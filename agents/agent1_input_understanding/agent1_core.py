"""
Agent 1: Input Understanding Agent
职责: 将异构输入数据(文本+时序+结构化)转换为标准化向量表示

核心组件:
1. ClinicalBERT: 编码心脏描述文本
2. LSTM: 编码血气时序数据
3. FeatureExtractor: 提取策略参数和患者特征
4. MedicalNER: 实体识别

负责人: 研究生 (NLP + 时序建模专家)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import re


@dataclass
class StandardizedInput:
    """Agent 1的标准化输出"""
    # 文本特征
    cardiac_text_embedding: torch.Tensor  # [768]
    cardiac_features: Dict[str, float]

    # 时序特征
    blood_gas_embedding: torch.Tensor  # [256]
    blood_gas_sequence: torch.Tensor  # [T, 6]
    metabolic_trajectory: Dict[str, float]

    # 策略特征
    strategy_params: torch.Tensor  # [20]
    strategy_features: Dict[str, float]

    # 患者特征
    patient_profile: torch.Tensor  # [50]
    risk_factors: List[str]

    # 实体
    extracted_entities: Dict[str, List[str]]

    # 原始数据（用于后续Agent）
    raw_text: str
    raw_blood_gas: List[Dict]


class ClinicalTextEncoder(nn.Module):
    """
    文本编码器: 使用ClinicalBERT编码心脏描述
    """
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__()

        # 加载预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # 微调层（针对心脏描述）
        self.fine_tune_layer = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 768)
        )

        # 特征提取头
        self.feature_extractors = nn.ModuleDict({
            'hypertrophy': nn.Linear(768, 1),  # 0-1 score
            'contractility': nn.Linear(768, 1),
            'valve_status': nn.Linear(768, 3),  # good/moderate/poor
            'scarring': nn.Linear(768, 1),
            'coronary_patency': nn.Linear(768, 1)
        })

    def forward(self, text: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        编码文本并提取特征

        Args:
            text: 心脏描述文本

        Returns:
            embedding: [768] 文本向量
            features: 提取的特征字典
        """
        # 1. BERT编码
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.bert(**inputs)

        # 使用[CLS] token的表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]

        # 2. 微调层
        refined_embedding = self.fine_tune_layer(cls_embedding)  # [1, 768]

        # 3. 提取特征
        features = {}

        # Hypertrophy (0-1)
        hypertrophy_score = torch.sigmoid(
            self.feature_extractors['hypertrophy'](refined_embedding)
        ).item()

        # Contractility (0-1)
        contractility_score = torch.sigmoid(
            self.feature_extractors['contractility'](refined_embedding)
        ).item()

        # Valve status (softmax -> categorical)
        valve_logits = self.feature_extractors['valve_status'](refined_embedding)
        valve_probs = torch.softmax(valve_logits, dim=-1)
        valve_status_idx = torch.argmax(valve_probs).item()
        valve_status_map = {0: 'good', 1: 'moderate', 2: 'poor'}

        # Scarring (0-1)
        scarring_score = torch.sigmoid(
            self.feature_extractors['scarring'](refined_embedding)
        ).item()

        # Coronary patency (0-1)
        coronary_score = torch.sigmoid(
            self.feature_extractors['coronary_patency'](refined_embedding)
        ).item()

        features = {
            'hypertrophy_level': hypertrophy_score,
            'contractility_score': contractility_score,
            'valve_status': valve_status_map[valve_status_idx],
            'valve_status_score': valve_probs[0, valve_status_idx].item(),
            'scarring_level': scarring_score,
            'coronary_patency': coronary_score,
            'visible_damage': scarring_score > 0.5
        }

        return refined_embedding.squeeze(0), features


class BloodGasLSTMEncoder(nn.Module):
    """
    时序编码器: 使用LSTM编码血气数据动态变化
    """
    def __init__(self,
                 input_size: int = 6,  # lactate, pH, pO2, pCO2, K+, glucose
                 hidden_size: int = 128,
                 num_layers: int = 2):
        super().__init__()

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        # 注意力层（识别关键时间点）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4
        )

        # 输出投影
        self.projection = nn.Linear(hidden_size * 2, 256)

    def forward(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码血气时序数据

        Args:
            sequence: [batch, time_steps, 6] 血气序列

        Returns:
            embedding: [batch, 256] 时序编码
            attention_weights: [batch, time_steps] 注意力权重
        """
        # 1. LSTM编码
        lstm_out, (h_n, c_n) = self.lstm(sequence)
        # lstm_out: [batch, time, hidden*2]

        # 2. 自注意力（找关键时间点）
        # 转换维度: [time, batch, hidden*2]
        lstm_out_transposed = lstm_out.transpose(0, 1)
        attn_output, attn_weights = self.attention(
            lstm_out_transposed,
            lstm_out_transposed,
            lstm_out_transposed
        )
        # attn_output: [time, batch, hidden*2]

        # 3. 取最后时间步
        final_state = attn_output[-1, :, :]  # [batch, hidden*2]

        # 4. 投影到固定维度
        embedding = self.projection(final_state)  # [batch, 256]

        return embedding, attn_weights.mean(dim=1)  # 平均所有头的权重

    def compute_trajectory_features(self, sequence: torch.Tensor) -> Dict[str, float]:
        """
        计算血气趋势特征

        Args:
            sequence: [time_steps, 6] 单个样本的血气序列

        Returns:
            trajectory_features: 趋势特征字典
        """
        # Lactate清除率 (负值表示下降，即好的)
        lactate = sequence[:, 0].numpy()
        time_points = np.arange(len(lactate))
        lactate_slope = np.polyfit(time_points, lactate, 1)[0]

        # pH稳定性 (标准差，越小越稳定)
        ph = sequence[:, 1].numpy()
        ph_std = np.std(ph)
        ph_stability = 1.0 / (1.0 + ph_std)  # 归一化到0-1

        # 氧合趋势
        po2 = sequence[:, 2].numpy()
        po2_slope = np.polyfit(time_points, po2, 1)[0]
        oxygenation_trend = 'improving' if po2_slope > 0 else 'stable' if abs(po2_slope) < 10 else 'declining'

        # pCO2趋势
        pco2 = sequence[:, 3].numpy()
        pco2_slope = np.polyfit(time_points, pco2, 1)[0]

        # K+稳定性
        k = sequence[:, 4].numpy()
        k_in_range = np.mean((k >= 3.5) & (k <= 5.0))  # 正常范围比例

        # Glucose变化
        glucose = sequence[:, 5].numpy()
        glucose_change = glucose[-1] - glucose[0]

        return {
            'lactate_clearance_rate': float(lactate_slope),  # mmol/L per time unit
            'lactate_initial': float(lactate[0]),
            'lactate_final': float(lactate[-1]),
            'ph_stability': float(ph_stability),
            'ph_mean': float(np.mean(ph)),
            'oxygenation_trend': oxygenation_trend,
            'po2_improvement': float(po2_slope),
            'pco2_trend': float(pco2_slope),
            'k_stability': float(k_in_range),
            'glucose_change': float(glucose_change)
        }


class StrategyFeatureExtractor:
    """
    策略参数特征提取器
    """
    def __init__(self):
        # 参考范围
        self.reference_ranges = {
            'pressure': (50, 80),      # mmHg
            'temperature': (2, 6),     # °C
            'flow_rate': (0.8, 1.5),   # L/min
            'duration': (180, 300)     # minutes
        }

        # 方法评分
        self.method_scores = {
            'HTK solution': 0.8,
            'Del Nido': 0.85,
            'Blood cardioplegia': 0.75,
            'Custodiol': 0.8,
            'St. Thomas': 0.7
        }

    def extract(self, strategy: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        提取策略特征

        Args:
            strategy: 策略字典

        Returns:
            feature_vector: [20] 特征向量
            feature_dict: 可解释特征字典
        """
        features = []
        feature_dict = {}

        # 1. 方法 (one-hot)
        method = strategy.get('method', 'HTK solution')
        method_score = self.method_scores.get(method, 0.5)
        features.append(method_score)
        feature_dict['method'] = method
        feature_dict['method_score'] = method_score

        # 2. 压力 (归一化)
        pressure = strategy.get('pressure', 60)
        pressure_normalized = (pressure - self.reference_ranges['pressure'][0]) / \
                             (self.reference_ranges['pressure'][1] - self.reference_ranges['pressure'][0])
        features.append(pressure_normalized)
        feature_dict['pressure'] = pressure
        feature_dict['pressure_normalized'] = pressure_normalized
        feature_dict['pressure_adequacy'] = 'adequate' if 65 <= pressure <= 80 else 'suboptimal'

        # 3. 温度
        temperature = strategy.get('temperature', 4)
        temp_normalized = (temperature - self.reference_ranges['temperature'][0]) / \
                         (self.reference_ranges['temperature'][1] - self.reference_ranges['temperature'][0])
        features.append(temp_normalized)
        feature_dict['temperature'] = temperature
        feature_dict['temperature_optimal'] = 2 <= temperature <= 6

        # 4. 流速
        flow_rate = strategy.get('flow_rate', 1.2)
        flow_normalized = (flow_rate - self.reference_ranges['flow_rate'][0]) / \
                         (self.reference_ranges['flow_rate'][1] - self.reference_ranges['flow_rate'][0])
        features.append(flow_normalized)
        feature_dict['flow_rate'] = flow_rate
        feature_dict['flow_adequacy'] = flow_normalized

        # 5. 持续时间
        duration = strategy.get('duration', 240)
        duration_normalized = (duration - self.reference_ranges['duration'][0]) / \
                             (self.reference_ranges['duration'][1] - self.reference_ranges['duration'][0])
        features.append(duration_normalized)
        feature_dict['duration'] = duration

        # 6. 添加剂 (binary features)
        common_additives = ['adenosine', 'insulin', 'magnesium', 'bicarbonate', 'mannitol']
        additives = strategy.get('additives', [])
        for additive in common_additives:
            features.append(1.0 if additive in additives else 0.0)
        feature_dict['additives'] = additives
        feature_dict['num_additives'] = len(additives)

        # 7. 递送模式
        delivery_mode = strategy.get('delivery_mode', 'antegrade')
        delivery_scores = {'antegrade': 0.7, 'retrograde': 0.5, 'combined': 0.9}
        features.append(delivery_scores.get(delivery_mode, 0.5))
        feature_dict['delivery_mode'] = delivery_mode

        # 8. 补充特征 (padding to 20)
        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32), feature_dict


class PatientRiskProfiler:
    """
    患者风险画像计算器
    """
    def __init__(self):
        # 风险权重
        self.comorbidity_weights = {
            'diabetes': 1.5,
            'hypertension': 1.2,
            'CKD': 2.0,
            'COPD': 1.3,
            'obesity': 1.2,
            'smoking': 1.1,
            'previous_MI': 1.8
        }

    def compute_profile(self, patient_record: Dict) -> Tuple[torch.Tensor, List[str]]:
        """
        计算患者风险画像

        Args:
            patient_record: 患者病历

        Returns:
            profile_vector: [50] 患者特征向量
            risk_factors: 风险因素列表
        """
        features = []
        risk_factors = []

        # 1. 人口学特征
        demographics = patient_record.get('demographics', {})
        age = demographics.get('age', 50)
        features.append(age / 100.0)  # 归一化

        weight = demographics.get('weight', 70)
        height = demographics.get('height', 170)
        bmi = weight / ((height / 100) ** 2)
        features.append(bmi / 40.0)  # 归一化
        if bmi > 30:
            risk_factors.append('obesity')

        gender = demographics.get('gender', 'male')
        features.append(1.0 if gender == 'male' else 0.0)

        # 2. 合并症 (binary features)
        comorbidities = patient_record.get('comorbidities', [])
        for condition, weight in self.comorbidity_weights.items():
            has_condition = condition in ' '.join(comorbidities).lower()
            features.append(1.0 if has_condition else 0.0)
            if has_condition:
                risk_factors.append(condition)

        # 3. 实验室指标
        lab_results = patient_record.get('lab_results', {})

        # Creatinine (肾功能)
        creatinine = lab_results.get('creatinine', 1.0)
        features.append(min(creatinine / 3.0, 1.0))  # 归一化，上限3.0
        if creatinine > 1.5:
            risk_factors.append('impaired_renal_function')

        # BNP (心功能)
        bnp = lab_results.get('BNP', 100)
        features.append(min(bnp / 5000.0, 1.0))  # 归一化

        # Troponin
        troponin = lab_results.get('troponin', 0.01)
        features.append(min(troponin / 1.0, 1.0))

        # Albumin
        albumin = lab_results.get('albumin', 4.0)
        features.append(albumin / 5.0)
        if albumin < 3.0:
            risk_factors.append('hypoalbuminemia')

        # 4. 血流动力学指标
        hemodynamics = patient_record.get('hemodynamics', {})

        # LVEF (左室射血分数)
        lvef = hemodynamics.get('LVEF', 60)
        features.append(lvef / 100.0)
        if lvef < 30:
            risk_factors.append('severe_LV_dysfunction')

        # PVR (肺血管阻力)
        pvr = hemodynamics.get('PVR', 2.0)
        features.append(min(pvr / 5.0, 1.0))
        if pvr > 3.0:
            risk_factors.append('pulmonary_hypertension')

        # Cardiac output
        co = hemodynamics.get('cardiac_output', 5.0)
        features.append(co / 8.0)

        # PCWP (肺毛细血管楔压)
        pcwp = hemodynamics.get('PCWP', 12)
        features.append(min(pcwp / 30.0, 1.0))

        # 5. 既往介入
        previous_interventions = patient_record.get('previous_interventions', [])
        has_lvad = 'LVAD' in ' '.join(previous_interventions)
        features.append(1.0 if has_lvad else 0.0)
        if has_lvad:
            risk_factors.append('LVAD')

        has_icd = 'ICD' in ' '.join(previous_interventions)
        features.append(1.0 if has_icd else 0.0)

        # 6. 综合风险评分
        risk_score = sum([
            self.comorbidity_weights.get(rf, 1.0)
            for rf in risk_factors
            if rf in self.comorbidity_weights
        ]) / 10.0  # 归一化
        features.append(risk_score)

        # Padding to 50
        while len(features) < 50:
            features.append(0.0)

        return torch.tensor(features, dtype=torch.float32), risk_factors


class MedicalNER:
    """
    医学命名实体识别
    使用规则+模式匹配 (可升级为transformer-based NER)
    """
    def __init__(self):
        # 实体词典
        self.entities = {
            'medications': [
                'adenosine', 'insulin', 'furosemide', 'carvedilol',
                'lisinopril', 'warfarin', 'aspirin', 'heparin',
                'magnesium', 'bicarbonate', 'mannitol'
            ],
            'perfusion_methods': [
                'HTK solution', 'Del Nido', 'Blood cardioplegia',
                'Custodiol', 'St. Thomas', 'Celsior'
            ],
            'biomarkers': [
                'lactate', 'troponin', 'BNP', 'creatinine', 'albumin',
                'pH', 'pO2', 'pCO2', 'glucose', 'potassium'
            ],
            'conditions': [
                'hypertrophy', 'scarring', 'ischemia', 'diabetes',
                'hypertension', 'CKD', 'COPD', 'cardiomyopathy'
            ],
            'devices': [
                'LVAD', 'ICD', 'pacemaker', 'ECMO', 'IABP'
            ]
        }

    def extract_all(self, text: str) -> Dict[str, List[str]]:
        """
        提取所有实体

        Args:
            text: 输入文本

        Returns:
            entities: 实体字典
        """
        text_lower = text.lower()

        extracted = {}
        for entity_type, entity_list in self.entities.items():
            found = []
            for entity in entity_list:
                if entity.lower() in text_lower:
                    found.append(entity)
            extracted[entity_type] = found

        return extracted


class InputUnderstandingAgent:
    """
    Agent 1 主类: 输入理解智能体
    """
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 初始化所有组件
        self.text_encoder = ClinicalTextEncoder().to(device)
        self.lstm_encoder = BloodGasLSTMEncoder().to(device)
        self.strategy_extractor = StrategyFeatureExtractor()
        self.patient_profiler = PatientRiskProfiler()
        self.ner = MedicalNER()

        print(f"✅ Agent 1 initialized on {device}")

    def process(self, raw_input: Dict) -> StandardizedInput:
        """
        处理原始输入，生成标准化表示

        Args:
            raw_input: 原始输入字典，包含:
                - perfusion_strategy: 灌注策略
                - cardiac_description: 心脏描述
                - blood_gas_data: 血气数据
                - recipient_medical_record: 患者病历

        Returns:
            StandardizedInput: 标准化输入对象
        """
        # 1. 文本理解
        cardiac_desc = raw_input.get('cardiac_description', {})
        full_text = ' '.join([
            cardiac_desc.get('visual_inspection', ''),
            cardiac_desc.get('palpation_notes', ''),
            cardiac_desc.get('procurement_notes', ''),
            cardiac_desc.get('surgeon_comments', '')
        ])

        text_embedding, cardiac_features = self.text_encoder(full_text)

        # 2. 时序数据处理
        blood_gas = raw_input.get('blood_gas_data', {})
        blood_gas_sequence = self._prepare_blood_gas_sequence(blood_gas)

        # LSTM编码
        blood_gas_embedding, attn_weights = self.lstm_encoder(
            blood_gas_sequence.unsqueeze(0).to(self.device)
        )
        blood_gas_embedding = blood_gas_embedding.squeeze(0).cpu()

        # 计算趋势特征
        metabolic_trajectory = self.lstm_encoder.compute_trajectory_features(
            blood_gas_sequence
        )

        # 3. 策略参数提取
        strategy = raw_input.get('perfusion_strategy', {})
        strategy_params, strategy_features = self.strategy_extractor.extract(strategy)

        # 4. 患者特征提取
        patient_record = raw_input.get('recipient_medical_record', {})
        patient_profile, risk_factors = self.patient_profiler.compute_profile(
            patient_record
        )

        # 5. 实体识别
        extracted_entities = self.ner.extract_all(full_text)

        # 6. 构建标准化输入
        standardized_input = StandardizedInput(
            # 文本
            cardiac_text_embedding=text_embedding.cpu(),
            cardiac_features=cardiac_features,

            # 时序
            blood_gas_embedding=blood_gas_embedding,
            blood_gas_sequence=blood_gas_sequence,
            metabolic_trajectory=metabolic_trajectory,

            # 策略
            strategy_params=strategy_params,
            strategy_features=strategy_features,

            # 患者
            patient_profile=patient_profile,
            risk_factors=risk_factors,

            # 实体
            extracted_entities=extracted_entities,

            # 原始数据
            raw_text=full_text,
            raw_blood_gas=blood_gas.get('during_perfusion', [])
        )

        return standardized_input

    def _prepare_blood_gas_sequence(self, blood_gas_data: Dict) -> torch.Tensor:
        """
        准备血气时序数据

        Args:
            blood_gas_data: 血气数据字典

        Returns:
            sequence: [T, 6] 时序tensor
        """
        during_perfusion = blood_gas_data.get('during_perfusion', [])

        if not during_perfusion:
            # 返回默认序列
            return torch.zeros(5, 6)

        # 提取6个指标: lactate, pH, pO2, pCO2, K+, glucose
        sequence = []
        for timepoint in during_perfusion:
            features = [
                timepoint.get('lactate', 1.0),
                timepoint.get('pH', 7.4),
                timepoint.get('pO2', 300),
                timepoint.get('pCO2', 40),
                timepoint.get('K+', 4.0),
                timepoint.get('glucose', 100)
            ]
            sequence.append(features)

        return torch.tensor(sequence, dtype=torch.float32)

    def summary(self, standardized_input: StandardizedInput) -> str:
        """
        生成可读的摘要
        """
        summary = f"""
╔════════════════════════════════════════════════════════════════╗
║                  Agent 1: 输入理解摘要                          ║
╚════════════════════════════════════════════════════════════════╝

【心脏状态】
  • 肥厚程度: {standardized_input.cardiac_features['hypertrophy_level']:.2f}
  • 收缩功能: {standardized_input.cardiac_features['contractility_score']:.2f}
  • 瓣膜状态: {standardized_input.cardiac_features['valve_status']}
  • 可见损伤: {'是' if standardized_input.cardiac_features['visible_damage'] else '否'}

【血气动态】
  • 乳酸清除率: {standardized_input.metabolic_trajectory['lactate_clearance_rate']:.3f} mmol/L/时间单位
  • 初始乳酸: {standardized_input.metabolic_trajectory['lactate_initial']:.2f} mmol/L
  • 最终乳酸: {standardized_input.metabolic_trajectory['lactate_final']:.2f} mmol/L
  • pH稳定性: {standardized_input.metabolic_trajectory['ph_stability']:.2f}
  • 氧合趋势: {standardized_input.metabolic_trajectory['oxygenation_trend']}

【灌注策略】
  • 方法: {standardized_input.strategy_features['method']}
  • 压力: {standardized_input.strategy_features['pressure']} mmHg ({standardized_input.strategy_features['pressure_adequacy']})
  • 流速充分性: {standardized_input.strategy_features['flow_adequacy']:.2f}
  • 温度最优: {'是' if standardized_input.strategy_features['temperature_optimal'] else '否'}
  • 添加剂: {', '.join(standardized_input.strategy_features['additives'])}

【患者风险】
  • 风险因素: {', '.join(standardized_input.risk_factors)}
  • 风险因素数量: {len(standardized_input.risk_factors)}

【提取实体】
  • 药物: {', '.join(standardized_input.extracted_entities['medications'])}
  • 生物标志物: {', '.join(standardized_input.extracted_entities['biomarkers'])}
  • 设备: {', '.join(standardized_input.extracted_entities['devices'])}

【向量维度】
  • 文本嵌入: {standardized_input.cardiac_text_embedding.shape}
  • 血气嵌入: {standardized_input.blood_gas_embedding.shape}
  • 策略参数: {standardized_input.strategy_params.shape}
  • 患者画像: {standardized_input.patient_profile.shape}

✅ 标准化完成，准备传递给 Agent 2
        """
        return summary.strip()
