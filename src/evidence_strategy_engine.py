#!/usr/bin/env python3
"""
Evidence-Driven Strategy Engine - 证据驱动的策略推荐引擎

核心设计：
1. KG证据收集层：从Neo4j精准查询相关证据
2. 证据聚合层：结构化整理、评分、去重
3. LLM策略生成层：强约束CoT推理，生成可溯源建议
4. 输出验证层：确保建议有证据支撑、有具体目标值
"""

import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvidenceType(Enum):
    """证据类型"""
    CAUSAL = "causal"           # 因果关系
    THERAPEUTIC = "therapeutic"  # 治疗关系
    RISK = "risk"               # 风险关系
    PHYSIOLOGICAL = "physiological"  # 生理机制
    CLINICAL_GUIDELINE = "guideline"  # 临床指南


class EvidenceStrength(Enum):
    """证据强度"""
    HIGH = "high"       # 多源验证、指南级
    MEDIUM = "medium"   # 单源可靠
    LOW = "low"         # 推断/间接
    UNKNOWN = "unknown"


@dataclass
class Evidence:
    """单条证据"""
    statement: str                    # 证据陈述
    evidence_type: EvidenceType       # 证据类型
    source_entity: str                # 源实体
    target_entity: str                # 目标实体
    relation: str                     # 关系类型
    strength: EvidenceStrength = EvidenceStrength.MEDIUM
    source_origin: str = "neo4j_kg"   # 来源（KG/文献/指南）
    numeric_value: Optional[float] = None  # 相关数值
    unit: Optional[str] = None        # 单位


@dataclass
class StrategyRecommendation:
    """策略推荐"""
    indicator: str                    # 目标指标
    current_value: float              # 当前值
    target_value: float               # 目标值
    target_range: Tuple[float, float] # 目标范围
    unit: str                         # 单位

    intervention: str                 # 干预措施
    intervention_type: str            # 干预类型
    priority: int                     # 优先级 (1=最高)

    reasoning_chain: List[str]        # CoT推理链
    supporting_evidence: List[Evidence]  # 支撑证据
    expected_effect: str              # 预期效果
    monitoring_points: List[str]      # 监测要点

    confidence: float = 0.0           # 置信度 0-1


@dataclass
class BaselineStrategyReport:
    """Baseline策略报告"""
    sample_id: str
    timestamp_min: int
    measurements: Dict[str, float]

    detected_abnormalities: List[Dict]
    recommendations: List[StrategyRecommendation]

    overall_risk_level: str
    summary: str


class KGEvidenceCollector:
    """知识图谱证据收集器"""

    # 指标到KG查询关键词的精确映射
    INDICATOR_QUERY_MAP = {
        'MAP': {
            'low': ['hypotension', 'low blood pressure', 'low MAP', 'shock', 'hemodynamic instability'],
            'keywords': ['mean arterial pressure', 'blood pressure', 'MAP'],
            'target_min': 65,
            'target_max': 90,
            'unit': 'mmHg'
        },
        'Lactate': {
            'high': ['lactic acidosis', 'hyperlactatemia', 'elevated lactate', 'tissue hypoperfusion'],
            'keywords': ['lactate', 'lactic acid'],
            'target_min': 0,
            'target_max': 4.0,  # 灌注期间允许稍高
            'unit': 'mmol/L'
        },
        'SvO2': {
            'low': ['tissue hypoxia', 'low SvO2', 'oxygen extraction', 'low cardiac output'],
            'high': ['shunting', 'sepsis', 'mitochondrial dysfunction'],
            'keywords': ['mixed venous oxygen', 'SvO2', 'venous saturation'],
            'target_min': 65,
            'target_max': 80,
            'unit': '%'
        },
        'CI': {
            'low': ['low cardiac output', 'cardiogenic shock', 'cardiac dysfunction', 'heart failure'],
            'keywords': ['cardiac index', 'cardiac output'],
            'target_min': 2.2,
            'target_max': 4.0,
            'unit': 'L/min/m²'
        },
        'Na_A': {
            'low': ['hyponatremia', 'cerebral edema', 'seizure'],
            'high': ['hypernatremia', 'dehydration', 'hyperosmolarity'],
            'keywords': ['sodium', 'Na+', 'electrolyte'],
            'target_min': 135,
            'target_max': 145,
            'unit': 'mmol/L'
        },
        'GluA': {
            'low': ['hypoglycemia', 'neuroglycopenia', 'altered consciousness'],
            'high': ['hyperglycemia', 'diabetic ketoacidosis', 'hyperosmolar'],
            'keywords': ['glucose', 'blood sugar'],
            'target_min': 4.0,
            'target_max': 10.0,  # 围术期允许稍高
            'unit': 'mmol/L'
        },
        'pO2V': {
            'low': ['venous hypoxemia', 'tissue hypoxia', 'increased oxygen extraction'],
            'keywords': ['venous pO2', 'mixed venous oxygen tension'],
            'target_min': 35,
            'target_max': 45,
            'unit': 'mmHg'
        },
        'pO2A': {
            'low': ['hypoxemia', 'respiratory failure', 'shunt'],
            'keywords': ['arterial pO2', 'oxygenation'],
            'target_min': 80,
            'target_max': 300,  # 高FiO2时可更高
            'unit': 'mmHg'
        },
        'pH': {
            'low': ['acidosis', 'metabolic acidosis', 'respiratory acidosis'],
            'high': ['alkalosis', 'metabolic alkalosis', 'respiratory alkalosis'],
            'keywords': ['pH', 'acid-base', 'hydrogen ion'],
            'target_min': 7.35,
            'target_max': 7.45,
            'unit': ''
        },
        'CF': {
            'low': ['coronary hypoperfusion', 'myocardial ischemia', 'graft dysfunction'],
            'keywords': ['coronary flow', 'myocardial perfusion'],
            'target_min': 0.5,  # L/min
            'target_max': 2.0,
            'unit': 'L/min'
        },
        'K_A': {
            'low': ['hypokalemia', 'low potassium'],
            'high': ['hyperkalemia', 'high potassium', 'arrhythmia'],
            'keywords': ['potassium', 'K+', 'electrolyte'],
            'target_min': 3.5,
            'target_max': 5.0,
            'unit': 'mmol/L'
        },
        'CvO2': {
            'low': ['tissue hypoxia', 'low oxygen delivery', 'oxygen extraction'],
            'keywords': ['venous oxygen content', 'CvO2'],
            'target_min': 12,
            'target_max': 15,
            'unit': 'mL/dL'
        },
        'HR': {
            'high': ['tachycardia', 'arrhythmia'],
            'low': ['bradycardia', 'heart block'],
            'keywords': ['heart rate'],
            'target_min': 60,
            'target_max': 100,
            'unit': 'bpm'
        },
        # === 移植共识新增指标 ===
        'PVR': {
            'high': ['pulmonary hypertension', 'right heart failure', 'RV dysfunction'],
            'keywords': ['pulmonary vascular resistance', 'PVR'],
            'target_min': 0,
            'target_max': 2.5,
            'unit': 'Wood Units'
        },
        'TPG': {
            'high': ['transpulmonary gradient', 'pulmonary hypertension', 'transplant contraindication'],
            'keywords': ['transpulmonary gradient', 'TPG'],
            'target_min': 0,
            'target_max': 12,
            'unit': 'mmHg'
        },
        'PASP': {
            'high': ['pulmonary artery pressure', 'right ventricular failure', 'RV dysfunction'],
            'keywords': ['pulmonary artery systolic pressure', 'PASP'],
            'target_min': 15,
            'target_max': 40,
            'unit': 'mmHg'
        },
        'Creatinine': {
            'high': ['renal dysfunction', 'CNI nephrotoxicity', 'acute kidney injury'],
            'keywords': ['creatinine', 'renal function'],
            'target_min': 0.5,
            'target_max': 1.5,
            'unit': 'mg/dL'
        },
        'GFR': {
            'low': ['chronic kidney disease', 'renal impairment', 'CNI nephrotoxicity'],
            'keywords': ['GFR', 'glomerular filtration rate'],
            'target_min': 60,
            'target_max': 120,
            'unit': 'mL/min/1.73m²'
        },
        'Bilirubin': {
            'high': ['hepatic dysfunction', 'cholestasis', 'liver failure'],
            'keywords': ['bilirubin', 'liver function'],
            'target_min': 0.1,
            'target_max': 1.2,
            'unit': 'mg/dL'
        },
        'EF': {
            'low': ['systolic dysfunction', 'graft failure', 'acute rejection'],
            'keywords': ['ejection fraction', 'left ventricular function'],
            'target_min': 50,
            'target_max': 70,
            'unit': '%'
        }
    }

    def __init__(self, neo4j_connector=None):
        self.neo4j = neo4j_connector
        self._evidence_cache = {}

    def set_neo4j(self, connector):
        self.neo4j = connector

    def collect_evidence_for_abnormality(
        self,
        indicator: str,
        current_value: float,
        direction: str  # 'high' or 'low'
    ) -> List[Evidence]:
        """为特定异常收集证据"""
        evidences = []

        if self.neo4j is None:
            logger.warning("Neo4j未连接，返回空证据")
            return evidences

        config = self.INDICATOR_QUERY_MAP.get(indicator, {})
        keywords = config.get(direction, []) + config.get('keywords', [])

        for keyword in keywords[:3]:  # 限制查询数量
            try:
                # 1. 查询原因
                causes = self._query_causes(keyword)
                for c in causes[:5]:
                    evidences.append(Evidence(
                        statement=f"{c['cause']} 可导致 {keyword}",
                        evidence_type=EvidenceType.CAUSAL,
                        source_entity=c['cause'],
                        target_entity=keyword,
                        relation=c.get('relation', 'causes'),
                        source_origin=f"KG:{c.get('cause_type', 'entity')}"
                    ))

                # 2. 查询后果
                consequences = self._query_consequences(keyword)
                for c in consequences[:5]:
                    evidences.append(Evidence(
                        statement=f"{keyword} 可导致 {c['consequence']}",
                        evidence_type=EvidenceType.RISK,
                        source_entity=keyword,
                        target_entity=c['consequence'],
                        relation=c.get('relation', 'leads_to'),
                        source_origin=f"KG:{c.get('consequence_type', 'entity')}"
                    ))

                # 3. 查询治疗
                treatments = self._query_treatments(keyword)
                for t in treatments[:5]:
                    evidences.append(Evidence(
                        statement=f"{t['treatment']} 可治疗/缓解 {keyword}",
                        evidence_type=EvidenceType.THERAPEUTIC,
                        source_entity=t['treatment'],
                        target_entity=keyword,
                        relation=t.get('relation', 'treats'),
                        strength=EvidenceStrength.MEDIUM,
                        source_origin=f"KG:{t.get('treatment_type', 'treatment')}"
                    ))

            except Exception as e:
                logger.error(f"查询 {keyword} 失败: {e}")

        # 去重
        seen = set()
        unique_evidences = []
        for e in evidences:
            key = (e.source_entity, e.target_entity, e.relation)
            if key not in seen:
                seen.add(key)
                unique_evidences.append(e)

        return unique_evidences[:15]  # 限制返回数量

    def _query_causes(self, keyword: str) -> List[Dict]:
        """查询原因"""
        query = """
        MATCH (cause:Entity)-[r:RELATION]->(target:Entity)
        WHERE toLower(target.name) CONTAINS toLower($keyword)
          AND r.type IN ['causes', 'leads_to', 'results_in', 'increases_risk']
        RETURN cause.name AS cause, cause.type AS cause_type,
               r.type AS relation, target.name AS target
        LIMIT 10
        """
        return self.neo4j._run_query(query, {'keyword': keyword})

    def _query_consequences(self, keyword: str) -> List[Dict]:
        """查询后果"""
        query = """
        MATCH (source:Entity)-[r:RELATION]->(consequence:Entity)
        WHERE toLower(source.name) CONTAINS toLower($keyword)
          AND r.type IN ['causes', 'leads_to', 'increases_risk', 'associated with']
        RETURN source.name AS source,
               consequence.name AS consequence,
               consequence.type AS consequence_type,
               r.type AS relation
        LIMIT 10
        """
        return self.neo4j._run_query(query, {'keyword': keyword})

    def _query_treatments(self, keyword: str) -> List[Dict]:
        """查询治疗方案"""
        query = """
        MATCH (treatment:Entity)-[r:RELATION]->(target:Entity)
        WHERE toLower(target.name) CONTAINS toLower($keyword)
          AND treatment.type IN ['treatment_regimen', 'medication', 'surgical_procedure', 'medical_device']
          AND r.type IN ['treats', 'alleviates', 'prevents', 'indicated_for', 'reduces_risk']
        RETURN treatment.name AS treatment,
               treatment.type AS treatment_type,
               r.type AS relation,
               target.name AS target
        LIMIT 10
        """
        return self.neo4j._run_query(query, {'keyword': keyword})

    def get_target_values(self, indicator: str) -> Dict:
        """获取指标的目标值"""
        config = self.INDICATOR_QUERY_MAP.get(indicator, {})
        return {
            'target_min': config.get('target_min'),
            'target_max': config.get('target_max'),
            'unit': config.get('unit', '')
        }


class LLMStrategyGenerator:
    """LLM策略生成器 - 强约束CoT"""

    # 策略生成提示词模板
    STRATEGY_PROMPT_TEMPLATE = """你是一位心脏移植灌注监测的临床决策支持专家。基于提供的证据，为异常指标生成精确的干预策略。

## 强制约束 (MUST FOLLOW)
1. **证据溯源**: 每条建议必须引用至少一条提供的证据，格式为 [Evidence-X]
2. **具体数值**: 必须给出精确的目标值和范围，不能使用模糊描述
3. **CoT推理**: 必须展示完整的推理链，包括：观察→分析→机制→干预→预期
4. **优先级**: 必须按紧迫性排序，1=最紧急
5. **监测要点**: 必须说明干预后的监测指标和频率

## 当前状态
- 指标: {indicator}
- 当前值: {current_value} {unit}
- 异常方向: {direction}
- 目标范围: {target_min} - {target_max} {unit}

## 知识图谱证据
{evidence_text}

## 输出格式 (JSON)
{{
  "reasoning_chain": [
    "Step 1 - 观察: ...",
    "Step 2 - 病理生理分析: ...",
    "Step 3 - 机制关联: ...",
    "Step 4 - 干预选择: ...",
    "Step 5 - 预期效果: ..."
  ],
  "intervention": "具体干预措施名称",
  "intervention_details": {{
    "drug": "药物名称（如适用）",
    "dose": "剂量和给药方式",
    "target": "调控目标"
  }},
  "target_value": 数值,
  "target_range": [最小值, 最大值],
  "expected_effect": "预期效果描述",
  "monitoring": [
    "监测项目1: 频率",
    "监测项目2: 频率"
  ],
  "evidence_used": ["Evidence-1", "Evidence-3"],
  "confidence": 0.0到1.0的置信度,
  "warnings": ["注意事项1", "注意事项2"]
}}

请严格按照JSON格式输出，确保可解析。"""

    # 预定义的干预策略库（当无LLM时使用）
    INTERVENTION_LIBRARY = {
        'MAP_Low': {
            'intervention': '血管活性药物支持',
            'intervention_details': {
                'drug': '去甲肾上腺素 (Norepinephrine)',
                'dose': '起始 0.05-0.1 μg/kg/min，滴定至目标MAP',
                'target': 'MAP ≥ 65 mmHg'
            },
            'target_value': 70,
            'target_range': [65, 80],
            'reasoning_chain': [
                'Step 1 - 观察: MAP < 50 mmHg，低于组织灌注安全阈值',
                'Step 2 - 病理生理分析: 低MAP导致冠脉灌注不足、组织缺氧',
                'Step 3 - 机制关联: [Evidence] 低血压→器官灌注不足→多器官功能障碍',
                'Step 4 - 干预选择: 首选去甲肾上腺素，α受体激动提升血管张力',
                'Step 5 - 预期效果: MAP提升至65-80 mmHg，改善组织灌注'
            ],
            'monitoring': ['MAP: 每5分钟', 'HR: 每5分钟', 'Lactate: 每30分钟', '尿量: 每小时'],
            'confidence': 0.85,
            'warnings': ['注意心率变化', '高剂量可致心律失常', '需同时评估容量状态']
        },
        'Lactate_High': {
            'intervention': '优化组织灌注',
            'intervention_details': {
                'drug': '多巴酚丁胺 (Dobutamine) + 补液',
                'dose': '多巴酚丁胺 2.5-5 μg/kg/min，晶体液 250-500 mL bolus',
                'target': 'Lactate < 2 mmol/L'
            },
            'target_value': 1.5,
            'target_range': [0.5, 2.0],
            'reasoning_chain': [
                'Step 1 - 观察: Lactate升高，提示组织缺氧/无氧代谢',
                'Step 2 - 病理生理分析: 乳酸升高是组织灌注不足的敏感标志',
                'Step 3 - 机制关联: [Evidence] 低灌注→无氧糖酵解→乳酸堆积',
                'Step 4 - 干预选择: 增强心输出量+补充血容量，改善氧输送',
                'Step 5 - 预期效果: 4-6小时内Lactate下降>20%'
            ],
            'monitoring': ['Lactate: 每1-2小时', 'SvO2: 每30分钟', 'CI: 每小时', 'CVP: 每30分钟'],
            'confidence': 0.80,
            'warnings': ['需排除肝功能障碍', '监测心率防止过度增快', '注意容量负荷']
        },
        'K_A_Low': {
            'intervention': '钾离子补充',
            'intervention_details': {
                'drug': '氯化钾 (KCl)',
                'dose': '10-20 mEq/h IV，最大浓度 40 mEq/L',
                'target': 'K+ 3.5-4.5 mmol/L'
            },
            'target_value': 4.0,
            'target_range': [3.5, 5.0],
            'reasoning_chain': [
                'Step 1 - 观察: K+ < 3.5 mmol/L，存在低钾血症',
                'Step 2 - 病理生理分析: 低钾可致心律失常、肌无力',
                'Step 3 - 机制关联: [Evidence] 低钾→细胞膜电位改变→心律失常风险',
                'Step 4 - 干预选择: 静脉补钾，速度不超过20 mEq/h',
                'Step 5 - 预期效果: 2-4小时K+恢复至正常范围'
            ],
            'monitoring': ['K+: 每2小时', 'ECG: 持续监测', 'Mg2+: 同时检测'],
            'confidence': 0.90,
            'warnings': ['补钾速度过快可致心脏骤停', '需同时纠正低镁', '肾功能不全需减量']
        },
        'K_A_High': {
            'intervention': '降钾治疗',
            'intervention_details': {
                'drug': '胰岛素+葡萄糖 / 钙剂',
                'dose': '胰岛素10U + 50%葡萄糖50mL IV；葡萄糖酸钙1g IV',
                'target': 'K+ < 5.5 mmol/L'
            },
            'target_value': 4.5,
            'target_range': [3.5, 5.0],
            'reasoning_chain': [
                'Step 1 - 观察: K+ > 5.5 mmol/L，存在高钾血症',
                'Step 2 - 病理生理分析: 高钾致心肌传导异常，可致室颤',
                'Step 3 - 机制关联: [Evidence] 高钾→静息膜电位降低→传导阻滞/心律失常',
                'Step 4 - 干预选择: 钙剂稳定心肌膜，胰岛素促钾进入细胞',
                'Step 5 - 预期效果: 30分钟内K+下降0.5-1.0 mmol/L'
            ],
            'monitoring': ['K+: 每1小时', 'ECG: 持续监测', '血糖: 每30分钟'],
            'confidence': 0.90,
            'warnings': ['钙剂与地高辛有相互作用', '胰岛素可致低血糖', '严重者需透析']
        },
        'CvO2_Low': {
            'intervention': '优化氧输送',
            'intervention_details': {
                'drug': '输血 + 正性肌力药',
                'dose': 'Hb < 7 g/dL时输红细胞；多巴酚丁胺 2.5-10 μg/kg/min',
                'target': 'CvO2 > 12 mL/dL'
            },
            'target_value': 14,
            'target_range': [12, 16],
            'reasoning_chain': [
                'Step 1 - 观察: CvO2降低，静脉血氧含量不足',
                'Step 2 - 病理生理分析: 反映组织氧摄取增加或氧输送减少',
                'Step 3 - 机制关联: [Evidence] DO2不足→组织缺氧→器官功能障碍',
                'Step 4 - 干预选择: 增加携氧能力（输血）+ 增加心输出量',
                'Step 5 - 预期效果: CvO2恢复至正常范围，组织氧供改善'
            ],
            'monitoring': ['CvO2: 每1小时', 'Hb: 每4小时', 'SvO2: 每30分钟', 'Lactate: 每2小时'],
            'confidence': 0.75,
            'warnings': ['输血有TRALI风险', '需评估心功能状态', '注意容量负荷']
        },
        'SvO2_Low': {
            'intervention': '改善氧供需平衡',
            'intervention_details': {
                'drug': '正性肌力药 + 镇静',
                'dose': '多巴酚丁胺 2.5-10 μg/kg/min；必要时丙泊酚镇静',
                'target': 'SvO2 > 65%'
            },
            'target_value': 70,
            'target_range': [65, 80],
            'reasoning_chain': [
                'Step 1 - 观察: SvO2 < 65%，混合静脉血氧饱和度降低',
                'Step 2 - 病理生理分析: 反映全身氧供需失衡',
                'Step 3 - 机制关联: [Evidence] 心输出量不足/氧耗增加→SvO2下降',
                'Step 4 - 干预选择: 增加心输出量 + 降低氧耗（镇静、降温）',
                'Step 5 - 预期效果: SvO2恢复至65-80%'
            ],
            'monitoring': ['SvO2: 持续监测', 'CI: 每小时', '体温: 每小时', 'Lactate: 每2小时'],
            'confidence': 0.80,
            'warnings': ['SvO2过高可能提示分流或组织利用障碍', '镇静过深可致呼吸抑制']
        },
        'SvO2_High': {
            'intervention': '评估组织氧利用',
            'intervention_details': {
                'drug': '病因治疗',
                'dose': '根据病因：脓毒症抗感染、线粒体功能障碍支持治疗',
                'target': 'SvO2 65-80%'
            },
            'target_value': 72,
            'target_range': [65, 80],
            'reasoning_chain': [
                'Step 1 - 观察: SvO2 > 80%，混合静脉血氧饱和度异常升高',
                'Step 2 - 病理生理分析: 可能提示组织氧摄取障碍、分流或线粒体功能障碍',
                'Step 3 - 机制关联: [Evidence] 脓毒症/线粒体损伤→组织无法利用氧→SvO2升高',
                'Step 4 - 干预选择: 排查脓毒症、评估微循环、支持治疗',
                'Step 5 - 预期效果: 明确病因后针对性治疗'
            ],
            'monitoring': ['SvO2: 持续监测', 'Lactate: 每1小时', '感染指标: 每4小时'],
            'confidence': 0.70,
            'warnings': ['SvO2高不代表氧供充足', '需结合Lactate综合判断', '警惕脓毒症']
        },
        'Na_A_Low': {
            'intervention': '纠正低钠血症',
            'intervention_details': {
                'drug': '高渗盐水 (3% NaCl)',
                'dose': '急性症状性：3% NaCl 100-150mL/20min；慢性：限水+口服盐',
                'target': 'Na+ > 130 mmol/L'
            },
            'target_value': 138,
            'target_range': [135, 145],
            'reasoning_chain': [
                'Step 1 - 观察: Na+ < 135 mmol/L，存在低钠血症',
                'Step 2 - 病理生理分析: 低钠致细胞水肿，严重者脑水肿、癫痫',
                'Step 3 - 机制关联: [Evidence] 低钠→细胞外渗透压降低→细胞水肿',
                'Step 4 - 干预选择: 急性有症状用高渗盐水，纠正速度≤10 mmol/L/24h',
                'Step 5 - 预期效果: Na+逐步恢复，症状改善'
            ],
            'monitoring': ['Na+: 每2-4小时', '神经系统症状: 持续', '尿量: 每小时'],
            'confidence': 0.85,
            'warnings': ['纠正过快可致渗透性脱髓鞘', '限速≤10 mmol/L/24h', '监测神经症状']
        },
        'Na_A_High': {
            'intervention': '纠正高钠血症',
            'intervention_details': {
                'drug': '低渗液体 (5% 葡萄糖 / 0.45% NaCl)',
                'dose': '计算自由水缺失，24-48小时缓慢纠正',
                'target': 'Na+ < 145 mmol/L'
            },
            'target_value': 142,
            'target_range': [135, 145],
            'reasoning_chain': [
                'Step 1 - 观察: Na+ > 145 mmol/L，存在高钠血症',
                'Step 2 - 病理生理分析: 高钠致细胞脱水，神经系统功能障碍',
                'Step 3 - 机制关联: [Evidence] 高钠→细胞外高渗→细胞脱水',
                'Step 4 - 干预选择: 补充自由水，降低渗透压',
                'Step 5 - 预期效果: Na+逐步恢复正常'
            ],
            'monitoring': ['Na+: 每4小时', '神经系统: 持续', '血容量: 评估'],
            'confidence': 0.80,
            'warnings': ['纠正过快可致脑水肿', '每小时降低≤0.5 mmol/L']
        },
        'GluA_High': {
            'intervention': '血糖控制',
            'intervention_details': {
                'drug': '胰岛素 (Regular Insulin)',
                'dose': '静脉泵注 0.5-2 U/h，目标血糖 6-10 mmol/L',
                'target': 'Glucose 6-10 mmol/L'
            },
            'target_value': 8.0,
            'target_range': [6.0, 10.0],
            'reasoning_chain': [
                'Step 1 - 观察: 血糖 > 10 mmol/L，存在高血糖',
                'Step 2 - 病理生理分析: 高血糖致渗透性利尿、免疫功能受损',
                'Step 3 - 机制关联: [Evidence] 应激/糖皮质激素→胰岛素抵抗→高血糖',
                'Step 4 - 干预选择: 胰岛素持续泵注，避免低血糖',
                'Step 5 - 预期效果: 血糖控制在目标范围'
            ],
            'monitoring': ['血糖: 每1小时', 'K+: 每4小时', '酮体: 必要时'],
            'confidence': 0.85,
            'warnings': ['避免低血糖', '胰岛素可致低钾', '监测血钾']
        },
        'GluA_Low': {
            'intervention': '纠正低血糖',
            'intervention_details': {
                'drug': '葡萄糖',
                'dose': '50% 葡萄糖 50mL IV，后10% 葡萄糖维持',
                'target': 'Glucose > 4 mmol/L'
            },
            'target_value': 6.0,
            'target_range': [4.0, 10.0],
            'reasoning_chain': [
                'Step 1 - 观察: 血糖 < 4 mmol/L，存在低血糖',
                'Step 2 - 病理生理分析: 低血糖致神经功能障碍、意识改变',
                'Step 3 - 机制关联: [Evidence] 葡萄糖不足→神经元能量缺乏→功能障碍',
                'Step 4 - 干预选择: 立即补充葡萄糖',
                'Step 5 - 预期效果: 血糖迅速恢复，症状改善'
            ],
            'monitoring': ['血糖: 每15-30分钟', '神经系统: 持续评估'],
            'confidence': 0.95,
            'warnings': ['低血糖是急症，需立即处理', '查找低血糖原因']
        },
        'pH_Low': {
            'intervention': '纠正酸中毒',
            'intervention_details': {
                'drug': '碳酸氢钠 (严重时) / 病因治疗',
                'dose': 'pH < 7.1时：NaHCO3 1-2 mEq/kg IV；优先纠正病因',
                'target': 'pH 7.35-7.45'
            },
            'target_value': 7.40,
            'target_range': [7.35, 7.45],
            'reasoning_chain': [
                'Step 1 - 观察: pH < 7.35，存在酸中毒',
                'Step 2 - 病理生理分析: 酸中毒抑制心肌收缩力、血管对儿茶酚胺反应性降低',
                'Step 3 - 机制关联: [Evidence] 乳酸堆积/CO2潴留→H+增加→酸中毒',
                'Step 4 - 干预选择: 优先纠正病因（改善灌注/通气），严重时补碱',
                'Step 5 - 预期效果: pH恢复正常'
            ],
            'monitoring': ['pH/血气: 每1-2小时', 'Lactate: 每1小时', 'K+: 每2小时'],
            'confidence': 0.80,
            'warnings': ['补碱可能加重细胞内酸中毒', '优先纠正病因', '注意低钾']
        },
        'pH_High': {
            'intervention': '纠正碱中毒',
            'intervention_details': {
                'drug': '病因治疗 / 氯化物补充',
                'dose': '代谢性：补充Cl-（0.9% NaCl）；呼吸性：调整通气',
                'target': 'pH 7.35-7.45'
            },
            'target_value': 7.40,
            'target_range': [7.35, 7.45],
            'reasoning_chain': [
                'Step 1 - 观察: pH > 7.45，存在碱中毒',
                'Step 2 - 病理生理分析: 碱中毒致低钾、低钙、心律失常',
                'Step 3 - 机制关联: [Evidence] 呕吐/利尿/过度通气→H+丢失→碱中毒',
                'Step 4 - 干预选择: 代谢性补氯，呼吸性调整通气参数',
                'Step 5 - 预期效果: pH恢复正常'
            ],
            'monitoring': ['pH/血气: 每2小时', 'K+: 每2小时', 'ECG: 持续'],
            'confidence': 0.75,
            'warnings': ['注意低钾血症', '碱中毒可致心律失常']
        },
        'pO2V_Low': {
            'intervention': '改善静脉血氧合',
            'intervention_details': {
                'drug': '正性肌力药 + 优化氧输送',
                'dose': '多巴酚丁胺 2.5-10 μg/kg/min；必要时输血',
                'target': 'pO2V 35-45 mmHg'
            },
            'target_value': 40,
            'target_range': [35, 45],
            'reasoning_chain': [
                'Step 1 - 观察: pO2V降低，静脉血氧分压不足',
                'Step 2 - 病理生理分析: 反映组织氧摄取增加或氧输送不足',
                'Step 3 - 机制关联: [Evidence] 心输出量低/Hb低→DO2不足→pO2V降低',
                'Step 4 - 干预选择: 增加心输出量，必要时输血',
                'Step 5 - 预期效果: pO2V恢复至正常范围'
            ],
            'monitoring': ['pO2V: 每1小时', 'SvO2: 每30分钟', 'CI: 每小时'],
            'confidence': 0.75,
            'warnings': ['需综合评估DO2', '注意Hb水平']
        },
        'pO2V_High': {
            'intervention': '评估高静脉氧分压',
            'intervention_details': {
                'drug': '病因排查',
                'dose': '评估是否存在分流、脓毒症、线粒体功能障碍',
                'target': 'pO2V 35-45 mmHg'
            },
            'target_value': 40,
            'target_range': [35, 45],
            'reasoning_chain': [
                'Step 1 - 观察: pO2V升高，静脉血氧分压过高',
                'Step 2 - 病理生理分析: 可能提示组织氧摄取障碍',
                'Step 3 - 机制关联: [Evidence] 分流/线粒体损伤→组织无法利用氧→pO2V升高',
                'Step 4 - 干预选择: 排查病因，针对性治疗',
                'Step 5 - 预期效果: 明确并处理病因'
            ],
            'monitoring': ['pO2V: 每1小时', 'Lactate: 每1小时', '感染指标: 评估'],
            'confidence': 0.65,
            'warnings': ['pO2V高不代表氧供充足', '需结合Lactate判断']
        },
        'CF_Low': {
            'intervention': '优化冠脉灌注',
            'intervention_details': {
                'drug': '血管活性药 + 容量优化',
                'dose': '提升灌注压，去甲肾上腺素滴定MAP > 65 mmHg',
                'target': 'CF > 0.5 L/min'
            },
            'target_value': 0.8,
            'target_range': [0.5, 1.5],
            'reasoning_chain': [
                'Step 1 - 观察: CF降低，冠脉灌注流量不足',
                'Step 2 - 病理生理分析: 冠脉灌注不足可致心肌缺血、移植心功能障碍',
                'Step 3 - 机制关联: [Evidence] 低灌注压/冠脉阻力高→CF降低→心肌缺血',
                'Step 4 - 干预选择: 提升灌注压，优化前负荷',
                'Step 5 - 预期效果: CF恢复，心肌灌注改善'
            ],
            'monitoring': ['CF: 每30分钟', 'MAP: 每5分钟', 'Lactate: 每1小时', 'ECG: 持续'],
            'confidence': 0.80,
            'warnings': ['CF降低提示移植心缺血风险', '监测心肌酶谱']
        },
        # ===== 移植共识新增策略 =====
        'PVR_High': {
            'intervention': '肺血管扩张治疗（共识）',
            'intervention_details': {
                'drug': '米力农 + NO + 前列腺素E1',
                'dose': '米力农 0.3-1 μg/kg/min + iNO 20-40ppm；前列腺素E1/硝普钠/西地那非备选',
                'target': 'PVR < 2.5 Wood Units'
            },
            'target_value': 2.0,
            'target_range': [0.5, 2.5],
            'reasoning_chain': [
                'Step 1 - 观察: PVR升高，右心后负荷增加',
                'Step 2 - 病理生理分析: 高PVR→右室难以承受→急性右心衰；共识>50mmHg肺动脉压为危险',
                'Step 3 - 机制关联: [Evidence-共识] 肺血管阻力升高→急性右心衰→移植心功能障碍',
                'Step 4 - 干预选择: 联合肺血管扩张（米力农+NO首选），共识方案',
                'Step 5 - 预期效果: PVR降至<2.5 Wood，右心功能改善'
            ],
            'monitoring': ['PVR: 每1小时', 'CVP: 每30分钟', 'CI: 每小时', 'MAP: 每15分钟'],
            'confidence': 0.85,
            'warnings': ['PVR>5 Wood为移植禁忌', '监测右心功能', '米力农可致低血压']
        },
        'TPG_High': {
            'intervention': '降低跨肺压差（共识）',
            'intervention_details': {
                'drug': '肺血管扩张剂',
                'dose': 'iNO 20-40ppm + 米力农 0.3-1 μg/kg/min',
                'target': 'TPG < 12 mmHg'
            },
            'target_value': 10,
            'target_range': [5, 12],
            'reasoning_chain': [
                'Step 1 - 观察: TPG >12 mmHg，跨肺压差升高',
                'Step 2 - 病理生理分析: TPG=mPAP-PCWP，反映固定性肺血管阻力',
                'Step 3 - 机制关联: [Evidence-共识] TPG>14-15 mmHg为移植禁忌',
                'Step 4 - 干预选择: 肺血管扩张剂试验，评估可逆性',
                'Step 5 - 预期效果: TPG降至<12 mmHg'
            ],
            'monitoring': ['TPG: 每2小时', 'mPAP: 每小时', 'PCWP: 每小时'],
            'confidence': 0.80,
            'warnings': ['TPG>14-15 mmHg为移植禁忌', '需评估扩张剂反应性']
        },
        'PASP_High': {
            'intervention': '降低肺动脉压（共识）',
            'intervention_details': {
                'drug': '联合肺血管扩张',
                'dose': '米力农 + iNO + 前列腺素E1',
                'target': 'PASP < 40 mmHg'
            },
            'target_value': 35,
            'target_range': [15, 40],
            'reasoning_chain': [
                'Step 1 - 观察: PASP升高，肺动脉高压',
                'Step 2 - 病理生理分析: 共识：>50mmHg右室难承受，>60-70mmHg禁忌',
                'Step 3 - 机制关联: [Evidence-共识] PASP↑→右室压力超负荷→急性右心衰',
                'Step 4 - 干预选择: 按共识右心衰处理流程：检查吻合→纠正缺氧酸中毒→肺血管扩张',
                'Step 5 - 预期效果: PASP降至<40 mmHg，右室功能改善'
            ],
            'monitoring': ['PASP: 每30分钟', 'CVP: 每30分钟', 'CI: 每小时'],
            'confidence': 0.80,
            'warnings': ['PASP>50为高危', '>60-70禁忌', '监测右心功能']
        },
        'Creatinine_High': {
            'intervention': '免疫抑制方案调整（共识）',
            'intervention_details': {
                'drug': '他克莫司减量 / ATG替代',
                'dose': 'Cr>1.7→CNI减量; Cr>2.0→ATG替代CNI',
                'target': '肌酐 < 1.5 mg/dL'
            },
            'target_value': 1.0,
            'target_range': [0.5, 1.5],
            'reasoning_chain': [
                'Step 1 - 观察: 肌酐升高，肾功能不全',
                'Step 2 - 病理生理分析: 长期CNI使用→肾毒性→1年20%受损，4年25%严重',
                'Step 3 - 机制关联: [Evidence-共识] 钙调磷酸酶抑制剂→肾小管毒性→肾功能下降',
                'Step 4 - 干预选择: 按共识方案调整免疫抑制剂',
                'Step 5 - 预期效果: 肾功能稳定/改善'
            ],
            'monitoring': ['肌酐: 每日', 'GFR: 每周', '他克莫司谷浓度: 每日'],
            'confidence': 0.85,
            'warnings': ['CNI肾毒性是移植后常见并发症', '减量需平衡排斥风险']
        },
        'GFR_Low': {
            'intervention': 'CKD监测方案（共识）',
            'intervention_details': {
                'drug': 'CNI减量/转换mTOR抑制剂',
                'dose': '根据CKD分期调整',
                'target': 'GFR稳定或改善'
            },
            'target_value': 60,
            'target_range': [60, 120],
            'reasoning_chain': [
                'Step 1 - 观察: GFR<60 mL/min，慢性肾病',
                'Step 2 - 病理生理分析: 术前心衰低灌注+长期利尿剂→肾储备差',
                'Step 3 - 机制关联: [Evidence-共识] 术后低心排+CNI肾毒性→GFR下降',
                'Step 4 - 干预选择: CKD分期管理，考虑mTOR抑制剂转换',
                'Step 5 - 预期效果: GFR下降速率减缓'
            ],
            'monitoring': ['GFR: 每月', '肌酐: 每周', '尿蛋白: 每月'],
            'confidence': 0.75,
            'warnings': ['年降>4 mL/min为警示', '需肾内科协同管理']
        },
        'EF_Low': {
            'intervention': '急性排斥评估与治疗（共识）',
            'intervention_details': {
                'drug': '甲泼尼龙冲击',
                'dose': '甲泼尼龙 1000mg/日×3天（<50kg: 15mg/kg）',
                'target': 'LVEF > 50%'
            },
            'target_value': 55,
            'target_range': [50, 70],
            'reasoning_chain': [
                'Step 1 - 观察: LVEF下降，供者标准>50%',
                'Step 2 - 病理生理分析: LVEF<50%提示急性排斥或移植心功能障碍',
                'Step 3 - 机制关联: [Evidence-共识] 急性排斥→心肌炎症→收缩功能下降',
                'Step 4 - 干预选择: 甲泼尼龙冲击治疗（共识急性排斥方案）',
                'Step 5 - 预期效果: 排斥控制后LVEF恢复'
            ],
            'monitoring': ['LVEF: 每12小时超声', 'cTnI: 每6小时', 'BNP: 每日'],
            'confidence': 0.80,
            'warnings': ['需排除其他LVEF下降原因', '活检确认排斥类型']
        },
        'Bilirubin_High': {
            'intervention': '肝功能评估（共识）',
            'intervention_details': {
                'drug': '对症处理 + 药物调整',
                'dose': '根据肝功能分级调整免疫抑制剂剂量',
                'target': '胆红素 < 2.0 mg/dL'
            },
            'target_value': 1.0,
            'target_range': [0.1, 1.2],
            'reasoning_chain': [
                'Step 1 - 观察: 胆红素升高，肝功能异常',
                'Step 2 - 病理生理分析: >2.5mg/dL提示肝功能不全',
                'Step 3 - 机制关联: [Evidence-共识] 右心衰/药物毒性→肝淤血→胆红素升高',
                'Step 4 - 干预选择: 明确病因（右心衰vs药物vs感染），针对性处理',
                'Step 5 - 预期效果: 肝功能改善'
            ],
            'monitoring': ['胆红素: 每日', '转氨酶: 每日', '凝血功能: 每2日'],
            'confidence': 0.70,
            'warnings': ['>2.5mg/dL为肝功能不全标志', '需排除药物性肝损']
        }
    }

    def __init__(self, llm_client=None, use_llm: bool = False):
        """
        初始化

        Args:
            llm_client: LLM客户端（如OpenAI/Anthropic客户端）
            use_llm: 是否使用LLM生成（False则使用预定义库）
        """
        self.llm_client = llm_client
        self.use_llm = use_llm and llm_client is not None

    def generate_strategy(
        self,
        indicator: str,
        current_value: float,
        direction: str,
        evidences: List[Evidence],
        target_config: Dict
    ) -> StrategyRecommendation:
        """
        生成策略推荐

        Args:
            indicator: 指标名称
            current_value: 当前值
            direction: 异常方向 ('high' or 'low')
            evidences: 支撑证据列表
            target_config: 目标值配置

        Returns:
            StrategyRecommendation
        """
        # 构建策略key
        strategy_key = f"{indicator}_{direction.capitalize()}"

        if self.use_llm:
            return self._generate_with_llm(
                indicator, current_value, direction,
                evidences, target_config, strategy_key
            )
        else:
            return self._generate_from_library(
                indicator, current_value, direction,
                evidences, target_config, strategy_key
            )

    def _generate_from_library(
        self,
        indicator: str,
        current_value: float,
        direction: str,
        evidences: List[Evidence],
        target_config: Dict,
        strategy_key: str
    ) -> StrategyRecommendation:
        """从预定义库生成策略"""

        # 获取预定义策略
        predefined = self.INTERVENTION_LIBRARY.get(strategy_key, {})

        if not predefined:
            # 没有预定义策略，生成通用建议
            return self._generate_generic_strategy(
                indicator, current_value, direction, evidences, target_config
            )

        # 整合证据到推理链
        reasoning_chain = predefined.get('reasoning_chain', []).copy()
        if evidences:
            # 在推理链中嵌入实际证据
            evidence_refs = []
            for i, ev in enumerate(evidences[:3]):
                evidence_refs.append(f"[Evidence-{i+1}] {ev.statement}")

            # 更新推理链中的证据引用
            if reasoning_chain and len(reasoning_chain) > 2:
                reasoning_chain[2] = f"Step 3 - 机制关联: 基于KG证据 - " + "; ".join(evidence_refs[:2])

        return StrategyRecommendation(
            indicator=indicator,
            current_value=current_value,
            target_value=predefined.get('target_value', target_config.get('target_min', 0)),
            target_range=(
                predefined.get('target_range', [target_config.get('target_min', 0)])[0],
                predefined.get('target_range', [0, target_config.get('target_max', 100)])[1]
            ),
            unit=target_config.get('unit', ''),
            intervention=predefined.get('intervention', '评估并干预'),
            intervention_type=predefined.get('intervention_details', {}).get('drug', 'general'),
            priority=1 if direction == 'low' and indicator in ['MAP', 'CI'] else 2,
            reasoning_chain=reasoning_chain,
            supporting_evidence=evidences,
            expected_effect=predefined.get('expected_effect', '指标恢复至目标范围'),
            monitoring_points=predefined.get('monitoring', []),
            confidence=predefined.get('confidence', 0.7)
        )

    def _generate_generic_strategy(
        self,
        indicator: str,
        current_value: float,
        direction: str,
        evidences: List[Evidence],
        target_config: Dict
    ) -> StrategyRecommendation:
        """生成通用策略"""

        target_min = target_config.get('target_min', current_value * 0.8)
        target_max = target_config.get('target_max', current_value * 1.2)
        unit = target_config.get('unit', '')

        # 基于证据构建推理链
        reasoning = [
            f"Step 1 - 观察: {indicator} = {current_value} {unit}，{'偏低' if direction == 'low' else '偏高'}",
            f"Step 2 - 病理生理分析: 指标异常可能影响器官功能",
        ]

        if evidences:
            ev_text = "; ".join([e.statement for e in evidences[:2]])
            reasoning.append(f"Step 3 - 机制关联: 根据证据 - {ev_text}")
        else:
            reasoning.append("Step 3 - 机制关联: 需进一步评估")

        reasoning.extend([
            f"Step 4 - 干预选择: 针对性调整以恢复{indicator}至目标范围",
            f"Step 5 - 预期效果: {indicator}恢复至 {target_min}-{target_max} {unit}"
        ])

        return StrategyRecommendation(
            indicator=indicator,
            current_value=current_value,
            target_value=(target_min + target_max) / 2,
            target_range=(target_min, target_max),
            unit=unit,
            intervention=f"评估并调整{indicator}",
            intervention_type="evaluation",
            priority=2,
            reasoning_chain=reasoning,
            supporting_evidence=evidences,
            expected_effect=f"{indicator}恢复至目标范围",
            monitoring_points=[f"{indicator}: 每30分钟监测"],
            confidence=0.5
        )

    def _generate_with_llm(
        self,
        indicator: str,
        current_value: float,
        direction: str,
        evidences: List[Evidence],
        target_config: Dict,
        strategy_key: str
    ) -> StrategyRecommendation:
        """使用LLM生成策略（需要LLM客户端）"""
        # 构建证据文本
        evidence_text = ""
        for i, ev in enumerate(evidences):
            evidence_text += f"[Evidence-{i+1}] {ev.statement} (来源: {ev.source_origin}, 关系: {ev.relation})\n"

        if not evidence_text:
            evidence_text = "无直接证据，请基于临床知识推理"

        # 构建prompt
        prompt = self.STRATEGY_PROMPT_TEMPLATE.format(
            indicator=indicator,
            current_value=current_value,
            unit=target_config.get('unit', ''),
            direction='偏低' if direction == 'low' else '偏高',
            target_min=target_config.get('target_min', '未知'),
            target_max=target_config.get('target_max', '未知'),
            evidence_text=evidence_text
        )

        # 调用LLM（这里需要实际的LLM客户端）
        # response = self.llm_client.generate(prompt)
        # result = json.loads(response)

        # 暂时返回预定义策略
        return self._generate_from_library(
            indicator, current_value, direction, evidences, target_config, strategy_key
        )


class EvidenceStrategyEngine:
    """证据驱动策略引擎 - 主接口"""

    def __init__(self, neo4j_connector=None, llm_client=None, use_llm: bool = False):
        self.evidence_collector = KGEvidenceCollector(neo4j_connector)
        self.strategy_generator = LLMStrategyGenerator(llm_client, use_llm)
        self.neo4j = neo4j_connector

    def set_neo4j(self, connector):
        """设置Neo4j连接器"""
        self.neo4j = connector
        self.evidence_collector.set_neo4j(connector)

    def analyze_baseline(
        self,
        measurements: Dict[str, float],
        sample_id: str = "unknown",
        timestamp_min: int = 30
    ) -> BaselineStrategyReport:
        """
        分析Baseline状态并生成策略报告

        Args:
            measurements: 指标测量值字典
            sample_id: 样本ID
            timestamp_min: 时间点（分钟）

        Returns:
            BaselineStrategyReport
        """
        abnormalities = []
        recommendations = []

        # 检测异常
        for indicator, value in measurements.items():
            target_config = self.evidence_collector.get_target_values(indicator)

            if not target_config.get('target_min'):
                continue

            # 判断异常方向
            direction = None
            if value < target_config['target_min']:
                direction = 'low'
            elif value > target_config['target_max']:
                direction = 'high'

            if direction:
                abnormalities.append({
                    'indicator': indicator,
                    'value': value,
                    'direction': direction,
                    'target_min': target_config['target_min'],
                    'target_max': target_config['target_max'],
                    'unit': target_config['unit']
                })

                # 收集证据
                evidences = self.evidence_collector.collect_evidence_for_abnormality(
                    indicator, value, direction
                )

                # 生成策略
                strategy = self.strategy_generator.generate_strategy(
                    indicator, value, direction, evidences, target_config
                )
                recommendations.append(strategy)

        # 按优先级排序
        recommendations.sort(key=lambda x: (x.priority, -x.confidence))

        # 计算总体风险
        risk_level = self._calculate_risk_level(abnormalities)

        # 生成摘要
        summary = self._generate_summary(abnormalities, recommendations)

        return BaselineStrategyReport(
            sample_id=sample_id,
            timestamp_min=timestamp_min,
            measurements=measurements,
            detected_abnormalities=abnormalities,
            recommendations=recommendations,
            overall_risk_level=risk_level,
            summary=summary
        )

    def _calculate_risk_level(self, abnormalities: List[Dict]) -> str:
        """计算总体风险等级"""
        if not abnormalities:
            return "LOW"

        critical_indicators = {'MAP', 'CI', 'K_A', 'PVR', 'PASP', 'pH', 'EF'}
        has_critical = any(a['indicator'] in critical_indicators for a in abnormalities)

        if len(abnormalities) >= 3 or has_critical:
            return "HIGH"
        elif len(abnormalities) >= 2:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_summary(
        self,
        abnormalities: List[Dict],
        recommendations: List[StrategyRecommendation]
    ) -> str:
        """生成报告摘要"""
        if not abnormalities:
            return "所有指标在正常范围内，建议继续监测。"

        summary_parts = [f"检测到 {len(abnormalities)} 项指标异常:"]

        for abn in abnormalities[:3]:
            direction_text = "偏低" if abn['direction'] == 'low' else "偏高"
            summary_parts.append(
                f"- {abn['indicator']}: {abn['value']} {abn['unit']} ({direction_text})"
            )

        if recommendations:
            top_rec = recommendations[0]
            summary_parts.append(f"\n首要干预: {top_rec.intervention}")
            summary_parts.append(f"目标: {top_rec.indicator} → {top_rec.target_range[0]}-{top_rec.target_range[1]} {top_rec.unit}")

        return "\n".join(summary_parts)

    def format_report(self, report: BaselineStrategyReport) -> str:
        """格式化输出报告"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"EVIDENCE-DRIVEN STRATEGY REPORT")
        lines.append(f"Sample: {report.sample_id} | Time: t={report.timestamp_min}min | Risk: {report.overall_risk_level}")
        lines.append("=" * 80)

        lines.append(f"\n## Summary\n{report.summary}")

        if report.recommendations:
            lines.append("\n## Detailed Recommendations")
            lines.append("-" * 80)

            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"\n### [{i}] {rec.indicator}: {rec.current_value} → {rec.target_value} {rec.unit}")
                lines.append(f"**Intervention**: {rec.intervention}")
                lines.append(f"**Target Range**: {rec.target_range[0]} - {rec.target_range[1]} {rec.unit}")
                lines.append(f"**Confidence**: {rec.confidence:.0%}")
                lines.append(f"**Priority**: {'URGENT' if rec.priority == 1 else 'Standard'}")

                lines.append("\n**Reasoning Chain (CoT)**:")
                for step in rec.reasoning_chain:
                    lines.append(f"  {step}")

                if rec.supporting_evidence:
                    lines.append("\n**Supporting Evidence**:")
                    for j, ev in enumerate(rec.supporting_evidence[:3], 1):
                        lines.append(f"  [{j}] {ev.statement}")
                        lines.append(f"      Source: {ev.source_origin} | Relation: {ev.relation}")

                lines.append("\n**Monitoring Points**:")
                for mp in rec.monitoring_points:
                    lines.append(f"  - {mp}")

                lines.append("-" * 40)

        return "\n".join(lines)


def test_evidence_strategy():
    """测试证据驱动策略引擎"""
    print("=" * 80)
    print("Testing Evidence-Driven Strategy Engine")
    print("=" * 80)

    # 创建引擎（无Neo4j）
    engine = EvidenceStrategyEngine()

    # 测试数据
    test_measurements = {
        'MAP': 45,      # 低
        'Lactate': 5.2, # 高
        'SvO2': 72,     # 正常
        'K_A': 3.0,     # 低
        'CvO2': 5.5,    # 低
    }

    # 生成报告
    report = engine.analyze_baseline(test_measurements, sample_id="TEST-001", timestamp_min=30)

    # 输出
    print(engine.format_report(report))


def test_with_neo4j():
    """测试Neo4j集成"""
    try:
        from neo4j_connector import Neo4jKnowledgeGraph

        print("\n" + "=" * 80)
        print("Testing with Neo4j Integration")
        print("=" * 80)

        # 连接Neo4j
        neo4j = Neo4jKnowledgeGraph()

        # 创建引擎
        engine = EvidenceStrategyEngine(neo4j_connector=neo4j)

        # 测试数据
        test_measurements = {
            'MAP': 42,
            'Lactate': 4.8,
            'K_A': 2.8,
            'CvO2': 5.2,
        }

        # 生成报告
        report = engine.analyze_baseline(test_measurements, sample_id="NEO4J-TEST", timestamp_min=30)

        print(engine.format_report(report))

        neo4j.close()

    except Exception as e:
        print(f"Neo4j test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "neo4j":
        test_with_neo4j()
    else:
        test_evidence_strategy()
