# Multi-Agent Neuro-Symbolic AI：心脏灌注预测系统
## 完整架构与实施方案

---

## 📋 文档概览

- **系统目标**：预测心脏灌注质量，提供实时介入建议
- **输入**：灌注策略 + 异构心脏数据（文本+时序+结构化） + 患者病历
- **输出**：质量评分 + 风险评估 + 策略评估 + 介入推荐
- **团队配置**：您（KG/RAG专家）+ Claude + 1名研究生
- **预计工期**：3.5-4个月

---

## 一、系统架构总览

### 1.1 核心理念

**不是单体RAG，而是7个独立Agent协同工作**

```
输入数据 → Agent1(理解) → Agent2(检索) → Agent3(推理) → Agent4(证据)
                                                    ↓
输出结果 ← Agent7(推荐) ← Agent6(评估) ← Agent5(预测)
                    ↑_____ Orchestrator协调 _____↑
```

### 1.2 7个Agent职责速查表

| Agent | 名称 | 输入 | 输出 | 核心技术 | 负责人 |
|-------|------|------|------|---------|-------|
| **Agent 1** | 输入理解 | 原始异构数据 | 标准化特征向量 | ClinicalBERT + LSTM | 研究生 |
| **Agent 2** | 知识检索 | 标准化特征 | 子图+文献Top-20 | Neo4j + ChromaDB | **您** |
| **Agent 3** | 神经符号推理 | 检索结果 | 推理路径+因果链 | Temporal-GNN + Prolog | Claude + 您 |
| **Agent 4** | 证据综合 | 推理结果 | 证据链+质量评分 | GRADE + Meta-analysis | Claude |
| **Agent 5** | 灌注预测 | 推理+证据 | 质量评分+风险概率 | GBM+LSTM+GNN集成 | 研究生 + Claude |
| **Agent 6** | 策略评估 | 预测结果 | 问题识别+偏差分析 | 规则引擎 | Claude |
| **Agent 7** | 介入推荐 | 评估结果 | Top-5推荐+监测方案 | 因果效应估计 | Claude |
| **Orchestrator** | 协调器 | 全部 | 最终报告 | Pipeline管理 | Claude |

---

## 二、输入输出详细规格

### 2.1 输入数据（三类异构）

#### 输入1: 拟定灌注策略
```json
{
  "perfusion_strategy": {
    "method": "HTK solution",
    "temperature": 4,        // °C
    "pressure": 60,          // mmHg ⚠️ 可能偏低
    "flow_rate": 1.2,        // L/min
    "duration": 240,         // minutes
    "additives": ["adenosine", "insulin"],
    "delivery_mode": "antegrade"
  }
}
```

#### 输入2: 异构心脏数据

**2.1 自由文本**（由Agent 1用ClinicalBERT编码）
```json
{
  "cardiac_description": {
    "visual_inspection": "Heart appears mildly hypertrophied with no visible scarring. Coronary arteries patent.",
    "palpation_notes": "Firm consistency, no areas of induration.",
    "procurement_notes": "Cross-clamp time 32 minutes."
  }
}
```

**2.2 时序血气数据**（由Agent 1用LSTM编码）
```json
{
  "blood_gas_data": {
    "pre_perfusion": {"lactate": 2.8, "pH": 7.32, "pO2": 280},
    "during_perfusion": [
      {"time": 60, "lactate": 1.8, "pH": 7.38},
      {"time": 120, "lactate": 1.2, "pH": 7.40},
      {"time": 180, "lactate": 0.9, "pH": 7.42}
    ],
    "post_perfusion": {"lactate": 0.6, "pH": 7.44}
  }
}
```

#### 输入3: 患者病历
```json
{
  "recipient_medical_record": {
    "demographics": {"age": 55, "gender": "male", "weight": 78},
    "diagnosis": "dilated cardiomyopathy",
    "comorbidities": ["diabetes", "hypertension", "CKD stage 3"],
    "hemodynamics": {"LVEF": 15, "PVR": 3.2}
  }
}
```

### 2.2 输出格式（完整决策报告）

```json
{
  "perfusion_outcome_prediction": {
    "overall_score": 0.78,              // 0-1量表
    "confidence_interval": [0.68, 0.88], // 95% CI
    "risk_assessment": {
      "ischemia_reperfusion_injury": 0.23,
      "endothelial_dysfunction": 0.18,
      "metabolic_recovery_failure": 0.12,
      "primary_graft_dysfunction": 0.15
    },
    "predicted_metrics": {
      "post_reperfusion_lactate": 1.2,  // mmol/L
      "cardiac_output_24h": 4.5,        // L/min
      "time_to_hemodynamic_stability": 6 // hours
    }
  },

  "strategy_evaluation": {
    "adequacy_level": "suboptimal",
    "identified_issues": [
      {
        "issue": "Perfusion pressure too low (60 mmHg)",
        "impact": "Inadequate coronary perfusion in hypertrophied heart",
        "severity": "moderate"
      },
      {
        "issue": "Lactate clearance slow",
        "impact": "Metabolic recovery delayed",
        "severity": "mild"
      }
    ]
  },

  "recommended_interventions": [
    {
      "intervention": "Increase perfusion pressure to 75-80 mmHg",
      "rationale": "8 studies (OR=1.87, p=0.004): Higher pressure improves outcomes in hypertrophied hearts",
      "expected_benefit": "Reduce ischemia risk from 23% to 14%",
      "priority": "high",
      "timing": "immediate",
      "evidence_level": "A"
    },
    {
      "intervention": "Add glucose-insulin-potassium (GIK)",
      "rationale": "Meta-analysis (n=1,243): Accelerates lactate clearance by 35%",
      "expected_benefit": "+0.12 quality score",
      "priority": "moderate",
      "timing": "next cycle",
      "evidence_level": "B"
    }
  ],

  "real_time_monitoring_plan": {
    "critical_parameters": ["lactate", "pH", "coronary_flow"],
    "alert_thresholds": {
      "lactate_increase": ">0.5 mmol/L per hour",
      "pH_drop": "<7.35"
    },
    "intervention_triggers": [
      {"condition": "lactate > 2.0 at 2h", "action": "Increase flow rate by 20%"}
    ]
  }
}
```

---

## 三、Agent详细设计

### Agent 1: Input Understanding Agent（输入理解）

**负责人：研究生（NLP+时序专家）**

#### 职责
将异构数据（文本+时序+结构化）转换为标准化向量表示

#### 技术实现

```python
class InputUnderstandingAgent:
    def __init__(self):
        # 1. 文本编码器
        self.text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # 2. 时序编码器
        self.lstm = nn.LSTM(
            input_size=6,    # lactate, pH, pO2, pCO2, K+, glucose
            hidden_size=128,
            num_layers=2,
            bidirectional=True
        )

        # 3. 实体识别
        self.ner = MedicalNER()  # 识别药物、疾病、指标

        # 4. 特征提取
        self.feature_extractor = FeatureExtractor()

    def process(self, raw_input: Dict) -> StandardizedInput:
        # 1. 文本理解
        text = raw_input['cardiac_description']['visual_inspection']
        text_embedding = self.text_encoder.encode(text)  # → 768-dim

        # 提取关键特征
        cardiac_features = {
            'hypertrophy_level': self._extract_hypertrophy(text),  # 0-1
            'contractility_score': self._extract_contractility(text),
            'valve_status': self._extract_valve(text)
        }

        # 2. 时序数据编码
        blood_gas_sequence = self._prepare_sequence(
            raw_input['blood_gas_data']['during_perfusion']
        )  # Shape: [T, 6]

        lstm_out, (h_n, c_n) = self.lstm(blood_gas_sequence)
        blood_gas_embedding = h_n[-1]  # → 128-dim

        # 计算趋势特征
        lactate_clearance_rate = self._compute_slope(
            blood_gas_sequence[:, 0]  # lactate column
        )

        # 3. 策略参数标准化
        strategy_vector = self.feature_extractor.extract(
            raw_input['perfusion_strategy']
        )  # → 20-dim

        # 归一化（相对于推荐范围）
        pressure_normalized = (strategy_vector['pressure'] - 50) / 30  # [50-80]范围

        # 4. 患者风险画像
        patient_profile = self._compute_risk_profile(
            raw_input['recipient_medical_record']
        )  # → 50-dim

        # 聚合输出
        return StandardizedInput(
            # 文本特征
            cardiac_text_embedding=text_embedding,
            cardiac_features=cardiac_features,

            # 时序特征
            blood_gas_embedding=blood_gas_embedding,
            lactate_clearance_rate=lactate_clearance_rate,
            ph_stability=self._compute_stability(blood_gas_sequence[:, 1]),

            # 策略特征
            strategy_params=strategy_vector,
            pressure_adequacy=pressure_normalized,

            # 患者特征
            patient_profile=patient_profile,
            risk_factors=['diabetes', 'hypertrophy', 'CKD'],

            # 实体
            extracted_entities=self.ner.extract_all(text)
        )
```

#### 关键输出
```python
StandardizedInput(
    # 维度设计
    cardiac_text_embedding=torch.tensor([...]),  # 768-dim
    blood_gas_embedding=torch.tensor([...]),     # 128-dim
    strategy_params=torch.tensor([...]),         # 20-dim
    patient_profile=torch.tensor([...]),         # 50-dim

    # 可解释特征
    cardiac_features={
        'hypertrophy_level': 0.6,
        'contractility_score': 0.8,
        'valve_competence': 'good'
    },
    metabolic_trajectory={
        'lactate_clearance_rate': -0.02,  # mmol/L per min (负值=下降=好)
        'pH_stability': 0.95,
        'oxygenation_trend': 'improving'
    }
)
```

#### 工作量估算
- **ClinicalBERT微调**：3-4天（在心脏描述数据上）
- **LSTM训练**：2-3天（血气时序预测）
- **特征工程**：2-3天（策略参数、患者特征）
- **集成测试**：2天
- **总计**：2-2.5周

---

### Agent 2: Knowledge Retrieval Agent（知识检索）

**负责人：您（KG/RAG专家）**

#### 职责
从知识图谱和向量库检索灌注相关知识

#### 图谱Schema设计

```cypher
// 核心节点类型
CREATE (s:PerfusionStrategy {
  method: 'HTK solution',
  temperature: 4,
  pressure: 75,
  flow_rate: 1.5
})

CREATE (o:Outcome {
  quality_score: 0.85,
  lactate_final: 0.8,
  complications: []
})

CREATE (m:BloodGasMarker {
  name: 'lactate',
  normal_range: [0.5, 1.0],
  critical_threshold: 2.0
})

CREATE (i:Intervention {
  name: 'Increase pressure to 75-80 mmHg',
  mechanism: 'Improve coronary flow',
  evidence_level: 'A'
})

CREATE (r:RiskFactor {
  name: 'cardiac_hypertrophy',
  odds_ratio: 2.3,
  p_value: 0.004
})

// 关系类型
CREATE (s)-[:RESULTS_IN {probability: 0.78, n_studies: 15}]->(o)
CREATE (m)-[:INDICATES {threshold: 2.0}]->(o)
CREATE (i)-[:IMPROVES {effect_size: 0.15, confidence: 0.85}]->(o)
CREATE (r)-[:PREDISPOSES_TO {mechanism: 'reduced_flow'}]->(c:Complication)
```

#### 检索策略

```python
class KnowledgeRetrievalAgent:
    def __init__(self):
        # 1. 图数据库
        self.graph_db = GraphDatabase.driver(
            "neo4j://localhost:7687",
            auth=("neo4j", "password")
        )

        # 2. 向量数据库
        self.vector_db = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.vector_db.get_or_create_collection(
            name="perfusion_literature",
            metadata={"hnsw:space": "cosine"}
        )

        # 3. 灌注本体
        self.ontology = PerfusionOntology()

    def retrieve(self, standardized_input: StandardizedInput) -> RetrievalResult:
        # 策略1: 基于策略参数的图检索
        similar_strategies = self._graph_query_strategies(
            method=standardized_input.strategy_params.method,
            pressure=standardized_input.strategy_params.pressure,
            temperature=standardized_input.strategy_params.temperature
        )

        # 策略2: 基于血气趋势的向量检索
        similar_cases = self._vector_search_cases(
            query_embedding=standardized_input.blood_gas_embedding,
            filters={
                'lactate_clearance': {'$lt': 0},  # 找lactate清除差的
                'has_hypertrophy': True
            },
            top_k=20
        )

        # 策略3: 基于患者特征的风险检索
        risk_pathways = self._graph_query_risks(
            comorbidities=['diabetes', 'CKD'],
            hypertrophy=True
        )

        # 策略4: 介入措施检索
        interventions = self._graph_query_interventions(
            current_issues=['low_pressure', 'slow_lactate_clearance']
        )

        # 聚合结果
        return RetrievalResult(
            similar_strategies=similar_strategies,     # 15个相似策略
            outcome_data=[s.outcome for s in similar_strategies],
            relevant_literature=similar_cases,         # Top-20文献
            risk_pathways=risk_pathways,               # 5条风险路径
            intervention_options=interventions,         # 12种介入措施
            subgraph=self._construct_subgraph(...)    # 子图
        )

    def _graph_query_strategies(self, method, pressure, temperature):
        """检索相似灌注策略"""
        query = """
        MATCH path = (s:PerfusionStrategy)-[r:RESULTS_IN]->(o:Outcome)
        WHERE s.method = $method
          AND s.pressure BETWEEN $pressure - 10 AND $pressure + 10
          AND s.temperature = $temperature
        RETURN s, r, o, r.probability AS prob
        ORDER BY prob DESC
        LIMIT 15
        """
        with self.graph_db.session() as session:
            result = session.run(query,
                method=method,
                pressure=pressure,
                temperature=temperature
            )
            return [record for record in result]

    def _vector_search_cases(self, query_embedding, filters, top_k):
        """向量检索相似案例"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            where=filters,
            n_results=top_k
        )
        return results

    def _graph_query_risks(self, comorbidities, hypertrophy):
        """检索风险路径"""
        query = """
        MATCH path = (p:PatientProfile)-[:HAS_RISK]->(r:RiskFactor)
                     -[:INCREASES_RISK]->(c:Complication)
        WHERE ALL(cond IN $comorbidities WHERE cond IN p.comorbidities)
          AND p.cardiac_hypertrophy = $hypertrophy
        RETURN path, r.odds_ratio AS or, r.p_value AS p
        ORDER BY or DESC
        """
        with self.graph_db.session() as session:
            result = session.run(query,
                comorbidities=comorbidities,
                hypertrophy=hypertrophy
            )
            return [record for record in result]

    def _graph_query_interventions(self, current_issues):
        """检索介入措施"""
        query = """
        MATCH (issue:PerfusionIssue)-[:TREATED_BY]->(i:Intervention)
        WHERE issue.type IN $issues
        RETURN i, i.evidence_level AS level, i.effect_size AS effect
        ORDER BY effect DESC
        """
        with self.graph_db.session() as session:
            result = session.run(query, issues=current_issues)
            return [record for record in result]
```

#### 工作量估算
- **Neo4j Schema设计**：3-4天
- **24k文章导入图谱**：5-7天（您擅长，可能更快）
- **ChromaDB向量库构建**：2-3天
- **Cypher查询优化**：3-4天
- **总计**：2.5-3周

---

### Agent 3: Neuro-Symbolic Reasoning Engine（神经符号推理）

**负责人：Claude（算法实现） + 您（图部分）**

#### 职责
基于检索结果进行因果推理和预测

#### 核心组件

##### 3.1 Temporal-GNN（时序图神经网络）

```python
class TemporalGNN(nn.Module):
    """
    处理时序血气数据的图神经网络
    结合知识图谱进行推理
    """
    def __init__(self, node_dim=128, hidden_dim=256, time_steps=10):
        super().__init__()

        # 1. 时序编码层
        self.temporal_encoder = nn.GRU(
            input_size=node_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 2. 图卷积层（多头注意力）
        self.graph_convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            for _ in range(3)
        ])

        # 3. 时序注意力
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8
        )

        # 4. 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self,
                blood_gas_sequence: torch.Tensor,  # [batch, time, features]
                graph_structure: Data):             # PyG Data object

        batch_size, time_steps, _ = blood_gas_sequence.shape

        # Step 1: 时序编码
        temporal_features, h_n = self.temporal_encoder(blood_gas_sequence)
        # temporal_features: [batch, time, hidden_dim]

        # Step 2: 对每个时间步进行图传播
        graph_enhanced_features = []
        for t in range(time_steps):
            x_t = temporal_features[:, t, :]  # [batch, hidden_dim]

            # 图卷积（融合知识图谱）
            for conv in self.graph_convs:
                x_t = conv(x_t, graph_structure.edge_index)
                x_t = F.relu(x_t)

            graph_enhanced_features.append(x_t)

        graph_enhanced_features = torch.stack(graph_enhanced_features, dim=1)
        # Shape: [batch, time, hidden_dim]

        # Step 3: 时序注意力（找关键时间点）
        attn_output, attn_weights = self.temporal_attention(
            query=graph_enhanced_features,
            key=graph_enhanced_features,
            value=graph_enhanced_features
        )

        # Step 4: 预测
        final_state = attn_output[:, -1, :]  # 取最后时间步
        prediction = self.predictor(final_state)

        return {
            'prediction': prediction,
            'attention_weights': attn_weights,
            'graph_enhanced_features': graph_enhanced_features
        }
```

##### 3.2 因果推断模块

```python
class CausalInferenceEngine:
    """
    估计介入措施的因果效应
    使用双重稳健估计（Doubly Robust）
    """
    def __init__(self):
        self.causal_model = CausalModel()

    def estimate_ate(self,
                     treatment: str,
                     outcome: str,
                     confounders: List[str],
                     data: pd.DataFrame) -> Dict:
        """
        估计平均处理效应（ATE）

        示例:
        treatment='increase_pressure'
        outcome='lactate_clearance_rate'
        confounders=['hypertrophy', 'ischemic_time', 'diabetes']
        """

        # 1. 构建因果图
        causal_graph = """
        digraph {
            hypertrophy -> increase_pressure;
            hypertrophy -> lactate_clearance_rate;
            ischemic_time -> increase_pressure;
            ischemic_time -> lactate_clearance_rate;
            diabetes -> lactate_clearance_rate;
            increase_pressure -> lactate_clearance_rate;
        }
        """

        # 2. 识别因果效应
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph
        )

        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True
        )

        # 3. 估计ATE（使用多种方法）
        estimates = {}

        # 3.1 倾向得分加权（IPW）
        estimate_ipw = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_weighting"
        )
        estimates['ipw'] = estimate_ipw.value

        # 3.2 双重稳健估计
        estimate_dr = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dr.LinearDRLearner"
        )
        estimates['doubly_robust'] = estimate_dr.value

        # 3.3 工具变量（如果有）
        # estimate_iv = model.estimate_effect(...)

        # 4. 反事实分析
        counterfactual = self._counterfactual_analysis(
            model, treatment, outcome, data
        )

        # 5. 敏感性分析
        sensitivity = model.refute_estimate(
            identified_estimand,
            estimate_ipw,
            method_name="random_common_cause"
        )

        return {
            'ate': np.mean(list(estimates.values())),
            'estimates': estimates,
            'confidence_interval': self._compute_ci(estimates),
            'counterfactual': counterfactual,
            'sensitivity': sensitivity,
            'interpretation': self._interpret_ate(np.mean(list(estimates.values())))
        }

    def _counterfactual_analysis(self, model, treatment, outcome, data):
        """
        反事实分析：如果改变treatment，outcome会如何？
        """
        # 示例：如果将pressure从60增加到75，lactate会如何？
        treated_data = data.copy()
        treated_data[treatment] = 1  # 干预

        control_data = data.copy()
        control_data[treatment] = 0  # 不干预

        # 预测两种情况下的outcome
        y_treated = model.predict(treated_data)
        y_control = model.predict(control_data)

        return {
            'individual_effects': y_treated - y_control,
            'mean_effect': np.mean(y_treated - y_control),
            'effect_distribution': np.percentile(y_treated - y_control, [25, 50, 75])
        }
```

##### 3.3 符号推理引擎（Prolog）

```python
class PrologReasoningEngine:
    """
    基于规则的符号推理
    """
    def __init__(self):
        self.prolog = Prolog()
        self._load_rules()

    def _load_rules(self):
        """加载灌注推理规则"""
        rules = """
        % 规则1: 低压力不足以应对肥厚心脏
        inadequate_perfusion(Strategy) :-
            pressure(Strategy, P), P < 65,
            cardiac_state(Heart, hypertrophy, Level), Level > 0.5.

        % 规则2: 糖尿病+高乳酸→高风险
        high_risk(Patient, ischemia_reperfusion) :-
            comorbidity(Patient, diabetes),
            blood_gas(PrePerfusion, lactate, L), L > 2.5.

        % 规则3: 乳酸清除慢→需要调整策略
        slow_lactate_clearance(BloodGas) :-
            time_point(BloodGas, T1, Lactate1),
            time_point(BloodGas, T2, Lactate2),
            T2 > T1 + 60,  % 60分钟后
            Clearance is (Lactate1 - Lactate2) / (T2 - T1),
            Clearance < 0.01.  % < 0.01 mmol/L/min

        % 规则4: 推荐增加压力
        recommend(increase_pressure) :-
            inadequate_perfusion(_),
            high_risk(_, ischemia_reperfusion).

        % 规则5: 推荐GIK（如果乳酸清除慢）
        recommend(add_gik) :-
            slow_lactate_clearance(_),
            not_contraindicated(gik).

        % 规则6: 监测建议
        monitor(troponin, frequent) :-
            high_risk(_, primary_graft_dysfunction).

        % 规则7: 风险传播
        risk_of(Patient, Complication) :-
            has_risk_factor(Patient, Risk),
            causes(Risk, Complication).

        % 事实库（从输入数据动态生成）
        % pressure(current_strategy, 60).
        % cardiac_state(donor_heart, hypertrophy, 0.6).
        % comorbidity(patient, diabetes).
        % blood_gas(pre, lactate, 2.8).
        """

        self.prolog.assertz(rules)

    def query(self, standardized_input: StandardizedInput) -> List[Dict]:
        """
        执行推理查询
        """
        # 1. 将输入数据转换为Prolog事实
        self._assert_facts(standardized_input)

        # 2. 查询推荐
        recommendations = list(self.prolog.query("recommend(X)"))
        # 返回: [{'X': 'increase_pressure'}, {'X': 'add_gik'}]

        # 3. 查询风险
        risks = list(self.prolog.query("risk_of(patient, X)"))
        # 返回: [{'X': 'ischemia_reperfusion'}, ...]

        # 4. 查询监测
        monitoring = list(self.prolog.query("monitor(X, Y)"))
        # 返回: [{'X': 'troponin', 'Y': 'frequent'}]

        # 5. 解释推理路径
        explanations = self._explain_reasoning(recommendations)

        return {
            'recommendations': [r['X'] for r in recommendations],
            'risks': [r['X'] for r in risks],
            'monitoring': monitoring,
            'explanations': explanations
        }

    def _assert_facts(self, input_data):
        """将输入数据转换为Prolog事实"""
        self.prolog.assertz(f"pressure(current_strategy, {input_data.strategy_params.pressure})")
        self.prolog.assertz(f"cardiac_state(donor_heart, hypertrophy, {input_data.cardiac_features['hypertrophy_level']})")

        for comorbidity in input_data.risk_factors:
            self.prolog.assertz(f"comorbidity(patient, {comorbidity})")

        # ... 更多事实
```

#### Agent 3完整流程

```python
class NeuroSymbolicReasoningEngine:
    def __init__(self):
        self.gnn = TemporalGNN()
        self.causal_engine = CausalInferenceEngine()
        self.logic_engine = PrologReasoningEngine()
        self.uncertainty_quantifier = BayesianNN()

    def reason(self,
               retrieval: RetrievalResult,
               input_data: StandardizedInput) -> ReasoningResult:

        # 1. 神经推理：时序GNN预测血气趋势
        gnn_output = self.gnn(
            blood_gas_sequence=input_data.blood_gas_sequence,
            graph_structure=retrieval.subgraph
        )
        predicted_trend = gnn_output['prediction']

        # 2. 因果推理：估计介入效应
        causal_effects = {}
        for intervention in ['increase_pressure', 'add_gik', 'increase_flow']:
            effect = self.causal_engine.estimate_ate(
                treatment=intervention,
                outcome='lactate_clearance_rate',
                confounders=['hypertrophy', 'ischemic_time', 'diabetes'],
                data=retrieval.similar_cases_df
            )
            causal_effects[intervention] = effect

        # 3. 符号推理：规则引擎
        logic_output = self.logic_engine.query(input_data)

        # 4. 不确定性量化
        uncertainty = self.uncertainty_quantifier.predict(
            input_data.all_features,
            return_epistemic_aleatoric=True
        )

        # 5. 融合三种推理结果
        final_reasoning = self._fuse_reasoning(
            neural=gnn_output,
            causal=causal_effects,
            symbolic=logic_output
        )

        return ReasoningResult(
            predicted_outcome=final_reasoning['outcome'],
            causal_effects=causal_effects,
            symbolic_recommendations=logic_output['recommendations'],
            uncertainty=uncertainty,
            reasoning_path=final_reasoning['explanation']
        )
```

#### 工作量估算
- **Temporal-GNN实现**：6-7天（Claude提供完整代码）
- **因果推断框架**：4-5天（Claude基于DoWhy）
- **Prolog规则库**：3-4天（Claude编写规则）
- **集成测试**：3天
- **总计**：3-3.5周（Claude承担大部分算法实现）

---

### Agent 4-7 简要设计

#### Agent 4: Evidence Synthesis（证据综合）
**负责人：Claude**

```python
class EvidenceSynthesisAgent:
    def __init__(self):
        self.grade_evaluator = GRADEEvaluator()
        self.meta_analyzer = MetaAnalyzer()

    def synthesize(self, reasoning: ReasoningResult) -> EvidenceSynthesis:
        # 1. GRADE评分
        evidence_quality = self.grade_evaluator.assess(
            claim="Pressure increase improves outcomes",
            studies=reasoning.supporting_studies
        )

        # 2. Meta分析
        if len(studies) >= 3:
            pooled_effect = self.meta_analyzer.pool(studies)

        # 3. 生成证据链
        return EvidenceSynthesis(
            evidence_chain=[...],
            overall_quality="Moderate to High",
            heterogeneity="Low (I²<25%)"
        )
```

**工作量：2周**

---

#### Agent 5: Perfusion Outcome Prediction（灌注预测）
**负责人：研究生 + Claude**

```python
class PerfusionOutcomePredictionAgent:
    def __init__(self):
        # 集成三种模型
        self.ensemble = EnsembleModel([
            GradientBoosting(),  # 静态特征
            LSTM(),              # 时序特征（研究生训练）
            GNN()                # 图特征（Claude提供）
        ])

    def predict(self, reasoning: ReasoningResult) -> PredictionResult:
        # 综合预测
        quality_score = self.ensemble.predict(features)

        # 风险评分
        risks = self.risk_scorer.calculate([
            'ischemia_reperfusion_injury',
            'endothelial_dysfunction',
            ...
        ])

        return PredictionResult(
            overall_quality_score=quality_score,
            risk_probabilities=risks,
            confidence_interval=(0.68, 0.88)
        )
```

**工作量：3-4周（研究生主导，Claude提供GNN）**

---

#### Agent 6: Strategy Evaluation（策略评估）
**负责人：Claude**

```python
class StrategyEvaluationAgent:
    def evaluate(self, strategy, prediction) -> StrategyEvaluation:
        # 1. 与指南对比
        guideline_compliance = self.guideline_checker.check(strategy)

        # 2. 识别问题
        issues = self._identify_issues(strategy, prediction)

        # 3. 敏感性分析
        sensitivity = self._sensitivity_analysis(strategy)

        return StrategyEvaluation(
            adequacy_level="suboptimal",
            identified_issues=issues,
            sensitivity_factors=sensitivity
        )
```

**工作量：1.5周**

---

#### Agent 7: Intervention Recommendation（介入推荐）
**负责人：Claude**

```python
class InterventionRecommendationAgent:
    def recommend(self, evaluation, prediction) -> RecommendationResult:
        # 1. 检索候选介入措施
        candidates = self.intervention_db.query(evaluation.issues)

        # 2. 估计因果效应
        for candidate in candidates:
            effect = self.effect_estimator.estimate(candidate)

        # 3. 优先级排序
        ranked = self.prioritizer.rank(candidates)

        return RecommendationResult(
            interventions=ranked[:5],
            monitoring_plan=self._generate_monitoring(...)
        )
```

**工作量：2周**

---

## 四、3人团队分工详细方案

### 分工总表

| 任务 | 负责人 | 时间 | 关键输出 |
|------|--------|------|---------|
| **Phase 1: 核心能力（2-3周）** |
| Agent 1: 多模态编码 | 研究生 | 2-2.5周 | ClinicalBERT + LSTM编码器 |
| Agent 2: 知识检索 | **您** | 2.5-3周 | Neo4j图谱 + ChromaDB |
| Agent 5: 基础预测 | 研究生+Claude | 1.5周 | LSTM预测模型 |
| **Phase 2: 增强推理（3-4周）** |
| Agent 3: Temporal-GNN | Claude+您 | 2.5周 | GNN算法+图集成 |
| Agent 3: 因果推断 | Claude | 1.5周 | ATE估计+反事实 |
| Agent 3: 符号推理 | Claude | 1周 | Prolog规则引擎 |
| Agent 4: 证据综合 | Claude | 2周 | GRADE+Meta分析 |
| **Phase 3: 完整系统（3-4周）** |
| Agent 5: 集成预测 | 研究生+Claude | 2周 | GBM+LSTM+GNN ensemble |
| Agent 6: 策略评估 | Claude | 1.5周 | 规则引擎+敏感性 |
| Agent 7: 介入推荐 | Claude | 2周 | 推荐引擎+监测 |
| Orchestrator | Claude | 1.5周 | Pipeline协调 |
| **Phase 4: 集成测试（2-3周）** |
| 系统集成 | 全员 | 2-3周 | 端到端测试 |

### 您的具体工作（25-30%工作量）

#### 核心职责
1. **Agent 2完整实现**（2.5-3周）
   - Neo4j图谱设计和构建
   - 24k篇文章数据导入
   - Cypher查询优化
   - ChromaDB向量库构建

2. **Agent 3图部分**（1-1.5周）
   - 为Temporal-GNN提供图结构
   - 图数据预处理
   - 与Claude协作调试GNN

3. **系统集成支持**（0.5周）
   - 检索API接口
   - 性能优化

#### 时间投入
- **前6周**：每周10-15小时（Agent 2开发）
- **第7-10周**：每周5-8小时（Agent 3协作）
- **第11-14周**：每周3-5小时（集成支持）
- **总计**：约120-150小时（1.5人月）

### 研究生的具体工作（100%工作量）

#### 核心职责
1. **Agent 1完整实现**（2-2.5周）
   - ClinicalBERT微调
   - LSTM训练
   - 特征工程

2. **Agent 5主要实现**（3-4周）
   - LSTM预测模型训练
   - 与GBM/GNN集成
   - 超参数调优

3. **数据管道**（贯穿全程）
   - 数据清洗和预处理
   - 实验跟踪（MLflow）

4. **系统部署**（最后2周）
   - FastAPI接口
   - Docker容器化

#### 所需技能
- ✅ Medical NLP（ClinicalBERT）
- ✅ 时序建模（LSTM/Transformer）
- ✅ PyTorch熟练
- ✅ 实验管理（MLflow）
- ✅ 基础深度学习

### Claude的具体工作

#### 核心职责
1. **所有核心算法实现**
   - Temporal-GNN完整代码
   - 因果推断框架
   - Prolog规则引擎
   - 集成预测模型

2. **Agent 4/6/7完整实现**
   - 证据综合
   - 策略评估
   - 介入推荐

3. **Orchestrator实现**
   - Agent协调逻辑
   - Pipeline管理

4. **技术支持**
   - 实时代码审查
   - Bug修复
   - 算法调优建议

---

## 五、实施路线图（14周详细计划）

### Week 1-3: Phase 1 - 核心能力

**目标：建立基础输入输出能力**

| Week | 任务 | 负责人 | 里程碑 |
|------|------|--------|--------|
| 1 | Agent 1: ClinicalBERT微调 | 研究生 | 文本编码器可用 |
| 1 | Agent 2: Neo4j Schema设计 | 您 | 图谱Schema确定 |
| 2 | Agent 1: LSTM训练 | 研究生 | 时序编码器可用 |
| 2 | Agent 2: 数据导入 | 您 | 5000篇文章入库 |
| 3 | Agent 1: 集成测试 | 研究生 | 完整输入理解可用 |
| 3 | Agent 2: Cypher查询 | 您 | 检索功能可用 |
| 3 | Agent 5: LSTM预测模型 | 研究生 | 基础预测可用 |

**Milestone 1（Week 3结束）**：
- ✅ 输入数据可以标准化
- ✅ 知识图谱可以检索
- ✅ 基础预测模型可以运行

---

### Week 4-7: Phase 2 - 增强推理

**目标：实现神经符号推理**

| Week | 任务 | 负责人 | 里程碑 |
|------|------|--------|--------|
| 4 | Agent 3: Temporal-GNN架构 | Claude | GNN代码框架 |
| 4 | Agent 2: 图结构准备 | 您 | 图数据ready |
| 5 | Agent 3: GNN训练 | Claude+研究生 | GNN第一版可用 |
| 5 | Agent 3: 因果推断框架 | Claude | DoWhy集成 |
| 6 | Agent 3: Prolog规则 | Claude | 规则引擎可用 |
| 6 | Agent 4: GRADE评估 | Claude | 证据评估可用 |
| 7 | Agent 3: 三种推理融合 | Claude | 完整推理引擎 |
| 7 | Agent 4: Meta分析 | Claude | 证据综合完成 |

**Milestone 2（Week 7结束）**：
- ✅ Temporal-GNN可以预测血气趋势
- ✅ 因果推断可以估计介入效应
- ✅ 符号推理可以生成规则推荐
- ✅ 证据综合可以评估质量

---

### Week 8-11: Phase 3 - 完整系统

**目标：完成所有7个Agent**

| Week | 任务 | 负责人 | 里程碑 |
|------|------|--------|--------|
| 8 | Agent 5: 集成模型（GBM） | 研究生 | GBM训练完成 |
| 8 | Agent 6: 策略评估 | Claude | 评估引擎可用 |
| 9 | Agent 5: Ensemble融合 | 研究生+Claude | 集成预测可用 |
| 9 | Agent 7: 介入推荐 | Claude | 推荐引擎可用 |
| 10 | Orchestrator: Pipeline | Claude | Agent协调可用 |
| 10 | Agent 7: 监测方案 | Claude | 完整推荐系统 |
| 11 | 端到端测试 | 全员 | 首个完整案例 |

**Milestone 3（Week 11结束）**：
- ✅ 7个Agent全部完成
- ✅ Orchestrator协调正常
- ✅ 端到端流程可运行

---

### Week 12-14: Phase 4 - 集成测试与优化

**目标：系统稳定可用**

| Week | 任务 | 负责人 | 里程碑 |
|------|------|--------|--------|
| 12 | 真实数据测试 | 全员 | 测试100个案例 |
| 12 | Bug修复 | 全员 | 主要问题解决 |
| 13 | 性能优化 | 您+研究生 | 推理<5秒 |
| 13 | FastAPI部署 | 研究生 | API可用 |
| 14 | 最终测试 | 全员 | 系统交付 |
| 14 | 文档编写 | 全员 | 技术文档完成 |

**Final Milestone（Week 14结束）**：
- ✅ 系统在真实数据上表现良好
- ✅ API部署完成
- ✅ 技术文档齐全
- ✅ 可以开始临床试验准备

---

## 六、关键风险与应对

### 风险1: Temporal-GNN效果不佳
**概率：中等**
- **影响**：Agent 3推理质量下降
- **应对**：
  1. 降级方案：使用纯LSTM替代（性能降10%但可用）
  2. 简化GNN架构（减少层数）
  3. 增加训练数据（从24k文章中提取更多案例）

### 风险2: 因果推断数据不足
**概率：中等**
- **影响**：ATE估计置信区间过宽
- **应对**：
  1. 使用Meta学习（从多个相似场景迁移）
  2. 贝叶斯先验（引入专家知识）
  3. 降级为关联分析（不做因果声明）

### 风险3: 研究生经验不足
**概率：低-中等**
- **影响**：Agent 1/5进度延迟2-3周
- **应对**：
  1. Claude提供更详细的代码示例
  2. 您提供额外指导（每周1-2小时）
  3. 简化部分功能（如减少特征工程复杂度）

### 风险4: 图谱数据质量问题
**概率：低**
- **影响**：Agent 2检索精度下降
- **应对**：
  1. 数据清洗和去重（您擅长）
  2. 引入置信度过滤
  3. 人工标注关键节点（100-200个核心概念）

---

## 七、成功标准

### 技术指标
- **预测准确率**：灌注质量评分MAE < 0.15（0-1量表）
- **风险预测**：AUC > 0.75（4类并发症）
- **推理速度**：端到端推理 < 5秒
- **可解释性**：每个预测都有证据链（至少3条支持证据）

### 系统指标
- **稳定性**：连续运行100个案例无崩溃
- **可扩展性**：支持批量处理（10个案例并行）
- **API可用性**：99% uptime

### 研究指标
- **新颖性**：Temporal-GNN + 因果推断 + 符号推理融合（学术创新）
- **实用性**：临床医生可理解和使用
- **发表潜力**：MICCAI/AAAI级别论文

---

## 八、总结

### 这个配置为什么可行？

1. **技能互补**
   - 您：KG/RAG专家 → 覆盖Agent 2（最核心的检索层）
   - Claude：算法专家 → 覆盖所有复杂推理（Agent 3/4/6/7）
   - 研究生：NLP/时序专家 → 覆盖数据处理（Agent 1/5）

2. **工作量合理**
   - 您：1.5人月（25%项目时间）
   - Claude：深度参与（实时支持）
   - 研究生：3.5人月（全职）
   - **总计**：5人月 vs 6-8.5人月需求（通过并行工作和Claude效率补齐）

3. **技术路线清晰**
   - Phase 1-3顺序推进
   - 每个阶段都有降级方案
   - Claude提供完整算法实现（不只是指导）

### 最终答案

**您+我+1个研究生 = 够！**

前提：
1. ✅ 研究生有NLP+时序项目经验
2. ✅ 您承担Agent 2+部分Agent 3（约25%时间）
3. ✅ 我提供所有核心算法的完整实现
4. ✅ 接受3.5-4个月工期

---

完整架构文档已保存至：
- `/home/user/SHU/docs/FINAL_INTEGRATED_ARCHITECTURE.md`

需要我详细展开某个具体模块的代码吗？比如：
- Temporal-GNN的完整PyTorch实现
- 因果推断的DoWhy代码
- Prolog规则库
- Orchestrator的完整流程
