# Multi-Agent Neuro-Symbolic System for Cardiac Perfusion Prediction
# 心脏灌注结果预测与介入决策系统

## 系统定位变更

**原架构**：心脏移植决策支持 → **新架构**：心脏灌注结果预测 + 实时介入措施推荐

**核心理念**：每个Layer是独立的智能体（Agent），协同完成灌注预测任务

---

## 一、系统输入输出规格（灌注场景）

### 输入数据（三类异构数据）

**1. 拟定灌注策略**
```json
{
  "perfusion_strategy": {
    "method": "HTK solution | Del Nido | Blood cardioplegia",
    "temperature": 4,  // °C
    "pressure": 60,    // mmHg
    "flow_rate": 1.2,  // L/min
    "duration": 240,   // minutes
    "additives": ["adenosine", "insulin"],
    "delivery_mode": "antegrade | retrograde | combined"
  }
}
```

**2. 异构心脏数据**

**2.1 自由文本描述**
```json
{
  "cardiac_description": {
    "visual_inspection": "Heart appears mildly hypertrophied with no visible scarring. Coronary arteries patent. Left ventricle shows good contractility.",
    "palpation_notes": "Firm consistency, no areas of induration. Valves appear competent.",
    "procurement_notes": "Cross-clamp time 32 minutes. Immediate cold perfusion initiated.",
    "surgeon_comments": "Excellent quality donor heart. Minimal ischemic time."
  }
}
```

**2.2 结构化血气数据**
```json
{
  "blood_gas_data": {
    "pre_perfusion": {
      "lactate": 2.8,      // mmol/L
      "pH": 7.32,
      "pO2": 280,          // mmHg
      "pCO2": 45,          // mmHg
      "K+": 4.2,           // mmol/L
      "glucose": 120       // mg/dL
    },
    "during_perfusion": [
      {"time": 60, "lactate": 1.8, "pH": 7.38, "pO2": 320, "pCO2": 42},
      {"time": 120, "lactate": 1.2, "pH": 7.40, "pO2": 340, "pCO2": 40},
      {"time": 180, "lactate": 0.9, "pH": 7.42, "pO2": 350, "pCO2": 38}
    ],
    "post_perfusion": {
      "lactate": 0.6,
      "pH": 7.44,
      "pO2": 360,
      "pCO2": 36
    }
  }
}
```

**3. 患者病历**
```json
{
  "recipient_medical_record": {
    "demographics": {"age": 55, "gender": "male", "weight": 78, "height": 175},
    "diagnosis": "dilated cardiomyopathy",
    "comorbidities": ["diabetes", "hypertension", "chronic kidney disease stage 3"],
    "lab_results": {
      "creatinine": 1.8,
      "BNP": 2400,
      "troponin": 0.15,
      "albumin": 3.5
    },
    "hemodynamics": {
      "LVEF": 15,
      "PVR": 3.2,
      "cardiac_output": 3.8,
      "PCWP": 28
    },
    "previous_interventions": ["LVAD", "ICD implantation"],
    "medications": ["furosemide", "carvedilol", "lisinopril", "warfarin"]
  }
}
```

---

### 输出格式（灌注预测结果）

```json
{
  "perfusion_outcome_prediction": {
    "overall_score": 0.78,  // 0-1, 灌注质量评分
    "risk_assessment": {
      "ischemia_reperfusion_injury": 0.23,  // 预测概率
      "endothelial_dysfunction": 0.18,
      "metabolic_recovery_failure": 0.12,
      "primary_graft_dysfunction": 0.15
    },
    "predicted_metrics": {
      "post_reperfusion_lactate": 1.2,  // mmol/L (预测值)
      "cardiac_output_24h": 4.5,        // L/min
      "troponin_peak": 8.2,             // ng/mL
      "time_to_hemodynamic_stability": 6  // hours
    }
  },

  "strategy_evaluation": {
    "current_strategy_adequacy": "suboptimal",  // optimal | adequate | suboptimal
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
      "rationale": "Evidence from 8 studies (OR=1.87, p=0.004): Higher pressure improves coronary flow in hypertrophied hearts",
      "expected_benefit": "Reduce ischemia-reperfusion injury risk from 23% to 14%",
      "timing": "immediate",
      "priority": "high",
      "evidence_level": "A"
    },
    {
      "intervention": "Add glucose-insulin-potassium (GIK) to perfusate",
      "rationale": "Meta-analysis (n=1,243): GIK improves metabolic recovery (p<0.001)",
      "expected_benefit": "Accelerate lactate clearance by 35%",
      "timing": "next perfusion cycle",
      "priority": "moderate",
      "evidence_level": "B"
    },
    {
      "intervention": "Monitor troponin at 2h intervals",
      "rationale": "Early detection of myocardial injury allows timely intervention",
      "expected_benefit": "Enable early detection of PGD",
      "timing": "post-transplant",
      "priority": "moderate",
      "evidence_level": "C"
    }
  ],

  "alternative_strategies": [
    {
      "strategy": "Switch to Del Nido cardioplegia",
      "predicted_improvement": "+12% in perfusion quality score",
      "trade_offs": {
        "pros": ["Single-dose delivery", "Better membrane stabilization"],
        "cons": ["Higher cost", "Less familiar to team"]
      },
      "evidence_support": 0.72
    }
  ],

  "real_time_monitoring_plan": {
    "critical_parameters": ["lactate", "pH", "coronary_flow"],
    "alert_thresholds": {
      "lactate_increase": ">0.5 mmol/L per hour",
      "pH_drop": "<7.35",
      "flow_decrease": "<0.8 L/min"
    },
    "intervention_triggers": [
      {
        "condition": "lactate > 2.0 at 2h",
        "action": "Increase flow rate by 20%"
      },
      {
        "condition": "pH < 7.30",
        "action": "Add bicarbonate buffer"
      }
    ]
  },

  "confidence_and_uncertainty": {
    "prediction_confidence": 0.82,
    "uncertainty_sources": [
      "Limited data on combined LVAD+CKD recipients (small sample size)",
      "Donor heart hypertrophy degree not quantified"
    ],
    "sensitivity_analysis": {
      "most_influential_factors": [
        "perfusion_pressure (+0.35 correlation)",
        "pre_perfusion_lactate (+0.28 correlation)",
        "recipient_PVR (+0.22 correlation)"
      ]
    }
  }
}
```

---

## 二、Multi-Agent架构（7个独立Agent）

### 总览图

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator Agent                   │
│          (协调各Agent，管理数据流和决策流程)              │
└────────────┬───────────────────────────────────────────┘
             ↓
┌────────────────────────────────────────────────────────────────┐
│                      Multi-Agent System                         │
│                                                                 │
│  Agent 1        Agent 2        Agent 3         Agent 4         │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────┐ │
│  │ Query    │→ │Retrieval │→ │Reasoning  │→ │Evidence      │ │
│  │Understanding│ │  Agent  │  │  Engine   │  │Synthesis     │ │
│  └──────────┘  └──────────┘  └───────────┘  └──────────────┘ │
│                                     ↓                           │
│  Agent 5        Agent 6        Agent 7                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐                │
│  │Prediction│← │Strategy  │← │Intervention  │                │
│  │  Agent   │  │Evaluation│  │Recommendation│                │
│  └──────────┘  └──────────┘  └──────────────┘                │
└────────────────────────────────────────────────────────────────┘
```

---

### Agent 1: Input Understanding Agent（输入理解智能体）

**职责**：解析异构输入数据，提取关键特征

**输入**：
- 拟定灌注策略（结构化）
- 心脏描述（自由文本）
- 血气数据（时间序列）
- 患者病历（混合）

**处理流程**：
```python
class InputUnderstandingAgent:
    """
    任务：异构数据标准化和特征提取
    """
    def __init__(self):
        self.text_encoder = ClinicalBERT()          # 文本编码器
        self.time_series_encoder = LSTM()           # 时序编码器
        self.feature_extractor = FeatureExtractor() # 特征工程
        self.entity_recognizer = MedicalNER()       # 实体识别

    def process(self, raw_input: Dict) -> StandardizedInput:
        # 1. 文本理解
        cardiac_features = self.text_encoder.encode(
            raw_input['cardiac_description']['visual_inspection']
        )
        # 提取：hypertrophy, scarring, contractility, valve_status

        # 2. 时序数据处理
        blood_gas_trajectory = self.time_series_encoder.encode(
            raw_input['blood_gas_data']['during_perfusion']
        )
        # 编码为固定维度向量，捕捉趋势

        # 3. 策略参数标准化
        strategy_vector = self.feature_extractor.extract(
            raw_input['perfusion_strategy']
        )
        # 归一化：temperature, pressure, flow_rate等

        # 4. 患者特征聚合
        patient_risk_profile = self._compute_risk_profile(
            raw_input['recipient_medical_record']
        )
        # 计算复合风险评分

        return StandardizedInput(
            cardiac_features=cardiac_features,        # 768-dim vector
            blood_gas_embedding=blood_gas_trajectory, # 128-dim vector
            strategy_params=strategy_vector,          # 20-dim vector
            patient_profile=patient_risk_profile,     # 50-dim vector
            raw_text_entities=self.entity_recognizer.extract_all(...)
        )
```

**关键能力**：
- 文本→向量：Clinical BERT编码心脏描述
- 时序→特征：LSTM提取血气数据趋势
- 实体识别：NER提取关键医学概念（药物、疾病、指标）

**输出**：
```python
StandardizedInput(
    cardiac_state={
        "hypertrophy_level": 0.6,  # 0-1 scale
        "contractility_score": 0.8,
        "valve_competence": "good",
        "visible_damage": False
    },
    perfusion_params={
        "pressure_normalized": 0.5,  # 相对于推荐范围
        "flow_adequacy": 0.7,
        "temperature_optimal": True
    },
    metabolic_trajectory={
        "lactate_clearance_rate": -0.02,  # mmol/L per min
        "pH_stability": 0.95,
        "oxygenation_trend": "improving"
    },
    patient_risk_factors=["diabetes", "hypertrophy", "CKD"],
    key_entities=["HTK solution", "adenosine", "LVAD", "troponin"]
)
```

---

### Agent 2: Knowledge Retrieval Agent（知识检索智能体）

**职责**：从知识图谱和文献库检索相关灌注知识

**核心变化**：
- **检索目标**：灌注策略、血气指标、介入措施（不再是移植决策）
- **图谱子集**：灌注相关的子图（perfusion_method, metabolic_marker, intervention）

**检索策略**：
```python
class KnowledgeRetrievalAgent:
    """
    任务：检索灌注相关知识
    """
    def __init__(self):
        self.graph_retriever = Neo4jRetriever()
        self.vector_retriever = ChromaRetriever()
        self.perfusion_ontology = PerfusionOntology()  # 灌注本体

    def retrieve(self, standardized_input: StandardizedInput) -> RetrievalResult:
        # 1. 基于策略参数的图检索
        subgraph = self.graph_retriever.query(f"""
            MATCH path = (strategy:PerfusionStrategy)-[r*1..3]-(outcome:Outcome)
            WHERE strategy.method = '{standardized_input.strategy_params.method}'
            AND strategy.pressure BETWEEN 50 AND 80
            RETURN path
        """)
        # 返回：相似策略的历史结果

        # 2. 基于血气趋势的向量检索
        similar_cases = self.vector_retriever.search(
            query_embedding=standardized_input.blood_gas_embedding,
            filter={"lactate_clearance": {"$lt": 0}},  # 找lactate清除不佳的案例
            top_k=20
        )
        # 返回：相似血气模式的文献

        # 3. 基于患者特征的风险检索
        risk_evidence = self.graph_retriever.query(f"""
            MATCH (patient:PatientProfile)-[:HAS_RISK]->(risk:RiskFactor)
                  -[:INCREASES_RISK]->(complication:Complication)
            WHERE patient.has_diabetes = true
            AND patient.has_CKD = true
            RETURN risk, complication, relationship_properties
        """)
        # 返回：糖尿病+CKD患者的灌注风险

        # 4. 介入措施检索
        interventions = self.graph_retriever.query(f"""
            MATCH (issue:PerfusionIssue)-[:TREATED_BY]->(intervention:Intervention)
            WHERE issue.type IN ['slow_lactate_clearance', 'low_pH']
            RETURN intervention, evidence_level, success_rate
        """)
        # 返回：针对当前问题的介入措施

        return RetrievalResult(
            similar_strategies=subgraph.nodes('PerfusionStrategy'),  # 15个相似策略
            outcome_data=subgraph.nodes('Outcome'),                  # 对应结果
            relevant_literature=similar_cases,                       # 20篇文献
            risk_pathways=risk_evidence,                             # 5条风险路径
            intervention_options=interventions                        # 12种介入措施
        )
```

**知识图谱Schema（灌注专用）**：
```cypher
// 节点类型
(:PerfusionStrategy {method, temperature, pressure, flow_rate})
(:BloodGasMarker {name, normal_range, clinical_significance})
(:Outcome {quality_score, complications, recovery_time})
(:Intervention {name, mechanism, evidence_level})
(:RiskFactor {name, prevalence, odds_ratio})

// 关系类型
(:PerfusionStrategy)-[:RESULTS_IN {probability, confidence}]->(:Outcome)
(:BloodGasMarker)-[:INDICATES {threshold}]->(:PerfusionQuality)
(:Intervention)-[:IMPROVES {effect_size}]->(:Outcome)
(:RiskFactor)-[:PREDISPOSES_TO]->(:Complication)
```

---

### Agent 3: Neuro-Symbolic Reasoning Engine（神经符号推理引擎）

**职责**：基于检索结果进行因果推理和预测

**推理任务**：
1. **因果推理**：灌注参数 → 血气变化 → 临床结局
2. **反事实推理**：如果改变策略X，结果Y会如何变化
3. **路径发现**：从当前状态到最优结局的推理路径

**双重推理机制**：
```python
class NeuroSymbolicReasoningEngine:
    """
    任务：因果推理和结局预测
    """
    def __init__(self):
        self.gnn_reasoner = TemporalGNN()        # 时序GNN（处理血气动态）
        self.causal_model = CausalModel()        # 因果模型（SCM）
        self.logic_engine = PrologEngine()       # 符号推理
        self.uncertainty_quantifier = BayesianNN()  # 不确定性量化

    def reason(self, retrieval: RetrievalResult, input_data: StandardizedInput):
        # 1. 时序GNN推理（处理动态血气数据）
        blood_gas_dynamics = self.gnn_reasoner.forward(
            x=retrieval.blood_gas_nodes,
            edge_index=retrieval.temporal_edges,  # t -> t+1 edges
            edge_attr=retrieval.temporal_relations
        )
        # 输出：预测未来血气趋势

        # 2. 因果推理（估计干预效应）
        causal_effects = self.causal_model.estimate_ate(
            treatment='increase_pressure',
            outcome='lactate_clearance_rate',
            confounders=['cardiac_hypertrophy', 'ischemic_time'],
            data=retrieval.similar_cases
        )
        # 输出：ATE = -0.3 mmol/L/h (增加压力可降低lactate)

        # 3. 符号推理（规则引擎）
        logic_inferences = self.logic_engine.query(f"""
            % 规则库
            inadequate_perfusion(Strategy) :-
                pressure(Strategy, P), P < 65,
                cardiac_hypertrophy(Heart, severe).

            risk_high(Patient, ischemia_reperfusion) :-
                diabetes(Patient, yes),
                lactate(PrePerfusion, L), L > 2.5.

            recommend_intervention(increase_pressure) :-
                inadequate_perfusion(_),
                risk_high(_, ischemia_reperfusion).

            % 查询
            ?- recommend_intervention(X).
        """)
        # 输出：X = increase_pressure (逻辑推导)

        # 4. 不确定性量化
        uncertainty = self.uncertainty_quantifier.predict(
            x=input_data.all_features,
            return_std=True
        )
        # 输出：预测值 ± 标准差

        # 5. 生成推理路径
        reasoning_path = self._construct_path(
            neural_prediction=blood_gas_dynamics,
            causal_effects=causal_effects,
            logic_proofs=logic_inferences
        )

        return ReasoningResult(
            predicted_outcome={
                "quality_score": 0.78,
                "ischemia_reperfusion_injury_risk": 0.23
            },
            causal_chain=[
                "low_pressure(60mmHg) → inadequate_coronary_flow → slow_lactate_clearance",
                "diabetes + high_lactate → increased_oxidative_stress → endothelial_dysfunction"
            ],
            intervention_recommendations=[
                {"action": "increase_pressure", "expected_benefit": "+0.15 quality_score"}
            ],
            confidence_interval=(0.68, 0.88),  # 95% CI for quality_score
            key_evidence_nodes=retrieval.top_k_most_relevant(10)
        )
```

**时序GNN架构（处理血气动态）**：
```python
class TemporalGNN(nn.Module):
    """
    处理时序血气数据的图神经网络
    """
    def __init__(self):
        self.temporal_conv = TemporalConvLayer()  # 时序卷积
        self.graph_conv = GATConv()               # 图卷积
        self.predictor = nn.Linear(128, 64)

    def forward(self, blood_gas_sequence, graph_structure):
        # blood_gas_sequence: [T, N, F]  T=时间步, N=节点, F=特征
        # graph_structure: 知识图谱的邻接矩阵

        # 1. 时序编码（捕捉血气动态）
        temporal_features = self.temporal_conv(blood_gas_sequence)
        # 输出：每个时间点的表示

        # 2. 图传播（结合知识图谱）
        for t in range(T):
            temporal_features[t] = self.graph_conv(
                x=temporal_features[t],
                edge_index=graph_structure
            )
        # 输出：融合知识图谱的时序表示

        # 3. 预测未来状态
        future_prediction = self.predictor(temporal_features[-1])
        return future_prediction
```

---

### Agent 4: Evidence Synthesis Agent（证据综合智能体）

**职责**：评估推理结果的证据强度，生成证据链

**针对灌注的证据评估**：
```python
class EvidenceSynthesisAgent:
    """
    任务：评估灌注预测的证据质量
    """
    def __init__(self):
        self.grade_evaluator = GRADEEvaluator()
        self.meta_analyzer = MetaAnalyzer()
        self.heterogeneity_checker = HeterogeneityChecker()

    def synthesize(self, reasoning: ReasoningResult, retrieval: RetrievalResult):
        # 1. 为每个预测评估证据
        evidence_for_pressure_increase = self.grade_evaluator.assess(
            claim="Increasing pressure from 60 to 75 mmHg improves outcomes",
            supporting_studies=[
                # 从retrieval中获取相关研究
                study for study in retrieval.relevant_literature
                if 'pressure' in study.keywords and 'hypertrophy' in study.context
            ]
        )
        # 输出：GRADE = Moderate (2 RCTs, 3 cohort studies)

        # 2. Meta分析（如有多个研究）
        if len(evidence_for_pressure_increase) >= 3:
            pooled_effect = self.meta_analyzer.pool(
                studies=evidence_for_pressure_increase,
                outcome='lactate_clearance_improvement'
            )
            # 输出：Pooled effect = 0.35 mmol/L/h (95% CI: 0.18-0.52), I²=15%

        # 3. 异质性检查
        heterogeneity = self.heterogeneity_checker.assess(
            studies=evidence_for_pressure_increase
        )
        # 输出：I² = 15% (低异质性，结果一致)

        # 4. 生成证据链
        evidence_chain = [
            {
                "claim": "Low perfusion pressure inadequate for hypertrophied hearts",
                "support_level": "strong",
                "evidence_sources": [
                    {"pmid": "33421385", "type": "RCT", "n": 234, "effect_size": 0.42},
                    {"pmid": "33253110", "type": "cohort", "n": 567, "effect_size": 0.38}
                ],
                "grade": "B",
                "confidence": 0.85
            },
            {
                "claim": "Diabetes increases oxidative stress during perfusion",
                "support_level": "moderate",
                "evidence_sources": [
                    {"pmid": "33290551", "type": "case_control", "OR": 2.1, "p": 0.003}
                ],
                "grade": "C",
                "confidence": 0.72
            }
        ]

        return EvidenceSynthesis(
            evidence_chain=evidence_chain,
            overall_quality="Moderate to High",
            heterogeneity_assessment="Low (I²<25% for most comparisons)",
            recommendation_strength="Strong for pressure adjustment, Moderate for GIK addition"
        )
```

---

### Agent 5: Perfusion Outcome Prediction Agent（灌注结局预测智能体）

**职责**：基于推理和证据，预测灌注结局

**预测模型**：
```python
class PerfusionOutcomePredictionAgent:
    """
    任务：预测灌注质量和临床结局
    """
    def __init__(self):
        self.outcome_predictor = EnsembleModel([
            GradientBoosting(),  # 基于表格特征
            LSTM(),              # 基于时序特征
            GNN()                # 基于图特征
        ])
        self.risk_scorer = RiskScoreCalculator()
        self.trajectory_predictor = SequencePredictor()

    def predict(self, reasoning: ReasoningResult, evidence: EvidenceSynthesis):
        # 1. 综合预测（集成多模型）
        outcome_scores = self.outcome_predictor.predict({
            'perfusion_params': reasoning.input_features['strategy'],
            'blood_gas_sequence': reasoning.input_features['blood_gas'],
            'patient_features': reasoning.input_features['patient'],
            'graph_embedding': reasoning.graph_representation
        })

        # 2. 风险评分
        risk_scores = self.risk_scorer.calculate([
            'ischemia_reperfusion_injury',
            'endothelial_dysfunction',
            'metabolic_recovery_failure',
            'primary_graft_dysfunction'
        ], features=outcome_scores)

        # 3. 预测未来指标轨迹
        predicted_trajectories = self.trajectory_predictor.forecast(
            current_state=reasoning.input_features['blood_gas'][-1],
            horizon=24  # 预测未来24小时
        )

        # 4. 计算置信区间（Bayesian ensemble）
        confidence_intervals = self._compute_ci(outcome_scores)

        return PredictionResult(
            overall_quality_score=outcome_scores['quality'],  # 0.78
            risk_probabilities={
                'ischemia_reperfusion_injury': 0.23,
                'endothelial_dysfunction': 0.18,
                'metabolic_recovery_failure': 0.12,
                'primary_graft_dysfunction': 0.15
            },
            predicted_metrics={
                'post_reperfusion_lactate': (1.2, 0.3),  # mean, std
                'cardiac_output_24h': (4.5, 0.8),
                'troponin_peak': (8.2, 2.1),
                'time_to_stability': (6, 2)  # hours
            },
            confidence=0.82,
            prediction_interval=(0.68, 0.88)  # for quality_score
        )
```

**集成预测架构**：
```python
class EnsembleModel:
    """
    三种模型的集成
    """
    def __init__(self, models):
        self.gbm = models[0]    # 处理静态特征
        self.lstm = models[1]   # 处理时序特征
        self.gnn = models[2]    # 处理图特征
        self.meta_learner = nn.Linear(3, 1)  # 元学习器

    def predict(self, features):
        # 1. 各模型独立预测
        pred_gbm = self.gbm.predict(features['perfusion_params'] + features['patient_features'])
        pred_lstm = self.lstm.predict(features['blood_gas_sequence'])
        pred_gnn = self.gnn.predict(features['graph_embedding'])

        # 2. 元学习器融合
        ensemble_pred = self.meta_learner(
            torch.cat([pred_gbm, pred_lstm, pred_gnn], dim=-1)
        )

        return {
            'quality': ensemble_pred.item(),
            'individual_preds': [pred_gbm, pred_lstm, pred_gnn],
            'agreement_score': self._compute_agreement([pred_gbm, pred_lstm, pred_gnn])
        }
```

---

### Agent 6: Strategy Evaluation Agent（策略评估智能体）

**职责**：评估拟定灌注策略的优劣，识别问题

**评估流程**：
```python
class StrategyEvaluationAgent:
    """
    任务：评估当前策略，识别改进点
    """
    def __init__(self):
        self.guideline_checker = GuidelineChecker()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.sensitivity_analyzer = SensitivityAnalyzer()

    def evaluate(self, strategy: PerfusionStrategy, prediction: PredictionResult):
        # 1. 与指南对比
        guideline_compliance = self.guideline_checker.check(strategy)
        # 输出：符合3/5条关键推荐，2条偏离

        # 2. 与最优策略对比
        optimal_strategy = self._retrieve_optimal_strategy(
            patient_profile=prediction.patient_features
        )
        deviation_analysis = self.comparative_analyzer.compare(
            current=strategy,
            optimal=optimal_strategy
        )
        # 输出：pressure偏低10 mmHg，flow偏低15%

        # 3. 敏感性分析（哪些参数影响最大）
        sensitivity = self.sensitivity_analyzer.analyze(
            strategy=strategy,
            outcome_predictor=prediction.model
        )
        # 输出：pressure影响最大(Δ=0.35)，temperature影响小(Δ=0.05)

        # 4. 识别具体问题
        identified_issues = []
        if strategy.pressure < optimal_strategy.pressure - 5:
            identified_issues.append({
                'issue': f'Perfusion pressure too low ({strategy.pressure} mmHg)',
                'impact': 'Inadequate coronary perfusion in hypertrophied heart',
                'severity': 'moderate',
                'evidence': guideline_compliance.pressure_recommendation
            })

        if prediction.risk_probabilities['metabolic_recovery_failure'] > 0.15:
            identified_issues.append({
                'issue': 'Lactate clearance predicted to be slow',
                'impact': 'Delayed metabolic recovery',
                'severity': 'mild',
                'evidence': prediction.predicted_metrics['post_reperfusion_lactate']
            })

        # 5. 综合评分
        adequacy_score = self._compute_adequacy(
            guideline_compliance, deviation_analysis, prediction
        )
        # 0-1: optimal(>0.9), adequate(0.7-0.9), suboptimal(<0.7)

        return StrategyEvaluation(
            adequacy_level="suboptimal" if adequacy_score < 0.7 else "adequate",
            adequacy_score=adequacy_score,
            identified_issues=identified_issues,
            deviation_from_optimal=deviation_analysis,
            sensitivity_factors=sensitivity.top_k(5)
        )
```

---

### Agent 7: Intervention Recommendation Agent（介入推荐智能体）

**职责**：基于问题和预测，推荐具体介入措施

**推荐引擎**：
```python
class InterventionRecommendationAgent:
    """
    任务：生成个性化介入建议
    """
    def __init__(self):
        self.intervention_db = InterventionDatabase()
        self.effect_estimator = CausalEffectEstimator()
        self.prioritizer = InterventionPrioritizer()
        self.contraindication_checker = ContraindicationChecker()

    def recommend(self,
                  evaluation: StrategyEvaluation,
                  prediction: PredictionResult,
                  patient: PatientProfile):

        recommendations = []

        # 1. 针对每个识别的问题，检索候选介入措施
        for issue in evaluation.identified_issues:
            candidates = self.intervention_db.query(
                problem_type=issue['issue'],
                severity=issue['severity']
            )
            # 返回：10-15个候选介入措施

            # 2. 估计每个介入的因果效应
            for candidate in candidates:
                effect = self.effect_estimator.estimate(
                    intervention=candidate,
                    current_state=prediction,
                    patient=patient
                )
                # 输出：预期改善量（如：risk降低9%）

                # 3. 检查禁忌症
                contraindications = self.contraindication_checker.check(
                    intervention=candidate,
                    patient=patient
                )

                if not contraindications:
                    recommendations.append({
                        'intervention': candidate.name,
                        'rationale': self._generate_rationale(candidate, issue),
                        'expected_benefit': effect.expected_improvement,
                        'evidence_level': candidate.evidence_level,
                        'timing': self._determine_timing(issue.severity),
                        'priority': None  # 待计算
                    })

        # 4. 优先级排序
        recommendations = self.prioritizer.rank(
            recommendations,
            criteria=['expected_benefit', 'evidence_level', 'feasibility', 'urgency']
        )

        # 5. 生成监测计划
        monitoring_plan = self._generate_monitoring_plan(
            recommendations, prediction
        )

        return RecommendationResult(
            interventions=recommendations[:5],  # Top-5
            monitoring_plan=monitoring_plan,
            alternative_strategies=self._generate_alternatives(evaluation),
            explanation=self._generate_explanation(recommendations)
        )

    def _generate_rationale(self, intervention, issue):
        # 从知识图谱检索证据
        evidence = self.intervention_db.get_evidence(intervention.id)

        rationale = f"""
        {intervention.name} addresses {issue['issue']}.
        Evidence: {evidence.study_design} (n={evidence.sample_size})
        Effect: {evidence.effect_description} (p={evidence.p_value})
        Mechanism: {intervention.mechanism_of_action}
        """
        return rationale.strip()
```

**介入措施示例**：
```python
# 介入措施库结构
interventions_db = {
    "increase_perfusion_pressure": {
        "id": "INT001",
        "name": "Increase perfusion pressure to 75-80 mmHg",
        "category": "parameter_adjustment",
        "mechanism": "Improves coronary flow, especially in hypertrophied hearts",
        "evidence": [
            {"study": "PMID:33421385", "type": "RCT", "n": 234, "OR": 1.87, "p": 0.004},
            {"study": "PMID:33253110", "type": "cohort", "n": 567, "effect": "+0.35 quality"}
        ],
        "contraindications": ["severe_aortic_regurgitation"],
        "timing": "immediate",
        "feasibility": "high"
    },
    "add_GIK": {
        "id": "INT002",
        "name": "Add glucose-insulin-potassium (GIK) to perfusate",
        "category": "additive",
        "mechanism": "Enhances metabolic recovery, reduces oxidative stress",
        "evidence": [
            {"study": "Meta-analysis", "n": 1243, "effect": "+35% lactate clearance", "p": 0.001}
        ],
        "contraindications": ["severe_hyperkalemia"],
        "timing": "next_cycle",
        "feasibility": "moderate"
    }
}
```

---

## 三、Orchestrator Agent（协调器智能体）

**职责**：管理7个Agent的交互和数据流

```python
class OrchestratorAgent:
    """
    Meta-Agent：协调所有子Agent
    """
    def __init__(self):
        self.agent_1 = InputUnderstandingAgent()
        self.agent_2 = KnowledgeRetrievalAgent()
        self.agent_3 = NeuroSymbolicReasoningEngine()
        self.agent_4 = EvidenceSynthesisAgent()
        self.agent_5 = PerfusionOutcomePredictionAgent()
        self.agent_6 = StrategyEvaluationAgent()
        self.agent_7 = InterventionRecommendationAgent()

        self.message_bus = MessageBus()  # Agent间通信
        self.state_manager = StateManager()  # 管理中间状态

    def execute_pipeline(self, raw_input: Dict):
        # 阶段1：输入理解
        standardized_input = self.agent_1.process(raw_input)
        self.state_manager.save('standardized_input', standardized_input)

        # 阶段2：知识检索
        retrieval_result = self.agent_2.retrieve(standardized_input)
        self.state_manager.save('retrieval', retrieval_result)

        # 阶段3：神经符号推理
        reasoning_result = self.agent_3.reason(retrieval_result, standardized_input)
        self.state_manager.save('reasoning', reasoning_result)

        # 阶段4：证据综合
        evidence_synthesis = self.agent_4.synthesize(reasoning_result, retrieval_result)
        self.state_manager.save('evidence', evidence_synthesis)

        # 阶段5：结局预测
        prediction = self.agent_5.predict(reasoning_result, evidence_synthesis)
        self.state_manager.save('prediction', prediction)

        # 阶段6：策略评估
        evaluation = self.agent_6.evaluate(
            strategy=standardized_input.perfusion_strategy,
            prediction=prediction
        )
        self.state_manager.save('evaluation', evaluation)

        # 阶段7：介入推荐
        recommendations = self.agent_7.recommend(
            evaluation=evaluation,
            prediction=prediction,
            patient=standardized_input.patient_profile
        )

        # 聚合最终输出
        final_output = self._assemble_output(
            prediction, evaluation, recommendations, evidence_synthesis
        )

        return final_output

    def _assemble_output(self, prediction, evaluation, recommendations, evidence):
        return {
            'perfusion_outcome_prediction': prediction.to_dict(),
            'strategy_evaluation': evaluation.to_dict(),
            'recommended_interventions': recommendations.interventions,
            'monitoring_plan': recommendations.monitoring_plan,
            'evidence_summary': evidence.summary(),
            'confidence': prediction.confidence,
            'uncertainty': prediction.prediction_interval
        }
```

---

## 四、完整数据流（灌注场景）

```
输入：灌注策略 + 心脏数据 + 血气数据 + 患者病历
    ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 1: Input Understanding                            │
│ • 文本→向量 (ClinicalBERT)                              │
│ • 时序→特征 (LSTM)                                      │
│ • 标准化：所有数据→统一表示                              │
│ 输出：StandardizedInput                                 │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 2: Knowledge Retrieval                            │
│ • 图检索：相似灌注策略的历史结果                          │
│ • 向量检索：相似血气模式的文献                            │
│ • 风险检索：患者特征对应的风险路径                        │
│ 输出：SubGraph + Documents (top-20)                     │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 3: Neuro-Symbolic Reasoning                       │
│ • 时序GNN：预测血气趋势                                  │
│ • 因果模型：估计介入效应                                 │
│ • 逻辑推理：规则引擎推导                                 │
│ 输出：ReasoningPath + CausalEffects                     │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 4: Evidence Synthesis                             │
│ • GRADE评分：证据质量分级                                │
│ • Meta分析：合并多研究效应                               │
│ • 异质性检查：I²统计                                     │
│ 输出：EvidenceChain + QualityScore                      │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 5: Perfusion Outcome Prediction                   │
│ • 集成预测：GBM + LSTM + GNN                            │
│ • 风险评分：4类并发症概率                                │
│ • 轨迹预测：未来24h指标                                  │
│ 输出：QualityScore(0.78) + RiskProbs + Trajectories    │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 6: Strategy Evaluation                            │
│ • 指南对比：符合度检查                                   │
│ • 偏差分析：vs最优策略                                   │
│ • 敏感性分析：关键参数识别                               │
│ 输出：Adequacy(suboptimal) + Issues(2个)                │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ Agent 7: Intervention Recommendation                    │
│ • 候选检索：针对identified issues                        │
│ • 效应估计：预期改善量                                   │
│ • 优先级排序：Top-5推荐                                  │
│ 输出：5条介入建议 + 监测方案                             │
└────────────────┬────────────────────────────────────────┘
                 ↓
输出：完整决策报告
    • 灌注质量预测 (0.78, CI: 0.68-0.88)
    • 风险评估 (4类并发症概率)
    • 策略评估 (suboptimal, 2个问题)
    • 介入推荐 (5条，优先级排序)
    • 监测方案 (关键参数+阈值)
```

---

## 五、技术栈（针对灌注场景）

| Agent | 核心技术 | 模型/框架 |
|-------|---------|----------|
| Agent 1 | 多模态编码 | ClinicalBERT, LSTM, FeatureExtractor |
| Agent 2 | 混合检索 | Neo4j (灌注KG), ChromaDB, Cypher |
| Agent 3 | 时序+因果推理 | Temporal-GNN, DoWhy (因果), Prolog |
| Agent 4 | 证据评估 | GRADE, R meta, I²统计 |
| Agent 5 | 集成预测 | GBM+LSTM+GNN ensemble, Bayesian NN |
| Agent 6 | 策略比对 | 指南库, Sensitivity analysis |
| Agent 7 | 介入推荐 | 介入措施库, 因果效应估计, 优先级算法 |

---

## 六、与原架构的关键差异

| 维度 | 原架构（移植决策） | 新架构（灌注预测） |
|-----|------------------|------------------|
| **预测目标** | 移植决策（是否移植） | 灌注质量 + 介入措施 |
| **输入数据** | 供受体特征 | 策略参数 + 血气时序 + 心脏描述 |
| **时序建模** | 无 | ✅ 时序GNN处理血气动态 |
| **因果推理** | 基础 | ✅ 深度因果模型（ATE估计） |
| **实时性** | 离线 | ✅ 实时监测 + 动态调整 |
| **Agent数量** | 单体系统 | 7个独立Agent |
| **反馈机制** | 无 | ✅ 监测→评估→调整闭环 |

---

## 七、实施路线图

### Phase 1: 核心预测能力（2-3周）
- Agent 1: 输入理解（多模态编码器）
- Agent 2: 基础检索（图+向量）
- Agent 5: 基础预测模型（单模型，非集成）

### Phase 2: 增强推理（3-4周）
- Agent 3: 时序GNN + 因果模型
- Agent 4: 证据综合
- Agent 6: 策略评估

### Phase 3: 完整系统（4-5周）
- Agent 7: 介入推荐
- Orchestrator: Agent协调
- 实时监测闭环

---

完整文档保存至：`/home/user/SHU/docs/MULTI_AGENT_PERFUSION_ARCHITECTURE.md`

需要我详细展开某个Agent的代码实现吗？比如时序GNN或因果推理模块？
