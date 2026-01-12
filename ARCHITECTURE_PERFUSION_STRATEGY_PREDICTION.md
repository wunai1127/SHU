# Architecture Analysis: Heart Perfusion Strategy Prediction

## Executive Summary

### Context
Machine perfusion generates massive amounts of data (blood gas, inflammatory markers, hemodynamic parameters), yet this data remains **underutilized** for rescuing marginal donors that could save more lives.

### Problem
Current clinical decisions rely on physicians' **"generic experience"**, ignoring the complex **nonlinear interactions** between:
- **Donor history** (text: anatomical descriptions, medical records)
- **Perfusion parameters** (time-series: blood gas, inflammatory markers, pressure/flow)

This leads to **latent ischemic injury being overlooked**, resulting in:
- Potentially viable hearts being discarded
- Suboptimal perfusion strategies being applied
- Delayed intervention when problems arise

### Solution
A **Neuro-Symbolic AI System** that:
1. **Predicts perfusion outcomes** before strategy execution
2. **Monitors real-time** perfusion data and predicts emerging issues
3. **Recommends interventions** based on knowledge graph + GNN + LLM reasoning

---

## 1. Current Progress Status

### ✅ Completed Components

| Component | Status | Details |
|-----------|--------|---------|
| **Triple Extraction** | ✅ Done | 15,068 articles processed |
| **Entity Types** | ✅ Done | treatment_regimen, complication, risk_factor, donor_characteristic, monitoring_indicator |
| **Relation Types** | ✅ Done | increases_risk, treats, mitigates, alleviates, precedes, part_of |
| **CSV Export** | ✅ Done | triple_nodes.csv, triple_edges.csv, text_nodes.csv, text_edges.csv |

### Triple Data Sample
```json
{
  "entities": [
    {"name": "prolonged ischemic times", "type": "risk_factor"},
    {"name": "ischemia-reperfusion injury (IRI)", "type": "complication"},
    {"name": "Lung-derived exosomes", "type": "treatment_regimen"}
  ],
  "relations": [
    {"head": "prolonged ischemic times", "relation": "increases_risk", "tail": "ischemia-reperfusion injury (IRI)"},
    {"head": "Lung-derived exosomes", "relation": "treats", "tail": "lung IRI"}
  ]
}
```

### 🔄 In Progress / Pending

| Component | Status | Priority |
|-----------|--------|----------|
| **Neo4j Import** | 🔄 Pending | High |
| **Temporal GNN Module** | ❌ Not Started | High |
| **Real-time Monitor** | ❌ Not Started | High |
| **Perfusion Schema Extension** | 🔄 Partial | Medium |

---

## 2. Architecture Overview

### Core Principle: Minimal Changes, Maximum Extension

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXISTING ARCHITECTURE                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Multi-Agent  │  │  Dual-Layer  │  │Neuro-Symbolic│          │
│  │  (LangGraph) │  │    Graph     │  │   Reasoning  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ✅                ✅                 ✅                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NEW EXTENSIONS (3 Modules)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Temporal    │  │  Perfusion   │  │  Real-time   │          │
│  │    GNN       │  │    Schema    │  │   Monitor    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         🆕                🆕                 🆕                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Extension 1: Temporal GNN Module

### Why Needed
- 移植预测是静态特征，灌注预测是**时序数据**（每5分钟采样）
- 需要捕捉血气指标的**变化趋势**和**异常模式**

### Architecture

```python
class TemporalPerfusionGNN(nn.Module):
    """
    Temporal Graph Neural Network for perfusion monitoring

    Combines:
    - LSTM: Captures temporal dependencies in blood gas / inflammatory markers
    - GraphSAGE: Captures structural relationships in knowledge graph
    """

    def __init__(self, config: TemporalGNNConfig):
        super().__init__()

        # Temporal encoder: process time-series features
        # 时序编码器：处理血气、炎症因子等时序特征
        self.temporal_encoder = nn.LSTM(
            input_size=config.temporal_features,  # [pH, PO2, PCO2, lactate, K+, IL-6, IL-8, ...]
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout
        )

        # Graph encoder: process heart features + strategy nodes
        # 图编码器：处理心脏特征节点 + 灌注策略节点
        self.graph_encoder = GraphSAGE(
            in_channels=config.node_features,
            hidden_channels=config.hidden_size,
            num_layers=3,
            out_channels=config.hidden_size
        )

        # Fusion layer: combine temporal + graph representations
        # 融合层：合并时序特征和图特征
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Prediction heads
        # 预测头：质量评分、风险预警、下一时刻预测
        self.quality_head = nn.Linear(config.hidden_size, 1)  # Quality score 0-100
        self.risk_head = nn.Linear(config.hidden_size, 4)     # Risk categories
        self.next_state_head = nn.Linear(config.hidden_size, config.temporal_features)

    def forward(
        self,
        temporal_features: torch.Tensor,  # [batch, time_steps, features]
        graph_data: Data                   # PyG graph with node features
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining temporal and graph information

        Args:
            temporal_features: Blood gas + inflammatory markers over time
                Shape: [batch_size, time_steps, num_features]
                Features: [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α,
                          pressure, flow_rate, temperature]

            graph_data: Knowledge subgraph containing:
                - Heart instance node (anatomical features)
                - Perfusion strategy node (parameters)
                - Related concept nodes (complications, risk factors)

        Returns:
            Dict containing:
                - quality_score: Predicted perfusion quality (0-100)
                - risk_level: Risk classification
                - next_state: Predicted next measurement values
                - need_intervention: Boolean flag
        """
        # 1. Encode temporal sequence
        # 编码时序数据，捕捉变化趋势
        lstm_out, (h_n, c_n) = self.temporal_encoder(temporal_features)
        h_temporal = h_n[-1]  # Last hidden state

        # 2. Encode graph structure
        # 编码图结构，捕捉知识关联
        h_graph = self.graph_encoder(graph_data.x, graph_data.edge_index)
        h_graph_pooled = global_mean_pool(h_graph, graph_data.batch)

        # 3. Fuse representations
        # 融合时序特征和图特征
        h_fused = self.fusion(torch.cat([h_temporal, h_graph_pooled], dim=-1))

        # 4. Generate predictions
        quality_score = torch.sigmoid(self.quality_head(h_fused)) * 100
        risk_logits = self.risk_head(h_fused)
        next_state = self.next_state_head(h_fused)

        return {
            'quality_score': quality_score,
            'risk_level': F.softmax(risk_logits, dim=-1),
            'next_state': next_state,
            'need_intervention': quality_score < 70  # Threshold-based flag
        }
```

### Feature Engineering

```python
def extract_temporal_features(blood_gas_timeline: Dict) -> np.ndarray:
    """
    Extract engineered features from time-series blood gas data
    从时序血气数据中提取工程特征

    Args:
        blood_gas_timeline: {
            'timestamps': [0, 5, 10, 15, ...],  # minutes
            'pH': [7.40, 7.38, 7.32, ...],
            'lactate': [1.2, 1.5, 2.1, ...],
            ...
        }

    Returns:
        Feature vector including:
        - Raw values (latest measurements)
        - Trend features (slopes)
        - Stability features (std dev)
        - Anomaly counts
    """
    features = {}

    for key in ['pH', 'PO2', 'PCO2', 'lactate', 'K_plus']:
        values = np.array(blood_gas_timeline[key])
        timestamps = np.array(blood_gas_timeline['timestamps'])

        # Raw features: latest value
        # 原始特征：最新值
        features[f'{key}_latest'] = values[-1]

        # Trend features: linear regression slope
        # 趋势特征：线性回归斜率（正=上升，负=下降）
        if len(values) >= 2:
            slope, _ = np.polyfit(timestamps, values, 1)
            features[f'{key}_trend'] = slope

        # Stability features: coefficient of variation
        # 稳定性特征：变异系数（越大越不稳定）
        features[f'{key}_cv'] = np.std(values) / (np.mean(values) + 1e-8)

        # Anomaly count: number of out-of-range values
        # 异常计数：超出正常范围的次数
        thresholds = NORMAL_RANGES[key]
        anomalies = np.sum((values < thresholds[0]) | (values > thresholds[1]))
        features[f'{key}_anomaly_count'] = anomalies

    return np.array(list(features.values()))


# Normal ranges for anomaly detection
# 正常范围定义，用于异常检测
NORMAL_RANGES = {
    'pH': (7.35, 7.45),
    'PO2': (300, 500),      # mmHg (in perfusion context)
    'PCO2': (35, 45),       # mmHg
    'lactate': (0.5, 2.0),  # mmol/L
    'K_plus': (3.5, 5.0),   # mEq/L
    'IL_6': (0, 50),        # pg/mL (elevated = inflammation)
}
```

---

## 4. Extension 2: Perfusion Knowledge Graph Schema

### New Entity Types

```python
# Schema extension for perfusion domain
# 灌注领域的 Schema 扩展

PERFUSION_ENTITY_TYPES = {
    # Perfusion strategy entities
    # 灌注策略相关实体
    "PerfusionStrategy": {
        "description": "Specific perfusion approach",
        "properties": {
            "name": "str",                    # e.g., "HTK antegrade perfusion"
            "solution_type": "enum",          # HTK, UW, Celsior, Custodiol
            "temperature_celsius": "float",   # 4, 10, 20, 37
            "pressure_mmhg": "float",         # 50-70 typical
            "flow_rate_ml_min": "float",      # varies by heart size
            "duration_minutes": "float"
        }
    },

    # Monitoring indicator entities
    # 监测指标相关实体
    "MonitoringIndicator": {
        "description": "Measurable parameter during perfusion",
        "properties": {
            "name": "str",                    # e.g., "arterial pH"
            "category": "enum",               # blood_gas, inflammatory, hemodynamic
            "unit": "str",                    # pH, mmHg, pg/mL, etc.
            "normal_range": "tuple",          # (min, max)
            "critical_threshold": "float",    # value requiring immediate action
            "sampling_frequency": "str"       # "every 5 minutes"
        }
    },

    # Intervention entities
    # 介入措施相关实体
    "Intervention": {
        "description": "Corrective action during perfusion",
        "properties": {
            "name": "str",                    # e.g., "NaHCO3 supplementation"
            "category": "enum",               # adjust_params, add_supplement, change_solution
            "trigger_condition": "str",       # "pH < 7.2"
            "dosage_protocol": "str",         # specific instructions
            "expected_effect": "str",         # "pH increase by 0.1-0.2"
            "time_to_effect_minutes": "int"
        }
    },

    # Perfusion outcome entities
    # 灌注结果相关实体
    "PerfusionOutcome": {
        "description": "Result of perfusion process",
        "properties": {
            "quality_score": "float",         # 0-100
            "usable_for_transplant": "bool",
            "contractility": "enum",          # excellent, good, fair, poor
            "rhythm": "enum",                 # sinus, afib, asystole
            "coronary_flow": "enum",          # patent, partial_occlusion, occluded
            "complications": "list[str]"
        }
    }
}


PERFUSION_RELATION_TYPES = {
    # Strategy relations
    "APPLIES_STRATEGY": {
        "description": "PerfusionCase uses a specific strategy",
        "source": "PerfusionCase",
        "target": "PerfusionStrategy",
        "properties": {"execution_time": "timestamp"}
    },

    # Monitoring relations
    "MONITORS": {
        "description": "Case monitors specific indicator",
        "source": "PerfusionCase",
        "target": "MonitoringIndicator",
        "properties": {
            "measurement_time": "timestamp",
            "measured_value": "float",
            "is_abnormal": "bool"
        }
    },

    # Intervention relations
    "TRIGGERS_INTERVENTION": {
        "description": "Abnormal indicator triggers intervention",
        "source": "MonitoringIndicator",
        "target": "Intervention",
        "properties": {
            "trigger_threshold": "float",
            "trigger_time": "timestamp"
        }
    },

    # Outcome relations
    "RESULTS_IN": {
        "description": "Strategy leads to outcome",
        "source": "PerfusionStrategy",
        "target": "PerfusionOutcome",
        "properties": {
            "success_rate": "float",
            "evidence_source": "str"
        }
    },

    # Causal relations (from existing triples)
    # 因果关系（从已有三元组继承）
    "INCREASES_RISK": {"source": "RiskFactor", "target": "Complication"},
    "TREATS": {"source": "TreatmentRegimen", "target": "Complication"},
    "MITIGATES": {"source": "Intervention", "target": "Complication"},
    "ALLEVIATES": {"source": "TreatmentRegimen", "target": "Complication"}
}
```

### Instance Layer Nodes

```python
@dataclass
class PerfusionCaseNode:
    """
    Instance node representing a single perfusion case
    实例节点：表示单个灌注案例
    """
    case_id: str
    heart_source: Literal["DBD", "DCD"]
    donor_age: float
    cold_ischemic_time_minutes: float
    warm_ischemic_time_minutes: float  # DCD only

    # Time-series data stored as arrays
    # 时序数据以数组形式存储
    blood_gas_timeline: Dict[str, List[float]] = field(default_factory=dict)
    # {
    #     'timestamps': [0, 5, 10, 15, ...],
    #     'pH': [7.40, 7.38, 7.32, ...],
    #     'PO2': [450, 420, 380, ...],
    #     ...
    # }

    inflammatory_timeline: Dict[str, List[float]] = field(default_factory=dict)
    # {
    #     'timestamps': [0, 15, 30, ...],  # less frequent
    #     'IL_6': [20, 35, 48, ...],
    #     'IL_8': [15, 28, 42, ...],
    #     ...
    # }

    perfusion_params_timeline: Dict[str, List[float]] = field(default_factory=dict)
    # {
    #     'timestamps': [0, 1, 2, ...],  # continuous
    #     'pressure': [55, 58, 52, ...],
    #     'flow_rate': [1.5, 1.4, 1.6, ...],
    #     'temperature': [4.0, 4.1, 4.0, ...],
    # }

    # Final outcome
    # 最终结果
    outcome: Optional[Dict] = None
    # {
    #     'quality_score': 85,
    #     'usable': True,
    #     'contractility': 'good',
    #     'rhythm': 'sinus',
    #     'complications': ['mild_edema']
    # }


@dataclass
class HeartInstanceNode:
    """
    Instance node representing a heart being perfused
    实例节点：表示待灌注的心脏
    """
    heart_id: str
    donor_type: Literal["DBD", "DCD"]

    # Extracted numerical features (from text descriptions)
    # 从文字描述中提取的数值特征
    anatomical_features: Dict[str, float] = field(default_factory=dict)
    # {
    #     'aortic_diameter_cm': 3.2,
    #     'lv_wall_thickness_cm': 1.1,
    #     'ejection_fraction_percent': 55,
    # }

    # Categorical features
    # 分类特征
    valve_condition: str = "normal"       # normal, mild_regurgitation, moderate, severe
    coronary_anatomy: str = "normal"      # normal, calcification, stenosis

    # Raw text for LLM reasoning
    # 原始文本，用于LLM推理
    raw_description: str = ""

    # Precomputed feature vector for GNN
    # 预计算的特征向量，用于GNN输入
    feature_vector: Optional[np.ndarray] = None
```

---

## 5. Extension 3: Real-time Monitoring System

### Monitor Architecture

```python
class RealTimePerfusionMonitor:
    """
    Real-time perfusion monitoring and intervention recommendation
    实时灌注监控与介入推荐系统

    Workflow:
    1. Receive new measurements →
    2. Update knowledge graph →
    3. Run GNN prediction →
    4. Check alerts →
    5. Recommend interventions
    """

    def __init__(
        self,
        kg_client: Neo4jClient,
        gnn_model: TemporalPerfusionGNN,
        llm_agent: MedicalExpertAgent
    ):
        self.kg = kg_client
        self.gnn = gnn_model
        self.llm = llm_agent

        # Alert thresholds
        # 警报阈值配置
        self.alert_thresholds = {
            'pH': {'low': 7.30, 'critical_low': 7.20, 'high': 7.50},
            'lactate': {'warning': 2.5, 'critical': 4.0},
            'K_plus': {'low': 3.0, 'high': 5.5, 'critical_high': 6.0},
            'IL_6': {'elevated': 50, 'high': 100}
        }

        # Active monitoring sessions
        self._active_cases: Dict[str, PerfusionSession] = {}

    def start_monitoring(
        self,
        case_id: str,
        strategy: PerfusionStrategy,
        heart: HeartInstanceNode
    ) -> None:
        """
        Initialize monitoring for a new perfusion case
        为新的灌注案例初始化监控
        """
        session = PerfusionSession(
            case_id=case_id,
            strategy=strategy,
            heart=heart,
            start_time=datetime.now(),
            measurements=[]
        )
        self._active_cases[case_id] = session

        # Create initial graph nodes
        # 创建初始图节点
        self.kg.create_perfusion_case(case_id, strategy, heart)

        logger.info(f"Started monitoring case {case_id}")

    def process_measurement(
        self,
        case_id: str,
        measurement: PerfusionMeasurement
    ) -> MonitoringResult:
        """
        Process new measurement and return monitoring result
        处理新的测量数据并返回监控结果

        Args:
            case_id: Active perfusion case ID
            measurement: New measurement data containing:
                - timestamp: minutes since perfusion start
                - blood_gas: {pH, PO2, PCO2, HCO3, lactate}
                - inflammatory: {IL_6, IL_8, TNF_alpha} (optional)
                - perfusion_params: {pressure, flow_rate, temperature}

        Returns:
            MonitoringResult with:
                - status: 'normal', 'warning', 'critical'
                - alerts: list of triggered alerts
                - predictions: GNN predictions for next state
                - recommendations: intervention suggestions (if needed)
        """
        session = self._active_cases.get(case_id)
        if not session:
            raise ValueError(f"No active session for case {case_id}")

        # 1. Store measurement
        # 存储测量数据
        session.measurements.append(measurement)

        # 2. Update knowledge graph
        # 更新知识图谱（添加时序节点和边）
        self.kg.add_temporal_measurement(case_id, measurement)

        # 3. Check for alerts (rule-based)
        # 基于规则检查警报
        alerts = self._check_alerts(measurement)

        # 4. Run GNN prediction
        # 运行GNN预测
        temporal_features = self._prepare_temporal_features(session)
        graph_data = self.kg.get_perfusion_subgraph(case_id)

        with torch.no_grad():
            predictions = self.gnn(temporal_features, graph_data)

        # 5. Determine status
        # 确定状态级别
        status = self._determine_status(alerts, predictions)

        # 6. Generate recommendations if needed
        # 如果需要，生成介入建议
        recommendations = None
        if status in ['warning', 'critical']:
            recommendations = self._generate_recommendations(
                alerts=alerts,
                predictions=predictions,
                session=session
            )

        return MonitoringResult(
            timestamp=measurement.timestamp,
            status=status,
            alerts=alerts,
            predictions={
                'quality_score': predictions['quality_score'].item(),
                'next_pH': predictions['next_state'][0].item(),
                'next_lactate': predictions['next_state'][3].item(),
                'need_intervention': predictions['need_intervention']
            },
            recommendations=recommendations
        )

    def _check_alerts(self, measurement: PerfusionMeasurement) -> List[Alert]:
        """
        Check measurement against alert thresholds
        检查测量值是否触发警报
        """
        alerts = []

        # pH alerts
        if measurement.blood_gas['pH'] < self.alert_thresholds['pH']['critical_low']:
            alerts.append(Alert(
                type='acidosis',
                severity='critical',
                indicator=f"pH = {measurement.blood_gas['pH']} (critical < 7.20)",
                message="Severe metabolic acidosis detected"
            ))
        elif measurement.blood_gas['pH'] < self.alert_thresholds['pH']['low']:
            alerts.append(Alert(
                type='acidosis',
                severity='warning',
                indicator=f"pH = {measurement.blood_gas['pH']} (normal: 7.35-7.45)",
                message="Mild acidosis - monitor trend"
            ))

        # Lactate alerts
        if measurement.blood_gas['lactate'] > self.alert_thresholds['lactate']['critical']:
            alerts.append(Alert(
                type='metabolic_stress',
                severity='critical',
                indicator=f"Lactate = {measurement.blood_gas['lactate']} mmol/L (critical > 4.0)",
                message="Severe metabolic stress - possible ischemia"
            ))
        elif measurement.blood_gas['lactate'] > self.alert_thresholds['lactate']['warning']:
            alerts.append(Alert(
                type='metabolic_stress',
                severity='warning',
                indicator=f"Lactate = {measurement.blood_gas['lactate']} mmol/L (normal < 2.0)",
                message="Elevated lactate - assess perfusion adequacy"
            ))

        # Potassium alerts (hyperkalemia is dangerous)
        if measurement.blood_gas.get('K_plus', 4.0) > self.alert_thresholds['K_plus']['critical_high']:
            alerts.append(Alert(
                type='hyperkalemia',
                severity='critical',
                indicator=f"K+ = {measurement.blood_gas['K_plus']} mEq/L (critical > 6.0)",
                message="Severe hyperkalemia - risk of cardiac arrest"
            ))

        return alerts

    def _generate_recommendations(
        self,
        alerts: List[Alert],
        predictions: Dict,
        session: PerfusionSession
    ) -> List[InterventionRecommendation]:
        """
        Generate intervention recommendations using KG + GNN + LLM
        使用知识图谱 + GNN + LLM 生成介入建议
        """
        recommendations = []

        # 1. Query knowledge graph for similar successful cases
        # 从知识图谱中查询相似的成功案例
        for alert in alerts:
            similar_interventions = self.kg.query(f"""
                MATCH (pc:PerfusionCase)-[:MONITORS]->(m:MonitoringIndicator)
                WHERE m.name = '{alert.type}'
                  AND m.measured_value BETWEEN {alert.value * 0.9} AND {alert.value * 1.1}
                  AND pc.outcome.usable = true
                MATCH (pc)-[:APPLIED_INTERVENTION]->(i:Intervention)
                RETURN i.name, i.dosage_protocol, i.expected_effect,
                       count(*) as success_count
                ORDER BY success_count DESC
                LIMIT 5
            """)

            for intervention in similar_interventions:
                recommendations.append(InterventionRecommendation(
                    intervention=intervention['name'],
                    protocol=intervention['dosage_protocol'],
                    expected_effect=intervention['expected_effect'],
                    confidence=intervention['success_count'] / 100,  # Normalize
                    source='knowledge_graph'
                ))

        # 2. Use LLM for personalized reasoning
        # 使用LLM进行个性化推理
        llm_context = {
            'current_state': session.get_latest_state(),
            'alerts': [a.to_dict() for a in alerts],
            'predictions': predictions,
            'heart_features': session.heart.to_dict(),
            'kg_recommendations': [r.to_dict() for r in recommendations[:3]]
        }

        llm_recommendation = self.llm.reason(
            prompt=INTERVENTION_REASONING_PROMPT,
            context=llm_context
        )

        # Add LLM recommendation with lower confidence (needs validation)
        # 添加LLM建议（置信度较低，需要验证）
        recommendations.append(InterventionRecommendation(
            intervention=llm_recommendation['action'],
            protocol=llm_recommendation['protocol'],
            expected_effect=llm_recommendation['expected_effect'],
            confidence=0.7,  # LLM recommendations have lower base confidence
            source='llm_reasoning',
            reasoning=llm_recommendation['reasoning']
        ))

        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)

        return recommendations


INTERVENTION_REASONING_PROMPT = """
You are a cardiac perfusion specialist AI assistant. Based on the following information,
recommend the most appropriate intervention.

Current perfusion state:
{current_state}

Triggered alerts:
{alerts}

GNN predictions for next time point:
{predictions}

Heart characteristics:
{heart_features}

Knowledge graph recommended interventions (from historical cases):
{kg_recommendations}

Please provide:
1. Your recommended intervention action
2. Specific protocol/dosage
3. Expected effect and timeline
4. Your reasoning (cite relevant factors)

Respond in JSON format:
{{
    "action": "...",
    "protocol": "...",
    "expected_effect": "...",
    "reasoning": "..."
}}
"""
```

---

## 6. Data Processing Pipeline

### Processing Different Data Types

| Data Type | Source | Processing Method | Graph Representation |
|-----------|--------|-------------------|---------------------|
| **Perfusion Strategy** | Structured input | Direct mapping | `PerfusionStrategy` node |
| **Blood Gas (time-series)** | Lab equipment | LSTM encoding + feature engineering | `Measurement` nodes with `NEXT` edges |
| **Inflammatory Markers** | Lab equipment | Same as blood gas | Merged with `Measurement` nodes |
| **Perfusion Parameters** | Machine sensors | Real-time streaming | Node properties (updated) |
| **Perfusion Outcome** | Clinical assessment | Labeling (supervision target) | `PerfusionOutcome` node |
| **Heart Description (text)** | Clinical notes | Regex + LLM extraction | `HeartInstance` node with feature vector |

### Text-to-Numerical Feature Extraction

```python
class HeartAnatomyExtractor:
    """
    Extract numerical features from free-text heart descriptions
    从自由文本的心脏描述中提取数值特征
    """

    # Regex patterns for common measurements
    # 常见测量值的正则表达式模式
    PATTERNS = {
        'aortic_diameter_cm': [
            r'aort(?:ic|a).*?(?:diameter|width).*?(\d+\.?\d*)\s*cm',
            r'(\d+\.?\d*)\s*cm.*?aort',
        ],
        'lv_wall_thickness_cm': [
            r'(?:left\s+)?ventricle.*?(?:wall\s+)?thickness.*?(\d+\.?\d*)\s*cm',
            r'LV.*?(\d+\.?\d*)\s*cm',
        ],
        'ejection_fraction_percent': [
            r'(?:ejection\s+fraction|EF|LVEF).*?(\d+\.?\d*)%?',
            r'(\d+\.?\d*)%?\s*(?:ejection|EF)',
        ],
        'lv_diameter_cm': [
            r'LV(?:ID)?[ds]?\s*[:=]?\s*(\d+\.?\d*)\s*cm',
        ]
    }

    def extract(self, text: str) -> Dict[str, Optional[float]]:
        """
        Extract all available numerical features from text
        从文本中提取所有可用的数值特征
        """
        features = {}
        text_lower = text.lower()

        for feature_name, patterns in self.PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        features[feature_name] = float(match.group(1))
                        break
                    except ValueError:
                        continue

            if feature_name not in features:
                features[feature_name] = None

        return features

    def extract_with_llm_fallback(
        self,
        text: str,
        llm: LLMClient
    ) -> Dict[str, Optional[float]]:
        """
        Extract features using regex first, then LLM for complex cases
        先用正则提取，复杂情况用LLM兜底
        """
        # Try regex first (fast)
        features = self.extract(text)

        # Count missing features
        missing = [k for k, v in features.items() if v is None]

        if missing and len(missing) < len(features):
            # Use LLM only for missing features
            # 仅对缺失特征使用LLM（节省成本）
            llm_prompt = f"""
            Extract the following measurements from this heart assessment:
            {missing}

            Text: {text}

            Return JSON with numerical values or null if not found.
            """

            llm_response = llm.invoke(llm_prompt)
            llm_features = json.loads(llm_response)

            for key in missing:
                if llm_features.get(key) is not None:
                    features[key] = llm_features[key]

        return features
```

---

## 7. Implementation Roadmap

### Phase 1: Knowledge Graph Import (Week 1-2)
- [ ] Import existing 15,068 triples to Neo4j
- [ ] Extend schema with perfusion entity/relation types
- [ ] Create indexes for efficient querying
- [ ] Validate graph connectivity

### Phase 2: Temporal GNN Development (Week 3-5)
- [ ] Implement `TemporalPerfusionGNN` class
- [ ] Prepare training data (blood gas time-series)
- [ ] Train model on historical perfusion cases
- [ ] Evaluate: MAE for quality score, F1 for risk classification

### Phase 3: Real-time Monitor (Week 6-7)
- [ ] Implement `RealTimePerfusionMonitor` class
- [ ] Integrate with knowledge graph queries
- [ ] Add LLM reasoning for recommendations
- [ ] Build alert/notification system

### Phase 4: Integration & Testing (Week 8)
- [ ] End-to-end pipeline testing
- [ ] Backtesting on historical cases
- [ ] Performance optimization (< 1s inference)
- [ ] Documentation and deployment

---

## 8. Summary

### Can We Achieve Perfusion Outcome Prediction and Intervention Recommendation?

**YES, and the architecture is more powerful than the original transplant prediction system.**

| Capability | Implementation | Confidence |
|------------|---------------|------------|
| **Predict perfusion outcome** | GNN + KG retrieval + LLM reasoning | ✅ High |
| **Real-time problem detection** | Threshold alerts + Temporal GNN trend prediction | ✅ High |
| **Intervention recommendation** | KG similar case retrieval + GNN simulation + LLM personalization | ✅ High |

### Key Advantages

1. **Multi-modal fusion**: Combines time-series (blood gas), text (heart description), and graph (knowledge) data
2. **Explainable**: Knowledge graph provides traceable reasoning paths
3. **Real-time capable**: Streaming architecture supports online inference
4. **Continuously improving**: New cases automatically enhance the knowledge graph
