"""
Real-time Perfusion Monitoring System

实时灌注监控系统，用于：
1. 实时接收灌注数据
2. 检测异常并触发警报
3. 使用 GNN 预测趋势
4. 推荐介入措施

Usage:
    monitor = RealTimePerfusionMonitor(kg_client, gnn_model, llm_agent)
    monitor.start_monitoring(case_id, strategy, heart)

    # During perfusion
    result = monitor.process_measurement(case_id, measurement)
    if result.status == 'critical':
        print(result.recommendations)
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging

import torch
import numpy as np

from .temporal_gnn import TemporalPerfusionGNN, TemporalGNNConfig, create_dummy_graph


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class MonitorStatus(Enum):
    """Monitoring status"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Represents an alert triggered during monitoring"""
    type: str                      # e.g., "acidosis", "hyperkalemia"
    severity: AlertSeverity
    indicator: str                 # e.g., "pH = 7.26 (normal: 7.35-7.45)"
    message: str                   # Human-readable message
    value: float = 0.0             # Actual measured value
    threshold: float = 0.0         # Threshold that was crossed
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'type': self.type,
            'severity': self.severity.value,
            'indicator': self.indicator,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class InterventionRecommendation:
    """Represents a recommended intervention"""
    intervention: str              # e.g., "NaHCO3 supplementation"
    protocol: str                  # e.g., "50 mEq IV over 10 minutes"
    expected_effect: str           # e.g., "pH increase by 0.1-0.2 within 15 minutes"
    confidence: float              # 0-1 confidence score
    source: str                    # "knowledge_graph", "llm_reasoning", "rule_based"
    reasoning: str = ""            # Explanation for the recommendation
    urgency: str = "medium"        # "low", "medium", "high", "immediate"

    def to_dict(self) -> Dict:
        return {
            'intervention': self.intervention,
            'protocol': self.protocol,
            'expected_effect': self.expected_effect,
            'confidence': self.confidence,
            'source': self.source,
            'reasoning': self.reasoning,
            'urgency': self.urgency
        }


@dataclass
class PerfusionMeasurement:
    """Single measurement during perfusion"""
    timestamp: int                 # Minutes since perfusion start
    blood_gas: Dict[str, float]    # {pH, PO2, PCO2, HCO3, lactate, K_plus, Na_plus}
    inflammatory: Dict[str, float] = field(default_factory=dict)  # {IL_6, IL_8, TNF_alpha}
    perfusion_params: Dict[str, float] = field(default_factory=dict)  # {pressure, flow_rate, temperature}

    def to_feature_vector(self) -> np.ndarray:
        """Convert to feature vector for GNN input"""
        features = [
            self.blood_gas.get('pH', 7.4),
            self.blood_gas.get('PO2', 400),
            self.blood_gas.get('PCO2', 40),
            self.blood_gas.get('lactate', 1.0),
            self.blood_gas.get('K_plus', 4.0),
            self.blood_gas.get('Na_plus', 140),
            self.inflammatory.get('IL_6', 20),
            self.inflammatory.get('IL_8', 20),
            self.inflammatory.get('TNF_alpha', 5),
            self.perfusion_params.get('pressure', 60),
            self.perfusion_params.get('flow_rate', 1.5),
            self.perfusion_params.get('temperature', 4)
        ]
        return np.array(features, dtype=np.float32)


@dataclass
class MonitoringResult:
    """Result of processing a measurement"""
    timestamp: int
    status: MonitorStatus
    alerts: List[Alert]
    predictions: Dict[str, Any]
    recommendations: Optional[List[InterventionRecommendation]] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'status': self.status.value,
            'alerts': [a.to_dict() for a in self.alerts],
            'predictions': self.predictions,
            'recommendations': [r.to_dict() for r in self.recommendations] if self.recommendations else None
        }


@dataclass
class PerfusionSession:
    """Active perfusion monitoring session"""
    case_id: str
    strategy: Dict[str, Any]       # Perfusion strategy parameters
    heart: Dict[str, Any]          # Heart characteristics
    start_time: datetime
    measurements: List[PerfusionMeasurement] = field(default_factory=list)
    alerts_history: List[Alert] = field(default_factory=list)

    def get_temporal_features(self, window_size: int = 10) -> np.ndarray:
        """Get recent measurements as temporal feature array"""
        recent = self.measurements[-window_size:] if len(self.measurements) >= window_size else self.measurements

        if not recent:
            # Return default values if no measurements yet
            return np.zeros((1, 12), dtype=np.float32)

        features = np.array([m.to_feature_vector() for m in recent])

        # Pad if necessary
        if len(features) < window_size:
            padding = np.tile(features[0], (window_size - len(features), 1))
            features = np.vstack([padding, features])

        return features

    def get_latest_state(self) -> Dict[str, Any]:
        """Get the latest measurement state"""
        if not self.measurements:
            return {}

        latest = self.measurements[-1]
        return {
            'timestamp': latest.timestamp,
            'blood_gas': latest.blood_gas,
            'inflammatory': latest.inflammatory,
            'perfusion_params': latest.perfusion_params
        }


class AlertThresholds:
    """Alert threshold configuration"""

    # Blood gas thresholds
    # 血气阈值配置
    pH = {
        'low': 7.30,
        'critical_low': 7.20,
        'high': 7.50,
        'critical_high': 7.60
    }

    PO2 = {
        'low': 300,
        'critical_low': 200,
        'high': 600
    }

    PCO2 = {
        'low': 30,
        'high': 50,
        'critical_high': 60
    }

    lactate = {
        'warning': 2.5,
        'high': 4.0,
        'critical': 6.0
    }

    K_plus = {
        'low': 3.0,
        'critical_low': 2.5,
        'high': 5.5,
        'critical_high': 6.0
    }

    Na_plus = {
        'low': 130,
        'high': 150
    }

    # Inflammatory markers
    # 炎症因子阈值
    IL_6 = {
        'elevated': 50,
        'high': 100,
        'critical': 200
    }

    IL_8 = {
        'elevated': 50,
        'high': 100
    }

    TNF_alpha = {
        'elevated': 20,
        'high': 50
    }


class RealTimePerfusionMonitor:
    """
    Real-time perfusion monitoring and intervention recommendation system

    Combines:
    1. Rule-based alert detection (fast, reliable)
    2. GNN-based trend prediction (ML-powered)
    3. Knowledge graph retrieval (evidence-based)
    4. LLM reasoning (personalized recommendations)
    """

    def __init__(
        self,
        kg_client: Any = None,
        gnn_model: Optional[TemporalPerfusionGNN] = None,
        llm_agent: Any = None,
        config: Optional[TemporalGNNConfig] = None
    ):
        """
        Initialize monitor

        Args:
            kg_client: Neo4j knowledge graph client
            gnn_model: Trained TemporalPerfusionGNN model
            llm_agent: LLM agent for reasoning
            config: GNN configuration
        """
        self.kg = kg_client
        self.llm = llm_agent

        # Initialize or load GNN model
        if gnn_model is not None:
            self.gnn = gnn_model
        else:
            config = config or TemporalGNNConfig()
            self.gnn = TemporalPerfusionGNN(config)

        self.gnn.eval()  # Set to evaluation mode

        # Active monitoring sessions
        self._active_sessions: Dict[str, PerfusionSession] = {}

        # Intervention rules (from knowledge graph or predefined)
        # 介入规则（从知识图谱加载或使用预定义规则）
        self._intervention_rules = self._load_intervention_rules()

    def _load_intervention_rules(self) -> Dict[str, List[Dict]]:
        """
        Load intervention rules from knowledge graph or use defaults

        Returns mapping from alert type to list of intervention options
        """
        # Default rules (can be overridden by KG query)
        # 默认规则（可被知识图谱查询覆盖）
        return {
            'acidosis': [
                {
                    'intervention': 'NaHCO3 supplementation',
                    'protocol': '50 mEq IV bolus, may repeat based on pH response',
                    'expected_effect': 'pH increase by 0.1-0.2 within 10-15 minutes',
                    'confidence': 0.85,
                    'conditions': {'pH': {'max': 7.30}}
                },
                {
                    'intervention': 'Increase perfusion flow',
                    'protocol': 'Increase flow rate by 10-20%',
                    'expected_effect': 'Improved tissue oxygenation and lactate clearance',
                    'confidence': 0.70,
                    'conditions': {'lactate': {'min': 2.5}}
                }
            ],
            'hyperkalemia': [
                {
                    'intervention': 'Calcium gluconate',
                    'protocol': '1g IV over 2-3 minutes for cardiac protection',
                    'expected_effect': 'Cardioprotection within 1-3 minutes',
                    'confidence': 0.90,
                    'conditions': {'K_plus': {'min': 6.0}}
                },
                {
                    'intervention': 'Insulin + Glucose',
                    'protocol': '10 units regular insulin with 25g dextrose',
                    'expected_effect': 'K+ decrease by 0.5-1.0 mEq/L within 30-60 minutes',
                    'confidence': 0.80,
                    'conditions': {'K_plus': {'min': 5.5}}
                }
            ],
            'metabolic_stress': [
                {
                    'intervention': 'Optimize perfusion parameters',
                    'protocol': 'Adjust pressure to 50-70 mmHg, flow to 1.5-2.0 L/min',
                    'expected_effect': 'Improved tissue perfusion, lactate stabilization',
                    'confidence': 0.75,
                    'conditions': {'lactate': {'min': 2.5}}
                }
            ],
            'inflammation': [
                {
                    'intervention': 'Methylprednisolone',
                    'protocol': '500mg IV bolus',
                    'expected_effect': 'Reduced inflammatory response within 1-2 hours',
                    'confidence': 0.70,
                    'conditions': {'IL_6': {'min': 100}}
                }
            ]
        }

    def start_monitoring(
        self,
        case_id: str,
        strategy: Dict[str, Any],
        heart: Dict[str, Any]
    ) -> None:
        """
        Start monitoring a new perfusion case

        Args:
            case_id: Unique case identifier
            strategy: Perfusion strategy parameters
            heart: Heart characteristics
        """
        session = PerfusionSession(
            case_id=case_id,
            strategy=strategy,
            heart=heart,
            start_time=datetime.now(),
            measurements=[],
            alerts_history=[]
        )

        self._active_sessions[case_id] = session
        logger.info(f"Started monitoring case {case_id}")

        # Create knowledge graph nodes if KG client available
        if self.kg:
            try:
                self._create_kg_nodes(case_id, strategy, heart)
            except Exception as e:
                logger.warning(f"Failed to create KG nodes: {e}")

    def stop_monitoring(self, case_id: str) -> Optional[PerfusionSession]:
        """
        Stop monitoring a case and return session data

        Args:
            case_id: Case to stop monitoring

        Returns:
            The completed session, or None if not found
        """
        session = self._active_sessions.pop(case_id, None)
        if session:
            logger.info(f"Stopped monitoring case {case_id}")
        return session

    def process_measurement(
        self,
        case_id: str,
        measurement: PerfusionMeasurement
    ) -> MonitoringResult:
        """
        Process a new measurement and return monitoring result

        This is the main entry point called for each new measurement during perfusion

        Args:
            case_id: Active case ID
            measurement: New measurement data

        Returns:
            MonitoringResult with status, alerts, predictions, and recommendations
        """
        session = self._active_sessions.get(case_id)
        if not session:
            raise ValueError(f"No active session for case {case_id}")

        # 1. Store measurement
        # 存储测量数据
        session.measurements.append(measurement)

        # 2. Check for rule-based alerts (fast)
        # 规则检查（快速）
        alerts = self._check_alerts(measurement)

        # 3. Run GNN prediction
        # GNN 预测
        predictions = self._run_gnn_prediction(session)

        # 4. Determine overall status
        # 确定整体状态
        status = self._determine_status(alerts, predictions)

        # 5. Generate recommendations if needed
        # 如需干预，生成建议
        recommendations = None
        if status in [MonitorStatus.WARNING, MonitorStatus.CRITICAL]:
            recommendations = self._generate_recommendations(
                alerts=alerts,
                predictions=predictions,
                session=session
            )

        # 6. Store alerts in history
        session.alerts_history.extend(alerts)

        # 7. Update knowledge graph if available
        if self.kg:
            try:
                self._update_kg_measurement(case_id, measurement, alerts)
            except Exception as e:
                logger.warning(f"Failed to update KG: {e}")

        return MonitoringResult(
            timestamp=measurement.timestamp,
            status=status,
            alerts=alerts,
            predictions=predictions,
            recommendations=recommendations
        )

    def _check_alerts(self, measurement: PerfusionMeasurement) -> List[Alert]:
        """
        Check measurement against alert thresholds

        使用规则检查测量值是否触发警报
        """
        alerts = []

        # pH checks
        ph = measurement.blood_gas.get('pH', 7.4)
        if ph < AlertThresholds.pH['critical_low']:
            alerts.append(Alert(
                type='acidosis',
                severity=AlertSeverity.CRITICAL,
                indicator=f"pH = {ph:.2f} (critical < 7.20)",
                message="Severe metabolic acidosis - immediate intervention required",
                value=ph,
                threshold=AlertThresholds.pH['critical_low']
            ))
        elif ph < AlertThresholds.pH['low']:
            alerts.append(Alert(
                type='acidosis',
                severity=AlertSeverity.WARNING,
                indicator=f"pH = {ph:.2f} (normal: 7.35-7.45)",
                message="Mild acidosis - monitor trend closely",
                value=ph,
                threshold=AlertThresholds.pH['low']
            ))

        # Lactate checks
        lactate = measurement.blood_gas.get('lactate', 1.0)
        if lactate > AlertThresholds.lactate['critical']:
            alerts.append(Alert(
                type='metabolic_stress',
                severity=AlertSeverity.CRITICAL,
                indicator=f"Lactate = {lactate:.1f} mmol/L (critical > 6.0)",
                message="Severe metabolic stress - possible severe ischemia",
                value=lactate,
                threshold=AlertThresholds.lactate['critical']
            ))
        elif lactate > AlertThresholds.lactate['high']:
            alerts.append(Alert(
                type='metabolic_stress',
                severity=AlertSeverity.HIGH,
                indicator=f"Lactate = {lactate:.1f} mmol/L (high > 4.0)",
                message="High lactate - assess perfusion adequacy",
                value=lactate,
                threshold=AlertThresholds.lactate['high']
            ))
        elif lactate > AlertThresholds.lactate['warning']:
            alerts.append(Alert(
                type='metabolic_stress',
                severity=AlertSeverity.WARNING,
                indicator=f"Lactate = {lactate:.1f} mmol/L (normal < 2.0)",
                message="Elevated lactate - monitor trend",
                value=lactate,
                threshold=AlertThresholds.lactate['warning']
            ))

        # Potassium checks
        k_plus = measurement.blood_gas.get('K_plus', 4.0)
        if k_plus > AlertThresholds.K_plus['critical_high']:
            alerts.append(Alert(
                type='hyperkalemia',
                severity=AlertSeverity.CRITICAL,
                indicator=f"K+ = {k_plus:.1f} mEq/L (critical > 6.0)",
                message="Severe hyperkalemia - cardiac arrest risk",
                value=k_plus,
                threshold=AlertThresholds.K_plus['critical_high']
            ))
        elif k_plus > AlertThresholds.K_plus['high']:
            alerts.append(Alert(
                type='hyperkalemia',
                severity=AlertSeverity.WARNING,
                indicator=f"K+ = {k_plus:.1f} mEq/L (high > 5.5)",
                message="Elevated potassium - monitor ECG",
                value=k_plus,
                threshold=AlertThresholds.K_plus['high']
            ))
        elif k_plus < AlertThresholds.K_plus['critical_low']:
            alerts.append(Alert(
                type='hypokalemia',
                severity=AlertSeverity.HIGH,
                indicator=f"K+ = {k_plus:.1f} mEq/L (low < 2.5)",
                message="Severe hypokalemia - arrhythmia risk",
                value=k_plus,
                threshold=AlertThresholds.K_plus['critical_low']
            ))

        # Inflammatory marker checks
        il6 = measurement.inflammatory.get('IL_6', 20)
        if il6 > AlertThresholds.IL_6['critical']:
            alerts.append(Alert(
                type='inflammation',
                severity=AlertSeverity.HIGH,
                indicator=f"IL-6 = {il6:.0f} pg/mL (critical > 200)",
                message="Severe inflammatory response",
                value=il6,
                threshold=AlertThresholds.IL_6['critical']
            ))
        elif il6 > AlertThresholds.IL_6['high']:
            alerts.append(Alert(
                type='inflammation',
                severity=AlertSeverity.WARNING,
                indicator=f"IL-6 = {il6:.0f} pg/mL (elevated > 100)",
                message="Elevated inflammation - consider anti-inflammatory",
                value=il6,
                threshold=AlertThresholds.IL_6['high']
            ))

        return alerts

    def _run_gnn_prediction(self, session: PerfusionSession) -> Dict[str, Any]:
        """
        Run GNN prediction for trend analysis

        使用 GNN 进行趋势预测
        """
        try:
            # Prepare temporal features
            temporal_features = session.get_temporal_features(window_size=10)
            temporal_tensor = torch.tensor(temporal_features, dtype=torch.float32).unsqueeze(0)

            # Create graph data (from KG subgraph or dummy)
            # TODO: Load actual subgraph from knowledge graph
            graph_data = create_dummy_graph()

            # Run prediction
            with torch.no_grad():
                output = self.gnn(temporal_tensor, graph_data)

            # Extract predictions
            predictions = {
                'quality_score': output['quality_score'].item(),
                'risk_probs': output['risk_probs'].squeeze().tolist(),
                'risk_level': ['low', 'medium', 'high', 'critical'][
                    output['risk_probs'].argmax().item()
                ],
                'intervention_prob': output['intervention_prob'].item(),
                'next_state': {
                    'predicted_pH': output['next_state'][0, 0].item(),
                    'predicted_lactate': output['next_state'][0, 3].item(),
                    'predicted_K_plus': output['next_state'][0, 4].item()
                }
            }

            return predictions

        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return {
                'quality_score': 50.0,
                'risk_level': 'unknown',
                'intervention_prob': 0.5,
                'error': str(e)
            }

    def _determine_status(
        self,
        alerts: List[Alert],
        predictions: Dict[str, Any]
    ) -> MonitorStatus:
        """
        Determine overall monitoring status

        基于警报和预测确定整体状态
        """
        # Check for critical alerts
        if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
            return MonitorStatus.CRITICAL

        # Check GNN predictions
        if predictions.get('intervention_prob', 0) > 0.8:
            return MonitorStatus.CRITICAL

        if predictions.get('risk_level') == 'critical':
            return MonitorStatus.CRITICAL

        # Check for high/warning alerts
        if any(a.severity in [AlertSeverity.HIGH, AlertSeverity.WARNING] for a in alerts):
            return MonitorStatus.WARNING

        if predictions.get('risk_level') in ['high', 'medium']:
            return MonitorStatus.WARNING

        return MonitorStatus.NORMAL

    def _generate_recommendations(
        self,
        alerts: List[Alert],
        predictions: Dict[str, Any],
        session: PerfusionSession
    ) -> List[InterventionRecommendation]:
        """
        Generate intervention recommendations

        结合规则、知识图谱和 LLM 生成介入建议
        """
        recommendations = []

        # 1. Rule-based recommendations
        # 基于规则的建议
        for alert in alerts:
            if alert.type in self._intervention_rules:
                for rule in self._intervention_rules[alert.type]:
                    # Check if conditions are met
                    applicable = True
                    for param, bounds in rule.get('conditions', {}).items():
                        value = alert.value if param == alert.type.split('_')[0] else 0
                        if 'min' in bounds and value < bounds['min']:
                            applicable = False
                        if 'max' in bounds and value > bounds['max']:
                            applicable = False

                    if applicable:
                        urgency = 'immediate' if alert.severity == AlertSeverity.CRITICAL else \
                                 'high' if alert.severity == AlertSeverity.HIGH else 'medium'

                        recommendations.append(InterventionRecommendation(
                            intervention=rule['intervention'],
                            protocol=rule['protocol'],
                            expected_effect=rule['expected_effect'],
                            confidence=rule['confidence'],
                            source='rule_based',
                            reasoning=f"Triggered by {alert.type}: {alert.indicator}",
                            urgency=urgency
                        ))

        # 2. Knowledge graph recommendations (if available)
        # 知识图谱建议
        if self.kg:
            try:
                kg_recommendations = self._query_kg_interventions(alerts, session)
                recommendations.extend(kg_recommendations)
            except Exception as e:
                logger.warning(f"KG query failed: {e}")

        # 3. LLM reasoning (if available and for complex cases)
        # LLM 推理（复杂情况）
        if self.llm and len(alerts) > 1:
            try:
                llm_recommendation = self._get_llm_recommendation(
                    alerts, predictions, session
                )
                if llm_recommendation:
                    recommendations.append(llm_recommendation)
            except Exception as e:
                logger.warning(f"LLM reasoning failed: {e}")

        # Sort by confidence and urgency
        urgency_order = {'immediate': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(
            key=lambda r: (urgency_order.get(r.urgency, 4), -r.confidence)
        )

        return recommendations[:5]  # Return top 5

    def _query_kg_interventions(
        self,
        alerts: List[Alert],
        session: PerfusionSession
    ) -> List[InterventionRecommendation]:
        """
        Query knowledge graph for similar successful interventions

        从知识图谱查询相似案例的成功介入
        """
        recommendations = []

        if not self.kg:
            return recommendations

        # Example query (adjust based on actual KG schema)
        for alert in alerts[:2]:  # Limit queries
            query = f"""
            MATCH (pc:PerfusionCase)-[:MONITORS]->(m:MonitoringIndicator)
            WHERE m.type = '{alert.type}'
              AND m.value >= {alert.value * 0.8}
              AND m.value <= {alert.value * 1.2}
            MATCH (pc)-[:APPLIED_INTERVENTION]->(i:Intervention)
            WHERE pc.outcome_quality_score > 70
            RETURN i.name as intervention,
                   i.protocol as protocol,
                   i.expected_effect as effect,
                   count(*) as success_count
            ORDER BY success_count DESC
            LIMIT 3
            """

            try:
                results = self.kg.query(query)
                for r in results:
                    recommendations.append(InterventionRecommendation(
                        intervention=r['intervention'],
                        protocol=r.get('protocol', 'Standard protocol'),
                        expected_effect=r.get('effect', 'Based on similar cases'),
                        confidence=min(0.9, 0.5 + r['success_count'] * 0.05),
                        source='knowledge_graph',
                        reasoning=f"Successful in {r['success_count']} similar cases"
                    ))
            except Exception as e:
                logger.warning(f"KG query failed for {alert.type}: {e}")

        return recommendations

    def _get_llm_recommendation(
        self,
        alerts: List[Alert],
        predictions: Dict[str, Any],
        session: PerfusionSession
    ) -> Optional[InterventionRecommendation]:
        """
        Get personalized recommendation from LLM

        使用 LLM 生成个性化建议
        """
        if not self.llm:
            return None

        prompt = f"""You are a cardiac perfusion specialist. Based on the following information,
provide the most appropriate intervention recommendation.

Current State:
{session.get_latest_state()}

Alerts:
{[a.to_dict() for a in alerts]}

GNN Predictions:
{predictions}

Heart Characteristics:
{session.heart}

Perfusion Strategy:
{session.strategy}

Please provide ONE concise recommendation in JSON format:
{{
    "intervention": "specific action",
    "protocol": "dosage/parameters",
    "expected_effect": "expected outcome",
    "reasoning": "brief explanation"
}}
"""

        try:
            response = self.llm.invoke(prompt)
            import json
            result = json.loads(response.content)

            return InterventionRecommendation(
                intervention=result['intervention'],
                protocol=result['protocol'],
                expected_effect=result['expected_effect'],
                confidence=0.75,  # LLM recommendations have moderate confidence
                source='llm_reasoning',
                reasoning=result.get('reasoning', ''),
                urgency='medium'
            )
        except Exception as e:
            logger.warning(f"LLM recommendation failed: {e}")
            return None

    def _create_kg_nodes(
        self,
        case_id: str,
        strategy: Dict,
        heart: Dict
    ) -> None:
        """Create knowledge graph nodes for new case"""
        if not self.kg:
            return

        # Create PerfusionCase node
        self.kg.query(f"""
            CREATE (pc:PerfusionCase {{
                id: '{case_id}',
                start_time: datetime(),
                strategy: '{strategy.get("name", "unknown")}',
                status: 'active'
            }})
        """)

    def _update_kg_measurement(
        self,
        case_id: str,
        measurement: PerfusionMeasurement,
        alerts: List[Alert]
    ) -> None:
        """Update knowledge graph with new measurement"""
        if not self.kg:
            return

        # Add measurement node and link to case
        self.kg.query(f"""
            MATCH (pc:PerfusionCase {{id: '{case_id}'}})
            CREATE (m:Measurement {{
                timestamp: {measurement.timestamp},
                pH: {measurement.blood_gas.get('pH', 7.4)},
                lactate: {measurement.blood_gas.get('lactate', 1.0)},
                K_plus: {measurement.blood_gas.get('K_plus', 4.0)}
            }})
            CREATE (pc)-[:HAS_MEASUREMENT]->(m)
        """)


if __name__ == "__main__":
    # Test the monitor
    print("Testing RealTimePerfusionMonitor...")

    monitor = RealTimePerfusionMonitor()

    # Start monitoring
    monitor.start_monitoring(
        case_id="TEST_001",
        strategy={"name": "HTK_antegrade", "temperature": 4, "pressure": 60},
        heart={"donor_type": "DCD", "aortic_diameter": 3.2}
    )

    # Simulate measurements
    measurements = [
        PerfusionMeasurement(
            timestamp=0,
            blood_gas={'pH': 7.40, 'PO2': 450, 'lactate': 1.2, 'K_plus': 4.0},
            inflammatory={'IL_6': 20},
            perfusion_params={'pressure': 60, 'flow_rate': 1.5, 'temperature': 4}
        ),
        PerfusionMeasurement(
            timestamp=5,
            blood_gas={'pH': 7.35, 'PO2': 420, 'lactate': 1.8, 'K_plus': 4.2},
            inflammatory={'IL_6': 35},
            perfusion_params={'pressure': 58, 'flow_rate': 1.5, 'temperature': 4}
        ),
        PerfusionMeasurement(
            timestamp=10,
            blood_gas={'pH': 7.25, 'PO2': 380, 'lactate': 3.5, 'K_plus': 5.8},  # Abnormal!
            inflammatory={'IL_6': 85},
            perfusion_params={'pressure': 55, 'flow_rate': 1.4, 'temperature': 4}
        )
    ]

    for m in measurements:
        print(f"\n=== Processing measurement at t={m.timestamp}min ===")
        result = monitor.process_measurement("TEST_001", m)

        print(f"Status: {result.status.value}")
        print(f"Alerts: {len(result.alerts)}")
        for alert in result.alerts:
            print(f"  - [{alert.severity.value}] {alert.type}: {alert.message}")

        print(f"Quality score: {result.predictions.get('quality_score', 'N/A'):.1f}")
        print(f"Risk level: {result.predictions.get('risk_level', 'N/A')}")

        if result.recommendations:
            print(f"Recommendations: {len(result.recommendations)}")
            for rec in result.recommendations[:2]:
                print(f"  - {rec.intervention} ({rec.confidence:.0%} confidence)")

    # Stop monitoring
    session = monitor.stop_monitoring("TEST_001")
    print(f"\n=== Session complete: {len(session.measurements)} measurements ===")
