"""
Perfusion Module - Machine Perfusion Monitoring and Prediction System

灌注模块 - 机械灌注监测与预测系统

This module provides components for real-time monitoring of heart perfusion
and predicting perfusion outcomes using temporal data and knowledge graphs.

Components:
- TemporalPerfusionGNN: Temporal Graph Neural Network for quality prediction
- RealTimePerfusionMonitor: Real-time monitoring with alerts
- HeartAnatomyExtractor: Extract numerical features from text reports
- PerfusionOutcomePredictor: Combined KG + GNN + LLM predictor
"""

from .temporal_gnn import (
    TemporalGNNConfig,
    TemporalEncoder,
    GraphEncoder,
    TemporalPerfusionGNN,
    PerfusionDataset,
    FeatureNormalizer,
    create_dummy_graph,
)

from .monitor import (
    RiskLevel,
    AlertType,
    AlertThresholds,
    Alert,
    PerfusionMeasurement,
    InterventionRecommendation,
    MonitoringResult,
    RuleBasedAlertChecker,
    KnowledgeGraphReasoner,
    RealTimePerfusionMonitor,
)

from .feature_extractor import (
    HeartDimensions,
    ValveConditions,
    CoronaryStatus,
    CardiacFunction,
    RiskFactors,
    HeartAnatomyFeatures,
    HeartAnatomyExtractor,
)

from .outcome_predictor import (
    PredictionConfig,
    PerfusionOutcome,
    GNNPathway,
    LLMPathway,
    PerfusionOutcomePredictor,
)

__all__ = [
    # Temporal GNN
    'TemporalGNNConfig',
    'TemporalEncoder',
    'GraphEncoder',
    'TemporalPerfusionGNN',
    'PerfusionDataset',
    'FeatureNormalizer',
    'create_dummy_graph',
    # Monitor
    'RiskLevel',
    'AlertType',
    'AlertThresholds',
    'Alert',
    'PerfusionMeasurement',
    'InterventionRecommendation',
    'MonitoringResult',
    'RuleBasedAlertChecker',
    'KnowledgeGraphReasoner',
    'RealTimePerfusionMonitor',
    # Feature Extractor
    'HeartDimensions',
    'ValveConditions',
    'CoronaryStatus',
    'CardiacFunction',
    'RiskFactors',
    'HeartAnatomyFeatures',
    'HeartAnatomyExtractor',
    # Outcome Predictor
    'PredictionConfig',
    'PerfusionOutcome',
    'GNNPathway',
    'LLMPathway',
    'PerfusionOutcomePredictor',
]

__version__ = '0.1.0'
