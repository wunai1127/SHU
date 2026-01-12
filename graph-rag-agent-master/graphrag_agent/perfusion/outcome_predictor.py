"""
Perfusion Outcome Predictor - Neuro-Symbolic Reasoning System

神经符号推理系统 - 结合知识图谱、GNN和LLM的灌注结果预测器

This module implements the neuro-symbolic approach combining:
1. GNN Pathway: Temporal pattern recognition from perfusion data
2. LLM Pathway: Medical knowledge reasoning from text reports
3. Knowledge Graph: Structured medical knowledge for context

Architecture Overview:
- Input: Heart anatomy text + Perfusion time-series data
- GNN Path: TemporalPerfusionGNN → quality_score, risk_level
- LLM Path: HeartAnatomyExtractor + KG context → reasoned_assessment
- Fusion: Weighted combination with uncertainty estimation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime

# Import sibling modules
from .temporal_gnn import (
    TemporalPerfusionGNN,
    TemporalGNNConfig,
    FeatureNormalizer,
    create_dummy_graph,
)
from .feature_extractor import (
    HeartAnatomyExtractor,
    HeartAnatomyFeatures,
)
from .monitor import (
    RiskLevel,
    PerfusionMeasurement,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionConfig:
    """Configuration for outcome prediction system

    预测系统配置
    """
    # Model paths
    gnn_model_path: Optional[str] = None
    llm_model_name: str = "gpt-4"  # or local model path

    # Fusion weights (sum to 1.0)
    # 融合权重配置
    gnn_weight: float = 0.6  # Weight for GNN pathway
    llm_weight: float = 0.4  # Weight for LLM pathway

    # Confidence thresholds
    min_confidence: float = 0.5
    high_confidence: float = 0.8

    # Knowledge graph settings
    kg_host: str = "localhost"
    kg_port: int = 7687
    kg_user: str = "neo4j"
    kg_password: str = ""

    # Decision thresholds
    # 决策阈值
    usability_threshold: float = 60.0  # Quality score threshold for usable
    risk_threshold: float = 0.7  # Risk probability threshold for alerts

    # Enable/disable pathways
    enable_gnn: bool = True
    enable_llm: bool = True
    enable_kg: bool = True


@dataclass
class PerfusionOutcome:
    """Comprehensive perfusion outcome prediction

    综合灌注结果预测
    """
    # Primary predictions
    # 主要预测结果
    usability_score: float  # 0-100, heart usability for transplant
    risk_level: RiskLevel
    risk_probabilities: Dict[str, float]  # {low, medium, high, critical}

    # Component predictions
    # 各组件预测
    gnn_prediction: Optional[Dict[str, Any]] = None
    llm_prediction: Optional[Dict[str, Any]] = None

    # Confidence and uncertainty
    # 置信度和不确定性
    confidence: float = 0.0
    uncertainty: float = 0.0

    # Reasoning traces
    # 推理轨迹
    gnn_reasoning: List[str] = field(default_factory=list)
    llm_reasoning: List[str] = field(default_factory=list)
    kg_evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Recommendations
    # 建议
    recommendations: List[str] = field(default_factory=list)
    intervention_needed: bool = False

    # Metadata
    timestamp: str = ""
    case_id: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'usability_score': self.usability_score,
            'risk_level': self.risk_level.value if isinstance(self.risk_level, RiskLevel) else self.risk_level,
            'risk_probabilities': self.risk_probabilities,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'gnn_prediction': self.gnn_prediction,
            'llm_prediction': self.llm_prediction,
            'gnn_reasoning': self.gnn_reasoning,
            'llm_reasoning': self.llm_reasoning,
            'kg_evidence': self.kg_evidence,
            'recommendations': self.recommendations,
            'intervention_needed': self.intervention_needed,
            'timestamp': self.timestamp,
            'case_id': self.case_id,
        }


class GNNPathway:
    """
    GNN-based prediction pathway

    基于GNN的预测路径
    - Processes temporal perfusion data
    - Extracts patterns from blood gas trends
    - Provides quantitative risk assessment
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.model: Optional[TemporalPerfusionGNN] = None
        self.normalizer = FeatureNormalizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_model()

    def _load_model(self):
        """Load GNN model from checkpoint"""
        gnn_config = TemporalGNNConfig()
        self.model = TemporalPerfusionGNN(gnn_config)
        self.model.to(self.device)

        if self.config.gnn_model_path:
            try:
                checkpoint = torch.load(
                    self.config.gnn_model_path,
                    map_location=self.device
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'normalizer' in checkpoint:
                    self.normalizer = checkpoint['normalizer']
                logger.info(f"Loaded GNN model from {self.config.gnn_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load GNN model: {e}. Using random weights.")

        self.model.eval()

    def predict(
        self,
        temporal_data: Union[np.ndarray, torch.Tensor],
        graph_context: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run GNN prediction on temporal data

        Args:
            temporal_data: Array of shape [time_steps, features] or [batch, time_steps, features]
                          Features: [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α,
                                    pressure, flow_rate, temperature]
            graph_context: Optional knowledge graph subgraph

        Returns:
            Dict with predictions and reasoning traces
        """
        if not self.config.enable_gnn:
            return {'enabled': False}

        # Prepare input
        # 准备输入数据
        if isinstance(temporal_data, np.ndarray):
            temporal_data = torch.tensor(temporal_data, dtype=torch.float32)

        if temporal_data.dim() == 2:
            temporal_data = temporal_data.unsqueeze(0)  # Add batch dimension

        temporal_data = temporal_data.to(self.device)

        # Use dummy graph if none provided
        # 如果没有提供图上下文，使用虚拟图
        if graph_context is None:
            graph_context = create_dummy_graph()
        graph_context = graph_context.to(self.device)

        # Run inference
        # 运行推理
        with torch.no_grad():
            output = self.model(temporal_data, graph_context)

        # Extract predictions
        quality_score = output['quality_score'].cpu().numpy()[0]
        risk_probs = output['risk_probs'].cpu().numpy()[0]
        intervention_prob = output['intervention_prob'].cpu().numpy()[0]
        next_state = output['next_state'].cpu().numpy()[0]

        # Map risk probabilities
        risk_labels = ['low', 'medium', 'high', 'critical']
        risk_dict = {label: float(prob) for label, prob in zip(risk_labels, risk_probs)}

        # Determine risk level
        # 确定风险等级
        max_risk_idx = np.argmax(risk_probs)
        risk_level = RiskLevel(risk_labels[max_risk_idx])

        # Generate reasoning traces
        # 生成推理轨迹
        reasoning = self._generate_reasoning(
            temporal_data.cpu().numpy()[0],
            quality_score,
            risk_dict,
            next_state
        )

        return {
            'enabled': True,
            'quality_score': float(quality_score),
            'risk_level': risk_level,
            'risk_probabilities': risk_dict,
            'intervention_probability': float(intervention_prob),
            'next_state_prediction': next_state.tolist(),
            'reasoning': reasoning,
            'confidence': float(1 - np.std(risk_probs)),  # Lower variance = higher confidence
        }

    def _generate_reasoning(
        self,
        temporal_data: np.ndarray,
        quality_score: float,
        risk_dict: Dict[str, float],
        next_state: np.ndarray
    ) -> List[str]:
        """Generate interpretable reasoning from GNN predictions

        从GNN预测生成可解释的推理轨迹
        """
        reasoning = []

        # Analyze temporal trends
        # 分析时序趋势
        feature_names = ['pH', 'PO2', 'PCO2', 'lactate', 'K+', 'Na+',
                        'IL-6', 'IL-8', 'TNF-α', 'pressure', 'flow', 'temp']

        # Calculate trends for each feature
        for i, name in enumerate(feature_names):
            if temporal_data.shape[0] >= 2:
                trend = temporal_data[-1, i] - temporal_data[0, i]
                if abs(trend) > 0.1 * abs(temporal_data[0, i] + 1e-8):
                    direction = "increasing" if trend > 0 else "decreasing"
                    reasoning.append(f"[GNN] {name} shows {direction} trend (Δ={trend:.3f})")

        # Quality assessment
        if quality_score >= 80:
            reasoning.append(f"[GNN] High quality score ({quality_score:.1f}) indicates good organ viability")
        elif quality_score >= 60:
            reasoning.append(f"[GNN] Moderate quality score ({quality_score:.1f}) suggests acceptable viability")
        else:
            reasoning.append(f"[GNN] Low quality score ({quality_score:.1f}) indicates potential concerns")

        # Risk assessment
        dominant_risk = max(risk_dict.items(), key=lambda x: x[1])
        reasoning.append(f"[GNN] Primary risk level: {dominant_risk[0]} ({dominant_risk[1]*100:.1f}%)")

        # Predicted trajectory
        if next_state is not None:
            # Check if predicted values are concerning
            # 检查预测值是否令人担忧
            if next_state[0] < 7.2:  # pH
                reasoning.append("[GNN] Predicted pH decline - potential acidosis risk")
            if next_state[3] > 4.0:  # lactate
                reasoning.append("[GNN] Predicted lactate elevation - metabolic stress indicator")

        return reasoning


class LLMPathway:
    """
    LLM-based reasoning pathway

    基于LLM的推理路径
    - Processes text reports and clinical notes
    - Leverages medical knowledge for reasoning
    - Provides qualitative risk assessment
    """

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.extractor = HeartAnatomyExtractor()
        self.llm_client = None  # Will be initialized based on config

        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client

        初始化LLM客户端
        """
        try:
            from openai import OpenAI

            # 从配置文件加载 API 设置
            # Load API settings from config
            try:
                from .training.config import openai_config
                api_key = openai_config.api_key
                base_url = openai_config.base_url
                self.llm_model = openai_config.model
                self.llm_temperature = openai_config.temperature
                self.llm_max_tokens = openai_config.max_tokens
            except ImportError:
                # 使用默认配置
                api_key = "sk-kzBbW2kbljrrOmp6EAgXcQR1F4cxhTMaCJmfyzZeIY8m1fPu"
                base_url = "https://yinli.one/v1"
                self.llm_model = "gpt-4"
                self.llm_temperature = 0.3
                self.llm_max_tokens = 1000

            self.llm_client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            logger.info(f"LLM client initialized with model: {self.llm_model}")
        except ImportError:
            logger.warning("OpenAI package not installed. LLM pathway will use rule-based fallback.")
            self.llm_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize LLM client: {e}. Using rule-based fallback.")
            self.llm_client = None

    def predict(
        self,
        text_report: str,
        kg_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Run LLM-based prediction on text report

        Args:
            text_report: Clinical text report about the donor heart
            kg_context: Related knowledge graph entities and relations

        Returns:
            Dict with predictions and reasoning traces
        """
        if not self.config.enable_llm:
            return {'enabled': False}

        # Extract structured features from text
        # 从文本中提取结构化特征
        features = self.extractor.extract_features(text_report)

        # Build prompt with KG context
        # 构建带有KG上下文的提示
        prompt = self._build_prompt(text_report, features, kg_context)

        # Run LLM inference (placeholder)
        # 运行LLM推理（占位符）
        llm_response = self._run_llm_inference(prompt, features)

        # Parse LLM response
        # 解析LLM响应
        parsed = self._parse_llm_response(llm_response, features)

        return {
            'enabled': True,
            'extracted_features': features.to_dict() if features else {},
            'quality_assessment': parsed.get('quality_assessment', 'unknown'),
            'risk_factors': parsed.get('risk_factors', []),
            'reasoning': parsed.get('reasoning', []),
            'recommendations': parsed.get('recommendations', []),
            'confidence': parsed.get('confidence', 0.5),
        }

    def _build_prompt(
        self,
        text_report: str,
        features: HeartAnatomyFeatures,
        kg_context: Optional[List[Dict]]
    ) -> str:
        """Build LLM prompt with context

        构建带上下文的LLM提示
        """
        prompt_parts = [
            "You are a cardiac transplant specialist evaluating a donor heart.",
            "Analyze the following information and provide a risk assessment.",
            "",
            "## Clinical Report",
            text_report,
            "",
            "## Extracted Features",
        ]

        if features:
            # Add extracted numerical features
            if features.dimensions:
                prompt_parts.append(f"- Heart Dimensions: {features.dimensions}")
            if features.cardiac_function:
                prompt_parts.append(f"- Cardiac Function: {features.cardiac_function}")
            if features.risk_factors:
                prompt_parts.append(f"- Risk Factors: {features.risk_factors}")

        if kg_context:
            prompt_parts.append("")
            prompt_parts.append("## Related Medical Knowledge")
            for item in kg_context[:5]:  # Limit to top 5
                if 'relation' in item:
                    prompt_parts.append(f"- {item['source']} --[{item['relation']}]--> {item['target']}")

        prompt_parts.extend([
            "",
            "## Task",
            "1. Assess the overall quality and usability of this heart for transplant (score 0-100)",
            "2. Identify key risk factors",
            "3. Provide specific recommendations",
            "4. Estimate your confidence level (0-1)",
            "",
            "Respond in JSON format:",
            '{"quality_score": <float>, "risk_level": "<low|medium|high|critical>",',
            '"risk_factors": [<list of factors>], "recommendations": [<list>],',
            '"reasoning": [<step by step reasoning>], "confidence": <float>}'
        ])

        return "\n".join(prompt_parts)

    def _run_llm_inference(
        self,
        prompt: str,
        features: HeartAnatomyFeatures
    ) -> Dict[str, Any]:
        """Run LLM inference

        运行LLM推理
        """
        # 如果有 LLM 客户端，调用真实 API
        # If LLM client is available, call real API
        if self.llm_client is not None:
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a cardiac transplant specialist. Always respond in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens,
                )

                # 解析响应
                response_text = response.choices[0].message.content

                # 尝试提取 JSON
                import re
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    # 确保所有必要字段存在
                    result.setdefault('quality_score', 70.0)
                    result.setdefault('risk_level', 'medium')
                    result.setdefault('risk_factors', [])
                    result.setdefault('recommendations', [])
                    result.setdefault('reasoning', [])
                    result.setdefault('confidence', 0.7)
                    return result
                else:
                    logger.warning("LLM response not in JSON format, using rule-based fallback")
            except Exception as e:
                logger.warning(f"LLM inference failed: {e}, using rule-based fallback")

        # 回退到基于规则的估计
        # Fallback to rule-based estimate
        quality_score = 75.0
        risk_factors = []
        recommendations = []
        reasoning = []

        if features:
            # Adjust based on extracted features
            # 根据提取的特征调整

            # Check ejection fraction
            if features.cardiac_function and features.cardiac_function.ejection_fraction:
                ef = features.cardiac_function.ejection_fraction
                if ef < 40:
                    quality_score -= 30
                    risk_factors.append("Low ejection fraction")
                    reasoning.append(f"[LLM] Ejection fraction {ef}% is below normal (>55%)")
                elif ef < 55:
                    quality_score -= 10
                    reasoning.append(f"[LLM] Ejection fraction {ef}% is mildly reduced")
                else:
                    reasoning.append(f"[LLM] Ejection fraction {ef}% is normal")

            # Check dimensions
            if features.dimensions:
                if features.dimensions.lv_diameter_cm and features.dimensions.lv_diameter_cm > 6.0:
                    quality_score -= 15
                    risk_factors.append("Dilated left ventricle")
                    reasoning.append("[LLM] LV diameter suggests cardiomegaly")

            # Check coronary status
            if features.coronary_status:
                if features.coronary_status.stenosis_percentage:
                    max_stenosis = max(features.coronary_status.stenosis_percentage.values())
                    if max_stenosis > 70:
                        quality_score -= 25
                        risk_factors.append("Significant coronary stenosis")
                        reasoning.append(f"[LLM] Coronary stenosis {max_stenosis}% may limit viability")

            # Check risk factors
            if features.risk_factors:
                if features.risk_factors.has_hypertension:
                    quality_score -= 5
                    risk_factors.append("History of hypertension")
                if features.risk_factors.has_diabetes:
                    quality_score -= 10
                    risk_factors.append("History of diabetes")
                    reasoning.append("[LLM] Diabetic donor hearts have higher rejection risk")
                if features.risk_factors.smoking_history:
                    quality_score -= 5
                    risk_factors.append("Smoking history")

        # Ensure score is in valid range
        quality_score = max(0, min(100, quality_score))

        # Determine risk level
        if quality_score >= 80:
            risk_level = "low"
        elif quality_score >= 60:
            risk_level = "medium"
        elif quality_score >= 40:
            risk_level = "high"
        else:
            risk_level = "critical"

        # Generate recommendations
        if quality_score < 60:
            recommendations.append("Consider extended perfusion evaluation")
        if "Low ejection fraction" in risk_factors:
            recommendations.append("Evaluate contractile reserve with dobutamine stress")
        if "Significant coronary stenosis" in risk_factors:
            recommendations.append("Consider coronary angiography before decision")

        return {
            'quality_score': quality_score,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'reasoning': reasoning,
            'confidence': 0.7 if features else 0.4,
        }

    def _parse_llm_response(
        self,
        response: Dict[str, Any],
        features: HeartAnatomyFeatures
    ) -> Dict[str, Any]:
        """Parse and validate LLM response

        解析和验证LLM响应
        """
        # In production, parse JSON from LLM text response
        # 在生产环境中，从LLM文本响应解析JSON
        return response


class PerfusionOutcomePredictor:
    """
    Main Neuro-Symbolic Predictor combining KG + GNN + LLM

    主神经符号预测器 - 融合知识图谱、GNN和LLM

    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Input Layer                                   │
    │  ┌─────────────────────┐     ┌─────────────────────┐           │
    │  │  Temporal Data      │     │  Text Reports       │           │
    │  │  (Blood gas, etc.)  │     │  (Donor info)       │           │
    │  └─────────────────────┘     └─────────────────────┘           │
    └─────────────────────────────────────────────────────────────────┘
                   │                         │
                   ▼                         ▼
    ┌──────────────────────┐     ┌──────────────────────┐
    │   GNN Pathway        │     │   LLM Pathway        │
    │  ┌────────────────┐  │     │  ┌────────────────┐  │
    │  │ Temporal LSTM  │  │     │  │ Text Extractor │  │
    │  │     ↓          │  │     │  │     ↓          │  │
    │  │ GraphSAGE      │  │     │  │ LLM Reasoner   │  │
    │  │     ↓          │  │     │  │     ↓          │  │
    │  │ Prediction     │  │     │  │ Assessment     │  │
    │  └────────────────┘  │     │  └────────────────┘  │
    └──────────────────────┘     └──────────────────────┘
                   │                         │
                   └────────────┬────────────┘
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Knowledge Graph                               │
    │  ┌─────────────────┐     ┌─────────────────┐                    │
    │  │ Concept Layer   │ ──► │ Instance Layer  │                    │
    │  │ (Medical KB)    │     │ (Case Data)     │                    │
    │  └─────────────────┘     └─────────────────┘                    │
    └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Fusion Layer                                  │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Weighted Combination + Uncertainty Estimation          │    │
    │  │  final = w_gnn * gnn_pred + w_llm * llm_pred           │    │
    │  └─────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        PerfusionOutcome
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        """Initialize the predictor

        初始化预测器

        Args:
            config: Configuration for prediction system
        """
        self.config = config or PredictionConfig()

        # Initialize pathways
        # 初始化预测路径
        self.gnn_pathway = GNNPathway(self.config) if self.config.enable_gnn else None
        self.llm_pathway = LLMPathway(self.config) if self.config.enable_llm else None

        # Initialize KG connection (placeholder)
        # 初始化知识图谱连接（占位符）
        self.kg_client = None
        if self.config.enable_kg:
            self._initialize_kg()

        logger.info(f"PerfusionOutcomePredictor initialized with GNN={self.config.enable_gnn}, "
                   f"LLM={self.config.enable_llm}, KG={self.config.enable_kg}")

    def _initialize_kg(self):
        """Initialize Knowledge Graph connection

        初始化知识图谱连接
        """
        try:
            from neo4j import GraphDatabase

            # 从配置文件加载连接参数
            # Load connection parameters from config
            try:
                from .training.config import neo4j_config
                uri = neo4j_config.uri
                user = neo4j_config.user
                password = neo4j_config.password
                self.kg_database = neo4j_config.database
            except ImportError:
                # 使用默认配置
                uri = "bolt://localhost:7687"
                user = "neo4j"
                password = "wunai1127"
                self.kg_database = "backup"

            self.kg_client = GraphDatabase.driver(
                uri,
                auth=(user, password)
            )
            # 测试连接
            with self.kg_client.session(database=self.kg_database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            logger.info(f"KG connection established to {uri}, database: {self.kg_database}")
        except ImportError:
            logger.warning("neo4j package not installed. KG will use mock data.")
            self.kg_client = None
        except Exception as e:
            logger.warning(f"Failed to connect to KG: {e}. Using mock data.")
            self.kg_client = None

    def query_kg_context(
        self,
        entities: List[str],
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph for relevant context

        查询知识图谱获取相关上下文

        Args:
            entities: List of entity names to query
            max_hops: Maximum number of hops for relation paths

        Returns:
            List of related triplets from KG
        """
        if not self.kg_client:
            return self._get_mock_kg_context(entities)

        try:
            # 实际查询 Neo4j
            # Query Neo4j for related entities
            query = '''
            MATCH (e)-[r]-(related)
            WHERE e.name IN $entities OR e.normalized_name IN $entities
            RETURN e.name as source, type(r) as relation, related.name as target
            LIMIT 30
            '''

            results = []
            with self.kg_client.session(database=self.kg_database) as session:
                result = session.run(query, entities=entities)
                for record in result:
                    results.append({
                        'source': record['source'],
                        'relation': record['relation'],
                        'target': record['target'],
                    })

            if results:
                logger.info(f"Found {len(results)} KG relations for entities: {entities[:3]}...")
                return results
            else:
                # 如果没有直接匹配，尝试模糊匹配
                # If no direct match, try fuzzy matching
                fuzzy_query = '''
                MATCH (e)-[r]-(related)
                WHERE any(ent IN $entities WHERE e.name CONTAINS ent OR e.normalized_name CONTAINS ent)
                RETURN e.name as source, type(r) as relation, related.name as target
                LIMIT 20
                '''
                result = session.run(fuzzy_query, entities=entities)
                for record in result:
                    results.append({
                        'source': record['source'],
                        'relation': record['relation'],
                        'target': record['target'],
                    })

            return results if results else self._get_mock_kg_context(entities)

        except Exception as e:
            logger.warning(f"KG query failed: {e}, using mock data")
            return self._get_mock_kg_context(entities)

    def _get_mock_kg_context(self, entities: List[str]) -> List[Dict[str, Any]]:
        """Get mock KG context for demonstration

        获取用于演示的模拟KG上下文
        """
        # Mock knowledge graph relations
        # 模拟的知识图谱关系
        mock_relations = [
            {"source": "normothermic machine perfusion", "relation": "PRESERVES", "target": "donor heart"},
            {"source": "lactate elevation", "relation": "INDICATES", "target": "metabolic stress"},
            {"source": "metabolic stress", "relation": "INCREASES_RISK_OF", "target": "primary graft dysfunction"},
            {"source": "ejection fraction", "relation": "MEASURES", "target": "cardiac function"},
            {"source": "low ejection fraction", "relation": "ASSOCIATED_WITH", "target": "heart failure"},
            {"source": "coronary artery disease", "relation": "AFFECTS", "target": "myocardial perfusion"},
            {"source": "IL-6", "relation": "MARKER_OF", "target": "inflammatory response"},
            {"source": "inflammatory response", "relation": "INCREASES_RISK_OF", "target": "rejection"},
            {"source": "hypothermia", "relation": "REDUCES", "target": "metabolic rate"},
            {"source": "reperfusion injury", "relation": "CAUSES", "target": "oxidative stress"},
        ]

        # Filter relevant relations
        relevant = []
        for rel in mock_relations:
            for entity in entities:
                if (entity.lower() in rel['source'].lower() or
                    entity.lower() in rel['target'].lower()):
                    relevant.append(rel)
                    break

        return relevant[:10]  # Limit to 10 relations

    def predict(
        self,
        temporal_data: Optional[Union[np.ndarray, torch.Tensor, List[PerfusionMeasurement]]] = None,
        text_report: Optional[str] = None,
        case_id: str = "",
    ) -> PerfusionOutcome:
        """
        Run full prediction pipeline

        运行完整的预测流水线

        Args:
            temporal_data: Time-series perfusion measurements
            text_report: Clinical text report about donor heart
            case_id: Case identifier

        Returns:
            PerfusionOutcome with comprehensive predictions
        """
        timestamp = datetime.now().isoformat()

        # Convert measurements to array if needed
        # 如果需要，将测量值转换为数组
        if temporal_data is not None and isinstance(temporal_data, list):
            if isinstance(temporal_data[0], PerfusionMeasurement):
                temporal_data = self._measurements_to_array(temporal_data)

        # Initialize result
        # 初始化结果
        gnn_result = {'enabled': False}
        llm_result = {'enabled': False}
        kg_evidence = []

        # 1. Run GNN pathway
        # 运行GNN路径
        if temporal_data is not None and self.gnn_pathway:
            try:
                gnn_result = self.gnn_pathway.predict(temporal_data)
            except Exception as e:
                logger.error(f"GNN pathway error: {e}")
                gnn_result = {'enabled': False, 'error': str(e)}

        # 2. Query KG for context
        # 查询KG获取上下文
        if text_report and self.config.enable_kg:
            # Extract key entities from text for KG query
            entities = self._extract_entities_for_kg(text_report)
            kg_evidence = self.query_kg_context(entities)

        # 3. Run LLM pathway
        # 运行LLM路径
        if text_report and self.llm_pathway:
            try:
                llm_result = self.llm_pathway.predict(text_report, kg_evidence)
            except Exception as e:
                logger.error(f"LLM pathway error: {e}")
                llm_result = {'enabled': False, 'error': str(e)}

        # 4. Fuse predictions
        # 融合预测
        fused_result = self._fuse_predictions(gnn_result, llm_result)

        # 5. Generate final outcome
        # 生成最终结果
        outcome = PerfusionOutcome(
            usability_score=fused_result['quality_score'],
            risk_level=fused_result['risk_level'],
            risk_probabilities=fused_result['risk_probabilities'],
            gnn_prediction=gnn_result if gnn_result.get('enabled') else None,
            llm_prediction=llm_result if llm_result.get('enabled') else None,
            confidence=fused_result['confidence'],
            uncertainty=fused_result['uncertainty'],
            gnn_reasoning=gnn_result.get('reasoning', []),
            llm_reasoning=llm_result.get('reasoning', []),
            kg_evidence=kg_evidence,
            recommendations=fused_result['recommendations'],
            intervention_needed=fused_result['intervention_needed'],
            timestamp=timestamp,
            case_id=case_id,
        )

        return outcome

    def _measurements_to_array(
        self,
        measurements: List[PerfusionMeasurement]
    ) -> np.ndarray:
        """Convert PerfusionMeasurement list to numpy array

        将PerfusionMeasurement列表转换为numpy数组
        """
        feature_order = ['pH', 'PO2', 'PCO2', 'lactate', 'K_plus', 'Na_plus',
                        'IL_6', 'IL_8', 'TNF_alpha', 'pressure', 'flow_rate', 'temperature']

        data = []
        for m in measurements:
            row = [
                m.pH,
                m.PO2,
                m.PCO2,
                m.lactate,
                m.K_plus,
                m.Na_plus,
                m.IL_6 if m.IL_6 else 0,
                m.IL_8 if m.IL_8 else 0,
                m.TNF_alpha if m.TNF_alpha else 0,
                m.pressure,
                m.flow_rate,
                m.temperature,
            ]
            data.append(row)

        return np.array(data, dtype=np.float32)

    def _extract_entities_for_kg(self, text: str) -> List[str]:
        """Extract entity names from text for KG query

        从文本中提取实体名称用于KG查询
        """
        # Simple keyword extraction
        # 简单的关键词提取
        keywords = [
            'perfusion', 'heart', 'ejection fraction', 'coronary',
            'lactate', 'inflammatory', 'rejection', 'stenosis',
            'dysfunction', 'ischemia', 'reperfusion'
        ]

        found = []
        text_lower = text.lower()
        for kw in keywords:
            if kw in text_lower:
                found.append(kw)

        return found

    def _fuse_predictions(
        self,
        gnn_result: Dict[str, Any],
        llm_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fuse predictions from GNN and LLM pathways

        融合GNN和LLM路径的预测

        Uses weighted combination with uncertainty estimation
        """
        gnn_enabled = gnn_result.get('enabled', False)
        llm_enabled = llm_result.get('enabled', False)

        # Calculate effective weights
        # 计算有效权重
        if gnn_enabled and llm_enabled:
            gnn_weight = self.config.gnn_weight
            llm_weight = self.config.llm_weight
        elif gnn_enabled:
            gnn_weight = 1.0
            llm_weight = 0.0
        elif llm_enabled:
            gnn_weight = 0.0
            llm_weight = 1.0
        else:
            # No predictions available
            return {
                'quality_score': 50.0,
                'risk_level': RiskLevel.MEDIUM,
                'risk_probabilities': {'low': 0.25, 'medium': 0.5, 'high': 0.2, 'critical': 0.05},
                'confidence': 0.1,
                'uncertainty': 0.9,
                'recommendations': ["Insufficient data for prediction"],
                'intervention_needed': False,
            }

        # Fuse quality scores
        # 融合质量分数
        gnn_quality = gnn_result.get('quality_score', 50.0) if gnn_enabled else 50.0
        llm_quality = llm_result.get('quality_score', 50.0) if llm_enabled else 50.0
        fused_quality = gnn_weight * gnn_quality + llm_weight * llm_quality

        # Fuse risk probabilities
        # 融合风险概率
        default_risk = {'low': 0.25, 'medium': 0.5, 'high': 0.2, 'critical': 0.05}
        gnn_risk = gnn_result.get('risk_probabilities', default_risk) if gnn_enabled else default_risk

        # Convert LLM risk level to probabilities if needed
        if llm_enabled:
            llm_risk_level = llm_result.get('risk_level', 'medium')
            if isinstance(llm_risk_level, str):
                llm_risk = self._risk_level_to_probs(llm_risk_level)
            else:
                llm_risk = default_risk
        else:
            llm_risk = default_risk

        fused_risk = {}
        for key in ['low', 'medium', 'high', 'critical']:
            fused_risk[key] = gnn_weight * gnn_risk.get(key, 0.25) + llm_weight * llm_risk.get(key, 0.25)

        # Normalize
        total = sum(fused_risk.values())
        fused_risk = {k: v/total for k, v in fused_risk.items()}

        # Determine final risk level
        # 确定最终风险等级
        max_risk = max(fused_risk.items(), key=lambda x: x[1])
        risk_level = RiskLevel(max_risk[0])

        # Calculate confidence and uncertainty
        # 计算置信度和不确定性
        gnn_conf = gnn_result.get('confidence', 0.5) if gnn_enabled else 0.0
        llm_conf = llm_result.get('confidence', 0.5) if llm_enabled else 0.0

        # Agreement between pathways increases confidence
        # 路径之间的一致性提高置信度
        if gnn_enabled and llm_enabled:
            score_diff = abs(gnn_quality - llm_quality)
            agreement_factor = max(0, 1 - score_diff / 50)  # 0-1
            confidence = (gnn_weight * gnn_conf + llm_weight * llm_conf) * (0.7 + 0.3 * agreement_factor)
        else:
            confidence = gnn_weight * gnn_conf + llm_weight * llm_conf

        confidence = min(1.0, confidence)
        uncertainty = 1.0 - confidence

        # Merge recommendations
        # 合并建议
        recommendations = []
        if gnn_enabled and 'recommendations' in gnn_result:
            recommendations.extend(gnn_result['recommendations'])
        if llm_enabled and 'recommendations' in llm_result:
            recommendations.extend(llm_result['recommendations'])
        recommendations = list(set(recommendations))  # Remove duplicates

        # Determine if intervention needed
        # 确定是否需要干预
        intervention_needed = (
            fused_quality < self.config.usability_threshold or
            fused_risk.get('critical', 0) > 0.2 or
            fused_risk.get('high', 0) > 0.4 or
            (gnn_enabled and gnn_result.get('intervention_probability', 0) > 0.5)
        )

        if intervention_needed:
            recommendations.append("Clinical team review recommended")

        return {
            'quality_score': fused_quality,
            'risk_level': risk_level,
            'risk_probabilities': fused_risk,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'recommendations': recommendations,
            'intervention_needed': intervention_needed,
        }

    def _risk_level_to_probs(self, level: str) -> Dict[str, float]:
        """Convert risk level string to probability distribution

        将风险等级字符串转换为概率分布
        """
        distributions = {
            'low': {'low': 0.7, 'medium': 0.2, 'high': 0.08, 'critical': 0.02},
            'medium': {'low': 0.15, 'medium': 0.6, 'high': 0.2, 'critical': 0.05},
            'high': {'low': 0.05, 'medium': 0.15, 'high': 0.6, 'critical': 0.2},
            'critical': {'low': 0.02, 'medium': 0.08, 'high': 0.3, 'critical': 0.6},
        }
        return distributions.get(level.lower(), distributions['medium'])

    def predict_batch(
        self,
        cases: List[Dict[str, Any]]
    ) -> List[PerfusionOutcome]:
        """Predict outcomes for multiple cases

        预测多个病例的结果

        Args:
            cases: List of dicts with 'temporal_data', 'text_report', 'case_id' keys

        Returns:
            List of PerfusionOutcome objects
        """
        results = []
        for case in cases:
            outcome = self.predict(
                temporal_data=case.get('temporal_data'),
                text_report=case.get('text_report'),
                case_id=case.get('case_id', ''),
            )
            results.append(outcome)

        return results


def demo():
    """Demonstrate the prediction system

    演示预测系统
    """
    print("=" * 60)
    print("Perfusion Outcome Predictor Demo")
    print("=" * 60)

    # Initialize predictor
    config = PredictionConfig(
        enable_gnn=True,
        enable_llm=True,
        enable_kg=True,
        gnn_weight=0.6,
        llm_weight=0.4,
    )
    predictor = PerfusionOutcomePredictor(config)

    # Create sample temporal data (10 time steps, 12 features)
    # 创建示例时序数据
    np.random.seed(42)
    temporal_data = np.array([
        # [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α, pressure, flow, temp]
        [7.40, 400, 40, 1.5, 4.0, 140, 10, 8, 5, 60, 1.5, 34],
        [7.38, 380, 42, 1.8, 4.1, 139, 12, 10, 6, 58, 1.5, 34],
        [7.36, 360, 44, 2.2, 4.2, 138, 15, 12, 8, 56, 1.4, 35],
        [7.34, 340, 46, 2.8, 4.3, 137, 20, 15, 10, 55, 1.4, 35],
        [7.32, 320, 48, 3.2, 4.4, 136, 25, 18, 12, 54, 1.3, 36],
        [7.30, 300, 50, 3.8, 4.5, 135, 30, 22, 15, 52, 1.3, 36],
        [7.28, 280, 52, 4.2, 4.6, 134, 35, 25, 18, 50, 1.2, 37],
        [7.26, 260, 54, 4.5, 4.7, 133, 40, 28, 20, 48, 1.2, 37],
        [7.25, 250, 55, 4.8, 4.8, 132, 42, 30, 22, 46, 1.1, 37],
        [7.24, 240, 56, 5.0, 4.9, 131, 45, 32, 24, 45, 1.1, 37],
    ], dtype=np.float32)

    # Sample text report
    text_report = """
    Donor Information:
    - 45-year-old male donor
    - History of hypertension, controlled with medication
    - Ejection fraction: 55%
    - No significant coronary artery disease
    - Left ventricle diameter: 5.2cm (normal)
    - Aortic diameter: 3.0cm

    During normothermic machine perfusion:
    - Initial lactate: 1.5 mmol/L, rising trend observed
    - pH showing gradual decline
    - Inflammatory markers (IL-6) elevated

    Assessment: Marginal donor with metabolic stress indicators during perfusion.
    """

    # Run prediction
    print("\nRunning prediction...")
    outcome = predictor.predict(
        temporal_data=temporal_data,
        text_report=text_report,
        case_id="DEMO-001"
    )

    # Display results
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)

    print(f"\nUsability Score: {outcome.usability_score:.1f}/100")
    print(f"Risk Level: {outcome.risk_level.value}")
    print(f"Confidence: {outcome.confidence:.2f}")
    print(f"Intervention Needed: {outcome.intervention_needed}")

    print("\nRisk Probabilities:")
    for level, prob in outcome.risk_probabilities.items():
        bar = "█" * int(prob * 20)
        print(f"  {level:8s}: {prob:.2%} {bar}")

    print("\n--- GNN Pathway Reasoning ---")
    for r in outcome.gnn_reasoning:
        print(f"  • {r}")

    print("\n--- LLM Pathway Reasoning ---")
    for r in outcome.llm_reasoning:
        print(f"  • {r}")

    print("\n--- Knowledge Graph Evidence ---")
    for e in outcome.kg_evidence[:5]:
        print(f"  • {e['source']} --[{e['relation']}]--> {e['target']}")

    print("\n--- Recommendations ---")
    for r in outcome.recommendations:
        print(f"  ★ {r}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
