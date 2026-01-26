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
            'target_max': 2.0,
            'unit': 'mmol/L'
        },
        'SvO2': {
            'low': ['tissue hypoxia', 'low SvO2', 'oxygen extraction', 'low cardiac output'],
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

        critical_indicators = {'MAP', 'CI', 'K_A'}
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
