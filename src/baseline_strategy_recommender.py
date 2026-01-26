#!/usr/bin/env python3
"""
Baseline Strategy Recommender - 完整的Baseline策略推荐系统

功能：
1. 加载YAML配置中的阈值和干预策略
2. 与Neo4j知识图谱对齐的证据查询
3. LLM集成接口（支持多种LLM后端）
4. 强约束CoT推理，生成可溯源建议
5. Robustness检查和一致性验证
"""

import os
import sys
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据结构定义
# =============================================================================

class Severity(Enum):
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    NORMAL = "normal"


@dataclass
class KGTriple:
    """知识图谱三元组"""
    subject: str
    predicate: str
    object: str
    source: str = "config"
    confidence: float = 1.0


@dataclass
class Evidence:
    """证据"""
    statement: str
    source: str
    triple: Optional[KGTriple] = None
    confidence: float = 0.8


@dataclass
class InterventionAction:
    """干预动作"""
    action: str
    drug: Optional[str] = None
    dose: Optional[str] = None
    target: Optional[str] = None
    rationale: Optional[str] = None
    source: Optional[str] = None


@dataclass
class StrategyRecommendation:
    """策略推荐"""
    indicator: str
    current_value: float
    target_value: float
    target_range: Tuple[float, float]
    unit: str

    severity: Severity
    condition: str

    immediate_actions: List[InterventionAction]
    followup_actions: List[InterventionAction]
    escalation_trigger: Optional[str]

    reasoning_chain: List[str]
    evidence: List[Evidence]
    kg_triples: List[KGTriple]

    monitoring: List[str]
    warnings: List[str]
    confidence: float


@dataclass
class BaselineReport:
    """Baseline报告"""
    sample_id: str
    timestamp_min: int
    measurements: Dict[str, float]

    abnormalities: List[Dict]
    recommendations: List[StrategyRecommendation]

    risk_level: str
    summary: str

    # Robustness检查结果
    consistency_check: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# LLM接口抽象
# =============================================================================

class LLMInterface(ABC):
    """LLM接口抽象基类"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查是否可用"""
        pass


class DummyLLM(LLMInterface):
    """占位LLM（用于无LLM时）"""

    def generate(self, prompt: str, **kwargs) -> str:
        return "LLM未配置"

    def is_available(self) -> bool:
        return False


class OpenAILLM(LLMInterface):
    """OpenAI LLM接口"""

    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("openai package not installed")
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        if client is None:
            return ""

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.1),
            max_tokens=kwargs.get("max_tokens", 2000)
        )
        return response.choices[0].message.content

    def is_available(self) -> bool:
        return self.api_key is not None and self._get_client() is not None


class AnthropicLLM(LLMInterface):
    """Anthropic Claude接口"""

    def __init__(self, api_key: str = None, model: str = "claude-3-sonnet-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic package not installed")
        return self._client

    def generate(self, prompt: str, **kwargs) -> str:
        client = self._get_client()
        if client is None:
            return ""

        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", 2000),
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def is_available(self) -> bool:
        return self.api_key is not None


# =============================================================================
# 配置加载器
# =============================================================================

class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        self.thresholds = self._load_yaml("thresholds.yaml")
        self.interventions = self._load_yaml("intervention_strategies.yaml")
        self.strategies = self._load_yaml("strategies.yaml")
        self.baseline = self._load_yaml("baseline.yaml")

        # 构建索引
        self._build_indices()

    def _load_yaml(self, filename: str) -> Dict:
        filepath = self.config_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _build_indices(self):
        """构建指标索引"""
        self.indicator_index = {}
        self.intervention_index = {}
        self.kg_triple_index = {}

        # 从thresholds构建指标索引
        for category, indicators in self.thresholds.items():
            if isinstance(indicators, dict) and category not in ['version', 'last_updated', 'update_notes']:
                for ind_key, ind_config in indicators.items():
                    if isinstance(ind_config, dict):
                        self.indicator_index[ind_key] = {
                            'category': category,
                            'config': ind_config
                        }

        # 从interventions构建干预索引和KG三元组索引
        for category, indicators in self.interventions.items():
            if isinstance(indicators, dict) and category not in ['version', 'last_updated', 'authoritative_sources']:
                for ind_key, ind_config in indicators.items():
                    if isinstance(ind_config, dict) and 'abnormalities' in ind_config:
                        self.intervention_index[ind_key] = ind_config

                        # 提取KG三元组
                        for abn_key, abn_config in ind_config.get('abnormalities', {}).items():
                            if 'kg_triples' in abn_config:
                                key = f"{ind_key}_{abn_key}"
                                self.kg_triple_index[key] = [
                                    KGTriple(t[0], t[1], t[2], source="intervention_strategies.yaml")
                                    for t in abn_config['kg_triples']
                                ]

    def get_threshold_config(self, indicator: str) -> Optional[Dict]:
        """获取指标阈值配置"""
        if indicator in self.indicator_index:
            return self.indicator_index[indicator]['config']
        return None

    def get_intervention_config(self, indicator: str) -> Optional[Dict]:
        """获取指标干预配置"""
        return self.intervention_index.get(indicator)

    def get_kg_triples(self, indicator: str, abnormality: str) -> List[KGTriple]:
        """获取指标异常对应的KG三元组"""
        key = f"{indicator}_{abnormality}"
        return self.kg_triple_index.get(key, [])

    def get_baseline_reference(self, indicator: str) -> Optional[float]:
        """获取指标的baseline参考值"""
        config = self.get_threshold_config(indicator)
        if config:
            return config.get('baseline_reference')

        # 从baseline.yaml查找
        for section in ['confirmed_baselines', 'pending_baselines']:
            if section in self.baseline and indicator in self.baseline[section]:
                return self.baseline[section][indicator].get('baseline_value')
        return None


# =============================================================================
# 策略生成器
# =============================================================================

class StrategyGenerator:
    """策略生成器"""

    # LLM提示词模板
    STRATEGY_PROMPT = """你是心脏移植灌注监测的临床决策支持专家。基于以下信息，生成精确的干预策略。

## 强制约束
1. 必须引用提供的证据，格式：[Evidence-N]
2. 必须给出精确数值目标（如MAP→70 mmHg）
3. 必须按5步CoT推理：观察→病理分析→机制关联→干预选择→预期效果
4. 必须说明监测频率

## 当前状态
- 指标: {indicator} ({indicator_name})
- 当前值: {current_value} {unit}
- 目标范围: {target_min} - {target_max} {unit}
- 异常类型: {abnormality}
- 严重程度: {severity}

## 配置的干预策略
{intervention_config}

## 知识图谱证据
{kg_evidence}

## 输出格式 (严格JSON)
{{
  "reasoning_chain": [
    "Step 1 - 观察: ...",
    "Step 2 - 病理生理分析: ...",
    "Step 3 - 机制关联: [Evidence-X] ...",
    "Step 4 - 干预选择: ...",
    "Step 5 - 预期效果: ..."
  ],
  "target_value": 数值,
  "immediate_actions": [
    {{"action": "...", "drug": "...", "dose": "...", "rationale": "..."}}
  ],
  "monitoring": ["指标: 频率", ...],
  "warnings": ["注意事项", ...],
  "confidence": 0.0-1.0
}}
"""

    def __init__(self, config_loader: ConfigLoader, llm: LLMInterface = None, neo4j_connector=None):
        self.config = config_loader
        self.llm = llm or DummyLLM()
        self.neo4j = neo4j_connector

    def set_neo4j(self, connector):
        """设置Neo4j连接器"""
        self.neo4j = connector

    def set_llm(self, llm: LLMInterface):
        """设置LLM"""
        self.llm = llm

    def analyze_indicator(self, indicator: str, value: float) -> Optional[Dict]:
        """分析单个指标状态"""
        threshold_config = self.config.get_threshold_config(indicator)
        if not threshold_config:
            return None

        thresholds = threshold_config.get('thresholds', {})
        unit = threshold_config.get('unit', '')

        # 确定异常类型和严重程度
        abnormality = None
        severity = Severity.NORMAL
        condition = ""

        # 检查critical
        if 'critical' in thresholds:
            crit = thresholds['critical']
            if self._check_condition(value, crit):
                abnormality = "critical"
                severity = Severity.CRITICAL
                condition = crit.get('description', '')

        # 检查red_line
        if abnormality is None:
            for key in ['red_line', 'red_line_low', 'red_line_high', 'reject']:
                if key in thresholds:
                    rl = thresholds[key]
                    if self._check_condition(value, rl):
                        abnormality = key
                        severity = Severity.CRITICAL
                        condition = rl.get('description', '')
                        break

        # 检查warning
        if abnormality is None:
            for key in ['warning', 'warning_low', 'warning_high', 'warning_range']:
                if key in thresholds:
                    warn = thresholds[key]
                    if self._check_condition(value, warn):
                        abnormality = key
                        severity = Severity.WARNING
                        condition = warn.get('description', '')
                        break

        if abnormality is None:
            return None

        # 确定方向
        direction = self._determine_direction(indicator, value, thresholds)

        return {
            'indicator': indicator,
            'value': value,
            'unit': unit,
            'abnormality': abnormality,
            'direction': direction,
            'severity': severity,
            'condition': condition,
            'threshold_config': threshold_config
        }

    def _check_condition(self, value: float, threshold_config: Dict) -> bool:
        """检查值是否满足阈值条件"""
        if 'operator' in threshold_config and 'value' in threshold_config:
            op = threshold_config['operator']
            th_value = threshold_config['value']

            ops = {
                '<': lambda v, t: v < t,
                '<=': lambda v, t: v <= t,
                '>': lambda v, t: v > t,
                '>=': lambda v, t: v >= t,
                '==': lambda v, t: v == t,
            }
            return ops.get(op, lambda v, t: False)(value, th_value)

        if 'min' in threshold_config and 'max' in threshold_config:
            return threshold_config['min'] <= value <= threshold_config['max']
        elif 'min' in threshold_config:
            return value >= threshold_config['min']
        elif 'max' in threshold_config:
            return value <= threshold_config['max']

        return False

    def _determine_direction(self, indicator: str, value: float, thresholds: Dict) -> str:
        """确定异常方向"""
        target = thresholds.get('target', {})
        baseline = self.config.get_baseline_reference(indicator)

        if 'min' in target and value < target['min']:
            return 'low'
        elif 'max' in target and value > target['max']:
            return 'high'
        elif baseline is not None:
            return 'high' if value > baseline else 'low'

        return 'abnormal'

    def generate_strategy(self, analysis: Dict) -> StrategyRecommendation:
        """生成策略推荐"""
        indicator = analysis['indicator']
        value = analysis['value']
        abnormality = analysis['abnormality']
        direction = analysis['direction']
        severity = analysis['severity']

        # 获取干预配置
        intervention_config = self.config.get_intervention_config(indicator)

        # 获取KG三元组
        kg_triples = self.config.get_kg_triples(indicator, abnormality)
        if not kg_triples:
            kg_triples = self.config.get_kg_triples(indicator, f"{direction}_{indicator.lower()}")

        # 查询Neo4j证据
        neo4j_evidence = self._query_neo4j_evidence(indicator, direction) if self.neo4j else []

        # 获取目标范围
        threshold_config = analysis['threshold_config']
        thresholds = threshold_config.get('thresholds', {})
        target = thresholds.get('target', {})
        target_min = target.get('min', value * 0.8)
        target_max = target.get('max', value * 1.2)
        target_value = (target_min + target_max) / 2

        # 提取配置的干预动作
        immediate_actions = []
        followup_actions = []
        escalation_trigger = None

        if intervention_config and 'abnormalities' in intervention_config:
            for abn_key, abn_config in intervention_config['abnormalities'].items():
                if direction in abn_key or abnormality in abn_key:
                    strategies = abn_config.get('strategies', {})

                    # immediate actions
                    for action in strategies.get('immediate', []):
                        immediate_actions.append(InterventionAction(
                            action=action.get('action', ''),
                            drug=action.get('drug') or (action.get('drugs', [None])[0] if 'drugs' in action else None),
                            dose=action.get('dose'),
                            target=action.get('target'),
                            rationale=action.get('rationale'),
                            source=action.get('source')
                        ))

                    # followup actions
                    for action in strategies.get('if_no_improvement', []):
                        followup_actions.append(InterventionAction(
                            action=action.get('action', ''),
                            drug=action.get('drug') or (action.get('drugs', [None])[0] if 'drugs' in action else None),
                            dose=action.get('dose'),
                            rationale=action.get('rationale'),
                            source=action.get('source')
                        ))

                    escalation_trigger = abn_config.get('escalation_trigger')
                    break

        # 构建推理链
        reasoning_chain = self._build_reasoning_chain(
            indicator, value, direction, severity,
            immediate_actions, kg_triples, neo4j_evidence
        )

        # 构建证据列表
        evidence = []
        for triple in kg_triples:
            evidence.append(Evidence(
                statement=f"{triple.subject} {triple.predicate} {triple.object}",
                source=triple.source,
                triple=triple
            ))
        evidence.extend(neo4j_evidence)

        # 监测点
        monitoring = self._get_monitoring_points(indicator, severity)

        # 警告
        warnings = self._get_warnings(indicator, direction, severity)

        # 如果有LLM，使用LLM增强
        if self.llm.is_available():
            enhanced = self._enhance_with_llm(
                indicator, value, analysis, intervention_config,
                kg_triples, neo4j_evidence
            )
            if enhanced:
                reasoning_chain = enhanced.get('reasoning_chain', reasoning_chain)
                monitoring = enhanced.get('monitoring', monitoring)
                warnings = enhanced.get('warnings', warnings)

        return StrategyRecommendation(
            indicator=indicator,
            current_value=value,
            target_value=target_value,
            target_range=(target_min, target_max),
            unit=analysis['unit'],
            severity=severity,
            condition=analysis['condition'],
            immediate_actions=immediate_actions,
            followup_actions=followup_actions,
            escalation_trigger=escalation_trigger,
            reasoning_chain=reasoning_chain,
            evidence=evidence,
            kg_triples=kg_triples,
            monitoring=monitoring,
            warnings=warnings,
            confidence=0.85 if immediate_actions else 0.6
        )

    def _query_neo4j_evidence(self, indicator: str, direction: str) -> List[Evidence]:
        """查询Neo4j获取证据"""
        if not self.neo4j:
            return []

        evidence = []
        keywords = self._get_search_keywords(indicator, direction)

        for keyword in keywords[:2]:
            try:
                result = self.neo4j.query_decision_support(keyword)

                for cause in result.get('causes', [])[:3]:
                    evidence.append(Evidence(
                        statement=f"{cause.get('cause', cause)} 可导致 {keyword}",
                        source="neo4j_kg",
                        confidence=0.7
                    ))

                for treatment in result.get('treatments', [])[:3]:
                    evidence.append(Evidence(
                        statement=f"{treatment.get('treatment', treatment)} 可治疗 {keyword}",
                        source="neo4j_kg",
                        confidence=0.7
                    ))
            except Exception as e:
                logger.error(f"Neo4j查询失败: {e}")

        return evidence

    def _get_search_keywords(self, indicator: str, direction: str) -> List[str]:
        """获取搜索关键词"""
        keyword_map = {
            ('MAP', 'low'): ['hypotension', 'low blood pressure'],
            ('Lactate', 'high'): ['lactic acidosis', 'hyperlactatemia'],
            ('K_A', 'high'): ['hyperkalemia', 'arrhythmia'],
            ('K_A', 'low'): ['hypokalemia'],
            ('SvO2', 'low'): ['tissue hypoxia', 'low cardiac output'],
            ('SvO2', 'high'): ['sepsis', 'shunting'],
            ('pH', 'low'): ['acidosis'],
            ('pH', 'high'): ['alkalosis'],
            ('CI', 'low'): ['low cardiac output', 'cardiogenic shock'],
        }
        return keyword_map.get((indicator, direction), [indicator])

    def _build_reasoning_chain(self, indicator: str, value: float, direction: str,
                               severity: Severity, actions: List[InterventionAction],
                               kg_triples: List[KGTriple], neo4j_evidence: List[Evidence]) -> List[str]:
        """构建推理链"""
        chain = []

        # Step 1 - 观察
        chain.append(f"Step 1 - 观察: {indicator} = {value}，{direction}，严重程度: {severity.value}")

        # Step 2 - 病理生理分析
        physio_map = {
            'MAP': '低MAP导致组织灌注不足，器官功能受损',
            'Lactate': '乳酸升高反映组织缺氧或无氧代谢增加',
            'K_A': '钾异常影响心肌电活动，可致心律失常',
            'SvO2': 'SvO2反映全身氧供需平衡状态',
            'pH': '酸碱失衡影响酶活性和器官功能',
            'CI': '心输出量不足导致器官灌注下降',
        }
        chain.append(f"Step 2 - 病理生理分析: {physio_map.get(indicator, '需进一步评估')}")

        # Step 3 - 机制关联
        if kg_triples:
            triple = kg_triples[0]
            chain.append(f"Step 3 - 机制关联: [Evidence-1] {triple.subject} --{triple.predicate}--> {triple.object}")
        elif neo4j_evidence:
            chain.append(f"Step 3 - 机制关联: [KG-Evidence] {neo4j_evidence[0].statement}")
        else:
            chain.append("Step 3 - 机制关联: 基于临床指南推荐")

        # Step 4 - 干预选择
        if actions:
            action = actions[0]
            chain.append(f"Step 4 - 干预选择: {action.action}" +
                        (f"，药物: {action.drug}" if action.drug else ""))
        else:
            chain.append("Step 4 - 干预选择: 评估后针对性干预")

        # Step 5 - 预期效果
        chain.append(f"Step 5 - 预期效果: {indicator}恢复至目标范围")

        return chain

    def _get_monitoring_points(self, indicator: str, severity: Severity) -> List[str]:
        """获取监测要点"""
        base_monitoring = {
            'MAP': ['MAP: 每5分钟', 'HR: 每5分钟', 'Lactate: 每30分钟'],
            'Lactate': ['Lactate: 每1-2小时', 'MAP: 每15分钟', '乳酸清除率: 评估'],
            'K_A': ['K+: 每2小时', 'ECG: 持续监测', 'Mg2+: 同时检测'],
            'SvO2': ['SvO2: 持续监测', 'CI: 每小时', 'Lactate: 每2小时'],
            'pH': ['血气分析: 每1-2小时', 'Lactate: 每1小时', 'K+: 每2小时'],
            'CI': ['CI: 每小时', 'MAP: 每15分钟', 'SvO2: 每30分钟'],
        }

        monitoring = base_monitoring.get(indicator, [f'{indicator}: 定期监测'])

        if severity == Severity.CRITICAL:
            monitoring = [m.replace('每', '每').replace('小时', '30分钟')
                         if '小时' in m else m for m in monitoring]

        return monitoring

    def _get_warnings(self, indicator: str, direction: str, severity: Severity) -> List[str]:
        """获取警告"""
        warnings_map = {
            ('MAP', 'low'): ['注意容量状态', '高剂量升压药可致心律失常'],
            ('K_A', 'high'): ['可致心脏骤停', '严重者需透析'],
            ('K_A', 'low'): ['补钾速度不超过20 mEq/h', '需同时纠正低镁'],
            ('Lactate', 'high'): ['需排除肝功能障碍', '持续升高提示预后不良'],
            ('pH', 'low'): ['补碱可能加重细胞内酸中毒', '优先纠正病因'],
        }
        return warnings_map.get((indicator, direction), [])

    def _enhance_with_llm(self, indicator: str, value: float, analysis: Dict,
                          intervention_config: Dict, kg_triples: List[KGTriple],
                          neo4j_evidence: List[Evidence]) -> Optional[Dict]:
        """使用LLM增强推荐"""
        try:
            # 构建证据文本
            evidence_text = ""
            for i, triple in enumerate(kg_triples, 1):
                evidence_text += f"[Config-{i}] {triple.subject} {triple.predicate} {triple.object}\n"
            for i, ev in enumerate(neo4j_evidence, len(kg_triples) + 1):
                evidence_text += f"[KG-{i}] {ev.statement}\n"

            if not evidence_text:
                evidence_text = "无直接证据，基于临床指南推理"

            # 构建干预配置文本
            interv_text = json.dumps(intervention_config, ensure_ascii=False, indent=2) if intervention_config else "无配置"

            prompt = self.STRATEGY_PROMPT.format(
                indicator=indicator,
                indicator_name=analysis['threshold_config'].get('name', indicator),
                current_value=value,
                unit=analysis['unit'],
                target_min=analysis['threshold_config'].get('thresholds', {}).get('target', {}).get('min', ''),
                target_max=analysis['threshold_config'].get('thresholds', {}).get('target', {}).get('max', ''),
                abnormality=analysis['abnormality'],
                severity=analysis['severity'].value,
                intervention_config=interv_text[:2000],
                kg_evidence=evidence_text
            )

            response = self.llm.generate(prompt, temperature=0.1)

            # 解析JSON响应
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"LLM增强失败: {e}")

        return None


# =============================================================================
# Robustness检查器
# =============================================================================

class RobustnessChecker:
    """鲁棒性和一致性检查器"""

    @staticmethod
    def check_consistency(recommendations: List[StrategyRecommendation]) -> Dict[str, Any]:
        """检查推荐的一致性"""
        issues = []

        # 检查是否有冲突的药物推荐
        all_drugs = []
        for rec in recommendations:
            for action in rec.immediate_actions:
                if action.drug:
                    all_drugs.append((rec.indicator, action.drug))

        # 检查药物交互
        drug_conflicts = [
            (['钙剂', '地高辛'], '钙剂与地高辛有相互作用风险'),
            (['胰岛素', '补钾'], '胰岛素会导致钾转移入细胞，需同时监测'),
        ]

        for (drug1_keywords, drug2_keywords), warning in drug_conflicts:
            found1 = any(any(k in d for k in drug1_keywords) for _, d in all_drugs)
            found2 = any(any(k in d for k in drug2_keywords) for _, d in all_drugs)
            if found1 and found2:
                issues.append({
                    'type': 'drug_interaction',
                    'warning': warning,
                    'severity': 'warning'
                })

        # 检查是否有相互矛盾的目标
        targets = {rec.indicator: rec.target_range for rec in recommendations}

        return {
            'consistent': len(issues) == 0,
            'issues': issues,
            'drug_count': len(all_drugs),
            'target_indicators': list(targets.keys())
        }

    @staticmethod
    def validate_evidence(recommendation: StrategyRecommendation) -> Dict[str, Any]:
        """验证证据支持"""
        evidence_count = len(recommendation.evidence)
        kg_triple_count = len(recommendation.kg_triples)

        has_config_evidence = any(e.source == 'config' or e.source.endswith('.yaml') for e in recommendation.evidence)
        has_kg_evidence = any(e.source == 'neo4j_kg' for e in recommendation.evidence)

        score = 0
        if evidence_count > 0:
            score += 0.3
        if kg_triple_count > 0:
            score += 0.3
        if has_config_evidence:
            score += 0.2
        if has_kg_evidence:
            score += 0.2

        return {
            'evidence_count': evidence_count,
            'kg_triple_count': kg_triple_count,
            'has_config_evidence': has_config_evidence,
            'has_kg_evidence': has_kg_evidence,
            'evidence_score': score,
            'well_supported': score >= 0.5
        }


# =============================================================================
# 主接口：BaselineStrategyRecommender
# =============================================================================

class BaselineStrategyRecommender:
    """Baseline策略推荐器 - 主接口"""

    def __init__(self, config_dir: str = None, llm: LLMInterface = None, neo4j_connector=None):
        self.config = ConfigLoader(config_dir)
        self.generator = StrategyGenerator(self.config, llm, neo4j_connector)
        self.checker = RobustnessChecker()

    def set_llm(self, llm: LLMInterface):
        """设置LLM"""
        self.generator.set_llm(llm)

    def set_neo4j(self, connector):
        """设置Neo4j连接器"""
        self.generator.set_neo4j(connector)

    def analyze_baseline(self, measurements: Dict[str, float], sample_id: str = "unknown",
                         timestamp_min: int = 30) -> BaselineReport:
        """分析Baseline并生成报告"""
        abnormalities = []
        recommendations = []

        # 分析每个指标
        for indicator, value in measurements.items():
            if value is None:
                continue

            analysis = self.generator.analyze_indicator(indicator, value)
            if analysis:
                abnormalities.append(analysis)
                recommendation = self.generator.generate_strategy(analysis)
                recommendations.append(recommendation)

        # 按严重程度排序
        recommendations.sort(key=lambda x: (
            0 if x.severity == Severity.CRITICAL else 1,
            -x.confidence
        ))

        # 一致性检查
        consistency_check = self.checker.check_consistency(recommendations)

        # 计算风险等级
        risk_level = self._calculate_risk_level(abnormalities)

        # 生成摘要
        summary = self._generate_summary(abnormalities, recommendations)

        return BaselineReport(
            sample_id=sample_id,
            timestamp_min=timestamp_min,
            measurements=measurements,
            abnormalities=[asdict(a) if hasattr(a, '__dataclass_fields__') else a for a in abnormalities],
            recommendations=recommendations,
            risk_level=risk_level,
            summary=summary,
            consistency_check=consistency_check
        )

    def _calculate_risk_level(self, abnormalities: List[Dict]) -> str:
        """计算风险等级"""
        critical_count = sum(1 for a in abnormalities if a.get('severity') == Severity.CRITICAL)
        warning_count = sum(1 for a in abnormalities if a.get('severity') == Severity.WARNING)

        critical_indicators = {'MAP', 'CI', 'K_A', 'pH'}
        has_critical_indicator = any(a['indicator'] in critical_indicators for a in abnormalities)

        if critical_count >= 2 or (critical_count >= 1 and has_critical_indicator):
            return "HIGH"
        elif critical_count >= 1 or warning_count >= 3:
            return "MEDIUM"
        elif warning_count >= 1:
            return "LOW"
        else:
            return "MINIMAL"

    def _generate_summary(self, abnormalities: List[Dict], recommendations: List[StrategyRecommendation]) -> str:
        """生成摘要"""
        if not abnormalities:
            return "所有指标在正常范围内，建议继续标准监测。"

        lines = [f"检测到 {len(abnormalities)} 项指标异常:"]

        for abn in abnormalities[:3]:
            direction = "偏低" if abn.get('direction') == 'low' else "偏高"
            lines.append(f"- {abn['indicator']}: {abn['value']} {abn['unit']} ({direction})")

        if recommendations:
            top = recommendations[0]
            lines.append(f"\n首要干预: {top.immediate_actions[0].action if top.immediate_actions else '评估后干预'}")
            lines.append(f"目标: {top.indicator} → {top.target_range[0]}-{top.target_range[1]} {top.unit}")

        return "\n".join(lines)

    def format_report(self, report: BaselineReport) -> str:
        """格式化报告输出"""
        lines = []
        lines.append("=" * 100)
        lines.append(f"BASELINE STRATEGY REPORT")
        lines.append(f"Sample: {report.sample_id} | Time: t={report.timestamp_min}min | Risk: {report.risk_level}")
        lines.append("=" * 100)

        lines.append(f"\n## Summary\n{report.summary}")

        if report.consistency_check.get('issues'):
            lines.append("\n## ⚠️ Consistency Warnings")
            for issue in report.consistency_check['issues']:
                lines.append(f"  - [{issue['severity']}] {issue['warning']}")

        if report.recommendations:
            lines.append("\n## Detailed Recommendations")
            lines.append("-" * 100)

            for i, rec in enumerate(report.recommendations, 1):
                lines.append(f"\n### [{i}] {rec.indicator}: {rec.current_value} → {rec.target_value:.1f} {rec.unit}")
                lines.append(f"**Severity**: {rec.severity.value.upper()}")
                lines.append(f"**Confidence**: {rec.confidence:.0%}")

                if rec.immediate_actions:
                    lines.append("\n**Immediate Actions**:")
                    for action in rec.immediate_actions:
                        lines.append(f"  → {action.action}")
                        if action.drug:
                            lines.append(f"    Drug: {action.drug}")
                        if action.dose:
                            lines.append(f"    Dose: {action.dose}")

                lines.append("\n**Reasoning Chain (CoT)**:")
                for step in rec.reasoning_chain:
                    lines.append(f"  {step}")

                if rec.evidence:
                    lines.append("\n**Evidence**:")
                    for j, ev in enumerate(rec.evidence[:3], 1):
                        lines.append(f"  [{j}] {ev.statement} (Source: {ev.source})")

                if rec.kg_triples:
                    lines.append("\n**KG Triples**:")
                    for triple in rec.kg_triples[:3]:
                        lines.append(f"  - {triple.subject} --{triple.predicate}--> {triple.object}")

                lines.append("\n**Monitoring**:")
                for m in rec.monitoring:
                    lines.append(f"  - {m}")

                if rec.warnings:
                    lines.append("\n**⚠️ Warnings**:")
                    for w in rec.warnings:
                        lines.append(f"  - {w}")

                lines.append("-" * 50)

        return "\n".join(lines)


# =============================================================================
# 测试函数
# =============================================================================

def test_baseline_recommender():
    """测试Baseline策略推荐器"""
    print("=" * 100)
    print("Testing Baseline Strategy Recommender")
    print("=" * 100)

    # 创建推荐器
    recommender = BaselineStrategyRecommender()

    # 测试数据
    test_measurements = {
        'MAP': 45,
        'Lactate': 4.5,
        'K_A': 6.2,
        'SvO2': 85,
        'pH': 7.28,
        'CI': 1.8,
        'CvO2': 5.5,
    }

    # 生成报告
    report = recommender.analyze_baseline(test_measurements, sample_id="TEST-001")

    # 输出
    print(recommender.format_report(report))

    # 验证证据
    print("\n" + "=" * 100)
    print("Evidence Validation")
    print("=" * 100)

    for rec in report.recommendations:
        validation = RobustnessChecker.validate_evidence(rec)
        print(f"\n{rec.indicator}:")
        print(f"  Evidence Score: {validation['evidence_score']:.0%}")
        print(f"  Well Supported: {validation['well_supported']}")


if __name__ == "__main__":
    test_baseline_recommender()
