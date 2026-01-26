#!/usr/bin/env python3
"""
Baseline Thresholds Module - 整合阈值配置与Neo4j知识图谱证据支持

功能：
1. 加载YAML阈值配置和baseline配置
2. 检测指标异常状态
3. 查询Neo4j获取生理学证据
4. 生成证据增强的干预建议
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """警报级别"""
    NORMAL = "normal"
    WARNING = "warning"
    RED_LINE = "red_line"
    CRITICAL = "critical"


class TrendDirection(Enum):
    """趋势方向"""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


@dataclass
class ThresholdResult:
    """阈值检查结果"""
    indicator: str
    value: float
    unit: str
    alert_level: AlertLevel
    threshold_description: str
    baseline_value: Optional[float] = None
    deviation: Optional[float] = None
    deviation_percent: Optional[float] = None
    trend: Optional[TrendDirection] = None
    confidence: str = "medium"
    source: List[str] = field(default_factory=list)


@dataclass
class EvidenceEnhancedIntervention:
    """证据增强的干预建议"""
    indicator: str
    alert_level: AlertLevel
    intervention: str
    intervention_type: str
    # Neo4j知识图谱证据
    causes: List[Dict] = field(default_factory=list)
    consequences: List[Dict] = field(default_factory=list)
    treatments: List[Dict] = field(default_factory=list)
    evidence_source: str = "neo4j_knowledge_graph"


# 指标异常状态到Neo4j查询关键词的映射
INDICATOR_TO_KG_MAPPING = {
    # 心功能
    "EF_Low": ["low ejection fraction", "systolic dysfunction", "heart failure"],
    "EF_Critical": ["severe systolic dysfunction", "cardiogenic shock"],
    "CI_Low": ["low cardiac output", "cardiogenic shock", "heart failure"],
    "CI_Critical": ["cardiogenic shock", "hemodynamic instability"],

    # 血压
    "MAP_Low": ["hypotension", "low blood pressure", "shock"],
    "MAP_Critical": ["severe hypotension", "circulatory shock"],

    # 氧代谢
    "SvO2_Low": ["tissue hypoxia", "oxygen extraction", "low cardiac output"],
    "SvO2_Critical": ["severe tissue hypoxia", "circulatory failure"],

    # 代谢
    "Lactate_High": ["lactic acidosis", "tissue hypoperfusion", "anaerobic metabolism"],
    "Lactate_Critical": ["severe lactic acidosis", "organ failure", "shock"],

    # 电解质
    "K_High": ["hyperkalemia", "arrhythmia", "cardiac arrest"],
    "K_Low": ["hypokalemia", "arrhythmia"],
    "K_Critical": ["severe hyperkalemia", "cardiac arrest"],
    "Na_High": ["hypernatremia", "dehydration"],
    "Na_Low": ["hyponatremia", "cerebral edema"],

    # 肺血管
    "PVR_High": ["pulmonary hypertension", "right heart failure"],

    # 心率
    "HR_High": ["tachycardia", "arrhythmia"],
    "HR_Low": ["bradycardia", "heart block"],

    # 灌注相关
    "Perfusion_Low": ["hypoperfusion", "ischemia", "organ dysfunction"],
    "Reperfusion_Injury": ["ischemia-reperfusion injury", "reperfusion injury", "IRI"],
    "Graft_Dysfunction": ["primary graft dysfunction", "graft failure", "PGD"],
}


class BaselineThresholds:
    """Baseline阈值管理器"""

    def __init__(self, config_dir: Optional[str] = None, neo4j_connector=None):
        """
        初始化

        Args:
            config_dir: 配置目录路径
            neo4j_connector: Neo4j连接器实例（可选，延迟注入）
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        # 加载配置
        self.thresholds_config = self._load_yaml("thresholds.yaml")
        self.baseline_config = self._load_yaml("baseline.yaml")

        # Neo4j连接器（可选）
        self.neo4j = neo4j_connector

        # 构建阈值索引
        self._build_threshold_index()

    def _load_yaml(self, filename: str) -> Dict:
        """加载YAML配置文件"""
        filepath = self.config_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _build_threshold_index(self):
        """构建阈值快速索引"""
        self.threshold_index = {}

        # 遍历所有分类
        for category, indicators in self.thresholds_config.items():
            if isinstance(indicators, dict) and category not in ['version', 'last_updated', 'update_notes']:
                for indicator_key, indicator_config in indicators.items():
                    if isinstance(indicator_config, dict) and 'thresholds' in indicator_config:
                        self.threshold_index[indicator_key] = {
                            'category': category,
                            'config': indicator_config
                        }

    def set_neo4j_connector(self, connector):
        """设置Neo4j连接器"""
        self.neo4j = connector

    def get_baseline(self, indicator: str) -> Optional[Dict]:
        """获取指标的baseline配置"""
        # 先查confirmed_baselines
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        if indicator in confirmed:
            return confirmed[indicator]

        # 再查pending_baselines
        pending = self.baseline_config.get('pending_baselines', {})
        if indicator in pending:
            return pending[indicator]

        return None

    def get_threshold_config(self, indicator: str) -> Optional[Dict]:
        """获取指标的阈值配置"""
        if indicator in self.threshold_index:
            return self.threshold_index[indicator]['config']
        return None

    def check_threshold(self, indicator: str, value: float,
                        baseline_value: Optional[float] = None) -> ThresholdResult:
        """
        检查单个指标的阈值状态

        Args:
            indicator: 指标名称
            value: 当前值
            baseline_value: baseline值（可选，不提供则使用配置中的值）

        Returns:
            ThresholdResult对象
        """
        config = self.get_threshold_config(indicator)
        if not config:
            return ThresholdResult(
                indicator=indicator,
                value=value,
                unit="",
                alert_level=AlertLevel.NORMAL,
                threshold_description="未找到阈值配置"
            )

        thresholds = config.get('thresholds', {})
        unit = config.get('unit', '')
        confidence = config.get('confidence', 'medium')
        source = config.get('source', [])

        # 确定baseline
        if baseline_value is None:
            baseline_config = self.get_baseline(indicator)
            if baseline_config:
                baseline_value = baseline_config.get('baseline_value') or baseline_config.get('provisional_baseline')

        # 检查阈值级别
        alert_level = AlertLevel.NORMAL
        threshold_description = "正常"

        # 检查critical
        if 'critical' in thresholds:
            crit = thresholds['critical']
            if self._check_condition(value, crit):
                alert_level = AlertLevel.CRITICAL
                threshold_description = crit.get('description', '危急')

        # 检查red_line
        if alert_level == AlertLevel.NORMAL:
            for key in ['red_line', 'red_line_low', 'red_line_high']:
                if key in thresholds:
                    rl = thresholds[key]
                    if self._check_condition(value, rl):
                        alert_level = AlertLevel.RED_LINE
                        threshold_description = rl.get('description', '红线')
                        break

        # 检查warning
        if alert_level == AlertLevel.NORMAL:
            for key in ['warning', 'warning_low', 'warning_high', 'warning_range']:
                if key in thresholds:
                    warn = thresholds[key]
                    if self._check_condition(value, warn):
                        alert_level = AlertLevel.WARNING
                        threshold_description = warn.get('description', '警戒')
                        break

        # 检查reject（用于EF等有accept/reject的指标）
        if alert_level == AlertLevel.NORMAL and 'reject' in thresholds:
            rej = thresholds['reject']
            if self._check_condition(value, rej):
                alert_level = AlertLevel.RED_LINE
                threshold_description = rej.get('description', '不合格')

        # 计算偏离
        deviation = None
        deviation_percent = None
        trend = None

        if baseline_value is not None and baseline_value != 0:
            deviation = value - baseline_value
            deviation_percent = (deviation / baseline_value) * 100

            # 评估趋势（需要知道指标的"好"方向）
            trend = self._assess_trend(indicator, deviation)

        return ThresholdResult(
            indicator=indicator,
            value=value,
            unit=unit,
            alert_level=alert_level,
            threshold_description=threshold_description,
            baseline_value=baseline_value,
            deviation=deviation,
            deviation_percent=deviation_percent,
            trend=trend,
            confidence=confidence,
            source=source
        )

    def _check_condition(self, value: float, threshold_config: Dict) -> bool:
        """检查值是否满足阈值条件"""
        if 'operator' in threshold_config and 'value' in threshold_config:
            op = threshold_config['operator']
            th_value = threshold_config['value']

            if op == '<':
                return value < th_value
            elif op == '<=':
                return value <= th_value
            elif op == '>':
                return value > th_value
            elif op == '>=':
                return value >= th_value
            elif op == '==':
                return value == th_value

        # 范围检查
        if 'min' in threshold_config and 'max' in threshold_config:
            return threshold_config['min'] <= value <= threshold_config['max']
        elif 'min' in threshold_config:
            return value >= threshold_config['min']
        elif 'max' in threshold_config:
            return value <= threshold_config['max']

        return False

    def _assess_trend(self, indicator: str, deviation: float) -> TrendDirection:
        """评估趋势方向"""
        # 需要增加为好的指标
        increase_is_good = ['EF', 'CI', 'MAP', 'SvO2', 'CF', 'AOP']
        # 需要减少为好的指标
        decrease_is_good = ['Lactate', 'PVR', 'chest_drainage']
        # 维持在目标范围内为好的指标
        maintain_target = ['K_A', 'Na_A', 'GluA', 'HR']

        baseline_config = self.get_baseline(indicator)
        acceptable_dev = 0
        if baseline_config:
            acceptable_dev = baseline_config.get('acceptable_deviation', 0)

        if abs(deviation) <= acceptable_dev:
            return TrendDirection.STABLE

        if indicator in increase_is_good:
            return TrendDirection.IMPROVING if deviation > 0 else TrendDirection.DETERIORATING
        elif indicator in decrease_is_good:
            return TrendDirection.IMPROVING if deviation < 0 else TrendDirection.DETERIORATING
        else:
            # 默认：偏离越大越不好
            return TrendDirection.DETERIORATING if abs(deviation) > acceptable_dev else TrendDirection.STABLE

    def get_abnormality_state(self, indicator: str, value: float) -> str:
        """
        获取异常状态标识（用于Neo4j查询映射）

        Returns:
            如 "Lactate_High", "MAP_Low", "K_Critical" 等
        """
        result = self.check_threshold(indicator, value)

        if result.alert_level == AlertLevel.NORMAL:
            return f"{indicator}_Normal"

        # 确定方向
        baseline = result.baseline_value
        config = self.get_threshold_config(indicator)

        direction = ""
        if config:
            thresholds = config.get('thresholds', {})
            # 检查是高还是低
            if 'target' in thresholds:
                target = thresholds['target']
                if 'min' in target and value < target['min']:
                    direction = "Low"
                elif 'max' in target and value > target['max']:
                    direction = "High"
            elif baseline is not None:
                direction = "High" if value > baseline else "Low"

        # 组合状态
        level_suffix = ""
        if result.alert_level == AlertLevel.CRITICAL:
            level_suffix = "_Critical"
        elif result.alert_level == AlertLevel.RED_LINE:
            level_suffix = ""  # 默认
        elif result.alert_level == AlertLevel.WARNING:
            level_suffix = ""

        return f"{indicator}_{direction}{level_suffix}" if direction else f"{indicator}_Abnormal"

    def query_evidence(self, abnormality_state: str) -> Dict[str, List[Dict]]:
        """
        查询Neo4j获取异常状态的证据支持

        Args:
            abnormality_state: 异常状态标识

        Returns:
            包含causes, consequences, treatments的字典
        """
        if self.neo4j is None:
            logger.warning("Neo4j连接器未设置，无法查询证据")
            return {'causes': [], 'consequences': [], 'treatments': []}

        # 获取查询关键词
        keywords = INDICATOR_TO_KG_MAPPING.get(abnormality_state, [abnormality_state])

        evidence = {
            'causes': [],
            'consequences': [],
            'treatments': []
        }

        for keyword in keywords[:2]:  # 限制查询数量
            try:
                result = self.neo4j.query_decision_support(keyword)
                evidence['causes'].extend(result.get('causes', []))
                evidence['consequences'].extend(result.get('consequences', []))
                evidence['treatments'].extend(result.get('treatments', []))
            except Exception as e:
                logger.error(f"Neo4j查询失败: {e}")

        # 去重
        for key in evidence:
            seen = set()
            unique = []
            for item in evidence[key]:
                item_key = str(item)
                if item_key not in seen:
                    seen.add(item_key)
                    unique.append(item)
            evidence[key] = unique[:10]  # 限制返回数量

        return evidence

    def generate_evidence_enhanced_intervention(
        self,
        indicator: str,
        value: float,
        include_evidence: bool = True
    ) -> Optional[EvidenceEnhancedIntervention]:
        """
        生成证据增强的干预建议

        Args:
            indicator: 指标名称
            value: 当前值
            include_evidence: 是否包含Neo4j证据

        Returns:
            EvidenceEnhancedIntervention对象
        """
        result = self.check_threshold(indicator, value)

        if result.alert_level == AlertLevel.NORMAL:
            return None

        # 获取异常状态
        abnormality_state = self.get_abnormality_state(indicator, value)

        # 基础干预建议（来自配置）
        config = self.get_threshold_config(indicator)
        intervention = ""
        intervention_type = "general"

        if config:
            if result.alert_level == AlertLevel.CRITICAL:
                intervention = config.get('escalation_rule', '紧急评估，考虑升级支持')
                intervention_type = "escalation"
            else:
                intervention = f"监测 {indicator}，当前 {result.threshold_description}"
                intervention_type = "monitoring"

        # 查询Neo4j证据
        evidence = {'causes': [], 'consequences': [], 'treatments': []}
        if include_evidence and self.neo4j:
            evidence = self.query_evidence(abnormality_state)

        return EvidenceEnhancedIntervention(
            indicator=indicator,
            alert_level=result.alert_level,
            intervention=intervention,
            intervention_type=intervention_type,
            causes=evidence['causes'],
            consequences=evidence['consequences'],
            treatments=evidence['treatments']
        )

    def check_all_indicators(self, indicators: Dict[str, float]) -> List[ThresholdResult]:
        """
        批量检查所有指标

        Args:
            indicators: 指标名称到值的映射

        Returns:
            ThresholdResult列表
        """
        results = []
        for indicator, value in indicators.items():
            if value is not None:
                results.append(self.check_threshold(indicator, value))
        return results

    def get_alerts(self, indicators: Dict[str, float],
                   min_level: AlertLevel = AlertLevel.WARNING) -> List[ThresholdResult]:
        """
        获取达到指定级别的警报

        Args:
            indicators: 指标值字典
            min_level: 最低警报级别

        Returns:
            警报列表
        """
        results = self.check_all_indicators(indicators)

        level_order = {
            AlertLevel.NORMAL: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.RED_LINE: 2,
            AlertLevel.CRITICAL: 3
        }

        min_order = level_order[min_level]

        return [r for r in results if level_order[r.alert_level] >= min_order]

    def generate_report(self, indicators: Dict[str, float],
                        include_evidence: bool = False) -> str:
        """
        生成阈值检查报告

        Args:
            indicators: 指标值字典
            include_evidence: 是否包含Neo4j证据

        Returns:
            报告文本
        """
        results = self.check_all_indicators(indicators)

        lines = ["=" * 60]
        lines.append("Baseline Threshold Check Report")
        lines.append("=" * 60)

        # 按级别分组
        critical = [r for r in results if r.alert_level == AlertLevel.CRITICAL]
        red_line = [r for r in results if r.alert_level == AlertLevel.RED_LINE]
        warning = [r for r in results if r.alert_level == AlertLevel.WARNING]
        normal = [r for r in results if r.alert_level == AlertLevel.NORMAL]

        if critical:
            lines.append("\n### CRITICAL ALERTS ###")
            for r in critical:
                lines.append(f"  [CRITICAL] {r.indicator}: {r.value} {r.unit}")
                lines.append(f"    - {r.threshold_description}")
                if r.deviation is not None:
                    lines.append(f"    - Baseline: {r.baseline_value}, Deviation: {r.deviation:+.2f} ({r.deviation_percent:+.1f}%)")
                if include_evidence:
                    intervention = self.generate_evidence_enhanced_intervention(r.indicator, r.value)
                    if intervention and intervention.treatments:
                        lines.append(f"    - Evidence-based treatments: {[t.get('treatment', t) for t in intervention.treatments[:3]]}")

        if red_line:
            lines.append("\n### RED LINE ALERTS ###")
            for r in red_line:
                lines.append(f"  [RED LINE] {r.indicator}: {r.value} {r.unit}")
                lines.append(f"    - {r.threshold_description}")

        if warning:
            lines.append("\n### WARNINGS ###")
            for r in warning:
                lines.append(f"  [WARNING] {r.indicator}: {r.value} {r.unit}")
                lines.append(f"    - {r.threshold_description}")

        lines.append(f"\n### Summary ###")
        lines.append(f"  Total indicators: {len(results)}")
        lines.append(f"  Critical: {len(critical)}, Red Line: {len(red_line)}, Warning: {len(warning)}, Normal: {len(normal)}")

        return "\n".join(lines)


def test_baseline_thresholds():
    """测试函数"""
    bt = BaselineThresholds()

    print("=" * 60)
    print("Baseline Thresholds Test")
    print("=" * 60)

    # 测试数据
    test_indicators = {
        'EF': 45,       # 灰区
        'CI': 1.8,      # 低于红线
        'MAP': 58,      # 低于红线
        'Lactate': 6.5, # 超过reject
        'K_A': 6.2,     # 高钾警戒
        'SvO2': 55,     # 低于红线
        'HR': 115,      # 警戒范围
    }

    print("\n### Individual Threshold Checks ###")
    for indicator, value in test_indicators.items():
        result = bt.check_threshold(indicator, value)
        print(f"\n{indicator}: {value} {result.unit}")
        print(f"  Level: {result.alert_level.value}")
        print(f"  Description: {result.threshold_description}")
        print(f"  Confidence: {result.confidence}")
        if result.baseline_value:
            print(f"  Baseline: {result.baseline_value}")
            print(f"  Deviation: {result.deviation:+.2f} ({result.deviation_percent:+.1f}%)")
            print(f"  Trend: {result.trend.value if result.trend else 'N/A'}")

    print("\n" + "=" * 60)
    print("Full Report:")
    print("=" * 60)
    print(bt.generate_report(test_indicators))

    print("\n### Abnormality States ###")
    for indicator, value in test_indicators.items():
        state = bt.get_abnormality_state(indicator, value)
        print(f"  {indicator}={value} -> {state}")


def test_with_neo4j():
    """测试Neo4j集成"""
    try:
        from neo4j_connector import Neo4jKnowledgeGraph

        print("\n" + "=" * 60)
        print("Testing with Neo4j Integration")
        print("=" * 60)

        # 创建连接
        neo4j = Neo4jKnowledgeGraph()

        # 创建阈值管理器并注入Neo4j
        bt = BaselineThresholds()
        bt.set_neo4j_connector(neo4j)

        # 测试证据增强的干预建议
        test_cases = [
            ('Lactate', 7.0),
            ('CI', 1.5),
            ('K_A', 6.5),
        ]

        for indicator, value in test_cases:
            print(f"\n### {indicator} = {value} ###")
            intervention = bt.generate_evidence_enhanced_intervention(indicator, value)

            if intervention:
                print(f"Alert Level: {intervention.alert_level.value}")
                print(f"Intervention: {intervention.intervention}")
                print(f"Type: {intervention.intervention_type}")

                if intervention.causes:
                    print(f"Causes from KG:")
                    for c in intervention.causes[:3]:
                        print(f"  - {c}")

                if intervention.consequences:
                    print(f"Consequences from KG:")
                    for c in intervention.consequences[:3]:
                        print(f"  - {c}")

                if intervention.treatments:
                    print(f"Treatments from KG:")
                    for t in intervention.treatments[:3]:
                        print(f"  - {t}")

        neo4j.close()

    except Exception as e:
        print(f"Neo4j integration test failed: {e}")
        print("Run without Neo4j to test basic functionality")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "neo4j":
        test_with_neo4j()
    else:
        test_baseline_thresholds()
