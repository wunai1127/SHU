"""
灌注监测模块 - 时序监测、报告生成、策略推荐
"""

import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .threshold_manager import ThresholdManager, ThresholdEvaluation, EvaluationResult
from .baseline_evaluator import BaselineEvaluator, BaselineComparison, Trend
from .knowledge_graph import KnowledgeGraph


class AlertLevel(Enum):
    """警报级别"""
    CRITICAL = "critical"    # 危急
    WARNING = "warning"      # 警告
    INFO = "info"           # 信息
    NORMAL = "normal"       # 正常


class InterventionStatus(Enum):
    """干预状态"""
    PENDING = "pending"          # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"      # 已完成
    EFFECTIVE = "effective"      # 有效
    INEFFECTIVE = "ineffective"  # 无效


@dataclass
class TimeWindowData:
    """时间窗数据"""
    time_minutes: int
    timestamp: datetime
    measurements: Dict[str, float]
    evaluations: Dict[str, ThresholdEvaluation] = field(default_factory=dict)
    comparisons: Dict[str, BaselineComparison] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""


@dataclass
class InterventionRecord:
    """干预记录"""
    time_initiated: int          # 开始时间窗
    indicator: str               # 相关指标
    abnormality: str            # 异常类型
    action: str                 # 干预措施
    status: InterventionStatus  # 状态
    time_windows_active: List[int] = field(default_factory=list)  # 活跃的时间窗
    outcome: Optional[str] = None


class PerfusionMonitor:
    """灌注监测器"""

    # 标准时间窗（分钟）
    TIME_WINDOWS = [30, 60, 120, 180, 240]

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化灌注监测器

        Args:
            config_dir: 配置目录
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        # 加载组件
        self.threshold_manager = ThresholdManager(config_dir)
        self.baseline_evaluator = BaselineEvaluator(config_dir)
        self.knowledge_graph = KnowledgeGraph(config_dir)

        # 加载干预策略
        self._load_intervention_strategies()

        # 数据存储
        self.baseline_data: Optional[TimeWindowData] = None
        self.time_window_history: Dict[int, TimeWindowData] = {}
        self.intervention_records: List[InterventionRecord] = []
        self.active_interventions: Dict[str, InterventionRecord] = {}

    def _load_intervention_strategies(self):
        """加载干预策略配置"""
        strategies_path = self.config_dir / "intervention_strategies.yaml"
        if strategies_path.exists():
            with open(strategies_path, 'r', encoding='utf-8') as f:
                self.intervention_strategies = yaml.safe_load(f)
        else:
            self.intervention_strategies = {}

    def set_baseline(self, measurements: Dict[str, float], time_minutes: int = 30):
        """
        设置baseline（t=30min）

        ⚠️ 重要：baseline设置时立即与阈值对比，识别需要紧急干预的指标
        这确保了在灌注开始时就能发现并处理异常

        Args:
            measurements: 测量值字典
            time_minutes: 时间点（默认30分钟）
        """
        self.baseline_data = self._process_time_window(measurements, time_minutes, is_baseline=True)

        # 设置动态baseline
        for indicator, value in measurements.items():
            self.baseline_evaluator.set_dynamic_baseline(indicator, value)

        # ⚠️ 立即检查是否有需要紧急干预的指标
        urgent_interventions = self._identify_urgent_interventions(self.baseline_data)

        if urgent_interventions:
            # 自动记录需要紧急关注的干预
            for ui in urgent_interventions:
                self.active_interventions[ui['indicator']] = InterventionRecord(
                    time_initiated=time_minutes,
                    indicator=ui['indicator'],
                    abnormality=ui['abnormality'],
                    action=ui['recommended_action'],
                    status=InterventionStatus.PENDING,
                    time_windows_active=[time_minutes]
                )
                self.intervention_records.append(self.active_interventions[ui['indicator']])

        return self.generate_baseline_report()

    def _identify_urgent_interventions(self, window: TimeWindowData) -> List[Dict[str, Any]]:
        """
        识别需要紧急干预的指标

        在baseline出现时立即评估，确保危急情况得到及时处理

        Args:
            window: 时间窗数据

        Returns:
            紧急干预列表
        """
        urgent = []

        for alert in window.alerts:
            if alert['level'] == AlertLevel.CRITICAL:
                indicator = alert['indicator']
                abnormality = self._get_abnormality_type(indicator, alert)

                # 从知识图谱获取紧急干预建议
                kg_interventions = self.knowledge_graph.find_interventions(abnormality)
                immediate_actions = [i for i in kg_interventions if i['urgency'] == 'immediate']

                # 从策略配置获取详细策略
                strategy_actions = self._get_strategy_interventions(indicator, abnormality)
                immediate_strategies = [s for s in strategy_actions if s['type'] == 'immediate']

                recommended_action = ""
                if immediate_actions:
                    recommended_action = immediate_actions[0]['action']
                elif immediate_strategies:
                    recommended_action = immediate_strategies[0]['action']
                else:
                    recommended_action = alert.get('action', '需要立即评估')

                urgent.append({
                    'indicator': indicator,
                    'abnormality': abnormality,
                    'value': alert['value'],
                    'message': alert['message'],
                    'recommended_action': recommended_action,
                    'kg_recommendations': immediate_actions,
                    'strategy_recommendations': immediate_strategies,
                    'risks': self.knowledge_graph.find_risks(abnormality)
                })

        return urgent

    def process_time_window(self, measurements: Dict[str, float], time_minutes: int) -> TimeWindowData:
        """
        处理时间窗数据

        Args:
            measurements: 测量值字典
            time_minutes: 时间点（分钟）

        Returns:
            TimeWindowData
        """
        if self.baseline_data is None:
            raise ValueError("请先设置baseline (调用 set_baseline)")

        window_data = self._process_time_window(measurements, time_minutes)
        self.time_window_history[time_minutes] = window_data

        # 检查之前的干预是否有效
        self._evaluate_intervention_effectiveness(time_minutes, window_data)

        return window_data

    def _process_time_window(self, measurements: Dict[str, float],
                             time_minutes: int, is_baseline: bool = False) -> TimeWindowData:
        """处理单个时间窗"""
        window = TimeWindowData(
            time_minutes=time_minutes,
            timestamp=datetime.now(),
            measurements=measurements.copy()
        )

        # 阈值评估
        for indicator, value in measurements.items():
            evaluation = self.threshold_manager.evaluate(indicator, value)
            window.evaluations[indicator] = evaluation

            # 生成警报
            if evaluation.result in [EvaluationResult.REJECT, EvaluationResult.RED_LINE]:
                window.alerts.append({
                    'level': AlertLevel.CRITICAL,
                    'indicator': indicator,
                    'value': value,
                    'message': evaluation.message,
                    'action': evaluation.action
                })
            elif evaluation.result in [EvaluationResult.WARNING, EvaluationResult.GRAY_ZONE]:
                window.alerts.append({
                    'level': AlertLevel.WARNING,
                    'indicator': indicator,
                    'value': value,
                    'message': evaluation.message,
                    'action': evaluation.action
                })

        # Baseline对比（非baseline时）
        if not is_baseline and self.baseline_data:
            for indicator, value in measurements.items():
                comparison = self.baseline_evaluator.compare(indicator, value)
                window.comparisons[indicator] = comparison

        # 生成干预建议
        window.interventions = self._generate_interventions(window)

        return window

    def _generate_interventions(self, window: TimeWindowData) -> List[Dict[str, Any]]:
        """生成干预建议"""
        interventions = []

        for alert in window.alerts:
            indicator = alert['indicator']
            abnormality = self._get_abnormality_type(indicator, alert)

            # 从知识图谱获取干预
            kg_interventions = self.knowledge_graph.find_interventions(abnormality)

            # 从策略配置获取详细干预
            strategy_interventions = self._get_strategy_interventions(indicator, abnormality)

            intervention = {
                'indicator': indicator,
                'abnormality': abnormality,
                'alert_level': alert['level'].value,
                'kg_recommendations': kg_interventions,
                'detailed_strategies': strategy_interventions,
                'rationale': self._get_intervention_rationale(abnormality)
            }

            interventions.append(intervention)

        return interventions

    def _get_abnormality_type(self, indicator: str, alert: Dict) -> str:
        """获取异常类型标识"""
        level = alert['level']
        value = alert['value']

        # 根据指标和级别生成异常类型
        if level == AlertLevel.CRITICAL:
            return f"{indicator}_Critical"
        else:
            return f"{indicator}_Low" if "低" in alert.get('message', '') or "下降" in alert.get('message', '') else f"{indicator}_High"

    def _get_strategy_interventions(self, indicator: str, abnormality: str) -> List[Dict]:
        """从策略配置获取干预"""
        interventions = []

        # 遍历策略配置查找匹配的干预
        for category, indicators in self.intervention_strategies.items():
            if isinstance(indicators, dict) and indicator in indicators:
                ind_config = indicators[indicator]
                if 'abnormalities' in ind_config:
                    for abn_name, abn_config in ind_config['abnormalities'].items():
                        if 'strategies' in abn_config:
                            strategies = abn_config['strategies']
                            if 'immediate' in strategies:
                                for action in strategies['immediate']:
                                    interventions.append({
                                        'type': 'immediate',
                                        'action': action.get('action', ''),
                                        'details': action
                                    })
                            if 'if_no_improvement' in strategies:
                                for action in strategies['if_no_improvement']:
                                    interventions.append({
                                        'type': 'escalation',
                                        'action': action.get('action', ''),
                                        'details': action
                                    })

        return interventions

    def _get_intervention_rationale(self, abnormality: str) -> str:
        """获取干预理由"""
        causes = self.knowledge_graph.find_causes(abnormality)
        risks = self.knowledge_graph.find_risks(abnormality)

        rationale_parts = []
        if causes:
            rationale_parts.append(f"可能原因: {', '.join(causes)}")
        if risks:
            rationale_parts.append(f"风险: {', '.join(risks)}")

        return "; ".join(rationale_parts) if rationale_parts else "基于临床指南推荐"

    def _evaluate_intervention_effectiveness(self, current_time: int, current_data: TimeWindowData):
        """评估之前干预的有效性"""
        for indicator, record in list(self.active_interventions.items()):
            # 检查该指标是否仍有警报
            still_abnormal = any(
                a['indicator'] == indicator and a['level'] in [AlertLevel.CRITICAL, AlertLevel.WARNING]
                for a in current_data.alerts
            )

            record.time_windows_active.append(current_time)

            # 如果超过2个时间窗仍无改善
            if len(record.time_windows_active) >= 2 and still_abnormal:
                record.status = InterventionStatus.INEFFECTIVE
                record.outcome = f"干预在{len(record.time_windows_active)}个时间窗后仍未见效"

                # 触发升级策略
                self._trigger_escalation(indicator, record, current_data)
            elif not still_abnormal:
                record.status = InterventionStatus.EFFECTIVE
                record.outcome = f"指标已恢复正常 (t={current_time}min)"
                del self.active_interventions[indicator]

    def _trigger_escalation(self, indicator: str, record: InterventionRecord,
                           current_data: TimeWindowData):
        """触发升级策略"""
        abnormality = record.abnormality

        # 从知识图谱搜索升级知识
        related_knowledge = self.knowledge_graph.search_related_knowledge(abnormality)

        # 查找升级路径
        escalation_path = self.knowledge_graph.get_escalation_path(indicator.lower())

        escalation_info = {
            'indicator': indicator,
            'previous_intervention': record.action,
            'time_windows_tried': record.time_windows_active,
            'related_knowledge': [str(t) for t in related_knowledge[:5]],
            'escalation_path': escalation_path,
            'recommendation': self._generate_escalation_recommendation(abnormality, record)
        }

        current_data.notes += f"\n⚠️ {indicator} 升级触发: {escalation_info['recommendation']}"

        return escalation_info

    def _generate_escalation_recommendation(self, abnormality: str, record: InterventionRecord) -> str:
        """生成升级建议"""
        # 从知识图谱查找升级措施
        escalation_triples = self.knowledge_graph.query(
            subject=abnormality,
            predicate='escalate_to'
        )

        if escalation_triples:
            return f"建议升级到: {escalation_triples[0].object}"

        # 通用升级建议
        return "当前策略无效，建议评估根本原因并考虑更高级别支持"

    def generate_baseline_report(self) -> str:
        """生成baseline报告"""
        if not self.baseline_data:
            return "未设置baseline"

        lines = [
            "=" * 70,
            "心脏灌注监测 - Baseline报告",
            f"时间点: t = {self.baseline_data.time_minutes} min",
            f"生成时间: {self.baseline_data.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]

        # 指标汇总
        lines.append("## 指标状态汇总\n")

        critical_count = sum(1 for a in self.baseline_data.alerts if a['level'] == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in self.baseline_data.alerts if a['level'] == AlertLevel.WARNING)

        lines.append(f"🔴 危急警报: {critical_count}")
        lines.append(f"🟡 警告: {warning_count}")
        lines.append(f"🟢 正常: {len(self.baseline_data.measurements) - critical_count - warning_count}")
        lines.append("")

        # ⚠️ 紧急干预提示（Baseline时立即显示）
        if critical_count > 0:
            lines.append("=" * 70)
            lines.append("⚠️⚠️⚠️ 立即需要干预 - IMMEDIATE ACTION REQUIRED ⚠️⚠️⚠️")
            lines.append("=" * 70)
            lines.append("")

            for indicator, record in self.active_interventions.items():
                if record.status == InterventionStatus.PENDING:
                    lines.append(f"### 🚨 {indicator} - {record.abnormality}")
                    lines.append(f"**建议干预**: {record.action}")

                    # 风险提示
                    risks = self.knowledge_graph.find_risks(record.abnormality)
                    if risks:
                        lines.append(f"**风险**: {', '.join(risks)}")

                    # 详细策略
                    detailed = self._get_strategy_interventions(indicator, record.abnormality)
                    if detailed:
                        lines.append("**详细策略**:")
                        for d in detailed[:3]:
                            if d['type'] == 'immediate':
                                lines.append(f"  - ⚡ {d['action']}")
                                if 'details' in d and 'drugs' in d['details']:
                                    lines.append(f"    药物: {', '.join(d['details']['drugs'])}")
                    lines.append("")

            lines.append("=" * 70)
            lines.append("")

        # 详细警报
        if self.baseline_data.alerts:
            lines.append("## 警报详情\n")
            for alert in self.baseline_data.alerts:
                icon = "🔴" if alert['level'] == AlertLevel.CRITICAL else "🟡"
                lines.append(f"{icon} **{alert['indicator']}** = {alert['value']}")
                lines.append(f"   {alert['message']}")
                lines.append(f"   建议: {alert['action']}")
                lines.append("")

        # 干预建议
        if self.baseline_data.interventions:
            lines.append("## 干预建议\n")
            for intervention in self.baseline_data.interventions:
                lines.append(f"### {intervention['indicator']} - {intervention['abnormality']}")
                lines.append(f"理由: {intervention['rationale']}")
                lines.append("")

                if intervention['detailed_strategies']:
                    lines.append("**策略:**")
                    for s in intervention['detailed_strategies']:
                        prefix = "⚡" if s['type'] == 'immediate' else "⬆️"
                        lines.append(f"  {prefix} {s['action']}")
                lines.append("")

        # 所有指标值
        lines.append("## 所有测量值\n")
        lines.append("| 指标 | 值 | 状态 |")
        lines.append("|------|-----|------|")
        for indicator, value in self.baseline_data.measurements.items():
            eval_result = self.baseline_data.evaluations.get(indicator)
            status = eval_result.result.value if eval_result else "unknown"
            lines.append(f"| {indicator} | {value} | {status} |")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def generate_time_window_report(self, time_minutes: int) -> str:
        """生成时间窗报告"""
        if time_minutes not in self.time_window_history:
            return f"无t={time_minutes}min的数据"

        window = self.time_window_history[time_minutes]
        baseline = self.baseline_data

        lines = [
            "=" * 70,
            f"心脏灌注监测 - 时间窗报告 (t = {time_minutes} min)",
            f"生成时间: {window.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]

        # 状态汇总
        critical_count = sum(1 for a in window.alerts if a['level'] == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in window.alerts if a['level'] == AlertLevel.WARNING)

        lines.append("## 当前状态")
        lines.append(f"🔴 危急: {critical_count} | 🟡 警告: {warning_count}")
        lines.append("")

        # 与baseline对比
        lines.append("## Baseline对比 (t=30min → t={}min)\n".format(time_minutes))
        lines.append("| 指标 | Baseline | 当前 | 变化 | 趋势 |")
        lines.append("|------|----------|------|------|------|")

        for indicator, current_value in window.measurements.items():
            baseline_value = baseline.measurements.get(indicator, "N/A") if baseline else "N/A"
            comparison = window.comparisons.get(indicator)

            if comparison:
                delta = f"{comparison.delta:+.2f}"
                trend_icon = {"improving": "📈", "stable": "➡️", "deteriorating": "📉", "unknown": "❓"}
                trend = trend_icon.get(comparison.trend.value, "❓")
            else:
                delta = "N/A"
                trend = "❓"

            lines.append(f"| {indicator} | {baseline_value} | {current_value} | {delta} | {trend} |")

        lines.append("")

        # 警报
        if window.alerts:
            lines.append("## 警报\n")
            for alert in window.alerts:
                icon = "🔴" if alert['level'] == AlertLevel.CRITICAL else "🟡"
                lines.append(f"{icon} **{alert['indicator']}**: {alert['message']}")
            lines.append("")

        # 干预建议
        if window.interventions:
            lines.append("## 干预建议\n")
            for intervention in window.interventions:
                lines.append(f"### {intervention['indicator']}")

                # 知识图谱推荐
                if intervention['kg_recommendations']:
                    first_checks = [i for i in intervention['kg_recommendations'] if i['urgency'] == 'first']
                    immediate = [i for i in intervention['kg_recommendations'] if i['urgency'] == 'immediate']

                    if first_checks:
                        lines.append("**首先检查:**")
                        for i in first_checks:
                            lines.append(f"  - {i['action']}")

                    if immediate:
                        lines.append("**立即干预:**")
                        for i in immediate:
                            lines.append(f"  - 🔴 {i['action']}")

                lines.append("")

        # 之前干预的效果评估
        if self.active_interventions:
            lines.append("## 干预效果评估\n")
            for indicator, record in self.active_interventions.items():
                status_icon = {
                    InterventionStatus.EFFECTIVE: "✅",
                    InterventionStatus.INEFFECTIVE: "❌",
                    InterventionStatus.IN_PROGRESS: "⏳",
                    InterventionStatus.PENDING: "🕐"
                }
                lines.append(f"{status_icon.get(record.status, '❓')} **{indicator}**: {record.action}")
                lines.append(f"   状态: {record.status.value}")
                if record.outcome:
                    lines.append(f"   结果: {record.outcome}")
            lines.append("")

        # 升级提示
        ineffective = [r for r in self.active_interventions.values()
                       if r.status == InterventionStatus.INEFFECTIVE]
        if ineffective:
            lines.append("## ⚠️ 需要升级策略\n")
            for record in ineffective:
                lines.append(f"**{record.indicator}**: {record.action} 无效")

                # 知识图谱搜索建议
                related = self.knowledge_graph.search_related_knowledge(record.abnormality)
                if related:
                    lines.append("知识图谱建议:")
                    for t in related[:3]:
                        lines.append(f"  - {t}")
                lines.append("")

        # 备注
        if window.notes:
            lines.append("## 备注\n")
            lines.append(window.notes)

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def generate_full_session_report(self) -> str:
        """生成完整会话报告"""
        lines = [
            "=" * 70,
            "心脏灌注监测 - 完整会话报告",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            ""
        ]

        # 时间线
        lines.append("## 监测时间线\n")
        all_times = sorted([self.baseline_data.time_minutes] + list(self.time_window_history.keys()))

        for t in all_times:
            if t == self.baseline_data.time_minutes:
                window = self.baseline_data
                label = "(Baseline)"
            else:
                window = self.time_window_history.get(t)
                label = ""

            if window:
                critical = sum(1 for a in window.alerts if a['level'] == AlertLevel.CRITICAL)
                warning = sum(1 for a in window.alerts if a['level'] == AlertLevel.WARNING)
                lines.append(f"### t = {t} min {label}")
                lines.append(f"状态: 🔴×{critical} 🟡×{warning}")
                if window.alerts:
                    for a in window.alerts[:3]:
                        lines.append(f"  - {a['indicator']}: {a['message'][:50]}...")
                lines.append("")

        # 干预历史
        if self.intervention_records:
            lines.append("## 干预历史\n")
            for record in self.intervention_records:
                lines.append(f"- **{record.indicator}** (t={record.time_initiated}min)")
                lines.append(f"  干预: {record.action}")
                lines.append(f"  状态: {record.status.value}")
                if record.outcome:
                    lines.append(f"  结果: {record.outcome}")
            lines.append("")

        # 最终状态
        if self.time_window_history:
            last_time = max(self.time_window_history.keys())
            last_window = self.time_window_history[last_time]

            lines.append(f"## 最终状态 (t={last_time}min)\n")
            for indicator, value in last_window.measurements.items():
                eval_result = last_window.evaluations.get(indicator)
                status = eval_result.result.value if eval_result else "unknown"
                lines.append(f"- {indicator}: {value} ({status})")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def record_intervention(self, indicator: str, abnormality: str, action: str, time_minutes: int):
        """记录执行的干预"""
        record = InterventionRecord(
            time_initiated=time_minutes,
            indicator=indicator,
            abnormality=abnormality,
            action=action,
            status=InterventionStatus.IN_PROGRESS,
            time_windows_active=[time_minutes]
        )

        self.intervention_records.append(record)
        self.active_interventions[indicator] = record


if __name__ == "__main__":
    # 测试示例
    monitor = PerfusionMonitor()

    print("=== 灌注监测测试 ===\n")

    # 设置baseline (t=30min)
    baseline_measurements = {
        "EF": 52,
        "CI": 2.4,
        "MAP": 68,
        "SvO2": 68,
        "Lactate": 2.0,
        "K_A": 4.2,
        "HR": 85,
        "PVR": 2.0,
        "CF": 600,
    }

    print("设置Baseline (t=30min)...")
    baseline_report = monitor.set_baseline(baseline_measurements)
    print(baseline_report)

    # 模拟t=60min (一些指标恶化)
    print("\n" + "=" * 70)
    print("处理 t=60min...")
    t60_measurements = {
        "EF": 48,      # 下降到灰区
        "CI": 2.1,     # 下降
        "MAP": 62,     # 下降
        "SvO2": 60,    # 警戒
        "Lactate": 3.5,  # 上升
        "K_A": 4.0,
        "HR": 95,
        "PVR": 2.3,
        "CF": 580,
    }

    monitor.process_time_window(t60_measurements, 60)
    print(monitor.generate_time_window_report(60))

    # 模拟记录干预
    monitor.record_intervention("CI", "CI_Low", "多巴酚丁胺 5 μg/kg/min", 60)
    monitor.record_intervention("Lactate", "Lactate_High", "优化灌注参数", 60)

    # 模拟t=120min (看干预效果)
    print("\n" + "=" * 70)
    print("处理 t=120min...")
    t120_measurements = {
        "EF": 50,      # 略改善
        "CI": 2.3,     # 改善
        "MAP": 66,     # 改善
        "SvO2": 64,    # 改善
        "Lactate": 3.0,  # 改善
        "K_A": 4.1,
        "HR": 88,
        "PVR": 2.1,
        "CF": 620,
    }

    monitor.process_time_window(t120_measurements, 120)
    print(monitor.generate_time_window_report(120))
