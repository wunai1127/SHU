"""
策略映射器 - 根据指标评估结果映射到临床决策策略
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from .threshold_manager import ThresholdManager, ThresholdEvaluation, EvaluationResult
from .baseline_evaluator import BaselineEvaluator, BaselineComparison, Trend


class ActionPriority(Enum):
    """操作优先级"""
    IMMEDIATE = 1      # 立即执行
    URGENT = 2         # 紧急
    ROUTINE = 3        # 常规
    MONITOR_ONLY = 4   # 仅监测


@dataclass
class StrategyAction:
    """策略操作"""
    indicator: str
    priority: ActionPriority
    action: str
    rationale: str
    escalation_needed: bool = False
    combined_triggers: List[str] = field(default_factory=list)


@dataclass
class ClinicalDecision:
    """临床决策结果"""
    timestamp: str
    overall_status: str
    primary_actions: List[StrategyAction]
    monitoring_only: List[str]
    escalation_protocol: Optional[str] = None
    summary: str = ""


class StrategyMapper:
    """策略映射器"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化策略映射器

        Args:
            config_dir: 配置文件目录
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        self.strategies: Dict[str, Any] = {}
        self.threshold_manager = ThresholdManager(config_dir)
        self.baseline_evaluator = BaselineEvaluator(config_dir)

        self._load_config()

    def _load_config(self):
        """加载策略配置"""
        strategies_path = self.config_dir / "strategies.yaml"
        if strategies_path.exists():
            with open(strategies_path, 'r', encoding='utf-8') as f:
                self.strategies = yaml.safe_load(f)

    def evaluate_all(self, measurements: Dict[str, float]) -> ClinicalDecision:
        """
        评估所有指标并生成临床决策

        Args:
            measurements: 测量值字典 {指标名: 值}

        Returns:
            ClinicalDecision 临床决策
        """
        from datetime import datetime

        primary_actions = []
        monitoring_only = []
        escalation_needed = False
        escalation_protocol = None

        # 收集所有评估结果
        threshold_results: Dict[str, ThresholdEvaluation] = {}
        baseline_results: Dict[str, BaselineComparison] = {}

        for indicator, value in measurements.items():
            threshold_results[indicator] = self.threshold_manager.evaluate(indicator, value)
            baseline_results[indicator] = self.baseline_evaluator.compare(indicator, value)

        # 根据评估结果映射策略
        for indicator, th_result in threshold_results.items():
            bl_result = baseline_results.get(indicator)

            # pending指标仅监测
            if th_result.result == EvaluationResult.PENDING:
                monitoring_only.append(indicator)
                continue

            # 获取策略配置
            strategy = self._get_indicator_strategy(indicator)
            action = self._map_to_action(indicator, th_result, bl_result, strategy)

            if action:
                primary_actions.append(action)
                if action.escalation_needed:
                    escalation_needed = True

        # 检查复合升级策略
        escalation_protocol = self._check_escalation_protocols(threshold_results, measurements)

        # 按优先级排序
        primary_actions.sort(key=lambda x: x.priority.value)

        # 生成总体状态
        if any(a.priority == ActionPriority.IMMEDIATE for a in primary_actions):
            overall_status = "CRITICAL - 需要立即干预"
        elif any(a.priority == ActionPriority.URGENT for a in primary_actions):
            overall_status = "WARNING - 需要关注"
        elif primary_actions:
            overall_status = "STABLE - 常规监测"
        else:
            overall_status = "NORMAL - 所有指标正常"

        # 生成摘要
        summary = self._generate_summary(primary_actions, monitoring_only, escalation_protocol)

        return ClinicalDecision(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            overall_status=overall_status,
            primary_actions=primary_actions,
            monitoring_only=monitoring_only,
            escalation_protocol=escalation_protocol,
            summary=summary
        )

    def _get_indicator_strategy(self, indicator: str) -> Optional[Dict[str, Any]]:
        """获取指标的策略配置"""
        indicator_strategies = self.strategies.get('indicator_strategies', {})
        return indicator_strategies.get(indicator)

    def _map_to_action(
        self,
        indicator: str,
        th_result: ThresholdEvaluation,
        bl_result: Optional[BaselineComparison],
        strategy: Optional[Dict[str, Any]]
    ) -> Optional[StrategyAction]:
        """
        将评估结果映射到策略操作

        Args:
            indicator: 指标名称
            th_result: 阈值评估结果
            bl_result: baseline对比结果
            strategy: 策略配置

        Returns:
            StrategyAction 或 None
        """
        # 确定优先级和操作
        if th_result.result in [EvaluationResult.REJECT, EvaluationResult.RED_LINE]:
            priority = ActionPriority.IMMEDIATE
            escalation = True
        elif th_result.result in [EvaluationResult.WARNING, EvaluationResult.GRAY_ZONE]:
            priority = ActionPriority.URGENT
            escalation = False
        elif th_result.result == EvaluationResult.NORMAL:
            # 正常情况下检查趋势
            if bl_result and bl_result.trend == Trend.DETERIORATING:
                priority = ActionPriority.ROUTINE
                escalation = False
            else:
                return None  # 不需要特殊操作
        else:
            return None

        # 从策略配置获取具体操作
        action_text = th_result.action or "评估并处理"
        rationale = th_result.message

        # 如果有策略配置，使用配置中的操作
        if strategy:
            level_key = self._result_to_level(th_result.result)
            if level_key and level_key in strategy:
                level_config = strategy[level_key]
                action_text = level_config.get('action', action_text)
                if 'escalation' in level_config:
                    escalation = level_config['escalation']

        # 结合baseline趋势信息
        if bl_result:
            trend_info = f" [趋势: {bl_result.trend.value}]"
            rationale += trend_info

        return StrategyAction(
            indicator=indicator,
            priority=priority,
            action=action_text,
            rationale=rationale,
            escalation_needed=escalation
        )

    def _result_to_level(self, result: EvaluationResult) -> Optional[str]:
        """将评估结果映射到策略级别"""
        mapping = {
            EvaluationResult.REJECT: 'red_line',
            EvaluationResult.RED_LINE: 'red_line',
            EvaluationResult.WARNING: 'warning',
            EvaluationResult.GRAY_ZONE: 'warning',
            EvaluationResult.NORMAL: 'normal',
            EvaluationResult.ACCEPT: 'normal',
        }
        return mapping.get(result)

    def _check_escalation_protocols(
        self,
        threshold_results: Dict[str, ThresholdEvaluation],
        measurements: Dict[str, float]
    ) -> Optional[str]:
        """
        检查是否触发复合升级策略

        Returns:
            触发的升级方案名称，或None
        """
        protocols = self.strategies.get('escalation_protocols', {})

        for protocol_name, protocol_config in protocols.items():
            triggers = protocol_config.get('triggers', [])
            triggered_count = 0

            for trigger in triggers:
                # 解析触发条件
                if self._check_trigger(trigger, measurements, threshold_results):
                    triggered_count += 1

            # 如果多个触发条件满足，返回该方案
            if triggered_count >= 2:  # 至少2个条件触发
                return protocol_config.get('name', protocol_name)

        return None

    def _check_trigger(
        self,
        trigger: str,
        measurements: Dict[str, float],
        threshold_results: Dict[str, ThresholdEvaluation]
    ) -> bool:
        """检查单个触发条件"""
        # 简单的条件解析
        # 格式如: "SvO2 < 50%", "CI < 2.0 L/min/m²", "MAP < 60 mmHg"
        pattern = r'(\w+)\s*([<>=]+)\s*([\d.]+)'
        match = re.match(pattern, trigger)

        if not match:
            return False

        indicator, operator, value_str = match.groups()
        threshold = float(value_str)

        if indicator not in measurements:
            return False

        current = measurements[indicator]

        if operator == '<':
            return current < threshold
        elif operator == '<=':
            return current <= threshold
        elif operator == '>':
            return current > threshold
        elif operator == '>=':
            return current >= threshold

        return False

    def _generate_summary(
        self,
        actions: List[StrategyAction],
        monitoring: List[str],
        escalation: Optional[str]
    ) -> str:
        """生成决策摘要"""
        lines = []

        if escalation:
            lines.append(f"⚠️ 触发升级方案: {escalation}")

        immediate = [a for a in actions if a.priority == ActionPriority.IMMEDIATE]
        urgent = [a for a in actions if a.priority == ActionPriority.URGENT]

        if immediate:
            lines.append(f"🔴 需立即处理: {len(immediate)} 项")
            for a in immediate:
                lines.append(f"   - {a.indicator}: {a.action}")

        if urgent:
            lines.append(f"🟡 需关注: {len(urgent)} 项")

        if monitoring:
            lines.append(f"⚪ 仅监测: {len(monitoring)} 项 ({', '.join(monitoring[:5])}{'...' if len(monitoring) > 5 else ''})")

        return "\n".join(lines)

    def generate_decision_report(self, measurements: Dict[str, float]) -> str:
        """
        生成完整的临床决策报告

        Args:
            measurements: 测量值字典

        Returns:
            格式化的报告字符串
        """
        decision = self.evaluate_all(measurements)

        report_lines = [
            "=" * 70,
            "临床决策支持报告",
            f"时间: {decision.timestamp}",
            f"状态: {decision.overall_status}",
            "=" * 70,
            "",
            decision.summary,
            "",
            "-" * 70,
            "详细操作建议:",
            "-" * 70,
        ]

        # 按优先级分组输出
        priority_groups = {
            ActionPriority.IMMEDIATE: ("🔴 立即执行", []),
            ActionPriority.URGENT: ("🟡 紧急处理", []),
            ActionPriority.ROUTINE: ("🟢 常规处理", []),
        }

        for action in decision.primary_actions:
            if action.priority in priority_groups:
                priority_groups[action.priority][1].append(action)

        for priority, (label, actions) in priority_groups.items():
            if actions:
                report_lines.append(f"\n{label}:")
                for a in actions:
                    report_lines.append(f"  [{a.indicator}]")
                    report_lines.append(f"    操作: {a.action}")
                    report_lines.append(f"    依据: {a.rationale}")
                    if a.escalation_needed:
                        report_lines.append(f"    ⚠️ 需要升级支持")

        # 输出仅监测的指标
        if decision.monitoring_only:
            report_lines.append(f"\n⚪ 仅监测 (无确认阈值):")
            for indicator in decision.monitoring_only:
                value = measurements.get(indicator, "N/A")
                report_lines.append(f"  - {indicator}: {value}")

        # 输出升级方案
        if decision.escalation_protocol:
            report_lines.append(f"\n" + "=" * 70)
            report_lines.append(f"⚠️ 触发升级方案: {decision.escalation_protocol}")
            report_lines.append(self._get_escalation_steps(decision.escalation_protocol))

        report_lines.append("\n" + "=" * 70)

        return "\n".join(report_lines)

    def _get_escalation_steps(self, protocol_name: str) -> str:
        """获取升级方案的步骤说明"""
        protocols = self.strategies.get('escalation_protocols', {})

        for name, config in protocols.items():
            if config.get('name') == protocol_name:
                steps = config.get('steps', {})
                lines = ["升级步骤:"]
                for step_num, step_info in sorted(steps.items()):
                    action = step_info.get('action', '')
                    condition = step_info.get('condition', '')
                    if condition:
                        lines.append(f"  {step_num}. [{condition}] {action}")
                    else:
                        lines.append(f"  {step_num}. {action}")
                return "\n".join(lines)

        return ""


def quick_decision(measurements: Dict[str, float], config_dir: Optional[str] = None) -> str:
    """
    快速生成临床决策报告

    Args:
        measurements: 测量值字典
        config_dir: 配置目录

    Returns:
        决策报告字符串
    """
    mapper = StrategyMapper(config_dir)
    return mapper.generate_decision_report(measurements)


if __name__ == "__main__":
    # 测试示例
    mapper = StrategyMapper()

    print("=== 临床决策支持测试 ===\n")

    # 模拟测量数据 - 包含一些异常值
    measurements = {
        # 心功能
        "EF": 48,        # 灰区
        "CI": 1.9,       # 红线！

        # 血压
        "MAP": 58,       # 红线！
        "AOP": 72,       # 正常

        # 氧代谢
        "SvO2": 52,      # 警戒
        "pO2V": 36,      # 正常（低信度）

        # 代谢
        "Lactate": 4.5,  # 警戒

        # 心率
        "HR": 115,       # 警戒范围

        # 肺血管
        "PVR": 2.2,      # 正常

        # 冠脉
        "CF": 550,       # 正常

        # Pending指标
        "Emax": 120,
        "max_dPdt": 1500,
        "GluA": 7.5,
    }

    # 生成决策报告
    report = mapper.generate_decision_report(measurements)
    print(report)
