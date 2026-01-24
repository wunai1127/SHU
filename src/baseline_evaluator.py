"""
Baseline 评估器 - t=0 基准对比和趋势分析
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class Trend(Enum):
    """变化趋势"""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    UNKNOWN = "unknown"


class DeviationSeverity(Enum):
    """偏离严重程度"""
    CRITICAL = "critical"    # 严重偏离
    MODERATE = "moderate"    # 中度偏离
    MILD = "mild"           # 轻度偏离
    NORMAL = "normal"       # 正常范围


@dataclass
class BaselineComparison:
    """Baseline 对比结果"""
    indicator: str
    current_value: float
    baseline_value: float
    delta: float
    delta_percent: float
    trend: Trend
    deviation_severity: DeviationSeverity
    within_acceptable: bool
    message: str
    recommendation: str


@dataclass
class TimeSeriesPoint:
    """时序数据点"""
    timestamp: datetime
    value: float


class BaselineEvaluator:
    """Baseline 评估器"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化 Baseline 评估器

        Args:
            config_dir: 配置文件目录
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        self.baseline_config: Dict[str, Any] = {}
        self.dynamic_baselines: Dict[str, float] = {}  # 运行时动态baseline
        self.measurement_history: Dict[str, List[TimeSeriesPoint]] = {}

        self._load_config()

    def _load_config(self):
        """加载baseline配置"""
        baseline_path = self.config_dir / "baseline.yaml"
        if baseline_path.exists():
            with open(baseline_path, 'r', encoding='utf-8') as f:
                self.baseline_config = yaml.safe_load(f)

    def get_baseline(self, indicator: str) -> Optional[float]:
        """
        获取指标的baseline值

        Args:
            indicator: 指标名称

        Returns:
            baseline值，如果不存在返回None
        """
        # 优先检查动态baseline
        if indicator in self.dynamic_baselines:
            return self.dynamic_baselines[indicator]

        # 检查confirmed baselines
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        if indicator in confirmed:
            return confirmed[indicator].get('baseline_value')

        # 检查pending baselines
        pending = self.baseline_config.get('pending_baselines', {})
        if indicator in pending:
            config = pending[indicator]
            # 如果有provisional baseline，使用它
            if config.get('provisional_baseline') is not None:
                return config['provisional_baseline']

        return None

    def set_dynamic_baseline(self, indicator: str, value: float):
        """
        设置动态baseline（用于pending指标的首次测量）

        Args:
            indicator: 指标名称
            value: baseline值
        """
        self.dynamic_baselines[indicator] = value

    def record_measurement(self, indicator: str, value: float, timestamp: Optional[datetime] = None):
        """
        记录测量值（用于趋势分析）

        Args:
            indicator: 指标名称
            value: 测量值
            timestamp: 时间戳，默认当前时间
        """
        if timestamp is None:
            timestamp = datetime.now()

        if indicator not in self.measurement_history:
            self.measurement_history[indicator] = []
            # 如果是首次测量且无baseline，设置为动态baseline
            if self.get_baseline(indicator) is None:
                self.set_dynamic_baseline(indicator, value)

        self.measurement_history[indicator].append(
            TimeSeriesPoint(timestamp=timestamp, value=value)
        )

    def compare(self, indicator: str, current_value: float) -> BaselineComparison:
        """
        将当前值与baseline对比

        Args:
            indicator: 指标名称
            current_value: 当前值

        Returns:
            BaselineComparison 对比结果
        """
        baseline = self.get_baseline(indicator)

        # 如果没有baseline
        if baseline is None:
            # 设置为动态baseline
            self.set_dynamic_baseline(indicator, current_value)
            return BaselineComparison(
                indicator=indicator,
                current_value=current_value,
                baseline_value=current_value,
                delta=0,
                delta_percent=0,
                trend=Trend.UNKNOWN,
                deviation_severity=DeviationSeverity.NORMAL,
                within_acceptable=True,
                message=f"{indicator}: 首次测量，已设为动态baseline",
                recommendation="继续监测"
            )

        # 计算偏离
        delta = current_value - baseline
        delta_percent = (delta / baseline * 100) if baseline != 0 else 0

        # 获取可接受偏差
        acceptable_deviation = self._get_acceptable_deviation(indicator)

        # 判断是否在可接受范围内
        within_acceptable = abs(delta) <= acceptable_deviation if acceptable_deviation else True

        # 判断趋势（需要结合指标特性）
        trend = self._determine_trend(indicator, delta)

        # 判断偏离严重程度
        deviation_severity = self._determine_severity(indicator, delta, acceptable_deviation)

        # 生成消息和建议
        message, recommendation = self._generate_message(
            indicator, current_value, baseline, delta, delta_percent,
            trend, deviation_severity, within_acceptable
        )

        return BaselineComparison(
            indicator=indicator,
            current_value=current_value,
            baseline_value=baseline,
            delta=round(delta, 3),
            delta_percent=round(delta_percent, 2),
            trend=trend,
            deviation_severity=deviation_severity,
            within_acceptable=within_acceptable,
            message=message,
            recommendation=recommendation
        )

    def _get_acceptable_deviation(self, indicator: str) -> Optional[float]:
        """获取可接受偏差"""
        confirmed = self.baseline_config.get('confirmed_baselines', {})
        if indicator in confirmed:
            return confirmed[indicator].get('acceptable_deviation')
        return None

    def _determine_trend(self, indicator: str, delta: float) -> Trend:
        """
        判断变化趋势

        Args:
            indicator: 指标名称
            delta: 变化量

        Returns:
            Trend 趋势
        """
        # 定义哪些指标增加是有利的，哪些是不利的
        higher_is_better = {'EF', 'CI', 'SvO2', 'pO2V', 'CF', 'MAP', 'AOP'}
        lower_is_better = {'Lactate', 'PVR', 'chest_drainage'}

        # 设置一个小阈值来判断稳定
        stability_threshold = 0.05  # 5%变化视为稳定

        if indicator in higher_is_better:
            if delta > stability_threshold * abs(self.get_baseline(indicator) or 1):
                return Trend.IMPROVING
            elif delta < -stability_threshold * abs(self.get_baseline(indicator) or 1):
                return Trend.DETERIORATING
            else:
                return Trend.STABLE

        elif indicator in lower_is_better:
            if delta < -stability_threshold * abs(self.get_baseline(indicator) or 1):
                return Trend.IMPROVING
            elif delta > stability_threshold * abs(self.get_baseline(indicator) or 1):
                return Trend.DETERIORATING
            else:
                return Trend.STABLE

        else:
            # 未知指标，只判断是否稳定
            if abs(delta) < stability_threshold * abs(self.get_baseline(indicator) or 1):
                return Trend.STABLE
            return Trend.UNKNOWN

    def _determine_severity(
        self,
        indicator: str,
        delta: float,
        acceptable_deviation: Optional[float]
    ) -> DeviationSeverity:
        """判断偏离严重程度"""
        if acceptable_deviation is None:
            return DeviationSeverity.NORMAL

        abs_delta = abs(delta)

        if abs_delta <= acceptable_deviation:
            return DeviationSeverity.NORMAL
        elif abs_delta <= acceptable_deviation * 2:
            return DeviationSeverity.MILD
        elif abs_delta <= acceptable_deviation * 3:
            return DeviationSeverity.MODERATE
        else:
            return DeviationSeverity.CRITICAL

    def _generate_message(
        self,
        indicator: str,
        current: float,
        baseline: float,
        delta: float,
        delta_pct: float,
        trend: Trend,
        severity: DeviationSeverity,
        within_acceptable: bool
    ) -> tuple:
        """生成消息和建议"""
        direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"

        message = (
            f"{indicator}: {current} (baseline: {baseline}, "
            f"Δ: {delta:+.2f} [{delta_pct:+.1f}%] {direction})"
        )

        if within_acceptable:
            recommendation = "维持监测"
        elif severity == DeviationSeverity.MILD:
            recommendation = "关注趋势变化"
        elif severity == DeviationSeverity.MODERATE:
            recommendation = "评估原因，考虑干预"
        else:  # CRITICAL
            recommendation = "需要立即评估和干预"

        # 根据趋势调整建议
        if trend == Trend.IMPROVING:
            recommendation += "（趋势向好）"
        elif trend == Trend.DETERIORATING:
            recommendation += "（趋势恶化）"

        return message, recommendation

    def get_rate_of_change(
        self,
        indicator: str,
        hours: float = 1.0
    ) -> Optional[float]:
        """
        计算指标的变化率（每小时）

        Args:
            indicator: 指标名称
            hours: 计算时间窗口（小时）

        Returns:
            每小时变化率，如果数据不足返回None
        """
        if indicator not in self.measurement_history:
            return None

        history = self.measurement_history[indicator]
        if len(history) < 2:
            return None

        # 获取最近的数据点
        latest = history[-1]
        # 找到指定时间窗口之前的数据点
        target_time = latest.timestamp.timestamp() - hours * 3600

        earlier_point = None
        for point in reversed(history[:-1]):
            if point.timestamp.timestamp() <= target_time:
                earlier_point = point
                break

        if earlier_point is None:
            earlier_point = history[0]

        # 计算变化率
        time_diff_hours = (latest.timestamp - earlier_point.timestamp).total_seconds() / 3600
        if time_diff_hours == 0:
            return None

        value_diff = latest.value - earlier_point.value
        return value_diff / time_diff_hours

    def generate_report(self, measurements: Dict[str, float]) -> str:
        """
        生成baseline对比报告

        Args:
            measurements: 当前测量值字典 {指标名: 值}

        Returns:
            格式化的报告字符串
        """
        report_lines = [
            "=" * 60,
            "Baseline 对比报告",
            f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            ""
        ]

        # 按严重程度分组
        critical = []
        warning = []
        normal = []

        for indicator, value in measurements.items():
            comparison = self.compare(indicator, value)

            if comparison.deviation_severity == DeviationSeverity.CRITICAL:
                critical.append(comparison)
            elif comparison.deviation_severity in [DeviationSeverity.MODERATE, DeviationSeverity.MILD]:
                warning.append(comparison)
            else:
                normal.append(comparison)

        # 输出严重偏离
        if critical:
            report_lines.append("🔴 严重偏离 (需立即关注):")
            report_lines.append("-" * 40)
            for c in critical:
                report_lines.append(f"  {c.message}")
                report_lines.append(f"    → {c.recommendation}")
            report_lines.append("")

        # 输出警告
        if warning:
            report_lines.append("🟡 警告 (需关注):")
            report_lines.append("-" * 40)
            for c in warning:
                report_lines.append(f"  {c.message}")
                report_lines.append(f"    → {c.recommendation}")
            report_lines.append("")

        # 输出正常
        if normal:
            report_lines.append("🟢 正常范围:")
            report_lines.append("-" * 40)
            for c in normal:
                report_lines.append(f"  {c.message}")
            report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


if __name__ == "__main__":
    # 测试示例
    evaluator = BaselineEvaluator()

    print("=== Baseline 对比测试 ===\n")

    # 模拟测量数据
    measurements = {
        "EF": 52,        # 接近baseline
        "CI": 2.1,       # 略低于baseline
        "Lactate": 3.5,  # 高于baseline
        "SvO2": 62,      # 低于baseline
        "HR": 95,        # 高于baseline
        "MAP": 68,       # 正常
        "PVR": 2.8,      # 高于baseline
        "Emax": 150,     # pending指标，首次测量
    }

    # 生成报告
    report = evaluator.generate_report(measurements)
    print(report)

    # 测试单个指标对比
    print("\n=== 单指标详细对比 ===\n")
    for indicator, value in measurements.items():
        comparison = evaluator.compare(indicator, value)
        print(f"{indicator}:")
        print(f"  当前值: {comparison.current_value}")
        print(f"  基线值: {comparison.baseline_value}")
        print(f"  变化量: {comparison.delta} ({comparison.delta_percent}%)")
        print(f"  趋势: {comparison.trend.value}")
        print(f"  严重程度: {comparison.deviation_severity.value}")
        print(f"  建议: {comparison.recommendation}")
        print()
