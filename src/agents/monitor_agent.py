"""
MonitorAgent - 感知层
=====================
工具:
  1. threshold_checker  - BaselineThresholds: 阈值检测
  2. baseline_evaluator - BaselineEvaluator: baseline对比+趋势
  3. threshold_manager  - ThresholdManager: 阈值评估分级

职责: 接收测量数据 → 检测异常 → 评估趋势 → 发出AlertEvent
"""

import logging
from pathlib import Path
from typing import Dict, List

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool, ToolResult,
    AgentMessage, AlertEvent, AlertLevel, Priority
)

logger = logging.getLogger(__name__)


class MonitorAgent(BaseAgent):
    """
    感知层Agent - 实时阈值监测 + 趋势评估

    Tools:
      - threshold_checker: 检查单个指标是否超阈值
      - baseline_comparator: 与baseline对比计算偏离度和趋势
      - batch_evaluator: 批量评估所有指标
    """

    def __init__(self, bus: EventBus, state: PatientState, config_dir: str = None):
        super().__init__("monitor", bus, state)
        self._config_dir = config_dir or str(Path(__file__).parent.parent.parent / "config")
        self._bt = None     # BaselineThresholds instance
        self._be = None     # BaselineEvaluator instance
        self._tm = None     # ThresholdManager instance
        self.setup_tools()

    def setup_tools(self):
        """初始化并注册工具"""
        # 延迟导入，避免循环依赖
        try:
            from baseline_thresholds import BaselineThresholds
            self._bt = BaselineThresholds(config_dir=self._config_dir)
            self.register_tool(AgentTool(
                name="threshold_checker",
                description="检查指标是否超越阈值（红线/警告/危急），返回AlertLevel和偏离信息",
                func=self._check_threshold
            ))
            self.register_tool(AgentTool(
                name="batch_alert_scan",
                description="批量扫描所有指标，返回异常告警列表",
                func=self._batch_scan
            ))
        except ImportError as e:
            logger.warning(f"MonitorAgent: BaselineThresholds not available: {e}")

        try:
            from baseline_evaluator import BaselineEvaluator
            self._be = BaselineEvaluator(config_dir=self._config_dir)
            self.register_tool(AgentTool(
                name="baseline_comparator",
                description="与t=0 baseline对比，计算偏离度、趋势方向（improving/stable/deteriorating）",
                func=self._compare_baseline
            ))
        except ImportError as e:
            logger.warning(f"MonitorAgent: BaselineEvaluator not available: {e}")

        try:
            from threshold_manager import ThresholdManager
            self._tm = ThresholdManager(config_dir=self._config_dir)
            self.register_tool(AgentTool(
                name="threshold_evaluator",
                description="对指标进行accept/reject/warning/red_line分级评估",
                func=self._evaluate_threshold
            ))
        except ImportError as e:
            logger.warning(f"MonitorAgent: ThresholdManager not available: {e}")

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _check_threshold(self, indicator: str, value: float) -> Dict:
        """工具: 单指标阈值检查"""
        if not self._bt:
            return {"error": "BaselineThresholds not loaded"}
        result = self._bt.check_threshold(indicator, value)
        if result is None:
            return {"indicator": indicator, "value": value, "level": "unknown"}
        return {
            "indicator": indicator,
            "value": value,
            "level": result.alert_level.value if hasattr(result.alert_level, 'value') else str(result.alert_level),
            "deviation": getattr(result, 'deviation', 0),
            "trend": getattr(result, 'trend', 'unknown'),
            "baseline": getattr(result, 'baseline_value', None),
        }

    def _compare_baseline(self, indicator: str, value: float) -> Dict:
        """工具: baseline对比"""
        if not self._be:
            return {"error": "BaselineEvaluator not loaded"}
        comparison = self._be.compare(indicator, value)
        if comparison is None:
            return {"indicator": indicator, "value": value, "baseline": None}
        return {
            "indicator": indicator,
            "value": value,
            "baseline": comparison.baseline_value,
            "delta": comparison.delta,
            "trend": comparison.trend.value if hasattr(comparison.trend, 'value') else str(comparison.trend),
            "severity": comparison.severity.value if hasattr(comparison.severity, 'value') else str(comparison.severity),
        }

    def _evaluate_threshold(self, indicator: str, value: float) -> Dict:
        """工具: 分级评估"""
        if not self._tm:
            return {"error": "ThresholdManager not loaded"}
        evaluation = self._tm.evaluate(indicator, value)
        if evaluation is None:
            return {"indicator": indicator, "value": value, "result": "unknown"}
        return {
            "indicator": indicator,
            "value": value,
            "result": evaluation.result.value if hasattr(evaluation.result, 'value') else str(evaluation.result),
            "confidence": evaluation.confidence.value if hasattr(evaluation.confidence, 'value') else str(evaluation.confidence),
            "message": evaluation.message,
        }

    def _batch_scan(self, measurements: Dict[str, float]) -> List[Dict]:
        """工具: 批量扫描"""
        if not self._bt:
            return []
        results = self._bt.check_all_indicators(measurements)
        alerts = []
        for r in results:
            level = getattr(r, 'alert_level', None)
            if level and str(level) != 'normal':
                alerts.append({
                    "indicator": r.indicator,
                    "value": r.value,
                    "level": level.value if hasattr(level, 'value') else str(level),
                    "deviation": getattr(r, 'deviation', 0),
                    "trend": getattr(r, 'trend', 'unknown'),
                })
        return alerts

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        """处理消息"""
        if msg.msg_type == "analyze":
            measurements = msg.payload
            if isinstance(measurements, dict):
                alerts = await self._analyze_all(measurements)
                await self.respond(msg, alerts)

        elif msg.msg_type == "check_single":
            indicator = msg.payload.get("indicator")
            value = msg.payload.get("value")
            result = self.use_tool("threshold_checker", indicator, value)
            await self.respond(msg, result.data)

    async def _analyze_all(self, measurements: Dict[str, float]) -> List[AlertEvent]:
        """分析所有指标，生成AlertEvent列表"""
        alerts = []
        self.log(f"开始扫描 {len(measurements)} 项指标")

        for indicator, value in measurements.items():
            if value is None:
                continue

            # 工具1: 阈值检查
            check_result = self.use_tool("threshold_checker", indicator, value)
            level_str = "normal"
            if check_result.success and check_result.data:
                level_str = check_result.data.get("level", "normal")

            # 工具2: baseline对比
            deviation = 0.0
            trend = "unknown"
            if "baseline_comparator" in self.tools:
                bl_result = self.use_tool("baseline_comparator", indicator, value)
                if bl_result.success and bl_result.data:
                    deviation = bl_result.data.get("delta", 0.0) or 0.0
                    trend = bl_result.data.get("trend", "unknown")

            # 判断告警级别
            level_map = {
                "critical": AlertLevel.CRITICAL,
                "red_line": AlertLevel.CRITICAL,
                "warning": AlertLevel.WARNING,
                "info": AlertLevel.INFO,
            }
            alert_level = level_map.get(level_str, None)

            if alert_level and alert_level != AlertLevel.NORMAL:
                # 确定方向
                threshold_val = 0.0
                if check_result.success and check_result.data:
                    baseline = check_result.data.get("baseline")
                    threshold_val = baseline or 0.0

                direction = "high" if deviation > 0 else "low" if deviation < 0 else "abnormal"

                alert = AlertEvent(
                    indicator=indicator,
                    value=value,
                    unit="",
                    level=alert_level,
                    direction=direction,
                    threshold=threshold_val,
                    deviation=deviation,
                    trend=trend,
                    message=f"{indicator}={value} [{level_str}] 偏离baseline {deviation:+.2f}"
                )
                alerts.append(alert)

        # 写入共享状态
        self.state.alerts = alerts
        self.log(f"扫描完成: {len(alerts)} 项异常 "
                 f"(critical={sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)}, "
                 f"warning={sum(1 for a in alerts if a.level == AlertLevel.WARNING)})")

        # 有critical则广播紧急告警
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        if critical_alerts:
            await self.send("*", "critical_alert", critical_alerts, priority=Priority.CRITICAL.value)

        return alerts

    # -------------------------------------------------------------------------
    # 同步接口（供非async环境调用）
    # -------------------------------------------------------------------------

    def analyze_sync(self, measurements: Dict[str, float]) -> List[AlertEvent]:
        """同步分析接口（给Streamlit等非async环境用）"""
        alerts = []
        for indicator, value in measurements.items():
            if value is None:
                continue
            check_result = self.use_tool("threshold_checker", indicator, value)
            if not check_result.success:
                continue

            level_str = check_result.data.get("level", "normal")
            if level_str in ("critical", "red_line", "warning"):
                deviation = 0.0
                trend = "unknown"
                if "baseline_comparator" in self.tools:
                    bl = self.use_tool("baseline_comparator", indicator, value)
                    if bl.success and bl.data:
                        deviation = bl.data.get("delta", 0) or 0
                        trend = bl.data.get("trend", "unknown")

                level_map = {"critical": AlertLevel.CRITICAL, "red_line": AlertLevel.CRITICAL,
                             "warning": AlertLevel.WARNING}
                alerts.append(AlertEvent(
                    indicator=indicator, value=value, unit="",
                    level=level_map.get(level_str, AlertLevel.WARNING),
                    direction="high" if deviation > 0 else "low",
                    threshold=0, deviation=deviation, trend=trend,
                    message=f"{indicator}={value} [{level_str}]"
                ))
        self.state.alerts = alerts
        return alerts
