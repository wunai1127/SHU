"""
DiagnosisAgent - 分析层
========================
工具:
  1. indicator_classifier - IndicatorManager: 指标分类(Setpoint/Readout/Injury)
  2. causal_tracer        - 因果关系图追踪: 从异常Readout追溯到上游Setpoint
  3. root_cause_analyzer  - 根因分析: 综合多指标异常推断根本原因

职责: 接收AlertEvent → 分类指标 → 追踪因果链 → 识别可调控Setpoint → 输出DiagnosisResult
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool,
    AgentMessage, AlertEvent, DiagnosisResult, Priority
)

logger = logging.getLogger(__name__)


class DiagnosisAgent(BaseAgent):
    """
    分析层Agent - 因果推理 + 根因分析

    Tools:
      - indicator_classifier: 判断指标类型(Setpoint/Readout/Injury Marker)
      - causal_tracer: 追踪因果链 (Readout → 哪些Setpoint影响它)
      - root_cause_analyzer: 多指标联合分析推断根因
    """

    def __init__(self, bus: EventBus, state: PatientState, config_dir: str = None):
        super().__init__("diagnosis", bus, state)
        self._config_dir = config_dir or str(Path(__file__).parent.parent.parent / "config")
        self._im = None  # IndicatorManager
        self.setup_tools()

    def setup_tools(self):
        try:
            from indicator_manager import IndicatorManager
            self._im = IndicatorManager(
                config_path=str(Path(self._config_dir) / "indicator_classification.yaml")
            )
            self.register_tool(AgentTool(
                name="indicator_classifier",
                description="判断指标类型: setpoint(可直接调控) / readout(间接) / injury_marker(仅监测)",
                func=self._classify_indicator
            ))
            self.register_tool(AgentTool(
                name="causal_tracer",
                description="追踪因果链: 异常Readout → 受哪些Setpoint影响 → 调控建议",
                func=self._trace_causal_chain
            ))
            self.register_tool(AgentTool(
                name="root_cause_analyzer",
                description="多指标联合分析: 从多个异常中推断共同根本原因",
                func=self._analyze_root_cause
            ))
            self.register_tool(AgentTool(
                name="adjustment_recommender",
                description="给出Setpoint调整方向建议（如pH需上调/下调）",
                func=self._get_adjustments
            ))
        except ImportError as e:
            logger.warning(f"DiagnosisAgent: IndicatorManager not available: {e}")

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _classify_indicator(self, indicator: str) -> Dict:
        """工具: 分类指标"""
        if not self._im:
            return {"indicator": indicator, "type": "unknown"}

        ind_type = self._im.get_indicator_type(indicator)
        result = {"indicator": indicator, "type": ind_type.value if ind_type else "unknown"}

        if ind_type and ind_type.value == "readout":
            readout = self._im.readouts.get(indicator)
            if readout:
                result["influenced_by"] = readout.influenced_by
                result["risk_threshold"] = readout.risk_threshold
                result["risk_direction"] = readout.risk_direction.value
        elif ind_type and ind_type.value == "setpoint":
            sp = self._im.setpoints.get(indicator)
            if sp:
                result["target_range"] = sp.target_range
                result["control_method"] = sp.control_method
        elif ind_type and ind_type.value == "injury_marker":
            marker = self._im.injury_markers.get(indicator)
            if marker:
                result["interpretation"] = marker.interpretation
                result["trend_is_key"] = marker.trend_is_key

        return result

    def _trace_causal_chain(self, indicator: str) -> Dict:
        """工具: 追踪因果链"""
        if not self._im:
            return {"indicator": indicator, "chain": []}

        chain = []
        upstream_setpoints = []

        # 1. 确认指标类型
        ind_type = self._im.get_indicator_type(indicator)
        if not ind_type:
            return {"indicator": indicator, "chain": [], "type": "unknown"}

        # 2. 如果是Readout，追踪影响它的Setpoint
        if ind_type.value == "readout":
            readout = self._im.readouts.get(indicator)
            if readout and readout.influenced_by:
                upstream_setpoints = readout.influenced_by
                for sp in upstream_setpoints:
                    chain.append(f"{sp} → {indicator}")

        # 3. 搜索共识因果关系中相关的链
        for rel in self._im.causal_relationships:
            if rel.get("to") == indicator or rel.get("from") == indicator:
                chain.append(
                    f"{rel.get('from', '?')} → {rel.get('to', '?')}: {rel.get('effect', '')}"
                )

        return {
            "indicator": indicator,
            "type": ind_type.value,
            "upstream_setpoints": upstream_setpoints,
            "causal_chain": chain,
        }

    def _analyze_root_cause(self, alert_indicators: List[str]) -> Dict:
        """工具: 多指标联合根因分析"""
        if not self._im:
            return {"root_causes": [], "shared_setpoints": []}

        # 收集所有异常Readout的上游Setpoint
        all_upstreams = {}
        for indicator in alert_indicators:
            readout = self._im.readouts.get(indicator)
            if readout and readout.influenced_by:
                for sp in readout.influenced_by:
                    if sp not in all_upstreams:
                        all_upstreams[sp] = []
                    all_upstreams[sp].append(indicator)

        # 按影响的异常Readout数排序 → 影响最多的Setpoint最可能是根因
        shared = sorted(all_upstreams.items(), key=lambda x: len(x[1]), reverse=True)

        root_causes = []
        for setpoint, affected in shared:
            sp_info = self._im.setpoints.get(setpoint)
            root_causes.append({
                "setpoint": setpoint,
                "name": sp_info.name if sp_info else setpoint,
                "affected_readouts": affected,
                "affected_count": len(affected),
                "control_method": sp_info.control_method if sp_info else "unknown",
                "target_range": sp_info.target_range if sp_info else (0, 0),
            })

        # 搜索共识因果关系中匹配的
        consensus_causes = []
        for rel in self._im.causal_relationships:
            if rel.get("to") in alert_indicators or rel.get("from") in alert_indicators:
                consensus_causes.append(rel)

        return {
            "root_causes": root_causes,
            "shared_setpoints": [s for s, _ in shared],
            "consensus_causal_matches": consensus_causes,
        }

    def _get_adjustments(self, indicator: str, value: float) -> List[Dict]:
        """工具: Setpoint调整建议"""
        if not self._im:
            return []
        recs = self._im.get_adjustment_recommendations(indicator, value)
        return [
            {
                "target_setpoint": r.target_setpoint,
                "direction": r.adjustment_direction,
                "rationale": r.rationale,
                "priority": r.priority,
            }
            for r in recs
        ]

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        if msg.msg_type == "analyze":
            alerts = msg.payload
            if isinstance(alerts, list):
                diagnoses = self._diagnose_alerts(alerts)
                await self.respond(msg, diagnoses)

        elif msg.msg_type == "classify":
            indicator = msg.payload
            result = self.use_tool("indicator_classifier", indicator)
            await self.respond(msg, result.data)

        elif msg.msg_type == "trace_causal":
            indicator = msg.payload
            result = self.use_tool("causal_tracer", indicator)
            await self.respond(msg, result.data)

    def _diagnose_alerts(self, alerts: List[AlertEvent]) -> List[DiagnosisResult]:
        """诊断所有告警"""
        diagnoses = []
        alert_indicators = [a.indicator for a in alerts]

        self.log(f"诊断 {len(alerts)} 项异常")

        # 1. 逐个指标分析
        for alert in alerts:
            # 分类
            classify_result = self.use_tool("indicator_classifier", alert.indicator)
            ind_type = "unknown"
            if classify_result.success and classify_result.data:
                ind_type = classify_result.data.get("type", "unknown")

            # 因果追踪
            causal_result = self.use_tool("causal_tracer", alert.indicator)
            causal_chain = []
            upstream = []
            if causal_result.success and causal_result.data:
                causal_chain = causal_result.data.get("causal_chain", [])
                upstream = causal_result.data.get("upstream_setpoints", [])

            diagnoses.append(DiagnosisResult(
                primary_cause=f"{alert.indicator} {alert.direction} ({alert.level.value})",
                causal_chain=causal_chain,
                affected_indicators=[alert.indicator],
                indicator_type=ind_type,
                upstream_setpoints=upstream,
                confidence=0.8 if causal_chain else 0.5,
            ))

        # 2. 联合根因分析
        if len(alerts) > 1:
            root_result = self.use_tool("root_cause_analyzer", alert_indicators)
            if root_result.success and root_result.data:
                root_causes = root_result.data.get("root_causes", [])
                if root_causes:
                    top = root_causes[0]
                    self.log(f"根因分析: {top['setpoint']} 影响 {top['affected_count']} 项异常Readout")

        # 写入共享状态
        self.state.diagnoses = diagnoses
        return diagnoses

    # -------------------------------------------------------------------------
    # 同步接口
    # -------------------------------------------------------------------------

    def diagnose_sync(self, alerts: List[AlertEvent]) -> List[DiagnosisResult]:
        """同步诊断接口"""
        return self._diagnose_alerts(alerts)
