"""
CommunicationAgent - 交互层
============================
工具:
  1. alert_formatter   - 格式化告警消息（中文/分级/颜色）
  2. voice_broadcaster - 语音播报文本生成（TTS文本）
  3. report_formatter  - 完整报告格式化（Markdown/HTML/JSON）
  4. qa_responder      - 问答响应（预构建QA数据库）

职责: 接收PatientState → 格式化展示 → 生成语音播报文本 → 输出给前端
"""

import json
import logging
import time
from typing import Any, Dict, List

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool,
    AgentMessage, AlertEvent, AlertLevel, StrategyRecommendation, Priority
)

logger = logging.getLogger(__name__)


class CommunicationAgent(BaseAgent):
    """
    交互层Agent - UI渲染/语音/告警

    Tools:
      - alert_formatter: 告警消息格式化
      - voice_broadcaster: TTS播报文本生成
      - report_formatter: 报告格式化(Markdown)
      - qa_responder: 临床问答
    """

    def __init__(self, bus: EventBus, state: PatientState):
        super().__init__("communication", bus, state)
        self._pending_alerts: List[Dict] = []
        self._pending_voice: List[str] = []
        self.setup_tools()

    def setup_tools(self):
        self.register_tool(AgentTool(
            name="alert_formatter",
            description="格式化告警消息: AlertEvent → 中文分级告警文本 (含颜色标记)",
            func=self._format_alerts
        ))
        self.register_tool(AgentTool(
            name="voice_broadcaster",
            description="生成语音播报文本: 策略推荐 → 中文TTS文本(含药物剂量)",
            func=self._generate_voice_text
        ))
        self.register_tool(AgentTool(
            name="report_formatter",
            description="完整报告格式化: PatientState → Markdown格式报告",
            func=self._format_report
        ))
        self.register_tool(AgentTool(
            name="qa_responder",
            description="临床问答: 输入问题 → 搜索PatientState和策略 → 返回答案",
            func=self._answer_question
        ))

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _format_alerts(self, alerts: List[AlertEvent]) -> List[Dict]:
        """工具: 格式化告警"""
        level_config = {
            AlertLevel.CRITICAL: {"emoji": "🔴", "color": "#ff4d4f", "prefix": "危急"},
            AlertLevel.WARNING: {"emoji": "🟡", "color": "#faad14", "prefix": "警告"},
            AlertLevel.INFO: {"emoji": "🔵", "color": "#1890ff", "prefix": "提示"},
            AlertLevel.NORMAL: {"emoji": "🟢", "color": "#52c41a", "prefix": "正常"},
        }

        formatted = []
        for alert in alerts:
            config = level_config.get(alert.level, level_config[AlertLevel.INFO])
            formatted.append({
                "indicator": alert.indicator,
                "value": alert.value,
                "level": alert.level.value,
                "emoji": config["emoji"],
                "color": config["color"],
                "title": f"{config['prefix']}: {alert.indicator} = {alert.value} {alert.unit}",
                "detail": alert.message,
                "trend": alert.trend,
                "deviation": alert.deviation,
            })

        # 按严重程度排序
        order = {AlertLevel.CRITICAL: 0, AlertLevel.WARNING: 1, AlertLevel.INFO: 2}
        formatted.sort(key=lambda x: order.get(AlertLevel(x["level"]), 3))

        return formatted

    def _generate_voice_text(self, state: PatientState = None) -> List[str]:
        """工具: 生成TTS播报文本"""
        if state is None:
            state = self.state

        texts = []

        # 1. 风险概览
        alert_count = len(state.alerts)
        critical_count = sum(1 for a in state.alerts if a.level == AlertLevel.CRITICAL)
        if critical_count > 0:
            texts.append(f"注意: 检测到 {critical_count} 项危急指标, 共 {alert_count} 项异常。")
        elif alert_count > 0:
            texts.append(f"提示: 检测到 {alert_count} 项指标异常。")
        else:
            texts.append("所有监测指标在正常范围内。")
            return texts

        # 2. 逐项播报critical
        for alert in state.alerts:
            if alert.level == AlertLevel.CRITICAL:
                texts.append(
                    f"危急: {alert.indicator} 当前值 {alert.value}, "
                    f"{'偏高' if alert.direction == 'high' else '偏低'}, "
                    f"趋势 {'恶化' if alert.trend == 'deteriorating' else '稳定' if alert.trend == 'stable' else '改善'}。"
                )

        # 3. 策略播报
        for strategy in state.strategies[:3]:
            text = f"建议: {strategy.action}"
            if strategy.drug:
                text += f", 药物 {strategy.drug}"
            if strategy.dose:
                text += f", 剂量 {strategy.dose}"
            texts.append(text + "。")

        # 4. 安全警告
        if state.drug_conflicts:
            texts.append(f"安全警告: {', '.join(state.drug_conflicts)}。")

        return texts

    def _format_report(self, state: PatientState = None) -> str:
        """工具: Markdown格式报告"""
        if state is None:
            state = self.state

        lines = []
        lines.append(f"# HTTG 灌注监测报告")
        lines.append(f"**样本**: {state.sample_id} | **时间点**: t={state.timestamp_min}min | **风险**: {state.risk_level}")
        lines.append("")

        # 异常指标
        if state.alerts:
            lines.append("## 异常指标")
            for alert in state.alerts:
                emoji = "🔴" if alert.level == AlertLevel.CRITICAL else "🟡"
                lines.append(f"- {emoji} **{alert.indicator}** = {alert.value} {alert.unit} "
                             f"({alert.direction}) [{alert.trend}]")
            lines.append("")

        # 诊断
        if state.diagnoses:
            lines.append("## 诊断分析")
            for diag in state.diagnoses:
                lines.append(f"- **{diag.primary_cause}**")
                if diag.causal_chain:
                    for chain in diag.causal_chain[:3]:
                        lines.append(f"  - 因果链: {chain}")
                if diag.upstream_setpoints:
                    lines.append(f"  - 可调控Setpoint: {', '.join(diag.upstream_setpoints)}")
            lines.append("")

        # 策略推荐
        if state.strategies:
            lines.append("## 策略推荐")
            for i, strat in enumerate(state.strategies, 1):
                lines.append(f"### [{i}] {strat.indicator}: {strat.action}")
                if strat.drug:
                    lines.append(f"- 药物: {strat.drug}")
                if strat.dose:
                    lines.append(f"- 剂量: {strat.dose}")
                if strat.reasoning_chain:
                    lines.append("- CoT推理:")
                    for step in strat.reasoning_chain:
                        lines.append(f"  - {step}")
                lines.append("")

        # 安全信息
        if state.safety_warnings or state.drug_conflicts:
            lines.append("## 安全警告")
            for w in state.safety_warnings:
                lines.append(f"- {w}")
            for c in state.drug_conflicts:
                lines.append(f"- 药物冲突: {c}")

        return "\n".join(lines)

    def _answer_question(self, question: str) -> str:
        """工具: 临床问答"""
        q_lower = question.lower()
        state = self.state

        # 问当前指标状态
        for indicator in state.measurements:
            if indicator.lower() in q_lower:
                value = state.measurements[indicator]
                alert = next((a for a in state.alerts if a.indicator == indicator), None)
                if alert:
                    answer = (f"{indicator} 当前值 {value}, "
                              f"状态: {alert.level.value}, "
                              f"趋势: {alert.trend}")
                    # 找对应策略
                    strat = next((s for s in state.strategies if s.indicator == indicator), None)
                    if strat:
                        answer += f"\n建议: {strat.action}"
                        if strat.drug:
                            answer += f", 药物: {strat.drug}, 剂量: {strat.dose}"
                    return answer
                else:
                    return f"{indicator} 当前值 {value}, 在正常范围内"

        # 问风险级别
        if "风险" in question or "risk" in q_lower:
            return f"当前风险等级: {state.risk_level}, 异常指标 {len(state.alerts)} 项"

        # 问策略/下一步
        if "策略" in question or "怎么办" in question or "建议" in question or "下一步" in question:
            if state.strategies:
                top = state.strategies[0]
                return f"首要建议: {top.action}" + (f", 药物 {top.drug} {top.dose}" if top.drug else "")
            return "当前无需干预"

        return f"当前状态: 风险{state.risk_level}, {len(state.alerts)}项异常, {len(state.strategies)}条策略"

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        if msg.msg_type == "display":
            # Coordinator发来完整状态，生成展示数据
            report = self.use_tool("report_formatter")
            voice = self.use_tool("voice_broadcaster")
            alerts = self.use_tool("alert_formatter", self.state.alerts)

            display_data = {
                "report_md": report.data if report.success else "",
                "voice_texts": voice.data if voice.success else [],
                "formatted_alerts": alerts.data if alerts.success else [],
            }
            self._pending_alerts = display_data.get("formatted_alerts", [])
            self._pending_voice = display_data.get("voice_texts", [])

            self.log(f"展示数据生成: {len(self._pending_alerts)} 条告警, {len(self._pending_voice)} 条播报")
            await self.respond(msg, display_data)

        elif msg.msg_type == "critical_alert":
            # 紧急告警广播 → 立即生成语音
            critical_alerts = msg.payload
            if isinstance(critical_alerts, list):
                for alert in critical_alerts:
                    self._pending_voice.append(
                        f"紧急: {alert.indicator} = {alert.value}, {alert.level.value}!"
                    )

        elif msg.msg_type == "question":
            answer = self.use_tool("qa_responder", msg.payload)
            await self.respond(msg, answer.data)

    # -------------------------------------------------------------------------
    # 前端访问接口
    # -------------------------------------------------------------------------

    def get_pending_alerts(self) -> List[Dict]:
        return self._pending_alerts

    def get_pending_voice(self) -> List[str]:
        return self._pending_voice

    def get_report_sync(self) -> str:
        result = self.use_tool("report_formatter")
        return result.data if result.success else ""

    def ask_sync(self, question: str) -> str:
        result = self.use_tool("qa_responder", question)
        return result.data if result.success else "无法回答"
