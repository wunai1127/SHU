"""
CoordinatorAgent - 协调层
==========================
职责:
  1. 编排流程: 感知→分析→决策 pipeline
  2. 安全校验: 药物冲突检查、剂量上限、禁忌证
  3. 冲突仲裁: 多个Strategy Agent建议冲突时仲裁
  4. 状态管理: 维护PatientState全局一致性
  5. 降级策略: 当Agent不可用时graceful降级

特殊地位: 唯一可以写入 final_decision 的Agent
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool,
    AgentMessage, AlertEvent, AlertLevel, DiagnosisResult,
    StrategyRecommendation, Priority
)

logger = logging.getLogger(__name__)


# =============================================================================
# 药物冲突知识库
# =============================================================================

DRUG_CONFLICTS = [
    {
        "drugs": ["钙剂", "地高辛"],
        "warning": "钙剂与地高辛有相互作用风险，可致心律失常",
        "severity": "critical"
    },
    {
        "drugs": ["胰岛素", "补钾"],
        "warning": "胰岛素会导致钾转移入细胞，合用时需密切监测血钾",
        "severity": "warning"
    },
    {
        "drugs": ["他克莫司", "环孢素"],
        "warning": "两种CNI不应同时使用（共识）",
        "severity": "critical"
    },
    {
        "drugs": ["米力农", "去甲肾上腺素"],
        "warning": "米力农扩张血管可能对抗去甲肾上腺素升压效果，需仔细滴定",
        "severity": "warning"
    },
    {
        "drugs": ["硝普钠", "去甲肾上腺素"],
        "warning": "扩张剂与缩血管剂同用需仔细平衡",
        "severity": "info"
    },
]

# 剂量上限（共识来源）
DOSE_LIMITS = {
    "多巴胺": {"max": "20 μg/kg/min", "source": "共识：供者标准，>20排除"},
    "肾上腺素": {"max": "0.2 μg/kg/min", "source": "共识：供者标准"},
    "去甲肾上腺素": {"max": "0.4 μg/kg/min", "source": "共识：供者标准"},
    "异丙肾上腺素": {"max": "0.1 μg/kg/min", "source": "共识：心脏去神经后"},
    "米力农": {"max": "1 μg/kg/min", "source": "共识：肺血管扩张"},
}


class CoordinatorAgent(BaseAgent):
    """
    协调器Agent - 编排所有Agent、安全校验、最终决策

    Tools:
      - safety_checker: 药物冲突和禁忌检查
      - conflict_resolver: 策略冲突仲裁
      - risk_calculator: 综合风险等级计算
      - pipeline_orchestrator: 流程编排
    """

    def __init__(self, bus: EventBus, state: PatientState,
                 monitor=None, diagnosis=None, strategy=None,
                 knowledge=None, communication=None):
        super().__init__("coordinator", bus, state)
        self.monitor = monitor
        self.diagnosis = diagnosis
        self.strategy = strategy
        self.knowledge = knowledge
        self.communication = communication
        self.setup_tools()

    def setup_tools(self):
        self.register_tool(AgentTool(
            name="safety_checker",
            description="安全校验: 检查药物冲突、剂量上限、共识禁忌证",
            func=self._check_safety
        ))
        self.register_tool(AgentTool(
            name="conflict_resolver",
            description="冲突仲裁: 多个策略矛盾时选择最优方案",
            func=self._resolve_conflicts
        ))
        self.register_tool(AgentTool(
            name="risk_calculator",
            description="综合风险评估: 基于异常数量和严重程度计算风险等级",
            func=self._calculate_risk
        ))
        self.register_tool(AgentTool(
            name="pipeline_orchestrator",
            description="流程编排: 按序调用 Monitor→Diagnosis→Knowledge→Strategy→Communication",
            func=self._orchestrate_sync
        ))

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _check_safety(self, strategies: List[StrategyRecommendation]) -> Dict:
        """工具: 安全校验"""
        warnings = []
        conflicts = []

        # 收集所有推荐的药物
        recommended_drugs = []
        for strat in strategies:
            if strat.drug:
                recommended_drugs.append(strat.drug)

        # 1. 药物冲突检查
        for conflict_rule in DRUG_CONFLICTS:
            conflict_drugs = conflict_rule["drugs"]
            found = []
            for drug in recommended_drugs:
                for cd in conflict_drugs:
                    if cd in drug:
                        found.append(drug)
            if len(found) >= 2:
                conflicts.append({
                    "drugs": found,
                    "warning": conflict_rule["warning"],
                    "severity": conflict_rule["severity"],
                })

        # 2. 剂量上限检查
        for strat in strategies:
            if strat.drug:
                for drug_name, limit in DOSE_LIMITS.items():
                    if drug_name in strat.drug:
                        warnings.append(
                            f"{drug_name}上限: {limit['max']} ({limit['source']})"
                        )

        # 3. 共识禁忌检查
        measurements = self.state.measurements
        if measurements.get("PVR", 0) > 5.0:
            warnings.append("PVR > 5.0 Wood: 移植禁忌（共识）")
        if measurements.get("TPG", 0) > 15:
            warnings.append("TPG > 15 mmHg: 移植禁忌（共识）")
        if measurements.get("ColdIschemiaTime", 0) > 8:
            warnings.append("冷缺血时间 > 8h: 超出最长极限（共识）")

        return {
            "safe": len(conflicts) == 0,
            "conflicts": conflicts,
            "warnings": warnings,
            "drug_count": len(recommended_drugs),
        }

    def _resolve_conflicts(self, strategies: List[StrategyRecommendation]) -> List[StrategyRecommendation]:
        """工具: 冲突仲裁"""
        if len(strategies) <= 1:
            return strategies

        # 按confidence排序，高优先
        sorted_strategies = sorted(strategies, key=lambda s: s.confidence, reverse=True)

        # 检查目标冲突（如同时升压和降压）
        resolved = []
        seen_actions = set()

        for strat in sorted_strategies:
            action_key = f"{strat.indicator}_{strat.action[:10]}"
            if action_key not in seen_actions:
                resolved.append(strat)
                seen_actions.add(action_key)
            else:
                self.log(f"冲突仲裁: 移除重复策略 {strat.indicator} - {strat.action}")

        return resolved

    def _calculate_risk(self, alerts: List[AlertEvent]) -> str:
        """工具: 风险等级计算"""
        if not alerts:
            return "MINIMAL"

        critical_count = sum(1 for a in alerts if a.level == AlertLevel.CRITICAL)
        warning_count = sum(1 for a in alerts if a.level == AlertLevel.WARNING)

        # 关键指标加权
        critical_indicators = {"MAP", "CI", "K_A", "pH", "PVR", "PASP", "EF"}
        has_critical_indicator = any(
            a.indicator in critical_indicators and a.level == AlertLevel.CRITICAL
            for a in alerts
        )

        if critical_count >= 2 or (critical_count >= 1 and has_critical_indicator):
            return "HIGH"
        elif critical_count >= 1 or warning_count >= 3:
            return "MEDIUM"
        elif warning_count >= 1:
            return "LOW"
        return "MINIMAL"

    # -------------------------------------------------------------------------
    # 核心: Pipeline编排（同步版本，供Streamlit使用）
    # -------------------------------------------------------------------------

    def _orchestrate_sync(self, measurements: Dict[str, float],
                          sample_id: str = "unknown",
                          timestamp_min: int = 0) -> PatientState:
        """同步编排Pipeline: Monitor → Diagnosis → Knowledge → Strategy → Safety → Communication"""

        start_time = time.time()

        # 初始化状态
        self.state.sample_id = sample_id
        self.state.timestamp_min = timestamp_min
        self.state.measurements = measurements
        self.state.processing_log = []

        self.log(f"=== Pipeline Start: {sample_id}, t={timestamp_min}min, {len(measurements)}项指标 ===")

        # Phase 1: 感知（Monitor Agent）
        self.log("Phase 1: 监测感知...")
        alerts = []
        if self.monitor:
            alerts = self.monitor.analyze_sync(measurements)
            self.state.alerts = alerts
            self.log(f"  检测到 {len(alerts)} 项异常")
        else:
            self.log("  [降级] Monitor Agent不可用，跳过阈值检测")

        if not alerts:
            self.state.risk_level = "MINIMAL"
            self.state.final_summary = "所有指标在正常范围内，继续标准监测。"
            self.log(f"=== Pipeline End: 无异常 ({time.time()-start_time:.2f}s) ===")
            return self.state

        # Phase 2: 诊断分析（Diagnosis Agent）
        self.log("Phase 2: 诊断分析...")
        diagnoses = []
        if self.diagnosis:
            diagnoses = self.diagnosis.diagnose_sync(alerts)
            self.state.diagnoses = diagnoses
            self.log(f"  生成 {len(diagnoses)} 项诊断")
        else:
            self.log("  [降级] Diagnosis Agent不可用，跳过因果分析")

        # Phase 3: 知识检索（Knowledge Agent）—— 可与Phase 2并行，但同步版串行
        self.log("Phase 3: 知识检索...")
        evidence = []
        if self.knowledge:
            evidence = self.knowledge.query_sync(alerts)
            self.state.evidence_pool = evidence
            self.log(f"  收集 {len(evidence)} 条证据")
        else:
            self.log("  [降级] Knowledge Agent不可用，使用配置默认策略")

        # Phase 4: 策略推荐（Strategy Agent）
        self.log("Phase 4: 策略推荐...")
        strategies = []
        if self.strategy:
            strategies = self.strategy.recommend_sync(diagnoses, evidence, measurements)
            self.state.strategies = strategies
            self.log(f"  生成 {len(strategies)} 条策略")
        else:
            self.log("  [降级] Strategy Agent不可用，无策略输出")

        # Phase 5: 安全校验（Coordinator自身）
        self.log("Phase 5: 安全校验...")
        safety_result = self.use_tool("safety_checker", strategies)
        if safety_result.success and safety_result.data:
            safety = safety_result.data
            self.state.safety_warnings = safety.get("warnings", [])
            self.state.drug_conflicts = [c["warning"] for c in safety.get("conflicts", [])]
            if not safety["safe"]:
                self.log(f"  ⚠️ 发现 {len(safety['conflicts'])} 个药物冲突")
            else:
                self.log(f"  ✓ 安全校验通过 ({safety.get('drug_count', 0)} 种药物)")

        # Phase 6: 冲突仲裁
        resolved = self.use_tool("conflict_resolver", strategies)
        if resolved.success and resolved.data:
            self.state.strategies = resolved.data

        # Phase 7: 风险等级
        risk_result = self.use_tool("risk_calculator", alerts)
        if risk_result.success:
            self.state.risk_level = risk_result.data

        # Phase 8: 生成摘要
        self.state.final_summary = self._generate_summary()

        elapsed = time.time() - start_time
        self.log(f"=== Pipeline End: risk={self.state.risk_level}, "
                 f"{len(strategies)}条策略 ({elapsed:.2f}s) ===")

        return self.state

    def _generate_summary(self) -> str:
        """生成最终摘要"""
        lines = []
        alerts = self.state.alerts
        strategies = self.state.strategies

        if not alerts:
            return "所有指标在正常范围内。"

        critical = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        warning = [a for a in alerts if a.level == AlertLevel.WARNING]

        lines.append(f"检测到 {len(alerts)} 项异常 "
                     f"(危急{len(critical)}, 警告{len(warning)})")

        for a in critical[:3]:
            lines.append(f"  🔴 {a.indicator} = {a.value} ({a.direction})")
        for a in warning[:3]:
            lines.append(f"  🟡 {a.indicator} = {a.value} ({a.direction})")

        if strategies:
            top = strategies[0]
            lines.append(f"首要干预: {top.action}")
            if top.drug:
                lines.append(f"  药物: {top.drug} {top.dose or ''}")

        if self.state.drug_conflicts:
            lines.append(f"⚠️ 药物冲突: {len(self.state.drug_conflicts)}个")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Async Pipeline（完整异步版）
    # -------------------------------------------------------------------------

    async def orchestrate_async(self, measurements: Dict[str, float],
                                sample_id: str = "unknown",
                                timestamp_min: int = 0) -> PatientState:
        """异步编排Pipeline（支持Agent并行）"""

        self.state.sample_id = sample_id
        self.state.timestamp_min = timestamp_min
        self.state.measurements = measurements
        self.state.processing_log = []

        self.log("=== Async Pipeline Start ===")

        # Phase 1: Monitor（同步优先，安全关键路径）
        monitor_msg = AgentMessage(
            source="coordinator", target="monitor",
            msg_type="analyze", payload=measurements,
            priority=Priority.CRITICAL.value
        )
        monitor_response = await self.bus.request(monitor_msg, timeout=3.0)
        alerts = monitor_response.payload if monitor_response else []
        self.state.alerts = alerts if isinstance(alerts, list) else []

        if not self.state.alerts:
            self.state.risk_level = "MINIMAL"
            self.state.final_summary = "所有指标正常。"
            return self.state

        # Phase 2 & 3: Diagnosis + Knowledge 并行
        diagnosis_msg = AgentMessage(
            source="coordinator", target="diagnosis",
            msg_type="analyze", payload=self.state.alerts,
            priority=Priority.HIGH.value
        )
        knowledge_msg = AgentMessage(
            source="coordinator", target="knowledge",
            msg_type="query", payload=self.state.alerts,
            priority=Priority.NORMAL.value
        )

        diagnosis_task = self.bus.request(diagnosis_msg, timeout=5.0)
        knowledge_task = self.bus.request(knowledge_msg, timeout=5.0)
        diagnosis_resp, knowledge_resp = await asyncio.gather(
            diagnosis_task, knowledge_task
        )

        diagnoses = diagnosis_resp.payload if diagnosis_resp else []
        evidence = knowledge_resp.payload if knowledge_resp else []
        self.state.diagnoses = diagnoses if isinstance(diagnoses, list) else []
        self.state.evidence_pool = evidence if isinstance(evidence, list) else []

        # Phase 4: Strategy
        strategy_msg = AgentMessage(
            source="coordinator", target="strategy",
            msg_type="recommend",
            payload={
                "diagnoses": self.state.diagnoses,
                "evidence": self.state.evidence_pool,
                "measurements": measurements,
            },
            priority=Priority.HIGH.value
        )
        strategy_resp = await self.bus.request(strategy_msg, timeout=10.0)
        strategies = strategy_resp.payload if strategy_resp else []
        self.state.strategies = strategies if isinstance(strategies, list) else []

        # Phase 5-7: Safety + Risk (sync tools)
        safety = self.use_tool("safety_checker", self.state.strategies)
        if safety.success and safety.data:
            self.state.safety_warnings = safety.data.get("warnings", [])
            self.state.drug_conflicts = [c["warning"] for c in safety.data.get("conflicts", [])]

        risk = self.use_tool("risk_calculator", self.state.alerts)
        if risk.success:
            self.state.risk_level = risk.data

        self.state.final_summary = self._generate_summary()

        # Phase 8: Communication
        await self.send("communication", "display", self.state, priority=Priority.NORMAL.value)

        self.log("=== Async Pipeline End ===")
        return self.state

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        if msg.msg_type == "new_data":
            measurements = msg.payload.get("measurements", {})
            sample_id = msg.payload.get("sample_id", "unknown")
            timestamp = msg.payload.get("timestamp_min", 0)
            state = await self.orchestrate_async(measurements, sample_id, timestamp)
            await self.respond(msg, state)

        elif msg.msg_type == "run_pipeline":
            measurements = msg.payload.get("measurements", {})
            sample_id = msg.payload.get("sample_id", "unknown")
            timestamp = msg.payload.get("timestamp_min", 0)
            state = self._orchestrate_sync(measurements, sample_id, timestamp)
            await self.respond(msg, state)

    # -------------------------------------------------------------------------
    # 公共接口
    # -------------------------------------------------------------------------

    def run_pipeline(self, measurements: Dict[str, float],
                     sample_id: str = "unknown",
                     timestamp_min: int = 0) -> PatientState:
        """同步运行完整Pipeline（Streamlit等同步环境使用）"""
        return self._orchestrate_sync(measurements, sample_id, timestamp_min)

    def get_agent_status(self) -> Dict:
        """获取所有Agent状态（用于前端展示）"""
        agents = {
            "monitor": self.monitor,
            "diagnosis": self.diagnosis,
            "strategy": self.strategy,
            "knowledge": self.knowledge,
            "communication": self.communication,
        }
        status = {}
        for name, agent in agents.items():
            if agent:
                status[name] = {
                    "status": agent.status.value,
                    "tools": agent.get_tool_list(),
                    "tool_count": len(agent.tools),
                }
            else:
                status[name] = {"status": "unavailable", "tools": [], "tool_count": 0}
        return status
