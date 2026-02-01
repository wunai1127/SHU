"""
StrategyAgent - 决策层
======================
工具:
  1. evidence_recommender  - EvidenceStrategyEngine: 证据驱动推荐+CoT推理
  2. strategy_mapper       - StrategyMapper: 阈值→临床决策映射
  3. intervention_library  - 干预库查询: 药物/剂量/方案
  4. cot_reasoner          - CoT推理链生成器

职责: 接收DiagnosisResult + Evidence → 生成策略推荐 → CoT推理链 → 输出StrategyRecommendation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool,
    AgentMessage, DiagnosisResult, StrategyRecommendation, Priority
)

logger = logging.getLogger(__name__)


class StrategyAgent(BaseAgent):
    """
    决策层Agent - 策略推荐 + CoT推理

    Tools:
      - evidence_recommender: 基于证据的策略推荐（含干预库查询）
      - strategy_mapper: 阈值结果→临床决策映射
      - cot_reasoner: 5步CoT推理链生成
      - drug_lookup: 药物/剂量查询
    """

    def __init__(self, bus: EventBus, state: PatientState, config_dir: str = None):
        super().__init__("strategy", bus, state)
        self._config_dir = config_dir or str(Path(__file__).parent.parent.parent / "config")
        self._ese = None   # EvidenceStrategyEngine
        self._sm = None    # StrategyMapper
        self._bsr = None   # BaselineStrategyRecommender
        self.setup_tools()

    def setup_tools(self):
        try:
            from evidence_strategy_engine import EvidenceStrategyEngine
            self._ese = EvidenceStrategyEngine()
            self.register_tool(AgentTool(
                name="evidence_recommender",
                description="证据驱动策略推荐: 查询干预库+KG三元组，生成含CoT推理链的完整推荐",
                func=self._evidence_recommend
            ))
            self.register_tool(AgentTool(
                name="drug_lookup",
                description="查询干预库中特定指标的药物/剂量/方案",
                func=self._drug_lookup
            ))
        except ImportError as e:
            logger.warning(f"StrategyAgent: EvidenceStrategyEngine not available: {e}")

        try:
            from strategy_mapper import StrategyMapper
            self._sm = StrategyMapper(config_dir=self._config_dir)
            self.register_tool(AgentTool(
                name="strategy_mapper",
                description="将阈值评估结果映射为临床决策（优先级/动作/升级方案）",
                func=self._map_strategy
            ))
        except ImportError as e:
            logger.warning(f"StrategyAgent: StrategyMapper not available: {e}")

        try:
            from baseline_strategy_recommender import BaselineStrategyRecommender
            self._bsr = BaselineStrategyRecommender(config_dir=self._config_dir)
            self.register_tool(AgentTool(
                name="full_report_generator",
                description="生成完整的Baseline策略报告（含异常检测+推荐+CoT+一致性检查）",
                func=self._generate_full_report
            ))
        except ImportError as e:
            logger.warning(f"StrategyAgent: BaselineStrategyRecommender not available: {e}")

        # 始终可用的CoT推理工具
        self.register_tool(AgentTool(
            name="cot_reasoner",
            description="5步CoT推理链生成器: 观察→病理分析→机制关联→干预选择→预期效果",
            func=self._build_cot_chain
        ))

    def set_llm_and_neo4j(self, llm=None, neo4j_connector=None):
        """设置LLM和Neo4j连接器，传递给内部引擎"""
        if self._bsr:
            if llm:
                self._bsr.set_llm(llm)
            if neo4j_connector:
                self._bsr.set_neo4j(neo4j_connector)

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _evidence_recommend(self, indicator: str, value: float, direction: str = "abnormal") -> Dict:
        """工具: 证据驱动推荐"""
        if not self._ese:
            return {"error": "EvidenceStrategyEngine not loaded"}

        abnormality_state = f"{indicator}_{direction.capitalize()}"
        report = self._ese.generate_baseline_report(
            sample_id="agent_query",
            measurements={indicator: value}
        )

        recommendations = []
        if hasattr(report, 'recommendations'):
            for rec in report.recommendations:
                recommendations.append({
                    "indicator": getattr(rec, 'indicator', indicator),
                    "intervention": getattr(rec, 'intervention', ''),
                    "target_value": getattr(rec, 'target_value', None),
                    "target_range": getattr(rec, 'target_range', None),
                    "reasoning_chain": getattr(rec, 'reasoning_chain', []),
                    "evidence": [str(e) for e in getattr(rec, 'supporting_evidence', [])],
                    "confidence": getattr(rec, 'confidence', 0.5),
                })

        return {
            "indicator": indicator,
            "value": value,
            "risk_level": getattr(report, 'risk_level', 'UNKNOWN'),
            "recommendations": recommendations,
        }

    def _drug_lookup(self, indicator: str, direction: str = "abnormal") -> Dict:
        """工具: 干预库药物查询"""
        if not self._ese:
            return {"error": "EvidenceStrategyEngine not loaded"}

        key = f"{indicator}_{direction.capitalize()}"
        lib = getattr(self._ese, 'INTERVENTION_LIBRARY', {})
        entry = lib.get(key, {})

        if not entry:
            # 尝试模糊匹配
            for k, v in lib.items():
                if indicator.lower() in k.lower():
                    entry = v
                    break

        if entry:
            return {
                "indicator": indicator,
                "intervention": entry.get("intervention", ""),
                "drug": entry.get("intervention_details", {}).get("drug", ""),
                "dose": entry.get("intervention_details", {}).get("dose", ""),
                "target": entry.get("intervention_details", {}).get("target", ""),
                "confidence": entry.get("confidence", 0.5),
                "warnings": entry.get("warnings", []),
            }
        return {"indicator": indicator, "found": False}

    def _map_strategy(self, measurements: Dict[str, float]) -> Dict:
        """工具: 阈值→决策映射"""
        if not self._sm:
            return {"error": "StrategyMapper not loaded"}

        decision = self._sm.evaluate_all(measurements)
        return {
            "overall_status": decision.overall_status,
            "primary_actions": [
                {
                    "indicator": a.indicator,
                    "priority": a.priority.value if hasattr(a.priority, 'value') else str(a.priority),
                    "action": a.action,
                    "rationale": a.rationale,
                }
                for a in decision.primary_actions
            ],
            "escalation": getattr(decision, 'escalation_protocol', None),
        }

    def _generate_full_report(self, measurements: Dict[str, float], sample_id: str = "agent") -> Dict:
        """工具: 完整报告生成"""
        if not self._bsr:
            return {"error": "BaselineStrategyRecommender not loaded"}

        report = self._bsr.analyze_baseline(measurements, sample_id=sample_id)
        return {
            "sample_id": report.sample_id,
            "risk_level": report.risk_level,
            "abnormality_count": len(report.abnormalities),
            "recommendation_count": len(report.recommendations),
            "summary": report.summary,
            "consistency": report.consistency_check,
        }

    def _build_cot_chain(self, indicator: str, value: float, direction: str,
                         evidence: List[str] = None, intervention: str = None) -> List[str]:
        """工具: CoT推理链生成"""
        chain = []

        # Step 1 - 观察
        chain.append(f"Step 1 - 观察: {indicator} = {value}, 方向: {direction}")

        # Step 2 - 病理生理分析
        physio_map = {
            "MAP": "低MAP导致组织灌注不足，可引起器官功能障碍",
            "Lactate": "乳酸升高反映组织缺氧或无氧代谢增加",
            "K_A": "钾异常影响心肌电活动，可致致命性心律失常",
            "pH": "酸碱失衡影响酶活性、电解质分布和器官功能",
            "CI": "心输出量不足导致全身器官灌注下降",
            "PVR": "肺血管阻力升高增加右室后负荷，可致急性右心衰(共识>50mmHg危险)",
            "TPG": "跨肺压差升高反映固定性肺血管阻力(共识>14-15mmHg禁忌)",
            "PASP": "肺动脉压升高→右室压力过负荷(共识>50mmHg右室难承受)",
            "Creatinine": "肌酐升高提示肾功能不全(共识>1.7减量, >2.0替代CNI)",
            "GFR": "GFR下降提示慢性肾病(共识<60需CKD监测)",
            "EF": "射血分数下降提示收缩功能障碍或急性排斥(共识<50%需立即治疗)",
            "Bilirubin": "胆红素升高提示肝功能异常(共识>2.5不全)",
        }
        chain.append(f"Step 2 - 病理生理分析: {physio_map.get(indicator, '需进一步评估')}")

        # Step 3 - 机制关联
        if evidence:
            chain.append(f"Step 3 - 机制关联: [Evidence] {evidence[0]}")
        else:
            chain.append("Step 3 - 机制关联: 基于临床指南和移植共识推荐")

        # Step 4 - 干预选择
        if intervention:
            chain.append(f"Step 4 - 干预选择: {intervention}")
        else:
            chain.append("Step 4 - 干预选择: 评估后针对性干预")

        # Step 5 - 预期效果
        chain.append(f"Step 5 - 预期效果: {indicator}恢复至目标范围")

        return chain

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        if msg.msg_type == "recommend":
            payload = msg.payload or {}
            diagnoses = payload.get("diagnoses", [])
            evidence = payload.get("evidence", [])
            measurements = payload.get("measurements", {})
            strategies = self._generate_strategies(diagnoses, evidence, measurements)
            await self.respond(msg, strategies)

        elif msg.msg_type == "drug_query":
            indicator = msg.payload.get("indicator", "")
            direction = msg.payload.get("direction", "abnormal")
            result = self.use_tool("drug_lookup", indicator, direction)
            await self.respond(msg, result.data)

        elif msg.msg_type == "full_report":
            measurements = msg.payload
            result = self.use_tool("full_report_generator", measurements)
            await self.respond(msg, result.data)

    def _generate_strategies(self, diagnoses: List[DiagnosisResult],
                             evidence: List[Dict],
                             measurements: Dict[str, float]) -> List[StrategyRecommendation]:
        """生成策略推荐列表"""
        strategies = []
        self.log(f"生成策略: {len(diagnoses)} 项诊断, {len(evidence)} 条证据")

        for diag in diagnoses:
            indicator = diag.affected_indicators[0] if diag.affected_indicators else ""
            value = measurements.get(indicator, 0)
            direction = "high" if "high" in diag.primary_cause.lower() else "low"

            # 工具1: 查询干预库
            drug_result = self.use_tool("drug_lookup", indicator, direction)
            drug_info = drug_result.data if drug_result.success else {}

            # 工具2: CoT推理链
            cot_evidence = [str(e) for e in evidence[:3]] if evidence else []
            intervention_str = drug_info.get("intervention", "") if drug_info else ""
            cot_result = self.use_tool(
                "cot_reasoner", indicator, value, direction,
                evidence=cot_evidence, intervention=intervention_str
            )
            cot_chain = cot_result.data if cot_result.success else []

            rec = StrategyRecommendation(
                indicator=indicator,
                action=drug_info.get("intervention", "评估后干预") if drug_info else "评估后干预",
                drug=drug_info.get("drug") if drug_info else None,
                dose=drug_info.get("dose") if drug_info else None,
                target_value=drug_info.get("target_value") if drug_info else None,
                reasoning_chain=cot_chain if isinstance(cot_chain, list) else [],
                evidence=cot_evidence,
                severity=diag.primary_cause.split("(")[-1].rstrip(")") if "(" in diag.primary_cause else "warning",
                confidence=drug_info.get("confidence", 0.5) if drug_info else 0.5,
                source="consensus" if drug_info and drug_info.get("found", True) else "default",
            )
            strategies.append(rec)

        # 按severity排序
        severity_order = {"critical": 0, "red_line": 0, "warning": 1, "info": 2}
        strategies.sort(key=lambda s: severity_order.get(s.severity, 3))

        self.state.strategies = strategies
        return strategies

    # -------------------------------------------------------------------------
    # 同步接口
    # -------------------------------------------------------------------------

    def recommend_sync(self, diagnoses: List[DiagnosisResult],
                       evidence: List[Dict],
                       measurements: Dict[str, float]) -> List[StrategyRecommendation]:
        """同步策略生成"""
        return self._generate_strategies(diagnoses, evidence, measurements)
