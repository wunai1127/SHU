"""
KnowledgeAgent - 知识层
========================
工具:
  1. neo4j_query     - Neo4jConnector: 图数据库查询(因果/治疗/证据)
  2. local_kg_query  - KnowledgeGraph: 本地三元组存储查询
  3. consensus_search- 共识知识检索: 从extracted_knowledge.json搜索
  4. evidence_scorer - 证据评分: 对收集到的证据按强度/来源排序

职责: 接收查询请求 → 多源检索(Neo4j + 本地KG + 共识) → 证据融合 → 输出Evidence列表
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseAgent, EventBus, PatientState, AgentTool,
    AgentMessage, AlertEvent, Priority
)

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """
    知识层Agent - 多源知识检索 + 证据融合

    Tools:
      - neo4j_query: Neo4j图数据库查询
      - local_kg_query: 本地知识图谱三元组查询
      - consensus_search: 共识文献知识检索
      - evidence_scorer: 证据评分与排序
    """

    def __init__(self, bus: EventBus, state: PatientState, config_dir: str = None):
        super().__init__("knowledge", bus, state)
        self._config_dir = config_dir or str(Path(__file__).parent.parent.parent / "config")
        self._project_root = Path(self._config_dir).parent
        self._neo4j = None
        self._local_kg = None
        self._consensus_data = None
        self.setup_tools()

    def setup_tools(self):
        # Tool 1: Neo4j查询
        try:
            from neo4j_connector import Neo4jKnowledgeGraph
            import os
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "")
            if password:
                self._neo4j = Neo4jKnowledgeGraph(uri=uri, user=user, password=password)
                self.register_tool(AgentTool(
                    name="neo4j_query",
                    description="Neo4j图数据库查询: 搜索因果关系、治疗方案、药物证据",
                    func=self._query_neo4j
                ))
        except (ImportError, Exception) as e:
            logger.info(f"KnowledgeAgent: Neo4j not available: {e}")

        # Tool 2: 本地知识图谱
        try:
            from knowledge_graph import KnowledgeGraph
            self._local_kg = KnowledgeGraph()
            self.register_tool(AgentTool(
                name="local_kg_query",
                description="本地知识图谱三元组查询: (主体, 关系, 客体)模式匹配",
                func=self._query_local_kg
            ))
        except ImportError as e:
            logger.info(f"KnowledgeAgent: KnowledgeGraph not available: {e}")

        # Tool 3: 共识知识检索（始终可用）
        self._load_consensus()
        self.register_tool(AgentTool(
            name="consensus_search",
            description="共识文献知识检索: 从7篇心脏移植共识中搜索阈值/药物/因果关系",
            func=self._search_consensus
        ))

        # Tool 4: 证据评分（始终可用）
        self.register_tool(AgentTool(
            name="evidence_scorer",
            description="证据评分: 按来源可信度(共识>KG>推断)和相关度排序",
            func=self._score_evidence
        ))

    def _load_consensus(self):
        """加载共识知识"""
        path = self._project_root / "extracted_knowledge.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    self._consensus_data = json.load(f)
                logger.info("KnowledgeAgent: loaded consensus knowledge")
            except Exception as e:
                logger.warning(f"KnowledgeAgent: failed to load consensus: {e}")
                self._consensus_data = {}
        else:
            self._consensus_data = {}

    # -------------------------------------------------------------------------
    # Tool implementations
    # -------------------------------------------------------------------------

    def _query_neo4j(self, keyword: str, query_type: str = "decision_support") -> Dict:
        """工具: Neo4j查询"""
        if not self._neo4j:
            return {"error": "Neo4j not connected", "results": []}
        try:
            if query_type == "decision_support":
                result = self._neo4j.query_decision_support(keyword)
                return {
                    "source": "neo4j",
                    "keyword": keyword,
                    "causes": result.get("causes", []),
                    "treatments": result.get("treatments", []),
                    "consequences": result.get("consequences", []),
                }
            elif query_type == "search":
                entities = self._neo4j.search_entities(keyword)
                return {"source": "neo4j", "keyword": keyword, "entities": entities}
            else:
                return {"error": f"Unknown query type: {query_type}"}
        except Exception as e:
            return {"error": str(e), "results": []}

    def _query_local_kg(self, subject: str = None, predicate: str = None,
                        obj: str = None) -> List[Dict]:
        """工具: 本地KG查询"""
        if not self._local_kg:
            return []
        results = []
        if subject:
            triples = self._local_kg.query_subject(subject)
            results.extend([{"s": t.subject, "p": t.predicate, "o": t.object} for t in triples])
        if predicate:
            triples = self._local_kg.query_predicate(predicate)
            results.extend([{"s": t.subject, "p": t.predicate, "o": t.object} for t in triples])
        return results

    def _search_consensus(self, keyword: str) -> Dict:
        """工具: 共识知识检索"""
        if not self._consensus_data:
            return {"keyword": keyword, "results": [], "source": "consensus"}

        results = []
        keyword_lower = keyword.lower()

        # 搜索阈值
        thresholds = self._consensus_data.get("阈值_与_指标", {})
        for key, val in thresholds.items():
            if keyword_lower in key.lower() or keyword_lower in json.dumps(val, ensure_ascii=False).lower():
                results.append({
                    "type": "threshold",
                    "indicator": key,
                    "data": val,
                    "source": val.get("来源", "共识")
                })

        # 搜索药物策略
        drugs = self._consensus_data.get("药物策略", {})
        for category, cat_data in drugs.items():
            if isinstance(cat_data, dict):
                text = json.dumps(cat_data, ensure_ascii=False).lower()
                if keyword_lower in text or keyword_lower in category.lower():
                    results.append({
                        "type": "drug_strategy",
                        "category": category,
                        "data": cat_data,
                        "source": "共识药物策略"
                    })

        # 搜索因果关系
        causals = self._consensus_data.get("因果关系", [])
        for rel in causals:
            if isinstance(rel, dict):
                text = json.dumps(rel, ensure_ascii=False).lower()
                if keyword_lower in text:
                    results.append({
                        "type": "causal",
                        "data": rel,
                        "source": "共识因果关系"
                    })

        return {
            "keyword": keyword,
            "result_count": len(results),
            "results": results,
            "source": "consensus_knowledge"
        }

    def _score_evidence(self, evidence_list: List[Dict]) -> List[Dict]:
        """工具: 证据评分"""
        source_weights = {
            "consensus": 0.9,
            "neo4j": 0.8,
            "local_kg": 0.7,
            "config": 0.6,
            "inferred": 0.4,
        }

        scored = []
        for ev in evidence_list:
            source = ev.get("source", "inferred")
            base_score = source_weights.get(source, 0.3)

            # 加分项: 含具体数值
            if any(key in str(ev.get("data", "")) for key in ["mg", "μg", "mmHg", "%", "Wood"]):
                base_score += 0.05

            scored.append({
                **ev,
                "evidence_score": min(base_score, 1.0),
                "strength": "high" if base_score >= 0.8 else "medium" if base_score >= 0.6 else "low"
            })

        # 按分数降序排列
        scored.sort(key=lambda x: x["evidence_score"], reverse=True)
        return scored

    # -------------------------------------------------------------------------
    # 综合检索（多源融合）
    # -------------------------------------------------------------------------

    def multi_source_query(self, keyword: str, indicators: List[str] = None) -> List[Dict]:
        """多源综合查询: Neo4j + 本地KG + 共识 → 融合评分"""
        all_evidence = []

        # 1. 共识检索（最可靠）
        consensus_result = self.use_tool("consensus_search", keyword)
        if consensus_result.success and consensus_result.data:
            for item in consensus_result.data.get("results", []):
                all_evidence.append({
                    "source": "consensus",
                    "type": item.get("type", "unknown"),
                    "data": item.get("data", {}),
                    "indicator": item.get("indicator", keyword),
                })

        # 2. Neo4j查询
        if "neo4j_query" in self.tools:
            neo4j_result = self.use_tool("neo4j_query", keyword)
            if neo4j_result.success and neo4j_result.data and "error" not in neo4j_result.data:
                for cause in neo4j_result.data.get("causes", []):
                    all_evidence.append({
                        "source": "neo4j",
                        "type": "causal",
                        "data": {"cause": str(cause), "keyword": keyword},
                    })
                for tx in neo4j_result.data.get("treatments", []):
                    all_evidence.append({
                        "source": "neo4j",
                        "type": "treatment",
                        "data": {"treatment": str(tx), "keyword": keyword},
                    })

        # 3. 本地KG
        if "local_kg_query" in self.tools:
            kg_result = self.use_tool("local_kg_query", subject=keyword)
            if kg_result.success and kg_result.data:
                for triple in kg_result.data:
                    all_evidence.append({
                        "source": "local_kg",
                        "type": "triple",
                        "data": triple,
                    })

        # 4. 评分排序
        scorer_result = self.use_tool("evidence_scorer", all_evidence)
        if scorer_result.success:
            return scorer_result.data

        return all_evidence

    # -------------------------------------------------------------------------
    # Message handling
    # -------------------------------------------------------------------------

    async def on_message(self, msg: AgentMessage):
        if msg.msg_type == "query":
            # 接收告警列表，为每个异常指标检索证据
            alerts = msg.payload
            all_evidence = []

            if isinstance(alerts, list):
                for alert in alerts:
                    keyword = alert.indicator if isinstance(alert, AlertEvent) else str(alert)
                    evidence = self.multi_source_query(keyword)
                    all_evidence.extend(evidence)
            elif isinstance(alerts, str):
                all_evidence = self.multi_source_query(alerts)

            self.state.evidence_pool = all_evidence
            self.log(f"知识检索完成: {len(all_evidence)} 条证据")
            await self.respond(msg, all_evidence)

        elif msg.msg_type == "consensus_search":
            keyword = msg.payload
            result = self.use_tool("consensus_search", keyword)
            await self.respond(msg, result.data)

        elif msg.msg_type == "neo4j_search":
            keyword = msg.payload
            result = self.use_tool("neo4j_query", keyword)
            await self.respond(msg, result.data)

    # -------------------------------------------------------------------------
    # 同步接口
    # -------------------------------------------------------------------------

    def query_sync(self, alerts: List[AlertEvent]) -> List[Dict]:
        """同步查询"""
        all_evidence = []
        for alert in alerts:
            evidence = self.multi_source_query(alert.indicator)
            all_evidence.extend(evidence)
        self.state.evidence_pool = all_evidence
        return all_evidence
