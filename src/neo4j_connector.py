#!/usr/bin/env python3
"""
Neo4j知识图谱连接器 - 用于查询生理学决策证据
"""

from neo4j import GraphDatabase
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jKnowledgeGraph:
    """Neo4j知识图谱接口"""

    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "qaz709394",
                 database: str = "backup"):
        """
        初始化连接

        Args:
            uri: Neo4j Bolt URI
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}, database: {database}")

    def close(self):
        """关闭连接"""
        self.driver.close()

    def _run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    # ==================== 探索性查询 ====================

    def get_schema_info(self) -> Dict[str, Any]:
        """获取Schema信息"""
        # 节点标签
        labels = self._run_query("CALL db.labels()")

        # 关系类型
        rel_types = self._run_query("CALL db.relationshipTypes()")

        # 属性键
        prop_keys = self._run_query("CALL db.propertyKeys()")

        # 节点数量
        node_count = self._run_query("MATCH (n) RETURN count(n) as count")[0]['count']

        # 关系数量
        rel_count = self._run_query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']

        return {
            'labels': [r['label'] for r in labels],
            'relationship_types': [r['relationshipType'] for r in rel_types],
            'property_keys': [r['propertyKey'] for r in prop_keys],
            'node_count': node_count,
            'relationship_count': rel_count
        }

    def get_entity_types(self) -> List[Dict]:
        """获取所有实体类型"""
        query = """
        MATCH (n:Entity)
        RETURN DISTINCT n.type AS type, count(*) AS count
        ORDER BY count DESC
        LIMIT 50
        """
        return self._run_query(query)

    def get_sample_entities(self, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """获取实体样本"""
        if entity_type:
            query = """
            MATCH (n:Entity {type: $type})
            RETURN n.id AS id, n.name AS name, n.type AS type, properties(n) AS props
            LIMIT $limit
            """
            return self._run_query(query, {'type': entity_type, 'limit': limit})
        else:
            query = """
            MATCH (n:Entity)
            RETURN n.id AS id, n.name AS name, n.type AS type, properties(n) AS props
            LIMIT $limit
            """
            return self._run_query(query, {'limit': limit})

    def get_sample_relations(self, limit: int = 20) -> List[Dict]:
        """获取关系样本"""
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        RETURN a.name AS from_name, a.type AS from_type,
               type(r) AS relation, properties(r) AS rel_props,
               b.name AS to_name, b.type AS to_type
        LIMIT $limit
        """
        return self._run_query(query, {'limit': limit})

    def search_entities_by_name(self, keyword: str, limit: int = 20) -> List[Dict]:
        """按名称搜索实体"""
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($keyword)
        RETURN n.id AS id, n.name AS name, n.type AS type
        LIMIT $limit
        """
        return self._run_query(query, {'keyword': keyword, 'limit': limit})

    # ==================== 灌注决策查询 ====================

    def find_complication_causes(self, complication: str) -> List[Dict]:
        """查找并发症的原因"""
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WHERE toLower(b.name) CONTAINS toLower($complication)
          AND r.type IN ['causes', 'increases_risk', 'leads_to']
        RETURN a.name AS cause, a.type AS cause_type,
               r.type AS relation_type, r.origin AS origin,
               b.name AS complication
        ORDER BY a.type
        LIMIT 30
        """
        return self._run_query(query, {'complication': complication})

    def find_treatment_for_complication(self, complication: str) -> List[Dict]:
        """查找并发症的治疗方案"""
        query = """
        MATCH (treatment:Entity)-[r:RELATION]->(comp:Entity)
        WHERE toLower(comp.name) CONTAINS toLower($complication)
          AND treatment.type IN ['treatment_regimen', 'medication', 'surgical_procedure']
          AND r.type IN ['treats', 'alleviates', 'prevents', 'reduces_risk', 'indicated_for']
        RETURN treatment.name AS treatment, treatment.type AS treatment_type,
               r.type AS relation_type,
               comp.name AS complication
        LIMIT 20
        """
        return self._run_query(query, {'complication': complication})

    def find_indicator_abnormality_consequences(self, indicator: str) -> List[Dict]:
        """查找指标异常的后果"""
        query = """
        MATCH (ind:Entity)-[r:RELATION]->(consequence:Entity)
        WHERE toLower(ind.name) CONTAINS toLower($indicator)
          AND ind.type = 'monitoring_indicator'
          AND r.type IN ['causes', 'increases_risk', 'indicates', 'associated with', 'monitors']
        RETURN ind.name AS indicator,
               r.type AS relation,
               consequence.name AS consequence,
               consequence.type AS consequence_type
        LIMIT 30
        """
        return self._run_query(query, {'indicator': indicator})

    def find_medication_effects(self, medication: str) -> List[Dict]:
        """查找药物的效果和适应症"""
        query = """
        MATCH (med:Entity)-[r:RELATION]->(target:Entity)
        WHERE toLower(med.name) CONTAINS toLower($medication)
          AND med.type = 'medication'
        RETURN med.name AS medication,
               r.type AS relation,
               target.name AS target,
               target.type AS target_type
        LIMIT 30
        """
        return self._run_query(query, {'medication': medication})

    def find_risk_factors_for_outcome(self, outcome: str) -> List[Dict]:
        """查找导致某结局的风险因素"""
        query = """
        MATCH (risk:Entity)-[r:RELATION]->(outcome:Entity)
        WHERE toLower(outcome.name) CONTAINS toLower($outcome)
          AND risk.type = 'risk_factor'
          AND r.type IN ['causes', 'increases_risk', 'associated with']
        RETURN risk.name AS risk_factor,
               r.type AS relation,
               outcome.name AS outcome,
               r.origin AS evidence_origin,
               r.score AS similarity_score
        ORDER BY r.score DESC NULLS LAST
        LIMIT 20
        """
        return self._run_query(query, {'outcome': outcome})

    def get_perfusion_related_knowledge(self) -> Dict[str, List[Dict]]:
        """获取灌注相关的知识"""
        results = {}

        # 灌注相关并发症
        query_complications = """
        MATCH (n:Entity)
        WHERE n.type = 'complication'
          AND (toLower(n.name) CONTAINS 'perfusion'
               OR toLower(n.name) CONTAINS 'ischemia'
               OR toLower(n.name) CONTAINS 'reperfusion'
               OR toLower(n.name) CONTAINS 'graft')
        RETURN n.name AS name, n.type AS type
        LIMIT 30
        """
        results['perfusion_complications'] = self._run_query(query_complications)

        # 灌注相关监测指标
        query_indicators = """
        MATCH (n:Entity)
        WHERE n.type = 'monitoring_indicator'
          AND (toLower(n.name) CONTAINS 'cardiac'
               OR toLower(n.name) CONTAINS 'lactate'
               OR toLower(n.name) CONTAINS 'pressure'
               OR toLower(n.name) CONTAINS 'flow'
               OR toLower(n.name) CONTAINS 'oxygen')
        RETURN n.name AS name, n.type AS type
        LIMIT 30
        """
        results['perfusion_indicators'] = self._run_query(query_indicators)

        # 相关治疗方案
        query_treatments = """
        MATCH (n:Entity)
        WHERE n.type IN ['treatment_regimen', 'medication']
          AND (toLower(n.name) CONTAINS 'vasopressor'
               OR toLower(n.name) CONTAINS 'inotrope'
               OR toLower(n.name) CONTAINS 'ecmo'
               OR toLower(n.name) CONTAINS 'perfusion')
        RETURN n.name AS name, n.type AS type
        LIMIT 30
        """
        results['perfusion_treatments'] = self._run_query(query_treatments)

        return results

    def query_decision_support(self, abnormality: str, indicator: str = None) -> Dict[str, Any]:
        """
        综合决策支持查询

        Args:
            abnormality: 异常状态描述 (如 "low MAP", "high lactate")
            indicator: 具体指标名称

        Returns:
            包含原因、后果、治疗建议的字典
        """
        result = {
            'abnormality': abnormality,
            'causes': [],
            'consequences': [],
            'treatments': [],
            'related_risks': []
        }

        # 查找原因
        causes_query = """
        MATCH (cause:Entity)-[r:RELATION]->(abnormal:Entity)
        WHERE toLower(abnormal.name) CONTAINS toLower($abnormality)
          AND r.type IN ['causes', 'leads_to', 'results_in']
        RETURN cause.name AS cause, cause.type AS type, r.type AS relation
        LIMIT 15
        """
        result['causes'] = self._run_query(causes_query, {'abnormality': abnormality})

        # 查找后果
        consequences_query = """
        MATCH (abnormal:Entity)-[r:RELATION]->(consequence:Entity)
        WHERE toLower(abnormal.name) CONTAINS toLower($abnormality)
          AND r.type IN ['causes', 'increases_risk', 'leads_to']
        RETURN consequence.name AS consequence, consequence.type AS type, r.type AS relation
        LIMIT 15
        """
        result['consequences'] = self._run_query(consequences_query, {'abnormality': abnormality})

        # 查找治疗
        treatments_query = """
        MATCH (treatment:Entity)-[r:RELATION]->(abnormal:Entity)
        WHERE toLower(abnormal.name) CONTAINS toLower($abnormality)
          AND treatment.type IN ['treatment_regimen', 'medication', 'surgical_procedure', 'medical_device']
          AND r.type IN ['treats', 'alleviates', 'prevents', 'indicated_for', 'reduces_risk']
        RETURN treatment.name AS treatment, treatment.type AS type, r.type AS relation
        LIMIT 15
        """
        result['treatments'] = self._run_query(treatments_query, {'abnormality': abnormality})

        return result

    def get_clinical_properties(self, entity_name: str) -> Dict:
        """获取实体的临床属性"""
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($name)
        RETURN properties(n) AS props, n.name AS name, n.type AS type
        LIMIT 5
        """
        results = self._run_query(query, {'name': entity_name})
        return results if results else []


def explore_database():
    """探索数据库结构"""
    kg = Neo4jKnowledgeGraph()

    print("="*60)
    print("Neo4j Knowledge Graph Exploration")
    print("="*60)

    # Schema信息
    print("\n### Schema Info ###")
    schema = kg.get_schema_info()
    print(f"Labels: {schema['labels']}")
    print(f"Relationship Types: {schema['relationship_types']}")
    print(f"Node Count: {schema['node_count']}")
    print(f"Relationship Count: {schema['relationship_count']}")
    print(f"Property Keys (first 30): {schema['property_keys'][:30]}")

    # 实体类型
    print("\n### Entity Types ###")
    types = kg.get_entity_types()
    for t in types[:20]:
        print(f"  {t['type']}: {t['count']}")

    # 样本实体
    print("\n### Sample Entities ###")
    samples = kg.get_sample_entities(limit=10)
    for s in samples:
        print(f"  [{s['type']}] {s['name']}")
        # 打印非空属性
        props = {k: v for k, v in s['props'].items() if v and k not in ['name', 'id', 'type']}
        if props:
            for k, v in list(props.items())[:3]:
                print(f"    - {k}: {v}")

    # 样本关系
    print("\n### Sample Relations ###")
    relations = kg.get_sample_relations(limit=15)
    for r in relations:
        print(f"  [{r['from_type']}]{r['from_name']} --{r['relation']}--> [{r['to_type']}]{r['to_name']}")
        if r['rel_props']:
            print(f"    Props: {r['rel_props']}")

    # 搜索心脏移植相关
    print("\n### Heart Transplant Related ###")
    for keyword in ['heart', 'cardiac', 'transplant', 'lactate', 'MAP', 'perfusion']:
        results = kg.search_entities_by_name(keyword, limit=5)
        if results:
            print(f"\n  '{keyword}':")
            for r in results:
                print(f"    [{r['type']}] {r['name']}")

    kg.close()


def test_perfusion_queries():
    """测试灌注相关查询"""
    kg = Neo4jKnowledgeGraph()

    print("="*60)
    print("Perfusion Decision Support Queries")
    print("="*60)

    # 1. 灌注相关知识
    print("\n### Perfusion Related Knowledge ###")
    perfusion_knowledge = kg.get_perfusion_related_knowledge()
    for category, items in perfusion_knowledge.items():
        print(f"\n{category}:")
        for item in items[:10]:
            print(f"  - {item['name']}")

    # 2. 缺血再灌注损伤的后果
    print("\n### Ischemia-Reperfusion Injury Consequences ###")
    consequences = kg.find_indicator_abnormality_consequences("ischemia-reperfusion")
    for c in consequences[:10]:
        print(f"  {c['indicator']} --{c['relation']}--> {c['consequence']} ({c['consequence_type']})")

    # 3. 综合决策支持
    print("\n### Decision Support: Primary Graft Dysfunction ###")
    decision = kg.query_decision_support("primary graft dysfunction")
    print(f"  Causes: {[c['cause'] for c in decision['causes'][:5]]}")
    print(f"  Consequences: {[c['consequence'] for c in decision['consequences'][:5]]}")
    print(f"  Treatments: {[t['treatment'] for t in decision['treatments'][:5]]}")

    # 4. 低心输出量的治疗
    print("\n### Treatments for Low Cardiac Output ###")
    treatments = kg.find_treatment_for_complication("low cardiac output")
    for t in treatments[:10]:
        print(f"  [{t['treatment_type']}] {t['treatment']} --{t['relation_type']}--> {t['complication']}")

    kg.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "perfusion":
        test_perfusion_queries()
    else:
        explore_database()
