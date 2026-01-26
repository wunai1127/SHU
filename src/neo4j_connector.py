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

    def find_indicator_relations(self, indicator_name: str) -> List[Dict]:
        """查找指标相关的关系"""
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($indicator)
           OR toLower(b.name) CONTAINS toLower($indicator)
        RETURN a.name AS from_entity, a.type AS from_type,
               properties(r) AS relation_props,
               b.name AS to_entity, b.type AS to_type
        LIMIT 50
        """
        return self._run_query(query, {'indicator': indicator_name})

    def find_intervention_evidence(self, intervention: str) -> List[Dict]:
        """查找干预措施的证据"""
        query = """
        MATCH (a:Entity)-[r:RELATION]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($intervention)
           OR toLower(b.name) CONTAINS toLower($intervention)
        RETURN a.name AS entity1, a.type AS type1,
               properties(r) AS relation,
               b.name AS entity2, b.type AS type2
        LIMIT 30
        """
        return self._run_query(query, {'intervention': intervention})

    def find_abnormality_path(self, abnormality: str) -> List[Dict]:
        """查找异常状态的因果路径"""
        query = """
        MATCH path = (a:Entity)-[r:RELATION*1..3]->(b:Entity)
        WHERE toLower(a.name) CONTAINS toLower($abnormality)
        RETURN [node IN nodes(path) | node.name] AS path_nodes,
               [node IN nodes(path) | node.type] AS path_types,
               length(path) AS path_length
        LIMIT 20
        """
        return self._run_query(query, {'abnormality': abnormality})

    def get_clinical_properties(self, entity_name: str) -> Dict:
        """获取实体的临床属性"""
        query = """
        MATCH (n:Entity)
        WHERE toLower(n.name) CONTAINS toLower($name)
        RETURN properties(n) AS props
        LIMIT 1
        """
        results = self._run_query(query, {'name': entity_name})
        if results:
            return results[0]['props']
        return {}


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


if __name__ == "__main__":
    explore_database()
