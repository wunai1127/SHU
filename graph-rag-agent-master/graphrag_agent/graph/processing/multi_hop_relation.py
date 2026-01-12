"""
多跳关系发现模块

在三元组/知识图谱中发现潜在的多跳（间接）关系

核心思想:
- 如果 A -[r1]-> B 且 B -[r2]-> C，则可能存在 A -[inferred_r]-> C
- 通过分析路径模式，推断隐含的直接关系
- 支持可配置的关系组合规则

使用方法:
    from graphrag_agent.graph.processing.multi_hop_relation import MultiHopRelationDiscoverer
    discoverer = MultiHopRelationDiscoverer(max_depth=3)
    result = discoverer.discover_all()
"""

import time
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.models.get_models import get_llm_model


@dataclass
class MultiHopPath:
    """多跳路径表示"""
    source: str                      # 起始实体
    target: str                      # 目标实体
    path: List[str]                  # 完整路径 [e1, e2, e3, ...]
    relations: List[str]             # 关系序列 [r1, r2, ...]
    depth: int                       # 跳数
    confidence: float = 1.0          # 置信度
    inferred_relation: str = ""      # 推断出的关系类型


@dataclass
class RelationInference:
    """关系推断结果"""
    source: str
    target: str
    inferred_relation: str
    supporting_paths: List[MultiHopPath]
    confidence: float
    evidence: str = ""


# 预定义的关系传递规则
# 格式: (relation1, relation2) -> inferred_relation
RELATION_COMPOSITION_RULES = {
    # 层级关系的传递
    ("属于", "属于"): "间接属于",
    ("包含", "包含"): "间接包含",
    ("是", "是"): "是",

    # 位置关系
    ("位于", "位于"): "位于",
    ("在", "在"): "在",

    # 因果关系
    ("导致", "导致"): "间接导致",
    ("引起", "导致"): "间接导致",
    ("导致", "引起"): "间接导致",

    # 时序关系
    ("之后", "之后"): "远后于",
    ("之前", "之前"): "远先于",

    # 关联关系
    ("相关", "相关"): "可能相关",
    ("关联", "关联"): "间接关联",

    # 作用关系
    ("影响", "影响"): "间接影响",
    ("作用于", "影响"): "间接影响",

    # 组成关系
    ("组成", "组成"): "组成",
    ("包括", "包括"): "包括",

    # 疾病-症状-治疗链
    ("表现为", "治疗"): "间接治疗目标",
    ("导致", "治疗"): "对症治疗",

    # 知识依赖
    ("需要", "需要"): "间接需要",
    ("依赖", "依赖"): "间接依赖",
}


class MultiHopRelationDiscoverer:
    """
    多跳关系发现器

    通过分析图谱中的路径模式，发现潜在的间接关系

    发现策略:
    1. 路径枚举 - 找出实体间的所有k跳路径
    2. 模式匹配 - 使用预定义规则推断关系
    3. 语义推理 - 使用LLM推断复杂关系
    4. 置信度计算 - 基于路径权重和出现频率
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_confidence: float = 0.5,
        use_llm_inference: bool = True,
        batch_size: int = 100
    ):
        """
        初始化多跳关系发现器

        Args:
            max_depth: 最大跳数（默认3跳）
            min_confidence: 最小置信度阈值
            use_llm_inference: 是否使用LLM进行语义推理
            batch_size: 批处理大小
        """
        self.graph = connection_manager.get_connection()
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.use_llm_inference = use_llm_inference
        self.batch_size = batch_size

        # LLM用于复杂推理
        self._llm = None

        # 性能统计
        self.stats = {
            'paths_found': 0,
            'paths_analyzed': 0,
            'relations_inferred': 0,
            'relations_written': 0,
            'llm_calls': 0
        }

        # 缓存已处理的实体对
        self.processed_pairs: Set[Tuple[str, str]] = set()

    @property
    def llm(self):
        """延迟加载LLM"""
        if self._llm is None and self.use_llm_inference:
            self._llm = get_llm_model()
        return self._llm

    def discover_all(self) -> Dict[str, Any]:
        """
        在整个图谱中发现多跳关系

        Returns:
            Dict: 发现结果统计
        """
        start_time = time.time()
        all_inferences = []

        print(f"开始多跳关系发现，最大深度: {self.max_depth}")

        # 获取种子实体（高连接度的实体）
        seed_entities = self._get_seed_entities()
        print(f"找到 {len(seed_entities)} 个种子实体")

        # 批量处理
        for i in range(0, len(seed_entities), self.batch_size):
            batch = seed_entities[i:i + self.batch_size]
            batch_inferences = self._process_entity_batch(batch)
            all_inferences.extend(batch_inferences)

            print(f"已处理 {min(i + self.batch_size, len(seed_entities))}/{len(seed_entities)} 个种子实体，"
                  f"发现 {len(all_inferences)} 个潜在关系")

        # 过滤并写入关系
        written = self._write_inferred_relations(all_inferences)

        elapsed = time.time() - start_time
        print(f"\n多跳关系发现完成:")
        print(f"  - 分析的路径数: {self.stats['paths_analyzed']}")
        print(f"  - 发现的潜在关系: {len(all_inferences)}")
        print(f"  - 写入的新关系: {written}")
        print(f"  - 总耗时: {elapsed:.2f}秒")

        return {
            'relations_found': len(all_inferences),
            'relations_written': written,
            'paths_analyzed': self.stats['paths_analyzed'],
            'elapsed_time': elapsed
        }

    def _get_seed_entities(self, limit: int = 1000) -> List[str]:
        """
        获取种子实体（高连接度的重要实体）

        Args:
            limit: 最大返回数量

        Returns:
            List[str]: 实体ID列表
        """
        query = """
        MATCH (e:`__Entity__`)
        WHERE e.id IS NOT NULL
        WITH e, COUNT { (e)--() } AS degree
        WHERE degree >= 2
        RETURN e.id AS entity_id
        ORDER BY degree DESC
        LIMIT $limit
        """

        result = self.graph.query(query, params={'limit': limit})
        return [r['entity_id'] for r in result]

    def _process_entity_batch(self, entities: List[str]) -> List[RelationInference]:
        """
        批量处理实体，发现多跳关系

        Args:
            entities: 实体列表

        Returns:
            List[RelationInference]: 推断结果列表
        """
        inferences = []

        for entity in entities:
            # 发现从该实体出发的多跳路径
            paths = self._find_multi_hop_paths(entity)
            self.stats['paths_found'] += len(paths)

            # 分析路径并推断关系
            for path in paths:
                self.stats['paths_analyzed'] += 1

                # 跳过已处理的实体对
                pair = (path.source, path.target)
                if pair in self.processed_pairs:
                    continue
                self.processed_pairs.add(pair)

                # 尝试推断关系
                inference = self._infer_relation(path)
                if inference and inference.confidence >= self.min_confidence:
                    inferences.append(inference)

        return inferences

    def _find_multi_hop_paths(self, start_entity: str) -> List[MultiHopPath]:
        """
        从指定实体出发，找出所有多跳路径

        Args:
            start_entity: 起始实体

        Returns:
            List[MultiHopPath]: 路径列表
        """
        paths = []

        # 使用Cypher查询多跳路径
        for depth in range(2, self.max_depth + 1):
            query = self._build_path_query(depth)

            try:
                result = self.graph.query(query, params={
                    'start_id': start_entity,
                    'limit': 50  # 每个深度限制路径数
                })

                for record in result:
                    path = MultiHopPath(
                        source=start_entity,
                        target=record['end_entity'],
                        path=record['path_entities'],
                        relations=record['path_relations'],
                        depth=depth
                    )
                    paths.append(path)

            except Exception as e:
                print(f"查询{depth}跳路径时出错: {e}")

        return paths

    def _build_path_query(self, depth: int) -> str:
        """
        构建指定深度的路径查询

        Args:
            depth: 路径深度

        Returns:
            str: Cypher查询语句
        """
        # 构建动态路径模式
        # 例如 depth=2: (e1)-[r1]->(e2)-[r2]->(e3)
        path_pattern = "(e0:`__Entity__` {id: $start_id})"
        for i in range(depth):
            path_pattern += f"-[r{i}]->(e{i+1}:`__Entity__`)"

        # 收集路径上的实体和关系
        entity_collect = ", ".join([f"e{i}.id" for i in range(depth + 1)])
        relation_collect = ", ".join([f"type(r{i})" for i in range(depth)])

        query = f"""
        MATCH {path_pattern}
        WHERE e0.id <> e{depth}.id
        AND NOT (e0)-[]-(e{depth})  // 排除已有直接关系的
        RETURN e{depth}.id AS end_entity,
               [{entity_collect}] AS path_entities,
               [{relation_collect}] AS path_relations
        LIMIT $limit
        """

        return query

    def _infer_relation(self, path: MultiHopPath) -> Optional[RelationInference]:
        """
        从路径推断关系

        Args:
            path: 多跳路径

        Returns:
            Optional[RelationInference]: 推断结果
        """
        # 策略1: 使用预定义规则
        inferred = self._apply_composition_rules(path.relations)
        if inferred:
            return RelationInference(
                source=path.source,
                target=path.target,
                inferred_relation=inferred,
                supporting_paths=[path],
                confidence=self._calculate_confidence(path),
                evidence=f"规则推断: {' -> '.join(path.relations)} => {inferred}"
            )

        # 策略2: 使用LLM推理（仅对复杂情况）
        if self.use_llm_inference and len(path.relations) >= 2:
            inferred = self._llm_infer_relation(path)
            if inferred:
                return RelationInference(
                    source=path.source,
                    target=path.target,
                    inferred_relation=inferred,
                    supporting_paths=[path],
                    confidence=self._calculate_confidence(path) * 0.8,  # LLM推理降低置信度
                    evidence=f"LLM推断: {' -> '.join(path.relations)} => {inferred}"
                )

        return None

    def _apply_composition_rules(self, relations: List[str]) -> Optional[str]:
        """
        应用关系组合规则

        Args:
            relations: 关系序列

        Returns:
            Optional[str]: 推断出的关系，或None
        """
        if len(relations) < 2:
            return None

        # 逐步组合关系
        current = relations[0]
        for i in range(1, len(relations)):
            next_rel = relations[i]

            # 查找组合规则
            key = (current, next_rel)
            if key in RELATION_COMPOSITION_RULES:
                current = RELATION_COMPOSITION_RULES[key]
            else:
                # 尝试通用规则
                current = self._apply_generic_rules(current, next_rel)
                if current is None:
                    return None

        return current

    def _apply_generic_rules(self, rel1: str, rel2: str) -> Optional[str]:
        """
        应用通用组合规则

        Args:
            rel1: 第一个关系
            rel2: 第二个关系

        Returns:
            Optional[str]: 组合后的关系
        """
        # 相同关系的传递
        if rel1 == rel2:
            return f"间接{rel1}"

        # 包含"相关"的关系
        if "相关" in rel1 or "相关" in rel2:
            return "可能相关"

        # 包含"影响"的关系
        if "影响" in rel1 or "影响" in rel2:
            return "间接影响"

        return None

    def _llm_infer_relation(self, path: MultiHopPath) -> Optional[str]:
        """
        使用LLM推断关系

        Args:
            path: 多跳路径

        Returns:
            Optional[str]: 推断出的关系
        """
        if not self.llm:
            return None

        self.stats['llm_calls'] += 1

        # 构建提示
        path_desc = " -> ".join([
            f"{path.path[i]} --[{path.relations[i]}]--> {path.path[i+1]}"
            for i in range(len(path.relations))
        ])

        prompt = f"""分析以下知识图谱路径，判断起点和终点之间是否存在潜在的直接关系。

路径: {path_desc}

起点实体: {path.source}
终点实体: {path.target}

如果存在潜在的直接关系，请只输出关系类型（如"间接影响"、"可能相关"等）。
如果不存在明确的关系，请输出"无"。

关系类型:"""

        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()

            # 验证结果
            if result and result != "无" and len(result) < 20:
                return result

        except Exception as e:
            print(f"LLM推理失败: {e}")

        return None

    def _calculate_confidence(self, path: MultiHopPath) -> float:
        """
        计算路径的置信度

        Args:
            path: 多跳路径

        Returns:
            float: 置信度（0-1）
        """
        # 基础置信度随跳数衰减
        base_confidence = 1.0 / path.depth

        # 如果关系有权重，考虑权重
        # 这里简化处理，实际可以查询边的权重属性

        return min(base_confidence * 1.5, 1.0)  # 适当提升并限制上限

    def _write_inferred_relations(self, inferences: List[RelationInference]) -> int:
        """
        将推断的关系写入图谱

        Args:
            inferences: 推断结果列表

        Returns:
            int: 写入的关系数
        """
        if not inferences:
            return 0

        written = 0

        for inference in inferences:
            try:
                # 创建推断关系
                query = """
                MATCH (source:`__Entity__` {id: $source_id})
                MATCH (target:`__Entity__` {id: $target_id})
                WHERE NOT (source)-[:INFERRED]->(target)
                CREATE (source)-[r:INFERRED {
                    relation_type: $relation_type,
                    confidence: $confidence,
                    evidence: $evidence,
                    hop_count: $hop_count,
                    created_at: datetime()
                }]->(target)
                RETURN r
                """

                result = self.graph.query(query, params={
                    'source_id': inference.source,
                    'target_id': inference.target,
                    'relation_type': inference.inferred_relation,
                    'confidence': inference.confidence,
                    'evidence': inference.evidence,
                    'hop_count': inference.supporting_paths[0].depth if inference.supporting_paths else 0
                })

                if result:
                    written += 1

            except Exception as e:
                print(f"写入推断关系失败: {e}")

        self.stats['relations_written'] = written
        return written

    def discover_between_entities(
        self,
        source: str,
        target: str
    ) -> List[RelationInference]:
        """
        发现两个特定实体之间的多跳关系

        Args:
            source: 源实体ID
            target: 目标实体ID

        Returns:
            List[RelationInference]: 推断结果列表
        """
        inferences = []

        # 查找两实体间的所有路径
        for depth in range(2, self.max_depth + 1):
            query = f"""
            MATCH path = (s:`__Entity__` {{id: $source}})-[*{depth}]->(t:`__Entity__` {{id: $target}})
            WHERE NOT (s)-[]-(t)  // 排除直接相连的情况
            WITH path, [r IN relationships(path) | type(r)] AS relations,
                 [n IN nodes(path) | n.id] AS entities
            RETURN entities AS path_entities,
                   relations AS path_relations
            LIMIT 10
            """

            try:
                result = self.graph.query(query, params={
                    'source': source,
                    'target': target
                })

                for record in result:
                    path = MultiHopPath(
                        source=source,
                        target=target,
                        path=record['path_entities'],
                        relations=record['path_relations'],
                        depth=depth
                    )

                    inference = self._infer_relation(path)
                    if inference:
                        inferences.append(inference)

            except Exception as e:
                print(f"查询路径时出错: {e}")

        return inferences

    def get_path_patterns(self) -> Dict[str, int]:
        """
        统计图谱中的路径模式

        Returns:
            Dict: 模式及其出现次数
        """
        pattern_counts = defaultdict(int)

        query = """
        MATCH (e1:`__Entity__`)-[r1]->(e2:`__Entity__`)-[r2]->(e3:`__Entity__`)
        WITH type(r1) + ' -> ' + type(r2) AS pattern
        RETURN pattern, count(*) AS count
        ORDER BY count DESC
        LIMIT 50
        """

        result = self.graph.query(query)

        for record in result:
            pattern_counts[record['pattern']] = record['count']

        return dict(pattern_counts)

    def suggest_composition_rules(self) -> List[Dict[str, Any]]:
        """
        基于图谱分析，建议新的关系组合规则

        Returns:
            List[Dict]: 建议的规则
        """
        suggestions = []

        # 获取频繁的2跳模式
        patterns = self.get_path_patterns()

        for pattern, count in patterns.items():
            if count < 5:  # 忽略低频模式
                continue

            parts = pattern.split(' -> ')
            if len(parts) != 2:
                continue

            rel1, rel2 = parts

            # 检查是否已有规则
            if (rel1, rel2) not in RELATION_COMPOSITION_RULES:
                # 建议可能的组合
                suggested_rel = self._suggest_relation_name(rel1, rel2)
                suggestions.append({
                    'pattern': (rel1, rel2),
                    'suggested_relation': suggested_rel,
                    'frequency': count
                })

        return suggestions

    def _suggest_relation_name(self, rel1: str, rel2: str) -> str:
        """
        建议组合关系的名称

        Args:
            rel1: 第一个关系
            rel2: 第二个关系

        Returns:
            str: 建议的关系名称
        """
        # 简单策略：如果两个关系相同，加"间接"前缀
        if rel1 == rel2:
            return f"间接{rel1}"

        # 否则用"经由"连接
        return f"{rel1}并{rel2}"


def add_composition_rule(rel1: str, rel2: str, inferred: str):
    """
    添加新的关系组合规则

    Args:
        rel1: 第一个关系
        rel2: 第二个关系
        inferred: 推断出的关系
    """
    RELATION_COMPOSITION_RULES[(rel1, rel2)] = inferred


if __name__ == "__main__":
    # 测试代码
    discoverer = MultiHopRelationDiscoverer(
        max_depth=3,
        min_confidence=0.5,
        use_llm_inference=False  # 测试时不使用LLM
    )

    # 获取路径模式
    print("分析路径模式...")
    patterns = discoverer.get_path_patterns()
    print(f"发现 {len(patterns)} 种路径模式")
    for pattern, count in list(patterns.items())[:10]:
        print(f"  {pattern}: {count}")

    # 建议规则
    print("\n建议的组合规则:")
    suggestions = discoverer.suggest_composition_rules()
    for s in suggestions[:5]:
        print(f"  {s['pattern']} => {s['suggested_relation']} (出现{s['frequency']}次)")

    # 执行发现
    print("\n执行多跳关系发现...")
    result = discoverer.discover_all()
    print(f"完成: {result}")
