"""
知识图谱模块 - 三元组存储、查询和推理
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Triple:
    """知识图谱三元组"""
    subject: str
    predicate: str
    object: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"({self.subject}, {self.predicate}, {self.object})"

    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)


class KnowledgeGraph:
    """心脏移植灌注知识图谱"""

    def __init__(self, config_dir: Optional[str] = None):
        """
        初始化知识图谱

        Args:
            config_dir: 配置目录路径
        """
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self.config_dir = Path(config_dir)

        # 三元组存储
        self.triples: List[Triple] = []

        # 索引结构
        self.subject_index: Dict[str, List[Triple]] = defaultdict(list)
        self.predicate_index: Dict[str, List[Triple]] = defaultdict(list)
        self.object_index: Dict[str, List[Triple]] = defaultdict(list)

        # 加载配置
        self._load_schema()
        self._load_strategies()

    def _load_schema(self):
        """加载知识图谱Schema"""
        schema_path = self.config_dir / "knowledge_graph_schema.yaml"
        if schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = yaml.safe_load(f)

            # 加载策略三元组
            strategy_triples = self.schema.get('strategy_triples', {})
            for category, triples in strategy_triples.items():
                for t in triples:
                    self.add_triple(
                        t['subject'],
                        t['predicate'],
                        t['object'],
                        {'category': category, 'source': 'schema'}
                    )

            # 加载药物知识
            drug_knowledge = self.schema.get('drug_knowledge', {})
            for category, triples in drug_knowledge.items():
                for t in triples:
                    self.add_triple(
                        t['subject'],
                        t['predicate'],
                        t['object'],
                        {'category': f'drug_{category}', 'source': 'schema'}
                    )

    def _load_strategies(self):
        """加载干预策略中的三元组"""
        strategies_path = self.config_dir / "intervention_strategies.yaml"
        if strategies_path.exists():
            with open(strategies_path, 'r', encoding='utf-8') as f:
                strategies = yaml.safe_load(f)

            # 提取所有策略中的kg_triples
            self._extract_triples_from_dict(strategies, 'intervention_strategies')

    def _extract_triples_from_dict(self, d: Dict, source: str):
        """递归提取字典中的kg_triples"""
        if isinstance(d, dict):
            if 'kg_triples' in d:
                for triple in d['kg_triples']:
                    if isinstance(triple, list) and len(triple) == 3:
                        self.add_triple(
                            triple[0], triple[1], triple[2],
                            {'source': source}
                        )
            for v in d.values():
                self._extract_triples_from_dict(v, source)
        elif isinstance(d, list):
            for item in d:
                self._extract_triples_from_dict(item, source)

    def add_triple(self, subject: str, predicate: str, obj: str,
                   metadata: Optional[Dict] = None):
        """
        添加三元组

        Args:
            subject: 主体
            predicate: 谓词/关系
            obj: 客体
            metadata: 元数据
        """
        triple = Triple(subject, predicate, obj, metadata or {})

        # 检查重复
        if triple.to_tuple() not in [t.to_tuple() for t in self.triples]:
            self.triples.append(triple)
            self.subject_index[subject].append(triple)
            self.predicate_index[predicate].append(triple)
            self.object_index[obj].append(triple)

    def query_by_subject(self, subject: str) -> List[Triple]:
        """按主体查询"""
        return self.subject_index.get(subject, [])

    def query_by_predicate(self, predicate: str) -> List[Triple]:
        """按谓词查询"""
        return self.predicate_index.get(predicate, [])

    def query_by_object(self, obj: str) -> List[Triple]:
        """按客体查询"""
        return self.object_index.get(obj, [])

    def query(self, subject: Optional[str] = None,
              predicate: Optional[str] = None,
              obj: Optional[str] = None) -> List[Triple]:
        """
        通用查询

        Args:
            subject: 主体（可选）
            predicate: 谓词（可选）
            obj: 客体（可选）

        Returns:
            匹配的三元组列表
        """
        results = self.triples

        if subject:
            results = [t for t in results if t.subject == subject]
        if predicate:
            results = [t for t in results if t.predicate == predicate]
        if obj:
            results = [t for t in results if t.object == obj]

        return results

    def find_interventions(self, abnormality: str) -> List[Dict[str, Any]]:
        """
        查找异常状态对应的干预措施

        Args:
            abnormality: 异常状态（如 "EF_Low", "Hyperkalemia"）

        Returns:
            干预措施列表
        """
        interventions = []

        # 查找直接干预
        for triple in self.query_by_subject(abnormality):
            if triple.predicate in ['requires_intervention', 'requires_immediate',
                                     'treat_with', 'stabilize_with', 'shift_with']:
                interventions.append({
                    'action': triple.object,
                    'type': triple.predicate,
                    'urgency': 'immediate' if 'immediate' in triple.predicate else 'routine'
                })

            # 首先检查项
            elif triple.predicate in ['first_check', 'check', 'assess']:
                interventions.append({
                    'action': f"检查: {triple.object}",
                    'type': 'diagnostic',
                    'urgency': 'first'
                })

            # 升级路径
            elif triple.predicate in ['escalate_to', 'if_no_improvement']:
                interventions.append({
                    'action': triple.object,
                    'type': 'escalation',
                    'urgency': 'if_no_improvement'
                })

        return interventions

    def find_causes(self, abnormality: str) -> List[str]:
        """查找异常可能的原因/指示"""
        causes = []
        for triple in self.query_by_subject(abnormality):
            if triple.predicate == 'indicates':
                causes.append(triple.object)
        return causes

    def find_risks(self, abnormality: str) -> List[str]:
        """查找异常的风险"""
        risks = []
        for triple in self.query_by_subject(abnormality):
            if triple.predicate in ['risk_of', 'causes', 'life_threatening']:
                risks.append(triple.object)
        return risks

    def find_drug_info(self, drug: str) -> Dict[str, Any]:
        """查找药物信息"""
        info = {}
        for triple in self.query_by_subject(drug):
            info[triple.predicate] = triple.object
        return info

    def get_escalation_path(self, condition: str) -> List[Dict]:
        """
        获取升级路径

        Args:
            condition: 初始条件

        Returns:
            升级步骤列表
        """
        paths = self.schema.get('escalation_pathways', {})
        for pathway_name, pathway in paths.items():
            if condition.lower() in pathway_name.lower():
                return pathway.get('path', [])
        return []

    def explain_intervention(self, abnormality: str) -> str:
        """
        生成干预解释

        Args:
            abnormality: 异常状态

        Returns:
            解释文本
        """
        lines = [f"## {abnormality} 干预指南\n"]

        # 原因分析
        causes = self.find_causes(abnormality)
        if causes:
            lines.append("### 可能原因")
            for c in causes:
                lines.append(f"- {c}")
            lines.append("")

        # 风险提示
        risks = self.find_risks(abnormality)
        if risks:
            lines.append("### 风险警告")
            for r in risks:
                lines.append(f"- ⚠️ {r}")
            lines.append("")

        # 干预措施
        interventions = self.find_interventions(abnormality)
        if interventions:
            # 按优先级分组
            first_checks = [i for i in interventions if i['urgency'] == 'first']
            immediate = [i for i in interventions if i['urgency'] == 'immediate']
            routine = [i for i in interventions if i['urgency'] == 'routine']
            escalation = [i for i in interventions if i['urgency'] == 'if_no_improvement']

            if first_checks:
                lines.append("### 首先检查")
                for i in first_checks:
                    lines.append(f"- {i['action']}")
                lines.append("")

            if immediate:
                lines.append("### 立即干预")
                for i in immediate:
                    lines.append(f"- 🔴 {i['action']}")
                    # 查找药物信息
                    drug_info = self.find_drug_info(i['action'])
                    if drug_info:
                        if 'dose_range' in drug_info:
                            lines.append(f"  - 剂量: {drug_info['dose_range']}")
                lines.append("")

            if routine:
                lines.append("### 常规干预")
                for i in routine:
                    lines.append(f"- {i['action']}")
                lines.append("")

            if escalation:
                lines.append("### 升级措施（如无改善）")
                for i in escalation:
                    lines.append(f"- ⬆️ {i['action']}")
                lines.append("")

        return "\n".join(lines)

    def search_related_knowledge(self, query: str, max_depth: int = 2) -> List[Triple]:
        """
        搜索相关知识（广度优先）

        Args:
            query: 查询词
            max_depth: 最大搜索深度

        Returns:
            相关三元组列表
        """
        visited: Set[str] = set()
        results: List[Triple] = []
        queue = [(query, 0)]

        while queue:
            current, depth = queue.pop(0)
            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            # 查找以当前词为主体的三元组
            for triple in self.query_by_subject(current):
                if triple not in results:
                    results.append(triple)
                if depth < max_depth:
                    queue.append((triple.object, depth + 1))

            # 查找以当前词为客体的三元组
            for triple in self.query_by_object(current):
                if triple not in results:
                    results.append(triple)
                if depth < max_depth:
                    queue.append((triple.subject, depth + 1))

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        return {
            'total_triples': len(self.triples),
            'unique_subjects': len(self.subject_index),
            'unique_predicates': len(self.predicate_index),
            'unique_objects': len(self.object_index),
            'predicates': list(self.predicate_index.keys())
        }

    def export_triples(self) -> List[Tuple[str, str, str]]:
        """导出所有三元组"""
        return [t.to_tuple() for t in self.triples]

    def to_networkx(self):
        """
        转换为NetworkX图（用于可视化）

        Returns:
            NetworkX DiGraph
        """
        try:
            import networkx as nx
            G = nx.DiGraph()

            for triple in self.triples:
                G.add_edge(
                    triple.subject,
                    triple.object,
                    relation=triple.predicate
                )

            return G
        except ImportError:
            print("需要安装 networkx: pip install networkx")
            return None


if __name__ == "__main__":
    # 测试示例
    kg = KnowledgeGraph()

    print("=== 知识图谱统计 ===")
    stats = kg.get_statistics()
    print(f"三元组总数: {stats['total_triples']}")
    print(f"唯一主体数: {stats['unique_subjects']}")
    print(f"唯一谓词数: {stats['unique_predicates']}")
    print(f"谓词列表: {stats['predicates']}")

    print("\n=== 查询测试 ===")

    # 查询高钾血症的干预
    print("\n高钾血症 (Hyperkalemia) 干预:")
    explanation = kg.explain_intervention("Hyperkalemia")
    print(explanation)

    # 查询CI危急的干预
    print("\nCI危急 (CI_Critical) 干预:")
    explanation = kg.explain_intervention("CI_Critical")
    print(explanation)

    # 搜索相关知识
    print("\n=== 搜索相关知识: Inotrope_Support ===")
    related = kg.search_related_knowledge("Inotrope_Support")
    for t in related[:10]:
        print(f"  {t}")
