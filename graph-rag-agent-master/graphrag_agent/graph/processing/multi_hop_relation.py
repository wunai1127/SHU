"""
Multi-hop Relation Discovery Module

Discover potential multi-hop (indirect) relations in triples/knowledge graphs

Core idea:
- If A -[r1]-> B and B -[r2]-> C, then possibly A -[inferred_r]-> C
- Analyze path patterns to infer implicit direct relationships
- Support configurable relation composition rules

Usage:
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
    """Multi-hop path representation"""
    source: str                      # Source entity
    target: str                      # Target entity
    path: List[str]                  # Full path [e1, e2, e3, ...]
    relations: List[str]             # Relation sequence [r1, r2, ...]
    depth: int                       # Number of hops
    confidence: float = 1.0          # Confidence score
    inferred_relation: str = ""      # Inferred relation type


@dataclass
class RelationInference:
    """Relation inference result"""
    source: str
    target: str
    inferred_relation: str
    supporting_paths: List[MultiHopPath]
    confidence: float
    evidence: str = ""


# Predefined relation composition rules
# Format: (relation1, relation2) -> inferred_relation
RELATION_COMPOSITION_RULES = {
    # Hierarchical relation transitivity
    ("belongs_to", "belongs_to"): "indirectly_belongs_to",
    ("contains", "contains"): "indirectly_contains",
    ("is_a", "is_a"): "is_a",
    ("part_of", "part_of"): "indirectly_part_of",

    # Location relations
    ("located_in", "located_in"): "located_in",
    ("located_at", "located_at"): "located_at",

    # Causal relations
    ("causes", "causes"): "indirectly_causes",
    ("leads_to", "causes"): "indirectly_causes",
    ("causes", "leads_to"): "indirectly_causes",
    ("results_in", "results_in"): "indirectly_results_in",

    # Temporal relations
    ("after", "after"): "long_after",
    ("before", "before"): "long_before",
    ("follows", "follows"): "eventually_follows",
    ("precedes", "precedes"): "eventually_precedes",

    # Association relations
    ("related_to", "related_to"): "possibly_related",
    ("associated_with", "associated_with"): "indirectly_associated",
    ("connected_to", "connected_to"): "indirectly_connected",

    # Effect relations
    ("affects", "affects"): "indirectly_affects",
    ("influences", "affects"): "indirectly_affects",
    ("impacts", "impacts"): "indirectly_impacts",

    # Composition relations
    ("composed_of", "composed_of"): "composed_of",
    ("includes", "includes"): "includes",
    ("has_part", "has_part"): "has_part",

    # Medical/disease-symptom-treatment chain
    ("manifests_as", "treats"): "indirect_treatment_target",
    ("causes", "treats"): "symptomatic_treatment",
    ("has_symptom", "treated_by"): "treatment_for_symptom",

    # Knowledge dependency
    ("requires", "requires"): "indirectly_requires",
    ("depends_on", "depends_on"): "indirectly_depends_on",

    # Ownership/possession
    ("owns", "owns"): "indirectly_owns",
    ("has", "has"): "indirectly_has",

    # Creation/production
    ("creates", "creates"): "indirectly_creates",
    ("produces", "produces"): "indirectly_produces",

    # Usage relations
    ("uses", "uses"): "indirectly_uses",
    ("used_by", "used_by"): "indirectly_used_by",
}


class MultiHopRelationDiscoverer:
    """
    Multi-hop Relation Discoverer

    Discover potential indirect relations by analyzing path patterns in the graph

    Discovery strategies:
    1. Path enumeration - Find all k-hop paths between entities
    2. Pattern matching - Use predefined rules to infer relations
    3. Semantic reasoning - Use LLM for complex relation inference
    4. Confidence calculation - Based on path weights and occurrence frequency
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_confidence: float = 0.5,
        use_llm_inference: bool = True,
        batch_size: int = 100
    ):
        """
        Initialize multi-hop relation discoverer

        Args:
            max_depth: Maximum number of hops (default 3)
            min_confidence: Minimum confidence threshold
            use_llm_inference: Whether to use LLM for semantic reasoning
            batch_size: Batch processing size
        """
        self.graph = connection_manager.get_connection()
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.use_llm_inference = use_llm_inference
        self.batch_size = batch_size

        # LLM for complex reasoning
        self._llm = None

        # Performance statistics
        self.stats = {
            'paths_found': 0,
            'paths_analyzed': 0,
            'relations_inferred': 0,
            'relations_written': 0,
            'llm_calls': 0
        }

        # Cache processed entity pairs
        self.processed_pairs: Set[Tuple[str, str]] = set()

    @property
    def llm(self):
        """Lazy load LLM"""
        if self._llm is None and self.use_llm_inference:
            self._llm = get_llm_model()
        return self._llm

    def discover_all(self) -> Dict[str, Any]:
        """
        Discover multi-hop relations across the entire graph

        Returns:
            Dict: Discovery result statistics
        """
        start_time = time.time()
        all_inferences = []

        print(f"Starting multi-hop relation discovery, max depth: {self.max_depth}")

        # Get seed entities (high-degree entities)
        seed_entities = self._get_seed_entities()
        print(f"Found {len(seed_entities)} seed entities")

        # Batch processing
        for i in range(0, len(seed_entities), self.batch_size):
            batch = seed_entities[i:i + self.batch_size]
            batch_inferences = self._process_entity_batch(batch)
            all_inferences.extend(batch_inferences)

            print(f"Processed {min(i + self.batch_size, len(seed_entities))}/{len(seed_entities)} seed entities, "
                  f"found {len(all_inferences)} potential relations")

        # Filter and write relations
        written = self._write_inferred_relations(all_inferences)

        elapsed = time.time() - start_time
        print(f"\nMulti-hop relation discovery complete:")
        print(f"  - Paths analyzed: {self.stats['paths_analyzed']}")
        print(f"  - Potential relations found: {len(all_inferences)}")
        print(f"  - New relations written: {written}")
        print(f"  - Total time: {elapsed:.2f}s")

        return {
            'relations_found': len(all_inferences),
            'relations_written': written,
            'paths_analyzed': self.stats['paths_analyzed'],
            'elapsed_time': elapsed
        }

    def _get_seed_entities(self, limit: int = 1000) -> List[str]:
        """
        Get seed entities (high-degree important entities)

        Args:
            limit: Maximum number to return

        Returns:
            List[str]: Entity ID list
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
        Batch process entities to discover multi-hop relations

        Args:
            entities: Entity list

        Returns:
            List[RelationInference]: Inference result list
        """
        inferences = []

        for entity in entities:
            # Discover multi-hop paths from this entity
            paths = self._find_multi_hop_paths(entity)
            self.stats['paths_found'] += len(paths)

            # Analyze paths and infer relations
            for path in paths:
                self.stats['paths_analyzed'] += 1

                # Skip already processed entity pairs
                pair = (path.source, path.target)
                if pair in self.processed_pairs:
                    continue
                self.processed_pairs.add(pair)

                # Try to infer relation
                inference = self._infer_relation(path)
                if inference and inference.confidence >= self.min_confidence:
                    inferences.append(inference)

        return inferences

    def _find_multi_hop_paths(self, start_entity: str) -> List[MultiHopPath]:
        """
        Find all multi-hop paths from a specified entity

        Args:
            start_entity: Starting entity

        Returns:
            List[MultiHopPath]: Path list
        """
        paths = []

        # Use Cypher to query multi-hop paths
        for depth in range(2, self.max_depth + 1):
            query = self._build_path_query(depth)

            try:
                result = self.graph.query(query, params={
                    'start_id': start_entity,
                    'limit': 50  # Limit paths per depth
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
                print(f"Error querying {depth}-hop paths: {e}")

        return paths

    def _build_path_query(self, depth: int) -> str:
        """
        Build path query for specified depth

        Args:
            depth: Path depth

        Returns:
            str: Cypher query
        """
        # Build dynamic path pattern
        # e.g. depth=2: (e1)-[r1]->(e2)-[r2]->(e3)
        path_pattern = "(e0:`__Entity__` {id: $start_id})"
        for i in range(depth):
            path_pattern += f"-[r{i}]->(e{i+1}:`__Entity__`)"

        # Collect entities and relations on path
        entity_collect = ", ".join([f"e{i}.id" for i in range(depth + 1)])
        relation_collect = ", ".join([f"type(r{i})" for i in range(depth)])

        query = f"""
        MATCH {path_pattern}
        WHERE e0.id <> e{depth}.id
        AND NOT (e0)-[]-(e{depth})  // Exclude existing direct relations
        RETURN e{depth}.id AS end_entity,
               [{entity_collect}] AS path_entities,
               [{relation_collect}] AS path_relations
        LIMIT $limit
        """

        return query

    def _infer_relation(self, path: MultiHopPath) -> Optional[RelationInference]:
        """
        Infer relation from path

        Args:
            path: Multi-hop path

        Returns:
            Optional[RelationInference]: Inference result
        """
        # Strategy 1: Use predefined rules
        inferred = self._apply_composition_rules(path.relations)
        if inferred:
            return RelationInference(
                source=path.source,
                target=path.target,
                inferred_relation=inferred,
                supporting_paths=[path],
                confidence=self._calculate_confidence(path),
                evidence=f"Rule inference: {' -> '.join(path.relations)} => {inferred}"
            )

        # Strategy 2: Use LLM reasoning (for complex cases only)
        if self.use_llm_inference and len(path.relations) >= 2:
            inferred = self._llm_infer_relation(path)
            if inferred:
                return RelationInference(
                    source=path.source,
                    target=path.target,
                    inferred_relation=inferred,
                    supporting_paths=[path],
                    confidence=self._calculate_confidence(path) * 0.8,  # Lower confidence for LLM inference
                    evidence=f"LLM inference: {' -> '.join(path.relations)} => {inferred}"
                )

        return None

    def _apply_composition_rules(self, relations: List[str]) -> Optional[str]:
        """
        Apply relation composition rules

        Args:
            relations: Relation sequence

        Returns:
            Optional[str]: Inferred relation, or None
        """
        if len(relations) < 2:
            return None

        # Compose relations step by step
        current = relations[0]
        for i in range(1, len(relations)):
            next_rel = relations[i]

            # Look up composition rule
            key = (current, next_rel)
            if key in RELATION_COMPOSITION_RULES:
                current = RELATION_COMPOSITION_RULES[key]
            else:
                # Try generic rules
                current = self._apply_generic_rules(current, next_rel)
                if current is None:
                    return None

        return current

    def _apply_generic_rules(self, rel1: str, rel2: str) -> Optional[str]:
        """
        Apply generic composition rules

        Args:
            rel1: First relation
            rel2: Second relation

        Returns:
            Optional[str]: Composed relation
        """
        # Transitivity for identical relations
        if rel1 == rel2:
            return f"indirectly_{rel1}"

        # Relations containing "related"
        if "related" in rel1.lower() or "related" in rel2.lower():
            return "possibly_related"

        # Relations containing "affects" or "influences"
        if "affect" in rel1.lower() or "affect" in rel2.lower():
            return "indirectly_affects"

        if "influence" in rel1.lower() or "influence" in rel2.lower():
            return "indirectly_influences"

        # Relations containing "causes"
        if "cause" in rel1.lower() or "cause" in rel2.lower():
            return "indirectly_causes"

        return None

    def _llm_infer_relation(self, path: MultiHopPath) -> Optional[str]:
        """
        Use LLM to infer relation

        Args:
            path: Multi-hop path

        Returns:
            Optional[str]: Inferred relation
        """
        if not self.llm:
            return None

        self.stats['llm_calls'] += 1

        # Build prompt
        path_desc = " -> ".join([
            f"{path.path[i]} --[{path.relations[i]}]--> {path.path[i+1]}"
            for i in range(len(path.relations))
        ])

        prompt = f"""Analyze the following knowledge graph path and determine if there is a potential direct relationship between the start and end entities.

Path: {path_desc}

Start entity: {path.source}
End entity: {path.target}

If a potential direct relationship exists, output only the relationship type (e.g., "indirectly_affects", "possibly_related", etc.).
If no clear relationship exists, output "none".

Relationship type:"""

        try:
            response = self.llm.invoke(prompt)
            result = response.content.strip()

            # Validate result
            if result and result.lower() != "none" and len(result) < 30:
                return result

        except Exception as e:
            print(f"LLM inference failed: {e}")

        return None

    def _calculate_confidence(self, path: MultiHopPath) -> float:
        """
        Calculate path confidence

        Args:
            path: Multi-hop path

        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence decays with hop count
        base_confidence = 1.0 / path.depth

        # If relations have weights, consider them
        # Simplified here, actual implementation could query edge weight properties

        return min(base_confidence * 1.5, 1.0)  # Boost slightly and cap at 1.0

    def _write_inferred_relations(self, inferences: List[RelationInference]) -> int:
        """
        Write inferred relations to graph

        Args:
            inferences: Inference result list

        Returns:
            int: Number of relations written
        """
        if not inferences:
            return 0

        written = 0

        for inference in inferences:
            try:
                # Create inferred relation
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
                print(f"Failed to write inferred relation: {e}")

        self.stats['relations_written'] = written
        return written

    def discover_between_entities(
        self,
        source: str,
        target: str
    ) -> List[RelationInference]:
        """
        Discover multi-hop relations between two specific entities

        Args:
            source: Source entity ID
            target: Target entity ID

        Returns:
            List[RelationInference]: Inference result list
        """
        inferences = []

        # Find all paths between the two entities
        for depth in range(2, self.max_depth + 1):
            query = f"""
            MATCH path = (s:`__Entity__` {{id: $source}})-[*{depth}]->(t:`__Entity__` {{id: $target}})
            WHERE NOT (s)-[]-(t)  // Exclude directly connected cases
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
                print(f"Error querying paths: {e}")

        return inferences

    def get_path_patterns(self) -> Dict[str, int]:
        """
        Get path pattern statistics from the graph

        Returns:
            Dict: Pattern and occurrence count
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
        Suggest new relation composition rules based on graph analysis

        Returns:
            List[Dict]: Suggested rules
        """
        suggestions = []

        # Get frequent 2-hop patterns
        patterns = self.get_path_patterns()

        for pattern, count in patterns.items():
            if count < 5:  # Ignore low-frequency patterns
                continue

            parts = pattern.split(' -> ')
            if len(parts) != 2:
                continue

            rel1, rel2 = parts

            # Check if rule already exists
            if (rel1, rel2) not in RELATION_COMPOSITION_RULES:
                # Suggest possible composition
                suggested_rel = self._suggest_relation_name(rel1, rel2)
                suggestions.append({
                    'pattern': (rel1, rel2),
                    'suggested_relation': suggested_rel,
                    'frequency': count
                })

        return suggestions

    def _suggest_relation_name(self, rel1: str, rel2: str) -> str:
        """
        Suggest composed relation name

        Args:
            rel1: First relation
            rel2: Second relation

        Returns:
            str: Suggested relation name
        """
        # Simple strategy: if same relation, add "indirectly_" prefix
        if rel1 == rel2:
            return f"indirectly_{rel1}"

        # Otherwise combine with "via"
        return f"{rel1}_via_{rel2}"


def add_composition_rule(rel1: str, rel2: str, inferred: str):
    """
    Add new relation composition rule

    Args:
        rel1: First relation
        rel2: Second relation
        inferred: Inferred relation
    """
    RELATION_COMPOSITION_RULES[(rel1, rel2)] = inferred


if __name__ == "__main__":
    # Test code
    discoverer = MultiHopRelationDiscoverer(
        max_depth=3,
        min_confidence=0.5,
        use_llm_inference=False  # Don't use LLM for testing
    )

    # Get path patterns
    print("Analyzing path patterns...")
    patterns = discoverer.get_path_patterns()
    print(f"Found {len(patterns)} path patterns")
    for pattern, count in list(patterns.items())[:10]:
        print(f"  {pattern}: {count}")

    # Suggest rules
    print("\nSuggested composition rules:")
    suggestions = discoverer.suggest_composition_rules()
    for s in suggestions[:5]:
        print(f"  {s['pattern']} => {s['suggested_relation']} (frequency: {s['frequency']})")

    # Execute discovery
    print("\nExecuting multi-hop relation discovery...")
    result = discoverer.discover_all()
    print(f"Complete: {result}")
