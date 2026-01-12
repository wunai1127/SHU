"""
Local Triple Normalization and Multi-hop Relation Discovery Script

本地运行的三元组归一化和多跳关系发现脚本
不依赖 Neo4j，直接在 CSV/JSON 文件上运行

Usage:
    python run_normalization.py --input triples_csv/ --output normalized_output/

Requirements:
    pip install pandas numpy scikit-learn sentence-transformers networkx tqdm
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional: for embedding-based similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not installed. Using string similarity only.")

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Multi-hop discovery will be limited.")


# ============================================================================
# Part 1: Entity Normalization (实体归一化)
# ============================================================================

@dataclass
class NormalizationConfig:
    """Configuration for entity normalization"""
    # String similarity threshold (Levenshtein-based)
    string_similarity_threshold: float = 0.85

    # Embedding similarity threshold (if using sentence-transformers)
    embedding_similarity_threshold: float = 0.90

    # Whether to use embeddings for similarity
    use_embeddings: bool = True

    # Minimum entity frequency to consider for normalization
    min_entity_frequency: int = 1

    # Output format
    output_format: str = "csv"  # or "json"


class EntityNormalizer:
    """
    Entity normalizer for triple data

    Normalizes entities by:
    1. Finding similar entities (string similarity + optional embedding similarity)
    2. Grouping them into clusters
    3. Selecting canonical names for each cluster
    4. Updating triples with canonical names
    """

    def __init__(self, config: NormalizationConfig = None):
        self.config = config or NormalizationConfig()
        self.embedding_model = None

        if self.config.use_embeddings and HAS_SENTENCE_TRANSFORMERS:
            print("Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_from_csv(
        self,
        nodes_csv: str,
        edges_csv: str,
        output_dir: str
    ) -> Dict[str, any]:
        """
        Main entry point: normalize entities from CSV files

        Args:
            nodes_csv: Path to nodes CSV (columns: name:ID, type, :LABEL)
            edges_csv: Path to edges CSV (columns: :START_ID, :END_ID, relation, :TYPE)
            output_dir: Directory to save normalized output

        Returns:
            Statistics dict
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading nodes from {nodes_csv}...")
        nodes_df = pd.read_csv(nodes_csv)

        print(f"Loading edges from {edges_csv}...")
        edges_df = pd.read_csv(edges_csv)

        # Extract unique entities
        entities = nodes_df['name:ID'].unique().tolist()
        print(f"Found {len(entities)} unique entities")

        # Step 1: Find similar entity pairs
        print("\nStep 1: Finding similar entities...")
        similar_pairs = self._find_similar_entities(entities)
        print(f"Found {len(similar_pairs)} similar entity pairs")

        # Step 2: Build clusters from similar pairs
        print("\nStep 2: Building entity clusters...")
        clusters = self._build_clusters(similar_pairs, entities)
        print(f"Built {len(clusters)} clusters (groups of similar entities)")

        # Step 3: Select canonical name for each cluster
        print("\nStep 3: Selecting canonical names...")
        canonical_mapping = self._select_canonical_names(clusters, edges_df)

        # Step 4: Update nodes and edges with canonical names
        print("\nStep 4: Updating triples with canonical names...")
        normalized_nodes, normalized_edges = self._apply_normalization(
            nodes_df, edges_df, canonical_mapping
        )

        # Save results
        print("\nSaving normalized data...")
        normalized_nodes.to_csv(
            os.path.join(output_dir, "normalized_nodes.csv"),
            index=False
        )
        normalized_edges.to_csv(
            os.path.join(output_dir, "normalized_edges.csv"),
            index=False
        )

        # Save canonical mapping
        mapping_df = pd.DataFrame([
            {"original": k, "canonical": v}
            for k, v in canonical_mapping.items()
            if k != v  # Only save non-trivial mappings
        ])
        mapping_df.to_csv(
            os.path.join(output_dir, "canonical_mapping.csv"),
            index=False
        )

        # Statistics
        stats = {
            "original_entities": len(entities),
            "normalized_entities": len(set(canonical_mapping.values())),
            "merged_entities": len(entities) - len(set(canonical_mapping.values())),
            "clusters": len(clusters),
            "similar_pairs": len(similar_pairs)
        }

        print(f"\n=== Normalization Complete ===")
        print(f"Original entities: {stats['original_entities']}")
        print(f"After normalization: {stats['normalized_entities']}")
        print(f"Merged: {stats['merged_entities']} ({100*stats['merged_entities']/stats['original_entities']:.1f}%)")

        return stats

    def _find_similar_entities(self, entities: List[str]) -> List[Tuple[str, str, float]]:
        """
        Find pairs of similar entities

        Returns list of (entity1, entity2, similarity_score)
        """
        similar_pairs = []

        # Precompute embeddings if available
        embeddings = None
        if self.embedding_model:
            print("  Computing embeddings...")
            embeddings = self.embedding_model.encode(entities, show_progress_bar=True)

        print("  Comparing entity pairs...")
        n = len(entities)

        # Use batched comparison for efficiency
        for i in tqdm(range(n), desc="  Finding similar pairs"):
            for j in range(i + 1, n):
                e1, e2 = entities[i], entities[j]

                # Quick filter: length difference
                if abs(len(e1) - len(e2)) > max(len(e1), len(e2)) * 0.5:
                    continue

                # String similarity (normalized Levenshtein)
                str_sim = self._string_similarity(e1, e2)

                if str_sim >= self.config.string_similarity_threshold:
                    # If embeddings available, also check embedding similarity
                    if embeddings is not None:
                        emb_sim = np.dot(embeddings[i], embeddings[j])
                        if emb_sim >= self.config.embedding_similarity_threshold:
                            similar_pairs.append((e1, e2, (str_sim + emb_sim) / 2))
                    else:
                        similar_pairs.append((e1, e2, str_sim))

        return similar_pairs

    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Compute normalized string similarity
        Uses a combination of:
        - Exact match after normalization
        - Jaccard similarity on tokens
        - Longest common subsequence ratio
        """
        # Normalize strings
        s1_norm = s1.lower().strip()
        s2_norm = s2.lower().strip()

        # Exact match after normalization
        if s1_norm == s2_norm:
            return 1.0

        # Jaccard similarity on tokens
        tokens1 = set(s1_norm.split())
        tokens2 = set(s2_norm.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union

        # LCS ratio (simplified)
        lcs_len = self._lcs_length(s1_norm, s2_norm)
        lcs_ratio = 2 * lcs_len / (len(s1_norm) + len(s2_norm))

        # Combine metrics
        return 0.5 * jaccard + 0.5 * lcs_ratio

    def _lcs_length(self, s1: str, s2: str) -> int:
        """Compute length of longest common subsequence"""
        m, n = len(s1), len(s2)

        # Optimization for long strings
        if m > 100 or n > 100:
            # Use approximate method for long strings
            common = set(s1) & set(s2)
            return sum(min(s1.count(c), s2.count(c)) for c in common)

        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _build_clusters(
        self,
        similar_pairs: List[Tuple[str, str, float]],
        all_entities: List[str]
    ) -> List[Set[str]]:
        """
        Build clusters of similar entities using Union-Find
        """
        # Union-Find structure
        parent = {e: e for e in all_entities}
        rank = {e: 0 for e in all_entities}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1

        # Union similar entities
        for e1, e2, _ in similar_pairs:
            union(e1, e2)

        # Build clusters
        cluster_map = defaultdict(set)
        for e in all_entities:
            root = find(e)
            cluster_map[root].add(e)

        # Filter clusters with more than one entity
        clusters = [c for c in cluster_map.values() if len(c) > 1]

        return clusters

    def _select_canonical_names(
        self,
        clusters: List[Set[str]],
        edges_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Select canonical name for each cluster

        Selection criteria:
        1. Frequency in edges (most connected)
        2. Length (prefer shorter, more concise names)
        3. Capitalization (prefer properly capitalized)
        """
        # Count entity frequency in edges
        entity_freq = defaultdict(int)
        for _, row in edges_df.iterrows():
            entity_freq[row[':START_ID']] += 1
            entity_freq[row[':END_ID']] += 1

        canonical_mapping = {}

        for cluster in clusters:
            # Score each entity in cluster
            scores = []
            for entity in cluster:
                freq_score = entity_freq.get(entity, 0)
                length_score = -len(entity)  # Prefer shorter
                cap_score = 1 if entity[0].isupper() else 0  # Prefer capitalized

                total_score = freq_score * 10 + length_score * 0.1 + cap_score
                scores.append((entity, total_score))

            # Select canonical name (highest score)
            canonical = max(scores, key=lambda x: x[1])[0]

            # Map all entities in cluster to canonical
            for entity in cluster:
                canonical_mapping[entity] = canonical

        # Add identity mapping for non-clustered entities
        all_entities = set(edges_df[':START_ID']) | set(edges_df[':END_ID'])
        for entity in all_entities:
            if entity not in canonical_mapping:
                canonical_mapping[entity] = entity

        return canonical_mapping

    def _apply_normalization(
        self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        canonical_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply canonical mapping to nodes and edges
        """
        # Update nodes
        normalized_nodes = nodes_df.copy()
        normalized_nodes['name:ID'] = normalized_nodes['name:ID'].map(
            lambda x: canonical_mapping.get(x, x)
        )
        # Remove duplicates (merged entities)
        normalized_nodes = normalized_nodes.drop_duplicates(subset=['name:ID'])

        # Update edges
        normalized_edges = edges_df.copy()
        normalized_edges[':START_ID'] = normalized_edges[':START_ID'].map(
            lambda x: canonical_mapping.get(x, x)
        )
        normalized_edges[':END_ID'] = normalized_edges[':END_ID'].map(
            lambda x: canonical_mapping.get(x, x)
        )
        # Remove self-loops created by normalization
        normalized_edges = normalized_edges[
            normalized_edges[':START_ID'] != normalized_edges[':END_ID']
        ]
        # Remove duplicate edges
        normalized_edges = normalized_edges.drop_duplicates(
            subset=[':START_ID', ':END_ID', 'relation']
        )

        return normalized_nodes, normalized_edges


# ============================================================================
# Part 2: Multi-hop Relation Discovery (多跳关系发现)
# ============================================================================

# Predefined relation composition rules
# 预定义的关系组合规则
RELATION_COMPOSITION_RULES = {
    # Causal chains
    ("increases_risk", "increases_risk"): "indirectly_increases_risk",
    ("causes", "causes"): "indirectly_causes",
    ("leads_to", "leads_to"): "eventually_leads_to",

    # Treatment chains
    ("treats", "prevents"): "treats_to_prevent",
    ("alleviates", "causes"): "alleviates_cause_of",
    ("mitigates", "increases_risk"): "mitigates_risk_factor",

    # Hierarchical
    ("part_of", "part_of"): "indirectly_part_of",
    ("is_a", "is_a"): "is_a",
    ("belongs_to", "belongs_to"): "indirectly_belongs_to",

    # Medical specific
    ("TRANSPLANT_RESULTS_IN_OUTCOME", "increases_risk"): "transplant_risk_factor",
    ("ORGAN_PRESERVED_WITH", "improves"): "preservation_improves",
    ("TREATMENT_USED_FOR", "treats"): "treatment_treats",

    # Generic
    ("related_to", "related_to"): "possibly_related",
    ("associated_with", "associated_with"): "indirectly_associated",
}


@dataclass
class MultiHopPath:
    """Represents a multi-hop path in the graph"""
    source: str
    target: str
    path: List[str]  # [entity1, entity2, entity3, ...]
    relations: List[str]  # [rel1, rel2, ...]
    depth: int
    confidence: float = 1.0


@dataclass
class InferredRelation:
    """Represents an inferred relation from multi-hop analysis"""
    source: str
    target: str
    inferred_relation: str
    confidence: float
    evidence_path: List[str]
    evidence_relations: List[str]


class MultiHopRelationDiscoverer:
    """
    Discover potential multi-hop relations in triple data

    Methods:
    1. Rule-based composition: Apply predefined rules to compose relations
    2. Path analysis: Find common path patterns and suggest relations
    """

    def __init__(
        self,
        max_depth: int = 3,
        min_confidence: float = 0.5,
        min_path_frequency: int = 2
    ):
        self.max_depth = max_depth
        self.min_confidence = min_confidence
        self.min_path_frequency = min_path_frequency

        self.graph = None  # NetworkX graph
        self.edges_df = None

    def discover_from_csv(
        self,
        edges_csv: str,
        output_dir: str
    ) -> Dict[str, any]:
        """
        Main entry point: discover multi-hop relations from edge CSV

        Args:
            edges_csv: Path to edges CSV
            output_dir: Directory to save discovered relations

        Returns:
            Statistics dict
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading edges from {edges_csv}...")
        self.edges_df = pd.read_csv(edges_csv)

        # Build graph
        print("Building graph...")
        self._build_graph()

        # Step 1: Analyze path patterns
        print("\nStep 1: Analyzing path patterns...")
        patterns = self._analyze_path_patterns()

        # Step 2: Apply composition rules
        print("\nStep 2: Applying relation composition rules...")
        inferred_relations = self._apply_composition_rules()

        # Step 3: Discover new relations from patterns
        print("\nStep 3: Discovering relations from patterns...")
        pattern_relations = self._discover_from_patterns(patterns)

        # Combine all inferred relations
        all_inferred = inferred_relations + pattern_relations

        # Filter by confidence
        high_confidence = [r for r in all_inferred if r.confidence >= self.min_confidence]

        # Save results
        print("\nSaving discovered relations...")

        # Save as CSV
        inferred_df = pd.DataFrame([
            {
                ":START_ID": r.source,
                ":END_ID": r.target,
                "relation": r.inferred_relation,
                "confidence": r.confidence,
                "evidence_path": " -> ".join(r.evidence_path),
                "evidence_relations": " -> ".join(r.evidence_relations),
                ":TYPE": "InferredRelation"
            }
            for r in high_confidence
        ])
        inferred_df.to_csv(
            os.path.join(output_dir, "inferred_relations.csv"),
            index=False
        )

        # Save path patterns
        pattern_df = pd.DataFrame([
            {"pattern": " -> ".join(p), "count": c}
            for p, c in patterns.items()
        ])
        pattern_df = pattern_df.sort_values("count", ascending=False)
        pattern_df.to_csv(
            os.path.join(output_dir, "path_patterns.csv"),
            index=False
        )

        # Statistics
        stats = {
            "total_edges": len(self.edges_df),
            "unique_entities": len(set(self.edges_df[':START_ID']) | set(self.edges_df[':END_ID'])),
            "path_patterns_found": len(patterns),
            "inferred_relations": len(all_inferred),
            "high_confidence_relations": len(high_confidence)
        }

        print(f"\n=== Multi-hop Discovery Complete ===")
        print(f"Path patterns found: {stats['path_patterns_found']}")
        print(f"Total inferred relations: {stats['inferred_relations']}")
        print(f"High confidence (>={self.min_confidence}): {stats['high_confidence_relations']}")

        return stats

    def _build_graph(self):
        """Build NetworkX graph from edges"""
        if not HAS_NETWORKX:
            print("Warning: NetworkX not installed. Using simple graph representation.")
            self.graph = None
            return

        self.graph = nx.DiGraph()

        for _, row in self.edges_df.iterrows():
            self.graph.add_edge(
                row[':START_ID'],
                row[':END_ID'],
                relation=row['relation']
            )

    def _analyze_path_patterns(self) -> Dict[Tuple[str, ...], int]:
        """
        Analyze 2-hop and 3-hop path patterns in the graph

        Returns dict of (relation1, relation2, ...) -> count
        """
        patterns = defaultdict(int)

        if self.graph is None:
            # Fallback without NetworkX
            return self._analyze_patterns_simple()

        # Find 2-hop patterns
        print("  Analyzing 2-hop patterns...")
        for node in tqdm(self.graph.nodes(), desc="  2-hop"):
            # Get outgoing edges
            out_edges = list(self.graph.out_edges(node, data=True))
            for _, mid, data1 in out_edges:
                # Get edges from intermediate node
                mid_out = list(self.graph.out_edges(mid, data=True))
                for _, end, data2 in mid_out:
                    if end != node:  # Avoid trivial cycles
                        pattern = (data1['relation'], data2['relation'])
                        patterns[pattern] += 1

        # Find 3-hop patterns (if not too many nodes)
        if len(self.graph.nodes()) < 5000:
            print("  Analyzing 3-hop patterns...")
            for node in tqdm(list(self.graph.nodes())[:1000], desc="  3-hop"):
                for _, n1, d1 in self.graph.out_edges(node, data=True):
                    for _, n2, d2 in self.graph.out_edges(n1, data=True):
                        if n2 == node:
                            continue
                        for _, n3, d3 in self.graph.out_edges(n2, data=True):
                            if n3 not in (node, n1):
                                pattern = (d1['relation'], d2['relation'], d3['relation'])
                                patterns[pattern] += 1

        return patterns

    def _analyze_patterns_simple(self) -> Dict[Tuple[str, ...], int]:
        """Fallback pattern analysis without NetworkX"""
        patterns = defaultdict(int)

        # Build adjacency list
        adj = defaultdict(list)
        for _, row in self.edges_df.iterrows():
            adj[row[':START_ID']].append((row[':END_ID'], row['relation']))

        # Find 2-hop patterns
        for start, neighbors in tqdm(adj.items(), desc="  Analyzing patterns"):
            for mid, rel1 in neighbors:
                if mid in adj:
                    for end, rel2 in adj[mid]:
                        if end != start:
                            patterns[(rel1, rel2)] += 1

        return patterns

    def _apply_composition_rules(self) -> List[InferredRelation]:
        """
        Apply predefined composition rules to infer new relations
        """
        inferred = []

        if self.graph is None:
            return self._apply_rules_simple()

        # For each 2-hop path, check if rule applies
        for node in tqdm(self.graph.nodes(), desc="  Applying rules"):
            for _, mid, data1 in self.graph.out_edges(node, data=True):
                for _, end, data2 in self.graph.out_edges(mid, data=True):
                    if end == node:
                        continue

                    # Check if direct edge already exists
                    if self.graph.has_edge(node, end):
                        continue

                    rel1, rel2 = data1['relation'], data2['relation']
                    key = (rel1, rel2)

                    if key in RELATION_COMPOSITION_RULES:
                        inferred_rel = RELATION_COMPOSITION_RULES[key]
                        confidence = 0.8  # Rule-based has high confidence

                        inferred.append(InferredRelation(
                            source=node,
                            target=end,
                            inferred_relation=inferred_rel,
                            confidence=confidence,
                            evidence_path=[node, mid, end],
                            evidence_relations=[rel1, rel2]
                        ))
                    else:
                        # Try generic rules
                        generic_rel = self._apply_generic_rule(rel1, rel2)
                        if generic_rel:
                            inferred.append(InferredRelation(
                                source=node,
                                target=end,
                                inferred_relation=generic_rel,
                                confidence=0.6,  # Lower confidence for generic
                                evidence_path=[node, mid, end],
                                evidence_relations=[rel1, rel2]
                            ))

        return inferred

    def _apply_rules_simple(self) -> List[InferredRelation]:
        """Fallback rule application without NetworkX"""
        inferred = []

        # Build adjacency and existing edge set
        adj = defaultdict(list)
        existing_edges = set()

        for _, row in self.edges_df.iterrows():
            adj[row[':START_ID']].append((row[':END_ID'], row['relation']))
            existing_edges.add((row[':START_ID'], row[':END_ID']))

        for start, neighbors in tqdm(adj.items(), desc="  Applying rules"):
            for mid, rel1 in neighbors:
                if mid in adj:
                    for end, rel2 in adj[mid]:
                        if end == start or (start, end) in existing_edges:
                            continue

                        key = (rel1, rel2)
                        if key in RELATION_COMPOSITION_RULES:
                            inferred.append(InferredRelation(
                                source=start,
                                target=end,
                                inferred_relation=RELATION_COMPOSITION_RULES[key],
                                confidence=0.8,
                                evidence_path=[start, mid, end],
                                evidence_relations=[rel1, rel2]
                            ))

        return inferred

    def _apply_generic_rule(self, rel1: str, rel2: str) -> Optional[str]:
        """
        Apply generic composition rules
        """
        # Same relation -> indirect version
        if rel1 == rel2:
            return f"indirectly_{rel1}"

        # Contains common keywords
        rel1_lower = rel1.lower()
        rel2_lower = rel2.lower()

        if "risk" in rel1_lower or "risk" in rel2_lower:
            return "related_risk_factor"

        if "treat" in rel1_lower or "treat" in rel2_lower:
            return "related_treatment"

        if "cause" in rel1_lower or "cause" in rel2_lower:
            return "related_cause"

        return None

    def _discover_from_patterns(
        self,
        patterns: Dict[Tuple[str, ...], int]
    ) -> List[InferredRelation]:
        """
        Discover new relations based on frequent path patterns
        """
        inferred = []

        # Filter frequent patterns
        frequent = {p: c for p, c in patterns.items()
                   if c >= self.min_path_frequency and len(p) == 2}

        if not frequent:
            return inferred

        print(f"  Found {len(frequent)} frequent 2-hop patterns")

        # Suggest new rules for patterns not in predefined rules
        new_rules = {}
        for pattern, count in sorted(frequent.items(), key=lambda x: -x[1])[:20]:
            if pattern not in RELATION_COMPOSITION_RULES:
                # Suggest composed relation name
                suggested = f"{pattern[0]}_then_{pattern[1]}"
                new_rules[pattern] = (suggested, count)

        if new_rules:
            print(f"\n  Suggested new composition rules:")
            for pattern, (suggested, count) in list(new_rules.items())[:10]:
                print(f"    {pattern} -> {suggested} (count: {count})")

        return inferred


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Triple Normalization and Multi-hop Relation Discovery"
    )
    parser.add_argument(
        "--nodes",
        type=str,
        required=True,
        help="Path to nodes CSV file"
    )
    parser.add_argument(
        "--edges",
        type=str,
        required=True,
        help="Path to edges CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory"
    )
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip entity normalization step"
    )
    parser.add_argument(
        "--skip-multihop",
        action="store_true",
        help="Skip multi-hop relation discovery"
    )
    parser.add_argument(
        "--string-threshold",
        type=float,
        default=0.85,
        help="String similarity threshold for normalization"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for multi-hop discovery"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for inferred relations"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Triple Normalization and Multi-hop Relation Discovery")
    print("=" * 60)

    # Paths
    nodes_csv = args.nodes
    edges_csv = args.edges
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    normalized_edges = edges_csv  # Default to original

    # Step 1: Entity Normalization
    if not args.skip_normalization:
        print("\n" + "=" * 60)
        print("PHASE 1: Entity Normalization")
        print("=" * 60)

        config = NormalizationConfig(
            string_similarity_threshold=args.string_threshold
        )
        normalizer = EntityNormalizer(config)

        norm_output = os.path.join(output_dir, "normalization")
        stats = normalizer.normalize_from_csv(nodes_csv, edges_csv, norm_output)

        # Use normalized edges for multi-hop discovery
        normalized_edges = os.path.join(norm_output, "normalized_edges.csv")

    # Step 2: Multi-hop Relation Discovery
    if not args.skip_multihop:
        print("\n" + "=" * 60)
        print("PHASE 2: Multi-hop Relation Discovery")
        print("=" * 60)

        discoverer = MultiHopRelationDiscoverer(
            max_depth=args.max_depth,
            min_confidence=args.min_confidence
        )

        multihop_output = os.path.join(output_dir, "multihop")
        stats = discoverer.discover_from_csv(normalized_edges, multihop_output)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
