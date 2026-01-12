"""
Entity Normalization Entry Script

Complete entity normalization workflow including:
1. Similar entity detection (KNN + WCC)
2. Entity disambiguation (string recall + vector rerank + NIL detection)
3. Entity alignment (conflict detection + LLM resolution)
4. Entity merging
5. Multi-hop relation discovery

Usage:
    from graphrag_agent.graph.processing.normalize import EntityNormalizer
    normalizer = EntityNormalizer()
    result = normalizer.normalize()
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from graphrag_agent.graph.core import connection_manager
from graphrag_agent.graph.processing.similar_entity import SimilarEntityDetector, GDSConfig
from graphrag_agent.graph.processing.entity_disambiguation import EntityDisambiguator
from graphrag_agent.graph.processing.entity_alignment import EntityAligner
from graphrag_agent.graph.processing.entity_merger import EntityMerger
from graphrag_agent.graph.processing.multi_hop_relation import MultiHopRelationDiscoverer


@dataclass
class NormalizationConfig:
    """Normalization configuration parameters"""
    # Similar entity detection
    enable_similar_detection: bool = True
    similarity_threshold: float = 0.8

    # Entity disambiguation
    enable_disambiguation: bool = True

    # Entity alignment
    enable_alignment: bool = True

    # Entity merging
    enable_merge: bool = True
    merge_batch_size: int = 20
    merge_max_workers: int = 4

    # Multi-hop relation discovery
    enable_multi_hop: bool = True
    multi_hop_max_depth: int = 3
    multi_hop_min_confidence: float = 0.5

    # Performance parameters
    batch_size: int = 500
    verbose: bool = True


@dataclass
class NormalizationResult:
    """Normalization result"""
    success: bool = False

    # Stage statistics
    similar_entities_detected: int = 0
    communities_found: int = 0
    entities_disambiguated: int = 0
    entities_aligned: int = 0
    entities_merged: int = 0
    multi_hop_relations_found: int = 0

    # Performance statistics
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)

    # Error messages
    errors: List[str] = field(default_factory=list)


class EntityNormalizer:
    """
    Entity Normalizer

    Integrates all normalization components to provide a complete entity normalization workflow

    Normalization workflow:
    1. Similar entity detection - Use KNN and WCC algorithms to find similar entity groups
    2. Entity disambiguation - Set canonical_id for each entity pointing to canonical entity
    3. Entity alignment - Resolve conflicts under the same canonical
    4. Entity merging - Merge duplicate entities into one
    5. Multi-hop relation discovery - Discover potential indirect relationships
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize normalizer

        Args:
            config: Normalization config, uses default if None
        """
        self.config = config or NormalizationConfig()
        self.console = Console()
        self.graph = connection_manager.get_connection()

        # Initialize result
        self.result = NormalizationResult()

        # Lazy initialization of components
        self._similar_detector = None
        self._disambiguator = None
        self._aligner = None
        self._merger = None
        self._multi_hop_discoverer = None

    @property
    def similar_detector(self) -> SimilarEntityDetector:
        """Similar entity detector"""
        if self._similar_detector is None:
            gds_config = GDSConfig(
                similarity_threshold=self.config.similarity_threshold
            )
            self._similar_detector = SimilarEntityDetector(gds_config)
        return self._similar_detector

    @property
    def disambiguator(self) -> EntityDisambiguator:
        """Entity disambiguator"""
        if self._disambiguator is None:
            self._disambiguator = EntityDisambiguator()
        return self._disambiguator

    @property
    def aligner(self) -> EntityAligner:
        """Entity aligner"""
        if self._aligner is None:
            self._aligner = EntityAligner()
        return self._aligner

    @property
    def merger(self) -> EntityMerger:
        """Entity merger"""
        if self._merger is None:
            self._merger = EntityMerger(
                batch_size=self.config.merge_batch_size,
                max_workers=self.config.merge_max_workers
            )
        return self._merger

    @property
    def multi_hop_discoverer(self) -> MultiHopRelationDiscoverer:
        """Multi-hop relation discoverer"""
        if self._multi_hop_discoverer is None:
            self._multi_hop_discoverer = MultiHopRelationDiscoverer(
                max_depth=self.config.multi_hop_max_depth,
                min_confidence=self.config.multi_hop_min_confidence
            )
        return self._multi_hop_discoverer

    def _create_progress(self) -> Progress:
        """Create progress display"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _display_stage_header(self, stage: str):
        """Display stage header"""
        if self.config.verbose:
            self.console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            self.console.print(f"[bold cyan]{stage}[/bold cyan]")
            self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """Display results table"""
        if not self.config.verbose:
            return

        table = Table(title=title, show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def normalize(self) -> NormalizationResult:
        """
        Execute complete normalization workflow

        Returns:
            NormalizationResult: Normalization result
        """
        start_time = time.time()

        if self.config.verbose:
            self.console.print(Panel(
                "[bold cyan]Starting Entity Normalization Workflow[/bold cyan]",
                border_style="cyan"
            ))

        try:
            # Stage 1: Similar entity detection
            if self.config.enable_similar_detection:
                self._run_similar_detection()

            # Stage 2: Entity disambiguation
            if self.config.enable_disambiguation:
                self._run_disambiguation()

            # Stage 3: Entity alignment
            if self.config.enable_alignment:
                self._run_alignment()

            # Stage 4: Entity merging
            if self.config.enable_merge:
                self._run_merge()

            # Stage 5: Multi-hop relation discovery
            if self.config.enable_multi_hop:
                self._run_multi_hop_discovery()

            self.result.success = True

        except Exception as e:
            self.result.errors.append(str(e))
            self.console.print(f"[red]Normalization error: {e}[/red]")

        finally:
            self.result.total_time = time.time() - start_time
            self._display_final_summary()

        return self.result

    def _run_similar_detection(self):
        """Run similar entity detection"""
        self._display_stage_header("Stage 1: Similar Entity Detection")
        stage_start = time.time()

        try:
            duplicates, stats = self.similar_detector.process_entities()

            self.result.similar_entities_detected = len(duplicates)
            self.result.communities_found = stats.get('communityCount', stats.get('community_count', 0))

            self._display_results_table("Similar Entity Detection Results", {
                "Candidate Entity Groups": len(duplicates),
                "Communities Found": self.result.communities_found,
                "Relationships": stats.get('relationshipsWritten', stats.get('relationships', 0))
            })

        except Exception as e:
            self.result.errors.append(f"Similar entity detection failed: {e}")
            self.console.print(f"[yellow]Warning: Similar entity detection failed - {e}[/yellow]")

        finally:
            self.result.stage_times['similar_detection'] = time.time() - stage_start

    def _run_disambiguation(self):
        """Run entity disambiguation"""
        self._display_stage_header("Stage 2: Entity Disambiguation")
        stage_start = time.time()

        try:
            disambiguated = self.disambiguator.apply_to_graph()
            self.result.entities_disambiguated = disambiguated

            self._display_results_table("Entity Disambiguation Results", {
                "Entities Disambiguated": disambiguated,
                "Time Elapsed": f"{time.time() - stage_start:.2f}s"
            })

        except Exception as e:
            self.result.errors.append(f"Entity disambiguation failed: {e}")
            self.console.print(f"[yellow]Warning: Entity disambiguation failed - {e}[/yellow]")

        finally:
            self.result.stage_times['disambiguation'] = time.time() - stage_start

    def _run_alignment(self):
        """Run entity alignment"""
        self._display_stage_header("Stage 3: Entity Alignment")
        stage_start = time.time()

        try:
            align_result = self.aligner.align_all()
            self.result.entities_aligned = align_result.get('entities_aligned', 0)

            self._display_results_table("Entity Alignment Results", {
                "Groups Processed": align_result.get('groups_processed', 0),
                "Conflicts Detected": align_result.get('conflicts_detected', 0),
                "Entities Aligned": self.result.entities_aligned,
                "Time Elapsed": f"{align_result.get('elapsed_time', 0):.2f}s"
            })

        except Exception as e:
            self.result.errors.append(f"Entity alignment failed: {e}")
            self.console.print(f"[yellow]Warning: Entity alignment failed - {e}[/yellow]")

        finally:
            self.result.stage_times['alignment'] = time.time() - stage_start

    def _run_merge(self):
        """Run entity merging"""
        self._display_stage_header("Stage 4: Entity Merging")
        stage_start = time.time()

        try:
            # Get similar entity detection results
            duplicates = self.similar_detector.find_potential_duplicates()

            if duplicates:
                merged_count, stats = self.merger.process_duplicates(duplicates)
                self.result.entities_merged = merged_count

                self._display_results_table("Entity Merge Results", {
                    "Entities Merged": merged_count,
                    "Time Elapsed": f"{time.time() - stage_start:.2f}s"
                })
            else:
                self.console.print("[blue]No entities to merge[/blue]")

        except Exception as e:
            self.result.errors.append(f"Entity merging failed: {e}")
            self.console.print(f"[yellow]Warning: Entity merging failed - {e}[/yellow]")

        finally:
            self.result.stage_times['merge'] = time.time() - stage_start

    def _run_multi_hop_discovery(self):
        """Run multi-hop relation discovery"""
        self._display_stage_header("Stage 5: Multi-hop Relation Discovery")
        stage_start = time.time()

        try:
            result = self.multi_hop_discoverer.discover_all()
            self.result.multi_hop_relations_found = result.get('relations_found', 0)

            self._display_results_table("Multi-hop Relation Discovery Results", {
                "Potential Relations Found": result.get('relations_found', 0),
                "Relations Written": result.get('relations_written', 0),
                "Paths Analyzed": result.get('paths_analyzed', 0),
                "Time Elapsed": f"{time.time() - stage_start:.2f}s"
            })

        except Exception as e:
            self.result.errors.append(f"Multi-hop relation discovery failed: {e}")
            self.console.print(f"[yellow]Warning: Multi-hop relation discovery failed - {e}[/yellow]")

        finally:
            self.result.stage_times['multi_hop'] = time.time() - stage_start

    def _display_final_summary(self):
        """Display final summary"""
        if not self.config.verbose:
            return

        # Status panel
        status = "[bold green]Success[/bold green]" if self.result.success else "[bold red]Failed[/bold red]"
        self.console.print(Panel(
            f"Normalization Complete - Status: {status}",
            border_style="green" if self.result.success else "red"
        ))

        # Summary table
        summary_table = Table(title="Normalization Summary", show_header=True)
        summary_table.add_column("Stage", style="cyan")
        summary_table.add_column("Count", justify="right")
        summary_table.add_column("Time (s)", justify="right")

        stages = [
            ("Similar Detection", self.result.similar_entities_detected, "similar_detection"),
            ("Disambiguation", self.result.entities_disambiguated, "disambiguation"),
            ("Alignment", self.result.entities_aligned, "alignment"),
            ("Merge", self.result.entities_merged, "merge"),
            ("Multi-hop Discovery", self.result.multi_hop_relations_found, "multi_hop")
        ]

        for stage_name, count, key in stages:
            time_taken = self.result.stage_times.get(key, 0)
            summary_table.add_row(stage_name, str(count), f"{time_taken:.2f}")

        summary_table.add_row(
            "[bold]Total[/bold]",
            "",
            f"[bold]{self.result.total_time:.2f}[/bold]",
            style="bold"
        )

        self.console.print(summary_table)

        # Display errors if any
        if self.result.errors:
            self.console.print("\n[yellow]Warnings/Errors:[/yellow]")
            for error in self.result.errors:
                self.console.print(f"  - {error}")

    def normalize_quick(self) -> NormalizationResult:
        """
        Quick normalization (skip time-consuming multi-hop relation discovery)

        Returns:
            NormalizationResult: Normalization result
        """
        original_multi_hop = self.config.enable_multi_hop
        self.config.enable_multi_hop = False

        try:
            return self.normalize()
        finally:
            self.config.enable_multi_hop = original_multi_hop

    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        Get current graph normalization statistics

        Returns:
            Dict: Statistics
        """
        stats_query = """
        MATCH (e:`__Entity__`)
        WITH count(e) AS total_entities,
             count(CASE WHEN e.canonical_id IS NOT NULL THEN 1 END) AS with_canonical,
             count(CASE WHEN e.disambiguated = true THEN 1 END) AS disambiguated,
             count(CASE WHEN e.aligned_from IS NOT NULL THEN 1 END) AS aligned
        RETURN total_entities, with_canonical, disambiguated, aligned
        """

        result = self.graph.query(stats_query)

        if result:
            return {
                'total_entities': result[0]['total_entities'],
                'with_canonical_id': result[0]['with_canonical'],
                'disambiguated': result[0]['disambiguated'],
                'aligned': result[0]['aligned']
            }

        return {}


def normalize(config: Optional[NormalizationConfig] = None) -> NormalizationResult:
    """
    Convenience function for entity normalization

    Args:
        config: Normalization config

    Returns:
        NormalizationResult: Normalization result
    """
    normalizer = EntityNormalizer(config)
    return normalizer.normalize()


if __name__ == "__main__":
    # Command line entry
    import argparse

    parser = argparse.ArgumentParser(description="Entity Normalization Processing")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip multi-hop discovery)")
    parser.add_argument("--no-merge", action="store_true", help="Skip entity merging")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--multi-hop-depth", type=int, default=3, help="Max depth for multi-hop relations")

    args = parser.parse_args()

    config = NormalizationConfig(
        enable_merge=not args.no_merge,
        enable_multi_hop=not args.quick,
        multi_hop_max_depth=args.multi_hop_depth,
        verbose=args.verbose
    )

    normalizer = EntityNormalizer(config)

    if args.quick:
        result = normalizer.normalize_quick()
    else:
        result = normalizer.normalize()

    # Return appropriate exit code
    exit(0 if result.success else 1)
