"""
归一化处理入口脚本

完整的实体归一化流程，包含：
1. 相似实体检测（KNN + WCC）
2. 实体消歧（字符串召回 + 向量重排 + NIL检测）
3. 实体对齐（冲突检测 + LLM解决）
4. 实体合并
5. 多跳关系发现

使用方法:
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
    """归一化配置参数"""
    # 相似实体检测
    enable_similar_detection: bool = True
    similarity_threshold: float = 0.8

    # 实体消歧
    enable_disambiguation: bool = True

    # 实体对齐
    enable_alignment: bool = True

    # 实体合并
    enable_merge: bool = True
    merge_batch_size: int = 20
    merge_max_workers: int = 4

    # 多跳关系发现
    enable_multi_hop: bool = True
    multi_hop_max_depth: int = 3
    multi_hop_min_confidence: float = 0.5

    # 性能参数
    batch_size: int = 500
    verbose: bool = True


@dataclass
class NormalizationResult:
    """归一化结果"""
    success: bool = False

    # 各阶段统计
    similar_entities_detected: int = 0
    communities_found: int = 0
    entities_disambiguated: int = 0
    entities_aligned: int = 0
    entities_merged: int = 0
    multi_hop_relations_found: int = 0

    # 性能统计
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)

    # 错误信息
    errors: List[str] = field(default_factory=list)


class EntityNormalizer:
    """
    实体归一化器

    整合所有归一化组件，提供完整的实体归一化流程

    归一化流程：
    1. 相似实体检测 - 使用KNN和WCC算法找出相似实体组
    2. 实体消歧 - 为每个实体设置canonical_id指向规范实体
    3. 实体对齐 - 解决同一canonical下的冲突
    4. 实体合并 - 将重复实体合并为一个
    5. 多跳关系发现 - 发现潜在的间接关系
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        初始化归一化器

        Args:
            config: 归一化配置，为None时使用默认配置
        """
        self.config = config or NormalizationConfig()
        self.console = Console()
        self.graph = connection_manager.get_connection()

        # 初始化结果
        self.result = NormalizationResult()

        # 延迟初始化组件
        self._similar_detector = None
        self._disambiguator = None
        self._aligner = None
        self._merger = None
        self._multi_hop_discoverer = None

    @property
    def similar_detector(self) -> SimilarEntityDetector:
        """相似实体检测器"""
        if self._similar_detector is None:
            gds_config = GDSConfig(
                similarity_threshold=self.config.similarity_threshold
            )
            self._similar_detector = SimilarEntityDetector(gds_config)
        return self._similar_detector

    @property
    def disambiguator(self) -> EntityDisambiguator:
        """实体消歧器"""
        if self._disambiguator is None:
            self._disambiguator = EntityDisambiguator()
        return self._disambiguator

    @property
    def aligner(self) -> EntityAligner:
        """实体对齐器"""
        if self._aligner is None:
            self._aligner = EntityAligner()
        return self._aligner

    @property
    def merger(self) -> EntityMerger:
        """实体合并器"""
        if self._merger is None:
            self._merger = EntityMerger(
                batch_size=self.config.merge_batch_size,
                max_workers=self.config.merge_max_workers
            )
        return self._merger

    @property
    def multi_hop_discoverer(self) -> MultiHopRelationDiscoverer:
        """多跳关系发现器"""
        if self._multi_hop_discoverer is None:
            self._multi_hop_discoverer = MultiHopRelationDiscoverer(
                max_depth=self.config.multi_hop_max_depth,
                min_confidence=self.config.multi_hop_min_confidence
            )
        return self._multi_hop_discoverer

    def _create_progress(self) -> Progress:
        """创建进度显示器"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        )

    def _display_stage_header(self, stage: str):
        """显示阶段标题"""
        if self.config.verbose:
            self.console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
            self.console.print(f"[bold cyan]{stage}[/bold cyan]")
            self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

    def _display_results_table(self, title: str, data: Dict[str, Any]):
        """显示结果表格"""
        if not self.config.verbose:
            return

        table = Table(title=title, show_header=True)
        table.add_column("指标", style="cyan")
        table.add_column("值", justify="right")

        for key, value in data.items():
            table.add_row(str(key), str(value))

        self.console.print(table)

    def normalize(self) -> NormalizationResult:
        """
        执行完整的归一化流程

        Returns:
            NormalizationResult: 归一化结果
        """
        start_time = time.time()

        if self.config.verbose:
            self.console.print(Panel(
                "[bold cyan]开始实体归一化流程[/bold cyan]",
                border_style="cyan"
            ))

        try:
            # 阶段1: 相似实体检测
            if self.config.enable_similar_detection:
                self._run_similar_detection()

            # 阶段2: 实体消歧
            if self.config.enable_disambiguation:
                self._run_disambiguation()

            # 阶段3: 实体对齐
            if self.config.enable_alignment:
                self._run_alignment()

            # 阶段4: 实体合并
            if self.config.enable_merge:
                self._run_merge()

            # 阶段5: 多跳关系发现
            if self.config.enable_multi_hop:
                self._run_multi_hop_discovery()

            self.result.success = True

        except Exception as e:
            self.result.errors.append(str(e))
            self.console.print(f"[red]归一化过程出错: {e}[/red]")

        finally:
            self.result.total_time = time.time() - start_time
            self._display_final_summary()

        return self.result

    def _run_similar_detection(self):
        """运行相似实体检测"""
        self._display_stage_header("阶段1: 相似实体检测")
        stage_start = time.time()

        try:
            duplicates, stats = self.similar_detector.process_entities()

            self.result.similar_entities_detected = len(duplicates)
            self.result.communities_found = stats.get('社区数量', 0)

            self._display_results_table("相似实体检测结果", {
                "候选实体组": len(duplicates),
                "社区数量": self.result.communities_found,
                "关系数量": stats.get('关系数量', 0)
            })

        except Exception as e:
            self.result.errors.append(f"相似实体检测失败: {e}")
            self.console.print(f"[yellow]警告: 相似实体检测失败 - {e}[/yellow]")

        finally:
            self.result.stage_times['相似实体检测'] = time.time() - stage_start

    def _run_disambiguation(self):
        """运行实体消歧"""
        self._display_stage_header("阶段2: 实体消歧")
        stage_start = time.time()

        try:
            disambiguated = self.disambiguator.apply_to_graph()
            self.result.entities_disambiguated = disambiguated

            self._display_results_table("实体消歧结果", {
                "消歧的实体数": disambiguated,
                "耗时": f"{time.time() - stage_start:.2f}秒"
            })

        except Exception as e:
            self.result.errors.append(f"实体消歧失败: {e}")
            self.console.print(f"[yellow]警告: 实体消歧失败 - {e}[/yellow]")

        finally:
            self.result.stage_times['实体消歧'] = time.time() - stage_start

    def _run_alignment(self):
        """运行实体对齐"""
        self._display_stage_header("阶段3: 实体对齐")
        stage_start = time.time()

        try:
            align_result = self.aligner.align_all()
            self.result.entities_aligned = align_result.get('entities_aligned', 0)

            self._display_results_table("实体对齐结果", {
                "处理的分组数": align_result.get('groups_processed', 0),
                "检测到的冲突": align_result.get('conflicts_detected', 0),
                "对齐的实体数": self.result.entities_aligned,
                "耗时": f"{align_result.get('elapsed_time', 0):.2f}秒"
            })

        except Exception as e:
            self.result.errors.append(f"实体对齐失败: {e}")
            self.console.print(f"[yellow]警告: 实体对齐失败 - {e}[/yellow]")

        finally:
            self.result.stage_times['实体对齐'] = time.time() - stage_start

    def _run_merge(self):
        """运行实体合并"""
        self._display_stage_header("阶段4: 实体合并")
        stage_start = time.time()

        try:
            # 获取相似实体检测的结果
            duplicates = self.similar_detector.find_potential_duplicates()

            if duplicates:
                merged_count, stats = self.merger.process_duplicates(duplicates)
                self.result.entities_merged = merged_count

                self._display_results_table("实体合并结果", {
                    "合并的实体数": merged_count,
                    "耗时": f"{time.time() - stage_start:.2f}秒"
                })
            else:
                self.console.print("[blue]没有需要合并的实体[/blue]")

        except Exception as e:
            self.result.errors.append(f"实体合并失败: {e}")
            self.console.print(f"[yellow]警告: 实体合并失败 - {e}[/yellow]")

        finally:
            self.result.stage_times['实体合并'] = time.time() - stage_start

    def _run_multi_hop_discovery(self):
        """运行多跳关系发现"""
        self._display_stage_header("阶段5: 多跳关系发现")
        stage_start = time.time()

        try:
            result = self.multi_hop_discoverer.discover_all()
            self.result.multi_hop_relations_found = result.get('relations_found', 0)

            self._display_results_table("多跳关系发现结果", {
                "发现的潜在关系": result.get('relations_found', 0),
                "已写入的关系": result.get('relations_written', 0),
                "分析的路径数": result.get('paths_analyzed', 0),
                "耗时": f"{time.time() - stage_start:.2f}秒"
            })

        except Exception as e:
            self.result.errors.append(f"多跳关系发现失败: {e}")
            self.console.print(f"[yellow]警告: 多跳关系发现失败 - {e}[/yellow]")

        finally:
            self.result.stage_times['多跳关系发现'] = time.time() - stage_start

    def _display_final_summary(self):
        """显示最终摘要"""
        if not self.config.verbose:
            return

        # 状态面板
        status = "[bold green]成功[/bold green]" if self.result.success else "[bold red]失败[/bold red]"
        self.console.print(Panel(
            f"归一化流程完成 - 状态: {status}",
            border_style="green" if self.result.success else "red"
        ))

        # 总体统计表
        summary_table = Table(title="归一化总结", show_header=True)
        summary_table.add_column("阶段", style="cyan")
        summary_table.add_column("处理数量", justify="right")
        summary_table.add_column("耗时(秒)", justify="right")

        stages = [
            ("相似实体检测", self.result.similar_entities_detected),
            ("实体消歧", self.result.entities_disambiguated),
            ("实体对齐", self.result.entities_aligned),
            ("实体合并", self.result.entities_merged),
            ("多跳关系发现", self.result.multi_hop_relations_found)
        ]

        for stage_name, count in stages:
            time_taken = self.result.stage_times.get(stage_name, 0)
            summary_table.add_row(stage_name, str(count), f"{time_taken:.2f}")

        summary_table.add_row(
            "[bold]总计[/bold]",
            "",
            f"[bold]{self.result.total_time:.2f}[/bold]",
            style="bold"
        )

        self.console.print(summary_table)

        # 显示错误（如果有）
        if self.result.errors:
            self.console.print("\n[yellow]警告/错误:[/yellow]")
            for error in self.result.errors:
                self.console.print(f"  - {error}")

    def normalize_quick(self) -> NormalizationResult:
        """
        快速归一化（跳过耗时的多跳关系发现）

        Returns:
            NormalizationResult: 归一化结果
        """
        original_multi_hop = self.config.enable_multi_hop
        self.config.enable_multi_hop = False

        try:
            return self.normalize()
        finally:
            self.config.enable_multi_hop = original_multi_hop

    def get_normalization_stats(self) -> Dict[str, Any]:
        """
        获取当前图谱的归一化统计信息

        Returns:
            Dict: 统计信息
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
    执行实体归一化的便捷函数

    Args:
        config: 归一化配置

    Returns:
        NormalizationResult: 归一化结果
    """
    normalizer = EntityNormalizer(config)
    return normalizer.normalize()


if __name__ == "__main__":
    # 命令行入口
    import argparse

    parser = argparse.ArgumentParser(description="实体归一化处理")
    parser.add_argument("--quick", action="store_true", help="快速模式（跳过多跳关系发现）")
    parser.add_argument("--no-merge", action="store_true", help="跳过实体合并")
    parser.add_argument("--verbose", action="store_true", default=True, help="详细输出")
    parser.add_argument("--multi-hop-depth", type=int, default=3, help="多跳关系最大深度")

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

    # 返回适当的退出代码
    exit(0 if result.success else 1)
