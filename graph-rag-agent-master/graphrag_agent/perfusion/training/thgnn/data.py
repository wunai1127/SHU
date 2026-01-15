"""
Perfusion THGNN Data Processing - 数据处理与图构建

为 THGNN 准备数据：
1. 加载 EVHP 数据
2. 构建病例间的关系图（正/负相关）
3. 提取时序特征
4. 数据增强（可选）
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler


@dataclass
class GraphData:
    """图数据结构"""
    features: torch.Tensor      # [N, seq_len, n_features] 时序特征
    pos_adj: torch.Tensor       # [N, N] 正关系邻接矩阵
    neg_adj: torch.Tensor       # [N, N] 负关系邻接矩阵
    labels: torch.Tensor        # [N] 标签（质量分数）
    case_ids: List[str]         # 病例ID列表
    mask: torch.Tensor          # [N] 有效标记


class PerfusionGraphBuilder:
    """
    灌注数据图构建器

    构建病例间的关系图：
    1. 基于时序相关性（Pearson）
    2. 基于初始状态相似性（可选）
    3. KNN 图（可选）
    """

    def __init__(
        self,
        pos_threshold: float = 0.3,
        neg_threshold: float = -0.3,
        method: str = 'correlation',  # 'correlation', 'knn', 'full'
        knn_k: int = 5,
    ):
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.method = method
        self.knn_k = knn_k

    def build_correlation_graph(
        self,
        features: np.ndarray  # [N, seq_len, n_features]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于时序相关性构建图

        计算病例间的 Pearson 相关系数，划分正/负关系

        Args:
            features: [N, seq_len, n_features]

        Returns:
            pos_adj: [N, N] 正关系
            neg_adj: [N, N] 负关系
        """
        N = features.shape[0]

        # 将时序展平为向量
        flat_features = features.reshape(N, -1)  # [N, seq_len * n_features]

        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(flat_features)

        # 处理 NaN（相同特征会导致 NaN）
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # 划分正/负关系
        pos_adj = (corr_matrix > self.pos_threshold).astype(np.float32)
        neg_adj = (corr_matrix < self.neg_threshold).astype(np.float32)

        # 移除自环
        np.fill_diagonal(pos_adj, 0)
        np.fill_diagonal(neg_adj, 0)

        return pos_adj, neg_adj

    def build_knn_graph(
        self,
        features: np.ndarray,
        k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于 KNN 构建图

        Args:
            features: [N, seq_len, n_features]
            k: 近邻数量

        Returns:
            pos_adj: [N, N] 正关系（最近邻）
            neg_adj: [N, N] 负关系（最远邻）
        """
        from sklearn.neighbors import NearestNeighbors

        k = k or self.knn_k
        N = features.shape[0]
        k = min(k, N - 1)

        if k < 1:
            # 样本太少，返回全连接图
            return self.build_full_graph(N)

        flat_features = features.reshape(N, -1)

        # 最近邻
        nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn.fit(flat_features)
        distances, indices = nn.kneighbors(flat_features)

        # 构建邻接矩阵
        pos_adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in indices[i][1:]:  # 排除自身
                pos_adj[i, j] = 1.0
                pos_adj[j, i] = 1.0  # 对称

        # 负关系：最远的 k 个邻居
        neg_adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            dists = np.linalg.norm(flat_features - flat_features[i], axis=1)
            farthest = np.argsort(dists)[-k-1:-1]  # 最远的 k 个
            for j in farthest:
                if j != i:
                    neg_adj[i, j] = 1.0
                    neg_adj[j, i] = 1.0

        return pos_adj, neg_adj

    def build_full_graph(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建全连接图（用于样本极少的情况）

        Args:
            n: 节点数量

        Returns:
            pos_adj: [N, N] 全 1（除对角线）
            neg_adj: [N, N] 全 0
        """
        pos_adj = np.ones((n, n), dtype=np.float32)
        np.fill_diagonal(pos_adj, 0)
        neg_adj = np.zeros((n, n), dtype=np.float32)
        return pos_adj, neg_adj

    def build(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建图

        Args:
            features: [N, seq_len, n_features]

        Returns:
            pos_adj, neg_adj
        """
        N = features.shape[0]

        if self.method == 'correlation':
            return self.build_correlation_graph(features)
        elif self.method == 'knn':
            return self.build_knn_graph(features)
        elif self.method == 'full':
            return self.build_full_graph(N)
        else:
            raise ValueError(f"Unknown method: {self.method}")


class PerfusionHeteroDataset(Dataset):
    """
    灌注异构图数据集

    处理流程：
    1. 加载 CSV 数据
    2. 按病例分组，提取时序特征
    3. 构建病例间关系图
    4. 归一化
    """

    FEATURE_COLS = [
        'pH', 'lactate', 'PO2', 'K_plus', 'Na_plus',
        'MAP_mmHg', 'AoF_L_min', 'cardiac_output',
        'ejection_fraction', 'heart_rate'
    ]

    def __init__(
        self,
        csv_path: str,
        seq_length: int = 6,
        feature_cols: List[str] = None,
        graph_method: str = 'correlation',
        pos_threshold: float = 0.3,
        neg_threshold: float = -0.3,
        normalize: bool = True,
        augment: bool = False,
        noise_std: float = 0.02,
    ):
        self.seq_length = seq_length
        self.feature_cols = feature_cols or self.FEATURE_COLS
        self.normalize = normalize
        self.augment = augment
        self.noise_std = noise_std

        # 加载数据
        self.df = pd.read_csv(csv_path)
        self.feature_cols = [c for c in self.feature_cols if c in self.df.columns]

        print(f"Using features: {self.feature_cols}")

        # 构建样本
        self.samples = self._build_samples()

        # 归一化
        if self.normalize and len(self.samples) > 0:
            self._fit_normalizer()

        # 构建图
        self.graph_builder = PerfusionGraphBuilder(
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            method=graph_method
        )

        print(f"Dataset: {len(self.samples)} cases")

    def _build_samples(self) -> List[Dict]:
        """构建样本列表"""
        samples = []

        for case_id, group in self.df.groupby('case_id'):
            group = group.sort_values('timestamp')

            # 提取特征
            features = group[self.feature_cols].values.astype(np.float32)

            # 填充 NaN
            for i in range(features.shape[1]):
                col = features[:, i]
                mask = np.isnan(col)
                if mask.any():
                    col_mean = np.nanmean(col) if not mask.all() else 0.0
                    features[mask, i] = col_mean

            # 序列长度处理
            if len(features) < self.seq_length:
                pad = np.repeat(features[-1:], self.seq_length - len(features), axis=0)
                features = np.vstack([features, pad])
            elif len(features) > self.seq_length:
                indices = np.linspace(0, len(features) - 1, self.seq_length, dtype=int)
                features = features[indices]

            # 标签
            quality = group['quality_score'].iloc[-1]

            samples.append({
                'case_id': case_id,
                'features': features,  # [seq_length, n_features]
                'quality_score': quality / 100.0,  # 归一化到 0-1
            })

        return samples

    def _fit_normalizer(self):
        """拟合归一化器"""
        all_features = np.vstack([s['features'] for s in self.samples])
        self.mean = np.mean(all_features, axis=0)
        self.std = np.std(all_features, axis=0) + 1e-8

    def set_normalizer(self, mean: np.ndarray, std: np.ndarray):
        """设置外部归一化参数"""
        self.mean = mean
        self.std = std

    def get_graph_data(self, indices: List[int] = None) -> GraphData:
        """
        获取图数据

        Args:
            indices: 样本索引列表，None 表示全部

        Returns:
            GraphData 对象
        """
        if indices is None:
            indices = list(range(len(self.samples)))

        # 提取特征和标签
        features_list = []
        labels_list = []
        case_ids = []

        for idx in indices:
            s = self.samples[idx]
            feat = s['features'].copy()

            # 数据增强
            if self.augment:
                noise = np.random.normal(0, self.noise_std, feat.shape)
                feat = feat + noise * self.std

            # 归一化
            if self.normalize:
                feat = (feat - self.mean) / self.std

            features_list.append(feat)
            labels_list.append(s['quality_score'])
            case_ids.append(s['case_id'])

        features = np.stack(features_list)  # [N, seq_len, n_features]
        labels = np.array(labels_list)

        # 构建图
        # 使用归一化后的特征构建图
        pos_adj, neg_adj = self.graph_builder.build(features)

        return GraphData(
            features=torch.tensor(features, dtype=torch.float32),
            pos_adj=torch.tensor(pos_adj, dtype=torch.float32),
            neg_adj=torch.tensor(neg_adj, dtype=torch.float32),
            labels=torch.tensor(labels, dtype=torch.float32),
            case_ids=case_ids,
            mask=torch.ones(len(indices), dtype=torch.bool)
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本（不包含图结构）"""
        s = self.samples[idx]
        feat = s['features'].copy()

        if self.augment:
            noise = np.random.normal(0, self.noise_std, feat.shape)
            feat = feat + noise * self.std

        if self.normalize:
            feat = (feat - self.mean) / self.std

        return {
            'features': torch.tensor(feat, dtype=torch.float32),
            'quality_score': torch.tensor(s['quality_score'], dtype=torch.float32),
            'case_id': s['case_id']
        }


def build_hetero_graph(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 6,
    method: str = 'correlation',
    pos_threshold: float = 0.3,
    neg_threshold: float = -0.3,
) -> GraphData:
    """
    从 DataFrame 构建异构图

    便捷函数，用于快速构建图数据

    Args:
        df: 包含时序数据的 DataFrame
        feature_cols: 特征列名
        seq_length: 序列长度
        method: 图构建方法
        pos_threshold: 正关系阈值
        neg_threshold: 负关系阈值

    Returns:
        GraphData
    """
    # 创建临时 CSV
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    try:
        dataset = PerfusionHeteroDataset(
            csv_path=temp_path,
            seq_length=seq_length,
            feature_cols=feature_cols,
            graph_method=method,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )
        return dataset.get_graph_data()
    finally:
        os.unlink(temp_path)


def split_graph_data(
    graph_data: GraphData,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[GraphData, GraphData]:
    """
    划分图数据为训练集和验证集

    注意：保持图结构完整，只划分标签

    Args:
        graph_data: 完整图数据
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        (train_data, val_data)
    """
    np.random.seed(seed)
    N = len(graph_data.case_ids)
    indices = np.random.permutation(N)

    n_train = int(N * train_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # 创建 mask
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_indices] = True

    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[val_indices] = True

    train_data = GraphData(
        features=graph_data.features,
        pos_adj=graph_data.pos_adj,
        neg_adj=graph_data.neg_adj,
        labels=graph_data.labels,
        case_ids=graph_data.case_ids,
        mask=train_mask
    )

    val_data = GraphData(
        features=graph_data.features,
        pos_adj=graph_data.pos_adj,
        neg_adj=graph_data.neg_adj,
        labels=graph_data.labels,
        case_ids=graph_data.case_ids,
        mask=val_mask
    )

    return train_data, val_data
