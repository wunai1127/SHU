"""
THGNN Training Script - 训练脚本

训练流程：
1. 加载数据并构建图
2. 交叉验证训练
3. 评估并保存模型
"""

import os
import argparse
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm

from .model import PerfusionTHGNN, SimplePerfusionTHGNN, THGNNConfig
from .data import PerfusionHeteroDataset, GraphData


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class THGNNTrainer:
    """
    THGNN 训练器

    支持：
    - K-Fold 交叉验证
    - Leave-One-Out 交叉验证
    - 早停
    - 学习率调度
    """

    def __init__(
        self,
        config: THGNNConfig,
        device: str = 'cpu',
        use_graph: bool = True,
    ):
        self.config = config
        self.device = torch.device(device)
        self.use_graph = use_graph

    def create_model(self) -> nn.Module:
        """创建模型"""
        if self.use_graph:
            model = PerfusionTHGNN(self.config)
        else:
            model = SimplePerfusionTHGNN(self.config)
        return model.to(self.device)

    def train_epoch(
        self,
        model: nn.Module,
        graph_data: GraphData,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
    ) -> float:
        """训练一个 epoch"""
        model.train()

        features = graph_data.features.to(self.device)
        pos_adj = graph_data.pos_adj.to(self.device)
        neg_adj = graph_data.neg_adj.to(self.device)
        labels = graph_data.labels.to(self.device)
        mask = graph_data.mask.to(self.device)

        optimizer.zero_grad()

        if self.use_graph:
            outputs = model(features, pos_adj, neg_adj)
        else:
            outputs = model(features)

        # 只计算 mask 内的损失
        loss = loss_fn(outputs[mask], labels[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    def evaluate(
        self,
        model: nn.Module,
        graph_data: GraphData,
    ) -> Dict[str, float]:
        """评估模型"""
        model.eval()

        features = graph_data.features.to(self.device)
        pos_adj = graph_data.pos_adj.to(self.device)
        neg_adj = graph_data.neg_adj.to(self.device)
        labels = graph_data.labels.to(self.device)
        mask = graph_data.mask.to(self.device)

        with torch.no_grad():
            if self.use_graph:
                outputs = model(features, pos_adj, neg_adj)
            else:
                outputs = model(features)

        # 提取 mask 内的预测
        preds = outputs[mask].cpu().numpy() * 100  # 转回 0-100
        targets = labels[mask].cpu().numpy() * 100

        metrics = {}
        if len(preds) > 1:
            metrics['mse'] = mean_squared_error(targets, preds)
            metrics['mae'] = mean_absolute_error(targets, preds)
            metrics['r2'] = r2_score(targets, preds)
        else:
            # 单样本
            metrics['mse'] = (preds[0] - targets[0]) ** 2
            metrics['mae'] = abs(preds[0] - targets[0])
            metrics['r2'] = 0.0

        return metrics, preds, targets

    def train_fold(
        self,
        train_data: GraphData,
        val_data: GraphData,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 20,
    ) -> Tuple[nn.Module, Dict]:
        """训练一个 fold"""
        model = self.create_model()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        loss_fn = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_r2': []}

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(model, train_data, optimizer, loss_fn)
            scheduler.step()

            # Validate
            val_metrics, _, _ = self.evaluate(model, val_data)
            val_loss = val_metrics['mse']
            val_r2 = val_metrics['r2']

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_r2'].append(val_r2)

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"val_mse={val_loss:.4f}, val_r2={val_r2:.4f}")

            # 早停
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, history


def create_fold_data(
    dataset: PerfusionHeteroDataset,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> Tuple[GraphData, GraphData]:
    """
    为 fold 创建图数据

    重要：使用所有样本构建图，但训练/验证 mask 不同
    """
    # 获取完整图数据
    all_data = dataset.get_graph_data()

    N = len(dataset)

    # 创建 mask
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[train_indices] = True

    val_mask = torch.zeros(N, dtype=torch.bool)
    val_mask[val_indices] = True

    train_data = GraphData(
        features=all_data.features,
        pos_adj=all_data.pos_adj,
        neg_adj=all_data.neg_adj,
        labels=all_data.labels,
        case_ids=all_data.case_ids,
        mask=train_mask
    )

    val_data = GraphData(
        features=all_data.features,
        pos_adj=all_data.pos_adj,
        neg_adj=all_data.neg_adj,
        labels=all_data.labels,
        case_ids=all_data.case_ids,
        mask=val_mask
    )

    return train_data, val_data


def train_thgnn(
    csv_path: str,
    output_dir: str = './checkpoints',
    n_folds: int = 5,
    epochs: int = 100,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    use_graph: bool = True,
    graph_method: str = 'correlation',
    device: str = 'cpu',
    seed: int = 42,
) -> Dict:
    """
    训练 THGNN 的便捷函数

    Args:
        csv_path: 数据 CSV 路径
        output_dir: 输出目录
        n_folds: K-Fold 数量（-1 表示 LOO）
        epochs: 训练轮数
        lr: 学习率
        hidden_dim: 隐藏层维度
        use_graph: 是否使用图结构
        graph_method: 图构建方法
        device: 设备
        seed: 随机种子

    Returns:
        训练结果字典
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    dataset = PerfusionHeteroDataset(
        csv_path=csv_path,
        graph_method=graph_method,
    )

    n_cases = len(dataset)
    print(f"Total cases: {n_cases}")

    # 配置
    config = THGNNConfig(
        in_features=len(dataset.feature_cols),
        temporal_hidden=hidden_dim,
        gat_out_features=hidden_dim // 4,
        gat_num_heads=4,
        sem_hidden=hidden_dim,
        predictor_hidden=hidden_dim // 2,
    )

    trainer = THGNNTrainer(config, device=device, use_graph=use_graph)

    # 交叉验证
    if n_folds == -1 or n_cases < 10:
        print("Using Leave-One-Out Cross-Validation")
        cv = LeaveOneOut()
        n_splits = n_cases
    else:
        n_splits = min(n_folds, n_cases)
        print(f"Using {n_splits}-Fold Cross-Validation")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    all_r2 = []
    all_mse = []
    all_predictions = []

    indices = np.arange(n_cases)

    for fold, (train_idx, val_idx) in enumerate(cv.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
        print(f"{'='*50}")

        # 创建 fold 数据
        train_data, val_data = create_fold_data(dataset, train_idx, val_idx)

        # 训练
        model, history = trainer.train_fold(
            train_data, val_data,
            epochs=epochs, lr=lr
        )

        # 评估
        metrics, preds, targets = trainer.evaluate(model, val_data)
        print(f"\nFold {fold + 1} Results: R²={metrics['r2']:.4f}, MSE={metrics['mse']:.4f}")

        all_r2.append(metrics['r2'])
        all_mse.append(metrics['mse'])

        # 保存预测
        val_case_ids = [dataset.samples[i]['case_id'] for i in val_idx]
        for case_id, pred, target in zip(val_case_ids, preds, targets):
            all_predictions.append({
                'case_id': case_id,
                'predicted': pred,
                'actual': target,
                'fold': fold
            })

    # 汇总
    pred_df = pd.DataFrame(all_predictions)
    overall_r2 = r2_score(pred_df['actual'], pred_df['predicted'])
    overall_mse = mean_squared_error(pred_df['actual'], pred_df['predicted'])
    overall_mae = mean_absolute_error(pred_df['actual'], pred_df['predicted'])

    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Per-fold R²: {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
    print(f"Overall R² (all predictions): {overall_r2:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall MAE: {overall_mae:.4f}")

    # 保存预测
    pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    print(f"\nPredictions saved to {output_dir}/predictions.csv")

    # 在全部数据上训练最终模型
    if overall_r2 > 0:
        print("\nTraining final model on all data...")
        all_data = dataset.get_graph_data()

        final_model = trainer.create_model()
        optimizer = optim.AdamW(final_model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        for epoch in tqdm(range(epochs), desc="Final training"):
            trainer.train_epoch(final_model, all_data, optimizer, loss_fn)

        # 保存
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'config': config,
            'feature_cols': dataset.feature_cols,
            'mean': dataset.mean,
            'std': dataset.std,
        }, os.path.join(output_dir, 'thgnn_model.pt'))

        print(f"Model saved to {output_dir}/thgnn_model.pt")

    return {
        'per_fold_r2': all_r2,
        'overall_r2': overall_r2,
        'overall_mse': overall_mse,
        'overall_mae': overall_mae,
        'predictions': pred_df,
    }


def main():
    parser = argparse.ArgumentParser(description='Train THGNN for Perfusion Prediction')
    parser.add_argument('--data', type=str, required=True, help='CSV data path')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--folds', type=int, default=5, help='K-fold (use -1 for LOO)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--no-graph', action='store_true', help='Disable graph structure')
    parser.add_argument('--graph-method', type=str, default='correlation',
                       choices=['correlation', 'knn', 'full'], help='Graph construction method')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    results = train_thgnn(
        csv_path=args.data,
        output_dir=args.output,
        n_folds=args.folds,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden,
        use_graph=not args.no_graph,
        graph_method=args.graph_method,
        device=args.device,
        seed=args.seed,
    )

    print(f"\nFinal Results:")
    print(f"  Overall R²: {results['overall_r2']:.4f}")
    print(f"  Overall MSE: {results['overall_mse']:.4f}")


if __name__ == '__main__':
    main()
