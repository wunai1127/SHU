#!/usr/bin/env python3
"""
GNN Training Script - Heart Perfusion Quality Prediction

GNN 训练脚本 - 心脏灌注质量预测模型

=============================================================================
使用方法 (How to Run):
=============================================================================

1. 准备数据后，直接运行：
   python train_gnn.py

2. 指定配置：
   python train_gnn.py --epochs 200 --batch_size 32 --lr 0.001

3. 从checkpoint恢复训练：
   python train_gnn.py --resume checkpoints/best_model.pt

4. 使用TensorBoard查看训练曲线：
   tensorboard --logdir=logs/

=============================================================================
数据格式要求 (Data Format Requirements):
=============================================================================

CSV 文件格式：
- 每行是一个时间点的测量值
- 必须包含 case_id 列来区分不同病例
- 必须包含 timestamp 列来排序
- 必须包含12个特征列和标签列

示例 CSV:
case_id,timestamp,pH,PO2,PCO2,lactate,K_plus,Na_plus,IL_6,IL_8,TNF_alpha,pressure,flow_rate,temperature,quality_score,risk_level,usable
CASE001,0,7.40,400,40,1.5,4.0,140,10,8,5,60,1.5,34,85,low,1
CASE001,1,7.38,380,42,1.8,4.1,139,12,10,6,58,1.5,34,85,low,1
...

=============================================================================
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from graphrag_agent.perfusion.temporal_gnn import (
    TemporalPerfusionGNN,
    TemporalGNNConfig,
    FeatureNormalizer,
    create_dummy_graph,
)
from graphrag_agent.perfusion.training.config import training_config, data_config


# =============================================================================
# 第一步：数据集类
# Step 1: Dataset Class
# =============================================================================

class PerfusionTrainingDataset(Dataset):
    """
    灌注数据集类

    这个类负责：
    1. 从CSV加载数据
    2. 按病例分组，构建时间序列
    3. 归一化特征
    4. 返回训练所需的张量
    """

    def __init__(
        self,
        csv_path: str,
        seq_length: int = 20,
        feature_columns: List[str] = None,
        normalize: bool = True,
    ):
        """
        初始化数据集

        Args:
            csv_path: CSV文件路径
            seq_length: 序列长度（每个样本包含多少时间点）
            feature_columns: 特征列名列表
            normalize: 是否归一化特征
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.seq_length = seq_length
        self.feature_columns = feature_columns or data_config.feature_columns

        # 检查必要的列是否存在
        # Check required columns exist
        self._validate_columns()

        # 按病例分组，构建序列
        # Group by case and build sequences
        self.samples = self._build_sequences()

        # 归一化
        # Normalize
        self.normalize = normalize
        if normalize:
            self.normalizer = FeatureNormalizer()
            self._fit_normalizer()

        print(f"Loaded {len(self.samples)} samples from {len(self.df['case_id'].unique())} cases")

    def _validate_columns(self):
        """验证CSV包含所有必要的列"""
        required = ['case_id', 'timestamp'] + self.feature_columns
        missing = [col for col in required if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # 检查标签列（至少需要一个）
        label_cols = list(data_config.label_columns.values())
        if not any(col in self.df.columns for col in label_cols):
            raise ValueError(f"Need at least one label column from: {label_cols}")

    def _build_sequences(self) -> List[Dict]:
        """
        按病例构建时间序列

        将每个病例的测量值按时间排序，然后切分成固定长度的序列
        """
        samples = []

        for case_id, group in self.df.groupby('case_id'):
            # 按时间排序
            group = group.sort_values('timestamp')

            # 提取特征矩阵
            features = group[self.feature_columns].values

            # 如果序列太短，跳过或填充
            if len(features) < self.seq_length:
                # 用最后一行填充
                padding = np.repeat(
                    features[-1:],
                    self.seq_length - len(features),
                    axis=0
                )
                features = np.vstack([features, padding])

            # 提取标签（使用最后一个时间点的标签）
            labels = {}
            if 'quality_score' in group.columns:
                labels['quality_score'] = float(group['quality_score'].iloc[-1])
            if 'risk_level' in group.columns:
                risk_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                risk_str = group['risk_level'].iloc[-1]
                labels['risk_level'] = risk_map.get(risk_str, 1)
            if 'usable' in group.columns:
                labels['usable'] = int(group['usable'].iloc[-1])

            # 滑动窗口采样（如果序列很长，可以获得多个样本）
            for start_idx in range(0, len(features) - self.seq_length + 1, self.seq_length // 2):
                end_idx = start_idx + self.seq_length
                sample = {
                    'case_id': case_id,
                    'features': features[start_idx:end_idx].copy(),
                    'labels': labels.copy(),
                }
                samples.append(sample)

        return samples

    def _fit_normalizer(self):
        """拟合归一化器（计算均值和标准差）"""
        all_features = np.vstack([s['features'] for s in self.samples])
        self.normalizer.fit(all_features)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            Dict with:
            - 'features': [seq_length, num_features] 特征张量
            - 'quality_score': scalar 质量分数
            - 'risk_level': scalar 风险等级 (0-3)
            - 'usable': scalar 是否可用 (0/1)
        """
        sample = self.samples[idx]
        features = sample['features'].copy()

        # 归一化
        if self.normalize:
            features = self.normalizer.transform(features)

        result = {
            'features': torch.tensor(features, dtype=torch.float32),
        }

        # 添加标签
        labels = sample['labels']
        if 'quality_score' in labels:
            result['quality_score'] = torch.tensor(labels['quality_score'] / 100.0, dtype=torch.float32)
        if 'risk_level' in labels:
            result['risk_level'] = torch.tensor(labels['risk_level'], dtype=torch.long)
        if 'usable' in labels:
            result['usable'] = torch.tensor(labels['usable'], dtype=torch.float32)

        return result


# =============================================================================
# 第二步：训练器类
# Step 2: Trainer Class
# =============================================================================

class GNNTrainer:
    """
    GNN模型训练器

    负责：
    1. 模型初始化
    2. 训练循环
    3. 验证和测试
    4. 保存检查点
    5. TensorBoard日志
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader = None,
        config: 'GNNTrainingConfig' = None,
        device: str = 'cuda',
    ):
        """
        初始化训练器

        Args:
            model: GNN模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
            config: 训练配置
            device: 训练设备 ('cuda' or 'cpu')
        """
        self.config = config or training_config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 模型
        self.model = model.to(self.device)

        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # 优化器 - Adam with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 学习率调度器 - 验证loss不下降时降低学习率
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,       # 降低为原来的一半
            patience=5,       # 5轮不下降就降低
            verbose=True,
        )

        # 损失函数
        self.quality_loss = nn.MSELoss()           # 质量分数 - 回归
        self.risk_loss = nn.CrossEntropyLoss()     # 风险等级 - 分类
        self.usable_loss = nn.BCEWithLogitsLoss()  # 是否可用 - 二分类

        # TensorBoard
        log_dir = Path(self.config.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")

        # 检查点目录
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 早停相关
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 检查模型是否使用图编码器
        # Check if model uses graph encoder
        self.use_graph = getattr(model, 'use_graph_encoder', True)

        # 虚拟图（仅在使用图编码器时需要）
        # Dummy graph (only needed when using graph encoder)
        if self.use_graph:
            self.dummy_graph = create_dummy_graph().to(self.device)
        else:
            self.dummy_graph = None

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            Dict of metrics
        """
        self.model.train()
        total_loss = 0
        quality_losses = []
        risk_losses = []

        # 进度条
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')

        for batch_idx, batch in enumerate(pbar):
            # 移动数据到设备
            features = batch['features'].to(self.device)

            # 前向传播
            outputs = self.model(features, self.dummy_graph)

            # 计算损失
            loss = 0

            # 质量分数损失
            if 'quality_score' in batch:
                quality_target = batch['quality_score'].to(self.device)
                quality_pred = outputs['quality_score'].squeeze()
                q_loss = self.quality_loss(quality_pred, quality_target)
                loss += q_loss
                quality_losses.append(q_loss.item())

            # 风险等级损失
            if 'risk_level' in batch:
                risk_target = batch['risk_level'].to(self.device)
                risk_pred = outputs['risk_logits']
                r_loss = self.risk_loss(risk_pred, risk_target)
                loss += r_loss
                risk_losses.append(r_loss.item())

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'q_loss': f'{np.mean(quality_losses):.4f}' if quality_losses else 'N/A',
            })

        avg_loss = total_loss / len(self.train_loader)
        return {
            'loss': avg_loss,
            'quality_loss': np.mean(quality_losses) if quality_losses else 0,
            'risk_loss': np.mean(risk_losses) if risk_losses else 0,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        在验证集上评估

        Returns:
            Dict of metrics
        """
        self.model.eval()
        total_loss = 0
        all_quality_preds = []
        all_quality_targets = []
        all_risk_preds = []
        all_risk_targets = []

        for batch in self.val_loader:
            features = batch['features'].to(self.device)
            outputs = self.model(features, self.dummy_graph)

            loss = 0

            if 'quality_score' in batch:
                quality_target = batch['quality_score'].to(self.device)
                quality_pred = outputs['quality_score'].squeeze()
                loss += self.quality_loss(quality_pred, quality_target)

                all_quality_preds.extend(quality_pred.cpu().numpy())
                all_quality_targets.extend(quality_target.cpu().numpy())

            if 'risk_level' in batch:
                risk_target = batch['risk_level'].to(self.device)
                risk_pred = outputs['risk_logits']
                loss += self.risk_loss(risk_pred, risk_target)

                pred_class = risk_pred.argmax(dim=1)
                all_risk_preds.extend(pred_class.cpu().numpy())
                all_risk_targets.extend(risk_target.cpu().numpy())

            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)

        metrics = {'loss': avg_loss}

        # 计算回归指标
        if all_quality_preds:
            mse = mean_squared_error(all_quality_targets, all_quality_preds)
            r2 = r2_score(all_quality_targets, all_quality_preds)
            metrics['quality_mse'] = mse
            metrics['quality_r2'] = r2

        # 计算分类指标
        if all_risk_preds:
            acc = accuracy_score(all_risk_targets, all_risk_preds)
            f1 = f1_score(all_risk_targets, all_risk_preds, average='weighted')
            metrics['risk_accuracy'] = acc
            metrics['risk_f1'] = f1

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }

        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点（用于恢复训练）

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']

        return checkpoint['epoch']

    def train(self, num_epochs: int = None, resume_from: str = None) -> Dict:
        """
        完整训练流程

        Args:
            num_epochs: 训练轮数
            resume_from: 从哪个检查点恢复

        Returns:
            训练历史
        """
        num_epochs = num_epochs or self.config.num_epochs
        start_epoch = 0

        # 恢复训练
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming from epoch {start_epoch}")

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
        }

        print("=" * 60)
        print("Starting Training")
        print("=" * 60)

        for epoch in range(start_epoch, num_epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            history['train_loss'].append(train_metrics['loss'])

            # 验证
            val_metrics = self.validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)

            # 学习率调度
            self.scheduler.step(val_metrics['loss'])

            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            if 'quality_r2' in val_metrics:
                self.writer.add_scalar('Metrics/quality_r2', val_metrics['quality_r2'], epoch)
            if 'risk_accuracy' in val_metrics:
                self.writer.add_scalar('Metrics/risk_accuracy', val_metrics['risk_accuracy'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # 打印进度
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            if 'quality_r2' in val_metrics:
                print(f"  Quality R²: {val_metrics['quality_r2']:.4f}")
            if 'risk_accuracy' in val_metrics:
                print(f"  Risk Acc:   {val_metrics['risk_accuracy']:.4f}")

            # 检查是否是最佳模型
            is_best = val_metrics['loss'] < self.best_val_loss - self.config.min_delta
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 保存检查点
            self.save_checkpoint(epoch, is_best)

            # 早停检查
            if self.patience_counter >= self.config.patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                print(f"   Validation loss hasn't improved for {self.config.patience} epochs")
                break

        # 关闭TensorBoard
        self.writer.close()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        print("=" * 60)

        return history

    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """
        在测试集上评估最终模型

        Returns:
            测试指标
        """
        if self.test_loader is None:
            print("No test loader provided")
            return {}

        # 加载最佳模型
        best_path = self.checkpoint_dir / 'best_model.pt'
        if best_path.exists():
            self.load_checkpoint(str(best_path))

        self.model.eval()

        all_quality_preds = []
        all_quality_targets = []
        all_risk_preds = []
        all_risk_targets = []

        for batch in tqdm(self.test_loader, desc='Testing'):
            features = batch['features'].to(self.device)
            outputs = self.model(features, self.dummy_graph)

            if 'quality_score' in batch:
                quality_pred = outputs['quality_score'].squeeze()
                all_quality_preds.extend(quality_pred.cpu().numpy())
                all_quality_targets.extend(batch['quality_score'].numpy())

            if 'risk_level' in batch:
                risk_pred = outputs['risk_logits'].argmax(dim=1)
                all_risk_preds.extend(risk_pred.cpu().numpy())
                all_risk_targets.extend(batch['risk_level'].numpy())

        metrics = {}

        if all_quality_preds:
            # 转换回 0-100 分数
            preds_100 = np.array(all_quality_preds) * 100
            targets_100 = np.array(all_quality_targets) * 100

            metrics['quality_mse'] = mean_squared_error(targets_100, preds_100)
            metrics['quality_rmse'] = np.sqrt(metrics['quality_mse'])
            metrics['quality_r2'] = r2_score(targets_100, preds_100)
            metrics['quality_mae'] = np.mean(np.abs(targets_100 - preds_100))

        if all_risk_preds:
            metrics['risk_accuracy'] = accuracy_score(all_risk_targets, all_risk_preds)
            metrics['risk_f1'] = f1_score(all_risk_targets, all_risk_preds, average='weighted')

        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics


# =============================================================================
# 第三步：创建模拟数据（用于测试）
# Step 3: Create Synthetic Data (for testing)
# =============================================================================

def create_synthetic_data(
    num_cases: int = 100,
    seq_length: int = 20,
    output_path: str = './data/synthetic_perfusion.csv'
) -> str:
    """
    创建模拟数据用于测试训练流程

    生成符合真实分布的模拟灌注数据

    Args:
        num_cases: 病例数量
        seq_length: 每个病例的时间点数量
        output_path: 输出CSV路径

    Returns:
        输出文件路径
    """
    print(f"Creating synthetic data with {num_cases} cases...")

    np.random.seed(42)
    rows = []

    for case_idx in range(num_cases):
        case_id = f"CASE{case_idx:04d}"

        # 随机决定这个心脏的基线状态
        # 好心脏 vs 差心脏
        is_good = np.random.random() > 0.3  # 70% 是好心脏

        # 基线值
        if is_good:
            base_ph = 7.38 + np.random.normal(0, 0.02)
            base_lactate = 1.5 + np.random.normal(0, 0.3)
            quality_score = 75 + np.random.normal(0, 10)
            risk_level = np.random.choice(['low', 'medium'], p=[0.7, 0.3])
        else:
            base_ph = 7.28 + np.random.normal(0, 0.04)
            base_lactate = 3.5 + np.random.normal(0, 0.8)
            quality_score = 45 + np.random.normal(0, 15)
            risk_level = np.random.choice(['medium', 'high', 'critical'], p=[0.3, 0.5, 0.2])

        quality_score = np.clip(quality_score, 0, 100)
        usable = 1 if quality_score >= 60 else 0

        # 生成时间序列
        for t in range(seq_length):
            # 添加时间趋势和噪声
            trend = t / seq_length * 0.1  # 轻微下降趋势

            row = {
                'case_id': case_id,
                'timestamp': t,
                'pH': base_ph - trend * 0.1 + np.random.normal(0, 0.01),
                'PO2': 400 - trend * 50 + np.random.normal(0, 20),
                'PCO2': 40 + trend * 5 + np.random.normal(0, 3),
                'lactate': base_lactate + trend * 1.5 + np.random.normal(0, 0.2),
                'K_plus': 4.0 + trend * 0.3 + np.random.normal(0, 0.1),
                'Na_plus': 140 - trend * 2 + np.random.normal(0, 1),
                'IL_6': 15 + trend * 20 + np.random.exponential(5),
                'IL_8': 10 + trend * 15 + np.random.exponential(3),
                'TNF_alpha': 8 + trend * 10 + np.random.exponential(2),
                'pressure': 60 - trend * 10 + np.random.normal(0, 3),
                'flow_rate': 1.5 - trend * 0.2 + np.random.normal(0, 0.1),
                'temperature': 34 + trend * 2 + np.random.normal(0, 0.5),
                'quality_score': quality_score,
                'risk_level': risk_level,
                'usable': usable,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Saved synthetic data to {output_path}")
    print(f"  - Total rows: {len(df)}")
    print(f"  - Unique cases: {df['case_id'].nunique()}")

    return output_path


# =============================================================================
# 第四步：主函数
# Step 4: Main Function
# =============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train GNN for Perfusion Prediction')

    # 数据参数
    parser.add_argument('--data', type=str, default='./data/synthetic_perfusion.csv',
                       help='Path to training data CSV')
    parser.add_argument('--create_synthetic', action='store_true',
                       help='Create synthetic data for testing')
    parser.add_argument('--num_cases', type=int, default=200,
                       help='Number of synthetic cases to create')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=training_config.num_epochs,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=training_config.batch_size,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=training_config.learning_rate,
                       help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=training_config.seq_length,
                       help='Sequence length')

    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建模拟数据（如果需要）
    if args.create_synthetic or not Path(args.data).exists():
        print("\n⚠️ Training data not found, creating synthetic data...")
        args.data = create_synthetic_data(num_cases=args.num_cases)

    # 加载数据集
    print("\n" + "=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    dataset = PerfusionTrainingDataset(
        csv_path=args.data,
        seq_length=args.seq_length,
    )

    # 划分数据集
    total_size = len(dataset)
    train_size = int(total_size * training_config.train_split)
    val_size = int(total_size * training_config.val_split)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # 创建模型
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)

    gnn_config = TemporalGNNConfig(
        hidden_dim=training_config.hidden_dim,
        num_layers=training_config.num_layers,
        dropout=training_config.dropout,
        num_features=training_config.num_features,
        # 禁用图编码器，因为目前没有真实的知识图谱数据
        # 只使用 LSTM 时序编码器进行训练
        # Disable graph encoder since we don't have real knowledge graph data
        # Only use LSTM temporal encoder for training
        use_graph_encoder=False,
    )

    model = TemporalPerfusionGNN(gnn_config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # 创建训练器
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config,
        device=args.device,
    )

    # 训练
    history = trainer.train(
        num_epochs=args.epochs,
        resume_from=args.resume,
    )

    # 测试
    test_metrics = trainer.test()

    # 保存训练历史
    history_path = Path(training_config.checkpoint_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'test_metrics': test_metrics,
        }, f, indent=2)

    print(f"\nTraining history saved to {history_path}")


if __name__ == '__main__':
    main()
