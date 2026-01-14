#!/usr/bin/env python3
"""
小数据集专用训练脚本 V2

针对31-32个病例的EVHP数据优化：
1. 添加趋势特征（变化率、斜率）
2. 降低正则化强度
3. 使用注意力机制
4. 添加基线模型对比
5. 在线数据增强
6. Leave-One-Out 交叉验证选项

修复的问题：
- dropout 0.5 太高 → 降到 0.2
- 没有趋势特征 → 添加差分和统计特征
- 只用最后时刻 → 使用注意力加权
"""

import os
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, LogisticRegression
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EnhancedPerfusionDataset(Dataset):
    """
    增强版数据集

    改进：
    1. 添加趋势特征（差分、斜率）
    2. 添加统计特征（均值、标准差、最大最小值）
    3. 在线数据增强
    """

    def __init__(
        self,
        df,
        seq_length=6,
        feature_cols=None,
        augment=False,
        noise_std=0.02,
    ):
        self.seq_length = seq_length
        self.augment = augment
        self.noise_std = noise_std

        self.raw_feature_cols = feature_cols or [
            'pH', 'lactate', 'PO2', 'K_plus', 'Na_plus',
            'MAP_mmHg', 'AoF_L_min', 'cardiac_output',
            'ejection_fraction', 'heart_rate'
        ]

        # 只保留存在的特征列
        self.raw_feature_cols = [c for c in self.raw_feature_cols if c in df.columns]

        self.samples = self._build_samples(df)

        # 计算归一化参数（在原始特征上）
        if len(self.samples) > 0:
            all_features = np.vstack([s['raw_features'] for s in self.samples])
            self.mean = np.nanmean(all_features, axis=0)
            self.std = np.nanstd(all_features, axis=0) + 1e-8
        else:
            self.mean = np.zeros(len(self.raw_feature_cols))
            self.std = np.ones(len(self.raw_feature_cols))

        print(f"Dataset: {len(self.samples)} samples, {len(self.raw_feature_cols)} raw features")

    def _build_samples(self, df):
        samples = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('timestamp')
            raw_features = group[self.raw_feature_cols].values.astype(np.float32)

            # 填充NaN（用列均值而不是0）
            for i in range(raw_features.shape[1]):
                col = raw_features[:, i]
                mask = np.isnan(col)
                if mask.any():
                    col_mean = np.nanmean(col) if not mask.all() else 0.0
                    raw_features[mask, i] = col_mean

            # 序列长度处理
            if len(raw_features) < self.seq_length:
                pad = np.repeat(raw_features[-1:], self.seq_length - len(raw_features), axis=0)
                raw_features = np.vstack([raw_features, pad])
            elif len(raw_features) > self.seq_length:
                # 均匀采样而不是截断
                indices = np.linspace(0, len(raw_features) - 1, self.seq_length, dtype=int)
                raw_features = raw_features[indices]

            # 获取标签
            quality = group['quality_score'].iloc[-1]
            usable = group['usable'].iloc[-1] if 'usable' in group.columns else (1 if quality >= 50 else 0)

            samples.append({
                'case_id': case_id,
                'raw_features': raw_features,
                'quality_score': quality / 100.0,
                'usable': float(usable),
            })
        return samples

    def _compute_enhanced_features(self, raw_features):
        """
        计算增强特征

        输入: [seq_len, n_features]
        输出: [seq_len, n_features * 2] (原始 + 差分)
        """
        # 归一化原始特征
        normalized = (raw_features - self.mean) / self.std

        # 计算差分（变化率）
        diff = np.zeros_like(normalized)
        diff[1:] = normalized[1:] - normalized[:-1]

        # 合并
        enhanced = np.concatenate([normalized, diff], axis=1)

        return enhanced.astype(np.float32)

    def _compute_summary_features(self, raw_features):
        """
        计算摘要统计特征（用于基线模型）

        输入: [seq_len, n_features]
        输出: [n_features * 6] (均值, 标准差, 最大, 最小, 首, 末)
        """
        normalized = (raw_features - self.mean) / self.std

        features = []
        features.append(np.mean(normalized, axis=0))      # 均值
        features.append(np.std(normalized, axis=0))       # 标准差
        features.append(np.max(normalized, axis=0))       # 最大值
        features.append(np.min(normalized, axis=0))       # 最小值
        features.append(normalized[0])                     # 首值
        features.append(normalized[-1])                    # 末值

        # 添加趋势（末值 - 首值）
        features.append(normalized[-1] - normalized[0])

        return np.concatenate(features).astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        raw_features = s['raw_features'].copy()

        # 在线数据增强
        if self.augment:
            noise = np.random.normal(0, self.noise_std, raw_features.shape)
            raw_features = raw_features + noise * self.std  # 按特征尺度添加噪声

        # 计算增强特征
        enhanced = self._compute_enhanced_features(raw_features)
        summary = self._compute_summary_features(raw_features)

        return {
            'features': torch.tensor(enhanced, dtype=torch.float32),
            'summary': torch.tensor(summary, dtype=torch.float32),
            'quality_score': torch.tensor(s['quality_score'], dtype=torch.float32),
            'usable': torch.tensor(s['usable'], dtype=torch.float32),
        }


class AttentionLSTM(nn.Module):
    """
    带注意力机制的LSTM模型

    改进：
    1. 注意力加权而不是只用最后时刻
    2. 降低 dropout
    3. 添加残差连接
    """

    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 注意力层
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # 质量分数预测头（简化）
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 可用性预测头
        self.usable_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]

        # 注意力权重
        attn_scores = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)

        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        context = self.dropout(context)

        quality = self.quality_head(context).squeeze(-1)
        usable_logit = self.usable_head(context).squeeze(-1)

        return {
            'quality_score': quality,
            'usable_logit': usable_logit,
            'attention_weights': attn_weights.squeeze(-1),
        }


def run_baseline(train_dataset, val_dataset):
    """
    运行基线模型（Ridge回归）
    用于验证数据本身的可预测性
    """
    # 提取摘要特征
    X_train = np.vstack([train_dataset[i]['summary'].numpy() for i in range(len(train_dataset))])
    y_train = np.array([train_dataset[i]['quality_score'].item() for i in range(len(train_dataset))])

    X_val = np.vstack([val_dataset[i]['summary'].numpy() for i in range(len(val_dataset))])
    y_val = np.array([val_dataset[i]['quality_score'].item() for i in range(len(val_dataset))])

    # Ridge 回归
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, 1)

    # 计算 R²
    r2 = r2_score(y_val * 100, y_pred * 100)

    return r2


def train_fold(model, train_loader, val_loader, device, epochs=100, lr=1e-3, quality_weight=1.0):
    """训练一个 fold"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 25

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            features = batch['features'].to(device)
            quality_target = batch['quality_score'].to(device)
            usable_target = batch['usable'].to(device)

            optimizer.zero_grad()
            outputs = model(features)

            # 加权损失
            loss = quality_weight * mse_loss(outputs['quality_score'], quality_target)
            loss += (1 - quality_weight) * bce_loss(outputs['usable_logit'], usable_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_quality_preds, val_quality_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                quality_target = batch['quality_score'].to(device)
                usable_target = batch['usable'].to(device)

                outputs = model(features)

                loss = quality_weight * mse_loss(outputs['quality_score'], quality_target)
                loss += (1 - quality_weight) * bce_loss(outputs['usable_logit'], usable_target)
                val_loss += loss.item()

                val_quality_preds.extend(outputs['quality_score'].cpu().numpy())
                val_quality_targets.extend(quality_target.cpu().numpy())

        val_loss /= max(len(val_loader), 1)

        # 计算 R²
        if len(val_quality_preds) > 1:
            preds_100 = np.array(val_quality_preds) * 100
            targets_100 = np.array(val_quality_targets) * 100
            r2 = r2_score(targets_100, preds_100)
        else:
            r2 = 0.0

        if epoch % 25 == 0:
            print(f"  Epoch {epoch}: loss={val_loss:.4f}, R²={r2:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")

        # 早停
        if val_loss < best_val_loss - 1e-5:
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
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser(description='Small Dataset Training Script V2')
    parser.add_argument('--data', type=str, required=True, help='CSV data path')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--folds', type=int, default=5, help='K-fold (use -1 for LOO)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--augment', action='store_true', help='Enable online augmentation')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quality-weight', type=float, default=0.8, help='Weight for quality loss (vs usable)')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Settings: hidden={args.hidden}, dropout={args.dropout}, lr={args.lr}")

    # 加载数据
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)

    # 过滤无效数据
    df = df.dropna(subset=['quality_score'])
    df = df[df['quality_score'] > 0]  # 去掉 0 分

    print(f"Total rows: {len(df)}, Cases: {df['case_id'].nunique()}")

    # 打印标签分布
    case_quality = df.groupby('case_id')['quality_score'].first()
    print(f"Quality score range: {case_quality.min():.1f} - {case_quality.max():.1f}")
    print(f"Quality score mean: {case_quality.mean():.1f} ± {case_quality.std():.1f}")

    # 获取唯一的病例ID
    case_ids = df['case_id'].unique()
    n_cases = len(case_ids)
    print(f"Unique cases: {n_cases}")

    # 选择交叉验证策略
    if args.folds == -1 or n_cases < 10:
        print("\nUsing Leave-One-Out Cross-Validation")
        cv = LeaveOneOut()
        n_splits = n_cases
    else:
        n_splits = min(args.folds, n_cases)
        print(f"\nUsing {n_splits}-Fold Cross-Validation")
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)

    all_quality_r2 = []
    all_baseline_r2 = []
    all_usable_acc = []
    all_predictions = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(case_ids)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_splits}")
        print(f"{'='*50}")

        train_cases = case_ids[train_idx]
        val_cases = case_ids[val_idx]

        print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)} cases")

        train_df = df[df['case_id'].isin(train_cases)]
        val_df = df[df['case_id'].isin(val_cases)]

        # 创建数据集
        train_dataset = EnhancedPerfusionDataset(train_df, augment=args.augment)
        val_dataset = EnhancedPerfusionDataset(val_df, augment=False)

        # 使用训练集的归一化参数
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std

        # 基线模型
        baseline_r2 = run_baseline(train_dataset, val_dataset)
        all_baseline_r2.append(baseline_r2)
        print(f"  Baseline (Ridge) R²: {baseline_r2:.4f}")

        # 神经网络
        train_loader = DataLoader(train_dataset, batch_size=min(8, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        # 输入维度 = 原始特征 * 2 (原始 + 差分)
        input_dim = len(train_dataset.raw_feature_cols) * 2

        model = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden,
            dropout=args.dropout
        ).to(device)

        # 训练
        model, best_loss = train_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, quality_weight=args.quality_weight
        )

        # 评估
        model.eval()
        quality_preds, quality_targets = [], []
        usable_preds, usable_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                outputs = model(features)

                quality_preds.extend(outputs['quality_score'].cpu().numpy() * 100)
                quality_targets.extend(batch['quality_score'].numpy() * 100)
                usable_preds.extend((torch.sigmoid(outputs['usable_logit']) > 0.5).float().cpu().numpy())
                usable_targets.extend(batch['usable'].numpy())

        # 保存预测结果
        for case, pred, target in zip(val_cases, quality_preds, quality_targets):
            all_predictions.append({
                'case_id': case,
                'predicted': pred,
                'actual': target,
                'fold': fold
            })

        if len(quality_targets) > 1:
            r2 = r2_score(quality_targets, quality_preds)
            mse = mean_squared_error(quality_targets, quality_preds)
        else:
            # LOO: 单个样本
            r2 = 0.0
            mse = (quality_preds[0] - quality_targets[0]) ** 2

        acc = accuracy_score(usable_targets, usable_preds) if len(usable_targets) > 0 else 0.0

        print(f"\nFold {fold + 1} Results:")
        print(f"  Neural Net R²: {r2:.4f} (Baseline: {baseline_r2:.4f})")
        print(f"  MSE: {mse:.4f}")
        print(f"  Usable Accuracy: {acc:.4f}")

        all_quality_r2.append(r2)
        all_usable_acc.append(acc)

    # 汇总预测结果计算整体 R²
    pred_df = pd.DataFrame(all_predictions)
    overall_r2 = r2_score(pred_df['actual'], pred_df['predicted'])
    overall_mse = mean_squared_error(pred_df['actual'], pred_df['predicted'])
    overall_mae = np.mean(np.abs(pred_df['actual'] - pred_df['predicted']))

    # 总结
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Neural Network:")
    print(f"  Per-fold R²: {np.mean(all_quality_r2):.4f} ± {np.std(all_quality_r2):.4f}")
    print(f"  Overall R² (all predictions): {overall_r2:.4f}")
    print(f"  Overall MSE: {overall_mse:.4f}")
    print(f"  Overall MAE: {overall_mae:.4f}")
    print(f"  Usable Accuracy: {np.mean(all_usable_acc):.4f} ± {np.std(all_usable_acc):.4f}")
    print(f"\nBaseline (Ridge Regression):")
    print(f"  R²: {np.mean(all_baseline_r2):.4f} ± {np.std(all_baseline_r2):.4f}")
    print(f"\nImprovement over baseline: {(np.mean(all_quality_r2) - np.mean(all_baseline_r2)):.4f}")

    # 保存预测结果
    pred_df.to_csv('./predictions.csv', index=False)
    print(f"\nPredictions saved to ./predictions.csv")

    # 训练最终模型
    if overall_r2 > 0:
        print("\nTraining final model on all data...")
        full_dataset = EnhancedPerfusionDataset(df, augment=False)
        full_loader = DataLoader(full_dataset, batch_size=min(8, len(full_dataset)), shuffle=True)

        input_dim = len(full_dataset.raw_feature_cols) * 2
        final_model = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden,
            dropout=args.dropout * 0.5  # 最终模型 dropout 更小
        ).to(device)

        optimizer = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        bce_loss = nn.BCEWithLogitsLoss()

        for epoch in tqdm(range(100), desc="Final training"):
            final_model.train()
            for batch in full_loader:
                features = batch['features'].to(device)
                quality_target = batch['quality_score'].to(device)
                usable_target = batch['usable'].to(device)

                optimizer.zero_grad()
                outputs = final_model(features)
                loss = args.quality_weight * mse_loss(outputs['quality_score'], quality_target)
                loss += (1 - args.quality_weight) * bce_loss(outputs['usable_logit'], usable_target)
                loss.backward()
                optimizer.step()

        # 保存
        os.makedirs('./checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': final_model.state_dict(),
            'feature_cols': full_dataset.raw_feature_cols,
            'mean': full_dataset.mean,
            'std': full_dataset.std,
            'input_dim': input_dim,
            'hidden_dim': args.hidden,
        }, './checkpoints/best_model_v2.pt')

        print(f"\nModel saved to ./checkpoints/best_model_v2.pt")
    else:
        print("\n⚠️ Model performance is poor (R² <= 0). Consider:")
        print("   1. Using data augmentation: --augment")
        print("   2. Checking data quality and label noise")
        print("   3. Using simpler model (baseline Ridge performs at:", f"{np.mean(all_baseline_r2):.4f})")


if __name__ == '__main__':
    main()
