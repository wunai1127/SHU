#!/usr/bin/env python3
"""
小数据集专用训练脚本

针对32个病例的EVHP数据优化：
1. 数据增强
2. 简化模型
3. 二分类（可用/不可用）
4. 交叉验证
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
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from tqdm import tqdm

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)


class SimplePerfusionDataset(Dataset):
    """简化的数据集"""

    def __init__(self, df, seq_length=6, feature_cols=None):
        self.seq_length = seq_length
        self.feature_cols = feature_cols or [
            'pH', 'lactate', 'PO2', 'K_plus', 'Na_plus',
            'MAP_mmHg', 'AoF_L_min', 'cardiac_output',
            'ejection_fraction', 'heart_rate'
        ]

        # 只保留存在的特征列
        self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        print(f"Using features: {self.feature_cols}")

        self.samples = self._build_samples(df)

        # 计算归一化参数
        all_features = np.vstack([s['features'] for s in self.samples])
        self.mean = np.nanmean(all_features, axis=0)
        self.std = np.nanstd(all_features, axis=0) + 1e-8

    def _build_samples(self, df):
        samples = []
        for case_id, group in df.groupby('case_id'):
            group = group.sort_values('timestamp')
            features = group[self.feature_cols].values.astype(np.float32)

            # 填充NaN
            features = np.nan_to_num(features, nan=0.0)

            # 如果太短就填充
            if len(features) < self.seq_length:
                pad = np.repeat(features[-1:], self.seq_length - len(features), axis=0)
                features = np.vstack([features, pad])
            elif len(features) > self.seq_length:
                features = features[:self.seq_length]

            # 获取标签
            quality = group['quality_score'].iloc[-1]
            usable = group['usable'].iloc[-1] if 'usable' in group.columns else (1 if quality >= 50 else 0)

            samples.append({
                'case_id': case_id,
                'features': features,
                'quality_score': quality / 100.0,  # 归一化到0-1
                'usable': usable,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        features = (s['features'] - self.mean) / self.std
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'quality_score': torch.tensor(s['quality_score'], dtype=torch.float32),
            'usable': torch.tensor(s['usable'], dtype=torch.float32),
        }


class SimpleGNN(nn.Module):
    """简化的模型 - 只用LSTM，不用图"""

    def __init__(self, input_dim=10, hidden_dim=64, dropout=0.5):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # 质量分数预测头
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

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后时刻的输出
        last_output = lstm_out[:, -1, :]  # [batch, hidden*2]
        last_output = self.dropout(last_output)

        quality = self.quality_head(last_output).squeeze(-1)
        usable_logit = self.usable_head(last_output).squeeze(-1)

        return {
            'quality_score': quality,
            'usable_logit': usable_logit,
        }


def train_fold(model, train_loader, val_loader, device, epochs=100, lr=5e-4):
    """训练一个fold"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0

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

            loss = mse_loss(outputs['quality_score'], quality_target)
            loss += bce_loss(outputs['usable_logit'], usable_target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        val_quality_preds, val_quality_targets = [], []
        val_usable_preds, val_usable_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                quality_target = batch['quality_score'].to(device)
                usable_target = batch['usable'].to(device)

                outputs = model(features)

                loss = mse_loss(outputs['quality_score'], quality_target)
                loss += bce_loss(outputs['usable_logit'], usable_target)
                val_loss += loss.item()

                val_quality_preds.extend(outputs['quality_score'].cpu().numpy())
                val_quality_targets.extend(quality_target.cpu().numpy())
                val_usable_preds.extend((torch.sigmoid(outputs['usable_logit']) > 0.5).float().cpu().numpy())
                val_usable_targets.extend(usable_target.cpu().numpy())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # 计算指标
        val_quality_preds = np.array(val_quality_preds) * 100
        val_quality_targets = np.array(val_quality_targets) * 100
        r2 = r2_score(val_quality_targets, val_quality_preds)
        acc = accuracy_score(val_usable_targets, val_usable_preds)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: loss={val_loss:.4f}, R²={r2:.4f}, Acc={acc:.4f}")

        # 早停
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= 30:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='CSV data path')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--folds', type=int, default=5, help='K-fold cross validation')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Total rows: {len(df)}, Cases: {df['case_id'].nunique()}")

    # 获取唯一的病例ID
    case_ids = df['case_id'].unique()
    print(f"Unique cases: {len(case_ids)}")

    # K-Fold 交叉验证
    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=42)

    all_quality_r2 = []
    all_usable_acc = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(case_ids)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{args.folds}")
        print(f"{'='*50}")

        train_cases = case_ids[train_idx]
        val_cases = case_ids[val_idx]

        train_df = df[df['case_id'].isin(train_cases)]
        val_df = df[df['case_id'].isin(val_cases)]

        train_dataset = SimplePerfusionDataset(train_df)
        val_dataset = SimplePerfusionDataset(val_df)

        # 使用训练集的归一化参数
        val_dataset.mean = train_dataset.mean
        val_dataset.std = train_dataset.std

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # 创建模型
        model = SimpleGNN(
            input_dim=len(train_dataset.feature_cols),
            hidden_dim=64,
            dropout=0.5
        ).to(device)

        # 训练
        model, best_loss = train_fold(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=5e-4
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

        r2 = r2_score(quality_targets, quality_preds)
        mse = mean_squared_error(quality_targets, quality_preds)
        acc = accuracy_score(usable_targets, usable_preds)
        f1 = f1_score(usable_targets, usable_preds)

        print(f"\nFold {fold + 1} Results:")
        print(f"  Quality R²: {r2:.4f}")
        print(f"  Quality MSE: {mse:.4f}")
        print(f"  Usable Accuracy: {acc:.4f}")
        print(f"  Usable F1: {f1:.4f}")

        all_quality_r2.append(r2)
        all_usable_acc.append(acc)

    # 总结
    print(f"\n{'='*60}")
    print("Cross-Validation Summary")
    print(f"{'='*60}")
    print(f"Quality R²: {np.mean(all_quality_r2):.4f} ± {np.std(all_quality_r2):.4f}")
    print(f"Usable Accuracy: {np.mean(all_usable_acc):.4f} ± {np.std(all_usable_acc):.4f}")

    # 保存最终模型（在全部数据上训练）
    print("\nTraining final model on all data...")
    full_dataset = SimplePerfusionDataset(df)
    full_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)

    final_model = SimpleGNN(
        input_dim=len(full_dataset.feature_cols),
        hidden_dim=64,
        dropout=0.3  # 最终模型dropout小一点
    ).to(device)

    optimizer = optim.AdamW(final_model.parameters(), lr=5e-4, weight_decay=1e-3)
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
            loss = mse_loss(outputs['quality_score'], quality_target)
            loss += bce_loss(outputs['usable_logit'], usable_target)
            loss.backward()
            optimizer.step()

    # 保存
    os.makedirs('./checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'feature_cols': full_dataset.feature_cols,
        'mean': full_dataset.mean,
        'std': full_dataset.std,
    }, './checkpoints/best_model.pt')

    print(f"\nModel saved to ./checkpoints/best_model.pt")


if __name__ == '__main__':
    main()
