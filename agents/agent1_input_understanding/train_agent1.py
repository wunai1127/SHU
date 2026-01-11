"""
Agent 1 训练脚本
用于微调ClinicalBERT和训练LSTM编码器

训练数据格式:
- 心脏描述文本 + 标注的特征 (hypertrophy, contractility等)
- 血气时序数据 + 对应的标签
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
from agent1_core import (
    ClinicalTextEncoder,
    BloodGasLSTMEncoder,
    InputUnderstandingAgent
)


class CardiacTextDataset(Dataset):
    """心脏描述文本数据集"""
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'text': item['text'],
            'hypertrophy': item['labels']['hypertrophy'],
            'contractility': item['labels']['contractility'],
            'valve_status': item['labels']['valve_status'],  # 0: good, 1: moderate, 2: poor
            'scarring': item['labels']['scarring'],
            'coronary_patency': item['labels']['coronary_patency']
        }


class BloodGasDataset(Dataset):
    """血气时序数据集"""
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 提取时序序列
        sequence = []
        for tp in item['sequence']:
            sequence.append([
                tp['lactate'],
                tp['pH'],
                tp['pO2'],
                tp['pCO2'],
                tp['K+'],
                tp['glucose']
            ])

        return {
            'sequence': torch.tensor(sequence, dtype=torch.float32),
            'outcome_score': item['outcome_score']  # 0-1
        }


class TextEncoderTrainer:
    """ClinicalBERT微调训练器"""
    def __init__(self,
                 model: ClinicalTextEncoder,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        # 优化器（只微调fine_tune_layer和feature_extractors）
        params_to_update = []
        for name, param in model.named_parameters():
            if 'bert' not in name:  # 不更新BERT参数
                params_to_update.append(param)

        self.optimizer = optim.AdamW(params_to_update, lr=1e-4, weight_decay=0.01)

        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training"):
            self.optimizer.zero_grad()

            # 前向传播
            texts = batch['text']
            losses = []

            for i, text in enumerate(texts):
                embedding, features = self.model(text)

                # 计算各特征的损失
                # Hypertrophy
                hypertrophy_pred = torch.sigmoid(
                    self.model.feature_extractors['hypertrophy'](embedding.unsqueeze(0))
                )
                hypertrophy_loss = self.mse_loss(
                    hypertrophy_pred,
                    batch['hypertrophy'][i].unsqueeze(0).unsqueeze(1).to(self.device)
                )

                # Contractility
                contractility_pred = torch.sigmoid(
                    self.model.feature_extractors['contractility'](embedding.unsqueeze(0))
                )
                contractility_loss = self.mse_loss(
                    contractility_pred,
                    batch['contractility'][i].unsqueeze(0).unsqueeze(1).to(self.device)
                )

                # Valve status (分类)
                valve_logits = self.model.feature_extractors['valve_status'](
                    embedding.unsqueeze(0)
                )
                valve_loss = self.ce_loss(
                    valve_logits,
                    batch['valve_status'][i].unsqueeze(0).to(self.device)
                )

                # Scarring
                scarring_pred = torch.sigmoid(
                    self.model.feature_extractors['scarring'](embedding.unsqueeze(0))
                )
                scarring_loss = self.mse_loss(
                    scarring_pred,
                    batch['scarring'][i].unsqueeze(0).unsqueeze(1).to(self.device)
                )

                # Coronary patency
                coronary_pred = torch.sigmoid(
                    self.model.feature_extractors['coronary_patency'](embedding.unsqueeze(0))
                )
                coronary_loss = self.mse_loss(
                    coronary_pred,
                    batch['coronary_patency'][i].unsqueeze(0).unsqueeze(1).to(self.device)
                )

                # 总损失
                loss = (hypertrophy_loss + contractility_loss + valve_loss +
                       scarring_loss + coronary_loss) / 5

                losses.append(loss)

            # 批次平均损失
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item()

        return total_loss / len(dataloader)

    def save(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to {path}")


class LSTMEncoderTrainer:
    """LSTM编码器训练器"""
    def __init__(self,
                 model: BloodGasLSTMEncoder,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device

        self.optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        # 预测头（用于训练）
        self.predictor = nn.Linear(256, 1).to(device)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc="Training LSTM"):
            self.optimizer.zero_grad()

            sequences = batch['sequence'].to(self.device)  # [batch, time, 6]
            outcomes = batch['outcome_score'].to(self.device)  # [batch]

            # 前向传播
            embeddings, attn_weights = self.model(sequences)

            # 预测结局
            predictions = self.predictor(embeddings).squeeze()

            # 损失
            loss = self.criterion(predictions, outcomes)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'encoder': self.model.state_dict(),
            'predictor': self.predictor.state_dict()
        }, path)
        print(f"✅ LSTM model saved to {path}")


def train_text_encoder(data_path: str,
                       output_dir: str,
                       epochs: int = 5,
                       batch_size: int = 16):
    """
    训练文本编码器

    Args:
        data_path: 训练数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print("=" * 60)
    print("训练 ClinicalBERT 文本编码器")
    print("=" * 60)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    dataset = CardiacTextDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型和训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ClinicalTextEncoder()
    trainer = TextEncoderTrainer(model, device)

    # 训练
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        loss = trainer.train_epoch(dataloader)
        print(f"   Loss: {loss:.4f}")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            trainer.save(f"{output_dir}/text_encoder_best.pth")

    print(f"\n✅ 训练完成！最佳Loss: {best_loss:.4f}")


def train_lstm_encoder(data_path: str,
                       output_dir: str,
                       epochs: int = 20,
                       batch_size: int = 32):
    """
    训练LSTM编码器

    Args:
        data_path: 训练数据路径
        output_dir: 输出目录
        epochs: 训练轮数
        batch_size: 批次大小
    """
    print("=" * 60)
    print("训练 LSTM 时序编码器")
    print("=" * 60)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 加载数据
    dataset = BloodGasDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型和训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BloodGasLSTMEncoder()
    trainer = LSTMEncoderTrainer(model, device)

    # 训练
    best_loss = float('inf')
    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")
        loss = trainer.train_epoch(dataloader)
        print(f"   Loss: {loss:.4f}")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            trainer.save(f"{output_dir}/lstm_encoder_best.pth")

    print(f"\n✅ 训练完成！最佳Loss: {best_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Agent 1 components")
    parser.add_argument('--component', type=str, required=True,
                       choices=['text', 'lstm', 'both'],
                       help='训练组件: text (文本编码器) / lstm (时序编码器) / both (两者)')
    parser.add_argument('--text_data', type=str,
                       default='data/cardiac_text_train.json',
                       help='文本训练数据路径')
    parser.add_argument('--lstm_data', type=str,
                       default='data/blood_gas_train.json',
                       help='时序训练数据路径')
    parser.add_argument('--output_dir', type=str,
                       default='checkpoints',
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')

    args = parser.parse_args()

    if args.component in ['text', 'both']:
        train_text_encoder(
            data_path=args.text_data,
            output_dir=args.output_dir,
            epochs=args.epochs
        )

    if args.component in ['lstm', 'both']:
        train_lstm_encoder(
            data_path=args.lstm_data,
            output_dir=args.output_dir,
            epochs=args.epochs if args.component == 'lstm' else 20
        )
