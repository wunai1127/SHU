"""
Configuration for Perfusion Module

灌注模块配置文件 - 包含所有敏感信息和超参数

注意：生产环境中应该使用环境变量或密钥管理服务
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Neo4jConfig:
    """Neo4j Knowledge Graph Connection

    知识图谱连接配置
    """
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "wunai1127"
    database: str = "backup"  # HTKG instance


@dataclass
class OpenAIConfig:
    """OpenAI API Configuration

    OpenAI API 配置
    """
    api_key: str = "sk-kzBbW2kbljrrOmp6EAgXcQR1F4cxhTMaCJmfyzZeIY8m1fPu"
    base_url: str = "https://yinli.one/v1"
    model: str = "gpt-4"  # or gpt-3.5-turbo for faster/cheaper
    temperature: float = 0.3  # Lower = more deterministic
    max_tokens: int = 1000


@dataclass
class GNNTrainingConfig:
    """GNN Model Training Configuration

    GNN模型训练配置

    这些是训练超参数，你可能需要根据数据量调整
    """
    # 数据相关
    # Data related
    train_split: float = 0.7      # 70% 训练集
    val_split: float = 0.15       # 15% 验证集
    test_split: float = 0.15      # 15% 测试集

    # 训练超参数
    # Training hyperparameters
    batch_size: int = 16          # 每批样本数，显存不够就减小
    num_epochs: int = 100         # 训练轮数
    learning_rate: float = 1e-3   # 学习率，如果loss不下降可以减小
    weight_decay: float = 1e-5    # L2正则化，防止过拟合

    # 早停策略
    # Early stopping
    patience: int = 15            # 验证集loss连续15轮不下降就停止
    min_delta: float = 1e-4       # 最小改进阈值

    # 模型结构
    # Model architecture
    hidden_dim: int = 128         # 隐藏层维度
    num_layers: int = 2           # LSTM/GNN层数
    dropout: float = 0.3          # Dropout比例，防止过拟合

    # 序列长度
    # Sequence length
    seq_length: int = 20          # 输入时序长度（多少个时间点）
    num_features: int = 12        # 特征数量（pH, PO2, etc.）

    # 输出路径
    # Output paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # 设备
    # Device
    device: str = "cuda"  # "cuda" or "cpu"

    # 随机种子（保证可复现）
    seed: int = 42


@dataclass
class DataConfig:
    """Data Configuration

    数据配置
    """
    # 数据路径
    raw_data_dir: str = "./data/raw"           # 原始数据
    processed_data_dir: str = "./data/processed"  # 处理后数据

    # 特征列名（CSV中的列名）
    feature_columns = [
        'pH',           # 血液酸碱度
        'PO2',          # 氧分压 (mmHg)
        'PCO2',         # 二氧化碳分压 (mmHg)
        'lactate',      # 乳酸 (mmol/L)
        'K_plus',       # 钾离子 (mEq/L)
        'Na_plus',      # 钠离子 (mEq/L)
        'IL_6',         # 白介素-6 (pg/mL)
        'IL_8',         # 白介素-8 (pg/mL)
        'TNF_alpha',    # 肿瘤坏死因子-α (pg/mL)
        'pressure',     # 灌注压力 (mmHg)
        'flow_rate',    # 流量 (L/min)
        'temperature',  # 温度 (°C)
    ]

    # 标签列名
    label_columns = {
        'quality_score': 'quality_score',    # 0-100 质量分数
        'risk_level': 'risk_level',          # low/medium/high/critical
        'usable': 'usable',                  # 0/1 是否可用
    }


# 全局配置实例
# Global config instances
neo4j_config = Neo4jConfig()
openai_config = OpenAIConfig()
training_config = GNNTrainingConfig()
data_config = DataConfig()


def get_full_config():
    """Get all configurations as a dictionary

    获取所有配置的字典形式（用于日志记录）
    """
    return {
        'neo4j': {
            'uri': neo4j_config.uri,
            'user': neo4j_config.user,
            'database': neo4j_config.database,
            # password 不记录
        },
        'openai': {
            'base_url': openai_config.base_url,
            'model': openai_config.model,
            # api_key 不记录
        },
        'training': training_config.__dict__,
        'data': {
            'feature_columns': data_config.feature_columns,
            'label_columns': data_config.label_columns,
        }
    }
