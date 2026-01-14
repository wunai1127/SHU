"""
针对小数据集优化的训练配置

Small Dataset Optimized Training Config
"""

from dataclasses import dataclass


@dataclass
class SmallDataTrainingConfig:
    """
    小数据集训练配置

    关键调整：
    1. 更小的模型防止过拟合
    2. 更强的正则化
    3. 更多的早停耐心
    4. 使用简化的2分类代替4分类
    """

    # 数据划分 - 小数据集用更大的训练集比例
    train_split: float = 0.8      # 80% 训练
    val_split: float = 0.1        # 10% 验证
    test_split: float = 0.1       # 10% 测试

    # 训练超参数
    batch_size: int = 8           # 小batch更稳定
    num_epochs: int = 200         # 更多轮次
    learning_rate: float = 5e-4   # 较小的学习率
    weight_decay: float = 1e-3    # 更强的L2正则化

    # 早停 - 更多耐心
    patience: int = 30
    min_delta: float = 1e-4

    # 模型结构 - 更简单的模型
    hidden_dim: int = 64          # 减小隐藏层 (原128)
    num_layers: int = 1           # 减少层数 (原2)
    dropout: float = 0.5          # 更强的dropout (原0.3)

    # 序列参数
    seq_length: int = 6           # 减少序列长度（数据里每个病例约6-7个时间点）
    num_features: int = 10        # 减少特征数（只用最重要的）

    # 输出
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    device: str = "cuda"
    seed: int = 42


# 简化的特征列（只保留最重要的）
IMPORTANT_FEATURES = [
    'pH',               # 最重要 - 酸碱状态
    'lactate',          # 最重要 - 代谢状态
    'PO2',              # 氧合
    'K_plus',           # 电解质
    'Na_plus',          # 电解质
    'MAP_mmHg',         # 血流动力学
    'AoF_L_min',        # 主动脉流量
    'cardiac_output',   # 心输出量
    'ejection_fraction',# 射血分数
    'heart_rate',       # 心率
]


# 简化为2分类（可用/不可用）而非4分类
USE_BINARY_CLASSIFICATION = True
