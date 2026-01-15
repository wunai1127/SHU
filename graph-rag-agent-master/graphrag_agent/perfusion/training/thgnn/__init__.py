"""
Temporal Heterogeneous Graph Neural Network (THGNN) for Perfusion Prediction

时序异构图神经网络 - 心脏灌注质量预测

Architecture:
    1. Temporal Encoder: GRU/LSTM 编码时序生理指标
    2. Heterogeneous Graph Attention:
       - Positive relation: 相似灌注模式的病例
       - Negative relation: 对比/互补的灌注模式
    3. Semantic Attention: 融合不同来源的信息
    4. Prediction: 质量分数回归

Reference: THGNN-main (金融场景), adapted for medical perfusion monitoring

Usage:
    # 使用图结构
    python -m graphrag_agent.perfusion.training.thgnn.train \\
        --data perfusion_data.csv --folds 5 --epochs 100

    # 不使用图结构（样本极少时）
    python -m graphrag_agent.perfusion.training.thgnn.train \\
        --data perfusion_data.csv --no-graph --folds -1
"""

from .model import (
    PerfusionTHGNN,
    SimplePerfusionTHGNN,
    THGNNConfig,
    GraphAttnMultiHead,
    PairNorm,
    SemanticAttention,
)

from .data import (
    PerfusionHeteroDataset,
    PerfusionGraphBuilder,
    GraphData,
    build_hetero_graph,
    split_graph_data,
)

from .train import (
    THGNNTrainer,
    train_thgnn,
)

__all__ = [
    # Model
    'PerfusionTHGNN',
    'SimplePerfusionTHGNN',
    'THGNNConfig',
    'GraphAttnMultiHead',
    'PairNorm',
    'SemanticAttention',
    # Data
    'PerfusionHeteroDataset',
    'PerfusionGraphBuilder',
    'GraphData',
    'build_hetero_graph',
    'split_graph_data',
    # Training
    'THGNNTrainer',
    'train_thgnn',
]
