"""
Temporal Graph Neural Network for Perfusion Monitoring

时序图神经网络模块，用于处理灌注过程中的动态时序数据

Components:
- TemporalEncoder: LSTM-based encoder for time-series features
- GraphEncoder: GraphSAGE-based encoder for knowledge graph structure
- TemporalPerfusionGNN: Combined model for perfusion quality prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class TemporalGNNConfig:
    """Configuration for Temporal Perfusion GNN"""
    # Temporal encoder config
    temporal_input_size: int = 12  # [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α, pressure, flow, temp]
    temporal_hidden_size: int = 64
    temporal_num_layers: int = 2
    temporal_dropout: float = 0.2

    # Graph encoder config
    node_feature_size: int = 64
    graph_hidden_size: int = 128
    graph_num_layers: int = 3

    # Fusion config
    fusion_hidden_size: int = 128
    output_size: int = 1  # Quality score

    # Training config
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder for time-series perfusion data

    Encodes sequences of blood gas, inflammatory markers, and perfusion parameters
    into a fixed-size representation capturing temporal dynamics
    """

    def __init__(self, config: TemporalGNNConfig):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=config.temporal_input_size,
            hidden_size=config.temporal_hidden_size,
            num_layers=config.temporal_num_layers,
            batch_first=True,
            dropout=config.temporal_dropout if config.temporal_num_layers > 1 else 0,
            bidirectional=True
        )

        # Project bidirectional output to hidden size
        # 将双向LSTM输出投影到隐藏维度
        self.projection = nn.Linear(
            config.temporal_hidden_size * 2,
            config.temporal_hidden_size
        )

        # Attention mechanism for time steps
        # 时间步注意力机制，关注重要的时间点
        self.attention = nn.Sequential(
            nn.Linear(config.temporal_hidden_size * 2, config.temporal_hidden_size),
            nn.Tanh(),
            nn.Linear(config.temporal_hidden_size, 1),
            nn.Softmax(dim=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
               Features: [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α,
                         pressure, flow_rate, temperature]
            lengths: Optional sequence lengths for packed sequence

        Returns:
            h_temporal: Aggregated temporal representation [batch_size, hidden_size]
            h_sequence: Full sequence output [batch_size, seq_len, hidden_size*2]
        """
        # LSTM forward
        h_sequence, (h_n, c_n) = self.lstm(x)

        # Attention-weighted aggregation
        # 注意力加权聚合，自动关注异常时间点
        attention_weights = self.attention(h_sequence)
        h_attended = torch.sum(attention_weights * h_sequence, dim=1)

        # Project to hidden size
        h_temporal = self.projection(h_attended)

        return h_temporal, h_sequence


class GraphEncoder(nn.Module):
    """
    GraphSAGE-based encoder for knowledge graph structure

    Encodes the heart features, perfusion strategy, and related concepts
    from the knowledge graph into node embeddings
    """

    def __init__(self, config: TemporalGNNConfig):
        super().__init__()

        self.convs = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(
            config.node_feature_size,
            config.graph_hidden_size
        ))

        # Hidden layers
        for _ in range(config.graph_num_layers - 1):
            self.convs.append(SAGEConv(
                config.graph_hidden_size,
                config.graph_hidden_size
            ))

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Node features [num_nodes, node_feature_size]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment for nodes (for batched graphs)

        Returns:
            h_graph: Graph-level representation [batch_size, hidden_size]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Global pooling
        # 全局池化，获取图级别表示
        if batch is not None:
            h_graph = global_mean_pool(x, batch)
        else:
            h_graph = x.mean(dim=0, keepdim=True)

        return h_graph


class TemporalPerfusionGNN(nn.Module):
    """
    Combined Temporal-Graph Neural Network for Perfusion Monitoring

    Fuses temporal dynamics (blood gas trends) with knowledge graph structure
    (heart features, perfusion strategies, medical concepts) for:
    1. Perfusion quality prediction
    2. Risk level classification
    3. Next state prediction (early warning)
    """

    def __init__(self, config: Optional[TemporalGNNConfig] = None):
        super().__init__()

        self.config = config or TemporalGNNConfig()

        # Sub-encoders
        # 子编码器：时序编码器和图编码器
        self.temporal_encoder = TemporalEncoder(self.config)
        self.graph_encoder = GraphEncoder(self.config)

        # Fusion layer
        # 融合层：合并时序和图表示
        fusion_input_size = (
            self.config.temporal_hidden_size +
            self.config.graph_hidden_size
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, self.config.fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.fusion_hidden_size, self.config.fusion_hidden_size),
            nn.ReLU()
        )

        # Prediction heads
        # 预测头：质量评分、风险分类、下一状态预测

        # Quality score: 0-100
        self.quality_head = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output 0-1, scale to 0-100
        )

        # Risk classification: [low, medium, high, critical]
        self.risk_head = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

        # Next state prediction: predict next measurement values
        self.next_state_head = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.config.temporal_input_size)
        )

        # Intervention flag
        self.intervention_head = nn.Sequential(
            nn.Linear(self.config.fusion_hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        temporal_features: torch.Tensor,
        graph_data: Data,
        temporal_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass combining temporal and graph information

        Args:
            temporal_features: Blood gas + inflammatory markers over time
                Shape: [batch_size, time_steps, num_features]
                Features: [pH, PO2, PCO2, lactate, K+, Na+, IL-6, IL-8, TNF-α,
                          pressure, flow_rate, temperature]

            graph_data: PyG Data object containing:
                - x: Node features [num_nodes, node_feature_size]
                - edge_index: Edge indices [2, num_edges]
                - batch: Batch assignment (optional)

            temporal_lengths: Sequence lengths for variable-length sequences

        Returns:
            Dict containing:
                - quality_score: Predicted perfusion quality (0-100)
                - risk_logits: Risk level logits [batch, 4]
                - risk_probs: Risk level probabilities [batch, 4]
                - next_state: Predicted next measurement values
                - intervention_prob: Probability that intervention is needed
        """
        # 1. Encode temporal sequence
        # 编码时序数据，捕捉血气变化趋势
        h_temporal, _ = self.temporal_encoder(temporal_features, temporal_lengths)

        # 2. Encode graph structure
        # 编码图结构，捕捉医学知识关联
        h_graph = self.graph_encoder(
            graph_data.x,
            graph_data.edge_index,
            graph_data.batch if hasattr(graph_data, 'batch') else None
        )

        # Handle batch size mismatch
        batch_size = temporal_features.size(0)
        if h_graph.size(0) != batch_size:
            h_graph = h_graph.expand(batch_size, -1)

        # 3. Fuse representations
        # 融合时序和图表示
        h_fused = self.fusion(torch.cat([h_temporal, h_graph], dim=-1))

        # 4. Generate predictions
        # 生成预测结果
        quality_score = self.quality_head(h_fused) * 100  # Scale to 0-100
        risk_logits = self.risk_head(h_fused)
        risk_probs = F.softmax(risk_logits, dim=-1)
        next_state = self.next_state_head(h_fused)
        intervention_prob = self.intervention_head(h_fused)

        return {
            'quality_score': quality_score.squeeze(-1),
            'risk_logits': risk_logits,
            'risk_probs': risk_probs,
            'next_state': next_state,
            'intervention_prob': intervention_prob.squeeze(-1)
        }

    def predict_trend(
        self,
        temporal_features: torch.Tensor,
        graph_data: Data,
        num_steps: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Predict multiple future time steps (autoregressive)

        Args:
            temporal_features: Historical measurements
            graph_data: Knowledge graph context
            num_steps: Number of future steps to predict

        Returns:
            Dict with predicted trajectories
        """
        predictions = []
        current_features = temporal_features

        for _ in range(num_steps):
            with torch.no_grad():
                output = self.forward(current_features, graph_data)
                next_state = output['next_state']
                predictions.append(next_state)

                # Append prediction to sequence
                # 将预测追加到序列中，进行下一步预测
                current_features = torch.cat([
                    current_features[:, 1:, :],  # Remove oldest
                    next_state.unsqueeze(1)       # Add newest
                ], dim=1)

        return {
            'predicted_trajectory': torch.stack(predictions, dim=1),
            'final_quality': output['quality_score'],
            'final_risk': output['risk_probs']
        }


class PerfusionDataset(torch.utils.data.Dataset):
    """
    Dataset for perfusion cases with temporal features and graph context
    """

    def __init__(
        self,
        temporal_data: List[np.ndarray],
        graph_data: List[Data],
        labels: List[Dict],
        feature_normalizer: Optional['FeatureNormalizer'] = None
    ):
        """
        Args:
            temporal_data: List of temporal feature arrays [time_steps, features]
            graph_data: List of PyG Data objects (knowledge subgraphs)
            labels: List of label dicts with 'quality_score', 'risk_level', etc.
            feature_normalizer: Optional normalizer for features
        """
        self.temporal_data = temporal_data
        self.graph_data = graph_data
        self.labels = labels
        self.normalizer = feature_normalizer

    def __len__(self):
        return len(self.temporal_data)

    def __getitem__(self, idx):
        temporal = torch.tensor(self.temporal_data[idx], dtype=torch.float32)

        if self.normalizer:
            temporal = self.normalizer.normalize(temporal)

        graph = self.graph_data[idx]
        label = self.labels[idx]

        return {
            'temporal': temporal,
            'graph': graph,
            'quality_score': torch.tensor(label.get('quality_score', 0), dtype=torch.float32),
            'risk_level': torch.tensor(label.get('risk_level', 0), dtype=torch.long),
            'usable': torch.tensor(label.get('usable', True), dtype=torch.bool)
        }


class FeatureNormalizer:
    """
    Normalizer for perfusion features

    Stores mean and std for each feature to normalize inputs
    """

    # Normal ranges for reference
    # 参考的正常范围
    FEATURE_RANGES = {
        'pH': (7.35, 7.45),
        'PO2': (300, 500),
        'PCO2': (35, 45),
        'lactate': (0.5, 2.0),
        'K_plus': (3.5, 5.0),
        'Na_plus': (135, 145),
        'IL_6': (0, 50),
        'IL_8': (0, 50),
        'TNF_alpha': (0, 20),
        'pressure': (40, 80),
        'flow_rate': (1.0, 2.0),
        'temperature': (4, 37)
    }

    FEATURE_ORDER = ['pH', 'PO2', 'PCO2', 'lactate', 'K_plus', 'Na_plus',
                     'IL_6', 'IL_8', 'TNF_alpha', 'pressure', 'flow_rate', 'temperature']

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """
        Fit normalizer to data

        Args:
            data: Array of shape [num_samples, time_steps, features] or [time_steps, features]
        """
        if data.ndim == 3:
            data = data.reshape(-1, data.shape[-1])

        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor"""
        if self.mean is None:
            return x

        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)

        return (x - mean) / std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor"""
        if self.mean is None:
            return x

        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)

        return x * std + mean


def create_dummy_graph(num_nodes: int = 10, feature_size: int = 64) -> Data:
    """
    Create a dummy graph for testing

    Args:
        num_nodes: Number of nodes
        feature_size: Node feature dimension

    Returns:
        PyG Data object
    """
    # Random node features
    x = torch.randn(num_nodes, feature_size)

    # Random edges (ensure connectivity)
    edge_list = []
    for i in range(num_nodes - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    # Add some random edges
    for _ in range(num_nodes):
        i, j = np.random.randint(0, num_nodes, 2)
        if i != j:
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


if __name__ == "__main__":
    # Test the model
    print("Testing TemporalPerfusionGNN...")

    config = TemporalGNNConfig()
    model = TemporalPerfusionGNN(config)

    # Create dummy inputs
    batch_size = 4
    seq_len = 10
    temporal_features = torch.randn(batch_size, seq_len, config.temporal_input_size)
    graph_data = create_dummy_graph()

    # Forward pass
    output = model(temporal_features, graph_data)

    print(f"Quality score shape: {output['quality_score'].shape}")
    print(f"Risk probs shape: {output['risk_probs'].shape}")
    print(f"Next state shape: {output['next_state'].shape}")
    print(f"Intervention prob shape: {output['intervention_prob'].shape}")

    print("\nSample outputs:")
    print(f"Quality scores: {output['quality_score']}")
    print(f"Risk probs: {output['risk_probs'][0]}")
    print(f"Intervention probs: {output['intervention_prob']}")

    # Test trend prediction
    print("\nTesting trend prediction...")
    trend = model.predict_trend(temporal_features, graph_data, num_steps=3)
    print(f"Predicted trajectory shape: {trend['predicted_trajectory'].shape}")

    print("\nAll tests passed!")
