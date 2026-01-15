"""
Perfusion THGNN Model - 心脏灌注时序异构图神经网络

基于 THGNN (Temporal Heterogeneous Graph Neural Network) 架构
原始应用：金融股票预测
适配应用：心脏灌注质量预测

核心思想：
1. GRU/LSTM 编码时序生理指标
2. 异构图注意力处理不同类型的病例关系：
   - 正相关关系：相似的灌注模式
   - 负相关关系：对比/互补的灌注模式
3. 语义注意力融合不同来源的信息
4. 预测质量分数

Reference: THGNN-main (金融场景)
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from dataclasses import dataclass
from typing import Optional


@dataclass
class THGNNConfig:
    """THGNN 配置"""
    # 输入特征
    in_features: int = 10  # 生理指标数量
    seq_length: int = 6    # 时序长度

    # 时序编码器
    temporal_hidden: int = 64
    temporal_layers: int = 1
    temporal_dropout: float = 0.1
    temporal_type: str = 'GRU'  # 'GRU' or 'LSTM'

    # 图注意力
    gat_out_features: int = 16
    gat_num_heads: int = 4
    gat_negative_slope: float = 0.2
    gat_residual: bool = True

    # 语义注意力
    sem_hidden: int = 64

    # 归一化
    pairnorm_mode: str = 'PN-SI'  # 'None', 'PN', 'PN-SI', 'PN-SCS'
    pairnorm_scale: float = 1.0

    # 预测头
    predictor_hidden: int = 32
    dropout: float = 0.2

    # 任务类型
    task: str = 'regression'  # 'regression' or 'classification'


class GraphAttnMultiHead(Module):
    """
    多头图注意力层

    用于聚合图上相邻节点的信息，支持多头注意力机制
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        negative_slope: float = 0.2,
        num_heads: int = 4,
        bias: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # 权重矩阵
        self.weight = Parameter(torch.FloatTensor(in_features, num_heads * out_features))
        self.weight_u = Parameter(torch.FloatTensor(num_heads, out_features, 1))
        self.weight_v = Parameter(torch.FloatTensor(num_heads, out_features, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        # 残差连接
        self.residual = residual
        if self.residual:
            self.project = nn.Linear(in_features, num_heads * out_features)
        else:
            self.project = None

        # 偏置
        if bias:
            self.bias = Parameter(torch.FloatTensor(1, num_heads * out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(-1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.weight_u.size(-1))
        self.weight_u.data.uniform_(-stdv, stdv)
        self.weight_v.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj_mat, requires_weight=False):
        """
        Args:
            inputs: [N, in_features] 节点特征
            adj_mat: [N, N] 邻接矩阵
            requires_weight: 是否返回注意力权重

        Returns:
            support: [N, num_heads * out_features]
            attn_weights: [num_heads, N, N] (if requires_weight)
        """
        # 线性变换
        support = torch.mm(inputs, self.weight)
        support = support.reshape(-1, self.num_heads, self.out_features).permute(dims=(1, 0, 2))

        # 计算注意力分数
        f_1 = torch.matmul(support, self.weight_u).reshape(self.num_heads, 1, -1)
        f_2 = torch.matmul(support, self.weight_v).reshape(self.num_heads, -1, 1)
        logits = f_1 + f_2

        # LeakyReLU 激活
        weight = self.leaky_relu(logits)

        # 应用邻接矩阵掩码并 softmax
        masked_weight = torch.mul(weight, adj_mat)

        # 处理全零行（避免 softmax 问题）
        row_sum = masked_weight.sum(dim=2, keepdim=True)
        row_sum = torch.where(row_sum == 0, torch.ones_like(row_sum), row_sum)
        attn_weights = masked_weight / row_sum

        # 聚合邻居信息
        support = torch.matmul(attn_weights, support)
        support = support.permute(dims=(1, 0, 2)).reshape(-1, self.num_heads * self.out_features)

        # 添加偏置
        if self.bias is not None:
            support = support + self.bias

        # 残差连接
        if self.residual:
            support = support + self.project(inputs)

        if requires_weight:
            return support, attn_weights
        else:
            return support, None


class PairNorm(nn.Module):
    """
    成对归一化层

    防止图神经网络的过平滑问题
    Reference: PairNorm: Tackling Oversmoothing in GNNs (ICLR 2020)
    """

    def __init__(self, mode='PN', scale=1):
        assert mode in ['None', 'PN', 'PN-SI', 'PN-SCS']
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)

        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class SemanticAttention(Module):
    """
    语义级别注意力

    用于融合不同来源的节点表示（自身、正关系、负关系）
    """

    def __init__(self, in_features: int, hidden_size: int = 128, act=nn.Tanh()):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            act,
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, inputs, requires_weight=False):
        """
        Args:
            inputs: [N, num_semantics, features] 多种语义的表示
            requires_weight: 是否返回注意力权重

        Returns:
            output: [N, features] 融合后的表示
            beta: [N, num_semantics, 1] 注意力权重
        """
        w = self.project(inputs)  # [N, num_semantics, 1]
        beta = torch.softmax(w, dim=1)

        output = (beta * inputs).sum(1)

        if requires_weight:
            return output, beta
        else:
            return output, None


class PerfusionTHGNN(nn.Module):
    """
    心脏灌注时序异构图神经网络

    架构：
    1. 时序编码器（GRU/LSTM）：编码生理指标时序变化
    2. 正关系图注意力：聚合相似病例的信息
    3. 负关系图注意力：聚合对比病例的信息
    4. 语义注意力：融合三种表示（自身、正、负）
    5. 预测头：输出质量分数

    Input:
        features: [N, seq_len, in_features] 时序生理指标
        pos_adj: [N, N] 正相关邻接矩阵
        neg_adj: [N, N] 负相关邻接矩阵

    Output:
        quality_score: [N] 质量分数 (0-1)
    """

    def __init__(self, config: Optional[THGNNConfig] = None):
        super().__init__()
        self.config = config or THGNNConfig()
        c = self.config

        # 1. 时序编码器
        if c.temporal_type == 'GRU':
            self.temporal_encoder = nn.GRU(
                input_size=c.in_features,
                hidden_size=c.temporal_hidden,
                num_layers=c.temporal_layers,
                batch_first=True,
                bidirectional=False,
                dropout=c.temporal_dropout if c.temporal_layers > 1 else 0
            )
        else:
            self.temporal_encoder = nn.LSTM(
                input_size=c.in_features,
                hidden_size=c.temporal_hidden,
                num_layers=c.temporal_layers,
                batch_first=True,
                bidirectional=False,
                dropout=c.temporal_dropout if c.temporal_layers > 1 else 0
            )

        # 2. 图注意力层
        gat_out_dim = c.gat_out_features * c.gat_num_heads

        self.pos_gat = GraphAttnMultiHead(
            in_features=c.temporal_hidden,
            out_features=c.gat_out_features,
            num_heads=c.gat_num_heads,
            negative_slope=c.gat_negative_slope,
            residual=c.gat_residual
        )

        self.neg_gat = GraphAttnMultiHead(
            in_features=c.temporal_hidden,
            out_features=c.gat_out_features,
            num_heads=c.gat_num_heads,
            negative_slope=c.gat_negative_slope,
            residual=c.gat_residual
        )

        # 3. 投影层（统一维度）
        self.mlp_self = nn.Linear(c.temporal_hidden, c.temporal_hidden)
        self.mlp_pos = nn.Linear(gat_out_dim, c.temporal_hidden)
        self.mlp_neg = nn.Linear(gat_out_dim, c.temporal_hidden)

        # 4. 成对归一化
        self.pairnorm = PairNorm(mode=c.pairnorm_mode, scale=c.pairnorm_scale)

        # 5. 语义注意力
        self.sem_attn = SemanticAttention(
            in_features=c.temporal_hidden,
            hidden_size=c.sem_hidden,
            act=nn.Tanh()
        )

        # 6. 预测头
        self.predictor = nn.Sequential(
            nn.Linear(c.temporal_hidden, c.predictor_hidden),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.predictor_hidden, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, pos_adj, neg_adj, requires_weight=False):
        """
        前向传播

        Args:
            features: [N, seq_len, in_features] 时序特征
            pos_adj: [N, N] 正关系邻接矩阵
            neg_adj: [N, N] 负关系邻接矩阵
            requires_weight: 是否返回注意力权重

        Returns:
            output: [N] 质量分数
            weights: (pos_attn, neg_attn, sem_attn) if requires_weight
        """
        # 1. 时序编码
        _, h_n = self.temporal_encoder(features)
        if isinstance(h_n, tuple):  # LSTM returns (h_n, c_n)
            h_n = h_n[0]
        support = h_n.squeeze(0)  # [N, temporal_hidden]

        # 2. 图注意力
        pos_support, pos_attn = self.pos_gat(support, pos_adj, requires_weight)
        neg_support, neg_attn = self.neg_gat(support, neg_adj, requires_weight)

        # 3. 投影到相同维度
        self_support = self.mlp_self(support)
        pos_support = self.mlp_pos(pos_support)
        neg_support = self.mlp_neg(neg_support)

        # 4. 堆叠三种语义表示
        all_embedding = torch.stack([self_support, pos_support, neg_support], dim=1)
        # [N, 3, temporal_hidden]

        # 5. 语义注意力融合
        all_embedding, sem_attn = self.sem_attn(all_embedding, requires_weight)
        # [N, temporal_hidden]

        # 6. 成对归一化
        all_embedding = self.pairnorm(all_embedding)

        # 7. 预测
        output = self.predictor(all_embedding).squeeze(-1)

        if requires_weight:
            return output, (pos_attn, neg_attn, sem_attn)
        else:
            return output


class SimplePerfusionTHGNN(nn.Module):
    """
    简化版 THGNN（无图结构）

    当病例数太少（<10）时，图结构可能不稳定
    此版本去掉图注意力，只使用时序编码 + 注意力
    """

    def __init__(self, config: Optional[THGNNConfig] = None):
        super().__init__()
        self.config = config or THGNNConfig()
        c = self.config

        # 时序编码器（双向）
        if c.temporal_type == 'GRU':
            self.temporal_encoder = nn.GRU(
                input_size=c.in_features,
                hidden_size=c.temporal_hidden,
                num_layers=c.temporal_layers,
                batch_first=True,
                bidirectional=True,
                dropout=c.temporal_dropout if c.temporal_layers > 1 else 0
            )
        else:
            self.temporal_encoder = nn.LSTM(
                input_size=c.in_features,
                hidden_size=c.temporal_hidden,
                num_layers=c.temporal_layers,
                batch_first=True,
                bidirectional=True,
                dropout=c.temporal_dropout if c.temporal_layers > 1 else 0
            )

        hidden_dim = c.temporal_hidden * 2  # 双向

        # 时序注意力
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, c.sem_hidden),
            nn.Tanh(),
            nn.Linear(c.sem_hidden, 1)
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, c.predictor_hidden),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.predictor_hidden, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features, pos_adj=None, neg_adj=None, requires_weight=False):
        """
        前向传播（忽略图结构）

        Args:
            features: [N, seq_len, in_features]
            pos_adj, neg_adj: 忽略

        Returns:
            output: [N] 质量分数
        """
        # 时序编码
        output, _ = self.temporal_encoder(features)  # [N, seq_len, hidden*2]

        # 时序注意力
        attn_scores = self.temporal_attn(output)  # [N, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = (attn_weights * output).sum(dim=1)  # [N, hidden*2]

        # 预测
        output = self.predictor(context).squeeze(-1)

        if requires_weight:
            return output, attn_weights
        else:
            return output
