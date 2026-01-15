# THGNN for Perfusion Prediction

时序异构图神经网络（Temporal Heterogeneous GNN）用于心脏灌注质量预测。

## 架构

基于 THGNN-main（金融场景）适配到医学灌注监测：

```
输入数据
    │
    ▼
┌─────────────────┐
│ GRU/LSTM 编码器  │  ← 编码时序生理指标 (pH, lactate, PO2...)
└────────┬────────┘
         │
    ┌────┴────┬────────┐
    ▼         ▼        ▼
┌───────┐ ┌───────┐ ┌───────┐
│ Self  │ │Pos GAT│ │Neg GAT│  ← 异构图注意力
└───┬───┘ └───┬───┘ └───┬───┘
    │         │        │
    └────┬────┴────────┘
         ▼
┌─────────────────┐
│  语义注意力融合   │  ← 融合不同来源的信息
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    PairNorm     │  ← 防止过平滑
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     预测头      │  → quality_score (0-1)
└─────────────────┘
```

### 图构建方式

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `correlation` | 基于时序 Pearson 相关性 | 默认，样本 > 20 |
| `knn` | K 近邻图 | 中等样本量 |
| `full` | 全连接图 | 样本极少 < 10 |

### 关系类型

- **正关系 (pos_adj)**: 相似的灌注模式（相关系数 > 0.3）
- **负关系 (neg_adj)**: 对比/互补的模式（相关系数 < -0.3）

## 使用方法

### 1. 准备数据

数据 CSV 格式：
```csv
case_id,timestamp,pH,lactate,PO2,K_plus,Na_plus,MAP_mmHg,AoF_L_min,cardiac_output,ejection_fraction,heart_rate,quality_score
case_1,0,7.35,2.1,150,...,75
case_1,1,7.38,1.9,155,...,75
...
```

### 2. 训练模型

```bash
# 基本用法（使用图结构）
python -m graphrag_agent.perfusion.training.thgnn.train \
    --data perfusion_data.csv \
    --folds 5 \
    --epochs 100

# 使用 KNN 图
python -m graphrag_agent.perfusion.training.thgnn.train \
    --data perfusion_data.csv \
    --graph-method knn \
    --folds 5

# 不使用图结构（样本极少时推荐）
python -m graphrag_agent.perfusion.training.thgnn.train \
    --data perfusion_data.csv \
    --no-graph \
    --folds -1  # Leave-One-Out

# 调整超参数
python -m graphrag_agent.perfusion.training.thgnn.train \
    --data perfusion_data.csv \
    --hidden 32 \
    --lr 5e-4 \
    --epochs 150
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必填) | CSV 数据路径 |
| `--output` | `./checkpoints` | 输出目录 |
| `--folds` | 5 | K-Fold 数量，-1 表示 LOO |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-3 | 学习率 |
| `--hidden` | 64 | 隐藏层维度 |
| `--no-graph` | False | 禁用图结构 |
| `--graph-method` | `correlation` | 图构建方法 |
| `--device` | `cpu` | 设备 |
| `--seed` | 42 | 随机种子 |

### 4. 输出文件

```
checkpoints/
├── predictions.csv      # 交叉验证预测结果
└── thgnn_model.pt       # 最终模型
```

## Python API

```python
from graphrag_agent.perfusion.training.thgnn import (
    train_thgnn,
    PerfusionTHGNN,
    THGNNConfig,
    PerfusionHeteroDataset,
)

# 方式1: 使用便捷函数
results = train_thgnn(
    csv_path='perfusion_data.csv',
    n_folds=5,
    epochs=100,
    use_graph=True,
)
print(f"Overall R²: {results['overall_r2']:.4f}")

# 方式2: 手动控制
config = THGNNConfig(
    in_features=10,
    temporal_hidden=64,
    gat_num_heads=4,
)

dataset = PerfusionHeteroDataset('data.csv')
graph_data = dataset.get_graph_data()

model = PerfusionTHGNN(config)
output = model(graph_data.features, graph_data.pos_adj, graph_data.neg_adj)
```

## 与原始 THGNN 的对比

| 方面 | 原始 THGNN (金融) | 本实现 (灌注) |
|------|------------------|--------------|
| 节点 | 股票 | 病例 |
| 时序特征 | OHLC 价格 | 生理指标 |
| 正关系 | 股票正相关 | 相似灌注模式 |
| 负关系 | 股票负相关 | 对比灌注模式 |
| 任务 | 涨跌预测 | 质量分数回归 |
| 图更新 | 每月更新 | 静态（小样本） |

## 小样本优化

对于 31 例 EVHP 数据的特殊处理：

1. **图稀疏性**: 调低阈值（0.3 → 0.2）增加边数量
2. **Leave-One-Out**: 使用 `--folds -1`
3. **禁用图**: 样本 < 10 时使用 `--no-graph`
4. **数据增强**: 在数据预处理时添加噪声

## 参考

- [THGNN-main](https://github.com/xxx/THGNN-main) - 原始金融场景实现
- PairNorm: Tackling Oversmoothing in GNNs (ICLR 2020)
