# GNN 训练指南 / GNN Training Guide

## 目录
1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [开始训练](#开始训练)
4. [查看训练结果](#查看训练结果)
5. [使用训练好的模型](#使用训练好的模型)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 进入项目目录
cd /home/user/SHU/graph-rag-agent-master

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pandas numpy scikit-learn tqdm tensorboard
pip install neo4j openai
```

### 2. 检查 GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

如果没有 GPU，训练会自动使用 CPU（但会慢很多）。

---

## 数据准备

### 数据格式要求

CSV 文件必须包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `case_id` | string | 病例唯一标识 |
| `timestamp` | int | 时间点编号（0, 1, 2, ...） |
| `pH` | float | 血液酸碱度 (7.2-7.5) |
| `PO2` | float | 氧分压 (mmHg) |
| `PCO2` | float | 二氧化碳分压 (mmHg) |
| `lactate` | float | 乳酸 (mmol/L) |
| `K_plus` | float | 钾离子 (mEq/L) |
| `Na_plus` | float | 钠离子 (mEq/L) |
| `IL_6` | float | 白介素-6 (pg/mL) |
| `IL_8` | float | 白介素-8 (pg/mL) |
| `TNF_alpha` | float | 肿瘤坏死因子-α (pg/mL) |
| `pressure` | float | 灌注压力 (mmHg) |
| `flow_rate` | float | 流量 (L/min) |
| `temperature` | float | 温度 (°C) |
| `quality_score` | float | **标签**: 质量分数 (0-100) |
| `risk_level` | string | **标签**: 风险等级 (low/medium/high/critical) |
| `usable` | int | **标签**: 是否可用 (0/1) |

### 示例数据

```csv
case_id,timestamp,pH,PO2,PCO2,lactate,K_plus,Na_plus,IL_6,IL_8,TNF_alpha,pressure,flow_rate,temperature,quality_score,risk_level,usable
CASE001,0,7.40,400,40,1.5,4.0,140,10,8,5,60,1.5,34,85,low,1
CASE001,1,7.38,380,42,1.8,4.1,139,12,10,6,58,1.5,34,85,low,1
CASE001,2,7.36,360,44,2.2,4.2,138,15,12,8,56,1.4,35,85,low,1
...
```

### 数据放置位置

```
graphrag_agent/perfusion/training/
├── data/
│   ├── raw/                    # 原始数据
│   │   └── perfusion_data.csv  # 你的数据放这里
│   └── processed/              # 处理后数据（自动生成）
├── checkpoints/                # 模型检查点（自动生成）
└── logs/                       # TensorBoard日志（自动生成）
```

---

## 开始训练

### 方法 1：使用模拟数据测试（推荐先做这一步）

```bash
cd /home/user/SHU/graph-rag-agent-master/graphrag_agent/perfusion/training

# 创建模拟数据并训练（测试整个流程）
python train_gnn.py --create_synthetic --num_cases 200 --epochs 50
```

这会：
1. 生成 200 个模拟病例数据
2. 训练 50 轮
3. 保存模型到 `checkpoints/best_model.pt`

### 方法 2：使用真实数据训练

```bash
# 假设你的数据在 data/raw/perfusion_data.csv
python train_gnn.py --data ./data/raw/perfusion_data.csv --epochs 100
```

### 训练参数说明

```bash
python train_gnn.py \
    --data ./data/perfusion_data.csv \  # 数据文件路径
    --epochs 100 \                       # 训练轮数
    --batch_size 16 \                    # 批次大小（显存不够就减小）
    --lr 0.001 \                         # 学习率
    --seq_length 20 \                    # 序列长度（每个样本多少时间点）
    --device cuda \                      # 使用GPU (cuda) 或 CPU (cpu)
    --seed 42                            # 随机种子（保证可复现）
```

### 从中断处恢复训练

```bash
python train_gnn.py --resume ./checkpoints/latest_checkpoint.pt
```

---

## 查看训练结果

### 1. TensorBoard（推荐）

```bash
# 启动 TensorBoard
tensorboard --logdir=./logs --port 6006

# 然后在浏览器打开 http://localhost:6006
```

TensorBoard 会显示：
- **Loss/train**: 训练损失曲线
- **Loss/val**: 验证损失曲线
- **Metrics/quality_r2**: 质量分数 R² 指标
- **Metrics/risk_accuracy**: 风险分类准确率
- **LR**: 学习率变化

### 2. 查看训练历史

```bash
# 训练完成后会生成 JSON 文件
cat ./checkpoints/training_history.json
```

### 3. 关键指标解读

| 指标 | 含义 | 好的值 |
|------|------|--------|
| `quality_mse` | 质量分数均方误差 | < 100 |
| `quality_r2` | 质量分数 R²（越接近1越好） | > 0.7 |
| `risk_accuracy` | 风险分类准确率 | > 0.8 |
| `risk_f1` | 风险分类 F1 分数 | > 0.75 |

---

## 使用训练好的模型

### 1. 更新配置指向模型文件

编辑 `config.py`：

```python
@dataclass
class GNNTrainingConfig:
    ...
    checkpoint_dir: str = "./checkpoints"  # 模型保存位置
```

### 2. 在预测中使用

```python
from graphrag_agent.perfusion import PerfusionOutcomePredictor, PredictionConfig

# 指定模型路径
config = PredictionConfig(
    gnn_model_path="./graphrag_agent/perfusion/training/checkpoints/best_model.pt",
    enable_gnn=True,
    enable_llm=True,
    enable_kg=True,
)

predictor = PerfusionOutcomePredictor(config)

# 使用真实数据预测
outcome = predictor.predict(
    temporal_data=your_perfusion_data,  # numpy array [time_steps, 12]
    text_report="Donor information...",
    case_id="REAL-001"
)

print(f"Usability Score: {outcome.usability_score:.1f}")
print(f"Risk Level: {outcome.risk_level}")
```

---

## 常见问题

### Q1: CUDA out of memory

**解决方案**：减小 batch_size

```bash
python train_gnn.py --batch_size 8
# 或更小
python train_gnn.py --batch_size 4
```

### Q2: Loss 不下降

**解决方案**：
1. 减小学习率：`--lr 0.0001`
2. 增加数据量
3. 检查数据是否有问题（NaN 值等）

### Q3: 过拟合（训练 loss 下降但验证 loss 上升）

**解决方案**：
1. 增加 dropout（在 `config.py` 中修改）
2. 增加 weight_decay
3. 早停（已自动实现）

### Q4: 没有 GPU

训练会自动使用 CPU，但速度会慢 10-50 倍。建议：
1. 减少 epochs：`--epochs 30`
2. 减少数据量
3. 使用云 GPU（Google Colab、AWS 等）

### Q5: 如何知道训练够了？

看 TensorBoard 中的曲线：
- 训练 loss 和验证 loss 都趋于平稳
- 验证 loss 不再下降（早停会自动触发）

---

## 完整训练流程示例

```bash
# 1. 进入目录
cd /home/user/SHU/graph-rag-agent-master/graphrag_agent/perfusion/training

# 2. 先用模拟数据测试
python train_gnn.py --create_synthetic --epochs 30

# 3. 启动 TensorBoard 查看
tensorboard --logdir=./logs &

# 4. 准备好真实数据后
python train_gnn.py --data ./data/raw/your_data.csv --epochs 100

# 5. 训练完成后，模型在
ls -la ./checkpoints/
# best_model.pt  <- 最佳模型
# latest_checkpoint.pt  <- 最新检查点
```

---

## 联系方式

如有问题，请检查日志文件或 TensorBoard 输出。
