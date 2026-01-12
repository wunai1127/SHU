# Heart Perfusion ML Pipeline

## 心脏移植灌注AI - 数据预处理与模型训练

针对EVHP（Ex Vivo Heart Perfusion，离体心脏灌注）实验数据的机器学习Pipeline。

---

## 数据概览

### EVHP实验数据结构

| 数据文件 | 说明 | 规模 |
|---------|------|------|
| `数据整理 - EVHP原始数据V1.xlsx` | 完整原始数据 | 203行 × 78列 |
| `数据整理 - EVHP数据V1 -计算用.xlsx` | 计算用数据 | 203行 × 73列 |
| `数据整理.xlsx` | 整理后数据 | 多Sheet |

### 数据维度 (5大类 70+特征)

```
1. Blood Gas - Arterial (动脉血气)
   pH, pO2, Na+, K+, Ca++, Cl-, Glu, Lac, tHb, HCO3-act, O2SAT

2. Blood Gas - Venous (静脉血气)
   同上

3. Hemodynamic Parameters (血流动力学)
   MAP (平均动脉压), AoF (主动脉流量), CVR (冠脉血管阻力)

4. Metabolic Parameters (代谢参数)
   MVO2 (心肌氧耗), Lac_Extrac (乳酸提取率), CaO2, CvO2

5. Cardiac Functional Parameters (心脏功能参数)
   ESPVR, EDPVR, PRSW, dP/dtmax, Emax, EF, CO, CI, HR, ...
   共35+参数
```

### 关键监测指标

| 指标 | 临床意义 | 正常范围 |
|------|---------|---------|
| **Lac (乳酸)** | 组织灌注/缺氧指标 | < 2.0 mmol/L |
| **pH** | 酸碱平衡 | 7.35-7.45 |
| **MAP** | 灌注压力 | 60-80 mmHg |
| **EF** | 心脏收缩功能 | > 50% |
| **Lac_Extrac** | 乳酸清除能力 | > 0 |

### 结局变量

- **Success_Wean**: 是否成功脱机 (二分类标签)

---

## Pipeline架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    数据预处理 Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│  1. 数据加载 (Excel)                                            │
│     └── load_excel() → parse_raw_data()                        │
│                                                                 │
│  2. 特征工程                                                    │
│     ├── 派生特征计算 (Lac_delta, Lac_clearance, trends...)     │
│     └── compute_derived_features()                             │
│                                                                 │
│  3. 数据清洗                                                    │
│     ├── 缺失值填充 (KNN / Forward Fill)                        │
│     └── 特征缩放 (StandardScaler)                              │
│                                                                 │
│  4. 数据集构建                                                  │
│     ├── 时序序列 (LSTM输入): create_sequences()                │
│     └── 分类数据集: create_classification_dataset()            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    模型训练 Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│  Task 1: 时序预测 (乳酸/pH变化)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  LSTM + Multi-Head Attention                            │   │
│  │  ├── Bidirectional LSTM (2 layers)                      │   │
│  │  ├── Self-Attention (4 heads)                           │   │
│  │  └── FC Output Layer                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Task 2: 结局分类 (脱机成功预测)                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Ensemble Classifiers                                    │   │
│  │  ├── XGBoost                                            │   │
│  │  ├── Random Forest                                       │   │
│  │  └── Logistic Regression (Baseline)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install pandas numpy scikit-learn torch openpyxl
# 可选: pip install xgboost tensorflow
```

### 2. 运行完整Pipeline

```bash
cd heart_perfusion_ml_pipeline

# 完整训练 (时序预测 + 分类)
python train_pipeline.py \
    --data "../数据整理 - EVHP原始数据V1.xlsx" \
    --output outputs \
    --task both

# 仅时序预测
python train_pipeline.py --data "../data.xlsx" --task prediction

# 仅分类
python train_pipeline.py --data "../data.xlsx" --task classification
```

### 3. 单独使用数据处理器

```python
from preprocessing.evhp_data_processor import EVHPDataProcessor

processor = EVHPDataProcessor(
    impute_method='knn',      # 'knn', 'median', 'forward'
    scale_method='standard'   # 'standard', 'minmax'
)

results = processor.process_pipeline(
    file_path="数据整理 - EVHP原始数据V1.xlsx",
    output_dir="outputs/processed"
)

# 获取处理后的数据
df = results['processed_df']
X_seq, y_seq = results['sequence_data']['X'], results['sequence_data']['y']
```

### 4. 单独训练模型

```python
from models.time_series_model import LSTMAttentionModel, ModelConfig

config = ModelConfig(
    input_dim=20,
    hidden_dim=64,
    num_layers=2,
    seq_length=4,
    epochs=50
)

model = LSTMAttentionModel(config)
history = model.train(X_train, y_train, X_val, y_val)

# 预测 (带注意力权重)
predictions, attention = model.predict(X_test, return_attention=True)
```

---

## 特征工程详解

### 派生特征

| 特征 | 计算方法 | 临床意义 |
|------|---------|---------|
| `Lac_delta` | Lac(t) - Lac(t-1) | 乳酸变化量 |
| `Lac_clearance` | -Lac_delta / Δt | 乳酸清除率 (每小时) |
| `Lac_pct_change` | 乳酸百分比变化 | 相对变化趋势 |
| `O2_extraction` | O2SAT_art - O2SAT_ven | 氧摄取率 |
| `Lac_av_diff` | Lac_art - Lac_ven | 动静脉乳酸差 |
| `Cardiac_Power` | CO × MAP / 451 | 心脏功率指数 |
| `pH_stability` | rolling std(pH) | pH稳定性 |
| `Lac_art_trend` | 线性趋势斜率 | 乳酸变化趋势 |

### 关键特征列表

```python
CRITICAL_FEATURES = [
    # 血气
    'Lac_art', 'pH_art', 'pO2_art', 'O2SAT_art',
    # 血流动力学
    'MAP', 'AoF', 'CVR',
    # 代谢
    'MVO2', 'Lac_Extrac', 'CaO2', 'CvO2',
    # 心功能
    'EF', 'CO', 'CI', 'HR', 'Stroke_Work',
    # 派生
    'Lac_delta', 'Lac_clearance', 'pH_stability',
]
```

---

## 与知识图谱整合

本Pipeline设计用于与graph-rag-agent知识图谱系统整合：

### 知识图谱实体类型

```python
ENTITY_TYPES = [
    "生理指标",    # Lac, pH, MAP等
    "干预措施",    # 加压、减压、药物调整
    "并发症",      # PGD、排斥反应
    "设备参数",    # OCS/XVIVO参数
    "临床指南",    # 中国心脏移植规范
]
```

### 整合流程

```
EVHP时序数据 ──┬──> 特征提取 ──> 实体识别 ──> 知识图谱
               │
               └──> 模型预测 ──> 推理引擎 ──> 决策建议
                                    ↑
                                    │
              PubMed文献 + 临床指南 ─┘
```

---

## 输出文件

```
outputs/
├── processed_data/
│   ├── processed_data.csv    # 处理后的完整数据
│   ├── X_sequences.npy       # 时序输入 (N, seq_len, features)
│   ├── y_sequences.npy       # 时序标签
│   ├── X_classification.npy  # 分类输入
│   └── y_classification.npy  # 分类标签
├── models/
│   ├── lstm_attention/       # 时序模型
│   │   ├── config.json
│   │   └── model.pt
│   └── lstm_training_history.json
└── training_summary.json     # 训练摘要
```

---

## 配置参考

### 数据处理配置

```python
EVHPDataProcessor(
    impute_method='knn',      # 缺失值填充方法
    scale_method='standard',  # 特征缩放方法
)
```

### 模型配置

```python
ModelConfig(
    input_dim=20,             # 输入特征数
    hidden_dim=128,           # LSTM隐藏维度
    num_layers=2,             # LSTM层数
    output_dim=1,             # 输出维度
    dropout=0.2,              # Dropout
    attention_heads=4,        # 注意力头数
    seq_length=4,             # 序列长度 (时间点数)
    learning_rate=1e-3,
    batch_size=32,
    epochs=100,
)
```

---

## TODO

- [ ] 添加因果推理模块
- [ ] 集成神经符号安全约束
- [ ] 添加在线学习能力
- [ ] 开发临床仪表盘可视化

---

## 参考

- 数据来源: EVHP离体心脏灌注实验
- 知识库: PubMed文献 (24,432篇) + 中国心脏移植技术规范(2019)
- 框架: graph-rag-agent多智能体系统
