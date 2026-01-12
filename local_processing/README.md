# Local Triple Processing

本地运行的三元组归一化和多跳关系发现工具。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python run_normalization.py \
    --nodes triples_csv/triple_nodes_heart_tx_pubmed_from_json_without_emb.csv \
    --edges triples_csv/triple_edges_heart_tx_pubmed_from_json_without_emb.csv \
    --output results/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--nodes` | 节点CSV文件路径 | (必填) |
| `--edges` | 边CSV文件路径 | (必填) |
| `--output` | 输出目录 | `output` |
| `--skip-normalization` | 跳过归一化步骤 | False |
| `--skip-multihop` | 跳过多跳关系发现 | False |
| `--string-threshold` | 字符串相似度阈值 | 0.85 |
| `--max-depth` | 多跳最大深度 | 3 |
| `--min-confidence` | 最小置信度 | 0.5 |

### 只运行归一化

```bash
python run_normalization.py \
    --nodes triples_csv/triple_nodes_heart_tx_pubmed_from_json_without_emb.csv \
    --edges triples_csv/triple_edges_heart_tx_pubmed_from_json_without_emb.csv \
    --skip-multihop \
    --output results/
```

### 只运行多跳关系发现

```bash
python run_normalization.py \
    --nodes triples_csv/triple_nodes_heart_tx_pubmed_from_json_without_emb.csv \
    --edges triples_csv/triple_edges_heart_tx_pubmed_from_json_without_emb.csv \
    --skip-normalization \
    --output results/
```

## 输出文件

### 归一化输出 (`results/normalization/`)

- `normalized_nodes.csv` - 归一化后的节点
- `normalized_edges.csv` - 归一化后的边
- `canonical_mapping.csv` - 实体映射表 (原名 → 规范名)

### 多跳关系发现输出 (`results/multihop/`)

- `inferred_relations.csv` - 推断出的新关系
- `path_patterns.csv` - 路径模式统计

## CSV 格式说明

### 输入节点 CSV
```csv
name:ID,type,concepts,synsets,:LABEL
normothermic machine perfusion,entity,[],[],Node
```

### 输入边 CSV
```csv
:START_ID,:END_ID,relation,concepts,synsets,:TYPE
normothermic machine perfusion,donor heart,ORGAN_PRESERVED_WITH,[],[],Relation
```

### 输出推断关系 CSV
```csv
:START_ID,:END_ID,relation,confidence,evidence_path,evidence_relations,:TYPE
entity_a,entity_c,indirectly_causes,0.8,entity_a -> entity_b -> entity_c,causes -> causes,InferredRelation
```

## 关系组合规则

预定义的关系组合规则：

| 关系1 | 关系2 | 推断关系 |
|-------|-------|----------|
| increases_risk | increases_risk | indirectly_increases_risk |
| causes | causes | indirectly_causes |
| treats | prevents | treats_to_prevent |
| part_of | part_of | indirectly_part_of |
| is_a | is_a | is_a |

可以在 `run_normalization.py` 中的 `RELATION_COMPOSITION_RULES` 字典添加自定义规则。

## 文件结构

```
local_processing/
├── README.md
├── requirements.txt
├── run_normalization.py      # 主脚本
└── triples_csv/              # 输入数据
    ├── triple_nodes_*.csv
    ├── triple_edges_*.csv
    └── ...
```
