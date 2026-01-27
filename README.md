# HTTG Perfusion Monitoring System

心脏移植灌注监测系统 - 证据驱动的策略推荐引擎

## 功能

1. **Baseline阈值检测** - 检测指标异常并分级
2. **证据驱动策略推荐** - 基于知识图谱和临床指南的干预建议
3. **LLM增强推理** - 可选的大语言模型增强推理
4. **Robustness检查** - 一致性和药物交互检查

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 🚀 启动Web前端（推荐）
streamlit run app.py

# 运行基础测试（无需Neo4j）
python test_evidence_strategy.py

# 运行完整策略测试
python test_full_strategy.py

# 运行Baseline策略推荐测试
python src/baseline_strategy_recommender.py
```

## Web前端功能

启动 `streamlit run app.py` 后访问 http://localhost:8501

- **📊 监测总览** - 实时指标状态卡片、时序趋势图
- **💊 策略推荐** - 干预建议+CoT推理链+证据溯源
- **📊 数据管理** - 上传/管理患者数据
- **⚙️ 后端配置** - Neo4j、LLM、阈值参数设置
- **📈 批量分析** - 多样本对比、成功/失败组统计

## Neo4j集成（可选）

```python
from src.neo4j_connector import Neo4jKnowledgeGraph
from src.baseline_strategy_recommender import BaselineStrategyRecommender

# 连接Neo4j
neo4j = Neo4jKnowledgeGraph(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your_password",
    database="backup"
)

# 创建推荐器并设置Neo4j
recommender = BaselineStrategyRecommender()
recommender.set_neo4j(neo4j)

# 分析数据
measurements = {'MAP': 45, 'Lactate': 4.5, 'K_A': 6.2}
report = recommender.analyze_baseline(measurements, sample_id="TEST-001")
print(recommender.format_report(report))
```

## LLM集成（可选）

```python
from src.baseline_strategy_recommender import BaselineStrategyRecommender, OpenAILLM

# 使用OpenAI
llm = OpenAILLM(api_key="your_api_key", model="gpt-4")
recommender = BaselineStrategyRecommender(llm=llm)

# 或使用Anthropic Claude
from src.baseline_strategy_recommender import AnthropicLLM
llm = AnthropicLLM(api_key="your_api_key")
recommender.set_llm(llm)
```

## 文件结构

```
├── app.py                                 # 🚀 Streamlit主应用入口
├── pages/
│   ├── 1_📊_数据管理.py                   # 数据上传/管理页面
│   ├── 2_⚙️_后端配置.py                   # Neo4j/LLM配置页面
│   └── 3_📈_批量分析.py                   # 批量分析页面
├── src/
│   ├── baseline_strategy_recommender.py  # 主推荐器
│   ├── baseline_thresholds.py            # 阈值管理
│   ├── evidence_strategy_engine.py       # 证据驱动引擎
│   ├── neo4j_connector.py                # Neo4j连接器
│   ├── perfusion_monitor.py              # 灌注监测器
│   └── ...
├── config/
│   ├── thresholds.yaml                   # 阈值配置
│   ├── intervention_strategies.yaml      # 干预策略
│   ├── baseline.yaml                     # Baseline配置
│   └── strategies.yaml                   # 策略映射
├── test_*.py                             # 测试脚本
└── requirements.txt
```

## 支持的指标

| 指标 | 单位 | 目标范围 | 干预策略 |
|------|------|----------|----------|
| MAP | mmHg | 65-80 | 血管活性药物支持 |
| CI | L/min/m² | 2.2-2.8 | 正性肌力药物 |
| Lactate | mmol/L | <4 | 优化灌注 |
| K_A | mmol/L | 4.0-5.0 | 补钾/降钾 |
| SvO2 | % | 65-80 | 氧供需平衡 |
| pH | - | 7.35-7.45 | 酸碱纠正 |
| ... | ... | ... | ... |

## 版本

- v2.0.0 - 完整的证据驱动策略推荐系统
