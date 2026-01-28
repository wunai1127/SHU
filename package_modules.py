#!/usr/bin/env python3
"""
打包本地运行所需的所有模块
"""

import os
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 需要打包的文件和目录
INCLUDE_PATTERNS = {
    # 核心源码
    'src/': [
        'baseline_strategy_recommender.py',
        'baseline_thresholds.py',
        'evidence_strategy_engine.py',
        'neo4j_connector.py',
        'perfusion_monitor.py',
        'threshold_manager.py',
        'baseline_evaluator.py',
        'knowledge_graph.py',
        '__init__.py',
    ],
    # 配置文件
    'config/': [
        'thresholds.yaml',
        'baseline.yaml',
        'intervention_strategies.yaml',
        'strategies.yaml',
    ],
    # 前端页面
    'pages/': [
        '1_📊_数据管理.py',
        '2_⚙️_后端配置.py',
        '3_📈_批量分析.py',
        '4_🎙️_语音交互.py',
    ],
    # 测试脚本
    '': [
        'test_perfusion_monitor.py',
        'test_full_strategy.py',
        'test_evidence_strategy.py',
        'app.py',  # Streamlit主应用
    ],
}

# 额外需要包含的文件
EXTRA_FILES = [
    'requirements.txt',
    'README.md',
]


def create_requirements():
    """创建requirements.txt"""
    requirements = """# HTTG Perfusion Monitoring System Requirements

# Core dependencies
pyyaml>=6.0
pandas>=1.5.0
openpyxl>=3.0.0
numpy>=1.21.0

# Neo4j integration (optional)
neo4j>=5.0.0

# LLM integration (optional)
openai>=1.0.0
anthropic>=0.18.0

# Visualization (optional)
matplotlib>=3.5.0
seaborn>=0.12.0
"""
    req_path = PROJECT_ROOT / 'requirements.txt'
    with open(req_path, 'w', encoding='utf-8') as f:
        f.write(requirements)
    print(f"Created: {req_path}")


def create_readme():
    """创建README.md"""
    readme = """# HTTG Perfusion Monitoring System

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

# 运行基础测试（无需Neo4j）
python test_evidence_strategy.py

# 运行完整策略测试
python test_full_strategy.py

# 运行Baseline策略推荐测试
python src/baseline_strategy_recommender.py
```

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
"""
    readme_path = PROJECT_ROOT / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme)
    print(f"Created: {readme_path}")


def package_modules():
    """打包所有模块"""
    # 创建必要文件
    create_requirements()
    create_readme()

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"httg_perfusion_system_{timestamp}.zip"
    zip_path = PROJECT_ROOT / zip_name

    print(f"\nPackaging to: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 打包指定的文件
        for prefix, files in INCLUDE_PATTERNS.items():
            for filename in files:
                src_path = PROJECT_ROOT / prefix / filename
                if src_path.exists():
                    arcname = f"httg_perfusion/{prefix}{filename}"
                    zipf.write(src_path, arcname)
                    print(f"  Added: {prefix}{filename}")
                else:
                    print(f"  [SKIP] {prefix}{filename} (not found)")

        # 打包额外文件
        for filename in EXTRA_FILES:
            src_path = PROJECT_ROOT / filename
            if src_path.exists():
                arcname = f"httg_perfusion/{filename}"
                zipf.write(src_path, arcname)
                print(f"  Added: {filename}")

        # 创建__init__.py如果不存在
        init_content = '"""HTTG Perfusion Monitoring System"""\n'
        zipf.writestr("httg_perfusion/__init__.py", init_content)
        zipf.writestr("httg_perfusion/src/__init__.py", init_content)

    print(f"\n✅ Package created: {zip_path}")
    print(f"   Size: {os.path.getsize(zip_path) / 1024:.1f} KB")

    return zip_path


def verify_package(zip_path):
    """验证打包内容"""
    print(f"\nVerifying package contents:")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        for name in zipf.namelist():
            info = zipf.getinfo(name)
            print(f"  {name} ({info.file_size} bytes)")


if __name__ == "__main__":
    zip_path = package_modules()
    verify_package(zip_path)
