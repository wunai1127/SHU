"""
Heart Transplant Perfusion AI - Configuration Settings
针对心脏移植灌注AI系统的配置
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ===== 基础路径 =====
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data_extracted"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"

# 创建必要目录
for d in [OUTPUT_DIR, CACHE_DIR, OUTPUT_DIR / "models", OUTPUT_DIR / "embeddings"]:
    d.mkdir(parents=True, exist_ok=True)

# ===== 知识图谱配置（心脏移植灌注领域） =====
KB_NAME = "心脏移植灌注决策支持系统"
THEME = "心脏移植机械灌注监测与干预决策"

# 实体类型 - 针对心脏移植灌注领域
ENTITY_TYPES = [
    # 临床实体
    "生理指标",        # 乳酸、pH、压力、流量、温度等
    "干预措施",        # 加压、减压、药物调整等
    "并发症",          # 原发性移植物功能障碍(PGD)、排斥反应等
    "器官状态",        # 供心状态、缺血时间等
    "设备参数",        # OCS/XVIVO设备参数

    # 医学知识实体
    "临床指南",        # 技术规范、操作标准
    "药物",           # 正性肌力药、免疫抑制剂等
    "检查指标",        # 实验室检查、影像学检查
    "供者特征",        # 年龄、体重、感染状态等
    "受者特征",        # PRA、血型、基础疾病等

    # 因果实体
    "风险因素",        # 导致不良预后的因素
    "保护因素",        # 改善预后的因素
]

# 关系类型 - 针对灌注决策
RELATIONSHIP_TYPES = [
    # 因果关系
    "导致",           # A导致B（如：高乳酸 -> 组织缺氧）
    "改善",           # A改善B（如：加压 -> 灌注改善）
    "恶化",           # A恶化B
    "预测",           # A预测B（如：乳酸清除率 -> 预后）

    # 操作关系
    "监测",           # 监测某指标
    "干预",           # 对某状态进行干预
    "禁忌",           # 某情况下禁止某操作
    "适应",           # 某情况下适合某操作

    # 知识关系
    "参考",           # 引用指南/文献
    "包含",           # 包含关系
    "相关",           # 相关但非因果
    "对比",           # 比较关系（如不同方案对比）
]

# ===== 文本处理配置 =====
CHUNK_SIZE = 512            # 文本分块大小
CHUNK_OVERLAP = 128         # 分块重叠
MAX_TEXT_LENGTH = 100000    # 最大文本长度

# ===== 模型配置 =====
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_DIM = 3072        # text-embedding-3-large维度
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# 本地备选Embedding模型
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LOCAL_EMBEDDING_DIM = 384

# ===== 训练配置 =====
TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eval_steps": 500,
    "save_steps": 1000,
    "logging_steps": 100,
}

# ===== 时序模型配置 =====
TIME_SERIES_CONFIG = {
    "sequence_length": 60,      # 60个时间点（如每分钟采样，1小时数据）
    "prediction_horizon": 15,   # 预测未来15个时间点
    "features": [
        "lactate",              # 乳酸 (mmol/L)
        "ph",                   # pH值
        "aortic_pressure",      # 主动脉压 (mmHg)
        "coronary_flow",        # 冠脉流量 (mL/min)
        "temperature",          # 温度 (°C)
        "pao2",                 # 动脉氧分压
        "pco2",                 # 动脉二氧化碳分压
    ],
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "attention_heads": 4,
    "dropout": 0.2,
}

# ===== 知识图谱配置 =====
NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://localhost:7687"),
    "username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", "12345678"),
}

# ===== 安全约束配置 =====
SAFETY_CONSTRAINTS = {
    "pressure": {"min": 40, "max": 100, "unit": "mmHg"},
    "temperature": {"min": 4, "max": 37, "unit": "°C"},
    "flow": {"min": 0.5, "max": 2.5, "unit": "L/min"},
    "lactate": {"warning": 4.0, "critical": 8.0, "unit": "mmol/L"},
    "ph": {"min": 7.2, "max": 7.5},
}

# ===== API配置 =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
