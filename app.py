#!/usr/bin/env python3
"""
HTTG 灌注监测系统 - Streamlit 前端
========================================

功能：
1. 状态卡片 - 实时显示指标状态（🔴危急/🟡警告/🟢正常）
2. 时序趋势图 - 多指标趋势可视化
3. 策略推荐面板 - 干预建议+CoT推理链
4. 证据溯源面板 - KG三元组+Neo4j查询结果
5. 样本选择器 - 切换不同病例

运行: streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
try:
    import plotly.express as px
except ImportError:
    px = None
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    make_subplots = None
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import sys

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# 添加src目录到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

# 导入后端模块
try:
    from baseline_thresholds import BaselineThresholds
    from evidence_strategy_engine import EvidenceStrategyEngine
    from baseline_strategy_recommender import BaselineStrategyRecommender
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.warning(f"后端模块导入警告: {e}")
    BACKEND_AVAILABLE = False

# 导入Agent系统
try:
    from agents.base import EventBus, PatientState
    from agents.monitor_agent import MonitorAgent
    from agents.diagnosis_agent import DiagnosisAgent
    from agents.strategy_agent import StrategyAgent
    from agents.knowledge_agent import KnowledgeAgent
    from agents.communication_agent import CommunicationAgent
    from agents.coordinator import CoordinatorAgent
    AGENT_AVAILABLE = True
except ImportError as e:
    AGENT_AVAILABLE = False

# 导入Neo4j和LLM模块
NEO4J_AVAILABLE = False
LLM_AVAILABLE = False
_neo4j_instance = None
_llm_instance = None

try:
    from neo4j_connector import Neo4jKnowledgeGraph
    from baseline_strategy_recommender import OpenAILLM
except ImportError:
    pass

# 导入灌注算法模块
try:
    from perfusion_algorithms import (
        SetpointReadoutCausalOptimizer,
        CUSUMDetector,
        CompositePerfusionRiskScore,
        LactateTrajectoryPredictor,
        run_all_algorithms,
    )
    ALGO_AVAILABLE = True
except ImportError:
    ALGO_AVAILABLE = False


@st.cache_resource
def init_neo4j():
    """初始化Neo4j连接（全局单例）"""
    try:
        kg = Neo4jKnowledgeGraph()
        return kg
    except Exception as e:
        return None


@st.cache_resource
def init_llm():
    """初始化LLM客户端（全局单例）"""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "deepseek-v3.2")
    if not api_key:
        return None, "OPENAI_API_KEY未设置"
    if not base_url:
        return None, "OPENAI_BASE_URL未设置"
    try:
        llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)
        if llm.is_available():
            return llm, None
        return None, "OpenAI客户端创建失败"
    except Exception as e:
        return None, str(e)


# =============================================================================
# 页面配置
# =============================================================================
st.set_page_config(
    page_title="HTTG 灌注监测系统",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 样式配置
# =============================================================================
st.markdown("""
<style>
    /* 状态卡片样式 */
    .status-card {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .status-critical {
        background: linear-gradient(135deg, #ff4d4f 0%, #cf1322 100%);
        color: white;
    }
    .status-warning {
        background: linear-gradient(135deg, #faad14 0%, #d48806 100%);
        color: white;
    }
    .status-normal {
        background: linear-gradient(135deg, #52c41a 0%, #389e0d 100%);
        color: white;
    }
    .status-pending {
        background: linear-gradient(135deg, #8c8c8c 0%, #595959 100%);
        color: white;
    }

    /* 指标值大数字 */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    /* 策略卡片 */
    .strategy-card {
        background: #f6f8fa;
        border-left: 4px solid #1890ff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }

    /* 证据项 */
    .evidence-item {
        background: #f0f5ff;
        border: 1px solid #adc6ff;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        border-radius: 4px;
        font-family: monospace;
    }

    /* 风险标签 */
    .risk-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-high { background: #ff4d4f; color: white; }
    .risk-medium { background: #faad14; color: white; }
    .risk-low { background: #52c41a; color: white; }

    /* 隐藏Streamlit默认footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 数据加载
# =============================================================================
@st.cache_data
def load_config():
    """加载配置文件"""
    config = {}
    config_dir = Path(__file__).parent / "config"

    for file in ["thresholds.yaml", "baseline.yaml", "intervention_strategies.yaml"]:
        file_path = config_dir / file
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                config[file.replace('.yaml', '')] = yaml.safe_load(f)

    return config

@st.cache_data
def load_patient_data():
    """加载患者数据"""
    data_file = Path(__file__).parent / "neo4j_query_table_data_2026-1-26.json"
    if data_file.exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_demo_data() -> Dict[str, Any]:
    """生成演示数据（含灌注调控参数 + 功能观测指标）"""
    return {
        "HTX-28": {
            "baseline": {
                # 灌注调控参数 (Setpoints)
                "Flow": 4.5, "Temperature": 22.0, "AoDP": 40, "PaO2": 150,
                "Hemoglobin": 45, "PacingRate": 0, "Dobutamine": 0, "Insulin": 2.0,
                # 功能观测指标 (Readouts)
                "Lactate": 2.8, "pH": 7.28, "K_A": 4.2, "EF": 0, "CI": 0,
                "SvO2": 82, "CvO2": 8.5, "MVO2": 6.5, "dPdt_max": 0,
                # 移植评估
                "MAP": 36, "HR": 0,
            },
            "60min": {
                "Flow": 4.5, "Temperature": 28.0, "AoDP": 38, "PaO2": 160,
                "Hemoglobin": 44, "PacingRate": 0, "Dobutamine": 0, "Insulin": 2.0,
                "Lactate": 3.5, "pH": 7.30, "K_A": 4.8, "EF": 15, "CI": 0,
                "SvO2": 78, "CvO2": 9.2, "MVO2": 7.2, "dPdt_max": 800,
                "MAP": 45, "HR": 0,
            },
            "120min": {
                "Flow": 4.4, "Temperature": 34.0, "AoDP": 42, "PaO2": 155,
                "Hemoglobin": 43, "PacingRate": 105, "Dobutamine": 4.0, "Insulin": 2.25,
                "Lactate": 3.2, "pH": 7.32, "K_A": 4.5, "EF": 22, "CI": 2.3,
                "SvO2": 75, "CvO2": 10.1, "MVO2": 9.5, "dPdt_max": 1350,
                "MAP": 50, "HR": 85,
            },
            "180min": {
                "Flow": 4.3, "Temperature": 36.5, "AoDP": 40, "PaO2": 145,
                "Hemoglobin": 42, "PacingRate": 105, "Dobutamine": 4.0, "Insulin": 2.25,
                "Lactate": 2.5, "pH": 7.33, "K_A": 4.3, "EF": 28, "CI": 2.7,
                "SvO2": 72, "CvO2": 11.5, "MVO2": 10.8, "dPdt_max": 1450,
                "MAP": 58, "HR": 82,
            },
            "240min": {
                "Flow": 4.2, "Temperature": 37.0, "AoDP": 40, "PaO2": 140,
                "Hemoglobin": 42, "PacingRate": 105, "Dobutamine": 3.0, "Insulin": 2.0,
                "Lactate": 2.0, "pH": 7.35, "K_A": 4.1, "EF": 35, "CI": 2.9,
                "SvO2": 70, "CvO2": 12.8, "MVO2": 12.0, "dPdt_max": 1500,
                "MAP": 65, "HR": 78,
            },
            "outcome": "success",
            "age": 45,
            "gender": "M"
        },
        "HTX-36": {
            "baseline": {
                "Flow": 4.6, "Temperature": 22.0, "AoDP": 38, "PaO2": 140,
                "Hemoglobin": 38, "PacingRate": 0, "Dobutamine": 0, "Insulin": 2.5,
                "Lactate": 3.9, "pH": 7.22, "K_A": 5.8, "EF": 0, "CI": 0,
                "SvO2": 85, "CvO2": 7.5, "MVO2": 5.8, "dPdt_max": 0,
                "MAP": 45, "HR": 0,
            },
            "60min": {
                "Flow": 4.7, "Temperature": 27.0, "AoDP": 35, "PaO2": 135,
                "Hemoglobin": 36, "PacingRate": 0, "Dobutamine": 0, "Insulin": 2.5,
                "Lactate": 4.9, "pH": 7.20, "K_A": 6.2, "EF": 12, "CI": 0,
                "SvO2": 82, "CvO2": 8.0, "MVO2": 6.2, "dPdt_max": 650,
                "MAP": 51, "HR": 0,
            },
            "120min": {
                "Flow": 4.8, "Temperature": 32.0, "AoDP": 36, "PaO2": 130,
                "Hemoglobin": 35, "PacingRate": 105, "Dobutamine": 5.0, "Insulin": 3.0,
                "Lactate": 4.1, "pH": 7.22, "K_A": 5.9, "EF": 14, "CI": 2.0,
                "SvO2": 80, "CvO2": 8.5, "MVO2": 7.0, "dPdt_max": 950,
                "MAP": 51, "HR": 108,
            },
            "180min": {
                "Flow": 4.9, "Temperature": 35.0, "AoDP": 34, "PaO2": 125,
                "Hemoglobin": 34, "PacingRate": 105, "Dobutamine": 6.0, "Insulin": 3.0,
                "Lactate": 5.2, "pH": 7.19, "K_A": 6.5, "EF": 11, "CI": 1.8,
                "SvO2": 78, "CvO2": 7.8, "MVO2": 6.0, "dPdt_max": 850,
                "MAP": 48, "HR": 115,
            },
            "240min": {
                "Flow": 5.0, "Temperature": 36.5, "AoDP": 32, "PaO2": 120,
                "Hemoglobin": 33, "PacingRate": 110, "Dobutamine": 6.0, "Insulin": 3.0,
                "Lactate": 6.8, "pH": 7.15, "K_A": 7.1, "EF": 8, "CI": 1.5,
                "SvO2": 75, "CvO2": 6.5, "MVO2": 4.8, "dPdt_max": 700,
                "MAP": 42, "HR": 125,
            },
            "outcome": "failure",
            "age": 58,
            "gender": "M"
        },
        "HTX-42": {
            "baseline": {
                "Flow": 4.4, "Temperature": 22.0, "AoDP": 42, "PaO2": 165,
                "Hemoglobin": 48, "PacingRate": 0, "Dobutamine": 0, "Insulin": 1.5,
                "Lactate": 2.5, "pH": 7.30, "K_A": 4.0, "EF": 0, "CI": 0,
                "SvO2": 78, "CvO2": 10.5, "MVO2": 7.5, "dPdt_max": 0,
                "MAP": 52, "HR": 0,
            },
            "60min": {
                "Flow": 4.3, "Temperature": 30.0, "AoDP": 41, "PaO2": 160,
                "Hemoglobin": 47, "PacingRate": 0, "Dobutamine": 0, "Insulin": 1.5,
                "Lactate": 2.2, "pH": 7.32, "K_A": 4.2, "EF": 20, "CI": 0,
                "SvO2": 75, "CvO2": 11.2, "MVO2": 8.5, "dPdt_max": 1100,
                "MAP": 58, "HR": 0,
            },
            "120min": {
                "Flow": 4.3, "Temperature": 35.5, "AoDP": 42, "PaO2": 155,
                "Hemoglobin": 46, "PacingRate": 105, "Dobutamine": 3.0, "Insulin": 2.0,
                "Lactate": 1.8, "pH": 7.34, "K_A": 4.1, "EF": 30, "CI": 2.8,
                "SvO2": 72, "CvO2": 12.0, "MVO2": 10.5, "dPdt_max": 1500,
                "MAP": 65, "HR": 75,
            },
            "180min": {
                "Flow": 4.2, "Temperature": 37.0, "AoDP": 41, "PaO2": 150,
                "Hemoglobin": 45, "PacingRate": 105, "Dobutamine": 3.0, "Insulin": 2.0,
                "Lactate": 1.5, "pH": 7.35, "K_A": 4.0, "EF": 38, "CI": 3.0,
                "SvO2": 70, "CvO2": 13.2, "MVO2": 12.0, "dPdt_max": 1580,
                "MAP": 70, "HR": 72,
            },
            "240min": {
                "Flow": 4.2, "Temperature": 37.0, "AoDP": 40, "PaO2": 145,
                "Hemoglobin": 45, "PacingRate": 105, "Dobutamine": 2.0, "Insulin": 1.5,
                "Lactate": 1.2, "pH": 7.35, "K_A": 3.9, "EF": 42, "CI": 3.2,
                "SvO2": 68, "CvO2": 14.0, "MVO2": 13.5, "dPdt_max": 1620,
                "MAP": 72, "HR": 70,
            },
            "outcome": "success",
            "age": 38,
            "gender": "F"
        }
    }

# =============================================================================
# 指标配置
# =============================================================================

# ------ 灌注调控参数 (Setpoints) — 灌注师可直接调控 ------
SETPOINT_CONFIG = {
    "Flow": {
        "name": "灌注流量", "unit": "L/min", "target": (4.2, 4.8),
        "red_line": 3.5, "critical": 3.8,
        "control": "离心泵/滚压泵转速", "device": "CPB泵",
        "priority": 1,
    },
    "Temperature": {
        "name": "灌注温度", "unit": "°C", "target": (34, 37),
        "red_line": None, "critical": None,
        "control": "热交换器", "device": "变温水箱",
        "priority": 2,
    },
    "AoDP": {
        "name": "灌注压(AoDP)", "unit": "mmHg", "target": (35, 45),
        "red_line": 25, "critical": 30,
        "control": "离心泵转速/反馈控制", "device": "离心泵",
        "priority": 3,
    },
    "PaO2": {
        "name": "动脉氧分压", "unit": "mmHg", "target": (100, 200),
        "red_line": 60, "critical": 80,
        "control": "氧合器FiO2/扫气", "device": "氧合器",
        "priority": 4,
    },
    "Hemoglobin": {
        "name": "血红蛋白", "unit": "g/L", "target": (40, 50),
        "red_line": 30, "critical": 35,
        "control": "RBC添加/稀释", "device": "储血罐",
        "priority": 5,
    },
    "PacingRate": {
        "name": "起搏心率", "unit": "bpm", "target": (100, 110),
        "red_line": None, "critical": None,
        "control": "起搏器设定", "device": "起搏器",
        "priority": 6,
    },
    "Dobutamine": {
        "name": "多巴酚丁胺", "unit": "μg/min", "target": (2, 6),
        "red_line": None, "critical": None,
        "control": "注射泵速率", "device": "注射泵",
        "priority": 7,
    },
    "Insulin": {
        "name": "胰岛素", "unit": "U/h", "target": (1.5, 3.0),
        "red_line": None, "critical": None,
        "control": "注射泵速率", "device": "注射泵",
        "priority": 8,
    },
}

# ------ 血气观测指标 (Blood Gas Readouts) — 反映代谢和氧合状态 ------
READOUT_BLOOD_GAS_CONFIG = {
    "Lactate": {"name": "乳酸", "unit": "mmol/L", "target": (0, 4.0), "red_line": 6.0, "critical": 4.0, "higher_is_worse": True},
    "pH": {"name": "动脉pH", "unit": "", "target": (7.25, 7.35), "red_line": 7.15, "critical": 7.20},
    "K_A": {"name": "动脉血钾", "unit": "mmol/L", "target": (3.5, 5.0), "red_line": 6.0, "critical": 5.5, "higher_is_worse": True},
    "SvO2": {"name": "混合静脉血氧饱和度", "unit": "%", "target": (65, 80), "red_line": 50, "critical": 60},
    "CvO2": {"name": "静脉血氧含量", "unit": "mL/dL", "target": (12, 16), "red_line": 8, "critical": 10},
}

# ------ 心功能观测指标 (Cardiac Function Readouts) — 反映心脏收缩/舒张功能 ------
READOUT_FUNCTION_CONFIG = {
    "EF": {"name": "射血分数", "unit": "%", "target": (18, 60), "red_line": 10, "critical": 18},
    "CI": {"name": "心指数", "unit": "L/min/m²", "target": (2.2, 4.0), "red_line": 1.8, "critical": 2.0},
    "MVO2": {"name": "心肌氧耗", "unit": "mLO₂/min/100g", "target": (8.8, 20), "red_line": 5, "critical": 8.8},
    "dPdt_max": {"name": "最大dP/dt", "unit": "mmHg/s", "target": (1200, 1800), "red_line": 800, "critical": 1000},
}

# ------ 合并所有Readout配置（兼容旧代码） ------
READOUT_CONFIG = {}
READOUT_CONFIG.update(READOUT_BLOOD_GAS_CONFIG)
READOUT_CONFIG.update(READOUT_FUNCTION_CONFIG)

# ------ 移植评估/术后指标 ------
TRANSPLANT_CONFIG = {
    "MAP": {"name": "平均动脉压", "unit": "mmHg", "target": (65, 90), "red_line": 50, "critical": 60},
    "HR": {"name": "心率", "unit": "bpm", "target": (60, 100), "red_line": None, "critical": None},
    "PVR": {"name": "肺血管阻力", "unit": "Wood", "target": (0.5, 2.5), "red_line": 5.0, "critical": 4.0, "higher_is_worse": True},
    "TPG": {"name": "跨肺压差", "unit": "mmHg", "target": (5, 12), "red_line": 15, "critical": 14, "higher_is_worse": True},
    "PASP": {"name": "肺动脉收缩压", "unit": "mmHg", "target": (15, 40), "red_line": 70, "critical": 50, "higher_is_worse": True},
    "Creatinine": {"name": "肌酐", "unit": "mg/dL", "target": (0.5, 1.5), "red_line": 2.0, "critical": 1.7, "higher_is_worse": True},
    "GFR": {"name": "肾小球滤过率", "unit": "mL/min", "target": (60, 120), "red_line": 30, "critical": 60},
    "Bilirubin": {"name": "胆红素", "unit": "mg/dL", "target": (0.1, 1.2), "red_line": 2.5, "critical": 2.0, "higher_is_worse": True},
}

# 合并所有指标配置（兼容旧代码）
INDICATOR_CONFIG = {}
INDICATOR_CONFIG.update(SETPOINT_CONFIG)
INDICATOR_CONFIG.update(READOUT_CONFIG)
INDICATOR_CONFIG.update(TRANSPLANT_CONFIG)

def get_status(indicator: str, value: float) -> Tuple[str, str]:
    """获取指标状态"""
    config = INDICATOR_CONFIG.get(indicator, {})
    target = config.get("target", (0, 100))
    red_line = config.get("red_line")
    critical = config.get("critical")
    higher_is_worse = config.get("higher_is_worse", False)

    if higher_is_worse:
        if red_line and value >= red_line:
            return "critical", "🔴"
        elif critical and value >= critical:
            return "warning", "🟡"
        elif target[0] <= value <= target[1]:
            return "normal", "🟢"
        else:
            return "warning", "🟡"
    else:
        if red_line and value <= red_line:
            return "critical", "🔴"
        elif critical and value <= critical:
            return "warning", "🟡"
        elif target[0] <= value <= target[1]:
            return "normal", "🟢"
        else:
            return "warning", "🟡"

# =============================================================================
# 组件函数
# =============================================================================
def render_header(sample_id: str, timepoint: str, risk_level: str):
    """渲染顶部Header"""
    risk_colors = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}
    risk_class = risk_colors.get(risk_level, "risk-medium")

    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    with col1:
        st.markdown("### 🫀 HTTG 灌注监测系统")
    with col2:
        st.markdown(f"**样本:** {sample_id}")
    with col3:
        st.markdown(f"**时间点:** {timepoint}")
    with col4:
        st.markdown(f'<span class="risk-badge {risk_class}">{risk_level} RISK</span>', unsafe_allow_html=True)

def _render_indicator_card(indicator: str, value: float, baseline_val: float, config: Dict,
                           show_device: bool = False):
    """渲染单个指标卡片"""
    status, icon = get_status(indicator, value)

    # 计算变化
    change = value - baseline_val
    change_pct = (change / baseline_val * 100) if baseline_val != 0 else 0
    trend = "↑" if change > 0 else "↓" if change < 0 else "→"

    # 状态颜色
    bg_colors = {
        "critical": "#ff4d4f",
        "warning": "#faad14",
        "normal": "#52c41a"
    }
    bg_color = bg_colors.get(status, "#8c8c8c")

    device_line = ""
    if show_device and config.get("device"):
        device_line = f'<div style="font-size:0.65rem; opacity:0.8; margin-top:2px;">🎛 {config["device"]}</div>'

    # 格式化值（小数位数根据单位）
    fmt = f"{value:.0f}" if config.get("unit") in ("bpm", "mmHg", "mmHg/s", "g/L", "%") else f"{value:.1f}"

    st.markdown(f"""
    <div style="background: {bg_color}; padding: 0.8rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 0.5rem;">
        <div style="font-size: 0.85rem; opacity: 0.9;">{icon} {config.get('name', indicator)}</div>
        <div style="font-size: 1.8rem; font-weight: bold; margin: 0.2rem 0;">{fmt}</div>
        <div style="font-size: 0.75rem; opacity: 0.8;">{config.get('unit', '')}</div>
        <div style="font-size: 0.7rem; margin-top: 0.2rem;">
            {trend} {abs(change):.1f} ({change_pct:+.1f}%) · 目标: {config.get('target', (0,0))[0]}-{config.get('target', (0,0))[1]}
        </div>
        {device_line}
    </div>
    """, unsafe_allow_html=True)


def render_status_cards(data: Dict[str, float], baseline: Dict[str, float]):
    """渲染状态卡片 — 灌注调控参数(Setpoints)优先 + 功能观测指标(Readouts)分血气/功能两部分"""

    # ===== 灌注调控参数 (Setpoints) =====
    st.markdown(
        '<div style="padding:6px 12px; margin-bottom:8px; border-left:4px solid #1890ff; '
        'background:rgba(24,144,255,0.06); border-radius:0 6px 6px 0;">'
        '<strong>🎛 灌注调控参数 (Setpoints)</strong> — 灌注师可直接调控：流量/温度/压力/药物</div>',
        unsafe_allow_html=True
    )
    setpoint_keys = [k for k in SETPOINT_CONFIG if k in data]
    cols = st.columns(min(len(setpoint_keys), 4) or 4)
    for i, indicator in enumerate(setpoint_keys):
        with cols[i % 4]:
            value = data.get(indicator, 0)
            baseline_val = baseline.get(indicator, value)
            config = SETPOINT_CONFIG[indicator]
            _render_indicator_card(indicator, value, baseline_val, config, show_device=True)

    # ===== 血气观测指标 (Blood Gas Readouts) =====
    st.markdown(
        '<div style="padding:6px 12px; margin:12px 0 8px 0; border-left:4px solid #52c41a; '
        'background:rgba(82,196,26,0.06); border-radius:0 6px 6px 0;">'
        '<strong>🩸 血气观测指标 (Blood Gas)</strong> — 反映代谢和氧合状态</div>',
        unsafe_allow_html=True
    )
    blood_gas_keys = [k for k in READOUT_BLOOD_GAS_CONFIG if k in data]
    cols2 = st.columns(min(len(blood_gas_keys), 4) or 4)
    for i, indicator in enumerate(blood_gas_keys):
        with cols2[i % 4]:
            value = data.get(indicator, 0)
            baseline_val = baseline.get(indicator, value)
            config = READOUT_BLOOD_GAS_CONFIG[indicator]
            _render_indicator_card(indicator, value, baseline_val, config)

    # ===== 心功能观测指标 (Cardiac Function Readouts) =====
    st.markdown(
        '<div style="padding:6px 12px; margin:12px 0 8px 0; border-left:4px solid #722ed1; '
        'background:rgba(114,46,209,0.06); border-radius:0 6px 6px 0;">'
        '<strong>💓 心功能观测指标 (Cardiac Function)</strong> — 反映心脏收缩/舒张功能</div>',
        unsafe_allow_html=True
    )
    function_keys = [k for k in READOUT_FUNCTION_CONFIG if k in data]
    cols3 = st.columns(min(len(function_keys), 4) or 4)
    for i, indicator in enumerate(function_keys):
        with cols3[i % 4]:
            value = data.get(indicator, 0)
            baseline_val = baseline.get(indicator, value)
            config = READOUT_FUNCTION_CONFIG[indicator]
            _render_indicator_card(indicator, value, baseline_val, config)

def render_time_series(patient_data: Dict[str, Any], selected_indicators: List[str]):
    """渲染时序趋势图"""
    if go is None or make_subplots is None:
        st.warning("plotly 未安装，无法显示趋势图。请运行: pip install plotly==5.24.1")
        return
    timepoints = ["baseline", "60min", "120min", "180min", "240min"]
    time_labels = ["Baseline\n(30min)", "60min", "120min", "180min", "240min"]

    fig = make_subplots(rows=len(selected_indicators), cols=1,
                        shared_xaxes=True,
                        subplot_titles=selected_indicators,
                        vertical_spacing=0.08)

    if px is not None:
        colors = px.colors.qualitative.Set2
    else:
        colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]

    for i, indicator in enumerate(selected_indicators, 1):
        config = INDICATOR_CONFIG.get(indicator, {})
        values = [patient_data.get(tp, {}).get(indicator, None) for tp in timepoints]

        # 主线
        fig.add_trace(
            go.Scatter(
                x=time_labels, y=values,
                mode='lines+markers',
                name=indicator,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=10)
            ),
            row=i, col=1
        )

        # 目标区域
        target = config.get("target", (0, 100))
        fig.add_hrect(
            y0=target[0], y1=target[1],
            fillcolor="green", opacity=0.1,
            line_width=0,
            row=i, col=1
        )

        # 红线
        red_line = config.get("red_line")
        if red_line:
            fig.add_hline(
                y=red_line, line_dash="dash", line_color="red",
                annotation_text="红线",
                row=i, col=1
            )

        # Y轴标签
        fig.update_yaxes(title_text=config.get("unit", ""), row=i, col=1)

    fig.update_layout(
        height=200 * len(selected_indicators),
        showlegend=False,
        margin=dict(l=60, r=20, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

def get_strategy_recommendations(data: Dict[str, float], baseline: Dict[str, float]) -> List[Dict]:
    """获取策略推荐（灌注调控优先）"""
    recommendations = []

    # 检查每个指标
    for indicator, value in data.items():
        config = INDICATOR_CONFIG.get(indicator, {})
        if not config:
            continue
        status, _ = get_status(indicator, value)

        if status in ["critical", "warning"]:
            target = config.get("target", (0, 100))

            rec = {
                "indicator": indicator,
                "name": config.get("name", indicator),
                "current": value,
                "target": sum(target) / 2,
                "unit": config.get("unit", ""),
                "priority": "URGENT" if status == "critical" else "Standard",
                "status": status
            }

            # ===== 灌注调控参数异常 =====
            if indicator == "Flow":
                rec["intervention"] = "调整泵流量"
                rec["drug"] = "调整离心泵/滚压泵转速"
                rec["dose"] = f"目标流量 4.2-4.8 L/min (当前 {value:.1f})"
                rec["reasoning"] = [
                    f"Step 1 - 观察: 灌注流量={value:.1f} L/min，偏离目标范围",
                    "Step 2 - 分析: 流量是灌注最核心参数，直接影响组织氧供",
                    "Step 3 - 机制: 流量↓→DO2↓→组织缺氧→Lactate↑",
                    "Step 4 - 干预: 调整泵转速，检查管路阻力和储血罐液面",
                    "Step 5 - 预期: 流量恢复至4.2-4.8 L/min"
                ]
                rec["monitoring"] = ["Flow: 持续", "Lactate: 每15分钟", "AoDP: 持续"]
                rec["caution"] = ["排除管路折叠/气泡", "注意储血罐液面"]

            elif indicator == "Temperature":
                rec["intervention"] = "调整变温水箱"
                rec["drug"] = "热交换器温度调整"
                rec["dose"] = f"复温方案: 22→37°C/30min (当前 {value:.1f}°C)"
                rec["reasoning"] = [
                    f"Step 1 - 观察: 灌注温度={value:.1f}°C",
                    "Step 2 - 分析: 温度影响范围最广（CVR、Tau、代谢率）",
                    "Step 3 - 机制: 低温→CVR↑→冠脉微循环阻力增加",
                    "Step 4 - 干预: 按复温方案调整热交换器",
                    "Step 5 - 预期: 按方案升温至目标"
                ]
                rec["monitoring"] = ["Temperature: 每5分钟", "CVR: 每15分钟", "Lactate: 每15分钟"]
                rec["caution"] = ["升温过快可致微循环损伤", "注意温差<10°C"]

            elif indicator == "AoDP":
                rec["intervention"] = "调整灌注压力"
                rec["drug"] = "调整离心泵转速/反馈控制"
                rec["dose"] = f"目标AoDP 35-45 mmHg (当前 {value:.0f})"
                rec["reasoning"] = [
                    f"Step 1 - 观察: AoDP={value:.0f} mmHg，灌注压偏离目标",
                    "Step 2 - 分析: AoDP是冠脉灌注驱动压",
                    "Step 3 - 机制: AoDP↓→冠脉灌注↓→心肌缺氧→EF↓",
                    "Step 4 - 干预: 调整泵转速，检查后负荷",
                    "Step 5 - 预期: AoDP稳定在40 mmHg"
                ]
                rec["monitoring"] = ["AoDP: 持续", "CF: 持续", "Lactate: 每15分钟"]
                rec["caution"] = ["过高压力可致水肿", "注意冠脉插管位置"]

            elif indicator == "Hemoglobin":
                rec["intervention"] = "血液管理"
                rec["drug"] = "RBC添加 / 灌注液调配"
                rec["dose"] = f"目标Hb 40-50 g/L (当前 {value:.0f})"
                rec["reasoning"] = [
                    f"Step 1 - 观察: Hb={value:.0f} g/L，携氧能力不足",
                    "Step 2 - 分析: Hb直接影响CaO2和DO2",
                    "Step 3 - 机制: Hb↓→CaO2↓→MVO2↓→心肌缺氧",
                    "Step 4 - 干预: 添加RBC至储血罐",
                    "Step 5 - 预期: Hb恢复至40-50 g/L"
                ]
                rec["monitoring"] = ["Hb: 每30分钟", "MVO2: 持续", "O2提取率: 持续"]
                rec["caution"] = ["注意容量负荷", "高Hb可致高粘滞"]

            elif indicator == "PaO2":
                rec["intervention"] = "调整氧合器参数"
                rec["drug"] = "调整FiO2和扫气流量"
                rec["dose"] = f"目标PaO2 100-200 mmHg (当前 {value:.0f})"
                rec["reasoning"] = [
                    f"Step 1 - 观察: PaO2={value:.0f} mmHg",
                    "Step 2 - 分析: PaO2影响溶解氧和CaO2",
                    "Step 3 - 机制: PaO2↓→CaO2↓→Lactate↑",
                    "Step 4 - 干预: 调整氧合器FiO2↑/扫气流量↑",
                    "Step 5 - 预期: PaO2恢复至目标范围"
                ]
                rec["monitoring"] = ["PaO2: 每15分钟", "Lactate: 每15分钟"]
                rec["caution"] = ["过高FiO2可致氧中毒"]

            # ===== 功能观测指标异常 =====
            elif indicator == "Lactate":
                rec["intervention"] = "优化灌注参数（Flow/AoDP/Hb）"
                rec["drug"] = "调整Setpoints改善灌注"
                rec["dose"] = "检查Flow、AoDP、Hb、PaO2"
                rec["reasoning"] = [
                    f"Step 1 - 观察: Lactate={value:.1f} mmol/L，组织缺氧/灌注不足",
                    "Step 2 - 分析: 乳酸是灌注质量核心监测指标",
                    "Step 3 - 机制: 灌注不足→无氧代谢→Lactate↑",
                    "Step 4 - 干预: 检查并优化Flow↑、AoDP→目标、Hb→目标",
                    "Step 5 - 预期: Lactate趋势下降，<5 mmol/L(OCS接受标准)"
                ]
                rec["monitoring"] = ["Lactate: 每15分钟", "乳酸清除率: 趋势", "Flow: 持续"]
                rec["caution"] = ["持续>5且上升→评估器官质量", "排除灌注液本身问题"]

            elif indicator == "K_A":
                rec["intervention"] = "电解质纠正（灌注液/打药）"
                rec["drug"] = "胰岛素+葡萄糖 / 钙剂 / 灌注液KCl调整"
                rec["dose"] = "高钾: 10U胰岛素+25g葡萄糖; 低钾: KCl 10-20mEq/h加入灌注液"
                rec["reasoning"] = [
                    f"Step 1 - 观察: K+={value:.1f} mmol/L",
                    "Step 2 - 分析: 钾异常可致致命性心律失常",
                    "Step 3 - 机制: 高钾→心肌传导异常; 可能与心肌保护液相关",
                    "Step 4 - 干预: 高钾→胰岛素降钾+钙剂护心; 低钾→灌注液补KCl",
                    "Step 5 - 预期: K+恢复至4.0-5.0 mmol/L"
                ]
                rec["monitoring"] = ["K+: 每30分钟", "ECG: 持续", "血糖: 每30分钟"]
                rec["caution"] = ["注意心肌保护液残余高钾", "库存血含钾也较高"]

            elif indicator == "pH":
                rec["intervention"] = "调整氧合器扫气 / NaHCO3"
                rec["drug"] = "扫气流量↑排CO2 或 NaHCO3纠酸"
                rec["dose"] = "扫气流量调整; NaHCO3根据BE计算"
                rec["reasoning"] = [
                    f"Step 1 - 观察: pH={value:.2f}",
                    "Step 2 - 分析: 酸中毒影响心肌收缩力和药物效应",
                    "Step 3 - 机制: pH↓→心肌抑制+血管反应性↓",
                    "Step 4 - 干预: 呼吸性→扫气流量↑排CO2; 代谢性→NaHCO3",
                    "Step 5 - 预期: pH恢复至7.25-7.35"
                ]
                rec["monitoring"] = ["血气: 每15分钟", "电解质: 同步"]
                rec["caution"] = ["区分呼吸性/代谢性", "过快纠正可致低钾"]

            elif indicator == "EF":
                rec["intervention"] = "正性肌力支持 + 优化灌注条件"
                rec["drug"] = "Dobutamine↑ / Levosimendan / 检查冠脉灌注"
                rec["dose"] = "多巴酚丁胺 2-10 μg/min; Levosimendan 45μg/kg bolus"
                rec["reasoning"] = [
                    f"Step 1 - 观察: EF={value:.0f}%",
                    "Step 2 - 分析: EF是供心收缩功能核心评估指标",
                    "Step 3 - 机制: EF↓→可能缺血再灌注损伤/心肌保护不良",
                    "Step 4 - 干预: Dobutamine增强收缩力; 检查冠脉灌注/温度是否到位",
                    "Step 5 - 预期: EF改善"
                ]
                rec["monitoring"] = ["EF: 持续", "dP/dt: 持续", "冠脉流量: 持续"]
                rec["caution"] = ["EF持续<40%且无改善→评估器官可用性"]

            elif indicator == "CI":
                rec["intervention"] = "正性肌力+容量优化"
                rec["drug"] = "Dobutamine / 米力农"
                rec["dose"] = "Dobutamine 5-10 μg/kg/min"
                rec["reasoning"] = [
                    f"Step 1 - 观察: CI={value:.1f} L/min/m²",
                    "Step 2 - 分析: 低CI导致组织灌注下降",
                    "Step 3 - 机制: 移植心功能不全→CO↓→器官灌注↓",
                    "Step 4 - 干预: 正性肌力药增强心肌收缩力",
                    "Step 5 - 预期: CI提升至>2.5 L/min/m²"
                ]
                rec["monitoring"] = ["CI: 持续", "CVP: 持续"]
                rec["caution"] = ["注意心律失常", "严重→VA-ECMO评估"]

            else:
                rec["intervention"] = "对症处理"
                rec["drug"] = "根据具体情况"
                rec["dose"] = "-"
                rec["reasoning"] = [f"指标{indicator}异常，需进一步评估"]
                rec["monitoring"] = [f"{indicator}: 每30分钟"]
                rec["caution"] = ["密切观察"]

            rec["confidence"] = 85 if status == "critical" else 75
            recommendations.append(rec)

    # 按优先级排序: setpoints先，readouts后; critical先，warning后
    def sort_key(r):
        is_setpoint = r["indicator"] in SETPOINT_CONFIG
        is_critical = r["status"] == "critical"
        return (0 if is_critical else 1, 0 if is_setpoint else 1)
    recommendations.sort(key=sort_key)
    return recommendations

def render_strategy_panel(recommendations: List[Dict]):
    """渲染策略推荐面板"""
    if not recommendations:
        st.success("✅ 所有指标在正常范围内，无需特殊干预")
        return

    critical_count = sum(1 for r in recommendations if r["status"] == "critical")
    warning_count = len(recommendations) - critical_count

    st.markdown(f"### 💊 策略推荐 (🔴 {critical_count} 危急 | 🟡 {warning_count} 警告)")

    for rec in recommendations:
        status_icon = "🔴" if rec["status"] == "critical" else "🟡"
        priority_color = "#ff4d4f" if rec["priority"] == "URGENT" else "#1890ff"

        with st.expander(f"{status_icon} {rec['name']}: {rec['current']:.1f} → {rec['target']:.1f} {rec['unit']} | {rec['priority']}", expanded=rec["status"]=="critical"):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**干预措施:** {rec['intervention']}")
                st.markdown(f"""
                <div style="background: #e6f7ff; border-left: 4px solid #1890ff; padding: 0.8rem; margin: 0.5rem 0; border-radius: 0 4px 4px 0;">
                    💉 <strong>{rec['drug']}</strong><br/>
                    剂量: {rec['dose']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**📋 CoT推理链:**")
                for step in rec.get("reasoning", []):
                    st.markdown(f"- {step}")

            with col2:
                st.markdown("**📊 监测要点:**")
                for item in rec.get("monitoring", []):
                    st.markdown(f"- {item}")

                st.markdown("**⚠️ 注意事项:**")
                for item in rec.get("caution", []):
                    st.markdown(f"- {item}")

                confidence = rec.get("confidence", 80)
                st.markdown(f"**置信度:** {confidence}%")
                st.progress(confidence / 100)

def render_evidence_panel():
    """渲染证据溯源面板"""
    st.markdown("### 🔬 证据溯源")

    tab1, tab2, tab3 = st.tabs(["📊 KG三元组", "🔍 Neo4j查询", "📚 临床指南"])

    with tab1:
        st.markdown("**来源: intervention_strategies.yaml**")
        evidence_triples = [
            ("MAP_Low", "first_check", "Volume_Status"),
            ("MAP_Low", "requires_intervention", "Vasopressor"),
            ("MAP_Low", "escalate_to", "ECMO_Evaluation"),
            ("Lactate_High", "indicates", "Tissue_Hypoxia"),
            ("Lactate_High", "requires", "Perfusion_Optimization"),
            ("K_High", "causes", "Arrhythmia_Risk"),
            ("K_High", "requires", "Potassium_Lowering"),
            ("CI_Low", "indicates", "Cardiac_Dysfunction"),
            ("CI_Low", "requires", "Inotrope_Support"),
            ("pH_Low", "affects", "Drug_Efficacy"),
            ("pH_Low", "requires", "Acid_Base_Correction")
        ]

        for s, p, o in evidence_triples:
            st.markdown(f"""
            <div style="background: #f0f5ff; border: 1px solid #adc6ff; padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 4px; font-family: monospace;">
                <span style="color: #1890ff;">{s}</span> ──<span style="color: #722ed1;">{p}</span>──► <span style="color: #52c41a;">{o}</span>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("**Neo4j Cypher 查询示例:**")
        st.code("""
MATCH (indicator:monitoring_indicator {name: 'MAP'})
-[r1:CAN_LEAD_TO]->(consequence)
RETURN indicator, r1, consequence

MATCH (symptom:symptom)-[r:TREATED_BY]->(treatment:treatment_regimen)
WHERE symptom.name CONTAINS 'hypotension'
RETURN symptom, treatment, r.dosage
        """, language="cypher")

        st.info("💡 连接Neo4j后可查询实时知识图谱证据")

    with tab3:
        st.markdown("**参考临床指南:**")
        guidelines = [
            {"name": "ISHLT 2014", "topic": "心脏移植受者血流动力学管理"},
            {"name": "EACTA 2019", "topic": "体外循环期间血压管理"},
            {"name": "STS 2021", "topic": "心脏手术围术期乳酸监测"}
        ]
        for g in guidelines:
            st.markdown(f"- **{g['name']}**: {g['topic']}")

def calculate_risk_level(data: Dict[str, float]) -> str:
    """计算整体风险等级"""
    critical_count = 0
    warning_count = 0

    for indicator, value in data.items():
        status, _ = get_status(indicator, value)
        if status == "critical":
            critical_count += 1
        elif status == "warning":
            warning_count += 1

    if critical_count >= 2:
        return "HIGH"
    elif critical_count >= 1 or warning_count >= 3:
        return "MEDIUM"
    else:
        return "LOW"

# =============================================================================
# 算法分析面板
# =============================================================================
def _extract_time_series(patient_data: Dict, selected_timepoint: str):
    """提取从baseline到当前时间点的时序数据"""
    all_tps = ["baseline", "60min", "120min", "180min", "240min"]
    all_mins = [0, 60, 120, 180, 240]
    idx = all_tps.index(selected_timepoint) + 1
    active_tps = all_tps[:idx]
    time_mins = all_mins[:idx]

    indicators = set()
    for tp in active_tps:
        indicators.update(patient_data.get(tp, {}).keys())

    series = {}
    for ind in indicators:
        vals = [patient_data.get(tp, {}).get(ind) for tp in active_tps]
        if all(v is not None for v in vals):
            series[ind] = vals
    return time_mins, series


def render_algorithm_panel(patient_data: Dict, current_data: Dict, selected_timepoint: str):
    """渲染智能算法分析面板"""
    if not ALGO_AVAILABLE:
        st.info("算法模块未加载")
        return

    st.markdown("### 🧠 智能算法分析")

    time_mins, time_series = _extract_time_series(patient_data, selected_timepoint)

    # 运行所有算法
    algo_results = run_all_algorithms(
        current_data=current_data,
        time_series=time_series,
        time_points=time_mins,
    )

    tab_srco, tab_cprs, tab_cusum, tab_lactate = st.tabs([
        "🎯 SRCO调控优化", "📊 CPRS风险评分", "📈 CUSUM早期预警", "🔬 乳酸轨迹预测"
    ])

    with tab_srco:
        _render_srco(algo_results.get("srco", []))

    with tab_cprs:
        _render_cprs(algo_results.get("cprs", {}))

    with tab_cusum:
        _render_cusum(algo_results.get("cusum", {}), time_mins)

    with tab_lactate:
        _render_lactate(algo_results.get("lactate_prediction", {}),
                        time_series.get("Lactate", []), time_mins)


def _render_srco(srco_results: List[Dict]):
    """渲染SRCO调控优化结果"""
    st.markdown("**SRCO算法**: 基于因果图反向推理，计算各Setpoint调控收益")

    if not srco_results:
        st.success("✅ 所有Readout正常，无需调控优化")
        return

    # 收益排名条形图
    if go:
        names = [f"{r['setpoint']} (P{r['priority']})" for r in srco_results[:8]]
        scores = [r['benefit_score'] for r in srco_results[:8]]
        colors = ['#ff4d4f' if s > 0.3 else '#faad14' if s > 0.1 else '#52c41a' for s in scores]

        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation='h',
            marker_color=colors,
            text=[f"{s:.3f}" for s in scores],
            textposition='outside'
        ))
        fig.update_layout(
            title="Setpoint调控收益排序 (Benefit Score)",
            xaxis_title="收益分数",
            yaxis=dict(autorange="reversed"),
            height=max(200, 40 * len(names)),
            margin=dict(l=140, r=50, t=40, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 详细推荐卡片
    for rec in srco_results[:5]:
        dir_emoji = "⬆️" if rec['direction'] == "increase" else "⬇️"
        with st.expander(f"{dir_emoji} {rec['setpoint']} | 收益={rec['benefit_score']:.4f} | 影响{rec['readout_count']}个Readout"):
            st.markdown(f"**建议方向:** {'增大' if rec['direction'] == 'increase' else '减小'}")
            st.markdown(f"**优先级:** P{rec['priority']}")
            st.markdown(f"**因果推理:** {rec['reasoning']}")
            if rec.get('affected_readouts'):
                st.markdown("**影响的Readout:**")
                for ar in rec['affected_readouts'][:5]:
                    st.caption(f"  → {ar['readout']}: 偏离{ar['deviation']:.0%}, 权重{ar['weight']}, {ar['mechanism']}")


def _render_cprs(cprs_result: Dict):
    """渲染CPRS综合风险评分"""
    st.markdown("**CPRS算法**: 多指标加权融合，计算0-100综合灌注风险分数")

    if not cprs_result:
        st.info("无法计算CPRS")
        return

    score = cprs_result.get('total_score', 0)
    level = cprs_result.get('risk_level', 'UNKNOWN')
    level_colors = {"LOW": "#52c41a", "MEDIUM": "#faad14", "HIGH": "#ff7a45", "CRITICAL": "#ff4d4f"}
    level_labels = {"LOW": "低风险", "MEDIUM": "中风险", "HIGH": "高风险", "CRITICAL": "极高风险"}
    color = level_colors.get(level, "#8c8c8c")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style="text-align:center; padding:1.2rem; background:{color}15; border:3px solid {color}; border-radius:15px;">
            <div style="font-size:0.9rem; color:{color}; font-weight:bold;">CPRS 综合风险</div>
            <div style="font-size:3.5rem; font-weight:bold; color:{color};">{score:.0f}</div>
            <div style="font-size:1.1rem; color:{color};">{level_labels.get(level, level)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 领域雷达图
        if go:
            domains = cprs_result.get('domain_scores', {})
            if domains:
                categories = [d['label'] for d in domains.values()]
                values = [d['score'] for d in domains.values()]
                categories.append(categories[0])
                values.append(values[0])

                fig = go.Figure(go.Scatterpolar(
                    r=values, theta=categories,
                    fill='toself',
                    fillcolor=f'{color}30',
                    line_color=color,
                    marker=dict(size=8)
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    height=280,
                    margin=dict(l=60, r=60, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    # 风险贡献排名
    if cprs_result.get('top_risks'):
        st.markdown("**风险贡献排名:**")
        for tr in cprs_result['top_risks'][:5]:
            risk_pct = tr['risk'] * 100
            st.markdown(f"- **{tr['indicator']}**: 风险={risk_pct:.0f}%, 贡献={tr['contribution']:.3f}")


def _render_cusum(cusum_results: Dict, time_mins: List[int]):
    """渲染CUSUM早期预警"""
    st.markdown("**CUSUM算法**: 累积和控制图，在阈值突破前检测趋势性恶化")

    if not cusum_results:
        st.info("需要至少2个时间点的数据才能进行CUSUM分析")
        return

    alarm_count = sum(1 for r in cusum_results.values() if r.get('alarm'))
    if alarm_count:
        st.error(f"⚠️ {alarm_count} 个指标触发CUSUM预警!")
    else:
        st.success("✅ 所有监测指标CUSUM趋势正常")

    # 状态网格
    cols = st.columns(min(len(cusum_results), 4) or 1)
    for i, (indicator, result) in enumerate(cusum_results.items()):
        with cols[i % 4]:
            alarm = result.get('alarm', False)
            severity = result.get('severity', 0)
            trend = result.get('trend', 'stable')
            trend_emoji = {"worsening": "📈⚠️", "improving": "📉✅", "stable": "➡️"}.get(trend, "❓")
            bg = "#ff4d4f" if alarm else "#faad14" if severity > 0.5 else "#52c41a"

            st.markdown(f"""
            <div style="background:{bg}18; border:2px solid {bg}; border-radius:8px; padding:0.6rem; text-align:center; margin-bottom:0.5rem;">
                <div style="font-weight:bold; font-size:0.85rem;">{indicator}</div>
                <div style="font-size:1.3rem;">{trend_emoji}</div>
                <div style="font-size:0.75rem;">严重度: {severity:.0%}</div>
            </div>
            """, unsafe_allow_html=True)

    # CUSUM曲线图（仅对有预警风险的指标）
    if go:
        alarming = {k: v for k, v in cusum_results.items() if v.get('severity', 0) > 0.3}
        if alarming:
            fig = make_subplots(rows=len(alarming), cols=1, shared_xaxes=True,
                                subplot_titles=list(alarming.keys()), vertical_spacing=0.12)
            for idx, (ind, result) in enumerate(alarming.items(), 1):
                cusum_upper = result.get('cusum_upper', [])
                cusum_lower = result.get('cusum_lower', [])
                h = result.get('threshold_h', 1)
                x_vals = time_mins[:len(cusum_upper)]

                fig.add_trace(go.Scatter(x=x_vals, y=cusum_upper, name=f'{ind} S⁺',
                                         line=dict(color='#ff4d4f', width=2)), row=idx, col=1)
                fig.add_trace(go.Scatter(x=x_vals, y=cusum_lower, name=f'{ind} S⁻',
                                         line=dict(color='#1890ff', width=2)), row=idx, col=1)
                fig.add_hline(y=h, line_dash="dash", line_color="red",
                              annotation_text="阈值h", row=idx, col=1)

            fig.update_layout(height=200 * len(alarming), showlegend=True,
                              margin=dict(l=50, r=20, t=30, b=30))
            st.plotly_chart(fig, use_container_width=True)


def _render_lactate(lactate_result: Dict, historical: List[float], time_mins: List[int]):
    """渲染乳酸轨迹预测"""
    st.markdown("**乳酸轨迹预测**: 指数衰减模型，预测乳酸清除速率和达标时间")

    if not lactate_result or lactate_result.get('trend') == 'unknown':
        st.info("需要乳酸时间序列数据才能预测")
        return

    quality = lactate_result.get('clearance_quality', 'unknown')
    quality_colors = {"good": "#52c41a", "marginal": "#faad14", "poor": "#ff7a45", "worsening": "#ff4d4f"}
    quality_labels = {"good": "良好", "marginal": "边缘", "poor": "不良", "worsening": "恶化"}
    color = quality_colors.get(quality, "#8c8c8c")

    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cr = lactate_result.get('clearance_rate', 0)
        st.metric("清除率", f"{cr:.1f}%/h",
                  delta="良好" if cr > 10 else "偏低",
                  delta_color="normal" if cr > 10 else "inverse")
    with col2:
        hl = lactate_result.get('half_life', 0)
        st.metric("半衰期", f"{hl:.0f} min" if hl > 0 else "N/A")
    with col3:
        pred_time = lactate_result.get('predicted_target_time')
        st.metric("预计达标", f"{pred_time:.0f} min" if pred_time and pred_time > 0 else "N/A")
    with col4:
        st.markdown(f"""
        <div style="text-align:center; padding:0.5rem; background:{color}20; border:2px solid {color}; border-radius:8px;">
            <div style="font-size:0.8rem;">清除质量</div>
            <div style="font-size:1.1rem; font-weight:bold; color:{color};">{quality_labels.get(quality, quality)}</div>
        </div>
        """, unsafe_allow_html=True)

    # 轨迹图
    if go and (historical or lactate_result.get('predicted_values')):
        fig = go.Figure()

        # 历史实测值
        if historical and time_mins:
            fig.add_trace(go.Scatter(
                x=time_mins, y=historical,
                mode='lines+markers', name='实测值',
                line=dict(color='#1890ff', width=3),
                marker=dict(size=10)
            ))

        # 预测轨迹
        predicted = lactate_result.get('predicted_values', [])
        if predicted:
            pred_x = [p['time_min'] for p in predicted]
            pred_y = [p['predicted_lactate'] for p in predicted]
            fig.add_trace(go.Scatter(
                x=pred_x, y=pred_y,
                mode='lines', name='预测轨迹',
                line=dict(color='#722ed1', width=2, dash='dash'),
                fill='tozeroy', fillcolor='rgba(114,46,209,0.05)'
            ))

        # 目标线
        target = lactate_result.get('target', 2.0)
        fig.add_hline(y=target, line_dash="dot", line_color="green",
                      annotation_text=f"目标 {target}")
        fig.add_hline(y=5.0, line_dash="dash", line_color="red",
                      annotation_text="OCS可接受上限 5.0")

        fig.update_layout(
            title="乳酸清除轨迹 & 预测",
            xaxis_title="时间 (min)",
            yaxis_title="Lactate (mmol/L)",
            height=320,
            margin=dict(l=50, r=20, t=40, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # 结论
    msg = lactate_result.get('message', '')
    if msg:
        st.info(f"💡 {msg}")


# =============================================================================
# 主应用
# =============================================================================
def main():
    # 加载数据
    config = load_config()
    demo_data = get_demo_data()

    # 初始化Neo4j和LLM
    neo4j_kg = init_neo4j()
    _llm_result = init_llm()
    llm_client = _llm_result[0]
    _llm_err = _llm_result[1]
    neo4j_connected = neo4j_kg is not None
    llm_configured = llm_client is not None

    # 侧边栏
    with st.sidebar:
        st.markdown("## ⚙️ 控制面板")

        # 样本选择
        st.markdown("### 📋 样本选择")
        sample_ids = list(demo_data.keys())
        selected_sample = st.selectbox("选择病例", sample_ids, index=1)

        # 时间点选择
        timepoints = ["baseline", "60min", "120min", "180min", "240min"]
        selected_timepoint = st.selectbox("选择时间点", timepoints, index=1)

        # 患者信息
        patient = demo_data[selected_sample]
        st.markdown("### 👤 患者信息")
        st.markdown(f"- **年龄:** {patient.get('age', 'N/A')} 岁")
        st.markdown(f"- **性别:** {patient.get('gender', 'N/A')}")
        outcome = patient.get("outcome", "unknown")
        outcome_color = "green" if outcome == "success" else "red"
        st.markdown(f"- **结局:** <span style='color:{outcome_color}'>{outcome}</span>", unsafe_allow_html=True)

        # 指标选择
        st.markdown("### 📈 趋势图指标")
        st.caption("🎛 调控参数")
        setpoint_list = list(SETPOINT_CONFIG.keys())
        st.caption("🩸 血气观测")
        blood_gas_list = list(READOUT_BLOOD_GAS_CONFIG.keys())
        st.caption("💓 心功能观测")
        function_list = list(READOUT_FUNCTION_CONFIG.keys())
        available_indicators = setpoint_list + blood_gas_list + function_list + list(TRANSPLANT_CONFIG.keys())
        selected_indicators = st.multiselect(
            "选择显示的指标",
            available_indicators,
            default=["Flow", "Temperature", "Lactate", "K_A"]
        )

        # Agent模式
        st.markdown("### 🤖 Agent系统")
        agent_mode = st.toggle("启用多Agent管线", value=AGENT_AVAILABLE, disabled=not AGENT_AVAILABLE)

        # 系统状态
        st.markdown("### 🔌 系统状态")
        st.markdown(f"- **后端模块:** {'✅ 已加载' if BACKEND_AVAILABLE else '⚠️ 部分加载'}")
        st.markdown(f"- **Agent系统:** {'✅ 可用' if AGENT_AVAILABLE else '⚪ 不可用'}")
        st.markdown(f"- **Neo4j:** {'✅ 已连接' if neo4j_connected else '⚪ 未连接'}")
        llm_model = os.getenv("LLM_MODEL", "N/A")
        if llm_configured:
            st.markdown(f"- **LLM:** ✅ {llm_model}")
        else:
            st.markdown(f"- **LLM:** ⚪ 未配置")
            if _llm_err:
                st.caption(f"  ({_llm_err})")

        # Agent详情
        if agent_mode and AGENT_AVAILABLE:
            with st.expander("Agent工具详情"):
                bus = EventBus()
                ps = PatientState()
                agent_info = {
                    "Monitor (感知)": MonitorAgent(bus, ps),
                    "Diagnosis (分析)": DiagnosisAgent(bus, ps),
                    "Strategy (决策)": StrategyAgent(bus, ps),
                    "Knowledge (知识)": KnowledgeAgent(bus, ps),
                    "Communication (交互)": CommunicationAgent(bus, ps),
                }
                for name, agent in agent_info.items():
                    tools = agent.get_tool_list()
                    st.markdown(f"**{name}**: {len(tools)} tools")
                    for t in tools:
                        st.caption(f"  - {t['name']}: {t['description'][:40]}...")

        st.markdown("---")
        st.markdown("*HTTG Perfusion Monitor v2.0 (Multi-Agent)*")

    # 获取当前数据
    current_data = patient.get(selected_timepoint, {})
    baseline_data = patient.get("baseline", {})

    # =================================================================
    # Agent Pipeline模式
    # =================================================================
    agent_state = None
    if agent_mode and AGENT_AVAILABLE:
        @st.cache_resource
        def init_agent_system(_neo4j=None, _llm=None):
            bus = EventBus()
            ps = PatientState()
            monitor = MonitorAgent(bus, ps)
            diagnosis = DiagnosisAgent(bus, ps)
            strategy_ag = StrategyAgent(bus, ps)
            knowledge = KnowledgeAgent(bus, ps)
            comm = CommunicationAgent(bus, ps)
            # 将LLM和Neo4j传递给策略Agent
            if _llm or _neo4j:
                strategy_ag.set_llm_and_neo4j(llm=_llm, neo4j_connector=_neo4j)
            coordinator = CoordinatorAgent(
                bus, ps,
                monitor=monitor, diagnosis=diagnosis,
                strategy=strategy_ag, knowledge=knowledge,
                communication=comm
            )
            return coordinator

        coordinator = init_agent_system(_neo4j=neo4j_kg, _llm=llm_client)
        agent_state = coordinator.run_pipeline(
            measurements=current_data,
            sample_id=selected_sample,
            timestamp_min=int(selected_timepoint.replace("min", "").replace("baseline", "0"))
        )
        risk_level = agent_state.risk_level
    else:
        risk_level = calculate_risk_level(current_data)

    # Header
    render_header(selected_sample, selected_timepoint, risk_level)
    st.markdown("---")

    # 状态卡片
    st.markdown("### 📊 实时指标状态")
    render_status_cards(current_data, baseline_data)

    st.markdown("---")

    # 主内容区域
    col_left, col_right = st.columns([3, 2])

    with col_left:
        # 时序趋势图
        st.markdown("### 📈 时序趋势监测")
        if selected_indicators:
            render_time_series(patient, selected_indicators)
        else:
            st.info("请在侧边栏选择要显示的指标")

    with col_right:
        # 策略推荐 - Agent模式 vs 传统模式
        if agent_state and agent_state.strategies:
            st.markdown("### 🤖 Agent策略推荐")
            for i, strat in enumerate(agent_state.strategies[:5], 1):
                severity_emoji = {"critical": "🔴", "red_line": "🔴", "warning": "🟡"}.get(strat.severity, "🔵")
                with st.expander(f"{severity_emoji} [{i}] {strat.indicator}: {strat.action}", expanded=(i<=2)):
                    if strat.drug:
                        st.markdown(f"**药物:** {strat.drug}")
                    if strat.dose:
                        st.markdown(f"**剂量:** {strat.dose}")
                    if strat.reasoning_chain:
                        st.markdown("**CoT推理链:**")
                        for step in strat.reasoning_chain:
                            st.markdown(f"- {step}")
                    if strat.evidence:
                        st.markdown("**证据:**")
                        for ev in strat.evidence[:3]:
                            st.caption(f"- {ev}")
                    st.progress(strat.confidence, text=f"置信度: {strat.confidence:.0%}")

            # 安全警告
            if agent_state.drug_conflicts:
                st.error("**药物冲突警告:**")
                for c in agent_state.drug_conflicts:
                    st.markdown(f"- {c}")
            if agent_state.safety_warnings:
                st.warning("**安全提示:**")
                for w in agent_state.safety_warnings[:5]:
                    st.markdown(f"- {w}")
        else:
            recommendations = get_strategy_recommendations(current_data, baseline_data)
            render_strategy_panel(recommendations)

    st.markdown("---")

    # 智能算法分析面板
    if ALGO_AVAILABLE:
        render_algorithm_panel(patient, current_data, selected_timepoint)
        st.markdown("---")

    # Agent处理日志 / 证据面板
    if agent_state:
        tab_evidence, tab_log, tab_agents = st.tabs(["📋 证据溯源", "📜 Agent处理日志", "🤖 Agent状态"])
        with tab_evidence:
            if agent_state.evidence_pool:
                st.markdown(f"**共收集 {len(agent_state.evidence_pool)} 条证据**")
                for ev in agent_state.evidence_pool[:10]:
                    source = ev.get("source", "unknown")
                    ev_type = ev.get("type", "")
                    score = ev.get("evidence_score", 0)
                    strength = ev.get("strength", "")
                    with st.expander(f"[{source}] {ev_type} (score: {score:.2f}, {strength})"):
                        st.json(ev.get("data", {}))
            else:
                st.info("无证据数据")
        with tab_log:
            for log_entry in agent_state.processing_log:
                st.text(log_entry)
        with tab_agents:
            if agent_mode and AGENT_AVAILABLE:
                agent_status = coordinator.get_agent_status()
                for name, info in agent_status.items():
                    status = info["status"]
                    tool_count = info["tool_count"]
                    emoji = "✅" if status != "unavailable" else "⚪"
                    st.markdown(f"{emoji} **{name}**: {tool_count} tools")
                    for t in info["tools"]:
                        st.caption(f"  - `{t['name']}`: {t['description'][:50]}")
    else:
        render_evidence_panel()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        neo4j_status = "✅ Connected" if neo4j_connected else "⚪ Disconnected"
        st.caption(f"📡 Neo4j: {neo4j_status}")
    with col3:
        llm_status = f"✅ {os.getenv('LLM_MODEL', 'Active')}" if llm_configured else "⚪ Not Configured"
        st.caption(f"🤖 LLM: {llm_status}")

if __name__ == "__main__":
    main()
