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
    """生成演示数据"""
    return {
        "HTX-28": {
            "baseline": {"MAP": 36, "Lactate": 2.8, "SvO2": 82, "K_A": 4.2, "CI": 2.1, "CvO2": 8.5, "HR": 88, "pH": 7.38},
            "60min": {"MAP": 45, "Lactate": 3.5, "SvO2": 78, "K_A": 4.8, "CI": 2.3, "CvO2": 9.2, "HR": 92, "pH": 7.35},
            "120min": {"MAP": 50, "Lactate": 3.2, "SvO2": 75, "K_A": 4.5, "CI": 2.5, "CvO2": 10.1, "HR": 85, "pH": 7.37},
            "180min": {"MAP": 58, "Lactate": 2.5, "SvO2": 72, "K_A": 4.3, "CI": 2.7, "CvO2": 11.5, "HR": 82, "pH": 7.40},
            "240min": {"MAP": 65, "Lactate": 2.0, "SvO2": 70, "K_A": 4.1, "CI": 2.9, "CvO2": 12.8, "HR": 78, "pH": 7.42},
            "outcome": "success",
            "age": 45,
            "gender": "M"
        },
        "HTX-36": {
            "baseline": {"MAP": 45, "Lactate": 3.9, "SvO2": 85, "K_A": 5.8, "CI": 1.9, "CvO2": 7.5, "HR": 105, "pH": 7.28},
            "60min": {"MAP": 51, "Lactate": 4.9, "SvO2": 82, "K_A": 6.2, "CI": 2.0, "CvO2": 8.0, "HR": 110, "pH": 7.25},
            "120min": {"MAP": 51, "Lactate": 4.1, "SvO2": 80, "K_A": 5.9, "CI": 2.1, "CvO2": 8.5, "HR": 108, "pH": 7.27},
            "180min": {"MAP": 48, "Lactate": 5.2, "SvO2": 78, "K_A": 6.5, "CI": 1.8, "CvO2": 7.8, "HR": 115, "pH": 7.22},
            "240min": {"MAP": 42, "Lactate": 6.8, "SvO2": 75, "K_A": 7.1, "CI": 1.5, "CvO2": 6.5, "HR": 125, "pH": 7.18},
            "outcome": "failure",
            "age": 58,
            "gender": "M"
        },
        "HTX-42": {
            "baseline": {"MAP": 52, "Lactate": 2.5, "SvO2": 78, "K_A": 4.0, "CI": 2.4, "CvO2": 10.5, "HR": 82, "pH": 7.40},
            "60min": {"MAP": 58, "Lactate": 2.2, "SvO2": 75, "K_A": 4.2, "CI": 2.6, "CvO2": 11.2, "HR": 78, "pH": 7.42},
            "120min": {"MAP": 65, "Lactate": 1.8, "SvO2": 72, "K_A": 4.1, "CI": 2.8, "CvO2": 12.0, "HR": 75, "pH": 7.43},
            "180min": {"MAP": 70, "Lactate": 1.5, "SvO2": 70, "K_A": 4.0, "CI": 3.0, "CvO2": 13.2, "HR": 72, "pH": 7.44},
            "240min": {"MAP": 72, "Lactate": 1.2, "SvO2": 68, "K_A": 3.9, "CI": 3.2, "CvO2": 14.0, "HR": 70, "pH": 7.45},
            "outcome": "success",
            "age": 38,
            "gender": "F"
        }
    }

# =============================================================================
# 指标配置
# =============================================================================
INDICATOR_CONFIG = {
    # 原有灌注指标
    "MAP": {"name": "平均动脉压", "unit": "mmHg", "target": (65, 90), "red_line": 50, "critical": 60},
    "Lactate": {"name": "乳酸", "unit": "mmol/L", "target": (0, 4.0), "red_line": 6.0, "critical": 4.0, "higher_is_worse": True},
    "SvO2": {"name": "混合静脉血氧饱和度", "unit": "%", "target": (65, 80), "red_line": None, "critical": None},
    "K_A": {"name": "动脉血钾", "unit": "mmol/L", "target": (3.5, 5.0), "red_line": 6.0, "critical": 5.5, "higher_is_worse": True},
    "CI": {"name": "心指数", "unit": "L/min/m²", "target": (2.2, 4.0), "red_line": 1.8, "critical": 2.0},
    "CvO2": {"name": "静脉血氧含量", "unit": "mL/dL", "target": (12, 16), "red_line": 8, "critical": 10},
    "HR": {"name": "心率", "unit": "bpm", "target": (60, 100), "red_line": None, "critical": None},
    "pH": {"name": "动脉pH", "unit": "", "target": (7.35, 7.45), "red_line": 7.20, "critical": 7.30},
    # 移植共识新增指标
    "PVR": {"name": "肺血管阻力", "unit": "Wood", "target": (0.5, 2.5), "red_line": 5.0, "critical": 4.0, "higher_is_worse": True},
    "TPG": {"name": "跨肺压差", "unit": "mmHg", "target": (5, 12), "red_line": 15, "critical": 14, "higher_is_worse": True},
    "PASP": {"name": "肺动脉收缩压", "unit": "mmHg", "target": (15, 40), "red_line": 70, "critical": 50, "higher_is_worse": True},
    "Creatinine": {"name": "肌酐", "unit": "mg/dL", "target": (0.5, 1.5), "red_line": 2.0, "critical": 1.7, "higher_is_worse": True},
    "GFR": {"name": "肾小球滤过率", "unit": "mL/min", "target": (60, 120), "red_line": 30, "critical": 60},
    "EF": {"name": "射血分数", "unit": "%", "target": (50, 70), "red_line": 30, "critical": 50},
    "Bilirubin": {"name": "胆红素", "unit": "mg/dL", "target": (0.1, 1.2), "red_line": 2.5, "critical": 2.0, "higher_is_worse": True},
}

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

def render_status_cards(data: Dict[str, float], baseline: Dict[str, float]):
    """渲染状态卡片"""
    # 选择主要指标显示
    main_indicators = ["MAP", "Lactate", "K_A", "CI", "CvO2", "pH", "HR", "SvO2"]

    cols = st.columns(4)
    for i, indicator in enumerate(main_indicators[:8]):
        with cols[i % 4]:
            value = data.get(indicator, 0)
            baseline_val = baseline.get(indicator, value)
            config = INDICATOR_CONFIG.get(indicator, {})
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

            st.markdown(f"""
            <div style="background: {bg_color}; padding: 1rem; border-radius: 10px; text-align: center; color: white; margin-bottom: 0.5rem;">
                <div style="font-size: 0.9rem; opacity: 0.9;">{icon} {config.get('name', indicator)}</div>
                <div style="font-size: 2rem; font-weight: bold; margin: 0.3rem 0;">{value:.1f}</div>
                <div style="font-size: 0.8rem; opacity: 0.8;">{config.get('unit', '')}</div>
                <div style="font-size: 0.75rem; margin-top: 0.3rem;">
                    {trend} {abs(change):.1f} ({change_pct:+.1f}%)
                </div>
                <div style="font-size: 0.7rem; opacity: 0.7;">
                    目标: {config.get('target', (0,0))[0]}-{config.get('target', (0,0))[1]}
                </div>
            </div>
            """, unsafe_allow_html=True)

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
    """获取策略推荐"""
    recommendations = []

    # 检查每个指标
    for indicator, value in data.items():
        config = INDICATOR_CONFIG.get(indicator, {})
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

            # 根据指标添加具体干预措施
            if indicator == "MAP":
                rec["intervention"] = "血管活性药物支持"
                rec["drug"] = "去甲肾上腺素 (Norepinephrine)"
                rec["dose"] = "0.05-0.1 μg/kg/min，滴定至目标MAP"
                rec["reasoning"] = [
                    f"Step 1 - 观察: MAP={value:.1f} mmHg，低于组织灌注安全阈值",
                    "Step 2 - 分析: 低MAP导致冠脉灌注不足、组织缺氧",
                    "Step 3 - 机制: [Evidence-1] 低血压→器官灌注不足→MOF风险",
                    "Step 4 - 干预: 首选去甲肾上腺素，α受体激动提升血管张力",
                    "Step 5 - 预期: MAP提升至65-80 mmHg，改善组织灌注"
                ]
                rec["monitoring"] = ["MAP: 每5分钟", "HR: 每5分钟", "Lactate: 每30分钟"]
                rec["caution"] = ["注意容量状态", "高剂量升压药可致心律失常"]

            elif indicator == "Lactate":
                rec["intervention"] = "改善组织灌注/氧合"
                rec["drug"] = "优化血流动力学 + 纠正贫血"
                rec["dose"] = "目标Hb>10g/dL，优化CI"
                rec["reasoning"] = [
                    f"Step 1 - 观察: Lactate={value:.1f} mmol/L，提示组织缺氧或灌注不足",
                    "Step 2 - 分析: 乳酸堆积反映无氧代谢增加",
                    "Step 3 - 机制: [Evidence-2] 组织缺氧→无氧糖酵解→乳酸产生↑",
                    "Step 4 - 干预: 优化氧输送(DO2)，改善组织灌注",
                    "Step 5 - 预期: Lactate下降至<2 mmol/L"
                ]
                rec["monitoring"] = ["Lactate: 每30分钟", "ScvO2: 持续", "尿量: 每小时"]
                rec["caution"] = ["排除肝功能不全", "注意是否存在肠系膜缺血"]

            elif indicator == "K_A":
                rec["intervention"] = "降钾治疗"
                rec["drug"] = "胰岛素+葡萄糖 / 钙剂"
                rec["dose"] = "10U胰岛素 + 50mL 50%葡萄糖，葡萄糖酸钙10mL静推"
                rec["reasoning"] = [
                    f"Step 1 - 观察: K+={value:.1f} mmol/L，存在高钾血症",
                    "Step 2 - 分析: 高钾可致心律失常，T波高尖",
                    "Step 3 - 机制: [Evidence-3] 高钾→心肌细胞膜电位异常→心律失常",
                    "Step 4 - 干预: 钙剂稳定心肌膜，胰岛素促钾内移",
                    "Step 5 - 预期: K+降至4.0-4.5 mmol/L"
                ]
                rec["monitoring"] = ["K+: 每30分钟", "ECG: 持续", "血糖: 每30分钟"]
                rec["caution"] = ["注意低血糖风险", "高钾>6.5需紧急处理"]

            elif indicator == "CI":
                rec["intervention"] = "强心治疗"
                rec["drug"] = "多巴酚丁胺 / 米力农"
                rec["dose"] = "多巴酚丁胺 5-10 μg/kg/min"
                rec["reasoning"] = [
                    f"Step 1 - 观察: CI={value:.1f} L/min/m²，心输出量不足",
                    "Step 2 - 分析: 低CI导致组织灌注下降",
                    "Step 3 - 机制: [Evidence-4] 移植心功能不全→CO↓→器官灌注↓",
                    "Step 4 - 干预: 正性肌力药增强心肌收缩力",
                    "Step 5 - 预期: CI提升至>2.5 L/min/m²"
                ]
                rec["monitoring"] = ["CI: 持续", "CVP: 持续", "PCWP: 每小时"]
                rec["caution"] = ["注意心律失常", "避免过度增加心肌耗氧"]

            elif indicator == "pH":
                rec["intervention"] = "纠正酸碱平衡"
                rec["drug"] = "碳酸氢钠 / 优化通气"
                rec["dose"] = "NaHCO3根据BE计算，或调整呼吸机参数"
                rec["reasoning"] = [
                    f"Step 1 - 观察: pH={value:.2f}，存在酸中毒",
                    "Step 2 - 分析: 酸中毒影响心肌收缩力和药物效应",
                    "Step 3 - 机制: [Evidence-5] 酸中毒→心肌抑制+血管反应性↓",
                    "Step 4 - 干预: 根据类型选择碱化或通气调整",
                    "Step 5 - 预期: pH恢复至7.35-7.40"
                ]
                rec["monitoring"] = ["血气: 每30分钟", "电解质: 每小时"]
                rec["caution"] = ["区分代谢性/呼吸性酸中毒", "过快纠正可致低钾"]

            else:
                rec["intervention"] = "对症处理"
                rec["drug"] = "根据具体情况"
                rec["dose"] = "-"
                rec["reasoning"] = [f"指标{indicator}异常，需进一步评估"]
                rec["monitoring"] = [f"{indicator}: 每30分钟"]
                rec["caution"] = ["密切观察"]

            rec["confidence"] = 85 if status == "critical" else 75
            recommendations.append(rec)

    # 按优先级排序
    recommendations.sort(key=lambda x: 0 if x["priority"] == "URGENT" else 1)
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
        available_indicators = list(INDICATOR_CONFIG.keys())
        selected_indicators = st.multiselect(
            "选择显示的指标",
            available_indicators,
            default=["MAP", "Lactate", "K_A", "CI"]
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
