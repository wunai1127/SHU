#!/usr/bin/env python3
"""
后端配置页面 - 配置Neo4j、LLM和阈值参数
"""

import streamlit as st
import yaml
from pathlib import Path

st.set_page_config(page_title="后端配置", page_icon="⚙️", layout="wide")

st.title("⚙️ 后端配置")

# =============================================================================
# Neo4j 配置
# =============================================================================
st.markdown("## 🔗 Neo4j 知识图谱配置")

with st.expander("Neo4j 连接设置", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        neo4j_uri = st.text_input("Neo4j URI", "bolt://localhost:7687")
        neo4j_user = st.text_input("用户名", "neo4j")

    with col2:
        neo4j_password = st.text_input("密码", type="password")
        neo4j_database = st.text_input("数据库名", "neo4j")

    if st.button("🔌 测试连接", key="test_neo4j"):
        with st.spinner("测试连接中..."):
            try:
                # 这里可以添加实际的Neo4j连接测试
                st.warning("⚠️ Neo4j 连接测试功能需要安装 neo4j 驱动")
                st.code("pip install neo4j")
            except Exception as e:
                st.error(f"❌ 连接失败: {e}")

    # 保存配置
    if st.button("💾 保存 Neo4j 配置"):
        config = {
            "uri": neo4j_uri,
            "user": neo4j_user,
            "password": neo4j_password,
            "database": neo4j_database
        }
        st.success("✅ 配置已保存到会话")
        st.session_state["neo4j_config"] = config

# =============================================================================
# LLM 配置
# =============================================================================
st.markdown("---")
st.markdown("## 🤖 LLM 配置")

with st.expander("大语言模型设置", expanded=True):
    llm_provider = st.selectbox(
        "LLM 提供商",
        ["OpenAI (GPT-4)", "Anthropic (Claude)", "本地模型", "无 (使用规则引擎)"]
    )

    col1, col2 = st.columns(2)

    with col1:
        if "OpenAI" in llm_provider:
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("模型", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"])
        elif "Anthropic" in llm_provider:
            api_key = st.text_input("Anthropic API Key", type="password")
            model = st.selectbox("模型", ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"])
        elif "本地" in llm_provider:
            api_key = ""
            local_endpoint = st.text_input("本地端点", "http://localhost:11434/api/generate")
            model = st.text_input("模型名称", "llama2")
        else:
            api_key = ""
            model = "rule-based"
            st.info("将使用基于规则的策略引擎，无需 LLM")

    with col2:
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1500)

    if st.button("💾 保存 LLM 配置"):
        llm_config = {
            "provider": llm_provider,
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        st.session_state["llm_config"] = llm_config
        st.success("✅ LLM 配置已保存")

# =============================================================================
# 阈值配置
# =============================================================================
st.markdown("---")
st.markdown("## 📏 阈值配置")

# 加载当前配置
config_path = Path(__file__).parent.parent / "config" / "thresholds.yaml"

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as f:
        thresholds_config = yaml.safe_load(f)
    st.success("✅ 已加载 thresholds.yaml")
else:
    thresholds_config = {}
    st.warning("⚠️ 未找到 thresholds.yaml")

with st.expander("指标阈值设置", expanded=True):
    st.markdown("### 核心指标阈值")

    indicators = {
        "MAP": {"name": "平均动脉压", "unit": "mmHg", "default_target": (65, 90), "default_red": 50, "default_critical": 60},
        "Lactate": {"name": "乳酸", "unit": "mmol/L", "default_target": (0, 4.0), "default_red": 6.0, "default_critical": 4.0},
        "K_A": {"name": "动脉血钾", "unit": "mmol/L", "default_target": (3.5, 5.0), "default_red": 6.0, "default_critical": 5.5},
        "CI": {"name": "心指数", "unit": "L/min/m²", "default_target": (2.2, 4.0), "default_red": 1.8, "default_critical": 2.0},
        "CvO2": {"name": "静脉血氧含量", "unit": "mL/dL", "default_target": (12, 16), "default_red": 8, "default_critical": 10},
        "pH": {"name": "动脉pH", "unit": "", "default_target": (7.35, 7.45), "default_red": 7.20, "default_critical": 7.30}
    }

    updated_thresholds = {}

    for ind, cfg in indicators.items():
        st.markdown(f"**{cfg['name']} ({ind}) - {cfg['unit']}**")

        cols = st.columns(4)
        with cols[0]:
            target_low = st.number_input(
                f"目标下限",
                value=float(cfg['default_target'][0]),
                key=f"{ind}_target_low"
            )
        with cols[1]:
            target_high = st.number_input(
                f"目标上限",
                value=float(cfg['default_target'][1]),
                key=f"{ind}_target_high"
            )
        with cols[2]:
            critical = st.number_input(
                f"警告阈值",
                value=float(cfg['default_critical']),
                key=f"{ind}_critical"
            )
        with cols[3]:
            red_line = st.number_input(
                f"红线阈值",
                value=float(cfg['default_red']),
                key=f"{ind}_red"
            )

        updated_thresholds[ind] = {
            "target": (target_low, target_high),
            "critical": critical,
            "red_line": red_line
        }

        st.markdown("---")

    if st.button("💾 保存阈值配置", type="primary"):
        st.session_state["thresholds"] = updated_thresholds
        st.success("✅ 阈值配置已保存到会话")

        # 可选：写入文件
        # with open(config_path, 'w', encoding='utf-8') as f:
        #     yaml.dump(updated_thresholds, f, allow_unicode=True)

# =============================================================================
# 当前配置预览
# =============================================================================
st.markdown("---")
st.markdown("## 📋 当前配置预览")

config_tabs = st.tabs(["Neo4j", "LLM", "阈值"])

with config_tabs[0]:
    if "neo4j_config" in st.session_state:
        st.json(st.session_state["neo4j_config"])
    else:
        st.info("暂未配置")

with config_tabs[1]:
    if "llm_config" in st.session_state:
        config = st.session_state["llm_config"].copy()
        config["api_key"] = "***" if config.get("api_key") else ""
        st.json(config)
    else:
        st.info("暂未配置")

with config_tabs[2]:
    if "thresholds" in st.session_state:
        st.json(st.session_state["thresholds"])
    else:
        st.info("使用默认配置")
