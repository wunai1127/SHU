#!/usr/bin/env python3
"""
数据管理页面 - 上传、查看和管理患者数据
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import yaml

st.set_page_config(page_title="数据管理", page_icon="📊", layout="wide")

st.title("📊 数据管理")

# =============================================================================
# 数据上传
# =============================================================================
st.markdown("## 📤 数据上传")

tab1, tab2, tab3 = st.tabs(["JSON 上传", "CSV 上传", "手动输入"])

with tab1:
    st.markdown("上传符合格式的 JSON 数据文件")
    uploaded_json = st.file_uploader("选择 JSON 文件", type=["json"], key="json_upload")

    if uploaded_json:
        try:
            data = json.load(uploaded_json)
            st.success(f"✅ 成功加载 {len(data)} 条记录")
            st.json(data)
        except Exception as e:
            st.error(f"❌ 解析错误: {e}")

    st.markdown("**JSON 格式示例:**")
    st.code("""
{
    "HTX-001": {
        "baseline": {"MAP": 45, "Lactate": 3.5, "K_A": 4.2, ...},
        "60min": {"MAP": 52, "Lactate": 3.0, "K_A": 4.0, ...},
        "120min": {...},
        "outcome": "success",
        "age": 45,
        "gender": "M"
    },
    "HTX-002": {...}
}
    """, language="json")

with tab2:
    st.markdown("上传 CSV 格式数据（宽表或长表）")
    uploaded_csv = st.file_uploader("选择 CSV 文件", type=["csv"], key="csv_upload")

    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            st.success(f"✅ 成功加载 {len(df)} 行, {len(df.columns)} 列")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"❌ 解析错误: {e}")

    st.markdown("**CSV 格式示例 (宽表):**")
    st.code("""
sample_id,timepoint,MAP,Lactate,K_A,SvO2,CI,CvO2,HR,pH,outcome
HTX-001,baseline,45,3.5,4.2,82,2.1,8.5,88,7.38,success
HTX-001,60min,52,3.0,4.0,78,2.3,9.2,85,7.40,success
HTX-002,baseline,38,4.8,5.5,85,1.8,7.2,105,7.28,failure
    """)

with tab3:
    st.markdown("手动输入单个时间点数据")

    col1, col2 = st.columns(2)

    with col1:
        sample_id = st.text_input("样本ID", "HTX-NEW")
        timepoint = st.selectbox("时间点", ["baseline", "60min", "120min", "180min", "240min"])
        outcome = st.selectbox("结局", ["unknown", "success", "failure"])

    with col2:
        age = st.number_input("年龄", min_value=0, max_value=100, value=50)
        gender = st.selectbox("性别", ["M", "F"])

    st.markdown("### 指标值")
    cols = st.columns(4)

    indicators = {
        "MAP": (0, 200, 65, "mmHg"),
        "Lactate": (0.0, 20.0, 2.0, "mmol/L"),
        "SvO2": (0, 100, 70, "%"),
        "K_A": (2.0, 10.0, 4.2, "mmol/L"),
        "CI": (0.0, 10.0, 2.5, "L/min/m²"),
        "CvO2": (0.0, 20.0, 12.0, "mL/dL"),
        "HR": (0, 200, 80, "bpm"),
        "pH": (6.8, 7.8, 7.40, "")
    }

    values = {}
    for i, (ind, (min_v, max_v, default, unit)) in enumerate(indicators.items()):
        with cols[i % 4]:
            values[ind] = st.number_input(
                f"{ind} ({unit})" if unit else ind,
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(default),
                step=0.1 if isinstance(default, float) else 1.0
            )

    if st.button("💾 保存数据", type="primary"):
        data_entry = {
            sample_id: {
                timepoint: values,
                "outcome": outcome,
                "age": age,
                "gender": gender
            }
        }
        st.success("✅ 数据已保存到会话")
        st.json(data_entry)

# =============================================================================
# 当前数据查看
# =============================================================================
st.markdown("---")
st.markdown("## 📋 当前数据")

# 加载本地数据
data_file = Path(__file__).parent.parent / "neo4j_query_table_data_2026-1-26.json"
if data_file.exists():
    with open(data_file, 'r', encoding='utf-8') as f:
        local_data = json.load(f)
    st.success(f"✅ 本地数据已加载: {len(local_data)} 条记录")

    # 转换为表格显示
    records = []
    for item in local_data:
        records.append({
            "样本ID": item.get("SampleNo", "N/A"),
            "MAP": item.get("MAP_value", "N/A"),
            "Lactate": item.get("Lactate_value", "N/A"),
            "SvO2": item.get("SvO2_value", "N/A"),
            "K_A": item.get("K_A_value", "N/A"),
            "时间点": item.get("timepoint", "N/A")
        })

    df = pd.DataFrame(records)
    st.dataframe(df, use_container_width=True)
else:
    st.info("📂 暂无本地数据文件")

# =============================================================================
# 数据导出
# =============================================================================
st.markdown("---")
st.markdown("## 📥 数据导出")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 导出为 JSON")
    if st.button("⬇️ 导出 JSON"):
        st.info("数据将导出为 JSON 格式")

with col2:
    st.markdown("### 导出为 CSV")
    if st.button("⬇️ 导出 CSV"):
        st.info("数据将导出为 CSV 格式")
