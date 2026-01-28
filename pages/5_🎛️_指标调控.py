#!/usr/bin/env python3
"""
指标调控页面 - 基于可调控性分类的灌注监测

设计理念：
- Setpoint (绿色): 可直接调控，显示"目标值"和"调控方法"
- Readout (橙色): 不可直接调控，显示"风险阈值"和"受哪些Setpoint影响"
- Injury Marker (红色): 只能监测，显示趋势
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 添加src目录
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="指标调控", page_icon="🎛️", layout="wide")

# 尝试导入后端模块
try:
    from indicator_manager import IndicatorManager, IndicatorType
    manager = IndicatorManager()
    BACKEND_AVAILABLE = True
except ImportError as e:
    st.warning(f"后端模块导入警告: {e}")
    BACKEND_AVAILABLE = False
    manager = None

st.title("🎛️ 指标调控面板")

st.markdown("""
**指标分类说明：**
- 🟢 **Setpoint (可直接调控)**: 通过设备/药物直接设定目标值
- 🟠 **Readout (可间接影响)**: 不可直接设定，需通过调整Setpoint来改变
- 🔴 **Injury Marker (只能监测)**: 反映损伤程度，无法干预
""")

if not BACKEND_AVAILABLE:
    st.error("后端模块未加载，无法显示指标信息")
    st.stop()

# =============================================================================
# Setpoints 区域
# =============================================================================
st.markdown("---")
st.markdown("## 🟢 Setpoints (可直接调控)")
st.markdown("*这些指标可以通过设备/药物直接设定目标值*")

setpoints = manager.get_all_setpoints()

# 按domain分组
domains = {}
for key, sp in setpoints.items():
    domain = sp.domain or "其他"
    if domain not in domains:
        domains[domain] = []
    domains[domain].append((key, sp))

# 显示每个domain
for domain, items in domains.items():
    with st.expander(f"📁 {domain} ({len(items)}项)", expanded=True):
        cols = st.columns(2)
        for i, (key, sp) in enumerate(items):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                            padding: 15px; border-radius: 10px; margin: 5px 0;
                            border-left: 4px solid #28a745;">
                    <h4 style="margin: 0; color: #155724;">{sp.name}</h4>
                    <p style="margin: 5px 0; color: #155724;">
                        <strong>目标:</strong> {sp.target_range[0]} - {sp.target_range[1]} {sp.unit}
                    </p>
                    <p style="margin: 5px 0; font-size: 0.9em; color: #155724;">
                        <strong>调控:</strong> {sp.control_method}
                    </p>
                    <p style="margin: 5px 0; font-size: 0.85em; color: #666;">
                        {sp.adjustment_direction or ''}
                    </p>
                </div>
                """, unsafe_allow_html=True)

# =============================================================================
# Readouts 区域
# =============================================================================
st.markdown("---")
st.markdown("## 🟠 Readouts (可间接影响)")
st.markdown("*这些指标不可直接设定，需要通过调整上方的Setpoint来改变*")

readouts = manager.get_all_readouts()

# 按domain分组
domains = {}
for key, rd in readouts.items():
    domain = rd.domain or "其他"
    if domain not in domains:
        domains[domain] = []
    domains[domain].append((key, rd))

for domain, items in domains.items():
    with st.expander(f"📁 {domain} ({len(items)}项)", expanded=True):
        for key, rd in items:
            dir_text = "⬆️越高风险越大" if rd.risk_direction.value == "higher_is_worse" else "⬇️越低风险越大"
            influenced = ", ".join(rd.influenced_by) if rd.influenced_by else "未定义"

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
                        padding: 15px; border-radius: 10px; margin: 5px 0;
                        border-left: 4px solid #ffc107;">
                <h4 style="margin: 0; color: #856404;">{rd.name}</h4>
                <p style="margin: 5px 0; color: #856404;">
                    <strong>风险阈值:</strong> {rd.risk_threshold} {rd.unit} ({dir_text})
                </p>
                <p style="margin: 5px 0; font-size: 0.9em; color: #856404;">
                    <strong>解读:</strong> {rd.interpretation}
                </p>
                <p style="margin: 5px 0; font-size: 0.85em; color: #856404;">
                    <strong>受影响于:</strong> {influenced}
                </p>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# Injury Markers 区域
# =============================================================================
st.markdown("---")
st.markdown("## 🔴 Injury Markers (只能监测)")
st.markdown("*这些指标反映损伤程度，无法直接干预，但可被Setpoint策略影响*")

for key, marker in manager.injury_markers.items():
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
                padding: 15px; border-radius: 10px; margin: 5px 0;
                border-left: 4px solid #dc3545;">
        <h4 style="margin: 0; color: #721c24;">{marker.name}</h4>
        <p style="margin: 5px 0; color: #721c24;">
            <strong>单位:</strong> {marker.unit}
        </p>
        <p style="margin: 5px 0; font-size: 0.9em; color: #721c24;">
            <strong>解读:</strong> {marker.interpretation}
        </p>
        <p style="margin: 5px 0; font-size: 0.85em; color: #721c24;">
            {'📈 趋势比绝对值更重要' if marker.trend_is_key else ''}
            {marker.notes if marker.notes else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# 实时分析
# =============================================================================
st.markdown("---")
st.markdown("## 🔍 实时指标分析")

st.markdown("输入当前测量值，系统将自动分析并给出调整建议")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Setpoints")
    input_setpoints = {}
    for key, sp in list(setpoints.items())[:5]:
        default = sp.typical_setpoint or (sp.target_range[0] + sp.target_range[1]) / 2
        input_setpoints[key] = st.number_input(
            f"{sp.name} ({sp.unit})",
            value=float(default),
            key=f"input_{key}"
        )

with col2:
    st.markdown("### Readouts")
    input_readouts = {}
    for key, rd in list(readouts.items())[:5]:
        default = rd.risk_threshold * 0.9 if rd.risk_direction.value == "higher_is_worse" else rd.risk_threshold * 1.1
        input_readouts[key] = st.number_input(
            f"{rd.name} ({rd.unit})",
            value=float(default),
            key=f"input_{key}"
        )

if st.button("🔍 分析并生成建议", type="primary"):
    # 合并所有输入
    all_measurements = {**input_setpoints, **input_readouts}

    # 分析
    analysis = manager.analyze_measurements(all_measurements)

    # 显示结果
    st.markdown("### 📊 分析结果")

    # Setpoint状态
    if analysis["setpoint_status"]:
        st.markdown("#### Setpoint 状态")
        for item in analysis["setpoint_status"]:
            if item["status"] == "on_target":
                st.success(f"✅ {item['message']}")
            else:
                st.warning(f"⚠️ {item['message']}")
                st.caption(f"   调控方法: {item['control_method']}")

    # Readout风险
    if analysis["readout_risks"]:
        st.markdown("#### ⚠️ Readout 风险")
        for item in analysis["readout_risks"]:
            st.error(f"🔴 {item['message']}")
            st.caption(f"   可通过调整: {', '.join(item['influenced_by'])}")

    # 调整建议
    if analysis["recommendations"]:
        st.markdown("#### 💡 调整建议 (按优先级)")
        for i, rec in enumerate(analysis["recommendations"], 1):
            with st.expander(f"{i}. {rec['direction']}", expanded=i<=3):
                st.markdown(f"**触发原因:** {rec['rationale']}")
                st.markdown(f"**预期效果:** {rec['expected_effect']}")
                st.markdown(f"**优先级:** {rec['priority']}")
    else:
        st.success("✅ 所有指标在正常范围内，无需调整")

# =============================================================================
# 因果关系图
# =============================================================================
st.markdown("---")
st.markdown("## 🔗 Setpoint → Readout 因果关系")

# 创建关系图数据
if manager.causal_relationships:
    fig = go.Figure()

    # 节点
    setpoint_keys = list(setpoints.keys())
    readout_keys = list(readouts.keys())

    # Setpoint节点（左侧）
    for i, key in enumerate(setpoint_keys):
        fig.add_trace(go.Scatter(
            x=[0], y=[i],
            mode='markers+text',
            marker=dict(size=30, color='#28a745'),
            text=[key],
            textposition='middle left',
            name='Setpoint',
            showlegend=False
        ))

    # Readout节点（右侧）
    for i, key in enumerate(readout_keys):
        fig.add_trace(go.Scatter(
            x=[2], y=[i * 0.8],
            mode='markers+text',
            marker=dict(size=30, color='#ffc107'),
            text=[key],
            textposition='middle right',
            name='Readout',
            showlegend=False
        ))

    # 连线
    for rel in manager.causal_relationships:
        from_key = rel.get("from")
        to_key = rel.get("to")
        if from_key in setpoint_keys and to_key in readout_keys:
            from_idx = setpoint_keys.index(from_key)
            to_idx = readout_keys.index(to_key)
            fig.add_trace(go.Scatter(
                x=[0.3, 1.7],
                y=[from_idx, to_idx * 0.8],
                mode='lines',
                line=dict(color='#6c757d', width=1),
                showlegend=False,
                hoverinfo='text',
                hovertext=rel.get("effect", "")
            ))

    fig.update_layout(
        title="Setpoint → Readout 影响关系",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("因果关系数据未配置")

# =============================================================================
# 配置编辑提示
# =============================================================================
st.markdown("---")
with st.expander("📝 如何修改指标配置"):
    st.markdown("""
    **配置文件位置:** `config/indicator_classification.yaml`

    **修改步骤:**
    1. 编辑 `indicator_classification.yaml` 文件
    2. 在 `setpoints` 下添加/修改可调控指标
    3. 在 `readouts` 下添加/修改可间接影响指标
    4. 在 `causal_relationships` 下定义因果关系
    5. 保存后刷新页面即可生效

    **示例 - 添加新的Setpoint:**
    ```yaml
    setpoints:
      NewIndicator:
        name: "新指标"
        domain: "分类域"
        unit: "单位"
        target_range: [下限, 上限]
        control_method: "调控方法"
    ```

    **示例 - 添加新的Readout:**
    ```yaml
    readouts:
      NewReadout:
        name: "新读数"
        domain: "分类域"
        unit: "单位"
        risk_threshold: 阈值
        risk_direction: "higher_is_worse"  # 或 "lower_is_worse"
        influenced_by: ["Setpoint1", "Setpoint2"]
    ```
    """)
