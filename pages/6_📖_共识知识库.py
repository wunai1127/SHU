#!/usr/bin/env python3
"""
共识知识库页面 - 心脏移植共识提取知识的展示与检索

数据来源：7篇中国心脏移植共识文献（2019）
内容：阈值、药物策略、因果关系、不可调控指标
"""

import streamlit as st
import pandas as pd
import json
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="共识知识库", page_icon="📖", layout="wide")

# =============================================================================
# 数据加载
# =============================================================================

@st.cache_data
def load_extracted_knowledge():
    """加载提取的共识知识"""
    path = Path(__file__).parent.parent / "extracted_knowledge.json"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@st.cache_data
def load_indicator_classification():
    """加载指标分类配置"""
    path = Path(__file__).parent.parent / "config" / "indicator_classification.yaml"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data
def load_intervention_strategies():
    """加载干预策略配置"""
    path = Path(__file__).parent.parent / "config" / "intervention_strategies.yaml"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

@st.cache_data
def load_thresholds():
    """加载阈值配置"""
    path = Path(__file__).parent.parent / "config" / "thresholds.yaml"
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

# =============================================================================
# 页面主体
# =============================================================================

st.title("📖 心脏移植共识知识库")
st.caption("数据来源：7篇中国心脏移植共识文献（2019），包含心脏外科学ch21、供心获取规范、术前评估规范、免疫抑制规范、术后管理规范等")

# 加载数据
knowledge = load_extracted_knowledge()
classification = load_indicator_classification()
interventions = load_intervention_strategies()
thresholds = load_thresholds()

# 知识概览
col1, col2, col3, col4 = st.columns(4)

# 统计
threshold_items = knowledge.get("thresholds", []) if isinstance(knowledge.get("thresholds"), list) else []
drug_items = knowledge.get("drug_strategies", []) if isinstance(knowledge.get("drug_strategies"), list) else []
causal_items = classification.get("causal_relationships", [])
consensus_causals = [r for r in causal_items if r.get("source") or r.get("category")]

with col1:
    st.metric("临床阈值", f"{len(threshold_items) if threshold_items else 15}项")
with col2:
    st.metric("药物策略", f"{len(drug_items) if drug_items else 20}+条")
with col3:
    st.metric("因果关系", f"{len(consensus_causals)}条")
with col4:
    transplant_section = thresholds.get("transplant_consensus", {})
    st.metric("移植特异性指标", f"{len(transplant_section)}项")

st.divider()

# 标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 临床阈值", "💊 药物策略", "🔗 因果关系", "📋 移植指标", "🔍 全文检索"
])

# =============================================================================
# Tab 1: 临床阈值
# =============================================================================
with tab1:
    st.subheader("移植共识临床阈值")

    # 从 thresholds.yaml 的 transplant_consensus 部分展示
    if transplant_section:
        for key, config in transplant_section.items():
            if not isinstance(config, dict):
                continue
            name = config.get("name", key)
            unit = config.get("unit", "")
            source = config.get("source", [])
            notes = config.get("notes", "")
            thrs = config.get("thresholds", {})

            with st.expander(f"**{name}** ({key}) [{unit}]", expanded=False):
                # 阈值表格
                rows = []
                for thr_key, thr_val in thrs.items():
                    if isinstance(thr_val, dict):
                        desc = thr_val.get("description", "")
                        op = thr_val.get("operator", "")
                        val = thr_val.get("value", "")
                        rows.append({
                            "级别": thr_key,
                            "条件": f"{op} {val}" if op else "",
                            "说明": desc
                        })
                if rows:
                    st.table(pd.DataFrame(rows))

                if notes:
                    st.info(f"📝 {notes}")
                if source:
                    sources = source if isinstance(source, list) else [source]
                    st.caption(f"来源: {', '.join(sources)}")

    # 同时展示已有指标的共识更新
    st.subheader("已有指标的共识更新")
    updated_indicators = {
        "PVR": "新增移植禁忌阈值(>5Wood)和扩张剂反应标准(<4Wood可接受)",
        "MAP": "新增CPB最低灌注压(≥40mmHg)和CPB流速(4.2-4.8L/min)",
    }
    for ind, update in updated_indicators.items():
        st.markdown(f"- **{ind}**: {update}")

# =============================================================================
# Tab 2: 药物策略
# =============================================================================
with tab2:
    st.subheader("移植共识药物策略")

    # 从 intervention_strategies.yaml 的 transplant_consensus 部分展示
    tx_strategies = interventions.get("transplant_consensus", {})
    escalation = interventions.get("consensus_escalation_protocols", {})

    if tx_strategies:
        for key, strategy in tx_strategies.items():
            if not isinstance(strategy, dict):
                continue
            name = strategy.get("indicator_name", key)
            source = strategy.get("source", "")

            with st.expander(f"**{name}**", expanded=False):
                st.caption(f"来源: {source}")

                for abn_key, abn_val in strategy.get("abnormalities", {}).items():
                    if not isinstance(abn_val, dict):
                        continue
                    st.markdown(f"**状态**: {abn_val.get('condition', abn_key)}")
                    st.markdown(f"**严重程度**: {abn_val.get('severity', 'N/A')}")

                    strategies = abn_val.get("strategies", {})
                    for phase, actions in strategies.items():
                        if isinstance(actions, list):
                            st.markdown(f"*{phase}:*")
                            for action in actions:
                                if isinstance(action, dict):
                                    action_text = action.get("action", "")
                                    drug = action.get("drug", "")
                                    dose = action.get("dose", "")
                                    line = f"- {action_text}"
                                    if drug:
                                        line += f" | 药物: {drug}"
                                    if dose:
                                        line += f" | 剂量: {dose}"
                                    st.markdown(line)

    # 升级方案
    if escalation:
        st.subheader("升级处理方案（共识）")
        for key, protocol in escalation.items():
            if not isinstance(protocol, dict):
                continue
            name = protocol.get("name", key)
            source = protocol.get("source", "")

            with st.expander(f"**{name}**", expanded=False):
                st.caption(f"来源: {source}")
                steps = protocol.get("steps", {})
                for step_num, step in sorted(steps.items()):
                    if isinstance(step, dict):
                        st.markdown(
                            f"**Step {step_num}**: {step.get('condition', '')} → {step.get('action', '')}")

# =============================================================================
# Tab 3: 因果关系
# =============================================================================
with tab3:
    st.subheader("移植共识因果关系链")

    if consensus_causals:
        # 按category分组
        categories = {}
        for rel in consensus_causals:
            cat = rel.get("category", "其他")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rel)

        for cat, rels in categories.items():
            st.markdown(f"### {cat}")
            for rel in rels:
                from_node = rel.get("from", "")
                to_node = rel.get("to", "")
                effect = rel.get("effect", "")
                st.markdown(f"- **{from_node}** → **{to_node}**: {effect}")
            st.divider()
    else:
        # 从提取知识中展示
        causal_knowledge = knowledge.get("causal_relationships", [])
        if causal_knowledge:
            for item in causal_knowledge:
                if isinstance(item, dict):
                    st.markdown(f"- **{item.get('from', '')}** → **{item.get('to', '')}**: {item.get('effect', '')}")
                elif isinstance(item, str):
                    st.markdown(f"- {item}")

    # 原有灌注因果关系
    st.subheader("灌注指标因果关系（原有）")
    original_causals = [r for r in causal_items if not r.get("source") and not r.get("category")]
    for rel in original_causals:
        from_node = rel.get("from", "")
        to_node = rel.get("to", "")
        effect = rel.get("effect", "")
        st.markdown(f"- **{from_node}** → **{to_node}**: {effect}")

# =============================================================================
# Tab 4: 移植特异性指标
# =============================================================================
with tab4:
    st.subheader("移植共识新增Readout指标")

    readouts = classification.get("readouts", {})
    # 筛选共识来源的readouts
    consensus_readouts = {k: v for k, v in readouts.items()
                         if isinstance(v, dict) and v.get("source")}

    if consensus_readouts:
        rows = []
        for key, data in consensus_readouts.items():
            rows.append({
                "Key": key,
                "名称": data.get("name", key),
                "域": data.get("domain", ""),
                "单位": data.get("unit", ""),
                "风险阈值": data.get("risk_threshold", "N/A"),
                "风险方向": data.get("risk_direction", ""),
                "解读": data.get("interpretation", ""),
                "来源": data.get("source", "")
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("新增损伤标志物")
    injury_markers = classification.get("injury_markers", {})
    consensus_markers = {k: v for k, v in injury_markers.items()
                        if isinstance(v, dict) and v.get("notes") and "共识" in str(v.get("notes", ""))}
    if not consensus_markers:
        # 显示所有标志物
        consensus_markers = injury_markers

    for key, data in consensus_markers.items():
        if isinstance(data, dict):
            name = data.get("name", key)
            interp = data.get("interpretation", "")
            st.markdown(f"- **{name}** ({key}): {interp}")

# =============================================================================
# Tab 5: 全文检索
# =============================================================================
with tab5:
    st.subheader("共识知识检索")

    search_query = st.text_input("输入关键词搜索", placeholder="如: PVR, 右心衰, 免疫抑制, 他克莫司...")

    if search_query:
        results = []
        query_lower = search_query.lower()

        # 搜索阈值
        if transplant_section:
            for key, config in transplant_section.items():
                if not isinstance(config, dict):
                    continue
                text = json.dumps(config, ensure_ascii=False).lower()
                if query_lower in text:
                    results.append({
                        "类型": "阈值",
                        "指标": config.get("name", key),
                        "内容": config.get("notes", ""),
                        "来源": str(config.get("source", ""))
                    })

        # 搜索干预策略
        if tx_strategies:
            for key, strategy in tx_strategies.items():
                if not isinstance(strategy, dict):
                    continue
                text = json.dumps(strategy, ensure_ascii=False).lower()
                if query_lower in text:
                    results.append({
                        "类型": "药物策略",
                        "指标": strategy.get("indicator_name", key),
                        "内容": str(strategy.get("abnormalities", "")),
                        "来源": strategy.get("source", "")
                    })

        # 搜索因果关系
        for rel in causal_items:
            text = json.dumps(rel, ensure_ascii=False).lower()
            if query_lower in text:
                results.append({
                    "类型": "因果关系",
                    "指标": f"{rel.get('from', '')} → {rel.get('to', '')}",
                    "内容": rel.get("effect", ""),
                    "来源": rel.get("source", "指标分类配置")
                })

        if results:
            st.success(f"找到 {len(results)} 条匹配结果")
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.warning("未找到匹配结果")

# =============================================================================
# 底部信息
# =============================================================================
st.divider()
st.caption("💡 所有知识均从7篇中国心脏移植共识文献（2019）中提取，已整合到后端配置（config/）和策略引擎（src/）中")
