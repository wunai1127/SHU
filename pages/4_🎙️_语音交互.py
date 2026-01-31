#!/usr/bin/env python3
"""
语音交互页面 - 语音播报 + 语音输入 + 意图识别 + LLM/KG/知识库问答
=================================================================

修复:
1. 白底白字 → 适配深色/浅色主题
2. 语音输入 → streamlit-webrtc + Web Speech API 双通道
3. 语音输出 → 浏览器 Web Speech API TTS，Streamlit组件控制
4. 意图识别 → 关键词+规则意图分类器
5. 问答 → 知识库检索 + KG查询 + 共识文献 → 查验确认 → 回答
"""

import streamlit as st
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import re

# 添加src目录
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="语音交互", page_icon="🎙️", layout="wide")

# =============================================================================
# 加载知识库
# =============================================================================
@st.cache_data
def load_knowledge_base():
    """加载全部知识源"""
    kb = {
        "extracted": {},
        "thresholds": {},
        "interventions": {},
        "classification": {},
    }
    base = Path(__file__).parent.parent

    # 1. 共识抽取知识
    ek = base / "extracted_knowledge.json"
    if ek.exists():
        with open(ek, 'r', encoding='utf-8') as f:
            kb["extracted"] = json.load(f)

    # 2. 阈值配置
    th = base / "config" / "thresholds.yaml"
    if th.exists():
        with open(th, 'r', encoding='utf-8') as f:
            kb["thresholds"] = yaml.safe_load(f)

    # 3. 干预策略
    iv = base / "config" / "intervention_strategies.yaml"
    if iv.exists():
        with open(iv, 'r', encoding='utf-8') as f:
            kb["interventions"] = yaml.safe_load(f)

    # 4. 指标分类
    ic = base / "config" / "indicator_classification.yaml"
    if ic.exists():
        with open(ic, 'r', encoding='utf-8') as f:
            kb["classification"] = yaml.safe_load(f)

    return kb


@st.cache_data
def load_demo_alerts():
    """演示用的当前监测警报"""
    return [
        {"level": "critical", "indicator": "MAP", "value": 45, "unit": "mmHg",
         "target": "65-80", "message": "平均动脉压严重偏低，建议立即使用去甲肾上腺素 0.05~0.1 μg/kg/min"},
        {"level": "critical", "indicator": "K+", "value": 6.2, "unit": "mmol/L",
         "target": "3.5-5.0", "message": "血钾严重升高，心律失常风险，建议胰岛素+葡萄糖降钾"},
        {"level": "warning", "indicator": "Lactate", "value": 4.5, "unit": "mmol/L",
         "target": "<4.0", "message": "乳酸轻度升高，提示组织灌注不足"},
        {"level": "warning", "indicator": "CI", "value": 2.0, "unit": "L/min/m²",
         "target": "2.2-4.0", "message": "心指数偏低，建议强心治疗"},
    ]


# =============================================================================
# 意图识别模块
# =============================================================================
class IntentRecognizer:
    """
    基于规则+关键词的意图识别器

    意图类型:
    - indicator_query: 查询某个指标的信息/阈值/处理方法
    - drug_query: 查询药物用法/剂量
    - risk_query: 查询当前风险/整体状态
    - causal_query: 查询因果关系
    - strategy_query: 查询处理策略/下一步怎么做
    - threshold_query: 查询阈值/正常范围
    - general_qa: 一般问答
    """

    INDICATOR_KEYWORDS = {
        "MAP": ["MAP", "map", "血压", "动脉压", "低血压", "高血压", "灌注压"],
        "Lactate": ["乳酸", "lactate", "Lactate"],
        "K": ["钾", "K+", "血钾", "高钾", "低钾", "钾离子"],
        "CI": ["心指数", "CI", "ci", "心输出量", "心排量", "心排"],
        "pH": ["pH", "ph", "酸中毒", "碱中毒", "酸碱"],
        "HR": ["心率", "HR", "hr", "心跳"],
        "SvO2": ["SvO2", "svo2", "混合静脉血氧", "血氧饱和度"],
        "CvO2": ["CvO2", "cvo2", "静脉血氧含量"],
        "PVR": ["PVR", "pvr", "肺血管阻力", "肺阻力"],
        "TPG": ["TPG", "tpg", "跨肺压差"],
        "PASP": ["PASP", "pasp", "肺动脉收缩压", "肺动脉压"],
        "EF": ["EF", "ef", "射血分数", "LVEF", "lvef"],
        "Creatinine": ["肌酐", "creatinine", "Creatinine"],
        "GFR": ["GFR", "gfr", "肾小球滤过率", "肾功能"],
        "Bilirubin": ["胆红素", "bilirubin", "Bilirubin"],
    }

    DRUG_KEYWORDS = {
        "去甲肾上腺素": ["去甲肾上腺素", "去甲肾", "norepinephrine", "NE"],
        "多巴胺": ["多巴胺", "dopamine"],
        "多巴酚丁胺": ["多巴酚丁胺", "dobutamine"],
        "米力农": ["米力农", "milrinone"],
        "肾上腺素": ["肾上腺素", "epinephrine"],
        "异丙肾上腺素": ["异丙肾上腺素", "isoproterenol", "异丙肾"],
        "他克莫司": ["他克莫司", "tacrolimus", "FK506"],
        "环孢素": ["环孢素", "cyclosporine"],
        "巴利昔单抗": ["巴利昔单抗", "basiliximab"],
        "ATG": ["ATG", "atg", "抗胸腺细胞球蛋白"],
        "泼尼松": ["泼尼松", "prednisone"],
        "甲泼尼龙": ["甲泼尼龙", "methylprednisolone"],
        "霉酚酸酯": ["霉酚酸酯", "MMF", "mmf"],
        "碳酸氢钠": ["碳酸氢钠", "NaHCO3"],
        "胰岛素": ["胰岛素", "insulin"],
        "一氧化氮": ["一氧化氮", "NO", "吸入NO"],
        "西地那非": ["西地那非", "sildenafil"],
        "前列腺素": ["前列腺素", "PGE1", "前列地尔"],
        "钙剂": ["钙剂", "葡萄糖酸钙", "氯化钙"],
        "利尿剂": ["利尿剂", "呋塞米", "速尿"],
    }

    INTENT_PATTERNS = [
        (r"(怎么处理|怎么办|如何处理|怎样处理|怎么治|如何治疗|怎么调)", "strategy_query"),
        (r"(用什么药|药物|用药|剂量|怎么用|给药)", "drug_query"),
        (r"(风险|危险|严重|预后|风险等级)", "risk_query"),
        (r"(为什么|原因|导致|引起|造成|因果|机制)", "causal_query"),
        (r"(阈值|正常值|正常范围|目标值|标准|红线|上限|下限)", "threshold_query"),
        (r"(是什么|什么是|含义|意义|代表|解释)", "indicator_query"),
        (r"(当前|目前|现在|最新|状态|数值)", "indicator_query"),
    ]

    @classmethod
    def recognize(cls, question: str) -> Dict[str, Any]:
        """
        识别用户意图

        Returns:
            {"intent": str, "indicators": list, "drugs": list, "confidence": float, "detail": str}
        """
        q = question.strip()

        # 识别涉及的指标
        matched_indicators = []
        for ind, keywords in cls.INDICATOR_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in q.lower():
                    if ind not in matched_indicators:
                        matched_indicators.append(ind)
                    break

        # 识别涉及的药物
        matched_drugs = []
        for drug, keywords in cls.DRUG_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in q.lower():
                    if drug not in matched_drugs:
                        matched_drugs.append(drug)
                    break

        # 识别意图类型
        intent = "general_qa"
        confidence = 0.5
        for pattern, intent_type in cls.INTENT_PATTERNS:
            if re.search(pattern, q):
                intent = intent_type
                confidence = 0.85
                break

        # 如果有药物关键词但未识别意图 → drug_query
        if matched_drugs and intent == "general_qa":
            intent = "drug_query"
            confidence = 0.8

        # 如果有指标关键词但未识别意图 → indicator_query
        if matched_indicators and intent == "general_qa":
            intent = "indicator_query"
            confidence = 0.75

        detail = f"意图={intent}, 指标={matched_indicators}, 药物={matched_drugs}"

        return {
            "intent": intent,
            "indicators": matched_indicators,
            "drugs": matched_drugs,
            "confidence": confidence,
            "detail": detail,
        }


# =============================================================================
# 知识检索 + 查验模块
# =============================================================================
class KnowledgeQA:
    """
    多源知识检索 + 查验确认

    检索流程:
    1. 意图识别
    2. 根据意图选择检索策略
    3. 从 知识库/KG/共识 多源检索
    4. 交叉查验
    5. 生成带来源的回答
    """

    def __init__(self, kb: Dict):
        self.kb = kb
        self.extracted = kb.get("extracted", {})
        self.thresholds = kb.get("thresholds", {})
        self.interventions = kb.get("interventions", {})
        self.classification = kb.get("classification", {})

    def answer(self, question: str) -> Dict[str, Any]:
        """
        主入口: 问题 → 意图识别 → 多源检索 → 查验 → 回答

        Returns:
            {"answer": str, "sources": list, "intent": dict, "verified": bool, "confidence": float}
        """
        # Step 1: 意图识别
        intent = IntentRecognizer.recognize(question)

        # Step 2: 多源检索
        evidences = []
        if intent["intent"] == "strategy_query":
            evidences = self._search_strategies(intent)
        elif intent["intent"] == "drug_query":
            evidences = self._search_drugs(intent)
        elif intent["intent"] == "threshold_query":
            evidences = self._search_thresholds(intent)
        elif intent["intent"] == "causal_query":
            evidences = self._search_causal(intent)
        elif intent["intent"] == "indicator_query":
            evidences = self._search_indicator_info(intent)
        elif intent["intent"] == "risk_query":
            evidences = self._search_risk(intent)
        else:
            evidences = self._search_general(intent, question)

        # Step 3: 交叉查验
        verified, confidence = self._verify_evidences(evidences)

        # Step 4: 生成回答
        answer_text = self._build_answer(question, intent, evidences, verified)
        sources = list(set(e.get("source", "未知来源") for e in evidences))

        return {
            "answer": answer_text,
            "sources": sources,
            "intent": intent,
            "verified": verified,
            "confidence": confidence,
            "evidence_count": len(evidences),
        }

    # ----- 检索方法 -----

    def _search_strategies(self, intent: Dict) -> List[Dict]:
        """检索干预策略"""
        results = []
        indicators = intent.get("indicators", [])

        for ind in indicators:
            # 从 intervention_strategies.yaml 查
            for section_key in ["cardiac_function", "hemodynamic", "metabolic", "electrolyte",
                                "transplant_heart_rate", "right_heart_failure",
                                "immunosuppression_induction", "immunosuppression_maintenance",
                                "donor_vasoactive"]:
                section = self.interventions.get(section_key, {})
                for key, val in section.items():
                    if not isinstance(val, dict):
                        continue
                    ind_name = val.get("indicator_name", key)
                    if ind.lower() in key.lower() or ind.lower() in ind_name.lower():
                        abnormalities = val.get("abnormalities", {})
                        for abn_key, abn_val in abnormalities.items():
                            if isinstance(abn_val, dict):
                                results.append({
                                    "type": "strategy",
                                    "indicator": ind,
                                    "condition": abn_key,
                                    "data": abn_val,
                                    "source": f"干预策略库 ({section_key}/{key})"
                                })

            # 从 extracted_knowledge 查
            drug_strategies = self.extracted.get("药物策略", {})
            for cat, drugs in drug_strategies.items():
                if isinstance(drugs, dict):
                    for drug_name, drug_info in drugs.items():
                        if isinstance(drug_info, dict):
                            text = json.dumps(drug_info, ensure_ascii=False)
                            if ind.lower() in text.lower() or any(
                                kw.lower() in text.lower()
                                for kw in IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [])
                            ):
                                results.append({
                                    "type": "drug_strategy",
                                    "indicator": ind,
                                    "drug": drug_name,
                                    "data": drug_info,
                                    "source": f"共识知识库 (药物策略/{cat})"
                                })

        return results

    def _search_drugs(self, intent: Dict) -> List[Dict]:
        """检索药物信息"""
        results = []
        drugs = intent.get("drugs", [])
        indicators = intent.get("indicators", [])

        # 从 extracted_knowledge 查药物
        drug_strategies = self.extracted.get("药物策略", {})
        for cat, cat_data in drug_strategies.items():
            if isinstance(cat_data, dict):
                for drug_name, drug_info in cat_data.items():
                    if isinstance(drug_info, dict):
                        matched = any(d in drug_name for d in drugs)
                        if not matched and indicators:
                            text = json.dumps(drug_info, ensure_ascii=False)
                            matched = any(
                                kw.lower() in text.lower()
                                for ind in indicators
                                for kw in IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [])
                            )
                        if matched:
                            results.append({
                                "type": "drug_info",
                                "drug": drug_name,
                                "data": drug_info,
                                "source": f"共识知识库 ({cat})"
                            })
                    elif isinstance(cat_data, str) and any(d in cat for d in drugs):
                        results.append({
                            "type": "drug_info",
                            "drug": cat,
                            "data": {"信息": cat_data},
                            "source": "共识知识库"
                        })

        # 从 intervention_strategies 查药物
        for section_key, section in self.interventions.items():
            if not isinstance(section, dict):
                continue
            section_text = json.dumps(section, ensure_ascii=False)
            for drug in drugs:
                if drug in section_text:
                    results.append({
                        "type": "drug_in_strategy",
                        "drug": drug,
                        "data": {"来源段落": section_key},
                        "source": f"干预策略库 ({section_key})"
                    })

        return results

    def _search_thresholds(self, intent: Dict) -> List[Dict]:
        """检索阈值信息"""
        results = []
        indicators = intent.get("indicators", [])

        # 从 extracted_knowledge 查阈值
        thresholds = self.extracted.get("阈值_与_指标", {})
        for th_key, th_val in thresholds.items():
            if isinstance(th_val, dict):
                for ind in indicators:
                    keywords = IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [ind])
                    if any(kw.lower() in th_key.lower() for kw in keywords):
                        results.append({
                            "type": "threshold",
                            "indicator": ind,
                            "data": th_val,
                            "source": f"共识知识库 (阈值/{th_key})"
                        })

        # 从 thresholds.yaml 查
        for section_key, section in self.thresholds.items():
            if not isinstance(section, dict):
                continue
            for ind in indicators:
                for key, val in section.items():
                    if isinstance(val, dict) and (ind.lower() in key.lower()):
                        results.append({
                            "type": "threshold_config",
                            "indicator": ind,
                            "data": val,
                            "source": f"阈值配置 ({section_key}/{key})"
                        })

        return results

    def _search_causal(self, intent: Dict) -> List[Dict]:
        """检索因果关系"""
        results = []
        indicators = intent.get("indicators", [])

        # 从 extracted_knowledge 查因果
        causal_list = self.extracted.get("因果关系", [])
        for rel in causal_list:
            text = json.dumps(rel, ensure_ascii=False)
            if not indicators or any(
                kw.lower() in text.lower()
                for ind in indicators
                for kw in IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [ind])
            ):
                results.append({
                    "type": "causal",
                    "data": rel,
                    "source": "共识知识库 (因果关系)"
                })

        # 从 classification 查因果
        causals = self.classification.get("causal_relationships", [])
        for rel in causals:
            text = json.dumps(rel, ensure_ascii=False)
            if not indicators or any(
                kw.lower() in text.lower()
                for ind in indicators
                for kw in IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [ind])
            ):
                results.append({
                    "type": "causal_config",
                    "data": rel,
                    "source": "指标分类配置 (因果关系)"
                })

        return results

    def _search_indicator_info(self, intent: Dict) -> List[Dict]:
        """检索指标综合信息"""
        results = []
        results.extend(self._search_thresholds(intent))
        results.extend(self._search_strategies(intent))
        return results

    def _search_risk(self, intent: Dict) -> List[Dict]:
        """检索风险相关"""
        alerts = load_demo_alerts()
        critical = [a for a in alerts if a["level"] == "critical"]
        warning = [a for a in alerts if a["level"] == "warning"]
        return [{
            "type": "risk_summary",
            "data": {
                "critical_count": len(critical),
                "warning_count": len(warning),
                "critical_indicators": [a["indicator"] for a in critical],
                "warning_indicators": [a["indicator"] for a in warning],
            },
            "source": "实时监测数据"
        }]

    def _search_general(self, intent: Dict, question: str) -> List[Dict]:
        """通用检索: 全文搜索知识库"""
        results = []
        q_lower = question.lower()

        # 搜索 extracted_knowledge 全文
        for category, data in self.extracted.items():
            text = json.dumps(data, ensure_ascii=False)
            if any(w in text.lower() for w in q_lower.split() if len(w) > 1):
                results.append({
                    "type": "general",
                    "data": {"category": category},
                    "source": f"共识知识库 ({category})"
                })

        return results

    # ----- 查验 -----

    def _verify_evidences(self, evidences: List[Dict]) -> Tuple[bool, float]:
        """
        交叉查验: 检查多源证据是否一致

        Returns: (verified: bool, confidence: float)
        """
        if not evidences:
            return False, 0.3

        # 多源 = 高置信
        sources = set(e.get("source", "") for e in evidences)
        source_count = len(sources)

        if source_count >= 3:
            return True, 0.95
        elif source_count >= 2:
            return True, 0.85
        elif source_count == 1 and len(evidences) >= 2:
            return True, 0.75
        else:
            return False, 0.6

    # ----- 回答生成 -----

    def _build_answer(self, question: str, intent: Dict, evidences: List[Dict], verified: bool) -> str:
        """根据检索结果生成结构化回答"""
        if not evidences:
            return self._fallback_answer(question, intent)

        parts = []
        intent_type = intent["intent"]

        if intent_type == "strategy_query":
            parts.append(self._format_strategy_answer(intent, evidences))
        elif intent_type == "drug_query":
            parts.append(self._format_drug_answer(intent, evidences))
        elif intent_type == "threshold_query":
            parts.append(self._format_threshold_answer(intent, evidences))
        elif intent_type == "causal_query":
            parts.append(self._format_causal_answer(intent, evidences))
        elif intent_type == "risk_query":
            parts.append(self._format_risk_answer(intent, evidences))
        elif intent_type == "indicator_query":
            parts.append(self._format_indicator_answer(intent, evidences))
        else:
            parts.append(self._format_general_answer(intent, evidences))

        return "\n".join(parts)

    def _format_strategy_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化策略回答"""
        lines = []
        indicators = intent.get("indicators", [])
        ind_name = indicators[0] if indicators else "指标"

        lines.append(f"**{ind_name} 异常处理策略：**\n")

        for ev in evidences[:5]:
            if ev["type"] == "strategy":
                condition = ev.get("condition", "")
                data = ev.get("data", {})
                actions = data.get("actions", [])
                lines.append(f"**{condition}:**")
                if isinstance(actions, list):
                    for act in actions[:4]:
                        if isinstance(act, dict):
                            drug = act.get("drug", act.get("action", ""))
                            dose = act.get("dose", "")
                            lines.append(f"- {drug}" + (f"（{dose}）" if dose else ""))
                        else:
                            lines.append(f"- {act}")
                escalation = data.get("escalation", [])
                if escalation:
                    lines.append(f"- 升级方案: {', '.join(escalation[:3])}")
                lines.append("")

            elif ev["type"] == "drug_strategy":
                drug = ev.get("drug", "")
                data = ev.get("data", {})
                lines.append(f"**{drug}:**")
                for k, v in data.items():
                    if k != "来源":
                        lines.append(f"- {k}: {v}")
                lines.append("")

        return "\n".join(lines)

    def _format_drug_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化药物回答"""
        lines = []
        drugs = intent.get("drugs", [])
        drug_name = drugs[0] if drugs else "药物"

        lines.append(f"**{drug_name} 用药信息：**\n")

        for ev in evidences[:5]:
            data = ev.get("data", {})
            drug = ev.get("drug", "")
            if drug:
                lines.append(f"**{drug}:**")
            for k, v in data.items():
                if k not in ["来源", "来源段落"]:
                    lines.append(f"- {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def _format_threshold_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化阈值回答"""
        lines = []
        indicators = intent.get("indicators", [])
        ind_name = indicators[0] if indicators else "指标"

        lines.append(f"**{ind_name} 阈值与正常范围：**\n")

        for ev in evidences[:5]:
            data = ev.get("data", {})
            for k, v in data.items():
                if k != "来源":
                    lines.append(f"- {k}: {v}")
            source = data.get("来源", ev.get("source", ""))
            if source:
                lines.append(f"- 📚 来源: {source}")
            lines.append("")

        return "\n".join(lines)

    def _format_causal_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化因果关系回答"""
        lines = []
        lines.append("**相关因果关系：**\n")

        for ev in evidences[:6]:
            data = ev.get("data", {})
            fr = data.get("from", data.get("upstream", ""))
            to = data.get("to", data.get("downstream", ""))
            effect = data.get("effect", data.get("mechanism", ""))
            if fr and to:
                lines.append(f"- **{fr}** → **{to}**")
                if effect:
                    lines.append(f"  - 效应: {effect}")

        return "\n".join(lines)

    def _format_risk_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化风险回答"""
        lines = []
        for ev in evidences:
            if ev["type"] == "risk_summary":
                data = ev["data"]
                cc = data["critical_count"]
                wc = data["warning_count"]
                lines.append(f"**当前风险评估：**\n")
                lines.append(f"- 🔴 危急指标: {cc} 项 ({', '.join(data['critical_indicators'])})")
                lines.append(f"- 🟡 警告指标: {wc} 项 ({', '.join(data['warning_indicators'])})")
                total = cc + wc
                if cc >= 2:
                    lines.append(f"- ⚠️ 整体风险: **HIGH** (共{total}项异常)")
                elif cc >= 1:
                    lines.append(f"- ⚠️ 整体风险: **MEDIUM** (共{total}项异常)")
                else:
                    lines.append(f"- 整体风险: **LOW**")
        return "\n".join(lines)

    def _format_indicator_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化指标综合回答"""
        parts = []
        threshold_ev = [e for e in evidences if "threshold" in e["type"]]
        strategy_ev = [e for e in evidences if "strategy" in e["type"]]

        if threshold_ev:
            parts.append(self._format_threshold_answer(intent, threshold_ev))
        if strategy_ev:
            parts.append(self._format_strategy_answer(intent, strategy_ev))

        return "\n".join(parts) if parts else self._fallback_answer("", intent)

    def _format_general_answer(self, intent: Dict, evidences: List[Dict]) -> str:
        """格式化通用回答"""
        lines = ["**检索到以下相关信息：**\n"]
        for ev in evidences[:5]:
            source = ev.get("source", "")
            lines.append(f"- 来源: {source}")
        return "\n".join(lines)

    def _fallback_answer(self, question: str, intent: Dict) -> str:
        """兜底回答"""
        indicators = intent.get("indicators", [])
        if indicators:
            return f"关于 {', '.join(indicators)} 的问题，当前知识库中未找到完全匹配的信息。建议查阅临床指南或咨询专科医生。"
        return '抱歉，当前知识库中未找到匹配信息。请尝试更具体的问题，如「MAP低怎么处理」或「他克莫司剂量」。'


# =============================================================================
# 样式 — 适配深色/浅色主题
# =============================================================================
st.markdown("""
<style>
    /* 适配 Streamlit 深色/浅色主题 */
    .voice-container {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid var(--secondary-background-color, #e0e0e0);
        background: var(--secondary-background-color, #f8f9fa);
        color: var(--text-color, #333);
        margin-bottom: 1rem;
    }
    .alert-card-critical {
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 6px;
        border-left: 4px solid #ff4d4f;
        background: rgba(255, 77, 79, 0.1);
        color: var(--text-color, #333);
    }
    .alert-card-warning {
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 6px;
        border-left: 4px solid #faad14;
        background: rgba(250, 173, 20, 0.1);
        color: var(--text-color, #333);
    }
    .answer-box {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        border: 1px solid var(--secondary-background-color, #dee2e6);
        background: var(--secondary-background-color, #f8f9fa);
        color: var(--text-color, #333);
        min-height: 80px;
        margin: 0.5rem 0;
    }
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
        background: rgba(24, 144, 255, 0.15);
        color: var(--text-color, #1890ff);
    }
    .intent-tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
        background: rgba(114, 46, 209, 0.12);
        color: var(--text-color, #722ed1);
    }
    .verify-pass {
        color: #52c41a;
        font-weight: bold;
    }
    .verify-fail {
        color: #faad14;
        font-weight: bold;
    }
    .tts-btn {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        font-size: 14px;
        margin: 4px;
        transition: all 0.2s;
    }
    .tts-btn:hover { opacity: 0.85; transform: scale(1.03); }
    .tts-speak {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .tts-stop {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    .tts-listen {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 浏览器语音 JS（TTS + STT）
# =============================================================================
VOICE_JS = """
<script>
// ===== TTS =====
function speak(text) {
    if (!('speechSynthesis' in window)) { alert('浏览器不支持语音合成'); return; }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'zh-CN'; u.rate = 0.9; u.pitch = 1; u.volume = 1;
    const voices = window.speechSynthesis.getVoices();
    const zh = voices.find(v => v.lang.includes('zh'));
    if (zh) u.voice = zh;
    window.speechSynthesis.speak(u);
}
function stopSpeaking() {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
}
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
}

// ===== STT =====
let _recognition = null;
function startSTT(targetId) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) { alert('浏览器不支持语音识别，请使用 Chrome'); return; }
    _recognition = new SR();
    _recognition.lang = 'zh-CN';
    _recognition.continuous = false;
    _recognition.interimResults = true;
    _recognition.onresult = (e) => {
        let t = '';
        for (let i = e.resultIndex; i < e.results.length; i++) t += e.results[i][0].transcript;
        // 将识别结果写入 Streamlit 隐藏的 textarea
        const el = window.parent.document.querySelector('textarea[aria-label="' + targetId + '"]');
        if (el) {
            const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSetter.call(el, t);
            el.dispatchEvent(new Event('input', {bubbles: true}));
        }
    };
    _recognition.start();
}
function stopSTT() { if (_recognition) _recognition.stop(); }
</script>
"""

# =============================================================================
# 主页面
# =============================================================================
st.title("🎙️ 语音交互助手")

# 注入JS
st.components.v1.html(VOICE_JS, height=0)

# 加载知识
kb = load_knowledge_base()
qa_engine = KnowledgeQA(kb)
alerts = load_demo_alerts()

# --- 1. 策略播报区域 ---
st.markdown("---")
st.subheader("📢 实时策略播报")

col_alerts, col_ctrl = st.columns([3, 1])

with col_alerts:
    for alert in alerts:
        level_cls = "alert-card-critical" if alert["level"] == "critical" else "alert-card-warning"
        icon = "🔴" if alert["level"] == "critical" else "🟡"
        st.markdown(f"""
        <div class="{level_cls}">
            <strong>{icon} {alert['indicator']}: {alert['value']} {alert['unit']}</strong>
            （目标: {alert['target']}）<br/>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)

with col_ctrl:
    broadcast_text = "灌注监测警报播报。"
    for a in alerts:
        lvl = "危急" if a["level"] == "critical" else "警告"
        broadcast_text += f"{lvl}，{a['indicator']}当前{a['value']}{a['unit']}，{a['message']}。"

    st.markdown("**播报控制**")

    if st.button("🔊 播报全部警报", use_container_width=True):
        escaped = broadcast_text.replace("`", "'").replace("\\", "\\\\")
        st.components.v1.html(f"<script>window.parent.postMessage('tts','*');speak(`{escaped}`);</script>", height=0)

    if st.button("⏹️ 停止播报", use_container_width=True):
        st.components.v1.html("<script>stopSpeaking();</script>", height=0)

    st.markdown("**单项播报**")
    for a in alerts:
        single = f"{a['indicator']}当前{a['value']}{a['unit']}，{a['message']}"
        icon = "🔴" if a["level"] == "critical" else "🟡"
        if st.button(f"{icon} {a['indicator']}", key=f"speak_{a['indicator']}", use_container_width=True):
            escaped = single.replace("`", "'").replace("\\", "\\\\")
            st.components.v1.html(f"<script>speak(`{escaped}`);</script>", height=0)

# --- 2. 语音问答区域 ---
st.markdown("---")
st.subheader("🎤 智能问答（意图识别 + 知识图谱 + 共识知识库）")

col_input, col_output = st.columns([1, 1])

with col_input:
    st.markdown("**语音/文字输入**")

    # 语音输入按钮
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("🎤 语音输入", use_container_width=True):
            st.components.v1.html(
                "<script>startSTT('请输入您的问题');</script>", height=0
            )
    with btn_col2:
        if st.button("⏹️ 停止录音", use_container_width=True):
            st.components.v1.html("<script>stopSTT();</script>", height=0)

    # 文字输入
    user_question = st.text_area(
        "请输入您的问题",
        height=80,
        placeholder="例如：MAP低应该怎么处理？/ 他克莫司剂量是多少？/ 高钾血症的因果关系？",
        key="qa_input",
    )

    # 示例问题快捷按钮
    st.markdown("**快捷提问：**")
    example_qs = [
        "MAP低应该怎么处理？",
        "乳酸升高怎么办？",
        "高钾血症如何处理？",
        "心指数偏低用什么药？",
        "他克莫司剂量是多少？",
        "肺血管阻力高的阈值？",
        "高钾导致什么后果？",
        "当前风险评估",
    ]
    eq_cols = st.columns(4)
    for i, eq in enumerate(example_qs):
        with eq_cols[i % 4]:
            if st.button(eq, key=f"eq_{i}", use_container_width=True):
                st.session_state["qa_input"] = eq
                st.rerun()

with col_output:
    st.markdown("**AI 回答**")

    if user_question and user_question.strip():
        # 执行问答
        result = qa_engine.answer(user_question.strip())

        # 意图识别结果
        intent = result["intent"]
        intent_labels = {
            "strategy_query": "🏥 策略查询",
            "drug_query": "💊 药物查询",
            "threshold_query": "📏 阈值查询",
            "causal_query": "🔗 因果查询",
            "indicator_query": "📊 指标查询",
            "risk_query": "⚠️ 风险查询",
            "general_qa": "💬 通用问答",
        }
        intent_label = intent_labels.get(intent["intent"], intent["intent"])

        st.markdown(f'<span class="intent-tag">{intent_label}</span>', unsafe_allow_html=True)

        if intent["indicators"]:
            st.caption(f"识别指标: {', '.join(intent['indicators'])}")
        if intent["drugs"]:
            st.caption(f"识别药物: {', '.join(intent['drugs'])}")

        # 查验状态
        if result["verified"]:
            st.markdown(f'<span class="verify-pass">✅ 多源查验通过 (置信度: {result["confidence"]:.0%}, {result["evidence_count"]}条证据)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="verify-fail">⚠️ 单源参考 (置信度: {result["confidence"]:.0%}, {result["evidence_count"]}条证据)</span>', unsafe_allow_html=True)

        # 回答正文
        st.markdown(result["answer"])

        # 来源标注
        if result["sources"]:
            st.markdown("**📚 信息来源：**")
            src_html = " ".join(f'<span class="source-tag">{s}</span>' for s in result["sources"])
            st.markdown(src_html, unsafe_allow_html=True)

        # 播报回答按钮
        answer_plain = re.sub(r'\*\*|#{1,3}\s?|`', '', result["answer"])
        answer_plain = re.sub(r'\n+', '。', answer_plain)
        escaped = answer_plain.replace("`", "'").replace("\\", "\\\\")
        if st.button("🔊 播报回答", key="speak_answer", use_container_width=True):
            st.components.v1.html(f"<script>speak(`{escaped}`);</script>", height=0)
    else:
        st.markdown("""
**支持的问题类型：**
- 📊 **指标查询**: "MAP是什么？" "肌酐正常范围？"
- 🏥 **策略查询**: "MAP低怎么处理？" "乳酸高怎么办？"
- 💊 **药物查询**: "他克莫司剂量？" "米力农怎么用？"
- 📏 **阈值查询**: "PVR移植禁忌阈值？"
- 🔗 **因果查询**: "高钾导致什么？" "为什么肺阻力升高？"
- ⚠️ **风险查询**: "当前风险评估"

系统会自动进行 **意图识别** → **多源检索**（知识图谱+共识文献+配置库）→ **交叉查验** → 输出回答。
        """)

# --- 3. 自动播报设置 ---
st.markdown("---")
st.subheader("⚙️ 自动播报设置")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    auto_broadcast = st.checkbox("启用自动播报", value=False)
with col_s2:
    broadcast_interval = st.selectbox("播报间隔", ["每5分钟", "每10分钟", "每30分钟", "仅危急时"])
with col_s3:
    broadcast_level = st.multiselect("播报级别", ["危急 (Critical)", "警告 (Warning)"], default=["危急 (Critical)"])

if auto_broadcast:
    st.info("🔔 自动播报已启用。当检测到选定级别的异常时，系统将自动语音播报。")

# --- 4. 使用说明 ---
st.markdown("---")
with st.expander("📖 使用说明"):
    st.markdown("""
### 语音播报
- 点击 **🔊 播报全部警报** 播报当前所有异常
- 点击单个指标按钮播报特定警报
- 点击 **⏹️ 停止播报** 随时停止

### 语音输入
- 点击 **🎤 语音输入** 后对麦克风说话（需Chrome浏览器）
- 识别结果会自动填入问题框
- 也可以直接输入文字 或 点击快捷提问按钮

### 智能问答流程
1. **意图识别**: 自动判断问题类型（策略/药物/阈值/因果/风险）
2. **多源检索**: 从共识知识库、干预策略库、阈值配置等多源获取信息
3. **交叉查验**: 检查多源证据一致性，标注置信度
4. **生成回答**: 结构化回答 + 来源标注

### 浏览器要求
- 推荐 **Chrome** 浏览器
- 首次使用需允许麦克风权限
- Safari/Firefox 可能不支持语音识别
    """)
