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

import os
import streamlit as st
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import re
import logging

logger = logging.getLogger(__name__)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

# 添加src目录
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

st.set_page_config(page_title="语音交互", page_icon="🎙️", layout="wide")

# =============================================================================
# 初始化 Neo4j 和 LLM（复用全局单例）
# =============================================================================
@st.cache_resource
def _init_neo4j():
    try:
        from neo4j_connector import Neo4jKnowledgeGraph
        kg = Neo4jKnowledgeGraph()
        return kg
    except Exception:
        return None


@st.cache_resource
def _init_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLM_MODEL", "deepseek-v3.2")
    if not api_key:
        return None, "OPENAI_API_KEY环境变量未设置"
    if not base_url:
        return None, "OPENAI_BASE_URL环境变量未设置"
    try:
        from baseline_strategy_recommender import OpenAILLM
        llm = OpenAILLM(api_key=api_key, model=model, base_url=base_url)
        if llm.is_available():
            return llm, None
        return None, "OpenAI客户端创建失败（检查openai包是否安装）"
    except ImportError:
        return None, "openai包未安装，请运行: pip install openai"
    except Exception as e:
        return None, str(e)


neo4j_kg = _init_neo4j()
_llm_result = _init_llm()
llm_client = _llm_result[0]
_llm_init_error = _llm_result[1]

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
        # 灌注调控参数 (Setpoints)
        "Flow": ["流量", "flow", "Flow", "灌注流量", "泵流量", "CPB流量", "泵速"],
        "Temperature": ["温度", "temperature", "体温", "复温", "降温", "变温", "灌注温度"],
        "AoDP": ["AoDP", "aodp", "灌注压", "主动脉舒张压", "后负荷"],
        "PaO2": ["PaO2", "pao2", "氧分压", "动脉氧分压", "FiO2"],
        "Hemoglobin": ["血红蛋白", "Hb", "hb", "HB", "Hemoglobin", "RBC", "红细胞", "携氧"],
        "PacingRate": ["起搏", "pacing", "起搏心率", "起搏器"],
        "Dobutamine": ["多巴酚丁胺", "dobutamine", "正性肌力"],
        "Insulin": ["胰岛素", "insulin", "血糖控制"],
        # 功能观测指标 (Readouts)
        "MAP": ["MAP", "map", "血压", "动脉压", "低血压", "高血压"],
        "Lactate": ["乳酸", "lactate", "Lactate"],
        "K": ["钾", "K+", "血钾", "高钾", "低钾", "钾离子"],
        "CI": ["心指数", "CI", "ci", "心输出量", "心排量", "心排"],
        "pH": ["pH", "ph", "酸中毒", "碱中毒", "酸碱"],
        "HR": ["心率", "HR", "hr", "心跳"],
        "SvO2": ["SvO2", "svo2", "混合静脉血氧", "血氧饱和度"],
        "CvO2": ["CvO2", "cvo2", "静脉血氧含量"],
        "EF": ["EF", "ef", "射血分数", "LVEF", "lvef"],
        "MVO2": ["MVO2", "mvo2", "心肌氧耗", "氧耗"],
        "dPdt": ["dPdt", "dpdt", "dp/dt", "压力变化率", "收缩力"],
        # 移植/术后评估
        "PVR": ["PVR", "pvr", "肺血管阻力", "肺阻力"],
        "TPG": ["TPG", "tpg", "跨肺压差"],
        "PASP": ["PASP", "pasp", "肺动脉收缩压", "肺动脉压"],
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

    def __init__(self, kb: Dict, llm=None, neo4j_kg=None):
        self.kb = kb
        self.extracted = kb.get("extracted", {})
        self.thresholds = kb.get("thresholds", {})
        self.interventions = kb.get("interventions", {})
        self.classification = kb.get("classification", {})
        self.llm = llm
        self.neo4j_kg = neo4j_kg

    def answer(self, question: str, phase: str = "intraop") -> Dict[str, Any]:
        """
        主入口: 问题 → 意图识别 → 多源检索 → KG查询 → 查验 → LLM增强 → 回答

        Args:
            question: 用户问题
            phase: 手术阶段 "intraop"(术中) 或 "postop"(术后)

        Returns:
            {"answer": str, "sources": list, "intent": dict, "verified": bool, "confidence": float}
        """
        # Step 1: 意图识别
        intent = IntentRecognizer.recognize(question)

        # Step 2: 多源检索（知识库）
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

        # Step 2.5: Neo4j知识图谱查询
        kg_evidences = self._search_neo4j(intent, question)
        evidences.extend(kg_evidences)

        # Step 3: 交叉查验
        verified, confidence = self._verify_evidences(evidences)

        # Step 4: 生成回答（优先LLM增强，回退到规则生成）
        self._last_llm_error = None
        llm_answer = self._llm_enhance(question, intent, evidences, phase=phase)
        llm_error = getattr(self, '_last_llm_error', None)

        if llm_answer:
            answer_text = llm_answer
            sources = list(set(e.get("source", "未知来源") for e in evidences))
            sources.append("LLM (DeepSeek)")
            confidence = min(confidence + 0.1, 1.0)
        else:
            answer_text = self._build_answer(question, intent, evidences, verified)
            sources = list(set(e.get("source", "未知来源") for e in evidences))

        return {
            "answer": answer_text,
            "sources": sources,
            "intent": intent,
            "verified": verified,
            "confidence": confidence,
            "evidence_count": len(evidences),
            "llm_error": llm_error,
            "_evidences": evidences,
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

    # ----- Neo4j 知识图谱查询 -----

    def _search_neo4j(self, intent: Dict, question: str) -> List[Dict]:
        """从Neo4j知识图谱检索结构化三元组证据"""
        if not self.neo4j_kg:
            return []

        results = []
        indicators = intent.get("indicators", [])
        drugs = intent.get("drugs", [])
        intent_type = intent.get("intent", "general_qa")

        try:
            for ind in indicators[:3]:
                keywords = IntentRecognizer.INDICATOR_KEYWORDS.get(ind, [ind])

                # 1. 综合决策支持查询（病因+后果+治疗）
                for kw in keywords[:2]:
                    decision = self.neo4j_kg.query_decision_support(kw)

                    for cause in decision.get("causes", [])[:5]:
                        results.append({
                            "type": "kg_triple",
                            "triple": (cause.get("cause", "?"), cause.get("relation", "causes"), kw),
                            "indicator": ind,
                            "data": cause,
                            "source": "Neo4j-KG(病因)"
                        })
                    for cons in decision.get("consequences", [])[:5]:
                        results.append({
                            "type": "kg_triple",
                            "triple": (kw, cons.get("relation", "leads_to"), cons.get("consequence", "?")),
                            "indicator": ind,
                            "data": cons,
                            "source": "Neo4j-KG(后果)"
                        })
                    for treat in decision.get("treatments", [])[:5]:
                        results.append({
                            "type": "kg_triple",
                            "triple": (treat.get("treatment", "?"), treat.get("relation", "treats"), kw),
                            "indicator": ind,
                            "data": treat,
                            "source": "Neo4j-KG(治疗)"
                        })

                # 2. 指标异常后果链
                if intent_type in ("causal_query", "risk_query", "strategy_query"):
                    consequences = self.neo4j_kg.find_indicator_abnormality_consequences(ind)
                    for c in consequences[:5]:
                        results.append({
                            "type": "kg_triple",
                            "triple": (c.get("indicator", ind), c.get("relation", "→"), c.get("consequence", "?")),
                            "indicator": ind,
                            "data": c,
                            "source": "Neo4j-KG(异常后果链)"
                        })

            # 3. 药物效应查询
            for drug in drugs[:3]:
                effects = self.neo4j_kg.find_medication_effects(drug)
                for eff in effects[:5]:
                    results.append({
                        "type": "kg_triple",
                        "triple": (eff.get("medication", drug), eff.get("relation", "→"), eff.get("target", "?")),
                        "indicator": drug,
                        "data": eff,
                        "source": "Neo4j-KG(药物效应)"
                    })

            # 4. 并发症治疗（针对策略/风险查询）
            if intent_type in ("strategy_query", "risk_query"):
                for ind in indicators[:2]:
                    treatments = self.neo4j_kg.find_treatment_for_complication(ind)
                    for t in treatments[:5]:
                        results.append({
                            "type": "kg_triple",
                            "triple": (t.get("treatment", "?"), t.get("relation_type", "treats"), t.get("complication", ind)),
                            "indicator": ind,
                            "data": t,
                            "source": "Neo4j-KG(并发症治疗)"
                        })

        except Exception as e:
            logger.warning(f"Neo4j查询失败: {e}")
            results.append({
                "type": "kg_error",
                "triple": None,
                "data": {"error": str(e)},
                "source": "Neo4j-KG(查询异常)"
            })

        return results

    # ----- LLM 增强回答 -----

    def _format_kg_triples(self, evidences: List[Dict]) -> str:
        """将KG证据格式化为结构化三元组文本"""
        triples = []
        for ev in evidences:
            if ev.get("type") == "kg_triple" and ev.get("triple"):
                s, p, o = ev["triple"]
                source_tag = ev.get("source", "KG")
                triples.append(f"  ({s}) --[{p}]--> ({o})  [{source_tag}]")
        if not triples:
            return ""
        return "\n".join(triples)

    def _format_kb_evidences(self, evidences: List[Dict]) -> str:
        """将知识库证据格式化为文本"""
        lines = []
        idx = 1
        for ev in evidences:
            if ev.get("type") == "kg_triple" or ev.get("type") == "kg_error":
                continue
            source = ev.get("source", "")
            data = ev.get("data", {})
            if isinstance(data, dict):
                data_str = json.dumps(data, ensure_ascii=False)[:300]
            else:
                data_str = str(data)[:300]
            lines.append(f"  [{idx}] ({source}) {data_str}")
            idx += 1
            if idx > 10:
                break
        return "\n".join(lines) if lines else ""

    def _build_intraop_prompt(self, question: str, intent: Dict,
                              kg_section: str, kb_section: str,
                              indicators_str: str, drugs_str: str) -> str:
        """构建术中阶段的LLM提示词"""
        return f"""# 角色
你是HTTG（心脏移植术中灌注监测）的临床决策支持AI，部署在心脏移植手术室中。
你的回答将直接用于**术中**的实时语音播报，辅助灌注师和主刀医生做出即时决策。

# 场景
心脏移植手术 **术中阶段** — 患者正在经历以下可能的时期之一：
- 体外循环（CPB）运转期
- CPB脱机/撤离期（最关键时期）
- 供心植入后早期评估期
- 鱼精蛋白中和期

**灌注调控参数（Setpoints — 灌注师可直接调控）：**
- Flow（灌注流量）: 目标4.2-4.8 L/min，泵转速直接控制
- Temperature（灌注温度）: 22→37°C复温方案，热交换器控制
- AoDP（灌注压）: 目标35-45 mmHg，泵转速/反馈控制
- PaO2（氧分压）: 氧合器FiO2/扫气流量调节
- Hemoglobin: 目标40-50 g/L，RBC添加管理
- PacingRate: 起搏器AAI模式 100-110 bpm
- Dobutamine: 正性肌力支持 2-6 μg/min
- Insulin: 代谢支持 1.5-3.0 U/h

**术中特征：**
- 灌注师通过调整Setpoints来优化Readouts（乳酸、EF、dP/dt等）
- 电解质紊乱（尤其高钾）可能与心肌保护液/库存血相关，需打药纠正
- 供心缺血再灌注损伤可能导致急性右心衰竭
- 需关注移植心脏的变时性/变力性功能（去神经心脏）
- 温度策略影响范围最广（CVR、Tau、代谢率、舒张功能）
- 出血/凝血问题（鱼精蛋白、血小板、纤维蛋白原）

# 输入

## 用户问题
{question}

## 意图分析
- 意图类型: {intent.get("intent", "general_qa")}
- 涉及指标: {indicators_str}
- 涉及药物: {drugs_str}

## 知识图谱三元组（来自Neo4j，结构化因果/治疗关系）
{kg_section}

## 知识库证据（来自临床共识/干预策略/阈值配置）
{kb_section}

# 输出要求

请严格按以下结构输出，适合**术中紧急播报**的风格（简短、直接、可操作）：

**【判断】** 一句话概括当前情况和紧急程度

**【机制】** 术中病理生理机制（结合供心状态、CPB影响、再灌注损伤等术中特有因素）

**【Setpoint调整】**（灌注师直接操作）
- Flow调整：目标流量 + 泵速方向
- Temperature调整：热交换器目标温度
- AoDP调整：灌注压目标
- 药物调整：Dobutamine/Insulin/KCl等 + 剂量 + 注射泵速率

**【打药方案】**（如需额外用药）
- 药物名称 + 剂量 + 给药途径 + 滴定目标

**【术中警示】** 移植心脏特殊注意事项（去神经化影响、右心保护、出血风险等）

**【证据等级】** KG三元组N条 / 共识知识库N条 / 临床经验

# 约束
1. 剂量、阈值必须来自提供的证据，不可编造
2. 证据不足时标注"基于临床经验补充"
3. 术中播报风格：每段≤2句话，直接给出可操作指令
4. 优先建议Setpoint调整（Flow/Temperature/AoDP/药物），而不是笼统的"处理"
5. 优先考虑术中安全性（出血、心律失常、右心衰竭）
6. 使用中文"""

    def _build_postop_prompt(self, question: str, intent: Dict,
                             kg_section: str, kb_section: str,
                             indicators_str: str, drugs_str: str) -> str:
        """构建术后阶段的LLM提示词"""
        return f"""# 角色
你是HTTG（心脏移植术后监护）的临床决策支持AI，部署在ICU/心外科病房中。
你的回答将用于**术后**的语音播报，辅助ICU医生和护理团队做出治疗决策。

# 场景
心脏移植手术 **术后阶段** — 患者处于以下可能的时期之一：
- ICU早期恢复（术后0-72h）：血流动力学稳定化、呼吸机撤离
- ICU中期（术后3-7天）：感染防控、肾功能保护、营养支持
- 病房恢复期（术后1-4周）：免疫抑制方案调整、排斥反应监测
- 出院前评估期：长期用药方案确定

**术后特征：**
- 免疫抑制是核心管理重点（他克莫司/环孢素谷浓度监测、MMF剂量、激素减量）
- 急性排斥反应的早期识别（心内膜活检、BNP/troponin趋势）
- 感染 vs 排斥的鉴别诊断
- 肾功能保护（CNI肾毒性、容量管理）
- 血压管理目标与术中不同（避免高血压→移植物血管病变）
- 血糖管理（激素相关高血糖）
- 心律监测（移植心脏窦房结功能恢复）

# 输入

## 用户问题
{question}

## 意图分析
- 意图类型: {intent.get("intent", "general_qa")}
- 涉及指标: {indicators_str}
- 涉及药物: {drugs_str}

## 知识图谱三元组（来自Neo4j，结构化因果/治疗关系）
{kg_section}

## 知识库证据（来自临床共识/干预策略/阈值配置）
{kb_section}

# 输出要求

请严格按以下结构输出，适合**术后管理播报**的风格（系统、全面、兼顾长期预后）：

**【判断】** 一句话概括当前问题及其对移植心脏的影响

**【机制】** 术后病理生理机制（结合免疫抑制状态、移植心脏特征、感染/排斥鉴别等术后特有因素）

**【处置方案】**
- 即时处理：药物调整 + 剂量 + 监测指标
- 免疫抑制相关：是否需要调整免疫抑制方案
- 后续计划：检查/检验安排、随访频率

**【长期注意】** 对移植预后的影响、需要警惕的远期并发症、患者教育要点

**【证据等级】** KG三元组N条 / 共识知识库N条 / 临床经验

# 约束
1. 剂量、阈值必须来自提供的证据，不可编造
2. 证据不足时标注"基于临床经验补充"
3. 术后播报风格：每段≤3句话，兼顾即时处理和长期管理
4. 所有建议需考虑免疫抑制背景下的特殊性
5. 使用中文"""

    def _llm_enhance(self, question: str, intent: Dict, evidences: List[Dict],
                     phase: str = "intraop") -> Optional[str]:
        """使用LLM基于KG三元组+知识库证据生成阶段特异性手术场景播报回答"""
        if not self.llm:
            self._last_llm_error = "LLM未配置（检查OPENAI_API_KEY和OPENAI_BASE_URL环境变量）"
            return None

        self._last_llm_error = None
        try:
            # 分离KG三元组 和 知识库证据
            kg_text = self._format_kg_triples(evidences)
            kb_text = self._format_kb_evidences(evidences)

            kg_section = kg_text if kg_text else "  （当前未从知识图谱中检索到相关三元组）"
            kb_section = kb_text if kb_text else "  （当前未从知识库中检索到直接证据）"

            indicators_str = ", ".join(intent.get("indicators", [])) or "未识别"
            drugs_str = ", ".join(intent.get("drugs", [])) or "未涉及"

            # 根据阶段选择不同的提示词
            if phase == "postop":
                prompt = self._build_postop_prompt(
                    question, intent, kg_section, kb_section, indicators_str, drugs_str)
            else:
                prompt = self._build_intraop_prompt(
                    question, intent, kg_section, kb_section, indicators_str, drugs_str)

            response = self.llm.generate(prompt, temperature=0.15, max_tokens=1200)
            if response and len(response.strip()) > 20:
                return response.strip()
            else:
                self._last_llm_error = "LLM返回空响应"
        except Exception as e:
            self._last_llm_error = f"LLM调用失败: {e}"
            logger.warning(f"LLM增强失败: {e}")

        return None

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

# TTS 辅助函数：每个 iframe 自带完整 speak 代码
def _tts_html(text: str) -> str:
    """生成自包含的 TTS iframe HTML"""
    escaped = text.replace("\\", "\\\\").replace("`", "'").replace("</", "<\\/")
    return f"""<script>
(function() {{
    const synth = window.parent.speechSynthesis || window.speechSynthesis;
    if (!synth) {{ return; }}
    synth.cancel();
    const u = new SpeechSynthesisUtterance(`{escaped}`);
    u.lang = 'zh-CN'; u.rate = 0.9; u.pitch = 1; u.volume = 1;
    const voices = synth.getVoices();
    const zh = voices.find(v => v.lang.includes('zh'));
    if (zh) u.voice = zh;
    synth.speak(u);
}})();
</script>"""


def _stt_html(target_label: str) -> str:
    """生成自包含的 STT iframe HTML（解决 iframe 隔离问题）"""
    return f"""<script>
(function() {{
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition
              || window.parent.SpeechRecognition || window.parent.webkitSpeechRecognition;
    if (!SR) {{ alert('浏览器不支持语音识别，请使用Chrome'); return; }}
    const rec = new SR();
    rec.lang = 'zh-CN';
    rec.continuous = false;
    rec.interimResults = false;
    rec.onresult = function(e) {{
        let t = '';
        for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
        if (!t) return;
        // 找到 Streamlit 主页面中的 textarea
        const doc = window.parent.document;
        const el = doc.querySelector('textarea[aria-label="{target_label}"]');
        if (el) {{
            const nativeSetter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype, 'value').set;
            nativeSetter.call(el, t);
            el.dispatchEvent(new Event('input', {{bubbles: true}}));
            // 自动触发提交 - 模拟 Ctrl+Enter
            setTimeout(function() {{
                el.dispatchEvent(new KeyboardEvent('keydown',
                    {{key:'Enter', code:'Enter', keyCode:13, ctrlKey:true, bubbles:true}}));
            }}, 300);
        }}
    }};
    rec.onerror = function(e) {{
        if (e.error !== 'no-speech') {{
            console.error('STT error:', e.error);
        }}
    }};
    rec.start();
}})();
</script>"""


def _tts_stop_html() -> str:
    """生成停止 TTS 的 iframe HTML"""
    return """<script>
(function() {
    const synth = window.parent.speechSynthesis || window.speechSynthesis;
    if (synth) synth.cancel();
})();
</script>"""

# =============================================================================
# 术中/术后 阶段配置
# =============================================================================

# 术中演示警报 — 以灌注调控参数为主
INTRAOP_ALERTS = [
    # 灌注调控参数异常（灌注师首要关注）
    {"level": "critical", "indicator": "Flow", "value": 3.6, "unit": "L/min",
     "target": "4.2-4.8", "message": "灌注流量偏低！检查泵转速、管路阻力、储血罐液面"},
    {"level": "warning", "indicator": "Temperature", "value": 28.5, "unit": "°C",
     "target": "复温至37°C", "message": "灌注温度偏低，复温进度滞后，调整热交换器"},
    {"level": "warning", "indicator": "AoDP", "value": 30, "unit": "mmHg",
     "target": "35-45", "message": "灌注压偏低，冠脉灌注可能不足，调整泵转速"},
    {"level": "warning", "indicator": "Hemoglobin", "value": 34, "unit": "g/L",
     "target": "40-50", "message": "Hb偏低，携氧能力下降，考虑添加RBC"},
    # 关键功能指标异常（需通过调Setpoints改善）
    {"level": "critical", "indicator": "Lactate", "value": 5.2, "unit": "mmol/L",
     "target": "<4.0", "message": "乳酸升高！检查Flow/AoDP/Hb，评估灌注充足性"},
    {"level": "critical", "indicator": "K+", "value": 6.2, "unit": "mmol/L",
     "target": "3.5-5.0", "message": "高钾血症，可能与心肌保护液/库存血相关，需打药降钾"},
    {"level": "warning", "indicator": "pH", "value": 7.18, "unit": "",
     "target": "7.25-7.35", "message": "酸中毒，调整氧合器扫气流量↑排CO2 / NaHCO3"},
]

# 术后演示警报
POSTOP_ALERTS = [
    {"level": "critical", "indicator": "MAP", "value": 105, "unit": "mmHg",
     "target": "70-90", "message": "血压偏高，移植物血管病变风险，调整降压方案"},
    {"level": "warning", "indicator": "Creatinine", "value": 2.1, "unit": "mg/dL",
     "target": "<1.2", "message": "肌酐升高，注意CNI肾毒性，评估他克莫司谷浓度"},
    {"level": "warning", "indicator": "K+", "value": 5.5, "unit": "mmol/L",
     "target": "3.5-5.0", "message": "血钾偏高，可能与CNI或肾功能不全相关"},
    {"level": "warning", "indicator": "HR", "value": 55, "unit": "bpm",
     "target": "80-110", "message": "心率偏低，移植心脏窦房结功能恢复不全，评估是否需异丙肾上腺素"},
    {"level": "warning", "indicator": "Lactate", "value": 3.2, "unit": "mmol/L",
     "target": "<2.0", "message": "乳酸轻度升高，评估心功能及组织灌注"},
]

# 术中快捷提问 — 灌注调控导向
INTRAOP_QUICK_QS = [
    "灌注流量低怎么调？",
    "乳酸升高怎么办？",
    "灌注温度怎么控制？",
    "高钾血症如何处理？",
    "血红蛋白低要加RBC吗？",
    "灌注压AoDP低怎么调？",
    "急性右心衰竭用什么药？",
    "当前风险评估",
]

# 术后快捷提问
POSTOP_QUICK_QS = [
    "术后血压高怎么处理？",
    "他克莫司谷浓度多少合适？",
    "肌酐升高怎么办？",
    "术后心率慢怎么处理？",
    "急性排斥反应怎么识别？",
    "感染和排斥怎么鉴别？",
    "激素减量方案？",
    "当前风险评估",
]


# =============================================================================
# 通用UI组件
# =============================================================================

def _render_alerts(alerts: List[Dict], phase_key: str):
    """渲染警报区域"""
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
        if st.button("🔊 播报全部警报", use_container_width=True, key=f"broadcast_all_{phase_key}"):
            st.components.v1.html(_tts_html(broadcast_text), height=0)
        if st.button("⏹️ 停止播报", use_container_width=True, key=f"stop_broadcast_{phase_key}"):
            st.components.v1.html(_tts_stop_html(), height=0)

        st.markdown("**单项播报**")
        for a in alerts:
            single = f"{a['indicator']}当前{a['value']}{a['unit']}，{a['message']}"
            icon = "🔴" if a["level"] == "critical" else "🟡"
            if st.button(f"{icon} {a['indicator']}", key=f"speak_{a['indicator']}_{phase_key}",
                         use_container_width=True):
                st.components.v1.html(_tts_html(single), height=0)


def _render_qa(qa_engine: KnowledgeQA, phase: str, phase_key: str, quick_qs: List[str]):
    """渲染问答区域"""
    phase_label = "术中" if phase == "intraop" else "术后"
    input_label = f"请输入{phase_label}问题"

    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("**语音/文字输入**")

        # 语音输入按钮
        if st.button("🎤 点击语音输入", use_container_width=True, key=f"stt_{phase_key}"):
            st.components.v1.html(_stt_html(input_label), height=0)
            st.info("🎤 正在录音，请说话...（说完自动停止）")

        # 文字输入
        typed_question = st.text_area(
            input_label,
            height=80,
            placeholder=f"例如：{quick_qs[0]} / {quick_qs[1]}",
            key=f"text_input_{phase_key}",
        )

        # 快捷提问
        st.markdown(f"**{phase_label}快捷提问：**")
        eq_cols = st.columns(4)
        for i, eq in enumerate(quick_qs):
            with eq_cols[i % 4]:
                if st.button(eq, key=f"eq_{phase_key}_{i}", use_container_width=True):
                    st.session_state[f"_quick_q_{phase_key}"] = eq

        # 有效问题
        user_question = st.session_state.pop(f"_quick_q_{phase_key}", None) or typed_question

    with col_output:
        st.markdown("**AI 回答**")

        if user_question and user_question.strip():
            result = qa_engine.answer(user_question.strip(), phase=phase)

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

            st.markdown(
                f'<span class="intent-tag">{intent_label}</span> '
                f'<span class="intent-tag">{phase_label}模式</span>',
                unsafe_allow_html=True
            )

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

            # LLM调用错误提示
            llm_err = result.get("llm_error")
            if llm_err:
                st.warning(f"LLM未参与回答: {llm_err}")

            # KG三元组证据溯源
            kg_triples = [e for e in result.get("_evidences", [])
                          if e.get("type") == "kg_triple" and e.get("triple")]
            if kg_triples:
                with st.expander(f"🔬 知识图谱证据溯源 ({len(kg_triples)} 条三元组)", expanded=False):
                    for tr in kg_triples:
                        s, p, o = tr["triple"]
                        src = tr.get("source", "KG")
                        st.markdown(
                            f'<div style="font-family:monospace; padding:4px 10px; margin:3px 0; '
                            f'border-left:3px solid #1890ff; background:var(--secondary-background-color,#f0f5ff); '
                            f'border-radius:0 4px 4px 0;">'
                            f'<span style="color:#1890ff;">{s}</span> '
                            f'──<span style="color:#722ed1;">{p}</span>──▸ '
                            f'<span style="color:#52c41a;">{o}</span> '
                            f'<span style="opacity:0.5; font-size:0.8em;">({src})</span>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # 来源标注
            if result["sources"]:
                st.markdown("**📚 信息来源：**")
                src_html = " ".join(f'<span class="source-tag">{s}</span>' for s in result["sources"])
                st.markdown(src_html, unsafe_allow_html=True)

            # 播报回答
            answer_plain = re.sub(r'\*\*|#{1,3}\s?|`', '', result["answer"])
            answer_plain = re.sub(r'\n+', '。', answer_plain)
            if st.button("🔊 播报回答", key=f"speak_answer_{phase_key}", use_container_width=True):
                st.components.v1.html(_tts_html(answer_plain), height=0)
        else:
            if phase == "intraop":
                st.markdown("""
**术中支持的问题类型：**
- 🎛 **Setpoint调整**: "流量低怎么调？" "温度怎么控制？" "灌注压低？"
- 💊 **打药方案**: "高钾打什么药？" "多巴酚丁胺怎么用？"
- 📊 **Readout异常**: "乳酸升高怎么办？" "EF低？"
- ⚡ **紧急情况**: "急性右心衰竭？" "高钾心律失常？"
- 🔗 **机制查询**: "温度影响哪些指标？" "为什么肺阻力升高？"
                """)
            else:
                st.markdown("""
**术后支持的问题类型：**
- 🏥 **术后管理**: "术后血压高怎么处理？" "心率慢怎么办？"
- 💊 **免疫抑制**: "他克莫司剂量？" "激素减量方案？"
- 🔬 **排斥/感染**: "急性排斥怎么识别？" "感染和排斥鉴别？"
- 📏 **术后阈值**: "肌酐目标值？" "他克莫司谷浓度？"
- 🔗 **因果查询**: "CNI肾毒性机制？"
                """)


# =============================================================================
# 主页面
# =============================================================================
st.title("🎙️ 语音交互助手")

# 连接状态指示
status_parts = []
if neo4j_kg:
    status_parts.append("Neo4j ✅")
else:
    status_parts.append("Neo4j ⚪")
if llm_client:
    model_name = os.getenv("LLM_MODEL", "LLM")
    status_parts.append(f"{model_name} ✅")
else:
    status_parts.append("LLM ⚪")
st.caption(f"🔌 {' | '.join(status_parts)}")

# 显示LLM初始化错误（帮助排查）
if _llm_init_error:
    st.warning(f"LLM初始化失败: {_llm_init_error}")

# 注入JS
st.components.v1.html(VOICE_JS, height=0)

# 加载知识
kb = load_knowledge_base()
qa_engine = KnowledgeQA(kb, llm=llm_client, neo4j_kg=neo4j_kg)

# =============================================================================
# 术中 / 术后 Tab 切换
# =============================================================================
st.markdown("---")

tab_intraop, tab_postop = st.tabs(["🔴 术中监测 (Intraoperative)", "🟢 术后管理 (Postoperative)"])

# ======================== 术中 Tab ========================
with tab_intraop:
    st.markdown(
        '<div style="padding:8px 16px; border-radius:8px; '
        'background:rgba(255,77,79,0.08); border-left:4px solid #ff4d4f; margin-bottom:1rem;">'
        '<strong>术中模式</strong> — 灌注调控为核心：Flow/Temperature/AoDP/打药 · '
        '灌注师直接调控Setpoints，观测Readouts评估灌注质量</div>',
        unsafe_allow_html=True
    )

    # 术中警报
    st.subheader("📢 术中实时警报")
    _render_alerts(INTRAOP_ALERTS, "intraop")

    # 术中问答
    st.markdown("---")
    qa_sources = ["意图识别"]
    if neo4j_kg:
        qa_sources.append("Neo4j-KG")
    if llm_client:
        qa_sources.append(f"LLM({os.getenv('LLM_MODEL', 'AI')})")
    qa_sources.append("共识知识库")
    st.subheader(f"🎤 术中智能问答（{' + '.join(qa_sources)}）")
    _render_qa(qa_engine, phase="intraop", phase_key="intraop", quick_qs=INTRAOP_QUICK_QS)

# ======================== 术后 Tab ========================
with tab_postop:
    st.markdown(
        '<div style="padding:8px 16px; border-radius:8px; '
        'background:rgba(82,196,26,0.08); border-left:4px solid #52c41a; margin-bottom:1rem;">'
        '<strong>术后模式</strong> — ICU恢复/免疫抑制管理/排斥监测/长期预后 · '
        '提示词针对术后综合管理优化，兼顾即时处理与远期预后</div>',
        unsafe_allow_html=True
    )

    # 术后警报
    st.subheader("📢 术后监测警报")
    _render_alerts(POSTOP_ALERTS, "postop")

    # 术后问答
    st.markdown("---")
    st.subheader(f"🎤 术后智能问答（{' + '.join(qa_sources)}）")
    _render_qa(qa_engine, phase="postop", phase_key="postop", quick_qs=POSTOP_QUICK_QS)

# =============================================================================
# 自动播报设置（通用）
# =============================================================================
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

# --- 使用说明 ---
st.markdown("---")
with st.expander("📖 使用说明"):
    st.markdown("""
### 术中/术后模式切换
- 页面顶部 **🔴 术中监测** / **🟢 术后管理** 标签页可切换阶段
- 两个阶段使用 **不同的LLM提示词**，针对各自的临床重点生成回答
- 术中强调：CPB管理、血管活性药即时调整、急性右心衰竭、出血凝血
- 术后强调：免疫抑制方案、排斥/感染鉴别、肾功能保护、长期预后

### 语音播报
- 点击 **🔊 播报全部警报** 播报当前阶段所有异常
- 点击单个指标按钮播报特定警报
- 点击 **⏹️ 停止播报** 随时停止

### 语音输入
- 点击 **🎤 语音输入** 后对麦克风说话（需Chrome浏览器）
- 识别结果会自动填入问题框
- 也可以直接输入文字 或 点击快捷提问按钮

### 智能问答流程
1. **意图识别**: 自动判断问题类型（策略/药物/阈值/因果/风险）
2. **多源检索**: Neo4j知识图谱 + 共识文献 + 配置库
3. **交叉查验**: 多源证据一致性检查
4. **阶段特异性LLM增强**: 根据术中/术后使用不同提示词生成回答

### 浏览器要求
- 推荐 **Chrome** 浏览器
- 首次使用需允许麦克风权限
- Safari/Firefox 可能不支持语音识别
    """)
