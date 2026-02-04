"""
教学DNA指纹系统
================
量化教师教学理念和逻辑的核心模块

核心理念：
每位教师都有独特的"教学DNA"，包含：
1. 语言指纹 - 词汇偏好、句式习惯、口头禅
2. 逻辑指纹 - 概念关联方式、推理路径
3. 互动指纹 - 提问模式、反馈风格、鼓励方式
4. 节奏指纹 - 难度曲线、详略分配、举例时机

通过量化这些维度，让AI能更精准地模拟教师风格
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import json
from collections import Counter


class LogicPattern(Enum):
    """逻辑推理模式"""
    DEDUCTIVE = "演绎型"      # 从一般到特殊：原理→推论→例子
    INDUCTIVE = "归纳型"      # 从特殊到一般：例子→规律→原理
    ANALOGICAL = "类比型"     # 通过相似性：A像B→理解A
    DIALECTICAL = "辩证型"    # 正反合：观点→反驳→综合
    SPIRAL = "螺旋型"         # 循环深入：概念→应用→深化概念


class InteractionPattern(Enum):
    """互动模式"""
    GUIDE = "引导型"          # 多用问句引导思考
    EXPLAIN = "讲解型"        # 系统性阐述
    DIALOGUE = "对话型"       # 来回交流
    CHALLENGE = "挑战型"      # 设置难题激发
    SCAFFOLDING = "支架型"    # 逐步搭建


class FeedbackStyle(Enum):
    """反馈风格"""
    ENCOURAGING = "鼓励式"    # 多肯定、正向激励
    ANALYTICAL = "分析式"     # 客观分析对错
    SOCRATIC = "追问式"       # 继续追问深入
    CORRECTIVE = "纠正式"     # 直接指出问题


@dataclass
class LanguageFingerprint:
    """
    语言指纹：捕捉教师的语言特征
    """
    # 高频词汇（教师常用的词）
    frequent_words: Dict[str, int] = field(default_factory=dict)

    # 句式模板（教师习惯的句式结构）
    sentence_patterns: List[str] = field(default_factory=list)
    # 例如: "首先...其次...最后..."
    #       "换句话说..."
    #       "这里要特别注意..."

    # 口头禅（特色表达）
    catchphrases: List[str] = field(default_factory=list)

    # 过渡词偏好
    transition_words: List[str] = field(default_factory=list)

    # 语气词使用（啊、呢、吧等）
    modal_particles: Dict[str, float] = field(default_factory=dict)

    # 正式度分数 (0-1, 0最口语化，1最书面化)
    formality_score: float = 0.5

    # 平均句长
    avg_sentence_length: float = 20.0

    # 问句使用频率
    question_frequency: float = 0.1

    def to_prompt_hints(self) -> str:
        """转换为Prompt提示"""
        hints = []

        if self.catchphrases:
            hints.append(f"常用表达：{', '.join(self.catchphrases[:5])}")

        if self.sentence_patterns:
            hints.append(f"句式习惯：{'; '.join(self.sentence_patterns[:3])}")

        if self.transition_words:
            hints.append(f"过渡词偏好：{', '.join(self.transition_words[:5])}")

        if self.question_frequency > 0.2:
            hints.append("善于使用提问引导思考")
        elif self.question_frequency < 0.05:
            hints.append("较少使用反问，以陈述为主")

        if self.formality_score > 0.7:
            hints.append("语言风格偏书面、正式")
        elif self.formality_score < 0.3:
            hints.append("语言风格口语化、亲切")

        return "\n".join(hints)


@dataclass
class LogicFingerprint:
    """
    逻辑指纹：捕捉教师的推理方式
    """
    # 主要推理模式
    primary_pattern: LogicPattern = LogicPattern.DEDUCTIVE

    # 概念展开顺序（教师习惯的讲解流程）
    concept_flow: List[str] = field(default_factory=lambda: [
        "定义", "性质", "例子", "应用", "总结"
    ])

    # 概念关联图（哪些概念教师习惯放在一起讲）
    concept_associations: Dict[str, List[str]] = field(default_factory=dict)

    # 抽象-具体平衡 (0-1, 0全抽象，1全具体)
    abstraction_balance: float = 0.5

    # 理论-实践平衡 (0-1, 0全理论，1全实践)
    theory_practice_balance: float = 0.5

    # 深度优先 vs 广度优先 (0-1, 0深度优先，1广度优先)
    depth_breadth_preference: float = 0.5

    # 因果解释偏好（是否喜欢解释"为什么"）
    causal_explanation_tendency: float = 0.7

    def to_prompt_hints(self) -> str:
        """转换为Prompt提示"""
        hints = []

        pattern_desc = {
            LogicPattern.DEDUCTIVE: "从原理推导到具体应用",
            LogicPattern.INDUCTIVE: "从具体例子归纳出规律",
            LogicPattern.ANALOGICAL: "善用类比帮助理解",
            LogicPattern.DIALECTICAL: "喜欢从正反两面分析",
            LogicPattern.SPIRAL: "循环深入、逐层递进"
        }
        hints.append(f"推理风格：{pattern_desc.get(self.primary_pattern, '')}")

        if self.concept_flow:
            hints.append(f"讲解顺序：{' → '.join(self.concept_flow)}")

        if self.abstraction_balance < 0.3:
            hints.append("偏重抽象理论阐述")
        elif self.abstraction_balance > 0.7:
            hints.append("偏重具体实例讲解")

        if self.causal_explanation_tendency > 0.7:
            hints.append("注重解释'为什么'，强调因果关系")

        return "\n".join(hints)


@dataclass
class InteractionFingerprint:
    """
    互动指纹：捕捉教师的互动特征
    """
    # 主要互动模式
    primary_mode: InteractionPattern = InteractionPattern.EXPLAIN

    # 反馈风格
    feedback_style: FeedbackStyle = FeedbackStyle.ENCOURAGING

    # 提问类型分布
    question_types: Dict[str, float] = field(default_factory=lambda: {
        "知识回忆型": 0.2,    # "什么是...?"
        "理解检验型": 0.3,    # "为什么...?"
        "应用引导型": 0.2,    # "如果...会怎样?"
        "反思启发型": 0.2,    # "你觉得...?"
        "挑战思辨型": 0.1     # "有没有可能...?"
    })

    # 等待时间（提问后是否给思考时间）
    wait_for_thinking: bool = True

    # 错误处理方式
    error_handling: str = "先肯定努力，再指出问题，最后给出正确方向"

    # 鼓励词库
    encouragement_phrases: List[str] = field(default_factory=list)

    # 追问深度（1-5）
    follow_up_depth: int = 2

    def to_prompt_hints(self) -> str:
        """转换为Prompt提示"""
        hints = []

        mode_desc = {
            InteractionPattern.GUIDE: "通过提问引导学生自己发现答案",
            InteractionPattern.EXPLAIN: "系统性讲解，逻辑清晰",
            InteractionPattern.DIALOGUE: "与学生来回对话，共同探讨",
            InteractionPattern.CHALLENGE: "设置有难度的问题激发思考",
            InteractionPattern.SCAFFOLDING: "搭建学习支架，循序渐进"
        }
        hints.append(f"互动方式：{mode_desc.get(self.primary_mode, '')}")

        feedback_desc = {
            FeedbackStyle.ENCOURAGING: "以鼓励为主，正向反馈",
            FeedbackStyle.ANALYTICAL: "客观分析，指出优缺点",
            FeedbackStyle.SOCRATIC: "继续追问，引导深入思考",
            FeedbackStyle.CORRECTIVE: "直接纠正错误，明确对错"
        }
        hints.append(f"反馈风格：{feedback_desc.get(self.feedback_style, '')}")

        hints.append(f"错误处理：{self.error_handling}")

        if self.encouragement_phrases:
            hints.append(f"鼓励用语：{', '.join(self.encouragement_phrases[:3])}")

        return "\n".join(hints)


@dataclass
class RhythmFingerprint:
    """
    节奏指纹：捕捉教师的教学节奏
    """
    # 难度曲线类型
    difficulty_curve: str = "渐进式"  # 渐进式/波浪式/阶梯式/倒金字塔

    # 详略分配（哪些内容详讲，哪些略讲）
    detail_allocation: Dict[str, str] = field(default_factory=lambda: {
        "核心概念": "详细",
        "推导过程": "适中",
        "应用例子": "详细",
        "边缘知识": "简略"
    })

    # 举例时机
    example_timing: str = "讲完概念立即举例"

    # 总结频率
    summary_frequency: str = "每个知识点后小结，章节末尾大总结"

    # 复习策略
    review_strategy: str = "新课前快速回顾上节重点"

    # 单次讲解时长偏好（字数）
    preferred_response_length: int = 300

    def to_prompt_hints(self) -> str:
        """转换为Prompt提示"""
        hints = [
            f"难度把控：{self.difficulty_curve}",
            f"举例时机：{self.example_timing}",
            f"总结习惯：{self.summary_frequency}",
            f"回复长度：约{self.preferred_response_length}字为宜"
        ]
        return "\n".join(hints)


@dataclass
class TeachingDNA:
    """
    教学DNA：教师的完整教学特征指纹

    这是量化教师个性的核心数据结构
    """
    teacher_id: str
    teacher_name: str

    # 四大指纹
    language: LanguageFingerprint = field(default_factory=LanguageFingerprint)
    logic: LogicFingerprint = field(default_factory=LogicFingerprint)
    interaction: InteractionFingerprint = field(default_factory=InteractionFingerprint)
    rhythm: RhythmFingerprint = field(default_factory=RhythmFingerprint)

    # DNA强度（各维度的权重，用于Prompt生成）
    weights: Dict[str, float] = field(default_factory=lambda: {
        "language": 0.25,
        "logic": 0.30,
        "interaction": 0.25,
        "rhythm": 0.20
    })

    def generate_dna_prompt(self) -> str:
        """
        生成DNA增强的Prompt片段

        这是让AI"听起来像老师本人"的核心
        """
        sections = []

        sections.append("# 教学DNA特征（请严格遵循以下特征，让回复风格与教师本人一致）\n")

        # 语言指纹
        sections.append("## 语言风格")
        sections.append(self.language.to_prompt_hints())

        # 逻辑指纹
        sections.append("\n## 思维逻辑")
        sections.append(self.logic.to_prompt_hints())

        # 互动指纹
        sections.append("\n## 互动特点")
        sections.append(self.interaction.to_prompt_hints())

        # 节奏指纹
        sections.append("\n## 教学节奏")
        sections.append(self.rhythm.to_prompt_hints())

        return "\n".join(sections)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "teacher_id": self.teacher_id,
            "teacher_name": self.teacher_name,
            "language": {
                "catchphrases": self.language.catchphrases,
                "sentence_patterns": self.language.sentence_patterns,
                "transition_words": self.language.transition_words,
                "formality_score": self.language.formality_score,
                "question_frequency": self.language.question_frequency
            },
            "logic": {
                "primary_pattern": self.logic.primary_pattern.value,
                "concept_flow": self.logic.concept_flow,
                "abstraction_balance": self.logic.abstraction_balance,
                "causal_explanation_tendency": self.logic.causal_explanation_tendency
            },
            "interaction": {
                "primary_mode": self.interaction.primary_mode.value,
                "feedback_style": self.interaction.feedback_style.value,
                "error_handling": self.interaction.error_handling,
                "encouragement_phrases": self.interaction.encouragement_phrases
            },
            "rhythm": {
                "difficulty_curve": self.rhythm.difficulty_curve,
                "example_timing": self.rhythm.example_timing,
                "summary_frequency": self.rhythm.summary_frequency,
                "preferred_response_length": self.rhythm.preferred_response_length
            },
            "weights": self.weights
        }


class DNAExtractor:
    """
    DNA提取器

    从教师的历史回答、教学材料中自动提取教学DNA
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def extract_from_samples(
        self,
        teacher_id: str,
        teacher_name: str,
        sample_answers: List[str],
        sample_materials: List[str] = None
    ) -> TeachingDNA:
        """
        从样本中提取教学DNA

        Args:
            teacher_id: 教师ID
            teacher_name: 教师姓名
            sample_answers: 教师的回答样本
            sample_materials: 教学材料样本

        Returns:
            提取的TeachingDNA
        """
        dna = TeachingDNA(teacher_id=teacher_id, teacher_name=teacher_name)

        # 提取语言指纹
        dna.language = self._extract_language_fingerprint(sample_answers)

        # 提取逻辑指纹
        dna.logic = self._extract_logic_fingerprint(sample_answers)

        # 提取互动指纹
        dna.interaction = self._extract_interaction_fingerprint(sample_answers)

        # 如果有LLM，使用LLM辅助提取
        if self.llm_client and sample_answers:
            dna = self._enhance_with_llm(dna, sample_answers)

        return dna

    def _extract_language_fingerprint(self, samples: List[str]) -> LanguageFingerprint:
        """提取语言指纹"""
        fp = LanguageFingerprint()

        if not samples:
            return fp

        all_text = " ".join(samples)

        # 统计词频
        words = re.findall(r'[\u4e00-\u9fa5]+', all_text)
        word_freq = Counter(words)
        fp.frequent_words = dict(word_freq.most_common(50))

        # 检测句式模板
        patterns = []
        if "首先" in all_text and "其次" in all_text:
            patterns.append("首先...其次...（递进式）")
        if "换句话说" in all_text or "也就是说" in all_text:
            patterns.append("换句话说...（解释式）")
        if "比如" in all_text or "例如" in all_text:
            patterns.append("比如/例如...（举例式）")
        if "注意" in all_text or "特别" in all_text:
            patterns.append("注意/特别...（强调式）")
        fp.sentence_patterns = patterns

        # 统计问句频率
        sentences = re.split(r'[。！？]', all_text)
        questions = [s for s in sentences if '？' in s or s.endswith('?')]
        fp.question_frequency = len(questions) / max(len(sentences), 1)

        # 计算平均句长
        fp.avg_sentence_length = sum(len(s) for s in sentences) / max(len(sentences), 1)

        # 检测过渡词
        transition_candidates = ["因此", "所以", "但是", "然而", "另外", "此外", "总之", "综上"]
        fp.transition_words = [w for w in transition_candidates if w in all_text]

        return fp

    def _extract_logic_fingerprint(self, samples: List[str]) -> LogicFingerprint:
        """提取逻辑指纹"""
        fp = LogicFingerprint()

        if not samples:
            return fp

        all_text = " ".join(samples)

        # 检测推理模式
        if "因为" in all_text and "所以" in all_text:
            fp.primary_pattern = LogicPattern.DEDUCTIVE
        elif "比如" in all_text and "从中可以看出" in all_text:
            fp.primary_pattern = LogicPattern.INDUCTIVE
        elif "就像" in all_text or "类似于" in all_text:
            fp.primary_pattern = LogicPattern.ANALOGICAL

        # 检测因果解释倾向
        causal_words = ["因为", "由于", "原因是", "导致", "之所以"]
        causal_count = sum(all_text.count(w) for w in causal_words)
        fp.causal_explanation_tendency = min(causal_count / max(len(samples), 1) * 0.3, 1.0)

        return fp

    def _extract_interaction_fingerprint(self, samples: List[str]) -> InteractionFingerprint:
        """提取互动指纹"""
        fp = InteractionFingerprint()

        if not samples:
            return fp

        all_text = " ".join(samples)

        # 检测互动模式
        question_count = all_text.count('？')
        if question_count > len(samples) * 2:
            fp.primary_mode = InteractionPattern.GUIDE
        else:
            fp.primary_mode = InteractionPattern.EXPLAIN

        # 检测反馈风格
        encouraging_words = ["很好", "不错", "对的", "正确", "棒", "好问题"]
        if any(w in all_text for w in encouraging_words):
            fp.feedback_style = FeedbackStyle.ENCOURAGING

        # 提取鼓励用语
        for word in encouraging_words:
            if word in all_text:
                fp.encouragement_phrases.append(word)

        return fp

    def _enhance_with_llm(self, dna: TeachingDNA, samples: List[str]) -> TeachingDNA:
        """使用LLM增强DNA提取"""
        if not self.llm_client:
            return dna

        sample_text = "\n---\n".join(samples[:5])  # 最多5个样本

        prompt = f"""分析以下教师回答样本，提取其教学风格特征。

教师回答样本：
{sample_text}

请以JSON格式返回以下信息：
{{
    "catchphrases": ["教师的口头禅或特色表达，最多5个"],
    "logic_pattern": "演绎型/归纳型/类比型/辩证型/螺旋型 之一",
    "interaction_mode": "引导型/讲解型/对话型/挑战型/支架型 之一",
    "feedback_style": "鼓励式/分析式/追问式/纠正式 之一",
    "difficulty_curve": "渐进式/波浪式/阶梯式 之一",
    "teaching_philosophy": "用一句话概括该教师的教学理念"
}}
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)

            # 更新DNA
            if result.get("catchphrases"):
                dna.language.catchphrases = result["catchphrases"]

            pattern_map = {
                "演绎型": LogicPattern.DEDUCTIVE,
                "归纳型": LogicPattern.INDUCTIVE,
                "类比型": LogicPattern.ANALOGICAL,
                "辩证型": LogicPattern.DIALECTICAL,
                "螺旋型": LogicPattern.SPIRAL
            }
            if result.get("logic_pattern") in pattern_map:
                dna.logic.primary_pattern = pattern_map[result["logic_pattern"]]

            mode_map = {
                "引导型": InteractionPattern.GUIDE,
                "讲解型": InteractionPattern.EXPLAIN,
                "对话型": InteractionPattern.DIALOGUE,
                "挑战型": InteractionPattern.CHALLENGE,
                "支架型": InteractionPattern.SCAFFOLDING
            }
            if result.get("interaction_mode") in mode_map:
                dna.interaction.primary_mode = mode_map[result["interaction_mode"]]

        except Exception as e:
            print(f"LLM增强失败: {e}")

        return dna


# 预置DNA模板
DNA_TEMPLATES = {
    "严谨学者型": TeachingDNA(
        teacher_id="template_rigorous",
        teacher_name="严谨学者",
        language=LanguageFingerprint(
            catchphrases=["准确地说", "从定义来看", "需要严格区分"],
            sentence_patterns=["定义→定理→证明→例题"],
            formality_score=0.8,
            question_frequency=0.1
        ),
        logic=LogicFingerprint(
            primary_pattern=LogicPattern.DEDUCTIVE,
            concept_flow=["定义", "性质", "定理", "证明", "应用"],
            abstraction_balance=0.3,
            causal_explanation_tendency=0.9
        ),
        interaction=InteractionFingerprint(
            primary_mode=InteractionPattern.EXPLAIN,
            feedback_style=FeedbackStyle.ANALYTICAL,
            error_handling="指出错误本质，分析错因，给出正确思路"
        ),
        rhythm=RhythmFingerprint(
            difficulty_curve="阶梯式",
            example_timing="讲完定理后用例题验证",
            preferred_response_length=400
        )
    ),

    "亲和启发型": TeachingDNA(
        teacher_id="template_inspiring",
        teacher_name="亲和启发",
        language=LanguageFingerprint(
            catchphrases=["你觉得呢", "想一想", "有没有发现", "是不是很有趣"],
            sentence_patterns=["问题引入→思考→发现→总结"],
            formality_score=0.3,
            question_frequency=0.4
        ),
        logic=LogicFingerprint(
            primary_pattern=LogicPattern.INDUCTIVE,
            concept_flow=["现象", "问题", "探索", "规律", "应用"],
            abstraction_balance=0.7,
            causal_explanation_tendency=0.6
        ),
        interaction=InteractionFingerprint(
            primary_mode=InteractionPattern.GUIDE,
            feedback_style=FeedbackStyle.ENCOURAGING,
            error_handling="首先肯定思考过程，然后引导发现问题",
            encouragement_phrases=["很好的思考", "你已经很接近了", "继续往下想"]
        ),
        rhythm=RhythmFingerprint(
            difficulty_curve="渐进式",
            example_timing="先举生活例子引起兴趣，再讲概念",
            preferred_response_length=250
        )
    ),

    "实战派教练": TeachingDNA(
        teacher_id="template_practical",
        teacher_name="实战教练",
        language=LanguageFingerprint(
            catchphrases=["来看个实际案例", "在工作中", "实际上", "常见的坑是"],
            sentence_patterns=["问题场景→解决方案→代码实现→注意事项"],
            formality_score=0.4,
            question_frequency=0.2
        ),
        logic=LogicFingerprint(
            primary_pattern=LogicPattern.INDUCTIVE,
            concept_flow=["需求", "方案", "实现", "优化", "总结"],
            abstraction_balance=0.8,
            theory_practice_balance=0.8,
            causal_explanation_tendency=0.5
        ),
        interaction=InteractionFingerprint(
            primary_mode=InteractionPattern.SCAFFOLDING,
            feedback_style=FeedbackStyle.CORRECTIVE,
            error_handling="直接指出问题，给出正确做法，解释为什么"
        ),
        rhythm=RhythmFingerprint(
            difficulty_curve="波浪式",
            example_timing="每个知识点都配实际代码演示",
            preferred_response_length=350
        )
    )
}
