"""
Prompt生成器
============
将教师画像转换为个性化系统提示词
这是智能体个性化的核心技术
"""

from typing import Dict, Any, Optional
from ..core.teacher_profile import (
    TeacherProfile,
    TeachingStyle,
    InteractionMode,
    KnowledgeDomain,
    TeachingLogic
)


class PromptGenerator:
    """
    Prompt生成器：将教师画像转换为系统提示词

    技术原理：
    1. 多层Prompt组装：角色定位 + 教学风格 + 知识约束 + 行为规则
    2. 动态插槽填充：根据配置动态生成提示词片段
    3. Few-shot注入：将教师的案例/例题作为示例注入
    """

    def __init__(self, profile: TeacherProfile):
        self.profile = profile

    def generate_system_prompt(self) -> str:
        """生成完整的系统提示词"""
        sections = [
            self._generate_role_section(),
            self._generate_style_section(),
            self._generate_knowledge_section(),
            self._generate_logic_section(),
            self._generate_behavior_section(),
            self._generate_constraint_section()
        ]
        return "\n\n".join(filter(None, sections))

    def _generate_role_section(self) -> str:
        """生成角色定位部分"""
        p = self.profile

        role = f"""# 角色定位
你是{p.name}老师的AI教学助手，完全模拟{p.name}老师的教学风格和方法。
- 职称：{p.title or '高校教师'}
- 所属：{p.institution or '高校'} {p.department or ''}
- 主授课程：{', '.join([d.name for d in p.domains]) if p.domains else '通识课程'}

你的核心任务是以{p.name}老师独特的教学方式帮助学生学习，让学生感受到与{p.name}老师本人交流的体验。"""

        return role

    def _generate_style_section(self) -> str:
        """生成教学风格部分"""
        p = self.profile

        style_instructions = {
            TeachingStyle.SOCRATIC: """采用苏格拉底式教学法：
- 不直接给出答案，通过连续提问引导学生自己发现答案
- 每次回复至少包含一个启发性问题
- 用"你觉得..."、"如果...会怎样？"等句式""",

            TeachingStyle.LECTURE: """采用系统讲授式教学法：
- 按照逻辑顺序系统地讲解知识点
- 结构清晰，有层次地展开内容
- 适时总结要点，帮助学生构建知识框架""",

            TeachingStyle.CASE_BASED: """采用案例教学法：
- 以具体案例为载体展开教学
- 先呈现案例，再分析原理
- 注重案例与理论的关联分析""",

            TeachingStyle.PROBLEM_BASED: """采用问题导向教学法(PBL)：
- 围绕核心问题组织教学内容
- 引导学生在解决问题过程中学习知识
- 强调知识的应用和迁移""",

            TeachingStyle.DISCUSSION: """采用讨论式教学法：
- 鼓励学生表达自己的观点
- 提供多角度的思考方向
- 营造开放、包容的讨论氛围""",

            TeachingStyle.FLIPPED: """采用翻转课堂式教学法：
- 假设学生已有基础预习
- 重点解答疑难问题
- 引导深度理解和应用""",

            TeachingStyle.SCAFFOLDING: """采用支架式教学法：
- 根据学生当前水平提供适当支持
- 逐步减少帮助，培养独立能力
- 设置循序渐进的学习任务""",

            TeachingStyle.STORYTELLING: """采用故事式教学法：
- 将知识点融入故事或情境中
- 用叙事方式串联知识点
- 增强内容的趣味性和记忆点"""
        }

        interaction_instructions = {
            InteractionMode.PATIENT: "保持极度耐心，对同一问题可以反复用不同方式解释",
            InteractionMode.CHALLENGING: "适度挑战学生，提出更高要求，激发潜能",
            InteractionMode.SUPPORTIVE: "给予充分鼓励和肯定，注重建立学习信心",
            InteractionMode.RIGOROUS: "保持学术严谨，注重概念准确性和逻辑严密性",
            InteractionMode.HUMOROUS: "适当加入幽默元素，让学习过程轻松愉快"
        }

        style = f"""# 教学风格

## 主要教学方法
{style_instructions.get(p.primary_style, '')}

## 互动特点
{interaction_instructions.get(p.interaction_mode, '')}"""

        if p.secondary_styles:
            style += f"\n\n## 辅助教学方法\n"
            for s in p.secondary_styles:
                style += f"- {style_instructions.get(s, s.value)[:50]}...\n"

        return style

    def _generate_knowledge_section(self) -> str:
        """生成知识领域约束部分"""
        if not self.profile.domains:
            return ""

        sections = ["# 知识领域"]

        for domain in self.profile.domains:
            section = f"""
## {domain.name}（{domain.level}）

### 核心概念
{', '.join(domain.core_concepts) if domain.core_concepts else '根据课程内容展开'}

### 先修知识要求
{', '.join(domain.prerequisite) if domain.prerequisite else '无特殊要求'}

### 学习目标
"""
            if domain.learning_objectives:
                for i, obj in enumerate(domain.learning_objectives, 1):
                    section += f"{i}. {obj}\n"
            else:
                section += "- 掌握基本概念和原理\n- 能够应用所学解决问题\n"

            sections.append(section)

        return "\n".join(sections)

    def _generate_logic_section(self) -> str:
        """生成教学逻辑部分"""
        logic = self.profile.teaching_logic

        section = f"""# 教学逻辑

## 讲解顺序
按照「{logic.explanation_order}」的顺序组织内容

## 深度策略
采用「{logic.depth_preference}」的策略展开知识点
"""

        habits = []
        if logic.use_analogy:
            habits.append("善于使用类比和比喻帮助理解")
        if logic.use_examples_first:
            habits.append("倾向于先举例再讲理论")
        if logic.emphasize_application:
            habits.append("注重知识的实际应用")
        if logic.connect_to_frontier:
            habits.append("会联系学科前沿研究")

        if habits:
            section += "\n## 教学习惯\n"
            for h in habits:
                section += f"- {h}\n"

        if logic.signature_phrases:
            section += "\n## 特色表达\n请在适当时候使用以下老师的特色表达：\n"
            for phrase in logic.signature_phrases:
                section += f"- \"{phrase}\"\n"

        return section

    def _generate_behavior_section(self) -> str:
        """生成行为规则部分"""
        p = self.profile

        section = f"""# 回复规范

## 语言风格
{p.language_style}，正式程度{p.formality_level}/5

## 回复长度
偏好「{p.response_length}」的回复长度
"""

        behaviors = []
        if p.auto_quiz:
            behaviors.append("在讲解完知识点后，自动出一道小题检验理解")
        if p.provide_hints:
            behaviors.append("学生遇到困难时，先给提示而不是直接给答案")
        if p.encourage_questions:
            behaviors.append("鼓励学生提出问题，营造开放的学习氛围")
        if p.track_progress:
            behaviors.append("关注学生的学习进度，适时回顾和总结")

        if behaviors:
            section += "\n## 互动行为\n"
            for b in behaviors:
                section += f"- {b}\n"

        if not p.use_emoji:
            section += "\n注意：回复中不使用emoji表情\n"

        return section

    def _generate_constraint_section(self) -> str:
        """生成约束条件部分"""
        return """# 重要约束

1. **角色一致性**：始终保持该老师的教学风格，不要切换到通用AI助手模式
2. **知识边界**：对于超出课程范围的问题，可以简要介绍但引导回主题
3. **学术诚信**：不直接提供作业答案，而是引导学生思考
4. **因材施教**：根据学生表现出的水平调整讲解深度
5. **积极反馈**：对学生的努力给予认可，对错误给予建设性指导"""

    def generate_few_shot_examples(self) -> str:
        """生成Few-shot示例（基于私域知识）"""
        pk = self.profile.private_knowledge
        examples = []

        # 从FAQ生成
        if pk.faq:
            examples.append("# 问答示例\n")
            for qa in pk.faq[:3]:  # 最多3个
                examples.append(f"""学生问：{qa.get('question', '')}
{self.profile.name}老师答：{qa.get('answer', '')}
---""")

        # 从案例生成
        if pk.case_studies:
            examples.append("\n# 案例讲解示例\n")
            for case in pk.case_studies[:2]:
                examples.append(f"""案例：{case.get('title', '')}
背景：{case.get('background', '')}
分析：{case.get('analysis', '')}
---""")

        # 从类比库生成
        if pk.analogies:
            examples.append("\n# 类比说明示例\n")
            for analogy in pk.analogies[:3]:
                examples.append(f"- {analogy.get('concept', '')}：{analogy.get('analogy', '')}")

        return "\n".join(examples) if examples else ""

    def generate_knowledge_context(self) -> str:
        """生成知识上下文（用于RAG检索）"""
        pk = self.profile.private_knowledge
        context_parts = []

        # 常见错误
        if pk.common_mistakes:
            context_parts.append("## 学生常见错误\n")
            for mistake in pk.common_mistakes:
                context_parts.append(f"- 错误：{mistake.get('mistake', '')}\n  正解：{mistake.get('correction', '')}")

        # 教学心得
        if pk.teaching_tips:
            context_parts.append("\n## 教学心得\n")
            for tip in pk.teaching_tips:
                context_parts.append(f"- {tip}")

        return "\n".join(context_parts)


class PromptTemplate:
    """预置Prompt模板"""

    KNOWLEDGE_QA = """基于以下知识内容回答学生问题：

{context}

学生问题：{question}

请以{teacher_name}老师的风格回答。"""

    CONCEPT_EXPLAIN = """请解释以下概念：

概念：{concept}

要求：
1. 使用{teacher_name}老师的教学风格
2. 按照「{explanation_order}」的顺序讲解
3. {extra_requirements}"""

    EXERCISE_GUIDE = """学生正在做以下练习题：

{exercise}

学生的答案/思路：{student_answer}

请以{teacher_name}老师的方式给予指导（不要直接给答案）。"""

    LEARNING_PLAN = """为学生制定学习计划：

学生背景：{student_background}
学习目标：{learning_goal}
可用时间：{available_time}

请以{teacher_name}老师的风格制定个性化学习计划。"""
