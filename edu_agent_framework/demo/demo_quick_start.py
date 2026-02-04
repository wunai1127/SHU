"""
快速开始Demo
============

展示如何用最少的代码创建一个个性化教学智能体
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from edu_agent_framework import EduAgentFactory, TeachingStyle


def demo_quick_create():
    """
    演示1: 极简创建（3行代码）
    """
    print("=" * 50)
    print("演示1: 极简创建智能体")
    print("=" * 50)

    # 创建工厂
    factory = EduAgentFactory()

    # 一行代码创建智能体
    agent = factory.create_quick(
        teacher_name="王老师",
        course_name="线性代数",
        style="讲授式"
    )

    print(f"智能体已创建: {agent.agent_id}")
    print(f"教师: {agent.profile.name}")
    print(f"课程: {[d.name for d in agent.profile.domains]}")
    print(f"风格: {agent.profile.primary_style.value}")

    # 模拟对话（需要配置API Key才能真正运行）
    print("\n模拟对话:")
    print("学生: 特征值怎么求？")
    print("王老师: [需要配置OPENAI_API_KEY才能运行]")

    return agent


def demo_template_create():
    """
    演示2: 使用预置模板创建
    """
    print("\n" + "=" * 50)
    print("演示2: 使用预置模板创建")
    print("=" * 50)

    factory = EduAgentFactory()

    # 使用"理工科严谨型"模板
    agent = factory.create_from_template(
        template_name="理工科严谨型",
        teacher_name="李教授",
        course_name="数据结构"
    )

    print(f"智能体已创建: {agent.agent_id}")
    print(f"教学风格: {agent.profile.primary_style.value}")
    print(f"互动模式: {agent.profile.interaction_mode.value}")
    print(f"讲解顺序: {agent.profile.teaching_logic.explanation_order}")

    return agent


def demo_full_profile():
    """
    演示3: 完整配置创建（展示所有可配置项）
    """
    print("\n" + "=" * 50)
    print("演示3: 完整配置创建")
    print("=" * 50)

    from edu_agent_framework.core.teacher_profile import (
        TeacherProfile,
        TeachingStyle,
        InteractionMode,
        AssessmentStyle,
        KnowledgeDomain,
        TeachingLogic,
        PrivateKnowledge
    )

    # 创建完整的教师画像
    profile = TeacherProfile(
        teacher_id="demo_001",
        name="张明远",
        title="副教授",
        institution="上海大学",
        department="计算机学院",

        # 教学风格
        primary_style=TeachingStyle.CASE_BASED,
        secondary_styles=[TeachingStyle.SOCRATIC],
        interaction_mode=InteractionMode.CHALLENGING,
        assessment_style=AssessmentStyle.FORMATIVE,

        # 知识领域
        domains=[
            KnowledgeDomain(
                name="人工智能导论",
                level="本科",
                core_concepts=["机器学习", "深度学习", "神经网络", "自然语言处理"],
                prerequisite=["高等数学", "线性代数", "概率论", "Python编程"],
                learning_objectives=[
                    "理解AI的基本概念和发展历程",
                    "掌握机器学习的核心算法",
                    "能够使用Python实现简单的AI应用"
                ]
            )
        ],

        # 教学逻辑
        teaching_logic=TeachingLogic(
            explanation_order="问题引入->概念讲解->案例分析->代码实践->总结提升",
            depth_preference="先广后深",
            use_analogy=True,
            use_examples_first=True,
            emphasize_application=True,
            connect_to_frontier=True,
            signature_phrases=[
                "这个问题很好，让我们从本质上来思考",
                "先不急着看答案，你觉得会是什么？",
                "在工业界，这个技术是这样应用的..."
            ]
        ),

        # 私域知识
        private_knowledge=PrivateKnowledge(
            faq=[
                {
                    "question": "深度学习和机器学习有什么区别？",
                    "answer": "深度学习是机器学习的子集。机器学习包括决策树、SVM等传统方法，而深度学习专指使用多层神经网络的方法。深度学习的优势在于能自动学习特征，但需要更多数据和算力。"
                },
                {
                    "question": "学AI需要什么数学基础？",
                    "answer": "主要需要三门数学：线性代数（理解矩阵运算、特征分解）、概率论（理解贝叶斯、分布）、微积分（理解梯度下降）。建议先把这三门的核心内容过一遍再深入学习。"
                }
            ],
            common_mistakes=[
                {
                    "mistake": "混淆过拟合和欠拟合",
                    "correction": "过拟合是模型在训练集上表现好但测试集差（记忆了训练数据），欠拟合是两个都差（模型太简单）"
                }
            ],
            analogies=[
                {
                    "concept": "神经网络",
                    "analogy": "可以把神经网络想象成一个黑盒工厂：输入原材料（数据），经过多道工序（层），最终产出产品（预测结果）"
                },
                {
                    "concept": "梯度下降",
                    "analogy": "就像在雾中下山——你看不到全局，但可以感受脚下哪个方向更陡，就往哪个方向走一小步"
                }
            ],
            teaching_tips=[
                "讲神经网络时先从感知机开始，再逐步堆叠",
                "CNN用图像处理的例子最直观",
                "RNN/LSTM用文本预测的例子学生容易理解"
            ]
        ),

        # 个性化配置
        language_style="技术性但接地气",
        response_length="详细",
        formality_level=3,
        auto_quiz=True,
        provide_hints=True,
        encourage_questions=True,
        track_progress=True
    )

    # 创建智能体
    factory = EduAgentFactory()
    agent = factory.create(profile)

    print(f"智能体已创建: {agent.agent_id}")
    print(f"教师: {profile.name} ({profile.title})")
    print(f"院校: {profile.institution} {profile.department}")
    print(f"课程: {[d.name for d in profile.domains]}")
    print(f"核心概念: {profile.domains[0].core_concepts}")
    print(f"特色表达: {profile.teaching_logic.signature_phrases}")

    # 导出配置供参考
    profile.to_yaml("demo/demo_profile.yaml")
    print("\n配置已导出到: demo/demo_profile.yaml")

    return agent


def demo_generate_prompt():
    """
    演示4: 查看生成的System Prompt
    """
    print("\n" + "=" * 50)
    print("演示4: 查看生成的System Prompt")
    print("=" * 50)

    from edu_agent_framework import EduAgentFactory
    from edu_agent_framework.generators import PromptGenerator

    factory = EduAgentFactory()
    agent = factory.create_quick("赵老师", "操作系统", style="苏格拉底式")

    print("生成的System Prompt预览（前1000字符）:")
    print("-" * 40)
    print(agent.system_prompt[:1000] + "...")

    return agent


def demo_batch_create():
    """
    演示5: 批量创建多个智能体
    """
    print("\n" + "=" * 50)
    print("演示5: 批量创建多个智能体")
    print("=" * 50)

    factory = EduAgentFactory()

    teachers = [
        ("王老师", "高等数学", "讲授式"),
        ("李老师", "大学物理", "案例式"),
        ("张老师", "程序设计", "问题导向式"),
        ("刘老师", "英语写作", "苏格拉底式"),
    ]

    agents = {}
    for name, course, style in teachers:
        agent = factory.create_quick(name, course, style=style)
        agents[name] = agent
        print(f"✓ 已创建: {name} - {course} ({style})")

    print(f"\n共创建 {len(agents)} 个智能体")
    return agents


if __name__ == "__main__":
    print("教育智能体框架 - 快速开始Demo\n")

    # 运行所有演示
    demo_quick_create()
    demo_template_create()
    demo_full_profile()
    demo_generate_prompt()
    demo_batch_create()

    print("\n" + "=" * 50)
    print("所有演示完成！")
    print("=" * 50)
