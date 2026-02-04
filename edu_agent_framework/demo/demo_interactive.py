"""
交互式Demo
==========

一个可以实际运行的交互式教学助手Demo
需要配置 OPENAI_API_KEY 环境变量
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def create_demo_agent():
    """创建Demo用的智能体"""
    from edu_agent_framework import EduAgentFactory
    from edu_agent_framework.core.teacher_profile import (
        TeacherProfile,
        TeachingStyle,
        InteractionMode,
        KnowledgeDomain,
        TeachingLogic,
        PrivateKnowledge
    )

    # 创建一个Python编程课程的教师画像
    profile = TeacherProfile(
        teacher_id="demo_python",
        name="陈老师",
        title="副教授",
        institution="示范大学",
        department="计算机系",

        primary_style=TeachingStyle.SCAFFOLDING,
        secondary_styles=[TeachingStyle.CASE_BASED],
        interaction_mode=InteractionMode.SUPPORTIVE,

        domains=[
            KnowledgeDomain(
                name="Python程序设计",
                level="本科",
                core_concepts=[
                    "变量与数据类型", "控制流", "函数", "面向对象",
                    "文件操作", "异常处理", "模块与包", "常用库"
                ],
                prerequisite=["基本的计算机操作"],
                learning_objectives=[
                    "掌握Python基础语法",
                    "能够编写解决实际问题的程序",
                    "理解面向对象编程思想",
                    "学会使用常用第三方库"
                ]
            )
        ],

        teaching_logic=TeachingLogic(
            explanation_order="概念->语法->示例代码->动手练习->常见错误->总结",
            depth_preference="先广后深",
            use_analogy=True,
            use_examples_first=True,
            emphasize_application=True,
            signature_phrases=[
                "编程最重要的是动手实践",
                "报错不可怕，每个错误都是学习机会",
                "先让代码跑起来，再考虑优化"
            ]
        ),

        private_knowledge=PrivateKnowledge(
            faq=[
                {
                    "question": "Python和其他语言比有什么优势？",
                    "answer": "Python的优势主要有：1)语法简洁易学，适合入门；2)库非常丰富，尤其在数据分析和AI领域；3)社区活跃，遇到问题容易找到解决方案。当然它也有缺点，比如运行速度相对较慢。"
                },
                {
                    "question": "列表和元组有什么区别？",
                    "answer": "最核心的区别是：列表可变(mutable)，元组不可变(immutable)。列表用[]，可以增删改元素；元组用()，创建后不能修改。选择建议：需要修改用列表，不需要修改且想防止意外修改就用元组。"
                }
            ],
            common_mistakes=[
                {
                    "mistake": "缩进不一致导致IndentationError",
                    "correction": "Python用缩进表示代码块，建议统一用4个空格，不要混用Tab和空格"
                },
                {
                    "mistake": "修改正在遍历的列表",
                    "correction": "遍历时不要直接修改列表，可以遍历副本或创建新列表"
                }
            ],
            analogies=[
                {
                    "concept": "变量",
                    "analogy": "变量就像贴在盒子上的标签，标签指向盒子里的数据。同一个盒子可以贴多个标签。"
                },
                {
                    "concept": "函数",
                    "analogy": "函数就像一台机器：投入原料（参数），经过加工，产出产品（返回值）。"
                },
                {
                    "concept": "类和对象",
                    "analogy": "类是蓝图/模具，对象是根据蓝图建造的房子/用模具做出的产品。一个类可以创建多个对象。"
                }
            ]
        ),

        language_style="亲切易懂，适合初学者",
        response_length="适中",
        formality_level=2,
        auto_quiz=False,
        provide_hints=True,
        encourage_questions=True
    )

    factory = EduAgentFactory()
    return factory.create(profile)


def run_interactive_demo():
    """运行交互式Demo"""
    print("=" * 60)
    print("  教育智能体框架 - 交互式Demo")
    print("  Python程序设计 - 陈老师AI助教")
    print("=" * 60)

    # 检查API Key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  未检测到 OPENAI_API_KEY 环境变量")
        print("请设置后重新运行：")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("\n以下进入模拟模式（不实际调用API）...")
        run_mock_demo()
        return

    print("\n正在创建智能体...")
    agent = create_demo_agent()
    print(f"✓ 智能体已就绪: {agent.profile.name}的AI助教\n")

    print("提示:")
    print("- 输入问题与AI助教对话")
    print("- 输入 'quit' 或 'exit' 退出")
    print("- 输入 'clear' 清空对话历史")
    print("- 输入 'summary' 查看学习摘要")
    print("-" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！祝学习愉快！")
                break

            if user_input.lower() == 'clear':
                agent.clear_history()
                print("对话历史已清空")
                continue

            if user_input.lower() == 'summary':
                summary = agent.get_summary()
                print("\n学习摘要:")
                print(f"  - 对话轮数: {summary['session_stats']['messages'] // 2}")
                print(f"  - 涉及主题: {summary['session_stats']['topics_covered'] or '暂无'}")
                print(f"  - 当前主题: {summary['session_stats']['current_topic'] or '暂无'}")
                continue

            # 调用智能体
            print(f"\n{agent.profile.name}:", end=" ", flush=True)

            # 使用流式输出
            for chunk in agent.chat_stream(user_input):
                print(chunk, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")


def run_mock_demo():
    """模拟Demo（不调用API）"""
    print("\n" + "=" * 60)
    print("  模拟对话演示")
    print("=" * 60)

    mock_conversations = [
        ("什么是变量？",
         "很高兴你问这个问题！变量就像贴在盒子上的标签，标签指向盒子里的数据。\n\n"
         "在Python中，创建变量非常简单：\n"
         "```python\n"
         "name = \"张三\"  # 字符串变量\n"
         "age = 20       # 整数变量\n"
         "score = 89.5   # 浮点数变量\n"
         "```\n\n"
         "注意Python不需要声明变量类型，它会自动推断。这是Python简洁的一个体现！\n\n"
         "你可以试着在Python解释器里创建几个变量练习一下。有什么问题随时问我。"),

        ("列表怎么用？",
         "列表是Python中最常用的数据结构之一，用来存储多个元素。\n\n"
         "**创建列表**：\n"
         "```python\n"
         "fruits = [\"苹果\", \"香蕉\", \"橙子\"]\n"
         "numbers = [1, 2, 3, 4, 5]\n"
         "mixed = [1, \"hello\", 3.14, True]  # 可以混合类型\n"
         "```\n\n"
         "**常用操作**：\n"
         "```python\n"
         "fruits.append(\"葡萄\")     # 添加元素\n"
         "fruits.remove(\"香蕉\")     # 删除元素\n"
         "print(fruits[0])          # 访问第一个元素\n"
         "print(len(fruits))        # 获取长度\n"
         "```\n\n"
         "编程最重要的是动手实践，建议你打开Python试一试这些操作！"),
    ]

    print("\n以下是模拟对话示例：\n")

    for question, answer in mock_conversations:
        print(f"学生: {question}")
        print(f"\n陈老师: {answer}")
        print("\n" + "-" * 40 + "\n")

    print("以上是模拟对话。配置OPENAI_API_KEY后可以进行真实交互。")


if __name__ == "__main__":
    run_interactive_demo()
