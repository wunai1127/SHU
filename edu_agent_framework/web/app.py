"""
教育智能体速成平台 - Web界面
==============================
端口: 8001

启动命令:
    streamlit run web/app.py --server.port 8001
"""

import streamlit as st
import sys
import os
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from edu_agent_framework.core import (
    TeacherProfile,
    TeachingStyle,
    InteractionMode,
    KnowledgeDomain,
    TeachingLogic,
    PrivateKnowledge,
    EduAgentFactory,
    TeachingDNA,
    DNA_TEMPLATES,
    LogicPattern,
    InteractionPattern,
    FeedbackStyle
)
from edu_agent_framework.generators import PromptGenerator

# 页面配置
st.set_page_config(
    page_title="教育智能体速成平台",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化session state"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'dna' not in st.session_state:
        st.session_state.dna = None
    if 'page' not in st.session_state:
        st.session_state.page = 'create'


def create_agent_page():
    """创建智能体页面"""
    st.markdown('<p class="main-header">🎓 教育智能体速成平台</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">3分钟创建具有您教学风格的AI助教</p>', unsafe_allow_html=True)

    # 创建方式选择
    create_mode = st.radio(
        "选择创建方式",
        ["快速创建", "使用模板", "完整配置"],
        horizontal=True
    )

    if create_mode == "快速创建":
        quick_create_form()
    elif create_mode == "使用模板":
        template_create_form()
    else:
        full_config_form()


def quick_create_form():
    """快速创建表单"""
    st.subheader("⚡ 快速创建")
    st.info("只需填写基本信息，即可创建一个可用的AI助教")

    col1, col2 = st.columns(2)

    with col1:
        teacher_name = st.text_input("您的姓名", placeholder="例如：张老师")
        course_name = st.text_input("课程名称", placeholder="例如：高等数学")

    with col2:
        style = st.selectbox(
            "教学风格",
            ["讲授式", "苏格拉底式", "案例式", "问题导向式", "讨论式", "支架式"]
        )
        interaction = st.selectbox(
            "互动模式",
            ["鼓励支持型", "耐心细致型", "挑战激励型", "严谨学术型", "幽默风趣型"]
        )

    # 可选：添加几个口头禅
    st.write("**可选：添加您的特色表达**")
    catchphrases = st.text_area(
        "口头禅/特色表达（每行一个）",
        placeholder="例如：\n这个问题很好...\n换个角度想...",
        height=100
    )

    if st.button("🚀 创建智能体", type="primary"):
        if not teacher_name or not course_name:
            st.error("请填写姓名和课程名称")
            return

        with st.spinner("正在创建智能体..."):
            try:
                # 创建Profile
                style_map = {s.value: s for s in TeachingStyle}
                mode_map = {m.value: m for m in InteractionMode}

                profile = TeacherProfile(
                    teacher_id=f"web_{teacher_name}",
                    name=teacher_name,
                    primary_style=style_map.get(style, TeachingStyle.LECTURE),
                    interaction_mode=mode_map.get(interaction, InteractionMode.SUPPORTIVE),
                    domains=[KnowledgeDomain(name=course_name, level="本科")]
                )

                # 添加口头禅
                if catchphrases.strip():
                    profile.teaching_logic.signature_phrases = [
                        p.strip() for p in catchphrases.strip().split('\n') if p.strip()
                    ]

                # 创建智能体
                factory = EduAgentFactory()
                agent = factory.create(profile)

                st.session_state.agent = agent
                st.session_state.profile = profile
                st.session_state.page = 'chat'
                st.rerun()

            except Exception as e:
                st.error(f"创建失败: {str(e)}")


def template_create_form():
    """模板创建表单"""
    st.subheader("📋 使用预置模板")

    col1, col2 = st.columns(2)

    with col1:
        teacher_name = st.text_input("您的姓名", placeholder="例如：李教授", key="tpl_name")
        course_name = st.text_input("课程名称", placeholder="例如：数据结构", key="tpl_course")

    with col2:
        template_name = st.selectbox(
            "选择模板",
            list(DNA_TEMPLATES.keys())
        )

    # 显示模板预览
    if template_name in DNA_TEMPLATES:
        dna = DNA_TEMPLATES[template_name]
        with st.expander("查看模板详情", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**语言风格**")
                st.write(f"- 口头禅: {', '.join(dna.language.catchphrases[:3])}")
                st.write(f"- 正式度: {dna.language.formality_score}")

            with col2:
                st.write("**教学逻辑**")
                st.write(f"- 推理模式: {dna.logic.primary_pattern.value}")
                st.write(f"- 互动方式: {dna.interaction.primary_mode.value}")

    if st.button("🚀 使用此模板创建", type="primary", key="tpl_create"):
        if not teacher_name or not course_name:
            st.error("请填写姓名和课程名称")
            return

        with st.spinner("正在创建智能体..."):
            try:
                # 获取DNA模板
                dna = DNA_TEMPLATES[template_name]
                dna.teacher_id = f"web_{teacher_name}"
                dna.teacher_name = teacher_name

                # 创建Profile
                profile = TeacherProfile(
                    teacher_id=f"web_{teacher_name}",
                    name=teacher_name,
                    domains=[KnowledgeDomain(name=course_name, level="本科")]
                )

                # 创建智能体（带DNA）
                factory = EduAgentFactory()
                agent = factory.create(profile)
                agent.teaching_dna = dna

                st.session_state.agent = agent
                st.session_state.profile = profile
                st.session_state.dna = dna
                st.session_state.page = 'chat'
                st.rerun()

            except Exception as e:
                st.error(f"创建失败: {str(e)}")


def full_config_form():
    """完整配置表单"""
    st.subheader("🛠️ 完整配置")

    # 基本信息
    st.write("### 基本信息")
    col1, col2 = st.columns(2)
    with col1:
        teacher_name = st.text_input("姓名", key="full_name")
        title = st.text_input("职称", placeholder="副教授", key="full_title")
    with col2:
        institution = st.text_input("学校", key="full_inst")
        department = st.text_input("院系", key="full_dept")

    # 课程信息
    st.write("### 课程信息")
    course_name = st.text_input("课程名称", key="full_course")
    core_concepts = st.text_input(
        "核心概念（逗号分隔）",
        placeholder="概念1, 概念2, 概念3"
    )

    # 教学风格
    st.write("### 教学风格")
    col1, col2 = st.columns(2)
    with col1:
        style = st.selectbox(
            "主要教学方法",
            [s.value for s in TeachingStyle],
            key="full_style"
        )
    with col2:
        interaction = st.selectbox(
            "互动模式",
            [m.value for m in InteractionMode],
            key="full_interaction"
        )

    # 教学逻辑
    st.write("### 教学逻辑")
    explanation_order = st.text_input(
        "讲解顺序",
        value="概念→原理→例子→应用→总结",
        key="full_order"
    )

    col1, col2 = st.columns(2)
    with col1:
        use_analogy = st.checkbox("善用类比", value=True)
        use_examples_first = st.checkbox("先举例后讲理论")
    with col2:
        emphasize_application = st.checkbox("注重实际应用", value=True)
        connect_to_frontier = st.checkbox("联系学科前沿")

    # 特色表达
    st.write("### 特色表达")
    catchphrases = st.text_area(
        "口头禅/特色表达（每行一个）",
        height=100,
        key="full_phrases"
    )

    # 私域知识
    st.write("### 私域知识")
    with st.expander("添加FAQ（学生常问问题）"):
        faq_q1 = st.text_input("问题1", key="faq_q1")
        faq_a1 = st.text_area("回答1", key="faq_a1", height=100)
        faq_q2 = st.text_input("问题2", key="faq_q2")
        faq_a2 = st.text_area("回答2", key="faq_a2", height=100)

    with st.expander("添加常见错误"):
        mistake1 = st.text_input("错误1", key="mistake1")
        correction1 = st.text_input("正确理解1", key="correction1")

    with st.expander("添加类比"):
        concept1 = st.text_input("难懂概念", key="concept1")
        analogy1 = st.text_input("类比说明", key="analogy1")

    if st.button("🚀 创建智能体", type="primary", key="full_create"):
        if not teacher_name or not course_name:
            st.error("请填写姓名和课程名称")
            return

        with st.spinner("正在创建智能体..."):
            try:
                style_map = {s.value: s for s in TeachingStyle}
                mode_map = {m.value: m for m in InteractionMode}

                # 构建FAQ
                faq = []
                if faq_q1 and faq_a1:
                    faq.append({"question": faq_q1, "answer": faq_a1})
                if faq_q2 and faq_a2:
                    faq.append({"question": faq_q2, "answer": faq_a2})

                # 构建常见错误
                mistakes = []
                if mistake1 and correction1:
                    mistakes.append({"mistake": mistake1, "correction": correction1})

                # 构建类比
                analogies = []
                if concept1 and analogy1:
                    analogies.append({"concept": concept1, "analogy": analogy1})

                # 创建完整Profile
                profile = TeacherProfile(
                    teacher_id=f"web_{teacher_name}",
                    name=teacher_name,
                    title=title,
                    institution=institution,
                    department=department,
                    primary_style=style_map.get(style, TeachingStyle.LECTURE),
                    interaction_mode=mode_map.get(interaction, InteractionMode.SUPPORTIVE),
                    domains=[
                        KnowledgeDomain(
                            name=course_name,
                            level="本科",
                            core_concepts=[c.strip() for c in core_concepts.split(',') if c.strip()]
                        )
                    ],
                    teaching_logic=TeachingLogic(
                        explanation_order=explanation_order,
                        use_analogy=use_analogy,
                        use_examples_first=use_examples_first,
                        emphasize_application=emphasize_application,
                        connect_to_frontier=connect_to_frontier,
                        signature_phrases=[p.strip() for p in catchphrases.split('\n') if p.strip()]
                    ),
                    private_knowledge=PrivateKnowledge(
                        faq=faq,
                        common_mistakes=mistakes,
                        analogies=analogies
                    )
                )

                # 创建智能体
                factory = EduAgentFactory()
                agent = factory.create(profile)

                st.session_state.agent = agent
                st.session_state.profile = profile
                st.session_state.page = 'chat'
                st.rerun()

            except Exception as e:
                st.error(f"创建失败: {str(e)}")


def chat_page():
    """聊天页面"""
    agent = st.session_state.agent
    profile = st.session_state.profile

    # 侧边栏显示智能体信息
    with st.sidebar:
        st.write("### 当前智能体")
        st.write(f"**教师**: {profile.name}")
        st.write(f"**课程**: {', '.join([d.name for d in profile.domains])}")
        st.write(f"**风格**: {profile.primary_style.value}")

        if st.button("🔄 创建新智能体"):
            st.session_state.agent = None
            st.session_state.profile = None
            st.session_state.messages = []
            st.session_state.page = 'create'
            st.rerun()

        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            if agent:
                agent.clear_history()
            st.rerun()

        # 显示学习摘要
        if agent and agent.learning_state.topics_covered:
            st.write("### 学习进度")
            st.write(f"涉及主题: {', '.join(agent.learning_state.topics_covered)}")

    # 主聊天区域
    st.markdown(f"## 💬 与{profile.name}的AI助教对话")

    # 显示聊天历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if prompt := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取AI回复
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # 检查API Key
            api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                response = f"""⚠️ **API Key 未配置**

请设置环境变量后重启服务：
```bash
export LLM_API_KEY="your-deepseek-api-key"
export LLM_BASE_URL="https://api.deepseek.com/v1"
```

---

**模拟回复**（{profile.name}老师的风格）：

您好！很高兴为您解答关于"{prompt}"的问题。

作为{profile.name}老师的AI助教，我会以{profile.primary_style.value}的方式帮助您理解这个概念。

{profile.teaching_logic.signature_phrases[0] if profile.teaching_logic.signature_phrases else '让我们一起来探讨这个问题。'}
"""
                message_placeholder.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                try:
                    # 流式输出
                    full_response = ""
                    for chunk in agent.chat_stream(prompt):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    error_msg = f"调用API失败: {str(e)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """主函数"""
    init_session_state()

    if st.session_state.page == 'create' or st.session_state.agent is None:
        create_agent_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
