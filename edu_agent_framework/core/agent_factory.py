"""
智能体工厂
==========
一键生成教育智能体的核心模块

核心理念：
教师只需提供配置文件 → 自动生成可用的智能体
"""

from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import json
import yaml

from .teacher_profile import TeacherProfile, create_quick_profile
from .base_edu_agent import BaseEduAgent
from ..generators.prompt_generator import PromptGenerator


class EduAgentFactory:
    """
    教育智能体工厂

    使用方式：
    1. 快速模式：factory.create_quick(name, course) → 即用智能体
    2. 配置模式：factory.create_from_config(yaml_path) → 定制智能体
    3. 完整模式：factory.create(profile) → 完全控制

    技术架构：
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  教师配置    │ → │  Prompt生成  │ → │  智能体实例  │
    │  (Profile)  │    │  (Generator) │    │  (Agent)    │
    └─────────────┘    └─────────────┘    └─────────────┘
    """

    # 注册的智能体类型
    _agent_types: Dict[str, Type[BaseEduAgent]] = {}

    # 默认智能体类型
    _default_agent_type: str = "openai"

    @classmethod
    def register_agent_type(cls, name: str, agent_class: Type[BaseEduAgent]):
        """注册新的智能体类型"""
        cls._agent_types[name] = agent_class

    @classmethod
    def set_default_type(cls, name: str):
        """设置默认智能体类型"""
        if name in cls._agent_types:
            cls._default_agent_type = name

    def __init__(
        self,
        agent_type: str = None,
        llm_config: Dict[str, Any] = None,
        rag_config: Dict[str, Any] = None
    ):
        """
        初始化工厂

        Args:
            agent_type: 智能体类型（openai/local/graphrag等）
            llm_config: LLM配置（api_key, model, base_url等）
            rag_config: RAG配置（向量库、知识图谱等）
        """
        self.agent_type = agent_type or self._default_agent_type
        self.llm_config = llm_config or {}
        self.rag_config = rag_config or {}

    def create(
        self,
        profile: TeacherProfile,
        agent_type: str = None,
        **kwargs
    ) -> BaseEduAgent:
        """
        从完整Profile创建智能体

        Args:
            profile: 教师画像配置
            agent_type: 指定智能体类型（覆盖默认）
            **kwargs: 额外参数传递给智能体

        Returns:
            配置好的教育智能体实例
        """
        # 生成Prompt
        generator = PromptGenerator(profile)
        system_prompt = generator.generate_system_prompt()
        few_shot = generator.generate_few_shot_examples()

        # 获取智能体类
        type_to_use = agent_type or self.agent_type
        agent_class = self._agent_types.get(type_to_use)

        if not agent_class:
            # 默认使用OpenAI智能体
            from ..agents.openai_agent import OpenAIEduAgent
            agent_class = OpenAIEduAgent

        # 创建实例
        return agent_class(
            profile=profile,
            system_prompt=system_prompt,
            few_shot_examples=few_shot,
            llm_config=self.llm_config,
            rag_config=self.rag_config,
            **kwargs
        )

    def create_quick(
        self,
        teacher_name: str,
        course_name: str,
        style: str = "讲授式",
        **kwargs
    ) -> BaseEduAgent:
        """
        快速创建智能体（最简模式）

        只需教师姓名和课程名即可创建一个可用的智能体

        Args:
            teacher_name: 教师姓名
            course_name: 课程名称
            style: 教学风格
            **kwargs: 额外配置

        Returns:
            即用的教育智能体

        Example:
            agent = factory.create_quick("张三", "高等数学", style="启发式")
            response = agent.chat("什么是导数？")
        """
        import uuid
        teacher_id = str(uuid.uuid4())[:8]

        profile = create_quick_profile(
            teacher_id=teacher_id,
            name=teacher_name,
            course_name=course_name,
            style=style,
            **kwargs
        )

        return self.create(profile)

    def create_from_config(
        self,
        config_path: str,
        agent_type: str = None
    ) -> BaseEduAgent:
        """
        从配置文件创建智能体

        支持JSON和YAML格式

        Args:
            config_path: 配置文件路径
            agent_type: 指定智能体类型

        Returns:
            配置好的教育智能体
        """
        path = Path(config_path)

        if path.suffix in ['.yaml', '.yml']:
            profile = TeacherProfile.from_yaml(config_path)
        elif path.suffix == '.json':
            profile = TeacherProfile.from_json(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {path.suffix}")

        return self.create(profile, agent_type)

    def create_from_template(
        self,
        template_name: str,
        teacher_name: str,
        course_name: str,
        **overrides
    ) -> BaseEduAgent:
        """
        从预置模板创建智能体

        Args:
            template_name: 模板名称（理工科严谨型/文科启发型/实践案例型）
            teacher_name: 教师姓名
            course_name: 课程名称
            **overrides: 覆盖模板的配置

        Returns:
            基于模板的智能体
        """
        from .teacher_profile import PROFILE_TEMPLATES, TeachingStyle, InteractionMode, TeachingLogic
        import uuid

        template = PROFILE_TEMPLATES.get(template_name, {})

        # 解析模板中的枚举值
        style_map = {s.value: s for s in TeachingStyle}
        mode_map = {m.value: m for m in InteractionMode}

        profile = TeacherProfile(
            teacher_id=str(uuid.uuid4())[:8],
            name=teacher_name,
            primary_style=style_map.get(
                template.get('primary_style', '讲授式'),
                TeachingStyle.LECTURE
            ),
            interaction_mode=mode_map.get(
                template.get('interaction_mode', '鼓励支持型'),
                InteractionMode.SUPPORTIVE
            ),
            teaching_logic=TeachingLogic(**template.get('teaching_logic', {})),
            language_style=template.get('language_style', '学术但易懂'),
            formality_level=template.get('formality_level', 3),
            **overrides
        )

        # 添加课程领域
        from .teacher_profile import KnowledgeDomain
        profile.domains = [KnowledgeDomain(name=course_name, level="本科")]

        return self.create(profile)

    def batch_create(
        self,
        config_paths: List[str]
    ) -> Dict[str, BaseEduAgent]:
        """
        批量创建智能体

        Args:
            config_paths: 配置文件路径列表

        Returns:
            {teacher_id: agent} 字典
        """
        agents = {}
        for path in config_paths:
            try:
                agent = self.create_from_config(path)
                agents[agent.profile.teacher_id] = agent
            except Exception as e:
                print(f"创建失败 {path}: {e}")
        return agents

    @staticmethod
    def generate_config_template(
        output_path: str,
        format: str = "yaml"
    ) -> str:
        """
        生成配置文件模板（供教师填写）

        Args:
            output_path: 输出路径
            format: 格式（yaml/json）

        Returns:
            生成的文件路径
        """
        template_profile = TeacherProfile(
            teacher_id="示例ID（将自动生成）",
            name="您的姓名",
            title="您的职称（如：副教授）",
            institution="您的学校",
            department="您的院系",
        )

        # 添加示例领域
        from .teacher_profile import KnowledgeDomain, PrivateKnowledge, TeachingLogic

        template_profile.domains = [
            KnowledgeDomain(
                name="课程名称（如：数据结构）",
                level="本科/硕士/博士",
                core_concepts=["概念1", "概念2", "概念3"],
                prerequisite=["先修课程1"],
                learning_objectives=["学习目标1", "学习目标2"]
            )
        ]

        template_profile.teaching_logic = TeachingLogic(
            explanation_order="概念->原理->例子->应用",
            signature_phrases=["您的口头禅或特色表达"]
        )

        template_profile.private_knowledge = PrivateKnowledge(
            faq=[{"question": "学生常问的问题？", "answer": "您的标准回答"}],
            common_mistakes=[{"mistake": "学生常犯的错误", "correction": "正确的理解"}],
            analogies=[{"concept": "难懂的概念", "analogy": "您常用的类比解释"}]
        )

        path = Path(output_path)
        if format == "yaml":
            template_profile.to_yaml(str(path))
        else:
            template_profile.to_json(str(path))

        return str(path)


# === 便捷函数 ===

def quick_agent(
    teacher_name: str,
    course_name: str,
    api_key: str = None,
    **kwargs
) -> BaseEduAgent:
    """
    最简单的创建方式

    Example:
        agent = quick_agent("李老师", "线性代数")
        print(agent.chat("特征值怎么求？"))
    """
    factory = EduAgentFactory(
        llm_config={"api_key": api_key} if api_key else {}
    )
    return factory.create_quick(teacher_name, course_name, **kwargs)
