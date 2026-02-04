"""
教育智能体速成框架 (Education Agent Framework)
==============================================

专为高校教师设计的个性化AI教学助手生成框架

核心功能：
1. 教师画像配置 - 定义教学风格、知识领域、私域素材
2. 教学DNA指纹 - 量化教学理念和逻辑，让AI更像教师本人
3. 智能体工厂 - 一键生成个性化教学智能体
4. 知识处理 - 处理PPT、PDF等教学文档
5. RAG检索 - 基于私域知识的精准回答
6. Web界面 - 可视化创建和管理智能体

快速开始：
```python
from edu_agent_framework import EduAgentFactory

# 方式1: 极简创建
factory = EduAgentFactory()
agent = factory.create_quick("张三", "高等数学")
response = agent.chat("什么是导数？")

# 方式2: 配置文件创建
agent = factory.create_from_config("teacher_config.yaml")

# 方式3: 模板创建（带教学DNA）
agent = factory.create_from_template("理工科严谨型", "李四", "数据结构")
```

启动Web界面：
```bash
python run_web.py  # 访问 http://localhost:8001
```
"""

__version__ = "0.2.0"
__author__ = "SHU Education AI Lab"

from .core import (
    # 教师画像
    TeacherProfile,
    TeachingStyle,
    InteractionMode,
    KnowledgeDomain,
    TeachingLogic,
    PrivateKnowledge,
    create_quick_profile,
    PROFILE_TEMPLATES,
    # 智能体
    EduAgentFactory,
    quick_agent,
    BaseEduAgent,
    # 教学DNA
    TeachingDNA,
    DNA_TEMPLATES,
    DNAExtractor
)

from .generators import PromptGenerator

from .knowledge import DocumentProcessor, KnowledgeIndexer

__all__ = [
    # 核心类
    'TeacherProfile',
    'TeachingStyle',
    'InteractionMode',
    'KnowledgeDomain',
    'TeachingLogic',
    'PrivateKnowledge',
    'create_quick_profile',
    'PROFILE_TEMPLATES',
    # 智能体
    'EduAgentFactory',
    'quick_agent',
    'BaseEduAgent',
    # 教学DNA
    'TeachingDNA',
    'DNA_TEMPLATES',
    'DNAExtractor',
    # 生成器
    'PromptGenerator',
    # 知识处理
    'DocumentProcessor',
    'KnowledgeIndexer',
]
