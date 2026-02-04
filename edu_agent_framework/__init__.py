"""
教育智能体速成框架 (Education Agent Framework)
==============================================

专为高校教师设计的个性化AI教学助手生成框架

核心功能：
1. 教师画像配置 - 定义教学风格、知识领域、私域素材
2. 智能体工厂 - 一键生成个性化教学智能体
3. 知识处理 - 处理PPT、PDF等教学文档
4. RAG检索 - 基于私域知识的精准回答

快速开始：
```python
from edu_agent_framework import EduAgentFactory

# 方式1: 极简创建
factory = EduAgentFactory()
agent = factory.create_quick("张三", "高等数学")
response = agent.chat("什么是导数？")

# 方式2: 配置文件创建
agent = factory.create_from_config("teacher_config.yaml")

# 方式3: 模板创建
agent = factory.create_from_template("理工科严谨型", "李四", "数据结构")
```
"""

__version__ = "0.1.0"
__author__ = "SHU Education AI Lab"

from .core import (
    TeacherProfile,
    TeachingStyle,
    KnowledgeDomain,
    EduAgentFactory,
    BaseEduAgent
)

from .generators import PromptGenerator

from .knowledge import DocumentProcessor, KnowledgeIndexer

__all__ = [
    # 核心类
    'TeacherProfile',
    'TeachingStyle',
    'KnowledgeDomain',
    'EduAgentFactory',
    'BaseEduAgent',
    # 生成器
    'PromptGenerator',
    # 知识处理
    'DocumentProcessor',
    'KnowledgeIndexer',
]
