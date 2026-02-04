# 教育智能体框架核心模块
from .teacher_profile import TeacherProfile, TeachingStyle, KnowledgeDomain
from .agent_factory import EduAgentFactory
from .base_edu_agent import BaseEduAgent

__all__ = [
    'TeacherProfile',
    'TeachingStyle',
    'KnowledgeDomain',
    'EduAgentFactory',
    'BaseEduAgent'
]
