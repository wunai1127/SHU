"""
基础教育智能体
==============
所有教育智能体的基类，封装核心功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
import json
import time

from .teacher_profile import TeacherProfile


@dataclass
class Message:
    """对话消息"""
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningState:
    """学习状态跟踪"""
    topics_covered: List[str] = field(default_factory=list)
    questions_asked: int = 0
    correct_answers: int = 0
    misconceptions: List[str] = field(default_factory=list)
    current_topic: str = ""
    understanding_level: Dict[str, float] = field(default_factory=dict)  # topic -> 0-1


class BaseEduAgent(ABC):
    """
    教育智能体基类

    核心功能：
    1. 个性化对话：基于教师画像生成回复
    2. 学习跟踪：记录学生学习状态
    3. 知识检索：从私域知识中检索相关内容
    4. 上下文管理：管理对话历史
    """

    def __init__(
        self,
        profile: TeacherProfile,
        system_prompt: str,
        few_shot_examples: str = "",
        max_history: int = 20
    ):
        self.profile = profile
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.max_history = max_history

        # 对话历史
        self.history: List[Message] = []

        # 学习状态
        self.learning_state = LearningState()

        # 知识缓存
        self._knowledge_cache: Dict[str, str] = {}

    @property
    def agent_id(self) -> str:
        """智能体唯一ID"""
        return f"edu_agent_{self.profile.teacher_id}"

    def _build_messages(self, user_input: str, context: str = "") -> List[Dict[str, str]]:
        """构建发送给LLM的消息列表"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # 添加few-shot示例
        if self.few_shot_examples:
            messages.append({
                "role": "system",
                "content": f"参考示例：\n{self.few_shot_examples}"
            })

        # 添加知识上下文
        if context:
            messages.append({
                "role": "system",
                "content": f"相关知识参考：\n{context}"
            })

        # 添加对话历史（限制长度）
        history_to_use = self.history[-self.max_history:]
        for msg in history_to_use:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

    def add_to_history(self, role: str, content: str, **metadata):
        """添加消息到历史"""
        self.history.append(Message(
            role=role,
            content=content,
            metadata=metadata
        ))

    def clear_history(self):
        """清空对话历史"""
        self.history = []
        self.learning_state = LearningState()

    @abstractmethod
    def chat(self, user_input: str, **kwargs) -> str:
        """
        核心对话方法（需子类实现）

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Returns:
            智能体回复
        """
        pass

    @abstractmethod
    def chat_stream(self, user_input: str, **kwargs) -> Generator[str, None, None]:
        """
        流式对话方法（需子类实现）

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Yields:
            回复文本片段
        """
        pass

    def retrieve_knowledge(self, query: str) -> str:
        """
        从私域知识中检索相关内容
        子类可覆写以实现更复杂的RAG
        """
        # 基础实现：简单的关键词匹配
        pk = self.profile.private_knowledge
        relevant = []

        # 搜索FAQ
        for qa in pk.faq:
            if query in qa.get('question', '') or query in qa.get('answer', ''):
                relevant.append(f"Q: {qa['question']}\nA: {qa['answer']}")

        # 搜索案例
        for case in pk.case_studies:
            case_text = json.dumps(case, ensure_ascii=False)
            if query in case_text:
                relevant.append(f"案例: {case.get('title', '')}\n{case.get('analysis', '')}")

        # 搜索常见错误
        for mistake in pk.common_mistakes:
            if query in mistake.get('mistake', '') or query in mistake.get('correction', ''):
                relevant.append(f"注意：{mistake['mistake']} → {mistake['correction']}")

        return "\n---\n".join(relevant[:5])  # 最多返回5条

    def update_learning_state(self, user_input: str, response: str):
        """更新学习状态（可被子类覆写）"""
        self.learning_state.questions_asked += 1

        # 简单的主题追踪
        for domain in self.profile.domains:
            for concept in domain.core_concepts:
                if concept in user_input or concept in response:
                    if concept not in self.learning_state.topics_covered:
                        self.learning_state.topics_covered.append(concept)
                    self.learning_state.current_topic = concept

    def get_summary(self) -> Dict[str, Any]:
        """获取学习会话摘要"""
        return {
            "agent_id": self.agent_id,
            "teacher": self.profile.name,
            "course": [d.name for d in self.profile.domains],
            "session_stats": {
                "messages": len(self.history),
                "topics_covered": self.learning_state.topics_covered,
                "questions_asked": self.learning_state.questions_asked,
                "current_topic": self.learning_state.current_topic
            }
        }

    def export_history(self) -> List[Dict[str, Any]]:
        """导出对话历史"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in self.history
        ]
