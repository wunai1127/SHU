"""
OpenAI教育智能体实现
====================
基于OpenAI API的教育智能体
"""

from typing import Dict, Any, Generator, Optional
import os

from ..core.base_edu_agent import BaseEduAgent
from ..core.teacher_profile import TeacherProfile


class OpenAIEduAgent(BaseEduAgent):
    """
    基于OpenAI API的教育智能体

    支持：
    - GPT-4o/GPT-4/GPT-3.5等模型
    - 流式输出
    - 自定义base_url（支持其他兼容API）
    """

    def __init__(
        self,
        profile: TeacherProfile,
        system_prompt: str,
        few_shot_examples: str = "",
        llm_config: Dict[str, Any] = None,
        rag_config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(profile, system_prompt, few_shot_examples)

        llm_config = llm_config or {}

        # LLM配置
        self.api_key = llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
        self.base_url = llm_config.get('base_url') or os.getenv('OPENAI_BASE_URL')
        self.model = llm_config.get('model') or os.getenv('OPENAI_LLM_MODEL', 'gpt-4o')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2000)

        # RAG配置
        self.rag_config = rag_config or {}
        self.enable_rag = self.rag_config.get('enable', False)

        # 初始化客户端
        self._client = None

    @property
    def client(self):
        """延迟初始化OpenAI客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("请安装openai: pip install openai")
        return self._client

    def chat(self, user_input: str, **kwargs) -> str:
        """
        同步对话

        Args:
            user_input: 用户输入
            **kwargs: 额外参数
                - use_rag: 是否启用RAG检索
                - context: 手动提供的上下文

        Returns:
            智能体回复
        """
        # 获取上下文
        context = kwargs.get('context', '')
        if kwargs.get('use_rag', self.enable_rag) and not context:
            context = self.retrieve_knowledge(user_input)

        # 构建消息
        messages = self._build_messages(user_input, context)

        # 调用API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        reply = response.choices[0].message.content

        # 更新历史和状态
        self.add_to_history("user", user_input)
        self.add_to_history("assistant", reply)
        self.update_learning_state(user_input, reply)

        return reply

    def chat_stream(self, user_input: str, **kwargs) -> Generator[str, None, None]:
        """
        流式对话

        Args:
            user_input: 用户输入
            **kwargs: 额外参数

        Yields:
            回复文本片段
        """
        # 获取上下文
        context = kwargs.get('context', '')
        if kwargs.get('use_rag', self.enable_rag) and not context:
            context = self.retrieve_knowledge(user_input)

        # 构建消息
        messages = self._build_messages(user_input, context)

        # 流式调用API
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        full_reply = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                yield content

        # 更新历史和状态
        self.add_to_history("user", user_input)
        self.add_to_history("assistant", full_reply)
        self.update_learning_state(user_input, full_reply)

    async def achat(self, user_input: str, **kwargs) -> str:
        """异步对话"""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("请安装openai: pip install openai")

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        context = kwargs.get('context', '')
        if kwargs.get('use_rag', self.enable_rag) and not context:
            context = self.retrieve_knowledge(user_input)

        messages = self._build_messages(user_input, context)

        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        reply = response.choices[0].message.content

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", reply)
        self.update_learning_state(user_input, reply)

        return reply


class LocalLLMEduAgent(BaseEduAgent):
    """
    本地LLM教育智能体

    支持通过Ollama或其他本地服务运行的模型
    """

    def __init__(
        self,
        profile: TeacherProfile,
        system_prompt: str,
        few_shot_examples: str = "",
        llm_config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(profile, system_prompt, few_shot_examples)

        llm_config = llm_config or {}
        self.base_url = llm_config.get('base_url', 'http://localhost:11434')
        self.model = llm_config.get('model', 'qwen2.5:7b')

    def chat(self, user_input: str, **kwargs) -> str:
        """调用本地LLM"""
        import requests

        context = kwargs.get('context', '')
        messages = self._build_messages(user_input, context)

        # Ollama API格式
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False
            }
        )

        result = response.json()
        reply = result.get('message', {}).get('content', '')

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", reply)
        self.update_learning_state(user_input, reply)

        return reply

    def chat_stream(self, user_input: str, **kwargs) -> Generator[str, None, None]:
        """流式调用本地LLM"""
        import requests

        context = kwargs.get('context', '')
        messages = self._build_messages(user_input, context)

        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": True
            },
            stream=True
        )

        full_reply = ""
        for line in response.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if 'message' in data and 'content' in data['message']:
                    content = data['message']['content']
                    full_reply += content
                    yield content

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", full_reply)
        self.update_learning_state(user_input, full_reply)
