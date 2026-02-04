"""
教育智能体实现
==============
支持DeepSeek和OpenAI兼容API
"""

from typing import Dict, Any, Generator, Optional
import os

from ..core.base_edu_agent import BaseEduAgent
from ..core.teacher_profile import TeacherProfile


class OpenAIEduAgent(BaseEduAgent):
    """
    基于OpenAI兼容API的教育智能体

    支持：
    - DeepSeek API (默认)
    - OpenAI API
    - 任何兼容OpenAI格式的代理API
    - 流式输出
    """

    def __init__(
        self,
        profile: TeacherProfile,
        system_prompt: str,
        few_shot_examples: str = "",
        llm_config: Dict[str, Any] = None,
        rag_config: Dict[str, Any] = None,
        teaching_dna=None,
        **kwargs
    ):
        super().__init__(profile, system_prompt, few_shot_examples)

        llm_config = llm_config or {}

        # LLM配置 - 优先使用传入参数，其次环境变量
        self.api_key = (
            llm_config.get('api_key') or
            os.getenv('LLM_API_KEY') or
            os.getenv('OPENAI_API_KEY', '')
        )

        # 获取并清理base_url
        raw_base_url = (
            llm_config.get('base_url') or
            os.getenv('LLM_BASE_URL') or
            os.getenv('OPENAI_BASE_URL', 'https://api.deepseek.com')
        )
        self.base_url = self._clean_base_url(raw_base_url)
        self.model = (
            llm_config.get('model') or
            os.getenv('LLM_MODEL') or
            os.getenv('OPENAI_LLM_MODEL', 'deepseek-chat')
        )
        self.temperature = llm_config.get(
            'temperature',
            float(os.getenv('LLM_TEMPERATURE', '0.7'))
        )
        self.max_tokens = llm_config.get(
            'max_tokens',
            int(os.getenv('LLM_MAX_TOKENS', '2000'))
        )

        # RAG配置
        self.rag_config = rag_config or {}
        self.enable_rag = self.rag_config.get('enable', False)

        # 教学DNA（个性化增强）
        self.teaching_dna = teaching_dna

        # 初始化客户端
        self._client = None

    @staticmethod
    def _clean_base_url(url: str) -> str:
        """
        清理base_url，移除多余的路径

        OpenAI客户端会自动添加 /chat/completions
        所以base_url应该只包含基础地址，如：
        - https://api.deepseek.com
        - https://api.deepseek.com/v1
        - https://api.openai.com/v1

        不应该包含：
        - /chat/completions
        - /completions
        """
        if not url:
            return 'https://api.deepseek.com'

        url = url.rstrip('/')

        # 移除常见的多余路径
        suffixes_to_remove = [
            '/chat/completions',
            '/completions',
            '/v1/chat/completions',
            '/v1/completions'
        ]

        for suffix in suffixes_to_remove:
            if url.endswith(suffix):
                url = url[:-len(suffix)]
                break

        # 确保URL不以斜杠结尾
        return url.rstrip('/')

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

    def _build_messages(self, user_input: str, context: str = "") -> list:
        """构建发送给LLM的消息列表（增强版）"""
        messages = [{"role": "system", "content": self.system_prompt}]

        # 注入教学DNA（个性化核心）
        if self.teaching_dna:
            dna_prompt = self.teaching_dna.generate_dna_prompt()
            messages.append({
                "role": "system",
                "content": dna_prompt
            })

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

        # 添加对话历史
        history_to_use = self.history[-self.max_history:]
        for msg in history_to_use:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})

        return messages

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

    async def achat_stream(self, user_input: str, **kwargs):
        """异步流式对话"""
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

        stream = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True
        )

        full_reply = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                yield content

        self.add_to_history("user", user_input)
        self.add_to_history("assistant", full_reply)
        self.update_learning_state(user_input, full_reply)


# 别名，保持向后兼容
DeepSeekEduAgent = OpenAIEduAgent
