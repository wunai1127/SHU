"""
配置管理模块
============
统一管理所有配置项，支持环境变量和默认值
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv是可选的


@dataclass
class LLMConfig:
    """LLM配置"""
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv('LLM_API_KEY', ''),
            base_url=os.getenv('LLM_BASE_URL', 'https://api.deepseek.com/v1'),
            model=os.getenv('LLM_MODEL', 'deepseek-chat'),
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.7')),
            max_tokens=int(os.getenv('LLM_MAX_TOKENS', '2000'))
        )


@dataclass
class EmbeddingConfig:
    """Embedding配置"""
    api_key: str
    base_url: str
    model: str

    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """从环境变量加载配置"""
        return cls(
            api_key=os.getenv('EMBEDDING_API_KEY', os.getenv('LLM_API_KEY', '')),
            base_url=os.getenv('EMBEDDING_BASE_URL', os.getenv('LLM_BASE_URL', '')),
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
        )


@dataclass
class WebConfig:
    """Web服务配置"""
    port: int = 8001
    host: str = "0.0.0.0"

    @classmethod
    def from_env(cls) -> 'WebConfig':
        return cls(
            port=int(os.getenv('WEB_PORT', '8001')),
            host=os.getenv('WEB_HOST', '0.0.0.0')
        )


class Settings:
    """
    全局配置单例

    使用方式:
        from config import settings
        print(settings.llm.model)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        self.llm = LLMConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.web = WebConfig.from_env()
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def reload(self):
        """重新加载配置"""
        self._init_config()

    def validate(self) -> bool:
        """验证必要配置是否存在"""
        if not self.llm.api_key:
            print("警告: LLM_API_KEY 未设置")
            return False
        return True


# 全局配置实例
settings = Settings()


def get_llm_client():
    """
    获取配置好的LLM客户端

    Returns:
        OpenAI兼容的客户端实例
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("请安装openai: pip install openai")

    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url
    )


def get_async_llm_client():
    """获取异步LLM客户端"""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("请安装openai: pip install openai")

    return AsyncOpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url
    )
