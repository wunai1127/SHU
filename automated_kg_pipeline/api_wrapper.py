#!/usr/bin/env python3
"""
DeepSeek API调用包装器 - 带限流保护和自动重试
"""

import time
import openai
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class RateLimitProtectedAPI:
    """带限流保护的API调用器"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

        # 配置openai客户端
        openai.api_key = api_key
        openai.api_base = base_url

        # 限流参数（根据DeepSeek文档）
        self.max_requests_per_minute = 60  # DeepSeek限制
        self.request_interval = 60.0 / self.max_requests_per_minute  # 1秒/请求
        self.last_request_time = 0

        # 重试参数
        self.max_retries = 5
        self.retry_delays = [2, 5, 10, 30, 60]  # 指数退避

    def call_with_retry(self, messages: List[Dict], max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """带重试机制的API调用"""

        for attempt in range(self.max_retries):
            try:
                # 限流控制：确保两次请求间隔>=1秒
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.request_interval:
                    sleep_time = self.request_interval - time_since_last
                    logger.debug(f"限流保护：等待 {sleep_time:.2f} 秒")
                    time.sleep(sleep_time)

                # 调用API
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=30  # 30秒超时
                )

                self.last_request_time = time.time()
                return response.choices[0].message.content

            except openai.error.RateLimitError as e:
                # 触发限流 - 等待后重试
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt]
                    logger.warning(f"触发限流，第 {attempt+1} 次重试，等待 {delay} 秒: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"达到最大重试次数，限流错误: {e}")
                    raise

            except openai.error.Timeout as e:
                # 超时 - 重试
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt]
                    logger.warning(f"请求超时，第 {attempt+1} 次重试，等待 {delay} 秒: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"达到最大重试次数，超时错误: {e}")
                    raise

            except openai.error.APIError as e:
                # API错误（可能是欠费）- 检查是否可恢复
                error_message = str(e)
                if "insufficient" in error_message.lower() or "balance" in error_message.lower():
                    logger.error(f"余额不足！请充值后重新运行。错误: {e}")
                    logger.info("提示：系统支持断点续传，充值后可从上次中断处继续")
                    raise  # 不重试，让用户充值
                else:
                    # 其他API错误，重试
                    if attempt < self.max_retries - 1:
                        delay = self.retry_delays[attempt]
                        logger.warning(f"API错误，第 {attempt+1} 次重试，等待 {delay} 秒: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"达到最大重试次数，API错误: {e}")
                        raise

            except Exception as e:
                # 未知错误
                logger.error(f"未知错误: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delays[attempt]
                    logger.warning(f"第 {attempt+1} 次重试，等待 {delay} 秒")
                    time.sleep(delay)
                else:
                    raise

        raise Exception("API调用失败：达到最大重试次数")


# 使用示例
if __name__ == '__main__':
    api = RateLimitProtectedAPI(
        api_key="sk-kzBbW2kbljrrOmp6EAgXcQR1F4cxhTMaCJmfyzZeIY8m1fPu",
        base_url="https://yinli.one/v1",
        model="deepseek-chat"
    )

    # 测试调用
    messages = [{"role": "user", "content": "测试连接"}]
    response = api.call_with_retry(messages, max_tokens=10)
    print(f"API响应: {response}")
