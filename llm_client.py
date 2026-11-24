import json
import os
import time
from typing import Callable, Dict, Iterator, List

import requests

from config import CONFIG


DEFAULT_API_BASE = "https://chat.intern-ai.org.cn/api/v1/chat/completions"
DEFAULT_MODEL = CONFIG["model"]


class InternChatClient:
    """Small OpenAI-compatible chat client for the competition sample."""

    def __init__(
        self,
        timeout: int = 120,
        retry: int = 3,
        api_key: str | None = None,
    ) -> None:
        raw_api_key = api_key or os.environ.get("INTERN_API_KEY")
        if not raw_api_key:
            raise RuntimeError("Missing API key. Set INTERN_API_KEY.")
        self.authorization = (
            raw_api_key if raw_api_key.startswith("Bearer ") else f"Bearer {raw_api_key}"
        )
        api_base = (os.environ.get("INTERN_API_BASE") or DEFAULT_API_BASE).strip().rstrip("/")
        self.api_base = api_base if api_base.endswith("/chat/completions") else f"{api_base}/chat/completions"
        self.model = os.environ.get("INTERN_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.retry = retry

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }

        last_error = None
        for attempt in range(self.retry):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - keep sample robust and simple.
                last_error = exc
                if attempt + 1 < self.retry:
                    time.sleep(2**attempt)

        raise RuntimeError(f"Chat completion failed after {self.retry} attempts: {last_error}")

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
        *,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        """Yield visible answer deltas without replaying a partially emitted answer."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "thinking_mode": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.authorization,
        }
        emitted = False
        last_error = None
        connect_timeout = min(10, self.timeout)
        for attempt in range(self.retry):
            try:
                with requests.post(
                    self.api_base,
                    headers=headers,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    timeout=(connect_timeout, self.timeout),
                    stream=True,
                ) as response:
                    response.raise_for_status()
                    finished = False
                    # requests' larger default buffer can hold several small SSE
                    # events before yielding the first answer to the browser.
                    for raw_line in response.iter_lines(chunk_size=1):
                        line = raw_line.decode("utf-8").strip()
                        if not line or line.startswith((":", "event:", "id:", "retry:")):
                            continue
                        if not line.startswith("data:"):
                            raise RuntimeError("模型没有返回有效的流式响应。")
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        event = json.loads(data)
                        if event.get("error") or event.get("object") == "error":
                            error = event.get("error") or event
                            detail = error.get("message", "模型请求失败") if isinstance(error, dict) else str(error)
                            raise RuntimeError(detail)
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        reasoning = delta.get("reasoning_content")
                        if isinstance(reasoning, str) and reasoning and on_reasoning is not None:
                            emitted = True
                            on_reasoning(reasoning)
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            emitted = True
                            yield content
                        reason = choice.get("finish_reason")
                        if reason == "length":
                            raise RuntimeError("回答达到模型输出上限，请缩短问题后重试。")
                        if reason is not None:
                            finished = True
                    if finished:
                        return
                    raise RuntimeError("模型输出流提前断开，请重试。")
            except Exception as exc:  # noqa: BLE001 - never duplicate visible deltas on retry.
                last_error = exc
                retryable = isinstance(exc, (requests.ConnectionError, requests.Timeout))
                if emitted or not retryable or attempt + 1 >= self.retry:
                    break
                time.sleep(2**attempt)
        if isinstance(last_error, requests.ConnectTimeout):
            detail = f"连接模型服务超时（单次建连上限 {connect_timeout:g} 秒），请检查网络或代理后重试。"
        elif isinstance(last_error, requests.ReadTimeout):
            detail = "等待模型输出超时，请稍后重试。"
        elif isinstance(last_error, requests.ConnectionError):
            detail = "与模型服务的连接失败或中断，请检查网络、代理或 HTTPS 证书设置后重试。"
        else:
            detail = str(last_error)
        raise RuntimeError(f"流式回答失败：{detail}") from last_error
