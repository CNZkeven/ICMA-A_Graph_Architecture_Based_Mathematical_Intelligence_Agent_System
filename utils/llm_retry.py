import logging
import time
from typing import Dict, List

from config import CONFIG


class LLMRetryWrapper:
    def __init__(self, client, max_retries: int = 3, backoff_factor: float = 2.0, logger=None):
        self.client = client
        self.max_retries = max(1, max_retries)
        self.backoff_factor = backoff_factor
        self.logger = logger or logging.getLogger("math_agent")

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        max_tokens: int = 4096,
    ) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - retry transient client failures.
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = 0 if self.backoff_factor == 0 else self.backoff_factor ** (attempt - 1)
                self.logger.warning("LLM call failed on attempt %s/%s: %s", attempt, self.max_retries, exc)
                if delay > 0:
                    time.sleep(delay)
        raise last_error


def chat_with_retry(client, messages, temperature=0.2, max_tokens=4096, logger=None):
    return LLMRetryWrapper(
        client,
        max_retries=CONFIG["llm_max_retries"],
        backoff_factor=CONFIG["backoff_factor"],
        logger=logger,
    ).chat(messages=messages, temperature=temperature, max_tokens=max_tokens)
