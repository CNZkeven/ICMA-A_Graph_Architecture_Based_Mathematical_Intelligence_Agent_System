"""Conversation-aware routing for follow-up questions."""

from __future__ import annotations

from typing import Any

from . import storage


def resolve_request(conversation_id: str, message: str) -> dict[str, Any]:
    conversation = storage.get_conversation(conversation_id) or {"messages": []}
    messages = conversation.get("messages", [])
    previous_user = next((item for item in reversed(messages) if item["role"] == "user"), None)
    previous_answer = next((item for item in reversed(messages) if item["role"] == "assistant"), None)
    if previous_answer and any(token in message for token in ("解释上一步", "解释这一步", "解释步骤", "解释")):
        return {
            "path": "follow_up",
            "resolved_problem": (
                f"请结合上一轮对话回答用户的追问：{message}\n\n"
                f"上一轮题目：{(previous_user or {}).get('content', '')}\n"
                f"上一轮答案：{previous_answer.get('content', '')}"
            ),
            "source_run_id": previous_answer.get("run_id"),
        }
    if previous_user and any(token in message for token in ("换一种方法", "另一种方法", "改一种方法")):
        return {
            "path": "follow_up",
            "resolved_problem": (
                f"原题：{previous_user.get('content', '')}\n"
                f"请针对原题回应用户的追问：{message}"
            ),
            "source_run_id": previous_user.get("run_id"),
        }
    if previous_user and any(token in message for token in ("再算", "改成", "如果", "参数")):
        return {
            "path": "follow_up",
            "resolved_problem": (
                f"原题：{previous_user.get('content', '')}\n"
                f"请保留原题上下文并处理新的条件：{message}"
            ),
            "source_run_id": previous_user.get("run_id"),
        }
    return {"path": "solve", "resolved_problem": message, "source_run_id": None}
