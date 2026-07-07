import logging
import traceback
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig


def _node_name(node_func: Callable, explicit_name: str | None = None) -> str:
    if explicit_name:
        return explicit_name
    name = getattr(node_func, "__name__", "unknown")
    return name[:-5] if name.endswith("_node") else name


def _error_payload(node: str, exc: Exception) -> dict:
    return {
        "node": node,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=5),
    }


def _fallback_for_node(node: str, state: dict, error: dict) -> dict:
    base = {"errors": [error]}
    if node == "classifier":
        return {
            **base,
            "category": "非基础及进阶课程",
            "category_confidence": 0.0,
            "candidate_categories": [],
            "classification_stages_used": ["error_fallback"],
        }
    if node == "reasoning_agent":
        return {
            **base,
            "reasoning_result": {
                "analysis": "",
                "steps": [],
                "answer": "",
                "validation_points": [],
            },
            "reasoning_trace": [{"attempt": 0, "status": "failed", "error": error["message"]}],
            "reasoning_attempts": 0,
        }
    if node == "python_agent":
        return {
            **base,
            "python_code": "",
            "python_output": {
                "success": False,
                "stdout": "",
                "stderr": error["message"],
                "answer": None,
                "execution_time": 0.0,
            },
            "python_trace": [{"attempt": 0, "status": "failed", "error": error["message"]}],
            "python_attempts": 0,
        }
    if node == "cross_validator":
        return {
            **base,
            "validation_status": "uncertain",
            "validation_details": {"status": "uncertain", "reason": error["message"]},
            "validated_answer": "",
            "next_node": "coordinator",
        }
    if node == "reconciliation":
        return {
            **base,
            "reconciliation_trace": list(state.get("reconciliation_trace") or [])
            + [{"round": state.get("reconciliation_round", 0), "action": "error_fallback"}],
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "coordinator",
        }
    if node == "coordinator":
        rr = state.get("reasoning_result") or {}
        answer = state.get("validated_answer") or rr.get("answer", "")
        if not answer:
            final = "解题过程中出现错误，无法给出完整答案。"
        else:
            ptype = (state.get("validation_details") or {}).get("problem_type", "computation")
            final = f"结论：{answer}" if ptype == "proof" else f"最终答案：{answer}"
        return {**base, "final_response": final}
    return {**base, "should_terminate": True, "next_node": "coordinator"}


class ErrorHandler:
    @staticmethod
    def safe_execute(func: Callable, *args: Any, **kwargs: Any) -> tuple[bool, Any]:
        try:
            return True, func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - convert any node failure into state.
            return False, {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }

    @staticmethod
    def node_wrapper(node_func: Callable, node_name: str | None = None) -> Callable:
        name = _node_name(node_func, node_name)
        logger = logging.getLogger("math_agent")

        def wrapped(state: dict, config: RunnableConfig) -> dict:
            try:
                return node_func(state, config)
            except Exception as exc:  # noqa: BLE001 - graph must continue per M4.
                logger.exception("Node %s failed", name)
                return _fallback_for_node(name, state, _error_payload(name, exc))

        wrapped.__name__ = f"{name}_wrapped"
        return wrapped


def node_wrapper(node_func: Callable, node_name: str | None = None) -> Callable:
    return ErrorHandler.node_wrapper(node_func, node_name=node_name)
