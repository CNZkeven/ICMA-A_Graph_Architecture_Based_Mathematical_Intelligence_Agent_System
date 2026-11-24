import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig

from utils.deps import get_deps
from utils.problem_anchor import verify_problem_anchor


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


def _clip(value: Any, limit: int = 1800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[truncated]"


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Keep event payloads serialisable even when a node returns a custom value."""
    if depth > 5:
        return _clip(value, 600)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth + 1) for item in value]
    return _clip(value, 1000)


def _stage_trace_payload(node: str, state: dict, result: dict, deps: Any) -> dict:
    """Expose the useful part of a node result in its completion event.

    The final trace is still assembled by ``ReasoningAgent``.  These smaller
    payloads make the same evidence available while a run is in progress,
    without serialising the full LangGraph state or raw model responses.
    """
    thinking: dict[str, Any] = {}
    tools: list[dict[str, Any]] = []
    category = result.get("category") or state.get("category")
    if node == "input":
        thinking = {
            "purpose": "固定原始题目并识别题型模式。",
            "question_mode": result.get("question_mode") or state.get("question_mode", ""),
            "status": "题目已接收",
        }
    elif node == "classifier":
        base_path = str(getattr(getattr(deps, "skills_loader", None), "base_path", "skills_pythonscripts"))
        category_dir = Path(base_path) / str(category or "")
        category_path = str(category_dir)
        skill_document = str(category_dir / f"{category or ''}skill.md")
        validation_script = str(category_dir / f"{category or ''}验证示例.py")
        thinking = {
            "category": category or "",
            "category_path": category_path,
            "skill_document": skill_document,
            "validation_script": validation_script,
            "question_mode": result.get("question_mode") or state.get("question_mode", ""),
            "confidence": result.get("category_confidence", 0.0),
            "classification_stages": result.get("classification_stages_used", []),
            "classification_basis": "从候选领域中选择与题面信号最匹配的技能目录。",
        }
        tools = [{
            "name": "category_candidates",
            "candidates": result.get("candidate_categories", []),
            "selected": category or "",
            "selected_path": category_path,
            "skill_document": skill_document,
            "validation_script": validation_script,
        }]
    elif node == "database_retrieval":
        examples = result.get("retrieved_examples") or []
        references = [
            {
                "rank": index,
                "similarity": item.get("similarity", 0.0),
                "source": item.get("source", ""),
                "problem": _clip(item.get("problem", ""), 900),
                "solution": _clip(item.get("solution", ""), 1200),
            }
            for index, item in enumerate(examples, 1)
            if isinstance(item, dict)
        ]
        thinking = {
            "purpose": "寻找相似题的解法结构，仅作为参考，不直接复制结论。",
            "query": state.get("problem", ""),
            "reference_count": len(references),
            "references": references,
        }
        tools = [{"name": "vector_database_query", "top_k": len(references), "results": references}]
    elif node == "reasoning_agent":
        reasoning = result.get("reasoning_result") or {}
        thinking = {
            "analysis": _clip(reasoning.get("analysis", ""), 2200),
            "steps": reasoning.get("steps", []),
            "result": reasoning.get("answer", ""),
            "final_answer": reasoning.get("answer", ""),
            "validation_points": reasoning.get("validation_points", []),
        }
        tools = [{
            "name": "reasoning_llm",
            "attempts": result.get("reasoning_trace", []),
            "max_attempts_used": result.get("reasoning_attempts", 0),
            "output_schema": ["analysis", "steps", "answer", "validation_points"],
        }]
    elif node == "python_agent":
        output = result.get("python_output") or {}
        code = _clip(result.get("python_code", ""), 5000)
        thinking = {
            "purpose": "用生成的 Python/SymPy 代码独立复核答案。",
            "python_code": code,
            "stdout": _clip(output.get("stdout", ""), 2200),
            "stderr": _clip(output.get("stderr", ""), 1400),
            "result": output.get("answer", ""),
            "verification_trusted": output.get("python_verification_trusted", True),
            "evidence_status": output.get("evidence_status", result.get("python_evidence_status", "")),
            "evidence_summary": _clip(output.get("evidence_summary", result.get("python_evidence_summary", "")), 1200),
        }
        tools = [{
            "name": "python_executor",
            "backend": output.get("execution_backend", ""),
            "success": output.get("success", False),
            "code": code,
            "stdout": _clip(output.get("stdout", ""), 1400),
            "stderr": _clip(output.get("stderr", ""), 1000),
            "result": output.get("answer", ""),
        }]
    elif node == "cross_validator":
        details = result.get("validation_details") or {}
        thinking = {
            "comparison": details.get("comparison", details.get("method", "")),
            "decision": result.get("validation_status", ""),
            "result": result.get("validated_answer", ""),
            "evidence_status": result.get("python_evidence_status", ""),
            "evidence_summary": _clip(result.get("python_evidence_summary", ""), 1200),
            "contradictions": result.get("python_contradictions", []),
        }
        tools = [{"name": "answer_matcher", "details": details}]
    elif node == "objective_review":
        thinking = {key: value for key, value in result.items() if key not in {"errors", "node_timings"}}
        tools = [{"name": "objective_review", "result": thinking}]
    elif node == "reconciliation":
        thinking = {
            "decision": result.get("validation_status", result.get("next_node", "")),
            "round": result.get("reconciliation_round", 0),
            "contradictions": result.get("python_contradictions", []),
        }
        tools = [{"name": "reconciliation_policy", "rounds": result.get("reconciliation_trace", [])}]
    elif node == "semantic_arbiter":
        thinking = {
            "decision": result.get("semantic_arbiter_decision", ""),
            "status": result.get("semantic_arbiter_status", ""),
            "result": result.get("validated_answer", ""),
        }
        tools = [{"name": "semantic_arbiter_llm", "rounds": result.get("semantic_arbiter_trace", [])}]
    elif node == "coordinator":
        thinking = {
            "purpose": "将推理步骤、验证结果与最终答案整理为面向读者的输出。",
            "result": _clip(result.get("final_response", ""), 2200),
        }
        tools = [{"name": "answer_formatter", "final_response_length": len(str(result.get("final_response", "")))}]
    return _json_safe({"thinking": thinking, "tools": tools}) if thinking or tools else {}


def _fallback_for_node(node: str, state: dict, error: dict, config: dict | None = None) -> dict:
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
    if node == "objective_review":
        # The review is an optional second candidate, not a terminal branch.
        # Keep the solving subgraph's objective_review -> cross_validator edge
        # alive when transport/parsing fails so the first candidate can still be
        # assessed and routed through the normal bounded fallback.
        return {
            **base,
            "objective_review_result": {
                "analysis": "",
                "steps": [],
                "answer": "",
                "validation_points": [],
            },
            "objective_review_trace": [
                {"attempt": 0, "status": "failed", "error": error["message"]}
            ],
            "objective_review_attempts": 0,
        }
    if node == "python_agent":
        from utils.verification_evidence import parse_verification_evidence

        output = parse_verification_evidence(
            {
                "success": False,
                "stdout": "",
                "stderr": error["message"],
                "answer": None,
                "execution_time": 0.0,
            },
            candidate_answer=(state.get("reasoning_result") or {}).get("answer", ""),
            problem=state.get("problem", ""),
        )
        return {
            **base,
            "python_code": "",
            "python_output": output,
            "python_trace": [{"attempt": 0, "status": "failed", "error": error["message"]}],
            "python_attempts": 0,
            "python_evidence_status": output["evidence_status"],
            "python_evidence_summary": output["evidence_summary"],
            "python_contradictions": output["contradictions"],
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
    if node == "semantic_arbiter":
        from nodes.cross_validator_node import _preferred_answer
        from utils.reconciliation_policy import reconciliation_retry_available

        can_retry = False
        try:
            can_retry = bool(config) and reconciliation_retry_available(state, config)
        except Exception:
            can_retry = False
        answer = "" if can_retry else _preferred_answer(
            state, state.get("validation_details") or {})
        return {
            **base,
            "semantic_arbiter_status": "error",
            "semantic_arbiter_trace": list(state.get("semantic_arbiter_trace") or [])
            + [{"status": "error", "error": error["message"]}],
            "semantic_arbiter_attempts": state.get("semantic_arbiter_attempts", 0) + 1,
            "validated_answer": answer,
            "answer_locked": False,
            "next_node": "reconciliation" if can_retry else "coordinator",
        }
    if node == "coordinator":
        rr = state.get("reasoning_result") or {}
        answer = state.get("validated_answer") or rr.get("answer", "")
        if not answer:
            # Fall back to raw reasoning text before giving up entirely.
            raw = (state.get("reasoning_raw_response") or "").strip()
            if raw:
                final = f"未能完成完整推导，以下为已获得的部分结果：\n{raw[-2000:]}"
            else:
                final = "解题过程中出现错误，无法给出完整答案。"
        else:
            ptype = (state.get("validation_details") or {}).get("problem_type", "computation")
            # 2026-08-24：响应只放答案本身，不附加标签（步骤在 trace）。
            final = answer
        return {
            **base,
            "final_response": final,
            "fallback_source": "validated_answer" if answer else (
                "reasoning_raw" if state.get("reasoning_raw_response") else "generic_error"),
        }
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
            from utils.timeout_control import resolve_timeout, run_with_timeout

            time_budget = None
            try:
                time_budget = getattr(get_deps(config), "time_budget", None)
            except Exception:  # noqa: BLE001 - a missing deps shape must not break routing.
                time_budget = None

            check = verify_problem_anchor(state)
            guarded_state = state
            integrity_events = []
            if check.event:
                integrity_events.append(check.event)
                if check.event.get("action") in {"repair", "anchor_corrupt"}:
                    guarded_state = {**state, "problem": check.problem}

            started = time.monotonic()
            event_sink = None
            try:
                event_sink = getattr(get_deps(config), "event_sink", None)
            except Exception:
                event_sink = None
            if event_sink is not None:
                try:
                    event_sink.emit("node.started", node=name, payload={
                        "summary": f"开始执行{name}",
                    })
                except Exception:
                    pass
            degraded = False
            try:
                timeout = resolve_timeout(name, time_budget)
                result = run_with_timeout(lambda: node_func(guarded_state, config), timeout)
            except Exception as exc:  # noqa: BLE001 - graph must continue per M4.
                logger.exception("Node %s failed", name)
                result = _fallback_for_node(name, guarded_state, _error_payload(name, exc), config=config)
                degraded = True
            elapsed = time.monotonic() - started
            if time_budget is not None:
                time_budget.record(f"node:{name}", elapsed)
            # Stage timings were unobservable before (judge report 4.7 had to
            # estimate them); recording them makes the next timeout diagnosable.
            if isinstance(result, dict):
                if integrity_events:
                    result.setdefault("problem_integrity_events", []).extend(integrity_events)
                if guarded_state is not state:
                    result.setdefault("problem", check.problem)
                result.setdefault("node_timings", []).append(
                    {"node": name, "elapsed_s": round(elapsed, 2)}
                )
            if event_sink is not None:
                try:
                    stage_payload = _stage_trace_payload(name, guarded_state, result, get_deps(config)) if isinstance(result, dict) else {}
                    event_sink.emit(
                        "node.degraded" if degraded else "node.completed",
                        node=name,
                        payload={
                            "summary": (
                                f"{name} 已降级处理" if degraded else f"{name} 完成"
                            ),
                            "elapsed_s": round(elapsed, 2),
                            **stage_payload,
                        },
                    )
                except Exception:
                    pass
            return result

        wrapped.__name__ = f"{name}_wrapped"
        return wrapped


def node_wrapper(node_func: Callable, node_name: str | None = None) -> Callable:
    return ErrorHandler.node_wrapper(node_func, node_name=node_name)
