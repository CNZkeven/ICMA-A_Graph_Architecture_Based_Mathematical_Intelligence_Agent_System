"""Cross-Validator: compare Reasoning vs Python results, route accordingly. §3.5.5.

M4 hardening: for computation problems, prefer the sympy-computed Python answer
when it succeeded (deterministic > LLM prose, which may echo placeholders or
compute wrong values). For proof problems or python-failed cases, use reasoning.
"""
from utils.answer_matcher import AnswerMatcher
from utils.answer_extractor import looks_incomplete_answer
from utils.cot_stripper import is_placeholder_answer
from config import CONFIG


def _clean_answer(answer: str) -> str:
    return "" if is_placeholder_answer(answer) else (answer or "")


def _preferred_answer(state: dict, match_result: dict) -> str:
    """Pick the answer to surface as validated_answer.

    Computation + python_success + 答案完整 → Python answer (sympy, deterministic).
    Python 答案为空/碎片（评委报告问题 2：'(Matrix([' 等截断片段曾污染
    final_response）→ 回退 reasoning answer；两者都不完整时取非空者兜底。
    """
    rr = state.get("reasoning_result") or {}
    po = state.get("python_output") or {}
    ptype = match_result.get("problem_type", "computation")
    python_answer = _clean_answer(po.get("answer", ""))
    reasoning_answer = _clean_answer(rr.get("answer", ""))
    python_ok = bool(po.get("success")) and bool(python_answer)
    if ptype == "computation" and python_ok and not looks_incomplete_answer(python_answer):
        return python_answer
    if reasoning_answer and not looks_incomplete_answer(reasoning_answer):
        return reasoning_answer
    return reasoning_answer or python_answer


def cross_validator_node(state, config):
    match_result = AnswerMatcher.match_answers(
        problem=state["problem"],
        reasoning_result=state.get("reasoning_result"),
        python_result=state.get("python_output"),
    )
    status = match_result["status"]
    validated_answer = ""
    next_node = "coordinator"

    if status == "match":
        validated_answer = _preferred_answer(state, match_result)
        next_node = "coordinator"
    elif status == "mismatch":
        # subgraph-level retry gated by reconciliation_round (NOT per-node attempts —
        # per-node attempts gate the agent's internal format/code retries; the plan
        # §4.2 keeps these separate).
        if state.get("reconciliation_round", 0) < CONFIG["reconciliation_max_rounds"]:
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            status = "mismatch_forced"
            validated_answer = _preferred_answer(state, match_result)
            next_node = "coordinator"
    else:  # uncertain
        po = state.get("python_output") or {}
        ptype = match_result.get("problem_type", "computation")
        python_failed = not po.get("success")
        can_reconcile = state.get("reconciliation_round", 0) < CONFIG["reconciliation_max_rounds"]
        if ptype == "computation" and python_failed and can_reconcile:
            # python failed on a computation problem → retry the subgraph; python may
            # succeed on re-run (non-determinism) and give a reliable sympy answer.
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            validated_answer = _preferred_answer(state, match_result)
            next_node = "coordinator"

    return {
        "validation_status": status,
        "validation_details": match_result,
        "validated_answer": validated_answer,
        "next_node": next_node,
    }
