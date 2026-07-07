"""Reconciliation: decide whether to retry the solving subgraph on mismatch.

M3 design note: the plan (§3.6) uses an LLM to make the retry decision, but
intern-s2-preview unreliably returns structured JSON (it emits a 'Thinking
Process:' CoT). A data-driven heuristic is more robust here:
- python failed → retry, hint python with the error.
- python succeeded but mismatch → retry, hint reasoning to match python's answer.
- round limit reached → give up, surface the preferred (sympy) answer, go to coordinator.
No LLM call → fast, deterministic, no CoT failure mode.
"""
from config import CONFIG
from nodes.cross_validator_node import _preferred_answer
from utils.deps import get_deps


def _max_rounds(config):
    try:
        deps = get_deps(config)
    except Exception:
        deps = None
    budget = getattr(deps, "token_budget", None)
    if budget and budget.is_tight():
        return 1
    return CONFIG["reconciliation_max_rounds"]


def reconciliation_node(state, config):
    round_num = state.get("reconciliation_round", 0) + 1
    max_rounds = _max_rounds(config)
    recon_trace = list(state.get("reconciliation_trace") or [])

    rr = state.get("reasoning_result") or {}
    po = state.get("python_output") or {}

    if round_num >= max_rounds:
        # circuit breaker — give up, surface best answer, go to coordinator
        match_result = state.get("validation_details") or {}
        validated = _preferred_answer(state, match_result)
        recon_trace.append({"round": round_num, "action": "give_up",
                            "validated_answer": validated})
        return {
            "reconciliation_round": round_num,
            "reconciliation_trace": recon_trace,
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "coordinator",
            "validated_answer": validated,
            "validation_status": "reconciled",
        }

    rs_hint = None
    py_hint = None
    if not po.get("success"):
        err = (po.get("stderr") or "")[:300]
        py_hint = (f"上一次 Python 代码执行失败。错误：{err}。"
                   f"请用 sympy 重新生成正确的 ```python``` 代码，结尾 print(\"最终答案:\", answer)。")
        action = "retry_python"
    elif po.get("answer"):
        # 不把 Python 答案灌给推理侧（Python 可能因 API 误用而错，如 jordan_form 解包
        # 反序——若锚定会把正确推理带偏）；两侧独立复核。
        py_ans = po.get("answer")
        rs_hint = ("你之前的推理答案与独立程序验证结果不一致。请重新独立推理，逐步复核每一步"
                   "关键计算，不要参考任何外部答案；严格按原格式输出（含 '## 问题分析'、"
                   "'## 详细解题步骤'、'## 最终答案' 章节），'## 最终答案' 后给出复核后的明确结果。")
        py_hint = (f"上一次代码计算结果为：{py_ans}，与理论推导不一致，代码逻辑可能有误。"
                   "请重点检查：多返回值函数的解包顺序（如 sympy 的 A.jordan_form() 返回 (P, J)，"
                   "Jordan 标准形是第二个返回值；A.diagonalize() 返回 (P, D)）、公式实现是否与题意"
                   "一致。修正后只输出一个 ```python``` 代码块，结尾 print(\"最终答案:\", answer)。")
        action = "retry_both"
    else:
        rs_hint = "请重新推理，给出明确的 '## 最终答案'。"
        py_hint = "请重新生成代码，确保最后一行 print(\"最终答案:\", answer)。"
        action = "retry_both"

    recon_trace.append({"round": round_num, "action": action})
    return {
        "reconciliation_round": round_num,
        "reconciliation_trace": recon_trace,
        "reasoning_retry_hint": rs_hint,
        "python_retry_hint": py_hint,
        "next_node": "solving",
    }
