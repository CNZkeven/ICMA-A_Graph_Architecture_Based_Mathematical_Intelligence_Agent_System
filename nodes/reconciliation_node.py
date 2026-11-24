"""Reconciliation: decide whether to retry the solving subgraph on mismatch.

The retry decision remains deterministic because intern-s2-preview-397b unreliably
returns structured JSON (it may emit a 'Thinking Process:' preamble):
- python failed → retry, hint python with the error.
- python succeeded but mismatch → retry, hint reasoning to match python's answer.
- round limit reached → route to the evidence-constrained semantic selector.
The selector itself can only choose an existing candidate or abstain.
"""
from config import CONFIG
from utils.deps import get_deps
from utils.reconciliation_policy import (
    projected_solving_seconds,
    reconciliation_retry_available,
    reconciliation_round_limit,
)


def _bounded(value, limit=1000):
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "...[truncated]"


def reconciliation_node(state, config):
    round_num = state.get("reconciliation_round", 0) + 1
    max_rounds = reconciliation_round_limit(config)
    recon_trace = list(state.get("reconciliation_trace") or [])

    po = state.get("python_output") or {}
    rr = state.get("reasoning_result") or {}
    details = state.get("validation_details") or {}
    evidence_summary = (
        details.get("python_evidence_summary")
        or state.get("python_evidence_summary")
        or po.get("evidence_summary")
        or ""
    )
    contradictions = details.get("python_contradictions") or state.get("python_contradictions") \
        or po.get("contradictions") or []
    evidence_context = _bounded(evidence_summary or "；".join(map(str, contradictions)))
    forced = bool(state.get("recheck_required")) and not state.get("forced_recheck_used")

    if round_num >= max_rounds:
        # Circuit breaker: give the semantic selector one final chance to adopt an
        # already-computed complete answer. It may still abstain and fall back to
        # the legacy preference without inventing a new result.
        recon_trace.append({"round": round_num, "action": "semantic_arbiter"})
        result = {
            "reconciliation_round": round_num,
            "reconciliation_trace": recon_trace,
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "semantic_arbiter",
            "validation_status": "reconciliation_exhausted_arbitrating",
        }
        if evidence_summary or contradictions or details.get("unresolved_contradiction"):
            result["validation_history"] = [{
                "round": round_num,
                "status": "unresolved_contradiction",
                "evidence_status": details.get(
                    "python_evidence_status",
                    state.get("python_evidence_status", "inconclusive"),
                ),
                "evidence_summary": _bounded(evidence_context),
                "contradictions": list(contradictions)[:10],
            }]
        return result

    # Reconciliation is normally reached through a policy check in the
    # cross-validator, but graph state can also be resumed or supplied directly.
    # Re-check the affordability here so a stale route cannot start a full pass
    # after the hard-clock reserve has become unavailable.
    if not reconciliation_retry_available(state, config, force=forced):
        try:
            deps = get_deps(config)
        except Exception:  # noqa: BLE001 - preserve the deterministic fallback.
            deps = None
        clock = getattr(deps, "time_budget", None)
        if clock is None:
            remaining_s = None
            required_s = 0.0
        else:
            remaining_fn = getattr(clock, "remaining_hard", None)
            remaining_s = float(
                remaining_fn() if callable(remaining_fn) else clock.remaining()
            )
            required_s = projected_solving_seconds(clock)
            if forced:
                required_s += float(CONFIG.get("forced_recheck_reserve_s", 240))
        blocked = {
            "remaining_s": remaining_s,
            "required_s": round(float(required_s), 2),
            "forced": forced,
        }
        recon_trace.append({
            "round": state.get("reconciliation_round", 0),
            "action": "budget_blocked",
            "budget_blocked": blocked,
        })
        return {
            "reconciliation_round": state.get("reconciliation_round", 0),
            "reconciliation_trace": recon_trace,
            "reasoning_retry_hint": None,
            "python_retry_hint": None,
            "next_node": "semantic_arbiter",
            "validation_status": "reconciliation_budget_blocked",
            "budget_blocked": True,
            "budget_blocked_details": blocked,
            "forced_recheck_used": bool(state.get("forced_recheck_used") or forced),
            "recheck_required": bool(state.get("recheck_required")),
        }

    rs_hint = None
    py_hint = None
    if not po.get("success"):
        err = (po.get("stderr") or "")[:300]
        py_hint = (f"上一次 Python 代码执行失败。错误：{err}。"
                   f"请用 sympy 重新生成正确的 ```python``` 代码，结尾 print(\"最终答案:\", answer)。")
        action = "retry_python"
    elif po.get("answer"):
        # Do not copy either candidate into the other branch.  A fresh pass must
        # rebuild the model from the problem and independently test its result.
        rs_hint = (
            "上一轮候选未形成独立一致的验证闭环（可能存在答案分歧、覆盖不足或证据强度不足）。"
            "请只根据原题重新推理，逐步复核关键计算，不要参考任何外部答案；严格按原格式输出"
            "（含 '## 问题分析'、'## 详细解题步骤'、'## 最终答案' 章节）。"
        )
        py_hint = (
            "上一轮程序验证未能独立支持最终结论。请只根据原题重新建模，检查公式、边界、"
            "状态转移和小规模基准；修正后只输出一个 ```python``` 代码块，结尾"
            "print(\"最终答案:\", answer)，并先打印验证状态: PASS|FAIL|INCONCLUSIVE 和验证证据。"
        )
        action = "retry_both"
    else:
        rs_hint = "请重新推理，给出明确的 '## 最终答案'。"
        py_hint = "请重新生成代码，确保最后一行 print(\"最终答案:\", answer)。"
        action = "retry_both"

    recon_trace.append({"round": round_num, "action": action})
    result = {
        "reconciliation_round": round_num,
        "reconciliation_trace": recon_trace,
        "reasoning_retry_hint": rs_hint,
        "python_retry_hint": py_hint,
        "next_node": "solving",
    }
    if forced:
        result["forced_recheck_used"] = True
        recon_trace[-1]["forced"] = True
    return result
