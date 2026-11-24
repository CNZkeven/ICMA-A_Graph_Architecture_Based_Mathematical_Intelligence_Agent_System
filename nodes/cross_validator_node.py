"""Cross-Validator: compare Reasoning vs Python results, route accordingly. §3.5.5.

M4 hardening: for computation problems, prefer the sympy-computed Python answer
when it succeeded (deterministic > LLM prose, which may echo placeholders or
compute wrong values). For proof problems or python-failed cases, use reasoning.
"""
import re
from utils.answer_matcher import AnswerMatcher
from utils.answer_contract import answer_part_count, missing_components
from utils.answer_extractor import is_multi_part_problem, looks_incomplete_answer
from utils.candidate_health import assess_candidate_health
from utils.cot_stripper import is_placeholder_answer
from utils.reconciliation_policy import reconciliation_retry_available
from utils.solution_cards import matched_cards
from utils.verification_evidence import (
    parse_verification_evidence,
    python_answer_is_trusted,
)
from utils.problem_profile import (
    classify_question_mode,
    fill_answer_matches_blanks,
    is_objective_mode,
    normalize_objective_answer,
    objective_answer_consistency,
    objective_answer_is_consistent,
    objective_answer_is_usable,
)


def _clean_answer(answer: str) -> str:
    return "" if is_placeholder_answer(answer) else (answer or "")


def _python_trusted(python_output: dict) -> bool:
    """Python 分支是否真的验证过——判断集中在 verification_evidence 里。

    保留这个名字是因为它已是本模块的既有词汇；实现委托给唯一权威入口，避免
    各消费点再各写一份等价判断而漏掉其中之一。
    """
    return python_answer_is_trusted(python_output)


def _same_scalar_or_text(a: str, b: str) -> bool:
    """两个候选是否本质相同（去空白/标点后一致）。"""
    import re as _re
    ka = _re.sub(r"[\s。．.,，;；]+", "", str(a or ""))
    kb = _re.sub(r"[\s。．.,，;；]+", "", str(b or ""))
    return bool(ka) and ka == kb


def _objective_key(answer: str, mode: str) -> str:
    """Canonical comparison key for two short objective candidates."""
    normalized = normalize_objective_answer(answer, mode)
    if mode == "choice":
        return ",".join(sorted(part for part in normalized.split("、") if part))
    if mode == "fill":
        return re.sub(r"\s+", "", normalized)
    return normalized


def _preferred_answer(state: dict, match_result: dict) -> str:
    """Pick the answer to surface as validated_answer.

    证据优先序（2026-08-09 评委建议 1，最高优先级）：
    ``有枚举支撑且反驳对手的 Python 答案 > evidence=support 的分支 > 推理答案 >
    结构化部分结论``。idx 20 事故——Python 枚举出 C(2k,k)² 且 evidence=contradict
    明确反驳了推理的 "4"，仲裁超时后系统仍输出被自家验证器反驳的 "4"——由第一
    条规则在架构上杜绝：证据指向谁，兜底就选谁。

    Computation + python_success + 答案完整 → Python answer (sympy, deterministic).
    Python 答案为空/碎片（评委报告问题 2：'(Matrix([' 等截断片段曾污染
    final_response）→ 回退 reasoning answer；两者都不完整时取非空者兜底。

    uncertain 时保持推理优先（去锚定），但 2026-07-07 评委报告显示：推理侧漏
    结论/漏问项/漏分配而 Python 侧完整时（265/271/283/93/113/275），残缺答案
    仍然胜出。故当推理答案缺契约字段或漏问项、且 Python 答案更完整时改选 Python。
    """
    rr = state.get("reasoning_result") or {}
    po = state.get("python_output") or {}
    problem = state.get("problem", "")
    ptype = match_result.get("problem_type", "computation")
    question_mode = state.get("question_mode") or ptype
    python_answer = _clean_answer(po.get("answer", ""))
    reasoning_answer = _clean_answer(rr.get("answer", ""))
    if question_mode == "fill":
        # A partial multi-blank payload is not a candidate answer.  Keep the
        # evidence in state for diagnostics, but never let it win a preference
        # fallback or an arbitration decision.
        if not fill_answer_matches_blanks(reasoning_answer, problem):
            reasoning_answer = ""
        if not fill_answer_matches_blanks(python_answer, problem):
            python_answer = ""
        if reasoning_answer and not objective_answer_is_consistent(
            reasoning_answer, problem, question_mode
        ):
            reasoning_answer = ""
        if python_answer and not objective_answer_is_consistent(
            python_answer, problem, question_mode
        ):
            python_answer = ""
    # 未验证即不可信（2026-08-19 idx=0）：Python 分支没做实质计算、或把结论抄进
    # print 时，它不是一个"验证过的答案"，不得参与答案选择——无论推理侧是否可用。
    # 该分支仍保留在 trace 与证据字段里，只是失去出厂资格。
    if not _python_trusted(po):
        python_answer = ""
    # 但"未通过验证强度检查"（缺独立基准比对）不等于"没算过"：只要它的计算
    # 结果与推理候选形成真实冲突，它就仍是仲裁/兜底要裁决的一方——2026-08-31
    # Q9 实测：Python 算出正确值 16，因缺基准被 trusted=False 清空，矛盾证据
    # 随之失效，错答出厂。伪造（fabricated）与未达标（unverified）在这里分开。
    python_raw = _clean_answer(po.get("answer", ""))
    python_ok = bool(po.get("success")) and bool(python_answer)
    reasoning_ok = bool(reasoning_answer) and not looks_incomplete_answer(reasoning_answer)
    if is_objective_mode(question_mode):
        # Keep every degradation path canonical.  In particular, an invalid or
        # unavailable objective candidate may still reach the semantic arbiter's
        # deterministic fallback; returning the raw rationale there would undo the
        # short-path formatting guarantee in coordinator_node.
        for candidate in (reasoning_answer, python_answer):
            normalized = normalize_objective_answer(candidate, question_mode)
            if objective_answer_is_usable(normalized, question_mode):
                return normalized
        return ""
    # 证据优先的最后防线：Python 证据明确反驳推理候选、且 Python 自己持有一个
    # 不同的、可用的、非伪造的答案时，兜底必须选 Python——被自家验证器反驳的
    # 答案不得再被输出。仅在此条件下打破"推理优先"，避免污染无争议路径。
    evidence_status = po.get("evidence_status", "")
    fabricated = bool((po.get("authenticity") or {}).get("fabricated"))
    # 例外（2026-09-02）：命中"解法直达"参考卡的题型不走这条否决。卡片是该族已核定
    # 的通解与取值（可信文档），而与之矛盾的 Python 候选往往是按**另一种口径**自行
    # 枚举的产物——实测 Q59：Python 把"4 轮后仍有人"实现成 q≥4 求和，得 2.09e13，
    # 与官方口径的单个欧拉数相矛盾，却凭"执行证据"在兜底里压过带卡片的推理答案。
    # 这类冲突只能由看过口径参考的语义仲裁裁决，兜底不得机械偏袒执行侧。
    if evidence_status == "contradict" and python_raw and not fabricated \
            and not looks_incomplete_answer(python_raw) \
            and not _same_scalar_or_text(python_raw, reasoning_answer) \
            and not matched_cards(problem):
        return python_raw
    if match_result.get("status") == "match" and reasoning_ok:
        # Python/SymPy supplies correctness evidence; retain the already validated
        # human-readable reasoning form for the public answer.
        return reasoning_answer
    if match_result.get("status") == "uncertain" and reasoning_ok:
        # uncertain + "谁缺的契约字段少就选谁"这条启发式必须先看真伪：2026-09-02 全量
        # 评测 idx 22 的推理已经算出 25502500，却因为答案里没写题目措辞的字段名而被
        # missing_components 判成"更残缺"，把一个被反伪造标记为字面量直答的
        # -1524914160 送出了厂。伪证据不参与完整度比较。
        if python_ok and not fabricated and not looks_incomplete_answer(python_answer):
            r_missing = missing_components(problem, reasoning_answer)
            p_missing = missing_components(problem, python_answer)
            if len(p_missing) < len(r_missing):
                return python_answer
            if is_multi_part_problem(problem) \
                    and answer_part_count(python_answer) > answer_part_count(reasoning_answer):
                return python_answer
        return reasoning_answer
    # 命中参考卡的题型同样跳过这条"计算题偏袒 Python"：手册里写的是该族已核定的
    # 通解，推理分支带着它作答；Python 分支虽也看得到卡片，却常按自创口径重新枚举
    # （口径差一格的枚举恰好是上面 Q59 的来源）。两侧冲突时交给带口径的仲裁裁决。
    if ptype == "computation" and python_ok and not looks_incomplete_answer(python_answer) \
            and match_result.get("status") != "mismatch" and not matched_cards(problem):
        # 反伪造：无实质计算的代码答案不得凭"执行成功"压过一个可用的推理答案。
        # mismatch（两分支真实冲突、仲裁又不可用）时不走这里：机械偏袒任何一方
        # 都是掷硬币（2026-08-31 Q40 实测：推理的 194 正确，Python 的 97 是
        # 公式打印，仍凭 computation+success 压过推理出厂）。冲突只能由带证据
        # 的语义仲裁裁决；兜底默认回退到可审计的完整推导（推理）。
        if not (fabricated and reasoning_ok):
            return python_answer
    if reasoning_ok:
        return reasoning_answer
    return reasoning_answer or python_answer


def cross_validator_node(state, config):
    reasoning_result = state.get("reasoning_result") or {}
    raw_python_output = state.get("python_output") or {}
    python_output = parse_verification_evidence(
        raw_python_output,
        candidate_answer=reasoning_result.get("answer", ""),
        code=state.get("python_code", ""),
        problem=state.get("problem", ""),
    )
    question_mode = state.get("question_mode") or classify_question_mode(state.get("problem", ""))
    if is_objective_mode(question_mode) and "objective_review_result" not in state:
        # Objective questions deliberately have no Python candidate.  A valid,
        # normalized answer from the concise reasoning path is the only candidate
        # worth emitting; sending it through the computation matcher would mark it
        # uncertain solely because Python was skipped and trigger an unnecessary
        # arbitration/reconciliation round.
        candidate = normalize_objective_answer(reasoning_result.get("answer", ""), question_mode)
        if not objective_answer_is_usable(candidate, question_mode):
            candidate = normalize_objective_answer(python_output.get("answer", ""), question_mode)
        semantic_reason = ""
        if candidate:
            consistent, semantic_reason = objective_answer_consistency(
                candidate, state.get("problem", ""), question_mode)
            if not consistent:
                candidate = ""
        if objective_answer_is_usable(candidate, question_mode):
            blank_gap = question_mode == "fill" and not fill_answer_matches_blanks(
                candidate, state.get("problem", ""))
            match_result = {
                "status": "uncertain" if blank_gap else "match",
                "verdict": True,
                "comparison_verdict": True,
                # 填空分项数不足题面空位数时降低置信（评委报告 idx 86：残缺答案
                # 曾以 0.78 置信直接放行，无任何完整性检查）。
                "confidence": 0.45 if blank_gap else 0.78,
                "reason": f"客观题快速路径已提取{question_mode}答案"
                          + ("；但分项数少于题面空位数，答案可能不完整" if blank_gap else ""),
                "method": "objective_blank_gap" if blank_gap else "objective_direct",
                "problem_type": question_mode,
                "text_similarity": 1.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.5 if blank_gap else 1.0,
            }
        else:
            match_result = {
                "status": "uncertain",
                "verdict": None,
                "comparison_verdict": None,
                "confidence": 0.0,
                "reason": ("客观题候选未通过语义核验：" + semantic_reason)
                          if semantic_reason else "客观题未提取到可提交的答案",
                "method": "objective_semantic_mismatch" if semantic_reason
                          else "objective_unparsed",
                "problem_type": question_mode,
                "text_similarity": 0.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
            }
    elif is_objective_mode(question_mode):
        review_result = state.get("objective_review_result") or {}
        first = normalize_objective_answer(
            reasoning_result.get("answer", ""), question_mode
        )
        second = normalize_objective_answer(
            review_result.get("answer", ""), question_mode
        )
        valid_first = objective_answer_is_usable(first, question_mode)
        valid_second = objective_answer_is_usable(second, question_mode)
        if valid_first:
            valid_first = objective_answer_consistency(
                first, state.get("problem", ""), question_mode
            )[0]
        if valid_second:
            valid_second = objective_answer_consistency(
                second, state.get("problem", ""), question_mode
            )[0]
        objective_candidates = [
            {"source": "reasoning", "answer": first, "usable": valid_first},
            {"source": "objective_review", "answer": second, "usable": valid_second},
        ]
        if valid_first and valid_second:
            same = _objective_key(first, question_mode) == _objective_key(
                second, question_mode
            )
            match_result = {
                "status": "match" if same else "mismatch",
                "verdict": same,
                "comparison_verdict": same,
                "confidence": 0.92 if same else 0.35,
                "reason": "独立客观复核候选一致" if same else "独立客观复核候选不一致",
                "method": "objective_review",
                "problem_type": question_mode,
                "text_similarity": 1.0 if same else 0.0,
                "matched_fields": ["objective_answer"] if same else [],
                "mismatched_fields": [] if same else ["objective_answer"],
                "field_coverage": 1.0,
                "objective_candidates": objective_candidates,
            }
            if not same:
                match_result["recheck_required"] = True
        elif valid_first or valid_second:
            match_result = {
                "status": "uncertain",
                "verdict": None,
                "comparison_verdict": None,
                "confidence": 0.35,
                "reason": "客观题只有一个可用候选，缺少独立复核",
                "method": "objective_single_candidate",
                "problem_type": question_mode,
                "text_similarity": 0.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 1.0,
                "objective_candidates": objective_candidates,
                "recheck_required": True,
            }
        else:
            match_result = {
                "status": "uncertain",
                "verdict": None,
                "comparison_verdict": None,
                "confidence": 0.0,
                "reason": "客观题候选未通过格式或语义核验",
                "method": "objective_unparsed",
                "problem_type": question_mode,
                "text_similarity": 0.0,
                "matched_fields": [],
                "mismatched_fields": [],
                "field_coverage": 0.0,
                "objective_candidates": objective_candidates,
                "recheck_required": True,
            }
    else:
        match_result = dict(AnswerMatcher.match_answers(
            problem=state["problem"],
            reasoning_result=reasoning_result,
            python_result=python_output,
        ))
    reasoning_health = assess_candidate_health(
        state.get("problem", ""),
        reasoning_result.get("answer", ""),
        mode=question_mode,
    )
    python_health = python_output.get("candidate_health") or assess_candidate_health(
        state.get("problem", ""),
        python_output.get("answer", ""),
        mode=question_mode,
        python_output=python_output,
    )
    objective_review_payload = state.get("objective_review_result") or {}
    objective_health = assess_candidate_health(
        state.get("problem", ""),
        objective_review_payload.get("answer", ""),
        mode=question_mode,
    ) if "objective_review_result" in state else None
    candidate_health = {"reasoning": reasoning_health, "python": python_health}
    if objective_health is not None:
        candidate_health["objective_review"] = objective_health
    evidence_status = python_output.get("evidence_status", "inconclusive")
    evidence_summary = python_output.get("evidence_summary", "")
    contradictions = list(python_output.get("contradictions") or [])
    match_result.update({
        "python_evidence_status": evidence_status,
        "python_evidence_summary": evidence_summary,
        "python_contradictions": contradictions,
        "candidate_health": candidate_health,
    })
    status = match_result["status"]
    validated_answer = ""
    next_node = "coordinator"
    problem_type = match_result.get("problem_type", "computation")

    history_entry = {
        "round": state.get("reconciliation_round", 0),
        "status": status,
        "evidence_status": evidence_status,
        "evidence_summary": evidence_summary[:1000],
        "contradictions": contradictions[:10],
        "candidate_health": candidate_health,
    }

    # Explicit evidence of a counterexample outranks textual/symbolic agreement.
    # A program can run successfully while proving that the candidate is false.
    if evidence_status == "contradict":
        match_result["routing_reason"] = "python_evidence_contradiction"
        match_result["recheck_required"] = True
        if reconciliation_retry_available(state, config, force=True):
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            status = "mismatch_arbitrating"
            next_node = "semantic_arbiter"
            match_result["unresolved_contradiction"] = True
        history_entry["status"] = status
        return {
            "validation_status": status,
            "validation_details": match_result,
            "validated_answer": validated_answer,
            "next_node": next_node,
            "python_output": python_output,
            "python_evidence_status": evidence_status,
            "python_evidence_summary": evidence_summary,
            "python_contradictions": contradictions,
            "candidate_health": candidate_health,
            "recheck_required": True,
            "validation_history": [history_entry],
        }

    if status == "match":
        # Identical text is not independent evidence.  If a Python branch was
        # present, it must provide auditable support and a healthy payload before
        # a match can bypass reconciliation.
        python_present = bool(
            raw_python_output.get("answer")
            or raw_python_output.get("success")
            or raw_python_output.get("stdout")
        )
        weak_match = (
            python_present
            and (
                python_health.get("evidence_quality") != "support"
                or not python_health.get("format_complete", False)
                or python_health.get("numeric_health") == "fail"
            )
        ) or (
            bool(reasoning_result.get("answer"))
            and not reasoning_health.get("format_complete", False)
        )
        if weak_match:
            match_result["recheck_required"] = True
            match_result["routing_reason"] = (
                "same_candidate_without_independent_healthy_evidence"
            )
            if reconciliation_retry_available(state, config, force=True):
                status = "mismatch_reconciling"
                next_node = "reconciliation"
            else:
                status = "match_arbitrating"
                next_node = "semantic_arbiter"
                match_result["budget_blocked"] = True
            history_entry["status"] = status
            return {
                "validation_status": status,
                "validation_details": match_result,
                "validated_answer": validated_answer,
                "next_node": next_node,
                "python_output": python_output,
                "python_evidence_status": evidence_status,
                "python_evidence_summary": evidence_summary,
                "python_contradictions": contradictions,
                "candidate_health": candidate_health,
                "recheck_required": True,
                "validation_history": [history_entry],
            }
        if problem_type == "computation" and python_output.get("success") \
                and match_result.get("method") in {"symbolic", "structured_symbolic"}:
            match_result["bypass_reason"] = "python_success_and_symbolic_equivalence"
        match_result["routing_reason"] = "validated_match"
        if match_result.get("method") == "objective_direct":
            validated_answer = normalize_objective_answer(
                reasoning_result.get("answer", ""), question_mode
            )
            if not validated_answer:
                validated_answer = normalize_objective_answer(
                    python_output.get("answer", ""), question_mode
                )
        else:
            # 传解析后的 python_output：可信度字段由 parse_verification_evidence
            # 写入，state 里的原始输出还没有它，否则可信分支会被误判为不可信。
            validated_answer = _preferred_answer(
                {**state, "python_output": python_output}, match_result)
        next_node = "coordinator"
    elif status == "mismatch":
        # subgraph-level retry gated by reconciliation_round (NOT per-node attempts —
        # per-node attempts gate the agent's internal format/code retries; the plan
        # §4.2 keeps these separate).
        # A candidate disagreement is itself an unresolved evidence gap.  Preserve
        # that fact in state even when the retry circuit is exhausted so the
        # arbiter/coordinator cannot treat the route as an ordinary match.
        match_result["recheck_required"] = True
        match_result.setdefault("routing_reason", "candidate_disagreement_requires_recheck")
        if reconciliation_retry_available(
            state,
            config,
            force=True,
        ):
            status = "mismatch_reconciling"
            next_node = "reconciliation"
        else:
            status = "mismatch_arbitrating"
            match_result["routing_reason"] = "mismatch_retry_exhausted_semantic_arbitration"
            next_node = "semantic_arbiter"
    else:  # uncertain
        # Parser/format uncertainty is exactly where keyword contracts are weakest.
        # Let a semantic judge choose between the two existing answers before paying
        # for another full reasoning+Python run. The arbiter may abstain and route
        # back to reconciliation without changing either candidate.
        if match_result.get("recheck_required") and reconciliation_retry_available(
            state, config, force=True
        ):
            status = "uncertain_reconciling"
            match_result["routing_reason"] = "insufficient_evidence_forced_recheck"
            next_node = "reconciliation"
        else:
            status = "uncertain_arbitrating"
            match_result["routing_reason"] = "insufficient_equivalence_evidence_semantic_arbitration"
            next_node = "semantic_arbiter"

    history_entry["status"] = status
    return {
        "validation_status": status,
        "validation_details": match_result,
        "validated_answer": validated_answer,
        "next_node": next_node,
        "python_output": python_output,
        "python_evidence_status": evidence_status,
        "python_evidence_summary": evidence_summary,
        "python_contradictions": contradictions,
        "candidate_health": candidate_health,
        "recheck_required": bool(
            match_result.get("recheck_required") or state.get("recheck_required")
        ),
        "validation_history": [history_entry],
    }
