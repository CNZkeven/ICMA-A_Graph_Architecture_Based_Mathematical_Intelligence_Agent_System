"""Independent short review for objective questions."""

from __future__ import annotations

import re
from typing import Any

from config import CONFIG
from utils.cot_stripper import strip_cot_prefix
from utils.deps import get_deps
from utils.llm_retry import chat_prefilled
from utils.problem_profile import (
    fill_answer_matches_blanks,
    judgement_item_count,
    normalize_objective_answer,
    objective_answer_consistency,
    objective_answer_is_usable,
)
from utils.prompt_templates import OBJECTIVE_REVIEW_PREFILL, OBJECTIVE_REVIEW_PROMPT
from utils.solution_cards import select_excerpt_with_cards as select_skill_excerpt
from utils.token_budget import estimate_tokens


def _parse_review_response(response: str, mode: str, problem: str) -> dict[str, Any]:
    text = strip_cot_prefix(response or "")
    raw_candidate = ""
    labelled = list(re.finditer(
        r"(?im)(?:^|\n)\s*(?:最终答案|答案|answer)\s*[：:]\s*([^\n]+)",
        text,
    ))
    if labelled:
        raw_candidate = labelled[-1].group(1).strip()
    elif mode == "choice":
        for line in reversed(text.splitlines()):
            if re.fullmatch(r"\s*[A-EＡ-Ｅ](?:\s*[、,，/和及&]\s*[A-EＡ-Ｅ])*\s*[。.]?\s*", line):
                raw_candidate = line.strip()
                break
    elif mode == "true_false":
        if judgement_item_count(problem) >= 2:
            # 多条判断：整段交给归一化器按条目抽取，不能只取首行（首行只有第 1 条）。
            raw_candidate = text.strip()
        else:
            raw_candidate = text.strip().splitlines()[0] if text.strip() else ""
    else:
        raw_candidate = text.strip()

    answer = normalize_objective_answer(raw_candidate, mode)
    if not objective_answer_is_usable(answer, mode):
        answer = ""
    if mode == "fill" and answer and not fill_answer_matches_blanks(answer, problem):
        return {
            "analysis": "",
            "steps": [],
            "answer": "",
            "incomplete_answer": answer,
            "validation_points": [],
        }
    if answer:
        consistent, reason = objective_answer_consistency(answer, problem, mode)
        if not consistent:
            return {
                "analysis": "",
                "steps": [],
                "answer": "",
                "invalid_answer": answer,
                "semantic_invalid_reason": reason,
                "validation_points": [],
            }
    evidence = re.search(r"(?:依据|理由|说明)\s*[：:]\s*(.+)", text, re.S)
    step_text = evidence.group(1).strip() if evidence else "独立按题面定义逐项核对。"
    return {
        "analysis": "独立复核原题定义与答案覆盖度。",
        "steps": [{"step_num": 1, "description": step_text[:600]}] if answer else [],
        "answer": answer,
        "validation_points": [],
    }


def objective_review_node(state: dict, config) -> dict:
    """Ask for one answer candidate without exposing the first branch's result."""
    deps = get_deps(config)
    problem = str(state.get("problem") or "")
    mode = str(state.get("question_mode") or "choice")
    category = str(state.get("category") or "")
    # 学科口径参考：盲复核不等于无教材。知识型客观题的判分口径来自技能文档
    # （2026-08-31 实测：无口径的盲复核在 Q99/101/103 全错，且经仲裁覆盖了
    # 正确首答）。它不暴露第一分支的候选，独立性体现在"独立采样"，不在"无知识"。
    skill_reference = ""
    try:
        skill_doc = deps.skills_loader.get_skill_document(category)
        # 额度与客观推理路径一致（2200）：低于文档前导长度时，"客观题高频口径"
        # 里的判分口径会被头部截断（2026-08-31 Q103 实测：1600 额度截掉了
        # "时间序列五要素全选"规则，复查分支据此答错并经仲裁覆盖正确首答）。
        skill_reference = select_skill_excerpt(skill_doc, problem, 2200)
    except Exception:  # noqa: BLE001 - 参考缺失时退化为纯盲复核。
        skill_reference = ""
    prompt = OBJECTIVE_REVIEW_PROMPT.format(
        question_mode=mode,
        category=category,
        problem=problem,
        skill_reference=skill_reference or "（无）",
    )
    trace: list[dict[str, Any]] = []
    response = ""
    attempts = 1
    try:
        response = chat_prefilled(
            deps.client,
            messages=[{"role": "user", "content": prompt}],
            prefix=OBJECTIVE_REVIEW_PREFILL,
            temperature=CONFIG["temperatures"].get(
                "objective_review",
                CONFIG["temperatures"].get("objective_reasoning", 0.2),
            ),
            max_tokens=CONFIG["max_tokens"].get(
                "objective_review", CONFIG["max_tokens"].get("objective_reasoning", 4096)
            ),
            logger=deps.logger,
            time_budget=deps.time_budget,
            expected_call_seconds=CONFIG.get("objective_review_expected_call_s", 35),
            label="objective_review",
        )
        if deps.token_budget:
            deps.token_budget.consume(estimate_tokens(prompt), estimate_tokens(response))
        parsed = _parse_review_response(response, mode, problem)
        trace.append({
            "attempt": attempts,
            "status": "success" if parsed.get("answer") else "unparsed",
            "response_chars": len(response or ""),
        })
    except Exception as exc:  # noqa: BLE001 - review is an optional evidence source.
        parsed = {
            "analysis": "",
            "steps": [],
            "answer": "",
            "validation_points": [],
        }
        trace.append({
            "attempt": attempts,
            "status": "failed",
            "error": str(exc)[:200],
        })
    return {
        "objective_review_result": parsed,
        "objective_review_trace": trace,
        "objective_review_attempts": attempts,
    }

