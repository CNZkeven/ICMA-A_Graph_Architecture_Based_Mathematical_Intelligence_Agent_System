"""Conservative, shared health checks for candidate answers.

The validator must distinguish a usable answer from evidence that actually
supports it.  This module deliberately reports ``unknown`` when a text-only
heuristic cannot decide; callers may request a recheck, but never manufacture a
value from the score.
"""

from __future__ import annotations

import math
import re
from typing import Any

from utils.answer_cleanliness import is_noise_answer
from utils.answer_contract import missing_components
from utils.answer_extractor import looks_incomplete_answer
from utils.cot_stripper import is_placeholder_answer
from utils.problem_profile import (
    fill_answer_matches_blanks,
    is_objective_mode,
)


_NONNEGATIVE_CUE_RE = re.compile(
    r"(?i)\b(?:count|counting|number\s+of|how\s+many|cardinality|"
    r"probability|probabilities|length|distance|area|volume|perimeter|"
    r"arrangements?|permutations?|combinations?|ways|quantity|amount)\b|"
    r"计数|个数|数量|概率|长度|距离|面积|体积|周长|排列|组合|方案|路径数|"
    r"总数|人数|次数"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_NONFINITE_RE = re.compile(r"(?i)\b(?:nan|inf(?:inity)?)\b|非有限|无穷大")


def _numeric_health(problem: str, answer: str) -> tuple[str, list[str]]:
    text = str(answer or "")
    if _NONFINITE_RE.search(text):
        return "fail", ["答案包含非有限数值"]
    if not _NONNEGATIVE_CUE_RE.search(str(problem or "")):
        return "unknown", []
    values: list[float] = []
    for token in _NUMBER_RE.findall(text):
        try:
            value = float(token)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            return "fail", ["答案包含非有限数值"]
        values.append(value)
    if not values:
        return "unknown", []
    if any(value < 0 for value in values):
        return "fail", ["题面要求非负量，但候选含负值"]
    return "pass", []


def _evidence_health(python_output: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not python_output:
        return "none", []
    status = str(python_output.get("evidence_status") or "").lower()
    trusted = python_output.get("python_verification_trusted")
    if trusted is False or (python_output.get("authenticity") or {}).get("fabricated"):
        return "inconclusive", ["Python 验证未通过可信度检查"]
    if status == "contradict":
        return "contradict", [
            str(item) for item in (python_output.get("contradictions") or [])[:3]
        ] or ["Python 验证包含反例或失败断言"]
    if status == "support" and python_output.get("success"):
        return "support", []
    if status == "inconclusive" or python_output.get("success"):
        return "inconclusive", ["Python 输出没有可审计的支持证据"]
    return "none", []


def assess_candidate_health(
    problem: str,
    answer: str,
    *,
    mode: str = "computation",
    python_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic health metadata for one candidate.

    ``usable`` means the text can participate in a selection; it does not mean
    the mathematics is proven.  A Python contradiction or a failed numeric
    invariant is definitive negative evidence, while missing evidence remains
    ``unknown`` and should normally trigger a bounded recheck.
    """
    text = str(answer or "").strip()
    reasons: list[str] = []
    format_complete = bool(text)
    if not text or is_placeholder_answer(text) or is_noise_answer(text):
        format_complete = False
        reasons.append("答案为空、占位或包含思维碎片")
    elif looks_incomplete_answer(text):
        format_complete = False
        reasons.append("答案疑似截断或未闭合")
    if format_complete:
        missing = missing_components(problem, text)
        if missing:
            format_complete = False
            reasons.append("答案缺少题面要求的组成部分: " + ", ".join(missing))
        if mode == "fill" and not fill_answer_matches_blanks(text, problem):
            format_complete = False
            reasons.append("填空分项数少于题面空位数")
    numeric_health, numeric_reasons = _numeric_health(problem, text)
    reasons.extend(numeric_reasons)
    evidence_quality, evidence_reasons = _evidence_health(python_output)
    reasons.extend(evidence_reasons)
    counterexample_free = {
        "contradict": "fail",
        "support": "pass",
    }.get(evidence_quality, "unknown")

    usable = format_complete and numeric_health != "fail" and evidence_quality != "contradict"
    # Objective candidates have no Python branch; their absence of executable
    # evidence is expected and must not make a well-formed choice unusable.
    if is_objective_mode(mode) and evidence_quality in {"none", "inconclusive"}:
        usable = format_complete and numeric_health != "fail"
    score = 0.0
    if format_complete:
        score += 0.45
    if numeric_health == "pass":
        score += 0.2
    elif numeric_health == "unknown":
        score += 0.1
    if evidence_quality == "support":
        score += 0.35
    elif evidence_quality == "none":
        score += 0.1
    return {
        "usable": bool(usable),
        "format_complete": bool(format_complete),
        "evidence_quality": evidence_quality,
        "counterexample_free": counterexample_free,
        "numeric_health": numeric_health,
        "reasons": reasons,
        "score": round(min(score, 1.0), 3),
    }

