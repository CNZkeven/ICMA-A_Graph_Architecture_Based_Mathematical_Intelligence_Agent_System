import re
from utils.deps import get_deps
from utils.llm_retry import chat_with_retry
from utils.prompt_templates import REASONING_PROMPT
from utils.token_budget import estimate_tokens
from utils.answer_extractor import looks_like_latex_fragment
from utils.cot_stripper import is_placeholder_answer, strip_cot_prefix
from config import CONFIG


def _parse_reasoning_output(response):
    response = strip_cot_prefix(response)
    r = {"analysis": "", "steps": [], "answer": "", "validation_points": []}
    m = re.search(r"## 问题分析\s+(.*?)(?=##|$)", response, re.DOTALL)
    if m:
        r["analysis"] = m.group(1).strip()
    sec = re.search(r"## 详细解题步骤\s+(.*?)(?=## 最终答案|$)", response, re.DOTALL)
    if sec:
        for sm in re.finditer(r"(?:步骤|Step)\s*(\d+)\s*[：:](.*?)(?=(?:步骤|Step)\s*\d+\s*[：:]|##|$)",
                              sec.group(1), re.DOTALL | re.IGNORECASE):
            r["steps"].append({"step_num": int(sm.group(1)), "description": sm.group(2).strip()})
    am = re.search(r"## 最终答案\s+(.*?)(?=##|$)", response, re.DOTALL)
    if am:
        r["answer"] = _distill_answer(am.group(1).strip())
    vm = re.search(r"## 关键验证点\s+(.*?)(?=##|$)", response, re.DOTALL)
    if vm:
        r["validation_points"] = re.findall(r"-\s*(.+)", vm.group(1))
    return r


def _reject_bad_answer(answer):
    """Placeholder 或被截头的 LaTeX 残片（评委报告 idx=112）一律不作为答案。"""
    answer = (answer or "").strip()
    if is_placeholder_answer(answer) or looks_like_latex_fragment(answer):
        return ""
    return answer


def _distill_answer(text):
    """Distill a concise answer from the '## 最终答案' section text.

    Multi-part enumeration (≥2 '名称=值' lines) → keep all parts joined. Short
    section → last non-empty line. Long prose → explicit '答案是X' pattern, then
    trailing '= value', then first line. Every candidate is rejected when it is a
    placeholder or a decapitated LaTeX fragment (e.g. a '= tail' grabbed from
    inside a \\begin{{cases}} environment), falling through to the next strategy.
    """
    text = (text or "").strip()
    if is_placeholder_answer(text):
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # 多问项答案节：形如每行"f_Z(z) = ..."、"P(Z>1) = 1/2"的枚举必须整体保留（idx=172 曾只取一行）
    eq_lines = [l.lstrip("-*• ").strip() for l in lines if "=" in l]
    if len(lines) <= 6 and len(eq_lines) >= 2 and len(text) <= 600 \
            and all(len(l) <= 160 for l in eq_lines):
        joined = "；".join(eq_lines)
        if not looks_like_latex_fragment(joined) and not is_placeholder_answer(joined):
            return joined
    if len(text) <= 80:
        return _reject_bad_answer(lines[-1] if lines else text)
    for pat in (r"答案[是为：:]\s*(.+?)(?:\n|$)", r"answer\s*[:is]+\s*(.+?)(?:\n|$)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            answer = _reject_bad_answer(m.group(1))
            if answer:
                return answer
    m = re.search(r"=\s*([^=\n]+?)\s*$", text, re.MULTILINE)
    if m:
        answer = _reject_bad_answer(m.group(1))
        if answer:
            return answer
    return _reject_bad_answer(lines[0] if lines else text)


def _is_complete(r):
    return bool(r["answer"]) and len(r["steps"]) >= 1


def reasoning_agent_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    client = deps.client
    budget = deps.token_budget
    max_attempts = 1 if budget and budget.is_tight() else CONFIG["max_retries_per_node"]
    problem, category = state["problem"], state["category"]
    skill_doc = sl.get_skill_document(category)
    base_prompt = REASONING_PROMPT.format(category=category, skill_document=skill_doc[:2000], problem=problem)
    hint = state.get("branch_hint")
    prompt = f"{hint}\n\n问题：{problem}" if hint else base_prompt
    trace = []
    attempts = 0
    parsed = {"analysis": "", "steps": [], "answer": "", "validation_points": []}
    for _ in range(max_attempts):
        attempts += 1
        resp = chat_with_retry(
            client,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["temperatures"]["reasoning"],
            max_tokens=CONFIG["max_tokens"]["reasoning"],
            logger=deps.logger,
        )
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(resp))
        parsed = _parse_reasoning_output(resp)
        trace.append({"attempt": attempts, "status": "success" if _is_complete(parsed) else "failed"})
        if _is_complete(parsed):
            return {"reasoning_result": parsed, "reasoning_trace": trace, "reasoning_attempts": attempts}
        # retry: keep base context (skill doc + category) + explicit format reminder
        prompt = (base_prompt + "\n\n注意：上一次输出缺少必需章节（必须含 '## 问题分析'、'## 详细解题步骤'、"
                  "'## 最终答案'，且'## 最终答案'后只给简洁答案值）。请重新严格按格式输出。")
    return {"reasoning_result": parsed, "reasoning_trace": trace, "reasoning_attempts": attempts}
