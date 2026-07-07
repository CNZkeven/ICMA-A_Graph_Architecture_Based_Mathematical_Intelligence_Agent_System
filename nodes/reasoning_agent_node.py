import re
from utils.deps import get_deps
from utils.llm_retry import chat_with_retry
from utils.prompt_templates import REASONING_PROMPT
from utils.token_budget import estimate_tokens
from utils.answer_extractor import looks_incomplete_answer, looks_like_latex_fragment
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
    """Placeholder、被截头的 LaTeX 残片或明显不完整的碎片一律不作为答案。"""
    answer = (answer or "").strip()
    if is_placeholder_answer(answer) or looks_like_latex_fragment(answer) \
            or looks_incomplete_answer(answer):
        return ""
    return answer


_ENUM_ITEM_RE = re.compile(r"^\s*(?:\d+\s*[.、)]|[-*•①②③④⑤])")
# 行是否携带答案信息：等式 / 数学式 / 数字 / 结论词。纯引导语（"…如下所示："）丢弃。
_PAYLOAD_LINE_RE = re.compile(r"[=$]|\d|拒绝|接受|显著|结论|因此|所以|故|建议|选择|综上|存在|唯一|收敛|成立")


def _strip_bullet(line):
    return line.lstrip("-*•①②③④⑤ ").strip()


def _join_parts(parts):
    """用全角分号连接各行；行尾自带分号时去重，避免出现 '；；'。"""
    cleaned = [p.rstrip("；; ").strip() for p in parts if p and p.strip("；; ")]
    return "；".join(cleaned)


def _rhs_of_top_level_eq(line):
    """行内最后一个"顶层" '=' 的右侧内容；没有顶层 '=' 时返回 ''。

    顶层 = 不在任何括号/花括号内，且不在 $...$ 数学模式内。这避免了从
    \\sum_{i=1}^n 的下标 'i=1' 或整条 LaTeX 公式中间切断（评委报告：
    '1}^n X_i$'、'\\sigma^2$ 且无自相关' 均由此产生）。
    """
    depth = 0
    in_math = False
    pos = -1
    for i, ch in enumerate(line):
        if ch == "$":
            in_math = not in_math
        elif ch in "{([":
            depth += 1
        elif ch in "})]":
            depth = max(0, depth - 1)
        elif ch == "=" and depth == 0 and not in_math:
            pos = i
    return line[pos + 1:].strip() if pos >= 0 else ""


def _distill_answer(text):
    """Distill a concise answer from the '## 最终答案' section text.

    策略顺序（每个候选都要通过 placeholder/残片检查，失败则落到下一策略）：
    1. 枚举节（≥2 个编号/列表项）→ 全部条目整体保留（评委报告：抽象代数__013
       曾只剩 1 个群）。
    2. 多行短节 → 保留所有信息行（含无 '=' 的结论行——统计推断__009~012 的
       "拒绝/不拒绝 H0" 结论行曾被只留等式行的旧逻辑丢弃）。
    3. 短节（≤80 字符）→ 最后一行。
    4. 显式 '答案是X' 模式。
    5. 行尾顶层 '=' 右值（LaTeX 感知，绝不切进公式内部）；右值残缺时保留整行。
    6. 首个信息行。
    """
    text = (text or "").strip()
    if is_placeholder_answer(text):
        return ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1. 枚举节：所有条目整体保留（保留带信息的引导行，如"共 4 种："）
    enum_items = [l for l in lines if _ENUM_ITEM_RE.match(l)]
    if len(enum_items) >= 2 and len(text) <= 1600:
        kept = [_strip_bullet(l) for l in lines
                if _ENUM_ITEM_RE.match(l) or _PAYLOAD_LINE_RE.search(l)]
        joined = _join_parts(kept)
        if not looks_like_latex_fragment(joined) and not is_placeholder_answer(joined):
            return joined

    # 2. 多行短节：保留所有信息行（等式行 + 结论行），过滤纯引导语
    eq_lines = [_strip_bullet(l) for l in lines if "=" in l]
    if 2 <= len(lines) <= 8 and len(text) <= 900 and eq_lines \
            and all(len(l) <= 200 for l in lines):
        kept = [_strip_bullet(l) for l in lines if _PAYLOAD_LINE_RE.search(l)]
        joined = _join_parts(kept)
        if kept and not looks_like_latex_fragment(joined) and not is_placeholder_answer(joined):
            return joined
    if len(eq_lines) >= 3:
        conclusion_extra = [_strip_bullet(l) for l in lines
                            if "=" not in l and re.search(r"拒绝|接受|显著|结论|建议|选择|综上", l)]
        joined = _join_parts(eq_lines + conclusion_extra)
        if len(joined) <= 1600 and not looks_like_latex_fragment(joined) and not is_placeholder_answer(joined):
            return joined

    # 3. 短节 → 最后一行
    if len(text) <= 80:
        return _reject_bad_answer(lines[-1] if lines else text)

    # 4. 显式"答案是X"
    for pat in (r"答案[是为：:]\s*(.+?)(?:\n|$)", r"answer\s*[:is]+\s*(.+?)(?:\n|$)"):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            answer = _reject_bad_answer(m.group(1))
            if answer:
                return answer

    # 5. 行尾顶层 '=' 右值；右值残缺时退回整行（不得切进 LaTeX 公式内部）
    eq_line = next((l for l in reversed(lines) if "=" in l), "")
    if eq_line:
        answer = _reject_bad_answer(_rhs_of_top_level_eq(eq_line))
        if answer:
            return answer
        answer = _reject_bad_answer(_strip_bullet(eq_line))
        if answer:
            return answer

    # 6. 首个信息行，最后兜底整节
    for cand in ([l for l in lines if _PAYLOAD_LINE_RE.search(l)][:1]
                 + ([lines[0]] if lines else [])
                 + ([text] if len(text) <= 600 else [])):
        answer = _reject_bad_answer(cand)
        if answer:
            return answer
    return ""


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
    base_prompt = REASONING_PROMPT.format(category=category, skill_document=skill_doc[:3000], problem=problem)
    hint = state.get("branch_hint")
    # 复核重试也保留 skill 文档与字段契约（旧实现只发 hint+题面，丢失了学科口径）
    prompt = f"{base_prompt}\n\n[复核提示] {hint}" if hint else base_prompt
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
                  "'## 最终答案'）。'## 最终答案' 下必须按题面要求完整列出各字段/各问项的结果"
                  "（含检验结论、区间上下限、全部枚举对象等），不得只给单个数值。请重新严格按格式输出。")
    return {"reasoning_result": parsed, "reasoning_trace": trace, "reasoning_attempts": attempts}
