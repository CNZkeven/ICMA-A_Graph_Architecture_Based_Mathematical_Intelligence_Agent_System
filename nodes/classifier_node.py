import json
import re
from utils.deps import get_deps
from utils.llm_retry import chat_with_retry
from utils.prompt_templates import CLASSIFICATION_PROMPT
from utils.token_budget import estimate_tokens
from utils.cot_stripper import strip_cot_prefix
from config import CONFIG


def _parse_json(text):
    text = strip_cot_prefix(text).strip()
    # try whole text
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first balanced {...} that parses
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break  # try next start
        start = text.find("{", start + 1)
    return None


_PUNCT_STRIP = re.compile(r"[\s　\"'.,。，、：:；;！!？?（）()【】\[\]{}「」『』]")


def _normalize_cat(s: str) -> str:
    """去除空白、引号与常见中英文标点，用于分类名的模糊匹配。"""
    return _PUNCT_STRIP.sub("", s or "")


def _lcs_length(a: str, b: str) -> int:
    """最长公共子串长度（连续字符）。对中文类别名比编辑距离更可靠。"""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def _resolve_category(raw, valid_categories, fallback):
    """将 LLM 返回的 category 归一到 skills_pythonscripts/ 下的真实文件夹名。

    严格保证返回值 ∈ valid_categories（即实际文件夹名），杜绝下游
    get_skill_document / get_validation_script 因类别名不存在而 FileNotFoundError。

    匹配顺序：精确 → 归一化 → 子串包含（双向）→ 最长公共子串（≥2）→ Levenshtein（≥0.6）→ fallback。
    fallback 始终是关键词检索 top1，必为真实文件夹名。
    """
    if not raw:
        return fallback
    s = raw.strip().strip("\"'「」『』").strip()
    if not s:
        return fallback
    # 1. 精确匹配
    if s in valid_categories:
        return s
    # 2. 归一化匹配（去空白/标点后比较）
    sn = _normalize_cat(s)
    for c in valid_categories:
        if _normalize_cat(c) == sn:
            return c
    # 3. 子串包含（任一方向）
    for c in valid_categories:
        if s in c or c in s:
            return c
    # 4. 最长公共子串（连续≥2字符）—— 对中文类别名比 Levenshtein 更可靠
    #    例：线性代数 ↔ 高等代数（共享"代数"）、概率统计 ↔ 概率论（共享"概率"）
    lcs_matches = [(c, _lcs_length(s, c)) for c in valid_categories]
    lcs_matches = [(c, l) for c, l in lcs_matches if l >= 2]
    if lcs_matches:
        max_lcs = max(l for _, l in lcs_matches)
        tied = [c for c, l in lcs_matches if l == max_lcs]
        if len(tied) == 1:
            return tied[0]
        # LCS 并列时用 Levenshtein ratio 精排（best-effort，区分长短不同的候选）
        try:
            from Levenshtein import ratio as lev_ratio
            return max(tied, key=lambda c: lev_ratio(s, c))
        except ImportError:
            return tied[0]
    # 5. Levenshtein 模糊匹配（兜底）
    try:
        from Levenshtein import ratio as lev_ratio
        best, best_score = max(((c, lev_ratio(s, c)) for c in valid_categories),
                               key=lambda x: x[1])
        if best_score >= 0.6:
            return best
    except ImportError:
        pass
    return fallback


def classifier_node(state, config):
    deps = get_deps(config)
    sl = deps.skills_loader
    client = deps.client
    budget = deps.token_budget
    problem = state["problem"]
    stages = ["all_categories", "keyword"]
    keyword_candidates = sl.find_candidate_categories(problem, top_k=CONFIG["classifier_top_k"])
    fallback = keyword_candidates[0][0] if keyword_candidates else "非基础及进阶课程"
    cand_names = list(sl.categories)
    keyword_hint = "、".join(c for c, _ in keyword_candidates) or fallback
    skill_docs = "\n".join(f"## {c}\n{sl.get_skill_document(c)[:350]}..." for c in cand_names)
    allowed_list = "、".join(cand_names)
    prompt = CLASSIFICATION_PROMPT.format(
        problem=problem,
        skill_docs=skill_docs,
        allowed_categories=allowed_list,
        keyword_hint=keyword_hint,
    )
    for _ in range(3):
        resp = chat_with_retry(
            client,
            messages=[{"role": "user", "content": prompt}],
            temperature=CONFIG["temperatures"]["classifier"],
            max_tokens=CONFIG["max_tokens"]["classifier"],
            logger=deps.logger,
        )
        if budget:
            budget.consume(estimate_tokens(prompt), estimate_tokens(resp))
        parsed = _parse_json(resp)
        if parsed and "category" in parsed:
            # 关键：LLM 返回的 category 必须归一到真实文件夹名，否则下游 FileNotFoundError
            resolved = _resolve_category(parsed["category"], sl.categories, fallback)
            stages.append("llm")
            return {"category": resolved,
                    "category_confidence": float(parsed.get("confidence", 0.0)),
                    "candidate_categories": cand_names,
                    "classification_stages_used": stages}
    return {"category": fallback, "category_confidence": 0.0,
            "candidate_categories": cand_names,
            "classification_stages_used": stages}
