import re


_COT_PREFIX_RE = re.compile(
    r"^\s*(?:Thinking Process|Thought Process|Thinking|Reasoning|Let me think|思考过程|推理过程|分析过程)\s*[：:]",
    re.IGNORECASE,
)

_CONTENT_START_RE = re.compile(
    r"(?m)^\s*(?:"
    r"\{"
    r"|```(?:python)?"
    r"|##\s+"
    r"|(?:\d+[.、]\s*)?问题理解"
    r"|(?:\d+[.、]\s*)?解题思路"
    r"|(?:\d+[.、]\s*)?详细步骤"
    r"|(?:\d+[.、]\s*)?答案验证"
    r"|(?:\d+[.、]\s*)?最终答案"
    r"|结论[：:]"
    r"|解[：:]"
    r"|证明[：:]"
    r")"
)

_VISIBLE_CHARS_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_PUNCT_ONLY_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

_PLACEHOLDERS = {
    "result",
    "answer",
    "finalanswer",
    "final_answer",
    "placeholder",
    "todo",
    "none",
    "null",
    "n/a",
    "na",
    "明确的最终结果",
    "最终结果",
    "待求",
    "占位",
    "无法确定",
    "不能确定",
    "未知",
}


def strip_cot_prefix(text: str) -> str:
    """Remove a leading CoT preamble when a real content marker follows it."""
    value = text or ""
    if not value.strip():
        return ""
    if not _COT_PREFIX_RE.search(value[:300]):
        return value.strip()

    prefix = _COT_PREFIX_RE.search(value[:300])
    prefix_end = prefix.end() if prefix else 0
    for match in _CONTENT_START_RE.finditer(value):
        if match.start() >= prefix_end:
            return value[match.start():].strip()
    return ""


def contains_cot_marker(text: str) -> bool:
    return bool(_COT_PREFIX_RE.search((text or "")[:300]))


def _normalize_placeholder_text(text: str) -> str:
    value = strip_cot_prefix(text or "")
    value = value.strip().strip("`").strip("$").strip()
    value = value.strip("[]【】()（）{}<>")
    value = re.sub(r"\s+", "", value).lower()
    return value.replace("-", "").replace("_", "")


def is_placeholder_answer(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    normalized = _normalize_placeholder_text(value)
    if not normalized:
        return True
    if normalized in {item.replace("_", "").replace("-", "") for item in _PLACEHOLDERS}:
        return True
    if _PUNCT_ONLY_RE.match(value) and not _VISIBLE_CHARS_RE.search(value):
        return True
    return False
