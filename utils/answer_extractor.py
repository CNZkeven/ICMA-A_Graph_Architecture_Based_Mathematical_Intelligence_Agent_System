import re


class AnswerExtractor:
    @staticmethod
    def normalize(answer: str) -> str:
        if not answer:
            return ""
        s = answer.strip().strip("。.").strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def extract_from_text(text: str):
        if not text:
            return None
        m = re.search(r"最终答案[：:]\s*(.+?)(?:\n|$)", text)
        return m.group(1).strip() if m else None


# 明显"话说一半"的结尾：冒号/逗号/运算符/引导语，后面本应还有内容
_DANGLING_TAIL_RE = re.compile(r"(?:[：:，,、;；=+\-*/^]|如下|分别为|分别是|包括|即)\s*$")


def looks_incomplete_answer(answer: str) -> bool:
    """判断答案是否为明显不完整的碎片（截断/引导语/未闭合括号）。

    评委报告主要问题 2/3：'(Matrix([' 与 'Z_18 的所有子群如下：' 这类碎片
    曾直接污染 final_response。此检测用于 cross_validator 与 formatter 的回退门。
    """
    s = (answer or "").strip()
    if not s:
        return True
    # 括号总开数 > 总闭数 → 截断（开闭跨类型合计，兼容半开区间 [0, 1) 记法）
    opens = sum(s.count(ch) for ch in "([{")
    closes = sum(s.count(ch) for ch in ")]}")
    if opens > closes:
        return True
    if _DANGLING_TAIL_RE.search(s):
        return True
    return False


def looks_like_latex_fragment(text: str) -> bool:
    """判断文本是否为被截头的 LaTeX 残片（如 '2 \\\\ 0, & n \\neq 2 \\end{cases} $$'）。

    特征：\\end{env} 无配对 \\begin{env}；$$ 未配对；对齐符 & 出现在任何环境之外。
    """
    s = (text or "").strip()
    if not s:
        return False
    for env in set(re.findall(r"\\end\{(\w+\*?)\}", s)):
        if ("\\begin{%s}" % env) not in s:
            return True
    for env in set(re.findall(r"\\begin\{(\w+\*?)\}", s)):
        if ("\\end{%s}" % env) not in s:
            return True
    if s.count("$$") % 2 == 1:
        return True
    if "&" in s and "\\begin" not in s and "\\&" not in s:
        return True
    return False


_ENUM_MARK_RE = re.compile(r"[（(]\s*[1-9一二三四五]\s*[）)]|[①②③④⑤]|第[一二三四五]问")
_MULTI_ASK_RE = re.compile(
    r"(?:求|计算|写出|给出|确定)[^。；;]*?(?:，?\s*并|及|以及|、并)[^。；;]*?(?:求|计算|证明|判断|讨论|写出|给出|确定)"
)


def is_multi_part_problem(problem: str) -> bool:
    """检测题目是否包含多个问项（评委报告问题 4：多问只答一问，如 idx=172）。"""
    p = (problem or "").strip()
    if not p:
        return False
    if len(_ENUM_MARK_RE.findall(p)) >= 2:
        return True
    if "分别" in p and re.search(r"求|计算|写出|给出|证明|讨论", p):
        return True
    if _MULTI_ASK_RE.search(p):
        return True
    return False
