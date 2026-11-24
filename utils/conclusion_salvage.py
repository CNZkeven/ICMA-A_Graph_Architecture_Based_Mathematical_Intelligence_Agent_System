"""Recover a conclusion from a response that never wrote its Markdown headers.

`intern-s2-preview-397b` bills `reasoning_content` against `max_tokens`, so on a hard
problem it can spend the whole budget thinking. The observed failure is worse than
plain truncation: the model reasons *to a conclusion* in prose and then runs out of
room before emitting `## 最终答案`, so a strictly header-driven parser scores the
attempt as producing nothing.

Measured on an ISL-level game-theory problem (2026-07-29, three runs): reasoning
returned 41,424 / 42,810 / 43,055 characters and the parser extracted zero sections
every time. Two of the three runs then finished with `解题过程中出现错误` after
1072s and 1165s — the model had done the work and the pipeline discarded it.

This is deliberately a *last resort*, used only when header parsing yields no answer:
  * it requires an explicit conclusion marker (因此/所以/综上/故/答案是), so ordinary
    exploratory prose ("我们先试 a=1") is never mistaken for a result;
  * it takes the LAST such sentence, because a long chain of reasoning revisits
    intermediate claims before settling;
  * every candidate passes the same placeholder/fragment gate as a parsed answer, so
    a salvaged value can never be lower quality than one that came from a header.

A salvaged answer is marked in the trace (`answer_source="salvaged_prose"`) so a
judge can tell it apart from a properly formatted one.
"""

from __future__ import annotations

import re

#: A sentence that announces a result rather than exploring toward one. `因此`/`所以`
#: /`综上` are the reliable Chinese conclusion markers; `答案` covers explicit
#: self-labelling. Requiring one of these is what keeps exploratory prose out.
_CONCLUSION_LEAD_RE = re.compile(
    r"(?:因此|所以|综上|故此|故\s|由此可(?:知|得)|答案(?:是|为)|结论(?:是|为|[：:])"
    r"|可以证明|我们(?:得到|证明了)|最终(?:答案|结论))")

#: A conclusion has to carry information: an equation, a number, a LaTeX quantity, or
#: an explicit qualitative verdict. Pure connective prose ("因此我们继续") is dropped.
_PAYLOAD_RE = re.compile(
    r"[=<>]|\d|\\frac|\\sqrt|\\lambda|\\pi|π|必胜|收敛|发散|成立|不成立|存在|唯一"
    r"|拒绝|接受|最大|最小|充分|必要")

#: Sentence boundaries. Newlines count, and so does the full stop, but a decimal
#: point must not split `0.5`, hence the lookarounds on the ASCII period.
_SENTENCE_SPLIT_RE = re.compile(r"(?:[。；;!?！？\n]|(?<!\d)\.(?!\d))+")

#: 试探排除句：模型在推导中否掉一族候选构造/量级，然后继续找真正的答案。
#: 2026-08-19 评委报告 idx 8：`所以任何 $O(1/n)$ 的收敛都不行` 带着结论词 `所以`
#: 与载荷词 `收敛`，旧闸门逐条放行，于是这句中间思考被当作最终答案出厂（标准
#: 答案是 603729）。这类句子的共同结构是「全称量词 + 否定判决」——它排除了一族
#: 候选，却没有说出所求的值，因此永远不是结论。
_TRIAL_REJECTION_RE = re.compile(
    r"(?:任何|所有|任意|每(?:个|一)|凡是?|一切)[^，。；\n]{0,40}"
    r"(?:都|均|皆)?\s*(?:不行|行不通|不可能|做不到|办不到|不满足|不成立|"
    r"不奏效|无法(?:满足|做到|实现|达到)|达不到|不够|失败)"
    r"|\b(?:any|every|all|no)\b[^,.;\n]{0,40}\b"
    r"(?:fails?|does\s*n[o']t\s+work|do\s+not\s+work|cannot\s+work|"
    r"is\s+impossible|are\s+impossible|won'?t\s+work)\b",
    re.IGNORECASE,
)

#: 模型给结论贴的显式标签。带标签的排除句是真的把"排除"当成了答案
#: （"答案是：不存在这样的 n"），此时不再按试探排除处理。
_ANSWER_LABEL_RE = re.compile(
    r"(?:最终)?答案\s*(?:是|为|[：:])|所求(?:为|是|的)|结论\s*(?:是|为|[：:])|"
    r"\bthe\s+answer\s+is\b",
    re.IGNORECASE,
)

#: Long enough to be a claim, short enough to be an answer rather than a paragraph.
_MIN_CHARS = 6
_MAX_CHARS = 400

#: How many trailing sentences to consider. The conclusion of a chain of reasoning is
#: at its end; scanning the whole 42k transcript would invite mid-derivation claims.
_TAIL_SENTENCES = 40


def _sentences(text: str) -> list[str]:
    return [s.strip().strip("*# 　") for s in _SENTENCE_SPLIT_RE.split(text or "")
            if s and s.strip()]


def salvage_conclusion(response: str) -> str:
    """Best conclusion sentence in `response`, or '' when there is none.

    Only for use when header parsing already failed — see the module docstring.
    """
    from utils.answer_cleanliness import is_noise_answer, looks_derivation_fragment
    from utils.answer_extractor import looks_incomplete_answer, looks_like_latex_fragment
    from utils.cot_stripper import is_placeholder_answer

    text = response or ""
    if not text.strip():
        return ""

    candidates = _sentences(text)[-_TAIL_SENTENCES:]
    # Last first: a long derivation restates intermediate results before concluding.
    for sentence in reversed(candidates):
        if not (_MIN_CHARS <= len(sentence) <= _MAX_CHARS):
            continue
        if not _CONCLUSION_LEAD_RE.search(sentence):
            continue
        if not _PAYLOAD_RE.search(sentence):
            continue
        if is_placeholder_answer(sentence) or looks_like_latex_fragment(sentence) \
                or looks_incomplete_answer(sentence):
            continue
        # 洁净度门（评委报告模式 B）：带结论词但实为自我怀疑/元叙述的句子
        # （"所以哪里出错了？"）不得作为答案输出。
        if is_noise_answer(sentence):
            continue
        # 对账门（2026-08-10 评委建议 4）：条件对/裸符号和/true-false 试错
        # 记录这类推导现场碎片，带结论词也不得出厂。
        if looks_derivation_fragment(sentence):
            continue
        # 试探排除门（2026-08-19 评委报告 idx 8）：「任何 X 都不行」排除的是一族
        # 候选，不是所求的值；除非模型自己把它标成了答案，否则继续往前找。
        if _TRIAL_REJECTION_RE.search(sentence) and not _ANSWER_LABEL_RE.search(sentence):
            continue
        return sentence
    return ""
