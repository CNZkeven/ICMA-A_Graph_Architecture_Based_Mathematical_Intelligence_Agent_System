"""题型画像与客观题快速路径。

题面通常不会显式携带 ``type`` 字段，且官方题集同时包含证明、计算、选择、
判断和填空。把题型识别放在一个小的、无副作用的模块里，避免每个节点各自用
不一致的关键词判断，尤其避免把“判断题”误当成证明题。
"""

from __future__ import annotations

import re
from typing import Any, Mapping


QUESTION_MODES = frozenset({"proof", "computation", "choice", "true_false", "fill"})
OBJECTIVE_MODES = frozenset({"choice", "true_false", "fill"})

_TYPE_TO_MODE = {
    "计算": "computation", "计算题": "computation", "computation": "computation",
    "proof": "proof", "证明": "proof", "证明题": "proof",
    "choice": "choice", "选择": "choice", "选择题": "choice", "单选": "choice", "多选": "choice",
    "fill": "fill", "填空": "fill", "填空题": "fill",
    "true_false": "true_false", "判断": "true_false", "判断题": "true_false",
}

# Option markers occur both at line starts and inline in the official JSONL set.
_OPTION_RE = re.compile(r"(?<![A-Za-z0-9])[A-EＡ-Ｅ]\s*[.．、:：)）]")
_OPTION_PAREN_RE = re.compile(r"(?<![A-Za-z0-9])[（(]\s*[A-EＡ-Ｅ]\s*[）)]")
_TRUE_FALSE_RE = re.compile(
    r"(?:^|[\n。；;])\s*(?:判断|正误|对错)\s*(?:题)?\s*[:：]?|"
    r"(?:判断题|正误题|对错题|正确与否|命题是否正确|说法是否正确|"
    r"(?:判断|判定)(?:下列|该|此)?(?:命题|说法|结论))",
    re.I,
)
_FILL_RE = re.compile(
    r"填空|填“|填\"|填入|_{3,}|＿{2,}|"
    r"[(（]\s*(?:\\(?:quad|qquad|square|Box)|\\\s*)?[)）]"
)
_PROOF_RE = re.compile(
    r"(?:^|[\n])\s*(?:证明|试证)|(?:^|[\n])\s*(?:prove|show)\b|"
    r"证明题|(?<!or )\b(?:prove|show|demonstrate|establish|justify)\s+that\b|"
    r"\b(?:prove|show|demonstrate|establish|justify)\b|"
    # 句中出现式证明/反例指令（2026-10-01 决赛 idx 48：题面「判断对错．正确的请给出
    # 证明，错误的请举出反例」——"证明"不在行首，旧规则判成 computation，协调器据此
    # 把 3400 字证明正文折叠成一句结论，违反 评委.md §2.5「推导题须保留关键步骤」）。
    # 只收明确的作答要求，不收泛述"证明/定理"字样，避免把计算题误判为证明题。
    r"请给出证明|给出证明|并证明|加以证明|请证明|证明之|举出反例|给出反例|举反例",
    re.I,
)


def mode_from_metadata(metadata: Mapping[str, Any] | None) -> str:
    """Read an evaluator-provided type without making it a runtime dependency.

    The public competition input normally contains only ``problem`` and ``idx``.
    Some harnesses also provide ``type``/``question_mode``; honoring those fields
    removes avoidable ambiguity while the text-only path remains unchanged.
    """
    if not isinstance(metadata, Mapping):
        return ""
    for key in ("question_mode", "type", "question_type", "problem_type"):
        value = str(metadata.get(key, "") or "").strip().lower()
        if value in QUESTION_MODES:
            return value
        if value in _TYPE_TO_MODE:
            return _TYPE_TO_MODE[value]
    return ""


def classify_question_mode(problem: str) -> str:
    """Return one stable mode name from raw problem text.

    Objective markers are checked before proof words: a multiple-choice stem may
    contain “判断下列命题” but should still use the short objective path.
    """

    text = str(problem or "").strip()
    if not text:
        return "computation"
    if _TRUE_FALSE_RE.search(text) or re.search(r"判断\s*(?:题)?\s*[：:]", text):
        return "true_false"
    # Official questions often use ``（）`` after the stem and place options
    # inline (``A.均值B.方差``).  Option detection must therefore precede the
    # generic blank marker; otherwise almost every objective item becomes fill-in.
    option_count = len(_OPTION_RE.findall(text)) + len(_OPTION_PAREN_RE.findall(text))
    if option_count >= 2 or re.search(r"(?:选择题|单选|多选|下列.*(?:正确|错误|是|为).*选)", text, re.I):
        return "choice"
    if _FILL_RE.search(text) or re.search(
        r"(?:填空题|fill\s*in|blank|指标是|分别为\s*$)", text, re.I
    ):
        return "fill"
    if _PROOF_RE.search(text):
        return "proof"
    return "computation"


_TF_ITEM_NUM_RE = re.compile(
    r"\\item\s*\[?\s*\(\s*([1-9])\s*\)|（\s*([1-9])\s*）|\(\s*([1-9])\s*\)"
)
_TF_BULLET_RE = re.compile(r"(?m)^\s*[-•・－]\s+\S")
_TF_BLANK_RE = re.compile(r"_{3,}")


def judgement_item_count(problem: str) -> int:
    """判断题内"并列小项"的个数：编号 (1)(2)…、行首列表项、下划线空位。

    ≥2 即视为"多条判断"，必须逐条作答。2026-10-02 决赛 idx 13（五连判断题）
    曾被单条契约折叠成"错误"一条；idx 32 是三条下划线空位的形态。
    """
    text = str(problem or "")
    numbers = {g for m in _TF_ITEM_NUM_RE.finditer(text) for g in m.groups() if g}
    return max(
        len(numbers),
        len(_TF_BULLET_RE.findall(text)),
        len(_TF_BLANK_RE.findall(text)),
    )


def is_objective_mode(mode: str) -> bool:
    return mode in OBJECTIVE_MODES


# ---- 题面结构探测（评委建议 6/7/8） ----

_BLANK_MARK_RE = re.compile(
    r"_{3,}|＿{2,}|—{3,}|"
    r"[（(]\s*(?:(?:\\(?:quad|qquad|square|Box))|\\)?\s*[）)]"
)

_EXPLICIT_BLANK_LABEL_RE = re.compile(r"(?:空|blank)\s*\d+\s*[：:]", re.I)

#: "求所有/找出全部" 型题：解空间可能含多个分支，单一构造不可作答。
_ALL_SOLUTIONS_RE = re.compile(
    r"(?i)(?:^|[\n:;。])\s*(?:find|determine)\s+all\b|"
    r"\b(?:find|determine)\s+all\s+(?:possible\s+)?(?:values?|solutions?|roots?)\b|"
    r"\bdetermine\s+every\s+(?:possible\s+)?(?:value|solution|root)s?\b|"
    r"\ball\s+possible\s+(?:values?|solutions?|roots?)\b|"
    r"\ball\s+solutions?\b|\ball\s+roots?\b|"
    r"所有可能|全部(?:的)?(?:解|值|情况|可能)|求所有|找出所有|确定所有"
)

_ALL_OPTIONS_RE = re.compile(
    r"(?i)\b(?:select|choose|mark|identify)\s+all\s+(?:the\s+)?options?\b|"
    r"\bwhich\s+(?:of\s+the\s+)?options?\b[^.\n]{0,80}\b(?:correct|true|false|apply)\b|"
    r"所有(?:正确|错误)?选项|多选题|不定项选择"
)

_EXPLICIT_NORM_RE = re.compile(
    r"(?i)(?:(?P<ascii>[123∞])\s*[- ]?norm\b|"
    r"norm\s*[_-]?\s*(?P<suffix>[123∞])\b|"
    r"范数\s*(?:为|是|下|取)?\s*(?P<cjk>[123∞])|"
    r"\|\|[^\n|]{1,80}\|\|\s*[_^]\s*\{?\s*(?P<latex>[123∞])\s*\}?)"
)

#: 极值/最优题：需要构造与上/下界双向闭环，且构造须与竞争构造比较。
_EXTREMAL_RE = re.compile(
    r"(?i)最大|最小|最多|最少|至多|至少|最优|最短|最长|极值|"
    r"\b(?:maximum|minimum|maximal|minimal|largest|smallest|greatest|least|"
    r"at\s+most|at\s+least|optimal)\b"
)

#: 值域为二元域/模 m 结构的函数声明（f: Q → F_2 等）。
_MOD_CODOMAIN_RE = re.compile(
    r"(?:\\rightarrow|\\to|→|->)\s*\\?(?:mathbb\s*\{?\s*F\s*\}?_\{?2\}?|F_?2\b|"
    r"GF\(2\)|\\?mathbb\s*\{?\s*Z\s*\}?\s*/\s*2)"
)

#: 同名函数值相加（f(a)+f(b)）——和落在函数值域所在的代数结构里。
_FUNC_SUM_RE = re.compile(r"([a-zA-Z])\s*\([^()]{1,60}\)\s*\+\s*\1\s*\(")

#: 开区间/严格不等式限定的取值范围。"between a and b" 是竞赛惯用的开区间表述；
#: 由 ≤ 推出的界在严格范围内取不到，临界值必须逐一核验可达性。
_OPEN_BOUND_RE = re.compile(
    r"(?i)\bbetween\b[^,.;]{1,40}\band\b|strictly\s+(?:between|less|greater)|"
    r"开区间|严格(?:小于|大于|介于|单调)|"
    # |x| < 1 型严格不等式（排除 ≤/\le/\leq）
    r"(?<![\\])<(?!=)"
)

#: "第 k 大/第 k 小/第 k 个" 型序号问法：通项与序号极易差一。
#: 序数词与数字之间常隔着 LaTeX 包裹（``$9$th``、``{10}th``），必须允许。
_ORDINAL_RANK_RE = re.compile(
    r"(?i)\b\d+\s*[$}\s]*(?:st|nd|rd|th)\b[^.\n]{0,40}"
    r"\b(?:largest|smallest|greatest|least|biggest|term|element|number|value)\b|"
    r"第\s*\$?\s*[0-9一二三四五六七八九十k]+\s*\$?\s*(?:大|小|个|项|位)"
)

#: 可行性 + 计数的双问项（"是否可能…若可能，最少需要多少"）。
_FEASIBILITY_RE = re.compile(
    r"(?i)\bis it possible\b|\bcan (?:we|you|one|it)\b|\bdoes there exist\b|"
    r"是否(?:可能|存在|能)|能否|可否"
)
_COUNT_ASK_RE = re.compile(
    r"(?i)\b(?:minimum|maximum|smallest|largest|fewest|greatest)\s+number\b|"
    r"\bhow many\b|最少(?:需要)?(?:多少|几)|最多(?:能|可以)?(?:多少|几)|"
    r"需要多少|数量是多少"
)


def has_open_interval_bound(problem: str) -> bool:
    """题面用开区间/严格不等式限定取值（端点不可达，临界值需 +1 核验）。"""
    return bool(_OPEN_BOUND_RE.search(str(problem or "")))


def asks_ordinal_rank(problem: str) -> bool:
    """题面问"第 k 大/小/个"（序号与通项必须逐项对齐核验）。"""
    return bool(_ORDINAL_RANK_RE.search(str(problem or "")))


def asks_feasibility_then_count(problem: str) -> bool:
    """题面同时问"是否可能"与"最少/最多需要多少"（不可能性论证须先证伪）。"""
    text = str(problem or "")
    return bool(_FEASIBILITY_RE.search(text)) and bool(_COUNT_ASK_RE.search(text))


def count_blanks(problem: str) -> int:
    """题面可见空位数（填空题输出契约用）。检测不到显式空位记号时返回 0。

    评委报告 idx 86：三空只答零空、置信 0.78 直接放行——缺的就是这个计数。
    """
    text = str(problem or "")
    visible = len(_BLANK_MARK_RE.findall(text))
    labelled = len({
        int(value)
        for value in re.findall(r"(?:空|blank)\s*(\d+)\s*[：:]", text, re.I)
    })
    return max(visible, labelled)


#: 括号对，用于"顶层分隔符"切分。空位之间的分隔符出现在括号外；出现在括号内的
#: 逗号属于答案自身的一部分（``Q(∜5, ζ₈)``、``\mathbb{Q}(\sqrt[4]{5},\zeta_8)``）。
_BRACKET_OPENERS = "([{（【｛"
_BRACKET_CLOSERS = ")]}）】｝"


def _split_top_level(text: str, separators: str) -> list[str]:
    """按 ``separators`` 切分，但忽略落在任何括号内部的分隔符。"""
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    for char in str(text or ""):
        if char in _BRACKET_OPENERS:
            depth += 1
        elif char in _BRACKET_CLOSERS:
            depth = max(0, depth - 1)
        elif depth == 0 and char in separators:
            parts.append("".join(buffer))
            buffer = []
            continue
        buffer.append(char)
    parts.append("".join(buffer))
    return [part.strip() for part in parts if part.strip()]


def fill_part_count(answer: str) -> int:
    """填空答案里的分项数（优先按显式空位标签，再按顶层分隔符切分）。

    分号/换行是首选分隔符，但模型经常把多个空写成一行并用逗号分隔。2026-08-19
    评委报告 idx 86 的复盘：模型三个空全答对了（``分裂域为 Q(∜5, ζ₈)，扩张次数为
    16，是 Galois 扩张``），只因分隔符是中文逗号而被计成 1 项，完整性闸门把这条
    正确答案当作残缺答案清空，最终输出退化成"无法确定"。逗号必须同样算分隔符；
    而 ``Q(∜5, ζ₈)`` 里的逗号在括号内，按顶层切分天然不会被误切。
    """
    text = str(answer or "").strip()
    if not text:
        return 0
    labels = {
        int(value)
        for value in re.findall(r"(?:空|blank)\s*(\d+)\s*[：:]", text, re.I)
    }
    if labels:
        return len(labels)
    strong = _split_top_level(text, "；;\n")
    if len(strong) > 1:
        return len(strong)
    return len(_split_top_level(text, "；;\n，,")) or 1


def recover_complete_fill_answer(response: str, problem: str) -> str:
    """Recover the latest explicit fill answer that covers every visible blank.

    Long objective responses often contain an early one-word answer followed by a
    complete labelled answer in the rationale.  The latter must win, but only when
    its number of labelled/separated parts covers the question's blank count.
    """
    expected = count_blanks(problem)
    if expected <= 1:
        return ""
    text = str(response or "")

    # The compressed recovery path uses the normal four-section response schema.
    # Its final-answer payload is commonly one labelled blank per line, which the
    # generic answer distiller must not collapse to the last line before the fill
    # contract sees it.  Only retain explicitly numbered fill lines: a prose
    # paragraph in a final-answer section is not evidence of complete coverage.
    sections = re.findall(
        r"(?ims)^\s*#{1,6}\s*(?:最终答案|答案|final\s+answer)\s*$\s*"
        r"(.*?)(?=^\s*#{1,6}\s|\Z)",
        text,
    )
    for section in reversed(sections):
        lines = [
            line.strip().lstrip("-*• ").strip()
            for line in section.splitlines()
            if re.search(r"(?:空|blank)\s*\d+\s*[：:]", line, re.I)
        ]
        candidate = "；".join(lines)
        if fill_part_count(candidate) >= expected:
            return candidate

    candidates = re.findall(
        r"(?im)(?:^|[\n。；;])\s*(?:最终答案|答案|answer)\s*[：:]\s*([^\n]+)",
        text,
    )
    for candidate in reversed(candidates):
        candidate = candidate.strip().strip("`*_ ")
        if fill_part_count(candidate) >= expected:
            return candidate
    return ""


def requires_all_solutions(problem: str) -> bool:
    """题面是否要求"所有解/全部值"（须触发解空间枚举，不得单支作答）。"""
    return bool(_ALL_SOLUTIONS_RE.search(str(problem or "")))


def asks_all_options(problem: str) -> bool:
    """题面是否明确要求逐项选择所有正确/适用选项。"""
    text = str(problem or "")
    if not _ALL_OPTIONS_RE.search(text):
        return False
    # Avoid treating a discussion sentence as an instruction when it has no
    # option markers at all.  Chinese explicit multi-select wording is already
    # authoritative and does not require ASCII markers.
    if re.search(r"多选题|不定项选择|所有(?:正确|错误)?选项", text):
        return True
    markers = len(_OPTION_RE.findall(text)) + len(_OPTION_PAREN_RE.findall(text))
    return markers >= 2


def explicit_norms(problem: str) -> set[str]:
    """Return norm indices explicitly named by the problem statement."""
    norms: set[str] = set()
    for match in _EXPLICIT_NORM_RE.finditer(str(problem or "")):
        value = next(
            (match.group(name) for name in ("ascii", "suffix", "cjk", "latex")
             if match.group(name)),
            "",
        )
        if value:
            norms.add(value)
    return norms


def is_extremal_problem(problem: str) -> bool:
    """题面是否为极值/最优化问题（构造须与竞争构造对照，防自洽陷阱）。"""
    return bool(_EXTREMAL_RE.search(str(problem or "")))


def has_mod2_valued_sum(problem: str) -> bool:
    """函数值域为 F₂/模 2 且所求为函数值之和（和必须在该结构内聚合）。

    评委报告 idx 7：六个 f 值全对，最后按整数相加得 3；题面 f:Q→F₂，
    和应在 F₂ 中为 1。通用口径清单不足以扭转，这里做题面条件触发的定向注入。
    """
    text = str(problem or "")
    return bool(_MOD_CODOMAIN_RE.search(text)) and bool(_FUNC_SUM_RE.search(text))


def mode_instruction(mode: str, problem: str = "") -> str:
    """Prompt suffix shared by reasoning/Python nodes.

    The instruction deliberately asks for the exact deliverable, not a guessed
    answer. It keeps objective questions concise while preserving the full
    four-section contract for the parser. ``problem`` is optional so legacy
    callers keep the single-verdict wording; multi-item true/false questions
    need it to demand one verdict per item.
    """

    if mode == "choice":
        return (
            "\n\n[题型画像：选择题快速路径]\n"
            "先判断单选还是多选：题面注明'多选/不定项'或多个选项各自独立成立时按多选，"
            "否则倾向单选。逐项按教材定义的**严格措辞**核对：概念被替换、方向说反、"
            "表述不精确的选项一律不选，宁缺勿滥。最后在'最终答案'中明确写出全部正确选项"
            "（如 A、C），不要把选项说明误当成答案。最多 3 个短步骤。"
        )
    if mode == "true_false":
        items = judgement_item_count(problem)
        if items >= 2:
            return (
                "\n\n[题型画像：判断题快速路径（多条并列）]\n"
                f"题面含 {items} 条并列判断：逐条核对并逐条给出判定，不要整体只给一个判断；"
                f"最终答案按题面顺序写全 {items} 条判定（有编号就带编号，如 (1) 正确；(2) 错误；"
                "……；无编号就按题面次序用分号分隔），只写一条视为未完成；"
                "判定词随题面措辞（题面问'是否成立'就写'成立/不成立'，问对错才写'正确/错误'）。"
            )
        return (
            "\n\n[题型画像：判断题快速路径]\n"
            "先给出‘正确’或‘错误/不正确’（题面以'是否成立'设问时写'成立/不成立'），再用一个反例或定理说明理由；"
            "不要把条件句当作最终判断。"
        )
    if mode == "fill":
        return (
            "\n\n[题型画像：填空题快速路径]\n"
            "按空位顺序逐项作答，答案行格式：空1: <结果>；空2: <结果>（只有一个空时直接给结果）。"
            "每个空必须填教材术语/具体数值/明确方向（如'有偏''低估或高估'），"
            "禁止填'不确定/视情况而定'；不要输出空位数量本身或任何操作说明，"
            "不要把一个空的同义解释拆成多个空，不输出与空位无关的长篇探索。"
        )
    if mode == "proof":
        return (
            "\n\n[题型画像：证明题]\n"
            "必须写出从假设到结论的关键蕴含链，至少列出两步，并检查边界条件/等号条件。"
        )
    return (
        "\n\n[题型画像：计算题]\n"
        "先列出题面要求的全部量，再计算并在最终答案中逐项标注；不要用一个中间量代替最终结论。"
        "若题目要求‘所有/确定集合/轨迹/最小或最大常数’，必须同时给出候选的必要性与充分性（或上界与构造）"
        "并核对边界，不能只报一个猜测值。"
    )


def structure_instruction(problem: str) -> str:
    """按题面结构追加的强制步骤（评委建议 6/7），推理与 Python 分支共用。

    与 mode_instruction 互补：mode 看题型，这里看题面语义（全部解/极值）。
    检测不到相应结构时返回 ""，不增加提示词负担。
    """
    parts: list[str] = []
    if has_mod2_valued_sum(problem):
        parts.append(
            "[值域聚合警告] 题面函数的值域是 F_2（模 2 的二元域）。所求的 f(⋯)+f(⋯)+… 是"
            " **F_2 中的加法**：先逐项求出各函数值（0 或 1），最后必须按模 2（异或）聚合，"
            "最终答案只能是 0 或 1；给出普通整数和（如 3）即错。"
        )
    if requires_all_solutions(problem):
        parts.append(
            "[全部解检查单] 本题要求给出**所有**解/值。禁止只验证一个已知解就作答："
            "必须先在小规模/截断版本上系统枚举解空间（列出发现的每一个解支），"
            "再证明再无其他分支；最终答案必须列出全部解支（含平凡支与例外支）。"
        )
    if is_extremal_problem(problem):
        parts.append(
            "[极值题对照检查] 本题求极值/最优。单一构造族内的自洽验证不构成最优性证明："
            "必须 (1) 用小规模精确解（暴力/DP）校准；(2) 至少比较两个结构不同的构造，"
            "取更优者；(3) 给出与构造值匹配的上界/下界论证。三者缺一即在验证点中声明未完成。"
        )
    # 以下三条针对 2026-08-10 复测轮的错因（评委建议 1/2/3/5）：口径与序号类错误
    # 都发生在"推理跑完了"之后，属于收尾核验缺失，成本极低但直接决定得分。
    if has_open_interval_bound(problem) and is_extremal_problem(problem):
        parts.append(
            "[端点可达性检查] 题面用开区间/严格不等式限定取值（如 between a and b、"
            "|S|<1）。由非严格不等式（≤）推出的界在严格范围内**取不到**："
            "推出 n ≥ K 之后必须对 n=K 显式写出**全部变量的具体数值或构造式**，"
            "并把它们逐条代回原式检验每一个等式与每一个严格不等号；"
            "禁止用“可以构造”“取 n=K 即可”一类断言代替检验。"
            "若检验发现只有让某个变量取到端点（或让某个严格不等式取等）才能成立，"
            "则 n=K 不可行，答案是 K+1，并在验证点中写出使等号无法成立的那一步。"
        )
    if asks_ordinal_rank(problem):
        parts.append(
            "[序号对齐检查] 本题问“第 k 大/第 k 小/第 k 个”。禁止直接写通项作答："
            "必须先显式列出 k=1、2、3 对应的前三个对象（写出具体表达式或数值），"
            "确认序号从 1 起数且与通项逐项对齐，再代入题面的 k；"
            "最终答案与该列表必须自洽（差一即错）。"
        )
    if asks_feasibility_then_count(problem):
        parts.append(
            "[不可能性论证的证伪要求] 本题先问可行性、再问所需数量。"
            "断言“不可能/不存在”之前必须完成两项证伪：(1) 在缩小规模的同类实例上"
            "（如把边长换成 3、5、7）实际构造或暴力搜索，只要小规模可行，"
            "不可能性论证即有误；(2) 复核不变量论证的每一步——同一形状/操作在染色或"
            "计数下的贡献若不唯一（例如一块拼图可覆盖 2 黑 1 白，也可覆盖 1 黑 2 白），"
            "总量整除性就**不构成**不可能性证明。若可行，最终答案必须给出具体数量，"
            "不得只回答“可能/不可能”。"
        )
    if not parts:
        return ""
    return "\n\n" + "\n".join(parts)


#: 约束跑遍一个无穷指标集（无穷序列、对所有正整数 m<n、n→∞ 的族）。这类题里
#: "最紧的那条约束"几乎从不落在前几项上，而模型习惯性地只验 m=1,n=2 就收工。
#: 2026-08-20 idx 8：正解 D=777²，模型取 y_n=M/n 后断言下确界在 (m,n)=(1,2) 处
#: 取到，得 3·777²/2；但同一表达式在 n=m+1、m→∞ 时趋于 0，构造根本不成立。
_INFINITE_INDEX_RE = re.compile(
    r"(?i)infinite\s+sequence|for\s+all\s+positive\s+integers|"
    r"for\s+every\s+(?:positive\s+)?integer|\bfor\s+all\s+n\b|"
    r"无穷(?:序列|数列|多)|对(?:所有|任意)(?:的)?正整数|每(?:个|一个)正整数|"
    r"x_?1\s*,\s*x_?2\s*,\s*\\?l?dots"
)


def has_infinite_index_family(problem: str) -> bool:
    """约束是否跑遍一个无穷指标集（极值自检必须验到 n→∞）。"""
    return bool(_INFINITE_INDEX_RE.search(str(problem or "")))


def answer_coverage_clause(problem: str) -> str:
    """压缩重试专用：必须保全的**答案覆盖度**（不是推导篇幅）。

    2026-08-10 复测轮 idx 14：截断后压缩重试的指令写着"沿最可行的一条路线、
    每章从简"，模型据此只报了三个可能值中的一个。压缩针对的是推导过程，
    答案该覆盖的分支一个都不能少——这句提醒随压缩指令一起发出。
    """
    clauses = []
    if requires_all_solutions(problem):
        clauses.append(
            "题目要求“所有可能的值/全部解”：压缩的是推导篇幅，不是解的个数——"
            "'## 最终答案' 必须把每一个解支都列出来（写成集合；英文 every/all "
            "possible 的每一支也必须保留），"
            "只报一支按缺解计零分"
        )
    if asks_all_options(problem):
        clauses.append(
            "[全选覆盖检查] 题面要求所有适用选项：先逐项判断 A、B、C…，"
            "最终答案必须保留完整集合；不得因首个正确项而提前停止。"
        )
    norms = explicit_norms(problem)
    if norms:
        clauses.append(
            "[范数口径检查] 题面明确指定范数 " + ", ".join(sorted(norms))
            + "，所有矩阵范数/条件数计算必须使用该范数；禁止套用未标注的默认 2-范数。"
        )
    if asks_all_options(problem):
        clauses.append(
            "题目要求选择所有适用选项：必须逐项核对并在最终答案列出完整选项集合，"
            "不能把一个正确选项当成单选答案"
        )
    if asks_feasibility_then_count(problem):
        clauses.append("题目同时问可行性与数量：最终答案必须给出具体数量，不能只答可能/不可能")
    if is_multi_ask(problem):
        clauses.append("题目含多个问项：每一问都要在最终答案中单独给出结果")
    if is_extremal_problem(problem):
        clauses.append(
            "题目求极值/最小规模：压缩的是推导篇幅，不是证明义务——"
            "必须同时给出匹配的下界或上界、达到该界的 explicit construction，"
            "并逐条核验最大（largest）/边界（boundary）要求；"
            "不得只由小样本或相似题类比公式，也不得把独立对象写成一次性符号来规避题面条件"
        )
        if has_infinite_index_family(problem):
            clauses.append(
                "约束跑遍无穷个指标：断言“最紧约束在某几项处取到”之前，必须先算出"
                "该表达式在指标趋于无穷（如 n=m+1, m→∞）时的极限——"
                "最紧的一条约束几乎从不落在前几项上；只验 m=1,n=2 就下结论是"
                "本项目已确认的失分模式（boundary at infinity）"
            )
    if not clauses:
        return ""
    return "\n**答案覆盖度要求（不受“从简”约束）**：" + "；".join(clauses) + "。"


def is_multi_ask(problem: str) -> bool:
    """题面是否有多个并列问项（复用答案抽取侧的多问项判定）。"""
    from utils.answer_extractor import is_multi_part_problem

    return is_multi_part_problem(problem)


_FULLWIDTH_TRANSLATION = str.maketrans("ＡＢＣＤＥａｂｃｄｅ", "ABCDEabcde")


def _strip_answer_wrappers(answer: str) -> str:
    text = str(answer or "").strip()
    # Prefer the last explicit label.  A response can quote an intermediate
    # ``答案：...`` and then commit to a second one; taking the last label avoids
    # treating the rationale as the submitted payload.
    labelled = list(re.finditer(
        r"(?:最终答案|答案|结论|选择)\s*[：:]\s*([^\n]+)", text, flags=re.I
    ))
    if labelled:
        text = labelled[-1].group(1).strip()
    else:
        text = re.sub(r"^\s*(?:最终答案|答案|结论|选择)\s*[：:]\s*", "", text, flags=re.I)
    # Keep the last boxed expression: an objective response may quote an option
    # in its rationale before committing to the answer line.
    boxed = re.findall(r"\\boxed\s*\{([^{}]*)\}", text, flags=re.S)
    if boxed:
        text = boxed[-1].strip()
    # The concise objective contract puts the evidence after a semicolon or a
    # dedicated ``依据/理由`` label.  Drop that suffix for choices/judgements;
    # fill answers keep internal semicolons because they can denote multiple blanks.
    text = re.split(r"(?:\n|[；;])\s*(?:依据|理由|说明)\s*[：:]", text, maxsplit=1)[0]
    return text.strip().strip("。．.;；\"'“”")


#: 判断词的归一化：肯定族映射为"正确"（成立保留原词，题面用的就是这个措辞）。
_TF_SINGLE_VERDICTS = {"正确": "正确", "对": "正确", "√": "正确", "成立": "成立"}
_TF_VERDICT_TOKEN = r"不成立|成立|不正确|错误|正确|不对|对|错|×|√"


def _verdict_token(value: str) -> str:
    cleaned = str(value or "").strip().rstrip("。.．,，")
    return _TF_SINGLE_VERDICTS.get(cleaned, "错误")


def _extract_multi_verdicts(text: str) -> str:
    """识别"多条判断"答案形态；识别不了就返回 ""（交回单条旧路径）。

    形态一（编号）：``(1) 错误；(2) 正确；……``（编号≥2 且互异，按编号排序）；
    形态二（纯序列）：整段只有判定词与分号/空白，形如 ``成立；成立；成立``。
    两条规则都刻意收得很窄：单条答案里"结论词+解释里再出现一个判定词"
    （如"错误，因为……正确……"）不会命中，仍按旧逻辑取第一个判定词。
    """
    pair_re = re.compile(
        r"\(?\s*([1-9])\s*[\)）]\s*[:：、.．]?\s*(" + _TF_VERDICT_TOKEN + r")"
    )
    pairs: dict[str, str] = {}
    for m in pair_re.finditer(text):
        if m.group(1) not in pairs:
            pairs[m.group(1)] = _verdict_token(m.group(2))
    if len(pairs) >= 2:
        ordered = sorted(pairs, key=int)
        return "；".join(f"({n}) {pairs[n]}" for n in ordered)
    tokens = [t for t in re.split(r"[；;\s]+", text) if t]
    if len(tokens) >= 2 and all(
        re.fullmatch(r"(?:" + _TF_VERDICT_TOKEN + r")[。.．]?", t) for t in tokens
    ):
        return "；".join(_verdict_token(t) for t in tokens)
    return ""


_TF_NUMBER_MARKER_RE = re.compile(r"\(?\s*([1-9])\s*[\)）]")


def _looks_multi_verdict(text: str) -> bool:
    """形似"多条判断"（出现 ≥2 个编号条目），用于把条目不全的答卷判为不可用。"""
    return len({m.group(1) for m in _TF_NUMBER_MARKER_RE.finditer(text)}) >= 2


def normalize_objective_answer(answer: str, mode: str) -> str:
    """Canonicalize a short choice/judgement/fill answer without solving it.

    This is intentionally a formatting operation.  It removes wrappers and,
    for choices, retains option letters in their appearance order; it never
    derives a missing option from the explanation.
    """
    text = _strip_answer_wrappers(answer)
    if not text:
        return ""
    if mode == "choice":
        text = text.translate(_FULLWIDTH_TRANSLATION)
        # 连写形态（"BD"、"\boxed{BD}" 剥壳后）：字母间无分隔符时逐字母展开。
        # 单词边界正则对连写必然失配（B 的后瞻撞上 D），评委报告 idx 87 的
        # \boxed{BD} 曾因此归一化为空。仅限 2-5 个纯 A-E 字母的短串，避免把
        # 普通英文单词错当选项串。
        if re.fullmatch(r"[A-E](?:\s*[A-E]){1,4}", text, flags=re.I):
            letters = re.findall(r"[A-E]", text, flags=re.I)
        else:
            letters = re.findall(r"(?<![A-Za-z0-9])[A-E](?![A-Za-z0-9])", text, flags=re.I)
        ordered = []
        for letter in letters:
            upper = letter.upper()
            if upper not in ordered:
                ordered.append(upper)
        return "、".join(ordered)
    if mode == "true_false":
        multi = _extract_multi_verdicts(text)
        if multi:
            return multi
        if _looks_multi_verdict(text):
            # 形似多条判断但条目不全（如只写了第 1 条）：判为不可用（返回空），
            # 交重试路径补全 —— 不能把"第 1 条判定"当作整题答案提交
            # （2026-10-02 决赛 idx 13 的折叠教训）。
            return ""
        # Prefer the two-character negative phrase before the bare ``正确``.
        matches = list(re.finditer(_TF_VERDICT_TOKEN, text, flags=re.I))
        if not matches:
            return ""
        # The first judgement token is the answer; later occurrences usually
        # belong to the explanatory sentence (e.g. ``错误，因为...正确``).
        return _verdict_token(matches[0].group(0))
    return text


_FILL_TEMPLATE_SLOT_RE = re.compile(
    r"(?:空|blank)\s*\d+\s*[:：]\s*(?:\.{2,}|…+|<[^>\n]{0,80}>)",
    re.I,
)
_FILL_FORMAT_META_RE = re.compile(r"\b(?:semicolon\s+separated|format(?:ting)?\s+constraints?)\b", re.I)


def objective_answer_is_usable(answer: str, mode: str) -> bool:
    normalized = normalize_objective_answer(answer, mode)
    if not normalized:
        return False
    if mode == "choice":
        return bool(re.fullmatch(r"[A-E](?:、[A-E])*", normalized))
    if mode == "true_false":
        if normalized in {"正确", "错误", "成立"}:
            return True
        # 多条判断：``(1) 错误；(2) 正确；…`` 或 ``成立；成立；成立``——
        # 每一节都必须含判定词，否则视为半成品交由重试路径处理。
        parts = [p for p in re.split(r"[；;]", normalized) if p.strip()]
        return bool(parts) and len(parts) >= 2 and all(
            re.search(_TF_VERDICT_TOKEN, p) for p in parts
        )
    if mode == "fill":
        # A section-less long CoT can be mistaken for a fill answer by the prose
        # salvage fallback (e.g. an English sentence ending in a formula).  Reject
        # obvious narrative so the bounded prefilled retry gets a chance to emit
        # the requested blank payload.  Mathematical expressions and multi-blank
        # textbook phrases remain allowed.
        from utils.answer_cleanliness import is_noise_answer
        from utils.answer_extractor import looks_incomplete_answer

        if looks_incomplete_answer(normalized):
            return False
        # A prompt-format example can satisfy the blank-count contract while
        # containing no mathematical value at all (e.g. ``空1: ...``).  Treat
        # that as a rejected placeholder so objective retries can produce a
        # real payload instead of submitting the instruction verbatim.
        if _FILL_TEMPLATE_SLOT_RE.search(normalized) or _FILL_FORMAT_META_RE.search(normalized):
            return False
        # 元指令回显 / 思维碎片（评委报告 idx 105 "Count blanks: 1"）与
        # 非答案术语（idx 108 "不确定"）都不是可提交的填空内容。
        if is_noise_answer(normalized):
            return False
        if re.fullmatch(r"不确定|不一定|视情况(?:而定)?|无法判断", normalized):
            return False
        # 带"空N:"前缀的不确定类回答（2026-08-09 冒烟回归 idx 108："空1: 不确定"）：
        # 逐空剥掉前缀后检查；全部空位都是不承诺答案时整体拒收，混合时保留部分得分。
        parts = [re.sub(r"^\s*空\s*\d+\s*[:：]\s*", "", part).strip()
                 for part in re.split(r"[；;]", normalized)]
        parts = [part for part in parts if part]
        if parts and all(
            re.fullmatch(r"不确定|不一定|视情况(?:而定)?|无法判断|无法确定|不能确定", part)
            for part in parts
        ):
            return False
        # 填空答案是术语/数值/短表达式；成段英文散句只可能是泄漏的思考文本
        # （2026-08-09 冒烟回归：idx 108 的答题纠结独白曾以 fill 答案形态出厂）。
        # 统计自然语言词时排除 LaTex 命令名；否则 ``\boxed{\text{...}
        # \mathbb{Q}(\sqrt{...},\zeta_8)}`` 会被命令名堆高到错误拒绝。
        prose_for_word_count = re.sub(r"\\[A-Za-z]+", "", normalized)
        if len(re.findall(r"[A-Za-z]{2,}", prose_for_word_count)) > 6:
            return False
        if len(normalized) > 420 or "\n" in normalized or "?" in normalized or "？" in normalized:
            return False
        if re.search(
            r"\b(?:let's|consider|under|because|therefore|we need|the phrase|actually|however)\b"
            r"|(?:如果|是否|请判断|请问|无法确定|不能确定)[^；;。]*[？?]?$",
            normalized,
            flags=re.I,
        ):
            return False
    return not re.fullmatch(r"(?:答案|结果|待求|无法确定|不能确定)(?:[：:].*)?", normalized)


_POSITIVE_QUARTIC_RE = re.compile(
    r"(?i)(?:\b[xX]|变量\s*[xX])\s*(?:\^\s*\{?4\}?|⁴)\s*\+\s*(\d+)"
)


def _positive_quartic_constant(problem: str) -> int | None:
    match = _POSITIVE_QUARTIC_RE.search(str(problem or ""))
    if not match:
        return None
    try:
        value = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _quartic_splitting_degree(constant: int) -> int | None:
    """Compute the splitting-field degree for a concrete positive quartic.

    This is a deterministic semantic check, not an answer lookup: SymPy derives
    the minimal polynomial of a primitive element for ``Q(c**1/4, zeta_8)``.
    If the optional algebra system cannot establish the degree, the caller leaves
    the candidate unresolved instead of guessing.
    """
    try:
        import sympy as sp
        from sympy.polys.numberfields import primitive_element

        x = sp.symbols("x")
        root = sp.real_root(sp.Integer(constant), 4)
        zeta8 = (1 + sp.I) / sp.sqrt(2)
        polynomial, _ = primitive_element([root, zeta8], x)
        return int(sp.degree(polynomial, x))
    except Exception:  # noqa: BLE001 - semantic gate must fail open on uncertainty.
        return None


def objective_answer_consistency(answer: str, problem: str, mode: str) -> tuple[bool, str]:
    """Check objective-answer semantics that formatting checks cannot prove.

    Currently this covers positive quartic splitting-field questions, where a
    model can emit a well-formed three-blank answer while silently replacing the
    required eighth-root-of-unity field or dropping a factor from the degree.
    Other question families remain unchanged and return ``(True, '')``.
    """
    if mode != "fill":
        return True, ""
    constant = _positive_quartic_constant(problem)
    text = str(answer or "")
    if constant is None or not re.search(r"分裂域|splitting\s+field", str(problem or ""), re.I):
        return True, ""

    # Models frequently use LaTeX spacing commands as field separators
    # (``\;``).  Remove the outer box and normalize only those separators before
    # locating the degree; otherwise the second blank can be misread as an
    # unlabelled fragment and an incorrect degree can slip through.
    try:
        from utils.answer_extractor import extract_boxed_answer
        boxed = extract_boxed_answer(text)
        if boxed:
            text = boxed
    except Exception:  # noqa: BLE001 - semantic checking must remain fail-open.
        pass
    semantic_text = text.replace(r"\;", ";")
    lowered = semantic_text.casefold()
    has_zeta8 = bool(re.search(r"zeta\s*[_-]?\s*8|ζ\s*[₈8]|sqrt\s*\{?\s*2|√\s*2", lowered))
    has_i = bool(re.search(r"(?:^|[,，(\s])(?:\\?i)(?:\b|[,，)\s])", lowered))
    expected_degree = _quartic_splitting_degree(constant)
    parts = [part.strip().strip("{}") for part in re.split(r"[;；\n]+", semantic_text)
             if part.strip()]
    degree_part = ""
    labelled = re.search(r"(?:空|blank)\s*2\s*[：:]\s*([^;；\n]+)", semantic_text, re.I)
    if labelled:
        degree_part = labelled.group(1)
    elif len(parts) >= 2:
        degree_part = parts[1]
    else:
        degree_match = re.search(
            r"(?:扩张次数|次数|degree)[^0-9]{0,24}(\d+)", semantic_text, re.I
        )
        degree_part = degree_match.group(1) if degree_match else ""
    degree_match = re.search(r"(?<!\d)(\d+)(?!\d)", degree_part)
    claimed_degree = int(degree_match.group(1)) if degree_match else None

    if has_i and not has_zeta8 and expected_degree and expected_degree > 8:
        return False, "候选只写 i，未包含正号四次式所需的 ζ_8/√2 生成元"
    if expected_degree and claimed_degree is not None and claimed_degree != expected_degree:
        return False, f"候选扩张次数为 {claimed_degree}，独立域计算得到 {expected_degree}"
    return True, ""


def objective_answer_is_consistent(answer: str, problem: str, mode: str) -> bool:
    return objective_answer_consistency(answer, problem, mode)[0]


def fill_answer_matches_blanks(answer: str, problem: str) -> bool:
    """填空答案的分项数是否覆盖题面的全部空位（空位 ≤1 时恒真）。

    评委建议 8：填空题必须"校验空数一致"。检测不到显式空位记号时不设限，
    以免误伤没有下划线记号的填空题。
    """
    blanks = count_blanks(problem)
    if blanks <= 1:
        return True
    return fill_part_count(answer) >= blanks
