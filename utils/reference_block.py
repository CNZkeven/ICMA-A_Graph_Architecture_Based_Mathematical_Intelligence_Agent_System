"""题库参考示例区块的唯一构造入口。

推理节点与 Python 节点各自内联拼装过同一段文本，两份实现的截断长度不同、
措辞也会各自漂移。检索条数与分支注入契约在这里集中实现，调用方只保留各自的
截断额度。

检索结果只能提供可迁移的方法线索。近邻题可能改变参数、操作角色、量词或边界，
所以本模块只生成反锚定提示和题面差异，绝不把任何评测题号或固定答案写进提示。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

_HEADER = "\n\n参考示例（来自数学竞赛题库的相似题目与解答）：\n"
_PYTHON_HEADER = "\n\n参考题面（来自题库的相似题，仅用于验证方法）：\n"

#: 反锚定说明。放在示例**之前**——写在后面时模型往往已经先读完解答并锚定了结论。
_ANTI_ANCHOR_NOTE = """
⚠️ 这些是按向量相似度检索出来的**近似题，不是本题**。相似度高只说明措辞接近，
不说明结论相同；把近似题的答案搬过来是本项目已确认的头号失分来源。使用规则：

1. 先逐项核对差异：参数数值（个数/上界/模数）、**奇偶性**、谁做选择、目标是
   最大化还是最小化、约束方向（≤ 还是 <、"所有解"还是"一个解"）。
2. 只要有一项不同，示例的**结论就不可迁移**——它的答案不是本题的答案。此时最多
   借用它的**方法**，并且必须在本题自己的参数下从头重算，再把结果代回本题题面验证。
3. 核对后若不能确认两题严格同构，就完全忽略示例，独立求解。宁可自己算，也不要
   抄一个参数不同的结论。
"""

#: 题面里有判别力的整数字面量：跳过 0/1 这类到处都是的数，也跳过 LaTeX 命令里
#: 的上下标数字（\sqrt[3]、x^{2}），只留下真正描述规模的参数。
_INT_RE = re.compile(r"(?<![\\^_{\w])(\d{2,})(?![}\w])")


def _scale_numbers(text: str) -> list[int]:
    seen: list[int] = []
    for match in _INT_RE.finditer(str(text or "")):
        value = int(match.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def _numeric_diff_line(problem: str, example_problem: str) -> str:
    """把两边题面的规模参数差异摆出来；无从比较时返回空串。"""
    ours = _scale_numbers(problem)
    theirs = _scale_numbers(example_problem)
    if not ours or not theirs:
        return ""
    only_ours = [n for n in ours if n not in theirs][:6]
    only_theirs = [n for n in theirs if n not in ours][:6]
    if not only_ours and not only_theirs:
        return ""
    parts = []
    if only_ours:
        parity = "奇" if only_ours[0] % 2 else "偶"
        parts.append(f"本题独有 {only_ours}（首个为{parity}数）")
    if only_theirs:
        parity = "奇" if only_theirs[0] % 2 else "偶"
        parts.append(f"示例独有 {only_theirs}（首个为{parity}数）")
    return ("**⚠ 参数差异**：" + "；".join(parts)
            + "。参数不同则结论不同，必须在本题参数下重算。\n")


def partition_reference_examples(
    examples: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Split retrieved examples into independent branch contexts.

    Retrieval is intentionally performed once, but the two solving branches must
    not see the same answer-bearing example.  Rank parity is deterministic and
    keeps the partition stable in traces and tests.  The Python side receives a
    copy with its solution removed, so it can use topic/parameter cues without
    inheriting a proposed conclusion.
    """
    usable: list[dict[str, Any]] = []
    for rank, example in enumerate(examples or []):
        if not isinstance(example, Mapping):
            continue
        item = dict(example)
        source = str(item.get("source") or "retrieved")
        item["reference_id"] = f"{source}:{rank}"
        item["retrieval_rank"] = rank
        usable.append(item)

    reasoning: list[dict[str, Any]] = []
    python: list[dict[str, Any]] = []
    for position, item in enumerate(usable):
        if position % 2 == 0:
            branch_item = dict(item)
            branch_item["reference_role"] = "reasoning"
            reasoning.append(branch_item)
        else:
            branch_item = dict(item)
            branch_item["reference_role"] = "python"
            branch_item["solution"] = ""
            branch_item["answer_suppressed"] = True
            python.append(branch_item)

    reasoning_ids = {item["reference_id"] for item in reasoning}
    python_ids = {item["reference_id"] for item in python}
    return {
        "reasoning": reasoning,
        "python": python,
        "strategy": "rank_parity_disjoint_solution_suppressed",
        "overlap": sorted(reasoning_ids & python_ids),
    }


def build_reference_block(
    examples: List[Dict[str, Any]] | None,
    problem_chars: int,
    solution_chars: int,
    problem: str = "",
    *,
    include_solutions: bool = True,
    role: str = "reasoning",
) -> str:
    """把检索到的每一条相似题拼成参考区块；无检索结果时返回空串。

    Args:
        examples: `retrieved_examples`，按相似度降序。全部注入，不再二次筛选。
        problem_chars: 每条题面的截断长度。
        solution_chars: 每条解答的截断长度。
        problem: 本题题面。给出时逐条附上与示例的规模参数差异（反锚定）。
        include_solutions: 是否包含示例解答；Python 分支必须关闭。
        role: 使用该区块的分支名称，写入提示以便审计。
    """
    usable = [ex for ex in (examples or []) if isinstance(ex, dict)]
    if not usable:
        return ""
    header = _HEADER if include_solutions else _PYTHON_HEADER
    parts = [header, f"\n[参考区块角色：{role}]\n", _ANTI_ANCHOR_NOTE]
    for i, example in enumerate(usable, 1):
        try:
            similarity = float(example.get("similarity") or 0.0)
        except (TypeError, ValueError):
            similarity = 0.0
        example_problem = str(example.get("problem") or "")
        parts.append(f"\n### 示例 {i} (相似度: {similarity:.3f}，**不是本题**)\n")
        diff = _numeric_diff_line(problem, example_problem) if problem else ""
        if diff:
            parts.append(diff)
        parts.append(f"**题目：**\n{example_problem[:problem_chars]}\n\n")
        if include_solutions:
            parts.append(f"**解答：**\n{str(example.get('solution') or '')[:solution_chars]}\n")
        else:
            parts.append("**验证用途：** 仅使用题面和参数差异核对方法；忽略任何示例结论。\n")
        if i < len(usable):
            parts.append("\n---\n")
    return "".join(parts)
