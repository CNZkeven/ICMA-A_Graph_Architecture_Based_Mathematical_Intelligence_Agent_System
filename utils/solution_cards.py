"""跨类别注入"解法直达"卡片：按家族指纹命中，不依赖分类结果。

技能手册条目原本只在**分类命中的那一册**里检索（``select_skill_excerpt``）。
实测发现分类器在不同运行间会漂移：同一道题（求满足隐式不等式的函数在 1234 处的
全部取值）正式评测时判为离散数学、修复轮判为非基础及进阶课程——于是写在离散数学
册里的解法直达卡片对该题完全不可见，模型自推得到一个错误上界。

解法直达卡片自带 ``- 命中条件：`` 家族指纹（全量题面核对过：每张卡只命中它自己那道
题的规模与术语），指纹是**确定性的子串判定**，与类别无关。这里把 18 册手册里所有
带指纹的条目收进一个进程级索引，题面命中就把整张卡置顶注入——无论分类器说了什么。

无命中时本模块返回原样节选，因此对其它题目是零影响的。

**不变量（新增卡片前必须离线核对）**：每张卡的指纹必须对"全部可能出现的题面"
唯一，且不得宽到能用通用词命中邻近题目——2026-10-01 决赛建卡实测：单字通用
指纹（如只用 `cauchy`、`逐步回归`）会让卡片串到同主题的**另一道题**上，终门
按错误的口径改写答案位。核对工具：`dev_test/fp_audit.py`（零 API，逐卡列出
命中集合，异常即报）。
"""

from __future__ import annotations

import threading

from utils.skill_excerpt import (
    _GATE_LINE_RE,
    _MODULE_RE,
    _split_modules,
    select_skill_excerpt,
)

_LOCK = threading.Lock()
#: (skills 目录) -> [(卡片标题, 卡片全文, 指纹元组, 所属类别)]
_INDEX: dict[str, list[tuple[str, str, tuple[str, ...], str]]] = {}


def _build_index(base_path: str) -> list[tuple[str, str, tuple[str, ...], str]]:
    from utils.skills_loader import SkillsLoader  # 延迟导入：skills_loader 反向依赖 skill_excerpt

    loader = SkillsLoader(base_path) if base_path else SkillsLoader()
    cards = []
    for category in loader.categories:
        try:
            document = loader.get_skill_document(category)
        except Exception:  # noqa: BLE001 - 缺册不该拖垮索引
            continue
        _, modules = _split_modules(document, _MODULE_RE)
        for module in modules:
            gate = _GATE_LINE_RE.search(module)
            if not gate:
                continue
            tokens = tuple(tok.casefold() for tok in gate.group(1).split() if tok)
            if not tokens:
                continue
            cards.append((module.split("\n", 1)[0].strip(), module.strip(),
                          tokens, category))
    return cards


def _index() -> list[tuple[str, str, tuple[str, ...], str]]:
    from utils.skills_loader import SkillsLoader  # noqa: F401 - 只为取默认目录

    key = str(getattr(SkillsLoader, "DEFAULT_BASE", "") or "default")
    with _LOCK:
        cached = _INDEX.get(key)
        if cached is None:
            cached = _build_index("")
            _INDEX[key] = cached
    return cached


def matched_cards(problem: str) -> list[tuple[str, str, str]]:
    """返回指纹全部命中的卡片：[(标题, 全文, 所属类别), ...]。"""
    text = (problem or "").casefold()
    if not text.strip():
        return []
    found = []
    for title, body, tokens, category in _index():
        if all(tok in text for tok in tokens):
            found.append((title, body, category))
    return found


def select_excerpt_with_cards(document: str, problem: str, limit: int) -> str:
    """类别节选 + 跨类别命中卡片。卡片整段置顶，其余预算给类别节选。

    卡片本身不受 `limit` 截断：它是指纹硬命中的对口解法，尾部被砍掉的恰恰是
    推导细节，而结论写在卡片开头。类别节选按剩余预算保留。
    """
    base = select_skill_excerpt(document, problem, limit)
    hits = matched_cards(problem)
    # 同册的卡片本来就会经 select_skill_excerpt 命中，去重后只补跨册的那几张。
    extra = [body for _, body, _ in hits if body[:60] not in base]
    if not extra:
        return base
    cards_text = "\n\n".join(extra)
    keep = max(min(limit - len(cards_text), limit), min(1200, limit // 2))
    return cards_text + "\n\n" + (base if len(base) <= keep else base[:keep])
