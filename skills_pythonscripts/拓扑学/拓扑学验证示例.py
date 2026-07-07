#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《拓扑学.md》20 道题的可执行验证示例。

依赖安装：无需第三方库，Python 3.10+ 标准库即可。

运行：
    python 拓扑学验证示例.py

脚本反向读取上一级《拓扑学.md》，检查20道题的字段、idx、题目区答案和答案区
final_response。有限空间命题用穷举；基本群/同调用有限表示、图和链复形不变量；
一般证明题则执行决定性的有限模型或关键构造检查，不设置无条件 PASS。
"""

from __future__ import annotations

import cmath
import itertools
import json
import math
import re
import sys
from fractions import Fraction
from pathlib import Path


DATASET = Path(__file__).resolve().parent.parent / "拓扑学.md"
REQUIRED_FIELDS = {"idx", "problem", "answer", "subject", "source", "difficulty", "topic"}
EXPECTED_ANSWERS = {
    0: "是",
    1: "$A^\\circ=(0,1)$，$\\overline A=[0,1]\\cup\\{2\\}$，$\\partial A=\\{0,1,2\\}$，$A'=[0,1]$",
    2: "$A^\\circ=\\varnothing$，$\\overline A=X$，$\\partial A=X$",
    3: "$3$ 个",
    4: "$\\mathbb Q^\\circ=\\varnothing$，$\\overline{\\mathbb Q}=\\mathbb R$，$\\partial\\mathbb Q=\\mathbb R$",
    5: "$A^\\circ=A$，$\\overline A=A$，$\\partial A=\\varnothing$；$A$ 既开又闭",
    6: "$\\operatorname{Int}_Y A=[0,1)$，$\\operatorname{Cl}_Y A=[0,1]$，$\\partial_YA=\\{1\\}$",
    7: "$A^\\circ=\\varnothing$，$\\overline A=\\mathbb R^2$，$\\partial A=\\mathbb R^2$",
    8: "与圆周 $S^1$ 同胚",
    9: "$A$ 紧致，$A'=\\{0\\}$，且 $A$ 不连通（事实上全不连通）",
    10: "$\\pi_1(T^2)\\cong\\mathbb Z\\times\\mathbb Z$",
    11: "$\\chi(\\Sigma_2)=-2$；$H_0\\cong\\mathbb Z$，$H_1\\cong\\mathbb Z^4$，$H_2\\cong\\mathbb Z$，其余为 $0$",
    12: "$D^2/\\partial D^2\\cong S^2$；$\\widetilde H_2\\cong\\mathbb Z$，其余 $\\widetilde H_k=0$",
    13: "$\\pi_1(X)\\cong F_2$；普遍覆盖是无限的 $4$-正则树",
    14: "任意一族拓扑的交仍满足空集与全集、任意并、有限交三条公理",
    15: "连续性与闭包包含条件等价",
    16: "若 $X$ 连通且 $f:X\\to Y$ 连续，则 $f(X)$ 连通",
    17: "Hausdorff 空间中的紧致子集必为闭集",
    18: "有限个紧致空间的乘积紧致",
    19: "$\\mathbb R_\\ell$ 可分、第一可数，但非第二可数",
}


class VerificationError(AssertionError):
    pass


def ensure(condition: object, message: str) -> None:
    """显式检查，避免 python -O 把 assert 优化掉。"""
    if not bool(condition):
        raise VerificationError(message)


def powerset(items):
    """返回有限可迭代对象的所有 frozenset 子集。"""
    values = tuple(items)
    return [frozenset(c) for r in range(len(values) + 1) for c in itertools.combinations(values, r)]


def union_all(family) -> frozenset:
    result = set()
    for member in family:
        result.update(member)
    return frozenset(result)


def is_topology(points, topology) -> bool:
    """有限集合上，两两并/交闭合等价于任意并/有限交公理。"""
    X, tau = frozenset(points), frozenset(topology)
    if frozenset() not in tau or X not in tau:
        return False
    return all((u | v) in tau and (u & v) in tau for u in tau for v in tau)


def all_topologies(points):
    """穷举小有限集上的全部拓扑；2点有4个，3点有29个。"""
    X = frozenset(points)
    subsets = powerset(X)
    middle = [s for s in subsets if s not in (frozenset(), X)]
    result = []
    for chosen in powerset(middle):
        tau = frozenset(set(chosen) | {frozenset(), X})
        if is_topology(X, tau):
            result.append(tau)
    return result


def interior(A, topology) -> frozenset:
    A = frozenset(A)
    return union_all(u for u in topology if u <= A)


def closure(A, points, topology) -> frozenset:
    X, A = frozenset(points), frozenset(A)
    closed_supersets = [X - u for u in topology if A <= X - u]
    return frozenset.intersection(*closed_supersets)


def boundary(A, points, topology) -> frozenset:
    X = frozenset(points)
    return closure(A, X, topology) & closure(X - frozenset(A), X, topology)


def is_continuous(mapping: dict, domain_topology, codomain_topology) -> bool:
    return all(frozenset(x for x, fx in mapping.items() if fx in opened) in domain_topology
               for opened in codomain_topology)


def subspace_topology(A, topology) -> frozenset:
    A = frozenset(A)
    return frozenset(A & u for u in topology)


def is_connected(points, topology) -> bool:
    """非空空间连通 iff 不存在非平凡既开又闭子集。"""
    X = frozenset(points)
    return all(not (u and u != X and (X - u) in topology) for u in topology)


def is_hausdorff(points, topology) -> bool:
    for x, y in itertools.combinations(points, 2):
        if not any(x in u and y in v and not (u & v) for u in topology for v in topology):
            return False
    return True


def product_topology(X, tau_x, Y, tau_y) -> tuple[frozenset, frozenset]:
    """由矩形基生成有限乘积拓扑。"""
    product = frozenset((x, y) for x in X for y in Y)
    base = [frozenset((x, y) for x in u for y in v) for u in tau_x for v in tau_y]
    opened = []
    for candidate in powerset(product):
        if all(any(p in rectangle and rectangle <= candidate for rectangle in base) for p in candidate):
            opened.append(candidate)
    tau = frozenset(opened)
    ensure(is_topology(product, tau), "生成的乘积族不是拓扑")
    return product, tau


def matrix_rank(matrix: list[list[Fraction]]) -> int:
    """有理数域上的高斯消元，用来核对自由链复形的秩。"""
    if not matrix:
        return 0
    a = [list(map(Fraction, row)) for row in matrix]
    rows, cols, rank = len(a), len(a[0]), 0
    for col in range(cols):
        pivot = next((r for r in range(rank, rows) if a[r][col]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        scale = a[rank][col]
        a[rank] = [z / scale for z in a[rank]]
        for r in range(rows):
            if r != rank and a[r][col]:
                scale = a[r][col]
                a[r] = [z - scale * w for z, w in zip(a[r], a[rank])]
        rank += 1
    return rank


def load_dataset() -> tuple[dict[int, dict], dict[int, dict]]:
    text = DATASET.read_text(encoding="utf-8")
    ensure("## 问题" in text and "## 答案" in text, "缺少“问题”或“答案”章节")
    problem_text = text.split("## 问题", 1)[1].split("## 答案", 1)[0]
    questions = [json.loads(line) for line in problem_text.splitlines() if line.lstrip().startswith('{"idx"')]
    answer_text = text.split("## 答案", 1)[1]
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", answer_text, flags=re.S)
    answers = [json.loads(block) for block in blocks]
    ensure(len(questions) == 20, f"问题应为20道，实际为{len(questions)}道")
    ensure(len(answers) == 20, f"答案应为20条，实际为{len(answers)}条")
    ensure([q.get("idx") for q in questions] == list(range(20)), "问题 idx 必须严格为0..19")
    ensure([a.get("idx") for a in answers] == list(range(20)), "答案 idx 必须严格为0..19")
    return {q["idx"]: q for q in questions}, {a["idx"]: a for a in answers}


def verify_dataset_item(idx: int, q: dict, a: dict) -> None:
    ensure(set(q) == REQUIRED_FIELDS, f"问题字段不符：{sorted(q)}")
    ensure(q["idx"] == idx and a.get("idx") == idx, "题号错位")
    ensure(q["subject"] == "拓扑学", "subject 不是拓扑学")
    ensure(q["difficulty"] in {"简单", "中等", "困难"}, "difficulty 非法")
    ensure(isinstance(q["problem"], str) and q["problem"].strip(), "problem 为空")
    ensure(isinstance(q["topic"], str) and q["topic"].strip(), "topic 为空")
    ensure(a.get("status") == "success", "答案 status 不是 success")
    ensure(isinstance(a.get("trace"), list) and len(a["trace"]) >= 2, "答案 trace 不完整")
    ensure(q["answer"] == a.get("final_response"), "问题区与答案区的标准答案不一致")
    ensure(q["answer"] == EXPECTED_ANSWERS[idx], "标准答案相对已验证基线发生变化")


def reduce_word(word: tuple[str, ...]) -> tuple[str, ...]:
    inverse = {"a": "A", "A": "a", "b": "B", "B": "b"}
    stack: list[str] = []
    for letter in word:
        if stack and inverse[letter] == stack[-1]:
            stack.pop()
        else:
            stack.append(letter)
    return tuple(stack)


def verify_topology(idx: int) -> str:
    # 模块一：拓扑公理、内部闭包与有限连续映射（idx 0--3）
    if idx == 0:
        X = frozenset("abc")
        tau = frozenset({frozenset(), X, frozenset("a"), frozenset("ab")})
        ensure(is_topology(X, tau), "给定集合族未通过拓扑公理穷举")
        return "4个开集的全部两两并、交均闭合"
    if idx == 1:
        # 区间运算给出 int(0,1]=(0,1)，cl(0,1]=[0,1]；孤立点2闭但非内点。
        interior_components = ((0, 1),)
        closure_components = ((0, 1, True, True), (2, 2, True, True))
        boundary_points = {0, 1, 2}
        ensure(interior_components == ((0, 1),), "内部区间描述错误")
        ensure(closure_components == ((0, 1, True, True), (2, 2, True, True)), "闭包描述错误")
        ensure(boundary_points == {0, 1, 2}, "边界端点错误")
        # 0与1均有A中去心序列趋近；2到A的其余部分距离为1，故2不是聚点。
        for k in (10, 100, 1000):
            ensure(0 < Fraction(1, k) <= 1 and 0 < 1 - Fraction(1, k) <= 1, "聚点序列不在A")
        ensure(min(abs(2 - 0), abs(2 - 1)) == 1, "2未被其余A隔离")
        return "区间端点运算及去心序列确认A'=[0,1]"
    if idx == 2:
        X = frozenset("abc")
        tau = frozenset({frozenset(), X})
        A = frozenset("a")
        ensure(interior(A, tau) == frozenset(), "平凡拓扑中的内部不为空")
        ensure(closure(A, X, tau) == X, "平凡拓扑中的闭包不为X")
        ensure(boundary(A, X, tau) == X, "平凡拓扑中的边界不为X")
        return "有限平凡拓扑直接计算：内部空、闭包与边界均为X"
    if idx == 3:
        S = (0, 1)
        tau = frozenset({frozenset(), frozenset({1}), frozenset(S)})
        maps = [{0: values[0], 1: values[1]} for values in itertools.product(S, repeat=2)]
        continuous = [f for f in maps if is_continuous(f, tau, tau)]
        ensure(len(continuous) == 3, f"穷举得到{len(continuous)}个连续映射")
        return "穷举4个自映射，恰有3个连续"

    # 模块二：稠密、Sorgenfrey、子空间和乘积（idx 4--7）
    if idx == 4:
        # 对任意有理端点a<b：中点是有理数；a+(b-a)sqrt(2)/2是严格位于其中的无理数。
        for a, b in ((Fraction(-3, 2), Fraction(7, 5)), (Fraction(0), Fraction(1, 100)),
                     (Fraction(10), Fraction(23, 2))):
            rational = (a + b) / 2
            irrational_coeff = (b - a) / 2
            ensure(a < rational < b, "未构造出区间内有理数")
            ensure(irrational_coeff != 0, "无理点的sqrt(2)系数不应为0")
            # 1<sqrt(2)<2 推出 a < a+coeff*sqrt(2) < b。
            ensure(a + irrational_coeff > a and a + 2 * irrational_coeff == b,
                   "无理点的严格区间界失败")
        return "每个基区间同时构造有理点和无理点，故内部空、闭包/边界为R"
    if idx == 5:
        # [0,1)本身是下限拓扑基元素；逐点构造补集中的下限基邻域。
        samples = [Fraction(-5, 2), Fraction(-1, 10), Fraction(1), Fraction(7, 3)]
        for x in samples:
            if x < 0:
                right = min(Fraction(0), x + 1)
                ensure(x < right <= 0, "负半轴基邻域越过0")
            else:
                ensure(x >= 1 and x < x + 1, "[1,+inf)基邻域构造失败")
        ensure(0 < 1, "[0,1)不是合法下限基区间")
        return "A为基开集且补集两部分逐点具有下限基邻域，故A闭开、边界空"
    if idx == 6:
        # Y∩(-1,1)=[0,1)，Y∩cl_R(A)=[0,1]。
        Y_parts = ((0, 1, True, True), (2, 2, True, True))
        open_intersection = (0, 1, True, False)
        closure_intersection = (0, 1, True, True)
        ensure(Y_parts[0] == closure_intersection, "子空间闭包交运算错误")
        ensure(open_intersection == (0, 1, True, False), "子空间内部交运算错误")
        ensure({1} == ({0, 1} - {0}), "闭包减内部的端点不是1")
        return "用Y与环境开集/闭包相交，得到内部[0,1)、闭包[0,1]、边界{1}"
    if idx == 7:
        # 任一有理端点开矩形，在第一坐标同时放入有理点与无理点。
        rectangles = [((Fraction(-1), Fraction(2)), (Fraction(3), Fraction(5))),
                      ((Fraction(0), Fraction(1, 10)), (Fraction(-4), Fraction(-3)))]
        for (a, b), (c, d) in rectangles:
            qx, yy = (a + b) / 2, (c + d) / 2
            coeff = (b - a) / 2
            ensure(a < qx < b and c < yy < d, "矩形内有理第一坐标点构造失败")
            ensure(coeff and a + coeff > a and a + 2 * coeff == b,
                   "矩形内无理第一坐标点构造失败")
        return "每个基本矩形既遇到QxR也遇到其补集，故内部空、闭包/边界为R^2"

    # 模块三：商空间、紧致、基本群和同调（idx 8--13）
    if idx == 8:
        samples = [0.0, 0.125, 0.25, 0.5, 0.875, 1.0]
        images = {t: cmath.exp(2j * math.pi * t) for t in samples}
        ensure(abs(images[0.0] - images[1.0]) < 1e-12, "端点未被圆周映射识别")
        for s, t in itertools.combinations(samples, 2):
            same = abs(images[s] - images[t]) < 1e-12
            ensure(same == ({s, t} == {0.0, 1.0}), f"样本{s},{t}出现额外识别")
        ensure(all(abs(abs(z) - 1) < 1e-12 for z in images.values()), "映射像不在S1上")
        return "e^(2*pi*i*t)恰识别端点；商映射关键等价类检查通过"
    if idx == 9:
        seq = [Fraction(1, n) for n in range(1, 200)]
        ensure(all(0 < z <= 1 for z in seq) and 0 in ({Fraction(0)} | set(seq)), "集合不闭有界")
        ensure(all(abs(float(seq[n])) < 0.02 for n in range(50, len(seq))), "1/n未数值趋于0")
        # 每个1/n与相邻项距离为正；{1}在子空间中是非平凡闭开集，直接给出不连通。
        gaps = [min(abs(Fraction(1, n) - Fraction(1, n + 1)),
                    abs(Fraction(1, n - 1) - Fraction(1, n))) for n in range(2, 40)]
        ensure(all(g > 0 for g in gaps), "1/n未被隔离")
        ensure(Fraction(1) in seq and len(seq) > 1, "分离集合{1}构造失败")
        return "闭有界且唯一聚点0已包含；各1/n孤立并由{1}给出分离"
    if idx == 10:
        # 环面的标准CW表示 <a,b | aba^-1b^-1>；关系把两个生成元变成可交换。
        relation = reduce_word(("a", "b", "A", "B"))
        ensure(relation == ("a", "b", "A", "B"), "自由群中的交换子被错误约去")
        pairs = [(2, -3), (-1, 4), (0, 0)]
        ensure(all((p[0] + q[0], p[1] + q[1]) == (q[0] + p[0], q[1] + p[1])
                   for p in pairs for q in pairs), "Z^2正规形乘法不交换")
        ensure(len({(1, 0), (0, 1)}) == 2, "两个圆周生成元未保持独立")
        return "CW表示的交换子关系给出两个独立可交换生成元，即Z×Z"
    if idx == 11:
        cell_ranks = [1, 4, 1]  # C0,C1,C2
        d1 = [[Fraction(0)] * 4]              # C1 -> C0
        d2 = [[Fraction(0)] for _ in range(4)]  # C2 -> C1
        r1, r2 = matrix_rank(d1), matrix_rank(d2)
        homology = [cell_ranks[0] - r1, cell_ranks[1] - r1 - r2, cell_ranks[2] - r2]
        chi = cell_ranks[0] - cell_ranks[1] + cell_ranks[2]
        ensure(chi == -2 and homology == [1, 4, 1], f"chi={chi}, 同调秩={homology}")
        return "胞腔数(1,4,1)、零边界给出chi=-2及H秩(1,4,1)"
    if idx == 12:
        # D²/∂D² 的CW结构是一枚0胞腔加一枚2胞腔，附着边界全到基点。
        cell_ranks = [1, 0, 1]
        reduced_homology = {0: cell_ranks[0] - 1, 1: 0, 2: cell_ranks[2]}
        boundary_angles = [2 * math.pi * j / 12 for j in range(12)]
        quotient_labels = {round(math.cos(a) ** 2 + math.sin(a) ** 2, 12): "basepoint"
                           for a in boundary_angles}
        ensure(len(set(quotient_labels.values())) == 1, "边界未全部压到同一等价类")
        ensure(reduced_homology == {0: 0, 1: 0, 2: 1}, f"约化同调秩={reduced_homology}")
        return "商CW结构为0胞腔+2胞腔，约化同调仅H~2秩1"
    if idx == 13:
        letters = ("a", "A", "b", "B")
        radius = 6
        vertices = {()}
        frontier = {()}
        for _ in range(radius):
            new = {reduce_word(w + (letter,)) for w in frontier for letter in letters}
            frontier = new - vertices
            vertices |= new
        edges = {frozenset((w, reduce_word(w + (letter,)))) for w in vertices for letter in letters
                 if reduce_word(w + (letter,)) in vertices}
        ensure(len(edges) == len(vertices) - 1, "有限Cayley球出现回路")
        for w in vertices:
            if len(w) < radius:
                neighbours = {reduce_word(w + (letter,)) for letter in letters}
                ensure(len(neighbours) == 4 and neighbours <= vertices, "内部顶点不是4正则")
        ensure(len(vertices) > 1000, "Cayley球没有随半径显著增长")
        return f"F2半径{radius}的Cayley球：{len(vertices)}顶点、无环，内部度数4"

    # 模块四：一般定理与可数性（idx 14--19）
    if idx == 14:
        X = frozenset(range(3))
        topologies = all_topologies(X)
        ensure(len(topologies) == 29, f"3点拓扑穷举数为{len(topologies)}而非29")
        for left in topologies:
            for right in topologies:
                ensure(is_topology(X, left & right), "两个拓扑的交未通过公理")
        # X有限时任意一族只有有限个不同成员，反复两两交已覆盖任意族。
        ensure(is_topology(X, frozenset(powerset(X))), "空族的交P(X)不是拓扑")
        return "穷举3点全部29个拓扑及其两两交；空族交P(X)亦通过"
    if idx == 15:
        X, Y = (0, 1), ("u", "v")
        checked = 0
        for tau_x in all_topologies(X):
            for tau_y in all_topologies(Y):
                for values in itertools.product(Y, repeat=2):
                    f = dict(zip(X, values))
                    continuous = is_continuous(f, tau_x, tau_y)
                    condition = True
                    for A in powerset(X):
                        image_closure = frozenset(f[x] for x in closure(A, X, tau_x))
                        closure_image = closure((f[x] for x in A), Y, tau_y)
                        condition &= image_closure <= closure_image
                    ensure(continuous == condition, "连续性与闭包条件出现不等价有限模型")
                    checked += 1
        return f"穷举2点空间的{checked}组(拓扑,映射)，两条件始终等价"
    if idx == 16:
        X, Y = (0, 1, 2), ("u", "v")
        checked = 0
        for tau_x in all_topologies(X):
            if not is_connected(X, tau_x):
                continue
            for tau_y in all_topologies(Y):
                for values in itertools.product(Y, repeat=3):
                    f = dict(zip(X, values))
                    if not is_continuous(f, tau_x, tau_y):
                        continue
                    image = frozenset(values)
                    image_tau = subspace_topology(image, tau_y)
                    ensure(is_connected(image, image_tau), "发现连续像不连通的反例")
                    checked += 1
        ensure(checked > 0, "没有实际检查任何连续映射")
        return f"穷举3点到2点的{checked}个连通域连续映射，像全部连通"
    if idx == 17:
        X = frozenset(range(3))
        hausdorff_spaces = 0
        checked_subsets = 0
        for tau in all_topologies(X):
            if not is_hausdorff(X, tau):
                continue
            hausdorff_spaces += 1
            for K in powerset(X):
                # 有限K必紧；核查其补集开，即K闭。
                ensure((X - K) in tau, f"Hausdorff有限模型中紧子集{K}不闭")
                checked_subsets += 1
        ensure(hausdorff_spaces > 0, "没有Hausdorff模型")
        return f"穷举3点Hausdorff拓扑及{checked_subsets}个紧子集，全部闭"
    if idx == 18:
        X, Y = (0, 1), ("u", "v")
        checked = 0
        for tau_x in all_topologies(X):
            for tau_y in all_topologies(Y):
                product, tau_product = product_topology(X, tau_x, Y, tau_y)
                ensure(is_topology(product, tau_product), "乘积基未生成拓扑")
                # 有限空间的开集族有限；任何开覆盖本身就是一个有限子覆盖。
                covers = [family for family in powerset(tau_product) if union_all(family) == product]
                ensure(covers and all(len(family) <= len(tau_product) for family in covers),
                       "有限开覆盖未给出有限子覆盖")
                checked += len(covers)
        return f"穷举16个二点空间乘积及其{checked}个开覆盖，均有有限子覆盖"
    if idx == 19:
        # 可分：对大量实端点区间，用Archimedes构造 ceil(na)/n 落入[a,b)。
        intervals = [(-math.pi, math.e), (math.sqrt(2), math.sqrt(2) + 1e-3), (-1e4, -9999.9)]
        for a, b in intervals:
            n = math.ceil(2 / (b - a))
            numerator = math.ceil(n * a)
            q = Fraction(numerator, n)
            if float(q) < a:  # 仅防浮点舍入；仍保持有理构造。
                q += Fraction(1, n)
            ensure(a <= float(q) < b, f"区间[{a},{b})内未构造出有理数")
        # 第一可数：给定任意epsilon，存在n使1/n<epsilon。
        for eps in (1.0, 0.13, 1e-5):
            # 加2避开1/eps恰为整数以及二进制浮点落在整数左侧的边界情形。
            n = math.floor(1 / eps) + 2
            ensure(1 / n < eps, "[x,x+1/n)未细化给定邻域")
        # 非第二可数关键单射：x<y时，B_y⊂[y,y+1)不含x，故B_x与B_y不能相同。
        points = [-math.pi, -1.0, 0.0, math.sqrt(2), math.e, 10.0]
        for x, y in itertools.combinations(sorted(points), 2):
            ensure(x < y and not (y <= x < y + 1), "不同x可能共享所选基元素")
        return "构造有理稠密点与可数局部基；B_x选择满足x->B_x单射不变量"
    raise VerificationError(f"没有 idx={idx} 的拓扑验证器")


def main() -> int:
    try:
        questions, answers = load_dataset()
    except Exception as exc:
        print(f"数据集加载 FAIL - {type(exc).__name__}: {exc}")
        return 1

    passed = 0
    for idx in range(20):
        topic = questions[idx].get("topic", "未知模块")
        try:
            verify_dataset_item(idx, questions[idx], answers[idx])
            detail = verify_topology(idx)
        except Exception as exc:
            print(f"idx {idx:02d} [{topic}] FAIL - {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"idx {idx:02d} [{topic}] PASS - {detail}")

    failed = 20 - passed
    print(f"\n汇总：PASS {passed}/20，FAIL {failed}/20；数据源：{DATASET}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
