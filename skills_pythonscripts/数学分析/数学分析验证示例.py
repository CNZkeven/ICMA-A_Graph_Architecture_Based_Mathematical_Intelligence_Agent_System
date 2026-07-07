#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《数学分析.md》20 道题的可执行验证示例。

依赖安装：
    python -m pip install "sympy>=1.12" "mpmath>=1.3"

运行：
    python 数学分析验证示例.py

脚本会反向读取上一级《数学分析.md》，逐题校验数据字段、题号以及题目区/答案区
的标准答案，再用 SymPy、mpmath 或可执行的关键逻辑不变量独立验证结论。
任何一题失败都会令进程以非零状态退出，适合人工学习及 CI 检查。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import mpmath as mp
import sympy as sp


DATASET = Path(__file__).resolve().parent.parent / "数学分析.md"
REQUIRED_FIELDS = {"idx", "problem", "answer", "subject", "source", "difficulty", "topic"}
EXPECTED_ANSWERS = {
    0: "$\\frac12$",
    1: "$\\frac12$",
    2: "$56$",
    3: "$-\\frac19$",
    4: "$\\frac{\\pi}{4a^3}$",
    5: "$\\frac34$",
    6: "$1+x-\\frac{x^3}{3}-\\frac{x^4}{6}+o(x^4)$",
    7: "$\\inf A=\\frac32$（可取到），$\\sup A=2$（不可取到）",
    8: "两个累次极限均为 $0$；二重极限不存在",
    9: "收敛半径 $R=1$；收敛区间 $[-1,1)$；和函数为 $-\\ln(1-x)$（$-1\\le x<1$）",
    10: "$4$",
    11: "$\\ln2$",
    12: "$-\\frac{\\pi}{4}$",
    13: "$\\ln2$",
    14: "在 $x=0$ 处连续但不可导",
    15: "取 $g=f$ 得 $\\int_0^1f^2=0$，由连续性推出 $f\\equiv0$",
    16: "在每个 $[0,a]$（$0<a<1$）上一致收敛；在 $[0,1)$ 上不一致收敛",
    17: "在 $[0,1]$ 上一致收敛，但对每个 $x\\in[0,1]$ 都不绝对收敛",
    18: "$f_n$ 一致收敛，且其极限 $f$ 满足 $f'=\\lim f_n'$",
    19: "$f_n\\to0$ 逐点但非一致；$\\int_0^1f_n\\,dx=\\frac16$，故不能交换极限与积分",
}


class VerificationError(AssertionError):
    """带有可读失败原因的验证异常。"""


def ensure(condition: object, message: str) -> None:
    """不能使用 assert：python -O 会关闭 assert，而验证脚本不应被静默绕过。"""
    if not bool(condition):
        raise VerificationError(message)


def load_dataset() -> tuple[dict[int, dict], dict[int, dict]]:
    """解析问题区的逐行 JSON 和答案区的 fenced JSON。"""
    text = DATASET.read_text(encoding="utf-8")
    ensure("## 问题" in text and "## 答案" in text, "缺少“问题”或“答案”章节")
    problem_text = text.split("## 问题", 1)[1].split("## 答案", 1)[0]
    questions = [json.loads(line) for line in problem_text.splitlines() if line.lstrip().startswith('{"idx"')]
    answer_text = text.split("## 答案", 1)[1]
    answer_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", answer_text, flags=re.S)
    answers = [json.loads(block) for block in answer_blocks]
    ensure(len(questions) == 20, f"问题应为20道，实际为{len(questions)}道")
    ensure(len(answers) == 20, f"答案应为20条，实际为{len(answers)}条")
    ensure([q.get("idx") for q in questions] == list(range(20)), "问题 idx 必须严格为0..19且依次排列")
    ensure([a.get("idx") for a in answers] == list(range(20)), "答案 idx 必须严格为0..19且依次排列")
    return {q["idx"]: q for q in questions}, {a["idx"]: a for a in answers}


def verify_dataset_item(idx: int, q: dict, a: dict) -> None:
    """检查字段和两处答案；EXPECTED_ANSWERS 使答案被同步篡改时也能检出。"""
    ensure(set(q) == REQUIRED_FIELDS, f"问题字段不符：{sorted(q)}")
    ensure(q["idx"] == idx and a.get("idx") == idx, "题号错位")
    ensure(q["subject"] == "数学分析", "subject 不是数学分析")
    ensure(q["difficulty"] in {"简单", "中等", "困难"}, "difficulty 非法")
    ensure(isinstance(q["problem"], str) and q["problem"].strip(), "problem 为空")
    ensure(isinstance(q["topic"], str) and q["topic"].strip(), "topic 为空")
    ensure(a.get("status") == "success", "答案 status 不是 success")
    ensure(isinstance(a.get("trace"), list) and len(a["trace"]) >= 2, "答案 trace 不完整")
    ensure(q["answer"] == a.get("final_response"), "问题区与答案区的标准答案不一致")
    ensure(q["answer"] == EXPECTED_ANSWERS[idx], "标准答案相对已验证基线发生变化")


def verify_math(idx: int) -> str:
    """逐题独立验算。证明题检查其决定性反例、估计式或逻辑不变量。"""
    x, y, t = sp.symbols("x y t", real=True)
    n = sp.symbols("n", integer=True, positive=True)

    # 模块一：极限、高阶导数、广义积分（idx 0--4）
    if idx == 0:
        value = sp.limit(n * (sp.sqrt(n**2 + 1) - n), n, sp.oo)
        ensure(value == sp.Rational(1, 2), f"极限算得 {value}")
        return "有理化后的数列极限=1/2"
    if idx == 1:
        value = sp.limit((1 - sp.cos(x)) / x**2, x, 0)
        ensure(value == sp.Rational(1, 2), f"极限算得 {value}")
        return "函数极限=1/2"
    if idx == 2:
        value = sp.diff(x**2 * sp.exp(x), x, 8).subs(x, 0)
        ensure(value == 56, f"八阶导数算得 {value}")
        return "八阶导数=56"
    if idx == 3:
        value = sp.integrate(x**2 * sp.log(x), (x, 0, 1))
        ensure(value == -sp.Rational(1, 9), f"积分算得 {value}")
        return "广义积分=-1/9"
    if idx == 4:
        a = sp.symbols("a", positive=True)
        value = sp.integrate(1 / (x**2 + a**2) ** 2, (x, 0, sp.oo))
        ensure(sp.simplify(value - sp.pi / (4 * a**3)) == 0, f"积分算得 {value}")
        return "含参积分=pi/(4a^3)"

    # 模块二：级数、泰勒展开、确界、多元极限（idx 5--8）
    if idx == 5:
        value = sp.summation(1 / (n * (n + 2)), (n, 1, sp.oo))
        ensure(value == sp.Rational(3, 4), f"级数和算得 {value}")
        return "裂项级数和=3/4"
    if idx == 6:
        polynomial = sp.series(sp.exp(x) * sp.cos(x), x, 0, 5).removeO().expand()
        target = 1 + x - x**3 / 3 - x**4 / 6
        ensure(sp.expand(polynomial - target) == 0, f"Taylor 多项式为 {polynomial}")
        return "0至4次系数全部吻合"
    if idx == 7:
        term = (2 * n + 1) / (n + 1)
        increment = sp.simplify(term.subs(n, n + 1) - term)
        ensure(sp.ask(sp.Q.positive(increment)) is True, f"未证得严格递增：差={increment}")
        ensure(term.subs(n, 1) == sp.Rational(3, 2), "首项不是3/2")
        ensure(sp.limit(term, n, sp.oo) == 2, "上确界候选不是2")
        ensure(sp.solve(sp.Eq(term, 2), n) == [], "存在有限 n 取到2")
        return "严格递增，最小值3/2，极限2且不取到"
    if idx == 8:
        f = x * y / (x**2 + y**2)
        first = sp.limit(sp.limit(f, y, 0), x, 0)
        second = sp.limit(sp.limit(f, x, 0), y, 0)
        path_diag = sp.simplify(f.subs(y, x))
        path_axis = sp.simplify(f.subs(y, 0))
        ensure((first, second) == (0, 0), f"累次极限为 {first}, {second}")
        ensure(path_diag == sp.Rational(1, 2) and path_axis == 0, "两条路径未产生不同极限")
        return "累次极限均0；路径y=x与y=0分别给1/2和0"

    # 模块三：幂级数、变差与特殊积分（idx 9--13）
    if idx == 9:
        radius_test = sp.limit((n / (n + 1)), n, sp.oo)  # |a_n/a_{n+1}|，a_n=1/n
        ensure(radius_test == 1, f"收敛半径算得 {radius_test}")
        ensure(sp.summation((-1) ** n / n, (n, 1, sp.oo)) == -sp.log(2), "x=-1端点值错误")
        ensure(sp.summation(1 / n, (n, 1, sp.oo)) == sp.oo, "x=1端点应发散")
        candidate = -sp.log(1 - x)
        ensure(sp.simplify(sp.diff(candidate, x) - 1 / (1 - x)) == 0 and candidate.subs(x, 0) == 0,
               "和函数未满足导数与初值")
        return "R=1，端点检查及和函数微分方程均通过"
    if idx == 10:
        values = [sp.sin(v) for v in (0, sp.pi / 2, 3 * sp.pi / 2, 2 * sp.pi)]
        variation = sum(abs(values[i + 1] - values[i]) for i in range(3))
        ensure(variation == 4, f"按单调段计算的全变差为 {variation}")
        return "三个单调段的变差和=4"
    if idx == 11:
        value = sp.summation((-1) ** (n - 1) / n, (n, 1, sp.oo))
        ensure(value == sp.log(2), f"交错调和级数和为 {value}")
        return "交错调和级数和=ln(2)"
    if idx == 12:
        # 直接符号积分在部分 SymPy 版本会保留复对数分支；改用题解中的 Beta 参数法。
        s = sp.symbols("s", real=True)
        beta_form = sp.gamma((s + 1) / 2) * sp.gamma((3 - s) / 2) / 2
        value = sp.simplify(sp.diff(beta_form, s).subs(s, 0))
        numeric = mp.quad(lambda z: mp.log(z) / (1 + z*z) ** 2, [0, 1, mp.inf])
        ensure(sp.simplify(value + sp.pi / 4) == 0, f"符号积分为 {value}")
        ensure(mp.almosteq(numeric, -mp.pi / 4, rel_eps=mp.mpf("1e-30")), f"高精度积分为 {numeric}")
        return "符号与高精度积分均为-pi/4"
    if idx == 13:
        pointwise = sp.limit(t ** (1 / n) / (1 + t), n, sp.oo)
        ensure(sp.simplify(pointwise - 1 / (1 + t)) == 0, f"换元后逐点极限为 {pointwise}")
        mp.mp.dps = 40
        approx = mp.quad(lambda z: z ** (mp.mpf(1) / 20000) / (1 + z), [0, 1])
        ensure(abs(approx - mp.log(2)) < mp.mpf("0.0001"), f"n=20000时数值为 {approx}")
        ensure(all(0 <= z ** (1 / 7) / (1 + z) <= 1 / (1 + z) for z in (0.01, 0.2, 0.7, 1.0)),
               "控制函数不等式抽查失败")
        return "边界层换元、控制条件与高精度极限均通过"

    # 模块四：连续性、一致收敛与换序定理（idx 14--19）
    if idx == 14:
        k = sp.symbols("k", integer=True, positive=True)
        xp = 1 / (sp.pi / 2 + 2 * sp.pi * k)
        xm = 1 / (3 * sp.pi / 2 + 2 * sp.pi * k)
        ensure(sp.limit(x * sp.sin(1 / x), x, 0) == 0, "函数值极限不是0")
        ensure(sp.simplify(sp.sin(1 / xp)) == 1 and sp.simplify(sp.sin(1 / xm)) == -1,
               "差商的两列反例没有给出1和-1")
        ensure(sp.limit(xp, k, sp.oo) == 0 and sp.limit(xm, k, sp.oo) == 0, "反例序列不趋于0")
        return "函数连续；差商沿两列分别为1和-1"
    if idx == 15:
        c, delta = sp.symbols("c delta", positive=True)
        lower_bound = 2 * delta * c**2
        ensure(sp.ask(sp.Q.positive(lower_bound)) is True, "连续性反证中的正积分下界未证为正")
        # g=f 后被积函数非负；若某点非零，连续性给出 |f|>=c 的长度2delta邻域。
        ensure(sp.integrate(c**2, (x, -delta, delta)) == lower_bound, "局部积分下界计算错误")
        return "g=f得到非负积分；非零点会强制产生正下界"
    if idx == 16:
        aa, N = sp.symbols("a N", positive=True)
        tail_bound = aa ** (N + 1) / (1 - aa)
        ensure(sp.limit(tail_bound.subs(aa, sp.Rational(1, 2)), N, sp.oo) == 0, "M判别尾界不趋零")
        ensure(sp.limit(-sp.log(1 - x), x, 1, dir="-") == sp.oo, "[0,1)上的和函数未检出无界")
        finite_partial = sum(x**j / j for j in range(1, 8))
        ensure(sp.limit(finite_partial, x, 1, dir="-").is_finite is True, "有限部分和应有界")
        return "闭子区间尾界趋0；半开区间极限无界而部分和有界"
    if idx == 17:
        N = sp.symbols("N", integer=True, positive=True)
        uniform_remainder_bound = 1 / (N + 1)
        ensure(sp.limit(uniform_remainder_bound, N, sp.oo) == 0, "一致Leibniz余项界不趋零")
        ensure(sp.summation(1 / (n + 1), (n, 1, sp.oo)) == sp.oo, "绝对值级数的下界未发散")
        ensure(sp.simplify(1 / n - 1 / (n + 1)) > 0, "通项未随n递减")
        return "一致交错余项界趋0；绝对值级数由调和尾项下界发散"
    if idx == 18:
        eps, length = sp.symbols("eps length", positive=True)
        point_bound = eps / 2
        derivative_bound = eps / (2 * length)
        total_bound = sp.simplify(point_bound + length * derivative_bound)
        ensure(total_bound == eps, "一致Cauchy估计未闭合到epsilon")
        # FTC 表示中的导数误差积分至多 length*||f_n'-g||；该算式正是换极限关键。
        eta = sp.symbols("eta", positive=True)
        ensure(sp.integrate(eta, (x, 0, length)) == eta * length, "导数误差积分界错误")
        return "一点值差+区间长度×导数上确界差给出一致Cauchy与FTC换序"
    if idx == 19:
        fn = n**2 * x * (1 - n * x)
        critical = sp.solve(sp.Eq(sp.diff(fn, x), 0), x)
        ensure(critical == [1 / (2 * n)], f"极值点算得 {critical}")
        maximum = sp.simplify(fn.subs(x, critical[0]))
        integral = sp.integrate(fn, (x, 0, 1 / n))
        ensure(maximum == n / 4 and sp.limit(maximum, n, sp.oo) == sp.oo, "非一致性最大值检查失败")
        ensure(integral == sp.Rational(1, 6), f"积分算得 {integral}")
        # 固定x>0时，任取整数n>1/x便落到支集外；用三个有理点执行该构造。
        for fixed in (sp.Rational(1, 10), sp.Rational(2, 7), sp.Rational(1, 1)):
            chosen_n = int(sp.floor(1 / fixed)) + 1
            ensure(sp.Rational(1, chosen_n) < fixed, "逐点最终离开支集的构造失败")
        return "逐点支集消失；上确界n/4；积分恒为1/6"
    raise VerificationError(f"没有 idx={idx} 的数学验证器")


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
            detail = verify_math(idx)
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
