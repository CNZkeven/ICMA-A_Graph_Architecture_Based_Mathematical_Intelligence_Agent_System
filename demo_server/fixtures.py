"""Curated home-page examples.

``EXAMPLES`` are the three sample problems, replayed from mathematically
checked fixtures.  ``COMPETITION_EXAMPLES`` are the "试试竞赛题" cards: past
全国大学生数学竞赛 problems whose stages replay a recorded real ICMA run
(``demo_server/staged/<id>.json``) with staged pacing, see ``staged.py``.
"""

from __future__ import annotations

from pathlib import Path


SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills_pythonscripts"


def _skill_files(category: str) -> dict[str, str]:
    category_dir = SKILLS_ROOT / category
    return {
        "category": category,
        "category_path": str(category_dir),
        "skill_document": str(category_dir / f"{category}skill.md"),
        "validation_script": str(category_dir / f"{category}验证示例.py"),
    }


STAGE_EVIDENCE = {
    "integral": {
        **_skill_files("数学分析"),
        "classification_basis": ["题面包含定积分", "被积函数是幂函数", "需要区间端点代入"],
        "reasoning_analysis": "先识别被积函数 x²，使用幂函数原函数 x³/3，再依次代入上、下限。",
        "reasoning_steps": [
            "识别 integrand：x²，积分区间为 [0, 1]。",
            "应用幂函数积分公式：∫x²dx = x³/3。",
            "计算端点差：1³/3 - 0³/3 = 1/3。",
        ],
        "reasoning_result": "候选答案为 1/3。",
        "python_code": "import sympy as sp\nx = sp.symbols('x')\nanswer = sp.integrate(x**2, (x, 0, 1))\nprint(\"最终答案:\", answer)",
        "python_stdout": "最终答案: 1/3",
        "python_result": "SymPy 得到 1/3，与推导结果一致。",
        "references": [
            {
                "rank": 1,
                "similarity": 0.91,
                "source": "竞赛题库 / 定积分基础",
                "problem": "计算积分 ∫₀¹ (3x² + 2x) dx。",
                "solution": "逐项积分，得到 [x³ + x²]₀¹ = 2。",
            },
            {
                "rank": 2,
                "similarity": 0.84,
                "source": "训练题库 / 幂函数积分",
                "problem": "计算积分 ∫₀² x² dx。",
                "solution": "使用原函数 x³/3，结果为 [x³/3]₀² = 8/3。",
            },
        ],
    },
    "optimization": {
        **_skill_files("高等代数"),
        "classification_basis": ["目标是二次型最小值", "含有对称矩阵", "需要驻点与正定性判断"],
        "reasoning_analysis": "把目标函数写成二次型，先令梯度为零求唯一驻点，再用 Hessian 正定性证明该驻点是全局最小点。",
        "reasoning_steps": [
            "计算梯度：∇f=(2x+y-3, x+2y-3)。",
            "令梯度为零，解线性方程组得到 x=y=1。",
            "Hessian 为 [[2,1],[1,2]]，特征值 1、3 均为正，因此严格凸。",
            "代入驻点：f(1,1)=1+1+1-3-3=-3。",
        ],
        "reasoning_result": "全局最小值为 -3，且仅在 (x,y)=(1,1) 取得。",
        "python_code": "import sympy as sp\nx, y = sp.symbols('x y')\nf = x**2 + x*y + y**2 - 3*x - 3*y\ncritical = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y], dict=True)[0]\nprint(\"驻点:\", critical)\nprint(\"最小值:\", sp.simplify(f.subs(critical)))\nprint(\"Hessian:\", sp.hessian(f, (x, y)))",
        "python_stdout": "驻点: {x: 1, y: 1}\n最小值: -3\nHessian: Matrix([[2, 1], [1, 2]])",
        "python_result": "Hessian 正定，计算出的最小值 -3 与推理结果一致。",
        "references": [
            {
                "rank": 1,
                "similarity": 0.89,
                "source": "竞赛题库 / 二次型",
                "problem": "求 g(x,y)=x²+2xy+2y²-4x-6y 的最小值。",
                "solution": "令梯度为零求驻点，再检查 Hessian 的正定性；该流程与当前题目相同。",
            },
            {
                "rank": 2,
                "similarity": 0.82,
                "source": "训练题库 / 凸二次函数",
                "problem": "判断 h(x,y)=2x²+2xy+2y²-4x-6y 是否存在唯一全局最小点。",
                "solution": "Hessian 的两个特征值均为正，所以函数严格凸，驻点就是唯一全局最小点。",
            },
        ],
    },
    "number-theory": {
        **_skill_files("离散数学"),
        "classification_basis": ["题面要求证明合数", "核心是整除与因式分解", "需要按 n 的奇偶分类"],
        "reasoning_analysis": "关键在于按 n 的奇偶性拆分：偶数直接提取因子 16，奇数把 4ⁿ 写成 4b⁴ 后使用 Sophie Germain 恒等式。",
        "reasoning_steps": [
            "n 为偶数时，n⁴ 与 4ⁿ 都被 16 整除，且整体大于 16。",
            "n 为奇数时令 b=2^((n-1)/2)，则 4ⁿ=4b⁴。",
            "套用 a⁴+4b⁴=(a²-2ab+2b²)(a²+2ab+2b²)，取 a=n。",
            "两种情形都得到非平凡因子，因此表达式是合数。",
        ],
        "reasoning_result": "奇偶两类均完成证明，对所有整数 n>1 成立。",
        "python_code": "import sympy as sp\nfor n in range(2, 9):\n    value = n**4 + 4**n\n    factors = sp.factorint(value)\n    print(n, value, factors)\nprint(\"恒等式检查: a^4 + 4b^4 可分解\")",
        "python_stdout": "2 32 {2: 5}\n3 145 {5: 1, 29: 1}\n4 512 {2: 9}\n5 1649 {17: 1, 97: 1}\n6 5392 {2: 4, 337: 1}\n7 18785 {5: 1, 13: 1, 17: 2}\n8 69632 {2: 12, 17: 1}\n恒等式检查: a^4 + 4b^4 可分解",
        "python_result": "样例 n=2..8 均出现非平凡因子；恒等式给出一般奇数情形的证明。",
        "references": [
            {
                "rank": 1,
                "similarity": 0.93,
                "source": "竞赛题库 / Sophie Germain 恒等式",
                "problem": "证明 a⁴+4b⁴ 在 a、b 满足适当条件时可分解为两个大于 1 的因子。",
                "solution": "使用 (a²-2ab+2b²)(a²+2ab+2b²) 展开验证。",
            },
            {
                "rank": 2,
                "similarity": 0.86,
                "source": "训练题库 / 奇偶整除",
                "problem": "证明偶数 n>1 时 n⁴+4ⁿ 具有固定的非平凡因子。",
                "solution": "n⁴ 和 4ⁿ 都被 16 整除，而数值大于 16，所以可直接判定为合数。",
            },
        ],
    },
}


EXAMPLES = [
    {
        "id": "integral",
        "difficulty": "基础",
        "title": "定积分",
        "problem": "计算积分 $\\int_0^1 x^2\\,dx$。",
        "plain_problem": "计算积分 ∫₀¹ x² dx。",
        "answer": "$$\\int_0^1 x^2\\,dx = \\left[\\frac{x^3}{3}\\right]_0^1 = \\boxed{\\frac{1}{3}}.$$",
        "summary": "将幂函数积分公式应用到区间端点。",
        "check": "答案为 1/3",
    },
    {
        "id": "optimization",
        "difficulty": "进阶",
        "title": "二元函数最优化",
        "problem": "求 $f(x,y)=x^2+xy+y^2-3x-3y$ 在实数范围内的最小值及取等点。",
        "plain_problem": "求 f(x,y)=x²+xy+y²−3x−3y 在实数范围内的最小值及取等点。",
        "answer": "令 $\\nabla f=(2x+y-3,\\ x+2y-3)=(0,0)$，得到 $x=y=1$。\n\n矩阵 $\\begin{pmatrix}2&1\\1&2\\end{pmatrix}$ 正定，因此这是全局最小点，且 $f(1,1)=\\boxed{-3}$。",
        "summary": "先用梯度求唯一驻点，再用 Hessian 正定性确认全局最小。",
        "check": "最小值 −3，在 x=y=1 取得",
    },
    {
        "id": "number-theory",
        "difficulty": "挑战",
        "title": "数论证明",
        "problem": "证明：对每个整数 $n>1$，$n^4+4^n$ 都是合数。",
        "plain_problem": "证明：对每个整数 n>1，n⁴+4ⁿ 都是合数。",
        "answer": "分奇偶讨论。\n\n当 $n$ 为偶数时，$n^4$ 与 $4^n$ 都被 $16$ 整除，而 $n^4+4^n>16$，所以它有非平凡因子 $16$。\n\n当 $n$ 为奇数时，令 $b=2^{(n-1)/2}$，则 $4^n=4b^4$。由 Sophie Germain 恒等式\n\n$$a^4+4b^4=(a^2-2ab+2b^2)(a^2+2ab+2b^2),$$\n\n取 $a=n$ 即得非平凡分解。因此对所有 $n>1$，$n^4+4^n$ 都是合数。",
        "summary": "偶数使用 16 的整除性，奇数使用 Sophie Germain 恒等式。",
        "check": "证明覆盖奇偶两类",
    },
]


def get_example(example_id: str | None) -> dict | None:
    return next((example for example in EXAMPLES if example["id"] == example_id), None)


# Each entry needs a recording at demo_server/staged/<id>.json, produced by
# ``python -m demo_server.record_staged``.  ``problem`` is what the card sends;
# ``aliases`` are other spellings a presenter may type into the composer.
COMPETITION_EXAMPLES = [
    {
        "id": "cmc7-riemann-sum",
        "difficulty": "中等",
        "title": "数列极限",
        "problem": (
            "求极限 $\\displaystyle\\lim_{n\\to\\infty} n\\left(\\frac{\\sin\\frac{\\pi}{n}}{n^2+1}"
            "+\\frac{\\sin\\frac{2\\pi}{n}}{n^2+2}+\\cdots+\\frac{\\sin\\pi}{n^2+n}\\right)$。"
        ),
        "aliases": [
            "求极限 lim_{n→∞} n(sin(π/n)/(n²+1)+sin(2π/n)/(n²+2)+…+sin π/(n²+n))。",
        ],
        # Checked against the original paper: 一 填空题 (1).
        "source": "第七届全国大学生数学竞赛（非数学类）预赛",
    },
    {
        "id": "cmc9-integral-equation",
        "difficulty": "中等",
        "title": "积分方程",
        "problem": "已知可导函数 $f(x)$ 满足 $f(x)\\cos x+2\\int_0^x f(t)\\sin t\\,\\mathrm{d}t=x+1$，求 $f(x)$。",
        "aliases": [
            "已知可导函数 f(x) 满足 f(x)cos x+2∫₀ˣ f(t)sin t dt=x+1，求 f(x)。",
        ],
        # Edition not yet checked against the original paper, so it is omitted.
        "source": "全国大学生数学竞赛（非数学类）预赛",
    },
]


def replay_events(example: dict) -> list[dict]:
    evidence = STAGE_EVIDENCE.get(example["id"], STAGE_EVIDENCE["integral"])
    references = evidence["references"]
    stages = [
        {
            "node": "input",
            "summary": "收到题目，建立 problem anchor",
            "thinking": {"purpose": "固定原始题目，避免后续阶段误读题面。", "status": "题目已接收"},
            "tools": [{"name": "problem_anchor", "action": "记录题面哈希与原始文本", "status": "completed"}],
        },
        {
            "node": "classifier",
            "summary": f"判定领域：{evidence['category']}；已定位技能目录",
            "thinking": {
                "category": evidence["category"],
                "category_path": evidence["category_path"],
                "skill_document": evidence["skill_document"],
                "validation_script": evidence["validation_script"],
                "question_mode": "computation",
                "confidence": 0.98,
                "signals": evidence["classification_basis"],
            },
            "tools": [{
                "name": "category_candidates",
                "selected": evidence["category"],
                "selected_path": evidence["category_path"],
                "candidates": list(dict.fromkeys([evidence["category"], "数学分析", "离散数学"])),
            }],
        },
        {
            "node": "database_retrieval",
            "summary": "检索相似题 2 条，作为参考证据",
            "thinking": {
                "purpose": "寻找相似题的解法结构，仅作为参考，不直接复制结论。",
                "query": example["plain_problem"],
                "reference_count": len(references),
                "references": references,
            },
            "tools": [{"name": "vector_database_query", "top_k": len(references), "results": references}],
        },
        {
            "node": "reasoning_agent",
            "summary": example["summary"],
            "thinking": {
                "analysis": evidence["reasoning_analysis"],
                "steps": evidence["reasoning_steps"],
                "result": evidence["reasoning_result"],
                "final_answer": example["answer"],
                "validation_points": ["推导覆盖题目所有条件", "结论可由 Python 阶段独立复核"],
            },
            "tools": [{"name": "reasoning_llm", "attempts": 1, "output_schema": ["analysis", "steps", "answer", "validation_points"], "result": evidence["reasoning_result"]}],
        },
        {
            "node": "python_agent",
            "summary": "独立验证通过，未发现候选矛盾",
            "thinking": {
                "purpose": "用独立计算路径复核候选答案。",
                "python_code": evidence["python_code"],
                "stdout": evidence["python_stdout"],
                "result": evidence["python_result"],
                "verification": "通过",
            },
            "tools": [{"name": "python_executor", "backend": "sympy", "success": True, "code": evidence["python_code"], "stdout": evidence["python_stdout"], "stderr": "", "result": evidence["python_result"]}],
        },
        {
            "node": "cross_validator",
            "summary": "推理结果与验证证据一致",
            "thinking": {"comparison": "推理候选与独立验证结果一致", "decision": "接受候选答案"},
            "tools": [{"name": "answer_matcher", "status": "consistent", "contradictions": []}],
        },
        {
            "node": "coordinator",
            "summary": "整理为可读答案并保留公式语义",
            "thinking": {"purpose": "把推导、验证和结论整理成面向读者的回答。", "format": "Markdown + LaTeX"},
            "tools": [{"name": "answer_formatter", "markdown": True, "latex": True}],
        },
    ]
    return [
        {
            **stage,
            "elapsed_s": round(0.25 + index * 0.11, 2),
        }
        for index, stage in enumerate(stages)
    ]
