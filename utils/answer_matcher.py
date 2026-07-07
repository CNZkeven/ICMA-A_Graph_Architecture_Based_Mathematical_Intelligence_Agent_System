"""Answer matching: computation (strict) vs proof (loose). Per 项目计划.md §3.5/§6.3.

等价匹配管线（评委报告问题 3：'E*I*pi' vs 'πie'、'2*sqrt(2)*pi' vs '2π√2' 曾被判
uncertain）：unicode 常量归一化（π√×÷²）→ LaTeX 转 sympy 文本（\\frac \\sqrt \\pi）→
隐式乘法解析 → simplify 差为零或数值（含复数）容差比较。
"""
import re

from utils.cot_stripper import is_placeholder_answer

PROOF_KEYWORDS = ["证明", "prove", "show that", "验证", "说明", "推导", "论证", "判断", "判别", "试证"]

_UNICODE_REPLACEMENTS = [
    ("−", "-"), ("–", "-"), ("—", "-"), ("×", "*"), ("·", "*"), ("⋅", "*"),
    ("÷", "/"), ("²", "**2"), ("³", "**3"), ("½", "(1/2)"), ("∞", "oo"),
]

# 纯"常量乘积"风格短答案（πie、2π√2）：全部字符落在此集合内才走逐字符翻译
_CONST_PRODUCT_CHARS_RE = re.compile(r"^[0-9πiIeE√+\-*/^(). ]{1,40}$")


def _const_product_to_expr(s: str) -> str:
    """把 'πie'/'2π√2' 这类紧凑常量乘积翻译成可解析文本 '(pi)(I)(E)'/'2(pi)sqrt(2)'。"""
    s = re.sub(r"√\s*\(([^()]*)\)", r"#R#(\1)", s)
    s = re.sub(r"√\s*([0-9.]+)", r"#R#(\1)", s)
    out = []
    for part in re.split(r"(#R#)", s):
        if part == "#R#":
            out.append("sqrt")
            continue
        buf = []
        for ch in part:
            if ch == "π":
                buf.append("(pi)")
            elif ch in "iI":
                buf.append("(I)")
            elif ch in "eE":
                buf.append("(E)")
            elif ch == "^":
                buf.append("**")
            else:
                buf.append(ch)
        out.append("".join(buf))
    return "".join(out)


_MATRIX_ENV_RE = re.compile(r"\\begin\{([pbvV]?matrix)\}(.*?)\\end\{\1\}", re.DOTALL)


def _latex_matrix_to_expr(s: str) -> str:
    """\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix} → Matrix([[1,2],[3,4]])。"""
    def conv(m):
        rows = [r.strip() for r in re.split(r"\\\\", m.group(2)) if r.strip()]
        rows_txt = ",".join(
            "[" + ",".join(c.strip() or "0" for c in row.split("&")) + "]" for row in rows)
        return f"Matrix([{rows_txt}])"
    return _MATRIX_ENV_RE.sub(conv, s)


def _latex_to_expr_text(s: str) -> str:
    s = _latex_matrix_to_expr(s)
    s = s.replace("$$", " ").replace("$", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for _ in range(4):
        new = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*\[\s*([^\[\]]+)\s*\]\s*\{([^{}]*)\}", r"((\2)**(1/(\1)))", s)
    for _ in range(4):
        new = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"(sqrt(\1))", s)
        if new == s:
            break
        s = new
    s = re.sub(r"\\sqrt\s*(\d+)", r"(sqrt(\1))", s)
    for cmd in ("operatorname", "mathrm", "text", "mathbf", "boldsymbol"):
        s = re.sub(r"\\%s\s*\{([^{}]*)\}" % cmd, r"\1", s)
    s = (s.replace("\\pi", " pi ").replace("\\cdot", "*").replace("\\times", "*")
         .replace("\\div", "/").replace("\\infty", "oo"))
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc|log|ln|exp|sinh|cosh|tanh|arcsin|arccos|arctan)\b", r"\1", s)
    if "\\" not in s:
        # 剩余的花括号只是分组（如 e^{2x} → e**(2x)）
        s = s.replace("{", "(").replace("}", ")")
    return s


def _normalize_expr_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^(?:最终答案|答案|结论)\s*[：:]\s*", "", s)
    s = s.strip().rstrip("。.，,；;")
    for src, dst in _UNICODE_REPLACEMENTS:
        s = s.replace(src, dst)
    s = re.sub(r"√\s*\(([^()]*)\)", r"(sqrt(\1))", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?|[a-zA-Z])", r"(sqrt(\1))", s)
    if "\\" in s or "$" in s:
        s = _latex_to_expr_text(s)
    s = s.replace("π", "(pi)")
    return s.strip()


def _try_parse_expr(text: str):
    """尽力把答案文本解析成 sympy 表达式；失败返回 None。"""
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations, implicit_multiplication_application,
        convert_xor)
    raw = (text or "").strip()
    if not raw or len(raw) > 400:
        return None
    # 剥离"J = "、"K = "这类命名标签，比较值本身（评委报告 idx=352：标签后接矩阵）
    label_stripped = re.sub(
        r"^\s*\$?\s*[A-Za-z][A-Za-z0-9]{0,3}(?:_\{?\w{1,8}\}?)?\s*=\s*", "", raw, count=1)
    variants = [raw] if label_stripped == raw else [label_stripped, raw]
    candidates = []
    for v in variants:
        if _CONST_PRODUCT_CHARS_RE.match(v) and re.search(r"[πiIeE√]", v):
            candidates.append(_const_product_to_expr(v))
    for v in variants:
        candidates.append(_normalize_expr_text(v))
    candidates.extend(variants)
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    local_dict = {"pi": sp.pi, "e": sp.E, "E": sp.E, "i": sp.I, "I": sp.I, "oo": sp.oo}
    for cand in candidates:
        cand = (cand or "").strip()
        if not cand:
            continue
        try:
            return sp.sympify(cand)
        except Exception:
            pass
        try:
            return parse_expr(cand, transformations=transformations, local_dict=local_dict)
        except Exception:
            continue
    return None


class AnswerMatcher:
    @staticmethod
    def detect_problem_type(problem: str) -> str:
        p = (problem or "").lower()
        for kw in PROOF_KEYWORDS:
            if kw in p:
                return "proof"
        return "computation"

    @staticmethod
    def match_computation_answer(answer1: str, answer2: str, tolerance: float = 1e-6):
        a1 = (answer1 or "").strip()
        a2 = (answer2 or "").strip()
        if is_placeholder_answer(a1) or is_placeholder_answer(a2):
            return False, 0.0, "占位符或空答案，无法判定匹配"
        if a1 == a2 and a1:
            return True, 1.0, "答案字符串完全匹配"
        # numeric
        try:
            v1, v2 = float(a1), float(a2)
            diff = abs(v1 - v2)
            if diff < 1e-10:
                return True, 1.0, "数值完全匹配"
            if diff < tolerance:
                return True, 0.95, f"数值近似匹配，误差{diff:.2e}"
            return False, 0.8, f"数值差异过大: {diff:.2e}"
        except Exception:
            pass
        # sympy symbolic / numeric equivalence (归一化 unicode/LaTeX/隐式乘法后)
        verdict = AnswerMatcher._sympy_equivalent(a1, a2, tolerance)
        if verdict is True:
            return True, 0.98, "符号/数值等价（归一化后）"
        if verdict is False:
            return False, 0.8, "符号表达式不等价"
        sim = AnswerMatcher._string_similarity(a1, a2)
        if sim > 0.9:
            return True, sim, f"高度相似({sim:.2f})"
        if sim > 0.7:
            return False, sim, f"部分相似({sim:.2f})"
        return False, 0.0, f"明显不同({sim:.2f})"

    @staticmethod
    def _sympy_equivalent(a1: str, a2: str, tolerance: float):
        """True=等价 / False=确定不等价 / None=无法判定（走字符串相似度）。"""
        import sympy as sp
        e1, e2 = _try_parse_expr(a1), _try_parse_expr(a2)
        if e1 is None or e2 is None:
            return None
        # 矩阵结构化比较：形状不同或差非零矩阵 → 确定不等价（触发 reconciliation 重试，
        # 而非落入低置信 uncertain 直接采信 Python 答案——评委报告 idx=352 教训）
        is_m1 = isinstance(e1, sp.MatrixBase)
        is_m2 = isinstance(e2, sp.MatrixBase)
        if is_m1 or is_m2:
            if is_m1 and is_m2:
                if e1.shape != e2.shape:
                    return False
                try:
                    return bool(sp.simplify(e1 - e2).is_zero_matrix)
                except Exception:
                    return None
            mat, scalar = (e1, e2) if is_m1 else (e2, e1)
            if mat.shape == (1, 1):
                e1, e2 = mat[0, 0], scalar
            else:
                return False
        diff = None
        try:
            diff = sp.simplify(e1 - e2)
            if diff == 0:
                return True
        except Exception:
            diff = None
        try:
            c1 = complex(sp.N(e1, 30, chop=True))
            c2 = complex(sp.N(e2, 30, chop=True))
            scale = max(1.0, abs(c1), abs(c2))
            return abs(c1 - c2) <= max(tolerance, 1e-9 * scale)
        except Exception:
            pass  # 含自由符号，无法数值比较
        if diff is not None:
            try:
                if not diff.free_symbols and abs(complex(sp.N(diff, 30, chop=True))) > tolerance:
                    return False
            except Exception:
                return None
        return None

    @staticmethod
    def match_proof_answer(reasoning_result, python_result):
        rr = reasoning_result or {}
        po = python_result or {}
        answer = rr.get("answer", "")
        reasoning_complete = (
            bool(rr.get("steps")) and len(rr["steps"]) >= 2
            and bool(answer) and len(answer) > 10 and not is_placeholder_answer(answer)
        )
        python_success = bool(po.get("success")) and len(po.get("stdout", "")) > 0
        if reasoning_complete and python_success:
            return True, 0.85, "证明题：推理完整且Python验证通过"
        if reasoning_complete and not python_success:
            return True, 0.70, "证明题：推理完整，Python验证未通过（证明题可能难以编程验证）"
        if not reasoning_complete and python_success:
            return False, 0.50, "证明题：Python验证通过但推理不完整"
        return False, 0.30, "证明题：推理和验证都不完整"

    @staticmethod
    def match_answers(problem, reasoning_result, python_result):
        ptype = AnswerMatcher.detect_problem_type(problem)
        if ptype == "proof":
            is_match, conf, reason = AnswerMatcher.match_proof_answer(reasoning_result, python_result)
        else:
            ra = (reasoning_result or {}).get("answer", "")
            pa = (python_result or {}).get("answer", "") if python_result else ""
            is_match, conf, reason = AnswerMatcher.match_computation_answer(ra, pa)
        if is_match and conf >= 0.8:
            status = "match"
        elif is_match and conf >= 0.6:
            status = "uncertain"
        elif (not is_match) and conf >= 0.6:
            status = "mismatch"
        else:
            status = "uncertain"
        return {"status": status, "confidence": conf, "reason": reason, "problem_type": ptype}

    @staticmethod
    def _string_similarity(a: str, b: str) -> float:
        try:
            from Levenshtein import ratio
            return ratio(a, b)
        except Exception:
            if not a or not b:
                return 0.0
            s = sum(1 for c in a if c in b)
            return s / max(len(a), len(b))
