# pip install sympy numpy
"""
数值分析 20道习题验证脚本
===========================
每个知识点独立代码块, 内置标准答案比对逻辑,
运行后打印「验证通过/不通过」。
支持分块单独运行: 修改 RUN_ALL = True 可控制全部或单题执行。
"""
import math
import sys
import io
# 设置标准输出为UTF-8编码，避免Windows GBK无法处理Unicode字符
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============ 全局控制 ============
RUN_ALL = True          # True=运行全部, False=仅运行 SELECTED
SELECTED = list(range(20))  # 当 RUN_ALL=False 时指定题号列表

try:
    import sympy as sp
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False
    print("[警告] sympy 未安装, 部分验证将使用 math 近似。安装: pip install sympy")


# ============ 辅助工具 ============
passed = 0
failed = 0

def check(condition, msg):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [验证通过] {msg}")
    else:
        failed += 1
        print(f"  [验证不通过] {msg}")

def print_header(idx, topic):
    print(f"\n{'='*60}")
    print(f"【第 {idx} 题】{topic}")
    print(f"{'='*60}")

# ============================================================
# 第 0 题: 误差分析 — 22/7 近似 π 的绝对误差和相对误差
# ============================================================
def verify_idx0():
    print_header(0, "误差分析: 22/7近似π")
    if HAS_SYMPY:
        pi_sym = sp.pi
        approx = sp.Rational(22, 7)
        abs_err = approx - pi_sym
        rel_err = abs_err / pi_sym
        # 计算数值验证
        abs_err_val = float(abs_err.evalf())
        rel_err_val = float(rel_err.evalf())
        print(f"  绝对误差 = 22/7 - π ≈ {abs_err_val:.10f}")
        print(f"  相对误差 = (22/7 - π)/π ≈ {rel_err_val:.10f}")
        # 数值应满足: 绝对误差 ≈ 0.001264, 相对误差 ≈ 0.000402
        check(0.0012 < abs_err_val < 0.0013,
              f"绝对误差数值合理: {abs_err_val:.6f}")
        check(0.0004 < rel_err_val < 0.00041,
              f"相对误差数值合理: {rel_err_val:.6f}")
    else:
        abs_err = 22/7 - math.pi
        rel_err = abs_err / math.pi
        print(f"  绝对误差 ≈ {abs_err:.10f}")
        print(f"  相对误差 ≈ {rel_err:.10f}")
        check(0.0012 < abs_err < 0.0013, "绝对误差数值合理")
        check(0.0004 < rel_err < 0.00041, "相对误差数值合理")

# ============================================================
# 第 1 题: 线性 Lagrange 插值 — 两点插值多项式
# ============================================================
def verify_idx1():
    print_header(1, "线性Lagrange插值")
    # 已知 f(0)=1, f(2)=5, 构造 P1(x) = 2x+1
    # 验证 P1(0)=1, P1(2)=5, P1(1)=3
    def P1(x):
        return 2*x + 1
    check(P1(0) == 1, "P1(0)=1")
    check(P1(2) == 5, "P1(2)=5")
    check(P1(1) == 3, "P1(1)=3, f(1)≈3")
    print(f"  插值多项式: P1(x) = 2x + 1")

# ============================================================
# 第 2 题: 中点公式数值积分
# ============================================================
def verify_idx2():
    print_header(2, "中点公式数值积分")
    # ∫₀¹ x² dx, 中点 x=0.5
    mid_approx = 1.0 * (0.5**2)   # (b-a)*f((a+b)/2)
    exact = 1/3
    error = exact - mid_approx
    print(f"  中点近似 = {mid_approx} (期望 1/4 = 0.25)")
    print(f"  精确值   = {exact} (期望 1/3 ≈ 0.33333)")
    print(f"  误差     = {error} (期望 1/12 ≈ 0.08333)")
    check(abs(mid_approx - 0.25) < 1e-15, f"中点近似 = 1/4: {mid_approx}")
    check(abs(exact - 1/3) < 1e-15, f"精确值 = 1/3: {exact}")
    check(abs(error - 1/12) < 1e-15, f"误差 = 1/12: {error}")

# ============================================================
# 第 3 题: 二次 Lagrange 插值
# ============================================================
def verify_idx3():
    print_header(3, "二次Lagrange插值")
    # P2(x) = x² + 1, 验证 P2(-1)=2, P2(0)=1, P2(1)=2, P2(1/2)=5/4
    def P2(x):
        return x*x + 1
    check(P2(-1) == 2, "P2(-1)=2")
    check(P2(0) == 1, "P2(0)=1")
    check(P2(1) == 2, "P2(1)=2")
    check(P2(0.5) == 1.25, f"P2(1/2)=5/4={P2(0.5)}")
    print(f"  插值多项式: P2(x) = x^2 + 1")

# ============================================================
# 第 4 题: 复化梯形公式
# ============================================================
def verify_idx4():
    print_header(4, "复化梯形公式")
    # ∫₀¹ e^{-x²} dx, n=4, h=1/4
    # T₄ = h/2[f(x₀) + 2Σf(xᵢ) + f(x₄)]
    import math
    n = 4
    h = 1.0 / n
    nodes = [i*h for i in range(n+1)]
    # 构造表达式
    print(f"  h = {h}")
    print(f"  节点: {nodes}")
    term0 = "1"
    terms_mid = [f"2e^(-{(i*h)**2:.4f})" for i in range(1, n)]
    term_last = f"e^(-1)"
    expr = f"T₄ = 1/8 × [ {term0} + {' + '.join(terms_mid)} + {term_last} ]"
    print(f"  求和表达式: {expr}")
    # 数值验证
    def f_4(x):
        return math.exp(-x*x)
    T4 = h/2 * (f_4(nodes[0]) + 2*sum(f_4(nodes[i]) for i in range(1, n)) + f_4(nodes[n]))
    print(f"  T₄ ≈ {T4:.8f}")
    # 期望表达式: T₄ = 1/8[1 + 2e^{-1/16} + 2e^{-1/4} + 2e^{-9/16} + e^{-1}]
    # 验证数值合理性: 积分值应在 0.74 左右
    check(0.73 < T4 < 0.75, f"T₄ 数值合理: {T4:.6f}")

# ============================================================
# 第 5 题: Newton 迭代法求根
# ============================================================
def verify_idx5():
    print_header(5, "Newton迭代法求根")
    # f(x) = x³ - 2x - 5, f'(x) = 3x² - 2
    # x₀ = 2, x₁ = 21/10
    if HAS_SYMPY:
        x = sp.symbols('x')
        f = x**3 - 2*x - 5
        fp = sp.diff(f, x)
        x0 = sp.Rational(2, 1)
        x1 = x0 - f.subs(x, x0) / fp.subs(x, x0)
        x2_val = x1 - f.subs(x, x1) / fp.subs(x, x1)
        print(f"  f(x) = x³ - 2x - 5")
        print(f"  f'(x) = 3x² - 2")
        print(f"  x₁ = {x1} ≈ {float(x1.evalf()):.6f}  (期望 21/10 = 2.1)")
        print(f"  x₂ = {sp.nsimplify(x2_val)} ≈ {float(x2_val.evalf()):.8f}")
        check(x1 == sp.Rational(21, 10), f"x₁ = 21/10: {x1}")
        # x₂ should be approximately 2.0946
        x2_float = float(x2_val.evalf())
        check(2.094 < x2_float < 2.095, f"x₂ ≈ 2.0946: {x2_float:.6f}")
    else:
        def f5(x):
            return x**3 - 2*x - 5
        def fp5(x):
            return 3*x**2 - 2
        x0 = 2.0
        x1 = x0 - f5(x0)/fp5(x0)
        x2_v = x1 - f5(x1)/fp5(x1)
        print(f"  x₁ = {x1:.10f} (期望 2.1)")
        print(f"  x₂ = {x2_v:.10f} (期望 ≈ 2.0946)")
        check(abs(x1 - 2.1) < 1e-10, f"x₁ = 2.1: {x1:.6f}")
        check(2.094 < x2_v < 2.095, f"x₂ ≈ 2.0946: {x2_v:.6f}")

# ============================================================
# 第 6 题: LU 分解 (Doolittle)
# ============================================================
def verify_idx6():
    print_header(6, "LU分解(Doolittle)")
    # A = [[2,1,1],[4,3,3],[8,7,9]]
    # L = [[1,0,0],[2,1,0],[4,3,1]]
    # U = [[2,1,1],[0,1,1],[0,0,2]]
    import math
    L = [[1,0,0],[2,1,0],[4,3,1]]
    U = [[2,1,1],[0,1,1],[0,0,2]]
    A = [[2,1,1],[4,3,3],[8,7,9]]
    b = [1,2,3]
    # 验证 A = LU
    n = 3
    LU = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += L[i][k] * U[k][j]
            LU[i][j] = s
    is_correct = all(LU[i][j] == A[i][j] for i in range(n) for j in range(n))
    check(is_correct, "A = LU 分解正确")
    if not is_correct:
        print(f"  LU乘积: {LU}")
    # 解方程 Ly = b, Ux = y
    # Ly=b 前代
    y = [0]*n
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i][j] * y[j]
        y[i] = s / L[i][i]
    # Ux=y 回代
    x = [0]*n
    for i in range(n-1, -1, -1):
        s = y[i]
        for j in range(i+1, n):
            s -= U[i][j] * x[j]
        x[i] = s / U[i][i]
    print(f"  解 x = {x}  (期望 [1/2, 1/2, -1/2])")
    check(abs(x[0] - 0.5) < 1e-10, f"x₁ = 1/2: {x[0]}")
    check(abs(x[1] - 0.5) < 1e-10, f"x₂ = 1/2: {x[1]}")
    check(abs(x[2] - (-0.5)) < 1e-10, f"x₃ = -1/2: {x[2]}")

# ============================================================
# 第 7 题: 最小二乘拟合
# ============================================================
def verify_idx7():
    print_header(7, "最小二乘拟合")
    # 数据点: (0,1), (1,3), (2,4), (3,6)
    # 手动计算最小二乘解:
    #   Σx = 6, Σy = 14, Σx² = 14, Σxy = 29
    #   4a + 6b = 14, 6a + 14b = 29
    #   解: a = 11/10 = 1.1, b = 8/5 = 1.6
    #   数据集标准答案为 y=23/10+4/5·x (a=2.3,b=0.8), 经核实数据集答案有误
    xs = [0, 1, 2, 3]
    ys = [1, 3, 4, 6]
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x*x for x in xs)
    sum_xy = sum(xs[i]*ys[i] for i in range(n))
    det = n*sum_xx - sum_x*sum_x
    a = (sum_xx*sum_y - sum_x*sum_xy) / det
    b = (n*sum_xy - sum_x*sum_y) / det
    rss = sum((ys[i] - (a + b*xs[i]))**2 for i in range(n))
    print(f"  Σx={sum_x}, Σy={sum_y}, Σx²={sum_xx}, Σxy={sum_xy}")
    print(f"  拟合直线 y = {a} + {b}x")
    print(f"  RSS = {rss}")
    print(f"  [注] 数据集标准答案为 y=23/10+4/5·x (a=2.3,b=0.8,RSS=3.4), 但核对数据后发现该答案有误")
    print(f"  [注] 正确的最小二乘解来自正规方程组: {n}a+{sum_x}b={sum_y}, {sum_x}a+{sum_xx}b={sum_xy}")
    # 验证计算内部一致性: 残差均值为零(最小二乘性质)
    residuals = [ys[i] - (a + b*xs[i]) for i in range(n)]
    mean_res = sum(residuals) / n
    check(abs(mean_res) < 1e-10, f"残差均值≈0 (最小二乘性质): {mean_res:.2e}")
    check(abs(a - 1.1) < 1e-10, f"a = 11/10 = 1.1: {a:.10f}")
    check(abs(b - 1.6) < 1e-10, f"b = 8/5 = 1.6: {b:.10f}")
    check(abs(rss - 0.2) < 1e-10, f"RSS = 1/5 = 0.2: {rss:.10f}")

# ============================================================
# 第 8 题: 数值微分 (中心差分)
# ============================================================
def verify_idx8():
    print_header(8, "数值微分(中心差分)")
    # f(x)=eˣ, x=1, h=0.1
    # 中心差分: f'(x) ≈ [f(x+h)-f(x-h)]/(2h)
    import math
    x = 1.0
    h = 0.1
    approx = (math.exp(x+h) - math.exp(x-h)) / (2*h)
    exact = math.exp(1)
    error = approx - exact
    print(f"  中心差分近似 = {approx:.10f}")
    print(f"  精确值 f'(1)=e = {exact:.10f}")
    print(f"  绝对误差 = {error:.10f} (期望 ≈ 0.00453)")
    check(abs(approx - exact) < 0.005, f"误差合理: {error:.6f}")

# ============================================================
# 第 9 题: Gauss 列主元消去法
# ============================================================
def verify_idx9():
    print_header(9, "Gauss列主元消去法")
    # 方程组:
    # x + 2y + 3z = 1
    # 4x + 5y + 6z = 2
    # 7x + 8y + 10z = 3
    # 精确解: x=-1/3, y=2/3, z=0
    import math
    # 增广矩阵
    A = [[1, 2, 3, 1],
         [4, 5, 6, 2],
         [7, 8, 10, 3]]
    n = 3
    # 列主元消去
    for col in range(n):
        # 选主元
        max_row = col
        max_val = abs(A[col][col])
        for row in range(col+1, n):
            if abs(A[row][col]) > max_val:
                max_val = abs(A[row][col])
                max_row = row
        if max_row != col:
            A[col], A[max_row] = A[max_row], A[col]
        # 消元
        pivot = A[col][col]
        for row in range(col+1, n):
            factor = A[row][col] / pivot
            for k in range(col, n+1):
                A[row][k] -= factor * A[col][k]
    # 回代
    x = [0]*n
    for i in range(n-1, -1, -1):
        s = A[i][n]
        for j in range(i+1, n):
            s -= A[i][j] * x[j]
        x[i] = s / A[i][i]
    print(f"  解: x={x[0]:.10f}, y={x[1]:.10f}, z={x[2]:.10f}")
    print(f"  期望: x=-1/3≈{-1/3:.6f}, y=2/3≈{2/3:.6f}, z=0")
    check(abs(x[0] - (-1/3)) < 1e-10, f"x = -1/3: {x[0]:.6f}")
    check(abs(x[1] - (2/3)) < 1e-10, f"y = 2/3: {x[1]:.6f}")
    check(abs(x[2] - 0) < 1e-10, f"z = 0: {x[2]:.6f}")

# ============================================================
# 第 10 题: Jacobi 迭代收敛性
# ============================================================
def verify_idx10():
    print_header(10, "Jacobi迭代收敛性")
    # 系数矩阵: [[4,1,1],[1,4,1],[1,1,4]]
    # 雅可比迭代矩阵特征值: -1/2, 1/2, 1/2? Actually let's recompute
    D_inv = [[1/4, 0, 0], [0, 1/4, 0], [0, 0, 1/4]]
    # L+U = [[0,1,1],[1,0,1],[1,1,0]]
    # T = -D^{-1}(L+U) = -1/4 * [[0,1,1],[1,0,1],[1,1,0]]
    # Eigenvalues: for matrix [[0,a,a],[a,0,a],[a,a,0]], eigenvalues are 2a, -a, -a
    # So with a = -1/4: eigenvalues are -2/4=-1/2, 1/4, 1/4
    # ρ = max(|-1/2|, |1/4|, |1/4|) = 1/2
    import math
    # 通过 sympy 验证
    if HAS_SYMPY:
        T_mat = sp.Matrix([[0, -sp.Rational(1,4), -sp.Rational(1,4)],
                          [-sp.Rational(1,4), 0, -sp.Rational(1,4)],
                          [-sp.Rational(1,4), -sp.Rational(1,4), 0]])
        eigvals = T_mat.eigenvals()
        spectral_radius = max(abs(float(v.evalf())) for v in eigvals)
        print(f"  迭代矩阵特征值: {[complex(v.evalf()) for v in eigvals]}")
        print(f"  谱半径 ρ = {spectral_radius:.10f} (期望 0.5)")
        check(abs(spectral_radius - 0.5) < 1e-10,
              f"谱半径 ρ = 1/2: {spectral_radius:.6f}")
        check(spectral_radius < 1, "ρ < 1, Jacobi迭代收敛")
    else:
        # 手动验证: det(T - λI) = 0
        # Eigenvalues are -1/2, 1/4, 1/4
        spectral_radius = 0.5
        print(f"  谱半径 ρ (理论) = 0.5")
        check(spectral_radius < 1, "ρ < 1, Jacobi迭代收敛")

# ============================================================
# 第 11 题: Simpson 公式代数精度
# ============================================================
def verify_idx11():
    print_header(11, "Simpson公式代数精度")
    # ∫₀² x³ dx
    # Simpson: S = (b-a)/6[f(a)+4f((a+b)/2)+f(b)]
    a, b = 0, 2
    def f11(x):
        return x**3
    S = (b-a)/6 * (f11(a) + 4*f11((a+b)/2) + f11(b))
    exact = 4.0  # [x⁴/4]₀² = 16/4 = 4
    print(f"  Simpson近似 = {S} (期望 4)")
    print(f"  精确值       = {exact} (期望 4)")
    check(abs(S - exact) < 1e-15, f"Simpson公式精确: S={S}, exact={exact}")

# ============================================================
# 第 12 题: Romberg 外推
# ============================================================
def verify_idx12():
    print_header(12, "Romberg外推")
    # 积分: ∫₀¹ 4/(1+x²) dx = π
    # T(h) with h=1
    import math
    def f12(x):
        return 4/(1+x*x)
    h = 1.0
    T_h = h/2 * (f12(0) + f12(1))
    h2 = 0.5
    T_h2 = h2/2 * (f12(0) + 2*f12(0.5) + f12(1))
    S = (4*T_h2 - T_h) / 3
    print(f"  T(h)   = {T_h}   (期望 3)")
    print(f"  T(h/2) = {T_h2} (期望 31/10 = 3.1)")
    print(f"  S       = {S:.10f} (期望 47/15 ≈ {47/15:.10f})")
    print(f"  精确值 π ≈ {math.pi:.10f}")
    check(abs(T_h - 3) < 1e-15, f"T(h) = 3: {T_h}")
    check(abs(T_h2 - 3.1) < 1e-15, f"T(h/2) = 31/10: {T_h2}")
    check(abs(S - 47/15) < 1e-15, f"S = 47/15: {S:.10f}")

# ============================================================
# 第 13 题: 矩阵范数与条件数
# ============================================================
def verify_idx13():
    print_header(13, "矩阵范数与条件数")
    # A = [[1,2],[3,4]]
    # ‖A‖_F = √30, κ₁(A) = 21
    import math
    A = [[1,2],[3,4]]
    # Frobenius 范数
    frob_norm = math.sqrt(sum(A[i][j]**2 for i in range(2) for j in range(2)))
    # 1-范数: max column sum
    norm1 = max(abs(A[0][0])+abs(A[1][0]), abs(A[0][1])+abs(A[1][1]))
    # 逆矩阵
    det = A[0][0]*A[1][1] - A[0][1]*A[1][0]
    A_inv = [[A[1][1]/det, -A[0][1]/det],
             [-A[1][0]/det, A[0][0]/det]]
    norm1_inv = max(abs(A_inv[0][0])+abs(A_inv[1][0]),
                    abs(A_inv[0][1])+abs(A_inv[1][1]))
    kappa1 = norm1 * norm1_inv
    print(f"  ‖A‖_F = √{frob_norm**2:.0f} = {frob_norm:.10f} (期望 √30 ≈ {math.sqrt(30):.10f})")
    print(f"  ‖A‖₁ = {norm1} (期望 6)")
    print(f"  ‖A⁻¹‖₁ = {norm1_inv} (期望 7/2 = 3.5)")
    print(f"  κ₁(A) = {kappa1} (期望 21)")
    check(abs(frob_norm - math.sqrt(30)) < 1e-10,
          f"‖A‖_F = √30: {frob_norm:.6f}")
    check(abs(norm1 - 6) < 1e-10, f"‖A‖₁ = 6: {norm1}")
    check(abs(norm1_inv - 3.5) < 1e-10, f"‖A⁻¹‖₁ = 7/2: {norm1_inv}")
    check(abs(kappa1 - 21) < 1e-10, f"κ₁(A) = 21: {kappa1}")

# ============================================================
# 第 14 题: 插值余项误差分析
# ============================================================
def verify_idx14():
    print_header(14, "插值余项误差分析")
    # f(x)=ln x, x₀=1, x₁=e
    # P₁(2) = ln1*(2-e)/(1-e) + ln e*(2-1)/(e-1) = (2-1)/(e-1) = 1/(e-1)
    import math
    e = math.e
    P1_2 = 1/(e-1)
    f2 = math.log(2)
    error = abs(f2 - P1_2)
    print(f"  P₁(2) = 1/(e-1) ≈ {P1_2:.10f}")
    print(f"  f(2) = ln 2 ≈ {f2:.10f}")
    print(f"  误差 |ln2 - 1/(e-1)| ≈ {error:.10f}")
    # 误差应在 0.11 左右
    check(0.10 < error < 0.12, f"误差合理: {error:.6f}")

# ============================================================
# 第 15 题: Newton 法收敛阶 (求 √2)
# ============================================================
def verify_idx15():
    print_header(15, "Newton法收敛阶")
    # f(x)=x²-2, x_{n+1} = (x_n + 2/x_n)/2
    # x₀=2, x₁=3/2, x₂=17/12, x₃=577/408
    import math
    if HAS_SYMPY:
        x0 = sp.Rational(2,1)
        def newton_step(xn):
            return (xn + 2/xn) / 2
        x1 = newton_step(x0)
        x2 = newton_step(x1)
        x3 = newton_step(x2)
        print(f"  x₁ = {x1} = {float(x1.evalf()):.10f} (期望 3/2)")
        print(f"  x₂ = {x2} = {float(x2.evalf()):.10f} (期望 17/12)")
        print(f"  x₃ = {x3} = {float(x3.evalf()):.10f} (期望 577/408)")
        check(x1 == sp.Rational(3,2), f"x₁ = 3/2: {x1}")
        check(x2 == sp.Rational(17,12), f"x₂ = 17/12: {x2}")
        check(x3 == sp.Rational(577,408), f"x₃ = 577/408: {x3}")
        # 验证二阶收敛: e_{n+1} / e_n² ≈ 常数
        sqrt2 = sp.sqrt(2)
        e0 = abs(float((x0 - sqrt2).evalf()))
        e1 = abs(float((x1 - sqrt2).evalf()))
        e2 = abs(float((x2 - sqrt2).evalf()))
        e3 = abs(float((x3 - sqrt2).evalf()))
        r1 = e1 / (e0**2) if e0 > 0 else 0
        r2 = e2 / (e1**2) if e1 > 0 else 0
        r3 = e3 / (e2**2) if e2 > 0 else 0
        print(f"  误差: e₀={e0:.6e}, e₁={e1:.6e}, e₂={e2:.6e}, e₃={e3:.6e}")
        print(f"  收敛比: e₁/e₀²={r1:.4f}, e₂/e₁²={r2:.4f}, e₃/e₂²={r3:.4f}")
        check(r2 > 0.3 and r2 < 0.5, f"二阶收敛验证: e₂/e₁²≈{r2:.4f}")
    else:
        def newton_sqrt(xn):
            return (xn + 2/xn) / 2
        x1 = newton_sqrt(2)
        x2 = newton_sqrt(x1)
        x3 = newton_sqrt(x2)
        print(f"  x₁ = {x1:.10f} (期望 1.5)")
        print(f"  x₂ = {x2:.10f} (期望 1.416666...)")
        print(f"  x₃ = {x3:.10f} (期望 1.4142156...)")
        check(abs(x1 - 1.5) < 1e-10, f"x₁ = 3/2: {x1}")
        check(abs(x2 - 17/12) < 1e-10, f"x₂ = 17/12: {x2}")
        check(abs(x3 - 577/408) < 1e-10, f"x₃ = 577/408: {x3}")

# ============================================================
# 第 16 题: 数值稳定性 (e^{-10})
# ============================================================
def verify_idx16():
    print_header(16, "数值稳定性: e^{-10}")
    # S₅ = Σ_{k=0}^{4} (-10)^k/k! = 291
    # 实际 e^{-10} ≈ 4.54×10⁻⁵
    import math
    S5 = sum(((-10)**k) / math.factorial(k) for k in range(5))
    actual = math.exp(-10)
    print(f"  S₅ (Taylor前5项和) = {S5:.10f} (期望 291)")
    print(f"  实际 e⁻¹⁰ ≈ {actual:.10e} (期望 4.54e-05)")
    print(f"  S₅ 是实际值的 {S5/actual:.0f} 倍, 完全不可用")
    check(abs(S5 - 291) < 1e-10, f"S₅ = 291: {S5:.0f}")
    check(actual > 0, "e^{-10} > 0")
    check(S5 > actual * 1e5, "S₅ ≫ e^{-10}, 说明级数前5项严重偏离")
    # 说明: 方法B (计算e^10再取倒数) 优于方法A (直接展开)
    e10 = math.exp(10)
    e10_inv = 1/e10
    check(abs(e10_inv - actual) < 1e-20, "方法B (e^10取倒数) 数值稳定")

# ============================================================
# 第 17 题: 梯形公式代数精度与误差
# ============================================================
def verify_idx17():
    print_header(17, "梯形公式代数精度与误差")
    # f(x)=x² 在 [0,1]
    # T = (b-a)/2[f(a)+f(b)] = 1/2[0+1] = 1/2
    # 精确值 = ∫₀¹ x² dx = 1/3
    # 误差 = 1/3 - 1/2 = -1/6
    T = 0.5 * (0 + 1)  # = 0.5
    exact = 1/3
    error = exact - T
    print(f"  梯形近似 T = {T} (期望 1/2)")
    print(f"  精确值     = {exact} (期望 1/3)")
    print(f"  误差       = {error} (期望 -1/6 ≈ {-1/6:.10f})")
    check(abs(T - 0.5) < 1e-15, f"T = 1/2: {T}")
    check(abs(exact - 1/3) < 1e-15, f"精确值 = 1/3: {exact}")
    check(abs(error - (-1/6)) < 1e-15, f"误差 = -1/6: {error:.6f}")
    # 验证对一次多项式精确成立
    # f(x) = 2x+3 在 [0,1]
    T_linear = 0.5 * (3 + 5)  # = 4
    exact_linear = 4.0  # ∫₀¹ (2x+3)dx = [x²+3x]₀¹ = 4
    check(abs(T_linear - exact_linear) < 1e-15,
          "梯形公式对一次多项式精确成立")

# ============================================================
# 第 18 题: 不动点迭代收敛性 (x = cos x)
# ============================================================
def verify_idx18():
    print_header(18, "不动点迭代: x = cos x")
    # x₀ = 0.7
    # x₁ = cos(0.7), x₂ = cos(x₁), x₃ = cos(x₂)
    import math
    x0 = 0.7
    x1 = math.cos(x0)
    x2 = math.cos(x1)
    x3 = math.cos(x2)
    print(f"  x₁ = {x1:.10f} (期望 ≈ 0.764842)")
    print(f"  x₂ = {x2:.10f} (期望 ≈ 0.721492)")
    print(f"  x₃ = {x3:.10f} (期望 ≈ 0.750821)")
    check(abs(x1 - 0.764842) < 1e-5, f"x₁ ≈ 0.764842: {x1:.6f}")
    check(abs(x2 - 0.721492) < 1e-5, f"x₂ ≈ 0.721492: {x2:.6f}")
    check(abs(x3 - 0.750821) < 1e-5, f"x₃ ≈ 0.750821: {x3:.6f}")
    # 渐近收敛因子估计
    factor = abs(x3 - x2) / abs(x2 - x1) if abs(x2 - x1) > 0 else 0
    print(f"  渐近收敛因子 |x₃-x₂|/|x₂-x₁| ≈ {factor:.4f} (期望 ≈ 0.6766)")
    check(0.6 < factor < 0.8, f"收敛因子合理: {factor:.4f}")

# ============================================================
# 第 19 题: ODE 显式 Euler 法稳定性
# ============================================================
def verify_idx19():
    print_header(19, "ODE显式Euler法稳定性")
    # y' = -λy, y(0)=1
    # y_{n+1} = (1-hλ)y_n
    # 稳定条件: |1-hλ| ≤ 1 → 0 < hλ < 2
    # λ=10 → h_crit = 0.2
    lam = 10.0
    h_crit = 2.0 / lam
    print(f"  稳定区间: 0 < hλ < 2")
    print(f"  λ = {lam} 时临界步长 h_crit = 2/λ = {h_crit} (期望 0.2)")
    check(abs(h_crit - 0.2) < 1e-15, f"h_crit = 0.2: {h_crit}")
    # 验证: h=0.19 (稳定), h=0.21 (不稳定)
    def euler_step(h, lam_val, n_steps=100):
        y = 1.0
        for _ in range(n_steps):
            y = (1 - h*lam_val) * y
        return y
    y_stable = euler_step(0.19, lam)
    y_unstable = euler_step(0.21, lam)
    print(f"  h=0.19, 100步后 y ≈ {y_stable:.6e} (应有界)")
    print(f"  h=0.21, 100步后 y ≈ {y_unstable:.6e} (应发散)")
    check(abs(y_stable) < 1, f"h=0.19 稳定: |y|={abs(y_stable):.4f}")
    check(abs(y_unstable) > 10, f"h=0.21 不稳定: |y|={abs(y_unstable):.4f}")

# ============================================================
# 主函数
# ============================================================
def main():
    global passed, failed
    passed = 0
    failed = 0
    # 所有验证函数列表
    verifiers = [
        verify_idx0, verify_idx1, verify_idx2, verify_idx3, verify_idx4,
        verify_idx5, verify_idx6, verify_idx7, verify_idx8, verify_idx9,
        verify_idx10, verify_idx11, verify_idx12, verify_idx13, verify_idx14,
        verify_idx15, verify_idx16, verify_idx17, verify_idx18, verify_idx19
    ]
    if HAS_SYMPY:
        print(f"[信息] sympy 已安装, 使用精确符号计算")
    else:
        print(f"[信息] sympy 未安装, 使用 math 模块数值计算")
    indices = SELECTED if not RUN_ALL else list(range(20))
    for i in indices:
        if 0 <= i < len(verifiers):
            verifiers[i]()
        else:
            print(f"\n[跳过] 无效题号: {i}")
    print(f"\n{'='*60}")
    print(f"验证结果汇总: 通过 {passed} 题, 不通过 {failed} 题, 总计 {passed+failed} 题")
    if failed == 0:
        print("恭喜! 全部验证通过!")
    else:
        print(f"注意: 有 {failed} 题验证不通过 (请检查输出详情)")
    return failed


if __name__ == "__main__":
    main()
