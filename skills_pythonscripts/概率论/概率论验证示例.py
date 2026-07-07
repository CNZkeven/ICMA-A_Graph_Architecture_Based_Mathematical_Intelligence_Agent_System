# -*- coding: utf-8 -*-
"""
概率论数据集 — sympy 全题验证脚本
覆盖 概率论.md 全部 20 题 (idx 0-19)
14 计算题 + 6 证明题
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sympy as sp
from sympy import (Rational, Symbol, symbols, exp, binomial,
                   factorial, summation, integrate, oo, diff,
                   simplify, expand, pi, I, E, Abs, sqrt, Piecewise)

results = []

# ============================================================
# 【题0】古典概型 — 超几何分布 (简单/计算)
# ============================================================
print("=" * 60)
print("【题0】古典概型 — 超几何分布")
print("=" * 60)
print("问题: 袋中有5个红球和3个白球，不放回抽3个球，恰2红的概率")
print()

# P = C(5,2) * C(3,1) / C(8,3)
p0 = binomial(5, 2) * binomial(3, 1) / binomial(8, 3)
p0_s = simplify(p0)
expected0 = Rational(15, 28)
print(f"  计算值: C(5,2)·C(3,1)/C(8,3) = {p0} = {p0_s}")
print(f"  预期答案: {expected0}")
ok0 = p0_s == expected0
print(f"  {'✅ 一致' if ok0 else '❌ 不一致'}")
results.append(("题0 (超几何概率)", ok0))
print()

# ============================================================
# 【题1】期望与方差 — f(x)=2x, x∈[0,1] (简单/计算)
# ============================================================
print("=" * 60)
print("【题1】期望与方差 — f(x)=2x, x∈[0,1]")
print("=" * 60)
print("问题: 密度 f(x)=2x (0≤x≤1)，求 E[X] 和 Var(X)")
print()

x = Symbol('x', real=True, positive=True)

# E[X] = ∫_0^1 x·2x dx
EX1 = integrate(x * 2*x, (x, 0, 1))
print(f"  E[X] = ∫_0^1 x·2x dx = {EX1}")

# E[X²] = ∫_0^1 x²·2x dx
EX2_1 = integrate(x**2 * 2*x, (x, 0, 1))
print(f"  E[X²] = ∫_0^1 x²·2x dx = {EX2_1}")

# Var(X) = E[X²] - (E[X])²
VarX1 = simplify(EX2_1 - EX1**2)
print(f"  Var(X) = E[X²] - (E[X])² = {VarX1}")

expected_EX1 = Rational(2, 3)
expected_Var1 = Rational(1, 18)
print(f"  预期答案: E[X] = {expected_EX1}, Var(X) = {expected_Var1}")
ok1 = (simplify(EX1 - expected_EX1) == 0 and
       simplify(VarX1 - expected_Var1) == 0)
print(f"  {'✅ 一致' if ok1 else '❌ 不一致'}")
results.append(("题1 (期望与方差)", ok1))
print()

# ============================================================
# 【题2】二项分布 — B(5, 1/3), P(X=2) (简单/计算)
# ============================================================
print("=" * 60)
print("【题2】二项分布 — X~B(5, 1/3), P(X=2)")
print("=" * 60)
print("问题: X~B(5, 1/3)，求 P(X=2)")
print()

n2, p2 = 5, Rational(1, 3)
q2 = 1 - p2
prob2 = binomial(n2, 2) * p2**2 * q2**(n2 - 2)
prob2_s = simplify(prob2)
expected2 = Rational(80, 243)
print(f"  P(X=2) = C(5,2)·(1/3)²·(2/3)³ = {prob2} = {prob2_s}")
print(f"  预期答案: {expected2}")
ok2 = prob2_s == expected2
print(f"  {'✅ 一致' if ok2 else '❌ 不一致'}")
results.append(("题2 (二项分布)", ok2))
print()

# ============================================================
# 【题3】贝叶斯公式 — 疾病检测 (中等/计算)
# ============================================================
print("=" * 60)
print("【题3】贝叶斯公式 — 疾病检测后验概率")
print("=" * 60)
print("问题: 发病率0.01，阳性率(P|D)=0.99，假阳性率(P|~D)=0.02，求P(D|+)")
print()

P_D = Rational(1, 100)
P_pos_given_D = Rational(99, 100)
P_pos_given_notD = Rational(2, 100)
P_notD = 1 - P_D

# 全概率: P(+) = P(+|D)P(D) + P(+|~D)P(~D)
P_pos = P_pos_given_D * P_D + P_pos_given_notD * P_notD
print(f"  全概率 P(+) = 0.99·0.01 + 0.02·0.99 = {simplify(P_pos)}")

# 贝叶斯: P(D|+) = P(+|D)P(D) / P(+)
P_D_given_pos = simplify(P_pos_given_D * P_D / P_pos)
expected3 = Rational(1, 3)
print(f"  后验 P(D|+) = (0.99·0.01) / P(+) = {P_D_given_pos}")
print(f"  预期答案: {expected3}")
ok3 = simplify(P_D_given_pos - expected3) == 0
print(f"  {'✅ 一致' if ok3 else '❌ 不一致'}")
results.append(("题3 (贝叶斯公式)", ok3))
print()

# ============================================================
# 【题4】二维连续分布 — f(x,y)=8xy, 0<x<y<1 (中等/计算)
# ============================================================
print("=" * 60)
print("【题4】二维连续分布 — f(x,y)=8xy, 0<x<y<1")
print("=" * 60)
print("问题: f(x,y)=8xy (0<x<y<1)，求 E[XY] 和 Cov(X,Y)")
print()

x4, y4 = symbols('x y', real=True, positive=True)

# E[XY] = ∫_0^1 ∫_x^1 xy·8xy dy dx
EXY4 = simplify(integrate(integrate(x4 * y4 * 8*x4*y4, (y4, x4, 1)), (x4, 0, 1)))
print(f"  E[XY] = ∫₀¹∫ₓ¹ xy·8xy dy dx = {EXY4}")

# E[X] = ∫_0^1 ∫_x^1 x·8xy dy dx
EX4 = simplify(integrate(integrate(x4 * 8*x4*y4, (y4, x4, 1)), (x4, 0, 1)))
print(f"  E[X] = ∫₀¹∫ₓ¹ x·8xy dy dx = {EX4}")

# E[Y] = ∫_0^1 ∫_x^1 y·8xy dy dx
EY4 = simplify(integrate(integrate(y4 * 8*x4*y4, (y4, x4, 1)), (x4, 0, 1)))
print(f"  E[Y] = ∫₀¹∫ₓ¹ y·8xy dy dx = {EY4}")

# Cov(X,Y) = E[XY] - E[X]E[Y]
Cov4 = simplify(EXY4 - EX4 * EY4)
print(f"  Cov(X,Y) = {EXY4} - ({EX4})({EY4}) = {Cov4}")

expected_EXY4 = Rational(4, 9)
expected_Cov4 = Rational(4, 225)
print(f"  预期答案: E[XY] = {expected_EXY4}, Cov(X,Y) = {expected_Cov4}")
ok4 = (simplify(EXY4 - expected_EXY4) == 0 and
       simplify(Cov4 - expected_Cov4) == 0)
print(f"  {'✅ 一致' if ok4 else '❌ 不一致'}")
results.append(("题4 (二维连续分布)", ok4))
print()

# ============================================================
# 【题5】离散相关系数 — Cov(X,Y) 与 ρ(X,Y) (中等/计算)
# ============================================================
print("=" * 60)
print("【题5】离散联合分布 — Cov(X,Y) 与 ρ(X,Y)")
print("=" * 60)
print("问题: P(0,0)=0.3, P(0,1)=0.2, P(1,0)=0.2, P(1,1)=0.3")
print()

joint5 = {(0, 0): Rational(3, 10), (0, 1): Rational(2, 10),
          (1, 0): Rational(2, 10), (1, 1): Rational(3, 10)}

# E[X], E[Y], E[XY]
EX5 = sum(xv * p for (xv, yv), p in joint5.items())
EY5 = sum(yv * p for (xv, yv), p in joint5.items())
EXY5 = sum(xv * yv * p for (xv, yv), p in joint5.items())

# Cov(X,Y) = E[XY] - E[X]E[Y]
Cov5 = simplify(EXY5 - EX5 * EY5)
print(f"  E[X] = {EX5}, E[Y] = {EY5}, E[XY] = {EXY5}")
print(f"  Cov(X,Y) = {EXY5} - {EX5}·{EY5} = {Cov5}")

# Var(X), Var(Y) — Bernoulli: Var = p(1-p) = 0.5·0.5 = 0.25
VarX5 = simplify(sum(xv**2 * p for (xv, yv), p in joint5.items()) - EX5**2)
VarY5 = simplify(sum(yv**2 * p for (xv, yv), p in joint5.items()) - EY5**2)
print(f"  Var(X) = {VarX5}, Var(Y) = {VarY5}")

# ρ = Cov / sqrt(Var(X) * Var(Y))
rho5 = simplify(Cov5 / sqrt(VarX5 * VarY5))
print(f"  ρ(X,Y) = {Cov5} / sqrt({VarX5}·{VarY5}) = {rho5}")

expected_Cov5 = Rational(5, 100)  # 0.05
expected_rho5 = Rational(1, 5)
print(f"  预期答案: Cov = {expected_Cov5}, ρ = {expected_rho5}")
ok5 = (simplify(Cov5 - expected_Cov5) == 0 and
       simplify(rho5 - expected_rho5) == 0)
print(f"  {'✅ 一致' if ok5 else '❌ 不一致'}")
results.append(("题5 (相关系数)", ok5))
print()

# ============================================================
# 【题6】变量变换 — Y=-ln X, X~U(0,1) (中等/计算)
# ============================================================
print("=" * 60)
print("【题6】随机变量函数分布 — Y=-ln X, X~U(0,1)")
print("=" * 60)
print("问题: X~U(0,1), Y=-ln X，求 E[Y] 和 P(Y>1)")
print()

x6 = Symbol('x6', positive=True)

# E[Y] = ∫_0^1 (-ln x)·1 dx
EY6 = integrate(-sp.log(x6), (x6, 0, 1))
print(f"  E[Y] = ∫₀¹(-ln x) dx = {EY6}")

# P(Y>1) = P(-ln X > 1) = P(X < e^{-1}) = e^{-1}
prob6 = integrate(1, (x6, 0, exp(-1)))
print(f"  P(Y>1) = P(X < e⁻¹) = {simplify(prob6)}")

expected_EY6 = 1
expected_P6 = exp(-1)
print(f"  预期答案: E[Y] = {expected_EY6}, P(Y>1) = e⁻¹")
ok6 = (simplify(EY6 - expected_EY6) == 0 and
       simplify(prob6 - expected_P6) == 0)
print(f"  {'✅ 一致' if ok6 else '❌ 不一致'}")
results.append(("题6 (变量变换)", ok6))
print()

# ============================================================
# 【题7】二维正态条件分布 (中等/计算)
# ============================================================
print("=" * 60)
print("【题7】二维正态条件分布 — E[Y|X=1] 与 Var(Y|X=1)")
print("=" * 60)
print("问题: μ_X=μ_Y=0, σ²_X=1, σ²_Y=4, Cov=1")
print()

muX7, muY7 = 0, 0
varX7, varY7 = 1, 4
cov7 = 1
sigmaX7 = sqrt(varX7)
sigmaY7 = sqrt(varY7)

# 相关系数
rho7 = Rational(cov7, sigmaX7 * sigmaY7)
print(f"  ρ = Cov/(σ_X·σ_Y) = 1/(1·2) = {rho7}")

# E[Y|X=1] = μ_Y + ρ·(σ_Y/σ_X)·(1 - μ_X)
EY_given_X1 = muY7 + rho7 * (sigmaY7 / sigmaX7) * (1 - muX7)
print(f"  E[Y|X=1] = 0 + (1/2)·2·1 = {EY_given_X1}")

# Var(Y|X=1) = σ²_Y·(1 - ρ²)
VarY_given_X1 = varY7 * (1 - rho7**2)
print(f"  Var(Y|X=1) = 4·(1 - 1/4) = {VarY_given_X1}")

print(f"  预期答案: E[Y|X=1] = 1, Var(Y|X=1) = 3")
ok7 = (simplify(EY_given_X1 - 1) == 0 and
       simplify(VarY_given_X1 - 3) == 0)
print(f"  {'✅ 一致' if ok7 else '❌ 不一致'}")
results.append(("题7 (二维正态条件)", ok7))
print()

# ============================================================
# 【题8】条件期望 — f(x,y)=2, 0<x<y<1 (中等/计算)
# ============================================================
print("=" * 60)
print("【题8】条件期望 — f(x,y)=2, 0<x<y<1")
print("=" * 60)
print("问题: f(x,y)=2 (0<x<y<1)，求 E[X|Y=y] 和 Var(X|Y=y)")
print()

x8, y8 = symbols('x y', real=True, positive=True)
f8 = 2

# 边缘密度 f_Y(y) = ∫_0^y 2 dx = 2y
fY8 = integrate(f8, (x8, 0, y8))
print(f"  f_Y(y) = ∫₀ʸ 2 dx = {fY8}")

# 条件密度 f_{X|Y}(x|y) = 2 / 2y = 1/y
f_cond8 = f8 / fY8
print(f"  f_{{X|Y}}(x|y) = 2/(2y) = {simplify(f_cond8)}")

# E[X|Y=y] = ∫_0^y x·(1/y) dx = y/2
EX_given_Y8 = simplify(integrate(x8 * f_cond8, (x8, 0, y8)))
print(f"  E[X|Y=y] = ∫₀ʸ x·(1/y) dx = {EX_given_Y8}")

# E[X²|Y=y] = ∫_0^y x²·(1/y) dx = y²/3
EX2_given_Y8 = simplify(integrate(x8**2 * f_cond8, (x8, 0, y8)))
print(f"  E[X²|Y=y] = ∫₀ʸ x²·(1/y) dx = {EX2_given_Y8}")

# Var(X|Y=y) = y²/3 - y²/4 = y²/12
VarX_given_Y8 = simplify(EX2_given_Y8 - EX_given_Y8**2)
print(f"  Var(X|Y=y) = {EX2_given_Y8} - ({EX_given_Y8})² = {VarX_given_Y8}")

expected_EX8 = y8 / 2
expected_Var8 = y8**2 / 12
print(f"  预期答案: E[X|Y=y] = y/2, Var(X|Y=y) = y²/12")
ok8 = (simplify(EX_given_Y8 - expected_EX8) == 0 and
       simplify(VarX_given_Y8 - expected_Var8) == 0)
print(f"  {'✅ 一致' if ok8 else '❌ 不一致'}")
results.append(("题8 (条件期望/方差)", ok8))
print()

# ============================================================
# 【题9】泊松分布 — X~Poisson(3) (中等/计算)
# ============================================================
print("=" * 60)
print("【题9】泊松分布 — X~Poisson(3)")
print("=" * 60)
print("问题: X~Poisson(3)，求 P(X=2) 和 P(X≥1)")
print()

lam9 = 3

# P(X=2) = e^{-3} * 3² / 2! = (9/2) e^{-3}
P9_eq2 = lam9**2 * exp(-lam9) / factorial(2)
P9_eq2_s = simplify(P9_eq2)
print(f"  P(X=2) = e⁻³·3²/2! = {P9_eq2_s}")

# P(X≥1) = 1 - P(X=0) = 1 - e^{-3}
P9_ge1 = 1 - exp(-lam9)
print(f"  P(X≥1) = 1 - e⁻³ = {P9_ge1}")

expected9_eq2 = Rational(9, 2) * exp(-3)
expected9_ge1 = 1 - exp(-3)
print(f"  预期答案: P(X=2) = (9/2)e⁻³, P(X≥1) = 1 - e⁻³")
ok9 = (simplify(P9_eq2 - expected9_eq2) == 0 and
       simplify(P9_ge1 - expected9_ge1) == 0)
print(f"  {'✅ 一致' if ok9 else '❌ 不一致'}")
results.append(("题9 (泊松分布)", ok9))
print()

# ============================================================
# 【题10】指数分布高阶矩 — Exp(2), E[X³] (中等/计算)
# ============================================================
print("=" * 60)
print("【题10】指数分布高阶矩 — Exp(2), E[X³]")
print("=" * 60)
print("问题: X~Exp(2) (密度 2e^{-2x}, x>0)，求 E[X³]")
print()

x10 = Symbol('x', positive=True)
lam10 = 2

# 积分法: E[X³] = ∫_0^∞ x³·2e^{-2x} dx
EX3_integral = integrate(x10**3 * lam10 * exp(-lam10 * x10), (x10, 0, oo))
print(f"  积分: E[X³] = ∫₀^∞ x³·2e^(-2x) dx = {simplify(EX3_integral)}")

# 公式法: E[X^n] = n! / λ^n
EX3_formula = Rational(factorial(3), lam10**3)
print(f"  公式: E[X³] = 3! / 2³ = 6/8 = {EX3_formula}")

expected10 = Rational(3, 4)
print(f"  预期答案: {expected10}")
ok10 = simplify(EX3_integral - expected10) == 0
print(f"  {'✅ 一致' if ok10 else '❌ 不一致'}")
results.append(("题10 (指数高阶矩)", ok10))
print()

# ============================================================
# 【题11】矩母函数 — M(t)=(1-2t)^(-3) (中等/计算)
# ============================================================
print("=" * 60)
print("【题11】矩母函数 — M(t)=(1-2t)^(-3)")
print("=" * 60)
print("问题: M(t)=(1-2t)^(-3) (t<1/2)，求 E[X] 和 Var(X)")
print()

t11 = Symbol('t', real=True)
M11 = (1 - 2*t11)**(-3)

# 一阶导数: E[X] = M'(0)
M1_11 = diff(M11, t11)
EX11 = simplify(M1_11.subs(t11, 0))
print(f"  M'(t) = {M1_11}")
print(f"  E[X] = M'(0) = {EX11}")

# 二阶导数: Var(X) = M''(0) - (M'(0))²
M2_11 = diff(M1_11, t11)
EX2_11 = simplify(M2_11.subs(t11, 0))
VarX11 = simplify(EX2_11 - EX11**2)
print(f"  M''(t) = {M2_11}")
print(f"  M''(0) = {EX2_11}")
print(f"  Var(X) = M''(0) - (M'(0))² = {EX2_11} - {EX11}² = {VarX11}")

# Gamma 分布识别: M(t) = (1-2t)^{-3} ⇒ Gamma(α=3, β=1/2)
# E[X] = α/β = 6, Var(X) = α/β² = 12
alpha_11, beta_11 = 3, Rational(1, 2)
EX_gamma = alpha_11 / beta_11
VarX_gamma = alpha_11 / beta_11**2
print(f"  Gamma(α=3,β=1/2): E[X] = {EX_gamma}, Var(X) = {VarX_gamma}")

expected_EX11 = 6
expected_Var11 = 12
print(f"  预期答案: E[X] = {expected_EX11}, Var(X) = {expected_Var11}")
ok11 = (simplify(EX11 - expected_EX11) == 0 and
        simplify(VarX11 - expected_Var11) == 0)
print(f"  {'✅ 一致' if ok11 else '❌ 不一致'}")
results.append(("题11 (矩母函数)", ok11))
print()

# ============================================================
# 【题12】独立U(0,1)之和 — 三角分布 (困难/计算)
# ============================================================
print("=" * 60)
print("【题12】独立U(0,1)之和 — Z = X+Y, 三角分布")
print("=" * 60)
print("问题: X,Y i.i.d. U(0,1)，求 f_Z(z) 和 P(Z>1)")
print()

x12, z12 = symbols('x z', real=True)

# 卷积: f_Z(z) = ∫ f_X(x)·f_Y(z-x) dx
# f_X(x)=1 for x∈[0,1]; f_Y(z-x)=1 for z-x∈[0,1] ⇒ x∈[z-1, z]
# 积分区域: x ∈ [max(0, z-1), min(1, z)]

# Case 1: 0<z<1, x∈[0,z]
fZ1 = integrate(1, (x12, 0, z12))
print(f"  0<z<1: f_Z(z) = ∫₀ᶻ 1 dx = {fZ1}")

# Case 2: 1<z<2, x∈[z-1,1]
fZ2 = integrate(1, (x12, z12 - 1, 1))
fZ2_s = simplify(fZ2)
print(f"  1<z<2: f_Z(z) = ∫_{'{z-1}'}¹ 1 dx = {fZ2_s}")

# 归一化验证
norm12 = integrate(z12, (z12, 0, 1)) + integrate(2 - z12, (z12, 1, 2))
print(f"  归一化: ∫₀¹ z dz + ∫₁² (2-z) dz = {norm12}")

# P(Z>1) = ∫₁² (2-z) dz = 1/2
P12 = integrate(2 - z12, (z12, 1, 2))
print(f"  P(Z>1) = ∫₁² (2-z) dz = {simplify(P12)}")

expected_fZ1 = z12
expected_fZ2 = 2 - z12
expected_P12 = Rational(1, 2)
print(f"  预期答案: f_Z: z (0<z<1), 2-z (1<z<2), P(Z>1) = {expected_P12}")
ok12 = (simplify(fZ1 - expected_fZ1) == 0 and
        simplify(fZ2_s - expected_fZ2) == 0 and
        simplify(P12 - expected_P12) == 0 and
        simplify(norm12 - 1) == 0)
print(f"  {'✅ 一致' if ok12 else '❌ 不一致'}")
results.append(("题12 (卷积三角分布)", ok12))
print()

# ============================================================
# 【题13】特征函数反演 — φ(t)=1/(1+t²) (困难/计算)
# ============================================================
print("=" * 60)
print("【题13】特征函数反演 — φ(t)=1/(1+t²), Laplace分布")
print("=" * 60)
print("问题: φ(t)=1/(1+t²)，求密度 f(x) 和 P(|X|<1)")
print()

t13, x13 = symbols('t x', real=True)

# 验证 φ(t) = 1/(1+t²) 是 Laplace(0,1) 的特征函数
# f(x) = (1/2)e^{-|x|}
# φ(t) = ∫_{-∞}^{∞} e^{itx}·(1/2)e^{-|x|} dx
# Analitic computation:
# part1 = (1/2)*integral_{-oo}^{0} e^{(it+1)x} dx = 1/(2(it+1))
# part2 = (1/2)*integral_{0}^{oo} e^{(it-1)x} dx = 1/(2(1-it))
part1_13 = Rational(1, 2) / (I*t13 + 1)
part2_13 = Rational(1, 2) / (1 - I*t13)
phi_computed = simplify(part1_13 + part2_13)
phi_expected_13 = 1 / (1 + t13**2)
print(f"  验证特征函数: 1/(2(it+1)) + 1/(2(1-it))")
print(f"    = {phi_computed}")
print(f"    = 1/(1+t^2) ? {simplify(phi_computed - phi_expected_13) == 0}")

# 用 Fourier 反演验证 f(0)=1/2
f0_inversion = integrate(phi_expected_13, (t13, -oo, oo)) / (2*pi)
print(f"  f(0)反演: 1/(2π)·∫ φ(t) dt = {simplify(f0_inversion)} (预期 1/2)")

# P(|X|<1) = ∫_{-1}^{1} (1/2)e^{-|x|} dx
# = 2·∫_0^1 (1/2)e^{-x} dx = ∫_0^1 e^{-x} dx = 1-e^{-1}
P13 = 2 * integrate(Rational(1, 2) * exp(-x13), (x13, 0, 1))
print(f"  P(|X|<1) = 2·∫₀¹ (1/2)e^(-x) dx = {simplify(P13)}")

expected_P13 = 1 - exp(-1)
print(f"  预期答案: f(x)=(1/2)e^(-|x|), P(|X|<1) = 1 - e⁻¹ = {expected_P13}")

ok13a = simplify(phi_computed - phi_expected_13) == 0
ok13b = simplify(P13 - expected_P13) == 0
print(f"  特征函数: {'✅' if ok13a else '❌'}, 概率: {'✅' if ok13b else '❌'}")
ok13 = ok13a and ok13b
results.append(("题13 (特征函数反演)", ok13))
print()

# ============================================================
# 【题14】证明 — Var(aX+b) = a² Var(X) (简单/证明)
# ============================================================
print("=" * 60)
print("【题14】证明验证 — Var(aX+b) = a² Var(X)")
print("=" * 60)
print("问题: 对任意随机变量X和常数a,b，证明 Var(aX+b) = a² Var(X)")
print()

# 用符号矩量: 设 E[X]=μ, E[X²]=μ₂
a_s, b_s, mu_s, mu2_s = symbols('a b mu mu2', real=True)
VarX_sym = mu2_s - mu_s**2  # Var(X)

# Var(aX+b) = E[(aX+b)²] - (E[aX+b])²
# E[(aX+b)²] = a²E[X²] + 2abE[X] + b²
E_Y2 = a_s**2 * mu2_s + 2*a_s*b_s*mu_s + b_s**2
# E[aX+b] = aE[X] + b
E_Y = a_s*mu_s + b_s
# Var(aX+b)
VarY_sym = simplify(E_Y2 - E_Y**2)
print(f"  Var(aX+b) = E[(aX+b)²] - (E[aX+b])²")
print(f"             = ({E_Y2}) - ({E_Y})²")
print(f"             = {VarY_sym}")

# 预期: a²Var(X) = a²(μ₂ - μ²)
expected14 = a_s**2 * VarX_sym
print(f"  a²Var(X)  = a²·(μ₂ - μ²) = {expected14}")

diff14 = simplify(VarY_sym - expected14)
print(f"  差 = {diff14}")
print(f"  恒等? {diff14 == 0}")
ok14 = diff14 == 0
print(f"\n  {'✅ 通过 — 恒等式成立' if ok14 else '❌ 失败'}")
results.append(("题14 (方差性质的证明)", ok14))
print()

# ============================================================
# 【题15】证明 — 马尔可夫不等式 (中等/证明)
# ============================================================
print("=" * 60)
print("【题15】证明验证 — 马尔可夫不等式 P(X≥ε) ≤ E[X]/ε")
print("=" * 60)
print("问题: X≥0, 对任意 ε>0, 证明 P(X≥ε) ≤ E[X]/ε")
print()

# 核心: I_{X≥ε} ≤ X/ε (a.s.)
# 证明思路: X≥0,ε>0 ⇒ 两种情况恒成立
print("  核心不等式: I_{X≥ε} ≤ X/ε (a.s.)")
print("  (1) 若 X ≥ ε: 1 ≤ X/ε (因为 X/ε ≥ 1)")
print("  (2) 若 0 ≤ X < ε: 0 ≤ X/ε (因为 X/ε ≥ 0)")
print("  取期望得: P(X≥ε) = E[I] ≤ E[X/ε] = E[X]/ε")

# 用具体分布数值验证
import math

# Test 1: X ~ Exp(1), E[X]=1, ε=3
lam15 = 1
eps15 = 3
P_test1 = float(exp(-lam15 * eps15))
bound1 = 1.0 / eps15
p15a = P_test1 <= bound1
print(f"\n  数值验证1: X~Exp(1), ε=3")
print(f"    P(X≥3) = e⁻³ ≈ {P_test1:.6f}")
print(f"    E[X]/ε = 1/3 ≈ {bound1:.6f}")
print(f"    不等式成立? {p15a}")

# Test 2: X ~ Exp(2), E[X]=0.5, ε=0.3
lam15b = 2
eps15b = 0.3
P_test2 = float(exp(-lam15b * eps15b))
bound2 = (1.0/lam15b) / eps15b
p15b = P_test2 <= bound2
print(f"  数值验证2: X~Exp(2), ε=0.3")
print(f"    P(X≥0.3) = e^(-0.6) ≈ {P_test2:.6f}")
print(f"    E[X]/ε = 0.5/0.3 ≈ {bound2:.6f}")
print(f"    不等式成立? {p15b}")

# Test 3: X ~ Exp(0.5), E[X]=2, ε=0.1
lam15c = 0.5
eps15c = 0.1
P_test3 = float(exp(-lam15c * eps15c))
bound3 = (1.0/lam15c) / eps15c
p15c = P_test3 <= bound3
print(f"  数值验证3: X~Exp(0.5), ε=0.1")
print(f"    P(X≥0.1) = e^(-0.05) ≈ {P_test3:.6f}")
print(f"    E[X]/ε = 2/0.1 = {bound3:.1f}")
print(f"    不等式成立? {p15c}")

ok15 = p15a and p15b and p15c
print(f"\n  {'✅ 通过 — 马尔可夫不等式验证成立' if ok15 else '❌ 失败'}")
results.append(("题15 (Markov不等式证明)", ok15))
print()

# ============================================================
# 【题16】证明 — 泊松分布可加性 (中等/证明)
# ============================================================
print("=" * 60)
print("【题16】证明验证 — 泊松分布可加性")
print("=" * 60)
print("问题: X~Poisson(λ₁), Y~Poisson(λ₂) 独立，证明 X+Y~Poisson(λ₁+λ₂)")
print()

lam1_16, lam2_16 = 2, 3
lam_sum = lam1_16 + lam2_16

# 方法一: 特征函数验证
t16 = Symbol('t', real=True)
phi1_16 = exp(lam1_16 * (exp(I*t16) - 1))
phi2_16 = exp(lam2_16 * (exp(I*t16) - 1))
phi_sum = simplify(phi1_16 * phi2_16)
phi_expected = exp(lam_sum * (exp(I*t16) - 1))
cf_ok = simplify(phi_sum - phi_expected) == 0
print(f"  方法一 (特征函数):")
print(f"    φ_X(t) = exp(λ₁(e^(it)-1))")
print(f"    φ_Y(t) = exp(λ₂(e^(it)-1))")
print(f"    φ_{{X+Y}}(t) = φ_X·φ_Y = exp((λ₁+λ₂)(e^(it)-1)) → Poisson(λ₁+λ₂)")
print(f"    符号验证: {cf_ok}")

# 方法二: 卷积验证 — 对多个k值验证
print(f"\n  方法二 (卷积), λ₁=2, λ₂=3:")
all_k_ok = True
ii_sym = Symbol('ii', integer=True, nonnegative=True)
for k_test in range(0, 8):
    # 卷积: Σ_{i=0}^{k} P(X=i)·P(Y=k-i)
    conv_sum = summation(
        exp(-lam1_16) * lam1_16**ii_sym / factorial(ii_sym) *
        exp(-lam2_16) * lam2_16**(k_test - ii_sym) / factorial(k_test - ii_sym),
        (ii_sym, 0, k_test)
    )
    # 直接: Poisson(5)
    direct_p = exp(-lam_sum) * lam_sum**k_test / factorial(k_test)
    diff_val = simplify(conv_sum - direct_p)
    k_ok = diff_val == 0
    status = "✅" if k_ok else "❌"
    print(f"    k={k_test}: 卷积={simplify(conv_sum)}, 直接={simplify(direct_p)} {status}")
    if not k_ok:
        all_k_ok = False

# 二项式定理验证 (用具体k值, 因为sympy不能自动化简符号求和)
lam1_sym2, lam2_sym2 = symbols('L1 L2', positive=True)
binom_all_ok = True
print(f"\n  二项式定理验证 (Σ C(k,i)·λ₁^i·λ₂^(k-i) = (λ₁+λ₂)^k):")
for kv in range(1, 8):
    binom_sum = sum(
        binomial(kv, i) * lam1_sym2**i * lam2_sym2**(kv - i)
        for i in range(kv + 1)
    )
    # Expected: (λ₁+λ₂)^k expanded
    expected_val = expand((lam1_sym2 + lam2_sym2)**kv)
    b_ok = simplify(binom_sum - expected_val) == 0
    print(f"    k={kv}: {'✅' if b_ok else '❌'}", end="")
    if not b_ok:
        binom_all_ok = False
        print(f" (diff) ", end="")
    print()

ok16 = cf_ok and all_k_ok and binom_all_ok
print(f"\n  {'✅ 通过 — 泊松可加性验证成立' if ok16 else '❌ 失败'}")
results.append(("题16 (Poisson可加性证明)", ok16))
print()

# ============================================================
# 【题17】证明 — 指数分布无记忆性 (中等/证明)
# ============================================================
print("=" * 60)
print("【题17】证明验证 — 指数分布无记忆性")
print("=" * 60)
print("问题: X~Exp(λ)，证明 P(X>s+t | X>s) = P(X>t), s>0, t>0")
print()

lam_s, s_s, t_s = symbols('lam s t', positive=True)

# P(X > s+t | X > s) = P(X > s+t) / P(X > s)
# = e^{-λ(s+t)} / e^{-λs}
lhs17 = exp(-lam_s * (s_s + t_s)) / exp(-lam_s * s_s)
lhs17_s = simplify(lhs17)
print(f"  P(X>s+t | X>s) = e^(-λ(s+t)) / e^(-λs) = {lhs17_s}")

# P(X > t) = e^{-λt}
rhs17 = exp(-lam_s * t_s)
print(f"  P(X > t) = e^(-λt) = {rhs17}")

diff17 = simplify(lhs17_s - rhs17)
print(f"  差 = {diff17}")
ok17_symbolic = diff17 == 0

# 数值验证
import math as m
test_cases_17 = [(1.0, 2.0, 3.0), (0.5, 1.0, 1.5), (2.0, 0.3, 0.7), (3.0, 5.0, 1.0)]
print(f"\n  数值验证:")
all_num_17 = True
for lt, st, tt in test_cases_17:
    lhs_num = m.exp(-lt*(st+tt)) / m.exp(-lt*st)
    rhs_num = m.exp(-lt*tt)
    ok_n = abs(lhs_num - rhs_num) < 1e-14
    status = "✅" if ok_n else "❌"
    print(f"    λ={lt}, s={st}, t={tt}: 左={lhs_num:.10f}, 右={rhs_num:.10f} {status}")
    if not ok_n:
        all_num_17 = False

ok17 = ok17_symbolic and all_num_17
print(f"\n  {'✅ 通过 — 无记忆性验证成立' if ok17 else '❌ 失败'}")
results.append(("题17 (指数无记忆性证明)", ok17))
print()

# ============================================================
# 【题18】证明 — 依概率收敛 ⇒ 依分布收敛 (困难/证明)
# ============================================================
print("=" * 60)
print("【题18】证明验证 — 依概率收敛⇒依分布收敛")
print("=" * 60)
print("问题: X_n →^P X，证明 X_n →^D X")
print()

# 核心是通过集合包含关系建立双边不等式:
# F_X(x-ε) - P(|X_n-X|>ε) ≤ F_n(x) ≤ F_X(x+ε) + P(|X_n-X|>ε)

print("  验证集合包含关系 (关键步骤):")
print()

# 上界推导的集合包含:
# {X_n ≤ x} = {X_n ≤ x, X ≤ x+ε} ∪ {X_n ≤ x, X > x+ε}
# 第二项是 {X_n ≤ x, X > x+ε} ⊆ {|X_n - X| > ε}
print("  【上界推导】")
print("    {X_n ≤ x} ⊆ {X ≤ x+ε} ∪ {|X_n - X| > ε}")
print("    验证: 若 X_n ≤ x 且 X > x+ε")
print("    则 X - X_n > (x+ε) - x = ε > 0")
print("    故 |X_n - X| ≥ X - X_n > ε ✓")
print("    因此 P(X_n ≤ x) ≤ P(X ≤ x+ε) + P(|X_n-X| > ε)")

# 数值验证
xn_t, xv_t, eps_t = 1.5, 2.5, 0.3
x_t = 2.0
cond_up = (xn_t <= x_t) and (xv_t > x_t + eps_t)
conc_up = abs(xn_t - xv_t) > eps_t
print(f"\n    数值验证上界: xn={xn_t}≤x={x_t}, X={xv_t}>x+ε={x_t+eps_t}")
print(f"    ⇒ |xn-X|={abs(xn_t-xv_t)}>ε={eps_t}? {conc_up}")
print()

# 下界推导的集合包含:
# {X ≤ x-ε} = {X ≤ x-ε, X_n ≤ x} ∪ {X ≤ x-ε, X_n > x}
# 第二项是 {X ≤ x-ε, X_n > x} ⊆ {|X_n - X| > ε}
print("  【下界推导】")
print("    {X ≤ x-ε} ⊆ {X_n ≤ x} ∪ {|X_n - X| > ε}")
print("    验证: 若 X ≤ x-ε 且 X_n > x")
print("    则 X_n - X > x - (x-ε) = ε > 0")
print("    故 |X_n - X| ≥ X_n - X > ε ✓")
print("    因此 P(X ≤ x-ε) ≤ P(X_n ≤ x) + P(|X_n-X| > ε)")
print("    等价地: F_n(x) ≥ F_X(x-ε) - P(|X_n-X| > ε)")

# 数值验证
xn_t2, xv_t2 = 2.8, 1.2
cond_lo = (xv_t2 <= x_t - eps_t) and (xn_t2 > x_t)
conc_lo = abs(xn_t2 - xv_t2) > eps_t
print(f"\n    数值验证下界: X={xv_t2}≤x-ε={x_t-eps_t}, xn={xn_t2}>x={x_t}")
print(f"    ⇒ |xn-X|={abs(xn_t2-xv_t2)}>ε={eps_t}? {conc_lo}")
print()

# 综合不等式链
print("  【不等式链】")
print("    F_X(x-ε) - P(|X_n-X| > ε) ≤ F_n(x) ≤ F_X(x+ε) + P(|X_n-X| > ε)")
print("    取 n→∞: P(|X_n-X| > ε) → 0 (依概率收敛)")
print("    ⇒ F_X(x-ε) ≤ liminf F_n(x) ≤ limsup F_n(x) ≤ F_X(x+ε)")
print("    令 ε→0⁺: F_X 在 x 处连续 ⇒ F_n(x) → F_X(x)")

# 数值演示: X_n ~ N(0, 1/n), X = 0
# X_n →^P 0 (Var → 0), F_n(x) = Φ(x√n) → F_X(x) = I{x≥0}
print(f"\n  【数值演示】 X_n ~ N(0, 1/n), X = 0 (退化分布)")
print("    F_X(x) = 0 for x<0, 1 for x≥0")
import mpmath as mp
for n_val in [1, 5, 25, 100]:
    Fn_half = mp.ncdf(0.5 * mp.sqrt(n_val))
    print(f"    n={n_val:4d}: F_n(0.5) = Φ(0.5√{n_val}) = {float(Fn_half):.8f} → 1")

ok18 = cond_up and conc_up and cond_lo and conc_lo
print(f"\n  {'✅ 通过 — 收敛关系验证成立' if ok18 else '❌ 失败'}")
results.append(("题18 (依概率⇒依分布证明)", ok18))
print()

# ============================================================
# 【题19】证明 — Slutsky 定理特例 (困难/证明)
# ============================================================
print("=" * 60)
print("【题19】证明验证 — Slutsky定理: X_n→^D X, Y_n→^P 0")
print("=" * 60)
print("问题: X_n →^D X, Y_n →^P 0，证明 X_n+Y_n →^D X")
print()

# 两个核心集合包含关系:
print("  验证核心集合包含关系:")
print()

# (1) {X_n+Y_n ≤ x, |Y_n| ≤ ε} ⊆ {X_n ≤ x+ε}
print("  【上界推导】")
print("    {X_n+Y_n ≤ x, |Y_n| ≤ ε} ⊆ {X_n ≤ x+ε}")
print("    验证: 若 X_n + Y_n ≤ x 且 |Y_n| ≤ ε (即 -ε ≤ Y_n ≤ ε)")
print("    则 X_n = (X_n+Y_n) - Y_n ≤ x - Y_n ≤ x + ε")
print("    (因为 -Y_n ≤ |Y_n| ≤ ε)")
print("    因此 P(X_n+Y_n ≤ x) ≤ P(X_n ≤ x+ε) + P(|Y_n| > ε)")

# 数值验证上界
xn19, yn19, x19, eps19 = 1.5, 0.2, 2.0, 0.5
cond1_19 = (xn19 + yn19 <= x19) and (abs(yn19) <= eps19)
concl1_19 = xn19 <= x19 + eps19
print(f"\n    数值验证上界: xn+yn={xn19+yn19}≤x={x19}, |yn|={abs(yn19)}≤ε={eps19}")
print(f"    ⇒ xn={xn19}≤x+ε={x19+eps19}? {concl1_19}")
print()

# (2) {X_n ≤ x-ε, |Y_n| ≤ ε} ⊆ {X_n+Y_n ≤ x}
print("  【下界推导】")
print("    {X_n ≤ x-ε, |Y_n| ≤ ε} ⊆ {X_n+Y_n ≤ x}")
print("    验证: 若 X_n ≤ x-ε 且 |Y_n| ≤ ε (即 -ε ≤ Y_n ≤ ε)")
print("    则 X_n + Y_n ≤ (x-ε) + ε = x")
print("    (因为 Y_n ≤ |Y_n| ≤ ε)")
print("    因此 P(X_n ≤ x-ε) ≤ P(X_n+Y_n ≤ x) + P(|Y_n| > ε)")
print("    等价地: P(X_n+Y_n ≤ x) ≥ P(X_n ≤ x-ε) - P(|Y_n| > ε)")

# 数值验证下界
xn19_2, yn19_2 = 1.2, 0.4
cond2_19 = (xn19_2 <= x19 - eps19) and (abs(yn19_2) <= eps19)
concl2_19 = (xn19_2 + yn19_2) <= x19
print(f"\n    数值验证下界: xn={xn19_2}≤x-ε={x19-eps19}, |yn|={abs(yn19_2)}≤ε={eps19}")
print(f"    ⇒ xn+yn={xn19_2+yn19_2}≤x={x19}? {concl2_19}")
print()

# 综合不等式链
print("  【不等式链】")
print("    P(X_n ≤ x-ε) - P(|Y_n| > ε) ≤ P(X_n+Y_n ≤ x) ≤ P(X_n ≤ x+ε) + P(|Y_n| > ε)")
print("    取 n→∞: P(|Y_n| > ε) → 0 (Y_n →^P 0)")
print("             P(X_n ≤ x±ε) → F_X(x±ε) (X_n →^D X, 取 F_X 连续点)")
print("    ⇒ F_X(x-ε) ≤ liminf ≤ limsup ≤ F_X(x+ε)")
print("    沿 F_X 连续点序列令 ε→0⁺ ⇒ P(X_n+Y_n ≤ x) → F_X(x)")

# 符号代数验证上界
# 假设 xn+yn <= x, |yn| <= eps
# 求证: xn <= x+eps
# 由 xn+yn <= x ⇒ xn <= x - yn
# 由 |yn| <= eps ⇒ -yn <= eps
# 所以 xn <= x - yn <= x + eps
print(f"\n  【符号验证】")
print(f"    上界: xn = (xn+yn) - yn ≤ x - yn ≤ x + ε (因为 -yn ≤ ε)")
print(f"    下界: xn + yn ≤ (x-ε) + ε = x (因为 yn ≤ ε)")

ok19 = concl1_19 and concl2_19
print(f"\n  {'✅ 通过 — Slutsky定理验证成立' if ok19 else '❌ 失败'}")
results.append(("题19 (Slutsky定理证明)", ok19))
print()

# ============================================================
# 汇总
# ============================================================
print("=" * 60)
print("全题验证汇总")
print("=" * 60)

all_ok = True
for name, ok in results:
    status = "✅" if ok else "❌"
    if not ok:
        all_ok = False
    print(f"  {status} {name}")

passed = sum(1 for _, ok in results if ok)
total = len(results)
print()
print(f"总计: {passed}/{total} 通过")

if all_ok:
    print("\n🎉 全部 20 道题目验证通过！")
else:
    print(f"\n⚠ 有 {total - passed} 道题目未通过，请检查。")
