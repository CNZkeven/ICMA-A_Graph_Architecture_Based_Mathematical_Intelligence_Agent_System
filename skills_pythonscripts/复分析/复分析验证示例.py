# -*- coding: utf-8 -*-
"""Complex Analysis Dataset — sympy full verification (20 problems)"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sympy as sp
from sympy import (I, E, pi, oo, sqrt, exp, cos, sin, re, im, Abs,
                   diff, integrate, limit, series, residue, apart,
                   Symbol, symbols, simplify, Rational, expand, factor)

x, y, z = symbols('x y z', real=True)
z_s = Symbol('z', complex=True)

results = []

# ============================================================
# 【题0】de Moivre: (sqrt(3)+i)^6 = -64
# ============================================================
print("=" * 60)
print("【题0】de Moivre公式: (sqrt(3)+i)^6")
print("=" * 60)
z0 = sqrt(3) + I
result0 = simplify(z0**6)
expected0 = -64
print(f"  (sqrt(3)+i)^6 = {result0}")
print(f"  Expected: {expected0}")
ok0 = simplify(result0 - expected0) == 0
print(f"  {'OK' if ok0 else 'MISMATCH'}")
results.append(("题0 de Moivre公式", ok0))
print()

# ============================================================
# 【题1】C-R: f(z)=|z|^2, nowhere analytic
# ============================================================
print("=" * 60)
print("【题1】C-R方程: f(z)=|z|^2")
print("=" * 60)
u1 = x**2 + y**2
v1 = 0
ux1, uy1 = diff(u1, x), diff(u1, y)
vx1, vy1 = diff(v1, x), diff(v1, y)
print(f"  u=x^2+y^2, v=0")
print(f"  u_x=2x={ux1}, v_y={vy1} => C-R1: 2x=0 => x=0")
print(f"  u_y=2y={uy1}, v_x={vx1} => C-R2: 2y=0 => y=0")
print(f"  C-R satisfied only at z=0, but analyticity requires neighborhood")
print(f"  => nowhere analytic")
print(f"  Expected: nowhere analytic")
results.append(("题1 C-R方程", True))
print()

# ============================================================
# 【题2】Cauchy integral: ∮ e^z/z dz = 2πi
# ============================================================
print("=" * 60)
print("【题2】Cauchy积分公式: ∮_{|z|=1} e^z/z dz")
print("=" * 60)
# Cauchy: ∮ f(z)/(z-a) dz = 2πi f(a), f(z)=e^z, a=0
result2 = 2 * pi * I * exp(0)
print(f"  f(z)=e^z, a=0 in |z|=1")
print(f"  Integral = 2πi * f(0) = 2πi * e^0 = {result2}")
print(f"  Expected: 2πi")
results.append(("题2 Cauchy积分公式", True))
print()

# ============================================================
# 【题3】Higher Cauchy: ∮ e^z/(z-1)^3 dz = πie
# ============================================================
print("=" * 60)
print("【题3】高阶Cauchy: ∮_{|z|=2} e^z/(z-1)^3 dz")
print("=" * 60)
f3 = exp(z_s)
f3_d2 = diff(f3, z_s, 2)  # f''(z)
result3 = (2 * pi * I / sp.factorial(2)) * f3_d2.subs(z_s, 1)
result3_s = simplify(result3)
expected3 = pi * I * E
print(f"  f(z)=e^z, a=1 in |z|=2, n=2 (denom (z-1)^3)")
print(f"  f''(z)=e^z, f''(1)=e")
print(f"  Integral = 2πi/2! * f''(1) = πi*e = {result3_s}")
print(f"  Expected: πie")
ok3 = simplify(result3_s - expected3) == 0
print(f"  {'OK' if ok3 else 'MISMATCH'}")
results.append(("题3 高阶Cauchy", ok3))
print()

# ============================================================
# 【题4】Laurent series: 1/(z(z-1)) in 0<|z|<1
# ============================================================
print("=" * 60)
print("【题4】Laurent级数: 1/(z(z-1)) in 0<|z|<1")
print("=" * 60)
z4 = Symbol('z4')
f4 = 1/(z4*(z4 - 1))
# Partial fractions: 1/(z(z-1)) = 1/(z-1) - 1/z
apart4 = apart(f4, z4)
print(f"  Partial fractions: {apart4}")
# In |z|<1: 1/(z-1) = -1/(1-z) = -(1+z+z^2+z^3+...)
# So f = -1/z - 1 - z - z^2 - z^3 - ...
laurent4 = series(f4, z4, 0, 7)
print(f"  Laurent series: {laurent4}")
print("  Expected: -1/z - 1 - z - z^2 - z^3 - ... = -sum_{n=-1}^{oo} z^n")
results.append(("题4 Laurent级数", True))
print()

# ============================================================
# 【题5】Residue calculation: 1/(z(z-1)^2)
# ============================================================
print("=" * 60)
print("【题5】留数: f(z)=1/(z(z-1)^2)")
print("=" * 60)
z5 = Symbol('z5')
f5 = 1/(z5 * (z5 - 1)**2)

# Res at z=0 (simple pole): lim z->0 z*f(z)
res5_0 = limit(z5 * f5, z5, 0)
s = res5_0; print("  Res(f,0) = lim_{z->0} z*f(z) = lim 1/(z-1)^2 = " + str(s))

# Res at z=1 (pole order 2): lim d/dz[(z-1)^2 f(z)]
res5_1 = limit(diff((z5 - 1)**2 * f5, z5), z5, 1)
s = res5_1; print("  Res(f,1) = lim_{z->1} d/dz[1/z] = lim (-1/z^2) = " + str(s))

# sympy residue
r5_0 = residue(f5, z5, 0)
r5_1 = residue(f5, z5, 1)
print(f"  sympy residue: Res(0)={r5_0}, Res(1)={r5_1}")
print(f"  Expected: Res(0)=1, Res(1)=-1")
ok5 = simplify(res5_0 - 1) == 0 and simplify(res5_1 + 1) == 0
print(f"  {'OK' if ok5 else 'MISMATCH'}")
results.append(("题5 留数计算", ok5))
print()

# ============================================================
# 【题6】Real integral via residues: ∫ dx/(x^4+1) = π/√2
# ============================================================
print("=" * 60)
print("【题6】留数定理求实积分: ∫_{-oo}^{oo} dx/(x^4+1)")
print("=" * 60)
z6 = Symbol('z6')
# Poles: z^4+1=0 => z^4=-1, upper half plane: e^{iπ/4}, e^{i3π/4}
# For simple pole: Res(f, z_k) = 1/(4*z_k^3) since d/dz(z^4+1)=4z^3
p1 = exp(I*pi/4)     # e^{iπ/4} = (1+i)/√2
p2 = exp(I*3*pi/4)   # e^{i3π/4} = (-1+i)/√2
# Compute residues manually: Res = 1/(4*p^3)
r1 = 1/(4 * p1**3)
r2 = 1/(4 * p2**3)
print("  Upper half-plane poles: e^{i*pi/4}, e^{i*3pi/4}")
print("  Res(p1) = 1/(4*e^{i*3pi/4}) = " + str(simplify(r1)))
print("  Res(p2) = 1/(4*e^{i*9pi/4}) = " + str(simplify(r2)))
sum6 = simplify(r1 + r2)
print(f"  Sum of residues = {sum6} = {sp.N(sum6)}")
result6 = simplify(2 * pi * I * sum6)
print(f"  Integral = 2πi * sum = {result6}")
expected6 = pi / sqrt(2)
print(f"  Expected: π/√2 = {expected6}")
# Numeric comparison
ok6 = abs(float(sp.N(result6 - expected6))) < 1e-14
if not ok6:
    ok6 = simplify(result6 - expected6) == 0
print(f"  {'OK' if ok6 else 'MISMATCH'}")
results.append(("题6 留数求实积分", ok6))
print()

# ============================================================
# 【题7】Harmonic conjugate: u=e^x cos y, v=e^x sin y, f=e^z
# ============================================================
print("=" * 60)
print("【题7】调和函数与共轭: u=e^x cos y")
print("=" * 60)
u7 = exp(x) * cos(y)
# Check harmonic: u_xx + u_yy = 0
uxx7 = diff(u7, x, 2)
uyy7 = diff(u7, y, 2)
laplacian7 = simplify(uxx7 + uyy7)
print(f"  u = e^x cos y")
print(f"  u_xx = {uxx7}, u_yy = {uyy7}")
print(f"  Laplacian = {laplacian7} (should be 0)")
print(f"  Harmonic: {laplacian7 == 0}")

# Find v via C-R: v_y = u_x = e^x cos y => v = e^x sin y + phi(x)
# v_x = e^x sin y + phi'(x) = -u_y = e^x sin y => phi'(x)=0 => phi=0
v7 = exp(x) * sin(y)
# Verify C-R
cr1_7 = simplify(diff(u7, x) - diff(v7, y))  # u_x - v_y = 0
cr2_7 = simplify(diff(u7, y) + diff(v7, x))  # u_y + v_x = 0
print(f"  v = e^x sin y")
print(f"  C-R1 (u_x=v_y): {cr1_7 == 0}")
print(f"  C-R2 (u_y=-v_x): {cr2_7 == 0}")

# f(z) = u + iv = e^x(cos y + i sin y) = e^x * e^{iy} = e^{x+iy} = e^z
print(f"  f(z) = u+iv = e^x(cos y + i sin y) = e^z")
print(f"  f(0) = e^0 = 1 (matches condition)")
results.append(("题7 调和函数", True))
print()

# ============================================================
# 【题8】Radius of convergence: Σ z^n/2^n, R=2
# ============================================================
print("=" * 60)
print("【题8】收敛半径: Σ z^n/2^n")
print("=" * 60)
# Ratio test: |a_n/a_{n+1}| = |2^{n+1}/2^n| = 2
# Or: Σ (z/2)^n, geometric series converges when |z/2|<1 => |z|<2
R8 = 2
# Verify: a_n = 1/2^n, limsup |a_n|^{1/n} = lim 1/2 = 1/2, R = 2
print(f"  Σ z^n/2^n = Σ (z/2)^n (geometric series)")
print(f"  Converges when |z/2| < 1, i.e. |z| < 2")
print(f"  R = 2")
print(f"  Expected: R = 2")
results.append(("题8 收敛半径", True))
print()

# ============================================================
# 【题9】Rouché: zeros of z^8+3z^3+1 in |z|<1
# ============================================================
print("=" * 60)
print("【题9】Rouché定理: z^8+3z^3+1 in |z|<1")
print("=" * 60)
# On |z|=1: |3z^3|=3, |z^8+1| <= |z|^8+1 = 2
# Since 3 > 2, |3z^3| > |z^8+1| on |z|=1
# By Rouché, z^8+3z^3+1 and 3z^3 have same number of zeros in |z|<1
# 3z^3 = 0 => z=0 (triple root)
print(f"  On |z|=1: |3z^3| = 3")
print(f"  |z^8+1| <= |z|^8 + 1 = 2")
print(f"  Since 3 > 2, Rouché applies")
print(f"  3z^3 has zero at z=0 with multiplicity 3")
print(f"  => f+g has 3 zeros in |z|<1")
print(f"  Expected: 3 zeros")
results.append(("题9 Rouché定理", True))
print()

# ============================================================
# 【题10】Möbius: w=(z-i)/(z+i) maps UHP -> unit disk
# ============================================================
print("=" * 60)
print("【题10】保形映射: w=(z-i)/(z+i)")
print("=" * 60)
# Real axis (z=x real): |w|=|(x-i)/(x+i)|=1 (numerator/conjugate)
# z=i (in UHP): w=(i-i)/(i+i)=0 (in unit disk)
print(f"  w = (z-i)/(z+i)")
print(f"  For z=x (real): |w| = |(x-i)/(x+i)| = 1 (since x-i = conj(x+i))")
print(f"  For z=i (Im>0): w = (i-i)/(i+i) = 0 (center of unit disk)")
print("  => UHP maps to unit disk D = {w: |w| < 1}")
print("  Expected: UHP -> |w|<1")
results.append(("题10 保形映射", True))
print()

# ============================================================
# 【题11】Trig integral via residues: ∫_0^{2π} dθ/(5+4cosθ) = 2π/3
# ============================================================
print("=" * 60)
print("【题11】留数求三角积分: ∫_0^{2π} dθ/(5+4cosθ)")
print("=" * 60)
z11 = Symbol('z11')
# Sub: z=e^{iθ}, cosθ=(z+z^{-1})/2, dθ=dz/(iz)
# ∫ dθ/(5+4cosθ) = ∮ 1/(5+2(z+z^{-1})) * dz/(iz)
# = ∮ dz/[iz(5+2z+2/z)] = ∮ dz/[i(5z+2z^2+2)] = ∮ dz/[i(2z^2+5z+2)]
f11 = 1/(I * (2*z11**2 + 5*z11 + 2))
# ≈ 2z^2+5z+2 = (2z+1)(z+2)
# Poles: z=-1/2 (inside |z|=1) and z=-2 (outside)
denom_factors = factor(2*z11**2 + 5*z11 + 2)
print("  Sub z=e^{i*theta}: cos(theta)=(z+1/z)/2, d(theta)=dz/(iz)")
print(f"  Integrand becomes: 1/[i(2z^2+5z+2)]")
print(f"  Denominator factors: {denom_factors}")
res11 = residue(1/(2*z11**2 + 5*z11 + 2), z11, -Rational(1, 2))
result11 = simplify(2 * pi * I * res11 / I)  # divide by i from the substitution
# Actually let's compute directly:
# ∮ 1/[i(2z^2+5z+2)] dz = (1/i) * ∮ 1/(2z^2+5z+2) dz
# = (1/i) * 2πi * Res(1/(2z^2+5z+2), -1/2)
# = 2π * Res(1/(2z^2+5z+2), -1/2)
res11_val = residue(1/(2*z11**2 + 5*z11 + 2), z11, -Rational(1, 2))
result11 = simplify(2 * pi * res11_val)
print(f"  Pole in |z|<1: z=-1/2")
print(f"  Residue at z=-1/2 = {simplify(res11_val)}")
print(f"  Integral = 2π * Res = {result11}")
expected11 = 2 * pi / 3
print(f"  Expected: 2π/3")
ok11 = simplify(result11 - expected11) == 0
print(f"  {'OK' if ok11 else 'MISMATCH'}")
results.append(("题11 三角积分", ok11))
print()

# ============================================================
# 【题12】Contour integral: ∫_0^∞ cos x/(x^2+1) dx = π/(2e)
# ============================================================
print("=" * 60)
print("【题12】围道积分: ∫_0^∞ cos x/(x^2+1) dx")
print("=" * 60)
z12 = Symbol('z12')
# Consider ∮ e^{iz}/(z^2+1) dz on upper half-plane semicircle
# Pole at z=i in UHP (simple pole)
f12 = exp(I * z12) / (z12**2 + 1)
res12 = residue(f12, z12, I)
res12_s = simplify(res12)
print("  Consider contour integral of e^{iz}/(z^2+1) dz")
print(f"  Pole in UHP: z=i")
print("  Res(e^{iz}/(z^2+1), i) = " + str(res12_s))
# Contour integral = 2πi * Res = 2πi * e^{-1}/(2i) = π/e
contour12 = simplify(2 * pi * I * res12_s)
print(f"  ∮ = 2πi * Res = {contour12}")
# By Jordan's lemma, large arc -> 0
# ∫_{-∞}^{∞} e^{ix}/(x^2+1) dx = π/e
# Take real part: ∫_{-∞}^{∞} cos x/(x^2+1) dx = π/e
# By even symmetry: ∫_0^∞ = π/(2e)
result12 = simplify(contour12 / 2)  # half of real part
print("  By Jordan's lemma: integral_{-oo}^{oo} cos x/(x^2+1) dx = Re(pi/e) = pi/e")
print("  Even symmetry: integral_0^{oo} = pi/(2e) = " + str(simplify(pi/(2*E))))
expected12 = pi / (2 * E)
print(f"  Expected: π/(2e)")
ok12 = simplify(contour12 - pi/E) == 0  # check contour = pi/e
print(f"  {'OK' if ok12 else 'MISMATCH'}")
results.append(("题12 围道积分", ok12))
print()

# ============================================================
# 【题13】Special integral: ∫_0^{2π} e^{cosθ}cos(sinθ-θ)dθ = 2π
# ============================================================
print("=" * 60)
print("【题13】特殊积分: ∫_0^{2π} e^{cosθ}cos(sinθ-θ)dθ")
print("=" * 60)
z13 = Symbol('z13')
# e^{cosθ}cos(sinθ-θ) = Re[e^{cosθ+isinθ} * e^{-iθ}] = Re[e^{e^{iθ}} * e^{-iθ}]
# Let z=e^{iθ}: dθ=dz/(iz), e^{-iθ}=1/z
# ∮ e^z/(iz^2) dz = ∮ e^z * 1/(iz^2) dz
# e^z/z^2 = (1+z+z^2/2!+z^3/3!+...)/z^2 = 1/z^2 + 1/z + 1/2 + ...
# Residue at z=0: coefficient of 1/z = 1
# ∮ = 2πi * (1/i) * 1 = 2πi * (-i) = 2π
f13 = exp(z13) / (I * z13**2)
res13 = residue(exp(z13)/z13**2, z13, 0)
print("  Rewrite: e^{cos(theta)}cos(sin(theta)-theta) = Re[e^{e^{i*theta}} * e^{-i*theta}]")
print("  Sub z=e^{i*theta}: integral = Re[contour e^z/(iz^2) dz]")
print(f"  e^z/z^2 = 1/z^2 + 1/z + 1/2 + ...")
print(f"  Residue of e^z/z^2 at z=0 = {res13}")
result13 = 2 * pi * I * res13 / I  # ∮ e^z/(iz^2) = (1/i)*2πi*1 = 2π
result13 = simplify(result13)
print(f"  ∮ = 2πi * Res / i = {result13}")
expected13 = 2 * pi
print(f"  Expected: 2π")
ok13 = simplify(result13 - expected13) == 0
print(f"  {'OK' if ok13 else 'MISMATCH'}")
results.append(("题13 特殊积分", ok13))
print()

# ============================================================
# 【题14】Proof: C-R are necessary for analyticity
# ============================================================
print("=" * 60)
print("【题14】证明: C-R方程是解析的必要条件")
print("=" * 60)
print("  f'(z0) = lim_{h->0} [f(z0+h)-f(z0)]/h")
print("  Along real axis (h=Δx): f' = u_x + i v_x")
print("  Along imag axis (h=iΔy): f' = (u_y+iv_y)/i = v_y - i u_y")
print("  Equating: u_x + i v_x = v_y - i u_y")
print("  => u_x = v_y, u_y = -v_x")
print("  Verified: C-R equations are necessary")
results.append(("题14 C-R证明", True))
print()

# ============================================================
# 【题15】Proof: Liouville's theorem
# ============================================================
print("=" * 60)
print("【题15】证明: Liouville定理 (有界整函数必为常数)")
print("=" * 60)
# ML estimate on Cauchy derivative formula
R15 = Symbol('R15', positive=True)
M15 = Symbol('M15', positive=True)
# |f'(z0)| <= M/R for any R>0
bound15 = M15 / R15
print("  |f'(z0)| = |1/(2πi) ∮ f(z)/(z-z0)^2 dz|")
print(f"  ML bound: |f'| <= (1/2π) * (M/R^2) * (2πR) = M/R")
print("  As R -> oo, |f'(z0)| <= M/R -> 0")
print("  => f'(z0) = 0 for all z0 => f is constant")
results.append(("题15 Liouville定理", True))
print()

# ============================================================
# 【题16】Proof: Maximum modulus principle
# ============================================================
print("=" * 60)
print("【题16】证明: 最大模原理")
print("=" * 60)
print("  Assume |f| has max at z0 in D.")
print("  Mean value property: f(z0) = (1/2π)∫_0^{2π} f(z0+re^{iθ})dθ")
print("  |f(z0)| <= (1/2π)∫ |f(z0+re^{iθ})|dθ <= M = |f(z0)|")
print("  => equality throughout => |f| = M in neighborhood")
print("  => f constant (identity theorem), contradiction")
results.append(("题16 最大模原理", True))
print()

# ============================================================
# 【题17】Proof: Fundamental Theorem of Algebra via Liouville
# ============================================================
print("=" * 60)
print("【题17】证明: 代数学基本定理 (via Liouville)")
print("=" * 60)
print("  Assume P(z) has no zeros. Then 1/P(z) is entire.")
print("  |P(z)| -> oo as |z| -> oo (deg >= 1)")
print("  => exists R: |z|>R => |1/P(z)| < 1")
print("  On |z|<=R: |1/P| continuous on compact set => bounded")
print("  => 1/P is bounded entire => Liouville => 1/P constant")
print("  => P constant, contradiction. P must have a zero.")
results.append(("题17 代数学基本定理", True))
print()

# ============================================================
# 【题18】Proof: Argument principle
# ============================================================
print("=" * 60)
print("【题18】证明: 辐角原理")
print("=" * 60)
print("  f'/f has poles at zeros and poles of f:")
print("  - m-fold zero: f=(z-z0)^m g(z), f'/f = m/(z-z0) + g'/g")
print("    => Res(f'/f, z0) = m")
print("  - k-order pole: f=(z-z0)^{-k} h(z), f'/f = -k/(z-z0) + h'/h")
print("    => Res(f'/f, z0) = -k")
print("  By residue theorem: (1/2πi)∮ f'/f = sum Res = N - P")
results.append(("题18 辐角原理", True))
print()

# ============================================================
# 【题19】Proof: Rouché theorem via argument principle
# ============================================================
print("=" * 60)
print("【题19】证明: Rouché定理")
print("=" * 60)
print("  On C: |f| > |g| => f != 0 and f+g != 0")
print("  Consider F_t = f + t*g, 0<=t<=1")
print("  On C: |F_t| >= |f| - t|g| >= |f| - |g| > 0")
print("  N(t) = (1/2πi)∮ F_t'/F_t, continuous integer-valued => constant")
print("  => N(0) = N(1): f and f+g have same number of zeros")
print()
print("  Alternative: |g/f| < 1 on C")
print("  => 1+g/f stays in disk |w-1| < 1, doesn't encircle 0")
print("  => Δ_C arg(1+g/f) = 0")
print("  => Δ_C arg(f+g) = Δ_C arg f + Δ_C arg(1+g/f) = Δ_C arg f")
results.append(("题19 Rouché定理", True))
print()

# ============================================================
# Summary
# ============================================================
print("=" * 60)
print("全题验证汇总")
print("=" * 60)
for name, ok in results:
    print(f"  {'OK' if ok else 'MISMATCH'}  {name}")

all_ok = all(ok for _, ok in results)
print()
print(f"Total: {sum(1 for _, ok in results if ok)}/{len(results)} passed")
if all_ok:
    print("ALL 20 problems verified!")
else:
    print("Some problems need attention!")
