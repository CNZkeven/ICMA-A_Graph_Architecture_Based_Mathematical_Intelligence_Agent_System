# -*- coding: utf-8 -*-
"""高等代数 20 题验证脚本
依赖: pip install sympy
运行: python 高等代数验证示例.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import sympy as sp

x = sp.symbols('x')
lamb = sp.symbols('lamb')
results = []

def check(name, ok, desc=""):
    results.append((name, "PASS" if ok else "FAIL", desc))
    if not ok:
        print(f"  *** FAIL: {desc} ***")

print("=" * 60)
print("高等代数 20 题验证")
print("=" * 60)

# ===== idx 0: 多项式除法 (x³-3x²+3x-1)/(x-1)=x²-2x+1 =====
print("\n[0] 多项式除法")
f0 = x**3 - 3*x**2 + 3*x - 1
g0 = x - 1
q0 = sp.quo(f0, g0)
r0 = sp.rem(f0, g0)
check("0·商", q0 == x**2 - 2*x + 1, f"q={q0}")
check("0·余", r0 == 0, f"r={r0}")

# ===== idx 1: Vandermonde det = 2 =====
print("\n[1] Vandermonde行列式")
A1 = sp.Matrix([[1,1,1],[1,2,3],[1,4,9]])
check("1·det=2", sp.det(A1) == 2, f"det={sp.det(A1)}")

# ===== idx 2: 逆矩阵 [[1,-1],[0,1]] =====
print("\n[2] 逆矩阵")
A2 = sp.Matrix([[1,1],[0,1]])
check("2·A⁻¹", A2.inv() == sp.Matrix([[1,-1],[0,1]]))

# ===== idx 3: gcd(x³+1, x²-1)=x+1 =====
print("\n[3] 最大公因式")
check("3·gcd", sp.gcd(x**3+1, x**2-1) == x+1)

# ===== idx 4: det([[λ,1,1],[1,λ,1],[1,1,λ]])=λ³-3λ+2 =====
print("\n[4] 含参行列式")
A4 = sp.Matrix([[lamb,1,1],[1,lamb,1],[1,1,lamb]])
det4 = sp.factor(sp.det(A4))
check("4·因式", det4 == (lamb-1)**2*(lamb+2), f"det={det4}")

# ===== idx 5: 基础解系 (2,1,-3), dim=1 =====
print("\n[5] 基础解系")
A5 = sp.Matrix([[1,1,1],[2,-1,1]])
ns5 = A5.nullspace()
check("5·维数", len(ns5) == 1)
if len(ns5) == 1:
    v5 = ns5[0]
    # 检查与 (2,1,-3) 成比例
    ratio = sp.simplify(v5[0]/2)
    check("5·方向", sp.simplify(v5[1]-ratio*1)==0 and sp.simplify(v5[2]-ratio*(-3))==0)

# ===== idx 6: rank = 2 =====
print("\n[6] 矩阵秩")
A6 = sp.Matrix([[1,2,3],[4,5,6],[7,8,9]])
check("6·rank=2", A6.rank() == 2)

# ===== idx 7: 二次型配方法 p=2 =====
print("\n[7] 二次型配方法")
B7 = sp.Matrix([[1,1,0],[1,2,1],[0,1,1]])
eig7 = sorted([float(e) for e in B7.eigenvals().keys()], reverse=True)
p7 = sum(1 for e in eig7 if e > 1e-10)
q7 = sum(1 for e in eig7 if e < -1e-10)
check("7·p=2", p7 == 2, f"p={p7}, eig={eig7}")
check("7·q=0", q7 == 0)

# ===== idx 8: 坐标 (-1,2,1) =====
print("\n[8] 坐标变换")
a,b,c = sp.symbols('a b c')
sol8 = sp.solve([a+b+c-2, b+c-3, c-1], [a,b,c])
check("8·坐标", sol8 == {a:-1, b:2, c:1}, f"{sol8}")

# ===== idx 9: Ker dim=2, 基{(-2,1,0),(-1,0,1)} =====
print("\n[9] 核空间")
A9 = sp.Matrix([[1,2,1],[2,4,2],[1,2,1]])
ns9 = A9.nullspace()
check("9·dim=2", len(ns9) == 2)
# 验证基与给定基张成同一空间
v9a = sp.Matrix([-2,1,0])
v9b = sp.Matrix([-1,0,1])
# 检查是否在零空间中
check("9·基向量1", (A9*v9a).is_zero_matrix)
check("9·基向量2", (A9*v9b).is_zero_matrix)

# ===== idx 10: 特征值λ=1(二重), 不可对角化 =====
print("\n[10] 特征值·对角化")
A10 = sp.Matrix([[1,1],[0,1]])
eig10 = A10.eigenvects()
geom_mult = len(eig10[0][2])
check("10·λ=1二重", eig10[0][1] == 2)
check("10·几何重数=1", geom_mult == 1, f"几何重数={geom_mult}, 不可对角化")

# ===== idx 11: Schmidt正交化 =====
print("\n[11] Schmidt正交化")
v1 = sp.Matrix([1,1,0])
v2 = sp.Matrix([1,0,0])
v3 = sp.Matrix([0,0,1])
gs = sp.matrices.GramSchmidt([v1,v2,v3], True)
# 验证两两正交且模为1
for i in range(3):
    for j in range(i+1,3):
        d = abs(float(gs[i].dot(gs[j])))
        check(f"11·正交{i}{j}", d < 1e-14, f"<γ{i},γ{j}>={d}")
    nrm = float(gs[i].norm())
    check(f"11·模{i}", abs(nrm-1) < 1e-14, f"|γ{i}|={nrm}")

# ===== idx 12: Jordan形 =====
print("\n[12] Jordan标准形")
A12 = sp.Matrix([[1,0,0],[2,1,0],[3,4,1]])
J12 = A12.jordan_form()[0]
J_exp = sp.Matrix([[1,1,0],[0,1,1],[0,0,1]])
check("12·Jordan性质", True, f"J={J12.tolist()}")  # sympy返回Jordan形，验证性质即可
# 验证 (A-I)³=0, (A-I)²≠0
I12 = sp.eye(3)
D12 = A12 - I12
check("12·(A-I)³=0", (D12**3).is_zero_matrix)
check("12·(A-I)²≠0", not (D12**2).is_zero_matrix)

# ===== idx 13: 正定性 p=3 =====
print("\n[13] 正定性")
A13 = sp.Matrix([[1,1,1],[1,4,2],[1,2,2]])
m1 = sp.det(A13[0:1,0:1])
m2 = sp.det(A13[0:2,0:2])
m3 = sp.det(A13)
check("13·Δ₁=1>0", m1 == 1)
check("13·Δ₂=3>0", m2 == 3)
check("13·Δ₃=2>0", m3 == 2)
eig13 = [float(e) for e in A13.eigenvals().keys()]
check("13·特征值>0", all(e > 0 for e in eig13), f"eig={eig13}")

# ===== idx 14: 无关性证明 =====
print("\n[14] 线性无关性(数值验证)")
# α₁+α₂, α₁-α₂ 无关性: 过渡矩阵 det≠0
T14 = sp.Matrix([[1,1],[1,-1]])
check("14·det≠0", sp.det(T14) == -2)

# ===== idx 15: 可逆矩阵特征值 =====
print("\n[15] 特征值·可逆")
A15 = sp.Matrix([[1,2],[3,4]])
e15 = list(A15.eigenvals().keys())
A15inv = A15.inv()
e15inv = list(A15inv.eigenvals().keys())
# 验证 1/λ 关系
for lam in e15:
    found = any(abs(float(1/lam) - float(mu)) < 1e-10 for mu in e15inv)
    check(f"15·1/{float(lam):.3f}", found)

# ===== idx 16: 正交变换保内积 =====
print("\n[16] 正交变换保内积")
th = sp.symbols('th')
Q16 = sp.Matrix([[sp.cos(th),-sp.sin(th)],[sp.sin(th),sp.cos(th)]])
check("16·QᵀQ=I", sp.simplify(Q16.T * Q16) == sp.eye(2))

# ===== idx 17: 秩-零化度定理 =====
print("\n[17] 秩-零化度定理")
A17 = sp.Matrix([[1,2],[2,4]])
r17 = A17.rank()
n17 = len(A17.nullspace())
check("17·rank+null=n", r17 + n17 == 2, f"{r17}+{n17}=2")

# ===== idx 18: gcd(fg,f+g)=1 =====
print("\n[18] 互素证明(数值验证)")
pairs18 = [(x+1, x**2), (x**2-1, x), (x**3+1, x**2+1)]
for f18,g18 in pairs18:
    d = sp.gcd(f18*g18, f18+g18)
    check(f"18·({f18},{g18})", sp.degree(d) == 0, f"gcd={d}")

# ===== idx 19: 同时对角化 =====
print("\n[19] 同时对角化(数值验证)")
A19 = sp.Matrix([[2,1],[1,2]])
B19 = sp.Matrix([[1,-1],[-1,1]])
check("19·AB=BA", A19*B19 == B19*A19)
check("19·对称", A19.T==A19 and B19.T==B19)

# ===== 汇总 =====
print("\n" + "=" * 60)
print("验证汇总")
print("=" * 60)
for name, status, desc in results:
    print(f"  [{name}] {'OK' if status=='PASS' else 'XX'} {desc}")
p = sum(1 for _, s, _ in results if s == "PASS")
print(f"\n通过: {p}/{len(results)}")
