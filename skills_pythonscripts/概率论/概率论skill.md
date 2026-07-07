---
name: probability-theory-verification
description: Use when verifying probability theory problems with sympy — classical probability, distributions, expectations, moment generating functions, characteristic functions, convergence theorems, and limit theorems
---

# 概率论解题与验证技能手册

## 概述

本技能手册覆盖本科概率论课程的核心知识体系，包括古典概型与组合计数、随机变量与分布（离散和连续）、多维随机变量及其数字特征、矩母函数与特征函数、极限定理（大数定律与中心极限定理）、以及常见不等式。全手册以20道精选习题为骨架，系统梳理每个知识模块的核心概念、常用公式、解题步骤和常见陷阱，并配套sympy验证方法，适用于概率论课程的学习、复习与自动解题验证。

---

## 知识点体系

### 模块1：古典概型与组合计数

#### 核心概念

- **古典概型**：样本空间有限且等可能，事件的概率为 $P(A) = \frac{|A|}{|\Omega|}$
- **排列**：$P_n^k = \frac{n!}{(n-k)!}$（有序不放回）
- **组合**：$C_n^k = \binom{n}{k} = \frac{n!}{k!(n-k)!}$（无序不放回）
- **超几何分布**：从 $N$ 个总体（含 $K$ 个"成功"）中不放回抽取 $n$ 个，恰好有 $k$ 个"成功"的概率为 $P = \frac{C_K^k \cdot C_{N-K}^{n-k}}{C_N^n}$
- **有放回抽样**：二项分布；**不放回抽样**：超几何分布

#### 常用公式

1. 加法原理与乘法原理
2. 组合恒等式：$\binom{n}{k} = \binom{n}{n-k}$，$\binom{n}{k} = \binom{n-1}{k} + \binom{n-1}{k-1}$
3. 超几何概率质量函数：
   $$P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$$

#### 解题步骤

1. 明确样本空间：确定总的等可能结果数
2. 确定有利事件：计算满足条件的结果数
3. 代入古典概型公式：$P = \frac{\text{有利结果数}}{\text{总结果数}}$
4. 对于不放回抽样，优先考虑超几何分布直接代入

#### 常见陷阱

- 混淆放回与不放回：放回用二项分布，不放回用超几何分布
- 组合数计算错误：注意 $\binom{n}{k}$ 的分母是 $k!(n-k)!$，不是 $k!$ 或 $(n-k)!$
- 未注意"恰好"与"至少"的区别

#### 对应习题

- **idx 0**: 袋中红球白球不放回抽取，超几何分布直接计算

---

### 模块2：随机变量与分布

#### 2.1 离散分布

##### 二项分布 $B(n, p)$

- **概率质量函数 (pmf)**：$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k=0,1,\ldots,n$
- **期望**：$E[X] = np$
- **方差**：$\mathrm{Var}(X) = np(1-p)$
- **矩母函数**：$M(t) = (pe^t + 1-p)^n$
- **特征函数**：$\varphi(t) = (pe^{it} + 1-p)^n$

##### 泊松分布 $\mathrm{Poisson}(\lambda)$

- **概率质量函数 (pmf)**：$P(X=k) = e^{-\lambda} \frac{\lambda^k}{k!}, \quad k=0,1,2,\ldots$
- **期望与方差**：$E[X] = \lambda, \quad \mathrm{Var}(X) = \lambda$
- **矩母函数**：$M(t) = \exp(\lambda(e^t - 1))$
- **特征函数**：$\varphi(t) = \exp(\lambda(e^{it} - 1))$
- **可加性**：若 $X \sim \mathrm{Poisson}(\lambda_1)$ 与 $Y \sim \mathrm{Poisson}(\lambda_2)$ 独立，则 $X+Y \sim \mathrm{Poisson}(\lambda_1+\lambda_2)$

#### 2.2 连续分布

##### 均匀分布 $U(a, b)$

- **密度函数**：$f(x) = \frac{1}{b-a}, \quad x \in [a,b]$
- **期望**：$E[X] = \frac{a+b}{2}$
- **方差**：$\mathrm{Var}(X) = \frac{(b-a)^2}{12}$
- **特征函数**：$\varphi(t) = \frac{e^{itb} - e^{ita}}{it(b-a)}$

##### 指数分布 $\mathrm{Exp}(\lambda)$

- **密度函数**：$f(x) = \lambda e^{-\lambda x}, \quad x > 0$（参数 $\lambda > 0$）
- **生存函数**：$P(X > x) = e^{-\lambda x}, \quad x > 0$
- **期望与方差**：$E[X] = \frac{1}{\lambda}, \quad \mathrm{Var}(X) = \frac{1}{\lambda^2}$
- **$n$ 阶矩**：$E[X^n] = \frac{n!}{\lambda^n}$（利用 $\Gamma$ 积分：$\int_0^\infty x^n e^{-\lambda x}dx = \frac{\Gamma(n+1)}{\lambda^{n+1}} = \frac{n!}{\lambda^{n+1}}$）
- **无记忆性**：$P(X > s+t \mid X > s) = P(X > t), \quad \forall s,t > 0$
- **矩母函数**：$M(t) = \frac{\lambda}{\lambda - t}, \quad t < \lambda$

##### 正态分布 $N(\mu, \sigma^2)$

- **密度函数**：$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
- **特征函数**：$\varphi(t) = \exp\left(i\mu t - \frac{1}{2}\sigma^2 t^2\right)$

##### Laplace分布 $\mathrm{Laplace}(\mu, b)$

- **密度函数**：$f(x) = \frac{1}{2b} e^{-|x-\mu|/b}$
- **特征函数**：$\varphi(t) = \frac{e^{i\mu t}}{1 + b^2 t^2}$
- 当 $\mu=0, b=1$ 时，$\varphi(t) = \frac{1}{1+t^2}$

#### 2.3 分布函数与密度函数

- **分布函数**：$F(x) = P(X \le x)$，右连续，单调不减，$\lim_{x\to-\infty}F(x)=0$，$\lim_{x\to+\infty}F(x)=1$
- **密度函数**：对连续型，$f(x) = F'(x)$（几乎处处）
- **随机变量函数的分布**：若 $Y = g(X)$，则
  - **分布函数法**：$F_Y(y) = P(g(X) \le y)$，再求导得 $f_Y(y)$
  - **公式法（g严格单调时）**：$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$

#### 常见陷阱

- 指数分布参数化差异：有的教材用 $f(x)=\lambda e^{-\lambda x}$（期望 $1/\lambda$），有的用 $f(x)=\frac{1}{\theta}e^{-x/\theta}$（期望 $\theta$）。做题前先确认参数化方式
- 泊松分布中 $P(X \ge 1) = 1 - P(X=0)$，不要漏掉取补
- 连续型随机变量单点概率为零：$P(X=c)=0$
- 对于 $X \sim U(0,1)$，$Y = -\ln X$ 服从 $\mathrm{Exp}(1)$ 是经典变换

#### 对应习题

- **idx 2**: 二项分布 pmf 直接代入计算
- **idx 9**: 泊松分布 $P(X=2)$ 和 $P(X \ge 1)$ 计算
- **idx 6**: 随机变量函数的分布 — $X \sim U(0,1)$，$Y=-\ln X$，通过分布函数法推导 $Y \sim \mathrm{Exp}(1)$
- **idx 10**: 指数分布高阶矩 — 利用 $\Gamma$ 积分 $E[X^n] = n! / \lambda^n$

---

### 模块3：多维随机变量

#### 3.1 联合分布与边缘分布

- **联合分布函数**：$F(x,y) = P(X \le x, Y \le y)$
- **联合密度**（连续型）：$f(x,y)$，满足 $P((X,Y)\in D) = \iint_D f(x,y)\,dx\,dy$
- **边缘密度**：
  $$f_X(x) = \int_{-\infty}^{\infty} f(x,y)\,dy, \quad f_Y(y) = \int_{-\infty}^{\infty} f(x,y)\,dx$$
- **独立性**：$f(x,y) = f_X(x) \cdot f_Y(y)$（或 $F(x,y) = F_X(x)F_Y(y)$）

#### 3.2 条件分布与条件期望

- **条件密度**（连续型）：$f_{X\mid Y}(x\mid y) = \frac{f(x,y)}{f_Y(y)}$（当 $f_Y(y) > 0$）


- **条件分布律**（离散型）：$P(X=x_i \mid Y=y_j) = \frac{P(X=x_i, Y=y_j)}{P(Y=y_j)}$

- **条件期望**（连续型）：
  $$E[X \mid Y=y] = \int_{-\infty}^{\infty} x \cdot f_{X\mid Y}(x\mid y)\,dx$$
  $$E[g(X) \mid Y=y] = \int_{-\infty}^{\infty} g(x) \cdot f_{X\mid Y}(x\mid y)\,dx$$

- **条件方差**：$\mathrm{Var}(X \mid Y=y) = E[X^2 \mid Y=y] - (E[X \mid Y=y])^2$

- **全期望公式**：$E[X] = E[E[X \mid Y]]$
- **全方差公式**：$\mathrm{Var}(X) = E[\mathrm{Var}(X\mid Y)] + \mathrm{Var}(E[X \mid Y])$

#### 3.3 协方差与相关系数

- **协方差**：$\mathrm{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])] = E[XY] - E[X]E[Y]$
- **相关系数**：
  $$\rho(X,Y) = \frac{\mathrm{Cov}(X,Y)}{\sqrt{\mathrm{Var}(X) \cdot \mathrm{Var}(Y)}}, \quad |\rho| \le 1$$
- **性质**：
  - $\rho = 0$ 表示不相关，但不一定独立（正态分布例外）
  - $\rho = \pm 1$ 当且仅当 $Y = aX + b$ 几乎处处成立（$a \neq 0$）
  - 独立 $\Rightarrow$ 不相关（$\mathrm{Cov}=0$），反之不成立
- **协方差的双线性**：$\mathrm{Cov}(aX+bY, Z) = a\,\mathrm{Cov}(X,Z) + b\,\mathrm{Cov}(Y,Z)$
- **和的方差**：$\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X,Y)$

#### 3.4 二维正态分布

若 $(X,Y) \sim N(\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2, \rho)$，则：

- **条件分布**：$Y \mid X=x \sim N(\mu_{Y\mid X}, \sigma_{Y\mid X}^2)$，其中
  $$\mu_{Y\mid X} = E[Y \mid X=x] = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X)$$
  $$\sigma_{Y\mid X}^2 = \mathrm{Var}(Y \mid X=x) = \sigma_Y^2(1-\rho^2)$$
- **条件均值是 $X$ 的线性函数**
- **条件方差与 $x$ 无关**（同方差性，homoscedasticity）
- **不相关 $\Leftrightarrow$ 独立**（仅对二维正态成立）

#### 常见陷阱

- 边缘密度计算时注意积分上下限：非矩形区域需要正确确定积分范围
- 条件密度只定义在 $f_Y(y) > 0$ 的区域
- 条件期望是 $y$ 的函数，书写时不要漏掉变量
- 协方差计算：$E[XY] - E[X]E[Y]$ 是最常用公式，不要直接用定义展开

#### 对应习题

- **idx 4**: 二维连续联合密度 $f(x,y)=8xy, 0<x<y<1$，计算 $E[XY]$ 和 $\mathrm{Cov}(X,Y)$
- **idx 5**: 二维离散联合分布律，计算 $\mathrm{Cov}(X,Y)$ 和 $\rho(X,Y)$
- **idx 7**: 二维正态分布条件期望与条件方差
- **idx 8**: 条件期望与条件方差 — $f(x,y)=2, 0<x<y<1$，求 $E[X\mid Y=y]$ 和 $\mathrm{Var}(X\mid Y=y)$

---

### 模块4：数字特征

#### 4.1 期望、方差与矩

- **期望**：$E[g(X)] = \int g(x)f(x)\,dx$（连续）或 $\sum g(x_i)P(X=x_i)$（离散）
- **方差**：$\mathrm{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2$
- **$k$ 阶原点矩**：$\mu_k = E[X^k]$
- **$k$ 阶中心矩**：$\nu_k = E[(X-E[X])^k]$
- **方差的性质（线性变换）**：$\mathrm{Var}(aX+b) = a^2 \mathrm{Var}(X)$
  - 平移不变性：加常数不改变方差
  - 缩放平方律：乘常数，方差乘该常数的平方

#### 4.2 矩母函数 (Moment Generating Function, MGF)

- **定义**：$M_X(t) = E[e^{tX}]$（存在性条件：$\exists h>0$ 使 $\forall t\in(-h,h)$ 期望有限）
- **性质**：
  - $E[X^k] = M^{(k)}(0)$（在 $t=0$ 处的 $k$ 阶导数）
  - $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$（$X,Y$ 独立）
  - $M_{aX+b}(t) = e^{bt} M_X(at)$
- **常见分布的 MGF**：
  - 二项 $B(n,p)$：$(pe^t + 1-p)^n$
  - 泊松 $\mathrm{Poisson}(\lambda)$：$\exp(\lambda(e^t-1))$
  - 指数 $\mathrm{Exp}(\lambda)$：$\frac{\lambda}{\lambda-t}, \; t<\lambda$
  - Gamma $\mathrm{Gamma}(\alpha,\beta)$：$(1 - t/\beta)^{-\alpha}, \; t<\beta$
  - 正态 $N(\mu,\sigma^2)$：$\exp(\mu t + \frac{1}{2}\sigma^2 t^2)$
- **识别技巧**：若 $M(t) = (1 - 2t)^{-3}$，则 $(1 - \frac{t}{1/2})^{-3}$ 对应 $\mathrm{Gamma}(3, 1/2)$，期望 $\alpha/\beta = 6$，方差 $\alpha/\beta^2 = 12$

#### 4.3 特征函数 (Characteristic Function, CF)

- **定义**：$\varphi_X(t) = E[e^{itX}]$（对所有 $t\in\mathbb{R}$ 存在）
- **性质**：
  - $|\varphi(t)| \le 1$，$\varphi(0) = 1$，$\varphi(-t) = \overline{\varphi(t)}$
  - $E[X^k] = i^{-k} \varphi^{(k)}(0)$
  - **唯一性定理**：特征函数与分布函数一一对应（反演公式）
  - **反演公式**：若 $\int |\varphi(t)|\,dt < \infty$，则 $X$ 是连续型且有密度：
    $$f(x) = \frac{1}{2\pi} \int_{-\infty}^{\infty} e^{-itx} \varphi(t)\,dt$$
  - **卷积定理**：若 $X,Y$ 独立，则 $\varphi_{X+Y}(t) = \varphi_X(t)\varphi_Y(t)$
- **常见分布的特征函数**：
  - 泊松 $\mathrm{Poisson}(\lambda)$：$\exp(\lambda(e^{it}-1))$
  - 正态 $N(\mu,\sigma^2)$：$\exp(i\mu t - \frac{1}{2}\sigma^2 t^2)$
  - Laplace：$\frac{1}{1+t^2}$（$\mu=0, b=1$）
  - 柯西分布：$e^{-|t|}$

#### 解题步骤

1. **MGF 求期望方差**：
   - 步骤1：对 $M(t)$ 求一阶导数，代入 $t=0$ 得 $E[X]$
   - 步骤2：求二阶导数，代入 $t=0$ 得 $E[X^2]$
   - 步骤3：$\mathrm{Var}(X) = E[X^2] - (E[X])^2$
   - 步骤4（可选）：若可识别分布类型，直接用该分布的参数公式更快捷
2. **特征函数反演求密度**：
   - 步骤1：识别 $\varphi(t)$ 的形式（对照常见分布的特征函数表）
   - 步骤2：若能识别，直接写出密度
   - 步骤3：若不能识别，通过反演积分 $f(x) = \frac{1}{2\pi}\int e^{-itx}\varphi(t)\,dt$ 计算

#### 常见陷阱

- MGF 对 $t$ 求导时注意链式法则
- 区分 MGF 与 CF：MGF 是 $E[e^{tX}]$，CF 是 $E[e^{itX}]$
- 特征函数反演积分的积分常数：前面有 $1/(2\pi)$
- Gamma 分布的 MGF：注意参数化方式（rate $\beta$ vs scale $\theta$）

#### 对应习题

- **idx 1**: 给定密度 $f(x)=2x, x\in[0,1]$，直接积分求期望 $E[X]$ 和方差 $\mathrm{Var}(X)$
- **idx 11**: 给定 MGF $M(t)=(1-2t)^{-3}$，求 $E[X]$ 和 $\mathrm{Var}(X)$
- **idx 13**: 给定 CF $\varphi(t)=1/(1+t^2)$，反演求得 Laplace 分布 $f(x)=\frac{1}{2}e^{-|x|}$
- **idx 14**: 证明 $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$

---

### 模块5：极限定理

#### 5.1 收敛模式

| 收敛类型 | 记号 | 定义 | 强弱关系 |
|---------|------|------|---------|
| 几乎必然收敛 | $X_n \xrightarrow{a.s.} X$ | $P(\lim X_n = X) = 1$ | 最强 |
| 依概率收敛 | $X_n \xrightarrow{P} X$ | $\forall\varepsilon>0, \lim_{n\to\infty} P(|X_n-X|>\varepsilon) = 0$ | 中等 |
| 依分布收敛 | $X_n \xrightarrow{D} X$ | $\lim F_n(x) = F(x)$ 对 $F$ 的所有连续点 $x$ | 最弱 |

**关系**：几乎必然收敛 $\Rightarrow$ 依概率收敛 $\Rightarrow$ 依分布收敛（但反向不成立）

#### 5.2 大数定律 (Law of Large Numbers, LLN)

- **弱大数定律 (WLLN)**：若 $\{X_n\}$ i.i.d. 且 $E[X_i] = \mu$，则样本均值依概率收敛到 $\mu$：$\bar{X}_n \xrightarrow{P} \mu$
- **强大数定律 (SLLN)**：在相同条件下，$\bar{X}_n \xrightarrow{a.s.} \mu$

#### 5.3 中心极限定理 (Central Limit Theorem, CLT)

- **经典 CLT**：若 $\{X_n\}$ i.i.d.，$E[X_i]=\mu$，$\mathrm{Var}(X_i)=\sigma^2<\infty$，则
  $$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{D} N(0, 1)$$
  即 $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{D} N(0, \sigma^2)$

#### 5.4 依概率收敛推出依分布收敛

- **定理**：$X_n \xrightarrow{P} X \;\Longrightarrow\; X_n \xrightarrow{D} X$
- **证明技巧（重要）**：对任意 $\varepsilon > 0$ 和 $F_X$ 的连续点 $x$：
  - **上界**：
    $$F_{X_n}(x) = P(X_n \le x) \le P(X \le x+\varepsilon) + P(|X_n - X| > \varepsilon)$$
    理由：若 $X_n \le x$ 且 $X > x+\varepsilon$，则 $|X_n - X| > \varepsilon$
  - **下界**：$P(X \le x-\varepsilon) \le P(X_n \le x) + P(|X_n - X| > \varepsilon)$，即
    $$F_{X_n}(x) \ge F_X(x-\varepsilon) - P(|X_n - X| > \varepsilon)$$
  - 取 $n\to\infty$，利用依概率收敛；再取 $\varepsilon \to 0^+$，利用连续性

#### 5.5 Slutsky 定理

- **定理**：若 $X_n \xrightarrow{D} X$ 且 $Y_n \xrightarrow{P} c$（常数），则：
  - $X_n + Y_n \xrightarrow{D} X + c$
  - $X_n \cdot Y_n \xrightarrow{D} cX$
  - $X_n / Y_n \xrightarrow{D} X/c$（若 $c \neq 0$）
- **特例**：当 $Y_n \xrightarrow{P} 0$ 时，$X_n + Y_n \xrightarrow{D} X$
- **证明技巧**：
  - 上界：$P(X_n + Y_n \le x) \le P(X_n \le x+\varepsilon) + P(|Y_n| > \varepsilon)$
  - 下界：$P(X_n + Y_n \le x) \ge P(X_n \le x-\varepsilon) - P(|Y_n| > \varepsilon)$
  - 取 $n\to\infty$ 和 $\varepsilon \to 0^+$（沿 $F_X$ 连续点序列），利用 $Y_n \xrightarrow{P} 0$ 和 $X_n \xrightarrow{D} X$
  - **关键技巧**：$F_X$ 的不连续点至多可数，可选取递减序列 $\varepsilon_k \to 0^+$ 使 $x\pm\varepsilon_k$ 均为连续点

#### 常见陷阱

- 依分布收敛只要求在 $F_X$ 的连续点收敛，不是所有点
- Slutsky 定理的证明中处理 $x\pm\varepsilon$ 是否为连续点需要额外注意（选择连续点序列 $\varepsilon_k$）
- 依概率收敛不保证依分布收敛的反向：$X_n \xrightarrow{D} X \not\Rightarrow X_n \xrightarrow{P} X$（除非 $X$ 是常数）
- 区分各种收敛记号：$P$、$D$、$a.s.$ 不要混淆

#### 对应习题

- **idx 18**: 证明：$X_n \xrightarrow{P} X \Rightarrow X_n \xrightarrow{D} X$
- **idx 19**: 证明 Slutsky 定理特例：$X_n \xrightarrow{D} X$ 且 $Y_n \xrightarrow{P} 0 \Rightarrow X_n+Y_n \xrightarrow{D} X$
- **idx 12**: 两独立 $U(0,1)$ 之和的分布（卷积，设问涉及极限定理的铺垫）

---

### 模块6：不等式与性质

#### 6.1 Markov 不等式

- **定理**：对非负随机变量 $X$（即 $X \ge 0$ a.s.）和任意 $\varepsilon > 0$，
  $$P(X \ge \varepsilon) \le \frac{E[X]}{\varepsilon}$$
- **证明思路**：引入指示函数 $I_{\{X \ge \varepsilon\}}$，证明 $I \le X/\varepsilon$ a.s.，再取期望
  - 当 $X \ge \varepsilon$ 时，$1 \le X/\varepsilon$
  - 当 $X < \varepsilon$ 时，$0 \le X/\varepsilon$（由非负性）

#### 6.2 Chebyshev 不等式

- **定理**：对任意随机变量 $X$（$E[X]=\mu$，$\mathrm{Var}(X)=\sigma^2 < \infty$），对任意 $k > 0$，
  $$P(|X - \mu| \ge k) \le \frac{\sigma^2}{k^2}$$
- 等价形式：$P(|X-\mu| \ge k\sigma) \le \frac{1}{k^2}$
- **推导**：将 Markov 不等式应用于非负随机变量 $(X-\mu)^2$

#### 6.3 方差性质

- 线性变换：$\mathrm{Var}(aX+b) = a^2 \mathrm{Var}(X)$
- 平移不变性：加常数 $b$ 不改变方差
- 尺度平方律：乘常数 $a$，方差乘 $a^2$

#### 常见陷阱

- Markov 不等式要求随机变量非负，这是关键前提
- Chebyshev 不等式给出的是上界，不一定紧
- 方差公式 $\mathrm{Var}(aX+b) = a^2 \mathrm{Var}(X)$ 中不要忘记平方

#### 对应习题

- **idx 15**: 证明 Markov 不等式：$P(X \ge \varepsilon) \le E[X]/\varepsilon$（$X \ge 0$）
- **idx 14**: 证明 $\mathrm{Var}(aX+b) = a^2 \mathrm{Var}(X)$

---

### 模块7：贝叶斯公式与全概率

#### 核心概念

- **全概率公式**：若 $\{B_i\}$ 是样本空间的一个划分，则
  $$P(A) = \sum_i P(A \mid B_i) P(B_i)$$
- **贝叶斯公式**：
  $$P(B_i \mid A) = \frac{P(A \mid B_i) P(B_i)}{P(A)} = \frac{P(A \mid B_i) P(B_i)}{\sum_j P(A \mid B_j) P(B_j)}$$
- **先验概率** $P(B_i)$ vs **后验概率** $P(B_i \mid A)$

#### 解题步骤

1. 定义事件：用字母表示题目中的每个事件（如 $D$ = 患病，$+$ = 阳性）
2. 写出已知概率：$P(D)$（先验），$P(+ \mid D)$（敏感度/真阳性率），$P(+ \mid \neg D)$（假阳性率）
3. 用全概率公式计算 $P(+)$
4. 用贝叶斯公式计算后验概率 $P(D \mid +)$

#### 常见陷阱

- 混淆 $P(A \mid B)$ 与 $P(B \mid A)$（"逆概率谬误"）
- 计算 $P(+ \mid \neg D)$ 时注意：假阳性率是 $P(+ \mid \neg D)$，不是 $P(\neg D \mid +)$
- 全概率公式分母中不要遗漏任何划分

#### 对应习题

- **idx 3**: 疾病检测贝叶斯问题 — 给定 $P(D)=0.01$，$P(+\mid D)=0.99$，$P(+\mid \neg D)=0.02$，求 $P(D\mid +) = 1/3$

---

### 模块8：独立随机变量和的分布（卷积）

#### 核心概念

- **卷积公式**（连续型）：若 $X,Y$ 独立且有密度 $f_X, f_Y$，则 $Z = X+Y$ 的密度为
  $$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z-x)\,dx = (f_X * f_Y)(z)$$
- **卷积公式**（离散型）：$P(Z=k) = \sum_i P(X=i) P(Y=k-i)$
- **MGF/CF 方法**：$\varphi_{X+Y}(t) = \varphi_X(t) \varphi_Y(t)$（由独立性），利用特征函数唯一性得到分布

#### 解题步骤

1. 写出 $f_Z(z) = \int f_X(x) f_Y(z-x)\,dx$
2. 确定 $x$ 的有效积分区域：**同时满足** $f_X(x) > 0$ 和 $f_Y(z-x) > 0$
3. **分情况讨论** $z$ 的取值范围，对不同的 $z$ 区间分别计算积分
4. 整理分段密度函数

#### 常见陷阱

- 积分上下限错误：未正确确定 $z$ 的分段
- 漏掉 $z$ 在某些区间密度为 $0$ 的情况
- 卷积积分的图形直观：$z$ 的范围是 $[a_X+b_X, a_Y+b_Y]$（$a,b$ 为分布支撑区间端点）

#### 对应习题

- **idx 12**: $X,Y \stackrel{i.i.d.}{\sim} U(0,1)$，$Z=X+Y$ 密度为三角分布：$f_Z(z)=z\,(0<z<1),\,2-z\,(1<z<2)$
- **idx 16**: 泊松分布可加性 — 离散卷积，或特征函数法证明

---

## 通用解题方法论

### 1. 题型识别

拿到题目后首先判断：

| 题型特征 | 对应模块 |
|---------|---------|
| 球/抽签/组合数 | 模块1：古典概型 |
| 具体分布名 + 求概率 | 模块2：离散/连续分布 |
| 联合密度/联合分布律 + 求协方差/相关系数 | 模块3：多维随机变量 |
| 已知密度/PMF + 求期望/方差 | 模块4：数字特征 |
| 已知MGF/CF + 求期望/分布 | 模块4：矩母函数/特征函数 |
| "证明XX收敛" / "X_n → Y_n" | 模块5：极限定理 |
| 条件概率 + 先验/后验 | 模块7：贝叶斯公式 |
| "证明不等式" / "Var(aX+b)" | 模块6：不等式与性质 |

### 2. 方法选择

| 问题类型 | 首选方法 | 备选方法 |
|---------|---------|---------|
| 求期望 | 直接积分/求和 | MGF求导、CF求导 |
| 求方差 | $E[X^2] - (E[X])^2$ | MGF $M''(0)-(M'(0))^2$ |
| 求 $X+Y$ 的分布 | 卷积积分 | MGF/CF乘积 |
| 判断可加性 | 特征函数法 | 卷积 + 归纳 |
| 求条件期望 | 先求条件密度 | 二维正态公式直接代 |
| 证明收敛 | 双边不等式 + 取极限 | 利用 Slutsky 等已知定理 |

### 3. 结果验证

- **合理性检查**：概率在 $[0,1]$ 范围内，相关系数在 $[-1,1]$ 范围内
- **量纲检查**：方差是期望平方的量纲
- **边界情况**：验证极端参数时结果是否合理
- **对称性检查**：若 $X,Y$ 对称，则期望和方差应对称
- **sympy 数值验证**：对计算题结果用独立方法验算

---

## sympy 验证技巧

本节给出与配套验证脚本 `test_probability.py` 对应的 sympy 验证方法。

### 积分验证

对于连续型随机变量的期望、方差计算：

```python
import sympy as sp

x = sp.Symbol('x', real=True)

# 例：密度 f(x)=2x, x∈[0,1]
f = 2*x
EX = sp.integrate(x * f, (x, 0, 1))       # 期望
EX2 = sp.integrate(x**2 * f, (x, 0, 1))   # 二阶矩
VarX = sp.simplify(EX2 - EX**2)           # 方差

# 验证结果：E[X]=2/3, Var(X)=1/18
```

对于二维联合分布的积分：

```python
x, y = sp.symbols('x y', real=True)

# 例：f(x,y)=8xy, 0<x<y<1
f_xy = 8*x*y
EXY = sp.integrate(sp.integrate(x*y * f_xy, (y, x, 1)), (x, 0, 1))
EX = sp.integrate(sp.integrate(x * f_xy, (y, x, 1)), (x, 0, 1))
EY = sp.integrate(sp.integrate(y * f_xy, (y, x, 1)), (x, 0, 1))
Cov = sp.simplify(EXY - EX*EY)
```

### 求和验证

对于离散型随机变量：

```python
k = sp.Symbol('k', integer=True, nonnegative=True)

# 例：二项分布 B(5, 1/3), P(X=2)
n, p = 5, sp.Rational(1,3)
prob = sp.binomial(n, 2) * p**2 * (1-p)**(n-2)
# 结果：80/243
```

对于泊松分布：

```python
import sympy as sp

lam = 3
# P(X=2) = e^{-λ} λ²/2!
p2 = sp.exp(-lam) * lam**2 / 2
# P(X≥1) = 1 - e^{-λ}
p_ge1 = 1 - sp.exp(-lam)
```

### 特征函数验证

```python
t = sp.Symbol('t', real=True)

# 验证特征函数 φ(t) = 1/(1+t²) 对应 Laplace(0,1)
# 通过 Fourier 反演验证：
phi = 1/(1 + t**2)
# sympy 的 inverse_fourier_transform 可用于验证
```

### 线性代数验证

对于协方差矩阵和相关系数的计算：

```python
# 二维正态条件分布验证
rho = sp.Rational(1,2)
sigma_X, sigma_Y = 1, 2
mu_X, mu_Y = 0, 0

# E[Y|X=1] = μ_Y + ρ·(σ_Y/σ_X)·(1 - μ_X)
EY_given_X1 = mu_Y + rho * (sigma_Y/sigma_X) * (1 - mu_X)
# = 0 + (1/2)·(2/1)·1 = 1

# Var(Y|X=1) = σ_Y²·(1 - ρ²)
VarY_given_X1 = sigma_Y**2 * (1 - rho**2)
# = 4·(1 - 1/4) = 3
```

### 数值近似验证

对于复杂积分，可使用 sympy 的数值积分 `.evalf()` 快速验证：

```python
# 例：验证 P(Z>1) = 1/2 (两独立U(0,1)之和)
z = sp.Symbol('z', real=True)
# f_Z(z) = z (0<z<1), 2-z (1<z<2)
p = sp.integrate(2 - z, (z, 1, 2))
sp.simplify(p)  # 1/2
```

### 卷积验证

```python
# 两独立 U(0,1) 之和
x, z = sp.symbols('x z', real=True)
# f_Z(z) = ∫ f_X(x) f_Y(z-x) dx
# f_X(x)=1 for x∈[0,1], f_Y(y)=1 for y∈[0,1]
# 当 0<z<1: x∈[0,z]
f1 = sp.integrate(1, (x, 0, z))          # = z
# 当 1<z<2: x∈[z-1,1]
f2 = sp.integrate(1, (x, z-1, 1))        # = 2-z
```

---

## 习题索引

| idx | 题型 | 难度 | 主题 | 关键知识点 | 核心公式/方法 | 验证方法 |
|-----|------|------|------|-----------|-------------|---------|
| 0 | 计算 | 简单 | 古典概型 | 超几何分布 | $P = \frac{C_5^2 C_3^1}{C_8^3} = \frac{15}{28}$ | 组合数直接计算 |
| 1 | 计算 | 简单 | 期望与方差 | 连续密度积分 | $E[X]=\int x f(x)dx$, $\mathrm{Var}(X)=E[X^2]-(E[X])^2$ | sympy积分 |
| 2 | 计算 | 简单 | 二项分布 | pmf公式 | $P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}$ | 组合数+幂运算 |
| 3 | 计算 | 中等 | 贝叶斯公式 | 全概率+后验概率 | $P(D\mid+)=\frac{P(+\mid D)P(D)}{P(+)}$ | 有理数运算 |
| 4 | 计算 | 中等 | 二维连续分布 | 联合密度积分/协方差 | $E[XY]=\iint xy f(x,y)$, $\mathrm{Cov}=E[XY]-E[X]E[Y]$ | sympy二重积分 |
| 5 | 计算 | 中等 | 相关系数 | 离散联合分布律 | $\rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X\sigma_Y}$ | 离散随机变量逐项计算 |
| 6 | 计算 | 中等 | 随机变量函数的分布 | 分布函数法 | $F_Y(y)=P(-\ln X \le y) \Rightarrow Y\sim\mathrm{Exp}(1)$ | 变量变换+分布识别 |
| 7 | 计算 | 中等 | 二维正态条件分布 | 条件期望/方差公式 | $E[Y\mid X=x]=\mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X)$ | 二维正态公式直接代入 |
| 8 | 计算 | 中等 | 条件期望 | 条件密度求法 | $f_{X\mid Y}(x\mid y)=\frac{f(x,y)}{f_Y(y)}$ | 边缘密度+条件密度积分 |
| 9 | 计算 | 中等 | 泊松分布 | pmf $e^{-\lambda}\lambda^k/k!$ | $P(X\ge 1)=1-P(X=0)$ | sympy求和/exp函数 |
| 10 | 计算 | 中等 | 指数分布高阶矩 | Gamma积分 | $E[X^n]=n!/\lambda^n$ | sympy积分验证 |
| 11 | 计算 | 中等 | 矩母函数 | MGF求导/分布识别 | $M(t)=(1-2t)^{-3}$, $E[X]=M'(0)$ | sympy求导+代入 |
| 12 | 计算 | 困难 | 独立RV和的分布 | 卷积公式 | $f_Z(z)=\int f_X(x)f_Y(z-x)dx$ | sympy分段积分 |
| 13 | 计算 | 困难 | 特征函数反演 | CF唯一性 | $\varphi(t)=1/(1+t^2) \Leftrightarrow f(x)=\frac{1}{2}e^{-|x|}$ | CF对照表+反演积分 |
| 14 | 证明 | 简单 | 方差的性质 | 线性变换 | $\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$ | 代数展开验证 |
| 15 | 证明 | 中等 | Markov不等式 | 指示函数技巧 | $P(X\ge\varepsilon)\le E[X]/\varepsilon$ | 不等式逻辑推导 |
| 16 | 证明 | 中等 | 泊松可加性 | CF法/卷积法 | $\varphi_{X+Y}=\exp((\lambda_1+\lambda_2)(e^{it}-1))$ | CF乘积+唯一性 |
| 17 | 证明 | 中等 | 指数无记忆性 | 条件概率+生存函数 | $\frac{P(X>s+t)}{P(X>s)} = e^{-\lambda t}$ | 代数化简 |
| 18 | 证明 | 困难 | 依概率→依分布 | 双边不等式 | $F_X(x-\varepsilon)-\delta \le F_n(x) \le F_X(x+\varepsilon)+\delta$ | $\varepsilon$-$\delta$ 型逻辑 |
| 19 | 证明 | 困难 | Slutsky定理 | 不等式+连续点选取 | $P(X_n\le x-\varepsilon)-o(1) \le P(X_n+Y_n\le x) \le P(X_n\le x+\varepsilon)+o(1)$ | 双极限论证 |

---

## 常见错误与陷阱

### 古典概型
1. **放回 vs 不放回混淆**：放回用二项分布 $C_n^k p^k(1-p)^{n-k}$；不放回用超几何分布 $\frac{C_K^k C_{N-K}^{n-k}}{C_N^n}$
2. **"至少"漏取补**：$P(\text{至少}1) = 1 - P(0)$，不要直接累加
3. **组合数与排列数混淆**：组合 $\binom{n}{k}$ 不计顺序，排列 $P_n^k$ 计顺序

### 连续型分布
4. **指数分布参数化混乱**：$f(x)=\lambda e^{-\lambda x}$ 的期望是 $1/\lambda$，$f(x)=\frac{1}{\theta}e^{-x/\theta}$ 的期望是 $\theta$。做题前先确认使用的是哪种参数化
5. **正态分布标准化错误**：$Z = \frac{X-\mu}{\sigma} \sim N(0,1)$，不要忘记除以 $\sigma$
6. **变量变换漏掉绝对值**：$f_Y(y) = f_X(g^{-1}(y)) \cdot |d(g^{-1})/dy|$，注意雅可比行列式的绝对值

### 多维随机变量
7. **积分上下限错误**：非矩形区域的二重积分需要正确分段，例如 $0<x<y<1$ 时 $x$ 的范围是 $[0, y]$ 或 $[0, 1]$
8. **协方差不等于独立**：$\mathrm{Cov}(X,Y)=0$ 不意味着 $X$ 与 $Y$ 独立（二维正态除外）
9. **条件期望是随机变量**：$E[X\mid Y]$ 是 $Y$ 的函数，不要写成常数
10. **相关系数计算**：$\rho = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}$，分母是标准差的乘积，不是方差的乘积

### 数字特征
11. **方差公式**：牢记 $\mathrm{Var}(X) = E[X^2] - (E[X])^2$，比直接展开定义式快得多
12. **MGF求导**：$M^{(k)}(0) = E[X^k]$，注意 $t=0$ 代入后才能得到矩
13. **Gamma 分布识别**：$M(t) = (1 - \beta t)^{-\alpha}$ 对应 $\mathrm{Gamma}(\alpha, \beta)$（rate参数化），期望 $\alpha/\beta$，方差 $\alpha/\beta^2$
14. **CF与MGF的区分**：$\varphi(t) = E[e^{itX}]$ vs $M(t) = E[e^{tX}]$，两者相差一个 $i$ 因子

### 极限定理
15. **依分布收敛点要求**：只在分布函数的连续点处收敛，证明时需处理不连续点（可数个）
16. **Slutsky 定理条件**：$Y_n \xrightarrow{P} c$ 要求 $c$ 是常数（不能是随机变量）
17. **收敛方向不可逆**：依分布收敛是最弱的，不能反推依概率收敛或几乎必然收敛
18. **$\varepsilon$ 与 $n$ 的取极限顺序**：先取 $n\to\infty$，再取 $\varepsilon\to 0^+$，顺序不可交换

### 不等式
19. **Markov 不等式的前提**：随机变量必须非负 $X \ge 0$，否则结论不成立
20. **Chebyshev 是 Markov 的推论**：将 Markov 不等式应用于 $(X - E[X])^2$ 即可得到 Chebyshev 不等式

### 贝叶斯公式
21. **颠倒先验和后验**：所求的是 $P(\text{患病} \mid \text{阳性})$，不是 $P(\text{阳性} \mid \text{患病})$。前者是后验概率，后者是似然函数
22. **遗漏全概率分母**：贝叶斯公式分母必须包含所有可能的路径（患病且阳性 + 未患病且阳性）

---

## 分布速查表

### 离散分布

| 分布 | 记号 | PMF $P(X=k)$ | $E[X]$ | $\mathrm{Var}(X)$ | MGF $M(t)$ | CF $\varphi(t)$ |
|------|------|-------------|--------|-------------------|------------|-----------------|
| 二项 | $B(n,p)$ | $\binom{n}{k}p^k q^{n-k}$ | $np$ | $npq$ | $(pe^t+q)^n$ | $(pe^{it}+q)^n$ |
| 泊松 | $\mathrm{Pois}(\lambda)$ | $e^{-\lambda}\lambda^k/k!$ | $\lambda$ | $\lambda$ | $\exp(\lambda(e^t-1))$ | $\exp(\lambda(e^{it}-1))$ |
| 几何 | $\mathrm{Geom}(p)$ | $pq^{k-1}$ | $1/p$ | $q/p^2$ | $\frac{pe^t}{1-qe^t}$ | $\frac{pe^{it}}{1-qe^{it}}$ |
| 超几何 | $\mathrm{Hyp}(N,K,n)$ | $\frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}$ | $nK/N$ | $n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}$ | — | — |

### 连续分布

| 分布 | 记号 | 密度 $f(x)$ | $E[X]$ | $\mathrm{Var}(X)$ | MGF $M(t)$ | CF $\varphi(t)$ |
|------|------|------------|--------|-------------------|------------|-----------------|
| 均匀 | $U(a,b)$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | $\frac{e^{tb}-e^{ta}}{t(b-a)}$ | $\frac{e^{itb}-e^{ita}}{it(b-a)}$ |
| 指数 | $\mathrm{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | $\frac{\lambda}{\lambda-t}$ | $\frac{\lambda}{\lambda-it}$ |
| 正态 | $N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | $\exp(\mu t+\frac{\sigma^2 t^2}{2})$ | $\exp(i\mu t-\frac{\sigma^2 t^2}{2})$ |
| Gamma | $\mathrm{Gamma}(\alpha,\beta)$ | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | $(1-t/\beta)^{-\alpha}$ | $(1-it/\beta)^{-\alpha}$ |
| Laplace | $\mathrm{Laplace}(\mu,b)$ | $\frac{1}{2b}e^{-|x-\mu|/b}$ | $\mu$ | $2b^2$ | $\frac{e^{\mu t}}{1-b^2 t^2}$ | $\frac{e^{i\mu t}}{1+b^2 t^2}$ |

---

## 参考资源

- 配套 sympy 验证脚本：`test_probability.py`
- 数据集来源：`概率论.md`（20题，涵盖全部核心知识点）
- 推荐教材：《概率论与数理统计》（茆诗松、程依明、濮晓龙 编著）
- 在线工具：Wolfram Alpha、sympy live
