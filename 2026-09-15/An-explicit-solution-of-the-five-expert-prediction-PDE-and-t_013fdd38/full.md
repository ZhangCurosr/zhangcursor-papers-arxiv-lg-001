# An explicit solution of the five-expert prediction PDE and the exact optimality set of COMB<sup>∗</sup>

Jef Calder<sup>1</sup> and Nadejda Drenska<sup>2</sup>

<sup>1</sup>School of Mathematics, University of Minnesota <sup>2</sup>Department of Mathematics, Louisiana State University

September 15, 2026

## Abstract

In this paper, we derive an explicit solution of the stationary prediction with expert advice PDE for five experts. The formula is given in three regions. In the first two regions, it is the four-expert solution plus a single integral with an elementary positive density. In the third region, it is a finite sum of hyperbolic products whose coeficients are determined by one scalar quadrature. Our formula establishes that the direction (1, 0, 1, 0, 0) is optimal throughout the ordered sector, and that the COMB strategy (1, 0, 1, 0, 1) is optimal only on a lower dimensional subset of the sector (where $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 } )$ . This disproves the COMB optimality conjecture of Gravin, Peres and Sivan [21]. The verification of the Hamiltonian inequalities is a tedious task, part of which is completed with a computer assisted proof. The verification reduces to 21 scalar inequalities, which we prove using 147 exact rational Bernstein polynomial certificates. The exact certificates and their independent arithmetic checks are included in a supplement to this paper.

## 1 Introduction

Prediction with expert advice is one of the oldest problems in online learning. A player (an “investor”), observing the past performance of n experts, repeatedly chooses which expert to follow; an adversary (a “market” or “nature”) simultaneously controls and decides which experts are correct. The player’s performance is measured by a function of regret, defined as the gap between the best expert’s cumulative gain and the player’s own gain. The problem originates with Cover [15] and Hannan [23]. Cover solved the two-expert case exactly and established the $O ( \sqrt { T } )$ regret rate over T rounds. The subsequent development, surveyed in Cesa-Bianchi and Lugosi [13], concentrated on algorithms with good worst-case guarantees rather than exact minimax play: the weighted-majority and aggregating strategies of Littlestone–Warmuth [29] and Vovk [33], and more generally the multiplicative-weights (Hedge) family [12, 20], achieve regret $O ( { \sqrt { T \log n } } )$ , and Haussler, Kivinen and Warmuth [24] showed this order is asymptotically unimprovable in the worst case. The setting above is the finite-horizon game, played for a known number of rounds $T .$ Variants change the stopping rule: play until an expert has incurred a fixed number of losses [2], or until a horizon that is random and unknown to the player [30]. An important special case of the latter variant is known as the geometric stopping game, which ends with probability $\delta$ at each step—a random horizon whose law is known to both players. The geometric stopping game is the main object of study in this paper.

Beyond the two expert work of Cover [15], exact minimax strategies remained out of reach for fifty years. The first breakthrough was due to Gravin, Peres and Sivan [21], who studied the geometric-stopping game with randomized strategies for both players and proved that for three experts the adversary’s optimal strategy is COMB: sort the experts by current regret, then advance all odd-ranked experts with probability $\frac { 1 } { 2 }$ and all even-ranked experts with probability ${ \frac { 1 } { 2 } } .$ . Their argument relies on an exponential ansatz for the value function that does not extend to $n \geq 4$ , and they conjectured that COMB is asymptotically optimal for any number of experts, computing the conjectured minimax regret $\pi / ( 4 \sqrt { 2 \delta } )$ in the four-expert case. In the finite-horizon setting, Abbasi-Yadkori, Bartlett and Gabillon [1] gave near-minimax strategies for three experts, again with a COMB-type adversary, and Gravin, Peres and Sivan [22] proved tight lower bounds for multiplicative-weights families, showing that no algorithm in that class matches the minimax constant.

A conceptually diferent route was opened by Drenska and Kohn [17, 18]. Following the approach of Kohn and Serfaty [27,28], who interpreted two-person games as discretizations of nonlinear PDEs, and the tug-of-war games of Peres, Schramm, Shefield and Wilson [31, 32] for the infinity- and p-Laplacians, they treat the dynamic programming principle of the prediction game as a numerical scheme for a nonlinear equation (“numerical analysis in reverse”). Using the Barles–Souganidis framework [3] and the viscosity theory of Crandall, Ishii and Lions [16], they proved that the rescaled value function converges, as $T \to \infty$ or $\delta \to 0$ , to the unique viscosity solution of a degenerate parabolic equation in the finite-horizon case and of the degenerate elliptic equation

$$
u ( x ) - \frac { 1 } { 2 } \operatorname* { m a x } _ { \mathbf { v } \in \{ 0 , 1 \} ^ { n } } \mathbf { v } ^ { \mathsf { T } } \nabla ^ { 2 } u ( x ) \mathbf { v } = \operatorname* { m a x } _ { i } x _ { i } , \qquad x \in \mathbb { R } ^ { n } ,\tag{1.1}
$$

in the geometric-stopping case. The solution encodes both minimax strategies: the player follows expert i with probability $\partial _ { i } u ,$ and the adversary’s optimal moves are the binary vectors attaining the maximum in (1.1). Their analysis covers a general class of symmetric, Lipschitz, translation-invariant payofs, and for $n = 3$ they solved the elliptic equation explicitly, recovering the Gravin–Peres–Sivan result as a continuum limit. The same framework has since been used to obtain potential-based regret bounds for general n by Kobzar, Kohn and Wang [25,26], to treat history-dependent experts [8,9,19], and to analyze limited or malicious adversaries [5, 7].

The four-expert case was settled by Bayraktar, Ekren and Zhang [6]. Working from the Drenska–Kohn PDE, they represented the value function under the conjectured COMB control as a discounted expectation of the local time of an obliquely reflected Brownian motion in an orthant [34] (from the orthant, one can extend the result to $\mathbb { R } ^ { n } )$ , the local time counting crossings between the two leading experts. Diferentiating the dynamic programming principle on the faces of the reflection domain led to a system of first-order hyperbolic equations for the boundary values, which they solved in closed form; a type of a maximum principle for this system then made the verification of the full nonlinear PDE tractable. The result is an explicit formula for $u ,$ the exact leading-order value $\pi / ( 4 \sqrt { 2 \delta } )$ , and the asymptotic optimality of COMB for $n = 4$ , confirming both conjectures of [21] in the four-expert case. A companion paper [4] treats the finite-horizon four-expert problem.

For $n \geq 5$ experts the picture changes substantially. Numerical experiments by Chase [14] gave the first indication that COMB is not asymptotically optimal for larger n. Calder, Drenska and Mosaphir [11] then developed numerical methods for the elliptic equation itself: a localization theorem shows the equation can be solved on a box $[ - T , T ] ^ { n }$ , with a bound on the efect of boundary errors on a smaller interior box that decays as $T$ grows. Permutation symmetry reduces the computational domain to the ordered sector $x _ { 1 } \geq \cdot \cdot \cdot \geq x _ { n - 1 } \geq x _ { n } = 0$ making fine-grid solutions feasible up to $n = 1 0$ experts. Their computations support the nonoptimality of COMB for $n \geq 5$ and single out $( 1 , 0 , 1 , 0 , 0 )$ , in rank order, as the adversary’s optimal direction for five experts. The theory is thus complete for $n \leq 4$ , and $n = 5$ is the first case in which the optimal adversarial strategy is expected to depart from COMB.

In this paper we give an explicit solution of the five-expert PDE (1.1) and use it to settle the question of COMB’s optimality for $n = 5$ . Our first main result, Theorem 2.1, is that the rank direction $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ , identified numerically in [11], attains the Hamiltonian maximum throughout the ordered sector $x _ { 1 } \geq \dots \geq x _ { 5 }$ . The solution is given in three regions of the sector. In the first two regions, it reduces to the four-expert solution of [6] plus a single integral of elementary functions against an explicit positive density; in the third, it involves a finite sum of hyperbolic products whose four coeficients are determined by a single scalar quadrature. Our formula proves that the value of the game at the origin is $u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ , to be compared with the values ${ \sqrt { 2 } } / 4 , { \sqrt { 2 } } / 3$ , and $\pi / ( 4 { \sqrt { 2 } } )$ for two, three, and four experts, the first two obtained from the exponential sums (3.2) and (3.3). We also obtain the second-order Taylor expansion of u at the origin, the five-expert analogue of [6, equation (3.4)].

The derivation is purely analytic and avoids the stochastic representation used in [6]. Solving (1.1) along the characteristic direction $\mathbf { v } _ { * }$ reduces the problem to a one-dimensional two-point boundary value problem, and the permutation-symmetry conditions on the faces of the sector produce coupled systems of first-order hyperbolic PDEs for the boundary traces. The key observation is that these systems can be solved explicitly: a family of truncated hyperbolic modes diagonalizes the trace operators, and the resulting propagators admit a positive Green’s function. As a by-product, the same method rederives the four-expert solution of [6] in a more direct way, and it explains why the cases $n = 2 , 3$ are elementary while $n \geq 4$ are not.

Verifying that the resulting formula is the viscosity solution of (1.1) is the bulk of the work in the paper. We prove that u is globally $C ^ { 2 }$ across the region interfaces and the permutation hyperplanes, that $u - \operatorname* { m a x } _ { i } x _ { i }$ is bounded, and that u is the unique viscosity solution in this class. The Hamiltonian inequalities $D _ { \mathbf { v } } ^ { 2 } u \le D _ { v _ { * } } ^ { 2 } u$ for all sixteen non-equivalent binary controls v are reduced, by multiafine interpolation in hyperbolic tangents and by sign principles for Laplace-type integrals, to twenty-one scalar inequalities in one variable.

We prove these with 147 exact rational Bernstein-polynomial certificates; the certificates and independent arithmetic checkers are included in a self-contained, ofline-reproducible supplement, and no floating-point or numerical sign sampling enters the proof.

Our second main result, Theorem 2.2, determines exactly where COMB is optimal for five experts. In decreasing rank order, COMB attains the Hamiltonian maximum if and only if $x _ { 1 } = x _ { 2 }$ and $x _ { 3 } = x _ { 4 } ;$ at every other point the curvature gap between $\mathbf { v } _ { * }$ and COMB is strictly positive. The optimality set is thus a three-dimensional set with empty interior in $\mathbb { R } ^ { 5 }$ . This confirms, in sharp form, the non-optimality of COMB conjectured on numerical grounds in [14] and [11], and shows that the pattern established for $n \leq 4$ does not persist. The optimal control is nevertheless not unique: several other rank-based controls attain the maximum on each region, and we list them in Remark 2.3. However, v<sub>∗</sub> is the only control optimal throughout the entire ordered sector, which also departs from the $n \leq 4$ cases where there were always two such strategies.

## 2 Main results

We consider the five expert PDE

$$
u ( x ) - \frac { 1 } { 2 } \operatorname* { m a x } _ { \mathbf { v } \in \{ 0 , 1 \} ^ { 5 } } \mathbf { v } ^ { T } \nabla ^ { 2 } u ( x ) \mathbf { v } = \varphi ( x ) , \qquad \varphi ( x ) = \operatorname* { m a x } _ { 1 \leq i \leq 5 } x _ { i } , \qquad x \in \mathbb { R } ^ { 5 } .\tag{2.1}
$$

We work in the ordered sector $\{ x _ { 1 } \geq x _ { 2 } \geq x _ { 3 } \geq x _ { 4 } \geq x _ { 5 } \}$ and set $k = \sqrt { 2 }$ . We define the scaled gaps and the correction F by

$$
y _ { i } = k ( x _ { i } - x _ { i + 1 } ) \geq 0 , \qquad u ( x ) = x _ { 1 } + { \frac { 1 } { k } } F ( y _ { 1 } , y _ { 2 } , y _ { 3 } , y _ { 4 } ) .\tag{2.2}
$$

Then the ordered sector is mapped to the positive orthant $\{ y _ { i } \geq 0 \}$ in gap coordinates. The explicit formula for the solution of (2.1) requires diferent coordinate systems in diferent regions in the sector. For the leading four experts we set

$$
z _ { 1 } = { \frac { x _ { 1 } + x _ { 2 } - x _ { 3 } - x _ { 4 } } { k } } , \qquad z _ { 2 } = { \frac { x _ { 1 } - x _ { 2 } + x _ { 3 } - x _ { 4 } } { k } } , \qquad z _ { 3 } = { \frac { x _ { 1 } - x _ { 2 } - x _ { 3 } + x _ { 4 } } { k } } , \qquad z _ { 4 } = { \frac { x _ { 2 } - x _ { 3 } - x _ { 4 } } { k } } .\tag{2.3}
$$

so that the ordered sector becomes $\{ z _ { 1 } \geq z _ { 2 } \geq | z _ { 3 } | \}$ , since $\begin{array} { r } { k / 2 = 1 / k , z _ { 1 } = y _ { 2 } + \frac { 1 } { 2 } ( y _ { 1 } + y _ { 3 } ) } \end{array}$ $\begin{array} { r } { z _ { 2 } = \frac { 1 } { 2 } ( y _ { 1 } + y _ { 3 } ) } \end{array}$ , and $z _ { 3 } = { \textstyle { \frac { 1 } { 2 } } } ( y _ { 1 } - y _ { 3 } )$ . The fifth expert enters through

$$
z _ { 4 } = y _ { 4 } - \operatorname* { m a x } ( z _ { 3 } , 0 ) .\tag{2.4}
$$

The sector splits into three regions:

<table><tr><td></td><td>Region Scaled gaps</td><td>Coordinates</td></tr><tr><td>I</td><td> $y _ { 1 } \le y _ { 3 }$ </td><td> $z _ { 3 } \le 0 \le z _ { 4 }$ </td></tr><tr><td>II</td><td> $y _ { 3 } \leq y _ { 1 } \leq y _ { 3 } + 2 y _ { 4 }$ </td><td> $z _ { 3 } \ge 0 , z _ { 4 } \ge 0$ </td></tr><tr><td>III</td><td> $y _ { 1 } \geq y _ { 3 } + 2 y _ { 4 }$ </td><td> $z _ { 4 } \le 0$ </td></tr></table>

The interfaces are $z _ { 3 } = 0$ , where $y _ { 1 } = y _ { 3 }$ , and $z _ { 4 } = 0$ , where $y _ { 1 } = y _ { 3 } + 2 y _ { 4 }$ . In Region III we instead use the coordinates

$$
a _ { 1 } = \frac { y _ { 1 } - y _ { 3 } - 2 y _ { 4 } } { 3 } , \quad a _ { 2 } = a _ { 1 } + y _ { 4 } , \quad a _ { 3 } = a _ { 2 } + y _ { 3 } , \quad a _ { 4 } = a _ { 3 } + y _ { 2 } ,\tag{2.5}
$$

for which $0 \leq a _ { 1 } \leq a _ { 2 } \leq a _ { 3 } \leq a _ { 4 } ;$ conversely

$$
y _ { 1 } = a _ { 1 } + a _ { 2 } + a _ { 3 } , \quad y _ { 2 } = a _ { 4 } - a _ { 3 } , \quad y _ { 3 } = a _ { 3 } - a _ { 2 } , \quad y _ { 4 } = a _ { 2 } - a _ { 1 } .\tag{2.6}
$$

The two systems are related by $a _ { 1 } = - { \textstyle \frac { 2 } { 3 } } z _ { 4 }$ and $a _ { 2 } = z _ { 3 } + { \textstyle \frac { 1 } { 3 } } z _ { 4 } , a _ { 3 } = z _ { 2 } + { \textstyle \frac { 1 } { 3 } } z _ { 4 } , a _ { 4 } = z _ { 1 } + { \textstyle \frac { 1 } { 3 } } z _ { 4 }$ so they coincide on the interface $z _ { 4 } = 0$ , where $a _ { 1 } = 0$ and $( a _ { 4 } , a _ { 3 } , a _ { 2 } ) = ( z _ { 1 } , z _ { 2 } , z _ { 3 } )$

The formula is built from the following functions. Write $\theta ( L ) = \arctan ( e ^ { - L } )$ and $\lambda ( L ) = \log \coth ( L / 2 )$ . Let

$$
I ( X ) = \int _ { 0 } ^ { X } \sinh ^ { 4 } t \cosh ^ { 2 } t d t = { \frac { \sinh 6 X } { 1 9 2 } } - { \frac { \sinh 4 X } { 6 4 } } - { \frac { \sinh 2 X } { 6 4 } } + { \frac { X } { 1 6 } } .\tag{2.7}
$$

The trace $e ,$ the only function in the formula that is not elementary, is defined by the quadrature

$$
e ( X ) = 3 \cosh X \int _ { X } ^ { \infty } { \frac { I ( t ) } { \cosh ^ { 2 } t \sinh ^ { 5 } t } } d t .\tag{2.8}
$$

The integrand is regular at zero, since $I ( t ) = t ^ { 5 } / 5 + \mathcal { O } ( t ^ { 7 } )$ , and decays like $e ^ { - t } / 3$ at infinity, so e is positive and bounded. It satisfies the first-order equation

$$
e ^ { \prime } = \operatorname { t a n h } X e - \frac { 3 I ( X ) } { \cosh X \sinh ^ { 5 } X } ,\tag{2.9}
$$

and consequently

$$
e ^ { \prime \prime } = - 5 \coth X e ^ { \prime } + 6 e - 3 \coth X ,\tag{2.10}
$$

$$
e ^ { \prime \prime \prime } = ( 3 1 + 3 0 \operatorname { c s c h } ^ { 2 } X ) e ^ { \prime } - 3 0 \coth X e + 1 5 + 1 8 \operatorname { c s c h } ^ { 2 } X ;
$$

these are proved in Section 3.3. We define the density

$$
p ( X ) = \frac { 1 } { 2 \sinh ^ { 6 } X } \int _ { 0 } ^ { X } \sinh ^ { 6 } t ~ \mathrm { s e c h } ^ { 2 } t d t ,\tag{2.11}
$$

which has the elementary form

$$
p ( X ) = \textstyle { \frac { 1 } { 2 } } \operatorname { t a n h } X - 3 \coth X + \frac { 1 5 I ( X ) } { \sinh ^ { 6 } X } .\tag{2.12}
$$

Finally, we define the truncated modes

$$
\Phi _ { t } ( X ) = \frac { \sinh ( t - X ) } { \sinh t } \quad ( 0 \leq X \leq t ) , \qquad \Phi _ { t } ( X ) = 0 \quad ( X > t ) .\tag{2.13}
$$

The four-expert solution enters the formula as a background term. We define

$$
\begin{array} { r } { F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) = \theta ( z _ { 1 } ) \cosh z _ { 1 } \cosh z _ { 2 } \cosh z _ { 3 } \qquad } \\ { + \frac 1 2 \lambda ( z _ { 1 } ) \sinh z _ { 1 } \sinh z _ { 2 } \sinh z _ { 3 } - \frac 1 2 \sinh ( z _ { 2 } + z _ { 3 } ) . } \end{array}\tag{2.14}
$$

Then $\begin{array} { r } { u _ { 4 } = x _ { 1 } + \frac { 1 } { k } F _ { 4 } } \end{array}$ is the explicit four-expert solution of Bayraktar, Ekren, and Zhang [6, Theorem 3.1] in the present notation; the identification is given in Section 3.5. We are now ready to state our main result.

Theorem 2.1. Let $\begin{array} { r } { u ( x ) = x _ { 1 } + \frac { 1 } { k } F ( y ) } \end{array}$ on the ordered sector, where in Regions I and II

$$
F = F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) + \int _ { z _ { 1 } } ^ { \infty } \sinh t p ( t ) e ^ { - 2 z _ { 4 } \coth t } \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( | z _ { 3 } | ) \Phi _ { t } ( z _ { 2 } ) d t ,\tag{2.15}
$$

and in Region III

$$
\begin{array} { r l } & { F = b _ { 0 } \cosh a _ { 1 } \cosh a _ { 2 } \cosh a _ { 3 } } \\ & { \qquad + b _ { 1 } ( \sinh a _ { 1 } \cosh a _ { 2 } \cosh a _ { 3 } + \cosh a _ { 1 } \sinh a _ { 2 } \cosh a _ { 3 } + \cosh a _ { 1 } \cosh a _ { 2 } \sinh a _ { 3 } ) } \\ & { \qquad + b _ { 2 } ( \sinh a _ { 1 } \sinh a _ { 2 } \cosh a _ { 3 } + \sinh a _ { 1 } \cosh a _ { 2 } \sinh a _ { 3 } + \cosh a _ { 1 } \sinh a _ { 2 } \sinh a _ { 3 } ) } \\ & { \qquad + b _ { 3 } \sinh a _ { 1 } \sinh a _ { 2 } \sinh a _ { 3 } , } \end{array}\tag{2.16}
$$

with e and its derivatives evaluated at $a _ { 4 }$

$$
\begin{array} { r l } & { b _ { 0 } = e , \qquad b _ { 1 } = ( e ^ { \prime } - 3 ) / 6 , } \\ & { b _ { 2 } = ( e ^ { \prime \prime } + 6 e + 1 8 \lambda ( a _ { 4 } ) \sinh a _ { 4 } ) / 4 2 , } \\ & { b _ { 3 } = ( e ^ { \prime \prime \prime } + 2 0 e ^ { \prime } ) / 3 3 6 + ( 6 \lambda ( a _ { 4 } ) \cosh a _ { 4 } - 1 3 ) / 1 4 , } \end{array}\tag{2.17}
$$

and extend u to $\mathbb { R } ^ { 5 }$ by sorting the coordinates. Then $u \in C ^ { 2 } ( \mathbb { R } ^ { 5 } ) , u - \varphi$ is bounded, and u is the unique viscosity solution of (2.1) in this class. In the ordered sector, the rank direction $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ attains the Hamiltonian maximum everywhere and

$$
u ( x ) = { \frac { 4 5 \pi ^ { 2 } } { 5 1 2 \sqrt { 2 } } } + { \frac { 1 } { 5 } } \sum _ { i = 1 } ^ { 5 } x _ { i } + { \frac { 1 5 \pi ^ { 2 } } { 2 5 6 \sqrt { 2 } } } \Bigl ( \sum _ { i = 1 } ^ { 5 } x _ { i } ^ { 2 } - { \frac { 1 } { 2 } } \sum _ { i < j } x _ { i } x _ { j } \Bigr ) + o ( | x | ^ { 2 } ) a s x \to 0 .\tag{2.18}
$$

The two expressions for F agree on the interface $z _ { 4 } = 0$ (Section 3.6). In Regions I and II the formula requires one quadrature of elementary functions; in Region III only the value $e ( a _ { 4 } )$ is not elementary, since (2.9) and (2.10) eliminate the derivatives in (2.17). The constant term $u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ in (2.18) is obtained in Lemma 3.1 by evaluating the quadrature (2.8) at $X = 0 ;$ it is the five-expert counterpart of the four-expert value $u _ { 4 } ( 0 ) = \pi / ( 4 \sqrt { 2 } )$ of [6]. The expansion (2.18) is the five-expert analogue of [6, equation (3.4)] and is proved in Section 4 from the $C ^ { 2 }$ regularity, the symmetries, and the equation at the origin.

Our second result identifies the set on which the COMB control is optimal. In decreasing rank order, COMB is $\mathbf { v } _ { C } = ( 1 , 0 , 1 , 0 , 1 )$ . Its complement $\mathbf { 1 } - \mathbf { v } _ { C } = ( 0 , 1 , 0 , 1 , 0 )$ has the same curvature, because $\nabla ^ { 2 } u \mathbb { 1 } = 0$ . Define the physical curvature gap

$$
\Delta ( x ) = D _ { { \bf v } _ { \ast } } ^ { 2 } u ( x ) - D _ { { \bf v } _ { C } } ^ { 2 } u ( x ) .\tag{2.19}
$$

Since $\mathbf { v } _ { * }$ attains the Hamiltonian maximum by Theorem 2.1, $\Delta \geq 0$ everywhere in the ordered sector. The following theorem identifies the set where equality holds.

Theorem 2.2. For $x _ { 1 } \geq \dots \geq x _ { 5 }$ , COMB attains the Hamiltonian maximum, i.e., $\Delta ( x ) = 0$ if and only if

$$
x _ { 1 } = x _ { 2 } a n d x _ { 3 } = x _ { 4 } .\tag{2.20}
$$

At every other point $\Delta > 0$

Remark 2.3. The optimal control is not unique even away from COMB’s equality set. In addition to $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ , the following controls are optimal throughout the indicated regions:

$$
{ \mathrm { I : ~ 1 0 0 1 1 , } } \qquad { \mathrm { I I : ~ 1 0 0 1 0 , } } \qquad { \mathrm { I I I : ~ 1 0 0 0 1 ~ a n d ~ 1 0 0 1 0 . } }\tag{2.21}
$$

Their directions are $\partial _ { z _ { 3 } }$ in Regions I and II, and $\partial _ { a _ { 1 } }$ and $\partial _ { a _ { 2 } }$ in Region III. This lists additional optimizers; further ties may occur on boundaries.

The rest of the paper is organized into two sections; in Section 3 we derive the formula and in Section 4 we verify the Hamiltonian inequalities that establish the formula indeed gives the solution of (2.1). Theorem 2.2 is an immediate consequence of this verification work. The verification is a tedious process, and uses a computer assisted proof to do much of the heavy lifting. This is described in detail in Section 4 and in the supplement.

## 3 Derivation of the five-expert formula

This section derives the formula of Theorem 2.1, following the steps outlined below. The proof that the result solves (2.1) is given in Section 4.

Before summarizing our approach, we make some observations about (1.1). Since the payof satisfies $\varphi ( x + c \mathbb { 1 } ) = \varphi ( x ) + c$ for all $c \in \mathbb { R }$ and is permutation invariant in the coordinates, the same is true for the viscosity solution u of (1.1), by uniqueness. Wherever u is diferentiable it follows that $\nabla u \cdot \mathbf { 1 } = 1$ and $u _ { x _ { i } } = u _ { x _ { j } }$ whenever $x _ { i } = x _ { j }$ . In particular, this leads to an origin boundary condition

$$
\nabla u ( 0 ) = \frac { 1 } { n } \mathbb { 1 }\tag{3.1}
$$

for the general n expert problem.

As a warmup and to illustrate the main ideas in this section, let us recall how to derive the solution formula for (1.1) in the case of $n = 2$ and $n = 3$ experts. For $n = 2$ , define $F ( x ) = u ( x , 0 )$ . Since $\mathbf { v } = ( 1 , 0 )$ is optimal for 2 experts (1.1) reduces to

$$
F - { \frac { 1 } { 2 } } F ^ { \prime \prime } ( x ) = x \quad { \mathrm { f o r ~ } } x \geq 0 .
$$

By (3.1) we have $F ^ { \prime } ( 0 ) = \frac { 1 } { 2 }$ , and since u has at most linear growth, we can ignore the exponentially growing solution $e ^ { \sqrt { 2 } x }$ . Thus

$$
F ( x ) = x + \frac { 1 } { 2 \sqrt { 2 } } e ^ { - \sqrt { 2 } x } .
$$

Since $\nabla u \cdot \mathbf { 1 } = 1$ we arrive at the $n = 2$ expert formula

$$
u ( x ) = x _ { 2 } + F ( x _ { 1 } - x _ { 2 } ) = x _ { 1 } + { \frac { 1 } { 2 { \sqrt { 2 } } } } e ^ { { \sqrt { 2 } } ( x _ { 2 } - x _ { 1 } ) } .\tag{3.2}
$$

A very similar argument for $n = 3$ with optimal strategy $\mathbf { v } = ( 1 , 0 , 0 )$ yields

$$
u ( x ) = x _ { 1 } + \frac { 1 } { 2 \sqrt { 2 } } e ^ { \sqrt { 2 } ( x _ { 2 } - x _ { 1 } ) } + \frac { 1 } { 6 \sqrt { 2 } } e ^ { \sqrt { 2 } ( 2 x _ { 3 } - x _ { 2 } - x _ { 1 } ) } .\tag{3.3}
$$

The key fact making both of these cases trivial is that there exists an optimal strategy, $\mathbf { v } = ( 1 , 0 )$ for $n = 2$ and $\mathbf { v } = ( 1 , 0 , 0 )$ for $n = 3$ , whose characteristic direction shoots of to infinity in one direction, allowing us to neglect the exponentially growing mode. This fails for the $n = 4$ and $n = 5$ expert problems. For $n = 4$ two strategies are optimal $\mathbf { v } = ( 1 , 0 , 1 , 0 )$ and $\mathbf { v } = ( 0 , 1 , 1 , 0 )$ [6], and neither has this property. Both characteristic directions intersect the boundary of the sector $\{ x _ { 1 } \geq x _ { 2 } \geq x _ { 3 } \geq x _ { 4 } \}$ in two places, requiring the solution of a two point boundary value problem, which greatly complicates the solution derivation. The case of $n = 5$ is more complicated, since the characteristic direction of the strategy $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ that we follow, which is optimal throughout the sector but not the only optimal strategy (see Remark 2.3), hits diferent faces of the sector boundary depending on the location, necessitating splitting the sector into regions and defining the solution diferently in each region.

Nevertheless, the overall idea from $n = 2$ and $n = 3$ experts—solving a one dimensional ODE in the optimal direction and resolving constants through boundary conditions—carries over to $n = 4$ and $n = 5$ at a high level, and is carried out in the rest of this section. In Section 3.1 we solve the equation (2.1) along the direction $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ , which was conjectured as optimal via numerics in [11]. The boundary conditions on the faces $x _ { i } = x _ { i + 1 }$ resulting from permutation of coordinates leads to a coupled system of first order PDEs, which are derived in Section 3.2. The key insight in this section is that we can (more or less) explicitly solve this system of PDEs; we show how to do this in Section 3.4. Then the final two subsections establish the 5 expert solution formula in the respective regions. Along the way, we give a rederivation of the $n = 4$ expert solution, avoiding the stochastic techniques used in [6] (see Section 3.5).

## 3.1 Solving the ODE along the $\mathbf { v } _ { * }$ characteristic

We work in the ordered sector with the scaled gaps and the correction F of (2.2). For a binary control, we define $b _ { \mathbf { v } } = ( v _ { 1 } - v _ { 2 } , v _ { 2 } - v _ { 3 } , v _ { 3 } - v _ { 4 } , v _ { 4 } - v _ { 5 } ) ^ { T }$ . Since $y _ { i } = k ( x _ { i } - x _ { i + 1 } )$ , a displacement of x in the direction v moves y in the direction $k b _ { \mathbf { v } } ,$ , so $\mathbf { v } ^ { T } \nabla ^ { 2 } u \mathbf { v } = k b _ { \mathbf { v } } ^ { T } \nabla _ { y } ^ { 2 } F b _ { \mathbf { v } }$ Writing $D _ { \mathbf { v } } ^ { 2 } u = \mathbf { v } ^ { T } \nabla ^ { 2 } u \mathbf { v }$ for the second derivative of u in the direction v, and using $k = 2 / k$ ，

$$
D _ { \mathbf { v } } ^ { 2 } u = \frac { 2 } { k } b _ { \mathbf { v } } ^ { T } \nabla _ { y } ^ { 2 } F b _ { \mathbf { v } } .\tag{3.4}
$$

We assume that $\mathbf { v } _ { * }$ attains the maximum in (2.1), which Section 4 verifies a posteriori. In the sector, where $\varphi = x _ { 1 }$ , the equation then reads $\begin{array} { r } { u - \frac { 1 } { 2 } D _ { \mathbf { v } _ { * } } ^ { 2 } u = x _ { 1 } } \end{array}$ , and with $u = x _ { 1 } + F / k$ and $b _ { \mathbf { v } _ { \ast } } = ( 1 , - 1 , 1 , 0 )$ this is the fixed equation

$$
( \partial _ { 1 } - \partial _ { 2 } + \partial _ { 3 } ) ^ { 2 } F = F .\tag{3.5}
$$

Since $u _ { x _ { 1 } } = 1 + F _ { y _ { 1 } } , u _ { x _ { 2 } } = - F _ { y _ { 1 } } + F _ { y _ { 2 } } , u _ { x _ { 3 } } = - F _ { y _ { 2 } } + F _ { y _ { 3 } } , u _ { x _ { 4 } } = - F _ { y _ { 3 } } + F _ { y _ { 4 } }$ , and $u _ { x 5 } = - F _ { y 4 }$ the face conditions $u _ { x _ { i } } = u _ { x _ { i + 1 } }$ when $x _ { i } = x _ { i + 1 } \ ( { \mathrm { o r } } \ y _ { i } = 0 )$ produce

$$
\begin{array} { r l r l r l } { { 2 } F _ { y 1 } - F _ { y 2 } = - 1 } & { ( y _ { 1 } = 0 ) , } & { - F _ { y 1 } + 2 F _ { y 2 } - F _ { y 3 } = 0 } & { ( y _ { 2 } = 0 ) , } \\ { - F _ { y 2 } + 2 F _ { y 3 } - F _ { y 4 } = 0 } & { ( y _ { 3 } = 0 ) , } & { - F _ { y 3 } + 2 F _ { y 4 } = 0 } & { ( y _ { 4 } = 0 ) . } \end{array}\tag{3.6}
$$

We now change variables so one direction aligns with $\mathbf { v } _ { * }$ . Let $A = y _ { 1 } + y _ { 2 } , B = y _ { 2 } + y _ { 3 } .$ $R = y _ { 4 }$ and $s = y _ { 2 }$ . In these new coordinates write ${ G ( A , B , R , s ) = F ( A - s , s , B - s , R ) }$ so that $( \partial _ { 1 } - \partial _ { 2 } + \partial _ { 3 } ) F = - G _ { s } .$ The sector becomes $0 \le s \le \operatorname* { m i n } ( A , B )$ , and (3.5) turns into $G _ { s s } = G$ . For fixed $( A , B , R )$ , this is a second-order linear equation in s on the interval $[ 0 , M ]$ , where $M = \operatorname* { m i n } ( A , B )$ . The solution to this two point boundary value problem is given by

$$
G ( A , B , R , s ) = \frac { \sinh ( M - s ) } { \sinh M } G ( A , B , R , 0 ) + \frac { \sinh s } { \sinh M } G ( A , B , R , M ) ( 0 \leq s \leq M ) .
$$

The endpoint $s = 0$ lies on the face $y _ { 2 } = 0$ . The endpoint $s = M$ lies on the face $y _ { 1 } = 0$ when $A \leq B$ , that is, when $y _ { 1 } \le y _ { 3 }$ , and on the face $y _ { 3 } = 0$ when $y _ { 3 } \le y _ { 1 }$ . To express the endpoint values, we introduce the four traces

$$
\begin{array} { r l } & { a ( X , Y , R ) = F ( X , 0 , X + Y , R ) , \quad \ell ( X , Y , R ) = F ( 0 , X , Y , R ) , } \\ & { b ( X , Y , R ) = F ( X + Y , 0 , X , R ) , \quad q ( X , Y , R ) = F ( Y , X , 0 , R ) . } \end{array}\tag{3.7}
$$

Here a and b parametrize the face $y _ { 2 } = 0$ on the two sides of the diagonal $y _ { 1 } = y _ { 3 }$ , while ℓ and q parametrize the faces $y _ { 1 } = 0$ and $y _ { 3 } = 0$ . For $y _ { 1 } \leq y _ { 3 } .$ set $X = y _ { 1 } + y _ { 2 }$ and $Y = y _ { 3 } - y _ { 1 }$ Then $Y \geq 0$ , so $M = \operatorname* { m i n } ( A , B ) = A$ , so $M = A = X$ and $B = X + Y$ , so the characteristic through y, by its construction, has endpoints $( X , 0 , X + Y , R )$ at $s = 0$ and $( 0 , X , Y , R )$ at $s = X$ , where F takes the values $a ( X , Y , R )$ and $\ell ( X , Y , R )$ . Since $M - s = y _ { 1 }$ and $s = y _ { 2 }$ the interpolation formula becomes

$$
F ( y ) = \frac { \sinh y _ { 1 } } { \sinh X } a ( X , Y , R ) + \frac { \sinh y _ { 2 } } { \sinh X } \ell ( X , Y , R ) .\tag{3.8}
$$

For $y _ { 3 } \le y _ { 1 }$ , set $X = y _ { 2 } + y _ { 3 }$ and $Y = y _ { 1 } - y _ { 3 }$ . Then $Y \geq 0$ , so $M = \operatorname* { m i n } ( A , B ) = B$ , so $M = B = X$ and $A = X + Y$ , so the endpoints are $( X + Y , 0 , X , R )$ and $( Y , X , 0 , R )$ , where F takes the values $b ( X , Y , R )$ and $q ( X , Y , R )$ . Now $M - s = y _ { 3 }$ , and we obtain

$$
F ( y ) = \frac { \sinh y _ { 3 } } { \sinh X } b ( X , Y , R ) + \frac { \sinh y _ { 2 } } { \sinh X } q ( X , Y , R ) .\tag{3.9}
$$

The quantities $a , b , \ell ,$ , and q satisfy coupled PDEs arising from the face conditions (3.6), which is the subject of the next section.

## 3.2 Deriving the trace systems

Substituting (3.8) into the first two face conditions gives

$$
a _ { X } = 2 \coth X a - 2 \operatorname { c s c h } X \ell , \qquad \ell _ { X } - 2 \ell _ { Y } = 2 \coth X \ell - 2 \operatorname { c s c h } X a - 1 .\tag{3.10}
$$

To see this, write (3.8) as $F = \sigma _ { 1 } a + \sigma _ { 2 } \ell$ with

$$
\sigma _ { 1 } = \frac { \sinh y _ { 1 } } { \sinh X } , \qquad \sigma _ { 2 } = \frac { \sinh y _ { 2 } } { \sinh X } ,
$$

where a and ℓ are evaluated at $( X , Y , R ) = ( y _ { 1 } + y _ { 2 } , y _ { 3 } - y _ { 1 } , y _ { 4 } )$ . Since $X _ { y _ { 1 } } = X _ { y _ { 2 } } = 1$ 2 $Y _ { y _ { 1 } } = - 1 , Y _ { y _ { 3 } } = 1$ , and $R _ { y _ { 4 } } = 1$ , the chain rule gives, for $g \in \{ a , \ell \}$

$$
\partial _ { 1 } g = g _ { X } - g _ { Y } , \qquad \partial _ { 2 } g = g _ { X } , \qquad \partial _ { 3 } g = g _ { Y } , \qquad \partial _ { 4 } g = g _ { R } .
$$

The weights depend only on $y _ { 1 }$ and $y _ { 2 }$ , through the numerator and through X, and

$$
\partial _ { 1 } \sigma _ { 1 } = \frac { \cosh y _ { 1 } } { \sinh X } - \coth X \sigma _ { 1 } , \partial _ { 2 } \sigma _ { 1 } = - \coth X \sigma _ { 1 } ,
$$

$$
\partial _ { 1 } \sigma _ { 2 } = - \coth X \sigma _ { 2 } , \qquad \partial _ { 2 } \sigma _ { 2 } = \frac { \cosh y _ { 2 } } { \sinh X } - \coth X \sigma _ { 2 } .
$$

Therefore

$$
\begin{array} { r l } & { F _ { y 1 } = ( \partial _ { 1 } \sigma _ { 1 } ) a + ( \partial _ { 1 } \sigma _ { 2 } ) \ell + \sigma _ { 1 } ( a _ { X } - a _ { Y } ) + \sigma _ { 2 } ( \ell _ { X } - \ell _ { Y } ) , } \\ & { F _ { y _ { 2 } } = ( \partial _ { 2 } \sigma _ { 1 } ) a + ( \partial _ { 2 } \sigma _ { 2 } ) \ell + \sigma _ { 1 } a _ { X } + \sigma _ { 2 } \ell _ { X } , } \\ & { F _ { y _ { 3 } } = \sigma _ { 1 } a _ { Y } + \sigma _ { 2 } \ell _ { Y } , \qquad F _ { y _ { 4 } } = \sigma _ { 1 } a _ { R } + \sigma _ { 2 } \ell _ { R } . } \end{array}
$$

On the face $y _ { 2 } = 0$ we have $X = y _ { 1 } , \textnormal { s o } \sigma _ { 1 } = 1 , \sigma _ { 2 } = 0 , \partial _ { 1 } \sigma _ { 1 } = \partial _ { 1 } \sigma _ { 2 } = 0 , \partial _ { 2 } \sigma _ { 1 } = - \coth X$ and $\partial _ { 2 } \sigma _ { 2 } = \operatorname { c s c h } X$ . Thus

$$
F _ { y _ { 1 } } = a _ { X } - a _ { Y } , \qquad F _ { y _ { 2 } } = a _ { X } - \coth X a + \operatorname { c s c h } X \ell , \qquad F _ { y _ { 3 } } = a _ { Y } , \qquad F _ { y _ { 4 } } = a _ { R } .
$$

The second face condition in (3.6) reads $2 F _ { y _ { 2 } } = F _ { y _ { 1 } } + F _ { y _ { 3 } } = a _ { X }$ , which is the first identity in (3.10). On the face $y _ { 1 } = 0$ we have $X = y _ { 2 } , \mathrm { ~ s o ~ } \sigma _ { 1 } = 0 , \sigma _ { 2 } = 1 , \partial _ { 2 } \sigma _ { 1 } = \partial _ { 2 } \sigma _ { 2 } = 0 , $ $\partial _ { 1 } \sigma _ { 1 } = \operatorname { c s c h } X$ , and $\partial _ { 1 } \sigma _ { 2 } = - \coth X$ . Thus

$$
F _ { y _ { 1 } } = \ell _ { X } - \ell _ { Y } - \coth X \ell + \operatorname { c s c h } X a , \qquad F _ { y _ { 2 } } = \ell _ { X } , \qquad F _ { y _ { 3 } } = \ell _ { Y } , \qquad F _ { y _ { 4 } } = \ell _ { R } .
$$

The first face condition reads $2 F _ { y _ { 1 } } - F _ { y _ { 2 } } = - 1$ , which is the second identity in (3.10).

The same calculation applies to $\left( 3 . 9 \right)$ , where $X = y _ { 2 } + y _ { 3 } , Y = y _ { 1 } - y _ { 3 }$ , the weights are sinh $y _ { 3 } /$ sinh X and sinh $y _ { 2 } /$ sinh X, and for $g \in \{ b , q \}$ the chain rule gives $\partial _ { 1 } g = g _ { Y }$ , $\partial _ { 2 } g = g _ { X } , \partial _ { 3 } g = g _ { X } - g _ { Y }$ , and $\partial _ { 4 } g = g _ { R }$ . On the face $y _ { 2 } = 0$ this yields

$$
F _ { y _ { 1 } } = b _ { Y } , \qquad F _ { y _ { 2 } } = b _ { X } - \coth X b + \operatorname { c s c h } X q , \qquad F _ { y _ { 3 } } = b _ { X } - b _ { Y } , \qquad F _ { y _ { 4 } } = b _ { R } , \qquad 
$$

and on the face $y _ { 3 } = 0$

$$
F _ { y _ { 1 } } = q _ { Y } , \qquad F _ { y _ { 2 } } = q _ { X } , \qquad F _ { y _ { 3 } } = q _ { X } - q _ { Y } - \coth X q + \operatorname { c s c h } X b , \qquad F _ { y _ { 4 } } = q _ { R } .
$$

The second face condition, $2 F _ { y _ { 2 } } = F _ { y _ { 1 } } + F _ { y _ { 3 } }$ , and the third face condition, $2 F _ { y _ { 3 } } = F _ { y _ { 2 } } + F _ { y _ { 4 } }$ give

$$
b _ { X } = 2 \coth X b - 2 \operatorname { c s c h } X q , \qquad q _ { X } - 2 q _ { Y } - q _ { R } = 2 \coth X q - 2 \operatorname { c s c h } X b .\tag{3.11}
$$

The fourth face condition, $F _ { y _ { 3 } } = 2 F _ { y _ { 4 } }$ , becomes

$$
a _ { Y } = 2 a _ { R } , \quad \ell _ { Y } = 2 \ell _ { R } , \quad b _ { X } - b _ { Y } = 2 b _ { R } , \quad q _ { X } = 3 q _ { R } \qquad ( R = 0 ) .\tag{3.12}
$$

The first three follow directly from the face values of $F _ { y 3 }$ and $F _ { y _ { 4 } }$ displayed above. At the q trace the third face says $2 F _ { y 3 } = q _ { X } + q _ { R }$ , while the bottom face says $F _ { y _ { 3 } } = 2 q _ { R }$

Our main task for much of this section will be to solve the coupled system of PDEs (3.10). A first step in this direction is to note that both equations can be integrated. Indeed, we note that

$$
{ \frac { d } { d X } } \left( { \frac { a } { \sinh ^ { 2 } X } } \right) = { \frac { a _ { X } - 2 \coth X a } { \sinh ^ { 2 } X } } = - { \frac { 2 \ell } { \sinh ^ { 3 } X } }
$$

and then integrate, using boundedness at infinity, to obtain $a = \tau \ell$ , where the integral operator I is defined by

$$
( \mathbb { Z } f ) ( X ) = 2 \sinh ^ { 2 } X \int _ { X } ^ { \infty } { \frac { f ( t ) } { \sinh ^ { 3 } t } } d t ,\tag{3.13}
$$

for $X > 0$ and by continuity for $X = 0$ . A similar argument yields $b = \mathcal { T } q$ . We also define, for later usage, the operators

$$
\ K = - \coth X + \operatorname { c s c h } X \bar { \ . }\tag{3.14}
$$

and

$$
\mathcal { L } _ { m } = \frac { 1 } { m } \partial _ { X } + \mathcal { K }\tag{3.15}
$$

for $m = 2 , 3$ . The remaining trace equations in (3.10) and (3.11) can now be written as

$$
\begin{array} { r } { \ell _ { Y } = \mathcal { L } _ { 2 } \ell + \frac { 1 } { 2 } , \qquad q _ { Y } + \frac { 1 } { 2 } q _ { R } = \mathcal { L } _ { 2 } q . } \end{array}\tag{3.16}
$$

## 3.3 The closed slice and its single scalar quadrature

Before solving the trace equations in general, we first note that we can solve them on the slice $y _ { 1 } = y _ { 3 } , y _ { 4 } = 0$ , since the equations close on this set. We define $\Psi ( a , b ) = { \cal F } ( a , b , a , 0 )$ Equations (3.5)–(3.6) imply

$$
\begin{array} { r } { ( \partial _ { a } - \partial _ { b } ) ^ { 2 } \Psi = \Psi , \qquad \Psi _ { a } - \frac { 7 } 6 \Psi _ { b } = - \frac { 1 } { 2 } { \bf \Gamma } ( a = 0 ) , \qquad \Psi _ { b } - \frac { 1 } { 2 } \Psi _ { a } = 0 { \bf \Gamma } ( b = 0 ) . } \end{array}\tag{3.17}
$$

The direction $( 1 , - 1 , 1 , 0 )$ of (3.5) is tangent to the slice and corresponds to $\partial _ { a } - \partial _ { b }$ there, since $\Psi _ { a } = F _ { y _ { 1 } } + F _ { y _ { 3 } }$ and $\Psi _ { b } = F _ { y _ { 2 } } ;$ this gives the first equation. At $a = 0$ the faces $y _ { 1 } = 0 , y _ { 3 } = 0$ , and $y _ { 4 } = 0$ all apply, and (3.6) gives $F _ { y _ { 1 } } = ( F _ { y _ { 2 } } - 1 ) / 2 , F _ { y _ { 4 } } = F _ { y _ { 3 } } / 2 ,$ and hence $F _ { y _ { 3 } } = 2 F _ { y _ { 2 } } / 3 .$ , so $\Psi _ { a } = 7 \Psi _ { b } / 6 - 1 / 2$ . At $b = 0$ the second condition in (3.6) gives $\begin{array} { r } { \Psi _ { b } = \dot { F } _ { y _ { 2 } } = \frac { 1 } { 2 } ( F _ { y _ { 1 } } + F _ { y _ { 3 } } ) = \frac { 1 } { 2 } \Psi _ { a } } \end{array}$ . Note that $X = a + b$ is the variable of the traces (3.7): the expressions $y _ { 1 } + y _ { 2 }$ and $y _ { 2 } + y _ { 3 }$ coincide on the slice, and the endpoint values $\Psi ( X , 0 ) = F ( X , 0 , X , 0 )$ and $\Psi ( 0 , X ) = { \cal F } ( 0 , X , 0 , 0 )$ are the traces $a = b$ and $\ell = q$ at $Y = R = 0$

Let $h ( X ) = \Psi ( X , 0 )$ . The direction $( 1 , - 1 )$ leaves X invariant. Along the characteristic through $( a , b )$ , parametrized by $s \ = \ b$ with X fixed, the first equation in (3.17) reads $\partial _ { s } ^ { 2 } \Psi ( X - s , s ) = \Psi ( X - s , s )$ , so

$$
\Psi ( X - s , s ) = h ( X ) \cosh s + c ( X ) \sinh s \qquad ( 0 \leq s \leq X )
$$

for some $c ,$ and diferentiating in s at $s \ = \ 0$ gives $c = - \Psi _ { a } ( X , 0 ) + \Psi _ { b } ( X , 0 )$ . Here $\Psi _ { a } ( X , 0 ) = h ^ { \prime } ( X )$ , since h is the restriction of Ψ to the face $b = 0$ , and the boundary condition at $b = 0$ gives $\begin{array} { r } { \Psi _ { b } ( X , 0 ) = \frac { 1 } { 2 } h ^ { \prime } ( X ) } \end{array}$ . Thus $c = - \textstyle { \frac { 1 } { 2 } } h ^ { \prime }$ , and

$$
\begin{array} { r } { \Psi ( a , b ) = h ( X ) \cosh b - \frac { 1 } { 2 } h ^ { \prime } ( X ) \sinh b . } \end{array}\tag{3.18}
$$

This is the hyperbolic interpolation (3.8) on the slice, after the boundary condition at $b = 0$ has been used to express the endpoint value $\Psi ( 0 , X )$ through $h$ and $h ^ { \prime } .$ . The boundary condition at $a = 0$ now determines h. Since (3.18) depends on a only through X,

$$
\Psi _ { a } = h ^ { \prime } \cosh b - { \textstyle \frac { 1 } { 2 } } h ^ { \prime \prime } \sinh b , \qquad \Psi _ { b } = \Psi _ { a } + h \sinh b - { \textstyle \frac { 1 } { 2 } } h ^ { \prime } \cosh b .
$$

Substituting into $\begin{array} { r } { \Psi _ { a } - \frac { 7 } { 6 } \Psi _ { b } = - \frac { 1 } { 2 } } \end{array}$ at $a = 0$ , where $b = X$ , gives

$$
\begin{array} { r } { \frac { 1 } { 1 2 } h ^ { \prime \prime } \sinh X + \frac { 5 } { 1 2 } h ^ { \prime } \cosh X - \frac { 7 } { 6 } h \sinh X = - \frac { 1 } { 2 } , } \end{array}
$$

and multiplying by 12/ sinh X gives

$$
h ^ { \prime \prime } + 5 \coth X h ^ { \prime } - 1 4 h = - 6 \operatorname { c s c h } X .\tag{3.19}
$$

It is convenient to obtain the endpoint value $e ( X ) = \Psi ( 0 , X )$ directly from this equation; we show that it is the function (2.8) of Section 2. By (3.18),

$$
\begin{array} { r } { e ( X ) = \Psi ( 0 , X ) = h ( X ) \cosh X - \frac 1 2 h ^ { \prime } ( X ) \sinh X . } \end{array}\tag{3.20}
$$

Diferentiating and eliminating $h ^ { \prime \prime }$ by (3.19) gives

$$
e ^ { \prime } = 3 h ^ { \prime } \cosh X - 6 h \sinh X + 3 ,\tag{3.21}
$$

and diferentiating once more, $e ^ { \prime \prime } = 3 h ^ { \prime \prime }$ cosh $X - 3 h ^ { \prime }$ sinh $X - 6 h$ cosh X. Substituting $h ^ { \prime \prime }$ from (3.19) into $e ^ { \prime \prime } + 5$ coth $X e ^ { \prime } - 6 e$ , the terms in $h ^ { \prime }$ and in h cancel and the constants leave

$$
e ^ { \prime \prime } + 5 \coth X e ^ { \prime } - 6 e = - 3 \coth X .\tag{3.22}
$$

Since (3.20) rearranges to $h ^ { \prime } = 2$ coth X $h - 2$ csch X e and h is bounded, the argument after (3.13) gives $h = \mathcal { T } e$ . These identities also give the compatibility relation

$$
e ^ { \prime } = 6 \mathcal { K } e + 3 .\tag{3.23}
$$

Indeed, (3.21) gives $e ^ { \prime } = 3 h ^ { \prime }$ cosh $X - 6 h$ sinh $X + 3 ,$ , while $\mathit { K e } = h ^ { \prime } \cosh X / 2 - h$ sinh X by $h = \mathcal { T } e$ and (3.20).

Conversely, let e be a solution of (3.22) with e and $e ^ { \prime }$ bounded, and put $\textit { h } = \mathcal { I } e$ Then (3.20) holds, since Ie solves $h ^ { \prime } = 2$ coth $X h - 2 \mathrm { c s c h } X e$ . Moreover, the function $\begin{array} { r } { \widetilde { h } = \frac { 1 } { 6 } } \end{array}$ sinh X $e ^ { \prime } +$ cosh $\begin{array} { l } { \displaystyle { X e - \frac { 1 } { 2 } } } \end{array}$ sinh X satisfies

$$
\begin{array} { r } { \widetilde { h } ^ { \prime } - 2 \coth X \widetilde { h } + 2 \operatorname { c s c h } X e = \frac { 1 } { 6 } \sinh X \left( e ^ { \prime \prime } + 5 \coth X e ^ { \prime } - 6 e + 3 \coth X \right) = 0 , } \end{array}
$$

so $\widetilde { h } - \mathcal { I } e = C \sinh ^ { 2 } X$ for a constant C. Since $\widetilde { h } - \mathcal { T } e = \mathcal { O } ( e ^ { X } ) , C = 0$ , and $\mathcal { I } e = \widetilde { h }$ is (3.23). Diferentiating $h ^ { \prime } = 2$ coth $X h - 2$ csch $X e$ and eliminating $e ^ { \prime }$ and e by (3.23) and (3.20) gives (3.19). Thus it sufices to solve (3.22).

To reduce (3.22) to a first-order equation, we set $\eta = e - e ^ { \prime \prime }$ . Diferentiating (3.22) gives $e ^ { \prime \prime \prime } = - 5$ coth X $e ^ { \prime \prime } + ( 5 \cosh ^ { 2 } X + 6 ) e ^ { \prime } + 3 \cosh ^ { 2 } X$ , so that $\eta ^ { \prime } = e ^ { \prime } - e ^ { \prime \prime \prime }$ and 6 coth $X \eta$ combine to

$$
\eta ^ { \prime } + 6 \coth X \eta = - \coth X \left( e ^ { \prime \prime } + 5 \coth X e ^ { \prime } - 6 e \right) - 3 \operatorname { c s c h } ^ { 2 } X = 3 \coth ^ { 2 } X - 3 \operatorname { c s c h } ^ { 2 } X = 3 .
$$

Thus

$$
\eta ^ { \prime } + 6 \coth X \eta = 3 , \qquad \eta ( X ) = { \frac { 3 } { \sinh ^ { 6 } X } } \int _ { 0 } ^ { X } \sinh ^ { 6 } t d t .\tag{3.24}
$$

To integrate, multiply by $\sinh ^ { 6 } X$ , whose logarithmic derivative is 6 coth $X$ , and integrate from zero:

$$
( \sinh ^ { 6 } X \eta ) ^ { \prime } = 3 \sinh ^ { 6 } X , \qquad \eta ( X ) = { \frac { 3 } { \sinh ^ { 6 } X } } \int _ { 0 } ^ { X } \sinh ^ { 6 } t d t + { \frac { C } { \sinh ^ { 6 } X } } .
$$

Since the trace $e$ is twice diferentiable at $X = 0 , \eta = e - e ^ { \prime \prime }$ is bounded there, whereas $C / \sinh ^ { 6 } X$ is not unless $C = 0$ . This gives the displayed formula, with $\eta ( X ) \sim 3 X / 7$ at zero and $\eta ( X ) \to { \frac { 1 } { 2 } }$ at infinity. Since $e ^ { \prime \prime } = e - \eta$ , equation (3.22) now gives $e ^ { \prime } =$ tanh $X ( e + \eta / 5 ) -$ $3 / 5$ . With the primitive I of $\left( 2 . 7 \right)$ , integrating $( \sinh ^ { 5 } X \cosh X ) ^ { \prime } = 5 \sinh ^ { 4 } X + 6 \sinh ^ { 6 } X$ from zero and using $\begin{array} { r } { I = \int _ { 0 } ^ { X } \sinh ^ { 4 } t d t + \int _ { 0 } ^ { X } \sinh ^ { 6 } t d t } \end{array}$ give

$$
\int _ { 0 } ^ { X } \sinh ^ { 6 } t d t = \sinh ^ { 5 } X \cosh X - 5 I ( X ) , \qquad \eta = 3 \coth X - { \frac { 1 5 I ( X ) } { \sinh ^ { 6 } X } } .
$$

Substituting this into $e ^ { \prime } =$ tanh $X ( e + \eta / 5 ) - 3 / 5$ , the terms $\frac { 3 } { 5 }$ cancel and the trace equation becomes the first-order equation (2.9). With the integrating factor sech X it reads $( \operatorname { s e c h } X e ) ^ { \prime } = - 3 I ( X ) / ( \cosh ^ { 2 } X \sinh ^ { 5 } X )$ , and since sech $X e  0$ at infinity for bounded e, integration from X to infinity yields the quadrature (2.8). The integrand there is regular at zero since $I ( t ) = t ^ { 5 } / 5 + \mathcal { O } ( t ^ { 7 } )$ , and decays exponentially at infinity. Thus $e$ is positive and bounded. Equation (2.9) gives

$$
\begin{array} { r } { e ^ { \prime } ( 0 ) = - \frac { 3 } { 5 } , \qquad e ^ { \prime \prime } ( 0 ) = e ( 0 ) , \qquad e ( X ) \longrightarrow \frac { 1 } { 2 } \quad ( X  \infty ) . } \end{array}\tag{3.25}
$$

Indeed, since sinh $\begin{array} { r } { \mathrm {  ~ \omega ~ } ^ { 4 } t \cosh ^ { 2 } t = t ^ { 4 } + \frac { 5 } { 2 } t ^ { 6 } + \mathcal { O } ( t ^ { 8 } ) } \end{array}$ , we have $\begin{array} { r } { I ( X ) = \frac { 1 } { 5 } X ^ { 5 } + \frac { 5 } { 2 1 } X ^ { 7 } + \mathcal { O } ( X ^ { 9 } ) } \end{array}$ and cosh X sinh $\textstyle { ^ { 5 } X = X ^ { 5 } + \frac { 4 } { 3 } X ^ { 7 } + { \mathcal { O } } ( X ^ { 9 } ) }$ , so the forcing term in (2.9) is

$$
\frac { 3 I ( X ) } { \cosh X \sinh ^ { 5 } X } = \textstyle { \frac { 3 } { 5 } } - \frac { 3 } { 3 5 } X ^ { 2 } + { \mathcal O } ( X ^ { 4 } ) .
$$

As tanh $X e  0$ at zero, (2.9) gives $\begin{array} { r } { e ^ { \prime } ( 0 ) = - \frac { 3 } { 5 } . } \end{array}$ The identity $e ^ { \prime \prime } ( 0 ) = e ( 0 ) \ \mathrm { i s } \ \eta ( 0 ) = 0$ which is immediate from $( 3 . 2 4 )$ ; it also follows by diferentiating (2.9), since the forcing term is even. At infinity, $I ( t ) \sim e ^ { 6 t } / 3 8 4$ and $\cosh ^ { 2 } t \mathrm { s i n h } ^ { 5 } t \sim e ^ { 7 t } / \bar { 1 2 8 }$ , so the integrand in (2.8) is asymptotic to $e ^ { - t } / 3$ , and $e ( X ) \sim 3 \cdot { \textstyle \frac { 1 } { 2 } } e ^ { X } \cdot { \textstyle \frac { 1 } { 3 } } e ^ { - X } = { \textstyle \frac { 1 } { 2 } }$ . The forcing term in (2.9) has a removable singularity at the origin and extends analytically there. Thus the trace is analytic at the origin. At infinity, substituting a series in $e ^ { - X }$ into (2.9) gives

$$
\begin{array} { r } { e ( X ) = \frac 1 2 + \frac 1 2 e ^ { - 2 X } - \frac 2 5 e ^ { - 4 X } + \mathcal { O } ( X e ^ { - 6 X } ) . } \end{array}\tag{3.26}
$$

Indeed, the forcing term $3 I / ( \cosh X \sinh ^ { 5 } X )$ is a rational function of $e ^ { - 2 X }$ apart from the term $X / 1 6$ of I, which first contributes at order $X e ^ { - 6 X }$ , and the coeficients of $e ^ { - 2 X }$ and $e ^ { - 4 X }$ are determined by matching; the remainder satisfies a linear equation with forcing $\mathcal { O } ( X e ^ { - 6 X } )$ , whose bounded solution is $\mathcal { O } ( X e ^ { - 6 X } )$ . Finally we evaluate $e ( 0 )$ in closed form.

Lemma 3.1. $e ( 0 ) = { 4 5 \pi ^ { 2 } } / { 5 1 2 }$ hence $u ( 0 ) = e ( 0 ) / \sqrt { 2 } = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ , the constant term in (2.18).

Proof. By $( 2 . 8 ) , e ( 0 ) = 3 J$ with $\begin{array} { r } { J = \int _ { 0 } ^ { \infty } I ( t ) w ( t ) } \end{array}$ dt and $w = \mathrm { s e c h } ^ { 2 } t \cosh ^ { 5 } t$ . Since $I ( t ) =$ $\begin{array} { r } { \int _ { 0 } ^ { t } \sinh ^ { 4 } s \cosh ^ { 2 } s d s } \end{array}$ by (2.7) and all integrands are positive, Fubini’s theorem gives

$$
J = \int _ { 0 } ^ { \infty } \sinh ^ { 4 } s \cosh ^ { 2 } s V ( s ) d s , \qquad V ( s ) = \int _ { s } ^ { \infty } w ( t ) d t .
$$

Writing $1 = \cosh ^ { 2 } t - \sinh ^ { 2 } t$ three times, $w = \cosh ^ { 5 } t - \operatorname { c s c h } ^ { 3 } t +$ csch $t - \sinh t \ \mathrm { s e c h } ^ { 2 } t .$ , and the standard antiderivatives, with $\lambda ^ { \prime } = - \cosh$ , give

$$
\begin{array} { r } { V ( s ) = \frac 1 4 \operatorname { c s c h } ^ { 3 } s \coth s - \frac 7 8 \operatorname { c s c h } s \coth s + \frac { 1 5 } 8 \lambda ( s ) - \operatorname { s e c h } s , } \end{array}
$$

each term vanishing at infinity. Hence

$$
\begin{array} { r } { \sinh ^ { 4 } s \cosh ^ { 2 } s V ( s ) = \frac { 1 } { 4 } \cosh ^ { 3 } s - \frac { 7 } { 8 } \sinh ^ { 2 } s \cosh ^ { 3 } s - \sinh ^ { 4 } s \cosh s + \frac { 1 5 } { 8 } \lambda ( s ) \sinh ^ { 4 } s \cosh ^ { 2 } s . } \end{array}
$$

We integrate over $[ 0 , T ]$ and let $T \to \infty$ . The first three terms are elementary. In the fourth, $\sinh ^ { 4 } s \cosh ^ { 2 } s = I ^ { \prime } ( s )$ and $\lambda ^ { \prime } = - \cosh$ , so integrating by parts, with $\lambda ( s ) I ( s ) \to 0 { \mathrm { ~ a t ~ } } s = 0$

$$
\int _ { 0 } ^ { T } \lambda ( s ) I ^ { \prime } ( s ) d s = \lambda ( T ) I ( T ) + \int _ { 0 } ^ { T } \cosh s I ( s ) d s ,
$$

and since sinh 2ks/ sinh $\begin{array} { r } { s = 2 \sum _ { j = 1 } ^ { k } \cosh ( 2 j - 1 ) s , } \end{array}$

$$
\operatorname { c s c h } s I ( s ) = { \textstyle { \frac { 1 } { 9 6 } } } \cosh 5 s - { \textstyle { \frac { 1 } { 4 8 } } } \cosh 3 s - { \textstyle { \frac { 5 } { 9 6 } } } \cosh s + { \textstyle { \frac { s } { 1 6 \sinh s } } } .
$$

Collecting the elementary terms,

$$
\begin{array} { r } { \displaystyle \int _ { 0 } ^ { T } \sinh ^ { 4 } s \cosh ^ { 2 } s V ( s ) d s = E ( T ) + \frac { 1 5 } { 1 2 8 } \int _ { 0 } ^ { T } \frac { s d s } { \sinh s } , \ ~ } \\ { E ( T ) = \frac { 1 } { 4 } \bigl ( \sinh T + \frac { 1 } { 3 } \sinh ^ { 3 } T \bigr ) - \frac { 7 } { 8 } \bigl ( \frac { 1 } { 3 } \sinh ^ { 3 } T + \frac { 1 } { 5 } \sinh ^ { 5 } T \bigr ) - \frac { 1 } { 5 } \sinh ^ { 5 } T } \\ { + \frac { 1 5 } { 8 } \Bigl ( \lambda ( T ) I ( T ) + \frac { 1 } { 4 8 0 } \sinh 5 T - \frac { 1 } { 1 4 4 } \sinh 3 T - \frac { 5 } { 9 6 } \sinh T \Bigr ) . \ ~ } \end{array}
$$

Expanding in $e ^ { - T } .$ , with $\begin{array} { r } { \lambda ( T ) = 2 \sum _ { n \mathrm { ~ o d d } } e ^ { - n T } / n , } \end{array}$ the growing exponentials cancel and $\begin{array} { r } { E ( \hat { T } ) = \big ( \frac { \hat { 1 5 } } { 6 4 } T - \frac { 2 3 } { 4 4 8 } \big ) e ^ { - T } + \mathcal { \dot { O } } ( T e ^ { - 3 T } ) \xrightarrow { \dots } 0 } \end{array}$ . Therefore

$$
J = { \frac { 1 5 } { 1 2 8 } } \int _ { 0 } ^ { \infty } { \frac { s d s } { \sinh s } } = { \frac { 1 5 } { 1 2 8 } } \cdot 2 \sum _ { k \geq 0 } { \frac { 1 } { ( 2 k + 1 ) ^ { 2 } } } = { \frac { 1 5 } { 1 2 8 } } \cdot { \frac { \pi ^ { 2 } } { 4 } } = { \frac { 1 5 \pi ^ { 2 } } { 5 1 2 } } ,
$$

and $e ( 0 ) = 3 J = 4 5 \pi ^ { 2 } / 5 1 2$

## 3.4 Truncated modes and the solution operators

We now proceed to solving the coupled trace equations (3.10) and (3.11). We will proceed in generality for the moment. Let $E ( X , Y )$ denote a lower trace, ℓ or $q ,$ , and let $A ( X , Y ) = \mathcal { T } E$ be the corresponding upper trace, a or b. For $m = 2 , 3$ consider

$$
A _ { X } = 2 \coth X A - 2 \operatorname { c s c h } X E , \qquad E _ { X } - m E _ { Y } = m \coth X E - m \operatorname { c s c h } X A .\tag{3.27}
$$

With $m = 2$ this is the common homogeneous form of (3.10) and (3.11): the constant −1 in (3.10) is removed by subtracting a stationary solution, and the term $q _ { R }$ in (3.11) is absorbed by a characteristic variable, as done in Section 3.5. On the bottom face $R = 0$ of Region III, where $q _ { X } = 3 q _ { R }$ by (3.12), equation (3.11) becomes the case $m = 3$ . Since $A = \mathcal { I } E$ solves the first equation, the second reads $\begin{array} { r } { E _ { Y } = \frac { 1 } { m } E _ { X } + \mathcal { K } E = \mathcal { L } _ { m } E } \end{array}$ with ${ \mathcal { L } } _ { m }$ from (3.15). The problem is therefore: given $E ( \cdot , 0 ) = f$ , find $E ( \cdot , Y )$ . We write $S _ { m } ( Y ) f = E ( \cdot , Y )$ for the solution operator, or propagator, of this Cauchy problem, and $\mathcal { T } _ { m } ( Y ) f = \mathcal { T } S _ { m } ( Y ) f = A ( \cdot , Y )$ for its upper-trace companion. We show these exist and give formulas for them below.

As a first step, it turns out we can easily solve the Cauchy problem with initial data given by $f = \Phi _ { t } { \mathrm { - } } \mathrm { t h e }$ mode defined in (2.13).

Proposition 3.2. Let $t > 0$ . The operators (3.13) act on a mode by

$$
{ \mathcal { T } } \Phi _ { t } = \Phi _ { t } ^ { 2 } , \qquad K \Phi _ { t } = - \coth t \Phi _ { t } .\tag{3.28}
$$

Moreover, for $m = 2 , 3$ the Cauchy problem (3.27) with data $E ( \cdot , 0 ) = \Phi _ { t }$ is solved by products of truncated modes: with $L = X + Y / m$ and $d = Y / m$ , as functions of X,

$$
S _ { m } ( Y ) \Phi _ { t } = \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m } , \qquad \mathcal { T } _ { m } ( Y ) \Phi _ { t } = \Phi _ { t } ( L ) ^ { 2 } \Phi _ { t } ( d ) ^ { m - 1 } .\tag{3.29}
$$

Both vanish for $t \leq L ,$ , and at $Y = 0$ they reduce to $\Phi _ { t }$ and $\Phi _ { t } ^ { 2 } = \mathcal { I } \Phi _ { t }$

Proof. For $X \geq t$ both sides of both identities vanish, since $\Phi _ { t } ( s ) = 0$ for $s \geq t$ . For $0 < s < t ,$ diferentiating the quotient $\Phi _ { t } ( s ) ^ { 2 } / ( 2 \sinh ^ { 2 } s ) = \sinh ^ { 2 } ( t - s ) / ( 2 \sinh ^ { 2 } t \sinh ^ { 2 } s )$ and using the addition formula $- \cosh ( t - s )$ sinh $s - \sinh ( t - s )$ cosh s = − sinh t gives

$$
\frac { d } { d s } \frac { \Phi _ { t } ( s ) ^ { 2 } } { 2 \sinh ^ { 2 } s } = - \frac { \Phi _ { t } ( s ) } { \sinh ^ { 3 } s } .
$$

Since $\Phi _ { t } ( t ) = 0$ , for $0 < X < t$ this yields

$$
\int _ { X } ^ { \infty } \frac { \Phi _ { t } ( s ) } { \sinh ^ { 3 } s } d s = \int _ { X } ^ { t } \frac { \Phi _ { t } ( s ) } { \sinh ^ { 3 } s } d s = \frac { \Phi _ { t } ( X ) ^ { 2 } } { 2 \sinh ^ { 2 } X } ,
$$

and multiplying by $2 \sinh ^ { 2 }$ X gives the first identity. Equivalently, $\Phi _ { t } ^ { 2 }$ is the bounded solution of the first trace equation $g ^ { \prime } = 2$ coth $X g - 2$ csch X $\Phi _ { t }$ . For the second identity,

$$
K \Phi _ { t } = - \coth X \Phi _ { t } + \operatorname { c s c h } X \Phi _ { t } ^ { 2 } = \Phi _ { t } \frac { \sinh ( t - X ) - \cosh X \sinh t } { \sinh X \sinh t } = - \coth t \Phi _ { t } ,
$$

since sinh $( t - X ) = \sinh t$ cosh X − cosh t sinh X reduces the numerator to − cosh t sinh X. We now verify that $E = \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m }$ and $A = \Phi _ { t } ( L ) ^ { 2 } \Phi _ { t } ( d ) ^ { m - 1 }$ satisfy (3.27). Put $N _ { t } ( a ) =$

coth t cosh a − sinh $a = \cosh ( t - a ) /$ sinh t, so that $\Phi _ { t } ^ { \prime } ( a ) = - N _ { t } ( a )$ for $0 < a < t$ . Writing $t - d = ( t - L ) + X$ and $t - L = ( t - d ) - X$ in the definition of $\Phi _ { t } .$ , the addition formula for sinh gives

$$
\Phi _ { t } ( d ) = \cosh X \Phi _ { t } ( L ) + \sinh X N _ { t } ( L ) , \qquad \Phi _ { t } ( L ) = \cosh X \Phi _ { t } ( d ) - \sinh X N _ { t } ( d ) .
$$

Let $X > 0$ and $L < t$ . In the variables $( L , d )$ we have $X = L - d , \partial _ { X } = \partial _ { L }$ and $\partial _ { Y } =$ $( \partial _ { L } + \partial _ { d } ) / m$ , so $\partial _ { X } - m \partial _ { Y } = - \partial _ { d }$ . The two sides of the first equation of (3.27) are

$$
\begin{array} { c } { { \displaystyle \partial _ { L } A = - 2 \Phi _ { t } ( L ) N _ { t } ( L ) \Phi _ { t } ( d ) ^ { m - 1 } , } } \\ { { \displaystyle 2 \coth X A - 2 \operatorname { c s c h } X E = \frac { 2 \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m - 1 } } { \sinh X } \bigl [ \cosh X \Phi _ { t } ( L ) - \Phi _ { t } ( d ) \bigr ] , } } \end{array}
$$

and the bracket equals − sinh X $N _ { t } ( L )$ by the first addition formula. The two sides of the second equation are

$$
\begin{array} { c } { { - \partial _ { d } E = m \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m - 1 } N _ { t } ( d ) , } } \\ { { { } } } \\ { { m \coth X E - m \operatorname { c s c h } X A = \displaystyle \frac { m \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m - 1 } } { \sinh X } \bigl [ \cosh X \Phi _ { t } ( d ) - \Phi _ { t } ( L ) \bigr ] , } } \end{array}
$$

and the bracket equals sinh $X N _ { t } ( d )$ by the second addition formula. For $L > t$ both E and A vanish identically, and since $\Phi _ { t } ( t ) = 0$ all four expressions above are continuous across $L = t$ . The function E itself is only Lipschitz across the front $L = t \colon$ its derivative $\partial _ { L } E =$ $\Phi _ { t } ^ { \prime } ( L ) \Phi _ { t } ( d ) ^ { m }$ jumps from $- \Phi _ { t } ( d ) ^ { m } /$ sinh t to 0 there, while the combination $E _ { X } - m E _ { Y } =$ $- \partial _ { d } E$ is continuous. Thus E solves (3.27) classically on either side of the characteristic $L = t$ and as a Lipschitz solution across it; the superpositions of modes with smooth data used below are classical solutions. At $Y = 0$ we have $L = X$ and $d = 0$ , and $\Phi _ { t } ( 0 ) = 1$ gives $E = \Phi _ { t }$ and $A = \Phi _ { t } ^ { 2 } = { \mathcal { T } } \Phi _ { t }$ . Finally A is bounded, and the homogeneous solutions of the first equation are the multiples of $\sinh ^ { 2 } X$ , so the first equation gives $A = \mathcal { I } E$ for every ${ \cal Y } ,$ as required of the upper trace. □

Now, we recall that the truncated mode $\Phi _ { t }$ of (2.13) is the Green’s function of $\partial _ { X } ^ { 2 } - 1$ with pole at t that vanishes for $X > t$ , normalized by $\Phi _ { t } ( 0 ) = 1$ . By the Green’s representation formula we have

$$
f ( X ) = \int _ { X } ^ { \infty } \sinh ( t - X ) \left( f ^ { \prime \prime } - f \right) ( t ) d t = \int _ { 0 } ^ { \infty } \sinh t \left( f ^ { \prime \prime } - f \right) ( t ) \Phi _ { t } ( X ) d t ,\tag{3.30}
$$

for any smooth compactly supported $f .$ Applying (3.29) under the integral in (3.30) yields

$$
\displaystyle { \mathcal { S } _ { m } ( Y ) f = \int _ { L } ^ { \infty } \sinh t \left( f ^ { \prime \prime } - f \right) ( t ) \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m } d t } .
$$

The integral starts at L because $\Phi _ { t } ( L ) = 0$ for $t < L$ . Integrating by parts twice moves the derivatives onto the kernel, and the resulting expression involves f alone, so it extends the solution operators to bounded data. The next lemma records the resulting Green’s function of the Cauchy problem for a general product of modes; the same Green’s function produces the Region III formula in Section 3.6.

Lemma 3.3. Let $L > 0$ , let $a _ { 1 } , \dots , a _ { N } \in [ 0 , L ]$ with at least one $a _ { i }$ equal to $L ,$ , and set $\begin{array} { r } { G ( t ) = \sinh t \prod _ { i = 1 } ^ { N } \Phi _ { t } ( a _ { i } ) } \end{array}$

(i) For $t > L$

$$
G ^ { \prime \prime } - G = \frac { 2 } { \sinh ^ { 3 } t } \sum _ { i < j } \sinh a _ { i } \sinh a _ { j } \prod _ { h \not \in \{ i , j \} } \Phi _ { t } ( a _ { h } ) ,\tag{3.31}
$$

which is nonnegative and at most $N ^ { 2 } \sinh ^ { 2 } L / \sinh ^ { 3 } { \it \Psi }$ t. Moreover $\begin{array} { r } { G ^ { \prime } ( L ) = \prod _ { h \neq i } \Phi _ { L } ( a _ { h } ) } \end{array}$ if $a _ { i }$ is the only entry equal to L, and $G ^ { \prime } ( L ) = 0$ if two or more entries equal $L$

(ii) For every smooth f with compact support in $[ 0 , \infty )$

$$
( 3 . 3 2 ) ~ { \mathcal { P } } _ { a _ { 1 } , \ldots , a _ { N } } f = \int _ { L } ^ { \infty } G ( t ) ( f ^ { \prime \prime } - f ) ( t ) d t = G ^ { \prime } ( L ) f ( L ) + \int _ { L } ^ { \infty } ( G ^ { \prime \prime } ( t ) - G ( t ) ) f ( t ) d t ,
$$

and the right side defines $\mathcal { P } _ { a _ { 1 } , . . . , a _ { N } } f$ for every bounded continuous $f .$

(iii) Let $L = X + Y / $ m and $d = Y / m$ , and define

$$
\begin{array} { r } { { S } _ { m } ( Y ) f = { \mathcal { P } } _ { L , d , . . . , d } f , \qquad { \mathcal { T } } _ { m } ( Y ) f = { \mathcal { P } } _ { L , L , d , . . . , d } f , } \end{array}\tag{3.33}
$$

with m copies of d in the first list and $m - 1$ in the second. For every bounded smooth f on $[ 0 , \infty )$ , the functions $E = S _ { m } ( Y ) f$ and $A = \mathcal { T } _ { m } ( Y ) f$ solve (3.27) on $X > 0 , Y \geq 0$ and

$$
S _ { m } ( 0 ) f = f , \qquad \mathcal { T } _ { m } ( 0 ) f = \mathcal { T } f , \qquad \mathcal { T } _ { m } ( Y ) = \mathcal { T } S _ { m } ( Y ) .\tag{3.34}
$$

In particular $\partial _ { Y } S _ { m } ( Y ) f = \mathcal { L } _ { m } S _ { m } ( Y ) f .$

Proof. For $0 \leq a \leq t$ , diferentiating $\Phi _ { t } ( a ) = \sinh ( t - a ) / \sinh t$ in t and using the addition formula gives

$$
\partial _ { t } \Phi _ { t } ( a ) = { \frac { \cosh ( t - a ) \sinh t - \sinh ( t - a ) \cosh t } { \sinh ^ { 2 } t } } = { \frac { \sinh a } { \sinh ^ { 2 } t } } .
$$

Write $\begin{array} { r } { P = \prod _ { i } \Phi _ { t } ( a _ { i } ) } \end{array}$ , so that $G \ = \ \sinh { t } P$ $G ^ { \prime } =$ cosh $t P +$ sinh $t P ^ { \prime }$ and $G ^ { \prime \prime } - G =$ 2 cosh $t P ^ { \prime } + \sinh t P ^ { \prime \prime }$ . By the product rule and $\partial _ { t } \cosh ^ { 2 } t = - 2 \coth t \operatorname { c s c h } ^ { 2 } t .$

$$
\begin{array} { l } { { \displaystyle P ^ { \prime } = \mathrm { c s c h } ^ { 2 } t \sum _ { i } \sinh a _ { i } \prod _ { h \ne i } \Phi _ { t } ( a _ { h } ) , } } \\ { { \displaystyle P ^ { \prime \prime } = - 2 \coth t \mathrm { c s c h } ^ { 2 } t \sum _ { i } \sinh a _ { i } \prod _ { h \ne i } \Phi _ { t } ( a _ { h } ) + \mathrm { c s c h } ^ { 4 } t \sum _ { i \ne j } \sinh a _ { i } \sinh a _ { j } \prod _ { h \ne \{ i , j \} } \Phi _ { t } ( a _ { h } ) . } }  \end{array}
$$

In 2 cosh t $P ^ { \prime } + \sinh { t } P ^ { \prime \prime }$ the two single sums cancel, since 2 cosh t csc $\boldsymbol { \mathrm { 1 } } ^ { 2 } t = 2$ coth t csch $t =$ sinh t · 2 coth $t \cosh ^ { 2 } t ,$ and the sum over ordered pairs $i \neq j$ is twice the sum over $i < j ;$ this gives (3.31). For $t \geq L$ every term is nonnegative, because sinh $a _ { i } \geq 0$ and $\Phi _ { t } ( a _ { h } ) =$ sinh $( t - a _ { h } ) / \sinh t \geq 0$ for $a _ { h } \ \leq \ L \ \leq \ t ,$ and the bound follows from sinh $a _ { i } \ \leq$ sinh $L ,$ $\Phi _ { t } ( a _ { h } ) \leq 1$ and the number $N ( N - 1 ) / 2$ of pairs. For the atom, $\Phi _ { L } ( L ) = 0$ gives $P ( L ) = 0$ so $G ^ { \prime } ( L )$ = sinh L $P ^ { \prime } ( L )$ . If $a _ { i }$ is the only entry equal to $L ,$ every term of $P ^ { \prime } ( L )$ other than the ith contains the factor $\Phi _ { L } ( a _ { i } ) = 0$ , and the ith term is $\mathrm { c s c h ^ { 2 } } \ .$ L sinh $L \prod _ { h \neq i } \Phi _ { L } ( a _ { h } )$ , so $\begin{array} { r } { G ^ { \prime } ( L ) = \prod _ { h \neq i } \Phi _ { L } ( a _ { h } ) } \end{array}$ . If two entries equal $L _ { ; }$ every term contains a vanishing factor and $G ^ { \prime } ( L ) = 0$

Let f be smooth with compact support. Two integrations by parts on $[ L , \infty )$ give

$$
\int _ { L } ^ { \infty } G f ^ { \prime \prime } d t = \Bigl [ G f ^ { \prime } - G ^ { \prime } f \Bigr ] _ { t = L } ^ { t = \infty } + \int _ { L } ^ { \infty } G ^ { \prime \prime } f d t = - G ( L ) f ^ { \prime } ( L ) + G ^ { \prime } ( L ) f ( L ) + \int _ { L } ^ { \infty } G ^ { \prime \prime } f d t ,
$$

the terms at infinity vanishing because $f$ has compact support. Since $G ( L ) = \sinh L P ( L ) =$ 0, subtracting $\textstyle \int _ { L } ^ { \infty } G f$ dt gives (3.32). The right side of (3.32) makes sense for bounded continuous f because $G ^ { \prime \prime } - G$ is integrable on $[ L , \infty )$ by (i).

Let $f$ be smooth with compact support in $[ 0 , \infty )$ and put $c ( t ) = \sinh t \left( f ^ { \prime \prime } - f \right) ( t )$ , which is bounded with compact support. Define

$$
E ( X , Y ) = \int _ { 0 } ^ { \infty } c ( t ) \Phi _ { t } ( L ) \Phi _ { t } ( d ) ^ { m } d t , \qquad A ( X , Y ) = \int _ { 0 } ^ { \infty } c ( t ) \Phi _ { t } ( L ) ^ { 2 } \Phi _ { t } ( d ) ^ { m - 1 } d t .
$$

The integrands vanish for $t < L$ , so by (ii), applied to the lists $( L , d , \ldots , d )$ and $( L , L , d , \ldots , d )$ $E = S _ { m } ( Y ) f$ and $A = \mathcal { T } _ { m } ( Y ) f$ . For each t the integrands solve (3.27) by Proposition 3.2. They are Lipschitz in $( X , Y )$ with constant bounded by a multiple of coth $t ,$ since $| \Phi _ { t } ^ { \prime } | \leq$ coth t on $[ 0 , t ]$ , and $c ( t )$ coth t = cosh $: ( f ^ { \prime \prime } - f ) ( t )$ is bounded, so diferentiation under the integral sign is permitted and $( E , A )$ solves (3.27).

Let $f$ be bounded and smooth, and let $\chi _ { n }$ be smooth with $0 \leq \chi _ { n } \leq 1 , \chi _ { n } = 1 { \mathrm { ~ o n ~ } } [ 0 , n ]$ and $\chi _ { n } = 0 ~ \mathrm { o n } ~ [ n + 1 , \infty )$ . The functions $f _ { n } = \chi _ { n } f$ are smooth with compact support, so $E _ { n } = S _ { m } ( Y ) f _ { n }$ and $A _ { n } = \tau _ { m } ( Y ) f _ { n }$ solve (3.27). Let $K = G ^ { \prime \prime } - G$ denote the kernel of either list. Where $L < n$ the atoms of f and $f _ { n }$ coincide, and $f = f _ { n }$ on $[ L , n ]$ , so

$$
S _ { m } ( Y ) f - E _ { n } = \int _ { n } ^ { \infty } K ( t ) \left( f - f _ { n } \right) ( t ) d t ,
$$

and likewise for $\tau _ { m } ( Y ) f - A _ { n }$ . The kernel K and its first derivatives in L and d are bounded by a multiple of sinh $^ { - 3 } t .$ locally uniformly in $( L , d )$ , and the lower limit n does not depend on $( X , Y )$ . Hence the diferences and their first derivatives are bounded by a multiple of sup $\begin{array} { r } { | f | \int _ { n } ^ { \infty } \sinh ^ { - 3 } t d t } \end{array}$ , which tends to zero uniformly on compact subsets of $\{ X > 0 , Y \geq 0 \}$ Thus $( \cdots , A _ { n } ) \to ( S _ { m } ( Y ) f , { \mathcal { T } } _ { m } ( Y ) f )$ together with first derivatives, and (3.27) passes to the limit.

We now prove (3.34). At $Y = 0$ we have $L = X$ and $d = 0$ . For the list $( X , 0 , \ldots , 0 )$ every pair $i < j$ contains the factor sinh $0 = 0$ , so $G ^ { \prime \prime } - G \equiv 0$ , and $G ^ { \prime } ( X ) = \Phi _ { X } ( 0 ) ^ { m } = 1 ;$ hence $S _ { m } ( 0 ) f = f$ . For the list $( X , X , 0 , \ldots , 0 )$ the atom vanishes, the only surviving pair is the two copies of X, and $G ^ { \prime \prime } - G = 2 \sinh ^ { 2 } X / \sinh ^ { 3 } t ;$ hence $\begin{array} { r } { T _ { m } ( 0 ) f = 2 \sinh ^ { 2 } X \int _ { X } ^ { \infty } f ( t ) } \end{array}$ sinh<sup>−3</sup> t dt = $\boldsymbol { \mathcal { T } f }$ by (3.13). Finally $A = \mathcal { T } _ { m } ( Y ) f$ tends to zero as $X  \infty$ with Y fixed, since by $\mathrm { ( i ) } \ | A | \ \leq \ N ^ { 2 }$ sup |f| sin $\begin{array} { r } { \mathrm { { 1 } } ^ { 2 } L \int _ { t } ^ { \infty } \sinh ^ { - 3 } t d t = \mathcal { O } ( e ^ { - L } ) } \end{array}$ . The first equation of (3.27) reads $( A / \sinh ^ { 2 } X ) _ { X } = - 2 E / \sinh ^ { 3 } \ddot { X }$ , and integrating it from X to infinity gives $A = \mathcal { I } E$ , that is, $\mathcal { T } _ { m } ( Y ) = \mathcal { I } S _ { m } ( Y )$ . Substituting this into the second equation gives $\begin{array} { r } { E _ { Y } = \frac { 1 } { m } E _ { X } + \mathcal { K } E = } \end{array}$ $\mathcal { L } _ { m } E$ □

In (3.33) the atom $G ^ { \prime } ( L ) f ( L ) = ( \sinh X / \sinh L ) ^ { m } f ( L )$ carries the data along the characteristic, with the integrating factor of the local part − coth X of K, and the kernel (3.31) collects the contribution of the nonlocal part csch $X { \mathcal { T } } ;$ together they form the Green’s function of the Cauchy problem. The name propagator is justified by the semigroup law $S _ { m } ( Y ^ { \prime } ) S _ { m } ( Y ) = S _ { m } ( Y + Y ^ { \prime } )$ : since $\Phi _ { t } ( X + d ) \ : = \ : \Phi _ { t } ( d ) \Phi _ { t - d } ( X )$ , formula (3.29) reads $S _ { m } ( Y ) \Phi _ { t } = \Phi _ { t } ( d ) ^ { m + 1 } \Phi _ { t - d } ,$ so a mode is sent to a multiple of the mode with pole $t - d ,$ and $\Phi _ { t } ( d ) \Phi _ { t - d } ( d ^ { \prime } ) = \Phi _ { t } ( d + d ^ { \prime } )$ gives the law on modes and hence, by superposition, on general data. Section 3.5 applies $S _ { 2 }$ to the interface data in Regions I and II, and Section 3.6 applies $S _ { 3 }$ followed by $S _ { 2 }$ in Region III; the regional formulas then follow from (3.29) mode by mode, and the positivity of the Green’s function of the Cauchy problem is used throughout Section 4.

## 3.5 Regions I and II: the single-integral formula

We now derive the solution formula (2.15) in Regions I and II. In this case, the five expert formula is an additive perturbation of the four expert formula established by Bayraktar, Ekren, and Zhang [6, Theorem 3.1]. To illustrate the characteristic method used below for five experts, we first give a simple rederivation of the four expert formula in the current notation; its $C ^ { 2 }$ regularity and the optimality of the COMB control are taken from [6, Theorems 3.1 and 3.2].

For $x _ { 1 } \geq x _ { 2 } \geq x _ { 3 } \geq x _ { 4 }$ , recall the coordinates (2.3), for which $z _ { 1 } \ge z _ { 2 } \ge | z _ { 3 } |$ . We write $u _ { 4 } = x _ { 1 } + F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } ) / k$ and seek a correction satisfying the two fixed-direction equations $\partial _ { z _ { 2 } } ^ { 2 } F _ { 4 } = \partial _ { z _ { 3 } } ^ { 2 } F _ { 4 } = F _ { 4 }$ . These correspond to the controls $( 1 , 0 , 1 , 0 )$ and $( 1 , 0 , 0 , 1 )$ . Equality of adjacent coordinate derivatives on the three faces gives

$$
\begin{array} { l l } { { \partial _ { z _ { 1 } } F _ { 4 } - \partial _ { z _ { 2 } } F _ { 4 } = 0 } } & { { ( z _ { 1 } = z _ { 2 } ) , } } \\ { { \partial _ { z _ { 2 } } F _ { 4 } + \partial _ { z _ { 3 } } F _ { 4 } = - 1 } } & { { ( z _ { 3 } = - z _ { 2 } ) , } } \\ { { \partial _ { z _ { 2 } } F _ { 4 } - \partial _ { z _ { 3 } } F _ { 4 } = 0 } } & { { ( z _ { 3 } = z _ { 2 } ) . } } \end{array}\tag{3.35}
$$

We first solve the equation in $z _ { 3 }$ , writing $F _ { 4 } = \alpha ( z _ { 1 } , z _ { 2 } )$ cosh $z _ { 3 } + \beta ( z _ { 1 } , z _ { 2 } )$ sinh $z _ { 3 }$ , so that $\partial _ { z _ { 2 } } F _ { 4 } = \alpha _ { z _ { 2 } } \cosh z _ { 3 } + \beta _ { z _ { 2 } }$ sinh $z _ { 3 }$ and $\partial _ { z _ { 3 } } F _ { 4 } = \alpha \sinh z _ { 3 } + \beta$ cosh $z _ { 3 }$ . On the faces $z _ { 3 } = - z _ { 2 }$ and $z _ { 3 } = z _ { 2 }$ the last two conditions in (3.35) read

$$
\begin{array} { r l } & { \alpha _ { z _ { 2 } } \cosh z _ { 2 } - \beta _ { z _ { 2 } } \sinh z _ { 2 } - \alpha \sinh z _ { 2 } + \beta \cosh z _ { 2 } = - 1 , } \\ & { \alpha _ { z _ { 2 } } \cosh z _ { 2 } + \beta _ { z _ { 2 } } \sinh z _ { 2 } - \alpha \sinh z _ { 2 } - \beta \cosh z _ { 2 } = 0 , } \end{array}
$$

and adding and subtracting them gives

$$
\partial _ { z _ { 2 } } \alpha \cosh z _ { 2 } - \alpha \sinh z _ { 2 } = - { \textstyle \frac { 1 } { 2 } } , \qquad \beta \cosh z _ { 2 } - \partial _ { z _ { 2 } } \beta \sinh z _ { 2 } = - { \textstyle \frac { 1 } { 2 } } .
$$

Dividing by $\cosh ^ { 2 } z _ { 2 }$ and by $\sinh ^ { 2 } z _ { 2 } .$ , respectively, these are $( \alpha / \cosh z _ { 2 } ) _ { z _ { 2 } } = - { \textstyle \frac { 1 } { 2 } } \operatorname { s e c h } ^ { 2 } z _ { 2 }$ and $( \beta / \sinh z _ { 2 } ) _ { z _ { 2 } } = { \textstyle { \frac { 1 } { 2 } } } \cosh ^ { 2 } z _ { 2 }$ , so integration in z<sub>2</sub> gives $\alpha = A ( z _ { 1 } )$ cosh z − sinh $z _ { 2 } / 2$ and $\beta = B ( z _ { 1 } )$ sinh $z _ { 2 } - \cosh z _ { 2 } / 2$ . Thus

$$
F _ { 4 } = A ( z _ { 1 } ) \cosh z _ { 2 } \cosh z _ { 3 } + B ( z _ { 1 } ) \sinh z _ { 2 } \sinh z _ { 3 } - { \textstyle \frac { 1 } { 2 } } \sinh ( z _ { 2 } + z _ { 3 } ) .
$$

This also satisfies the equation in $z _ { 2 }$ . On the remaining face $z _ { 1 } = z _ { 2 } > 0$

$$
\begin{array} { r l } & { \partial _ { z _ { 1 } } F _ { 4 } - \partial _ { z _ { 2 } } F _ { 4 } = \left( A ^ { \prime } \cosh z _ { 2 } - A \sinh z _ { 2 } + \frac { 1 } { 2 } \cosh z _ { 2 } \right) \cosh z _ { 3 } } \\ & { \qquad + \left( B ^ { \prime } \sinh z _ { 2 } - B \cosh z _ { 2 } + \frac { 1 } { 2 } \sinh z _ { 2 } \right) \sinh z _ { 3 } , } \end{array}
$$

and the coeficients of cosh $z _ { 3 }$ and sinh $z _ { 3 }$ must vanish separately, since $- z _ { 1 } \le z _ { 3 } \le z _ { 1 }$ . With $z _ { 2 } = z _ { 1 }$ we obtain

$$
\begin{array} { l l } { { A ^ { \prime } - \operatorname { t a n h } z _ { 1 } A = - \frac { 1 } { 2 } , ~ } } & { { B ^ { \prime } - \coth z _ { 1 } B = - \frac { 1 } { 2 } . } } \end{array}\tag{3.36}
$$

The integrating factors are sech $z _ { 1 }$ and csch $z _ { 1 }$ , respectively. Boundedness of the correction fixes the constants of integration: first take $z _ { 2 } = z _ { 3 } = 0$ and let $z _ { 1 } \to \infty$ to exclude a homogeneous term proportional to cosh $z _ { 1 }$ in $A ;$ then take $z _ { 2 } = z _ { 3 } > 0$ fixed to exclude a term proportional to sinh $z _ { 1 }$ in $B .$ . Integrating from infinity therefore gives

$$
\begin{array} { l } { { A ( z _ { 1 } ) = { \frac { 1 } { 2 } } \cosh z _ { 1 } \displaystyle \int _ { z _ { 1 } } ^ { \infty } \operatorname { s e c h } t d t = \theta ( z _ { 1 } ) \cosh z _ { 1 } , \hfill } } \\ { { B ( z _ { 1 } ) = { \frac { 1 } { 2 } } \sinh z _ { 1 } \displaystyle \int _ { z _ { 1 } } ^ { \infty } \operatorname { c s c h } t d t = { \frac { 1 } { 2 } } \lambda ( z _ { 1 } ) \sinh z _ { 1 } . } } \end{array}\tag{3.37}
$$

Consequently $F _ { 4 }$ is the function (2.14). At $z _ { 1 } = 0$ , the domain forces $z _ { 2 } = z _ { 3 } = 0$ , and the formula is interpreted by continuity. In particular, the product containing $\lambda ( z _ { 1 } )$ sinh $z _ { 1 }$ tends to zero. To identify (2.14) with [6, equation (3.1)], reverse their ascending order, $( x ^ { ( 1 ) } , \ldots , x ^ { ( 4 ) } ) \ = \ ( x _ { 4 } , x _ { 3 } , x _ { 2 } , x _ { 1 } )$ : their exponential argument becomes $- z _ { 1 }$ , their three hyperbolic arguments become − $\cdot z _ { 2 } , - z _ { 3 } , z _ { 1 } , k ( x _ { 1 } - x _ { 2 } ) = z _ { 2 } + z _ { 3 }$ , and atanh $( e ^ { - z _ { 1 } } ) = \lambda ( z _ { 1 } ) / 2$ Substitution gives exactly (2.14). By [6, Theorems 3.1 and 3.2], $u _ { 4 } = x _ { 1 } + F _ { 4 } / k$ is the global $C ^ { 2 }$ solution of the four-expert equation and $( 1 , 0 , 1 , 0 )$ attains its Hamiltonian maximum throughout the closed ordered sector; Section 4 uses these two facts.

To lift the four expert solution to five experts, we need to use the density p defined earlier in (2.11). The density is related to the trace by

$$
p ( X ) = \textstyle { \frac { 1 } { 2 } } \operatorname { t a n h } X - \eta ( X ) .
$$

Indeed, by (3.24) the right side satisfies

$$
\begin{array} { r } { p ^ { \prime } + 6 \coth X p = \frac { 1 } { 2 } \operatorname { s e c h } ^ { 2 } X , } \end{array}\tag{3.38}
$$

since $( { \textstyle { \frac { 1 } { 2 } } } \operatorname { t a n h } X ) ^ { \prime } + 3$ coth X tanh $X = { \textstyle { \frac { 1 } { 2 } } } \mathrm { s e c h } ^ { 2 } X + 3$ and $\eta ^ { \prime } + 6$ coth $X \eta = 3$ . With the integrating factor $\sinh ^ { 6 } X$ this reads $( \sinh ^ { 6 } X p ) ^ { \prime } = { \textstyle { \frac { 1 } { 2 } } } \sinh ^ { 6 } X \operatorname { s e c h } ^ { 2 } X$ , and, exactly as for $\eta ,$ the solution that is regular at zero is the positive integral in (2.11). The elementary form (2.12) follows from $\eta = 3 \coth X - 1 5 I ( X ) / \sinh ^ { 6 } X$ . Now, the corresponding interface endpoint and upper trace are

$$
e _ { 4 } ( X ) = \cosh X \theta ( X ) , \qquad h _ { 4 } ( X ) = \cosh ^ { 2 } X \theta ( X ) - { \textstyle { \frac { 1 } { 2 } } } \sinh X = { \cal T } e _ { 4 } ( X ) .\tag{3.39}
$$

Since $\eta = e - e ^ { \prime \prime }$ and $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ tanh $X - \eta$ , we have $\begin{array} { r } { e ^ { \prime \prime } - e = p - \frac { 1 } { 2 } } \end{array}$ tanh $X ;$ and $\theta ^ { \prime } = - \frac { 1 } { 2 }$ sech X gives $e _ { 4 } ^ { \prime \prime } - e _ { 4 } = - \frac { 1 } { 2 }$ tanh $X$ , so the decaying diference $g = e - e _ { 4 }$ obeys $g ^ { \prime \prime } - g = p$ . Its Green representation, which is the expansion of $g$ in truncated modes with density sinh $t p ( t )$ , is

$$
g ( X ) = \int _ { X } ^ { \infty } \sinh ( t - X ) p ( t ) d t = \int _ { 0 } ^ { \infty } \sinh t p ( t ) \Phi _ { t } ( X ) d t .\tag{3.40}
$$

The integral converges, since (2.11) gives $p ( t ) = \mathcal { O } ( e ^ { - 2 t } )$ . There is no free decaying homogeneous term: (3.26) and the expansion of $e _ { 4 }$ show $g ( X ) = \mathcal { O } ( e ^ { - 2 X } )$ , excluding both $e ^ { X }$ and $e ^ { - X }$ homogeneous terms. Indeed, every other solution of $g ^ { \prime \prime } - g = p$ that decays at infinity difers from (3.40) by $B e ^ { - X }$ . The stronger decay $g = \mathcal { O } ( e ^ { - 2 X } )$ forces $B = 0$ , so no additional boundary assumption at zero is needed.

We now solve the equations at the interface where $Y = 0 .$ , or rather $y _ { 1 } = y _ { 3 }$ . Here, the traces coincide pairwise by (3.7), $a = b = F ( X , 0 , X , R )$ and $\ell = q = F ( 0 , X , 0 , R )$ ). We write $H ( X , R )$ and $E ( X , R )$ for these common values. They satisfy

$$
H = \mathcal { I } E , \qquad E _ { R } = 2 \mathcal { K } E + 1 , \qquad E ( X , 0 ) = e ( X ) .\tag{3.41}
$$

The first equation is $a = \tau \ell$ of Section 3.2, the integrated form of the first equation in $\left( 3 . 1 0 \right)$ , at $Y = 0$ . For the second equation, add the two equations (3.16). At $Y = 0$ we have $\ell = q = E$ , hence $\ell _ { X } = q _ { X } = E _ { X } , \ell _ { R } = q _ { R } = E _ { R }$ , and $\begin{array} { r } { { \cal K } \ell = { \cal K } q = { \cal K } E } \end{array}$ , so by (3.15) the sum is

$$
\begin{array} { r } { \ell _ { Y } + q _ { Y } = E _ { X } + 2 \mathcal { K } E + \frac { 1 } { 2 } - \frac { 1 } { 2 } E _ { R } . } \end{array}
$$

On the other hand, by (3.7), $\ell _ { Y } = F _ { y _ { 3 } } ( 0 , X , Y , R )$ and $q _ { Y } = F _ { y _ { 1 } } ( Y , X , 0 , R )$ . Therefore at $Y = 0$ both are derivatives of F at the point $( 0 , X , 0 , R )$ , which lies on the faces $y _ { 1 } = 0$ and $y _ { 3 } = 0$ . The first and third conditions in (3.6) give $\begin{array} { r } { F _ { y _ { 1 } } = \frac { 1 } { 2 } ( F _ { y _ { 2 } } - 1 ) } \end{array}$ and $\begin{array} { r } { F _ { y _ { 3 } } = \frac { 1 } { 2 } ( F _ { y _ { 2 } } + F _ { y _ { 4 } } ) } \end{array}$ there, while $E ( X , R ) = F ( 0 , X , 0 , R )$ gives $E _ { X } = F _ { y _ { 2 } }$ and $E _ { R } = F _ { y _ { 4 } }$ . Hence

$$
\ell _ { Y } + q _ { Y } = E _ { X } + \textstyle { \frac { 1 } { 2 } } E _ { R } - \frac { 1 } { 2 } .
$$

Equating the two expressions gives $E _ { R } = 2 \mathcal { K } E + 1$ . Finally, $E ( X , 0 ) = F ( 0 , X , 0 , 0 ) =$ $\Psi ( 0 , X ) = e ( X )$ by (3.20), and likewise $H ( X , 0 ) = \Psi ( X , 0 ) = h ( X )$ , in agreement with $h = \mathcal { T } e$

The pair $( h _ { 4 } , e _ { 4 } )$ of (3.39) is a stationary solution of the first two equations in (3.41): $h _ { 4 } = \tau e _ { 4 }$ by (3.39), and

$$
{ \cal K } e _ { 4 } = - \coth X e _ { 4 } + \operatorname { c s c h } X h _ { 4 } = - \frac { \cosh ^ { 2 } X } { \sinh X } \theta ( X ) + \frac { \cosh ^ { 2 } X } { \sinh X } \theta ( X ) - { \textstyle \frac { 1 } { 2 } } = - { \textstyle \frac { 1 } { 2 } } .
$$

Therefore $\boldsymbol { G } = \boldsymbol { E } - \boldsymbol { e } _ { 4 }$ satisfies $G _ { R } = 2 \mathcal { K } G$ with $G ( X , 0 ) = e - e _ { 4 } = g$ . Expanding $g$ in truncated modes by $( 3 . 4 0 )$ and using $\mathcal { K } \Phi _ { t } = - \coth t \Phi _ { t }$ from (3.28), each mode evolves in R by the factor $e ^ { - 2 R \mathrm { c o t h } t } ;$ applying $\mathcal { T } \Phi _ { t } = \Phi _ { t } ^ { 2 }$ , also from (3.28), to the result gives H. Thus

$$
\begin{array} { l } { { E ( X , R ) = e _ { 4 } ( X ) + \displaystyle \int _ { X } ^ { \infty } \sinh t p ( t ) e ^ { - 2 R \coth t } \Phi _ { t } ( X ) d t , } } \\ { { H ( X , R ) = h _ { 4 } ( X ) + \displaystyle \int _ { X } ^ { \infty } \sinh t p ( t ) e ^ { - 2 R \coth t } \Phi _ { t } ( X ) ^ { 2 } d t . } } \end{array}\tag{3.42}
$$

Substituting these expressions directly into (3.41) verifies the interface equations.

We now use the trace characteristics to extend the interface data into Regions I and II, by means of the propagator $S _ { 2 }$ of Lemma 3.3.

In Region I, $y _ { 1 } \le y _ { 3 }$ , use $X = y _ { 1 } + y _ { 2 } , Y = y _ { 3 } - y _ { 1 } , R = y _ { 4 }$ , as in (3.8). The left system (3.10) contains no R-derivative, so R is a parameter, and its second equation is $\begin{array} { r } { \ell _ { Y } = \mathcal { L } _ { 2 } \ell + \frac { 1 } { 2 } } \end{array}$

by (3.16), with the data $\ell ( \cdot , 0 , R ) = E ( \cdot , R )$ at the interface. The constant $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ is removed by a stationary solution: the pair

$$
E _ { 3 } = { \textstyle \frac { 1 } { 6 } } ( 3 + e ^ { - 2 X } ) , \qquad A _ { 3 } = { \textstyle \frac { 2 } { 3 } } e ^ { - X }
$$

satisfies the first equation of (3.10), $A _ { 3 } ^ { \prime } = 2$ coth X $A _ { 3 } - 2$ csch X $E _ { 3 } ,$ and $A _ { 3 }$ is bounded, so $A _ { 3 } = \tau E _ { 3 }$ as in Section 3.2; a direct computation then gives $\begin{array} { r } { \mathcal { L } _ { 2 } E _ { 3 } = \frac { 1 } { 2 } E _ { 3 } ^ { \prime } - } \end{array}$ coth X $E _ { 3 } +$ csch X $A _ { 3 } = - \frac { 1 } { 2 }$ . Hence $\ell - E _ { 3 }$ solves the homogeneous Cauchy problem $( \ell { - } E _ { 3 } ) _ { Y } = \mathcal { L } _ { 2 } ( \ell { - } E _ { 3 } )$ with the bounded smooth data $E ( \cdot , R ) - E _ { 3 }$ , and Lemma 3.3(iii) together with $a = \tau \ell$ from Section 3.2 gives

$$
\ell = E _ { 3 } + { \cal S } _ { 2 } ( Y ) \big ( E ( \cdot , R ) - E _ { 3 } \big ) , \qquad a = \mathcal { I } \ell .\tag{3.43}
$$

In Region II, $y _ { 3 } \leq y _ { 1 } \leq y _ { 3 } + 2 y _ { 4 }$ , use $X = y _ { 2 } + y _ { 3 } , Y = y _ { 1 } - y _ { 3 } , R = y _ { 4 }$ , as in $\left( 3 . 9 \right)$ and put $Z = R - Y / 2 ;$ the condition $Z \ge 0$ is the inequality $y _ { 1 } \le y _ { 3 } + 2 y _ { 4 }$ defining the region. The right system (3.11) contains $q _ { R } .$ , and its second equation is $q _ { Y } + { \textstyle \frac { 1 } { 2 } } q _ { R } = \mathcal { L } _ { 2 } q$ by (3.16). The operator $\partial _ { Y } + { \textstyle { \frac { 1 } { 2 } } } \partial _ { R }$ diferentiates along the lines $R - Y / 2 = Z$ of the $( Y , R ) \mathrm { - }$ plane, so for fixed Z the function $\widetilde { q } ( X , Y ) = q ( X , Y , Z + Y / 2 )$ satisfies $\widetilde { q } _ { Y } = \mathcal { L } _ { 2 } \widetilde { q }$ with data $\widetilde { q } ( \cdot , 0 ) = q ( \cdot , 0 , Z ) = E ( \cdot , Z )$ : the backward characteristic through $( Y , R )$ reaches the interface $Y = 0$ at $R = Z$ . Lemma 3.3(iii) and $b = \mathcal { T } q$ give

$$
\begin{array} { r } { q = S _ { 2 } ( Y ) E ( \cdot , Z ) , \qquad b = \boldsymbol { \mathcal { T } } q . } \end{array}\tag{3.44}
$$

When $Z < 0$ the characteristic meets the bottom face $R = 0$ first; this is Region III, treated in Section 3.6.

We are now equipped to derive (2.15). In the coordinates (2.3)–(2.4) of Section 2, Region II is $z _ { 3 } \ge 0 , z _ { 4 } \ge 0$ and Region I is $z _ { 3 } \le 0 \le z _ { 4 }$ . The characteristic coordinates of Lemma 3.3 for $m = 2$ are $L = X + Y / 2$ and $d = Y / 2$ . In Region II, $\begin{array} { r } { L = y _ { 2 } + \frac { 1 } { 2 } ( y _ { 1 } + y _ { 3 } ) = z _ { 1 } , } \end{array}$ $\begin{array} { r } { d = \frac { 1 } { 2 } ( y _ { 1 } - y _ { 3 } ) = z _ { 3 } } \end{array}$ , and $Z = R - Y / 2 = y _ { 4 } - z _ { 3 } = z _ { 4 } ;$ in Region I, $\begin{array} { r } { L = y _ { 2 } + \frac { 1 } { 2 } ( y _ { 1 } + y _ { 3 } ) = z _ { 1 } } \end{array}$ $\begin{array} { r } { d = \frac { 1 } { 2 } ( y _ { 3 } - y _ { 1 } ) = - z _ { 3 } } \end{array}$ , and $R = y _ { 4 } = z _ { 4 }$ . With $d = | z _ { 3 } |$ , both regions are therefore described by $0 \leq d \leq z _ { 2 } \leq z _ { 1 }$ and $z _ { 4 } \geq 0$ , and the propagations (3.43) and (3.44) both start from the interface data $E ( \cdot , z _ { 4 } )$ of (3.42).

Consider Region II. By $( 3 . 4 2 ) , E ( \cdot , Z )$ is e<sub>4</sub> plus a superposition of truncated modes with density sinh $t p ( t ) e ^ { - 2 Z \coth t }$ , so (3.29), applied mode by mode in (3.44), gives

$$
\begin{array} { l } { \displaystyle { q = S _ { 2 } ( Y ) e _ { 4 } + \int _ { z _ { 1 } } ^ { \infty } \sinh t p ( t ) e ^ { - 2 z _ { 4 } \coth t } \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 3 } ) ^ { 2 } d t , } } \\ { \displaystyle { b = \mathbb { Z } S _ { 2 } ( Y ) e _ { 4 } + \int _ { z _ { 1 } } ^ { \infty } \sinh t p ( t ) e ^ { - 2 z _ { 4 } \coth t } \Phi _ { t } ( z _ { 1 } ) ^ { 2 } \Phi _ { t } ( z _ { 3 } ) d t , } } \end{array}\tag{3.45}
$$

the integrals starting at $z _ { 1 }$ because $\Phi _ { t } ( z _ { 1 } ) = 0$ for $t < z _ { 1 }$ . Reconstructing by (3.9), with $X = y _ { 2 } + y _ { 3 }$ , the integrands combine to

$$
\frac { \sinh y _ { 3 } \Phi _ { t } ( z _ { 1 } ) + \sinh y _ { 2 } \Phi _ { t } ( z _ { 3 } ) } { \sinh ( y _ { 2 } + y _ { 3 } ) } \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 3 } ) = \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 3 } ) \Phi _ { t } ( z _ { 2 } ) ,
$$

because $z _ { 1 } = z _ { 2 } + y _ { 2 }$ and $z _ { 3 } = z _ { 2 } - y _ { 3 }$ , so that with $u = t - z _ { 2 }$ the addition formula gives

$$
\sinh y _ { 3 } \sinh ( u - y _ { 2 } ) + \sinh y _ { 2 } \sinh ( u + y _ { 3 } ) = \sinh u \sinh ( y _ { 2 } + y _ { 3 } ) .\tag{3.46}
$$

This is the integral in (2.15). The four-expert parts of (3.45), $S _ { 2 } ( Y ) e _ { 4 }$ and $\boldsymbol { \mathcal { I } S _ { 2 } ( Y ) e _ { 4 } }$ , are the traces q and b of $F _ { 4 }$ itself: $F _ { 4 }$ is independent of R and satisfies the face conditions (3.35), so its traces solve the same Cauchy problem with the interface data $e _ { 4 }$ . Reconstructing them returns $F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } )$ , as one can also confirm by substituting (2.14) directly.

Region I is identical, with (3.43) and (3.8) in place of (3.44) and (3.9). The stationary pair $( E _ { 3 } , A _ { 3 } )$ belongs to the four-expert part, $E _ { 3 } + S _ { 2 } ( Y ) ( e _ { 4 } - E _ { 3 } )$ being the trace ℓ of $F _ { 4 } ;$ the modes carry the factors $\Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( d ) ^ { 2 }$ and $\Phi _ { t } ( z _ { 1 } ) ^ { 2 } \Phi _ { t } ( d )$ with $d = - z _ { 3 } ;$ ; and the addition formula holds with $y _ { 1 }$ in place of $y _ { 3 }$ and $- z _ { 3 }$ in place of $z _ { 3 } ,$ since $z _ { 1 } = z _ { 2 } + y _ { 2 }$ 2 and $- z _ { 3 } = z _ { 2 } - y _ { 1 }$ . Both regions therefore produce the integrand of $( 2 . 1 5 )$ , in which $z _ { 3 }$ enters only through $\left| z _ { 3 } \right|$ , and the two formulas match on the interface $y _ { 1 } = y _ { 3 }$ , where $a = b$ and $\ell = q$ by (3.7). This proves (2.15), which requires only one quadrature of elementary functions.

## 3.6 Region III: the finite formula

We now prove (2.16), in the coordinates (2.5) of Section 2, by propagating the slice trace e along the bottom face and then of it.

We begin with the propagation. In Region III, $y _ { 1 } \geq y _ { 3 } + 2 y _ { 4 }$ , we keep the Region II coordinates $X = y _ { 2 } + y _ { 3 } , Y = y _ { 1 } - y _ { 3 } , R = y _ { 4 }$ of (3.9) and put $W = Y - 2 R \ge 0 ;$ in the coordinates (2.5), $W = 3 a _ { 1 } , R = a _ { 2 } - a _ { 1 }$ , and $X = a _ { 4 } - a _ { 2 }$ . The right system (3.11) again gives $q _ { Y } + { \textstyle { \frac { 1 } { 2 } } } q _ { R } = { \mathcal { L } } _ { 2 } q$ by (3.16), but now $Z = R - Y / 2 < 0$ , so the backward characteristic $R - Y / 2 = Z$ through $( Y , R )$ meets the bottom face $R = 0$ before the interface, at $Y = W$ On the bottom face the fourth face condition gives $q _ { X } = 3 q _ { R }$ by (3.12), so there, by (3.15),

$$
\begin{array} { r } { q _ { Y } = \mathcal { L } _ { 2 } q - \frac { 1 } { 2 } q _ { R } = \frac { 1 } { 2 } q _ { X } + K q - \frac { 1 } { 6 } q _ { X } = \frac { 1 } { 3 } q _ { X } + K q = \mathcal { L } _ { 3 } q . } \end{array}
$$

The data at $Y = R = 0$ is $q ( X , 0 , 0 ) = F ( 0 , X , 0 , 0 ) = \Psi ( 0 , X ) = e ( X )$ by (3.7) and (3.20). Hence Lemma 3.3(iii) with $m = 3$ gives $q ( \cdot , W , 0 ) = S _ { 3 } ( W ) e$ on the bottom face, and with $m = 2 .$ , applied along the characteristic as in Region II to $\widetilde { q } ( X , s ) = q ( X , W + s , s / 2 )$ for $0 \leq s \leq 2 R$ , it gives

$$
q = S _ { 2 } ( 2 R ) S _ { 3 } ( W ) e , \qquad b = \mathcal { T } q ,\tag{3.47}
$$

the second identity being $b = \mathcal { T } q$ of Section 3.2.

We next compute the efect of the two propagators on a single mode. Apply (3.47) to a truncated mode $\Phi _ { t }$ with $t > a _ { 4 }$ . For $S _ { 3 } ( W )$ the characteristic coordinates of Lemma 3.3 are $L = X + W / 3 = X + a _ { 1 }$ and $d = W / 3 = a _ { 1 }$ , so (3.29) and the shift identity $\Phi _ { t } ( u + a _ { 1 } ) =$ $\Phi _ { t } ( a _ { 1 } ) \Phi _ { t - a _ { 1 } } ( u )$ give

$$
\begin{array} { r } { S _ { 3 } ( W ) \Phi _ { t } = \Phi _ { t } ( X + a _ { 1 } ) \Phi _ { t } ( a _ { 1 } ) ^ { 3 } = \Phi _ { t } ( a _ { 1 } ) ^ { 4 } \Phi _ { t - a _ { 1 } } ( X ) . } \end{array}
$$

For $S _ { 2 } ( 2 R )$ they are $L = X + R$ and $d = R$ . Applying (3.29) to the mode $\Phi _ { t - a _ { 1 } }$ and using the shift identity twice more, with $X + R + a _ { 1 } = a _ { 4 }$ and $R + a _ { 1 } = a _ { 2 }$

$$
q = \Phi _ { t } ( a _ { 1 } ) ^ { 4 } \Phi _ { t - a _ { 1 } } ( X + R ) \Phi _ { t - a _ { 1 } } ( R ) ^ { 2 } = \Phi _ { t } ( a _ { 1 } ) \Phi _ { t } ( a _ { 4 } ) \Phi _ { t } ( a _ { 2 } ) ^ { 2 } ,
$$

$$
b = \Phi _ { t } ( a _ { 1 } ) ^ { 4 } \Phi _ { t - a _ { 1 } } ( X + R ) ^ { 2 } \Phi _ { t - a _ { 1 } } ( R ) = \Phi _ { t } ( a _ { 1 } ) \Phi _ { t } ( a _ { 4 } ) ^ { 2 } \Phi _ { t } ( a _ { 2 } ) .
$$

Reconstructing by (3.9), the identity (3.46) with $u = t - a _ { 3 }$ , since $a _ { 4 } = a _ { 3 } + y _ { 2 }$ and $a _ { 2 } = a _ { 3 } - y _ { 3 }$ 2 gives

$$
\frac { \sinh y _ { 3 } \Phi _ { t } ( a _ { 4 } ) + \sinh y _ { 2 } \Phi _ { t } ( a _ { 2 } ) } { \sinh ( y _ { 2 } + y _ { 3 } ) } = \Phi _ { t } ( a _ { 3 } ) ,
$$

so a mode produces the four-factor product $\textstyle \prod _ { i = 1 } ^ { 4 } \Phi _ { t } ( a _ { i } )$

We now pass from modes to the datum e and obtain the kernel form of $F .$ For compactly supported smooth data $f$ in place of $e ,$ the Green representation (3.30) and the linearity of every step give

$$
F = \int _ { a _ { 4 } } ^ { \infty } \sinh t \prod _ { i = 1 } ^ { 4 } \Phi _ { t } ( a _ { i } ) \left( f ^ { \prime \prime } - f \right) ( t ) d t = { \mathcal { P } } _ { a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } } f
$$

by Lemma 3.3(ii), the list $( a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } )$ having its entries in $[ 0 , a _ { 4 } ]$ with $a _ { 4 } = L ;$ the integral starts at $a _ { 4 }$ because $\Phi _ { t } ( a _ { 4 } ) = 0$ for $t < a _ { 4 }$ . Both sides depend on $f$ through kernels of the type in Lemma 3.3, and the cutof argument in its proof extends the identity from compactly supported to bounded smooth data, so $F = \mathcal { P } _ { a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } } e .$ . Only the factor $\Phi _ { t } ( a _ { 4 } )$ vanishes at $t = a _ { 4 } ,$ so by Lemma 3.3(i) the atom is $\begin{array} { r } { G ^ { \prime } ( a _ { 4 } ) = \prod _ { i = 1 } ^ { 3 } \Phi _ { a _ { 4 } } ( a _ { i } ) } \end{array}$ and the kernel (3.31) is a sum over the six pairs $i < j \colon$

$$
\begin{array} { l } { { \displaystyle F = A ( a _ { 4 } , a ) e ( a _ { 4 } ) + 2 \int _ { a _ { 4 } } ^ { \infty } \frac { e ( t ) } { \sinh ^ { 3 } t } \sum _ { i < j } \sinh a _ { i } \sinh a _ { j } \prod _ { h \not \in \{ i , j \} } \Phi _ { t } ( a _ { h } ) d t , } } \\ { { \displaystyle A ( a _ { 4 } , a ) = \prod _ { i = 1 } ^ { 3 } \frac { \sinh ( a _ { 4 } - a _ { i } ) } { \sinh a _ { 4 } } . } } \end{array}\tag{3.48}
$$

This is the representation whose positivity Section 4 uses.

Finally we derive the coeficients of the finite formula. Return to compactly supported smooth data $f ,$ for which $F [ f ] = \int _ { a _ { 4 } } ^ { \infty }$ sinh $t \prod _ { i = 1 } ^ { 4 } \Phi _ { t } ( a _ { i } ) ( f ^ { \prime \prime } - f ) ( t ) d t$ , and write the three factors $\Phi _ { t } ( a _ { i } ) =$ cosh $a _ { i } \textrm { -- }$ coth t sinh $a _ { i } , \ i \ \leq \ 3$ , in the mode product. The product is multiafine in the pairs (cosh $a _ { i } , \sinh { a _ { i } } )$ , and collecting the terms with $j$ factors sinh $a _ { i }$ gives

$$
{ \cal F } [ f ] = \sum _ { j = 0 } ^ { 3 } b _ { j } [ f ] \sigma _ { j } , \qquad b _ { j } [ f ] = \int _ { a _ { 4 } } ^ { \infty } \sinh t ( - \coth t ) ^ { j } \Phi _ { t } ( a _ { 4 } ) ( f ^ { \prime \prime } - f ) ( t ) d t ,
$$

where $\sigma _ { 0 } = \cosh { a _ { 1 } }$ cosh $a _ { 2 }$ cosh a<sub>3</sub>, $\sigma _ { 3 } = \sinh { a _ { 1 } }$ sinh $a _ { 2 }$ sinh $a _ { 3 }$ , and $\sigma _ { 1 } , \sigma _ { 2 }$ are the sums of three products displayed in (2.16). By the eigenfunction identity (3.28), $( - \coth t ) ^ { j } \Phi _ { t } = K ^ { j } \Phi _ { t } ,$ with K acting on the variable $a _ { 4 } ;$ since $f ^ { \prime \prime } - f$ has compact support, $\mathcal { K } ^ { j }$ may be taken outside the integral, and the Green representation (3.30) gives $b _ { j } [ f ] = \mathcal { K } ^ { j } f$ evaluated at $a _ { 4 }$ . The integral defining $b _ { j } [ f ]$ cannot be used with e in place of $f \colon$ since $e ^ { \prime \prime } - e = - \eta \to - { \frac { 1 } { 2 } }$ and sinh $t \Phi _ { t } ( a _ { 4 } ) = \sinh ( t - a _ { 4 } )$ , its integrand would grow like $e ^ { t }$ . Integrating by parts twice instead, with $W _ { j } ( t ) = ( - \coth t ) ^ { j } \sinh ( t - a _ { 4 } ) , W _ { j } ( a _ { 4 } ) = 0$ , and $W _ { j } ^ { \prime } ( a _ { 4 } ) = ( - \coth a _ { 4 } ) ^ { j }$ , gives

$$
b _ { j } [ f ] = ( - \coth a _ { 4 } ) ^ { j } f ( a _ { 4 } ) + \int _ { a _ { 4 } } ^ { \infty } ( W _ { j } ^ { \prime \prime } - W _ { j } ) ( t ) f ( t ) d t ,
$$

whose density $W _ { j } ^ { \prime \prime } - W _ { j }$ is $\mathcal { O } ( e ^ { - 3 t } )$ for fixed $a _ { 4 } > 0 ;$ it vanishes for $j = 0$ and equals 2 sinh $a _ { 4 } \mathrm { c s c h } ^ { 3 } t \ \mathrm { f o r } \ j = 1$ . This form and the kernel form $F [ f ] = \mathcal { P } _ { a _ { 1 } , a _ { 2 } , a _ { 3 } , a _ { 4 } } f$ both pass to the bounded datum e under the cutofs $f _ { n }  e$ used above, by dominated convergence, and so does ${ \mathcal { K } } ^ { j } f _ { n } \to { \mathcal { K } } ^ { j } e$ , locally uniformly on $( 0 , \infty )$ , because I acts on bounded functions through an absolutely convergent integral. Hence $\begin{array} { r } { F = \sum _ { j = 0 } ^ { 3 } b _ { j } \sigma _ { j } } \end{array}$ with $b _ { j } = \mathcal { K } ^ { j } e$ evaluated at $a _ { 4 } , j = 0 , 1 , 2 , 3$ , and $b _ { 0 } = e$ . It remains to compute ${ \kappa } ^ { j } e$ for $j \le 3$

Two integration-by-parts identities are

$$
\partial _ { X } \mathcal { T } = \mathcal { T } ( \partial _ { X } + \mathcal { K } ) , \qquad \mathcal { K } ^ { 2 } - [ \partial _ { X } , \mathcal { K } ] = \mathrm { I d } .\tag{3.49}
$$

For the second identity, we set

$$
J _ { j } = \int _ { X } ^ { \infty } f ( t ) \coth ^ { j } t / \sinh ^ { 3 } t d t .
$$

Both $\kappa ^ { 2 } f$ and $[ \partial _ { X } , { \cal K } ] f$ contain 2 cosh $X J _ { 0 } \mathrm { ~ - ~ } 6$ sinh $X J _ { 1 } ;$ their multiplication terms are respectively coth<sup>2</sup> Xf and $\operatorname { c s c h } ^ { 2 } X f$ . This proves their diference is $f .$ The first identity follows by diferentiating (3.13) and integrating $\mathcal { T } \partial _ { X } f$ by parts. Also, the antiderivatives $\begin{array} { r } { \int \cosh ^ { 3 } t \dot { d t } = - \frac { 1 } { 2 } } \end{array}$ csch t coth $t + { \textstyle \frac { 1 } { 2 } } \lambda ( t )$ and $\begin{array} { r } { \int \lambda ( t ) \cosh ^ { 2 } t d t = - \lambda ( t ) } \end{array}$ coth t + csch $t ,$ which use $\mathrm {  ~ \lambda ~ } ^ { \prime } = - \cosh { t } .$ , give I1 = cosh $\bar { X ( \bar { \mathbf { \alpha } } - \lambda ( X ) }$ sinh<sup>2</sup> X and $\mathcal { T } ( \lambda$ sinh $X ) = 2 \lambda ( X )$ sinh $X$ cosh $X -$ 2 sinh X, hence

$$
\begin{array} { r } { K 1 = - \lambda ( X ) \sinh X , \qquad K ^ { 2 } 1 = 2 - \lambda ( X ) \cosh X . } \end{array}\tag{3.50}
$$

The compatibility relation (3.23) is $\begin{array} { r } { K e \ = \ \frac { 1 } { 6 } ( e ^ { \prime } - 3 ) } \end{array}$ , that is, $6 b _ { 1 } = e ^ { \prime } - 3$ . The second identity in (3.49) says $\mathcal { K } f ^ { \prime } = ( \mathcal { K } f ) ^ { \prime } - \mathcal { K } ^ { 2 } f + f$ . With $f = e$ it gives $\begin{array} { r } { K e ^ { \prime } = b _ { 1 } ^ { \prime } - b _ { 2 } + e . } \end{array}$ , so $\begin{array} { r } { b _ { 2 } = K b _ { 1 } = \frac { 1 } { 6 } K e ^ { \prime } - \frac { 1 } { 2 } K 1 = \frac { 1 } { 6 } ( b _ { 1 } ^ { \prime } - b _ { 2 } + e ) - \frac { 1 } { 2 } K 1 } \end{array}$ , that is, $7 b _ { 2 } = b _ { 1 } ^ { \prime } + e - 3 \mathcal { K } 1$ . With $f = b _ { 1 }$ it gives $\begin{array} { r } { K b _ { 1 } ^ { \prime } = \check { b } _ { 2 } ^ { \prime } - { b _ { 3 } } + { b _ { 1 } } , \mathrm { s o } \check { b } _ { 3 } = K { b _ { 2 } } = \frac { 1 } { 7 } ( K \bar { b } _ { 1 } ^ { \prime } + K e - 3 K ^ { 2 } { 1 } ) = \frac { 1 } { 7 } ( { b } _ { 2 } ^ { \prime } - { b _ { 3 } } + 2 { b _ { 1 } } - 3 K ^ { 2 } { 1 } ) } \end{array}$ , that $\mathrm { i s } , 8 b _ { 3 } = b _ { 2 } ^ { \prime } + 2 b _ { 1 } - 3 \mathcal { K } ^ { 2 } 1$ . Inserting (3.50) and (λ sinh $X ) { ' } = \lambda$ cosh $X - 1$ ，

$$
4 2 b _ { 2 } = e ^ { \prime \prime } + 6 e + 1 8 \lambda \sinh X , \qquad 3 3 6 b _ { 3 } = e ^ { \prime \prime \prime } + 2 0 e ^ { \prime } + 1 4 4 \lambda \cosh X - 3 1 2 ,
$$

which are the coeficients (2.17) evaluated at $a _ { 4 }$ . This completes the derivation of the finite formula (2.16). In this representation, only $e ( a _ { 4 } )$ requires the quadrature (2.8); the derivatives of e are eliminated by (2.9) and (2.10), the latter being (3.22) and its derivative.

On the interface $z _ { 4 } = 0$ the two coordinate systems coincide, with $a _ { 1 } = 0 .$ , and $\Phi _ { t } ( 0 ) = 1$ remove the factor $\Phi _ { t } ( a _ { 1 } )$ , so (3.48) becomes the three-factor Green’s representation $\mathcal { P } _ { a _ { 2 } , a _ { 3 } , a _ { 4 } } e$ of the interface datum e. The Region II formula (2.15) at $z _ { 4 } = 0$ is the same representation applied to $e = e _ { 4 } + g \colon$ by Section 3.5, its four-expert part is $F _ { 4 }$ and its density part is the integral with $p = g ^ { \prime \prime } - g ,$ , the Laplace factor $e ^ { - 2 z _ { 4 } \coth t }$ being 1. The two formulas therefore have the same value on the interface. Their integrands are diferent, one involving e and the other p and $F _ { 4 } { \mathrm { : } }$ the agreement of their transverse derivatives up to second order is proved in Section 4.1.

## 4 Verification, regularity, and the optimality set of COMB

The derivation of the five expert solution formula given in Section 3 does not prove the formula is the viscosity solution of the expert PDE (2.1). This section addresses this in multiple steps. We first verify in Section 4.1 that the solution is $C ^ { 2 }$ across the sector boundaries, and that it has linear growth (bounded correction from the payof). Then we verify in Sections 4.2, 4.3, and 4.4 that $\mathbf { v } _ { * } = ( 1 , 0 , 1 , 0 , 0 )$ attains the Hamiltonian maximum throughout the whole ordered sector, which allows us to establish the formula as the unique viscosity solution of (2.1) in Section 4.6, and complete our proof of non-optimality of the COMB strategy.

Many of the proofs in this section require tedious algebraic work, and part of it is delegated to a computer. The division of labor is as follows. The arguments that reduce Theorems 2.1 and 2.2 to finite computations are given in full in the text: the face operators and the $C ^ { 2 }$ matching across the interfaces, the symmetry and multiafine reductions from the sector to a finite list of corner cases, the sign principles of Lemmas 4.4 and 4.5, the moment recursion, the integration of diferential sign identities from infinity, and the viscosity-solution argument. What remains after these reductions is finite and of two kinds. The first is a list of exact algebraic identities among explicit hyperbolic, rational, and trace expressions, among them the face conditions, the value and the first two transverse derivatives at the interfaces, and the control tables, which comprise 64 Hessian contractions in Region III and 64 corner gaps in Regions I and II. The second is the sign of 21 explicit scalar functions of one variable, 8 in Region III and 13 in Regions I and II. Each has the form $a ( L ) e ( L ) + b ( L )$ with a rational in $\varrho = e ^ { L }$ and b rational in ϱ and linear in $L$ and $\lambda ( L )$ ; the diferential sign identities of the form (4.12) and (4.13) eliminate the trace $e ,$ and their residuals are then of the rational-log form (4.34).

Both kinds are verified by the computational supplement described in Section 4.5, and the text states at each such point which check of the supplement verifies it. The identities are checked symbolically: after the substitution $\varrho = e ^ { L }$ , and its analogues for the other coordinates, every hyperbolic identity becomes an identity of rational functions, which is decided by exact polynomial arithmetic. Each identity is also derived in reduced form in the text, so any one of them can be verified by hand. The scalar signs are established by exact certificates: the logarithms in (4.34) are enclosed between rational functions on the seven intervals (4.36), and the resulting 147 polynomial lower bounds are shown to be nonnegative through their Bernstein coeficients (4.38), which are exact rational numbers. No floating-point arithmetic, numerical quadrature, or sign sampling is used anywhere in the proof, so there are no rounding errors to control. The trusted software consists of the exact integer and rational arithmetic of $\mathrm { C P y }$ thon and the exact algebra and polynomial root counting of SymPy, and the certificates are finite lists of integers that the supplement rechecks with a separate program using only fraction arithmetic. In this sense the computer-assisted part of the proof is an exact symbolic computation with the same logical status as a long hand calculation, not a formalization in a proof assistant. The supplement is a fixed, versioned release with pinned dependencies that runs ofline in a few minutes; its review guide maps every check to an equation label of this paper and lists the analytic steps that remain for the reader.

## 4.1 Regularity and boundedness

The aim of this section is to prove regularity and boundedness of the solution formula for u.   
In particular, we prove the following result.

Proposition 4.1. Let u be the function of Theorem 2.1: on the ordered sector $u = x _ { 1 } + F ( y ) / k$ as in (2.2), with F given by (2.15) in Regions I and II and by (2.16) in Region III, and u is

extended to $\mathbb { R } ^ { 5 }$ by sorting the coordinates. Then u $\in C ^ { 2 } ( \mathbb { R } ^ { 5 } )$ , satisfies $\nabla u \cdot \mathbf { 1 } = 1$ , and has bounded correction to $\varphi .$ . Furthermore, in the ordered sector, $D _ { \mathbf { v } _ { * } } ^ { 2 } u = 2 ( u - x _ { 1 } )$

The proof of Proposition 4.1 contains many steps that occupy the rest of this section. Throughout we write $d = | z _ { 3 } |$ . In Region I, where $\begin{array} { r } { d = - z _ { 3 } = \frac { 1 } { 2 } ( y _ { 3 } - y _ { 1 } ) } \end{array}$ and $z _ { 4 } = y _ { 4 }$ , and in Region II, where $d = z _ { 3 }$ and $z _ { 4 } = y _ { 4 } - z _ { 3 }$ , the chain rule for the coordinates (2.3)–(2.4) gives

$$
\begin{array} { r l r } & { \mathrm { I } { : } } & { \partial _ { y 1 } = \frac { 1 } { 2 } ( \partial _ { z 1 } + \partial _ { z 2 } - \partial _ { d } ) , \qquad \partial _ { y 3 } = \frac { 1 } { 2 } ( \partial _ { z 1 } + \partial _ { z 2 } + \partial _ { d } ) , } \\ & { \mathrm { I I } { : } } & { \partial _ { y 1 } = \frac { 1 } { 2 } ( \partial _ { z 1 } + \partial _ { z 2 } + \partial _ { d } - \partial _ { z 4 } ) , \partial _ { y 3 } = \frac { 1 } { 2 } ( \partial _ { z 1 } + \partial _ { z 2 } - \partial _ { d } + \partial _ { z 4 } ) , } \end{array}\tag{4.1}
$$

with $\partial _ { y _ { 2 } } = \partial _ { z _ { 1 } }$ and $\partial _ { y _ { 4 } } = \partial _ { z _ { 4 } }$ in both regions, and in Region III, by (2.5),

$$
\begin{array} { l l } { { \partial _ { y 1 } = { \frac { 1 } { 3 } } ( \partial _ { a 1 } + \partial _ { a 2 } + \partial _ { a 3 } + \partial _ { a 4 } ) , } } & { { \partial _ { y 2 } = \partial _ { a _ { 4 } } , } } \\ { { } } & { { } } \\ { { \partial _ { y 3 } = { \frac { 1 } { 3 } } ( - \partial _ { a 1 } - \partial _ { a 2 } + 2 \partial _ { a 3 } + 2 \partial _ { a 4 } ) , } } & { { \partial _ { y 4 } = { \frac { 1 } { 3 } } ( - 2 \partial _ { a 1 } + \partial _ { a 2 } + \partial _ { a 3 } + \partial _ { a 4 } ) . } } \end{array}\tag{4.2}
$$

In particular the direction $b _ { \mathbf { v } _ { * } } = ( 1 , - 1 , 1 , 0 )$ of (3.5) is $\partial _ { y _ { 1 } } - \partial _ { y _ { 2 } } + \partial _ { y _ { 3 } } = \partial _ { z _ { 2 } }$ in Regions I and II and $\partial _ { a _ { 3 } }$ in Region III, and the displayed formulas satisfy the corresponding secondderivative equation immediately: every term of (2.16) has degree one in (cosh a , sinh a ), and in (2.15) both $F _ { 4 }$ and $\Phi _ { t } ( z _ { 2 } )$ satisfy $f ^ { \prime \prime } = f$ in $z _ { 2 }$ . This proves (3.5) in each regional interior. One-variable functions such as $e , p , \theta , \lambda$ , the moments below, and the scalar certificates are written in a generic variable $L > 0$ , which stands for $z _ { 1 }$ in Regions I and II and for $a _ { 4 }$ in Region III. We abbreviate S = sinh L and $C = \cosh L$

The density p enters every estimate below through the following bounds.

Lemma 4.2. For $L > 0$

$$
\frac { 1 } { 1 4 } \operatorname { t a n h } L \operatorname { s e c h } ^ { 2 } L \leq p ( L ) \leq \frac { 1 } { 8 } \operatorname { t a n h } L \operatorname { s e c h } ^ { 2 } L .\tag{4.3}
$$

Moreover $p ( L ) \sim L / 1 4$ at zero and $p ( L ) \sim e ^ { - 2 L } / 2$ at infinity.

Proof. We apply $\partial _ { L } +$ 6 coth L to each of the proposed barriers and subtract the right side ${ \frac { 1 } { 2 } } \operatorname { s e c h } ^ { 2 } L$ of (3.38). For the lower and upper barriers, respectively, this gives

$$
\begin{array} { r } { - \frac { 3 } { 1 4 } \operatorname { t a n h } ^ { 2 } L \operatorname { s e c h } ^ { 2 } L \leq 0 \qquad \mathrm { a n d } \qquad \frac { 3 } { 8 } \operatorname { s e c h } ^ { 4 } L \geq 0 . } \end{array}
$$

Multiplying by the positive integrating factor $\sinh ^ { 6 } L$ and integrating from zero proves the two bounds. The asymptotic statements follow from (2.11). Near zero the integrand is $t ^ { 6 } + \mathcal { O } ( t ^ { 8 } )$ , so

$$
p ( L ) = \frac { L ^ { 7 } / 7 + \mathcal { O } ( L ^ { 9 } ) } { 2 L ^ { 6 } + \mathcal { O } ( L ^ { 8 } ) } \sim \frac { L } { 1 4 } .
$$

At infinity sinh $\mid ^ { 6 } t \mathrm { \ s e c h } ^ { 2 } t \sim e ^ { 4 t } / 1 6$ , so the integral is asymptotic to $e ^ { 4 L } / 6 4$ while $2 \sinh ^ { 6 } L \sim$ $e ^ { 6 L } / 3 2$ , and $p ( L ) \sim e ^ { - 2 L } / 2$ . In particular the lower bound in (4.3) is sharp at zero and the upper bound is sharp at infinity. □

We next verify the four boundary conditions (3.6) on the faces of the ordered sector directly from the formula. Section 3 used these conditions only through the trace systems (3.10)–(3.12), which were solved as Cauchy problems from data on an edge, imposing the remaining conditions only on that edge; it also assumed growth conditions at infinity and smooth traces. Since each equation in (3.6) is the condition $\partial _ { n } u = 0$ on a permutation hyperplane, (3.6) is part of the $C ^ { 2 }$ claim of Theorem 2.1 and must be established for the final formulas. We use (4.1) to express each face operator in the regional coordinates. The four-expert term of (2.15) is $F _ { 4 } ( z _ { 1 } , z _ { 2 } , - d )$ in Region I and $F _ { 4 } ( z _ { 1 } , z _ { 2 } , d )$ in Region II, with $F _ { 4 }$ from (2.14); we write $F _ { 4 } ^ { \mathrm { I } }$ and $F _ { 4 } ^ { \mathrm { I I } }$ for these. The correction in (2.15) is symmetric in $d , z _ { 2 }$ . On $y _ { 1 } = 0$ in Region I, $d = z _ { 2 }$ and $2 \partial _ { y _ { 1 } } - \partial _ { y _ { 2 } } = \partial _ { z _ { 2 } } - \partial _ { d }$ , which annihilates every function symmetric in $( d , z _ { 2 } )$ on the diagonal $d = z _ { 2 }$ . The only asymmetric term of F is $- { \textstyle { \frac { 1 } { 2 } } } \sinh ( z _ { 2 } - d )$ in $F _ { 4 } ^ { \mathrm { I } }$ , and $( \partial _ { z _ { 2 } } - \partial _ { d } ) \bigl ( - { \textstyle \frac { 1 } { 2 } } \sinh ( z _ { 2 } - d ) \bigr ) = - \cosh ( z _ { 2 } - d ) = - 1$ at $d = z _ { 2 }$ which is the required value. On $y _ { 2 } = 0$ in Regions I and II, $z _ { 1 } = z _ { 2 }$ and the face operator $- \partial _ { y _ { 1 } } + 2 \partial _ { y _ { 2 } } - \partial _ { y _ { 3 } }$ is $\partial _ { z _ { 1 } } - \partial _ { z _ { 2 } }$ , the $\partial _ { d }$ and $\partial _ { z _ { 4 } }$ terms cancelling. The background satisfies this identity by its face condition (3.35), and the correction does too by symmetry of $\Phi _ { t } \big ( z _ { 1 } \big ) \Phi _ { t } \big ( z _ { 2 } \big )$ its moving-endpoint term is zero since $\Phi _ { z _ { 1 } } ( z _ { 1 } ) = 0$ . On $y _ { 3 } = 0$ in Region $\operatorname { I I } , d = z _ { 2 }$ and the operator $- \partial _ { y _ { 2 } } + 2 \partial _ { y _ { 3 } } - \partial _ { y _ { 4 } }$ is $\partial _ { z _ { 2 } } - \partial _ { d }$ , so both terms vanish by symmetry.

It remains to check the boundary condition on $y _ { 4 } = 0$ in Region I. Here $z _ { 4 } = 0$ , and by (4.1) the condition $- F _ { y _ { 3 } } + 2 F _ { y _ { 4 } } = 0$ reads

$$
4 F _ { z _ { 4 } } = F _ { z _ { 1 } } + F _ { d } + F _ { z _ { 2 } } , \qquad 0 \le d \le z _ { 2 } \le z _ { 1 } .\tag{4.4}
$$

Put c = coth t, $P = \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( d ) \Phi _ { t } ( z _ { 2 } )$ , and $N _ { t } ( a ) = c \cosh a - \sinh a = - \partial _ { a } \Phi _ { t } ( a )$ . Since $F _ { 4 } ^ { \mathrm { I } }$ does not depend on $z _ { 4 }$ , while in the correction $\partial _ { z _ { 4 } }$ produces the factor $- 2 c$ , the tangential derivatives act on $P$ through $\partial _ { a } \Phi _ { t } ( a ) = - N _ { t } ( a )$ , and the moving lower limit contributes nothing because $\Phi _ { z _ { 1 } } ( z _ { 1 } ) = 0$ , condition (4.4) at $z _ { 4 } = 0$ is equivalent to

$$
\begin{array} { l } { { ( \partial _ { z _ { 1 } } + \partial _ { d } + \partial _ { z _ { 2 } } ) F _ { 4 } ^ { \mathrm { I } } = \displaystyle \int _ { z _ { 1 } } ^ { \infty } \sinh { t p ( t ) } \big [ - 8 c P + N _ { t } ( z _ { 1 } ) \Phi _ { t } ( d ) \Phi _ { t } ( z _ { 2 } ) } } \\ { { { } } } \\ { { \qquad + \Phi _ { t } ( z _ { 1 } ) N _ { t } ( d ) \Phi _ { t } ( z _ { 2 } ) + \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( d ) N _ { t } ( z _ { 2 } ) \big ] d t . } } \end{array}
$$

The bracket is a polynomial in c of degree at most four. We replace $c ^ { j }$ by the moments $M _ { j } ( z _ { 1 } )$ from (4.22)–(4.23), for $0 \le j \le 4$ . Substituting the moment formulas gives the left side exactly; no additional relation between e and $p$ is needed. We verify this finite symbolic identity in checks/faces.py in the proof supplement.

In Region III the face operators on $y _ { 2 } = 0 , y _ { 3 } = 0$ , and $y _ { 4 } = 0$ are, respectively, $\partial _ { a _ { 4 } } - \partial _ { a _ { 3 } }$ at $a _ { 3 } = a _ { 4 } , \partial _ { a _ { 3 } } - \partial _ { a _ { 2 } }$ at $a _ { 2 } = a _ { 3 }$ , and $\partial _ { a _ { 2 } } - \partial _ { a _ { 1 } }$ at $a _ { 1 } = a _ { 2 }$ . The last two identities follow from symmetry of (2.16). The first follows by substituting (2.17) and the trace ODE into the derivative of that finite formula; it is checked in the same audit. These direct checks cover every relatively open physical face. Their intersections follow from the continuity of the value, gradient, and Hessian up to the boundary of the sector, established below.

We now show that the correction to the payof is bounded. This follows directly from the integral representations. The four-expert correction is bounded, and the correction integral in (2.15) is bounded above by $\textstyle \int _ { 0 } ^ { \infty }$ sinh $t p ( t ) d t < \infty$ . In (3.48), $0 \leq A \leq 1$ , every $\Phi _ { t } ( a _ { i } ) \leq 1$ and sinh $a _ { i } \leq$ sinh $a _ { 4 }$ . Also

$$
2 \sinh ^ { 2 } a _ { 4 } \int _ { a _ { 4 } } ^ { \infty } { \frac { d t } { \sinh ^ { 3 } t } } \leq 2 \sinh ^ { 2 } a _ { 4 } \int _ { a _ { 4 } } ^ { \infty } { \frac { \cosh t } { \sinh ^ { 3 } t } } d t = 1 .
$$

There are six pairs in the kernel, so $0 \leq F \leq 7 \| e \| _ { \infty }$ in Region III. In particular $u - \varphi$ is bounded throughout the sector. This is the one place where Section 4 uses Section 3 beyond the definition of $e \mathrm { : }$ the upper bound in Region III rests on the equality of (2.16) with (3.48), established in Section 3.6 by the passage from compactly supported data to $e ,$ and it is not visible from (2.16) alone, whose products $\sigma _ { j }$ grow like $e ^ { a _ { 1 } + a _ { 2 } + a _ { 3 } }$ . The lower bound $F \geq 0$ in Region III also follows from Section 4.2: the gap of the control 00000 is F itself, and its vertex values $e , C _ { 1 } + 3 T _ { 1 } + 4 D _ { 1 } , 3 ( A _ { 2 } + B _ { 2 } + T _ { 2 } )$ , and $3 T _ { 3 } + 6 P _ { 3 }$ in the tables there are nonnegative.

The formula in each region is smooth for $z _ { 1 } > 0$ , respectively $a _ { 4 } > 0$ , up to its relative boundary. We show next that the values and derivatives agree across the two interfaces.

At the first interface keep $( z _ { 1 } , z _ { 2 } , R )$ fixed and vary the signed coordinate $z _ { 3 } = ( y _ { 1 } - y _ { 3 } ) / 2$ The four-expert term (2.14) is already smooth in $z _ { 3 }$ . For a fixed $c = \coth t ,$ , the only changing factor in the correction is $e ^ { - 2 z _ { 4 } c } \Phi _ { t } ( | z _ { 3 } | )$ , where $z _ { 4 } = R$ in Region I and $z _ { 4 } = R - z _ { 3 }$ in Region II, and $\Phi _ { t } ( \mp z _ { 3 } ) = \cosh z _ { 3 } \pm c \sinh z _ { 3 }$ by the addition formula; that is,

$$
e ^ { - 2 R c } \left\{ \begin{array} { l l } { \cosh z _ { 3 } + c \sinh z _ { 3 } , } & { z _ { 3 } \leq 0 , } \\ { e ^ { 2 c z _ { 3 } } ( \cosh z _ { 3 } - c \sinh z _ { 3 } ) , } & { z _ { 3 } \geq 0 . } \end{array} \right.\tag{4.5}
$$

Both branches have value 1, first derivative $c ,$ and second derivative 1 at $z _ { 3 } = 0$ : for the second branch the derivative is $e ^ { 2 c z _ { 3 } }$ -c cosh $z _ { 3 } + ( 1 - 2 c ^ { 2 } )$ sinh $z _ { 3 } \big ]$ , whose value and derivative at zero are c and $2 c ^ { 2 } + ( 1 - 2 c ^ { 2 } ) \stackrel { . } { = } 1$ Since the integration endpoint $z _ { 1 }$ is fixed, this proves equality of the value and the first two transverse derivatives as functions of the tangential variables. These three identities hold at every point of the interface, so they may be diferentiated with respect to $z _ { 1 } , z _ { 2 }$ , and $R ;$ this yields the tangential first derivatives and the tangential–tangential and mixed second derivatives, and hence the value, gradient, and Hessian of the two regional formulas agree at $y _ { 1 } = y _ { 3 }$

For the second interface use the Region III coordinates on both sides and write $\tau = a _ { 1 }$ keeping $( a _ { 2 } , a _ { 3 } , a _ { 4 } )$ fixed. By the relations $a _ { 1 } = - { \frac { 2 } { 3 } } z _ { 4 }$ and $a _ { i } = z _ { 5 - i } + { \textstyle { \frac { 1 } { 3 } } } z _ { 4 }$ recorded after (2.6), the Region II coordinates become

$$
( z _ { 1 } , z _ { 2 } , z _ { 3 } , z _ { 4 } ) = ( a _ { 4 } + \tau / 2 , a _ { 3 } + \tau / 2 , a _ { 2 } + \tau / 2 , - 3 \tau / 2 ) ,\tag{4.6}
$$

so that $\begin{array} { r } { \partial _ { \tau } z _ { i } = \frac { 1 } { 2 } } \end{array}$ for $i \leq 3$ and $\partial _ { \tau } z _ { 4 } = - { \frac { 3 } { 2 } }$ . With $s _ { i } = \sinh a _ { i } , c _ { i } = \cosh a _ { i }$ , define

$$
\begin{array} { r } { B _ { 0 } = b _ { 0 } c _ { 2 } c _ { 3 } + b _ { 1 } ( s _ { 2 } c _ { 3 } + c _ { 2 } s _ { 3 } ) + b _ { 2 } s _ { 2 } s _ { 3 } , } \\ { B _ { 1 } = b _ { 1 } c _ { 2 } c _ { 3 } + b _ { 2 } ( s _ { 2 } c _ { 3 } + c _ { 2 } s _ { 3 } ) + b _ { 3 } s _ { 2 } s _ { 3 } . } \end{array}
$$

Formula (2.16) is $F = \cosh \tau B _ { 0 } + \sinh \tau B _ { 1 }$ on the Region III side. On the Region II side, direct diferentiation gives

$$
( F , F _ { \tau } , F _ { \tau \tau } ) | _ { \tau = 0 ^ { - } } = ( B _ { 0 } , B _ { 1 } , B _ { 0 } ) .\tag{4.7}
$$

To verify this identity, we must account for the moving endpoint of the integral. We set $P = \Phi _ { t } ( a _ { 4 } ) \Phi _ { t } ( a _ { 2 } ) \Phi _ { t } ( a _ { 3 } )$ and

$$
\begin{array} { r } { Q _ { 0 } = \frac { 1 } { 2 } ( \partial _ { a _ { 4 } } + \partial _ { a _ { 2 } } + \partial _ { a _ { 3 } } ) , \qquad Q _ { c } = Q _ { 0 } + 3 c . } \end{array}
$$

By $( 4 . 6 ) , \partial _ { \tau }$ acts as $Q _ { 0 }$ on $F _ { 4 } ( z _ { 1 } , z _ { 2 } , z _ { 3 } )$ . The integrand of (2.15) is sinh $t p ( t ) e ^ { - 2 z _ { 4 } c }$ times $\Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( z _ { 3 } ) \Phi _ { t } ( z _ { 2 } )$ , and since $\partial _ { \tau } e ^ { - 2 z _ { 4 } c } = 3 c e ^ { - 2 z _ { 4 } c }$ , the operator $\partial _ { \tau }$ acts on it as $Q _ { c }$ . The lower limit $z _ { 1 } = a _ { 4 } + \tau / 2$ moves as well. Its first derivative contributes nothing, because the integrand vanishes at $t = z _ { 1 } ;$ its second derivative contributes $- \frac { 1 } { 2 } \partial _ { \tau } ( \mathrm { i n t e g r a n d } ) | _ { t = z _ { 1 } }$ , where only the factor $\Phi _ { t } ( z _ { 1 } )$ has a nonzero derivative, namely $\partial _ { \tau } \Phi _ { t } ( z _ { 1 } ) = { \textstyle \frac { 1 } { 2 } } \Phi _ { t } ^ { \prime } ( z _ { 1 } )$ , which equals $- 1 / ( 2 \sinh t )$ at $t = z _ { 1 }$ . For $j = 0 , 1 , 2$ the left side of (4.7) is therefore

$$
Q _ { 0 } ^ { j } F _ { 4 } + \int _ { a _ { 4 } } ^ { \infty } \sinh t p ( t ) Q _ { c } ^ { j } P d t + \mathbb { 1 } _ { \{ j = 2 \} } \frac { p ( a _ { 4 } ) } { 4 } \Phi _ { a _ { 4 } } ( a _ { 2 } ) \Phi _ { a _ { 4 } } ( a _ { 3 } ) ,\tag{4.8}
$$

the last term being $\left( - { \frac { 1 } { 2 } } \right)$ · sinh $\begin{array} { r } { a _ { 4 } p ( a _ { 4 } ) \cdot ( - \frac { 1 } { 2 \sinh a _ { 4 } } ) \Phi _ { a _ { 4 } } ( a _ { 2 } ) \Phi _ { a _ { 4 } } ( a _ { 3 } ) } \end{array}$ at $\tau = 0$ , where $z _ { 4 } = 0$ Expand $Q _ { c } ^ { j } P$ as a polynomial in c and use the moments (4.22)–(4.23) below. Substitution of (2.12) and (2.10) gives respectively $B _ { 0 } , B _ { 1 } , B _ { 0 }$ , coeficient by coeficient in c<sub>2</sub>c<sub>3</sub>, s<sub>2</sub>c<sub>3</sub>, c<sub>2</sub>s<sub>3</sub>, s<sub>2</sub>s<sub>3</sub>. The symbolic audit also checks this identity directly. The value and first two transverse derivatives agree as functions of all three tangential variables, and tangential diferentiation, as at the first interface, gives equality of the value, gradient, and Hessian of the Region II and Region III formulas on the second interface.

It remains to show that the derivatives extend continuously to $z _ { 1 } = 0$ in Regions I and II and to $a _ { 4 } = 0$ in Region III. In Regions I and II, Lemma 4.2 gives

$$
\sinh t p ( t ) = \mathcal { O } ( t ^ { 2 } ) \quad ( t \downarrow 0 ) , \qquad \sinh t p ( t ) = \mathcal { O } ( e ^ { - t } ) \quad ( t \to \infty ) .
$$

For $0 \leq a \leq z _ { 1 } \leq t$ , we have $0 \leq \Phi _ { t } ( a ) \leq 1$ and $| \partial _ { a } \Phi _ { t } ( a ) | \leq$ coth t. Every derivative of order at most two of the integrand in (2.15) is therefore bounded by

$$
C _ { 0 } \sinh t p ( t ) ( 1 + \coth ^ { 2 } t ) ,
$$

uniformly on the physical domain, since $z _ { 4 } ~ \ge ~ 0$ . This is integrable both at zero and infinity. The first endpoint term vanishes because $\begin{array} { r } { \Phi _ { z _ { 1 } } ( z _ { 1 } ) = 0 ; } \end{array}$ the second endpoint terms are $\mathcal { O } ( p ( z _ { 1 } ) ) = \mathcal { O } ( z _ { 1 } )$ . Dominated convergence proves continuous limits for the function, gradient, and Hessian at $z _ { 1 } = 0$ , including the top-four collision with arbitrary $y _ { 4 } \geq 0$ . The four-expert term has the same property: its only nonanalytic factor at zero is $\lambda ( z _ { 1 } )$ sinh $z _ { 1 }$ sinh $z _ { 2 }$ sinh $z _ { 3 }$ whose second derivatives tend to zero.

In Region III the trace e is analytic at zero. The nonanalytic part of (2.16) is a linear combination of

$$
\lambda ( a _ { 4 } ) \sinh a _ { 4 } \sum _ { i < j } s _ { i } s _ { j } c _ { h } , \qquad \lambda ( a _ { 4 } ) \cosh a _ { 4 } s _ { 1 } s _ { 2 } s _ { 3 } ,
$$

where h is the remaining index. On $0 \leq a _ { i } \leq a _ { 4 }$ these terms are $\mathcal { O } ( a _ { 4 } ^ { 3 } | \log { a _ { 4 } } | )$ and all their second partial derivatives are $\mathcal { O } ( a _ { 4 } | \log a _ { 4 } | )$ . The remaining terms are analytic. The value, gradient, and Hessian therefore have unique limits at the origin in each region. The interface identities prove that those limits agree: approach the origin along either common interface, where the value, gradient, and Hessian of the two regional formulas have already been shown to agree.

To pass from these one-sided identities to $C ^ { 2 }$ regularity we use the following gluing lemma, whose proof is standard and given in Appendix A.

Lemma 4.3. Let $\Omega \subset \mathbb { R } ^ { n }$ be open and convex, let $K _ { 1 } , \ldots , K _ { N }$ be closed convex sets with nonempty interiors whose union contains Ω, and let $u \colon \Omega  \mathbb { R }$ be continuous. Suppose that for each j the restriction of u to the intersection of Ω with the interior of $K _ { j }$ is $C ^ { 2 }$ , that its gradient and Hessian extend continuously to $\Omega \cap K _ { j }$ , and that at every point $o f \Omega$ the extended gradients and Hessians of all the $K _ { j }$ containing that point agree. Then $u \in C ^ { 2 } ( \Omega )$

We first apply the lemma inside the sector. Let S be the closed ordered sector in $\mathbb { R } ^ { 5 }$ write u for the function defined on S by the formula of Theorem 2.1, and take Ω to be the interior of $S _ { i }$ , where all $y _ { i } > 0$ , and $K _ { 1 } , K _ { 2 } , K _ { 3 }$ the closures of Regions I, II, and III, which are closed convex cones. Inside each region the formula is smooth, as noted before the interface computations, and its gradient and Hessian extend continuously to the closed region: in Regions I and II by the dominated convergence bound just established, which is uniform on the physical domain, together with the $C ^ { 2 }$ regularity of $F _ { 4 }$ , and in Region III because (2.16) is analytic for $a _ { 4 } > 0$ and has the limits at the origin found above. A point of the open sector lying in two regions lies on an interface, and a point lying in all three lies on both interfaces, at $z _ { 3 } = z _ { 4 } = 0 $ ; the interface identities give the agreement of the extended gradients and Hessians at such points, and the agreement of the values makes u continuous. Lemma 4.3 therefore shows that u is $C ^ { 2 }$ in the interior of S. Moreover ∇u and $\nabla ^ { 2 } \overline { { u } }$ extend continuously to S: at boundary points with $z _ { 1 } > 0$ by the extensions from the closed regions, and at $z _ { 1 } = 0$ and at the origin by the limits established above, where the extensions from diferent regions agree by continuity, since they agree on the interfaces.

We now extend u to $\mathbb { R } ^ { 5 }$ by permutation invariance. For $x \in \mathbb { R } ^ { 5 }$ let $x ^ { \downarrow }$ be the vector obtained by sorting the coordinates of x in decreasing order, and set $u ( x ) = \overline { { u } } ( x ^ { \downarrow } )$ . This is well defined and continuous, since the kth coordinate of $x ^ { \downarrow }$ , the kth largest coordinate of $x ,$ is a continuous function of x. Let each permutation σ of $\{ 1 , \ldots , 5 \}$ act on $\mathbb { R } ^ { 5 }$ by permuting coordinates, and write $\sigma$ also for the corresponding orthogonal matrix. The sectors $\sigma ( S )$ are closed convex cones with nonempty interiors covering $\mathbb { R } ^ { 5 }$ , and on $\sigma ( S )$ we have $u = \overline { { u } } \circ \sigma ^ { - 1 }$ whose gradient and Hessian are $\sigma \nabla \overline { { u } } \circ \sigma ^ { - 1 }$ and $\sigma \nabla ^ { 2 } \overline { { u } } \circ \sigma ^ { - 1 } \sigma ^ { T }$ in the interior and extend continuously to $\sigma ( S )$ . To apply Lemma 4.3 with $\Omega = \mathbb { R } ^ { 5 }$ it remains to check the agreement hypothesis, and by permutation invariance it sufices to do so at points $x \in S$ . The sectors containing x are exactly the $\sigma ( S )$ with $\sigma x = x \colon$ indeed $x \in \sigma ( S )$ means that $\sigma ^ { - 1 } \boldsymbol { x }$ is sorted in decreasing order, that is, $\sigma ^ { - 1 } x = x ^ { \downarrow } = x$ . On such a sector the extended gradient and Hessian at x are $\sigma \nabla \overline { { u } } ( x )$ and $\sigma \nabla ^ { 2 } \overline { { u } } ( x ) \sigma ^ { T }$ , so the agreement hypothesis at x states that $\nabla \overline { { u } } ( \boldsymbol { x } )$ and $\nabla ^ { 2 } { \overline { { u } } } ( x )$ are invariant under every σ with $\sigma x = x$ . These permutations preserve each block of tied coordinates of $x _ { i }$ and they are generated by the transpositions $\sigma _ { i }$ of i and $i + 1$ over the indices i with $x _ { i } = x _ { i + 1 }$ , so it sufices to check invariance under these. Fix such an i and let $\nu \in \mathbb { R } ^ { 5 }$ be the unit vector with $\nu _ { i } = 1 / \sqrt { 2 } , \nu _ { i + 1 } = - 1 / \sqrt { 2 } .$ and all other components zero, so that $\sigma _ { i } = \operatorname { I d } - 2 \nu \nu ^ { T }$ is the reflection across the hyperplane $x _ { i } = x _ { i + 1 }$ The face condition in (3.6) on $y _ { i } = 0$ is $u _ { x _ { i } } = u _ { x _ { i + 1 } }$ , that is, $\nu \cdot \nabla \overline { { u } } = 0$ . It was verified on the relatively open face of S in this hyperplane, where all the other inequalities are strict, and it holds on the closed face by continuity of $\nabla \overline { { u } }$ . The relatively open face is an open subset of the hyperplane, so the identity may be diferentiated there in every direction t orthogonal to $\nu ,$ giving $t ^ { T } \nabla ^ { 2 } \overline { { u } } \nu = 0$ , which again extends to the closed face by continuity. Thus at x the vector ν is orthogonal to $\nabla \overline { { u } } ( \boldsymbol { x } )$ and is an eigenvector of $\nabla ^ { 2 } { \overline { { u } } } ( x )$ , say $\nabla ^ { 2 } \overline { { u } } ( x ) \nu = \gamma \nu$ . Hence $\sigma _ { i } \nabla \overline { { u } } ( x ) = \nabla \overline { { u } } ( x ) - 2 \big ( \nu \cdot \nabla \overline { { u } } ( x ) \big ) \nu = \nabla \overline { { u } } ( x )$ and

$$
\begin{array} { r } { \sigma _ { i } \nabla ^ { 2 } \overline { { u } } ( x ) \sigma _ { i } = \nabla ^ { 2 } \overline { { u } } ( x ) - 2 \nu \nu ^ { T } \nabla ^ { 2 } \overline { { u } } ( x ) - 2 \nabla ^ { 2 } \overline { { u } } ( x ) \nu \nu ^ { T } + 4 \nu \nu ^ { T } \nabla ^ { 2 } \overline { { u } } ( x ) \nu \nu ^ { T } = \nabla ^ { 2 } \overline { { u } } ( x ) , } \end{array}
$$

since the last three terms equal − ${ } ^ { - 2 \gamma \nu \nu ^ { T } , - 2 \gamma \nu \nu ^ { T } }$ , and $4 \gamma \nu \nu ^ { T }$ . This verifies the agreement hypothesis, and Lemma 4.3 gives $u \in C ^ { 2 } ( \mathbb { R } ^ { 5 } )$ . This completes the proof of Proposition 4.1.

## 4.2 The Hamiltonian inequalities in Region III

It remains to prove $D _ { \mathbf { v } } ^ { 2 } u \leq D _ { \mathbf { v } _ { \tau } } ^ { 2 }$ u for every binary control. We treat Region III first, because the finite formula (2.16) is afine in each tanh $a _ { i } , \ i \leq 3$ , so the inequalities there reduce by algebra alone to finitely many scalar inequalities in L. In Regions I and II the fifth expert enters (2.15) through the factor $e ^ { - }$ <sup>−2z4</sup> <sup>coth</sup> <sup>t</sup> inside the integral, and each gap is a Laplace transform in $2 z _ { 4 } ;$ the reduction to scalar inequalities then needs the sign principles of Section 4.3.

Translation invariance gives $\nabla ^ { 2 } u \mathbb { 1 } = 0$ , so complementary controls have equal curvature; we use the sixteen representatives with $v _ { 1 } = 0$ . In the coordinates (2.5), define

$$
q _ { \mathbf { v } } = { \frac { 1 } { 3 } } \left( { \begin{array} { c c c c } { 1 } & { 0 } & { - 1 } & { - 2 } \\ { 1 } & { 0 } & { - 1 } & { 1 } \\ { 1 } & { 0 } & { 2 } & { 1 } \\ { 1 } & { 3 } & { 2 } & { 1 } \end{array} } \right) b _ { \mathbf { v } } ,\tag{4.9}
$$

where the matrix is the Jacobian $\partial a / \partial y$ of (2.5), so that $b _ { \mathbf { v } } ^ { T } \nabla _ { y } ^ { 2 } F b _ { \mathbf { v } } = q _ { \mathbf { v } } ^ { T } \nabla _ { a } ^ { 2 } F q _ { \mathbf { v } }$ in (3.4); the direction $b _ { \mathbf { v } _ { * } } = ( 1 , - 1 , 1 , 0 )$ is sent to $( 0 , 0 , 1 , 0 )$ , and $D _ { { \bf v } _ { \star } } ^ { 2 } \dot { u } = ( 2 / k ) F$ by (3.5). The physical curvature gap $D _ { \mathbf { v } _ { * } } ^ { 2 } u - D _ { \mathbf { v } } ^ { 2 } u$ is therefore $( 2 / k ) ( F - q _ { \mathbf { v } } ^ { T } \nabla _ { a } ^ { 2 } F q _ { \mathbf { v } } )$ . Every term of (2.16) has degree one in each pair (cosh $a _ { i } .$ sinh $a _ { i } ) , i \leq 3$ , and diferentiation preserves this, so after division by $\textstyle \prod _ { i = 1 } ^ { 3 }$ cosh $a _ { i }$ the gap is afine separately in each of tanh $a _ { i }$ . Recall that $L$ denotes $a _ { 4 }$ in Region III, and put $\xi _ { i } = \operatorname { t a n h } a _ { i }$ for $i \leq 3$ . The closed Region III is then the simplex $0 \leq \xi _ { 1 } \leq \xi _ { 2 } \leq \xi _ { 3 } \leq$ tanh $L$ , and the normalized gap is afine in each $\xi _ { i }$ separately. A function afine in each variable separately on the cube [0, tanh $L ] ^ { 3 }$ is a convex combination of its values at the eight vertices of the cube, so it sufices to prove that the normalized gap is nonnegative at these vertices, where every $a _ { i } \in \{ 0 , L \}$ . Permutations of $a _ { 1 } , a _ { 2 } , a _ { 3 }$ leave (2.16) invariant and permute $v _ { 3 } , v _ { 4 } , v _ { 5 }$ , since they correspond to permutations of experts 3, 4, and $5 ;$ they map the gap for one control at a vertex to the gap for the permuted control at the permuted vertex. It therefore sufices to consider, for all sixteen controls, the four vertices

$$
\omega _ { 0 } = ( 0 , 0 , 0 ) , \qquad \omega _ { 1 } = ( 0 , 0 , L ) , \qquad \omega _ { 2 } = ( 0 , L , L ) , \qquad \omega _ { 3 } = ( L , L , L )
$$

of the simplex, where $\omega _ { j }$ has $j$ coordinates equal to L. At $( a _ { 1 } , a _ { 2 } , a _ { 3 } ) = \omega _ { j }$ the normalized gap is a function of $a _ { 4 } = L$ alone, through the coeficients (2.17) and the derivatives in the direction $a _ { 4 } ;$ we write

$$
G _ { j } ^ { \mathbf { v } } ( L ) = \frac { F - q _ { \mathbf { v } } ^ { T } \nabla _ { a } ^ { 2 } F q _ { \mathbf { v } } } { \cosh a _ { 1 } \cosh a _ { 2 } \cosh a _ { 3 } } \bigg | _ { ( a _ { 1 } , a _ { 2 } , a _ { 3 } ) = \omega _ { j } } , \qquad j = 0 , 1 , 2 , 3 .\tag{4.10}
$$

For $j = 0$ , we obtain the following nonzero expressions. The gap is zero for every control omitted from the table.

<table><tr><td>Controls  $G _ { 0 } ^ { \mathbf { v } }$ </td></tr><tr><td>00000 e</td></tr><tr><td>00001,00010,00100  $e / 3 + 2 \eta / 2 1 + 2 S \lambda / 7$ </td></tr><tr><td>00011,00101,00110  $( 3 \eta + 2 S \lambda ) / 7$ </td></tr><tr><td>00111 η</td></tr><tr><td>01000,01111  $( 7 e + 5 \eta - 6 S \lambda ) / 2 1$ </td></tr></table>

Only $V = 7 e + 5 \eta - 6 S \lambda$ needs more than positivity of the displayed factors. Diferentiate with (2.9) for $e ^ { \prime } { \mathrm { . } }$ , (3.24) for $\eta ^ { \prime } .$ , and $( S \lambda ) ^ { \prime } = C \lambda - 1$ ; then eliminate I through $\eta = 3$ coth $L -$ $1 5 I / \sinh ^ { 6 } L$ and η through $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ tanh $L - \eta$ . This gives

$$
\begin{array} { r } { V ^ { \prime } - \operatorname { t a n h } L V = \frac { 9 } { 5 } \operatorname { s e c h } ^ { 2 } L - 6 \operatorname { s e c h } L \lambda + ( 3 0 \coth L + \frac { 1 8 } { 5 } \operatorname { t a n h } L ) p . } \end{array}
$$

Using the upper bound $p \ \leq$ tanh $L \mathrm { s e c h } ^ { 2 } L / 8$ of Lemma 4.2 in the last term and $\lambda \ : =$ atanh(sech $L ) \geq$ sech L in the second,

$$
\begin{array} { r } { V ^ { \prime } - \operatorname { t a n h } L V \leq \left( \frac { 9 } { 5 } + \frac { 1 5 } { 4 } - 6 \right) \mathrm { s e c h } ^ { 2 } L + \frac { 9 } { 2 0 } \operatorname { t a n h } ^ { 2 } L \mathrm { s e c h } ^ { 2 } L = - \frac { 9 } { 2 0 } \mathrm { s e c h } ^ { 4 } L < 0 , } \end{array}
$$

since $\textstyle { \frac { 9 } { 5 } } + { \frac { 1 5 } { 4 } } - 6 = - { \frac { 9 } { 2 0 } }$ and tan $1 ^ { 2 } L = 1 - \mathrm { s e c h } ^ { 2 } L$ . Since $V /$ cosh $L \to 0$ , integration from infinity proves $V \geq ( 9 / 2 0 )$ cosh $\begin{array} { r } { L \int _ { L } ^ { \infty } \operatorname { s e c h } ^ { 5 } t d t > 0 } \end{array}$

For the other vertices define eight generators:

$$
\begin{array} { r l r } & { C _ { 1 } = G _ { 1 } ^ { 0 0 0 1 1 } , } & { D _ { 1 } = G _ { 1 } ^ { 0 0 1 0 1 } , } & { T _ { 1 } = G _ { 1 } ^ { 0 1 1 1 1 } , } \\ & { A _ { 2 } = G _ { 2 } ^ { 0 0 0 1 } , } & { B _ { 2 } = G _ { 2 } ^ { 0 0 1 1 0 } , } & { T _ { 2 } = G _ { 2 } ^ { 0 1 1 1 1 } , } \\ & { P _ { 3 } = G _ { 3 } ^ { 0 0 0 1 1 } , } & { T _ { 3 } = G _ { 3 } ^ { 0 1 1 1 1 } . } \end{array}\tag{4.11}
$$

The following table lists all controls. We obtain each entry by diferentiating (2.16) and using (3.22); all entries are exact identities.
<table><tr><td>V</td><td>GY G2</td><td></td><td> $G _ { 3 } ^ { \mathbf { v } }$ </td></tr><tr><td>00000</td><td> $C _ { 1 } + 3 T _ { 1 } + 4 D _ { 1 }$ </td><td> $3 ( A _ { 2 } + B _ { 2 } + T _ { 2 } )$ </td><td> $3 T _ { 3 } + 6 P _ { 3 }$ </td></tr><tr><td>00001</td><td> $C _ { 1 } + T _ { 1 } + 2 D _ { 1 }$ </td><td> $3 A _ { 2 } + T _ { 2 }$ </td><td> $T _ { 3 } + 3 P _ { 3 }$ </td></tr><tr><td>00010</td><td> $C _ { 1 } + T _ { 1 } + 2 D _ { 1 }$ </td><td> $A _ { 2 } + 2 B _ { 2 } + T _ { 2 }$ </td><td> $T _ { 3 } + 3 P _ { 3 }$ </td></tr><tr><td>00011</td><td> $C _ { 1 }$ </td><td> $A _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>00100</td><td> $T _ { 1 } + 2 D _ { 1 }$ </td><td> $A _ { 2 } + 2 B _ { 2 } + T _ { 2 }$ </td><td> $T _ { 3 } + 3 P _ { 3 }$ </td></tr><tr><td>00101</td><td> $D _ { 1 }$ </td><td> $A _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>00110</td><td> $D _ { 1 }$ </td><td> $B _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>00111</td><td> $0$ </td><td>0</td><td>0</td></tr><tr><td>01000</td><td> $T _ { 1 } + 2 D _ { 1 }$ </td><td> $A _ { 2 } + 2 B _ { 2 } + T _ { 2 }$ </td><td> $T _ { 3 } + 3 P _ { 3 }$ </td></tr><tr><td>01001</td><td> $D _ { 1 }$ </td><td> $A _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>01010</td><td> $D _ { 1 }$ </td><td> $B _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>01011</td><td>0</td><td>0</td><td>0</td></tr><tr><td>01100</td><td>0</td><td> $B _ { 2 }$ </td><td> $P _ { 3 }$ </td></tr><tr><td>01101</td><td>0</td><td>0</td><td>0</td></tr><tr><td>01110</td><td> $0$ </td><td> $0$ </td><td>0</td></tr><tr><td>01111</td><td> $T _ { 1 }$ </td><td> $T _ { 2 }$ </td><td> $T _ { 3 }$ </td></tr></table>

It therefore sufices to prove nonnegativity of the eight generators.

Substitute $( 2 . 9 )$ into a generator and write $G = a ( L ) e ( L ) + b ( L )$ . The coeficient a is rational in $\varrho = e ^ { L } ;$ b is rational in $\varrho$ and linear in $L , \lambda .$ For $h = a$ cosh L the trace cancels from

$$
R = \left( \frac { G } { h } \right) ^ { \prime } = \left( \frac { b } { a \cosh L } \right) ^ { \prime } - \frac { 3 I } { \cosh ^ { 2 } L \sinh ^ { 5 } L } ,\tag{4.12}
$$

because $G / h = e /$ cosh $L + b / ( a \cosh L )$ and, by (2.9),

$$
\left( \frac { e } { \cosh L } \right) ^ { \prime } = \frac { e ^ { \prime } - \operatorname { t a n h } L e } { \cosh L } = - \frac { 3 I } { \cosh ^ { 2 } L \sinh ^ { 5 } L } .
$$

For six generators, a has a fixed sign and the certified sign of R is the opposite sign:

<table><tr><td>Generator</td><td> $a ( L )$  , with  $\varrho = e ^ { L }$ </td><td>Sign of R</td></tr><tr><td> $C _ { 1 }$ </td><td> $- 5 ( \varrho ^ { 2 } - 1 ) ^ { 2 } / [ 8 ( \varrho ^ { 2 } + 1 ) ^ { 2 } ]$ </td><td> $R \geq 0$ </td></tr><tr><td> $D _ { 1 }$ </td><td> $5 ( \varrho ^ { 2 } - 1 ) ^ { 2 } / [ 1 \dot { 2 } ( \varrho ^ { 2 } + 1 ) ^ { 2 } ] ^ { }$ </td><td> $R \leq 0$ </td></tr><tr><td> $T _ { 1 }$ </td><td> $( \varrho ^ { 4 } + 3 0 \varrho ^ { 2 } + 1 ) / [ 2 4 ( \varrho ^ { 2 } + 1 ) ^ { 2 } ]$ </td><td> $R \leq 0$ </td></tr><tr><td> $A _ { 2 }$ </td><td> $5 ( \varrho ^ { 2 } - 1 ) ^ { 2 } / [ 2 4 ( \varrho ^ { 2 } + 1 ) ^ { 2 } ]$ </td><td> $R \leq 0$ </td></tr><tr><td> $B _ { 2 }$ </td><td> $5 ( \varrho ^ { 2 } - 1 ) ^ { 2 } / [ 6 ( \varrho ^ { 2 } + 1 ) ^ { 2 } ]$ </td><td> $R \leq 0$ </td></tr><tr><td> $P _ { 3 }$ </td><td> $5 ( \varrho ^ { 2 } - 1 ) ^ { 2 } ( \mathrm { i } 1 \varrho ^ { 4 } + 1 8 \varrho ^ { 2 } + 1 1 ) / [ 4 8 ( \varrho ^ { 2 } + 1 ) ^ { 4 } ]$ </td><td> $R \leq 0$ </td></tr></table>

The exact sign procedure is given in Section 4.5. Here b grows at most linearly at infinity (in fact it has a finite limit, as the audit checks), a tends to a nonzero constant, and $e \to 1 / 2 ,$ , so $G / h  0$ . Integrating the certified derivative sign from infinity proves $G \geq 0$ in all six cases: if $a > 0$ and $R \leq 0$ , then $G / h$ decreases to zero and is therefore nonnegative; if $a < 0$ and $R \geq 0 .$ , then $G / h$ increases to zero and is nonpositive, and $h < 0$ again gives $G \geq 0$

For $T _ { j } , j = 2 , 3$ , a changes sign. Instead use

$$
R _ { j } = T _ { j } ^ { \prime \prime } - \frac { h _ { j } ^ { \prime \prime } } { h _ { j } ^ { \prime } } T _ { j } ^ { \prime } \ge 0 , \qquad h _ { j } = a _ { j } \cosh L .\tag{4.13}
$$

This operator also eliminates the trace: writing $T _ { j } = h _ { j } E + b _ { j }$ with ${ \boldsymbol { E } } = { \boldsymbol { e } } /$ cosh $L ,$ the terms containing E itself cancel in $R _ { j }$ , and only $E ^ { \prime } = - 3 I / ( \cosh ^ { 2 } L \sinh ^ { 5 } L )$ and $E ^ { \prime \prime }$ remain, both explicit by (2.9). The same device is used in (4.33) below. The exact derivatives are

$$
\begin{array} { r l } & { h _ { 2 } ^ { \prime } = - \frac { ( \varrho ^ { 2 } - 1 ) \left( 1 3 \varrho ^ { 4 } + 1 1 0 \varrho ^ { 2 } + 1 3 \right) } { 4 8 \varrho ( \varrho ^ { 2 } + 1 ) ^ { 2 } } < 0 , } \\ & { h _ { 3 } ^ { \prime } = - \frac { ( \varrho ^ { 2 } - 1 ) \left( 7 7 \varrho ^ { 8 } + 7 1 6 \varrho ^ { 6 } + 8 4 6 \varrho ^ { 4 } + 7 1 6 \varrho ^ { 2 } + 7 7 \right) } { 9 6 \varrho ( \varrho ^ { 2 } + 1 ) ^ { 4 } } < 0 . } \end{array}
$$

Thus the operator has no interior singularity. The explicit formulas give $T _ { j } \to 0 , T _ { i } ^ { \prime }$ bounded, and $h _ { i } ^ { \prime } \to - \infty$ . Since $( T _ { j } ^ { \prime } / h _ { j } ^ { \prime } ) ^ { \prime } = R _ { j } / h _ { j } ^ { \prime } \leq 0$ and $T _ { j } ^ { \prime } / h _ { j } ^ { \prime }  0$ , we obtain $T _ { j } ^ { \prime } / h _ { j } ^ { \prime } \stackrel { \cdot } { = } 0$ , hence $T _ { j } ^ { \prime } \leq \dot { 0 }$ and $T _ { j } \geq 0$ . This proves every Region III comparison once the eight scalar signs are certified.

## 4.3 Regions I and II: corner gaps and sign principles

For these regions let $q = M b _ { \mathbf { v } }$ in the coordinates $( z _ { 1 } , d , z _ { 2 } , z _ { 4 } )$ , where M is the Jacobian $\partial ( z _ { 1 } , d , z _ { 2 } , z _ { 4 } ) / \partial y$ of the regional coordinates, transposed from (4.1):

$$
M _ { \mathrm { I } } = \left( \begin{array} { c c c c } { { 1 / 2 } } & { { 1 } } & { { 1 / 2 } } & { { 0 } } \\ { { - 1 / 2 } } & { { 0 } } & { { 1 / 2 } } & { { 0 } } \\ { { 1 / 2 } } & { { 0 } } & { { 1 / 2 } } & { { 0 } } \\ { { 0 } } & { { 0 } } & { { 0 } } & { { 1 } } \end{array} \right) , \qquad M _ { \mathrm { I I } } = \left( \begin{array} { c c c c } { { 1 / 2 } } & { { 1 } } & { { 1 / 2 } } & { { 0 } } \\ { { 1 / 2 } } & { { 0 } } & { { - 1 / 2 } } & { { 0 } } \\ { { 1 / 2 } } & { { 0 } } & { { 1 / 2 } } & { { 0 } } \\ { { - 1 / 2 } } & { { 0 } } & { { 1 / 2 } } & { { 1 } } \end{array} \right) .\tag{4.14}
$$

Write $\mathcal { G } _ { \mathbf { v } } = F - q ^ { T } \nabla ^ { 2 } F q$ . After division by cosh d cosh $z _ { 2 }$ , this is multiafine in tanh d and tanh $z _ { 2 } .$ , including the moving-endpoint term, which is bilinear in those variables. Enlarge the domain to the square $0 \leq d , z _ { 2 } \leq z _ { 1 }$ . In Region II the formula is symmetric in $d , z _ { 2 }$ , and interchanging experts 3 and 4 gives the corresponding control permutation. In Region I, interchanging experts 1 and 2 instead gives

$$
\mathcal { G } _ { \mathbf { v } } ( z _ { 1 } , 0 ) - \mathcal { G } _ { \mathbf { v } ^ { \prime } } ( 0 , z _ { 1 } ) = \sinh z _ { 1 } \left[ 1 - ( v _ { 1 } - v _ { 2 } ) ^ { 2 } \right] \geq 0 ,\tag{4.15}
$$

where $\mathbf { v } ^ { \prime }$ is v with $v _ { 1 } , v _ { 2 }$ interchanged. Indeed, this interchange reverses the sign of $y _ { 1 }$ which exchanges d and $z _ { 2 }$ and sends $b _ { \mathbf { v } }$ to $b _ { \mathbf { v } ^ { \prime } }$ , so the part of F symmetric in $( d , z _ { 2 } )$ namely the integral and all of $F _ { 4 } ^ { \mathrm { I } }$ except $F ^ { \mathrm { a s } } = - { \textstyle \frac { 1 } { 2 } } \sinh ( z _ { 2 } - d )$ , contributes equally to the two sides. For the antisymmetric part, $q ^ { T } \nabla ^ { 2 } \bar { F ^ { \mathrm { a s } } } q = - \textstyle \frac { 1 } { 2 } \sinh ( z _ { 2 } - d ) ( q _ { z _ { 2 } } - q _ { d } ) ^ { 2 }$ with $q _ { z _ { 2 } } - q _ { d } = v _ { 1 } - v _ { 2 }$ by (4.14), so $\begin{array} { r } { F ^ { \mathrm { a s } } - q ^ { T } \nabla ^ { 2 } F ^ { \mathrm { a s } } q = - \frac { 1 } { 2 } \sinh ( \bar { z } _ { 2 } - d ) [ 1 - ( v _ { 1 } - v _ { 2 } ) ^ { 2 } ] } \end{array}$ , which equals $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ sinh $z _ { 1 } [ 1 - ( v _ { 1 } - v _ { 2 } ) ^ { 2 } ]$ at $( d , z _ { 2 } ) = ( z _ { 1 } , 0 )$ and its negative at $( 0 , z _ { 1 } )$ . Consequently only $( \bar { d , } z _ { 2 } ) = ( 0 , 0 ) , ( 0 , z _ { 1 } ) , ( z _ { 1 } , z _ { 1 } )$ need be checked. The first two are shared by Regions I and II. The four distinct families are

<table><tr><td></td><td>Family Scaled gaps (y1, y2, y3, y4) n</td><td></td></tr><tr><td> $V _ { 0 }$ </td><td> $( 0 , z _ { 1 } , 0 , z _ { 4 } )$ </td><td>0</td></tr><tr><td> $V _ { 1 }$ </td><td> $( z _ { 1 } , 0 , z _ { 1 } , z _ { 4 } )$ </td><td>1</td></tr><tr><td> $V _ { \mathrm { I } }$ </td><td> $( 0 , 0 , 2 z _ { 1 } , z _ { 4 } )$ </td><td>2</td></tr><tr><td> $V _ { \mathrm { I I } }$ </td><td> $( 2 z _ { 1 } , 0 , 0 , z _ { 4 } + z _ { 1 } )$ </td><td>2</td></tr></table>

The integer n counts how many of $d , z _ { 2 }$ equal $z _ { 1 }$ .

Put $A =$ coth $L , s = 2 z _ { 4 }$ , and change variables $c =$ coth t in the integral, so that $e ^ { - 2 z _ { 4 } \coth t } = e ^ { - s c } , d t = - d c / ( c ^ { 2 } - 1 )$ , and sinh $t = ( c ^ { 2 } - 1 ) ^ { - 1 / 2 }$ . Its positive density is

$$
\mu ( c ) = \frac { p ( \operatorname { a r c c o t h } c ) } { ( c ^ { 2 } - 1 ) ^ { 3 / 2 } } , \qquad 1 < c \leq A ,\tag{4.16}
$$

that is, $\textstyle \int _ { L } ^ { \infty }$ sinh $\begin{array} { r } { t p ( t ) f ( \coth t ) d t = \int _ { 1 } ^ { A } \mu ( c ) f ( c ) } \end{array}$ dc for every integrable $f ,$ the reversed limits absorbing the sign of dt. Each corner gap has the form

$$
\mathcal { G } _ { \mathbf { v } } ( s ) = B _ { \mathbf { v } } ( L ) + \int _ { 1 } ^ { A } \mu ( c ) K _ { \mathbf { v } } ( L , c ) e ^ { - s c } d c + a _ { \mathbf { v } } ( L ) e ^ { - s A } .\tag{4.17}
$$

For the four-expert contribution, we use the two facts from [6, Theorems 3.1 and 3.2] recorded after (2.14): the function $u _ { 4 } ( x _ { 1 } , \ldots , x _ { 4 } ) = x _ { 1 } + F _ { 4 } / k$ is the global $C ^ { 2 }$ solution of

the four-expert equation, and $( 1 , 0 , 1 , 0 )$ attains its Hamiltonian maximum throughout the closed ordered sector. The identity $\partial _ { z _ { 2 } } ^ { 2 } F _ { 4 } = F _ { 4 }$ identifies the stated maximizing control. The inequality extends to ties by the $C ^ { 2 }$ regularity in [6, Theorem 3.1].

For $\overline { { \mathbf { v } } } = ( v _ { 1 } , \ldots , v _ { 4 } )$ the four-expert contribution is therefore

$$
B _ { \mathbf { v } } = F _ { 4 } - q ^ { T } \nabla ^ { 2 } F _ { 4 } q = { \frac { k } { 2 } } { \big [ } 2 ( u _ { 4 } - x _ { 1 } ) - D _ { \mathbf { v } } ^ { 2 } u _ { 4 } { \big ] } \geq 0 .
$$

The fifth component of v has no efect. Every corner retained above is in the closed four-expert sector, including $V _ { \mathrm { I } }$ and $V _ { \mathrm { I I } }$ . The nonphysical corner $( d , z _ { 2 } ) = ( z _ { 1 } , 0 )$ is handled by symmetry and (4.15); no four-expert inequality outside the ordered sector is used. The atom comes from the moving lower limit $z _ { 1 }$ of the integral: as in (4.8), the first z -derivative produces no endpoint term, and the second produces $- \partial _ { z _ { 1 } } ( \mathrm { i n t e g r a n d } ) | _ { t = z _ { 1 } } = p ( L ) e ^ { - s A } \Phi _ { L } ( d ) \Phi _ { L } ( z _ { 2 } )$ which vanishes unless $d = z _ { 2 } = 0$ because $\Phi _ { L } ( L ) = 0$ . In $\mathcal { G } _ { \mathbf { v } } = F - q ^ { T } \nabla ^ { 2 } F q$ it carries the coeficient $- q _ { z _ { 1 } } ^ { 2 }$ , so $a _ { \mathbf { v } } = - q _ { z _ { 1 } } ^ { 2 } p ( L )$ when $n = 0$ and $a _ { \mathbf { v } } = 0$ otherwise. It must be retained, since the second derivative moves the integration endpoint.

We can express all of the kernels using one polynomial formula. We set $( a _ { 0 } , a _ { 1 } , a _ { 2 } ) =$ $( L , d , z _ { 2 } )$ , with $d , z _ { 2 } \in \{ 0 , L \}$ at the corners, ψ = cosh $a _ { i } - c$ sinh $a _ { i } , N _ { i } = c$ cosh a − sinh ${ { a } _ { i } } ,$ and $P = \psi _ { 0 } \psi _ { 1 } \psi _ { 2 }$ , so that for fixed $c$ the integrand of the correction is $e ^ { - s c } P$ with

$$
\partial _ { a _ { i } } \psi _ { i } = - N _ { i } , \qquad \partial _ { a _ { i } } ^ { 2 } \psi _ { i } = \psi _ { i } , \qquad \partial _ { z 4 } e ^ { - s c } = - 2 c e ^ { - s c } , \qquad \partial _ { z 4 } ^ { 2 } e ^ { - s c } = 4 c ^ { 2 } e ^ { - s c } .
$$

Expanding $q ^ { T } \nabla ^ { 2 } ( e ^ { - s c } P ) q$ with these rules and subtracting it from $e ^ { - s c } P$ gives

$$
K _ { \mathbf { v } } = \left( 1 - \sum _ { i = 0 } ^ { 2 } q _ { i } ^ { 2 } - 4 q _ { z _ { 4 } } ^ { 2 } c ^ { 2 } \right) P - 2 \sum _ { i < j } q _ { i } q _ { j } N _ { i } N _ { j } \psi _ { h } - 4 q _ { z _ { 4 } } c \sum _ { i } q _ { i } N _ { i } \prod _ { j \ne i } \psi _ { j } ,\tag{4.18}
$$

where h is the remaining index in the pair sum. We use this formula to generate each of the 64 comparisons.

We next give three elementary sign criteria that will be used in the Hamiltonian comparisons. We state them for an integrable density h on $[ 1 , A ]$

Lemma 4.4. Let $\begin{array} { r } { G ( s ) = B + \int _ { 1 } ^ { A } h ( c ) e ^ { - s c } d c + a e ^ { - s A } } \end{array}$ , with $B \geq 0 , a \leq 0$ , and $G ( 0 ) \geq 0$

1. $I f h \leq 0$ , then $G ( s ) \geq G ( 0 )$ . If $h \geq 0 _ { i }$ , then $G ( s ) \geq ( 1 - e ^ { - s A } ) B + e ^ { - s A } G ( 0 )$

2. $H a = 0$ and h changes sign at most once, from positive to negative, then $G ( s ) \geq 0$

3. If $a = 0$ , h changes sign at most once from negative to positive, and $\begin{array} { r } { - \int _ { 1 } ^ { A } c h ( c ) d c \geq 0 . } \end{array}$ then $G ^ { \prime } ( s ) \geq 0$ and $G ( s ) \geq 0$

Proof. For the first assertion, if $h \leq 0$ then

$$
G ( s ) - G ( 0 ) = \int _ { 1 } ^ { A } h ( c ) \big ( e ^ { - s c } - 1 \big ) d c + a \big ( e ^ { - s A } - 1 \big ) \geq 0 ,
$$

both integrand and atom being products of two nonpositive factors; if $h \geq 0$ then $e ^ { - s c } \geq e ^ { - s A }$ on [1, A] gives

$$
G ( s ) \geq B + e ^ { - s A } \Bigl ( \int _ { 1 } ^ { A } h d c + a \Bigr ) = B + e ^ { - s A } \bigl ( G ( 0 ) - B \bigr ) .
$$

For the second assertion let $c _ { * }$ be the transition point. Then $h ( c ) ( e ^ { - s c } - e ^ { - s c _ { * } } ) \geq 0$ for every $^ { c , }$ since both factors change sign at $c _ { * }$ in the same direction, so with $a = 0$

$$
G ( s ) \geq B + e ^ { - s c _ { * } } \int _ { 1 } ^ { A } h d c = ( 1 - e ^ { - s c _ { * } } ) B + e ^ { - s c _ { * } } G ( 0 ) \geq 0 .
$$

In the last case, multiplication by $c > 0$ preserves the negative-to-positive ordering of $h ,$ , so $c h ( c ) ( e ^ { - s c } - e ^ { - s c _ { * } } ) \leq 0$ and

$$
G ^ { \prime } ( s ) = - \int _ { 1 } ^ { A } c h ( c ) e ^ { - s c } d c \geq - e ^ { - s c _ { * } } \int _ { 1 } ^ { A } c h ( c ) d c \geq 0 ,
$$

whence $G ( s ) \geq G ( 0 ) \geq 0$ . Zero densities and absent sign changes follow by the same comparisons. □

For the remaining comparisons, we use a second cumulative integral.

Lemma 4.5. For the same $G ,$ define

$$
C _ { 2 } ( b ) = B b + \int _ { 1 } ^ { b } ( b - c ) h ( c ) d c \quad ( 1 \leq b \leq A ) , \qquad C _ { 2 } ( b ) = B b \quad ( 0 \leq b \leq 1 ) .
$$

$I f G ( 0 ) \geq 0$ and $C _ { 2 } \geq 0$ on [0, A], then $G ( s ) \geq 0$ for $s \geq 0$

Proof. Let $\begin{array} { r } { C _ { 1 } ( b ) = B + \int _ { 1 } ^ { b } } \end{array}$ h dc for $b \geq 1$ and $C _ { 1 } ( b ) = B$ for $0 \leq b \leq 1$ , so that $C _ { 2 } ^ { \prime } = C _ { 1 }$ 2 $C _ { 2 } ( 0 ) = 0$ , and $C _ { 1 } ( A ) + a = G ( 0 )$ . Integrating by parts once,

$$
G ( s ) = B + \int _ { 0 } ^ { A } e ^ { - s c } C _ { 1 } ^ { \prime } ( c ) d c + a e ^ { - s A } = e ^ { - s A } { \bigl ( } C _ { 1 } ( A ) + a { \bigr ) } + s \int _ { 0 } ^ { A } e ^ { - s c } C _ { 1 } ( c ) d c ,
$$

and integrating by parts once more, with $C _ { 1 } = C _ { 2 } ^ { \prime }$ , gives the exact identity

$$
G ( s ) = e ^ { - s A } G ( 0 ) + s e ^ { - s A } C _ { 2 } ( A ) + s ^ { 2 } \int _ { 0 } ^ { A } e ^ { - s b } C _ { 2 } ( b ) d b .\tag{4.19}
$$

The atom at A enters $G ( 0 )$ , and its contribution to $C _ { 2 } ( A )$ is zero. All terms on the right are nonnegative. □

We can now give the full classification. The symbols +, −, 0 denote the sign of the polynomial kernel, not the sign or vanishing of the full ${ \mathrm { g a p } } ;$ PN denotes a positive-to-negative transition, M uses the moment condition in Lemma 4.4, and C uses Lemma 4.5. M and C each reduce to three distinct gaps, listed below; the audit identifies these cases by exact equality of the triple (base, kernel, atom). The standing hypotheses $B \geq 0$ and $\mathcal { G } _ { \mathbf { v } } ( 0 ) \geq 0$ of Lemmas 4.4 and 4.5 are supplied for all 64 cases in Section 4.4, through the boundary values at $z _ { 4 } = 0$ and (4.20).

$$
 \begin{array} { r l } { { \frac { \mathbf { v } } { ( 0 . 0 ) ( 0 ) } } } &  =   \begin{array} { r l r l r l r l } & { V _ { 0 } } & { = V _ { 1 } } & { V _ { 1 } } & { V _ { 1 } } & { } & { V _ { 0 } } & { } \\ & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } \\ { { \mathrm { o n d i n } } } & { } & { } & { } & { } & { } & { } & { } & { } & { } & { } &  \end{array} \end{array}
$$

We emphasize that the base and endpoint atom in (4.17) are still included in each comparison. For example, at $V _ { 0 }$ with $\mathbf { v } = 0 0 1 1 1$ , the table records a zero kernel, but

$$
\begin{array} { r } { \mathcal G _ { \mathbf v } ( s ) = \frac 1 2 \operatorname { t a n h } L - p ( L ) e ^ { - s \coth L } , \qquad \mathcal G _ { \mathbf v } ( 0 ) = \eta ( L ) > 0 . } \end{array}
$$

Lemma 4.4 accounts for the negative atom, and the gap is positive for every $L > 0$ and $s \geq 0$ We verify the polynomial sign classifications as follows. We divide $K _ { \mathbf { v } }$ by $S ^ { n + 1 }$ , write it as a polynomial in $A , c ,$ and substitute $A = 1 + a , c = 1 + a y$ , where $a > 0$ and $0 \leq y \leq 1$ . Its Bernstein coeficients in $y$ are polynomials in $a .$ . The signs of their power coeficients establish fixed signs or at most one sign variation. For a sequence with one uncertain coeficient, either of its signs gives the same allowable variation bound. Bernstein variation reduction follows by substituting $y = t / ( 1 + t )$ and applying $\mathrm { D e s c a r t e s } ^ { \prime }$ rule to the resulting power polynomial. There is one exceptional M kernel, $V _ { \mathrm { I } }$ with $\mathbf { v } = 0 0 1 0 1$ . Its Bernstein coeficients are

$$
- \frac { 5 } { 4 } a ^ { 3 } , \quad - \frac { a ^ { 3 } ( 6 a - 7 ) } { 1 0 } , \quad - \frac { 3 a ^ { 3 } ( 2 a ^ { 2 } - 8 a - 1 7 ) } { 4 0 } , \quad \frac { a ^ { 3 } ( a + 2 ) ( 3 a + 5 ) } { 1 0 } , \quad \frac { a ^ { 3 } ( a + 2 ) ^ { 2 } } { 1 0 } , \quad 0 .
$$

The two uncertain coeficients cannot have signs $+ , -$ , since $2 a ^ { 2 } - 8 a - 1 7 < 0$ on $0 \leq a \leq 7 / 6$ Thus this kernel also has at most one negative-to-positive transition. The exact audit reconstructs each kernel from (4.18) before performing these sign checks.

## 4.4 Regions I and II: the scalar inequalities

At $z _ { 4 } = 0$ , the families $V _ { 0 } , V _ { 1 } , V _ { \mathrm { I I } }$ coincide with the Region III vertices with respectively $\operatorname { z e r o } ,$ one, and two entries equal to $L .$ . The $C ^ { 2 }$ matching therefore gives all 48 boundary inequalities there. For the remaining family $V _ { \mathrm { I } }$ , the comparisons at $z _ { 4 } = 0$ reduce to the three generators

$$
C _ { * } = \mathcal { G } _ { 0 0 0 1 1 } , \qquad B _ { * } = \mathcal { G } _ { 0 0 1 0 0 } , \qquad D _ { * } = \mathcal { G } _ { 0 0 1 1 0 } .
$$

The kernels of $B _ { * } , D _ { * }$ are nonnegative, these being the + entries at $( V _ { \mathrm { I } } , 0 0 1 0 0 )$ and (V<sub>I</sub>, 00110) of the table, and their bases are nonnegative by the four-expert inequality. The full remaining list is

$$
\begin{array} { r l } & { \mathcal { G } _ { 0 0 0 0 1 } = \mathcal { G } _ { 0 0 0 1 0 } = C _ { * } + B _ { * } + D _ { * } , } \\ & { \mathcal { G } _ { 0 1 0 0 0 } = \mathcal { G } _ { 0 1 1 1 1 } = B _ { * } , } \\ & { \mathcal { G } _ { 0 0 1 0 1 } = \mathcal { G } _ { 0 1 0 0 1 } = \mathcal { G } _ { 0 1 1 1 0 } = \mathcal { G } _ { 0 1 0 1 0 } = \mathcal { G } _ { 0 1 1 0 1 } = \mathcal { G } _ { * } , } \\ & { \mathcal { G } _ { 0 0 1 1 1 } = \mathcal { G } _ { 0 1 0 1 1 } = \mathcal { G } _ { 0 1 1 0 0 } = 0 . } \end{array}\tag{4.20}
$$

Control 00000 is harmless because $F \geq 0$ . Only $C _ { * } \geq 0$ needs an additional scalar sign; we certify $C _ { * } / C ^ { 2 }$

All needed scalar integrals reduce to seven moments:

$$
M _ { j } ( L ) = \int _ { 1 } ^ { \coth L } c ^ { j } \mu ( c ) d c = \int _ { L } ^ { \infty } \sinh t p ( t ) \coth ^ { j } t d t , \qquad 0 \leq j \leq 6 .\tag{4.21}
$$

Let $d _ { 0 } = p ( L ) S$ . The first two are

$$
M _ { 1 } = { \frac { d _ { 0 } + 1 / ( 2 C ) } { 5 } } , \qquad M _ { 0 } = { \frac { e - C \theta + S M _ { 1 } } { C } } .\tag{4.22}
$$

The second identity is (3.40) at $X = L .$ , since sinh(t − L) = sinh t cosh L − cosh t sinh $L$ gives $g ( L ) = C M _ { 0 } - S M _ { 1 }$ and $g = e - e _ { 4 } = e - C \theta$ . The first follows from the density equation (3.38), which gives (sinh t p)<sup>′</sup> = cosh $t p +$ sinh $t p ^ { \prime } = - 5$ cosh $t p + { \frac { 1 } { 2 } }$ tanh t sech $t ;$ integrating from L to infinity, where sinh $t p  0$ , yields $\begin{array} { r } { 5 M _ { 1 } = d _ { 0 } + \frac { 1 } { 2 } } \end{array}$ sech $\bar { L }$ . In the c variable, with $c ^ { \prime } = - ( c ^ { 2 } - 1 )$ and sec $\mathrm { h } ^ { 2 } t = ( c ^ { 2 } - 1 ) / c ^ { 2 }$ , the density equation reads

$$
\mu ^ { \prime } - { \frac { 3 c } { c ^ { 2 } - 1 } } \mu = - { \frac { 1 } { 2 c ^ { 2 } ( c ^ { 2 } - 1 ) ^ { 3 / 2 } } } .
$$

Hence

$$
\frac { d } { d c } \big [ c ^ { j } ( c ^ { 2 } - 1 ) \mu \big ] = ( j + 5 ) c ^ { j + 1 } \mu - j c ^ { j - 1 } \mu - \frac { c ^ { j - 2 } } { 2 \sqrt { c ^ { 2 } - 1 } } ,
$$

and integrating from 1 to A, with $( A ^ { 2 } - 1 ) \mu ( A ) = d _ { 0 }$ , gives

$$
M _ { j + 1 } = \frac { A ^ { j } d _ { 0 } + j M _ { j - 1 } + Q _ { j - 2 } / 2 } { j + 5 } , \qquad j \geq 1 ,\tag{4.23}
$$

where $\begin{array} { r } { Q _ { r } = \int _ { 1 } ^ { A } c ^ { r } / \sqrt { c ^ { 2 } - 1 } d c = \int _ { L } ^ { \infty } } \end{array}$ coth $\textsuperscript { \textit { r } } t$ csch t dt and the needed values are

$$
\begin{array} { r } { Q _ { - 1 } = 2 \theta , \quad Q _ { 0 } = \lambda , \quad Q _ { 1 } = S ^ { - 1 } , \quad Q _ { 2 } = \frac { 1 } { 2 } ( C / S ^ { 2 } + \lambda ) , \quad Q _ { 3 } = \frac { 1 } { 3 S ^ { 3 } } + S ^ { - 1 } . } \end{array}\tag{4.24}
$$

The boundary term at $c = 1$ is zero, since $( c ^ { 2 } - 1 ) \mu = \mathcal { O } ( \sqrt { c - 1 } )$ . These formulas are also checked by $M _ { i } ^ { \prime } = - S p A ^ { j }$ and their zero limits at infinity.

There are only three distinct M comparisons: $( V _ { 1 } , 0 0 1 0 1 ) , ( V _ { \mathrm { I } } , 0 0 1 0 1 )$ , and $( V _ { \mathrm { I I } } , 0 0 0 1 1 )$ For these the moment condition of Lemma $4 . 4 ( 3 )$ , with $\begin{array} { r } { h = \mu K _ { \mathbf { v } } , \mathrm { i s } - \int _ { 1 } ^ { A } c \mu ( c ) K _ { \mathbf { v } } ( L , c ) d c \geq } \end{array}$ $0 ;$ if $\begin{array} { r } { K _ { \mathbf { v } } = \sum _ { j } k _ { j } ( L ) c ^ { j } } \end{array}$ this reads

$$
N _ { \mathbf { v } } ( L ) = - C ^ { - n } \sum _ { j } k _ { j } ( L ) M _ { j + 1 } ( L ) \geq 0 .\tag{4.25}
$$

The factor $C ^ { - n } > 0$ is only a convenient normalization. The moment recursion makes all three finite expressions, linear in $e , p , \theta , \lambda ;$ the certificates below establish their signs.

We next treat the three cumulative comparisons, marked C in the sign table. The distinct C comparisons are (V<sub>0</sub>, 01001), (V<sub>1</sub>, 01110), and $( V _ { \mathrm { I I } } , 0 1 1 1 0 )$ . All have $\mathcal { G } _ { \mathbf { v } } ( 0 ) = 0$ . For $t \geq L$ define

$$
J ( L , t ) = B _ { \mathbf { v } } ( L ) + \int _ { 1 } ^ { \coth t } ( 1 - c \operatorname { t a n h } t ) \mu ( c ) K _ { \mathbf { v } } ( L , c ) d c .\tag{4.26}
$$

This is $C _ { 2 } ( \coth t ) /$ coth t. Let $m = n + 1$ , which takes the values 1, 2, 3 in these three cases, and put

$$
x = e ^ { - 2 L } , \quad y = e ^ { - 2 t } , \qquad H ( x , y ) = e ^ { - m L } J ( L , t ) , \qquad 0 < y \leq x < 1 .\tag{4.27}
$$

For fixed $c , e ^ { - m L } K _ { \mathbf { v } } ( L , c )$ is a polynomial of degree at most m in $x .$ Thus its $( m + 1 )$ st derivative vanishes, and only the four-expert base contributes to the next derivative. We obtain the positive rational function

$$
\partial _ { x } ^ { m + 1 } H ( x , y ) = \frac { m ! \varrho ^ { 2 m + 5 } } { ( \varrho ^ { 2 } - 1 ) ^ { m } ( \varrho ^ { 2 } + 1 ) ^ { 3 } } > 0 , \qquad \varrho = e ^ { L } = x ^ { - 1 / 2 } .\tag{4.28}
$$

This identity follows by diferentiating (2.14) with the specified control; it is checked exactly for all three cases.

It remains to prove the nine diagonal signs

$$
A _ { j } ( t ) = \partial _ { x } ^ { j } H ( x , y ) \big | _ { x = y } \geq 0 , \qquad 0 \leq j \leq m .\tag{4.29}
$$

For fixed $y > 0$ , the kernel part of $H ( \cdot , y )$ is polynomial and its base is smooth for $L > 0$ , so $H ( \cdot , y ) \in C ^ { m + 1 } ( [ y , x ] )$ whenever $y \le x < 1$ . Taylor’s formula with integral remainder then gives

$$
H ( x , y ) = \sum _ { j = 0 } ^ { m } \frac { A _ { j } ( t ) } { j ! } ( x - y ) ^ { j } + \frac { 1 } { m ! } \int _ { y } ^ { x } ( x - \xi ) ^ { m } \partial _ { x } ^ { m + 1 } H ( \xi , y ) d \xi \geq 0 .\tag{4.30}
$$

Thus $C _ { 2 } \geq 0$ and Lemma 4.5 applies.

To write the scalar functions explicitly, we set

$$
b _ { j } = \partial _ { x } ^ { j } ( e ^ { - m L } B _ { \mathbf { v } } ) , \qquad \partial _ { x } ^ { j } ( e ^ { - m L } K _ { \mathbf { v } } ) = \sum _ { r } k _ { j r } ( L ) c ^ { r } , \qquad \partial _ { x } = - \frac { 1 } { 2 } e ^ { 2 L } \partial _ { L } .
$$

Diferentiate with c fixed before taking the diagonal. Then

$$
A _ { j } ( L ) = b _ { j } ( L ) + \sum _ { r } k _ { j r } ( L ) \big [ M _ { r } ( L ) - \operatorname { t a n h } L M _ { r + 1 } ( L ) \big ] .\tag{4.31}
$$

Equations (4.18), (4.22), (4.23), and (4.31) specify all nine scalars by finite rational operations and the trace. We use these exact expressions as inputs to the certificate calculation.

We now prove the thirteen remaining scalar inequalities. Substitute (2.12) into the nine $A _ { j }$ , the three $N _ { \mathbf { v } }$ , and $C _ { * } / C ^ { 2 }$ . Every θ term cancels. The resulting expressions have the same form $G = a e + b$ as in the Region III proof. Eleven have first-order certificates (4.12), with $- \operatorname { s g n } ( a ) R \geq 0 ;$

<table><tr><td>Scalar</td><td>Indices</td><td>Signs of a</td></tr><tr><td> $A _ { j } { \mathrm { ~ a t ~ } } ( V _ { 0 } , 0 1 0 0 1 )$ </td><td> $j = 0 , 1$ </td><td> $+ , -$ </td></tr><tr><td> $A _ { j } { \mathrm { ~ a t ~ } } ( V _ { 1 } , 0 1 1 1 0 )$ </td><td> $j = 0 , 1 , 2$ </td><td> $+ , - , -$ </td></tr><tr><td> $A _ { j } \mathrm { ~ a t ~ } ( V _ { \mathrm { I I } } , 0 1 1 1 0 )$ </td><td> $j = 0 , 2 , 3$ </td><td> $+ , - , -$ </td></tr><tr><td> $\dot { C _ { * } } / { C ^ { 2 } } _ { \mathrm { ~ a t ~ } } V _ { \mathrm { I } }$ </td><td></td><td>一</td></tr><tr><td> $N _ { 0 0 1 0 1 } ~ \mathrm { a t } ~ V _ { \mathrm { I } }$ </td><td></td><td>十</td></tr><tr><td> $N _ { 0 0 0 1 1 } ~ \mathrm { a t } ~ V _ { \mathrm { I I } }$ </td><td></td><td>一</td></tr></table>

Fixed signs of a are verified by exact polynomial root counts on $\varrho > 1$ , after removing zeros at $\varrho = 1$ , and evaluation at a rational test point. The endpoint calculation gives $b / ( a C )  0 .$ while $e / C \to 0$ . Integration from infinity proves all eleven scalar inequalities.

The remaining moment at (V<sub>1</sub>, 00101) is independent of e:

$$
\begin{array} { r l } & { N _ { 0 0 1 0 1 } = \big [ 1 0 5 \lambda \varrho ^ { 8 } + 6 \lambda \varrho ^ { 6 } - 6 \lambda \varrho ^ { 2 } - 1 0 5 \lambda - 2 5 6 p \varrho ^ { 5 } - 2 5 6 p \varrho ^ { 3 } } \\ & { \qquad \quad - 2 1 0 \varrho ^ { 7 } - 8 2 \varrho ^ { 5 } + 8 2 \varrho ^ { 3 } + 2 1 0 \varrho \big ] / \big [ 1 0 0 8 \varrho ( \varrho ^ { 2 } - 1 ) ( \varrho ^ { 2 } + 1 ) ^ { 2 } \big ] . } \end{array}\tag{4.32}
$$

After substitution for $p ,$ it is certified directly.

Finally $A _ { 1 }$ at $( V _ { \mathrm { I I } } , 0 1 1 1 0 )$ has the sign-changing coeficient

$$
a = \frac { 3 ( \varrho ^ { 2 } - 1 ) ( 7 \varrho ^ { 4 } - 1 4 \varrho ^ { 2 } - 3 3 ) } { 2 5 6 \varrho ^ { 3 } ( \varrho ^ { 2 } + 1 ) ^ { 2 } } .
$$

Multiply by $\sigma = 5 1 2 { \varrho ^ { 4 } ( \varrho ^ { 2 } + 1 ) } / [ 3 ( \varrho ^ { 2 } - 1 ) ] > 0$ and put $\widetilde { G } = \sigma A _ { 1 }$ . Its homogeneous solution is $h = \sigma a C = 7 \varrho ^ { 4 } - 1 4 \varrho ^ { 2 } - 3 3$ , with $h ^ { \prime } = 2 8 \varrho ^ { 2 } ( \varrho ^ { 2 } - 1 ) > 0$ . The exact certificate is

$$
\widetilde { G } ^ { \prime \prime } - \frac { h ^ { \prime \prime } } { h ^ { \prime } } \widetilde { G } ^ { \prime } \geq 0 .\tag{4.33}
$$

Expansion (3.26) gives $\widetilde G \to 0$ and ${ \widetilde { G } } ^ { \prime } / h ^ { \prime } \to 0 ;$ only the accuracy $o ( e ^ { - 5 L } )$ of its three displayed terms is needed for these two limits, since the coeficients of $e$ in $\widetilde { G }$ and in $\widetilde { G } ^ { \prime } / h ^ { \prime }$ are $\mathcal O ( \varrho ^ { 5 } )$ . Hence $( \widetilde { G } ^ { \prime } / h ^ { \prime } ) ^ { \prime } \geq 0$ , so $\widetilde { G } ^ { \prime } \le 0$ and $\widetilde { G } \geq 0$ . The purpose of the positive rescaling is to ensure that the homogeneous derivative has no zero in the interior.

## 4.5 Exact rational certificates and the supplement

We now describe the exact certificates used to verify the scalar inequalities. The full arithmetic data are included in the supplement. Every signed residual just used can be written as

$$
{ \frac { A ( q ) ( - \log q ) + B ( q ) 2 \operatorname { a t a n h } q + C ( q ) } { D ( q ) } } , \qquad q = e ^ { - L } \in ( 0 , 1 ) , \quad D ( q ) > 0 ,\tag{4.34}
$$

where the coeficients are rational polynomials. Derivatives are evaluated exactly using $\varrho \partial _ { \varrho } ,$ $L ^ { \prime } = 1 , \lambda ^ { \prime } = - 1 /$ sinh $L ,$ and (2.9). Polynomial root counts check every denominator sign and exclude interior singularities.

For $0 \leq x < 1$ and $N = 1 2$ , let

$$
T _ { N } ( x ) = 2 \sum _ { j = 0 } ^ { N - 1 } { \frac { x ^ { 2 j + 1 } } { 2 j + 1 } } , \qquad T _ { N } ( x ) \leq 2 \operatorname { a t a n h } x \leq T _ { N } ( x ) + { \frac { 2 x ^ { 2 N + 1 } } { ( 2 N + 1 ) ( 1 - x ^ { 2 } ) } } .\tag{4.35}
$$

The last inequality bounds the tail by a geometric series. Use the seven intervals with endpoints

$$
0 , \quad 1 / 6 4 , \quad 1 / 3 2 , \quad 1 / 1 6 , \quad 1 / 8 , \quad 1 / 4 , \quad 1 / 2 , \quad 1 .\tag{4.36}
$$

On a middle dyadic interval $[ 2 ^ { - j } , 2 ^ { 1 - j } ]$ write

$$
- \log q = j \log 2 - 2 \operatorname { a t a n h } { \frac { 2 ^ { j } q - 1 } { 2 ^ { j } q + 1 } } , \qquad \log 2 = 2 \operatorname { a t a n h } ( 1 / 3 ) .
$$

On $[ 1 / 2 , 1 )$ use − log $q = 2$ atanh $( ( 1 - q ) / ( 1 + q ) )$ . On $( 0 , 1 / 6 4 ]$ use 6 log $2 \leq -$ log $q \ \leq$ $( q ^ { - 1 } - q ) / 2$ . All the resulting logarithm bounds are rational functions.

If a coeficient of a logarithm has a fixed sign, substitute its appropriate upper or lower bound to obtain a lower bound for the entire residual. There is also a fully explicit rule for a mixed-sign coeficient $A ( q )$ . Let $\widehat { A } ( q )$ be the polynomial with absolute values of its power coeficients, so $| A ( q ) | \leq { \widehat { A } } ( q )$ for $q \geq 0$ . If $\underline { { f } } \le f \le \overline { { f } }$ , then

$$
A ( q ) f ( q ) \geq A ( q ) \underline { { { f } } } ( q ) - \widehat { A } ( q ) \big ( \overline { { { f } } } ( q ) - \underline { { { f } } } ( q ) \big ) .\tag{4.37}
$$

Thus coeficient sign changes create no unresolved interval decision.

After clearing denominators whose positivity has been verified, each lower bound is a rational polynomial $P ( q )$ . For an interval $[ a , b ]$ , write

$$
\begin{array} { l } { { P ( a + ( b - a ) t ) = \displaystyle \sum _ { j = 0 } ^ { d } c _ { j } t ^ { j } = \displaystyle \sum _ { i = 0 } ^ { d } B _ { i } { \binom { d } { i } } t ^ { i } ( 1 - t ) ^ { d - i } , } } \\ { { B _ { i } = \displaystyle \sum _ { j = 0 } ^ { i } c _ { j } \displaystyle \frac { { \binom { i } { j } } } { { \binom { d } { j } } } . } } \end{array}\tag{4.38}
$$

Every $B _ { i }$ is nonnegative as an exact rational number. Therefore $P \geq 0$ on its whole interval.   
Positive common-denominator rescaling gives the integer coeficient lists in the supplement.

There are eight scalar certificates in Region III and thirteen in Regions I and II. Each uses the seven intervals (4.36), with no additional subdivision:

<table><tr><td>Region</td><td>Scalar signs</td><td>Polynomial certificates</td><td>Largest degree</td></tr><tr><td>III</td><td>8</td><td>56</td><td>79</td></tr><tr><td>I and II</td><td>13</td><td>91</td><td>72</td></tr><tr><td>Total</td><td>21</td><td>147</td><td>79</td></tr></table>

All polynomial power coeficients, rational intervals, integer Bernstein coeficients, and exact scalar expressions are included in the two certificate archives. Separate checkers recompute the Bernstein coeficients using only standard-library fractions; no symbolic library or floatingpoint calculation is used in that independent arithmetic check. The identity audits reconstruct the scalar expressions from the formula and match them to the certificate inputs. The released verifier independently reconstructs every rational lower bound from (4.35)–(4.37), checks all denominator signs, and matches the resulting polynomial to its certificate before the

Bernstein check. This gives a reproducible exact verification of all the scalar inequalities used above.

The certificates and checks described in this section can be rerun from the computational supplement, as follows. The self-contained supplement is the versioned directory five-expert-proof-1.0.6, archived at doi:10.5281/zenodo.22723924 [10] and distributed as five-expert-proof-1.0.6.zip, whose SHA-256 checksum is

## a58aae99f888ecdd55fe402027f3928b8fabd90807000bcf0d520ce264e32486

It contains all exact proof inputs, checkers, and the pinned dependency wheels. From the directory containing the extracted release, install and run entirely ofline:

python3.12 -m venv proof-env

proof-env/bin/python -m pip install --no-index --require-hashes \

--find-links five-expert-proof-1.0.6/wheels \

-r five-expert-proof-1.0.6/requirements.txt

proof-env/bin/python five-expert-proof-1.0.6/verify.py \

--output proof-results

The output directory must be new and outside the release. If pip is configured for user installs, the install step needs the environment setting PIP\_USER=0. The audit executes checks/trace\_kernels.py, checks/formula.py, checks/faces.py, checks/region3.py, checks/region12.py, checks/comb.py, checks/log\_bounds.py, and checks/rejection\_ tests.py. In particular, it reconstructs every polynomial bound from its scalar expression before checking its exact Bernstein coeficients; no numerical sign sampling is used. Adding -O to the Python command exercises the same checks under optimization. Logs, dependency versions, exit statuses, and the release manifest hash are written to the output directory.

The supplement’s README.md gives the installation and audit procedure, and REVIEW\_ GUIDE.md maps every computational check, and the checker’s variable names, to the paper and states the analytic steps requiring mathematical review.

## 4.6 Completion of the proofs of the main theorems

We now complete the proofs of our main results from Section 2.

Proof of Theorem 2.1. Since $u \in C ^ { 2 }$ is a classical solution, it is also a viscosity solution, and it has linear growth because $u - \varphi$ is bounded. By [6, Proposition 2.2], the viscosity solution of (2.1) is unique in the class of functions with linear growth, so u is the unique viscosity solution with bounded correction. We turn to the proof of the expansion (2.18). By Proposition 4.1, u is $C ^ { 2 }$ at the origin. Permutation symmetry and $u ( x + c \mathbb { 1 } ) = u ( x ) + c$ give $\boldsymbol { \nabla } \boldsymbol { u } ( 0 ) = \frac { 1 } { 5 } \mathbb { 1 }$ (recall (3.1)) and they force the Hessian $\nabla ^ { 2 } u ( 0 )$ to commute with all permutations and to annihilate 1, so $\begin{array} { r } { \nabla ^ { 2 } u ( 0 ) = \alpha ( I - \frac { 1 } { 5 } \mathbb { 1 } \mathbb { 1 } ^ { T } ) } \end{array}$ for some α. For a binary v with m entries equal to one, $\mathbf { v } ^ { T } \nabla ^ { 2 } u ( 0 ) \mathbf { v } = \alpha m ( 5 - m ) / 5$ . If $\alpha \leq 0$ the maximum over v is zero and (2.1) would give $u ( 0 ) = 0$ , contradicting $u ( 0 ) > 0 ;$ hence $\alpha > 0$ , the maximum is $6 \alpha / 5 .$ , attained for $m = 2 , 3$ , and (2.1) at the origin gives $u ( 0 ) = 3 \alpha / 5$ . Taylor’s formula with $\alpha = { \textstyle \frac { 5 } { 3 } } u ( 0 )$ and $\begin{array} { r } { x ^ { T } ( I - \frac { 1 } { 5 } \mathbb { 1 } \mathbb { 1 } ^ { T } ) x = \frac { 4 } { 5 } \big ( \sum _ { i } x _ { i } ^ { 2 } - \frac { 1 } { 2 } \sum _ { i < j } x _ { i } x _ { j } \big ) } \end{array}$ is (2.18), with $u ( 0 ) = 4 5 \pi ^ { 2 } / ( 5 1 2 \sqrt { 2 } )$ by Lemma 3.1 and $\scriptstyle { \frac { 2 } { 3 } } u ( 0 ) = 1 5 \pi ^ { 2 } / ( 2 5 6 { \sqrt { 2 } } )$ . The same argument with four experts gives $\begin{array} { r } { \nabla ^ { 2 } u _ { 4 } ( 0 ) = 2 u _ { 4 } ( 0 ) ( I - \frac { 1 } { 4 } \Im { \mathbb { 1 } \Im ^ { T } } ) } \end{array}$ and recovers [6, equation (3.4)] from $u _ { 4 } ( 0 ) = \pi / ( 4 \sqrt { 2 } )$ □

We now prove Theorem 2.2. The Hamiltonian inequalities of this section give $\Delta \geq 0$ for the gap (2.19) but do not identify where equality holds. The proof upgrades this nonnegativity to strict positivity: in Regions I and II through an exact positive-kernel formula for $\Delta ,$ , and in Region III through the strict positivity of the generators of (4.11).

Proof of Theorem 2.2. We treat Regions I and II first, through an exact formula for the gap. Both coordinate matrices (4.14) send $b _ { \mathbf { v } _ { \ast } } = \left( 1 , - 1 , 1 , 0 \right) \mathrm { t o } \left( 0 , 0 , 1 , 0 \right)$ and $b _ { \mathbf { v } _ { C } } = \left( 1 , - 1 , 1 , - 1 \right)$ to $( 0 , 0 , 1 , - 1 )$ in the coordinates $( z _ { 1 } , d , z _ { 2 } , z _ { 4 } ) , d = | z _ { 3 } |$ . By (3.4), $D _ { { \mathbf { v } _ { * } } } ^ { 2 } u = ( 2 / k ) F _ { z _ { 2 } z _ { 2 } }$ and $D _ { \mathbf { v } _ { C } } ^ { 2 } u = ( 2 / k ) ( F _ { z _ { 2 } z _ { 2 } } - 2 F _ { z _ { 2 } z _ { 4 } } + F _ { z _ { 4 } z _ { 4 } } )$ , hence

$$
\Delta = \frac { 2 } { k } ( 2 F _ { z _ { 2 } z _ { 4 } } - F _ { z _ { 4 } z _ { 4 } } ) .\tag{4.39}
$$

Since the four-expert contribution is independent of $z _ { 4 } ,$ it cancels from the gap. Neither direction changes $z _ { 1 } ,$ , so diferentiating the integral produces no endpoint term. With $c =$ coth t, the integrand of (2.15) carries the factor $e ^ { - 2 z _ { 4 } c } \Phi _ { t } ( z _ { 2 } )$ , on which $\partial _ { z _ { 4 } }$ produces −2c and $\partial _ { z _ { 2 } } \Phi _ { t } ( z _ { 2 } ) = - N _ { t } ( z _ { 2 } )$ with $N _ { t } ( z _ { 2 } ) = { \mathfrak { c } }$ c cosh $z _ { 2 }$ − sinh $z _ { 2 } ;$ so $2 \partial _ { z _ { 2 } } \partial _ { z _ { 4 } } - \partial _ { z _ { 4 } } ^ { 2 }$ replaces $\Phi _ { t } ( z _ { 2 } )$ by 4c $\left[ N _ { t } ( z _ { 2 } ) - c \Phi _ { t } ( z _ { 2 } ) \right]$ , and the identity

$$
( c \cosh z _ { 2 } - \sinh z _ { 2 } ) - c ( \cosh z _ { 2 } - c \sinh z _ { 2 } ) = ( c ^ { 2 } - 1 ) \sinh z _ { 2 }
$$

reduces the diferentiated kernel to a positive one:

$$
\Delta = \frac { 8 } { k } \sinh z _ { 2 } \int _ { z _ { 1 } } ^ { \infty } \sinh t p ( t ) \coth t ( \coth ^ { 2 } t - 1 ) e ^ { - 2 z _ { 4 } \coth t } \Phi _ { t } ( z _ { 1 } ) \Phi _ { t } ( | z _ { 3 } | ) d t .\tag{4.40}
$$

For $z _ { 1 } > 0$ , every factor in the integrand is strictly positive for $t > z _ { 1 }$ . Therefore $\Delta = 0$ exactly when $z _ { 2 } = 0$ , or $y _ { 1 } = y _ { 3 } = 0$ . If $z _ { 1 } = 0$ , the first four coordinates coincide. The controls $\mathbf { v } _ { * } = 1 0 1 0 0$ and ${ \bf 1 } - { \bf v } _ { C } = 0 1 0 1 0$ each select two of those four tied coordinates, so their curvatures agree by permutation symmetry, and $\mathbf { v } _ { C }$ has the same curvature as its complement. This proves the theorem in Regions I and II, including their boundary collisions.

In Region III we use the multiafine structure of the gap. For $a _ { 4 } > 0$ put

$$
\xi _ { i } = \frac { \operatorname { t a n h } a _ { i } } { \operatorname { t a n h } a _ { 4 } } \in [ 0 , 1 ] , \qquad { \widehat { G } } = \frac { k \Delta } { 2 \cosh a _ { 1 } \cosh a _ { 2 } \cosh a _ { 3 } } .
$$

The COMB direction in these coordinates is $q = ( - 2 , 1 , - 2 , 1 ) / 3$ , the image of $b _ { \mathbb { 1 } - \mathbf { v } _ { C } } =$ $( - 1 , 1 , - 1 , 1 )$ under (4.9); the sign of $q$ is immaterial in the quadratic form. The eight values of its multiafine gap are

<table><tr><td> $( \xi _ { 1 } , \xi _ { 2 } , \xi _ { 3 } )$  000</td><td>001</td><td>010 011</td><td>100 101</td></tr><tr><td> $\widehat { G }$ </td><td></td></tr><tr><td></td><td>110</td></tr><tr><td> $D _ { 1 }$   $B _ { 2 }$   $D _ { 1 }$ </td><td></td></tr><tr><td>0 0</td><td> $A _ { 2 }$ </td></tr><tr><td></td><td> $B _ { 2 }$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td> $P _ { 3 }$ </td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

The coeficients are the generators defined in (4.11); permuting the three $a _ { i }$ explains the repeated values.

To characterize the equality set, we need to show that these generators are strictly positive. For each $G \in \{ D _ { 1 } , B _ { 2 } , A _ { 2 } , P _ { 3 } \}$ its first-order certificate has $a > 0$ and $R = ( G / ( a C ) ) ^ { \prime } \leq 0$ . In each of the seven polynomial certificates for $- R$ , at least one Bernstein coeficient is strictly

positive and the others are nonnegative. Every Bernstein basis function is positive in the open interval. Thus $R < 0$ in each interval interior. Since $G / ( a C )  0$ , integration from infinity gives

$$
D _ { 1 } ( L ) > 0 , \qquad B _ { 2 } ( L ) > 0 , \qquad A _ { 2 } ( L ) > 0 , \qquad P _ { 3 } ( L ) > 0 \quad ( L > 0 ) .\tag{4.41}
$$

The strict inequalities follow from 28 of the 56 exact polynomial certificates already used in the Region III verification.

Using multiafine interpolation of the table, grouping the vertex terms by their dependence on $\xi _ { 2 }$ , we obtain the factorization

$$
\begin{array} { r l } & { \widehat { G } = ( \xi _ { 1 } + \xi _ { 3 } - 2 \xi _ { 1 } \xi _ { 3 } ) \big [ ( 1 - \xi _ { 2 } ) D _ { 1 } + \xi _ { 2 } B _ { 2 } \big ] } \\ & { \qquad + \xi _ { 1 } \xi _ { 3 } \big [ ( 1 - \xi _ { 2 } ) A _ { 2 } + \xi _ { 2 } P _ { 3 } \big ] . } \end{array}\tag{4.42}
$$

Both brackets are strictly positive. Also $\xi _ { 1 } + \xi _ { 3 } - 2 \xi _ { 1 } \xi _ { 3 } = \xi _ { 1 } ( 1 - \xi _ { 3 } ) + ( 1 - \xi _ { 1 } ) \xi _ { 3 } \geq 0 \mathrm { ~ , ~ }$ Consequently the gap vanishes exactly when $\xi _ { 1 } = \xi _ { 3 } = 0$ . The physical ordering $0 \leq a _ { 1 } \leq$ $a _ { 2 } \leq a _ { 3 }$ forces all three $a _ { i }$ to vanish. By (2.6), this is $y _ { 1 } = y _ { 3 } = y _ { 4 } = 0$ . It is precisely the part of (2.20) lying in Region III. The case $a _ { 4 } = 0$ is the full collision and follows by continuity. Together with (4.40), this completes the proof. □

## A Proof of the gluing lemma

We prove Lemma 4.3 in the notation of its statement: $\Omega \subset \mathbb { R } ^ { n }$ is open and convex, $K _ { 1 } , \ldots , K _ { N }$ are closed convex sets with nonempty interiors whose union contains Ω, and u is continuous on $\Omega ,$ is $C ^ { 2 }$ in the interior of each $K _ { j }$ , and has a gradient and Hessian that extend continuously to each $\Omega \cap K _ { j }$ and agree at every point of Ω for all the $K _ { j }$ containing it.

Proof of Lemma $4 . 3 .$ Let $g$ and h be the common extended gradient and Hessian. They are well defined by the agreement hypothesis and continuous on $\Omega ,$ since each is continuous on the finitely many relatively closed sets $\Omega \cap K _ { j }$ covering Ω. Fix $x , y \in \Omega ;$ the segment $[ x , y ]$ lies in Ω. Each intersection $[ x , y ] \cap K _ { j }$ is a closed subsegment, and these finitely many subsegments cover $[ x , y ]$ , so taking all their endpoints as partition points gives $0 = s _ { 0 } < s _ { 1 } < \cdot \cdot \cdot < s _ { m } = 1$ such that every segment $[ x + s _ { i - 1 } ( y - x ) , x + s _ { i } ( y - x ) ]$ lies in a single $K _ { j }$ . For a segment $[ a , b ]$ in the interior of $K _ { j }$ , the fundamental theorem of calculus gives

$$
u ( b ) - u ( a ) = \int _ { 0 } ^ { 1 } g \big ( a + s ( b - a ) \big ) \cdot ( b - a ) d s .\tag{A.1}
$$

If $[ a , b ] \subset K _ { j }$ meets the boundary of $K _ { j }$ , choose a point z of Ω in the interior of $K _ { j }$ , which exists because Ω is open and meets $K _ { j }$ . For $0 < \varepsilon \le 1$ the segment with endpoints $( 1 - \varepsilon ) a + \varepsilon z$ and $( 1 - \varepsilon ) b + \varepsilon z$ lies in Ω and in the interior of $K _ { j }$ by convexity, so (A.1) holds for it, and letting $\varepsilon \to 0$ gives (A.1) for $[ a , b ]$ , by the continuity of u and $g$ on $\Omega \cap K _ { j }$ . Summing over the subsegments,

$$
u ( y ) - u ( x ) = \int _ { 0 } ^ { 1 } g { \big ( } x + s ( y - x ) { \big ) } \cdot ( y - x ) d s ,
$$

so $u ( y ) - u ( x ) - g ( x ) \cdot ( y - x ) = o ( | y - x | )$ as $y  x .$ by the continuity of $g .$ . Thus u is diferentiable with $\nabla u = g _ { ; }$ , and $u \in C ^ { 1 } ( \Omega )$ . The same argument applied to $^ { g , }$ , whose restriction to the interior of each $K _ { j }$ is $C ^ { 1 }$ with derivative $h ,$ , gives $\nabla g = h$ , so $u \in C ^ { 2 } ( \Omega )$ □

## References

[1] Y. Abbasi-Yadkori, P. L. Bartlett, and V. Gabillon. Near minimax optimal players for the finite-time 3-expert prediction problem. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017.

[2] J. Abernethy, M. K. Warmuth, and J. Yellin. When random play is optimal against an adversary. In Proceedings of the 21st Annual Conference on Learning Theory (COLT 2008), pages 437–446, 2008.

[3] G. Barles and P. E. Souganidis. Convergence of approximation schemes for fully nonlinear second order equations. Asymptotic Analysis, 4(3):271–283, 1991.

[4] E. Bayraktar, I. Ekren, and X. Zhang. Finite-time 4-expert prediction problem. Communications in Partial Diferential Equations, 45(7):714–757, 2020.

[5] E. Bayraktar, I. Ekren, and X. Zhang. Prediction against a limited adversary. Journal of Machine Learning Research, 22(1):5673–5704, 2021.

[6] E. Bayraktar, I. Ekren, and Y. Zhang. On the asymptotic optimality of the comb strategy for prediction with expert advice. Ann. Appl. Probab., 30(6):2517–2546, 2020.

[7] E. Bayraktar, H. V. Poor, and X. Zhang. Malicious experts versus the multiplicative weights algorithm in online prediction. IEEE Transactions on Information Theory, 67(1):559–565, 2021.

[8] J. Calder and N. Drenska. Asymptotically optimal strategies for online prediction with history-dependent experts. Journal of Fourier Analysis and Applications, 27(2):20, 2021.

[9] J. Calder and N. Drenska. Online prediction with history-dependent experts: the general case. Communications on Pure and Applied Mathematics, 76(9):1678–1727, 2023.

[10] J. Calder and N. Drenska. Five-expert PDE: exact proof supplement, version 1.0.6, 2026. DOI:10.5281/zenodo.22723924.

[11] J. Calder, N. Drenska, and D. Mosaphir. Numerical solution of a PDE arising from prediction with expert advice. European J. Appl. Math., 37(1):96–122, 2026.

[12] N. Cesa-Bianchi, Y. Freund, D. Haussler, D. P. Helmbold, R. E. Schapire, and M. K. Warmuth. How to use expert advice. Journal of the ACM, 44(3):427–485, 1997.

[13] N. Cesa-Bianchi and G. Lugosi. Prediction, learning, and games. Cambridge University Press, 2006.

[14] Z. Chase. Experimental evidence for asymptotic non-optimality of comb adversary strategy, 2019. arXiv:1912.01548.

[15] T. M. Cover. Behavior of sequential predictors of binary sequences. In Transactions of the Fourth Prague Conference on Information Theory, Statistical Decision Functions, Random Processes (Prague, 1965), pages 263–272, Prague, 1967. Academia.

[16] M. G. Crandall, H. Ishii, and P.-L. Lions. User’s guide to viscosity solutions of second order partial diferential equations. Bulletin of the American Mathematical Society (N.S.), 27(1):1–67, 1992.

[17] N. Drenska. A PDE Approach to a Prediction Problem Involving Randomized Strategies. PhD thesis, New York University, 2017.

[18] N. Drenska and R. V. Kohn. Prediction with expert advice: a PDE perspective. J. Nonlinear Sci., 30:137–173, 2020.

[19] N. Drenska and R. V. Kohn. A PDE approach to the prediction of a binary sequence with advice from two history-dependent experts. Communications on Pure and Applied Mathematics, 76(4):843–897, 2023.

[20] Y. Freund and R. E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997.

[21] N. Gravin, Y. Peres, and B. Sivan. Towards optimal algorithms for prediction with expert advice. In Proceedings of the Twenty-Seventh Annual ACM-SIAM Symposium on Discrete Algorithms (SODA 2016), pages 528–547. SIAM, 2016.

[22] N. Gravin, Y. Peres, and B. Sivan. Tight lower bounds for multiplicative weights algorithmic families. In 44th International Colloquium on Automata, Languages, and Programming (ICALP 2017), volume 80 of LIPIcs, pages 48:1–48:14, 2017.

[23] J. Hannan. Approximation to Bayes risk in repeated play. In Contributions to the Theory of Games, Vol. III, volume 39 of Annals of Mathematics Studies, pages 97–139. Princeton University Press, 1957.

[24] D. Haussler, J. Kivinen, and M. K. Warmuth. Tight worst-case loss bounds for predicting with expert advice. In Computational Learning Theory (Barcelona, 1995), volume 904 of Lecture Notes in Computer Science, pages 69–83, Berlin, 1995. Springer.

[25] V. A. Kobzar, R. V. Kohn, and Z. Wang. New potential-based bounds for prediction with expert advice. In Proceedings of the 33rd Conference on Learning Theory (COLT 2020), volume 125 of Proceedings of Machine Learning Research, pages 2370–2405, 2020.

[26] V. A. Kobzar, R. V. Kohn, and Z. Wang. New potential-based bounds for the geometricstopping version of prediction with expert advice. In Proceedings of the First Mathematical and Scientific Machine Learning Conference (MSML 2020), volume 107 of Proceedings of Machine Learning Research, pages 537–554, 2020.

[27] R. V. Kohn and S. Serfaty. A deterministic-control-based approach to motion by curvature. Communications on Pure and Applied Mathematics, 59(3):344–407, 2006.

[28] R. V. Kohn and S. Serfaty. A deterministic-control-based approach to fully nonlinear parabolic and elliptic equations. Communications on Pure and Applied Mathematics, 63(10):1298–1350, 2010.

[29] N. Littlestone and M. K. Warmuth. The weighted majority algorithm. Information and Computation, 108(2):212–261, 1994.

[30] H. Luo and R. E. Schapire. Towards minimax online learning with unknown time horizon. In Proceedings of the 31st International Conference on Machine Learning (ICML 2014), volume 32 of Proceedings of Machine Learning Research, pages 226–234, 2014.

[31] Y. Peres, O. Schramm, S. Shefield, and D. B. Wilson. Tug-of-war and the infinity Laplacian. Journal of the American Mathematical Society, 22(1):167–210, 2009.

[32] Y. Peres and S. Shefield. Tug-of-war with noise: a game-theoretic view of the p-Laplacian. Duke Mathematical Journal, 145(1):91–120, 2008.

[33] V. G. Vovk. Aggregating strategies. In Proceedings of the Third Annual Workshop on Computational Learning Theory (COLT 1990), pages 371–386. Morgan Kaufmann, 1990.

[34] R. J. Williams. Semimartingale reflecting Brownian motions in the orthant. In Stochastic Networks, volume 71 of IMA Volumes in Mathematics and its Applications, pages 125–137. Springer, New York, 1995.