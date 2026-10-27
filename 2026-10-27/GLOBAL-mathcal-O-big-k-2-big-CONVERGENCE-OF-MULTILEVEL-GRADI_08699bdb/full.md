# GLOBAL $\mathcal { O } \big ( k ^ { - 2 } \big )$ CONVERGENCE OF MULTILEVEL GRADIENT-REGULARIZED NEWTON METHODS

NICK TSIPINAKIS<sup>∗</sup> AND MATTHIAS VOIGT<sup>∗</sup>

Abstract. We develop and analyze multilevel gradient-regularized Newton methods for unconstrained convex optimization with Lipschitz-continuous Hessians. By computing Newton corrections in lower-dimensional subspaces, the proposed methods aim to reduce computational costs while preserving the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence rate in function values of the full-space method. To establish this rate for multilevel methods, we introduce a class of row-nested restriction operators that ensures monotonicity of the reduced-gradient norms across levels. Numerical experiments with row-nested operators demonstrate convergence behavior comparable to full-space gradient-regularized Newton, while ofering substantial computational savings on large-scale instances.

Key words. multilevel optimization, regularized Newton methods, convex optimization, rownested restriction operators

MSC codes. 90C25, 90C30, 65K05, 90C06

1. Introduction. Cubic regularization of Newton’s method [12, 23] is a fundamental second-order approach for smooth, possibly nonconvex optimization, achieving an $\mathcal { O } ( \varepsilon ^ { - 3 / 2 } )$ complexity for first-order stationarity in nonconvex problems and a global $\dot { \mathcal { O } } \big ( k ^ { - 2 } \big )$ function-value rate for convex objectives with Lipschitz-continuous Hessians. Subsequent developments include accelerated cubic Newton methods attaining an $\mathcal { O } ( k ^ { - 3 } )$ rate [21] and Adaptive Regularization with Cubics (ARC) [6–8]. However, cubic regularization requires solving a potentially expensive subproblem at each iteration.

Gradient-regularized Newton methods provide a simpler alternative, with iterations of the form

$$
x _ { k + 1 } = x _ { k } - \left( \nabla ^ { 2 } f ( x _ { k } ) + a _ { k } I \right) ^ { - 1 } \nabla f ( x _ { k } ) ,\tag{1.1}
$$

where $a _ { k } > 0$ . Following the early work of Polyak [24], Mishchenko [19] and Doikov and Nesterov [9] established that choosing $a _ { k }$ proportional to $\sqrt { L \| \nabla f ( x _ { k } ) \| }$ yields the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function-value rate for convex objectives with Lipschitz-continuous Hessians, improving upon the $\mathcal { O } ( k ^ { - 1 } )$ rate of standard gradient descent. Nevertheless, computing full-dimensional Hessians and solving the corresponding linear systems can be prohibitively expensive for large-scale problems.

Inspired by classical multigrid methods [4], multilevel optimization reduces these costs by computing corrections in lower-dimensional spaces, replacing full-dimensional Newton systems with smaller ones. Multilevel strategies have been successfully developed for Newton and regularized Newton methods [11, 14, 25, 26, 29], higher-order regularization methods, including ARC [5], as well as line-search, accelerated gradient, and proximal methods [1, 16, 30].

In a foundational work, Gratton, Sartenaer, and Toint [11] introduced recursive multilevel trust-region (RMTR) methods, establishing an $\mathcal { O } \big ( \bar { \varepsilon } ^ { - 2 } \big )$ worst-case complexity bound for first-order stationarity in possibly nonconvex problems. Their framework permits recursion to a coarser level when

$$
\| R _ { i } g _ { i , k } \| \ge \kappa _ { g } \| g _ { i , k } \| , \qquad \| R _ { i } g _ { i , k } \| > \varepsilon _ { i - 1 } ^ { g } ,\tag{1.2}
$$

where $g _ { i , k }$ is the gradient of the current level-dependent model, $\kappa _ { g } > 0$ is a prescribed parameter, and $R _ { i } \in \mathbb { R } ^ { n _ { i - 1 } \times n _ { i } }$ <sup>i</sup> restricts information to the next coarser level. Importantly, this eligibility condition is evaluated before computing a coarse correction, avoiding additional trial steps for level selection.

In general, second- or higher-order multilevel methods need not inherit the fast convergence rates of their full-space counterparts, since coarse corrections exploit only partial second- or higher-order information. Notable results preserving full-space convergence guarantees include RMTR [11] and the high-order multilevel methods of Calandra et al. [5], which attain the $\bar { \mathcal { O } } \big ( \varepsilon ^ { - 3 / 2 } \big )$ first-order stationarity complexity of cubic regularization for possibly nonconvex problems. More recently, for convex objectives, the multilevel gradient-regularized Newton method [27] was shown to achieve the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function-value rate, while the adaptive multilevel Newton method [28] achieves local quadratic convergence for strongly convex objectives.

However, the deterministic formulations of these last two methods rely on a stronger level-selection criterion, namely

$$
\begin{array} { r } { \| R _ { i } \nabla f ( x _ { k } ) \| \ge \sigma \left\| \nabla f ( x _ { k } ) \right\| , \qquad \left\| R _ { i } \nabla f ( x _ { k + 1 } ) \right\| \ge \sigma \left\| \nabla f ( x _ { k + 1 } ) \right\| . } \end{array}\tag{1.3}
$$

The second condition requires computing a trial coarse step to evaluate the gradient at the prospective next iterate, potentially increasing the computational cost. In contrast, the classical eligibility criterion (1.2) uses only current-iterate information. To the best of our knowledge, a direct global $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence analysis for deterministic multilevel gradient-regularized Newton methods employing such a pre-step level-selection criterion has remained unavailable. Establishing this rate will provide theoretical justification for employing multilevel methods in convex optimization, combining the fast convergence guarantees of full-space Newton methods with the potential computational savings of lower-dimensional corrections.

Contributions. We develop multilevel gradient-regularized Newton methods for convex objectives with Lipschitz-continuous Hessians, exploiting lower-dimensional subspaces and the classical level-selection criterion (1.2) (cf. [11]). We show that these methods preserve the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function-value rate of their full-space counterpart. For two levels, Theorem 2.5 establishes this rate under arbitrary coarse/fine alternation. For multiple levels with general restriction operators, Theorem 3.3 requires an explicit fine-iteration frequency condition, which is eventually satisfied by the algorithm (Subsection 3.1) but may be restrictive during coarse-dominated phases. To eliminate this condition, we introduce row-nested restriction operators (Subsection 3.2), whose structure guarantees monotonicity of reduced-gradient norms across levels. Exploiting this property, we establish the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate without imposing any fine-iteration frequency condition (Theorem 3.8).

Our analysis builds on the ideas of [19], but addresses a fundamental dificulty absent from full-space methods: coarse and fine corrections control diferent gradient norms, whose convergence properties cannot be directly transferred across levels. To overcome this, we introduce a level-dependent weighted gradient measure that provides uniform control across all possible coarse-to-fine, coarse-to-coarse, and fineto-coarse transitions. This, together with the proposed class of row-nested restriction operators (Definition 3.4), constitutes the main theoretical novelty of our analysis, enabling us to preserve the convergence rate of the full-space method without imposing any condition on the frequency of fine-level iterations.

We support our theoretical analysis with numerical experiments comparing the proposed methods against the full-space gradient-regularized Newton method. The results demonstrate that the two-level and row-nested multilevel methods exhibit convergence behavior comparable to that of their full-dimensional counterpart, while achieving substantial reductions in computational time on large-scale instances. Furthermore, the experiments highlight the importance of the restriction operator structure, with row-nested hierarchies generally exhibiting more favorable convergence behavior than their non-nested counterparts. These findings suggest that row-nested restriction operators can ofer an attractive choice from both theoretical and computational perspectives.

1.1. Related work. Early multilevel optimization frameworks were developed by Nash [20] and Lewis and Nash [18], including implementations based on truncated-Newton methods. A common feature of subsequent multilevel methods [5, 11, 13, 14] is the use of coarse-level eligibility conditions of the form (1.2). Wen and Goldfarb [30] developed a line-search multigrid method establishing R-linear convergence under strong convexity and suitable conditions on the search directions. Calandra et al. [5] extended higher-order regularization to multilevel optimization. Other multilevel Newton-type methods [14, 25, 29] establish global convergence or fast local rates under specific assumptions. However, unlike our approach, these Newton-type methods do not employ gradient-dependent Hessian regularization to handle potentially singular Hessians, nor do they establish the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function-value rate for general convex objectives.

Tsipinakis and Parpas [27] developed the multilevel gradient-regularized Newton method achieving the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function-value rate for convex objectives. However, unlike the methods proposed in this paper, their method requires the look-ahead condition (1.3), introducing additional computational costs. This condition avoids directly analyzing transitions between coarse and fine gradient measures and thus their proof cannot be applied to our setting. In contrast, our analysis preserves the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate under classical level selection without requiring look-ahead trial steps.

Nested subspaces are widely used in Krylov-based trust-region methods, reducedbasis approximation, and multilevel optimization [2, 10, 13]. The row-nested restriction operators considered here additionally guarantee monotonicity of the restrictedgradient norms across levels. Moreover, the weighted-gradient analysis is closely related to multiple-Lyapunov-function techniques for switched systems [3, 31]. In our setting, we combine a suitably weighted gradient measure with regularized Newton decrease bounds to establish a global $\mathcal { O } \big ( k ^ { - 2 } \big )$  convergence rate in function values under arbitrary coarse/fine alternation.

1.2. Main assumptions and notation. We consider the unconstrained optimization problem

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { n } } f ( x ) ,
$$

where $f : \mathbb { R } ^ { n }  \mathbb { R }$ is a twice continuously diferentiable convex function. Given an initial point $x _ { 0 } \in \mathbb { R } ^ { n }$ , our goal is to develop iterative methods achieving the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence rate in function values. Throughout, we impose the following assumptions.

Assumption 1.1. The initial sublevel set $\mathcal { L } ( x _ { 0 } ) : = \{ x \in \mathbb { R } ^ { n } : f ( x ) \leq f ( x _ { 0 } ) \}$ is bounded.

Since $f$ is continuous, $\mathcal { L } ( \boldsymbol { x } _ { 0 } )$ is compact, ensuring the existence of a global minimizer $x ^ { * } \in \mathbb { R } ^ { n }$ . We denote $f ^ { * } : = f ( x ^ { * } )$

Assumption 1.2. There exists an $L > 0$ such that

$$
\left\| \nabla ^ { 2 } f ( x ) - \nabla ^ { 2 } f ( y ) \right\| \leq L \left\| x - y \right\| \qquad \forall x , y \in \mathbb { R } ^ { n } .
$$

These assumptions are standard in the analysis of second-order optimization methods; see, e.g., [22]. Throughout, ∥·∥ denotes the Euclidean norm for vectors and the induced spectral norm for matrices. For convenience, we write

$$
\nabla f _ { k } : = \nabla f ( x _ { k } ) , \qquad \nabla ^ { 2 } f _ { k } : = \nabla ^ { 2 } f ( x _ { k } ) .
$$

Moreover, by compactness of $\mathcal { L } ( x _ { 0 } )$ , the constants

$$
H : = \operatorname* { s u p } _ { x \in \mathcal { L } ( x _ { 0 } ) } \left\| \nabla ^ { 2 } f ( x ) \right\| , \qquad D : = \operatorname* { s u p } _ { x \in \mathcal { L } ( x _ { 0 } ) } \| x - x ^ { * } \|\tag{1.4}
$$

are finite.

Organization. In Section 2, we introduce the two-level gradient-regularized Newton method and establish its convergence rate. In Section 3, we extend the analysis to multiple coarse levels, establishing convergence guarantees under a fine-iteration frequency condition and, alternatively, without this condition using row-nested restriction operators introduced in Subsection 3.2. Numerical experiments comparing the proposed methods with full-space gradient-regularized Newton are presented in Section 4. Finally, Section 5 concludes the paper.

2. The two-level Newton method with gradient regularization. We consider the Newton method with gradient regularization that alternates between the fine and one coarse level. For the two-level algorithm we consider a single restriction operator that transfers the information from fine to coarse space, i.e., $R : \mathbb { R } ^ { n }  \mathbb { R } ^ { d }$ ， where $d < n$ are the coarse space dimensions. Moreover, the prolongation operator is defined in a standard way, i.e., $P : = R ^ { \top }$ . For the operators we take the standard assumptions (see for instance [11, 14]).

Assumption 2.1. Let $R \in \mathbb { R } ^ { d \times n }$ be a full-row-rank restriction operator, with $d \leq$ n, and let $P = R ^ { \top }$ be its corresponding prolongation operator. We define

$$
\omega : = \left\| R \right\| .
$$

Let $\sigma \in ( 0 , \operatorname* { m i n } \{ 1 , \omega \} )$ and $\varepsilon > 0$ . We define the sets of coarse and fine iterations

$$
\mathcal { C } : = \{ k \in { \mathbb { N } } : \| R \nabla f _ { k } \| \geq \sigma \| \nabla f _ { k } \| , \| R \nabla f _ { k } \| \geq \varepsilon \} , \quad \mathcal { F } : = { \mathbb { N } } \setminus \mathcal { C } .
$$

We can consider C as the set of efective coarse iterations and $\mathcal { F }$ the set of fine iterations. Thus, if $m \in { \mathcal { C } }$ then the algorithm will perform a coarse iteration otherwise a fine iteration. The two-level gradient-regularized Newton algorithm is presented in Algorithm 2.1.

The following result will be needed for the analysis of Algorithm 2.1; it has been proved in [27, Lemma 3.3] and [19, Lemma 3].

Lemma 2.1 ([19, 27]). Suppose that Assumptions 1.2 and 2.1 hold. Let $x _ { k } \in \mathbb { R } ^ { n }$ and suppose that $x _ { k + 1 }$ is obtained by either the coarse or fine update of Algorithm 2.1, whenever the corresponding update is well-defined. Then

$$
f ( x _ { k } ) - f ( x _ { k + 1 } ) \geq { \frac { \| R \nabla f _ { k + 1 } \| ^ { 2 } } { 8 { \sqrt { \omega ^ { 3 } L \left\| R \nabla f _ { k } \right\| / 2 } } } }\tag{2.1}
$$

for a coarse update,

$$
f ( x _ { k } ) - f ( x _ { k + 1 } ) \geq { \frac { \| \nabla f _ { k + 1 } \| ^ { 2 } } { 6 { \sqrt { L \left\| \nabla f _ { k } \right\| / 2 } } } }\tag{2.2}
$$

for a fine update.

```latex
Algorithm 2.1 Two-Level Regularized Newton (TLRN)
Inputs: $x _ { 0 } \in \mathbb { R } ^ { n } , R : \mathbb { R } ^ { n } \to \mathbb { R } ^ { d } , \sigma \in ( 0 , \operatorname* { m i n } \{ 1 , \omega \} ) , \varepsilon > 0 .$
Outputs: Sequence of iterates (x<sub>0</sub>, x<sub>1</sub>, x<sub>2</sub>, . . .).
for k = 0, 1, . . . do
if $\nabla f _ { k } = 0$ then
stop
end if
Set
$x _ { k + 1 } : = \left\{ \begin{array} { l l } { x _ { k } - P \left( R \nabla ^ { 2 } f _ { k } P + \sqrt { \frac { \omega ^ { 3 } L \| R \nabla f _ { k } \| } { 2 } } I _ { d } \right) ^ { - 1 } R \nabla f _ { k } , } & { k \in \mathcal { C } , } \\ { x _ { k } - \left( \nabla ^ { 2 } f _ { k } + \sqrt { \frac { L \| \nabla f _ { k } \| } { 2 } } I _ { n } \right) ^ { - 1 } \nabla f _ { k } , } & { k \in \mathcal { F } . } \end{array} \right.$
end for
```

Corollary 2.2. Suppose that Assumptions 1.2 and 2.1 hold and that the sequence $( x _ { k } ) _ { k \in \mathbb { N } }$ is generated by Algorithm 2.1. Then the sequence of function values $( f ( x _ { k } ) ) _ { k \in \mathbb { N } }$ is non-increasing, i.e.,

$$
f ( x _ { k + 1 } ) \leq f ( x _ { k } ) , \qquad k \in \mathbb { N } .
$$

The following stability estimates for the coarse and fine corrections were also established in [27] and [19], and hold for all coarse and fine updates, respectively:

(2.3)

$$
\begin{array} { r l } { \left\| R \nabla f _ { k + 1 } \right\| \leq 2 \left\| R \nabla f _ { k } \right\| \quad } & { \mathrm { f o r ~ a ~ c o a r s e ~ u p d a t e } , } \\ { \left\| \nabla f _ { k + 1 } \right\| \leq 2 \left\| \nabla f _ { k } \right\| \quad } & { \mathrm { f o r ~ a ~ f i n e ~ u p d a t e } . } \end{array}\tag{2.4}
$$

2.1. Convergence analysis. When ${ \mathcal { C } } = \emptyset$ , then Algorithm 2.1 coincides with the gradient regularization of Newton’s method (see [9, 19]) which converges with the fast $\mathcal { O } ( k ^ { - 2 } )$ rate. In what follows we show that this rate can be attained when the sequence $( \dot { x } _ { k } ) _ { k \in \mathbb { N } }$ is always generated from the coarse space, i.e.,

$$
x _ { k + 1 } = x _ { k } - P \left( R \nabla ^ { 2 } f _ { k } P + \sqrt { \frac { \omega ^ { 3 } L \left\| R \nabla f _ { k } \right\| } { 2 } } I _ { d } \right) ^ { - 1 } R \nabla f _ { k } .
$$

For the purposes of this result, the condition $\| R \nabla f _ { k } \| \ge \varepsilon$ is therefore not imposed.

Proposition 2.3. Suppose that Assumptions 1.1 and 1.2 hold, and let $( x _ { k } ) _ { k \in \mathbb { N } }$ be generated by applying the coarse update of Algorithm 2.1 at every iteration. Assume that

$$
\begin{array} { r } { \| R \nabla f _ { k } \| \ge \sigma \| \nabla f _ { k } \| , \qquad k \in \mathbb { N } . } \end{array}
$$

Then

$$
f ( x _ { k } ) - f ^ { * } = { \mathcal { O } } { \left( k ^ { - 2 } \right) } .
$$

Moreover, the constant in the polynomial bound is proportional to $\sigma ^ { - 3 }$

Proof. Fix $k \in \mathbb N$ and set $\begin{array} { r } { I ^ { k } : = \left\{ m \in \{ 0 , \ldots , k - 1 \} : \| R \nabla f _ { m + 1 } \| > \frac { 1 } { 4 } \| R \nabla f _ { m } \| \right\} } \end{array}$ We distinguish two cases.

Case I: $\vert I ^ { k } \vert \le k / 2$ . By the definition of $I ^ { k }$ and (2.3),

$$
\| R \nabla f _ { m + 1 } \| \leq \left\{ { \frac { 1 } { 4 } } \| R \nabla f _ { m } \| , \quad m \notin I ^ { k } , \right.
$$

Therefore,

$$
\| R \nabla f _ { k } \| \le \frac { 2 ^ { \left| I ^ { k } \right| } } { 4 ^ { k - | I ^ { k } | } } \| R \nabla f _ { 0 } \| \le 2 ^ { - k / 2 } \| R \nabla f _ { 0 } \| .
$$

Using convexity and the coarse-level condition,

$$
f ( \boldsymbol { x } _ { k } ) - f ^ { * } \leq D \| \nabla f _ { k } \| \leq \frac { D } { \sigma } \| R \nabla f _ { k } \| \leq \frac { \omega D } { \sigma } 2 ^ { - k / 2 } \| \nabla f _ { 0 } \| .
$$

Thus, in this case the function value gap decreases geometrically and, in particular, is $\mathcal { O } \big ( k ^ { - 2 } \big )$

Case II: $\vert I ^ { k } \vert > k / 2$ . Let $\tau : = | I ^ { k } |$ , enumerate $I ^ { k } = \{ k _ { 0 } , \ldots , k _ { \tau - 1 } \} , \ k _ { 0 } < \cdots <$ $k _ { \tau - 1 } < k$ , and set $k _ { \tau } : = k$ . By (2.1) and the definition of $I ^ { k }$ , for $\ell = 0 , \dots , \tau - 1$ ，

$$
f ( x _ { k _ { \ell } } ) - f ( x _ { k _ { \ell } + 1 } ) \geq \frac { \| R \nabla f _ { k _ { \ell } + 1 } \| ^ { 2 } } { 8 ( \omega ^ { 3 } L \| R \nabla f _ { k _ { \ell } } \| / 2 ) ^ { 1 / 2 } } \geq \frac { \| R \nabla f _ { k _ { \ell } } \| ^ { 3 / 2 } } { 1 2 8 ( \omega ^ { 3 } L / 2 ) ^ { 1 / 2 } } \geq \frac { \sigma ^ { 3 / 2 } \| \nabla f _ { k _ { \ell } } \| ^ { 3 / 2 } } { 1 2 8 ( \omega ^ { 3 } L / 2 ) ^ { 1 / 2 } } .
$$

Since $I ^ { k } \subseteq \{ 0 , \ldots , k - 1 \}$ , it follows that $k _ { \ell } + 1 \leq k _ { \ell + 1 }$ for $\ell = 0 , \dots , \tau - 1$ . Moreover, by (2.1), the sequence $( f ( x _ { j } ) ) _ { j \in \mathbb { N } }$ is non-increasing. Hence,

$$
f ( x _ { k _ { \ell } } ) - f ( x _ { k _ { \ell + 1 } } ) \geq { \frac { \sigma ^ { 3 / 2 } } { 1 2 8 ( \omega ^ { 3 } L / 2 ) ^ { 1 / 2 } } } \| \nabla f _ { k _ { \ell } } \| ^ { 3 / 2 } .
$$

Combining this with convexity, i.e., $f ( x _ { k _ { \ell } } ) - f ^ { * } \leq \left. \nabla f _ { k _ { \ell } } , x _ { k _ { \ell } } - x ^ { * } \right. \leq D \| \nabla f _ { k _ { \ell } } \|$ , we obtain

$$
f ( x _ { k _ { \ell + 1 } } ) - f ^ { * } \leq f ( x _ { k _ { \ell } } ) - f ^ { * } - C _ { \mathcal { C } } { \big ( } f ( x _ { k _ { \ell } } ) - f ^ { * } { \big ) } ^ { 3 / 2 } ,
$$

where

$$
C _ { \cal C } : = \frac { \sigma ^ { 3 / 2 } } { 1 2 8 ( D ^ { 3 } \omega ^ { 3 } L / 2 ) ^ { 1 / 2 } } .
$$

Hence, Proposition A.1 yields

$$
f ( x _ { k } ) - f ^ { * } \leq \frac { 1 } { C _ { \mathcal { C } } ^ { 2 } ( 1 + \tau / 2 ) ^ { 2 } } \leq \frac { 1 } { C _ { \mathcal { C } } ^ { 2 } ( 1 + k / 4 ) ^ { 2 } } = \mathcal { O }  { \left( \frac { 1 } { \sigma ^ { 3 } k ^ { 2 } } \right) } .
$$

Combining the two cases proves the result.

Proposition 2.3 analyzes the limiting case in which only coarse corrections are performed and shows that the coarse-only iteration retains the $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate of the full-space regularized Newton method. Its proof adapts the strategy of [19, Theorem 1] to the reduced-gradient setting. A key feature of this analysis is that every iteration controls the same quantity, namely $\| R \nabla f _ { k } \|$ , so the corresponding estimates can be propagated consistently along the sequence.

The situation is fundamentally diferent for Algorithm 2.1. Coarse iterations naturally control $\| R \nabla f _ { k } \|$ , whereas fine iterations control $\| \nabla f _ { k } \|$ , and neither of these quantities is necessarily contractive when the algorithm switches to the other level. Thus, the fact that both the coarse-only and fine-only schemes achieve an $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate does not by itself imply convergence, or the same rate, for an arbitrary mixture of the two iteration types. The main dificulty is therefore not the behavior within either level, but the transitions between them.

The following lemma provides the cross-level stability estimate needed to bridge this mismatch: although a coarse correction need not decrease the full gradient norm, it can increase it by at most a uniform factor.

Lemma 2.4. Suppose that Assumptions 1.1, 1.2, and 2.1 hold and that $( x _ { k } ) _ { k \in \mathbb { N } }$ is generated by Algorithm 2.1. Let

$$
M _ { \mathcal { C } } : = 1 + \frac { H \sqrt { 2 \omega } } { \sqrt { L \varepsilon } } ,
$$

where H is defined in (1.4). Then, for any $k \in { \mathcal { C } }$ we have

$$
\| \nabla f _ { k + 1 } \| \le M _ { \mathcal { C } } \| \nabla f _ { k } \|
$$

Proof. Denote

$$
\alpha _ { k } : = \sqrt { \frac { \omega ^ { 3 } L \left. R \nabla f _ { k } \right. } { 2 } } .
$$

Next, we have

$$
\nabla f ( x _ { k + 1 } ) - \nabla f ( x _ { k } ) = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } f { \big ( } x _ { k } + t ( x _ { k + 1 } - x _ { k } ) { \big ) } { \big ( } x _ { k + 1 } - x _ { k } { \big ) } \mathrm { d } t .
$$

By Corollary 2.2, $x _ { k } , x _ { k + 1 } \in \mathcal L ( x _ { 0 } )$ . Since f is convex, $\mathcal { L } ( \boldsymbol { x } _ { 0 } )$ is convex, and hence $x _ { k } + t ( x _ { k + 1 } - x _ { k } ) \in \mathcal L ( x _ { 0 } )$ for every $t \in [ 0 , 1 ]$ . Therefore,

$$
\| \nabla f ( x _ { k + 1 } ) - \nabla f ( x _ { k } ) \| \leq H \| x _ { k + 1 } - x _ { k } \| .\tag{2.5}
$$

Then, by using the definitions of ω and $\alpha _ { k }$ as well as $k \in { \mathcal { C } }$ and the symmetric positive semi-definiteness of $R \nabla ^ { 2 } f _ { k } P$ , we obtain

$$
\begin{array} { r l } & { \| \nabla f _ { k + 1 } \| \leq \| \nabla f _ { k + 1 } - \nabla f _ { k } \| + \| \nabla f _ { k } \| } \\ & { \qquad \leq H \left\| P \left( R \nabla ^ { 2 } f _ { k } P + \sqrt { \frac { \omega ^ { 2 } L } { 2 } \| R \nabla f _ { k } \| } _ { { \boldsymbol { J } } } _ { t } \right) ^ { - 1 } R \nabla f _ { k } \right\| + \| \nabla f _ { k } \| } \\ & { \qquad \leq { I I } \left\| \left( R \nabla ^ { 2 } f _ { k } P + \sqrt { \frac { \mu \Delta _ { { \boldsymbol { J } } } } { 2 } \| R \nabla f _ { k } \| } _ { { \boldsymbol { J } } } _ { t } \right) ^ { - 1 } R \nabla f _ { k } \right\| + \| \nabla f _ { k } \| } \\ & { \qquad \leq { I I } \left\| \frac { \| { \boldsymbol { J } } \nabla f _ { k + 1 } \| } { \alpha _ { k } } + \| \nabla f _ { k } \| \right. } \\ & { \qquad \leq { I I } \frac { \| { \boldsymbol { J } } \nabla f _ { k + 1 } \| } { \sqrt { \alpha _ { k } ^ { 3 } L \varepsilon } } \| { I } \nabla f _ { k } \| + \| \nabla f _ { k } \| } \\ & { \qquad \leq \left( 1 + \frac { H \sqrt { 2 \omega _ { k } } } { \sqrt { L \varepsilon } } \right) \| \nabla f _ { k } \| . } \end{array}
$$

Lemma 2.4 shows that a coarse correction is stable with respect to the full gradi ent, in the sense that the gradient norm can increase by at most a uniformly bounded

factor. Together with the level-specific stability estimates (2.1) and (2.2), this bound will allow us to construct a level-dependent weighted gradient measure that controls transitions between the two levels.

Theorem 2.5. Let f be convex and suppose that Assumptions 1.1, 1.2, and 2.1 hold. Then the sequence generated by Algorithm 2.1 satisfies

$$
f ( x _ { k } ) - f ^ { * } = \mathcal { O } \left( \frac { 1 } { k ^ { 2 } } \right) .
$$

Proof. To compare the reduced and full gradient norms across changes of level, we introduce a level-dependent weighted gradient measure. The weights are chosen so that all four possible coarse/fine transitions admit uniform contraction or stability bounds. Choose

$$
a > 0 , \qquad \rho \in \left( 0 , \frac { \sigma } { \omega M _ { C } } \right) , \qquad t \in \left( \rho \omega , \frac { \sigma } { M _ { C } } \right) ,
$$

and set $b : = a t$ . Since $\| R \| \| \nabla f _ { k } \| \ge \| R \nabla f _ { k } \| \ge \sigma \| \nabla f _ { k } \|$ implies $\omega \geq \sigma$ , we get

$$
q : = \operatorname* { m a x } \left\{ \rho , \frac { t M _ { \mathcal { C } } } { \sigma } , \frac { \omega \rho } { t } \right\} < 1 .
$$

Moreover, define the scalar

$$
B : = \operatorname* { m a x } \left\{ 2 , \frac { 2 \omega } { t } , \frac { t M _ { \mathscr { C } } } { \sigma } \right\} ,
$$

and the sets

$$
\begin{array} { r } { \mathcal { C } ^ { k } : = \mathcal { C } \cap \{ 0 , \dotsc , k - 1 \} , \qquad \mathcal { F } ^ { k } : = \mathcal { F } \cap \{ 0 , \dotsc , k - 1 \} , } \end{array}
$$

$$
\mathcal { Z } _ { \mathcal { C } } ^ { k } ( \rho ) : = \{ m \in \mathcal { C } \cap \{ 0 , \ldots , k - 1 \} : \| R \nabla f _ { m + 1 } \| > \rho \| R \nabla f _ { m } \| \} ,
$$

$$
\begin{array} { r } { \mathcal { Z } _ { \mathcal { F } } ^ { k } ( { \boldsymbol \rho } ) : = \{ m \in \mathcal { F } \cap \{ 0 , \dots , k - 1 \} : \ \| \nabla f _ { m + 1 } \| > \rho \| \nabla f _ { m } \| \} . } \end{array}
$$

We set

$$
\begin{array} { r } { \mathcal { T } ^ { k } ( \rho ) : = \mathcal { T } _ { \mathcal { C } } ^ { k } ( \rho ) \cup \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho ) . } \end{array}
$$

Now, choose $\begin{array} { r } { \theta \in \left( 0 , \frac { - \ln q } { \ln B - \ln q } \right) } \end{array}$ and fix a $k \in \mathbb N$ . We distinguish two cases.

Case I: $\left. \mathcal { T } ^ { k } ( \rho ) \right. \le \theta k$ . Define

$$
V _ { k } : = \left\{ \begin{array} { l l } { a \| R \nabla f _ { k } \| , } & { k \in \mathcal { C } , } \\ { b \| \nabla f _ { k } \| , } & { k \in \mathcal { F } , } \end{array} \right.
$$

For $m \in ( \mathcal { Z } ^ { k } ( \rho ) ) ^ { \mathrm { c } } : = \{ 0 , \ldots , k - 1 \} \setminus \mathbb { Z } ^ { k } ( \rho ) = ( \mathcal { C } ^ { k } \setminus \mathcal { Z } _ { \mathcal { C } } ^ { k } ( \rho ) ) \cup ( \mathcal { F } ^ { k } \setminus \mathcal { Z } _ { \mathcal { F } } ^ { k } ( \rho ) )$ , we show that $V _ { m + 1 } \leq q V _ { m } .$ . Let $m \in \mathcal { T } ^ { k } ( \rho ) ^ { \mathrm { c } }$ . Then, for m and $m + 1$ we have four possibilities:

(a) $m \in \mathcal { C } ^ { k } \setminus \mathcal { T } _ { \mathcal { C } } ^ { k } ( \rho ) , \ m + 1 \in \mathcal { C } .$ . Then,

$$
{ V _ { m + 1 } } \stackrel { m + 1 } { = } \stackrel { \in } a \left\| R \nabla f _ { m + 1 } \right\| \stackrel { m \in \mathcal { C } \backslash \mathcal { T } _ { c } ^ { k } ( \rho ) } { \leq } a \rho \left\| R \nabla f _ { m } \right\| \stackrel { m \in \mathcal { C } } { = } \rho V _ { m } .
$$

(b) $m \in \mathcal { C } ^ { k } \setminus \mathcal { T } _ { \mathcal { C } } ^ { k } ( \rho ) , m + 1 \in \mathcal { F }$ . Then, with similar arguments we obtain

$$
\begin{array} { r l } & { { \cal V } _ { m + 1 } \stackrel { m + 1 \in { \mathcal F } } { = } b \left\| \nabla f _ { m + 1 } \right\| \stackrel { \mathrm { L e m . ~ } ^ { 2 . 4 } } { \leq } b M _ { \mathcal { C } } \left\| \nabla f _ { m } \right\| } \\ & { \quad \leq \displaystyle \frac { b M _ { \mathcal { C } } } { \sigma } \left\| R \nabla f _ { m } \right\| \stackrel { m \in { \mathcal C } } { \leq } \frac { t M _ { \mathcal { C } } } { \sigma } { \cal V } _ { m } . } \end{array}
$$

(c) $m \in \mathcal { F } ^ { k } \setminus \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho ) , m + 1 \in \mathcal { F }$ . Then,

$$
V _ { m + 1 } \stackrel { m + 1 \in \mathcal { F } } { = } b \left\| \nabla f _ { m + 1 } \right\| \stackrel { m \in \mathcal { F } \backslash \mathcal { Z } _ { \mathcal { F } } ^ { k } ( \rho ) } { \leq } b \rho \left\| \nabla f _ { m } \right\| \stackrel { m \in \mathcal { F } } { = } \rho V _ { m } .
$$

(d) m ∈ F<sup>k</sup> \ I<sup>k</sup> (ρ), m + 1 ∈ C. Then,

$$
{ V _ { m + 1 } } ^ { m + 1 } \stackrel {  } { = } { \boldsymbol { \varepsilon } } \boldsymbol { c } _ { a } \| { \boldsymbol { R } } \nabla f _ { m + 1 } \| \leq a \omega \| \nabla f _ { m + 1 } \| \stackrel { m \in \mathcal { F } \backslash \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho ) } { \leq } a \omega \rho \| \nabla f _ { m } \| \stackrel { m \in \mathcal { F } } { = } \frac { \omega \rho } { t } V _ { m } .
$$

Hence,

$$
V _ { m + 1 } \leq q V _ { m } \mathrm { f o r \ a n y } m \in \mathbb { Z } ^ { k } ( \rho ) ^ { \mathrm { c } } .\tag{2.6}
$$

Now let $m \in \mathcal { T } ^ { k } ( \rho )$ . We again consider four cases as above:

(a) $m \in \mathcal { T } _ { \mathcal { C } } ^ { k } ( \rho ) , \ m + 1 \in \mathcal { C } .$ Then,

$$
V _ { m + 1 } = a \| R \nabla f _ { m + 1 } \| \stackrel { ( 2 . 3 ) } { \leq } 2 V _ { m } .
$$

(b) $m \in \mathcal { T } _ { \mathcal { C } } ^ { k } ( \rho ) , m + 1 \in \mathcal { F }$ . Then,

$$
V _ { m + 1 } \leq \frac { t M _ { \cal C } } { \sigma } V _ { m } .
$$

(c) $n \in \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho ) , \ m + 1 \in \mathcal { F }$ . Then,

$$
V _ { m + 1 } = b \| \nabla f _ { m + 1 } \| \overset { ( 2 . 4 ) } { \leq } 2 V _ { m } .
$$

(d) $m \in \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho ) , \ m + 1 \in \mathcal { C }$ . Then,

$$
V _ { m + 1 } \leq \frac { 2 \omega } { t } V _ { m } .
$$

As a result,

$$
V _ { m + 1 } \leq B V _ { m } \mathrm { f o r ~ a n y } m \in \mathcal { T } ^ { k } ( \rho ) .\tag{2.7}
$$

Let $\eta : = q ^ { 1 - \theta } B ^ { \theta }$ . Thus, we have

$$
\ln \left( q ^ { 1 - \theta } B ^ { \theta } \right) = ( 1 - \theta ) \ln q + \theta \ln B = \ln q + \theta ( \ln B - \ln q ) ,
$$

and hence, by the definition of $\theta , \eta < 1$ . Combining now (2.6) and (2.7) we obtain

$$
V _ { k } \leq q ^ { k - \left| \mathcal { T } ^ { k } \left( \rho \right) \right| } B ^ { \left| \mathcal { T } ^ { k } \left( \rho \right) \right| } V _ { 0 } \leq \eta ^ { k } V _ { 0 } .\tag{2.8}
$$

Furthermore, for any $k \in \mathbb N$ , we have that

$$
\left\| \nabla f _ { k } \right\| \leq A V _ { k } , \quad A : = \operatorname* { m a x } \left\{ { \frac { 1 } { a \sigma } } , { \frac { 1 } { b } } \right\} .\tag{2.9}
$$

Then, by convexity and Assumption 1.1, we obtain

$$
f ( x _ { k } ) - f ^ { * } \leq D \left\| \nabla f _ { k } \right\| \overset { ( 2 . 8 ) , ( 2 . 9 ) } { \leq } \eta ^ { k } A D V _ { 0 } = O \left( \frac { 1 } { k ^ { 2 } } \right) ,\tag{2.10}
$$

and thus the desired rate of convergence is established when $\left. \mathcal { T } ^ { k } ( \rho ) \right. \le \theta k$

Case II: $\left| \mathcal { T } ^ { k } ( \rho ) \right| > \theta k$ . Let $\tau : = | \mathcal { I } ^ { k } ( \rho ) |$ and enumerate $\mathcal { T } ^ { k } ( \rho ) = \{ k _ { 0 } , \dots k _ { \tau - 1 } \}$ with $k _ { 0 } < k _ { 1 } < \dots < k _ { \tau - 1 } < k , ~ k _ { \tau } : = k$ . Let $\ell \in \{ 0 , \ldots , \tau - 1 \}$ . Thus, $k _ { \ell } \in \mathcal { T } ^ { k } ( \rho )$ . If $k _ { \ell } \in \mathcal { T } _ { C } ^ { k } ( \rho )$ , then

$$
\begin{array} { r l } & { f ( x _ { k _ { \ell } } ) - f ( x _ { k _ { \ell } + 1 } ) \overset { { ( 2 . 1 ) } } { \geq } \displaystyle \frac { \| R \nabla f _ { k _ { \ell } + 1 } \| ^ { 2 } } { 8 \left( \omega ^ { 3 } L / 2 \| R \nabla f _ { k _ { \ell } } \| \right) ^ { 1 / 2 } } \overset { k _ { \ell } \in \mathcal { Z } _ { c } ^ { k } ( \rho ) } { \geq } \displaystyle \frac { \rho ^ { 2 } \| R \nabla f _ { k _ { \ell } } \| ^ { 3 / 2 } } { 8 \left( \omega ^ { 3 } L / 2 \right) ^ { 1 / 2 } } } \\ & { \qquad \overset { k _ { \ell } \in \mathcal { C } } { \geq } \left( \frac { \sigma } { \omega } \right) ^ { 3 / 2 } \displaystyle \frac { \rho ^ { 2 } \| \nabla f _ { k _ { \ell } } \| ^ { 3 / 2 } } { 8 \sqrt { L / 2 } } . } \end{array}
$$

Using similar arguments, if $k _ { \ell } \in \mathcal { T } _ { \mathcal { F } } ^ { k } ( \rho )$ , then

$$
f ( x _ { k _ { \ell } } ) - f ( x _ { k _ { \ell } + 1 } ) \geq \frac { \rho ^ { 2 } \left. \nabla f _ { k _ { \ell } } \right. ^ { 3 / 2 } } { 6 \sqrt { L / 2 } } ,
$$

and thus, since $\sigma \leq \omega$ , we get

$$
\begin{array} { c } { { f ( x _ { k _ { \ell } } ) - f ( x _ { k _ { \ell } + 1 } ) \geq \operatorname* { m i n } \left\{ \left( \displaystyle \frac { \sigma } { \omega } \right) ^ { 3 / 2 } \displaystyle \frac { \rho ^ { 2 } } { 8 \sqrt { L / 2 } } , \displaystyle \frac { \rho ^ { 2 } } { 6 \sqrt { L / 2 } } \right\} | | \nabla f _ { k _ { \ell } } | | ^ { 3 / 2 } } } \\ { { = \left( \displaystyle \frac { \sigma } { \omega } \right) ^ { 3 / 2 } \displaystyle \frac { \rho ^ { 2 } | | \nabla f _ { k _ { \ell } } | | ^ { 3 / 2 } } { 8 \sqrt { L / 2 } } } } \end{array}
$$

for all $\ell \in \{ 0 , \ldots , \tau - 1 \}$ . Since $I ^ { k } ( \rho ) \subseteq \{ 0 , \ldots , k - 1 \}$ , it follows that $k _ { \ell } + 1 \le k _ { \ell + 1 }$ for $\ell = 0 , \dots , \tau - 1$ . Following the same argument as in Case II of Proposition 2.3, monotonicity of the function values and convexity yield

$$
\Delta _ { \ell + 1 } \leq \Delta _ { \ell } - C \Delta _ { \ell } ^ { 3 / 2 } , \qquad \Delta _ { \ell } : = f ( x _ { k _ { \ell } } ) - f ^ { * } ,
$$

where

$$
C : = \frac { \sqrt { 2 } \rho ^ { 2 } } { 8 \sqrt { L } } \left( \frac { \sigma } { \omega D } \right) ^ { 3 / 2 } .
$$

Hence, Proposition A.1 and $\tau > \theta k$ give

$$
f ( x _ { k } ) - f ^ { * } = \Delta _ { \tau } \leq \frac { 1 } { C ^ { 2 } ( 1 + \tau / 2 ) ^ { 2 } } \leq \frac { 1 } { C ^ { 2 } ( 1 + \theta k / 2 ) ^ { 2 } } = \mathcal { O } \big ( k ^ { - 2 } \big ) ,
$$

which completes the proof of Case II.

Combining now the results of the two cases we have

$$
f ( x _ { k } ) - f ^ { * } \leq \operatorname* { m a x } \bigg \{ A D V _ { 0 } \eta ^ { k } , \frac { 1 } { C ^ { 2 } ( 1 + \theta k / 2 ) ^ { 2 } } \bigg \} = \mathcal { O } \big ( k ^ { - 2 } \big ) ,
$$

as required.

Theorem 2.5 shows that the use of coarse corrections does not deteriorate the global convergence rate of the regularized Newton method. In particular, Algorithm 2.1 achieves the $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate in function values while allowing arbitrary switching between coarse and fine iterations according to the level-selection criterion. A key feature of the analysis is that coarse and fine iterations naturally control diferent quantities, namely $\| R \nabla f _ { k } \|$ and $\| \nabla f _ { k } \|$ , respectively. The proof resolves this mismatch through a level-dependent weighted gradient measure that provides uniform control across all possible coarse/fine transitions: it contracts on iterations outside $\mathcal { T } ^ { k } ( \rho )$ while its growth on the remaining iterations is uniformly bounded. Consequently, the analysis does not require the full gradient norm to decrease at every coarse iteration; uniform stability of the coarse corrections is suficient. This provides a mechanism for preserving convergence rates in multilevel schemes even when diferent iteration types control diferent measures of stationarity.

3. The multilevel Newton method with gradient regularization. In this section we consider the gradient regularization of Newton’s method that alternates between ℓ coarse levels (plus the fine level) when generating the updates. In particular, we consider ℓ restriction operators as in [11], but unlike their recursive framework, our operators map directly between each coarse level and the full space, allowing coarse corrections to be transferred directly to the fine level. More precisely, for each $i \in$ $\mathcal { I } : = \{ 1 , \ldots , \ell \}$ , let $R _ { i } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n _ { i } }$ be a restriction operator, where $n _ { 1 } < \cdots < n _ { \ell } < n$ Further, $P _ { i } : = R _ { i } ^ { \top }$ is the prolongation operator at level $i \in \mathcal { I }$ . We will see that this nonrecursive formulation simplifies both the algorithm and its convergence analysis while retaining the same $\mathcal { O } \bigl ( n _ { i } ^ { 2 } + n _ { i } ^ { 3 } \bigr )$ computational cost at the ith coarse level.

Accordingly, Assumption 2.1 is now replaced by the following assumption.

Assumption 3.1. For each $i \in \mathcal { I }$ , let $R _ { i } \in \mathbb { R } ^ { n _ { i } \times n }$ be a full-row-rank restriction operator, with $n _ { i } \leq n$ , and let $P _ { i } = R _ { i } ^ { \top }$ be its corresponding prolongation operator. We define

$$
\omega _ { i } : = \| R _ { i } \| , \qquad \zeta : = \operatorname* { m a x } _ { i \in \mathcal { T } } \omega _ { i } .
$$

Let $\sigma \in ( 0 .$ , min $\{ 1 , \operatorname* { m i n } _ { i \in \mathcal { I } } \omega _ { i } \} )$ and $\varepsilon > 0$ . For each $i \in \mathcal { I }$ , the sets of coarse and fine iterations are now defined as follows. First define the eligible sets

$$
\mathcal { E } _ { i } : = \left\{ k \in \mathbb { N } : \| R _ { i } \nabla f _ { k } \| \geq \sigma \| \nabla f _ { k } \| , \| R _ { i } \nabla f _ { k } \| \geq \varepsilon \right\} .
$$

The set $\mathcal { E } _ { i }$ contains the iterations for which level i is eligible. For the multilevel method to become eficient, we need to select the lowest-dimensional eligible level

$$
\mathcal { C } _ { i } : = \mathcal { E } _ { i } \setminus \bigcup _ { j < i } \mathcal { E } _ { j } , \qquad i \in \mathcal { I } .
$$

Then the sets $\mathcal { C } _ { i }$ are pairwise disjoint and, whenever $\{ i \in \mathcal { I } : k \in \mathcal { E } _ { i } \} \neq \emptyset$

$$
k \in { \mathcal { C } } _ { i } \quad \iff \quad i = i ^ { * } ( k ) : = \operatorname* { m i n } \{ j \in { \mathcal { I } } : k \in { \mathcal { E } } _ { j } \} .
$$

Moreover, we set

$$
{ \mathcal { F } } ^ { ( \ell ) } : = \mathbb { N } \setminus { \mathcal { C } } ^ { ( \ell ) } , \qquad { \mathcal { C } } ^ { ( \ell ) } : = \bigcup _ { i \in { \mathcal { I } } } { \mathcal { C } } _ { i } .
$$

The multilevel gradient-regularized Newton method is presented in Algorithm 3.1.

The next lemma shows that Algorithm 3.1 generates a non-increasing sequence of function values. It also shows that it guarantees the reduced-gradient stability of a coarse correction step. These results arise by [27].

Lemma 3.1. Suppose that Assumptions 1.2 and 3.1 hold, and let $( x _ { k } ) _ { k \in \mathbb { N } }$ be generated by Algorithm 3.1. Then, for any $k \in \mathcal { C } _ { i } , i \in \mathcal { I }$ 2

(3.1)

$$
f ( x _ { k } ) - f ( x _ { k + 1 } ) \geq { \frac { \| R _ { i } \nabla f _ { k + 1 } \| ^ { 2 } } { 8 { \sqrt { \omega _ { i } ^ { 3 } L \| R _ { i } \nabla f _ { k } \| / 2 } } } } ,\tag{3.2}
$$

$$
\| R _ { i } \nabla f _ { k + 1 } \| \leq 2 \| R _ { i } \nabla f _ { k } \| .
$$

```latex
Algorithm 3.1 MultiLevel Regularized Newton (MLRN)
Inputs: $x _ { 0 } \in \mathbb { R } ^ { n } , \ell \in \mathbb { N } , R _ { i } : \mathbb { R } ^ { n } \to \mathbb { R } ^ { n _ { i } }$ for $i \in \mathcal { I } , \sigma \in \left( 0 , \operatorname* { m i n } \left\{ 1 , \operatorname* { m i n } _ { i \in \mathcal { I } } \omega _ { i } \right\} \right) , \varepsilon > 0 .$
Outputs: Sequence of iterates $( x _ { 0 } , x _ { 1 } , x _ { 2 } , . . . )$
for $k = 0 , 1 , \ldots$ . do
if $\nabla f _ { k } = 0$ then
stop
end if
Set
$\begin{array} { r } { x _ { k + 1 } : = \left\{ \begin{array} { l l } { x _ { k } - P _ { i } \left( R _ { i } \nabla ^ { 2 } f _ { k } P _ { i } + \sqrt { \frac { \omega _ { i } ^ { 3 } L \left. R _ { i } \nabla f _ { k } \right. } { 2 } } I _ { n _ { i } } \right) ^ { - 1 } R _ { i } \nabla f _ { k } , } & { k \in \mathcal { C } _ { i } , i \in \mathcal { I } , } \\ { x _ { k } - \left( \nabla ^ { 2 } f _ { k } + \sqrt { \frac { L \left. \nabla f _ { k } \right. } { 2 } } I _ { n } \right) ^ { - 1 } \nabla f _ { k } , } & { k \in \mathcal { F } ^ { ( \ell ) } . } \end{array} \right. } \end{array}$
end for
```

Proof. The proof is outlined in Subsection A.1.

Lemma 3.1 guarantees that $f ( x _ { k + 1 } ) \leq f ( x _ { k } ) , k \in \mathcal { C } _ { i }$ , and by $( 2 . 4 ) , ( f ( x _ { k } ) ) _ { k \in \mathbb { N } }$ is non-increasing. In particular, this means that the multilevel iterates stay in $\mathcal { L } ( \boldsymbol { x } _ { 0 } )$ Finally, in Lemma A.2 we show that Algorithm 3.1 performs only finitely many coarse iterations.

3.1. Conditional rate of convergence. Let $\rho \in ( 0 , 1 )$ . Similar to the analysis of the two-level method, we will need the following definitions,

$$
\begin{array} { r l r } {  { \mathcal { C } _ { i } ^ { k } : = \mathcal { C } _ { i } \cap \{ 0 , \ldots , k - 1 \} , \mathcal { C } ^ { ( \ell ) , k } : = \bigcup _ { i \in \mathcal { I } } \mathcal { C } _ { i } ^ { k } , } } \\ & { } & \\ & { } & { \mathcal { F } ^ { ( \ell ) , k } : = \mathcal { F } ^ { ( \ell ) } \cap \{ 0 , \ldots , k - 1 \} , } \\ & { } & { \mathcal { T } _ { \mathcal { C } _ { i } } ^ { k } ( \rho ) : = \{ m \in \mathcal { C } _ { i } ^ { k } : \| R _ { i } \nabla f _ { m + 1 } \| > \rho \| R _ { i } \nabla f _ { m } \| \} , } \\ & { } & { \mathcal { Z } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) : = \{ m \in \mathcal { F } ^ { ( \ell ) , k } : \| \nabla f _ { m + 1 } \| > \rho \| \nabla f _ { m } \| \} , } \end{array}
$$

where $i \in \mathcal { I }$ . We further set

$$
\mathcal { T } _ { \ell } ^ { k } ( \rho ) : = \left( \bigcup _ { i \in \mathcal { I } } I _ { \mathcal { C } _ { i } } ^ { k } ( \rho ) \right) \cup \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) .
$$

Hence, the disjointness of the sets $\mathcal { C } _ { i } , \ : i \in \mathcal { I }$ , and $\mathcal { F } ^ { ( \ell ) }$ gives

$$
\left. \mathcal { T } _ { \ell } ^ { k } ( \rho ) \right. = \sum _ { i = 1 } ^ { \ell } \left. I _ { \mathcal { C } _ { i } } ^ { k } ( \rho ) \right. + \left. \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) \right. .
$$

The following lemma shows the $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate of Algorithm 3.1 when $\left| \mathcal { T } _ { \ell } ^ { k } ( \rho ) \right|$ is suficiently large.

Lemma 3.2. Suppose that Assumptions 1.1, 1.2, and 3.1 hold. Let $k \in \mathbb N$ and

$\theta , \rho \in ( 0 , 1 ) . \ I f \left| \mathcal { T } _ { \ell } ^ { k } ( \rho ) \right| > \theta k$ , then the sequence generated by Algorithm 3.1 satisfies

$$
f ( x _ { k } ) - f ^ { * } \leq \frac { 1 } { C _ { \ell } ^ { 2 } \left( 1 + \frac { \theta k } { 2 } \right) ^ { 2 } } , \qquad C _ { \ell } : = \sqrt { \frac { \rho ^ { 4 } \sigma ^ { 3 } } { 3 2 L \zeta ^ { 3 } D ^ { 3 } } } .
$$

Proof. The proof is given in Subsection A.2.

Further, the coarse correction is also stable with respect to the full gradient. In particular, by Lemma 2.4 there exist $M _ { \mathcal { C } _ { i } } \in ( 0 , \infty ) , i \in \mathcal { I }$ , such that

$$
\begin{array} { r } { \left. \nabla f _ { k + 1 } \right. \leq M _ { \mathcal { C } _ { i } } \left. \nabla f _ { k } \right. , \qquad k \in \mathcal { C } _ { i } . } \end{array}
$$

In what follows we show that Algorithm 3.1 converges with $\mathcal { O } \big ( k ^ { - 2 } \big )$ when it performs a suficiently large numbersuficiently of fine steps.

Theorem 3.3. Suppose that Assumptions 1.1, 1.2, and 3.1 hold. Let

$$
M _ { { \mathcal { C } } ^ { ( \ell ) } } : = \operatorname* { m a x } \left\{ 1 , \operatorname* { m a x } _ { i \in { \mathcal { T } } } M _ { { \mathcal { C } } _ { i } } \right\} , \qquad h : = \log _ { 2 } M _ { { \mathcal { C } } ^ { ( \ell ) } } .
$$

Let $\theta \in ( 0 , 2 / 3 )$ and choose

$$
a \in \left( { \frac { h + 3 \theta } { h + 2 } } , 1 \right) .
$$

Suppose also that, for all suficiently large k,

$$
\left| { \mathcal { F } } ^ { ( \ell ) , k } \right| \geq a k .
$$

Then the sequence generated by Algorithm 3.1 satisfies

$$
f ( x _ { k } ) - f ^ { * } = { \mathcal { O } } \left( k ^ { - 2 } \right) .
$$

Proof. Fix a suficiently large $k \in \mathbb N$ such that $\left| { \mathcal { F } } ^ { ( \ell ) , k } \right| \geq a k$ , and set $\mathcal { T } _ { \ell } ^ { k } : =$ $\mathcal { T } _ { \ell } ^ { k } ( 1 / 4 ) , \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } : = \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( 1 / 4 ) . \mathrm { ~ I f ~ } \left| \mathcal { T } _ { \ell } ^ { k } \right| > \theta k$ , then Lemma 3.2 yields

$$
f ( x _ { k } ) - f ^ { * } = { \mathcal { O } } \left( k ^ { - 2 } \right) .
$$

It remains to consider the case $| \mathcal { T } _ { \ell } ^ { k } | \le \theta k$ . For every $m \in \{ 0 , \ldots , k - 1 \}$ , exactly one of the following possibilities occurs:

(a) If $m \in { \mathcal { C } } ^ { ( \ell ) , k }$ , then $\lVert \nabla f _ { m + 1 } \rVert \leq M _ { \mathcal { C } ^ { ( \ell ) } } \lVert \nabla f _ { m } \rVert .$

(b) If $m \in \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k }$ , then, by $( 2 . 4 ) , \| \nabla f _ { m + 1 } \| \leq 2 \| \nabla f _ { m } \|$

(c) If $m \in \mathring { \mathcal { F } } ^ { ( \ell ) , k } \setminus \mathcal { T } _ { \mathcal { F } ( \ell ) } ^ { k }$ , then $\begin{array} { r l } {  { \bigl \| \nabla f _ { m + 1 } \bigr \| \leq \frac { 1 } { 4 } \| \nabla f _ { m } \| } } \end{array}$

Since $\left| \mathcal { C } ^ { ( \ell ) , k } \right| + \left| \mathcal { F } ^ { ( \ell ) , k } \right| = k$ , iterating these bounds gives

$$
\begin{array} { r l } & { \| \nabla f _ { k } \| \leq M _ { \mathcal { C } ^ { ( \ell ) } } ^ { k - \left| \mathcal { F } ^ { ( \ell ) , k } \right| } 2 ^ { \left| \mathcal { Z } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } \right| } \left( \frac { 1 } { 4 } \right) ^ { \left| \mathcal { F } ^ { ( \ell ) , k } \right| - \left| \mathcal { Z } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } \right| } \| \nabla f _ { 0 } \| } \\ & { \qquad = M _ { \mathcal { C } ^ { ( \ell ) } } ^ { k - \left| \mathcal { F } ^ { ( \ell ) , k } \right| } 2 ^ { - 2 \left| \mathcal { F } ^ { ( \ell ) , k } \right| + 3 \left| \mathcal { Z } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } \right| } \| \nabla f _ { 0 } \| . } \end{array}
$$

Since $\left| \mathcal { F } ^ { ( \ell ) , k } \right| \geq a k$ and $\left. \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } \right. \le \left. \mathcal { T } _ { \ell } ^ { k } \right. \le \theta k$ , we obtain

$$
\begin{array} { r } { \| \nabla f _ { k } \| \le \beta ^ { k } \| \nabla f _ { 0 } \| , } \end{array}
$$

where

$$
\beta : = M _ { \mathcal { C } ^ { ( \ell ) } } ^ { 1 - a } 2 ^ { - 2 a + 3 \theta } .
$$

Moreover,

$$
\log _ { 2 } \beta = h + 3 \theta - a ( h + 2 ) < 0
$$

by the choice of $^ { a , }$ and hence $\beta < 1$

Therefore, by convexity and Assumption 1.1,

$$
f ( x _ { k } ) - f ^ { * } \leq D \| \nabla f _ { k } \| \leq D \beta ^ { k } \| \nabla f _ { 0 } \| = { \mathcal { O } } \big ( \beta ^ { k } \big ) = { \mathcal { O } } \big ( k ^ { - 2 } \big ) .
$$

Combining the two cases proves the result.

Theorem 3.3 provides a quantitative condition under which the multilevel scheme retains the convergence rate of the underlying full-space method. Although this rate need not be inherited by an arbitrary multilevel realization, the theorem shows that the $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate is preserved whenever a suficiently large fraction of the iterations is performed on the fine level. Such preservation of complexity has also been established for other multilevel high-order regularization frameworks; see, e.g., [5] for the multilevel ARC method and possibly non-convex functions. To the best of our knowledge, Theorem 3.3 is the first result showing that a classical multilevel gradient-regularized Newton method can retain the global $\mathcal { O } \big ( k ^ { - 2 } \big )$ function value rate of the full-space method. This is possible to achieve under an explicit quantitative condition on the proportion of fine-level iterations.

We emphasize that the fine-iteration frequency condition is not restrictive. Indeed, the eligibility condition $\| R _ { i } \nabla f _ { k } \| \ge \varepsilon$ implies that only finitely many coarse iterations can occur, see Lemma A.2 for details. Consequently, the algorithm eventually performs exclusively fine-level iterations, and the required frequency condition is automatically satisfied for all suficiently large k. Nevertheless, this condition is not computationally irrelevant. For instance, for large k and $\theta = 0 . 1 , M _ { \mathscr { C } ^ { ( \ell ) } } = 2$ and $M _ { C ^ { ( \ell ) } } = 2 0 0 0$ yield approximately $a > 0 . 4 4$ and $a > 0 . 8 7$ , respectively. This indicates that arbitrarily long sequences of coarse corrections cannot not occur before the algorithm enters the fine-only regime. In Subsection 3.3 we develop analysis establishing an explicit convergence bound valid at every iteration, without relying on the eventual transition to the fine-only regime, and which also allows the method to perform arbitrarily long sequences of coarse corrections.

3.2. Row-nested restriction operators. In the previous subsection, we established an $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence rate under the condition that a suficiently large fraction of the iterations is performed on the fine level. Such a condition may lead to a large number of computationally expensive fine-level iterations before Algorithm 3.1 enters its fine-level regime. Here we introduce an additional structure on the restriction operators and discuss its most important properties. This new structure will, later, allow us to show a global $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate for Algorithm 3.1 without this condition.

Definition 3.4 (Row-nested restriction operators). Let $R _ { i } \in \mathbb { R } ^ { n _ { i } \times n } , \ i \in \mathcal { I }$ with

$$
n _ { 1 } < n _ { 2 } < \cdots < n _ { \ell } < n .
$$

The family $( R _ { i } ) _ { i = 1 } ^ { \ell }$ is said to be row-nested if, for every $i = 1 , \ldots , \ell - 1$ , there exists

$$
\widehat { R } _ { i } \in \mathbb { R } ^ { ( n _ { i + 1 } - n _ { i } ) \times n }
$$

such that

$$
R _ { i + 1 } = \left[ { \frac { R _ { i } } { \widehat { R } _ { i } } } \right] .
$$

Nested hierarchies of coarse spaces are standard in multigrid and have also been employed in multilevel optimization; see, e.g., [13, 17]. Here we impose a more specific algebraic structure: all restriction operators act on the same fine space, and the rows defining a lower-dimensional restriction are retained at every subsequent level. This row-nested construction yields the monotonicity $\| R _ { i } v \| \leq \| R _ { j } v \|$ for $i < j$ , which will play a central role in our convergence analysis.

For every $i < j$ there exists $\widehat { R } _ { i j } \in \mathbb { R } ^ { ( n _ { j } - n _ { i } ) \times n }$ such that

$$
R _ { j } = \left[ \begin{array} { l } { R _ { i } } \\ { \widehat { R } _ { i j } } \end{array} \right] .
$$

Consequently, for every $v \in \mathbb { R } ^ { n }$

$$
\| R _ { j } v \| ^ { 2 } = \| R _ { i } v \| ^ { 2 } + \big \| \widehat { R } _ { i j } v \big \| ^ { 2 } ,
$$

and therefore,

$$
\begin{array} { r } { \| R _ { i } v \| \le \| R _ { j } v \| , \qquad 1 \le i < j \le \ell . } \end{array}\tag{3.3}
$$

Further, Definition 3.4 implies nestedness in multilevel spaces. By the definition of $R _ { j }$ we have

$$
P _ { j } = { R _ { j } ^ { \top } = \left[ P _ { i } \quad \widehat { R } _ { i j } ^ { \top } \right] } .
$$

Therefore, since $n _ { i } < n _ { j }$ ，

$$
\mathrm { r a n g e } ( P _ { i } ) \subset \mathrm { r a n g e } ( P _ { j } ) , \qquad i < j ,\tag{3.4}
$$

and thus, as in classical multilevel methods with nested subspaces, each successive level enriches the previous coarse correction space with additional directions. However, the stronger row-nested structure additionally implies the norm monotonicity (3.3). There also is a corresponding reverse nesting of null spaces, namely

$$
\ker ( R _ { j } ) \subset \ker ( R _ { i } ) , \qquad i < j ,
$$

where the inclusion is strict since dim ker $( R _ { j } ) = n - n _ { j } < n - n _ { i } = \dim \ker ( R _ { i } )$ . Thus, larger coarse levels miss fewer directions of the fine-space gradient.

Definition 3.4 yields properties that significantly simplify the multilevel algorithm and provides us with further insights. In particular,

$$
\omega _ { 1 } \leq \omega _ { 2 } \leq \cdot \cdot \cdot \leq \omega _ { \ell } = \zeta ,
$$

and the admissible range for $\sigma$ simplifies to $\sigma \in ( 0 , \operatorname* { m i n } \{ 1 , \omega _ { 1 } \} )$ . Moreover, (3.3) implies that the eligible sets satisfy

$$
{ \mathcal { E } } _ { 1 } \subseteq { \mathcal { E } } _ { 2 } \subseteq \cdots \subseteq { \mathcal { E } } _ { \ell } , \qquad \bigcup _ { j < i } { \mathcal { E } } _ { j } = { \mathcal { E } } _ { i - 1 } , \quad i = 2 , \ldots , \ell .
$$

Hence,

$$
\mathcal C _ { 1 } = \mathcal E _ { 1 } , \qquad \mathcal C _ { i } = \mathcal E _ { i } \setminus \mathcal E _ { i - 1 } , \quad i = 2 , \dots , \ell ,
$$

and,

$$
\mathcal { C } ^ { ( \ell ) } = \bigcup _ { i = 1 } ^ { \ell } \mathcal { C } _ { i } = \mathcal { E } _ { \ell } , \qquad \mathcal { F } ^ { ( \ell ) } = \mathbb { N } \setminus \mathcal { E } _ { \ell } .
$$

Therefore, $k \in \mathcal { C } _ { i }$ means that level i is the lowest-dimensional level satisfying the eligibility criterion at iteration k. Every level $j > i$ is then automatically eligible, whereas no level $j < i$ is.

Remark 3.5. The nested structure also induces a hierarchy of reduced Hessian matrices. Indeed, for $i < j$

$$
R _ { j } \nabla ^ { 2 } f ( x ) P _ { j } = \left[ \begin{array} { c c } { { R _ { i } \nabla ^ { 2 } f ( x ) P _ { i } } } & { { R _ { i } \nabla ^ { 2 } f ( x ) \widehat { R } _ { i j } ^ { \top } } } \\ { { \widehat { R } _ { i j } \nabla ^ { 2 } f ( x ) P _ { i } } } & { { \widehat { R } _ { i j } \nabla ^ { 2 } f ( x ) \widehat { R } _ { i j } ^ { \top } } } \end{array} \right] .
$$

Hence, $R _ { i } \nabla ^ { 2 } f ( x ) P _ { i }$ is a principal submatrix of $R _ { j } \nabla ^ { 2 } f ( x ) P _ { j }$ . Consequently, by the Cauchy interlacing theorem [15], their eigenvalues satisfy

$$
\begin{array} { r } { \lambda _ { r } \bigl ( R _ { j } \nabla ^ { 2 } f ( x ) P _ { j } \bigr ) \leq \lambda _ { r } \bigl ( R _ { i } \nabla ^ { 2 } f ( x ) P _ { i } \bigr ) \leq \lambda _ { r + n _ { j } - n _ { i } } \bigl ( R _ { j } \nabla ^ { 2 } f ( x ) P _ { j } \bigr ) , } \end{array}
$$

for $r = 1 , \ldots , n _ { i }$ , where the eigenvalues are ordered increasingly. Moreover, if $f$ is convex, then

$$
R _ { i } \nabla ^ { 2 } f ( x ) P _ { i } \succeq 0 , \qquad i \in \mathcal { I } ,
$$

while $\nabla ^ { 2 } f ( x ) \succ 0$ together with the full row rank of $R _ { i }$ implies

$$
R _ { i } \nabla ^ { 2 } f ( x ) P _ { i } \succ 0 .
$$

Thus, increasing the level enriches the reduced model while retaining the curvature information contained in the preceding levels. However, note that the principalsubmatrix relation concerns the unregularized reduced Hessians; the regularized matrices need not be nested because the regularization parameter depends on the level.

Remark 3.6. An important special case of Definition 3.4 arises when the restriction operators have orthonormal rows, i.e.,

$$
R _ { i } R _ { i } ^ { \top } = I _ { n _ { i } } , \qquad i \in \mathcal { I } .
$$

Then

$$
Q _ { i } : = P _ { i } R _ { i } = R _ { i } ^ { \top } R _ { i }
$$

is the orthogonal projector onto range(P<sub>i</sub>). Moreover, (3.4) implies

$$
Q _ { i } Q _ { j } = Q _ { j } Q _ { i } = Q _ { i } , \qquad Q _ { i } \preceq Q _ { j } , \qquad i < j .
$$

Thus, increasing the level enlarges the projected subspace while retaining all directions represented at the preceding levels.

A simple and computationally eficient construction satisfying Assumption 3.1, Definition 3.4, and the orthonormal-row condition considered above is obtained by selecting nested subsets of the rows of the identity matrix. Specifically, let $S _ { 1 } \subset \cdots \subset$ $S _ { \ell } \subset \{ 1 , \ldots , n \}$ with $| S _ { i } | = n _ { i }$ , and let $R _ { i }$ consist of the rows of $I _ { n }$ indexed by $S _ { i }$ . For each $i = 1 , \ldots , \ell - 1$ , choosing $\widehat { R } _ { i }$ as the rows of $I _ { n }$ indexed by $S _ { i + 1 } \setminus S _ { i }$ yields

$$
R _ { i + 1 } = \left[ { \frac { R _ { i } } { \widehat { R } _ { i } } } \right] .
$$

Since the selected rows are distinct standard basis vectors, $R _ { i } R _ { i } ^ { \top } = I _ { n _ { i } }$ for every $i \in \mathcal { I }$ . Thus, Assumption 3.1 is satisfied with $\omega _ { i } = 1 , i \in \mathcal { I }$ , and further $\sigma \in ( 0 , 1 )$ .

Remark 3.7. A general construction satisfying both Assumption 3.1 and Definition 3.4 is obtained by choosing any full row-rank restriction operator $R _ { \ell } \in \mathbb { R } ^ { n _ { \ell } \times n }$ and defining $R _ { i }$ as its first $n _ { i }$ rows, for $i = 1 , \ldots , \ell - 1$ . Indeed, the full row-rank of $R _ { \ell }$ guarantees that rank $( R _ { i } ) = n _ { i }$ , while the construction yields $\boldsymbol { R _ { i + 1 } } = \left[ \boldsymbol { R _ { i } ^ { \top } } , \boldsymbol { \widehat { R } _ { i } ^ { \top } } \right] ^ { \top }$ ， where $\widehat { R } _ { i }$ consists of rows $n _ { i } + 1 , \ldots , n _ { i + 1 }$ of $R _ { \ell }$ . This construction accommodates, for instance, Gaussian matrices, suitable sparse sketching matrices, and coordinate restrictions.

3.3. Convergence analysis using the row-nested restriction operators. We now establish an $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence rate for Algorithm 3.1 under the row-nested structure of Definition 3.4, without imposing any condition on the frequency of finelevel iterations. The analysis extends the weighted-gradient argument developed for the two-level method. An additional dificulty arises from transitions between diferent coarse levels, since the corresponding updates control distinct reduced-gradient norms $\| R _ { i } \nabla f _ { k } \|$ . For general restriction operators, these quantities need not be comparable across levels. The row-nested structure provides the necessary norm comparisons, which, together with suitable level-dependent weights, allow us to control transitions among all coarse and fine levels.

Theorem 3.8. Let f be convex and suppose that Assumptions 1.1 and 1.2 hold. If the sequence $( x _ { k } ) _ { k \in \mathbb { N } }$ is generated by Algorithm 3.1 with row-nested operators for which Assumption 3.1 hold, then

$$
f ( x _ { k } ) - f ^ { * } = { \mathcal { O } } { \left( k ^ { - 2 } \right) } .
$$

Proof. The proof makes use of the set $\mathcal { T } _ { \ell } ^ { k } ( \rho )$ and as previously we split the analysis into two cases, according to the magnitude of $\mathcal { T } _ { \ell } ^ { k } ( \rho )$

Recall that, due to the row-nestedness, $\zeta = \omega _ { \ell }$ , and define $\bar { \omega } : = \mathrm { m a x } \{ 1 , \omega _ { \ell } \}$ . We choose

$$
\theta \in \left( 0 , \frac { 1 } { 1 + \ell } \right) , \qquad r \in \left( 0 , \operatorname* { m i n } \left\{ \frac { 1 } { \Lambda } , B \right\} \right) , \qquad \rho \in \left( 0 , \frac { \Lambda } { \bar { \omega } } r ^ { 1 + \ell } \right) ,
$$

where

$$
\Lambda : = \frac { \bar { \omega } M _ { \mathcal { C } ^ { ( \ell ) } } } { \sigma } , \qquad B : = \exp \left( - \frac { \left( 1 - \theta \right) \ln ( \Lambda ) + \theta \ln ( 2 \bar { \omega } ) } { 1 - ( 1 + \ell ) \theta } \right) ,
$$

and note that $\Lambda \geq 1$ and $B , r , \rho \in ( 0 , 1 )$ . Moreover, for $i \in \mathcal { I }$ , we define

$$
a _ { i } : = r ^ { i - 1 } , \qquad b : = r ^ { \ell } ,
$$

therefore $1 = a _ { 1 } > \cdot \cdot \cdot > a _ { \ell } > 0$ , and note also that $\begin{array} { r } { \frac { a _ { j } } { a _ { i } } = r ^ { j - i } > 1 } \end{array}$ , if $j < i$ , and $r ^ { j - i } < 1 , { \mathrm { i f ~ } } j > i .$

Let us fix a $k \in \mathbb N$ . If $\left. T _ { \ell } ^ { k } ( \rho ) \right. > \theta k$ , then Lemma 3.2 provides the desired rate of convergence. Assume now that $\left. \dot { T } _ { \ell } ^ { k } ( \rho ) \right. \leq \theta k$ . Define the weight gradient measure for ℓ coarse levels

$$
V _ { k } : = \left\{ \begin{array} { l l } { a _ { i } \| R _ { i } \nabla f _ { k } \| , } & { k \in \mathcal { C } _ { i } , ~ i \in \mathcal { I } , } \\ { b \| \nabla f _ { k } \| , } & { k \in \mathcal { F } ^ { ( \ell ) } . } \end{array} \right.
$$

Thus, by the definition of $a _ { i }$ and $V _ { k }$ ,

$$
\left\| \nabla f _ { k } \right\| \leq \operatorname* { m a x } \left\{ { \frac { 1 } { a _ { 1 } \sigma } } , \ldots , { \frac { 1 } { a _ { \ell } \sigma } } , { \frac { 1 } { b } } \right\} V _ { k } = \operatorname* { m a x } \left\{ { \frac { 1 } { a _ { \ell } \sigma } } , { \frac { 1 } { b } } \right\} V _ { k } .\tag{3.5}
$$

Next, it holds

$$
\left( \mathcal { Z } _ { \ell } ^ { k } ( \rho ) \right) ^ { \mathrm { c } } : = \{ 0 , 1 , \ldots , k - 1 \} \setminus \mathcal { Z } _ { \ell } ^ { k } ( \rho ) = \left( \bigcup _ { i \in \mathcal { I } } C _ { i } ^ { k } \setminus \mathcal { Z } _ { \mathcal { C } _ { i } } ^ { k } ( \rho ) \right) \cup \left( \mathcal { F } ^ { ( \ell ) , k } \setminus \mathcal { Z } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) \right) .
$$

We first examine the case $m \in ( \mathbb { Z } _ { \ell } ^ { k } ( \rho ) ) ^ { \mathrm { c } }$ , where, in particular, we will show that $V _ { k }$ is a contraction. To achieve this, in comparison to the proof of Theorem 2.5, we need to examine an extra case which concerns the transitions between the ℓ coarse levels:

1. m $\in \mathcal { C } _ { i } ^ { k } \setminus \mathcal { T } _ { \mathcal { C } _ { i } } ^ { k } , m + 1 \in \mathcal { C } _ { j }$ . We analyze this case by splitting into additional three subcases according to the ordering of the coarse levels:

(a) $j < i .$ . Then, by the row-nestedness (3.3), we have

$$
\begin{array} { r l } & { V _ { m + 1 } = a _ { j } \left\| R _ { j } \nabla f _ { m + 1 } \right\| \leq a _ { j } \left\| R _ { i } \nabla f _ { m + 1 } \right\| } \\ & { \qquad k \notin \mathcal { X } _ { c } ^ { k } } \\ & { \qquad \leq \quad \rho a _ { j } \left\| R _ { i } \nabla f _ { m } \right\| = r ^ { j - i } \rho V _ { k } \leq r ^ { 1 - \ell } \rho V _ { k } , } \end{array}
$$

where the last inequality uses the fact that $x \mapsto r ^ { x }$ is non-increasing for $r \in ( 0 , 1 )$

(b) $i < j$ . Then,

$$
\begin{array} { r l } & { { \cal V } _ { m + 1 } = a _ { j } \left\| { \cal R } _ { j } \nabla f _ { m + 1 } \right\| \le a _ { j } \bar { \omega } \left\| \nabla f _ { m + 1 } \right\| } \\ & { \qquad \le a _ { j } \bar { \omega } M _ { \mathcal { C } ^ { ( \ell ) } } \left\| \nabla f _ { m } \right\| \le \Lambda r ^ { j - i } { \cal V } _ { m } \le \Lambda r { \cal V } _ { m } , } \end{array}
$$

where we have use the uniform stability of the coarse corrections, i.e., $\| \nabla f _ { k + 1 } \| \le M _ { \mathcal { C } ^ { ( \ell ) } } \| \nabla f _ { k } \|$ with $M _ { { \cal C } ^ { ( \ell ) } }$ defined in Theorem 3.3.

(c) $i = j$ . Then we immediately have

$$
V _ { m + 1 } \leq \rho V _ { m } .
$$

Combining the above subcases yields

$$
V _ { m + 1 } \leq \operatorname* { m a x } \left\{ \rho , \Lambda r , r ^ { 1 - \ell } \rho \right\} V _ { m } = \Lambda r V _ { m } ,
$$

where the last equality hold by the choice on $\rho .$ . This concludes the first case of arbitrary transitions between coarse spaces.

2. m $\in \mathcal { C } _ { i } ^ { k } \setminus \bar { \mathcal { T } } _ { \mathcal { C } _ { i } } ^ { k } , m + 1 \in \mathcal { F } ^ { ( \ell ) }$ . Then

$$
V _ { m + 1 } \leq b \left\| \nabla f _ { m + 1 } \right\| \leq b M _ { \mathcal { C } ^ { ( \ell ) } } \left\| \nabla f _ { m } \right\| \overset { k \in \mathcal { C } _ { i } ^ { k } } { \leq } r ^ { \ell - i + 1 } \frac { M _ { \mathcal { C } ^ { ( \ell ) } } } { \sigma } \left\| \nabla f _ { m } \right\| \leq \Lambda r V _ { m } .
$$

3. $m \in \mathcal { F } ^ { ( \ell ) , k } \setminus \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) , m + 1 \in \mathcal { C } _ { i }$ . Then,

$$
V _ { m + 1 } = a _ { i } \left\| R _ { i } \nabla f _ { m + 1 } \right\| \le a _ { i } \bar { \omega } \rho \left\| \nabla f _ { m } \right\| = \frac { \bar { \omega } \rho } { r ^ { \ell - i + 1 } } V _ { m } \le \frac { \bar { \omega } \rho } { r ^ { \ell } } V _ { m } .
$$

4. $m \in \mathcal { F } ^ { ( \ell ) , k } \setminus \mathcal { T } _ { \mathcal { F } ^ { ( \ell ) } } ^ { k } ( \rho ) , m + 1 \in \mathcal { F } ^ { ( \ell ) }$ , then we directly obtain

$$
V _ { m + 1 } \leq \rho V _ { m } .
$$

Therefore, by the definition of $\rho ,$

$$
V _ { m + 1 } \leq \left\{ \rho , \Lambda r , \frac { \bar { \omega } \rho } { r ^ { \ell } } \right\} V _ { m } = \Lambda r V _ { m } < V _ { m } , \qquad m \in \left( \mathcal { T } _ { \ell } ^ { k } ( \rho ) \right) ^ { \mathrm { c } } ,\tag{3.6}
$$

which shows that $V _ { k }$ is a contraction for elements of $( \mathcal { T } _ { \ell } ^ { k } ( \rho ) ) ^ { \mathrm { c } }$

Assume now that m $\in \mathcal { T } _ { \ell } ^ { k } ( \rho )$ . Similarly, we split our arguments into four cases, to obtain a recurrent relation for $V _ { k }$ . However, since $\mathcal { T } _ { \ell } ^ { k } ( \rho )$ contains a few elements only, $V _ { k }$ is not required to be a contraction.

1. $m \in C _ { i } ^ { k } , m + 1 \in C _ { j }$ . Then,

$$
V _ { m + 1 } = a _ { j } \left\| R _ { j } \nabla f _ { m + 1 } \right\| \le a _ { j } \bar { \omega } M _ { \mathcal { C } ^ { ( \ell ) } } \left\| \nabla f _ { m } \right\| \le \Lambda r ^ { j - i } V _ { m } \le r ^ { - \ell } V _ { m } ,
$$

where the last inequality uses the monotonicity of $x \mapsto r ^ { x } , r \in ( 0 , 1 )$ 2. m $\in C _ { i } ^ { k } , m + 1 \in \mathcal { F } ^ { ( \ell ) }$ . Then,

$$
V _ { m + 1 } \leq r ^ { \ell } M _ { \mathcal { C } ^ { ( \ell ) } } \left\| \nabla f _ { m } \right\| \leq r ^ { \ell - i + 1 } \Lambda V _ { m } \leq V _ { m } .
$$

3. $m , \in \mathcal { F } ^ { ( \ell ) , k } , m + 1 \in \mathcal { F } ^ { ( \ell ) }$ . Then, by (2.4),

$$
V _ { m + 1 } \leq 2 V _ { m } .
$$

4. $m \in \mathcal { F } ^ { ( \ell ) , k } , m + 1 \in \mathcal { C } _ { i }$ . Then,

$$
V _ { m + 1 } \leq a _ { i } \| R _ { i } \nabla f _ { m + 1 } \| \leq 2 a _ { i } \bar { \omega } \| \nabla f _ { m } \| \leq 2 r ^ { - \ell } \bar { \omega } V _ { m } .
$$

As a result, the above four cases imply

$$
V _ { m + 1 } \leq 2 \bar { \omega } r ^ { - \ell } V _ { m } , \qquad m \in \mathcal { T } _ { \ell } ^ { k } ( \rho ) .\tag{3.7}
$$

Let us denote

$$
\eta : = r ^ { 1 - ( 1 + \ell ) \theta } \Lambda ^ { 1 - \theta } ( 2 \bar { \omega } ) ^ { \theta } .
$$

By the choices of θ and $\rho$ it easy to see that $r ^ { 1 - ( 1 + \ell ) \theta } < 1$ and since $r < B$ we have $\eta < 1$ . Hence, by (3.6) and (3.7), we obtain

$$
V _ { k } \le \left( \Lambda r \right) ^ { k - \left| \mathcal { Z } _ { \ell } ^ { k } \left( \rho \right) \right| } \left( 2 r ^ { - \ell } \bar { \omega } \right) ^ { \left| \mathcal { Z } _ { \ell } ^ { k } \left( \rho \right) \right| } V _ { 0 } \overset { \left| \mathcal { Z } _ { \ell } ^ { k } \left( \rho \right) \right| \le \theta k } { \le } \eta ^ { k } V _ { 0 } ,
$$

which shows that $V _ { k }$ is a contraction. Now the desired result follows from the last inequality, convexity, and (3.5) as follows. Let

$$
A _ { \ell } = \operatorname* { m a x } \left\{ \frac { 1 } { \sigma r ^ { \ell - 1 } } , \frac { 1 } { r ^ { \ell } } \right\} .
$$

Then

$$
f ( x _ { k } ) - f ^ { * } \leq D \left\| \nabla f _ { k } \right\| \leq D A _ { \ell } V _ { k } \leq D A _ { \ell } V _ { 0 } \eta ^ { k }
$$

which shows exponential rate for the case $\left. \mathcal { T } _ { \ell } ^ { k } ( \rho ) \right. \leq \theta k$ . Hence the two cases imply

$$
f ( x _ { k } ) - f ^ { * } \le \operatorname* { m a x } \left\{ D A _ { \ell } V _ { 0 } \eta ^ { k } , \frac { 1 } { C _ { \ell } ^ { 2 } ( 1 + \theta k / 2 ) ^ { 2 } } \right\} ,
$$

which concludes the proof.

Theorem 3.8 establishes a direct $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence bound for the multilevel method with row-nested restriction operators, without imposing any condition on the frequency of fine-level iterations. Although the asymptotic rate also follows for general restriction operators from the eventual fine-only behavior, the present analysis controls the convergence rate throughout the entire sequence, including coarse-dominated phases. In particular, the resulting bound does not depend on the iteration at which the method enters the fine-only regime.

The main dificulty for achieving this result is that a coarse correction at level $i$ controls $\| R _ { i } \nabla f _ { k + 1 } \|$ , but not necessarily the reduced gradient at the subsequently selected level. Row-nestedness resolves this mismatch through monotonicity of the reduced-gradient norms, which, together with level-dependent weights and uniform full-gradient stability, allows us to control arbitrary transitions between levels. The resulting weighted gradient measure contracts outside a set of non-ρ-contracting iterations, while its growth remains uniformly bounded otherwise. When non-ρ-contracting iterations are suficiently frequent, the function-value decrease estimates instead yield the $\mathcal { O } \big ( k ^ { - 2 } \big )$ rate. Thus, the analysis accommodates arbitrarily long sequences of coarse corrections without requiring a prescribed frequency of fine iterations or a decrease of the full gradient at every coarse step.

4. Experiments. In this section, we present numerical experiments to support our theoretical findings. In particular, we compare the performance of the two-level regularized Newton (TLRN), multilevel regularized Newton (MLRN), and row-nested multilevel regularized Newton (MLRN nested) methods with that of the full-space gradient-regularized Newton method (RN) [19]. Although our methods retain the $\mathcal { O } \big ( k ^ { - 2 } \big )$ convergence rate of RN, the constants in their complexity bounds depend on $\sigma < 1$ , see Lemma 3.2, potentially leading to less favorable theoretical bounds. Nevertheless, since the multilevel methods compute some of their updates in lowerdimensional spaces, they may ofer significant computational savings. Our experiments aim to illustrate that the multilevel methods exhibit convergence behavior comparable to RN while potentially reducing the overall computational time.

The restriction operators for MLRN are constructed by selecting rows of the identity matrix as in [28], whereas for the row-nested method we use the construction described in Theorem 3.6. We consider $\ell = 5$ coarse levels with dimensions $n _ { 1 } = 0 . 1 n$ $n _ { 2 } = 0 . 3 n , n _ { 3 } = 0 . 5 n , n _ { 4 } = 0 . 7 n$ , and $n _ { 5 } ~ = ~ 0 . 9 n$ , whereas for TLRN we select $d = 0 . 5 n$

Since the Hessian’s Lipschitz constant L is generally unknown, we employ an adaptive line-search procedure, following the approach of [22, Section 4.1.4.2] and [19]. At each iteration, the estimate $L _ { k }$ is increased until the corresponding suficient-decrease and gradient-stability conditions are satisfied. We set $\varepsilon = 1 0 ^ { - 1 0 }$ and terminate the algorithms when $f ( \boldsymbol { x } _ { k } ) - f ^ { * } \le \varepsilon _ { f } ~ \mathrm { o r } ~ \| \nabla f ( \boldsymbol { x } _ { k } ) \| \le \varepsilon _ { g } ,$ , where $\varepsilon _ { f } = 1 0 ^ { - 8 }$ and $\varepsilon _ { g } = \varepsilon$ We use L-BFGS to compute a numerical approximation of the optimal objective value $f ^ { * }$ . We set $\sigma = 0 . 5$ , balancing the frequency of coarse corrections against their effectiveness (note that larger values of $\sigma$ yield more favorable convergence constants but impose a stricter coarse-level eligibility condition, potentially resulting in more fine-level iterations). All experiments are initialized at the origin.

4.1. Log-sum-exp. Given data $a _ { 1 } , \dots , a _ { m } \in \mathbb { R } ^ { n }$ and $b _ { 1 } , \dots , b _ { m } \in \mathbb { R }$ , we consider minimizing the log-sum-exp function

$$
f _ { \gamma } ( x ) = \gamma \ln \left( \sum _ { i = 1 } ^ { m } \exp \left( \frac { a _ { i } ^ { \top } x - b _ { i } } { \gamma } \right) \right) , \qquad \gamma > 0 .
$$

![](images/507980d6142bb7ce8974d923dd4d7581a2e2aa7107b06919112dcd32a6e72d1e.jpg)  
(a) $\gamma = 0 . 0 5 ,$ , iterations

![](images/91a96e3c003c103fa32f7f3e15aded76d50477ddc8da2c36b0f40a7fbf18bc51.jpg)  
(b) $\gamma = 0 . 0 5 ,$ , runtime

![](images/6c008464b68276bd94c1933fbe0ec0ad6ac4b5edec1911272566b3926001ee38.jpg)  
(c) $\gamma = 0 . 1$ , iterations

![](images/6ec2db9a048c180b31c99ba2615f5f14bc25ea1914c965ddaa78e643ccccdfca.jpg)  
(d) $\gamma = 0 . 1$ , runtime

![](images/6f4d8c2201526b169cbbc4838e90e41ed6635972e14fca62cd9e602b2cde1f59.jpg)  
(e) $\gamma = 0 . 5 ,$ iterations

![](images/412255c2b832b054da54f990a83b0731027f4f55c5866d4d265014221c7272a4.jpg)  
(f) $\gamma = 0 . 5 ,$ , runtime  
Fig. 1. Convergence comparison of MLRN, row-nested MLRN, regularized Newton (RN), and two-level regularized Newton (TLRN) for the log-sum-exp objective with $\gamma \in \{ 0 . 0 5 , 0 . 1 , 0 . 5 \}$ The left column shows the objective gap $f ( x _ { k } ) - f ^ { * }$ versus iterations, while the right column shows the objective gap versus wall-clock time. Row-nested MLRN and TLRN methods demonstrate comparable convergence behavior to RN.

The parameter γ controls the smoothing of the log-sum-exp objective. We generate a random dataset with $m = 1 0 0 0$ observations and $n = 8 0 0$ variables and consider three values of this parameter, namely $\gamma \in \{ 0 . 0 5 , 0 . 1 , 0 . 5 \}$ , to investigate the sensitivity of the algorithms to diferent degrees of smoothing.

The results are presented in Figure 1. For all three values of γ, row-nested MLRN and TLRN exhibit convergence behavior comparable to that of RN, both in terms of iterations and computational time. In contrast, the non-nested MLRN method converges substantially more slowly. We suspect that this diference may be related to the fine-iteration frequency condition in Theorem 3.3, which need not be satisfied during the observed iterations. Nevertheless, these findings suggest that row-nested restriction operators can be an attractive choice from both theoretical and computational perspectives.

![](images/a7e69ea0dc9b7dba5306c0d27b7eca4774af3969dc327d6aed48aa1e4aff62d6.jpg)  
(a) Gisette, iterations

![](images/83245c42cac8792d5bb283fbfdc316d2e23d8f09cfe5f83000c5713834d3213a.jpg)  
(b) Gisette, runtime

![](images/08fd4759765901dbc3d39499831df9af468931af748fa003bbbeb802d680f281.jpg)  
(c) Duke, iterations

![](images/70926d571bc9e9095912624dfc01c1d9a3c239bedb41fe41364aecbee1e89deb.jpg)  
(d) Duke, runtime  
Fig. 2. Convergence comparison of MLRN, row-nested MLRN, regularized Newton (RN), and two-level regularized Newton (TLRN) using the large Gisette and Duke datasets for logistic regression. The left column reports the objective gap $f ( x _ { k } ) - f ^ { * }$ versus iterations, while the right column reports the objective gap versus wall-clock time. The multilevel gradient-regularized Newton methods ofer substantial computational savings compared to their full-space counterpart.

4.2. Logistic regression. In the second experiment, we consider logistic regression problems using real-world datasets. The objective function is given by

$$
f ( \boldsymbol { x } ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \ln \left( 1 + \exp \big ( - b _ { i } a _ { i } ^ { \top } \boldsymbol { x } \big ) \right) + \frac { \lambda _ { 2 } } { 2 } \left. \boldsymbol { x } \right. ^ { 2 } ,\tag{4.1}
$$

where $b _ { i } \in \{ - 1 , 1 \}$ and we set $\lambda _ { 2 } = 1 0 ^ { - 1 0 }$ to make the problem ill-conditioned. The results are presented in Figure 2 and Figure 3.

Across the considered datasets, nested MLRN and TLRN exhibit convergence behavior comparable to that of the full-space RN method. Moreover, as shown in Figure 2, when applied to large datasets, both methods achieve substantial reductions in computational time, illustrating the benefits of performing corrections in lower-dimensional spaces. In contrast, non-row-nested MLRN exhibits slower convergence on some instances, particularly those shown in Figure 3, which may prevent the method from converging faster than RN in computational time. These observations suggest that the structure of the restriction operators, particularly row-nestedness, can play an important role in the practical performance of multilevel gradient-regularized

![](images/588c8066fb20c5b32e3c7b36b9604b29176e39ee01024c646eadebcb78a466e7.jpg)  
(a) a9a, iterations

![](images/0c1754589e5b2dc372c295d7c9f26b8c7eaafc84eab4ae9995e4803d0187608e.jpg)  
(b) a9a, runtime

![](images/e03766a5179a7fb60cafe80c6ee2144772c64cb372bd7d0886d99197200875a4.jpg)  
(c) Mushrooms, iterations

![](images/42e10ba11a302cdbd25332f003ac3311fd947d7125c1011faf63aa5e4ffb309f.jpg)  
(d) Mushrooms, runtime

![](images/94135d6ba582506eaa7f195de10b5fa410586e38b31298c509af6083f578a02c.jpg)  
(e) w8t, iterations

![](images/fd7887f677fd1ea0bcbdd8f7a97d3e73e7b2894db059763b02ca679f9431c9f2.jpg)  
(f) w8t, runtime  
Fig. 3. Convergence comparison of MLRN, row-nested MLRN, regularized Newton (RN), and two-level regularized Newton (TLRN) on the a9a, Mushrooms, and w8t datasets for logistic regression. The left column reports the objective gap $f ( x _ { k } ) - f ^ { * }$ versus iterations, while the right column reports the objective gap versus wall-clock time. Row-nested restriction operators significantly improve the convergence behavior of MLRN.

Newton methods. Overall, the results demonstrate that row-nested restriction operators can yield computationally eficient multilevel methods, complementing the theoretical convergence guarantees established in this paper.

5. Conclusions. We developed and analyzed multilevel gradient-regularized Newton methods for unconstrained convex optimization, establishing a global O k<sup>−2</sup> convergence rate in function values. Among our results, the proof of Theorem 3.8 is of particular interest as it combines a weighted gradient measure with row-nestedness, a special structure of the restriction operators introduced in Subsection 3.2 for the purposes of this analysis. These ideas are new in multilevel optimization theory and we believe that they could prove useful for establishing fast convergence rates for other multilevel methods, such as cubic-regularized Newton and higher-order methods for convex optimization.

Code and data availability. The code and data for reproducing the numerical results is available at

https://github.com/Ntsip/Multilevel-Gradient-Regularized-Newton-Method.

It is authored by Nick Tsipinakis and published under the MIT license.

CRediT author statement. Nick Tsipinakis: Conceptualization, Methodology, Software, Formal Analysis, Investigation, Writing — Original Draft, Visualization; Matthias Voigt: Writing — Review & Editing, Visualization, Supervision.

Acknowledgments. The authors used AI tools to assist with language and figure editing, mathematical proof verification, and the development and debugging of numerical implementations. All mathematical results, algorithms, and computational experiments were reviewed and validated by the authors. The authors assume responsibility for all content.

## References.

[1] A. Ang, H. De Sterck, and S. Vavasis, MGProx: A nonsmooth multigrid proximal gradient method with adaptive restriction for strongly convex optimization, SIAM J. Optim., 34 (2024), pp. 2788–2820, https://doi.org/10.1137/ 23M1552140.

[2] P. Binev, A. Cohen, W. Dahmen, R. DeVore, G. Petrova, and P. Wojtaszczyk, Convergence rates for greedy algorithms in reduced basis methods, SIAM J. Math. Anal., 43 (2011), pp. 1457–1472, https://doi.org/10.1137/ 100795772.

[3] M. S. Branicky, Multiple Lyapunov functions and other analysis tools for switched and hybrid systems, IEEE Trans. Automat. Control, 43 (1998), pp. 475– 482, https://doi.org/10.1109/9.664150.

[4] W. L. Briggs, V. E. Henson, and S. F. McCormick, A Multigrid Tutorial, SIAM, Philadelphia, PA, 2nd ed., 2000, https://doi.org/10.1137/1. 9780898719505.

[5] H. Calandra, S. Gratton, E. Riccietti, and X. Vasseur, On high-order multilevel optimization strategies, SIAM J. Optim., 31 (2021), pp. 307–330, https: //doi.org/10.1137/19M1255355.

[6] C. Cartis, N. I. M. Gould, and P. L. Toint, Adaptive cubic regularisation methods for unconstrained optimization. Part I: Motivation, convergence and numerical results, Math. Program., 127 (2011), pp. 245–295, https: //doi.org/10.1007/s10107-009-0286-5.

[7] C. Cartis, N. I. M. Gould, and P. L. Toint, Adaptive cubic regularisation methods for unconstrained optimization. Part II: Worst-case functionand derivative-evaluation complexity, Math. Program., 130 (2011), pp. 295–319, https://doi.org/10.1007/s10107-009-0337-y.

[8] C. Cartis, N. I. M. Gould, and P. L. Toint, Evaluation complexity of adaptive cubic regularization methods for convex unconstrained optimization, Optim. Methods Softw., 27 (2012), pp. 197–219, https://doi.org/10.1080/10556788.2011. 602076.

[9] N. Doikov and Y. Nesterov, Gradient regularization of Newton method with Bregman distances, Math. Program., 204 (2024), pp. 1–25, https://doi.org/10. 1007/s10107-023-01943-7.

[10] N. I. M. Gould, S. Lucidi, M. Roma, and P. L. Toint, Solving the trustregion subproblem using the Lanczos method, SIAM J. Optim., 9 (1999), pp. 504– 525, https://doi.org/10.1137/S1052623497322735.

[11] S. Gratton, A. Sartenaer, and P. L. Toint, Recursive trust-region methods for multiscale nonlinear optimization, SIAM J. Optim., 19 (2008), pp. 414–444, https://doi.org/10.1137/050623012.

[12] A. Griewank, The modification of Newton’s method for unconstrained optimization by bounding cubic terms, Technical Report NA/12, Department of Applied Mathematics and Theoretical Physics, University of Cambridge, Cambridge, 1981.

[13] C. Gross and R. Krause, On the convergence of recursive trust-region methods for multiscale nonlinear optimization and applications to nonlinear mechanics, SIAM J. Numer. Anal., 47 (2009), pp. 3044–3069, https://doi.org/10.1137/ 08071819X.

[14] C. P. Ho, M. Kocvara, and P. Parpas<sup>ˇ</sup> , Newton-type multilevel optimization method, Optim. Methods Softw., 37 (2022), pp. 45–78, https://doi.org/10.1080/ 10556788.2019.1700256.

[15] R. A. Horn and C. R. Johnson, Matrix Analysis, Cambridge University Press, Cambridge, 2nd ed., 2012, https://doi.org/10.1017/CBO9780511810817.

[16] V. Hovhannisyan, P. Parpas, and S. Zafeiriou, MAGMA: Multilevel accelerated gradient mirror descent algorithm for large-scale convex composite minimization, SIAM J. Imaging Sci., 9 (2016), pp. 1829–1857, https://doi.org/10. 1137/15M104013X, https://doi.org/10.1137/15M104013X.

[17] M. Kocvara and S. Mohammed<sup>ˇ</sup> , A first-order multigrid method for boundconstrained convex optimization, Optim. Methods Softw., 31 (2016), pp. 622–644, https://doi.org/10.1080/10556788.2016.1146267.

[18] R. M. Lewis and S. G. Nash, Model problems for the multigrid optimization of systems governed by diferential equations, SIAM J. Sci. Comput., 26 (2005), pp. 1811–1837, https://doi.org/10.1137/S1064827502407792.

[19] K. Mishchenko, Regularized Newton method with global O(1/k<sup>2</sup>) convergence, SIAM J. Optim., 33 (2023), pp. 1440–1462, https://doi.org/10.1137/ 22M1488752.

[20] S. G. Nash, A multigrid approach to discretized optimization problems, Optim. Methods Softw., 14 (2000), pp. 99–116, https://doi.org/10.1080/ 10556780008805795.

[21] Y. Nesterov, Accelerating the cubic regularization of Newton’s method on convex problems, Math. Program., 112 (2008), pp. 159–181, https://doi.org/10.1007/ s10107-006-0089-x.

[22] Y. Nesterov, Lectures on Convex Optimization, vol. 137 of Springer Optim. Appl., Springer, Cham, 2nd ed., 2018, https://doi.org/10.1007/ 978-3-319-91578-4.

[23] Y. Nesterov and B. T. Polyak, Cubic regularization of Newton method and its global performance, Math. Program., 108 (2006), pp. 177–205, https://doi. org/10.1007/s10107-006-0706-8.

[24] R. A. Polyak, Regularized Newton method for unconstrained convex optimization, Math. Program., 120 (2009), pp. 125–145, https://doi.org/10.1007/ s10107-007-0143-3.

[25] N. Tsipinakis and P. Parpas, A multilevel method for self-concordant minimization, J. Optim. Theory Appl., 203 (2024), pp. 2509–2559, https://doi.org/ 10.1007/s10957-024-02509-z.

[26] N. Tsipinakis and P. Parpas, Convergence rates of Newton’s method for strongly self-concordant minimization, arXiv preprint arXiv:2507.23558, 2025, https://doi.org/10.48550/arXiv.2507.23558.

[27] N. Tsipinakis and P. Parpas, Multilevel regularized Newton methods with fast convergence rates, SIAM J. Sci. Comput., (2025), pp. S232–S257, https: //doi.org/10.1137/24M1677307.

[28] N. Tsipinakis, P. Parpas, and M. Voigt, Adaptive Multilevel Newton: A quadratically convergent optimization method, arXiv preprint arXiv:2510.24967, 2025, https://doi.org/10.48550/arXiv.2510.24967.

[29] N. Tsipinakis, P. Tigas, and P. Parpas, A multilevel low-rank Newton method with super-linear convergence rate and its application to non-convex problems, Trans. Mach. Learn. Res., (2026), https://openreview.net/forum?id= PKakPzVVja.

[30] Z. Wen and D. Goldfarb, A line search multigrid method for large-scale nonlinear optimization, SIAM J. Optim., 20 (2009), pp. 1478–1503, https: //doi.org/10.1137/08071524X.

[31] X. Zhao, P. Shi, Y. Yin, and S. K. Nguang, New results on stability of slowly switched systems: A multiple discontinuous Lyapunov function approach, IEEE Trans. Automat. Control, 62 (2017), pp. 3502–3509, https://doi.org/10. 1109/TAC.2016.2614911.

## Appendix A. Background and proofs of auxiliary results.

Proposition A.1. Let $\left( \Delta _ { \ell } \right) _ { \ell = 0 } ^ { \tau }$ be a tuple of nonnegative numbers satisfying

$$
\Delta _ { \ell + 1 } \leq \Delta _ { \ell } - C \Delta _ { \ell } ^ { 3 / 2 } , \qquad \ell = 0 , \dots , \tau - 1 ,
$$

for some $C > 0$ and $\tau \geq 1$ . Then

$$
\Delta _ { \tau } \leq \frac { 1 } { C ^ { 2 } ( 1 + \tau / 2 ) ^ { 2 } } .
$$

Proof. If $\Delta _ { \ell } = 0$ for some ℓ, the recurrence and nonnegativity imply $\Delta _ { j } = 0$ for all $j \geq \ell ,$ , so the result follows immediately. Otherwise, nonnegativity implies $C \sqrt { \Delta _ { \ell } } < 1$ and hence

$$
\frac { 1 } { \sqrt { \Delta _ { \ell + 1 } } } \geq \frac { 1 } { \sqrt { \Delta _ { \ell } } } \frac { 1 } { \sqrt { 1 - C \sqrt { \Delta _ { \ell } } } } \geq \frac { 1 } { \sqrt { \Delta _ { \ell } } } + \frac { C } { 2 } ,
$$

where we used $( 1 - s ) ^ { - 1 / 2 } \geq 1 + s / 2$ for $s \in [ 0 , 1 )$ . Summing $\frac { 1 } { \sqrt { \Delta _ { \ell + 1 } } } - \frac { 1 } { \sqrt { \Delta _ { \ell } } }$ over $\ell = 0 , \dots , \tau - 1$ , exploiting the telescoping property, and using $1 / \sqrt { \Delta _ { 0 } } \geq C$ gives

$$
\frac { 1 } { \sqrt { \Delta _ { \tau } } } \geq C \left( 1 + \frac { \tau } { 2 } \right) ,
$$

which proves the claim.

We also refer to [22, Theorem 4.1.4] for a related recurrence estimate.

Lemma A.2. Suppose that Assumptions 1.1, 1.2, and 3.1 hold. Then Algorithm 3.1 performs only finitely many coarse iterations.

Proof. Let $k \in \mathcal { C } _ { i }$ for some $i \in \mathcal { I }$ , and define

$$
\alpha _ { i , k } : = \sqrt { \frac { \omega _ { i } ^ { 3 } L \left\| R _ { i } \nabla f _ { k } \right\| } { 2 } } , \qquad d _ { i , k } : = - \left( R _ { i } \nabla ^ { 2 } f _ { k } P _ { i } + \alpha _ { i , k } I _ { n _ { i } } \right) ^ { - 1 } R _ { i } \nabla f _ { k } .
$$

By the coarse-level eligibility condition,

$$
\alpha _ { i , k } \geq \sqrt { \frac { \omega _ { i } ^ { 3 } L \varepsilon } { 2 } } > 0 .
$$

Moreover, since the iterates remain in the compact set $\mathcal { L } ( x _ { 0 } )$ , there exists

$$
G : = \operatorname* { m a x } _ { x \in \mathcal { L } ( x _ { 0 } ) } \| \nabla f ( x ) \| < \infty .
$$

Consequently,

$$
\alpha _ { i , k } \leq \bar { \alpha } _ { i } : = \omega _ { i } ^ { 2 } \sqrt { \frac { L G } { 2 } } .
$$

Using the definition of $d _ { i , k }$ , we obtain

$$
\varepsilon \leq \| R _ { i } \nabla f _ { k } \| \leq \left( \omega _ { i } ^ { 2 } H + \alpha _ { i , k } \right) \| d _ { i , k } \| \leq \left( \omega _ { i } ^ { 2 } H + \bar { \alpha } _ { i } \right) \| d _ { i , k } \| .
$$

Hence,

$$
\| d _ { i , k } \| \ge \frac { \varepsilon } { \omega _ { i } ^ { 2 } H + \bar { \alpha } _ { i } } .
$$

By the descent estimate in [27, Lemma 3.3], the coarse update satisfies

$$
f ( x _ { k } ) - f ( x _ { k + 1 } ) \geq { \frac { \alpha _ { i , k } } { 2 } } \left\| d _ { i , k } \right\| ^ { 2 } .
$$

Combining the preceding estimates yields

$$
f ( x _ { k } ) - f ( x _ { k + 1 } ) \geq \underbrace { { \frac { 1 } { 2 } } { \sqrt { \frac { \omega _ { i } ^ { 3 } L \varepsilon } { 2 } } } \left( { \frac { \varepsilon } { \omega _ { i } ^ { 2 } H + { \bar { \alpha } } _ { i } } } \right) ^ { 2 } } _ { = : \delta _ { i } > 0 } .
$$

Since the number of coarse levels is finite, $\delta : = \mathrm { m i n } _ { i \in \mathcal { I } } \delta _ { i } > 0$ . Therefore, every coarse iteration decreases the objective function by at least δ. By monotonicity of the function values and their lower boundedness by $f ^ { * }$

$$
\begin{array} { r } { | \mathcal { C } ^ { ( \ell ) } | \delta \leq f ( x _ { 0 } ) - f ^ { * } . } \end{array}
$$

Thus, $\vert { \mathcal { C } } ^ { ( \ell ) } \vert < \infty$ , proving the result.

## A.1. Proof of Lemma 3.1.

Proof. The result follows from the analysis in [27, Lemma 3.2 and Corollary 3.4], which allows the restriction operator to vary with the iteration. Indeed, at an iteration $k \in \mathcal { C } _ { i }$ , applying the corresponding estimates with $R _ { k } = R _ { i }$ and $\omega _ { i } = \| R _ { i } \|$ yields

$$
\| R _ { i } \nabla f _ { k + 1 } \| \le 2 \alpha _ { k } \| d _ { i , k } \| , \qquad \| d _ { i , k } \| \le \frac { \| R _ { i } \nabla f _ { k } \| } { \alpha _ { k } } ,
$$

where

$$
\alpha _ { k } = \sqrt { \frac { \omega _ { i } ^ { 3 } L \| R _ { i } \nabla f _ { k } \| } { 2 } } , \qquad d _ { i , k } : = - \left( R _ { i } \nabla ^ { 2 } f _ { k } P _ { i } + \alpha _ { k } I _ { n _ { i } } \right) ^ { - 1 } R _ { i } \nabla f _ { k } .
$$

These inequalities yield (3.2), while the proof of [27, Corollary 3.4] gives (3.1).

## A.2. Proof of Lemma 3.2.

Proof. The proof of Case II in Theorem 2.5 extends naturally to ℓ coarse levels. Thus, below, only the main steps of the proof will be discussed. Let $\tau : = \vert \mathcal { T } _ { \ell } ^ { k } ( \rho ) \vert$ and enumerate $\mathcal { T } _ { \ell } ^ { k } ( \rho ) = \{ k _ { 0 } , \dots , k _ { \tau - 1 } \}$ increasingly, with $k _ { \tau } : = k$ . If $k _ { j } \in \mathcal { T } _ { \mathcal { C } _ { i } } ^ { k } ( { \dot { \rho } } )$ for some $i \in \mathcal { I }$ , then, using the coarse decrease estimate $\omega _ { i } \leq \zeta$ and $\| R _ { i } \nabla \bar { f } _ { k _ { j } } \| \ge \sigma \| \nabla f _ { k _ { j } } \|$ , we obtain

$$
f ( x _ { k _ { j } } ) - f ( x _ { k _ { j } + 1 } ) \geq \frac { \sqrt { 2 } \rho ^ { 2 } } { 8 \sqrt { L } } \left( \frac { \sigma } { \zeta } \right) ^ { 3 / 2 } \| \nabla f _ { k _ { j } } \| ^ { 3 / 2 } .
$$

The same bound holds for $k _ { j } \in \mathcal { T } _ { \mathcal { F } ( \ell ) } ^ { k } \left( \rho \right)$ by the fine-level decrease estimate. Hence, arguing as in Case II of Theorem 2.5,

$$
\Delta _ { j + 1 } \leq \Delta _ { j } - C _ { \ell } \Delta _ { j } ^ { 3 / 2 } , \qquad \Delta _ { j } : = f ( x _ { k _ { j } } ) - f ^ { * } .
$$

$$
f ( x _ { k } ) - f ^ { * } \leq \frac { 1 } { C _ { \ell } ^ { 2 } ( 1 + \theta k / 2 ) ^ { 2 } } .
$$

Since $\tau > \theta k$ , Proposition A.1 yields