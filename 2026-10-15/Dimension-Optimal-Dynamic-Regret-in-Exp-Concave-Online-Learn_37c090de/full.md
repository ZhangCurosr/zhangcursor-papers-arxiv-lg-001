# Dimension-Optimal Dynamic Regret in Exp-Concave Online Learning

Yanyi Li<sup>1,∗</sup> Wendao Wu<sup>1,∗</sup> Haihan Zhang<sup>1,∗</sup> Qinshi Wang<sup>1,∗</sup> Chenheng Zhang<sup>1,∗</sup> Chunyuan Zheng<sup>1</sup> Cong Fang<sup>1,†</sup> Haoxuan Li<sup>1,†</sup> Zhouchen Lin<sup>1,†</sup>

<sup>1</sup>Peking University

liyanyi26@stu.pku.edu.cn wuwendao@stu.pku.edu.cn zhanghaihan@stu.pku.edu.cn

wangqinshi24@gmail.com chenhengz@stu.pku.edu.cn cyzheng@stu.pku.edu.cn

fangcong@pku.edu.cn hxli@pku.edu.cn ZLIN@pku.edu.cn

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## Abstract

Tracking changing comparators is a central problem in online learning. For exp-concave losses, previous work established the $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ dependence on the horizon $T$ and Euclidean path length $P _ { T }$ for fixed dimension, while the optimal cost of dimension remained unresolved. $\mathrm { W e }$ resolve this question by establishing a sharp $d ^ { 1 / 3 }$ curved dimension factor, improving the $d ^ { 2 / 3 }$ factor of recent switching and interval reductions [1, 2]. To achieve this improvement, we develop a proper, full-information learner on compact convex subsets of $\mathbb { R } ^ { d }$ that adapts to the unknown path length and, under fixed dimension-independent diameter, gradient, and curvature parameters, guarantees

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq \widetilde { O } \left( d + \operatorname* { m i n } \left\{ d ^ { 1 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } , \sqrt { T P _ { T } } , T \right\} \right) .
$$

A matching lower bound on a diameter-one Euclidean ball establishes the minimax frontier up to logarithmic factors when $T \gtrsim d ^ { 2 }$ , including the transition from the curved rate to larger path budgets. The lower bound already holds for losses that depend on only one coordinate per round, isolating an unavoidable cost of learning across directions. Our upper bound follows from square-root Gaussian KL control of comparator movement and a cancellation argument that pays for history restarts. The curved learner maintains $O ( \log ^ { 2 } T )$ Gaussian states, uses one gradient per round, and runs in time near-linear in T under a Mahalanobis-projection oracle.

## 1 Introduction

Online learning provides a principled framework for applications such as recommendation, portfolio selection, and tracking evolving networks, where decisions must be updated as data arrive [3, 4]. These sequential decision problems are commonly studied through online convex optimization (OCO): at round t, a learner chooses $w _ { t }$ from a compact convex set $\mathcal { W } \subset \mathbb { R } ^ { d }$ before observing a convex loss $f _ { t } : \mathcal { W } \xrightarrow { } \mathbb { R }$ . To evaluate its ability to track a changing benchmark, we measure the dynamic regret against a comparator sequence $u _ { 1 : T } \in \mathcal { W } ^ { T }$ [5]:

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) = \sum _ { t = 1 } ^ { T } \bigl [ f _ { t } ( w _ { t } ) - f _ { t } ( u _ { t } ) \bigr ] .\tag{1}
$$

For general convex losses, adaptive methods achieve the optimal $O ( \sqrt { T ( 1 + P _ { T } ) } )$ rate under fixed diameter and gradient bounds [6, 7], where the path length $P _ { T }$ is defined as the total Euclidean distance traveled by the comparator sequence. Beyond this general convex setting, widely used prediction losses (e.g., squared and logistic losses on bounded prediction domains) possess additional curvature that permits faster learning [8]. For such curved losses, Baby and Wang [9, 10] established the sharper $T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ temporal dependence, up to logarithmic factors for fixed dimension.

However, the model dimension also afects the cost of tracking, and its role depends on the geometry of the losses. Strong convexity supplies curvature in every direction and admits this fast rate without an explicit dimension factor under fixed Euclidean regularity parameters [1]. By contrast, exp-concavity permits curvature to concentrate in a subspace, suggesting a possible dimension cost for tracking across directions.

For exp-concave losses, Baby and Wang obtained curved-term factors of $d ^ { 7 / 2 } \ [ 9 ]$ and $d ^ { 3 }$ [10] in their respective settings, using $\ell _ { 1 }$ path length.<sup>1</sup> With Euclidean path length, Zhang et al. [11] obtain a factor d through mixability. Recent switching-to-dynamic [1] and interval-to-dynamic [2] reductions attain $d ^ { 2 / 3 }$ . For proper learners, whose predictions remain in the decision set, these advances motivate the following question in full-information OCO:

Question. What is the unavoidable dimension dependence of dynamic regret for general exp-concave losses?

Dimension-optimal answer. We show that the curved dimension factor is $d ^ { 1 / 3 }$ , attained by an implementable proper learner and required by a matching lower bound in the curved regime. We also characterize the transition to larger path budgets. With fixed dimension-independent regularity parameters and without knowing $P _ { T }$ , our learner guarantees

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq \widetilde { O } \left[ d + \operatorname* { m i n } \left\{ d ^ { 1 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } , \sqrt { T P _ { T } } , T \right\} \right] .\tag{2}
$$

The upper bound holds on every admissible compact convex domain. For $T \gtrsim d ^ { 2 }$ , a matching lower bound establishes the minimax frontier over the normalized class of domains and losses (Corollary 5.2). Table 1 summarizes the improvement over previous bounds in the curved regime.

An eficient upper bound. We develop SparseGauss from projected Gaussian histories [12]. This proper learner attains the $d ^ { 1 / 3 }$ curved factor without knowing the path length (Theorem 4.1). The key is to charge comparator displacement through square-root Gaussian $\mathrm { K L } ,$ reducing the movement coeficient from d to ${ \sqrt { d } } ;$ combining this control with rank-one smoothing yields the improved rate. Negative excess-regret prefixes pay for history restarts, so the learner needs only $O ( \log ^ { 2 } T )$ Gaussian states. Under a Mahalanobis-projection oracle, the curved algorithm uses one gradient per round, ${ \cal \tilde { O } } ( d ^ { 2 } )$ memory, and time near-linear in $T .$ . Combining it with adaptive projected-gradient learners yields the full upper bound in Corollary 4.2.

A matching lower bound. We show that the $d ^ { 1 / 3 }$ factor is unavoidable, even for losses that depend on only one coordinate per round (Theorem 5.1). Our construction places independent hidden signs across coordinates and time blocks, forcing the learner to repeatedly estimate changing coordinate means. It keeps the domain diameter, gradient bound, and exp-concavity constant independent of $d ,$ so the lower bound isolates an intrinsic cost of dimension. Varying the number of active coordinates also recovers the square-root and linear regimes, matching (2) up to logarithmic factors when $T \gtrsim d ^ { 2 }$

Table 1: Dynamic regret for proper exp-concave OCO on arbitrary compact convex domains, with Euclidean path length $P _ { T }$ . Diameter, gradient bounds, and curvature are fixed independently of d; logarithmic factors are suppressed. The displayed curved regime is $T \gtrsim d ^ { 2 }$ and $0 \leq P _ { T } \lesssim T / d ^ { 2 }$ The matching lower bound is attained on a diameter-one ball.
<table><tr><td>Work</td><td>Upper bound</td><td>Lower bound</td><td>Mechanism</td></tr><tr><td>Zhang et al. [11]</td><td> $\widetilde { O } ( d + d T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } )$ </td><td></td><td>Gaussian fixed share; KL projection</td></tr><tr><td>Wang et al. [1]</td><td> $\widetilde { O } ( d + d ^ { 2 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } )$ </td><td></td><td>Switching reduction with IRESET/ONS</td></tr><tr><td>Wang et al. [2]</td><td> $\widetilde { O } ( d + d ^ { 2 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } )$ </td><td></td><td>Interval-to-dynamic reduction</td></tr><tr><td>This Paper (Corollary 5.2)</td><td> $\widetilde { \Theta } ( d + d ^ { 1 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } )$ </td><td></td><td>Sparse Gaussian histories; drawdown cancellation</td></tr></table>

Organization. Section 2 reviews prior work, and Section 3 defines the model and oracle assumptions. Section 4 presents the algorithm, upper bounds, and their proof ideas. Section 5 establishes the lower bound and minimax characterization, followed by discussion in Section 6. Section A collects the preliminaries; Sections B and C provide the proofs for Sections 4 and 5, including the finite-precision analysis.

## 2 Related work

Path-length regret and other measures of change. Zinkevich’s projected-gradient analysis [5] established path length as a measure of comparator movement. Ader [6] aggregates learning rates to adapt to an unknown path budget, while Cutkosky [7] connects dynamic regret with parameter-free and strongly adaptive learning. These methods supply the general convex benchmark and motivate the projected-gradient branch of our full frontier. Hall and Willett [4] instead measure deviation from a family of dynamical models, allowing a predictable moving path to have small tracking complexity. Besbes, Gur, and Zeevi [13] study nonstationary stochastic optimization under variation of the loss sequence. Their benchmark and feedback model distinguish that problem from simultaneous regret guarantees for every comparator path. More recently, Jacobsen et al. [14] reduce dynamic regret to static regret in a suitable reproducing kernel Hilbert space, making the choice of comparator geometry part of the reduction.

Curvature, mixability, and variation-bounded prediction. Exp-concavity supports the O(d log T) static guarantee of ONS [15]; mixability more broadly permits aggregation through a suitable prediction rule [16]. Baby and Wang [9, 10] derive curved dynamic rates by analyzing the ofline comparator’s KKT conditions. Their proper strongly convex bound has factor $d ^ { 1 / 3 }$ in terms of $\begin{array} { r } { P _ { T } ^ { ( 1 ) } = \sum _ { t } \Vert u _ { t } - u _ { t - 1 } \Vert _ { 1 } } \end{array}$ , and their proper exp-concave result treats a box domain. Since $P _ { T } \leq P _ { T } ^ { ( 1 ) } \leq \sqrt { d } P _ { T }$ , dimension comparisons must keep the path norm and regularity normalization explicit. Zhang et al. [11] replace KKT analysis with mixability and extend Gaussian aggregation to general exp-concave OCO. Zhang et al. [12] subsequently give a tractable implementation with projected Gaussian histories. Their Theorem 10 has a curved factor d under bounded $\ell _ { 1 }$ diameter, bounded $\ell _ { \infty }$ gradients, and $\ell _ { 1 }$ path length. Our advance combines sparse histories, square-root KL movement control, and drawdown cancellation to obtain the sharp Euclidean dimension dependence. The scalar variation problem also connects to online nonparametric regression through sequentia entropy [17] and adaptive forecasting of total-variation-bounded signals [18].

Adaptive histories and switching reductions. Fixed share tracks a changing best expert by continually redistributing weight [19]. Follow-the-Leading-History [20] and strongly adaptive reductions [21] organize learners born at diferent times; coin-betting methods improve interval guarantees with comparable time overhead [22]. Gokcesu and Gokcesu [23] develop near-linear tracking for mixable or exp-concave losses. RESET [24] obtains optimal switching regret for all segmentations simultaneously, and IRESET [25] extends logarithmic switching guarantees to curved OCO. Building on this interface, Wang et al. [1] approximate a moving comparator by an unbiased random sequence with controlled variance and expected switches. Their exp-concave instantiation gives a $d ^ { 2 / 3 }$ factor. Wang, Zhao, and Zhou [2] obtain the same dimension factor through an interval-to-dynamic reduction on arbitrary compact convex domains; their Theorem 3 gives the bound compared in Table 1. Our improvement charges comparator movement directly through Gaussian potential.

Eficient exp-concave optimization and alternative reductions. Mahalanobis projection is a principal cost of second-order online algorithms. Mhammedi and Gatmiry [26] reduce this cost through quasi-Newton steps, and LightONS [27] preserves optimal static regret with total time $O ( d ^ { 2 } T + d ^ { \omega } { \sqrt { T \log T } } )$ , where $\omega$ is the matrix-multiplication exponent. Transferring such projection savings to the Gaussian states in our algorithm would strengthen the computational result. Xie et al. [28] use a discounted-to-dynamic reduction for FTRL-based linear and logistic regression, illustrating another route that exploits the structure of particular curved losses. Our arithmetic bound counts the projection cost explicitly and applies to any domain admitting the stated oracle. Classical rank-one eigensystem modification [29, 30] motivates a possible acceleration; our conditional computational statement specifies the full spectral-update interface it requires.

## 3 Problem setup

At round $t \in [ T ]$ , the learner chooses $w _ { t } \in \mathcal W$ , then observes a diferentiable convex loss $f _ { t } : \mathcal { W } $ R and incurs $f _ { t } ( w _ { t } )$ . For $u _ { 1 : T } \in \mathcal { W } ^ { T }$ , define

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) = \sum _ { t = 1 } ^ { T } [ f _ { t } ( w _ { t } ) - f _ { t } ( u _ { t } ) ] , \qquad P _ { T } ( u ) = \sum _ { t = 2 } ^ { T } \| u _ { t } - u _ { t - 1 } \| _ { 2 } .
$$

The learner may depend on $d , T , D , G , \eta$ and the domain, but not on the comparator or its path length. Full information permits evaluation of the revealed loss and its gradient at additional points. We count these evaluations separately from projection and arithmetic costs. The curved learner queries one gradient at its own prediction; the outer ensemble used for the full frontier evaluates its base learners after observing $f _ { t }$ . All upper bounds hold for every realized admissible loss sequence. The lower bound uses an oblivious distribution of losses and applies to randomized learners. We write $\widetilde O$ for bounds hiding only logarithmic factors in $T$ when regularity parameters are fixed, and use natural logarithms unless a base is indicated.

Assumption 3.1 (Geometry and curvature). The set $\mathcal { W } \subset \mathbb { R } ^ { d }$ is nonempty, compact, and convex, with Euclidean diameter at most $D > 0$ . Every $f _ { t }$ is η-exp-concave on W and satisfies $\Vert \nabla f _ { t } ( w ) \Vert _ { 2 } \leq G$ for all $w \in \mathcal W$

Section A recalls the definition of exp-concavity, its curvature criterion, and representative losses satisfying this assumption. The analysis uses the following curvature and logarithmic parameters:

$$
\gamma = \operatorname* { m i n } \left\{ \frac { \eta } { 2 } , \frac { 1 } { 4 G D } \right\} , \qquad \lambda = \frac { \gamma } { 2 } , \qquad \Lambda _ { T } = 1 + \log ( 2 T ) .\tag{3}
$$

The degenerate case $G D = 0$ has zero regret and will be omitted.

The statistical assumptions above specify the loss class. To state the algorithm’s computational cost on a general domain, we also make explicit the projection operation used to keep its predictions feasible.

Assumption 3.2 (Exact-arithmetic projection oracle). The oracle returns an exact solution to

$$
\operatorname* { m i n } _ { w \in \mathcal { W } } ( w - a ) ^ { \top } A ( w - a )
$$

for every $a \in \mathbb { R } ^ { d }$ and $A \succ 0$ , at arithmetic cost ${ \mathsf { C } } _ { \mathrm { p r o j } } ( d )$

This oracle measures the cost of enforcing the domain constraint in the geometry of each Gaussian state. It enters the upper-bound implementation; the lower bound applies to every learner, regardless of its computational resources.

The arithmetic bounds use the dense spectral model: a symmetric eigendecomposition and reconstruction are charged $O ( d ^ { 3 } )$ operations. This is an idealized real-arithmetic convention. Section B.4 gives a separate rational implementation and its bit cost under an explicit approximateprojection oracle.

## 4 Algorithm and upper bounds

We first construct the curved learner, SparseGauss, and state its regret and computational guarantees. We then explain why its movement analysis gives the $d ^ { 1 / 3 }$ factor and how to extend the guarantee to all path budgets.

## 4.1 A common rank-one surrogate

After the master predicts $w _ { t } \in \mathcal W$ and observes $g _ { t } = \nabla f _ { t } ( w _ { t } )$ , all histories use

$$
\widetilde { f } _ { t } ( v ) = \langle g _ { t } , v - w _ { t } \rangle + \frac { \gamma } { 2 } \langle g _ { t } , v - w _ { t } \rangle ^ { 2 } , \qquad \gamma = \operatorname* { m i n } \left\{ \frac { \eta } { 2 } , \frac { 1 } { 4 G D } \right\} .\tag{4}
$$

This surrogate dominates the original one-step regret and is itself $\gamma / 2 \mathrm { - e x p \mathrm { - c o n c a v e } }$ . Its Hessian is rank one, which is essential: smoothing a comparator by $\mathcal { N } ( u _ { t } , \tau ^ { 2 } I )$ costs exactly

$$
\frac { \gamma \tau ^ { 2 } } { 2 } \lVert g _ { t } \rVert _ { 2 } ^ { 2 } ,
$$

with no ambient trace factor d. Thus the same surrogate both transfers regret back to the origina losses and keeps the smoothing error independent of dimension; Lemma B.1 proves the required domination and exp-concavity properties.

## 4.2 Sparse Gaussian histories

For dyadic scales $\tau _ { k } = D 2 ^ { - k } , k = 0 , \dots , \lfloor \log _ { 2 } T \rfloor$ , we run a deterministic logarithmic active set of histories. A history born at time i and scale k maintains a single Gaussian $\mathcal { N } ( m _ { t } , \Sigma _ { t } )$ satisfying

$$
m _ { t } \in \mathcal { W } , \qquad \tau _ { k } ^ { 2 } I \preceq \Sigma _ { t } \preceq D ^ { 2 } I .
$$

The exp-concave master averages the means, so the actual prediction is proper. Exponential tilting by (4) preserves Gaussianity. We then floor the covariance at $\tau _ { k } ^ { 2 } I$ and project the mean onto $\mathcal { W }$ in the resulting Mahalanobis metric. Both operations decrease KL from every comparator Gaussian $\mathcal { N } ( u , \tau _ { k } ^ { 2 } I )$

Birth times let the learner adapt to changes in the comparator, while the scale grid adapts to its unknown path length. Algorithm 1 summarizes one round of the resulting learner. The active-start schedule and master weights are specified in Section B.1.2, and the Gaussian updates in Section B.1.3.

```latex
Algorithm 1 SparseGauss: sparse Gaussian histories for the curved branch
Require: Horizon $T ,$ scales $\tau _ { k } = D 2 ^ { - k }$ , projection oracle, prior mean $w _ { 0 } \in \mathcal { W }$
1: Initialize the streaming active-start schedule and one Gaussian $\mathcal { N } ( w _ { 0 } , D ^ { 2 } I )$ for each newborn
scale
2: for $t = 1 , \dots , T$ do
3: Collect the means $\ v { x } _ { t } ^ { i , k }$ of all active histories and predict $\begin{array} { r } { w _ { t } = \sum _ { i , k } p _ { t } ^ { i , k } x _ { t } ^ { i , k } } \end{array}$
4: Observe $f _ { t }$ , evaluate $g _ { t } = \nabla f _ { t } ( w _ { t } )$ , and form the common surrogate (4)
5: Update master weights by exponential weighting on $\widetilde { f _ { t } } ( x _ { t } ^ { i , k } )$
6: for each active history $( i , k )$ do
7: Exponentially tilt its Gaussian by exp $[ - ( \gamma / 2 ) \widetilde { f } _ { t } ]$
8: Floor the covariance at $\tau _ { k } ^ { 2 } I$ and Mahalanobis-project the mean onto W
9: Prune expired starts; insert the new start $t + 1$ at every scale and renormalize
```

## 4.3 Regret and computational guarantees

The learner above runs all candidate scales simultaneously, so it adapts to the unknown path length. We state the resulting rate with fixed regularity parameters to make its dependence on $d , T ,$ , and $P _ { T }$ explicit. Recall that $\Lambda _ { T } = 1 + \log ( 2 T )$ ; the $O ( \cdot )$ notation below hides constants, while retaining all logarithmic factors.

Theorem 4.1 (Curved dynamic regret). Under Assumptions 3.1 and 3.2, fix D, $G , \eta > 0$ independently of d and T. The proper full-information learner SparseGauss does not require $P _ { T } ( u )$ and, for every comparator path with $P _ { T } ( u ) \leq c T / ( \sqrt { d } \Lambda _ { T } ^ { 2 } )$ , guarantees

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq O \left( d \Lambda _ { T } ^ { 2 } + d ^ { 1 / 3 } \Lambda _ { T } ^ { 4 / 3 } T ^ { 1 / 3 } P _ { T } ( u ) ^ { 2 / 3 } \right) .\tag{5}
$$

Here $c > 0$ and the constant hidden by $O ( \cdot )$ depend only on $D , G , \eta .$ The learner uses one gradient query per round, requires $O ( ( d ^ { 3 } + \mathsf { C } _ { \mathrm { p r o j } } ( d ) ) \log ^ { 2 } ( 2 T ) )$ operations per round in the dense spectral model, and stores $O ( d ^ { 2 } \log ^ { 2 } ( 2 T ) )$ real numbers. If an additional primitive performs the complete rank-one update and covariance flooring in $O ( d ^ { 2 } )$ operations, including reconstruction in the original coordinates, the $d ^ { 3 }$ term becomes $d ^ { 2 }$

The $d ^ { 1 / 3 }$ factor follows by optimizing the movement–smoothing tradeof. The path condition ensures that the balancing scale does not exceed the domain diameter. Theorem B.16 gives the full dependence on $D , G , \eta$ and the clipped-scale bound valid for every path. The next corollary removes the path restriction by adding projected-gradient learners.

At any time there are ${ \cal O } ( \log T )$ active starts and ${ \cal O } ( \log T )$ scales, hence $O ( \log ^ { 2 } T )$ Gaussian states. Each state requires one rank-one precision update, covariance flooring, and one Mahalanobis projection. Dense spectral computation gives the stated $O ( d ^ { 3 } )$ cost. Classical rank-one eigensystem methods [29, 30] motivate the additional update primitive; the conditional $O ( d ^ { 2 } )$ cost includes the full change of basis and covariance reconstruction. Storing one Gaussian per history keeps the total state count logarithmic in the number of starts and scales.

These state counts give the per-round cost in Theorem 4.1, hence $O ( T ( d ^ { 3 } + C _ { \mathrm { p r o j } } ( d ) ) \log ^ { 2 } ( 2 T ) )$ arithmetic operations over the horizon. A polynomial-bit implementation is given in Section B.4.

## 4.4 Why the dimension factor improves

The proof of Theorem 4.1 has three steps: control displacement within one history, pay for history restarts, and optimize the smoothing scale. The central obstacle is that a moving comparator prevents the usual KL potential from telescoping directly. For a fixed history and scale, let

$$
F _ { t } ( u ) = \mathrm { K L } \Big ( \mathcal { N } ( u , \tau ^ { 2 } I ) \lVert \mathcal { N } ( m _ { t } , \Sigma _ { t } ) \Big ) .
$$

For a fixed comparator, the one-step inequality in Lemma B.6 telescopes. Comparator movement introduces an extra term, which Lemma B.7 controls through the square root of the potential:

$$
\sqrt { F _ { t + 1 } ( v ) } \leq \sqrt { F _ { t + 1 } ( u ) } + \frac { \| v - u \| _ { 2 } } { \sqrt { 2 } \tau } .
$$

A single history has no time-uniform fixed-share envelope. Let $r _ { t }$ be its surrogate excess against the Gaussian comparator and $\textstyle S _ { q } = \sum _ { t = i } ^ { q } r _ { t }$ . If $S _ { q } \geq - H$ throughout an interval, the shifted energy

$$
E _ { t } = F _ { t } ( u _ { t } ) + \frac { \gamma } { 2 } ( S _ { t - 1 } + H )
$$

is nonnegative and controls movement throughout the interval (Lemma B.8). If $S _ { q } < - H$ for the first time, that negative prefix pays the master cost and the recursion restarts on the remaining sufix. The streaming-history property ensures that every interval exposes a fresh history in its first half. Deleting negative nodes leaves O(log T) positive nodes per component. This cancellation gives the low-variation interval guarantee in Lemma B.9; it is what allows sparse histories to retain the improved movement cost.

Finally, partition the comparator path into low-variation intervals and sum their costs. Adding the rank-one smoothing error yields the fixed-scale bound of Lemma B.11:

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \lesssim \frac { d \mathrm { \ p o l y l o g } T } { \gamma } + \frac { \sqrt { d } \mathrm { \ p o l y l o g } T } { \gamma \tau } P _ { T } + \gamma G ^ { 2 } T \tau ^ { 2 } .
$$

The second term is the cost of movement and the third is the smoothing error. Balancing them produces $d ^ { 1 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 }$ ; Lemma B.12 handles scales whose unconstrained optimizer falls outside $[ D / T , D ]$ . This completes the route from the local potential inequalities to Theorem 4.1.

## 4.5 Extension to all path budgets

For larger path budgets, projected-gradient learners can give a smaller regret than the curved branch. We combine SparseGauss with a logarithmic grid of their learning rates and use exp-concavity to compete with the best branch. This yields the next corollary, with additional gradient and projection calls because each base learner is evaluated at its own prediction.

Corollary 4.2 (Full upper frontier). Under Assumptions 3.1 and 3.2, fix $D , G , \eta > 0$ independently of d and T. A proper full-information learner that does not require $P _ { T } ( u )$ guarantees

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \le { \cal O } \biggl ( d \Lambda _ { T } ^ { 2 } + \operatorname* { m i n } \left\{ d ^ { 1 / 3 } \Lambda _ { T } ^ { 4 / 3 } T ^ { 1 / 3 } P _ { T } ( u ) ^ { 2 / 3 } , \sqrt { T ( 1 + P _ { T } ( u ) ) } , T \right\} \biggr ) .\tag{6}
$$

for every comparator path. The constant hidden by $O ( \cdot )$ depends only on $D , G , \eta$ . The learner uses ${ \cal O } ( \log T )$ loss evaluations, gradient evaluations, and Euclidean projection calls per round in addition to the curved branch’s internal work.

The first branch in (6) agrees with (5); the other branches control larger path budgets and unrestricted movement. The logarithmic aggregation cost is absorbed by the additive term. Corollary B.17 records the full parameter dependence, and Section B.3 proves it from Theorem B.16 and the projected-gradient bound in Lemma B.13.

## 5 Lower bounds and minimax optimality

We now ask whether any learner can improve the dimension dependence of the upper bound. The construction below uses the same Euclidean path length and fixes the diameter, gradient bound, and exp-concavity constant independently of d. Thus any remaining dimension factor reflects the dificulty of tracking the comparator.

## 5.1 A matching lower bound

Each loss reveals information about only one coordinate. Although the learner observes the entire loss function, the other coordinates remain unobserved on that round. The following theorem turns this restriction into a lower bound that applies even to randomized learners.

Theorem 5.1 (Matching normalized lower frontier). There are universal constants $c , c _ { 0 } > 0$ such that the following holds. Let

$$
\mathcal { W } _ { d } = \{ w \in \mathbb { R } ^ { d } : \| w \| _ { 2 } \leq 1 / 2 \}
$$

and let the adversary use only the losses

$$
f _ { j , y } ( w ) = \frac { 1 } { 2 } ( w _ { j } - y ) ^ { 2 } , \qquad j \in [ d ] , \quad y \in \{ - 1 , + 1 \} .\tag{7}
$$

This class has diameter $D = 1$ , gradient bound $G \leq 3 / 2$ , and exp-concavity coeficient at least $4 / 9$ uniformly in d. $I f T \geq c _ { 0 } d ^ { 2 }$ , then for every $P \geq 0$

$$
\begin{array} { r l } & { \underset { \textnormal { \tiny A } _ { f _ { 1 : T } } ~ o f ~ t h e ~ f o r m } { \operatorname* { i n f } } ~ \mathbb { E } _ { \mathsf { A } } \mathrm { R e g } _ { T } ( u _ { 1 : T } ) \ge c \left[ d + \operatorname* { m i n } \left\{ d ^ { 1 / 3 } T ^ { 1 / 3 } P ^ { 2 / 3 } , \sqrt { T P } , T \right\} \right] . } \\ & { \quad \quad \quad \quad u _ { 1 : T } \in { \mathcal { W } } _ { d } ^ { T } , ~ P _ { T } ( u ) \le P } \end{array}\tag{8}
$$

Why the construction is hard. In each time block, the comparator encodes independent hidden signs across a chosen number m of coordinates. Noisy labels reveal too little information to estimate those signs accurately before the next block. Lemma C.1 quantifies the one-coordinate prediction error, and Lemma C.2 combines these errors while respecting the Euclidean norm and path constraints.

Using all d coordinates gives the curved factor $d ^ { 1 / 3 }$ . As the path budget grows, choosing fewer active coordinates yields the square-root branch, followed by linear saturation. The stationary construction in Lemma C.3 supplies the additive d term even when the comparator never moves. The complete construction and the choices of m are given in Section C.

## 5.2 The minimax frontier

The upper bound applies to every admissible compact convex domain, whereas the lower bound already holds on a Euclidean ball. Together they determine the worst-case regret over the normalized class. The next corollary combines the two results and records the condition on the horizon under which they match.

Corollary 5.2 (Computational minimax closure, up to logarithms). For $D = 1$ ， $G = 3 / 2$ , and $\eta = 4 / 9$ , take the worst case over compact convex domains of diameter at most one and losses satisfying Assumption 3.1. For horizons $T \gtrsim d ^ { 2 }$ , the proper full-information minimax dynamic regret with path budget $P _ { T }$ has, up to polylogarithmic factors, the frontier

$$
\left| d + \operatorname * { m i n } \{ d ^ { 1 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } , \sqrt { T P _ { T } } , T \} . \right|
$$

With a Mahalanobis-projection oracle, the upper frontier is attained in near-linear time in T.

Proof. The lower bound follows from Theorem 5.1, since its ball and coordinate-square losses belong to the stated class. For the upper bound, Corollary 4.2 gives the curved branch, $\sqrt { T ( 1 + P _ { T } ) }$ , and the linear cap, together with an additive d, up to logarithms. For $P _ { T } \ge 1$ , the square-root branch is within a constant factor of $\sqrt { T P _ { T } }$ . For $0 < P _ { T } < 1$ and $T \geq d ^ { 2 }$ , the ratio of the curved term to $\sqrt { T P _ { T } }$ is $( d ^ { 2 } P _ { T } / T ) ^ { 1 / 6 } \leq 1$ , so the curved term is already the smaller branch. The case $P _ { T } = 0$ follows directly from the static term. This proves the displayed normalized frontier. □

## 6 Discussion and conclusion

Sparse Gaussian histories attain the sharp $d ^ { 1 / 3 }$ curved dimension factor, and coordinate-square losses show why this cost is unavoidable. The upper analysis couples rank-one smoothing with square-root KL movement control; negative excess-regret prefixes pay for the history restarts needed to maintain only logarithmically many active states. Combining the curved learner with projected-gradient learners gives the full minimax frontier when $T \gtrsim d ^ { 2 }$

The arithmetic result depends on the cost of Mahalanobis projection onto the domain. Projection deferral, as in LightONS [27], is a natural direction for reducing that cost while preserving the comparator-KL inequality. A second question is whether one gradient at the final prediction can support the entire frontier: the current curved learner has this property, whereas the full-frontier ensemble evaluates its bases separately. The role of more restrictive domain or loss geometry in reducing the efective dimension also remains open.

AI Usage. Nearly the entire research pipeline for this paper was carried out by Colombo, our laboratory’s internal auto-research system, powered by GPT-5.6 Sol. The system also conducted a Lean-backed article audit of the resulting manuscript. The authors subsequently reviewed and approved the mathematical claims, presentation, and formal artifacts, and take responsibility for the final manuscript. The complete Lean audit report and the system’s technical report will be made public at a later date.

## References

[1] Y. Wang, W. Yang, S. Yang, W. Jiang, Y. Wan, and L. Zhang. From switching to dynamic regret: a simple reduction via unbiased random sequences. arXiv:2609.20968, 2026.

[2] Y.-H. Wang, P. Zhao, and Z.-H. Zhou. On the relation between interval regret and dynamic regret. arXiv:2609.34423, 2026.

[3] E. Hazan. Introduction to Online Convex Optimization, second edition. MIT Press, 2022.

[4] E. C. Hall and R. M. Willett. Dynamical models and tracking regret in online convex programming. In Proceedings of ICML, PMLR 28(1):579–587, 2013.

[5] M. Zinkevich. Online convex programming and generalized infinitesimal gradient ascent. In Proceedings of ICML, pages 928–936, 2003.

[6] L. Zhang, S. Lu, and Z.-H. Zhou. Adaptive online learning in dynamic environments. In Advances in Neural Information Processing Systems 31, 2018.

[7] A. Cutkosky. Parameter-free, dynamic, and strongly-adaptive online learning. In Proceedings of ICML, PMLR 119:2250–2259, 2020.

[8] T. Koren and K. Y. Levy. Fast rates for exp-concave empirical risk minimization. In Advances in Neural Information Processing Systems 28, 2015.

[9] D. Baby and Y.-X. Wang. Optimal dynamic regret in exp-concave online learning. In Proceedings of COLT, PMLR 134:359–409, 2021.

[10] D. Baby and Y.-X. Wang. Optimal dynamic regret in proper online learning with strongly convex losses and beyond. In Proceedings of AISTATS, PMLR 151:1805–1845, 2022.

[11] Y.-J. Zhang, P. Zhao, and M. Sugiyama. Non-stationary online learning for curved losses: improved dynamic regret via mixability. In Proceedings of ICML, PMLR 267:77044–77068, 2025.

[12] Y.-J. Zhang, Y.-X. Wang, P. Zhao, and K. Jamieson. Universal dynamic portfolios. arXiv:2609.34643, 2026.

[13] O. Besbes, Y. Gur, and A. Zeevi. Non-stationary stochastic optimization. Operations Research, 63(5):1227–1244, 2015.

[14] A. Jacobsen, A. Rudi, F. Orabona, and N. Cesa-Bianchi. Dynamic regret reduces to kernelized static regret. In Advances in Neural Information Processing Systems 38, 2025.

[15] E. Hazan, A. Agarwal, and S. Kale. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2–3):169–192, 2007.

[16] V. Vovk. Competitive on-line statistics. International Statistical Review, 69(2):213–248, 2001.

[17] A. Rakhlin and K. Sridharan. Online non-parametric regression. In Proceedings of COLT, PMLR 35:1232–1264, 2014.

[18] D. Baby and Y.-X. Wang. Online forecasting of total-variation-bounded sequences. In Advances in Neural Information Processing Systems 32, 2019.

[19] M. Herbster and M. K. Warmuth. Tracking the best expert. Machine Learning, 32:151–178, 1998.

[20] E. Hazan and C. Seshadhri. Adaptive algorithms for online optimization. ECCC Report TR07-088, revision 1, 2007.

[21] A. Daniely, A. Gonen, and S. Shalev-Shwartz. Strongly adaptive online learning. In Proceedings of ICML, PMLR 37:1405–1411, 2015.

[22] K.-S. Jun, F. Orabona, S. Wright, and R. Willett. Improved strongly adaptive online learning using coin betting. In Proceedings of AISTATS, PMLR 54:943–951, 2017.

[23] K. Gokcesu and H. Gokcesu. Near-linear time algorithm with near-logarithmic regret per switch for mixable/exp-concave losses. arXiv:2109.13786, 2021.

[24] S. Pasteris, C. Hicks, V. Mavroudis, and M. Herbster. Online convex optimisation: the optimal switching regret for all segmentations simultaneously. In Advances in Neural Information Processing Systems 37, pages 78278–78298, 2024.

[25] W. Yang, Y. Wang, Y. Wan, and L. Zhang. Logarithmic switching regret for online convex optimization. In Proceedings of ICML, PMLR 306:146297–146314, 2026.

[26] Z. Mhammedi and K. Gatmiry. Quasi-Newton steps for eficient online exp-concave optimization. In Proceedings of COLT, PMLR 195:4473–4503, 2023.

[27] Y.-H. Wang, P. Zhao, and Z.-H. Zhou. A simple, optimal and eficient algorithm for online exp-concave optimization. In Proceedings of COLT, PMLR 336:6651–6691, 2026.

[28] Y.-F. Xie, Y.-J. Zhang, P. Zhao, and Z.-H. Zhou. Dynamic regret via discounted-to-dynamic reduction with applications to curved losses and Adam optimizer. arXiv:2602.08372, 2026.

[29] J. R. Bunch, C. P. Nielsen, and D. C. Sorensen. Rank-one modification of the symmetric eigenproblem. Numerische Mathematik, 31:31–48, 1978.

[30] M. Gu and S. C. Eisenstat. A stable and eficient algorithm for the rank-one modification of the symmetric eigenproblem. SIAM Journal on Matrix Analysis and Applications, 15(4):1266–1276, 1994.

## A Preliminaries

We collect the curvature facts used in the upper and lower bounds, together with examples that clarify the loss class in Assumption 3.1.

## A.1 Exp-concavity and directional curvature

A function $f : \mathcal { W } \to \mathbb { R }$ is η-exp-concave, for $\eta > 0$ , if $w \mapsto e ^ { - \eta f ( w ) }$ is concave on the convex set W [3, 15]. For diferentiable $f ,$ the supporting-hyperplane inequality for this concave function gives

$$
e ^ { - \eta f ( v ) } \leq e ^ { - \eta f ( w ) } \bigl ( 1 - \eta \langle \nabla f ( w ) , v - w \rangle \bigr ) , \qquad v , w \in \mathcal { W } .\tag{9}
$$

This is the first-order inequality used to obtain surrogate domination in Lemma B.1. Exp-concavity with parameter $\eta$ also implies exp-concavity with every parameter $\eta ^ { \prime } \in ( 0 , \eta ]$ , since $x \mapsto x ^ { \eta / \eta }$ is increasing and concave on $( 0 , \infty )$

For a $C ^ { 2 }$ loss on an open convex domain, the equivalent curvature criterion is

$$
\nabla ^ { 2 } f ( w ) \succeq \eta \nabla f ( w ) \nabla f ( w ) ^ { \top } .\tag{10}
$$

Diferentiating the exponential gives

$$
\nabla ^ { 2 } e ^ { - \eta f ( w ) } = \eta e ^ { - \eta f ( w ) } \big ( \eta \nabla f ( w ) \nabla f ( w ) ^ { \top } - \nabla ^ { 2 } f ( w ) \big ) .
$$

The same criterion sufices on a compact convex subset when the loss has a smooth extension to a neighborhood; on a lower-dimensional domain it is enough to check directions in its afine hull. We use this criterion to verify the smooth examples below; the main results assume diferentiability and exp-concavity as stated in Assumption 3.1.

Strong convexity requires curvature in every direction: in the smooth case, $\nabla ^ { 2 } f ( w ) \succeq \sigma I$ for some $\sigma > 0$ . If also $\| \nabla f ( w ) \| _ { 2 } \leq G$ with $G > 0$ , then (10) holds for $\eta = \sigma / G ^ { 2 }$ . Exp-concavity allows the Hessian to be rank one, as the examples below illustrate, so it need not imply strong convexity in the ambient space.

## A.2 Representative exp-concave losses

For a linear prediction $z = x ^ { \top }$ w and a scalar loss $\ell ( z )$ , the gradient and Hessian of $f ( w ) = \ell ( x ^ { \top } w )$ are $\ell ^ { \prime } ( z ) x$ and $\ell ^ { \prime \prime } ( z ) x x ^ { \top }$ Thus the scalar condition $\ell ^ { \prime \prime } ( z ) \geq \eta ( \ell ^ { \prime } ( z ) ) ^ { 2 }$ verifies (10) without a dimension factor.

Squared prediction loss. For $\begin{array} { r } { f ( w ) = \frac { 1 } { 2 } ( x ^ { \top } w - y ) ^ { 2 } } \end{array}$ on a domain where $| x ^ { \top } w - y | \leq B$ with $B > 0$ , the scalar derivatives are $\ell ^ { \prime } ( z ) = z - y$ and $\ell ^ { \prime \prime } ( z ) = 1$ . The loss is therefore 1/B<sup>2</sup>-exp-concave. Its gradient norm is at most $B \Vert { \boldsymbol { x } } \Vert _ { 2 }$ . The coordinate-square losses in Theorem 5.1 take $\boldsymbol { x } = \boldsymbol { e } _ { j }$ and $B = 3 / 2$ , giving the coeficient $4 / 9$ used there.

Logistic prediction loss. For $y \in \{ - 1 , 1 \}$ and $f ( w ) = \log ( 1 + e ^ { - y x ^ { \top } w } )$ , the derivatives satisfy

$$
\ell ^ { \prime } ( z ) = { \frac { - y } { 1 + e ^ { y z } } } , \qquad \ell ^ { \prime \prime } ( z ) = { \frac { e ^ { y z } } { ( 1 + e ^ { y z } ) ^ { 2 } } } , \qquad { \frac { \ell ^ { \prime \prime } ( z ) } { ( \ell ^ { \prime } ( z ) ) ^ { 2 } } } = e ^ { y z } .
$$

On a domain with $| x ^ { \top } w | \leq B$ , this loss is $e ^ { - B _ { - } } \mathrm { e x p - c o n c a v e }$ and has gradient norm at most $\| { \boldsymbol { x } } \| _ { 2 }$ The bounded prediction range supplies a uniform positive exp-concavity parameter.

Negative logarithmic return. For a return vector r and a domain on which $r ^ { \top } w > 0$ , the portfolio loss $f ( w ) = - \log ( r ^ { \top } w )$ is 1-exp-concave because $e ^ { - f ( w ) } = r ^ { \top } w$ is afine [3]. $\boldsymbol { \mathrm { I f } } \ : \boldsymbol { r } ^ { \top } \boldsymbol { w } \geq b > 0$ its gradient norm is at most $\| r \| _ { 2 } / b$ . The lower bound on returns is needed for the uniform gradient bound in Assumption 3.1.

## A.3 Aggregation under exp-concavity

For predictions $w _ { i } \in \mathcal W$ and weights $p _ { i } \geq 0$ summing to one, convexity of the domain makes $\bar { w } = \textstyle \sum _ { i } p _ { i } w _ { i }$ feasible. Concavity of $e ^ { - \eta f }$ then yields

$$
f ( \bar { w } ) \leq - \frac { 1 } { \eta } \log \left( \sum _ { i } p _ { i } e ^ { - \eta f ( w _ { i } ) } \right) .\tag{11}
$$

This inequality is the aggregation step for both the surrogate master in Section B.1.2 and the outer ensemble in Section B.3. The former uses the surrogate’s exp-concavity parameter λ, whereas the latter uses η for the original losses.

## B Proofs for Section 4

We collect the supporting lemmas with their proofs, then prove the parameter-explicit versions of Theorem 4.1 and corollary 4.2. The final subsection establishes the finite-precision implementation and its anytime extension.

## B.1 Useful lemmas

The argument proceeds from surrogate domination and history aggregation to Gaussian movement control and restart cancellation. The path-partition and projected-gradient lemmas then supply the two branches of the full frontier. We also record the numerical robustness statement used in the implementation theorem.

## B.1.1 Rank-one surrogate: domination and curvature

After the algorithm forms its proper prediction $w _ { t }$ , let $g _ { t } = \nabla f _ { t } ( w _ { t } )$ and define

$$
\widetilde { f } _ { t } ( v ) = \langle g _ { t } , v - w _ { t } \rangle + \frac { \gamma } { 2 } \langle g _ { t } , v - w _ { t } \rangle ^ { 2 } .\tag{12}
$$

The first step is to justify analyzing the quadratic surrogate instead of the original losses. The next lemma gives both ingredients needed later: regret domination for the final guarantee and exp-concavity for the master aggregation.

Lemma B.1 (Domination and surrogate exp-concavity). For every $v \in \mathcal W$

$$
f _ { t } ( w _ { t } ) - f _ { t } ( v ) \leq \widetilde { f } _ { t } ( w _ { t } ) - \widetilde { f } _ { t } ( v ) = - \widetilde { f } _ { t } ( v ) .\tag{13}
$$

Moreover, $\widetilde { f } _ { t }$ is λ-exp-concave on W.

Proof. Because $2 \gamma \leq \eta$ , the loss is also 2γ-exp-concave. Put $a \ = \ \langle g _ { t } , v \ - \ w _ { t } \rangle$ . Concavity of $e ^ { - 2 \gamma \bar { f } _ { t } }$ gives

$$
e ^ { - 2 \gamma f _ { t } ( v ) } \leq e ^ { - 2 \gamma f _ { t } ( w _ { t } ) } ( 1 - 2 \gamma a ) .
$$

The choice of $\gamma$ implies $| 2 \gamma a | \le 2 \gamma G D \le 1 / 2$ . For $| x | \le 1 / 2 , - \log ( 1 - x ) \ge x + x ^ { 2 } / 4$ . Taking logarithms yields

$$
f _ { t } ( v ) - f _ { t } ( w _ { t } ) \geq a + \frac { \gamma } { 2 } a ^ { 2 } = \widetilde { f } _ { t } ( v ) ,
$$

which is (13).

For exp-concavity, write $s = \langle g _ { t } , v - w _ { t } \rangle$ . Then

$$
\nabla \widetilde { f } _ { t } ( v ) = ( 1 + \gamma s ) g _ { t } , \qquad \nabla ^ { 2 } \widetilde { f } _ { t } ( v ) = \gamma g _ { t } g _ { t } ^ { \top } .
$$

Since $| \gamma s | \le 1 / 4$

$$
\nabla ^ { 2 } \widetilde { f } _ { t } ( \boldsymbol { v } ) \boldsymbol { \Sigma } \stackrel { \gamma } { = } \frac { \gamma } { 2 } ( 1 + \gamma s ) ^ { 2 } g _ { t } g _ { t } ^ { \top } = \lambda \nabla \widetilde { f } _ { t } ( \boldsymbol { v } ) \nabla \widetilde { f } _ { t } ( \boldsymbol { v } ) ^ { \top } .
$$

This is the Hessian criterion for λ-exp-concavity.

## B.1.2 Sparse histories and the exp-concave master

The deterministic streaming schedule follows Hazan and Seshadhri [20]. Its role is to expose a fresh history early in any interval while keeping only logarithmically many states. For an integer $i \geq 1$ let $\nu _ { 2 } ( i )$ be the largest $k \geq 0$ such that $2 ^ { k }$ divides $i ,$ and define its lifetime

$$
\ell ( i ) = 2 ^ { \nu _ { 2 } ( i ) + 2 } .\tag{14}
$$

At time t, keep

$$
S _ { t } = \{ i \in [ t ] : t - i \leq \ell ( i ) \} .\tag{15}
$$

Lemma B.2 (Streaming-history properties). The sets (15) satisfy:

$( H 1 ) \ t \in S _ { t }$ and $| S _ { t } | \le c _ { H }$ log(2t) for a universal c<sub>H</sub>;

(H2) once i leaves the active set, it never returns;

(H3) for every interval $[ r , s ] \subseteq [ T ]$ , there is an $i \in { \mathcal { S } } _ { s } \cap [ r , \lfloor ( r + s ) / 2 \rfloor ]$

Consequently, the history born at such an i is active on every round from i through s.

Proof. The first assertion in (H1) and (H2) are immediate from (15). For the size bound, fix k. Integers with $\nu _ { 2 } ( i ) = k$ are separated by $2 ^ { k + 1 }$ , and an active such integer lies in an interval of length $2 ^ { k + 2 }$ . Hence only a universal constant number occur for each $k ,$ and $k \leq \lfloor \log _ { 2 } t \rfloor$

For (H3), put $\Delta = s - r$ . If $\Delta < 2$ , the claim is immediate. Otherwise choose the largest $2 ^ { k } \leq \Delta / 2$ . The interval $[ r , ( r + s ) / 2 ]$ has length at least $2 ^ { k }$ , so it contains an integer i divisible by $2 ^ { k }$ . Thus $\ell ( i ) \geq 2 ^ { k + 2 } > \Delta \geq s - i ,$ and $i \in S _ { s }$ . Property (H2) ensures activity throughout [i, s].

Define the smoothing grid and its prior by

$$
K = \lfloor \log _ { 2 } T \rfloor , \qquad \tau _ { k } = D 2 ^ { - k } , \quad k = 0 , \dots , K , \qquad q _ { k } = \frac { 1 } { K + 1 } .
$$

For every active pair $( i , k )$ , maintain one Gaussian history learner $\mathcal { E } _ { i , k }$ . At round $t ,$ it predicts a point $x _ { t } ^ { i , k } \in \mathcal { W }$ . Initialize $p _ { 1 } ^ { 1 , k } = q _ { k }$ . The master has weights $p _ { t } ^ { i , k }$ on active pairs and predicts

$$
w _ { t } = \sum _ { k = 0 } ^ { K } \sum _ { i \in \mathcal { S } _ { t } } p _ { t } ^ { i , k } x _ { t } ^ { i , k } \in \mathcal { W } .\tag{16}
$$

After observing (12), perform the exponential update

$$
\widehat { p } _ { t + 1 } ^ { i , k } = \frac { p _ { t } ^ { i , k } e ^ { - \lambda \widetilde { f } _ { t } ( x _ { t } ^ { i , k } ) } } { \sum _ { j \in \mathcal { S } _ { t } , \ell } p _ { t } ^ { j , \ell } e ^ { - \lambda \widetilde { f } _ { t } ( x _ { t } ^ { j , \ell } ) } } .
$$

Before round $t + 1$ , give total weight $1 / ( t + 1 )$ to the new start $t + 1$ , split according to $q _ { k }$ , multiply surviving old weights by $t / ( t + 1 )$ , delete starts outside $\boldsymbol { S } _ { t + 1 }$ , and renormalize.

The schedule guarantees that a suitable history exists on any interval. We next bound the price of following that history. A guarantee for every prefix, rather than only its full lifetime, is needed because the cancellation argument may stop a history at its first large negative excess-regret prefix.

Lemma B.3 (Prefix overhead to every surviving history). If $\mathcal { E } _ { i , k }$ is active through round $b \geq i$ then every prefix $[ i , q ] \subseteq [ i , b ]$ satisfies

$$
\sum _ { t = i } ^ { q } [ \widetilde f _ { t } ( w _ { t } ) - \widetilde f _ { t } ( x _ { t } ^ { i , k } ) ] \leq a _ { k } , \qquad a _ { k } = \frac { 1 } { \lambda } \log \frac { T + 1 } { q _ { k } } .\tag{17}
$$

For the uniform scale prior, write

$$
a = \lambda ^ { - 1 } \log ( ( T + 1 ) ( K + 1 ) ) .\tag{18}
$$

Proof. By exp-concavity of the surrogate and (16),

$$
e ^ { - \lambda \widetilde { f _ { t } } ( w _ { t } ) } \geq \sum _ { j , \ell } p _ { t } ^ { j , \ell } e ^ { - \lambda \widetilde { f _ { t } } ( x _ { t } ^ { j , \ell } ) } .
$$

Rearranging this inequality gives the one-step comparison

$$
\lambda [ \widetilde f _ { t } ( w _ { t } ) - \widetilde f _ { t } ( x _ { t } ^ { i , k } ) ] \leq \log \frac { \widehat { p } _ { t + 1 } ^ { i , k } } { p _ { t } ^ { i , k } } .\tag{19}
$$

At birth, pruning renormalization can only increase mass, so $p _ { i } ^ { i , k } \geq q _ { k } / i$ . For $t > i ,$ , survival and renormalization give $p _ { t } ^ { i , k } \geq ( ( t - 1 ) / t ) \widehat { p } _ { t } ^ { i , k }$ . Summing (19),

$$
\begin{array} { r l } { \displaystyle \lambda \sum _ { t = i } ^ { q } [ \widetilde f _ { t } ( w _ { t } ) - \widetilde f _ { t } ( x _ { t } ^ { i , k } ) ] \leq - \log p _ { i } ^ { i , k } + \log \widehat p _ { q + 1 } ^ { i , k } + \displaystyle \sum _ { t = i + 1 } ^ { q } \log \frac { \widehat p _ { t } ^ { i , k } } { p _ { t } ^ { i , k } } } & { } \\ { \leq \log \displaystyle \frac { i } { q _ { k } } + 0 + \displaystyle \sum _ { t = i + 1 } ^ { q } \log \frac { t } { t - 1 } } & { } \\ { \displaystyle } & { = \log \displaystyle \frac { q } { q _ { k } } \leq \log \frac { T + 1 } { q _ { k } } . } \end{array}
$$

The terminal term is nonpositive because $\widehat { p } _ { q + 1 } ^ { i , k } \leq 1$

## B.1.3 One Gaussian history: update, feasibility, and KL control

Fix a scale $\tau \in [ D / T , D ]$ . A history maintains $P _ { t } = { \mathcal { N } } ( m _ { t } , \Sigma _ { t } )$ with

$$
m _ { t } \in \mathcal { W } , \qquad \tau ^ { 2 } I _ { d } \preceq \Sigma _ { t } \preceq D ^ { 2 } I _ { d } .\tag{20}
$$

At birth, $P _ { i } = \mathcal { N } ( w _ { 0 } , D ^ { 2 } I _ { d } )$ for a fixed $w _ { 0 } \in \mathcal { W }$

Closed-form rank-one Gaussian update. Let $A _ { t } = \Sigma _ { t } ^ { - 1 }$ . Multiplying the Gaussian density by $e ^ { - \lambda \widetilde { f _ { t } } }$ gives a Gaussian $R _ { t + 1 } = \mathscr { N } ( a _ { t + 1 } , C _ { t + 1 } )$ with

$$
A _ { t + 1 } ^ { + } = A _ { t } + \lambda \gamma g _ { t } g _ { t } ^ { \top } ,\tag{21}
$$

$$
b _ { t + 1 } ^ { + } = A _ { t } m _ { t } - \lambda g _ { t } + \lambda \gamma ( g _ { t } ^ { \top } w _ { t } ) g _ { t } ,\tag{22}
$$

$$
C _ { t + 1 } = ( A _ { t + 1 } ^ { + } ) ^ { - 1 } , \qquad a _ { t + 1 } = C _ { t + 1 } b _ { t + 1 } ^ { + } .\tag{23}
$$

These identities follow by collecting the quadratic and linear terms in the exponent. Sherman– Morrison and the matrix determinant lemma evaluate the update in $O ( d ^ { 2 } )$ arithmetic operations once an eigensystem of $A _ { t }$ is stored.

Let $C _ { t + 1 } = U \mathrm { d i a g } ( c _ { 1 } , \ldots , c _ { d } ) U ^ { \top }$ and define the covariance floor

$$
\begin{array} { r } { C _ { t + 1 } ^ { \flat } = U \mathrm { d i a g } ( c _ { j } \vee \tau ^ { 2 } ) U ^ { \top } . } \end{array}\tag{24}
$$

Then project the tilted mean in the corresponding Mahalanobis metric:

$$
m _ { t + 1 } \in \arg \operatorname* { m i n } _ { m \in \mathcal { W } } ( m - a _ { t + 1 } ) ^ { \top } ( C _ { t + 1 } ^ { \flat } ) ^ { - 1 } ( m - a _ { t + 1 } ) , \qquad \Sigma _ { t + 1 } = C _ { t + 1 } ^ { \flat } .\tag{25}
$$

Because the tilt only increases precision, $C _ { t + 1 } \preceq \Sigma _ { t } \preceq D ^ { 2 } I ;$ flooring raises eigenvalues only to $\tau ^ { 2 } \le D ^ { 2 }$ , so (20) is preserved.

For $u \in \mathcal W$ , set

$$
Q _ { u } = { \mathcal { N } } ( u , \tau ^ { 2 } I _ { d } ) .\tag{26}
$$

Lemma B.4 (Comparator-KL monotonicity of the feasibility correction). For every $u \in \mathcal W$

$$
\mathrm { K L } ( Q _ { u } \| \mathcal { N } ( m _ { t + 1 } , \Sigma _ { t + 1 } ) ) \leq \mathrm { K L } ( Q _ { u } \| R _ { t + 1 } ) .\tag{27}
$$

Proof. First keep the mean $a = a _ { t + 1 }$ fixed and replace $C = C _ { t + 1 }$ by $C ^ { \flat } = C _ { t + 1 } ^ { \flat }$ . In an eigenbasis of $C ,$ twice the covariance and mean-dependent part of the KL divergence is a sum of

$$
h _ { j } ( c ) = \frac { \tau ^ { 2 } + r _ { j } ^ { 2 } } { c } + \log c ,
$$

where $r _ { j }$ is the corresponding coordinate of $u - a$ . If $c \leq \tau ^ { 2 }$

$$
h _ { j } ^ { \prime } ( c ) = \displaystyle \frac { c - \tau ^ { 2 } - r _ { j } ^ { 2 } } { c ^ { 2 } } \leq 0 .
$$

Thus increasing every eigenvalue below $\tau ^ { 2 }$ to $\tau ^ { 2 }$ cannot increase KL.

Now put $A = ( C ^ { \flat } ) ^ { - 1 }$ and let $m = m _ { t + 1 }$ be the metric projection of a onto $\mathcal { W }$ . Its variational inequality is

$$
( a - m ) ^ { \top } A ( u - m ) \leq 0 \qquad ( u \in \mathcal { W } ) .
$$

Expanding squared norms gives

$$
\left\| u - m \right\| _ { A } ^ { 2 } \leq \left\| u - a \right\| _ { A } ^ { 2 } - \left\| a - m \right\| _ { A } ^ { 2 } .
$$

The covariance terms are unchanged during mean projection, so KL again cannot increase. □

Mix loss and one-step potential. For the Gaussian $P _ { t }$ , define

$$
m _ { t } ^ { \operatorname* { m i x } } = - \frac { 1 } { \lambda } \log \mathbb { E } _ { U \sim P _ { t } } e ^ { - \lambda \widetilde { f } _ { t } ( U ) } .
$$

Lemma B.5 (Gaussian mixability gap). The history mean satisfies

$$
\begin{array} { r } { \tilde { f } _ { t } ( m _ { t } ) \leq m _ { t } ^ { \operatorname* { m i x } } . } \end{array}\tag{28}
$$

Proof. Under $P _ { t } ,$ the scalar $X \ = \ \langle g _ { t } , U \ - \ w _ { t } \rangle$ is Gaussian with mean $\textit { a } = \langle g _ { t } , m _ { t } \textit { - } w _ { t } \rangle$ and variance $v = g _ { t } ^ { \top } \Sigma _ { t } g _ { t }$ . Since $\lambda = \gamma / 2 , - \lambda \widetilde { f } _ { t } ( U ) = - \lambda X - \lambda ^ { 2 } X ^ { 2 }$ . With $s = 2 \lambda ^ { 2 } v$ , direct Gaussian integration gives

$$
\log \frac { \mathbb { E } e ^ { - \lambda X - \lambda ^ { 2 } X ^ { 2 } } } { e ^ { - \lambda a - \lambda ^ { 2 } a ^ { 2 } } } = - \frac { 1 } { 2 } \log ( 1 + s ) + \frac { s ( 1 + 2 \lambda a ) ^ { 2 } } { 4 ( 1 + s ) } .\tag{29}
$$

Both $m _ { t }$ and $w _ { t }$ lie in W, so $| \lambda a | \le \gamma G D / 2 \le 1 / 8$ and $( 1 + 2 \lambda a ) ^ { 2 } \leq 2 5 / 1 6$ . Since $\log ( 1 + s ) \geq s / ( 1 + s )$ ， the right-hand side of (29) is at most $- 7 s / [ 6 4 ( 1 + s ) ] \leq 0$ . Therefore

$$
\mathbb { E } e ^ { - \lambda \widetilde { f } _ { t } ( U ) } \le e ^ { - \lambda \widetilde { f } _ { t } ( m _ { t } ) } ,
$$

which is equivalent to (28).

For a comparator path and one history, write

$$
\begin{array} { r } { Q _ { t } = \mathcal { N } ( u _ { t } , \tau ^ { 2 } I _ { d } ) , \qquad F _ { t } ( u ) = \mathrm { K L } ( Q _ { u } \| P _ { t } ) , \qquad r _ { t } = \widetilde { f } _ { t } ( m _ { t } ) - \mathbb { E } _ { Q _ { t } } \widetilde { f } _ { t } . } \end{array}\tag{30}
$$

We can now combine the mix-loss bound with the feasibility correction. The resulting inequality pays for one round of surrogate excess by a decrease in KL to a fixed comparator; the next step will account for changes in that comparator.

Lemma B.6 (One-step KL inequality). Every history update satisfies

$$
\lambda r _ { t } \le F _ { t } ( u _ { t } ) - F _ { t + 1 } ( u _ { t } ) .\tag{31}
$$

Proof. The exponential-update identity gives

$$
m _ { t } ^ { \mathrm { m i x } } - \mathbb { E } _ { Q _ { t } } \widetilde { f } _ { t } = \frac { 1 } { \lambda } \left[ { \mathrm { K L } } ( Q _ { t } \Vert P _ { t } ) - { \mathrm { K L } } ( Q _ { t } \Vert R _ { t + 1 } ) \right] .
$$

Use lemma B.5 on the left and lemma B.4 on the right.

To telescope the preceding inequality along a moving path, we must compare potentials at consecutive comparator points. Taking the square root makes the Gaussian KL Lipschitz in the comparator mean, with sensitivity controlled by the covariance floor. This is the source of the improved movement coeficient.

Lemma B.7 (Square-root KL movement). For every $u , v \in \mathcal { W }$

$$
\sqrt { F _ { t + 1 } ( v ) } \leq \sqrt { F _ { t + 1 } ( u ) } + \frac { \Vert v - u \Vert _ { 2 } } { \sqrt { 2 } \tau } .\tag{32}
$$

At a history’s birth time $i ,$

$$
F _ { i } ( u _ { i } ) \leq B _ { \tau } , \qquad B _ { \tau } = \frac { 1 } { 2 } \left[ 1 + d ( \rho ^ { 2 } - 1 - \log \rho ^ { 2 } ) \right] , \quad \rho = \frac { \tau } { D } .\tag{33}
$$

Proof. For fixed $P _ { t + 1 } = \mathcal { N } ( m , \Sigma )$ ,

$$
F _ { t + 1 } ( u ) = c + { \frac { 1 } { 2 } } ( u - m ) ^ { \mathsf { T } } \Sigma ^ { - 1 } ( u - m ) , \qquad c = { \frac { 1 } { 2 } } \left[ \tau ^ { 2 } \operatorname { t r } ( \Sigma ^ { - 1 } ) - d + \log { \frac { \operatorname* { d e t } \Sigma } { \tau ^ { 2 d } } } \right] \geq 0 .
$$

Define the augmented vector

$$
z ( u ) = \left( \sqrt { c } , \frac { 1 } { \sqrt { 2 } } \Sigma ^ { - 1 / 2 } ( u - m ) \right) .
$$

Then $\sqrt { F _ { t + 1 } ( u ) } = \| z ( u ) \| _ { 2 }$ . By the triangle inequality and $\Sigma \succeq \tau ^ { 2 } I$

$$
\sqrt { F _ { t + 1 } ( v ) } \leq \sqrt { F _ { t + 1 } ( u ) } + \frac { 1 } { \sqrt { 2 } } \left. \Sigma ^ { - 1 / 2 } ( v - u ) \right. _ { 2 } \leq \sqrt { F _ { t + 1 } ( u ) } + \frac { \left. v - u \right. _ { 2 } } { \sqrt { 2 } \tau } .
$$

At birth, $P _ { i } = \mathcal { N } ( w _ { 0 } , D ^ { 2 } I )$ ; the Gaussian KL formula and $\| u _ { i } - w _ { 0 } \| \le D$ give (33).

## B.1.4 Shifted drawdown energy and recursive cancellation

The movement inequality in Lemma B.7 is useful only while its starting potential is controlled. Negative excess regret can increase that potential, so we first absorb cumulative excess into a shifted energy. The resulting bound applies until a large negative prefix appears. We then charge the extra history pieces exposed by recursion to that negative prefix, closing the regret bound.

Lemma B.8 (Shifted drawdown energy). Consider a fresh scale-τ history on [i, b]. If

$$
\sum _ { t = i } ^ { q } r _ { t } \geq - H \qquad f o r \ e v e r y \ q \in [ i , b ] ,\tag{34}
$$

then the cumulative excess satisfies

$$
\sum _ { t = i } ^ { b } r _ { t } \leq \frac { 1 } { \lambda } \left( \sqrt { B _ { \tau } + \lambda H } + \frac { P _ { i : b } } { \sqrt { 2 } \tau } \right) ^ { 2 } - H ,\tag{35}
$$

where $\begin{array} { r } { P _ { i : b } = \sum _ { t = i + 1 } ^ { b } \| u _ { t } - u _ { t - 1 } \| _ { 2 } } \end{array}$

Proof. Put $\begin{array} { r } { S _ { q } = \sum _ { t = i } ^ { q } r _ { t } , S _ { i - 1 } = 0 } \end{array}$ , and

$$
E _ { t } = F _ { t } ( u _ { t } ) + \lambda ( S _ { t - 1 } + H ) .
$$

The drawdown assumption makes the added term nonnegative. By (31),

$$
F _ { t + 1 } ( u _ { t } ) + \lambda ( S _ { t } + H ) \leq E _ { t } .
$$

For $t < b ,$ use (32) and the fact that adding the same nonnegative scalar under both square roots preserves the movement inequality:

$$
\sqrt { E _ { t + 1 } } \leq \sqrt { E _ { t } } + \frac { \Vert u _ { t + 1 } - u _ { t } \Vert _ { 2 } } { \sqrt { 2 } \tau } .
$$

Iteration and (33) give

$$
\sqrt { E _ { b } } \leq \sqrt { B _ { \tau } + \lambda H } + \frac { P _ { i : b } } { \sqrt { 2 } \tau } .
$$

Applying (31) at t = b yields $F _ { b + 1 } ( u _ { b } ) + \lambda ( S _ { b } + H ) \le E _ { b }$ . Drop the terminal KL term.

Fix one scale and abbreviate

$$
L = \lceil \log _ { 2 } ( T + 1 ) \rceil , \quad A _ { \tau } = a + \frac { B _ { \tau } } { \lambda } , \quad H _ { \tau } = 1 2 8 L A _ { \tau } , \quad C _ { \tau } = B _ { \tau } + \lambda H _ { \tau } , \quad V _ { \tau } = \frac { \tau \sqrt { C _ { \tau } } } { 5 1 2 L } .\tag{36}
$$

For an interval I, define

$$
\mathcal { R } _ { \tau } ( I ) = \sum _ { t \in I } [ \widetilde { f } _ { t } ( w _ { t } ) - \mathbb { E } _ { Q _ { t } } \widetilde { f } _ { t } ] .\tag{37}
$$

The shifted-energy bound concerns one fresh history and assumes that its cumulative excess never falls below a threshold. The next lemma removes that condition: whenever it fails, the negative prefix pays for the additional histories introduced by recursion. This converts a conditional history bound into an unconditional interval bound for the master.

Lemma B.9 (Low-variation interval bound). If $P _ { I } \leq V _ { \tau }$ , then

$$
\mathcal { R } _ { \tau } ( I ) \leq \frac { H _ { \tau } } { 6 4 } .\tag{38}
$$

Proof. Recursively decompose $I = [ r , s ]$ . By lemma B.2, choose $i \in [ r , \lfloor ( r + s ) / 2 \rfloor ]$ whose history remains active through s. Recurse on $[ r , i - 1 ]$ . On [i, s], define

$$
S _ { q } = \sum _ { t = i } ^ { q } [ \widetilde { f } _ { t } ( x _ { t } ^ { i , k } ) - \mathbb { E } _ { Q _ { t } } \widetilde { f } _ { t } ] .
$$

If $S _ { q } \geq - H _ { \tau }$ for every $q ,$ call the node good and terminate its right branch. Otherwise let $q$ be the first crossing of $- H _ { \tau }$ , charge $[ i , q ]$ as a bad node, and recurse on $[ q + 1 , s ]$ . The charged pieces and recursive subintervals are disjoint and partition I.

For a good node, lemmas B.3 and B.8 and $P _ { i : s } \leq P _ { I } \leq V _ { \tau }$ imply

$$
\begin{array} { l } { \displaystyle \mathcal { R } _ { \tau } ( [ i , s ] ) \leq a + \frac { 1 } { \lambda } \left( \sqrt { C _ { \tau } } + \frac { V _ { \tau } } { \sqrt { 2 } \tau } \right) ^ { 2 } - H _ { \tau } } \\ { \displaystyle \leq \frac { H _ { \tau } } { 6 4 L } . } \end{array}
$$

Indeed, with $\varepsilon = V _ { \tau } / ( \sqrt { 2 } \tau \sqrt { C _ { \tau } } ) = 1 / ( 5 1 2 \sqrt { 2 } L )$ , the first line equals

$$
A _ { \tau } + ( 2 \varepsilon + \varepsilon ^ { 2 } ) \frac { C _ { \tau } } { \lambda } .
$$

Since $C _ { \tau } / \lambda = B _ { \tau } / \lambda + H _ { \tau } \leq ( 1 + 1 2 8 L ) A _ { \tau }$ , this is at most $2 A _ { \tau } = H _ { \tau } / ( 6 4 L )$

For a bad node, lemma B.3 and the crossing definition give the charge

$$
a - H _ { \tau } \leq - { \frac { 1 2 7 } { 1 2 8 } } H _ { \tau } .
$$

Remove all bad nodes from the recursion tree. Every remaining connected component consists only of left-child edges, because a right child is created only by a bad node. Along a left edge the interval length at least halves, so each component has at most $L$ good nodes. Removing $N _ { \mathrm { b a d } }$ binary-tree nodes creates at most $2 N _ { \mathrm { b a d } } + 1$ components. Hence

$$
N _ { \mathrm { g o o d } } \leq L ( 2 N _ { \mathrm { b a d } } + 1 ) .
$$

The total charge is at most

$$
\frac { H _ { \tau } } { 6 4 L } L ( 2 N _ { \mathrm { b a d } } + 1 ) - \frac { 1 2 7 H _ { \tau } } { 1 2 8 } N _ { \mathrm { b a d } } \leq \frac { H _ { \tau } } { 6 4 } .
$$

## B.1.5 Path partition, scale optimization, and endpoint regimes

Lemma B.9 gives an interval cost in terms of its internal path length. Partitioning a general path at a scale-dependent variation threshold converts this local statement into a bound linear in $P _ { T } / \tau$ . We optimize the scale and include both endpoints.

Lemma B.10 (Variation partition). For every $V > 0$ , every comparator path can be partitioned into consecutive intervals $I _ { 1 } , \ldots , I _ { M }$ satisfying

$$
P _ { I _ { j } } \leq V , \qquad M \leq 1 + { \frac { P _ { T } } { V } } .
$$

Proof. Greedily extend the current interval until the next jump would make its internal variation exceed V, then cut immediately before that jump. For every completed interval, its internal variation plus the triggering boundary jump exceeds V. These collections of jumps are disjoint, so $( M - 1 ) V < P _ { T }$ □

Combining the variation partition with the low-variation interval bound gives a guarantee for an arbitrary comparator path. The remaining term is the error from Gaussian smoothing. The next lemma displays these two contributions before optimizing the scale.

Lemma B.11 (Explicit fixed-scale guarantee). For every $\tau = \tau _ { k } ,$ , simultaneously for every comparator path,

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq \frac { H _ { \tau } } { 6 4 } \left( 1 + \frac { P _ { T } } { V _ { \tau } } \right) + \frac { \gamma G ^ { 2 } T \tau ^ { 2 } } { 2 } .\tag{39}
$$

Consequently, for a universal C,

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq C \left[ \frac { L \overline { { B } } _ { \tau } } { \gamma } + \frac { L ^ { 3 / 2 } \sqrt { \overline { { B } } _ { \tau } } } { \gamma \tau } P _ { T } + \gamma G ^ { 2 } T \tau ^ { 2 } \right] ,\tag{40}
$$

where the combined initial budget is

$$
\overline { { B } } _ { \tau } = B _ { \tau } + \log ( ( T + 1 ) ( K + 1 ) ) .\tag{41}
$$

Proof. Apply lemma B.10 with $V = V _ { \tau }$ and lemma B.9 on each bin. For $Q _ { t } = \mathcal { N } ( u _ { t } , \tau ^ { 2 } I )$

$$
\mathbb { E } _ { Q _ { t } } \widetilde { f } _ { t } - \widetilde { f } _ { t } ( u _ { t } ) = \frac { \gamma } { 2 } \tau ^ { 2 } \lVert g _ { t } \rVert _ { 2 } ^ { 2 } \leq \frac { \gamma } { 2 } G ^ { 2 } \tau ^ { 2 } .
$$

Together with (13), this proves (39). Substitute (36), use $a + B _ { \tau } / \lambda = \overline { { B } } _ { \tau } / \lambda$ , and $\lambda = \gamma / 2$ to obtain (40). □

Lemma B.11 holds simultaneously on the dyadic grid, so we can now compare with the best scale in hindsight. The formal upper bound must also cover paths for which the unconstrained balancing scale lies outside the grid range. The next lemma handles both endpoints and the discretization loss.

Lemma B.12 (Scale endpoints). Define the coeficients of the scale-dependent bound by

$$
\overline { { B } } = C _ { 0 } ( d + 1 ) \Lambda _ { T } , \qquad A = \frac { L \overline { { B } } } { \gamma } , \qquad B = \frac { L ^ { 3 / 2 } \sqrt { \overline { { B } } } } { \gamma } , \qquad C _ { 2 } = \gamma G ^ { 2 } T ,
$$

where $C _ { 0 }$ is large enough that $\overline { { B } } _ { \tau } ~ \leq ~ \overline { { B } }$ for every $\tau \in [ D / T , D ]$ . Put $\tau _ { * } ~ = ~ ( B P _ { T } / C _ { 2 } ) ^ { 1 / 3 }$ and $P _ { 0 } = C _ { 2 } D ^ { 3 } / B$ . Then some dyadic scale satisfies

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq C \left[ A + \operatorname* { m i n } _ { \tau \in [ D / T , D ] } \left( \frac { B P _ { T } } { \tau } + C _ { 2 } \tau ^ { 2 } \right) \right] \wedge G D T .\tag{42}
$$

Moreover, whenever $P _ { T } \leq P _ { 0 }$ 2

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq C \left[ A + ( B ^ { 2 } C _ { 2 } P _ { T } ^ { 2 } ) ^ { 1 / 3 } \right] .\tag{43}
$$

Proof. If $\tau _ { * } \in [ D / T , D ]$ , choose a dyadic scale within a factor two and optimize $B P _ { T } / \tau + C _ { 2 } \tau ^ { 2 }$ If $\tau _ { * } < D / T$ , choose $\tau _ { K } \in [ D / T , 2 D / T )$ . Since $B P _ { T } / C _ { 2 } < D ^ { 3 } / T ^ { 3 }$ , we obtain

$$
\frac { B P _ { T } } { \tau _ { K } } + C _ { 2 } \tau _ { K } ^ { 2 } \leq 5 \gamma G ^ { 2 } \frac { D ^ { 2 } } { T } .
$$

Because $\gamma G D \le 1 / 4 , \gamma G ^ { 2 } D ^ { 2 } / T \le 1 / ( 1 6 \gamma T ) \le A$ after adjusting the universal constant. Thus the static term absorbs the lower endpoint and (43) remains valid.

If $\tau _ { * } > D _ { * }$ , choosing D is within a universal factor of the clipped minimum in (42); independently, Lipschitzness gives the cap GDT. This proves the all-path clipped statement. Finally, $P _ { T } \leq P _ { 0 }$ is exactly the condition $\tau _ { * } \leq D$ , so the preceding two cases establish (43). □

## B.1.6 The projected-gradient bound

The curved learner is most efective for slowly moving comparators. We combine it with a grid of projected-gradient learners to control larger path budgets, using the classical dynamic projectedgradient argument [5, 6].

Lemma B.13 (Dynamic projected gradient descent). Let $z _ { t + 1 } = \Pi _ { \mathcal { W } } ( z _ { t } - \alpha \nabla f _ { t } ( z _ { t } ) )$ ) for a fixed $\alpha > 0$ , where $\Pi _ { \mathcal { W } }$ is Euclidean projection. Then every comparator path satisfies

$$
\sum _ { t = 1 } ^ { T } [ f _ { t } ( z _ { t } ) - f _ { t } ( u _ { t } ) ] \leq \frac { D ^ { 2 } + 2 D P _ { T } ( u ) } { 2 \alpha } + \frac { \alpha G ^ { 2 } T } { 2 } .\tag{44}
$$

Proof. Convexity and projection nonexpansiveness give

$$
2 \alpha [ f _ { t } ( z _ { t } ) - f _ { t } ( u _ { t } ) ] \leq \| z _ { t } - u _ { t } \| ^ { 2 } - \| z _ { t + 1 } - u _ { t } \| ^ { 2 } + \alpha ^ { 2 } G ^ { 2 } .
$$

The change in comparator also satisfies

$$
\begin{array} { r } { \left\| z _ { t + 1 } - u _ { t } \right\| ^ { 2 } - \left\| z _ { t + 1 } - u _ { t + 1 } \right\| ^ { 2 } \geq - 2 D \left\| u _ { t + 1 } - u _ { t } \right\| , } \end{array}
$$

because all three points lie in a set of diameter D. Sum and telescope, using $\| z _ { 1 } - u _ { 1 } \| \leq D$ □

## B.1.7 Numerical model and robustness

We now separate exact arithmetic from an actual bit model.

Assumption B.14 (Polynomial-bit online oracle model). All scalar parameters and the initial point have rational encodings of at most $L _ { 0 }$ bits. On a rational query point, the online loss oracle returns exact rational loss and gradient values with at most $L _ { 0 }$ bits per coordinate. Given rational $A \succ 0$ and a with at most B bits per coordinate, and a precision request $p ,$ the projection oracle returns a feasible rational point m $\in \mathcal W$ whose Mahalanobis objective is within $2 ^ { - p }$ of optimum. Its coordinates have at most poly $( d , L _ { 0 } , B , p )$ bits, and its cost is $\mathsf { C } _ { \mathrm { p r o j } } ^ { \mathrm { b i t } } ( d , L _ { 0 } , B , p )$ bit operations. Rational convex combinations of feasible points can be formed exactly in polynomial bit complexity. For the near-linear-in-T conclusion, $\mathsf { C } _ { \mathrm { p r o j } } ^ { \mathrm { b i t } }$ is polynomial in its displayed arguments.

This model specifies both feasibility and the size of the returned encodings. We analyze the implemented predictions directly, with exact oracle replies at those predictions.

Lemma B.15 (Robustness to inexact history and master updates). Suppose every selected Gaussian update satisfies, uniformly over $u \in \mathcal W$ 2

$$
\lambda r _ { t } \le F _ { t } ( u _ { t } ) - F _ { t + 1 } ( u _ { t } ) + \varepsilon _ { t } ,\tag{45}
$$

and the implemented master satisfies, for every surviving-history prefix $[ i , q ]$

$$
\sum _ { t = i } ^ { q } [ \widetilde f _ { t } ( w _ { t } ) - \widetilde f _ { t } ( x _ { t } ^ { i , k } ) ] \leq a + \lambda ^ { - 1 } \sum _ { t = i } ^ { q } \zeta _ { t } .\tag{46}
$$

Here $\varepsilon _ { t } , \zeta _ { t } \geq 0$ bound the errors uniformly over active histories, including weight changes at birth and pruning. Then all conclusions of theorem B.16 hold with the additive term

$$
{ \frac { 1 } { \lambda } } \sum _ { t = 1 } ^ { T } ( \varepsilon _ { t } + \zeta _ { t } ) .\tag{47}
$$

Proof. Replace $r _ { t }$ by $\bar { r } _ { t } = r _ { t } - \varepsilon _ { t } / \lambda$ . Then the exact one-step inequality holds for $\bar { r } _ { t } ,$ so the shiftedenergy and recursion arguments apply unchanged. Similarly, (46) changes every selected prefix overhead by at most the sum of its $\zeta _ { t } / \lambda$ . The charged and recursively treated pieces are disjoint, so every round’s numerical error is counted at most once. Restoring $r _ { t }$ gives (47). □

## B.2 Full version and proof of Theorem 4.1

The next statement makes the dependence on curvature, diameter, gradient bound, and smoothing scale explicit. It also records the clipped-scale guarantee for every comparator path.

Theorem B.16 (Full version of Theorem 4.1). There are universal constants $C , c > 0$ such that, under Assumptions 3.1 and 3.2, there is a proper full-information algorithm, independent of $P _ { T } ( u )$ such that every comparator path satisfies

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \le C \bigg \{ \frac { ( d + 1 ) \Lambda _ { T } ^ { 2 } } { \gamma } + \operatorname* { m i n } _ { \tau \in [ D / T , D ] } \left[ \frac { \sqrt { d + 1 } \Lambda _ { T } ^ { 2 } } { \gamma \tau } P _ { T } ( u ) + \gamma G ^ { 2 } T \tau ^ { 2 } \right] \bigg \} \wedge G D T .\tag{48}
$$

In particular, if

$$
P _ { T } ( u ) \leq c \frac { \gamma ^ { 2 } G ^ { 2 } T D ^ { 3 } } { \sqrt { d + 1 } \Lambda _ { T } ^ { 2 } } ,\tag{49}
$$

then the bound simplifies to

$$
\mathrm { R e g } _ { T } ( u _ { 1 : T } ) \leq C \left[ \frac { ( d + 1 ) \Lambda _ { T } ^ { 2 } } { \gamma } + \gamma ^ { - 1 / 3 } G ^ { 2 / 3 } ( d + 1 ) ^ { 1 / 3 } \Lambda _ { T } ^ { 4 / 3 } T ^ { 1 / 3 } P _ { T } ( u ) ^ { 2 / 3 } \right] .\tag{50}
$$

The curved algorithm uses one gradient query per round at its actual prediction and has

$$
\mathrm { T i m e } _ { \mathrm { a r i t h } } \leq C T \log ^ { 2 } ( 2 T ) [ d ^ { 3 } + \mathsf { C } _ { \mathrm { p r o j } } ( d ) ] ,\tag{51}
$$

$$
\mathrm { M e m o r y } \leq C d ^ { 2 } \log ^ { 2 } ( 2 T ) .\tag{52}
$$

If a primitive implements the complete rank-one precision update, covariance flooring, and reconstruction in the original coordinates in $O ( d ^ { 2 } )$ operations, the $d ^ { 3 }$ term in (51) can be replaced by $d ^ { 2 }$

Proof. Lemma B.11 and lemma B.12 give

$$
A \leq C \frac { ( d + 1 ) \Lambda _ { T } ^ { 2 } } { \gamma } , \qquad B \leq C \frac { \sqrt { d + 1 } \Lambda _ { T } ^ { 2 } } { \gamma } .
$$

This proves the clipped bound (48). Under (49), $P _ { T } \leq P _ { 0 }$ up to a universal adjustment, and

$$
( B ^ { 2 } C _ { 2 } P _ { T } ^ { 2 } ) ^ { 1 / 3 } \leq C \gamma ^ { - 1 / 3 } G ^ { 2 / 3 } ( d + 1 ) ^ { 1 / 3 } \Lambda _ { T } ^ { 4 / 3 } T ^ { 1 / 3 } P _ { T } ^ { 2 / 3 } ,
$$

which proves (50).

By lemma B.2, there are ${ \cal O } ( \log T )$ active starts and ${ \cal O } ( \log T )$ scales, hence $O ( \log ^ { 2 } T )$ Gaussian states. A standard symmetric eigendecomposition, covariance clipping, and Mahalanobis projection cost $O ( d ^ { 3 } + \mathsf { C } _ { \mathrm { p r o j } } )$ per state in the dense spectral model. The conditional acceleration substitutes an $O ( d ^ { 2 } )$ primitive for this complete spectral update, including reconstruction of the floored covariance in the original coordinates. Interlacing limits the number of precision eigenvalues crossing the cap to one; the primitive must also account for the change of eigenbasis. This proves (51)–(52) under the respective computational models. For fixed $D , G , \eta .$ , the constant $\gamma$ is fixed and $d + 1 \leq 2 d$ Absorbing these regularity-dependent constants gives Theorem 4.1, with the displayed logarithmic factors unchanged. □

## B.3 Full version and proof of Corollary 4.2

We now combine the curved learner with the projected-gradient bound to cover all path budgets.

Run a grid

$$
\alpha _ { j } = 2 ^ { j } \frac { D } { G \sqrt { T } } , \qquad j = 0 , \dots , \left\lceil \log _ { 2 } \sqrt { 2 T } \right\rceil ,
$$

plus the curved algorithm of theorem 4.1. Use an outer exponential-weights master on the original losses $f _ { t } ,$ at learning rate $\eta .$ Every base prediction is in W, so their weighted mean is proper; exp-concavity gives outer meta-regret at most $\eta ^ { - 1 } \log O ( \log T )$ to every base. The curved base is run counterfactually at its own prediction and queries one gradient there. Each gradient-descent base queries a gradient at its own prediction; this is why corollary 4.2 states ${ \cal O } ( \log T )$ gradient evaluations per round.

Corollary B.17 (Full version of Corollary 4.2). Under Assumptions 3.1 and 3.2, a proper fullinformation learner, independent of $P _ { T } ( u )$ , satisfies

$$
\begin{array} { r l r } {  { \mathrm { R e g } _ { T } ( u _ { 1 : T } ) \le C \Big [ \frac { ( d + 1 ) \Lambda _ { T } ^ { 2 } } { \gamma } + \operatorname* { m i n } \Big \{ \gamma ^ { - 1 / 3 } G ^ { 2 / 3 } ( d + 1 ) ^ { 1 / 3 } \Lambda _ { T } ^ { 4 / 3 } T ^ { 1 / 3 } P _ { T } ( u ) ^ { 2 / 3 } , } \\ & { } & { G \sqrt { T ( D ^ { 2 } + D P _ { T } ( u ) ) } , \ G D T \Big \} \Big ] } } \end{array}\tag{53}
$$

for a universal constant $C > 0$ . It uses ${ \cal O } ( \log T )$ loss evaluations, gradient evaluations, and Euclidean projection calls per round in addition to the curved branch’s internal work.

Proof. For the optimal fixed step in (44),

$$
\alpha _ { * } = \sqrt { \frac { D ^ { 2 } + 2 D P _ { T } } { G ^ { 2 } T } } ,
$$

which lies in the range covered by the grid up to a factor two. Therefore one grid expert has regret

$$
U _ { T } = O \bigg ( G \sqrt { T ( D ^ { 2 } + D P _ { T } ) } \bigg ) .
$$

The outer exp-concave master pays only $O ( \eta ^ { - 1 } \log \log T )$ to the better base.

It remains to justify writing the unconstrained curved expression inside the minimum even when the curved smoothing scale would exceed D. With the notation of lemma B.12, let $Q _ { T } = ( B ^ { 2 } C _ { 2 } P _ { T } ^ { 2 } ) ^ { 1 / 3 }$ and $P _ { 0 } = C _ { 2 } D ^ { 3 } / B$ . For $P _ { T } \leq P _ { 0 }$ , theorem B.16 supplies $A + Q _ { T }$ . For $P _ { T } \ge P _ { 0 }$ , direct algebra gives

$$
G \sqrt { T D P _ { T } } \leq Q _ { T } , \qquad G D \sqrt { T } \leq \sqrt { A Q _ { T } } \leq \frac { A + Q _ { T } } { 2 } ,
$$

because $\gamma B \geq 1$ after the universal constants are fixed. Hence $U _ { T } \leq C ( A + Q _ { T } )$ also outside the curved-scale range. Thus the better of the curved and OGD bases is bounded by the first two branches in (53). The GDT cap follows from Lipschitzness and the diameter bound. Substituting the definitions of $A , B , C _ { 2 }$ gives the displayed logarithmic factors; the outer aggregation cost is absorbed by $( d + 1 ) \Lambda _ { T } ^ { 2 } / \gamma$

With $D , G , \eta$ fixed, absorbing regularity-dependent constants gives Corollary 4.2. For the fixedparameter form (2), the replacement of $\sqrt { T ( 1 + P _ { T } ) }$ by $\sqrt { T P _ { T } }$ is valid inside the displayed minimum, with the additive d retained. If $P _ { T } \ge 1$ , the two square-root expressions difer by at most $\sqrt { 2 }$ . If $P _ { T } < 1$ and $T \geq d ^ { 2 }$ , the curved term is at most $\sqrt { T P _ { T } }$ . If $P _ { T } < 1$ and $T < d ^ { 2 }$ , the extra $\sqrt { T }$ is at most d. This also justifies the abstract’s upper bound for all horizons. □

## B.4 Finite precision and Turing complexity

Under the rational-oracle model in Assumption B.14, the local error control of Lemma B.15 yields the following implementation guarantee.

Theorem B.18 (Polynomial-bit implementation). Under Assumptions B.14 and 3.1, for every $\delta \in ( 0 , 1 )$ there is a Turing implementation of the curved algorithm whose regret is the right-hand side of (48) plus δ. In addition to one loss/gradient oracle call per round, its total bit complexity is

$$
\tilde { O } \Big ( T \log ^ { 2 } ( 2 T ) \left[ \mathrm { p o l y } \left( d , L _ { 0 } , \log T , \log \frac { 1 } { \delta } , \log ( 1 + 1 / \gamma ) , \log ( 1 + D + G ) \right) + C _ { \mathrm { p r o j } } ^ { \mathrm { b i t } } ( d , L _ { 0 } , B , p ) \right] \Big ) .\tag{54}
$$

where the oracle precision parameters satisfy

$$
B , p = \mathrm { p o l y } ( d , L _ { 0 } , \log ( T / \delta ) , \log ( 1 + 1 / \gamma ) , \log ( 1 + D + G ) ) .
$$

The horizon dependence is near-linear under the polynomial-bit oracle model above.

Proof. We control local update errors at the implemented state, so their contribution is additive through Lemma B.15. Write $N = O ( \log ^ { 2 } ( 2 T ) )$ for a uniform bound on the number of active states and set the target local errors to $\varepsilon _ { t } , \zeta _ { t } \leq \lambda \delta / ( 2 T )$ .

Uniform conditioning. Work in coordinates translated by the rational initial point w<sub>0</sub>. Every feasible mean then has norm at most D. The stored covariances satisfy $\tau ^ { 2 } I \preceq \Sigma \preceq D ^ { 2 } I$ , with $\tau \geq D / T$ . Before flooring, the precision matrix is bounded between $D ^ { - 2 } I$ and $( \tau ^ { - 2 } + \lambda \gamma G ^ { 2 } ) I$ The tilt mean is obtained by solving this positive-definite system with a bounded right-hand side. Consequently, all relevant magnitudes, inverse eigenvalues, and derivatives of the Gaussian KL formula have bounds of the form $2 ^ { \mathrm { p o l y } ( L _ { 0 } ) } \mathrm { p o l y } ( d , T , 1 + 1 / \gamma , 1 + D + G )$ . This includes inverse powers of $D ,$ since its positive rational encoding has at most $L _ { 0 }$ bits. Thus polynomially many guard bits in the stated parameters sufice for any prescribed local KL error.

A rational covariance floor. The scale $\tau = D$ has the exact stored covariance $D ^ { 2 } I$ . For the other dyadic scales, $\tau \leq D / 2$ . Compute a rational approximation to the spectral clipping of the tilted covariance into $[ \tau ^ { 2 } , D ^ { 2 } ]$ , first moving the two clipping thresholds inward by a guard margin h and then approximating the matrix to operator-norm error at most $h / 2$ . The stored rational covariance then remains in the certified interval $[ \tau ^ { 2 } , D ^ { 2 } ]$ and difers from the exact floor by $O ( h )$

This operation has polynomial bit complexity without an eigenvalue separation assumption. One explicit construction replaces the scalar positive part $x _ { + }$ by $( x + \sqrt { x ^ { 2 } + \epsilon ^ { 2 } } ) / 2$ , with uniform error at most $\epsilon / 2$ , and applies this formula successively at the lower and upper thresholds. For a symmetric matrix $M$ , the required positive-definite square root is that of $M ^ { 2 } + \epsilon ^ { 2 } I$ . Starting from a scalar upper bound times the identity, Newton’s iteration $X  ( X + X ^ { - 1 } ( M ^ { 2 } + \epsilon ^ { 2 } I ) ) / 2$ reduces to the scalar square-root iteration in the eigenbasis of M. Its iteration count is polynomial in the logarithmic spectral bounds and the requested precision. Rational rounding with guard bits gives the required operator-norm error; entrywise error e contributes at most de in operator norm. Choosing the smoothing, iteration, and rounding errors below fixed fractions of $h$ proves the claimed polynomial construction. The uniform KL bounds above turn its $O ( h )$ covariance error into the desired local KL error.

Feasible mean updates. Round the tilt mean and the positive-definite projection metric to the working precision before calling the projection oracle. Their input encodings therefore have at most B bits per coordinate, independent of earlier oracle-output lengths. Perturbing these inputs changes the quadratic objective uniformly on W by a quantity controlled by the same magnitude bounds. Hence a suficiently accurate oracle output has objective gap at most $\xi$ for the exact floored metric $A _ { * } = ( C ^ { \flat } ) ^ { - 1 }$ . If $m ^ { * }$ is its exact projection and $\widehat { m }$ is the feasible returned point, the projection variational inequality gives

$$
\| \widehat { m } - m ^ { * } \| _ { A _ { * } } ^ { 2 } \leq \xi , \qquad \| u - \widehat { m } \| _ { A _ { * } } ^ { 2 } - \| u - m ^ { * } \| _ { A _ { * } } ^ { 2 } \leq 2 ( D / \tau ) \sqrt { \xi } + \xi .
$$

The covariance and mean errors therefore yield (45) uniformly over $u \in \mathcal W$ . The oracle’s outputlength bound makes each stored feasible mean polynomial in the working precision. At the next update, its derived projection inputs are rounded again to B bits, so encoding lengths do not grow with the number of rounds.

Master weights. Absolute rounding alone can erase a small expert weight. We instead round each normalized probability upward. For an exact normalized vector v with at most N entries, compute rational upper approximations $b _ { j }$ such that

$$
v _ { j } + h \le b _ { j } \le v _ { j } + 3 h , \qquad q _ { j } = \frac { b _ { j } } { \sum _ { \ell } b _ { \ell } } .
$$

Certified absolute approximations to the exponentials sufice to construct these bounds. They imply

$$
q _ { j } \ge \frac { v _ { j } } { 1 + 3 N h } , \qquad q _ { j } \ge \frac { h } { 1 + 3 N h } .
$$

Apply this operation after the exponential update and after birth, deletion, and normalization. Each operation decreases an individual weight by at most a factor $1 + 3 N h$ relative to its ideal value at that implemented state. In the telescoping proof of Lemma B.3, this adds at most CNh per round to the logarithmic overhead, including the birth weight. It proves (46) with $\zeta _ { t } \le C N h$ . All stored probabilities have polynomial encoding length and a known positive lower bound. Since $| \lambda \widetilde { f _ { t } } ( x ) |$ is uniformly bounded on W, the required normalized exponential probabilities can be evaluated to this absolute accuracy in polynomial bit complexity.

The prediction is the exact rational convex combination of feasible means and the stored rational weights. The loss oracle is queried at this actual prediction, so surrogate domination and the Gaussian mix-gap argument apply directly to the implemented sequence. Choose h and the projection objective accuracy so that $\varepsilon _ { t } , \zeta _ { t } \leq \lambda \delta / ( 2 T )$ . The preceding bounds require only the displayed polynomial values of $B , p .$ Lemma B.15 then bounds the additional regret by δ. There are $O ( \log ^ { 2 } ( 2 T ) )$ states per round, giving (54). □

Remark B.19 (Anytime implementation). A doubling schedule removes prior knowledge of T. For the curved learner, fix a comparison scale $\tau \in [ D / T , D ]$ and in epoch j compare with a dyadic scale within a factor two of max $\{ \tau , D / H _ { j } \}$ , where $H _ { j }$ is that epoch’s planned doubling horizon. Its movement term is bounded by a constant times $P _ { j } / \tau$ . The additional smoothing cost caused by the lower endpoint is at most $C \gamma G ^ { 2 } D ^ { 2 } / H _ { j }$ , and the geometric sum of these costs is absorbed by the static term. Summing the fixed-scale bounds and then optimizing this common τ recovers the clipped-scale guarantee with additional logarithmic factors.

Restarting the full-frontier ensemble likewise gives the full frontier with logarithmic overhead. If epoch j has planned horizon $H _ { j }$ and internal path length $P _ { j }$ , then Hölder’s inequality gives

$$
\sum _ { j } H _ { j } ^ { 1 / 3 } P _ { j } ^ { 2 / 3 } \leq \left( \sum _ { j } H _ { j } \right) ^ { 1 / 3 } \left( \sum _ { j } P _ { j } \right) ^ { 2 / 3 } .
$$

The planned horizons satisfy $\begin{array} { r } { \sum _ { j } H _ { j } \ \leq \ 4 T } \end{array}$ , including the last partially completed epoch. The square-root branch therefore sums to at most $2 \sqrt { T ( J + \textstyle \sum _ { j } P _ { j } ) }$ over $J = { \cal O } ( \log ( 2 T ) )$ epochs, by Cauchy–Schwarz, and the actual linear caps sum to $T$ . For a total numerical regret budget $\delta ,$ allocate $\delta _ { j } = \delta / 2 ^ { j + 1 }$ to epoch $j ,$ then $\textstyle \sum _ { j } \delta _ { j } \leq \delta$ and its precision still grows only polynomially in log $H _ { j }$ and $\log ( 1 / \delta )$

## C Proofs for Section 5

The lower construction allocates uncertainty across time blocks and coordinates. Each coordinate carries a hidden sign whose noisy labels reveal only a constant amount of information within one block. The comparator follows the blockwise conditional mean. The norm constraint limits the total signal across coordinates, while the path budget limits how often that signal can change. Choosing the number of active coordinates recovers the curved, square-root, and saturation regimes in Theorem 5.1.

## C.1 Useful lemmas

The first lemma bounds sequential estimation of one hidden sign. The next two convert that risk into movement-dependent and stationary lower bounds.

Lemma C.1 (Sequential two-point risk). Let S be uniform on $\{ - 1 , + 1 \}$ and, conditional on $S$ let $Y _ { 1 } , \dots , Y _ { n } \in \{ - 1 , + 1 \}$ be independent with $\mathbb { P } ( Y _ { r } = 1 \mid S ) = ( 1 + S \theta ) / 2$ , where $0 < \theta \leq 1 / 2$ . If $n \theta ^ { 2 } \leq 1 / 3 2$ , then every sequential prediction $X _ { r } \in [ - 1 , 1 ]$ satisfies

$$
\sum _ { r = 1 } ^ { n } \mathbb { E } ( X _ { r } - S \theta ) ^ { 2 } \geq c n \theta ^ { 2 }\tag{55}
$$

for a universal $c > 0$ , where $X _ { r }$ is measurable with respect to $Y _ { 1 } , \dots , Y _ { r - 1 }$ and auxiliary randomness independent of $( S , Y _ { 1 : n } )$ . Independent observations from other hidden coordinates may be included in this auxiliary randomness.

Proof. The one-sample KL divergence between the two sign models is

$$
\theta \log { \frac { 1 + \theta } { 1 - \theta } } \leq 4 \theta ^ { 2 } .
$$

Before round $r ,$ Pinsker’s inequality bounds the total variation between the two histories by $\sqrt { 2 ( r - 1 ) } \theta \leq 1 / 4$ . Thus every sign test errs with probability at least $3 / 8$ . The test sign(X<sub>r</sub>) has $| X _ { r } - S \theta | \ge \theta$ on a sign error. Sum over r. □

The two-point lemma gives a constant amount of prediction error while one hidden sign remains uncertain. We replicate that experiment over coordinates and time blocks. The efective dimension m controls how much independent uncertainty can be packed into the fixed Euclidean ball without exceeding the path budget.

Lemma C.2 (Block-hypercube lower bound in efective dimension m). There are universal $c , C > 0$ such that the following holds. Let $1 \leq m \leq d$ and suppose

$$
C { \frac { m } { \sqrt { T } } } \leq P \leq c { \frac { T } { m ^ { 2 } } } .\tag{56}
$$

Then the loss class (7) admits a distribution and a comparator path of length at most P for which every learner sufers

$$
\mathbb { E } \mathrm { { R e g } } _ { T } \geq c m ^ { 1 / 3 } T ^ { 1 / 3 } P ^ { 2 / 3 } .\tag{57}
$$

Proof. Use only the first m coordinates. Choose

$$
K = \left\lceil C _ { 1 } \left( { \frac { T P ^ { 2 } } { m ^ { 2 } } } \right) ^ { 1 / 3 } \right\rceil , \qquad n = \left\lfloor { \frac { T } { K m } } \right\rfloor , \qquad a = { \frac { P } { 4 K } } , \qquad \theta = { \frac { a } { \sqrt { m } } } .
$$

For suitable constants in (56), $2 \leq K \leq T / ( 2 m )$ and $a \leq 1 / 2$ . Partition Knm rounds into K blocks. In block $k ,$ draw independent signs $S _ { k , 1 } , \ldots , S _ { k , m }$ and set

$$
\boldsymbol { u } ^ { ( k ) } = \frac { a } { \sqrt { m } } ( S _ { k , 1 } , \ldots , S _ { k , m } , 0 , \ldots , 0 ) .
$$

This lies in $\mathcal { W } _ { d } .$ , and the total path is at most $2 a ( K - 1 ) \leq P$ . Within each block, cycle through the m active coordinates n times. When coordinate $j$ is queried, draw $Y \in \{ - 1 , + 1 \}$ with mean $\theta S _ { k , j }$ and reveal $f _ { j , Y }$ . Keep the last block comparator fixed on any leftover rounds and continue the same conditional-mean square-loss experiment. Those leftover rounds have nonnegative conditional expected excess, so they may be discarded in the lower-bound calculation.

Conditional on the signs and the past observations $\mathcal { H } _ { t - 1 }$ , every prediction $w _ { j }$ satisfies

$$
\mathbb { E } \left[ f _ { j , Y } ( w ) - f _ { j , Y } ( u ^ { ( k ) } ) \mid S _ { k , j } , \mathcal { H } _ { t - 1 } \right] = \frac { 1 } { 2 } ( w _ { j } - \theta S _ { k , j } ) ^ { 2 } .\tag{58}
$$

The information available for each hidden sign is controlled by

$$
n \theta ^ { 2 } \leq \frac { T P ^ { 2 } } { 1 6 K ^ { 3 } m ^ { 2 } } ,
$$

which is at most $1 / 3 2$ for a suficiently large $C _ { 1 }$ . Apply lemma C.1 independently to every blockcoordinate pair. Since $K n m \ge T / 2$

$$
\mathbb { E } \mathrm { R e g } _ { T } \geq c T \theta ^ { 2 } = c \frac { T P ^ { 2 } } { m K ^ { 2 } } \geq c m ^ { 1 / 3 } T ^ { 1 / 3 } P ^ { 2 / 3 } .
$$

The calculation holds after conditioning on any learner randomness. Averaging over the generated loss sequences and comparator paths therefore implies that, for every randomized learner, at least one admissible deterministic pair attains the same expected lower bound. □

The block construction captures the movement-dependent part of regret, but its stated regime excludes very small path budgets. A stationary comparator supplies the missing initialization cost. The next lemma provides this additive dimension term without using any movement.

Lemma C.3 (Static lower bound). $I f T \geq c _ { 0 } d ^ { 2 } .$ , then the same class has static minimax regret $\Omega ( d )$

Proof. Draw independent fixed signs $S _ { 1 } , \ldots , S _ { d }$ and set the fixed comparator $u _ { j } ~ = ~ S _ { j } / ( 8 \sqrt { d } )$ Query each coordinate for $n = d$ initial rounds with labels of mean $S _ { j } / ( 8 \sqrt { d } )$ . On all remaining rounds continue to use losses of the form (7) with the same conditional means. Their conditional expected excess over the fixed comparator is nonnegative, so it is enough to lower-bound the initial observations. The product $n \theta ^ { 2 } = 1 / 6 4$ satisfies Lemma C.1, and lemma C.1 and equation (58) yield constant regret per coordinate and hence $\Omega ( d )$ total regret. Averaging over the signs and observations yields an admissible deterministic loss sequence and fixed comparator with at least this regret in expectation over learner randomness. □

## C.2 Proof of Theorem 5.1

The following restatement collects the normalized lower frontier. Its proof verifies the loss class and then combines the preceding lemmas across all path budgets.

Theorem C.4 (Restatement of Theorem 5.1). There are universal constants $c , c _ { 0 } > 0$ such that the following holds. Let

$$
\mathcal { W } _ { d } = \{ w \in \mathbb { R } ^ { d } : \| w \| _ { 2 } \leq 1 / 2 \}
$$

and let the adversary use only the losses

$$
f _ { j , y } ( w ) = \frac { 1 } { 2 } ( w _ { j } - y ) ^ { 2 } , \qquad j \in [ d ] , \quad y \in \{ - 1 , + 1 \} .\tag{59}
$$

This class has diameter $D = 1$ , gradient bound $G \leq 3 / 2$ , and exp-concavity coeficient at least $4 / 9$ uniformly in d. $I f T \geq c _ { 0 } d ^ { 2 }$ , then for every $P \geq 0$

$$
\begin{array} { r l } & { \underset { \textnormal { \tiny A } _ { f _ { 1 : T } } ~ \mathcal { O } f ~ t h e ~ f o r m ~ ( 7 ) } { \operatorname* { i n f } } \mathbb { E } _ { \boldsymbol { \mathrm { A } } } \mathrm { R e g } _ { T } ( u _ { 1 : T } ) \geq c \left[ d + \operatorname* { m i n } \left\{ d ^ { 1 / 3 } T ^ { 1 / 3 } P ^ { 2 / 3 } , \sqrt { T P } , T \right\} \right] . } \\ & { \quad \underset { u _ { 1 : T } \in { \mathcal { W } _ { d } ^ { T } } , ~ P _ { T } ( u ) \leq P } { \operatorname* { s u p } } } \end{array}\tag{60}
$$

Proof. We first verify the stated regularity. For (7),

$$
\nabla f _ { j , y } ( w ) = ( w _ { j } - y ) e _ { j } , \qquad \nabla ^ { 2 } f _ { j , y } ( w ) = e _ { j } e _ { j } ^ { \top } .
$$

On $\mathcal { W } _ { d } , | w _ { j } - y | \leq 3 / 2$ , so

$$
\nabla ^ { 2 } f _ { j , y } ( w ) \succeq \frac { 4 } { 9 } \nabla f _ { j , y } ( w ) \nabla f _ { j , y } ( w ) ^ { \top } ,
$$

and the claimed uniform regularity follows. Let $C _ { * } , c _ { * }$ be the constants in Lemma C.2, decreasing $c _ { * }$ so that $c _ { * } \leq 1$ . Fix $\kappa = \sqrt { c _ { * } } / 2$ and $b = \kappa ^ { 2 } / 4$ . We may enlarge the universal constant in $T \geq c _ { 0 } d ^ { 2 }$ so that

$$
C _ { * } d / \sqrt { T } \le \operatorname * { m i n } \{ c _ { * } T / d ^ { 2 } , b T \} , \qquad c _ { * } T / d ^ { 2 } \ge ( C _ { * } \kappa ) ^ { 2 / 3 } .
$$

If $P < C _ { * } d / \sqrt { T }$ , the curved expression is at most $C _ { * } ^ { 2 / 3 } d ,$ so Lemma C.3 sufices. For $C _ { * } d / \sqrt { T } \leq$ $P \leq c _ { * } T / d ^ { 2 }$ , use Lemma C.2 with $m = d .$

Suppose next that $c _ { * } T / d ^ { 2 } < P \leq b T$ and choose

$$
m = \left\lfloor \kappa \sqrt { T / P } \right\rfloor .
$$

The upper bound on $P$ ensures that the quantity inside the floor is at least two, so ${ \scriptstyle { \frac { \kappa } { 2 } } } { \sqrt { T / P } } \leq$ $m \leq \kappa \sqrt { T / P } < d .$ . In particular, $P \leq c _ { * } T / m ^ { 2 }$ . The second displayed condition gives $C _ { * } m / \sqrt { T } \leq$ ${ C _ { * } \kappa } / \sqrt { P } \leq P$ . Thus both conditions of Lemma C.2 hold and its lower bound is $\Omega ( { \sqrt { T P } } )$

At $P = b T$ , one of the preceding cases gives $\Omega ( T )$ : the all-coordinate bound is at least a constant times T if its regime reaches $b T$ , and otherwise the efective-dimension bound gives $\sqrt { b } T$ . Monotonicity of the admissible comparator class extends this lower bound to every $P \geq b T$ These cases cover every budget, including the constant-factor transition ranges. Finally, the static and moving constructions belong to the same loss class; taking their maximum and using max $\{ a , b \} \ge ( a + b ) / 2$ yields the additive d term. □