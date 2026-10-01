# Convergence of Practical Muon with Finite Newton–Schulz Iterations and Nesterov Momentum

Hanyang Peng, Hui Wang, Yue Yu Pengcheng Laboratory, Shenzhen, China {penghy, wangh06, yuy}@pcl.ac.cn

## Abstract

Practical Muon maintains momentum and performs a small, fixed number of Newton–Schulz iterations separately for each parameter matrix, often with a Nesterov correction. We analyze these layer-wise finite-step updates jointly on a coupled nonconvex objective, rather than replacing them by exact polar factors or one global orthogonalization. Under gradient-dependent $( \mathcal { L } _ { 0 } , \mathcal { L } _ { 1 } , q )$ )-smoothness and conditionally unbiased stochastic gradients with bounded layer-wise variance, we establish an O(T<sup>−1/4</sup>) bound on the expected average Frobenius gradient norm. The analysis retains the Nesterov recursion and requires neither bounded stochastic gradients, symmetric noise, nor a uniform positive lower bound on the nonzero output singular values. Its constants contain no explicit matrix-dimension or rank factors when the number of blocks and problem constants are fixed. The proof follows a descent inequality and a decomposition of the momentum tracking error into initialization, noise, and drift. For the original five-step quintic, we verify the required scalar-map bounds analytically; the result also allows step-dependent coeficients satisfying the same bounds. A complementary nuclear-norm result quantifies rank dependence under a stronger spectral condition. The vanishing rate uses coupled learning-rate and momentum schedules, including the standard single-coeficient Nesterov rule.

## 1 Introduction

Muon orthogonalizes a momentum-based update separately for each matrix-valued parameter. The reference implementation maintains a momentum bufer for each parameter matrix, applies Frobenius normalization and a small number of quintic Newton–Schulz iterations to that matrix, and includes an optional Nesterov correction (Jordan, 2024). These three features—layer-wise updates, finite orthogonalization, and Nesterov momentum—all matter for convergence analysis. A single global normalization is not the layer-wise operation, a finite polynomial need not return an exact polar factor, and the Nesterov correction changes the update’s noise and gradient-tracking error.

Our analysis preserves this layer-wise structure. Each matrix block has its own momentum, normalization, and orthogonalization, and the blocks may have diferent dimensions and noise levels. Their updates nevertheless act on one coupled objective: all stochastic gradient blocks are evaluated at the same iterate, and changing one block may change the gradients in other blocks. We therefore use joint smoothness to control simultaneous updates, while retaining layer-specific variance bounds and tracking errors. The analysis does not assume a separable loss or independent noise across layers. Here a “layer” indexes a parameter matrix; several such matrices may belong to the same architectural layer.

Existing analyses address diferent aspects of Muon. Pethick et al. (2025) place non-Nesterov Muon in an exact-LMO framework and develop layer-wise norm choices. Riabinin et al. (2025) explicitly analyze layer-wise LMOs under block generalized smoothness. Shen et al. (2025) study exact-polar Muon and its matrix geometry, whereas Kim and Oh (2025) analyze finite Newton–Schulz iterations through polar-approximation bounds. Choudhury et al. (2026) already combine Nesterov momentum with inexact polar decomposition and allow gradient-dependent heavy-tailed noise. Li and Tsuchiya (2026) exploit finite-iteration smoothing through online-to-nonconvex conversion. Table 1 compares the algorithms, assumptions, and guarantees in these results. Our contribution is not any one of these ingredients in isolation, but a direct analysis of their combination: layer-wise Frobenius-normalized finite polynomial updates with Nesterov momentum, under joint gradientdependent smoothness and bounded layer-wise conditional variance, yielding an average Frobenius stationarity bound without explicit matrix-size or rank factors. The assumptions are not uniformly weaker than those of all prior work; in particular, our bounded-variance model does not cover the heavier-tailed regimes of Choudhury et al. (2026).

Table 1: Comparison of representative convergence analyses relevant to practical Muon. NS denotes Newton–Schulz iteration; LMO denotes a linear minimization oracle. Entries describe the cited theoretical results, not every implementation studied in each paper.
<table><tr><td>Work</td><td>Block structure</td><td>in theory</td><td></td><td>Orthogonalization Nesterov Smoothness and noise Stationarity and</td><td>dimension factors</td></tr><tr><td>Pethick et al. (2025) (Scion/uSCG)</td><td>General- norm theorem; layer-wise design</td><td>Exact LMO; polar factor for Muon</td><td>Noª</td><td>Norm smoothness; bounded Euclidean variance</td><td>Dual-norm gradient; geometry-dependent constants</td></tr><tr><td>Riabinin et al. (2025) (Gluon)</td><td>Explicitly layer-wise</td><td>Exact block LMOs No</td><td></td><td>Block generalized smoothness; block dual-norm variance</td><td>Weighted block dual norms; norm-specific constants</td></tr><tr><td>Shen et al. (2025)</td><td>Single matrix</td><td>Exact SVD-polar factor</td><td>No</td><td>Frobenius or spectral smoothness; bounded Frobenius variance</td><td>Frobenius/nuclear gradient; rank factors in stochastic bounds</td></tr><tr><td>Kim and Oh (2025)</td><td>Single matrix</td><td>Finite Taylor NS; polar-error analysis</td><td>No</td><td>Spectral smoothness; bounded Frobenius variance</td><td>Average nuclear gradient; explicit rank factors</td></tr><tr><td>Choudhury et al. (2026)</td><td>Single matrix</td><td>Inexact polar, including NS; relative alignment bound</td><td>Yes</td><td>Frobenius L-smoothness; Best-iterate Frobenius gradient-dependent α-moment noise,  $1 < \alpha \leq 2$ </td><td>gradient; do-dependent constantsb</td></tr><tr><td>Li and Tsuchiya Single- (2026)</td><td>matrix online learner</td><td>Finite Taylor NS; fixed normalization</td><td>No</td><td>Nonsmooth objectives allowed; moment bounds and almost-sure operator-norm gradient bound</td><td>Online-to-nonconvex stationarity; rank-dependent bounds</td></tr><tr><td>This work Theorem 3.4</td><td>Explicitly layer-wise</td><td>Fixed finite NS; includes the practical five-step quintic</td><td>Yes</td><td>Joint  $( \mathcal { L } _ { 0 } , \mathcal { L } _ { 1 } , q )$  -smoothness; layer-wise conditional Frobenius variance</td><td>Average Frobenius gradient,  $\mathcal { O } ( T ^ { - 1 / 4 } ) ;$  no size/rank factors°</td></tr></table>

<sup>a</sup>The Muon specialization in the uSCG theory uses non-Nesterov momentum; Scion also develops layer-wise norm choices. $^ { \mathrm { b } } d _ { 0 } = \operatorname* { m i n } \{ m , n \}$ <sup>c</sup>With fixed block count, NS depth and coeficients, and problem constants; the vanishing rate uses coupled momentum and learning-rate schedules. “Single matrix” describes the theorem’s formulation, not an impossibility of extending it. The norms, noise models, and output criteria difer, so the rows are not ordered by a common strength of assumptions. For our implementation scope, see Remark 4.5.

The main obstacle is not only the approximation error relative to a polar factor. For a fixed number J of iterations, let $\phi _ { J }$ be the composite scalar polynomial acting on normalized singular values. Since $\phi _ { J } ( s ) \to 0$ as $s \downarrow 0$ , a uniform positive lower bound on all nonzero output singular values is unavailable over arbitrary spectra. However, the ratio $\phi _ { J } ( s ) / s$ can remain bounded above and away from zero. Together with Frobenius normalization of each block, this yields descent and update-energy bounds that do not introduce a rank factor. We use these bounds without replacing the finite map by an exact polar direction.

The resulting theorem controls the expected average Frobenius gradient norm along the original stochastic iterates. Its proof retains the same three parts of each block’s Nesterov tracking error: the initial momentum bias, the accumulated stochastic noise, and the drift of the true gradient under joint updates. Layer-specific noise levels remain visible through $\begin{array} { r } { S _ { \mathsf { F } } = \sum _ { l = 1 } ^ { L } \sigma _ { ( l ) } } \end{array}$ , rather than through a matrix-rank conversion. The $\mathcal { O } ( T ^ { - 1 / 4 } )$ rate follows by coupling the learning rate and momentum parameters. Taking the same coeficient in the momentum and Nesterov correction is included, not removed for the analysis. For the original quintic, the scalar-map requirements hold for every fixed $J \geq 1$ , in particular for $J = 5 ;$ they can also be checked for schedules using diferent coeficients at diferent steps.

Section 3 develops these results within one proof framework. It first gives a rank-sensitive nuclear-norm bound under a positive output singular-value floor, then removes that condition by using the finite map and measuring stationarity in Frobenius norm. The latter result uses neither a bounded-gradient nor a noise-symmetry assumption. Dimension independence here concerns the optimization bound with fixed block count and problem constants; it is not a claim about how those constants change across neural-network architectures.

## 2 Preliminary

We minimize $F ( X ) = \mathbb { E } _ { \xi } [ f ( X ; \xi ) ]$ over matrix blocks $\boldsymbol { X } = ( X ^ { ( 1 ) } , \ldots , X ^ { ( L ) } )$ . Algorithm 1 uses ${ \mathbf { } } M _ { t } ^ { ( l ) }$ for the first-order momentum and $N _ { t } ^ { ( l ) }$ for the direction passed to the orthogonalization routine. The setting $\beta _ { 2 } = 1$ gives ordinary momentum, whereas $\beta _ { 2 } = \beta _ { 1 } = \beta$ gives

$$
N _ { t } ^ { ( l ) } = \beta M _ { t } ^ { ( l ) } + ( 1 - \beta ) G _ { t } ^ { ( l ) } ,\tag{1}
$$

the single-coeficient Nesterov correction used in the reference implementation.<sup>1</sup> The routine may be inexact; its finite Newton–Schulz specialization is specified in Section 3.1.

Layer-wise refers to the separate matrix operations in Algorithm 1: the normalization in each call to orthogonalization uses that block’s $N _ { t } ^ { ( l ) }$ , not the norm of the full momentum tuple. All $G _ { t } ^ { ( l ) }$ are evaluated at $X _ { t }$ before the joint iterate $X _ { t + 1 }$ is formed. This is not cyclic block-coordinate descent, and no cross-block derivatives are set to zero. The common sample $\xi _ { t }$ may also correlate the noise across blocks.

Here “practical” refers to retaining the layer-wise matrix updates, finite Newton–Schulz map, and Nesterov correction, rather than analyzing an exact-polar or globally orthogonalized surrogate. The mathematical iteration is studied in exact arithmetic, without decoupled weight decay or a separate optimizer for other parameter groups. Remark 4.5 treats normalization stabilizers and fixed output scaling separately.

Proposition 2.1 (Spectral Structure of Inexact Orthogonalization). Let the compact SVD of the extrapolated momentum be $N _ { t } = U \Sigma V ^ { \top } \in \mathbb { R } ^ { m \times n } ~ w i t h ~ \Sigma = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { r } )$ , where $r = { \mathrm { r a n k } } ( N _ { t } )$ . A singular-vector-preserving orthogonalization routine gives $O _ { t } = U \varsigma V ^ { \top }$ , where $\boldsymbol { \varsigma } = \mathrm { d i a g } ( \varsigma _ { 1 } , \ldots , \varsigma _ { r } )$ For a finite polynomial routine with input normalization $\nu _ { t } > 0$ , the modified values are $\varsigma _ { i } = \phi _ { J } ( \sigma _ { i } / \nu _ { t } )$ where J denotes the number of polynomial steps and is distinct from the smoothness exponent $q .$ Suppose these values are nonnegative. Then:

Algorithm 1 Practical layer-wise Muon with Nesterov momentum and inexact orthogonalization   
Require: learning rate $\gamma > 0 ,$ , momentum $\overline { { \beta _ { 1 } \in [ 0 , 1 ) , \beta _ { 2 } \in [ 0 , 1 ] } }$ , total iterations $T _ { i }$ , number of   
layers $L$   
1: Initialize: $M _ { 0 } ^ { ( l ) } \gets 0 , X _ { 1 } ^ { ( l ) } \in \mathcal { S } _ { l }$ for each layer $l \in \{ 1 , \ldots , L \}$   
2: for $t = 1$ to $T$ do   
3: for each layer $l = 1$ to $L$ do   
4: $G _ { t } ^ { ( l ) } \gets \nabla _ { ( l ) } f ( X _ { t } ; \xi _ { t } )$ ▷ Compute layer-wise batch gradients   
5: $M _ { t } ^ { ( l ) } \gets \beta _ { 1 } M _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 1 } ) G _ { t } ^ { ( l ) }$ ▷ Update first-order momentum   
6: $N _ { t } ^ { ( l ) } \gets \beta _ { 2 } M _ { t } ^ { ( l ) } + ( 1 - \beta _ { 2 } ) G _ { t } ^ { ( l ) }$ ▷ Apply Nesterov correction   
7: $O _ { t } ^ { ( l ) }$ ← Inexact-Orthogonalization $( N _ { t } ^ { ( l ) } )$ ▷ Generate approximate polar proxy   
8: $\dot { X _ { t + 1 } ^ { ( l ) } }  X _ { t } ^ { ( l ) } - \gamma O _ { t } ^ { ( l ) }$ ▷ Update parameters   
9: end for   
10: end for

1. Directional Alignment: $\langle N _ { t } , O _ { t } \rangle \geq \varsigma _ { \mathrm { m i n } } \| N _ { t } \| ,$ , where $\varsigma _ { \mathrm { m i n } } = \operatorname* { m i n } _ { i } \varsigma _ { i }$

2. Frobenius Norm Bound: $\| O _ { t } \| _ { \mathsf { F } } \leq \varsigma _ { \operatorname* { m a x } } \sqrt { r }$ , where $\operatorname { S m a x } = \operatorname* { m a x } _ { i } \operatorname { S } i$

$I f N _ { t } = 0$ , take $O _ { t } = 0$ and interpret both inequalities directly, without a minimum over an empty spectrum. Uniform positive lower bounds on the nonzero $\varsigma _ { i }$ are an additional condition in Theorem ${ \ 3 . 1 ; }$ they do not follow merely from using finitely many Newton–Schulz steps.

Assumption 2.2 (Objective Lower Boundedness). The objective function F is lower bounded on $s ;$ that is, there exists a constant ${ F } ^ { * } > - \infty$ such that

$$
F ( X ) \geq F ^ { * } , \qquad \forall X \in { \mathcal { S } } .\tag{2}
$$

Assumption 2.3 (Global (L<sub>0</sub>, L<sub>1</sub>, q)-Smoothness). Let ${ \mathcal { S } } = \mathbb { R } ^ { n _ { 1 } \times m _ { 1 } } \times \cdot \cdot \cdot \times \mathbb { R } ^ { n _ { L } \times m _ { L } }$ , and equip the complete parameter tuple with $\begin{array} { r } { \| \dot { \boldsymbol { X } } \| _ { \mathsf { F } } ^ { 2 } = \sum _ { l = 1 } ^ { L } \widetilde { \| \boldsymbol { X } ^ { ( l ) } \| _ { \mathsf { F } } ^ { 2 } } } \end{array}$ . The diferentiable objective $F : S $ R satisfies, for some $\mathcal { L } _ { 0 } > 0 , \mathcal { L } _ { 1 } \ge 0$ , and $q \in [ 0 , 1 ]$ , and all $X , Y \in { \mathcal { S } }$

$$
\begin{array} { r l } & { \| \nabla F ( X ) - \nabla F ( Y ) \| _ { \mathsf { F } } } \\ & { \quad \leq \left( \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \| \nabla F ( X ) \| _ { \mathsf { F } } ^ { q } \right) \| X - Y \| _ { \mathsf { F } } . } \end{array}\tag{3}
$$

For $q = 0$ , the power is interpreted as 1, including at a zero gradient. The condition applies when several blocks change simultaneously, as in Algorithm 1. A bound only for $X , Y$ difering in one block would not justify the joint descent and momentum-drift estimates used below.

Assumption 2.4 (Layer-wise Variance). The stochastic gradient oracle is assumed to have a layer-wise variance bound, where the parameter space $\pmb { S } = \mathbb { R } ^ { n _ { 1 } \times m _ { 1 } } \times \dots \times \mathbb { R } ^ { n _ { L } \times m _ { L } }$ is the Cartesian product of the weight spaces of all L layers. Specifically, for each layer $l \in \{ 1 , \ldots , L \}$ , there exists a constant $\sigma _ { ( l ) } \geq 0$ such that, for any $X \in S$ , the noisy gradient $\nabla _ { ( l ) } f ( X ; \xi )$ with respect to the l-th layer satisfies:

$$
\begin{array} { r l } & { \mathbb { E } _ { \xi } \left[ \nabla _ { ( l ) } f ( X ; \xi ) \right] = \nabla _ { ( l ) } F ( X ) , } \\ & { \mathbb { E } _ { \xi } \left[ \left\| \nabla _ { ( l ) } f ( X ; \xi ) - \nabla _ { ( l ) } F ( X ) \right\| _ { \mathsf { F } } ^ { 2 } \right] \leq \sigma _ { ( l ) } ^ { 2 } , } \end{array}\tag{4}
$$

where $\xi$ denotes the random sampling index and $\| \cdot \| _ { \mathsf { F } }$ denotes the Frobenius norm. At iteration $t ,$ $\xi _ { t }$ is sampled independently of the past $\mathcal { F } _ { t - 1 }$ , which includes $X _ { t }$ . Thus the same statements hold conditionally on $\mathcal { F } _ { t - 1 }$ . Independence of the noise across layers is not required.

Scope of the assumptions. Assumption 2.3 includes ordinary global $\mathcal { L } _ { 0 } .$ -smoothness when $\mathcal { L } _ { 1 } = 0$ and otherwise allows the smoothness bound to depend on the gradient norm. We use the joint condition because all blocks are updated together. Assumption 2.4 controls conditional variance, not the magnitude of every stochastic gradient; no almost-sure gradient bound, noise symmetry, or sub-Gaussian tail assumption is imposed. The finite-step result below further dispenses with a positive lower bound on nonzero output singular values. These are the specific relaxations used in the analysis; no additional architecture assumption is needed.

## 3 Main Results

Throughout, $\nabla F _ { ( l ) } ( X )$ and $\nabla _ { ( l ) } F ( X )$ denote the same gradient block. We develop one descent– tracking argument for Algorithm 1, first under output singular-value bounds and then with the geometry of the finite Newton–Schulz map. The block sums retain layer-specific noise levels, while gradient drift is controlled for the joint move $X _ { t + 1 } - X _ { t }$ on the coupled objective.

## 3.1 A common descent and tracking argument

The proof has four steps: derive and telescope a descent inequality; expand the Nesterov tracking error; bound its initialization, noise, and gradient-drift terms; and substitute these bounds to absorb the remaining gradient term. The orthogonalization routine enters through alignment and update-energy bounds. We establish these bounds in two forms below.

Control through output singular values. The first form uses Proposition 2.1 and gives a nuclear-norm stationarity bound with explicit rank dependence. The positive output singular-value floor is an additional spectral condition. The finite-map estimates later in this subsection lead to the Frobenius result in Section 3.2 without that condition.

Theorem 3.1 (Convergence of Layer-wise Muon). Suppose that Assumptions 2.2, 2.3, and $\it 2 . 4$ hold. Let $\{ X _ { t } \} _ { t = 1 } ^ { T + 1 }$ be generated by Algorithm 1 with $\beta _ { 1 } \in [ 0 , 1 ) , \beta _ { 2 } \in [ 0 , 1 ]$ and $\gamma > 0$ . For each layer l, let $r ^ { ( l ) }$ bound both rank $( N _ { t } ^ { ( l ) } )$ and rank $( N _ { t } ^ { ( l ) } - \nabla _ { ( l ) } F ( X _ { t } ) )$ along the iterates (the choice $r ^ { ( l ) } = \operatorname* { m i n } \{ n _ { l } , m _ { l } \}$ always sufices), and let the inexact orthogonalization satisfy Proposition 2.1 with fixed constants $\varsigma _ { \mathrm { m i n } } > 0$ and $\varsigma _ { \mathrm { m a x } } > 0$ uniformly over all layers and iterates. Define $\hat { \mathcal { L } } = \mathcal { L } _ { 0 } + ( 1 - q ) \mathcal { L } _ { 1 }$ $\begin{array} { r } { R _ { 0 } = \sum _ { l = 1 } ^ { L } r ^ { ( l ) } , S _ { 0 } = \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \sigma _ { ( l ) } , g _ { l } = \sqrt { r ^ { ( l ) } } \left. \nabla F _ { ( l ) } ( X _ { 1 } ) \right. _ { \mathrm { F } } , a n d G _ { 0 } = \sum _ { l = 1 } ^ { L } g _ { l } , g _ { 0 } = \sum _ { l = 1 } ^ { L } \sum _ { \tau } g _ { \tau } \mathcal { U } _ { 0 } . } \end{array}$ . Set

$$
\Delta _ { \gamma , \beta _ { 1 } } : = \varsigma _ { \mathrm { m i n } } - \frac { \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { 2 } - \frac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { 1 - \beta _ { 1 } } .
$$

Then

$$
\begin{array} { r l } {  { \Delta _ { \gamma , \beta _ { 1 } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } [ \| \nabla F _ { ( l ) } ( X _ { t } ) \| _ { * } ] } \quad } & { } \\ & { \leq \frac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + \frac { 2 \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } G _ { 0 } } { T ( 1 - \beta _ { 1 } ) } } \\ & { + \ 2 ( \sqrt { 1 - \beta _ { 1 } } + \sqrt { 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } ) \varsigma _ { \operatorname* { m a x } } S _ { 0 } } \\ & { + \ \frac { 2 \gamma \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } ^ { 2 } \hat { L } } { 1 - \beta _ { 1 } } R _ { 0 } + \frac { \gamma \varsigma _ { \operatorname* { m a x } } ^ { 2 } \hat { L } R _ { 0 } } { 2 } . } \end{array}\tag{5}
$$

Corollary 3.2 (Rate of Layer-wise Muon with scheduled Nesterov correction). Suppose that the conditions of Theorem 3.1 hold, with $F ( X _ { 1 } ) - F ^ { * } > 0$ and $S _ { 0 } > 0$ for the optimized parameter formulas below. Denote $\begin{array} { r } { R _ { 0 } = \sum _ { l = 1 } ^ { L } r ^ { ( l ) } , S _ { 0 } = \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \sigma _ { ( l ) } } \end{array}$ , and $\begin{array} { r } { G _ { 0 } = \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \left. \nabla F _ { ( l ) } ( X _ { 1 } ) \right. _ { \mathsf { F } } , } \end{array}$ Let $\beta _ { 2 } = 1 - ( 1 - \beta _ { 1 } ) ^ { a }$ for $\begin{array} { r } { a > \frac { 1 } { 2 } } \end{array}$ . Since $\beta _ { 2 } \leq 1$ , choose $C _ { 1 } , C _ { 2 } , \gamma , \beta _ { 1 }$ by minimizing the leading upper envelope

$$
\begin{array} { r l r } {  { \Phi ( \gamma , \beta _ { 1 } ) : = \frac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + 2 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } } } \\ & { } & { + \frac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal L } R _ { 0 } } { 1 - \beta _ { 1 } } , } \end{array}
$$

whose Young equality condition is

$$
\frac { 4 ( F ( X _ { 1 } ) - F ^ { * } ) } { \gamma T } = 4 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } = \frac { 8 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal { L } } R _ { 0 } } { 1 - \beta _ { 1 } } .
$$

This gives

$$
\begin{array} { r l } & { C _ { 1 } = \frac { \left( F ( X _ { 1 } ) - F ^ { * } \right) ^ { 3 / 4 } } { 2 ^ { 1 / 4 } \varsigma _ { \mathrm { m a x } } S _ { 0 } ^ { 1 / 2 } \hat { \mathcal { L } } ^ { 1 / 4 } R _ { 0 } ^ { 1 / 4 } } , } \\ & { C _ { 2 } = \frac { 2 ^ { 1 / 2 } \hat { \mathcal { L } } ^ { 1 / 2 } \left( F ( X _ { 1 } ) - F ^ { * } \right) ^ { 1 / 2 } R _ { 0 } ^ { 1 / 2 } } { S _ { 0 } } , } \\ & { ~ \gamma = \frac { C _ { 1 } } { T ^ { 3 / 4 } } , ~ \beta _ { 1 } = 1 - \frac { C _ { 2 } } { T ^ { 1 / 2 } } . } \end{array}
$$

Assume that T is large enough such that $\beta _ { 1 } \in [ 0 , 1 )$ and

$$
T \geq \left( \left( \frac { C _ { 1 } \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { \varsigma _ { \mathrm { m i n } } } \right) ^ { 1 / 3 } + \frac { 4 C _ { 1 } \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { C _ { 2 } \varsigma _ { \mathrm { m i n } } } \right) ^ { 4 } .\tag{6}
$$

Then the iterates generated by Algorithm 1 satisfy

$$
\begin{array} { l } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \left[ \left\| \nabla F ( t _ { l } ) ( X _ { t } ) \right\| _ { * } \right] \leq \frac { 2 \cdot 5 1 2 ^ { 1 / 4 } \zeta _ { \operatorname* { m a x } } \hat { Z } ^ { 1 / 4 } S _ { 0 } ^ { 1 / 2 } R _ { 0 } ^ { 1 / 4 } } { \operatorname* { c m i n } T ^ { 1 / 4 } } } \\ { \displaystyle \quad \cdot ( F ( X _ { 1 } ) - F ^ { * } ) ^ { 1 / 4 } + \frac { 4 \zeta _ { \operatorname* { m a x } } G _ { 0 } } { C _ { 2 } \zeta _ { \operatorname* { m i n } } T ^ { 1 / 2 } } } \\ { \displaystyle \quad + \frac { 4 \sqrt { 2 } \zeta _ { \operatorname* { m a x } } \hat { S } _ { 0 } } { \zeta _ { \operatorname* { m i n } } } \left( \frac { C _ { 2 } ^ { ( 1 + a ) / 2 } } { T ^ { ( 1 + a ) / 4 } } + \frac { C _ { 2 } ^ { a } } { T ^ { a / 2 } } \right) } \\ { \displaystyle \qquad + \frac { C _ { 1 } \zeta _ { \operatorname* { m a x } } ^ { 2 } \hat { Z } R _ { 0 } } { \zeta _ { \operatorname* { m i n } } T ^ { 3 / 4 } } . } \end{array}\tag{7}
$$

Consequently,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \left[ \left. \nabla F _ { ( l ) } ( X _ { t } ) \right. _ { * } \right] = \mathcal { O } \left( T ^ { - 1 / 4 } \right) .\tag{8}
$$

Control through the finite Newton–Schulz map. To use the same proof for a fixed number of polynomial steps, we derive alignment and energy bounds from the actual scalar map. Its output can approach zero on small input singular values, so the preceding theorem’s positive output floor is not generally available. The ratio of output to input singular values provides the required control instead. For $N \neq 0$ , initialize $Z _ { 0 } = N / \| N \| _ { \mathsf { F } }$ and take a fixed number J of polynomial steps

$$
\begin{array} { r l } & { Z _ { j } = a _ { j } Z _ { j - 1 } + b _ { j } ( Z _ { j - 1 } Z _ { j - 1 } ^ { \top } ) Z _ { j - 1 } } \\ & { \qquad + c _ { j } ( Z _ { j - 1 } Z _ { j - 1 } ^ { \top } ) ^ { 2 } Z _ { j - 1 } , \qquad j = 1 , \ldots , J , } \end{array}\tag{9}
$$

with $O = Z _ { J } ;$ set $O = 0$ when $N = 0$ . The coeficients may vary with $j ,$ but J and their values are fixed independently of matrix size and T. All statements concern exact arithmetic. Write

$$
p _ { j } ( s ) = a _ { j } s + b _ { j } s ^ { 3 } + c _ { j } s ^ { 5 } , \qquad \phi _ { J } = p _ { J } \circ \cdot \cdot \cdot \circ p _ { 1 } .\tag{10}
$$

We require the following property of this scalar map:

$$
\begin{array} { c c } { 0 < \ell _ { J } : = \displaystyle \operatorname* { i n f } _ { 0 < s \leq 1 } \frac { \phi _ { J } ( s ) } { s } , } & { u _ { J } : = \displaystyle \operatorname* { s u p } _ { 0 < s \leq 1 } \frac { \phi _ { J } ( s ) } { s } < \infty , } \\ { c _ { J } : = \displaystyle \operatorname* { m a x } _ { 0 \leq s \leq 1 } \phi _ { J } ( s ) . } \end{array}\tag{11}
$$

The ratio has the continuous extension $\begin{array} { r } { \phi _ { J } ^ { \prime } ( 0 ) = \prod _ { j = 1 } ^ { J } a _ { j } } \end{array}$ at zero. Thus these are scalar constants, not bounds on a matrix’s smallest nonzero singular value. Positive lower bounds and finite upper bounds can also be used in place of the extrema in (11).

Proposition 3.3 (Finite-map alignment and energy). Let $N = U \Sigma V ^ { \top } \neq 0 , R = \| N \| _ { \mathsf { F } }$ , and $s _ { i } = \sigma _ { i } ( N ) / R$ . Then $\textstyle \sum _ { i } s _ { i } ^ { 2 } = 1$ and the direction in (9) is $O = U \mathrm { d i a g } ( \phi _ { J } ( s _ { i } ) ) V ^ { \top }$ . Define

$$
A ( N ) = \sum _ { i } s _ { i } \phi _ { J } ( s _ { i } ) , \qquad B ( N ) = \sum _ { i } \phi _ { J } ( s _ { i } ) ^ { 2 } .\tag{12}
$$

The exact identities and bounds are

$$
\begin{array} { r l } { \langle N , O \rangle = R A ( N ) , \quad } & { \| O \| _ { \mathsf { F } } ^ { 2 } = B ( N ) , } \\ { \ell _ { J } \leq A ( N ) \leq u _ { J } , \quad } & { B ( N ) \leq u _ { J } A ( N ) \leq u _ { J } ^ { 2 } , } \\ { \| O \| _ { \mathrm { o p } } \leq c _ { J } . } \end{array}\tag{13}
$$

Moreover, with

$$
\bar { \rho } _ { J } : = \frac { u _ { J } + \ell _ { J } } { 2 \sqrt { u _ { J } \ell _ { J } } } ,\tag{14}
$$

we have $\sqrt { B ( N ) } \leq \bar { \rho } _ { J } A ( N )$ . Consequently, for any matrix H and $E = N - H$

$$
\langle H , O \rangle \geq A ( N ) { \big ( } \| N \| _ { \mathsf { F } } - { \bar { \rho } } _ { J } \| E \| _ { \mathsf { F } } { \big ) } .\tag{15}
$$

No rank factor appears in these statements.

An explicit five-step instance.

For the fixed quintic coeficients $( a , b , c ) = ( 3 . 4 4 4 5 , - 4 . 7 7 5 , 2 . 0 3 1 5 )$ , as used in the original Muon implementation (Jordan, 2024), one may use the fully analytic bounds

$$
\begin{array} { l } { { \ell _ { J } \geq \displaystyle \frac { 1 7 } { 2 5 } , ~ u _ { J } = 3 . 4 4 4 5 ^ { J } , } } \\ { { \begin{array} { l } { { c _ { J } \leq \displaystyle \frac { 5 } { 4 } , ~ J \geq 1 . } } \end{array} } } \end{array}\tag{16}
$$

The lower bound uses invariant intervals of the whole scalar map; it is stronger than multiplying a separate worst-case lower bound at every step. For $J = 5 , u _ { 5 } = 3 . 4 4 4 5 ^ { 5 } \simeq 4 8 4 . 8 7 6 3$ , and the displayed bounds imply ${ \bar { \rho } } _ { 5 } < 1 3 . 3 7 1$ . These are certified bounds, not an assertion that a numerical scan has found the sharp lower or angular constant. The details are given in Appendix D.2.

The bounds $\langle N , O \rangle \geq \ell _ { J } \| N \| _ { \mathsf { F } }$ and $\| O \| _ { \mathsf { F } } \le u _ { J }$ now replace the nuclear-norm alignment and rank-dependent update bound in the same four-step argument. The Nesterov error expansion is unchanged; joint gradient drift is controlled using $\| X _ { t + 1 } - X _ { t } \| _ { \mathsf { F } } \leq \gamma u _ { J } \sqrt { L }$

## 3.2 Dimension-independent convergence of practical Muon

We apply the argument of Section 3.1 with its finite-map estimates. The resulting stationarity measure is the sum of block Frobenius norms, which also bounds the Frobenius norm of the complete gradient. No positive output singular-value floor or rank bound for the momentum or tracking error is required.

Theorem 3.4 (Dimension-independent Frobenius stationarity). Suppose Assumptions 2.2, 2.3, and 2.4 hold. Use (9) in Algorithm 1, with (11), $\beta _ { 1 } \in [ 0 , 1 ) , \beta _ { 2 } \in [ 0 , 1 ]$ , and $\gamma > 0$ . Define

$$
\begin{array} { r l } & { \quad \delta = 1 - \beta _ { 1 } , \qquad D = F \bigl ( X _ { 1 } \bigr ) - F ^ { * } , } \\ & { \quad S _ { \mathsf { F } } = \displaystyle \sum _ { l = 1 } ^ { L } \sigma _ { ( l ) } , } \\ & { \quad G _ { \mathsf { F } } = \displaystyle \sum _ { l = 1 } ^ { L } \| \nabla _ { ( l ) } F ( X _ { 1 } ) \| _ { \mathsf { F } } , } \\ & { \quad v _ { \beta } = \sqrt { \delta } + \sqrt { 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } , } \\ & { \quad \hat { \mathcal { L } } = \mathcal { L } _ { 0 } + ( 1 - q ) \mathcal { L } _ { 1 } . } \end{array}\tag{17}
$$

Set

$$
\Delta _ { \gamma , \beta _ { 1 } } ^ { \mathsf { F } } : = \ell _ { J } - \frac { \gamma L u _ { J } ^ { 2 } q \mathcal { L } _ { 1 } } { 2 } - \frac { \gamma L u _ { J } ( \ell _ { J } + u _ { J } ) q \mathcal { L } _ { 1 } } { \delta } .\tag{18}
$$

Then

$$
\begin{array} { r l } & { \displaystyle \Delta _ { \gamma , \beta _ { 1 } } ^ { \sf F } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \| \nabla _ { ( l ) } F ( X _ { t } ) \| _ { \sf F } } \\ & { \displaystyle \leq \frac { D } { \gamma T } + \frac { \beta _ { 2 } \left( \ell _ { J } + u _ { J } \right) G _ { \sf F } } { T \delta } + ( \ell _ { J } + u _ { J } ) v _ { \beta } S _ { \sf F } } \\ & { \quad + \frac { \gamma \beta _ { 2 } L u _ { J } \left( \ell _ { J } + u _ { J } \right) \hat { \mathcal { L } } } { \delta } + \frac { \gamma L u _ { J } ^ { 2 } \hat { \mathcal { L } } } { 2 } . } \end{array}\tag{19}
$$

In particular, a positive $\Delta _ { \gamma , \beta _ { 1 } } ^ { \mathsf { F } }$ gives a bound on the original, unstopped sequence of iterates.

Corollary 3.5 (Dimension-independent $T ^ { - 1 / 4 } \ \mathrm { r a t e } )$ . Under Theorem $\ 3 . 4 ,$ fix $\gamma _ { 0 } , \delta _ { 0 } > 0$ and $a > 1 / 2$ and set

$$
\gamma = \gamma _ { 0 } T ^ { - 3 / 4 } , \quad 1 - \beta _ { 1 } = \delta _ { 0 } T ^ { - 1 / 2 } , \quad \beta _ { 2 } = 1 - ( 1 - \beta _ { 1 } ) ^ { a } .\tag{20}
$$

For all suficiently large $T , \ \beta _ { 1 } , \beta _ { 2 } \in [ 0 , 1 ]$ and $\Delta _ { \gamma , \beta _ { 1 } } ^ { \mathsf { F } } \geq \ell _ { J } / 2$ . Writing $b _ { J } = \ell _ { J } + u _ { J }$

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { \ell = 1 } ^ { T } \mathbb { E } \| \nabla _ { ( h ) } F ( X _ { t } ) \| _ { \mathbf { F } } } \\ & { \le \displaystyle \frac { 2 } { \ell } \Bigg [ \left( \frac { D } { \gamma _ { 0 } } + b _ { J } S _ { \mathrm { F } } \sqrt { \delta _ { 0 } } + \frac { \gamma _ { 0 } I M _ { J } b _ { J } \hat { Z } } { \delta _ { 0 } } \right) T ^ { - 1 / 4 } } \\ & { \qquad + \displaystyle \frac { b _ { J } G _ { \mathrm { F } } } { \delta _ { 0 } T ^ { 1 / 2 } } } \\ & { \qquad + \sqrt { 2 } b _ { J } S _ { \mathrm { F } } \left( \frac { \delta _ { 0 } ^ { ( 1 + \delta ) / 2 } } { T ^ { 1 + \delta + 1 / 4 } } + \frac { \delta _ { 0 } ^ { \alpha } } { T ^ { \alpha / 2 } } \right) } \\ & { \qquad + \displaystyle \frac { \gamma _ { 0 } I M _ { J } ^ { 2 } \hat { Z } ^ { \star } } { 2 T ^ { 3 / 4 } } \Bigg ] . } \end{array}\tag{21}
$$

The same upper bound holds for $\begin{array} { r } { T ^ { - 1 } \sum _ { t } \mathbb { E } \| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } } \end{array}$ . Its constant contains no $n _ { l } , \ m _ { l } , \ r ^ { ( l ) }$ , or $\begin{array} { r } { d = \sum _ { l } n _ { l } m _ { l } } \end{array}$

The standard Nesterov choice is included. Setting $a = 1$ in (20) gives $\beta _ { 2 } = \beta _ { 1 }$ and hence (1). In this case,

$$
v _ { \beta } = \sqrt { \delta } + \delta \sqrt { 2 \beta _ { 1 } ( 1 + \beta _ { 1 } ) } \le \sqrt { \delta } + 2 \delta .\tag{22}
$$

Thus the extra Nesterov noise term is $\mathcal { O } ( T ^ { - 1 / 2 } )$ , smaller than the leading $\mathcal { O } ( T ^ { - 1 / 4 } )$ term. This specialization uses the usual Nesterov algebra, with a horizon-dependent momentum coeficient; it does not assert the same vanishing bound for a fixed numerical momentum coeficient and a fixed batch size.

A suficient explicit horizon in this corollary is

$$
\begin{array} { r l } & { \quad T \geq \operatorname* { m a x } \{ 1 , \delta _ { 0 } ^ { 2 } , ( A _ { J } + B _ { J } ^ { 1 / 3 } ) ^ { 4 } \} , } \\ & { \quad A _ { J } = \frac { 2 \gamma _ { 0 } L u _ { J } \left( \ell _ { J } + u _ { J } \right) q \mathcal { L } _ { 1 } } { \delta _ { 0 } \ell _ { J } } , \qquad B _ { J } = \frac { \gamma _ { 0 } L u _ { J } ^ { 2 } q \mathcal { L } _ { 1 } } { \ell _ { J } } . } \end{array}\tag{23}
$$

For $q \mathcal { L } _ { 1 } = 0$ , the absorption condition is automatic. The initial-gradient constant can also be eliminated: joint generalized smoothness and lower boundedness imply

$$
G _ { \mathsf { F } } \leq \sqrt { L } \left[ D q \mathcal { L } _ { 1 } + \sqrt { D ^ { 2 } q ^ { 2 } \mathcal { L } _ { 1 } ^ { 2 } + 2 D \hat { \mathcal { L } } } \right] .\tag{24}
$$

Thus, for fixed $L , J ,$ , coeficient schedule, $\gamma _ { 0 } , \delta _ { 0 } , a , \mathcal { L } _ { 0 } , \mathcal { L } _ { 1 } , q , D$ , and $S _ { \mathsf { F } }$ , the complete displayed bound is independent of parameter dimension. This is not a proof that these problem quantities stay fixed when a model family changes.

## 4 Discussion

We discuss the coupled momentum and learning-rate schedules, the meaning of dimension independence, and the scope of the orthogonalization and normalization choices covered by the analysis.

## 4.1 Momentum schedules

Remark 4.1 (Coupled Muon schedules versus fixed momentum). There is a growing body of evidence that stable and transferable learning rates are not architecture-agnostic. The $\mu \mathrm { P }$ framework gives width-aware hyperparameter transfer rules for large neural networks (Yang et al., 2022), and recent Muon pretraining studies combine Muon with µP-style transfer and carefully chosen matrix update scales (Liu et al., 2025; Shah et al., 2025). For SGD/GD-style training, prior analyses have also identified nontrivial depth dependence of maximal or efective learning rates, including maximal initial learning rates in deep ReLU networks (Iyer et al., 2023), depth-dependent $\mu \mathrm { P }$ learning rates (Jelassi et al., 2023), and architecture-aware learning-rate rules that depend on depth, width, kernel size, and graph topology (Chen et al., 2024). In Transformer training with Adam, Xiong et al. (2020) also showed that depth-dependent gradient behavior is closely related to learning-rate warmup and normalization design.

The rank-sensitive bound makes explicit how matrix structure enters one suficient Muon stepsize. Corollary 3.2 should be interpreted as a theoretical coupled hyperparameter schedule for Muon. The learning-rate rule

$$
\gamma \asymp \frac { 1 } { S _ { 0 } ^ { 1 / 2 } R _ { 0 } ^ { 1 / 4 } T ^ { 3 / 4 } }
$$

is obtained together with the momentum schedule

$$
1 - \beta _ { 1 } \asymp \frac { R _ { 0 } ^ { 1 / 2 } } { S _ { 0 } \sqrt { T } } , \qquad 1 - \beta _ { 2 } = ( 1 - \beta _ { 1 } ) ^ { a } , \quad a > \frac 1 2 .
$$

Thus, the depth-rank dependence of $\gamma$ is not a stand-alone law under an arbitrary fixed momentum coeficient. It is the result of balancing the three leading terms in the upper envelope

$$
\frac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + 2 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } + \frac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal { L } } R _ { 0 } } { 1 - \beta _ { 1 } } .
$$

In particular, if one fixes $\delta : = 1 - \beta _ { 1 }$ as in conventional training $( e . g . , \delta = 0 . 1 \ \mathrm { o r } \ 0 . 0 5 )$ , then the same envelope becomes

$$
\frac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + 2 \sqrt { \delta } \varsigma _ { \mathrm { m a x } } S _ { 0 } + \frac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal { L } } R _ { 0 } } { \delta } .
$$

Optimizing only over $\gamma$ yields the diferent scale $\gamma \asymp \sqrt { \delta / ( R _ { 0 } T ) }$ , and the stochastic term $2 \sqrt { \delta } \varsigma _ { \mathrm { m a x } } S _ { 0 }$ does not vanish with $T$ . Therefore, the rate and the learning-rate schedule in Corollary $3 . 2$ rely on the scheduled first-momentum gap $1 - \beta _ { 1 } = \Theta ( R _ { 0 } ^ { 1 / 2 } / ( S _ { 0 } T ^ { 1 / 2 } ) )$ . The rank bound $R _ { 0 }$ directly enters the learning-rate scale because Muon’s orthogonalized matrix direction satisfies $\| O _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \lesssim \sqrt { r ^ { ( l ) } }$ so the smoothness cost accumulates as $\textstyle \sum _ { l } r ^ { ( l ) }$ . This rank-sensitive term is specific to the matrix geometry of Muon and is not present in the same form for coordinate-wise Adam or Euclidean SGD. By contrast, the leading schedule for $1 - \beta _ { 1 }$ depends on the layer-wise noise aggregate $S _ { 0 }$ the accumulated rank $R _ { 0 }$ , and the horizon $T ;$ remaining problem-dependent efects enter through constants such as $\hat { \mathcal { L } }$ and the initial optimality gap.

These rank factors belong to the nuclear-norm bound in Corollary 3.2. The finite-step Frobenius bound in Corollary 3.5 uses diferent spectral estimates and needs no explicit rank factor in its schedule.

This theoretical schedule should not be confused with empirical neural scaling laws of the Kaplan/Chinchilla type (Kaplan et al., 2020; Hofmann et al., 2022). Our analysis characterizes how Muon’s hyperparameters should depend on depth, accumulated rank bound, and training horizon to guarantee stationarity; it does not by itself imply a power-law relation between test loss, dataset size, model size, and compute.

Remark 4.2 (Role of the Nesterov momentum correction). The Nesterov momentum correction plays a diferent role from the coupled learning-rate and first-momentum schedules above. When $\beta _ { 2 } = 1$ $N _ { t } ^ { ( l ) }$ reduces to the standard momentum ${ \mathbf { } } M _ { t } ^ { ( l ) }$ . From Theorem 3.1, the Nesterov coeficient enters the initialization term and the following noise–drift terms

$$
2 \sqrt { 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } \varsigma _ { \mathrm { m a x } } S _ { 0 } + \frac { 2 \gamma \beta _ { 2 } \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal { L } } R _ { 0 } } { 1 - \beta _ { 1 } } .
$$

The bound therefore exhibits a trade-of: a smaller $\beta _ { 2 }$ reduces the initialization and drift coeficients, while changing the stochastic term. This does not by itself prove an empirical early- or late-stage advantage. The extra stochastic contribution and the change in the drift coeficient relative to $\beta _ { 2 } = 1$ are lower-order under $\beta _ { 2 } = 1 - ( 1 - \beta _ { 1 } ) ^ { a }$ with $\begin{array} { r } { a > \frac { 1 } { 2 } } \end{array}$ ; the drift term itself remains leading-order. Thus the Nesterov correction does not change the leading $\mathcal { O } ( T ^ { - 1 / 4 } )$ convergence rate.

## 4.2 Dimension independence

The bound in Corollary 3.5 contains no explicit matrix-dimension or rank factor when the block count and problem constants are fixed. The constants $\ell _ { J }$ and $u J$ depend on the finite polynomial map; dimension independence does not imply that these constants are small or that the noise and smoothness constants stay fixed as an architecture changes.

Remark 4.3 (What the sharper spectral identities do and do not imply). Equations (12)–(15) retain the full finite map and give a useful descent-or-small-gradient test for an individual block. However, $\| O \| _ { \mathrm { o p } } \leq c _ { J }$ cannot replace $\| O \| _ { \mathsf { F } } \le u _ { J }$ in a Frobenius-smoothness bound without an additional curvature condition. Similarly, an expected tracking-error bound at each deterministic time does not imply the same bound at the first time that alignment becomes unreliable. We therefore prove the averaged stochastic result directly, using the same three-term tracking decomposition, rather than relying on a stopping-time conversion. Appendix D.5 gives the precise geometric certificate and explains this distinction. The present theorem establishes dimension independence; it does not claim a uniformly small NS constant.

## 4.3 Implementation scope

Remark 4.4 (Generality beyond exact Newton–Schulz orthogonalization). Our analysis does not require $O _ { t } ^ { ( l ) }$ to be an exact orthogonal or polar factor. Exact orthogonalization corresponds to the special case $\varsigma _ { i } = 1$ for all nonzero singular directions. In contrast, the proof of Theorem 3.1 only uses the two inequalities in Proposition 2.1,

$$
\langle N _ { t } ^ { ( l ) } , O _ { t } ^ { ( l ) } \rangle \geq \varsigma _ { \mathrm { m i n } } \| N _ { t } ^ { ( l ) } \| _ { * } , \qquad \| O _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \leq \varsigma _ { \mathrm { m a x } } \sqrt { r ^ { ( l ) } } .
$$

Therefore, exact polar decomposition is not needed. It is suficient that the orthogonalization routine preserves enough directional alignment with $N _ { t } ^ { ( l ) }$ and keeps the update magnitude uniformly bounded.

This viewpoint is related to, but more algorithm-agnostic than, the recent analysis of Muon with Newton–Schulz orthogonalization (Kim and Oh, 2025). That work studies the practical Newton– Schulz iteration directly and shows that, for a fixed number of Newton–Schulz steps, Muon converges at the same rate as the ideal SVD-polar version up to a constant factor, with the factor converging to one rapidly as the number of Newton–Schulz steps increases. Our argument does not require the modified singular values to be symmetric around 1, nor does it require $\varsigma _ { \mathrm { m i n } }$ and $\varsigma _ { \mathrm { m a x } }$ to approach 1 as the number of Newton–Schulz iterations grows. The constants in our bound depend on $\varsigma _ { \mathrm { m i n } }$ and $\zeta _ { \mathrm { m a x } } .$ , and the same rate is preserved as long as $\varsigma _ { \mathrm { m i n } } > 0$ and the distortion ratio $\varsigma _ { \mathrm { m a x } } / \varsigma _ { \mathrm { m i n } }$ remains controlled. Consequently, the proof applies not only to Newton–Schulz approximations, but also to other inexact orthogonalization or spectral-flattening procedures satisfying the same alignment and magnitude conditions. For Frobenius-normalized finite polynomial steps, however, $\phi _ { J } ( s ) \to 0$ as $s \to 0 ,$ . Hence a uniform $\varsigma _ { \mathrm { m i n } } > 0$ need not exist over arbitrary spectra. Sections 3.1 and 3.2 avoid this condition by using $\phi _ { J } ( s ) / s$ and proving Frobenius, rather than nuclear-norm, stationarity.

Remark 4.5 (Normalization and implementation scope). If the initial normalization is $N / ( \kappa \| N \| _ { \mathsf { F } } )$ for a fixed $\kappa \geq 1$ , apply the results to the efective map $\tilde { \phi } _ { J } ( s ) = \phi _ { J } ( s / \kappa )$ on [0, 1]. In particular, the slope at zero is $( \prod _ { j } a _ { j } ) / \kappa$ , not $\textstyle \prod _ { j } a _ { j }$ . For $N / ( \| N \| _ { \mathsf { F } } + \varepsilon )$ with fixed $\varepsilon > 0$ , the same energy bound holds and

$$
\langle N , O \rangle \ge \ell _ { J } \frac { \| N \| _ { \mathsf { F } } ^ { 2 } } { \| N \| _ { \mathsf { F } } + \varepsilon } \ge \ell _ { J } ( \| N \| _ { \mathsf { F } } - \varepsilon ) .
$$

Accordingly, add $L \ell _ { J } \varepsilon$ to the right-hand side of (19); (21) acquires the residual $2 L \varepsilon$

If the output in block l is multiplied by a fixed $\alpha _ { l } > 0$ , the same proof applies with $\ell _ { J }$ replaced by $\alpha _ { \mathrm { m i n } } \ell _ { J }$ and $u J$ by $\alpha _ { \operatorname* { m a x } } u J$ , where $\alpha _ { \mathrm { m i n } } ~ = ~ \mathrm { m i n } _ { l } \alpha _ { l }$ and $\alpha _ { \mathrm { m a x } } = \operatorname* { m a x } _ { l } \alpha _ { l } ;$ the operator-norm bound becomes $\alpha _ { \mathrm { m a x } } c _ { J }$ . This follows by applying the blockwise inequalities with these common bounds. Such scaling preserves dimension independence only when $\alpha _ { \mathrm { m a x } }$ and $1 / \alpha _ { \mathrm { m i n } }$ remain bounded independently of matrix size. Unbounded size-dependent multipliers cannot be absorbed into a dimension-independent constant. No finite-precision error guarantee is asserted here.

## 5 Conclusion

We established convergence guarantees for layer-wise Muon with finite Newton–Schulz updates and Nesterov momentum under joint gradient-dependent smoothness and bounded conditional variance. A common descent-and-tracking argument separates the momentum error into initialization, stochastic noise, and gradient drift. Controlling the finite polynomial through $\phi _ { J } ( s ) / s$ yields an $\mathcal { O } ( T ^ { - 1 / 4 } )$ bound on the expected average Frobenius gradient norm under coupled learning-rate and momentum schedules, including the standard equal-coeficient Nesterov rule. The bound has no explicit matrixdimension or rank factors when the block count and problem constants are fixed. We also verified the scalar-map conditions analytically for the original five-step quintic iteration.

The central implication is that finite orthogonalization provides suficient alignment and updateenergy control for convergence without a positive lower bound on its nonzero output singular values. The complementary nuclear-norm result identifies where rank dependence enters under stronger spectral control. Extending the analysis to finite-precision arithmetic and to fixed momentum with increasing batch sizes would further connect these guarantees to training practice.

## References

Chen, W., Wu, J., Wang, Z., and Hanin, B. (2024). Principled architecture-aware scaling of hyperparameters. In International Conference on Learning Representations.

Choudhury, S., Cheng, X., Takáč, M., Na, S., and Kolar, M. (2026). Muon with Nesterov momentum: Heavy-tailed noise and (randomized) inexact polar decomposition. arXiv preprint arXiv:2605.06884.

Hofmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., Casas, D. d. L., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. W., and Sifre, L. (2022). Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Iyer, G., Hanin, B., and Rolnick, D. (2023). Maximal initial learning rates in deep ReLU networks. In International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 14500–14530. PMLR.

Jelassi, S., Hanin, B., Ji, Z., Reddi, S. J., Bhojanapalli, S., and Kumar, S. (2023). Depth dependence of µP learning rates in ReLU MLPs. arXiv preprint arXiv:2305.07810.

Jordan, K. (2024). Muon: An optimizer for hidden layers in neural networks. Online technical report. Published December 8, 2024.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. (2020). Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Kim, G. Y. and Oh, M.-h. (2025). Convergence of Muon with Newton–Schulz. arXiv preprint arXiv:2510.20377.

Li, M. and Tsuchiya, T. (2026). Muon with finite Newton–Schulz: The smoothing benefit in nonsmooth nonconvex optimization. arXiv preprint arXiv:2608.26288.

Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., Qin, Y., Xu, W., Lu, E., Yan, J., et al. (2025). Muon is scalable for llm training. arXiv preprint arXiv:2502.16982.

Pethick, T., Xie, W., Antonakopoulos, K., Zhu, Z., Silveti-Falls, A., and Cevher, V. (2025). Training deep learning models with norm-constrained LMOs. arXiv preprint arXiv:2502.07529.

Riabinin, A., Shulgin, E., Gruntkowska, K., and Richtárik, P. (2025). Gluon: Making Muon & Scion great again! (bridging theory and practice of LMO-based optimizers for LLMs). arXiv preprint arXiv:2505.13416.

Shah, I., Polloreno, A. M., Stratos, K., Monk, P., Chaluvaraju, A., Hojel, A., Ma, A., Thomas, A., Tanwer, A., Shah, D. J., et al. (2025). Practical eficiency of muon for pretraining. arXiv preprint arXiv:2505.02222.

Shen, W., Huang, R., Huang, M., Shen, C., and Zhang, J. (2025). On the convergence analysis of Muon. arXiv preprint arXiv:2505.23737.

Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L., and Liu, T. (2020). On layer normalization in the transformer architecture. In International Conference on Machine Learning, pages 10524–10533. PMLR.

Yang, G., Hu, E. J., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W., and Gao, J. (2022). Tensor programs V: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466.

## A Useful Lemmas

Lemma A.1. Under Assumption 2.3, for any $X , Y \in { \mathcal { S } }$

$$
F ( \boldsymbol { Y } ) \le F ( \boldsymbol { X } ) + \langle \nabla F ( \boldsymbol { X } ) , \boldsymbol { Y } - \boldsymbol { X } \rangle + \frac { \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \| \nabla F ( \boldsymbol { X } ) \| _ { \mathrm { F } } ^ { q } } { 2 } \sum _ { l = 1 } ^ { L } \| Y ^ { ( l ) } - X ^ { ( l ) } \| _ { \mathrm { F } } ^ { 2 } .\tag{25}
$$

Proof. Writing $V = Y - X$ and applying the fundamental theorem of calculus gives

$$
\begin{array} { r l } & { F ( \boldsymbol { Y } ) - F ( \boldsymbol { X } ) - \langle \nabla F ( \boldsymbol { X } ) , \boldsymbol { V } \rangle = \displaystyle \int _ { 0 } ^ { 1 } \langle \nabla F ( \boldsymbol { X } + t \boldsymbol { V } ) - \nabla F ( \boldsymbol { X } ) , \boldsymbol { V } \rangle d t } \\ & { \quad \quad \quad \le \displaystyle \int _ { 0 } ^ { 1 } \| \nabla F ( \boldsymbol { X } + t \boldsymbol { V } ) - \nabla F ( \boldsymbol { X } ) \| _ { \mathsf { F } } \| _ { \mathsf { F } } d t } \\ & { \quad \quad \le \displaystyle \frac { \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \| \nabla F ( \boldsymbol { X } ) \| _ { \mathsf { F } } ^ { q } } { 2 } \| \boldsymbol { V } \| _ { \mathsf { F } } ^ { 2 } . } \end{array}\tag{26}
$$

The last identity $\begin{array} { r } { \| V \| _ { \mathsf { F } } ^ { 2 } = \sum _ { l } \| V ^ { ( l ) } \| _ { \mathsf { F } } ^ { 2 } } \end{array}$ proves the claim.

Lemma A.2. Let $\{ N _ { t } ^ { ( l ) } \} _ { t = 1 } ^ { T }$ be the generalized Nesterov momentum sequence generated by Algorithm 1. Then, for $t \geq 2 , N _ { t } ^ { ( l ) }$ admits the following equivalent recursive form (with $N _ { 1 } ^ { ( l ) } = ( 1 - \beta _ { 1 } \beta _ { 2 } ) G _ { 1 } ^ { ( l ) } ) { : }$

$$
N _ { t } ^ { ( l ) } = \beta _ { 1 } N _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 1 } \beta _ { 2 } ) G _ { t } ^ { ( l ) } - \beta _ { 1 } ( 1 - \beta _ { 2 } ) G _ { t - 1 } ^ { ( l ) } = \beta _ { 1 } N _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 1 } ) G _ { t } ^ { ( l ) } + \beta _ { 1 } ( 1 - \beta _ { 2 } ) ( G _ { t } ^ { ( l ) } - G _ { t - 1 } ^ { ( l ) } ) .\tag{27}
$$

Proof. By the update rule for $N _ { t } ^ { ( l ) }$ , we obtain

$$
\begin{array} { r l } & { N _ { t } ^ { ( l ) } - \beta _ { 1 } N _ { t - 1 } ^ { ( l ) } = \beta _ { 2 } M _ { t } ^ { ( l ) } + ( 1 - \beta _ { 2 } ) G _ { t } ^ { ( l ) } - \beta _ { 1 } ( \beta _ { 2 } M _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 2 } ) G _ { t - 1 } ^ { ( l ) } ) } \\ & { \qquad = \beta _ { 2 } ( \beta _ { 1 } M _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 1 } ) G _ { t } ^ { ( l ) } ) + ( 1 - \beta _ { 2 } ) G _ { t } ^ { ( l ) } - \beta _ { 1 } \beta _ { 2 } M _ { t - 1 } ^ { ( l ) } - \beta _ { 1 } ( 1 - \beta _ { 2 } ) G _ { t - 1 } ^ { ( l ) } } \\ & { \qquad = ( 1 - \beta _ { 1 } \beta _ { 2 } ) G _ { t } ^ { ( l ) } - \beta _ { 1 } ( 1 - \beta _ { 2 } ) G _ { t - 1 } ^ { ( l ) } } \\ & { \qquad = ( 1 - \beta _ { 1 } ) G _ { t } ^ { ( l ) } + \beta _ { 1 } ( 1 - \beta _ { 2 } ) ( G _ { t } ^ { ( l ) } - G _ { t - 1 } ^ { ( l ) } ) . } \end{array}\tag{28}
$$

Rearranging terms gives the desired recursion.

Lemma A.3. Let $a , b > 0$ and $0 < \alpha < \beta$ . Define

$$
\bar { x } : = \left( a + b ^ { \alpha / \beta } \right) ^ { 1 / \alpha } .
$$

Then, for every $x \geq \bar { x }$

$$
\frac { a } { x ^ { \alpha } } + \frac { b } { x ^ { \beta } } \leq 1 .
$$

Moreover, $i f x ,$ <sub>⋆</sub> denotes the smallest positive solution of $a / x ^ { \alpha } + b / x ^ { \beta } \leq 1$ , then

$$
x _ { \star } \le \bar { x } \le 2 ^ { 1 / \alpha } x _ { \star } .
$$

Equivalently, the corresponding threshold in the variable $s = x ^ { \alpha }$ is within a factor of 2 of the optimal threshold.

Proof. Let $s = x ^ { \alpha }$ and $\gamma = \beta / \alpha > 1$ . The desired inequality is equivalent to

$$
\frac { a } { s } + \frac { b } { s ^ { \gamma } } \leq 1\tag{29}
$$

with $s > 0$ . Any feasible s for (29) must satisfy

$$
s \geq \operatorname* { m a x } \{ a , b ^ { 1 / \gamma } \} ,\tag{30}
$$

since each term on the left-hand side of (29) is nonnegative.

Now set $\bar { s } : = a + b ^ { 1 / \gamma }$ . Since $\bar { s } > b ^ { 1 / \gamma }$ , we have $b / \bar { s } ^ { \gamma } \leq b ^ { 1 / \gamma } / \bar { s }$ . Therefore,

$$
\frac { a } { \bar { s } } + \frac { b } { \bar { s } ^ { \gamma } } \leq \frac { a } { \bar { s } } + \frac { b ^ { 1 / \gamma } } { \bar { s } } = 1 .\tag{31}
$$

Thus s¯ is feasible, and consequently every $x \ge \bar { x } : = \bar { s } ^ { 1 / \alpha }$ satisfies the claimed inequality.

Let $s _ { \star }$ be the smallest feasible value of s in (29). By (30), $s _ { \star } \geq \operatorname* { m a x } \{ a , b ^ { 1 / \gamma } \}$ . Hence

$$
\bar { s } = a + b ^ { 1 / \gamma } \leq 2 \operatorname* { m a x } \{ a , b ^ { 1 / \gamma } \} \leq 2 s _ { \star } .
$$

Taking the power $1 / \alpha$ gives $\bar { x } \leq 2 ^ { 1 / \alpha } x ,$ , where $x _ { \star } = s _ { \star } ^ { 1 / \alpha }$ . The inequality $x _ { \star } \leq \bar { x }$ follows from the feasibility of ${ \bar { x } } .$ □

## B Proof of Theorem 3.1

Proof. Step 1: Descent and telescoping. Applying Lemma A.1 with $Y = X _ { t + 1 }$ and $X = X _ { t }$ gives

$$
F ( X _ { t + 1 } ) \leq F ( X _ { t } ) + \langle \nabla F ( X _ { t } ) , X _ { t + 1 } - X _ { t } \rangle + \sum _ { l = 1 } ^ { L } \frac { \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \| \nabla F ( X _ { t } ) \| _ { \mathrm { F } } ^ { q } } { 2 } \| X _ { t + 1 } ^ { ( l ) } - X _ { t } ^ { ( l ) } \| _ { \mathrm { F } } ^ { 2 } .\tag{32}
$$

For compactness in this proof, write $K _ { t } = \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \Vert \nabla F ( X _ { t } ) \Vert _ { \mathsf { F } } ^ { q }$ . Substituting the update rule $X _ { t + 1 } ^ { ( l ) } = X _ { t } ^ { ( l ) } - \gamma O _ { t } ^ { ( l ) }$ into the preceding inequality yields

$$
\begin{array} { r l } { \varepsilon _ { \mathrm { ( 2 , 3 ) , 4 , 5 , 7 , 6 , 7 } } } & { \gamma _ { \mathrm { ( 3 , 4 ) , 4 , 5 , 7 , 6 , 7 , 6 , 7 } } } \\ & { \varepsilon _ { \mathrm { ( 4 , 5 ) , 5 , 6 , 7 , 6 , 7 } } } \\ & { \varepsilon _ { \mathrm { ( 5 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { = \varepsilon _ { \mathrm { ( 1 , 5 ) , 5 , 7 , 6 , 7 } } } \\ & { - \varepsilon _ { \mathrm { ( 1 , 5 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { \varepsilon _ { \mathrm { ( 4 , 5 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { + \frac { 3 } { 2 } \varepsilon _ { \mathrm { ( 1 , 5 ) , 4 , 5 , 7 } } } \\ & { - \varepsilon _ { \mathrm { ( 2 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { - \varepsilon _ { \mathrm { ( 3 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { \varepsilon _ { \mathrm { ( 4 , 5 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { + \frac { 3 } { 2 } \varepsilon _ { \mathrm { ( 2 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { + \frac { 3 } { 2 } \varepsilon _ { \mathrm { ( 2 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { + \frac { 3 } { 2 } \varepsilon _ { \mathrm { ( 3 , 6 ) , 4 , 5 , 7 , 6 , 7 } } } \\ & { - \varepsilon _ { \mathrm { ( 4 , 5 ) , 5 , 7 , 6 , 7 } } } \\ & { \varepsilon _ { \mathrm { ( 4 , 5 ) , 4 , 5 , 7 , 6 , 7 } } } \end{array}\tag{33}
$$

By Proposition 2.1, the inexact orthogonalization direction satisfies $\langle N _ { t } ^ { ( l ) } , O _ { t } ^ { ( l ) } \rangle \geq \varsigma _ { \mathrm { m i n } } \| N _ { t } ^ { ( l ) } \| ,$ and $\| O _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \leq \mathsf { \varsigma _ { \operatorname* { m a x } } } \sqrt { r ^ { ( l ) } }$ . Write $D _ { t } ^ { ( l ) } = N _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } )$ and $H _ { t } = \mathcal { L } _ { 0 } + ( 1 - q ) \mathcal { L } _ { 1 } + q \mathcal { L } _ { 1 } \| \nabla F ( X _ { t } ) \| _ { \mathsf { F } }$

Then

$$
\begin{array} { r l } & { \Gamma ( { \mathbf { \bar { \lambda } } } _ { 1 } ( \lambda _ { 1 } ) ; { \mathbf { \bar { \lambda } } } _ { 2 } ( \lambda _ { 2 } ) , \gamma _ { 1 } ) } \\ & { \quad = \left. { \bar { \lambda } } _ { 2 } ( \lambda _ { 2 } ) , \bar { \lambda } _ { 2 } ( \lambda _ { 1 } ) , \bar { \lambda } _ { 2 } ( \lambda _ { 2 } ) , \bar { \lambda } _ { 1 } ( \lambda _ { 2 } ) , \bar { \lambda } _ { 2 } ( \lambda _ { 1 } ) , \bar { \lambda } _ { 2 } ( \lambda _ { 2 } ) , \bar { \lambda } _ { 1 } ( \lambda _ { 2 } ) , \bar { \lambda } _ { 2 } ( \lambda _ { 1 } ) \right. } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad  \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad  \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \quad \quad \quad \quad \quad  \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}\tag{34}
$$

Here, (i) follows from $a ^ { q } \leq ( 1 - q ) + q a$ for $a \geq 0$ and $q \in [ 0 , 1 ] ; ( i i )$ follows from the reverse triangle inequality $\| A \| _ { * } \geq \| B \| _ { * } - \| B - A \| _ { * } ;$ and (iii) uses $\| A \| _ { * } \leq { \sqrt { r } } \| A \| _ { \mathsf { I } }$ where r bounds the rank of the error matrix $A = \nabla _ { ( l ) } F ( X _ { t } ) - N _ { t } ^ { ( l ) }$ , as stipulated in Theorem 3.1, and $\varsigma _ { \mathrm { m i n } } \leq \varsigma _ { \mathrm { m a x } }$

Taking expectations and summing the resulting descent inequality over $t = 1 , \dots , T$ , the intermediate terms telescope and give

$$
\begin{array} { r l } { \displaystyle \mathbb { E } [ F ( X _ { T + 1 } ) ] \leq F ( X _ { 1 } ) - \gamma \varsigma _ { \operatorname* { m i n } } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \| \nabla F _ { ( l ) } ( X _ { t } ) \| _ { * } } & { } \\ { \displaystyle + 2 \gamma \varsigma _ { \operatorname* { m a x } } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \mathbb { E } \| D _ { t } ^ { ( l ) } \| _ { \mathsf { F } } } & { } \\ { \displaystyle + \frac { \gamma ^ { 2 } \varsigma _ { \operatorname* { m a x } } ^ { 2 } } { 2 } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } r ^ { ( l ) } \mathbb { E } [ H _ { t } ] . } \end{array}\tag{35}
$$

Rearranging the terms and using Assumption 2.2, we obtain

$$
\begin{array} { r l } & { \displaystyle \frac { \varsigma _ { \mathrm { m i n } } } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \left[ \left\| \nabla F _ { ( l ) } ( X _ { t } ) \right\| _ { * } \right] - \frac { \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } q } { 2 T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } r ^ { ( l ) } \mathcal { L } _ { 1 } \mathbb { E } \left[ \left\| \nabla F ( X _ { t } ) \right\| _ { \mathsf { F } } \right] } \\ & { \displaystyle \leq \frac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + \frac { 2 \varsigma _ { \mathrm { m a x } } } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \mathbb { E } \left[ \left\| N _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } ) \right\| _ { \mathsf { F } } \right] + \frac { \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } } { 2 } \sum _ { l = 1 } ^ { L } r ^ { ( l ) } \hat { \mathcal { L } } , } \end{array}\tag{36}
$$

where $\hat { \mathcal { L } } : = \mathcal { L } _ { 0 } + ( 1 - q ) \mathcal { L } _ { 1 }$

Step 2: The Nesterov tracking-error expansion. We next control the discrepancy between the extrapolated momentum and the corresponding true gradient block. From the recursion $N _ { t } ^ { ( l ) } = \beta _ { 1 } \bar { N } _ { t - 1 } ^ { ( l ) } + ( 1 - \beta _ { 1 } \beta _ { 2 } ) G _ { t } ^ { ( l ) } - \beta _ { 1 } ( 1 - \beta _ { 2 } ) \bar { G } _ { t - 1 } ^ { ( l ) }$ , we have

$$
\begin{array} { r l } & { N _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } ) = \beta _ { 1 } \left( N _ { t - 1 } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t - 1 } ) \right) + ( 1 - \beta _ { 1 } \beta _ { 2 } ) \left( G _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } ) \right) } \\ & { \qquad - \beta _ { 1 } ( 1 - \beta _ { 2 } ) \left( G _ { t - 1 } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t - 1 } ) \right) + \beta _ { 1 } \beta _ { 2 } \left( \nabla F _ { ( l ) } ( X _ { t - 1 } ) - \nabla F _ { ( l ) } ( X _ { t } ) \right) } \end{array}\tag{37}
$$

Unrolling this recursion, together with the initialization $M _ { 0 } ^ { ( l ) } = 0$ , gives

$$
\begin{array} { r l } { { \displaystyle N _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } ) = - \beta _ { 1 } ^ { t } \beta _ { 2 } \nabla F _ { ( l ) } ( X _ { 1 } ) + ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( G _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } ) ) } } & { { } } \\ { { \displaystyle ~ + \beta _ { 2 } ( 1 - \beta _ { 1 } ) \sum _ { k = 1 } ^ { t - 1 } \beta _ { 1 } ^ { t - k } ( G _ { k } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { k } ) ) } } & { { } } \\ { { \displaystyle ~ + \beta _ { 1 } \beta _ { 2 } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } ( \nabla F _ { ( l ) } ( X _ { k - 1 } ) - \nabla F _ { ( l ) } ( X _ { k } ) ) , } } & { { } } \end{array}\tag{38}
$$

Taking norms, expectations, and then averaging over $t = 1 , \dots , T$ , we decompose the error into three terms: Write $\varepsilon _ { t } ^ { ( l ) } = G _ { t } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { t } )$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| D _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \leq \mathcal { T } _ { 1 } + \mathcal { T } _ { 2 } + \mathcal { T } _ { 3 } ,\tag{39}
$$

where the initialization, noise, and drift terms are, respectively,

$$
\begin{array} { r l } & { \mathcal { T } _ { 1 } = \displaystyle \frac { \beta _ { 2 } } { T } \sum _ { \ell = 1 } ^ { T } \phi _ { 1 } ^ { t } \| \nabla F _ { ( i ) } ( X _ { 1 } ) \| \mathrm { F } , } \\ & { \mathcal { T } _ { 2 } = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \bigg \| ( 1 - \beta _ { 1 } \beta _ { 2 } ) z _ { \ell } ^ { ( t ) } } \\ & { \qquad + \beta _ { 2 } \big ( 1 - \beta _ { 1 } \big ) \displaystyle \sum _ { k = 1 } ^ { t - 1 } \beta _ { 1 } ^ { t - k } \xi _ { k } ^ { ( t ) } \Big \| _ { \mathrm { F } } , } \\ & { \mathcal { T } _ { 3 } = \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \bigg \| \beta _ { 1 } \beta _ { 2 } \displaystyle \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } } \\ & { \qquad \cdot \big ( \nabla F _ { ( i ) } ( X _ { k - 1 } ) - \nabla F _ { ( i ) } ( X _ { k } ) \big ) \Big \| _ { \mathrm { F } } . } \end{array}
$$

Step 3: Initialization, noise, and gradient drift. The first term is controlled directly by the geometric series:

$$
\mathcal { T } _ { 1 } = \frac { \beta _ { 2 } } { T } \sum _ { t = 1 } ^ { T } \beta _ { 1 } ^ { t } \left. \nabla F _ { ( l ) } ( X _ { 1 } ) \right. _ { \mathsf { F } } \leq \frac { \beta _ { 2 } \left. \nabla F _ { ( l ) } ( X _ { 1 } ) \right. _ { \mathsf { F } } } { T ( 1 - \beta _ { 1 } ) } .\tag{40}
$$

For the stochastic-gradient noise term $\mathcal { T } _ { 2 }$ , Jensen’s inequality and the martingale-diference

property of the stochastic gradients yield

$$
\begin{array} { r l } { \gamma _ { - } - \frac { 1 } { L _ { 1 } } \displaystyle \sum _ { i = 1 } ^ { N } \| \boldsymbol { \hat { x } } - \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } ^ { i } \boldsymbol { x } ^ { i } \boldsymbol { x } ^ { i } \boldsymbol { x } ^ { i } | + \boldsymbol { \hat { x } } ( \boldsymbol { \hat { x } } - \boldsymbol { \hat { x } } ) \{ \boldsymbol { \hat { x } } - \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } ^ { i } \} \{ \boldsymbol { x } \} } \\ { \boldsymbol { \hat { x } } ^ { i } } & { = \frac { 1 } { L _ { 1 } } \displaystyle \sum _ { i = 1 } ^ { N } \| \boldsymbol { \hat { x } } \{ 1 - \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } ^ { i } \} ^ { i } \boldsymbol { x } ^ { i } \boldsymbol { x } ^ { i } \boldsymbol { x } ^ { i } | + \boldsymbol { \hat { x } } \{ 1 - \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } ^ { i } } { \boldsymbol { \hat { x } } ^ { i } \boldsymbol { x } ^ { i } \} ^ { 2 } \boldsymbol { x } ^ { i } } \\ { \boldsymbol { \hat { x } } ^ { i } } & { = \frac { 1 } { L _ { 1 } } \displaystyle \sum _ { i = 1 } ^ { N } \{ \boldsymbol { \hat { x } } ^ { i } ( \boldsymbol { \hat { x } } - \boldsymbol { \hat { x } } \boldsymbol { \hat { x } } ^ { i } ) ^ { 2 } \boldsymbol { \hat { x } } ^ { i } \} ^ { 1 / 2 } } \\ & { \qquad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \boldsymbol { \hat { x } } ^ { i } } &  = \frac { 1 } { L _ { 1 } } \displaystyle \sum _ { i = 1 } ^ { N } \{ \boldsymbol { \hat { x } } ^ { i } ( \boldsymbol { \hat { x } } \end{array}\tag{41}
$$

Here, (i) is Jensen’s inequality in the form $\mathbb { E } [ Z ] \le \sqrt { \mathbb { E } [ Z ^ { 2 } ] }$ ; (ii) uses conditional unbiasedness $\begin{array} { r } { \mathbb { E } [ G _ { k } ^ { ( l ) } - \nabla F _ { ( l ) } ( X _ { k } ) \ | \ \mathcal { F } _ { k - 1 } ] = 0 } \end{array}$ from Assumption 2.4, which eliminates the cross terms; (iii) applies the variance bound in Assumption 2.4; and $( i v )$ follows from ${ \sqrt { a + b } } \leq { \sqrt { a } } + { \sqrt { b } }$

It remains to bound the drift term $\tau _ { 3 }$ , which is caused by the change of the true gradient along the trajectory. Write $\mathcal { T } _ { 3 , l }$ for this term in layer l. Cauchy–Schwarz across layers and the triangle inequality give

$$
\begin{array} { r l } { \displaystyle \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \mathcal { T } _ { 3 , l } \leq \frac { \beta _ { 1 } \beta _ { 2 } \sqrt { R _ { 0 } } } { T } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } \mathbb { E } \| \nabla F ( X _ { k - 1 } ) - \nabla F ( X _ { k } ) \| _ { \mathbf { F } } } & { } \\ { \leq \frac { \gamma \beta _ { 1 } \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } R _ { 0 } } { T } \displaystyle \sum _ { t = 1 } ^ { T } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } \mathbb { E } \big [ \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } \| \nabla F ( X _ { k } ) \| _ { \mathbf { F } } ^ { q } \big ] } & { } \\ { \leq \frac { \gamma \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } R _ { 0 } } { 1 - \beta _ { 1 } } \left( \hat { \mathcal { L } } + \frac { q \mathcal { L } _ { 1 } } { T } \displaystyle \sum _ { k = 1 } ^ { T } \mathbb { E } \| \nabla F ( X _ { k } ) \| _ { \mathbf { F } } \right) . } \end{array}\tag{42}
$$

The second inequality applies Assumption 2.3 with base point $X _ { k }$ and uses $X _ { k } - X _ { k - 1 } = - \gamma O _ { k } .$ −1 and $\| O _ { k - 1 } \| _ { \mathsf { F } } \leq \mathsf { \varsigma } _ { \operatorname* { m a x } } \sqrt { R _ { 0 } }$ . The last inequality uses $z ^ { q } \leq ( 1 - q ) + q z$ , exchanges the finite sums, and bounds the geometric series by $( 1 - \beta _ { 1 } ) ^ { - 1 }$ . In particular, no one-layer bound is applied to a displacement in all layers.

Combining the bounds for $\mathcal { T } _ { 1 } , \mathcal { T } _ { 2 }$ , and $\tau _ { 3 }$ in Eq. (39), we obtain

$$
\begin{array} { r } { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \mathbb { E } \| N _ { t } ^ { ( l ) } - \nabla _ { ( l ) } F ( X _ { t } ) \| _ { \mathsf { F } } \leq \frac { \beta _ { 2 } G _ { 0 } } { T ( 1 - \beta _ { 1 } ) } + \left( \sqrt { 1 - \beta _ { 1 } } + \sqrt { 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } \right) S _ { 0 } } \\ { + \frac { \gamma \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } R _ { 0 } \hat { C } } { 1 - \beta _ { 1 } } + \frac { \gamma \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } R _ { 0 } q \hat { C } _ { 1 } } { 1 - \beta _ { 1 } } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } . } \end{array}\tag{43}
$$

Step 4: Substitution and absorption. Finally, substituting Eq. (43) into Eq. (36) and using $\begin{array} { r } { \| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } \leq \sum _ { l } \| \nabla _ { ( l ) } F ( X _ { t } ) \| } \end{array}$ <sub>∗</sub> and $\beta _ { 2 } \leq 1$ gives

$$
\begin{array} { r l } & { \quad \left( \displaystyle { \operatorname* { s m i n } - \frac { \gamma \zeta _ { \operatorname* { m a x } } ^ { 2 } q \mathcal L _ { 1 } R _ { 0 } } { 2 } - \frac { 2 \gamma \zeta _ { \operatorname* { m a x } } ^ { 2 } q \mathcal L _ { 1 } R _ { 0 } } { 1 - \beta _ { 1 } } } \right) \frac { 1 } { T } \sum _ { t = 1 } ^ { T } { \sum _ { l = 1 } ^ { L } \mathbb E \left[ \left\| \nabla F _ { ( l ) } ( X _ { t } ) \right\| _ { * } \right] } } \\ & { \le \frac { F \left( X _ { 1 } \right) - F ^ { * } } { \gamma T } + \frac { 2 \beta 2 \varsigma _ { \operatorname* { m a x } } } { T \left( 1 - \beta _ { 1 } \right) } \displaystyle \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \left\| \nabla F _ { ( l ) } ( X _ { 1 } ) \right\| _ { \mathrm { F } } } \\ & { \quad + 2 \left( \sqrt { 1 - \beta _ { 1 } } + \sqrt { 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } \right) \varsigma _ { \operatorname* { m a x } } \sum _ { l = 1 } ^ { L } \sqrt { r ^ { ( l ) } } \sigma _ { ( l ) } } \\ & { \quad + \frac { 2 \gamma \beta _ { 2 } \varsigma _ { \operatorname* { m a x } } ^ { 2 } \hat { L } \sum _ { l = 1 } ^ { L } r ^ { ( l ) } } { 1 - \beta _ { 1 } } + \frac { \gamma \varsigma _ { \operatorname* { m a x } } ^ { 2 } \hat { L } \sum _ { l = 1 } ^ { L } r ^ { ( l ) } } { 2 } , } \end{array}\tag{44}
$$

where $\begin{array} { r } { R _ { 0 } = \sum _ { l = 1 } ^ { L } r ^ { ( l ) } } \end{array}$ . This is precisely the claimed convergence bound. The use of $R _ { 0 }$ in the absorption coeficient accounts for simultaneous, coupled block updates. □

## C Proof of Corollary 3.2

Proof. Applying $\beta _ { 2 } = 1 - ( 1 - \beta _ { 1 } ) ^ { a }$ for $\begin{array} { r } { a > \frac { 1 } { 2 } } \end{array}$ , we have

$$
\sqrt { \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } = \sqrt { \beta _ { 1 } ( 1 - \beta _ { 1 } + \beta _ { 1 } ( 1 - \beta _ { 1 } ) ^ { a } ) ( 1 - \beta _ { 1 } ) ^ { a } } = \sqrt { \beta _ { 1 } ( 1 - \beta _ { 1 } ) ^ { 1 + a } + \beta _ { 1 } ^ { 2 } ( 1 - \beta _ { 1 } ) ^ { 2 a } + \beta _ { 2 } ^ { 2 } }\tag{45}
$$

We need the term $\sqrt { \beta _ { 1 } ( 1 - \beta _ { 1 } ) ^ { 1 + a } + \beta _ { 1 } ^ { 2 } ( 1 - \beta _ { 1 } ) ^ { 2 a } }$ to be of no larger order than $\sqrt { 1 - \beta _ { 1 } }$ so that the Nesterov momentum projection in Muon does not afect the final convergence rate. This is equivalent to $\beta _ { 1 } ( 1 - \beta _ { 1 } ) ^ { 1 + a } + \beta _ { 1 } ^ { 2 } ( 1 - \beta _ { 1 } ) ^ { 2 a } = O ( 1 - \beta _ { 1 } )$ , which holds for $\begin{array} { r } { a > \frac { 1 } { 2 } } \end{array}$

Since $\beta _ { 2 } \leq 1$ , we minimize the following upper envelope of the three leading terms in Eq. (44). By Young’s inequality,

$$
\begin{array} { r l } & { \cfrac { F ( X _ { 1 } ) - F ^ { * } } { \gamma T } + 2 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } + \cfrac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } { \hat { \mathcal { L } } } R _ { 0 } } { 1 - \beta _ { 1 } } } \\ & { \geq \bigg ( \cfrac { 4 ( F ( X _ { 1 } ) - F ^ { * } ) } { \gamma T } \bigg ) ^ { 1 / 4 } \cdot \Big ( 4 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } \Big ) ^ { 1 / 2 } \cdot \bigg ( \cfrac { 8 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } { \hat { \mathcal { L } } } R _ { 0 } } { 1 - \beta _ { 1 } } \bigg ) ^ { 1 / 4 } } \\ & { = \cfrac { 5 1 2 ^ { 1 / 4 } \varsigma _ { \mathrm { m a x } } { \hat { \mathcal { L } } } ^ { 1 / 4 } ( F ( X _ { 1 } ) - F ^ { * } ) ^ { 1 / 4 } S _ { 0 } ^ { 1 / 2 } R _ { 0 } ^ { 1 / 4 } } { T ^ { 1 / 4 } } . } \end{array}\tag{46}
$$

The equality condition is

$$
\frac { 4 ( F ( X _ { 1 } ) - F ^ { * } ) } { \gamma T } = 4 \sqrt { 1 - \beta _ { 1 } } \varsigma _ { \mathrm { m a x } } S _ { 0 } = \frac { 8 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } \hat { \mathcal { L } } R _ { 0 } } { 1 - \beta _ { 1 } } .\tag{47}
$$

Solving these two balancing equations gives

$$
\gamma = \frac { ( F ( X _ { 1 } ) - F ^ { * } ) ^ { 3 / 4 } } { 2 ^ { 1 / 4 } \varsigma _ { \operatorname* { m a x } } S _ { 0 } ^ { 1 / 2 } \hat { \mathcal { L } } ^ { 1 / 4 } R _ { 0 } ^ { 1 / 4 } T ^ { 3 / 4 } } , \quad \beta _ { 1 } = 1 - \frac { ( 2 \hat { \mathcal { L } } ( F ( X _ { 1 } ) - F ^ { * } ) R _ { 0 } ) ^ { 1 / 2 } } { S _ { 0 } T ^ { 1 / 2 } } .\tag{48}
$$

At this equality point, the minimized upper envelope is

$$
\frac { 5 1 2 ^ { 1 / 4 } \varsigma _ { \mathrm { m a x } } S _ { 0 } ^ { 1 / 2 } R _ { 0 } ^ { 1 / 4 } \hat { \mathcal { L } } ^ { 1 / 4 } ( F ( X _ { 1 } ) - F ^ { * } ) ^ { 1 / 4 } } { T ^ { 1 / 4 } } .\tag{49}
$$

Define $C _ { 1 } , C _ { 2 }$ as in Corollary 3.2. Choosing the horizon in (6), Lemma A.3 (or a direct check if $q \mathcal { L } _ { 1 } = 0 )$ gives

$$
\frac { \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { 2 } + \frac { 2 \gamma \varsigma _ { \mathrm { m a x } } ^ { 2 } q \mathcal { L } _ { 1 } R _ { 0 } } { 1 - \beta _ { 1 } } \leq \frac { \varsigma _ { \mathrm { m i n } } } { 2 } .\tag{50}
$$

Then, we reformulate Eq. (5) as

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } [ \| \nabla F ( \boldsymbol { \chi } _ { t } ) ( \boldsymbol { X } _ { t } ) \| _ { * } ] \leq \frac { 2 } { \operatorname* { S m a x } } \biggl ( \frac { 5 1 2 ^ { | \boldsymbol { \jmath } | } \boldsymbol { \mathcal { A } } _ { \operatorname { S m a x } } S _ { 0 } ^ { 1 / 2 } R _ { 0 } ^ { 1 / 4 } \hat { C } ^ { 1 / 4 } ( F ( \boldsymbol { X } _ { 1 } ) - F ^ { * } ) ^ { 1 / 4 } } { T ^ { 1 / 4 } } } \\ & { \qquad + \frac { 2 \operatorname* { S m a x } G _ { 0 } } { C _ { 2 } T ^ { 1 / 2 } } } \\ & { \qquad + 2 \sqrt { 2 } \varsigma _ { \operatorname* { m a x } } S _ { 0 } \biggl ( \frac { C _ { 2 } ^ { ( 1 + a ) / 2 } } { T ^ { ( 1 + a ) / 4 } } + \frac { C _ { 2 } ^ { a } } { T ^ { a / 2 } } \biggr ) } \\ & { \qquad + \frac { C _ { 1 } \varsigma _ { \operatorname* { m a x } } ^ { 2 } \hat { C } R _ { 0 } } { 2 T ^ { 3 / 4 } } \biggr ) . } \end{array}\tag{51}
$$

## D Proofs of the dimension-independent results

## D.1 Proof of Proposition 3.3

Proof. Each polynomial step preserves the representation in the original left and right singular-vector bases and applies $p _ { j }$ to its signed diagonal entries. The final entries are positive by (11), giving the displayed SVD formula for O. For $s _ { i } > 0$ , put $r _ { i } = \phi _ { J } ( s _ { i } ) / s _ { i }$ and $w _ { i } = s _ { i } ^ { 2 }$ , so $\textstyle \sum _ { i } w _ { i } = 1$ and $r _ { i } \in [ \ell _ { J } , u _ { J } ]$ . Direct calculation gives

$$
A = \sum _ { i } w _ { i } r _ { i } , \qquad B = \sum _ { i } w _ { i } r _ { i } ^ { 2 } , \qquad \langle N , O \rangle = \| N \| _ { \mathsf { F } } A .\tag{52}
$$

The energy estimates follow from $\ell _ { J } \leq r _ { i } \leq u _ { J }$ and $r _ { i } ^ { 2 } \le u _ { J } r _ { i }$ . The operator-norm estimate follows from the largest output singular value.

For the angular estimate, $( r _ { i } - \ell _ { J } ) ( u _ { J } - r _ { i } ) \geq 0$ implies

$$
B \leq ( u _ { J } + \ell _ { J } ) A - u _ { J } \ell _ { J } .
$$

Consequently,

$$
\frac { B } { A ^ { 2 } } \leq \frac { u _ { J } + \ell _ { J } } { A } - \frac { u _ { J } \ell _ { J } } { A ^ { 2 } } \leq \frac { ( u _ { J } + \ell _ { J } ) ^ { 2 } } { 4 u _ { J } \ell _ { J } } .\tag{53}
$$

The last expression is the maximum over $A > 0 ,$ , attained at $A = 2 u _ { J } \ell _ { J } / ( u _ { J } + \ell _ { J } )$ . This is the Kantorovich angle bound, derived here directly. Finally,

$$
\langle H , O \rangle = \langle N , O \rangle - \langle E , O \rangle \geq \| N \| _ { \mathsf { F } } A - \| E \| _ { \mathsf { F } } \sqrt { B } \geq A ( \| N \| _ { \mathsf { F } } - \bar { \rho } _ { J } \| E \| _ { \mathsf { F } } ) .
$$

The estimates needed by the convergence theorem also hold for $N = 0$ , where $O = 0$

## D.2 Analytic constants for the fixed quintic

Let $p ( s ) = a s + b s ^ { 3 } + c s ^ { 5 }$ , with the exact decimal coeficients in (16), and let $h ( z ) = a + b z + c z ^ { 2 }$ Completing the square gives

$$
h ( z ) \geq a - { \frac { b ^ { 2 } } { 4 c } } = 0 . 6 3 8 6 1 4 5 7 0 5 1 4 \ldots > 0 , \qquad z \geq 0 .\tag{54}
$$

For completeness, the two positive stationary points of $p$ are $\sqrt { z _ { - } }$ and $\sqrt { z _ { + } }$ , where

$$
z _ { \pm } = \frac { - 3 b \pm \sqrt { 9 b ^ { 2 } - 2 0 a c } } { 1 0 c } .
$$

Evaluating p at these points and at the endpoints proves the interval inclusions

$$
p ( [ 0 , 5 / 4 ] ) \subseteq [ 0 , 5 / 4 ] , \qquad p ( [ 1 7 / 2 5 , 5 / 4 ] ) \subseteq [ 1 7 / 2 5 , 5 / 4 ] .\tag{55}
$$

For example, the local maximum is between 1.2023 and 1.2024, the local minimum is between 0.6818 and 0.6819, $p ( 1 7 / 2 5 ) = 1 . 1 3 6 2 1 3 8 0 4 3 3 9 2$ , and $p ( 5 / 4 ) = 1 . 1 7 9 0 9 9 1 2 1 0 9 3 7 5$ . These loose rational brackets sufice for (55); no optimization of a high-degree composite polynomial is needed. Also, h is decreasing on $[ 0 , ( 1 7 / 2 5 ) ^ { 2 } ]$ and $h ( ( 1 7 / 2 5 ) ^ { 2 } ) > 1$ , so $p ( s ) \geq s$ for $0 \leq s \leq 1 7 / 2 5$

Consider an initial $s \in ( 0 , 1 ]$ . While an iterate is below $1 7 / 2 5 ,$ it cannot decrease. Once it reaches $[ 1 7 / 2 5 , 5 / 4 ]$ , it remains there by (55). If $s \geq 1 7 / 2 5$ , its later iterates are at least $1 7 / 2 5 \geq ( 1 7 / 2 5 ) s ;$ if $s < 1 7 / 2 5$ , they are at least s. Thus $\phi _ { J } ( s ) \geq ( 1 7 / 2 5 ) \varepsilon$ s for every $J \geq 1$ . All iterates lie in $[ 0 , 5 / 4 ]$ and on this interval $h ( s ^ { 2 } ) \leq a .$ , because $( 5 / 4 ) ^ { 2 } < - b / c$ . It follows that

$$
{ \frac { \phi _ { J } ( s ) } { s } } = \prod _ { j = 0 } ^ { J - 1 } h ( \phi _ { j } ( s ) ^ { 2 } ) \leq a ^ { J } .
$$

The limit at $s \downarrow 0$ is $a ^ { J }$ , proving $u _ { J } = a ^ { J }$ . This proves all of (16). The general theorem permits diferent coeficients at diferent steps; for such schedules positivity and reachable intervals must be checked for that schedule rather than inferred from this fixed-coeficient example.

## D.3 Proof of Theorem 3.4

Proof. We follow the four steps of the proof of Theorem 3.1 in Appendix B. Only the alignment, update-energy, and joint-drift bounds change. Write

$$
E _ { t } ^ { ( l ) } = N _ { t } ^ { ( l ) } - \nabla _ { ( l ) } F ( X _ { t } ) , \qquad H _ { t } = \sum _ { l = 1 } ^ { L } \| \nabla _ { ( l ) } F ( X _ { t } ) \| _ { \mathsf { F } } , \qquad b _ { J } = \ell _ { J } + u _ { J } ,
$$

and set $\begin{array} { r } { \bar { H } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \mathbb { E } H _ { t } } \end{array}$ and $\begin{array} { r } { \bar { E } = T ^ { - 1 } \sum _ { t = 1 } ^ { T } \sum _ { l = 1 } ^ { L } \mathbb { E } \| E _ { t } ^ { ( l ) } \| _ { \mathsf { F } } } \end{array}$

Step 1: Descent and telescoping. Proposition 3.3 gives $\langle N _ { t } ^ { ( l ) } , O _ { t } ^ { ( l ) } \rangle \geq \ell _ { J } \Vert N _ { t } ^ { ( l ) } \Vert _ { \mathsf { F } }$ and $\| O _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \leq u _ { J }$ Splitting the true gradient into $N _ { t } ^ { ( l ) } - E _ { t } ^ { ( l ) }$ and applying the reverse triangle inequality, just as in the nuclear-norm proof, yields

$$
\begin{array} { r l } & { \langle \nabla _ { ( l ) } F ( X _ { t } ) , O _ { t } ^ { ( l ) } \rangle \geq \ell _ { J } \| N _ { t } ^ { ( l ) } \| _ { \mathsf { F } } - u _ { J } \| E _ { t } ^ { ( l ) } \| _ { \mathsf { F } } } \\ & { \qquad \geq \ell _ { J } \| \nabla _ { ( l ) } F ( X _ { t } ) \| _ { \mathsf { F } } - b _ { J } \| E _ { t } ^ { ( l ) } \| _ { \mathsf { F } } . } \end{array}\tag{56}
$$

These inequalities also hold when $N _ { t } ^ { ( l ) } = 0$ and $O _ { t } ^ { ( l ) } = 0$ . Moreover, $\begin{array} { r } { \sum _ { l } \| O _ { t } ^ { ( l ) } \| _ { \mathsf { F } } ^ { 2 } \leq L u _ { J } ^ { 2 } } \end{array}$ . Lemma A.1, $z ^ { q } \leq ( 1 - q ) + q z .$ , and $\| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } \leq H _ { t }$ therefore imply

$$
F ( X _ { t + 1 } ) \leq F ( X _ { t } ) - \gamma \ell _ { J } H _ { t } + \gamma b _ { J } \sum _ { l } \| E _ { t } ^ { ( l ) } \| _ { \mathsf { F } } + \frac { \gamma ^ { 2 } L u _ { J } ^ { 2 } } { 2 } \big ( \hat { \mathcal { L } } + q \mathcal { L } _ { 1 } H _ { t } \big ) .
$$

Taking expectations, summing over t, and using $F ( X _ { T + 1 } ) \geq F ^ { * }$ gives

$$
\left( \ell _ { J } - \frac { \gamma L u _ { J } ^ { 2 } q \mathcal L _ { 1 } } { 2 } \right) \bar { H } \leq \frac { D } { \gamma T } + b _ { J } \bar { E } + \frac { \gamma L u _ { J } ^ { 2 } \hat { \mathcal L } } { 2 } .\tag{57}
$$

This is the counterpart of (36), with Frobenius stationarity and an unweighted sum of block tracking errors.

Step 2: The Nesterov tracking-error expansion. Put $\varepsilon _ { t } ^ { ( l ) } = G _ { t } ^ { ( l ) } - \nabla _ { ( l ) } F ( X _ { t } )$ . The momentum recursion and $M _ { 0 } ^ { ( l ) } = 0$ give exactly the expansion (38), which we write as

$$
E _ { t } ^ { ( l ) } = - \beta _ { 1 } ^ { t } \beta _ { 2 } \nabla _ { ( l ) } F ( X _ { 1 } ) + Z _ { t } ^ { ( l ) } + V _ { t } ^ { ( l ) } ,
$$

where

$$
Z _ { t } ^ { ( l ) } = ( 1 - \beta _ { 1 } \beta _ { 2 } ) \varepsilon _ { t } ^ { ( l ) } + \beta _ { 2 } \delta \sum _ { k = 1 } ^ { t - 1 } \beta _ { 1 } ^ { t - k } \varepsilon _ { k } ^ { ( l ) } ,
$$

$$
{ V } _ { t } ^ { ( l ) } = \beta _ { 1 } \beta _ { 2 } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } \big ( \nabla _ { ( l ) } F ( X _ { k - 1 } ) - \nabla _ { ( l ) } F ( X _ { k } ) \big ) .
$$

Thus the same three terms as in (39) satisfy

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \| E _ { t } ^ { ( l ) } \| _ { \mathsf { F } } \leq \mathcal { T } _ { 1 , l } + \mathcal { T } _ { 2 , l } + \mathcal { T } _ { 3 , l } ,
$$

where $\mathcal { T } _ { 1 , l }$ is the averaged norm of the initialization term, $\begin{array} { r } { \mathcal { T } _ { 2 , l } = T ^ { - 1 } \sum _ { t } \mathbb { E } \| Z _ { t } ^ { ( l ) } \| _ { \mathsf { F } } } \end{array}$ , and $\mathcal { T } _ { 3 , l } =$ $\begin{array} { r } { T ^ { - 1 } \sum _ { t } \mathbb { E } \| V _ { t } ^ { ( l ) } \| _ { \mathsf { F } } } \end{array}$

Step 3: Initialization, noise, and gradient drift. The initialization bound uses the same geometric series:

$$
\sum _ { l } \mathcal { T } _ { 1 , l } = \frac { \beta _ { 2 } } { T } \sum _ { t = 1 } ^ { T } \beta _ { 1 } ^ { t } G _ { \mathsf { F } } \leq \frac { \beta _ { 2 } G _ { \mathsf { F } } } { T \delta } .
$$

For the noise term, conditional unbiasedness eliminates temporal cross terms within each block. The variance bound and geometric series give

$$
\begin{array} { r l } & { \mathbb { E } \Vert Z _ { t } ^ { ( l ) } \Vert _ { \mathsf { F } } ^ { 2 } \leq \left( ( 1 - \beta _ { 1 } \beta _ { 2 } ) ^ { 2 } + \beta _ { 2 } ^ { 2 } \delta ^ { 2 } \displaystyle \sum _ { k = 1 } ^ { t - 1 } \beta _ { 1 } ^ { 2 ( t - k ) } \right) \sigma _ { ( l ) } ^ { 2 } } \\ & { \qquad \leq \frac { \delta + 2 \beta _ { 1 } ( 1 - \beta _ { 1 } \beta _ { 2 } ) ( 1 - \beta _ { 2 } ) } { 1 + \beta _ { 1 } } \sigma _ { ( l ) } ^ { 2 } . } \end{array}
$$

Jensen’s inequality and ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ imply $\Sigma _ { l } \mathcal { T } _ { 2 , l } \leq v _ { \beta } S _ { \mathsf { F } }$ . This estimate does not require independence across layers.

For drift, apply Cauchy–Schwarz across blocks before invoking joint smoothness. In place of the rank-weighted bound in (42), use $\begin{array} { r } { \sum _ { l } \| W ^ { ( l ) } \| _ { \mathsf F } \leq \sqrt { L } \| W \| _ { \mathsf F } } \end{array}$ and $\| X _ { k } - X _ { k - 1 } \| _ { \mathsf { F } } \leq \gamma u _ { J } \sqrt { L }$ :

$$
\begin{array} { r l } {  { \sum _ { l } \mathcal { T } _ { 3 , l } \leq \frac { \beta _ { 1 } \beta _ { 2 } \sqrt { L } } { T } \sum _ { t = 1 } ^ { T } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } \mathbb { E } \| \nabla F ( X _ { k - 1 } ) - \nabla F ( X _ { k } ) \| \mathrm { r } } } \\ & { \leq \frac { \gamma \beta _ { 1 } \beta _ { 2 } L u _ { J } } { T } \sum _ { t = 1 } ^ { T } \sum _ { k = 2 } ^ { t } \beta _ { 1 } ^ { t - k } \mathbb { E } \big [ \hat { \mathcal { L } } + q \mathcal { L } _ { 1 } H _ { k } \big ] } \\ & { \leq \frac { \gamma \beta _ { 2 } L u _ { J } } { \delta } \big ( \hat { \mathcal { L } } + q \mathcal { L } _ { 1 } \bar { H } \big ) . } \end{array}\tag{58}
$$

The second inequality uses Assumption 2.3 with base point $X _ { k }$ and $X _ { k } - X _ { k - 1 } = - \gamma O _ { k - 1 }$ . The last exchanges the finite sums and bounds $\begin{array} { r } { \beta _ { 1 } \sum _ { t = k } ^ { T } \beta _ { 1 } ^ { t - k } \le \delta ^ { - 1 } } \end{array}$ . Combining the three terms gives the counterpart of (43):

$$
\bar { E } \leq \frac { \beta _ { 2 } G _ { \mathsf { F } } } { T \delta } + v _ { \beta } S _ { \mathsf { F } } + \frac { \gamma \beta _ { 2 } L u _ { J } \hat { \mathcal { L } } } { \delta } + \frac { \gamma \beta _ { 2 } L u _ { J } q \mathcal { L } _ { 1 } } { \delta } \bar { H } .\tag{59}
$$

Step 4: Substitution and absorption. Substituting (59) into (57) and moving its last term to the left gives

$$
\begin{array} { r l } & { \left( \ell _ { J } - \frac { \gamma L u _ { J } ^ { 2 } q \mathcal L _ { 1 } } { 2 } - \frac { \gamma \beta _ { 2 } L u _ { J } b _ { J } q \mathcal L _ { 1 } } { \delta } \right) \bar { H } } \\ & { \qquad \leq \displaystyle \frac { D } { \gamma T } + \frac { \beta _ { 2 } b _ { J } G _ { \mathsf { F } } } { T \delta } + b _ { J } v _ { \beta } S _ { \mathsf { F } } + \frac { \gamma \beta _ { 2 } L u _ { J } b _ { J } \hat { \mathcal L } } { \delta } + \frac { \gamma L u _ { J } ^ { 2 } \hat { \mathcal L } } { 2 } . } \end{array}
$$

Using $\beta _ { 2 } \leq 1$ in the coeficient on the left and $\bar { H } \geq 0$ yields (19). This completes the same descent, tracking, and absorption argument as for Theorem 3.1, with the finite-map bounds supplying all geometric constants. □

## D.4 Proof of Corollary 3.5

Proof. As in the proof of Corollary 3.2, the schedules balance the three leading terms of the convergence bound. With $b _ { J } = \ell _ { J } + u _ { J }$ , the corresponding envelope is

$$
\Phi _ { \mathsf { F } } ( \gamma , \delta ) = \frac { D } { \gamma T } + b _ { J } S _ { \mathsf { F } } \sqrt { \delta } + \frac { \gamma L u _ { J } b _ { J } \hat { \mathcal { L } } } { \delta } .
$$

For $D , S _ { \mathsf { F } } > 0$ , weighted Young’s inequality gives

$$
\Phi _ { \mathsf { F } } ( \gamma , \delta ) \geq \left( \frac { 6 4 D ( b _ { J } S _ { \mathsf { F } } ) ^ { 2 } L u _ { J } b _ { J } \hat { \mathscr { L } } } { T } \right) ^ { 1 / 4 } ,
$$

with equality when

$$
\frac { 4 D } { \gamma T } = 2 b _ { J } S _ { \mathsf { F } } \sqrt { \delta } = \frac { 4 \gamma L u _ { J } b _ { J } \hat { \mathcal { L } } } { \delta } .
$$

These balancing equations give $\gamma \asymp T ^ { - 3 / 4 }$ and $\delta \asymp T ^ { - 1 / 2 }$ . The corollary allows any positive $\gamma _ { 0 } , \delta _ { 0 }$ with these exponents; the verification below also covers $D = 0$ or $S _ { \mathsf { F } } = 0$ without using the equality conditions.

Under (20), $\delta = \delta _ { 0 } T ^ { - 1 / 2 }$ and

$$
1 - \beta _ { 1 } \beta _ { 2 } = \delta + \beta _ { 1 } \delta ^ { a } , \qquad v _ { \beta } \le \sqrt { \delta } + \sqrt { 2 } \bigl ( \delta ^ { ( 1 + a ) / 2 } + \delta ^ { a } \bigr ) .
$$

The condition $\Delta _ { \gamma , \beta _ { 1 } } ^ { \mathsf { F } } \geq \ell J / 2$ follows from $A _ { J } T ^ { - 1 / 4 } + B _ { J } T ^ { - 3 / 4 } \leq 1$ . The suficient threshold (23) follows by the same elementary argument as Lemma $\mathrm { { A . 3 } ; }$ if $A _ { J } = B _ { J } = 0$ there is nothing to check. Substitution into (19), followed by division by $\ell _ { J } / 2 ,$ , proves (21). For $a > 1 / 2$ , all exponents except the leading $1 / 4$ are strictly larger than $1 / 4$ . Finally $\| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } \leq H _ { t }$ , so the Euclidean/Frobenius stationarity claim follows directly.

To prove (24), put $g \ : = \ : \| \nabla F ( X _ { 1 } ) \| _ { \mathsf F }$ and $L _ { g } = \mathcal { L } _ { 0 } + \mathcal { L } _ { 1 } g ^ { q } > 0$ . Applying Lemma A.1 to $Y = X _ { 1 } - \nabla F ( X _ { 1 } ) / L _ { g }$ and using $F ( Y ) \ge F ^ { * }$ gives $g ^ { 2 } \leq 2 D L _ { g } \leq 2 D ( \hat { \mathscr { L } } + q \mathscr { L } _ { 1 } g )$ . Solving this quadratic inequality and using $G _ { \mathsf { F } } \leq { \sqrt { L } } g$ proves the claim. This argument concerns the stated optimization assumptions, not a network-width limit. □

## D.5 Scope of the angular refinement

For one nonzero block momentum N with true gradient H and error $E = N - H$ , (15) gives the valid implication

$$
\| N \| _ { \mathsf { F } } \geq 2 \bar { \rho } _ { J } \| E \| _ { \mathsf { F } } \quad \Longrightarrow \quad \langle H , O \rangle \geq \frac { 1 } { 2 } \ell _ { J } \| N \| _ { \mathsf { F } } .
$$

If the displayed suficient condition fails, the triangle inequality gives

$$
\| H \| _ { \mathsf { F } } < ( 2 \bar { \rho } _ { J } + 1 ) \| E \| _ { \mathsf { F } } .
$$

Failure of this test is not equivalent to actual negative alignment; it only means that this suficient lower bound is unavailable. Nor does smallness of one block gradient imply stationarity of the complete parameter tuple.

To make precise the stronger small-constant argument, consider the single-block case and separately assume spectral-direction smoothness

$$
F ( X + V ) \leq F ( X ) + \langle \nabla F ( X ) , V \rangle + \frac { \mathcal { L } _ { \mathrm { s p } } } { 2 } \| V \| _ { \mathrm { o p } } ^ { 2 } .\tag{60}
$$

This is an additional hypothesis, not a consequence of Assumption 2.3 with a dimension-independent constant. Define $h _ { \gamma } = \mathcal { L } _ { \mathrm { s p } } \gamma c _ { J } ^ { 2 } / \ell _ { J } . \mathrm { ~ I f ~ } \| N _ { t } \| _ { \mathsf { F } } \geq 2 \bar { \rho } _ { J } \| E _ { t } \| _ { \mathsf { F } } + h _ { \gamma }$ , (15) and (60) yield

$$
\begin{array} { r } { F ( X _ { t + 1 } ) \leq F ( X _ { t } ) - \frac { 1 } { 2 } \gamma \ell _ { J } \| N _ { t } \| _ { \mathsf F } . } \end{array}
$$

On any sample path for which ma $\mathrm { x } _ { t \leq T } \| E _ { t } \| _ { \mathsf { F } } \leq \bar { e } .$ , either this condition first fails, or it holds at every step. In the first case the gradient at the failed test is at most $( 2 \bar { \rho } _ { J } + 1 ) \bar { e } + h _ { \gamma } ;$ in the second case telescoping gives min $\| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } \leq 2 D / ( \gamma \ell _ { J } T ) + \bar { e }$ . Therefore the precise deterministic certificate is

$$
\operatorname* { m i n } _ { t \leq T } \| \nabla F ( X _ { t } ) \| _ { \mathsf { F } } \leq \frac { 2 D } { \gamma \ell _ { J } T } + ( 2 \bar { \rho } _ { J } + 1 ) \bar { e } + \frac { \mathcal { L } _ { \mathrm { s p } } \gamma c _ { J } ^ { 2 } } { \ell _ { J } } .\tag{61}
$$

An averaged or pointwise expected error bound is not a bound on ma $\mathrm { x } _ { t } \parallel E _ { t } \parallel _ { \mathsf { F } }$ , and it cannot simply be substituted for e¯. Initialization transients must also be included in any such uniform bound. Equation (61) is thus a conditional geometric certificate, not a second stochastic convergence claim under the three stated assumptions.

The size of $u J$ cannot in general be removed from a uniform Frobenius update bound by sharpening a scalar inequality. Let $A _ { 0 } = \phi _ { J } ^ { \prime } ( 0 ) > 0$ and take rank r with all normalized singular values $s _ { i } = r ^ { - 1 / 2 }$ . Then

$$
A ( N ) = \sqrt { r } \phi _ { J } ( r ^ { - 1 / 2 } ) \longrightarrow A _ { 0 } , \qquad B ( N ) = r \phi _ { J } ( r ^ { - 1 / 2 } ) ^ { 2 } \longrightarrow A _ { 0 } ^ { 2 } .
$$

Thus any uniform bound $\| O \| _ { \mathsf { F } } \leq U$ needs $U \geq A _ { 0 }$ , and any $B \le K A$ needs $K \geq A _ { 0 }$ . For the fixed quintic, $A _ { 0 } = u _ { J }$ . The large universal magnitude constant and the absence of dimension factors are distinct issues.