# Geometric Kernel Diversity: An Axiomatic Diversity Measure in Euclidean Space

Ditbul Ban   
MIT CSAIL   
firefly7@mit.edu

Younghun Roh MIT CSAIL yhunroh@mit.edu

Julian Shun MIT CSAIL jshun@mit.edu

Vihan Lakshman MIT CSAIL vihan@mit.edu

## Abstract

Quantifying the diversity of a set of objects is a fundamental problem in machine learning and statistics. Mironov and Prokhorenkova (2025) recently showed that three axioms— monotonicity, uniqueness, and continuity— are crucial for a reliable diversity measure and that none of the standard metrics in the literature satisfy all three. They also constructed two measures that do satisfy the axioms, but both are NP-hard to compute, leaving open the question of whether eficient axiomatic diversity measures can exist. We study a Euclidean-restricted version of this question. We introduce Geometric Kernel Diversity (GKD), which associates a radial kernel solid with each point and measures the volume of their union. We show that canonical GKD satisfies all three Mironov–Prokhorenkova axioms on Euclidean-realizable inputs. For canonical GKD, we give exact polynomial-time algorithms in dimensions $d \ \leq \ 3 .$ In arbitrary dimensions, we give a randomized algorithm that returns a $( 1 \pm \varepsilon )$ -approximation with probability at least $1 ~ - ~ \delta$ in time $\begin{array} { r } { O \left( { \frac { n d } { \varepsilon ^ { 2 } } } \log { \frac { 2 } { \delta } } \right) } \end{array}$ for n points. The axiomatic guarantee applies to exact canonical GKD; the randomized algorithm approximates this deterministic quantity. We further show that the two diversity measures of Mironov and Prokhorenkova (2025) remain NP-hard on Euclidean-realizable inputs and, unless NP = RP, admit no FPRAS even on this restricted domain.

## 1 INTRODUCTION

Measuring diversity is a fundamental problem in machine learning and statistics, from evaluating generative samples to quantifying dataset coverage and redundancy. Despite the ubiquity of this task, commonly used diversity measures can behave unintuitively. Mironov and Prokhorenkova (2025) proposed three axioms—monotonicity under nontrivial increases of pairwise distances, uniqueness relative to exact duplicates, and continuity—and showed that none of the common diversity measures in the literature satisfy all three. They also constructed two measures that do satisfy all three axioms, but both are NP-hard to compute, leaving open the question of whether an axiomatic diversity function can be computationally tractable.

We study a Euclidean-restricted version of this question, motivated in part by settings where objects are represented by learned embeddings. We introduce Geometric Kernel Diversity (GKD): each point induces the under-graph solid of a normalized radial kernel, and GKD is the volume of their union, equivalently the integral of the kernel value determined by the nearest point. This gives a geometric efective-number notion ranging from complete redundancy toward the support size as the points become widely separated.

For a broad class of kernels, we prove strict monotonicity under pairwise-distance expansions; in particular, our canonical Student-t<sub>2</sub> kernel satisfies monotonicity, uniqueness, and continuity on Euclidean-realizable distance matrices. We then ask whether this axiomatic diversity function can also be computed eficiently. We show that canonical GKD is exactly computable in polynomial time for d ≤ 3 using Voronoi-based algorithms, and in arbitrary dimensions admits a randomized (1 ± ε)-approximation with probability at least 1 − δ in $\begin{array} { r } { O \big ( \frac { n \bar { d } } { \varepsilon ^ { 2 } } \operatorname { l o g } \frac { 2 } { \delta } \big ) } \end{array}$ time for n points in our randomized real-RAM model. The axiomatic guarantee applies to exact canonical GKD; the randomized algorithm approximates this deterministic quantity rather than defining a new diversity measure.

In summary, our main contributions are as follows:

• We introduce Geometric Kernel Diversity (GKD), a new geometric diversity measure in Euclidean space based on the union volume of radial-kernel solids. For a broad class of radial kernels, we prove that GKD satisfies the monotonicity, uniqueness, and continuity axioms of Mironov and Prokhorenkova (2025) on Euclidean-realizable inputs in every dimension.

• We provide, to our knowledge, the first positive answer to the open problem posed by Mironov and Prokhorenkova (2025) in the restricted setting of low-dimensional Euclidean space: for the canonical GKD kernel, we give exact polynomial-time algorithms in dimensions $d \leq 3 .$ . The unrestricted Mironov–Prokhorenkova problem remains open.

• We give a randomized relative-error approximation algorithm for canonical GKD in arbitrary dimensions that returns a (1 ± ε)-approximation with probability at least 1 − δ in time $O \left( \textstyle { \frac { n d } { \varepsilon ^ { 2 } } } \log { \frac { 2 } { \delta } } \right)$ for n points. In contrast, we show that the two diversity measures constructed by Mironov and Prokhorenkova (2025) remain NP-hard even on Euclidean-realizable inputs and, unless NP = RP, admit no FPRAS on this restricted domain. Thus, our results provide the first eficiently computable randomized approximation scheme for an axiomatic diversity measure on Euclideanrealizable inputs.

## 2 RELATED WORK

Axiomatic Diversity. Mironov and Prokhorenkova (2025) study diversity functions defined on general symmetric nonnegative distance matrices, without requiring the triangle inequality or Euclidean realizability. They propose three desirable properties. Monotonicity requires diversity to strictly increase when no pairwise distance decreases and at least one increases. Uniqueness requires diversity to strictly decrease when a distinct object is replaced by a dupli cate of another object. Continuity requires diversity to vary continuously with the pairwise distances. We provide the full formal definitions of these three axioms in Appendix A.6. Mironov and Prokhorenkova (2025) show that none of the measures they consider satisfies all three axioms and construct MultiDimVolume and IntegralMaxClique, which satisfy all three axioms but are NP-hard to compute. We instead restrict the domain to Euclidean-realizable distance matrices. This restriction alone does not make their two diversity functions tractable: we show that both remain NP-hard on Euclidean-realizable inputs and that, unless $\mathrm { N P } ~ = ~ \mathrm { R P }$ , neither admits an FPRAS on this domain (Appendix E). Moreover, the restriction to the Euclidean domain does not enable existing diversity measures such as the Vendi Score (Friedman and Dieng, 2023), Determinantal Point Processes (DPP) (Wilhelm et al., 2018), and R´enyi Kernel Entropy (RKE) (Jalali et al., 2023) to satisfy the axioms (Section 6.2).

Geometric and volume-based diversity. GKD lies within a broader family of geometric upperenvelope constructions. Its Gaussian specialization coincides, up to normalization and scale, with the Gaussian functionals studied by Mourtada (2025) and Cˆot´e (2025), while our definition allows a broader class of radial kernels. At the other extreme, GKD is the equalweight, all-maximum specialization of the more general flower-weight framework of Bezdek and Connelly (2008), which allows more general combinations of radial weight functions. These works arise in diferent contexts: in particular, the flower-weight framework is developed in connection with the Kneser-Poulsen problem rather than axiomatic diversity.

Our focus is the Mironov–Prokhorenkova axiomatic question on Euclidean-realizable inputs. Building on the Gaussian comparison theorem of Mourtada (2025), the Bernstein-Widder representation (Widder, 1931), and the large-radius strictness theorem of Gorbovickis (2012), we establish strict monotonicity under arbitrary pairwise-distance expansions for a broader kernel class, a novel result not addressed by prior work. Furthermore, we go beyond prior work in volume-based geometric diversity by designing eficient algorithms. We identify a canonical Student-t kernel and give exact algorithms in dimensions $d \leq 3$ together with a randomized relative-error approximation in arbitrary dimensions.

Metric-space magnitude. Limbeck et al. (2024) propose magnitude-based measures for evaluating the diversity of latent representations. For a finite metric space, magnitude is interpreted as an efective number of points across multiple distance scales, and their MagArea measure aggregates this information over scales. This motivation is closely related to GKD’s efective number interpretation, but the theoretical frameworks difer. They emphasize, among other properties, that adding a genuinely new observation should not decrease diversity and that duplicate observations should be redundant. They also discuss monotonicity under increases in pairwise distances, but only in a nonstrict form and without establishing it in full generality. In contrast, we study the Mironov–Prokhorenkova axioms, which require strict distance monotonicity and explicitly compare configurations containing duplicates, and establish all three axioms on Euclideanrealizable inputs.

Union-volume approximation. Karp et al. (1989) give the self-adjusting coverage estimator for unions of finite sets, and Bringmann and Friedrich (2010) develop an oracle-based framework for geometric unionvolume approximation. Canonical GKD solids have known unit volume and admit O(d)-time exact sampling and membership oracles, enabling the arbitrarydimensional estimator of Section 5; our exact lowdimensional algorithms additionally exploit Voronoi and divergence structure.

## 3 GEOMETRIC KERNEL DIVERSITY

Throughout the paper, $\| \cdot \|$ denotes the Euclidean norm, $B _ { d } ( { \pmb x } , r )$ the Euclidean ball of radius r centered at ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ and $\mathrm { V o l } _ { k }$ the k-dimensional Lebesgue measure. We write $s d \sb { - 1 }$ for the surface area of the unit sphere in $\mathbb { R } ^ { d }$ . For a tuple $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )$ , possibly containing duplicate points, define supp(X) to be the set of distinct points appearing in X.

## 3.1 Definition and Under-Graph Representation

We begin with a radial kernel that quantifies how strongly a point covers the surrounding space.

Definition 3.1 (d-admissible radial kernel). A function $\phi : [ 0 , \infty )  [ 0 , \infty )$ is a d-admissible radial kernel if it is continuous, strictly decreasing, and

$$
0 < \int _ { 0 } ^ { \infty } r ^ { d - 1 } \phi ( r ) d r < \infty .
$$

Its kernel mass is

$$
Z _ { \phi , d } : = \int _ { \mathbb { R } ^ { d } } \phi ( \| z \| ) d z = s _ { d - 1 } \int _ { 0 } ^ { \infty } r ^ { d - 1 } \phi ( r ) d r .
$$

We call $\phi$ normalized if $Z _ { \phi , d } = 1$ , and assume normalization unless stated otherwise.

For a nonempty tuple $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ , define

$$
d _ { X } ( z ) : = \operatorname* { m i n } _ { 1 \leq i \leq n } \| z - \pmb { x } _ { i } \| .
$$

Definition 3.2 (Geometric Kernel Diversity). For a normalized d-admissible radial kernel ϕ, define

$$
{ \mathcal { D } } _ { \phi , d } ( X ) : = \int _ { \mathbb { R } ^ { d } } \phi ( d _ { X } ( z ) ) d z .
$$

Since $\phi$ is strictly decreasing,

$$
\phi ( d _ { X } ( z ) ) = \operatorname* { m a x } _ { \pmb { x } \in \mathrm { s u p p } ( X ) } \phi ( \| z - \pmb { x } \| ) .
$$

For $\pmb { x } \in \mathbb { R } ^ { d }$ , define its under-graph solid by

$$
S _ { \phi , d } ( x ) : = \big \{ ( z , u ) \in \mathbb { R } ^ { d } \times \mathbb { R } _ { \geq 0 } \ | \ 0 \leq u \leq \phi ( \| z - x \| ) \big \} .
$$

(a) Single-point solid  
(b) Union volume  
![](images/579ef4f0a593b1b64b81425e554400f70acc4db23c90fe90801e51de060de1a7.jpg)

![](images/33157ca46da5ecda68000acea8f63671175fc686b93562fd15bb1b2ca707aac6.jpg)  
Figure 1: Union-volume interpretation of GKD in $\mathbb { R } ^ { 2 }$ . (a) A point $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ induces the under-graph solid $S _ { \phi , 2 } ( \pmb { x } _ { i } )$ . (b) For a configuration $X = \{ { \pmb x } _ { 1 } , { \pmb x } _ { 2 } , { \pmb x } _ { 3 } \}$ GKD is the volume of the union of the corresponding solids, whose upper boundary is $u = \phi ( d _ { X } ( z ) )$ .

Proposition 3.3 (Under-graph representation and well-definedness). For every nonempty $X \in ( \mathbb { R } ^ { d } ) ^ { n }$

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = { \mathrm { V o l } } _ { d + 1 } \left( \bigcup _ { \pmb { x } \in \mathrm { s u p p } ( X ) } S _ { \phi , d } ( \pmb { x } ) \right) .
$$

Moreover, every single-point solid has unit volume:

$$
\mathrm { V o l } _ { d + 1 } ( S _ { \phi , d } ( { \pmb x } ) ) = 1 .
$$

In particular, $\mathcal { D } _ { \phi , d } ( X ) < \infty$

This union-volume viewpoint (Figure 1) will drive both the axiomatic analysis and the computational al gorithms below.

## 3.2 Basic Properties

The union-volume representation immediately yields several structural properties of GKD.

Theorem 3.4 (Basic and Coverage Properties). Let $d \ge 1$ and let $\phi$ be a normalized d-admissible radial kernel. Then, $\mathcal { D } _ { \phi , d }$ is permutation invariant, duplicate invariant, continuous in the input points, and invariant under Euclidean isometries. Moreover, adding any point outside the current support strictly increases GKD. With the extension $\mathcal { D } _ { \phi , d } ( \mathcal { O } ) = 0$ , its restriction to finite subsets of $\mathbb { R } ^ { d }$ is submodular.

These properties are natural from the union-volume viewpoint. Permuting the input points or adding a duplicate leaves the same collection of distinct solids. Adding a genuinely new point introduces a solid with a region of positive volume not covered by the existing ones, so the union volume strictly increases. Small perturbations of the input points change the corresponding solids continuously, while Euclidean isometries preserve their union volume. Finally, submodu larity expresses the usual diminishing-returns behavior of coverage: as more solids are already present, the additional volume contributed by a new solid can only decrease. The proof is given in Appendix A.2.

Normalization also gives GKD a natural efectivenumber interpretation, making its numerical scale directly comparable to the number of distinct points in the configuration. If $m = | \operatorname { s u p p } ( X ) |$ |, then ${ \mathcal { D } } _ { \phi , d } ( X ) =$ 1 if and only if $m = 1$ ; for $m \geq 2$

$$
1 < { \mathcal { D } } _ { \phi , d } ( X ) < m ,
$$

and the supremum over configurations with m distinct points is m, approached as the points become widely separated. Thus, duplicate points behave like a single efective point, while suficiently separated points contribute nearly one unit each. Proposition A.1 in $\mathrm { A p \mathrm { - } }$ pendix A.3 states and proves these efective-number properties formally.

## 3.3 Distance Monotonicity

We now turn to the main nontrivial property of GKD: increasing pairwise distances should increase diversity. In the union-volume picture, separating the centers should reduce overlap among their solids and therefore increase the volume of their union. However, a pairwise-distance expansion is only a comparison between two configurations: in general, there need not be a continuous path of configurations between them along which all pairwise distances are nondecreasing. Thus, this geometric intuition alone does not give a proof.

For labeled configurations $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )$ and $Y =$ $\left( \pmb { y } _ { 1 } , \dots , \pmb { y } _ { n } \right)$ in $( \mathbb { R } ^ { d } ) ^ { n }$ , write $X \preceq Y$ if

$$
\begin{array} { r } { \| \pmb { x } _ { i } - \pmb { x } _ { j } \| \leq \| \pmb { y } _ { i } - \pmb { y } _ { j } \| \quad \mathrm { f o r ~ a l l } ~ i , j , } \end{array}
$$

and write $X \prec Y$ if at least one inequality is strict. We write $X \cong Y$ when all pairwise distances agree, equivalently when the labeled configurations are congruent.

To identify kernels for which monotonicity can be proved, write $\phi ( r ) = g ( r ^ { 2 } )$ . A function $g : [ 0 , \infty ) $ $[ 0 , \infty )$ is completely monotone if it is continuous at zero, infinitely diferentiable on $( 0 , \infty )$ , and

$$
( - 1 ) ^ { k } g ^ { ( k ) } ( s ) \geq 0 { \mathrm { ~ f o r ~ e v e r y ~ } } k \geq 0 { \mathrm { ~ a n d ~ } } s > 0 .
$$

By the Bernstein-Widder theorem (Widder, 1931), such a function has a unique representation

$$
g ( s ) = \int _ { [ 0 , \infty ) } e ^ { - t s } d \mu _ { g } ( t )
$$

for a finite nonnegative measure $\mu _ { g }$ . Thus, $\phi$ can be expressed as an integral of Gaussian kernels $e ^ { - t r ^ { 2 } }$ with nonnegative weights. This is useful because distance monotonicity can first be proved for each Gaussian kernel separately and then integrated over t. We say that g has accumulating Laplace support if $\operatorname { s u p p } ( \mu _ { g } )$ has an accumulation point in $( 0 , \infty )$

Theorem 3.5 (Distance Monotonicity). Let $d \ge 1$ and let $\phi$ be a normalized d-admissible radial kernel of the form $\phi ( r ) = g ( r ^ { 2 } )$ where g is completely monotone. $I f X \preceq Y$ , then

$$
{ \mathcal { D } } _ { \phi , d } ( X ) \leq { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

$H ,$ in addition, $g$ has accumulating Laplace support, then

$$
\mathcal { D } _ { \phi , d } ( X ) = \mathcal { D } _ { \phi , d } ( Y ) \iff X \cong Y .
$$

Consequently, under this additional assumption, $X \prec$ Y implies $\mathcal { D } _ { \phi , d } ( X ) < \mathcal { D } _ { \phi , d } ( Y )$

For a fixed Gaussian scale $t > 0$ , define

$$
G _ { S } ( t ) : = \int _ { \mathbb { R } ^ { d } } e ^ { - t d _ { S } ( z ) ^ { 2 } } d z .
$$

A comparison theorem of Mourtada (2025) gives $X \preceq$ $Y \Rightarrow G _ { X } ( t ) \leq G _ { Y } ( t )$ for $t > 0$ . Since the representation above gives $\begin{array} { r } { \mathcal { D } _ { \phi , d } ( S ) = \int _ { ( 0 , \infty ) } G _ { S } ( t ) d \mu _ { g } ( t ) } \end{array}$ , integrating these inequalities proves weak distance monotonicity.

Under the additional accumulating-support assumption, the inequality is strict for every noncongruent expansion; the proof requires a further analytic argument and is given in Appendix A.4.

## 3.4 Mironov–Prokhorenkova Axioms on Euclidean Inputs

The axioms of Mironov and Prokhorenkova (2025) are formulated for diversity functions of general symmetric nonnegative distance matrices. We restrict their framework to distance matrices realizable by points in a prescribed Euclidean space. Since two labeled Euclidean configurations with the same pairwise distances are congruent, GKD defines a well-defined function of such distance matrices.

Corollary 3.6 (Mironov–Prokhorenkova axioms on Euclidean-realizable distance matrices). Let $d \ge 1$ and let $\phi$ be a normalized d-admissible radial kernel of the form

$$
\phi ( r ) = g ( r ^ { 2 } ) ,
$$

where g is completely monotone and has accumulating Laplace support. For every fixed $n \geq 1$ , the diversity function induced by GKD on Euclidean-realizable npoint distance matrices is permutation invariant and satisfies the monotonicity, uniqueness, and continuity axioms of Mironov and Prokhorenkova (2025).

Each axiom follows from the geometric properties established above. Monotonicity is exactly the strictdistance monotonicity result of Theorem 3.5. For uniqueness, replacing a genuinely distinct point by a duplicate removes one distinct solid from the union, and Theorem 3.4 shows that this strictly decreases GKD. Continuity requires one additional step because the axiom is stated in terms of distance matrices, while GKD is defined from point coordinates. If a sequence of Euclidean distance matrices converges, their realizations can be translated and rotated so that the corresponding point coordinates converge. Continuity of GKD in the point coordinates then gives continuity as a function of the distance matrix. The formal construction of the function and proof are given in $\mathrm { A p } \cdot$ pendix A.6.

## 4 THE CANONICAL KERNEL AND COMPUTATION IN $d \leq 3$

## 4.1 The Canonical Kernel

The preceding section establishes the axiomatic properties of GKD for a broad class of radial kernels. We now specialize to a particular kernel that retains these guarantees while also admitting useful computational structure.

For $d \geq 1$ and $\sigma > 0$ , define

$$
\psi _ { d , \sigma } ( r ) : = \frac { \Gamma ( d / 2 + 1 ) } { \pi ^ { d / 2 } \sigma ^ { d } } \left( 1 + \frac { r ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - ( d / 2 + 1 ) } ,
$$

where Γ denotes the Gamma function.

Then, for $z \in \mathbb { R } ^ { d } , z \longmapsto \psi _ { d , \sigma } ( \| z \| )$ is the density of the spherical d-dimensional Student-t distribution with two degrees of freedom and scale matrix $( \sigma ^ { 2 } / 2 ) I _ { d } .$ where $I _ { d }$ denotes the d-dimensional identity matrix. We call $\psi _ { d , \sigma }$ the canonical kernel, and GKD using this kernel canonical GKD.

This choice has several useful properties. Lemma 4.1 shows that the canonical kernel satisfies the assumptions of Theorem 3.5; hence canonical GKD satisfies the Mironov–Prokhorenkova axioms on Euclideanrealizable inputs. It is also consistent under changes of ambient dimension. Finally, as we show later, the choice of two degrees of freedom gives especially simple geometric identities that lead to exact low-dimensional algorithms and an eficient sampling procedure for randomized approximation.

Lemma 4.1 (Canonical kernel properties). For every $d \geq 1$ and $\sigma > 0$ , ψ<sub>d,σ</sub> is a normalized d-admissible radial kernel. Moreover, $\psi _ { d , \sigma } ( \boldsymbol { r } ) = g _ { d , \sigma } ( r ^ { 2 } )$ , where $g _ { d , \sigma }$ is completely monotone and has accumulating Laplace support.

Hence Theorem 3.5 and Corollary 3.6 apply to canonical GKD.

The parameter σ sets the geometric length scale. In particular,

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) = \mathcal { D } _ { \psi _ { d , 1 } , d } ( X / \sigma ) ,
$$

so changing σ is equivalent to rescaling the input. Consequently, the computational arguments below may assume $\sigma = 1$ after rescaling. See Lemma B.1 for the formal statement and proof.

Proposition 4.2 (Ambient dimension invariance). Let $g \ge d \ge 1 , \sigma > 0$ , and let $T : \mathbb { R } ^ { d }  \mathbb { R } ^ { g }$ be an afine isometric embedding. Then,

$$
\mathcal { D } _ { \psi _ { g , \sigma } , g } ( T ( X ) ) = \mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) ,
$$

for every nonempty $X \in ( \mathbb { R } ^ { d } ) ^ { n }$

For two points, canonical GKD has a particularly simple, dimension-independent form. If $r = \| \pmb { x } - \pmb { y } \|$ , then

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( \pmb { x } , \pmb { y } ) = 1 + \frac { r } { \sqrt { r ^ { 2 } + 4 \sigma ^ { 2 } } } .
$$

Thus, the value is 1 for coincident points and approaches 2 as the points become widely separated. This value depends only on the ratio $r / \sigma ,$ making explicit how σ controls the distance scale over which two points are regarded as distinct. The derivation is given in Proposition B.3.

The choice of two degrees of freedom is computationally useful. The canonical density admits a simple transformation to the uniform distribution on the unit ball, which gives the exact sampling procedure used by the randomized algorithm in Section 5. It also admits a simple divergence representation, allowing integrals in the low-dimensional exact algorithms of Section 4.3 to be reduced to boundary integrals. The correspond ing identities and their derivations are given in $\mathrm { A p \mathrm { - } }$ pendix B.1.

## 4.2 Computation Model and Input Conventions

The exact low-dimensional algorithms take as input the point coordinates $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ together with $\sigma > 0$ , where $d \in \{ 1 , 2 , 3 \}$ is fixed. If instead a Euclidean distance matrix is given, coordinates can be reconstructed in polynomial time (Alen car et al., 2019); this preprocessing cost is not included in the running times below.

We work in an elementary-function real-RAM model in which real arithmetic, comparisons, square roots, and evaluations of arctan and arccos are exact unit-cost operations. Thus, our running times are real-RAM bounds rather than bit-complexity bounds. Duplicate points can be removed in O(n log n) time, and the scaling identity of Lemma B.1 allows the proofs below to assume $\sigma = 1$

## 4.3 Exact Algorithms in Low Dimensions

We now give exact algorithms for the canonical GKD in dimensions $d \in \{ 1 , 2 , 3 \}$ . By Lemma B.1, the proofs may assume $\sigma = 1$

Theorem 4.3 (Exact computation in one dimension). Let $x _ { 1 } , \ldots , x _ { n } \ \in \ \mathbb { R } , \ X \ = \ ( x _ { 1 } , \ldots , x _ { n } )$ and $\sigma > 0$ Then ${ \mathcal { D } } _ { \psi _ { 1 , \sigma } , 1 } ( X )$ can be computed in $O ( n \log n )$ time. $I f x _ { 1 } \leq x _ { 2 } \leq \cdots \leq x _ { n }$ and $\Delta _ { i } = x _ { i + 1 } - x _ { i }$ , then

$$
\mathcal { D } _ { \psi _ { 1 , \sigma } , 1 } ( X ) = 1 + \sum _ { i = 1 } ^ { n - 1 } \frac { \Delta _ { i } } { \sqrt { \Delta _ { i } ^ { 2 } + 4 \sigma ^ { 2 } } } .
$$

After sorting the points, decomposing the line according to the nearest input point and integrating over the resulting intervals yields the formula above. The formula then can be evaluated in $O ( n )$ time, so the overall running time is O(n log n). See Appendix B.2.

The same nearest-point decomposition is given in higher dimensions by the Euclidean Voronoi diagram. Inside each Voronoi cell, the nearest input point is fixed, so GKD decomposes into one integral per cell. For the canonical kernel, the divergence theorem converts each cell integral into integrals over its boundary faces, reducing the dimension of the remaining integration by one. The full reduction is given in $\mathrm { A p \mathrm { - } }$ pendix B.3.

Theorem 4.4 (Exact computation in two dimensions). For $X \in \mathsf { ( \mathbb { R } ^ { 2 } ) } ^ { n }$ and $\sigma > 0 , \ \mathcal { D } _ { \psi _ { 2 , \sigma } , 2 } ( X )$ can be computed in $O ( n \log n )$ time in the elementary real-RAM model.

In two dimensions, Voronoi boundaries are onedimensional, and each resulting boundary integral has an elementary closed form. A planar Voronoi diagram can be constructed in O(n log n) time (Fortune, 1987), after which all boundary contributions can be evaluated in $O ( n )$ total time. The construction is given in Appendix B.4.

Theorem 4.5 (Exact computation in three dimensions). For $X \in ( \mathbb { R } ^ { 3 } ) ^ { n }$ and $\sigma > 0 , { \mathcal { D } } _ { \psi _ { 3 , \sigma } , 3 } ( X )$ can be computed in $O ( n ^ { 2 } )$ time in the elementary real-RAM model.

In three dimensions, the boundary terms are integrals over polygonal Voronoi faces. Appendix B.5 gives an elementary boundary formula for evaluating each such face integral. A three-dimensional Voronoi diagram can have $O ( n ^ { 2 } )$ total complexity and can be constructed in $O ( n ^ { 2 } )$ time (Chazelle, 1993); Appendix B.6 shows that the total work needed to evaluate all face contributions is also $O ( n ^ { 2 } )$

Higher dimensions. For $d \geq 4 .$ , this approach becomes less attractive for two reasons. First, the total size of a d-dimensional Voronoi diagram—the number of faces of all dimensions that may need to be processed—can be $\Theta ( n ^ { \lceil d / 2 \rceil } )$ in the worst case. For fixed $d ,$ this is still polynomial in $n ,$ but the exponent grows with $d ;$ without a fixed-dimensional restriction, an n-point Euclidean distance matrix may require dimension as large as $n - 1$ . Second, the divergence reduction leaves (d−1)-dimensional integrals over Voronoi faces, and we do not have comparably simple exact evaluations for these higher-dimensional integrals. We therefore turn to randomized approximation, which avoids explicit Voronoi construction and applies in arbitrary dimensions.

## 5 ESTIMATOR IN ARBITRARY DIMENSIONS

Canonical GKD itself is the deterministic diversity function defined in the previous sections, and its axiomatic guarantees concern this exact function. In arbitrary dimensions, rather than compute its value exactly, we use a randomized algorithm to approximate it. The randomized estimator is a computational procedure for approximating this deterministic function and is not itself an axiomatic diversity measure; its output may vary across runs and does not satisfy the continuity axioms. Our estimator is, to our knowledge, the first eficient randomized relative-error approximation scheme for a diversity function that satisfies all three Mironov–Prokhorenkova axioms since the prior two diversity measures constructed by Mironov and Prokhorenkova (2025), namely IntegralMaxClique and MultiDimVolume, do not admit an FPRAS, unless NP=RP (Appendix E.3).

We augment the elementary real-RAM model of Section 4.2 with exact unit-cost independent sampling from $\mathrm { U n i f } ( 0 , 1 ) , \mathcal { N } ( 0 , 1 )$ , and finite uniform distributions. The sampling and membership oracles needed below are constructed in Appendix C.

For $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ , let

$$
S _ { i } : = S _ { \psi _ { d , \sigma } , d } ( \pmb { x } _ { i } ) \subseteq \mathbb { R } ^ { d + 1 } .
$$

By Proposition 3.3,

$$
{ \mathscr { D } _ { \psi _ { d , \sigma } , d } ( X ) = \mathrm { V o l } _ { d + 1 } \left( \bigcup _ { i = 1 } ^ { n } S _ { i } \right) , \quad \mathrm { V o l } _ { d + 1 } ( S _ { i } ) = 1 . }
$$

Thus, estimating canonical GKD is an instance of estimating the volume of a union of n equal-volume sets. We specialize the self-adjusting coverage estimator of Karp et al. (1989) to this setting.

A trial first chooses i uniformly from $\{ 1 , \ldots , n \}$ and samples a point $Y$ uniformly from $S _ { i }$ . It then repeatedly chooses independent uniform indices $j$ and tests whether $Y \in S _ { j }$ , stopping at the first success.

For a point y in the union, let

$$
c ( \pmb { y } ) : = | \{ j \in \{ 1 , . . . , n \} \mid \pmb { y } \in S _ { j } \} |
$$

be the number of solids covering y. Points with larger $c ( y )$ are more likely to be sampled: the density of $Y$ at y is $c ( y ) / n$ . On the other hand, once $Y = y$ has been sampled, a random membership query succeeds with probability $c ( y ) / n$ . Hence, if L is the number of membership queries made in one trial,

$$
L \mid ( Y = y ) \sim \operatorname { G e o m } \left( { \frac { c ( y ) } { n } } \right) ,
$$

and therefore

$$
\mathbb { E } [ L \mid Y = y ] = { \frac { n } { c ( y ) } } .
$$

These two efects cancel: regions covered by many solids are sampled more often, but require few mem bership queries on average. Consequently,

$$
\begin{array} { r l } & { \mathbb { E } [ L ] = \displaystyle \int _ { \cup _ { i } S _ { i } } \frac { n } { c ( { \pmb y } ) } \frac { c ( { \pmb y } ) } { n } d { \pmb y } } \\ & { \qquad = \mathrm { V o l } _ { d + 1 } \left( \bigcup _ { i = 1 } ^ { n } S _ { i } \right) = { \mathscr { D } _ { \psi _ { d , \sigma } , d } ( X ) } . } \end{array}
$$

Thus, the expected length of one trial is exactly the canonical GKD value. The fixed-budget estimator below turns this observation into a relative-error approximation with high probability.

To obtain a high-probability relative-error guarantee, we use a fixed total number of membership queries. For $\epsilon , \delta \in ( 0 , 1 )$ , set

$$
B : = \left\lceil \frac { 8 ( 1 + \epsilon ) n \log ( 2 / \delta ) } { \epsilon ^ { 2 } } \right\rceil .
$$

Run consecutive trials until the B-th membership query, and let $N _ { B }$ be the number of trials completed within this budget. Since the expected length of

each trial is the GKD value, $N _ { B }$ is approximately $B / { D _ { \psi _ { d , \sigma } , d } ( X ) }$ , motivating the estimator

$$
\widehat { D } = \left\{ \begin{array} { l l } { \displaystyle \frac { B } { N _ { B } } } & { ( N _ { B } > 0 ) , } \\ { \displaystyle n } & { ( N _ { B } = 0 ) . } \end{array} \right.
$$

The case $N _ { B } = 0$ occurs exactly when the first trial has not finished within the B-query budget. We then set $\widehat { D } = n$ only to make the estimator well-defined. With probability at least $1 - \delta$ , the analysis guarantees that $N _ { B } > 0$ and the required relative-error bound, so this fallback value does not afect the theorem.

Theorem 5.1 (Randomized relative approximation). For every nonempty $X \in ( \mathbb { R } ^ { d } ) ^ { n } , \ d \geq 1$ , and $\epsilon , \delta \in$ (0, 1),

$$
\operatorname* { P r } \Bigl [ \Bigl | \widehat { D } - \mathscr { D } _ { \psi _ { d , \sigma } , d } ( X ) \Bigr | \le \epsilon \mathscr { D } _ { \psi _ { d , \sigma } , d } ( X ) \Bigr ] \geq 1 - \delta .
$$

$I n$ the randomized elementary real-RAM model, the running time is

$$
O \left( \frac { n d } { \epsilon ^ { 2 } } \log \frac { 2 } { \delta } \right) .
$$

The fixed-budget analysis of the self-adjusting coverage estimator gives the stated relative-error guarantee. Sampling uniformly from a canonical solid and testing membership both take $O ( d )$ time, so the B membership queries and at most B trial initializations require $O ( B d )$ total time. The full analysis and oracle constructions are given in Appendix C.

Comparison with prior diversity measures. The Euclidean restriction does not make the two diversity measures of Mironov and Prokhorenkova (2025) tractable. In Appendix E, we show that MultiDimVolume and IntegralMaxClique remain NP-hard even on Euclidean-realizable inputs and, unless $\mathrm { N P } = \mathrm { R P }$ , neither admits an FPRAS on this restricted domain. The reductions encode arbitrary graphs as small perturbations of a regular simplex.

## 6 AXIOM STRESS TESTS

Mironov and Prokhorenkova (2025) show that standard diversity measures violate their axioms on general symmetric distance matrices, a domain that includes matrices violating the triangle inequality and matrices not realizable by any point configuration. A priori, these axiomatic failures could be artifacts of that broader domain, and restricting to Euclidean inputs might be enough to restore the axioms. This section shows that this is not the case. The violations persist both on simple, explicitly constructed low-dimensional Euclidean configurations and under Euclidean-realizable perturbations of real learned representations (deferred to Appendix D.3). These experiments indicate that the Euclidean restriction does not by itself resolve the tension between the axioms and eficient computation.

## 6.1 Comparison Measures

We consider several commonly used diversity measures in addition to GKD: average pairwise distance, the Vendi Score (Friedman and Dieng, 2023), determinantbased DPP diversity (Mironov and Prokhorenkova, 2025; Wilhelm et al., 2018), and R´enyi kernel entropy (RKE) (Jalali et al., 2023). Full definitions are given in Appendix D.1. Our goal in this section is not to argue that one diversity measure is superior to another for a particular application, but rather to characterize their behavior in relation to the Mironov–Prokhorenkova axioms.

For the Vendi Score, DPP, and RKE, we use the RBF similarity matrix

$$
K _ { i j } : = \exp \left( - \frac { \| x _ { i } - x _ { j } \| ^ { 2 } } { 2 h ^ { 2 } } \right) ,
$$

with a common bandwidth h within each experiment.

All of these comparison measures are continuous on Euclidean inputs, so our experiments focus on the monotonicity and uniqueness axioms that they can violate.

## 6.2 Controlled Euclidean Stress Tests

We first use small Euclidean configurations to show that several standard comparison measures can violate the monotonicity and uniqueness axioms. These examples are designed to isolate a single axiom rather than to model a particular application. We set $\sigma = h = 1$ and compute canonical GKD exactly.

Distance monotonicity. Consider three points with

$$
d _ { 1 2 } = d _ { 1 3 } = 1 , \quad d _ { 2 3 } = c \in [ 1 . 7 6 , 1 . 9 9 ] .
$$

Increasing c changes exactly one pairwise distance while leaving the other two fixed, so strict distance monotonicity requires the diversity score to increase. Canonical GKD does so, while the Vendi score and determinant-based DPP diversity both decrease along this family.

Uniqueness. Consider the one-dimensional configu ration

$$
X ( t ) = \left( 0 , 0 , \frac { 1 } { 2 } , \frac { 1 } { 2 } - \frac { t } { 4 } \right) , \quad t \in [ 0 , 1 ] .
$$

At $t = 0$ , the third and fourth points coincide. For every $t > 0$ , the fourth point becomes distinct while the duplicate pair at the origin remains unchanged. The uniqueness axiom therefore requires $D ( X ( t ) ) >$ $D ( X ( 0 ) )$ for every $t > 0$ . Canonical GKD increases as required. Average pairwise distance, the Vendi Score, and RKE decrease on this family, while the determinant-based DPP score remains zero.

These examples show that axiom violations can occur even in simple low-dimensional Euclidean configurations. Appendix D.2 gives the explicit point configurations, plots of the diversity scores along both perturbations, and numerical endpoint values for all measures.

Additional stress tests under Euclidean-realizable perturbations around ImageNet/DINOv2 embeddings are given in Appendix D.3.

## 7 CONCLUSION

We studied the question of Mironov and Prokhorenkova (2025) under the restriction that the input distance matrix is Euclidean-realizable, and introduced Geometric Kernel Diversity (GKD). For a broad class of radial kernels, GKD satisfies monotonicity, uniqueness, and continuity on this domain. For the canonical Student $\scriptstyle - t _ { 2 }$ kernel, GKD is exactly computable in O(n log n) time in one and two dimensions and in $O ( n ^ { 2 } )$ time in three dimensions, while admitting a randomized $( 1 \pm \epsilon )$ -approximation in arbitrary dimensions. We also show that the two diversity measures of Mironov and Prokhorenkova (2025) remain NP-hard on Euclidean-realizable inputs and, unless $\mathrm { N P } = \mathrm { R P }$ , do not admit an FPRAS even under this restriction.

Several questions remain open. Most importantly, it is still unknown whether a diversity function satisfying all three axioms can be computed eficiently on unrestricted symmetric distance matrices. For GKD itself, it remains open whether canonical GKD admits eficient exact algorithms in dimensions $d \geq 4 .$ , and whether other kernels retaining the axiomatic guarantees lead to simpler or faster computation. Each of these questions represents exciting directions for future work, building on the results we present in this paper.

## AI use statement

In this work, we used generative AI tools for generating synthetic data sets; helping develop theoretical models or conceptual frameworks; summarizing or analyzing existing literature; formulating mathematical claims; providing critical ingredients for proving mathematical claims; assisting in the writing of proofs; drafting parts of the paper; improving the writing of the paper; proposing or refining hypotheses; designing or providing feedback on research methodology or experiments; implementing methods; cleaning and reformatting datasets; creating or modifying figures or images; suggesting experimental parameters; creating or editing software code; creating artifacts; and interpreting results. We have not used generative AI tools for any other applicable tasks with required disclosure. Assisting with translation and supporting qualitative and thematic data analysis were not applicable to this work.

We reviewed all AI-assisted work. Mathematical claims and proofs developed with AI assistance were checked by the authors. AI-assisted experimental and data-processing code was tested through exact calculations, reproducibility checks, and comparisons with recorded experimental outputs, as applicable. Suggested literature and citations were checked against the corresponding source material. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Alencar, J., Lavor, C., and Liberti, L. (2019). Realiz ing euclidean distance matrices by sphere intersection. Discrete Applied Mathematics, 256:5–10. Distance Geometry Theory and Applications (DGTA 16).

Bezdek, K. and Connelly, R. (2002). Pushing disks apart - the kneser-poulsen conjecture in the plane. Journal f¨ur die reine und angewandte Mathematik, 2002(553):221–236.

Bezdek, K. and Connelly, R. (2008). On the weighted Kneser-Poulsen conjecture. Periodica Mathematica Hungarica, 57(2):121–129.

Bringmann, K. and Friedrich, T. (2010). Approximating the volume of unions and intersections of highdimensional geometric objects. Computational Geometry, 43(6-7):601–610.

Chazelle, B. (1993). An optimal convex hull algorithm in any fixed dimension. Discrete & Computational Geometry, 10(4):377–409.

Cˆot´e, N. (2025). A Value-Free, Diversity-Sensitive Measure of Freedom. Ethical Theory and Moral Practice, 28(4):569–583.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248– 255. IEEE.

Ding, P. (2016). On the conditional distribution of the multivariate t distribution. The American Statistician, 70(3):293–295.

Fortune, S. (1987). A sweepline algorithm for Voronoi diagrams. Algorithmica, 2(1):153–174.

Friedman, D. and Dieng, A. B. (2023). The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023.

Gorbovickis, I. (2012). Strict kneser–poulsen conjecture for large radii. Geometriae Dedicata, 162:95 – 107.

Jalali, M., Li, C. T., and Farnia, F. (2023). An information-theoretic evaluation of generative models in learning multi-modal distributions. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S., editors, Advances in Neural Information Processing Systems, volume 36, pages 9931–9943. Curran Associates, Inc.

Jones, M. C. (2013). Perlman and wellner’s circular and transformed circular copulas are particular beta and t copulas. Symmetry, 5(1):81–85.

Karp, R. M. (1972). Reducibility among combinatorial problems. In Miller, R. E., Thatcher, J. W., and Bohlinger, J. D., editors, Complexity of Computer Computations: Proceedings of a symposium on the Complexity of Computer Computations, held March 20–22, 1972, at the IBM Thomas J. Watson Research Center, Yorktown Heights, New York, and sponsored by the Ofice of Naval Research, Mathematics Program, IBM World Trade Corporation, and the IBM Research Mathematical Sciences Department, pages 85–103. Springer US, Boston, MA.

Karp, R. M., Luby, M., and Madras, N. (1989). Monte-carlo approximation algorithms for enumeration problems. Journal of Algorithms, 10(3):429– 448.

Limbeck, K., Andreeva, R., Sarkar, R., and Rieck, B. (2024). Metric space magnitude for evaluating the diversity of latent representations. In Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and Zhang, C., editors, Advances in Neural Information Processing Systems, volume 37, pages 123911–123953. Curran Associates, Inc.

Mijatovi´c, A., Mramor, V., and Uribe Bravo, G. (2018). Projections of spherical brownian motion. Electronic Communications in Probability, 23.

Mironov, M. and Prokhorenkova, L. (2025). Measuring diversity: axioms and challenges. In Proceedings of the 42nd International Conference on Machine Learning, ICML’25. JMLR.org.

Motwani, R. and Raghavan, P. (1995). Randomized Algorithms. Cambridge International Series on Parallel Computation. Cambridge University Press.

Mourtada, J. (2025). Universal coding, intrinsic volumes, and metric complexity. Journal of the European Mathematical Society.

Muller, M. E. (1959). A note on a method for generating points uniformly on n-dimensional spheres. Commun. ACM, 2(4):19–20.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H. V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research.

Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A. C., and Fei-Fei, L. (2015). ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV), 115(3):211–252.

van Oosterom, A. and Strackee, J. (1983). The solid angle of a plane triangle. IEEE transactions on biomedical engineering, 30(2):125–126.

Widder, D. V. (1931). Necessary and Suficient Conditions for the Representation of a Function as a Laplace Integral. Transactions of the American Mathematical Society, 33(4):851–892.

Wilhelm, M., Ramanathan, A., Bonomo, A., Jain, S., Chi, E. H., and Gillenwater, J. (2018). Practical diversified recommendations on youtube with determinantal point processes. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management, pages 2165–2173.

## Supplementary Materials

## CONTENTS

A Properties of Geometric Kernel Diversity 12   
A.1 Under-Graph Representation 12   
A.2 Basic and Coverage Properties 12   
A.3 Efective-Number Interpretation . 14   
A.4 Distance Monotonicity 15   
A.5 Kneser-Poulsen Connection 18   
A.6 Mironov–Prokhorenkova Axioms on Euclidean Inputs . 20   
B Canonical Kernel and Exact Computation 23   
B.1 Canonical-Kernel Properties . 23   
B.2 Exact Computation in One Dimension 28   
B.3 Voronoi-Divergence Reduction . 29   
B.4 Exact Computation in Two Dimensions 31   
B.5 Planar Face Integration 32   
B.6 Exact Computation in Three Dimensions 35   
C Randomized Approximation 37   
C.1 The Self-Adjusting Union-Volume Estimator . 37   
C.2 Sampling and Membership Oracles for Canonical Solids 38   
C.3 Proof of the Randomized Approximation Theorem 40   
D Additional Axiom-Stress-Test Details 41   
D.1 Comparison Measures and Implementation 41   
D.2 Controlled Euclidean Stress Tests 41   
D.3 ImageNet/DINOv2 Stress Tests 43   
E Hardness of Existing Diversity Measures on Euclidean Inputs 46   
E.1 A Euclidean Graph Encoding 47   
E.2 Exact Hardness on Euclidean Inputs 48   
E.3 Hardness of Randomized Approximation 50

# A PROPERTIES OF GEOMETRIC KERNEL DIVERSITY

## A.1 Under-Graph Representation

Proof of Proposition 3.3. For every $z \in \mathbb { R } ^ { d }$ , since $\phi$ is strictly decreasing,

$$
\phi ( d _ { X } ( z ) ) = \operatorname* { m a x } _ { \pmb { x } \in \mathrm { s u p p } ( X ) } \phi ( \| z - \pmb { x } \| ) .
$$

Hence

$$
\begin{array} { c } { \left\{ ( z , u ) \in \mathbb { R } ^ { d } \times \mathbb { R } _ { \geq 0 } \ | \ 0 \leq u \leq \phi ( d _ { X } ( z ) ) \right\} = \displaystyle \bigcup _ { x \in \mathrm { s u p p } ( X ) } \left\{ ( z , u ) \in \mathbb { R } ^ { d } \times \mathbb { R } _ { \geq 0 } \ | \ 0 \leq u \leq \phi ( \| z - x \| ) \right\} } \\ { = \displaystyle \bigcup _ { x \in \mathrm { s u p p } ( X ) } S _ { \phi , d } ( x ) . } \end{array}
$$

These sets are measurable because $d _ { X }$ and $\phi$ are continuous. Therefore, by Tonelli’s theorem,

$$
\begin{array} { r l } & { \mathrm { V o l } _ { d + 1 } \left( \underset { x \in \mathrm { s u p p } ( X ) } { \bigcup } S _ { \phi , d } ( x ) \right) = \displaystyle \int _ { \mathbb { R } ^ { d } } \int _ { 0 } ^ { \infty } \mathbf { 1 } _ { u \leq \phi ( d _ { X } ( z ) ) } d u d z } \\ & { \qquad \quad = \displaystyle \int _ { \mathbb { R } ^ { d } } \phi ( d _ { X } ( z ) ) d z } \\ & { \qquad = \mathcal { D } _ { \phi , d } ( X ) . } \end{array}
$$

For every $\pmb { x } \in \mathbb { R } ^ { d }$ , translation invariance of Lebesgue measure and normalization of $\phi$ give

$$
\operatorname { V o l } _ { d + 1 } ( S _ { \phi , d } ( \pmb { x } ) ) = \int _ { \mathbb { R } ^ { d } } \phi ( \| \pmb { z } - \pmb { x } \| ) d \pmb { z } = Z _ { \phi , d } = 1 .
$$

Since X is nonempty, the union contains at least one such unit-volume solid, and hence

$$
{ \mathcal { D } } _ { \phi , d } ( X ) \geq 1 .
$$

On the other hand, subadditivity of Lebesgue measure gives

$$
\mathcal { D } _ { \phi , d } ( X ) = \operatorname { V o l } _ { d + 1 } \left( \bigcup _ { x \in \operatorname { s u p p } ( X ) } S _ { \phi , d } ( x ) \right) \leq \sum _ { x \in \operatorname { s u p p } ( X ) } \operatorname { V o l } _ { d + 1 } \left( S _ { \phi , d } ( x ) \right) = | \operatorname { s u p p } ( X ) | .
$$

Finally, $| \operatorname { s u p p } ( X ) | \leq n$ , and therefore,

$$
1 \leq \mathcal { D } _ { \phi , d } ( X ) \leq | \operatorname { s u p p } ( X ) | \leq n .
$$

In particular, ${ \mathcal { D } } _ { \phi , d } ( X )$ is finite.

## A.2 Basic and Coverage Properties

Proof of Theorem 3.4.

Permutation and duplicate invariance. The nearest-distance function depends only on the set of distinct input points:

$$
d _ { X } ( z ) = \operatorname* { m i n } _ { \pmb { x } \in \mathrm { s u p p } ( X ) } \| z - \pmb { x } \| .
$$

Permuting the entries of $X$ , or adding another copy of an existing point, therefore leaves $d _ { X }$ unchanged and hence leaves ${ \mathcal { D } } _ { \phi , d } ( X )$ unchanged.

Strict addition monotonicity. Let X be nonempty and let w $\not \in \operatorname { s u p p } ( X )$ . Set

$$
\ell : = d _ { X } ( \pmb { w } ) > 0 .
$$

For every $z \in B _ { d } ( w , \ell / 3 )$ and every $\pmb { x } \in \mathrm { s u p p } ( X )$ ),

$$
\| z - x \| \geq \| w - x \| - \| z - w \| \geq \frac { 2 \ell } { 3 } ,
$$

whereas

$$
\| z - w \| \leq \frac { \ell } { 3 } .
$$

Hence,

$$
d _ { X \cup \{ w \} } ( z ) = \| z - w \| < d _ { X } ( z )
$$

throughout $B _ { d } ( w , \ell / 3 )$ . Since $\phi$ is strictly decreasing,

$$
\phi ( d _ { X \cup \{ w \} } ( z ) ) > \phi ( d _ { X } ( z ) )
$$

on this ball. Moreover, adding a point can only decrease the nearest-point distance, so for every $z \in \mathbb { R } ^ { d }$

$$
d _ { X \cup \{ w \} } ( z ) \leq d _ { X } ( z )
$$

and therefore

$$
\phi ( d _ { X \cup \{ w \} } ( z ) ) \geq \phi ( d _ { X } ( z ) ) .
$$

Since $B _ { d } ( w , \ell / 3 )$ has positive Lebesgue measure, integrating these inequalities gives

$$
\mathcal { D } _ { \phi , d } ( X \cup \{ \pmb { w } \} ) > \mathcal { D } _ { \phi , d } ( X ) .
$$

For $X = \varnothing$ , the extension ${ \mathcal { D } } _ { \phi , d } ( \emptyset ) = 0$ gives

$$
\begin{array} { r } { \mathcal { D } _ { \phi , d } ( \{ \pmb { w } \} ) = 1 > 0 . } \end{array}
$$

Continuity. Suppose

$$
X ^ { ( k ) } = ( \pmb { x } _ { 1 } ^ { ( k ) } , \pmb { x } _ { 2 } ^ { ( k ) } , \dots , \pmb { x } _ { n } ^ { ( k ) } ) \longrightarrow X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )
$$

in $( \mathbb { R } ^ { d } ) ^ { n }$ . For every fixed $z \in \mathbb { R } ^ { d }$ , continuity of the minimum gives

$$
d _ { X ^ { ( k ) } } ( z ) \longrightarrow d _ { X } ( z ) ,
$$

and therefore

$$
\phi ( d _ { X ^ { ( k ) } } ( z ) ) \longrightarrow \phi ( d _ { X } ( z ) ) .
$$

Since $X ^ { ( k ) } \to X$ , there exists $R > 0$ such that $\| \pmb { x } _ { i } ^ { ( k ) } \| \leq R$ for all i and k. Therefore,

$$
d _ { X ^ { ( k ) } } ( z ) \geq \operatorname* { m a x } \{ 0 , \| z \| - R \} ,
$$

so, because $\phi$ is decreasing,

$$
0 \leq \phi ( d _ { X ^ { ( k ) } } ( z ) ) \leq h ( z ) : = \phi ( \operatorname* { m a x } \{ 0 , \| z \| - R \} ) .
$$

The function h is integrable. Indeed, it is bounded by $\phi ( 0 )$ on $B _ { d } ( \mathbf { 0 } , 2 R )$ , while for $\| z \| > 2 R$

$$
h ( z ) \leq \phi ( \| z \| / 2 ) ,
$$

and

$$
\int _ { \mathbb { R } ^ { d } } \phi ( \| z \| / 2 ) d z = 2 ^ { d } \int _ { \mathbb { R } ^ { d } } \phi ( \| \boldsymbol { u } \| ) d \boldsymbol { u } = 2 ^ { d } .
$$

The dominated convergence theorem therefore yields

$$
{ \mathcal { D } } _ { \phi , d } ( X ^ { ( k ) } ) \longrightarrow { \mathcal { D } } _ { \phi , d } ( X ) .
$$

Euclidean Isometry Invariance. Let $T : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ be a Euclidean isometry. Since T preserves distances,

$$
d _ { T ( X ) } ( T z ) = d _ { X } ( z ) .
$$

It also preserves Lebesgue measure, so the change of variables $w = T z$ gives

$$
\begin{array} { l } { \displaystyle \mathcal { D } _ { \phi , d } \big ( T ( X ) \big ) = \int _ { \mathbb { R } ^ { d } } \phi \big ( d _ { T ( X ) } ( w ) \big ) d w } \\ { \displaystyle \qquad = \int _ { \mathbb { R } ^ { d } } \phi \big ( d _ { T ( X ) } ( T z ) \big ) d z } \\ { \displaystyle \qquad = \int _ { \mathbb { R } ^ { d } } \phi \big ( d _ { X } ( z ) \big ) d z = \mathcal { D } _ { \phi , d } \big ( X \big ) . } \end{array}
$$

Submodularity. For a finite set $C \subseteq \mathbb { R } ^ { d }$ , define

$$
U _ { C } : = \bigcup _ { \pmb { x } \in C } S _ { \phi , d } ( \pmb { x } ) , \quad U _ { \emptyset } : = \emptyset .
$$

By Proposition 3.3,

$$
{ \mathcal { D } } _ { \phi , d } ( C ) = \operatorname { V o l } _ { d + 1 } ( U _ { C } ) .
$$

Let $A \subseteq B$ be finite sets and let w $\notin B$ . Then $U _ { A } \subseteq U _ { B }$ , and

$$
\begin{array} { r l } & { \mathcal { D } _ { \phi , d } ( A \cup \{ w \} ) - \mathcal { D } _ { \phi , d } ( A ) = \mathrm { V o l } _ { d + 1 } ( S _ { \phi , d } ( w ) \setminus U _ { A } ) , } \\ & { \mathcal { D } _ { \phi , d } ( B \cup \{ w \} ) - \mathcal { D } _ { \phi , d } ( B ) = \mathrm { V o l } _ { d + 1 } ( S _ { \phi , d } ( w ) \setminus U _ { B } ) . } \end{array}
$$

Since

$$
S _ { \phi , d } ( \pmb { w } ) \setminus U _ { A } \supseteq S _ { \phi , d } ( \pmb { w } ) \setminus U _ { B } ,
$$

we obtain

$$
\mathcal { D } _ { \phi , d } ( A \cup \{ w \} ) - \mathcal { D } _ { \phi , d } ( A ) \geq \mathcal { D } _ { \phi , d } ( B \cup \{ w \} ) - \mathcal { D } _ { \phi , d } ( B ) .
$$

Thus, the restriction of $\mathcal { D } _ { \phi , d }$ to finite sets is submodular.

## A.3 Efective-Number Interpretation

The normalization of each single-point solid gives GKD a natural efective-number interpretation.

Proposition A.1 (Efective-number interpretation). Let $d \geq 1$ , let ϕ be a normalized d-admissible radial kernel, and let $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ be nonempty. Define

$$
m : = | \operatorname { s u p p } ( X ) | .
$$

Then

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = 1 \iff m = 1 .
$$

If $m \geq 2 ,$ then

$$
1 < { \mathcal { D } } _ { \phi , d } ( X ) < m .
$$

Moreover, for every integer m $\geq 1$

$$
\operatorname* { s u p } \left\{ { \mathcal { D } } _ { \phi , d } ( X ) \mid X \subset \mathbb { R } ^ { d } , \mid X \mid = m \right\} = m .
$$

For $m \geq 2 .$ , this supremum is not attained.

Proof. If $m = 1$ , then duplicate invariance and normalization give

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = 1 .
$$

Conversely, if $m \geq 2$ , start from any one point of $\operatorname { s u p p } ( X )$ and add the remaining distinct points one at a time. Every addition is strict by Theorem 3.4, so

$$
{ \mathcal { D } } _ { \phi , d } ( X ) > 1 .
$$

It remains to prove the strict upper bound. By Proposition 3.3,

$$
\mathcal { D } _ { \phi , d } ( X ) = \mathrm { V o l } _ { d + 1 } \Bigg ( \bigcup _ { \substack { x \in \mathrm { s u p p } ( X ) } } S _ { \phi , d } ( \pmb { x } ) \Bigg ) \leq m .
$$

Suppose $m \geq 2$ , and choose distinct $\pmb { x } , \pmb { y } \in \operatorname { s u p p } ( X )$ . Let

$$
\pmb { c } : = \frac { \pmb { x } + \pmb { y } } { 2 } , \quad a : = \phi \bigg ( \frac { \| \pmb { x } - \pmb { y } \| } { 2 } \bigg ) .
$$

Since $\phi$ is strictly decreasing and nonnegative, $a > 0$ . By continuity, there exists an open neighborhood $U$ of c such that

$$
\phi ( \| z - x \| ) > \frac { a } { 2 } , \quad \phi ( \| z - y \| ) > \frac { a } { 2 }
$$

for every $z \in U$ . Hence

$$
U \times [ 0 , a / 2 ] \subseteq S _ { \phi , d } ( { \pmb x } ) \cap S _ { \phi , d } ( { \pmb y } ) ,
$$

so two of the union-volume solids overlap on a set of positive $( d + 1 )$ -dimensional measure. Therefore,

$$
\mathcal { D } _ { \phi , d } ( X ) < m .
$$

Finally, fix $m \geq 1$ and $\epsilon > 0 .$ . Since $\phi$ is normalized, choose $R > 0$ such that

$$
\int _ { B _ { d } ( \mathbf { 0 } , R ) } \phi ( \| z \| ) d z > 1 - \frac { \epsilon } { m } .
$$

Let $e _ { 1 }$ be the first standard basis vector and set

$$
\pmb { x } _ { i } : = 3 R i \pmb { e } _ { 1 } , \quad i = 1 , \ldots , m .
$$

The balls $B _ { d } ( { \pmb x } _ { i } , R )$ are pairwise disjoint, and every point of $B _ { d } ( { \pmb x } _ { i } , R )$ is closer to $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ than to any other $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { j } }$ Therefore,

$$
\begin{array} { l } { \displaystyle \mathcal { D } _ { \phi , d } \big ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { m } \big ) \geq \sum _ { i = 1 } ^ { m } \int _ { B _ { d } ( \pmb { x } _ { i } , R ) } \phi \big ( \| \pmb { z } - \pmb { x } _ { i } \| \big ) d \pmb { z } } \\ { \displaystyle = m \int _ { B _ { d } ( \pmb { 0 } , R ) } \phi \big ( \| \pmb { z } \| \big ) d \pmb { z } } \\ { \displaystyle \geq m - \epsilon . } \end{array}
$$

Together with the universal upper bound $\mathcal { D } _ { \phi , d } \leq m$ , this shows

$$
\operatorname* { s u p } \left\{ { \mathcal { D } } _ { \phi , d } ( X ) \mid X \subset \mathbb { R } ^ { d } , | X | = m \right\} = m .
$$

When $m \geq 2 .$ , the strict upper bound above shows that the supremum is not attained.

## A.4 Distance Monotonicity

For a nonempty finite configuration $S \subset \mathbb { R } ^ { d }$ and $t > 0 ,$ , define its Gaussian functional by

$$
G _ { S } ( t ) : = \int _ { \mathbb { R } ^ { d } } e ^ { - t d _ { S } ( z ) ^ { 2 } } d z .
$$

We first record the Gaussian comparison supplied by the nonconvex Wills functional.

Lemma A.2 (Gaussian Comparison). Let X, $Y \in ( \mathbb { R } ^ { d } ) ^ { n }$ satisfy $X \preceq Y$ . Then, for every $t > 0$

$$
G _ { X } ( t ) \leq G _ { Y } ( t ) .
$$

Proof. Consider

$$
\tau : \operatorname { s u p p } ( Y ) \to \operatorname { s u p p } ( X ) , \quad \tau ( \pmb { y } _ { i } ) : = \pmb { x } _ { i } .
$$

This map is well-defined: if ${ \bf { \nabla } } y _ { i } = { \bf { \nabla } } y _ { j }$ , then

$$
\| { \pmb x } _ { i } - { \pmb x } _ { j } \| \leq \| { \pmb y } _ { i } - { \pmb y } _ { j } \| = 0 ,
$$

so $\mathbf { \Delta } _ { \mathbf { x } _ { i } } = \mathbf { \Delta } _ { \mathbf { x } _ { j } }$ . It is surjective because every point of $\operatorname { s u p p } ( X )$ is $\pmb { x } _ { i } = \tau ( \pmb { y } _ { i } )$ for some i. Moreover,

$$
\begin{array} { r } { \| \tau ( \pmb { y } _ { i } ) - \tau ( \pmb { y } _ { j } ) \| = \| \pmb { x } _ { i } - \pmb { x } _ { j } \| \leq \| \pmb { y } _ { i } - \pmb { y } _ { j } \| . } \end{array}
$$

Thus, τ is distance-contracting.

By the comparison theorem for the nonconvex Wills functional (Mourtada, 2025, Theorem 4.1, Eq. (2.16)),

$$
\int _ { \mathbb { R } ^ { d } } e ^ { - \pi d _ { X } ( z ) ^ { 2 } } d z \leq \int _ { \mathbb { R } ^ { d } } e ^ { - \pi d _ { Y } ( z ) ^ { 2 } } d z .
$$

To obtain the result for arbitrary $t > 0$ , define

$$
a : = { \sqrt { \frac { t } { \pi } } } .
$$

Since $X ~ \preceq ~ Y$ , we also have $a X \preceq a Y$ . Applying the preceding inequality to aX and $a Y$ , and using the substitution ${ \pmb w } = a z$ , gives

$$
\begin{array} { r l r } {  { \int _ { { \mathbb { R } } ^ { d } } e ^ { - \pi d _ { a X } ( \pmb { w } ) ^ { 2 } } d \pmb { w } = a ^ { d } \int _ { { \mathbb { R } } ^ { d } } e ^ { - \pi a ^ { 2 } d _ { X } ( \pmb { z } ) ^ { 2 } } d \pmb { z } } } \\ & { } & { = a ^ { d } G _ { X } ( t ) , } \end{array}
$$

and similarly

$$
\int _ { \mathbb { R } ^ { d } } e ^ { - \pi d _ { a Y } ( \pmb { w } ) ^ { 2 } } d \pmb { w } = a ^ { d } G _ { Y } ( t ) .
$$

Dividing by $a ^ { d }$ yields

$$
G _ { X } ( t ) \leq G _ { Y } ( t ) .
$$

Lemma A.3 (Analyticity of the Gaussian functional). For every nonempty finite configuration $S \subset \mathbb { R } ^ { d }$ , the function $G _ { S }$ is real analytic on $( 0 , \infty )$

Proof. For complex w with $\mathrm { R e } ( w ) > 0$ , define

$$
G _ { S } ( w ) : = \int _ { \mathbb { R } ^ { d } } e ^ { - w d _ { S } ( z ) ^ { 2 } } d z .
$$

Let K be a compact subset of the half-plane $\{ w \in \mathbb { C } \mid \operatorname { R e } ( w ) > 0 \}$ . Choose $\eta > 0$ such that $\mathrm { R e } ( w ) \geq \eta$ for every $w \in K$ . Choose $R > 0$ so that $S \subset B _ { d } ( \mathbf { 0 } , R )$ . For $\| z \| \geq 2 R$

$$
d _ { S } ( z ) \geq \| z \| - R \geq { \frac { \| z \| } { 2 } } .
$$

Therefore,

$$
\left| e ^ { - w d _ { S } ( z ) ^ { 2 } } \right| \le e ^ { - \eta d _ { S } ( z ) ^ { 2 } } \le e ^ { - \eta \| z \| ^ { 2 } / 4 }
$$

outside $B _ { d } ( \mathbf { 0 } , 2 R )$ , while the integrand is bounded by 1 inside this ball. Thus, the integrand is dominated, uniformly over $w \in K$ , by an integrable function.

For every fixed z, the integrand is holomorphic in w, and the preceding bounds give an integrable dominating function locally uniformly in w. It follows that $G _ { S }$ is holomorphic on the half-plane $\mathrm { R e } ( w ) > 0$ . Therefore, its restriction to $( 0 , \infty )$ is real analytic. □

Proof of Theorem 3.5.

Gaussian-mixture representation and weak monotonicity. By the Bernstein-Widder theorem, there exists a finite nonnegative Borel measure $\mu _ { g }$ on $[ 0 , \infty )$ such that

$$
g ( s ) = \int _ { [ 0 , \infty ) } e ^ { - t s } d \mu _ { g } ( t ) .
$$

We first observe that

$$
\mu _ { g } ( \{ 0 \} ) = 0 .
$$

Indeed, if $\mu _ { g } ( \{ 0 \} ) > 0$ , then,

$$
g ( s ) \geq \mu _ { g } ( \{ 0 \} ) > 0
$$

for every $s \geq 0$ . Hence, for every nonempty configuration S,

$$
\mathcal { D } _ { \phi , d } ( S ) = \int _ { \mathbb { R } ^ { d } } g ( d _ { S } ( z ) ^ { 2 } ) d z = \infty ,
$$

contradicting the assumption that $\phi$ is a d-admissible kernel. Therefore,

$$
g ( s ) = \int _ { ( 0 , \infty ) } e ^ { - t s } d \mu _ { g } ( t ) .
$$

Using Tonelli’s theorem,

$$
\begin{array} { l } { \displaystyle \mathcal { D } _ { \phi , d } ( S ) = \int _ { \mathbb R ^ { d } } g ( d _ { S } ( z ) ^ { 2 } ) d z } \\ { \displaystyle = \int _ { \mathbb R ^ { d } } \int _ { ( 0 , \infty ) } e ^ { - t d _ { S } ( z ) ^ { 2 } } d \mu _ { g } ( t ) d z } \\ { \displaystyle = \int _ { ( 0 , \infty ) } G _ { S } ( t ) d \mu _ { g } ( t ) . } \end{array}
$$

Now let $X , Y \in ( \mathbb { R } ^ { d } ) ^ { n }$ satisfy $X \preceq Y$ . By Lemma A.2,

$$
G _ { X } ( t ) \leq G _ { Y } ( t )
$$

for every $t > 0$ . Hence,

$$
{ \mathcal { D } } _ { \phi , d } ( Y ) - { \mathcal { D } } _ { \phi , d } ( X ) = \int _ { ( 0 , \infty ) } ( G _ { Y } ( t ) - G _ { X } ( t ) ) d \mu _ { g } ( t ) \geq 0 .
$$

This proves weak distance monotonicity.

Equality at every Gaussian scale. Assume now that $g$ has accumulating Laplace support and that $\mathcal { D } _ { \phi , d } ( X ) = \mathcal { D } _ { \phi , d } ( Y )$ . Define

$$
h ( t ) : = G _ { Y } ( t ) - G _ { X } ( t ) .
$$

Then $h ( t ) \geq 0$ for every $t > 0$ , while

$$
\int _ { ( 0 , \infty ) } h ( t ) d \mu _ { g } ( t ) = 0 .
$$

By Lemma A.3, $h = G _ { Y } - G _ { X }$ is real analytic on $( 0 , \infty )$ , and hence continuous.

We claim that $h ( t ) = 0$ for every $t \in \mathrm { s u p p } ( \mu _ { g } ) \cap ( 0 , \infty )$ . Indeed, if $h ( t _ { 0 } ) > 0$ at some such point $t _ { 0 } ,$ , continuity of h gives an open neighborhood $I \subset ( 0 , \infty )$ of t<sub>0</sub> and a constant $c > 0$ such that

$$
h ( t ) \geq c
$$

for every $t \in I$ . Since $t _ { 0 } \in \operatorname { s u p p } ( \mu _ { g } ) , \mu _ { g } ( I ) > 0$ , and consequently

$$
\int _ { ( 0 , \infty ) } h ( t ) d \mu _ { g } ( t ) \geq \int _ { I } h ( t ) d \mu _ { g } ( t ) \geq c \mu _ { g } ( I ) > 0 ,
$$

which is a contradiction. Thus, h vanishes on supp $( \mu _ { g } ) \cap ( 0 , \infty )$ . Since this set has an accumulation point in $( 0 , \infty )$ , the identity theorem for real-analytic functions gives

$$
G _ { X } ( t ) = G _ { Y } ( t )
$$

for every $t > 0 ,$

Equality of equal-radius union volumes. For a finite configuration $S ,$ define the equal-radius union-volume function

$$
U _ { S } ( r ) : = \mathrm { V o l } _ { d } \left( \bigcup _ { s \in \mathrm { s u p p } ( S ) } B _ { d } ( s , r ) \right) , \quad r \geq 0 .
$$

For every $t > 0$

$$
e ^ { - t d _ { S } ( z ) ^ { 2 } } = \int _ { d _ { S } ( z ) ^ { 2 } } ^ { \infty } t e ^ { - t s } d s = \int _ { 0 } ^ { \infty } t e ^ { - t s } { \bf 1 } _ { d _ { S } ( z ) ^ { 2 } \leq s } d s .
$$

Tonelli’s theorem therefore gives

$$
\begin{array} { c } { { G _ { S } ( t ) = \displaystyle \int _ { \mathbb R ^ { d } } e ^ { - t d _ { S } ( z ) ^ { 2 } } d z } } \\ { { = t \displaystyle \int _ { 0 } ^ { \infty } e ^ { - t s } U _ { S } ( \sqrt s ) d s . } } \end{array}
$$

Since $G _ { X } ( t ) = G _ { Y } ( t )$ for every $t > 0$

$$
\int _ { 0 } ^ { \infty } e ^ { - t s } \left( U _ { Y } ( { \sqrt { s } } ) - U _ { X } ( { \sqrt { s } } ) \right) d s = 0
$$

for every $t > 0$ . Define

$$
q ( s ) : = U _ { Y } ( { \sqrt { s } } ) - U _ { X } ( { \sqrt { s } } ) .
$$

The function q is measurable and locally integrable. Moreover,

$$
| q ( s ) | \le U _ { X } \bigl ( \sqrt { s } \bigr ) + U _ { Y } \bigl ( \sqrt { s } \bigr ) \le 2 n v _ { d } s ^ { d / 2 } ,
$$

so $q$ has at most polynomial growth. By the standard uniqueness theorem for Laplace transforms of locally integrable functions of exponential order, $q ( s ) = 0$ for almost every $s > 0$ . Equivalently,

$$
U _ { X } ( r ) = U _ { Y } ( r )
$$

for almost every $r > 0$

Strictness. Suppose X and $Y$ are not congruent. Since $X \preceq Y$ , Y is a noncongruent expansion of $X .$ By the large-radius strictness theorem of Gorbovickis (2012, Theorem 1), there exists $r _ { 0 } > 0$ such that

$$
U _ { X } ( r ) < U _ { Y } ( r )
$$

for every $r > r _ { 0 }$ . Consequently,

$$
q ( s ) = U _ { Y } ( { \sqrt { s } } ) - U _ { X } ( { \sqrt { s } } ) > 0
$$

for every $s > r _ { 0 } ^ { 2 } .$ , contradicting $q ( s ) = 0$ for almost every $s > 0$

Conversely, if $X \cong Y$ , Euclidean isometry invariance from Theorem 3.4 gives

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

Hence,

$$
\mathcal { D } _ { \phi , d } ( X ) = \mathcal { D } _ { \phi , d } ( Y ) \iff X \cong Y .
$$

## A.5 Kneser-Poulsen Connection

For a nonempty finite configuration X, define its equal-radius union-volume function by

$$
U _ { X } ( r ) : = \mathrm { V o l } _ { d } \left( \bigcup _ { \pmb { x } \in \mathrm { s u p p } ( X ) } B _ { d } ( \pmb { x } , r ) \right) , \quad r \geq 0 .
$$

The equal-radius Kneser-Poulsen conjecture asserts that an expansive rearrangement of the centers of congruent Euclidean balls cannot decrease the volume of their union; in our notation,

$$
X \preceq Y \implies U _ { X } ( r ) \leq U _ { Y } ( r ) { \mathrm { ~ f o r ~ e v e r y ~ } } r \geq 0 .
$$

This is elementary in dimension $d = 1$ and follows in dimension $d = 2$ from the planar Kneser-Poulsen theorem of Bezdek and Connelly (2002). It remains open in general for $d \geq 3$

We first express GKD directly in terms of these union volumes.

Lemma A.4 (Equal-radius union-volume representation). Let $d \geq 1$ , let $\phi$ be a normalized d-admissible radial kernel, and let $X \in ( \mathbb { R } ^ { d } ) ^ { n }$ be nonempty. Then

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = \int _ { ( 0 , \phi ( 0 ) ] } U _ { X } ( \phi ^ { - 1 } ( t ) ) d t .
$$

Proof of Lemma $A . 4 .$ Since $\phi$ is nonnegative, strictly decreasing, and integrable with respect to $r ^ { d - 1 } d r .$ , we have

$$
\operatorname * { l i m } _ { r \to \infty } \phi ( r ) = 0 .
$$

Hence

$$
\phi ^ { - 1 } : ( 0 , \phi ( 0 ) ] \to [ 0 , \infty )
$$

is well-defined.

For every $z \in \mathbb { R } ^ { d }$

$$
\phi ( d _ { X } ( z ) ) = { \int } _ { ( 0 , \phi ( 0 ) ] } { \bf 1 } _ { t \le \phi ( d _ { X } ( z ) ) } d t .
$$

Tonelli’s theorem gives

$$
\mathcal { D } _ { \phi , d } ( X ) = \int _ { ( 0 , \phi ( 0 ) ] } \int _ { \mathbb { R } ^ { d } } \mathbf { 1 } _ { t \leq \phi ( d _ { X } ( z ) ) } d z d t .
$$

Because $\phi$ is strictly decreasing,

$$
t \leq \phi ( d _ { X } ( z ) ) \iff d _ { X } ( z ) \leq \phi ^ { - 1 } ( t ) .
$$

Therefore,

$$
\left\{ z \in \mathbb { R } ^ { d } \mid d _ { X } ( z ) \leq \phi ^ { - 1 } ( t ) \right\} = \bigcup _ { \pmb { x } \in \mathrm { s u p p } ( X ) } B _ { d } ( \pmb { x } , \phi ^ { - 1 } ( t ) ) ,
$$

whose volume is $U _ { X } ( \phi ^ { - 1 } ( t ) )$ . Substitution yields the claimed identity.

Corollary A.5 (Kneser-Poulsen monotonicity). Suppose the equal-radius Kneser-Poulsen inequality holds in dimension d. Let ϕ be a normalized d-admissible radial kernel and let $X , Y \in ( \mathbb { R } ^ { d } ) ^ { n }$ satisfy $X \preceq Y$ . Then

$$
{ \mathcal { D } } _ { \phi , d } ( X ) \leq { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

Moreover,

$$
\mathcal { D } _ { \phi , d } ( X ) = \mathcal { D } _ { \phi , d } ( Y ) \iff X \cong Y .
$$

Consequently, $i f X \prec Y$ , then

$$
{ \mathcal { D } } _ { \phi , d } ( X ) < { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

In particular, this conclusion holds unconditionally for $d \in \{ 1 , 2 \}$ , and conditionally on the equal-radius Kneser-Poulsen conjecture for every $d \geq 3$

Proof of Corollary A.5. By the assumed Kneser-Poulsen inequality,

$$
U _ { X } ( r ) \leq U _ { Y } ( r ) { \mathrm { ~ f o r ~ e v e r y ~ } } r \geq 0 .
$$

Lemma A.4 therefore gives

$$
{ \mathcal { D } } _ { \phi , d } ( X ) \leq { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

Now suppose that $X \not \cong Y$ . Since $X \preceq Y , Y$ is a noncongruent expansion of X. By the large-radius strictness theorem of Gorbovickis (2012), there exists $r _ { 0 } > 0$ such that

$$
U _ { X } ( r ) < U _ { Y } ( r ) \mathrm { ~ f o r ~ e v e r y ~ } r > r _ { 0 } .
$$

Since $\phi ( r _ { 0 } ) > 0$ , the interval $( 0 , \phi ( r _ { 0 } ) )$ has positive Lebesgue measure. For every t in this interval,

$$
\phi ^ { - 1 } ( t ) > r _ { 0 } ,
$$

and hence

$$
U _ { X } ( \phi ^ { - 1 } ( t ) ) < U _ { Y } ( \phi ^ { - 1 } ( t ) ) .
$$

The layer-cake representation therefore yields

$$
{ \mathcal { D } } _ { \phi , d } ( X ) < { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

Conversely, if $X \cong Y$ , Euclidean-isometry invariance from Theorem 3.4 gives

$$
{ \mathcal { D } } _ { \phi , d } ( X ) = { \mathcal { D } } _ { \phi , d } ( Y ) .
$$

This proves the equality characterization. Since $X \prec Y$ implies $X \not \cong Y$ , the strict inequality follows as well.

Finally, the equal-radius Kneser-Poulsen inequality is elementary for $d = 1$ and follows for $d = 2$ from Bezdek and Connelly (2002); for $d \geq 3 .$ the conclusion is conditional on the conjecture. □

## A.6 Mironov–Prokhorenkova Axioms on Euclidean Inputs

We first recall the formal axioms of Mironov and Prokhorenkova (2025).

Definition A.6 (Mironov–Prokhorenkova axioms). Fix $n \geq 1$ . Let $\mathcal { M } _ { n }$ be the set of symmetric nonnegative $n \times n$ matrices $A = \left( a _ { i j } \right)$ with zero diagonal satisfying

$$
a _ { i j } = 0 \Rightarrow a _ { i k } = a _ { j k } { \mathrm { ~ f o r ~ a l l ~ } } i , j , k .
$$

Let $F : { \mathcal { M } } _ { n } \to \mathbb { R }$ be a permutation-invariant diversity function. The axioms of Mironov and Prokhorenkova (2025) are:

(M) Monotonicity. Let $A = \left( a _ { i j } \right)$ and $B = ( b _ { i j } )$ belong to $\mathcal { M } _ { n }$ . If $a _ { i j } \geq b _ { i j }$ for all $i , j$ , and at least one of the inequalities is strict, then $F ( A ) > F ( B )$

(U) Uniqueness. Let $A = \left( a _ { i j } \right)$ and $B = \left( b _ { i j } \right)$ belong to $\mathcal { M } _ { n }$ and satisfy

$$
\begin{array} { r l r } & { a _ { i j } = b _ { i j } \quad \mathrm { f o r ~ a l l } \ i > 2 , j > 2 , } & \\ & { b _ { 1 j } = b _ { 2 j } \quad } & { \mathrm { f o r ~ a l l } \ j , } \\ & { a _ { 2 j } = b _ { 2 j } \quad } & { \mathrm { f o r ~ a l l } \ j > 2 , } \\ & { a _ { 1 j } > 0 \quad } & { \mathrm { f o r ~ a l l } \ j > 1 . } \end{array}
$$

Then $F ( A ) > F ( B )$

(C) Continuity. The function F is continuous on this domain with respect to the subspace topology inherited from $\mathbb { R } ^ { n \times n }$

For a labeled configuration $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ , let

$$
( D _ { X } ) _ { i j } : = \| \pmb { x } _ { i } - \pmb { x } _ { j } \| ,
$$

and define the set of Euclidean-realizable n-point distance matrices by

$$
\mathcal { E } _ { n , d } : = \left\{ D _ { X } \in \mathbb { R } ^ { n \times n } \vert X \in ( \mathbb { R } ^ { d } ) ^ { n } \right\} ,
$$

equipped with the subspace topology inherited from $\mathbb { R } ^ { n \times n }$ . We call X a realization of $D \in { \mathcal { E } } _ { n , d } { \mathrm { ~ i f ~ } } D _ { X } = D$ Define

$$
\overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { X } ) : = \mathscr { D } _ { \phi , d } ( X ) .
$$

This is well-defined because labeled Euclidean configurations with the same pairwise distance matrix are congruent, and GKD is invariant under Euclidean isometries.

We first establish the continuity fact needed to pass from point configurations to their distance matrices.

For two labeled configurations $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )$ and $Y = \left( { \pmb y } _ { 1 } , \dots , { \pmb y } _ { n } \right)$ in $( \mathbb { R } ^ { d } ) ^ { n }$ , write

$$
\| X - Y \| _ { \mathrm { p r o d } } : = \left( \sum _ { i = 1 } ^ { n } \| \pmb { x } _ { i } - \pmb { y } _ { i } \| ^ { 2 } \right) ^ { 1 / 2 } .
$$

Lemma A.7 (Alignment of convergent realizations). Let $D ^ { ( k ) } , D \in \mathcal { E } _ { n , d }$ satisfy

$$
D ^ { ( k ) } \longrightarrow D
$$

in $\mathbb { R } ^ { n \times n }$ . Let $X \in ( \mathbb { R } ^ { d } ) ^ { n }$ be any realization of D, and for each k, let $Y ^ { ( k ) } \in ( \mathbb { R } ^ { d } ) ^ { n }$ be any realization of $D ^ { ( k ) }$ Then there exist Euclidean isometries $T _ { k } : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ such that

$$
T _ { k } ( Y ^ { ( k ) } ) \longrightarrow X
$$

in $( \mathbb { R } ^ { d } ) ^ { n }$

Proof. Translate each configuration so that its first point is the origin. Define

$$
\widetilde { X } : = ( 0 , \pmb { x } _ { 2 } - \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } - \pmb { x } _ { 1 } )
$$

and

$$
\widetilde { Y } ^ { ( k ) } : = ( 0 , \pmb { y } _ { 2 } ^ { ( k ) } - \pmb { y } _ { 1 } ^ { ( k ) } , \ldots , \pmb { y } _ { n } ^ { ( k ) } - \pmb { y } _ { 1 } ^ { ( k ) } ) .
$$

For every $i ,$

$$
\| \pmb { y } _ { i } ^ { ( k ) } - \pmb { y } _ { 1 } ^ { ( k ) } \| = D _ { 1 i } ^ { ( k ) } \longrightarrow D _ { 1 i } .
$$

Therefore, the sequence $\widetilde Y ^ { ( k ) }$ is bounded in $( \mathbb { R } ^ { d } ) ^ { n }$ . We claim that

$$
\operatorname* { i n f } _ { Q \in \operatorname { O } ( d ) } \left\| Q \widetilde { Y } ^ { ( k ) } - \widetilde { X } \right\| _ { \mathrm { p r o d } } \longrightarrow 0 ,
$$

where $\mathrm { O } ( d )$ is the orthogonal group and $Q$ acts on each point of the configuration.

Suppose otherwise. Then there exist $\epsilon > 0$ and a subsequence, still indexed by k, such that

$$
\operatorname* { i n f } _ { Q \in \operatorname { O } ( d ) } \left\| Q \widetilde { Y } ^ { ( k ) } - \widetilde { X } \right\| _ { \mathrm { p r o d } } \geq \epsilon
$$

for every k. Since the sequence is bounded in a finite-dimensional Euclidean space, it has a further subsequence converging to some

$$
Z = ( 0 , z _ { 2 } , \ldots , z _ { n } ) \in ( \mathbb R ^ { d } ) ^ { n } .
$$

For every i, j $i , j$

$$
\| z _ { i } - z _ { j } \| = \operatorname* { l i m } _ { k \to \infty } D _ { i j } ^ { ( k ) } = D _ { i j } .
$$

Thus, $Z$ and $\widetilde { X }$ have the same pairwise distance matrix. Since their first points are both zero, the polarization identity gives

$$
\langle z _ { i } , z _ { j } \rangle = \frac { D _ { 1 i } ^ { 2 } + D _ { 1 j } ^ { 2 } - D _ { i j } ^ { 2 } } { 2 } = \langle { \pmb x } _ { i } - { \pmb x } _ { 1 } , { \pmb x } _ { j } - { \pmb x } _ { 1 } \rangle .
$$

Hence the two configurations have the same Gram matrix, so the correspondence

$$
{ \pmb z } _ { i } \longmapsto { \pmb x } _ { i } - { \pmb x } _ { 1 }
$$

extends to a linear isometry between their spans. Extending this isometry to the orthogonal complements gives some $Q \in \mathrm { O } ( d )$ such that

$$
Q Z = { \widetilde { X } } .
$$

Along the convergent subsequence,

$$
Q { \widetilde { Y } } ^ { ( k ) } \longrightarrow Q Z = { \widetilde { X } } ,
$$

contradicting the lower bound by ϵ. This proves the claim.

For every k, compactness of $\mathrm { O } ( d )$ implies that the infimum is attained by some $Q _ { k } \in \operatorname { O } ( d )$ . Therefore,

$$
Q _ { k } { \widetilde { Y } } ^ { ( k ) } \longrightarrow { \widetilde { X } } .
$$

Undoing the initial translations, define

$$
T _ { k } ( z ) : = x _ { 1 } + Q _ { k } ( z - y _ { 1 } ^ { ( k ) } ) .
$$

Each $T _ { k }$ is a Euclidean isometry, and

$$
T _ { k } ( Y ^ { ( k ) } ) \longrightarrow X .
$$

Proof of Corollary 3.6.

Permutation invariance. Let $\pi \in S _ { n } .$ , and let $P _ { \pi }$ be its permutation matrix. For $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )$ , define

$$
\pi X : = ( \pmb { x } _ { \pi ( 1 ) } , \ldots , \pmb { x } _ { \pi ( n ) } ) .
$$

Then,

$$
{ \cal D } _ { \pi X } = P _ { \pi } { \cal D } _ { X } P _ { \pi } ^ { T } .
$$

By permutation invariance from Theorem 3.4,

$$
\begin{array} { r l } & { \overline { { \mathscr { D } } } _ { \phi , d , n } \left( P _ { \pi } D _ { X } P _ { \pi } ^ { T } \right) = \mathscr { D } _ { \phi , d } ( \pi X ) } \\ & { \qquad = \mathscr { D } _ { \phi , d } ( X ) = \overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { X } ) . } \end{array}
$$

Monotonicity. Let $D _ { X } , D _ { Y } \in \mathcal { E } _ { n , d }$ satisfy

$$
( D _ { X } ) _ { i j } \leq ( D _ { Y } ) _ { i j }
$$

for every $i , j$ . Equivalently, $X \preceq Y$ . Theorem 3.5 gives

$$
\overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { X } ) = \mathscr { D } _ { \phi , d } ( X ) \leq \mathscr { D } _ { \phi , d } ( Y ) = \overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { Y } ) .
$$

If at least one entry inequality is strict, then $X \prec Y$ , and the same theorem gives

$$
\overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { X } ) < \overline { { \mathscr { D } } } _ { \phi , d , n } ( D _ { Y } ) .
$$

This is exactly the monotonicity axiom restricted to ${ \mathcal { E } } _ { n , d }$

Uniqueness. Let

$$
A = ( a _ { i j } ) , B = ( b _ { i j } ) \in \mathcal { E } _ { n , d }
$$

satisfy the hypotheses of the uniqueness axiom above.

Choose realizations $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } )$ and $Y = ( { \pmb y } _ { 1 } , \dots , { \pmb y } _ { n } )$ of A and $B ,$ respectively.

The condition $b _ { 1 j } = b _ { 2 j }$ for all j implies, by taking $j = 1$ , that $\left\| y _ { 1 } - y _ { 2 } \right\| = b _ { 1 2 } = b _ { 1 1 } = 0 , \mathrm { s o }  y _ { 1 } = y _ { 2 }$ . Moreover, the subconfigurations $( { \pmb x } _ { 2 } , \dots , { \pmb x } _ { n } )$ and $\left( { \pmb y } _ { 2 } , \dots , { \pmb y } _ { n } \right)$ have the same labeled pairwise distances and are therefore congruent. Applying a Euclidean isometry to $X$ , we may assume

$$
\begin{array} { r } { { \pmb x } _ { i } = { \pmb y } _ { i } \ \mathrm { f o r \ e v e r y } \ i \geq 2 . } \end{array}
$$

Let $S : = ( \pmb { x } _ { 2 } , \dots , \pmb { x } _ { n } )$ . Since $a _ { 1 j } > 0$ for every $j > 1$ , the point $\scriptstyle { \mathbf { { \vec { x } } } } _ { 1 }$ is distinct from every point in $\operatorname { s u p p } ( S )$ . Strict addition monotonicity from Theorem 3.4 therefore gives

$$
{ \mathcal { D } } _ { \phi , d } ( S ) < { \mathcal { D } } _ { \phi , d } ( X ) .
$$

On the other hand, ${ \bf { y } } _ { 1 } = { \bf { y } } _ { 2 }$ and $( { \pmb y } _ { 2 } , \dots , { \pmb y } _ { n } ) = S$ , so duplicate invariance gives

$$
\mathcal { D } _ { \phi , d } ( Y ) = \mathcal { D } _ { \phi , d } ( S ) .
$$

Consequently,

$$
\overline { { \mathcal { D } } } _ { \phi , d , n } ( A ) > \overline { { \mathcal { D } } } _ { \phi , d , n } ( B ) ,
$$

which is the uniqueness axiom restricted to $\mathcal { E } _ { n , d } .$

Continuity. Let $D ^ { ( k ) } \to D$ be a convergent sequence in ${ \mathcal { E } } _ { n , d } .$ Choose any realization $X \in ( \mathbb { R } ^ { d } ) ^ { n }$ of $D ,$ and for every $k ,$ choose any realization $Y ^ { ( k ) } \in ( \mathbb { R } ^ { \breve { d } } ) ^ { n }$ of $\bar { D } ^ { ( k ) }$

By Lemma A.7, there exist Euclidean isometries $T _ { k }$ such that

$$
T _ { k } ( Y ^ { ( k ) } ) \longrightarrow X .
$$

Using Euclidean isometry invariance and continuity from Theorem 3.4,

$$
\begin{array} { r l } { \overline { { \mathcal { D } } } _ { \phi , d , n } ( D ^ { ( k ) } ) = \mathcal { D } _ { \phi , d } ( Y ^ { ( k ) } ) } & { } \\ { = \mathcal { D } _ { \phi , d } ( T _ { k } ( Y ^ { ( k ) } ) ) } & { } \\ { \longrightarrow \mathcal { D } _ { \phi , d } ( X ) = \overline { { \mathcal { D } } } _ { \phi , d , n } ( D ) . } \end{array}
$$

Therefore, $\overline { { \mathcal { D } } } _ { \phi , d , n } : \mathcal { E } _ { n , d } \to$ R is continuous in the subspace topology on $\mathcal { E } _ { n , d } .$

## B CANONICAL KERNEL AND EXACT COMPUTATION

## B.1 Canonical-Kernel Properties

Proof of Lemma $4 . 1 .$ . Fix $d \geq 1$ and $\sigma > 0$ , and write

$$
\alpha : = { \frac { d } { 2 } } + 1 .
$$

The canonical kernel is

$$
\psi _ { d , \sigma } ( r ) = \frac { \Gamma ( \alpha ) } { \pi ^ { d / 2 } \sigma ^ { d } } \left( 1 + \frac { r ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - \alpha } .
$$

It is continuous, nonnegative, and strictly decreasing on [0, ∞).

We first verify normalization. Using polar coordinates and the substitution

$$
u : = \frac { r ^ { 2 } } { \sigma ^ { 2 } } ,
$$

we obtain

$$
\begin{array} { l } { \displaystyle \int _ { \mathbb R ^ { d } } \psi _ { d , \sigma } ( | | z | | ) d z = s _ { d - 1 } \frac { \Gamma ( \alpha ) } { \pi ^ { d / 2 } \sigma ^ { d } } \int _ { 0 } ^ { \infty } r ^ { d - 1 } \left( 1 + \frac { r ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - \alpha } d r } \\ { = \displaystyle \frac { \Gamma ( \alpha ) } { \Gamma ( d / 2 ) } \int _ { 0 } ^ { \infty } u ^ { d / 2 - 1 } ( 1 + u ) ^ { - \alpha } d u } \end{array}
$$

The remaining integral is

$$
B \left( \frac { d } { 2 } , 1 \right) = \frac { \Gamma ( d / 2 ) \Gamma ( 1 ) } { \Gamma ( d / 2 + 1 ) } = \frac { \Gamma ( d / 2 ) } { \Gamma ( \alpha ) } .
$$

Hence

$$
\int _ { \mathbb { R } ^ { d } } \psi _ { d , \sigma } ( \| z \| ) d z = 1 .
$$

Thus, $\psi _ { d , \sigma }$ is a normalized d-admissible radial kernel.

Now define

$$
g _ { d , \sigma } ( s ) : = \frac { \Gamma ( \alpha ) } { \pi ^ { d / 2 } \sigma ^ { d } } \left( 1 + \frac { s } { \sigma ^ { 2 } } \right) ^ { - \alpha } , \quad s \geq 0 .
$$

Then

$$
\psi _ { d , \sigma } ( \boldsymbol { r } ) = g _ { d , \sigma } ( \boldsymbol { r } ^ { 2 } ) .
$$

For every integer $k \geq 0$

$$
( - 1 ) ^ { k } g _ { d , \sigma } ^ { ( k ) } ( s ) = \frac { \Gamma ( \alpha ) } { \pi ^ { d / 2 } \sigma ^ { d } } \frac { 1 } { \sigma ^ { 2 k } } \left( 1 + \frac { s } { \sigma ^ { 2 } } \right) ^ { - \alpha - k } \prod _ { i = 0 } ^ { k - 1 } ( \alpha + i ) ,
$$

where the product is interpreted as 1 for $k = 0$ . Thus,

$$
( - 1 ) ^ { k } g _ { d , \sigma } ^ { ( k ) } ( s ) > 0
$$

for every $k \geq 0$ and $s \geq 0 ,$ , so $g _ { d , \sigma }$ is completely monotone.

To identify its Bernstein-Widder representing measure, use

$$
( s + \sigma ^ { 2 } ) ^ { - \alpha } = \frac { 1 } { \Gamma ( \alpha ) } \int _ { 0 } ^ { \infty } t ^ { \alpha - 1 } e ^ { - ( s + \sigma ^ { 2 } ) t } d t .
$$

Since

$$
\left( 1 + \frac { s } { \sigma ^ { 2 } } \right) ^ { - \alpha } = \sigma ^ { 2 \alpha } ( s + \sigma ^ { 2 } ) ^ { - \alpha } ,
$$

we obtain

$$
\begin{array} { l } { { \displaystyle g _ { d , \sigma } ( s ) = \frac { \Gamma ( \alpha ) } { \pi ^ { d / 2 } \sigma ^ { d } } \frac { \sigma ^ { 2 \alpha } } { \Gamma ( \alpha ) } \int _ { 0 } ^ { \infty } t ^ { \alpha - 1 } e ^ { - s t } e ^ { - \sigma ^ { 2 } t } d t } } \\ { { \displaystyle \quad = \frac { \sigma ^ { 2 } } { \pi ^ { d / 2 } } \int _ { 0 } ^ { \infty } t ^ { d / 2 } e ^ { - s t } e ^ { - \sigma ^ { 2 } t } d t . } } \end{array}
$$

Therefore, its representing measure has density

$$
d \mu _ { d , \sigma } ( t ) = \frac { \sigma ^ { 2 } } { \pi ^ { d / 2 } } t ^ { d / 2 } e ^ { - \sigma ^ { 2 } t } d t .
$$

This density is strictly positive for every $t > 0$ , so $\operatorname { s u p p } ( \mu _ { d , \sigma } )$ has accumulation points throughout $( 0 , \infty )$ . Hence $g _ { d , \sigma }$ has accumulating Laplace support. □

Lemma B.1 (Scaling identity). For every nonempty $X \in ( \mathbb { R } ^ { d } ) ^ { n }$

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) = \mathcal { D } _ { \psi _ { d , 1 } , d } ( X / \sigma ) ,
$$

where $X / \sigma = ( { \pmb x } _ { 1 } / \sigma , \ldots , { \pmb x } _ { n } / \sigma )$

Proof. The canonical kernel satisfies

$$
\psi _ { d , \sigma } ( r ) = \sigma ^ { - d } \psi _ { d , 1 } ( r / \sigma ) .
$$

Therefore,

$$
{ \mathcal { D } } _ { \psi _ { d , \sigma } , d } ( X ) = \int _ { \mathbb { R } ^ { d } } \sigma ^ { - d } \psi _ { d , 1 } \left( { \frac { d _ { X } ( z ) } { \sigma } } \right) d z .
$$

Using the substitution $z = \sigma u .$ , we have $d z = \sigma ^ { d } d u$ and

$$
\frac { d _ { X } ( \sigma u ) } { \sigma } = \operatorname* { m i n } _ { \substack { x \in \mathrm { s u p p } ( X ) } } \left\| u - \frac { x } { \sigma } \right\| = d _ { X / \sigma } ( \boldsymbol { u } ) .
$$

Consequently,

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) = \int _ { \mathbb { R } ^ { d } } \psi _ { d , 1 } ( d _ { X / \sigma } ( \pmb { u } ) ) d \pmb { u } = \mathcal { D } _ { \psi _ { d , 1 } , d } ( X / \sigma ) .
$$

Lemma B.2 (Dimensional marginal consistency). Let $g \geq d \geq 1 , \sigma > 0$ , and $r \geq 0$ . Then

$$
\int _ { \mathbb { R } ^ { g - d } } \psi _ { g , \sigma } \left( { \sqrt { r ^ { 2 } + \| v \| ^ { 2 } } } \right) d v = \psi _ { d , \sigma } ( r ) .
$$

Proof. Let $( \mathbf { u } , \mathbf { v } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { g - d }$ have density

$$
( { \pmb u } , { \pmb v } ) \longmapsto \psi _ { g , \sigma } \left( \sqrt { \| { \pmb u } \| ^ { 2 } + \| { \pmb v } \| ^ { 2 } } \right) .
$$

By Section 4.1, this is the density of

$$
t _ { g } \left( \mathbf { 0 } , { \frac { \sigma ^ { 2 } } { 2 } } I _ { g } , 2 \right) .
$$

By the standard marginal-closure property of the multivariate Student-t distribution (Ding, 2016), the first d coordinates satisfy

$$
\mathbf { u } \sim t _ { d } \left( \mathbf { 0 } , \frac { \sigma ^ { 2 } } { 2 } I _ { d } , 2 \right) .
$$

Their density is therefore

$$
\mathbf { \Omega } \bullet \longmapsto \psi _ { d , \sigma } ( \| \mathbf { \boldsymbol { u } } \| ) .
$$

On the other hand, by the definition of a marginal density, it is

$$
u \longmapsto \int _ { \mathbb { R } ^ { g - d } } \psi _ { g , \sigma } \left( { \sqrt { \| u \| ^ { 2 } + \| v \| ^ { 2 } } } \right) d v .
$$

Equating these expressions and choosing any u with $\lVert \mathbf { \boldsymbol { u } } \rVert = r$ gives the claimed identity.

Proof of Proposition 4.2. Let $T : \mathbb { R } ^ { d }  \mathbb { R } ^ { g }$ be an afine isometric embedding. After composing $T$ with a Euclidean isometry of $\mathbb { R } ^ { g }$ , which does not change canonical GKD, we may assume without loss of generality that

$$
\begin{array} { r } { T ( \pmb { u } ) = ( \pmb { u } , \pmb { 0 } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { g - d } . } \end{array}
$$

For $( \boldsymbol { u } , \boldsymbol { v } ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { g - d }$

$$
\begin{array} { l } { { d _ { T ( X ) } ( ( { \pmb u } , { \pmb v } ) ) = \underset { { \pmb x } \in \mathrm { s u p p } ( X ) } { \operatorname* { m i n } } \sqrt { \| { \pmb u } - { \pmb x } \| ^ { 2 } + \| { \pmb v } \| ^ { 2 } } } } \\ { { = \sqrt { d _ { X } ( \pmb u ) ^ { 2 } + \| { \pmb v } \| ^ { 2 } } . } } \end{array}
$$

Therefore, Tonelli’s theorem and Lemma B.2 give

$$
\begin{array} { r l } & { \displaystyle \mathcal { D } _ { \psi _ { g , \sigma } , g } ( T ( X ) ) = \int _ { \mathbb R ^ { d } } \int _ { \mathbb R ^ { g - d } } \psi _ { g , \sigma } \left( \sqrt { d _ { X } ( \pmb u ) ^ { 2 } + \| \pmb v \| ^ { 2 } } \right) d \pmb v d \pmb u } \\ & { \quad \quad \quad = \int _ { \mathbb R ^ { d } } \psi _ { d , \sigma } ( d _ { X } ( \pmb u ) ) d \pmb u } \\ & { \quad \quad \quad = \mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) . } \end{array}
$$

Proposition B.3 (Pairwise canonical GKD). Let $d \geq 1 , \sigma > 0 _ { }$ , and x $\mathbf { \Psi } : \pmb { y } \in \mathbb { R } ^ { d } . \ \mathbf { \backslash } f r = \| \pmb { x } - \pmb { y } \|$ , then

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( \pmb { x } , \pmb { y } ) = 1 + \frac { r } { \sqrt { r ^ { 2 } + 4 \sigma ^ { 2 } } } .
$$

Proof. Let $r : = \| \pmb { x } - \pmb { y } \|$ . By Euclidean isometry invariance, we may assume

$$
{ \pmb x } = - \frac { r } { 2 } { \pmb e } _ { 1 } , \quad { \pmb y } = \frac { r } { 2 } { \pmb e } _ { 1 } .
$$

Their perpendicular bisector is the hyperplane $z _ { 1 } = 0$ . The point x is the nearest on the half-space $z _ { 1 } \leq 0 ,$ while y is nearest on the half-space $z _ { 1 } \geq 0$ . By symmetry,

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( { \pmb x } , { \pmb y } ) = 2 \int _ { \{ z _ { 1 } \leq 0 \} } \psi _ { d , \sigma } ( \| z - { \pmb x } \| ) d z .
$$

Translating by x gives

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( { \pmb x } , { \pmb y } ) = 2 \int _ { \{ u _ { 1 } \leq r / 2 \} } \psi _ { d , \sigma } ( \| { \pmb u } \| ) d { \pmb u } .
$$

Let u have density

$$
\mathbf { \Omega } \bullet \longmapsto \psi _ { d , \sigma } ( \| \mathbf { \boldsymbol { u } } \| ) .
$$

By Lemma B.2, its first coordinate has density

$$
p _ { \sigma } ( t ) = \frac { 1 } { 2 \sigma } \left( 1 + \frac { t ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - 3 / 2 } .
$$

Hence

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( { \pmb x } , { \pmb y } ) = 2 \mathrm { P r } \left[ \mathrm { u } _ { 1 } \leq \frac { r } { 2 } \right] .
$$

An antiderivative of $p _ { \sigma }$ is

$$
{ \frac { t } { 2 { \sqrt { t ^ { 2 } + \sigma ^ { 2 } } } } } ,
$$

and therefore

$$
\operatorname* { P r } [ \mathrm { u } _ { 1 } \leq t ] = \frac { 1 } { 2 } + \frac { t } { 2 \sqrt { t ^ { 2 } + \sigma ^ { 2 } } } .
$$

Substituting $t = r / 2$ yields

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( \pmb { x } , \pmb { y } ) = 1 + \frac { r } { \sqrt { r ^ { 2 } + 4 \sigma ^ { 2 } } } .
$$

In particular,

$$
\mathcal { D } _ { \psi _ { d , \sigma } , d } ( { \pmb x } , { \pmb y } ) = \frac { 3 } { 2 } \Longleftrightarrow r = \frac { 2 } { \sqrt { 3 } } \sigma .
$$

□

Why two degrees of freedom? Ambient-dimensional consistency alone does not single out the choice $\nu = 2 ,$ For any $\nu > 0$ , define

$$
\psi _ { d , \sigma } ^ { ( \nu ) } ( r ) : = \frac { \Gamma ( ( d + \nu ) / 2 ) } { \Gamma ( \nu / 2 ) \pi ^ { d / 2 } \sigma ^ { d } } \left( 1 + \frac { r ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - ( d + \nu ) / 2 } .
$$

Then

$$
\psi _ { d , \sigma } ^ { ( 2 ) } = \psi _ { d , \sigma } ,
$$

and ${ \pmb u } \longmapsto \psi _ { d , \sigma } ^ { ( \nu ) } ( \left\| { \pmb u } \right\| )$ is the density of

$$
t _ { d } \left( \mathbf { 0 } , { \frac { \sigma ^ { 2 } } { \nu } } I _ { d } , \nu \right) .
$$

For every fixed $\nu ,$ this family has the same dimensional marginal-consistency property. Two additional simplifications used by our algorithms occur specifically when $\nu = 2$

Uniform-ball transformation. Define

$$
T ( \mathbf { \boldsymbol { u } } ) : = \frac { \mathbf { \boldsymbol { u } } } { \sqrt { \sigma ^ { 2 } + \| \mathbf { \boldsymbol { u } } \| ^ { 2 } } } .
$$

This is a smooth bijection from $\mathbb { R } ^ { d }$ onto the unit ball $B _ { d } ( \mathbf { 0 } , 1 )$ , with inverse

$$
T ^ { - 1 } ( \pmb { y } ) = \frac { \sigma \pmb { y } } { \sqrt { 1 - \| \pmb { y } \| ^ { 2 } } } .
$$

Lemma B.4 (Jacobian of the radial transformation). For

$$
T ( \mathbf { \boldsymbol { u } } ) : = \frac { \mathbf { \boldsymbol { u } } } { \sqrt { \sigma ^ { 2 } + \| \mathbf { \boldsymbol { u } } \| ^ { 2 } } } ,
$$

$$
| \mathrm { d e t } D T ( { \pmb u } ) | = \frac { 1 } { \sigma ^ { d } } \left( 1 + \frac { \| { \pmb u } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - ( d / 2 + 1 ) } .
$$

Equivalently, for ${ \pmb y } = { \pmb T } ( { \pmb u } )$ 2

$$
\left| \operatorname* { d e t } D T ^ { - 1 } ( \pmb { y } ) \right| = \sigma ^ { d } ( 1 - \| \pmb { y } \| ^ { 2 } ) ^ { - ( d / 2 + 1 ) } .
$$

Proof. For $a , b \in \{ 1 , \ldots , d \}$

$$
\frac { \partial T _ { a } } { \partial u _ { b } } = ( \sigma ^ { 2 } + \Vert \pmb { u } \Vert ^ { 2 } ) ^ { - 1 / 2 } \delta _ { a b } - ( \sigma ^ { 2 } + \Vert \pmb { u } \Vert ^ { 2 } ) ^ { - 3 / 2 } u _ { a } u _ { b } .
$$

Hence

$$
D T ( \boldsymbol { \mathbf { u } } ) = ( \sigma ^ { 2 } + \| \boldsymbol { \mathbf { u } } \| ^ { 2 } ) ^ { - 1 / 2 } \left( I - \frac { \boldsymbol { \mathbf { u } } \boldsymbol { \mathbf { u } } ^ { T } } { \sigma ^ { 2 } + \| \boldsymbol { \mathbf { u } } \| ^ { 2 } } \right) .
$$

By the matrix determinant lemma,

$$
\operatorname* { d e t } \left( I - \frac { { \pmb u } { \pmb u } ^ { T } } { \sigma ^ { 2 } + \| { \pmb u } \| ^ { 2 } } \right) = 1 - \frac { \| { \pmb u } \| ^ { 2 } } { \sigma ^ { 2 } + \| { \pmb u } \| ^ { 2 } } = \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + \| { \pmb u } \| ^ { 2 } } .
$$

Therefore,

$$
\left| \operatorname* { d e t } D T ( \pmb { u } ) \right| = \sigma ^ { 2 } ( \sigma ^ { 2 } + \| \pmb { u } \| ^ { 2 } ) ^ { - ( d / 2 + 1 ) } = \frac { 1 } { \sigma ^ { d } } \left( 1 + \frac { \| \pmb { u } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - ( d / 2 + 1 ) } .
$$

Since

$$
1 - \| \pmb { y } \| ^ { 2 } = \frac { \sigma ^ { 2 } } { \sigma ^ { 2 } + \| \pmb { u } \| ^ { 2 } } ,
$$

taking reciprocals gives

$$
\left| \operatorname* { d e t } D T ^ { - 1 } ( \pmb { y } ) \right| = \sigma ^ { d } ( 1 - \| \pmb { y } \| ^ { 2 } ) ^ { - ( d / 2 + 1 ) } .
$$

Let u have density $\psi _ { d , \sigma } ^ { ( \nu ) } ( \left\| \boldsymbol { u } \right\| )$ and set ${ \bf y } = T ( { \bf u } )$ . Since

$$
1 + { \frac { \| { \pmb u } \| ^ { 2 } } { \sigma ^ { 2 } } } = { \frac { 1 } { 1 - \| { \pmb y } \| ^ { 2 } } } ,
$$

Lemma B.4 and the change of variables formula give

$$
f _ { \mathbf { y } } ( \pmb { y } ) = \frac { \Gamma ( ( d + \nu ) / 2 ) } { \Gamma ( \nu / 2 ) \pi ^ { d / 2 } } ( 1 - \| \pmb { y } \| ^ { 2 } ) ^ { \nu / 2 - 1 } , \quad \| \pmb { y } \| < 1 .
$$

This is the spherical beta distribution associated with the Student-t radial transformation (Jones, 2013). When $\nu = 2$ , the exponent is zero and

$$
f _ { \mathbf { y } } ( \pmb { y } ) = \frac { \Gamma ( d / 2 + 1 ) } { \pi ^ { d / 2 } } = \frac { 1 } { \mathrm { V o l } _ { d } ( B _ { d } ( \mathbf { 0 } , 1 ) ) } .
$$

Hence

$$
\mathbf { y } \sim \mathrm { U n i f } ( B _ { d } ( \mathbf { 0 } , 1 ) ) .
$$

This is the uniform-ball representation used by the exact sampling oracle in Appendix C.2.

Divergence simplification. After scaling to $\sigma = 1$ , define

$$
\mathbf { F } _ { \nu } ( \pmb { u } ) : = \frac { \pmb { u } } { ( 1 + \Vert \pmb { u } \Vert ^ { 2 } ) ^ { ( d + \nu - 2 ) / 2 } } .
$$

Writing

$$
p : = { \frac { d + \nu - 2 } { 2 } } ,
$$

a direct diferentiation gives

$$
\begin{array} { c } { { \nabla \cdot { \bf F } _ { \nu } ( { \pmb u } ) = d ( 1 + \| { \pmb u } \| ^ { 2 } ) ^ { - p } - 2 p \| { \pmb u } \| ^ { 2 } ( 1 + \| { \pmb u } \| ^ { 2 } ) ^ { - p - 1 } } } \\ { { { } } } \\ { { { } = \left( d + ( 2 - \nu ) \| { \pmb u } \| ^ { 2 } \right) ( 1 + \| { \pmb u } \| ^ { 2 } ) ^ { - ( d + \nu ) / 2 } . } } \end{array}
$$

Thus, $\nu = 2$ is exactly the value at which the quadratic term disappears:

$$
\nabla \cdot \left( \frac { \pmb { u } } { ( 1 + \| \pmb { u } \| ^ { 2 } ) ^ { d / 2 } } \right) = \frac { d } { ( 1 + \| \pmb { u } \| ^ { 2 } ) ^ { d / 2 + 1 } } .
$$

Since

$$
c _ { d } : = \frac { \Gamma ( d / 2 ) } { 2 \pi ^ { d / 2 } }
$$

satisfies

$$
c _ { d } d = \frac { \Gamma ( d / 2 + 1 ) } { \pi ^ { d / 2 } } ,
$$

we obtain

$$
c _ { d } \nabla \cdot \left( \frac { \pmb { u } } { ( 1 + \| \pmb { u } \| ^ { 2 } ) ^ { d / 2 } } \right) = \psi _ { d , 1 } ( \| \pmb { u } \| ) .
$$

This is precisely the divergence identity used in Appendix B.3.

## B.2 Exact Computation in One Dimension

Proof of Theorem $4 . 3 .$ Sort the input points so that

$$
x _ { 1 } \leq x _ { 2 } \leq \cdot \cdot \cdot \leq x _ { n } ,
$$

and define $\Delta _ { i } : = x _ { i + 1 } - x _ { i }$ for $i \in \{ 1 , \ldots , n - 1 \}$ . Repeated points cause no dificulty: if $x _ { i } = x _ { i + 1 }$ , then $\Delta _ { i } = 0$ and the corresponding term in the formula vanishes.

For $d = 1$ , the canonical kernel is

$$
\psi _ { 1 , \sigma } ( r ) = \frac { 1 } { 2 \sigma } \left( 1 + \frac { r ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - 3 / 2 } .
$$

An antiderivative on $[ 0 , \infty )$ is

$$
F _ { \sigma } ( r ) : = \frac { r } { 2 \sqrt { r ^ { 2 } + \sigma ^ { 2 } } } , \quad F _ { \sigma } ^ { \prime } ( r ) = \psi _ { 1 , \sigma } ( r ) .
$$

Up to shared endpoints of measure zero, R is decomposed into the two unbounded intervals

$$
( - \infty , x _ { 1 } ] , \quad [ x _ { n } , \infty )
$$

and the intervals $[ x _ { i } , x _ { i + 1 } ]$ for $i \in \{ 1 , \ldots , n - 1 \}$ . Intervals corresponding to duplicate consecutive points are degenerate and contribute zero.

On $( - \infty , x _ { 1 } )$ , the point $x _ { 1 }$ is a nearest input point, so

$$
\int _ { - \infty } ^ { x _ { 1 } } \psi _ { 1 , \sigma } ( d _ { X } ( z ) ) d z = \int _ { 0 } ^ { \infty } \psi _ { 1 , \sigma } ( r ) d r = \frac { 1 } { 2 } .
$$

Similarly,

$$
\int _ { x _ { n } } ^ { \infty } \psi _ { 1 , \sigma } ( d _ { X } ( z ) ) d z = { \frac { 1 } { 2 } } .
$$

Thus, the two unbounded intervals contribute a total of 1.

Now fix $i \in \{ 1 , \ldots , n - 1 \}$ . On $[ x _ { i } , x _ { i + 1 } ]$ , the nearest input point is $x _ { i }$ to the left of the midpoint and $x _ { i + 1 }$ to the right. Therefore, by symmetry,

$$
\begin{array} { l l l } { \displaystyle \int _ { x _ { i } } ^ { x _ { i + 1 } } \psi _ { 1 , \sigma } ( { d _ { X } ( z ) } ) d z = 2 \int _ { 0 } ^ { \Delta _ { i } / 2 } \psi _ { 1 , \sigma } ( \boldsymbol { r } ) d \boldsymbol { r } } \\ { \displaystyle \quad \quad = 2 \left[ \frac { \boldsymbol { r } } { 2 \sqrt { r ^ { 2 } + \sigma ^ { 2 } } } \right] _ { 0 } ^ { \Delta _ { i } / 2 } } \\ { \displaystyle \quad \quad = \frac { \Delta _ { i } } { \sqrt { \Delta _ { i } ^ { 2 } + 4 \sigma ^ { 2 } } } . } \end{array}
$$

Summing all contributions gives

$$
\mathcal { D } _ { \psi _ { 1 , \sigma } , 1 } ( X ) = 1 + \sum _ { i = 1 } ^ { n - 1 } \frac { \Delta _ { i } } { \sqrt { \Delta _ { i } ^ { 2 } + 4 \sigma ^ { 2 } } } .
$$

For $n = 1$ , the sum is empty and the same formula gives ${ \mathcal D } _ { \psi _ { 1 , \sigma } , 1 } ( X ) = 1$

Sorting takes $O ( n \log n )$ time. Once the points are sorted, evaluating the $n - 1$ terms requires $O ( n )$ elementary real-RAM operations. Hence ${ \mathcal { D } } _ { \psi _ { 1 , \sigma } , 1 } ( X )$ can be computed in $O ( n \log n )$ time. □

## B.3 Voronoi-Divergence Reduction

By Lemma B.1, it sufices to consider the standard canonical kernel $\psi _ { d , 1 }$

After removing duplicate points, write $X ~ = ~ ( x _ { 1 } , \ldots , x _ { m } )$ for the pairwise distinct points. If $m = 1$ , then $\mathcal { D } _ { \psi _ { d , 1 } , d } ( X ) = 1$ , so assume $m \geq 2 .$

For each $i \in \{ 1 , \ldots , m \}$ , let

$$
P _ { i } : = \left\{ z \in \mathbb { R } ^ { d } \ | \ \| z - \pmb { x } _ { i } \| \leq \| z - \pmb { x } _ { j } \| \ \mathrm { f o r \ e v e r y } \ j \right\}
$$

be the Voronoi cell of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$

For adjacent cells $P _ { i }$ and $P _ { j }$ , meaning cells whose intersection contains a (d − 1)-dimensional face, let

$$
F _ { i j } : = P _ { i } \cap P _ { j } , \quad d _ { i j } : = \| { \pmb x } _ { i } - { \pmb x } _ { j } \| , \quad m _ { i j } : = \frac { { \pmb x } _ { i } + { \pmb x } _ { j } } { 2 } ,
$$

and let

$$
{ \pmb n } _ { i j } : = \frac { { \pmb x } _ { j } - { \pmb x } _ { i } } { d _ { i j } } , \quad { \pmb n } _ { j i } = - { \pmb n } _ { i j } .
$$

The face $F _ { i j }$ lies in the perpendicular-bisector hyperplane

$$
\begin{array} { r } { H _ { i j } : = { m _ { i j } + n _ { i j } ^ { \perp } } . } \end{array}
$$

Choose a linear isometry

$$
L _ { i j } : \mathbb { R } ^ { d - 1 }  \pmb { n } _ { i j } ^ { \perp }
$$

and define

$$
\begin{array} { r } { Q _ { i j } : = \left\{ \pmb { u } \in \mathbb { R } ^ { d - 1 } \ \middle | \ m _ { i j } + L _ { i j } \pmb { u } \in F _ { i j } \right\} . } \end{array}
$$

Thus, $Q _ { i j }$ is the common Voronoi face expressed in coordinates centered at the midpoint of $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { j } }$ Lemma B.5 (Voronoi-divergence reduction). For $d \in \{ 2 , 3 \}$ ,

$$
\mathcal { D } _ { \psi _ { d , 1 } , d } ( X ) = 1 + c _ { d } \sum _ { \{ i , j \} } d _ { i j } \int _ { Q _ { i j } } \frac { d \boldsymbol { u } } { \left( 1 + d _ { i j } ^ { 2 } / 4 + \| \boldsymbol { u } \| ^ { 2 } \right) ^ { d / 2 } } ,
$$

where

$$
c _ { d } : = \frac { \Gamma ( d / 2 ) } { 2 \pi ^ { d / 2 } } ,
$$

and the sum ranges over unordered adjacent pairs $\{ i , j \}$

Proof. Since the Voronoi cells cover $\mathbb { R } ^ { d }$ and their boundaries have d-dimensional Lebesgue measure zero,

$$
\mathcal { D } _ { \psi _ { d , 1 } , d } ( X ) = \sum _ { i = 1 } ^ { m } \int _ { P _ { i } } \psi _ { d , 1 } ( \| z - x _ { i } \| ) d z .
$$

Define

$$
\mathbf { F } _ { i } ( z ) : = c _ { d } \frac { z - { \pmb x } _ { i } } { \left( 1 + \| z - { \pmb x } _ { i } \| ^ { 2 } \right) ^ { d / 2 } } .
$$

The divergence identity from Appendix B.1, translated by ${ \mathbf { } } x _ { i } .$ , gives

$$
\nabla \cdot \mathbf { F } _ { i } ( z ) = \psi _ { d , 1 } ( \| z - x _ { i } \| ) .
$$

For suficiently large $R > 0$ , let $B _ { R } : = B _ { d } ( \mathbf { 0 } , R )$ , so that all points of $X$ lie in its interior. Each $P _ { i } \cap B _ { R }$ is a bounded convex region whose boundary is piecewise smooth, so the divergence theorem applies and gives

$$
\int _ { P _ { i } \cap B _ { R } } \psi _ { d , 1 } ( \| z - x _ { i } \| ) d z = \int _ { \partial ( P _ { i } \cap B _ { R } ) } \mathbf { F } _ { i } ( z ) \cdot n d S .
$$

Summing over i yields

$$
\int _ { B _ { R } } \psi _ { d , 1 } ( d _ { X } ( z ) ) d z = S _ { R } + \sum _ { \{ i , j \} } C _ { i j } ( R ) ,
$$

where the sum is over unordered pairs of adjacent Voronoi cells, meaning pairs whose intersection contains a (d − 1)-dimensional face.

The outer-boundary contribution is

$$
S _ { R } : = \sum _ { i = 1 } ^ { m } \int _ { P _ { i } \cap \partial B _ { R } } \mathbf { F } _ { i } ( z ) \cdot { \frac { z } { R } } d S
$$

and the contribution of a common Voronoi face is

$$
C _ { i j } ( R ) : = \int _ { { F _ { i j } \cap B _ { R } } } \left( \mathbf { F } _ { i } ( z ) \cdot { \pmb n } _ { i j } + { \bf F } _ { j } ( z ) \cdot { \pmb n } _ { j i } \right) d S .
$$

No general-position assumption is required here. Intersections of three or more distinct Voronoi cells have dimension at most $d - 2$ . Indeed, for distinct $i , j , k ,$

$$
P _ { i } \cap P _ { j } \cap P _ { k } \subseteq H _ { i j } \cap H _ { j k } ,
$$

where $H _ { i j }$ and $H _ { j k }$ are distinct perpendicular-bisector hyperplanes. If they were the same, then the hyperplane would reflect $x _ { j }$ to both $x _ { i }$ and $x _ { k } ,$ contradicting $x _ { i } \neq x _ { k }$ . Their intersection is therefore either empty or has dimension at most $d - 2$ . Such sets have zero $( d - 1 )$ )-dimensional measure and contribute no flux.

Outer boundary. Write $z = R w$ where $\lVert \boldsymbol { \mathbf { \mathit { w } } } \rVert = 1$ . Then,

$$
\begin{array} { r l } & { { R ^ { d - 1 } } \mathbf { F } _ { i } ( R \pmb { w } ) \cdot \pmb { w } = c _ { d } \frac { R ^ { d - 1 } \left( R - { \pmb x } _ { i } \cdot \pmb { w } \right) } { \left( 1 + \| R \pmb { w } - { \pmb x } _ { i } \| ^ { 2 } \right) ^ { d / 2 } } } \\ & { \qquad = c _ { d } \frac { 1 - { \pmb x } _ { i } \cdot \pmb { w } / R } { \left( 1 - \frac { 2 { \pmb x } _ { i } \cdot \pmb { w } } { R } + \frac { 1 + \| { \pmb x } _ { i } \| ^ { 2 } } { R ^ { 2 } } \right) ^ { d / 2 } } . } \end{array}
$$

Because the set of points is finite,

$$
R ^ { d - 1 } \mathbf { F } _ { i } ( R \pmb { w } ) \cdot \pmb { w } \longrightarrow c _ { d }
$$

uniformly in both i and w as $R \to \infty$

The sets $P _ { i } \cap \partial B _ { R }$ partition $\partial B _ { R }$ up to a set of surface measure zero. Under $z = R w$

$$
d S = R ^ { d - 1 } d \omega ,
$$

where $d \omega$ is the surface measure on the unit sphere. Hence

$$
\operatorname * { l i m } _ { R  \infty } S _ { R } = c _ { d } s _ { d - 1 } = 1 .
$$

Voronoi-face contribution. For every $z \in F _ { i j } , \| z - \pmb { x } _ { i } \| = \| z - \pmb { x } _ { j } \|$ . Therefore,

$$
\begin{array} { l } { { \displaystyle { \bf F } _ { i } ( z ) \cdot n _ { i j } + { \bf F } _ { j } ( z ) \cdot n _ { j i } = c _ { d } \left( \frac { z - x _ { i } } { ( 1 + \| z - x _ { i } \| ^ { 2 } ) ^ { d / 2 } } - \frac { z - x _ { j } } { ( 1 + \| z - x _ { j } \| ^ { 2 } ) ^ { d / 2 } } \right) \cdot n _ { i j } } } \\ { { = c _ { d } \displaystyle \frac { d _ { i j } } { ( 1 + \| z - x _ { i } \| ^ { 2 } ) ^ { d / 2 } } . } } \end{array}
$$

Thus,

$$
C _ { i j } ( R ) = c _ { d } d _ { i j } \int _ { F _ { i j } \cap B _ { R } } \frac { d S } { ( 1 + \| z - \pmb { x } _ { i } \| ^ { 2 } ) ^ { d / 2 } } .
$$

For

$$
z = m _ { i j } + L _ { i j } \pmb { u } ,
$$

we have

$$
m _ { i j } - x _ { i } = \frac { d _ { i j } } { 2 } n _ { i j } ,
$$

which is orthogonal to $\boldsymbol { L } _ { i j } \boldsymbol { u }$ . Therefore

$$
\| z - \pmb { x } _ { i } \| ^ { 2 } = \frac { d _ { i j } ^ { 2 } } { 4 } + \| \pmb { u } \| ^ { 2 } .
$$

As $R \to \infty$ , the sets $F _ { i j } \cap B _ { R }$ increase to $F _ { i j }$ . Since the integrand is nonnegative and $L _ { i j }$ preserves $( d - 1 )$ dimensional measure, the monotone convergence theorem gives

$$
\operatorname* { l i m } _ { R  \infty } C _ { i j } ( R ) = c _ { d } d _ { i j } \int _ { Q _ { i j } } \frac { d { \pmb u } } { ( 1 + d _ { i j } ^ { 2 } / 4 + \| { \pmb u } \| ^ { 2 } ) ^ { d / 2 } } .
$$

Finally,

$$
\int _ { B _ { R } } \psi _ { d , 1 } ( d _ { X } ( z ) ) d z \longrightarrow { \mathcal { D } } _ { \psi _ { d , 1 } , d } ( X )
$$

by monotone convergence. There are finitely many adjacent pairs, so we may pass to the limit term-by-term in the face sum. Combining the face limits with

$$
\operatorname* { l i m } _ { R \to \infty } S _ { R } = 1
$$

gives

$$
\mathcal { D } _ { \psi _ { d , 1 } , d } ( X ) = 1 + c _ { d } \sum _ { \{ i , j \} } d _ { i j } \int _ { Q _ { i j } } \frac { d \boldsymbol { u } } { \left( 1 + d _ { i j } ^ { 2 } / 4 + \| \boldsymbol { u } \| ^ { 2 } \right) ^ { d / 2 } } ,
$$

which proves Lemma B.5.

## B.4 Exact Computation in Two Dimensions

Proof of Theorem 4.4. By Lemma B.1, after rescaling the input it sufices to assume $\sigma = 1$ . By duplicate invariance, we may also remove duplicate points.

Let $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { m } )$ be the resulting pairwise distinct points. If $m = 1$ , then the GKD value is 1, so assume $m \geq 2$

Construct the planar Voronoi diagram of X. For each unordered adjacent pair $\{ i , j \}$ , let $E _ { i j } : = P _ { i } \cap P _ { j }$ be their common Voronoi edge and set $d _ { i j } : = \| \pmb { x } _ { i } - \pmb { x } _ { j } \|$

Since $\begin{array} { r } { c _ { 2 } = \frac { 1 } { 2 \pi } } \end{array}$ , Lemma B.5 gives

$$
\mathcal { D } _ { \psi _ { 2 , 1 } , 2 } ( X ) = 1 + \frac { 1 } { 2 \pi } \sum _ { \{ i , j \} } d _ { i j } \int _ { Q _ { i j } } \frac { d t } { 1 + d _ { i j } ^ { 2 } / 4 + t ^ { 2 } } .
$$

Because a Voronoi edge is a convex subset of its supporting line, $Q _ { i j }$ is an interval with ordered endpoints

$$
- \infty \leq a _ { i j } < b _ { i j } \leq + \infty
$$

where the interval may be bounded, a ray, or the entire line. Define

$$
A _ { i j } : = \sqrt { 1 + \frac { d _ { i j } ^ { 2 } } { 4 } } .
$$

Then,

$$
\int _ { Q _ { i j } } \frac { d t } { 1 + d _ { i j } ^ { 2 } / 4 + t ^ { 2 } } = \int _ { a _ { i j } } ^ { b _ { i j } } \frac { d t } { A _ { i j } ^ { 2 } + t ^ { 2 } } = \frac { 1 } { A _ { i j } } \left[ \arctan \left( \frac { t } { A _ { i j } } \right) \right] _ { a _ { i j } } ^ { b _ { i j } }
$$

where

$$
\arctan ( + \infty ) = { \frac { \pi } { 2 } } , \quad \arctan ( - \infty ) = - { \frac { \pi } { 2 } } .
$$

Hence the contribution of $E _ { i j }$ is

$$
\frac { d _ { i j } } { 2 \pi A _ { i j } } \left( \arctan \left( \frac { b _ { i j } } { A _ { i j } } \right) - \arctan \left( \frac { a _ { i j } } { A _ { i j } } \right) \right) .
$$

Therefore,

$$
\mathcal { D } _ { \psi _ { 2 , 1 } , 2 } ( X ) = 1 + \sum _ { \left\{ i , j \right\} } \frac { d _ { i j } } { 2 \pi A _ { i j } } \left( \arctan \left( \frac { b _ { i j } } { A _ { i j } } \right) - \arctan \left( \frac { a _ { i j } } { A _ { i j } } \right) \right) .
$$

A planar Voronoi diagram of m points has $O ( m )$ total combinatorial complexity and can be constructed in $O ( m \log m )$ time (Fortune, 1987). No general-position assumption is needed. Cocircular points may create Voronoi vertices incident to more than three cells, but the one-dimensional Voronoi edges are still segments or rays. If all points are collinear, the nonempty Voronoi edges are complete lines, which are handled by taking $a _ { i j } = - \infty$ and $b _ { i j } = + \infty$

Once the diagram is constructed, its geometric representation provides the endpoints or directions of every Voronoi edge. Relative to the coordinate system defining $Q _ { i j }$ in Appendix B.3, the corresponding finite endpoints or ray directions can be converted to $a _ { i j }$ and $b _ { i j }$ in $O ( 1 )$ elementary real-RAM operations per edge. Each edge contribution then requires only $O ( 1 )$ additional operations, so evaluating all contributions takes $O ( m ) = O ( n )$ time.

Including the duplicate removal and the $O ( n )$ rescaling from Lemma B.1, the total running time is $O ( n$ log n) for every $\sigma > 0$ . This proves Theorem 4.4. □

## B.5 Planar Face Integration

Lemma B.6 (Planar face integration). Let $Q \subseteq \mathbb { R } ^ { 2 }$ be a nonempty convex polygonal region with nonempty interior, possibly unbounded, whose boundary consists of k line segments, rays, or lines. For every $A > 0$

$$
I _ { A } ( Q ) : = \int _ { Q } { \frac { d \pmb { u } } { ( A ^ { 2 } + \| \pmb { u } \| ^ { 2 } ) ^ { 3 / 2 } } }
$$

can be evaluated using $O ( k + 1 )$ elementary real-RAM operations.

Proof. For bounded $Q ,$ the quantity $A I _ { A } ( Q )$ is the solid angle subtended by $Q .$ , embedded in a plane, from a point at perpendicular distance A. Thus, the bounded case could alternatively be evaluated by triangulating Q and using classical closed-form formulas for the solid angle of a triangle (van Oosterom and Strackee, 1983). We instead give a boundary formula that also handles unbounded regions.

Define

$$
\mathbf { G } _ { A } ( \pmb { u } ) : = \frac { \pmb { u } } { A \sqrt { A ^ { 2 } + \| \pmb { u } \| ^ { 2 } } \left( A + \sqrt { A ^ { 2 } + \| \pmb { u } \| ^ { 2 } } \right) } .
$$

Let

$$
r : = \| \pmb { u } \| , \quad s : = \sqrt { A ^ { 2 } + r ^ { 2 } } ,
$$

and write

$$
\mathbf { G } _ { A } ( \pmb { u } ) = \pmb { u } g ( r ) , \quad g ( r ) : = \frac { 1 } { A s ( A + s ) } .
$$

For a radial scalar function g in two dimensions,

$$
\nabla \cdot ( \boldsymbol { u } \boldsymbol { g } ( \boldsymbol { r } ) ) = 2 \boldsymbol { g } ( \boldsymbol { r } ) + \boldsymbol { r } \boldsymbol { g } ^ { \prime } ( \boldsymbol { r } ) .
$$

Since

$$
g ^ { \prime } ( r ) = - \frac { r ( 2 s + A ) } { A s ^ { 3 } ( A + s ) ^ { 2 } } = - \frac { r ( 2 s + A ) } { s ^ { 2 } ( A + s ) } g ( r ) ,
$$

we obtain

$$
\begin{array} { l } { \nabla \cdot \mathbf { G } _ { A } ( u ) = g ( r ) \left( 2 - \frac { r ^ { 2 } ( 2 s + A ) } { s ^ { 2 } ( A + s ) } \right) } \\ { = g ( r ) \left( 2 - \frac { ( s - A ) ( 2 s + A ) } { s ^ { 2 } } \right) } \\ { = g ( r ) \frac { A ( A + s ) } { s ^ { 2 } } } \\ { = \displaystyle \frac { 1 } { s ^ { 3 } } = \frac { 1 } { ( A ^ { 2 } + \| u \| ^ { 2 } ) ^ { 3 / 2 } } . } \end{array}
$$

For $R > 0$ , let

$$
\begin{array} { r } { Q _ { R } : = Q \cap B _ { 2 } ( \mathbf { 0 } , R ) . } \end{array}
$$

For all suficiently large $R , \ Q _ { R }$ is a nonempty bounded convex region with piecewise smooth boundary. The divergence theorem gives

$$
\int _ { Q _ { R } } \frac { d \pmb { u } } { ( A ^ { 2 } + \| \pmb { u } \| ^ { 2 } ) ^ { 3 / 2 } } = \int _ { \partial Q _ { R } } \mathbf { G } _ { A } ( \pmb { u } ) \cdot \pmb { n } ( \pmb { u } ) d s .
$$

Up to finitely many intersection points, $\partial Q _ { R }$ consists of portions of the straight boundary pieces of $Q$ together with circular arcs on $\partial B _ { 2 } ( { \bf 0 } , R )$

Straight boundary pieces. Let E be a straight boundary piece of $Q ,$ which may be a segment, ray, or complete line. Let $\mathbf { \nabla } _ { \mathbf { \Omega } ^ { n } E }$ be its outward unit normal and choose a unit tangent vector $\tau _ { E }$ perpendicular to $\mathbf { \nabla } _ { \mathbf { \Omega } ^ { n } E }$ The supporting line of E has an equation

$$
\pmb { n } _ { E } \cdot \pmb { u } = c _ { E }
$$

for some constant $c _ { E } \in \mathbb { R }$ , so we may parameterize

$$
\begin{array} { r } { \pmb { u } _ { E } ( t ) = c _ { E } \pmb { n } _ { E } + t \pmb { \tau } _ { E } , \quad t \in { \cal J } _ { E } , } \end{array}
$$

where $J _ { E }$ is an interval whose endpoints $a _ { E } , b _ { E }$ lie in $\mathbb { R } \cup \{ \pm \infty \}$

Since

$$
\| \pmb { u } _ { E } ( t ) \| ^ { 2 } = c _ { E } ^ { 2 } + t ^ { 2 } , \quad \pmb { u } _ { E } ( t ) \cdot \pmb { n } _ { E } = c _ { E } ,
$$

we have

$$
\mathbf { G } _ { A } ( \pmb { u } _ { E } ( t ) ) \cdot \pmb { n } _ { E } = \frac { c _ { E } } { A \sqrt { A ^ { 2 } + c _ { E } ^ { 2 } + t ^ { 2 } } \left( A + \sqrt { A ^ { 2 } + c _ { E } ^ { 2 } + t ^ { 2 } } \right) } .
$$

If $c _ { E } = 0$ , this flux vanishes identically. For $c _ { E } \neq 0$ , define

$$
\Phi _ { A , c _ { E } } ( t ) : = \frac { c _ { E } } { A | c _ { E } | } \left[ \arctan \left( \frac { t } { | c _ { E } | } \right) - \arctan \left( \frac { A t } { | c _ { E } | \sqrt { A ^ { 2 } + c _ { E } ^ { 2 } + t ^ { 2 } } } \right) \right] .
$$

Direct diferentiation gives

$$
\Phi _ { A , c _ { E } } ^ { \prime } ( t ) = \mathbf { G } _ { A } ( \pmb { u } _ { E } ( t ) ) \cdot \pmb { n } _ { E } .
$$

For all suficiently large $R ,$ the portion of $E$ contained in $\partial Q _ { R }$ is parameterized by

$$
J _ { E , R } = J _ { E } \cap \left[ - \sqrt { R ^ { 2 } - c _ { E } ^ { 2 } } , \sqrt { R ^ { 2 } - c _ { E } ^ { 2 } } \right] .
$$

Let ${ a } _ { E , R }$ and $b _ { E , R }$ denote its endpoints. As $R \to \infty$ ，

$$
a _ { E , R }  a _ { E } , \quad b _ { E , R }  b _ { E }
$$

in the extended real line, so the limiting flux contribution of E is

$$
\Phi _ { A , c _ { E } } ( b _ { E } ) - \Phi _ { A , c _ { E } } ( a _ { E } ) .
$$

For infinite endpoints,

$$
\Phi _ { A , c _ { E } } ( + \infty ) = \frac { c _ { E } } { A | c _ { E } | } \left( \frac { \pi } { 2 } - \arctan \frac { A } { | c _ { E } | } \right) , \quad \Phi _ { A , c _ { E } } ( - \infty ) = - \Phi _ { A , c _ { E } } ( + \infty ) .
$$

Thus, each straight boundary piece contributes a constant-size elementary expression.

Boundary at infinity. Write

$$
{ \pmb w } ( \theta ) : = ( \cos \theta , \sin \theta )
$$

and define

$$
\begin{array} { r } { \Theta _ { R } : = \left\{ \theta \in \left[ 0 , 2 \pi \right) \vert \ R \pmb { w } ( \theta ) \in Q \right\} . } \end{array}
$$

On $\partial B _ { 2 } ( { \bf 0 } , R )$ , the outward normal of $Q _ { R }$ along its circular boundary is ${ \pmb w } ( \theta )$ and $d s = R d \theta$ . Hence the total circular flux equals

$$
\frac { R ^ { 2 } } { A \sqrt { A ^ { 2 } + R ^ { 2 } } \left( A + \sqrt { A ^ { 2 } + R ^ { 2 } } \right) } | \Theta _ { R } | .
$$

The scalar factor converges to $1 / A$ as $R \to \infty$

Define the recession cone of $Q$ by

$$
\operatorname { r e c } ( Q ) : = \left\{ w \in \mathbb { R } ^ { 2 } \mid u + \lambda w \in Q { \mathrm { ~ f o r ~ e v e r y ~ } } u \in Q , \ \lambda \geq 0 \right\}
$$

and let

$$
\beta ( Q ) : = | \{ \theta \in [ 0 , 2 \pi ) \mid { \pmb w } ( \theta ) \in \mathrm { r e c } ( Q ) \} | .
$$

We claim that

$$
| \Theta _ { R } | \longrightarrow \beta ( Q ) .
$$

Write $Q$ as the intersection of finitely many half-planes,

$$
Q = \bigcap _ { i = 1 } ^ { p } \left\{ { \pmb u } \in \mathbb { R } ^ { 2 } \ | \ n _ { i } \cdot { \pmb u } \leq h _ { i } \right\} .
$$

Then

$$
R { \pmb w } ( \theta ) \in Q \iff { \pmb n } _ { i } \cdot { \pmb w } ( \theta ) \leq \frac { h _ { i } } { R } { \mathrm { ~ f o r ~ e v e r y ~ } } i .
$$

For every direction satisfying

$$
\begin{array} { r } { \pmb { n } _ { i } \cdot \pmb { w } ( \theta ) \neq 0 \mathrm { ~ f o r ~ e v e r y ~ } i , } \end{array}
$$

these inequalities eventually have the same truth values as

$$
n _ { i } \cdot w ( \theta ) \leq 0 { \mathrm { ~ f o r ~ e v e r y ~ } } i ,
$$

which is precisely the condition ${ \pmb w } ( \theta ) \in \mathrm { r e c } ( Q )$ . Only finitely many directions satisfy $\pmb { n } _ { i } \cdot \pmb { w } ( \theta ) = 0$ for some $i ,$ so the exceptional set has angular measure zero. Thus,

$$
{ \mathbf { 1 } } _ { \theta \in \Theta _ { R } } \longrightarrow { \mathbf { 1 } } _ { w ( \theta ) \in \mathrm { r e c } ( Q ) }
$$

for almost every θ. Dominated convergence yields

$$
| \Theta _ { R } | \longrightarrow \beta ( Q ) .
$$

Consequently, the limiting circular-boundary contribution is

$$
{ \frac { \beta ( Q ) } { A } } .
$$

Since $Q _ { R }$ increases to $Q$ and the integrand is nonnegative, the monotone convergence theorem gives

$$
I _ { A } ( Q _ { R } ) \longrightarrow I _ { A } ( Q ) .
$$

Combining the straight-boundary and circular-boundary limits yields

$$
I _ { A } ( Q ) = \sum _ { E \in \mathcal { E } ( Q ) } \left( \Phi _ { A , c _ { E } } ( b _ { E } ) - \Phi _ { A , c _ { E } } ( a _ { E } ) \right) + \frac { \beta ( Q ) } { A } ,
$$

where $\mathcal { E } ( Q )$ denotes the set of straight boundary pieces of $Q ,$ and the summand is interpreted as zero when $c _ { E } = 0 $

It remains to compute $\beta ( Q )$ . From the polygonal boundary representation, the possibilities are as follows:

$\mathrm { I f } \ Q$ is bounded, then $\beta ( Q ) = 0$

• If $Q = \mathbb { R } ^ { 2 }$ , then $\beta ( Q ) = 2 \pi$

• If Q is a half-plane, then $\beta ( Q ) = \pi$

• If Q is a strip between two parallel lines, then $\operatorname { r e c } ( Q )$ is a line and $\beta ( Q ) = 0$

• In every remaining unbounded case, the unbounded polygonal boundary has two rays. Let $\mathbf { \Delta } _ { \mathbf { r } _ { 1 } }$ and $\mathbf { \boldsymbol { r } } _ { 2 }$ be unit vectors in their recession directions. Then,

$$
\operatorname { r e c } ( Q ) = \left\{ \lambda _ { 1 } r _ { 1 } + \lambda _ { 2 } r _ { 2 } \mid \lambda _ { 1 } , \lambda _ { 2 } \geq 0 \right\} ,
$$

and $\beta ( Q ) = \operatorname { a r c c o s } ( \pmb { r } _ { 1 } \cdot \pmb { r } _ { 2 } )$

These cases and the corresponding recession directions can be identified in one scan of the k boundary pieces. Thus, $\beta ( Q )$ can be computed in $O ( k + 1 )$ elementary real-RAM operations. Each straight boundary contribution requires O(1) operations, so the entire integral $I _ { A } ( Q )$ can be evaluated using $O ( k + 1 )$ operations. □

## B.6 Exact Computation in Three Dimensions

Proof of Theorem 4.5. By Lemma B.1, after rescaling the input it sufices to assume $\sigma = 1$ . By duplicate invariance, we may also remove duplicate points.

If only one distinct point remains, then the GKD value is 1. Hence assume that

$$
X = ( x _ { 1 } , \ldots , x _ { m } )
$$

contains $m \geq 2$ pairwise distinct points.

Construct the three-dimensional Voronoi diagram of X. For every unordered adjacent pair $\{ i , j \}$ , let $F _ { i j } : = P _ { i } \cap P _ { j }$ denote their common two-dimensional Voronoi face and set $d _ { i j } : = \| \pmb { x } _ { i } - \pmb { x } _ { j } \|$ . Since

$$
c _ { 3 } = { \frac { \Gamma ( 3 / 2 ) } { 2 \pi ^ { 3 / 2 } } } = { \frac { 1 } { 4 \pi } } ,
$$

Lemma B.5 gives

$$
\mathcal { D } _ { \psi _ { 3 , 1 } , 3 } \big ( X \big ) = 1 + \frac { 1 } { 4 \pi } \sum _ { \{ i , j \} } d _ { i j } \int _ { Q _ { i j } } \frac { d \boldsymbol { u } } { \big ( 1 + d _ { i j } ^ { 2 } / 4 + \| \boldsymbol { u } \| ^ { 2 } \big ) ^ { 3 / 2 } } ,
$$

where $Q _ { i j } \subseteq \mathbb { R } ^ { 2 }$ is the planar representation of $F _ { i j }$ introduced in Appendix B.3.

For each face, define

$$
A _ { i j } : = \sqrt { 1 + \frac { d _ { i j } ^ { 2 } } { 4 } } .
$$

Then its integral is

$$
I _ { A _ { i j } } ( Q _ { i j } ) = \int _ { Q _ { i j } } \frac { d \pmb { u } } { ( A _ { i j } ^ { 2 } + \| \pmb { u } \| ^ { 2 } ) ^ { 3 / 2 } } .
$$

Relative to the planar coordinate system defining $Q _ { i j }$ in Appendix B.3, the vertices and recession directions of $F _ { i j }$ can be converted to the corresponding line segments, rays, or lines of $Q _ { i j }$ using $O ( k _ { i j } + 1 )$ elementary real-RAM operations. Hence, after constructing the Voronoi diagram, the total coordinate-conversion and integral cost is

$$
O \left( | { \mathcal { F } } | + \sum _ { F \in { \mathcal { F } } } k ( F ) \right) ,
$$

where $\mathcal { F }$ denotes the set of two-dimensional Voronoi faces and $k ( F )$ is the number of one-dimensional boundary pieces of $F .$

We now bound the incidence term

$$
\sum _ { F \in { \mathcal { F } } } k ( F ) .
$$

For each Voronoi cell $P _ { i }$ , let $e ( P _ { i } )$ denote the number of its one-dimensional boundary pieces, including bounded edges, rays, and lines. Since $P _ { i }$ is the intersection of at most $m - 1 \leq n - 1$ half-spaces, it has at most $n - 1$ two-dimensional faces.

We claim that $e ( P _ { i } ) = O ( n )$ for every i. If $P _ { i }$ is bounded, this follows from the standard planar-graph bound for the boundary graph of a three-dimensional convex polytope. If $P _ { i }$ is unbounded, truncate it by a suficiently large cube C centered at the origin, chosen so that all bounded vertices and edges of $P _ { i }$ lie in the interior of $C$ and no one-dimensional boundary piece of $P _ { i }$ is contained in $\partial C$ . Such a choice is possible because only finitely many cube sizes violate the latter condition. The polytope

$$
P _ { i } ^ { \prime } : = P _ { i } \cap C
$$

has at most $n - 1 + 6 = n + 5$ faces, and every one-dimensional boundary piece of $P _ { i }$ is preserved as an edge of $P _ { i } ^ { \prime }$ , possibly after truncation. Since a three-dimensional convex polytope with $f$ faces has at most $3 f - 6$ edges, $e \dot { ( } P _ { i } ) \le e ( P _ { i } ^ { \prime } ) = O ( n )$

For a fixed cell $P _ { i } .$ , every one-dimensional piece is incident to exactly two of its two-dimensional faces. Therefore,

$$
\sum _ { F \in { \mathcal { F } } } k ( F ) = 2 e ( P _ { i } ) .
$$

Every two-dimensional Voronoi face is incident to exactly two Voronoi cells. Summing the preceding identity over all cells thus gives

$$
2 \sum _ { F \in \mathcal { F } } k ( F ) = 2 \sum _ { i = 1 } ^ { m } e ( P _ { i } ) ,
$$

and hence

$$
\sum _ { F \in { \mathcal { F } } } k ( F ) = \sum _ { i = 1 } ^ { m } e ( P _ { i } ) = O ( n ^ { 2 } ) .
$$

Also,

$$
| { \mathcal { F } } | \leq { \binom { m } { 2 } } = O ( n ^ { 2 } ) ,
$$

since a pair of convex Voronoi cells shares at most one two-dimensional face. Consequently,

$$
| { \mathcal { F } } | + \sum _ { F \in { \mathcal { F } } } k ( F ) \in O ( n ^ { 2 } ) ,
$$

so all face integrals can be evaluated in $O ( n ^ { 2 } )$ elementary real-RAM operations once the Voronoi diagram is available.

A three-dimensional Voronoi diagram can be constructed in $O ( n ^ { 2 } )$ time in the real-RAM model (Chazelle, 1993). Therefore, including Voronoi construction, duplicate removal, and the initial rescaling from Lemma B.1, the total running time is $O ( n ^ { 2 } )$ for every $\sigma > 0$ □

## C RANDOMIZED APPROXIMATION

## C.1 The Self-Adjusting Union-Volume Estimator

We use the self-adjusting coverage estimator of Karp et al. (1989). Their algorithm and analysis apply to unions of finite sets. Bringmann and Friedrich (2010) adapted the same approach to unions of measurable geometric bodies using volume, sampling, and membership oracles, and further allowed these oracles to be approximate. For completeness, we state the exact-oracle version needed here, retaining the arbitrary failure probability δ from the analysis of Karp et al. (1989).

Let $( \Omega , \Sigma , \mu )$ be a measure space, and let $B _ { 1 } , \ldots , B _ { m } \in \Sigma$ satisfy $0 < \mu ( B _ { i } ) < \infty$ for every $i \in \{ 1 , \ldots , m \}$ . Define

$$
V : = \mu \left( \bigcup _ { i = 1 } ^ { m } B _ { i } \right) , \quad W : = \sum _ { i = 1 } ^ { m } \mu ( B _ { i } ) .
$$

Assume access to the following operations:

1. compute $\mu ( B _ { i } )$ ;

2. sample a point y from $B _ { i }$ according to

$$
\operatorname { P r } [ \operatorname { y } \in A ] = { \frac { \mu ( A \cap B _ { i } ) } { \mu ( B _ { i } ) } }
$$

for every measurable $A \subseteq \Omega ;$

3. given $y \in \Omega$ and $i \in \{ 1 , \ldots , m \}$ , determine whether $y \in B _ { i }$

A trial first chooses an index $I \in \{ 1 , \ldots , m \}$ according to

$$
\operatorname* { P r } [ I = i ] = \frac { \mu ( B _ { i } ) } { W }
$$

and then samples $_ \mathrm { y }$ from the normalized restriction of $\mu$ to $B _ { I }$ . It next repeatedly chooses independent indices

$$
J \sim \operatorname { U n i f } ( \{ 1 , \dots , m \} )
$$

and tests whether $\mathrm { y } \in B _ { J }$ , ending the trial at the first successful query.

For $\epsilon , \delta \in ( 0 , 1 )$ , define the membership-query budget

$$
B : = \left\lceil \frac { 8 ( 1 + \epsilon ) m \log ( 2 / \delta ) } { \epsilon ^ { 2 } } \right\rceil .
$$

Run consecutive trials and stop after the B-th membership query. Let $N _ { B }$ denote the number of trials completed within these $B$ queries; an incomplete final trial is not counted. Define

$$
\widehat { V } : = \left\{ \begin{array} { l l } { \displaystyle \frac { B W } { m N _ { B } } , } & { ( N _ { B } > 0 ) , } \\ { W , } & { ( N _ { B } = 0 ) . } \end{array} \right.
$$

Lemma C.1 (Self-adjusting union-volume estimator). Under the assumptions above,

$$
\begin{array} { r } { \operatorname* { P r } \left[ ( 1 - \epsilon ) V \le \widehat { V } \le ( 1 + \epsilon ) V \right] \ge 1 - \delta . } \end{array}
$$

Proof. For $y \in \Omega ,$ , let

$$
c ( \boldsymbol { y } ) : = \sum _ { i = 1 } ^ { m } \mathbf { 1 } _ { \boldsymbol { y } \in B _ { i } } ,
$$

and for $k \in \{ 1 , \ldots , m \}$ define

$$
R _ { k } : = \{ y \in \Omega \mid c ( y ) = k \} , \quad r _ { k } : = \mu ( R _ { k } ) .
$$

Since the solids $B _ { i }$ are measurable, so are c and the sets $R _ { k }$ . Also, from the definition above,

$$
V = \sum _ { k = 1 } ^ { m } r _ { k } , \quad W = \int _ { \Omega } c ( y ) d \mu ( y ) = \sum _ { k = 1 } ^ { m } k r _ { k } .
$$

Moreover, the point y sampled at the beginning of a trial satisfies

$$
\mathrm { P r } [ \mathrm { y } \in R _ { k } ] = \sum _ { i = 1 } ^ { m } \frac { \mu ( B _ { i } ) } { W } \frac { \mu ( R _ { k } \cap B _ { i } ) } { \mu ( B _ { i } ) } = \frac { k r _ { k } } { W } .
$$

Conditional on $\mathrm { ~ y ~ } \in \ R _ { k }$ , each membership query succeeds independently with probability $k / m$ , and hence the trial length satisfies

$$
L \mid ( \mathrm { y } \in R _ { k } ) \sim \operatorname { G e o m } \left( { \frac { k } { m } } \right)
$$

where Geom(p) counts the number of membership queries through the first success.

These are exactly the coverage classes and conditional trial-length distributions used by Karp et al. (1989). The same replacement of finite cardinalities by volumes is made explicitly by Bringmann and Friedrich (2010) in their continuous-body formulation. Thus, the self-adjusting coverage analysis applies to the present exact-oracle setting. In particular, Theorem I of Karp et al. (1989) gives the stated fixed-budget guarantee.

For the ceiling in our definition of $B ,$ , define

$$
\delta _ { B } : = 2 \exp \left( - \frac { \epsilon ^ { 2 } B } { 8 ( 1 + \epsilon ) m } \right) .
$$

By the definition of B, we have $\delta _ { B } \leq \delta$ , and

$$
B = \frac { 8 ( 1 + \epsilon ) m \log ( 2 / \delta _ { B } ) } { \epsilon ^ { 2 } } .
$$

Applying Theorem I of Karp et al. (1989) with failure probability $\delta _ { B }$ therefore gives

$$
\operatorname* { P r } \bigg [ N _ { B } > 0 \mathrm { ~ a n d ~ } ( 1 - \epsilon ) V \le \frac { B W } { m N _ { B } } \le ( 1 + \epsilon ) V \bigg ] \ge 1 - \delta _ { B } \ge 1 - \delta .
$$

On this event, $\widehat { V } = B W / m N _ { B } .$ , which proves the claimed probability bound. The fallback value for $N _ { B } = 0$ is used only outside this success event and therefore does not afect the guarantee. □

## C.2 Sampling and Membership Oracles for Canonical Solids

For $X = ( \pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } ) \in ( \mathbb { R } ^ { d } ) ^ { n }$ and $\sigma > 0$ , define

$$
S _ { i } : = S _ { \psi _ { d , \sigma } , d } ( \pmb { x } _ { i } ) = \left. ( \pmb { z } , h ) \in \mathbb { R } ^ { d } \times \mathbb { R } _ { \geq 0 } \ \middle | \ 0 \leq h \leq \psi _ { d , \sigma } ( \Vert \pmb { z } - \pmb { x } _ { i } \Vert ) \right. .
$$

We now construct the volume, membership, and uniform-sampling oracles required by Lemma C.1.

Lemma C.2 (Canonical-solid oracles). Under the randomized elementary real-RAM model of Section 5, each canonical solid $S _ { i }$ supports the following operations:

1. its volume can be returned in O(1) time;

2. membership of a point $( z , h ) \in \mathbb { R } ^ { d + 1 }$ can be tested in $O ( d )$ time;

3. a point can be sampled exactly from the normalized Lebesgue measure on $S _ { i }$ in $O ( d )$ time.

Proof. By normalization of the canonical kernel,

$$
\operatorname { V o l } _ { d + 1 } ( S _ { i } ) = 1 ,
$$

so the volume oracle is immediate.

For membership,

$$
( z , h ) \in S _ { i } \iff 0 \leq h \leq \psi _ { d , \sigma } ( \| z - x _ { i } \| ) .
$$

Computing $\| \boldsymbol { z } - \boldsymbol { x } _ { i } \| ^ { 2 }$ takes $O ( d )$ arithmetic operations. Since d is an integer, the integer or half-integer powers and normalization constant in $\psi _ { d , \sigma }$ can also be evaluated in $O ( d )$ elementary real-RAM operations. Thus, membership can be tested in $O ( d )$ time.

It remains to construct an exact uniform sampler.

Sampling the under-graph solid. First consider the solid centered at the origin,

$$
S _ { 0 } = S _ { \psi _ { d , \sigma } , d } ( \mathbf { 0 } ) .
$$

Suppose z is sampled from the probability density

$$
p ( z ) = \psi _ { d , \sigma } ( \| z \| ) ,
$$

and, conditional on z, independently sample

$$
H \sim \operatorname { U n i f } ( 0 , \psi _ { d , \sigma } ( \| z \| ) ) .
$$

The joint density of $( z , H )$ on $S _ { 0 }$ is

$$
\psi _ { d , \sigma } ( \| z \| ) \cdot \frac { 1 } { \psi _ { d , \sigma } ( \| z \| ) } = 1 .
$$

Since $\mathrm { V o l } _ { d + 1 } ( S _ { 0 } ) = 1$ , this is exactly the uniform probability distribution on $S _ { 0 }$ . Thus, it remains only to sample z from the canonical density.

Sampling the canonical density. Recall from Appendix B.1 the radial map

$$
T ( \pmb { u } ) = \frac { \pmb { u } } { \sqrt { \sigma ^ { 2 } + \| \pmb { u } \| ^ { 2 } } } , \quad T ^ { - 1 } ( \pmb { y } ) = \frac { \sigma \pmb { y } } { \sqrt { 1 - \| \pmb { y } \| ^ { 2 } } }
$$

By Lemma B.4,

$$
| \mathrm { d e t } D T ( { \pmb u } ) | = \frac { 1 } { \sigma ^ { d } } \left( 1 + \frac { \| { \pmb u } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ^ { - ( d / 2 + 1 ) } .
$$

Since

$$
\psi _ { d , \sigma } ( \| { \pmb u } \| ) = \frac { \Gamma ( d / 2 + 1 ) } { \pi ^ { d / 2 } } \left| \operatorname* { d e t } D T ( \pmb u ) \right| = \frac { 1 } { v _ { d } } \left| \operatorname* { d e t } D T ( \pmb u ) \right| ,
$$

where

$$
v _ { d } = \mathrm { V o l } _ { d } ( B _ { d } ( { \bf 0 } , 1 ) ) = \frac { \pi ^ { d / 2 } } { \Gamma ( d / 2 + 1 ) } ,
$$

the change of variables formula shows that T maps the canonical density to the uniform probability distribution on the unit ball. Consequently, if

$$
\pmb { y } \sim \mathrm { U n i f } ( B _ { d } ( \mathbf { 0 } , 1 ) )
$$

then

$$
z = { \frac { \sigma y } { \sqrt { 1 - \| y \| ^ { 2 } } } }
$$

has density $\psi _ { d , \sigma } ( | | z | | )$

To sample y exactly, draw $G _ { 1 } , \dots , G _ { d + 2 } \overset { \mathrm { i . i . d } } { \sim } \mathcal { N } ( 0 , 1 )$ and set

$$
{ \pmb y } = \frac { ( G _ { 1 } , \dots , G _ { d } ) } { \sqrt { G _ { 1 } ^ { 2 } + \dots + G _ { d + 2 } ^ { 2 } } } .
$$

Normalizing a standard Gaussian vector gives a uniform point on $S ^ { d + 1 }$ (Muller, 1959), and projecting onto its first d coordinates gives the uniform distribution on $B _ { d } ( \mathbf { 0 } , 1 )$ (Mijatovi´c et al., 2018).

Therefore, an exact sampler for $S _ { i }$ is:

1. sample $G _ { 1 } , \dots , G _ { d + 2 } \stackrel { \mathrm { i . i . d } } { \sim } \mathcal { N } ( 0 , 1 )$ and set

$$
{ \pmb y } = \frac { ( G _ { 1 } , \dots , G _ { d } ) } { \sqrt { G _ { 1 } ^ { 2 } + \dots + G _ { d + 2 } ^ { 2 } } } ;
$$

2. set

$$
z = { \frac { \sigma y } { \sqrt { 1 - \| y \| ^ { 2 } } } } ;
$$

3. independently sample $U \sim \mathrm { U n i f } ( 0 , 1 )$ and set

$$
H : = U \psi _ { d , \sigma } ( \| z \| ) ;
$$

4. return $( { \pmb x } _ { i } + z , H )$

The denominator in the second step is positive almost surely because the boundary of the unit ball has probability zero.

Generating and processing the $d + 2$ Gaussian variables, applying $T ^ { - 1 }$ , evaluating the kernel, sampling $U ,$ and translating by $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ require $O ( d )$ elementary real-RAM operations. Hence the complete sampler runs in $O ( d )$ time. 口

## C.3 Proof of the Randomized Approximation Theorem

Proof of Theorem 5.1. For each $i \in \{ 1 , \ldots , n \}$ , let

$$
S _ { i } : = S _ { \psi _ { d , \sigma } , d } ( \pmb { x } _ { i } ) .
$$

By Proposition 3.3,

$$
{ \mathcal { D } } _ { \psi _ { d , \sigma } , d } ( X ) = { \mathrm { V o l } } _ { d + 1 } \left( \bigcup _ { i = 1 } ^ { n } S _ { i } \right) ,
$$

and each $S _ { i }$ has volume 1. By Lemma C.2, these solids provide the exact volume, uniform-sampling, and membership oracles required by Lemma C.1.

Apply that lemma to

$$
( \Omega , \Sigma , \mu ) = \left( \mathbb { R } ^ { d + 1 } , { \boldsymbol { B } } ( \mathbb { R } ^ { d + 1 } ) , \operatorname { V o l } _ { d + 1 } \right)
$$

with $B _ { i } = S _ { i }$ . In its notation,

$$
m = n , \quad W = \sum _ { i = 1 } ^ { n } { \mathrm { V o l } } _ { d + 1 } ( S _ { i } ) = n , \quad V = { \mathcal { D } } _ { \psi _ { d , \sigma } , d } ( X ) .
$$

Hence its estimator specializes to

$$
\widehat { V } = \left\{ \frac { B } { N _ { B } } , \ : \ : \ : ( N _ { B } > 0 ) , \right.
$$

which is exactly the estimator $\widehat { D }$ defined in Section 5. Therefore, Lemma C.1 gives

$$
\begin{array} { r } { \operatorname* { P r } \left[ \Big | \widehat { D } - \mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) \Big | \leq \epsilon \mathcal { D } _ { \psi _ { d , \sigma } , d } ( X ) \right] \geq 1 - \delta . } \end{array}
$$

It remains to bound the running time. By Lemma C.2, sampling uniformly from a canonical solid and performing a membership query take $O ( d )$ time. Because all solids have volume 1, the initial index of each trial is uniform

on $\{ 1 , \ldots , n \}$ . The membership-query indices are also uniform by construction. These indices can be sampled in O(1) time in the randomized real-RAM model.

The algorithm performs exactly B membership queries. Since every initiated trial performs at least one membership query, at most B trials are initiated, and hence at most B uniform-solid samples are generated. Thus, the total running time is

$$
O ( B d ) = O \left( \frac { n d } { \epsilon ^ { 2 } } \log \frac { 2 } { \delta } \right) ,
$$

which proves the theorem.

## D ADDITIONAL AXIOM-STRESS-TEST DETAILS

## D.1 Comparison Measures and Implementation

Comparison measures. For a configuration $X ~ = ~ ( x _ { 1 } , \ldots , x _ { n } )$ , we compare GKD with average pairwise distance, the Vendi Score (Friedman and Dieng, 2023), determinant-based DPP diversity (Mironov and Prokhorenkova, 2025; Wilhelm et al., 2018), and R´enyi kernel entropy (RKE) (Jalali et al., 2023).

For the similarity-based measures, we use the RBF similarity matrix

$$
K _ { i j } = \exp \left( - \frac { \| x _ { i } - x _ { j } \| ^ { 2 } } { 2 h ^ { 2 } } \right) .
$$

Average pairwise distance is

$$
D _ { \mathrm { A P D } } ( X ) = \frac { 2 } { n ( n - 1 ) } \sum _ { i < j } \| x _ { i } - x _ { j } \| .
$$

The determinant-based DPP score is

$$
D _ { \mathrm { D P P } } ( X ) = \operatorname* { d e t } K .
$$

Because $K _ { i i } = 1$ , we have tr $K = n$ . Let $\lambda _ { 1 } , \ldots , \lambda _ { n }$ denote the eigenvalues of $K / n$ . The Vendi Score is

$$
D _ { \mathrm { V e n d i } } ( X ) = \exp \left( - \sum _ { i = 1 } ^ { n } \lambda _ { i } \log \lambda _ { i } \right) ,
$$

with the convention 0 log $0 = 0$ . The order-two R´enyi kernel entropy (RKE) is

$$
D _ { \mathrm { R K E } } ( X ) = - \log \left( \frac { 1 } { n ^ { 2 } } \sum _ { i , j } K _ { i j } ^ { 2 } \right) = - \log \mathrm { t r } \Bigg ( \Bigg ( \frac { K } { n } \Bigg ) ^ { 2 } \Bigg ) .
$$

All three similarity-based measures use the same RBF bandwidth within a given experiment.

Implementation and reproducibility. All experimental code is implemented in Python. Exact software dependencies, package versions, and the computing environment used for the final reproducibility audit are provided in the supplementary material. Our empirical results concern diversity values and statistical comparison frequencies rather than wall-clock performance, and we do not make hardware-dependent runtime comparisons.

## D.2 Controlled Euclidean Stress Tests

This appendix gives the explicit constructions and numerical endpoint values for the controlled examples in Section 6.2. These examples isolate the monotonicity and uniqueness axioms in simple low-dimensional Euclidean configurations. Figure 2 illustrates the monotonicity construction and plots the GKD and Vendi Score behavior for both examples.

Throughout this subsection, GKD uses the canonical kernel with $\sigma = 1$ , and the similarity-based measures use bandwidth $h = 1$ . GKD is evaluated exactly, so these comparisons contain no Monte Carlo error.

Table 1: Numerical endpoint behavior in the controlled axiom stress tests. Relative change is computed from the first endpoint to the second. For DPP in the uniqueness example, both endpoint values are zero, so we report 0% to indicate no change.
<table><tr><td>Measure</td><td>Initial</td><td>Final</td><td>Relative change</td></tr><tr><td colspan="3">Monotonicity:  $\overline { { c = 1 . 7 6 \to 1 . 9 9 } }$ </td><td></td></tr><tr><td>GKD</td><td>1.88847</td><td>1.89442</td><td>+0.315%</td></tr><tr><td>Average pairwise distance</td><td>1.25333</td><td>1.33000</td><td>+6.12%</td></tr><tr><td>Vendi Score</td><td>2.29378</td><td>2.28758</td><td>-0.270%</td></tr><tr><td>DPP</td><td>0.37543</td><td>0.34676</td><td>-7.64%</td></tr><tr><td>RKE</td><td>0.67950</td><td>0.69101</td><td>+1.69%</td></tr><tr><td colspan="3">Uniqueness:  $\overline { { t = 0 \to 1 } }$ </td><td></td></tr><tr><td>GKD</td><td>1.24254</td><td>1.24807</td><td>+0.445%</td></tr><tr><td>Average pairwise distance</td><td>0.33333</td><td>0.29167</td><td>-12.5%</td></tr><tr><td>Vendi Score</td><td>1.25047</td><td>1.18768</td><td>-5.02%</td></tr><tr><td>DPP</td><td>0</td><td>0</td><td>0%</td></tr><tr><td>RKE</td><td>0.11721</td><td>0.08123</td><td>-30.7%</td></tr></table>

Distance monotonicity. We use the three-point planar family

$$
x _ { 1 } = ( 0 , 0 ) , \quad x _ { 2 } = ( 1 , 0 ) ,
$$

and

$$
x _ { 3 } ( c ) = \left( 1 - { \frac { c ^ { 2 } } { 2 } } , { \sqrt { 1 - \left( 1 - { \frac { c ^ { 2 } } { 2 } } \right) ^ { 2 } } } \right) , \quad c \in [ 1 . 7 6 , 1 . 9 9 ] .
$$

A direct calculation gives

$$
\left\| x _ { 1 } - x _ { 2 } \right\| = \left\| x _ { 1 } - x _ { 3 } ( c ) \right\| = 1 , \quad \left\| x _ { 2 } - x _ { 3 } ( c ) \right\| = c .
$$

Thus, increasing c changes exactly one pairwise distance while leaving the other two unchanged. Any diversity measure satisfying strict distance monotonicity must therefore increase along this path.

At the endpoints $c = 1 . 7 6$ and c = 1.99, exact GKD increases from 1.88847 to 1.89442. In contrast, the Vendi Score decreases from 2.29378 to 2.28758, and the determinant DPP score decreases from 0.37543 to 0.34676. Average pairwise distance and RKE increase along this particular path.

Uniqueness. For the uniqueness example, we use the one-dimensional family

$$
X ( t ) = \left( 0 , 0 , \frac { 1 } { 2 } , \frac { 1 } { 2 } - \frac { t } { 4 } \right) , \quad t \in [ 0 , 1 ] .
$$

At $t = 0 ,$ , the third and fourth points coincide. For every $t > 0 ,$ , the fourth point is distinct, while the duplicate pair at the origin is unchanged. Hence the uniqueness axiom requires

$$
D ( X ( t ) ) > D ( X ( 0 ) ) \quad { \mathrm { f o r ~ e v e r y ~ } } t > 0 .
$$

At the endpoints t = 0 and $t = 1$ , exact GKD increases from 1.24254 to 1.24807. Average pairwise distance decreases from 0.33333 to 0.29167, the Vendi Score decreases from 1.25047 to 1.18768, and RKE decreases from 0.11721 to 0.08123. The determinant DPP score is zero throughout because the persistent duplicate pair makes the RBF similarity matrix singular.

![](images/7500edaf0e811afe384ffc21d7486fc59b1e7a63dc6870554a4e1711ade05cf7.jpg)

![](images/487eb384ee1ce2762c62e96e781cd8c274e64e5a01d8e584718f05ba698d8e88.jpg)

![](images/31db12986678914640d1734b4a809faa02f78713c1b59501a70fdb0aba518ad4.jpg)  
Figure 2: Controlled axiom stress tests. (a) Three-point monotonicity construction with $d _ { 1 2 } = d _ { 1 3 } = 1$ and $d _ { 2 3 } = c ,$ where c increases from 1.76 to 1.99. (b) GKD increases along this expansion, while the Vendi Score decreases. (c) In the uniqueness construction, separating one member of a duplicate pair increases GKD while the Vendi Score decreases. The behavior of all comparison measures is reported in Table 1.

## D.3 ImageNet/DINOv2 Stress Tests

ImageNet representations and batching. We use the 50,000 images in the ImageNet-1k validation set (Deng et al., 2009; Russakovsky et al., 2015) and embed each image using DINOv2 ViT-L/14 (Oquab et al., 2024). We convert the resulting embeddings to double precision and $\ell _ { 2 ^ { - } }$ -normalize each vector. Each of the 1,000 ImageNet classes contains 50 validation images. Within each class, we deterministically partition the images into five disjoint batches of size 10. Thus, every image appears in exactly one base configuration, giving 5,000 base configurations in total.

Bandwidth selection. For each image $x _ { i } ,$ let

$$
r _ { i } : = \operatorname* { m i n } _ { j \neq i \atop j { \mathrm { ~ i n ~ t h e ~ s a m e ~ b a t c h } } } \| x _ { i } - x _ { j } \| ,
$$

and define

$$
r _ { 0 } : = { \mathrm { m e d i a n } } _ { i } r _ { i } .
$$

For the fixed batching used in our experiments,

$$
r _ { 0 } \approx 0 . 7 4 2 5 4 3 9 4 0 1 .
$$

We choose the primary bandwidth $h _ { 0 }$ so that two points separated by $r _ { 0 }$ have RBF similarity $1 / 2 .$ Equivalently,

$$
h _ { 0 } : = \frac { r _ { 0 } } { \sqrt { 2 \log 2 } } \approx 0 . 6 3 0 6 5 8 7 5 6 0 .
$$

We use $h _ { 0 }$ for the primary analysis and $0 . 5 h _ { 0 }$ and $2 h _ { 0 }$ for robustness checks.

We exhaustively apply Euclidean-realizable perturbations to these 5,000 base configurations, with each perturbation designed to isolate one axiom. The base configurations are observed image embeddings, while the perturbed configurations are valid Euclidean configurations that need not themselves correspond to embeddings of actual image batches. We use these controlled perturbations because the axioms concern specific changes to a configuration: monotonicity compares configurations in which pairwise distances only increase, while uniqueness compares a configuration with one obtained by introducing an exact duplicate. Constructing such perturbations lets us isolate the relevant axiom while retaining the geometry of real learned representations.

Single-edge Euclidean expansion. Fix a batch

$$
X = ( x _ { 1 } , \dots , x _ { 1 0 } )
$$

and an unordered pair $\{ i , j \}$ . Let H be the smallest afine subspace containing the other eight points. Let $p _ { i }$ and $p _ { j }$ be the orthogonal projections of $x _ { i }$ and $x _ { j }$ onto $H$ , and write

$$
x _ { i } = p _ { i } + u _ { i } , \quad x _ { j } = p _ { j } + u _ { j } ,
$$

where $u _ { i }$ and $u _ { j }$ are perpendicular to the direction space of $H$ . For every anchor point $x _ { k } \in H$

$$
\| x _ { i } - x _ { k } \| ^ { 2 } = \| p _ { i } - x _ { k } \| ^ { 2 } + \| u _ { i } \| ^ { 2 } ,
$$

and analogously for $x _ { j }$ . Therefore, keeping $p _ { i } , p _ { j }$ and the residual norms $\| u _ { i } \| , \| u _ { j } \|$ fixed preserves every distance from $x _ { i }$ and $x _ { j }$ to the other eight points.

The selected distance satisfies

$$
\| x _ { i } - x _ { j } \| ^ { 2 } = \| p _ { i } - p _ { j } \| ^ { 2 } + \| u _ { i } \| ^ { 2 } + \| u _ { j } \| ^ { 2 } - 2 \langle u _ { i } , u _ { j } \rangle .
$$

We can therefore increase only this distance by decreasing the inner product between $u _ { i }$ and $u _ { j }$ while preserving their norms. Its largest attainable value occurs when the residual vectors are antiparallel:

$$
d _ { i j , \operatorname* { m a x } } ^ { 2 } = \| p _ { i } - p _ { j } \| ^ { 2 } + { \bigl ( } \| u _ { i } \| + \| u _ { j } \| { \bigr ) } ^ { 2 } .
$$

In every batch, the eight anchor points have afine rank 7. Since the embeddings lie in $\mathbb { R } ^ { 1 0 2 4 }$ , the space perpendicular to the direction space of H therefore has dimension

$$
1 0 2 4 - 7 = 1 0 1 7 .
$$

In particular, it has dimension at least two, so whenever the selected pair has positive expansion capacity, the angle between $u _ { i }$ and $u _ { j }$ can be varied continuously while preserving both residual norms. Consequently, every squared distance between $d _ { i j } ^ { 2 }$ and $d _ { i j , \operatorname* { m a x } } ^ { 2 }$ is realizable while all other pairwise distances remain unchanged.

For $\lambda \in [ 0 , 1 ]$ , define

$$
d _ { i j } ( \lambda ) ^ { 2 } = d _ { i j } ^ { 2 } + \lambda \bigl ( d _ { i j , \operatorname* { m a x } } ^ { 2 } - d _ { i j } ^ { 2 } \bigr ) .
$$

Let $X _ { \lambda }$ denote any Euclidean realization of the resulting distance matrix. Since all measures considered here depend only on pairwise distances, the choice of realization does not matter. Thus, $\lambda = 0$ gives the observed configuration and $\lambda = 1$ gives the largest selected-edge expansion attainable while preserving all other pairwise distances; intermediate values interpolate in squared distance.

Every one of the

$$
5 , 0 0 0 { \binom { 1 0 } { 2 } } = 2 2 5 , 0 0 0
$$

candidate pairs has positive expansion capacity. Hence no pair is filtered out of the experiment. As an implementation check, the modified endpoint distance matrices remain Euclidean to floating-point precision: across all reconstructed endpoint matrices, the smallest eigenvalue of the corresponding Gram matrix is approximately $- 8 . 6 \times 1 0 ^ { - 1 4 }$ , and the maximum error after reconstructing pairwise distances from Euclidean coordinates is below $1 . 8 \times 1 0 ^ { - 1 5 }$

The parameter λ is the fraction of the available squared-distance expansion, not the relative increase in the distance itself. Define

$$
c _ { i j } = \frac { d _ { i j , \operatorname* { m a x } } ^ { 2 } - d _ { i j } ^ { 2 } } { d _ { i j } ^ { 2 } } .
$$

Then

$$
\frac { d _ { i j } ( \lambda ) } { d _ { i j } } = \sqrt { 1 + \lambda c _ { i j } } ,
$$

so the actual relative increase in the selected distance is

$$
\frac { d _ { i j } ( \lambda ) - d _ { i j } } { d _ { i j } } = \sqrt { 1 + \lambda c _ { i j } } - 1 .
$$

Across all 225,000 edges, the median distance increases for

$$
\lambda = 0 . 1 , 0 . 2 5 , 0 . 5 , 1
$$

are respectively

$$
4 . 6 \% , \quad 1 1 . 1 \% , \quad 2 1 . 1 \% , \quad 3 9 . 1 \% .
$$

Table 2: Observed monotonicity reversal rates at the primary bandwidth $h _ { 0 }$ . Each column contains all 225,000 single-edge expansions at the indicated value of λ.
<table><tr><td>Measure</td><td> $\lambda = 0 . 1$ </td><td> $\overline { { \lambda = 0 . 2 5 } }$ </td><td> $\overline { { \lambda = 0 . 5 } }$ </td><td> $\overline { { \lambda = 1 } }$ </td></tr><tr><td>Average pairwise distance</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Vendi Score</td><td>0.089%</td><td>0.438%</td><td>4.40%</td><td>33.61%</td></tr><tr><td>DPP</td><td>20.53%</td><td>39.46%</td><td>67.57%</td><td>88.21%</td></tr><tr><td>RKE</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr></table>

Monotonicity reversal rates. For each λ, we compare the diversity of the observed configuration with that of its corresponding single-edge expansion. A reversal occurs when the diversity score decreases even though one pairwise distance has increased and every other pairwise distance is unchanged.

Table 2 and Figure $\mathrm { 3 ( a ) }$ summarize the observed reversal rates at the primary bandwidth $h _ { 0 } .$ . Average pairwise distance and RKE exhibit no reversals. DPP reversals are already common at the smallest perturbation and become increasingly frequent as the expansion grows. Vendi reversals are rare for small expansions but become frequent at the largest expansion.

The DPP reversals also become more severe as the perturbation grows. For each reversed comparison, we measure the relative decrease

$$
{ \frac { D _ { \mathrm { D P P } } ( X ) - D _ { \mathrm { D P P } } ( X _ { \lambda } ) } { D _ { \mathrm { D P P } } ( X ) } } .
$$

Conditional on a reversal, the median relative decreases at $\lambda = 0 . 1 , 0 . 2 5 , 0 . 5 , 1$ are respectively

$$
0 . 1 0 \% , \quad 0 . 3 4 \% , \quad 1 . 1 2 \% , \quad 3 . 7 1 \% .
$$

At λ = 1, 65.7% of all tested expansions decrease DPP by at least 1%, and 39.7% decrease it by at least 5%. Vendi’s large-expansion reversals are substantially milder: at $\lambda = 1$ , the median decrease among reversed comparisons is 0.045%, and no tested comparison decreases Vendi by as much as 1%.

Uniqueness perturbation. For every ordered pair $i \neq j$ in each batch, we replace $x _ { i }$ by an exact copy of $x _ { j }$ . The observed configuration contains distinct embeddings, while the modified configuration contains one additional duplicate. The uniqueness axiom therefore requires the observed configuration to have strictly larger diversity.

There are

$$
5 , 0 0 0 \cdot 1 0 \cdot 9 = 4 5 0 , 0 0 0
$$

ordered replacements. Figure 3(b) summarizes the corresponding uniqueness reversal rates. At the primary bandwidth $h _ { 0 } ,$ the observed reversal rates are 18.90% for average pairwise distance, 1.83% for Vendi, 0% for DPP, and 12.70% for RKE. There are no numerically unresolved ties at this bandwidth, so the reversal rates coincide with the observed strict-axiom failure rates.

The reversals can also be substantial. Among reversed comparisons, the median wrong-way changes are 1.52% for average pairwise distance, 1.62% for Vendi, and 2.70% for RKE. Across all 450,000 ordered replacements, the fractions producing a reversal of at least 1% are respectively

$$
1 1 . 8 3 \% , \quad 1 . 2 1 \% , \quad 9 . 7 0 \% .
$$

Bandwidth robustness. For the kernel-based measures, we repeat the experiment with the pre-specified bandwidths $0 . 5 h _ { 0 }$ and $2 h _ { 0 }$ . Tables 3 and 4 give the complete reversal rates. Although the precise frequencies vary with bandwidth, the qualitative failure modes persist.

Class-level bootstrap summaries. The reversal rates above are exact descriptive proportions for the fixed ImageNet/DINOv2 configurations and perturbations used in the experiment. To summarize how these rates vary with the composition of ImageNet classes, we additionally compute class-level percentile bootstrap intervals. We use the ImageNet class as the resampling unit: all batches and perturbations belonging to a class are kept together whenever that class is selected.

![](images/162f1f02f454a052e27a07940a3a9a2302cbbba418a4616c66e456bf0cc27713.jpg)

![](images/5e40c3a2c8c7cd2e83d35b00d7277be361c605ebfb67b432e73b07f241946c1e.jpg)  
Figure 3: Axiom stress tests on ImageNet/DINOv2 geometry. (a) Reversal rates under exhaustive single-edge Euclidean expansions. (b) Reversal rates under all 450,000 ordered duplicate replacements. Error bars are 95% percentile bootstrap intervals from resampling ImageNet classes.

Table 3: Observed monotonicity reversal rates across RBF bandwidths. Each entry is the percentage of 225,000 single-edge expansions producing a wrong-direction change. Average pairwise distance and RKE have zero reversals throughout.
<table><tr><td>Measure</td><td>λ</td><td> $\overline { { 0 . 5 h _ { 0 } } }$ </td><td> $\overline { { h _ { 0 } } }$ </td><td> $\overline { { 2 h _ { 0 } } }$ </td></tr><tr><td rowspan="3">Vendi</td><td>0.1</td><td>0.334%</td><td>0.089%</td><td>0.004%</td></tr><tr><td>0.25</td><td>0.877%</td><td>0.438%</td><td>0.029%</td></tr><tr><td>0.5</td><td>2.58%</td><td>4.40%</td><td>1.09%</td></tr><tr><td rowspan="4">DPP</td><td>1</td><td>6.94%</td><td>33.61%</td><td>62.24%</td></tr><tr><td>0.1</td><td>6.02%</td><td>20.53%</td><td>33.96%</td></tr><tr><td>0.25</td><td>11.54%</td><td>39.46%</td><td>61.46%</td></tr><tr><td>0.5 1</td><td>20.24% 28.77%</td><td>67.57% 88.21%</td><td>89.37% 99.50%</td></tr></table>

Table 4: Observed uniqueness reversal rates across RBF bandwidths. Average pairwise distance is bandwidthindependent and reverses 18.90% of the ordered replacements.
<table><tr><td>Measure</td><td>0.5h0</td><td> $h _ { 0 }$ </td><td> $\overline { { 2 h _ { 0 } } }$ </td></tr><tr><td>Vendi Score</td><td>0.008%</td><td>1.83%</td><td>8.96%</td></tr><tr><td>DPP</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>RKE</td><td>0.74%</td><td>12.70%</td><td>26.18%</td></tr></table>

For each of 2,000 bootstrap replicates, we sample the 1,000 ImageNet classes with replacement and recompute the mean class-level reversal rate. We report the 2.5th and 97.5th percentiles of the resulting bootstrap distribution.

At λ = 1, the resulting 95% percentile bootstrap intervals are [87.50%, 88.86%] for DPP and [32.50%, 34.71%] for Vendi. For the uniqueness perturbation, the corresponding intervals are [18.40%, 19.37%] for average pairwise distance, [1.61%, 2.07%] for Vendi, and [12.02%, 13.39%] for RKE.

## E HARDNESS OF EXISTING DIVERSITY MEASURES ON EUCLIDEAN INPUTS

Mironov and Prokhorenkova (2025) construct two diversity functions, MultiDimVolume and IntegralMaxClique, that satisfy all three of their axioms but are NP-hard to compute on general dissimilarity matrices. In this section, we show that both measures remain NP-hard when the input is restricted to Euclidean-realizable distance matrices. Moreover, unless NP = RP, neither measure admits an FPRAS on Euclidean-realizable inputs. The key observation is that an arbitrary graph can be encoded by a suficiently small perturbation of the distances of a regular simplex.

## E.1 A Euclidean Graph Encoding

Lemma E.1 (Near-equilateral Euclidean realization). Let $G = ( [ n ] , E )$ be a simple graph with $n \geq 2 ,$ and let $0 < \epsilon \leq 1 / n$ . Define the symmetric matrix $D = \left( d _ { i j } \right)$ by $d _ { i i } = 0$ and, $f o r \ i \ne j$

$$
d _ { i j } = \left\{ \begin{array} { l l } { 2 + \epsilon , } & { \left\{ i , j \right\} \in E , } \\ { 2 , } & { \left\{ i , j \right\} \notin E . } \end{array} \right.
$$

Then D is a Euclidean distance matrix: there exist $\pmb { x } _ { 1 } , \dots , \pmb { x } _ { n } \in \mathbb { R } ^ { n - 1 }$ such that $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert = d _ { i j }$ for all $i , j$

Proof. Let A be the adjacency matrix of $G ,$ let

$$
H : = I - { \frac { 1 } { n } } \mathbf { 1 } \mathbf { 1 } ^ { T } ,
$$

and define

$$
q : = ( 2 + \epsilon ) ^ { 2 } - 4 = 4 \epsilon + \epsilon ^ { 2 } .
$$

Consider the symmetric matrix

$$
B : = 2 H - \frac { q } { 2 } H A H .
$$

We first show that $B$ is positive semidefinite. For any $\pmb { v } \in \mathbb { R } ^ { n }$ , let $\mathbf { \boldsymbol { y } } : = H \mathbf { \boldsymbol { v } }$ . Since H is the orthogonal projection onto $\mathbf { 1 } ^ { \perp }$

$$
{ \pmb v } ^ { T } B { \pmb v } = 2 \| { \pmb y } \| ^ { 2 } - \frac { q } { 2 } { \pmb y } ^ { T } A { \pmb y } .
$$

Since A is a 0-1 adjacency matrix,

$$
\| A \| _ { 1 } \leq n - 1 , \quad \| A \| _ { \infty } \leq n - 1 .
$$

Hence

$$
\| A \| _ { 2 } \leq { \sqrt { \| A \| _ { 1 } \| A \| _ { \infty } } } \leq n - 1 .
$$

Therefore,

$$
\begin{array} { r } { \pmb { y } ^ { T } \pmb { A } \pmb { y } \le \| \pmb { A } \| _ { 2 } \| \pmb { y } \| ^ { 2 } \le ( n - 1 ) \| \pmb { y } \| ^ { 2 } , } \end{array}
$$

and consequently

$$
{ \pmb v } ^ { T } { \cal B } { \pmb v } \geq \left( 2 - \frac { q ( n - 1 ) } { 2 } \right) \| { \pmb y } \| ^ { 2 } .
$$

Because $\epsilon \leq 1 / n$

$$
q ( n - 1 ) \leq \left( { \frac { 4 } { n } } + { \frac { 1 } { n ^ { 2 } } } \right) ( n - 1 ) = 4 - { \frac { 3 } { n } } - { \frac { 1 } { n ^ { 2 } } } < 4 .
$$

Thus, $B \succeq 0 .$

Since B is positive semidefinite, let $r : = \mathrm { r a n k } ( B )$ and take a Gram factorization

$$
B = X ^ { T } X , \quad X \in \mathbb { R } ^ { r \times n } .
$$

Let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ be the i-th column of $X ,$ so that

$$
B _ { i j } = \langle { \pmb x } _ { i } , { \pmb x } _ { j } \rangle .
$$

Moreover, $H \mathbf { 1 } = 0$ implies $B { \bf 1 } = 0$ , and hence

$$
\operatorname { r a n k } ( B ) \leq n - 1 .
$$

Thus, we may regard $\pmb { x } _ { 1 } , \ldots , \pmb { x } _ { n }$ as points in $\mathbb { R } ^ { n - 1 }$

It remains to verify their pairwise distances. For $i \neq j ,$ , the vector $e _ { i } - e _ { j }$ lies in $\mathbf { 1 } ^ { \perp }$ , so

$$
H ( e _ { i } - e _ { j } ) = e _ { i } - e _ { j } .
$$

Consequently,

$$
\| { \pmb x } _ { i } - { \pmb x } _ { j } \| ^ { 2 } = ( e _ { i } - e _ { j } ) ^ { T } B ( e _ { i } - e _ { j } ) = 2 \| { \pmb e } _ { i } - { \pmb e } _ { j } \| ^ { 2 } - \frac { q } { 2 } ( e _ { i } - e _ { j } ) ^ { T } A ( e _ { i } - e _ { j } ) .
$$

Since G is simple, $A _ { i i } = A _ { j j } = 0$ and $A _ { i j } = A _ { j i }$ , giving

$$
( e _ { i } - e _ { j } ) ^ { T } A ( e _ { i } - e _ { j } ) = - 2 A _ { i j } .
$$

Therefore,

$$
\| { \pmb x } _ { i } - { \pmb x } _ { j } \| ^ { 2 } = 4 + q A _ { i j } .
$$

If $\{ i , j \} \not \in E$ , this equals 4, so $\| { \pmb x } _ { i } - { \pmb x } _ { j } \| = 2 . \ \mathrm { I f } \ \{ i , j \} \in E$ , this equals $4 + q = ( 2 + \epsilon ) ^ { 2 }$ , so $\lVert \pmb { x } _ { i } - \pmb { x } _ { j } \rVert = 2 + \epsilon$ Hence the resulting Euclidean configuration realizes D. □

## E.2 Exact Hardness on Euclidean Inputs

For completeness, we recall the definition of two diversity measures of Mironov and Prokhorenkova (2025). For a distance matrix $D = \left( d _ { i j } \right)$ on n objects and $k \in \{ 2 , \ldots , n \}$ , define

$$
M _ { k } ( D ) : = \operatorname* { m a x } _ { S \subseteq [ n ] } \prod _ { i , j \in S \atop i < j } d _ { i j } .
$$

The MultiDimVolume diversity is

$$
\operatorname { M u l t i D i m V o l u m e } ( D ) : = \sum _ { k = 2 } ^ { n } M _ { k } ( D ) .
$$

For $t \geq 0 ,$ , let $G _ { t } ( D )$ be the graph on [n] containing an edge $\{ i , j \}$ whenever $d _ { i j } \geq t ,$ , with edge weight $d _ { i j }$ . Among the maximum-cardinality cliques of $G _ { t } ( D )$ , let $C _ { t }$ be one with maximum total edge weight, and write

$$
w _ { t } ( D ) : = \sum _ { \stackrel { i , j \in C _ { t } } { i < j } } d _ { i j } .
$$

The IntegralMaxClique diversity is

$$
\operatorname { I n t e g r a l M a x C l i q u e } ( D ) : = \int _ { 0 } ^ { \infty } w _ { t } ( D ) d t .
$$

We now apply Lemma E.1 to the two axiomatic diversity measures of Mironov and Prokhorenkova (2025).

Proposition E.2 (Hardness on Euclidean inputs). MultiDimVolume and IntegralMaxClique are NP-hard to compute even when restricted to Euclidean-realizable distance matrices.

Proof. We reduce from Maximum $C l i q u e ,$ the NP-hard problem of computing the maximum clique size $\omega ( G )$ of a graph G (Karp, 1972).

MultiDimVolume The NP-hardness reduction of Mironov and Prokhorenkova (2025) already uses a twodistance encoding of a graph: edges have distance $2 + \epsilon$ and nonedges have distance 2, for a suficiently small $\epsilon > 0$ . Let $D _ { G }$ denote this distance matrix. We record the quantitative details needed to ensure that the resulting instance is also Euclidean-realizable.

For $k \in \{ 2 , \ldots , n \}$ , define

$$
c _ { k } : = { \binom { k } { 2 } } , \quad r _ { k } : = \operatorname* { m a x } _ { S \subseteq [ n ] } | E ( G [ S ] ) | .
$$

For any k-vertex subset $S ,$ , its product of pairwise distances is

$$
( 2 + \epsilon ) ^ { | E ( G [ S ] ) | } 2 ^ { c _ { k } - | E ( G [ S ] ) | } .
$$

Hence the k-th term in MultiDimVolume is

$$
2 ^ { c _ { k } } \left( 1 + \frac { \epsilon } { 2 } \right) ^ { r _ { k } } .
$$

Writing $m : = { \binom { n } { 2 } }$ , choose

$$
\epsilon : = 2 ^ { - ( 2 m + \lceil \log _ { 2 } n \rceil + 3 ) } .
$$

This number has $O ( n ^ { 2 } )$ bits, thus the distance matrix $D _ { G }$ can be constructed with polynomial bit complexity.   
It also satisfies $\epsilon \leq 1 / n ,$ , so Lemma E.1 guarantees that the resulting distance matrix is Euclidean-realizable.

Expanding by the binomial theorem gives

$$
\mathrm { M u l t i D i m V o l u m e } ( D _ { G } ) = A _ { 0 } + \epsilon M + R ,
$$

where

$$
A _ { 0 } : = \sum _ { k = 2 } ^ { n } 2 ^ { c _ { k } } , \quad M : = \sum _ { k = 2 } ^ { n } r _ { k } 2 ^ { c _ { k } - 1 } ,
$$

and $R \geq 0$ contains all terms of degree at least two in ϵ. Since $r _ { k } \le c _ { k } \le m$ and $\epsilon / 2 \le 1$

$$
\begin{array} { l } { { \displaystyle R = \sum _ { k = 2 } ^ { n } 2 ^ { c _ { k } } \sum _ { j = 2 } ^ { r _ { k } } \binom { r _ { k } } { j } \left( \frac { \epsilon } { 2 } \right) ^ { j } } } \\ { { \le n 2 ^ { m } \left( \frac { \epsilon } { 2 } \right) ^ { 2 } 2 ^ { m } } } \\ { { = n \epsilon ^ { 2 } 2 ^ { 2 m - 2 } . } } \end{array}
$$

By our choice of $\epsilon ,$

$$
0 \leq \frac { R } { \epsilon } < \frac { 1 } { 2 } .
$$

Therefore the exact MultiDimVolume value determines the integer

$$
M = \left\lfloor \frac { \mathrm { M u l t i D i m V o l u m e } ( D _ { G } ) - A _ { 0 } } { \epsilon } \right\rfloor .
$$

It remains to recover the values $r _ { k }$ from M. Define

$$
w _ { k } : = 2 ^ { c _ { k } - 1 } .
$$

As observed in the original reduction of Mironov and Prokhorenkova (2025), these weights satisfy

$$
w _ { k } > \sum _ { i = 2 } ^ { k - 1 } c _ { i } w _ { i }
$$

for every $k \geq 3$ . The inequality holds for $k = 3$ , and the inductive step follows from

$$
\frac { w _ { k + 1 } } { w _ { k } } = 2 ^ { k } > \binom { k } { 2 } + 1 .
$$

Since $0 \leq r _ { k } \leq c _ { k }$ , we can therefore recover $r _ { n } , r _ { n - 1 } , \ldots , r _ { 2 }$ greedily from

$$
M = \sum _ { k = 2 } ^ { n } r _ { k } w _ { k } .
$$

Moreover, M has $O ( n ^ { 2 } )$ bits, so this greedy recovery requires only polynomial-time integer arithmetic. Finally,

$$
r _ { k } = c _ { k }
$$

if and only if G contains a clique of size k. Hence

$$
\omega ( G ) = \operatorname* { m a x } \left\{ k \biggm | r _ { k } = \binom { k } { 2 } \right\} ,
$$

so exact computation of MultiDimVolume on these Euclidean inputs is NP-hard.

IntegralMaxClique. For this reduction, set $\epsilon : = 1 / n$ and define $D _ { G } = \left( d _ { i j } \right)$ where $d _ { i i } = 0$ , and for $i \neq j$

$$
d _ { i j } = \left\{ \begin{array} { l l } { 2 + \epsilon , } & { \left\{ i , j \right\} \in E , } \\ { 2 , } & { \left\{ i , j \right\} \notin E . } \end{array} \right.
$$

By Lemma E.1, this distance matrix is Euclidean-realizable.

Let $\textstyle S : = \sum _ { i < j } d _ { i j }$ and let $s : = \omega ( G )$ . Recall that IntegralMaxClique constructs, for each threshold $t \geq 0$ , the graph containing an edge $\{ i , j \}$ whenever $d _ { i j } \geq t ,$ and lets $w _ { t } ( D _ { G } )$ be the total edge weight of a maximumcardinality clique, breaking ties by total weight.

For $0 \leq t \leq 2$ , the threshold graph is complete, so

$$
w _ { t } ( D _ { G } ) = S .
$$

For $2 < t \leq 2 + \epsilon$ , the threshold graph is exactly $G .$ Its maximum-cardinality clique has s vertices, and every surviving edge has weight $2 + \epsilon ,$ so

$$
w _ { t } ( D _ { G } ) = ( 2 + \epsilon ) \binom { s } { 2 } .
$$

For $t > 2 + \epsilon$ , the threshold graph has no edges, so

$$
w _ { t } ( D _ { G } ) = 0 .
$$

Consequently,

$$
\mathrm { I n t e g r a l M a x C l i q u e } ( D _ { G } ) = 2 S + \epsilon ( 2 + \epsilon ) \binom { s } { 2 } .
$$

Since $S$ and ϵ are known from the constructed instance, exact evaluation of IntegralMaxClique determines

$$
\binom { s } { 2 } = \frac { \mathrm { I n t e g r a l M a x C l i q u e } ( D _ { G } ) - 2 S } { \epsilon ( 2 + \epsilon ) } ,
$$

and therefore determines $s = \omega ( G )$

Thus, IntegralMaxClique is also NP-hard to compute on Euclidean-realizable inputs.

## E.3 Hardness of Randomized Approximation

We next show that allowing randomized relative approximation does not make either of the two measures eficiently computable. For a graph G, write $\omega ( G )$ for its clique number. We use the standard notion of a fully polynomial randomized approximation scheme (FPRAS) (Motwani and Raghavan, 1995): given an instance with value $F > 0$ and parameters $\eta , \delta \in ( 0 , 1 )$ , the algorithm returns an estimate $\widehat F$ satisfying

$$
\operatorname* { P r } \Big [ ( 1 - \eta ) F \le \widehat { F } \le ( 1 + \eta ) F \Big ] \ge 1 - \delta ,
$$

in time polynomial in the input size, $1 / \eta .$ , and $\log ( 1 / \delta )$

Theorem E.3 (Nonexistence of FPRAS on Euclidean inputs). Unless $\mathrm { N P = R P }$ , neither MultiDimVolume nor IntegralMaxClique admits an FPRAS, even when restricted to Euclidean-realizable distance matrices.

Proof. We first show that an FPRAS for either measure will allow us to recover the maximum clique size $\omega ( G )$ of an arbitrary graph with constant success probability.

MultiDimVolume. Let $G = ( [ n ] , E )$ with $n \geq 2 .$ , and write $\omega : = \omega ( G )$ . Set $r : = 1 0 0 n$ . Construct a graph H by replacing each vertex $i \in [ n ]$ with a clique $V _ { i }$ of $r$ vertices. For distinct $i , j$ , connect every vertex of $V _ { i }$ to every vertex of $V _ { j } { \mathrm { ~ i f ~ } } \{ i , j \} \in E$ , and place no edges between the two clusters otherwise. Then

$$
N : = | V ( H ) | = r n , \quad \omega ( H ) = r \omega .
$$

Expanding the maximum clique of G gives a clique of size rω in $H .$ . Conversely, the clusters intersected by any clique of H correspond to a clique of G, so any clique of H contains vertices from at most ω clusters and therefore has size at most rω.

Define

$$
\rho : = 1 - \frac { 1 } { 2 N + 1 } ,
$$

and construct a distance matrix $D _ { H }$ by defining $d _ { u u } = 0$ , and for distinct $u , v ,$

$$
d _ { u v } = \left\{ \begin{array} { l l } { 1 , } & { \{ u , v \} \in E ( H ) , } \\ { \rho , } & { \{ u , v \} \notin E ( H ) . } \end{array} \right.
$$

This matrix is Euclidean-realizable, since multiplying every distance by $2 / \rho$ gives distance 2 on nonedges and $\begin{array} { r } { \frac { 2 } { \rho } = 2 + \frac { 1 } { N } } \end{array}$ on edges. Lemma E.1 therefore applies to the scaled matrix, and scaling the realizing configuration b $\rho / 2$ yields a realization of $D _ { H }$

Let $M _ { k } ( D _ { H } )$ be the k-th term of MultiDimVolume. Since every distance is at most 1,

$$
M _ { k } ( D _ { H } ) \leq 1 .
$$

Since H contains a clique of size $r \omega , M _ { k } ( D _ { H } ) = 1$ for $2 \leq k \leq r \omega$

We next bound the remaining terms. Fix $s \geq 1$ such that $r \omega + s \leq N$ , and consider any set $Q \subseteq V ( H )$ with

$$
| Q | = r \omega + s .
$$

For each $i \in [ n ]$ , let

$$
a _ { i } : = | Q \cap V _ { i } | , \quad x _ { i } : = { \frac { a _ { i } } { r } } .
$$

Thus, $x _ { i }$ is the fraction of vertices of $V _ { i }$ contained in $Q ,$ and

$$
\sum _ { i = 1 } ^ { n } x _ { i } = \omega + { \frac { s } { r } } .
$$

Let $q ( Q )$ denote the number of unordered pairs in $Q$ that are nonadjacent in $H ,$ equivalently the number of nonedges in the induced subgraph $H [ Q ]$ . Two vertices from distinct clusters $V _ { i }$ and $V _ { j }$ are nonadjacent in H exactly when $\{ i , j \} \not \in E ( G )$ . Therefore,

$$
q ( Q ) = \sum _ { \stackrel { 1 \leq i < j \leq n } { \{ i , j \} \notin E ( G ) } } a _ { i } a _ { j } = r ^ { 2 } \sum _ { \stackrel { 1 \leq i < j \leq n } { \{ i , j \} \notin E ( G ) } } x _ { i } x _ { j } .
$$

For any subset $R \subseteq [ n ]$ , let $\overline { { e } } _ { G } ( R )$ denote the number of unordered pairs $\{ i , j \} \subseteq R$ such that $\{ i , j \} \not \in E ( G )$ Then

$$
| R | - { \overline { { e } } } _ { G } ( R ) \leq \omega .
$$

Indeed, choose one endpoint from each missing edge inside $R ,$ and let C be the set of chosen vertices. Then $| C | \leq { \overline { { e } } } _ { G } ( R )$ , and every pair of vertices in $R \backslash C$ is adjacent in G. Hence $R \backslash C$ is a clique, so

$$
\omega \geq | R \rangle C | \geq | R | - \overline { { e } } _ { G } ( R ) .
$$

Now form a random subset $R \subseteq [ n ]$ by independently including each vertex i with probability $x _ { i }$ . Then

$$
\mathbb { E } [ | R | ] = \sum _ { i = 1 } ^ { n } x _ { i } = \omega + \frac { s } { r } ,
$$

while

$$
\mathbb { E } [ \overline { { e } } _ { G } ( R ) ] = \sum _ { \begin{array} { l } { 1 \leq i < j \leq n } \\ { \{ i , j \} \notin E ( G ) } \end{array} } x _ { i } x _ { j } .
$$

Taking expectations in the inequality above gives

$$
\omega + \frac { s } { r } - \sum _ { \stackrel { 1 \leq i < j \leq n } { \{ i , j \} \notin E \left( G \right) } } x _ { i } x _ { j } \leq \omega .
$$

Therefore,

$$
\sum _ { \stackrel { 1 \leq i < j \leq n } { \{ i , j \} \notin E ( G ) } } x _ { i } x _ { j } \geq \frac { s } { r } .
$$

Recalling the expression for $q ( Q )$ , we conclude that $q ( Q ) \geq r s$

Every nonedge of $H$ contributes a factor $\rho$ to the pairwise distance product, whereas every edge contributes a factor 1. Therefore,

$$
\begin{array} { r } { M _ { r \omega + s } ( D _ { H } ) \le \rho ^ { r s } . } \end{array}
$$

Writing

$$
V _ { H } : = \mathrm { M u l t i D i m V o l u m e } ( D _ { H } ) ,
$$

we obtain

$$
V _ { H } = r \omega - 1 + T ,
$$

where

$$
0 \leq T \leq \sum _ { s = 1 } ^ { \infty } \rho ^ { r s } = { \frac { \rho ^ { r } } { 1 - \rho ^ { r } } } .
$$

Since

$$
\rho = 1 - \frac { 1 } { 2 N + 1 } \leq \exp \left( - \frac { 1 } { 2 N + 1 } \right) ,
$$

we have

$$
T \leq { \frac { 1 } { \exp ( r / ( 2 N + 1 ) ) - 1 } } \leq { \frac { 2 N + 1 } { r } } = 2 n + { \frac { 1 } { r } } .
$$

Suppose now that MultiDimVolume admits an FPRAS. Run it on $D _ { H }$ with

$$
\eta : = \frac 1 { 1 0 n } , \quad \delta : = \frac 1 4 .
$$

Since $M _ { k } ( D _ { H } ) \le 1$ for every $k ,$

$$
V _ { H } \le N - 1 < N = r n .
$$

Thus, on the success event,

$$
| \widehat { V } _ { H } - V _ { H } | \leq \eta V _ { H } < \frac { r } { 1 0 } .
$$

Consequently,

$$
\begin{array} { c } { \displaystyle { \left| \frac { \widehat { V } _ { H } + 1 } { r } - \omega \right| \leq \frac { | \widehat { V } _ { H } - V _ { H } | } { r } + \frac { T } { r } } } \\ { \displaystyle { < \frac { 1 } { 1 0 } + \frac { 2 n + 1 / r } { 1 0 0 n } < \frac { 1 } { 2 } . } } \end{array}
$$

Therefore, rounding $( \widehat { V } _ { H } + 1 ) / r$ to the nearest integer recovers $\omega ( G )$ exactly whenever the FPRAS succeeds.

Finally, $N = 1 0 0 n ^ { 2 }$ , the rational distance $\rho = 1 - 1 / ( 2 N + 1 )$ has polynomial bit complexity, and $1 / \eta = 1 0 n .$ Hence the construction and the FPRAS call both have polynomial complexity in the size of G.

IntegralMaxClique. Let $G = ( [ n ] , E )$ with $n \geq 2$ , and construct the Euclidean-realizable distance matrix $D _ { G }$ from Proposition E.2 using $\epsilon = 1 / n$ . Write

$$
s : = \omega ( G ) , \quad S : = \sum _ { i < j } d _ { i j } ,
$$

and write

$$
I _ { G } : = \mathrm { I n t e g r a l M a x C l i q u e } ( D _ { G } ) .
$$

The proof of Proposition E.2 gives

$$
I _ { G } = 2 S + a { \binom { s } { 2 } } , \quad a : = \epsilon ( 2 + \epsilon ) .
$$

Since every distance is at most $2 + 1 / n \leq 5 / 2 .$

$$
S \leq { \frac { 5 } { 2 } } { \binom { n } { 2 } } < { \frac { 5 } { 4 } } n ^ { 2 } .
$$

Also,

$$
a = { \frac { 1 } { n } } \left( 2 + { \frac { 1 } { n } } \right) \leq { \frac { 5 } { 2 n } } , \quad { \binom { s } { 2 } } \leq { \binom { n } { 2 } } ,
$$

so

$$
I _ { G } < \frac { 5 } { 2 } n ^ { 2 } + \frac { 5 } { 2 n } \frac { n ^ { 2 } } { 2 } < 4 n ^ { 2 } .
$$

On the other hand,

$$
a \geq 2 \epsilon = { \frac { 2 } { n } } .
$$

Suppose IntegralMaxClique has an FPRAS, and run it with

$$
\eta : = \frac { 1 } { 8 n ^ { 3 } } , \quad \delta : = \frac { 1 } { 4 } .
$$

On the success event, define

$$
\widehat { T } : = \frac { \widehat { I } _ { G } - 2 S } { a } .
$$

Since the true value is $T : = { \binom { s } { 2 } }$ , we have

$$
| \widehat { T } - T | = \frac { | \widehat { I } _ { G } - I _ { G } | } { a } \leq \eta \frac { I _ { G } } { a } < \eta \cdot 2 n ^ { 3 } = \frac { 1 } { 4 } .
$$

Rounding $\widehat { T }$ to the nearest integer therefore recovers $\begin{array} { r } { T = \binom { s } { 2 } } \end{array}$ exactly, and hence recovers $s = \omega ( G )$ . Since $1 / \eta = 8 n ^ { 3 }$ , the entire procedure is a randomized polynomial-time procedure.

From clique-number recovery to one-sided error. It remains to convert either randomized clique-number procedure above into a one-sided-error algorithm for Clique. Given a graph $G$ and an integer $k ,$ the decision problem Clique asks whether $G$ contains a clique of size at least k. The problem is NP-complete (Karp, 1972). The cases $k \leq 1$ are trivial.

The two recovery procedures above remain valid with any requested FPRAS failure probability $\delta ^ { \prime } { } .$ , since their recovery guarantees hold whenever the corresponding FPRAS call succeeds. For each clique-number query below, set $\delta ^ { \prime } = 1 / ( 4 n )$ . By the FPRAS running-time guarantee, this incurs only a polynomial dependence on $\log ( 4 n ) = O ( \log n )$

Starting with $J : = G$ , process its current vertices one by one. For a current vertex $v ,$ compute $\omega ( J - v )$ using the corresponding clique-number recovery procedure. If the result is at least $k ,$ delete $v ;$ otherwise, keep it.

Condition on the event that every clique-number query returns the true clique number. If G is a YES-instance, then $J$ continues to contain a k-clique throughout the process. Moreover, once a vertex v is kept because $\omega ( J - v ) \ < \ k _ { \cdot }$ , subsequent vertex deletions cannot make $\omega ( J - v )$ increase to $k .$ Thus, at termination, no remaining vertex can be deleted while preserving a k-clique.

Thus, the final graph contains a k-clique. If it contained more than k vertices, choose any k-clique C in the remaining graph. Any vertex outside C could be deleted while leaving C intact, contradicting the termination condition. Therefore, the final graph has at most k vertices. Since it contains a k-clique, it has exactly k vertices and is complete.

Finally, accept only if the remaining graph has exactly k vertices and is complete. Hence a NO-instance is never accepted. For a YES-instance, the algorithm accepts whenever every clique-number query is correct. At most n queries are made, and each has failure probability at most $1 / ( 4 n )$ ; by the union bound, the probability that any query fails is at most $1 / 4$ . Thus, a YES-instance is accepted with probability at least $3 / 4$

Hence $\mathrm { C _ { L I Q U E } } \in \mathrm { R P }$ . Since Clique is NP-complete and $\mathrm { R P \subseteq N P } .$ , this implies $\mathrm { N P } = \mathrm { R P } .$