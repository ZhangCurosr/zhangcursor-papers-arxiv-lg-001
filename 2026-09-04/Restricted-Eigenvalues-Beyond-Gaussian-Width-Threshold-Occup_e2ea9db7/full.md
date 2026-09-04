# Restricted Eigenvalues Beyond Gaussian Width: Threshold Occupancy under Heavy Tails

Shi Fu and Huibo Xu and Qixin Zhang and Dacheng Tao

Generative AI Lab, College of Computing and Data Science,

Nanyang Technological University, Singapore

## Abstract

Restricted eigenvalue (RE) bounds govern stable recovery by norm-regularized estimators. For isotropic sub-Gaussian measurements, the benchmark sample size is $1 + w ( A ) ^ { 2 }$ , where w(A) is the Gaussian width of the normalized descent cone. The COLT 2015 open-problem note (Banerjee et al., 2015) asked whether the same law follows for heavy-tailed designs from a uniform small-ball condition alone. We give an explicit and systematic negative answer to the general question as formulated there: the proposed law fails in its full dimension-free, arbitrary-set form, and the missing obstruction is simultaneous threshold occupancy. A constant-width polyhedral descent cone with fixed small-ball constants has zero empirical RE on every sample path up to half the ambient dimension. More generally, every finite range space admits exact threshold encoding in an arbitrarily narrow spherical cap and a lift to a full polyhedral descent-cone section. For every fixed threshold VC dimension d, as $\beta \downarrow 0 .$ , the sharp worst-case sample complexity is $\Theta ( \beta ^ { - 1 } [ d \log ( 1 / \beta ) + $ log(1/δ)]). The separation persists under exact isotropy and all finite moments: on the same constant-width cone, Gaussian measurements succeed with $O ( 1 + \log ( 1 / \delta ) )$ samples, whereas an isotropic heavy-tailed design fails pathwise for $n \lesssim { \sqrt { p / \log p } } .$ Gaussian smoothing yields an everywhere-positive $C ^ { \infty }$ density while retaining arbitrarily poor RE. Under isotropy, a distribution-free fallback governed by affine dimension times squared enclosing radius is sharp on this family.

Keywords: restricted eigenvalues; heavy-tailed designs; small-ball method; Gaussian width; VC dimension

## 1. Introduction

Many high-dimensional estimators recover a structured parameter by minimizing a norm subject to fitting the data. The Lasso is the canonical example: the $\ell _ { 1 }$ norm favors sparse vectors, but it cannot distinguish the target $\theta ^ { \star }$ from a perturbation that both preserves the measurements and does not increase the norm. More generally, the perturbations that a regularizer cannot exclude form its descent cone. Recovery is possible only when the measurement matrix is well conditioned on this cone.

The restricted eigenvalue (RE) quantifies this condition. Let $\ b { X } \in \mathbb { R } ^ { n \times p }$ have i.i.d. rows $Z _ { 1 } , \ldots , Z _ { n }$ with common law $P _ { \mathbf { \varepsilon } }$ , and let $A \subseteq \mathbb { S } ^ { p - 1 }$ be the normalized set of relevant directions, typically the spherical section of a descent cone. We write

$$
\Lambda _ { n } ( X ; A ) = \operatorname* { i n f } _ { u \in A } { \frac { \| X u \| _ { 2 } ^ { 2 } } { n } } = \operatorname* { i n f } _ { u \in A } { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \langle Z _ { i } , u \rangle ^ { 2 } .\tag{1}
$$

If $\Lambda _ { n } ( X ; A ) ~ > ~ 0$ , no relevant direction lies in the kernel. A quantitative lower bound controls the sensitivity of recovery to noise and, in statistical models, the estimation error (Bickel et al., 2009; Chandrasekaran et al., 2012; Negahban et al., 2012).

For centered isotropic sub-Gaussian rows, a clean geometric law holds up to distributiondependent constants: the benchmark sample size is controlled by $1 + w ( A ) ^ { 2 }$ , where $w ( A ) =$ $\mathbb { E } \operatorname* { s u p } _ { u \in A } \langle g , u \rangle$ is the Gaussian width. The $w ( A ) ^ { 2 }$ term acts as an effective dimension of the cone, while the additive constant covers even zero-width sets. For Gaussian measurements, statistical-dimension theory sharpens the picture to precise phase transitions for sparse recovery, low-rank recovery, and many other convex inverse problems (Gordon, 1988; Chandrasekaran et al., 2012; Amelunxen et al., 2014; Tropp, 2015a).

Heavy-tailed measurements motivate a different route. Upper-tail concentration may be unavailable, but each fixed direction may still be visible with nontrivial probability. The small-ball condition asks that, for some $\alpha , \beta > 0$

$$
\mathsf { Q } _ { \alpha } ( P , A ) : = \operatorname* { i n f } _ { u \in A } P \{ | \langle Z , u \rangle | \ge \alpha \} \ge \beta .\tag{2}
$$

The apparent analogy hides a change in quantifiers. The small-ball condition says that, for every fixed direction u, a fresh population draw detects u with probability at least $\beta .$ An RE bound asks for more: with high probability, one common sample must detect every direction in A simultaneously. Under sub-Gaussian assumptions, increment control couples nearby directions and Gaussian width measures the remaining uniformity cost. A bare small-ball condition provides no such coupling.

The COLT 2015 open problem of Banerjee et al. (2015) asked whether, for arbitrary spherical A, one can nevertheless prove

$$
\operatorname* { i n f } _ { u \in A } \| X u \| _ { 2 } ^ { 2 } \geq c _ { 1 } ( \alpha , \beta ) n - c _ { 2 } ( \alpha , \beta ) w ( A ) ^ { 2 }\tag{3}
$$

with high probability. We make explicit the sample-complexity interpretation of the “suitable constants” in that question: $c _ { 1 } > 0$ and $c _ { 2 } < \infty$ may depend on the small-ball parameters but not on the ambient dimension or on A. Without this dimension-free uniformity, the proposed Gaussian-width law would not express the same-order sample-complexity principle posed in the original note. A positive answer would have made the Gaussian-width phase diagram universal far beyond Gaussian designs, without upper-tail concentration or high-order moment assumptions.

Our answer. We show that the dimension-free Gaussian-width law posed in the COLT 2015 note does not follow from a bare small-ball condition, even when the design is exactly isotropic. Our results identify simultaneous threshold occupancy as the general obstruction, determine its sharp worst-case complexity, and delineate what isotropy does and does not recover.

The missing information is simultaneous coverage. Gaussian width measures Euclidean proximity between directions, but a small-ball hypothesis need not couple their visibility events. Imagine many latent row types and one direction for each subset of these types. Direction u is visible precisely when the sampled row type belongs to its associated subset. All directions can be placed in an arbitrarily narrow spherical cap, so their Gaussian width is tiny, while a finite sample must still hit every subset in a complicated range system. Missing one range leaves a direction with zero empirical energy. Thus the problem is not Gaussian-process geometry but threshold coverage.

Main contributions. We establish four complementary results that expose the obstruction, quantify its complexity, and identify a positive boundary.

• A pathwise counterexample at constant width. For every even m, we give a polyhedral norm and a centered, directionally heavy-tailed design with $w ( A ) = O ( 1 )$ and fixed small-ball constants, yet $\Lambda _ { n } ( X ; A ) = 0$ on every sample path whenever $n \leq m / 2$ (Theorem 1). This is a deterministic finite-sample obstruction, not a failure event hidden in a tail bound.

• Range-space universality and the sharp low-mass law. We represent every finite probability range space exactly by threshold events at arbitrarily small Gaussian width (Theorem 3), then lift missed-range witnesses to a full polyhedral descent-cone section while preserving uniform small-ball mass over the cone (Theorem 4). Thus occupancy is intrinsic. At threshold VC dimension $d ,$ uniform RE holds at $n \gtrsim \beta ^ { - 1 } \{ d \log ( 2 / \beta ) + \log ( 1 / \delta ) \}$ and this order is necessary for every fixed d as $\beta \downarrow 0$ (Theorem 5). This sharp law concerns general spherical sets; the cone lift does not claim to preserve the generating range space’s VC dimension.

• An isotropic and smooth separation on the same geometry. On one common descent cone, Gaussian measurements succeed after ${ \cal O } ( 1 + \log ( 1 / \delta ) )$ ) samples, whereas a centered isotropic heavy-tailed design with all moments has zero RE pathwise for $n \lesssim \sqrt { p / \log p }$ (Theorem 7 and Corollary 8). Gaussian smoothing gives a positive $C ^ { \infty }$ density while leaving the restricted singular value arbitrarily small (Corollary 9). Thus neither covariance normalization nor removing atoms restores Gaussian-width control.

• A positive boundary under isotropy. Isotropy yields a coarser distribution-free guarantee: if $q ( A )$ is the affine dimension and $r ( A )$ the minimum enclosing radius, then $q ( A ) r ( A ) ^ { 2 }$ replaces $w ( A ) ^ { 2 }$ in a small-ball RE bound (Theorem 10 and Corollary 11). The anchoredframe family matches this scale. This is a genuine positive consequence of isotropy, without asserting a joint minimax formula for all complexity pairs.

Relation to earlier sparse-recovery constructions. The spiky construction in the 2014 arXiv version of Lecue and Mendelson´ (2017b) predates the COLT note and can be retrospectively interpreted as a negative instance of the fully uniform Gaussian-width implication for a particular $\ell _ { 1 } { \mathrm { - r e l a t e d } }$ sparse-recovery geometry. The COLT note cited that work in its discussion of unit s-sparse directions and, after reviewing it together with the threshold-VC approach of Koltchinskii and Mendelson (2015), nevertheless stated that the known results covered only “certain special cases of $A ^ { \ast }$ and that the problem for general A under the small-ball property remained open (Banerjee et al., 2015, p. 1754). Our contribution is a structural treatment of that arbitrary-set formulation: constant-width pathwise failure, exact encoding of arbitrary finite range spaces, the sharp occupancy law, and same-geometry isotropic and smooth separations. Section 7 gives the full comparison.

Scope. Our impossibility concerns what follows uniformly from marginal small-ball information alone. Moment or increment assumptions and sparsity-specific structure can couple nearby directions and thereby exclude our unrestricted encodings. The appendices provide all constants, measurability conventions, and proofs.

## 2. Problem setup and proof ideas

Let V be a finite-dimensional Euclidean space, write $S ( V ) = \{ v \in V : \| v \| _ { 2 } = 1 \}$ , and let $A \subseteq S ( V )$ be nonempty. For a norm R and $\theta \neq 0$ , set ${ \mathcal { D } } ( { \mathcal { R } } , \theta ) = \{ h : { \mathcal { R } } ( \theta + t h ) \leq$ ${ \mathcal { R } } ( \theta )$ for some $t > 0 \}$ . All cones below are closed and polyhedral.

We use the fixed radial variable $L = 1 + e ^ { G _ { 0 } }$ , where $G _ { 0 } \sim \mathsf { N } ( 0 , 1 )$ . It has finite moments of every order but $\mathbb { E } e ^ { s L } = \infty$ for every $s > 0$ . An independent Rademacher sign centers our designs. We call a random vector directionally heavy-tailed if

$$
\mathbb { E } e ^ { s | \langle Z , v \rangle | } = \infty \qquad \mathrm { f o r e v e r y } v \neq 0 \mathrm { a n d e v e r y } s > 0 .\tag{4}
$$

For range-space encodings with possibly degenerate marginals, “heavy-tailed radial multiplier” refers to this same L.

The proof first encodes incidence in a narrow cap and then lifts it to a descent cone; isotropic negative and positive extensions complete the picture.

Narrow-cap incidence encoding. For a finite range space $\big \{ C _ { 1 } , \dots , C _ { M } \big \}$ , place each $u _ { j }$ near one common anchor and let a row store $( \mathbf { 1 } \{ Y \in C _ { j } \} ) _ { j \leq M }$ . Then $| \langle Z , u _ { j } \rangle | \ge 1$ exactly when $Y \in C _ { j }$ . Shrinking the perturbation drives width to zero without changing these events.

From finitely many directions to a descent cone. A finite spherical set is not yet a regularizer geometry. Transporting the positive orthant through a narrow embedding yields a polyhedral norm whose cone directions are normalized mixtures of the generators. Averaging preserves uniform small-ball mass over the cone, while a missed range annihilates an extreme ray.

Restoring isotropy without losing the witness. Start with polynomially many anchored Rademacher row types having near-identity covariance, constant slab mass, and wellconditioned small submatrices; whitening gives a tight frame. After observing types $D _ { ; }$ normalize the residual obtained by projecting the anchor off their span. It annihilates every observed row and stays within $O ( \sqrt { | D | / p } )$ of the anchor. A cap-to-cone lemma controls the full cone’s width.

Why isotropy still yields a positive theorem. For a centered isotropic design, translate A by a minimum enclosing center a and project the symmetrized process onto the linear space span $( A - a )$ parallel to af(A). Its expected supremum is at most $r ( A ) \sqrt { q ( A ) }$ , giving the dimension–radius bound. On our family, $q ( A ) = p$ and $r ( A ) ^ { 2 } \asymp k / p ,$ matching the pathwise scale k.

## 3. A constant-width pathwise counterexample

The first theorem already rules out (3). Its role is to expose the coverage obstruction in an elementary form before the universal encoding.

Theorem 1 (Explicit constant-width counterexample) For every even $m \geq 2 ,$ , there are a Euclidean space $V _ { m }$ ofdimension m, a polyhedral norm $\mathcal { R } _ { m } ,$ , a point ${ \theta } _ { m } ^ { \star } ,$ , and a centered, directionally heavy-tailed $Z _ { m } \in V _ { m }$ with law $P _ { m }$ . Set $A _ { m } = { \mathcal { D } } ( { \mathcal { R } } _ { m } , \theta _ { m } ^ { \star } ) \cap S ( V _ { m } )$ . Then

$$
\mathsf { Q } _ { 1 / ( 2 \sqrt { 2 } ) } ( P _ { m } , A _ { m } ) \ge \frac { 1 } { 3 } , \qquad w ( A _ { m } ) \le 2 \sqrt { 2 / \pi } , \qquad \mathrm { V C } ( \mathcal { T } _ { 1 } ( A _ { m } ) | _ { \mathrm { s u p p } P _ { m } } ) \ge \frac { m } { 2 } .\tag{5}
$$

Moreover,for every $n \leq m / 2$ and every realization of the sample,

$$
\Lambda _ { n } ( X ; A _ { m } ) = 0 .\tag{6}
$$

The radial law L is the same in every dimension.

Here ${ \mathcal { T } } _ { \alpha } ( A ) = \{ \{ z : | \langle z , u \rangle | \geq \alpha \} : u \in A \}$ . The construction separates an anchor coordinate t from a balanced perturbation y. Work in $\begin{array} { r } { V _ { m } = \{ ( t , y / m ) : \sum _ { i } y _ { j } = 0 \} } \end{array}$ and define

$$
\begin{array} { c } { C _ { m } = \{ ( t , y / m ) : t \geq 0 , | y _ { j } | \leq t \} , } \\ { \mathcal { R } _ { m } ( t , y / m ) = \operatorname* { m a x } \{ | t | , \underset { j } { \operatorname* { m a x } } | t - y _ { j } | , \underset { j } { \operatorname* { m a x } } | t + y _ { j } | \} . } \end{array}\tag{7}
$$

The norm makes $C _ { m } = \mathcal { D } ( \mathcal { R } _ { m } , ( - 1 , 0 ) )$ . Since $\| y / m \| _ { 2 } \leq t / { \sqrt { m } }$ , its normalized directions occupy an $O ( m ^ { - 1 / 2 } )$ cap, explaining the constant width. For visibility, let J be uniform on [m] and σ an independent Rademacher sign, set $\begin{array} { r } { r _ { j } = e _ { 0 } + m e _ { j } - \sum _ { \ell = 1 } ^ { m } e _ { \ell } , } \end{array}$ , and take $Z _ { m } = \sigma L r _ { J } . \mathrm { A t } v = ( t , y / m ) \in C _ { m } ,$ , type j has score $t + y _ { j }$ . These scores lie in [0, 2t] and average to t, so at least $m / 3$ are at least $t / 2$ . Normalization gives the uniform small-ball bound.

For the pathwise witness, let D be the distinct observed types. When $| D | \le m / 2$ , the assignments $s _ { j } ~ = ~ - 1$ on $D$ extend to a balanced $s \in \{ - 1 , 1 \} ^ { m }$ . Then $u _ { s } \propto ( 1 , s / m )$ belongs to $A _ { m }$ , and every observed score vanishes because $\langle r _ { j } , u _ { s } \rangle \propto 1 + s _ { j } = 0$ . Thus one data-dependent direction annihilates the sample. Extending arbitrary signs on any prescribed $m / 2$ types also gives shattering. Appendix B supplies the norm, width, tail, and boundary calculations.

Corollary 2 (Uniform failure of the Gaussian-width law) Fix any $c _ { 1 } > 0 , 0 \le c _ { 2 } < \infty$ and integer $n _ { 0 } \geq 1$ . There are $n \geq n _ { 0 }$ , a spherical descent-cone section A, and a design with thefixed small-ball constants in Theorem 1for which in $\operatorname { f } _ { u \in A } \| X u \| _ { 2 } ^ { 2 } < c _ { 1 } n - c _ { 2 } w ( A ) ^ { 2 }$ with probability one.

Indeed, take $m = 2 n$ and then n large enough. Thus a cone that has constant complexity according to Gaussian width can require a number of heavy-tailed measurements proportional to its ambient dimension. The obstruction is not a rare failure of concentration.

## 4. Threshold occupancy governs the small-ball worst case

For $u \in A$ , let $I _ { \alpha } ( u ) = \{ i \in [ n ] : | \langle Z _ { i } , u \rangle | \ge \alpha \}$ collect the rows that α-detect u, and write ${ \widehat { q } } _ { n , \alpha } ( A ) = n ^ { - 1 } \operatorname* { i n f } _ { \boldsymbol { u } \in A } | I _ { \alpha } ( \boldsymbol { u } ) |$ |. Every detected row contributes at least $\alpha ^ { 2 }$ to the empirical energy, so deterministically

$$
\begin{array} { r } { \Lambda _ { n } ( X ; A ) \geq \alpha ^ { 2 } \widehat { q } _ { n , \alpha } ( A ) . } \end{array}\tag{8}
$$

Thus uniform threshold coverage is sufficient. It is not pointwise necessary for a fixed law—a few unusually large observations may still supply quadratic energy—but the results below give the relevant minimax converse: every finite coverage obstruction can be realized as an RE obstruction at arbitrarily small Gaussian width.

## 4.1. Arbitrary range spaces in narrow caps and descent cones

Let $( \Omega , \mu , \mathcal { C } )$ be a finite probability range space with full support, meaning $\Omega = \operatorname { s u p p } ( \mu )$ and let $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { M } \}$ with $\mu ( C _ { j } ) \ge \beta$ . Its VC dimension is the largest number of row types on which all hit–miss labelings occur. For a latent sample $Y _ { 1 } , \dots , Y _ { n }$ , write $\begin{array} { r } { \mu _ { n } ( C ) = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \{ Y _ { i } \in C \} } \end{array}$

Theorem 3 (Exact range-space representation) For every $\eta > 0$ there are a centered design with the fixed heavy-tailed radial multiplier and a finite $A = \{ u _ { 1 } , \ldots , u _ { M } \} \subset \mathbb { S } ^ { M }$ such that $w ( A ) \leq \eta , \mathsf { Q } _ { 1 } ( P , A ) \geq \beta$ , and, on the design support,

$$
\mathbf { 1 } \{ | \langle Z , u _ { j } \rangle | \ge 1 \} = \mathbf { 1 } \{ Y \in C _ { j } \} \qquad ( j \in [ M ] ) .\tag{9}
$$

Consequently the support-restricted threshold class and C have the same VC dimension. For every sample,

$$
\operatorname* { i n f } _ { u \in A } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \operatorname* { m i n } \{ 1 , \langle Z _ { i } , u \rangle ^ { 2 } \} = \operatorname* { m i n } _ { j \leq M } \mu _ { n } ( C _ { j } ) ,\tag{10}
$$

and $\Lambda _ { n } ( X ; A ) > 0$ exactly when the latent sample hits every $C _ { j }$

The representation is explicit. Take $u _ { j } = ( e _ { 0 } + \varepsilon e _ { j } ) / \sqrt { 1 + \varepsilon ^ { 2 } }$ and

$$
Z = \frac { \sqrt { 1 + \varepsilon ^ { 2 } } } { \varepsilon } \sigma L \sum _ { j } { \bf 1 } \{ Y \in C _ { j } \} e _ { j } \in \mathbb { R } ^ { M + 1 } .
$$

Then $\langle Z , u _ { j } \rangle = \sigma L \mathbf { 1 } \{ Y \in C _ { j } \}$ , while $w ( A ) \leq \varepsilon \sqrt { 2 \log ( 2 M ) }$ . Decreasing ε hides an arbitrary incidence pattern in an arbitrarily narrow cap without changing a single threshold event. This exact identity, rather than a comparison inequality, is what makes the subsequent lower bounds lossless.

The finite representation does not by itself establish a statement about norm-regularized recovery. The next theorem shows that the same missed-range witness survives after passing to the full spherical section of a descent cone.

Theorem 4 (Descent-cone universality) For every finite range space above and $\eta > 0 .$ there are an explicit polyhedral norm $\mathcal { R } _ { \varepsilon }$ on an M-dimensional space $V _ { \varepsilon } ,$ , a point $\theta _ { \varepsilon } ^ { \star }$ and a centered design with the fixed heavy-tailed radial multiplier such that, for $A _ { \varepsilon } =$ ${ \mathcal { D } } ( { \mathcal { R } } _ { \varepsilon } , \theta _ { \varepsilon } ^ { \star } ) \cap S ( V _ { \varepsilon } )$

$$
w ( A _ { \varepsilon } ) \leq \eta , \qquad \mathsf { Q } _ { 1 } ( P , A _ { \varepsilon } ) \geq \frac { \beta } { 2 - \beta } .\tag{11}
$$

Ifa latent sample misses one $C _ { j } ,$ , then $\Lambda _ { n } ( X ; A _ { \varepsilon } ) = 0 .$

For intuition, the injective map $T _ { \varepsilon } a = ( \mathbf { 1 } ^ { \top } a , \varepsilon a )$ transports the $\ell _ { \infty }$ norm to $V _ { \varepsilon } = T _ { \varepsilon } \mathbb { R } ^ { M }$ Write $\Delta _ { M } = \{ q \in \mathbb { R } _ { + } ^ { M } : \mathbf { 1 } ^ { \top } q = 1 \}$ for the probability simplex. A $\mathbf { t } \theta _ { \varepsilon } ^ { \star } = - T _ { \varepsilon } \mathbf { 1 }$ ，

$$
\mathcal { D } ( \mathcal { R } _ { \varepsilon } , \theta _ { \varepsilon } ^ { \star } ) = T _ { \varepsilon } \mathbb { R } _ { + } ^ { M } , \qquad u ( q ) = \frac { e _ { 0 } + \varepsilon q } { \sqrt { 1 + \varepsilon ^ { 2 } \| q \| _ { 2 } ^ { 2 } } } , \quad q \in \Delta _ { M } .\tag{12}
$$

If $\begin{array} { r } { f _ { q } ( Y ) = \sum _ { i } q _ { j } { \bf 1 } \{ Y \in C _ { j } \} } \end{array}$ , then $0 \leq f _ { q } \leq 1$ and $\mathbb { E } f _ { q } \geq \beta$ , so $P \{ f _ { q } \ge \beta / 2 \} \ge \beta / ( 2 - \beta )$ The scaling of Z turns this event into the threshold at one. A missed $C _ { j }$ annihilates the generator $u ( e _ { j } )$ , while a direct Gaussian calculation controls the full cone width. See Appendix D.

The obstruction is therefore not tied to one specially chosen cone: arbitrarily narrow Euclidean geometry can hide any finite combinatorial visibility pattern.

## 4.2. The VC envelope and its sharp low-mass regime

For a given law P, call A pointwise measurable when $\{ \mathbf { 1 } _ { C } | _ { \mathrm { s u p p } P } : C \in \mathcal { T } _ { 1 } ( A ) \}$ has a countable subclass whose pointwise sequential closure contains the whole class. Let $\mathsf { N } _ { \mathrm { R E } } ( d , \beta , \delta )$ be the least integer n such that, for every $N \geq n$ , every pointwise-measurable spherical set A in a finite-dimensional Euclidean space, and every design satisfying $\mathsf Q _ { 1 } ( P , A ) \ge \beta$ and $\operatorname { V C } ( { \mathcal { T } } _ { 1 } ( A ) | _ { \operatorname { s u p p } P } ) \leq d ,$ , one has $\Lambda _ { N } ( X ; A ) \ge \beta / 2$ with probability at least $1 - \delta$ . The requirement is marginal for each N. No event simultaneous over all N is asserted.

Theorem 5 (VC envelope and sharp fixed-d low-mass law) There are universal $c , C >$ 0 such that, for every integer $d \geq 1$ and $0 < \beta , \delta \le 1 / 4$

$$
c \frac { d + \log ( 2 / \beta ) + \log ( 1 / \delta ) } { \beta } \leq \mathsf { N } _ { \mathrm { R E } } ( d , \beta , \delta ) \leq C \frac { d \log ( 2 / \beta ) + \log ( 1 / \delta ) } { \beta } .\tag{13}
$$

Moreover,for everyfixed $d \geq 1$ there is $\beta _ { d } > 0$ such that, whenever $0 < \beta \le \beta _ { d }$

$$
\mathsf { N } _ { \mathrm { R E } } ( d , \beta , \delta ) \ge c \frac { d \log ( 2 / \beta ) + \log ( 1 / \delta ) } { \beta } .\tag{14}
$$

Thus the upper bound is optimal up to universal multiplicative constants in the low-mass regimefor eachfixed VC dimension.

The upper bound is a constant-relative-error VC approximation followed by (8). The general lower envelope combines three transparent mechanisms: all d-subsets of a uniform ground set give $d / \beta$ , disjoint singleton ranges give $\beta ^ { - 1 } \log ( 1 / \beta )$ already at $\mathrm { V C }$ dimension one, and one range of mass $\beta$ gives $\beta ^ { - 1 } \log ( 1 / \delta )$ . The fixed-d product term follows from the sharp ε-net lower bound of Komlos et al.´ (1992). Pach and Tardos (2013) exhibit the logarithm even for geometric VC-two systems. Appendix E states the needed consequence of the external result and proves every reduction. The classical range-space inequalities are not new. Their exact realization as arbitrarily narrow-width RE problems is the new structural statement.

## 5. Isotropy does not restore Gaussian width

The preceding encodings use unconstrained row magnitudes. One might therefore hope that exact covariance normalization reconnects Euclidean proximity with common visibility. The following construction shows that this hope is false.

Lemma 6 (Anchored tight frame) There are absolute $a _ { 0 } , b _ { 0 } , c _ { 0 } , c _ { 1 } , C _ { 1 } > 0$ such that,for all sufficiently large $p$ and $M = p ^ { 4 }$ , vectors $b _ { 1 } , \dots , b _ { M } \in \mathbb { R } ^ { p }$ and a unit $u _ { \star }$ exist with

$$
\frac { 1 } { M } \sum _ { j } b _ { j } b _ { j } ^ { \top } = I _ { p } , \qquad \langle b _ { j } , u _ { \star } \rangle = 1 , \qquad \frac { 1 } { M } | \{ j : | \langle b _ { j } , v \rangle | \ge a _ { 0 } \} | \ge b _ { 0 }\tag{15}
$$

for every unit v. $H B _ { D }$ has rows $b _ { j } ^ { \top }$ , then, whenever $| D | \leq c _ { 0 } p / \log p$

$$
c _ { 1 } p I _ { | D | } \preceq B _ { D } B _ { D } ^ { \top } \preceq C _ { 1 } p I _ { | D | } .\tag{16}
$$

Start from $a _ { j } = \left( 1 , \xi _ { j } \right)$ with independent Rademacher vectors $\xi _ { j } \in \{ - 1 , 1 \} ^ { p - 1 }$ . Their empirical covariance is close to identity. A uniform slab law gives global small-ball mass, while a net argument and a union bound make every small row submatrix well conditioned. Whitening then yields the exact tight frame without losing the common anchor. Appendix F records the probability budget and the net-to-operator step.

For $D \subseteq [ M ]$ with $| D | \leq c _ { 0 } p /$ log p, let $P _ { D }$ project onto span $\{ b _ { j } : j \in D \}$ , with $P _ { \mathcal { O } } = 0$ and set $u _ { D } = ( I - P _ { D } ) u _ { \star } / \| ( I - P _ { D } ) u _ { \star } \| _ { 2 }$ . The Gram bound gives $\| u _ { D } - u _ { \star } \| _ { 2 } ^ { 2 } \asymp | D | / p$ and $B _ { D } u _ { D } = 0$ . Define $C _ { p , k } = \mathrm { c o n e } \{ u _ { D } : | D | \leq k \}$ and $A _ { p , k } = C _ { p , k } \cap \mathbb { S } ^ { p - 1 }$

Theorem 7 (Isotropic heavy-tail separation) There are absolute $c , C , \alpha _ { 0 } , \beta _ { 0 } > 0$ such that, for large p and $1 \leq k \leq c p /$ log p, thefollowing statements hold.

(i) $A _ { p , k }$ is the spherical section of a polyhedral norm descent cone and

$$
w ( A _ { p , k } ) \leq C \left( k { \sqrt { \frac { \log p } { p } } } + { \frac { k } { \sqrt { p } } } \right) .\tag{17}
$$

(ii) There are centered isotropic laws $P _ { \mathsf { H } }$ and $P _ { \mathsf { G } } = { \mathsf { N } } ( 0 , I _ { p } )$ with common absolute smallball constants

$$
\operatorname* { i n f } _ { v \in \mathbb { S } ^ { p - 1 } } P _ { s } \{ | \langle Z , v \rangle | \ge \alpha _ { 0 } \} \ge \beta _ { 0 } , \qquad s \in \{ \mathsf { H } , \mathsf { G } \} .\tag{18}
$$

The law $P _ { \mathsf { H } }$ has finite moments of every order and no positive exponential moment in any nonzero marginal, yet $\Lambda _ { n } ( X _ { \mathsf { H } } ; A _ { p , k } ) = 0$ on every sample path for every $n \leq k .$ (iii) Under $P _ { \mathsf { G } } ,$ , for every $t > 0$ , with probability at least $1 - e ^ { - t ^ { 2 } / 2 }$

$$
\operatorname* { i n f } _ { u \in A _ { p , k } } \| X _ { \mathsf { G } } u \| _ { 2 } \geq \sqrt { n - 1 } - w ( A _ { p , k } ) - t .\tag{19}
$$

The heavy-tailed law is $Z _ { \mathsf { H } } = \sigma \rho b _ { J }$ , where J is uniform on $[ M ]$ and $\rho = L / \sqrt { \mathbb { E } L ^ { 2 } }$ Tightness gives isotropy and the slab law gives global small-ball. Given a sample, the set D of distinct row types has size at most n, and $u _ { D }$ annihilates every row. Conversely, (19) is Gordon’s escape theorem. The cap-to-cone argument gives (17) for the full cone rather than only its generators. The complete proof is in Appendix G.

Corollary 8 (Constant-width separation on one descent cone) Universal constants can be chosen so that the following holds. For every sufficiently large p, there is a descentcone section $A _ { p } \subset \mathbb { S } ^ { p - 1 }$ with $w ( A _ { p } ) ~ \leq ~ C _ { G }$ and two centered isotropic designs with common absolute small-ball constants. For every $0 < \delta < 1$ , Gaussian measurements satisfy $\Lambda _ { n } ( X _ { \mathsf { G } } ; A _ { p } ) \geq c _ { G }$ with probability at least $1 - \delta$ whenever $n \geq C _ { G } ( 1 + \log ( 1 / \delta ) )$ ), whereas the heavy-tailed design has $\Lambda _ { n } ( X _ { \mathsf { H } } ; A _ { p } ) = 0$ on every sample path for every $n \leq c _ { H } \sqrt { p / \log p } .$

Choose $k = \lfloor a \sqrt { p / \log p } \rfloor$ with the sufficiently small universal constant $a > 0$ fixed in Appendix G. The two designs have the same covariance and the same marginal small-ball constants. What differs is how visibility events are coupled across directions. Alternatively, $k = \lfloor p ^ { 1 / 3 } \rfloor$ gives $w ( A _ { p , k } )  0$ while the deterministic failure horizon diverges.

## 5.1. Smooth-density robustness

The isotropic counterexample above is discrete in its row type. To show that the obstruction is not caused by atoms, smooth it with an independent Gaussian. For $H \sim \mathsf { N } ( 0 , I _ { p } )$ and $\tau > 0$ , set

$$
Z _ { \mathsf { H } , \tau } = \frac { Z _ { \mathsf { H } } + \tau H } { \sqrt { 1 + \tau ^ { 2 } } } .\tag{20}
$$

The normalization preserves isotropy, while Gaussian convolution gives an everywherepositive $C ^ { \infty }$ density. Smoothing removes the exact kernel, but the next result shows that the restricted energy remains of order $\tau ^ { 2 }$

Corollary 9 (Smooth isotropic robustness) Under the construction ofTheorem 7, there are absolute $\tau _ { 0 } , \bar { \alpha } , \bar { \beta } , c _ { s } > 0$ such that, for every sufficiently large p, every admissible $1 \leq k \leq c p /$ log p, and every $0 < \tau \leq \tau _ { 0 }$ , the law of $Z _ { \mathsf { H } , \tau }$ is centered and isotropic, has finite moments of every order and no exponential moment in any nonzero direction, has an everywhere-positive $C ^ { \infty }$ density on $\mathbb { R } ^ { p }$ , and satisfies the global small-ball condition $( { \bar { \alpha } } , { \bar { \beta } } )$ For each fixed integer $1 \leq n \leq k$

$$
\mathbb { P } \{ \Lambda _ { n } ( X _ { \mathsf { H } , \tau } ; A _ { p , k } ) \le 2 \tau ^ { 2 } \} \ge 1 - e ^ { - c _ { s } n } .\tag{21}
$$

Conditionally on the row types, the adaptive witness $u _ { D }$ is independent of the Gaussian perturbations, and its normalized empirical energy is $( \tau ^ { 2 } / ( 1 + \tau ^ { 2 } ) ) ( \chi _ { n } ^ { 2 } / n )$ . A standard chi-square bound gives the stated probability. The witness is no longer an exact kernel direction, but $\sqrt { \Lambda _ { n } }$ can be arbitrarily small while the small-ball constants stay fixed.

## 5.2. A distribution-free isotropic fallback

For bounded nonempty A, let $r ( A ) = \operatorname* { i n f } _ { a } \operatorname* { s u p } _ { u \in A } \| u - a \| _ { 2 } \operatorname { a n d } q ( A ) = \dim \operatorname { a f f } ( A )$

Theorem 10 (Dimension–radius RE bound) Let Z be centered and isotropic and let $\varnothing \neq A \subseteq \mathbb { S } ^ { p - 1 }$ . Suppose $\mathsf Q _ { \alpha } ( P , A ) \ge \beta f o r$ some $\alpha > 0$ and $0 < \beta \leq 1$ . For every $t > 0$ with probability at least $1 - e ^ { - t ^ { 2 } / 2 } ,$

$$
\operatorname* { i n f } _ { u \in A } \| X u \| _ { 2 } \geq { \frac { \alpha \beta } { 2 } } { \sqrt { n } } - 2 r ( A ) { \sqrt { q ( A ) } } - { \frac { \alpha t } { 2 } } .\tag{22}
$$

Corollary 11 (Sample-complexity consequence) Under the assumptions ofTheorem $I O ,$ for every $0 < \delta < 1$ there is a universal C such that

$$
n \geq C \left( { \frac { q ( A ) r ( A ) ^ { 2 } } { \alpha ^ { 2 } \beta ^ { 2 } } } + { \frac { \log ( 1 / \delta ) } { \beta ^ { 2 } } } \right) \quad \Longrightarrow \quad \Lambda _ { n } ( X ; A ) \geq { \frac { \alpha ^ { 2 } \beta ^ { 2 } } { 1 6 } }\tag{23}
$$

with probability at least $1 - \delta .$

The proof of Theorem 10 is short. For $H _ { n } = n ^ { - 1 / 2 } \sum _ { i } \varepsilon _ { i } Z _ { i }$ , translate by a minimum enclosing center a and project onto $L = \operatorname { s p a n } ( A - a )$ . Isotropy gives

$$
\mathbb { E } \operatorname* { s u p } _ { u \in A } \langle H _ { n } , u \rangle \leq r ( A ) \mathbb { E } \| P _ { L } H _ { n } \| _ { 2 } \leq r ( A ) { \sqrt { q ( A ) } } .\tag{24}
$$

Substitution into the empirical small-ball inequality proves (22). For $A _ { p , k } ,$ Appendix G proves $q ( A _ { p , k } ) = p$ and $r ( A _ { p , k } ) ^ { 2 } \asymp k / p$ . At fixed confidence and fixed small-ball constants, the worst-case sample complexity on this family is therefore $\Theta ( k )$ . This is a family-wise sharpness statement, not a joint minimax characterization for every prescribed pair of threshold and radius complexities. Thus isotropy does not recover Gaussian width, but it prevents arbitrarily bad behavior once the set is controlled by its affine dimension and enclosing radius.

## 6. Consequences and scope

Our polyhedral descent-cone sections are closed, so their spherical sections are compact. Whenever $\Lambda _ { n } ( X ; A ) = 0$ , the minimum is attained by some $u \in A \cap$ ker X. Consequently the noiseless program

$$
\operatorname* { m i n } _ { \theta } \mathcal { R } ( \theta ) \quad \mathrm { s u b j e c t { t o } } \quad X \theta = X \theta ^ { \star }\tag{25}
$$

does not uniquely recover $\theta ^ { \star }$ . The explicit and isotropic counterexamples therefore translate directly into failure of norm-regularized recovery.

For the smoothed construction, write $\kappa _ { n } ( X ; A ) = \sqrt { \Lambda _ { n } ( X ; A ) }$ for the normalized restricted singular value and define the realized-design inverse modulus $K _ { n } ( X ; A ) \ =$ $1 / \kappa _ { n } ( X ; A )$ , with $K _ { n } = \infty$ when $\kappa _ { n } = 0$ . On the event in Corollary 9, $K _ { n } \ge 1 / ( \sqrt { 2 } \tau )$ and the corresponding quadratic modulus $1 / \Lambda _ { n }$ is at least $1 / ( 2 \tau ^ { 2 } )$ . This is a deterministic conditioning statement for each realized design; its witness may depend on X, so it is not a two-fixed-parameter statistical minimax claim. The obstruction is nevertheless relevant to noisy recovery, not only exact interpolation.

The two positive mechanisms in the paper are complementary. Relative threshold occupancy is distribution-adapted and requires no moments. The dimension–radius bound is Euclidean but uses isotropy. Stronger increment or moment assumptions can recover finer chaining and, in suitable models, Gaussian-width behavior. We do not claim that threshold VC complexity and dimension–radius complexity form a joint minimax formula for every distribution class.

## 7. Related work

Restricted eigenvalues and Gaussian geometry. RE theory underpins the Lasso, Dantzig selector, and decomposable regularizers (Bickel et al., 2009; Negahban et al., 2012). For Gaussian measurements, escape through a mesh, Gaussian width, and statistical dimension connect descent cones to sharp recovery transitions (Gordon, 1988; Chandrasekaran et al., 2012; Amelunxen et al., 2014; Tropp, 2015a). Related bounds cover sub-Gaussian, correlated, anisotropic, and stable-rank models (Banerjee et al., 2014; Raskutti et al., 2010; Rudelson and Zhou, 2013; Kasiviswanathan and Rudelson, 2018); broader universality laws still require entry or tail structure beyond marginal visibility (Oymak and Tropp, 2018). All constrain the joint process indexed by A.

Small-ball methods and distribution-dependent complexity. The small-ball method lower-bounds nonnegative empirical processes without upper-tail concentration (Koltchinskii and Mendelson, 2015; Mendelson, 2015; Tropp, 2015a). Its distribution-dependent Rademacher or localized complexities become quadratic and multiplier fixed points in regularized estimation (Lecue and Mendelson´ , 2018, 2017a). Extensions relax uniform small-ball assumptions (Mendelson, 2021), and moment-based covariance bounds offer another route (Oliveira, 2016). These retain information absent from ordinary Gaussian width.

Positive results under stronger tail assumptions. Weak coordinate moments can suffice for sparse recovery (Lecue and Mendelson´ , 2017b; Dirksen et al., 2018). Under a uniform $\psi _ { 1 }$ assumption, Sivakumar et al. (2015) use exponential width and threshold VC dimension at scale $n \gtrsim d / \beta ^ { 2 }$ . Sub-Weibull bounds (Kuchibhotla and Chakrabortty, 2022), mixed-tail chaining (Dirksen, 2015; Genzel and Kipp, 2022), and thresholding (Wei, 2018) likewise add coupling or robustness that excludes our encoding.

Earlier sparse-recovery results and the 2015 open problem. First circulated in 2014, the spiky construction of Lecue and Mendelson ´ (2017b) has centered isotropic rows with bounded fourth moments and hence a uniform small-ball condition. For an $\ell _ { 1 } \cdot$ -related sparserecovery geometry, it can force the relevant RE and compatibility quantities to vanish with constant probability when n log $n \lesssim p ;$ the companion note (Lecue and Mendelson´ , 2014) records a basis-pursuit failure. Thus this construction already gives a special-case counterexample to a fully uniform implication from marginal small-ball behavior to Gaussianwidth-controlled RE. The COLT note cited Lecue–Mendelson in its discussion of unit´ s-sparse directions and the Koltchinskii–Mendelson threshold-VC method, but explicitly characterized the available results as covering only “certain special cases of $A ^ { \prime \prime }$ and left the question for general $A \subseteq \mathbb { S } ^ { p - 1 }$ under the small-ball property open (Banerjee et al., 2015, p. 1754). Our results go beyond that geometry-specific instance by establishing constantwidth pathwise failure, exact realization of arbitrary finite threshold systems, descent-cone universality, the sharp fixed-d occupancy law, and same-geometry isotropic and smooth separations.

Threshold classes and range spaces. The COLT note highlighted VC-controlled threshold bounds (Koltchinskii and Mendelson, 2015; Banerjee et al., 2015). Classical random ε-nets and relative approximations depend on VC dimension (Haussler and Welzl, 1987; Li et al., 2001; Har-Peled and Sharir, 2011); Komlos et al.´ (1992) prove the logarithmic lower bound, and Pach and Tardos (2013) give a geometric VC-two construction. Those inequalities are not new here. Our contribution is their exact realization on arbitrarily narrow spherical sets and a descent-cone lift turning every missed range into a kernel witness.

## 8. Conclusion

We show that the Gaussian-width principle posed in the COLT 2015 open-problem note does not follow from a uniform small-ball condition: it can fail pathwise on a constant-width descent cone. Empirical RE requires one sample to cover all relevant directions, whereas bare small-ball visibility is only pointwise. Gaussian and sub-Gaussian increment control couples nearby directions, but a marginal small-ball condition does not. Encoding arbitrary threshold range spaces in narrow spherical caps yields pathwise counterexamples and the sharp low-mass law at each fixed threshold VC dimension. The separation survives isotropy, moments of every order, and positive smooth densities, while isotropy still gives a sharp dimension–radius fallback. A natural next question is which minimal structure between marginal visibility and full increment control restores a Gaussian-width law.

## References

Dennis Amelunxen, Martin Lotz, Michael B. McCoy, and Joel A. Tropp. Living on the edge: Phase transitions in convex programs with random data. Information and Inference, 3(3):224–294, 2014.

Arindam Banerjee, Sheng Chen, Farideh Fazayeli, and Vidyashankar Sivakumar. Estimation with norm regularization. In Advances in Neural Information Processing Systems 27, pages 1556–1564, 2014.

Arindam Banerjee, Sheng Chen, and Vidyashankar Sivakumar. Open problem: Restricted eigenvalue condition for heavy tailed designs. In Peter Grunwald, Elad Hazan, and Satyen¨ Kale, editors, Proceedings of the 28th Conference on Learning Theory, volume 40 of Proceedings ofMachine Learning Research, pages 1752–1755, Paris, France, 3–6 July 2015. PMLR. URL https://proceedings.mlr.press/v40/Banerjee15. html.

Peter J. Bickel, Ya’acov Ritov, and Alexandre B. Tsybakov. Simultaneous analysis of lasso and dantzig selector. The Annals ofStatistics, 37(4):1705–1732, 2009.

Stephane Boucheron, G ´ abor Lugosi, and Pascal Massart. ´ Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 2013.

Venkat Chandrasekaran, Benjamin Recht, Pablo A. Parrilo, and Alan S. Willsky. The convex geometry of linear inverse problems. Foundations of Computational Mathematics, 12(6): 805–849, 2012.

Sjoerd Dirksen. Tail bounds via generic chaining. Electronic Journal of Probability, 20(53): 1–29, 2015. doi: 10.1214/EJP.v20-3760.

Sjoerd Dirksen, Guillaume Lecue, and Holger Rauhut. On the gap between restricted ´ isometry properties and sparse recovery conditions. IEEE Transactions on Information Theory, 64(8):5478–5487, August 2018. doi: 10.1109/TIT.2016.2570244.

Martin Genzel and Christian Kipp. Generic error bounds for the generalized lasso with sub-exponential data. Sampling Theory, Signal Processing, and Data Analysis, 20:15, 2022. doi: 10.1007/s43670-022-00032-8.

Yehoram Gordon. On milman’s inequality and random subspaces which escape through a mesh in R<sup>n</sup>. In Joram Lindenstrauss and Vitali D. Milman, editors, Geometric Aspects of Functional Analysis, volume 1317 of Lecture Notes in Mathematics, pages 84–106. Springer, Berlin, Heidelberg, 1988.

Sariel Har-Peled and Micha Sharir. Relative (p, ε)-approximations in geometry. Discrete & Computational Geometry, 45(3):462–496, 2011. doi: 10.1007/s00454-010-9248-1.

David Haussler and Emo Welzl. ε-nets and simplex range queries. Discrete & Computational Geometry, 2:127–151, 1987.

Shiva Prasad Kasiviswanathan and Mark Rudelson. Restricted eigenvalue from stable rank with applications to sparse linear regression. In Proceedings of the 31st Conference on Learning Theory, volume 75 of Proceedings of Machine Learning Research, pages 1011–1041. PMLR, 2018.

Vladimir Koltchinskii and Shahar Mendelson. Bounding the smallest singular value of a random matrix without concentration. International Mathematics Research Notices, 2015 (23):12991–13008, 2015. doi: 10.1093/imrn/rnv096.

Janos Koml ´ os, J ´ anos Pach, and Gerhard J. Woeginger. Almost tight bounds for ´ ε-nets. Discrete & Computational Geometry, 7(1):163–173, 1992. doi: 10.1007/BF02187833.

Arun Kumar Kuchibhotla and Abhishek Chakrabortty. Moving beyond sub-gaussianity in high-dimensional statistics: Applications in covariance estimation and linear regression. Information and Inference, 11(4):1389–1456, 2022. doi: 10.1093/imaiai/iaac012.

Guillaume Lecue and Shahar Mendelson. Necessary moment conditions for exact recon-´ struction via basis pursuit. arXiv preprint arXiv:1404.3116, 2014.

Guillaume Lecue and Shahar Mendelson. Regularization and the small-ball method ii:´ Complexity dependent error rates. Journal of Machine Learning Research, 18(146):1–48, 2017a.

Guillaume Lecue and Shahar Mendelson. Sparse recovery under weak moment assumptions.´ Journal of the European Mathematical Society, 19(3):881–904, 2017b. doi: 10.4171/ JEMS/682. First circulated as arXiv:1401.2188 in 2014.

Guillaume Lecue and Shahar Mendelson. Regularization and the small-ball method i: Sparse´ recovery. The Annals ofStatistics, 46(2):611–641, 2018. doi: 10.1214/17-AOS1562.

Yi Li, Philip M. Long, and Aravind Srinivasan. Improved bounds on the sample complexity of learning. Journal of Computer and System Sciences, 62(3):516–527, 2001. doi: 10.1006/jcss.2000.1741.

Shahar Mendelson. Learning without concentration. Journal of the ACM, 62(3):21:1–21:25, 2015. doi: 10.1145/2699439.

Shahar Mendelson. Extending the scope of the small-ball method. Studia Mathematica, 256 (2):147–167, 2021. doi: 10.4064/sm190420-21-11.

Sahand N. Negahban, Pradeep Ravikumar, Martin J. Wainwright, and Bin Yu. A unified framework for high-dimensional analysis of m-estimators with decomposable regularizers. Statistical Science, 27(4):538–557, 2012.

Roberto Imbuzeiro Oliveira. The lower tail of random quadratic forms with applications to ordinary least squares. Probability Theory and Related Fields, 166(3–4):1175–1194, 2016. doi: 10.1007/s00440-016-0738-9.

Samet Oymak and Joel A. Tropp. Universality laws for randomized dimension reduction, with applications. Information and Inference, 7(3):337–446, 2018. doi: 10.1093/imaiai/ iax011.

Janos Pach and G ´ abor Tardos. Tight lower bounds for the size of ´ ε-nets. Journal of the American Mathematical Society, 26(3):645–658, 2013.

Garvesh Raskutti, Martin J. Wainwright, and Bin Yu. Restricted eigenvalue properties for correlated gaussian designs. Journal of Machine Learning Research, 11(78):2241–2259, 2010.

Mark Rudelson and Shuheng Zhou. Reconstruction from anisotropic random measurements. IEEE Transactions on Information Theory, 59(6):3434–3447, 2013. doi: 10.1109/TIT. 2013.2243201.

Vidyashankar Sivakumar, Arindam Banerjee, and Pradeep K. Ravikumar. Beyond subgaussian measurements: High-dimensional structured estimation with sub-exponential designs. In Advances in Neural Information Processing Systems 28, pages 2206–2214. Curran Associates, Inc., 2015.

Joel A. Tropp. Convex recovery of a structured signal from independent random linear measurements. In Gotz E. Pfander, editor,¨ Sampling Theory, a Renaissance, pages 67–101. Birkhauser, 2015a.¨

Joel A. Tropp. An introduction to matrix concentration inequalities. Foundations and Trends in Machine Learning, 8(1–2):1–230, 2015b. doi: 10.1561/2200000048.

Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge University Press, 2018.

Xiaohan Wei. Structured recovery with heavy-tailed measurements: A thresholding procedure and optimal rates. arXiv preprint arXiv:1804.05959, 2018.

## Appendix A. Auxiliary facts

This appendix records all proofs and the external tools used in them. Appendix A collects the empirical small-ball inequality, Gordon’s Gaussian escape bound, range-space estimates, and two geometric lemmas proved here. Appendix B proves the explicit constant-width counterexample. Appendices C to E establish exact range-space representation, its descentcone lifting, and the sharp threshold-complexity law. Appendices F and G construct the anchored isotropic frame and prove the isotropic separation. Appendices H to J treat smoothing, the dimension–radius guarantee, and the consequences for convex recovery. Throughout, numerical constants may change from one display to the next.

Unless stated otherwise, all auxiliary random objects introduced within a construction— including $L , \sigma , J , Y , H$ , and their sample copies—are mutually independent. Suprema over non-countable sets are interpreted in outer expectation or outer probability. For the finite or compact polyhedral constructions used here, the relevant suprema are measurable and this distinction disappears.

When an index set lies in a Euclidean subspace V of an ambient space, its Gaussian width is computed using a standard Gaussian vector on $V .$ . Equivalently, one may use an ambient standard Gaussian and project it orthogonally onto $V$ , since all relevant inner products are unchanged.

## A.1. Empirical small-ball and Gaussian escape inequalities

For a set $A \subseteq \mathbb { S } ^ { p - 1 }$ and a distribution $P$ on $\mathbb { R } ^ { p }$ , define

$$
\mathsf { Q } _ { \xi } ( P , A ) = \operatorname* { i n f } _ { u \in A } \mathbb { P } \big ( | \langle Z , u \rangle | \ge \xi \big ) , \qquad W _ { n } ( A ; P ) = \mathbb { E } \operatorname* { s u p } _ { u \in A } \left. \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } Z _ { i } , u \right. ,
$$

where $Z , Z _ { 1 } , \ldots , Z _ { n }$ are distributed according to $P$ and the $\varepsilon _ { i }$ are independent Rademacher variables. We use the following form of the small-ball inequality, which is Proposition 5.1 of Tropp (2015a).

Lemma 12 (Small-ball inequality) For every $\xi , t > 0$ , with probability at least $1 - e ^ { - t ^ { 2 } / 2 }$

$$
\operatorname* { i n f } _ { u \in A } \| X u \| _ { 2 } \geq \xi \sqrt { n } \mathsf { Q } _ { 2 \xi } ( P , A ) - 2 W _ { n } ( A ; P ) - \xi t .
$$

For a standard Gaussian matrix $G \in \mathbb { R } ^ { n \times p }$ , Gordon’s escape theorem gives the companion inequality

$$
\mathbb { P } \left( \operatorname* { i n f } _ { u \in A } \| G u \| _ { 2 } \geq \mathbb { E } \| g _ { n } \| _ { 2 } - w ( A ) - t \right) \geq 1 - e ^ { - t ^ { 2 } / 2 } ,\tag{A.1}
$$

where $g _ { n } \sim \mathsf { N } ( 0 , I _ { n } )$ . We use the elementary estimate $\mathbb { E } \Vert g _ { n } \Vert _ { 2 } \geq { \sqrt { n - 1 } }$

## A.2. Two geometric lemmas

Lemma 13 (From a finite cap to its cone) Let $u _ { \star } \in \mathbb { S } ^ { p - 1 }$ and let $\mathcal { U } \subseteq \mathbb { S } ^ { p - 1 }$ be finite. Suppose

$$
\operatorname* { m a x } _ { u \in \mathcal { U } } \| u - u _ { \star } \| _ { 2 } \leq r < 1 .
$$

Set $C = \mathrm { c o n e } ( \mathcal { U } )$ and $A = C \cap { \mathbb S } ^ { p - 1 }$ . Then

$$
w ( A ) \leq r { \sqrt { 2 \log \left| { \mathcal { U } } \right| } } + { \frac { r ^ { 2 } } { 2 } } { \sqrt { p } } .\tag{A.2}
$$

Moreover, every $v \in A$ satisfies

$$
\| v - u _ { \star } \| _ { 2 } \leq r + r ^ { 2 } / 2 .\tag{A.3}
$$

Proof Let $K = \operatorname { c o n v } ( \mathcal { U } )$ . Every nonzero point in $C$ is a positive multiple of a point in $K ,$ hence each $v \in A$ can be written as $v = x / \lVert x \rVert _ { 2 }$ for some $x \in K$ . Since every $u \in \mathcal { U }$ is a unit vector,

$$
\langle u , u _ { \star } \rangle = 1 - \frac { 1 } { 2 } \| u - u _ { \star } \| _ { 2 } ^ { 2 } \geq 1 - r ^ { 2 } / 2 .
$$

The same inequality holds for $x \in K$ . Also $\| x \| _ { 2 } \leq 1$ , and therefore

$$
\left\| \frac { x } { \| x \| _ { 2 } } - x \right\| _ { 2 } = 1 - \| x \| _ { 2 } \leq 1 - \langle x , u _ { \star } \rangle \leq r ^ { 2 } / 2 .
$$

This proves (A.3). For the width, write $u = u _ { \star } + \left( u - u _ { \star } \right)$ and apply the Gaussian maximal inequality to the centered Gaussian family $\{ \langle g , u - u _ { \star } \rangle : u \in \mathcal { U } \}$

$$
w ( K ) = w ( \mathcal { U } ) \leq r \sqrt { 2 \log | \mathcal { U } | } .
$$

For each $v = x / \lVert x \rVert _ { 2 }$ as above,

$$
\langle g , v \rangle \leq \langle g , x \rangle + { \frac { r ^ { 2 } } { 2 } } \| g \| _ { 2 } .
$$

Taking the supremum and expectation, and using $\mathbb { E } \Vert g \Vert _ { 2 } \leq \sqrt { p } ,$ gives (A.2).

Lemma 14 (Cone realization) Let $C \subset \mathbb { R } ^ { p }$ be a full-dimensional pointed polyhedral cone generated by $c _ { 1 } , \ldots , c _ { N }$ . There is a centrally symmetric full-dimensional polytope B, a point $x ^ { \star } \in \partial B$ , and a polyhedral norm R whose unit ball is B such that

$$
{ \mathcal { D } } ( { \mathcal { R } } , x ^ { \star } ) = C .
$$

The construction can be made explicit from the generators.

Proof Choose $c _ { \circ } \in \operatorname { i n t } ( C )$ . Since the interior is open and is preserved by positive scaling, for each i we have $c _ { \circ } - c _ { i } / ( 2 L ) \in \operatorname { i n t } ( C )$ once L is large enough. Because there are finitely many generators, one may choose a common $L > 0$ such that

$$
2 L c _ { \circ } - c _ { i } \in \operatorname { i n t } ( C ) \qquad ( i \in [ N ] ) .
$$

Set $x ^ { \star } = - L c _ { \mathrm { c } }$ and

$$
B = \operatorname { c o n v } \big ( \{ x ^ { \star } , x ^ { \star } + c _ { i } : i \in [ N ] \} \cup \{ - x ^ { \star } , - x ^ { \star } - c _ { i } : i \in [ N ] \} \big ) .
$$

The set $B$ is centrally symmetric and full dimensional because the $c _ { i }$ span $\mathbb { R } ^ { p }$ . In particular, its center 0 lies in its interior, so the Minkowski functional $\gamma _ { B }$ is a norm. Every displacement from $x ^ { \star }$ to a listed point of $B$ is one of

$$
c _ { i } , \qquad 2 L c _ { \circ } , \qquad 2 L c _ { \circ } - c _ { i } ,
$$

and hence lies in C. Conversely, every generator $c _ { i }$ occurs, so

$$
\mathrm { c o n e } ( B - x ^ { \star } ) = \mathrm { c o n e } \{ c _ { i } , 2 L c _ { \circ } , 2 L c _ { \circ } - c _ { i } : i \in [ N ] \} = C .\tag{A.4}
$$

To verify that $x ^ { \star }$ is on the boundary, choose a linear functional $\ell \in \operatorname { i n t } ( C ^ { * } )$ . Pointedness and full dimensionality imply $\langle \ell , z \rangle > 0$ for every nonzero $z \in C$ . Every other listed point $y$ satisfies $y - x ^ { \star } \in C \setminus \{ 0 \}$ , hence $\langle \ell , y \rangle > \langle \ell , x ^ { \star } \rangle$ . Thus $x ^ { \star }$ is an exposed vertex of $B$ and $\gamma _ { B } ( x ^ { \star } ) = 1$

Finally, the definition of the gauge gives

$$
{ \mathcal { D } } ( \gamma _ { B } , x ^ { \star } ) = \{ h : x ^ { \star } + t h \in B { \mathrm { ~ f o r ~ s o m e ~ } } t > 0 \} = \operatorname { c o n e } ( B - x ^ { \star } ) .
$$

The second equality holds in both directions by writing $h = t ^ { - 1 } ( y - x ^ { \star } )$ or choosing t as the reciprocal of the conic multiplier. Combining this identity with (A.4) proves the claim with $\mathcal { R } = \gamma _ { B }$

## A.3. Range-space estimates

We use two classical consequences of VC theory. First, if a class $\mathcal { C }$ has VC dimension at most d and every $C \in \mathcal { C }$ has probability at least $\beta ,$ , then

$$
n \geq C \frac { d \log ( 2 / \beta ) + \log ( 1 / \delta ) } { \beta }\tag{A.5}
$$

implies, with probability at least $1 - \delta$

$$
\operatorname* { i n f } _ { C \in \mathcal { C } } \mu _ { n } ( C ) \geq \beta / 2 .
$$

This is a constant-relative-error specialization of relative approximation bounds (Li et al., 2001; Har-Peled and Sharir, 2011).

For a probability range space $( \Omega , \mu , \mathcal { C } )$ , a finite set $S \subseteq \Omega$ is a $\beta \mathrm { - n e t }$ if it intersects every $C \in \mathcal { C }$ with $\mu ( C ) \geq \beta$ . Write $f _ { d } ( \beta )$ for the supremum, over finite probability range spaces of VC dimension at most $d ,$ of the minimum cardinality of $\tt a \beta / \mathrm { - n e t }$ . Theorem 2.1 of Komlos´ et al. (1992) implies

$$
\operatorname* { l i m i n f } _ { \beta \downarrow 0 } \frac { f _ { d } ( \beta ) } { \beta ^ { - 1 } \log ( 1 / \beta ) } \geq d - 2 + \frac { 2 } { d + 2 } \qquad ( d \geq 2 ) .\tag{A.6}
$$

The following finite form is the interface needed below.

Lemma 15 (Finite $\beta { \bf - n e t }$ lower bound) There is a universal $c _ { \mathrm { n e t } } > 0$ such that,for every fixed $d \geq 2 ,$ , there is $\beta _ { d } > 0$ with the following property. For each $0 < \beta \le \beta _ { d } .$ , there is a finite full-support probability range space $( \Omega , \mu , \mathcal { C } )$ ofVC dimension at most d such that every set $S \subseteq \Omega$ with

$$
| S | < c _ { \mathrm { n e t } } { \frac { d } { \beta } } \log { \frac { 1 } { \beta } }
$$

misses some range $C \in \mathcal { C }$ of mass $\mu ( C ) \geq \beta .$

Proof The coefficient on the right of (A.6) satisfies

$$
{ \frac { d - 2 + 2 / ( d + 2 ) } { d } } \geq { \frac { 1 } { 4 } } \qquad ( d \geq 2 ) .
$$

Hence, for every fixed $d ,$ the liminf statement gives the asserted order with a universal numerical constant once $\beta$ is below a threshold $\beta _ { d }$ that may depend on d. Because $f _ { d }$ is a supremum, one chooses a finite range space whose minimum net size is, say, at least half the displayed lower bound; no attainment of the supremum is needed. The constructions in the cited theorem use a finite ground set with the uniform measure, hence have full support.

Our convention uses $\mu ( C ) \geq \beta$ . If the strict convention $\mu ( C ) > \varepsilon$ is used in the cited formulation, apply it with $\varepsilon = 2 \beta$ and retain the ranges of mass greater than $2 \beta$ . They all have mass at least $\beta _ { ; }$ , their VC dimension cannot increase, and, after decreasing $\beta _ { d }$

$$
\frac { d } { 2 \beta } \log \frac { 1 } { 2 \beta } \geq c \frac { d } { \beta } \log \frac { 1 } { \beta } .
$$

This changes only the universal constant and proves the stated weak-threshold form.

The threshold $\beta _ { d }$ is not asserted to be uniform in $d .$ Pach and Tardos (2013) give a geometric VC-two construction with the same logarithmic phenomenon. For VC dimension one, deterministic net size alone does not contain this logarithm, but random sampling does, by the coupon-collector argument in Appendix E.

## Appendix B. Proof of the explicit counterexample Proof

## B.1. The norm and its descent cone

Recall

$$
V _ { m } = \left\{ ( t , y / m ) : t \in \mathbb { R } , y \in \mathbb { R } ^ { m } , \sum _ { j = 1 } ^ { m } y _ { j } = 0 \right\}
$$

and

$$
\mathcal { R } _ { m } ( t , y / m ) = \operatorname* { m a x } \left\{ | t | , \operatorname* { m a x } _ { j } | t - y _ { j } | , \operatorname* { m a x } _ { j } \left| t + y _ { j } \right| \right\} .
$$

This is the maximum of finitely many absolute linear functionals. If it vanishes, then $t = 0$ and $t \pm y _ { i } = 0$ for every $j ,$ , hence $y = 0$ . It is therefore a norm on $V _ { m }$

Let $\theta _ { m } ^ { \star } = ( - 1 , 0 )$ and let $v = ( t , y / m )$ . For $\lambda > 0$

$$
\mathcal { R } _ { m } ( \theta _ { m } ^ { \star } + \lambda v ) = \operatorname* { m a x } \left\{ | - 1 + \lambda t | , \operatorname* { m a x } _ { j } | - 1 + \lambda ( t - y _ { j } ) | , \operatorname* { m a x } _ { j } | - 1 + \lambda ( t + y _ { j } ) | \right\} .
$$

There exists a sufficiently small $\lambda > 0$ for which this maximum is at most one exactly when

$$
t \geq 0 , \qquad t - y _ { j } \geq 0 , \qquad t + y _ { j } \geq 0 \quad ( j \in [ m ] ) .
$$

Thus

$$
\mathcal { D } ( \mathcal { R } _ { m } , \theta _ { m } ^ { \star } ) = C _ { m } = \{ ( t , y / m ) : t \geq 0 , | y _ { j } | \leq t , \sum _ { j } y _ { j } = 0 \} .
$$

## B.2. Uniform small-ball bound

For

$$
r _ { j } = e _ { 0 } + m e _ { j } - \sum _ { \ell = 1 } ^ { m } e _ { \ell }
$$

and $v = ( t , y / m ) \in C _ { m }$ , direct calculation gives

$$
\langle r _ { j } , v \rangle = t + y _ { j } .
$$

Set $a _ { j } = t + y _ { j }$ . Then $0 \leq a _ { j } \leq 2 t$ and m<sup>−</sup> ${ } ^ { - 1 } \sum _ { j } a _ { j } = t$ . If fewer than $m / 3$ of the $a _ { j }$ were at least $t / 2$ , then

$$
\frac { 1 } { m } \sum _ { j } a _ { j } < \frac { 1 } { 3 } ( 2 t ) + \frac { 2 } { 3 } ( t / 2 ) = t ,
$$

a contradiction. Also

$$
\| \boldsymbol { v } \| _ { 2 } ^ { 2 } = t ^ { 2 } + \frac { 1 } { m ^ { 2 } } \sum _ { j } y _ { j } ^ { 2 } \leq t ^ { 2 } + \frac { 1 } { m } t ^ { 2 } \leq 2 t ^ { 2 } .
$$

Since $L \geq 1$ , for $Z _ { m } = \sigma L r _ { J }$

$$
\mathbb { P } \left( | \langle Z _ { m } , v \rangle | \geq \frac { 1 } { 2 \sqrt { 2 } } \| v \| _ { 2 } \right) \geq \frac { 1 } { 3 } .
$$

Homogeneity gives the claimed bound on $A _ { m }$ .

## B.3. Gaussian width

For $v ~ = ~ ( t , y / m ) ~ \in ~ A _ { m }$ , the cone constraints and $\| v \| _ { 2 } = 1$ imply $0 ~ \leq ~ t ~ \leq ~ 1$ and $| y _ { j } | \leq t$ . Let $g = ( g _ { 0 } , g _ { 1 } , \dotsc , g _ { m } )$ be standard Gaussian in the ambient $\mathbb { R } ^ { m + 1 }$ . Its orthogonal projection onto $V _ { m }$ is a standard Gaussian vector on $V _ { m }$ , and inner products with $v \in V _ { m }$ are unchanged. Hence

$$
\operatorname* { s u p } _ { v \in A _ { m } } \left. g , v \right. \leq \vert g _ { 0 } \vert + \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \vert g _ { j } \vert .
$$

Taking expectations yields

$$
\begin{array} { r } { w ( A _ { m } ) \leq 2 \mathbb { E } | g _ { 0 } | = 2 \sqrt { 2 / \pi } . } \end{array}
$$

## B.4. Deterministic kernel and VC dimension

Let $D \subseteq [ m ]$ denote the distinct indices observed among $J _ { 1 } , \ldots , J _ { n }$ . If $n \leq m / 2$ , choose a set $N \subseteq [ m ]$ of cardinality $m / 2$ with $D \subseteq N$ , and define $s _ { j } = - 1$ on $N$ and $s _ { j } = 1$ off $N _ { ☉ }$ Then $s \in \{ - 1 , 1 \} ^ { m }$ and $\textstyle \sum _ { j } s _ { j } = 0$ . Therefore

$$
u _ { s } = \frac { ( 1 , s / m ) } { \sqrt { 1 + 1 / m } } \in A _ { m } .
$$

For each row type,

$$
\langle r _ { j } , u _ { s } \rangle = \frac { 1 + s _ { j } } { \sqrt { 1 + 1 / m } } .
$$

All observed rows have indices in $D \subseteq N$ , hence $X u _ { s } = 0$ . This proves (6) for every realization.

To prove the VC lower bound, fix any $I \subseteq [ m ]$ of cardinality $m / 2$ and choose one support point $z _ { j } = \ell _ { 0 } r _ { j }$ for each $j \in I$ , where $\ell _ { 0 } > 1$ belongs to the support of L. Given any label set $B \subseteq I$ , prescribe $s _ { j } = 1$ for $j \in B$ and $s _ { j } = - 1$ for $j \in I \setminus B$ . Among the remaining $m / 2$ coordinates, choose signs so that the total sum is zero. This is possible because the partial sum on I has the same parity as $m / 2$ and lies between $- m / 2$ and $m / 2$ For the resulting balanced $s ,$

$$
\mathbf { 1 } \{ | \langle z _ { j } , u _ { s } \rangle | \geq 1 \} = \mathbf { 1 } \{ s _ { j } = 1 \} \qquad ( j \in I ) ,
$$

since $2 \ell _ { 0 } / \sqrt { 1 + 1 / m } > 1$ . Thus I is shattered and $\mathrm { V C } ( \mathcal T _ { 1 } ( A _ { m } ) | _ { \mathrm { s u p p } P _ { m } } ) \geq m / 2$

The random vector is centered because of $\sigma$ and has finite moments of every order because L does. To verify the spanning claim, observe that $m ^ { - 1 } \textstyle \sum _ { i } r _ { j } = e _ { 0 }$ and $r _ { j } - r _ { m } =$ $m ( e _ { j } - e _ { m } )$ for $j < m$ . These vectors span the anchor coordinate and the zero-sum subspace, hence all of $V _ { m }$ . Therefore, for every nonzero $v \in V _ { m }$ , some $\langle r _ { j } , v \rangle$ is nonzero. Conditioning on $J = j$ shows $\mathbb { E } e ^ { s | \langle Z _ { m } , v \rangle | } = \infty$ for all $s > 0$ ■

## Appendix C. Exact range-space representation

Proof Let $A = \{ u _ { 1 } , \dotsc , u _ { M } \}$ and $Z$ be as in Theorem 3. The independent sign centers $Z .$ Since $L > 1$ almost surely,

$$
\langle Z , u _ { j } \rangle = \sigma L \mathbf { 1 } \{ Y \in C _ { j } \}
$$

implies the claimed small-ball inequality and (9). To justify the VC equality carefully, map a support point $z \ \mathbf { t o }$ the incidence vector $I ( z ) = ( \mathbf { 1 } \{ | \langle z , u _ { j } \rangle | \ge 1 \} ) _ { j \le M }$ . The displayed identity says that the set of incidence vectors on $\operatorname { s u p p } ( P )$ equals the set of incidence vectors $( \mathbf { 1 } \{ y \in C _ { j } \} ) _ { j \leq M }$ on $\operatorname { s u p p } ( \mu )$ . By the full-support convention in the theorem, $\operatorname { s u p p } ( \mu ) = \Omega$ Repeated latent points, signs, and radial values only duplicate an incidence vector. Choosing one representative from each nonempty incidence fiber transfers shattered sets in both directions, proving equality of the support-restricted VC dimensions.

For each sample and each $j ,$

$$
\operatorname* { m i n } \{ 1 , \langle Z _ { i } , u _ { j } \rangle ^ { 2 } \} = \mathbf { 1 } \{ Y _ { i } \in C _ { j } \} ,
$$

which proves (10). The untruncated sum in direction $u _ { j }$ is positive exactly when some $Y _ { i }$ belongs to $C _ { j }$ . Taking the minimum over j proves the hitting equivalence.

Finally,

$$
\operatorname* { s u p } _ { j \leq M } \langle g , u _ { j } \rangle = \frac { g _ { 0 } + \varepsilon \operatorname* { m a x } _ { j \leq M } g _ { j } } { \sqrt { 1 + \varepsilon ^ { 2 } } } .
$$

The expectation of $g _ { 0 }$ is zero and $\mathbb { E }$ max<sub>j</sub> $g _ { j } \leq \sqrt { 2 \log ( 2 M ) }$ . Hence the claimed width bound follows whenever

$$
\varepsilon \leq { \frac { \eta } { \sqrt { 2 \log ( 2 M ) } } } .
$$

This formula also covers $M = 1$ and shows that the width can be sent to zero without altering the represented range system.

## Appendix D. Descent-cone range universality

Proof The map $T _ { \varepsilon }$ in (12) is injective, so

$$
\mathcal { R } _ { \varepsilon } ( T _ { \varepsilon } a ) = \| a \| _ { \infty }
$$

defines a norm on $V _ { \varepsilon }$ . Its unit ball is $T _ { \varepsilon } [ - 1 , 1 ] ^ { M }$ , a centrally symmetric polytope in $V _ { \varepsilon }$ ; hence $\mathcal { R } _ { \varepsilon }$ is a polyhedral norm. For $h = T _ { \varepsilon } a$

$$
\begin{array} { r } { \mathcal { R } _ { \varepsilon } ( - T _ { \varepsilon } \mathbf { 1 } + t h ) \leq 1 \quad \Longleftrightarrow \quad \lVert - \mathbf { 1 } + t a \rVert _ { \infty } \leq 1 . } \end{array}
$$

The latter holds for some $t > 0$ if and only if $a \geq 0$ coordinatewise. This proves the cone identity in (12). Every nonzero $a \geq 0$ can be written as $( \mathbf { 1 } ^ { \top } a ) q$ with $q \in \Delta _ { M }$ , and normalization gives the parametrization in (12).

Let

$$
\widetilde { Z } = \frac { 2 \sqrt { 1 + \varepsilon ^ { 2 } } } { \beta \varepsilon } \sigma L \sum _ { j = 1 } ^ { M } \mathbf { 1 } \{ Y \in C _ { j } \} e _ { j } \in \mathbb { R } ^ { M + 1 } ,
$$

and let Z be its orthogonal projection onto $V _ { \varepsilon } .$ . Inner products with $u ( q ) \in V _ { \varepsilon }$ are unchanged, and

$$
\begin{array} { r l r } {  { | \langle Z , u ( q ) \rangle | = \frac { 2 \sqrt { 1 + \varepsilon ^ { 2 } } } { \beta \sqrt { 1 + \varepsilon ^ { 2 } \| q \| _ { 2 } ^ { 2 } } } L f _ { q } ( Y ) } } \\ & { } & { \geq \displaystyle \frac { 2 } { \beta } f _ { q } ( Y ) . } \end{array}
$$

Because $0 \leq f _ { q } \leq 1$ and $\mathbb { E } f _ { q } \ge \beta , \operatorname { i f } p _ { q } = \mathbb { P } ( f _ { q } \ge \beta / 2 )$ , then

$$
\begin{array} { r } { \beta \le \mathbb { E } f _ { q } \le p _ { q } + ( 1 - p _ { q } ) \beta / 2 , } \end{array}
$$

and therefore $p _ { q } \ge \beta / ( 2 - \beta )$ . This proves the uniform small-ball bound.

For the width, let $g = ( g _ { 0 } , g ^ { \prime } )$ be standard Gaussian in $\mathbb { R } ^ { M + 1 }$ . Put $d _ { q } = \sqrt { 1 + \varepsilon ^ { 2 } \| q \| _ { 2 } ^ { 2 } }$ $\mathrm { I f } \ g _ { 0 } \ge 0$ , then $g _ { 0 } / d _ { q } \le g _ { 0 } . \mathrm { ~ H ~ } g _ { 0 } < 0$ , then

$$
\frac { g _ { 0 } } { d _ { q } } = g _ { 0 } + | g _ { 0 } | \left( 1 - \frac { 1 } { d _ { q } } \right) \leq g _ { 0 } + \frac { \varepsilon ^ { 2 } } { 2 } | g _ { 0 } | .
$$

Also

$$
\frac { \varepsilon \langle g ^ { \prime } , q \rangle } { d _ { q } } \leq \varepsilon \operatorname* { m a x } \{ 0 , g _ { 1 } , \dots , g _ { M } \} .
$$

Consequently,

$$
w ( A _ { \varepsilon } ) \leq \varepsilon \mathbb { E } \operatorname* { m a x } \{ 0 , g _ { 1 } , \dots , g _ { M } \} + { \frac { \varepsilon ^ { 2 } } { 2 } } \mathbb { E } | g _ { 0 } | \leq \varepsilon { \sqrt { 2 \log ( 2 M ) } } + { \frac { \varepsilon ^ { 2 } } { \sqrt { 2 \pi } } } .
$$

Choosing ε small gives any prescribed $\eta .$ . Finally, if the latent sample misses $C _ { j }$ , then $f _ { e _ { j } } ( Y _ { i } ) = 0$ for all i, hence $X u ( e _ { j } ) = 0$

## Appendix E. Sharp threshold-complexity law

Proof The definition of $\mathsf { N } _ { \mathrm { R E } }$ asks for a threshold valid for every larger sample size. If the universal conclusion fails at some $N _ { 0 }$ , then no $n \leq N _ { 0 }$ satisfies the definition, and hence $\mathsf { N } _ { \mathrm { R E } } \geq N _ { 0 } + 1$ . Each upper bound below applies marginally at every larger sample size.

Upper bound. Apply the relative VC estimate (A.5) to the support-restricted threshold class $\mathcal { T } _ { 1 } ( A )$ . For every $N \geq C [ d \log ( 2 / \beta ) + \log ( 1 / \delta ) ] / \beta$ , with probability at least $1 - \delta$ $\widehat { q } _ { N , 1 } ( A ) \geq \beta / 2$ . The deterministic inequality (8) then gives $\Lambda _ { N } ( X ; A ) \ge \beta / 2$

A universal $d / \beta$ lower bound. Assume first that $0 < \beta \leq 1 / 4$ . Let $N _ { 0 } = \lfloor d / \beta \rfloor$ and put the uniform measure on $[ N _ { 0 } ]$ . Take as ranges all d-element subsets. Each range has mass $d / N _ { 0 } \ge \beta$ , and the class has VC dimension exactly d: it shatters a fixed d-set, using points outside that set to complete each trace to cardinality $d ,$ and no $( d + 1 )$ -set can be shattered. Any hitting set must have at least $N _ { 0 } - d + 1$ points, since the complement of a smaller set contains a d-range. Therefore every sample with fewer than $N _ { 0 } - d + 1 \ge c d / \beta$ distinct points misses a range. The exact representation theorem converts this deterministic miss into $\Lambda _ { n } = 0$ while preserving threshold VC dimension. This proves $\mathsf { N } _ { \mathrm { R E } } \geq c d / \beta$

For reference, the preceding $d / \beta$ construction has no hidden endpoint loss: with $n _ { 0 } =$ $N _ { 0 } - d ,$ , every sample of size $n _ { 0 }$ misses a range, so

$$
{ \mathsf { N } } _ { \mathrm { R E } } ( d , \beta , \delta ) \geq n _ { 0 } + 1 = N _ { 0 } - d + 1 \geq \frac { 3 } { 4 } \frac { d } { \beta } .
$$

Here $\lfloor d / \beta \rfloor + 1 \ge d / \beta$ and $\beta \leq 1 / 4$ give the last inequality.

The coupon-collector term, including $d = 1$ . For $0 < \beta \leq 1 / 1 6$ , let $M = \lfloor 1 / ( 2 \beta ) \rfloor$ points each have mass $\beta ,$ , put the remaining mass on one extra point, and let the ranges be the M singletons. This class has VC dimension one. Let $U$ be the number of singleton ranges missed by n independent observations. Then

$$
\mathbb { E } U = M ( 1 - \beta ) ^ { n } , \quad \quad \mathrm { V a r } ( U ) \leq \mathbb { E } U ,\tag{E.1}
$$

because two missing indicators have covariance $( 1 - 2 \beta ) ^ { n } - ( 1 - \beta ) ^ { 2 n } \leq 0$ . If $n \ \leq$ $( 4 \beta ) ^ { - 1 }$ log $M ,$ , then $\bar { ( 1 - \beta ) } ^ { n } \ge e ^ { - 2 \beta n } \ge M ^ { - 1 / 2 }$ , so $\mathbb { E } U \geq \sqrt { M }$ . Chebyshev’s inequality yields $P \{ U = 0 \} \le \mathrm { V a r } ( U ) / ( \mathbb { E } U ) ^ { 2 } \le M ^ { - 1 / 2 }$ . Here $M \geq 8 .$ , so the probability of hitting every singleton is strictly below $3 / 4$ . Moreover, $M \geq 1 / ( 4 \beta )$ and hence log $M \geq c \log ( 2 / \beta )$ on $0 < \beta \leq 1 / 1 6$ . The exact representation therefore gives the claimed order on this interval. For $1 / 1 6 < \beta \leq 1 / 4$ , a single range of mass $\beta$ is missed at sample size one with probability $1 - \beta \geq 3 / 4$ , while $\beta ^ { - 1 } \log ( 2 / \beta ) \le 1 6 \log 3 2$ . Decreasing the same universal constant covers this remaining interval and yields

$$
\mathsf { N } _ { \mathrm { R E } } ( d , \beta , 1 / 4 ) \ge c { \frac { \log ( 2 / \beta ) } { \beta } } \qquad ( d \ge 1 ) .\tag{E.2}
$$

Indeed, on the first interval one may take $n _ { 0 } = \lfloor ( 4 \beta ) ^ { - 1 } \log M \rfloor$ ; the probability of success at $n _ { 0 }$ is below $3 / 4$ , whence $\mathsf { N } _ { \mathrm { R E } } \geq n _ { 0 } + 1 > ( 4 \beta ) ^ { - 1 }$ log M. This makes the integer reduction explicit. This is why the random-sample problem retains a logarithm for VC dimension one, even though a smallest deterministic net need not. Since $\mathsf { N } _ { \mathrm { R E } }$ is nonincreasing as $\delta$ increases, (E.2) also holds for every $\delta \leq 1 / 4$

The confidence term. An explicit range space is $\Omega = \{ a , b \}$ with $\mu ( a ) = \beta$ and $\mathcal { C } =$ $\{ \{ a \} \}$ ; it has full support and VC dimension zero. Take one range of mass exactly $\beta .$ . It is missed with probability $( 1 - \beta ) ^ { n } \geq e ^ { - 2 \beta n }$ for $\beta \leq 1 / 4$ . Consequently, every integer $n < ( 2 \beta ) ^ { - 1 } \log ( 1 / \delta )$ has failure probability strictly larger than $\delta .$ Accounting for the integer endpoint only changes a universal constant, and therefore

$$
\mathsf { N } _ { \mathrm { R E } } ( d , \beta , \delta ) \geq c \frac { 1 } { \beta } \log \frac { 1 } { \delta } .\tag{E.3}
$$

More precisely, set $T = ( 2 \beta ) ^ { - 1 } \log ( 1 / \delta )$ and $n _ { 0 } = \lceil T \rceil - 1 < T$ . Failure at $n _ { 0 }$ gives $\mathsf { N } _ { \mathrm { R E } } \geq n _ { 0 } + 1 = \lceil T \rceil \geq T$ . The maximum of the three lower bounds above is at least one third of their sum. This proves the left side of (13).

Sharp fixed-d low-mass regime. Fix $d \geq 2$ and take the finite full-support range space supplied by Lemma 15; decrease $\beta _ { d }$ if necessary so that $\beta _ { d } \leq 1 / 4$ . Removing any ranges of mass below $\beta$ cannot increase VC dimension. By the lemma, every subset of Ω with fewer than $c _ { \mathrm { n e t } } ( d / \beta ) \log ( 1 / \beta )$ points misses a retained range of mass at least $\beta .$ . The distinct support of a multiset sample has cardinality at most the sample size, so every sample path below this threshold misses such a range. Applying Theorem 3 gives $\Lambda _ { n } = 0$ on every path while preserving the support-restricted VC dimension. For $d = 1$ , (E.2) gives the same order. Finally, $\log ( 1 / \beta ) \asymp \log ( 2 / \beta )$ on this range. Combining this term with (E.3) proves (14) for every fixed d and all $0 < \beta \le \beta _ { d }$ . For the strict inequality in Lemma 15, put $T _ { \beta } = c _ { \mathrm { n e t } } ( d / \beta ) \log ( 1 / \beta )$ and $n _ { 0 } = \lceil T _ { \beta } \rceil - 1 < T _ { \beta }$ . Every $\mathrm { s i z e } { \mathrm { - } } n _ { 0 }$ sample then fails pathwise, so $\mathsf { N } _ { \mathrm { R E } } \geq n _ { 0 } + 1 = \lceil T _ { \beta } \rceil \geq T _ { \beta }$

## Appendix F. The anchored isotropic frame

Proof We use the probabilistic method. The proof has four steps. We first control the empirical covariance, then establish a uniform slab lower bound, and next condition every row submatrix of size $O ( p / \log p )$ . A final whitening step makes the frame exactly tight while preserving the common anchor and the three preceding estimates. Let $M = p ^ { 4 }$ and let

$$
a _ { j } = ( 1 , \xi _ { j } ) \in \mathbb { R } \times \mathbb { R } ^ { p - 1 } , \qquad j \in [ M ] ,
$$

where the $\xi _ { j }$ are independent vectors with independent Rademacher coordinates. Define

$$
\Sigma = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } a _ { j } a _ { j } ^ { \top } .
$$

We establish three simultaneous events of positive probability.

## F.1. Covariance control

The vectors $a _ { j }$ are isotropic in the sense that $\mathbb { E } a _ { j } a _ { j } ^ { \top } = I _ { p }$ and satisfy $\| a _ { j } \| _ { 2 } ^ { 2 } = p$ deterministically. Matrix Bernstein, applied to $a _ { j } a _ { j } ^ { \top } - I _ { p } ,$ , yields

$$
\begin{array} { r } { \mathbb { P } \left( \| \Sigma - I _ { p } \| _ { \mathrm { o p } } > 1 / 4 \right) \le 2 p \exp ( - c M / p ) . } \end{array}\tag{F.1}
$$

For completeness, if $\begin{array} { r } { Y _ { j } = a _ { j } a _ { j } ^ { \top } - I _ { p } } \end{array}$ , then $\| Y _ { j } \| _ { \mathrm { o p } } \leq p + 1 \leq 2 p$ and

$$
\mathbb { E } Y _ { j } ^ { 2 } = ( p - 1 ) I _ { p } .
$$

The self-adjoint matrix Bernstein inequality (Tropp, 2015b, Theorem 6.1.1), applied at deviation level $M / 4$ , has variance proxy $M ( p - 1 )$ and range bound $2 p .$ . Substitution gives an exponent at most $- c M / p ,$ proving the displayed estimate. In particular, on the complementary event

$$
\frac { 3 } { 4 } I _ { p } \preceq \Sigma \preceq \frac { 5 } { 4 } I _ { p } .\tag{F.2}
$$

## F.2. A uniform empirical slab bound

For fixed $v = ( v _ { 0 } , v ^ { \prime } ) \in \mathbb { S } ^ { p - 1 }$ and $a = \left( 1 , \xi \right)$ , put $S = \langle a , v \rangle = v _ { 0 } + \langle \xi , v ^ { \prime } \rangle$ . Then

$$
\mathbb { E } S ^ { 2 } = 1
$$

and a direct fourth-moment expansion gives

$$
\mathbb { E } S ^ { 4 } = v _ { 0 } ^ { 4 } + 6 v _ { 0 } ^ { 2 } \| v ^ { \prime } \| _ { 2 } ^ { 2 } + 3 \| v ^ { \prime } \| _ { 2 } ^ { 4 } - 2 \sum _ { \ell = 1 } ^ { p - 1 } v _ { \ell } ^ { 4 } \leq 3 .
$$

Paley–Zygmund applied to $S ^ { 2 }$ gives

$$
\mathbb { P } \big ( | \langle a , v \rangle | \geq 1 / \sqrt { 2 } \big ) \geq 1 / 1 2 .\tag{F.3}
$$

The class of two-sided homogeneous slabs

$$
\{ x : | \langle x , v \rangle | \geq \| v \| _ { 2 } / \sqrt { 2 } \} , \qquad v \neq 0 ,
$$

has VC dimension at most $C _ { 0 } p$ . Indeed, it is a subclass of unions of two affine halfspaces. On N points the halfspace growth function is at most $( e N / ( p + 1 ) ) ^ { p + 1 }$ by Sauer’s lemma, so the growth function of two-fold unions is at most its square. For a sufficiently large numerical $C _ { 0 } ,$ , this is smaller than $2 ^ { N }$ whenever $N > C _ { 0 } p ,$ proving the claim. We use the following explicit form of the VC uniform law (Boucheron et al., 2013). For a class of sets with VC dimension v and $M \geq v$ , symmetrization, Hoeffding’s inequality, and Sauer’s lemma give

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { C } | \mu _ { M } ( C ) - \mu ( C ) | > r \right\} \le 8 \left( \frac { 2 e M } { v } \right) ^ { v } e ^ { - M r ^ { 2 } / 3 2 } .
$$

Take $v = C _ { 0 } p$ and $r = 1 / 2 4$ . Since $M = p ^ { 4 }$ , the right side is at most $e ^ { - c M }$ for all sufficiently large $p .$ . Combining this event with (F.3) gives

$$
\operatorname* { i n f } _ { v \neq 0 } \frac { 1 } { M } \left| \left\{ j : | \langle a _ { j } , v \rangle | \geq \| v \| _ { 2 } / \sqrt { 2 } \right\} \right| \geq 1 / 2 4 .\tag{F.4}
$$

## F.3. All small row submatrices are well conditioned

Fix $D \subseteq [ M ]$ with $s = | D |$ and write $\Xi _ { D } \in \mathbb { R } ^ { ( p - 1 ) \times s }$ for the matrix whose columns are $\xi _ { j }$ $j \in D$ . For fixed $x \in \mathbb { S } ^ { s - 1 }$ , the coordinates $\begin{array} { r } { Y _ { \ell } = \sum _ { i \in D } x _ { j } \xi _ { j \ell } \operatorname* { o f } \Xi _ { D } x } \end{array}$ are independent, meanzero, and variance-one. Hoeffding’s lemma gives $\| \bar { Y _ { \ell } } \| _ { \psi _ { 2 } } \leq C \| x \| _ { 2 } = C , \operatorname { s o } \| Y _ { \ell } ^ { 2 } - 1 \| _ { \psi _ { 1 } } \leq$ $C ^ { \prime }$ . Bernstein’s inequality for these centered squared values gives

$$
\begin{array} { r } { \mathbb { P } \left( \left| \| \Xi _ { D } x \| _ { 2 } ^ { 2 } - ( p - 1 ) \right| > ( p - 1 ) / 2 \right) \leq 2 e ^ { - c p } . } \end{array}\tag{F.5}
$$

A $1 / 8 \cdot$ -net $\mathcal { N }$ of $\mathbb { S } ^ { s - 1 }$ has cardinality at most $1 7 ^ { s }$ (Vershynin, 2018). To spell out the net step, put $H = \Xi _ { D } ^ { \top } \Xi _ { D } - ( p - 1 ) I _ { s }$ . For every symmetric H and every ε-net with $\varepsilon < 1 / 2$

$$
\| H \| _ { \mathrm { o p } } \leq { \frac { 1 } { 1 - 2 \varepsilon } } \operatorname* { m a x } _ { x \in { \mathcal { N } } } | x ^ { \top } H x | .
$$

This follows by choosing, for a maximizing unit vector $y ,$ a net point x with $\| x - y \| \leq \varepsilon$ and bounding $| y ^ { \top } H y - x ^ { \top } H x | \leq 2 \varepsilon \| H \| _ { \mathrm { o p } }$ . Apply (F.5) to every $x \in \mathcal N$ . On the resulting event, the last display with $\varepsilon = 1 / 8$ puts all eigenvalues of $\Xi _ { D } ^ { \top } \Xi _ { D }$ between absolute multiples of $p .$ Hence

$$
\begin{array} { r } { \mathbb { P } \left( c p I _ { s } \neq \Xi _ { D } ^ { \top } \Xi _ { D } \ \mathrm { o r } \ \Xi _ { D } ^ { \top } \Xi _ { D } \neq C p I _ { s } \right) \leq 2 \exp ( - c ^ { \prime } p + C s ) . } \end{array}\tag{F.6}
$$

There are at most $( e M / s ) ^ { s }$ subsets of cardinality s. Thus the total failure probability for all sets of a fixed size s is at most

$$
2 \exp \{ - c ^ { \prime } p + C s + s \log ( e M / s ) \} .
$$

For $M = p ^ { 4 }$ and $s \leq s _ { \mathrm { m a x } } = \lfloor c _ { 0 } p / \log p \rfloor$ , the two positive terms are at most $6 c _ { 0 } p$ for large $p .$ Choose $c _ { 0 } < c ^ { \prime } / 1 2$ . The failure probability for each size $s \leq s _ { \mathrm { m a x } }$ is then at most $2 e ^ { - c ^ { \prime } p / 2 }$ and summing over the at most $s _ { \mathrm { m a x } } \leq p$ possible sizes gives

$$
2 s _ { \mathrm { m a x } } e ^ { - c ^ { \prime } p / 2 } \leq e ^ { - c ^ { \prime \prime } p }
$$

for all sufficiently large $p .$ Thus, with probability at least $1 - e ^ { - c ^ { \prime \prime } p }$

$$
c p I _ { | D | } \preceq \Xi _ { D } ^ { \top } \Xi _ { D } \preceq C p I _ { | D | } \qquad \mathrm { f o r e v e r y } | D | \leq c _ { 0 } p / \log p ,\tag{F.7}
$$

Let $A _ { D }$ be the matrix with rows $a _ { j } ^ { \top } , j \in D$ . For $\boldsymbol { x } \in \mathbb { R } ^ { D }$

$$
\| A _ { D } ^ { \top } x \| _ { 2 } ^ { 2 } = ( \mathbf { 1 } ^ { \top } x ) ^ { 2 } + \| \boldsymbol { \Xi } _ { D } x \| _ { 2 } ^ { 2 } .
$$

The lower inequality in (F.7) is unchanged, while $( \mathbf { 1 } ^ { \top } \boldsymbol { x } ) ^ { 2 } \leq s \| \boldsymbol { x } \| _ { 2 } ^ { 2 } \leq c _ { 0 } p \| \boldsymbol { x } \| _ { 2 } ^ { 2 } / \log p$ . Thus (F.7) implies

$$
c p I _ { | D | } \preceq A _ { D } A _ { D } ^ { \top } \preceq C p I _ { | D | }\tag{F.8}
$$

for all such $D$

## F.4. Whitening

The covariance, slab, and restricted-Gram failure probabilities are at most 2pe<sup>−</sup> $^ { - c p ^ { 3 } } , e ^ { - c p ^ { 4 } }$ and $e ^ { - c ^ { \prime \prime } p }$ , respectively. Their sum is below one for all sufficiently large $p .$ Fix a realization on the three events and define

$$
b _ { j } = \Sigma ^ { - 1 / 2 } a _ { j } , \qquad u _ { \star } = \Sigma ^ { 1 / 2 } e _ { 0 } .
$$

Then

$$
\frac { 1 } { M } \sum _ { j } b _ { j } b _ { j } ^ { \top } = I _ { p } .
$$

Also

$$
\| u _ { \star } \| _ { 2 } ^ { 2 } = e _ { 0 } ^ { \top } \Sigma e _ { 0 } = 1 , \qquad \langle b _ { j } , u _ { \star } \rangle = \langle a _ { j } , e _ { 0 } \rangle = 1 .
$$

For a unit vector v, set $q = \Sigma ^ { - 1 / 2 } v$ . By (F.2), $\| \boldsymbol { q } \| _ { 2 } \ge \sqrt { 4 / 5 }$ . Applying (F.4) to q gives at least $M / 2 4$ indices with

$$
| \langle b _ { j } , v \rangle | = | \langle a _ { j } , q \rangle | \geq \| q \| _ { 2 } / \sqrt { 2 } \geq \sqrt { 2 / 5 } .
$$

Thus one may take $a _ { 0 } = 1 / 2$ and $b _ { 0 } = 1 / 2 4$ . Finally,

$$
B _ { D } B _ { D } ^ { \top } = A _ { D } \Sigma ^ { - 1 } A _ { D } ^ { \top } ,
$$

and (F.2), (F.8) give (16).

## Appendix G. The isotropic separation

Proof Fix the frame from Lemma 6, choose the theorem constant $c \leq c _ { 0 } ,$ , and assume $1 \leq k \leq c p / \log p .$ . Throughout this proof, $D \subseteq [ M ]$ ranges only over $| D | \leq k ,$ , and $B _ { D }$ denotes the corresponding row submatrix. If $D \neq \emptyset$ , (16) makes $G _ { D } = B _ { D } B _ { D } ^ { \top }$ positive definite, and we set

$$
\begin{array} { r } { P _ { D } = B _ { D } ^ { \top } G _ { D } ^ { - 1 } B _ { D } . } \end{array}
$$

For $D = \emptyset$ , set $P _ { D } = 0$ and $s _ { D } = 0$ . In either case $P _ { D }$ is the orthogonal projector onto the row span of $B _ { D }$ . We first quantify the normalized residual $u _ { D }$ , then control its generated cone, and finally verify the distributional claims. Since $B _ { D } u _ { \star } = { \bf 1 }$ for nonempty $D .$

$$
\| P _ { D } u _ { \star } \| _ { 2 } ^ { 2 } = \mathbf { 1 } ^ { \top } G _ { D } ^ { - 1 } \mathbf { 1 } .\tag{G.1}
$$

The Gram bounds imply, for $d = | D | \geq 1$ 9

$$
\frac { d } { C _ { 1 } p } \leq s _ { D } ^ { 2 } : = \| P _ { D } u _ { \star } \| _ { 2 } ^ { 2 } = \mathbf { 1 } ^ { \top } G _ { D } ^ { - 1 } \mathbf { 1 } \leq \frac { d } { c _ { 1 } p } .\tag{G.2}
$$

For $D = \emptyset$ , the same bounds hold with $d = s _ { D } = 0$ . For all sufficiently large $p ,$ the upper bound is at most $c / ( c _ { 1 } \log p ) \leq 1 / 2$ . Thus $( I - P _ { D } ) u _ { \star } \neq 0$ and every $u _ { D }$ used in the construction is well defined.

Since $( I - P _ { D } ) u _ { \star }$ is orthogonal to $P _ { D } u _ { \star }$

$$
\left. u _ { D } , u _ { \star } \right. = \sqrt { 1 - s _ { D } ^ { 2 } }
$$

and

$$
\| u _ { D } - u _ { \star } \| _ { 2 } ^ { 2 } = 2 \left( 1 - \sqrt { 1 - s _ { D } ^ { 2 } } \right) = \frac { 2 s _ { D } ^ { 2 } } { 1 + \sqrt { 1 - s _ { D } ^ { 2 } } } .\tag{G.3}
$$

In particular, $s _ { D } ^ { 2 } \leq \| u _ { D } - u _ { \star } \| _ { 2 } ^ { 2 } \leq 2 s _ { D } ^ { 2 }$ when $s _ { D } ^ { 2 } \leq 1 / 2$ , so (G.2) gives $\| u _ { D } - u _ { \star } \| _ { 2 } ^ { 2 } \asymp d / p$ with explicit absolute constants. Also $B _ { D } u _ { D } = 0$

## G.1. Geometry and width of the cone

Let

$$
\mathcal { U } _ { p , k } = \{ \boldsymbol { u } _ { D } : | D | \leq k \} .
$$

Its cardinality satisfies

$$
| \mathcal { U } _ { p , k } | \leq \sum _ { d = 0 } ^ { k } \binom { M } { d } \leq 2 \left( \frac { e M } { k } \right) ^ { k }\tag{G.4}
$$

for $1 \le k \le M / 2$ . By (G.3), all generators lie in a cap of radius

$$
r \leq C \sqrt { k / p }
$$

about $u _ { \star }$ . Since $k \leq c p / \log p .$ , choosing c small and $p$ large ensures $r < 1 \AA$ , as required by Lemma 13. Applying that lemma with $M = p ^ { 4 }$ gives

$$
\begin{array} { l } { \displaystyle { w ( A _ { p , k } ) \leq C \sqrt { \frac { k } { p } } \sqrt { k \log ( e M / k ) } + C \frac { k } { \sqrt { p } } } } \\ { \displaystyle { \qquad \leq C \left[ k \sqrt { \frac { \log p } { p } } + \frac { k } { \sqrt { p } } \right] . } } \end{array}
$$

The cone is full dimensional. Indeed, $u _ { \emptyset } = u _ { \star }$ , and for $D = \{ j \}$ the vector $u _ { \{ j \} }$ is a nonzero normalization of

$$
u _ { \star } - \frac { 1 } { \| b _ { j } \| _ { 2 } ^ { 2 } } b _ { j } .
$$

Thus every $b _ { j }$ belongs to the linear span of $u _ { \star }$ <sub>⋆</sub> and $u _ { \{ j \} }$ . The tight-frame identity implies that the $b _ { j }$ span $\mathbb { R } ^ { p }$ . The cone is pointed because

$$
\operatorname* { i n f } _ { | D | \leq k } \langle u _ { D } , u _ { \star } \rangle \geq 1 / \sqrt { 2 } > 0 .
$$

It is polyhedral because it has finitely many generators. Lemma 14 therefore realizes it as the descent cone of a polyhedral norm.

## G.2. Heavy-tailed design

Let J be uniform on [M], let $\sigma$ be a Rademacher sign independent of $( J , L )$ , and set

$$
\rho = L / \sqrt { \mathbb { E } L ^ { 2 } } , \qquad Z _ { \mathsf { H } } = \sigma \rho b _ { J } .
$$

Then

$$
\mathbb { E } Z _ { \mathsf { H } } = 0 , \qquad \mathbb { E } Z _ { \mathsf { H } } Z _ { \mathsf { H } } ^ { \top } = \mathbb { E } \rho ^ { 2 } \frac { 1 } { M } \sum _ { j } b _ { j } b _ { j } ^ { \top } = I _ { p } .
$$

Since $\rho \ge ( \mathbb { E } L ^ { 2 } ) ^ { - 1 / 2 }$ and the slab property in (15) holds,

$$
\operatorname* { i n f } _ { v \in \mathbb { S } ^ { p - 1 } } \mathbb { P } \left( | \langle Z _ { \mathsf { H } } , v \rangle | \geq a _ { 0 } / \sqrt { \mathbb { E } L ^ { 2 } } \right) \geq b _ { 0 } .
$$

For the Gaussian law, $\langle G , v \rangle \sim \mathsf { N } ( 0 , 1 )$ for every unit v. Thus the explicit common choice

$$
\alpha _ { 0 } = \operatorname* { m i n } \left\{ \frac { a _ { 0 } } { \sqrt { \mathbb { E } L ^ { 2 } } } , \frac { 1 } { 2 } \right\} , \qquad \beta _ { 0 } = \operatorname* { m i n } \left\{ b _ { 0 } , \mathbb { P } ( | G _ { 1 } | \geq \alpha _ { 0 } ) \right\}
$$

gives (18) for both designs.

The law of $Z _ { \mathsf { H } }$ has finite moments of every order. If $v \neq 0$ , the tight-frame identity gives

$$
\frac { 1 } { M } \sum _ { j } \langle b _ { j } , v \rangle ^ { 2 } = \| v \| _ { 2 } ^ { 2 } > 0 ,
$$

so some row type has nonzero inner product with v. Conditioning on that type shows

$$
\mathbb { E } e ^ { s | \langle Z _ { \mathsf { H } } , v \rangle | } = \infty \qquad ( s > 0 ) .
$$

Given any sample of size $n \leq k$ , let D be the set of distinct values among $J _ { 1 } , \ldots , J _ { n }$ Then $u _ { D } \in A _ { p , k }$ and

$$
\langle Z _ { \mathsf { H } , i } , u _ { D } \rangle = { \sigma } _ { i } { \rho } _ { i } \langle b _ { J _ { i } } , u _ { D } \rangle = 0
$$

for every i. This proves deterministic failure. The Gaussian claim follows directly from (A.1).

## G.3. Constant-width Gaussian specialization

Fix a sufficiently small absolute $a > 0$ and take

$$
k = \left\lfloor a \sqrt { \frac { p } { \log p } } \right\rfloor .
$$

For all sufficiently large $p ,$ this k is admissible in Theorem 7, and (17) gives

$$
w ( A _ { p , k } ) \leq C \left( a + \frac { a } { \sqrt { \log p } } \right) \leq C _ { \star }
$$

for an absolute $C _ { \star }$ . Let $t = \sqrt { 2 \log ( 1 / \delta ) }$ . If

$$
n \geq \operatorname* { m a x } \{ 2 , 1 6 C _ { \star } ^ { 2 } , 3 2 \log ( 1 / \delta ) \} ,
$$

then $\sqrt { n - 1 } \geq \sqrt { n / 2 } , C _ { \star } \leq \sqrt { n } / 4$ , and $t \leq \sqrt { n } / 4$ . Therefore (19) yields

$$
{ \frac { 1 } { \sqrt { n } } } \operatorname* { i n f } _ { u \in A _ { p , k } } \| X _ { \mathsf { G } } u \| _ { 2 } \geq { \frac { 1 } { \sqrt { 2 } } } - { \frac { 1 } { 2 } } = : c _ { \star } > 0
$$

with probability at least $1 - \delta$ . Hence $\Lambda _ { n } ( X _ { \mathsf { G } } ; A _ { p , k } ) \ge c _ { \star } ^ { 2 }$ . Enlarging one universal constant converts the displayed condition on n to $n \ge \bar { C } _ { G } [ 1 + \log ( 1 / \delta ) ]$ . The heavy-tailed failure holds on every path for $n \leq k$ , and, after reducing a by a factor of two to absorb the floor, $k \geq c _ { H } \sqrt { p / \log p }$ . This proves Corollary 8.

## G.4. Radius and affine dimension

The radius upper bound follows from (A.3):

$$
r ( A _ { p , k } ) \leq C \sqrt { k / p } .
$$

Choose a set D of cardinality k. By (G.3),

$$
\| u _ { D } - u _ { \star } \| _ { 2 } \geq c \sqrt { k / p } ,
$$

so every enclosing ball has radius at least half this distance. Therefore

$$
r ( A _ { p , k } ) ^ { 2 } \asymp k / p .
$$

The cone $C _ { p , k }$ is full dimensional. Its intersection with the sphere therefore contains a nonempty open spherical patch. If this patch lay in a proper affine hyperplane $\{ x : \langle a , x \rangle =$ $b \}$ , then the affine function $x \mapsto \langle a , x \rangle - b$ would vanish on an open subset of the sphere. Differentiating along all tangent directions on that subset forces a to be parallel to every point in the patch, which is impossible unless $a = 0$ and then $b = 0$ . Hence no proper affine hyperplane contains the patch, and $q ( A _ { p , k } ) = p .$

## Appendix H. Smooth-density robustness

Proof The convolution in (20) is centered and isotropic. Write $c _ { \tau } = \sqrt { 1 + \tau ^ { 2 } }$ and let $\phi _ { \tau }$ be the density of $\tau H$ . The smoothed row has density

$$
f _ { \tau } ( x ) = c _ { \tau } ^ { p } \mathbb { E } \phi _ { \tau } ( c _ { \tau } x - Z _ { \mathsf { H } } ) ,
$$

which is strictly positive for every $x \in \mathbb { R } ^ { p }$ . Every derivative of a Gaussian density is a polynomial times that density and is bounded on $\mathbb { R } ^ { p }$ . Dominated differentiation under the expectation therefore gives $f _ { \tau } \in C ^ { \infty } (  { \mathbb { R } } ^ { p } )$ . Polynomial moments remain finite.

We also verify the stronger directional heavy-tail claim. Fix $v \neq 0$ . Tightness of the frame provides a j with $c _ { j } = \langle b _ { j } , v \rangle \neq 0$ . Conditional on $J = j$ , and on the independent event $| \tau \langle H , v \rangle | \leq 1$ , which has positive probability,

$$
| \sigma \rho c _ { j } + \tau \langle H , v \rangle | \geq | c _ { j } | \rho - 1 .
$$

The event is independent of $\rho .$ Hence, after the harmless factor $1 / \sqrt { 1 + \tau ^ { 2 } }$ , every positive exponential moment of $| \langle Z _ { \mathsf { H } , \tau } , v \rangle |$ is infinite.

Let $\alpha _ { 0 } , \beta _ { 0 }$ be global small-ball constants for $Z _ { \mathsf { H } }$ . Choose $\tau _ { 0 } > 0$ such that

$$
\begin{array} { r } { \mathbb { P } ( | \tau H _ { 1 } | \le \alpha _ { 0 } / 2 ) \ge 1 / 2 \qquad ( 0 < \tau \le \tau _ { 0 } ) . } \end{array}
$$

For a unit v, the heavy and Gaussian components are independent. On the event

$$
| \langle Z _ { \mathsf { H } } , v \rangle | \geq \alpha _ { 0 } , \qquad | \tau \langle H , v \rangle | \leq \alpha _ { 0 } / 2 ,
$$

one has

$$
\begin{array} { r } { \left| \langle Z _ { \mathsf { H } , \tau } , v \rangle \right| \geq \frac { \alpha _ { 0 } } { 2 \sqrt { 1 + \tau _ { 0 } ^ { 2 } } } . } \end{array}
$$

Thus one may take $\bar { \alpha } = \alpha _ { 0 } / ( 2 \sqrt { 1 + \tau _ { 0 } ^ { 2 } } )$ and $\bar { \beta } = \beta _ { 0 } / 2$

For each fixed sample size $n \leq k$ , let D be its latent row-type set. The direction $u _ { D }$ depends only on the $J _ { i }$ and is independent of the Gaussian perturbations. Since the heavy component is annihilated,

$$
\Lambda _ { n } ( X _ { \mathsf { H } , \tau } ; A _ { p , k } ) \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle Z _ { \mathsf { H } , \tau , i } , u _ { D } \rangle ^ { 2 } = \frac { \tau ^ { 2 } } { 1 + \tau ^ { 2 } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \langle H _ { i } , u _ { D } \rangle ^ { 2 } .
$$

Conditionally on $\left( J _ { 1 } , \ldots , J _ { n } \right)$ , the final sum has the $\chi _ { n } ^ { 2 }$ law. Thus $\chi _ { n } ^ { 2 } \ \leq \ 2 n$ implies $\Lambda _ { n } \leq 2 \tau ^ { 2 }$ . For completeness, Markov’s inequality and $\mathbb { E } e ^ { \bar { \chi } _ { n } ^ { 2 } / 4 } = 2 ^ { n / 2 } \mathrm { ~ g i v e ~ }$

$$
\mathbb { P } ( \chi _ { n } ^ { 2 } > 2 n ) \le e ^ { - n / 2 } 2 ^ { n / 2 } = \exp \left[ - \frac { 1 - \log 2 } { 2 } n \right] .
$$

This proves (21) with $c _ { s } = ( 1 - \log 2 ) / 2$

## Appendix I. The dimension–radius bound

Proof Choose a countable dense subset $A _ { 0 } \subseteq A$ . For every matrix X and every $h \in \mathbb { R } ^ { p }$ continuity gives

$$
\operatorname* { i n f } _ { u \in A _ { 0 } } \| X u \| _ { 2 } = \operatorname* { i n f } _ { u \in A } \| X u \| _ { 2 } , \qquad \operatorname* { s u p } _ { u \in A _ { 0 } } \langle h , u \rangle = \operatorname* { s u p } _ { u \in A } \langle h , u \rangle .
$$

Moreover, ${ \sf Q } _ { \alpha } ( P , A _ { 0 } ) \geq { \sf Q } _ { \alpha } ( P , A ) \geq \beta$ . We may therefore apply the countable, measurable form of the small-ball lemma to $A _ { 0 } ;$ the displayed identities transfer its conclusion back to A.

Because A is bounded and the ambient space is finite dimensional, the continuous coercive function $a \mapsto \operatorname* { s u p } _ { u \in A } \| u - a \| _ { 2 }$ attains its minimum (passing to the closure of A does not change the supremum). Orthogonally projecting any minimizer onto af $\because ( A )$ cannot increase its distance to any $u \in A$ . We may therefore choose a minimum enclosing center $a \in \mathrm { a f f } ( A )$ . Let

$$
L _ { A } = \operatorname { s p a n } ( A - a ) , \qquad \dim L _ { A } = q ( A ) .
$$

For

$$
H _ { n } = \frac { 1 } { \sqrt { n } } \sum _ { i = 1 } ^ { n } \varepsilon _ { i } Z _ { i } ,
$$

the Rademacher signs give $\mathbb { E } H _ { n } = 0$ . For each realization, $\begin{array} { r } { \operatorname* { s u p } _ { u \in A } \langle H _ { n } , u \rangle = \langle H _ { n } , a \rangle + } \end{array}$ $\textstyle \operatorname* { s u p } _ { u \in A } \langle H _ { n } , u - a \rangle$ , and $\mathbb { E } \langle H _ { n } , a \rangle = 0$ . Hence

$$
\begin{array} { r l } & { W _ { n } ( A ; P ) = \mathbb { E } \operatorname* { s u p } _ { u \in A } \langle H _ { n } , u \rangle = \mathbb { E } \operatorname* { s u p } _ { u \in A } \langle H _ { n } , u - a \rangle } \\ & { \qquad \leq r ( A ) \mathbb { E } \| P _ { L _ { A } } H _ { n } \| _ { 2 } } \\ & { \qquad \leq r ( A ) \sqrt { \mathbb { E } \| P _ { L _ { A } } H _ { n } \| _ { 2 } ^ { 2 } } . } \end{array}
$$

Isotropy gives $\mathbb { E } H _ { n } H _ { n } ^ { \top } = I _ { p }$ , and therefore

$$
\mathbb { E } \| P _ { L _ { A } } H _ { n } \| _ { 2 } ^ { 2 } = \operatorname { t r } ( P _ { L _ { A } } ) = q ( A ) .
$$

Thus

$$
W _ { n } ( A ; P ) \leq r ( A ) { \sqrt { q ( A ) } } .
$$

Apply Lemma 12 with $\xi = \alpha / 2$ . Since $\mathsf Q _ { \alpha } ( P , A ) \ge \beta$ , this yields (22).

To derive (23), choose $t = \sqrt { 2 \log ( 1 / \delta ) }$ and require separately

$$
2 r ( A ) \sqrt { q ( A ) } \leq \frac { \alpha \beta } { 8 } \sqrt { n } , \qquad \frac { \alpha t } { 2 } \leq \frac { \alpha \beta } { 8 } \sqrt { n } .
$$

Under the resulting sample-size condition, the right side of (22) is at least $\alpha \beta \sqrt { n } / 4$ , so $\Lambda _ { n } ( X ; A ) \ge \alpha ^ { 2 } \beta ^ { 2 } / 1 6$

Finally, Gaussian width is always at most the dimension–radius scale. Indeed,

$$
w ( A ) = \mathbb { E } \operatorname* { s u p } _ { u \in A } \langle g , u - a \rangle \leq r ( A ) \mathbb { E } \| P _ { L _ { A } } g \| _ { 2 } \leq r ( A ) { \sqrt { q ( A ) } } .
$$

The matching identities for $A _ { p , k }$ were proved in the preceding appendix.

## Appendix J. Convex-recovery consequences

Proof Suppose $A = { \mathcal { D } } ( { \mathcal { R } } , \theta ^ { \star } ) \cap { \mathbb { S } } ^ { p - 1 }$ and $X u = 0$ for some $u \in A$ . By the definition of the descent cone, there is $t > 0$ such that

$$
\mathcal { R } ( \theta ^ { \star } + t u ) \leq \mathcal { R } ( \theta ^ { \star } ) .
$$

At the same time $X ( \theta ^ { \star } + t u ) = X \theta ^ { \star }$ . Hence $\theta ^ { \star }$ cannot be the unique solution of (25). This proves the exact-recovery statements in Section 6.

For the smoothed construction, (21) supplies a unit descent direction u with $\| X u \| _ { 2 } \leq$ $\sqrt { 2 } \tau \sqrt { n }$ . In normalized units,

$$
\kappa _ { n } ( X ; A ) : = \frac { 1 } { \sqrt { n } } \operatorname* { i n f } _ { v \in A } \| X v \| _ { 2 } = \sqrt { \Lambda _ { n } ( X ; A ) } \leq \sqrt { 2 } \tau .
$$

Equivalently, the realized-design inverse modulus

$$
K _ { n } ( X ; A ) : = { \frac { 1 } { \kappa _ { n } ( X ; A ) } }
$$

is at least $1 / ( \sqrt { 2 } \tau )$ , with the convention $1 / 0 = \infty$ , and the quadratic modulus $1 / \Lambda _ { n }$ is at least $1 / ( 2 \tau ^ { 2 } )$ . This conclusion is deterministic after X is realized. In particular, the witness u and a descending comparison point $\theta _ { 1 } = \theta ^ { \star } +$ su may depend on that realization. Their normalized observation distance is at most $\sqrt { 2 } \tau s$ , so the statement controls the conditioning of the realized inverse problem; it does not assert a minimax lower bound for two parameters fixed before sampling.