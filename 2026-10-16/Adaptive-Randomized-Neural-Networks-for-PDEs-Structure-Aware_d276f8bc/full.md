# Adaptive Randomized Neural Networks for PDEs: Structure-Aware Feature Redistribution and Residual-Driven Block Enrichment

You Yang<sup>1</sup> and Fei Wang<sup>2</sup>

Abstract. Randomized neural networks retain an output-linear least-squares solve, but globally sampled random features may fail to capture the spatial and scale structure of PDE solutions. We propose SFRBE–RaNN, a two-stage adaptive trial-space construction. SFR redistributes a fixed feature budget according to regional indicator masses and adapts regional weight scales through low-dimensional residual-based optimization. RBE constructs nonredundant enrichments by screening multiple activation families, removing represented response directions, and selecting the smallest admissible prefix. For solution-dependent operators, candidates are screened with a frozen operator and accepted only after nonlinear residual verification. For fixed linear problems, we establish an exact discrete residual-reduction identity and a conditional solution-error estimate. Building on this identity, we combine safeguarded SFR with RBE capture factors to derive a finite-horizon error bound involving baseline best approximation and empirical consistency, together with a conditional random-search success-probability bound. Numerical experiments on localized, multiscale, and nonlinear space–time problems demonstrate the method’s efectiveness.

Keywords. Randomized neural networks, adaptive trial spaces, feature redistribution, residual-informed enrichment, activation-family selection

Mathematics Subject Classification. 68T07, 68T05, 65N35, 65F20, 65M22

## 1 Introduction

Classical numerical methods, including finite diference, finite element, discontinuous Galerkin, and spectral methods, remain central to PDE computation. Nevertheless, localized peaks, oscillations, layers, and evolving interfaces often require adaptive approximation spaces whose resolution follows the underlying solution structure.

Neural PDE solvers ofer flexible trial spaces, including Deep Ritz, DGM, PINNs, and Galerkin neural methods [10, 19, 16, 1]. Their performance, however, can depend heavily on architecture, sampling, and nonlinear training.

In contrast to fully trained neural PDE solvers, randomized neural networks (RaNNs) generate and freeze the hidden representation and determine only the output coeficients from the governing equations. For a shallow representation,

$$
u _ { h } ( \pmb { x } ) = \sum _ { j = 1 } ^ { m } c _ { j } \rho \big ( \pmb { w } _ { j } ^ { \top } \pmb { x } + b _ { j } \big ) ,\tag{1.1}
$$

where the hidden parameters are fixed during each solve and the coeficients are obtained from a linear or linearized least-squares system. This linear output structure avoids nonlinear optimization of all network parameters and i compatible with a range of discretizations [6, 2, 18, 17, 20, 21, 13, 14, 23].

The quality of a RaNN trial space depends on how its finite feature budget is configured: the distribution of hidden weights controls the characteristic length scales and directional variation of the features, feature anchors

determine spatial placement, and the activation family determines the local response profile. A common scale mode is

$$
{ \pmb w } _ { j } \sim { \mathcal { U } } ( - { \pmb r } , { \pmb r } ) = \prod _ { k = 1 } ^ { d } { \mathcal { U } } ( - r _ { k } , r _ { k } ) , \qquad { \pmb r } = ( r _ { 1 } , \dots , r _ { d } ) ,\tag{1.2}
$$

where r controls characteristic scales and directional variation. Feature anchors ${ \pmb a } _ { j } \in \Omega$ are introduced by setting $b _ { j } = - { \pmb w } _ { j } ^ { \top } { \pmb a } _ { j }$ , so that the preactivation vanishes at ${ \pmb a } _ { j }$ . Thus, a randomized trial space depends not only on its width, but also on its feature configuration and response span. The activation family controls the feature profile: tanh produces smooth transitions, Gaussian features produce localized responses, and trigonometric features produce oscillatory variation.

Prior work has adapted several components of neural trial spaces. Global scales and low-dimensional feature distributions have been adjusted to control the overall randomized representation [7, 28]; adaptive activation functions have been used to modify neural PDE responses [12]; local trial spaces have been constructed using residual indicators [21]; residual-guided neural basis construction has been used to progressively expand approximation spaces [5, 25]; and randomized greedy methods have used randomly discretized dictionaries for weak-greedy enrichment [27]. Related residual-driven mechanisms in PINN methods guide sampling, weighting, or network updates [15, 26, 11]. These approaches operate at diferent levels of adaptation. Adaptive activation methods generally learn slopes or shape parameters together with the network parameters, whereas a strict RaNN solve keeps the hidden representation fixed and determines only the output coeficients from a linear or linearized least-squares system. Methods that adapt global feature distributions or local trial spaces do not, by themselves, specify how a fixed feature budget should be redistributed or how features aligned with missing residual directions should be added. This motivates a progressive two-stage approach to adaptive trial-space design: first adapting the existing trial space, and then expanding it with features better suited to the problem at hand.

We propose SFRBE–RaNN as a two-stage adaptive trial-space construction tailored to this setting. SFR reconfigures the existing feature budget using residual, derivative, geometric, or problem-specific indicators to identify structure and background regions, reposition feature anchors, and refresh regional scales without increasing the total width. RBE then adaptively constructs residual-informed feature blocks: candidate families are residualized against the current response space, ranked by a common capture score, and truncated at the smallest admissible nonredundant prefix. Fixed residual operators use block-incremental updates, whereas solution-dependent operators use frozen-operator screening followed by safeguarded reassembly.

The main contributions of this paper are as follows:

• Structure-aware feature redistribution. We introduce SFR to identify localized or physics-defined structures, redistribute a fixed feature budget among the detected structure regions and the background according to their relative indicator masses, and adapt region-specific random weight scales through low dimensional residual-based optimization.

• Residual-informed feature-block construction. RBE compares candidate blocks from diferent activation families, residualizes their responses against the current response space, ranks them by residual capture, and retains the smallest admissible nonredundant prefix.

• Residual reduction and finite-horizon error control. For frozen linear least-squares subproblems, we derive an exact residual-reduction identity for the residual-orthogonal RBE update and a conditional PDE-error estimate. For fixed linear problems, we further combine safeguarded SFR with the cumulative RBE capture factors to establish a finite-horizon error bound, together with a success-probability bound under a conditional coverage assumption for the complete random candidate search. We also provide a separate local one-step safeguard estimate for Picard or Newton updates.

The remainder is organized as follows. Sections 2–4 present the randomized residual formulation and develop the SFR and RBE modules. Section 5 gives the analysis; Section 6 reports the experiments; and the conclusion and appendices summarize the findings and implementation details.

# 2 Randomized Neural PDE Formulation and Adaptive Framework

## 2.1 Randomized neural residual formulation

A shallow randomized neural network defines the trial space and the corresponding approximation as

$$
\mathcal { V } _ { m } = \operatorname { s p a n } \{ \phi _ { 1 } , \dots , \phi _ { m } \} , \qquad \phi _ { j } ( \pmb { x } ) = \rho ( \pmb { w } _ { j } ^ { \top } \pmb { x } + b _ { j } ) , \qquad \boldsymbol { u } _ { h } ( \pmb { x } ) = \sum _ { j = 1 } ^ { m } c _ { j } \phi _ { j } ( \pmb { x } ) .\tag{2.1}
$$

The hidden parameters are fixed during each residual solve, while the output coeficients $\pmb { c } = ( c _ { 1 } , \ldots , c _ { m } ) ^ { \top }$ are determined by least squares.

Consider

$$
\begin{array} { r l } & { \mathcal { L } u = f , \quad \mathrm { i n ~ } \Omega , } \\ & { \beta u = g , \quad \mathrm { o n ~ } \partial \Omega , } \\ & { \mathcal { T } u = h , \quad \mathrm { o n ~ } \Omega _ { 0 } . } \end{array}\tag{2.2}
$$

Let $A _ { r } , A _ { b } ,$ and $A _ { 0 }$ be the residual-response matrices evaluated at the interior, boundary, and initial collocation points, respectively. Let $\eta _ { b } > 0$ and $\eta _ { 0 } > 0$ denote the boundary and initial residual weights. The weighted residual system is defined as

$$
A = \left[ \sqrt { \eta _ { b } } A _ { b } \right] , \qquad { \pmb { y } } = \left[ \sqrt { \eta _ { b } } { \pmb { g } } _ { b } \right] .\tag{2.3}
$$

The output coeficients are computed from

$$
\pmb { c } = \arg \operatorname* { m i n } _ { \pmb { v } } \| \boldsymbol { A } \pmb { v } - \pmb { y } \| _ { 2 } ^ { 2 } .\tag{2.4}
$$

For nonlinear PDEs, each outer iteration produces a linear or linearized least-squares problem of the form

$$
\pmb { c } ^ { k + 1 } = \arg \operatorname* { m i n } _ { \pmb { v } } \| \boldsymbol { A } ^ { k } \pmb { v } - \pmb { y } ^ { k } \| _ { 2 } ^ { 2 } .\tag{2.5}
$$

Here $A ^ { k }$ and $y ^ { k }$ denote the matrix and right-hand side associated with the k-th linearized subproblem. For a fixed-operator Picard iteration, $A ^ { k }$ remains fixed while $y ^ { k }$ is updated; for a Newton-type iteration, $A ^ { k }$ and $y ^ { k }$ are assembled from the current linearization.

## 2.2 Overall adaptive randomized neural network framework

SFRBE–RaNN updates the randomized trial space through two complementary modules:

$$
\mathcal { V } _ { 0 } \xrightarrow { \mathrm { S F R } } \widetilde { \mathcal { V } } _ { 0 } \xrightarrow { \mathrm { R B E } } \mathcal { V } _ { 1 } \xrightarrow { \mathrm { R B E } } \mathcal { V } _ { 2 } \xrightarrow { \mathrm { R B E } } \cdots .
$$

SFR reconfigures the existing fixed width space through feature anchors, neuron allocations, biases, and regional hidden-weight scales, without changing the total number of raw features. RBE then adds a nonredundant space aligned with the unresolved residual:

$$
\mathscr { V } _ { L + 1 } = \mathscr { V } _ { L } + \mathscr { W } _ { L } ^ { \mathrm { n e w } } .
$$

Here $\mathcal { W } _ { L } ^ { \mathrm { n e w } }$ is the accepted enrichment space, represented by accepted composite features for fixed residual operators and by accepted raw features when the operator is updated. Here $m _ { \mathrm { f u l l } }$ and $m _ { \mathrm { s c r } }$ denote the full candidate-block size and the screening size, respectively. Figure 1 summarizes this workflow: the initial randomized space is diagnosed and refreshed by $\mathrm { S F R , }$ and the remaining residual is enriched by RBE.

![](images/ca64d9bf1d9f33f5f0b256cdbccde8217beced077897ee2ea91e94045a14f1e8.jpg)  
Figure 1: Standard randomized neural network followed by the two adaptive modules. SFR diagnoses the residual structure and refreshes the existing feature budget through regional anchors and scales. RBE screens candidate activation families, removes represented directions, and appends only an accepted residual-aligned block.

## 3 Structure-Aware Feature Redistribution

Let $u _ { h }$ denote the current randomized neural-network approximation and let $\mathcal { T } _ { d } = \{ K _ { i } \} _ { i \in \mathbb { Z } _ { d } }$ be a Cartesian diagnostic grid, which is used for marking. SFR keeps the total number of features fixed and relocates their anchors only when the diagnostic indicator is spatially concentrated.

## 3.1 Indicator and global marking

For two-dimensional diagnostic grids, each valid cell $K _ { i }$ with efective measure $\mu _ { i }$ is evaluated at four tensorproduct Gauss–Legendre points $z _ { i q } , q = 1 , \ldots , 4$ . Let $Q _ { i } = \{ q \in \{ 1 , \dots , 4 \} : z _ { i q } \in \Omega \}$ and ${ \mathcal { T } } _ { d } ^ { \mathrm { v } } = \{ i \in { \mathbb { Z } } _ { d } : Q _ { i } \neq \emptyset \}$ ; cells outside $\mathcal { T } _ { d } ^ { \mathrm { v } }$ are excluded from marking and sampling. For $i \in \mathcal { T } _ { d } ^ { \mathrm { v } }$ , set $\omega _ { i q } = \mu _ { i } / | Q _ { i } |$ and define

$$
r _ { i } = \frac { 1 } { \mu _ { i } } \sum _ { j = 1 } ^ { J } \lambda _ { j } \sum _ { q \in Q _ { i } } \omega _ { i q } \left| \mathcal { R } _ { j } ( u _ { h } ; z _ { i q } ) \right| , \qquad g _ { i } = \frac { 1 } { \mu _ { i } } \sum _ { j = 1 } ^ { J } \lambda _ { j } \sum _ { q \in Q _ { i } } \omega _ { i q } \left\| \nabla u _ { h , j } ( z _ { i q } ) \right\| .\tag{3.1}
$$

Here $\mathcal { R } _ { j } ( u _ { h } ; \cdot )$ denotes the residual of the jth equation, including any prescribed row weighting. Also, $J = 1$ for the scalar problems considered in the experiments and $\lambda _ { j } = 1$ . For space–time problems, the diagnostic cells and their efective measures include all independent coordinates. In the three-dimensional space–time Burgers run, the indicators are evaluated at the center of each Cartesian cell with measure $\mu _ { i j k } = h _ { x } h _ { y } h _ { t }$ . With a small numerical constant $\varepsilon _ { n } .$ , the normalized score is

$$
\widehat { r } _ { i } = \frac { r _ { i } } { \operatorname* { m a x } _ { k \in \mathbb { Z } _ { d } ^ { \mathrm { v } } } r _ { k } + \varepsilon _ { n } } , \qquad \widehat { g } _ { i } = \frac { g _ { i } } { \operatorname* { m a x } _ { k \in \mathbb { Z } _ { d } ^ { \mathrm { v } } } g _ { k } + \varepsilon _ { n } } , \qquad s _ { i } = \frac { \widehat { r } _ { i } + \widehat { g } _ { i } } { \operatorname* { m a x } _ { k \in \mathbb { Z } _ { d } ^ { \mathrm { v } } } \widehat { ( r _ { k } + \widehat { g } _ { k } ) } + 2 \varepsilon _ { n } } .\tag{3.2}
$$

The area-aware cell mass is $\eta _ { i } = \mu _ { i } s _ { i }$

The marked set is one global Dörfler set,

$$
\mathcal { M } \in \mathop { \arg \operatorname* { m i n } } _ { \boldsymbol { \mathcal { A } } \subseteq \mathcal { T } _ { d } ^ { \mathrm { v } } } \left\{ \lvert \boldsymbol { \mathcal { A } } \rvert : \sum _ { i \in \mathcal { A } } \eta _ { i } \geq \vartheta \sum _ { i \in \mathcal { T } _ { d } ^ { \mathrm { v } } } \eta _ { i } \right\}
$$

For two-dimensional diagnostic grids, the marked cells are labeled by eight-neighbour connectivity, producing candidate components $\widetilde { \mathcal { C } } _ { \ell } , \ell = 1 , \dots , K _ { \mathrm { r a w } }$ . A candidate component is retained when

$$
\frac { \sum _ { i \in \widetilde { \mathcal { C } } _ { \ell } } \eta _ { i } } { \sum _ { i \in \mathcal { T } _ { d } ^ { \mathrm { v } } } \eta _ { i } + \varepsilon _ { n } } \ge \tau _ { \mathrm { c o n n } } .\tag{3.3}
$$

After filtering, reindex the survivors as $\mathcal { C } _ { \ell } , \ell = 1 , \ldots , K _ { S }$ , set $\begin{array} { r } { \mathcal { I } _ { S } = \bigcup _ { \ell = 1 } ^ { K _ { S } } \mathcal { C } _ { \ell } } \end{array}$ and $\mathcal { T } _ { B } = \mathcal { T } _ { d } ^ { \mathrm { v } } \setminus \mathcal { T } _ { S }$ , and use ℓ for a structure component and B for the complementary background. The connectivity and indicator mass filter removes isolated cells with low scores while preserving small components with substantial accumulated residual and gradient scores.

## 3.2 Anchor redistribution and regional scales

For a ridge feature,

$$
\phi _ { j } ( \pmb { x } ) = \rho ( \pmb { w } _ { j } ^ { \top } \pmb { x } + b _ { j } ) , \qquad b _ { j } = - \pmb { w } _ { j } ^ { \top } \pmb { a } _ { j } ,\tag{3.4}
$$

the anchor ${ \pmb a } _ { j }$ controls the feature location, whereas ${ \pmb w } _ { j }$ controls its direction and scale. To retain a nonzero sampling probability in low-score cells, SFR defines

$$
\bar { \eta } = \frac { \sum _ { i \in \mathcal { Z } _ { d } ^ { \mathrm { v } } } \eta _ { i } } { \sum _ { i \in \mathcal { T } _ { d } ^ { \mathrm { v } } } \mu _ { i } } , \qquad d _ { i } = \eta _ { i } + \delta _ { f } \bar { \eta } \mu _ { i } , \qquad \delta _ { f } = 1 0 ^ { - 3 } .\tag{3.5}
$$

Let m be the fixed total feature budget, let $\textstyle { \mathcal { C } } = \bigcup _ { \ell = 1 } ^ { K _ { S } } { \mathcal { C } } _ { \ell }$ denote the union of the retained structure components, and let ${ \mathcal { T } } _ { B } = { \mathcal { T } } _ { d } ^ { \mathrm { v } } \setminus { \mathcal { C } }$ denote the complementary background. The structure–background allocation is determined by the normalized density mass

$$
\alpha _ { S } = \frac { \sum _ { i \in \mathcal { C } } d _ { i } } { \sum _ { i \in \mathcal { T } _ { d } ^ { \vee } } d _ { i } } , \qquad m _ { S } = \mathrm { r o u n d } ( \alpha _ { S } m ) , \qquad m _ { B } = m - m _ { S } .\tag{3.6}
$$

Thus $m _ { S }$ features are assigned to the retained structure region and $m _ { B }$ features to the background, so that the total width remains fixed. If several structure components are retained, the structure budget $m _ { S }$ is distributed among them in proportion to their density masses. Conditional on the selected region, cells are sampled according to

$$
\mathbb { P } ( \boldsymbol { a } _ { j } \ \mathrm { s e l e c t s } \ K _ { i } \ | \ \ell ) = \frac { d _ { i } } { \sum _ { \boldsymbol { k } \in \mathcal { C } _ { \ell } } d _ { \boldsymbol { k } } } , \qquad \mathbb { P } ( \boldsymbol { a } _ { j } \ \mathrm { s e l e c t s } \ K _ { i } \ | \ \boldsymbol { B } ) = \frac { d _ { i } } { \sum _ { \boldsymbol { k } \in \mathcal { T } _ { \boldsymbol { B } } } d _ { \boldsymbol { k } } } .\tag{3.7}
$$

After a cell is selected, each anchor is sampled uniformly from the physical extent of that cell. Each retained component, as well as the background, is assigned its own adaptively selected scale parameters; admissible scale selection mechanisms are described in Appendix B. After the regional features are built, all output coeficients are recomputed by the same weighted least-squares procedure as in the global solve.

## 3.3 Refresh gate and safeguard

Let $N _ { \mathrm { v a l i d } } = | \mathcal { L } _ { d } ^ { \mathrm { v } } |$ denote the number of valid diagnostic cells. When $K _ { S } \geq 1$ , let $m _ { \ell }$ be the feature count allocated to component $\mathcal { C } _ { \ell } .$ , with $\begin{array} { r } { \sum _ { \ell = 1 } ^ { K _ { S } } m _ { \ell } = m _ { S } } \end{array}$ , and let $\widehat { \alpha } _ { \ell }$ be the fraction of current anchors in that component. Set $\alpha _ { \ell } ^ { \mathrm { t a r } } = m \ell / m$ . The efective indicator support and the coverage deficit (when $K _ { S } \geq 1 )$ are defined by

$$
\chi _ { \mathrm { e f f } } = \frac { \left( \sum _ { i \in \mathbb { Z } _ { d } ^ { \mathrm { v } } } \eta _ { i } \right) ^ { 2 } } { N _ { \mathrm { v a l i d } } \sum _ { i \in \mathbb { Z } _ { d } ^ { \mathrm { v } } } \eta _ { i } ^ { 2 } + \varepsilon _ { n } } , \quad D _ { \mathrm { c o v } } = \operatorname* { m a x } _ { \ell } \frac { [ \alpha _ { \ell } ^ { \mathrm { t a r } } - \widehat \alpha _ { \ell } ] _ { + } } { \alpha _ { \ell } ^ { \mathrm { t a r } } + \varepsilon _ { n } } .\tag{3.8}
$$

SFR is refreshed only if the residual–gradient indicator is suficiently localized, at least one component survives the mass filter, and the current anchors do not cover the target components:

$$
\chi _ { \mathrm { e f f } } \leq \tau _ { \mathrm { e f f } } , \qquad K _ { S } \geq 1 , \qquad D _ { \mathrm { c o v } } \geq \tau _ { \mathrm { c o v e r } } .\tag{3.9}
$$

The redistributed basis is compared with the old basis on the same collocation set. Let $E _ { \mathrm { a c c } } ^ { \mathrm { o l d } }$ and $E _ { \mathrm { a c c } } ^ { \mathrm { n e w } }$ denote the corresponding weighted $\ell ^ { 2 }$ residual norms. The refresh is accepted only when

$$
E _ { \mathrm { a c c } } ^ { \mathrm { n e w } } \leq ( 1 - \eta _ { \mathrm { a c c } } ) E _ { \mathrm { a c c } } ^ { \mathrm { o l d } } , \qquad \eta _ { \mathrm { a c c } } = 0\tag{3.10}
$$

for the default non-increasing safeguard. For nonlinear problems, the acceptance residual is recomputed using the full nonlinear operator after the configured refinement step. If the acceptance criterion is not satisfied, the old basis and coeficients are restored. Thus, for both linear and nonlinear problems, the safeguard ensures that the post-SFR weighted empirical residual does not exceed its pre-SFR value. For fixed linear problems, this property is also used to establish the finite-horizon error bound in Theorem 5.5.

## 4 Residual-Driven Block Enrichment

RBE enlarges the trial space when the current residual still contains unresolved directions. At stage L, let $A _ { L }$ denote the weighted residual-response matrix of the current features, assembled according to (2.3). The corresponding least-squares system is

$$
A _ { L } \pmb { c } _ { L } \approx \pmb { y } .\tag{4.1}
$$

## 4.1 Residual-weighted candidates and screening

Let F denote the problem-specific set of candidate families. In the implementation considered here, we use an optional family scheduling rule to diversify successive RBE enrichments and avoid repeatedly adding block from the same activation family. After a block from family F is accepted, F is excluded from the next candidate construction and screening step. All families are available at the first RBE step, and F becomes available again once a block from another family has been accepted. RBE candidate centers are sampled from the current discrete interior PDE point set $X _ { \mathrm { i n t } } = \{ z _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { i n t } } }$ , using the residual-weighted probabilities below. With $r _ { i }$ denoting the ith interior entry of $r _ { L } ,$ the implementation uses

$$
\widehat { r } _ { i } = \frac { \left| r _ { i } \right| } { \underset { 1 \leq k \leq N _ { \mathrm { i n t } } } { \operatorname* { m a x } } \left| r _ { k } \right| + \varepsilon _ { n } } , \qquad s _ { i } = \delta + \widehat { r } _ { i } ^ { q } , \qquad p _ { i } = ( 1 - \alpha _ { \operatorname* { m i x } } ) \frac { 1 } { N _ { \mathrm { i n t } } } + \alpha _ { \operatorname* { m i x } } \frac { s _ { i } } { \sum _ { k = 1 } ^ { N _ { \mathrm { i n t } } } s _ { k } } ,
$$

where the standard values are $\delta = 3 \times 1 0 ^ { - 3 } , q = 2$ , and $\alpha _ { \mathrm { m i x } } = 0 . 8 5$ . Hence every point retains a nonzero probability while large residuals are preferentially sampled. A screening block contains $m _ { \mathrm { s c r } }$ features. Its low-dimensional family parameters (for example, a scale, frequency, or shape parameter) may be refined by the optional SFRBE parameter search described in Appendix B. The selected parameters are frozen when the full candidate block is formed.

The candidate families and their typical structures are summarized in Table 1.

Table 1: Candidate activation families screened by RBE. The family is chosen from the residual-capture score after residualization; no single activation is prescribed for all PDEs.
<table><tr><td>Family</td><td>Representative feature</td><td>Residual structure typically captured</td></tr><tr><td>tanh</td><td> $\operatorname { t a n h } ( { \pmb w } ^ { \top } { \pmb x } + b )$ </td><td>smooth transitions and anisotropic layers</td></tr><tr><td>ridge_gauss</td><td> $\exp [ - ( { \pmb w } ^ { \top } { \pmb x } + b ) ^ { 2 } ]$ </td><td>localized ridge-like defects</td></tr><tr><td>sin</td><td> $\sin ( \pi ( \boldsymbol { \mathbf { \em w } } ^ { \top } \boldsymbol { \mathbf { x } } + b ) )$ </td><td>oscillatory or high-frequency residuals</td></tr><tr><td>radial_gauss</td><td> $\begin{array} { r } { \exp [ - \sum _ { \ell } r _ { \ell } ^ { 2 } ( x _ { \ell } - a _ { \ell } ) ^ { 2 } ] } \end{array}$ </td><td>isolated peaks and compact defects</td></tr><tr><td>mexican_hat</td><td> $\begin{array} { r } { ( d - s ) \exp ( - s / 2 ) , \quad s = \sum _ { \ell = 1 } ^ { d } r _ { \ell } ^ { 2 } ( x _ { \ell } - a _ { \ell } ) ^ { 2 } } \end{array}$ </td><td>sign-changing localized residuals</td></tr></table>

Let $Q _ { L }$ be an orthonormal basis for the current retained response space and let $r _ { L }$ be the current weighted residual. For any candidate response matrix $C _ { F }$ , form

$$
C _ { \perp , F } = ( I - Q _ { L } Q _ { L } ^ { \top } ) C _ { F } = U _ { F } \Sigma _ { F } V _ { F } ^ { \top } + E _ { F } ,\tag{4.2}
$$

where $U _ { F } \Sigma _ { F } V _ { F } ^ { \mathsf { T } }$ contains the singular components retained by the configured relative SVD threshold and $E _ { F }$ contains the discarded components. Define the regularized capture and map-amplification measures by

$$
s _ { F } = { \frac { \| U _ { F } ^ { \mathsf { T } } { \pmb { r } } _ { L } \| _ { 2 } ^ { 2 } } { \| { \pmb { r } } _ { L } \| _ { 2 } ^ { 2 } + \varepsilon _ { n } } } , \qquad \kappa _ { F } = \| V _ { F } { \pmb { \Sigma } } _ { F } ^ { - 1 } \| _ { 2 } .\tag{4.3}
$$

For the initial family screening, (4.3) is applied to the $m _ { \mathrm { s c r } }$ -feature screening block $C _ { F } ^ { \mathrm { s c r } }$ , and the resulting quantities are denoted by $s _ { F } ^ { \mathrm { s c r } }$ and $\kappa _ { F } ^ { \mathrm { s c r } }$ . The threshold $\tau _ { \mathrm { c a p } } = 1 0 ^ { - 4 }$ is the minimum screening capture score required for a candidate family to proceed to full-block construction.

For the exact identities below, define the corresponding unregularized score for any retained SVD-truncated block by

$$
s _ { F } ^ { ( 0 ) } = \frac { \| U _ { F } ^ { \top } { \pmb r } _ { L } \| _ { 2 } ^ { 2 } } { \| { \pmb r } _ { L } \| _ { 2 } ^ { 2 } } , \qquad { \pmb r } _ { L } \neq { \bf 0 } .\tag{4.4}
$$

Thus, $s _ { F }$ and $s _ { F } ^ { ( 0 ) }$ use the same retained SVD basis and difer only in the denominator regularization.

## 4.2 Prefix selection and safeguarded enrichment

For a family passing the initial screening, RBE generates a full block of $m _ { \mathrm { f u l l } }$ features and ranks its columns by one-column residual capture, computed by applying (4.3) to each residualized column individually. It then tests the prescribed prefix sizes $p \in \mathcal P$ . Let $C _ { \perp , F , p }$ denote the residualized matrix formed by the first $p$ ranked features, and $U _ { F , p }$ denote the retained left singular vectors after SVD truncation. For each prefix, $s _ { F , p }$ and $\kappa _ { F , p }$ are defined by applying (4.3) to the retained SVD of $C _ { \perp , F , p }$ . Likewise, $s _ { F , \mathrm { f u l l } }$ and $\kappa _ { F , \mathrm { f u l l } }$ denote the corresponding quantities obtained from the full block. The smallest admissible prefix is the smallest $p \in \mathcal P$ satisfying

$$
s _ { F , p } \geq \theta _ { \mathrm { p r e f } } s _ { F , \mathrm { f u l l } } ; \quad \kappa _ { F , p } \leq \kappa _ { \operatorname* { m a x } } ^ { \mathrm { m a p } } ; \quad \dim \mathrm { R a n g e } ( U _ { F , p } ) \geq 1 ; \quad s _ { F , p } \geq \tau _ { \mathrm { s t o p } } .\tag{4.5}
$$

Families are examined in decreasing order of $s _ { F } ^ { \mathrm { s c r } }$ , and the first family with such a prefix is selected. The mapconditioning cap is denoted by $\kappa _ { \mathrm { m a x } } ^ { \mathrm { m a p } } ;$ it is $1 0 ^ { 6 }$ for all cases. If no prefix satisfies these conditions, the candidate block is not added.

For a fixed linear operator, let $C _ { \perp , * } = ( I - Q _ { L } Q _ { L } ^ { \mathsf { T } } ) C ;$ be the residualized accepted block, and let $U _ { * }$ denote the orthonormal basis obtained from its retained SVD components. By construction, $Q _ { L } ^ { \mathsf { T } } U _ { * } = 0$ up to numerical roundof. The least-squares correction contributed by the new block is therefore the projection of the current residual onto Range(U<sub>∗</sub>):

$$
\gamma _ { * } = U _ { * } ^ { \mathsf { T } } r _ { L } , \qquad r _ { L + 1 } = r _ { L } - U _ { * } \gamma _ { * } , \qquad Q _ { L + 1 } = [ Q _ { L } , U _ { * } ] .\tag{4.6}
$$

At outer iteration $k ,$ the solution-dependent operator is frozen for candidate screening. Once a block is selected, its raw feature columns are appended, the frozen augmented problem is reassembled, and the coeficients are recomputed:

$$
A _ { L + 1 } ^ { ( k ) } = [ A _ { L } ^ { ( k ) } , C _ { * } ^ { ( k ) } ] , \qquad \pmb { c } _ { L + 1 } ^ { ( k + 1 ) } = \arg \operatorname* { m i n } _ { \pmb { v } } \| A _ { L + 1 } ^ { ( k ) } \pmb { v } - \pmb { y } ^ { ( k ) } \| _ { 2 } ^ { 2 } .\tag{4.7}
$$

The resulting trial state is then refined with the full nonlinear operator before the acceptance test. The block is accepted only if the recomputed nonlinear residual decreases,

$$
\lVert \mathcal { R } _ { \mathrm { n l } } ( u _ { L + 1 } ) \rVert _ { 2 } < \lVert \mathcal { R } _ { \mathrm { n l } } ( u _ { L } ) \rVert _ { 2 } ,\tag{4.8}
$$

Otherwise the trial state is rolled back and enrichment stops. This residual safeguard is the key distinction between candidate screening and an accepted RBE update.

## 4.3 SFR–RBE workflow

The complete procedure is therefore:

1. generate the initial randomized features and solve for the output coeficients;

2. evaluate the residual–gradient indicator and, if (3.9) holds, perform one safeguarded SFR refresh;

3. construct residual-weighted candidate blocks for the configured families, refine their low-dimensional family parameters, and screen them by (4.3);

4. select the smallest admissible prefix satisfying (4.5), append ${ \mathrm { i t } } ,$ and re-solve the augmented problem;

5. for fixed linear operators, accept the selected block and update by (4.6); for solution-dependent operators, accept the trial block only if (4.8) holds; repeat the enrichment procedure.

The same workflow is used throughout; the block update and final acceptance test depend on whether the residual operator is fixed or solution-dependent.

## 5 Residual Reduction and Error Analysis

This section analyzes residual reduction and error control for SFRBE–RaNN. For fixed linear problems, we establish the exact residual reduction produced by an accepted RBE block, connect the residual capture score with the PDE solution error, and derive a finite-horizon error bound for safeguarded SFRBE with a conditional success-probability bound for the randomized candidate search. For solution-dependent operators, we give a separate one-step residual estimate for frozen Picard and Newton updates.

The fixed-linear results assume one fixed afine residual map, including its right-hand side, one weighted collocation rule, and exact unregularized projection onto the retained orthonormal SVD response spaces. Ridge regularization and inexact solves are implementation choices. The finite-horizon probabilistic statement is conditional on candidate coverage and empirical consistency, while the nonlinear result is treated separately because the residual operator may change with the current iterate.

## 5.1 Discrete residual reduction

Consider the weighted least-squares problem over the current retained residual-response space, represented by an orthonormal basis $Q _ { L }$

$$
\pmb { d } _ { L } = \arg \operatorname* { m i n } _ { \pmb { d } } \| Q _ { L } \pmb { d } - \pmb { y } \| _ { 2 } ^ { 2 } = Q _ { L } ^ { \top } \pmb { y } , \qquad \pmb { r } _ { L } = \pmb { y } - Q _ { L } \pmb { d } _ { L } .\tag{5.1}
$$

For a candidate block C, define

$$
C _ { \perp } = ( I - Q _ { L } Q _ { L } ^ { \top } ) C ,
$$

and let

$$
\begin{array} { r } { C _ { \perp } = U \Sigma V ^ { \top } + E _ { \mathrm { t r u n c } } } \end{array}
$$

denote its decomposition into retained and discarded singular components, with $E _ { \mathrm { t r u n c } }$ containing the discarded components. The increment is formed only from Range(U), so the discarded directions are not part of the augmented response space. For the fixed-linear retained block, the equivalent transformed features can be written as $Z = C V \Sigma ^ { - 1 }$ ; hence $( I - Q _ { L } Q _ { L } ^ { \top } ) Z = U$ when the retained SVD relation is exact.

Theorem 5.1 (Residual reduction identity for the block incremental least-squares update). Assume that the current and augmented coeficient vectors are obtained by exact unregularized least-squares projection in the retained spaces. In exact arithmetic, let $\begin{array} { r } { Q _ { L } ^ { \top } Q _ { L } = I , Q _ { L } ^ { \top } U = 0 , U ^ { \top } U = I , } \end{array}$ and $\pmb { r } _ { L } = ( I - Q _ { L } Q _ { L } ^ { \top } ) \pmb { y } \neq \mathbf { 0 }$ . Let Range(U) denote the retained nonredundant residual-response subspace of the SVD-truncated candidate block. For the accepted block C, define $s _ { C } ^ { ( 0 ) } = \| U ^ { \top } { \pmb { r } } _ { L } \| _ { 2 } ^ { 2 } / \| { \pmb { r } } _ { L } \| _ { 2 } ^ { 2 }$ . After augmenting the current response space by Range(U), the block incremental least-squares residual satisfies

$$
\begin{array} { r } { \pmb { r } _ { L + 1 } = \pmb { r } _ { L } - U U ^ { \top } \pmb { r } _ { L } , } \end{array}\tag{5.2}
$$

and

$$
\begin{array} { r l } { \| \pmb { r } _ { L + 1 } \| _ { 2 } ^ { 2 } = \| \pmb { r } _ { L } \| _ { 2 } ^ { 2 } - \| \pmb { U } ^ { \top } \pmb { r } _ { L } \| _ { 2 } ^ { 2 } } & { = \left( 1 - s _ { C } ^ { ( 0 ) } \right) \| \pmb { r } _ { L } \| _ { 2 } ^ { 2 } . } \end{array}\tag{5.3}
$$

Proof. By construction, $Q _ { L } ^ { \top } U = 0$ and $U ^ { \top } U = I$ . Hence $[ Q _ { L } , U ]$ is an orthonormal basis of the augmented response space, and its least-squares residual is

$$
\begin{array} { r l } & { { \pmb r } _ { L + 1 } = { \pmb y } - [ Q _ { L } , U ] [ Q _ { L } , U ] ^ { \top } { \pmb y } } \\ & { \qquad = ( I - Q _ { L } Q _ { L } ^ { \top } ) { \pmb y } - U U ^ { \top } { \pmb y } = { \pmb r } _ { L } - U U ^ { \top } { \pmb r } _ { L } , } \end{array}
$$

where the last equality follows from $U ^ { \top } Q _ { L } = 0$ . Since $U U ^ { \top }$ is an orthogonal projector,

$$
\| \pmb { r } _ { L + 1 } \| _ { 2 } ^ { 2 } = \| \pmb { r } _ { L } \| _ { 2 } ^ { 2 } - \| \pmb { U } ^ { \top } \pmb { r } _ { L } \| _ { 2 } ^ { 2 } .
$$

The definition of $s _ { C } ^ { ( 0 ) }$ completes the proof.

Remark 5.2 (Implementation score). Implementation uses the regularized score $s _ { F }$ for screening and stopping, whereas the exact identity uses $s _ { C } ^ { ( 0 ) }$ on the same retained block. Solution-dependent trials are reassembled and accepted only when the original nonlinear residual decreases.

## 5.2 Conditional PDE solution-error estimate

Let V be a Hilbert solution space on which the diferential operator and the relevant boundary and initial traces are well defined, and let $Y _ { r } , Y _ { b }$ , and $Y _ { 0 }$ be the corresponding residual spaces. For a fixed linear problem, define the continuous residual loss by

$$
\mathcal { L } ( v ) = \| \mathcal { L } v - f \| _ { Y _ { r } } ^ { 2 } + \eta _ { b } \| \mathcal { B } v - g \| _ { Y _ { b } } ^ { 2 } + \eta _ { 0 } \| \mathcal { Z } v - h \| _ { Y _ { 0 } } ^ { 2 } .\tag{5.4}
$$

Assume that there exist constants $C _ { \mathrm { s } } , C _ { \mathrm { c } } > 0$ such that

$$
\begin{array} { r } { C _ { \mathrm { s } } \| v \| _ { V } ^ { 2 } \leq \| \mathcal { L } v \| _ { Y _ { r } } ^ { 2 } + \eta _ { b } \| \mathcal { B } v \| _ { Y _ { b } } ^ { 2 } + \eta _ { 0 } \| \mathcal { L } v \| _ { Y _ { 0 } } ^ { 2 } \leq C _ { \mathrm { c } } \| v \| _ { V } ^ { 2 } , \qquad v \in V . } \end{array}\tag{5.5}
$$

This graph-norm stability condition is problem-dependent, as in [5]. For the two-peak Poisson benchmark in Section 6.2, this framework yields an explicit $L ^ { 2 } .$ -error estimate; its proof is given in Appendix A. For general linear problems, the corresponding residual-to-solution control is expressed through the graph-norm condition above.

Let $u ^ { * } \in V$ be the exact solution satisfying

$$
\begin{array} { r } { \mathcal { L } u ^ { * } = f , \qquad \mathcal { B } u ^ { * } = g , \qquad \mathcal { T } u ^ { * } = h . } \end{array}
$$

Since ${ \mathcal { L } } , B ,$ and I are linear, applying (5.5) to $v - u ^ { * }$ gives

$$
C _ { \mathrm { s } } \| v - u ^ { * } \| _ { V } ^ { 2 } \leq \mathcal { L } ( v ) \leq C _ { \mathrm { c } } \| v - u ^ { * } \| _ { V } ^ { 2 } .\tag{5.6}
$$

Let $R _ { r } ( v ) , R _ { b } ( v )$ , and $ { \boldsymbol { R } } _ { 0 } ( v )$ collect the sampled interior, boundary, and initial residuals, respectively, and let $\mathbf { \boldsymbol { W } } _ { r } = \mathrm { d i a g } ( \omega _ { i } ^ { r } ) , \mathbf { \boldsymbol { W } } _ { b } = \mathrm { d i a g } ( \omega _ { i } ^ { b } )$ , and $W _ { 0 } = \mathrm { d i a g } ( \omega _ { i } ^ { 0 } )$ . The empirical residual loss is

$$
\widehat { \mathcal { L } } _ { N } ( v ) = \| \boldsymbol { W } _ { r } ^ { 1 / 2 } \boldsymbol { R } _ { r } ( v ) \| _ { 2 } ^ { 2 } + \eta _ { b } \| \boldsymbol { W } _ { b } ^ { 1 / 2 } \boldsymbol { R } _ { b } ( v ) \| _ { 2 } ^ { 2 } + \eta _ { 0 } \| \boldsymbol { W } _ { 0 } ^ { 1 / 2 } \boldsymbol { R } _ { 0 } ( v ) \| _ { 2 } ^ { 2 } .\tag{5.7}
$$

The weights are therefore incorporated into the weighted least-squares matrix and right-hand-side vector. Consequently,

$$
\widehat { \mathcal { L } } _ { N } ( u _ { L } ) = \| Q _ { L } Q _ { L } ^ { \top } \pmb { y } - \pmb { y } \| _ { 2 } ^ { 2 } = \| \pmb { r } _ { L } \| _ { 2 } ^ { 2 } .
$$

For two consecutive RBE approximations computed in the retained composite trial spaces, define

$$
\Delta _ { N , L } = \operatorname* { m a x } _ { v \in \{ u _ { L } , u _ { L + 1 } \} } \left| \mathcal { L } ( v ) - \widehat { \mathcal { L } _ { N } } ( v ) \right| .\tag{5.8}
$$

Thus $\Delta _ { N , L }$ measures the empirical-to-continuous residual discrepancy at the two consecutive approximations $u _ { L }$ and $u _ { L + 1 }$

Theorem 5.3 (Conditional solution-error estimate for RBE). Suppose that the graph-norm stability condition (5.5) holds and that the two consecutive approximations are exact unregularized least-squares projections for the same fixed linear residual operator and empirical residual rule. Let $s _ { C } ^ { ( 0 ) }$ denote the unregularized capture score of the accepted SVD-truncated block. Then the main one-step estimate is

$$
\| u _ { L + 1 } - u ^ { * } \| _ { V } ^ { 2 } \leq \frac { ( 1 - s _ { C } ^ { ( 0 ) } ) C _ { \mathrm { c } } } { C _ { \mathrm { s } } } \| u _ { L } - u ^ { * } \| _ { V } ^ { 2 } + \frac { 2 - s _ { C } ^ { ( 0 ) } } { C _ { \mathrm { s } } } \Delta _ { N , L } .\tag{5.9}
$$

Proof. By (5.6) and (5.8),

$$
\begin{array} { r l } & { C _ { \mathrm { s } } \| u _ { L + 1 } - u ^ { * } \| _ { V } ^ { 2 } \leq \widehat { \mathcal { L } } _ { N } ( u _ { L + 1 } ) + \Delta _ { N , L } = ( 1 - s _ { C } ^ { ( 0 ) } ) \widehat { \mathcal { L } } _ { N } ( u _ { L } ) + \Delta _ { N , L } } \\ & { \qquad \leq ( 1 - s _ { C } ^ { ( 0 ) } ) \big ( C _ { \mathrm { c } } \| u _ { L } - u ^ { * } \| _ { V } ^ { 2 } + \Delta _ { N , L } \big ) + \Delta _ { N , L } , } \end{array}
$$

where the equality follows from (5.3).

Remark 5.4 (Conditional error interpretation). Theorem 5.3 separates the efect of residual projection from the empirical-to-continuous consistency term. When the two losses agree, the estimate reduces to a contraction criterion:

$$
s _ { C } ^ { ( 0 ) } > 1 - \frac { C _ { \mathrm { s } } } { C _ { \mathrm { c } } } .
$$

## 5.3 Finite-horizon randomized enrichment and SFRBE error control

The actual capture factors can be accumulated along accepted fixed-linear updates. Together with the SFR safeguard, this relates the final error to the pre-SFR baseline without imposing a uniform statistical bound over an unbounded trial space. Let $V _ { \mathrm { b a s e } } \subset V$ be the finite-dimensional pre-SFR trial space, $u _ { \mathrm { b a s e } }$ its exact empirical least-squares minimizer, and $u _ { 0 }$ the accepted or restored post-SFR state. Define

$$
v _ { \mathrm { b a s e } } ^ { \star } \in \arg \operatorname* { m i n } _ { v \in V _ { \mathrm { b a s e } } } \| v - u ^ { * } \| _ { V } ^ { 2 } , \qquad A _ { \mathrm { b a s e } } = \| v _ { \mathrm { b a s e } } ^ { \star } - u ^ { * } \| _ { V } ^ { 2 } ,
$$

and, for L accepted RBE blocks,

$$
\Pi _ { L } = \prod _ { \ell = 0 } ^ { L - 1 } ( 1 - s _ { \ell } ^ { ( 0 ) } ) , \qquad \Pi _ { 0 } = 1 ,
$$

where $s _ { \ell } ^ { ( 0 ) }$ is the unregularized capture score of the retained block from (4.4).

Theorem 5.5 (Finite-horizon RBE capture and safeguarded SFRBE error). Fix $L \geq 1$ and consider the same fixed afine residual map (including its right-hand side), weighted collocation rule, and exact unregularized projection onto retained response spaces throughout SFR and the L accepted RBE updates. Assume the graph-norm equivalence (5.5) and the SFR safeguard (3.10). Suppose, for this realized final state, that the two-point empirical-consistency estimates hold:

$$
\left| \mathcal { L } ( v _ { \mathrm { b a s e } } ^ { \star } ) - \widehat { \mathcal { L } } _ { N } ( v _ { \mathrm { b a s e } } ^ { \star } ) \right| \leq { \varepsilon } _ { \mathrm { b a s e } } , \qquad \left| \mathcal { L } ( u _ { L } ) - \widehat { \mathcal { L } } _ { N } ( u _ { L } ) \right| \leq { \varepsilon } _ { L } ,\tag{5.10}
$$

where $\varepsilon _ { \mathrm { b a s e } } , \varepsilon _ { L } \geq 0$ . Then every such successful trajectory satisfies

$$
\Vert u _ { L } - u ^ { * } \Vert _ { V } ^ { 2 } \leq \frac { C _ { \mathrm { c } } } { C _ { \mathrm { s } } } \Pi _ { L } A _ { \mathrm { b a s e } } + \frac { \Pi _ { L } \varepsilon _ { \mathrm { b a s e } } + \varepsilon _ { L } } { C _ { \mathrm { s } } } .\tag{5.11}
$$

In particular, if the implementation score of every accepted block is at least $\tau _ { \mathrm { s t o p } } \in ( 0 , 1 )$ , then $\Pi _ { L } \leq ( 1 - \tau _ { \mathrm { s t o p } } ) ^ { L } .$

For the random-search statement, let $G _ { \ell }$ be the event that the complete random candidate-generation, SVD truncation, conditioning, and prefix-selection procedure at stage ℓ finds an admissible block passing the stopping threshold. $H ,$ conditional on every preceding successful history $\mathcal { F } _ { \ell }$

$$
\mathbb { P } ( G _ { \ell } \mid \mathcal { F } _ { \ell } ) \ge p _ { 0 } > 0 , \qquad \ell = 0 , \ldots , L - 1 ,\tag{5.12}
$$

then $\mathbb { P } ( G _ { 0 } \cap \cdot \cdot \cdot \cap G _ { L - 1 } ) \geq p _ { 0 } ^ { L }$ . If (5.10) holds with $\varepsilon _ { \mathrm { b a s e } } , \varepsilon _ { L } \leq \varepsilon _ { N }$ on every such successful history, then, with probability at least $p _ { 0 } ^ { L }$ ,

$$
\| u _ { L } - u ^ { * } \| _ { V } ^ { 2 } \leq \frac { C _ { \mathrm { c } } } { C _ { \mathrm { s } } } ( 1 - \tau _ { \mathrm { s t o p } } ) ^ { L } A _ { \mathrm { b a s e } } + \frac { 2 \varepsilon _ { N } } { C _ { \mathrm { s } } } .\tag{5.13}
$$

More generally, suppose an event $H _ { L }$ on the ambient probability space satisfies $\mathbb { P } ( H _ { L } ) \ge 1 - \delta _ { N }$ and enforces the same two-point bounds on every successful path in $H _ { L }$ . Then (5.13) holds with probability at least $\operatorname* { m a x } \{ 0 , p _ { 0 } ^ { L } - \delta _ { N } \}$

Proof. By the SFR safeguard and baseline empirical optimality, $\widehat { \mathcal { L } _ { N } } ( u _ { 0 } ) \leq \widehat { \mathcal { L } _ { N } } ( v _ { \mathrm { b a s e } } ^ { \star } )$ . Iterating (5.3) for the unchanged weighted afine residual map gives $\widehat { \mathcal { L } _ { N } } ( u _ { L } ) = \Pi _ { L } \widehat { \mathcal { L } _ { N } } ( u _ { 0 } )$ . Consequently, (5.6) and the two estimates in (5.10) imply

$$
\begin{array} { r l } & { C _ { \mathrm { s } } \| u _ { L } - u ^ { * } \| _ { V } ^ { 2 } \leq \widehat { \mathcal { L } } _ { N } ( u _ { L } ) + \varepsilon _ { L } } \\ & { \qquad \leq \Pi _ { L } \big ( \mathcal { L } ( v _ { \mathrm { b a s e } } ^ { \star } ) + \varepsilon _ { \mathrm { b a s e } } \big ) + \varepsilon _ { L } } \\ & { \qquad \leq \Pi _ { L } C _ { \mathrm { c } } A _ { \mathrm { b a s e } } + \Pi _ { L } \varepsilon _ { \mathrm { b a s e } } + \varepsilon _ { L } . } \end{array}
$$

The regularized capture score is no greater than its unregularized counterpart, giving the geometric factor. Conditional multiplication of success probabilities yields $p _ { 0 } ^ { L }$ ; intersection with the consistency event and the union bound yield the final probability claim. □

Remark 5.6 (Conditional coverage and scope). The event $G _ { \ell }$ concerns the final admissible block, not an unscreened neuron. For example, set $Z _ { \ell } = 0$ on an unsuccessful full search and otherwise let it be the selected block’s regularized capture score. Since $0 \le Z _ { \ell } \le \mathbf { 1 } _ { G _ { \ell } }$ , a conditional bound $\mathbb { E } [ Z _ { \ell } \mid { \mathcal { F } } _ { \ell } ] \geq a > 0$ implies (5.12) with $p _ { 0 } = a$ . Neither this coverage condition nor the two-point consistency bound follows automatically from residual-weighted sampling; in particular, consistency for the data-selected $u _ { L }$ requires independent validation or suitable complexity control. The positive stopping threshold and possible search failure do not establish unconditional infinite-horizon convergence.

Remark 5.7 (Nonlinear updates). For solution-dependent Picard or Newton methods, the fixed-linear results describe the exact projection step for the corresponding frozen subproblem. After the operator is updated, the enlarged system is reassembled and the original nonlinear residual is recomputed for the acceptance test (4.8).

Lemma 5.8 (One-step estimate for a frozen nonlinear operator). Let $\mathcal { F } , \mathcal { F } _ { k } : X  Y$ act between normed spaces and satisfy $\mathcal { F } _ { k } ( u _ { k } ) = \mathcal { F } ( u _ { k } )$ . Suppose the trial state $\widetilde { u } _ { k + 1 }$ satisfies

$$
\| \mathcal { F } _ { k } ( \widetilde { u } _ { k + 1 } ) \| _ { Y } \le \alpha _ { k } \| \mathcal { F } ( u _ { k } ) \| _ { Y } , \quad 0 \le \alpha _ { k } < 1 ,
$$

and $\Vert \mathcal { F } ( \widetilde { u } _ { k + 1 } ) - \mathcal { F } _ { k } ( \widetilde { u } _ { k + 1 } ) \Vert _ { Y } \leq \omega _ { k } \Vert \widetilde { u } _ { k + 1 } - u _ { k } \Vert _ { X }$ . Then

$$
\| \mathcal { F } ( \widetilde { u } _ { k + 1 } ) \| _ { Y } \leq \alpha _ { k } \| \mathcal { F } ( u _ { k } ) \| _ { Y } + \omega _ { k } \| \widetilde { u } _ { k + 1 } - u _ { k } \| _ { X } .\tag{5.14}
$$

In particular, a non-increase of the true residual follows if

$$
\alpha _ { k } \| \mathcal { F } ( u _ { k } ) \| _ { Y } + \omega _ { k } \| \widetilde u _ { k + 1 } - u _ { k } \| _ { X } \leq \| \mathcal { F } ( u _ { k } ) \| _ { Y } .\tag{5.15}
$$

For the Newton model $\mathcal F _ { k } ( v ) = \mathcal F ( u _ { k } ) + \mathcal F ^ { \prime } ( u _ { k } ) ( v - u _ { k } )$ , a Jacobian with Lipschitz constant $L _ { J }$ on the trial segment yields $\begin{array} { r } { \| \mathcal { F } ( \widetilde u _ { k + 1 } ) - \mathcal { F } _ { k } ( \widetilde u _ { k + 1 } ) \| _ { Y } \le \frac { 1 } { 2 } L _ { J } \| \widetilde u _ { k + 1 } - u _ { k } \| _ { X } ^ { 2 } } \end{array}$

Proof. The first estimate is the triangle inequality. For Newton, with $d = \widetilde { u } _ { k + 1 } - u _ { k }$ , write the operator diference as $\begin{array} { r } { \int _ { 0 } ^ { 1 } [ \mathcal { F } ^ { \prime } ( u _ { k } + t d ) - \mathcal { F } ^ { \prime } ( u _ { k } ) ] d d t } \end{array}$ and use $\begin{array} { r } { \int _ { 0 } ^ { 1 } t L _ { J } \| \dot { d } \| _ { X } ^ { 2 } d t = \frac { 1 } { 2 } L _ { J } \| d \| _ { X } ^ { 2 } } \end{array}$ □

## 5.4 Scope of the theoretical statements

Theorems 5.1–5.5 apply to a single fixed afine residual map, including its right-hand side and collocation weights. The final-state consistency assumption is conditional; uniform statistical bounds of this type, of order $m \sqrt { \log m / N }$ in the width m and the number of collocation points N, are available for fixed-width trial spaces with bounded hidden parameters and i.i.d. uniform collocation points [5]. For Picard/Newton iterations, the fixed-linear projection identity concerns only a frozen subproblem, not the original nonlinear PDE; accepted nonlinear trials are separately checked by Lemma 5.8.

## 6 Numerical experiments

## 6.1 Experimental setup

We test the two adaptive modules on localized and multiscale elliptic problems, variable-coeficient Darcy flow, and nonlinear space–time KdV and Burgers equations. In most experiments, the value error is measured by the relative $L ^ { 2 }$ error:

$$
\mathrm { R e l .  ~ } L ^ { 2 } = \frac { \| u _ { h } - u _ { \mathrm { r e f } } \| _ { L ^ { 2 } ( \Omega ) } } { \| u _ { \mathrm { r e f } } \| _ { L ^ { 2 } ( \Omega ) } } .\tag{6.1}
$$

For the reported gradient errors, we use the full-coordinate $H ^ { 1 }$ norm

$$
\| v \| _ { H ^ { 1 } ( \Omega ) } ^ { 2 } = \| v \| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } + \| \nabla _ { z } v \| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } , \qquad \mathrm { R e l . ~ } H ^ { 1 } = \frac { \| u _ { h } - u _ { \mathrm { r e f } } \| _ { H ^ { 1 } ( \Omega ) } } { \| u _ { \mathrm { r e f } } \| _ { H ^ { 1 } ( \Omega ) } } .\tag{6.2}
$$

Here $u _ { h }$ denotes the computed solution and $u _ { \mathrm { r e f } }$ denotes either the exact solution, when available, or a high-accuracy reference solution generated by a reliable traditional numerical method. The vector z contains all independent coordinates, so the gradient includes the temporal derivative in space–time problems. The integrals in (6.1) and (6.2) are evaluated by numerical quadrature on the prescribed evaluation grid.

Here “AD–RaNN” denotes the global-scale adaptive-distribution initialization used as the initial space and a comparison baseline; it is distinct from the parameter search in Appendix B.

All numerical experiments are implemented in MATLAB, where the least-squares systems, residual evaluations, SFR updates, and RBE procedures are assembled and solved. The schematic neural-network and algorithmic illustrations are generated in Python using Matplotlib. The two-peak Poisson and periodic KdV results are averaged over ten MATLAB random seeds $0 , \ldots , 9$ , whereas the other reported experiments use seed 4 unless otherwise stated.

All spatial and temporal derivatives of the random features are evaluated analytically. Unless otherwise stated, “Features” and “Total features” denote the stage-wise and cumulative numbers of raw randomized features, respectively.

## 6.2 Two-peak Poisson benchmark: ablation and error analysis

Problem setup, numerical results, and ablation The first test considers the Poisson problem

$$
\begin{array} { r l } { - \Delta u = f , } & { { } ( x , y ) \in \Omega , } \\ { u = g , } & { { } ( x , y ) \in \partial \Omega , } \end{array} \quad \quad \Omega = ( 0 , 1 ) ^ { 2 } .\tag{6.3}
$$

The source term f and the Dirichlet boundary data g are generated from the manufactured exact solution

$$
u = \exp \left( - \frac { ( x - 0 . 3 0 ) ^ { 2 } + ( y - 0 . 3 5 ) ^ { 2 } } { 0 . 0 2 ^ { 2 } } \right) + 0 . 8 5 \exp \left( - \frac { ( x - 0 . 7 2 ) ^ { 2 } + ( y - 0 . 6 8 ) ^ { 2 } } { 0 . 0 5 ^ { 2 } } \right) .
$$

The manufactured solution contains two localized Gaussian peaks and is used to test whether the proposed structure-aware mechanism can detect multiple separated residual structures and redistribute the randomized feature anchors accordingly. After the SFR refresh, RBE is applied to further resolve the remaining localized and oscillatory residual components.

Table 2: Numerical results for the two-peak Poisson equation (6.3) (mean $/$ best $/$ worst over seeds $0 , \ldots , 9 )$ “Features” denotes the cumulative raw randomized-feature count reported at each stage; SFR keeps a fixed-width trial space. The values after ± denote one standard deviation over the ten seeds. Times are mean stage wall-clock times.
<table><tr><td>Stage</td><td>Features (avg.)</td><td> ${ \mathrm { T i m e ~ } } ( { \mathrm { s } } , { \mathrm { a v g . } } )$ </td><td>Rel.  $L ^ { 2 } \ \mathrm { ( a v g . ) }$ </td><td> ${ \mathrm { R e l . ~ } } L ^ { 2 } \ ( \mathrm { b e s t } )$ </td><td>Rel. L² (worst)</td></tr><tr><td>AD-RaNN</td><td>3500</td><td>8.477</td><td> $2 . 0 6 8 \mathrm { e } { - } 0 2 \pm 1 . 9 5 0 \mathrm { e - } 0 3$ </td><td> $1 . 8 2 9 \mathrm { { e } - 0 2 }$ </td><td> $2 . 3 2 7 \mathrm { e } { - 0 2 }$ </td></tr><tr><td>SFR</td><td>3500</td><td>24.555</td><td> $4 . 8 6 6 \mathrm { { e } - 0 3 \pm 1 . 1 7 1 \mathrm { { e } - 0 3 } }$ </td><td>3.609e-03</td><td>6.974e-03</td></tr><tr><td>RBE</td><td>6450</td><td>93.250</td><td> $1 . 1 6 4 \mathrm { e } { - } 0 6 \pm 2 . 1 1 4 \mathrm { e } { - } 0 6$ </td><td>8.588e-08</td><td>7.096e-06</td></tr></table>

Table 3: Mean accepted RBE features by activation family over Poisson seeds $0 , \ldots , 9 .$
<table><tr><td>Family</td><td></td><td></td><td></td><td></td><td>tanh ridge_gauss sin radial_gauss mexican_hat</td></tr><tr><td>Mean accepted features 380</td><td></td><td>370</td><td>830</td><td>480</td><td>890</td></tr></table>

![](images/189fd54b286bc262795463d12219619c6ccbf4140490958a0f54142f80e82e10.jpg)

![](images/584774e8b3fe8b3e0e4ed0fcbf974b69ba1ac07963f4ccebfaf97d3a6a3aad0a.jpg)

![](images/1f4737e714a230383f3b0da746865bc9d893030b03a3307350aea584ff1015c5.jpg)  
Figure 2: Two-peak Poisson equation. Exact solution (left), final numerical prediction after RBE enrichment (middle), and SFR feature-anchor distribution (right). In the feature-anchor plot, diferent colors distinguish the detected structure regions and the background region.

As shown in Table $2 ,$ the initial AD–RaNN state gives an average relative $L ^ { 2 }$ error of $2 . 0 6 8 \times 1 0 ^ { - 2 }$ over MATLAB seeds $0 , \ldots , 9$ (best $1 . 8 2 9 \times 1 0 ^ { - 2 }$ , worst $2 . 3 2 7 \times 1 0 ^ { - 2 } )$ . The SFR refresh reduces the average error $\mathrm { t o ~ 4 . 8 6 6 \times 1 0 ^ { - 3 } }$ (best $3 . 6 0 9 \times 1 0 ^ { - 3 }$ , worst $6 . 9 7 4 \times 1 0 ^ { - 3 } )$ by reallocating the original randomized features between the detected structure regions and the background region. Starting from this SFR-refreshed approximation, RBE further decreases the average relative $L ^ { 2 }$ error to $1 . 1 6 4 \times 1 0 ^ { - 6 }$ (best $8 . 5 8 8 \times 1 0 ^ { - 8 }$ , worst $7 . 0 9 6 \times 1 0 ^ { - 6 } )$ . Figures 2 and 3 visualize the final prediction, the SFR feature-anchor distribution, and the progressive reduction of the pointwise error.

The reported wall-clock times are implementation-dependent and are included only as practical reference values. Table 4 compares SFRBE–RaNN with Expert-guided PINN [24], HyResPINN [4], adaptive finite-element baselines, AG–RaNN [5], Adaptive RFM [2], and KAPI–ELM [9], together with feature-count-matched AD–RaNN, SFR–RaNN and RBE–RaNN ablations.

The adaptive finite-element results use the iFEM MATLAB package with the standard solve–estimate–mark– refine loop and multigrid-preconditioned conjugate gradients [3, 8, 22]. The meshes in Figure 4 concentrate around the two Gaussian peaks. Adaptive P2 FEM obtains $3 . 1 9 0 \times 1 0 ^ { - 7 }$ in relative $L ^ { 2 }$ and $1 . 1 5 8 \times 1 0 ^ { - 5 }$ in full $H ^ { 1 }$ , using $5 . 7 2 5 \times 1 0 ^ { 6 }$ degrees of freedom.

![](images/9eee18174772695b8029dec62201a92860ae60424f20203db52443004b412083.jpg)

![](images/6ea9df9cf2a1e18c4c74c75e97582b63df2cfbb997b2f8bda3e5dd9e55d935ee.jpg)

![](images/51f67492f6ec4e3394ade4243fc6f0595b9b0b324897503afddfbf1d8ceb54d4.jpg)  
Figure 3: Pointwise absolute error maps for the two-peak Poisson equation. From left to right: initial AD–RaNN, after $\mathrm { S F R , }$ and after RBE.

Table 4: Comparison and mechanism ablation on the two-peak Poisson equation. The upper panel compares SFRBE–RaNN with representative external baselines. The lower panel isolates the efects of global initialization, fixed-budget feature redistribution, and residual-driven enrichment, with the wider SFR–RaNN row serving as a capacity control. Times and errors for the AD–RaNN, SFR–RaNN, RBE–RaNN (no SFR), and SFRBE–RaNN rows are means over seeds $0 , \ldots , 9 ;$ the remaining rows are representative comparison or capacity-control results. The columns report absolute and relative $L ^ { 2 }$ errors, together with the corresponding errors measured in the full $H ^ { 1 }$ norm.
<table><tr><td colspan="7">(a) Comparison with external baselines</td></tr><tr><td>Method</td><td>DOFs / features</td><td>Time (s)</td><td> $\mathrm { \ A b s . } \ L ^ { 2 }$ </td><td> $\mathrm { R e l } , \ L ^ { 2 }$ </td><td> $\mathrm { A b s . } ~ H ^ { 1 }$ </td><td> $\mathrm { R e l . } \ H ^ { 1 }$ </td></tr><tr><td>Expert-guided PINN</td><td>462593 params.</td><td>5062.382</td><td>1.366e-02</td><td>2.321e-01</td><td>7.375e-02</td><td>3.168e-02</td></tr><tr><td>HyResPINN</td><td>256711 params.</td><td>6806.590</td><td>1.220e-03</td><td>2.073e-02</td><td>1.564e-02</td><td>6.717e-03</td></tr><tr><td>Adaptive P1 FEM</td><td>7247310 DOFs</td><td>279.380</td><td>2.029e-07</td><td>3.447e-06</td><td>2.670e-03</td><td>1.147e-03</td></tr><tr><td>Adaptive P2 FEM</td><td>5724817 DOFs</td><td>195.110</td><td>1.878e-08</td><td>3.190e-07</td><td>2.695e-05</td><td>1.158e-05</td></tr><tr><td>AG-RaNN-1</td><td>1200 features</td><td>3.742</td><td>4.462e-04</td><td>7.581e-03</td><td>4.755e-02</td><td>2.043e-02</td></tr><tr><td>AG-RaNN-2</td><td>6450 features</td><td>40.203</td><td>2.246e-04</td><td>3.816e-03</td><td>3.704e-02</td><td>1.591e-02</td></tr><tr><td>Adaptive RFM</td><td>6450 features</td><td>31.204</td><td>1.731e-04</td><td>2.947e-03</td><td>1.210e-02</td><td>5.212e-03</td></tr><tr><td>KAPI-ELM</td><td>6450 features</td><td>178.071</td><td>4.993e-06</td><td>8.503e-05</td><td>2.195e-04</td><td>9.457e-05</td></tr></table>

<table><tr><td colspan="8">(b) Mechanism ablation and capacity controls</td></tr><tr><td>Method</td><td>DOFs / features</td><td>Time (s)</td><td> $\mathrm { \ A b s . } \ L ^ { 2 }$ </td><td> $\mathrm { R e l } , \ L ^ { 2 }$ </td><td> $\mathrm { A b s } , \ H ^ { 1 }$ </td><td> $\mathrm { R e l . } \ H ^ { 1 }$ </td></tr><tr><td>AD-RaNN (matched)</td><td>6450 features</td><td>14.489</td><td> $3 . 4 7 2 \mathrm { e } - 0 4$ </td><td> $5 . 8 9 7 \mathrm { e } { - 0 3 }$ </td><td> $7 . 3 5 4 \mathrm { e } { - 0 2 }$ </td><td>3.160e-02</td></tr><tr><td>SFR-RaNN (ours)</td><td>6450 features</td><td>81.414</td><td>2.676e-05</td><td>4.545e-04</td><td>5.719e-03</td><td>2.458e-03</td></tr><tr><td>SFR-RaNN (ours, wider)</td><td>12000 features</td><td>434.678</td><td>3.530e-07</td><td>5.997e-06</td><td>1.057e-04</td><td>4.541e-05</td></tr><tr><td>RBE-RaNN (no SFR)</td><td>6630 features  $\left( \mathrm { a v g . } \right)$ </td><td>116.075</td><td>4.849e-07</td><td>8.236e-06</td><td>9.213e-05</td><td>3.959e-05</td></tr><tr><td>SFRBE-RaNN (ours)</td><td>6450 features (avg.)</td><td>126.282</td><td>6.854e-08</td><td>1.164e-06</td><td>1.613e-05</td><td>6.933e-06</td></tr></table>

SFRBE–RaNN obtains an average relative $L ^ { 2 }$ error of $1 . 1 6 4 \times 1 0 ^ { - 6 }$ (best $8 . 5 8 8 \times 1 0 ^ { - 8 }$ , worst $7 . 0 9 6 \times 1 0 ^ { - 6 }$ over seeds 0–9) and an average full $H ^ { 1 }$ error of $6 . 9 3 3 \times 1 0 ^ { - 6 }$ , using an average final total of 6450 randomized features and 126.282 seconds of compute time, excluding figure generation. Feature counts and finite-element degrees of freedom are not directly equivalent; this comparison highlights the mesh-free representation and direct evaluation of SFRBE–RaNN rather than a claim of uniformly lower error.

![](images/a7079c69f2a30edf6631ee90d95126d273bbe66384090e95f2dfbe8470db93ae.jpg)  
Figure 4: Adaptive finite element results for the two-peak Poisson equation (6.3). From left to right: adaptive P1 mesh, absolute and relative $L ^ { 2 }$ and full $H ^ { 1 }$ error decay for adaptive P1 FEM, adaptive P2 mesh, and the corresponding error decay for adaptive P2 FEM. Both meshes are shown at the 20th AFEM iteration.

At a matched final feature count, AD–RaNN, SFR–RaNN, and SFRBE–RaNN achieve mean relative $L ^ { 2 }$ errors of $( 5 . 8 9 7 \times 1 0 ^ { - 3 } ) , ( 4 . 5 4 5 \times 1 0 ^ { - 4 } )$ , and $( 1 . 1 6 4 \times 1 0 ^ { - 6 } )$ , respectively. These results compare the three feature-space configurations at comparable final widths. As a complementary ablation, RBE–RaNN (no SFR) starts from the same 3500-feature $\mathrm { \ A D - R a N N }$ space, accepts nine RBE blocks, and reaches a ten-seed mean of 6630 features with relative $L ^ { 2 }$ error $8 . 2 3 6 \times 1 0 ^ { - 6 }$ . Thus, RBE alone improves the initial trial space, while applying SFR first yields a further gain; the wider SFR–RaNN result is reported separately as a capacity control.

Poisson residual-to-error estimate For the two-peak Poisson problem (6.3), define $R ( v ) = - \Delta v - f$ and $b ( v ) = v | _ { \partial \Omega } - g _ { \mathrm { : } }$ , with $\| \cdot \| _ { \Omega } = \| \cdot \| _ { L ^ { 2 } ( \Omega ) }$ and $\| \cdot \| _ { \partial \Omega } = \| \cdot \| _ { L ^ { 2 } ( \partial \Omega ) }$ . For $u ^ { \ast } , v \in H ^ { 2 } ( \Omega )$ , the following explicit stability estimate holds:

$$
\| v - u ^ { * } \| _ { L ^ { 2 } ( \Omega ) } \leq \frac { \| R ( v ) \| _ { L ^ { 2 } ( \Omega ) } } { 2 \pi ^ { 2 } } + \frac { \| b ( v ) \| _ { L ^ { 2 } ( \partial \Omega ) } } { \sqrt { 2 } } .\tag{6.4}
$$

The proof is given in Appendix A.

For the seed-4 two-peak Poisson run, let $\rho _ { L } = \| R ( u _ { L } ) \| _ { L ^ { 2 } ( \Omega ) } , \beta _ { L } = \| b ( u _ { L } ) \| _ { L ^ { 2 } ( \partial \Omega ) }$ , and $e _ { L } = \Vert u _ { L } - u ^ { * } \Vert _ { L ^ { 2 } ( \Omega ) }$ denote the interior residual norm, boundary residual norm, and absolute $L ^ { 2 } ( \Omega )$ solution error, respectively. Define $B _ { L } = \rho _ { L } / ( 2 \pi ^ { 2 } ) + \beta _ { L } / \sqrt { 2 } .$ . For the accepted block from stage L to stage $L + 1$ , let $\varrho _ { L } = \| r _ { L } \| _ { 2 }$ be the weighted discrete residual norm. Then $q _ { L } ^ { \mathrm { o b s } } = { \varrho _ { L + 1 } } / { \varrho _ { L } }$ and $q _ { L } ^ { \mathrm { p r e d } } = \sqrt { 1 - s _ { L } ^ { ( 0 ) } }$ . The observed and predicted ratios are (0.332022, 0.332024) for RBE 2 and (0.638280, 0.638279) for RBE 8. The explicit Poisson bound is conservative, with $\boldsymbol { B } _ { L } / \boldsymbol { e } _ { L }$ remaining on the order of $1 0 ^ { 3 }$ . Complete stagewise diagnostics are reported in Table 5.

Table 5: Residual-to-error and RBE projection diagnostics for the detailed seed-4 Poisson run.
<table><tr><td colspan="5">Poisson residual-to-error diagnostics</td><td rowspan="2"></td><td colspan="4">RBE projection diagnostics</td></tr><tr><td>Stage</td><td> $\rho _ { L }$ </td><td> $\beta _ { L }$ </td><td> $e _ { L }$ </td><td> $B _ { L }$ </td><td> $B _ { L } / e _ { L }$ </td><td> $\varrho _ { L }$ </td><td> $s _ { L } ^ { ( 0 ) }$ </td><td> $q _ { L } ^ { \mathrm { o b s } }$   $q _ { L } ^ { \mathrm { p r e d } }$ </td></tr><tr><td></td><td>Initial 3.7801e+01 2.0260e-04 1.0768e-03 1.9152e+00</td><td></td><td></td><td></td><td> $1 . 7 8 \mathrm { e } + 0 3$ </td><td></td><td></td><td></td><td></td></tr><tr><td>SFR</td><td></td><td></td><td>1.0008e+01 1.1999e-04 2.4459e-04 5.0710e-01 2.07e+03 5.5174e-02</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 1 1.1027e-01 9.0593e-07 2.2659e-06 5.5870e-03 2.47e+03 6.0005e-04 0.99988 0.010876 0.010954</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 2 3.6376e-02 3.7292e-07 1.2297e-061.8431e-03 1.50e+03 1.9923e-04 0.88976 0.332022 0.332024</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 3 1.7380e-02 1.4947e-074.1892e-07 8.8059e-04 2.10e+03 9.4681e-05 0.77416 0.475235 0.475226</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 4 9.7973e-03 9.4211e-08 1.9670e-07 4.9640e-04 2.52e+03 5.3544e-050.68019 0.565520 0.565517</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 5 5.7025e-03 3.9802e-08 9.1348e-08 2.8892e-04 3.16e+03 3.0929e-050.66633 0.577637 0.577642</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 6 3.3086e-03 2.4162e-08 5.4388e-08 1.6763e-04 3.08e+03 1.7960e-05 0.66282 0.580685 0.580672</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 7 1.4387e-03 1.1447e-08 2.1626e-08 7.2893e-05 3.37e+03 7.8229e-06 0.81027 0.435573 0.435580</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>RBE 8 9.2044e-04 6.4937e-09 1.2849e-08 4.6635e-05 3.63e+034.9932e-06 0.59260 0.638280 0.638279</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 6.3 Multiscale and variable-coeficient elliptic problems

Case 1: Multiscale Helmholtz equation We consider the Helmholtz equation

$$
- u _ { x x } - u _ { y y } - u = f , \qquad ( x , y ) \in ( 0 , 1 ) ^ { 2 } ,\tag{6.5}
$$

with Dirichlet boundary data prescribed from the exact solution

$$
\begin{array} { r } { u ( x , y ) = \sin ( 3 \pi x ) \sin ( 2 \pi y ) + 0 . 5 \sin ( 2 5 \pi x ) \sin ( 2 2 \pi y ) } \\ { + 0 . 2 \sin ( 5 0 \pi x ) \sin ( 5 7 \pi y ) . \phantom { \sin ( 5 0 \pi x ) \sin ( 5 7 \pi y ) } . } \end{array}\tag{6.6}
$$

This example contains several separated spatial frequencies and therefore tests the ability of the proposed enrichment strategy to resolve strongly oscillatory multiscale residuals. Unlike the two-peak Poisson example, the dominant residual is not concentrated in a small number of localized regions. Consequently, the SFR gate is not activated, and the improvement is mainly achieved by the subsequent RBE block enrichment.

![](images/c6732eb1a760a759ca82794dfa29c3f41433d43df698485cdd21982d8a2d7171.jpg)

![](images/602128e5521d2adb7cb87f4707a577f3da2f446009139332c0f6a7dc68756937.jpg)

![](images/e81d260fa347b5a6f7b46c6b3474290b5d9c106c9c37cc533e8792ee852bc381.jpg)

![](images/37dc9f73f31a3c8cde014617dea382a8a0f66cf0f3be55d475eaa5fae693eb60.jpg)  
Figure 5: Numerical results for the multiscale Helmholtz equation (6.5). From left to right: exact solution, final numerical prediction after RBE enrichment, initial pointwise absolute error, and pointwise absolute error after RBE enrichment.

The SFR gate remains inactive because the residual is globally oscillatory rather than localized. Starting from the 3500-feature AD–RaNN state Rel. $L ^ { 2 } = 2 . 0 6 9 \times 1 0 ^ { - 1 }$ , RBE accepts ten blocks drawn from the families sin, tanh, ridge\_gauss, and mexican\_hat, reducing the relative $L ^ { 2 }$ error to $1 . 6 4 7 \times 1 0 ^ { - 5 }$ . The corresponding reduction in the pointwise error is shown in Figure $5 ;$ the per-block scales are listed in Appendix D.

Table 6: Stage-wise results for the multiscale Helmholtz equation (6.5). The SFR row records the inactive gate and unchanged 3500-feature allocation. “Total features” is cumulative, and the reported times are stage wall-clock values.
<table><tr><td>Stage</td><td>Family</td><td>Features</td><td>Total features</td><td>Scale r</td><td>Time (s)</td><td>Rel.  $L ^ { 2 }$ </td></tr><tr><td>Initial AD-RaNN</td><td>tanh</td><td>3500</td><td>3500</td><td>[63.667,66.557]</td><td>15.303</td><td>2.069e-01</td></tr><tr><td>SFR</td><td>Not triggered</td><td></td><td>3500</td><td></td><td>10.247</td><td>2.069e-01</td></tr><tr><td>RBE</td><td>sin/tanh/ridge_gauss/mexican_hat 5600 (10 blocks)</td><td></td><td>9100</td><td></td><td>501.654</td><td>1.647e-05</td></tr></table>

Case 2: Multiscale Darcy equation We next consider the linear variable-coeficient Darcy problem

$$
\begin{array} { r l } & { - \nabla \cdot ( K _ { \sigma } ( x , y ) \nabla u ) = 1 , \quad ( x , y ) \in ( 0 , 1 ) ^ { 2 } , } \\ & { u ( 0 , y ) = 1 , u ( 1 , y ) = 0 , \quad 0 \leq y \leq 1 , } \\ & { \quad K _ { \sigma } \nabla u \cdot n = 0 , \quad y \in \{ 0 , 1 \} , } \end{array}\tag{6.7}
$$

$$
\begin{array} { l } { { K _ { \sigma } ( x , y ) = \exp \Bigl ( 5 \sin ( 4 \pi x ) \sin ( 4 \pi y ) + 2 . 5 \sin ( 1 0 \pi x ) \sin ( 6 \pi y ) } } \\ { { +  7 \exp [ - \frac { ( x - 0 . 7 ) ^ { 2 } + ( y - 0 . 3 ) ^ { 2 } } { \sigma ^ { 2 } } ] \Bigr ) , \qquad \sigma = 0 . 0 3 . } } \end{array}
$$

This is a linear problem with two oscillatory coeficient scales and a localized, smooth, high-contrast feature. On the evaluation grid, $K _ { \sigma }$ ranges from approximately $9 . 5 \times 1 0 ^ { - 4 }$ to $1 . 1 \times 1 0 ^ { 3 }$ , so the coeficient contrast is about $1 . 1 \times 1 0 ^ { 6 }$ The resulting solution gradient and flux are strongly nonuniform even though the coeficient remains smooth; this makes the case a compact test of regional redistribution and residual-driven enrichment without introducing an interface condition. The reference field is computed by quadratic finite elements with $H _ { \mathrm { m a x } } = 1 0 ^ { - 3 }$ and stored on a $1 0 1 \times 1 0 1$ evaluation grid.

Table 7: Stage-wise results for the Multiscale Darcy problem (6.7). The two SFR rows report the structure- and background-region neuron counts $m _ { S }$ and $m _ { B } ;$ “Total” is the cumulative feature count. Times are stage wall-clock times, and the detailed RBE blocks are listed in Appendix D. The run uses seed 4 and times exclude figure generation.
<table><tr><td>Stage</td><td>Family</td><td>Features</td><td>Total</td><td>Scale r</td><td>Time (s)</td><td>Rel.  $L ^ { 2 }$ </td></tr><tr><td>Initial AD-RaNN</td><td>tanh</td><td>3000</td><td>3000</td><td>[18.530, 18.510]</td><td>63.924</td><td>1.239e-01</td></tr><tr><td rowspan="2">SFR</td><td>tanh</td><td> $m s = 1 4 8 2$ </td><td rowspan="2">3000</td><td> $S _ { 1 } : [ 3 4 . 8 7 5 , 3 5 . 5 7 3 ]$   $S _ { 2 } : \left. 3 5 . 5 7 2 , 3 6 . 6 7 1 \right.$ </td><td rowspan="2"></td><td rowspan="2">24.560 8.188e-02</td></tr><tr><td>tanh</td><td> $m _ { B } = 1 5 1 8$ </td><td>[29.131, 22.165]</td></tr><tr><td>RBE</td><td>mexican_hat/ridge_gauss/tanh 4900 (7 blocks) 7900</td><td></td><td></td><td></td><td>141.411</td><td>8.362e-03</td></tr></table>

The initial AD–RaNN approximation has relative $L ^ { 2 }$ error $1 . 2 3 9 \times 1 0 ^ { - 1 }$ . SFR accepts the regional refresh without increasing the 3000-feature budget and reduces this error to $8 . 1 8 8 \times 1 0 ^ { - 2 }$ ; seven accepted RBE blocks then reduce it to $8 . 3 6 2 \times 1 0 ^ { - 3 }$ . Figure 6 shows the six saved diagnostics: the reference solution, final prediction, and SFR-anchor distribution, followed by the initial error, post-SFR error, and RBE error.

![](images/d004e0c3a8ae933a1a8b8b1aadbe4f00343eb3919e4d6976a670d8f3f0f4c56c.jpg)

![](images/0c587ad2488a1dfb23f6a8a8a44963aac04f21c980bbe8aa56d12d01c80d8a1a.jpg)

![](images/30de638fed80a65892f9fcd9c6ce93e25a6e21f8430aec6745656013b421b000.jpg)

![](images/c906d6819193d98565bdb76864238eeb17f25742d6bead77e1cd1cb0c8c52435.jpg)

![](images/fbc2f9b0e644384d697f2dbb938fbe00278f5019e3f95e83119dc30a0acc82f7.jpg)

![](images/e51fe8fb62185cc984bb041e0a642a682b6ff17cbddee04e446a5cc4b90a0ddb.jpg)  
Figure 6: Multiscale Darcy results for (6.7). The first row shows the reference solution, the final numerical solution after RBE enrichment, and the SFR feature-anchor distribution. The second row shows the initial pointwise absolute error, the pointwise absolute error after SFR, and the pointwise absolute error after RBE.

## 6.4 Nonlinear space–time problems

Case 1: Generalized KdV localized traveling wave We consider the generalized KdV equation in the space–time domain:

$$
u _ { t } + u _ { x } + u ^ { 3 } u _ { x } + \varepsilon u _ { x x x } = 0 , \qquad ( x , t ) \in ( - 2 , 3 ) \times ( 0 , 2 ) ,\tag{6.8}
$$

where $\varepsilon = 0 . 2 0 5 8 \times 1 0 ^ { - 4 }$ . The exact solution and its parameters are written compactly as

$$
\begin{array} { l } { { \displaystyle \vphantom { \bigg ( } u _ { \mathrm { e x } } ( x , t ) = A \mathrm { s e c h } ^ { 2 / 3 } ( K ( x - x _ { 0 } ) - \omega t ) , } } \\ { { \displaystyle A = 0 . 2 2 7 5 , \qquad K = 3 \left( \frac { A ^ { 3 } } { 4 0 \varepsilon } \right) ^ { 1 / 2 } , \qquad \omega = K \left( 1 + \frac { A ^ { 3 } } { 1 0 } \right) , \qquad x _ { 0 } = 0 . 5 . } } \end{array}\tag{6.9}
$$

The initial and boundary conditions are prescribed from the exact solution:

$$
\begin{array} { c c } { { u ( x , 0 ) = u _ { \mathrm { e x } } ( x , 0 ) , } } & { { u ( - 2 , t ) = u _ { \mathrm { e x } } ( - 2 , t ) , } } \\ { { } } & { { u _ { x } ( 3 , t ) = \partial _ { x } u _ { \mathrm { e x } } ( 3 , t ) , } } & { { u _ { x x } ( 3 , t ) = \partial _ { x x } u _ { \mathrm { e x } } ( 3 , t ) . } } \end{array}\tag{6.10}
$$

This nonlinear example contains a moving localized wave structure. The reported solver uses a fixed-operator Picard iteration:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { f i x } } v = v _ { t } + v _ { x } + \varepsilon v _ { x x x } , \qquad \mathcal { L } _ { \mathrm { f i x } } u ^ { ( k + 1 ) } = - ( u ^ { ( k ) } ) ^ { 3 } u _ { x } ^ { ( k ) } . } \end{array}\tag{6.11}
$$

In the reported run, AD–RaNN updates the initial scale from [10, 10] to [10.199, 10.194], SFR reallocates the fixed 2000-feature budget as $m _ { S } = 9 9 5$ and $m _ { B } = 1 0 0 5$ , and the accepted residual-aligned RBE blocks are then appended using the same fixed operator.

![](images/8d085d13c447991ac0edf9378d77c790a5534001f34d292ded8abc355c968db3.jpg)

![](images/567f65e93f2d72a4c9e7a61f93bccbc120600139f4b64a7d942c791266710034.jpg)

![](images/85128d1a0f54be50e497f20de9c5259cb1529d19574cae3a6eb68d88361ac376.jpg)  
Figure 7: Generalized KdV equation (6.8). Exact solution (left), final numerical prediction after RBE enrichment (middle), and SFR feature-anchor distribution (right). The SFR feature anchors are concentrated around the moving localized wave structure.

The fixed-operator initial solve has relative $L ^ { 2 }$ error $6 . 7 7 8 \times 1 0 ^ { - 3 }$ . SFR reduces it to $1 . 0 3 9 \times 1 0 ^ { - 5 }$ while retaining 2000 features. Two accepted RBE blocks then add 400 residual-aligned features from tanh and ridge\_gauss, giving 2400 total features and a final relative $L ^ { 2 }$ error of $7 . 4 2 3 \times 1 0 ^ { - 7 }$

Case 2: Periodic KdV two-soliton interaction We further consider the periodic KdV equation with doublesoliton interaction:

$$
u _ { t } + u u _ { x } + \varepsilon u _ { x x x } = 0 , \qquad ( x , t ) \in ( 0 , 2 ) \times ( 0 , 2 ) ,\tag{6.12}
$$

where $\varepsilon = 4 . 8 4 0 0 \times 1 0 ^ { - 4 }$ . The periodic boundary conditions $u ( 0 , t ) = u ( 2 , t ) , u _ { x } ( 0 , t ) = u _ { x } ( 2 , t )$ , and $u _ { x x } ( 0 , t ) =$ $u _ { x x } ( 2 , t )$ are imposed. The two-pulse initial state is

$$
u ( x , 0 ) = 3 c _ { 1 } \mathrm { s e c h } ^ { 2 } ( k _ { 1 } ( x - x _ { 1 } ) ) + 3 c _ { 2 } \mathrm { s e c h } ^ { 2 } ( k _ { 2 } ( x - x _ { 2 } ) ) ,\tag{6.13}
$$

with $c _ { 1 } = 0 . 3 , c _ { 2 } = 0 . 1 , x _ { 1 } = 0 . 4 , x _ { 2 } = 0 . 8 , \varepsilon = 4 . 8 4 \times 1 0 ^ { - 4 }$ , and $\begin{array} { r } { k _ { i } = \frac { 1 } { 2 } \sqrt { c _ { i } / \varepsilon } . } \end{array}$

Table 8: Stage-wise results for the fixed-operator generalized KdV equation (6.8). The SFR rows report two structure components and the background, with m denoting the total features allocated to the structure components. “Total” is the cumulative feature count. Times are stage wall-clock times; detailed RBE blocks are listed in Appendix D.
<table><tr><td>Stage</td><td>Family</td><td>Features</td><td>Total</td><td>Scale r</td><td>Time (s)</td><td> $\mathrm { R e l . } ~ L ^ { 2 }$ </td></tr><tr><td>Initial AD-RaNN</td><td>tanh</td><td>2000</td><td>2000</td><td>[10.199,10.194]</td><td>34.005</td><td>6.778e-03</td></tr><tr><td>SFR</td><td>tanh tanh</td><td> $m _ { S } = 9 9 5$   $m _ { B } = 1 0 0 5$ </td><td>2000</td><td> $S _ { 1 } : [ 1 5 . 5 2 1 , 1 3 . 1 1 2 ]$   $S _ { 2 } : [ 1 3 . 1 1 3 , 1 1 . 2 9 0 ]$  [13.482, 9.377]</td><td>20.495</td><td>1.039e-05</td></tr><tr><td>RBE</td><td>tanh/ridge_gauss 400 (2 blocks)</td><td></td><td>2400</td><td></td><td>70.834</td><td>7.423e-07</td></tr></table>

The equation solve uses damped Newton iterations. At the current iterate $u ^ { ( k ) }$

$$
\begin{array} { r } { \mathcal { T } _ { k } [ \delta u ] = \delta u _ { t } + u ^ { ( k ) } \delta u _ { x } + u _ { x } ^ { ( k ) } \delta u + \varepsilon \delta u _ { x x x } , \qquad } \\ { \delta c ^ { ( k ) } = \arg \operatorname* { m i n } _ { \delta c } \| J _ { k } \delta c + R _ { k } \| _ { 2 } ^ { 2 } , \qquad c ^ { ( k + 1 ) } = c ^ { ( k ) } + \alpha _ { k } \delta c ^ { ( k ) } . } \end{array}\tag{6.14}
$$

Here $\alpha _ { k } \in \{ 1 , 2 ^ { - 1 } , 2 ^ { - 2 } , . . . \}$ is selected by backtracking to ensure a non-increasing nonlinear residual. The same refinement is applied after the initial solve, the SFR update, and each accepted RBE block.

Unlike the preceding benchmarks, this periodic KdV problem has no closed-form reference solution, so errors are measured against a high-accuracy Fourier spectral solution. The moving solitons and interaction region challenge globally sampled random features.

![](images/9e055418df533945518ce67c867c523dcc0980d428c863a2036a3514dd03b036.jpg)

![](images/278e1dc973eb0e5ba76dd65822aab1e74299bf7e803fdddc925e3399d4829cd3.jpg)

![](images/4157524edf0e89a05db4528ae385101107a75ce836ccc3e08bfca2c5f1a38a54.jpg)  
Figure 8: Periodic KdV two-soliton interaction problem (6.12). Left: high-accuracy Fourier spectral reference solution. Middle: final numerical prediction after RBE enrichment. Right: SFR feature-anchor distribution.

To assess seed sensitivity, we repeat the same solver for seeds $0 , \ldots , 9$ without saving figures. The representative seed-4 run uses damped Newton refinement, changes the $\mathrm { A D - R a N N }$ scale from [10, 10] to [14.8043, 9.5169], allocates the fixed 2000-feature SFR budget as $m _ { S } = 9 8 4$ and $m _ { B } = 1 0 1 6$ , and accepts three RBE blocks. Across the ten seeds, the relative $L ^ { 2 }$ error decreases from $2 . 9 7 6 \times 1 0 ^ { - 1 }$ (best $1 . 7 8 8 \times 1 0 ^ { - 1 }$ , worst $3 . 8 2 2 \times 1 0 ^ { - 1 } )$ initially to $4 . 2 2 5 \times 1 0 ^ { - 3 }$ after SFR (best $2 . 2 1 9 \times 1 0 ^ { - 4 }$ , worst $1 . 4 3 5 \times 1 0 ^ { - 2 } )$ and to $9 . 0 6 1 \times 1 0 ^ { - 5 }$ after RBE (best $6 . 6 4 8 \times 1 0 ^ { - 5 }$ worst $1 . 2 6 2 \times 1 0 ^ { - 4 } )$ . The accepted block count ranges from two to five (mean 3.7), with a mean final width of 3130 features. The representative seed-4 blocks and their scales are listed in Appendix D.

Case 3: Burgers equation We finally consider the nonlinear Burgers equation on the three-dimensional space–time domain:

$$
u _ { t } + u ( u _ { x } + u _ { y } ) - \varepsilon ( u _ { x x } + u _ { y y } ) = 0 , \qquad ( x , y , t ) \in ( 0 , 1 ) ^ { 2 } \times ( 0 , 1 ) , \qquad \varepsilon = 3 \times 1 0 ^ { - 2 } .\tag{6.15}
$$

Table 9: Stage-wise numerical results for the periodic KdV two-soliton problem (6.12) over seeds $0 , \ldots , 9 .$ The SFR entry also reports the mean allocations to the structure and background regions; the RBE entry reports the mean cumulative feature count, with the mean number of added features and accepted blocks given in parentheses. Times are mean stage wall-clock times, and the values after ± denote one standard deviation over the ten seeds.
<table><tr><td>Stage</td><td>Features (avg.)</td><td> ${ \mathrm { T i m e ~ } } ( \mathrm { s } , \mathrm { a v g . } )$ </td><td> $\mathrm { R e l . } \ L ^ { 2 } \ \mathrm { ( a v g . ) }$ </td><td> ${ \mathrm { R e l . ~ } } L ^ { 2 } \ ( \mathrm { b e s t } )$ </td><td> $\mathrm { R e l . } \ L ^ { 2 } \ \mathrm { ( w o r s t ) }$ </td></tr><tr><td>Initial AD-RaNN</td><td>2000</td><td>87.203</td><td> $2 . 9 7 6 \mathrm { { e } - 0 1 \pm 6 . 3 5 4 \mathrm { { e } - 0 2 } }$ </td><td> $1 . 7 8 8 \mathrm { e } { - 0 1 }$ </td><td> $3 . 8 2 2 \mathrm { e } { - 0 1 }$ </td></tr><tr><td>SFR</td><td>2000  $( m _ { S } = 9 9 5 . 4 , ~ m _ { B } = 1 0 0 4 . 6 )$ </td><td>96.761</td><td> $4 . 2 2 5 \mathrm { { e } - 0 3 \pm 4 . 8 0 7 \mathrm { { e } - 0 3 } }$ </td><td> $2 . 2 1 9 \mathrm { e } { - 0 4 }$ </td><td> $1 . 4 3 5 \mathrm { e } { - 0 2 }$ </td></tr><tr><td>RBE</td><td>3130 (1130 added, 3.7 blocks)</td><td>579.581</td><td> $9 . 0 6 1 \mathrm { e } \mathrm { - } 0 5 \pm 2 . 3 3 3 \mathrm { e - } 0 5$ </td><td>6.648e-05</td><td>1.262e-04</td></tr></table>

The initial and boundary data are prescribed from $\begin{array} { r } { u _ { \mathrm { e x } } ( x , y , t ) = \frac { 1 } { 1 + \exp ( ( x + y - t ) / ( 2 \varepsilon ) ) } , } \end{array}$

We use the stabilized fixed-operator iteration with the nonlinear convection term lagged from the previous iterate; the global marked set is used as one SFR structure mask. The relative $L ^ { 2 }$ error decreases from $8 . 2 0 2 \times 1 0 ^ { - 3 }$

![](images/f46bb8497a1f1558bbf287487912a37084f15219397aad49fd91dbe0fe1fa254.jpg)

![](images/bce7534d0d71d0526e7984724b4138ffbf9e9f5eb8bceeb768e546d54ceb2f1e.jpg)

![](images/90a41cf2ead7fac472a1ff9718a1f38dc2d6af3e98035c05914484844f15a14b.jpg)

![](images/3a0950822d30926452453964f9fa75775f34c1de82f152cf2f9fc4f630277c31.jpg)  
Figure 9: Space–time visualization for the Burgers problem (6.15): exact solution, final prediction after RBE enrichment, pointwise absolute error, and the SFR feature-anchor distribution.

for the initial AD–RaNN space to $1 . 6 2 2 \times 1 0 ^ { - 4 }$ after SFR and to $2 . 2 7 1 \times 1 0 ^ { - 5 }$ after four accepted RBE blocks. The accepted blocks use the families tanh, $\mathtt { r i d g e \_ g a u s s } _ { \mathtt { d a s s } _ { \mathtt { d } } }$ , tanh, and $\mathtt { r i d g e \_ g a u s s } _ { \mathtt { d a s s } _ { \mathtt { d } } }$ , adding 1000 features in total.

Table 10: Stage-wise results for the Burgers problem (6.15). The two SFR rows report the structure- and backgroundregion neuron counts $m _ { S }$ and $m _ { B } ; \mathrm { { ^ { 6 6 } T o t a l } ^ { 3 } }$ is the cumulative feature count. Times are stage wall-clock values and exclude figure generation.
<table><tr><td>Stage</td><td>Family</td><td>Features</td><td>Total</td><td>Scale r</td><td>Time (s)</td><td> $\mathrm { R e l } , \ L ^ { 2 }$ </td></tr><tr><td>Initial AD-RaNN</td><td>tanh</td><td>3000</td><td>3000</td><td>[10.864,9.549,7.862]</td><td>182.302</td><td>8.202e-03</td></tr><tr><td>SFR</td><td>tanh tanh</td><td> $m _ { S } = 1 4 9 9$   $m _ { B } = 1 5 0 1$ </td><td>3000</td><td>[12.948, 12.071, 11.089] [8.500, 8.986, 8.639]</td><td></td><td>160.578 1.622e-04</td></tr><tr><td>RBE</td><td>tanh/ridge_gauss 1000 (4 blocks)</td><td></td><td>4000</td><td></td><td>564.769</td><td>2.271e-05</td></tr></table>

## 7 Conclusion

This paper introduced SFRBE–RaNN, an adaptive randomized neural PDE solver that preserves the outputlinear least-squares structure while combining structure-aware feature redistribution with residual-driven block enrichment. SFR reallocates a fixed feature budget through adaptive anchors, regional neuron counts, and local scales, whereas RBE selects activation families and nonredundant feature blocks according to their residual-capture ability. Numerical experiments on localized, multiscale, variable-coeficient elliptic, and nonlinear space–time problems demonstrate the complementary efects of the two stages, while the analysis establishes exact fixed-linear residual reduction, a conditional finite-horizon RBE success-probability estimate and a safeguarded baseline-to-final error bound, together with a separate nonlinear acceptance estimate. Future work will develop eficient large-scale least-squares solvers for growing randomized feature systems, extend the framework to operator learning and parametric or multiphysics problems, and establish uniform candidate-coverage and finite-sample consistency bounds for fully data-dependent trial-space adaptation.

## References

[1] Mark Ainsworth and Justin Dong. Galerkin neural networks: A framework for approximating variational equations with error control. SIAM Journal on Scientific Computing, 43(4):A2474–A2501, 2021.

[2] Jingrun Chen, Xurong Chi, Weinan E, and Zhouwang Yang. Bridging Traditional and Machine Learning-Based Algorithms for Solving PDEs: The Random Feature Method. Journal of Machine Learning, 1(3):268–298, 2022.

[3] Long Chen. iFEM: an innovative finite element methods package in MATLAB. Preprint, University of Maryland, 2008.

[4] Madison Cooley, Mike Kirby, Shandian Zhe, and Varun Shankar. HyResPINNs: A Hybrid Residual Physics Informed Neural Network Architecture Designed to Balance Expressiveness and Trainability. Transactions on Machine Learning Research, 2025.

[5] Haoning Dang, Fei Wang, and Song Jiang. Adaptive growing randomized neural networks for solving partial diferential equations. arXiv preprint arXiv:2408.17225, 2024.

[6] Suchuan Dong and Zongwei Li. Local extreme learning machines and domain decomposition for solving linear and nonlinear partial diferential equations. Computer Methods in Applied Mechanics and Engineering, 387:114129, 2021.

[7] Suchuan Dong and Jielin Yang. On computing the hyperparameter of extreme learning machines: Algorithm and application to computational PDEs, and comparison with classical and high-order finite elements. Journal of Computational Physics, 463:111290, 2022.

[8] Willy Dörfler. A Convergent Adaptive Algorithm for Poisson’s Equation. SIAM Journal on Numerical Analysis, 33(3):1106–1124, 1996.

[9] Vikas Dwivedi, Balaji Srinivasan, Monica Sigovan, and Bruno Sixou. Kernel-Adaptive PI-ELMs for Forward and Inverse Problems in PDEs with Sharp Gradients. arXiv preprint arXiv:2507.10241, 2025.

[10] Weinan E and Bing Yu. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1):1–12, 2018.

[11] Zhiwei Gao, Liang Yan, and Tao Zhou. Failure-informed adaptive sampling for PINNs. SIAM Journal on Scientific Computing, 45(4):A1971–A1994, 2023.

[12] Ameya D Jagtap, Kenji Kawaguchi, and George Em Karniadakis. Adaptive activation functions accelerate convergence in deep and physics-informed neural networks. Journal of Computational Physics, 404:109136, 2020.

[13] Yunlong Li and Fei Wang. Local randomized neural networks with finite diference methods for interface problems. Journal of Computational Physics, 529:113847, 2025.

[14] Yunlong Li and Fei Wang. Structure-Oriented Randomized Neural Networks for Poisson–Nernst–Planck and Poisson–Nernst–Planck–Navier–Stokes Systems. arXiv preprint arXiv:2606.19912, 2026.

[15] Levi D McClenny and Ulisses M Braga-Neto. Self-adaptive physics-informed neural networks. Journal of Computational Physics, 474:111722, 2023.

[16] Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019.

[17] Yong Shang and Fei Wang. Randomized neural networks with Petrov–Galerkin methods for solving linear elasticity and Navier–Stokes equations. Journal of Engineering Mechanics, 150(4):04024010, 2024.

[18] Yong Shang, Fei Wang, and Jingbo Sun. Randomized neural network with Petrov–Galerkin methods for solving linear and nonlinear partial diferential equations. Communications in Nonlinear Science and Numerical Simulation, 127:107518, 2023.

[19] Justin Sirignano and Konstantinos Spiliopoulos. DGM: A deep learning algorithm for solving partial diferential equations. Journal of Computational Physics, 375:1339–1364, 2018.

[20] Jingbo Sun, Suchuan Dong, and Fei Wang. Local randomized neural networks with discontinuous Galerkin methods for partial diferential equations. Journal of Computational and Applied Mathematics, 445:115830, 2024.

[21] Jingbo Sun and Fei Wang. Local randomized neural networks with discontinuous Galerkin methods for difusive-viscous wave equation. Computers & Mathematics with Applications, 154:128–137, 2024.

[22] Rüdiger Verfürth. A posteriori error estimation techniques for finite element methods. OUP Oxford, 2013.

[23] Haixin Wang, Haoning Dang, Fei Wang, and Shimin Guo. Domain-decomposed randomized neural networks for partial diferential equations in unbounded domains. arXiv preprint arXiv:2606.31342, 2026.

[24] Sifan Wang, Shyam Sankaran, Hanwen Wang, and Paris Perdikaris. An expert’s guide to training physicsinformed neural networks. arXiv preprint arXiv:2308.08468, 2023.

[25] Mingxing Weng, Zhiping Mao, and Jie Shen. Deep Collocation Method: A Framework for Solving PDEs Using Neural Networks with Error Control. SIAM Journal on Scientific Computing, 48(1):C77–C102, 2026.

[26] Chenxi Wu, Min Zhu, Qinyang Tan, Yadhu Kartha, and Lu Lu. A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. Computer Methods in Applied Mechanics and Engineering, 403:115671, 2023.

[27] Jinchao Xu and Xiaofeng Xu. Randomized greedy algorithms for neural network optimization in solving partial diferential equations. Journal of Scientific Computing, 105(1):26, 2025.

[28] You Yang and Fei Wang. Adaptive-Distribution Randomized Neural Networks for PDEs: A Low-Dimensional Distribution-Learning Framework. arXiv preprint arXiv:2604.23999, 2026.

## A Proof of the Poisson residual-to-error estimate

Proof of the Poisson stability estimate. Write $v - u ^ { * } = w + z$ , where $- \Delta w = R ( v ) , w | _ { \partial \Omega } = 0$ , and z is harmonic with trace $b ( v )$ . Since the first Dirichlet eigenvalue of the unit square is $2 \pi ^ { 2 }$

$$
\lVert w \rVert _ { L ^ { 2 } ( \Omega ) } \leq \frac { \lVert R ( v ) \rVert _ { L ^ { 2 } ( \Omega ) } } { 2 \pi ^ { 2 } } .
$$

Let $- \Delta \tau = 1$ in Ω with $\tau | _ { \partial \Omega } = 0$ . The maximum principle and comparison give $0 \leq - \partial _ { n } \tau \leq 1 / 2$ on $\partial \Omega$ . Green’s identity then yields

$$
\| z \| _ { L ^ { 2 } ( \Omega ) } ^ { 2 } \le - \int _ { \partial \Omega } b ( v ) ^ { 2 } \partial _ { n } \tau \le \frac { 1 } { 2 } \| b ( v ) \| _ { L ^ { 2 } ( \partial \Omega ) } ^ { 2 } ,
$$

and hence $\| z \| _ { L ^ { 2 } ( \Omega ) } \leq \frac { \| b ( v ) \| _ { L ^ { 2 } ( \partial \Omega ) } } { \sqrt { 2 } }$ . The triangle inequality now gives (6.4).

## B Optional family-parameter refinement for SFRBE–RaNN

The scale and family parameters in SFRBE–RaNN can be selected by several admissible mechanisms, including frequency-informed rules, bounded evolutionary searches such as diferential evolution, and low-dimensional adaptive distribution optimization [5, 7, 28]. In this paper, both SFR and RBE use the AD–RaNN-style low-dimensional search, in which Adam updates the parameter vector $\vartheta _ { F } \in \Theta _ { F }$ while the sampled anchors and directions remain fixed. Here $\Theta _ { F }$ denotes the admissible parameter set for family $F$ used in the bounded parameter search. For a candidate family $F _ { \mathrm { { ; } } }$ let

$$
C _ { F , \perp } ^ { \mathrm { s c r } } ( \vartheta _ { F } ) = ( I - Q _ { L } Q _ { L } ^ { \mathsf { T } } ) C _ { F } ^ { \mathrm { s c r } } ( \vartheta _ { F } ) .\tag{B.1}
$$

Starting from the initial value $\vartheta _ { F } ^ { ( 0 ) }$ , Adam minimizes the ridge-regularized reduced objective

$$
\mathcal { I } _ { F , L } ^ { \mathrm { r e d } , \lambda } \big ( \vartheta _ { F } \big ) = \operatorname* { m i n } _ { a } \left\{ \big \| \boldsymbol { C } _ { F , \bot } ^ { \mathrm { s c r } } ( \vartheta _ { F } ) { \boldsymbol { a } } - \boldsymbol { r } _ { L } \big \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { s c a l e } } \big \| { \boldsymbol { a } } \big \| _ { 2 } ^ { 2 } \right\} , \qquad \vartheta _ { F } \in \Theta _ { F } ,\tag{B.2}
$$

for at most $T _ { \mathrm { t u n e } }$ bounded updates. The same regularization parameter $\lambda _ { \mathrm { s c a l e } } = 1 0 ^ { - 6 }$ is used in the reduced parameter searches for the initial AD–RaNN construction, SFR, and RBE. The ridge term is used only during this low-dimensional parameter search; after $\vartheta _ { F }$ is selected, the regularization is removed and the output coeficients are recomputed by the unregularized weighted least-squares solve.

## C Additional implementation details

This appendix summarizes the implementation settings used in the numerical experiments. Unless otherwise stated, the output coeficients are computed by unregularized linear or linearized least-squares solves. The regularization parameter $\lambda _ { \mathrm { s c a l e } } = 1 0 ^ { - 6 }$ is used to stabilize the reduced parameter-search objectives in the initial AD–RaNN construction, SFR, and RBE. Tables 11 and 12 summarize problem-dependent settings and shared controls, respectively.

Table 11: Problem-dependent discretization, features, and constraint-weight settings.
<table><tr><td>Problem</td><td colspan="2">Points</td><td>Initial features</td><td>Full block size mfull</td><td>Weights  $( \sqrt { \eta _ { b } } , \sqrt { \eta _ { 0 } } )$ </td></tr><tr><td>Two-peak Poisson</td><td> $N _ { \mathrm { p d e } } = 2 2 5 0 0 , \ N _ { b } = 8 0 0$ </td><td></td><td>3500</td><td>1000</td><td> $( { { 1 0 } ^ { 5 } } , - )$ </td></tr><tr><td>Multiscale Helmholtz</td><td> $N _ { \mathrm { p d e } } = 6 2 5 0 0 , \ N _ { b } = 8 0 0$ </td><td></td><td>3500</td><td>2000</td><td> $( { 1 0 } ^ { 5 } , - )$ </td></tr><tr><td>Multiscale Darcy</td><td> $\dot { N _ { \mathrm { p d e } } } = 2 2 5 0 0 , \ N _ { b } = 8 0 0$ </td><td></td><td>3000</td><td>1000</td><td>(100,−)</td></tr><tr><td>Generalized KdV</td><td></td><td> $N _ { \mathrm { p d e } } = 2 2 5 0 0 , \ N _ { b } = 4 0 0 , \ N _ { 0 } = 2 0 0$ </td><td>2000</td><td>1000</td><td>(100,100)</td></tr><tr><td>Periodic KdV</td><td></td><td> $N _ { \mathrm { p d e } } = 3 2 4 0 0 , \ N _ { b } = 4 0 0 , \ N _ { 0 } = 2 0 0$ </td><td>2000</td><td>1000</td><td>(100, 100)</td></tr><tr><td>Burgers</td><td> $N _ { \mathrm { p d e } } = 3 7 0 2 6 , \ N _ { b } = 4 7 6 0 , \ N _ { 0 } =$ </td><td>1225</td><td>3000</td><td>1000</td><td>(300,300)</td></tr></table>

The prescribed prefix-size set is defined as

$$
\mathcal { P } = \{ \lceil { \alpha m _ { \mathrm { f u l l } } } \rceil : \alpha \in \{ 0 . 2 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 , 0 . 8 , 0 . 9 , 1 . 0 \} \}
$$

. The same sampling-cap rule is applied independently to the structure region, the background region, and the RBE candidate blocks. For low-dimensional parameter optimization, we subsample features and collocation equations

Table 12: Shared SFR/RBE and numerical implementation controls.
<table><tr><td>Module</td><td>Parameters</td><td>Values</td></tr><tr><td>SFR</td><td>θ, Teff, Tconn, Tcover</td><td>0.50, 0.35, 0.01, 0.35</td></tr><tr><td>RBE screening</td><td>Tsvd, Tcap</td><td>10−10,10−4</td></tr><tr><td>RBE screening size</td><td>mscr</td><td>min{500, [0.3mfull]}</td></tr><tr><td>RBE prefix/stop</td><td>θpref , Tstop</td><td>0.80, 0.25</td></tr><tr><td>Stabilization</td><td>En</td><td>10-12</td></tr><tr><td>SFR parameter refinement</td><td> ${ \mathrm { l r } } _ { S } , T _ { S }$ </td><td>(0.7,30)</td></tr><tr><td>Family-parameter search</td><td>lrF , Ttune</td><td>(0.5,30)</td></tr></table>

separately for the SFR structure and background regions and RBE candidate blocks. Let $q \in \{ S , B , R \}$ , with $m _ { q }$ selected features and $N _ { \mathrm { e q } , q }$ available equation rows. The sampling cap is

$$
N _ { q } ^ { \operatorname * { m a x } } = \operatorname * { m i n } \left\{ N _ { \mathrm { e q } , q } , \lceil m _ { q } \log ( 1 + m _ { q } ) \rceil \right\} , \qquad q \in \{ S , B , R \} .\tag{C.1}
$$

## D Stagewise activation-family and scale records

The following per-round records use representative seed-4 runs. Poisson and periodic KdV aggregate statistics remain in the main text.

Table 13: Activation-family choices and residual decrease during RBE enrichment. Each row gives the postacceptance weighted relative residual for the indicated block; the two panels are a compact continuation of one table.
<table><tr><td>Round Family</td><td></td><td>m</td><td>Scale/shape</td><td>Rel. residual</td></tr><tr><td colspan="5">Two-peak Poisson (Seed 4)</td></tr><tr><td></td><td>1 radial_gauss</td><td>200</td><td>[47.872,50.505]</td><td>6.001e-04</td></tr><tr><td></td><td>2 mexican_hat</td><td>200</td><td>[55.822,50.104]</td><td>1.992e-04</td></tr><tr><td></td><td>3 ridge_gauss</td><td>400</td><td>[69.134,66.557]</td><td>9.468e-05</td></tr><tr><td></td><td>4 mexican_hat</td><td>500</td><td>[48.317,57.077]</td><td>5.354e-05</td></tr><tr><td></td><td>5 sin</td><td>600</td><td>[48.802,50.577]</td><td>3.093e-05</td></tr><tr><td></td><td>6 mexican_hat</td><td>500</td><td>[46.264,71.383]</td><td>1.796e-05</td></tr><tr><td></td><td>7 sin</td><td>500</td><td>[51.141,52.627]</td><td>7.823e-06</td></tr><tr><td></td><td>8 ridge_gauss</td><td>400</td><td>[76.559,73.034]</td><td>4.993e-06</td></tr><tr><td colspan="5">Multiscale Helmholtz</td></tr><tr><td></td><td>1 sin</td><td>400</td><td>[61.490,67.019]</td><td>2.507e-01</td></tr><tr><td></td><td>2 tanh</td><td>400</td><td>[67.066,70.204]</td><td>1.381e-01</td></tr><tr><td></td><td>3 sin</td><td>400</td><td>[61.165,67.295]</td><td>6.122e-02</td></tr><tr><td></td><td>4 ridge_gauss</td><td>400</td><td>[80.840,84.406]</td><td>3.355e-02</td></tr><tr><td></td><td>5 sin</td><td>400</td><td>[61.015,66.926]</td><td>9.133e-03</td></tr><tr><td></td><td>6 ridge_gauss</td><td>400</td><td>[82.788,86.523]</td><td>4.532e-03</td></tr><tr><td></td><td>7 sin</td><td>400</td><td>[61.121,68.547]</td><td>7.007e-04</td></tr><tr><td></td><td>8 ridge_gauss</td><td>800</td><td>[82.173,88.652]</td><td>3.576e-04</td></tr><tr><td></td><td>9 sin</td><td>800</td><td>[60.006,68.498]</td><td>1.328e-04</td></tr><tr><td></td><td>10 mexican_hat</td><td>1200</td><td>[5.239,23.696]</td><td>1.072e-04</td></tr></table>

<table><tr><td>Round Family</td><td></td><td>m</td><td>Scale/shape</td><td>Rel. residual</td></tr><tr><td colspan="5">Multiscale Darcy</td></tr><tr><td></td><td>1 mexican_hat</td><td>500</td><td>[35.224,36.122]</td><td>9.792e-05</td></tr><tr><td></td><td>2 ridge_gauss</td><td>700</td><td>[35.261,36.979]</td><td>5.996e-05</td></tr><tr><td></td><td>3 tanh</td><td>500</td><td>[45.750,44.683]</td><td>3.699e-05</td></tr><tr><td></td><td>4 ridge_gauss</td><td>700</td><td>[35.228,35.666]</td><td>2.608e-05</td></tr><tr><td></td><td>5 tanh</td><td>800</td><td>[48.606,43.850]</td><td>1.976e-05</td></tr><tr><td></td><td>6 ridge_gauss</td><td>900</td><td>[33.089,34.127]</td><td>1.587e-05</td></tr><tr><td></td><td>7 tanh</td><td>800</td><td>[53.972,48.430]</td><td>1.307e-05</td></tr><tr><td colspan="5">Generalized KdV (fixed operator)</td></tr><tr><td>1 tanh</td><td></td><td>200</td><td>[15.718,13.432]</td><td>3.964e-07</td></tr><tr><td></td><td>2 ridge_gauss 200</td><td></td><td>[28.334,20.426]</td><td>1.676e-07</td></tr><tr><td colspan="5">Periodic KdV two-soliton (Seed 4)</td></tr><tr><td></td><td>1 ridge_gauss 200</td><td></td><td>[27.148,14.723]</td><td>5.518e-05</td></tr><tr><td></td><td>2 tanh</td><td>200</td><td>[33.628,14.985]</td><td>4.000e-05</td></tr><tr><td></td><td>3 sin</td><td>600</td><td>[23.463,15.269]</td><td>3.486e-05</td></tr><tr><td colspan="5">Burgers</td></tr><tr><td></td><td>1 tanh</td><td>200</td><td>[13.118,15.097,12.596]</td><td>3.540e-05</td></tr><tr><td></td><td>2 ridge_gauss</td><td>400</td><td>[18.520,20.163,12.438]</td><td>2.714e-05</td></tr><tr><td></td><td>3 tanh</td><td>200</td><td>[20.770,24.351,15.088]</td><td>1.963e-05</td></tr><tr><td></td><td>4 ridge_gauss 200</td><td></td><td>[25.983,32.015,19.770]</td><td>1.572e-05</td></tr></table>

## Code availability

The repository address for the MATLAB implementation, experiment configurations, and plotting scripts is https://github.com/XJTU-AI4SciComp-Lab/sfrbe-rann-adaptive-paper-code.