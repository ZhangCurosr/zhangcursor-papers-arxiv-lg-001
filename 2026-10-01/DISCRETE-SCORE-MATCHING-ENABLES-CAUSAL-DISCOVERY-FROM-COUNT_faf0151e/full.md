# DISCRETE SCORE MATCHING ENABLES CAUSAL DISCOVERY FROM COUNT DATA

Euijong Song Hyewon Park Gunwoong Park<sup>∗</sup>

Department of Statistics

Seoul National University, Korea

brejsong@gmail.com hwonpark18@gmail.com gw.park23@gmail.com

## ABSTRACT

Count data pose a challenge for score-matching-based causal discovery: derivatives are unavailable, and simply replacing them with finite differences does not generally suffice for causal discovery. We generalize SCORE’s constant-curvature criterion (Rolland et al., 2022) by conditioning on the node’s value, yielding the conditional curvature score (CCS) for ordering. We also extend curvature-based parent recovery through the off-diagonal curvature score (OCS), enabling directed acyclic graph (DAG) recovery with both scores constructed from score functions for continuous data and concrete scores for counts. In the bivariate setting, zero CCS exactly characterizes a semiparametric generalized linear model (GLM) conditional form in which the conditional family need not be specified in advance, unlike in classical GLMs. For bivariate semiparametric GLM DAGs under our regularity condition, canonical-parameter nonlinearity is necessary and sufficient for identifiability. In multivariate DAGs, this nonlinearity enables DAG recovery through CCS and OCS. Our framework identifies a new class of semiparametric GLM DAGs that strictly contains the nonlinear Gaussian ANM class identified by SCORE. We introduce DISCO (DIscrete SCOre), a count-DAG recovery algorithm that estimates CCS and OCS using discrete diffusion. Experiments demonstrate accurate DAG recovery across Poisson, negative binomial, binomial, and mixed-family settings, as well as scalability to 1,000-node DAGs on a single GPU.

## 1 INTRODUCTION

Many causal-discovery problems are naturally count-valued: gene-regulatory analysis from RNAseq counts (Love et al., 2014), community structure from species abundances (Stoklosa et al., 2022), and topic structure from word counts (Blei et al., 2003). Such counts rarely follow a single distributional family: sequencing studies model expression counts as negative binomial (NB) but allele counts as binomial (Love et al., 2014; Robinson et al., 2010; Castel et al., 2015), and ecological surveys mix Poisson and overdispersed NB (Stoklosa et al., 2022).

Order-based methods provide a scalable alternative to DAG search by estimating a causal order and then selecting parents (Ghoshal and Honorio, 2018; Gao et al., 2020; Peters et al., 2014). Scorematching-based methods follow this approach using log-density derivatives (Rolland et al., 2022; Montagna et al., 2023a;b; Sanchez et al., 2023). Concrete score matching extends score estimation to discrete data (Meng et al., 2022), but finite-difference curvature need not satisfy SCORE’s sink criterion: even at a sink, the node’s conditional density can contribute curvature that varies with its value. We call this contribution own-factor curvature.

Existing count-DAG methods obtain identifiability by exploiting properties of specified conditional families, such as Poisson or known quadratic variance function (QVF) families (Park and Raskutti, 2015; 2018). More flexible conditionally parametric models allow general parent effects but still specify the family itself (Bodik and Chavez-Demoulin, 2025). Their guarantees consequently depend on choosing appropriate conditional families. Extending score-matching-based causal discovery to counts without prespecified families therefore requires an ordering criterion that accommodates unknown own-factor curvature.

We address these limitations by generalizing SCORE’s constant-curvature criterion through the CCS and extending curvature-based parent recovery through the OCS. Our conditional curvature score (CCS) removes the own-factor contribution to curvature variance while retaining variation induced by children. The CCS is obtained from derivatives of the continuous score or finite differences of log concrete-score entries.

We establish identifiability for a new class of semiparametric GLM DAGs. In the bivariate setting, zero CCS exactly characterizes a semiparametric GLM conditional form without prespecifying the conditional family. Within this class, canonical-parameter nonlinearity is necessary and sufficient for identifiability. In multivariate settings, this nonlinearity enables the CCS to recover a causal order, after which the off-diagonal curvature score (OCS) recovers parents. On continuous data, the identifiable class strictly contains SCORE’s nonlinear Gaussian additive noise models (ANMs).

We introduce DISCO (DIscrete SCOre), a scalable count-DAG recovery algorithm that estimates the CCS and OCS. We first estimate joint concrete scores using score-entropy discrete diffusion, then project them to the marginal concrete scores needed to compute the CCS and OCS. With OCSbased parent selection, a rank transform makes the recovered DAG invariant to strictly increasing coordinate-wise transformations.

Our contributions are as follows.

• Discrete score matching for count-DAG discovery. We generalize SCORE’s constantcurvature principle to count data through the conditional curvature score (CCS), which identifies sinks despite node-specific own-factor curvature. We also extend curvature-based parent recovery through the OCS.

• A new identifiable class of semiparametric GLM DAGs. We derive the semiparametric GLM conditional form from the bivariate zero-CCS condition without prespecifying the family. Canonical-parameter nonlinearity is necessary and sufficient for bivariate identifiability. This nonlinearity also yields multivariate DAG identifiability. In the continuous setting, the resulting identifiable class strictly contains the nonlinear Gaussian ANM class identified by SCORE.

• Scalable count-DAG recovery with discrete diffusion. We develop DISCO, which uses score-entropy discrete diffusion to estimate joint concrete scores and projects them to the marginal concrete scores needed to compute the CCS and OCS. DISCO scales to 1,000- node count DAGs on a single GPU.

## 2 BACKGROUND

## 2.1 RELATED WORK

Score-matching-based causal discovery. For a continuous density $p ,$ the score function is $s ( x ) =$ $\nabla _ { x } \log p ( x )$ (Hyvarinen, 2005). SCORE identifies sinks in nonlinear Gaussian ANMs through the¨ constant-curvature criterion, $\mathrm { V a r } [ \partial _ { j } s _ { j } ( X ) ] = 0$ (Rolland et al., 2022). Subsequent work allows non-Gaussian noise (NoGAM; Montagna et al., 2023b), improves Hessian estimation (DiffAN and SciNO; Sanchez et al., 2023; Kang et al., 2025), and uses off-diagonal Hessian components for parent recovery (DAS; Montagna et al., 2023a).

For discrete causal discovery, Vo et al. (2026) use a marginalization-based generalized score (Lyu, 2009) to recover a causal order under a non-decreasing conditional-randomness assumption. In contrast, we use concrete scores (Meng et al., 2022), estimated through discrete diffusion with SEDD’s score-entropy objective (Lou et al., 2024), and establish DAG identifiability within our semiparametric GLM class under the stated nonlinearity and regularity conditions, without requiring a cross-node ordering of conditional randomness.

Count-DAG identifiability. Count-DAG identifiability typically relies on distribution-specific conditional structure: Poisson and QVF DAGs assume known conditional variance structure (Park and Raskutti, 2015; Park and Park, 2019b; Park and Raskutti, 2018), while conditionally parametric causal models still prespecify the conditional family (Bodik and Chavez-Demoulin, 2025). Our semiparametric GLM DAG class leaves conditional families, links, and canonical-parameter functions unspecified.

## 2.2 NOTATION

Let $G = ( V , E )$ be a directed acyclic graph with $V = [ d ] : = \{ 1 , \ldots , d \}$ . Let $\mathrm { P a } _ { G } ( j )$ and $\operatorname { C h } _ { G } ( j )$ denote the parents and children of node j in G. A source is a node $j$ with $\operatorname { P a } _ { G } ( j ) = { \overset { \cdot } { \varnothing } }$ , a non-source is a node j with $\mathrm { P a } _ { G } ( j ) \neq \varnothing$ , and a sink is a node $j$ with ${ \mathrm { C h } } _ { G } ( j ) = \varnothing$ . For $R \subseteq V , G _ { R }$ denotes the induced subgraph of G on R. A causal order is a permutation $\pi : V \to [ d ]$ with $\pi ( j ) < \pi ( i )$ for every edge $j  i ;$ we also call it an ordering of $G$

Let $P$ be the distribution of $X = ( X _ { 1 } , \ldots , X _ { d } )$ . For each $j \in V ,$ , the coordinate $X _ { j }$ takes values in $\mathcal { X } _ { j } \subseteq \mathbb { R }$ and has a dominating measure $\nu _ { j }$ (counting or Lebesgue). With $\nu = \otimes _ { j = 1 } ^ { d } \nu _ { j }$ , write $p = d P / d \nu$ for the density; throughout, this term includes probability mass functions. The set $\mathcal { X } _ { j }$ is the declared effective support. We call $X _ { j }$ a count coordinate when $\chi _ { j }$ is a finite or countably infinite set of consecutive integers. Define $\mathring { \mathcal { X } } _ { j } = \operatorname { i n t } ( \mathcal { X } _ { j } )$ for a continuous coordinate and $\mathring { \mathcal { X } } _ { j } = \mathcal { X } _ { j }$ for a count coordinate.

For $A \subseteq V$ , write $\begin{array} { r } { X _ { A } = ( X _ { i } ) _ { i \in A } , \mathcal { X } _ { A } = \prod _ { i \in A } \mathcal { X } _ { i } } \end{array}$ , and $\begin{array} { r } { \mathring { \mathcal { X } } _ { A } = \prod _ { i \in { \cal A } } \mathring { \mathcal { X } } _ { i } } \end{array}$ . For $j \in A , x _ { A } \in \mathcal { X } _ { A }$ and $y \in { \mathcal { X } } _ { j }$ , let $x _ { A } ^ { j  y }$ be the vector obtained by replacing coordinate $j$ with $y .$ . For $R \subseteq V$ , let $P _ { R }$ denote the marginal distribution of $X _ { R }$ and $p _ { R }$ its density.

We use a single coordinate operator $\mathcal { D } _ { j }$ for both continuous and count coordinates. With $e _ { j }$ the jth unit vector,

$$
\bigl ( \mathcal D _ { j } f , \mathcal D _ { j } ^ { 2 } f \bigr ) ( x ) = \left\{ \begin{array} { l l } { \bigl ( \partial _ { x _ { j } } f ( x ) , \partial _ { x _ { j } } ^ { 2 } f ( x ) \bigr ) , } & { X _ { j } \mathrm { c o n t i n u o u s } , } \\ { \bigl ( f ( x + e _ { j } ) - f ( x ) , f ( x + 2 e _ { j } ) - 2 f ( x + e _ { j } ) + f ( x ) \bigr ) , } & { X _ { j } \mathrm { c o u n t } . } \end{array} \right.
$$

On a count coordinate these are partial operators: every expectation or conditional variance involving a finite-difference curvature is restricted to states where all required shifted values remain in the support. Appendix A gives the formal domains.

## 3 CURVATURE SCORES

We introduce two curvature scores: the conditional curvature score (CCS) for ordering and the off-diagonal curvature score (OCS) for parent recovery. Define the diagonal and off-diagonal logdensity curvatures $H _ { j } : = \mathcal { D } _ { i } ^ { 2 }$ log p and $\bar { H } _ { i j } : = \mathcal { D } _ { i } \mathcal { D } _ { j }$ log p for $i \neq j$ . For the marginal distribution $P _ { R } ,$ , we denote the corresponding curvatures by $H _ { j , R }$ and $H _ { i j , R }$

For node $j ,$ we ask whether its log-density curvature still varies with the other variables after fixing $X _ { j }$ . The conditional curvature score measures this remaining variation.

Definition 1 (Conditional curvature score). For node j with $H _ { j } \in L ^ { 2 }$

$$
S _ { j } = \mathbb { E } \big [ \mathrm { V a r } \{ H _ { j } ( X ) \mid X _ { j } \} \big ] .\tag{1}
$$

For the marginal distribution $P _ { R }$ , write $S _ { j } ( R ) = \mathbb { E } _ { P _ { R } } [ \operatorname { V a r } _ { P _ { R } } \{ H _ { j , R } ( X _ { R } ) \mid X _ { j } \} ]$ , so $S _ { j } = S _ { j } ( V )$ with the same admissible-domain convention. The CCS vanishes exactly when $H _ { j } ( { \bar { X } } )$ is almost surely a function of $X _ { j }$ alone. By contrast, the constant-curvature criterion requires $H _ { j } ( X )$ to be almost surely constant, equivalently $\operatorname { V a r } [ H _ { j } ( X ) ] = 0$ . For continuous data, $H _ { j } = \partial _ { j } s _ { j }$ is a diagonal component of the log-density Hessian, and this is SCORE’s sink criterion. The CCS therefore allows curvature to vary across values of $X _ { j }$ . Section 5 establishes when the CCS separates sinks from nonsinks.

To define the OCS, fix a causal order π of G. For each node $i ,$ define its predecessors by ${ \mathrm { P r e d } } ( i ) =$ $\{ j : \pi ( j ) < \pi ( i ) \}$ } and let $R _ { i } = \{ i \} \cup \mathrm { P r e d } ( i )$ . The OCS is computed under the marginal distribution $P _ { R _ { i } }$

Definition 2 (Off-diagonal curvature score). For a node i and a predecessor $j \in \mathrm { P r e d } ( i )$ with $H _ { i j , R _ { i } } \in L ^ { 1 }$

$$
\begin{array} { r } { \mathrm { O C S } _ { j  i } = \mathbb { E } \big [ | H _ { i j , R _ { i } } ( X _ { R _ { i } } ) | \big ] . } \end{array}\tag{2}
$$

For count coordinates, $H _ { i j , R _ { i } }$ records how the log concrete-score entry for a shift in coordinate $j$ changes when $X _ { i }$ steps by one level. The OCS averages its absolute magnitude under $P _ { R _ { i } }$

## 4 SEMIPARAMETRIC GLM DAGS

A classical generalized linear model (GLM) specifies a conditional family and a link function that maps the conditional mean to a linear predictor (Nelder and Wedderburn, 1972; McCullagh and Nelder, 1989). Semiparametric GLMs relax the requirement to prespecify the conditional family (Rathouz and Gao, 2009; Yang et al., 2018). This conditional form naturally arises from the zero-CCS condition in Theorem 5, motivating the DAG class introduced below.

Definition 3 (Semiparametric GLM DAG). A distribution P follows a semiparametric GLM DAG on $G$ if its density factorizes as $\begin{array} { r } { p ( x ) = \prod _ { j = 1 } ^ { d } p _ { j } \big ( x _ { j } ~ \vert ~ x _ { \mathrm { P a } _ { G } ( j ) } \big ) } \end{array}$ . Source marginal densities $p _ { j }$ on $\chi _ { j }$ are arbitrary and need not belong to an exponential family. For every non-source node $j ,$ its conditional density has the form

$$
p _ { j } ( x _ { j } \mid x _ { \mathrm { P a } _ { G } ( j ) } ) = b _ { j } ( x _ { j } ) \exp \bigl \{ x _ { j } \eta _ { j } \bigl ( x _ { \mathrm { P a } _ { G } ( j ) } \bigr ) - A _ { j } \bigl ( \eta _ { j } \bigl ( x _ { \mathrm { P a } _ { G } ( j ) } \bigr ) \bigr ) \bigr \} ,\tag{3}
$$

where $\eta _ { j }$ is an unspecified canonical-parameter function and $\begin{array} { r } { A _ { j } ( t ) : = \log \int _ { \mathcal { X } _ { j } } b _ { j } ( u ) e ^ { u t } d \nu _ { j } ( u ) } \end{array}$ . The support $\mathcal { X } _ { j }$ and base function $b _ { j }$ are unspecified but do not depend on the parent values. Additionally, each $\eta _ { j }$ depends nontrivially on every parent.

The support $\mathcal { X } _ { j }$ and base function $b _ { j }$ determine the conditional family, while $\eta _ { j }$ captures all parent dependence. The canonical-parameter function $\eta _ { j }$ can be represented through a strictly increasing link $g _ { j }$ and an index $f _ { j }$

$$
\begin{array} { r } { g _ { j } \big ( \mathbb { E } [ X _ { j } \mid X _ { \mathrm { P a } _ { G } ( j ) } ] \big ) = f _ { j } ( X _ { \mathrm { P a } _ { G } ( j ) } ) , \qquad f _ { j } : = g _ { j } \circ A _ { j } ^ { \prime } \circ \eta _ { j } . } \end{array}
$$

This link–index pair is nonunique, so we state all structural assumptions directly in terms of $\eta _ { j }$

Definition 4 (Nonlinear semiparametric GLM DAG). A semiparametric GLM DAG is nonlinear if, for every edge $k  i .$ , there are fixed values z of the other parents in the product-support interior such that $\eta _ { i } ( \cdot , z )$ is nonlinear on $\mathring { \mathcal { X } } _ { k }$ . Equivalently, $\mathcal { D } _ { k } ^ { 2 } \eta _ { i } \notin 0$ on the corresponding derivative or finite-difference domain.

For Gaussian conditionals with fixed variance, canonical-parameter nonlinearity coincides with the mechanism nonlinearity used by SCORE (Rolland et al., 2022). Definition 4 extends this restriction to other conditional families. Parent-dependent nuisance parameters lie outside the class, for example a negative binomial whose size depends on the parents.

Table 1: Example conditional distributions, with index $f ~ = ~ f _ { j } ( x _ { \mathrm { P a } _ { G } ( j ) } )$ and logistic function sigmoid. Noncanonical links can yield nonlinear η even for affine f. NB and Gamma use the mean as their second parameter, with fixed size r and shape α, respectively.
<table><tr><td>Conditional  $X _ { j } \mid x _ { \operatorname { P a } _ { G } ( j ) }$ </td><td>Conditional family</td><td>Support</td><td>Link g</td></tr><tr><td>Poisson(exp f)</td><td>Poisson</td><td> $ { \mathbb { N } } _ { 0 }$ </td><td>log</td></tr><tr><td> ${ \mathrm { P o i s s o n } } ( \operatorname { s o f t p l u s } f )$ </td><td>Poisson</td><td> $ { \mathbb { N } } _ { 0 }$ </td><td> $\mathrm { s o f t p l u s } ^ { - 1 }$ </td></tr><tr><td> $\mathrm { N B } ( r , \exp f )$ </td><td>NB(r)</td><td> $ { \mathbb { N } } _ { 0 }$ </td><td>log</td></tr><tr><td> $\operatorname { B i n o m i a l } ( M , \operatorname { s i g m o i d } ( f ) )$ </td><td>Binomial(M)</td><td> $\{ 0 , \ldots , M \}$ </td><td> $\mu \mapsto \mathrm { l o g i t } ( \mu / M )$ </td></tr><tr><td> $\mathcal { N } ( f , \tau ^ { 2 } )$ </td><td>Gaussian</td><td>R</td><td>identity</td></tr><tr><td> ${ \mathrm { G a m m a } } ( \alpha , \exp f )$ </td><td>Gamma(α)</td><td> $( 0 , \infty )$ </td><td>log</td></tr></table>

Regularity conditions. We assume strictly positive densities on product-support interiors and at least three support values for count coordinates. The results in Section 5 are stated under the corresponding regularity conditions in Appendix B.

## 5 CURVATURE CHARACTERIZATION AND DAG IDENTIFIABILITY

## 5.1 BIVARIATE CHARACTERIZATION AND IDENTIFIABILITY

For both continuous and count variables, we first characterize the conditional distributions with zero CCS. We then establish when the causal direction is identifiable within the resulting model class.

Theorem 5 (Bivariate characterization by the CCS). $S _ { Y } = 0$ if and only $i f p ( y \mid x )$ has the semiparametric GLM conditionalform in (3).

If $S _ { Y } = 0 _ { ; }$ , the log-density curvature in y depends only on y. Integrating derivatives or summing finite differences gives log $p ( x , y ) = a ( x ) + \log b _ { Y } ( y ) + y \eta ( x )$ , which yields this conditional form after normalization.

Theorem 6 (Bivariate identifiability characterization). Let P have a semiparametric GLM representation with edge $X  Y$ , with Condition R1 (affine reversibility; Appendix B.2) imposed in the affine case. Within this model class,

$$
X  Y i s i d e n t i f a b l e \quad \Longleftrightarrow \quad \eta i s n o n l i n e a r o n { \mathring X } _ { X } .
$$

To relate this characterization to the CCS, consider $X  Y$ . The child conditional satisfies

$$
\log p ( \boldsymbol { y } \mid \boldsymbol { x } ) = \underbrace { \log b _ { Y } ( \boldsymbol { y } ) } _ { \mathrm { f a m i l y - s p e c i f i c ~ t e r m } } + \underbrace { \boldsymbol { y } \boldsymbol { \eta } ( \boldsymbol { x } ) - \boldsymbol { A } _ { Y } ( \boldsymbol { \eta } ( \boldsymbol { x } ) ) } _ { \mathrm { p a r e n t ~ c o n t r i b u t i o n , a f f u e i n } \boldsymbol { y } } .
$$

Taking a second derivative or second difference in y removes the affine parent contribution, leaving $H _ { Y } = \kappa _ { Y } ( Y )$ , where $\kappa _ { Y } ( y ) : = \mathcal { D } _ { Y } ^ { 2 } \log p ( y ~ | ~ x ) = \mathcal { D } _ { Y } ^ { 2 } \log b _ { Y } ( y )$ is the child’s own-factor curvature. This curvature need not be constant: for a negative binomial conditional with fixed size $\begin{array} { r } { r > 0 , H _ { Y } = \log \frac { ( Y + r + 1 ) ( Y + 1 ) } { ( Y + r ) ( Y + 2 ) } } \end{array}$ , which is zero for $r = 1$ and nonconstant otherwise. Thus, simply replacing derivatives with finite differences in SCORE does not ensure constant curvature at a sink. Conditioning on Y, however, gives $S _ { Y } = 0$

For the parent X, the curvature separates as

$$
\begin{array} { r } { { \cal H } _ { X } = \kappa _ { X } ( X ) + \underbrace { Y \mathcal { D } _ { X } ^ { 2 } \eta ( X ) - \mathcal { D } _ { X } ^ { 2 } \left[ A _ { Y } ( \eta ( X ) ) \right] } _ { \psi _ { X } ( X , Y ) : \mathrm { \ r e s i d u a l \ c u r v a t u r e } } . } \end{array}
$$

Here $\kappa _ { X } ( x ) : = \mathcal { D } _ { X } ^ { 2 } \log p _ { X } ( x )$ is the source’s own-factor curvature. Conditioning on X makes all terms except $Y \hat { \mathcal { D } } _ { X } ^ { 2 } \dot { \eta } ( X )$ deterministic. Hence $S _ { X } = \mathbb { E } [ \{ { \mathcal { D } } _ { X } ^ { 2 } \eta ( X ) \} ^ { 2 } \operatorname { V a r } ( Y \mid X ) ]$ . Since $\mathrm { V a r } ( Y \mid$ $X ) > 0 , S _ { Y } = 0 \overset { \vartriangle } { < } S _ { X }$ exactly when η is nonlinear.

Remark 7 (Translation to link and index). Identifiability is determined by nonlinearity of the canonical-parameter function η, not by the link or index in isolation. Appendix D.2 gives the link– index characterization.

## 5.2 MULTIVARIATE IDENTIFIABILITY

We extend the CCS argument to recover a causal order in multivariate DAGs and use the OCS to recover parents. A remaining set R obtained by repeatedly removing sinks is ancestral. Its marginal $P _ { R }$ retains the semiparametric GLM factorization on $G _ { R }$ and the nonlinearity condition.

Theorem 8 (DAG recovery by the CCS and OCS). Let P follow a semiparametric GLM DAG on G.

(i) If P follows a nonlinear semiparametric GLM DAG, then, for every ancestral remaining set R and $j \in R , j$ is a sink ofG<sub>R</sub> ifand only $i f S _ { j } ( R ) = 0 .$

(ii) Given a causal order and $j \in { \mathrm { P r e d } } ( i ) , j \in { \mathrm { P a } } _ { G } ( i )$ if and only $i f \mathrm { O C S } _ { j  i } > 0$

On an ancestral remaining set R, the bivariate decomposition extends to $H _ { j , R } ( x _ { R } ) = \kappa _ { j } ( x _ { j } ) +$ $\psi _ { j , R } ( x _ { R } )$ , where $\kappa _ { j } ( x _ { j } ) : = \mathcal { D } _ { j } ^ { 2 } \log p _ { j } ( x _ { j } \mid x _ { \mathrm { P a } _ { G } ( j ) } )$ is the own-factor curvature and $\psi _ { j , R }$ is the residual curvature from the child contributions. Conditioning on $X _ { j }$ makes $\kappa _ { j } ( X _ { j } )$ deterministic, so the CCS measures the conditional variation of $\psi _ { j , R }$ . This residual vanishes at a sink; at a nonsink, the nonlinearity condition makes its averaged conditional variance positive.

For parent recovery, node i is a sink of $G _ { R _ { i } }$ . Under $P _ { R _ { i } } , H _ { i j , R _ { i } } = \mathcal { D } _ { j } \eta _ { i } \mathrm { i f } j \in \mathrm { P a } _ { G } ( i )$ and ${ \cal H } _ { i j , { R _ { i } } } =$ 0 otherwise. Thus a nonconstant parent effect suffices for OCS; nonlinearity is not required.

Corollary 9 (DAG identifiability). Let Pfollow a nonlinear semiparametric GLM DAG on G. Then G is identifiablefrom P within this nonlinear semiparametric GLM class.

This class strictly contains SCORE’s nonlinear Gaussian ANMs. For example, the Gamma model in Appendix D.1 is identifiable despite its nonconstant sink curvature $H _ { Y } \overset { \cdot } { = } - ( \alpha - 1 ) / Y ^ { 2 }$ , while $S _ { Y } = 0$

## 6 DISCO ALGORITHM

DISCO consists of concrete-score estimation, CCS-based ordering, and parent selection given the estimated order. After applying the rank transform, we estimate joint concrete scores using scoreentropy discrete diffusion and project them to the marginal concrete scores needed for CCS-based ordering and OCS-based parent selection. Parent recovery can use CAM pruning (Buhlmann et al.,¨ 2014), PCM-GAM (Lundborg et al., 2024), or the OCS-based rule. The OCS-based rule reuses the fitted score networks, providing a scalable alternative to regression- and testing-based parent selection. Appendix I.2 gives detailed comparisons of these methods. Appendix H.2.8 presents a continuous-data version and a comparison with DiffAN (Sanchez et al., 2023).

## 6.1 CONCRETE-SCORE ESTIMATION

We first apply the rank transform, mapping the distinct training values of each count variable j to the empirical rank grid $\{ 0 , \ldots , K _ { j } \}$ , where $K _ { j } + 1$ is the number of distinct training values. We then estimate concrete scores on the transformed data using discrete diffusion.

The rank transform makes neighboring differences invariant to strictly increasing transformations on the support of each variable. For estimation, $X , X _ { j }$ , and $p _ { R }$ refer to the transformed data, rank grid, and corresponding marginal density.

For $j ~ \in ~ R ~ \subseteq ~ V$ and $y ~ \in ~ { \mathcal { X } } _ { j }$ , the concrete-score entry is the probability ratio $s _ { j , y } ^ { R } ( x _ { R } ) \ : =$ $p _ { R } ( x _ { R } ^ { j  y } ) / p _ { R } ( x _ { R } )$ . On the transformed data, we train a joint concrete-score network $s _ { \theta } ( x ; \sigma )$ to estimate the concrete score of the noise-corrupted joint distribution by minimizing a neighborrestricted denoising score-entropy objective (Lou et al., 2024). Here σ denotes the diffusion noise level. $\mathbf { A s } \sigma  0$ , the concrete score of the noise-corrupted distribution converges to that of the data distribution.

Marginal concrete-score projection. Computing the CCS after each sink removal requires the concrete scores of the remaining variables’ marginal distribution. These scores cannot generally be obtained by simply restricting the joint concrete score to the remaining variables. The following identity estimates them via conditional-mean regression.

Proposition 10 (Marginalization of the concrete score). Let $j \in R \subseteq V$ with count-valued $X _ { j }$ . For each neighboring replacement $y \in \mathcal { X } _ { j }$ , the concrete-score entries satisfy

$$
s _ { j , y } ^ { R } ( x _ { R } ) = \mathbb { E } \big [ s _ { j , y } ^ { V } ( X _ { V } ) \mid X _ { R } = x _ { R } \big ] .
$$

Applying Proposition 10 to the noise-corrupted distribution gives the same conditional-mean identity at each noise level σ. Using predictions from $s _ { \theta } ( x ; \sigma )$ as targets, we learn a single mask-conditioned projection network with remaining set R as an input. The resulting estimator $\widehat { s } _ { \theta , \phi } ^ { R } ( \boldsymbol { x } _ { R } ; \boldsymbol { \sigma } )$ , with projection parameters $\phi ,$ provides marginal concrete scores for different sets R without refitting. Appendices F and G provide the theoretical analysis and implementation details, respectively.

## 6.2 CCS AND OCS COMPUTATION

We compute the CCS and OCS at near-zero noise levels on held-out folds, using estimators fitted on the other folds. Fold indices on fitted quantities are suppressed.

Let $\widehat { \ell } _ { j , R } ( x _ { R } ; \sigma )$ denote the projection network’s predicted log concrete-score entry for increasing $x _ { j }$ by one. Where all required neighbors lie in the grid, it equals log $\left[ \widehat { s } _ { \theta , \phi } ^ { R } ( x _ { R } ; \sigma ) \right] _ { j , x _ { j } + 1 }$ , and for $i , j \in R$ we estimate curvature by

$$
\begin{array} { r } { \widehat { H } _ { i j , R } ( x _ { R } ; \sigma ) : = \widehat { \ell } _ { j , R } ( x _ { R } + e _ { i } ; \sigma ) - \widehat { \ell } _ { j , R } ( x _ { R } ; \sigma ) . } \end{array}\tag{4}
$$

For the CCS, we first average diagonal curvature estimates over a finite set of noise levels $\mathcal { G } _ { \sigma }$ without taking absolute values: $\begin{array} { r } { \widehat { H } _ { j , R } : = | \mathcal { G } _ { \sigma } | ^ { - 1 } \sum _ { \sigma \in \mathcal { G } _ { \sigma } } \widehat { H } _ { j j , R } ( \cdot ; \sigma ) } \end{array}$

To estimate the CCS, we first restrict evaluation to observations where the required finite differences are defined. We group these observations by the rank value of $X _ { j }$ and take a sample-size-weighted average of the within-group sample variances of $\widehat { H } _ { j , R }$ . For these observations, let ${ \mathcal { I } } _ { j , v } \ = \ \{ a \ :$ $X _ { a j } = v \}$ and $n _ { v } = | \mathcal { I } _ { j , v } |$ . Keeping the levels ${ \mathcal { V } } _ { j } = \mathbf { \bar { \{ v : n _ { v } \geq n _ { \operatorname* { m i n } } \} } }$ gives

$$
\widehat S _ { j } ( R ) = \frac { 1 } { \sum _ { v \in \mathcal { V } _ { j } } n _ { v } } \sum _ { v \in \mathcal { V } _ { j } } n _ { v } \widehat { \mathrm { V a r } } \Big ( \{ \widehat H _ { j , R } ( X _ { a } ) : a \in \mathcal { I } _ { j , v } \} \Big ) .\tag{5}
$$

Here $\widehat { \mathrm { V a r } }$ is the sample variance.

We compute the OCS separately on each held-out fold. Given an estimated causal order ${ \widehat { \pi } } ,$ define Pred<sub>π</sub> $( i ) = \{ j : \widehat { \pi } ( j ) < \widehat { \pi } ( i ) \}$ and $\widehat { R } _ { i } = \{ i \} \cup \mathrm { P r e d } _ { \widehat { \pi } } ( i )$ . Let $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { B }$ denote these folds. For candidate edge $j  i ,$ let $\mathcal { T } _ { b } ^ { i j }$ index the evaluation observations used for that edge. For nonempty $\mathcal { T } _ { b } ^ { i j }$ , we take absolute values before averaging over noise levels and observations:

$$
\widehat { \mathrm { O C S } } _ { j  i } ^ { ( b ) } = \frac { 1 } { | \mathcal { T } _ { b } ^ { i j } | | \mathcal { G } _ { \sigma } | } \sum _ { a \in \mathcal { T } _ { b } ^ { i j } } \sum _ { \sigma \in \mathcal { G } _ { \sigma } } \big | \widehat { H } _ { i j , \widehat { R } _ { i } } ( \boldsymbol { X } _ { a , \widehat { R } _ { i } } ; \sigma ) \big | .\tag{6}
$$

## 6.3 DAG RECOVERY

Starting from $R = V .$ , DISCO repeatedly removes a minimizer of ${ \widehat { S } } _ { j } ( R )$ and assigns that node the last available position. Given this order, parents can be selected using CAM, PCM-GAM, or OCS. Algorithm 1 presents the OCS-based implementation.

For scalable parent selection, we use the OCS estimates without fitting additional regressions or performing conditional-independence tests. Estimated absolute curvatures can be positive even for nonparents. To account for this noise baseline, we standardize the OCS estimates for each target node and fold, obtaining $Z _ { j  i } ^ { ( b ) }$ . We retain edges whose selection frequency $\Pi _ { j \to i } ( \tau _ { Z } ) : =$ $\begin{array} { r } { B ^ { - 1 } \sum _ { b = 1 } ^ { B } \mathbf { 1 } \{ Z _ { j  i } ^ { ( b ) } > \tau _ { Z } \} } \end{array}$ is at least $\pi _ { \mathrm { m i n } }$

Algorithm 1 DISCO with OCS-based parent selection.   
Input: marginal concrete-score estimator $\widehat { s } _ { \theta , \phi } ^ { R }$ and evaluation folds $\{ \mathcal { T } _ { b } \} _ { b = 1 } ^ { B }$   
Ordering Parent selection given $\widehat { \pi }$   
$R  V$ for each node $i \in V$ do   
while $| R | > 1$ do $\widehat { R } _ { i }  \{ i \} \cup \mathrm { P r e d } _ { \widehat { \pi } } ( i )$   
Compute ${ \widehat { S } } _ { j } ( R )$ for all $j \in R$ Compute the foldwise $\widehat { \mathrm { O C S } } _ { j  i } ^ { ( b ) }$   
$\widehat { j } _ { R } \in$ arg min ${ \cdot j \in } { R } \widehat { S } _ { j } ( R )$ Standardize and form $\Pi _ { j  i } ( \tau _ { Z } )$   
$\widehat { \pi } ( \widehat { j } _ { R } ) \gets | R |$ Select $j \to i \mathrm { i f } \Pi _ { j \to i } ( \tau _ { Z } ^ { \setminus } ) \geq \pi _ { 1 }$ min   
$R \gets R \backslash \{ \widehat { j } _ { R } \}$ end for   
end while Output: DAG $\widehat { G }$   
Assign the sole remaining node to position 1

For the consistency result, we consider a variant of Algorithm 1 that selects parents by thresholding the raw OCS at a deterministic level $\lambda _ { n } > 0$

Theorem 11 (Plug-in Consistency of DISCO with raw OCS thresholding). Under the assumptions of Theorem 8 and Assumption 15, this variant recovers a causal order of G with probability tending to one as n → ∞. $I f \lambda _ { n }  0$ and the curvature-estimation error $\varepsilon _ { n }$ from Assumption 15 satisfies $\varepsilon _ { n } = o _ { p } ( \lambda _ { n } )$ , then $\operatorname* { P r } \{ { \widehat { G } } = G \} \longrightarrow 1$

Transformation invariance. For the population result on count data, let $T _ { j }$ map each value in the ordered support of coordinate $j$ to its support rank. The fitted map $\widehat { T } _ { j }$ is its sample counterpart.

Proposition 12 (Invariance to strictly increasing transformations). DISCO with OCS-based parent selection returns the same graph under strictly increasing transformations of each variable when using the same seed.

Proofs of the formal results in this section are provided in Appendix E.

Computational complexity. Following DiffAN (Sanchez et al., 2023), we fix the training epochs and evaluation sample size, and count each score evaluation at unit cost. With fixed hidden widths and mask draws per observation, the rank transform and training take $O ( n d )$ for n training samples on consecutive counts. Ordering requires $O ( d ^ { 2 } )$ shifted-state evaluations in ${ \dot { O } } ( d )$ sequential batches. OCS-based parent selection uses $O ( d )$ batches and $O ( d ^ { 2 } )$ candidate-edge operations. These counts give $O ( n d \bar { + } d ^ { 2 } )$ for the OCS-based implementation; CAM and PCM-GAM require additional regressions or tests.

## 7 NUMERICAL EXPERIMENTS

We evaluate whether conditional curvature restores the ordering signal when nonconstant own-factor curvature invalidates the constant-curvature criterion, whether DISCO achieves family-agnostic count-DAG recovery within the semiparametric GLM DAG class, and whether it scales to large DAGs. We compare DISCO with DiffAN (Sanchez et al., 2023), NOTEARS-MLP (Zheng et al., 2020), ODS (Park and Raskutti, 2018), and MRS (Park and Park, 2019a). ODS (Oracle) receives the true nodewise conditional families and their fixed parameters, whereas DISCO receives no family labels. For parent selection, we compare CAM’s additive-regression pruning (Buhlmann et al.,¨ 2014) and PCM-GAM’s conditional-mean independence testing (Lundborg et al., 2024). Unless otherwise stated, DISCO uses OCS-based parent selection. Experimental settings and evaluation metrics are given in Appendix H.1. Additional experiments and algorithm diagnostics appear in Appendices H and I.

DAG recovery versus sample size. Figure 1 compares DISCO with the baselines on DAGs with $d = 1 0 0$ as $n \in \{ 5 0 0$ , 1000, 2000, 3000, 4000, 5000} increases: NB with $r = 1$ in panel (a), NB with $r = 6$ in panel (b), and the mixed NB → Bin → Poi configuration in panel (c). DISCO attains the highest median $F _ { 1 }$ at larger sample sizes under r = 6 and mixed families, whereas ODS (Oracle) remains strongest under $r = 1$ . As illustrated by the negative binomial example in Section 5.1, ownfactor curvature vanishes when $r = 1$ , so the constant-curvature criterion is already sufficient and estimating the more general conditional curvature criterion can incur an additional finite-sample efficiency cost. Nevertheless, DISCO shows improved recovery at larger sample sizes. DiffAN shows no consistent gains with more data under $r \ = \ 6$ and mixed families, where the constantcurvature criterion fails because sink curvature is nonconstant.

Conditional curvature ablation. We compare CCS and the constant-curvature criterion on the same sets of remaining nodes using the same fitted networks $( d = 5 0 , n = 5 0 0 0 )$ . Table $2 ( \mathrm { a } )$ shows that CCS improves the accuracy of identifying sinks, with larger gains for Poisson and negative binomial data than for binomial data. This improvement does not always increase DAG recovery accuracy: for binomial data, mean $F _ { 1 }$ is 0.877 with CCS and 0.884 with the constant-curvature criterion. Table 6 in Appendix H.1 reports DAG recovery for all three conditional families.

Parent recovery given an estimated order. We compare OCS, CAM, and PCM-GAM given the same estimated DISCO order at $d = 5 0$ and $n = 5 0 0 0$ , using disjoint ordering and parent-recovery samples. CAM improves mean $F _ { 1 }$ over OCS for NB and Bin, whereas OCS performs best for Poi (Table 2(b)). Appendix I.2 gives the test settings, SHD, and true-order comparisons.

![](images/54112101e83031dd0fd0720cbd8cfe3f7dc3161e065d8602a4cc22eb208ef90a.jpg)  
(a) NB (r = 1)

![](images/ddac2fb94627dfa27507df2c8b21b1b1639225b2a3ed0c956ce35a4a02643925.jpg)  
sample size n  
(b) NB (r = 6)

![](images/933dd808b64124b0c8846ddf0a39845522bc52f1826490cf2ddf86c689cea85e.jpg)  
(c) NB → Bin → Poi  
Figure 1: End-to-end $F _ { 1 }$ versus sample size on DAGs with $d = 1 0 0 \colon$ (a) NB with r = 1; (b) NB with $r \ = \ 6 ;$ and (c) the mixed NB → Bin → Poi configuration. Points and error bars show the median and interquartile range. ODS (Oracle) uses the true nodewise conditional families and their fixed parameters.

Table 2: Ordering and parent-selection comparisons $( d = 5 0 , n = 5 0 0 0 )$ . (a) Sink-selection accuracy: Constant denotes the constant-curvature criterion; gain is in percentage points. (b) Mean $F _ { 1 }$ given the same estimated DISCO order.  
(a) Sink selection (%)
<table><tr><td>Family</td><td>Constant</td><td>CCS</td><td>Gain</td></tr><tr><td>Poi</td><td>68.3</td><td>81.7</td><td>+13.3</td></tr><tr><td>NB</td><td>73.3</td><td>93.3</td><td>+20.0</td></tr><tr><td>Bin</td><td>80.0</td><td>83.3</td><td>+3.3</td></tr></table>

(b) Parent recovery $( F _ { 1 } )$
<table><tr><td>Selector</td><td>Poi</td><td>NB</td><td>Bin</td></tr><tr><td>OCS</td><td>0.858</td><td>0.751</td><td>0.828</td></tr><tr><td>CAM</td><td>0.850</td><td>0.823</td><td>0.898</td></tr><tr><td>PCM-GAM</td><td>0.712</td><td>0.703</td><td>0.738</td></tr></table>

Scalability with graph size. Table 3 reports $F _ { 1 }$ and total wall-clock time as d grows on Poisson DAGs with $n = 5 0 0 0$ . At $d = 1 0 0 0$ , DISCO attains median $F _ { 1 } = 0 . 7 6 9$ in $5 7 . 4$ minutes on a single GPU, whereas all baselines exceed the 240-minute time limit. This demonstrates practical scalability; the runtimes are not intended as a hardware-matched speed comparison.

Table 3: Scalability on Poisson DAGs $( n = 5 0 0 0$ ; numerical entries are medians). Timeout: runtime exceeds 240 minutes.
<table><tr><td></td><td colspan="2">d = 100</td><td colspan="2">d = 200</td><td colspan="2"> $d = 5 0 0$ </td><td colspan="2">d = 1000</td></tr><tr><td>Method</td><td> $F _ { 1 }$ </td><td>Total time (min)</td><td> $F _ { 1 }$ </td><td>Total time (min)</td><td> $F _ { 1 }$ </td><td>Total time (min)</td><td> $F _ { 1 }$ </td><td>Total time (min)</td></tr><tr><td>DISCO</td><td>0.915</td><td>2.7</td><td>0.835</td><td>3.5</td><td>0.769</td><td>10.8</td><td>0.769</td><td>57.4</td></tr><tr><td>ODS (Oracle)</td><td>0.725</td><td>12.5</td><td>0.713</td><td>37.0</td><td>0.729</td><td>204.7</td><td></td><td>Timeout</td></tr><tr><td>MRS</td><td>0.786</td><td>3.5</td><td>0.763</td><td>10.8</td><td>0.742</td><td>96.3</td><td></td><td>Timeout</td></tr><tr><td>NOTEARS-MLP</td><td>0.467</td><td>174.1</td><td></td><td>Timeout</td><td></td><td>Timeout</td><td></td><td>Timeout</td></tr><tr><td>DiffAN</td><td></td><td>Timeout</td><td>一</td><td>Timeout</td><td></td><td>Timeout</td><td>一</td><td>Timeout</td></tr></table>

Family-agnostic DAG learning on real MLB data. We revisit the Lahman batting data previously analyzed for count-DAG learning by Park and Park (2019b), using $d = 1 7$ count variables from $n = 1 \small { , } 4 0 0$ player–seasons during 2019–2025. All methods in this diagnostic use the same four-level quantile representation (q4), chosen to increase observations per level for CCS estimation. We evaluate ordering against seven domain-informed accounting and containment relations among standard baseball statistics (Major League Baseball, n.d.). ODS requires a conditional-family specification, and its mean ordering agreement $A _ { \mathrm { t o p } }$ with these references varies from 0.333 to 0.429 across family choices. In contrast, DISCO requires no family or link input and attains mean

$A _ { \mathrm { t o p } } = 0 . 9 0 5$ . Although the partial references do not define a ground-truth DAG, DISCO shows higher agreement with these domain-informed references than the three fixed-family ODS variants, without requiring a family choice. Appendix H.3 provides the complete diagnostics, including pairwise family-specification comparisons.

## 8 CONCLUSION

DISCO enables score-matching-based causal discovery from count data without prespecifying nodewise conditional families, relaxing a central assumption of existing identifiable count-DAG methods. The CCS and OCS provide ordering and parent-recovery criteria for nonlinear semiparametric GLM DAGs under the stated assumptions. A joint concrete-score network and marginal projection allow DISCO to estimate both criteria, with OCS-based recovery scaling to 1,000-node DAGs in our experiments. The main remaining theoretical challenge is to establish end-to-end consistency of DISCO, from finite-noise discrete-diffusion score estimation and marginal projection to curvature estimation and DAG recovery.

## AI USE STATEMENT

Generative AI tools were used to assist with aspects of the proofs and experimental work. The authors reviewed the AI-assisted material and take responsibility for the final manuscript and results.

## REPRODUCIBILITY STATEMENT

Implementation details, including preprocessing, training objectives, mask sampling, and parent selection, are provided in Appendix G. Appendix H specifies the data-generating mechanisms, baseline configurations, evaluation metrics, and replication protocols. Appendix I describes the algorithm diagnostics, and Appendix H.3 gives the Lahman cohort construction, paired cross-validation design, and reference sets. The appendix also contains the assumptions and proofs of the theoretical results.

## REFERENCES

D. M. Blei, A. Y. Ng, and M. I. Jordan. Latent Dirichlet allocation. Journal of Machine Learning Research, 3:993–1022, 2003.

J. Bodik and V. Chavez-Demoulin. Identifiability of causal graphs under non-additive conditionally parametric causal models. Journal ofMachine Learning Research, 26(264):1–55, 2025.

P. Buhlmann, J. Peters, and J. Ernest. CAM: Causal additive models, high-dimensional order search¨ and penalized regression. The Annals of Statistics, 42(6):2526–2556, 2014.

S. E. Castel, A. Levy-Moonshine, P. Mohammadi, E. Banks, and T. Lappalainen. Tools and best practices for data processing in allelic expression analysis. Genome Biology, 16:195, 2015. doi:10.1186/s13059-015-0762-6.

M. Gao, Y. Ding, and B. Aragam. A polynomial-time algorithm for learning nonparametric causal graphs. In Advances in Neural Information Processing Systems 33 (NeurIPS), pages 11599– 11611, 2020.

A. Ghoshal and J. Honorio. Learning linear structural equation models in polynomial time and sample complexity. In Proceedings ofthe 21st International Conference on Artificial Intelligence and Statistics (AISTATS), volume 84 of Proceedings ofMachine Learning Research, pages 1466– 1475, 2018.

A. Huang. Mean-parametrized Conway–Maxwell–Poisson regression models for dispersed counts. Statistical Modelling, 17(6):359–380, 2017. doi:10.1177/1471082X17697749.

A. Hyvarinen. Estimation of non-normalized statistical models by score matching.¨ Journal of Machine Learning Research, 6(24):695–709, 2005.

J. Kang, S. Kim, C. Lee, D. Hwang, J. Chung, Y. Ko, S. Lee, S. Kim, and S. Lim. Score-informed neural operator for enhancing ordering-based causal discovery. In Advances in Neural Information Processing Systems 38 (NeurIPS), pages 113109–113151, 2025.

S. Lahman. Lahman baseball database. Society for American Baseball Research. https:// sabr.org/lahman-database/, accessed September 22, 2026.

A. Lou, C. Meng, and S. Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning (ICML), volume 235 of Proceedings of Machine Learning Research, pages 32819–32848, 2024.

M. I. Love, W. Huber, and S. Anders. Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2. Genome Biology, 15(12):550, 2014.

A. R. Lundborg, I. Kim, R. D. Shah, and R. J. Samworth. The projected covariance measure for assumption-lean variable significance testing. The Annals of Statistics, 52(6):2851–2878, 2024. doi:10.1214/24-AOS2447.

S. Lyu. Interpretation and generalization of score matching. In Proceedings ofthe 25th Conference on Uncertainty in Artificial Intelligence (UAI), pages 359–366, 2009.

Major League Baseball. Glossary of standard statistics: At-bat, Hit, and Intentional walk. MLB.com, n.d. Accessed September 21, 2026.

P. McCullagh and J. A. Nelder. Generalized Linear Models. Chapman & Hall, 2nd edition, 1989.

C. Meng, K. Choi, J. Song, and S. Ermon. Concrete score matching: Generalized score matching for discrete data. In Advances in Neural Information Processing Systems 35 (NeurIPS), pages 34532–34545, 2022.

F. Montagna, N. Noceti, L. Rosasco, K. Zhang, and F. Locatello. Scalable causal discovery with score matching. In Proceedings of the 2nd Conference on Causal Learning and Reasoning (CLeaR), volume 213 of Proceedings of Machine Learning Research, pages 752–771, 2023a.

F. Montagna, N. Noceti, L. Rosasco, K. Zhang, and F. Locatello. Causal discovery with score matching on additive models with arbitrary noise. In Proceedings of the 2nd Conference on Causal Learning and Reasoning (CLeaR), volume 213 of Proceedings of Machine Learning Research, pages 726–751, 2023b.

J. A. Nelder and R. W. M. Wedderburn. Generalized linear models. Journal ofthe Royal Statistical Society: Series A (General), 135(3):370–384, 1972.

G. Park and H. Park. Identifiability of generalized hypergeometric distribution (GHD) directed acyclic graphical models. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 89 of Proceedings of Machine Learning Research, pages 158–166, 2019a.

G. Park and S. Park. High-dimensional Poisson structural equation model learning via ℓ<sub>1</sub>-regularized regression. Journal ofMachine Learning Research, 20(95):1–41, 2019b.

G. Park and G. Raskutti. Learning large-scale Poisson DAG models based on overdispersion scoring. In Advances in Neural Information Processing Systems 28 (NeurIPS), pages 631–639, 2015.

G. Park and G. Raskutti. Learning quadratic variance function (QVF) DAG models via overdispersion scoring (ODS). Journal ofMachine Learning Research, 18(224):1–44, 2018.

J. Peters, J. M. Mooij, D. Janzing, and B. Scholkopf. Causal discovery with continuous additive¨ noise models. Journal ofMachine Learning Research, 15(58):2009–2053, 2014.

P. J. Rathouz and L. Gao. Generalized linear models with unspecified reference distribution. Biostatistics, 10(2):205–218, 2009.

M. D. Robinson, D. J. McCarthy, and G. K. Smyth. edgeR: a Bioconductor package for differential expression analysis of digital gene expression data. Bioinformatics, 26(1):139–140, 2010.

P. Rolland, V. Cevher, M. Kleindessner, C. Russell, D. Janzing, B. Scholkopf, and F. Locatello.¨ Score matching enables causal discovery of nonlinear additive noise models. In Proceedings of the 39th International Conference on Machine Learning (ICML), volume 162 of Proceedings of Machine Learning Research, pages 18741–18753, 2022.

P. Sanchez, X. Liu, A. Q. O’Neil, and S. A. Tsaftaris. Diffusion models for causal discovery via topological ordering. In International Conference on Learning Representations (ICLR), 2023.

J. Stoklosa, R. V. Blakey, and F. K. C. Hui. An overview of modern applications of negative binomial modelling in ecology and biodiversity. Diversity, 14(5):320, 2022.

V. Vo, T. Le, H. Zhao, E. V. Bonilla, and D. Phung. Ordering-based causal discovery via generalized score matching. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD), pages 4706–4717, 2026. doi:10.1145/3770855.3817672.

Z. Yang, Y. Ning, and H. Liu. On semiparametric exponential family graphical models. Journal of Machine Learning Research, 19(57):1–59, 2018.

K. Zhang, S. Zhu, M. Kalander, I. Ng, J. Ye, Z. Chen, and L. Pan. gCastle: A Python toolbox for causal discovery. arXiv preprint arXiv:2111.15155, 2021.

X. Zheng, C. Dan, B. Aragam, P. Ravikumar, and E. P. Xing. Learning sparse nonparametric DAGs. In Proceedings ofthe 23rd International Conference on Artificial Intelligence and Statistics (AIS-TATS), volume 108 of Proceedings of Machine Learning Research, pages 3414–3425, 2020.

## APPENDIX CONTENTS

A Notation . 14   
B Regularity Conditions . 15   
B.1 Common Regularity 15   
B.2 Semiparametric GLM Representations 15   
B.3 Moment Conditions 15   
C Proofs for Section 5 16   
C.1 Proof of Theorem 5 16   
C.2 Proof of Theorem 6 16   
C.3 Proof of Theorem 8 17   
C.4 Proof of Corollary 9 18   
D Additional Identifiability Results 18   
D.1 Strict Extension of SCORE 18   
D.2 Link–Index Characterization 19   
E Proofs for Section 6 19   
E.1 Proof of Proposition 10 19   
E.2 Proof of Theorem 11 20   
E.3 Proof of Proposition 12 22   
F Theoretical Properties of Concrete-Score Estimation 22   
F.1 Denoising Score-Entropy Objective 22   
F.2 Vanishing-Noise Limit 23   
G DISCO Implementation 23   
G.1 Rank Transform . 23   
G.2 Joint Concrete-Score Training 23   
G.3 Marginal Concrete-Score Projection 24   
G.4 Curvature Estimation 24   
G.5 Parent Selection . 24   
H Experimental Details and Additional Results 25   
H.1 Experimental Setup 25   
H.2 Additional Benchmarks 27   
H.3 Real Data: Lahman Batting Counts . 31   
I Algorithm Diagnostics . 34   
I.1 Marginal Projection Methods 34   
I.2 Parent Selection . . 34

## A NOTATION

Table 4: Recurring symbols.
<table><tr><td>Symbol</td><td>Meaning (defined in)</td></tr><tr><td> $G = ( V , E ) , | V | = d$ </td><td>DAG and number of nodes (Sec. 2.2)</td></tr><tr><td> $\widehat { G }$ </td><td>estimated DAG (Sec. 6.3)</td></tr><tr><td> $\mathrm { P a } _ { G } ( j ) , \mathrm { C h } _ { G } ( j )$ </td><td>parents and children of j (a sink has  $\operatorname { C h } _ { G } ( j ) = \varnothing )$ </td></tr><tr><td>source node</td><td>a node  $j$  with  $\operatorname { P a } _ { G } ( j ) = \varnothing$ </td></tr><tr><td> $R \subseteq V$ </td><td>remaining node set, ancestral when obtained by recursive sink re- moval</td></tr><tr><td> $G _ { R } , P _ { R } , p _ { R }$ </td><td>induced subgraph, marginal distribution of  $X _ { R } ,$  and its density</td></tr><tr><td> $p _ { R , \sigma }$   $\pi , { \mathrm { P r e d } } ( i ) , R _ { i }$ </td><td>density of the noise-corrupted R-marginal (App. F) causal order, predecessors  $\{ j : \pi ( j ) < \pi ( i ) \}$  , and  $R _ { i } ~ = ~ \{ i \} \cup$ </td></tr><tr><td> $\widehat { \pi } , \mathrm { P r e d } _ { \widehat { \pi } } ( i ) , \widehat { R } _ { i }$ </td><td>Pred(i) (Sec. 3)</td></tr><tr><td></td><td>estimated counterparts of the preceding row (Sec. 6.2)</td></tr><tr><td> $x _ { A } ^ { j  y }$ </td><td> $x _ { A }$  with coordinate  $j$  reset to y (Sec. 2.2)</td></tr><tr><td> $\bar { \mathcal { D } _ { j } } , \mathcal { D } _ { j } ^ { 2 }$   $b _ { j }$ </td><td>unit coordinate (second) difference, with  $\partial _ { j } , \partial _ { j } ^ { 2 }$  if continuous</td></tr><tr><td> $g _ { j } , \ f _ { j }$ </td><td>unspecified base function of a non-source conditional (Def. 3)</td></tr><tr><td> $\eta _ { j }$ </td><td>link and its corresponding index</td></tr><tr><td> $A _ { j }$ </td><td>canonical-parameter function encoding parent dependence log-partition of a non-source conditional&#x27;s natural exponential fam-</td></tr><tr><td> $\kappa _ { j } ( \boldsymbol { x } _ { j } )$ </td><td>ily</td></tr><tr><td> $\psi _ { X } , \psi _ { j , R }$ </td><td>own-factor curvature of node j (Sec. 5.2) residual curvature in the bivariate or marginal decomposition</td></tr><tr><td> $H _ { j , R } = \mathcal { D } _ { j } ^ { 2 } \log { p _ { R } }$ </td><td>(Secs. 5.1–5.2)</td></tr><tr><td> $H _ { i j , R } = \bar { \mathcal { D } } _ { i } \mathcal { D } _ { j } \log p _ { R }$ </td><td>diagonal curvature of the marginal (Sec. 3) off-diagonal curvature (parent recovery)</td></tr><tr><td> $\mathcal { A } _ { j } ^ { ( 2 ) } , \mathcal { A } _ { i j } ^ { ( 1 , 1 ) }$ </td><td></td></tr><tr><td></td><td>admissible-difference events for  $\mathcal { D } _ { j } ^ { 2 } , \mathcal { D } _ { i } \mathcal { D } _ { j }$ </td></tr><tr><td> $\bar { S _ { j } } ( R )$ </td><td>conditional curvature score (CCS), (1)</td></tr><tr><td> ${ \mathrm { O C S } } _ { j  i }$ </td><td>off-diagonal curvature score, (2)</td></tr><tr><td> $O _ { j \to i } ( R )$ </td><td>OCS on a general admissible marginal in the consistency proof</td></tr><tr><td> $T _ { j }$   $\widehat { T } _ { i }$ </td><td>population rank transform (Sec. 6.3)</td></tr><tr><td></td><td>fitted rank transform from the score-training split (Sec. 6.1)</td></tr><tr><td> $s _ { j , y } ^ { \check { R } }$ </td><td>concrete-score entry, neighboring when  $y = x _ { j } \pm 1$ </td></tr><tr><td> $s _ { \theta }$ </td><td>joint concrete-score network</td></tr><tr><td> $\widehat { s } _ { \theta , \phi } ^ { R }$ </td><td>R-marginal concrete-score estimate</td></tr><tr><td> $\widehat { \ell } _ { j , R } ( \cdot ; \sigma )$ </td><td>predicted log concrete-score entry for increasing  $x _ { j }$  by one</td></tr><tr><td></td><td>(Sec. 6.2)</td></tr><tr><td> $\widehat { S } _ { j } ( R ) , \widehat { \mathrm { O C S } } _ { j  i } ^ { ( b ) }$ </td><td>estimated CCS and foldwise estimated OCS (Sec. 6.2)</td></tr></table>

Admissible domains. A finite difference is defined only where all required shifted values remain in the support; we call this its admissible domain. For $r \in \{ 1 , 2 \}$ , define

$$
\mathcal { X } _ { j } ^ { \circ , r } = \displaystyle \int \bigl \{ \begin{array} { l l } { \{ x _ { j } \in \mathcal { X } _ { j } : x _ { j } , x _ { j } + 1 , \ldots , x _ { j } + r \in \mathcal { X } _ { j } \} , } & { X _ { j } \mathrm { ~ c o u n t } , } \\ { \tilde { \mathcal { X } } _ { j } , } & { X _ { j } \mathrm { ~ c o n t i n u o u s } . } \end{array} 
$$

Thus $\mathcal { D } _ { j } ^ { r }$ requires $x _ { j } \in \mathcal { X } _ { j } ^ { \circ , r }$ , and, for $i \neq j , \mathcal { D } _ { i } \mathcal { D } _ { j }$ requires $x _ { i } \ \in \ \mathcal { X } _ { i } ^ { \circ , 1 }$ and $x _ { j } \ \in \ \mathcal { X } _ { j } ^ { \circ , 1 }$ . The corresponding curvature events are

$$
\begin{array} { r } { \boldsymbol { \mathcal { A } } _ { j } ^ { ( 2 ) } = \{ \boldsymbol { X } _ { j } \in \mathcal { X } _ { j } ^ { \circ , 2 } \} , \qquad \boldsymbol { \mathcal { A } } _ { i j } ^ { ( 1 , 1 ) } = \{ \boldsymbol { X } _ { i } \in \mathcal { X } _ { i } ^ { \circ , 1 } , \boldsymbol { X } _ { j } \in \mathcal { X } _ { j } ^ { \circ , 1 } \} , \quad i \neq j . } \end{array}
$$

Continuous coordinates impose only the support-interior restriction, which has probability one.

For these curvatures, population moments and almost-sure statements use the marginal distribution conditioned on the corresponding event. For the CCS, both the outer expectation and the conditional variance use $P _ { R } ( \cdot \mid A _ { j } ^ { ( 2 ) } )$ ; the OCS uses $P _ { R _ { i } } ( { \bf \cdot } { \bf \sigma } | { \bf \mathcal { A } } _ { i j } ^ { ( 1 , 1 ) } )$ . If the event is determined by the conditioning variables, this restriction leaves the conditional distribution of the other variables unchanged.

## B REGULARITY CONDITIONS

## B.1 COMMON REGULARITY

We assume finite, strictly positive densities on product-support interiors. Each coordinate support is a connected interval with nonempty interior or a consecutive integer set with at least three levels. For the distribution-level characterization, log p is jointly $C ^ { 2 }$ in the continuous coordinates for each fixed choice of count coordinates.

## B.2 SEMIPARAMETRIC GLM REPRESENTATIONS

For the semiparametric GLM representations in Definition 3, count supports are additionally bounded below. Source densities and non-source base functions are finite and strictly positive on their support interiors, and their logarithms are $C ^ { 2 }$ in continuous coordinates. Each $\eta _ { j }$ is finite and jointly $C ^ { 2 }$ in its continuous parent coordinates for each fixed choice of count parents. Required finite differences are finite on their admissible domains, and the canonical parameters lie in the interior of the canonical-parameter domain:

$$
\eta _ { j } \big ( \mathring { \mathcal { X } } _ { \mathrm { P a } _ { G } ( j ) } \big ) \subseteq \operatorname* { i n t } \Xi _ { j } , \qquad \Xi _ { j } : = \{ t \in \mathbb { R } : A _ { j } ( t ) < \infty \} .
$$

This interiority justifies differentiation under the integral, so $A _ { j }$ is smooth and $A _ { j } ^ { \prime \prime } ( t ) = \operatorname { V a r } _ { t } ( X _ { j } ) \in$ $( 0 , \infty )$ on int $\Xi _ { j }$

Condition R1 (Affine reversibility). In the affine branch of Theorem 6, write $\eta ( x ) = \beta _ { 0 } + \beta _ { 1 } x$ with $\beta _ { 1 } \neq 0$ , and define

$$
\widetilde { b } _ { X } ( x ) = p _ { X } ( x ) e ^ { - A _ { Y } ( \beta _ { 0 } + \beta _ { 1 } x ) } , \qquad \widetilde { \Xi } _ { X } = \left\{ t : \int _ { \mathcal { X } _ { X } } \widetilde { b } _ { X } ( x ) e ^ { t x } d \nu _ { X } ( x ) < \infty \right\} .
$$

For count ${ \cal Y } ,$ , require $\beta _ { 1 } e \in \operatorname { i n t } { \widetilde { \Xi } } _ { X }$ at each support endpoint $e _ { \cdot }$ . There are two endpoints for finite support and one for one-sided infinite support. No additional restriction is imposed for continuous $Y$

Integrating out a sink removes its conditional density from the DAG factorization. Therefore an ancestral marginal is finite and strictly positive and retains the remaining conditional densities and parent sets. The nonlinearity condition is also preserved when present. The own-factor curvature is $\dot { \kappa } _ { j } ( x _ { j } ) = \mathcal { D } _ { j } ^ { 2 } \log b _ { j } ( x _ { j } )$ for a non-source and $\bar { \kappa } _ { j } ( x _ { j } ) = \mathcal { D } _ { j } ^ { 2 } \log ^ { } p _ { j } ( x _ { j } )$ for a source, consistent with Section 5.2. Its curvature decomposition is

$$
\begin{array} { r l } & { H _ { j , R } = \kappa _ { j } ( x _ { j } ) + \psi _ { j , R } ( x _ { R } ) , } \\ & { \psi _ { j , R } = \displaystyle \sum _ { c \in \mathrm { C h } _ { G _ { R } } ( j ) } \mathcal { D } _ { j } ^ { 2 } \Big \{ x _ { c } \eta _ { c } \big ( x _ { \mathrm { P a } _ { G _ { R } } ( c ) } \big ) - A _ { c } \big ( \eta _ { c } \big ( x _ { \mathrm { P a } _ { G _ { R } } ( c ) } \big ) \big ) \Big \} . } \end{array}\tag{7}
$$

The child base functions do not depend on $x _ { j }$ and disappear under $\mathcal { D } _ { j } ^ { 2 }$

## B.3 MOMENT CONDITIONS

Score moments. Where the CCS is used, we assume $H _ { j , R } \in L ^ { 2 } ( P _ { R } ( \cdot \mid A _ { j } ^ { ( 2 ) } ) )$ for every evaluated ancestral set R and $j \in R .$ In the bivariate setting, this means $H _ { Z } \in L ^ { 2 } ( P ( \cdot \ | \ A _ { Z } ^ { ( 2 ) } ) )$ for $Z \in { }$ $\{ X , Y \}$ . Where the OCS is used, we assume $H _ { i j , R _ { i } } \in L ^ { 1 } ( P _ { R _ { i } } ( \cdot \mid \mathcal { A } _ { i j } ^ { ( 1 , 1 ) } ) )$ for every evaluated predecessor marginal $R _ { i }$ and $j \in { \mathrm { P r e d } } ( i )$ .

Score-entropy regression. For each positive active target $U$ , assume $\mathbb { E } [ U \log ^ { + } U ] < \infty$ , where $\log ^ { + } u = \operatorname* { m a x } \{ \log u , 0 \}$ . This moment condition holds on finite population supports, but a finite empirical rank grid alone does not ensure it.

## C PROOFS FOR SECTION 5

All derivatives, finite differences, and conditional moments below use the notation and admissibleevent convention in Appendix A.

## C.1 PROOF OF THEOREM 5

Proof. Write $F ( x , y ) = \log p ( x , y )$ . By Definition $1 , S _ { Y } = 0$ exactly when ${ \mathcal { D } } _ { Y } ^ { 2 } F ( X , Y )$ is almost surely a function of Y alone. Support positivity and smoothness extend this equality to the admissible domain, so $\mathcal { D } _ { Y } ^ { 2 } F ( x , y )$ is independent of $x .$

Fix $x _ { 0 } \in \mathring { \mathcal { X } } _ { X }$ and let $r _ { x } ( y ) = F ( x , y ) - F ( x _ { 0 } , y )$ . Since $\mathcal { D } _ { Y } ^ { 2 } r _ { x } = 0$ , a connected continuous support or a consecutive count support implies that $r _ { x }$ is affine. Thus there are functions $a ( x )$ and $\eta ( x )$ such that $r _ { x } ( y ) = a ( x ) + y \eta ( x )$ ). Define $b _ { Y } ( y ) : = \exp \{ F ( x _ { 0 } , y ) \}$ . Then

$$
\log p ( x , y ) = a ( x ) + \log b _ { Y } ( y ) + y \eta ( x ) .\tag{8}
$$

For any fixed distinct $y _ { 0 } , y _ { 1 }$ in the support interior, $\eta ( x ) = \{ r _ { x } ( y _ { 1 } ) - r _ { x } ( y _ { 0 } ) \} / ( y _ { 1 } - y _ { 0 } )$ . Thus a and $\eta$ are finite, and they are $C ^ { 2 }$ when X is continuous.

Define the log-partition function $\begin{array} { r } { A _ { Y } ( t ) = \log \int _ { \chi _ { Y } } b _ { Y } ( u ) e ^ { u t } d \nu _ { Y } ( u ) } \end{array}$ . Exponentiating (8) and integrating over y gives

$$
p _ { X } ( x ) = e ^ { a ( x ) } \int _ { \chi _ { Y } } b _ { Y } ( u ) e ^ { u \eta ( x ) } d \nu _ { Y } ( u ) = \exp \{ a ( x ) + A _ { Y } ( \eta ( x ) ) \} .
$$

The marginal density $p _ { X } ( x )$ is finite and positive for $P _ { X }$ -almost every $x .$ For each such $x ,$ the integral is therefore finite, and division by $p _ { X } ( x )$ yields

$$
p ( y \mid x ) = \frac { e ^ { a ( x ) } b _ { Y } ( y ) e ^ { y \eta ( x ) } } { e ^ { a ( x ) + A _ { Y } ( \eta ( x ) ) } } = b _ { Y } ( y ) \exp \{ y \eta ( x ) - A _ { Y } ( \eta ( x ) ) \} .
$$

This gives (3). Finiteness of $A _ { Y } ( \eta ( x ) )$ alone does not imply that $\eta ( x )$ lies in the interior of the canonical-parameter domain.

Conversely, suppose the conditional density has this form. In log $p ( x , y ) = \log p _ { X } ( x ) + \log b _ { Y } ( y ) +$ $y \eta ( x ) { - } A _ { Y } ( \eta ( x ) )$ , all terms other than log $b _ { Y } ( y )$ are constant or affine in y. Hence $\mathcal { D } _ { Y } ^ { 2 } \log p ( x , y ) =$ $\mathcal { D } _ { Y } ^ { 2 } \log b _ { Y } ( y )$ , which depends only on y, and $S _ { Y } = 0$ □

## C.2 PROOF OF THEOREM 6

Proof. In the given $X  Y$ representation, η is nonconstant and $A _ { Y } ^ { \prime \prime } > 0$ , so the conditional mean $\mathbb { E } [ Y ^ { ^ { \bullet } } | \ X = { \breve { x } } ] = A _ { Y } ^ { \prime } ( \eta ( x ) )$ ) is nonconstant. Thus X and Y are dependent, leaving only the two edge directions to compare.

First suppose $\eta ( x ) ~ = ~ \beta _ { 0 } + \beta _ { 1 } x .$ , where $\beta _ { 1 } ~ \neq ~ 0$ Use $\widetilde { b } _ { X }$ and $\widetilde { \Xi } _ { X }$ from Condition R1, and write $\begin{array} { r } { \widetilde { A } _ { X } ( t ) = \log \int _ { \mathcal { X } _ { \mathbf { Y } } } \widetilde { b } _ { X } ( x ) e ^ { t x } d \nu _ { X } ( x ) } \end{array}$ . The joint density can be written as $p ( x , y ) ~ =$ $b _ { Y } ( y ) e ^ { \beta _ { 0 } y } \widetilde { b } _ { X } ( x ) e ^ { x \beta _ { 1 } y }$ . Integrating over x gives $p _ { Y } ( y ) = b _ { Y } ( y ) \exp \{ \beta _ { 0 } y + \widetilde { A } _ { X } ( \beta _ { 1 } y ) \} .$ Dividing the joint density by this marginal density therefore gives $p ( x \mid y ) = { \widetilde { b } } _ { X } ( x ) \exp \{ x { \beta } _ { 1 } y - { \widetilde { A } } _ { X } ( \beta _ { 1 } y ) \}$

To show that the constructed $Y  X$ representation belongs to the same model class, we check its regularity. First consider canonical-parameter interiority. If Y is continuous, the integral defining $p _ { Y }$ is finite for almost every y. Around any interior support point $y ,$ choose one such finite point on each side. Holder’s inequality makes the finiteness domain¨ $\widetilde { \Xi } _ { X }$ convex, so $\beta _ { 1 } y$ lies in its interior. For count $Y _ { \textrm { \scriptsize i } }$ , the integral is finite at every support value because $p _ { Y } ( y ) \leq 1$ . Every level with neighbors on both sides therefore gives an interior parameter. Condition R1 covers the endpoints.

The reverse parameter $\widetilde { \eta } ( y ) = \beta _ { 1 } y$ is nonconstant because $\beta _ { 1 } \neq 0$ . To check smoothness, observe that

$$
\begin{array} { l } { \log \widetilde { b } _ { X } ( x ) = \log p _ { X } ( x ) - A _ { Y } ( \beta _ { 0 } + \beta _ { 1 } x ) , } \\ { \log p _ { Y } ( y ) = \log b _ { Y } ( y ) + \beta _ { 0 } y + \widetilde { A } _ { X } ( \beta _ { 1 } y ) . } \end{array}
$$

Canonical-parameter interiority and the stated smoothness make these expressions $C ^ { 2 }$ in continuous variables and finite at count values. Finally, its affine-reversibility base is $p _ { Y } ( y ) e ^ { - \widetilde { A } _ { X } ( \beta _ { 1 } y ) } =$ $b _ { Y } ( y ) e ^ { \beta _ { 0 } y }$ , with parameter domain $\Xi _ { Y } - \beta _ { 0 }$ . If X is count, then at each support endpoint $e ,$ forward canonical-parameter interiority gives $\beta _ { 0 } + \beta _ { 1 } e \in \mathrm { i n t } \Xi _ { Y } ,$ or equivalently $\dot { \beta _ { 1 } e } \in \mathrm { i n t } ( \dot { \Xi } _ { Y } - \beta _ { 0 } )$ . Thus the reverse representation also satisfies Condition R1. Taking $P _ { Y }$ as the source therefore gives a reverse representation in the same model class. The direction is consequently not identifiable when η is affine.

Conversely, suppose a reverse semiparametric GLM representation exists, with canonical-parameter function $\widetilde { \eta } .$ Choose distinct $y _ { 0 } , y _ { 1 }$ for which both conditional representations hold. Positivity permits any two support values for count Y. When Y is continuous, Fubini’s theorem gives a full-measure set of valid values, from which two distinct points can be chosen. The forward representation and Bayes’ rule give

$$
\log p ( x \mid y _ { 1 } ) - \log p ( x \mid y _ { 0 } ) = ( y _ { 1 } - y _ { 0 } ) \eta ( x ) + C ,
$$

where C is independent of x. The reverse representation expresses the same difference as $x \{ \widetilde { \eta } ( y _ { 1 } ) -$ $\widetilde { \eta } ( y _ { 0 } ) \} + C ^ { \prime }$ , where $C ^ { \prime }$ is also independent of $x .$ Equating the two expressions and dividing by $y _ { 1 } - y _ { 0 } \neq 0$ shows that $\eta ( x )$ is affine almost everywhere. Support positivity and smoothness extend this equality to the support interior. Therefore a reverse representation exists exactly when η is affine, proving the result. □

## C.3 PROOF OF THEOREM 8

Proof. (I) Sink characterization. Fix an ancestral remaining set $R \subseteq V$ and a node $j \in R$ . Write $P _ { R } ^ { ( j ) } = P _ { R } ( \cdot \mid \mathcal { A } _ { j } ^ { ( 2 ) } )$ , which equals $P _ { R }$ when $X _ { j }$ is continuous. All probabilities, expectations, and variances in this part use $P _ { R } ^ { ( j ) }$ , with the convention in Appendix A. We first establish

$$
j \mathrm { i s  a s i n k o f } G _ { R } \quad \Longleftrightarrow \quad H _ { j , R } ( X _ { R } ) = h _ { j , R } ( X _ { j } ) \quad P _ { R } ^ { ( j ) } \mathrm { - a . s . \ f o r \ s o m e \ m e a s u r a b l e \ } h _ { j , R } . \quad ( 9 )
$$

If j is a sink of $G _ { R } , ( 7 )$ gives $H _ { j , R } = \kappa _ { j } ( X _ { j } )$ , proving the forward implication in (9) with $h _ { j , R } = \kappa _ { j }$ For the reverse implication, we prove the contrapositive. Suppose $j$ is a nonsink, and choose a sink $c ^ { \star }$ in the subgraph induced by $\operatorname { C h } _ { { G } _ { R } } ( j )$ . Such a node exists because this subgraph is a nonempty finite DAG. Then $c ^ { \star }$ is not a parent of any other child of $j ,$ so only its own child contribution in (7) depends on $X _ { c ^ { \star } }$ . Let $\bar { U } = \bar { X _ { R \backslash \{ c ^ { \star } \} } }$ . All parents of $c ^ { \star }$ , including $j ,$ , belong to $R \backslash \{ c ^ { \star } \}$ . Define the function $a ( u ) : = ( \mathcal { D } _ { j } ^ { 2 } \eta _ { c ^ { \star } } ) \big ( u _ { \mathrm { P a } _ { G _ { R } } ( c ^ { \star } ) } \big )$ . Thus $\eta _ { c ^ { \star } } ( X _ { \mathrm { P a } _ { G _ { R } } ( c ^ { \star } ) } )$ and its log-partition term are functions of $U$ alone. Since $\mathcal { D } _ { j } ^ { 2 }$ leaves $X _ { c ^ { \star } }$ fixed, linearity gives $H _ { j , R } = a ( U ) X _ { c ^ { \star } } + B ( U )$ , where $B ( U )$ collects all terms independent of $X _ { c ^ { \star } }$ . Definition 4 gives an admissible u with $a ( u ) \neq 0$ . Support positivity and smoothness then give $\mathbb { P } \{ a ( U ) \neq 0 \} \ > \ 0$ . Support positivity also makes $X _ { c ^ { \star } } \ | \ U$ nondegenerate, including under the admissible-event restriction. Hence $H _ { j , R } \mid U$ is nondegenerate on $\{ a ( \boldsymbol { U } ) \ne 0 \}$ . If $H _ { j , R }$ were a function of $X _ { j }$ alone, it would be fixed given $U .$ , since $U$ contains $X _ { j }$ . This contradiction proves the reverse implication in (9).

For a nonsink, $H _ { j , R } \in L ^ { 2 }$ and the identity $X _ { c ^ { \star } } = \{ H _ { j , R } - B ( U ) \} / a ( U )$ give a finite conditional second moment for $X _ { c ^ { \star } }$ given U on $\{ a ( \dot { U } ) \ne 0 \}$ . The conditional nondegeneracy above therefore implies $\operatorname { V a r } ( H _ { j , R } \mid U ) { \mathit { \Sigma } } = a ( U ) ^ { 2 } \operatorname { V a r } ( X _ { c ^ { \star } } ^ { \prime } \mid { \mathit { \hat { U } } } ) > 0$ on this event. Since U includes $X _ { j }$ , the conditional law of total variance gives

$$
0 < \operatorname { \mathbb { E } } \{ \operatorname { V a r } ( H _ { j , R } \mid U ) \} \leq \operatorname { \mathbb { E } } \{ \operatorname { V a r } ( H _ { j , R } \mid X _ { j } ) \} = S _ { j } ( R ) < \infty .
$$

Thus every nonsink has positive CCS, and every sink has zero CCS.

(II) Parent recovery. Fix a causal order, a node $i ,$ and a predecessor $j \in \mathrm { P r e d } ( i )$ Write $R _ { i } \ =$ {i} ∪ Pred(i). Expectations below use the admissible-event convention for $\mathcal { A } _ { i j } ^ { ( 1 , 1 ) }$

The predecessor set contains all parents of i and no descendants of i. The local Markov property therefore gives $p ( x _ { i } \mid x _ { \mathrm { P r e d } ( i ) } ) = p _ { i } ( x _ { i } \mid x _ { \mathrm { P a } _ { G } ( i ) } )$ . Since log $p _ { R _ { i } } = \log p ( x _ { i } \mid x _ { \mathrm { P r e d } ( i ) } ) +$ log $p _ { \mathrm { P r e d } ( i ) }$ and the second term is independent of $x _ { i }$ , we have $H _ { i j , R _ { i } } = \mathcal { D } _ { i } \mathcal { D } _ { j } \log p _ { i } ( x _ { i } \mid x _ { \mathrm { P a } _ { G } ( i ) } )$ If i is a source, its conditional density equals its marginal density $p _ { i } ( x _ { i } )$ , so this off-diagonal curvature is zero for every predecessor, as required.

For a non-source, applying $\mathcal { D } _ { i }$ to the log conditional density in (3) gives $\mathcal { D } _ { i } \log { b _ { i } ( x _ { i } ) } + \eta _ { i } \big ( x _ { \mathrm { P a } _ { G } ( i ) } \big )$ The mixed operators commute on the admissible domain. Applying $\mathcal { D } _ { j }$ therefore removes the first term and yields

$$
H _ { i j , R _ { i } } = \mathcal { D } _ { j } \eta _ { i } , \qquad \mathrm { O C S } _ { j \to i } = \mathbb { E } \big [ | \mathcal { D } _ { j } \eta _ { i } ( X _ { \mathrm { P a } _ { G } ( i ) } ) | \big ] .
$$

$\operatorname { I f } j$ is not a parent, $\eta _ { i }$ is independent of $x _ { j }$ . For a parent, the nonconstant dependence in Definition 3 gives a point where $\mathcal { D } _ { j } \eta _ { i } \neq 0 ;$ ; support positivity and smoothness give positive probability to this event. Hence

$$
j \in \mathrm { P a } _ { G } ( i ) \quad \Longleftrightarrow \quad \mathbb { P } \Big ( | H _ { i j , R _ { i } } | > 0 \Big | \mathcal { A } _ { i j } ^ { ( 1 , 1 ) } \Big ) > 0 .\tag{10}
$$

Since $H _ { i j , R _ { i } } \in L ^ { 1 }$ , the OCS is finite and positive exactly for parents. For example, $\eta _ { i } ( x _ { j } ) = a + b x _ { j }$ with $b \neq 0$ gives $\mathrm { O C S } _ { j  i } = | b | > 0$ , so nonlinearity is not needed in this part. 口

## C.4 PROOF OF COROLLARY 9

Proof. Suppose that P has nonlinear semiparametric GLM DAG representations on both G and $\widetilde { G }$ For any remaining set R that is ancestral for both representations, the right-hand side of (9) depends only on $P _ { R }$ . It therefore identifies the same sink set in $G _ { R }$ and $\widetilde { G } _ { R }$ . Starting from $R = V$ , choose a common sink using a fixed tie-breaking rule and remove it. The new set remains ancestral for both graphs, so induction gives a single ordering π compatible with both DAG representations.

For each node $i , R _ { i } = \{ i \} \cup \mathrm { P r e d } ( i )$ is ancestral for both representations, so (10) applies to both graphs. Its right-hand side depends only on $P _ { R _ { 1 } }$ and hence recovers the same parent set for i. Thus $G = \widetilde G$ . Both (9) and (10) were established without moment conditions. □

## D ADDITIONAL IDENTIFIABILITY RESULTS

## D.1 STRICT EXTENSION OF SCORE

Corollary 13 (Strict extension of SCORE). SCORE’s identifiable nonlinear Gaussian ANMs form a proper subclass ofthe continuous nonlinear semiparametric GLM DAGs identified by Corollary 9. For a nonlinear semiparametric GLM DAG with constant own-factor curvature at every node, including these Gaussian $A N M s ,$ every ancestral remaining set R and every $j \in R s a t i s f y$

$$
\mathrm { V a r } ( H _ { j , R } ) = 0 \quad \Longleftrightarrow \quad S _ { j } ( R ) = 0 \quad \Longleftrightarrow \quad j i s a s i n k o f { G } _ { R } .
$$

Proof. For every non-source node in a Gaussian ANM, choose

$$
b _ { i } ( x ) = ( 2 \pi \tau _ { i } ^ { 2 } ) ^ { - 1 / 2 } e ^ { - x ^ { 2 } / ( 2 \tau _ { i } ^ { 2 } ) } , A _ { i } ( t ) = \frac { \tau _ { i } ^ { 2 } t ^ { 2 } } { 2 } .
$$

Then $\eta _ { i } = f _ { i } / \tau _ { i } ^ { 2 }$ , so positive scaling preserves the nonlinearity condition and Corollary 9 applies under the regularity conditions in Appendix B.2. Gaussian source marginals are also allowed, and every node has constant own-factor curvature $\kappa _ { i } ( x _ { i } ) = - 1 / \tau _ { i } ^ { 2 }$

With constant own-factor curvature, a sink has $H _ { j , R } \ = \ \kappa _ { j } ( X _ { j } )$ , so both scores are zero. At a nonsink, Theorem 8(i) and the law of total variance give $\mathrm { V a r } ( H _ { j , R } ) \ge S _ { j } ( R ) > 0$

To see that the inclusion is strict, let $X _ { 1 } \sim \mathrm { E x p } ( 1 )$ and

$$
X _ { 2 } \mid X _ { 1 } = x \sim \operatorname { G a m m a } ( \alpha , { \mathrm { s c a l e } } = 1 + x ) , \qquad \alpha > 4 .
$$

This conditional has

$$
b _ { 2 } ( y ) = \frac { y ^ { \alpha - 1 } } { \Gamma ( \alpha ) } , \qquad \eta _ { 2 } ( x ) = - \frac { 1 } { 1 + x } , \qquad A _ { 2 } ( t ) = - \alpha \log ( - t ) .
$$

Here $\eta _ { 2 }$ is nonlinear, while $\kappa _ { 2 } ( y ) = - ( \alpha - 1 ) / y ^ { 2 }$ is nonconstant. Hence $S _ { 2 } = 0$ but $\mathrm { V a r } ( H _ { 2 } ) > 0 ;$ $\alpha > 4$ ensures a finite second curvature moment. The CCS therefore identifies a model for which SCORE’s constant-curvature criterion fails. □

## D.2 LINK–INDEX CHARACTERIZATION

For a conditional family–link–index parameterization of $Y \mid X$ , let $g _ { Y , c } : = ( A _ { Y } ^ { \prime } ) ^ { - 1 }$ be the canonical link, set $h : = g _ { Y , c } \circ g _ { Y } ^ { - 1 }$ , and write $I _ { f } : = f ( \mathring { \mathcal { X } } _ { X } )$ . Then the canonical-parameter function is $\eta = h \circ f$ For any link $^ { g , }$ its affine class is $[ g ] : = \{ \alpha \circ g$ : α is increasing affine}. Thus a canonical-class link is a member of $\left[ g _ { Y , c } \right]$

Corollary 14 (Link–index characterization). In the setting of Theorem 6, its characterization becomes

X → Y identifiable ⇐⇒ h ◦ f is nonlinear on $\mathring { \mathcal { X } } _ { X }$

(i) If $g _ { Y } ~ \in ~ [ g _ { Y , c } ]$ , then h is affine and $\eta = \lambda f + \mu$ for some $\lambda \neq 0 .$ . Hence $X  Y$ is identifiable ifand only iff is nonlinear on $\breve { \mathscr { X } } _ { X }$

(ii) $I f g _ { Y } \notin [ g _ { Y , c } ] .$ , then h is nonlinear on itsfull domain, but its restriction to $I _ { f }$ may be affine. For a nonconstant affine index $f ( x ) = \alpha + \beta x ,$ , the direction is identifiable exactly when h is nonlinear on $I _ { f } = \alpha + \beta \stackrel { \circ } { \mathcal { X } } _ { X }$ . For a nonlinear index, the directionfails to be identifiable exactly when $f = h ^ { - 1 } \circ$ a on $\breve { \mathscr { X } } _ { X }$ for an affine function a whose values lie in $h ( I _ { f } )$

Every unidentifiable case also has a canonical-link representation with an affine index.

Proof. Theorem 6 reduces the problem to determining when $\eta = h \circ f$ is affine on $\mathring { \mathcal { X } } _ { X } . \mathrm { I f } f = h ^ { - 1 }$ ◦a for an affine function $^ { a , }$ , then $\eta = a$ is affine. Conversely, suppose that $h \circ f = a$ for an affine a. The map h is strictly increasing on $I _ { f }$ , so its inverse is well defined on $h ( I _ { f } )$ . Applying this inverse pointwise yields $f = h ^ { - 1 } \circ a$ on $\mathring { \mathcal { X } } _ { X }$

For clause $( \mathrm { i } ) , g _ { Y } \in [ g _ { Y , c } ]$ holds exactly when h is affine. Therefore, $\eta = \lambda f + \mu$ with $\lambda \neq 0$ , and η is nonlinear exactly when f is nonlinear.

For clause (ii), first consider a nonconstant affine index $f ( x ) = \alpha + \beta x$ with $\beta \neq 0 .$ . It maps $\mathring { \mathcal { X } } _ { X }$ bijectively and affinely onto $I _ { f } = \alpha + \beta \mathring { \mathcal { X } } _ { X }$ . Consequently, $h \circ f$ is affine on $\overset { \circ } { \mathcal { X } } _ { X }$ exactly when h is affine on $I _ { f }$ . For a nonlinear index, the preceding inverse argument shows that the exceptions are exactly the functions $f = h ^ { - 1 }$ ◦ a with a affine. Condition R1 supplies the reverse representation in the affine case. Taking $g _ { Y } ~ = ~ g _ { Y , c }$ then makes $f = \eta$ affine as well, which proves the final statement. □

## E PROOFS FOR SECTION 6

## E.1 PROOF OF PROPOSITION 10

Proof. Let $S = V \backslash R$ and choose a regular conditional distribution of $X _ { S }$ given $X _ { R }$ . For $P _ { R }$ -almost every $x _ { R }$ with $p _ { R } ( x _ { R } ) > 0$ , Bayes’ rule gives

$$
\begin{array} { r l } & { \mathbb { E } [ s _ { j , y } ^ { V } ( X _ { V } ) \mid X _ { R } = x _ { R } ] = \displaystyle \int \frac { p _ { V } ( x _ { R } ^ { j  y } , x _ { S } ) } { p _ { V } ( x _ { R } , x _ { S } ) } \frac { p _ { V } ( x _ { R } , x _ { S } ) } { p _ { R } ( x _ { R } ) } d \nu _ { S } ( x _ { S } ) } \\ & { \quad \quad \quad \quad = \frac { p _ { R } ( x _ { R } ^ { j  y } ) } { p _ { R } ( x _ { R } ) } = s _ { j , y } ^ { R } ( x _ { R } ) . } \end{array}
$$

Here $y$ is a neighboring value in $\chi _ { j }$ , and $\nu _ { S }$ is the product of the dominating measures on the removed coordinates. This proves the stated identity.

It remains to verify integrability for the neighboring entries used by DISCO. Fix $\delta \in \{ - 1 , + 1 \}$ For $A \in \{ R , V \}$ , define the zero-extended entry

$$
\begin{array} { r } { \bar { s } _ { j } ^ { A , \delta } ( x _ { A } ) = \left\{ { \begin{array} { l l } { s _ { j , x _ { j } + \delta } ^ { A } ( x _ { A } ) , } & { x _ { j } + \delta \in \mathcal { X } _ { j } , } \\ { 0 , } & { x _ { j } + \delta \notin \mathcal { X } _ { j } . } \end{array} } \right. } \end{array}
$$

The support event depends only on $x _ { j }$ , so the preceding identity implies $\bar { s } _ { j } ^ { R , \delta } ( X _ { R } ) = \mathbb { E } [ \bar { s } _ { j } ^ { V , \delta } ( X _ { V } )$ $X _ { R } ]$ . Moreover,

$$
\mathbb { E } [ \bar { s } _ { j } ^ { V , \delta } ( X _ { V } ) ] = \sum _ { x _ { j } : x _ { j } + \delta \in \mathcal { X } _ { j } } \int p _ { V } ( x _ { j } + \delta , x _ { - j } ) d \nu _ { - j } ( x _ { - j } ) \le 1 .
$$

Thus these zero-extended neighboring entries belong to $L ^ { 1 }$

## E.2 PROOF OF THEOREM 11

For count-valued P, let ${ \mathcal { A } } ( G )$ be the nonempty ancestral remaining sets of $G ,$ , and write $H _ { j j , R } : =$ $H _ { j , R }$ . For $R \in { \mathcal { A } } ( G )$ and $i , j \in R$ , use the events from Appendix A: $A _ { j j , R } : = \mathcal { A } _ { j } ^ { ( 2 ) }$ and $A _ { i j , R } : =$ $\mathcal { A } _ { i j } ^ { ( 1 , 1 ) }$ for $i \neq j . \operatorname { S e t } Q _ { i j , R } : = P _ { R } ( { } \cdot { } | \ A _ { i j , R } )$ ; the support assumptions give $P _ { R } ( A _ { i j , R } ) > 0$

Training and evaluation samples. Let $\mathcal { F } _ { n }$ contain the training sample of size $n$ , the fitting randomness, and a finite nonempty noise grid ${ \mathcal G } _ { \sigma , n } \subset ( 0 , \infty )$ . For every $\sigma \in { \mathcal G } _ { \sigma , n }$ , the noise-specific estimated curvatures $\widehat { H } _ { i j , R } ^ { ( n ) } ( \cdot ; \boldsymbol { \sigma } )$ are ${ \mathcal { F } } _ { n }$ -measurable. They use ordinary, unclipped shifts and are evaluated only on $A _ { i j , R }$ . The evaluation observations $X ^ { ( 1 ) } , \ldots , X ^ { ( m _ { n } ) }$ are iid from $P _ { - }$ , independent of $\mathcal { F } _ { n }$ , with $m _ { n } $ ∞ as $n \to \infty$

Estimators and graph threshold. For parent selection, let

$$
\mathcal T ( G ) : = \{ ( R , i , j ) : R \in \mathcal A ( G ) , \ i \mathrm { i s } \ a \mathrm { s i n k } \ o f G _ { R } , \ j \in R \backslash \{ i \} \} ,
$$

and write $O _ { j  i } ( R ) : = \mathbb { E } _ { Q _ { i j , R } } | H _ { i j , R } |$ . For each triple, some causal order places $R \setminus \{ i \}$ first and i next, so Theorem 8(ii) applies. We use the $L ^ { 2 }$ and $L ^ { 1 }$ moment conditions of Appendix B.3 for all diagonal entries with $R \in { \mathcal { A } } ( G ) , j \in R ,$ , and all triples in ${ \mathcal { T } } ( G )$ , respectively.

Write $\begin{array} { r } { \overline { { H } } _ { i j , R } ^ { ( n ) } : = | \mathcal { G } _ { \sigma , n } | ^ { - 1 } \sum _ { \sigma \in \mathcal { G } _ { \sigma , n } } \widehat { H } _ { i j , R } ^ { ( n ) } ( \cdot ; \sigma ) } \end{array}$ . Let ${ \widehat { S } } _ { j , n } ( R )$ be the grouped estimator (5) applied to $\overline { { H } } _ { j j , R } ^ { ( n ) }$ on $A _ { j j , R }$ , with fixed $n _ { \mathrm { m i n } } = k \geq 2$ and unbiased within-group sample variances. Let ${ \widehat { O } } _ { j  i , n } ( R )$ use the averaging rule in (6) for marginal R and noise grid ${ \mathcal { G } } _ { \sigma , n } ,$ restricting observations to $A _ { i j , R }$ and using the evaluation sample as one fold; absolute values are taken before noise averaging. Empty evaluation averages and maxima are zero. Given $\widehat { \pi } _ { n }$ , write $\widehat { R } _ { i , n } : = \{ i \} \cup \mathrm { P r e d } _ { \widehat { \pi } _ { n } } ( i )$ The graph $\widehat { G }$ includes $j \to i$ exactly when $j \in \mathrm { P r e d } _ { \widehat { \pi } _ { n } } ( i )$ and $\widehat { O } _ { j  i , n } ( \widehat { R } _ { i , n } ) > \lambda _ { n }$ . The procedure may be completed arbitrarily outside ${ \mathcal { A } } ( G )$

Assumption 15 (Marginal curvature estimation). Under the setup above, the noise-specific fitted marginal curvatures converge uniformly over the evaluation noise grid to the clean curvatures:

$$
\varepsilon _ { n } : = \operatorname* { m a x } _ { \stackrel { R \in \mathcal { A } ( G ) , ~ i , j \in R } { \sigma \in \mathcal { G } _ { \sigma , n } } } \big \| \widehat { H } _ { i j , R } ^ { ( n ) } ( \cdot ; \sigma ) - H _ { i j , R } \big \| _ { L ^ { 2 } ( Q _ { i j , R } ) } \stackrel { p } { \to } 0 ,\tag{11}
$$

with the error norms finite almost surely.

Proof. We first show

$$
\operatorname* { m a x } _ { R \in \mathcal { A } ( G ) } \operatorname* { m a x } _ { j \in R } | \widehat { S } _ { j , n } ( R ) - S _ { j } ( R ) | \overset { p } {  } 0 ,\tag{12}
$$

$$
\operatorname* { m a x } _ { ( R , i , j ) \in \mathcal { T } ( G ) } | \widehat { O } _ { j  i , n } ( R ) - O _ { j  i } ( R ) | \stackrel { p } {  } 0 .\tag{13}
$$

Step 1: Consistency of the CCS estimator. For the CCS, fix $R , j$ and write $Q = Q _ { j j , R } , Z = X _ { j }$ $f = H _ { j j , R } \in L ^ { 2 } ( Q )$ , and $\mathsf { S } _ { N } ( g )$ for the grouped estimator (5) using curvature $g$ on the $N$ admissible observations. Here $N  \infty$ in probability, and conditional on N these observations are iid from $Q .$ . Let $N _ { v }$ be the count at level v and $\begin{array} { r } { D _ { N } = \sum _ { v } N _ { v } { \bf 1 } \{ N _ { v } \ge k \} } \end{array}$ . For any finite set $F$ of positiveprobability levels, each level is eventually retained, giving lim inf $\dot { \phantom { } _ { N } } D _ { N } / N \geq Q ( Z \in \bar { F } )$ almost surely. Taking F to the countable support yields $\bar { D _ { N } } / \bar { N } \overset { \cdot } {  } 1$ ; this also holds in probability at the random admissible count.

The contribution of levels in $\textit { F t o } { \sf S } _ { N } ( f )$ converges to $\begin{array} { r } { \sum _ { v \in F } Q ( Z = v ) \operatorname { V a r } _ { Q } ( f \mid Z = v ) } \end{array}$ . On $D _ { N } > 0$ , the remaining contribution is bounded by

$$
\frac { c _ { k } } { D _ { N } } \sum _ { a = 1 } ^ { N } f ( X _ { R } ^ { ( a ) } ) ^ { 2 } { \bf 1 } \{ Z _ { a } \notin F \} , \qquad c _ { k } : = \frac { k } { k - 1 } \le 2 .
$$

By the law of large numbers and $D _ { N } / N \to 1$ , this bound converges to $c _ { k } \mathbb { E } _ { Q } [ f ^ { 2 } { \bf 1 } \{ Z \notin F \} ]$ , which tends to zero as $\breve { F }$ increases. The population tail has the same bound, so $\mathsf { S } _ { N } \mathsf { \bar { ( } f ) } \to _ { p } \mathsf { \bar { E } } _ { Q } [ \mathsf { \bar { V } a r } _ { Q } ( f \mid$ $Z ) ] = S _ { j } ( R )$

Now put $f _ { n } = \overline { { H } } _ { j j , R } ^ { ( n ) }$ and $h _ { n } = f _ { n } - f .$ The triangle inequality and Assumption 15 give

$$
\begin{array} { r l } & { \| h _ { n } \| _ { L ^ { 2 } ( Q ) } \leq \displaystyle \frac { 1 } { | { \mathcal G } _ { \sigma , n } | } \sum _ { \sigma \in { \mathcal G } _ { \sigma , n } } \big \| \widehat H _ { j j , R } ^ { ( n ) } ( \cdot ; \sigma ) - H _ { j j , R } \big \| _ { L ^ { 2 } ( Q ) } } \\ & { \qquad \leq \varepsilon _ { n } . } \end{array}
$$

Since $\sqrt { \mathsf { S } _ { N } ( \cdot ) }$ is a seminorm, on $D _ { N } > 0$

$$
\big | \sqrt { \mathsf { S } _ { N } ( f _ { n } ) } - \sqrt { \mathsf { S } _ { N } ( f ) } \big | ^ { 2 } \leq \mathsf { S } _ { N } ( h _ { n } ) \leq c _ { k } \frac { N } { D _ { N } } T _ { n } , \qquad T _ { n } : = \frac { 1 } { N } \sum _ { a = 1 } ^ { N } h _ { n } ( X _ { R } ^ { ( a ) } ) ^ { 2 } ,
$$

with $T _ { n } = 0 { \mathrm { ~ i f ~ } } N = 0$ . Independence gives $\mathbb { E } [ T _ { n } \ | \ \mathcal { F } _ { n } ] \ \leq \ \varepsilon _ { n } ^ { 2 }$ . For every $t > 0$ , conditional Markov’s inequality gives

$$
\operatorname* { P r } ( T _ { n } > t ) \leq \mathbb { E } [ \operatorname* { m i n } \{ 1 , \varepsilon _ { n } ^ { 2 } / t \} ] \longrightarrow 0 ,
$$

because the bounded integrand converges to zero in probability. Together with $D _ { N } / N \to 1$ , this proves ${ \widehat { S } } _ { j , n } ( R ) \to _ { p } S _ { j } ( R )$ and, by finiteness of the index set, (12).

Step 2: Consistency of the OCS estimator. For the OCS, fix $( R , i , j ) \in { \mathcal { T } } ( G )$ and let $\widetilde { O } _ { j  i , n } ( R )$ average $| H _ { i j , R } |$ on the same admissible observations. The $L ^ { 1 } ( Q _ { i j , R } )$ moment and the law of large numbers give $\widetilde { O } _ { j  i , n } ( R )  _ { p } O _ { j  i } ( R )$ , while

$$
\begin{array} { r l } & { \mathbb { E } \Big [ \vert \widehat { O } _ { j  i , n } ( R ) - \widetilde { O } _ { j  i , n } ( R ) \vert \mid \mathcal { F } _ { n } \Big ] } \\ & { \quad \leq \frac { 1 } { \vert \mathcal { G } _ { \sigma , n } \vert } \displaystyle \sum _ { \sigma \in \mathcal { G } _ { \sigma , n } } \| \widehat { H } _ { i j , R } ^ { ( n ) } ( \cdot ; \sigma ) - H _ { i j , R } \big \| _ { L ^ { 1 } ( Q _ { i j , R } ) } } \\ & { \qquad \leq \varepsilon _ { n } . } \end{array}
$$

The first inequality uses $\left| \left| a \right| - \left| b \right| \right| \leq \left| a - b \right|$ before averaging over noise levels; the second uses the $L ^ { 2 }$ error bound. The same conditional Markov argument and finiteness of ${ \mathcal { T } } ( G )$ yield (13).

Step 3: Recovery of a causal order. By Theorem $8 ( \mathrm { i } )$ , sinks have zero CCS and nonsinks have positive CCS on every ancestral set. If nonsinks exist, their scores have a positive minimum $\Delta$ over this finite collection. When all CCS errors are below $\Delta / 3$ , every empirical minimizer is a sink. Starting from $V ,$ , each removal therefore preserves ancestrality and yields a causal order. By (12), this event has probability tending to one. If no nonsinks exist, every order is valid.

Step 4: Recovery of the DAG. For a nonparent triple in ${ \mathcal { T } } ( G )$ , Theorem 8(ii) gives $H _ { i j , R } = 0$ $Q _ { i j , R }$ -almost surely, so $\widetilde { O } _ { j  i , n } ( R ) = 0$ . Hence

$$
\begin{array} { r } { \operatorname* { P r } \{ \widehat { O } _ { j  i , n } ( R ) > \lambda _ { n } \} \leq \mathbb { E } [ \operatorname* { m i n } \{ 1 , \varepsilon _ { n } / \lambda _ { n } \} ] \longrightarrow 0 . } \end{array}
$$

For a true parent, $O _ { j \to i } ( R ) > 0 , \operatorname { s o } \left( 1 3 \right)$ and $\lambda _ { n } \to 0$ give selection with probability tending to one. A union bound over ${ \mathcal { T } } ( G )$ , together with the correct-order event, proves $\operatorname* { P r } \{ { \widehat { G } } = G \} \to 1$ □

Remark (Scope of the consistency result). Consistency of common-grid clipping and median/IQR standardization is not established here. Applying the theorem to fitted grids requires control of rank and truncation errors, and Assumption 15 requires any evaluation-noise bias to vanish.

## E.3 PROOF OF PROPOSITION 12

Proof. Let $Y _ { j } = \phi _ { j } ( X _ { j } )$ with $\phi _ { j }$ strictly increasing, and write the ordered support as $v _ { j , 0 } < v _ { j , 1 } <$ $\cdot \cdot \cdot$ . Strict monotonicity gives $\ \bar { T } _ { j } ^ { Y } ( \phi _ { j } ( v _ { j , k } ) ) \ = \ k \ = \ T _ { j } ^ { X } ( v _ { j , k } )$ for every support value. Thus $T ^ { Y } ( Y ) = T ^ { X } ( X )$ almost surely, so their marginal CCS and OCS values, and hence their causal orders and parent sets, coincide. The fitted transform also depends only on this order, including its clipping and between-value rules. It therefore produces identical transformed training and evaluation inputs, proving the fixed-seed claim. □

## F THEORETICAL PROPERTIES OF CONCRETE-SCORE ESTIMATION

Fix a remaining set $R$ and a noise level $\sigma > 0$ . Starting from $x _ { 0 } \sim p _ { R }$ , corrupt the coordinates independently with kernels $q _ { \sigma } ^ { j } ( b \mid a ) \ : = \ : [ \exp ( \sigma Q ^ { j } ) ] _ { a b }$ , where $Q ^ { j }$ is a continuous-time Markovchain rate matrix. We assume $q _ { \sigma } ^ { j } ( b \mid a ) > 0$ for every pair of states $a , b$ in the coordinate state space and every positive noise level used below. The finite-grid reflecting birth–death kernel used in DISCO satisfies this condition. The resulting transition probability and noisy marginal are

$$
p _ { \sigma | 0 } ( x _ { R } \mid x _ { 0 } ) = \prod _ { j \in R } q _ { \sigma } ^ { j } ( x _ { j } \mid x _ { 0 j } ) , \qquad p _ { R , \sigma } ( x _ { R } ) = \sum _ { x _ { 0 } } p _ { \sigma | 0 } ( x _ { R } \mid x _ { 0 } ) p _ { R } ( x _ { 0 } ) .
$$

The corresponding noisy concrete score is $s _ { j , y } ^ { R } ( x _ { R } ; \sigma ) = p _ { R , \sigma } ( x _ { R } ^ { j  y } ) / p _ { R , \sigma } ( x _ { R } ) .$

## F.1 DENOISING SCORE-ENTROPY OBJECTIVE

For $u , v > 0$ , the score-entropy loss is

$$
\ell _ { \mathrm { S E } } ( u , v ) = u - v + v \log ( v / u ) .\tag{14}
$$

Denoising objective. Using the corruption and noisy concrete-score notation above, the countstructured loss below is the birth–death restriction of the denoising score-entropy objective of Lou et al. (2024). Each $Q ^ { j }$ has nonnegative off-diagonal entries and zero row sums. We estimate each non-self entry $y \neq x _ { j }$ by $\left[ s _ { \theta } ( x _ { R } ; \bar { \sigma } ) \right] _ { j , y } :$ , fix the self entry to 1, and minimize

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { { D S E } } } ( \theta ) = \mathbb { E } _ { t } \mathbb { E } _ { x _ { 0 } \sim p _ { R } } \mathbb { E } _ { x _ { R } \sim p _ { \sigma | 0 } ( \cdot | x _ { 0 } ) } \displaystyle \sum _ { j \in R } \sum _ { y \neq x _ { j } } w _ { x _ { j } y } } \\ & { \qquad \quad \times \dot { \sigma } ( t ) \ell _ { \mathrm { S E } } \Big ( \big [ s _ { \theta } ( x _ { R } ; \sigma ) \big ] _ { j , y } , \rho _ { j , y } ^ { x _ { 0 } } \Big ) , } \end{array}\tag{15}
$$

where $\ell _ { \mathrm { S E } }$ is defined in $( 1 4 ) , t \sim \mathrm { U n i f } ( \epsilon _ { t } , 1 )$ with $\epsilon _ { t } \in ( 0 , 1 )$ , and $\sigma = \sigma ( t ) > 0$ with $0 < \dot { \sigma } ( t ) < \infty$ on the sampled interval. We set $\rho _ { j , y } ^ { x _ { 0 } } = q _ { \sigma } ^ { j } ( y \mid x _ { 0 j } ) / q _ { \sigma } ^ { j } ( x _ { j } \mid x _ { 0 j } )$ and $w _ { x _ { j } y } = Q ^ { j } ( x _ { j } , y )$ , matching the training specification in Appendix G.2.

Population minimizer. Restrict the objective to modeled entries with $w _ { x _ { i } y } > 0$ and omit entries with zero generator rate. At a fixed sampled time t, for a corrupted state $x _ { R }$ and modeled entry $( j , y )$ with $p _ { R , \sigma } ( x _ { R } ) > 0$ and $p _ { R , \sigma } ( x _ { R } ^ { j  y } ) > 0 .$ , set $\bar { \rho } _ { j , y } ( x _ { R } ) = \mathbb { E } [ \rho _ { j , y } ^ { X _ { 0 } } \mid X _ { R } = x _ { R } ]$ ]. Expanding the loss gives, almost surely, the finite conditional risk $u - \bar { \rho } _ { j , y } ( x _ { R } )$ log $u + C$ at prediction $u > 0$ , where $C$ is finite and does not depend on $u .$ . Its derivative $1 - \bar { \rho } _ { j , y } ( x _ { R } ) / u$ shows that the unique minimizer is $u \ : = \ : \bar { \rho } _ { j , y } ( x _ { R } )$ The strictly positive generator and time weights do not alter this conditional minimizer. Kernel positivity justifies cancellation of the transition probabilities, and the product form of the corruption kernel gives

$$
\begin{array} { l } { \bar { \rho } _ { j , y } ( x _ { R } ) = \displaystyle \sum _ { x _ { 0 } } \frac { p _ { R } ( x _ { 0 } ) p _ { \sigma | 0 } ( x _ { R } \mid x _ { 0 } ) } { p _ { R , \sigma } ( x _ { R } ) } \frac { q _ { \sigma } ^ { j } ( y \mid x _ { 0 j } ) } { q _ { \sigma } ^ { j } ( x _ { j } \mid x _ { 0 j } ) } \qquad } \\ { \displaystyle = \frac { 1 } { p _ { R , \sigma } ( x _ { R } ) } \sum _ { x _ { 0 } } p _ { R } ( x _ { 0 } ) p _ { \sigma | 0 } ( x _ { R } ^ { j  y } \mid x _ { 0 } ) = \frac { p _ { R , \sigma } ( x _ { R } ^ { j  y } ) } { p _ { R , \sigma } ( x _ { R } ) } = s _ { j , y } ^ { R } ( x _ { R } ; \sigma ) . } \end{array}
$$

On the finite common corruption-model grid, both the generator weights and the number of modeled entries are bounded. These entrywise minima therefore make the corrupted concrete score the unique population minimizer of (15) on modeled entries.

## F.2 VANISHING-NOISE LIMIT

Proposition 16 (Vanishing-noise limit). Suppose $q _ { \sigma } ^ { j }$ tends to the identity as $\sigma \downarrow 0 ,$ , which holds for the finite-grid reflecting birth–death kernel used in DISCO. Then $p _ { R , \sigma }  p _ { R }$ pointwise, and for every $x _ { R }$ with $p _ { R } ( x _ { R } ) > 0$ and every $y \in \mathcal { X } _ { j }$

$$
s _ { j , y } ^ { R } ( x _ { R } ; \sigma ) \xrightarrow [ \sigma \downarrow 0 ] { } s _ { j , y } ^ { R } ( x _ { R } ) .
$$

Proof. On the countable support, $\begin{array} { r } { p _ { R , \sigma } ( x _ { R } ) = \sum _ { x _ { 0 } } p _ { \sigma | 0 } ( x _ { R } \ | \ x _ { 0 } ) p _ { R } ( x _ { 0 } ) } \end{array}$ . For each fixed $x _ { 0 } ,$ , the summand converges to ${ \bf 1 } \{ x _ { R } = x _ { 0 } \} p _ { R } ( x _ { 0 } )$ as $\sigma \downarrow 0 .$ . It is bounded by the summable function $p _ { R } ( x _ { 0 } )$ because $0 \leq p _ { \sigma | 0 } \dot { ( } x _ { R } \ | \ x _ { 0 } \rangle \leq 1$ . Dominated convergence therefore gives $p _ { R , \sigma } ( x _ { R } ) $ $p _ { R } ( x _ { R } )$ for every $x _ { R } . \operatorname { I f } p _ { R } ( x _ { R } ) > 0$ , continuity of the ratio yields

$$
s _ { j , y } ^ { R } ( x _ { R } ; \sigma ) = \frac { p _ { R , \sigma } ( x _ { R } ^ { j  y } ) } { p _ { R , \sigma } ( x _ { R } ) } \longrightarrow \frac { p _ { R } ( x _ { R } ^ { j  y } ) } { p _ { R } ( x _ { R } ) } = s _ { j , y } ^ { R } ( x _ { R } ) .
$$

## G DISCO IMPLEMENTATION

Cross-fitting. We partition the observations into evaluation folds $\mathcal { T } _ { 1 } , \ldots , \mathcal { T } _ { B }$ . For each fold, we fit the rank transform, joint concrete-score network, and projection network on the other folds. We then apply the fitted transform and networks to the held-out fold to compute the CCS and OCS.

## G.1 RANK TRANSFORM

For coordinate j, let $v _ { j , 0 } < \cdots < v _ { j , K _ { j } }$ be the distinct values observed in the score-training split and set $\widehat { T } _ { j } ( v _ { j , k } ) = k$ . Apply this fitted transform to the held-out evaluation split without using its empirical distribution. A held-out value below $v _ { j , 0 }$ is assigned rank 0, a value above $v _ { j , K _ { j } }$ is assigned rank $K _ { j } ,$ and a value strictly between $v _ { j , k }$ and $v _ { j , k + 1 }$ is assigned rank $\operatorname* { m a x } ( k , 1 )$ . The first interior gap is assigned rank 1 to keep the observed minimum as a separate state. Thus the evaluation values are clipped or coarsened onto the rank grid according to order rather than Euclidean distance.

## G.2 JOINT CONCRETE-SCORE TRAINING

Training objective. We train the joint concrete-score network once on the score-training split, using SEDD’s denoising score-entropy objective (Lou et al., 2024) in (15) with $R = V \colon$ each minibatch draws $t \sim \mathrm { U n i f } ( \epsilon _ { t } , 1 )$ , perturbs $x _ { 0 } \sim p _ { V } t o \ x \sim p _ { \sigma ( t ) | 0 } ( \cdot \ | \ x _ { 0 } )$ , and weights the loss by $\dot { \sigma } ( t )$ We use the geometric schedule $\sigma ( t ) = \sigma _ { \operatorname* { m i n } } ^ { 1 - t } \sigma _ { \operatorname* { m a x } } ^ { t }$ with $( \epsilon _ { t } , \sigma _ { \operatorname* { m i n } } , \sigma _ { \operatorname* { m a x } } ) = ( 1 0 ^ { - 3 } , 1 0 ^ { - 3 } , 3 )$ The training objective is restricted to the neighboring replacements $y = x _ { j } \pm 1$ and masks replacements outside the common corruption-model grid $\bar { \{ 0 , \ldots , K \} }$ , where $K \stackrel { \cdot } { = } \operatorname* { m a x } _ { j } K _ { j }$ . We choose a reflecting birth–death corruption on this grid for every coordinate: $Q ^ { j } ( a , b ) = \bar { 1 } \mathrm { f o r } | a - b | = 1$ zero for other off-diagonal entries, and $\begin{array} { r } { Q ^ { j } ( \bar { a } , a ) = - \sum _ { b \neq a } Q ^ { j } ( a , b ) } \end{array}$ . The common upper bound K may exceed a coordinate’s largest training rank $K _ { j } .$ , so this grid is not the coordinate’s population support.

Network architecture. The count network is a multilayer perceptron with four hidden layers of width 512 and SiLU activations, without hidden-layer normalization or dropout. Writing $\mathbf { \bar { \mathbf { K } } ^ { \vee } }$ = $\operatorname* { m a x } ( K , 1 )$ for the common grid bound, its value features are $[ x / K ^ { \vee } , \log ( 1 + x ) / \log ( 1 + K ^ { \vee } ) ]$ These features are standardized using training-split means and standard deviations, with standard deviations floored at $1 0 ^ { - 4 }$ ; log σ is then appended. The output is reshaped to $\mathbb { R } ^ { d \times 2 }$ , with the last axis indexing the two directions.

## G.3 MARGINAL CONCRETE-SCORE PROJECTION

## G.3.1 MASK DISTRIBUTION

The score-entropy estimator draws a remaining-set size uniformly from $\{ 2 , \ldots , d \}$ and then a subset uniformly conditional on that size, using one mask per training observation and epoch. Independently with probability 0.2, it replaces the sampled mask by the full set V. For $d \geq 2$ , the resulting distribution is

$$
\mathcal { Q } ( R ) = \frac { 0 . 8 } { \left( d - 1 \right) \binom { d } { | R | } } + 0 . 2 \mathbf { 1 } \{ R = V \} , \qquad 2 \leq | R | \leq d .
$$

Thus every subset of size at least two has positive sampling probability; for $d > 2 ,$ neither subsets nor final mask sizes are uniformly distributed.

## G.3.2 SCORE-ENTROPY TRAINING

Each projection update draws $X _ { 0 } \sim p _ { V }$ , samples log σ uniformly from $\left[ \log 1 0 ^ { - 3 } , \log 3 \right]$ , and corrupts the full observation to $X _ { \sigma } \sim p _ { \sigma | 0 } ( \cdot \mid X _ { 0 } )$ . The joint network evaluates $X _ { \sigma }$ without parameter updates, while the projection network receives $( X _ { \sigma , R } , R , \sigma )$ with removed coordinates zero-masked. Masks are independent of the observation, and projection training uses no σ˙ loss multiplier.

A mask-conditioned MLP with the same hidden architecture as the joint network learns a log correction to a positive baseline. The baseline is obtained by filling removed coordinates with values from one training observation sampled once and held fixed, then evaluating the joint network.

We apply the score-entropy loss in (14) to the numerically stabilized, baseline-normalized projected and joint-network scores. Log clipping and positive numerical floors are used for stability; their constants are specified in the code. We average the loss over neighboring replacements within the grid for $j \in R$ and add a full-set $( R = V )$ loss with weight 1.

## G.4 CURVATURE ESTIMATION

Evaluation. We use five geometrically spaced noise levels from $1 0 ^ { - 3 } \mathrm { t o } 5 \times 1 0 ^ { - 3 }$ in $\mathcal { G } _ { o }$ and direct log concrete-score outputs from the projection network.

We evaluate diagonal curvature only when $x _ { j } + 2 \leq K$ , so both upward log concrete-score entries correspond to transitions inside the fitted model grid. Invalid entries are excluded separately for each node, rather than set to zero or removed from every node’s evaluation sample. Counts and weights in (5) use admissible observations; a node with no repeated admissible level is reported as unavailable and stops ordering. This model-grid restriction does not identify the fitted grid with the population support.

For the grouped conditional variance (5), we use rank levels with at least $n _ { \mathrm { m i n } } = 2$ admissible evaluation observations and weight each by the number of these observations. We average the resulting scores across folds.

Symmetric off-diagonal curvature evaluation. For parent selection, we use the commutingdifference identity

$$
H _ { i j , \widehat { R } _ { i } } = \mathcal { D } _ { i } \mathcal { D } _ { j } \log p _ { \widehat { R } _ { i } } = \mathcal { D } _ { j } \mathcal { D } _ { i } \log p _ { \widehat { R } _ { i } } = H _ { j i , \widehat { R } _ { i } } .\tag{16}
$$

For each node $i ,$ we obtain the $\widehat { R } _ { i }$ -marginal concrete score for $\widehat { R } _ { i } \ : = \ : \{ i \} \cup \mathrm { P r e d } _ { \widehat { \pi } } ( i )$ , evaluate the projection network on the unshifted batch and the batch shifted in coordinate $i , x _ { \hat { R } _ { i } } + e _ { i }$ on admissible observations, and extract the entries for all predecessor coordinates $j \in \mathrm { P r e d } _ { \widehat { \pi } } ( i )$ from those two outputs. Thus the off-diagonal curvatures in (4) require only one shifted batch per node. We compute these curvatures for all predecessors simultaneously, averaging their absolute values over noise levels and admissible observations for each edge. All candidate parents use the same unshifted score predictions and remaining-set mask.

## G.5 PARENT SELECTION

We estimate OCS with $B = 3$ cross-fitting folds. We use $\begin{array} { r } { \mathcal { T } _ { b } ^ { i j } = \{ a \in \mathbb { Z } _ { b } : X _ { a i } + 1 \leq K , X _ { a j } + 1 \leq } \end{array}$ K} in (6), with denominator $| \mathcal { T } _ { b } ^ { i j } |$ for each edge. An empty set is reported as unavailable and

stops parent selection. Clipped-boundary evaluation uses $\mathcal { T } _ { b } ^ { i j } = \mathcal { T } _ { b }$ . For a node i with at least two candidate parents, we robustly standardize these estimates within each fold,

$$
Z _ { j \to i } ^ { ( b ) } = \frac { \widehat { \mathrm { O C S } } _ { j \to i } ^ { ( b ) } - \mathrm { m e d i a n } _ { k \in \mathrm { P r e d } _ { \widehat { \pi } } ( i ) } \widehat { \mathrm { O C S } } _ { k \to i } ^ { ( b ) } } { s _ { i } ^ { ( b ) } } .
$$

Here $s _ { i } ^ { ( b ) }$ is the IQR of the candidate OCS estimates, falling back successively to the scaled median absolute deviation, standard deviation, and unit scale when degenerate. We apply the selectionfrequency rule in Section 6.3.

We use $( \tau _ { Z } , \pi _ { \operatorname* { m i n } } ) = ( 2 , 1 )$ unless otherwise stated. For a single candidate, we omit centering and use the median robust scale across children with at least two candidates in the same fold. If none exist, we use the robust scale of all candidate scores in that fold. The same threshold and selectionfrequency rule still apply. Candidate sets of size two or three remain unable to pass the default threshold in exact arithmetic. This practical rule does not provide a finite-sample error-control guarantee. Appendix I.2 evaluates threshold sensitivity and compares parent-selection methods.

## H EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

## H.1 EXPERIMENTAL SETUP

Graph and sampling protocol. For count-data synthetic experiments, ERk denotes an Erdos–˝ Renyi DAG obtained from a random permutation used as the causal order, with each admissible´ earlier-to-later edge included independently with probability $2 k / d$ (expected edge count $k ( d - 1 ) ,$ . Unless stated otherwise, we use ER3 DAGs. Within each setting and seed, all methods receive the same graph and observations. For sample-size curves, smaller samples are row prefixes of the largest sample. The continuous-data comparison uses the fixed-edge-count convention specified in Appendix H.2.8.

Replication and pairing. Unless stated otherwise, we use 10 paired seeds for each synthetic experiment. The Lahman comparisons use three runs on each fixed cohort; these are computational repetitions, not independent sampled cohorts. Ablations also share learned scores whenever applicable; graph seeds are the independent replication units.

Evaluation metrics. We use $\begin{array} { r } { A _ { \mathrm { t o p } } = | \mathcal { E } | ^ { - 1 } \sum _ { ( u , v ) \in \mathcal { E } } \mathbf { 1 } \{ \widehat { \pi } ( u ) < \widehat { \pi } ( v ) \} } \end{array}$ for ordering agreement. The nonempty evaluation set E is the true DAG edge set E for synthetic data and the partial directed reference set R for real data. DAG recovery comparisons report $F _ { 1 }$ , precision, recall, and structural Hamming distance (SHD), defined as $\mathrm { F P } + \mathrm { F N }$ , so reversing a true edge counts twice.

Training settings. On count data, we train the joint network for up to 800 epochs and the projection network for 800 epochs, unless stated otherwise. The COM–Poisson experiment (Table 11), parent-selection experiments (Tables 17 and 18), and 2019–2025 Lahman DISCO analysis use 400 projection epochs. In the count-data experiments, DISCO clips upper-grid shifts to $K$ and includes boundary outputs, using all fold observations for OCS, except where admissible-domain evaluation is specified. Separate continuous-data settings are given in Appendix H.2.8.

Both networks use Adam with batch size 512. The joint and projection learning rates start at $1 0 ^ { - 3 }$ and $1 . 5 \times 1 0 ^ { - 3 }$ , respectively, and follow cosine schedules to $1 \dot { 0 } ^ { - 5 } \mathrm { a n d 3 } \times 1 0 ^ { - 5 } \ddagger$ their gradient-norm clipping thresholds are 1 and 5. Joint validation uses a 10% internal holdout (at most 4096 observations), begins at epoch 300, and runs every 100 epochs with patience 2 and minimum improvement $1 0 ^ { - 4 }$

Data generation. For non-source nodes, write the index as $\begin{array} { r } { z _ { j } \ = \ b _ { 0 } + \sum _ { k \in \mathrm { P a } _ { G } ( j ) } \beta _ { k j } \phi ( X _ { k } ) } \end{array}$ Table 5 gives the six benchmark mechanisms; $\mu _ { j }$ denotes the conditional mean and $q _ { j }$ the binomial success probability. The intercept is fixed within each mechanism. We do not standardize the generated counts, normalize by in-degree, or modify observations after generation.

Here $\widetilde { z } _ { j } = \mathrm { c l i p } ( z _ { j } , - 3 , 3 . 5 )$ limits the exponential response, and $\Phi$ is the standard normal CDF. Source means are 2 for the softplus mechanisms and 1.5 for the exponential mechanisms. Negative binomial nodes have size $r = 6 .$ Binomial nodes have size $M = 5 0$ and source probability 0.10; generated probabilities are clipped to $[ 1 0 ^ { - 8 } , 1 - 1 0 ^ { - 8 } ] .$ . Mixed-family benchmarks use the softplus/affine rows for Poisson and negative binomial children and the sigmoid/square-root row for binomial children, with the feature map determined by the child’s conditional family.

Table 5: The six single-family benchmark mechanisms. Coefficients are drawn independently and uniformly from the stated intervals.
<table><tr><td>Conditional family</td><td>Conditional response</td><td> $\phi ( x )$ </td><td> $b _ { 0 }$ </td><td>Coefficients</td></tr><tr><td>Poi</td><td> $\mu _ { j } = 0 . 2 + \mathrm { s o f t p l u s } ( z _ { j } )$   $\mu _ { j } = \exp ( \widetilde { z } _ { j } )$ </td><td> $x$   $\log ( 1 + x )$ </td><td>0.5 0.4</td><td>[0.30, 0.45] [0.15, 0.30]</td></tr><tr><td>NB</td><td> $\mu _ { j } = 0 . 2 + \mathrm { s o f t p l u s } ( z _ { j } )$   $\mu _ { j } = \exp ( \widetilde { z } _ { j } )$ </td><td>x  $\log ( 1 + x )$ </td><td>0.5 0.4</td><td>[0.30, 0.45] [0.15, 0.30]</td></tr><tr><td>Bin</td><td> $q _ { j } = \Phi ( z _ { j } )$   $q _ { j } = \mathrm { s i g m o i d } ( z _ { j } )$ </td><td> $x / 5 0$   $\sqrt { x / 5 0 }$ </td><td>-1.28 -2.197</td><td>[0.50, 1.00] [0.60, 0.90]</td></tr></table>

CCS comparison. Table 2(a) uses the softplus/affine Poisson and negative-binomial mechanisms and the sigmoid/square-root binomial mechanism in Table 5, at $d = 5 0$ and $n = 5 0 0 0$ . CCS and the constant-curvature criterion use the same fitted networks. Across 10 seeds, we evaluate nested prefixes of the true causal order with sizes 2–9, excluding prefixes in which every node is a sink; this leaves 60 matched nontrivial ancestral remaining sets per conditional family. Accuracy is the fraction of sets for which the selected node is a true sink; sets within a graph seed are not independent replicates.

For DAG recovery, each criterion orders all nodes using the same fitted networks, and the default OCS rule then selects parents. Table 6 reports the results. CCS attains higher mean $A _ { \mathrm { t o p } }$ and $F _ { 1 }$ and lower mean SHD for Poisson and negative binomial data. For binomial data, the two criteria give similar $A _ { \mathrm { t o p } }$ , and the constant-curvature criterion gives slightly higher mean $F _ { 1 }$ and lower mean SHD.

Table 6: DAG recovery with CCS and the constant-curvature criterion $( d = 5 0 , n = 5 0 0 0 )$ . Entries are means ± sample standard deviations over 10 seeds. Constant denotes the constant-curvature criterion.
<table><tr><td>Conditional family</td><td>Criterion</td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td>SHD</td></tr><tr><td rowspan="2">Poi</td><td>Constant</td><td> $0 . 9 1 5 \pm 0 . 0 2 2$ </td><td> $0 . 8 8 3 \pm 0 . 0 3 7$ </td><td> $3 1 . 5 \pm 9 . 4$ </td></tr><tr><td>CCS</td><td> $0 . 9 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 8 9 2 \pm 0 . 0 3 4$ </td><td> $2 8 . 9 \pm 8 . 7$ </td></tr><tr><td rowspan="2">NB</td><td>Constant</td><td> $0 . 9 0 7 \pm 0 . 0 1 9$ </td><td> $0 . 7 6 3 \pm 0 . 0 7 0$ </td><td> $5 7 . 9 \pm 1 5 . 5$ </td></tr><tr><td>CCS</td><td> $0 . 9 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 7 9 2 \pm 0 . 0 6 6$ </td><td> $5 1 . 2 \pm 1 5 . 4$ </td></tr><tr><td rowspan="2">Bin</td><td>Constant</td><td> $0 . 8 9 9 \pm 0 . 0 2 8$ </td><td> $0 . 8 8 4 \pm 0 . 0 2 7$ </td><td> $3 0 . 8 \pm 7 . 7$ </td></tr><tr><td>CCS</td><td> $0 . 9 0 1 \pm 0 . 0 2 7$ </td><td> $0 . 8 7 7 \pm 0 . 0 3 1$ </td><td> $3 2 . 3 \pm 8 . 4$ </td></tr></table>

## H.1.1 BASELINE METHODS

ODS. ODS (Poi) follows the QVF overdispersion ordering criterion of Park and Raskutti (2018). The multivariate experiments use regression estimates of the conditional mean and variance because exact-match conditioning cells collapse at the graph densities considered here. The fixed-family ODS variants (Poisson, negative binomial, and binomial) and ODS (Oracle) differ only in the supplied QVF coefficient and family-specific regression link. For ODS (Oracle), the fixed parameters are the negative binomial size r and binomial trial count M, giving QVF coefficients $1 / r { \mathrm { a n d - } } 1 / M$ respectively; Poisson has coefficient zero and requires no additional fixed parameter.

NOTEARS-MLP. We use the gCastle nonlinear MLP (Zhang et al., 2021) with squared loss on standardized log(1 + x) counts.

DiffAN. We use the DiffAN implementation of Sanchez et al. (2023), which includes CAM pruning (Buhlmann et al., 2014). Runtime includes diffusion ordering and parent pruning.¨

MRS. We use the MRS implementation (Park and Park, 2019a) with GES skeleton estimation. Mixed-family and scalability experiments use its Poisson setting. Single-family experiments use the Poisson and binomial settings for the corresponding families, and a hyper-Poisson proxy for negative binomial data.

## H.2 ADDITIONAL BENCHMARKS

## H.2.1 INVARIANCE TO STRICTLY INCREASING TRANSFORMATIONS

Table 7 compares DISCO with DISCO-NORANK after applying $\log ( 1 + x )$ , Anscombe’s $2 { \sqrt { x + 3 / 8 } }$ transform, and $x ^ { 2 } + x$ to the same Poisson data, generated by the softplus/affine mechanism in Table 5. With the rank transform, median $A _ { \mathrm { t o p } } = 0 . 9 2 2$ and $F _ { 1 } = 0 . 8 9 4$ remain unchanged across transformations. Without it, median $F _ { 1 }$ falls from 0.899 on raw counts to 0.301 under $x ^ { 2 } + x .$

Table 7: Invariance to strictly increasing transformations on Poisson DAGs $( d = 5 0 , n = 5 0 0 0 )$ Entries are medians. DISCO-NORANK omits the rank transform.
<table><tr><td rowspan="2">Method</td><td colspan="2">Raw counts</td><td colspan="2"> $\log ( 1 + x )$ </td><td colspan="2">Anscombe</td><td colspan="2"> $x ^ { 2 } + x$ </td></tr><tr><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td></tr><tr><td>DISCO</td><td>0.922</td><td>0.894</td><td>0.922</td><td>0.894</td><td>0.922</td><td>0.894</td><td>0.922</td><td>0.894</td></tr><tr><td>DISCO-NORANK</td><td>0.937</td><td>0.899</td><td>0.897</td><td>0.726</td><td>0.890</td><td>0.860</td><td>0.834</td><td>0.301</td></tr></table>

## H.2.2 SINGLE-FAMILY DAG RECOVERY

The binomial–probit panel uses ODS (Poi) with a Poisson QVF, a non-oracle specification for binomial data.

## H.2.3 MIXED-FAMILY DAG RECOVERY

We split a causal order into three consecutive groups of nearly equal size and assign Poisson, negative binomial, and binomial mechanisms to these groups in all six permutations. The arrows in the panel titles indicate this assignment order. Each node uses the mechanism for its assigned conditional family in Table 5. Figure 1(c) reports the NB → Bin → Poi configuration. To examine sensitivity to conditional-family specification, Figure 3 compares DISCO with ODS (Oracle), ODS (Poi), ODS (NB), and ODS (Bin) across all six configurations.

## H.2.4 SCALABILITY WITH GRAPH SIZE

Table 8 reports $A _ { \mathrm { t o p } }$ for the Poisson scalability experiments in Table 3, using the softplus/affine mechanism in Table 5 under the same 240-minute time limit.

Figure 2: Extended single-family comparison across all six mechanisms and six sample sizes. Columns correspond to the Poi, NB, and Bin conditional families; the top and bottom rows use the first and second mechanisms for each family in Table 5, respectively. Points and error bars show the median and interquartile range.  
![](images/bb1dabfbe193d02359a582d8050ddb9c5f46e9d4d82616e516cbf41d774ceac9.jpg)  
Figure 3: Mixed-family DAG recovery across six conditional-family configurations. Each panel compares DISCO with ODS (Oracle), ODS (Poi), ODS (NB), and ODS (Bin). ODS (Oracle) uses the true nodewise conditional families and their fixed parameters. Points and error bars show the median and interquartile range.

Table 8: Ordering accuracy in the scalability experiments at n = 5000. Entries are medians; config urations marked Timeout in Table 3 are omitted.
<table><tr><td>Method</td><td>d  $A _ { \mathrm { t o p } }$ </td></tr><tr><td>DISCO</td><td>100 0.937 200 0.902 500 0.914 1000 0.908</td></tr><tr><td>ODS (Oracle)</td><td>100 0.819 200 0.816 500 0.811</td></tr><tr><td>MRS</td><td>100 0.797 200 0.779 500 0.790</td></tr><tr><td>NOTEARS-MLP</td><td>100 0.541</td></tr></table>

Total time in Table 3 covers the complete DAG recovery procedure, including parent selection. MRS, NOTEARS-MLP, and DiffAN were first evaluated with a four-hour limit. Configurations finishing within this limit were evaluated on the remaining seeds. For DiffAN at $d = 1 0 0$ and $d = 2 0 0$ , the time limit was reached during parent pruning after ordering had finished.

CPU baselines use one thread, while DISCO uses GPU training and CPU post-processing. The runtime comparison is therefore descriptive and does not imply matched hardware.

## H.2.5 SPARSE DAG RECOVERY

We repeat the single-family and mixed-family designs with ER1 graphs, holding the mechanisms, coefficient distributions, and sample-size grid fixed. Table 9 reports medians for DISCO at $n =$ 5000.

Table 9: Single-family and mixed-family DAG recovery on ER1 graphs $( d = 1 0 0 , n = 5 0 0 0 )$ Entries are medians for DISCO.
<table><tr><td colspan="2">Conditional</td><td rowspan="2">Index/configuration</td><td rowspan="2"> $A _ { \mathrm { t o p } }$ </td><td rowspan="2"> $F _ { 1 }$ </td><td rowspan="2">SHD</td></tr><tr><td>Setting</td><td>family</td></tr><tr><td rowspan="4">Single- family</td><td>Poi</td><td>Affine Nonlinear</td><td>0.816 0.897</td><td>0.750 0.873</td><td>49.0 25.0</td></tr><tr><td>NB</td><td>Affine</td><td>0.841</td><td>0.788</td><td>42.5</td></tr><tr><td></td><td>Nonlinear Affine</td><td>0.907 0.886</td><td>0.884 0.860</td><td>22.0</td></tr><tr><td>Bin</td><td>Nonlinear</td><td>0.904</td><td>0.886</td><td>28.5 23.0</td></tr><tr><td rowspan="5">Mixed- family</td><td>Poi → NB → Bin Poi → Bin → NB</td><td></td><td>0.891</td><td>0.845</td><td>29.5</td></tr><tr><td>NB → Poi → Bin</td><td></td><td>0.845</td><td>0.796</td><td>44.0</td></tr><tr><td></td><td></td><td>0.810</td><td>0.777</td><td>44.0</td></tr><tr><td>NB → Bin → Poi</td><td></td><td>0.766</td><td>0.705</td><td>58.0</td></tr><tr><td>Bin → Poi → NB  $\mathrm { B i n }  \mathrm { N B }  \mathrm { P o i }$ </td><td></td><td>0.739 0.723</td><td>0.685 0.633</td><td>67.0 75.0</td></tr></table>

## H.2.6 SCALE-FREE DAG RECOVERY

We replace the ER graph generator with the Barabasi–Albert preferential-attachment generator,´ keeping d = 50 and $n = 5 0 0 0$ for each of Poisson, negative binomial, and binomial. SF1 and SF3 attach each new node to one and three existing nodes, respectively, yielding 49 and 141 edges.

A random node ordering orients the skeleton acyclically; attachment time is not supplied as a causal order. The conditional mechanisms and coefficient ranges are unchanged from the parent-selection study in Appendix I.2. Seed labels identify independent repetitions within a graph type, not identical observations across ER and SF graphs.

Table 10: Scale-free count-DAG recovery $( d = 5 0 , n = 5 0 0 0 )$ . Entries are means ± sample standard deviations.
<table><tr><td>Graph</td><td>Conditional family</td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td>SHD</td></tr><tr><td rowspan="3">SF1</td><td>Poi</td><td> $0 . 7 8 4 \pm 0 . 0 6 5$ </td><td> $0 . 7 0 5 \pm 0 . 0 9 1$ </td><td> $2 8 . 6 \pm 8 . 5$ </td></tr><tr><td>NB</td><td> $0 . 8 2 2 \pm 0 . 0 5 4$ </td><td> $0 . 7 4 1 \pm 0 . 0 9 4$ </td><td> $2 4 . 8 \pm 8 . 9$ </td></tr><tr><td>Bin</td><td> $0 . 7 7 8 \pm 0 . 0 6 0$ </td><td> $0 . 7 8 2 \pm 0 . 0 4 3$ </td><td> $2 0 . 1 \pm 4 . 0$ </td></tr><tr><td rowspan="3">SF3</td><td>Poi</td><td> $0 . 9 1 8 \pm 0 . 0 1 9$ </td><td> $0 . 7 1 9 \pm 0 . 0 7 4$ </td><td> $6 5 . 3 \pm 1 4 . 4$ </td></tr><tr><td>NB</td><td> $0 . 9 0 4 \pm 0 . 0 3 5$ </td><td> $0 . 6 2 0 \pm 0 . 1 1 4$ </td><td> $8 2 . 4 \pm 1 9 . 7$ </td></tr><tr><td>Bin</td><td> $0 . 7 6 9 \pm 0 . 0 5 0$ </td><td> $0 . 6 9 3 \pm 0 . 0 7 7$ </td><td> $7 1 . 2 \pm 1 4 . 8$ </td></tr></table>

Recovery varies with both graph structure and conditional family. In particular, the higher $A _ { \mathrm { t o p } }$ on SF3 for Poisson and negative binomial does not imply a corresponding increase in $F _ { 1 }$ . These results extend the empirical evaluation beyond ER graphs; they do not establish performance independent of graph structure or conditional family.

## H.2.7 DAG RECOVERY BEYOND STANDARD COUNT FAMILIES

We test recovery beyond the Poisson, negative binomial, and binomial benchmark conditional families using mean-parametrized Conway–Maxwell–Poisson (COM–Poisson) distributions (Huang, 2017). For a parent-independent shape $\nu > 0 .$

$$
p ( \boldsymbol { y } \mid x _ { \mathrm { P a } _ { G } ( j ) } ) = \exp \{ y \eta ( x _ { \mathrm { P a } _ { G } ( j ) } ) - \nu \log ( y ! ) - A _ { \nu } ( \eta ( x _ { \mathrm { P a } _ { G } ( j ) } ) ) \} , \qquad \boldsymbol { y } \in \mathbb { N } _ { 0 } .
$$

We use $\nu \in \{ 0 . 5 , 1 , 2 \}$ on DAGs with $d = 5 0$ and $n = 5 0 0 0 ; \nu = 1$ reduces to the Poisson distribution. Graphs and coefficients are paired by seed. At fixed parent values, all three settings share source mean 2 and non-source conditional mean $\begin{array} { r } { 0 . 2 + \mathrm { s o f t p l u s } ( 0 . 5 + \sum _ { k \in \mathrm { P a } _ { G } ( j ) } \beta _ { k j } X _ { k } ) } \end{array}$ with $\beta _ { k j } \sim \mathrm { U } ( 0 . 3 0 , 0 . 4 5 )$ . We solve for the canonical parameter to match this mean, rather than treating the usual COM–Poisson rate as its expectation. Hence conditional means are matched, while the corresponding canonical parameters and samples may differ. Sampling approximates the unbounded distribution through adaptive support expansion, with bounds on the omitted probability mass and its first-moment contribution. DISCO receives neither the conditional family nor $\nu ;$ the same configuration is used for all three ν values.

Table 11: Recovery beyond the three benchmark conditional families $( d = 5 0 , n = 5 0 0 0 )$ . Entries are means ± sample standard deviations. The $\nu = 1$ row uses results from the corresponding Poisson experiment; graph metrics include DISCO parent selection.
<table><tr><td>Conditional family</td><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td>SHD</td></tr><tr><td>COM-Poisson  $( \nu = 0 . 5 )$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 8$ </td><td> $0 . 8 5 3 \pm 0 . 0 3 3$ </td><td> $3 7 . 8 \pm 8 . 5$ </td></tr><tr><td>COM–Poisson (ν = 1)</td><td> $0 . 9 2 8 \pm 0 . 0 2 0$ </td><td> $0 . 8 7 7 \pm 0 . 0 4 3$ </td><td> $3 2 . 4 \pm 1 0 . 2$ </td></tr><tr><td>COM-Poisson  $( \nu = 2 )$ </td><td> $0 . 9 2 0 \pm 0 . 0 1 9$ </td><td> $0 . 8 7 3 \pm 0 . 0 2 6$ </td><td> $3 3 . 6 \pm 6 . 5$ </td></tr></table>

Ordering remains accurate in both non-Poisson settings, while DAG recovery varies: $\nu = 0 . 5$ has lower mean $F _ { 1 }$ than $\nu = 1$ . This supports recovery on additional fixed-carrier one-parameter families, not arbitrary count conditional distributions or parent-dependent shape parameters.

## H.2.8 CONTINUOUS-DATA ORDERING RECOVERY

Algorithm. DISCO-CONT builds on DiffAN’s score network (Sanchez et al., 2023), adds marginal projection, and uses CCS for sink selection. We train an unmasked joint score network for 1000 epochs and a width-256 mask-conditioned projection network for 400 epochs, using squared error against noise predictions from the fixed joint network on corrupted inputs. Both stages use a learning rate of $1 0 ^ { - 3 }$ and a batch size of 1000; neither network is refitted during ordering. Inputs are standardized for network training, but CCS uses curvature in the original data units at $t = 0 . 0 5$ and held-out squared residuals from five-fold cross-fitted cubic splines of $X _ { j }$ (six quantile knots; ridge penalty $1 0 ^ { - 3 } )$ .

The DiffAN comparator uses masking without residue updates. For Gaussian data, it is trained separately with a 3000-epoch cap and the implementation’s early-stopping and voting rules. For Gamma data, both ordering procedures share the same joint network, with DiffAN evaluated at $t = 0 . 0 5$ without retraining. The methods differ in both marginal-score approximation and sink selection.

The Gamma joint network omits the first hidden-layer normalization to retain amplitude information relevant to score derivatives. We use paired Gaussian perturbations ϵ and −ϵ. Half the noise levels come from $t \sim \mathrm { U n i f o r m } ( 0 , 0 . 5 )$ and half from noise standard deviations drawn log-uniformly on $[ 0 . 0 5 , 0 . 8 ]$ . This mixture broadens the range of noise scales used for joint training. Projection uses $\bar { t } \sim \mathrm { U n i f o r m } ( 0 , 0 . 5 )$

Experimental setup. We compare Gaussian and Gamma conditional families at $d \ : = \ : 3 0$ and $ { n _ { \mathrm { ~  ~ } } } = \ 5 0 0 0$ for both ER1 and ER3. A random node permutation defines the possible forward edges, of which exactly kd are sampled for ERk. Each mechanism $f _ { j }$ is drawn independently from a zero-mean Gaussian process with a multivariate RBF kernel of unit variance and unit length scale. In the Gaussian model, sources are standard normal and non-sources follow $X _ { j } = f _ { j } ( X _ { \mathrm { P a } _ { G } ( j ) } ) + \varepsilon _ { j }$ with independent $\varepsilon _ { j } \sim \mathcal { N } ( 0 , 1 )$ . Gamma conditionals have fixed shape 6 and mean $2 \exp \{ 0 . 5 f _ { j } ( X _ { \mathrm { P a } _ { G } ( j ) } ) \}$ ; sources have the same shape and mean 2. Within each family and seed, both methods use the same graph and observations.

Table 12: Continuous-data ordering recovery $( d = 3 0 , n = 5 0 0 0 )$ . Entries report $A _ { \mathrm { t o p } }$ (higher is better) as means ± sample standard deviations.
<table><tr><td>Graph Method</td><td></td><td>Gaussian</td><td>Gamma</td></tr><tr><td>ER1</td><td>DISCO-CONT DiffAN</td><td> $0 . 8 7 7 \pm 0 . 0 6 3$   $0 . 8 5 0 \pm 0 . 0 7 2$ </td><td> $0 . 6 4 0 \pm 0 . 1 2 6$   $0 . 5 4 0 \pm 0 . 1 3 7$ </td></tr><tr><td>ER3</td><td>DISCO-CONT DiffAN</td><td> $0 . 8 6 6 \pm 0 . 0 4 1$   $0 . 8 8 2 \pm 0 . 0 4 9$ </td><td> $0 . 7 1 8 \pm 0 . 0 2 1$   $0 . 6 5 7 \pm 0 . 0 5 7$ </td></tr></table>

Results. For Gaussian data, DISCO-CONT has higher mean $A _ { \mathrm { t o p } }$ on ER1, whereas DiffAN has higher mean $A _ { \mathrm { t o p } }$ on ER3 (Table 12).

For Gamma data, the two methods share the same joint network. DISCO-CONT has higher mean $A _ { \mathrm { t o p } }$ on both ER1 and ER3. Both methods have lower ordering accuracy on ER1 than on ER3.

## H.3 REAL DATA: LAHMAN BATTING COUNTS

Data and preprocessing. We use the Lahman batting table through 2025 (Lahman, n.d.). For each season from 2019 to 2025, we retain complete stints, select the 200 players with the most at-bats, aggregate retained stints by player, and pool the seven seasons, yielding $n = 1 , 4 0 0$ player– season observations from 490 players on $d = 1 7$ batting-count variables. The main analysis uses a four-level quantile representation (q4) to increase the number of observations per count level for CCS estimation. Each pooled column is cut at its empirical quartiles, with no additional supportrank recoding after binning. This support compression is a finite-sample preprocessing choice and need not preserve the conditional-independence structure of the raw counts; we therefore examine alternative support resolutions and raw counts below.

Table 13: Pairwise sensitivity of ODS to the supplied conditional family on the Lahman q4 representation. Entries are mean ± standard deviation across three paired cross-validation assignments. Pairwise SHD compares two fitted directed graphs, not an estimated graph against a causal ground truth.
<table><tr><td></td><td>ODS pair Order agreement Edge Jaccard Pairwise SHD</td><td></td><td></td></tr><tr><td>Poi-NB</td><td> $0 . 8 8 2 \pm 0 . 0 0 7$ </td><td> $0 . 7 4 7 \pm 0 . 0 2 3$ </td><td> $1 6 . 7 \pm 1 . 5$ </td></tr><tr><td>Poi-Bin</td><td> $0 . 9 8 8 \pm 0 . 0 0 8$ </td><td> $0 . 9 2 4 \pm 0 . 0 2 1$ </td><td> $4 . 3 \pm 1 . 2$ </td></tr><tr><td>NB-Bin</td><td> $0 . 8 8 0 \pm 0 . 0 0 4$ </td><td> $0 . 7 1 3 \pm 0 . 0 1 9$ </td><td> $1 9 . 0 \pm 1 . 0$ </td></tr></table>

Order evaluation. Standard batting definitions (Major League Baseball, n.d.) motivate the seven domain-informed directions

$$
{ \mathcal { R } } _ { 7 } = \{ { \mathrm { A B } } \to { \mathrm { H } } , { \mathrm { A B } } \to { \mathrm { S O } } , { \mathrm { A B } } \to { \mathrm { G I D P } } , { \mathrm { H } } \to 2 { \mathrm { B } } , { \mathrm { H } } \to 3 { \mathrm { B } } , { \mathrm { H } } \to { \mathrm { H R } } , { \mathrm { B B } } \to { \mathrm { I B B } } \} .
$$

These accounting or containment relations are used only for post-fit diagnostics and are not supplied during fitting. Because several imply parent-dependent feasible ranges, they need not satisfy the parent-independent-support assumptions of Definition 3. We therefore interpret $A _ { \mathrm { t o p } }$ as agreement with a partial directional reference, not as DAG causal accuracy or as evidence that the Lahman distribution belongs exactly to the proposed model class.

For reference-set sensitivity, we additionally use $\mathcal { R } _ { 1 7 } = \mathcal { R } _ { 7 } \cup \mathcal { R } _ { \mathrm { P P } }$ , where

$$
\begin{array} { r l } & { \mathcal { R } _ { \mathrm { P P } } = \{ \mathrm { G }  \mathrm { H } , \mathrm { G }  \mathrm { B B } , \mathrm { G }  \mathrm { S O } , \mathrm { G }  \mathrm { R B I } , \mathrm { A B }  \mathrm { B B } , \mathrm { A B }  \mathrm { R B I } , } \\ & { \qquad \mathrm { H }  \mathrm { R } , \mathrm { H }  \mathrm { S O } , \mathrm { H R }  \mathrm { I B B } , \mathrm { S B }  \mathrm { C S } \} . } \end{array}
$$

The ten additional directions are extracted from the qualitative relationships discussed in the MLB analysis of Park and Park (2019b); neither $\mathcal { R } _ { 7 }$ nor $\mathcal { R } _ { 1 7 }$ is treated as a known causal DAG.

Method settings. General replication, evaluation, and estimator settings follow Appendix H.1, and baseline implementations follow Appendix H. DISCO uses seeds 0–2, while the ODS familyspecification analysis uses three paired cross-validation assignments shared across the Poisson, negative-binomial, and binomial fits so that only the supplied family specification changes. All fits use the same fixed cohort, so the reported variation reflects fitting or cross-validation variability rather than sampling uncertainty.

Family-specification sensitivity. Under the main q4 setting, the three DISCO orders respect $6 / 7 .$ $7 / 7 ,$ and $\mathbf { \bar { 6 } } / 7$ relations in $\mathcal { R } _ { 7 }$ , yielding mean $A _ { \mathrm { t o p } } \stackrel { - } { = } 0 . 9 0 \bar { 5 }$ . The corresponding mean values for ODS under the Poisson, negative-binomial, and binomial specifications are 0.429, 0.333, and 0.429, respectively. Figure 4 summarizes ordering support and majority parent-edge recovery on these seven reference relations. Because the reference set is incomplete, non-reference edges are not interpreted as false causal edges.

The three ODS fits are compared pairwise in Table 13. Order agreement is the fraction of unordered node pairs ranked in the same relative order by two fitted orders. Edge Jaccard is $| \widehat { E } _ { 1 } \cap \widehat { E } _ { 2 } | / | \widehat { E } _ { 1 } \cup \widehat { E } _ { 2 } |$ for their directed edge sets, treating opposite directions as distinct edges.

Poisson and binomial yield similar estimated orders, whereas comparisons involving the negativebinomial specification show larger changes in both ordering and edge set. Because the q4 codes are compressed counts rather than the original count scale, these results are interpreted as sensitivity to the supplied ODS specification rather than as correctly specified family comparisons.

Boundary and support-resolution sensitivity. The main q4 implementation evaluates upper-grid shifts by clipping, whereas the population CCS is defined on the admissible finite-difference domain. As a reproduction check, the original q4 configuration again yields $6 / 7 , 7 / 7$ , and $6 / 7$ agreement across the three runs, giving mean $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 7 } } = 0 . \dot { 9 } 0 5$ We then hold the DISCO training configuration fixed and replace only curvature evaluation by the admissible-domain rule. Under this rule, $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 7 } } = 0 . 7 6 2$ for each of q4, q6, and q8, while the corresponding $\mathcal { R } _ { 1 7 }$ values are 0.804, 0.765, and

![](images/bc54bf6b3b2102536ea28af019ea2bfeeee92acda23bb5040393f80faef29d58.jpg)  
Figure 4: Ordering and parent-edge diagnostics on the Lahman 2019–2025 q4 representation. The top row shows ODS with Poisson and negative-binomial specifications; the bottom row shows ODS with a binomial specification and DISCO. Blue arrows agree with the reference direction and orange arrows reverse it. Arrow labels report ordering support across the three runs. Solid arrows denote parent edges selected in at least two runs, whereas dashed arrows denote majority ordering without majority parent-edge selection. The reference pairs are used only for post-fit evaluation.

Table 14: Boundary and support-resolution sensitivity of DISCO on Lahman. The first row reproduces the main q4 analysis; the remaining rows use admissible finite differences with the same DISCO training configuration.
<table><tr><td>Representation</td><td>Boundary</td><td> $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 7 } }$ </td><td> $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 1 7 } }$ </td></tr><tr><td>q4</td><td>clipped</td><td>0.905</td><td>0.843</td></tr><tr><td>q4</td><td>admissible</td><td>0.762</td><td>0.804</td></tr><tr><td>q6</td><td>admissible</td><td>0.762</td><td>0.765</td></tr><tr><td>q8</td><td>admissible</td><td>0.762</td><td>0.784</td></tr><tr><td>raw</td><td>admissible</td><td>0.000</td><td>0.137</td></tr></table>

0.784. Thus the absolute q4 result is sensitive to boundary handling, whereas the admissible-domain ordering agreement is stable across the examined compressed support resolutions.

The substantially lower raw-count agreement indicates that support compression is important in this finite-sample Lahman analysis.

Preprocessing sensitivity. Because quantile compression changes the raw count mean–variance relationship used by QVF-based ODS, we additionally evaluate ODS on the uncompressed Lahman counts. For this sensitivity analysis, the Poisson and negative-binomial variants use log-link regressions, while the binomial variant uses a logit link with empirical nodewise support maxima. On the raw counts, $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 7 } }$ is 0.333, 0.857, and 0.286 for the Poisson, negative-binomial, and binomial specifications, respectively, with corresponding $\mathcal { R } _ { 1 7 }$ values 0.412, 0.843, and 0.392. Thus ODS can attain high directional agreement under one supplied family, but the result varies substantially with that specification; these results are therefore treated as preprocessing and family-specification sensitivity rather than as an oracle comparison with DISCO.

Table 15: ODS sensitivity on the raw Lahman counts. The binomial row uses empirical nodewise support maxima and is not an oracle binomial specification.
<table><tr><td>Supplied family</td><td> $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 7 } }$ </td><td> $A _ { \mathrm { t o p } } ^ { \mathcal { R } _ { 1 7 } }$  Mean  $| \widehat { E } |$ </td></tr><tr><td>Poi</td><td>0.333 0.412</td><td>73.7</td></tr><tr><td>NB</td><td>0.857 0.843</td><td>69.3</td></tr><tr><td>Bin</td><td>0.286 0.392</td><td>79.7</td></tr></table>

Additional robustness checks. For the 2019–2025 q4 analysis, the three DISCO orders agree with $\mathcal { R } _ { 1 7 }$ on 15/17, 15/17, and 13/17 relations, giving mean $A _ { \mathrm { t o p } } = 0 . 8 4 3$ . Repeating the same cohort construction and q4 preprocessing for 2012–2018 gives mean agreement 0.952 on $\mathcal { R } _ { 7 }$ and 0.922 on $\mathcal { R } _ { 1 7 }$ . These checks show that the reported directional pattern is not confined to one reference set or one analysis period, while remaining descriptive partial-order diagnostics rather than causal validation.

## I ALGORITHM DIAGNOSTICS

## I.1 MARGINAL PROJECTION METHODS

We compare three projection procedures on DAGs with d = 50 and $n = 5 0 0 0$ , using the same mechanisms for the three conditional families as in Appendix I.2. They share the data, three-fold splits, fixed joint networks, the first sink selected from all variables, and initial projection networks.

Additional training settings. We use the joint-network validation protocol, initial projection learning rate, and batch size specified in Appendix H.1.

Amortized projection uses this projection network throughout ordering without further training. After each removal, Stagewise direct projection initializes the projection network with its previous parameters and trains it for 100 additional epochs, using predictions from the fixed joint network as regression targets. Stagewise recursive projection uses the same training budget but obtains targets from the preceding projection network, whose parameters are held fixed. Both procedures use the joint network for the first reduced-set fit. Thus each stagewise procedure adds 48 fits per fold, for remaining-set sizes 49 through 2. Training targets and projection predictions use the same fully corrupted sample and noise level. No joint concrete-score network is refitted on a reduced set.

Parent selection uses the recovered orders with the same fixed amortized projection networks and default OCS rule, isolating the ordering procedure. Total computation budgets are not matched. Projection time includes fitting the shared initial projection network, the additional training at each removal, and saving network parameters, but excludes joint training, CCS evaluation, and parent selection.

Amortized projection achieves comparable or higher mean $A _ { \mathrm { t o p } } ,$ , higher mean $F _ { 1 }$ , lower mean SHD, and shorter projection time in each conditional family. These differences are not uniform across seeds.

## I.2 PARENT SELECTION

We use the first Poi and NB mechanisms and the second Bin mechanism in Table 5. The held-out folds collectively cover all 5000 observations, and no network is retrained for this comparison.

For the threshold-sensitivity analysis, the median/IQR standardization is unchanged. We vary τ<sub>Z</sub> ∈ {1, 1.5, 2, 2.5, 3} and $\pi _ { \operatorname* { m i n } } \in \{ 2 / 3 , 1 \}$ : a candidate must satisfy the strict upper-tail test $Z _ { j \to i } ^ { ( b ) } > \tau _ { Z }$ in at least two or all three folds, respectively.

Table 16: Comparison of marginal concrete-score projection methods $( d = 5 0 , n = 5 0 0 0 )$ . Recovery metrics are means ± sample standard deviations, with seeds paired across methods within each conditional family. Projection time is the mean wall-clock time for the projection stage, in seconds.
<table><tr><td rowspan="2">Conditional family Projection</td><td rowspan="2"></td><td colspan="4"></td></tr><tr><td> $A _ { \mathrm { t o p } }$ </td><td> $F _ { 1 }$ </td><td>SHD</td><td>Projection time (s)</td></tr><tr><td rowspan="3">Poi</td><td>Amortized</td><td> $0 . 9 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 8 9 2 \pm 0 . 0 3 4$ </td><td> $2 8 . 9 \pm 8 . 7$ </td><td>92.6</td></tr><tr><td>Stagewise direct</td><td> $0 . 9 1 7 \pm 0 . 0 1 6$ </td><td> $0 . 8 7 4 \pm 0 . 0 3 4$ </td><td> $3 4 . 0 \pm 9 . 2$ </td><td>511.9</td></tr><tr><td>Stagewise recursive</td><td> $0 . 9 1 6 \pm 0 . 0 1 3$ </td><td> $0 . 8 6 9 \pm 0 . 0 3 2$ </td><td> $3 5 . 1 \pm 8 . 2$ </td><td>544.5</td></tr><tr><td rowspan="3">NB</td><td>Amortized</td><td> $0 . 9 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 7 9 2 \pm 0 . 0 6 6$ </td><td> $5 1 . 2 \pm 1 5 . 4$ </td><td>94.0</td></tr><tr><td>Stagewise direct</td><td> $0 . 9 2 3 \pm 0 . 0 1 9$ </td><td> $0 . 7 8 5 \pm 0 . 0 5 5$ </td><td> $5 3 . 3 \pm 1 3 . 1$ </td><td>515.0</td></tr><tr><td>Stagewise recursive</td><td> $0 . 9 2 4 \pm 0 . 0 2 1$ </td><td> $0 . 7 8 8 \pm 0 . 0 5 3$ </td><td> $5 2 . 6 \pm 1 2 . 9$ </td><td>545.2</td></tr><tr><td rowspan="3">Bin</td><td>Amortized</td><td> $0 . 9 0 1 \pm 0 . 0 2 7$ </td><td> $0 . 8 7 7 \pm 0 . 0 3 1$ </td><td> $3 2 . 3 \pm 8 . 4$ </td><td>94.2</td></tr><tr><td>Stagewise direct</td><td> $0 . 9 0 1 \pm 0 . 0 3 2$ </td><td> $0 . 8 7 3 \pm 0 . 0 3 6$ </td><td> $3 3 . 4 \pm 9 . 7$ </td><td>510.3</td></tr><tr><td>Stagewise recursive</td><td> $0 . 9 0 0 \pm 0 . 0 2 5$ </td><td> $0 . 8 7 4 \pm 0 . 0 3 5$ </td><td> $3 3 . 1 \pm 9 . 2$ </td><td>542.0</td></tr></table>

Table 17: Parent-selection sensitivity for the threshold settings shown, with fixed learned orders and OCS estimates $( d = 5 0 , n = 5 0 0 0 )$ . Entries are arithmetic means over the same seeds. Prec. and Rec. denote precision and recall against the full true DAG, including true edges excluded by the estimated order.
<table><tr><td rowspan="2" colspan="2">Rule</td><td rowspan="2"></td><td colspan="3">Poi</td><td colspan="3">NB</td><td colspan="3">Bin</td></tr><tr><td> $F _ { 1 }$ </td><td>Prec.</td><td>Rec.</td><td> $F _ { 1 }$ </td><td>Prec.</td><td>Rec.</td><td> $F _ { 1 }$ </td><td>Prec.</td><td>Rec.</td></tr><tr><td></td><td> $\tau _ { Z } = 2 , \pi _ { \operatorname* { m i n } } = 1$ </td><td></td><td>0.877</td><td>0.934</td><td>0.827</td><td>0.797</td><td>0.917</td><td>0.708</td><td>0.878</td><td>0.943</td><td>0.821</td></tr><tr><td></td><td> $\tau _ { Z } = 2 , \pi _ { \operatorname* { m i n } } = 2 / 3$ </td><td></td><td>0.891</td><td>0.898</td><td>0.884</td><td>0.870</td><td>0.898</td><td>0.845</td><td>0.890</td><td>0.895</td><td>0.885</td></tr><tr><td></td><td> $\tau _ { Z } = 1 , \pi _ { \operatorname* { m i n } } = 1$ </td><td></td><td>0.900</td><td>0.898</td><td>0.901</td><td>0.866</td><td>0.892</td><td>0.843</td><td>0.896</td><td>0.906</td><td>0.885</td></tr><tr><td></td><td> $\tau _ { Z } = 1 , \tau _ { \mathrm { m i n } } = 2 / 3$ </td><td></td><td>0.800</td><td>0.709</td><td>0.917</td><td>0.813</td><td>0.735</td><td>0.910</td><td>0.809</td><td>0.732</td><td>0.907</td></tr></table>

Table 17 shows that lowering τ<sub>Z</sub> from 2 to 1 while retaining all-fold agreement, or requiring two rather than three folds at $\tau _ { Z } = 2$ , increases mean $F _ { 1 }$ in all three conditional families. For negative binomial, relaxing fold agreement alone raises $F _ { 1 }$ from 0.797 to 0.870. Relaxing both requirements increases false positives and lowers $F _ { 1 }$ relative to the one-at-a-time changes. Thus the same OCS estimates can yield different recovery under different selection rules. No threshold is selected or tuned against the true graph; this is not an error-control procedure and does not replace the default $\tau _ { Z } = 2 , \pi _ { \operatorname* { m i n } } = 1$

Comparison of parent recovery methods. To separate parent selection from ordering, we compare the default DISCO OCS selector with CAM pruning, an additive-regression variable-selection procedure (Buhlmann et al., 2014), and PCM-GAM, a conditional-mean independence test (Lund-¨ borg et al., 2024). Under a valid causal order in our semiparametric GLM model, $\mathbb { E } [ X _ { i } \mid X _ { \mathrm { P r e d } ( i ) } ] =$ $A _ { i } ^ { \prime } ( \eta _ { i } )$ . Since $A _ { i } ^ { \prime \prime } > 0$ , the nonconstant parent effects in Definition 3, together with the support regularity conditions, make conditional-mean dependence on a predecessor equivalent to that predecessor being a parent of i. Thus the population criterion targeted by PCM identifies true parents; this does not guarantee exact recovery by the finite-sample PCM-GAM test.

We use a separate study with $d = 5 0$ and $n = 5 0 0 0$ , under both the true order and the same estimated DISCO order. Family-specific generators are unchanged from the study above. Each dataset is split into 2500 ordering observations and 2500 parent-recovery observations; PCM further splits the latter into 1250 nuisance-training and 1250 testing observations. This disjoint-sample protocol is distinct from the cross-fitted threshold study above.

We reuse the CAM implementation described in Appendix H.1.1, with cutoff 0.001. PCM-GAM uses the public comets::pcm implementation of the projected covariance measure (Lundborg et al., 2024), with GAM mean/projection regressions (basis parameter $k = 4 )$ and a 500-tree random forest for the variance nuisance. Each candidate is tested at $\alpha = 0 . 0 5$ without multiplicity correction. CAM and PCM-GAM receive raw counts without the true conditional family or link. These are method-specific settings, not tests at a common nominal level; using GAM nuisance estimates does not itself establish count-specific test calibration.

Table 18: Parent recovery under fixed orders $( d = 5 0 , n = 5 0 0 0 )$ . Entries are arithmetic means over seeds shared by all methods and both order conditions. Metrics use the full true DAG, including edges excluded by an estimated order. Estimated denotes the shared DISCO order; OCS denotes the default DISCO parent selector.
<table><tr><td colspan="2">Conditional family Order</td><td>Parent selector</td><td> $F _ { 1 }$ </td><td>SHD</td></tr><tr><td rowspan="6">Poi</td><td></td><td>OCS</td><td>0.878</td><td>30.8</td></tr><tr><td>True</td><td>CAM</td><td>0.989</td><td>3.2</td></tr><tr><td></td><td>PCM-GAM</td><td>0.839</td><td>54.1</td></tr><tr><td rowspan="3">Estimated</td><td>OCS</td><td>0.858</td><td>37.2</td></tr><tr><td>CAM</td><td>0.850</td><td>46.3</td></tr><tr><td>PCM-GAM</td><td>0.712</td><td>105.6</td></tr><tr><td rowspan="6">NB</td><td rowspan="3">True</td><td>OCS</td><td>0.780</td><td>50.5</td></tr><tr><td>CAM</td><td>0.959</td><td>12.1</td></tr><tr><td>PCM-GAM</td><td>0.832</td><td>56.3</td></tr><tr><td rowspan="3">Estimated</td><td>OCS</td><td>0.751</td><td>59.2</td></tr><tr><td>CAM</td><td>0.823</td><td>54.9</td></tr><tr><td>PCM-GAM</td><td>0.703</td><td>108.3</td></tr><tr><td rowspan="5">Bin</td><td rowspan="3">True</td><td>OCS</td><td>0.820</td><td>43.1</td></tr><tr><td>CAM</td><td>0.993</td><td>1.9</td></tr><tr><td>PCM-GAM</td><td>0.827</td><td>58.3</td></tr><tr><td>OCS</td><td>0.828</td><td>42.6</td></tr><tr><td rowspan="3">Estimated</td><td>CAM</td><td>0.898</td><td>28.7</td></tr><tr><td>PCM-GAM</td><td></td><td></td></tr><tr><td></td><td>0.738</td><td>90.3</td></tr></table>

With the true order, CAM has the highest mean $F _ { 1 }$ and lowest SHD in every conditional family (Table 18). With the shared estimated order, OCS outperforms PCM-GAM in all three families and has slightly higher $F _ { 1 }$ and lower SHD than CAM for Poi; CAM remains strongest for NB and Bin. The OCS gap under the true order reflects limitations of learned scores and the practical selector, rather than the population OCS characterization; this comparison does not separate score-estimation error from thresholding error.