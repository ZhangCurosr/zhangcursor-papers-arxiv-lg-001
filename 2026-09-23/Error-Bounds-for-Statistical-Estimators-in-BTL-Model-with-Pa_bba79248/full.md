# Error Bounds for Statistical Estimators in BTL Model with Parametric Multivariate Utility Functions

Yicheng Li<sup>1</sup> and Huifu Xu<sup>2</sup>

September 23, 2026

## Abstract

We study preference elicitation under the Bradley-Terry-Luce (BTL) model where the true partworth vector is unknown and has to be estimated as a parameter with elicited preference information. The set of selected pairwise queries is non-uniform, deterministic, and arbitrary over a collection of alternatives, provided that it satisfies a joint identifiability condition. We focus on understanding when the canonical maximum likelihood estimator (MLE) is finite and admits sharp error bounds without explicit compactness constraints on the feasible set or external regularizers. To this end, we derive minimax lower bounds under the standard bounded dynamic range condition, and find that the same Fisher-information geometry in the classic Cram´er-Rao lower bounds underpins the finite-sample dificulty of the estimation problem. By combining a non-asymptotic expansion of the likelihood score equation with a fixed-point localization argument, we identify a design-dependent sample size threshold above which the unconstrained canonical MLE exists and is unique with high probability. The same expansion yields a decomposition of the estimation error into a linear stochastic term, an explicit second-order bias, and a higher-order remainder. A refined analysis gives suficient sample size conditions under which the canonical MLE attains the minimax rates up to logarithmic and constant factors. These results provide a unified non-asymptotic theory for parametric utility elicitation and reveal when the inference is determined by response data alone rather than by external regularization. Preliminary numerical results are consistent with the theoretical findings.

Keywords. Parametric multivariate utility function, BTL model, preference elicitation, pairwise comparison, MLE, minimax lower bounds

## 1 Introduction

Preference elicitation seeks to infer a decision maker’s (DM’s) latent utility function from a limited collection of observed preference statements. In many applications, including but not limited to marketing and transportation [65; 78], interactive recommendation systems [30; 5], and reinforcement learning from human feedback [56; 80], such statements are collected through survey-based statistical procedures. In particular, given d alternatives (e.g., products or services), the experimenter may present selected pairs $( i , j ) \in \mathbb { I } : = \{ ( k , l ) \mid 1 \leq k < l \leq d \}$ to the DM and record which alternative is preferred. Let $\mathcal { X } : = \{ \mathbf { x } _ { 1 } , \dots , \mathbf { x } _ { d } \} \subseteq \mathbb { R } ^ { n }$ , where $\mathbf { x } _ { i }$ denotes the characteristic vector of alternative i, and let $U : \mathcal { X }  \mathbb { R }$ denote the DM’s latent utility function. We are then interested in quantifying the statistical efectiveness of estimating $U ( \cdot )$ from a prescribed questionnaire design and the resulting ordinal responses.

In some practical applications, the preference statements are not necessarily consistent, which means the observed pairwise choices are not transitive [71] and the DM’s responses to the same pair of alternatives may be diferent [32], either because the DM’s preferences are truly random or there are random errors in the elicitation process [60; 66; 6; 14]. The phenomena cannot be described by deterministic utility models (including von Neumann-Morgenstern’s expected utility theory) such as [65; 79; 75], because no single admissible utility function is consistent with all observed responses. Likewise, set-based approaches remain non-probabilistic, but represent incomplete preference information through families of deterministic utility functions [29; 26]. On the other hand, random utility theory models such response variability by treating utilities as random variables [12; 33]. The most well-known one is the Bradley-Terry-Luce (BTL) model [8; 46] in the discrete-choice literature. Bradley and Terry [8] first use the formula

$$
\mathbb { P } ( i \mid \{ i , j \} ) = \frac { U ( \mathbf { x } _ { i } ) } { U ( \mathbf { x } _ { i } ) + U ( \mathbf { x } _ { j } ) }\tag{1.1}
$$

to calculate the probability of alternative i winning a comparison $( i , j )$ . Luce develops an axiomatic description of DM’s preference to justify (1.1) when $U ( \cdot ) : \mathcal { X } \to \mathbb { R } _ { + + }$ , see [46, Theorem 3]. By setting $U ( \cdot ) : = \sigma \log U ( \cdot )$ , where $\sigma \in ( 0 , \infty )$ is a positive constant, McFadden [49] [67, Chapter 2] shows that the DM’s preference can be described by the random utility function

$$
V ( \mathbf { x } _ { i } ) = U ( \mathbf { x } _ { i } ) + \epsilon _ { i } ,\tag{1.2}
$$

where $\epsilon _ { i } : ( \Omega , \mathcal { F } , \mathbb { P } ) \to \mathbb { R } , \ i \in [ d ]$ , are identically and independently distributed (i.i.d.) with $\epsilon _ { i } \sim \mathrm { G u m b e l } ( 0 , \sigma )$ , and

$$
\mathbb { P } ( i \mid \{ i , j \} ) = \frac { e ^ { U ( \mathbf { x } _ { i } ) / \sigma } } { e ^ { U ( \mathbf { x } _ { i } ) / \sigma } + e ^ { U ( \mathbf { x } _ { j } ) / \sigma } } .\tag{1.3}
$$

For repeated presentations of a pair $( i , j )$ , we assume that the draws $( \epsilon _ { i } , \epsilon _ { j } )$ are independent.   
The logit form in (1.3) is widely known as the BTL model.

When the utility function $U ( \cdot )$ is linear in its parameters, maximum likelihood estimation under the BTL model is down to solving a logistic regression problem, see [48, Chapter 4] and [21, Chapter 7]. Nevertheless, the statistical question depends on how the questionnaire is generated. In preference elicitation, the experimenter controls the comparisons manually [27; 80; 50], or sometimes through an optimal design criterion [66; 78; 28]. Thus, covariates here need not satisfy specific distributional properties used in, e.g., [52; 13]. In addition, many survey-based data collection procedures involve a limited number of responses because of cost, logistics, or respondent fatigue. Consequently, the reliability of asymptotic maximum likelihood theory and experimental design based on the Fisher information matrix (FIM) is questionable [78, Section 2]. These two concerns raise the basic well-posedness issue of the maximum likelihood estimator (MLE). Classical results [36; 3; 42] show that the positive definiteness of the FIM does not exclude separation in the realized responses, in which case a finite MLE fails to exist. A common remedy to this challenge is to stabilize the estimator externally. For example, one may restrict the parameter to a prescribed compact set [61; 80; 63; 14], regularize the likelihood via ridge [4; 18] or Firth correction [23; 37], and introduce a Bayesian prior [60; 45]. The constrained MLE guarantees the existence of an empirical minimizer and a uniform lower bound on the likelihood curvature over the feasible set, but it requires the diameter of that set to be specified a priori. Firth correction remains finite under separation and has a shrinkage property when applied to logistic regression [37, Theorem 2], yet the resulting loss function is generally non-convex. These strategies produce finite estimates even under separation, but finiteness alone does not establish that the likelihood is informative in every identifiable direction. When the response data remain separated, the magnitude of the estimate along an escaping direction is determined primarily by the intensity of the external regularizer. This leads to our central question:

For a fixed and possibly nonuniform questionnaire design, how many responses sufice for the unconstrained canonical MLE to exist with high probability and exhibit sharp non-asymptotic error bounds?

This issue has been well addressed in pairwise ranking models, where every query vector is a canonical graph edge. In this setting, the DM assigns a structureless latent preference score to each alternative, and the FIM reduces to a weighted graph Laplacian [61] (see also Definition 2.2 and Section 2.3.2). The special geometry leads to graph-topology treatment of the estimation problem, e.g., information-theoretic lower bounds established through Laplacian spectral quantities [31; 61; 39; 40], finite-sample MLE existence criteria derived via Ford’s condition on the directed win-loss graph [24; 9; 7], and upper bounds obtained from graph-based localization [61; 51; 16; 76; 77] and leave-one-out (LOO) arguments [19; 18; 25]. Even though related techniques also appear in, e.g., structured von Neumann–Morgenstern utility elicitation [14], downstream policy optimality quantification [80], K-wise questionnaire design under the Plackett–Luce model [50], covariate-assisted ranking under Erd¨os-R´enyi graph design [22], and $\ell _ { \infty }$ error analysis over general comparison graphs [43], the unified graph-theoretic framework does not transfer mechanically once the canonical edge vectors are replaced by general covariate directions. Some of these error bounds remain conservative when the heterogeneity of the query directions is taken into account. For example, they may depend inversely on the smallest eigenvalue of the FIM, leading to deterioration in Euclidean error and sample complexity, see e.g., [61, Theorem 2] and [63, Lemma 6]; or be inherently suboptimal when the comparison graph is irregular, see discussion in [16; 76].

For general covariates, the overlapping condition [3] conveys not only the graph topology but also the conic geometry of the covariates in the parameter space [62]. A sharp and explicit sample threshold for MLE existence requires additional assumptions on how the covariates are generated. For example, Cand\`es and Sur [11] derive an asymptotic phase transition for MLE existence under Gaussian covariates, while Chardon et al. [13] obtain non-asymptotic guarantees for MLE existence and excess risk under Gaussian and other regular random designs. They also show that for discrete covariates, existence may depend strongly on the orientation of the ground truth [13, Section 3.3]. Our aim is instead to derive a design-dependent suficient condition that applies to an arbitrary but fixed collection of pairwise queries.

![](images/9f319b4f48bb30f11d599e7d1882c8111687c68619e63651b214e28aa0357644.jpg)  
Figure 1: Flowchart of theoretical results.

We use a two-step analysis based on the pseudo self-concordance property of the logistic likelihood function [4; 52]. First, we localize the estimator uniformly along each query direction, which keeps the empirical Hessian comparable to its value at the ground truth. Second, we use a likelihood score expansion to connect this local behavior with global excess risk and Euclidean error. Due to the non-asymptotic nature of the results to be presented, the expansion retains the second-order bias of the canonical MLE. Classical works show explicit bias formulas for generalized linear models [20], and show how Firth’s correction removes the leading bias asymptotically [23]. More recent finite-sample analysis for smooth functionals of Z-estimators by Lin et al. [44] identifies how the second-order bias produces a quadratic sample size barrier, complementing the growing-dimensional asymptotic literature by Portnoy [53; 54].

## 1.1 Main contributions

In this paper, we study the parametric utility elicitation under the BTL model with a fixed and possibly nonuniform collection of pairwise queries. To facilitate reading, we plot a flowchart (see Figure 1) outlining key theoretical developments. The main contributions are three-fold.

• We consider a linear-in-parameter utility function subject to linear equality constraints (Definition 2.1). This covers several classical specifications, e.g., pairwise ranking model [18; 61; 22], multivariate linear utility function [67; 60; 75], piecewise linear approximation of nonlinear univariate utilities [14; 79]. We derive a necessary and suficient identifiability condition for the population-level MLE to be well-posed (Proposition 2.3), equivalent to the positive definiteness of the design matrix (Corollary 2.1). Under the identifiability condition, we establish both Cram´er–Rao Lower Bounds (CRLB) for locally unbiased estimators (Corollary 3.1), and minimax lower bounds over bounded dynamic range parameter class (Theorems 3.1, 3.2 and Corollary 3.2). These results demonstrate that trace and coherence-based Fisher-information criteria govern both classical local eficiency and finite-sample minimax lower bounds, when measured by the Euclidean norm and along individual query directions, respectively.

• We develop a two-step convex localization strategy for the finite-sample analysis of the unconstrained MLE. Motivated by the induction-type localization arguments in ranking literature [18; 16], we express the likelihood score equation as a fixed-point system. By combining this construction with the expansion strategy of score equation in [44; 63], and the pseudo self-concordance trick in [4; 52], we derive a design-dependent suficient sample size above which the unconstrained MLE exists and is unique with high probability (Theorem 4.1). Instead of imposing boundedness constraints or introducing explicit regularization for the MLE, we localize the likelihood minimizer in a convex and compact set around the ground truth in the proof.

• We distinguish the sample sizes suficient for existence of the MLE from those suficient for its minimax optimality. A refined decomposition of the estimation error isolates the contribution of the linear stochastic term, deterministic second-order bias, and higher-order remainder (Theorem 4.2). When measured by the Euclidean norm, beyond the baseline localization requirements, the suficient sample size for the MLE to attain the minimax rate depends on the inverse of the FIM via its efective rank but not its extreme eigenvalue (Corollary 4.2), thereby improving the previous conclusions in [61; 63]. Numerical results in Section 5 illustrate how the bias and nonlinear residuals deteriorate the statistical eficiency of the MLE.

The rest of the paper is organized as follows. In Section 2, we introduce the parametric utility model and its general identifiability condition. Several classical problems that this model subsumes are sequentially presented as examples. In Sections 3 and 4, we discuss lower error bounds and finite-sample guarantees for the unconstrained MLE, respectively. In Section 5, we provide numerical evidence that verifies the established theory. In Section 6, we conclude with some remarks. Proofs of the main results are put in Section 7, with auxiliary arguments deferred to Appendix A.

Throughout the paper, bold lowercase and uppercase letters denote column vectors and matrices, respectively. 0, 1, and ${ \mathbf I } _ { n }$ denote the all-zero vector, the all-one vector, and the identity matrix in $\mathbb { R } ^ { n \times n }$ , respectively. $\mathbf { e } _ { i }$ is the i-th canonical Euclidean basis with dimensions clear from context. For a matrix A, $\mathbf { A } _ { i j } = [ \mathbf { A } ] _ { i j }$ denotes its $( i , j )$ -th entry. diag(a) and diag(A) stand for a diagonal matrix formed by a or a vector formed by the diagonal of A, respectively. $( a _ { i j } ) _ { i , j \leq n }$ is a matrix whose (i, j)-th element is $a _ { i j }$ . We write $\mathbf { A } ^ { \dagger }$ for the Moore-Penrose pseudoinverse of A. For any conformable matrices A, B, $\langle \mathbf { A } , \mathbf { B } \rangle = \mathrm { t r } ( \mathbf { A } ^ { \top } \mathbf { B } )$ denotes their Frobenius inner product, and A◦B is the Hadamard product between them. $\mathbf { A } \succ \mathbf { 0 } \left( \mathbf { A } \succeq \mathbf { 0 } \right)$ means A is positive (semi)definite. We use $\| \mathbf { A } \| , \| \mathbf { A } \| _ { F }$ for the spectral and Frobenius norms. $\| \mathbf { a } \| _ { 2 } , \| \mathbf { a } \| _ { \mathbf { H } } : = { \sqrt { \mathbf { a } ^ { \top } \mathbf { H } \mathbf { a } } }$ stand for the Euclidean norm and H-reweighted seminorm of a, respectively, where $\mathbf { H } \succeq \mathbf { 0 }$ . For $\Theta \subseteq \mathbb { R } ^ { n }$ , int(Θ) denotes its interior. For a random variable Z, let $\| Z \| _ { \psi _ { 2 } }$ denote its sub-Gaussian norm. For a random vector $\pmb { \xi } \in \mathbb { R } ^ { n }$ , define $\| \pmb { \xi } \| _ { \psi _ { 2 } } : = \operatorname* { s u p } _ { \mathbf { u } \in \mathbb { S } ^ { n - 1 } } \| \mathbf { u } ^ { \top } \pmb { \xi } \| _ { \psi _ { 2 } }$ , where $\mathbb { S } ^ { n - 1 }$ is the unit sphere in $\mathbb { R } ^ { n }$ . For nonnegative quantities f and g, relations $f \lesssim g , f \gtrsim g$ and $f \asymp g$ mean, respectively, that $f \leq C g , f \geq c g , \mathrm { o r } c g \leq f \leq C g$ for some absolute constants $C \geq c > 0$

## 2 Model Setup and Identifiability

In this section, we spell out the details of the parametric BTL model, and discuss parameter identifiability issue. See Figure 2 for an overview of the statistical estimation procedure considered in this paper.

![](images/55bd0a06110352ba984606664c962e30ad7d915256b759dbef1b8840299a9fd4.jpg)  
Figure 2: Survey-based statistical procedures, from alternative features to utility estimation.

## 2.1 The Parametric Random Utility Model

Under the BTL framework (1.3), choice probabilities depend on the alternatives through their utility diferences and the scale parameter σ. We will handle them separately. We first specify $U ( \cdot )$ through an appropriate feature map, allowing nonlinear dependence on the characteristic vectors while retaining linearity in the unknown parameter.

Definition 2.1 (Linear-in-parameter Utility Representation) Let $\mathcal { X } : = \{ \mathbf { x } _ { 1 } , \ldots , \mathbf { x } _ { d } \}$ denote the characteristic vectors of the d alternatives, and let ϕ : $\mathcal { X } \to \mathbb { R } ^ { p }$ be a fixed vector-valued feature map. The utility values $\{ U ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { d }$ admit the following linear form w.r.t. parameter $\mathbf { w } \in \mathbb { R } ^ { p } , \ i . e .$

$$
U ( \mathbf { x } _ { i } ) : = \mathbf { w } ^ { \top } \phi ( \mathbf { x } _ { i } ) = \sum _ { k = 1 } ^ { p } w _ { k } [ \phi ( \mathbf { x } _ { i } ) ] _ { k } , ~ f o r ~ i \in [ d ] ,\tag{2.4}
$$

where $\mathbf { w } = [ w _ { 1 } , \ldots , w _ { p } ] ^ { \top }$ satisfies Aw = b for some $\mathbf { A } \in \mathbb { R } ^ { l \times p } , \ \mathbf { b } \in \mathbb { R } ^ { l }$

We mainly focus on the case where $\{ { \bf x } _ { i } \} _ { i = 1 } ^ { d }$ are deterministic attribute vectors of the alternatives, though our results can be modified to cover random lotteries with finite outcomes [14], and highly nonlinear specifications [4, Section 4.1]. The equality constraints Aw = b are adopted to encode the known structural restrictions of the utility function or to eliminate non-identifiable directions, as discussed in Proposition 2.3. They should not be confused with polyhedral cuts generated from observed preferences in deterministic elicitation methods (see e.g., [65]). Definition 2.1 places several commonly used utility specifications within a common parametric framework, as illustrated next.

Example 2.1 (Discrete Choice Model) For every $i \in [ d ] .$ , set $\mathbf { x } _ { i } \in \{ 0 , 1 \} ^ { p } , \ \phi ( \mathbf { x } _ { i } ) \ = \ \mathbf { x } _ { i }$ in (2.4), and remove the linear equality constraint Aw = b. Then $U ( \mathbf { x } _ { i } ) = \mathbf { w } ^ { \top } \mathbf { x } _ { i }$ , and it can be viewed as a multivariate linear utility function, where $\mathbf { x } _ { i }$ is the vector of attributes and w represents the partworth vector. This recovers the discrete choice model used in classical conjoint analysis [67, Chapter 2], [60, Section 3], and [75].

Example 2.2 (Piecewise-linear Approximation) For each $i \in [ d ]$ , let $x _ { i } \in [ a , b ] \subset \mathbb { R }$ denote the scalar attribute value ofalternative i. Fix a set ofbreakpoints $a = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { J } = b ,$ and let $u : [ a , b ]  [ 0 , 1 ]$ be a normalized univariate, nonlinear utility function satisfying $u ( a ) = 0$ and u(b) = 1. Define $\mathbf { w } _ { j } : = u ( t _ { j } ) - u ( t _ { j - 1 } ) , j \in [ J ]$ , and let $\mathbf { w } : = [ \mathbf { w } _ { 1 } , \hdots , \mathbf { w } _ { J } ] ^ { \top } \in \mathbb { R } ^ { J }$ . For every $\zeta \in [ a , b ]$ , define $\phi ( \zeta ) \in \mathbb { R } ^ { J }$ coordinatewise, $i . e .$

$$
[ \phi ( \zeta ) ] _ { j } : = \left\{ \begin{array} { l l } { 0 , } & { f o r \zeta \le t _ { j - 1 } , } \\ { \zeta - t _ { j - 1 } } & { f o r t _ { j - 1 } < \zeta \le t _ { j } , } \\ { t _ { j } - t _ { j - 1 } } & { } \\ { 1 , } & { f o r \zeta > t _ { j } . } \end{array} \right.
$$

Then $U ( \zeta ) : = \mathbf { w } ^ { \top } \phi ( \zeta )$ is a piecewise-linear approximation of u, and hence $U ( x _ { i } ) = \mathbf { w } ^ { \top } \phi ( x _ { i } )$ gives the corresponding utility scores $\left[ 1 4 ; \ 7 9 \right]$ . The normalization is encoded by $\mathbf { 1 } ^ { \top } \mathbf { w } = 1$ , while the monotonicity ofu implies $\mathbf { w } \geq \mathbf { 0 }$ . The same representation extends to additive multi-attribute utility functions following [33, Proposition 2].

Example 2.3 (Pairwise Ranking Model) By setting $\phi ( \mathbf { x } _ { i } ) = \mathbf { e } _ { i } \in \mathbb { R } ^ { d } , \forall i \in [ d ]$ , and imposing $\mathbf { 1 } ^ { \top } \mathbf { w } = 0$ , we can interpret the $\{ U ( \mathbf { x } _ { i } ) \} _ { i = 1 } ^ { d }$ in (2.4) as structure-less latent preference scores that the DM assigns to the items. This recovers the ranking model used in [31; 61; 18; 76].

Example 2.4 (Covariate Assisted Ranking Model) Let $\phi ( \mathbf { x } _ { i } ) = [ \mathbf { e } _ { i } ^ { \top } , \mathbf { x } _ { i } ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { d + n }$ $\mathbf { w } =$ $[ \mathbf { w } _ { 1 } ^ { \top } , \mathbf { w } _ { 2 } ^ { \top } ] ^ { \top }$ , where $\mathbf { w } _ { 1 } \in \mathbb { R } ^ { d } , \ \mathbf { w } _ { 2 } \in \mathbb { R } ^ { n }$ . Then $U ( \mathbf { x } _ { i } ) = \mathbf { e } _ { i } ^ { \top } \mathbf { w } _ { 1 } + \mathbf { x } _ { i } ^ { \top } \mathbf { w } _ { 2 } ,$ where $\mathbf { e } _ { i } ^ { \top } \mathbf { w } _ { 1 }$ represents the residual scores that cannot be explained by the contribution from covariates $\mathbf { w } _ { 2 } ^ { \top } \mathbf { x } _ { i }$ alone [22]. Under the condition that

$$
\bar { \mathbf { X } } : = \left[ \begin{array} { l l l } { 1 } & { \cdots } & { 1 } \\ { \mathbf { x } _ { 1 } } & { \cdots } & { \mathbf { x } _ { d } } \end{array} \right] \in \mathbb { R } ^ { ( n + 1 ) \times d } , \ \mathrm { r a n k } ( \bar { \mathbf { X } } ) = n + 1 ,
$$

the linear equality constraint $[ \bar { \mathbf { X } } , \mathbf { 0 } _ { ( n + 1 ) \times n } ] \mathbf { w } = \mathbf { 0 }$ is imposed to ensure identifiability. We will come back to this in Example 2.6.

We now discuss the specification of $\sigma .$ . To preserve strictly positive choice probabilities and a nondegenerate stochastic model, we assume $0 < \sigma < \infty$ throughout. When $\sigma \downarrow 0$ the DM chooses alternative i without uncertainty whenever $U ( \mathbf { x } _ { i } ) > U ( \mathbf { x } _ { j } )$ This noise-free limit violates the strict positivity condition underlying Luce’s representation [46, Lemma 1]. As $\sigma  \infty$ , the choice probability converges to ${ \frac { 1 } { 2 } } ,$ so the responses become uninformative about utility diferences. From standard treatment of generalized linear models (see e.g., [67, Chapter 3.2], [48, Chapter 4], and [39]), we treat σ as a fixed scale parameter and do not estimate it along with w. This is justified by the fact that the response law is invariant within $[ ( \mathbf { w } , \sigma ) ] : =$ $\{ ( c \mathbf { w } , c \sigma ) \mid c \in \mathbb { R } _ { + + } \}$ unless the scale of $U ( \cdot )$ is fixed exogenously, $\mathrm { e . g . }$ , in Examples 2.1, 2.3, and 2.4, we set $\sigma = 1$ . When the utility function is normalized as in Example 2.2, one may select a unique representation from $[ ( \mathbf { w } , \sigma ) ]$ as in [14, Theorem 1], but the interpretation of σ remains relative to that normalization.

## 2.2 Questionnaire and Responses

Let $\pmb { \alpha } : = ( i , j ) \in \mathbb { I }$ denote a pairwise query, and define

$$
\mathfrak { A } : = [ \phi ( \mathbf { x } _ { 1 } ) , \ldots , \phi ( \mathbf { x } _ { d } ) ] ^ { \top } \in \mathbb { R } ^ { d \times p } ,\tag{2.5a}
$$

$$
\begin{array} { r } { \mathbf { a } _ { \alpha } : = \mathfrak { A } ^ { \top } ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) . } \end{array}\tag{2.5b}
$$

$\mathbf { a } _ { \alpha }$ is called a query vector associated with query $\pmb { \alpha }$ , which is also known as a query direction. Next, we discuss how the sequence of responses is generated and specify several mild conditions on the ground-truth parameter underlying the responses. By substituting (2.4) and (2.5b) into (1.3), we obtain a scalar choice probability in terms of $\mathbf { a } _ { \alpha }$ parameterized by w

$$
p _ { \alpha } ( \mathbf { w } ) : = \mathbb { P } ( i \mid \{ i , j \} ) = \frac { 1 } { 1 + \exp \left( - \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } \right) } .\tag{2.6}
$$

Let $\mathcal { E } _ { s } \subseteq \mathbb { I }$ be the selected pairs, and let $n _ { \alpha } \geq 1$ denote the number of responses collected for pair α $\in { \mathcal { E } } _ { s }$ . Repeated presentations of the same query are allowed but not required for our theoretical results to hold. Thus, $\mathcal { E } _ { s }$ and $\{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } }$ specify the questionnaire design. For query $\alpha ,$ , let $( \mathscr { y } _ { \alpha } , \mathscr { y } _ { \alpha } ) : = ( \{ 0 , 1 \} , 2 ^ { \{ 0 , 1 \} } )$ , and let $\mathrm { P } _ { \mathbf { w } } ^ { \alpha }$ denote the Bernoulli probability measure on this space with success rate $p _ { \alpha } ( \mathbf { w } )$ . The response to presentation $t \in [ n _ { \alpha } ]$ is denoted by $Y _ { \alpha } ^ { ( t ) }$ , with value one when $i$ is chosen and zero otherwise. Its distribution under the BTL model (2.6) is $\mathrm { P } _ { \mathbf { w } } ^ { \alpha }$ . The next proposition describes this via the exponential-family representation.

Proposition 2.1 ([48, Chapters 2 and 4]) For any but fixed α and $t \in [ n _ { \alpha } ]$ , suppose that $Y _ { \alpha } ^ { ( t ) } \sim \mathrm { P } _ { \bf w } ^ { \alpha }$ . Then the probability mass function of $Y _ { \alpha } ^ { ( t ) }$ , evaluated at $y \in \{ 0 , 1 \}$ , can be expressed by

$$
f _ { Y _ { \alpha } ^ { ( t ) } } ( y \mid \mathbf { w } ) = \exp \left[ y \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } - \log \left( 1 + e ^ { \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } } \right) \right] .\tag{2.7}
$$

Moreover, let $\Psi ( \eta ) : = \log ( 1 + e ^ { \eta } )$ . Then $\begin{array} { r } { \mathbb { E } _ { \mathbf { w } } ^ { \alpha } [ Y _ { \alpha } ^ { ( t ) } ] = \Psi ^ { \prime } \left( \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } \right) } \end{array}$ and $\begin{array} { r } { \operatorname { V a r } _ { \bf w } ^ { \alpha } [ Y _ { \alpha } ^ { ( t ) } ] = \Psi ^ { \prime \prime } \left( \frac { { \bf w } ^ { \top } { \bf a } _ { \alpha } } { \sigma } \right) } \end{array}$ where E<sup>α</sup>[Y ] and $\mathrm { V a r } _ { \mathbf { w } } ^ { \alpha } [ Y ]$ denote the expectation and the variance under $\mathrm { P } _ { \mathbf { w } } ^ { \alpha } ,$ respectively.

The proof can be found in, $\mathrm { { e . g . , } \ [ 4 8 }$ , Chapter 2.2.2]. $\Psi ( \eta )$ is also known as the cumulant function, as its derivatives determine the moment statistics of $Y _ { \alpha } ^ { ( t ) }$ , with $| \Psi ^ { ( 3 ) } ( \eta ) | \le \Psi ^ { \prime \prime } ( \eta )$ ， $| \Psi ^ { ( 4 ) } ( \eta ) | \le \Psi ^ { \prime \prime } ( \eta ) , \forall \eta \in \mathbb { R }$ . Equation (2.7) has the canonical logistic regression form given a single sample-covariate pair $( Y _ { \alpha } ^ { ( t ) } , \frac { { \bf a } _ { \alpha } } { \sigma } )$ , and the structure of the collection of such pairs may be conveniently described by the notion of comparison graph (see e.g. [61; 7; 40; 80]).

Definition 2.2 (Comparison Graph) Consider an undirected weighted graph $\mathcal { G } ( [ d ] , \mathcal { E } _ { s } , n _ { \alpha } )$ with vertex set [d], edge set $\mathcal { E } _ { s } \subseteq \mathbb { I }$ , and edge weight $n _ { \alpha }$ . It is called a comparison graph because any two distinct vertices i and $j$ are adjacent if the pair $\boldsymbol { \alpha } = ( i , j )$ is presented to the DM at least once. The edge weight is equal to the number of times the corresponding query is presented to the DM.

With the notion of comparison graph, we are ready to make an assumption that ensures that model (2.7) is correctly specified.

Assumption 2.1 (Well Specified Model) There exists a ground-truth parameter $\mathbf { w } ^ { \star } \in \mathbb { R } ^ { p }$ satisfying $\mathbf { A } \mathbf { w } ^ { \star } = \mathbf { b }$ , such that $Y _ { \alpha } ^ { ( t ) } \sim \mathrm { P } _ { \mathbf { w } } ^ { \alpha } ,$ <sub>⋆</sub> for every $\alpha \in \mathcal { E } _ { s } , t \in [ n _ { \alpha } ]$

Assumption 2.1 specifies the population-level response law. Testing whether the observed choices are compatible with the model is beyond the scope of this work. Ruan et al. [57, Theorem 1, Section 5] formulate the representability of a marginal distribution model (including the BTL model as a special case) as a linear feasibility problem. Applying such verification requires the choice probabilities or suitable estimates. Without repeated presentations of the same query, empirical choice frequencies provide no reliable estimate of the corresponding choice probabilities [48, Chapter 4.4.3].

Assumption 2.2 (Independent Samples) For a fixed edge set $\mathcal { E } _ { s }$ and the edge weights $\{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } }$ of the comparison graph defined as in Definition 2.2, the responses $\{ Y _ { \alpha } ^ { ( t ) } \} _ { \alpha \in \mathcal { E } _ { s } } ^ { t \in [ n _ { \alpha } ] }$ are mutually independent, and for each fixed $\pmb { \alpha } \in \mathcal { E } _ { s } , \{ Y _ { \pmb { \alpha } } ^ { ( t ) } \} _ { t \in [ n _ { \pmb { \alpha } } ] }$ are i.i.d.

The next assumption states that log odds of $p _ { \alpha } ( \mathbf { w } ^ { \star } )$ , i.e., $\frac { \mathbf { a } _ { \alpha } ^ { \top } \mathbf { w } ^ { \star } } { \sigma }$ is uniformly bounded for every queried pair.

Assumption 2.3 (Bounded Dynamic Range) There exists a positive constant $B > 1$ such that the ground truth parameter w<sup>⋆</sup> has a dynamic range bounded by log B over the set of selected queries, i.e.,

$$
\operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } | \mathbf { a } _ { \alpha } ^ { \top } \mathbf { w } ^ { \star } / \sigma | \leq \log B < \infty .\tag{2.8}
$$

The assumption follows from the bounded dynamic range condition used in ranking [61, Section 2.1] [18, Section 2.1]. When B ↓ 1, the queried choice probabilities approach ${ \frac { 1 } { 2 } } ,$ meaning that the DM becomes indiferent throughout the comparisons. On the other hand, as $B  \infty$ the bound (2.8) allows choice probabilities to approach zero or one. Assumption 2.3 is also needed for uniform statistical guarantees, i.e., by Proposition 3.2, the minimax lower bound may go to infinity without Assumption 2.3. We therefore define

$$
\begin{array} { r } { \mathcal { W } _ { B } : = \left\{ \mathbf { w } \in \mathbb { R } ^ { p } \big | \mathbf { A } \mathbf { w } = \mathbf { b } , \big | \mathbf { a } _ { \alpha } ^ { \top } \mathbf { w } / \sigma \big | \leq \log B , \forall \alpha \in \mathcal { E } _ { s } \right\} . } \end{array}\tag{2.9}
$$

Proposition 2.2 below is a standard consequence of Assumption 2.3, we restate it here for completeness. It shows that $\mathrm { V a r } _ { \mathbf { w } } ^ { \alpha } ( Y _ { \alpha } ^ { ( t ) } )$ is uniformly bounded away from zero over $\mathcal { W } _ { B }$

Proposition 2.2 (Non-degenerate Variance) Under Assumption 2.3,

$$
\frac { 1 } { 4 B } \leq \mathrm { V a r } _ { \mathbf { w } } ^ { \alpha } ( Y _ { \alpha } ^ { ( t ) } ) \leq \frac { 1 } { 4 } , \forall \mathbf { w } \in \mathcal { W } _ { B } , \alpha \in \mathcal { E } _ { s } , t \in [ n _ { \alpha } ] .\tag{2.10}
$$

We omit the proof as it follows directly by mimicking [61, Appendix A.2] or [76, Lemma 13].

## 2.3 Identifiability and Likelihood Function

This section serves a twofold purpose. First, we remove the equality constraints by reparameterization and establish the identifiability condition of the general model introduced above through the joint structure of the features, constraints, and comparison graph. We then deduce the likelihood function and express the same condition equivalently by positive definiteness of an approximation of the FIM.

## 2.3.1 Identifiability

Let $r _ { A } : = \mathrm { r a n k } ( \mathbf { A } )$ , and let $\mathbf { V } _ { A ^ { \perp } } \in \mathbb { R } ^ { p \times ( p - r _ { A } ) }$ have orthonormal columns that span $\operatorname { N u l l } ( \mathbf { A } )$ With $\mathbf { w } _ { o } : = \mathbf { A } ^ { \dagger } \mathbf { b }$ being the minimum-norm solution of $\mathbf { A } \mathbf { w } \ = \ \mathbf { b }$ , every feasible w has the representation

$$
\mathbf { w } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb \theta , \quad \pmb \theta \in \mathbb { R } ^ { d _ { \theta } } , \quad d _ { \theta } : = p - r _ { A } .\tag{2.11}
$$

Since $\mathbf { V } _ { A ^ { \perp } }$ <sub>⊥</sub> has orthogonal columns, the map $\pmb { \theta } \mapsto \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb { \theta }$ is injective. Therefore $\mathcal { W } _ { B }$ induces the equivalent parameter space

$$
\begin{array} { r } { \Theta _ { B } : = \left\{ \pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } } \ \big | \ \big | \mathbf { a } _ { \alpha } ^ { \top } \big ( \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb { \theta } \big ) / \sigma \big | \leq \log B , \forall \alpha \in \mathcal { E } _ { s } \right\} . } \end{array}\tag{2.12}
$$

Assumption 2.4 Let B be given as in Assumption 2.3 and $B _ { 0 }$ be a positive constant such that $\begin{array} { r } { \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \left| \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } / \sigma \right| = \log ( B _ { 0 } ) } \end{array}$ . Assume that $B > B _ { 0 }$

By Assumption 2.1, we may denote by $\theta ^ { \star } \in \Theta _ { B }$ the reparameterized ground truth satisfying $\mathbf { w } ^ { \star } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \bot } } \pmb { \theta } ^ { \star }$ . For the pairwise ranking model, setting $\mathbf { A } = \mathbf { 1 } ^ { \top }$ and $\mathbf { b } = 0$ gives rise to ${ \bf w } _ { o } = { \bf 0 }$ and makes the columns of $\mathbf { V } _ { A } .$ <sub>⊥</sub> an orthonormal basis of $\{ \mathbf { 1 } \} ^ { \perp }$ , Assumption 2.4 then follows from $B > 1$ . The self-centering constraint removes the translation ambiguity inherent in the ranking model, otherwise, the vectors w and $\mathbf { w } + c \mathbf { 1 }$ induce the same choice probability for every constant $c ,$ see [40; 61]. An analogous result holds under the general setup in this paper. Its proof is deferred to Appendix $\mathrm { A . 1 }$

Proposition 2.3 (Identifiability) Assume that the comparison graph $\mathcal { G } ( [ d ] , \mathcal { E } _ { s } , n _ { \alpha } )$ has K connected components, i.e., there exist subsets $\mathcal { C } _ { k } \subseteq [ d ] , f o r k \in [ K ]$ such that $\begin{array} { r } { [ d ] = \bigcup _ { k \in [ K ] } \mathcal { C } _ { k } } \end{array}$ and $\mathcal { C } _ { k } \cap \mathcal { C } _ { l } = \mathcal { O }$ for any $k \neq l$ . Let $\mathbf { C } _ { I } : = [ \mathbb { 1 } _ { \mathcal { C } _ { 1 } } , \dots , \mathbb { 1 } _ { \mathcal { C } _ { K } } ] \in \mathbb { R } ^ { d \times K }$ where the $k - t h$ column $o f \mathbf { C } _ { I }$ is the indicator vector of $\mathcal { C } _ { k } , \ i . e . , \ [ \mathbb { 1 } _ { \mathcal { C } _ { k } } ] _ { i } = \left\{ { \begin{array} { l l } { 1 , \ } & { \ f o r \ i \in \mathcal { C } _ { k } } \\ { 0 , \ } & { \ f o r \ i \not \in \mathcal { C } _ { k } } \end{array} } \right.$ . Then the following two statements are equivalent.

(i) For any w, $\mathbf { w } ^ { \prime } \in \{ \mathbf { w } \in \mathbb { R } ^ { p } \mid \mathbf { A w } = \mathbf { b } \} , \{ p _ { \alpha } ( \mathbf { w } ) = p _ { \alpha } ( \mathbf { w } ^ { \prime } ) \} _ { \alpha \in \mathcal { E } _ { s } }$ <sub>s</sub> implies $\mathbf { w } = \mathbf { w } ^ { \prime }$

(ii) The joint rank condition holds, i.e.,

$$
\operatorname { r a n k } \left( \left[ \mathbf { C } _ { I } , \mathfrak { 2 } \mathbf { 1 } \mathbf { V } _ { A ^ { \perp } } \right] \right) = d _ { \theta } + K .\tag{2.13}
$$

Consequently, under Assumptions 2.1 and 2.3, condition (2.13) uniquely identifies the groundtruth parameter $\mathbf { w ^ { \star } } \in \mathcal { W } _ { B }$ from the pairwise choice probabilities $\{ p _ { \alpha } ( \mathbf { w } ^ { \star } ) \} _ { \alpha \in \mathcal { E } _ { s } }$ Part (ii) of the proposition, partly inspired by [22, Proposition 1], provides a verifiable condition for identifiability, and is equivalent to the positive definiteness of the design matrix, as shown in Corollary 2.1 below. Unlike classical ranking models [61; 31; 18; 76], the general linear utility model in Definition 2.1 can remain identifiable on a disconnected comparison graph as the utility parameters are shared across alternatives. Examples 2.5 and 2.6 illustrate the disconnected and connected cases.

Example 2.5 (Discrete Choice Model) This example corresponds to the unconstrained case, where $r _ { A } = 0 , \ \mathbf { V } _ { A ^ { \perp } } = \mathbf { I } _ { p } ,$ , and ${ \bf w } _ { o } ~ = ~ { \bf 0 }$ . For instance, let $n ~ = ~ p ~ = ~ d _ { \theta } ~ = ~ 2 , ~ d ~ = ~ 4$ and the comparison graph have two connected components $\mathcal { C } _ { 1 } = \{ 1 , 2 \}$ and $\mathcal { C } _ { 2 } = \{ 3 , 4 \}$ . Let $\mathbf { x } _ { 1 } = [ 0 , 0 ] ^ { \top } , ~ \mathbf { x } _ { 2 } = [ 1 , 1 ] ^ { \top } , ~ \mathbf { x } _ { 3 } = [ 0 , 1 ] ^ { \top } , ~ \mathbf { x } _ { 4 } = [ 1 , 0 ] ^ { \top }$ , and $\mathfrak { A } = [ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \mathbf { x } _ { 3 } , \mathbf { x } _ { 4 } ] ^ { \top }$ . Then the matrix

$$
[ \mathbb { 1 } _ { \mathcal { C } _ { 1 } } , \mathbb { 1 } _ { \mathcal { C } _ { 2 } } , \mathfrak { A } ] = \left[ \begin{array} { l l l l } { 1 } & { 0 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 1 } & { 1 } \\ { 0 } & { 1 } & { 0 } & { 1 } \\ { 0 } & { 1 } & { 1 } & { 0 } \end{array} \right] ,
$$

has rank $4 = d _ { \theta } + K$ , where $K = 2$ is the number of connected components.

Example 2.6 (Covariate Assisted Ranking) In [22, Proposition $1 ] ,$ the authors show that, $i f$ the comparison graph is further assumed to be connected, then the equality constraint in $E x -$ ample $\it 2 . 4$ always ensures identifiability. To see this, let $d = 4 , n = 1 , K = 1$ , and take the non-binary covariates as $x _ { 1 } = 0 , x _ { 2 } = 7 , x _ { 3 } = 2 , x _ { 4 } = 1$ . Then the feature matrix A and linear constraint matrix A are

$$
\mathfrak { A } = \left[ \begin{array} { c c c c c } { 1 } & { 0 } & { 0 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } & { 7 } \\ { 0 } & { 0 } & { 1 } & { 0 } & { 2 } \\ { 0 } & { 0 } & { 0 } & { 1 } & { 1 } \end{array} \right] , \mathbf { A } = \left[ \begin{array} { c c c c c } { 1 } & { 1 } & { 1 } & { 1 } & { 0 } \\ { 0 } & { 7 } & { 2 } & { 1 } & { 0 } \end{array} \right] ,
$$

which gives rank $\mathbf { \partial } ( \mathbf { A } ) = 2 , p = d + n = 5$ , and $d _ { \theta } = 3$ . We may use any basis of Null(A) in place of the orthonormal basis $\mathbf { V } _ { A ^ { \perp } } \ f o r$ the rank check, $e . g .$ ，

$$
\mathbf { v } _ { 1 } = [ - 6 , - 1 , 0 , 7 , 0 ] ^ { \top } , \mathbf { v } _ { 2 } = [ - 5 , - 2 , 7 , 0 , 0 ] ^ { \top } , \mathbf { v } _ { 3 } = [ 0 , 0 , 0 , 0 , 1 ] ^ { \top } ,
$$

and therefore

$$
[ \mathbf { 1 } , \mathfrak { A } [ \mathbf { v } _ { 1 } , \mathbf { v } _ { 2 } , \mathbf { v } _ { 3 } ] ] = \left[ \begin{array} { c c c c } { 1 } & { - 6 } & { - 5 } & { 0 } \\ { 1 } & { - 1 } & { - 2 } & { 7 } \\ { 1 } & { 0 } & { 7 } & { 2 } \\ { 1 } & { 7 } & { 0 } & { 1 } \end{array} \right] ,
$$

has rank $4 = d _ { \theta } + K = 3 + 1$

## 2.3.2 Likelihood Function

We now derive the MLE of $\pmb { \theta } ^ { \star }$ under the above reparameterization. Let $\begin{array} { r } { N : = \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } } \end{array}$ denote the total number of comparisons. For the fixed deterministic design $( { \mathcal E } _ { s } , \{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } } )$ , define the full sample space $( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ , where

$$
\mathcal { Y } _ { N } : = \prod _ { \alpha \in \mathcal { E } _ { s } } \prod _ { t = 1 } ^ { n _ { \alpha } } \mathcal { Y } _ { \alpha } , \quad \mathcal { Y } _ { N } : = \bigotimes _ { \alpha \in \mathcal { E } _ { s } } \bigotimes _ { t = 1 } ^ { n _ { \alpha } } \mathcal { Y } _ { \alpha } .
$$

The canonical sample variable $Y ^ { ( N ) } : \mathcal { V } _ { N } \to \mathcal { V } _ { N }$ consists of the coordinate projections $Y _ { \alpha } ^ { ( t ) }$ $\mathcal { V } _ { \mathcal { N } } \to \mathcal { V } _ { \alpha }$ and is identified with an element in $\{ 0 , 1 \} ^ { N }$ under a fixed ordering of the index set. Let $\mathcal { P } ( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ denote the set of all probability measures over $( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ . For each $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ and $\pmb { \alpha } \in \mathcal { E } _ { s }$ , define $\mathrm { P } _ { \pmb \theta } ^ { \alpha } : = \mathrm { P } _ { \mathbf { w } _ { o } + \mathbf { V } _ { A } \bot \pmb \theta } ^ { \alpha } \cdot$ . Under Assumption 2.2, the joint distribution of $Y ^ { ( N ) }$ is the product of the marginals $\mathrm { P } _ { \theta } ^ { \alpha }$

$$
\mathbb { P } _ { \pmb { \theta } } ^ { ( N ) } : = \bigotimes _ { \pmb { \alpha } \in \mathcal { E } _ { s } } ( \mathrm { P } _ { \pmb { \theta } } ^ { \alpha } ) ^ { \otimes n _ { \alpha } } \in \mathcal { P } ( \mathcal { Y } _ { N } , \mathcal { Y } _ { N } ) .\tag{2.14}
$$

Any measurable statistic $\hat { T } = \hat { T } ( Y ^ { ( N ) } )$ on $( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ is a random element under $\mathbb { P } _ { \pmb { \theta } } ^ { ( N ) }$ , whose explicit dependence on $Y ^ { ( N ) }$ is suppressed whenever there is no confusion in context. For an integrable random variable Z on $( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ , let $\begin{array} { r } { \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } [ Z ] : = \int _ { \mathcal { V } _ { N } } Z ( \pmb { y } ) \mathbb { P } _ { \pmb { \theta } } ^ { ( N ) } ( \mathrm { d } \pmb { y } ) } \end{array}$ , and we use $\operatorname { V a r } _ { \pmb { \theta } } ^ { ( N ) } ( \cdot )$ and $\mathrm { C o v } _ { \pmb { \theta } } ^ { ( N ) } ( \cdot )$ analogously.

Under Assumptions 2.1 and 2.2, it follows by Proposition 2.1 that the likelihood function of $Y ^ { ( N ) }$ given parameter w is

$$
f _ { Y ^ { ( N ) } } ( Y ^ { ( N ) } \mid \mathbf { w } ) = \prod _ { \alpha \in \mathcal { E } _ { s } } \prod _ { t = 1 } ^ { n _ { \alpha } } \exp \left[ Y _ { \alpha } ^ { ( t ) } \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } - \log \left( 1 + \exp \left( \frac { \mathbf { w } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } \right) \right) \right] .\tag{2.15}
$$

The reparameterized negative log-likelihood can then be written as

$$
\begin{array} { l } { \displaystyle \ell ( \theta ) : = - \frac { 1 } { N } \log \left( f _ { Y ^ { ( N ) } } \left( Y ^ { ( N ) } \mid \mathbf { w } _ { o } + \mathbf { V } _ { A } . \theta \right) \right) } \\ { \displaystyle \quad = \frac { 1 } { N } \sum _ { \alpha \in \mathcal { E } _ { s } } \left[ - Y _ { \alpha } \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \theta ^ { \top } \mathbf { V } _ { A } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } \right) + n _ { \alpha } \Psi \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \theta ^ { \top } \mathbf { V } _ { A } ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } \right) \right] , } \end{array}\tag{2.16}
$$

where $\begin{array} { r } { Y _ { \pmb { \alpha } } : = \sum _ { t = 1 } ^ { n _ { \pmb { \alpha } } } Y _ { \pmb { \alpha } } ^ { ( t ) } . \{ Y _ { \pmb { \alpha } } \} _ { \pmb { \alpha } \in \mathcal { E } _ { s } } } \end{array}$ is a suficient statistic under the model specified above and $\Psi ( \cdot )$ is defined as in Proposition 2.1.

Definition 2.3 (MLE and Firth Correction) Under the identifiability condition (2.13), the MLE is the unique solution of the unconstrained convex program

$$
{ \hat { \pmb \theta } } \in \arg \operatorname* { m i n } _ { { \pmb \theta } \in \mathbb { R } ^ { d _ { \theta } } } \ell ( { \pmb \theta } ) ,\tag{2.17}
$$

provided that it exists. The corresponding estimator in the original parameterization is recovered with $\hat { \mathbf { w } } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \hat { \pmb { \theta } }$ . When $\ell ( \pmb \theta )$ is regularized by the Jefreys’ invariant prior,

$$
\hat { \pmb { \theta } } _ { \mathrm { F i r t h } } \in \arg \operatorname* { m i n } _ { \pmb { \theta } \in \mathbb { R } ^ { d _ { \pmb { \theta } } } } \ell ( \pmb { \theta } ) - \frac { 1 } { 2 N } \log \operatorname* { d e t } ( \nabla ^ { 2 } \ell ( \pmb { \theta } ) ) ,\tag{2.18}
$$

is called the Firth correction [23].

Note that MLE may not exist in that $\ell ( \pmb \theta )$ is not coercive due to the separation in the realization of response data [3]. In contrast, since the regularization term in (2.18) goes to infinity when $\| \pmb \theta \| _ { \infty }  \infty$ , the existence of $\widehat { \pmb { \theta } } _ { \mathrm { F i r t h } }$ is always guaranteed under the identifiability condition [37, Theorem 1]. In Sections 3 and 4 we will derive error bounds of MLE under various metrics, and use Firth correction for numerical comparison in Section 5.

For notational simplicity, define

$$
\mathbf { q } _ { \alpha } : = \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathbf { a } _ { \alpha } , \quad \eta _ { \alpha } ( \pmb \theta ) : = \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \pmb \theta } { \sigma } , \quad \alpha \in \mathcal { E } _ { s } ,\tag{2.19}
$$

where $\eta _ { \alpha } ( \pmb \theta )$ denotes the log odds of choice probability $p _ { \alpha } ( \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb \theta )$ . By the chain rule

$$
\nabla \ell ( \pmb \theta ) = \frac { 1 } { N \sigma } \sum _ { \alpha \in \mathcal { E } _ { s } } \left[ - Y _ { \alpha } + n _ { \alpha } \Psi ^ { \prime } \big ( \eta _ { \alpha } ( \pmb \theta ) \big ) \right] \mathbf q _ { \alpha } ,\tag{2.20a}
$$

$$
\nabla ^ { 2 } \ell ( \pmb \theta ) = \frac { 1 } { N \sigma ^ { 2 } } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \Psi ^ { \prime \prime } \big ( \eta _ { \alpha } ( \pmb \theta ) \big ) \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } .\tag{2.20b}
$$

Equation (2.20b) shows that the diference between $\nabla ^ { 2 } \ell ( \pmb \theta )$ and $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } )$ is governed by the query-wise predictor deviations $\begin{array} { r } { \eta _ { \alpha } ( \pmb { \theta } ) - \eta _ { \alpha } ( \pmb { \theta } ^ { \star } ) = \frac { \ P _ { \alpha } ^ { \top } ( \pmb { \theta } - \pmb { \theta } ^ { \star } ) } { \sigma } } \end{array}$ . This observation motivates us to

develop lower and upper bounds on this quantity, we will come back to this in Theorems 3.2 and 4.1, respectively. The FIM can be subsequently calculated from (2.20b)

$$
\mathcal { Z } ( \pmb { \theta } ^ { \star } ) : = \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } [ \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ] = \frac { 1 } { \sigma ^ { 2 } } \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathfrak { A } ^ { \top } \left[ \frac { 1 } { N } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \Psi ^ { \prime \prime } \left( \eta _ { \alpha } ( \pmb { \theta } ^ { \star } ) \right) \left( \mathbf { e } _ { i } - \mathbf { e } _ { j } \right) \left( \mathbf { e } _ { i } - \mathbf { e } _ { j } \right) ^ { \top } \right] \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ,\tag{2.21}
$$

where the second equality follows from (2.19). Note that the graph Laplacian of the comparison graph $\mathcal { G } ( [ d ] , \mathcal { E } _ { s } , \{ n _ { \alpha } \} )$ is defined as

$$
\mathbf { L } : = \frac { 1 } { N } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ^ { \top } .\tag{2.22}
$$

By Propositions 2.1 and 2.2, it is then straightforward to verify that the bracketed term in (2.21) is an approximation of L up to multiplicative constants depending only on B, i.e., for each $\pmb { \alpha } \in \mathcal { E } _ { s }$ $\begin{array} { r } { \frac { 1 } { 4 B } \leq \Psi ^ { \prime \prime } ( \eta _ { \alpha } ( \pmb \theta ^ { \star } ) ) \leq \frac { 1 } { 4 } } \end{array}$ implies that

$$
\frac { 1 } { 4 B } { \bf L } \preceq \frac { 1 } { N } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \Psi ^ { \prime \prime } \left( \eta _ { \alpha } ( \pmb { \theta } ^ { \star } ) \right) ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ^ { \top } \preceq \frac { 1 } { 4 } { \bf L } .\tag{2.23}
$$

For fixed $\mathcal { E } _ { s }$ , the empirical frequencies $\frac { Y _ { \alpha } } { n _ { \alpha } }$ converge in probability to $p _ { \alpha } ( \mathbf { w } ^ { \star } )$ as $n _ { \alpha } \to \infty$ . Thus the population-level problem concerns the recovery of parameters from these choice probabilities. The identifiability of population-level MLE is stated as a corollary of Proposition 2.3. Its proof is deferred to Appendix A.2.

Corollary 2.1 Let $\mathbf { V } _ { A ^ { \perp } } , \mathfrak { A }$ , and L be defined as in (2.11), (2.5), and (2.22), respectively. The rank condition (2.13) holds if the matrix

$$
\mathbf { W } : = \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathfrak { A } ^ { \top } \mathbf { L } \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \in \mathbb { R } ^ { d _ { \theta } \times d _ { \theta } } ,\tag{2.24}
$$

is positive definite. Moreover, if (2.13) fails, let $\mathbf { Q } _ { + }$ and $\mathbf { Q } _ { \perp }$ have orthonormal columns spanning the range and null space of W, respectively. Augmenting Aw = b with $( \mathbf { V } _ { A ^ { \perp } } \mathbf { Q } _ { \perp } ) ^ { \top } \mathbf { w } = \mathbf { 0 }$ yields an identifiable model. Further, for any ground-truth parameter $\mathbf { w } ^ { \star } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \bot } } \pmb { \theta } ^ { \star }$ satisfying Assumption 2.1, there exists an identifiable representative

$$
\tilde { \mathbf { w } } ^ { \star } : = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } ,\tag{2.25}
$$

such that $\tilde { \mathbf { w } } ^ { \star }$ is the unique parameter in the augmented model satisfying $p _ { \alpha } ( \tilde { \mathbf { w } } ^ { \star } ) = p _ { \alpha } ( \mathbf { w } ^ { \star } )$ for every $\pmb { \alpha } \in \mathcal { E } _ { s }$

Equation (2.25) projects $\pmb { \theta } ^ { \star }$ onto the span of the selected query directions without changing the queried choice probabilities. Unlike the classical ranking models, however, the comparison graph enters the information geometry through both the feature matrix A and the null space of A, so that connectivity alone is no longer decisive. For fixed A and A, the selected edges and their multiplicities determine W, which approximates the FIM through (2.21)-(2.23). In Sections 3 and 4, we will describe this observation with finite-sample theory.

## 3 Lower Bounds

Classical likelihood theory states that the MLE is asymptotically normal, with covariance $\frac { \mathcal { I } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } } { N }$ to first order [72, Chapters 5, 7] under regularity and non-singularity conditions, but does not specify the finite-sample accuracy of this approximation. In this section, we address part of these limitations from two complementary perspectives. In Proposition 3.1, we derive the Cram´er–Rao inequality for locally unbiased estimators at a fixed ground-truth parameter. Theorems 3.1 and 3.2 then establish minimax lower bounds over $\Theta _ { B }$ for estimators with finite worst-case risk. Together, these results identify the information geometry relevant to the local estimation and worst-case finite-sample dificulty. We work in the reparameterized model<sup>1</sup>.

## 3.1 The Cram´er-Rao Lower Bound

We begin with a classic local result stated for a fixed ground-truth parameter. Fix $q \in \mathbb { Z } _ { + }$ with $q \leq d _ { \theta }$ , and consider a continuously diferentiable vector-valued mapping $h : \Theta _ { B } \to \mathbb { R } ^ { q }$ . An estimator of $h ( \pmb \theta )$ is a measurable mapping $\hat { \psi } : ( \mathcal { V } _ { N } , \mathcal { Y } _ { N } )  ( \mathbb { R } ^ { q } , \mathcal { B } ( \mathbb { R } ^ { q } ) )$ . Given $\pmb { \theta } ^ { \star } \in \operatorname { i n t } ( \Theta _ { B } )$ , we say $\hat { \psi }$ is a locally unbiased estimator $\mathrm { i f ^ { 2 } }$

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } [ \hat { \psi } ] = h ( \pmb { \theta } ^ { \star } ) , \quad \mathrm { a n d } \quad \nabla _ { \pmb { \theta } } \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } [ \hat { \psi } ] \Big | _ { \pmb { \theta } = \pmb { \theta } ^ { \star } } = \nabla h ( \pmb { \theta } ^ { \star } ) .
$$

Proposition 3.1 (CRLB) Let Assumptions 2.1 and 2.2 hold, and ${ \mathcal { T } } ( \theta ^ { \star } )$ be defined as in (2.21) where $\pmb { \theta } ^ { \star } \in \operatorname { i n t } ( \Theta _ { B } )$ . Let $\hat { \psi } : \mathcal { V } _ { N } \to { \mathbb R } ^ { q }$ be any locally unbiased estimator with $\bar { \mathbb { E } } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } [ \| \hat { \psi } \| _ { 2 } ^ { 2 } ] \stackrel { \cdot } { < } \infty$ Then

$$
\operatorname { C o v } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } ( \hat { \psi } ) \succeq \frac { \nabla h ( \pmb { \theta } ^ { \star } ) ^ { \top } \mathcal { Z } ( \pmb { \theta } ^ { \star } ) ^ { \dagger } \nabla h ( \pmb { \theta } ^ { \star } ) } { N } ,\tag{3.26a}
$$

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \| \hat { \pmb { \psi } } - h ( \pmb { \theta } ^ { \star } ) \| _ { 2 } ^ { 2 } \right] \geq \frac { \mathrm { t r } ( \nabla h ( \pmb { \theta } ^ { \star } ) ^ { \top } \mathcal { I } ( \pmb { \theta } ^ { \star } ) ^ { \dagger } \nabla h ( \pmb { \theta } ^ { \star } ) ) } { N } .\tag{3.26b}
$$

We omit the proof as it directly follows from the proof of the Cram´er–Rao inequality via Schur complement, see [55, Section 6.1.1]. Proposition 3.1 covers two specific cases when $\hat { \psi }$ and $h ( \theta )$ take particular forms, the next corollary addresses this. These lower error bounds will be used to compare with the minimax rates in Section 3.2.

Corollary 3.1 (Lower error bounds for locally unbiased estimators of $\pmb { \theta } ^ { \star } )$ Let $\hat { \pmb { \theta } }$ be any estimator of $\pmb \theta$ which is locally unbiased at $\pmb { \theta } ^ { \star }$ . Then the following assertions hold.

(i) Let ${ \pmb { \alpha } } \in { \mathcal { E } } _ { s }$ and $\mathbf { q } _ { \alpha }$ be defined as in (2.19). $\begin{array} { r } { I f h ( \pmb { \theta } ) = \mathbf q _ { \alpha } ^ { \top } \pmb { \theta } } \end{array}$ , then $\hat { \psi } = \mathbf q _ { \alpha } ^ { \top } \hat { \pmb \theta }$ is a locally unbiased estimator of $h ( \pmb \theta )$ at $\pmb { \theta } ^ { \star }$ , and $\hat { \pmb { \theta } }$ satisfies

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ | \mathbf { q } _ { \alpha } ^ { \top } ( \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } ) | ^ { 2 } \right] \geq \frac { \mathbf { q } _ { \alpha } ^ { \top } \mathcal { T } ( \pmb { \theta } ^ { \star } ) ^ { \dagger } \mathbf { q } _ { \alpha } } { N } .\tag{3.27}
$$

(ii) If ${ \mathcal { I } } ( \pmb { \theta } ^ { \star } ) \succ \mathbf { 0 }$ and $h ( \pmb \theta ) = \pmb \theta$ , then $\hat { \pmb { \theta } }$ satisfies

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \lVert \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \rVert _ { 2 } ^ { 2 } \right] \geq \frac { \mathrm { t r } ( \mathcal { T } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } ) } { N } .\tag{3.28}
$$

Proof. The conclusion follows straightforwardly from Proposition 3.1.

## 3.2 Minimax Lower Bounds

To formally link the Fisher information geometry with the intrinsic dificulty of the estimation problem at hand, we follow [70, Chapter 2] and [74, Chapter 15] to introduce the minimax lower bound to the utility elicitation problem. Let $\rho : \mathbb { R } ^ { d _ { \theta } } \times \mathbb { R } ^ { d _ { \theta } } \to [ 0 , \infty )$ be a measurable semi-distance, i.e., $\rho$ is $B ( \mathbb { R } ^ { d _ { \theta } } ) \otimes B ( \mathbb { R } ^ { d _ { \theta } } ) / B ( [ 0 , \infty ) )$ -measurable, and satisfies all properties of a distance, except that $\rho ( \pmb \theta , \pmb \theta ^ { \prime } ) = 0$ does not necessarily imply $\pmb \theta = \pmb \theta ^ { \prime }$ . Consider any estimator $\hat { \pmb { \theta } }$ of $\pmb \theta$ as a measurable mapping $\hat { \pmb { \theta } } : ( \mathcal { V } _ { N } , \mathcal { V } _ { N } )  ( \mathbb { R } ^ { d _ { \theta } } , \mathcal { B } ( \mathbb { R } ^ { d _ { \theta } } ) )$ . Define the class of candidate estimators

$$
\mathcal { A } _ { N } : = \{ \boldsymbol { \hat { \theta } } : \mathcal { V } _ { N }  \mathbb { R } ^ { d _ { \boldsymbol { \theta } } } \enspace \middle | \enspace \operatorname* { s u p } _ { \boldsymbol { \theta } \in \Theta _ { B } } \mathbb { E } _ { \boldsymbol { \theta } } ^ { ( N ) } [ \rho ^ { 2 } ( \boldsymbol { \hat { \theta } } , \boldsymbol { \theta } ) ] < \infty \} .\tag{3.29}
$$

If $\Theta _ { B }$ has finite $\rho \mathrm { - }$ −diameter, i.e., $\mathrm { s u p } _ { \theta , \theta ^ { \prime } \in \Theta _ { B } } \rho ( \theta , \theta ^ { \prime } ) < \infty$ , then every finite-valued estimator belongs to $\mathcal { A } _ { N }$ . One special case is when the estimator is Lipschitz continuous w.r.t. samples under the semi-distance $\rho .$ To see this, let ${ \pmb y } ^ { ( N ) } , \tilde { \pmb y } ^ { ( N ) } \in \mathcal { V } _ { N }$ be two realizations of the response sequence. If the estimator satisfies

$$
\rho ( \hat { \pmb \theta } ( \pmb y ^ { ( N ) } ) , \hat { \pmb \theta } ( \tilde { \pmb y } ^ { ( N ) } ) ) \leq L \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \bigg | \frac { \sum _ { t = 1 } ^ { n _ { \alpha } } \big ( \pmb y _ { \alpha } ^ { ( t ) } - \tilde { \pmb y } _ { \alpha } ^ { ( t ) } \big ) } { n _ { \alpha } } \bigg | , \ \forall \hat { \pmb \theta } ( \pmb y ^ { ( N ) } ) , \hat { \pmb \theta } ( \tilde { \pmb y } ^ { ( N ) } ) \in \Theta _ { B } ,
$$

where $L < \infty$ is a constant independent of the realizations, then

$$
\begin{array} { r } { \rho \left( \hat { \pmb { \theta } } ( \pmb { y } ^ { ( N ) } ) , \pmb { \theta } \right) \leq \rho \left( \hat { \pmb { \theta } } ( \pmb { y } ^ { ( N ) } ) , \hat { \pmb { \theta } } ( \tilde { \pmb { y } } ^ { ( N ) } ) \right) + \rho \left( \hat { \pmb { \theta } } ( \tilde { \pmb { y } } ^ { ( N ) } ) , \pmb { \theta } \right) < \infty , \forall \pmb { \theta } \in \Theta _ { B } , } \end{array}
$$

which implies $\hat { \pmb { \theta } } \in \mathcal { A } _ { N }$

Definition 3.1 (Minimax Risk) For any $\hat { \pmb { \theta } } \in \mathcal { A } _ { N }$ and $\pmb { \theta } \in \Theta _ { B }$ , let

$$
\begin{array} { r } { \mathcal { R } _ { N } ( \hat { \pmb { \theta } } , \pmb { \theta } ) : = \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } \left[ \rho ^ { 2 } ( \hat { \pmb { \theta } } , \pmb { \theta } ) \right] , } \end{array}\tag{3.30}
$$

denote the risk when $\hat { \pmb { \theta } }$ is used to estimate $\pmb \theta$ . The minimax risk under the squared semi-distance $\rho ^ { 2 }$ is defined by

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \rho ^ { 2 } ) : = \operatorname* { i n f } _ { \hat { \theta } \in \mathcal { A } _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathcal { R } _ { N } ( \hat { \theta } , \theta ) .\tag{3.31}
$$

Equation (3.31) describes the performance of the best possible estimator over the adversarial design of the ground truth. By specializing $\rho$ to be the Euclidean norm and assuming $\mathbf { W } \succ \mathbf { 0 } ^ { 3 }$ 2 where W is defined as in (2.24), we have

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { 2 } ^ { 2 } ) = \operatorname* { i n f } _ { \hat { \theta } \in \mathcal { A } _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } \left[ \| \hat { \pmb { \theta } } - \pmb { \theta } \| _ { 2 } ^ { 2 } \right] .\tag{3.32}
$$

The first result in this section aims to find a sharp lower bound on $\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { 2 } ^ { 2 } )$ . To this end, we introduce the coherence level.

Definition 3.2 (Coherence Level) Let $\mathbf { q } _ { \alpha }$ and W be defined as in (2.19) and (2.24) respectively. When $\mathbf { W } \succ \mathbf { 0 }$ , the quantity

$$
\mu _ { Q } : = \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \frac { 1 } { d _ { \theta } } \mathbf { q } _ { \alpha } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { \alpha }\tag{3.33}
$$

is called the coherence level of the questionnaire design.

<sup>3</sup>Indeed, $\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { 2 } ^ { 2 } )$ can be driven to infinity if W has non-trivial null space, i.e., $\begin{array} { r } { \operatorname* { s u p } _ { \pmb { \theta } \in \Theta _ { B } } \| \pmb { \theta } - \pmb { \theta } ^ { \prime } \| _ { 2 } = \infty } \end{array}$ for any finite $\pmb { \theta } ^ { \prime } \in \mathbb { R } ^ { d _ { \theta } }$ , rendering any lower bound trivial.

$\mu _ { Q }$ quantifies how evenly the selected query vectors cover the parameter space. A similar concept has also appeared in [22, Section 3], and our definition of $\mu _ { Q }$ coincides with theirs. By the standard matrix leverage identities [10, Definition 1.2] [17, Section II], $\mu _ { Q }$ satisfies $1 \leq \mu _ { Q } \leq$ $\frac { N } { ( \operatorname* { m i n } _ { \pm \varepsilon \varepsilon _ { s } } n _ { \pm \alpha } ) d _ { \theta } } { } ^ { 4 }$ . Both upper and lower bounds are attainable, as the following example shows.

Example 3.1 Consider the setting of Example ${ \it 2 . 5 , }$ with $d _ { \theta } = 2$ . Let $\mathbf { q } _ { ( 1 , 2 ) } = \left[ - 1 , - 1 \right] ^ { \top }$ and $\mathbf { q } _ { ( 3 , 4 ) } = \lbrack - 1 , 1 \rbrack ^ { \top }$ . With one observation per pair,

$$
\mathbf { W } = { \frac { 1 } { 2 } } \left( \mathbf { q } _ { ( 1 , 2 ) } \mathbf { q } _ { ( 1 , 2 ) } ^ { \top } + \mathbf { q } _ { ( 3 , 4 ) } \mathbf { q } _ { ( 3 , 4 ) } ^ { \top } \right) = \mathbf { I } _ { 2 } \succ \mathbf { 0 } ,
$$

and $\mathbf { q } _ { ( 1 , 2 ) } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { ( 1 , 2 ) } = \mathbf { q } _ { ( 3 , 4 ) } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { ( 3 , 4 ) } = 2$ , therefore $\mu _ { Q } = 1$ . Consider now the other case where $n _ { ( 1 , 2 ) } = 1 , n _ { ( 3 , 4 ) } \geq 2$ , and $N = n _ { ( 1 , 2 ) } + n _ { ( 3 , 4 ) }$ . In this case,

$$
\mathbf { W } = \frac { 2 } { N } \left( \frac { \mathbf { q } _ { ( 1 , 2 ) } } { \sqrt { 2 } } \left( \frac { \mathbf { q } _ { ( 1 , 2 ) } } { \sqrt { 2 } } \right) ^ { \top } + n _ { ( 3 , 4 ) } \frac { \mathbf { q } _ { ( 3 , 4 ) } } { \sqrt { 2 } } \left( \frac { \mathbf { q } _ { ( 3 , 4 ) } } { \sqrt { 2 } } \right) ^ { \top } \right) \succ \mathbf { 0 } ,
$$

where the middle term is the spectral decomposition of W. Thus $\mathbf { q } _ { ( 1 , 2 ) } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { ( 1 , 2 ) } = N$ and $\begin{array} { r } { \mathbf { q } _ { ( 3 , 4 ) } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { ( 3 , 4 ) } = \frac { N } { n _ { ( 3 , 4 ) } } } \end{array}$ . Subsequently, $\begin{array} { r } { \mu _ { Q } = \frac { 1 } { 2 } \operatorname* { m a x } ( N , \frac { N } { n _ { ( 3 , 4 ) } } ) = \frac { N } { 2 } } \end{array}$ , which attains the upper bound. Moreover, we can see that $\mu _ { Q }$ is afected by both the structure of $\{ \mathbf { q } _ { \alpha } \} _ { \alpha \in \mathcal { E } _ { s } }$ and $\{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } }$

We will come back to discuss the role of $\mu _ { Q }$ in Theorem 3.2. With all the notation introduced above, we prove the following result, its proof is deferred to Section 7.1.1.

Theorem 3.1 Let Assumptions $\it { 2 . 1 - 2 . 4 }$ hold, and $\mathbf { W } \succ \mathbf { 0 }$ . Let $\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { 2 } ^ { 2 } )$ be defined as in (3.32). Then

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { 2 } ^ { 2 } ) \gtrsim \sigma ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} ,\tag{3.34a}
$$

$$
\operatorname* { i n f } _ { \hat { \theta } \in \mathcal { A } _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { P } _ { \theta } ^ { ( N ) } \left\{ \| \hat { \pmb { \theta } } - \pmb { \theta } \| _ { 2 } ^ { 2 } \geq \frac { \sigma ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } { 1 6 } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} \right\} \geq \frac { 1 } { 1 5 } .\tag{3.34b}
$$

We are particularly interested in the case where the minimum on the right-hand side of (3.34a) is attained by the first term, namely when $\begin{array} { r } { N \ge \frac { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } { \log ^ { 2 } ( B / B _ { 0 } ) } } \end{array}$ , the minimax lower bound scales as $\frac { \sigma ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } { N }$ . For fixed B, this isolates the geometry of the questionnaire design in the minimax lower bound, including the case when W is nearly singular. We will discuss this phenomenon in detail in Corollary 4.2.

To see necessities of the assumptions, we note that Assumptions 2.1 and 2.2 specify the basic probability model whereas Assumptions 2.3 and 2.4 play an important role. For fixed W, the lower bound in (3.34) tends to zero when $B \downarrow B _ { 0 }$ . This trivial case is excluded by Assumption 2.4. On the other hand, if we allow $B = \infty$ (violating Assumption 2.3), then the following proposition holds.

Proposition 3.2 Suppose Assumptions 2.1 and 2.2 hold, and let $\mathbf { W } \succ \mathbf { 0 }$ be fixed. Then for every fixed $N < \infty$ , and any estimator $\hat { \textbf { \theta } } _ { O f } \pmb { \theta }$ that may depend on B

$$
\operatorname* { l i m } _ { B \to \infty } \operatorname* { s u p } _ { \pmb { \theta } \in \Theta _ { B } } \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } \left[ \lVert \hat { \pmb { \theta } } - \pmb { \theta } \rVert _ { 2 } ^ { 2 } \right] = \infty .\tag{3.35}
$$

The proof is deferred to Appendix A.3. Equation (3.35) implies that if $\Theta _ { B }$ is unbounded, every finite-valued estimator has infinite worst-case risk. Moreover, when $B = \infty , \mathcal { A } _ { N } = \emptyset$ . Both cases render the minimax risk being infinity.

Remark 3.1 It might be interesting to discuss the case when B is fixed whereas W and N are allowed to vary. We concentrate on the case when $\begin{array} { r } { N \ge \frac { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } { \log ^ { 2 } ( B / B _ { 0 } ) } } \end{array}$

(i) Theorem 3.1 gives a lower bound on the worst-case risk over $\Theta _ { B }$ . In particular, it provides a non-asymptotic lower bound for the MLE analysis in Section $\it 4 .$ From (2.21)–(2.24), we have $\begin{array} { r } { \frac { 1 } { 4 \sigma ^ { 2 } } \mathbf { W } \succeq \mathcal { T } ( \pmb { \theta } ^ { \star } ) \succeq \frac { 1 } { 4 B \sigma ^ { 2 } } \mathbf { W } } \end{array}$ , i.e., $\sigma ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } )$ and $\operatorname { t r } ( { \mathcal { I } } ( \theta ^ { \star } ) ^ { - 1 } )$ are comparable when B is treated as an absolute constant. Thus, the finite-sample Euclidean norm lower error bound of any admissible estimator can be characterized by the same Fisher-information geometry as in (3.28).

(ii) From the experimental design perspective, Theorem 3.1 shows that the statistical quality of questionnaire design is underpinned by $\mathbf { W } = \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathfrak { A } ^ { \top } \mathbf { L } \mathfrak { A } \mathbf { V } _ { A ^ { \perp } }$ in (2.24). Unlike a graph Laplacian by itself, W can be positive definite even when the graph is disconnected, as proved in Proposition 2.3. Hence, for fixed A and A, an A-optimal design allocates the limited budget to comparisons that minimize tr(W<sup>−1</sup>), thereby targeting the design criterion in (3.34).

(iii) Inequality (3.34a) improves the best known bound in pairwise ranking literature under the BTL model $/ \psi ,$ Theorem 3] proved by a Bayesian approach [39, Lemma 10]. The proof strategy of Theorem 3.1 can be applied to generalized linear models with canonical link function and yields similar results by assuming that the cumulant function has a bounded second-order derivative. Inequality (3.34b) shows that the lower bound is not driven by rare events. In Section 4, we further show that the canonical MLE attains this minimax benchmark up to logarithmic factors with high probability once N exceeds an explicit designdependent threshold involving B and $\mu _ { Q }$

We close the discussion of Theorem 3.1 by studying the minimax lower bound for the excess risk. Let $\pmb { v } \in \mathbb { R } ^ { d _ { \theta } }$ be any but fixed. Consider the population-level negative log-likelihood function $\mathcal { L } _ { \pmb { \theta } } ( \pmb { v } ) : = \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } [ \ell ( \pmb { v } ) ]$ , and the excess risk $\mathcal { E } _ { \mathcal { L } } ( v , \pmb { \theta } ) : = \mathcal { L } _ { \pmb { \theta } } ( v ) - \mathcal { L } _ { \pmb { \theta } } ( \pmb { \theta } )$ . We can mimic the proof of Theorem 3.1 to derive a lower bound $\mathrm { f o r ^ { 5 } }$

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \mathcal { E } _ { \mathcal { L } } ( \cdot , \cdot ) ) : = \operatorname* { i n f } _ { \hat { \theta } \in \mathcal { A } _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \mathcal { L } _ { \theta } ( \hat { \theta } ) - \mathcal { L } _ { \theta } ( \theta ) \right] .\tag{3.36}
$$

$\mathcal { E } _ { \mathcal { L } } ( \hat { \pmb { \theta } } , \pmb { \theta } ^ { \star } )$ is widely used to measure the performance of an estimator $\hat { \pmb { \theta } }$ of $\pmb { \theta } ^ { \star }$ in the literature of logistic regression, see e.g., [13, Section 2] and [52, Theorem 1.1]. The result presented below can be viewed as an analogue of [61, Theorem 1] but stated and proved in a diferent form. Its proof is deferred to Section 7.1.2.

Corollary 3.2 Assume the settings and conditions of Theorem 3.1, let $\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \mathcal { E } _ { \mathcal { L } } ( \cdot , \cdot ) )$ be defined as in (3.36). Then

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \mathcal { E } _ { \mathcal { L } } ( \cdot , \cdot ) ) \gtrsim \frac { d _ { \theta } } { B } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} ,\tag{3.37a}
$$

$$
\operatorname* { i n f } _ { \theta \in \mathcal { A } _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { P } _ { \theta } ^ { ( N ) } \left\{ \varepsilon _ { \scriptscriptstyle \mathcal { L } } ( \hat { \theta } , \theta ) \geq \frac { d _ { \theta } } { 1 2 8 B } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} \right\} \geq \frac { 1 } { 1 5 } .\tag{3.37b}
$$

The second result in this section concerns a minimax lower bound in the form of

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \parallel \cdot \parallel _ { Q , \infty } ^ { 2 } ) = \operatorname* { i n f } _ { \hat { \theta } \in A _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \left| \mathbf { q } _ { \alpha } ^ { \top } ( \hat { \theta } - \pmb { \theta } ) \right| ^ { 2 } \right] ,\tag{3.38}
$$

where $\| \cdot \| _ { Q , \infty } : = \operatorname* { m a x } _ { \alpha \in { \mathcal { E } } _ { s } } | \mathbf { q } _ { \alpha } ^ { \top } ( \cdot )$ |measures the largest error along queried directions, and it serves as an analogue of the $\ell _ { \infty }$ norm bound used in ranking literature [18]. This geometry is also central to the MLE analysis in Section 4, where we will show that controlling $\| \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \| _ { Q , \infty }$ bounds the drift of $\nabla ^ { 2 } \ell ( { \hat { \pmb { \theta } } } )$ when compared with $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } )$ . We assume $\mathbf { W } \succ \mathbf { 0 } .$ , under which $\| \cdot \| _ { Q , \infty }$ is a norm. This slightly restrictive assumption is well-justified by Corollary 2.1.

Theorem 3.2 Let $\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { Q , \infty } ^ { 2 } )$ be defined as in (3.38) and $\textbf { W } \succ \textbf { 0 }$ . Under Assumptions 2.1–2.4,

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { Q , \infty } ^ { 2 } ) \gtrsim \operatorname* { m i n } \left\{ \frac { \mu _ { Q } d _ { \theta } \sigma ^ { 2 } } { N } , \sigma ^ { 2 } \log ^ { 2 } ( B / B _ { 0 } ) \right\} ,\tag{3.39a}
$$

$$
\operatorname* { i n f } _ { \hat { \theta } \in A _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { P } _ { \theta } ^ { ( N ) } \left\{ \| \hat { \theta } - \theta \| _ { Q , \infty } ^ { 2 } \geq \operatorname* { m i n } \left\{ \frac { \mu _ { Q } d _ { \theta } \sigma ^ { 2 } } { N } , \sigma ^ { 2 } \log ^ { 2 } ( B / B _ { 0 } ) \right\} \right\} \geq \frac { 1 } { 4 } .\tag{3.39b}
$$

Its proof is deferred to Section 7.1.3.

Remark 3.2 Theorem 3.2 complements Theorem 3.1 and identifies the coherence level $\mu _ { Q }$ as the design quantity afecting the query-wise minimax rate. Some subsequent conclusions can be drawn.

(i) When $\begin{array} { r } { N \geq \frac { \mu _ { Q } d _ { \theta } } { \log ^ { 2 } \left( B / B _ { 0 } \right) } } \end{array}$ , the lower bound in (3.39a) has scale $\frac { \mu _ { Q } d _ { \theta } \sigma ^ { 2 } } { N }$ . By (2.21)–(2.23), this is comparable, for fixed B, to the largest query-wise CRLB deduced in (3.27). Together with Theorem 3.1, this observation implies that the trace- and coherence-type Fisher information geometry characterizes the best possible performance of any estimator in the non-asymptotic regime, when measured by the Euclidean and $\| \cdot \| _ { Q , \infty }$ norms, respectively.

(ii) Under the pairwise ranking model, the coherence level reduces, up to normalization, to the maximum efective resistance over queried edges $[ 1 6 ]$ . One may check $\mu _ { Q } = \mathcal { O } ( 1 )$ with high probability under the widely used Erd¨os-R´enyi comparison graph design, see, $e . g . , \left. \left. 1 8 \right. \right. \left. 6 9 , \right.$ Section 5.3.3]. Moreover, $\| \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \| _ { Q , \infty }$ can be upper bounded by $\| \hat { \mathbf { w } } - \mathbf { w } ^ { \star } \| _ { \infty }$ up to an absolute constant [19, Section 2.3]. Thus, together with the Erd¨os-R´enyi graph design, Theorem 3.2 recovers a lower bound for the usual coordinatewise $\ell _ { \infty }$ norm error bound up to logarithmic factors, see, e.g., [18, Theorem 7] and [19, Theorem 1]. For general query directions, Madan et al. $[ \langle \mathcal { I } \mathcal { I } ]$ show that the discrete D-optimal construction yields an explicit bound on $\mu _ { Q } ,$ , see Section 5.

## 4 The MLE Error Rate

With theoretical preparations for lower error bounds of general estimators which are applicable to MLE, we are ready to discuss the MLE error rate by deriving upper error bounds and conditions under which the upper and lower bounds meet up to log d and B factors. A precondition in Section 3 is that the estimators are finite-valued. However, MLE may not exist (see e.g., [36; $3 ; 4 2 ] )$ . This motivates us to investigate existence of MLE in the first place before discussing its upper error bounds. In view of Corollary 2.1, we focus on the identifiable case $\textbf { W } \succ \textbf { 0 }$ throughout this section. We start by specifying the linear stochastic and deterministic bias terms

$$
\Delta _ { l i n } : = - \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \nabla \ell ( \pmb \theta ^ { \star } ) ,\tag{4.40a}
$$

$$
\Delta _ { b i a s } : = - \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left( \frac { 1 } { 2 N } \sum _ { \alpha \in \mathcal { E } _ { \star } } n _ { \alpha } \frac { \mathbf { q } _ { \alpha } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \mathbf { q } _ { \alpha } } { N \sigma ^ { 2 } } \Psi ^ { ( 3 ) } \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta ^ { \star } } { \sigma } \right) \frac { \mathbf { q } _ { \alpha } } { \sigma } \right) ,\tag{4.40b}
$$

where $\Psi ^ { ( 3 ) } ( \cdot )$ is the third-order derivative of the smooth cumulant function $\Psi ( \cdot )$ . The two quantities in (4.40) will be extensively used throughout the analysis in this section. Observe that the linear term is centered, i.e., $\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \Delta _ { l i n } \right] = \mathbf { 0 }$ , and its covariance satisfies

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \Delta _ { l i n } \Delta _ { l i n } ^ { \top } \right] = \frac { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } } { N } , \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ | \mathbf { q } _ { \alpha } ^ { \top } \Delta _ { l i n } | ^ { 2 } \right] = \frac { \mathbf { q } _ { \alpha } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { q } _ { \alpha } } { N } , \forall \pmb { \alpha } \in \mathcal { E } _ { s } .
$$

Thus, $\Delta _ { l i n }$ has exactly the Fisher information geometry predicted by classical asymptotic likelihood theory. The quantity $\Delta _ { b i a s }$ corresponds to the deterministic second-order bias of the canonical logistic regression, see, $\mathrm { e . g . }$ , [48, Chapter 4], [20, Section 4] and [54; 23; 37]. It can be alternatively identified from the second-order Taylor expansion of the likelihood score equation, as detailed in Section 7.2.1.1. For a fixed questionnaire design and ground truth, $\Delta _ { b i a s }$ is deterministic. However, it may dominate the linear stochastic term over a nontrivial finite-sample regime in the absence of additional structural restrictions on the questionnaire (Lemmas A.2 and A.3). Consequently, the resulting suficient sample size for the MLE to be minimax optimal depends quadratically on $d _ { \theta }$ . We refer interested readers to [44] and references therein for more discussion.

Theorem 4.1 Let Assumptions 2.1–2.3 hold and $\mathbf { W } \succ \mathbf { 0 }$ , let $\mu _ { Q }$ be the coherence level defined in (3.33). If the number of independent samples satisfies $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d ,$ then the following hold.

(i) There exists a positive absolute constant $c > 1$ , such that with probability at least $1 - d ^ { - c }$ the MLE $\hat { \pmb { \theta } }$ in Definition 2.3 exists and is unique.

(ii) The estimation error $\hat { \pmb { \theta } } - \pmb { \theta } ^ { \star }$ satisfies the following inequalities:

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ \| \hat { \theta } - \theta ^ { \star } \| _ { Q , \infty } \lesssim \sqrt { \frac { \sigma ^ { 2 } B \mu _ { Q } d _ { \theta } \log d } { N } } + \frac { \sigma B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } \right\} \geq 1 - d ^ { - c } ,\tag{4.41a}
$$

$$
\mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left\{ \| \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } \lesssim \sqrt { \frac { B d _ { \theta } \log d } { N } } \right\} \geq 1 - d ^ { - c } .\tag{4.41b}
$$

We do not integrate (4.41) to obtain error bounds in expectation form as in [61], since the unconstrained MLE may not exist. We assess minimax optimality by comparing our highprobability upper bounds with the corresponding probability lower bounds in (3.34b), (3.37b), and (3.39b). See, e.g., [15, Section 3] for a similar treatment.

The proof of Theorem 4.1 is deferred to Section 7.2.1, where Assumption 2.3 enters the proof via Proposition 2.2 and Definition 3.2, see Lemma A.1. The first term on the r.h.s. of (4.41a) matches the minimax rate in Theorem 3.2 up to log d and B factors. The second term in (4.41a) signifies that the suficient condition for existence alone does not immediately yield the same error rate with minimax scaling. By isolating the second-order bias and higher-order remainder, Theorem 4.2 below makes this distinction precise. In the present fixed-design setting, the $d _ { \theta } ^ { 3 / 2 }$ sample size scaling improves upon the quadratic one by Ostrovskii and Bach [52, Theorem 1.1], up to log d and B factors. Consequently, inequalities (4.41a) and (4.41b), together with Lemma 7.7, yield the excess risk bound stated in the following corollary.

Corollary 4.1 Assume the settings and conditions of Theorem 4.1. If the number of independent samples satisfies $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d ;$ then there exists a positive absolute constant $c > 1$ such that

$$
\mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left\{ \mathcal { L } _ { \pmb { \theta } ^ { \star } } ( \hat { \pmb { \theta } } ) - \mathcal { L } _ { \pmb { \theta } ^ { \star } } ( \pmb { \theta } ^ { \star } ) \lesssim \frac { B d _ { \theta } \log d } { N } \right\} \geq 1 - d ^ { - c } ,\tag{4.42}
$$

where $\mathcal { L } _ { \pmb { \theta } ^ { \star } } ( \cdot ) : = \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } [ \ell ( \cdot ) ]$ is defined as in Corollary 3.2.

We omit the proof as it follows directly from mimicking the proof of [52, Theorem 3.1]. Under the assumptions of Corollary 3.2, the minimax lower bound measured in excess risk has scale $\frac { d _ { \theta } } { B N }$ whenever $\begin{array} { r } { N \ge \frac { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } { \log ^ { 2 } ( B / B _ { 0 } ) } } \end{array}$ (see (3.37)). Comparing this with Corollary 4.1 shows the optimality of (4.42) up to logarithmic and B factors, see also [13, Section 2] for related excess risk guarantees in the logistic regression literature.

The main technical challenge in proving Theorem 4.1 is that positive definiteness of $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } )$ does not provide a uniform non-trivial lower bound on $\nabla ^ { 2 } \ell ( \pmb \theta )$ over $\mathbb { R } ^ { d _ { \theta } }$ , since $\Psi ^ { \prime \prime } ( \eta ) \downarrow 0$ as $| \eta |  \infty$ . We must localize the MLE near $\pmb { \theta } ^ { \star }$ before obtaining a curvature lower bound of the negative log-likelihood function. However, by performing a constrained MLE with prescribed compact feasible set as in [61; 80; 14; 63], we can derive an error bound of the form (4.41b) from the standard argument below.

Lemma 4.1 (Strong Convexity Localization) Let $\Theta \subseteq \mathbb { R } ^ { d _ { \theta } } , \pmb { \theta } ^ { \star } \in \Theta$ , and $\nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \ \succ \ \mathbf { 0 }$ Assume that, for some $\kappa > 0 ,$ , the negative log-likelihood function ℓ(θ) is κ-strongly convex at $\pmb { \theta } ^ { \star }$ under the norm $\| \cdot \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } , i . e .$

$$
\ell ( \pmb \theta ) - \ell ( \pmb \theta ^ { \star } ) - \langle \nabla \ell ( \pmb \theta ^ { \star } ) , \pmb \theta - \pmb \theta ^ { \star } \rangle \geq \kappa \| \pmb \theta - \pmb \theta ^ { \star } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } ^ { 2 } , \forall \pmb \theta \in \Theta .
$$

Then every $\begin{array} { r } { \pmb { \theta } \in \{ \pmb { \theta } \in \Theta \mid \ell ( \pmb { \theta } ) \leq \ell ( \pmb { \theta } ^ { \star } ) \} ~ s a t i s f i e s ~ \| \pmb { \theta } - \pmb { \theta } ^ { \star } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } \leq \frac { 1 } { \kappa } \| \nabla \ell ( \pmb { \theta } ^ { \star } ) \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } } . } \end{array}$

This localization strategy appears in [61, Lemma 9] and [18, Lemma 12], which underlies the seminorm bounds for constrained estimators. For the fixed design, $\| \nabla \ell ( { \pmb \theta } ^ { \star } ) \| _ { \nabla ^ { 2 } \ell ( { \pmb \theta } ^ { \star } ) ^ { - 1 } } ^ { 2 }$ tightly concentrates around its mean, and $\sqrt { N } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 / 2 } \nabla \ell ( \pmb { \theta } ^ { \star } )$ is isotropic, see e.g., Lemma A.4. Consequently, the main results in $\left[ 1 4 ; 6 1 ; 8 0 \right]$ may be interpreted, up to logarithmic and constant factors, as high-probability upper bounds in form of $\begin{array} { r } { \| \Delta \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } ^ { 2 } \lesssim \frac { B d _ { \theta } } { N } } \end{array}$ , thus $\begin{array} { r } { \| \Delta \| _ { 2 } ^ { 2 } \lesssim \frac { B d _ { \theta } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } \end{array}$ where $\Delta = \hat { \pmb { \theta } } - { \pmb { \theta } } ^ { \star }$ . This bound can be substantially larger than the minimax lower bound in Theorem 3.1. We therefore seek conditions under which the MLE is minimax optimal under both $\| \cdot \| _ { Q , \infty }$ and $\| \cdot \| _ { 2 }$

Theorem 4.2 Assume the settings and conditions ofTheorem 4.1. Ifthe number ofindependent samples satisfies $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d _ { \cdot }$ , then there exists a positive absolute constant $c > 1$ , such that

$$
\begin{array} { r } { \begin{array} { r } { \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \Bigg \{ \| \hat { \theta } - \theta ^ { \star } - \Delta _ { l i n } - \Delta _ { b i a s } \| _ { Q , \infty } \lesssim \sigma \frac { B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } \left[ \frac { 1 } { d _ { \theta } ^ { 1 / 4 } } + \frac { 1 } { B ^ { 3 / 2 } \log ^ { 2 } d } \right] } \\ { + \sqrt { \frac { \sigma ^ { 2 } B \mu _ { Q } d _ { \theta } \log d } { N } } \frac { 1 } { d _ { \theta } ^ { 1 / 4 } \log d } \Bigg \} \geq 1 - d ^ { - c } . } \end{array} } \end{array}\tag{4.43}
$$

If, in addition, $N \gtrsim$ max $\left\{ B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d , B ^ { 4 } \mu _ { Q } d _ { \theta } \log d \right\}$ , then

$$
\begin{array} { r } { \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \Bigg \{ \| \hat { \theta } - \theta ^ { \star } - \Delta _ { l i n } - \Delta _ { b i a s } \| _ { 2 } \lesssim \frac { \sqrt { \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| B \mu _ { Q } } d _ { \theta } } { N } \left[ \frac { 1 } { d _ { \theta } ^ { 1 / 4 } } + \frac { 1 } { B ^ { 3 / 2 } \log ^ { 2 } d } \right] } \\ { + \sqrt { \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) \log d _ { \theta } } { N } } \Bigg \} \geq 1 - d ^ { - c } . } \end{array}\tag{4.44}
$$

The proof is deferred to Section 7.2.2. The role of Theorem 4.2 is parallel to that of [44, Theorem 1], which refines the coarse error bound of Theorem 4.1 by a second-order expansion and makes the transition toward the minimax regime explicit. By Lemmas $\mathrm { A . 2 }$ and A.4

$$
\| \Delta _ { b i a s } \| _ { Q , \infty } \lesssim \sigma \frac { B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } , \quad \| \Delta _ { l i n } \| _ { Q , \infty } \lesssim \sqrt { \frac { \sigma ^ { 2 } B \mu _ { Q } d _ { \theta } \log d } { N } } ,
$$

where the second inequality holds with high probability under the assumptions of Theorem 4.1. By comparing these bounds with (4.43), we can argue that the canonical MLE attains the $\| \cdot \| _ { Q , \infty }$ minimax rate up to log d and B factors whenever $N \gtrsim$ max $\left\{ { \frac { B \mu _ { Q } d _ { \theta } ^ { 2 } } { \log d } } , B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d \right\} ^ { 6 }$ , and the resulting $d _ { \theta } ^ { 2 }$ dependency is consistent with the quadratic barrier for smooth plug-in estimators based on the MLE [44].

We pause to relate Corollary 4.1 and Theorem 4.2 to Theorem 3.2 and Proposition 3.1 of [54], respectively. For a canonically parameterized exponential family with dimension $d _ { \theta }$ and sample size N, Portnoy shows [54, Section 3], under suitable moment conditions, that $\frac { d _ { \theta } ^ { 3 / 2 } } { N }  0$ is suficient to guarantee normal approximation of the centered likelihood ratio statistic, but it is in general not enough for obtaining the asymptotic normality of a fixed linear function of the uncorrected MLE, and hence for the query-wise linear predictor in the present setting. Inequality (4.43) may be viewed as a design-dependent, non-asymptotic analogue of [54, Proposition 3.1]. Both results isolate the leading linear stochastic term and the second-order bias, while controlling the higher-order remainders. On the other hand, (4.42) bounds the excess risk, whereas [54, Theorem 3.2] gives a distributional likelihood-ratio approximation. Both results exhibit a $d _ { \theta } ^ { \bar { 3 } / 2 } / N$ dimensional scaling<sup>7</sup>. The same $d _ { \theta } ^ { 3 / 2 }$ scaling has also appeared in recent analysis of broader classes of $Z -$ estimators under stronger regularity conditions [63; 44], see Section 7.2.1 for further discussion.

![](images/279cb6faf0ea34af1b76419b00f3f841e8794804c6caf3b105ceaf67f267cb72.jpg)  
Figure 3: Flowchart of the numerical experiments.

Similar to (4.43), inequality (4.44) admits a related but more geometry-sensitive structure stated in the following corollary. The proof is deferred to Appendix A.10.

Corollary 4.2 Assume the settings and conditions of Theorem 4.2. Suppose further

$$
N \gtrsim \operatorname* { m a x } \left\{ B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d , \ B ^ { 4 } \mu _ { Q } d _ { \theta } \log d , \frac { B \mu _ { Q } d _ { \theta } \log d _ { \theta } } { r _ { \mathrm { e f f } } } , \frac { B \mu _ { Q } d _ { \theta } ^ { 2 } } { r _ { \mathrm { e f f } } \log d _ { \theta } } \right\} ,\tag{4.45}
$$

where $\begin{array} { r } { r _ { \mathrm { e f f } } : = \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } ) } { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } } \end{array}$ denotes the efective rank of $\nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 }$ . Then there exists a positive absolute constant $c > 1$ , such that

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ \| \hat { \theta } - \theta ^ { \star } \| _ { 2 } ^ { 2 } \lesssim \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) \log d _ { \theta } } { N } \right\} \geq 1 - 2 d ^ { - c } - d _ { \theta } ^ { - c } .\tag{4.46}
$$

Comparing (4.46) with (3.34), we can see that the upper bound coincides with the lower bound up to log $d _ { \theta }$ and B factors. Beyond the baseline sample-size requirements inherited from Theorems 4.1 and 4.2, the dependence of the sample-size requirement on $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } )$ is entirely summarized by the efective rank of $\nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 }$ . Since $1 \leq r _ { \mathrm { e f f } } \leq d _ { \theta }$ , the resulting dimensional scaling in (4.45) ranges from $d _ { \theta } ^ { 3 / 2 }$ to $d _ { \theta } ^ { 2 }$ if $B$ and $\mu _ { Q }$ are independent of $d _ { \theta }$ . For example, when $r _ { \mathrm { e f f } } = \mathcal { O } ( 1 )$ , the quadratic dependence on $d _ { \theta }$ in (4.45) is then recovered. Neither the smallest eigenvalue nor the condition number of $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } )$ appears in the sample complexity for the MLE to attain the Euclidean minimax rate in Theorem 3.1. Consequently, near-singularity of the FIM can enlarge the absolute estimation error at the r.h.s. of (4.46) without causing a corresponding explosion of sample size at the r.h.s. of (4.45).

## 5 Numerical Experiments

This section presents some numerical studies that evaluate the finite-sample performance of the MLE and the Firth correction. We examine how the gap between empirical error of the MLE and the CRLB ((3.27) and (3.28) in Corollary 3.1) changes with the sample size. To this end, we first use synthetic query pools to control the design spectrum and parameter orientation, and then consider an experiment based on flight survey data in [59]. Figure 3 gives a flowchart of the key steps in this section.

## 5.1 Implementation Details

We consider a linear utility model without equality constraints throughout this section, i.e., $\mathbf { V } _ { A ^ { \perp } } = \mathbf { I } _ { p } , \mathbf { w } _ { o } = \mathbf { 0 } , \sigma = 1$ . Let $\mathbb { I } = \{ ( i , j ) ~ | ~ 1 \le i < j \le d \}$ . In each experiment, we specify a fixed pool of candidate queries $\{ \mathbf { q } _ { \alpha } \} _ { \alpha \in \mathbb { I } }$ that spans $\mathbb { R } ^ { d _ { \theta } }$ and a ground-truth parameter $\pmb { \theta } ^ { \star } \in \mathbb { R } ^ { d _ { \theta } }$ . Their constructions are given in Sections 5.2 and 5.3. Since $\mathbf { V } _ { A ^ { \perp } } = \mathbf { I } _ { p } , \mathbf { q } _ { \alpha } = \mathbf { a } _ { \alpha }$ and $\pmb { \theta } ^ { \star } = \mathbf { w } ^ { \star }$ . The MLE and Firth correction are solved by standard trust-region method. To obtain an identifiable design with controlled coherence level, we identify the edge weights $\{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } }$ by solving the discrete D-optimal problem [47], i.e.,

$$
\begin{array} { l } { \displaystyle \operatorname* { m a x } _ { n _ { \alpha } } \log \operatorname* { d e t } \left( \sum _ { \alpha \in \mathbb { I } } n _ { \alpha } \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } \right) } \\ { \displaystyle \mathrm { s . t . } \sum _ { \alpha \in \mathbb { I } } n _ { \alpha } \leq N _ { b a s e } , n _ { \alpha } \geq 0 , n _ { \alpha } \in \mathbb { Z } , \forall \pmb { \alpha } \in \mathbb { I } , } \end{array}\tag{5.47}
$$

where $N _ { b a s e } \asymp d _ { \theta }$ is the design budget to be specified later. Using the solution, we define $\mathcal { E } _ { s } : = \{ \pmb { \alpha } \in \mathbb { I } \mid n _ { \pmb { \alpha } } > 0 \}$ , and set

$$
\mathbf { W } : = \frac { 1 } { N _ { b a s e } } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } .\tag{5.48}
$$

We repeat the base blocks $\{ n _ { \alpha } \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } \} _ { \alpha \in \mathcal { E } _ { s } }$ by a replication factor $r \in \mathbb { Z } _ { + }$ to increase the sample size from $N = N _ { b a s e }$ to $N = r N _ { b a s e }$ while keeping all other design-dependent coeficients unchanged. Alternatively, we may adopt D-optimal design without repetition<sup>8</sup>, i.e., replace $n _ { \alpha } \in \mathbb { Z }$ by $n _ { \alpha } \in \{ 0 , 1 \}$ in (5.47), but this would involve unnecessary subtlety in the design of numerical experiments for validating the efectiveness of both the MLE and Firth correction.

By the Kiefer-Wolfowitz theorem (see e.g., [35], [64, Chapter 2.2], and [38, Chapter 21]), and the strong duality between the convex relaxation of the D-optimal design and the minimum volume enclosing ellipsoid problem [64, Chapter 2], the solution to the convex relaxation of (5.47) is automatically G-optimal. Ahipasaoglu et al. [2] use Fedorov’s exchange method (see, e.g. [34, Section 4.1.1]) to refine an initial integer allocation obtained by rounding a continuous solution of the convex relaxation of (5.47). We adopt their approach to solve (5.47) due to its adaptation to candidate sets with large cardinality. Moreover, Algorithm 3 in [2] guarantees $\mu _ { Q } \leq$ min $\left\{ \frac { N _ { b a s e } } { d _ { \theta } } , \frac { N _ { b a s e } } { N _ { b a s e } - d _ { \theta } + 1 } \right\} \mathrm { ~ < ~ 2 ~ }$ for any $N _ { b a s e } \geq d _ { \theta }$ based on intermediate results in [47, Section $2 . 2 ] ^ { \grave { 9 } }$

For given $( d _ { \theta } , N , \{ \mathbf { q } _ { \alpha } \} _ { \alpha \in \mathcal { E } _ { s } } , \{ n _ { \alpha } \} _ { \alpha \in \mathcal { E } _ { s } } )$ , we simulate the response $Y _ { \alpha } ^ { ( t ) } \in \{ 0 , 1 \}$ as a realization of a Bernoulli random variable with success rate ${ p } _ { \alpha } ( \pmb { \theta } ^ { \star } )$ for every $t \in [ n _ { \alpha } ] , \alpha \in \mathcal { E } _ { s }$ , independently. For each questionnaire design and fixed sample size, we run $L = 3 0 0 0$ independent Monte Carlo trials for the response vector $Y ^ { ( N ) }$ and calculate the MLE and Firth correction in each trial. Since MLE may not exist when the sample size is small, we use Konis’s linear programming method [36] to test the existence of MLE in each trial. When there is no MLE, we use the optimal solution obtained in the box $[ - 1 0 0 \| \pmb { \theta } ^ { \star } \| _ { \infty } , 1 0 0 \| \pmb { \theta } ^ { \star } \| _ { \infty } ] ^ { d _ { \theta } }$ . For trial l, define

$$
\Delta _ { l } : = \hat { \pmb \theta } ^ { l } - \pmb \theta ^ { \star } , \quad \Delta _ { \mathrm { F i r t h } } ^ { l } : = \hat { \pmb \theta } _ { \mathrm { F i r t h } } ^ { l } - \pmb \theta ^ { \star } ,
$$

where $\hat { \theta } ^ { l } \ ( \hat { \theta } _ { \mathrm { F i r t h } } ^ { l } )$ denotes the MLE (Firth correction) for the l-th trial. Let $\mathfrak { L } \subseteq [ L ]$ index the trials included in a summary. We report the root mean square errors (RMSE)

$$
\| \Delta \| _ { 2 } \mathrm { \ R M S E : } \sqrt { \frac { 1 } { | \mathfrak { L } | } \sum _ { l \in \mathfrak { L } } \| \Delta _ { l } \| _ { 2 } ^ { 2 } } , \quad \| \Delta \| _ { Q , \infty } \mathrm { \ R M S E : } \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \sqrt { \frac { 1 } { | \mathfrak { L } | } \sum _ { l \in \mathfrak { L } } | \mathbf { q } _ { \alpha } ^ { \top } \Delta _ { l } | ^ { 2 } } .
$$

The same definitions apply to $\Delta _ { \mathrm { F i r t h } }$ and the random residuals identified in Section 4. For RMSEs conditioned on MLE existence, L contains only trials for which Konis’s method reports no separation, so that the MLE exists. We compare the estimators’ RMSEs with the corresponding CRLBs in Corollary 3.1.

## 5.2 Synthetic Questionnaire

We construct synthetic query pools to examine how $\Delta _ { l i n } , \ \Delta _ { b i a s }$ , and the higher-order terms depend on the design spectrum (see (5.49)) and parameter orientation (see (5.50)) in the nonasymptotic regime.

Let $\begin{array} { r } { \pmb { g _ { i } } \overset { i . i . d . } { \sim } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { d _ { \theta } } ) } \end{array}$ , for $i \in [ d ]$ . For a user-specified parameter $m \ \leq \ { \binom { d } { 2 } }$ , we sample uniformly without replacement from the set $\mathbb { I } = \{ ( i , j ) \mid 1 \leq i < j \leq d \}$ to construct a subset $\tilde { \mathbb { I } } \subseteq \mathbb { I }$ with cardinality $m _ { : }$ , and solve (5.47) with I replaced by <sup>˜</sup>I. With a Rademacher sequence $\{ \xi _ { \alpha } \} _ { \alpha \in \tilde { \mathbb { I } } } \in \{ - 1 , 1 \} ^ { m }$ , define

$$
\tilde { \mathbf { q } } _ { \alpha } : = \xi _ { \alpha } ( \pmb { g } _ { i } - \pmb { g } _ { j } ) , \quad \tilde { \mathbf { W } } : = \frac { 1 } { m } \sum _ { \pmb { \alpha } \in \tilde { \mathbb { I } } } \tilde { \mathbf { q } } _ { \alpha } \tilde { \mathbf { q } } _ { \alpha } ^ { \top } .
$$

Let $\mathbf { u } \sim$ Unif $( \mathbb { S } ^ { d _ { \theta } - 1 } )$ and $\mathbf { v } \ \sim \ \mathrm { U n i f } \ ( \mathbb { S } ^ { d _ { \theta } - 1 } \cap \mathbf { u } ^ { \perp } )$ such that $\mathbf { u } ^ { \top } \mathbf { v } = \mathrm { ~ 0 ~ }$ . For $\epsilon \in \mathsf { \Gamma } ( 0 , 1 ]$ , set $\mathbf { R } : = \mathbf { I } _ { d _ { \theta } } + ( \sqrt { \epsilon } - 1 ) \mathbf { u } \mathbf { u } ^ { \top }$ and reshape the spectrum of $\tilde { \mathbf { W } }$ as

$$
\mathbf { q } _ { \alpha } : = \mathbf { R } \tilde { \mathbf { W } } ^ { - 1 / 2 } \tilde { \mathbf { q } } _ { \alpha } , \hat { \mathbf { W } } : = \frac { 1 } { m } \sum _ { \alpha \in \tilde { \mathbb { I } } } \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } = \mathbf { R } \mathbf { R } ^ { \top } = \mathbf { I } _ { d _ { \theta } } + ( \epsilon - 1 ) \mathbf { u } \mathbf { u } ^ { \top } .\tag{5.49}
$$

Thus, $\hat { \mathbf { W } }$ has eigenvalue ϵ along u and eigenvalue 1 on $\{ { \bf u } \} ^ { \perp }$ . Its inverse has efective rank

$$
r _ { e , t a r } : = \frac { \mathrm { t r } ( \hat { \mathbf { W } } ^ { - 1 } ) } { \Vert \hat { \mathbf { W } } ^ { - 1 } \Vert } = 1 + ( d _ { \theta } - 1 ) \epsilon .
$$

Note that $\hat { \mathbf { W } }$ uniformly averages all queries in the candidate pool, whereas W in (5.48) uses the selected D-optimal design, and ${ \mathcal { T } } ( \theta ^ { \star } )$ additionally incorporates the ground-truth variance, see (2.21)–(2.24). The efective ranks of their inverses therefore may not coincide. We specify the ground truth by

$$
{ \pmb \theta } ^ { \star } = { \bf R } ^ { - 1 } \left( \rho { \bf u } + \sqrt { 1 - \rho ^ { 2 } } { \bf v } \right) , \rho \in [ 0 , 1 ] .\tag{5.50}
$$

For fixed $\{ \widetilde { \mathbf { q } } _ { \alpha } \} _ { \alpha \in \widetilde { \mathbb { I } } }$ , the pair $( \mathbf { u } , \mathbf { v } )$ and $\rho ,$

$$
\eta _ { \alpha } ( \pmb \theta ^ { \star } ) = \frac { \mathbf { w _ { o } } ^ { \top } \mathbf { a } _ { \alpha } + \left( \mathbf { V } _ { A ^ { \bot } } \pmb \theta ^ { \star } \right) ^ { \top } \mathbf { a } _ { \alpha } } { \sigma } = \mathbf { q } _ { \alpha } ^ { \top } \pmb \theta ^ { \star } = \tilde { \mathbf { q } } _ { \alpha } ^ { \top } \tilde { \mathbf { W } } ^ { - 1 / 2 } \left( \rho \mathbf { u } + \sqrt { 1 - \rho ^ { 2 } } \mathbf { v } \right)
$$

and it satisfies $\begin{array} { r } { \frac { 1 } { m } \sum _ { \alpha \in \tilde { \mathbb { I } } } \eta _ { \alpha } ( \pmb { \theta } ^ { \star } ) ^ { 2 } = 1 } \end{array}$ . Since $\eta _ { \alpha } ( \pmb { \theta } ^ { \star } )$ does not depend on $\epsilon , \epsilon$ does not afect the choice probabilities in (2.6). From direct calculation, $\begin{array} { r } { \frac { | \mathbf { u } ^ { \top } \pmb { \theta } ^ { \star } | } { \| \pmb { \theta } ^ { \star } \| _ { 2 } } = \frac { \rho } { \sqrt { \rho ^ { 2 } + \epsilon ( 1 - \rho ^ { 2 } ) } } } \end{array}$ , thus $\rho$ and ϵ jointly

Table 1: Exponent $\alpha _ { \mathrm { c r o s s } } : = \log _ { d _ { \theta } } N _ { \mathrm { c r o s s } }$ across the parameter grid. Each entry reports the Monte Carlo mean ± standard deviation of $\alpha _ { \mathrm { { c r o s s } } }$  
(a) d<sub>θ</sub> = 30, N<sub>base</sub> = 45
<table><tr><td rowspan="2"></td><td colspan="5"> $r _ { e , t a r }$ </td></tr><tr><td>2</td><td>3.94</td><td>7.75</td><td>15.2</td><td>30</td></tr><tr><td>0.5</td><td></td><td></td><td></td><td> $\mid 1 . 1 3 \pm 0 . 0 3 7 \ 0 . 9 9 \pm 0 . 0 3 1 \ 0 . 8 8 \pm 0 . 0 2 8 \ 0 . 8 0 \pm 0 . 0 2 8 \ 0 . 7 4 \pm 0 . 0 2 9$ </td><td></td></tr><tr><td>0.6</td><td>1.22 ± 0.032 1.07 ± 0.028 0.93 ± 0.027 0.82 ± 0.027</td><td></td><td></td><td></td><td> $0 . 7 4 \pm 0 . 0 2 8$ </td></tr><tr><td>0.7</td><td></td><td></td><td></td><td> $1 . 3 0 \pm 0 . 0 3 0 \ 1 . 1 3 \pm 0 . 0 2 7 \ 0 . 9 8 \pm 0 . 0 2 6 \ 0 . 8 5 \pm 0 . 0 2 7 \ 0 . 7 4 \pm 0 . 0 2 8$ </td><td></td></tr><tr><td>0.8</td><td></td><td> $1 . 3 6 \pm 0 . 0 2 9 1 . 1 9 \pm 0 . 0 2 7 1 . 0 3 \pm 0 . 0 2 7 0 . 8 7 \pm 0 . 0 2 7$ </td><td></td><td></td><td>0.74 ± 0.028</td></tr><tr><td>0.9</td><td></td><td> $\left| 1 . 4 2 \pm 0 . 0 2 9 \ 1 . 2 5 \pm 0 . 0 2 6 \ 1 . 0 7 \pm 0 . 0 2 7 \ 0 . 9 0 \pm 0 . 0 2 8 \ 0 . 7 4 \pm 0 . 0 3 0 \ \right.$ </td><td></td><td></td><td></td></tr></table>

(c) d<sub>θ</sub> = 70, N<sub>base</sub> = 105
<table><tr><td colspan="6"> $r _ { e , t a r }$ </td></tr><tr><td>ρ</td><td>2</td><td>4.86</td><td>11.8</td><td>28.8</td><td>70</td></tr><tr><td>0.5</td><td></td><td> $\left| 1 . 2 1 \pm 0 . 0 1 9 \ 1 . 0 4 \pm 0 . 0 1 4 \ 0 . 8 9 \pm 0 . 0 1 3 \ 0 . 7 8 \pm 0 . 0 1 1 \ 0 . 7 1 \pm 0 . 0 1 1 \right.$ </td><td></td><td></td><td></td></tr><tr><td>0.6</td><td></td><td> $\left| 1 . 2 9 \pm 0 . 0 1 8 \ 1 . 1 1 \pm 0 . 0 1 3 \ 0 . 9 4 \pm 0 . 0 1 2 \ 0 . 8 0 \pm 0 . 0 1 2 \ 0 . 7 1 \pm 0 . 0 1 1 \right.$ </td><td></td><td></td><td></td></tr><tr><td>0.7</td><td> $1 . 3 5 \pm 0 . 0 1 7$ </td><td></td><td></td><td> $1 . 1 7 \pm 0 . 0 1 3 ~ 0 . 9 9 \pm 0 . 0 1 2 ~ 0 . 8 3 \pm 0 . 0 1 2 ~ 0 . 7 1 \pm 0 . 0 1 2$ </td><td></td></tr><tr><td>0.8</td><td></td><td> $1 . 4 1 \pm 0 . 0 1 7 \ 1 . 2 2 \pm 0 . 0 1 2 \ 1 . 0 3 \pm 0 . 0 1 2 \ 0 . 8 6 \pm 0 . 0 1 2 \ 0 . 7 1 \pm 0 . 0 1 2$ </td><td></td><td></td><td></td></tr><tr><td>0.9</td><td> $1 . 4 6 \pm 0 . 0 1 7$ </td><td> $1 . 2 7 \pm 0 . 0 1 2 1 . 0 8 \pm 0 . 0 1 2 0 . 8 9 \pm 0 . 0 1 3 0 . 7 1 \pm 0 . 0 1 3$ </td><td></td><td></td><td></td></tr></table>

(b) $d _ { \theta } = 5 0 , N _ { \mathrm { b a s e } } = 7 5$
<table><tr><td rowspan="2">ρ</td><td colspan="5"> $r _ { e , t a r }$ </td></tr><tr><td>2</td><td>4.47</td><td>10</td><td>22.4</td><td>50</td></tr><tr><td>0.5</td><td></td><td></td><td></td><td> $1 . 1 8 \pm 0 . 0 2 6 ~ 1 . 0 2 \pm 0 . 0 2 0 ~ 0 . 8 8 \pm 0 . 0 1 7 ~ 0 . 7 8 \pm 0 . 0 1 6 ~ 0 . 7 2 \pm 0 . 0 1 6$ </td><td></td></tr><tr><td>0.6</td><td> $1 . 2 6 \pm 0 . 0 2 5$ </td><td></td><td></td><td>1.09 ± 0.019 0.93 ± 0.017 0.80 ± 0.016 0.72 ± 0.016</td><td></td></tr><tr><td>0.7</td><td></td><td></td><td></td><td> $1 . 3 3 \pm 0 . 0 2 5 ~ 1 . 1 5 \pm 0 . 0 1 9 ~ 0 . 9 8 \pm 0 . 0 1 7 ~ 0 . 8 3 \pm 0 . 0 1 6 ~ 0 . 7 2 \pm 0 . 0 1 6$ </td><td></td></tr><tr><td>0.8</td><td> $1 . 3 9 \pm 0 . 0 2 4$ </td><td></td><td></td><td> $1 . 2 1 \pm 0 . 0 1 9 1 . 0 3 \pm 0 . 0 1 7 0 . 8 6 \pm 0 . 0 1 7 0 . 7 2 \pm 0 . 0 1 7$ </td><td></td></tr><tr><td>0.9</td><td> $\left| 1 . 4 4 \pm 0 . 0 2 4 \ 1 . 2 6 \pm 0 . 0 1 9 \ 1 . 0 7 \pm 0 . 0 1 7 \ 0 . 8 9 \pm 0 . 0 1 7 \ 0 . 7 2 \pm 0 . 0 1 8 \right.$ </td><td></td><td></td><td></td><td></td></tr></table>

(d) d<sub>θ</sub> = 90, N<sub>base</sub> = 135
<table><tr><td rowspan="2">p</td><td colspan="5"> $r _ { e , t a r }$ </td></tr><tr><td>2</td><td>5.18</td><td>13.4</td><td>34.7</td><td>90</td></tr><tr><td>0.5</td><td></td><td></td><td></td><td> $\mid 1 . 2 4 \pm 0 . 0 1 7 \ 1 . 0 5 \pm 0 . 0 1 3 \ 0 . 8 9 \pm 0 . 0 1 2 \ 0 . 7 8 \pm 0 . 0 1 0 \ 0 . 7 1 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>0.6</td><td> $1 . 3 1 \pm 0 . 0 1 6 \ 1 . 1 2 \pm 0 . 0 1 2 \ 0 . 9 5 \pm 0 . 0 1 1 \ 0 . 8 0 \pm 0 . 0 1 0 \ 0 . 7 1 \pm 0 . 0 1 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td>0.7</td><td> $1 . 3 7 \pm 0 . 0 1 5 ~ 1 . 1 8 \pm 0 . 0 1 1 ~ 0 . 9 9 \pm 0 . 0 1 0 ~ 0 . 8 3 \pm 0 . 0 1 0 ~ 0 . 7 1 \pm 0 . 0 1 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td>0.8</td><td> $1 . 4 3 \pm 0 . 0 1 5 ~ 1 . 2 3 \pm 0 . 0 1 0 ~ 1 . 0 4 \pm 0 . 0 1 0 ~ 0 . 8 6 \pm 0 . 0 1 0 ~ 0 . 7 1 \pm 0 . 0 1 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td>0.9</td><td> $1 . 4 7 \pm 0 . 0 1 5 1 . 2 8 \pm 0 . 0 1 0 1 . 0 8 \pm 0 . 0 0 9 0 . 8 9 \pm 0 . 0 0 9 0 . 7 1 \pm 0 . 0 0 9$ </td><td></td><td></td><td></td><td></td></tr></table>

control the alignment between $\pmb { \theta } ^ { \star }$ and u. Moreover, the solution of (5.47) is also invariant to ϵ because

$$
\log \operatorname* { d e t } \left( \sum _ { \alpha \in \tilde { \mathbb { T } } } n _ { \alpha } \mathbf { q } _ { \alpha } \mathbf { q } _ { \alpha } ^ { \top } \right) = \log \epsilon + \log \operatorname* { d e t } \left( \sum _ { \alpha \in \tilde { \mathbb { T } } } n _ { \alpha } \tilde { \mathbf { W } } ^ { - 1 / 2 } \tilde { \mathbf { q } } _ { \alpha } \tilde { \mathbf { q } } _ { \alpha } ^ { \top } \tilde { \mathbf { W } } ^ { - 1 / 2 } \right) .\tag{5.51}
$$

We draw 100 independent realizations of the candidate query pool and the ground-truth parameter as described above with $r _ { e , t a r } = d _ { \theta } { } ^ { 1 0 } , \rho = 0 . 5 , m = 2 0 0 0 , d = 4 d _ { \theta } , d _ { \theta } \in \{ 3 0 , 5 0 , 7 0 , 9 0 \}$ and solve (5.47) with $N _ { b a s e } = 1 . 5 d _ { \theta }$ . According to (5.49), (5.50) and (5.51), the solution of (5.47) is invariant to $( r _ { e , t a r } , \rho )$ . Moreover, for a fixed design $\{ n _ { \alpha } \} _ { \alpha \in { \mathcal E } _ { s } }$ , the CRLB (see (3.28)) and $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ (see (4.40b)) are functions of $( r _ { e , t a r } , \rho )$ . Thus, for diferent $( r _ { e , t a r } , \rho )$ values, we can calculate the CRLB and $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ based on the case when $( r _ { e , t a r } , \rho ) \ = \ ( d _ { \theta } , 0 . 5 )$ . Note that when a base query design is replicated, ${ \mathcal { T } } ( \theta ^ { \star } )$ remains unchanged $( n _ { \alpha }$ and N has the same scale in (2.21)). Thus

$$
\sqrt { \frac { \mathrm { t r } ( \mathcal { Z } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } ) } { N } } = \sqrt { \frac { \mathrm { t r } ( \mathcal { Z } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } ) } { N _ { b a s e } } \frac { 1 } { \sqrt { r } } } , \quad \| \Delta _ { b i a s } ( N ) \| _ { 2 } = \frac { \| \Delta _ { b i a s } ( N _ { b a s e } ) \| _ { 2 } } { r } .
$$

The two quantities are equal when $\begin{array} { r } { N _ { c r o s s } = N _ { b a s e } ^ { 2 } \frac { \| \Delta _ { b i a s } ( N _ { b a s e } ) \| _ { 2 } ^ { 2 } } { \mathrm { t r } ( \mathbb { Z } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } ) } } \end{array}$ . We report the mean values and standard deviations of $\alpha _ { \mathrm { c r o s s } } : = \log _ { d _ { \theta } } N _ { \mathrm { c r o s s } }$ in Table 1 for diferent values of $( r _ { e , t a r } , \rho )$ . In general, $N _ { c r o s s }$ increases with $\rho$ and decreases with $r _ { e , t a r }$ , and can be substantially larger than $d _ { \theta }$ . When $r _ { e , t a r } = d _ { \theta }$ , the pool spectrum is isotropic and the distribution of $N _ { c r o s s }$ is invariant to $\rho .$

We next fix $\rho = 0 . 9 , r _ { e , t a r } = 2$ and examine $d _ { \theta } \in \{ 5 0 , 7 0 \}$ with $N _ { b a s e } = 1 . 5 d _ { \theta }$ . For each $d _ { \theta }$ , we select the underlying realization in the Table 1 experiment whose $N _ { c r o s s }$ is closest to the empirical median, and fix the corresponding candidate query pool and ground-truth parameter in the subsequent Monte Carlo simulation. We record the trials in which the MLE exists and the trust-region algorithm converges, and plot the corresponding empirical probabilities in Figure 4.

![](images/5a055a9885f8cdceb77f90c9980d74aab46238cbcf5785429d74df4fbdbb6911.jpg)  
(a)

![](images/c51bf74d405de1a513eba7d0686e6ac876bd33772f6eb9a5e9f9f3c30aba5496.jpg)  
(b)

Figure 4: MLE existence and trust-region convergence probabilities versus N under base-design replication and D-optimal redesign, estimated from 3000 Monte Carlo trials, using the synthetic query pool: (a) $d _ { \theta } = 5 0 ;$ (b) $d _ { \theta } = 7 0$ . An MLE is classified as finite when Konis’s method reports no separation. An MLE is classified as converged if the trust-region method returns a solution with gradient norm below $1 0 ^ { - 6 }$ . The vertical dashed lines indicate the values of d<sub>θ</sub>, $d _ { \theta } ^ { 3 / 2 }$ and $d _ { \theta } ^ { 2 } .$  
![](images/ac7808d130316972f65947ec430a37fbb3e69805850a9c99c77cb2ce1dfeaed2.jpg)  
(a)

![](images/465534883a5a3e5616caff26f00af261bb87ec23a38350723e9d02c00bdce77c.jpg)  
(b)

d<sub>3</sub> =50, redesign, RMSE (Cond. on MLE exists)  
![](images/3f5f89fa72e922bf173968cc28ba27a90080ffe85c11e75415f2cba901ac9d87.jpg)  
(c)

d<sub>3</sub> =50, redesign, RMSE (Cond. on MLE exists)  
![](images/b378eaaea70a7aa9a08c3d10036d8af91eb64284432ffe39e7ba8ef4e7285c29.jpg)  
(d)  
Figure 5: Finite-sample performance using the synthetic query pool under a replicated D-optimal design with $( d _ { \theta } , N _ { b a s e } ) = ( 5 0 , 7 5 )$ (subplots (a)–(b)) and a D-optimal design recomputed for each N (subplots (c)–(d)). These plots compare the MLE and Firth RMSEs with the CRLB, the second-order bias, and higher-order residuals. For each $N ,$ the MLE-based RMSEs are calculated conditional on the existence of a finite MLE, and are omitted when the MLE existence probability is less than 0.02. Firth RMSEs are computed over all 3000 trials. The vertical dashed lines indicate the values of d , $d _ { \theta } ^ { 3 / 2 }$ and $d _ { \theta } ^ { 2 } .$ . For compact figure labels, $\boldsymbol { \mathfrak { s } } \boldsymbol { \mathfrak { s } } \| \cdot \| _ { Q , \infty } \mathrm { C R L B } ^ { \prime \prime }$ denotes the square root of the r.h.s. of (3.27), maximized over $\pmb { \alpha } \in \mathcal { E } _ { s }$

![](images/ee5583e3767770f9d0bcbdd002cfcdc1aca2fa3eb05a66926298085f1f8c1158.jpg)  
(a)

![](images/90f0c1777d2f2991ead7f1f715c62f6d77a66ebc745effd486dbd5f5afaf1b34.jpg)  
(b)

![](images/187261ee6f6950588ed94c886591e635a6102863623f58efc2b40a412e20ccce.jpg)  
(c)

![](images/86abe7451e8bb14e3dac1e30c17856d119954b403dac356f9e4966c801567a09.jpg)  
(d)  
Figure 6: Same as Figure 5, but with $( d _ { \theta } , N _ { b a s e } ) = ( 7 0 , 1 0 5 )$

To determine whether the observed behavior is an artifact of replicating a base design, we compare the replicated design with a discrete D-optimal design recomputed for each N. Under both schemes, we evaluate the MLE and Firth correction using the RMSE criterion defined in Section 5.1, with results for $d _ { \theta } = 5 0$ and 70 plotted in Figures 5 and 6, respectively.

Figures 4–6 reveal that the sample size required for finiteness of the MLE is smaller than the sample size required for statistical eficiency (i.e., its RMSE approaches the CRLB). Figure 4 shows that the MLE does not exist in most Monte Carlo trials when $d _ { \theta } \leq N \leq 2 d _ { \theta } ( d _ { \theta } =$ 50 in Figure 4 (a), and $d _ { \theta } ~ = ~ 7 0$ in Figure 4 (b)). On the other hand, the probability of the MLE being finite approaches one before N reaches $d _ { \theta } ^ { 3 / 2 }$ . Thus, the suficient $d _ { \theta } ^ { 3 / 2 }$ -scale condition in Theorem 4.1 is conservative for existence in this specific instance. Next, we use Figure 5 (a) to describe the finite-sample behavior of the MLE. The RMSE curves involving ∆ are computed conditional on MLE existence and are displayed only when the MLE existence probability exceeds 0.02. The ∥ · ∥<sub>2</sub> CRLB, $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ , and $\lVert \Delta _ { \mathrm { F i r t h } } \rVert _ { 2 }$ RMSE curves are plotted over the full sample-size range. The $\| \Delta \| _ { 2 }$ RMSE curve can be described in roughly four stages according to sample size. When $N \leq 1 5 0$ , the MLE does not exist in most trials. For $1 5 0 \leq$ $N \leq 4 5 0$ , the decrease in $\| \Delta \| _ { 2 }$ RMSE primarily reflects the decay of the nonlinear residuals, as shown by the $\lVert \Delta - \Delta _ { l i n } - \Delta _ { b i a s } \rVert _ { 2 }$ and $\lVert \Delta - \Delta _ { l i n } \rVert _ { 2 }$ RMSE curves. At $N \approx 4 5 0$ (about

![](images/f1f0837499a9515145cdbf775159a940c2d8f758a0a4491bfe88742c15bfe1ae.jpg)  
(a)

![](images/45cc09dfeaa99870658c045736024b45f7749532b17f829ba2ed3acbea384ac9.jpg)  
(b)  
Figure 7: MLE existence and trust-region convergence probabilities versus N under D-optimal redesign estimated from 3000 Monte Carlo trials, using the Green-Flight query pool: (a) DC; (b) London. An MLE is classified as finite when Konis’s method reports no separation. An MLE is classified as converged if the trust-region method returns a solution with gradient norm below $1 0 ^ { - 6 }$ . The vertical dashed lines indicate the values of $d _ { \theta } , d _ { \theta } ^ { 3 / 2 }$ and $d _ { \theta } ^ { 2 }$

$1 . 2 7 d _ { \theta } ^ { 3 / 2 } )$ , the $\lVert \Delta - \Delta _ { l i n } - \Delta _ { b i a s } \rVert _ { 2 }$ RMSE falls below $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ and tends to vanish, while the $\lVert \Delta - \Delta _ { l i n } \rVert _ { 2 }$ RMSE eventually approaches $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ as the sample size grows. This observation is consistent with the error bound in Theorem 4.2. In the case $4 5 0 \leq N \leq 9 0 0$ , the log-log slope of the $\| \Delta \| _ { 2 }$ RMSE curve closely tracks that of $\| \Delta _ { b i a s } \| _ { 2 }$ , although the bias is already smaller than the CRLB. Finally, for $N \geq 1 6 5 0$ , the $\| \Delta \| _ { 2 }$ RMSE nearly coincides with the CRLB. This final transition is consistent with Corollary 4.2, which requires both the higher-order remainder and the second-order bias to be small relative to the linear stochastic term. Similar behavior is observed under the query-wise metric and when the D-optimal design is recomputed for each $N _ { ; }$ , as shown in the remaining subplots of Figures 5 and 6.

By contrast, as seen from Figure 5 (a), the Firth correction exhibits more stable finite-sample behavior. When $1 5 0 \leq N \leq 4 5 0$ , the $\lVert \Delta _ { \mathrm { F i r t h } } \rVert _ { 2 }$ RMSE is comparable to $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ , and then it approaches the CRLB as N increases beyond 450. Analogous phenomena are observed in the remaining subplots of Figures 5 and 6. These results support the Firth correction as a more reliable default for the finite-sample, fixed-design preference elicitation problem considered here, and we conjecture that it is minimax optimal when $N \gtrsim d _ { \theta } ^ { 3 / 2 }$ , up to $B , \mu _ { Q }$ , and logarithmic factors. To the best of our knowledge, however, non-asymptotic error bounds for the Firth correction in high-dimensional settings are not currently available.

## 5.3 The Green-Flight Questionnaire

We construct a semi-synthetic experiment from a real-world survey conducted among 450 employees of the University of California, Davis [59]. Respondents compared hypothetical flights to Washington DC and London, described by price (in \$), $\mathrm { C O _ { 2 } }$ emissions (in lb), departure airport (SMF or SFO), and non-stop status. Certain features of the respondents are also recorded, from which we retain age, number of flights during the past year, preferred airport (SMF or SFO), current position in the university (staf, faculty, post-doc, graduate student, and others), and preferred flight arrangement method (institution-mediated or via web portal), as the contextual information of a respondent. The survey data are publicly available at https://doi.org/10.25338/B81S5M.

![](images/85da8bc9702c2bf08a6710d46054c0661c0770c70c190997f9327ccad4523ef2.jpg)  
(a)

![](images/f55eaf52ad687eb77764a6a7b44eaaa3eb41ea620db6ce0ac1c783b1e61c877c.jpg)  
(b)

![](images/a436124466bfbcd2cf93a5322848bee88425630c2b89d8d38c1d01fdfd4a1663.jpg)  
(c)

![](images/89751ad0e1ca547055a92940749906e9eaeecf091e42498559ea0d5f09237e4f.jpg)  
(d)  
Figure 8: Finite-sample performance for DC flights using the Green-Flight query pool, with the D-optimal design recomputed for each N. MLE-based RMSEs are calculated conditional on the existence of a finite MLE and are omitted when the MLE existence probability is below 0.02. Firth RMSEs are computed over all 3000 trials. The dashed curves and shaded regions in subplots (a)—(b) report the corresponding medians and empirical 10%–90% percentile ranges, using the same trial sets as their RMSE counterparts. For each $N ,$ the query-wise medians and percentile intervals are computed from the absolute errors along the query direction that attains the corresponding maximum RMSE. Subplots $\mathrm { ( c ) - ( d ) }$ show the error decompositions defined as in Figure 5. The vertical dashed lines indicate the values of $d _ { \theta } , d _ { \theta } ^ { 3 / 2 }$ and $d _ { \theta } ^ { 2 }$

After excluding respondents with missing contextual information or incomplete response sequences, we obtain a dataset with 367 respondents and 48 pairwise flight ticket templates for each destination. Each respondent answered 5 DC and 6 London questions, yielding $3 6 7 \times 5 =$ 1835 and $3 6 7 \times 6 = 2 2 0 2$ observed choices, respectively. For a destination $\mathfrak { D } \in \{ \mathrm { D C } , \mathrm { L o n d o n } \}$ , a pairwise flight ticket template $_ { \pmb { \alpha } }$ , and an alternative $j \in \{ A , B \}$ , define feature vectors of tickets $\mathbf { q } _ { \mathfrak { D } , \alpha _ { j } }$ and respondents $\mathbf { c } _ { n }$ as

$$
\mathbf { q } _ { \mathfrak { D } , \alpha _ { j } } = \left[ \frac { \mathrm { C O } _ { 2 \mathfrak { D } , \alpha _ { j } } } { 1 0 0 0 \mathrm { ~ l b } } , \frac { \mathrm { C o s t } _ { \mathfrak { D } , \alpha _ { j } } } { 1 0 0 \mathrm { ~ \mathfrak { G } ~ } } , \mathbb { 1 } \{ \mathrm { n o n } \mathrm { - s t o p } \} , \mathbb { 1 } \{ \mathrm { d e p a r t u r e ~ f r o m ~ S F O } \} \right] ^ { \top } \in \mathbb { R } ^ { 4 } ,
$$

$$
\mathbf { c } _ { n } = [ 1 , { \mathrm { a g e } } _ { n } , { \mathrm { f i g h t s } } _ { n } , 1 \{ { \mathrm { p r e f e r r e d ~ S F O } } \} , 1 \{ { \mathrm { F a c u l t y } } \} , 1 \{ { \mathrm { i n s t i t u t i o n } } \mathrm { { . } } { \mathrm { m e d i a t e d } } \} ] ^ { \top } \in \mathbb { R } ^ { 6 } ,
$$

where the last indicator in $\mathbf { c } _ { n }$ equals one when booking is made through the Aggie Travel portal or arranged by administrative staf, and ${ \pmb { \alpha } } _ { A } \left( { \pmb { \alpha } } _ { B } \right)$ denotes the first (second) ticket in a pairwise

![](images/026792d67ad39f907b760f96346f848719ec3475042a2d3616f7871d88721909.jpg)  
(a)

![](images/5e89b0eb448582ced6971dcc0e626d333dff485251386ec7b99b2a76b77ab5f3.jpg)  
(b)

![](images/2a4d275a01b185457b28b651edd575b028b8ba702371025799058bfc0745b797.jpg)  
(c)

![](images/1ee8ed77b9cc40ad3ec1dd7c78b73c77065fe02e4bf009e5a0c23fd259c08fee.jpg)  
(d)  
Figure 9: Same as Figure 8, but for flights to London.

template. Following the contextual choice model in [41], we adopt a bilinear utility specification with the query vector

$$
\mathbf { q } _ { \mathfrak { D } , \alpha , n } = \mathrm { v e c } \left( ( \mathbf { q } _ { \mathfrak { D } , \alpha _ { A } } - \mathbf { q } _ { \mathfrak { D } , \alpha _ { B } } ) \mathbf { c } _ { n } ^ { \top } \right) \in \mathbb { R } ^ { 2 4 } ,\tag{5.52}
$$

which encodes interactions between the ticket feature diferences in template α and contextual information of respondent n. With $\Theta _ { \mathfrak { D } } ^ { \star } \in \mathbb { R } ^ { 4 \times 6 }$ and $\theta _ { \mathfrak { D } } ^ { \star } : = \mathrm { v e c } ( \Theta _ { \mathfrak { D } } ^ { \star } )$ ), the corresponding utility score diference is

$$
U ( \mathbf { q } _ { \mathfrak { D } , \alpha _ { A } } , \mathbf { c } _ { n } ) - U ( \mathbf { q } _ { \mathfrak { D } , \alpha _ { B } } , \mathbf { c } _ { n } ) : = ( \mathbf { q } _ { \mathfrak { D } , \alpha _ { A } } - \mathbf { q } _ { \mathfrak { D } , \alpha _ { B } } ) ^ { \top } \mathbf { Q } _ { \mathfrak { D } } ^ { \star } \mathbf { c } _ { n } = ( \theta _ { \mathfrak { D } } ^ { \star } ) ^ { \top } \mathbf { q } _ { \mathfrak { D } , \alpha , n } .
$$

Here we assume that there is no model mis-specification error. Statistical guarantees of this contextual choice model then follow from results established in Sections 3–4 with $d _ { \theta } = 2 4$

For a destination $\mathfrak { D } \ \in \ \{ \mathrm { D C } , \mathrm { L o n d o n } \}$ , we cross each of the 367 retained $\mathbf { c } _ { n }$ with all 48 pairwise-comparison templates ${ \bf q } _ { \mathfrak { D } , { \pmb { \alpha } } _ { A } } - { \bf q } _ { \mathfrak { D } , { \pmb { \alpha } } _ { B } }$ as in (5.52), yielding $3 6 7 \times 4 8 = 1 7 6 1 6$ candidate query instances, which include ticket–respondent combinations that are not observed in the survey. We then solve the D-optimal design problem over this candidate query pool with design budget N. Since the ground truth is unknown, we follow [50, Section 6] and use the Firth correction to estimate $\pmb { \theta } _ { \mathrm { D C } } ^ { \star }$ and $\pmb { \theta } _ { \mathrm { L o n d o n } } ^ { \star }$ based on the 1835 and 2202 real-world responses. We rescale $\theta _ { \mathrm { D C } } ^ { \star } \left( \theta _ { \mathrm { L o n d o n } } ^ { \star } \right)$ to ensure log $B = 3$ over its candidate query pool and use the resulting parameter as the ground truth for simulation. Responses are then generated from the corresponding BTL model.

Figure 7 plots the existence and convergence probabilities of the canonical MLE, while Figures 8 and 9 report the RMSEs, medians, and empirical $1 0 \% { - } 9 0 \%$ percentile ranges of the Euclidean and query-wise errors over 3000 Monte Carlo trials for the DC and London query pools, respectively. Figure 7 shows that when $2 4 \leq N \leq 4 8$ , the MLE fails to exist in almost all trials, and the MLE existence probabilities increase with N and become close to one by approximately $N = 1 9 2$ , but the Euclidean and query-wise RMSEs remain substantially above the corresponding CRLBs at this sample size, as depicted by Figures 8 (a)–(b) and $9 \ ( \mathrm { a } ) \mathrm { - ( b ) }$ . For example, Figure 8 (a) shows a gap between the $\| \Delta \| _ { 2 }$ RMSE and the corresponding median for $5 5 \le N \le 2 9 5$ , together with wide empirical percentile ranges depicted by the shaded-in-green region. This observation indicates that large variability across trials and right-skewed errors inflate the $\| \Delta \| _ { 2 }$ RMSE even when the MLE exists.

We apply the same decomposition method as in Section 5.2 when plotting Figure 8 (c) and examine the source of these large errors. After rescaling the ground-truth parameter, $\| \Delta _ { b i a s } \| _ { 2 }$ 2 lies below the $\| \cdot \| _ { 2 }$ CRLB throughout the plotted sample-size range, see (4.40b). When N is small, the $\lVert \Delta - \Delta _ { l i n } \rVert _ { 2 }$ and $\lVert \Delta - \Delta _ { l i n } - \Delta _ { b i a s } \rVert _ { 2 }$ RMSE curves nearly coincide and remain close to the $\| \Delta \| _ { 2 }$ RMSE curve. Thus, the large error in this regime is driven primarily by higher-order nonlinear terms. For $N \geq 2 5 0$ , these nonlinear residuals fall below the CRLB and continue to decay more rapidly, after which the $\| \Delta \| _ { 2 }$ RMSE approaches the CRLB. Similar behavior is observed for destination London in Figure 9 (c) and under the query-wise metric in Figures 8 (d) and 9 (d). The Firth correction, however, exhibits substantially more stable behavior. Its Euclidean RMSE decreases steadily for $N \geq 1 2 8$ and nearly coincides with the CRLB at $N = 5 7 6 = d _ { \theta } ^ { 2 }$

## 6 Concluding Remarks

In this paper, we investigate error rates of the maximum likelihood estimator for eliciting linearin-parameter utility functions under the BTL framework. Under some moderate conditions, we derive the Cram´er-Rao and minimax lower bounds for general estimators including MLE. We then establish existence of the MLE and conditions under which the upper and lower bounds meet up to some constants.

There are several promising directions for future research. First, the feasible parameter set considered in this paper is characterized by a system of linear equations. Extending the current analysis to more general polyhedral parameter spaces would accommodate linear inequalities that encode monotonicity or convexity of the utility function [33; 14]. Second, our results are developed for linear-in-parameter multivariate utility functions under the BTL framework. It remains unclear whether analogous theoretical treatments apply to nonlinear-in-parameter univariate utility functions or to the Plackett-Luce model. Third, the MLE provides a point estimate of the unknown parameters. An important direction for future work is to investigate whether the theoretical results can be extended to Bayesian learning approaches [60; 30; 45] to describe posterior concentration rates with finite samples, which may also include posterior based on Jefreys’ invariant prior, whose mode yields the Firth correction, as one special case.

## 7 Proofs of Main Results

We now turn to the proofs of our main results developed in Sections 3 and 4.

## 7.1 Proofs of Minimax Lower Bounds

We use two standard tools to transform the minimax risk into a hypothesis-testing problem, namely, Assouad’s lemma and Le Cam’s two-point method.

Lemma 7.1 ([70, Lemma 2.12]) Let $\pmb { S } = \{ - 1 , 1 \} ^ { m }$ be the set of all {±1} sequences of length $m \in \mathbb { Z } _ { + }$ , and equip this space with Hamming distance $\rho _ { H } ( \mathbf { s } ^ { 1 } , \mathbf { s } ^ { 2 } ) : = \sum _ { j = 1 } ^ { m } \Im \{ s _ { j } ^ { 1 } \neq s _ { j } ^ { 2 } \}$ , for any $\mathbf { s } _ { 1 } , \mathbf { s } _ { 2 } \in S$ . Let $\{ \mathbb { P } _ { \mathbf { s } } ^ { ( N ) } \mid \mathbf { s } \in \mathcal { S } \} \subset \mathcal { P } ( \mathcal { V } _ { N } , \mathcal { Y } _ { N } )$ contain $2 ^ { m }$ probability measures indexed by $\mathbf { s } \in { \mathcal { S } }$ With slight notation abuse, let $\mathbb { E } _ { \mathbf { s } } ^ { ( N ) } [ \cdot ]$ denote expectation under $\mathbb { P } _ { \mathbf { s } } ^ { \breve { ( } N ) }$ . Consider all measurable estimators $\hat { \mathbf { s } } : ( \mathcal { V } _ { N } , \mathcal { Y } _ { N } )  ( { \mathcal { S } } , 2 ^ { S } )$ and some positive weights $\{ a _ { j } \} _ { j = 1 } ^ { m }$ . Then

$$
\frac { 1 } { 2 ^ { m } } \sum _ { \mathbf { s } \in \mathcal { S } } \mathbb { E } _ { \mathbf { s } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { m } a _ { j } \mathbb { 1 } \left\{ \hat { s } _ { j } \neq s _ { j } \right\} \right] \geq \frac { 1 } { 2 } \sum _ { j = 1 } ^ { m } a _ { j } \left( 1 - \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \in \mathcal { S } } \mathbb { d } _ { T V } \left( \mathbb { P } _ { \mathbf { s } } ^ { ( N ) } , \mathbb { P } _ { \mathbf { s } ^ { \prime } } ^ { ( N ) } \right) \right) .\tag{7.53}
$$

This weighted form of Assouad’s lemma follows from the proof of [70, Lemma 2.12].

Lemma 7.2 ([70, Theorem 2.2]) Let ρ be a semi-distance, and $\{ \mathbb { P } _ { \pmb { \theta } } ^ { ( N ) } \ | \ \pmb { \theta } \in \ \{ \pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } \} \} \ \subset$ $\mathcal { P } ( \mathcal { V } _ { N } , \mathcal { V } _ { N } )$ be the set of parametric probability measures indexed by $\{ \pmb \theta _ { 1 } , \pmb \theta _ { 2 } \}$ . If there exists a 2δ−separated pair $\pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } \in \Theta _ { B }$ with $\rho ( \pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } ) \geq 2 \delta$ , then

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \rho ^ { 2 } ) \geq \frac { \delta ^ { 2 } } { 2 } ( 1 - \mathsf { d } _ { T V } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) ) ,\tag{7.54a}
$$

$$
\operatorname* { i n f } _ { \hat { \theta } \in \mathcal A _ { N } } \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb P _ { \theta } ^ { ( N ) } \left\{ \rho ^ { 2 } ( \hat { \theta } , \pmb \theta ) \geq \delta ^ { 2 } \right\} \geq \frac 1 2 ( 1 - \mathsf d | _ { T V } ( \mathbb P _ { \pmb \theta _ { 1 } } ^ { ( N ) } , \mathbb P _ { \pmb \theta _ { 2 } } ^ { ( N ) } ) ) .\tag{7.54b}
$$

We also need to upper bound the TV distance between two joint Bernoulli distributions $\mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) }$ induced by some $\pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } \in \mathbb { R } ^ { d _ { \theta } }$ . By Pinsker’s inequality [70, Lemma 2.5], i.e., $\| _ { T V } ( \mathrm { P } , \mathrm { Q } ) \ \leq$ $\sqrt { { \sf d l } _ { K L } ( \mathrm { P } , \mathrm { Q } ) / 2 }$ , and the additivity of KL divergence for product measures, i.e., dl $\mathbf { \Psi } _ { K L } ( \mathbb { P } _ { \pmb { \theta } _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \pmb { \theta } _ { 2 } } ^ { ( N ) } ) =$ $\begin{array} { r } { \sum _ { \pmb { \alpha } \in \mathcal { E } _ { s } } n _ { \pmb { \alpha } } \mathsf { d } \mathsf { I } _ { K L } ( \mathbf { P } _ { \pmb { \theta _ { 1 } } } ^ { \alpha } , \mathbf { P } _ { \pmb { \theta _ { 2 } } } ^ { \alpha } ) } \end{array}$ , it is enough to find an upper bound on d $K L ( \mathrm { P } _ { \theta _ { 1 } } ^ { \alpha } , \mathrm { P } _ { \theta _ { 2 } } ^ { \alpha } )$ . This estimate is standard and due to [61].

Lemma 7.3 ([61, Lemma 8]) Let $\theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d _ { \theta } }$ be arbitrary but fixed. For two joint distributions of the $f u l l$ observation $\begin{array} { r } { \hat { Y ^ { ( N ) } } , \ i . e . , \mathbb { P } _ { \pmb { \theta } _ { 1 } } ^ { ( \bar { N } ) } , \ \mathbb { P } _ { \pmb { \theta } _ { 2 } } ^ { ( N ) } } \end{array}$ defined as in (2.14), we have

$$
\mathsf { d } _ { K L } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) \leq \frac { N } { 8 \sigma ^ { 2 } } ( \theta _ { 1 } - \theta _ { 2 } ) ^ { \top } \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathfrak { A } ^ { \top } \mathbf { L } \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ( \theta _ { 1 } - \theta _ { 2 } ) .\tag{7.55}
$$

The following univariate lower bound on the sum of two KL divergences between Bernoulli distributions complements Lemma 7.3.

Lemma 7.4 Let $a , b \in [ - \log B$ , log $B ] , \ z \in \mathbb { R }$ , and $\mathrm { P } _ { \Psi ^ { \prime } } ^ { a } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { b } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z }$ be Bernoulli distributions with success rates $\Psi ^ { \prime } ( a ) , \Psi ^ { \prime } ( b )$ , and $\Psi ^ { \prime } ( z )$ , respectively. Then

$$
{ \sf d } _ { K L } ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { a } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } ) + { \sf d } _ { K L } ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { b } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } ) \geq \frac { 1 } { 1 6 B } ( a - b ) ^ { 2 } .
$$

Moreover, let $\theta _ { 1 } , \theta _ { 2 } \in \Theta _ { B }$ , and let

$$
\begin{array} { r l } & { \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( v ) = - \Psi ^ { \prime } ( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 1 } } { \sigma } ) \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } + \Psi ( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } ) , } \\ & { \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( v ) = - \Psi ^ { \prime } ( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 2 } } { \sigma } ) \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } + \Psi ( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } ) , } \end{array}
$$

denote the population-level negative log-likelihood of one observation associated with query $\pmb { \alpha } \in$ $\mathcal { E } _ { s }$ , evaluated at some $\pmb { v } \in \mathbb { R } ^ { d _ { \theta } }$ . Then

$$
\mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( \theta _ { 1 } ) + \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( \theta _ { 2 } ) \geq \frac { 1 } { 1 6 B \sigma ^ { 2 } } \Big | \mathbf { q } _ { \alpha } ^ { \top } ( \theta _ { 1 } - \theta _ { 2 } ) \Big | ^ { 2 } , \forall v \in \mathbb { R } ^ { d _ { \theta } } .\tag{7.56}
$$

The proof is deferred to Appendix A.4.

## 7.1.1 Proof of Theorem 3.1

Proof of (3.34a). The proof proceeds in two main steps: we first define a binary precoder/decoder pair and relate the mean square error to Hamming distance, and then use Lemmas 7.1 and 7.3 to obtain a lower bound on the weighted Hamming distance and upper bound on the TV distance of two properly designed probability measures, respectively.

STEP 1. We start by defining the precoder, which is similar to those in [61, Theorem 1] and [43, Theorem 5], but we use a Rademacher concentration argument to retain a suficiently large subset of feasible points in $\Theta _ { B }$ . Let $\mathcal { S } : = \{ - 1 , 1 \} ^ { d _ { \theta } }$ , and $( S , 2 ^ { S } , \mathbb { P } _ { \xi } )$ denote the probability space that supports a Rademacher sequence $\pmb { \xi } : = [ \xi _ { 1 } , \dots , \xi _ { d _ { \theta } } ] ^ { \top }$ . That is, for $A \subseteq S , \mathbb { P } _ { \xi } ( A ) = 2 ^ { - d _ { \theta } } | A |$ and $j \in [ d _ { \theta } ] , \xi _ { j } : S \to \{ - 1 , 1 \}$ is the coordinate projection. We denote a realization of ${ \pmb { \xi } } \ \mathrm { a s } \ \mathbf { s } \in { \mathcal { S } }$ and set $\mathbb { E } _ { \xi } [ \cdot ]$ be the expectation under $\mathbb { P } _ { \xi }$ . Let $\mathbf { W } : = \mathbf { H } \mathbf { A } \mathbf { H } ^ { \top }$ denote the spectral decomposition of W, with $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { d _ { \theta } } )$ and $\mathbf { H } = [ \mathbf { h } _ { 1 } , \ldots , \mathbf { h } _ { d _ { \theta } } ]$ , where $\mathbf { h } _ { j } \in \mathbb { R } ^ { d _ { \theta } }$ . For some $\delta > 0$ define

$$
\pmb \theta ^ { ( \pmb \xi ) } : = \delta \mathbf { H } \mathbf { A } ^ { - 1 / 2 } \pmb \xi = \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta \xi _ { j } \mathbf { h } _ { j } } { \sqrt { \lambda _ { j } } } , \quad \mathbf { w } ^ { ( \pmb \xi ) } : = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb \theta ^ { ( \pmb \xi ) } .\tag{7.57}
$$

Consider the set $\begin{array} { r } { \mathcal { Q } _ { B } : = \{ \mathbf { s } \in \mathcal { S } \mid \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \left| \mathbf { q } _ { \alpha } ^ { \top } \pmb { \theta } ^ { ( \mathbf { s } ) } \right| \leq \sigma \log ( B / B _ { 0 } ) \} } \end{array}$ . By Assumption 2.4, for any $\mathbf { s } \in \mathcal { Q } _ { B }$

$$
\operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \frac { \left| \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \pmb { \theta } ^ { ( \mathbf { s } ) } \right| } { \sigma } \leq \log B _ { 0 } + \log ( B / B _ { 0 } ) = \log B .
$$

Thus, $\{ \pmb { \theta } ^ { ( \mathbf { s } ) } \ | \ \mathbf { s } \in \mathcal { Q } _ { B } \} \subseteq \Theta _ { B }$ by Assumption 2.3 and (2.12), from which we have

$$
\begin{array} { r l } & { \underset { \theta \in \Theta _ { B } } { \operatorname* { s u p } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \left. \hat { \theta } - \theta \right. _ { 2 } ^ { 2 } \right] \geq \frac { 1 } { | Q _ { B } | } \sum _ { \mathrm { s } \in \mathcal { Q } _ { B } } \mathbb { E } _ { \theta ^ { ( \mathrm { s } ) } } ^ { ( N ) } \left[ \left. \hat { \theta } - \theta ^ { ( \mathrm { s } ) } \right. _ { 2 } ^ { 2 } \right] } \\ & { = \frac { 2 ^ { - d _ { \theta } } \sum _ { \mathrm { s } \in \mathcal { Q } _ { B } } \mathbb { E } _ { \theta ^ { ( \mathrm { s } ) } } ^ { ( N ) } \left[ \left. \hat { \theta } - \theta ^ { ( \mathrm { s } ) } \right. _ { 2 } ^ { 2 } \right] } { 2 ^ { - d _ { \theta } } | Q _ { B } | } = \frac { 2 ^ { - d _ { \theta } } \sum _ { \mathrm { s } \in \mathcal { S } } \mathbb { 1 } \left\{ \mathrm { s } \in Q _ { B } \right\} \mathbb { E } _ { \theta ^ { ( \mathrm { s } ) } } ^ { ( N ) } \left[ \left. \hat { \theta } - \theta ^ { ( \mathrm { s } ) } \right. _ { 2 } ^ { 2 } \right] } { 2 ^ { - d _ { \theta } } | Q _ { B } | } } \\ & { = \frac { \mathbb { E } _ { \xi } \left[ \mathbb { 1 } \left\{ \xi \in \mathcal { Q } _ { B } \right\} \mathbb { E } _ { \theta ^ { ( \mathrm { s } ) } } ^ { ( N ) } \left[ \left. \hat { \theta } - \theta ^ { ( \mathrm { s } ) } \right. _ { 2 } ^ { 2 } \right] \right] } { \mathbb { P } _ { \xi } \left\{ \xi \in \mathcal { Q } _ { B } \right\} } . } \end{array}
$$

Then our primary interest is on a single realization s of $\boldsymbol { \xi }$ and $\pmb \theta ^ { ( \mathbf { s } ) }$ . By (7.57) and the projection property of $\mathbf { H } \mathbf { H } ^ { \top }$ , we have<sup>11</sup>

$$
\begin{array} { l } { \displaystyle \| \hat { \pmb \theta } - \pmb \theta ^ { ( \mathrm { s } ) } \| _ { 2 } ^ { 2 } \geq \| \mathbf H \mathbf H ^ { \top } ( \hat { \pmb \theta } - \pmb \theta ^ { ( \mathrm { s } ) } ) \| _ { 2 } ^ { 2 } = \| \mathbf H ^ { \top } \hat { \pmb \theta } - \mathbf H ^ { \top } \pmb \theta ^ { ( \mathrm { s } ) } \| _ { 2 } ^ { 2 } } \\ { \displaystyle = \| \mathbf H ^ { \top } \hat { \pmb \theta } - \delta \mathbf \Lambda ^ { - 1 / 2 } \mathbf s \| _ { 2 } ^ { 2 } = \displaystyle \sum _ { j = 1 } ^ { d _ { \theta } } \left( \mathbf h _ { j } ^ { \top } \hat { \pmb \theta } - \frac \delta { \sqrt { \lambda _ { j } } } s _ { j } \right) ^ { 2 } , \forall \mathbf s \in \mathcal S . } \end{array}\tag{7.59}
$$

We use $\mathbf { h } _ { j } ^ { \top } \hat { \pmb { \theta } }$ to define the decoder

$$
\hat { s } _ { j } : = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { ~ i f ~ } \mathbf h _ { j } ^ { \top } \hat { \pmb \theta } \ge 0 , } \\ { - 1 , } & { \mathrm { ~ i f ~ } \mathbf h _ { j } ^ { \top } \hat { \pmb \theta } < 0 . } \end{array} \right.
$$

For any $j \in [ d _ { \theta } ] , \mathrm { i f } \hat { s } _ { j } \neq s _ { j }$ , then $\begin{array} { r } { \left| \mathbf { h } _ { j } ^ { \top } \hat { \pmb { \theta } } - \frac { \delta } { \sqrt { \lambda _ { j } } } s _ { j } \right| \geq \frac { \delta } { \sqrt { \lambda _ { j } } } } \end{array}$ . Thus

$$
\sum _ { j = 1 } ^ { d _ { \theta } } \left( \mathbf { h } _ { j } ^ { \top } \hat { \pmb { \theta } } - \frac { \delta } { \sqrt { \lambda _ { j } } } s _ { j } \right) ^ { 2 } \geq \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq s _ { j } \} , \forall \mathbf { s } \in \mathcal { S } .\tag{7.60}
$$

By substituting (7.59) and (7.60) into (7.58), we note that it is enough to derive an upper bound on $\mathbb { P } _ { \xi } \{ \xi \notin \mathcal { Q } _ { B } \}$ and a lower bound on $\begin{array} { r } { \mathbb { E } _ { \pm } \left[ \mathbb { E } _ { \pmb { \theta } ^ { ( \pm ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] } \end{array}$ . Indeed

$$
\begin{array} { r l } & { \underset { \theta \in \Theta _ { B } } { \operatorname* { s u p } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \lVert \hat { \theta } - \theta \rVert _ { 2 } ^ { 2 } \right] \geq \mathbb { E } _ { \xi } \left[ \mathbb { 1 } \{ \xi \in \mathcal { Q } _ { B } \} \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { \xi } \left[ \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] - \mathbb { E } _ { \xi } \left[ \mathbb { 1 } \{ \xi \notin \mathcal { Q } _ { B } \} \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] } \end{array}
$$

$$
\ge \mathbb { E } _ { \pm } \left[ \mathbb { E } _ { \pmb { \theta } ^ { ( \pmb { \mathscr { s } } ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] - \mathbb { P } _ { \pmb { \xi } } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) ,\tag{7.61}
$$

where the last inequality is from the deterministic upper bound $\begin{array} { r } { \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \leq \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } \end{array}$

STEP 2. We first derive an upper bound for $\mathbb { P } _ { \xi } \{ \xi ~ \notin ~ \mathcal { Q } _ { B } \}$ . By Hoefding’s inequality [73, Theorem 2.2.5],

$$
\mathbb { P } _ { \xi } \{ \xi \notin \mathcal { Q } _ { B } \} = \mathbb { P } _ { \xi } \left\{ \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \left. \mathbf { q } _ { \alpha } ^ { \top } \pmb { \theta } ^ { ( \xi ) } \right. > \sigma \log ( B / B _ { 0 } ) \right\} \le 2 d ^ { 2 } \exp \left( - \frac { \sigma ^ { 2 } \log ^ { 2 } ( B / B _ { 0 } ) } { 2 \delta ^ { 2 } \mu _ { Q } d _ { \theta } } \right) ,\tag{7.62}
$$

where the last inequality follows from $\begin{array} { r } { \| \mathbf { \Delta } ^ { - 1 / 2 } \mathbf { H } ^ { \top } \mathbf { q } _ { \alpha } \| _ { 2 } ^ { 2 } \leq \operatorname* { m a x } _ { \alpha \in \mathcal { E } _ { s } } \mathbf { q } _ { \alpha } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { \alpha } = \mu _ { Q } d _ { \theta } } \end{array}$ for any ${ \pmb { \alpha } } \in { \mathcal { E } } _ { s }$ , and the fact that $| { \mathcal { E } } _ { s } | \leq d ^ { 2 }$ . By setting $\begin{array} { r } { \delta = \operatorname* { m i n } \left\{ \frac { \sigma } { \sqrt { N } } , \frac { \sigma \log \left( B / B _ { 0 } \right) } { \sqrt { 4 \mu _ { Q } d _ { \theta } \log \left( 4 d \right) } } \right\} } \end{array}$ , we obtain by inequality (7.62) that $\begin{array} { r } { \mathbb { P } _ { \pmb { \xi } } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \le \frac { 1 } { 8 } } \end{array}$

Next, by Lemma 7.1 and Pinsker’s inequality

$$
\mathbb { E } _ { \xi } \left[ \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq \xi _ { j } \} \right] \right] = \frac { 1 } { 2 ^ { d _ { \theta } } } \sum _ { \mathbf { s } \in S } \mathbb { E } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } \left[ \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq s _ { j } \} \right]
$$

$$
\begin{array} { r l } & { \geq \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \left( 1 - \displaystyle \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \colon \boldsymbol { \rho } _ { H } ( \mathbf { s } , \mathbf { s } ^ { \prime } ) = 1 } \mathsf { d } _ { T V } ( \mathbb { P } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } , \mathbb { P } _ { \theta ^ { ( \mathbf { s } ^ { \prime } ) } } ^ { ( N ) } ) \right) } \\ & { \geq \displaystyle \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \left( 1 - \displaystyle \operatorname* { s u p } _ { s , s ^ { \prime } \colon \boldsymbol { \rho } _ { H } ( s , s ^ { \prime } ) = 1 } \sqrt { \displaystyle \frac { 1 } { 2 } \mathsf { d } _ { K L } ( \mathbb { P } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } , \mathbb { P } _ { \theta ^ { ( s ^ { \prime } ) } } ^ { ( N ) } ) } \right) . } \end{array}\tag{7.63}
$$

We then use Lemma 7.3 to derive an upper bound for the KL divergence term

$$
\begin{array} { r } { \displaystyle \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \colon \rho _ { H } ( \mathbf { s } , \mathbf { s } ^ { \prime } ) = 1 } \mathrm { d } _ { K L } ( { \mathbb { P } } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } , { \mathbb { P } } _ { \theta ^ { ( \mathbf { s } ^ { \prime } ) } } ^ { ( N ) } ) \leq \displaystyle \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \colon \rho _ { H } ( \mathbf { s } , \mathbf { s } ^ { \prime } ) = 1 } \frac { N } { 8 \sigma ^ { 2 } } ( \theta ^ { ( \mathbf { s } ) } - \theta ^ { ( \mathbf { s } ^ { \prime } ) } ) ^ { \top } \mathbf { W } ( \theta ^ { ( \mathbf { s } ) } - \theta ^ { ( \mathbf { s } ^ { \prime } ) } ) } \\ { = \displaystyle \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \colon \rho _ { H } ( \mathbf { s } , \mathbf { s } ^ { \prime } ) = 1 } \frac { N \delta ^ { 2 } } { 8 \sigma ^ { 2 } } \| \mathbf { s } - \mathbf { s } ^ { \prime } \| _ { 2 } ^ { 2 } = \frac { N \delta ^ { 2 } } { 2 \sigma ^ { 2 } } \leq \frac { 1 } { 2 } , } \end{array}\tag{7.64}
$$

where the last equality is from the fact that $\mathbf { s } , \mathbf { s } ^ { \prime } \in \{ \pm 1 \} ^ { d _ { \theta } }$ , and the last inequality is due to the specification of δ. By combining (7.61)-(7.64), and the choice of $\delta ,$ we have

$$
\begin{array} { r l r } & { } & { \displaystyle \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } \left[ \| \hat { \pmb { \theta } } - \pmb { \theta } \| _ { 2 } ^ { 2 } \right] \geq \frac { 1 } { 4 } \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } - \frac { 1 } { 8 } \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) = \frac { 1 } { 8 } \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } \\ & { } & { \displaystyle = \frac { \sigma ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } { 8 } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} . } \end{array}
$$

Taking infimum over $\hat { \pmb { \theta } }$ on both sides of the above inequality gives rise to (3.34a).

Proof of (3.34b). First, by (7.59) and (7.60)

$$
\lVert \hat { { \boldsymbol { \theta } } } - { \boldsymbol { \theta } } ^ { ( \mathbf { s } ) } \rVert _ { 2 } ^ { 2 } \geq \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq s _ { j } \} , \forall \mathbf { s } \in \mathcal { Q } _ { B } ,\tag{7.65}
$$

$\mathbb { P } _ { \pmb { \theta } ( \mathbf { s } ) } ^ { ( N ) }$ almost surely, where

$$
0 \leq \sum _ { j = 1 } ^ { d _ { \theta } } \frac { \delta ^ { 2 } } { \lambda _ { j } } \mathbb { 1 } \{ \hat { s } _ { j } \neq s _ { j } \} \leq \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) .\tag{7.66}
$$

We adopt the same choice of $\delta$ as in the derivation above. That is, for all $t \in ( 0 , \delta ^ { 2 } \mathrm { t r } ( { \bf W } ^ { - 1 } ) )$

$$
\begin{array} { r l } { \underset { \theta \in \Theta _ { B } } { \operatorname* { s u p } } \frac { \mathbb { P } ( \theta ) } { \theta } \{ | \hat { \theta } - \theta | _ { 2 } ^ { 2 } \geq t \} \geq \frac { 1 } { | \mathcal { Q } _ { B } | } \underset { \epsilon \in \mathcal { Q } _ { B } } { \sum \operatorname* { s u p } } \frac { \mathbb { P } ( \theta ^ { ( 1 ) } \cdot ) } { \theta ( \theta ^ { ( 1 ) } - \theta ^ { ( 1 ) } | _ { 2 } ^ { 2 } \geq t ) } \} } & { } \\ & { \geq \frac { 1 } { | \mathcal { Q } _ { B } | } \underset { \epsilon \in \mathcal { Q } _ { B } } { \sum \operatorname* { s u p } } \frac { \mathbb { P } ( \theta ^ { ( 1 ) } ) } { \theta ^ { ( 1 ) } } \{ \sum _ { j = 1 } ^ { \theta } \frac { \hat { \alpha } ^ { 2 } } { \lambda _ { j } } \Im \{ \hat { s } _ { j } - s _ { j } \} \geq t \} \geq t \} } \\ & { \geq \frac { | \mathcal { Q } _ { B } | ^ { - 1 } \sum _ { \kappa \in \mathcal { Q } _ { B } } \mathbb { E } ( \theta ^ { ( 1 ) } ) } { \delta ^ { 2 } \operatorname* { t r } ( \mathbb { W } ^ { - 1 } ) - l } \{ \sum _ { j = 1 } ^ { \theta } \frac { \hat { \alpha } ^ { 2 } } { \lambda _ { j } } \{ \delta _ { j } \} ^ { 2 } \{ s _ { j } \} \} - l } \\ &  = \frac { \frac { 2 \hat { \alpha } ^ { 2 } \theta } { | \mathcal { Q } _ { B } | } \cdot \mathbb { E } \{ [ \mathbb { 1 } \{ \xi \in \mathcal { Q } _ { B } } ] \sum _ { \theta \in \mathcal { Q } _ { B } } ^ { \mathbb { P } ( \theta ^ { ( 1 ) } ) } [ \frac { \sum _ { \hat { \alpha } ^ { 2 } } ^ { \theta } }  \sum _ { j = 1 } ^ { \theta } \sum _ { l } \{ \hat { s } _ { j } \end{array}\tag{7.67}
$$

where the third inequality is from Lemma B.1, and in the last inequality we use $\frac { 2 ^ { d _ { \theta } } } { | \mathcal { Q } _ { B } | } \geq 1$ and the lower bound on the r.h.s. of (7.61) deduced in the proof of (3.34a). By setting $\begin{array} { r } { t = \frac { \delta ^ { 2 } \mathrm { t r } ( \mathbf { W } ^ { - 1 } ) } { 1 6 } } \end{array}$

and $\begin{array} { r } { \delta = \operatorname* { m i n } \left\{ \frac { \sigma } { \sqrt { N } } , \frac { \sigma \log \left( B / B _ { 0 } \right) } { \sqrt { 4 \mu _ { Q } d _ { \theta } \log \left( 4 d \right) } } \right\} } \end{array}$ , inequality (7.67) yields

$$
\begin{array} { r l } & { \underset { \theta \in \Theta _ { R } } { \operatorname* { s u p } } \mathbb { P } _ { \theta } ^ { ( N ) } \left\{ \| \hat { \theta } - \theta \| _ { 2 } ^ { 2 } \geq \frac { \hat { \theta } ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) } { 1 6 } \right\} = \underset { \theta \in \Theta _ { R } } { \operatorname* { s u p } } \mathbb { P } _ { \theta } ^ { ( N ) } \left\{ \| \hat { \theta } - \theta \| _ { 2 } ^ { 2 } \geq \frac { \sigma ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) } { 1 6 } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} \right\} } \\ & { \quad \quad \quad \quad \geq \frac { \hat { \theta } ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) / 8 - \delta ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) / 1 6 } { \delta ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) - \delta ^ { 2 } \operatorname { t r } ( \mathbf { W } ^ { - 1 } ) / 1 6 } = \frac { 1 } { 1 5 } , } \end{array}
$$

taking infimum over $\hat { \pmb { \theta } }$ on both sides proves (3.34b).

## 7.1.2 Proof of Corollary 3.2

A direct corollary from Lemma 7.4 lower bounds the excess risk, i.e.,

$$
\begin{array} { r l } & { \displaystyle \mathcal { E } _ { \mathcal { L } } ( \pmb { v } , \pmb { \theta } _ { 1 } ) + \mathcal { E } _ { \mathcal { L } } ( \pmb { v } , \pmb { \theta } _ { 2 } ) = \mathcal { L } _ { \pmb { \theta _ { 1 } } } ( \pmb { v } ) - \mathcal { L } _ { \pmb { \theta _ { 1 } } } ( \pmb { \theta _ { 1 } } ) + \mathcal { L } _ { \pmb { \theta _ { 2 } } } ( \pmb { v } ) - \mathcal { L } _ { \pmb { \theta _ { 2 } } } ( \pmb { \theta _ { 2 } } ) } \\ & { \quad \quad \quad \quad = \displaystyle \frac { 1 } { N } \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } \left( \mathcal { L } _ { \pmb { \theta _ { 1 } } } ^ { \alpha } ( \pmb { v } ) - \mathcal { L } _ { \pmb { \theta _ { 1 } } } ^ { \alpha } ( \pmb { \theta _ { 1 } } ) + \mathcal { L } _ { \pmb { \theta _ { 2 } } } ^ { \alpha } ( \pmb { v } ) - \mathcal { L } _ { \pmb { \theta _ { 2 } } } ^ { \alpha } ( \pmb { \theta _ { 2 } } ) \right) } \\ & { \quad \quad \quad \quad \geq \displaystyle \frac { 1 } { 1 6 B \sigma ^ { 2 } } \sum _ { \alpha \in \mathcal { E } _ { s } } \frac { n _ { \alpha } } { N } \Big \vert \mathbf { q } _ { \alpha } ^ { \top } ( \pmb { \theta _ { 1 } } - \pmb { \theta _ { 2 } } ) \Big \vert ^ { 2 } = \frac { 1 } { 1 6 B \sigma ^ { 2 } } \Vert \pmb { \theta _ { 1 } } - \pmb { \theta _ { 2 } } \Vert _ { \mathbb { W } } ^ { 2 } , } \end{array}\tag{7.68}
$$

for all $\pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } \in \Theta _ { B }$ and $\pmb { v } \in \mathbb { R } ^ { d _ { \theta } }$ . We adopt the notation introduced in Section 7.1.1.

First, let $\pmb \theta ^ { ( \pmb \varepsilon ) }$ be defined as in (7.57). Thus $\begin{array} { r } { \mathbb { P } _ { \pm } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \le \frac { 1 } { 8 } \mathrm { w h e n } \delta = \operatorname* { m i n } \left\{ \frac { \sigma } { \sqrt { N } } , \frac { \sigma \log \left( B / B _ { 0 } \right) } { \sqrt { 4 \mu _ { Q } d _ { \theta } \log \left( 4 d \right) } } \right\} } \end{array}$ Fix any $\hat { \pmb { \theta } } \in \mathcal { A } _ { N }$ , and define the decoder

$$
\hat { \mathbf { s } } : = \arg \operatorname* { m i n } _ { \mathbf { t } \in \mathcal { Q } _ { B } } \mathcal { E } _ { \mathcal { L } } ( \hat { \pmb { \theta } } , \pmb { \theta } ^ { ( \mathbf { t } ) } ) ,
$$

where we break ties according to a fixed ordering of $\mathcal { Q } _ { B }$ , and we have $\pmb { \theta } ^ { ( \hat { \mathbf { s } } ) } \in \Theta _ { B }$ . Thus

$$
\begin{array} { r l } { \underset { \omega \in \{ S _ { \delta } \} _ { \leq } } { \operatorname* { s u p } } \mathcal { A } _ { \delta } ^ { ( N ) } [ \mathcal { E } _ { \xi } \{ \hat { \theta } _ { \delta } \theta _ { \delta } \} ] \leq \frac { 1 } { | \mathcal { E } _ { \epsilon } | } \displaystyle \sum _ { \omega \in \mathcal { J } _ { \delta } } \frac { d ( \delta ) ^ { N } } { | \mathcal { E } _ { \epsilon } \langle \hat { \theta } _ { \delta } ^ { N } | } \{ E _ { \epsilon } \hat { \theta } _ { \delta } ^ { N } \hat { \theta } _ { \delta } ^ { N } \} ] } & { } \\ & { \leq \frac { 1 } { 2 } \displaystyle \sum _ { \vec { u } \in \{ S } _ { \epsilon } } \frac { \mathbb { E } _ { \epsilon } \mathbb { E } _ { \epsilon } \mathbb { E } _ { \epsilon } ^ { N } } { \exp ^ { \mathcal { I } _ { \epsilon } } [ \mathcal { E } _ { \epsilon } \hat { u } _ { \delta } ^ { N } \hat { \theta } _ { \epsilon } ^ { N } ] [ \mathcal { E } _ { \epsilon } \hat { u } _ { \delta } ^ { N } \hat { \theta } _ { \epsilon } ^ { N } ] }  \\ & { \overset { \leq \exp ^ { \mathcal { I } _ { \epsilon } } } { \geq } \frac { \mathbb { E } _ { \epsilon } \mathbb { E } _ { \epsilon } \mathbb { E } _ { \epsilon } } { \geq \exp ^ { \mathcal { I } _ { \epsilon } } [ \mathbb { E } _ { \epsilon } \hat { \theta } _ { \epsilon } ^ { N } - \mathbb { E } _ { \epsilon } ^ { N } ] [ \mathbb { E } _ { \epsilon } ^ { N } - \mathbb { E } _ { \epsilon } ^ { N } ] [ \mathbb { E } _ { \epsilon } ^ { N } ] } } \\ &  - \frac { \rho ^ { N } }  3 2 N ^ { 2 } ( \mathscr { D } _ { \epsilon } \hat { \theta } _ { \epsilon } ^ { N } ) \displaystyle \sum _ { \omega \in \mathcal { J } _ { \epsilon } } \mathbb { E } _ { \epsilon } ^ \end{array}\tag{7.69}
$$

According to (7.61), the r.h.s. of (7.69) can be lower bounded as

$$
\mathbb { E } _ { \xi } \left[ \mathbb { 1 } \{ \pmb { \xi } \in \mathcal { Q } _ { B } \} \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \rho _ { H } ( \pmb { \xi } , \hat { \mathbf { s } } ) \right] \right] = \mathbb { E } _ { \pmb { \xi } } \left[ \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \rho _ { H } ( \pmb { \xi } , \hat { \mathbf { s } } ) \right] \right] - \mathbb { E } _ { \pmb { \xi } } \left[ \mathbb { 1 } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \mathbb { E } _ { \theta ^ { ( \xi ) } } ^ { ( N ) } \left[ \rho _ { H } ( \pmb { \xi } , \hat { \mathbf { s } } ) \right] \right]
$$

$$
\begin{array} { r l r } & { \displaystyle = \frac { 1 } { 2 ^ { d _ { \theta } } } \sum _ { \mathbf { s } \in \mathcal { S } } \mathbb { E } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } \left[ \rho _ { H } \big ( \mathbf { s } , \hat { \mathbf { s } } \big ) \right] - \mathbb { E } _ { \xi } \left[ \mathbb { 1 } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \mathbb { E } _ { \theta ^ { ( \pm ) } } ^ { ( N ) } \left[ \rho _ { H } \big ( \pmb { \xi } , \hat { \mathbf { s } } \big ) \right] \right] } & \\ & { \displaystyle \geq \frac { d _ { \theta } } { 2 } \left( 1 - \operatorname* { s u p } _ { \mathbf { s } , \mathbf { s } ^ { \prime } \colon \rho _ { H } ( \mathbf { s } , \mathbf { s } ^ { \prime } ) = 1 } \mathbb { d } _ { T V } \big ( \mathbb { P } _ { \theta ^ { ( \mathbf { s } ) } } ^ { ( N ) } , \mathbb { P } _ { \theta ^ { ( \mathbf { s } ^ { \prime } ) } } ^ { ( N ) } \big ) \right) - d _ { \theta } \mathbb { P } _ { \xi } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} } & \\ & { \displaystyle \geq \frac { d _ { \theta } } { 4 } - d _ { \theta } \mathbb { P } _ { \xi } \{ \pmb { \xi } \notin \mathcal { Q } _ { B } \} \geq \frac { d _ { \theta } } { 8 } , } & { ( 7 . 7 } \end{array}\tag{0}
$$

where the first inequality is from Lemma 7.1 and the worst-case upper bound $\rho _ { H } ( \pmb { \xi } , \hat { \mathbf { s } } ) \leq d _ { \theta }$ over all ${ \pmb { \xi } } , { \hat { \bf s } } \in S$ . The second and last inequalities in (7.70) follow from analogous derivation as in (7.64) and (7.62), respectively.

By substituting (7.70) into (7.69), we obtain

$$
\operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \mathcal { E } _ { \mathcal { L } } ( \hat { \theta } , \theta ) \right] \geq \frac { \delta ^ { 2 } d _ { \theta } } { 6 4 B \sigma ^ { 2 } } = \frac { d _ { \theta } } { 6 4 B } \operatorname* { m i n } \left\{ \frac { 1 } { N } , \frac { \log ^ { 2 } ( B / B _ { 0 } ) } { 4 \mu _ { Q } d _ { \theta } \log ( 4 d ) } \right\} .
$$

Taking infimum over $\hat { \pmb { \theta } }$ on both sides gives rise to (3.37a). Inequality (3.37b) follows from the deterministic inequality implied by (7.69)

$$
\mathcal { E } _ { \mathcal { L } } ( \hat { \pmb { \theta } } , \pmb { \theta } ^ { ( \mathbf { s } ) } ) \geq \frac { \delta ^ { 2 } } { 8 B \sigma ^ { 2 } } \rho _ { H } ( \mathbf { s } , \hat { \mathbf { s } } ) , \forall \mathbf { s } \in \mathcal { Q } _ { B } ,
$$

and

$$
0 \leq \rho _ { H } ( \mathbf { s } , \hat { \mathbf { s } } ) \leq d _ { \theta } , \forall \mathbf { s } \in \mathcal { S } ,
$$

and then mimicking the proof of (3.34b) with $\begin{array} { r } { t = \frac { \delta ^ { 2 } d _ { \theta } } { 1 2 8 B \sigma ^ { 2 } } } \end{array}$

## 7.1.3 Proof of Theorem 3.2

It is enough to prove (3.39a) via Lemma 7.2. For any but fixed ${ \pmb { \alpha } } \in { \mathcal { E } } _ { s }$ such that ${ \bf q } _ { \alpha } \neq { \bf 0 }$ , let $\begin{array} { r } { \mu _ { \pmb { \alpha } } : = \frac { 1 } { d _ { \pmb { \alpha } } } \pmb { \mathrm { q } } _ { \pmb { \alpha } } ^ { \top } \pmb { \mathrm { W } } ^ { - 1 } \pmb { \mathrm { q } } _ { \pmb { \alpha } } } \end{array}$ and $\begin{array} { r } { \mathbf { h } _ { \alpha } : = \frac { \mathbf { W } ^ { - 1 } \mathbf { q } _ { \alpha } } { \sqrt { \mu _ { \alpha } d _ { \theta } } } } \end{array}$ . Suppose that we can find $\delta > 0$ such that $\pmb { \theta } _ { 1 } : = \delta \mathbf { h } _ { \alpha }$ $\pmb { \theta } _ { 2 } : = - \delta \mathbf { h } _ { \alpha } \in \Theta _ { B }$ . Then

$$
\begin{array} { r l } & { \| \pmb { \theta } _ { 1 } - \pmb { \theta } _ { 2 } \| _ { Q , \infty } = \underset { \beta \in \mathcal { E } _ { s } } { \operatorname* { m a x } } | \mathbf { q } _ { \beta } ^ { \top } ( \pmb { \theta } _ { 1 } - \pmb { \theta } _ { 2 } ) | = 2 \delta \underset { \beta \in \mathcal { E } _ { s } } { \operatorname* { m a x } } | \mathbf { q } _ { \beta } ^ { \top } \mathbf { h } _ { \alpha } | } \\ & { \qquad \geq 2 \delta | \mathbf { q } _ { \alpha } ^ { \top } \mathbf { h } _ { \alpha } | = 2 \delta \frac { \mathbf { q } _ { \alpha } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { \alpha } } { \sqrt { \mu _ { \alpha } d _ { \theta } } } = 2 \delta \sqrt { \mu _ { \alpha } d _ { \theta } } , } \end{array}
$$

from which we construct a $2 \delta \sqrt { \mu _ { \alpha } d _ { \theta } }$ −separated pair measured in $\| \cdot \| _ { Q , \infty }$ . By Lemma 7.3 and setting $\delta = \frac { \sigma } { \sqrt { N } }$

$$
\begin{array} { l } { \displaystyle \mathrm { d } _ { T V } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) \leq \sqrt { \mathrm { d } _ { K L } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) / 2 } \leq \sqrt { \frac { N } { 1 6 \sigma ^ { 2 } } ( \theta _ { 1 } - \theta _ { 2 } ) ^ { \top } \mathbf { W } ( \theta _ { 1 } - \theta _ { 2 } ) } } \\ { \displaystyle \quad \quad = \sqrt { \frac { \delta ^ { 2 } N } { 4 \sigma ^ { 2 } } \mathbf { h } _ { \alpha } ^ { \top } \mathbf { W } \mathbf { h } _ { \alpha } } = \sqrt { \frac { \delta ^ { 2 } N } { 4 \sigma ^ { 2 } } } = \frac { 1 } { 2 } , } \end{array}
$$

where the first inequality is from Pinsker’s inequality. Therefore, invoking the first part of Lemma 7.2 yields

$$
\begin{array} { r l } & { \mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \parallel \cdot \parallel _ { Q , \infty } ^ { 2 } ) \geq \underset { \alpha \in \mathcal { E } _ { s } } { \operatorname* { m a x } } \frac { \delta ^ { 2 } \mu _ { \alpha } d _ { \theta } } { 2 } \left( 1 - \mathsf { d } _ { T V } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) \right) } \\ & { \qquad \geq \underset { \alpha \in \mathcal { E } _ { s } } { \operatorname* { m a x } } \frac { \delta ^ { 2 } \mu _ { \alpha } d _ { \theta } } { 4 } = \underset { \alpha \in \mathcal { E } _ { s } } { \operatorname* { m a x } } \frac { \mu _ { \alpha } d _ { \theta } \sigma ^ { 2 } } { 4 N } = \frac { \mu _ { Q } d _ { \theta } \sigma ^ { 2 } } { 4 N } . } \end{array}\tag{7.71}
$$

Next, we verify the claim that $\pmb { \theta } _ { 1 } , \pmb { \theta } _ { 2 } \in \Theta _ { B }$ . That is, by setting $\begin{array} { r } { \delta = \operatorname* { m i n } \left\{ \frac { \sigma } { \sqrt { N } } , \frac { \sigma \log \left( { B } / { B _ { 0 } } \right) } { \sqrt { \mu _ { Q } d _ { \theta } } } \right\} } \end{array}$

$$
\begin{array} { r l } { \displaystyle \operatorname* { m a x } _ { \beta \in \mathcal { E } _ { s } } \frac { \vert \mathbf { a } _ { \beta } ^ { \top } \mathbf { W } _ { o } + \mathbf { a } _ { \beta } ^ { \top } \mathbf { V } _ { A ^ { \perp } } \pmb { \theta } _ { i } \vert } { \sigma } \leq \log ( B _ { 0 } ) + \displaystyle \operatorname* { m a x } _ { \beta \in \mathcal { E } _ { s } } \frac { \vert \mathbf { q } _ { \beta } ^ { \top } \pmb { \theta } _ { i } \vert } { \sigma } = \log ( B _ { 0 } ) + \displaystyle \operatorname* { m a x } _ { \beta \in \mathcal { E } _ { s } } \frac { \delta \vert \mathbf { q } _ { \beta } ^ { \top } \mathbf { W } ^ { - 1 } \mathbf { q } _ { \alpha } \vert } { \sigma \sqrt { \mu _ { \alpha } d _ { \theta } } } } & { } \\ { \leq \log ( B _ { 0 } ) + \displaystyle \operatorname* { m a x } _ { \beta \in \mathcal { E } _ { s } } \frac { \delta \vert \vert \mathbf { W } ^ { - 1 / 2 } \mathbf { q } _ { \beta } \vert \vert 2 \vert \vert \mathbf { W } ^ { - 1 / 2 } \mathbf { q } _ { \alpha } \vert \vert _ { 2 } } { \sigma \sqrt { \mu _ { \alpha } d _ { \theta } } } } & { } \\ { \leq \log ( B _ { 0 } ) + \frac { \delta \sqrt { \mu _ { Q } d _ { \theta } } } { \sigma } \leq \log ( B ) , } & { } \end{array}
$$

for $i = 1 , 2 ,$ , where the last inequality follows by setting $\begin{array} { r } { \delta \le \frac { \sigma \log \left( B / B _ { 0 } \right) } { \sqrt { \mu _ { Q } d _ { \theta } } } } \end{array}$ . Whenever $\begin{array} { r } { \frac { \sigma \log ( B / B _ { 0 } ) } { \sqrt { \mu _ { Q } d _ { \theta } } } \leq } \end{array}$ $\frac { \sigma } { \sqrt { N } }$ , we have $\mathsf { d } _ { T V } ( \mathbb { P } _ { \theta _ { 1 } } ^ { ( N ) } , \mathbb { P } _ { \theta _ { 2 } } ^ { ( N ) } ) \leq \frac { 1 } { 2 }$ . This, together with (7.71), implies that

$$
\mathcal { R } _ { N } ^ { \star } ( \Theta _ { B } , \| \cdot \| _ { Q , \infty } ^ { 2 } ) \geq \operatorname* { m i n } \left\{ \frac { \mu _ { Q } d _ { \theta } \sigma ^ { 2 } } { 4 N } , \frac { \sigma ^ { 2 } \log ^ { 2 } ( B / B _ { 0 } ) } { 4 } \right\} .
$$

Inequality (3.39b) then follows from (7.54b) and simple adjustment of the constants.

## 7.2 Proofs of MLE Error Bounds

We begin with some extra notations that will be used throughout the proof. It is convenient to reindex the observations by a single sample index k. Since $\begin{array} { r } { N = \sum _ { \alpha \in \mathcal { E } _ { s } } n _ { \alpha } } \end{array}$ , fix an arbitrary bijection $\pi : [ N ]  \{ ( \alpha , t ) : \alpha \in \mathcal { E } _ { s } , \ t \in [ n _ { \alpha } ] \} , \pi ( k ) = ( \alpha _ { k } , t _ { k } )$ . We then write $Y _ { k } : = Y _ { \alpha _ { k } } ^ { ( t _ { k } ) } , k \in$ [N]. For each $\pmb { \alpha } \in \mathcal { E } _ { s } .$ , define

$$
S _ { \alpha } : = \{ k \in [ N ] \mid \alpha _ { k } = \alpha \} , [ N ] = \bigcup _ { \alpha \in \mathcal { E } _ { s } } S _ { \alpha } , | S _ { \alpha } | = n _ { \alpha } .\tag{7.72}
$$

The particular choice of the bijection π is immaterial. For each $k \in [ N ]$ , define $\mathbf { q } _ { k } : = \mathbf { q } _ { \alpha _ { k } }$

$$
\varepsilon _ { k } : = Y _ { k } - \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { \alpha _ { k } } [ Y _ { k } ] = Y _ { k } - \Psi ^ { \prime } ( \eta _ { \alpha _ { k } } ( \pmb { \theta } ^ { \star } ) ) , \quad \boldsymbol { v } _ { k } ^ { \star } : = \mathrm { V a r } _ { \pmb { \theta } ^ { \star } } ^ { \alpha _ { k } } ( Y _ { k } ) = \Psi ^ { \prime \prime } ( \eta _ { \alpha _ { k } } ( \pmb { \theta } ^ { \star } ) ) ,\tag{7.73}
$$

where $\eta _ { \alpha } ( \pmb \theta )$ is defined in (2.19). Let the collection of the design vectors and ground-truth variances be

$$
\mathbf { x } _ { k } : = \frac { \mathbf { q } _ { k } } { \sigma } , \quad \mathbf { X } : = [ \mathbf { x } _ { 1 } , \hdots , \mathbf { x } _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d _ { \theta } } , \quad \mathbf { V } ^ { \star } : = \mathrm { d i a g } ( v _ { 1 } ^ { \star } , \hdots , v _ { N } ^ { \star } ) ,\tag{7.74}
$$

respectively. It is also convenient to introduce $\eta _ { k } ^ { \star } : = \eta _ { \alpha _ { k } } ( \pmb { \theta } ^ { \star } )$ . Moreover, consider the matrix $\mathbf { A } \in \mathbb { R } ^ { N \times N }$ with its (i, j)-th entry $\mathbf { A } _ { i j }$ being $\frac { \mathbf { x } _ { i } ^ { \top } \nabla ^ { 2 } \overset { . . . } { \ell } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } } { N }$ , and let $A _ { \mathrm { m a x } } : = \operatorname* { m a x } _ { i \in [ N ] , j \in [ N ] } | \mathbf { A } _ { i j } |$

## 7.2.1 Proof of Theorem 4.1

The core of the proof is to establish (4.41a). This reduces to understanding the components of ∆, for which the high-level road-map is partly inspired by [63, Theorem 1] and [44, Theorem 1], where the authors show that a sample complexity on the order of $d _ { \theta } ^ { 3 / 2 }$ sufices to isolate the second-order bias efect for the canonical MLE with sub-Gaussian design vectors [63, Section 2.2] and for the corresponding Z-estimator with deterministic design that admits a solution in the interior of a compact set [44, Section 2.1], respectively. However, these two regularity conditions cannot hold in our context, and we construct a compact and convex set on which the residual map is a self-map and obtain a finite MLE from its fixed point (Lemma 7.8). Within this set, the Hessian of the negative log-likelihood function tends to be well-behaved, resulting in ℓ(θ)

exhibiting local strong convexity. For this step, Chen [16] and Yang et al.’s [76; 77] strategy is the most analogous to ours among existing literature. They consider running preconditioned gradient descent, starting from the ground truth

$$
\pmb { \theta } _ { t + 1 } = \pmb { \theta } _ { t } - \eta \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \nabla \ell ( \pmb { \theta } _ { t } ) , \quad \pmb { \theta } _ { 0 } = \pmb { \theta } ^ { \star } .
$$

Performing this recursively, it yields [77]

$$
\Delta ^ { t + 1 } = - [ 1 - ( 1 - \eta ) ^ { t + 1 } ] \underbrace { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \nabla \ell ( \theta ^ { \star } ) } _ { - \Delta _ { l i n } } - \eta \sum _ { j = 0 } ^ { t } ( 1 - \eta ) ^ { t - j } \underbrace { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } [ \bar { \mathbf { H } } ^ { j } - \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ] \Delta ^ { j } } _ { \mathrm { Q u a d r a t i c ~ a n d ~ h i g h e r - o r d e r ~ r e s i d u a l s } } ,\tag{7.75}
$$

where $\begin{array} { r } { \bar { \mathbf { H } } ^ { t } = \int _ { 0 } ^ { 1 } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } + s \Delta _ { t } ) \mathrm { d } } \end{array}$ s and $\Delta ^ { t } : = \pmb { \theta } ^ { t } - \pmb { \theta } ^ { \star }$ . The localization step is accomplished by proving by induction that $\theta _ { t }$ will remain close to $\pmb { \theta } ^ { \star }$ , and this procedure will converge to the MLE solution. Yet the induction steps in [76, Appendix B] heavily rely on the specific graphical structure of the pairwise ranking model, which in general does not hold under our setup. Taking (4.41a) as given, we can establish local strong convexity within a small region around the ground truth, from which we prove (4.41b). This step utilizes some corollaries from Bach et al. [52; 4].

## 7.2.1.1 Taylor Expansion of High-Order Residuals

The first step of the proof is to rephrase the “main order + residual” structure depicted in (7.75) as a nonlinear equality system that encodes the first-order optimality condition, and write out the second-order bias explicitly. To ease the notation, let the nonlinear remainder around $\Psi ^ { \prime } ( \eta _ { j } ^ { \star } )$ be defined as

$$
r _ { j } ( \Delta ) : = \Psi ^ { \prime } ( \eta _ { j } ^ { \star } + \mathbf { x } _ { j } ^ { \top } \Delta ) - \Psi ^ { \prime } ( \eta _ { j } ^ { \star } ) - \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } ^ { \top } \Delta , \forall j \in [ N ] ,\tag{7.76}
$$

where $\Delta = \theta - \theta ^ { \star }$ and $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ . By construction, (7.75) can be viewed as a Taylor expansion of the likelihood score

$$
\begin{array} { l } { \displaystyle \nabla \ell ( \pmb { \theta } ^ { \star } + \Delta ) = \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \left( \Delta - \Delta _ { l i n } \right) + \int _ { 0 } ^ { 1 } \left[ \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } + s \Delta ) - \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \right] \Delta \mathrm { d } s } \\ { \displaystyle = \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \left( \Delta - \Delta _ { l i n } \right) + \frac { 1 } { N } \sum _ { j = 1 } ^ { N } r _ { j } ( \Delta ) \mathbf { x } _ { j } . } \end{array}\tag{7.77}
$$

We are particularly interested in the structure of the second term on the r.h.s. of (7.77), the following elementary expansion separates the leading quadratic component of $r _ { j } ( \Delta )$

Lemma 7.5 Let $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ , and $\Delta = \theta - \theta ^ { \star }$ . For any $j \in [ N ]$ , the nonlinear remainder defined in (7.76) can be equivalently expressed as

$$
r _ { j } ( \Delta ) = \frac { 1 } { 2 } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) | \mathbf { x } _ { j } ^ { \top } \Delta | ^ { 2 } + \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } \Psi ^ { ( 4 ) } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } \Delta ) ( \mathbf { x } _ { j } ^ { \top } \Delta ) ^ { 3 } \mathrm { d } s .\tag{7.78}
$$

Proof. This follows directly from Taylor expansion of a three-times continuously diferentiable function $g : \mathbb { R }  \mathbb { R }$ to the third term with integral remainder, i.e.,

$$
g ( a + h ) = g ( a ) + g ^ { \prime } ( a ) h + \frac { 1 } { 2 } g ^ { \prime \prime } ( a ) h ^ { 2 } + \int _ { 0 } ^ { 1 } \frac { h ^ { 3 } } { 2 } ( 1 - s ) ^ { 2 } g ^ { ( 3 ) } ( a + s h ) \mathrm { d } s ,
$$

for $g ( \cdot ) = \Psi ^ { \prime } ( \cdot )$ with $a = \eta _ { j } ^ { \star }$ and $h = \mathbf { x } _ { j } ^ { \top } \Delta$

Lemma 7.5 also explains the particular form of the deterministic bias correction in (4.40b). Indeed, by the definition of $\Delta _ { l i n }$ and A, we have

$$
\mathbb { E } _ { \theta ^ { \star } } ^ { ( N ) } \left[ \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left[ \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) | \mathbf { x } _ { j } ^ { \top } \Delta _ { l i n } | ^ { 2 } \mathbf { x } _ { j } \right] \right] = \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left[ \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { A } _ { j j } \mathbf { x } _ { j } \right]\tag{7.79}
$$

Inspired by similar results under stronger regularity conditions [44, Section 2.2] and the asymptotic rate proved by Portnoy [54, Section 3], we infer that $\Delta - \Delta _ { l i n } - \Delta _ { b i a s }$ is suficiently small, when measured in the $\| \cdot \| _ { Q , \infty }$ norm with high probability, provided that θ satisfies the first-order optimality condition (see Section 7.2.1.2). The following lemma converts the likelihood score equation into a fixed-point system for the modified residual vector $\Delta - \Delta _ { l i n } - \Delta _ { b i a s }$

Lemma 7.6 (Residual Expansion with Bias) For any $\Delta \in \mathbb { R } ^ { d _ { \theta } }$ , and $\Delta _ { l i n } , \Delta _ { b i a s }$ defined as in (4.40a), (4.40b), respectively, let $\pmb { \theta } : = \Delta + \pmb { \theta } ^ { \star }$ , and h $: = \Delta - \Delta _ { l i n } - \Delta _ { b i a s }$ . Then θ satisfies the first-order optimality condition $\nabla \ell ( \pmb \theta ) = \mathbf 0$ if h satisfies

$$
{ \bf h } = - \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } { \bf x } _ { j } r _ { j } ( \Delta _ { l i n } + \Delta _ { b i a s } + { \bf h } ) - \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } { \bf A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) { \bf x } _ { j } \right] .\tag{7.80}
$$

Proof. See Appendix A.5.

We next recall the pseudo self-concordance property of the cumulant function $\Psi ( \cdot )$ , introduced by Bach et al. [4; 52], which will be utilized to control both the higher-order remainder and the variation of the Hessian of the negative log-likelihood function.

Definition 7.1 Let $g : [ 0 , 1 ] \to$ R be a three-times continuously diferentiable function. If $g ^ { \prime \prime } ( 0 ) > 0$ , and there exists some $S \in [ 0 , \infty )$ such that

$$
| g ^ { ( 3 ) } ( t ) | \leq S g ^ { \prime \prime } ( t ) , \forall t \in [ 0 , 1 ] ,
$$

then g is said to be pseudo self-concordant with parameter S (see $\mathit { 1 4 ; 5 2 | }$

By the chain rule and the fact that $| \Psi ^ { ( 3 ) } ( \eta ) | \le \Psi ^ { \prime \prime } ( \eta )$ , it is straightforward to verify that for any $t \in [ 0 , 1 ] , \Psi ( \eta _ { j } ^ { \star } + t \mathbf { x } _ { j } ^ { \top } \Delta )$ , as a function of t, is pseudo self-concordant with parameter $| \mathbf { x } _ { j } ^ { \top } \Delta |$ , i.e., $\begin{array} { r } { \left| \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } + t \mathbf { x } _ { j } ^ { \top } \Delta ) \cdot ( \mathbf { x } _ { j } ^ { \top } \Delta ) ^ { 3 } \right| \leq | \mathbf { x } _ { j } ^ { \top } \Delta | \cdot \left( \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } + t \mathbf { x } _ { j } ^ { \top } \Delta ) | \mathbf { x } _ { j } ^ { \top } \Delta | ^ { 2 } \right) , \forall t \in [ 0 , 1 ] , \forall j \in [ N ] . } \end{array}$ (7.81)

Analogously, define the nonlinear remainder around $\Psi ( \eta _ { j } ^ { \star } )$ by

$$
\delta _ { j } ( \Delta ) : = \Psi ( \eta _ { j } ^ { \star } + \mathbf { x } _ { j } ^ { \top } \Delta ) - \Psi ( \eta _ { j } ^ { \star } ) - \Psi ^ { \prime } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } ^ { \top } \Delta , \ : \forall j \in [ N ] .\tag{7.82}
$$

Then, recursively integrating both sides of (7.81) yields the following result.

Lemma 7.7 ([4, Lemma 1]) Let $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } } , \Delta = \pmb { \theta } - \pmb { \theta } ^ { \star }$ , and $t \in [ 0 , 1 ]$ . Then for any $j \in [ N ] ^ { 1 2 }$

$$
e ^ { - | \mathbf { x } _ { j } ^ { \top } \Delta | t } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) \leq \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } + t \mathbf { x } _ { j } ^ { \top } \Delta ) \leq e ^ { | \mathbf { x } _ { j } ^ { \top } \Delta | t } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) ,\tag{7.83a}
$$

$$
\frac { 1 - e ^ { - | \mathbf { x } _ { j } ^ { \top } \Delta | t } } { | \mathbf { x } _ { j } ^ { \top } \Delta | } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) \leq \int _ { 0 } ^ { t } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } \Delta ) \mathrm { d } s \leq \frac { e ^ { | \mathbf { x } _ { j } ^ { \top } \Delta | t } - 1 } { | \mathbf { x } _ { j } ^ { \top } \Delta | } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) ,\tag{7.83b}
$$

$$
\begin{array} { r } { ( e ^ { - | \mathbf x _ { j } ^ { \top } \Delta | t } + | \mathbf x _ { j } ^ { \top } \Delta | t - 1 ) \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) \leq \delta _ { j } ( t \Delta ) \leq ( e ^ { | \mathbf x _ { j } ^ { \top } \Delta | t } - | \mathbf x _ { j } ^ { \top } \Delta | t - 1 ) \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) . } \end{array}\tag{7.83c}
$$

When $\mathbf { x } _ { j } ^ { \top } \Delta = 0$ , all three inequalities hold trivially.

Bach’s convex localization strategy [52, Proposition B.4] can be generalized to the unconstrained case. However, inequality (7.83c) is not sharp enough to derive sample complexity of order $d _ { \theta } ^ { 3 / 2 }$ Furthermore, Lemma 7.7 indicates that the core of the proof lies in proper control of the term m $\mathrm { a x } _ { m \in [ N ] } | \mathbf x _ { m } ^ { \top } ( \hat { \pmb \theta } - { \pmb \theta } ^ { \star } )$ |. This will be the main focus of the next section.

## 7.2.1.2 Existence of Fixed Point

The existence of canonical MLE can be derived by existence of a fixed point of the nonlinear equality system (7.80), underpinned by Brouwer’s fixed point theorem. Specifically, Theorem 4.1 builds on the following lemma.

Lemma 7.8 (Self-mapping Property) When $N \gtrsim B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d$ , there exist some positive constants $r _ { \infty }$ and $r _ { H }$ , where

$$
r _ { \infty } \asymp \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + B A _ { \operatorname* { m a x } } \log d ,\tag{7.84}
$$

$$
r _ { H } \asymp \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } + \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } B \log ^ { 2 } d } { N } + A _ { \operatorname* { m a x } } \log ( d ) \sqrt { \frac { d _ { \theta } B \log d } { N } } } ,\tag{7.85}
$$

such that with probability at least $1 - d ^ { - c }$ 2

$$
\Phi ( \mathbf { h } ) : = - \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left[ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } r _ { j } ( \Delta _ { l i n } + \Delta _ { b i a s } + \mathbf { h } ) - \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } \right]
$$

is a self-mapping on the non-empty, closed, convex and compact set

$$
\mathcal { N } ( r _ { H } , r _ { \infty } ) : =  \mathbf { h } \in \mathbb { R } ^ { d _ { \theta } } | \| \mathbf { h } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \leq r _ { H } , \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { h } | \leq r _ { \infty }  ,\tag{7.86}
$$

that is,

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \Big \{ \forall \mathbf { h } \in \mathcal { N } ( r _ { H } , r _ { \infty } ) , \Phi ( \mathbf { h } ) \in \mathcal { N } ( r _ { H } , r _ { \infty } ) \Big \} \geq 1 - d ^ { - c } .\tag{7.87}
$$

The proof is deferred to Appendix A.6.

Proof of Theorem 4.1. With Lemmas 7.6 and 7.8, we are ready to prove Theorem 4.1. First, observe that $\Phi : \mathcal { N } ( r _ { H } , r _ { \infty } )  \mathcal { N } ( r _ { H } , r _ { \infty } )$ is continuous because each $r _ { j } ( \cdot )$ is continuous, and $\Phi ( \cdot )$ is a finite linear combination of $r _ { j } ( \Delta _ { l i n } + \Delta _ { b i a s } + ( \cdot ) )$ . Since $\mathcal { N } ( r _ { H } , r _ { \infty } )$ is nonempty, compact and convex, the self-mapping property enables us to apply Brouwer’s fixed-point theorem which guarantees existence of a fixed point $\hat { \mathbf { h } } \in \mathcal { N } ( r _ { H } , r _ { \infty } )$ such that

$$
\hat { \mathbf { h } } = \Phi ( \hat { \mathbf { h } } ) .\tag{7.88}
$$

Let $\hat { \Delta } : = \Delta _ { l i n } + \Delta _ { b i a s } + \hat { \mathbf { h } }$ and accordingly $\hat { \pmb \theta } : = { \pmb \theta } ^ { \star } + \hat { \Delta }$ . By Lemma 7.6, the fixed point identity (7.88) implies $\nabla \ell (  { \hat { \mathbf { \theta } } } ) = \mathbf { 0 }$ . Moreover, since $\Psi ^ { \prime \prime } ( \eta ) > 0$ for any $\eta \in \mathbb { R }$ and $\textbf { W } \succ \textbf { 0 }$ by assumption, the negative log-likelihood function $\ell ( \pmb \theta )$ is strictly convex, which means that $\hat { \pmb { \theta } }$ is a unique global minimizer. Therefore the fixed point h<sup>ˆ</sup> in (7.88) is unique. Finally, since $\hat { \mathbf { h } } \in \mathcal { N } ( r _ { H } , r _ { \infty } )$ , with probability at least $1 - d ^ { - c }$ , then

$$
\operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \hat { \boldsymbol { \Delta } } | \leq \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } { \Delta } _ { l i n } | + \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } { \Delta } _ { b i a s } | + r _ { \infty }
$$

$$
\begin{array} { r l r } {  { \lesssim \sqrt { A _ { m a x } \log d } + A _ { m a x } \log d + \frac { 1 } { 2 } A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + r _ { \infty } } } & { ( \mathrm { b y ~ L e m m a s ~ A . 4 ~ a n d ~ A . 2 } ) } \\ & { \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + B A _ { \operatorname* { m a x } } \log d } & { ( \mathrm { b y ~ L e m m a ~ 7 . 8 } ) } \\ & { \lesssim \sqrt { \frac { B \mu _ { Q } d _ { \theta } \log d } { N } } + \frac { B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } , } \end{array}
$$

where the last inequality is from Lemma A.1, for $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta }$ log d. By noticing $\| \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \| _ { Q , \infty } =$ $\begin{array} { r } { \| \hat { \boldsymbol { \Delta } } \| _ { Q , \infty } = \sigma \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \hat { \boldsymbol { \Delta } } | } \end{array}$ , we obtain (4.41a).

## 7.2.1.3 Local Strong Convexity

Inequality (4.41b) is proved by the strong convexity of $\ell ( \pmb \theta )$ over the set

$$
\mathfrak { M } _ { \infty , 1 / 2 } : = \left\{ \theta \in \mathbb { R } ^ { d _ { \theta } } \Big | \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } ( \theta - \theta ^ { \star } ) | \leq \frac { 1 } { 2 } \right\} .
$$

To prove this assertion, let $\Delta = \theta - \theta ^ { \star }$ for any $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ and by the first inequality in (7.83a) and the structure of $\nabla ^ { 2 } \ell ( \pmb \theta )$ in (2.20b)

$$
\begin{array} { r l } {  { \Delta ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } + t \Delta ) \Delta = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } + t \mathbf { x } _ { j } ^ { \top } \Delta ) | \mathbf { x } _ { j } ^ { \top } \Delta | ^ { 2 } \geq \frac { 1 } { N } \sum _ { j = 1 } ^ { N } e ^ { - | \mathbf { x } _ { j } ^ { \top } \Delta | t } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) | \mathbf { x } _ { j } ^ { \top } \Delta | ^ { 2 } } } \\ & { \geq \exp ( - t \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta | ) \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } ) | \mathbf { x } _ { j } ^ { \top } \Delta | ^ { 2 } } \\ & { = \exp ( - t \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta | ) \| \Delta \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } ^ { 2 } , \forall t \in [ 0 , 1 ] . } \end{array}
$$

By integrating both sides of the above inequality twice w.r.t. t, we obtain (see (7.83c) or [4, Proposition 1] [52, Proposition B.3])<sup>13</sup>

$$
\begin{array} { r } { \ell ( \pmb { \theta } ) - \ell ( \pmb { \theta } ^ { \star } ) - \langle \nabla \ell ( \pmb { \theta } ^ { \star } ) , \Delta \rangle \geq \frac { e ^ { - \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta | } + \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta | - 1 } { ( \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta | ) ^ { 2 } } \| \Delta \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } ^ { 2 } . } \end{array}\tag{7.89}
$$

Using the basic inequality $\begin{array} { r } { \frac { x ^ { 2 } } { 4 } \leq e ^ { - x } + x - 1 } \end{array}$ for $\begin{array} { r } { 0 \leq x \leq \frac { 1 } { 2 } } \end{array}$ , we deduce from (7.89) that

$$
\ell ( \pmb \theta ) - \ell ( \pmb \theta ^ { \star } ) - \langle \nabla \ell ( \pmb \theta ^ { \star } ) , \Delta \rangle \geq \frac { 1 } { 4 } \| \Delta \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } ^ { 2 } , \forall \pmb \theta \in \mathfrak { M } _ { \infty , 1 / 2 } ,\tag{7.90}
$$

where we prove the ${ \frac { 1 } { 4 } } \cdot$ -local strong convexity at $\pmb { \theta } ^ { \star }$ of $\ell ( \pmb \theta )$ . Moreover, by Lemma 4.1 and the assumption that $\nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) \succ \mathbf 0$ 2

$$
\| \theta - \theta ^ { \star } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \leq 4 \| \nabla \ell ( \theta ^ { \star } ) \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } , \forall \theta \in \mathfrak { M } _ { \infty , 1 / 2 } , \ell ( \theta ) \leq \ell ( \theta ^ { \star } ) .\tag{7.91}
$$

Finally, under the assumptions of Lemma 7.8, by (7.84), the fact that $\ell ( { \hat { \pmb \theta } } ) ~ \leq ~ \ell ( { \pmb \theta } ^ { \star } )$ , and Lemma $\mathrm { A . 4 } .$ we conclude that with probability at least $1 - d ^ { - c } , \hat { \pmb { \theta } } \in \mathfrak { M } _ { \infty , 1 / 2 }$ , and

$$
\| \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } \| _ { \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) } \lesssim \left( \sqrt { \frac { d _ { \theta } } { N } } + \sqrt { \frac { B d _ { \theta } \log d } { N } } \right) \lesssim \sqrt { \frac { B d _ { \theta } \log d } { N } } .
$$

This concludes the proof of (4.41b).

## 7.2.2 Proof of Theorem 4.2

The proof relies on the proof of Lemma 7.8, and we will use the notation introduced there. Denote the MLE solution derived in Theorem 4.1 and the fixed point deduced in (7.88) by $\hat { \pmb { \theta } }$ and $\hat { \mathbf { h } } ,$ respectively. Let $\hat { \Delta } : = \hat { \pmb { \theta } } - \pmb { \theta } ^ { \star } = \Delta _ { l i n } + \Delta _ { b i a s } +$ h<sup>ˆ</sup> and denote $\hat { \mathbf { e } } _ { e r r } : = \Delta _ { b i a s } + \hat { \mathbf { h } }$ . By the derivation in Section 7.2.1.2, h<sup>ˆ</sup> satisfies the equality system (7.80). This, together with Lemma 7.5 and identity (7.79) gives

$$
\begin{array} { r l } & { \hat { \Delta } = \Delta _ { l i n } + \Delta _ { b i a s } + \Phi ( \hat { \mathbf { h } } ) } \\ & { \quad = \Delta _ { l i n } + \Delta _ { b i a s } - \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left[ Q _ { 2 } ( \Delta _ { l i n } + \hat { \mathbf { e } } _ { e r r } ) + Q _ { 3 } ( \Delta _ { l i n } + \hat { \mathbf { e } } _ { e r r } ) - \mathbb { E } _ { \theta ^ { \star } } ^ { ( N ) } [ Q _ { 2 } ( \Delta _ { l i n } ) ] \right] } \\ & { \quad = \Delta _ { l i n } + \Delta _ { b i a s } + \Delta _ { q u a d } + \Delta _ { \ge 3 } , } \end{array}\tag{7.92}
$$

where

$$
\Delta _ { q u a d } : = - \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \left[ Q _ { 2 } ( \Delta _ { l i n } ) - \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ Q _ { 2 } ( \Delta _ { l i n } ) \right] \right] ,\tag{7.93a}
$$

$$
\Delta _ { \geq 3 } : = - \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \left[ Q _ { 2 } ( \Delta _ { l i n } + \hat { \mathbf { e } } _ { e r r } ) - Q _ { 2 } ( \Delta _ { l i n } ) + Q _ { 3 } ( \Delta _ { l i n } + \hat { \mathbf { e } } _ { e r r } ) \right] .\tag{7.93b}
$$

Proof of (4.43). It sufices to derive the upper bounds on $\| \Delta _ { q u a d } \| _ { Q , \infty }$ and $\| \Delta _ { \geq 3 } \| _ { Q , \infty }$ . From STEP 2 in the proof of Lemma 7.8 and the assumption $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d .$

$$
\begin{array} { r l } { \frac { \operatorname* { m a x } _ { i } } { \operatorname* { m a x } _ { i } \left| \mathbf { Y } _ { i } \right| } | \mathbf { x } _ { \mathrm { m } } ^ { \top } \partial _ { \mathbf { S } } | \leq \frac { \operatorname* { m a x } _ { i } } { \operatorname* { m a x } _ { i } \left| \mathbf { X } _ { i } \right| } \left| \mathbf { X } _ { i } ^ { \top } \nabla ^ { \top } \hat { \mathcal { H } } ( \theta ) ^ { - 1 } \left[ Q _ { 2 } ( \Delta _ { \mathbf { H } + 1 } + \hat { \mathcal { \ell } } _ { \mathrm { a } \times \mathbf { r } } ) - Q _ { 2 } ( \Delta _ { \mathbf { H } + 1 } ) \right] \right| } & { \frac { \operatorname* { m a x } _ { i } } { \operatorname* { m a x } _ { i } \left| \mathbf { X } _ { i } ^ { \top } \right| } \left| \mathbf { X } _ { i } ^ { \top } \nabla ^ { \top } \hat { \mathcal { H } } ( \theta ^ { * } ) ^ { - 1 } Q _ { 4 } ( \Delta _ { \mathbf { H } + 1 } ~ \hat { \mathcal { \ell } } _ { \mathrm { a } \times \mathbf { r } } ) \right| } \\ &  \leq \sqrt { 4 } \operatorname* { m a x } _ { i } \left| \hat { \mathcal { \ell } } _ { \mathrm { a } \times \mathbf { r } } \right| \left| \mathbf { r } _ { i } \right| \otimes \hat { \mathcal { \ell } } _ { i } \left| \nabla ^ { \top } \hat { \mathcal { H } } ( \theta ) \right| \ \ \sqrt { Q } \Delta _ { \mathbf { H } - 1 } \log \ d + \frac { \operatorname* { m a x } _ { i } \times \times \times \times \times \times \times \times \times \times \times \times \times \times \times \times } \\ & { + \left( \sqrt { 4 } \operatorname* { m a x } _ { i } \log \hat { \mathcal { \ell } } _ { + } 4 \operatorname* { m a x } _ { i } \log \hat { \mathcal { \ell } } _ { + } \right) \left( N \Delta _ { \mathbf { H } - 1 } \log \phi + 4 \operatorname* { m a x } _ { i } \times \log \sqrt { \phi } _ { i } \right) } \\ \end{array}
$$

where the second inequality follows by the upper bounds on $I _ { 2 }$ and $I _ { 3 }$ derived in Appendix A.6.3. On the other hand, by (A.125), ma $\begin{array} { r } { \mathrm { i X } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta _ { q u a d } | \lesssim B A _ { \mathrm { m a x } } } \end{array}$ log d. Then, by combining the above two upper bounds and substituting the upper bound on $A _ { \mathrm { m a x } } .$ , we obtain

$$
\begin{array} { r } { \| \hat { \boldsymbol { \Delta } } - \boldsymbol { \Delta } _ { l i n } - \boldsymbol { \Delta } _ { b i a s } \| _ { Q , \infty } \leq \sigma \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \boldsymbol { \Delta } _ { \geq 3 } | + \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \boldsymbol { \Delta } _ { q u a d } | \right) } \\ { \lesssim \frac { \sigma B ^ { 2 } \mu _ { Q } d _ { \theta } \log d } { N } \left[ \sqrt { \frac { B \mu _ { Q } d _ { \theta } ^ { 2 } \log d } { N } } + \frac { \sqrt { B } \mu _ { Q } d _ { \theta } ^ { 2 } } { N } + 1 \right] . } \end{array}
$$

Inequality (4.43) is then directly recovered when $N \gtrsim B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d .$

Proof of (4.44). By (7.92), it sufices to find upper bounds on $\| \Delta _ { \geq 3 } \| _ { 2 }$ and $\| \Delta _ { q u a d } \| _ { 2 }$ . While the former can be derived via $\| \Delta _ { \geq 3 } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) }$ , i.e., (A.119) and (A.121), the latter requires some nontrivial tools for bounding the supremum of a collection of sub-Gaussian quadratic forms. Here,

we utilize the Hanson-Wright inequality in vector space (which can be viewed as a corollary of the general theory developed by Adamczak et al. [1], restated in Lemma B.3). We refer interested readers to the paper and the references therein.

Bound for $\| \Delta _ { q u a d } \| _ { 2 }$ . We first recast $\Delta _ { q u a d }$ in the form depicted in Lemma B.3. From the proof of (A.113a) in Lemma $\mathrm { A . 6 }$ , we can verify $\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } [ \Delta _ { q u a d } ] = \mathbf { 0 }$ , and $\Delta _ { q u a d }$ can be written as follows

$$
\begin{array} { l } { { \displaystyle \Delta _ { q u a d } = - \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } { \bf x } _ { j } \left[ ( { \bf e } _ { j } ^ { \top } { \bf P } \pmb { \xi } ) ^ { 2 } - { \bf P } _ { j j } \right] } } \\ { { \displaystyle \qquad = \sum _ { ( k , l ) \in [ N ] ^ { 2 } } { \bf a } ^ { ( k l ) } \left( \xi _ { k } \xi _ { l } - \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \xi _ { k } \xi _ { l } \right] \right) , } } \end{array}
$$

where $\xi _ { i } = \varepsilon _ { i } / \sqrt { v _ { i } ^ { \star } } , i \in [ N ]$ are independent, zero mean, isotropic, and $\sqrt { B } \mathrm { - s u b \mathrm { - } G a u s s i a n }$ , and $\pmb { \xi } : = [ \xi _ { 1 } , \ldots , \xi _ { N } ] ^ { \top }$ , and $\begin{array} { r } { \mathbf { a } ^ { ( k l ) } : = - \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \frac { \Psi ^ { ( 3 ) } ( \boldsymbol \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \mathbf { P } _ { j k } \mathbf { P } _ { j l } \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } \in \mathbb { R } ^ { d _ { \theta } } } \end{array}$ . In what follows, j we derive bounds for the terms $\begin{array} { r } { T ^ { 2 } : = \sum _ { ( k , l ) \in [ N ] ^ { 2 } } \| \mathbf { a } ^ { ( k l ) } \| _ { 2 } ^ { 2 } } \end{array}$ and the $U , V$ defined in Lemma B.3.

For $T ^ { 2 }$ , we use the fact that $\mathbf { P } \succeq \mathbf { 0 }$ and $\mathbf { P } ^ { 2 } = \mathbf { P }$

$$
\begin{array} { l } { { \displaystyle T ^ { 2 } = \frac { 1 } { 4 N ^ { 2 } } \sum _ { ( m , n ) \in [ N ] ^ { 2 } } \frac { \Psi ^ { ( 3 ) } ( \eta _ { m } ^ { \star } ) } { v _ { m } ^ { \star } } \frac { \Psi ^ { ( 3 ) } ( \eta _ { n } ^ { \star } ) } { v _ { n } ^ { \star } } \left( \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 2 } \mathbf { x } _ { n } \right) \sum _ { k \in [ N ] } \mathbf { P } _ { m k } \mathbf { P } _ { k n } \sum _ { l \in [ N ] } \mathbf { P } _ { m l } \mathbf { P } _ { l n } } } \\ { { \displaystyle \quad = \frac { 1 } { 4 N ^ { 2 } } \left. \mathbf { P } \circ \mathbf { P } , \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { m } ^ { \star } ) } { v _ { m } ^ { \star } } \frac { \Psi ^ { ( 3 ) } ( \eta _ { n } ^ { \star } ) } { v _ { n } ^ { \star } } \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 2 } \mathbf { x } _ { n } \right) _ { m , n \leq N } \right. } } \\ { { \displaystyle \qquad \leq \frac { 1 } { 4 N ^ { 2 } } \| \mathbf { P } \circ \mathbf { P } \| \sum _ { j = 1 } ^ { N } \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \right) ^ { 2 } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } \leq \frac { 1 } { 4 N ^ { 2 } } \| \mathbf { P } \circ \mathbf { P } \| \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } , } } \ \end{array}
$$

where the second last inequality follows from the fact that both matrices in the inner product are positive semidefinite. Then, by the Gershgorin circle theorem and Lemma A.1

$$
\begin{array} { r l } & { \| \mathbf { P } \circ \mathbf { P } \| \leq \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \sum _ { j = 1 } ^ { N } \left| [ \mathbf { P } \circ \mathbf { P } ] _ { i j } \right| = \displaystyle \operatorname* { m a x } _ { i \in [ N ] } \sum _ { j = 1 } ^ { N } v _ { i } ^ { \star } v _ { j } ^ { \star } \mathbf { A } _ { i j } ^ { 2 } = \displaystyle \operatorname* { m a x } _ { i \in [ N ] } v _ { i } ^ { \star } \mathbf { A } _ { i i } = \operatorname* { m a x } _ { i \in [ N ] } \mathbf { P } _ { i i } } \\ & { \qquad \leq \displaystyle \frac { A _ { \operatorname* { m a x } } } { 4 } . } \end{array}
$$

On the other hand, by Proposition 2.2, $\begin{array} { r } { \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \mathbf { x } _ { j } ^ { \top } \preceq 4 B \mathbf { X } ^ { \top } \mathbf { V } ^ { \star } \mathbf { X } = 4 B N \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } \end{array}$ . Thus

$$
\begin{array} { r l } & { \displaystyle \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } = \left. \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } , \displaystyle \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \mathbf { x } _ { j } ^ { \top } \right. } \\ & { \qquad \leq \langle \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } , 4 B N \nabla ^ { 2 } \ell ( \theta ^ { \star } ) \rangle = 4 B N \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) , } \end{array}
$$

from which we have

$$
T \leq \frac { 1 } { 2 } \sqrt { \frac { B A _ { \operatorname* { m a x } } \mathrm { t r } ( \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } ) } { N } } .\tag{7.94}
$$

Next, we derive an upper bound for the second term in U. Observe

$$
U _ { 2 } : = \operatorname* { s u p } _ { \| \mathbf { M } \| _ { F } \leq 1 } \left\| \sum _ { ( k , l ) \in [ N ] ^ { 2 } } \mathbf { a } ^ { ( k l ) } [ \mathbf { M } ] _ { k l } \right\| _ { 2 } = \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \operatorname* { s u p } _ { \| \mathbf { M } \| _ { F } \leq 1 } \left| \sum _ { ( k , l ) \in [ N ] ^ { 2 } } \mathbf { v } ^ { \top } \mathbf { a } ^ { ( k l ) } [ \mathbf { M } ] _ { k l } \right|
$$

$$
\mathbf { \Psi } = \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \left\| \left( \mathbf { v } ^ { \top } \mathbf { a } ^ { ( k l ) } \right) \right\| _ { F } = \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \frac { 1 } { 2 N } \| \mathbf { P } \mathbf { D } _ { v } \mathbf { P } \| _ { F } ,
$$

where $\mathbf { D } _ { v } = \mathrm { d i a g } ( \mathbf { d } _ { v } )$ , and $\begin{array} { r } { \mathbf { d } _ { v } : = \left( ( \frac { \Psi ^ { ( 3 ) } ( \eta _ { i } ^ { \star } ) } { v _ { i } ^ { \star } } ) \mathbf { v } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { i } \right) _ { i \leq N } . } \end{array}$ . Thus

$$
\begin{array} { l } { ( { \boldsymbol { U } } _ { 2 } ) ^ { 2 } = \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \frac { 1 } { 4 N ^ { 2 } } \mathrm { t r } ( \mathbf { D } _ { v } \mathbf { P } \mathbf { D } _ { v } \mathbf { P } ) = \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \frac { 1 } { 4 N ^ { 2 } } \mathbf { d } _ { v } ^ { \top } ( \mathbf { P } \circ \mathbf { P } ) \mathbf { d } _ { v } } \\ { \leq \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \frac { 1 } { 4 N ^ { 2 } } \| \mathbf { P } \circ \mathbf { P } \| \| \mathbf { d } _ { v } \| _ { 2 } ^ { 2 } \leq \frac { A _ { \operatorname* { m a x } } } { 1 6 N ^ { 2 } } \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \| \mathbf { d } _ { v } \| _ { 2 } ^ { 2 } } \\ { \leq \displaystyle \frac { A _ { \operatorname* { m a x } } } { 1 6 N ^ { 2 } } \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \sum _ { j = 1 } ^ { N } \Big ( \mathbf { v } ^ { \top } \nabla ^ { 2 } \boldsymbol { \ell } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } \Big ) ^ { 2 } \leq \frac { B A _ { \operatorname* { m a x } } } { 4 N } \displaystyle \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } \leq 1 } \mathbf { v } ^ { \top } \nabla ^ { 2 } \boldsymbol { \ell } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { v } } \\ { \leq \frac { B A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \boldsymbol { \ell } ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { A N } . } \end{array}\tag{7.95}
$$

We then find an upper bound on V . Let $\| \mathbf { u } \| _ { 2 } , \| \mathbf { v } \| _ { 2 } \leq 1$ and y be a unit vector. Then

$$
\begin{array} { r l } & { \bigg \vert \displaystyle \sum _ { ( k , l ) \in [ N ] ^ { 2 } } \mathbf { a } ^ { ( k l ) } \mathbf { u } _ { k } \mathbf { v } _ { l } \bigg \vert \bigg \vert _ { 2 } = \underset { \| \mathbf { y } \| _ { 2 } = 1 } { \operatorname* { s u p } } \frac { 1 } { 2 N } \bigg \vert \displaystyle \sum _ { j = 1 } ^ { N } \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \left( \mathbf { y } ^ { \top } \nabla ^ { 2 } \boldsymbol \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } \right) \left( \mathbf { e } _ { j } ^ { \top } \mathbf { P u } \right) \left( \mathbf { e } _ { j } ^ { \top } \mathbf { P v } \right) \bigg \vert } \\ & { \qquad \leq \underset { \| \mathbf { y } \| _ { 2 } = 1 } { \operatorname* { s u p } } \frac { 1 } { 2 N } \underset { m \in [ N ] } { \operatorname* { m a x } } \bigg \vert \mathbf { y } ^ { \top } \nabla ^ { 2 } \boldsymbol \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { m } \bigg \vert \displaystyle \sum _ { j = 1 } ^ { N } \bigg \vert \mathbf { e } _ { j } ^ { \top } \mathbf { P u } \bigg \vert \cdot \Big \vert \mathbf { e } _ { j } ^ { \top } \mathbf { P v } \bigg \vert } \\ & { \qquad \leq \frac { \operatorname* { m a x } _ { m \in [ N ] } \| \nabla ^ { 2 } \boldsymbol \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { m } \| _ { 2 } } { 2 N } \| \mathbf { P u } \| _ { 2 } \| \mathbf { P v } \| _ { 2 } , } \end{array}
$$

from which we have

$$
\begin{array} { r l } & { V : = \underset { \| \mathbf { u } \| _ { 2 } , \| \mathbf { v } \| _ { 2 } \leq 1 } { \operatorname* { s u p } } \left\| \underset { ( k , l ) \in [ N ] ^ { 2 } } { \sum } \mathbf { a } ^ { ( k l ) } \mathbf { u } _ { k } \mathbf { v } _ { l } \right\| _ { 2 } } \\ & { \leq \frac { \operatorname* { m a x } _ { m \in [ N ] } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { m } \| _ { 2 } } { 2 N } \leq \frac { 1 } { 2 } \sqrt { \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } . } \end{array}\tag{7.96}
$$

Finally, for the first term in $U .$ , let $\| \mathbf { y } \| _ { 2 } \leq 1$ and ${ \bf d } _ { y } : = \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { i } ^ { \star } ) } { v _ { i } ^ { \star } } \right) _ { i < N } \circ { \bf P } { \bf y }$ . Then

$$
\begin{array} { r l r } {  { \| \sum _ { k = 1 } ^ { N } \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \| _ { 2 } ^ { 2 } = \frac { 1 } { 4 N ^ { 2 } } \| \sum _ { j = 1 } ^ { N } \frac { \Psi ^ { ( 3 ) } ( \boldsymbol \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } \mathbf { P } _ { j l } ( \mathbf { e } _ { j } ^ { \top } \mathbf { P } \mathbf { y } ) \| _ { 2 } ^ { 2 } } } \\ & { } & { = \frac { 1 } { 4 N ^ { 2 } } \Bigg \| \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathrm { d i a g } ( \mathbf { d } _ { y } ) \mathbf { P } \mathbf { e } _ { l } \Bigg \| _ { 2 } ^ { 2 } , \qquad } \end{array}
$$

from which we have

$$
\begin{array} { c c } { \displaystyle \sum _ { l = 1 } ^ { N } \left\| \displaystyle \sum _ { k = 1 } ^ { N } \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \right\| _ { 2 } ^ { 2 } = \frac { 1 } { 4 N ^ { 2 } } \bigg \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathrm { d i a g } ( \mathbf { d } _ { y } ) \mathbf { P } \bigg \| _ { F } ^ { 2 } \leq \frac { 1 } { 4 N ^ { 2 } } \bigg \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathrm { d i a g } ( \mathbf { d } _ { y } ) \bigg \| _ { F } ^ { 2 } } \\ { = \frac { 1 } { 4 N ^ { 2 } } \displaystyle \sum _ { j = 1 } ^ { N } \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \right) ^ { 2 } ( \mathbf { e } _ { j } ^ { \top } \mathbf { P y } ) ^ { 2 } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } } \\ { \leq \frac { 1 } { 4 N ^ { 2 } } \operatorname* { m a x } _ { m \in [ N ] } \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \mathbf { x } _ { m } \| \mathbf { P y } \| _ { 2 } ^ { 2 } } \\ { \leq \frac { 1 } { 4 N ^ { 2 } } \operatorname* { m a x } _ { m \in [ N ] } \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \mathbf { x } _ { m } \leq \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| } { 4 N } . } \end{array}\tag{97}
$$

Therefore

$$
\operatorname* { s u p } _ { \| \mathbf { y } \| _ { 2 } \leq 1 } \sqrt { \sum _ { l = 1 } ^ { N } \left\| \sum _ { k = 1 } ^ { N } \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \right\| _ { 2 } ^ { 2 } } \leq \frac { 1 } { 2 } \sqrt { \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } .
$$

Observe that every single term $\| \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \| _ { 2 } ^ { 2 }$ can be upper bounded by the r.h.s. of (7.97) up to a constant factor, that is

$$
\begin{array} { r l } & { U _ { 1 } : = \displaystyle \operatorname* { s u p } _ { \| \mathbf { y } \| _ { 2 } \leq 1 }  { \displaystyle \sum _ { l = 1 } ^ { N } \| \displaystyle \sum _ { k \neq l } ^ { N } \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \| _ { 2 } ^ { 2 } } } \\ & { \leq \displaystyle \operatorname* { s u p } _ { \| \mathbf { y } \| _ { 2 } \leq 1 } \sqrt { \displaystyle \sum _ { l = 1 } ^ { N } \| \displaystyle \sum _ { k = 1 } ^ { N } \mathbf { a } ^ { ( k l ) } \mathbf { y } _ { k } \| _ { 2 } ^ { 2 } } + \displaystyle \operatorname* { s u p } _ { \| \mathbf { y } \| _ { 2 } \leq 1 } \sqrt { \displaystyle \sum _ { l = 1 } ^ { N } \| \mathbf { a } ^ { ( l l ) } \mathbf { y } _ { l } \| _ { 2 } ^ { 2 } } } \\ & { \leq \displaystyle \frac { 1 } { 2 } \sqrt { \displaystyle \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } + \displaystyle \operatorname* { m a x } _ { l \in [ N ] } \| \mathbf { a } ^ { ( l l ) } \| _ { 2 } } \\ & { \leq \displaystyle \frac { 1 } { 2 } \sqrt { \displaystyle \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } + V \leq \sqrt { \displaystyle \frac { A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } } . } \end{array}\tag{7.98}
$$

Now, using Lemma B.3 and the aforementioned upper bounds (7.94), (7.95), (7.98), and (7.96), we have with probability at least $1 - 2 e ^ { - t }$

$$
\begin{array} { r l } & { \| \Delta _ { q u a d } \| _ { 2 } \lesssim B \left\{ T + U \sqrt { t } + V t \right\} } \\ & { \qquad \lesssim \sqrt { \frac { B ^ { 3 } A _ { \operatorname* { m a x } } \{ \mathrm { r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) } } { N } } + \sqrt { \frac { t B ^ { 3 } A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| } { N } } + \sqrt { \frac { t ^ { 2 } B ^ { 2 } A _ { \operatorname* { m a x } } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| } { N } } .  \end{array}\tag{7.99}
$$

Bound for $\| \Delta _ { \geq 3 } \| _ { 2 }$ . It sufices to find upper bounds on $\| \Delta _ { \geq 3 } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) }$ , which, by the analysis in Section A.6.3, reduces to the upper bounds on $\| Q _ { 2 } ( \hat { \Delta } ) - Q _ { 2 } ( \Delta _ { l i n } ) \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } }$ and $\| Q _ { 3 } ( \hat { \Delta } ) \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } }$ By (A.119), we have

$$
\Vert Q _ { 2 } ( \hat { \Delta } ) - Q _ { 2 } ( \Delta _ { l i n } ) \Vert _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } } \lesssim \left[ \operatorname* { m a x } _ { m \in [ N ] } \vert \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } \vert + \frac { 1 } { 2 } \operatorname* { m a x } _ { l \in [ N ] } \vert \mathbf { x } _ { l } ^ { \top } \hat { \mathbf { e } } _ { e r r } \vert \right] \Vert \hat { \mathbf { e } } _ { e r r } \Vert _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } .
$$

Moreover, by Theorem 4.1, ma $\begin{array} { r } { \mathrm { x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \hat { \Delta } | \leq \frac { 1 } { 2 } } \end{array}$ with probability at least $1 - d ^ { - c }$ for suficiently large N. Then by following the argument used to prove (A.121), we have

$$
\| Q _ { 3 } ( \hat { \Delta } ) \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } \lesssim \left( \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \right) ^ { 2 } \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } + \left( \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } ( \hat { \mathbf { e } } _ { e r r } ) | \right) ^ { 2 } \| \hat { \mathbf { e } } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } .
$$

By Lemma 7.8, whenever Theorem 4.1 holds, with probability at least $1 - d ^ { - c }$ , we have

$$
\operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \log d ,
$$

$$
\operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \hat { \mathbf { e } } _ { e r r } | \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + B A _ { \operatorname* { m a x } } \log d ,
$$

$$
\| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } \lesssim \sqrt { \frac { B d _ { \theta } \log d } { N } } ,
$$

where to obtain the second inequality, we also utilize Lemma A.2. Under the setting of Theorem 4.1, with probability at least $1 - d ^ { - c }$ , we have

$$
\begin{array} { r l } & { \| \Delta _ { \geq 3 } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) \lesssim \| Q _ { 2 } ( \Delta ) - Q _ { 2 } ( \Delta _ { l i n } ) \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } + \| Q _ { 3 } ( \Delta ) \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } \\ & { \qquad \lesssim \left[ \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + B A _ { \operatorname* { m a x } } \log d \right] \left( \| \Delta _ { b i a s } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) + \| \Delta _ { q u a d } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) + \| \Delta _ { \geq 3 } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) \right) } \end{array}
$$

$$
\begin{array} { r l } & { + A _ { \operatorname* { m a x } } \log d \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } } \\ & { \lesssim \Big [ \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + B A _ { \operatorname* { m a x } } \log d \Big ] \left( \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } + \| \Delta _ { q u a d } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \right) } \\ & { + A _ { \operatorname* { m a x } } \log d \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } } \\ & { \lesssim \Big [ \sqrt { B A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } \Big ] \sqrt { \frac { A _ { \operatorname* { m a x } } B d _ { \theta } \log ^ { 2 } d } { N } } + A _ { \operatorname* { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } } \\ & { \lesssim B A _ { \operatorname* { m a x } } \sqrt { \frac { d _ { \theta } \log ^ { 3 } d } { N } } + \sqrt { B } A _ { \operatorname* { m a x } } ^ { 3 / 2 } \frac { d _ { \theta } \log d } { \sqrt { N } } , } \end{array}
$$

where the second inequality is from driving N suficiently large such that the square bracketed term is smaller than $^ { \frac { 1 } { 2 } , }$ , thus the $\| \Delta _ { \geq 3 } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) }$ can be absorbed into the l.h.s. Using Lemma A.1, we conclude that

$$
\begin{array} { r l } & { \| \Delta _ { \geq 3 } \| _ { 2 } \leq \| \Delta _ { \geq 3 } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } \sqrt { \| \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \| } } \\ & { \qquad \lesssim \sqrt { \| \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } \| } \left[ \frac { B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 / 2 } d } { N ^ { 3 / 2 } } + \frac { B ^ { 2 } \mu _ { Q } ^ { 3 / 2 } d _ { \theta } ^ { 5 / 2 } \log d } { N ^ { 2 } } \right] . } \end{array}\tag{7.100}
$$

Combining the Bounds. Setting $t = C$ log d for some suficiently large $C > 0$ in (7.99), combining it with (7.100), and using Lemma A.1 again to upper bound $A _ { \mathrm { m a x } }$ gives

$$
\begin{array} { r } { \| \dot { \Delta } - \Delta _ { l i n } - \Delta _ { b i a s } \| _ { 2 } \lesssim \frac { \sqrt { \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| B \mu _ { Q d } } d _ { \theta } } { N } \left[ \frac { B ^ { 3 / 2 } \sqrt { \mu _ { Q d \theta } } \log ^ { 3 / 2 } d } { \sqrt { N } } + \frac { B ^ { 3 / 2 } \mu Q d _ { \theta } ^ { 3 / 2 } \log d } { N } \right] } \\ { + \sqrt { \frac { \operatorname { t r } \left( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \right) \log d _ { \theta } } { N } } B ^ { 2 } \left[ \sqrt { \frac { \mu _ { Q } d _ { \theta } } { N \log d _ { \theta } } } + \sqrt { \frac { \mu _ { Q } d _ { \theta } \log d } { N \log d _ { \theta } } } + \sqrt { \frac { \mu _ { Q } d _ { \theta } \log d } { B N \log d _ { \theta } } } \right] . } \end{array}
$$

where we use the fact that $\lVert \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \rVert \leq \mathrm { t r } ( \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } )$ to further simplify (7.99). Then by setting $N \gtrsim$ max $\left\{ B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d , B ^ { 4 } \mu _ { Q } d _ { \theta } \log d \right\}$ the above inequality yields (4.44). □

## A Proofs of Auxiliary Results

In this appendix, we include proofs of a number of intermediate results needed in the main part of this paper.

## A.1 Proof of Proposition 2.3

The reparameterization map $\pmb \theta \mapsto \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb \theta$ is injective, so it sufices to work with $\pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } }$ Since the function $\frac { 1 } { 1 + e ^ { - t } }$ is strictly increasing in t, for any two feasible parameters $\mathbf { w } = \mathbf { w } _ { o } +$ $\mathbf { V } _ { A ^ { \perp } } \pmb { \theta }$ and $\mathbf { w } ^ { \prime } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \perp } } \pmb \theta ^ { \prime }$ , the condition $p _ { \pmb { \alpha } } ( \mathbf { w } ) = p _ { \pmb { \alpha } } ( \mathbf { w } ^ { \prime } ) , \forall \pmb { \alpha } \in \mathcal { E } _ { s }$ is equivalent to

$$
( { \bf e } _ { i } - { \bf e } _ { j } ) ^ { \top } \mathfrak { A } { \bf V } _ { A ^ { \perp } } ( \pmb \theta - \pmb \theta ^ { \prime } ) = 0 , \forall \pmb \alpha = ( i , j ) \in \mathcal { E } _ { s } .\tag{A.101}
$$

Equation (A.101) states that the vector ${ \mathfrak { A } } \mathbf { V } _ { A ^ { \perp } } ( \pmb \theta - \pmb \theta ^ { \prime } )$ is constant on connected components of $\mathcal { G } ( [ d ] , \mathcal { E } _ { s } , 1 )$ , i.e., there exist coeficients $[ c _ { 1 } , \ldots , c _ { K } ]$ such that

$$
{ \mathfrak { A } } \mathbf { V } _ { A ^ { \perp } } ( \pmb { \theta } - \pmb { \theta } ^ { \prime } ) = \sum _ { t = 1 } ^ { K } c _ { t } \mathbf { l } | _ { \mathcal { C } _ { t } } \Leftrightarrow \left[ \mathbf { C } _ { I } \quad { \mathfrak { A } } \mathbf { V } _ { A ^ { \perp } } \right] \left[ { - [ { \boldsymbol { c } } _ { 1 } , \dots , { \boldsymbol { c } } _ { K } ] ^ { \top } } \right] = \mathbf { 0 } .\tag{A.102}
$$

We first prove (2.13) implies identifiability. Suppose that $[ \mathbf { C } _ { I } , \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ] \in \mathbb { R } ^ { d \times ( d _ { \theta } + K ) }$ has full column rank $d _ { \theta } + K$ . Then (A.102) implies $[ c _ { 1 } , \ldots , c _ { K } ] ^ { \top } = \mathbf { 0 }$ and ${ \pmb \theta } - { \pmb \theta } ^ { \prime } = { \pmb 0 }$ . Since the reparameterization is injective, we have $\mathbf { w } = \mathbf { w } ^ { \prime }$ accordingly.

Conversely, suppose that (2.13) fails. Then rank $( [ \mathbf { C } _ { I } , \mathfrak { 2 } \imath \mathbf { V } _ { A ^ { \perp } } ] ) < d _ { \theta } + K$ and hence there exists a non-zero solution $\left[ - [ c _ { 1 } , \ldots , c _ { K } ] , \pmb \theta - \pmb \theta ^ { \prime } ] ^ { \top } \right.$ satisfying the linear system of equations at the r.h.s. of (A.102). Since the columns of $\mathbf { C } _ { I }$ are linearly independent, we assert that $\pmb \theta \ne \pmb \theta ^ { \prime }$ because otherwise, we would have $\mathbf { C } _ { I } [ c _ { 1 } , \ldots , c _ { K } ] ^ { \top } = \mathbf { 0 }$ for $[ c _ { 1 } , \ldots , c _ { K } ] ^ { \top } \neq \mathbf { 0 }$ . Let w, $\mathbf { w } ^ { \prime } \in \{ \mathbf { w } \in$ $\mathbb { R } ^ { p } \mid \mathbf { A } \mathbf { w } = \mathbf { b } \}$ be defined according to the $\theta , \theta ^ { \prime }$ . Then by the equivalence between (A.102) and (A.101), $\mathbf { a } _ { \alpha } ^ { \top } \mathbf { w } = \mathbf { a } _ { \alpha } ^ { \top } \mathbf { w } ^ { \prime }$ holds for all $\pmb { \alpha } \in \mathcal { E } _ { s }$ . This further implies $p _ { \alpha } ( \mathbf { w } ) = p _ { \alpha } ( \mathbf { w } ^ { \prime } ) , \forall \pmb { \alpha } \in \mathcal { E } _ { s }$ for $\mathbf { w } \neq \mathbf { w } ^ { \prime }$ , which means that the model is not identifiable over $\left\{ \mathbf { w } \in \mathbb { R } ^ { p } \mid \mathbf { A } \mathbf { w } = \mathbf { b } \right\}$ □

## A.2 Proof of Corollary 2.1

The first part is derived from identifying the null space of W. From standard results in spectral graph theory, span $( \{ \mathbf { 1 } _ { \mathcal { C } _ { t } } \} _ { t = 1 } ^ { K } ) = \operatorname { s p a n } ( \mathbf { C } _ { I } ) = \operatorname { N u l l } ( \mathbf { L } )$ . Suppose that there exists $\pmb \theta \neq \mathbf 0$ such that $\pmb { \theta } ^ { \top } \mathbf { W } \pmb { \theta } = 0$ . Since $\mathrm { ~ \bf ~ L ~ } \succeq \mathrm { ~ \bf ~ 0 ~ }$ , equivalently, this gives $( \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \pmb \theta ) ^ { \top } \mathbf { L } ( \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \pmb \theta ) = 0$ , and ${ \mathfrak { A } } { \mathbf { V } } _ { A ^ { \perp } } \pmb { \theta } \in$ Null(L). Thus

$$
\operatorname { N u l l } ( \mathbf { W } ) = \{ \pmb { \theta } \in \mathbb { R } ^ { d _ { \theta } } \ | \ \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \pmb { \theta } \in \mathrm { s p a n } ( \mathbf { C } _ { I } ) \}\tag{A.103}
$$

Since the matrix $\mathbf { C } _ { I }$ has linearly independent columns, whenever (2.13) holds, we will have

$$
\mathrm { s p a n } ( \mathbf { C } _ { I } ) \cap \mathrm { R a n g e } ( \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ) = \{ \mathbf { 0 } \} , \quad \mathrm { r a n k } ( \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ) = d _ { \theta } ,\tag{A.104}
$$

and vice versa. If $\mathrm { N u l l } ( \mathbf { W } ) = \{ \mathbf { 0 } \}$ , then (A.103) yields (A.104), and consequently (2.13). Conversely, suppose that (2.13) holds. Then (A.104) implies that the only θ which satisfies (A.103) is identically zero. Thus $\mathrm { N u l l } ( \mathbf { W } ) = \{ \mathbf { 0 } \}$

For the second part, let rank ${ \mathbf { \rho } } ( \mathbf { W } ) = r < d _ { \theta }$ , and the full eigen decomposition of W be

$$
\mathbf { W } = [ \mathbf { Q } _ { + } , \mathbf { Q } _ { \perp } ] \left[ \mathbf { 0 } \quad \mathbf { 0 } \right] [ \mathbf { Q } _ { + } , \mathbf { Q } _ { \perp } ] ^ { \top } .
$$

Then, the refined linear equality system can be expressed as

$$
\begin{array} { r } { \tilde { \mathbf { A } } \mathbf { w } = \tilde { \mathbf { b } } , \quad \tilde { \mathbf { A } } : = \left[ \mathbf { A } \mathbf { \Xi } \right] , \quad \tilde { \mathbf { b } } : = \left[ \mathbf { b } \right] . } \end{array}
$$

By following the same procedure and reparameterizing as in (2.11), the counterpart of W in the refined model is

$$
\begin{array} { r } { \tilde { \mathbf { W } } : = \mathbf { Q } _ { + } ^ { \top } \mathbf { V } _ { A ^ { \perp } } ^ { \top } \mathfrak { A } ^ { \top } \mathbf { L } \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \mathbf { Q } _ { + } = \mathbf { Q } _ { + } ^ { \top } \mathbf { W } \mathbf { Q } _ { + } = \mathbf { A } \succ \mathbf { 0 } . } \end{array}
$$

By the first part of the proof, we know the refined model enforces (2.13) and rank $( [ \mathbf { C } _ { I } , \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \mathbf { Q } _ { + } ] ) =$ $r + K$ , thus the refined model is identifiable.

Finally, fix an arbitrary $\mathbf { w } ^ { \star }$ that satisfies Assumption 2.1, and set $\tilde { \mathbf { w } } ^ { \star } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \bot } } \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star }$ We verify that: $( \mathrm { i } ) p _ { \alpha } ( \tilde { \mathbf { w } } ^ { \star } ) = p _ { \alpha } ( \mathbf { w } ^ { \star } ) , \forall \alpha \in \mathcal { E } _ { s }$ , and $( \mathrm { i i } ) ~ \tilde { \mathbf { w } } ^ { \star }$ is unique under the refined model, thereby completing the proof. For (i), notice that $\pmb { \theta } ^ { \star } - \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } = \mathbf { Q } _ { \perp } \mathbf { Q } _ { \perp } ^ { \top } \pmb { \theta } ^ { \star } \in \mathrm { N u l l } ( \mathbf { W } )$ . By (A.103) we know $\mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \big ( \pmb { \theta } ^ { \star } - \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } \big ) \in \mathrm { s p a n } \big ( \mathbf { C } _ { I } \big )$ , thus for any ${ \pmb { \alpha } } = ( i , j ) \in \mathcal { E } _ { s }$

$$
( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ^ { \top } \mathfrak { A } \mathbf { V } _ { A ^ { \perp } } ( \pmb { \theta } ^ { \star } - \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } ) = ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) ^ { \top } \mathfrak { A } ( \mathbf { w } ^ { \star } - \tilde { \mathbf { w } } ^ { \star } ) = \mathbf { a } _ { \alpha } ^ { \top } ( \mathbf { w } ^ { \star } - \tilde { \mathbf { w } } ^ { \star } ) = 0 ,
$$

which proves (i). For (ii), suppose there exists $\tilde { \mathbf { w } } _ { o } ^ { \star } = \mathbf { w } _ { o } + \mathbf { V } _ { A ^ { \bot } } \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } \mathbf { \Lambda } _ { c }$ that satisfies $\tilde { \textbf { A w } } = \tilde { \textbf { b } }$ and induces the same probability distribution, then following (A.102), we have

$\mathfrak { A } \mathbf { V } _ { A ^ { \perp } } \big ( \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } - \mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } \pmb { \theta } ^ { \star } \mathstrut _ { o } \big ) \in \mathrm { s p a n } ( \mathbf { C } _ { I } )$ . By (A.103) and the definition of $\mathbf { Q } _ { + }$ , this implies $\mathbf { Q } _ { + } \mathbf { Q } _ { + } ^ { \top } ( \pmb { \theta } ^ { \star } - \pmb { \theta } ^ { \star } \lrcorner ) \in \mathrm { N u l l } ( \mathbf { W } ) \bigcap \mathrm { R a n g e } ( \mathbf { W } ) = \{ \mathbf { 0 } \}$ hence $\tilde { \mathbf { w } } ^ { \star } = \tilde { \mathbf { w } } _ { o } ^ { \star }$ □

## A.3 Proof of Proposition 3.2

This result can be viewed as a generalization of [61, Proposition 17]. Fix a unit vector $\pmb { \theta } _ { o } \in \mathbb { R } ^ { d _ { \theta } }$ and consider the feasible ray $\{ \pmb { \theta } ^ { \prime } \in \mathbb { R } ^ { d _ { \theta } } \ | \ \pmb { \theta } ^ { \prime } = c \pmb { \theta } _ { o } , c \geq 0 \}$ . For query α the utility diference along the ray is $\mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + c \mathbf { q } _ { \alpha } ^ { \top } \pmb { \theta } _ { o }$ . Since $\mathcal { E } _ { s }$ is a finite set, there exists some $c _ { 0 } < \infty$ such that for every $\pmb { \alpha } \in \mathcal { E } _ { s } , \mathrm { s g n } ( \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \pmb { \alpha } } + c \mathbf { q } _ { \alpha } ^ { \top } \pmb { \theta } _ { o } )$ remains unchanged for all $c \geq c _ { 0 }$ . Hence, by [61, Proposition 17], there exists a realization $\tilde { \pmb { y } } ^ { ( N ) } \in \{ 0 , 1 \} ^ { N }$ of the response sequence $Y ^ { ( N ) }$ agreeing with these signs such that

$$
\mathbb { P } _ { c \theta _ { o } } ^ { ( N ) } \left\{ Y ^ { ( N ) } = \tilde { \pmb y } ^ { ( N ) } \right\} = \prod _ { \pmb { \alpha } \in \mathcal { E } _ { s } } \prod _ { t \in [ n _ { \alpha } ] } \mathbb { P } _ { c \theta _ { o } } ^ { \boldsymbol { \alpha } } \left\{ Y _ { \pmb { \alpha } } ^ { ( t ) } = \tilde { y } _ { \pmb { \alpha } } ^ { ( t ) } \right\} \geq \frac { 1 } { 2 ^ { N } } , \forall c \geq c _ { 0 } .
$$

Fix any $\bar { c } \geq c _ { 0 }$ , then $\{ \pmb { \theta } ^ { \prime } \in \mathbb { R } ^ { d _ { \pmb { \theta } } } \ \vert \ \pmb { \theta } ^ { \prime } = c \pmb { \theta } _ { o } , c \in [ c _ { 0 } , \bar { c } ] \} \ \subseteq \ \Theta _ { B }$ for every suficiently large B. Consequently, for any estimator θ<sup>ˆ</sup>

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { \theta \in \Theta _ { B } } \mathbb { E } _ { \theta } ^ { ( N ) } \left[ \| \hat { \theta } - \theta \| _ { 2 } ^ { 2 } \right] \geq \operatorname* { s u p } _ { c \in [ c _ { 0 } , \hat { \sigma } ] } \mathbb { E } _ { c \theta _ { o } } ^ { ( N ) } \left[ \| \hat { \theta } - c \theta _ { o } \| _ { 2 } ^ { 2 } \right] = \displaystyle \operatorname* { s u p } _ { c \in [ c _ { 0 } , \hat { \sigma } ] } \sum _ { y ^ { ( N ) } \in y ^ { N } } \mathbb { F } _ { c \theta _ { o } } ^ { ( N ) } \left\{ Y ^ { ( N ) } = y ^ { ( N ) } \right\} \| \hat { \theta } ( y ^ { ( N ) } ) - c \theta _ { o } \| _ { 2 } ^ { 2 } } & \\ { \geq \displaystyle \operatorname* { s u p } _ { c \in [ c _ { 0 } , \hat { \sigma } ] } \mathbb { P } _ { \theta _ { o } } ^ { ( N ) } \left\{ Y ^ { ( N ) } = \tilde { y } ^ { ( N ) } \right\} \| \hat { \theta } ( \tilde { y } ^ { ( N ) } ) - c \theta _ { o } \| _ { 2 } ^ { 2 } \geq \frac { 1 } { 2 ^ { N } } \displaystyle \operatorname* { s u p } _ { c \in [ c _ { 0 } , \hat { \sigma } ] } \| \hat { \theta } ( \tilde { y } ^ { ( N ) } ) - c \theta _ { o } \| _ { 2 } ^ { 2 } } & \\ { \geq \frac { 1 } { 2 ^ { N } } \operatorname* { m a x } \left\{ \| \hat { \theta } ( \tilde { y } ^ { ( N ) } ) - c _ { 0 } \theta _ { o } \| _ { 2 } ^ { 2 } , \| \hat { \theta } ( \tilde { y } ^ { ( N ) } ) - \bar { c } \theta _ { o } \| _ { 2 } ^ { 2 } \right\} \geq \frac { ( \bar { c } - c _ { 0 } ) ^ { 2 } } { 2 ^ { N + 2 } } . } & \end{array}
$$

Since $\bar { c } > c _ { 0 }$ is arbitrary, for every $M \ > \ 0$ , one may choose ¯c suficiently large such that $\frac { ( \bar { c } - c _ { 0 } ) ^ { 2 } } { 2 ^ { N + 2 } } > M$ . Consequently, lim $B { \to } \infty \operatorname* { s u p } _ { \pmb { \theta } \in \Theta _ { B } } \mathbb { E } _ { \pmb { \theta } } ^ { ( N ) } \left[ \| \hat { \pmb { \theta } } - \pmb { \theta } \| _ { 2 } ^ { 2 } \right] = \infty .$ □

## A.4 Proof of Lemma 7.4

The KL divergence between Bernoulli distributions can be written as the first-order Taylor remainder of $\Psi ( \cdot )$ , see, e.g. [61]. That is

$$
\begin{array} { r l } & { { \mathrm { d } } _ { K L } \big ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { \alpha } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } \big ) = \mathbb { E } _ { \mathrm { P } _ { \Psi ^ { \prime } } ^ { \alpha } } \left[ \log \left( \frac { \mathrm { d } \mathrm { P } _ { \Psi ^ { \prime } } ^ { \alpha } ( Y ) } { \mathrm { d } \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } ( Y ) } \right) \right] } \\ & { \qquad = \mathbb { E } _ { \mathrm { P } _ { \Psi ^ { \prime } } ^ { \alpha } } \left[ Y a - \Psi ( a ) - Y z + \Psi ( z ) \right] = \Psi ( z ) - \Psi ( a ) - \Psi ^ { \prime } ( a ) ( z - a ) , } \end{array}
$$

where $\mathbb { E } _ { \mathrm { P } _ { \Psi ^ { \prime } } ^ { a } }$ denotes the expectation under $\mathrm { P } _ { \Psi ^ { \prime } } ^ { a }$ , and the last two equalities are from Proposition 2.1. Observe that

$$
\begin{array} { r l } & { \mathrm { d } \| _ { K L } \big ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { \alpha } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } \big ) + \mathrm { d } \| _ { K L } \big ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { b } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } \big ) = \Psi ( z ) - \Psi ( a ) - \Psi ^ { \prime } ( a ) ( z - a ) + \Psi ( z ) - \Psi ( b ) - \Psi ^ { \prime } ( b ) ( z - b ) } \\ & { \qquad \geq \underset { z \in \mathbb { R } } { \operatorname* { i n f } } \Psi ( z ) - \Psi ( a ) - \Psi ^ { \prime } ( a ) ( z - a ) + \Psi ( z ) - \Psi ( b ) - \Psi ^ { \prime } ( b ) ( z - b ) , } \end{array}
$$

and the infimum is attained at $z ^ { \star }$ such that $\begin{array} { r } { \Psi ^ { \prime } ( z ^ { \star } ) = \frac { \Psi ^ { \prime } ( a ) + \Psi ^ { \prime } ( b ) } { 2 } } \end{array}$ . Since $\Psi ^ { \prime } ( \cdot )$ is strictly increasing, w.l.o.g. assume that $a \leq b .$ , we have $z ^ { \star } \in [ a , b ]$ . Thus, there exist $\xi _ { 1 } , \xi _ { 2 } \in [ - \log B , \log B ]$ such that

$$
\begin{array} { l } { \displaystyle \mathrm { d } | _ { K L } ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { a } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } ) + \mathrm { d } | _ { K L } ( \mathrm { P } _ { \Psi ^ { \prime } } ^ { b } , \mathrm { P } _ { \Psi ^ { \prime } } ^ { z } ) \geq \frac { 1 } { 2 } \left( \Psi ^ { \prime \prime } ( \xi _ { 1 } ) ( z ^ { \star } - a ) ^ { 2 } + \Psi ^ { \prime \prime } ( \xi _ { 2 } ) ( z ^ { \star } - b ) ^ { 2 } \right) } \\ { \displaystyle \qquad \geq \frac { 1 } { 8 B } \left( ( z ^ { \star } - a ) ^ { 2 } + ( z ^ { \star } - b ) ^ { 2 } \right) \geq \frac { 1 } { 1 6 B } ( a - b ) ^ { 2 } , } \end{array}\tag{A.105}
$$

where the second last inequality is by Proposition 2.2 and the fact that Var $\operatorname { \mathrm { { P } } } _ { \mathrm { { r } } , \mathrm { { r } } } ^ { a } , ( Y ) = \Psi ^ { \prime \prime } ( a )$ , and Ψ the last inequality follows from $\begin{array} { r } { ( z ^ { \star } - a ) ^ { 2 } + ( z ^ { \star } - b ) ^ { 2 } = 2 ( z ^ { \star } - \frac { a + b } { 2 } ) ^ { 2 } + \frac { ( a - \dot { b } ) ^ { 2 } } { 2 } \geq \frac { ( a - b ) ^ { 2 } } { 2 } } \end{array}$ . Next, observe that

$$
\begin{array} { r l } & { \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( \theta _ { 1 } ) = \Psi \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } \right) - \Psi \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 1 } } { \sigma } \right) - \Psi ^ { \prime } \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 1 } } { \sigma } \right) \frac { \mathbf { q } _ { \alpha } ^ { \top } ( v - \theta _ { 1 } ) } { \sigma } } \\ & { \qquad = \mathbf { q } _ { K L } ( \mathrm { P } _ { \theta _ { 1 } } ^ { \alpha } , \mathrm { P } _ { \sigma } ^ { \alpha } ) , } \\ & { \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( \theta _ { 2 } ) = \Psi \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } v } { \sigma } \right) - \Psi \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 2 } } { \sigma } \right) - \Psi ^ { \prime } \left( \frac { \mathbf { w } _ { o } ^ { \top } \mathbf { a } _ { \alpha } + \mathbf { q } _ { \alpha } ^ { \top } \theta _ { 2 } } { \sigma } \right) \frac { \mathbf { q } _ { \alpha } ^ { \top } ( v - \theta _ { 2 } ) } { \sigma } } \\ &  \qquad = \mathbf { q } _ { K L } ( \mathrm { P } _ { \theta _ { 2 } } ^ { \alpha } , \mathrm { P } _ { \sigma } ^ { \alpha } ) . \end{array}
$$

Thus, by (A.105)

$$
\begin{array} { r l } & { \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 1 } } ^ { \alpha } ( \theta _ { 1 } ) + \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( v ) - \mathcal { L } _ { \theta _ { 2 } } ^ { \alpha } ( \theta _ { 2 } ) = \mathrm { d } | _ { K L } ( \mathrm { P } _ { \theta _ { 1 } } ^ { \alpha } , \mathrm { P } _ { v } ^ { \alpha } ) + \mathrm { d } | _ { K L } ( \mathrm { P } _ { \theta _ { 2 } } ^ { \alpha } , \mathrm { P } _ { v } ^ { \alpha } ) } \\ & { \qquad \geq \displaystyle \frac { 1 } { 1 6 B \sigma ^ { 2 } } \Big | \mathbf { q } _ { \alpha } ^ { \top } ( \theta _ { 1 } - \theta _ { 2 } ) \Big | ^ { 2 } , } \end{array}
$$

which proves (7.56).

## A.5 Proof of Lemma 7.6

First, by adding and subtracting terms, the gradient $\nabla \ell ( \pmb \theta ^ { \star } + \Delta )$ can be equivalently written as

$$
\begin{array} { l } { \displaystyle \nabla \ell ( \pmb { \theta } ^ { \star } + \Delta ) = \nabla \ell ( \pmb { \theta } ^ { \star } ) + \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \Delta + \big ( \nabla \ell ( \pmb { \theta } ^ { \star } + \Delta ) - \nabla \ell ( \pmb { \theta } ^ { \star } ) - \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \Delta \big ) } \\ { \displaystyle \qquad = \nabla \ell ( \pmb { \theta } ^ { \star } ) + \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) \Delta + \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { x } _ { j } r _ { j } ( \Delta ) , } \end{array}
$$

where $r _ { j } ( \Delta )$ is defined in (7.76). Using the relation $\Delta = \Delta _ { l i n } + \Delta _ { b i a s } + \mathbf { h }$ , and the definition of $\Delta _ { l i n } , \Delta _ { b i a s }$ in (4.40), we can present the first-order optimality condition as

$$
\begin{array} { l } { { \displaystyle { \bf 0 } = \nabla \ell ( { \pmb { \theta } } ^ { \star } + \Delta ) = \nabla \ell ( { \pmb { \theta } } ^ { \star } ) + \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) \Delta _ { l i n } + \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) \Delta _ { b i a s } + \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) { \bf h } + \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } { \bf x } _ { j } r _ { j } ( \Delta ) } } \\ { ~ = - \displaystyle \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } { \bf A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) { \bf x } _ { j } + \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) { \bf h } + \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } { \bf x } _ { j } r _ { j } ( \Delta ) } \\ { ~ = - \displaystyle \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } { \bf A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) { \bf x } _ { j } + \nabla ^ { 2 } \ell ( { \pmb { \theta } } ^ { \star } ) { \bf h } + \displaystyle \frac { 1 } { N } \sum _ { j = 1 } ^ { N } { \bf x } _ { j } r _ { j } ( \Delta _ { l i n } + \Delta _ { b i a s } + { \bf h } ) , } \end{array}
$$

where the second last equality follows from the definition of $\Delta _ { l i n }$ and $\Delta _ { b i a s }$ . Rearranging the terms gives rise to (7.80). □

## A.6 Proof of Lemma 7.8

Lemma 7.8 relies on several intermediate results stated below, which control the $\| \cdot \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) }$ and $\| \cdot \| _ { Q , \infty }$ norms of the deterministic second-order bias, and tail behavior of certain sub-Gaussian quadratic forms. We use the notation introduced at the beginning of Section 7.2.

## A.6.1 Deterministic residuals

Lemma A.1 Let $\begin{array} { r } { \mathbf { A } : = \frac { 1 } { N } \mathbf { X } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { X } ^ { \top } \in \mathbb { R } ^ { N \times N } } \end{array}$ . Then the following assertions hold.

(i) AV<sup>⋆</sup>A = A, i.e.,

$$
\sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { i j } ^ { 2 } = \mathbf { A } _ { i i } , \quad \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } = d _ { \theta } .\tag{A.106}
$$

$$
\begin{array} { r } { ( i i ) ~ A _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ N ] , j \in [ N ] } | { \bf A } _ { i j } | = \operatorname* { m a x } _ { i \in [ N ] } { \bf A } _ { i i } \leq \frac { 4 B \mu _ { Q } d _ { \theta } } { N } . } \end{array}
$$

The proof is deferred to Appendix A.7. As a straightforward corollary from Lemma A.1, the matrix

$$
\mathbf { P } : = \sqrt { \mathbf { V } ^ { \star } } \mathbf { A } \sqrt { \mathbf { V } ^ { \star } }\tag{A.107}
$$

is an orthogonal projection, i.e., $\mathbf { P } ^ { \top } = \mathbf { P }$ and $\mathbf { P } ^ { 2 } = { \sqrt { \mathbf { V } ^ { \star } } } \mathbf { A } \mathbf { V } ^ { \star } \mathbf { A } { \sqrt { \mathbf { V } ^ { \star } } } = { \sqrt { \mathbf { V } ^ { \star } } } \mathbf { A } { \sqrt { \mathbf { V } ^ { \star } } } = \mathbf { P }$

Lemma A.2 Consider the following linear functional

$$
- \mathbf { x } _ { m } ^ { \mathsf { T } } \Delta _ { b i a s } = \mathbf { x } _ { m } ^ { \mathsf { T } } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \left( \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } \right) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { A } _ { j j } , m \in [ N ] ,
$$

where $\Delta _ { b i a s }$ is defined as in (4.40b). It satisfies

$$
\operatorname* { m a x } _ { m \in [ N ] } \bigg | - \mathbf { x } _ { m } ^ { \top } \Delta _ { b i a s } \bigg | \leq \frac { 1 } { 2 } A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } \lesssim \frac { B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } .\tag{A.108}
$$

Proof. The upper bound follows by Cauchy–Schwarz inequality

$$
\begin{array} { r l r } {  {  - \mathbf { x } _ { m } ^ { \top } \Delta _ { b i a s }  \leq \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } | \mathbf { A } _ { m j } | v _ { j } ^ { \star } \mathbf { A } _ { j j } \leq \frac { 1 } { 2 } ( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } ) ^ { 1 / 2 } ( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } ^ { 2 } ) ^ { 1 / 2 } } } \\ & { } & { \leq \frac { 1 } { 2 } \sqrt { \mathbf { A } _ { m m } } \sqrt { A _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } } \leq \frac { 1 } { 2 } A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } , } \end{array}
$$

where the last two inequalities follow from (A.106).

Lemma A.3 Let $\Delta _ { b i a s }$ be defined as in (4.40b). Then

$$
\| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } \leq \frac { 1 } { 2 } \sqrt { \frac { A _ { \mathrm { m a x } } d _ { \theta } } { N } } .\tag{A.109}
$$

Proof. This follows from reproducing the orthogonal projection P, and then using $\| \mathbf { P } \| \leq 1$ Let $\begin{array} { r } { \mathbf { d } _ { v ^ { \star } } : = \left( \mathbf { A } _ { i i } \frac { \Psi ^ { ( 3 ) } ( \eta _ { i } ^ { \star } ) } { v _ { i } ^ { \star } } \sqrt { v _ { i } ^ { \star } } \right) _ { i \le N } } \end{array}$ for the moment, then

$$
\begin{array} { r l } & { \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } ^ { 2 } = \displaystyle \frac { 1 } { 4 N } \mathbf { d } _ { v ^ { \star } } ^ { \top } \mathbf { P } \mathbf { d } _ { v ^ { \star } } \leq \displaystyle \frac { 1 } { 4 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } ^ { 2 } \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } \right) ^ { 2 } v _ { j } ^ { \star } \leq \frac { 1 } { 4 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } ^ { 2 } v _ { j } ^ { \star } } \\ & { \qquad \leq A _ { \operatorname* { m a x } } \displaystyle \frac { 1 } { 4 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } v _ { j } ^ { \star } \leq \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { 4 N } , } \end{array}
$$

where the second inequality follows from the fact that $| \Psi ^ { ( 3 ) } ( \eta ) | \le \Psi ^ { \prime \prime } ( \eta )$ for any $\eta \in \mathbb { R }$ (see Proposition 2.1), and the last inequality follows by Lemma A.1. □

## A.6.2 Random residuals

Lemma A.4 (Tail Bounds on $\Delta _ { l i n } )$ For $m \in [ N ]$ , the term $\Delta _ { l i n }$ satisfies

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ \vert \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } \vert \geq t \right\} \lesssim \exp \left( { - c \operatorname* { m i n } \left( \frac { t ^ { 2 } } { A _ { \operatorname* { m a x } } } , \frac { t } { A _ { \operatorname* { m a x } } } \right) } \right) ,\tag{A.110}
$$

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ \left| \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } ^ { 2 } - \frac { \operatorname { t r } ( \mathbf { I } _ { d _ { \theta } } ) } { N } \right| \geq t \right\} \lesssim \exp \left( - c \operatorname* { m i n } \left( \frac { N ^ { 2 } t ^ { 2 } } { B ^ { 2 } d _ { \theta } } , \frac { N t } { B } \right) \right) ,\tag{A.111}
$$

for every $t \geq 0$

Its proof is deferred to Appendix A.8. Lemma A.6 to be presented below can be viewed as a corollary from the general tail behavior of a certain sub-Gaussian quadratic form.

Lemma A.5 Let $\pmb { \xi } \in \mathbb { R } ^ { N }$ be a zero-mean, isotropic, sub-Gaussian random vector with independent coordinates, such that ma $\mathfrak { c } _ { m \in [ N ] } \| \pmb { \xi } _ { m } \| _ { \psi _ { 2 } } \leq \sqrt { K }$ . Let P, $\mathbf { D } \in \mathbb { R } ^ { N \times N }$ be any but fixed orthogonal projection and diagonal matrix, respectively. Then, for M := PDP and every $t \geq 0$

$$
\mathbb { P } _ { \xi } \left\{ \left. \xi ^ { \top } \mathbf { M } \xi - \mathrm { t r } ( \mathbf { M } ) \right. \geq t \right\} \lesssim \exp \left( - c \mathrm { m i n } \left( \frac { t ^ { 2 } } { K ^ { 2 } \mathrm { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } ) } , \frac { t } { K D _ { \mathrm { m a x } } } \right) \right)\tag{A.112}
$$

where $D _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ N ] } | { \bf D } _ { i i } |$

Proof. Our focus is on controlling $\lVert \mathbf { M } \rVert$ and $\| \mathbf { M } \| _ { F }$ , where

$$
\| \mathbf { M } \| = \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } = 1 } \left| \mathbf { v } ^ { \top } \mathbf { M } \mathbf { v } \right| \leq \operatorname* { m a x } _ { j \in [ N ] } \| \mathbf { D } _ { j j } | \operatorname* { s u p } _ { \| \mathbf { v } \| _ { 2 } = 1 } \| \mathbf { P } \mathbf { v } \| _ { 2 } \leq D _ { \operatorname* { m a x } } .
$$

On the other hand $\| \mathbf { M } \| _ { F } ^ { 2 } = \| \mathbf { P D P } \| _ { F } ^ { 2 } \leq \| \mathbf { D P } \| _ { F } ^ { 2 } = \operatorname { t r } ( \mathbf { D ^ { 2 } P } )$ . Then by Lemma B.2 we conclude (A.112). □

Lemma A.6 (Centered Quadratic Biases) For $m \in [ N ]$ , let

$$
Q _ { l i n } ^ { m } : = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \left[ \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } - \mathbf { A } _ { j j } \right] ,\tag{A.113a}
$$

$$
Q _ { E r r , C } ^ { m } : = \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \left[ \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } - \mathbf { A } _ { j j } \right] ,\tag{A.113b}
$$

$$
Q _ { A b s } ^ { m } : = \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { A } _ { m j } | \left[ \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } - \mathbf { A } _ { j j } \right] ,\tag{A.113c}
$$

be centered sub-Gaussian quadratic forms. Then $Q _ { l i n } ^ { m } , \ Q _ { E r r , C } ^ { m }$ and $Q _ { A b s } ^ { m }$ satisfy

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ | Q _ { l i n } ^ { m } | \ge t \right\} \lesssim \exp \left( - c \operatorname* { m i n } \left( \frac { t ^ { 2 } } { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 2 } } , \frac { t } { B A _ { \operatorname* { m a x } } } \right) \right) ,\tag{A.114a}
$$

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ | Q _ { E r r , C } ^ { m } | \geq t \right\} \lesssim \exp \left( - c \operatorname* { m i n } \left( \frac { t ^ { 2 } } { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 4 } } , \frac { t } { B A _ { \operatorname* { m a x } } ^ { 2 } } \right) \right) ,\tag{A.114b}
$$

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ | Q _ { A b s } ^ { m } | \ge t \right\} \lesssim \exp \left( - c \operatorname* { m i n } \left( \frac { t ^ { 2 } } { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 2 } } , \frac { t } { B A _ { \operatorname* { m a x } } } \right) \right) ,\tag{A.114c}
$$

for every $t \geq 0$ , respectively. Moreover, the following upper bounds hold

$$
\sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \mathbf { A } _ { j j } \leq A _ { \operatorname* { m a x } } ^ { 2 } , \quad \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { A } _ { m j } | \mathbf { A } _ { j j } \leq A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } .
$$

The proofs of these bounds are deferred to Appendix A.9.

## A.6.3 Self-mapping property

We are now ready to prove Lemma 7.8. It is straightforward to verify that the set $\mathcal { N } ( r _ { H } , r _ { \infty } )$ is non-empty, closed, convex and compact, since $\{ \mathbf { x } _ { m } \} _ { m = 1 } ^ { N }$ linearly spans $\mathbb { R } ^ { d _ { \theta } }$ and $\mathbf { 0 } \in \mathcal { N } ( r _ { H } , r _ { \infty } )$ Consider the following events

$$
\mathcal { E } _ { \infty } : = \left\{ \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \log d \right\} ,
$$

$$
\begin{array} { r } { \mathcal { E } _ { H } : = \left\{ \left| \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } ^ { 2 } - \frac { \mathrm { t r } ( \mathbf { I } _ { d _ { \theta } } ) } { N } \right| ^ { 1 / 2 } \lesssim \sqrt { \frac { B d _ { \theta } \log d } { N } } \right\} , } \end{array}
$$

$$
\mathcal { E } _ { Q , l i n } : = \left\{ \operatorname* { m a x } _ { m \in \left[ N \right] } \big | Q _ { l i n } ^ { m } \big | \lesssim \sqrt { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 2 } \log d } + B A _ { \operatorname* { m a x } } \log d \right\} ,
$$

$$
\mathcal { E } _ { Q , E r r } : = \left\{ \operatorname* { m a x } _ { m \in [ N ] } | Q _ { E r r , C } ^ { m } | \lesssim \sqrt { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 4 } \log d } + B A _ { \operatorname* { m a x } } ^ { 2 } \log d \right\} ,
$$

$$
\mathcal { E } _ { Q , A b s } : = \left\{ \operatorname* { m a x } _ { m \in \left[ N \right] } \big | Q _ { A b s } ^ { m } \big | \lesssim \sqrt { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 2 } \log d } + B A _ { \operatorname* { m a x } } \log d \right\} ,
$$

$$
\begin{array} { r } { \mathcal { E } : = \mathcal { E } _ { \infty } \cap \mathcal { E } _ { H } \cap \mathcal { E } _ { Q , l i n } \cap \mathcal { E } _ { Q , E r r } \cap \mathcal { E } _ { Q , A b s } . } \end{array}
$$

Let $c ^ { \prime } > 5$ be a fixed absolute constant. By Lemma A.4 and a union bound, there exists a suficiently large absolute constant $C > 0$ such that

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { \infty } \} \le | \mathcal { E } _ { s } | \operatorname* { m a x } _ { m \in [ N ] } \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left\{ | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \ge C ( \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \log d ) \right\} \le d ^ { - c ^ { \prime } + 2 } ,
$$

where the first inequality is from the fact that $\begin{array} { r } { \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | = \operatorname* { m a x } _ { \beta \in \mathcal { E } _ { s } } | \mathbf { x } _ { \beta } ^ { \top } \Delta _ { l i n } | } \end{array}$ , and the second inequality is due to $| { \mathcal { E } } _ { s } | \leq d ^ { 2 }$ . Similar bounds can be established for $\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { H } \}$ , $\mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , l i n } \} , \mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , E r r } \}$ , and $\mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , A b s } \}$ by virtue of Lemmas A.4, A.6, respectively. Consequently, we have

$$
\begin{array} { r l } & { \bar { \mathbb { P } } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } \} \leq \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { H } \} + \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { \infty } \} + \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , l i n } \} + \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , E r r } \} + \mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \{ \bar { \mathcal { E } } _ { Q , A b s } \} } \\ & { \qquad \leq 5 d ^ { - c ^ { \prime } + 2 } \leq d ^ { - c ^ { \prime } + 5 } = d ^ { - c } , } \end{array}
$$

and the last inequality follows from $d \geq 2$ . Setting $c ^ { \prime } > 6 , c = c ^ { \prime } - 5$ is enough to ensure $c > 1$

We are now ready to prove inequality (7.87) on the high-probability event $\mathcal { E } ^ { \mathcal { C } } .$ . We proceed with the proof by applying the Taylor expansion in Lemma 7.5 in the first place. For notational simplicity, let $\mathbf { e } _ { e r r } : = \Delta _ { b i a s } + \mathbf { h }$ . Then

$$
Q _ { h } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } r _ { j } ( \Delta _ { l i n } + \Delta _ { b i a s } + \mathbf { h } ) = \underbrace { \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) | \mathbf { x } _ { j } ^ { \top } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) | ^ { 2 } } _ { Q _ { 2 } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) }
$$

$$
\begin{array} { r l } & { + \underbrace { \cfrac { 1 } { 2 N } \displaystyle \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \Psi ^ { ( 4 ) } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) ) ( \mathbf { x } _ { j } ^ { \top } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) ) ^ { 3 } \mathrm { d } s } _ { Q _ { 3 } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) } } \\ & { = Q _ { 2 } ( \Delta _ { l i n } ) + [ Q _ { 2 } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) - Q _ { 2 } ( \Delta _ { l i n } ) ] + Q _ { 3 } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) . } \end{array}
$$

It sufices to derive bounds of the three terms at the r.h.s. of the last equality respectively. Observe that $Q _ { 2 } \big ( \Delta _ { l i n } \big )$ contributes to the second-order bias of the MLE, and the other two terms are higher-order residuals. We proceed in two steps.

STEP $\underline { { \mathbf { 1 } } } \colon \| \cdot \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) }$ Self-mapping. We use the dual-norm definition of $\| \cdot \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } }$ to derive bounds, that is

$$
\| \cdot \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) ^ { - 1 } } : = \operatorname* { s u p } _ { \| \mathbf v \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } = 1 } \mathbf v ^ { \top } ( \cdot ) .
$$

To show $\| \Phi ( { \mathbf { h } } ) \| _ { \nabla ^ { 2 } { \ell } ( \pmb { \theta } ^ { \star } ) } \leq r _ { H }$ , it is enough to find proper upper bounds on

$$
\begin{array} { l } { { \displaystyle H _ { 1 } : = \left\| Q _ { 2 } ( \Delta _ { l i n } ) - \frac { 1 } { 2 N } { \sum _ { j = 1 } ^ { N } } { \bf A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) { \bf x } _ { j } \right\| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } } \ , }  \\ { { \displaystyle H _ { 2 } : = \left\| Q _ { 2 } ( \Delta _ { l i n } + { \bf e } _ { e r r } ) - Q _ { 2 } ( \Delta _ { l i n } ) \right\| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } , \ ~ } } \\ { { \displaystyle H _ { 3 } : = \left\| Q _ { 3 } ( \Delta _ { l i n } + { \bf e } _ { e r r } ) \right\| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } } . } \end{array}
$$

Bound of $H _ { 1 }$ . We prove that on the event of $\mathcal { E } ^ { \mathcal { C } } .$

$$
H _ { 1 } \lesssim \sqrt { \frac { A _ { \mathrm { m a x } } d _ { \theta } } { N } } + \sqrt { A _ { \mathrm { m a x } } \log d } \sqrt { \frac { B d _ { \theta } \log d } { N } } .\tag{A.115}
$$

By the triangle inequality and Lemma A.3

$$
\begin{array} { r l } & { H _ { 1 } \leq \| Q _ { 2 } ( \Delta _ { l i n } ) \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } + \left\| \frac { 1 } { 2 N } \displaystyle \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } \right\| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } } \\ & { \quad \quad = \| Q _ { 2 } ( \Delta _ { l i n } ) \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } + \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } } \\ & { \quad \quad \leq \| Q _ { 2 } ( \Delta _ { l i n } ) \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } + \displaystyle \frac { 1 } { 2 } \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } . } \end{array}\tag{A.116}
$$

Using the relation $\begin{array} { r } { | \mathbf { x } _ { j } ^ { \top } \Delta _ { l i n } | ^ { 2 } = \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } } \end{array}$ , we have

$$
\begin{array} { r l } {  { \| Q _ { 2 } ( \Delta _ { l i n } ) \| \nabla \cdot \boldsymbol { \varepsilon } ( \theta ^ { \star } ) ^ { - 1 } = \underset { \| \nabla \times \| \nabla _ { \xi } Z ( \theta ^ { \star } ) ^ { - 1 } } { \operatorname* { s u p } } \langle \nabla , Q _ { 2 } ( \Delta _ { l i n } ) \rangle \leq \underset { \| \nabla \| _ { \xi } \nabla _ { 2 : ( \theta ^ { \star } ) } ^ { - 1 } } { \operatorname* { s u p } } | \langle \nabla , Q _ { 2 } ( \Delta _ { l i n } ) \rangle | } } \\ & { \leq \underset { \| \nabla \| _ { \nabla ^ { 2 } : ( \theta ^ { \star } ) } ^ { - 1 } } { \operatorname* { s u p } } \frac { 1 } { 1 2 N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { T } \forall | ( \displaystyle \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } ) ^ { 2 } } \\ & { \leq \frac { 1 } { 2 } \operatorname* { m a x } | \displaystyle \sum _ { i = 1 } ^ { N } \mathbf { A } _ { m i } \varepsilon _ { i } | \cdot \underset { \| \nabla \| _ { \xi } ^ { 2 } \varepsilon ( \theta ^ { \star } ) } { \operatorname* { s u p } } = \mathrm { i n } ( \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \star } \mathbf { \nabla } | ^ { 2 } ) ^ { 1 / 2 } \cdot ( \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } ( \displaystyle \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } ) ^ { 2 } ) ^ { 1 / 2 } } \\ &  \leq \frac { 1 } { 2 } \operatorname* { m a x } | \displaystyle \sum _ { i = 1 } ^ { N } \mathbf { A } _ { m i } \varepsilon _ { i } | \cdot \underset  \| \nabla \| _ { \xi } \nabla ^ { 2 } \varepsilon ( \ \end{array}
$$

$$
= \frac { 1 } { 2 } \operatorname* { m a x } _ { m \in \left[ N \right] } \left. \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } \right. \cdot \left. \left. \Delta _ { l i n } \right. \right. _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } .\tag{A.117}
$$

On the other hand, on event $\mathcal { E } _ { \infty } \cap \mathcal { E } _ { H }$

$$
\begin{array} { r l r } {  { \operatorname* { m a x } _ { m \in [ N ] }  \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n }  \cdot \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \leq \operatorname* { m a x } _ { m \in [ N ] }  \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n }  \cdot [  \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } ^ { 2 } - \frac { \mathrm { t r } ( \mathbf { I } _ { d _ { \theta } } ) } { N }  ^ { 1 / 2 } + \sqrt { \frac { \mathrm { t r } ( \mathbf { I } _ { d _ { \theta } } ) } { N } } ] } } \\ & { } & { \lesssim ( \sqrt { A _ { \mathrm { m a x } } \log d } + A _ { \mathrm { m a x } } \log d ) ( \sqrt { \frac { d _ { \theta } } { N } } + \sqrt { \frac { B d _ { \theta } \log d } { N } } ) } \\ & { } & { \lesssim \sqrt { A _ { \mathrm { m a x } } \log d } \sqrt { \frac { B d _ { \theta } \log d } { N } } , \qquad ( \mathrm { A . l i } \mathrm { ~ } ] } \end{array}\tag{8}
$$

where the last inequality follows from the assumption on sample complexity, i.e., $N \gtrsim B \mu _ { Q } d _ { \theta }$ log d.   
A combination of (A.116)–(A.118) gives rise to (A.115).

Bound of $H _ { 2 }$ . We prove that on the event of $\mathcal { E } ^ { \mathcal { C } }$

$$
H _ { 2 } \lesssim \left( \sqrt { A _ { \mathrm { m a x } } \log d } + \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \right) \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } .\tag{A.119}
$$

Observe that

$$
Q _ { 2 } ( \Delta _ { l i n } + { \bf e } _ { e r r } ) - Q _ { 2 } ( \Delta _ { l i n } ) = \underbrace { \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } { \bf x } _ { j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \left( | { \bf x } _ { j } ^ { \top } ( \Delta _ { l i n } + { \bf e } _ { e r r } ) | ^ { 2 } - | { \bf x } _ { j } ^ { \top } ( \Delta _ { l i n } ) | ^ { 2 } \right) } _ { H _ { 2 , 1 } }
$$

For $H _ { 2 , 1 }$ , recall $| \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) | \leq v _ { j } ^ { \star }$ , then use the dual norm method again

$$
\begin{array} { r l } & { \| H _ { 2 , 1 } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { * } ) ^ { - 1 } } \leq \underset { \| \mathbf { v } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { * } ) } = 1 } { \operatorname* { s u p } } \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { * } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | \mathbf { x } _ { j } ^ { \top } \boldsymbol { \Delta } _ { l i n } | \cdot | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | } \\ & { \leq \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \boldsymbol { \Delta } _ { l i n } | \cdot \underset { \| \mathbf { v } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { * } ) } = 1 } { \operatorname* { s u p } } \left( \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { * } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | ^ { 2 } \right) ^ { 1 / 2 } \cdot \left( \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { * } | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | ^ { 2 } \right) ^ { 1 / 2 } } \\ & { = \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \boldsymbol { \Delta } _ { l i n } | \cdot \| \mathbf { e } _ { e r r } \| \nabla ^ { 2 } \ell ( \theta ^ { * } ) \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } \| \mathbf { e } _ { e r r } \| \mathbf { v } ^ { 2 } \ell ( \theta ^ { * } ) , } \end{array}
$$

where the last step follows from the event $\mathcal { E } _ { \infty }$ and the fact that $N \gtrsim B \mu _ { Q } d _ { \theta }$ log d. For the deterministic term $H _ { 2 , 2 }$ , we have

$$
\begin{array} { r l } & { \| H _ { 2 , 2 } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } \leq \underset { \| \mathbf { v } \| _ { \infty } \ell ( \theta ^ { \star } ) ^ { - 1 } } { \operatorname* { s u p } } \frac { 1 } { 1 } \frac { 1 } { 2 N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | ^ { 2 } } \\ & { \qquad \leq \frac { 1 } { 2 } \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \cdot \left( \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | ^ { 2 } \right) ^ { 1 / 2 } = \frac { 1 } { 2 } \| \mathbf { e } _ { e r r } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | . } \end{array}\tag{A.120}
$$

By combining $H _ { 2 , 1 }$ and $H _ { 2 , 2 }$ , we arrive at (A.119).

Bound of $H _ { 3 }$ . We prove that on the event of $\mathcal { E } ^ { \mathcal { C } }$

$$
H _ { 3 } \lesssim A _ { \mathrm { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } + \left( \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \right) ^ { 2 } \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } .\tag{A.121}
$$

First, observe that for each $j \in [ N ]$ , the coeficient $Q _ { 3 , j } ^ { c }$ in $Q _ { 3 } \big ( \Delta _ { l i n } + { \bf e } _ { e r r } \big )$ admits the following bound

$$
\begin{array} { r l } {  { | Q _ { 3 , j } ^ { c } | : = | \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } \Psi ^ { ( 4 ) } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) \big ) \big ( \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) \big ) ^ { 3 } \mathrm { d } s | } } \\ & { \le \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } | \Psi ^ { ( 4 ) } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) \big ) | \cdot | \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) | ^ { 3 } \mathrm { d } s } \\ & { \le \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } \Psi ^ { \prime \prime } ( \eta _ { j } ^ { \star } + s \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) ) \cdot | \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) | ^ { 3 } \mathrm { d } s } \\ & { \le \int _ { 0 } ^ { 1 } ( 1 - s ) ^ { 2 } \exp \bigg ( s \operatorname* { m a x } _ { m \in [ N ] } \Big | \mathbf { x } _ { m } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) \Big | \bigg ) v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) | ^ { 3 } \mathrm { d } s } \\ &  \le \frac { 1 } { \sigma } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \big ( \Delta _ { l i n } + \mathbf { e } _  \end{array}\tag{A.122}
$$

where the second last inequality is from the pseudo self-concordance property introduced in Lemma $7 . 7 .$ , and the last inequality is from the assumption on sample complexity, i.e., $N \gtrsim$ $B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d ,$ under which

$$
\begin{array} { r l } & { \begin{array} { l } { \underset { m \in [ N ] } { \operatorname* { m a x } } \left| { \mathbf { x } } _ { m } ^ { \top } ( \Delta _ { l i n } + { \mathbf { e } } _ { e r r } ) \right| \leq \underset { m \in [ N ] } { \operatorname* { m a x } } \left| { \mathbf { x } } _ { m } ^ { \top } \Delta _ { l i n } \right| + \underset { m \in [ N ] } { \operatorname* { m a x } } \left| { \mathbf { x } } _ { m } ^ { \top } \Delta _ { b i a s } \right| + \underset { m \in [ N ] } { \operatorname* { m a x } } \left| { \mathbf { x } } _ { m } ^ { \top } { \mathbf { h } } \right| } \\ & { \qquad \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \log d + \frac 1 2 A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + r _ { \infty } < 1 . } \end{array} } \end{array}
$$

Here, the second inequality is from event $\mathcal { E } _ { \infty }$ and Lemma A.2. Next, substituting (A.122) back into $H _ { 3 }$ yields

$$
\begin{array} { l } { \displaystyle { H _ { 3 } = \left\| \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } { \mathbf { x } _ { j } Q _ { 3 , j } ^ { \mathrm { c } } } \right\| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } } \leq \underset { | \mathbf { v } | | _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } = 1 } { \operatorname* { s u p } } \frac { 1 } { 2 N } \underset { j = 1 } { \overset { N } { \sum } } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | Q _ { 3 , j } ^ { \mathrm { c } } | } } \\ { \displaystyle { \leq \underset { | \mathbf { v } | | _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } = 1 } { \operatorname* { s u p } } \frac { 1 } { 4 N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | \mathbf { x } _ { j } ^ { \top } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) | ^ { 3 } } } \\ { \displaystyle { \leq \underset { | \mathbf { v } | | _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } = 1 } { \operatorname* { s u p } } \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | \mathbf { x } _ { j } ^ { \top } \Delta _ { l i n } | ^ { 3 } } + \underset { | \mathbf { v } | | _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } = 1 } { \operatorname* { s u p } } \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | \cdot | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | ^ { 3 } , }  \end{array}\tag{A.123}
$$

where the last inequality follows from $| a + b | ^ { 3 } \leq 4 | a | ^ { 3 } + 4 | b | ^ { 3 }$ . The terms $H _ { 3 } ^ { 1 } , H _ { 3 } ^ { 2 }$ can be estimated in a similar manner as (A.117) (A.118) and (A.120) respectively. That is, on event $\mathcal { E } _ { \infty } \cap \mathcal { E } _ { H }$

$$
\begin{array} { r l } & { H _ { 3 } ^ { 1 } \leq \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \right) ^ { 2 } \underset { \| \mathbf { v } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { * } ) } = 1 } { \operatorname* { s u p } } \left( \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \mathbf { v } | ^ { 2 } \right) ^ { 1 / 2 } \left( \frac { 1 } { N } \underset { j = 1 } { \overset { N } { \sum } } v _ { j } ^ { \star } | \mathbf { x } _ { j } ^ { \top } \Delta _ { l i n } | ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \quad \leq \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | \right) ^ { 2 } \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { * } ) } \lesssim A _ { \operatorname* { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } , } \end{array}
$$

where the last step is from the assumption on sample complexity, i.e., $N \gtrsim B \mu _ { Q } d _ { \theta }$ log d. Likewise,

$$
H _ { 3 } ^ { 2 } \leq \left( \underset { m \in [ N ] } { \operatorname* { m a x } } ~ | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \right) ^ { 2 } \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } .
$$

Combining the above inequalities, we obtain (A.121).

Combining the Bounds: To complete STEP 1, we verify $\| \Phi ( { \bf h } ) \| _ { \nabla ^ { 2 } \ell ( \pmb \theta ^ { \star } ) } \leq r _ { H }$ for any $\mathbf { h } \in$ $\mathcal { N } ( r _ { H } , r _ { \infty } )$ on event $\mathcal { E } ^ { \mathcal { C } } .$ , that is

$$
\begin{array} { r l } & { \left. { \Phi ( \mathbf { h } ) } \right. \left. { \nabla ^ { \scriptscriptstyle 2 } \ell ( \theta ^ { \star } ) } \right. + H _ { 1 } + H _ { 2 } + H _ { 3 } } \\ & { \stackrel { ( i ) } { \lesssim } \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } + \sqrt { A _ { \operatorname* { m a x } } \log d } \sqrt { \frac { B d _ { \theta } \log d } { N } } + A _ { \operatorname* { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } } \\ & { ~ + \left( \sqrt { A _ { \operatorname* { m a x } } \log d } + \underbrace { \operatorname* { m a x } } _ { m \in [ N ] } \left. { \bf x } _ { m } ^ { \top } { \bf e } _ { e r r } \right. + \left( \underset { m \in [ N ] } { \operatorname* { m a x } } \left. { \bf x } _ { m } ^ { \top } { \bf e } _ { e r r } \right. \right) ^ { 2 } \right) \left. { \bf e } _ { e r r } \right. _ { \nabla ^ { \scriptscriptstyle 2 } \ell ( \theta ^ { \star } ) } } \\ & { ~ { \stackrel { ( i i ) } { \lesssim } } ~ \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } + \sqrt { A _ { \operatorname* { m a x } } \log d } \sqrt { \frac { B d _ { \theta } \log d } { N } } + A _ { \operatorname* { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } + \frac { 1 } { 4 } \left. { \bf e } _ { e r r } \right. _ { \nabla ^ { \scriptscriptstyle 2 } \ell ( \theta ^ { \star } ) } } \\ &  ~ { \stackrel { ( i i i ) } { \lesssim } } ~ \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } + \sqrt { A _ { \operatorname* { m a x } } \log d } \sqrt { \frac { B d _ { \theta } \log d } { N } } + A _ { \operatorname* { m a x } } \log d \sqrt { \frac { B d _ { \theta } \log d } { N } } + \frac { 1 } { 4 } r _ { H } \leq r _  \end{array}
$$

Here, step (i) is obtained by (A.115), (A.119), and (A.121). Step (ii) is obtained by setting N suficiently large, the bracketed term is smaller than <sup>1</sup><sub>4</sub> , i.e.,

$$
\begin{array} { r l } { \sqrt { A _ { \mathrm { m a x } } \log d } + \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | + \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \right) ^ { 2 } \lesssim \sqrt { A _ { \mathrm { m a x } } \log d } + \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | } & { } \\ { \lesssim \sqrt { \frac { B \mu Q d \theta \log d } { N } } + \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \Delta _ { b i a s } | + \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } \mathbf { h } | } & { } \end{array}
$$

$$
\mathrm { ( b y ~ L e m m a ~ A . 2 ) } \quad \mathrm { \lesssim } \quad \sqrt { \frac { B \mu _ { Q } d _ { \theta } \log d } { N } } + \frac { B \mu _ { Q } d _ { \theta } ^ { 3 / 2 } } { N } + r _ { \infty } \leq \frac { 1 } { 4 } ,
$$

when $N \gtrsim B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 }$ log d. Finally, step (iii) is established by observing

$$
\| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \leq \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } + \| \mathbf { h } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \leq \frac { 1 } { 2 } \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } + r _ { H } ,
$$

where the last inequality is from Lemma A.3.

STEP 2: $\| \cdot \| _ { Q , \infty }$ Self-mapping. We aim to find upper bounds on the following three terms

$$
I _ { 1 } : = \operatorname* { m a x } _ { m \in [ N ] } \left. \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \left[ Q _ { 2 } ( \Delta _ { l i n } ) - \frac { 1 } { 2 N } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { j j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \mathbf { x } _ { j } \right] \right. ,\tag{A.124a}
$$

$$
I _ { 2 } : = \operatorname* { m a x } _ { m \in [ N ] } \Big | \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \big [ Q _ { 2 } \big ( \Delta _ { l i n } + \mathbf { e } _ { e r r } \big ) - Q _ { 2 } \big ( \Delta _ { l i n } \big ) \big ] \Big | ,\tag{A.124b}
$$

$$
I _ { 3 } : = \operatorname* { m a x } _ { m \in [ N ] } \bigg | \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } Q _ { 3 } ( \Delta _ { l i n } + \mathbf { e } _ { e r r } ) \bigg | .\tag{A.124c}
$$

Bound of $I _ { 1 }$ : By plugging the definition of $Q _ { 2 } ( \Delta _ { l i n } )$ in $I _ { 1 }$ , we obtain

$$
\mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } Q _ { 2 } ( \Delta _ { l i n } ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } .
$$

The upper bound follows directly from (A.114a), i.e., on event $\mathcal { E } _ { Q , l i n }$ , we have

$$
I _ { 1 } \lesssim \sqrt { B ^ { 2 } A _ { \operatorname* { m a x } } ^ { 2 } \log d } + B A _ { \operatorname* { m a x } } \log d \lesssim B A _ { \operatorname* { m a x } } \log d .\tag{A.125}
$$

Bound of $I _ { 2 } \colon$ From direct calculation

$$
I _ { 2 } \leq \underbrace { \operatorname* { m a x } _ { m \in [ N ] } \left| \sum _ { j = 1 } ^ { N } { \mathbf { A } _ { m j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) \left( \sum _ { i = 1 } ^ { N } { \mathbf { A } _ { j i } \varepsilon _ { i } } \right) \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } } \right| } _ { I _ { 2 } ^ { 1 } } + \underbrace { \operatorname* { m a x } _ { m \in [ N ] } \left| \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } { \mathbf { A } _ { m j } \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) ( \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } ) ^ { 2 } } \right| } _ { I _ { 2 } ^ { 2 } } .
$$

The term $I _ { 2 } ^ { 1 }$ can be bounded using the Cauchy–Schwarz inequality, (A.114b), and a union bound. That is, on event $\mathcal { E } Q , E r r$

$$
\begin{array} { l } { I _ { 2 } ^ { 1 } \le \displaystyle \operatorname* { m a x } _ { m \in [ N ] } \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { l = 1 } ^ { N } v _ { l } ^ { \star } ( \mathbf { x } _ { l } ^ { \top } \mathbf { e } _ { e r r } ) ^ { 2 } \right) ^ { 1 / 2 } } \\ { \quad \lesssim \sqrt { B A _ { \operatorname* { m a x } } ^ { 2 } \log d } \cdot \sqrt { N } \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } . } \end{array}
$$

The upper bound for the deterministic part $I _ { 2 } ^ { 2 }$ is meanwhile straightforward

$$
\begin{array} { l } { { \displaystyle I _ { 2 } ^ { 2 } \leq \operatorname* { m a x } _ { m \in [ N ] } \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } | \mathbf { A } _ { m j } | v _ { j } ^ { \star } ( \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } ) ^ { 2 } \leq \frac { 1 } { 2 } \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } ( \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } ) ^ { 2 } \right) ^ { 1 / 2 } } } \\ { { \displaystyle \quad \leq \frac { 1 } { 2 } \sqrt { A _ { m a x } N } \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \cdot \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } } . } \end{array}
$$

Bound of $I _ { 3 } { \mathrm { : } }$ From (A.122), (A.123), and setting N suficiently large, we have

$$
I _ { 3 } = \operatorname* { m a x } _ { m \in [ N ] } \left. \frac { 1 } { 2 } \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } Q _ { 3 , j } ^ { c } \right. \leq \underbrace { \operatorname* { m a x } _ { m \in [ N ] } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \lvert \mathbf { A } _ { m j } \rvert \cdot \lvert \mathbf { x } _ { j } ^ { \top } \Delta _ { l i n } \rvert ^ { 3 } } _ { I _ { 3 } ^ { 1 } } + \underbrace { \operatorname* { m a x } _ { m \in [ N ] } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \lvert \mathbf { A } _ { m j } \rvert \cdot \lvert \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } \rvert ^ { 3 } } _ { I _ { 3 } ^ { 2 } } .
$$

The upper bounds on $I _ { 3 } ^ { 1 } , I _ { 3 } ^ { 2 }$ are relatively easy to establish. On event $\mathcal { E } _ { \infty } \cap \mathcal { E } _ { Q } $ ,Abs

$$
\begin{array} { r l r } {  { I _ { 3 } ^ { 1 } \leq \operatorname* { m a x } _ { l \in [ N ] } | \mathbf { x } _ { l } ^ { \top } \Delta _ { l i n } | \cdot \operatorname* { m a x } _ { m \in [ N ] } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { A } _ { m j } | ( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } ) ^ { 2 } } } \\ & { } & { \lesssim ( \sqrt { A _ { \operatorname* { m a x } } \log d } + A _ { \operatorname* { m a x } } \log d ) ( B A _ { \operatorname* { m a x } } \log d + A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } ) \quad \mathrm { ( b y ~ L e m m a ~ \Delta A . ~ 4 ~ a n d ~ ( A . 1 1 4 c ) ) } } \\ & { } & { \lesssim \sqrt { A _ { \operatorname* { m a x } } \log d } . \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \mathrm { ( b y ~ \mathcal { N } \gtrsim B ^ { 2 } \mu \mathcal { Q } ^ { d _ { \theta } ^ { 3 / 2 } } \log d ) } } \end{array}
$$

Moreover,

$$
\begin{array} { r l r } {  { I _ { 3 } ^ { 2 } \leq ( \operatorname* { m a x } _ { l \in [ N ] } \boldsymbol { | \mathbf { x } } _ { l } ^ { \top } \mathbf { e } _ { e r r } ) ^ { 2 } \cdot \operatorname* { m a x } _ { m \in [ N ] } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { A } _ { m j } | \cdot | \mathbf { x } _ { j } ^ { \top } \mathbf { e } _ { e r r } | } } \\ & { \leq ( \operatorname* { m a x } _ { l \in [ N ] } | \mathbf { x } _ { l } ^ { \top } \mathbf { e } _ { e r r } | ) ^ { 2 } \sqrt { A _ { \operatorname* { m a x } } N } \| \mathbf { e } _ { e r r } \| _ { \mathbb { V } ^ { 2 } \ell ( \theta ^ { \star } ) } . } & { \quad \mathrm { ( b y ~ C a u c h y - S c h w a r z ~ a n d ~ L e m m a ~ A . l ) } } \end{array}
$$

Combining the Bounds: The final step of this part is to verify ma $\mathrm { x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Phi ( \mathbf { h } ) | \leq r _ { \infty }$ for any $\mathbf { h } \in \mathcal { N } ( r _ { H } , r _ { \infty } )$ on event ${ \mathcal { E } } .$ . We proceed by first showing

$$
\begin{array} { r l } { \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \sqrt { A _ { \operatorname* { m a x } } N } \le \| \mathbf { e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \sqrt { A _ { \operatorname* { m a x } } B N } } & { } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \le \left( \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } + \| \mathbf { h } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \right) \sqrt { A _ { \operatorname* { m a x } } B N } } & { } \\ { \quad \quad \quad \quad \mathrm { ( b y ~ L e m m a ~ A . 3 ) } } & { { } \quad \lesssim A _ { \operatorname* { m a x } } \sqrt { B d _ { \theta } } + B A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } \log d + B A _ { \operatorname* { m a x } } ^ { 3 / 2 } \sqrt { d _ { \theta } } ( \log d ) ^ { 3 / 2 } < \frac { 1 } { 4 } , } \end{array}
$$

when $N \gtrsim B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 2 } d .$ , and

$$
\operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \mathbf { e } _ { e r r } | \leq \operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Delta _ { b i a s } | + r _ { \infty } \lesssim A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } + r _ { \infty } < 1 , \quad \mathrm { ( b y ~ L e m m a ~ A . 2 ) }
$$

when $N \gtrsim B ^ { 2 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 2 } d .$ Then, on the event of $\mathcal { E }$

$$
\operatorname* { m a x } _ { m \in [ N ] } | \mathbf { x } _ { m } ^ { \top } \Phi ( \mathbf { h } ) | \leq I _ { 1 } + I _ { 2 } + I _ { 3 }
$$

$$
\begin{array} { r l } & { \lesssim B A _ { \mathrm { m a x } } \log d + \sqrt { A _ { \mathrm { m a x } } \log d } + \| { \bf e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \sqrt { A _ { \mathrm { m a x } } N } \times } \\ & { \quad \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } { \bf e } _ { e r r } | + \left( \underset { l \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { l } ^ { \top } { \bf e } _ { e r r } | \right) ^ { 2 } \right) + \| { \bf e } _ { e r r } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } \sqrt { A _ { \mathrm { m a x } } B N } \sqrt { A _ { \mathrm { m a x } } \log d } } \\ & { \lesssim B A _ { \mathrm { m a x } } \log d + \sqrt { A _ { \mathrm { m a x } } \log d } + \frac { 1 } { 4 } \left( \underset { m \in [ N ] } { \operatorname* { m a x } } | \mathbf { x } _ { m } ^ { \top } { \bf e } _ { e r r } | + \sqrt { A _ { \mathrm { m a x } } \log d } \right) } \\ & { \lesssim B A _ { \mathrm { m a x } } \log d + \sqrt { A _ { \mathrm { m a x } } \log d } + A _ { \mathrm { m a x } } \sqrt { d _ { \theta } } + \frac { 1 } { 4 } r _ { \infty } \leq r _ { \infty } . } \end{array}
$$

By combining STEP 1 and STEP 2, we conclude that on the event $\mathcal { E }$ with $\mathbb { P } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \{ \mathcal { E } \} \geq 1 - d ^ { - c }$ uniformly over h $\in \mathcal { N } ( r _ { H } , r _ { \infty } )$ , the event $\Phi ( \mathbf { h } ) \in \mathcal { N } ( r _ { H } , r _ { \infty } )$ occurs, this proves (7.87). □

## A.7 Proof of Lemma A.1

Part (i). By (2.20b), $\begin{array} { r } { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) = \frac { 1 } { N } \mathbf { X } ^ { \top } \mathbf { V } ^ { \star } \mathbf { X } } \end{array}$ , where X and $\mathbf { V } ^ { \star }$ are defined in (7.74). Thus

$$
\begin{array} { l } { { \displaystyle { \bf A } { \bf V } ^ { \star } { \bf A } = \frac { 1 } { N ^ { 2 } } { \bf X } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ( { \bf X } ^ { \top } { \bf V } ^ { \star } { \bf X } ) \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } { \bf X } ^ { \top } } \ ~ } \\ { { \displaystyle ~ = \frac { 1 } { N ^ { 2 } } { \bf X } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ( N \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ) \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } { \bf X } ^ { \top } = \frac { 1 } { N } { \bf X } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } { \bf X } ^ { \top } = { \bf A } } . } \end{array}
$$

By the symmetry of A, $\begin{array} { r } { \mathbf { A } _ { i i } = \mathbf { e } _ { i } ^ { \top } \mathbf { A } \mathbf { e } _ { i } = \mathbf { e } _ { i } ^ { \top } \mathbf { A } \mathbf { V } ^ { \star } \mathbf { A } \mathbf { e } _ { i } = \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { i j } ^ { 2 } } \end{array}$ . Moreover, by the definition of A and the cyclic property of trace

$$
\sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } = \operatorname { t r } ( \mathbf { V } ^ { \star } \mathbf { A } ) = \operatorname { t r } \left( \mathbf { V } ^ { \star } { \frac { 1 } { N } } \mathbf { X } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \mathbf { X } ^ { \mathsf { T } } \right) = \operatorname { t r } \left( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } { \frac { 1 } { N } } \mathbf { X } ^ { \mathsf { T } } \mathbf { V } ^ { \star } \mathbf { X } \right) = d _ { \theta } ,\tag{A.126}
$$

where $d _ { \theta }$ is defined in (2.11).

Part (ii). Since A is symmetric, and $\mathbf { A } \succeq \mathbf { 0 }$ , there exists some B such that $\mathbf { A } = \mathbf { B } \mathbf { B } ^ { \top }$ . Thus $\begin{array} { r } { | \mathbf { A } _ { i j } | = | \mathbf { e } _ { i } ^ { \top } \mathbf { B } \mathbf { B } ^ { \top } \mathbf { e } _ { j } | \leq \| \mathbf { B } ^ { \top } \mathbf { e } _ { i } \| _ { 2 } \| \mathbf { B } ^ { \top } \mathbf { e } _ { j } \| _ { 2 } = \sqrt { \mathbf { A } _ { i i } \mathbf { A } _ { j j } } \leq \operatorname* { m a x } _ { i \in [ N ] } \mathbf { A } _ { i i } } \end{array}$ . The upper bound on $\operatorname* { m a x } _ { i \in [ N ] } { \bf A } _ { i i }$ is by Definition 3.2, and Proposition 2.2. □

## A.8 Proof of Lemma A.4

We prove inequalities (A.110) and (A.111) by virtue of Bernstein’s inequality and Hanson-Wright inequality (restated in Lemma B.2), respectively.

Proof of (A.110). Observe that

$$
| \mathbf { x } _ { m } ^ { \top } \Delta _ { l i n } | = \left| \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \nabla \ell ( \pmb { \theta } ^ { \star } ) \right| = \left| \mathbf { x } _ { m } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \varepsilon _ { j } \right| = \left| \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } \varepsilon _ { j } \right| .
$$

$\{ \mathbf { A } _ { m j } \varepsilon _ { j } \} _ { j \in [ N ] }$ are zero-mean, bounded random variables, i.e.,

$$
| \mathbf { A } _ { m j } \varepsilon _ { j } | \leq | \mathbf { A } _ { m j } | \leq A _ { \operatorname* { m a x } } , a . s . ,
$$

where the last inequality is by Lemma A.1. The summation of these random variables has a bounded variance in that

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \left( \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } \varepsilon _ { j } \right) ^ { 2 } \right] = \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } ^ { 2 } v _ { j } ^ { \star } = \mathbf { A } _ { m m } \leq A _ { \operatorname* { m a x } } .
$$

With the established bounds, we can obtain (A.110) by virtue of Bernstein’s inequality (see, e.g., [73, Theorem 2.8.4]).

Proof of (A.111). By a simple maneuver, we can obtain the following relationship

$$
\left| \| \Delta _ { l i n } \| _ { \nabla ^ { 2 } \ell ( \theta ^ { \star } ) } ^ { 2 } - \frac { \mathrm { t r } ( { \bf I } _ { d _ { \theta } } ) } { N } \right| = \frac { 1 } { N } \Bigg | \pmb { \xi } ^ { \top } { \bf P } \pmb { \xi } - \mathrm { t r } ( { \bf P } ) \Bigg | ,
$$

for the isotropic, $\sqrt { B }$ sub-Gaussian random vector $\begin{array} { r } { \pmb { \xi } : = \left\lceil \frac { \varepsilon _ { 1 } } { \sqrt { v _ { 1 } ^ { \star } } } , \ldots , \frac { \varepsilon _ { N } } { \sqrt { v _ { N } ^ { \star } } } \right\rceil ^ { \top } , \mathrm { ~ i . e . , ~ } \| \pmb { \xi } \| _ { \psi _ { 2 } } \lesssim \sqrt { B } . } \end{array}$ Since tr $\left( \mathbf { I } _ { d _ { \theta } } \right) = \operatorname { t r } ( \mathbf { P } ) = d _ { \theta }$ , we can directly verify that $\begin{array} { r } { \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \pmb { \xi } ^ { \top } \mathbf { P } \pmb { \xi } \right] = \operatorname { t r } ( \mathbf { P } ) , \| \mathbf { P } \| _ { F } ^ { 2 } = \operatorname { t r } ( \mathbf { P } ^ { 2 } ) = } \end{array}$ $d _ { \theta }$ , and $\| \mathbf { P } \| \leq 1$ . Applying Lemma B.2 to this quadratic form yields

$$
\mathbb { P } _ { \theta ^ { \star } } ^ { ( N ) } \left( \frac { 1 } { N } \bigg | \xi ^ { \top } \mathbf { P } \xi - \mathrm { t r } ( \mathbf { P } ) \bigg | \geq t \right) \lesssim \exp \left( - c \operatorname* { m i n } \left\{ \frac { N ^ { 2 } t ^ { 2 } } { d _ { \theta } B ^ { 2 } } , \frac { N t } { B } \right\} \right) ,
$$

which implies (A.111).

## A.9 Proof of Lemma A.6

Proof of (A.114a). By plain calculation,

$$
\mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \left( \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } \varepsilon _ { i } \right) ^ { 2 } \right] = \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } ^ { 2 } \varepsilon _ { i } ^ { 2 } \right] = \sum _ { i = 1 } ^ { N } \mathbf { A } _ { j i } ^ { 2 } v _ { i } ^ { \star } = \mathbf { A } _ { j j } ,
$$

where the first equality is due to the independence of samples. Consider

$$
{ \bf D } : = \mathrm { d i a g } \left( \left( \frac { \Psi ^ { ( 3 ) } ( \eta _ { i } ^ { \star } ) } { v _ { i } ^ { \star } } { \bf A } _ { m i } \right) _ { i \le N } \right) , \quad { \bf M } _ { m } : = { \bf P } { \bf D } { \bf P } ,
$$

where P is defined in (A.107), and the isotropic sub-Gaussian random vector $\begin{array} { r } { \pmb { \xi } : = \left[ \frac { \varepsilon _ { 1 } } { \sqrt { v _ { 1 } ^ { \star } } } , . . . , \frac { \varepsilon _ { N } } { \sqrt { v _ { N } ^ { \star } } } \right] ^ { \top } } \end{array}$ Then $Q _ { l i n } ^ { m }$ admits the following equivalent form

$$
Q _ { l i n } ^ { m } = \frac { 1 } { 2 } \left( \pmb { \xi } ^ { \top } \mathbf { M } _ { m } \pmb { \xi } - \mathrm { t r } ( \mathbf { M } _ { m } ) \right) .\tag{A.127}
$$

Since $\begin{array} { r } { D _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { j \in [ N ] } \left| { \frac { \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) } { v _ { j } ^ { \star } } } \mathbf { A } _ { m j } \right| \ \leq \ A _ { \operatorname* { m a x } } } \end{array}$ by Lemma A.1, our focus is on controlling $\operatorname { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } )$ , that is

$$
\mathrm { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } ) = \sum _ { j = 1 } ^ { N } \left( \frac { \Psi ^ { ( 3 ) } \bigl ( \eta _ { j } ^ { \star } \bigr ) } { v _ { j } ^ { \star } } \mathbf { A } _ { m j } \right) ^ { 2 } v _ { j } ^ { \star } \mathbf { A } _ { j j } \leq \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } \mathbf { A } _ { m j } ^ { 2 } \leq A _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \leq A _ { \operatorname* { m a x } } ^ { 2 } ,
$$

where the first inequality is due to $| \Psi ^ { ( 3 ) } ( \eta _ { j } ^ { \star } ) | \le v _ { j } ^ { \star }$ . With the established bounds of $D _ { \mathrm { m a x } }$ and $\operatorname { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } )$ , we can obtain the tail bound (A.114a) via inequality (A.112).

Proof of (A.114b). First, notice that

$$
\sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \mathbf { A } _ { j j } \leq A _ { \operatorname* { m a x } } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } = A _ { \operatorname* { m a x } } \mathbf { A } _ { m m } \leq A _ { \operatorname* { m a x } } ^ { 2 } ,
$$

which yields the deterministic upper bound. Again, we can identify the matrix

$$
\mathbf { D } : = \mathrm { d i a g } \left( \mathbf { A } _ { m 1 } ^ { 2 } , \ldots , \mathbf { A } _ { m N } ^ { 2 } \right) , \quad \mathbf { M } _ { m , 2 } : = \mathbf { P D P } ,
$$

and the isotropic sub-Gaussian random vector $\begin{array} { r } { \pmb { \xi } : = \left[ \frac { \varepsilon _ { 1 } } { \sqrt { v _ { 1 } ^ { \star } } } , \ldots , \frac { \varepsilon _ { N } } { \sqrt { v _ { N } ^ { \star } } } \right] ^ { \top } } \end{array}$ such that $\| \pmb { \xi } \| _ { \psi _ { 2 } } \lesssim \sqrt { B } .$ and

$$
\begin{array} { r } { Q _ { E r r , C } ^ { m } = \pmb { \xi } ^ { \top } \mathbf { M } _ { m , 2 } \pmb { \xi } - \mathrm { t r } ( \mathbf { M } _ { m , 2 } ) . } \end{array}
$$

Moreover $\begin{array} { r } { D _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ N ] } \mathbf { A } _ { m i } ^ { 2 } \leq A _ { \operatorname* { m a x } } ^ { 2 } , } \end{array}$ and

$$
\mathrm { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } ) = \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } ^ { 4 } v _ { j } ^ { \star } \mathbf { A } _ { j j } \leq A _ { \operatorname* { m a x } } ^ { 2 } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } \mathbf { A } _ { m j } ^ { 2 } \leq A _ { \operatorname* { m a x } } ^ { 4 } .
$$

Using inequality (A.112), we obtain (A.114b).

Proof of (A.114c). The deterministic upper bound is from Cauchy-Schwarz inequality,

$$
\sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } | \mathbf { A } _ { m j } | \mathbf { A } _ { j j } \leq \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { m j } ^ { 2 } \right) ^ { 1 / 2 } \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { A } _ { j j } ^ { 2 } \right) ^ { 1 / 2 } \leq \sqrt { A _ { \operatorname* { m a x } } } \sqrt { A _ { \operatorname* { m a x } } d _ { \theta } } = A _ { \operatorname* { m a x } } \sqrt { d _ { \theta } } .
$$

The tail bound is from identifying the matrix

$$
\mathbf M _ { m , | \cdot | } : = \mathbf P \mathrm { d i a g } \left( | \mathbf A _ { m 1 } | , \dots , | \mathbf A _ { m N } | \right) \mathbf P ,
$$

the isotropic sub-Gaussian random vector $\begin{array} { r } { \pmb { \xi } : = \left[ \frac { \varepsilon _ { 1 } } { \sqrt { v _ { 1 } ^ { \star } } } , \ldots , \frac { \varepsilon _ { N } } { \sqrt { v _ { N } ^ { \star } } } \right] ^ { \top } , \mathrm { i . e . , } \| \pmb { \xi } \| _ { \psi _ { 2 } } \lesssim \sqrt { B } . } \end{array}$ such that

$$
Q _ { A b s } ^ { m } = \xi ^ { \top } \mathbf { M } _ { m , | \cdot | } \xi - \mathrm { t r } ( \mathbf { M } _ { m , | \cdot | } ) .
$$

Then $D _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { j \in [ N ] } | \mathbf { A } _ { m j } | \leq A _ { \operatorname* { m a x } } .$ , and

$$
\mathrm { t r } ( \mathbf { D } ^ { 2 } \mathbf { P } ) = \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } ^ { 2 } \mathbf { P } _ { j j } = \sum _ { j = 1 } ^ { N } \mathbf { A } _ { m j } ^ { 2 } v _ { j } ^ { \star } \mathbf { A } _ { j j } \leq \mathbf { A } _ { m m } A _ { \operatorname* { m a x } } \leq A _ { \operatorname* { m a x } } ^ { 2 } .
$$

Using inequality (A.112), we conclude the proof.

## A.10 Proof of Corollary 4.2

It sufices to control $\| \Delta _ { l i n } \| _ { 2 }$ and $\lVert \Delta _ { b i a s } \rVert _ { 2 }$ . The latter can be directly estimated from $\| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) }$

$$
\begin{array} { r l } & { \| \Delta _ { b i a s } \| _ { 2 } \leq \sqrt { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } \| \Delta _ { b i a s } \| _ { \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) } \lesssim \sqrt { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } \sqrt { \frac { A _ { \operatorname* { m a x } } d _ { \theta } } { N } } } \\ & { \qquad \lesssim \sqrt { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } \frac { \sqrt { B \mu _ { Q } } d _ { \theta } } { N } , } \end{array}\tag{A.128}
$$

where the second and last inequalities follow by Lemmas A.3 and A.1, respectively. The upper bound on $\| \Delta _ { l i n } \| _ { 2 }$ can be derived from the matrix Bernstein inequality, restated here as Lemma B.4. Recall that $\begin{array} { r } { \nabla \ell ( \pmb { \theta } ^ { \star } ) = - \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \mathbf { x } _ { j } \varepsilon _ { j } } \end{array}$ , and $\begin{array} { r } { \nabla ^ { 2 } \ell \bigl ( \pmb { \theta } ^ { \star } \bigr ) = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { x } _ { j } \mathbf { x } _ { j } ^ { \top } } \end{array}$ , from which $\Delta _ { l i n }$ can be written as the sum of zero-mean, independent random vectors,

$$
\Delta _ { l i n } = \sum _ { j = 1 } ^ { N } \frac { 1 } { N } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \mathbf { x } _ { j } \varepsilon _ { j } : = \sum _ { j = 1 } ^ { N } \mathbf { z } _ { j } .
$$

The random vector $\mathbf { z } _ { j } \in \mathbb { R } ^ { d _ { \theta } }$ has Euclidean norm bounded a.s.,

$$
\| \mathbf { z } _ { j } \| _ { 2 } \leq \sqrt { \frac { 1 } { N ^ { 2 } } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } } \leq \sqrt { \frac { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N } \mathbf { A } _ { j j } } \leq \sqrt { \frac { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| A _ { \operatorname* { m a x } } } { N } } .
$$

And the variance terms satisfy

$$
\bigg \| \sum _ { j = 1 } ^ { N } \mathbb { E } _ { \pmb { \theta } ^ { \star } } ^ { ( N ) } \left[ \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right] \bigg \| = \bigg \| \frac { 1 } { N ^ { 2 } } \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \left( \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { x } _ { j } \mathbf { x } _ { j } ^ { \top } \right) \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \bigg \| = \frac { \| \nabla ^ { 2 } \ell ( \pmb { \theta } ^ { \star } ) ^ { - 1 } \| } { N }
$$

$$
\left| \begin{array} { l } { \displaystyle \sum _ { j = 1 } ^ { N } \mathbb { E } _ { \theta ^ { \star } } ^ { ( N ) } \left[ \mathbf { z } _ { j } ^ { \top } \mathbf { z } _ { j } \right] \bigg | = \left| \frac { 1 } { N ^ { 2 } } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { x } _ { j } ^ { \top } \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \mathbf { x } _ { j } \right| = \frac { 1 } { N } \mathrm { t r } \left( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 2 } \frac { 1 } { N } \sum _ { j = 1 } ^ { N } v _ { j } ^ { \star } \mathbf { x } _ { j } \mathbf { x } _ { j } ^ { \top } \right) } \\ { = \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) } { N } . } \end{array} \right.
$$

Applying Lemma B.4 and substituting the upper bound on $A _ { \mathrm { m a x } }$ , it yields

$$
\begin{array} { r l } & { \left\| \Delta _ { l i n } \right\| _ { 2 } = \left\| \displaystyle \sum _ { j = 1 } ^ { N } \mathbf { z } _ { j } \right\| _ { 2 } \lesssim \sqrt { \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) \log d _ { \theta } } { N } } + \sqrt { \frac { B \mu _ { Q } d _ { \theta } \| \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } \| \log ^ { 2 } d _ { \theta } } { N ^ { 2 } } } } \\ & { \qquad \lesssim \sqrt { \frac { \mathrm { t r } ( \nabla ^ { 2 } \ell ( \theta ^ { \star } ) ^ { - 1 } ) \log d _ { \theta } } { N } } , } \end{array}\tag{A.129}
$$

with probability at least $1 - d _ { \theta } ^ { - c }$ , where the last inequality is from setting $\begin{array} { r } { N \gtrsim \frac { B \mu _ { Q } d _ { \theta } \log d _ { \theta } } { r _ { \mathrm { e f f } } } } \end{array}$ Comparing (A.129) with (A.128), setting $\begin{array} { r } { N \gtrsim \frac { B \mu _ { Q } d _ { \theta } ^ { 2 } } { r _ { \mathrm { e f f } } \log d _ { \theta } } } \end{array}$ sufices to ensure that the upper bound in (A.129) dominates that in (A.128). By setting

$$
N \gtrsim \operatorname* { m a x } \left\{ B ^ { 3 } \mu _ { Q } d _ { \theta } ^ { 3 / 2 } \log ^ { 3 } d , \ B ^ { 4 } \mu _ { Q } d _ { \theta } \log d , \frac { B \mu _ { Q } d _ { \theta } \log d _ { \theta } } { r _ { \mathrm { e f f } } } , \frac { B \mu _ { Q } d _ { \theta } ^ { 2 } } { r _ { \mathrm { e f f } } \log d _ { \theta } } \right\} ,
$$

the r.h.s. of (4.44) is also controlled by the r.h.s. of (A.129). Using the triangle inequality then gives (4.46). □

## B Supporting Lemmas

Throughout this appendix, all random objects appearing in the same statement are assumed to be defined on a common probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ , and E denotes expectation under P. The underlying probability space may vary from lemma to lemma. When applying these results to the statistical model developed in the main part of the paper, (P, E) is instantiated as $( \mathbb { P } _ { \theta } ^ { ( N ) } , \mathbb { E } _ { \theta } ^ { ( N ) } )$ for the relevant value of θ.

Lemma B.1 Let X be a bounded, non-negative random variable with $0 \leq X \leq C < \infty$ almost surely. Then

$$
\mathbb { P } ( X \geq t ) \geq \frac { \mathbb { E } [ X ] - t } { C - t } , \forall t \in ( 0 , C ) .\tag{B.130}
$$

Proof. By direct decomposition, we have

$$
\begin{array} { r } { \mathbb { E } [ X ] = \mathbb { E } [ X \cdot \mathbb { 1 } \{ X < t \} ] + \mathbb { E } [ X \cdot \mathbb { 1 } \{ X \geq t \} ] \leq t \mathbb { P } ( X < t ) + C \mathbb { P } ( X \geq t ) = t + ( C - t ) \mathbb { P } ( X \geq t ) , } \end{array}
$$

rearranging the inequality gives rise to (B.130).

Lemma B.2 ([58, Theorem 1.1]) Let $\pmb { \xi } = [ \xi _ { 1 } , \dots , \xi _ { n } ] ^ { \top } \in \mathbb { R } ^ { n }$ be a random vector with independent components $\xi _ { i }$ which satisfy $\mathbb { E } [ \xi _ { i } ] = 0$ and $\| \xi _ { i } \| _ { \psi _ { 2 } } \le K$ . Then for any but fixed matrix $\mathbf { M } \in \mathbb { R } ^ { n \times n }$ and every $t \geq 0$

$$
\mathbb { P } \left\{ \left| \boldsymbol { \xi } ^ { \top } \mathbf { M } \boldsymbol { \xi } - \mathbb { E } [ \boldsymbol { \xi } ^ { \top } \mathbf { M } \boldsymbol { \xi } ] \right| \geq t \right\} \leq 2 \exp \left( - c \operatorname* { m i n } \left\{ \frac { t ^ { 2 } } { K ^ { 4 } \| \mathbf { M } \| _ { F } ^ { 2 } } , \frac { t } { K ^ { 2 } \| \mathbf { M } \| } \right\} \right) .\tag{B.131}
$$

Lemma B.3 ([1, Corollary 16]) Let $\xi _ { 1 } , \ldots , \xi _ { n }$ be independent mean-zero $K - s u b$ -Gaussian random variables and let $\mathbf { H } : = ( \mathbf { h } ^ { ( i j ) } ) _ { i , j \leq n }$ be any but fixed symmetric matrix with values in the normed space $( \mathbb { R } ^ { m } , \Vert \cdot \Vert _ { 2 } )$ . For $t \geq C K ^ { 2 } \sqrt { \textstyle \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } \| \mathbf { h } ^ { ( i j ) } \| _ { 2 } ^ { 2 } }$ , we have

$$
\mathbb { P } \left\{ \left\| \sum _ { ( i , j ) \in [ n ] ^ { 2 } } \mathbf { h } ^ { ( i j ) } \left( \xi _ { i } \xi _ { j } - \mathbb { E } [ \xi _ { i } \xi _ { j } ] \right) \right\| _ { 2 } \geq t \right\} \leq 2 \exp \left( - \frac { 1 } { C } \operatorname* { m i n } \left\{ \frac { t ^ { 2 } } { K ^ { 4 } U ^ { 2 } } , \frac { t } { K ^ { 2 } V } \right\} \right) ,
$$

where

$$
\begin{array} { l } { { \displaystyle { U = \operatorname* { s u p } _ { \| { \bf u } \| _ { 2 } \leq 1 } \sqrt { \displaystyle \sum _ { j = 1 } ^ { n } \left\| \sum _ { i \neq j } ^ { n } { \bf h } ^ { ( i j ) } { \bf u } _ { i } \right\| _ { 2 } ^ { 2 } + \operatorname* { s u p } _ { \| { \bf Y } \| _ { F } \leq 1 } \left\| \sum _ { ( i , j ) \in [ n ] ^ { 2 } } { \bf h } ^ { ( i j ) } [ { \bf Y } ] _ { i j } \right\| _ { 2 } } , } } } \\ { { \displaystyle { V = \operatorname* { s u p } _ { \| { \bf u } \| _ { 2 } \leq 1 , \| { \bf v } \| _ { 2 } \leq 1 } \left\| \sum _ { ( i , j ) \in [ n ] ^ { 2 } } { \bf h } ^ { ( i j ) } { \bf u } _ { i } { \bf v } _ { j } \right\| _ { 2 } } . } } \end{array}
$$

Lemma B.4 ([68, Theorem 1.6]) Let $\left\{ { \bf H } _ { k } \right\}$ be a finite sequence of independent, random matrices in $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ satisfying $\mathbb { E } [ \mathbf { H } _ { k } ] = \mathbf { 0 } , \| \mathbf { H } _ { k } \| \leq L$ almost surely. Let the norm of total variance be

$$
\boldsymbol { \nu } ^ { 2 } = \operatorname* { m a x } \left\{ \left. \sum _ { k } \mathbb { E } \left[ \mathbf { H } _ { k } \mathbf { H } _ { k } ^ { \top } \right] \right. , \left. \sum _ { k } \mathbb { E } \left[ \mathbf { H } _ { k } ^ { \top } \mathbf { H } _ { k } \right] \right. \right\} .
$$

Then for all $t \geq 0$

$$
\mathbb { P } \left\{ \left. \sum _ { k } \mathbf { H } _ { k } \right. \geq t \right\} \leq \left( n _ { 1 } + n _ { 2 } \right) \exp \left( \frac { - t ^ { 2 } } { 2 \nu ^ { 2 } + 2 L t / 3 } \right) .
$$

## References

[1] Rados law Adamczak, Rafa l Lata la, and Rafa l Meller. Hanson–Wright inequality in Banach spaces. Annales de l’Institut Henri Poincar´e, Probabilit´es et Statistiques, 56(4):2356–2376, 2020.

[2] Selin Damla Ahipa¸sao˘glu, Stefano Cipolla, and Jacek Gondzio. A column generation approach to exact experimental design. Mathematical Programming Computation, 2026.

[3] A. Albert and J. A. Anderson. On the existence of maximum likelihood estimates in logistic regression models. Biometrika, 71(1):1–10, 1984.

[4] Francis Bach. Self-concordant analysis for logistic regression. Electronic Journal of Statistics, 4:384– 414, 2010.

[5] Linas Baltrunas, Tadas Makcinskas, and Francesco Ricci. Group recommendations with rank aggregation and collaborative filtering. In Proceedings of the Fourth ACM Conference on Recommender Systems, pages 119–126, 2010.

[6] Dimitris Bertsimas and Allison O’Hair. Learning preferences under noise and loss aversion: An optimization approach. Operations Research, 61(5):1190–1199, 2013.

[7] Heejong Bong and Alessandro Rinaldo. Generalized results for the existence and consistency of the MLE in the Bradley–Terry–Luce model. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pages 2160–2177, 2022.

[8] Ralph Allan Bradley and Milton E. Terry. Rank analysis of incomplete block designs: The method of paired comparisons. Biometrika, 39(3–4):324–345, 1952.

[9] Kenneth Butler and John T. Whelan. The existence of maximum likelihood estimates in the Bradley– Terry model and its extensions. arXiv:math/0412232, 2004.

[10] Emmanuel J. Cand\`es and Benjamin Recht. Exact matrix completion via convex optimization. Foundations of Computational Mathematics, 9(6):717–772, 2009.

[11] Emmanuel J. Cand\`es and Pragya Sur. The phase transition for the existence of the maximum likelihood estimate in high-dimensional logistic regression. The Annals of Statistics, 48(1):27–42, 2020.

[12] Ennio Cascetta. Random utility theory. In Transportation Systems Analysis: Models and Applications, pages 89–167. Springer US, Boston, MA, 2009.

[13] Hugo Chardon, Matthieu Lerasle, and Jaouad Mourtada. Finite-sample performance of the maximum likelihood estimator in logistic regression. arXiv:2411.02137, 2024.

[14] Bo Chen and Jia Liu. Eliciting von Neumann–Morgenstern utility from discrete choices with response error. Mathematical Programming, 2026.

[15] Pinhan Chen, Chao Gao, and Anderson Y. Zhang. Partial recovery for top-K ranking: Optimality of MLE and suboptimality of the spectral method. The Annals of Statistics, 50(3):1618–1652, 2022.

[16] Yanxi Chen. Ranking from pairwise comparisons in general graphs and graphs with locality. arXiv:2304.06821, 2023.

[17] Yudong Chen. Incoherence-optimal matrix completion. IEEE Transactions on Information Theory, 61(5):2909–2923, 2015.

[18] Yuxin Chen, Jianqing Fan, Cong Ma, et al. Spectral method and regularized MLE are both optimal for top-K ranking. The Annals of Statistics, 47(4):2204–2235, 2019.

[19] Yuxin Chen and Changho Suh. Spectral MLE: Top-K rank aggregation from pairwise comparisons. In Proceedings of the 32nd International Conference on Machine Learning, volume 37, pages 371–380, 2015.

[20] Gauss M. Cordeiro and Peter McCullagh. Bias correction in generalized linear models. Journal of the Royal Statistical Society: Series B (Methodological), 53(3):629–643, 1991.

[21] Annette J Dobson and Adrian G Barnett. An introduction to generalized linear models. Chapman and Hall/CRC, 4th edition, 2018.

[22] Jianqing Fan, Jikai Hou, and Mengxin Yu. Uncertainty quantification of MLE for entity ranking with covariates. Journal of Machine Learning Research, 25(358):1–83, 2024.

[23] David Firth. Bias reduction of maximum likelihood estimates. Biometrika, 80(1):27–38, 1993.

[24] L. R. Ford, Jr. Solution of a ranking problem from binary comparisons. The American Mathematical Monthly, 64(8P2):28–33, 1957.

[25] Chao Gao, Yandi Shen, and Anderson Y. Zhang. Uncertainty quantification in the Bradley–Terry– Luce model. Information and Inference: A Journal of the IMA, 12(2):1073–1140, 2023.

[26] Alfio Giarlotta and Salvatore Greco. Necessary and possible preference structures. Journal of Mathematical Economics, 49(2):163–172, 2013.

[27] Mark E. Glickman and Shane T. Jensen. Adaptive paired comparison design. Journal of Statistical Planning and Inference, 127(1–2):279–293, 2005.

[28] Peter Goos, Bart Vermeulen, and Martina Vandebroek. D-optimal conjoint choice designs with nochoice options for a nested logit model. Journal ofStatistical Planning and Inference, 140(4):851–861, 2010.

[29] Salvatore Greco, Vincent Mousseau, and Roman S lowi´nski. Ordinal regression revisited: Multiple criteria ranking using a set of additive value functions. European Journal of Operational Research, 191(2):416–436, 2008.

[30] Shengbo Guo, Scott Sanner, and Edwin V. Bonilla. Gaussian process preference elicitation. In Advances in Neural Information Processing Systems, volume 23, pages 262–270, 2010.

[31] Bruce Hajek, Sewoong Oh, and Jiaming Xu. Minimax-optimal inference from partial rankings. In Advances in Neural Information Processing Systems, volume 27, pages 1475–1483, 2014.

[32] John D. Hey. Does repetition improve consistency? Experimental Economics, 4(1):5–54, 2001.

[33] Jian Hu, Dali Zhang, Huifu Xu, et al. Distributional utility preference robust optimization models in multi-attribute decision making. Mathematical Programming, 212(1–2):519–565, 2025.

[34] Xun Huan, Jayanth Jagalur, and Youssef Marzouk. Optimal experimental design: Formulations and computations. Acta Numerica, 33:715–840, 2024.

[35] J. Kiefer and Jacob Wolfowitz. The equivalence of two extremum problems. Canadian Journal of Mathematics, 12:363–366, 1960.

[36] K Konis. Linear programming algorithms for detecting separated data in binary logistic regression models. PhD thesis, University of Oxford, 2007.

[37] Ioannis Kosmidis and David Firth. Jefreys-prior penalty, finiteness and shrinkage in binomialresponse generalized linear models. Biometrika, 108(1):71–82, 2021.

[38] Tor Lattimore and Csaba Szepesv´ari. Bandit Algorithms. Cambridge University Press, Cambridge, 2020.

[39] Kuan-Yun Lee and Thomas A. Courtade. Minimax bounds for generalized linear models. In Advances in Neural Information Processing Systems, volume 33, pages 9372–9382, 2020.

[40] Kuan-Yun Lee and Thomas A. Courtade. Minimax bounds for generalized pairwise comparisons. In 2021 International Conference on Machine Learning (ICML) Workshop on Information-Theoretic Methods for Rigorous, Responsible, and Reliable Machine Learning, 2021.

[41] Seong Jin Lee, Will Wei Sun, and Yufeng Liu. Low-rank online dynamic assortment with dual contextual information. Journal of the American Statistical Association, pages 1–13, 2026.

[42] Emmanuel Lesafre and Heinz Kaufmann. Existence and uniqueness of the MLE for a multivariate probit model. Journal of the American Statistical Association, 87(419):805–811, 1992.

[43] Wanshan Li, Shamindra Shrotriya, and Alessandro Rinaldo. ℓ<sub>∞</sub>-bounds of the MLE in the BTL model under general comparison graphs. In Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180, pages 1178–1187, 2022.

[44] Licong Lin, Fangzhou Su, Wenlong Mou, et al. When is it worthwhile to jackknife? Breaking the quadratic barrier for Z-estimators. arXiv:2411.02909, 2024.

[45] Jiapeng Liu, Mi losz Kadzi´nski, and Xiuwu Liao. Modeling contingent decision behavior: A Bayesian nonparametric preference-learning approach. INFORMS Journal on Computing, 35(4):764–785, 2023.

[46] R. Duncan Luce. Individual Choice Behavior: A Theoretical Analysis. Dover Books on Mathematics. Dover Publications, 2012.

[47] Vivek Madan, Mohit Singh, Uthaipon Tantipongpipat, et al. Combinatorial algorithms for optimal design. In Proceedings of the Thirty-Second Conference on Learning Theory, volume 99, pages 2210– 2258, 2019.

[48] P. McCullagh and John A. Nelder. Generalized Linear Models. Chapman and Hall, London, 2nd edition, 1989.

[49] Daniel McFadden. Conditional logit analysis of qualitative choice behavior. In Paul Zarembka, editor, Frontiers in Econometrics, pages 105–142. Academic Press, New York, 1974.

[50] Subhojyoti Mukherjee, Anusha Lalitha, Kousha Kalantari, et al. Optimal design for human preference elicitation. In Advances in Neural Information Processing Systems, volume 37, pages 90132– 90159, 2024.

[51] Sahand Negahban, Sewoong Oh, and Devavrat Shah. Rank centrality: Ranking from pairwise comparisons. Operations Research, 65(1):266–287, 2016.

[52] Dmitrii M. Ostrovskii and Francis Bach. Finite-sample analysis of M-estimators using selfconcordance. Electronic Journal of Statistics, 15(1):326–391, 2021.

[53] Stephen Portnoy. Asymptotic behavior of M-estimators of p regression parameters when $p ^ { 2 } / n$ is large. I. Consistency. The Annals of Statistics, 12(4):1298–1309, 1984.

[54] Stephen Portnoy. Asymptotic behavior of likelihood methods for exponential families when the number of parameters tends to infinity. The Annals of Statistics, 16(1):356–366, 1988.

[55] Simo Puntanen and George P. H. Styan. Schur complements in statistics and probability. In Fuzhen Zhang, editor, The Schur Complement and Its Applications, pages 163–226. Springer US, Boston, MA, 2005.

[56] Rafael Rafailov, Archit Sharma, Eric Mitchell, et al. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741, 2023.

[57] Yanqiu Ruan, Xiaobo Li, Karthyek Murthy, et al. A nonparametric approach with marginals for modeling consumer choice. Management Science, 2026.

[58] Mark Rudelson and Roman Vershynin. Hanson-Wright inequality and sub-Gaussian concentration. Electronic Communications in Probability, 18:1–9, 2013.

[59] Angela Sanguinetti and Nina Amenta. Nudging consumers toward greener air travel by adding carbon to the equation in online flight search. Transportation Research Record, 2676(2):788–799, 2022.

[60] Denis Saur´e and Juan Pablo Vielma. Ellipsoidal methods for adaptive choice-based conjoint analysis. Operations Research, 67(2):315–338, 2019.

[61] Nihar Shah, Sivaraman Balakrishnan, Joseph Bradley, et al. Estimation from pairwise comparisons: Sharp minimax bounds with topology dependence. Journal of Machine Learning Research, 17(58):1– 47, 2016.

[62] Paul L. Speckman, Jaeyong Lee, and Dongchu Sun. Existence of the MLE and propriety of posteriors for a general multinomial choice model. Statistica Sinica, 19(2):731–748, 2009.

[63] Fangzhou Su, Wenlong Mou, Peng Ding, et al. When is the estimated propensity score better? High-dimensional analysis and bias correction. arXiv:2303.17102, 2023.

[64] Michael J. Todd. Minimum-Volume Ellipsoids: Theory and Algorithms. MOS-SIAM Series on Optimization. Society for Industrial and Applied Mathematics, 2016.

[65] Olivier Toubia, John R. Hauser, and Duncan I. Simester. Polyhedral methods for adaptive choicebased conjoint analysis. Journal of Marketing Research, 41(1):116–131, 2004.

[66] Olivier Toubia, Eric Johnson, Theodoros Evgeniou, et al. Dynamic experiments for estimating preferences: An adaptive method of eliciting time and risk parameters. Management Science, 59(3):613– 640, 2013.

[67] Kenneth E. Train. Discrete Choice Methods with Simulation. Cambridge University Press, Cambridge, 2nd edition, 2009.

[68] Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Foundations of Computational Mathematics, 12(4):389–434, 2012.

[69] Joel A. Tropp. An introduction to matrix concentration inequalities. Foundations and Trends® in Machine Learning, 8(1–2):1–230, 2015.

[70] Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer New York, NY, 2009.

[71] Amos Tversky. Intransitivity of preferences. Psychological Review, 76(1):31–48, 1969.

[72] A. W. van der Vaart. Asymptotic Statistics. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 1998.

[73] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2018.

[74] Martin J. Wainwright. High-Dimensional Statistics: A Non-Asymptotic Viewpoint. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, 2019.

[75] Jiaxin Wei, Jia Liu, and Huifu Xu. Coordinate-wise polyhedral method for eliciting multivariate linear utility and univariate nonlinear utility functions. arXiv:2606.13418, 2026.

[76] Yuepeng Yang, Antares Chen, Lorenzo Orecchia, et al. Top-K ranking with a monotone adversary. In Proceedings of Thirty Seventh Conference on Learning Theory, volume 247, pages 5123–5162, 2024.

[77] Yuepeng Yang and Cong Ma. Random pairing MLE for estimation of item parameters in the Rasch model. Journal of the American Statistical Association, 121(554):1683–1694, 2026.

[78] Jie Yu, Peter Goos, and Martina Vandebroek. A comparison of diferent Bayesian design criteria for setting up stated preference studies. Transportation Research Part B: Methodological, 46(7):789–807, 2012.

[79] Sainan Zhang, Shaoyan Guo, Melvyn Sim, et al. Modified polyhedral method for elicitation of shape-free utility and conservatism reduction in robust optimization. arXiv:2503.23269, 2025.

[80] Banghua Zhu, Michael Jordan, and Jiantao Jiao. Principled reinforcement learning with human feedback from pairwise or K-wise comparisons. In Proceedings of the 40th International Conference on Machine Learning, volume 202, pages 43037–43067, 2023.