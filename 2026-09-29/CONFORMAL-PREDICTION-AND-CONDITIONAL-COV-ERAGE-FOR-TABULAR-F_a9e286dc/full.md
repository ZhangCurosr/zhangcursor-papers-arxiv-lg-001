# CONFORMAL PREDICTION AND CONDITIONAL COV-ERAGE FOR TABULAR FOUNDATION MODELS

Sungwoo Park<sup>∗</sup> Sunghee Park<sup>∗</sup> Won Chang<sup>†</sup>

Department of Statistics

Seoul National University

Seoul, Republic of Korea

{park.sw,psh2002,wonchang}@snu.ac.kr

## ABSTRACT

Tabular foundation models (TFMs) provide predictive distributions for regression, but their prediction regions can exhibit undercoverage or overcoverage even when point predictions are accurate. We introduce C-USIM (Conditionally-Uniformized Score Integration Method), a lightweight application of highest predictive density split conformal prediction that accommodates multimodal predictions. Given calibration and test outputs, it requires no additional training or model inference. It provides finite-sample marginal validity under our assumptions. We bound conditional–marginal coverage gaps using distribution-estimation error and score discreteness, and examine coverage heterogeneity through percentile rank– score plots. Experiments with TabPFN and TabICL show improved marginal coverage accuracy and lower average conditional and group coverage errors. Under a fixed data budget, allocating more observations to calibration can reduce marginal coverage error despite less accurate point predictions.

## 1 INTRODUCTION

Tabular foundation models (TFMs) built on prior-data fitted networks (PFNs) approximate posterior predictive distributions (PPD) through pretraining on synthetic data and in-context learning from a labeled context (Muller et al., 2024). Mismatch between pretraining and downstream distributions, ¨ together with approximation errors, can overestimate or underestimate these predictions despite accurate point estimates, motivating calibration and coverage assessment across inputs.

We introduce C-USIM (Conditionally-Uniformized Score Integration Method), a split conformal procedure that calibrates highest predictive density (HPD) regions from TFM outputs. HPD regions can comprise disjoint intervals for multimodal predictions. Uncalibrated plug-in HPD can under- or overcover when the predictive distribution is inaccurate. Following HPD-split (Izbicki et al., 2022), C-USIM sets its density-rank threshold to an empirical quantile of held-out scores. Calibration requires no additional training, model inference, or separate density or score-correction model.

Under exchangeability, calibration guarantees finite-sample marginal coverage of at least the target but can leave systematic undercoverage in some covariate regions and overcoverage in others (Vovk et al., 2005). We analyze conditional–marginal coverage gaps through distribution-estimation error and score discreteness, using percentile rank–score plots to show coverage improvements after calibration and remaining variation across inputs.

Under a fixed label budget, we compare reserving labels for calibration with using all labels as context for uncalibrated plug-in HPD. TabPFN and TabICL experiments show improved marginal coverage accuracy, lower mean absolute conditional coverage error across synthetic mechanisms, and lower aggregate group coverage errors in most real-world comparisons. We further examine the training–calibration ratio within C-USIM and find that allocating more labels to calibration can improve coverage accuracy despite less accurate point predictions.

![](images/7da39db1de3faffe813afdfd019c5fbc890a1f9cadca5b78a9fc7435d981ecfd.jpg)

![](images/94c02e7f133c73be38027c1063bf74a2d5aeb43d6e5e23314854fb13477384ec.jpg)  
Figure 1: C-USIM overview with $n _ { \mathrm { c a l } }$ calibration scores and target coverage 1 − α. Using a pretrained TFM, the procedure predicts densities, computes density-rank scores, calibrates a threshold, and constructs prediction regions without fitting an additional model. See Section 4 for details.

We make three contributions. First, we formulate C-USIM by applying HPD-split to TFM binprobability and quantile outputs, calibrating potentially disjoint prediction regions without additional model fitting or inference. Second, under the stated assumptions, we bound conditional– marginal coverage gaps using bin-probability estimation error and score discreteness, and relate a shared threshold to coverage at individual inputs through our percentile rank–score representation. Third, our experiments under a fixed label budget provide evidence of coverage improvements over plug-in HPD and guidance on context–calibration allocation: more calibration observations can improve marginal coverage accuracy despite less accurate point predictions.

## 2 PRELIMINARIES

## 2.1 PRIOR-DATA FITTED NETWORK

Given a training dataset $D _ { \mathrm { t r a i n } } ~ = ~ \{ ( x _ { 1 } , y _ { 1 } ) , \cdot \cdot \cdot ~ , ( x _ { n _ { \mathrm { t r a i n } } } , y _ { n _ { \mathrm { t r a i n } } } ) \}$ , we estimate the conditional distribution of the response $Y$ at $X = x _ { \mathrm { t e s t } } .$ In the Bayesian framework for supervised learning, let ϕ denote a data-generating mechanism in a space Φ, sampled from the prior $p ( \phi )$ , with datasets D generated independently conditional on $\phi .$ The PPD $p ( \cdot \mid x _ { \mathrm { t e s t } } , D _ { \mathrm { t r a i n } } )$ marginalizes over the posterior of $\phi \colon$

$$
p ( y \mid x , D ) \propto \int _ { \Phi } p ( y \mid x , \phi ) p ( D \mid \phi ) p ( \phi ) d \phi
$$

Prior-data Fitted Networks (PFNs) use Transformers pretrained on synthetic datasets to approximate this Bayesian inference, predicting the PPD from a training context. In univariate regression, where $y \in \mathbb { R } ,$ TabPFN learns probability masses over response bins using cross-entropy loss (Grinsztajn et al., 2026), whereas TabICL estimates conditional quantiles using pinball loss (Qu et al., 2026).

## 2.2 PLUG-IN HPD REGIONS

Let $f _ { x }$ denote an estimated conditional density of Y given $X = x .$ . For $0 < \alpha < 1$ , the plug-in HPD region is defined by the highest-density-region construction (Hyndman, 1996):

$$
\begin{array} { r l } & { \widehat { C } _ { 1 - \alpha } ^ { \mathrm { p l u g } } ( x ) : = \{ y : f _ { x } ( y ) \geq \lambda _ { \alpha } ( x ) \} , } \\ & { \quad \lambda _ { \alpha } ( x ) : = \operatorname* { s u p } \left\{ t > 0 : \displaystyle \int _ { \{ z : f _ { x } ( z ) \geq t \} } f _ { x } ( z ) d z \geq 1 - \alpha \right\} . } \end{array}
$$

The cutoff $\lambda _ { \alpha } ( x )$ retains at least $1 - \alpha$ predictive mass at the largest possible density threshold. For piecewise-constant densities, whole positive-density levels are included in decreasing order until their cumulative mass first reaches or exceeds the target, retaining all boundary ties. These potentially disconnected regions require no calibration, and their predictive mass does not guarantee coverage under the true data distribution.

## 2.3 SPLIT CONFORMAL PREDICTION

Split conformal prediction (CP) guarantees finite-sample marginal coverage under exchangeability (Vovk et al., 2005). After fitting a model on training data, a held-out calibration set $\{ ( X _ { i } , \mathbf { \bar { Y } } _ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } }$ and a nonconformity rule $s ( x , y )$ yield scores $\{ S _ { i } \stackrel { - } { = } s ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } }$ <sup>l</sup>. For target coverage $1 - \alpha$ define $\widehat { C } _ { 1 - \alpha } ( x ) : = \{ y : s ( x , y ) \leq \widehat { q } _ { 1 - \alpha } \}$ where $\widehat { q } _ { 1 - \alpha }$ is the $\left\lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) ^ { - } \right\rceil$ ⌉th order statistic of $\{ S _ { i } \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } } \cup \{ \dot { \infty } \}$ . However, nontrivial distribution-free guarantees of exact conditional coverage are generally impossible in finite samples without additional assumptions (Barber et al., 2021).

For a real-valued random variable $W$ with CDF $F _ { W } ( w )$ , the probability integral transform (PIT) $U : = F _ { W } ( W )$ is uniform when $F _ { W }$ is continuous: $\dot { U } \sim \bar { \mathrm { U n i f } } ( 0 , 1 )$ A score is pivotal if its conditional law is independent of the covariate. For $S = s ( X , Y )$ with conditional CDF $F _ { S \mid X = x } ,$ if $F _ { S \mid X = x }$ is continuous for every x, the oracle conditional PIT correction $\tilde { s } ( x , y ) : = F _ { S | X = x } \dot { ( } s ( x , y ) )$ satisfies ${ \tilde { s } } ( X , Y ) \mid X = x \sim \operatorname { U n i f } ( 0 , 1 )$ . Since this conditional distribution is the same across covariates, the corrected score is pivotal. The region $\{ y : \tilde { s } ( x , y ) \leq 1 - \alpha \}$ therefore has conditional coverage $1 - \alpha$ for every x (Laplante, 2026).

C-USIM computes the model-based score CDF from the TFM predictive distribution without fitting a separate conditional score model. Piecewise-constant predictive densities can also induce discrete scores whose deterministic PIT need not be uniform even under the predictive model. We therefore analyze distribution-estimation error and score discreteness.

## 3 RELATED WORK

Conformal prediction provides distribution-free marginal coverage guarantees (Gammerman et al., 1998; Saunders et al., 1999; Vovk et al., 2005), with extensive work on regression (Shafer & Vovk, 2008; Lei & Wasserman, 2014). HPD-split uses density-based scores to construct adaptive prediction regions for heteroscedastic, skewed, or multimodal conditional distributions (Izbicki et al., 2022). Its HPD-based score is also studied in the C-HDR formulation alongside a broader family of CDF-based conformity scores (Dheur et al., 2025).

Conformalized quantile regression (CQR) calibrates residuals around estimated lower and upper conditional quantiles to form adaptive prediction intervals (Romano et al., 2019). Conformalized histogram regression (CHR) calibrates a nested family of shortest contiguous intervals from conditional histograms using region probability mass as the conformity score; its interval construction can affect length and computational cost (Sesia & Romano, 2021). When applied to the same TFM predictive distribution, CQR and CHR return contiguous intervals, which include the intervening lowdensity region whenever they cover separated modes. C-USIM instead permits disjoint high-density regions, allowing such gaps to be excluded and adapting prediction-set geometry to multimodal predictive densities. This flexibility allows C-USIM to handle broader families of distributions, a capability that is particularly important for TFMs.

JAPAN thresholds densities estimated by normalizing flows, while DSPS combines conditional normalizing flows with density-based ranking and conformal calibration (English & Lippert, 2026; Luo & Zhou, 2026). PIT-CP-MDN and PIT-CP-CNF fit additional models for conditional score distributions using mixture density networks and conditional normalizing flows, respectively, to seek approximate conditional coverage for different base scores (Laplante, 2026).

Finally, for tabular models, CP has been applied to various settings using absolute-error scores for regression and adaptive prediction-set scores for classification (van Leeuwen, 2025; Costa et al., 2026; Kenfack et al., 2026). We apply HPD-split and its PIT interpretation to pretrained TFM outputs directly, without fitting an additional density or score-correction model. Our contribution combines this lightweight construction with conditional-coverage analysis, percentile rank–score diagnostics, and fixed-budget context–calibration allocation experiments.

## 4 C-USIM: CONDITIONALLY-UNIFORMIZED SCORE INTEGRATION METHOD

C-USIM constructs prediction regions by reconstructing densities from TFM outputs, computing density-rank scores, and calibrating a threshold on held-out observations.

## 4.1 SETUP AND NOTATION

For a random variable W, we denote its cumulative distribution function and probability density function, when it exists, by $F _ { W }$ and $f _ { W }$ , respectively.

In this paper, we consider d-dimensional covariates and a univariate response. We denote the training and calibration datasets by $D _ { \mathrm { t r a i n } }$ and $D _ { \mathrm { c a l } } ,$ consisting of $n _ { \mathrm { t r a i n } }$ and $n _ { \mathrm { c a l } }$ covariate-response pairs in $\mathbb { R } ^ { d } \times \mathbb { R }$ , respectively, drawn independently from a common distribution. We also denote a PFN model by $\mathcal { M } ( \cdot , D _ { \mathrm { t r a i n } } ) : \mathbb { R } ^ { d }  \mathcal { P } ( \mathbb { R } )$ , which uses $D _ { \mathrm { t r a i n } }$ as context and maps each covariate x to an estimator of the conditional distribution of $Y$ given $X = x ,$ , assuming that this conditional distribution exists. Unless otherwise stated, we omit the second argument and write $\mathcal M ( x )$ for $\mathcal { M } ( x , D _ { \mathrm { t r a i n } } )$

Assumption 1 (Row-permutation equivariance). For fixed $D _ { \mathrm { t r a i n } } ,$ , let $\mathcal { M } ( \mathbf { x } )$ denote the ordered predictive distributions returned jointly for the calibration and test covariates $\mathbf { x } = ( x _ { 1 } , \dots , x _ { m } )$ We assume $\mathcal { M } ( \pi \mathbf { x } ) = \pi \mathcal { M } ( \mathbf { x } )$ for every row permutation π.

## 4.2 GENERAL SETTING OF C-USIM

For $x \in \mathbb { R } ^ { d }$ , our goal is to construct ${ \widehat { C } } _ { 1 - \alpha } ( x )$ from density predictions with the following properties:

$\mathbb { P } _ { ( X , Y ) } ( Y \in { \widehat { C } } _ { 1 - \alpha } ( X ) ) \geq 1 - \alpha$ , i.e., marginal coverage achieves the target.

• Conditional coverage $\mathbb { P } _ { Y } ( Y \in { \widehat { C } } _ { 1 - \alpha } ( x ) | X = x )$ concentrates around $1 - \alpha$

• The total length of the prediction region ${ \widehat { C } } _ { 1 - \alpha } ( X )$ is as small as possible.

C-USIM uses the CP framework with the nonconformity score $s _ { \mathcal { M } }$ obtained by applying the modelbased PIT to the negated predictive density. Specifically, we take $s _ { 0 } ( x , y ) = - f _ { \mathcal { M } ( x ) } ( y )$ as the base score and evaluate its CDF under $Z \sim \mathcal { M } ( x )$ at $s _ { 0 } ( x , y )$ . Since $s _ { 0 } ( x , Z ) \leq s _ { 0 } ( x , y )$ is equivalent to $f _ { \mathcal M ( x ) } ( Z ) \ge f _ { \mathcal M ( x ) } ( y )$ , this gives

$$
s _ { \mathcal { M } } ( x , y ) : = \mathbb { P } _ { Z \sim \mathcal { M } ( x ) } ( f _ { \mathcal { M } ( x ) } ( y ) \le f _ { \mathcal { M } ( x ) } ( Z ) )
$$

Algorithm 1 gives the complete prediction-set construction. Density-level ordering, cumulative probability masses, and an empirical calibration quantile determine the regions without additional model fitting or inference.

Models used in this work For all analyses and subsequent experiments, we use ${ \mathrm { T a b P F N } } { \bf - v } 3 ^ { 1 }$ and TabICL-regressor- $\mathbf { v } 2 ^ { 2 }$ with the default inference settings of their respective implementations. Hereafter, we refer to these models simply as TabPFN and TabICL, respectively. For both models, we assume the permutation-equivariance condition in Assumption 1 for calibration and test rows, which is natural for their Transformer architectures with suitably masked test attention.

```tcl
Algorithm 1 C-USIM prediction-set construction
Require: Pretrained TFM M, labeled data $D _ { \mathrm { g i v e n } }$ , calibration size $n _ { \mathrm { c a l } }$ , test covariates $\bf { x } _ { \mathrm { t e s t } }$ , and
miscoverage level $\alpha \in ( 0 , 1 )$
Ensure: Prediction sets $\widehat { C } _ { 1 - \alpha } ( x )$ for each test covariate $x .$
1: Split $D _ { \mathrm { g i v e n } } = D _ { \mathrm { t r a i n } }$ ∪˙ $D _ { \mathrm { c a l } }$ with $| D _ { \mathrm { c a l } } | = n _ { \mathrm { c a l } }$
2: Use $D _ { \mathrm { t r a i n } }$ as labeled context for ${ \dot { \mathcal { M } } } .$
3: Query the calibration and test covariates jointly, as in Assumption 1.
4: Construct densities $f _ { x } : = f _ { \mathcal { M } ( x ) }$ from the outputs (Appendix B.1).
5: for each $( X _ { i } , Y _ { i } ) \in D _ { \mathrm { c a l } }$ do
6: $S _ { i } \gets \mathbb { P } _ { Z \sim \mathcal { M } ( X _ { i } ) } ( f _ { X _ { i } } ( Y _ { i } ) \leq f _ { X _ { i } } ( Z ) )$
7: end for
8: $k  \lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \rceil .$
9: $\widehat { q } _ { 1 - \alpha } \gets$ the k-th smallest value in $\{ S _ { i } \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } } \cup \{ \infty \}$
10: return $\widehat { C } _ { 1 - \alpha } ( x ) = \{ y : s _ { { \mathcal { M } } } ( x , y ) \leq \widehat { q } _ { 1 - \alpha } \}$ for each $x \in \mathbf { x } _ { \mathrm { t e s t } }$
```

## 5 THEORETICAL ANALYSIS OF CONDITIONAL COVERAGE FOR C-USIM

In this section and its proofs (Appendix A), we use a stronger assumption than Assumption 1.

Assumption 2 (Test-set invariance). For fixed $D _ { \mathrm { t r a i n } } ,$ consider any two test-covariate datasets $\mathbf { x } _ { 1 } =$ $( x _ { 1 1 } , \ldots , x _ { 1 m } )$ and ${ \bf x } _ { 2 } = ( x _ { 2 1 } , \dots , x _ { 2 n } )$ . For any indices p and q satisfying $x _ { 1 p } = x _ { 2 q } ,$ , we assume that

$$
\mathcal { M } ( x _ { 1 p } , D _ { \mathrm { t r a i n } } ; \mathbf { x } _ { 1 } ) = \mathcal { M } ( x _ { 2 q } , D _ { \mathrm { t r a i n } } ; \mathbf { x } _ { 2 } ) ,
$$

where $\mathcal { M } ( x , D _ { \mathrm { t r a i n } } ; \mathbf { x } )$ denotes the response distribution returned for x when prediction is performed on the test dataset x.

We assume that the calibration and test observations are i.i.d.; under Assumption 2, their scores are therefore i.i.d. evaluations of a fixed scoring rule.

## 5.1 CONDITIONAL COVERAGE GAP BOUND

The conditional coverage gap of conformal prediction at $x ,$ with respect to the scoring rule $s ,$ is defined as the equation below (Laplante, 2026):

$$
\Delta ( x ) : = \operatorname* { s u p } _ { \alpha \in ( 0 , 1 ) } \left| \mathbb { P } \left( Y \in { \widehat { C } } _ { 1 - \alpha } ( x ) \mid X = x \right) - \mathbb { P } \left( Y \in { \widehat { C } } _ { 1 - \alpha } ( X ) \right) \right|
$$

We proved that this can be bounded in terms of the discrete $L ^ { 1 }$ error of the bin probability vectors. Theorem 1 (Conditional coverage gap bound for the PIT-transformed score). Fix $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , and let

$$
- \infty = \hat { y } _ { 0 } ^ { ( x ) } < \hat { y } _ { 1 } ^ { ( x ) } < \dots < \hat { y } _ { r + 1 } ^ { ( x ) } = \infty .
$$

Suppose that the nonconformity score $s _ { \mathcal { M } } ( \boldsymbol { x } , \cdot )$ is constant on each interval $[ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } )$ , with constant value $S _ { i } ^ { ( x ) }$ , and that these values are distinct across intervals. Let

$$
\begin{array} { r } { \hat { p } _ { i } ^ { ( x ) } : = \mathbb { P } _ { Z _ { x } \sim \mathcal { M } ( x ) } \left( Z _ { x } \in [ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } ) \right) , \quad \tilde { p } _ { i } ^ { ( x ) } : = \mathbb { P } \left( Y \in [ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } ) \mid X = x \right) , } \end{array}
$$

and define

$$
B ( x ) : = \frac { 1 } { 2 } \left\| \tilde { p } ^ { ( x ) } - \hat { p } ^ { ( x ) } \right\| _ { 1 } + \operatorname* { m a x } _ { 1 \leq i \leq r + 1 } \hat { p } _ { i } ^ { ( x ) } .
$$

Here, $\tilde { p } ^ { ( x ) } = ( \tilde { p } _ { 1 } ^ { ( x ) } , \tilde { p } _ { 2 } ^ { ( x ) } , \dots , \tilde { p } _ { r + 1 } ^ { ( x ) } ) , \ \hat { p } ^ { ( x ) } = ( \hat { p } _ { 1 } ^ { ( x ) } , \hat { p } _ { 2 } ^ { ( x ) } , \dots , \hat { p } _ { r + 1 } ^ { ( x ) } ) \in \mathbb { R } ^ { r + 1 }$

Let s˜ be the PIT-transformed scoring rule induced by $^ { s } { \mathcal { M } } ,$ and let $\Delta _ { \widetilde { s } } ( x )$ denote the conditional coverage gap ofconformal prediction constructed with s˜. Then,

$$
\begin{array} { r } { \Delta _ { \widetilde { s } } ( x ) \leq \mathbb { E } _ { X } [ B ( X ) ] + B ( x ) . } \end{array}
$$

This result characterizes a trade-off between model complexity and the performance of CP with a PIT-corrected score. $\mathbf { A } \mathbf { s } \ r$ increases, the partition becomes finer and can reduce the discretization term $\mathrm { m a x } _ { 1 \leq i \leq r + 1 } \hat { p } _ { i } ^ { ( x ) }$ . However, estimating the resulting higher-dimensional probability vector may increase the $L ^ { 1 }$ error term. Conversely, using a smaller r can yield a more stable probability estimate but incurs a larger discretization error.

Since this analysis yields a global bound for $\alpha \in ( 0 , 1 )$ , it does not characterize the distribution of conditional coverage over $X$ . To extend the discussion, in the following section, we use the percentile rank–score plot to visualize and examine the distribution of conditional coverage across inputs empirically.

## 5.2 CONDITIONAL COVERAGE THROUGH PERCENTILE RANK–SCORE PLOTS

Assumption 3 (Continuous score distribution). For the results in this section, we assume that the C-USIM score $s _ { \mathcal { M } } ( x , Y ^ { ( x ) } )$ has a continuous distribution, where $Y ^ { ( x ) } \sim Y \mid X = x$ for every covariate x.

Let $T ( x , y ) : = s _ { \mathcal { M } } ( x , y ) , P ( x , y ) : = F _ { s _ { \mathcal { M } } ( X , Y ) \mid X = x } ( T ( x , y ) )$ denote the score and its conditional percentile rank, respectively. Let Z be a conditionally independent copy of $Y$ given $X , \mathrm { i . e . , } ( Z \mid$ $X = x ) \overset { d } { = } \left( Y \mid X = x \right)$ and $Z \perp \perp Y \mid X$

Under Assumption $3 , P ( x , y ) = \mathbb { P } ( s _ { \mathcal { M } } ( x , Z ) < s _ { \mathcal { M } } ( x , y ) | X = x ) = \mathbb { P } ( Z \in \hat { C } _ { T ( x , y ) } ( x ) | X = x )$ Hence, $P ( x , y )$ can be interpreted as the true conditional coverage obtained when $T ( x , y )$ is used as the score threshold. Moreover, by PIT, $P ( X , Y ) \mid X \ = \ { \bar { x ^ { * } } } \sim \ \operatorname { U n i f } ( 0 , 1 )$ and consequently $P ( X , Y ) \sim \operatorname { U n i f } ( 0 , 1 )$

For a fixed score threshold $t ,$ define

$$
\begin{array} { r } { P _ { t } ( x ) : = F _ { s _ { \mathrm { { \scriptscriptstyle M } } } ( X , Y ) \mid X = x } ( t ) = { \mathbb { P } } \left( s _ { \mathrm { { \scriptscriptstyle M } } } ( X , Y ) \leq t \mid X = x \right) = { \mathbb { P } } \left( Y \in \widehat { C } _ { t } ( X ) \mid X = x \right) . } \end{array}
$$

Thus, $P _ { t } ( x )$ is the conditional coverage at covariate vector x obtained by applying t as the score threshold. Thus, $P _ { t } ( X )$ describes the distribution of conditional coverage associated with $\widehat { C } _ { t }$

In C-USIM with target coverage $1 - \alpha ,$ , let $\{ T _ { i } \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } + 1 }$ denote the calibration scores, where $T _ { i } =$ $T ( X _ { i } , Y _ { i } )$ for $i = 1 , \ldots , n _ { \mathrm { c a l } }$ and $T _ { n _ { \mathrm { c a l } } + 1 } = \infty$ . The calibration threshold τˆ is then set as

$$
\hat { \tau } : = T _ { ( k ) } , \qquad k = \lceil ( n _ { c a l } + 1 ) ( 1 - \alpha ) \rceil .
$$

This is the k-th order statistic of $\{ T _ { i } \} _ { i = 1 } ^ { n _ { \mathrm { c a l } } + 1 }$ . Then, the distribution of C-USIM’s conditional coverage C can be written as $C = P _ { \hat { \tau } } ( \mathbf { \dot { \boldsymbol { X } } } )$

Under Assumptions 2 and 3, if $n _ { \mathrm { c a l } } \to \infty$ , the threshold $\hat { \tau }$ converges in probability to $q _ { 1 - \alpha } .$ , which is the $( 1 - \alpha )$ quantile of the marginal score distribution, and $\bar { P _ { \hat { \tau } } ( X ) }$ converges in distribution to $P _ { q _ { 1 } - \alpha } ( X )$ . The ideal case for C-USIM is as follows.

Theorem 2 (Conditional coverage distribution in the ideal case). Under Assumptions 2 and 3, suppose that $\alpha > ( n _ { \mathrm { c a l } } + 1 ) ^ { - 1 }$ and $T ( X , Y ) = f ( P ( X , Y ) )$ for some strictly increasing function f almost surely. Then C-USIM’s conditional coverage C follows $\mathrm { B e t a } ( k , n _ { \mathrm { c a l } } + 1 - k )$ .

For each test covariate $X _ { i } ,$ consider the percentile rank–score curve $( P ( X _ { i } , y ) , T ( X _ { i } , y ) )$ . For a given threshold $t , P _ { t } ( X _ { i } )$ is the horizontal coordinate at which this curve intersects the horizontal line $T = t .$ . More generally, $P _ { t } ( X )$ describes the conditional-coverage distribution at threshold t. Figure 2 illustrates these percentile rank–score plots (a) and the resulting conditional-coverage distributions (b) in an example setting.

The diagonal $T \ = \ P$ indicates agreement between the score and its conditional percentile rank given ${ \bar { X = x } } .$ . At the nominal threshold, curves above (below) the diagonal indicate undercoverage (or overcoverage). For fixed predictive distributions, the calibration shifts the horizontal line while leaving the curves unchanged. In this example, C-USIM moves this line toward $q _ { 1 - \alpha } \approx 0 . 9 7 3$ If the model satisfies Assumption $^ { 2 , }$ this is the point to which its calibrated threshold converges as the calibration sample size increases. Meanwhile, plug-in HPD uses the fixed score threshold

![](images/a2dbfa81cdba123169642566628ed538c6a5e135dc93827aa3fc3a329b3f2906.jpg)

![](images/1d1a89fc7a0f0ae9993c9b5cfd06a5f248ab1f22b0ed92be7bb7259dafac93b9.jpg)

![](images/76914034bcb8f4bdc4e43b286331d6f1d107f88c889f7c0be04d46fcb0519964.jpg)  
Figure 2: Percentile rank–score plots over test covariates (plot (a)), obtained using TabPFN on the sine with shifted exponential noise function (1D-1). Plot (b) approximates $\bar { P _ { t } } ( \bar { X _ { i } } )$ at the nominal score threshold $t = 1 - \alpha$ (blue) and the ideal C-USIM threshold $t = q _ { 1 - \alpha }$ (red), which is estimated by the empirical $( 1 - \alpha )$ -quantile of the pooled Monte Carlo scores. The histogram summarizes interpolated estimates of $\hat { P } _ { t } ( X _ { 1 } ) , \ldots , P _ { t } ( X _ { n _ { t e s t } } )$ , approximating the distribution of $P _ { t } ( X )$ . See Appendices B.2 and B.3 for details.

0.95. Figure 2(a) shows the intersections, and Figure 2(b) summarizes their horizontal coordinates $( P _ { 0 . 9 7 3 } ( X _ { i } )$ for ideal C-USIM and $P _ { 0 . 9 5 } ( X _ { i } )$ for plug-in HPD).

This representation also explains the roles of training and calibration data. Increasing the training sample aims to improve the predictive distributions and bring the curves closer to $T \stackrel { = } { = } P$ , whereas more calibration data can stabilize the estimated threshold. For TabPFN and TabICL, we observed diminishing gains beyond a certain training size. In this regime, reallocating observations from training to calibration improved coverage more than further enlarging the training set. More illustrations of percentile rank–score plots and conditional coverage distributions using TabPFN and TabICL are attached in Appendix B.3. Section 6.3 examines this trade-off under a fixed label budget.

## 6 EXPERIMENTS

We compare C-USIM with plug-in HPD on synthetic and real-world regression tasks, evaluating coverage accuracy and prediction-set length.

Calibrating method. Throughout all experiments, we obtain distribution estimates for the calibration and test covariates by providing the row-merged dataset $D _ { \mathrm { c a l } } \dot { \cup } D _ { \mathrm { t e s t } }$ , following Algorithm 1. Because TabPFN and TabICL produce outputs in different formats, we use model-specific reconstruction rules to convert them into the finite piecewise-uniform conditional densities described in Appendix B.1. For the plug-in HPD baseline, we include the entire boundary density level required to attain at least 95% model mass, following Section 2.2. In contrast, C-USIM uses the exact score sublevel set determined by its calibration order statistic, following Algorithm 1. Appendix B.5 provides the common settings, random seeds, and grouping scheme.

Evaluation measures. We report marginal coverage and total prediction-set length alongside conditional coverage absolute deviation (CCAD) for synthetic data and CEC-X for real data. For synthetic data with known true conditional densities, CCAD measures mean absolute conditional coverage error across inputs. For real data, where these densities are unknown, CEC-X averages absolute empirical coverage errors across covariate-defined groups, weighted by their test sizes. These measures correspond to the three goals in Section 4.2: marginal validity, conditional coverage close to the target, and small prediction sets. Appendix B.4 provides the formal definitions for these.

Fixed-context comparison. Our main comparison uses a total budget of 1,536 labels: plug-in HPD uses all labels as context, while C-USIM uses 512 for context and 1,024 for calibration. To examine the effect of calibrating the cutoff with the predictive densities held fixed, we also compare C-USIM with plug-in HPD using the same 512-observation context. Only C-USIM uses the additional

![](images/bc0bf93201596274f12fc99445a1bded8a734c58bbab0719812836e7f96e210f.jpg)  
Figure 3: Synthetic and Journal SJR overview. (a,b) Synthetic seed means, averaged equally over the selected mechanisms. (c,d) Journal SJR seed means of marginal coverage and CEC-X, respectively; (d) uses a representative covariate grouping. See Table 1 for experiment settings and Appendix B.6 for detailed results.

1,024 calibration responses in this comparison. We report this fixed-context comparison as a supplementary analysis in Appendix B.8, since our primary comparison concerns context–calibration allocation under a common total label budget.

## 6.1 FUNCTION-BASED SYNTHETIC EXAMPLES

We compare plug-in HPD and C-USIM using the settings in Table 1 in Appendix B.5. We consider three one-dimensional and three multidimensional synthetic examples, described in Appendix B.2. We selected these examples to assess coverage in nonlinear settings with oscillatory or discontinuous response functions, skewed noise, and covariate-dependent scale mixtures. For each model, we report averages across all six mechanisms.

Figure 3(a,b) summarizes results from the six selected mechanisms. C-USIM brings mean marginal coverage closer to the target and reduces mean CCAD for both models, with a larger reduction for TabPFN. The lower mean CCAD shows that the gains extend to average conditional coverage accuracy across these mechanisms. Under the same total label budget, these results support reserving labels for calibration when constructing prediction regions from TFM outputs. Appendix B.6.1 gives the case-specific seed distributions, numerical results, and set lengths.

## 6.2 REAL-WORLD REGRESSION EXAMPLES

We evaluate C-USIM on three real-world datasets: Journal SJR, JP Anime, and Allstate Claims Severity. Journal SJR, from the CARTE benchmark (Kim et al., 2024), contains publication meta data and journal H-indices from SCImago.<sup>3</sup> Descriptions and results for the other two datasets are provided in Appendix B.6.3. We use K-means to construct multiple covariate groupings for CEC-X evaluation. Detailed settings and grouping procedures are provided in Table 1 and Appendix B.5.

Figure 3(c,d) shows that C-USIM brings mean marginal coverage closer to the target and lowers CEC-X for both models. Mean group error also decreases for both models. With the same total label budget as plug-in HPD, C-USIM thus improves both marginal coverage accuracy and average coverage accuracy within groups. Appendix B.6.2 provides detailed group results and comparisons.

## 6.3 SPLIT RATIO

We examine how the training–calibration split affects point-prediction accuracy and coverage stability in C-USIM. An 8:2 training–calibration split is a customary starting point for point prediction, but Das et al. (2026) show that the preferred split for prediction-interval length depends on the learning regime. To assess coverage stability, we use the rank–score representation in Section 5.2: training context shapes the input-specific curves, whereas, for a fixed predictor, more calibration data can stabilize the horizontal line $T = { \hat { \tau } }$ . When additional training yields little improvement in the curves’ concentration around $T = P$ , allocating more labels to calibration may improve coverage stability.

![](images/e5186c4e8d6d1973493737da711c85b0fd1bf7d7d7fc799115cff9d1da1ee147.jpg)  
Figure 4: Split-ratio sensitivity on MD-2; settings follow Table 1. Rows correspond to TabPFN and TabICL. Columns show mean-function RMSE, absolute marginal coverage error, and CCAD; lower values are better. Standard boxplots summarize variation across seeds; connected diamonds mark the means.

Figure 4 shows how the training–calibration allocation affects coverage stability for C-USIM on MD-2 from Section 6.1, using the fixed label budget and settings in Table 1. The first two columns show clear trade-offs between RMSE and absolute coverage error, with a smaller calibration size leading to better point prediction accuracy while increasing the absolute coverage error. For TabPFN, a 2:1 or 1:1 split seems to strike a good balance between prediction accuracy and coverage stability. For TabICL, there is no such balancing point, showing clear trade-offs between the two goals. CCAD also increases from the 1:2 split through 5:1. The reduction in marginal coverage error holds across all evaluated case–model pairs (Appendix B.7), supporting larger calibration sets when stable coverage is a priority.

## 7 LIMITATIONS AND FUTURE WORK

The distribution estimates produced by current TFMs may depend on internal model settings, the calibration and test covariates jointly processed in a prediction call, and the ordering of the training rows; Assumption 2 rules out the dependence on the other jointly processed test covariates. Although these factors do not necessarily violate the exchangeability assumption required for conformal prediction, they make it difficult to characterize the generalized behavior of conditional coverage distributions of C-USIM.

C-USIM can be extended to other scoring criteria by computing model-based PIT transformations from the TFM predictive distribution. For instance, if the conditional response distribution is symmetric, a PIT-corrected symmetric density score $( s ( x , y ) = \mathbb { P } _ { Z \sim \mathcal { M } ( x ) } ( f _ { \mathcal { M } ( x ) } ( y ) + f _ { \mathcal { M } ( x ) } ( 2 \hat { y } ^ { \mathrm { m e d } } -$ $y ) \leq f _ { \mathcal { M } ( x ) } ( Z ) + f _ { \mathcal { M } ( x ) } ( 2 \hat { y } ^ { \mathrm { m e d } } - Z ) ) )$ ), where $\hat { \boldsymbol y } ^ { \mathrm { m e d } }$ is the estimated median response, may be considered. Future work could establish other generalized pivotal scores and evaluate their conditional coverage and efficiency on synthetic and real-world data.

## 8 CONCLUSION

C-USIM provides lightweight HPD-split calibration with marginal validity under the stated exchangeability assumptions. It accommodates multimodal predictions through potentially disjoint regions, using pretrained TFM calibration and test outputs without further training or inference. Our analysis bounds the conditional coverage gap and uses percentile rank–score plots to characterize how calibration thresholds affect coverage across inputs. Experiments with TabPFN and TabICL show that, under a fixed label budget, C-USIM improves marginal coverage accuracy over plug-in HPD, reduces CCAD in most evaluated synthetic cases, and lowers aggregate group coverage errors in most comparisons on real data cases. The split-ratio results further suggest that allocations for conformal prediction may differ from the training-heavy splits motivated by point prediction.

## AI USE STATEMENT

In this work, we used generative AI tools for implementing methods. We have not used generative AI tools for proposing or refining hypotheses, interpreting results, or constructing mathematical proofs, and qualitative and thematic data analysis is not applicable to this work. Additionally, we used generative AI tools for creating or editing software code, creating or modifying scientific figures, editing the paper to improve readability, and checking mathematical proofs. We have reviewed all AI-assisted work. In particular, we checked the correctness of the AI-generated code. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Code and detailed configurations will be released on GitHub upon publication. Algorithm 1 specifies the C-USIM procedure, and Appendix A provides proofs of the theoretical results under the stated assumptions. Appendix B documents how predictive densities are constructed from TFM outputs, the synthetic data-generating processes, and the evaluation measures. The experimental settings in Appendices B.5 and B.6.3 include sample sizes, random seeds, training–calibration allocations, and preprocessing and group construction for the real-world datasets. Appendix B.7 gives further details of the allocation experiments.

## ACKNOWLEDGMENTS

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT) (RS-2025-00523567), the Global-LAMP Program of the National Research Foundation of Korea (NRF) grant funded by the Ministry of Education (RS-2023- 00301976), and New Faculty Startup Fund from Seoul National University (326-20240027).

## REFERENCES

Rina Foygel Barber, Emmanuel J. Candes, Aaditya Ramdas, and Ryan J. Tibshirani. The limits of \` distribution-free conditional predictive inference. Information and Inference: A Journal of the IMA, 10(2):455–482, 2021. doi: 10.1093/imaiai/iaaa017.

Jose Lucas De Melo Costa, Fabrice Popineau, Arpad Rimmel, and Bich-Li´ en Doan. High perfor-ˆ mance, low reliability: Uncertainty benchmarking for tabular foundation models. In Proceedings of the European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, pp. 115–120. i6doc.com, 2026. doi: 10.14428/esann/2026.ES2026-261.

Sayan Das, Bahram Yaghooti, Todd A Kuffner, and Soumendra N Lahiri. On optimal data splitting for split conformal prediction. arXiv:2606.31600, 2026. URL https://arxiv.org/abs/ 2606.31600.

Victor Dheur, Matteo Fontana, Yorick Estievenart, Naomi Desobry, and Souhaib Ben Taieb. A unified comparative study with generalized conformity scores for multi-output conformal regression. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 13444–13485. PMLR, 2025.

Eshant English and Christoph Lippert. JAPAN: Joint adaptive prediction areas with normalising flow. In International Conference on Learning Representations, volume 2026, pp. 131010– 131047, 2026.

Alex Gammerman, Volodya Vovk, and Vladimir Vapnik. Learning by transduction. In Proceedings of the Fourteenth Conference on Uncertainty in Artificial Intelligence, pp. 148–155, San Francisco, CA, 1998. Morgan Kaufmann.

Leo Grinsztajn, Klemens Fl´ oge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Mihir Ma-¨ nium, Shi Bin Hoo, Magnus Buhler, Anurag Garg, Dominik Safaric, Jake Robertson, Ben-¨ jamin Jager, Simone Alessi, Adrian Hayler, Vladyslav Moroshan, Lennart Purucker, Philipp¨ Singer, Alan Arazi, Julien Siems, Jan Hendrik Metzen, Georg Grab, Nick Erickson, Siyuan Guo, Eliott Kalfon, Simon Bing, David Salinas, Clara Cornu, Lilly Charlotte Wehrhahn, Diana Kriuchkova, Kursat Kaya, Lydia Sidhoum, Marie Salmon, Jerry Chen, Madelon Hulsebos, Yann LeCun, Samuel Muller, Bernhard Sch¨ olkopf, Sauraj Gambhir, Noah Hollmann, and Frank Hutter.¨ TabPFN-3: Technical report. arXiv:2605.13986, 2026. URL https://arxiv.org/abs/ 2605.13986.

Rob J. Hyndman. Computing and graphing highest density regions. The American Statistician, 50 (2):120–126, 1996.

Rafael Izbicki, Gilson Shimizu, and Rafael B. Stern. CD-split and HPD-split: efficient conformal regions in high dimensions. Journal of Machine Learning Research, 23(87):1–32, 2022. ISSN 1532-4435.

Patrik Kenfack, Samira Ebrahimi Kahou, and Ulrich A¨ıvodji. Towards fair in-context learning with tabular foundation models. Transactions on Machine Learning Research, 2026.

Myung Jun Kim, Leo Grinsztajn, and Gael Varoquaux. CARTE: Pretraining and transfer for tabular learning. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 23843–23866. PMLR, 2024.

Felix Laplante. A post-processing conformal prediction approach for conditional coverage via piv-´ otal scores. arXiv:2605.25852, 2026. URL https://arxiv.org/abs/2605.25852.

Jing Lei and Larry Wasserman. Distribution-free prediction bands for non-parametric regression. Journal of the Royal Statistical Society Series B: Statistical Methodology, 76(1):71–96, 2014.

Rui Luo and Zhixin Zhou. Density-sorted prediction set: Efficient conformal prediction for multitarget regression. Pattern Recognition, 172:112513, 2026. doi: 10.1016/j.patcog.2025.112513.

Samuel Muller, Noah Hollmann, and Frank Hutter. Bayes’ power for explaining in-context learning¨ generalizations. arXiv:2410.01565, 2024. URL https://arxiv.org/abs/2410.01565.

Jingang Qu, David Holzmuller, Ga ¨ el Varoquaux, and Marine Le Morvan. TabICLv2: A better, faster,¨ scalable, and open tabular foundation model. In International Conference on Machine Learning, 2026.

Yaniv Romano, Evan Patterson, and Emmanuel J. Candes. Conformalized quantile regression. In\` Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019.

Craig Saunders, Alex Gammerman, and Volodya Vovk. Transduction with confidence and credibility. In Proceedings of the Sixteenth International Joint Conference on Artificial Intelligence, volume 2, pp. 722–726. Morgan Kaufmann, 1999.

Matteo Sesia and Yaniv Romano. Conformal prediction using conditional histograms. In Advances in Neural Information Processing Systems, volume 34, pp. 6304–6315. Curran Associates, Inc., 2021.

Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(12):371–421, 2008.

Florian D. van Leeuwen. Conformal prediction for tabular prior-data fitted networks with missing data. In EurIPS 2025 Workshop: AIfor Tabular Data, 2025.

Vladimir Vovk, Alex Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer, New York, NY, 2005. doi: 10.1007/b106715.

## A PROOFS

## A.1 PROOF OF THEOREM 1

Proof. The quantity $\Delta _ { \widetilde { s } } ( x )$ is bounded by the Kolmogorov–Smirnov (KS) distance between the conditional and marginal score distributions (Laplante, 2026):

$$
\Delta ( x ) _ { \tilde { s } } \leq d _ { K S } ( F _ { \tilde { s } | X = x } , F _ { \tilde { s } } )
$$

Let $( t _ { 1 } ^ { ( x ) } , \dots , t _ { r + 1 } ^ { ( x ) } )$ be a permutation of $( 1 , \ldots , r + 1 )$ such that

$$
S _ { t _ { i } ^ { ( x ) } } ^ { ( x ) } \leq S _ { t _ { j } ^ { ( x ) } } ^ { ( x ) } \qquad { \mathrm { f o r ~ } } i < j .
$$

The probability measure of the PIT-corrected score s˜ is:

$$
\mu _ { \tilde { s } | X = x } = \sum _ { i = 1 } ^ { r + 1 } \tilde { p } _ { t _ { i } } ^ { ( x ) } \delta _ { \sum _ { j = 1 } ^ { i } \hat { p } _ { t _ { j } } ^ { ( x ) } }
$$

$$
\begin{array} { r } { d _ { K S } ( F _ { \widehat { s } | X = x } , \mathrm { U n i f } [ 0 , 1 ] ) = \displaystyle \operatorname* { m a x } _ { i \in [ r + 1 ] } \operatorname* { m a x } _ { 0 \leq x } \left( \left| \sum _ { j = 1 } ^ { i } ( \widetilde { p } _ { t _ { j } } ^ { ( x ) } - \widehat { p } _ { t _ { j } } ^ { ( x ) } ) \right| , \displaystyle \left| \sum _ { j = 1 } ^ { i - 1 } \widetilde { p } _ { t _ { j } } ^ { ( x ) } - \sum _ { j = 1 } ^ { i } \widehat { p } _ { t _ { j } } ^ { ( x ) } \right| \right) } \\ { \leq \frac { 1 } { 2 } \| \widetilde { p } ^ { ( x ) } - \widehat { p } ^ { ( x ) } \| _ { 1 } + \displaystyle \operatorname* { m a x } _ { i \in [ r + 1 ] } \widehat { p } _ { i } ^ { ( x ) } = B ( x ) } \end{array}
$$

By the convexity of the KS distance, and since $\mu _ { \tilde { s } }$ is the expectation of the conditional distribution $\mu _ { \tilde { s } | X = x }$ , the inequality below holds.

$$
\begin{array} { r l } { d _ { \mathrm { K S } } ( F _ { \widetilde { s } } , \mathrm { U n i f } ( 0 , 1 ) ) \leq \mathbb { E } _ { X } \left[ d _ { \mathrm { K S } } ( F _ { \widetilde { s } | X } , \mathrm { U n i f } ( 0 , 1 ) ) \right] } & { } \\ { \leq \mathbb { E } _ { X } [ B ( X ) ] } \end{array}
$$

Therefore, by the triangle inequality of the KS distance,

$$
\begin{array} { r l } & { d _ { K S } ( F _ { \widetilde { s } | X = x } , F _ { \widetilde { s } } ) \leq d _ { K S } ( F _ { \widetilde { s } } , \mathrm { U n i f } [ 0 , 1 ] ) + d _ { K S } ( F _ { \widetilde { s } | X = x } , \mathrm { U n i f } [ 0 , 1 ] ) } \\ & { \qquad \leq \mathbb { E } _ { X } [ B ( X ) ] + B ( x ) . } \end{array}
$$

## A.2 PROOF OF THEOREM 2

Proof. The theorem’s assumption on the strictly increasing function $f$ implies that $\begin{array} { r l } { f ^ { - 1 } } & { { } = } \end{array}$ $F _ { s _ { \mathcal { M } } ( X , Y ) \mid X = x }$ almost surely regardless of x. Thus,

$$
\begin{array} { r l } & { P _ { t } ( x ) = \mathbb { P } \left( s _ { \mathcal { M } } ( X , Y ) \le t \mid X = x \right) } \\ & { \qquad = \mathbb { E } [ \mathbb { P } \left( s _ { \mathcal { M } } ( X , Y ) \le t \mid X = x \right) ] } \\ & { \qquad = P \left( s _ { \mathcal { M } } ( X , Y ) \le t \right) } \\ & { \qquad = F _ { s _ { \mathcal { M } } ( X , Y ) } ( t ) } \end{array}
$$

Because of the assumption, $F _ { s _ { \mathcal { M } } ( X , Y ) } ( s _ { \mathcal { M } } ( X , Y ) ) \sim \mathrm { U n i f } ( 0 , 1 )$ , and since a cumulative distribution function preserves order, $C = P _ { \hat { \tau } } ( X ) = F _ { s _ { \mathcal { M } } ( X , Y ) } ( \hat { \tau } )$ has the distribution of the k-th order statistic of $n _ { \mathrm { c a l } }$ independent $\mathrm { U n i f } ( 0 , 1 )$ random variables, where $k = \lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \rceil \leq n _ { \mathrm { c a l } } .$ This order statistic follows Beta $\left( \dot { k } , n _ { \mathrm { c a l } } + 1 - k \right)$ □

## B EXPERIMENTAL DETAILS

## B.1 DENSITY RECONSTRUCTION FROM TFM OUTPUTS

The following reconstruction rules convert TFM outputs into piecewise-constant densities without fitting a separate density estimator.

Binning-based Outputs. For binning-based models, we represent the estimated conditional distribution at x by contiguous predicted intervals $\{ [ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } ) \} _ { i = 1 } ^ { r + 1 }$ , where $\hat { y } _ { 0 } ^ { ( x ) } < \hat { y } _ { 1 } ^ { ( x ) } < \ldots < \hat { y } _ { r + 1 } ^ { ( x ) }$ 1 and their corresponding probability weights $\{ \hat { p } _ { i } ^ { ( x ) } \} _ { i = 1 } ^ { r + 1 }$ . We construct a piecewise-constant density estimate by dividing each predicted probability weight by the width of its corresponding interval. Specifically,

$$
f _ { \mathcal { M } ( x ) } ( y ) = \sum _ { i = 1 } ^ { r + 1 } \frac { \hat { p } _ { i } ^ { ( x ) } } { \hat { y } _ { i } ^ { ( x ) } - \hat { y } _ { i - 1 } ^ { ( x ) } } \mathbf { 1 } \Big \{ y \in [ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } ) \Big \} .
$$

For TabPFN, we consider the default setting $( r = 4 9 9 9 , \hat { y } _ { i } ^ { ( x ) }$ is fixed with respect to x) (Grinsztajn et al., 2026).

Quantile-based Outputs. For quantile-based models, we represent the estimated conditional distribution at x by the set of quantile level-value pairs $\{ ( q _ { i } ^ { ( x ) } , y _ { i } ^ { ( x ) } ) \} _ { i = 0 } ^ { r + 1 }$ such that $0 = \hat { q } _ { 0 } ^ { ( x ) } < \hat { q } _ { 1 } ^ { ( x ) } <$ $\hat { q } _ { 2 } ^ { ( x ) } < . . . < \hat { q } _ { r + 1 } ^ { ( x ) } = 1 , \hat { y } _ { 0 } ^ { ( x ) } < \hat { y } _ { 1 } ^ { ( x ) } < \hat { y } _ { 2 } ^ { ( x ) } < . . . < \hat { y } _ { r + 1 } ^ { ( x ) }$ . We construct a piecewise-constant density estimate by dividing each percentile gap by the gap between quantiles. Specifically,

$$
f _ { \mathcal { M } ( x ) } ( y ) = \sum _ { i = 1 } ^ { r + 1 } \frac { \hat { q } _ { i } ^ { ( x ) } - \hat { q } _ { i - 1 } ^ { ( x ) } } { \hat { y } _ { i } ^ { ( x ) } - \hat { y } _ { i - 1 } ^ { ( x ) } } \mathbf { 1 } \Big \{ y \in [ \hat { y } _ { i - 1 } ^ { ( x ) } , \hat { y } _ { i } ^ { ( x ) } ) \Big \} .
$$

For TabICL, we use the default quantile grid $\left( r = 9 9 9 , \hat { q } _ { i } ^ { ( x ) } = \frac { i } { r + 1 } \right)$ . The model does not predict the boundary quantiles $\hat { y } _ { 0 } ^ { ( x ) } , \hat { y } _ { r + 1 } ^ { ( x ) }$ (Qu et al., 2026). Before constructing the density, we retain strictly increasing raw quantile sequences and repair only those with crossings or duplicate values, using float64 arithmetic. For a raw sequence $v _ { 0 } , \ldots ,$ , v<sub>998</sub>, let R be its range, replaced by max(1, |v<sub>0</sub>|) if the sequence is constant, and set $g = \operatorname* { m a x } \{ 1 0 ^ { - 6 } R / 9 9 8$ $8 \mathrm { u l p } ( \mathrm { m a \bar { x } } _ { j } | v _ { j } | ) \}$ , where ulp denotes the spacing to the next larger float64 value. We apply unweighted least-squares isotonic regression to $v _ { j } - j g$ using the pool-adjacent-violators algorithm and then restore the ramp $j g$ . The resulting strictly increasing sequence supplies the interior quantiles in the density above. After this correction, we set the boundary quantile values to $\hat { y } _ { 1 } ^ { ( x ) } { - P \cdot ( \bar { y } _ { 2 } ^ { ( x ) } { - } \hat { y } _ { 1 } ^ { ( x ) } } ,$ and $\hat { y } _ { r } ^ { ( x ) } { + } \hat { P } { \cdot } ( \hat { y } _ { r } ^ { ( x ) } { - } \hat { y } _ { r - 1 } ^ { ( x ) } )$ , respectively, with P fixed at 3 throughout our experiments. This gives 1,000 finite intervals, each with probability mass 0.001.

## B.2 SYNTHETIC DATA-GENERATING PROCESSES

Within each example, context, calibration, and test observations (and hence their covariates) are sampled i.i.d. from the same joint distribution.

## B.2.1 ONE-DIMENSIONAL EXAMPLES

The three one-dimensional examples use $X \ \sim \ \mathrm { U n i f } ( 0 , 2 \pi )$ . Throughout this appendix, $Z \sim$ $\mathcal { N } ( 0 , 1 )$ is independent of the covariates and any mixture-component draw. All Gaussian scale parameters below are standard deviations.

1D-1: Sine with shifted exponential noise. For $E \sim \mathrm { E x p } ( 1 )$ independent of $X ,$ the response is

$$
Y = \sin ( 1 0 X ) + 0 . 1 8 ( E - 1 ) .
$$

The exponential distribution has rate one, so the noise is centered and has variance $0 . 1 8 ^ { 2 }$

1D-2: Periodic Gaussian scale mixture. Given $X = x ,$ draw $B \sim \mathrm { B e r n o u l l i } ( p ( x ) )$ , independently of Z, where

$$
p ( x ) = 0 . 0 8 + 0 . 3 0 ( 0 . 5 + 0 . 5 \sin ( 5 x ) ) .
$$

Then

$$
Y = 0 . 8 \sin ( 1 2 X ) + 0 . 2 \cos ( 3 X ) + \sigma _ { B } Z , \qquad \sigma _ { B } = \left\{ \begin{array} { l l } { { 0 . 4 8 , } } & { { B = 1 , } } \\ { { 0 . 0 7 , } } & { { B = 0 . } } \end{array} \right.
$$

Thus, $p ( x )$ is the probability of the wider Gaussian component.

## 1D-3: Step function. The response is

$$
Y = m ( X ) + 0 . 1 3 Z , \qquad m ( x ) = \left\{ \begin{array} { l l } { { 0 . 8 5 , } } & { { \sin ( 1 0 x ) \ge 0 , } } \\ { { - 0 . 8 5 , } } & { { \sin ( 1 0 x ) < 0 . } } \end{array} \right.
$$

## B.2.2 MULTI-DIMENSIONAL EXAMPLES MD-1 – MD-3

The three multi-dimensional examples use i.i.d. coordinates $X _ { j } \sim \operatorname { U n i f } ( 0 , 1 ) , j = 1 , \ldots , d .$ These examples share the mean

$$
m ( x ) = \sin ( 2 \pi x _ { 1 } ) + 0 . 6 5 ( 2 x _ { 2 } - 1 ) ( 2 x _ { 3 } - 1 ) .
$$

Given $X = x ,$ , draw $B \sim \mathrm { B e r n o u l l i } ( p ( x ) )$ independently of $Z \sim \mathcal { N } ( 0 , 1 )$ and set $Y = m ( X ) +$ $\sigma _ { B } ( \boldsymbol { X } ) Z$ . The parameters below specify the probability and standard deviation of each mixture component. These mechanisms were selected during earlier exploratory screening; the displayed evaluation includes these three mechanisms for both models.

MD-1: Input-dependent severity, d = 5. The mixture probability is $p ( x ) = 0 . 2 0$ , with $\sigma _ { 0 } ( x ) =$ 0.05 and $\sigma _ { 1 } ( x ) = 0 . 3 0 + 0 . 9 0 x _ { 4 }$ . The fourth coordinate changes the scale of the wider component.

## MD-2: Radial mixture weight, d = 10. Define

$$
r ( x ) = \left\{ \frac { 1 } { 7 } \sum _ { j = 4 } ^ { 1 0 } ( 2 x _ { j } - 1 ) ^ { 2 } \right\} ^ { 1 / 2 } , \qquad p ( x ) = 0 . 0 3 + 0 . 6 5 \exp \left( \frac { r ( x ) - 0 . 2 5 } { 0 . 6 0 } , 0 , 1 \right) ,
$$

where $\mathrm { c l i p } ( a , 0 , 1 ) = \operatorname* { m i n } ( 1 , \operatorname* { m a x } ( 0 , a ) )$ . The component scales are $\sigma _ { 0 } = 0 . 0 5 \mathrm { a n d } \sigma _ { 1 } = 1 . 0 0$

MD-3: Input-dependent scale mixture, $d = 2 0 .$ . Here $p ( x ) = 0 . 0 5 + 0 . 5 0 x _ { 4 } , \sigma _ { 0 } = 0 . 0 7$ , and $\sigma _ { 1 } = 0 . 9 0$ . The remaining 16 coordinates do not affect the conditional response law.

## B.3 CONSTRUCTION OF PERCENTILE RANK–SCORE PLOTS AND CONDITIONAL COVERAGE DISTRIBUTIONS

First, for each fixed $X _ { i } ,$ , we independently draw two sets of Monte Carlo samples:

$$
\widetilde { Y } _ { i } ^ { ( 1 ) } , \ldots , \widetilde { Y } _ { i } ^ { ( B ) } \stackrel { \mathrm { i . i . d . } } { \sim } Y \mid X = X _ { i } , \qquad Y _ { i } ^ { ( 1 ) } , \ldots , Y _ { i } ^ { ( L ) } \stackrel { \mathrm { i . i . d . } } { \sim } Y \mid X = X _ { i } .
$$

The first set serves as a reference sample for approximating the conditional percentile rank:

$$
{ \widehat { P } } ( X _ { i } , y ) : = { \frac { 1 } { B } } \sum _ { b = 1 } ^ { B } \mathbf { 1 } \Big \{ T ( X _ { i } , { \widetilde { Y } } _ { i } ^ { ( b ) } ) \leq T ( X _ { i } , y ) \Big \} .
$$

For each $X _ { i }$ , we plot an empirical percentile rank–score curve by sorting and connecting the L points

$$
\left\{ \widehat { P } ( X _ { i } , Y _ { i } ^ { ( \ell ) } ) , T ( X _ { i } , Y _ { i } ^ { ( \ell ) } ) \right\} _ { \ell = 1 } ^ { L }
$$

We overlay these curves for $i = 1 , \ldots , m$ . The conditional-coverage distribution is approximated from the horizontal coordinates of the intersections between these interpolated curves and the horizontal line at each score threshold. We approximate $q _ { 1 - \alpha }$ by the empirical $( 1 - \alpha )$ -quantile of the pooled Monte Carlo scores.

Under Assumptions 2 and 3, the population threshold satisfies $\mathbb { E } [ P _ { q _ { 1 - \alpha } } ( X ) ] = \mathbb { P } [ s _ { \mathcal { M } } ( X , Y ) \leq$ $q _ { 1 - \alpha } ] = 1 - \alpha$ . For a discrete marginal score law, coverage at this threshold is at least $1 - \alpha$ and can exceed it because of a jump in the score CDF.

Generally, the predictive distributions of TabPFN and TabICL in our setting are piecewise uniform, so Assumption 3 does not hold exactly.

In particular, the identity in Section 5.2,

$$
P ( x , y ) = \mathbb { P } ( s _ { \mathcal { M } } ( x , Z ) < s _ { \mathcal { M } } ( x , y ) \mid X = x )
$$

need not hold rigorously, since the inequality

$$
\mathbb { P } \Big ( Z \in \widehat { C } _ { T ( x , y ) } ( x ) \mid X = x \Big ) \geq \mathbb { P } ( s _ { { \mathcal M } } ( x , Z ) < s _ { { \mathcal M } } ( x , y ) \mid X = x )
$$

is strict whenever that level has positive probability under the true conditional response law; this probability is the difference between the two sides. Discrete scores do not satisfy exact PIT uniformity but both models yield sufficiently fine discretizations that this discrepancy can reasonably be regarded as negligible and as having no material effect on the overall patterns in the plots.

Unless otherwise noted, the top two rows of each figure show TabPFN, and the bottom two rows show TabICL. Rows with $n _ { \mathrm { t r a i n } } = 4 0 9 6$ alternate with rows with $n _ { \mathrm { t r a i n } } = 1 0 2 4 .$ , starting from the top. We set $B = L = 1 0 0 0 0$ . Plot (a) shows percentile rank–score plots, whereas plot (b) shows the interpolated conditional-coverage distributions at ideal C-USIM and plug-in HPD.

TabPFN: Sine with shifted exponential noise (1D-1)  
(a) Percentile rank - Score plot  
![](images/18fa4e8695e4c0be1ce1697844d5994310a167454e0614439593e0a11f976828.jpg)

![](images/da950b5ae1558f6203e80afeb7bfd344604618da2bbc7a2b172f22f73faab7da.jpg)

![](images/8b88c208a16ae6c1b94faa9c9333f3dd46ffaf1f291dfab8a1090ed6ad501e1f.jpg)  
TabPFN: Sine with shifted exponential noise (1D-1)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/f50a9b30ae2fac4d0f68e71da829b0cc3d9eb2c68e6b3143bbd8267aee2143f7.jpg)

![](images/2f5a9c084b29f8f1bc34ddf1ae7d2050a2215b220d8cb9fd933ed461954be266.jpg)

![](images/d7c9d5d2341a039d6dec19bb3feace7ea8c8718a5af23ddd8734e4ecdea82eb4.jpg)  
TabICL: Sine with shifted exponential noise (1D-1)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/9f0803535eb25b0dd09eec2bee499f9e1ea8e2ec1ca5574c77718e2720fbfb40.jpg)

![](images/d8f5ca7dd7fb9cec3f46668c185b5abb152223b8130eb1f28ec059c5af845bfb.jpg)

![](images/f1a0e38e42919aaf76fd1f0f37f9d4d7457e853e75cbea75e3e5451c506b02e8.jpg)  
TabICL: Sine with shifted exponential noise (1D-1)

(a) Percentile rank - Score plot  
![](images/b3d88e7e81b056a37cb7c6e654a2efe27dae5d9262503b8ba4e771f2c8bed27c.jpg)

![](images/e9452e8cbacd89633ca062215e1713b8c35c227116a1650b348213641672309d.jpg)

(b) Coverage distribution  
![](images/9a88d5fda8dbcf28005ef4b1c144aed2ec5e2c1731a19c0e60a53e5ac982d223.jpg)  
Figure 5: Percentile rank–score diagnostics for 1D-1, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plug-in HPD.

TabPFN: Periodic Gaussian scale mixture (1D-2)  
(a) Percentile rank - Score plot  
![](images/5da5dcfe9050b8733a59f9e0caba1a1bfd84c2a473dfcb145949c12ac6c12513.jpg)

![](images/8104d20a91fb96e39e15a09cbcb56b199df936c767288de1552336f470f5cd4e.jpg)  
TabPFN: Periodic Gaussian scale mixture (1D-2)

![](images/4e7ba11cf0b6ef7391933efd5ac5a380a4466893ddd2774834639f84acbf6aa4.jpg)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/fd9edc90a3d98edcaeb79d790cc5c8fb6c3771612c2bb3ba9ea283dd28ef58f4.jpg)

![](images/ec697bd419a21f8e18984a49c03cdf1f733bbf47a8db2955d8329eacf8e453b5.jpg)

![](images/7172ea4cbdbd3ee1ae120912e2496e35aafaf17ea53fdb3b45fc245b8ae7d1ce.jpg)  
TabICL: Periodic Gaussian scale mixture (1D-2)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/1ae550842fcf4b8db2c2d3a5ac8d81213b02fd2a2513164aa1dc10d5adc52f9c.jpg)

![](images/88afaafd3067e2df9cea7f6b408d0f2b265244b2451611b2a5acd32b78ad9ba8.jpg)  
TabICL: Periodic Gaussian scale mixture (1D-2)

![](images/7cd4ebb66046a0ea72878f935ee4d09e6fb14aaced7291c816402b165dee1cc1.jpg)

(a) Percentile rank - Score plot  
![](images/e825b91746edc5d011fffec89700f52344e822caa03dce7fab0be9d3bcb7bd28.jpg)

![](images/4e452ecfd4ba636a5a4884df51029a8a963d61300f136ca4f648efdf4b0aafe5.jpg)

(b) Coverage distribution  
![](images/bca43f0d8715444363163741e53911d9a68a0461b51ff896dd63354816386c01.jpg)  
Figure 6: Percentile rank–score diagnostics for 1D-2, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plug-in HPD.

![](images/42b3e69847e43862f3df467657856ebda0547a70ad1457b4b437beec24e148ff.jpg)  
Figure 7: Percentile rank–score diagnostics for 1D-3, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plug-in HPD.

![](images/12e09e60f35ef1b14bb6d0d986f5c9a770c5be093045d59520d21a509850c095.jpg)  
(b) Coverage distribution  
(a) Percentile rank - Score plot

(b) Coverage distribution  
![](images/f6d2e71cd476b71df38308f6e571411b6603287791db2e835a34ad49b10da684.jpg)

![](images/d74201340a924df0ca34eb804fad0d18060e4bc5d72a5f60a698d6e449afacd5.jpg)  
TabPFN: Input-dependent severity (MD-1)

![](images/d39e81e9ffb78d19b38dd23020d605a826312aaf3693ccdb7bc670ecadbf0995.jpg)

![](images/b7576f9b8755718763dd79e37a36337098120cb8a371d938ae07f86bd9189b36.jpg)

![](images/f35a487e92582b6f636d04ac5b5874dd88f14ac3b7f7d81ed16025ac035a765e.jpg)  
TabICL: Input-dependent severity (MD-1)

![](images/0e7f8944a5dbed22d73dcc897c18305f5fcf995af0fa891e57751d91e60f4666.jpg)

![](images/850d740762828f9faa9c5a99223464ab7ae3a8b3acefdc1cd4f67b68a5047002.jpg)

![](images/3d6730f73957171e9e8b2c4cff20bb7ed13d8b18b7f968118c82908f42f59de4.jpg)  
TabICL: Input-dependent severity (MD-1)

![](images/7cff51510ed822a51342f9c15862ad91da18037203ecd1efab34f0858afacbdc.jpg)  
(b) Coverage distribution

![](images/2e4c81ac4df3ed6f4fa8efcaee5608050206100c093594a37150e5f189a5ada3.jpg)

![](images/bc9b6ad41535539fd45bf63bb38a9e778a2cb538d413eb56b2bd39b9962c253a.jpg)  
Figure 8: Percentile rank–score diagnostics for MD-1, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plug-in HPD.

![](images/f8af41b58a542fad162cea8b1759c2ded788ab4eef7750648e94966cb8082551.jpg)  
Figure 9: Percentile rank–score diagnostics for MD-2, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plug-in HPD.

TabPFN: Input-dependent scale mixture (MD-3)  
(a) Percentile rank - Score plot  
![](images/df95c6e806359e7ed69193bafc0216341cb9a2496abe1bd630143870cd354d21.jpg)

![](images/8f15f31c0ab693c3ced7cd561ce910396c2ce3277e58abb69e088e3c60dd5d14.jpg)

![](images/54c778c5eb760d833e49790fc8cc493d17fd5978dcc6be97a1c33c7fffe2f4e3.jpg)  
TabPFN: Input-dependent scale mixture (MD-3)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/aea14f2f0669b1d78a6bd90e333a6ca20f978f00d06cb6acbfd2bea4f3906f6f.jpg)

![](images/60ce84de986c2215b7e2edf4d8d59501152e69a78431550b45b64b85c665f920.jpg)

![](images/55015ab865c0bcd1d769d258b48ee98d5d89455d4a5b98e50c92a6b3a6330364.jpg)  
TabICL: Input-dependent scale mixture (MD-3)  
(b) Coverage distribution

(a) Percentile rank - Score plot  
![](images/134c537a8720ff261543c5ad730d35515f2156c641050f5efe4f4450e46cb7e9.jpg)

![](images/3e769fd0d9a251a064ea6ad9bd4cd56bbd3af28b7e7c0b23870fda741b93a4a5.jpg)

![](images/7ca9f84956919f9659987dd0000093110d164d57a089d174ca17928f743988a6.jpg)  
TabICL: Input-dependent scale mixture (MD-3)

(a) Percentile rank - Score plot  
![](images/84e89d8656961ee1004529e3846bdd8bef98e5853950d2f34374a7bda2c076b6.jpg)

![](images/56edd885986dcc74de5fc43b1fdfd03ca5f5e9814db619fc60017b73c623e962.jpg)

(b) Coverage distribution  
![](images/29176b07d5ed4ee6462ff59c3a40c25b91f43b3ada829a0027c28920c5b083a3.jpg)  
Figure 10: Percentile rank–score diagnostics for MD-3, seed 2026: C-USIM $( n _ { \mathrm { c a l } } \to \infty )$ and plugin HPD.

## B.4 EVALUATION MEASURE DEFINITIONS

For each run, we condition on the fitted predictor, calibration data, and algorithmic randomness, treating $\widehat { C } _ { 1 - \alpha }$ as a fixed prediction-set function. Its conditional coverage, marginal coverage, and conditional coverage absolute deviation (CCAD) are

$$
\begin{array} { r l } & { c ( x ) : = \mathbb { P } \Big ( Y \in \widehat { C } _ { 1 - \alpha } ( x ) \mid X = x \Big ) , } \\ & { \mathrm { C o v } : = \mathbb { E } _ { X } [ c ( X ) ] , } \end{array} \qquad \mathrm { C C A D } : = \mathbb { E } _ { X } [ | c ( X ) - ( 1 - \alpha ) | ] .
$$

The expectation is over the test covariate distribution. Below, we describe how these quantities are evaluated on synthetic and real data, followed by prediction-set length and aggregation across runs.

For synthetic data, let $X _ { i }$ be a test input and let $Y _ { i j } \sim P ( \cdot \mid X _ { i } ) , j = 1 , \ldots , M .$ , be independent conditional response draws, with $M \stackrel { - } { = } 1 0 0 0$ . Both methods use the same inputs and response draws. With 1{·} denoting the indicator function, we compute

$$
\begin{array} { r l r } & { \widehat { c } _ { i } = \displaystyle \frac { 1 } { M } \displaystyle \sum _ { j = 1 } ^ { M } \mathbf { 1 } \Big \{ Y _ { i j } \in \widehat { C } _ { 1 - \alpha } ( X _ { i } ) \Big \} , } & \\ & { \widehat { \mathrm { C o v } _ { \mathrm { M C } } } = \displaystyle \frac { 1 } { n _ { \mathrm { t e s t } } } \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } \widehat { c } _ { i } , } & { \widehat { \mathrm { C C A D } } = \displaystyle \frac { 1 } { n _ { \mathrm { t e s t } } } \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } | \widehat { c } _ { i } - ( 1 - \alpha ) | . } \end{array}
$$

The empirical CCAD retains Monte Carlo error from both the test inputs and conditional response draws.

For real data, let $A _ { 1 } , \ldots , A _ { K }$ be a disjoint covariate partition learned from validation inputs alone. For the test observations $( X _ { i } , Y _ { i } )$ , define $I _ { g } = \{ i : X _ { i } \in A _ { g } \} , n _ { g } = | I _ { g } |$ , and

$$
h _ { i } = { \bf 1 } \Big \{ Y _ { i } \in \widehat C _ { 1 - \alpha } ( X _ { i } ) \Big \} , \qquad \widehat c _ { g } = \frac { 1 } { n _ { g } } \sum _ { i \in I _ { g } } h _ { i } \quad ( n _ { g } > 0 ) .
$$

Observed marginal coverage is $\widehat { \mathrm { C o v } } _ { \mathrm { o b s } } = n _ { \mathrm { t e s t } } ^ { - 1 } \sum _ { i = 1 } ^ { n _ { \mathrm { t e s t } } } h _ { i }$ . We summarize absolute group coverage errors in two ways. Mean group error gives equal weight to each nonempty test group. CEC-X instead weights each group’s absolute error by its share of the test observations:

$$
\widehat { \mathrm { C E C } } _ { X , 1 } = \sum _ { g : n _ { g } > 0 } \frac { n _ { g } } { n _ { \mathrm { t e s t } } } \left| \widehat { c } _ { g } - ( 1 - \alpha ) \right| ,
$$

where empty test groups contribute zero. This is an absolute, not squared, coverage error over a finite partition and is not a direct estimate of pointwise CCAD.

We report prediction-set length as the sum of the lengths of all component intervals, excluding gaps between disjoint intervals. Infinite lengths remain in the averages. For each real-world dataset, coverage uses the entire test set, whereas mean length uses the first 256 inputs in the fixed test order, shared across seeds, models, and methods. The test sets contain 9,731 observations for Journal SJR and Allstate, and 5,372 for JP Anime.

In seed-repeated analyses, absolute gaps, CCAD, mean group error, and CEC-X are computed within each seed before averaging across seeds. Taking the absolute gap of the mean coverage can give a different result.

## B.5 EXPERIMENTAL SETTINGS

Table 1 summarizes the main experimental settings. Settings specific to JP Anime and Allstate Claims Severity are given in Appendix B.6.3. In the synthetic and real-world method comparisons, the two methods share evaluation samples and the model random seed within each run, with a total budget of 1,536 labels and distinct contexts of 1,536 and 512 labels. At target coverage $1 - \alpha = 0 . 9 5$ the calibrated cutoff in these comparisons is the 974th of 1,024 calibration scores. The split-ratio sweep instead varies the context and calibration sizes for C-USIM while keeping $n _ { \mathrm { t r a i n } } + n _ { \mathrm { c a l } } =$ 1536.

Table 1: Experimental settings. Synthetic and split-ratio sample counts are per mechanism and run; conditional response draws are additional Monte Carlo samples. Split-ratio coverage and CCAD use the known conditional CDF rather than the stored response draws. A dash denotes an unused setting.
<table><tr><td>Setting</td><td>Synthetic</td><td>Journal SJR</td><td>Split ratio</td></tr><tr><td>Dataset size</td><td>1,792 per run</td><td>27,803 cleaned rows</td><td>1,792 per run</td></tr><tr><td>Total label budget</td><td>1,536</td><td>1,536</td><td>1,536</td></tr><tr><td>Input dimensions</td><td>1,5, 10, 20</td><td>7 (categorical)</td><td>1,5, 10,20</td></tr><tr><td>Run seeds</td><td>100-109</td><td>12100-12299</td><td>25600-25649</td></tr><tr><td>Number of seeds</td><td>10</td><td>200</td><td>50</td></tr><tr><td>Plug-in context observations</td><td>1,536</td><td>1,536</td><td></td></tr><tr><td>C-USIM context observations</td><td>512</td><td>512</td><td>256-1,280</td></tr><tr><td>Calibration observations</td><td>1,024</td><td>1,024</td><td>256-1,280</td></tr><tr><td>Validation observations</td><td></td><td>512</td><td></td></tr><tr><td>Test observations</td><td>256</td><td>9,731</td><td>256</td></tr><tr><td>Conditional draws per test input</td><td>1,000</td><td></td><td>1,000 (stored)</td></tr><tr><td>Target coverage</td><td>95%</td><td>95%</td><td>95%</td></tr><tr><td>Group counts (K) Clustering seeds</td><td></td><td>5, 10, 15, 20, 30, 40</td><td></td></tr><tr><td></td><td></td><td>1717, 2717, 3717, 4717, 5717</td><td></td></tr><tr><td>Representative grouping</td><td></td><td>K = 10, seed 1717</td><td></td></tr></table>

Split-ratio training–calibration allocations: 256:1280, 512:1024, 768:768, 1024:512, 1229:307, and 1280:256. The fifth allocation is the closest integer split to 8:2. All six synthetic mechanisms are evaluated with both models.

Synthetic sampling. Conditional response draws are independent at each test input and shared across models and allocation arms. Table 1 reports the number of test inputs and response draws; generating equations are given in Appendix B.2.

Journal SJR preprocessing and groups. The categorical inputs are publication type, country, region, publisher, coverage years, subject categories, and subject areas. The response and predictionset lengths use the log $\overline { { { \bf 1 0 } } } ( \bar { H } { \bf - i n d e x } \ i \not { + } 1 )$ scale. For TFM inference, missing categories receive a dedicated token, and categorical inputs are retained. Each seed uses the same context/calibration row allocation for both models. For each fitted context, all query covariates are processed in one prediction call: 9,731 test inputs for plug-in HPD, and 1,024 calibration plus 9,731 test inputs for C-USIM. Calibration scores and the cutoff are computed from this joint output, using the finite-density reconstruction described above. For Journal SJR, the validation and test sets remain fixed across run seeds and are shared by both models. For group construction, we impute missing categorical values with their validation-set modes and apply one-hot encoding before fitting K-means with 10 initializations per clustering seed. The imputer, encoder, and cluster centers are fitted on validation covariates alone, without using responses. Test covariates undergo the same transformation and are assigned to their nearest cluster centers by Euclidean distance in the encoded feature space; categories absent from validation are encoded as all zeros in the corresponding feature block. We evaluate every combination of the six group counts and five clustering seeds in Table 1, giving 30 groupings in total. Each grouping uses its specified number of groups, with group assignments shared by both models and methods and reused across run seeds. The representative grouping was chosen before the expanded evaluation. For each seed, we compute $| \widehat { c } _ { g } - 0 . 9 5 |$ for every nonempty test group before averaging.

Numerical implementation and checks. Both models are configured with eight estimators and FP32 parameters and outputs. For one-dimensional inputs, the default TabICL ensemble generator produces two distinct preprocessing configurations; for the multidimensional inputs it produces eight. TabICL computes scaled dot-product attention in FP64 before returning FP32 activations; this numerical policy is shared by the synthetic and real-world studies. For every synthetic fitted context, numerical checks compare repeated predictions and predictions under singleton, short, extreme-mixed, and permuted queries. For every real-world seed and fitted context, we permute the entire query table, restore the original row order, and compare predictions at every query input. We also verify that each ensemble forward receives the full query table; internal batching over ensemble members preserves all query rows. For the additional datasets, FP64 attention may also be batched over independent leading batch items to limit memory use; the query and key dimensions of each attention problem are kept intact. All checks use the fixed tolerance 10<sup>−4</sup> + 10<sup>−4</sup>|reference|. These checks support the reported numerical implementation but do not establish query independence for every possible input.

Computational cost. The lightweight aspect of C-USIM is that, once the joint calibration and test outputs are available, it requires only post-processing, with no additional training or model inference. No separate density or score-correction model is fitted. For a reconstructed density with B intervals, the implementation sorts density levels once and computes cumulative probability masses, combining all intervals with equal density into the same score level. This requires O(B log B) time and O(B) storage per query; extracting the prediction set from these quantities takes $O ( B )$ time. Calibration adds an empirical order-statistic calculation. These costs exclude TFM inference.

## B.6 DETAILED EXPERIMENTAL RESULTS

We report the synthetic results first, followed by Journal SJR and the two additional real-world datasets. The tables summarize performance across seeds, and the figures show the corresponding distributions and illustrative prediction sets.

## B.6.1 SYNTHETIC

Table 2 and Figure 17 summarize the six selected mechanisms over ten seeds per model. Figures 11– 16 show prediction sets and estimated conditional-coverage distributions for each mechanism, with TabPFN above TabICL. These illustrations use seed 100, fixed before evaluation. The comparison allocates the same 1,536 labels to either the plug-in context or the C-USIM context and calibration sets; the fitted predictors therefore differ between methods.

For TabICL, marginal coverage improves on 1D-1 while CCAD increases. Undercoverage and overcoverage across inputs can offset each other in marginal coverage, whereas both contribute to the absolute errors measured by CCAD. With the training context and the resulting TFM predictive distributions held unchanged, raising only the shared calibration threshold in the rank–score plot can reduce undercoverage at some inputs while increasing overcoverage at others. If the added overcoverage outweighs the reduction in undercoverage on average, CCAD increases even as marginal coverage approaches the target. This balance depends on the model-induced score distributions across inputs. The fitted predictors also differ in the present comparison, so the observed CCAD difference does not isolate the effect of threshold adjustment.

Reading the prediction-set plots. For each model and mechanism, plots (a) and (b) show Plug-in HPD(1536) and C-USIM, respectively. They display 80 of the 256 test inputs, selected at evenly spaced ranks of the first covariate using the same rows for both methods and models. Each prediction set is centered at its observed test response, so the black vertical line at zero marks that response; gray sets miss it. The two methods share an interval-axis range within each model and mechanism. All finite interval endpoints are retained, and arrows indicate unbounded components. In the multidimensional examples, ordering by the first covariate does not define a one-dimensional conditional slice.

Reading the coverage distributions. Plot (c) uses all 256 test inputs with equal weight. Conditional coverage at each input is estimated from 1,000 shared draws from the known conditional response distribution. Both methods use the same 35 histogram bins, spanning their combined observed range with padding; no observations are removed. A dashed line marks the 0.95 target.

Observed response

![](images/9bff277bb92c438c5015bd7d3e5afa539b5060d32b258c93dce8d5111795f4c1.jpg)

![](images/841274c5dbd53f45ca121ec086c9ba0e55a619329aa48072d1047fea8f5d5975.jpg)  
Set misses observed response

(c) Coverage distribution  
![](images/b71b27d4a08c9a063691d19797d8f06e8e3b5634e51fa8ca4155f6b34f4d9876.jpg)

![](images/73a4c3f372a9dfe303733c3d791be4b6f06bf2917c86b00d45d508fc8f1fb204.jpg)  
Observed response

![](images/052bcaf583c55be25f96f89522b9c469488e09bb65ebcafd90fd32ad653f0e66.jpg)  
Set misses observed response

(c) Coverage distribution  
![](images/365f7187fac34f09c4adfea51ba6031757b5663918f648e8a902e9800b917e7f.jpg)  
Figure 11: 1D-1, seed 100. Plotting details: Appendix B.6.1.

(a) Plug-in HPD  
![](images/c26e8823d148060167c5853346f16937b8f722c067a5e867aee4b36d0a1586f8.jpg)

![](images/ec0502919cb2eb4bdfdbc4d4b09b7010b70adc10bafdf443bab74926853321b1.jpg)

(c) Coverage distribution  
![](images/029e4bfbd627f8064def51bd2561daaeca07780736f8fb27a940ba44de3d5316.jpg)

(a) Plug-in HPD  
![](images/e1132afc06b90f4daa9fb4ce65305294f2d55efe6fe2571e2578bef2f213b84f.jpg)  
Observed response

![](images/40c13066c06f2735107e314782149135905e6b22364b6671bcd17489004e80ca.jpg)  
Set misses observed response

![](images/b4606aeb9748db4f0b7084baa563739fd01fec529900f7cb5946c8bd1bc92123.jpg)  
Figure 12: 1D-2, seed 100. Plotting details: Appendix B.6.1.

Observed response

(a) Plug-in HPD  
![](images/036e1a1ec366ac5b964df267c4174a9f22db937ffe3f0aa7ccb28cfad6659abe.jpg)

![](images/001c6ca5d895832b3d0d1892e2726c6b17c3354824142e8da11ccf4083913f28.jpg)

(c) Coverage distribution  
![](images/d3102f4d471703977797a4caedc271adb105b43177e1783543e888d37b8f6069.jpg)  
(c) Coverage distribution

![](images/40b8d4b29f94b9d127c1052e40e227880a84a31b7ba0ce6521fbf581e45a47a6.jpg)

![](images/9ca332557f56b36f5bf2aaecc2125fd480ea4bf2f244db54cc5b8cc69a3d062d.jpg)

![](images/2895f2946ced9c098b4c1ceba5ffd6154d68013abdb7deec62fa0569fbf1f624.jpg)  
Figure 13: 1D-3, seed 100. Plotting details: Appendix B.6.1.

![](images/408fb75ffdb2df74c5e092b15fbdfbcbe418138bbd1752b299cc4c90fdf11687.jpg)

![](images/2dd030846b8c9ed49f90916f43c1b80099f7741b6b2272937b18a7d7c4b53631.jpg)

(c) Coverage distribution  
![](images/a67cc8fa8e79caa721041afa163faa1c9010e86b8949af83e446e3131c4e7c8f.jpg)

![](images/c819c839e1c875bdc313bcae456e480632501110b60f094b226655ea38d85495.jpg)

![](images/f482af111c51eff22bc808862dc8d22e7a49031b892c3b6c89e8ce80dd2e8991.jpg)  
Set misses observed response

![](images/65067d5905a8d53eb9b1de140ee23f8efc57ddd9f0a5f40458ab2732b59504ac.jpg)  
Figure 14: MD-1, seed 100. Plotting details: Appendix B.6.1.

(a) Plug-in HPD  
Set misses observed response  
Set misses observed response  
![](images/4a367cb4303b0d0df8ba5a91fb22eaf62a2cc2199f10e311669ef76c78a27152.jpg)

![](images/6b3141dfac639fdc6a513c1036dd5f4e1d33305cc892cf5022612298284b24d3.jpg)

(c) Coverage distribution  
![](images/9a415f1ca398d12d4df14c5c3f8e2f808d0b52dfefbd025d505d421e7fe92665.jpg)  
(c) Coverage distribution

![](images/741603e33ce48c758a30c4f3ab5e76048a58af9239e837f96fa514324bc9cd66.jpg)

![](images/26affda315034b9c2315d986394b997ec5bb9fceb6ec3a73c5ae92afbc640ffb.jpg)

![](images/a0edd24acaeef029d1f9417a472cf69c2ce9032df134b0d4e8b64b7cc888126f.jpg)  
Figure 15: MD-2, seed 100. Plotting details: Appendix B.6.1.

![](images/0050b36348374df440e418177e6642aaa9425ef2c2549deac4747850f7bb8cdf.jpg)

![](images/b0a7344d51586437dbd7297af72913ebf31c8e6a872b56dd5e31e889a38fcce2.jpg)

(c) Coverage distribution  
![](images/0e40d67082bcaccfe174fe74dff10a28ad387f174bf1d5e8c76d82616f7108e5.jpg)

![](images/955c1ce40c661c9173c583e4eead5b7d6a1d56f22a71d4b197445812421c7b47.jpg)  
Observed response

![](images/b5077a22ce0502a3838946428feab2c8a47545243f84e1f97e26475d6c40a70a.jpg)  
— Set misses observed response

(c) Coverage distribution  
![](images/4017414863fb42cf2859abb5730c78ccf2e6c1a6c6d9093bd79dc1c356969a3e.jpg)  
Figure 16: MD-3, seed 100. Plotting details: Appendix B.6.1.

Table 2: Synthetic comparison under a fixed budget of 1,536 labels, averaged over ten seeds per model and case. Each arrow denotes Plug-in HPD(1536) → C-USIM(512+1024). Coverage is in percent and CCAD in percentage points (pp). Cases are defined in Appendix B.2.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Example</td><td>Coverage (%)</td><td>CCAD (pp)</td><td>Mean length</td></tr><tr><td>Plug-in HPD(1536) → C-USIM</td><td></td><td></td></tr><tr><td rowspan="5">TabPFN</td><td>1D-1</td><td>91.92 → 95.28</td><td>3.572 → 2.159</td><td>0.571 → 0.717</td></tr><tr><td>1D-2</td><td>89.77 → 94.74</td><td>5.384 → 2.186</td><td>0.754 → 1.290</td></tr><tr><td>1D-3</td><td>91.63 → 94.91</td><td>3.513 → 1.708</td><td>0.483 → 0.580</td></tr><tr><td>MD-1</td><td>89.05 → 95.33</td><td>5.986 → 2.201</td><td>0.915 → 2.100</td></tr><tr><td>MD-2</td><td>88.24 → 94.81</td><td>6.854 → 1.642</td><td>2.274 → 3.313</td></tr><tr><td rowspan="6">TabICL</td><td>MD-3</td><td>89.64 → 95.07</td><td>5.533 → 2.054</td><td>1.706 → 2.706</td></tr><tr><td>1D-1</td><td>94.19 → 95.31</td><td>2.961 → 3.385</td><td>0.617 → 0.747</td></tr><tr><td>1D-2</td><td>93.48 → 94.93</td><td>2.741 → 2.632</td><td>1.044 → 1.287</td></tr><tr><td>1D-3 MD-1</td><td>92.93 → 95.13</td><td>3.186 → 3.168</td><td>0.535 → 0.725</td></tr><tr><td>MD-2</td><td>93.08 → 95.43</td><td>2.930 → 2.644</td><td>1.402 → 1.901</td></tr><tr><td>MD-3</td><td>93.42 → 95.01</td><td>2.220 → 1.572</td><td>2.759 → 3.109</td></tr><tr><td></td><td></td><td>93.11 → 95.17</td><td>2.679 → 2.266</td><td>2.087 → 2.609</td></tr></table>

![](images/4c408306b58d2bd7cce41c792df69873b2987430476ef875d58d73aad97941bc.jpg)

![](images/08ba3b0a8e7558cf3f536ad09e65e6a68608350b3921d13c9b858dc981263a74.jpg)

(c)  
![](images/f7528ba1a26de42b6860c54164da1d8f20e81291bd1d00577ea3f856aa2793df.jpg)

(d)  
![](images/fa0e7d74b067179594ffe259f41ef7cbb2d5a9edb019ce19740143e9b5e4792f.jpg)  
Figure 17: Synthetic coverage (a,b) and CCAD (c,d) across ten seeds for Plug-in HPD(1536) and C-USIM(512+1024); methods and label budgets follow Table 2. Boxes show Q1–Q3 and medians, with 1.5-IQR whiskers and all outliers. Dashed lines mark 95% coverage. CCAD is in percentage points on model-specific scales. These seed distributions are not confidence intervals.

## B.6.2 JOURNAL SJR

Table 3 and Figure 18 report the fixed-budget comparison between Plug-in HPD(1536) and C-USIM for Journal SJR.

![](images/dd4a8cc040a13c57776b012b32a2b3f66bf34730e1afe9ed45e4f3abdb9ffa5f.jpg)

![](images/c092809afcfb27d0feeb4f784d7c25be7a1e6d1e5657621ea592f2454291b2ce.jpg)

![](images/9600f0030bd2547e3f9279421053b7a963f3e763c72427f6ad32bdfb0eef6eee.jpg)

![](images/bc419b0c513e45a5c105e7a5ea971d6be405b40df2716e65af707b29205ea13f.jpg)

![](images/a64f34df26b07f271114410aef5f70d0a64cf3c6f79d3cb46664bf43df8a636b.jpg)

![](images/500d5089a08b88a0b31d4b5f669698564507e7b07d31c92c86453eaee9d9857b.jpg)  
Figure 18: Journal SJR across 200 seeds for Plug-in HPD(1536) and C-USIM(512+1024); methods and label budgets follow Table 3. (a,b) CEC-X, weighted by group size; (c,d) group coverage; (e,f) absolute group error. Errors are computed within each seed. Boxes show Q1–Q3 and medians, with 1.5-IQR whiskers and all outliers, not confidence intervals. Red bands mark higher mean absolute group error with C-USIM; dashed lines mark 95% coverage. Errors are in percentage points. CEC-X shares a scale across models; group plots use model-specific scales with visual padding below zero.

Variation across groups and groupings under a fixed label budget. In the representative grouping, mean absolute error decreases in 7 of the ten groups for TabPFN and 5 for TabICL; 62.20% and 55.35% of seed–group pairs, respectively, move closer to 95% coverage. Across the 30 groupings, C-USIM reduces absolute coverage error in 47.56–70.60% of seed–group pairs for TabPFN and 40.64–61.00% for TabICL. All nonempty groups remain in the reported averages. These aggregate gains coexist with higher errors in some covariate groups and longer mean prediction sets.

## B.6.3 ADDITIONAL REAL-WORLD DATASETS

Datasets and experimental settings. We add JP Anime from CARTE (Kim et al., 2024) and Allstate Claims Severity from OpenML (dataset 42571, version 1).<sup>4</sup> JP Anime predicts the supplied natural logarithm of the anime score from ten features: genres, type, episodes, producers, studios, source, duration, content rating, start date, and end date. Eight features are categorical; episodes and duration are numeric. The prepared cohort retains the first row for each distinct feature profile. We use the source response without applying another logarithm, and report prediction-set lengths on this log-score scale. Allstate predicts the supplied claim loss from 116 categorical and 14 continuous anonymized features. Claim identifiers are excluded from the predictors, and repeated feature profiles with distinct claim identifiers remain separate observations. No response transformation is applied to Allstate.

Table 3: Journal SJR under a fixed budget of 1,536 labels at 95% target coverage, averaged over 200 seeds. Each arrow denotes Plug-in HPD(1536) → C-USIM(512+1024). Absolute marginal and group gaps and CEC-X are computed within each seed before averaging, and are reported in percentage points (pp).
<table><tr><td rowspan="2">Measure</td><td>TabPFN</td><td>TabICL</td></tr><tr><td>Plug-in HPD(1536) → C-USIM</td><td></td></tr><tr><td>Marginal coverage (%)</td><td> $\overline { { 9 1 . 5 0 \to 9 4 . 8 3 } }$ </td><td> $\overline { { 9 2 . 9 3 \to 9 4 . 8 7 } }$ </td></tr><tr><td>Mean marginal gap (pp)</td><td> $3 . 4 9 9  0 . 5 5 3$ </td><td> $2 . 2 4 1  0 . 6 1 1$ </td></tr><tr><td>Mean group gap (pp)</td><td> $2 . 9 7 9  1 . 7 6 0$ </td><td> $2 . 4 5 8 \to 1 . 8 5 6$ </td></tr><tr><td>CEC-X (pp)</td><td> $3 . 6 9 8 \to 1 . 6 6 2$ </td><td> $2 . 7 8 2  1 . 7 0 7$ </td></tr><tr><td>Mean set length</td><td> $1 . 1 4 9  1 . 4 5 3$ </td><td> $1 . 2 5 5 \to 1 . 5 4 0$ </td></tr></table>

![](images/c3782e6c6304fe482b95e0f6595e126f854be59c68a99d37bcefacf1434daeb7.jpg)

![](images/e77033631d032df765ca5763348dea4c8346b372fab07ce247fb4b36dd0d52be.jpg)

![](images/cfe02250eb895d0051872b768a5097eb753c2f2c551d34155fd628254041b4f9.jpg)

![](images/72c694d0d2c9a842bf84ac825d11ded436b5fc315467c9e1060c86a99c8db864.jpg)  
Figure 19: Additional real-world examples under a total budget of 1,536 labels. Points are means over 50 seeds, not confidence intervals. (a,b) JP Anime; $^ { ( \mathrm { c } , \mathrm { d } ) }$ Allstate Claims Severity. Dashed lines mark 95% coverage. CEC-X uses the representative covariate grouping and is computed within each seed before averaging. Settings and detailed results are in Table 4 and Appendix B.6.3.

Table 4 gives the additional sample counts. Both datasets use 50 run seeds, 12100–12149, fixed before the new evaluation, with the same model and label-budget settings as Journal SJR. For each fitted context, plug-in HPD processes the full test set in one prediction call, and C-USIM processes all 1,024 calibration covariates together with that test set. Calibration scores are computed anew for each fitted context and joint query table. For model inference, missing numeric values are imputed using context medians and missing categorical values receive a dedicated token. Groups use the same 30 combinations of group counts and clustering seeds as Journal SJR. Numeric variables are median-imputed and standardized, while categorical variables are mode-imputed and one-hot encoded; all preprocessing and cluster centers are fitted on the 512 validation covariates only. The representative grouping remains $K = 1 0$ with clustering seed 1717.

Figure 19 summarizes the results, and Table 5 reports the comparison between Plug-in HPD(1536) and C-USIM(512+1024) over 50 seeds per model. For both models on both datasets, C-USIM reduces mean absolute marginal coverage error to approximately 0.54–0.60 percentage points. On JP Anime, CEC-X decreases from 6.598 to 2.429 percentage points for TabPFN and from 4.211 to 2.521 for TabICL. On Allstate, it decreases from 2.186 to 1.517 for TabPFN. Mean predictionset length increases in all four comparisons; lengths are measured on the dataset-specific response scales in Table 4.

Table 4: Additional real-world datasets.
<table><tr><td>Dataset</td><td>Rows</td><td>Features</td><td>Categorical</td><td>Test rows</td><td>Response scale</td></tr><tr><td>JP Anime</td><td>15,351</td><td>10</td><td>8</td><td>5,372</td><td>ln(Score)</td></tr><tr><td>Allstate</td><td>188,318</td><td>130</td><td>116</td><td>9,731</td><td>Claim loss</td></tr></table>

Variation across groups and groupings. In the representative grouping, mean absolute group error decreases in 7 of ten groups for TabPFN and 5 for TabICL on JP Anime, and in 8 and 5 groups, respectively, on Allstate. Across all 30 groupings, CEC-X decreases for both models on JP Anime and for TabPFN on Allstate. The Allstate TabICL comparison is more sensitive to grouping: CEC-X decreases in 17 of the 30 groupings. In the representative grouping, its CEC-X changes from 1.481 to 1.561 percentage points, while mean group error changes from 1.540 to 1.523. The group plots in Figures 22 and 23 retain all ten groups and place both plug-in baselines alongside C-USIM.

Table 5: Additional real-world datasets under a fixed total budget of 1,536 labels, averaged over 50 seeds. Each arrow denotes Plug-in HPD(1536) → C-USIM(512+1024). Marginal and group errors are computed within each seed before averaging; group measures use the representative grouping. Length uses the first 256 fixed test inputs.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Measure</td><td colspan="2">TabPFN</td></tr><tr><td>Plug-in HPD (1536) → C-USIM</td><td>TabICL</td></tr><tr><td rowspan="5">JP Anime</td><td>Marginal coverage (%)</td><td>88.54 → 94.96</td><td>91.40 → 94.96</td></tr><tr><td>Mean marginal gap (pp)</td><td>6.461 → 0.590</td><td>3.598 → 0.540</td></tr><tr><td>Mean group gap (pp)</td><td>6.224 → 2.770</td><td>4.235 → 2.837</td></tr><tr><td>CEC-X (pp)</td><td>6.598 → 2.429</td><td>4.211 → 2.521</td></tr><tr><td>Mean set length</td><td>0.447 → 0.565</td><td>0.494 → 0.539</td></tr><tr><td rowspan="5">Allstate</td><td>Marginal coverage (%)</td><td>92.86 → 94.91</td><td>93.78 → 95.05</td></tr><tr><td>Mean marginal gap (pp)</td><td>2.142 → 0.601</td><td>1.226 → 0.540</td></tr><tr><td>Mean group gap (pp)</td><td>2.240 → 1.515</td><td>1.540 → 1.523</td></tr><tr><td>CEC-X (pp)</td><td>2.186 → 1.517</td><td>1.481 → 1.561</td></tr><tr><td>Mean set length</td><td>5862.19 → 6831.03</td><td>6195.59 → 6883.54</td></tr></table>

## B.7 SPLIT-RATIO SENSITIVITY

The sweep follows the synthetic protocol in Appendix B.5, with allocations and seeds listed in Table 1. Within each mechanism and seed, all allocations and both models share a labeled pool and independent test inputs: the first $n _ { \mathrm { t r a i n } }$ pool observations form the context and the remainder form the calibration set.

Table 6: Split-ratio sensitivity across all six selected mechanisms with 1,536 total labels. Entries are means over the same 50 seeds. Gap is the per-seed absolute deviation of marginal coverage from 95%; gap and CCAD use percentage points (pp).
<table><tr><td rowspan="2">Case</td><td colspan="2">TabPFN</td><td colspan="2">TabICL</td></tr><tr><td>Gap (pp)</td><td>CCAD (pp)</td><td>Gap (pp)</td><td>CCAD (pp)</td></tr><tr><td></td><td></td><td>1229:307 (≈ 8:2) → 512:1024 (1:2)</td><td></td><td></td></tr><tr><td>1D-1</td><td>1.033 → 0.554</td><td>2.110 → 2.152</td><td>0.911 → 0.440</td><td>2.830 → 3.381</td></tr><tr><td>1D-2</td><td>0.964 → 0.497</td><td>1.998 → 2.093</td><td>1.013 → 0.561</td><td>2.370 → 2.541</td></tr><tr><td>1D-3</td><td>0.857 → 0.546</td><td>1.231 → 1.379</td><td>1.098 → 0.552</td><td>2.486 → 3.104</td></tr><tr><td>MD-1</td><td>1.026 → 0.583</td><td>2.271 → 2.218</td><td>1.119 → 0.650</td><td>2.569 → 2.634</td></tr><tr><td>MD-2</td><td>0.924 → 0.500</td><td> $1 . 6 2 6 \to 1 . 5 4 3$ </td><td>0.969 → 0.640</td><td>1.676 → 1.606</td></tr><tr><td>MD-3</td><td>1.223 → 0.624</td><td> $1 . 7 8 4  1 . 9 3 1$ </td><td>1.223 → 0.562</td><td>2.045 → 2.155</td></tr></table>

Coverage and CCAD follow Appendix B.4, with conditional coverage computed using the known CDF rather than response sampling. Mean-function RMSE is $\{ 2 5 6 ^ { - 1 } \textstyle \bar { \sum _ { i } } [ \bar { \hat { m } } ( X _ { i } ) - \bar { m ( X _ { i } ) } ] ^ { 2 } \} ^ { 1 / 2 }$ where mˆ is the mean of the reconstructed predictive density and m is the true conditional mean. Metrics are computed within each seed before averaging. Table 6 reports all six mechanisms, including cases with worse CCAD.

## B.8 SUPPLEMENTARY COMPARISON WITH IDENTICAL PREDICTIVE DENSITIES

The main experiments compare Plug-in HPD(1536) with C-USIM(512+1024) under the same total label budget. Here, we additionally compare Plug-in HPD(512) with C-USIM to examine the effect of calibrating the cutoff while holding the reconstructed predictive densities fixed. This supplementary comparison uses different numbers of response labels: Plug-in HPD(512) uses only the 512 context labels, whereas C-USIM additionally uses 1,024 held-out calibration responses.

Settings and shared predictions. Within each run, Plug-in HPD(512) and C-USIM share the context observations, model randomness, query covariates, and reconstructed densities. The plug-in regions use the boundary density level needed to attain at least 95% model mass; C-USIM uses the 974th of the 1,024 calibration scores. Both use the finite-density reconstruction in Appendix B.1 and the evaluation measures in Appendix B.4. The data splits and seeds follow Table 1, with the additional real-world datasets specified in Table 4. Synthetic comparisons use the same 1,000 con ditional response draws at each test input across all three configurations.

For Journal SJR, both 512-context configurations use the same joint table of 1,024 calibration and 9,731 test covariates in one prediction call. Plug-in HPD(512) uses the test densities from this shared output without using the calibration responses. Coverage uses all 9,731 test rows, and mean prediction-set length uses the first 256 inputs in the fixed test order for all three configurations. The inference and density reconstruction steps are shared; calibration changes the cutoff used to extract the set without fitting an additional model.

## B.8.1 SYNTHETIC

Table 7 compares Plug-in HPD(512) and C-USIM, using the same reconstructed densities and conditional response draws in each run. Mean CCAD decreases from 5.053 to 1.992 percentage points for TabPFN and from 3.540 to 2.611 for TabICL when averaged equally across the six mechanisms. Mean CCAD decreases in all twelve model–mechanism combinations, and mean predictionset length increases in each combination. Figure 20 places this comparison alongside Plug-in HPD(1536), showing how the two context sizes affect the uncalibrated regions and how calibra tion changes coverage and length at the 512 context.

This comparison also clarifies the TabICL result discussed in Appendix B.6.1 on 1D-1: mean CCAD decreases from 4.313 to 3.385 percentage points after calibration, whereas Plug-in HPD(1536) has a lower CCAD of 2.961. Thus, the increase relative to the larger-context baseline cannot be attributed to threshold calibration alone.

Table 7: Synthetic comparison using identical predictive densities from a 512-observation context. Each arrow denotes Plug-in HPD(512) → C-USIM(512+1024), averaged over ten seeds. C-USIM additionally uses 1,024 calibration responses.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Example</td><td>Coverage (%)</td><td>CCAD (pp)</td><td>Mean length</td></tr><tr><td colspan="3">Plug-in HPD(512) → C-USIM</td></tr><tr><td rowspan="6">TabPFN</td><td>1D-1</td><td>92.95 → 95.28</td><td>2.947 → 2.159</td><td>0.618 → 0.717</td></tr><tr><td>1D-2</td><td>90.24 → 94.74</td><td>5.074 → 2.186</td><td>0.805 → 1.290</td></tr><tr><td>1D-3</td><td>92.11 → 94.91</td><td>3.268 → 1.708</td><td>0.516 → 0.580</td></tr><tr><td>MD-1</td><td>88.49 → 95.33</td><td>6.569 → 2.201</td><td>0.841 → 2.100</td></tr><tr><td>MD-2</td><td>88.42 → 94.81</td><td>6.714 → 1.642</td><td>2.281 → 3.313</td></tr><tr><td>MD-3</td><td>89.77 → 95.07</td><td>5.747 → 2.054</td><td>1.783 → 2.706</td></tr><tr><td rowspan="6">TabICL</td><td>1D-1</td><td>92.77 → 95.31</td><td>4.313 → 3.385</td><td>0.643 → 0.747</td></tr><tr><td>1D-2</td><td>91.98 → 94.93</td><td>3.915 → 2.632</td><td>0.970 → 1.287</td></tr><tr><td>1D-3</td><td>92.74 → 95.13</td><td>4.058 → 3.168</td><td>0.617 → 0.725</td></tr><tr><td>MD-1</td><td>92.58 → 95.43</td><td>3.611 → 2.644</td><td>1.343 → 1.901</td></tr><tr><td>MD-2</td><td>93.57 → 95.01</td><td>2.222 → 1.572</td><td>2.833 → 3.109</td></tr><tr><td>MD-3</td><td>93.41 → 95.17</td><td>3.123 → 2.266</td><td>2.278 → 2.609</td></tr></table>

Plug-in HPD (1536) Plug-in HPD (512) C-USIM (512+1024)  
![](images/bed90b4e785ad9417ad63426d8ab8480d6f5be0277e17bbc73a3d4e22b173976.jpg)

![](images/69c2e7053e3e9b8a7d1f799f0404e46a0e9e96083e2d6ec49e65b2abbcccc7e4.jpg)

![](images/9e102ebb4151fd8a2bc3e2195ece9d13400c09a19e8b83d1e6458bf43da8227e.jpg)

![](images/6f71adcf802cc0d81fc7fb2ec5b116bf057519b8bb7ee554b132f09ecf6ef6db.jpg)

![](images/1fd09306a876491fa12ecc761a78be106dc15b8a1694ccd32d223c21950219b3.jpg)

![](images/39a976e8fd7f06e66ea36ab0273a4db7195bba996450c6db1f4df3a9806f5445.jpg)  
Figure 20: Synthetic coverage, CCAD, and prediction-set length for all three configurations. Points are means over ten seeds, not confidence intervals. Horizontal lines separate mechanisms. The two 512-context methods share predictive densities. Dashed lines mark 95% coverage. All interval components contribute to length, excluding gaps.

![](images/869f4dce3c472b72fef190be450e51837592481a08602a0ac09e32a690c53aa3.jpg)

![](images/dfafef8877262bc6400e63accc07771da1d47b9db9ca956c6e76756237ba1db3.jpg)

![](images/6a0d95423adbdc47d2144b77665ccd1cb2cfb97127bb07d5dfd5352dd0aaa48d.jpg)

![](images/498e1bc07771c85d62f5a253889352498ba3d18f2db07b5682f895bfd1e87ff7.jpg)

![](images/8eac5bbd8c1e3e261df500ea99df2f5fcc5268c383ed50ea58b081ab3577fba1.jpg)

![](images/6c524cff69d42b4a79d5009eab66d957032965d357e5d2bd4f69bb42f5674882.jpg)  
Figure 21: Journal SJR with both plug-in baselines and C-USIM, using the representative grouping. (a,d) CEC-X over 200 seeds: boxes show quartiles and medians, whiskers extend to 1.5 IQR, all outliers are retained, and colored markers denote means. (b,e) Mean coverage in each group. (c,f) Mean within-seed absolute group error. Horizontal lines in the group plots separate groups. These summaries are not confidence intervals. Dashed lines mark 95% coverage; all ten groups are retained.

Table 8: Journal SJR using identical predictive densities from a 512-observation context, averaged over 200 seeds. Each arrow denotes Plug-in HPD(512) → C-USIM(512+1024). Both configurations use the same joint calibration/test query table; only C-USIM uses the 1,024 calibration responses. Group errors use the representative grouping. Coverage uses all 9,731 test rows; length uses the first 256 inputs in the fixed test order.
<table><tr><td rowspan="2">Measure</td><td>TabPFN</td><td>TabICL</td></tr><tr><td>Plug-in HPD(512) → C-USIM</td><td></td></tr><tr><td>Marginal coverage (%)</td><td>91.06 → 94.83</td><td>91.72 → 94.87</td></tr><tr><td>Mean marginal gap (pp)</td><td>3.964 → 0.553</td><td>3.431 → 0.611</td></tr><tr><td>Mean group gap (pp)</td><td>3.846 → 1.760</td><td>3.630 → 1.856</td></tr><tr><td>CEC-X (pp)</td><td>4.333 → 1.662</td><td>3.863 → 1.707</td></tr><tr><td>Mean set length</td><td>1.261 → 1.453</td><td>1.359 → 1.540</td></tr></table>

## B.8.2 JOURNAL SJR

Table 8 reports the results over 200 seeds per model. In the representative grouping, mean CEC-X decreases from 4.333 to 1.662 percentage points for TabPFN and from 3.863 to 1.707 for TabICL. Relative to Plug-in HPD(512), C-USIM lowers mean CEC-X in all 30 groupings for each model. In the representative grouping, mean absolute group error decreases from 3.846 to 1.760 percentage points for TabPFN and from 3.630 to 1.856 for TabICL. Mean error decreases in 7 of the ten groups for TabPFN and 6 for TabICL; 66.05% and 61.50% of seed–group pairs, respectively, move closer to the target. Across all groupings, these fractions range from 46.38% to 70.00% for TabPFN and from 43.26% to 64.00% for TabICL. All nonempty groups remain in the averages and Figure 21 shows all ten representative groups.

The discussion in Appendix B.6.2 also applies here. Mean prediction-set length increases from 1.261 to 1.453 for TabPFN and from 1.359 to 1.540 for TabICL, as reported in Table 8.

## B.8.3 ADDITIONAL REAL-WORLD DATASETS

Table 9 compares Plug-in HPD(512) with C-USIM using identical predictive densities. Both methods use the same joint calibration/test table for each seed; only C-USIM uses the 1,024 calibration responses. On JP Anime, mean CEC-X decreases from 4.681 to 2.429 percentage points for TabPFN and from 3.103 to 2.521 for TabICL. On Allstate, the corresponding changes are 2.667 to 1.517 and 2.277 to 1.561. CEC-X decreases in all 30 groupings for each of the four dataset–model pairs, and mean prediction-set length increases in each pair.

The Allstate TabICL result illustrates the distinction between the two comparisons: calibrating the same 512-context densities lowers CEC-X, while comparison with Plug-in HPD(1536) also changes the predictor and gives the smaller, grouping-sensitive differences described in Appendix B.6.3.

Table 9: Additional real-world datasets using identical predictive densities from a 512-observation context, averaged over 50 seeds. Each arrow denotes Plug-in HPD(512) → C-USIM(512+1024). Only C-USIM uses the additional 1,024 calibration responses. Group measures use the representative grouping; response scales and test sizes follow Table 4.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Measure</td><td>TabPFN</td><td>TabICL</td></tr><tr><td>Plug-in HPD (512) → C-USIM</td><td></td></tr><tr><td rowspan="5">JP Anime</td><td>Marginal coverage (%)</td><td>91.05 → 94.96</td><td>94.01 → 94.96</td></tr><tr><td>Mean marginal gap (pp)</td><td>3.974 → 0.590</td><td>1.853 → 0.540</td></tr><tr><td>Mean group gap (pp)</td><td>4.697 → 2.770</td><td>3.330 → 2.837</td></tr><tr><td>CEC-X (pp)</td><td>4.681 → 2.429</td><td>3.103 → 2.521</td></tr><tr><td>Mean set length</td><td>0.486 → 0.565</td><td>0.522 → 0.539</td></tr><tr><td rowspan="5">Allstate</td><td>Marginal coverage (%)</td><td>92.43 → 94.91</td><td>93.03 → 95.05</td></tr><tr><td>Mean marginal gap (pp)</td><td>2.568 → 0.601</td><td>2.001 → 0.540</td></tr><tr><td>Mean group gap (pp)</td><td>2.646 → 1.515</td><td>2.279 → 1.523</td></tr><tr><td>CEC-X (pp)</td><td>2.667 → 1.517</td><td>2.277 → 1.561</td></tr><tr><td>Mean set length</td><td>5935.33 → 6831.03</td><td>6145.43 → 6883.54</td></tr></table>

![](images/20a2e4fe929d9937856e7cefd0c868bb1d16265ef344f3463e91c734f5916c42.jpg)

![](images/ce9616a0458c9aa5422f37a6a3289e0bd60fb607111242d0c1b7e3aa3ad21d6b.jpg)

![](images/d63b6de5fe7999ca2b94dfe580d251e687c1e325ee98c7b2fc342fd0cfd9b85c.jpg)

![](images/76fe1409ebc1fc5a566f0a1db5c2d2e46849b25bf0729ae00f387c8f27acf9a9.jpg)

![](images/f4e6a1b356d7d3c7749e85e8b5aea7385defd49432a2365751b6e9903c96d85c.jpg)

![](images/e925c36e46756fd6fa8c5d766f8653e6fc7f6c67a4076d4b6de6897ba68c94e4.jpg)

![](images/9ef73b5dcd5e6c2bb86de92543d9ac8d54086bdceb7cd23b33faf1415a580055.jpg)  
Figure 22: JP Anime with both plug-in baselines and C-USIM, using the representative grouping. (a,d) CEC-X over 50 seeds: boxes show quartiles and medians, whiskers extend to 1.5 IQR, all outliers are retained, and colored markers denote means. (b,e) Mean group coverage. (c,f) Mean within-seed absolute group error. Horizontal lines separate groups; group 9 has 15 test observations. Dashed lines mark 95% coverage. These summaries are not confidence intervals.  
Figure 23: Allstate Claims Severity with both plug-in baselines and C-USIM, using the representative grouping. (a,d) CEC-X over 50 seeds: boxes show quartiles and medians, whiskers extend to 1.5 IQR, all outliers are retained, and colored markers denote means. (b,e) Mean group coverage. (c,f) Mean within-seed absolute group error. Horizontal lines separate all ten groups; dashed lines mark 95% coverage. These summaries are not confidence intervals.