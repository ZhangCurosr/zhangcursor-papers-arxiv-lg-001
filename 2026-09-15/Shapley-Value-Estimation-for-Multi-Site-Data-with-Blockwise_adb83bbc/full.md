# Shapley Value Estimation for Multi-Site Data with Blockwise-Missing Features

Siqi Li<sup>∗1</sup>, Wangxuan Fan<sup>2</sup>, Yiming Li<sup>3</sup>, Doudou Zhou<sup>4</sup>, and Molei Liu<sup>5,6</sup>

<sup>1</sup>Centre for Biomedical Data Science, Duke-NUS Medical School <sup>2</sup>School of Data Science, Chinese University of Hong Kong, Shenzhen <sup>3</sup>Department of Biostatistics, Columbia University   
<sup>4</sup>Department of Statistics and Data Science, National University of Singapore <sup>5</sup>Department of Biostatistics, Peking University   
<sup>6</sup>Beijing International Center for Mathematical Research, Peking University

## Abstract

Shapley value (SV)-based methods are the prevailing framework for feature attribution in machine learning, yet existing population-level Shapley estimators generally assume that observations used to evaluate the coalitional game are fully observed under a common feature space. This assumption is routinely violated in multi-site studies across biomedicine, social science, and environmental monitoring, where institutions record diferent features under diferent protocols, producing systematic blockwise missingness across sources. We first show that the standard remedy of imputing missing features before computing Shapley values introduces systematic, coalition-dependent bias into the resulting attributions. We then propose FUSHAP (Fusion Shapley Attribution from Partially-observed data), a method that leverages partially-observed auxiliary sites to reduce the variance of a preliminary single-site Shapley estimate without imputation. A permutation-based screening step detects and excludes sites whose data distributions are incompatible with the target population. In synthetic experiments, FUSHAP achieves 3–8× lower MSE than the single-site estimator and 2–3× lower MSE than imputation baselines without incurring imputation-induced bias, and the screening procedure identifies misaligned sites with 82% power at moderate misalignment and 100% for strong misalignment. On multi-site air quality and multi-center clinical data, FUSHAP reduces MSE by approximately 3–7× relative to the single-site estimator; in the clinical application, standard imputation can increase MSE above the single-site baseline.

Keywords: Shapley values, feature attribution, blockwise missing data, multi-source data fusion, model interpretability, influence functions, variance reduction, control variates.

## 1 Introduction

Model interpretability has become an important component of trustworthy machine learning, particularly in high-stakes domains such as medicine, credit scoring, and criminal justice where practitioners must understand why a model produces a given prediction before acting on it. Among approaches to post-hoc explanation, Shapley value (SV)-based feature attribution methods, exemplified by SHAP (Lundberg and Lee, 2017) and SAGE (Covert et al., 2020), have emerged as the principled standard, owing to their axiomatic foundation in cooperative game theory and modelagnostic applicability (Molnar, 2020; Mosca et al., 2022; Li et al., 2024; Salih et al., 2025). Given $p$ input features and a predictive model $f ,$ , SV methods define a cooperative game $v ( S )$ that measures the predictive performance of $f$ when only features in $S \subseteq \{ 1 , \ldots , p \}$ are available, with absent features marginalized over a reference distribution. The $\mathrm { S V } ~ \phi _ { j }$ of feature $j$ is then the weighted average of $j ^ { \dag } \mathrm { s }$ marginal contribution $v ( S \cup \{ j \} ) - v ( S )$ across all $2 ^ { p - 1 }$ coalitions not containing $j ;$ estimation of $\phi = ( \phi _ { 1 } , \ldots , \phi _ { p } )$ requires evaluating $v ( S )$ for many coalitions against a reference sample.

A fundamental assumption underlying most existing population-level SV estimators, including KernelSHAP (Lundberg and Lee, 2017; Covert and Lee, 2021), SAGE (Covert et al., 2020), Fast-SHAP (Jethani et al., 2022), and SIM-Shapley (Fan et al., 2025), is that this reference sample is drawn from a single, fully-observed data source. In practice, any scientific study that integrates data across multiple sources, whether clinical registries, sensor networks, or multi-cohort social surveys, risks violating this assumption, as diferent sources typically measure diferent variables and produce blockwise missingness in which entire groups of features are systematically absent at certain sites. Clinical research provides a particularly prominent example: large-scale healthcare consortia aggregate records from dozens of institutions, each administering diferent diagnostic protocols, so that certain imaging, laboratory, or cognitive assessments are entirely unavailable at certain centers (Li et al., 2025, 2026). To our knowledge, existing SV methods do not explicitly address data partitioned across sites with disjoint blocks of unobserved features and potential distributional shift.

Two natural strategies exist for computing SV from blockwise-missing multi-site data, but neither is fundamentally adequate. The most straightforward approach is to compute feature-level SV independently at each site and average the results, which ignores distributional diferences across sites and provides no mechanism for detecting sources whose data are incompatible with the target population. The other natural remedy is to impute the missing features and proceed with standard SV estimation, but this can be more problematic: imputation alters the covariance structure on which Shapley attributions depend, introducing method-dependent bias into feature importance rankings that does not necessarily diminish with improved predictive accuracy (Vo et al., 2025).

## 1.1 Related Work

Shapley values under incomplete data. Computing SV requires specifying how absent features are handled within each coalition $S ;$ the choice among conditional, marginal, and baseline removal strategies materially afects the resulting attributions (Chen et al., 2023; Covert et al., 2021), and which convention is preferable remains open and context-dependent. The majority of recent methodological work has focused on improving the computational eficiency of SV estimation under a fixed removal convention (Covert and Lee, 2021; Fan et al., 2025; Mitchell et al., 2022), leaving the statistical challenge of heterogeneous, partially-observed evaluation data largely unaddressed.

A compounding dificulty arises when the evaluation data themselves contain missing entries. Under missingness, standard imputation yields a surrogate distribution ${ \widetilde { P } } _ { X } \neq P _ { X }$ that does not correct for the shift between observed and full data (Shannon et al., 2026; N¨af et al., 2026). Since all common SV formulations define the coalitional game through expectations with respect to $P _ { X }$ this distributional error propagates directly into the value function $\mathcal { V } ( S )$ and hence into every attribution. Empirically, Vo et al. (2025) confirm that diferent imputation strategies produce systematically divergent Shapley attributions. Yet no existing work provides a correction for this bias under structured blockwise missingness.

Data fusion under blockwise missingness. Several recent works address estimation from multi-source data in which diferent sources observe diferent variable subsets. Xue and Qu (2021) integrate multiple conditional-mean imputations, each derived from a distinct overlap of observed covariates across block-wise missing-pattern groups, within a penalized generalized method of moments (GMM). Jin and Rothenh¨ausler (2023) propose a modular regression framework that leverages auxiliary variables satisfying a conditional independence structure to improve estimation eficiency and prediction accuracy. Li et al. (2025) develop a data-adaptive control-variate framework that handles both blockwise missingness and distributional shift for generalized linear model coefficients. Xu et al. (2025) and Huang et al. (2025) extend similar ideas to broader parameter classes under block-missing designs.

In all such cases, the target estimand is defined by a single estimating equation or a small system of moment conditions. The Shapley attribution vector $\phi \in \mathbb { R } ^ { p }$ , while also finite-dimensional, is defined through $2 ^ { p }$ coalition-level value functions, each involving a separate conditional expectation, which is a structure absent from prior data-fusion targets. How to extend variance-reduction techniques from scalar estimands to this combinatorial setting remains an open problem.

Shapley values in multi-site and federated settings. Several works employ Shapley values in multi-site contexts, but target fundamentally diferent estimands from ours. One active line assigns a single Shapley value to each site, i.e., quantifying how much each data source contributes to the overall model, rather than to each feature within the model (Wang et al., 2020; Zheng et al., 2023; Liu et al., 2022). This is client-level data valuation: the players in the cooperative game are institutions, not input variables, and blockwise missingness plays no role. Wu et al. (2021) use SV to explain performance disparities across clinical sites, but treat site-level confounders (demographics, equipment type) as the players rather than model features. In all of these formulations, each site has access to the same feature space; the heterogeneous feature coverage that defines blockwise missingness is absent.

## 1.2 Contributions

We propose FUSHAP (Fusion Shapley Attribution from Partially-observed data), a framework for estimating Shapley feature attributions from multi-site data with blockwise-missing covariates, without resorting to imputation. Our contributions are as follows.

1. Imputation bias in blockwise-missing settings. Extending the empirical findings of Vo et al. (2025), we confirm that imputing missing features before computing Shapley values introduces systematic, coalition-dependent bias that persists across the standard imputation methods considered, with MSE up to 3.1× that of imputation-free alternatives.

2. Variance-reduced estimation without imputation. We derive the influence function of the constrained WLS Shapley estimator and use it to construct control variate corrections from blockwise-missing auxiliary sites, reducing variance without imputing unobserved features.

3. Adaptive source screening and calibration. We develop a permutation-based screening procedure that detects incompatible sites, and a total-variance calibration that optimally weights each site’s contribution.

4. Empirical validation. Across simulations and multi-site real data, FUSHAP reduces MSE by 3–8× relative to the single-site estimator and 2–3× relative to imputation baselines in simulations, with approximately 3–7× improvements over the single-site estimator on real data.

## 2 Problem Formulation

## 2.1 Data structure

Let Y denote the outcome of interest and $\mathbf { X } = ( X _ { 1 } , \ldots , X _ { p } ) ^ { \top }$ a p-dimensional feature vector. We consider a multi-site setting with three types of data source:

• Labeled complete (LC), of size n: both (Y, X) are jointly observed.

• Labeled missing $( \mathcal { L M } _ { r } , r = 1 , \ldots , R )$ , of size $n _ { r }$ : the outcome Y and a subset $\mathbf { X } _ { \Gamma _ { r } }$ of covariates are observed, where $\Gamma _ { r } \subsetneq \{ 1 , \dotsc , p \}$ . The remaining covariates $\mathbf { X } _ { \Gamma _ { r } ^ { c } }$ are entirely unobserved.

• Unlabeled complete (UC), of size $N \gg n \colon$ all covariates X are observed but Y is unavailable.

The UC sample defines the target population on which inference is desired. We assume centralized access to row-level data from all sources. Throughout, we denote by $\mathcal { D } = \{ 1 , \ldots , p \}$ the full index set and $\rho _ { r } = n _ { r } / n$ the sample-size ratio of the r-th labeled-missing source to the complete source. Sources may difer in their marginal covariate distributions; identification relies on conditional alignment of the outcome and remaining features given an observed alignment set, formalized below.

Assumption 1 (Missing at random with suficient alignment). For each site $\begin{array} { r } { \mathcal { L M } _ { r } , r = 1 , \ldots , R , } \end{array}$ there exists a suficient alignment set $\Omega _ { r } \subseteq \Gamma _ { r }$ such that

$$
\begin{array} { r } { p _ { \mathcal { U C } } \left( Y , \mathbf { X } _ { \Omega _ { r } ^ { c } } \mid \mathbf { X } _ { \Omega _ { r } } \right) ~ = ~ p _ { \mathcal { L M } _ { r } } \left( Y , \mathbf { X } _ { \Omega _ { r } ^ { c } } \mid \mathbf { X } _ { \Omega _ { r } } \right) . } \end{array}\tag{1}
$$

Assumption 1 requires that, conditional on the alignment variables $\mathbf { X } _ { \Omega _ { r } }$ , the joint distribution of the outcome and remaining features is the same at site r and in the target population. This generalizes the missing-completely-at-random (MCAR) condition commonly adopted in the blockwise-missing literature (Xue and Qu, 2021; Jin and Rothenh¨ausler, 2023): when $\Omega _ { r } ~ = ~ \mathcal { O }$ 2 (1) reduces to ${ \mathrm { M C A R } } ;$ when $\Omega _ { r } = \Gamma _ { r } $ , arbitrary marginal shift in X is permitted provided the conditional distributions agree (Li et al., 2025).

## 2.2 Shapley feature attribution

Let $f : \mathbb { R } ^ { p } $ R be a fixed, pre-trained predictive model and $\ell : \mathbb { R } \times \mathbb { R } \to \mathbb { R } _ { \geq 0 }$ a loss function. For each coalition $S \subseteq { \mathcal { D } }$ , define the restricted prediction

$$
\bar { f } _ { S } ( \mathbf { x } _ { S } ) = \mathbb { E } _ { p u c } \big [ f ( \mathbf { x } _ { S } , \mathbf { X } _ { S ^ { c } } ) \big ] ,\tag{2}
$$

which marginalizes the absent features $\mathbf { X } _ { S ^ { c } }$ over their marginal distribution under $p u c$ , independently of $\mathbf { X } _ { S }$ . This is the marginal feature removal convention (Lundberg and Lee, 2017; Fan et al., 2025); see Remark 1 for the conditional alternative.

The SAGE cooperative game (Covert et al., 2020) assigns to each coalition $S \subseteq { \mathcal { D } }$ the value

$$
\mathcal { V } ( S ) = - \mathbb { E } _ { p _ { U C } } [ \ell ( \bar { f } _ { S } ( \mathbf { X } _ { S } ) , \ : Y ) ] ,\tag{3}
$$

the negated expected loss under coalition S, with larger values indicating better predictive performance. Encoding coalitions as binary vectors $\mathbf { z } \in \{ 0 , 1 \} ^ { p }$ via $S ( \mathbf { z } ) = \{ j : z _ { j } = 1 \}$ , we write $\mathcal { V } ( \mathbf { z } )$ and $\mathcal { V } ( S )$ interchangeably. The population value function admits a per-observation decomposition $\mathcal { V } ( \mathbf { z } ) = \mathbb { E } _ { p _ { \mathit { M C } } } [ \eta ( \mathbf { z } , \mathbf { X } , Y ) ]$ , where

$$
\eta ( \mathbf { z } , \mathbf { x } , y ) ~ = ~ - \ell \big ( \bar { f } _ { S ( \mathbf { z } ) } ( \mathbf { x } _ { S ( \mathbf { z } ) } ) , ~ y \big )\tag{4}
$$

records the negated loss for a single observation $\left( \mathbf { x } , y \right)$ under coalition z.

The SV of feature j is the weighted average of its marginal contribution $\mathcal { V } ( S \cup \{ j \} ) - \mathcal { V } ( S )$ over

all coalitions $S \geqslant j$ , uniquely characterized by the eficiency, symmetry, linearity, and null-player axioms (Covert et al., 2020). Equivalently, $\bar { \boldsymbol { \phi } } = ( \phi _ { 1 } , \ldots , \phi _ { p } ) ^ { \top }$ is the solution to the constrained weighted least squares (WLS) problem (Lundberg and Lee, 2017; Covert and Lee, 2021)

$$
\bar { \phi } ~ = ~ \arg \operatorname* { m i n } _ { \beta \in \mathbb { R } ^ { p } } ~ \mathbb { E } _ { \mu \mathrm { S h } } \left[ \left( \mathcal { V } ( \mathbf { 0 } ) + \mathbf { z } ^ { \top } \beta - \mathcal { V } ( \mathbf { z } ) \right) ^ { 2 } \right] ~ \mathrm { s . t . } ~ \mathbf { 1 } ^ { \top } \beta = \mathcal { V } ( \mathbf { 1 } ) - \mathcal { V } ( \mathbf { 0 } ) ,\tag{5}
$$

where $\mu _ { \mathrm { S h } }$ is the Shapley kernel, the distribution over coalitions of intermediate size $( 0 < | { \mathbf { z } } | < p )$ with probability mass $\mu _ { \mathrm { S h } } ( \mathbf { z } ) \propto [ \binom { p } { | \mathbf { z } | } | \mathbf { z } | ( p - | \mathbf { z } | ) ] ^ { - 1 }$ , and the eficiency constraint ensures that the attributions sum to the diference between full-model and null-model performance. The KKT conditions yield (Covert and Lee, 2021; Fan et al., 2025)

$$
\bar { \phi } = \Sigma ^ { - 1 } \bigg [ \mathbf { b } + \mathbf { 1 } \frac { c - \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 } \mathbf { b } } { \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 } \mathbf { 1 } } \bigg ] ,\tag{6}
$$

with

$$
\Sigma = \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ \mathbf { z } \mathbf { z } ^ { \top } ] , \qquad \mathbf { b } = \mathbb { E } _ { \mu _ { \mathrm { S h } } } \big [ \mathbf { z } \big ( \mathcal { V } ( \mathbf { z } ) - \mathcal { V } ( \mathbf { 0 } ) \big ) \big ] , \qquad c = \mathcal { V } ( \mathbf { 1 } ) - \mathcal { V } ( \mathbf { 0 } ) .\tag{7}
$$

In practice, the expectation over the Shapley kernel is approximated using m sampled coalitions $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { m } \stackrel { \mathrm { i . i . d . } } { \sim } \mu _ { \mathrm { S h } }$ , where m denotes the coalition-sampling budget. Let $\begin{array} { r } { A = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } } \end{array}$ , b<sup>¯</sup> = <sup>1</sup><sub>m</sub> $\begin{array} { r } { \sum _ { j = 1 } ^ { m } \mathbf { z } _ { j } ( \widehat { \mathcal { V } } ( \mathbf { z } _ { j } ) - \widehat { \mathcal { V } } ( \mathbf { 0 } ) ) } \end{array}$ , and $\widehat { c } = \widehat { \mathcal { V } } ( \mathbf { 1 } ) - \widehat { \mathcal { V } } ( \mathbf { 0 } )$ . Note that $A { \stackrel { p } { \to } } \Sigma$ and $\bar { \mathbf { b } } \overset { p } {  }$ b as $m \to \infty$

Since Σ depends only on the Shapley kernel (not on the data), the data-dependent part of $\phi$ enters entirely through b and c. Both are expectations over the UC population:

$$
\begin{array} { r } { \mathbf { b } = \mathbb { E } _ { \mathbf { z } } \big [ \mathbf { z } \mathbb { E } _ { ( \mathbf { X } , Y ) \sim p _ { U C } } \big [ \eta ( \mathbf { z } , \mathbf { X } , Y ) \big ] \big ] - \mathcal { V } ( \mathbf { 0 } ) \mathbb { E } _ { \mathbf { z } } [ \mathbf { z } ] , } \end{array}\tag{8}
$$

$$
c = \mathbb { E } _ { ( \mathbf { X } , Y ) \sim p _ { U C } } [ \eta ( \mathbf { 1 } , \mathbf { X } , Y ) - \eta ( \mathbf { 0 } , \mathbf { X } , Y ) ] .\tag{9}
$$

The estimand $\bar { \phi }$ is defined via expectations under $p _ { \mathcal { U } \mathcal { C } }$ , but evaluating $\eta ( \mathbf { z } , \mathbf { x } , y )$ requires both the outcome Y and all features X. Only LC possesses both, yet its sample size n is typically small, yielding a high-variance estimate, and its covariate distribution may difer from the target population $p _ { \mathcal { U } \mathcal { C } }$ , introducing bias. The $\mathcal { L } \mathcal { M } _ { r }$ sources provide additional labeled observations but lack the features in $\Gamma _ { r } ^ { c }$ ; the UC source provides the complete feature vector but no outcome. The central question addressed in this paper is whether these partially-observed data sources can reduce the variance of the LC-only estimator without introducing bias.

Remark 1 (Feature removal convention). Equation (2) adopts the marginal removal convention, in which $\mathbf { X } _ { S ^ { c } }$ is drawn independently of $\mathbf { X } _ { S }$ . The conditional alternative $\bar { f } _ { S } ( \mathbf { x } _ { S } ) = \mathbb { E } _ { p _ { \mathcal { U C } } } [ f ( \mathbf { X } )$ | $\mathbf { X } _ { S } ~ = ~ \mathbf { x } _ { S } ]$ preserves feature dependencies but requires estimating high-dimensional conditional distributions. The estimation framework in Section 3 is agnostic to this choice: it requires only that $\mathcal { V } ( \mathbf { z } ) = \mathbb { E } _ { p _ { \mathit { M C } } } [ \eta ( \mathbf { z } , \mathbf { X } , Y ) ]$ for some per-observation function η, a property satisfied under either convention.

## 3 Method

The goal is to estimate the population Shapley vector $\bar { \phi } ( p _ { U C } , f )$ , defined with respect to the $\mathcal { U } \mathcal { C }$ covariate distribution and the fixed, pre-trained model $f ,$ using data from the three source types described in Section 2.1.

## 3.1 Preliminary estimator from the complete-data site

Given m coalitions $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { m }$ drawn from the Shapley kernel $\mu _ { \mathrm { S h } }$ , the value function $\mathcal { V } ( \mathbf { z } )$ is estimated from LC by the importance-weighted sample average

$$
{ \widehat { \mathcal { V } } } ( \mathbf { z } ) = { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } { \widehat { w } } ( \mathbf { x } _ { i } ) \eta ( \mathbf { z } , \mathbf { x } _ { i } , y _ { i } ) ,\tag{10}
$$

where the summation runs over $\mathcal { L } \mathcal { C }$ observations. The density ratio $\widehat { w } ( \mathbf { x } ) \ = \ \widehat { p } \varkappa c ( \mathbf { x } ) / \widehat { p } \angle c ( \mathbf { x } )$ reweights $\mathcal { L } \mathcal { C }$ to the target distribution $p _ { \mathcal { U } \mathcal { C } }$ and is estimated separately by training a binary classifier on $\mathcal { L C } \cup \mathcal { U } \mathcal { C }$ with source indicators; $\widehat { w } \equiv 1$ when no covariate shift is present. The preliminary Shapley estimator $\widetilde { \phi }$ is then the closed-form solution (6) with $( \Sigma , \mathbf { b } , c )$ replaced by their sample analogues

$$
\begin{array} { r } { \widehat { \Sigma } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } , \quad \widehat { \mathbf { b } } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { z } _ { j } \big ( \widehat { \mathcal { V } } ( \mathbf { z } _ { j } ) - \widehat { \mathcal { V } } ( \mathbf { 0 } ) \big ) , \quad \widehat { c } = \widehat { \mathcal { V } } ( \mathbf { 1 } ) - \widehat { \mathcal { V } } ( \mathbf { 0 } ) . } \end{array}\tag{11}
$$

## 3.2 Influence function of the WLS Shapley estimator

The preliminary estimator $\widetilde { \phi }$ depends on $p _ { \mathcal { U } \mathcal { C } }$ only through b and c in (7); the matrix Σ is determined by the Shapley kernel alone. Replacing the population expectation by the $\mathcal { L } \mathcal { C }$ sample average perturbs b and $^ { c , }$ and the resulting perturbation of $\phi$ can be expressed in terms of per-observation contributions via the chain rule.

For each observation $\left( \mathbf { x } , y \right)$ and coalition $\mathbf { z } ,$ let

$$
\epsilon ( { \bf z } , { \bf x } , y ) = \eta ( { \bf z } , { \bf x } , y ) - \mathcal { V } ( { \bf z } )\tag{12}
$$

denote the residual of the per-observation value contribution about its population mean, and define

$$
\mathbf { g } ( \mathbf { x } , y ) = \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ \mathbf { z } \epsilon ( \mathbf { z } , \mathbf { x } , y ) ] .\tag{13}
$$

The influence function of the WLS Shapley estimator (6) is (the full derivation is given in $\mathrm { A p \mathrm { - } }$ pendix B)

$$
\bar { \psi } ( \mathbf { x } , y ) = \Sigma ^ { - 1 } \big [ \mathbf { g } ( \mathbf { x } , y ) + \lambda ( \mathbf { x } , y ) \mathbf { 1 } \big ] ,\tag{14}
$$

where

$$
\lambda ( \mathbf { x } , y ) ~ = ~ { \frac { \left( { \boldsymbol { \eta } } ( \mathbf { 1 } , \mathbf { x } , y ) - { \boldsymbol { \eta } } ( \mathbf { 0 } , \mathbf { x } , y ) \right) - c - \mathbf { 1 } ^ { \top } { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { g } ( \mathbf { x } , y ) } { \mathbf { 1 } ^ { \top } { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { 1 } } }\tag{15}
$$

is the Lagrange correction enforcing the eficiency constraint at the observation level. Since the closed-form solution (6) is linear in $( \mathbf { b } , c )$ with Σ fixed, the estimation error decomposes as

$$
\tilde { \phi } - \bar { \phi } \ = \ \frac { 1 } { n } \sum _ { i \in \mathcal { L C } } \bar { \psi } ( \mathbf { x } _ { i } , y _ { i } ) \ + \ O _ { p } ( m ^ { - 1 / 2 } ) ,\tag{16}
$$

where $\hat { \psi }$ is given by (14)–(15) and the remainder arises from approximating Σ and b with m sampled coalitions. The leading term is exact in the data-sampling component, as no higher-order remainder in n is incurred.

Remark 2 (Extension to covariate shift). The influence function in (14)–(15) is derived under $p _ { \mathcal { L } \mathcal { C } } = p _ { \mathcal { U } \mathcal { C } }$ . Suppose instead that the target covariate distribution is absolutely continuous with respect to the $\mathcal { L } \mathcal { C }$ distribution, with density ratio $w ( { \bf x } ) = p _ { \ / { M C } } ( { \bf x } ) / p _ { \ / { C C } } ( { \bf x } )$ . When $w$ is known, the same derivation applies after replacing $\eta ( \mathbf { z } , \mathbf { x } , y )$ by $w ( { \bf x } ) \eta ( { \bf z } , { \bf x } , y )$ . In particular, the weighted residual is $\epsilon _ { w } ( \mathbf { z } , \mathbf { x } , y ) = w ( \mathbf { x } ) \eta ( \mathbf { z } , \mathbf { x } , y ) - \mathcal { V } ( \mathbf { z } )$ , and $\mathbf { g } , \lambda ,$ , and $\hat { \psi }$ are defined analogously using $\epsilon _ { w }$

In practice, w is replaced by an estimate $\widehat { w }$ obtained from the $\mathcal { L } \mathcal { C }$ and $\mathcal { U } \mathcal { C }$ covariates. The linearization in (16) continues to hold with the same first-order influence function whenever the contribution from density-ratio estimation is asymptotically negligible. A suficient condition is $\| \widehat { w } - w \| _ { \infty } = o _ { p } ( n ^ { - 1 / 2 } )$ , together with appropriate moment and regularity conditions on $\eta .$ . We treat this condition as an assumption in the present analysis; more generally, when density-ratio estimation contributes at first order, an orthogonal/debiased construction is required to account for this additional nuisance-estimation error.

## 3.3 Variance reduction via partially-observed sites

The decomposition (16) reveals the structure that enables variance reduction. The estimation error is a sample average of per-observation contributions $\bar { \psi } ( { \bf x } _ { i } , y _ { i } )$ whose population mean is zero: $\mathbb { E } _ { p _ { \mathcal { U } } c } [ \bar { \psi } ( \mathbf { X } , Y ) ] = \mathbf { 0 }$ . Each $\bar { \psi } ( \mathbf { x } , y )$ depends on the full feature vector x and the outcome $y ,$ both partially available at $\mathcal { L } \mathcal { M } _ { r }$ , which observes $( \mathbf { X } _ { \Gamma _ { r } } , Y )$

The component of $\hat { \psi }$ predictable from these observed variables is the conditional expectation

$$
\boldsymbol { \tau } _ { r } ^ { * } ( \mathbf { X } _ { \Gamma _ { r } } , Y ) \ = \ \mathbb { E } \big [ \bar { \psi } ( \mathbf { X } , Y ) \big | \mathbf { X } _ { \Gamma _ { r } } , Y \big ] .\tag{17}
$$

In practice $\boldsymbol { \tau } _ { r } ^ { * }$ is unknown. We estimate it from the $\mathcal { L } \mathcal { C }$ sample by regressing the estimated influence function $\widetilde { \psi } ( \mathbf { x } _ { i } , y _ { i } )$ on $( \mathbf { X } _ { \Gamma _ { r } } , Y )$ via cross-fitted ridge regression, yielding $\widehat { \tau } _ { r } ( \mathbf { X } _ { \Gamma _ { r } } , Y )$

The augmented estimator. The preliminary estimate is corrected by adding, for each site, the diference between the $\mathcal { L } \mathcal { M } _ { \tau }$ <sub>r</sub> and $\mathcal { L } \mathcal { C }$ averages of the estimated control variate:

$$
\widehat { \phi } _ { \mathrm { a u g \it ~ = ~ \widetilde { \phi } ~ + ~ } } \sum _ { r = 1 } ^ { R } \bigg \{ \frac { 1 } { n _ { r } } \sum _ { ( { \bf x } , y ) \in \mathcal { L } \mathcal { M } _ { r } } \widehat { \tau } _ { r } ( { \bf x } _ { \Gamma _ { r } } , y ) - \frac { 1 } { n } \sum _ { ( { \bf x } , y ) \in \mathcal { L } \mathcal { C } } \widehat { \tau } _ { r } ( { \bf x } _ { \Gamma _ { r } } , y ) \bigg \} .\tag{18}
$$

When $p _ { \mathcal { L } \mathcal { C } } = p _ { \mathcal { L } \mathcal { M } _ { r } } = p _ { \mathcal { U } \mathcal { C } }$ , each correction term has population mean zero and the augmentation

reduces variance without introducing bias. Under covariate shift, both averages require importance weighting to the target distribution $p _ { \mathcal U C } ;$ the generalization is given in (21) of Section 3.5.

## 3.4 Screening for misaligned sites

The augmented estimator (18) benefits from site r only if the correction term $\widehat { \tau } _ { r }$ estimated on $\mathcal { L } \mathcal { M } _ { r }$ is consistent with the same quantity estimated on $\mathcal { L } \mathcal { C } .$ . When the two disagree systematically, whether due to distributional incompatibility between site r and the target population or because the regression $\widehat { \tau } _ { r }$ extrapolates poorly on $\mathcal { L } \mathcal { M } _ { r }$ data, including site r degrades rather than improves the estimate.

For each site $r ,$ we compare the importance-weighted averages of $\widehat { \tau } _ { r }$ computed on $\mathcal { L } \mathcal { M } _ { r }$ and $\mathcal { L } \mathcal { C }$ . Define

$$
\bar { \tau } _ { r } ^ { \mathcal { L M } } = \frac { 1 } { n _ { r } } \sum _ { ( \mathbf x , y ) \in \mathcal { L M } _ { r } } \widehat { w } _ { r } ( \mathbf x _ { \Gamma _ { r } } ) \widehat { \tau } _ { r } ( \mathbf x _ { \Gamma _ { r } } , y ) , \qquad \bar { \tau } _ { r } ^ { \mathcal { L C } } = \frac { 1 } { n } \sum _ { ( \mathbf x , y ) \in \mathcal { L C } } \widehat { w } ( \mathbf x ) \widehat { \tau } _ { r } ( \mathbf x _ { \Gamma _ { r } } , y ) ,\tag{19}
$$

where $\widehat { w } _ { r } ( \mathbf { x } _ { \Gamma _ { r } } ) = \widehat { p } _ { \mathcal { U C } } ( \mathbf { x } _ { \Gamma _ { r } } ) / \widehat { p } _ { \mathcal { L M } _ { r } } ( \mathbf { x } _ { \Gamma _ { r } } )$ and $\widehat { w } ( \mathbf { x } ) = \widehat { p } \varkappa c ( \mathbf { x } ) / \widehat { p } c c ( \mathbf { x } )$ are the density ratios. Both reweight to $p _ { \mathcal { U C } }$ , so under Assumption 1 their diference $\mathbf { d } _ { r } = \bar { \pmb { \tau } } _ { r } ^ { \mathcal { L } \mathcal { M } } - \bar { \pmb { \tau } } _ { r } ^ { \mathcal { L } \mathcal { C } } \in \mathbb { R } ^ { p }$ has population mean zero. To ensure that coordinates with noisier importance weights do not dominate the comparison, we studentize the diference. The test statistic is

$$
T _ { r } \ = \ \sum _ { j = 1 } ^ { p } \frac { d _ { r , j } ^ { 2 } } { \sqrt { \mathrm { a r } } ( d _ { r , j } ) } ,\tag{20}
$$

where $\widehat { \mathrm { V a r } } ( d _ { r , j } ) = \widehat { \mathrm { V a r } } _ { \mathcal { L } \mathcal { C } } ( \widehat { w } \widehat { \tau } _ { r , j } ) / n + \widehat { \mathrm { V a r } } _ { \mathcal { L } \mathcal { M } _ { r } } ( \widehat { w } _ { r } \widehat { \tau } _ { r , j } ) / n _ { \eta }$ is the estimated variance of the $j \mathrm { - t h }$ coordinate of the weighted mean diference.

Since the null distribution of $T _ { r }$ depends on the estimated importance weights and control variates in a complex way, we assess significance via a permutation test rather than a $\chi _ { p } ^ { 2 }$ approximation. Let $\pmb { \tau } _ { i } ^ { \mathcal { L C } } = \widehat { w } ( \mathbf { x } _ { i } ) \widehat { \pmb { \tau } } _ { r } ( \mathbf { x } _ { i , \Gamma _ { r } } , y _ { i } )$ denote the weighted control variate value for $\mathcal { L } \mathcal { C }$ observation $i ,$ and define $\tau _ { k } ^ { \mathcal { L } M }$ analogously for $\mathcal { L } \mathcal { M } _ { r }$ observation k. The permutation procedure is:

1. Pool $\{ \tau _ { i } ^ { \mathcal { L C } } \} _ { i = 1 } ^ { n }$ and $\{ \tau _ { k } ^ { \mathcal { L } \mathcal { M } } \} _ { k = 1 } ^ { n _ { r } }$ into a combined set of $n + n _ { r }$ observations.

2. For $b = 1 , \dots , B _ { \mathrm { p e r m } } ;$ : randomly assign n observations to the $\mathcal { L } \mathcal { C }$ group and $n _ { r }$ to the $\mathcal { L M }$ group; compute the permuted test statistic $T _ { r } ^ { ( b ) }$ as in (20).

3. The p-value is $\begin{array} { r } { \hat { p } _ { r } = \left( 1 + \sum _ { b = 1 } ^ { B _ { \mathrm { p e r m } } } \mathbf { 1 } \{ T _ { r } ^ { ( b ) } \geq T _ { r } \} \right) / \left( 1 + B _ { \mathrm { p e r m } } \right) } \end{array}$

We use $B _ { \mathrm { p e r m } } = 1 { , } 0 0 0$ throughout. This approach avoids parametric distributional assumptions on the test statistic. While exact exchangeability under estimated nuisance parameters is not formally guaranteed, we empirically assess the calibration and power of the resulting screening procedure under both aligned and misaligned sources in Section 4.2.3. Sites with $\hat { p } _ { r } < \alpha$ are excluded from the summation in (21).

## 3.5 Calibration

The augmented estimator (18) applies unit weight to each site’s correction. In practice, the optimal weight should depend on how well $\widehat { \tau } _ { \mathit { r } }$ predicts $\hat { \psi }$ at site r. We therefore introduce a scalar calibration weight $\delta _ { r }$ for each site:

$$
\widehat { \phi } _ { \mathrm { a u g \ } } = \widetilde { \phi } + \sum _ { r = 1 } ^ { R } \delta _ { r } \bigg \{ \frac { 1 } { n _ { r } } \sum _ { ( \mathbf { x } , y ) \in \mathcal { L } \mathcal { M } _ { r } } \widehat { \tau } _ { r } ( \mathbf { x } _ { \Gamma _ { r } } , y ) - \frac { 1 } { n } \sum _ { ( \mathbf { x } , y ) \in \mathcal { L } \mathcal { C } } \widehat { \tau } _ { r } ( \mathbf { x } _ { \Gamma _ { r } } , y ) \bigg \} .\tag{21}
$$

The weight $\delta _ { r }$ is chosen to minimize the empirical variance of $\widehat { \phi } _ { \mathrm { a u g } } .$ . For each feature $j ,$ this reduces to a quadratic program in R variables with closed-form solution $\delta _ { r , j } ^ { * } = [ A _ { j } ^ { - 1 } \mathbf { b } _ { j } ]$ <sub>r</sub> (derived in Appendix C), where

$$
[ A _ { j } ] _ { r s } = \widehat { \mathrm { C o v } } _ { \mathscr { L C } } ( \widehat { \tau } _ { r , j } , \widehat { \tau } _ { s , j } ) + \mathbf { 1 } _ { r = s } \frac { n } { n _ { r } } \widehat { \mathrm { V a r } } _ { \mathscr { L M } _ { r } } ( \widehat { \tau } _ { r , j } ) ,\tag{22}
$$

$$
[ \mathbf { b } _ { j } ] _ { r } = \widehat { \mathrm { C o v } } _ { \mathcal { L C } } ( \widetilde { \psi } _ { j } , \ \widehat { \tau } _ { r , j } ) .\tag{23}
$$

The per-site weight minimizes the total variance $\textstyle \sum _ { j = 1 } ^ { p } \operatorname { V a r } ( \widehat { \phi } _ { \mathrm { a u g } , j } )$ directly:

$$
\delta ^ { * } \ = \ \bigg ( \sum _ { j = 1 } ^ { p } A _ { j } \bigg ) ^ { - 1 } \bigg ( \sum _ { j = 1 } ^ { p } \mathbf { b } _ { j } \bigg ) ,\tag{24}
$$

a single R-dimensional linear system obtained by summing the per-feature quadratic objectives.

The control variate $\widehat { \tau } _ { r }$ is trained via cross-fitting on $\mathcal { L } \mathcal { C }$ . The complete procedure is summarized in Algorithm 1.

## 4 Simulations

We compare FUSHAP against six types of baselines that represent the principal strategies available when labeled data are distributed across sites with blockwise-missing features. All methods target the same estimand (the global Shapley attribution vector $\bar { \phi }$ on the $\mathcal { U } \mathcal { C }$ population) and share the same $\mathcal { U } \mathcal { C }$ background sample for the restricted prediction (2). They difer only in which labeled observations are used to estimate the value function $\mathcal { V } ( \mathbf { z } )$ . The same baselines are used in the real-data applications of Section 5.

(A) Single-site estimator. The WLS Shapley estimator applied to $\mathcal { L } \mathcal { C }$ alone, with $\mathcal { U } \mathcal { C }$ as the background sample for marginalizing absent features. Unbiased but potentially high-variance due to the small $\mathcal { L } \mathcal { C }$ sample.

(B) Single-site with importance weighting. Identical to (A) but with each $\mathcal { L } \mathcal { C }$ observation reweighted by the estimated density ratio $\hat { w } ( x ) = \hat { p } _ { \mathcal { U C } } ( x ) / \hat { p } _ { \mathcal { L C } } ( x )$ to correct for covariate shift. Equivalent to FUSHAP with all calibration weights set to zero; serves as a direct ablation.

(C) Impute-then-estimate. Missing features at each $\mathcal { L } \mathcal { M } _ { r }$ are imputed from $\mathcal { L } \mathcal { C }$ reference values and the completed data are pooled with $\mathcal { L } \mathcal { C }$ . Since prior work and our simulations indicate that switching among standard imputers does not necessarily resolve attribution bias under blockwise missingness (Vo et al., 2025), we report two representative methods: mean imputation and MICE (iterative conditional imputation).

(D) Per-site averaging. Each site independently imputes, estimates $\widehat { \mathcal { V } } _ { r } ( \mathbf { z } )$ from its own labeled observations, solves the WLS, and the resulting Shapley vectors are averaged weighted by sample size.

(E) Complete-case. Only features observed at every site $( \cap _ { r } \Gamma _ { r } )$ are retained; $\eta$ is averaged over all labeled observations using this reduced feature set.

(F) Oracle. All $\mathcal { L } \mathcal { C }$ and $\mathcal { L } \mathcal { M } _ { r }$ observations are pooled with the missing features at each $\mathcal { L } \mathcal { M } _ { r }$ site treated as observed, yielding $n + \textstyle \sum _ { r } n _ { r }$ labeled observations with complete feature vectors on which the WLS Shapley estimator is applied.

All baselines compute $\hat { \phi }$ via the WLS characterization (5) with m sampled coalitions from $\mu _ { \mathrm { S h } }$ For each simulation configuration, the reference $\bar { \phi } ^ { \mathrm { t r u e } }$ is a high-precision Monte Carlo approximation to the population Shapley vector, computed by exact enumeration over all $2 ^ { p }$ coalitions using the combinatorial Shapley formula, with each value function $\mathcal { V } ( S )$ evaluated as the sample average of $\eta ( \mathbf { z } _ { S } , \mathbf { x } _ { i } , y _ { i } )$ over a large independent evaluation set $( n _ { \mathrm { e v a l } } = 5 0 , 0 0 0$ observations from $p _ { \mathcal { U } \mathcal { C } }$ , with $K = 5 0 0$ background samples for the marginal imputation). This evaluation set is generated independently of the UC sample used by the methods. The reference is computed once per configuration and held fixed across all B replications; therefore diferences in MSE across methods reflect the estimation strategy rather than variation in the evaluation target.

## 4.1 Simulation Setup

## 4.1.1 Data-generating process

The target population UC has features drawn as $\mathbf { X } \ \sim \ N ( \mathbf { 0 } , I _ { p } )$ with $p = 1 0$ . The outcome is generated as $Y = f ( \mathbf { X } ) + \varepsilon$ with $\varepsilon \sim N ( 0 , 0 . 2 5 )$ , independently of X. We consider three outcome models of increasing complexity.

Model I (linear).

$$
f _ { \mathrm { I } } ( { \bf x } ) { \bf \phi } = \sum _ { j = 1 } ^ { p } \beta _ { j } x _ { j } , \qquad \beta _ { j } = { \frac { p + 1 - j } { p } } ,\tag{25}
$$

a linear predictor with monotonically decreasing coeficients. Under squared-error loss, the perobservation value contribution $\eta ( \mathbf { z } , \mathbf { x } , y )$ is quadratic in $\left( \mathbf { x } , y \right)$ , and the influence function $\hat { \psi }$ inherits this polynomial structure.

Model II (polynomial interactions).

$$
f _ { \mathrm { I I } } ( { \bf x } ) = f _ { \mathrm { I } } ( { \bf x } ) + 0 . 5 x _ { 1 } x _ { 2 } + 0 . 3 x _ { 3 } x _ { 4 } + 0 . 4 x _ { 5 } ^ { 2 } ,\tag{26}
$$

augmenting Model I with pairwise interactions and a quadratic term. The influence function retains polynomial dependence on the data, though of higher degree than in Model I.

Model III (non-polynomial).

$$
\begin{array} { r } { f _ { \mathrm { I I I } } ( \mathbf { x } ) = \sin ( x _ { 1 } x _ { 2 } ) + 0 . 8 \operatorname* { m a x } ( x _ { 3 } + x _ { 4 } , 0 ) + 0 . 5 x _ { 5 } ^ { 2 } - 0 . 3 x _ { 6 } , } \end{array}\tag{27}
$$

a non-polynomial model whose influence function cannot be fully captured by polynomial control variates.

## 4.1.2 Multi-site data construction

From each model, we construct the multi-site data structure of Section 2.1 by drawing $n { + } \sum _ { r } n _ { r } { + } N$ independent observations and allocating them to $\mathcal { L C } ~ ( n = 3 0 0 )$ 2 $R = 3$ labeled-missing sources $( n _ { r } = 2 , 0 0 0 \ \mathrm { e a c h } )$ , and UC $( N = 5 , 0 0 0 )$ . The missing-feature blocks are non-overlapping: site r observes all features except $\{ 2 ( r { - } 1 ) { + } 1 , 2 r \}$ , so that each site lacks 20% of the feature set and $\mathcal { L } \mathcal { C }$ is the only source with complete coverage.

To reflect the heterogeneity typical of multi-site studies, each data source is subject to both location and scale shifts: features at source s are drawn as $\mathbf { X } ^ { ( s ) } \sim N ( \delta _ { s } \mathbf { 1 } , \sigma _ { s } ^ { 2 } I _ { p } )$ , with $( \delta , \sigma ) = ( 0 , 1 )$ for $\mathcal { U C } , ( 0 . 1 , 1 . 0 5 )$ for $\mathcal { L C } , ( - 0 . 1 5 , 0 . 9 )$ for $\mathcal { L } \mathcal { M } _ { 1 } , ( 0 . 2 , 1 . 1 )$ for $\mathcal { L } \mathcal { M } _ { 2 }$ , and $\left( - 0 . 1 , 0 . 9 5 \right)$ for $\mathcal { L } \mathcal { M } _ { 3 }$ . The density ratio $\widehat { w } ( \mathbf { x } ) = \widehat { p } _ { \mathcal { U C } } ( \mathbf { x } ) / \widehat { p } _ { s } ( \mathbf { x } )$ is estimated via a gradient-boosted classifier on the combined sample with source indicators. In Experiments 1, 2, and 4, the location-scale shift is applied to all $p$ features at each site, including those subsequently declared missing. Consequently, these experiments deliberately introduce moderate violations of Assumption 1 and evaluate FUSHAP beyond its exact alignment regime. In Experiment 3, where screening calibration and power are the quantities of interest, aligned sites are instead constructed to satisfy Assumption 1 exactly (the shift is applied only to the observed features $\mathbf { X } _ { \Gamma _ { r } }$ , while the missing features $\mathbf { X } _ { \Gamma _ { r } ^ { c } }$ are drawn from the target distribution $p u c )$

## 4.2 Experiments

All methods receive the same fixed, pre-trained predictive model and compute its Shapley attribution vector. In the simulation studies, this is the true data-generating function $f ;$ in the real-data applications (Section 5), this is a model trained on held-out data. Across B independent replications, we report the mean squared error $\mathrm { M S E } = \mathrm { B i a s ^ { 2 } + V a r }$ , where $\mathrm { B i a s } ^ { 2 } = | | \bar { \hat { \phi } } - \bar { \phi } | | ^ { 2 }$ and $\begin{array} { r } { \mathrm { V a r } = \sum _ { j } \widehat { \mathrm { V a r } } _ { B } \big ( \hat { \phi } _ { j } \big ) } \end{array}$ , together with Spearman’s rank correlation $\rho _ { s }$ between each estimate and the ground truth to assess agreement in the feature importance ranking. Where appropriate, we report the variance ratio $\mathrm { \Delta V R = V a r ( m e t h o d ) / V a r ( L C \mathrm { - } o n l y ) }$

Table 1: Experiment 1: MSE decomposition and Spearman rank correlation $( B = 1 0 0 , p = 1 0$ $n = 3 0 0 , n _ { r } = 2 , 0 0 0 , R = 3 )$ . Best feasible method in bold.  
Model I (linear)
<table><tr><td>Method</td><td>MSE</td><td> $\mathrm { B i a s ^ { 2 } }$ </td><td> $\mathrm { V a r }$ </td><td> $\rho _ { s }$ </td></tr><tr><td>(F) Oracle</td><td>0.032</td><td>0.027</td><td>0.004</td><td>1.00</td></tr><tr><td>FUSHAP</td><td>0.052</td><td>0.003</td><td>0.049</td><td>0.98</td></tr><tr><td>(A) Single-site</td><td>0.175</td><td>0.087</td><td>0.088</td><td>0.98</td></tr><tr><td>(B) Single+IPW</td><td>0.092</td><td>0.019</td><td>0.074</td><td>0.98</td></tr><tr><td>(C) Impute-mean</td><td>0.118</td><td>0.111</td><td>0.008</td><td>0.99</td></tr><tr><td>(C) Impute-MICE</td><td>0.120</td><td>0.112</td><td>0.008</td><td>0.99</td></tr><tr><td>(D) Per-site avg</td><td>0.119</td><td>0.111</td><td>0.009</td><td>0.99</td></tr><tr><td>(E) Complete-case</td><td>2.492</td><td>2.491</td><td>0.001</td><td>-0.75</td></tr></table>

Model II (sparse interactions)
<table><tr><td>Method</td><td>MSE</td><td> $\mathrm { B i a s ^ { 2 } }$ </td><td>Var</td><td> $\rho _ { s }$ </td></tr><tr><td>(F) Oracle</td><td>0.065</td><td>0.059</td><td>0.006</td><td>0.99</td></tr><tr><td>FUSHAP</td><td>0.065</td><td>0.005</td><td>0.060</td><td>0.98</td></tr><tr><td>(A) Single-site</td><td>0.550</td><td>0.367</td><td>0.183</td><td>0.97</td></tr><tr><td>(B Single+IPW</td><td>0.109</td><td>0.015</td><td>0.093</td><td>0.97</td></tr><tr><td>(C) Impute-mean</td><td>0.129</td><td>0.118</td><td>0.011</td><td>0.98</td></tr><tr><td>(C) Impute-MICE</td><td>0.130</td><td>0.119</td><td>0.011</td><td>0.98</td></tr><tr><td>(D) Per-site avg</td><td>0.130</td><td>0.118</td><td>0.013</td><td>0.98</td></tr><tr><td>(E) Complete-case</td><td>3.336</td><td>3.334</td><td>0.002</td><td>-0.75</td></tr></table>

<table><tr><td colspan="5">Model III (non-polynomial)</td></tr><tr><td>Method</td><td>MSE</td><td> $\mathrm { B i a s ^ { 2 } }$ </td><td> $\mathrm { V a r }$ </td><td> $\rho _ { s }$ </td></tr><tr><td>(F) Oracle</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.96</td></tr><tr><td>FUSHAP</td><td>0.020</td><td>0.009</td><td>0.010</td><td>0.94</td></tr><tr><td>(A) Single-site</td><td>0.061</td><td>0.036</td><td>0.025</td><td>0.95</td></tr><tr><td>(B) Single+IPW</td><td>0.025</td><td>0.015</td><td>0.011</td><td>0.95</td></tr><tr><td>(C) Impute-mean</td><td>0.062</td><td>0.060</td><td>0.002</td><td>0.86</td></tr><tr><td>(C) Impute-MICE</td><td>0.062</td><td>0.060</td><td>0.002</td><td>0.86</td></tr><tr><td>(D) Per-site avg</td><td>0.062</td><td>0.060</td><td>0.002</td><td>0.86</td></tr><tr><td>(E) Complete-case</td><td>0.374</td><td>0.374</td><td>0.000</td><td></td></tr></table>

## 4.2.1 Experiment 1 (imputation bias).

Under the default configuration with all three outcome models, we compare all baselines and FUSHAP over B = 100 replications. Table 1 reports the results. FUSHAP achieves the lowest MSE among all feasible methods across all three models, with improvements of 3.4× (Model I), $8 . 5 \times$ (Model II), and 3.1× (Model III) over the single-site estimator. The gains are largest for Models I and II, where the polynomial control variate is well-specified and captures a substantial fraction of the influence function’s variability. Under Model III (non-polynomial), the control variate approximation is less efective, yet FUSHAP still achieves 3.1× lower MSE than imputation.

The bias–variance decomposition reveals the mechanism. The single-site estimator has low bias but high variance (0.088 in Model I); imputation baselines reduce variance (0.008) but introduce substantial bias (0.111). FUSHAP achieves both low bias (0.003) and moderate variance (0.049), outperforming all alternatives in total MSE. Notably, mean imputation and MICE produce nearly identical results, suggesting that switching between these standard imputation procedures alone does not eliminate the attribution bias (Vo et al., 2025).

## 4.2.2 Experiment 2 (variance reduction).

We examine how FUSHAP’s MSE depends on three design parameters: the LM sample size $n _ { r } \in$ {200, 500, 1,000, 2,000, 5,000} with $R = 3$ fixed (left panels), the number of sites $R \in \{ 1 , 2 , 3 , 5 \}$ with $n _ { r } = 2 , 0 0 0$ fixed (center panels), and the $\mathcal { L } \mathcal { C }$ sample size $n \in \{ 1 0 0 , 2 0 0 , 3 0 0 , 5 0 0 , 1 , 0 0 0 \}$ with $R = 3$ and $n _ { r } = 2 , 0 0 0 \mathrm { f i x e d }$ (right panels). Figure 1 reports results under all three outcome models.

Three patterns are consistent across models. First, FUSHAP’s improvement increases with $n _ { r }$ but exhibits diminishing returns beyond $n _ { r } \approx 1 , 0 0 0$ (left panels). Second, adding sites monotonically reduces MSE: at $R = 5$ , FUSHAP achieves 2.3× lower MSE than the single-site estimator under Model I (center panels). Third, the LC sample size has a critical lower bound: at $n = 1 0 0$ the control variate regression overfits and FUSHAP degrades; for $n \geq 2 0 0$ , FUSHAP consistently improves upon LC-only (right panels). The variance reduction is largest under Model I, where the polynomial control variate is well-specified and smallest under Model III, where the approximation is less efective. FUSHAP requires a suficient number of complete observations to learn the influence-function projection reliably; when the $\mathcal { L } \mathcal { C }$ sample is very small $( n = 1 0 0 )$ , the control variate regression overfits and auxiliary data cannot compensate.

## 4.2.3 Experiment 3 (screening).

Under Model I with $R = 4$ sites, the first three sites satisfy Assumption 1 exactly: the covariate shift is applied only to the observed features $\mathbf { X } _ { \Gamma _ { r } }$ , while the missing features $\mathbf { X } _ { \Gamma _ { r } ^ { c } }$ are drawn from $p _ { \mathcal { U } \mathcal { C } }$ The fourth site is misaligned: its outcome is generated under perturbed coeficients $\tilde { \beta } _ { j } = \beta _ { j } + \Delta$ for $j \in \Gamma _ { 4 } ^ { c }$

Figure 2 reports the rejection rate (left) and MSE (right) as $\Delta$ varies from 0 to 3 over $B = 1 0 0$ replications. At $\Delta = 0$ (no misalignment), the average rejection rate of the three aligned sites is

(c) Varying n (R = 3, n<sub>r</sub> = 2000)  
![](images/f528d26f60b1157d34c741f57c84f11a40ebb3c8b1260eeda13df612f8b6630b.jpg)

![](images/9fdd38265bec224ad7e631ef6e8721d7925d9545e228199c759b58e2d50c6c2b.jpg)  
(a) Model I (linear)

![](images/be049793f33a150c8cf4f723e129bcad5ce4df0a42692c49f33e25605c90a8f9.jpg)  
(c) Varying n (R = 3, n<sub>r</sub> = 2000)

![](images/3aa66aaab9128a4158e83869e8fce1fd18fe22ec9e6a315c4bb294b71a0339db.jpg)

![](images/18fa315f511d8c9b3600d16ddf600986fd13ea4b738117d4c3f96ac6b9d9c1a4.jpg)  
(b) Model II (sparse interactions)

![](images/a9f48e5fe905e8520307aa291e89ce824e3be8d8d5445aea9d2fc29abf5aefb1.jpg)

![](images/c4c0734d86e9ffba7692f7960b3a50e7369f36035f5737456f552e25625e7ca7.jpg)

![](images/6bf07c45e28a71ed6e0b10e2488165f2c6204f9e1dbe5bd9cc594a8cd5ebd417.jpg)  
(c) Model III (non-polynomial)

(c) Varying n (R = 3, n<sub>r</sub> = 2000)  
![](images/b1bb8c2db11e73ebffda8583b1a6e8d38032f4f30b30c875f7200e207bc2bc72.jpg)  
Figure 1: Experiment 2: MSE of FUSHAP (solid) and LC-only (dashed) as a function of $n _ { r }$ (left), R (center), and n (right), B = 100.

![](images/30a80a8b0b6156d46a5fe72a42375ab0640db050971f4db3c0aaf47daaf8b2a0.jpg)

![](images/fc91cb5ca85de7d9da45d4ace1852eb6e68e02ef9f3fbf232f4a219c38f179d9.jpg)  
Figure 2: Experiment 3: screening power and MSE $( B = 1 0 0 , \alpha = 0 . 0 5 )$ . Left: rejection rate of the misaligned site and aligned sites (average). Right: MSE of FUSHAP with and without screening.

4.3%, close to the nominal $\alpha = 0 . 0 5$ (Table 5). The misaligned site is detected with 82% power at $\Delta = 0 . 2 5$ and 100% for $\Delta \ge 0 . 5$ . Without screening, FUSHAP’s MSE degrades from 0.061 at $\Delta = 0$ to $0 . 2 4 9$ at $\Delta = 3$ , exceeding the single-site baseline (0.164). With screening, MSE stabilizes between 0.061 and 0.066 across all values of $\Delta ,$ , a roughly 2.5× improvement over the single-site estimator. Detailed per-site results are reported in Table 5 of Appendix D.

## 4.2.4 Experiment 4 (computational cost).

Table 2 reports wall-clock time as the number of features increases. FUSHAP’s overhead relative to the single-site estimator is modest (3–5×) and arises from the control variate regression and calibration steps. Compared to the impute-then-pool baseline, FUSHAP is 2.7–4.6× faster because it computes Shapley values on the small $\mathcal { L } \mathcal { C }$ sample $( n \ : = \ : 3 0 0 )$ rather than the pooled dataset $\begin{array} { r } { ( n + \sum _ { r } n _ { r } = 6 { , } 3 0 0 ) } \end{array}$ . The speedup grows with $p$ because the pooled Shapley computation scales with both sample size and the number of coalitions.

Table 2: Experiment 4: Wall-clock time in seconds (mean ± std over 3 runs, $n = 3 0 0$ 2 $n _ { r } = 2 { , } 0 0 0$ $R = 2 )$ .
<table><tr><td> $p$ </td><td> $\mathrm { L C - o n l y }$ </td><td>Impute-pool</td><td>FUSHAP</td></tr><tr><td>4</td><td> $0 . 4 7 \pm 0 . 0 1$ </td><td> $6 . 7 9 \pm 0 . 0 8$ </td><td> $2 . 5 2 \pm 0 . 0 2$ </td></tr><tr><td>6</td><td> $0 . 6 7 \pm 0 . 0 2$ </td><td> $9 . 3 9 \pm 0 . 0 0$ </td><td> $2 . 7 9 \pm 0 . 0 2$ </td></tr><tr><td>8</td><td> $0 . 7 7 \pm 0 . 0 1$ </td><td> $1 1 . 6 5 \pm 0 . 0 7$ </td><td> $3 . 0 4 \pm 0 . 0 4$ </td></tr><tr><td>10</td><td> $0 . 9 9 \pm 0 . 0 1$ </td><td> $1 4 . 4 3 \pm 0 . 1 8$ </td><td> $3 . 4 1 \pm 0 . 0 3$ </td></tr><tr><td>12</td><td> $1 . 1 0 \pm 0 . 0 2$ </td><td> $1 6 . 4 8 \pm 0 . 2 0$ </td><td> $3 . 6 2 \pm 0 . 1 2$ </td></tr></table>

## 5 Real-data Applications

We evaluate FUSHAP on two real-world datasets with controlled blockwise missingness imposed on the auxiliary sites, enabling quantitative comparison against a ground-truth attribution vector. Since mean imputation and MICE produce nearly identical Shapley attributions under blockwise missingness, both in our simulations (Table 1) and in prior work (Vo et al., 2025), only mean imputation is reported below.

## 5.1 Beijing Multi-Site Air Quality

We apply FUSHAP to the Beijing Multi-Site Air Quality dataset (Zhang et al., 2017), which records daily averages of six pollutants and five meteorological variables at 12 monitoring stations. The task is to attribute a model for PM2.5 concentration using the remaining $p = 1 0$ features. One urban station (Dongsi, $n = 3 0 0 )$ serves as ${ \mathcal { L } } { \mathcal { C } } ;$ three suburban stations serve as $\mathcal { L } \mathcal { M }$ sites, each missing a diferent pair of features (Table 3); the remaining eight stations form UC. Four models are trained on 15,646 observations from all non-LC stations and held fixed during Shapley estimation. The reference attribution vector is computed from 2,000 held-out training observations with $m = 3 0 0$ sampled coalitions.

FUSHAP achieves the lowest MSE across all four models, with MSEs of 5.8, 7.3, 7.4, and $8 . 8 \ : \ : ( \times 1 0 ^ { - 3 } )$ for the linear, RF, GBM, and MLP models, respectively. Relative to the singlesite estimator, these correspond to improvements of 7.4×, 5.8×, 5.3×, and 4.4×. FUSHAP also improves over the best imputation-based baseline by 2.1× (linear), 1.6× (RF), 1.7× (GBM), and 2.6× (MLP). The bias-variance decomposition (Panel c) shows that FUSHAP achieves both low bias and low variance across all models (e.g., $\mathrm { B i a s ^ { 2 } = 4 . 2 }$ , Var = 1.7 under the linear model), whereas the imputation baselines trade reduced variance for substantial bias $( \mathrm { B i a s ^ { 2 } } = 1 0 . 7 $ , Var $= 1 . 4 )$

## 5.2 NACC Alzheimer’s Disease

We apply FUSHAP to multi-center clinical data from the National Alzheimer’s Coordinating Center (NACC), predicting Mini-Mental State Examination (MMSE) scores from $p = 9$ demographic and clinical features. A random sample of $n = 3 0 0$ patients serves as ${ \mathcal { L } } { \mathcal { C } } ;$ the remaining patients are partitioned into three $\mathcal { L M }$ sites with complementary synthetic missingness and a UC pool (Table 4; preprocessing details in Appendix D.3). Complete-case analysis is infeasible because the three missingness blocks are fully complementary, leaving no features common to all sites. Because $\mathcal { L } \mathcal { C }$ is a random sample from the pooled population, no covariate shift correction is needed and Baseline (B) is omitted. Four regression models are trained on 16,900 non-LC patients and held fixed during Shapley estimation. The reference attribution vector is computed from the 16,900 non-LC training observations with $m = 3 0 0$ sampled coalitions.

FUSHAP achieves the lowest MSE across all four models, with MSEs of 1.03, 3.25, 2.79, and 1.79 for the ridge, RF, GBM, and MLP models, respectively. Relative to the single-site estimator, these correspond to improvements of $6 . 6 \times , 2 . 6 \times , 2 . 6 \times$ , and 4.0×. FUSHAP also improves over the best imputation baseline by 6.0× (ridge), 5.0× (RF), 3.8× (GBM), and $1 2 . 0 \times \ \mathrm { ( M L P ) }$ . For three of the four models, imputation performs worse than the single-site estimator, with MSEs up to 3.0× higher (MLP), demonstrating that imputation bias can outweigh its variance reduction in this setting. The bias-variance decomposition (Panel c) reveals that the single-site estimator has moderate bias and high variance (∼ 6.5), imputation has low variance but dominant bias (6–21), and FUSHAP achieves both low bias (0.2–1.6) and substantially reduced variance (0.9–1.9).

Table 3: Beijing Air Quality: data partition and results $( \times 1 0 ^ { - 3 } , B = 5 0 )$ . Best feasible method in bold.  
(a) Partition
<table><tr><td>Source</td><td>Station</td><td>n</td><td>Missing</td></tr><tr><td> $\mathcal { L } \mathcal { C }$ </td><td>Dongsi</td><td>300</td><td></td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 1 }$ </td><td>Changping</td><td>1,450</td><td>CO, O3</td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 2 }$ </td><td>Huairou</td><td>1,445</td><td>SO2, NO2</td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 3 }$ </td><td>Shunyi</td><td>1,384</td><td>DEWP, PRES</td></tr><tr><td> $\mathcal { U } \mathcal { C }$ </td><td>8 stations</td><td>11,375</td><td></td></tr></table>

(b) MSE $( \times 1 0 ^ { - 3 } )$
<table><tr><td>Method</td><td>Linear</td><td>RF</td><td>GBM</td><td>MLP</td></tr><tr><td>(A) Single-site</td><td>42.7</td><td>42.1</td><td>39.4</td><td>39.0</td></tr><tr><td>(B) Single+IPW</td><td>20.9</td><td>22.4</td><td>22.0</td><td>25.0</td></tr><tr><td>(C) Impute-mean</td><td>12.1</td><td>11.8</td><td>12.7</td><td>23.3</td></tr><tr><td>(D) ) Per-site avg</td><td>11.8</td><td>11.6</td><td>12.5</td><td>22.8</td></tr><tr><td>(E) Complete-case</td><td>62.7</td><td>79.7</td><td>77.1</td><td>136.1</td></tr><tr><td>FUSHAP</td><td>5.8</td><td>7.3</td><td>7.4</td><td>8.8</td></tr></table>

(c) $B i a s ^ { 2 } \mathrm { ~ / ~ } V a r \mathrm { ~ ( \times 1 0 ^ { - 3 } ) ~ }$
<table><tr><td>Method</td><td>Linear</td><td>RF</td><td>GBM</td><td>MLP</td></tr><tr><td>(A) Single-site</td><td>13.4 / 29.2</td><td>13.3 / 28.7</td><td>12.4 / 27.0</td><td>12.2 / 26.8</td></tr><tr><td>(B) Single+IPW</td><td>11.9 / 9.0</td><td>14.9 / 7.6</td><td>14.4 / 7.6</td><td>14.1 / 10.9</td></tr><tr><td>(C) Impute-mean</td><td>10.7 / 1.4</td><td>10.6 / 1.2</td><td>11.6 / 1.2</td><td>20.4 / 2.8</td></tr><tr><td>(D) Per-site avg</td><td>10.6 / 1.2</td><td>10.6 / 1.0</td><td>11.5 / 1.0</td><td>20.7 / 2.1</td></tr><tr><td>(E) Complete-case</td><td>61.6 / 1.1</td><td>79.5 / 0.2</td><td>76.8 / 0.2</td><td>134.7 / 1.5</td></tr><tr><td>FUSHAP</td><td>4.2 / 1.7</td><td>5.5 / 1.8</td><td>5.5 / 1.8</td><td>5.5 / 3.3</td></tr></table>

Table 4: NACC Alzheimer’s: data partition and results $( B = 5 0 )$ . Best feasible method in bold.  
(a) Partition
<table><tr><td>Source</td><td>n</td><td>Missing</td></tr><tr><td>LC (random)</td><td>300</td><td></td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 1 }$ </td><td>3,756</td><td>WEIGHT, HEIGHT, NACCLIVS</td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 2 }$ </td><td>3,756</td><td>ALCOHOL, TOBAC100, CDRLANG</td></tr><tr><td> $\mathcal { L } \mathcal { M } _ { 3 }$ </td><td>3,755</td><td>SEX, RACE, NACCAGE</td></tr><tr><td> $\mathcal { U } \mathcal { C }$ </td><td>5,633</td><td></td></tr></table>

(b) MSE
<table><tr><td>Method</td><td>Ridge</td><td>RF</td><td>GBM</td><td>MLP</td></tr><tr><td>(A) Single-site</td><td>6.83</td><td>8.54</td><td>7.35</td><td>7.07</td></tr><tr><td>(B) Single+IPW</td><td>6.23</td><td>16.25</td><td>10.67</td><td>21.53</td></tr><tr><td>(C) Impute-mean (D) Per-site avg</td><td>6.22</td><td>16.26</td><td>10.67</td><td>21.59</td></tr><tr><td>(E) Complete-case</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>FUSHAP</td><td>1.03</td><td>3.25</td><td>2.79</td><td>1.79</td></tr></table>

(c) $B i a s ^ { 2 }$ / Var
<table><tr><td>Method</td><td>Ridge</td><td>RF</td><td>GBM</td><td>MLP</td></tr><tr><td>(A) Single-site</td><td>0.80 / 6.03</td><td>1.83 / 6.71</td><td>0.97 / 6.38</td><td>0.61 / 6.47</td></tr><tr><td>(B) Single+IPW</td><td></td><td></td><td></td><td></td></tr><tr><td>(C) Impute-mean</td><td>6.12 / 0.11</td><td>16.12 / 0.13</td><td>10.54 / 0.13</td><td>21.09 / 0.45</td></tr><tr><td>(D) Per-site avg</td><td>6.11 / 0.11</td><td>16.13 / 0.13</td><td>10.54 / 0.13</td><td>21.15 / 0.44</td></tr><tr><td>(E) Complete-case FUSHAP</td><td> $\mathrm { N } / \mathrm { A }$  0.18 / 0.85</td><td> $\mathrm { N } / \mathrm { A }$  1.55 / 1.70</td><td> $\mathrm { N } / \mathrm { A }$  0.93 / 1.86</td><td> $\mathrm { N } / \mathrm { A }$  0.36 / 1.43</td></tr></table>

## 6 Discussion

We proposed FUSHAP, a method for estimating global Shapley feature attributions from multi-site data with blockwise missingness. By deriving the influence function of the constrained WLS Shapley estimator and constructing site-specific control variates, FUSHAP reduces variance without imputing missing features, avoiding the systematic coalition-dependent bias that imputation introduces. A permutation-based screening procedure protects against incompatible auxiliary sites, and data-adaptive calibration weights ensure that each site’s contribution is proportional to its informativeness.

Several directions provide opportunities for extending the current framework. First, the current framework assumes that the unlabeled sample UC has complete feature coverage. Extending FUSHAP to settings where the target covariate distribution is only partially observed would broaden its applicability to more general missing-data configurations and would require additional identification assumptions. Second, while our current implementation assumes centralized access to row-level data, the FUSHAP augmentation and calibration are constructed from site-level averages and covariance summaries. This structure provides a natural starting point for privacy-preserving distributed implementations that communicate summary statistics rather than individual-level data.

## Acknowledgments

The NACC database is funded by NIA/NIH Grant U24 AG072122. NACC data are contributed by the NIA-funded ADRCs: P30 AG062429 (PI James Brewer, MD, PhD), P30 AG066468 (PI Oscar Lopez, MD), P30 AG062421 (PI Teresa Gomez-Isla, MD), P30 AG066509 (PI Thomas Grabowski, MD), P30 AG066514 (PI Mary Sano, PhD), P30 AG066530 (PI Helena Chui, MD, Arthur Toga, PhD), P30 AG066507 (PI Marilyn Albert, PhD), P30 AG066444 (PI David Holtzman, MD), P30 AG066518 (PIs Lisa Silbert, MD, Kevin Duf, PhD), P30 AG066512 (PI Thomas Wisniewski, MD), P30 AG066462 (PI Scott Small, MD), P30 AG072979 (PI David Wolk, MD), P30 AG072972 (PIs Charles DeCarli, MD, Rachel Whitmer, PhD), P30 AG072976 (PI Andrew Saykin, PsyD), P30 AG072975 (PI Julie Schneider, MD, MS), P30 AG072978 (PI Ann McKee, MD), P30 AG072977 (PI Robert Vassar, PhD), P30 AG066519 (PI Joshua Grill, PhD), P30 AG062677 (PIs Brad Boeve, MD, Ronald Petersen, MD, PhD), P30 AG079280 (PI Jessica Langbaum, PhD), P30 AG062422 (PI Gi Rabinovici, MD), P30 AG066511 (PI Allan Levey, MD, PhD), P30 AG072946 (PI Linda Van Eldik, PhD), P30 AG062715 (PI Sanjay Asthana, MD, FRCP), P30 AG072973 (PI Russell Swerdlow, MD), P30 AG066506 (PIs Glenn Smith, PhD, ABPP, David Lowenstein, PhD, Ranjan Duara, MD), P30 AG066508 (PIs Stephen Strittmatter, MD, PhD, Christopher Van Dyck, MD), P30 AG066515 (PI Victor Henderson, MD, MS), P30 AG072947 (PI Suzanne Craft, PhD), P30 AG072931 (PI Henry Paulson, MD, PhD), P30 AG066546 (PIs Sudha Seshadri, MD, Gladys Maestre, MD, PhD), P30 AG086401 (PI Erik Roberson, MD, PhD), P30 AG086404 (PI Gary Rosenberg, MD), P30 AG086403 (PI Angela Jeferson, PhD), P30 AG072958 (PIs Heather Whitson, MD, Gwenn Garden, MD, PhD), P30 AG072959 (PI Jagan Pillai, MD, PhD), P30 AG092752 (Ihab Hajjar, MD, MS).

## Data Availability

The Beijing Multi-Site Air Quality dataset is publicly available from the UCI Machine Learning Repository (https://doi.org/10.24432/C5RK5G) (Zhang et al., 2017). NACC data are available upon request through https://naccdata.org/data-request-process/.

## Code Availability

The FUSHAP implementation is available at https://github.com/siqili0325/FUSHAP.

## References

Hugh Chen, Ian C Covert, Scott M Lundberg, and Su-In Lee. Algorithms to estimate shapley value feature attributions. Nature Machine Intelligence, 5(6):590–601, 2023.

Ian Covert and Su-In Lee. Improving KernelSHAP: practical Shapley value estimation via linear regression. In International Conference on Artificial Intelligence and Statistics (AISTATS), 2021.

Ian Covert, Scott Lundberg, and Su-In Lee. Understanding global feature contributions with additive importance measures. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

Ian Covert, Scott Lundberg, and Su-In Lee. Explaining by removing: A unified framework for model explanation. Journal of Machine Learning Research, 22(209):1–90, 2021.

Wangxuan Fan, Siqi Li, Doudou Zhou, Yohei Okada, Chuan Hong, Molei Liu, and Nan Liu. SIM-Shapley: a stable and computationally eficient approach to Shapley value approximation. arXiv preprint arXiv:2505.08198, 2025.

Jingyue Huang, Huiyuan Wang, Yuqing Lei, and Yong Chen. Eficient semiparametric inference for distributed data with blockwise missingness, 2025. URL https://arxiv.org/abs/2508.16902.

Neil Jethani, Mukund Sudarshan, Ian Covert, Su-In Lee, and Rajesh Ranganath. FastSHAP: realtime Shapley value estimation. In International Conference on Learning Representations (ICLR), 2022.

Ying Jin and Dominik Rothenh¨ausler. Modular regression: Improving linear models by incorporating auxiliary data, 2023. URL https://arxiv.org/abs/2211.10032.

Meng Li, Hengyang Sun, Yanjun Huang, and Hong Chen. Shapley value: from cooperative game to explainable artificial intelligence. Autonomous Intelligent Systems, 4(1):2, 2024.

Siqi Li, Chuan Hong, Ziye Tian, Benjamin Sieu-Hon Leong, Koshi Nakagawa, Hideharu Tanaka, Sang Do Shin, Khuong Quoc Dai, Do Ngoc Son, Marcus Eng Hock Ong, Nan Liu, and Molei Liu. Distributionally robust transfer learning with structurally missing covariates, with application to cross-national cardiac arrest prediction, 2026. URL https://arxiv.org/abs/2605.24212.

Yiming Li, Ying Wei, and Molei Liu. Adaptive learning with blockwise missing and semi-supervised data. arXiv preprint arXiv:2405.18722v3, 2025.

Zelei Liu, Yuanyuan Chen, Han Yu, Yang Liu, and Lizhen Cui. GTG-Shapley: eficient and accurate participant contribution evaluation in federated learning. ACM Transactions on Intelligent Systems and Technology, 13(4):1–21, 2022.

Scott M. Lundberg and Su-In Lee. A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Rory Mitchell, Joshua Cooper, Eibe Frank, and Geofrey Holmes. Sampling permutations for shapley value estimation. Journal of Machine Learning Research, 23(43):1–46, 2022.

Christoph Molnar. Interpretable machine learning. Lulu. com, 2020.

Edoardo Mosca, Ferenc Szigeti, Stella Tragianni, Daniel Gallagher, and Georg Groh. Shap-based explanation methods: a review for nlp interpretability. In Proceedings of the 29th international conference on computational linguistics, pages 4593–4603, 2022.

Jefrey N¨af, Erwan Scornet, and Julie Josse. What is a good imputation under mar missingness?, 2026. URL https://arxiv.org/abs/2403.19196.

Ahmed M Salih, Zahra Raisi-Estabragh, Ilaria Boscolo Galazzo, Petia Radeva, Stefen E Petersen, Karim Lekadir, and Gloria Menegaz. A perspective on explainable artificial intelligence methods: Shap and lime. Advanced Intelligent Systems, 7(1):2400304, 2025.

Luke Shannon, Song Liu, and Katarzyna Reluga. Distribution shift in missing data imputation: A risk-based perspective and importance-weighted correction under mar, 2026. URL https: //arxiv.org/abs/2602.06713.

Tuan L. Vo, Thu Nguyen, Luis M. Lopez-Ramos, Hugo L. Hammer, Michael A. Riegler, and Pal Halvorsen. Explainability of machine learning models under missing data, 2025. URL https: //arxiv.org/abs/2407.00411.

Tianhao Wang, Johannes Rausch, Ce Zhang, Ruoxi Jia, and Dawn Song. A principled approach to data valuation for federated learning, 2020. URL https://arxiv.org/abs/2009.06192.

Eric Wu, Kevin Wu, and James Zou. Explaining medical ai performance disparities across sites with confounder shapley value analysis, 2021. URL https://arxiv.org/abs/2111.08168.

Qi Xu, Lorenzo Testa, Jing Lei, and Kathryn Roeder. Blockwise missingness meets ai: A tractable solution for semiparametric inference, 2025. URL https://arxiv.org/abs/2509.24158.

Fei Xue and Annie Qu. Integrating multisource block-wise missing data in model selection. Journal of the American Statistical Association, 116(536):1914–1927, 2021.

Shuyi Zhang, Bin Guo, Anlan Dong, Jing He, Ziping Xu, and Song Xi Chen. Cautionary tales on air-quality improvement in Beijing. Proceedings of the Royal Society A, 473(2205):20170457, 2017.

Shuyuan Zheng, Yang Cao, and Masatoshi Yoshikawa. Secure shapley value for cross-silo federated learning. Proceedings of the VLDB Endowment, 16(7):1657–1670, March 2023. ISSN 2150-8097. doi: 10.14778/3587136.3587141. URL http://dx.doi.org/10.14778/3587136.3587141.

## A Summary of the FUSHAP Algorithm

```latex
Algorithm 1 FUSHAP: Variance-Reduced Shapley Attribution
Require: LC, $\{ \mathcal { L } \mathcal { M } _ { r } \} _ { r = 1 } ^ { R } , \mathcal { U C } ,$ model $f ,$ loss $\ell ,$ significance level $\alpha$
1: Step 1: Preliminary estimator
2: Estimate density ratio $\widehat { w } ( \mathbf { x } ) = \widehat { p } \varkappa c ( \mathbf { x } ) / \widehat { p } c c ( \mathbf { x } )$ via gradient-boosted classifier on $\mathcal { L C \cup U C }$
3: Sample m coalitions $\mathbf { z } _ { 1 } , \ldots , \mathbf { z } _ { m } \sim \mu _ { \mathrm { S h } }$
4: Compute importance-weighted value functions $\begin{array} { r } { \widehat { \mathcal { V } } ( \mathbf { z } _ { j } ) = n ^ { - 1 } \sum _ { i \in \mathcal { L C } } \widehat { w } ( \mathbf { x } _ { i } ) \eta ( \mathbf { z } _ { j } , \mathbf { x } _ { i } , y _ { i } ) } \end{array}$
5: Solve WLS via $( 6 ) ~  ~ \widetilde { \phi }$
6: Step 2: Influence function
7: Compute $\Sigma ^ { - 1 }$ from the Shapley kernel
8: Compute weighted residuals $\epsilon _ { w } ( \mathbf { z } _ { j } , \mathbf { x } _ { i } , y _ { i } ) = \widehat { w } ( \mathbf { x } _ { i } ) \eta ( \mathbf { z } _ { j } , \mathbf { x } _ { i } , y _ { i } ) - \widehat { \mathcal { V } } ( \mathbf { z } _ { j } )$
9: Compute $\begin{array} { r } { \mathbf { g } _ { w } ( \mathbf { x } _ { i } , y _ { i } ) = m ^ { - 1 } \sum _ { j = 1 } ^ { m } \mathbf { z } _ { j } \epsilon _ { w } ( \mathbf { z } _ { j } , \mathbf { x } _ { i } , y _ { i } ) } \end{array}$
10: Compute $\underset { \sim } { \lambda } { } _ { w } ( \mathbf { x } _ { i } , y _ { i } )$ via (15)
11: Compute $\widetilde { \psi } _ { w } ( \mathbf { x } _ { i } , y _ { i } ) = \Sigma ^ { - 1 } [ \mathbf { g } _ { w } ( \mathbf { x } _ { i } , y _ { i } ) + \lambda _ { w } ( \mathbf { x } _ { i } , y _ { i } ) \mathbf { 1 } ]$
12: Step 3: Control variates
13: for $r = 1 , \ldots , R$ do
14: Estimate per-site density ratio $\widehat { w } _ { \underline { { r } } } ( \mathbf { x } _ { \Gamma _ { r } } )$ via classifier on $\mathcal { L } \mathcal { M } _ { r } \cup \mathcal { U } \mathcal { C }$ using features $\Gamma _ { r }$
15: Cross-fit $\widehat { \tau } _ { r } \colon$ ridge regression of $\psi$ on $\mathrm { p o l y } ( \mathbf { X } _ { \Gamma _ { r } } , Y )$ with 5-fold CV on $\mathcal { L } \mathcal { C }$
16: Refit on all $\mathcal { L } \mathcal { C } ;$ evaluate $\widehat { \tau } _ { r }$ on $\mathcal { L } \mathcal { M } _ { r }$
17: end for
18: Step 4: Screening
19: for $r = 1 , \ldots , R$ do
20: Compute studentized $T _ { r }$ via (20); assess significance by permutation test $( B _ { \mathrm { p e r m } } = 1 , 0 0 0 )$
21: if $\hat { p } _ { r } < \alpha$ then
22: Exclude site $r$
23: end if
24: end for
25: Step 5: Calibration
26: for $j = 1 , \dotsc , p$ do
27: Compute $A _ { j }$ and $\mathbf { b } _ { j }$ via (22)–(23)
28: Add adaptive ridge: $A _ { j }  A _ { j } + \lambda _ { \mathrm { r e g } } I _ { R }$
29: end for
30: Solve $\begin{array} { r } { \pmb { \delta } ^ { * } = \big ( \sum _ { j = 1 } ^ { p } A _ { j } \big ) ^ { - 1 } \big ( \sum _ { j = 1 } ^ { p } \mathbf { b } _ { j } \big ) } \end{array}$ via (24)
31: Output: $\begin{array} { r } { \widehat { \phi } _ { \mathrm { a u g } } = \widetilde { \phi } + \sum _ { r \in \mathrm { a l i g n e d } } \delta _ { r } ^ { * } \big \{ \bar { \pmb { \tau } } _ { r } ^ { \mathcal { L } \mathcal { M } } - \bar { \pmb { \tau } } _ { r } ^ { \mathcal { L } \mathcal { C } } \big \} } \end{array}$
```

## B Derivation of the Influence Function

The population Shapley vector (6) is a function of $( \mathbf { b } , c )$

$$
\bar { \phi } = h ( { \bf b } , c ) = \Sigma ^ { - 1 } \bigg [ { \bf b } + \frac { c - { \bf 1 } ^ { \top } \Sigma ^ { - 1 } { \bf b } } { { \bf 1 } ^ { \top } \Sigma ^ { - 1 } { \bf 1 } } { \bf 1 } \bigg ] ,\tag{28}
$$

where $\boldsymbol { \Sigma } = \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ \mathbf { z } \mathbf { z } ^ { \top } ]$ depends only on p and the Shapley kernel. For notational convenience, define

$$
\mathbf { s } = \Sigma ^ { - 1 } \mathbf { 1 } , \qquad \kappa = \mathbf { 1 } ^ { \top } \mathbf { s } = \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 } \mathbf { 1 } .\tag{29}
$$

Expanding (28):

$$
\begin{array} { l } { { \displaystyle h ( { \mathbf b } , c ) = \Sigma ^ { - 1 } { \mathbf b } + \frac { c } { \kappa } { \mathbf s } - \frac { { \mathbf s } ^ { \top } { \mathbf b } } { \kappa } } \mathbf s } \\ { { \displaystyle \qquad = \underbrace { \left( \Sigma ^ { - 1 } - \frac { { \mathbf s } { \mathbf s } ^ { \top } } { \kappa } \right) } _ { M _ { 1 } } { \mathbf b } + \underbrace { \frac { \mathbf s } { \kappa } } _ { M _ { 2 } } c . } } \end{array}\tag{30}
$$

Since $M _ { 1 }$ and $M _ { 2 }$ are constant matrices (depending only on $\Sigma )$ , the map h is linear in $( \mathbf { b } , c )$ . The first-order expansion is therefore exact with no higher-order remainder. The partial derivatives follow from (30):

$$
\frac { \partial h } { \partial { \bf b } } = M _ { 1 } = \Sigma ^ { - 1 } - \frac { { \bf s \bf s } ^ { \top } } { \kappa } ,\tag{31}
$$

$$
\frac { \partial h } { \partial c } = M _ { 2 } = \frac { \bf s } { \kappa } = \frac { \Sigma ^ { - 1 } { \bf 1 } } { { \bf 1 } ^ { \top } \Sigma ^ { - 1 } { \bf 1 } } .\tag{32}
$$

Recall that b is defined as a weighted average over coalitions b $= \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ { \mathbf z } ( \mathcal { V } ( { \mathbf z } ) - \mathcal { V } ( \mathbf { 0 } ) ) ]$ When each $\mathcal { V } ( \mathbf { z } )$ is perturbed by $\epsilon ( \mathbf { z } , \mathbf { x } , y )$ (including the null coalition $\mathcal { V } ( \mathbf { 0 } )$ which is perturbed by $\epsilon ( \mathbf { 0 } , \mathbf { x } , y ) )$ , the perturbed b becomes

$$
\begin{array} { r l } & { \mathbf { b } + \delta \mathbf { b } = \mathbb { E } _ { \mu _ { \mathrm { S h } } } \Big [ \mathbf { z } \Big ( \big ( \mathcal { V } ( \mathbf { z } ) + \epsilon ( \mathbf { z } , \mathbf { x } , y ) \big ) - \big ( \mathcal { V } ( \mathbf { 0 } ) + \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) \big ) \Big ) \Big ] } \\ & { \qquad = \underbrace { \mathbb { E } _ { \mu _ { \mathrm { S h } } } \big [ \mathbf { z } \big ( \mathcal { V } ( \mathbf { z } ) - \mathcal { V } ( \mathbf { 0 } ) \big ) \big ] } _ { \mathrm { b } } + \mathbb { E } _ { \mu _ { \mathrm { S h } } } \big [ \mathbf { z } \big ( \epsilon ( \mathbf { z } , \mathbf { x } , y ) - \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) \big ) \big ] . } \end{array}
$$

Hence

$$
\delta \mathbf { b } = \mathbb { E } _ { \mu _ { \mathrm { S h } } } \bigl [ \mathbf { z } \bigl ( \epsilon ( \mathbf { z } , \mathbf { x } , y ) - \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) \bigr ) \bigr ] .\tag{33}
$$

Since $\epsilon ( \mathbf { 0 } , \mathbf { x } , y ) = \eta ( \mathbf { 0 } , \mathbf { x } , y ) - \mathcal { V } ( \mathbf { 0 } )$ does not depend on the coalition z, it is a scalar that factors out of the expectation:

$$
\delta \mathbf { b } = \underbrace { \mathbb { E } _ { \mu _ { \mathrm { S h } } } \bigl [ \mathbf { z } \epsilon ( \mathbf { z } , \mathbf { x } , y ) \bigr ] } _ { \mathbf { g } ( \mathbf { x } , y ) } - \underbrace { \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ \mathbf { z } ] } _ { \bar { \mathbf { z } } } \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) .\tag{34}
$$

Similarly, from $c = \mathcal { V } ( { \bf 1 } ) - \mathcal { V } ( { \bf 0 } )$ , the perturbation of c is

$$
\begin{array} { r l } & { \delta c = \left( \mathcal { V } ( \mathbf { 1 } ) + \epsilon ( \mathbf { 1 } , \mathbf { x } , y ) \right) - \left( \mathcal { V } ( \mathbf { 0 } ) + \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) \right) - \underbrace { \left( \mathcal { V } ( \mathbf { 1 } ) - \mathcal { V } ( \mathbf { 0 } ) \right) } _ { c } } \\ & { \quad \quad = \epsilon ( \mathbf { 1 } , \mathbf { x } , y ) - \epsilon ( \mathbf { 0 } , \mathbf { x } , y ) } \\ & { \quad = \left[ \eta ( \mathbf { 1 } , \mathbf { x } , y ) - \mathcal { V } ( \mathbf { 1 } ) \right] - \left[ \eta ( \mathbf { 0 } , \mathbf { x } , y ) - \mathcal { V } ( \mathbf { 0 } ) \right] } \\ & { \quad = \left[ \eta ( \mathbf { 1 } , \mathbf { x } , y ) - \eta ( \mathbf { 0 } , \mathbf { x } , y ) \right] - \underbrace { \left[ \mathcal { V } ( \mathbf { 1 } ) - \mathcal { V } ( \mathbf { 0 } ) \right] } _ { = c } . } \end{array}\tag{35}
$$

Define the influence function $\psi ( \mathbf { x } , y )$ as the per-observation contribution to the estimation error $\tilde { \phi } - \bar { \phi }$ , so that

$$
\widetilde { \phi } - \bar { \phi } = \frac { 1 } { n } \sum _ { i \in \mathcal { L C } } \bar { \psi } ( \mathbf { x } _ { i } , y _ { i } ) .
$$

Since $h ( { \bf b } , c ) = M _ { 1 } { \bf b } + M _ { 2 } c$ is linear, each observation’s contribution is

$$
{ \bar { \psi } } ( \mathbf { x } , y ) = M _ { 1 } \delta \mathbf { b } ( \mathbf { x } , y ) + M _ { 2 } \delta c ( \mathbf { x } , y ) = { \frac { \partial h } { \partial \mathbf { b } } } \delta \mathbf { b } + { \frac { \partial h } { \partial c } } \delta c .\tag{36}
$$

Substituting (31)–(35) and writing $\epsilon _ { 0 } = \epsilon ( \mathbf { 0 } , \mathbf { x } , y )$

$$
\begin{array} { l } { { \displaystyle { \bar { \psi } = \left( { \boldsymbol { \Sigma } } ^ { - 1 } - \frac { { \bf { s s } } ^ { \top } } { \kappa } \right) \left( { \bf { g } } - { \bar { \bf { z } } } \epsilon _ { 0 } \right) + \frac { { \bf { s } } } { \kappa } \delta c } \ ~ } } \\ { { \displaystyle ~ = { \boldsymbol { \Sigma } } ^ { - 1 } \left( { \bf { g } } - { \bar { \bf { z } } } \epsilon _ { 0 } \right) - \frac { { \bf { s s } } ^ { \top } \left( { \bf { g } } - { \bar { \bf { z } } } \epsilon _ { 0 } \right) } { \kappa } + \frac { { \bf { s } } \delta c } { \kappa } } } \\ { { \displaystyle ~ = { \boldsymbol { \Sigma } } ^ { - 1 } \left[ { \bf { g } } - { \bar { \bf { z } } } \epsilon _ { 0 } + \frac { \delta c - { \bf { s } } ^ { \top } \left( { \bf { g } } - { \bar { \bf { z } } } \epsilon _ { 0 } \right) } { \kappa } { \bf 1 } \right] } , } \end{array}\tag{37}
$$

where the last equality uses $\mathbf { s } = \Sigma ^ { - 1 } \mathbf { 1 }$ to factor $\Sigma ^ { - 1 }$ from the second and third terms.

The Shapley kernel $\mu _ { \mathrm { S h } } ( \mathbf { z } )$ depends on z only through $| \mathbf { z } | = \mathbf { 1 } ^ { \top } \mathbf { z }$ . Since each feature appears symmetrically across all coalitions of a given size,

$$
\begin{array} { r } { \bar { \bf z } = \mathbb { E } _ { \mu _ { \mathrm { S h } } } [ { \bf z } ] = \frac { 1 } { 2 } { \bf 1 } . } \end{array}\tag{38}
$$

Substituting into (37), the expression inside the brackets becomes

$$
{ \bf g } \ - \ { \textstyle \frac { 1 } { 2 } } { \bf 1 } \epsilon _ { 0 } \ + \ \frac { \delta c - { \bf s } ^ { \top } ( { \bf g } - { \textstyle \frac { 1 } { 2 } } { \bf 1 } \epsilon _ { 0 } ) } { \kappa } { \bf 1 } .
$$

Since $- \frac { 1 } { 2 } { \bf 1 } \epsilon _ { 0 }$ is proportional to 1, it can be merged with the last term. To do $\operatorname { s o } .$ , first expand $\mathbf { s } ^ { \top } ( \mathbf { g } - \frac { 1 } { 2 } \mathbf { 1 } \epsilon _ { 0 } )$ :

$$
\begin{array} { r } { \mathbf { s } ^ { \top } \big ( \mathbf { g } - \frac { 1 } { 2 } \mathbf { 1 } \boldsymbol { \epsilon } _ { 0 } \big ) = \mathbf { s } ^ { \top } \mathbf { g } - \frac { 1 } { 2 } \underbrace { \big ( \mathbf { s } ^ { \top } \mathbf { 1 } \big ) } _ { = \kappa } \boldsymbol { \epsilon } _ { 0 } = \mathbf { s } ^ { \top } \mathbf { g } - \frac { \kappa } { 2 } \boldsymbol { \epsilon } _ { 0 } . } \end{array}\tag{39}
$$

Now collect all scalar multiples of 1 from both terms and denote their sum by $\lambda ^ { * }$ :

$$
\begin{array} { l } { { \displaystyle \lambda ^ { * } = - \frac { 1 } { 2 } \epsilon _ { 0 } ~ + ~ \frac { \delta c - \mathbf { s } ^ { \top } \mathbf { g } + \frac { \kappa } { 2 } \epsilon _ { 0 } } { \kappa } } } \\ { ~ } \\ { { \displaystyle ~ = - \frac { 1 } { 2 } \epsilon _ { 0 } ~ + ~ \frac { \delta c - \mathbf { s } ^ { \top } \mathbf { g } } { \kappa } ~ + ~ \frac { \frac { \kappa } { 2 } \epsilon _ { 0 } } { \kappa } } } \\ { { \displaystyle ~ = \frac { \delta c - \mathbf { s } ^ { \top } \mathbf { g } } { \kappa } . } } \end{array}\tag{40}
$$

Since all $\epsilon _ { \mathrm { 0 } }$ terms have canceled, the non-1 part of (37) reduces to g alone. Expanding δc and $\mathbf { s } ^ { \top } \mathbf { g }$ in (40) using (35) and $\mathbf { s } ^ { \top } = \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 }$ :

$$
\begin{array} { l } { { \displaystyle \lambda ^ { * } = \frac { \delta c - \mathbf { s } ^ { \top } \mathbf { g } } { \kappa } } \ ~ } \\ { { \displaystyle ~ = \frac { \left[ \eta ( \mathbf { 1 } , \mathbf { x } , y ) - \eta ( \mathbf { 0 } , \mathbf { x } , y ) \right] - c - \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 } \mathbf { g } ( \mathbf { x } , y ) } { \mathbf { 1 } ^ { \top } \Sigma ^ { - 1 } \mathbf { 1 } } . } } \end{array}\tag{41}
$$

Substituting back into (37):

$$
\begin{array} { r } { \boxed { \bar { \boldsymbol { \psi } } ( \mathbf { x } , y ) = \Sigma ^ { - 1 } \big [ \mathbf { g } ( \mathbf { x } , y ) + \lambda ( \mathbf { x } , y ) \mathbf { 1 } \big ] , } } \end{array}\tag{42}
$$

with $\lambda ( \mathbf { x } , y )$ given by (41), recovering (14)–(15) in the main text.

## C Derivation of the Calibration Weights

We derive the optimal calibration weights $\delta _ { \cdot , j } ^ { * } = A _ { j } ^ { - 1 } \mathbf { b } _ { j }$ stated in (22)–(23).

For feature $j ,$ , the estimation error of the calibrated estimator (21) is approximately

$$
\widehat { \phi } _ { \mathrm { a u g } , j } - \bar { \phi } _ { j } \approx \frac { 1 } { n } \sum _ { i \in \mathcal { L } } \bigg [ \widetilde { \psi } _ { j } ( \mathbf { x } _ { i } , y _ { i } ) - \sum _ { r = 1 } ^ { R } \delta _ { r } \widehat { \tau } _ { r , j } ( \mathbf { x } _ { i } , y _ { i } ) \bigg ] \ + \ \sum _ { r = 1 } ^ { R } \frac { \delta _ { r } } { n _ { r } } \sum _ { k \in \mathcal { L } \backslash u _ { r } } \widehat { \tau } _ { r , j } ( \mathbf { x } _ { k } , y _ { k } ) .\tag{43}
$$

Since $\mathcal { L } \mathcal { C }$ and $\mathcal { L } \mathcal { M } _ { r }$ are independent, the variance decomposes as

$$
\mathrm { V a r } ( \widehat { \phi } _ { \mathrm { a u g } , j } ) = \frac { 1 } { n } \mathrm { V a r } \bigg ( \widetilde { \psi } _ { j } - \sum _ { r } \delta _ { r } \widehat { \tau } _ { r , j } \bigg ) + \sum _ { r = 1 } ^ { R } \frac { \delta _ { r } ^ { 2 } } { n _ { r } } \mathrm { V a r } _ { \mathcal { L } \mathcal { M } _ { r } } ( \widehat { \tau } _ { r , j } ) .\tag{44}
$$

Expanding the first term:

$$
\begin{array} { r } { \mathrm { V a r } \displaystyle \left( \widetilde { \psi } _ { j } - \sum _ { r } \delta _ { r } \widehat { \tau } _ { r , j } \right) = \mathrm { V a r } ( \widetilde { \psi } _ { j } ) - 2 \sum _ { r } \delta _ { r } \mathrm { C o v } ( \widetilde { \psi } _ { j } , \widehat { \tau } _ { r , j } ) } \\ { + \sum _ { r } \sum _ { s } \delta _ { r } \delta _ { s } \mathrm { C o v } ( \widehat { \tau } _ { r , j } , \widehat { \tau } _ { s , j } ) . } \end{array}\tag{45}
$$

Substituting (45) into (44) and combining the $\delta _ { r } ^ { 2 }$ terms:

$$
\begin{array} { l } { { \displaystyle V _ { j } ( \pmb \delta ) = \frac { 1 } { n } \operatorname { V a r } ( \widetilde \psi _ { j } ) - \frac { 2 } { n } \sum _ { r } \delta _ { r } \operatorname { C o v } ( \widetilde \psi _ { j } , \widehat \tau _ { r , j } ) } } \\ { { \displaystyle \quad \quad + \frac { 1 } { n } \sum _ { r } \sum _ { s } \delta _ { r } \delta _ { s } \operatorname { C o v } ( \widehat \tau _ { r , j } , \widehat \tau _ { s , j } ) + \sum _ { r } \frac { \delta _ { r } ^ { 2 } } { n _ { r } } \operatorname { V a r } _ { \mathcal { L } \mathcal { M } _ { r } } ( \widehat \tau _ { r , j } ) } . } \end{array}\tag{46}
$$

This is quadratic in δ. Taking the derivative with respect to $\delta _ { r }$ and setting it to zero:

$$
\frac { \partial V _ { j } } { \partial \delta _ { r } } = - \frac { 2 } { n } \operatorname { C o v } ( \widetilde { \psi } _ { j } , \widehat { \tau } _ { r , j } ) + \frac { 2 } { n } \sum _ { s } \delta _ { s } \operatorname { C o v } ( \widehat { \tau } _ { r , j } , \widehat { \tau } _ { s , j } ) + \frac { 2 \delta _ { r } } { n _ { r } } \operatorname { V a r } _ { \angle { M _ { r } } } ( \widehat { \tau } _ { r , j } ) = 0 .\tag{47}
$$

Dividing by $2 / n$ and rearranging:

$$
\sum _ { s = 1 } ^ { R } \delta _ { s } \biggl [ \mathrm { C o v } ( \widehat { \tau } _ { r , j } , \widehat { \tau } _ { s , j } ) + \mathbf { 1 } _ { r = s } \frac { n } { n _ { r } } \mathrm { V a r } _ { \mathcal { L M } _ { r } } ( \widehat { \tau } _ { r , j } ) \biggr ] = \mathrm { C o v } ( \widetilde { \psi } _ { j } , \widehat { \tau } _ { r , j } ) .\tag{48}
$$

This is the linear system $A _ { j } \pmb { \delta } _ { \cdot , j } = \mathbf { b } _ { j }$ with

$$
[ A _ { j } ] _ { r s } = \operatorname { C o v } _ { \mathit { \mathcal { L } } \mathit { \mathcal { C } } } ( \widehat { \tau } _ { r , j } , \widehat { \tau } _ { s , j } ) + \mathbf { 1 } _ { r = s } \frac { n } { n _ { r } } \operatorname { V a r } _ { \mathit { \mathcal { L } } \mathit { \mathcal { M } } _ { r } } ( \widehat { \tau } _ { r , j } ) ,
$$

$$
[ \mathbf { b } _ { j } ] _ { r } = \mathrm { C o v } _ { \mathcal { L C } } ( \widetilde { \psi } _ { j } , \widehat { \tau } _ { r , j } ) ,
$$

recovering (22)–(23) in the main text. When a single per-site scalar $\delta _ { r }$ is used across all features, the total variance $\textstyle \sum _ { j = 1 } ^ { p } V _ { j } ( \delta )$ is minimized by summing (48) over $j ,$ yielding $\begin{array} { r } { \big ( \sum _ { j } A _ { j } \big ) \pmb { \delta } = \sum _ { j } \mathbf { b } _ { j } } \end{array}$ □

## D Additional Experimental Results

## D.1 Simulation Experiment 3: Detailed Screening Results

Table 5 reports per-site rejection rates and MSE across all misalignment strengths tested in Experiment 3. Panel (a) confirms that the three aligned sites maintain rejection rates between 2% and 7% across all values of $\Delta ,$ , consistent with the nominal $\alpha = 0 . 0 5$ . Panel (b) shows that FUSHAP with screening stabilizes MSE between 0.061 and 0.066 across all misalignment strengths, while FUSHAP without screening degrades beyond the single-site baseline for $\Delta \geq 1$

## D.2 Beijing Air Quality

The Beijing Multi-Site Air Quality dataset (Zhang et al., 2017) contains hourly measurements from 12 monitoring stations over March 2013 to February 2017. We aggregate to daily averages and remove days with any missing values within each station.

The target variable is daily mean PM2.5 concentration. The $p = 1 0$ predictor features are: PM10, SO2, NO2, CO, O3 (pollutants) and TEMP, PRES, DEWP, RAIN, WSPM (meteorological

Table 5: Experiment 3: per-site rejection rates and MSE across misalignment strengths $( \alpha = 0 . 0 5$ B = 100). Aligned sites satisfy Assumption 1 exactly.  
(a) Rejection rates (%)
<table><tr><td>∆</td><td>Site 1 (aligned)</td><td>Site 2 (aligned)</td><td>Site 3 (aligned)</td><td>Site 4 (misaligned)</td></tr><tr><td>0</td><td>4</td><td>7</td><td>2</td><td>5</td></tr><tr><td>0.25</td><td>4</td><td>7</td><td>2</td><td>82</td></tr><tr><td>0.5</td><td>4</td><td>7</td><td>2</td><td>100</td></tr><tr><td>1.0</td><td>4</td><td>7</td><td>2</td><td>100</td></tr><tr><td>1.5</td><td>4</td><td>7</td><td>2</td><td>100</td></tr><tr><td>2.0</td><td>4</td><td>7</td><td>2</td><td>100</td></tr><tr><td>3.0</td><td>4</td><td>7</td><td>2</td><td>100</td></tr></table>

(b) MSE comparison
<table><tr><td>∆ LC-only</td><td>FUSHAP (no screen)</td><td>FUSHAP (screen)</td></tr><tr><td>0 0.164</td><td>0.061</td><td>0.061</td></tr><tr><td>0.25 0.164</td><td>0.064</td><td>0.066</td></tr><tr><td>0.5 0.164</td><td>0.075</td><td>0.066</td></tr><tr><td>1.0 0.164</td><td>0.124</td><td>0.066</td></tr><tr><td>1.5 0.164</td><td>0.187</td><td>0.066</td></tr><tr><td>2.0 0.164</td><td>0.232</td><td>0.066</td></tr><tr><td>3.0 0.164</td><td>0.249</td><td>0.066</td></tr></table>

variables).

To construct a controlled multi-site scenario with complementary blockwise missingness, we designate one urban station (Dongsi) as $\mathcal { L } \mathcal { C }$ with all features observed $( n = 3 0 0$ days subsampled per replication). Three suburban stations serve as $\mathcal { L } \mathcal { M }$ sites, each with a diferent pair of features artificially removed: CO and O3 at Changping, SO2 and NO2 at Huairou, and DEWP and PRES at Shunyi. The remaining eight stations form UC (N = 11,375 days) with complete feature coverage but no outcome variable used during Shapley estimation. This design ensures complementary missingness patterns across sites while preserving real inter-station distributional heterogeneity.

All features and the outcome are standardized using UC means and standard deviations. Four regression models (linear, random forest, GBM, MLP) are trained on 15,646 observations from all $\mathrm { n o n } { - } \mathcal { L } \mathcal { C }$ stations and held fixed during Shapley estimation. The reference attribution vector is computed from 2,000 held-out training observations.

## D.3 NACC Alzheimer’s Data

The National Alzheimer’s Coordinating Center (NACC) Uniform Data Set (UDS) aggregates clinical assessments from over 40 Alzheimer’s Disease Research Centers across the United States. We use the investigator dataset which contains demographic, clinical, and cognitive variables collected under heterogeneous protocols across centers.

The outcome variable is the Mini-Mental State Examination (MMSE) score, a continuous measure of cognitive function ranging from 0 (severe impairment) to 30 (no impairment). The $p = 9$ predictor features are: sex, race, age at visit (NACCAGE), weight, height, living situation (NAC-CLIVS), alcohol use history, tobacco use history (TOBAC100), and CDR language domain score (CDRLANG). Observations with missing values in any feature or the outcome are excluded, yielding 17,200 complete cases from 34 centers.

To construct a controlled multi-site scenario with complementary blockwise missingness, we randomly sample $n = 3 0 0$ patients from the pooled complete cases to serve as ${ \mathcal { L C } } .$ . The remaining patients are partitioned into three $\mathcal { L } \mathcal { M }$ sites and one UC pool. Each $\mathcal { L } \mathcal { M }$ site has a diferent triplet of features artificially removed (Table 4), ensuring complementary missingness patterns. This design satisfies Assumption 1 by construction, as all partitions are drawn from the same population.

Continuous features (NACCAGE, WEIGHT, HEIGHT, CDRLANG) are standardized using the UC means and standard deviations. Four regression models (ridge, random forest, GBM, MLP) are trained on all non-LC patients $( n _ { \mathrm { t r a i n } } = 1 6 , 9 0 0 )$ and held fixed during Shapley estimation. The reference attribution vector is computed from all non-LC patients $( n _ { \mathrm { t r a i n } } = 1 6 , 9 0 0 )$ with $m = 3 0 0$ sampled coalitions, providing an evaluation target independent of the $\mathcal { L } \mathcal { C }$ sample used for attribution estimation.