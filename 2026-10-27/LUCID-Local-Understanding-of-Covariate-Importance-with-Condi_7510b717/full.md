# LUCID: Local Understanding of Covariate Importance with Conditional Density Estimation

Yixiao Wang<sup>1</sup>, Srikar Katta<sup>1</sup>, Jon Donnelly<sup>1,2</sup>, Emanuele Borgonovo<sup>3</sup>, and Cynthia Rudin<sup>1</sup>

<sup>1</sup>Duke University, Durham, USA

<sup>2</sup>Johns Hopkins University, Baltimore, USA

<sup>3</sup>Bocconi University, Milan, Italy

yixiao.wang@duke.edu, srikar.katta@duke.edu, jdonnelly@jhu.edu, emanuele.borgonovo@unibocconi.it, cynthia@cs.duke.edu

## Abstract

Computing local-level variable importance (VI) is critical for understanding machine learning models. Local permutation-based VI metrics ofer a simple yet powerful approach, measuring how much a model’s output changes when perturbing its inputs. However, existing perturbation strategies often ignore feature dependence or account for it only through restricted approximations. Perturbing a feature without fully respecting this dependence can generate unrealistic counterfactuals, thereby evaluating the model in regions unsupported by the observed data distribution. We address this gap by introducing a new model-agnostic local VI metric and estimator, Local Understanding of Covariate Importance with conditional Density estimation (LUCID). LUCID perturbs features of interest using plausible values drawn from the full conditional distribution given the remaining observed features. We establish theoretical connections between LUCID and existing VI measures, and provide an estimator with convergence guarantees and uncertainty quantification. Finally, we demonstrate LUCID’s stable performance and practical utility across simulated and real-world examples.

## 1 Introduction

Variable importance (VI) is a fundamental tool for interpreting machine learning models, enabling the characterization of features’ usage and their efects on the model’s predictions. VI is particularly useful for probing the otherwise opaque behavior of black-box machine learning systems (Guidotti et al., 2018; Casalicchio et al., 2019; Fisher et al., 2019). Most established VI methods summarize each feature with a single populationlevel score or ranking (Breiman, 2001; Gregorutti et al., 2015). However, a feature may strongly influence the model in one region of the covariate space while being irrelevant in another, so averaging importance across the population can obscure when, where, and for which observations a feature matters. For example, in the Bike Sharing dataset (Fanaee-T and Gama, 2014), the importance of temperature and humidity changes substantially over the course of a day, while their global VI scores, shown as dashed lines, collapse this variation into single values and obscure the underlying heterogeneity (Figure 1(a)). More fundamentally, distinct local importance patterns can share the same population average, meaning that global importance alone cannot identify how importance is distributed across observations. This motivates local VI and its accurate estimation, which makes observation-level feature usage accessible and characterizes heterogeneous feature usage across individual predictions.

A central challenge for local VI is determining how to perturb the feature. Existing local perturbation-based approaches typically sample counterfactuals from the marginal distribution of the feature of interest (Casalicchio et al., 2019; Bladen et al., 2026), which can break feature dependence and generate implausible counterfactuals. For example, in the Bike Sharing data, plausible temperature values vary across seasons, whereas marginal permutation draws replacements from the same pool regardless of the remaining covariates (Fig. 1(b)). Although global conditional importance methods provide dependence-aware alternatives, their perturbation schemes often rely on restricted approximations whose validity requires additional assumptions (Fisher et al., 2019; Chamma et al., 2023; Borgonovo et al., 2026; Strobl et al., 2008; Debeer and Strobl, 2020; Molnar et al., 2024; Blesch et al., 2025). Such schemes can recover population-average importance while producing systematically biased local scores, because errors across observations may cancel after averaging (Section 5.1 DGP 3). Accurate local estimation therefore requires capturing the observationspecific conditional structure. This motivates flexible estimation of the full conditional distribution, which existing perturbation schemes may struggle to capture, particularly under heteroskedasticity, multimodality, high dimensionality, or complex interactions.

![](images/6cc0a0365420eeb82d02af2cfdd2591e2ef428045674f671b735b09c5b020d56.jpg)

![](images/06ea0a19dcc076a9d15e16d17103065750effebfa8ca06f7533b92f720f8508e.jpg)  
Figure 1: Two motivations for LUCID on the Bike Sharing dataset (Fanaee-T and Gama, 2014). (a) LUCID values for temperature and humidity LUCID (solid lines) vary substantially across hours, while their corresponding global VI values (dashed lines) obscure this heterogeneity. (b) For temperature, each split violin compares the estimated conditional replacement distribution from DR-BART (left; Orlandi et al., 2021) with the marginal replacement distribution (right). Conditional densities adapt across seasons, whereas the marginal distribution does not.

In this work, we introduce Local Understanding of Covariate Importance with conditional Density estimation (LUCID), a modelagnostic framework for measuring feature importance at the observation level. For a fixed predictive model, LUCID replaces a feature of interest with plausible values drawn from its conditional density given the remaining observed features, and measures the resulting change in model behavior. Our main contributions are:

• We establish an interpretable decomposition of LUCID that captures local leave-one-covariateout (LOCO) importance without refitting the predictive model and additionally quantifies conditional prediction variability omitted by local LOCO.

• We develop a flexible estimation framework for LUCID that accommodates modern conditional density estimators (CDEs), establish convergence guarantees, and provide a Bayesian implementation with posterior uncertainty quantification.

• We extend LUCID to explain feature groups, enabling observation-level assessment of their joint importance conditional on the observed covariates outside the group.

• Through extensive synthetic and real-data experiments, we demonstrate LUCID’s ability to capture heterogeneous local importance under complex covariate dependence, together with the high accuracy, robustness, and computational scalability of our implementation.

## 2 Related Work

Permutation-based variable importance measures feature relevance through changes in model behavior under feature perturbations, beginning with the Random Forest permutation importance of Breiman (2001). Subsequent work has developed a range of marginal and conditional variable importance methods at both global and local levels. We provide a detailed discussion of these methods in Section A. In contrast, LUCID defines observation-level importance with respect to the full conditional feature distribution. Accurate estimation of this distribution yields replacements that preserve the dependence between the feature of interest and the remaining covariates

Other notions of feature importance and local explanation target diferent estimands. We discuss connections to LOCO (Lei et al., 2018), SHAP (Lundberg and Lee, 2017; Aas et al., 2021), LIME (Ribeiro et al., 2016), grouped variable importance (Gregorutti et al., 2015; Au et al., 2022; Chamma et al., 2024; Little et al., 2026), and other explanation paradigms in Section A.

## 3 Our Method: LUCID

## 3.1 Notation and Background

Let Z denote the feature of interest and $\boldsymbol { X } =$ $( X _ { 1 } , \ldots , X _ { k } )$ denote the remaining covariates. We write $\pmb { W } = ( Z , \pmb { X } )$ for the complete feature vector and let $g ( W ) = g ( Z , X )$ denote a given prediction model to be explained, with ℓ denoting a prediction loss. We assume the analyst observes a dataset of $n$ independent and identically distributed observations $\{ W _ { i } \} _ { i = 1 } ^ { n }$ from some distribution $P .$ . For an observation ${ \pmb w } _ { i } = ( z _ { i } , { \pmb x } _ { i } )$ . We write $P _ { Z }$ and $P ( \cdot \mid x )$ to denote the marginal and conditional (on all other features) distributions of feature $z ,$ respectively. For distributions with densities $p$ and $q ,$ we denote the total variation distance by $\textstyle \mathrm { T V } ( p , q ) : = { \frac { 1 } { 2 } } \int | p ( z ) - q ( z ) | d z$ , and, for a measurable function $h$ , the $L ^ { 2 }$ norm by $\| h \| _ { L ^ { 2 } ( P ) } : = \left( \mathbb { E } _ { P } [ h ^ { 2 } ] \right) ^ { 1 / 2 }$

Permutation feature importance (PFI) is the population-average discrepancy between the original prediction and the prediction obtained by replacing $Z$ with an independent draw from its marginal distribution:

$$
\operatorname { P F I } : = \mathbb { E } _ { Z , X } \left\{ \mathbb { E } _ { Z ^ { \prime } \sim P _ { Z } } \left[ \ell ( g ( Z ^ { \prime } , \pmb { X } ) , g ( Z , \pmb { X } ) ) \right] \right\}\tag{1}
$$

Conditional permutation feature importance (CPFI) instead draws the replacement from the conditional distribution of $Z$ given the

remaining features:

$$
\mathrm { C P F I } : = \mathbb { E } _ { Z , X } \left\{ \mathbb { E } _ { Z ^ { \prime } \sim P ( \cdot \vert X ) } \big [ \ell \big ( g ( Z ^ { \prime } , X ) , g ( Z , X ) \big ) \big ] \right\} .\tag{2}
$$

Leave-one-covariate-out importance (LOCO) measures the expected prediction loss when the full model is replaced by a reduced model using only the remaining features (Lei et al., 2018): $\operatorname { L O C O } : = \mathbb { E } _ { Z , X } [ \ell ( g _ { - Z } ( X ) , g ( Z , X ) ) ]$ , where $g _ { - Z }$ denotes a reduced model, typically obtained by refitting without feature $Z .$

## 3.2 LUCID Definition

We now introduce the instance-level counterpart of conditional permutation feature importance. For an observation $w ,$ , Local Understanding of Covariate Importance with conditional Density estimation (LUCID) of feature Z is defined as

$$
\operatorname { L U C I D } ( w ) : = \mathbb { E } _ { Z ^ { \prime } \sim P ( \cdot | x ) } \big [ \ell \big ( g ( Z ^ { \prime } , x ) , g ( z , x ) \big ) \big ] .\tag{3}
$$

LUCID measures the expected prediction discrepancy when z is replaced by a draw from $P ( \cdot \mid x )$ with the remaining covariates fixed at x. By the law of total expectation, integrating LUCID in Equation (3) over the population distribution of W recovers the CPFI in Equation (2).

## 3.3 Understanding LUCID

Under squared loss, LUCID admits a simple decomposition into two terms, each asking a separate question: (i) how varied are the predictions from $g$ across diferent counterfactual values of $Z ;$ (ii) how extreme was the prediction for the observed value of $Z$ compared to the predictions for possible counterfactual values of $Z ?$ We first establish this decomposition for squared-error loss, illustrating the theoretical connection to local LOCO. We then extend the decomposition to Bregman divergences generated by diferentiable, strictly convex functions.

Below, expectations and variances are over $Z ^ { \prime } \sim P ( \cdot \mid x )$ , and we assume the black box’s predictions admit a finite variance over diferent perturbations.

![](images/583dd8dac6ea81446f2d1edeff7404490f403c19d91f5dd8116438d5a56f6955.jpg)  
Figure 2: Three examples of conditional distributions of perturbed outputs $g ( Z ^ { \prime } , \pmb { x } )$ The purple dashed line marks the mean $\mathbb { E } [ g ( Z ^ { \prime } , x ) ]$ , the yellow arrow shows the square root of the displacement, while the purple arrow illustrates the conditional standard deviation, the square root of the dispersion term.

Theorem 1 (Decomposition of LUCID). Under the prediction reference and squared loss, LUCID(w admits the decomposition

$$
\underbrace { \mathrm { V a r } \big [ g ( Z ^ { \prime } , \pmb { x } ) \big ] } _ { d i s p e r s i o n } + \underbrace { \big ( g ( z , \pmb { x } ) - { \mathbb { E } } \big [ g ( Z ^ { \prime } , \pmb { x } ) \big ] \big ) ^ { 2 } } _ { d i s p l a c e m e n t } ,\tag{4}
$$

We refer to the two components in (4) as the dispersion and displacement terms, respectively. The proof for Theorem 1 is in Section C.2. Globally, dispersion and displacement contribute equally on average over the data distribution (Corollary C.2). At the local level, however, they capture distinct mechanisms through which a feature can be important for a particular observation. The dispersion term measures how much the model output varies across conditionally plausible replacements of Z.

In contrast, the displacement term measures how far the original output lies from the average prediction across perturbations. So, if we observe a value of Z that leads to an extreme prediction far from the central tendency, displacement will be high. Both quantities measure how important the feature is for making the prediction, but in diferent ways. Dispersion asks if predictions vary wildly across possible values of Z | x while displacement asks if the prediction for the observed data point is itself an extreme/outlier compared to the average prediction. As illustrated in Figure 2, LUCID can therefore be large because the perturbed outputs vary substantially (high dispersion), because the observed output lies far from their mean (high displacement), or both. Conversely, since both terms are nonnegative, by

Corollary C.3,

$$
\operatorname { L U C I D } ( z , \pmb { x } ) = 0 \quad \Longleftrightarrow \quad g ( Z ^ { \prime } , \pmb { x } ) = g ( z , \pmb { x } ) \operatorname { a . s . }\tag{5}
$$

This can occur (i) because the model does not use feature $Z$ in predicting $\pmb { x } \mathrm { \hat { s } }$ outcome, (ii) because Z is determined by the remaining features, or (iii) because its conditionally plausible values all produce the same model output.

Relationship with LOCO. The decomposition also clarifies the relationship between LUCID and LOCO, both of which assess a feature’s contribution beyond the remaining features. Under squared loss with Bayes-optimal full and reduced predictors, CPFI equals twice LOCO (Corollary C.2) (Reyero-Lobo et al., 2025). However, this proportionality does not generally hold at the local level.

For a comparison LUCID, if the reduced model is the optimal reduced predictor of $g ( Z , X )$ under squared loss, then $g _ { - Z } ( \pmb { x } ) = \mathbb { E } [ g ( Z , \pmb { x } ) ]$ . The corresponding local LOCO score is $\mathrm { L O C O } ( \pmb { w } ) : =$ $( g ( z , \pmb { x } ) - g _ { - Z } ( \pmb { x } ) ) ^ { 2 }$

Proposition 1 (Relationship between LUCID and local LOCO). Under squared loss and the prediction reference,

$$
\operatorname { L U C I D } ( \pmb { w } ) = \underbrace { \operatorname { V a r } \big [ g ( Z ^ { \prime } , \pmb { x } ) \big ] } _ { d i s p e r s i o n } + \underbrace { \operatorname { L O C O } ( \pmb { w } ) } _ { d i s p l a c e m e n t } .
$$

The result follows directly from Theorem 1 and the proof for Proposition 1 is in Section C.2. Under this comparison, local LOCO measures the discrepancy between the original output and the optimal reduced prediction, while LUCID additionally captures variation among conditionally plausible outputs. This additional dispersion term can change local feature rankings even when the corresponding global importance measures difer only by a factor of 2 (see Section E.8).

Beyond Squared Loss. The dispersion–displacement interpretation of LUCID extends to Bregman divergences (Bregman, 1967; Adlam et al., 2022). For a diferentiable, strictly convex function ϕ, define $D _ { \phi } ( a , b ) = \phi ( a ) - \phi ( b ) - \langle \nabla \phi ( b ) , a - b \rangle$

Proposition 2 (Bregman Decomposition of LU-CID). Under the prediction reference and Bregman loss $\ell ( a , b ) = D _ { \phi } ( a , b )$ , assuming the relevant expectations are finite, LUCID(w) admits the decomposition

$$
\begin{array} { r } { \mathrm { L U C I D } ( w ) = \underbrace { \mathbb { E } \left[ D _ { \phi } \left( g ( Z ^ { \prime } , x ) , \mathbb { E } \left[ g ( Z ^ { \prime } , x ) \right] \right) \right] } _ { d i s p e r s i o n } } \\ { + \underbrace { D _ { \phi } \big ( \mathbb { E } \left[ g ( Z ^ { \prime } , x ) \right] , g ( z , x ) \big ) } _ { d i s p l a c e m e n t } . \quad ( } \end{array}\tag{6}
$$

For asymmetric Bregman divergences, the divergence direction matters. When averaging, the squared-loss factor-of-two identity between global CPFI and LOCO need not hold. We provide detailed proofs for Proposition 2 and more illustrative examples in Section C.3.

## 3.4 LUCID for Feature Groups

Individual features’ LUCID may fail to capture joint efects when features carry overlapping information. For example, suppose a prediction model depends on both temperature and season, while these features are strongly predictable from one another. Conditioning on temperature can substantially restrict plausible replacements of season, and vice versa, yielding small individual LUCID scores. Such scores reflect limited unique information, however, and do not imply that the two features are jointly unimportant. We therefore extend LUCID to feature groups (Gregorutti et al., 2015; Au et al., 2022; Chamma et al., 2024). For a group of features $Z ,$ let X denote the remaining conditioning covariates. We define the group LUCID as

$$
\operatorname { L U C I D } ( z , x ) : = \mathbb { E } _ { Z ^ { \prime } \sim P ( \cdot | x ) } \left[ \ell \big ( g ( Z ^ { \prime } , x ) , g ( z , x ) \big ) \right] .\tag{7}
$$

The definition extends directly to a group of features by replacing the scalar $Z$ with a vector $z$ Let G denote a feature group with $d = | G |$ . In general, group LUCID need not equal the sum of the corresponding individual LUCID scores, because individual replacements condition on the other group members, whereas group replacement allows all members to vary jointly. Examining individual and group LUCID together provides a fuller picture, as illustrated in Section 5.2.

## 4 Estimating LUCID

Estimating LUCID requires sampling from an unknown conditional feature distribution. We first introduce a plug-in Monte Carlo estimator applicable to any conditional density estimator (CDE), and then introduce an implementation with DR-BART.

## 4.1 Conditional-Sampling Estimation

Let $\mathcal { D } _ { n }$ denote the sample used to estimate the conditional distribution of $Z$ given $\boldsymbol { X }$ , and let ${ \pmb w } _ { i } = ( z _ { i } , { \pmb x } _ { i } ) , i = 1 , . . . , N$ , denote the observations to be explained. We continue to treat g as fixed and use prediction-reference LUCID.

Given an estimate ${ \widehat { P } } _ { Z | X } ( \cdot | { \pmb x } )$ , draw $Z _ { m } ^ { \prime } \stackrel { \mathrm { i . i . d . } } { \sim }$ $\widehat { P } _ { Z | X } ( \cdot \mid x ) , m = 1 , \ldots , M$ , and define

$$
\widehat { \mathrm { L U C I D } } ( w ) : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \ell \bigl ( g \bigl ( Z _ { m } ^ { \prime } , \pmb { x } \bigr ) , g \bigl ( z , \pmb { x } \bigr ) \bigr )\tag{8}
$$

Estimation therefore requires only conditionally plausible draws and evaluations of the fixed model, making the procedure agnostic to the choice of CDE.

## 4.2 Stability and Convergence

To study sensitivity to conditional-distribution estimation, write $\operatorname { L U C I D } [ p ] ( \pmb { w } )$ for predictionreference LUCID induced by conditional density $p ( \cdot \mid x )$ , and let $p ^ { * }$ denote the truth. Let d be a distance between conditional distributions satisfying, for some constant A,

$$
\| \mathrm { T V } \{ p ( \cdot \mid X ) , q ( \cdot \mid X ) \} \| _ { L ^ { 2 } ( P _ { X } ) } \leq A d ( p , q ) .\tag{9}
$$

Theorem 2 (Stability of LUCID). Suppose that the loss $\ell ( g ( z ^ { \prime } , \pmb { x } ) , g ( z , \pmb { x } ) )$ is uniformly bounded by some $C _ { \ell } < \infty ,$ for $P _ { Z , X }$ -almost every $( z , \pmb { x } )$ and all replacement values $z ^ { \prime }$ in the supports of the conditional distributions being compared. If the estimated conditional density satisfies $d ( \widehat { p } , p ^ { * } ) =$ $O _ { P } ( \epsilon _ { n } )$ , then the induced LUCID function converges at the same rate in $L ^ { 2 } ( P _ { Z , X } )$

$$
\| \mathrm { L U C I D } [ \hat { p } ] - \mathrm { L U C I D } [ p ^ { * } ] \| _ { L ^ { 2 } ( P _ { Z , X } ) } = O _ { P } ( \epsilon _ { n } ) .
$$

The theorem shows that conditional distribution estimation error passes to LUCID without a loss in convergence rate under the stated bound. For squared loss, bounded model outputs are suficient. Proof is given in Section C.4.

## 4.3 Implementing LUCID with DR-BART

Our main implementation estimates each conditional distribution $P ( \cdot \mid X )$ using Density Regression Bayesian Additive Regression Trees (DR-BART) (Orlandi et al., 2021). Posterior draws of this conditional distribution induce posterior draws of LUCID, allowing point estimation and uncertainty quantification.

Posterior point estimates and credible intervals. For each retained posterior draw $b =$ $1 , \ldots , B$ , generate M replacements and compute

$$
\widehat { \mathrm { L U C I D } } ^ { ( b ) } ( \pmb { w } ) : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \ell \Big ( g ( Z _ { m } ^ { \prime ( b ) } , \pmb { x } ) , g ( z , \pmb { x } ) \Big ) .\tag{10}
$$

We use the average of $\widehat { \{ \mathrm { L U C I D } ( \pmb { w } ) ^ { ( b ) } \} _ { b = 1 } ^ { B } }$ as the posterior point estimate and their empirical quantiles to construct posterior credible intervals.

Convergence under DR-BART. Under the bounded-loss assumption in Theorem 2 and the regularity and prior conditions of Orlandi et al. (2021, Theorem 2), the DR-BART density posterior contracts in integrated Hellinger distance, which satisfies the required distance-control condition in Theorem 2. The induced LUCID posterior therefore contracts at the same rate in $L ^ { 2 } ( P _ { Z , X } )$ Under these conditions, Sections C.5–C.7 further establish $O _ { P } ( \epsilon _ { n } ^ { 2 } )$ posterior second-moment control for LUCID (Corollary C.7) and an $O _ { P } ( \epsilon _ { n } )$ convergence rate for the posterior-mean conditional density and its induced LUCID function (Proposition C.8).

Section D.1 also provides implementation details, including direct conditional sampling from the DR-BART latent-variable representation, binary and categorical features, and joint conditional sampling for feature groups.

Table 1: Methods considered in our experiments.
<table><tr><td>Category</td><td>Method</td></tr><tr><td>Attribution</td><td>SHAP (Lundberg and Lee, 2017) LIME (Ribeiro et al., 2016)</td></tr><tr><td>Removal</td><td>Cond. SHAP (Aas et al., 2021; Jullum et al., 2025) local version of LOCO (Lei et al., 2018)</td></tr><tr><td>Permutation</td><td>ICI (Casalicchio et al., 2019)</td></tr><tr><td></td><td></td></tr><tr><td></td><td>CMR (Fisher et al., 2019)</td></tr><tr><td></td><td>GCMR (Borgonovo et al., 2026)</td></tr><tr><td></td><td>CLIQUE (Bladen et al., 2026)</td></tr><tr><td></td><td>local version of cs-PFI (Molnar et al., 2024)</td></tr><tr><td></td><td>local version of cARFi (Blesch et al., 2025)</td></tr></table>

Note: We refer to ICI as “Permutation” in this paper. ARF denotes the conditional replacement sampler used for our local adaptation of cARFi.

Computational scalability with DR-BART. Although fitting DR-BART is computationally intensive, fitted conditional samplers can be reused across observations. Section D.2 provides theoretical time and space complexity analyses, while Section F.6 reports empirical runtime and memory scaling on higher-dimensional OpenXAI (Agarwal et al., 2022) benchmark datasets, including datasets with more than $1 0 ^ { 5 }$ observations and more than 50 features.

## 5 Experiments

We evaluate LUCID in controlled synthetic settings with known models and in a real-data bikedemand case study with learned prediction models. For each dataset, we use five-fold crossfitting: we train a conditional density estimator on four folds and estimate LUCID on the held-out fold (Chernozhukov et al., 2018). For simplicity, throughout the experiments we use $I _ { j }$ to denote the LUCID of feature j. We list baseline methods in Table 1 and provide complete experimental details and additional results in Sections E and F.

## 5.1 Synthetic Experiments

Because the synthetic experiments admit analytical LUCID, we first compare estimated and analytical importance using mean squared error (MSE) for feature of interest $X _ { 1 }$ in Table 2. Across these controlled settings, our DR-BART implementation is the most consistently accurate conditional estimator overall, attaining the best average MSE rank while remaining robust across discontinuity (DGP 1), correlation (DGP 2), heteroskedasticity (DGP 3), interaction structure (DGP 4), and extreme tail (DGP 5). Competing methods can perform well when their structural assumptions align with the data-generating process, but their errors can increase sharply when those assumptions are violated. We further repeat the synthetic experiments over 50 random seeds in Section E.3, confirming that these qualitative conclusions are not driven by a particular realization.

Table 2: MSE between estimated and analytical LUCID, averaged over all evaluated instances of $X _ { 1 }$ . The lowest MSE (from unrounded MSE values) in each DGP is highlighted in purple bold. Mean rank averages within-DGP rankings; lower is better.
<table><tr><td></td><td>DR-BART</td><td>ARF</td><td>cs-PFI</td><td>CMR</td><td>GCMR</td><td>CLIQUE</td><td>Permutation</td></tr><tr><td>DGP 1</td><td>0.022</td><td>0.144</td><td>0.008</td><td>0.590</td><td>0.543</td><td>0.001</td><td>0.008</td></tr><tr><td>DGP 2</td><td>0.004</td><td>0.005</td><td>0.015</td><td>0.011</td><td>0.011</td><td>4.460</td><td>4.431</td></tr><tr><td>DGP 3</td><td>0.449</td><td>0.454</td><td>1.173</td><td>2.685</td><td>2.485</td><td>2.060</td><td>2.065</td></tr><tr><td>DGP 4</td><td>0.401</td><td>1.298</td><td>83.329</td><td>1.809</td><td>4.315</td><td>83.673</td><td>83.135</td></tr><tr><td>DGP 5</td><td>0.066</td><td>0.185</td><td>0.036</td><td>0.849</td><td>0.851</td><td>0.251</td><td>0.269</td></tr><tr><td>Mean rank</td><td>1.800</td><td>2.800</td><td>3.400</td><td>5.200</td><td>5.400</td><td>4.600</td><td>4.800</td></tr></table>

We organize subsequent contents around four questions motivating our framework: “why local importance?” (DGP 1); “why conditional perturbations?” (DGP 2); “why density estimation?” (DGP 3); and “why estimate the conditional density directly?” (irrelevant-feature robustness test). We also consider DGP 4 to show why subgroup-based methods can struggle to recover interaction structure, and DGP 5 to further explore the relationship between LUCID and local LOCO; we defer these two experiments, along with analytical LUCID calculations and additional experimental details and results, to Section E.

DGP 1: Discontinuity. LUCID handles feature heterogeneity, unlike global VI measures. This case study uses three independent standard Gaussian features, $X _ { 1 } , X _ { 2 } , X _ { 3 } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 )$ and a simple tree predictor $g ( \pmb { x } ) = x _ { 1 } \mathbf { 1 } _ { x _ { 3 } > 0 } +$ $x _ { 2 } \mathbf { 1 } _ { x _ { 3 } \leq 0 }$ , so that $X _ { 1 }$ is active only when $X _ { 3 } > 0$ whereas $X _ { 2 }$ is active only when $X _ { 3 } \leq 0$ (a setup similar to the one considered in Bladen et al., 2026). As shown in the DR-BART panel in Figure 3, because $X _ { 1 }$ and $X _ { 2 }$ play symmetric roles across the discontinuity, their global importance (dashed line) is identical, masking the fact that only one of them matters at any given observation. In contrast, LUCID captures this heterogeneity exactly, with the yellow and purple points representing observations on the two sides of the discontinuity.

![](images/c37c4f6e54a3cdd5549cd8f624accb75ef594e4657eb14b77058b01a6b5c7b06.jpg)  
Figure 3: Discontinuity. Local importance for $X _ { 1 }$ across methods. Red boxes mark representative nonzero importance in the inactive region. Solid and dashed curves denote analytical LUCID and CPFI, respectively.

LUCID correctly identifies locally irrelevant features. When a feature is locally inactive, perturbing it leaves the prediction unchanged, so LUCID is zero by Equation (5). In contrast, LIME, SHAP, conditional SHAP, and local LOCO assign nonzero scores to $X _ { 1 }$ at every inactive instance $( x _ { 3 } \le 0 )$ ; a representative point is highlighted by the red box in Figure 3. This results in importance leakage for diferent reasons: LIME fits a locally linear model that does not recognize boundaries, while SHAP and conditional SHAP average over subsets that omit $X _ { 3 }$ , mixing contributions across the discontinuity. Moreover, finite-sample errors in model refitting can lead to local LOCO producing nonzero scores. Section E.4 contains detailed comparisons and experiment results.

![](images/e72ca656386a2df4888fb279dae8635aec599085a954d91b76e0bc891ac5a174.jpg)  
Figure 4: Heteroskedastic. Resampling distributions of $X _ { 1 }$ at five $X _ { 2 }$ quantile points across samplers.

DGP 2: Correlated. Conditioning in LU-CID accounts for correlated features. When features are dependent, marginal permutation breaks the observed dependence structure and may evaluate the fitted model on unrealistic feature combinations (Strobl et al., 2008; Hooker et al., 2021). We isolate this efect using two jointly Gaussian features, $( X _ { 1 } , X _ { 2 } ) ^ { \top } \sim { \mathcal { N } } _ { 2 } ( 0 , \Sigma _ { \rho } )$ with unit variances and correlation $\rho ,$ and the predictor $g ( \pmb { x } ) = x _ { 1 }$ . In contrast to marginal permutation, conditional perturbations preserve features’ dependency and therefore measure the additional information in $X _ { 1 }$ beyond $X _ { 2 }$ . Accordingly, in this DGP, the analytical-average LUCID equals $2 ( 1 - \rho ^ { 2 } )$ (where $\rho$ denotes the correlation between features), whereas global marginal permutation importance remains equal to 2. Thus, as $\rho  1$ (i.e., as features become increasingly correlated), conditional importance reflects the increasing redundancy, whereas permutation-based methods fail to respect the feature dependence structure, leading to poor performance in the MSE comparison in Table $2 ;$ see Section E.5 for experiment results.

DGP 3: Heteroskedastic. Residual-based conditional permutation importance fail to capture heteroskedasticity. We isolate heteroskedastic dependence by letting $X _ { 2 } \sim { \mathcal { N } } ( 0 , 1 )$ and $X _ { 1 } ~ \mid ~ X _ { 2 } ~ = ~ x _ { 2 } ~ \sim ~ \mathcal { N } ( 0 , x _ { 2 } ^ { 2 } )$ , with predictor $g ( \pmb { x } ) ~ = ~ x _ { 1 }$ Thus, the conditional mean $\mathbb { E } [ X _ { 1 } \mid X _ { 2 } ] = 0$ is constant, while the conditional variance Var $( X _ { 1 } \mid X _ { 2 } = x _ { 2 } ) = x _ { 2 } ^ { 2 }$ varies with $x _ { 2 }$ Even with the correct conditional mean, methods based on globally permuted residuals (e.g., Fisher et al., 2019; Borgonovo et al., 2026) draw perturbations with the pooled residual variance $\mathbb { E } [ X _ { 2 } ^ { 2 } ] = 1$ , failing to preserve this heteroskedasticity. As a result, while the analytical LUCID is $x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 }$ , these methods asymptotically yield $x _ { 1 } ^ { 2 } + 1$ Importantly, this local misspecification is invisible after global averaging because $\mathbb { E } [ X _ { 2 } ^ { 2 } ] = 1 $ : the incorrect local scores recover the correct global average importance despite systematically misrepresenting where the feature matters. Moreover, Section E.6 shows that any common replacement distribution incurs a population local MSE of at least 2 in this DGP, further demonstrating why accurate conditional density estimation is essential for local variable importance. In Figure 4, we show the resampling distributions induced by each method. Only CDE-based approaches capture the changing conditional spread, with DR-BART most closely matching the analytical CDE.

![](images/87a8f001601034bfaa423104bd0a2be34e14c0c1c57196e74b6d57addc2322b6.jpg)  
Figure 5: LUCID estimation MSE as independent irrelevant features are added across five DGPs.

Conditional density estimation is more robust to high-dimensional covariate spaces than joint density estimation. Similar to our DR-BART estimator of LUCID, cARFi also generates perturbations by sampling from a learned density (Blesch et al., 2025). cARFi obtains conditional replacements from the joint distribution using Adversarial Random Forest (ARF; Watson et al., 2023), whereas we instead fit the conditional density directly. Because joint distribution estimation generally becomes more dificult as the ambient dimension grows (Izbicki and Lee, 2016), we consider a naive stress test: we augment each of the five DGPs with 0, 5, 10, or 20 independent standard Gaussian noise features and compare ARF with DR-BART. These additions increase the ambient dimension without changing the true conditional distribution of $X _ { 1 }$ or its analytical LUCID. As shown in Figure 5, the LUCID estimation MSE increases sharply for ARF as irrelevant features are added, whereas DR-BART’s estimated conditional density is far more stable. In Section E.12, we further consider more challenging high-dimensional settings with additional relevant features, where the degradation of joint-density-based estimation is even more pronounced.

Table 3: Empirical coverage (C) and width (W) of LUCID posterior credible intervals for active $X _ { 1 }$ observations across five DGPs.
<table><tr><td></td><td>DGP 1</td><td>DGP 2</td><td>DGP 3</td><td>DGP 4</td><td>DGP 5</td></tr><tr><td>90% C</td><td>98.43%</td><td>91.70%</td><td>91.34%</td><td>93.20%</td><td>97.12%</td></tr><tr><td>90% W</td><td>0.766</td><td>0.155</td><td>0.944</td><td>0.969</td><td>0.956</td></tr><tr><td>95% C</td><td>99.37%</td><td>95.76%</td><td>92.94%</td><td>95.64%</td><td>98.52%</td></tr><tr><td>95% W</td><td>0.910</td><td>0.184</td><td>1.123</td><td>1.152</td><td>1.138</td></tr></table>

Posterior credible intervals. A major advantage of LUCID is that its estimation admits posterior credible intervals. Table 3 evaluates the uncertainty quantification provided by the DR-BART posterior for LUCID over observations where $X _ { 1 }$ is active in the model. Across the five DGPs, empirical coverage is generally close to the nominal 90% and 95% levels, with DGP 1 somewhat conservative, consistent with its particularly simple structure, and DGP 3 slightly below nominal at 95%, mainly because of the sparse, highly dispersed observations in the tails. As expected, interval widths increase with the credible level. We exclude inactive observations, whose LUCIDs are identically zero and recovered exactly.

## 5.2 Real-Dataset Experiments

We next examine individual and group LUCID for a learned prediction model using the 2011 subset of the UCI Bike Sharing dataset (n = 8,645 hourly records), which combines Capital Bikeshare rental counts with calendar and weather measurements (Fanaee-T and Gama, 2014). Following Blesch et al. (2025), we retain hour, workingday status, normalized temperature, normalized humidity, and season. For additional experimental details and results, see Section F.

LUCID reveals local heterogeneity across <sup>5</sup> real covariate contexts. Figure 1(a) shows that temperature LUCID varies substantially over the day, so its global importance masks when temperature matters most. As shown in Figure 1(b), its conditional replacement distribution also shifts across seasons, whereas marginal permutation fails to capture this heterogeneity by drawing replacements from the same marginal distribution throughout. Figure 6 shows a complementary pattern for hour: on working days, importance peaks sharply during the morning and evening commuting periods, whereas this structure is largely absent on non-working days. Unlike temperature, however, hour is only weakly predictable from the remaining features, so its local importance is less constrained by information already captured by those covariates. The observed heterogeneity therefore mainly reflects changes in the model’s sensitivity to hour. Together, these results show that LUCID identifies when a learned model relies on a feature and how its plausible perturbations vary given the remaining covariates. In the next paragraph, we show how group LUCID complements this individuallevel view by capturing shared information among related features, as motivated in Section 3.4.

Group LUCID complements individual local importance. Figure 7(a) compares the joint LUCID of the weather group (temperature, humidity, and season) with the sum of their individual LUCIDs. Both are small overnight, whereas the group LUCID substantially exceeds the individual sum during the day, particularly during evening rush hours. Thus, assessing the weather features individually does not fully capture their joint local predictive importance in these periods.

Figure 7(b) shows that the temperature–season pair exhibits a gap similar to that of the full weather group. Its pattern closely tracks that of the full weather group, while the temperature– humidity and humidity–season pairs contribute much smaller gaps. This might be because, for the other two pairs, season or temperature remains in the conditioning set, respectively, and can constrain the joint replacements through its dependence on the selected features. Thus, evaluating these features separately can understate their collective local importance when they share information.

![](images/fc1b7a6c9a9847a2252f4a8983a1b9aa61239008f145a50048a89e21c592d9a8.jpg)

![](images/713322a11f316ce2353ea3c267d9eabc7eb3202c2bcdd004856f2ab2ef608bb7.jpg)

![](images/0a04d2c5999c620113f1b9673fb4856e82907902f2cefc7ce395c3fb089f97b9.jpg)

![](images/82cd107b47224644f31253d45a7ca196650ec36fb3a18a313db4daa866527771.jpg)  
Figure 6: Local importance and conditional replacements for hour. (a) LUCID for hour varies substantially over the day and between working and non-working days (solid lines). (b–d) Split violins compare the estimated conditional distribution of replacement hour values using DR-BART (left) with the marginal permutation distribution (right).

![](images/6d5b32b71ce68c0d0ff7ff0f9b4d6b28f230ae4dc5a860ae533b2cbb86de37b8.jpg)

![](images/7bc5a38f40c7fab3344dd38388cdf03d942167e7c20670bd702606ebdedc0c6e.jpg)  
Figure 7: weather-group LUCID (a) The purple line shows the joint LUCID of temperature, humidity, and season, while the orange dashed line shows the sum of their individual LUCIDs; the shaded region highlights their gap. (b) Group gaps are shown for the (T, H, S), (T, S), (T, H), and (H, S) groups.

## 6 Conclusion

We present LUCID, a model-agnostic framework for local variable importance based on the full conditional feature distribution. LUCID characterizes where and how strongly a fixed model responds to conditionally plausible feature changes, with an interpretable decomposition, convergence guarantees, and Bayesian uncertainty quantification. Its flexible estimation framework accommodates modern conditional density estimators and feature groups. Future work could extend LU-CID to outcome-reference importance and model class reliance (Fisher et al., 2019), moving from explaining a fixed model toward assessing importance for the prediction problem. Also, since global conditional permutation importance is the expectation of LUCID, accurate local estimation also supports accurate global conditional importance.

## References

Aas, K., Jullum, M., and Løland, A. (2021). Explaining individual predictions when features are dependent: More accurate approximations to Shapley values. Artificial Intelligence, 298:103502.

Adlam, B., Gupta, N., Mariet, Z., and Smith, J. (2022). Understanding the bias-variance tradeof of Bregman divergences. arXiv preprint arXiv:2202.04167.

Agarwal, C., Krishna, S., Saxena, E., Pawelczyk, M., Johnson, N., Puri, I., Zitnik, M., and Lakkaraju, H. (2022). OpenXAI: Towards a transparent evaluation of model explanations. In Advances in Neural Information Processing Systems, volume 35, pages 15784–15799.

Au, Q., Herbinger, J., Stachl, C., Bischl, B., and Casalicchio, G. (2022). Grouped feature importance and combined features efect plot. Data Mining and Knowledge Discovery, 36(4):1401– 1450.

Bladen, K. K., Cutler, A., Cutler, D. R., and Moon, K. R. (2026). Conditional local importance by quantile expectations. Transactions on Machine Learning Research.

Blesch, K., Koenen, N., Kapar, J., Golchian, P., Burk, L., Loecher, M., and Wright, M. N. (2025). Conditional feature importance with generative modeling using adversarial random forests. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 15596–15604.

Borgonovo, E., Cappelli, F., Lu, X., Plischke, E., and Rudin, C. (2026). Trustworthy feature importance avoids unrestricted permutations. arXiv preprint arXiv:2604.11253.

Bregman, L. M. (1967). The relaxation method of finding the common point of convex sets and its application to the solution of problems in convex programming. USSR Computational Mathematics and Mathematical Physics, 7(3):200–217.

Breiman, L. (2001). Random forests. Machine Learning, 45(1):5–32.

Casalicchio, G., Molnar, C., and Bischl, B. (2019). Visualizing the feature importance for black box models. In Machine Learning and Knowledge Discovery in Databases, volume 11051 of Lecture Notes in Computer Science, pages 655– 670.

Chamma, A., Engemann, D. A., and Thirion, B. (2023). Statistically valid variable importance assessment through conditional permutations. In Advances in Neural Information Processing Systems, volume 36, pages 67662–67685.

Chamma, A., Thirion, B., and Engemann, D. (2024). Variable importance in highdimensional settings requires grouping. In Pro-

ceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 11195–11203.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 21(1):C1–C68.

Chipman, H. A., George, E. I., and McCulloch, R. E. (2010). BART: Bayesian additive regression trees. The Annals of Applied Statistics, 4(1):266–298.

Debeer, D. and Strobl, C. (2020). Conditional permutation importance revisited. BMC Bioinformatics, 21(1):307.

Fanaee-T, H. and Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2–3):113–127.

Fisher, A., Rudin, C., and Dominici, F. (2019). All models are wrong, but many are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously. Journal of Machine Learning Research, 20(177):1–81.

Ghosal, S., Ghosh, J. K., and van der Vaart, A. W. (2000). Convergence rates of posterior distributions. The Annals of Statistics, 28(2):500–531.

Gregorutti, B., Michel, B., and Saint-Pierre, P. (2015). Grouped variable importance with random forests and application to multiple functional data analysis. Computational Statistics & Data Analysis, 90:15–35.

Guidotti, R., Monreale, A., Ruggieri, S., Turini, F., Giannotti, F., and Pedreschi, D. (2018). A survey of methods for explaining black box models. ACM Computing Surveys, 51(5):93:1– 93:42.

Hooker, G., Mentch, L., and Zhou, S. (2021). Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance. Statistics and Computing, 31(6). Article 82.

Hornung, R. and Boulesteix, A.-L. (2022). Interaction forests: Identifying and exploiting interpretable quantitative and qualitative interaction efects. Computational Statistics & Data Analysis, 171:107460.

Izbicki, R. and Lee, A. B. (2016). Nonparametric conditional density estimation in a highdimensional regression setting. Journal of Computational and Graphical Statistics, 25(4):1297– 1316.

Jullum, M., Olsen, L. H. B., Lachmann, J., and Redelmeier, A. (2025). shapr: Explaining machine learning models with conditional Shapley values in R and Python. arXiv preprint arXiv:2504.01842.

Lei, J., G’Sell, M., Rinaldo, A., Tibshirani, R. J., and Wasserman, L. (2018). Distributionfree predictive inference for regression. Journal of the American Statistical Association, 113(523):1094–1111.

Little, C. O., Zheng, L., and Allen, G. I. (2026). iLOCO: Distribution-free inference for feature interactions. Transactions on Machine Learning Research.

Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. In Advances in Neural Information Processing Systems, volume 30, pages 4768–4777.

Molnar, C., König, G., Bischl, B., and Casalicchio, G. (2024). Model-agnostic feature importance and efects with dependent features: a conditional subgroup approach. Data Mining and Knowledge Discovery, 38(5):2903–2941.

Orlandi, V., Murray, J., Linero, A., and Volfovsky, A. (2021). Density regression with Bayesian additive regression trees. arXiv preprint arXiv:2112.12259.

Quiroga, M., Garay, P. G., Alonso, J. M., Loyola, J. M., and Martin, O. A. (2022). Bayesian additive regression trees for probabilistic programming. arXiv preprint arXiv:2206.03619.

Reyero-Lobo, A., Neuvial, P., and Thirion, B. (2025). Conditional feature importance revisited: Double robustness, eficiency and inference. arXiv preprint arXiv:2501.17520.

Ribeiro, M. T., Singh, S., and Guestrin, C. (2016). “why should i trust you?”: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1135–1144.

Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., and Zeileis, A. (2008). Conditional variable importance for random forests. BMC Bioinformatics, 9(1).

Watson, D. S., Blesch, K., Kapar, J., and Wright, M. N. (2023). Adversarial random forests for density estimation and generative modeling. In Proceedings of the 26th International Conference on Artificial Intelligence and Statistics, volume 206 of Proceedings of Machine Learning Research, pages 5357–5375. PMLR.

Road map. Section A summarizes the related work. Section B summarizes the notation used throughout the paper. Section C provides the complete theoretical results, proofs, and convergence guarantees. Section D presents implementation details, computational complexity, experimental setup, software, and data information. Section E provides complete synthetic-experiment definitions, calculations, and additional results. Section F provides additional Bike Sharing analyses and results. Section F.6 reports runtime and memory scalability results on OpenXAI benchmark datasets.

## A Related Work

Perturbation-based variable importance. Breiman (2001) introduced permutation-based VI in his seminal work on Random Forests, recognizing that if predictions change when shufling the inputs to the forest (i.e., resampling them from their respective marginal distributions), then the forest likely depends on the feature. This core principle has inspired subsequent approaches to global perturbation-based variable importance that go beyond sampling from marginal distributions, but the sampling distribution in these methods is often restricted and its validity requires additional assumptions. Residual-based approaches (Fisher et al., 2019; Chamma et al., 2023; Borgonovo et al., 2026) estimate the conditional mean of the feature of interest and use the residual distribution to generate counterfactuals, which can yield biased importance estimates when the residual distribution varies with the conditioning covariates, such as under heteroskedasticity or other changes in the conditional residual distribution. Tree- or subgroup-based approaches (Strobl et al., 2008; Debeer and Strobl, 2020; Molnar et al., 2024; Chamma et al., 2023) permute or resample feature values within regions defined by tree-based partitions, which may overlook local variation in the conditional distribution, while overly fine partitions leave too few observations for resampling. Approaches based on joint-distribution estimation (Blesch et al., 2025) instead require modeling the full joint distribution, which can become increasingly challenging as the ambient dimension grows (Izbicki and Lee, 2016). There are several recent methods that extend permutation-based importance to the local level, including Individual Conditional Importance (ICI; Casalicchio et al., 2019) and Conditional Local Importance by Quantile Expectations (CLIQUE; Bladen et al., 2026). However, these existing local perturbation-based methods ignore dependence between the perturbed feature and the remaining covariates. In contrast, LUCID defines observation-level importance with respect to the full conditional feature distribution. Accurate estimation of this distribution yields replacements that preserve dependence between the feature of interest and the remaining covariates.

Other explanation and importance paradigms. Several related methods target diferent notions of feature importance. (i) Local explanation methods also explain individual predictions by assigning contributions to features. SHapley Additive exPlanations (SHAP)-based methods (Lundberg and Lee, 2017; Aas et al., 2021) evaluate changes in model outputs as features are added or marginalized over, with conditional SHAP using conditional feature densities to account for dependencies between features (Aas et al., 2021; Jullum et al., 2025). However, their estimand difers from LUCID, as SHAP allocates the deviation of an individual prediction from a baseline across feature coalitions, whereas LUCID measures the change in model behavior induced by conditionally plausible replacements of a specified feature while holding the remaining context fixed. LIME (Ribeiro et al., 2016), by contrast, constructs a locally linear surrogate approximation of the prediction function, using the local linear regression’s parameters as explanations for the black box. Both of these methods can unfaithfully capture nonzero variable attribution to features that are inactive at the observation being explained, especially when the data exhibits discontinuities. As we show in Figure 3, LIME, SHAP, and SHAPr (a SHAP variant) all yield non-zero feature attribution even when the feature is completely unused by some specific regions of the known outcome generating function. (ii) Feature-removal methods such as leave-one-covariate-out (LOCO; Lei et al., 2018) importance measure how much predictive information is lost by retraining the predictive model with all-but-one of the original features. In practice, this change can arise from both information loss due to feature removal and errors introduced by model refitting, making it dificult to disentangle these efects. Moreover, because LOCO compares two diferent models, it does not directly quantify how the original fitted model relies on the omitted feature. Instead, it describes how much the algorithm that discovered the model relies on the feature (Fisher et al., 2019). At the population level and under squared loss, however, LOCO with optimal reduced predictor is proportional to and closely related to conditional permutation feature importance (Reyero-Lobo et al., 2025). We further explore this relationship at the observation level in Section 3.3. (iii) Grouped VI methods measure the importance of sets of related features jointly to account for shared or redundant information (Gregorutti et al., 2015; Au et al., 2022; Chamma et al., 2024; Little et al., 2026). Our framework naturally extends this idea but to the local level, quantifying conditional importance for feature groups at the observation level.

## B Notation

Table 4 summarizes the notation used in the main theory and this appendix. Throughout, $g$ is a fixed prediction model. The scalar feature of interest is denoted by $Z ,$ and the vector X contains the remaining covariates. For group importance, the feature group and its replacement are written $z$ and $\boldsymbol { Z ^ { \prime } }$ . All population expectations refer to the true covariate distribution.

Vector symbols use $\mathbf { } X , x , W , w$ , whereas individual coordinates, such as $X _ { j } , x _ { j }$ , and the scalar replacements $Z ^ { \prime } , z ^ { \prime }$ remain unbold. The model $g$ is scalar-valued in the squared-loss subsection and may have vector-valued outputs in the Bregman subsection. We use dispersion and displacement with the same colors as the main paper.

Table 4: Notation for LUCID and its theoretical guarantees.
<table><tr><td>Notation</td><td>Definition</td></tr><tr><td> $Z , \ z$ </td><td>feature of interest and its observed value.</td></tr><tr><td> $\pmb { X } = ( X _ { 1 } , \ldots , X _ { k } ) , \pmb { x }$ </td><td>Remaining covariates and their observed values; k is the number of conditioning covariates.</td></tr><tr><td> $\pmb { W } = ( Z , \pmb { X } ) , \ \pmb { w } = ( z , \pmb { x } )$ </td><td>Complete feature vector and one observation.</td></tr><tr><td> $Z ^ { \prime } , \ z ^ { \prime }$   $z , \ z , \ z ^ { \prime } , \ z ^ { \prime }$ </td><td>conditional replacement and its integration variable. Feature group, its observed vector, its joint conditional replacement, and the</td></tr><tr><td> $P _ { Z , X } , \ P _ { X } , \ P _ { Z }$ </td><td>vector integration variable True joint covariate distribution and its marginals.</td></tr><tr><td> $P ( \cdot \mid x )$ </td><td>True conditional distribution of Z given  $X = x .$ </td></tr><tr><td> $p ^ { * } ( z \mid x )$ </td><td>True conditional density, with respect to the stated dominating measure.</td></tr><tr><td> $p , \ q , \ \widehat { p }$ </td><td>Candidate conditional densities and a fitted conditional density.</td></tr><tr><td> $g ( z , \pmb { x } )$ </td><td>Fixed prediction model being explained.</td></tr><tr><td> $\ell ( a , b )$ </td><td>Scalar-valued loss comparing alternative and reference predictions; for vector arguments, write  $\ell ( a , b )$ </td></tr><tr><td> $\mathrm { L U C I D } ( w )$ </td><td>Local importance under the true conditional distribution.</td></tr><tr><td> $\operatorname { L U C I D } [ p ] ( \pmb { w } )$ </td><td>Local importance induced by a candidate conditional density  $p .$ </td></tr><tr><td> $g _ { - Z } ( { \pmb x } )$ </td><td>Conditional mean of the fixed model output given  $X = x ;$  the optimal squared-loss reduced predictor.</td></tr><tr><td> $\mathrm { L O C O } ( w )$ </td><td>Squared discrepancy between  $g ( z , \pmb { x } )$  and  $g _ { - Z } ( { \pmb x } )$ </td></tr><tr><td> $\mathrm { L O C O } _ { \phi } ( w )$ </td><td> ${ \cal D } _ { \phi } ( g _ { - Z } ( { \pmb x } ) , g ( z , { \pmb x } ) )$  , using the same conditional-mean reduced predictor.</td></tr><tr><td> $I _ { j } ( w )$ </td><td>Experimental shorthand for LUCID when the jth original feature is selected as  $Z .$ </td></tr><tr><td> $\phi , \ D _ { \phi }$ </td><td>Scalar-valued strictly convex differentiable generator and its Bregman divergence; vector arguments are written  $a , b ,$  with scalar coordinates</td></tr><tr><td> $\mathrm { T V } ( p , q )$ </td><td>Total variation distance, with the convention  ${ \frac { 1 } { 2 } } \int \left| p - q \right|$ </td></tr><tr><td> $d ( p , q )$ </td><td>Conditional-distribution distance satisfying the main paper distance-control</td></tr><tr><td> $h ( p , q )$ </td><td>condition. Integrated Hellinger distance defined in Equation (C.26);  $h ( p ( \cdot \mid x ) , q ( \cdot \mid x ) )$  denotes its fixed-context version.</td></tr><tr><td> $\| \cdot \| _ { 2 }$ </td><td> $L ^ { 2 } ( P _ { Z , { \pmb X } } )$   $L ^ { 2 } ( P _ { X } )$  norm for importance functions; is written explicitly when used.</td></tr><tr><td> $C _ { \ell } , \ A$ </td><td>Uniform loss bound and constant in the distance-control condition.</td></tr><tr><td> $\epsilon _ { n }$ </td><td>Conditional-density contraction rate, transferred to LUCID.</td></tr><tr><td> $\textstyle { \mathcal { D } } _ { n } , n$ </td><td>I.i.d. covariate sample used to fit the conditional density, and its size.</td></tr><tr><td> $w _ { i } , \textit { N }$ </td><td>Observations to be explained, and their number.</td></tr><tr><td> $\Pi , \Pi ( \cdot \mid \mathcal { D } _ { n } )$ </td><td>Prior and posterior on conditional densities.</td></tr><tr><td> $\boldsymbol { p } ^ { ( b ) } , \boldsymbol { B }$ </td><td>Retained posterior density draw and number of retained draws.</td></tr><tr><td> $\overline { { p } }$ </td><td>Complete posterior mean density.</td></tr><tr><td> $b _ { m } , ~ M$ </td><td>Random posterior index for replacement  $m ,$  and number of replacements in an estimate.</td></tr><tr><td> $Z _ { m } ^ { \prime } , \ Z _ { m } ^ { \prime ( b ) }$ </td><td>Replacement from a specified sampler, and replacement generated with posterior draw b held fixed.</td></tr><tr><td> $\widehat { \mathrm { L U C I D } _ { M } [ p ] }$ </td><td>Monte Carlo approximation of  $\mathrm { L U C I D } [ p ]$  using M replacements.</td></tr><tr><td> $\mathrm { K L } ( p ^ { * } \| p ) , \ V ( p ^ { * } , p )$ </td><td>Integrated log-density ratio and its second moment under the truth.</td></tr><tr><td> ${ \mathcal { G } } _ { n } , \ { \mathcal { N } } ( \epsilon , { \mathcal { G } } _ { n } , h )$ </td><td>Density sieve and its Hellinger covering number.</td></tr><tr><td> $L _ { n } ( p ) , \psi _ { n }$ </td><td>Likelihood ratio and binary test used only in the posterior-tail proof.</td></tr><tr><td> $\alpha , ~ \beta , ~ d _ { 0 }$ </td><td>DR-BART smoothness,  $\beta = \operatorname* { m i n } \{ \alpha , 1 \}$  , and effective dimension, including the response coordinate.</td></tr><tr><td> $\gamma , \ \widehat { Q } _ { w }$ </td><td>Tail probability for a  $( 1 - \gamma )$  credible interval, and empirical posterior quantile function at w.</td></tr></table>

## C Theoretical Results and Proofs

We first prove the structural results for the true conditional distribution. We then establish stability under conditional-distribution error and distinguish posterior contraction from the moment control needed for posterior averaging.

## C.1 Setup and Global–Local Identity

Let ${ \pmb W } = ( Z , { \pmb X } ) \sim P _ { Z , { \pmb X } }$ . A conditional replacement is constructed so that

$$
Z ^ { \prime } \mid X \sim P ( \cdot \mid X ) , \qquad Z ^ { \prime } \bot Z \mid X .\tag{C.1}
$$

For a candidate conditional density $p ,$ write

$$
\operatorname { L U C I D } [ p ] ( w ) : = \int \ell ( g ( z ^ { \prime } , x ) , g ( z , x ) ) p ( z ^ { \prime } \mid x ) d z ^ { \prime } , \qquad \operatorname { L U C I D } ( w ) : = \operatorname { L U C I D } [ p ^ { * } ] ( w ) .\tag{C.2}
$$

At fixed ${ \pmb w } = ( z , { \pmb x } )$ , expectations over $Z ^ { \prime }$ use $P ( \cdot \mid x )$ unless another distribution is specified. All conditional densities are defined with respect to a common dominating measure $\mu ;$ for discrete features, the corresponding integrals are sums.

Proposition C.1 (Global–local identity). $I f \operatorname { \mathbb { E } } | \ell ( g ( Z ^ { \prime } , \mathbf { X } ) , g ( Z , \mathbf { X } ) ) | < \infty$ , then

$$
\mathrm { C P F I } = \mathbb { E } [ \mathrm { L U C I D } ( \pmb { W } ) ] .\tag{C.3}
$$

Proof. By Equation (C.1), the conditional distribution of $Z ^ { \prime }$ given (Z, X) is $P ( \cdot \mid X )$ . Thus

$$
\begin{array} { r } { \mathbb { E } [ \ell ( g ( Z ^ { \prime } , \pmb { X } ) , g ( Z , \pmb { X } ) ) \mid Z , \pmb { X } ] = \mathrm { L U C I D } ( Z , \pmb { X } ) . } \end{array}
$$

Taking expectations yields the stated identity and the prediction-based definition of global CPFI in the main paper, Equation 2. □

## C.2 Squared-Loss Decomposition and the Relationship with LOCO

Assume first that $g$ is real-valued and the relevant second moments are finite. Define the conditionalmean reduced predictor by

$$
g _ { - Z } ( \pmb { x } ) : = \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] = \mathbb { E } [ g ( Z , \pmb { X } ) \mid \pmb { X } = \pmb { x } ] .\tag{C.4}
$$

This definition refers to the given model $g .$

Proof of Theorem 1. Under squared loss, adding and subtracting the conditional mean gives

$$
\begin{array} { r l } & { \mathrm { L U C I D } ( w ) = \mathbb { E } [ ( g ( Z ^ { \prime } , \pmb { x } ) - g ( z , \pmb { x } ) ) ^ { 2 } ] } \\ & { \qquad = \mathbb { E } [ ( g ( Z ^ { \prime } , \pmb { x } ) - \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] ) ^ { 2 } ] + ( \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] - g ( z , \pmb { x } ) ) ^ { 2 } } \\ & { \qquad + 2 ( \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] - g ( z , \pmb { x } ) ) \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) - \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] ] } \\ & { \qquad = \underbrace { \mathrm { V a r } ( g ( Z ^ { \prime } , \pmb { x } ) ) } _ { \mathrm { d i s p e r s i o n } } + \underbrace { ( g ( z , \pmb { x } ) - \mathbb { E } [ g ( Z ^ { \prime } , \pmb { x } ) ] ) ^ { 2 } } _ { \mathrm { d i s p l a c e m e n t } } . } \end{array}\tag{C.5}
$$

The cross term vanishes by Equation (C.4).

The conditional mean is the optimal squared-loss reduced predictor: for every real number $^ { a , }$

$$
\begin{array} { r } { \mathbb { E } [ ( g ( Z ^ { \prime } , \pmb { x } ) - a ) ^ { 2 } ] = \mathrm { V a r } ( g ( Z ^ { \prime } , \pmb { x } ) ) + ( g _ { - Z } ( \pmb { x } ) - a ) ^ { 2 } . } \end{array}\tag{C.6}
$$

Consequently, $g _ { - Z } ( { \pmb x } )$ uniquely minimizes this conditional risk. Recall the local LOCO comparison in the main paper,

$$
\operatorname { L O C O } ( { \pmb w } ) : = ( g ( z , x ) - g _ { - Z } ( x ) ) ^ { 2 } .\tag{C.7}
$$

A reduced model obtained by finite-sample refitting need not equal this projection, so the following identity does not assert equality for a refitted model.

Proof of Proposition 1. Substituting Equation (C.7) into Equation (C.5) gives

$$
\mathrm { L U C I D } ( \boldsymbol { w } ) = \mathrm { V a r } ( g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) ) + \mathrm { L O C O } ( \boldsymbol { w } ) .\tag{C.8}
$$

Corollary C.2 (Conditional and global averages). Following Reyero-Lobo et al. (2025), $i f \mathbb { E } [ g ( Z , \pmb { X } ) ^ { 2 } ] <$ $\infty ,$ then

$$
\mathbb { E } [ \mathrm { L O C O } ( W ) \mid X ] = \mathrm { V a r } ( g ( Z , X ) \mid X ) ,\tag{C.9}
$$

$$
\operatorname { \mathbb { E } } [ \operatorname { L U C I D } ( W ) \mid X ] = 2 \operatorname { V a r } ( g ( Z , X ) \mid X ) .\tag{C.10}
$$

In particular, dispersion and displacement have the same population mean, and

$$
\operatorname { C P F I } = 2 \mathbb { E } [ \operatorname { L O C O } ( W ) ] = 2 \mathbb { E } [ \operatorname { V a r } ( g ( Z , X ) \mid X ) ] .\tag{C.11}
$$

Proof. Equation (C.9) is the definition of conditional variance. At $\pmb { W } = ( Z , \pmb { X } )$ , the dispersion in Equation (C.8) equals $\operatorname { V a r } ( g ( Z , X ) \mid X )$ because $Z ^ { \prime }$ and Z have the same conditional distribution. Taking conditional expectations proves Equation (C.10). Averaging again and applying Proposition C.1 proves Equation (C.11). □

## C.3 Bregman Decomposition and Global Identities

Let the model outputs lie in a convex domain in a finite-dimensional Euclidean space, and let $\phi$ be strictly convex and continuously diferentiable on its relative interior. Its Bregman divergence (Bregman, 1967) is

$$
D _ { \phi } ( \mathbf { a } , b ) : = \phi ( \mathbf { a } ) - \phi ( b ) - \langle \nabla \phi ( b ) , \mathbf { a } - b \rangle .\tag{C.12}
$$

For each fixed ${ \mathbf { } } ^ { \mathbf { } } \mathbf { { \mathbf { x } } } ,$ assume that $g ( Z ^ { \prime } , \pmb { x } )$ has a finite mean and that both $g ( z , \pmb { x } )$ and this mean, $g _ { - Z } ( { \pmb x } )$ , lie in the relative interior of the domain of $\phi .$ . Assume also that the expected Bregman divergences appearing below are finite. For vector-valued outputs, $g _ { - Z } ( { \pmb x } )$ is obtained by taking the conditional expectation of each output component in Equation (C.4). If the output domain lies in a lower-dimensional afine space, gradients are taken within that space; for example, for probability vectors, derivatives are taken along directions that preserve the constraint that the components sum to one.

Proof of Proposition 2. Expanding Equation (C.12) gives the three-point identity

$$
\begin{array} { r l } { D _ { s } \langle \theta ( \mathcal { Z } ^ { \prime } , x , x ) , \theta ( \mathcal { Z } ^ { \prime } , x ) \rangle = \phi ( \theta ( \mathcal { Z } ^ { \prime } , x ) ) - \phi ( \theta ( \mathcal { Z } ^ { \prime } , x ) ) - \langle \theta ( \mathcal { Z } ^ { \prime } ( s , x ) ) , \theta ( \mathcal { Z } ^ { \prime } , x ) \rangle - g ( \mathcal { Z } ^ { \prime } , x ) \rangle } & { } \\ & { = \phi ( \theta ( \mathcal { Z } ^ { \prime } , x ) ) - \phi ( \theta ( \mathcal { Z } ^ { \prime } , x ) ) + \phi ( \mathcal { Z } ^ { \prime } , 0 ) \rangle } & { + \phi ( \mathcal { Z } ^ { \prime } , 0 ) | ) - \phi ( \mathcal { Z } ^ { \prime } , x ) | ) - \phi ( \mathcal { Z } ^ { \prime } , x ) | \rangle } \\ & { \quad - \langle \nabla \phi ( \theta ( \mathcal { Z } , x ) ) , \theta ( \mathcal { Z } ^ { \prime } , x ) \rangle - \mathbb { E } [ \phi ( \mathcal { Z } ^ { \prime } , x ) ] \rangle } \\ & { \quad - \langle \nabla \phi ( \theta ( \mathcal { Z } ^ { \prime } , x ) ) , \theta ( \mathcal { Z } ^ { \prime } , x ) \rangle - \mathbb { E } [ \phi ( \mathcal { Z } ^ { \prime } , x ) ] \rangle } \\ & { = [ \phi ( \mathcal { Z } ^ { \prime } , x ) ) - \phi ( \mathbb { E } [ \mathcal { Z } ^ { \prime } , x ) ] ] - g ( \mathcal { Z } ^ { \prime } , x ) \rangle } \\ & { \quad - \langle \nabla \phi ( \mathbb { E } [ \mathcal { Z } ^ { \prime } , x ) ) , \phi ( \mathcal { Z } ^ { \prime } , x ) - \mathbb { E } [ \mathcal { Z } ( \mathcal { Z } ^ { \prime } , x ) ] \rangle ] } \\ & { \quad + [ \phi ( \mathcal { W } _ { \mathcal { Z } } ( \mathcal { Z } ^ { \prime } , x ) ) - \phi ( \mathcal { Z } ( \mathcal { Z } ^ { \prime } , x ) ) ] } \\ &  \quad \quad - \langle \nabla \phi ( \mathcal { Z } ^ { \prime } , \end{array}
$$

The last term has mean zero. Hence, for $\ell = D _ { \phi }$

$$
\begin{array} { r } { \mathrm { L U C I D } ( \boldsymbol { w } ) = \underbrace { \mathbb { E } [ D _ { \phi } ( g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) , \mathbb { E } [ g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) ] ) ] } _ { \mathrm { d i s p e r s i o n } } + \underbrace { D _ { \phi } ( \mathbb { E } [ g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) ] , g ( \boldsymbol { z } , \boldsymbol { x } ) ) } _ { \mathrm { d i s p l a c e m e n t } } . } \end{array}\tag{C.13}
$$

For $\phi ( \pmb { a } ) = \| \pmb { a } \| ^ { 2 }$ , the Bregman divergence reduces to squared Euclidean distance, so Equation (C.13) becomes

$$
\mathrm { L U C I D } ( \boldsymbol { w } ) = \underbrace { \mathbb { E } [ \| g ( Z ^ { \prime } , \boldsymbol { x } ) - g _ { - Z } ( \boldsymbol { x } ) \| ^ { 2 } ] } _ { \mathrm { d i s p e r s i o n } } + \underbrace { \| g _ { - Z } ( \boldsymbol { x } ) - g ( \boldsymbol { z } , \boldsymbol { x } ) \| ^ { 2 } } _ { \mathrm { d i s p l a c e m e n t } } .\tag{C.14}
$$

For scalar outputs, the dispersion term is $\mathrm { V a r } ( g ( Z ^ { \prime } , { \pmb x } ) )$ , recovering the squared-loss decomposition.

For strictly positive probability-vector outputs and negative entropy $\begin{array} { r } { \phi ( { \pmb a } ) = \sum _ { c } a _ { c } } \end{array}$ log $\textstyle { a _ { c } , }$ , the Bregman divergence is $\mathrm { K L } ( { \pmb a } | | { \pmb b } )$ . Thus, whenever the expectations are finite,

$$
\mathrm { L U C I D } ( \boldsymbol { w } ) = \underbrace { \mathbb { E } [ \mathrm { K L } ( g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) | | g _ { - Z } ( \boldsymbol { x } ) ) ] } _ { \mathrm { d i s p e r s i o n } } + \underbrace { \mathrm { K L } ( g _ { - Z } ( \boldsymbol { x } ) | | g ( \boldsymbol { z } , \boldsymbol { x } ) ) } _ { \mathrm { d i s p l a c e m e n t } } .\tag{C.15}
$$

The direction of the LOCO comparison. The main paper places the alternative prediction first and the reference prediction second in ℓ to show that we measure the alternative prediction using the original model output as the reference. Keeping this order, the Bregman comparison with the conditional-mean reduced predictor is

$$
\mathrm { L O C O } _ { \phi } ( \pmb { w } ) : = D _ { \phi } ( g _ { - Z } ( \pmb { x } ) , g ( z , \pmb { x } ) ) .\tag{C.16}
$$

It is exactly the displacement term in Equation (C.13). Here, we retain the conditional-mean reduced predictor, which need not minimize the Bregman loss when placed in its first argument. Reversing the comparison leads to an analogous formulation in dual coordinates, with a corresponding dualmean predictor. As this extension is beyond the main scope of our paper, we omit its details and refer readers to Adlam et al. (2022) for further discussion about Bregman divergence.

Corollary C.3 (Zero importance). For squared loss or a well-defined Bregman divergence with a strictly convex generator,

$$
\mathrm { L U C I D } ( \pmb { w } ) = 0 \quad \Longleftrightarrow \quad g ( Z ^ { \prime } , \pmb { x } ) = g ( z , \pmb { x } ) \quad P ( \cdot \mid \pmb { x } ) - a l m o s t s u r e l y .\tag{C.17}
$$

Proof. By the definition of LUCID,

$$
\mathrm { L U C I D } ( \boldsymbol { w } ) = \mathbb { E } \big [ \ell ( g ( \boldsymbol { Z } ^ { \prime } , \boldsymbol { x } ) , g ( z , \boldsymbol { x } ) ) \big ] , \qquad \boldsymbol { Z } ^ { \prime } \sim P ( \cdot \mid \boldsymbol { x } ) .
$$

For squared loss, $\ell ( { \pmb a } , { \pmb b } ) = \| { \pmb a } - { \pmb b } \| ^ { 2 }$ is nonnegative and equals zero if and only if ${ \pmb a } = { \pmb b }$ . For Bregman divergence, strict convexity of ϕ implies

$$
\phi ( \pmb { a } ) > \phi ( \pmb { b } ) + \langle \nabla \phi ( \pmb { b } ) , \pmb { a } - \pmb { b } \rangle \qquad \mathrm { w h e n e v e r } \ \pmb { a } \neq \pmb { b } .
$$

Together with Equation (C.12), this gives $D _ { \phi } ( { \pmb a } , { \pmb b } ) \geq 0$ , with equality if and only if ${ \pmb a } = { \pmb b }$ . Thus, if $\mathrm { L U C I D } ( \pmb { w } ) = 0$ , the nonnegative loss inside the expectation must vanish $P ( \cdot \mid x )$ -almost surely, implying $g ( Z ^ { \prime } , \pmb { x } ) = g ( z , \pmb { x } )$ almost surely. Conversely, this almost-sure equality makes the loss zero almost surely and hence gives $\mathrm { L U C I D } ( \pmb { w } ) = 0$ □

Proposition C.4 (Global Bregman identity). Under the preceding conditions and population integrability of the displayed terms,

$$
\mathrm { C P F I } = \mathbb { E } [ D _ { \phi } ( g ( Z , X ) , g _ { - Z } ( X ) ) ] + \mathbb { E } [ D _ { \phi } ( g _ { - Z } ( X ) , g ( Z , X ) ) ] .\tag{C.18}
$$

Proof. Average Equation (C.13) over $( Z , X )$ . Conditional on $\boldsymbol { X }$ , the outputs $g ( Z ^ { \prime } , X )$ and $g ( Z , X )$ have the same distribution and conditional mean $g _ { - Z } ( X )$ . Therefore, the average dispersion equals $\mathbb { E } [ D _ { \phi } ( g ( Z , X ) , g _ { - Z } ( X ) ) ]$ . The average displacement is the second term in Equation (C.18).

When the relevant generator expectations are finite, the first term also satisfies

$$
\begin{array} { r } { { \mathbb { E } } [ D _ { \phi } ( g ( Z , { \pmb X } ) , g _ { - Z } ( { \pmb X } ) ) ] = { \mathbb { E } } [ \phi ( g ( Z , { \pmb X } ) ) ] - { \mathbb { E } } [ \phi ( g _ { - Z } ( { \pmb X } ) ) ] , } \end{array}\tag{C.19}
$$

since the conditional mean of $g ( Z , X ) - g _ { - Z } ( X )$ given X is zero. For $\phi ( \pmb { a } ) = \| \pmb { a } \| ^ { 2 }$ , the two divergences in Equation (C.18) coincide, recovering the factor-of-two identity. For a strictly positive probability-vector output and negative entropy $\begin{array} { r } { \phi ( \mathbf { a } ) = \sum _ { c } a _ { c } \log a _ { c } } \end{array}$ , direct substitution gives

$$
\mathrm { C P F I } = \mathbb { E } [ \mathrm { K L } ( g ( Z , \pmb { X } ) \| g _ { - Z } ( \pmb { X } ) ) ] + \mathbb { E } [ \mathrm { K L } ( g _ { - Z } ( \pmb { X } ) \| g ( Z , \pmb { X } ) ) ] .\tag{C.20}
$$

These two expectations need not agree. Consequently, even though local displacement is the comparison in Equation (C.16), global CPFI need not be twice its population mean.

Feature groups. All results above also hold for a feature group $Z ,$ with $\boldsymbol { Z ^ { \prime } }$ drawn jointly conditional on the remaining covariates X. In the corresponding formulas, replace $Z , z , Z ^ { \prime } , z ^ { \prime }$ by $Z , z , Z ^ { \prime } , z ^ { \prime }$ , respectively. The proofs use the conditional distribution of the model output and do not require scalar feature replacements.

## C.4 Stability Under Conditional-Distribution Error

Throughout this subsection, suppose

$$
| \ell ( g ( z ^ { \prime } , \pmb { x } ) , g ( z , \pmb { x } ) ) | \leq C _ { \ell } < \infty\tag{C.21}
$$

for $P _ { Z , X }$ -almost every $( z , \pmb { x } )$ and replacement values in the supports of the candidate distributions. Bounded outputs of a fixed model sufice for squared loss. Write $\| \cdot \| _ { 2 } = \| \cdot \| _ { L ^ { 2 } ( P _ { Z , X } ) }$

Proof of Theorem 2. For every pair $p , q$ satisfying Equation (C.21),

$$
\begin{array} { r l } { | \operatorname { L U C I D } [ p ] ( \pmb { w } ) - \operatorname { L U C I D } [ q ] ( \pmb { w } ) | \leq C _ { \ell } \displaystyle \int | p ( z ^ { \prime } \mid \pmb { x } ) - q ( z ^ { \prime } \mid \pmb { x } ) | d z ^ { \prime } } & { } \\ { = 2 C _ { \ell } \operatorname { T V } \{ p ( \cdot \mid \pmb { x } ) , q ( \cdot \mid \pmb { x } ) \} . } & { } \end{array}\tag{C.22}
$$

The right-hand side depends only on x. Squaring and integrating with respect to $P _ { Z , X }$ , and applying the main paper distance-control assumption, yields

$$
\begin{array} { r l } & { \| \operatorname { L U C I D } [ p ] - \operatorname { L U C I D } [ q ] \| _ { 2 } \leq 2 C _ { \ell } \| \operatorname { T V } \{ p ( \cdot \mid X ) , q ( \cdot \mid X ) \} \| _ { L ^ { 2 } ( P _ { X } ) } } \\ & { \qquad \leq 2 C _ { \ell } A d ( p , q ) . } \end{array}\tag{C.23}
$$

Taking $p = \widehat { p }$ and $q = p ^ { * }$ proves the stated $O _ { P } ( \epsilon _ { n } )$ rate whenever $d ( \widehat { p } , p ^ { * } ) = O _ { P } ( \epsilon _ { n } )$

Local estimation accuracy also controls the error in population-average importance. Since $\mathrm { C P F I } = \mathbb { E } [ \mathrm { L U C I D } [ p ^ { * } ] ( W ) ]$ , Cauchy–Schwarz gives

$$
\begin{array} { r l } & { \mathbb { E } [ \mathrm { L U C I D } [ p ] ( W ) ] - \mathrm { C P F I } | = | \mathbb { E } [ \mathrm { L U C I D } [ p ] ( W ) - \mathrm { L U C I D } [ p ^ { * } ] ( W ) ] | } \\ & { \qquad \leq \left( \mathbb { E } \Big [ | \mathrm { L U C I D } [ p ] ( W ) - \mathrm { L U C I D } [ p ^ { * } ] ( W ) | ^ { 2 } \Big ] \right) ^ { 1 / 2 } } \\ & { \qquad = \| \mathrm { L U C I D } [ p ] - \mathrm { L U C I D } [ p ^ { * } ] \| _ { 2 } . } \end{array}\tag{C.24}
$$

Here, expectations are taken over the true covariate distribution of W. Thus, the root mean squared error of the local importance function bounds the absolute error of its population average.

When the population average is approximated using N evaluation observations, the total error decomposes as

$$
\begin{array} { r l } { \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { L U C I D } [ p ] ( { \boldsymbol w } _ { i } ) \mathrm { - C P F I } = \underbrace { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { L U C I D } [ p ] ( { \boldsymbol w } _ { i } ) \mathrm { - } \mathbb { E } [ \mathrm { L U C I D } [ p ] ( { \boldsymbol W } ) ] } _ { \mathrm { e v a l u a t i o n - s a m p l e ~ e r r o r } } } & { } \\ { \displaystyle + \underbrace { \mathbb { E } [ \mathrm { L U C I D } [ p ] ( { \boldsymbol W } ) ] - \mathrm { C P F I } } _ { \mathrm { c o n d i t i o n a l - d e n s i t y ~ e s t i m a t i o n ~ e r r o r } } . } \end{array}\tag{C.25}
$$

Equation (C.24) controls the second term. The first is a separate sampling error arising from the finite evaluation sample.

Hellinger distance. Define

$$
h ^ { 2 } ( p , q ) : = \int \int \{ \sqrt { p ( z ^ { \prime } \mid x ) } - \sqrt { q ( z ^ { \prime } \mid x ) } \} ^ { 2 } d z ^ { \prime } P _ { X } ( d x ) .\tag{C.26}
$$

The corresponding fixed-context distance omits the integral over $P _ { X }$ . Factoring $p - q = ( { \sqrt { p } } -$ ${ \sqrt { q } } ) ( { \sqrt { p } } + { \sqrt { q } } )$ and applying Cauchy–Schwarz gives

$$
\int | p ( z ^ { \prime } \mid x ) - q ( z ^ { \prime } \mid x ) | d z ^ { \prime } \leq 2 h ( p ( \cdot \mid x ) , q ( \cdot \mid x ) ) ,
$$

because $\textstyle \int ( { \sqrt { p } } + { \sqrt { q } } ) ^ { 2 } \leq 4$ . Thus the distance-control condition holds with $d = h$ and $A = 1$ , and

$$
\| \mathrm { L U C I D } [ p ] - \mathrm { L U C I D } [ q ] \| _ { 2 } \leq 2 C _ { \ell } h ( p , q ) .\tag{C.27}
$$

The normalization in Equation (C.26) also gives $h ^ { 2 } ( p , q ) \leq 2$

## C.5 Posterior Contraction and Posterior Moment Control

We now establish convergence guarantees for DR-BART-based (Orlandi et al., 2021) LUCID under the regularity, prior, and bounded-loss conditions stated below. Readers interested only in posterior contraction for LUCID may proceed directly to Corollary C.5 and then Section C.6, where we verify the conditions needed to apply Equation (C.28). The remainder of this subsection addresses a further question relevant to our implementation: does the posterior mean conditional density

$$
{ \bar { p } } ( z \mid \mathbf { \mathscr { x } } ) : = \mathbb { E } _ { \Pi } [ p ( z \mid \mathbf { \mathscr { x } } ) \mid { \mathcal { D } } _ { n } ]
$$

inherit the same convergence rate? Posterior contraction alone does not automatically guarantee this rate, because the posterior mean also includes contributions from the posterior tails. We therefore establish quantitative tail bounds and posterior second-moment control, which allow us to retain the rate when passing from posterior draws to the posterior mean density and its induced LUCID.

Let $\mathcal { D } _ { n } = \{ ( Z _ { i } , \pmb { X } _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. from $P _ { Z , X }$ , and let $\Pi ( { \cdot \mid } D _ { n } )$ be the posterior on conditional densities. All $O _ { P }$ statements are with respect to the randomness of the data under the true sampling distribution, together with any posterior or Monte Carlo sampling used to compute the estimator.

Corollary C.5 (Posterior contraction transfer). Suppose Equation (C.21) holds over the posterior support. If, for some $K > 0$

$$
\Pi \{ d ( p , p ^ { * } ) > K \epsilon _ { n } \ | \ { \mathcal { D } } _ { n } \} \longrightarrow 0 \quad i n \ p r o b a b i l i t y ,
$$

and d satisfies the distance-control condition with constant A, then

$$
\Pi \{ \| \mathrm { L U C I D } [ p ] - \mathrm { L U C I D } [ p ^ { * } ] \| _ { 2 } > 2 C _ { \ell } A K \epsilon _ { n } | \mathcal { D } _ { n } \} \longrightarrow 0\tag{C.28}
$$

in probability.

Proof. By Equation (C.23) and the distance-control condition,

$$
\| \operatorname { L U C I D } [ p ] - \operatorname { L U C I D } [ p ^ { * } ] \| _ { 2 } \leq 2 C _ { \ell } A d ( p , p ^ { * } )
$$

over the posterior support. Hence, an importance error exceeding $2 C _ { \ell } A K \epsilon _ { n }$ requires $d ( p , p ^ { * } ) > K \epsilon _ { n }$ so

$$
\Pi \{ \| \operatorname { L U C I D } [ p ] - \operatorname { L U C I D } [ p ^ { * } ] \| _ { 2 } > 2 C _ { \ell } A K \epsilon _ { n } \mid \mathcal { D } _ { n } \} \le \Pi \{ d ( p , p ^ { * } ) > K \epsilon _ { n } \mid \mathcal { D } _ { n } \} .
$$

The right-hand side converges to zero in probability by assumption, proving the claim.

This conclusion describes concentration of the posterior distribution. To establish the same rate for posterior averages, we next control a posterior second moment. The following lemma makes explicit the additional consequence of the standard sieve-and-testing conditions used in density contraction proofs. Its argument follows the testing framework of Ghosal et al. (2000); Orlandi et al. (2021) .

Define the integrated log-likelihood moments by

$$
\mathrm { K L } ( p ^ { * } \| p ) : = \mathbb { E } _ { P _ { Z , X } } \left[ \log \frac { p ^ { * } ( Z \mid X ) } { p ( Z \mid X ) } \right] , \qquad V ( p ^ { * } , p ) : = \mathbb { E } _ { P _ { Z , X } } \left[ \left\{ \log \frac { p ^ { * } ( Z \mid X ) } { p ( Z \mid X ) } \right\} ^ { 2 } \right] .\tag{C.29}
$$

Lemma C.6 (Posterior tail and second-moment control). Let $\epsilon _ { n } \to 0$ and $n \epsilon _ { n } ^ { 2 } \to \infty$ . Suppose there are constants $C _ { 1 } , \ldots , C _ { 5 } > 0$ and deterministic density sieves ${ \mathcal { G } } _ { n }$ such that, for all suficiently large $n ,$

$$
G 1 : \Pi \{ p : \mathrm { K L } ( p ^ { * } \| p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 } , \ V ( p ^ { * } , p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 } \} \geq e ^ { - C _ { 2 } n \epsilon _ { n } ^ { 2 } } ,\tag{C.30}
$$

$$
G 2 : \Pi ( { \mathcal { G } } _ { n } ^ { c } ) \leq C _ { 3 } e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } ,\tag{C.31}
$$

$$
G 3 : \log \mathcal { N } ( C _ { 5 } \epsilon _ { n } , \mathcal { G } _ { n } , h ) \leq C _ { 4 } n \epsilon _ { n } ^ { 2 } .\tag{C.32}
$$

Then, for a suficiently large fixed K,

$$
P _ { Z , X } ^ { \otimes n } \left[ \Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \ | \ \mathcal { D } _ { n } \} > 2 e ^ { - n \epsilon _ { n } ^ { 2 } } \right] \leq \frac { C _ { 1 } ^ { 2 } } { n \epsilon _ { n } ^ { 2 } } + 3 e ^ { - n \epsilon _ { n } ^ { 2 } }\tag{C.33}
$$

for all suficiently large n. If also $\log ( 1 / \epsilon _ { n } ) = o ( n \epsilon _ { n } ^ { 2 } )$ , then

$$
\mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid { \mathcal { D } } _ { n } ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) .\tag{C.34}
$$

Under Equation (C.21), it follows that

$$
\mathbb { E } _ { \Pi } [ \| \mathrm { L U C I D } [ p ] - \mathrm { L U C I D } [ p ^ { * } ] \| _ { 2 } ^ { 2 } \ | \ \mathcal { D } _ { n } ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) .\tag{C.35}
$$

Proof. Write

$$
L _ { n } ( p ) : = \prod _ { i = 1 } ^ { n } { \frac { p ( Z _ { i } \mid X _ { i } ) } { p ^ { * } ( Z _ { i } \mid X _ { i } ) } } .
$$

Proof of Equation C.33 By Bayes’ theorem, the posterior tail probability $\Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \ |$ $\mathcal { D } _ { n } \big \}$ can be written as

$$
\begin{array} { r l } & { \Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \ | \ \mathcal { D } _ { n } \} = \frac { \int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } \prod _ { i = 1 } ^ { n } p ( Z _ { i } \ | \ X _ { i } ) d \Pi ( p ) } { \int \prod _ { i = 1 } ^ { n } p ( Z _ { i } \ | \ X _ { i } ) d \Pi ( p ) } } \\ & { \quad \quad \quad \quad = \frac { \int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } \prod _ { i = 1 } ^ { n } p ( Z _ { i } | X _ { i } ) } { \int \displaystyle \prod _ { i = 1 } ^ { n } p ^ { * } ( Z _ { i } | X _ { i } ) } d \Pi ( p ) } \\ & { \quad \quad \quad \quad = \frac { \int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } L _ { n } ( p ) d \Pi ( p ) } { \int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } L _ { n } ( p ) d \Pi ( p ) } } \end{array}
$$

Our goal is to show that this posterior tail probability $\Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \mid { \mathcal D } _ { n } \}$ is large only on an event of small sampling probability.

(i) The denominator $\textstyle \int L _ { n } ( p ) d \Pi ( p )$ is large with high probability

We consider the denominator $\textstyle \int L _ { n } ( p ) d \Pi ( p )$ . Let $B _ { n }$ be the KL neighborhood in Equation (C.30), and let $\Pi _ { B _ { n } }$ denote the prior normalized on $B _ { n }$ , defined by

$$
d \Pi _ { B _ { n } } ( p ) : = \frac { \mathbf { 1 } _ { \{ p \in \mathcal { B } _ { n } \} } d \Pi ( p ) } { \Pi ( \mathcal { B } _ { n } ) } .
$$

By G1 (Equation C.30), every $p \in B _ { n }$ satisfies $\mathrm { K L } ( p ^ { * } \| p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 } , \ V ( p ^ { * } , p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 }$ and $B _ { n }$ satisfies $\Pi ( B _ { n } ) \overset { \cdot } { \geq } e ^ { - C _ { 2 } n \epsilon _ { n } ^ { 2 } }$

Since the prior mass $\Pi ( B _ { n } )$ is bounded below, we restrict the integral to $B _ { n }$ and express it in terms of the normalized prior $\Pi _ { B _ { n } }$ :

$$
\int L _ { n } ( p ) d \Pi ( p ) \geq \int _ { { \mathcal { B } } _ { n } } L _ { n } ( p ) d \Pi ( p ) = \Pi ( { \mathcal { B } } _ { n } ) \int L _ { n } ( p ) d \Pi _ { { \mathcal { B } } _ { n } } ( p ) .
$$

Taking logarithms and applying Jensen’s inequality gives

$$
\log \int L _ { n } ( p ) d \Pi ( p ) \geq \log \Pi ( \mathcal { B } _ { n } ) + \int \log L _ { n } ( p ) d \Pi _ { \mathcal { B } _ { n } } ( p ) .
$$

For the first term: log $\Pi ( B _ { n } ) \ge - C _ { 2 } n \epsilon _ { n } ^ { 2 }$

For the second term: We first compute the mean and variance of the second term and then use Chebyshev’s inequality to show that the denominator is not too small with high probability.

$$
\begin{array} { r l r } & { } & { \displaystyle { \int \log L _ { n } ( p ) d \Pi _ { \mathcal B _ { n } } ( p ) } = { \int \log \prod _ { i = 1 } ^ { n } \frac { p ( Z _ { i } \mid X _ { i } ) } { p ^ { * } ( Z _ { i } \mid X _ { i } ) } d \Pi _ { \mathcal B _ { n } } ( p ) } } \\ & { } & { \displaystyle { = \int \sum _ { i = 1 } ^ { n } \log \frac { p ( Z _ { i } \mid X _ { i } ) } { p ^ { * } ( Z _ { i } \mid X _ { i } ) } d \Pi _ { \mathcal B _ { n } } ( p ) } } \\ & { } & { \displaystyle { = \sum _ { i = 1 } ^ { n } \int \log \frac { p ( Z _ { i } \mid X _ { i } ) } { p ^ { * } ( Z _ { i } \mid X _ { i } ) } d \Pi _ { \mathcal B _ { n } } ( p ) } } \end{array}
$$

Let’s denote

$$
A _ { i } = \int \log \frac { p ( Z _ { i } \mid \mathbf { X } _ { i } ) } { p ^ { * } ( Z _ { i } \mid \mathbf { X } _ { i } ) } d \Pi _ { \mathcal { B } _ { n } } ( p )
$$

Since each $A _ { i }$ is a measurable function of $( Z _ { i } , \pmb { X } _ { i } )$ , the variables $A _ { 1 } , \ldots , A _ { n }$ are independent under the true sampling distribution $P _ { Z , X } ^ { \otimes n }$ . We have

$$
\begin{array} { r l } { \mathbb { E } _ { P _ { S , \kappa } ^ { \rho } } \Bigg [ \Bigg | \log L _ { n } ( p ) \mathrm { d } \Pi _ { B _ { n } } ( p ) \Bigg | = \mathbb { E } _ { P _ { S , \kappa } ^ { \rho * } } \Bigg [ \Bigg | _ { \Omega } ^ { n } A _ { \star } \Bigg ] } & { } \\ & { = \pi \mathbb { E } _ { P _ { S , \kappa } } [ A ] \Bigg | } \\ & { = \pi \mathbb { E } _ { P _ { S , \kappa } } \Bigg [ \Bigg / \log \frac { p ( Z _ { 1 } \mid X _ { 1 } ) } { p ^ { * } ( Z _ { 1 } \mid X _ { 1 } ) } \mathrm { d } \Pi _ { B _ { n } } ( p ) \Bigg ] } \\ & { = \pi \int \mathbb { E } _ { P _ { Z , \kappa } } \Bigg [ \log \frac { p ( Z _ { 1 } \mid X _ { 1 } ) } { p ^ { * } ( Z _ { 1 } \mid X _ { 1 } ) } \Bigg | \mathrm { d } \Pi _ { B _ { n } } ( p ) \Bigg ] } \\ & { = - n \int \mathrm { K L } ( p ^ { * } \mid p ) \mathrm { d } \Pi _ { B _ { n } } ( p ) } \\ & { \geq - n \mathcal { D } _ { 1 } ^ { \circ } \epsilon _ { n } ^ { 2 } \Bigg / \mathrm { d } \Pi _ { B _ { n } } ( p ) } \\ & { = - n \mathcal { D } _ { 1 } ^ { \circ } \epsilon _ { n } ^ { 2 } } \end{array}
$$

$$
\begin{array} { r l } { \Lambda _ { 4 , 5 , 5 } \int _ { \Omega } \Lambda _ { 4 , 6 , 7 } ( g ) d \Omega _ { 1 , 5 } } & { = - \alpha _ { 2 , 2 , 3 } [ \sum _ { k = 1 } ^ { 2 } \Lambda _ { 4 , 1 }  } \\ & { \qquad \quad - \alpha _ { 3 , 2 , 4 } ( g ^ { \prime } ) \Lambda _ { 4 , 1 } ^ { \prime } ( g ^ { \prime } ) } \\ & { \qquad - \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 1 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 2 } ^ { \prime } ( g ^ { \prime } ) ] } \\ & { \quad - \alpha _ { 3 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 1 } ^ { \prime } ( g ^ { \prime } ) } \\ & { \qquad \quad - \alpha _ { 4 , 3 } ^ { \prime } ( \sum _ { k = 1 } ^ { 2 } \alpha _ { 2 , 2 } \Lambda _ { 3 , 4 } ^ { \prime } ) \Lambda _ { 4 , 4 , 4 } ( g ^ { \prime } ) - \alpha _ { 3 , 3 } [  | \int _ { 0 } ^ { 1 } \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 2 } ^ { \prime } ( g ^ { \prime } ) | ^ { 2 } ] } \\ & { \qquad \quad - \alpha _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \Lambda _ { 4 , 4 } ^ { \prime } ( g ^ { \prime } ) } \\ &  \qquad \quad - \alpha _ { 5 , 3 } ^ { \prime } ( g ^ { \prime } ) ( g ^ { \prime } ) \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \int _ { 0 } ^ { 1 } \Lambda _ { 4 , 3 } ^ { \prime } ( g ^ { \prime } ) \ \end{array}
$$

Thus, the expectation is at least $- C _ { 1 } ^ { 2 } n \epsilon _ { n } ^ { 2 }$ and the variance is at most $C _ { 1 } ^ { 2 } n \epsilon _ { n } ^ { 2 }$ On the event that R log $L _ { n } ( p ) d \Pi _ { B _ { n } } ( p ) \geq - ( C _ { 1 } ^ { 2 } + 1 ) n \epsilon _ { n } ^ { 2 }$ , the preceding bounds imply log $\begin{array} { r } { \int L _ { n } ( p ) d \Pi ( p ) \geq } \end{array}$

$- ( C _ { 1 } ^ { 2 } + 1 + C _ { 2 } ) n \epsilon _ { n } ^ { 2 }$ . Chebyshev’s inequality therefore yields

$$
\begin{array} { r l } { P _ { \xi , x } ^ { \mathrm { o s } } \bigg \{ \int L _ { n } ( \psi ) \mathrm { d } [ \eta ] \varphi \big \} \leq \epsilon \ ( \mathcal { Q } _ { 2 } | \mathcal { C } _ { 1 } ^ { 2 } | 1 ) | \mathcal { H } _ { n } ^ { 2 } \bigg \} = P _ { \xi , x } ^ { \mathrm { o s } } \bigg \{ \ln ( \int L _ { n } ( \psi ) \mathrm { d } \Pi ( \bar { p } ) \zeta - \zeta ( \mathcal { Q } _ { 2 } + C _ { 1 } ^ { 2 } + 1 ) ) | \alpha _ { n } ^ { 2 } \bigg \} } \\ & { \leq P _ { \xi , x } ^ { \mathrm { o s } } \bigg \{ \int \exp { L _ { n } ( \psi ) } \mathrm { d } \ln _ { s } ( \psi ) < ( \mathcal { C } _ { 1 } ^ { 2 } + 1 ) | \alpha _ { n } ^ { 2 } \bigg \} } \\ & { = P _ { \mathcal { Z } , X } ^ { \mathrm { o s } } \bigg \{ \int \exp { L _ { n } ( \psi ) } \mathrm { d } \ln _ { s } ( \psi ) + C _ { 1 } ^ { 2 } w _ { n } ^ { 2 } < - w _ { n } ^ { 2 } \bigg \} } \\ & { = P _ { \mathcal { Z } , X } ^ { \mathrm { o s } } \bigg \{ \int \exp { L _ { n } ( \psi ) } \mathrm { d } \ln _ { s } ( \psi ) - \mathbb { E } _ { P _ { \mathcal { Z } , X } ^ { \mathrm { o s } } } ^ { 2 } \bigg \{ \int \log { L _ { n } ( \psi ) } \mathrm { d } \Pi _ { X _ { n } } ( \bar { p } ) \bigg \} < - w _ { n } ^ { 2 } \bigg \} } \\ &  \leq P _ { \mathcal { Z } , X } ^ { \mathrm { o s } } \bigg \{ \int \exp { L _ { n } ( \psi ) } \mathrm { d } \Pi _ { X _ { n } } ( \psi ) - \mathbb { E } _ { P _ { \mathcal { Z } , X } ^ { \mathrm { o s } } } ^ { 2 } \bigg [ \int \log { L _ { n } ( \psi ) } \mathrm { d } \Pi _ { X _ { n } } ( \bar { p } ) \bigg ] \ \leq \end{array}
$$

Combining the above bounds, with probability at least

$$
1 - \frac { C _ { 1 } ^ { 2 } } { n \epsilon _ { n } ^ { 2 } } ,
$$

the denominator is bounded below by

$$
e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 1 ) n \epsilon _ { n } ^ { 2 } } .
$$

(ii) The numerator $\begin{array} { r } { \int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } L _ { n } ( p ) d \Pi ( p ) } \end{array}$ is large only with low probability We use deterministic density sieves ${ \mathcal { G } } _ { n }$ satisfying G2 (Equation C.31) to control $\Pi ( \mathcal { G } _ { n } ^ { c } ) \leq$ $C _ { 3 } e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } }$ The main dificulty is to control the contribution from ${ \cal A } _ { n } : = \{ h ( p , p ^ { * } ) ~ >$ $K \epsilon _ { n } \} \cap { \mathcal { G } } _ { n }$ . More precisely,

$$
\int _ { h ( p , p ^ { * } ) > K \epsilon _ { n } } L _ { n } ( p ) d \Pi ( p ) \leq \int _ { \mathcal { A } _ { n } } L _ { n } ( p ) d \Pi ( p ) + \int _ { \mathcal { G } _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p )
$$

For the second term, the argument is simpler. Since we will ultimately apply Markov inequality to bound the probability, we first bound $\begin{array} { r l } { { \mathbb E } _ { P _ { Z . X } ^ { \otimes n } } \biggl [ \int _ { \mathcal G _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p ) \biggr ] } \end{array}$ . For later use, we establish the following more general bound. Let $S _ { n } : = ( \mathcal { Z } \times \mathcal { X } ) ^ { n }$ denote the sample space for $\mathcal { D } _ { n }$

$$
S _ { n } ^ { * } : = \left\{ { \mathcal { D } } _ { n } \in S _ { n } : \prod _ { i = 1 } ^ { n } p ^ { * } ( z _ { i } \mid x _ { i } ) > 0 \right\} .
$$

For any nonnegative measurable function $F$ of $\mathcal { D } _ { n }$ ,

$$
\begin{array} { r l } {  { \mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ \int _ { \mathcal { G } _ { n } ^ { c } } F ( \mathcal { D } _ { n } ) L _ { n } ( p ) d \Pi ( p ) ] = \int _ { \mathcal { G } _ { n } ^ { c } } \mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ F ( \mathcal { D } _ { n } ) L _ { n } ( p ) ] \ d \Pi ( p ) } } \\ & { \quad \quad = \int _ { \mathcal { G } _ { n } ^ { c } } [ \int _ { S _ { n } ^ { * } } F ( \mathcal { D } _ { n } ) \prod _ { i = 1 } ^ { n } \frac { p ( z _ { i } \mid x _ { i } ) } { p ^ { * } ( z _ { i } \mid x _ { i } ) } d P _ { Z , X } ^ { \otimes n } ( \mathcal { D } _ { n } ) ] d \Pi ( p ) } \end{array}\tag{C.36}
$$

(C.37)

$$
= \int _ { \mathcal { G } _ { n } ^ { c } } \left[ \int _ { S _ { n } ^ { * } } F ( \mathcal { D } _ { n } ) \prod _ { i = 1 } ^ { n } p ( z _ { i } \mid \pmb { x } _ { i } ) d ( \mu \otimes P _ { X } ) ^ { \otimes n } ( \mathcal { D } _ { n } ) \right] d \Pi ( p )\tag{C.38}
$$

$$
\leq \int _ { { \mathcal { G } } _ { n } ^ { c } } \left[ \int _ { S _ { n } } F ( { \mathcal { D } } _ { n } ) \prod _ { i = 1 } ^ { n } p ( z _ { i } \mid { \pmb x } _ { i } ) d ( \mu \otimes P _ { { \pmb X } } ) ^ { \otimes n } ( { \mathcal { D } } _ { n } ) \right] d \Pi ( p )\tag{C.39}
$$

$$
= \int _ { \mathcal { G } _ { n } ^ { c } } \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ F ( \mathcal { D } _ { n } ) ] d \Pi ( p ) ,\tag{C.40}
$$

where $P _ { p } ^ { \otimes n }$ is the joint law of n independent observations from $P _ { p }$ , defined by

$$
d P _ { p } ( z , \pmb { x } ) : = p ( z \mid \pmb { x } ) d \mu ( z ) d P _ { X } ( \pmb { x } ) ,
$$

with $\mu$ the dominating measure for $Z$ and $P x$ the fixed true marginal distribution of X.

Taking $F \equiv 1$ , we obtain

$$
\mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } \bigg [ \int _ { \mathcal { G } _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p ) \bigg ] \leq \int _ { \mathcal { G } _ { n } ^ { c } } \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 ] d \Pi ( p ) = \Pi ( \mathcal { G } _ { n } ^ { c } )
$$

Thus, by Markov’s inequality, we have

$$
P _ { Z , X } ^ { \otimes n } \left[ \int _ { \mathcal { G } _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p ) > e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 2 ) n \epsilon _ { n } ^ { 2 } } \right] \leq \frac { C _ { 3 } e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } } { e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 2 ) n \epsilon _ { n } ^ { 2 } } } = C _ { 3 } e ^ { - 2 n \epsilon _ { n } ^ { 2 } } \leq e ^ { - n \epsilon _ { n } ^ { 2 } }
$$

The last inequality holds for all suficiently large $n ,$ since $n \epsilon _ { n } ^ { 2 } \to \infty$

For the first term, we use the standard Hellinger testing construction of Ghosal et al. (2000). We construct a test that distinguishes the true distribution from all candidate distributions indexed by $A _ { n }$ . More specifically, we test

$$
H _ { 0 } : { \mathcal { D } } _ { n } \sim P _ { Z , X } ^ { \otimes n } { \mathrm { a g a i n s t } } \qquad H _ { 1 } : { \mathcal { D } } _ { n } \sim P _ { p } ^ { \otimes n } { \mathrm { f o r ~ s o m e ~ } } p \in { \mathcal { A } } _ { n } .
$$

We seek a test that equals 1 when $H _ { 0 }$ is rejected and 0 otherwise.

By Equation (C.32), there exists a cover of ${ \mathcal { G } } _ { n }$ by $N _ { n }$ Hellinger balls of radius $C _ { 5 } \epsilon _ { n }$ , with centers $q _ { 1 } , \ldots , q _ { N _ { n } }$ , such that

$$
N _ { n } \leq e ^ { C _ { 4 } n \epsilon _ { n } ^ { 2 } } .
$$

Retain only the balls that intersect ${ \mathcal { A } } _ { n } ,$ , and denote their index set by $J _ { n }$ . For each $j \in J _ { n }$ , choose $p _ { j } \in A _ { n }$ in the corresponding ball. The triangle inequality gives

$$
h ( q _ { j } , p ^ { * } ) \geq h ( p _ { j } , p ^ { * } ) - h ( p _ { j } , q _ { j } ) > ( K - C _ { 5 } ) \epsilon _ { n } .
$$

In particular, if $K > 3 C _ { 5 }$ , then

$$
C _ { 5 } \epsilon _ { n } < \frac { 1 } { 2 } h ( q _ { j } , p ^ { * } ) .
$$

The robust Hellinger testing result in Ghosal et al. (2000, Section 7, Equations (7.1)–(7.2)) therefore provides, for each $j \in J _ { n }$ , a binary test $\phi _ { n , j }$ and a universal constant $c > 0$ such that

$$
\mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ \phi _ { n , j } ] \le e ^ { - c n h ^ { 2 } ( q _ { j } , p ^ { * } ) } \le e ^ { - c ( K - C _ { 5 } ) ^ { 2 } n \epsilon _ { n } ^ { 2 } } ,
$$

$$
\operatorname* { s u p } _ { p : h ( p , q _ { j } ) \leq C _ { 5 } \epsilon _ { n } } \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 - \phi _ { n , j } ] \leq e ^ { - c n h ^ { 2 } ( q _ { j } , p ^ { * } ) } \leq e ^ { - c ( K - C _ { 5 } ) ^ { 2 } n \epsilon _ { n } ^ { 2 } } .
$$

Define $\psi _ { n } : = \mathrm { m a x } _ { j \in J _ { n } } \phi _ { n , j }$ , with $\psi _ { n } : = 0$ if $J _ { n }$ is empty. By the union bound,

$$
\begin{array} { r l } & { \mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ \psi _ { n } ] \le \displaystyle \sum _ { j \in J _ { n } } \mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ \phi _ { n , j } ] } \\ & { \qquad \le N _ { n } e ^ { - c ( K - C _ { 5 } ) ^ { 2 } n \epsilon _ { n } ^ { 2 } } } \\ & { \qquad \le e ^ { - \{ c ( K - C _ { 5 } ) ^ { 2 } - C _ { 4 } \} n \epsilon _ { n } ^ { 2 } } . } \end{array}
$$

Moreover, every $p \in { \mathcal { A } } _ { n }$ belongs to at least one retained ball, say the ball indexed by $j$ . Since $\psi _ { n } \geq \phi _ { n , j }$ ，

$$
\mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 - \psi _ { n } ] \le \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 - \phi _ { n , j } ] \le e ^ { - c ( K - C _ { 5 } ) ^ { 2 } n \epsilon _ { n } ^ { 2 } } .
$$

Choose a fixed $K > 3 C _ { 5 }$ suficiently large that

$$
c ( K - C _ { 5 } ) ^ { 2 } \geq \operatorname* { m a x } \{ C _ { 4 } + 1 , \ C _ { 2 } + C _ { 1 } ^ { 2 } + 4 \} .
$$

Then

$$
\mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } [ \psi _ { n } ] \le e ^ { - n \epsilon _ { n } ^ { 2 } } , \qquad \operatorname* { s u p } _ { p \in \mathcal { A } _ { n } } \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 - \psi _ { n } ] \le e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } .\tag{C.41}
$$

We now apply the same change-of-measure argument in Equation (C.40), replacing the integration set ${ \mathcal { G } } _ { n } ^ { c }$ by $A _ { n }$ and taking $F ( { \mathcal { D } } _ { n } ) = 1 - \psi _ { n } ( { \mathcal { D } } _ { n } ) \geq 0$ . Together with Equation (C.41), this yields

$$
\begin{array} { r l } { \mathbb { E } _ { P _ { Z , X } ^ { \otimes n } } \bigg [ ( 1 - \psi _ { n } ) \int _ { A _ { n } } L _ { n } ( p ) d \Pi ( p ) \bigg ] \leq \int _ { A _ { n } } \mathbb { E } _ { P _ { p } ^ { \otimes n } } [ 1 - \psi _ { n } ] d \Pi ( p ) } & { } \\ & { \leq e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } \Pi ( \mathcal { A } _ { n } ) } \\ & { \leq e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } . } \end{array}
$$

Consequently, Markov’s inequality gives

$$
P _ { Z , X } ^ { \otimes n } \bigg \{ ( 1 - \psi _ { n } ) \int _ { \mathcal { A } _ { n } } L _ { n } ( p ) d \Pi ( p ) > e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 2 ) n \epsilon _ { n } ^ { 2 } } \bigg \} \leq e ^ { - 2 n \epsilon _ { n } ^ { 2 } } .\tag{C.42}
$$

Since $\psi _ { n }$ is binary, Equation (C.41) also gives $P _ { Z , X } ^ { \otimes n } ( \psi _ { n } = 1 ) \leq e ^ { - n \epsilon _ { n } ^ { 2 } }$ . Thus, with probability at least $1 - e ^ { - n \epsilon _ { n } ^ { 2 } } - e ^ { - 2 n \epsilon _ { n } ^ { 2 } }$ , we have $\psi _ { n } = 0$ and

$$
\int _ { A _ { n } } L _ { n } ( p ) d \Pi ( p ) \leq e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 2 ) n \epsilon _ { n } ^ { 2 } } .
$$

(iii) Combining the preceding bounds. Recall that

$$
\int _ { \{ p : h ( p , p ^ { * } ) > K \epsilon _ { n } \} } L _ { n } ( p ) d \Pi ( p ) \leq \int _ { \mathcal { A } _ { n } } L _ { n } ( p ) d \Pi ( p ) + \int _ { \mathcal { G } _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p ) .
$$

By the union bound, the denominator lower bound and both numerator upper bounds hold simultaneously with probability at least

$$
1 - \left\{ \frac { C _ { 1 } ^ { 2 } } { n \epsilon _ { n } ^ { 2 } } + 2 e ^ { - n \epsilon _ { n } ^ { 2 } } + e ^ { - 2 n \epsilon _ { n } ^ { 2 } } \right\}
$$

under $P _ { Z , X } ^ { \otimes n }$ , for all suficiently large n. On this event,

$$
\Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \mid \mathcal { D } _ { n } \} = \frac { \int _ { \{ p : h ( p , p ^ { * } ) > K \epsilon _ { n } \} } L _ { n } ( p ) d \Pi ( p ) } { \int L _ { n } ( p ) d \Pi ( p ) }
$$

$$
\begin{array} { l l } {  { \leq \frac { \int _ { A _ { n } } L _ { n } ( p ) d \Pi ( p ) + \int _ { \mathcal { G } _ { n } ^ { c } } L _ { n } ( p ) d \Pi ( p ) } { \int L _ { n } ( p ) d \Pi ( p ) } } } \\ & { \leq \frac { 2 e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 2 ) n \epsilon _ { n } ^ { 2 } } } { e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 1 ) n \epsilon _ { n } ^ { 2 } } } } \\ & { = 2 e ^ { - n \epsilon _ { n } ^ { 2 } } . } \end{array}
$$

Since $e ^ { - 2 n \epsilon _ { n } ^ { 2 } } \leq e ^ { - n \epsilon _ { n } ^ { 2 } }$ , we conclude that

$$
P _ { Z , X } ^ { \otimes n } \Big \{ \Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \ | \ \mathcal { D } _ { n } \} > 2 e ^ { - n \epsilon _ { n } ^ { 2 } } \Big \} \leq \frac { C _ { 1 } ^ { 2 } } { n \epsilon _ { n } ^ { 2 } } + 3 e ^ { - n \epsilon _ { n } ^ { 2 } } .
$$

Proof of Equation C.34 Since $h ^ { 2 } \leq 2$ , we have the deterministic posterior bound

$$
\mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } ] \le K ^ { 2 } \epsilon _ { n } ^ { 2 } + 2 \Pi \{ h ( p , p ^ { * } ) > K \epsilon _ { n } \mid \mathcal { D } _ { n } \} .
$$

On the high-probability event established above,

$$
\mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } ] \le K ^ { 2 } \epsilon _ { n } ^ { 2 } + 4 e ^ { - n \epsilon _ { n } ^ { 2 } } .
$$

The additional condition log $\cdot ( 1 / \epsilon _ { n } ) = o ( n \epsilon _ { n } ^ { 2 } )$ implies

$$
\frac { 2 \log ( 1 / \epsilon _ { n } ) } { n \epsilon _ { n } ^ { 2 } } \longrightarrow 0 .
$$

Thus,

$$
- n \epsilon _ { n } ^ { 2 } + 2 \log ( 1 / \epsilon _ { n } ) = - n \epsilon _ { n } ^ { 2 } \left( 1 - \frac { 2 \log ( 1 / \epsilon _ { n } ) } { n \epsilon _ { n } ^ { 2 } } \right) \longrightarrow - \infty ,
$$

Therefore, we get

$$
\frac { e ^ { - n \epsilon _ { n } ^ { 2 } } } { \epsilon _ { n } ^ { 2 } } = \exp \{ - n \epsilon _ { n } ^ { 2 } + 2 \log ( 1 / \epsilon _ { n } ) \} \longrightarrow 0 .
$$

Thus, for all suficiently large n, we have

$$
4 e ^ { - n \epsilon _ { n } ^ { 2 } } \leq \epsilon _ { n } ^ { 2 } .
$$

Hence, we can build

$$
P _ { Z , X } ^ { \otimes n } \Big \{ \mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } ] > ( K ^ { 2 } + 1 ) \epsilon _ { n } ^ { 2 } \Big \} \leq \frac { C _ { 1 } ^ { 2 } } { n \epsilon _ { n } ^ { 2 } } + 3 e ^ { - n \epsilon _ { n } ^ { 2 } } \longrightarrow 0 .
$$

Consequently,

$$
\mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) ,
$$

where $O _ { P }$ is with respect to the true sampling law $P _ { Z , X } ^ { \otimes n }$

Proof of Equation C.35

$$
\mathbb { E } _ { \Pi } \big [ \big \lVert \mathrm { L U C I D } [ p ] - \mathrm { L U C I D } [ p ^ { * } ] \big \rVert _ { 2 } ^ { 2 } \mid \mathcal { D } _ { n } \big ] \le 4 C _ { \ell } ^ { 2 } \mathbb { E } _ { \Pi } \big [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } \big ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) .
$$

Note: mere convergence of a posterior tail probability to zero, without a quantitative bound such as Equation (C.33), is insuficient to infer the same rate for a posterior mean.

## C.6 DR-BART Specialization

We apply Conditions F and P1–P6 of Orlandi et al. (2021, Section 4) to response Z and conditioning covariates X. These conditions are adopted in full. They impose smoothness and sparsity of the log conditional density, restrictions on the true conditional quantile function, and specified support-selection, tree, leaf, scale, and split-point priors, including an approximation condition. They are stronger than choosing the default tuning parameters of a density estimator.

Corollary C.7 (DR-BART contraction and posterior moments). Under these conditions, with k conditioning coordinates, let

$$
\epsilon _ { n } = \left( { \frac { n } { \log n } } \right) ^ { - { \frac { \alpha } { \alpha + 1 } } { \frac { \beta } { 2 \beta + d _ { 0 } } } } + \sqrt { \frac { d _ { 0 } \log ( k + 1 ) } { n } } , \qquad \beta = \operatorname* { m i n } \{ \alpha , 1 \} .\tag{C.43}
$$

Here $0 < \alpha \leq 2$ is the smoothness index and $d _ { 0 }$ counts active coordinates including the response. The density posterior contracts at this rate in integrated Hellinger distance. Its proof conditions also yield Equation (C.34). If Equation (C.21) holds, the LUCID posterior satisfies both Equations (C.28) and (C.35).

Proof. The density contraction statement follows from Orlandi et al. (2021, Theorem 2). We verify that the bounds in its proof satisfy the assumptions of Lemma C.6.

First, condition G1, verified in Section 8.3 of the cited paper, gives constants $C _ { 1 } , C _ { 2 } > 0$ such that

$$
\Pi \{ p : \mathrm { K L } ( p ^ { * } \| p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 } , \ V ( p ^ { * } , p ) \leq C _ { 1 } ^ { 2 } \epsilon _ { n } ^ { 2 } \} \geq e ^ { - C _ { 2 } n \epsilon _ { n } ^ { 2 } } .
$$

This is Equation (C.30).

Second, the proof of G2 constructs sieves $\mathcal { G } _ { n , \kappa }$ indexed by a suficiently large fixed constant κ and establishes

$$
\Pi ( \mathcal { G } _ { n , \kappa } ^ { c } ) \leq \exp ( - C \varsigma \kappa n \epsilon _ { n } ^ { 2 } ) ,
$$

where $C _ { \mathcal { G } } > 0$ depends only on the prior. Choose

$$
\kappa > \frac { C _ { 2 } + C _ { 1 } ^ { 2 } + 4 } { C _ { \mathscr { G } } }
$$

and fix $\mathcal { G } _ { n } = \mathcal { G } _ { n , \kappa }$ . Then

$$
\Pi ( { \mathcal G } _ { n } ^ { c } ) \le e ^ { - ( C _ { 2 } + C _ { 1 } ^ { 2 } + 4 ) n \epsilon _ { n } ^ { 2 } } ,
$$

which verifies Equation (C.31) with $C _ { 3 } = 1$ . This choice changes only the sieve used in the proof, not the prior.

Third, the covering construction establishing G3 remains valid for this fixed κ. It gives finite constants $C _ { 4 } , C _ { 5 } > 0$ , which may depend on κ, such that

$$
\log \mathcal { N } ( C _ { 5 } \epsilon _ { n } , \mathcal { G } _ { n } , h ) \leq C _ { 4 } n \epsilon _ { n } ^ { 2 } .
$$

This is Equation (C.32).

It remains to check the growth conditions. Assume $\epsilon _ { n }  0$ , as stipulated in the corollary, and write

$$
a = { \frac { \alpha } { \alpha + 1 } } { \frac { \beta } { 2 \beta + d _ { 0 } } } , \qquad \beta = \mathrm { m i n } \{ \alpha , 1 \} .
$$

Since $0 < \alpha \leq 2$ and $d _ { 0 } \geq 1$ 2

$$
0 < a \leq \frac { 2 } { 3 } \cdot \frac { 1 } { 3 } = \frac { 2 } { 9 } .
$$

For all suficiently large $n ,$

$$
\epsilon _ { n } \geq ( n / \log n ) ^ { - a } \geq ( n / \log n ) ^ { - 2 / 9 } ,
$$

and hence

$$
n \epsilon _ { n } ^ { 2 } \ge n ^ { 5 / 9 } ( \log n ) ^ { 4 / 9 } \longrightarrow \infty .
$$

Moreover, once $\epsilon _ { n } < 1$ 2

$$
0 \leq \log ( 1 / \epsilon _ { n } ) \leq { \frac { 2 } { 9 } } \log ( n / \log n ) \leq { \frac { 2 } { 9 } } \log n = o ( n \epsilon _ { n } ^ { 2 } ) .
$$

Lemma C.6 therefore yields Equation (C.34). Under Equation (C.21), it also gives Equation (C.35). Finally, the density contraction result, together with Corollary C.5 and Equation (C.27), establishes the stated contraction of the LUCID posterior. □

## C.7 Convergence Rate of Posterior Averaging

The linear dependence of LUCID on the replacement distribution allows posterior averaging without explicitly constructing an average density. For analysis, define

$$
\overline { { p } } ( z ^ { \prime } \mid \pmb { x } ) : = \mathbb { E } _ { \Pi } [ p ( z ^ { \prime } \mid \pmb { x } ) \mid \mathcal { D } _ { n } ] .\tag{C.44}
$$

This is a conceptual replacement distribution used for the analysis; it need not be fitted or evaluated explicitly.

Proposition C.8 (Posterior averaging and its rate). Let ${ \overline { { p } } } ( z \mid \mathbf { x } ) : = \mathbb { E } _ { \Pi } [ p ( z \mid \mathbf { x } ) \mid { \mathcal { D } } _ { n } ]$ denote the posterior mean conditional density.

If Equation (C.34) holds, then

$$
h ( \overline { { p } } , p ^ { * } ) = O _ { P } ( \epsilon _ { n } ) .\tag{C.45}
$$

Under Equation (C.21),

$$
\mathrm { L U C I D } [ \overline { { { p } } } ] ( \pmb { w } ) = \mathbb { E } _ { \Pi } [ \mathrm { L U C I D } [ p ] ( \pmb { w } ) \ | \ \mathcal { D } _ { n } ] .\tag{C.46}
$$

If Equation (C.35) also holds, then

$$
\| \operatorname { L U C I D } [ \overline { { { p } } } ] - \operatorname { L U C I D } [ p ^ { * } ] \| _ { 2 } = O _ { P } ( \epsilon _ { n } ) .\tag{C.47}
$$

Proof. Tonelli’s theorem ensures that $\overline { { p } } ( \cdot \mid x )$ integrates to one. Convexity of squared Hellinger distance in its first argument gives

$$
\begin{array} { r } { h ^ { 2 } ( \overline { { p } } , p ^ { * } ) \leq \mathbb { E } _ { \Pi } [ h ^ { 2 } ( p , p ^ { * } ) \mid \mathcal { D } _ { n } ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) , } \end{array}
$$

where the last equality follows from Equation (C.34). Taking square roots proves Equation (C.45).

Under Equation (C.21), Fubini’s theorem and Equation (C.2) yield

$$
\begin{array} { r l } & { \displaystyle \mathrm { L U C I D } [ \overline { { p } } ] ( \boldsymbol { w } ) = \int \ell \big ( g ( z ^ { \prime } , \boldsymbol { x } ) , g ( z , \pmb { x } ) \big ) \overline { { p } } \big ( z ^ { \prime } \mid \boldsymbol { x } \big ) d z ^ { \prime } } \\ & { \qquad = \mathbb { E } _ { \Pi } [ \int \ell \big ( g ( z ^ { \prime } , \boldsymbol { x } ) , g ( z , \pmb { x } ) \big ) p ( z ^ { \prime } \mid \boldsymbol { x } \big ) d z ^ { \prime } | \mathcal { D } _ { n } ] } \\ & { \qquad = \mathbb { E } _ { \Pi } [ \mathrm { L U C I D } [ p ] ( \boldsymbol { w } ) \mid \mathcal { D } _ { n } ] , } \end{array}
$$

proving Equation (C.46). Jensen’s inequality in $L ^ { 2 } ( P _ { Z , X } )$ then gives

$$
\begin{array} { r } { \| \operatorname { L U C I D } [ \overline { { p } } ] - \operatorname { L U C I D } [ p ^ { * } ] \| _ { 2 } ^ { 2 } \leq \operatorname { \mathbb { E } } _ { \Pi } [ \| \operatorname { L U C I D } [ p ] - \operatorname { L U C I D } [ p ^ { * } ] \| _ { 2 } ^ { 2 } \ | \ \mathcal { D } _ { n } ] = O _ { P } ( \epsilon _ { n } ^ { 2 } ) , } \end{array}
$$

where the last equality follows from Equation (C.35). Taking square roots proves Equation (C.47).

These results establish the convergence rate of the exact posterior mean. Approximating it with finitely many retained posterior draws and conditional replacements introduces additional Monte Carlo error, which must be controlled separately.

## D Implementation Details and Experimental Setup

## D.1 Implementation Details

This section describes conditional replacement sampling, posterior summaries, and the experimental settings used to evaluate LUCID.

Pseudocode. We first present the pseudocode for LUCID estimation without batching in Algorithm 1.

Algorithm 1 Estimating LUCID by conditional sampling   
Require: Fixed model ${ g } ;$ evaluation observations $\{ { \pmb w } _ { i } = ( z _ { i } , { \pmb x } _ { i } ) \} _ { i = 1 } ^ { N } ;$ conditional sampler   
$\widehat { P } _ { Z | X = { \pmb x } _ { i } } ( { \pmb x } \mid { \pmb x } _ { i } )$ ; loss $\ell ;$ number of replacements M   
1: for $i = 1 , \ldots , N$ do   
2: Draw $z _ { i , 1 } ^ { \prime } , \ldots , z _ { i , M } ^ { \prime }$ independently from ${ \widehat { P } } _ { Z | X = x _ { i } } ( \cdot \mid x _ { i } )$   
3: Form perturbed observations ${ \pmb w } _ { i , m } ^ { \prime } = ( z _ { i , m } ^ { \prime } , { \pmb x } _ { i } ) , m = 1 , \ldots , M$   
Compute $\widehat { \mathrm { L U C I D } _ { i } }$ using Equation (8)   
5: end for   
6: return $\{ \widehat { \mathrm { L U C I D } _ { i } } \} _ { i = 1 } ^ { N }$

Direct conditional sampling. For a continuous feature, a posterior draw b of DR-BART has the latent-variable representation

$$
Z ^ { \prime } = f ^ { ( b ) } ( { \pmb x } , U ) + \sigma ^ { ( b ) } ( { \pmb x } , U ) \varepsilon , \qquad U \sim \mathrm { U n i f } ( 0 , 1 ) , \quad \varepsilon \sim \mathcal { N } ( 0 , 1 ) ,\tag{D.1}
$$

where U and $\varepsilon$ are independent, and $f ^ { ( b ) }$ and $\boldsymbol { \sigma } ^ { ( b ) }$ are the location and standard-deviation functions from that posterior draw. Here $U$ is a continuous latent variable; b indexes a draw from the posterior over the model parameters.

We use this representation to generate conditional replacements directly, without first evaluating the fitted conditional density on a grid. Conditional on a posterior draw, this procedure samples exactly from the corresponding fitted conditional distribution, avoiding grid discretization and numerical inversion of the cumulative distribution function. It also eliminates the need to construct and store a density-evaluation matrix for sampling. More specifically, first draw $U _ { : }$ , evaluate the location and standard deviation at $( { \pmb x } , U )$ , and then draw $Z ^ { \prime }$ from the resulting normal distribution. Algorithm 2 gives the sampling procedure.

Algorithm 2 Direct posterior-predictive sampling with DR-BART   
Require: Conditioning vector ${ \pmb x } ;$ retained posterior functions $\{ ( f ^ { ( b ) } , \sigma ^ { ( b ) } ) \} _ { b = 1 } ^ { B } ;$ number of replace  
ments M   
1: for $m = 1 , \ldots , M$ do   
2: Draw $b _ { m } \sim \operatorname { U n i f } \{ 1 , \dots , B \}$   
3: Draw $u _ { m } \sim \mathrm { U n i f } ( 0 , 1 )$   
4: Set $\mu _ { m } \gets f ^ { ( b _ { m } ) } ( { \pmb x } , u _ { m } )$ and $s _ { m } \gets \sigma ^ { ( b _ { m } ) } ( { \pmb x } , u _ { m } )$   
5: Draw $Z _ { m } ^ { \prime } \sim \mathcal N ( \mu _ { m } , s _ { m } ^ { 2 } )$   
6: end for   
7: return $Z _ { 1 } ^ { \prime } , \ldots , Z _ { M } ^ { \prime }$

Binary and categorical features. For binary and categorical features, the conditional distribution is fully determined by the conditional category probabilities. We therefore use classification BART (Chipman et al., 2010; Quiroga et al., 2022) to estimate these probabilities directly. For a binary feature, we estimate $P ( Z = 1 \mid X = { \pmb x } )$ and draw replacements from the corresponding Bernoulli distribution. For a feature with multiple categories, we estimate a vector of conditional category probabilities that sum to one and draw replacements from the resulting categorical distribution. Each posterior draw of these probabilities specifies a complete conditional distribution, which can be used in the same LUCID estimation and posterior summarization procedures described below.

Posterior point estimates and credible intervals. For each retained posterior draw $b =$ $1 , \ldots , B _ $ , generate M replacements using Algorithm 2, however, keeping b fixed across all replacements, and compute

$$
\widehat { \mathrm { L U C I D } } ^ { ( b ) } ( \pmb { w } ) : = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \ell \Big ( g ( Z _ { m } ^ { \prime ( b ) } , \pmb { x } ) , g ( z , \pmb { x } ) \Big ) .\tag{D.2}
$$

The posterior mean is approximated by

$$
\widehat { \mathrm { L U C I D } } ^ { \mathrm { p o s t } } ( \boldsymbol { w } ) : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \widehat { \mathrm { L U C I D } } ^ { ( b ) } ( \boldsymbol { w } ) ,\tag{D.3}
$$

When only a posterior point estimate is required, we can alternatively sample from the posterior predictive distribution. For each replacement, we select an index b uniformly from $\{ 1 , \ldots , B \}$ and generate a conditional replacement using Equation (D.1). Because LUCID is linear in the conditional replacement distribution, averaging the resulting loss contrasts estimates the same posterior mean, conditional on the retained draws, without requiring a separate LUCID estimate for every draw.

To construct posterior credible intervals, we instead compute a separate LUCID estimate for each posterior draw using Equation (D.2). The empirical quantiles of these estimates $\widehat { \{ \mathrm { L U C I D } }  ^ { ( b ) } ( \pmb { w } ) \} _ { b = 1 } ^ { B }$ describe posterior uncertainty in LUCID induced by the conditional density estimator. Writing $\widehat { Q } _ { w } ( \gamma )$ for the empirical γ-quantile of $\widehat { \{ \mathrm { L U C I D } }  ^ { ( b ) } ( \pmb { w } ) \} _ { b = 1 } ^ { B }$ , an approximate $( 1 - \gamma )$ posterior credible interval is

$$
\left[ \widehat { Q } _ { w } ( \gamma / 2 ) , \widehat { Q } _ { w } ( 1 - \gamma / 2 ) \right] .\tag{D.4}
$$

These empirical quantiles approximate posterior uncertainty in LUCID when the Monte Carlo error within each posterior draw is negligible; otherwise, they also reflect finite-M sampling variability.

Sampling feature groups. Let $\pmb { Z } = ( Z _ { 1 } , \ldots , Z _ { d } )$ denote a feature group with a fixed ordering, and let X denote the remaining covariates. We construct a joint conditional sampler through the factorization

$$
p ( \boldsymbol { z } \mid \boldsymbol { x } ) = \prod _ { r = 1 } ^ { d } p ( z _ { r } \mid \boldsymbol { x } , z _ { 1 } , \ldots , z _ { r - 1 } ) .\tag{D.5}
$$

We fit each factor using the remaining covariates and the preceding group members as predictors.   
For continuous group members, we fit these factors with DR-BART.

To generate one replacement vector, first sample $Z _ { 1 } ^ { \prime }$ given x, then sample $Z _ { 2 } ^ { \prime }$ given $( \pmb { x } , Z _ { 1 } ^ { \prime } )$ , and continue through the group. Each step conditions on the values already sampled, allowing the sampler to represent dependence within the group. Substitute the resulting joint draw $\boldsymbol { Z ^ { \prime } }$ into Equation (7), and average the resulting losses as in Equation (8).

To quantify posterior uncertainty in group LUCID, we additionally draw the factor models from their joint posterior. With independent priors for the factor models, a posterior draw of the joint sampler is formed by selecting a posterior draw from each factor independently. We then hold these selected functions fixed across the M joint replacements used to compute one group LUCID draw. Repeating this procedure yields posterior point estimates and credible intervals for LUCID(w) for the feature group, in the same way as for individual features.

Feature bounds. The experimental implementation optionally restricts continuous features to bounds $[ L _ { j } , H _ { j } ]$ . These bounds may be specified by the user; when they are not provided, we set them to the observed dataset range padded by 10% on each side. Candidates outside this interval are rejected while holding the posterior index fixed and redrawing only $U _ { i m }$ and $\varepsilon _ { i m }$ . Thus, each posterior-specific replacement follows the corresponding conditional distribution truncated to $[ L _ { j } , H _ { j } ]$

After at most 10,000 rejection rounds, any remaining replacements are generated using a bounded-mixture fallback based on latent-variable tree cutpoints, which samples from the fitted conditional distribution further restricted to $X _ { j } \in [ L _ { j } , H _ { j } ]$

## D.2 Additional Implementation Details for Direct Conditional Sampling and Batched LUCID Computation

This section provides implementation-level details for our LUCID computation, including direct conditional sampling, batching, and the corresponding time and space complexity analysis.

Notation and computational scope. Since the analysis is carried out feature by feature and we aim to make the full update procedure explicit, we introduce a separate notation system with additional subscripts to indicate precisely which observation, feature, posterior draw, and replacement each quantity corresponds to.

Consider a fixed feature j and N evaluation observations $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { N }$ , with $\pmb { x } _ { i } \in \mathbb { R } ^ { p }$ . Here, $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ denotes the full observation previously written as ${ \pmb w } _ { i }$ . The predictive model $g$ is held fixed throughout the calculation. This notation system will continue to be used in Section E and Section F.6.

Let $\{ ( f _ { j } ^ { ( b ) } , \sigma _ { j } ^ { ( b ) } ) \} _ { b = 1 } ^ { B }$ denote the retained conditional-density posterior functions. We use M replacements per observation, an evaluation batch size $A _ { \mathrm { b a t c h } }$ , and a replacement-prediction batch size $C _ { \mathrm { b a t c h } }$

The following complexity statements concern evaluation after fitting. They exclude the input dataset and the storage of the fitted predictive model and posterior tree ensembles. In particular, direct sampling does not eliminate the storage required for the $B$ retained posterior functions themselves. Let $\tau _ { f }$ denote the cost of evaluating both $f _ { j } ^ { ( b ) }$ and $\sigma _ { j } ^ { ( b ) }$ at one conditioning– latent pair.

Batched implementation. In our implementation, feature-specific calculations are submitted as independent parallel jobs. Within each job, conditional replacements are generated in evaluation batches of size $A _ { \mathrm { b a t c h } }$ . The implementation then processes each observation separately, evaluating its perturbed inputs in replacement batches of size $C _ { \mathrm { b a t c h } }$ and accumulating their losses. Thus, sampling is batched across observations, whereas predictive-model evaluation is batched across replacements of a single observation.

An independent job executes Algorithm 3 for each feature. These jobs can run concurrently subject to available resources; each job writes its feature-specific result separately.

Algorithm 3 Batched prediction-reference LUCID estimation per feature   
Require: Fixed model ${ g } ;$ evaluation observations $\{ \pmb { x } _ { i } \} _ { i = 1 } ^ { N } \subset \mathbb { R } ^ { p } ;$ feature $j ;$ fitted direct conditional   
sampler for feature $j ;$ number of replacements $M ;$ evaluation batch size $A _ { \mathrm { b a t c h } } ;$ replacement  
prediction batch size $C _ { \mathrm { b a t c h } }$   
1: Evaluate the fixed model on all original observations: $g ( \pmb { x } _ { i } ) , i = 1 , . . . , N$   
2: for each consecutive observation batch ${ \mathcal { T } } \subseteq \{ 1 , \ldots , N \}$ of size at most $A _ { \mathrm { b a t c h } }$ do   
3: Use direct posterior-predictive (in Algorithm 2) sampling to generate $\pmb { R } \in \mathbb { R } ^ { | \mathcal { T } | \times M }$ , where   
$R _ { i m }$ replaces feature $j$ conditional on $\boldsymbol { x } _ { i , - j }$   
4: for each observation $i \in \mathcal { Z }$ do   
5: Set accumulated loss $T _ { i } \gets 0$   
6: for each consecutive replacement batch ${ \mathcal { C } } \subseteq \{ 1 , \ldots , M \}$ of size at most $C _ { \mathrm { b a t c h } }$ do   
7: Form $\pmb { x } _ { i m } ^ { \prime }$ by replacing coordinate $j$ of $\mathbf { \mathcal { x } } _ { i }$ with $R _ { i m } .$ , for $m \in { \mathcal { C } }$   
8: Evaluate $\{ g ( \pmb { x } _ { i m } ^ { \prime } ) \} _ { m \in \mathcal { C } }$ in one model call   
9: Accumulate $\begin{array} { r } { T _ { i }  T _ { i } + \sum _ { m \in \mathcal { C } } \ell ( g ( \pmb { x } _ { i m } ^ { \prime } ) , g ( \pmb { x } _ { i } ) ) } \end{array}$   
10: end for   
11: Compute $\widehat { L U C I D } ( \pmb { x } _ { i } ) \gets T _ { i } / M$   
12: end for   
13: end for   
14: Save and return $\widehat { L U C I D }$ for feature $j$

Direct sampling and posterior conditioning. For a continuous feature, the unbounded conditional sampler uses

$$
\begin{array} { r l } & { U _ { i m } \sim \mathrm { U n i f } ( 0 , 1 ) , } \\ & { \varepsilon _ { i m } \sim \mathcal { N } ( 0 , 1 ) , } \\ & { X _ { i , j , m } ^ { \prime } = f _ { j } ^ { ( b _ { i m } ) } ( \pmb { x } _ { i , - j } , U _ { i m } ) + \sigma _ { j } ^ { ( b _ { i m } ) } ( \pmb { x } _ { i , - j } , U _ { i m } ) \varepsilon _ { i m } . } \end{array}\tag{D.6}
$$

For a posterior-predictive point estimate, the implementation draws $b _ { i m } \sim \mathrm { U n i f } \{ 1 , \dots , B \}$ independently for every observation–replacement pair. For a posterior-specific LUCID estimate, it instead fixes $b _ { i m } = b$ for all observations and all M replacements.

The conditional standard deviation follows the fitted variance mode:

$$
\sigma _ { j } ^ { ( b ) } ( \pmb { x } _ { - j } , u ) = \left\{ \begin{array} { l l } { \left[ \phi _ { j , b } ^ { * } \tau _ { j } ^ { ( b ) } ( \pmb { x } _ { - j } , u ) \right] ^ { - 1 / 2 } , } & { \mathrm { u x } , } \\ { \left[ \phi _ { j , b } ^ { * } \tau _ { j } ^ { ( b ) } ( \pmb { x } _ { - j } ) \right] ^ { - 1 / 2 } , } & { \mathrm { x } , } \\ { \sigma _ { j , b } , } & { \mathrm { c o n s t . } } \end{array} \right.\tag{D.7}
$$

Here, $\tau _ { j } ^ { ( b ) }$ is the fitted precision-tree function, and ${ \phi } _ { j , b } ^ { * } > 0$ is the global precision scale retained in posterior draw b for the conditional model of feature $j$ . Their product gives the conditional precision. For the constant-variance model, $\sigma _ { j , b }$ is the retained posterior standard deviation. This is the parameterization used by the default DR-BART implementation; we make the conversion from the retained precision-tree representation to the corresponding conditional variance explicit only because the latter is not directly exposed by the default API.

Complexity of direct sampling. The following bounds concern unbounded fixed-posterior sampling and exclude fitted-model storage, model loading, and subsequent predictive-model evaluation.

Proposition 3 (Complexity of direct sampling per feature). For one feature and one fixed posterior draw, generating M replacements for each of N evaluation observations takes

$$
T _ { \mathrm { s a m p l e } } = O ( N M ( p + \tau _ { f } ) )
$$

time and

$$
S _ { \mathrm { s a m p l e } } = O ( A _ { b a t c h } M p )
$$

peak sampling storage.

Proof. For a batch of $a \le A _ { \mathrm { b a t c h } }$ observations (the final batch may contain fewer than $A _ { \mathrm { b a t c h } }$ observations), the implementation materializes an $a M \times ( p - 1 )$ conditioning array and evaluates the conditional mean and standard deviation once per replacement. Thus, processing all NM replacements takes $O ( N M ( p + \tau _ { f } ) )$ time, while the current batch requires $O ( A _ { \mathrm { b a t c h } } M p )$ storage.

Materializing each batch enables vectorized tree evaluation and reduces per-call overhead. Using $A _ { \mathrm { b a t c h } } \ll N$ reduces peak sampling storage from $O ( N M p )$ to $O ( A _ { \mathrm { b a t c h } } M p )$ . Subsequent predictivemodel evaluation uses replacement batches of size $C _ { \mathrm { b a t c h } }$ , requiring an additional $O ( C _ { \mathrm { b a t c h } } p )$ perturbed-input workspace; it does not reduce the replacements already generated by the sampler. Feature-specific calculations can be distributed across independent parallel jobs, subject to available computational resources. The storage bound above is per job; concurrent jobs increase aggregate memory usage.

Note: Direct sampling does not require storing a conditional-density evaluation matrix. By generating one replacement per observation at a time and immediately accumulating its loss contribution, the sampling workspace can, in principle, be reduced to $O ( A _ { \mathrm { b a t c h } } p )$ while still using M replacements per observation. The current implementation instead materializes all M replacements within each evaluation batch, using $O ( A _ { \mathrm { b a t c h } } M p )$ sampling workspace to enable vectorized tree evaluation and reduce per-call overhead. This creates a trade-of between memory usage and computational eficiency when processing very large datasets.

## D.3 Experimental Setup

Real-world datasets. Our primary real-data case study uses the 2011 hourly subset of the UCI Bike Sharing dataset (Fanaee-T and Gama, 2014), containing 8,645 observations and distributed as Bikeshare in the ISLR2 R package.<sup>1</sup> The UCI dataset is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.<sup>2</sup>

To further evaluate scalability and robustness on higher-dimensional tabular data, we additionally use four datasets from the OpenXAI (Agarwal et al., 2022) benchmark: Framingham Heart Study (Heart), German Credit, HELOC, and Give Me Some Credit (GMSC). We obtain the benchmark versions of these datasets from the OpenXAI Harvard Dataverse archive, where the deposited dataset is listed under the CC0 1.0 Universal Public Domain Dedication.<sup>3</sup>

Experiment Platform. Experiments were submitted as Slurm jobs on a shared high-performance computing cluster. The nodes used included Intel Xeon Gold 6226 and AMD EPYC 9554 processors, with node memory ranging from 768 GB to approximately 1 TB. Standard feature-specific worker tasks requested one CPU core and 32 GB RAM, with no GPU resources requested. Independent feature and fold tasks were scheduled concurrently. The software environment used Python 3.14.4 and R 4.5.2.

Software Packages. We used the following upstream packages together with the feature-specific sampling and importance calculations implemented in our repository.

DR-BART: Continuous conditional models were fitted using the R package drbart (development version 0.0.0.9000), available at https://github.com/vittorioorlandi/drbart. Direct latent-variable sampling, batching, and posterior LUCID calculations were implemented in our repository. Binary and categorical conditional models used pymc-bart 0.13.1 and pymc 6.3.2, available at https://github.com/pymc-devs/pymc-bart and https://github.com/pymc-devs/pymc, respectively.

ARF: We used the R package arf 0.2.5, available at https://github.com/bips-hb/arf, for adversarial random-forest fitting, density estimation, and conditional generation. Its random-forest backend was ranger 0.18.0.

cs-PFI: Our conditional-subgroup sampler used trtf 0.4-3, mlt 1.8-2, and partykit 1.3- 0. These packages are available at https://CRAN.R-project.org/package=trtf, https://CRAN. R-project.org/package=mlt, and https://CRAN.R-project.org/package=partykit. The conditional subgroup method’s original code is available at https://github.com/christophM/paper\_conditional\_ subgroups. Our repository implements the sampler and its integration with the common LUCID evaluation routine.

LIME: We used lime 0.2.0.1 and its LimeTabularExplainer, available at https://github. com/marcotcr/lime.

SHAP: We used shap 0.52.0 through shap.Explainer, available at https://github.com/ shap/shap.

Conditional SHAP: We used pyshapr 0.6.0 with the R package shapr 1.1.0, available at https://github.com/NorskRegnesentral/shapr. Conditional expectations were estimated using its arf approach. In some experimental figures and tables, conditional SHAP is labeled “SHAPr”.

Other baselines and predictive models: CMR, GCMR, CLIQUE, ICI, local LOCO, and the analytical LUCID/LOCO benchmarks were implemented in our repository. CMR, GCMR, and local LOCO used xgboost 3.4.1, available at https://github.com/dmlc/xgboost. The same package supplied predictive models for the real-data experiments; synthetic experiments used the known DGP prediction functions. The CLIQUE baseline used our quantile-replacement implementation, rather than the upstream CLIQUE R package.

Numerical and data-processing tools: We used NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, and scikit-learn 1.9.1. Figures were generated using Matplotlib 3.11.2. The Python interface to the R SHAPr implementation used rpy2 3.6.8, and resource measurements used psutil 7.2.2.

Parameters. We did not perform hyperparameter tuning. Unless specified below, package hyperparameters were left at their defaults.

For continuous-feature DR-BART models, we used 2,000 burn-in iterations and retained 2,000 posterior draws without thinning (nburn=2,000, nsim=2,000, nthin=1). The conditional mean and precision ensembles contained 200 and 100 trees, respectively (m\_mean=200, m\_var=100). We used variance mode ux, allowing the conditional variance to depend on both the conditioning features and the latent variable. Group conditional models used the same settings.

For binary and categorical features, the conditional BART models used 50 trees, 1,000 tuning iterations, and 1,000 posterior draws from one chain.

Point-estimation experiments used M = 1, 000 replacements per observation, with evaluation and replacement-prediction batch sizes $A _ { \mathrm { b a t c h } } = C _ { \mathrm { b a t c h } } = 1 0 0$ . The default experiment seed was 2026.

For real-data experiments, we used five-fold cross-fitting. For each held-out fold, we trained an

XGBoost predictive model g on the other four folds using default hyperparameters, without tuning, and held it fixed when computing LUCID on the held-out observations. This yielded five separately fitted predictive models, each shared across the compared methods within its corresponding fold.

## E Additional DGP Results

In this section, we provide the full experimental results, analytical calculations, and additional analyses for the synthetic DGPs considered in the main text. Because this section focuses on concrete simulation settings, we slightly simplify the notation used in the main paper. We use $X _ { j }$ to denote the feature of interest and $X _ { - j }$ to denote the remaining features, with lowercase $x _ { j }$ and $\mathbf { \delta } _ { \pmb { x } _ { - j } }$ denoting their observed values. We use n (from now on) for the number of observations and $p$ for the number of features. For the DGPs in the main paper, we set $n = 5 0 0 0$ . Unless otherwise stated, all LUCID quantities are defined with respect to the fixed prediction function used in each DGP.

## E.1 Synthetic Data-Generating Processes

We consider five synthetic DGPs in the main paper designed to isolate diferent aspects of local conditional variable importance. Again, for each DGP, Table 5 reports the data-generating process, the analytical LUCID for each relevant feature, and the corresponding global conditional permutation feature importance (CPFI).

Table 5: Synthetic DGPs and analytical LUCID and analytical CPFI under squared loss.
<table><tr><td>DGP</td><td>Data-generating process</td><td>Analytical LUCID</td><td>CPFI</td></tr><tr><td>Discontinuity</td><td> $X _ { 1 } , X _ { 2 } , X _ { 3 } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 )$   $g ( \pmb { x } ) = x _ { 1 } \mathbf { 1 } _ { x _ { 3 } > 0 } + x _ { 2 } \mathbf { 1 } _ { x _ { 3 } \leq 0 } .$ </td><td> $I _ { 1 } = ( x _ { 1 } ^ { 2 } + 1 ) \mathbf { 1 } _ { x _ { 3 } > 0 } ;$   $I _ { 2 } = ( x _ { 2 } ^ { 2 } + 1 ) \mathbf { 1 } _ { x _ { 3 } \leq 0 } ;$   $\begin{array} { r } { I _ { 3 } = \frac 1 2 ( x _ { 1 } - x _ { 2 } ) ^ { 2 } . } \end{array}$ </td><td> $\mathrm { { C P F I } _ { 1 } = 1 ; }$   $\mathrm { { C P F I } _ { 2 } = 1 ; }$   $\mathrm { { C P F I } _ { 3 } = 1 . }$ </td></tr><tr><td>Correlated</td><td> $( X _ { 1 } , X _ { 2 } ) ^ { \top } \sim { \mathcal { N } } _ { 2 } ( 0 , \Sigma _ { \rho } ) ,$   $\Sigma _ { \rho , 1 1 } = \Sigma _ { \rho , 2 2 } = 1 , \Sigma _ { \rho , 1 2 } = \rho ,$   $\rho = 0 . 9 ; g ( \pmb { x } ) = x _ { 1 } .$ </td><td> $I _ { 1 } = 1 - \rho ^ { 2 } + ( x _ { 1 } - \rho x _ { 2 } ) ^ { 2 } ;$   $I _ { 2 } = 0 .$ </td><td> $\mathrm { { C P F I _ { 1 } } = }$   $2 ( 1 - \rho ^ { 2 } ) ;$   $\mathrm { C P F I _ { 2 } } = 0 .$ </td></tr><tr><td>Heteroskedast</td><td> $\mathbf { i c } X _ { 2 } \sim \mathcal { N } ( 0 , 1 )$   $X _ { 1 } \mid X _ { 2 } = x _ { 2 } \sim { \mathcal { N } } ( 0 , x _ { 2 } ^ { 2 } ) ;$   $g ( \pmb { x } ) = x _ { 1 } .$ </td><td> $I _ { 1 } = x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 } ; I _ { 2 } = 0 .$ </td><td> ${ \mathrm { C P F I } } _ { 1 } = 2 ;$   $\mathrm { C P F I _ { 2 } } = 0 .$ </td></tr><tr><td>Interacted discontinuity</td><td> $X _ { 2 } , X _ { 3 } \stackrel { \mathrm { i i d } } { \sim } { \mathcal { N } } ( 0 , 1 ) .$   $\mu = 4 ( { \bf 1 } _ { X _ { 2 } > 0 } \oplus { \bf 1 } _ { X _ { 3 } > 0 } ) - 2 ;$   $X _ { 1 } \mid X _ { 2 } , X _ { 3 } \sim { \mathcal { N } } ( \mu , 1 ) ; g ( \pmb { x } ) = x _ { 1 } .$ </td><td> $I _ { 1 } = 1 + ( x _ { 1 } - \mu ) ^ { 2 } ;$   $I _ { 2 } = I _ { 3 } = 0 .$ </td><td> ${ \mathrm { C P F I } } _ { 1 } = 2 ;$   $\mathrm { C P F I _ { 2 } = }$   $\mathrm { C P F I _ { 3 } } = 0 .$ </td></tr><tr><td>Extreme tail</td><td> $X _ { 2 } \sim { \mathcal { N } } ( 0 , 1 )$   $X _ { 1 } \mid X _ { 2 } = x _ { 2 } \sim { \mathcal { N } } ( 0 , v ( x _ { 2 } ) ) ;$   $v ( x _ { 2 } ) = 1 + \mathbf { 1 } _ { x _ { 2 } > 0 } ,$   $H ( x _ { 2 } ) = \mathbf { 1 } _ { | x _ { 2 } | > \Phi ^ { - 1 } ( . 9 9 ) } ;$   $g ( \pmb { x } ) = x _ { 1 } + 2 0 H ( x _ { 2 } )$ </td><td> $I _ { 1 } = x _ { 1 } ^ { 2 } + v ( x _ { 2 } ) ;$   $I _ { 2 } = 8 + 3 8 4 H ( x _ { 2 } )$ </td><td> $\mathrm { { C P F I } _ { 1 } = 3 ; }$   $\mathrm { C P F I _ { 2 } } = 1 5 . 6 8 .$ </td></tr></table>

## E.2 Analytical LUCID Calculations

We derive the expressions in Table 5. Under squared loss and using the prediction itself as the reference, the LUCID of feature j at observation x reduces to

$$
I _ { j } ( \pmb { x } ) = \mathbb { E } \left[ \left\{ g ( X _ { j } ^ { \prime } , \pmb { x } _ { - j } ) - g ( \pmb { x } ) \right\} ^ { 2 } \mid X _ { - j } = \pmb { x } _ { - j } \right] ,\tag{E.1}
$$

where $X _ { j } ^ { \prime } \sim P ( \cdot \mid X _ { - j } = x _ { - j } )$ . The corresponding global conditional permutation feature importance is

$$
\mathrm { C P F I } _ { j } = \mathbb { E } \left[ I _ { j } ( \boldsymbol { X } ) \right] .\tag{E.2}
$$

DGP 1: Discontinuity. Since the features are mutually independent, the conditional replacement distributions equal their marginals. For $X _ { 1 }$ , replacing $x _ { 1 }$ afects the prediction only when $x _ { 3 } > 0$ giving

$$
\begin{array} { r l } & { I _ { 1 } ( \pmb { x } ) = \mathbf { 1 } _ { x _ { 3 } > 0 } \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } ] } \\ & { \qquad = \mathbf { 1 } _ { x _ { 3 } > 0 } \left\{ \mathrm { V a r } ( X _ { 1 } ^ { \prime } ) + ( \mathbb { E } [ X _ { 1 } ^ { \prime } ] - x _ { 1 } ) ^ { 2 } \right\} } \\ & { \qquad = ( 1 + x _ { 1 } ^ { 2 } ) \mathbf { 1 } _ { x _ { 3 } > 0 } . } \end{array}\tag{E.3}
$$

Similarly,

$$
I _ { 2 } ( \pmb { x } ) = ( 1 + x _ { 2 } ^ { 2 } ) \mathbf { 1 } _ { x _ { 3 } \leq 0 } .\tag{E.4}
$$

For $X _ { 3 }$ , a replacement changes the prediction only when $X _ { 3 } ^ { \prime }$ falls on the opposite side of zero from the observed $x _ { 3 }$ . Since $\mathrm { P r } ( X _ { 3 } ^ { \prime } > 0 ) = \mathrm { P r } ( X _ { 3 } ^ { \prime } \leq 0 ) = 1 / 2$

$$
I _ { 3 } ( { \pmb x } ) = \frac { 1 } { 2 } ( x _ { 1 } - x _ { 2 } ) ^ { 2 } .\tag{E.5}
$$

Averaging over the data distribution gives

$$
\mathrm { C P F I } _ { 1 } = \mathbb { E } [ ( 1 + X _ { 1 } ^ { 2 } ) \mathbf { 1 } _ { X _ { 3 } > 0 } ] = 2 \times { \frac { 1 } { 2 } } = 1 ,\tag{E.6}
$$

$$
\mathrm { C P F I } _ { 2 } = \mathbb { E } [ ( 1 + X _ { 2 } ^ { 2 } ) \mathbf { 1 } _ { X _ { 3 } \leq 0 } ] = 2 \times { \frac { 1 } { 2 } } = 1 ,\tag{E.7}
$$

$$
\operatorname { C P F I } _ { 3 } = { \frac { 1 } { 2 } } \mathbb { E } [ ( X _ { 1 } - X _ { 2 } ) ^ { 2 } ] = { \frac { 1 } { 2 } } ( 1 + 1 ) = 1 .\tag{E.8}
$$

DGP 2: Correlated. For the bivariate Gaussian distribution,

$$
X _ { 1 } \mid X _ { 2 } = x _ { 2 } \sim { \mathcal { N } } ( \rho x _ { 2 } , 1 - \rho ^ { 2 } ) .\tag{E.9}
$$

Because $g ( \pmb { x } ) = x _ { 1 }$ , only perturbations of $X _ { 1 }$ afect the prediction. Hence

$$
\begin{array} { r l } & { I _ { 1 } ( \pmb { x } ) = \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } \ | \ X _ { 2 } = x _ { 2 } ] } \\ & { \qquad = \operatorname { V a r } ( X _ { 1 } ^ { \prime } \ | \ X _ { 2 } = x _ { 2 } ) + \{ \mathbb { E } [ X _ { 1 } ^ { \prime } \ | \ X _ { 2 } = x _ { 2 } ] - x _ { 1 } \} ^ { 2 } } \\ & { \qquad = 1 - \rho ^ { 2 } + ( x _ { 1 } - \rho x _ { 2 } ) ^ { 2 } . } \end{array}\tag{E.10}
$$

Since $g$ does not depend on $x _ { 2 }$ directly,

$$
I _ { 2 } ( { \pmb x } ) = 0 .\tag{E.11}
$$

Moreover, $X _ { 1 } - \rho X _ { 2 }$ has variance $1 - \rho ^ { 2 }$ , so

$$
\begin{array} { r l } & { \mathrm { C P F I _ { 1 } } = 1 - \rho ^ { 2 } + \mathbb { E } [ ( X _ { 1 } - \rho X _ { 2 } ) ^ { 2 } ] } \\ & { \qquad = 2 ( 1 - \rho ^ { 2 } ) , } \end{array}\tag{E.12}
$$

while $\mathrm { { C P F I } _ { 2 } = 0 }$

DGP 3: Heteroskedastic. Here

$$
X _ { 1 } \mid X _ { 2 } = x _ { 2 } \sim { \mathcal { N } } ( 0 , x _ { 2 } ^ { 2 } ) ,\tag{E.13}
$$

so the conditional variance of $X _ { 1 }$ depends on the local value of $X _ { 2 }$ . Since $g ( \pmb { x } ) = x _ { 1 }$

$$
\begin{array} { r l } & { I _ { 1 } ( \pmb { x } ) = \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } \ | \ X _ { 2 } = x _ { 2 } ] } \\ & { \qquad = \operatorname { V a r } ( X _ { 1 } ^ { \prime } \ | \ X _ { 2 } = x _ { 2 } ) + \{ \mathbb { E } [ X _ { 1 } ^ { \prime } \ | \ X _ { 2 } = x _ { 2 } ] - x _ { 1 } \} ^ { 2 } } \\ & { \qquad = x _ { 2 } ^ { 2 } + x _ { 1 } ^ { 2 } . } \end{array}\tag{E.14}
$$

Again, g does not depend directly on $X _ { 2 } ,$ , so $I _ { 2 } ( { \pmb x } ) = 0$ . Since

$$
\operatorname { \mathbb { E } } [ X _ { 1 } ^ { 2 } ] = \operatorname { \mathbb { E } } [ \operatorname { V a r } ( X _ { 1 } \mid X _ { 2 } ) ] = \operatorname { \mathbb { E } } [ X _ { 2 } ^ { 2 } ] = 1 ,\tag{E.15}
$$

we obtain

$$
\mathrm { C P F I } _ { 1 } = \mathbb { E } [ X _ { 1 } ^ { 2 } + X _ { 2 } ^ { 2 } ] = 2 , \qquad \mathrm { C P F I } _ { 2 } = 0 .\tag{E.16}
$$

DGP 4: Interacted discontinuity. Define

$$
\mu ( X _ { 2 } , X _ { 3 } ) = 4 \left\{ { \bf 1 } _ { X _ { 2 } > 0 } \oplus { \bf 1 } _ { X _ { 3 } > 0 } \right\} - 2 .\tag{E.17}
$$

Conditional on $( X _ { 2 } , X _ { 3 } )$ 2

$$
X _ { 1 } \mid X _ { 2 } , X _ { 3 } \sim { \mathcal { N } } ( \mu , 1 ) .\tag{E.18}
$$

Since $g ( \pmb { x } ) = x _ { 1 }$ , the LUCID of $X _ { 1 }$ is

$$
\begin{array} { c } { I _ { 1 } ( \pmb { x } ) = \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } \ | \ X _ { 2 } = x _ { 2 } , X _ { 3 } = x _ { 3 } ] } \\ { = 1 + ( x _ { 1 } - \mu ) ^ { 2 } . } \end{array}\tag{E.19}
$$

Although $X _ { 2 }$ and $X _ { 3 }$ determine the conditional distribution of $X _ { 1 }$ , neither enters the prediction function once $x _ { 1 }$ is held fixed. Therefore,

$$
I _ { 2 } ( { \pmb x } ) = I _ { 3 } ( { \pmb x } ) = 0 .\tag{E.20}
$$

Finally, because $X _ { 1 } - \mu \mid X _ { 2 } , X _ { 3 } \sim \mathcal { N } ( 0 , 1 )$

$$
\mathrm { C P F I } _ { 1 } = 1 + \mathbb { E } [ ( X _ { 1 } - \mu ) ^ { 2 } ] = 2 , \qquad \mathrm { C P F I } _ { 2 } = \mathrm { C P F I } _ { 3 } = 0 .\tag{E.21}
$$

DGP 5: Extreme tail. Let

$$
v ( x _ { 2 } ) = 1 + { \bf 1 } _ { x _ { 2 } > 0 } , \qquad H ( x _ { 2 } ) = { \bf 1 } _ { | x _ { 2 } | > c } , \qquad c = \Phi ^ { - 1 } ( 0 . 9 9 ) .\tag{E.22}
$$

For $X _ { 1 }$ , conditional on $X _ { 2 } = x _ { 2 }$ we have $X _ { 1 } ^ { \prime } \sim \mathcal { N } ( 0 , v ( x _ { 2 } ) )$ . Since changing $X _ { 1 }$ leaves the tail indicator fixed,

$$
\begin{array} { l } { I _ { 1 } ( \pmb { x } ) = \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } \ | \ X _ { 2 } = x _ { 2 } ] } \\ { \quad \quad = v ( x _ { 2 } ) + x _ { 1 } ^ { 2 } . } \end{array}\tag{E.23}
$$

Averaging gives

$$
\mathrm { C P F I _ { 1 } } = \mathbb { E } [ v ( X _ { 2 } ) ] + \mathbb { E } [ X _ { 1 } ^ { 2 } ]
$$

$$
\begin{array} { l } { \displaystyle = 2 \mathbb { E } [ v ( X _ { 2 } ) ] } \\ { \displaystyle = 2 \left( 1 + \frac { 1 } { 2 } \right) = 3 . } \end{array}\tag{E.24}
$$

For $X _ { 2 }$ , the value of $X _ { 1 }$ remains fixed under perturbation, and therefore

$$
g ( x _ { 1 } , X _ { 2 } ^ { \prime } ) - g ( x _ { 1 } , x _ { 2 } ) = 2 0 \{ H ( X _ { 2 } ^ { \prime } ) - H ( x _ { 2 } ) \} .\tag{E.25}
$$

Although $X _ { 1 }$ and $X _ { 2 }$ are dependent through $v ( X _ { 2 } )$ , the conditional probability that a replacement lies in the extreme tail remains 0.02. To see this, note that

$$
p ( x _ { 2 } \mid x _ { 1 } ) \propto \phi ( x _ { 2 } ) \phi \{ x _ { 1 } ; 0 , v ( x _ { 2 } ) \} ,\tag{E.26}
$$

where $\phi$ is a Gaussian density<sup>4</sup>, and $v ( x _ { 2 } )$ is constant within each of the two regions $x _ { 2 } > 0$ and $x _ { 2 } \leq 0$ . Thus, conditional on either sign of $X _ { 2 } ,$ its distribution remains a standard Gaussian truncated to that half-line. Consequently,

$$
\operatorname* { P r } ( X _ { 2 } ^ { \prime } > c \mid X _ { 1 } , X _ { 2 } ^ { \prime } > 0 ) = { \frac { \operatorname* { P r } ( X _ { 2 } > c ) } { \operatorname* { P r } ( X _ { 2 } > 0 ) } } = { \frac { 0 . 0 1 } { 0 . 5 } } = 0 . 0 2 ,\tag{E.27}
$$

$$
\operatorname* { P r } ( X _ { 2 } ^ { \prime } < - c \mid X _ { 1 } , X _ { 2 } ^ { \prime } \leq 0 ) = { \frac { \operatorname* { P r } ( X _ { 2 } < - c ) } { \operatorname* { P r } ( X _ { 2 } \leq 0 ) } } = { \frac { 0 . 0 1 } { 0 . 5 } } = 0 . 0 2 ,\tag{E.28}
$$

and hence

$$
\mathrm { P r } \{ H ( X _ { 2 } ^ { \prime } ) = 1 \mid X _ { 1 } \} = 0 . 0 2 .\tag{E.29}
$$

Therefore, when $H ( x _ { 2 } ) = 0$ , a perturbation changes the indicator with probability 0.02, while when $H ( x _ { 2 } ) = 1$ , it changes the indicator with probability 0.98. It follows that

$$
\begin{array} { c } { { I _ { 2 } ( x ) = 4 0 0 \left[ 0 . 0 2 \{ 1 - H ( x _ { 2 } ) \} + 0 . 9 8 H ( x _ { 2 } ) \right] } } \\ { { { } } } \\ { { = 8 + 3 8 4 H ( x _ { 2 } ) . } } \end{array}\tag{E.30}
$$

Finally, since $\operatorname* { P r } \{ H ( X _ { 2 } ) = 1 \} = 0 . 0 2$

$$
\begin{array} { r l } { \mathrm { C P F I _ { 2 } } = \mathbb { E } [ 8 + 3 8 4 H ( X _ { 2 } ) ] } \\ { = 8 + 3 8 4 ( 0 . 0 2 ) } \\ { = 1 5 . 6 8 . } \end{array}\tag{E.31}
$$

## E.3 Stress Test Across Random Seeds

To assess the robustness of our synthetic experiments to random variation, we repeat all five DGP experiments over 50 independent random seeds. For each method and DGP, we compute the MSE between the estimated and analytical LUCID for $X _ { 1 }$ . Table 6 reports the mean MSE together with its standard deviation across the 50 repetitions, thereby showing both average estimation accuracy and its variation across random realizations. We additionally rank the seven LUCID estimators within each paired DGP–seed experiment and report the resulting mean rank across all $5 \times 5 0 = 2 5 0$ settings in Table 7.

The repeated experiments confirm that the qualitative conclusions of the main experiments are stable across random seeds. DR-BART achieves the lowest mean MSE in DGPs 2–4 and remains highly competitive in DGP 5. In DGP 1, where the relevant features are independent and no conditional density estimation is required, simpler permutation-based approaches attain smaller errors. DGP 5 is similarly favorable to cs-PFI, although DR-BART remains close and exhibits low variability across repetitions. Most importantly, DR-BART does not rely on a single favorable dependence structure: its MSE remains consistently low across linear dependence, heteroskedasticity, discontinuous interactions, and extreme-tail behavior. The relatively small standard deviations further indicate that this performance is not driven by a small number of favorable random realizations.

Table 6: MSE of estimated LUCID for $X _ { 1 }$ across 50 independent random seeds. Each entry reports mean ± standard deviation across seeds. Lower values indicate more accurate and more stable estimation.
<table><tr><td>Method</td><td>DGP 1</td><td>DGP 2</td><td>DGP 3</td><td>DGP 4</td><td>DGP 5</td></tr><tr><td>DR-BART</td><td> $0 . 0 2 2 5 \pm 0 . 0 0 3 6$ </td><td> $\mathbf { 0 . 0 0 2 9 \pm 0 . 0 0 0 6 }$ </td><td> $\mathbf { 0 . 3 6 6 8 \pm 0 . 1 0 9 5 }$ </td><td> $\mathbf { 0 . 5 0 9 2 \pm 0 . 1 2 9 3 }$ </td><td> $0 . 0 8 6 5 \pm 0 . 0 1 2 2$ </td></tr><tr><td>ARF</td><td> $0 . 1 1 8 7 \pm 0 . 0 1 4 7$ </td><td> $0 . 0 0 4 1 \pm 0 . 0 0 0 6$ </td><td> $0 . 6 5 8 2 \pm 0 . 2 8 8 8$ </td><td> $1 . 2 8 5 9 \pm 0 . 1 7 3 8$ </td><td> $0 . 2 0 6 3 \pm 0 . 0 1 9 4$ </td></tr><tr><td>cs-PFI</td><td> $0 . 0 0 6 4 \pm 0 . 0 0 1 9$ </td><td> $0 . 0 1 1 6 \pm 0 . 0 0 1 6$ </td><td> $1 . 0 8 0 4 \pm 0 . 2 3 8 6$ </td><td> $7 8 . 0 9 0 2 \pm 5 . 7 0 7 8$ </td><td> $\mathbf { 0 . 0 5 1 0 \pm 0 . 0 1 2 5 }$ </td></tr><tr><td>CLIQUE</td><td> $\mathbf { 0 . 0 0 0 8 \pm 0 . 0 0 0 7 }$ </td><td> $4 . 5 7 2 0 \pm 0 . 2 4 9 7$ </td><td> $1 . 9 9 2 2 \pm 0 . 1 1 2 2$ </td><td> $8 0 . 3 2 7 1 \pm 1 . 7 6 6 0$ </td><td> $0 . 2 5 3 6 \pm 0 . 0 0 2 7$ </td></tr><tr><td>CMR</td><td> $0 . 5 5 5 6 \pm 0 . 0 5 8 6$ </td><td> $0 . 0 1 0 6 \pm 0 . 0 0 1 0$ </td><td> $2 . 5 5 9 7 \pm 0 . 2 1 2 8$ </td><td> $1 . 5 9 9 4 \pm 0 . 2 7 8 0$ </td><td> $0 . 8 8 0 6 \pm 0 . 0 5 8 3$ </td></tr><tr><td>GCMR</td><td> $0 . 5 4 4 7 \pm 0 . 0 6 0 3$ </td><td> $0 . 0 1 0 8 \pm 0 . 0 0 1 0$ </td><td> $2 . 4 3 9 9 \pm 0 . 1 5 6 3$ </td><td> $3 . 9 2 1 0 \pm 0 . 3 3 1 3$ </td><td> $0 . 8 8 8 3 \pm 0 . 0 5 9 0$ </td></tr><tr><td>Permutation</td><td> $0 . 0 0 6 3 \pm 0 . 0 0 1 7$ </td><td> $4 . 5 3 8 1 \pm 0 . 2 4 6 2$ </td><td> $2 . 0 1 4 5 \pm 0 . 1 1 5 0$ </td><td> $8 0 . 0 1 1 0 \pm 1 . 7 7 6 6$ </td><td> $0 . 2 7 8 7 \pm 0 . 0 0 7 3$ </td></tr></table>

Table 7: Mean MSE rank for $X _ { 1 }$ across the five DGPs and 50 independent random seeds, giving 250 paired DGP–seed comparisons. Lower is better.
<table><tr><td>Method</td><td>Mean rank</td></tr><tr><td>DR-BART</td><td>1.80</td></tr><tr><td>ARF</td><td>2.84</td></tr><tr><td>cs-PFI</td><td>3.29</td></tr><tr><td>CLIQUE</td><td>4.60</td></tr><tr><td>Permutation</td><td>4.80</td></tr><tr><td>CMR</td><td>5.16</td></tr><tr><td>GCMR</td><td>5.51</td></tr></table>

DR-BART also obtains the best overall mean rank of 1.80 across the 250 paired settings, followed by ARF (2.84) and cs-PFI (3.29). Together, the MSE distributions and ranking results show that the relative performance observed in the main experiments persists across repeated data realizations rather than being driven by a particular random seed.

## E.4 DGP 1: Discontinuity.

Additional results are provided in Figures 9 and 14.

## E.5 DGP 2: Correlated Features

We show the results for $\rho = 0 . 9$ in Figure 15. The first row contains the conditional permutationbased methods, which accurately capture the correlation-induced restriction on the replacement

distribution in this relatively simple setting. In contrast, marginal permutation does not account for this dependence structure and therefore fail to recover the corresponding local importance pattern. Additional results are provided in Figures 10.

## E.6 DGP 3: Heteroskedastic Conditional Dependence

Recall that $X _ { 2 } \sim N ( 0 , 1 ) , X _ { 1 } \mid X _ { 2 } = x _ { 2 } \sim N ( 0 , x _ { 2 } ^ { 2 } )$ , and $g ( \pmb { x } ) = x _ { 1 }$ . Under squared loss, the analytical importance of $X _ { 1 }$ is

$$
I _ { 1 } ( \pmb { x } ) = \mathbb { E } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } \ | \ X _ { 2 } = x _ { 2 } ] = x _ { 1 } ^ { 2 } + x _ { 2 } ^ { 2 } .
$$

Now consider any replacement scheme that uses the same distribution Q for every observation, with $\mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] < \infty$ . Its population local score is

$$
I _ { 1 } ^ { Q } ( { \pmb x } ) : = \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } - x _ { 1 } ) ^ { 2 } ] = x _ { 1 } ^ { 2 } - 2 x _ { 1 } \mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] + \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] .
$$

Consequently,

$$
I _ { 1 } ^ { Q } ( { \pmb x } ) - I _ { 1 } ( { \pmb x } ) = \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] - x _ { 2 } ^ { 2 } - 2 x _ { 1 } \mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] .
$$

Since $\mathbb { E } [ X _ { 1 } \mid X _ { 2 } ] = 0$ , the cross term vanishes:

$$
{  { \mathbb E } } \Big [ X _ { 1 } \Big \{ {  { \mathbb E } } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] - X _ { 2 } ^ { 2 } \Big \} \Big ] = {  { \mathbb E } } \Big [ {  { \mathbb E } } [ X _ { 1 } \ | \ X _ { 2 } ] \Big \{ {  { \mathbb E } } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] - X _ { 2 } ^ { 2 } \Big \} \Big ] = 0 .
$$

Using $\mathbb { E } [ X _ { 1 } ^ { 2 } ] = \mathbb { E } [ X _ { 2 } ^ { 2 } ] = 1$ and $\mathbb { E } [ X _ { 2 } ^ { 4 } ] = 3$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ ( I _ { 1 } ^ { Q } ( \pmb { X } ) - I _ { 1 } ( \pmb { X } ) ) ^ { 2 } ] = \mathbb { E } \bigg [ \Big \{ \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] - X _ { 2 } ^ { 2 } \Big \} ^ { 2 } \bigg ] + 4 \{ \mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] \} ^ { 2 } \mathbb { E } [ X _ { 1 } ^ { 2 } ] } \\ & { \quad \quad \quad \quad \quad = \{ \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] \} ^ { 2 } - 2 \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] + 3 + 4 \{ \mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] \} ^ { 2 } } \\ & { \quad \quad \quad \quad = \Big \{ \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] - 1 \Big \} ^ { 2 } + 2 + 4 \{ \mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] \} ^ { 2 } } \\ & { \quad \quad \quad \quad \geq 2 . } \end{array}
$$

Thus, every common replacement distribution incurs a population local MSE of at least 2 in this DGP, even when its induced score is computed exactly.

Nevertheless, the global averages satisfy

$$
\mathbb { E } [ I _ { 1 } ( \boldsymbol { X } ) ] = 2 , \qquad \mathbb { E } [ I _ { 1 } ^ { Q } ( \boldsymbol { X } ) ] = 1 + \mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] .
$$

In particular, the marginal distribution of $X _ { 1 }$ , which is also the pooled residual distribution under the correct conditional mean, satisfies $\mathbb { E } _ { Q } [ X _ { 1 } ^ { \prime } ] = 0$ and $\mathbb { E } _ { Q } [ ( X _ { 1 } ^ { \prime } ) ^ { 2 } ] = 1$ . It therefore yields

$$
I _ { 1 } ^ { Q } ( { \pmb x } ) = x _ { 1 } ^ { 2 } + 1 , \qquad \mathbb { E } [ I _ { 1 } ^ { Q } ( { \pmb X } ) ] = \mathbb { E } [ I _ { 1 } ( { \pmb X } ) ] = 2 , \qquad \mathbb { E } [ ( I _ { 1 } ^ { Q } ( { \pmb X } ) - I _ { 1 } ( { \pmb X } ) ) ^ { 2 } ] = 2 .
$$

Hence, exact recovery of global-average importance does not imply accurate local importance. The local error arises from replacing the context-dependent conditional variance $x _ { 2 } ^ { 2 }$ with a common variance and persists even with the correct conditional mean and arbitrarily many Monte Carlo replacements.

Additional results are provided in Figures 11 and 16.

![](images/0446fb8cf409c5cd5462a4f897077a834547b299fe05125bb3eeedac58f1e5e9.jpg)

![](images/5bd5cfa46a734242a4471e588e9569063d46a34e1369e61fa40f5f9ca5e27ddc.jpg)  
Figure 8: Extreme tail. Match and mismatch regions for the analytical and fitted LUCID structures in DGP 5. Match and mismatch regions between LUCID and local LOCO feature rankings in DGP 5, using analytical and estimated importance measures.

## E.7 DGP 4: Interacted Discontinuity.

What happens to local VI when the covariate dependence involves interactions? This case examines dependence driven by an XOR-type interaction between $X _ { 2 }$ and $X _ { 3 }$ in the conditional distribution of $X _ { 1 }$ . Such interactions can be dificult for recursive tree methods based on univariate split selection, because their joint signal is not apparent from individual univariate splits (Hornung and Boulesteix, 2022). We use this setting to assess how this dificulty afects local importance estimation. As shown in the results for DGP 4 in Table 2, DR-BART (which explores ensembles of tree structures through Markov Chain Monte Carlo simulations (Orlandi et al., 2021)) yields estimates closest to the analytical LUCID. The cs-PFI estimates closely resemble marginal permutation, consistent with its transformation-tree partition failing to recover the pure interaction. ARF substantially improves on marginal permutation, but its MSE is still approximately 3× higher than our DR-BART implementation. These results highlight the importance of recovering interaction structure in the conditional distribution when estimating LUCID. Additional results are provided in Figures 12 and 17.

## E.8 DGP 5: Extreme Tail.

How does LUCID difer from local LOCO? LOCO provides a natural comparison for LUCID because both assess a feature’s contribution given the information available in the remaining features. Although global LOCO and global conditional permutation importance difer only by a factor of two under squared loss and an oracle regression predictor (Reyero-Lobo et al., 2025), this proportionality does not generally hold locally, as established in Section 3.3. Compared to local LOCO, LUCID additionally accounts for prediction variability under conditional feature replacement, reflecting how important it is to know the exact feature value; omitting this component can substantially alter local feature rankings. We construct a heteroskedastic DGP with an extreme-tail component in the prediction function to examine the consequences of this distinction. Analytical local LOCO and analytical LUCID yield diferent feature rankings with an analytical probability of 67.4% (Analytical local LOCO and LUCID assign diferent feature rankings to 68.6% of the evaluated observations.). Consistent with this theoretical discrepancy, estimated local LOCO disagrees with estimated LUCID on 69.6% of the evaluated instances, as shown in Figure 8. The disagreement between the analytical rankings demonstrates that this discrepancy is intrinsic to the two importance measures and cannot be eliminated by improving LOCO estimation alone. This example shows that LUCID provides fuller local feature information about dispersion, leading to substantially diferent feature rankings from local LOCO despite their proportionality at the global level. Additional results are provided in Figures 13 and 18.

## E.9 Complete Conditional Replacement Distributions

We further show complete results of conditional replacement distributions produced by each sampling method across representative covariates. For each DGP, we evaluate the conditional distribution of $X _ { 1 }$ at five empirical quantiles of $X _ { 2 }$ and compare the estimated replacement distributions with the analytical conditional distribution. Across the five settings, DR-BART is the most stable method overall: it consistently tracks changes in the location, scale, and shape of the true conditional distribution. The remaining methods can perform well when their structural assumptions align with the DGP, but their performance is less uniform across the diferent conditional structures.

DGP 1: Discontinuity. Figure 9 shows the conditional replacement distributions for the discontinuity DGP. Because $X _ { 1 }$ is independent of the remaining features, its true conditional distribution does not vary with $X _ { 2 }$ . DR-BART closely preserves this invariant distribution across all five conditioning points. Most methods also perform reasonably well in this simple setting, as no conditional dependence needs to be learned, although ARF exhibits noticeable shifts in location or scale at the more extreme conditioning points.

DGP 2: Correlated features. Figure 10 illustrates the efect of strong linear dependence. As $X _ { 2 }$ moves from its lower to upper quantiles, the true conditional distribution of $X _ { 1 }$ shifts correspondingly while remaining substantially narrower than its marginal distribution. DR-BART closely follows both the changing conditional location and the restricted conditional spread across the full range of $X _ { 2 }$ . In contrast, marginal permutation remains centered around the marginal distribution and therefore entirely misses the correlation-induced restriction, while several other conditional methods capture the trend but exhibit larger errors in location or scale at the extremes.

DGP 3: Heteroskedastic conditional dependence. Figure 11 isolates changes in conditional variance while keeping the conditional mean fixed at zero. The analytical distribution becomes increasingly concentrated as $X _ { 2 }$ approaches zero and substantially more dispersed in the tails. DR-BART reproduces this local scale variation particularly closely, including the near-degenerate distribution around the median of $X _ { 2 }$ and the much wider distributions at the extreme quantiles. Methods that rely on permuting residual distributions cannot preserve this heteroskedastic structure, while subgroup-based methods recover some scale variation but less uniformly across conditioning points.

![](images/195ebf52abc6165e3f4637cfcab6c681928045d0f342e6b819b74ab6831a1059.jpg)  
Figure 9: Estimated conditional replacement distributions for $X _ { 1 }$ in the discontinuity DGP at five empirical quantiles of $X _ { 2 }$ , with $X _ { 3 } = 1$ . The analytical conditional distribution is shown as Optimal.

![](images/f4b7a6ee5d1cf3cfa2c044370695e93447459207705382a13d1d54e86de67ed6.jpg)  
Figure 10: Estimated conditional replacement distributions for $X _ { 1 }$ in the correlated-features DGP at five empirical quantiles of $X _ { 2 }$

![](images/9e3254330cb4fcd77e18629d45bf127f720dcdb6e1f769e3b46ef87a0639e761.jpg)  
Figure 11: Estimated conditional replacement distributions for $X _ { 1 }$ in the heteroskedastic DGP at five empirical quantiles of $X _ { 2 }$

DGP 4: Interacted discontinuity. Figure 12 presents the most structurally challenging conditional mean in our five DGPs. With $X _ { 3 } = 1$ , the conditional distribution of $X _ { 1 }$ undergoes an abrupt location shift as $X _ { 2 }$ crosses zero while maintaining approximately constant conditional variance. DR-BART accurately captures both sides of this discontinuity and transitions sharply at the relevant boundary. ARF, CMR, and GCMR also capture much of the location shift, whereas cs-PFI and marginal permutation fail to reproduce the correct regime-specific conditional location in this configuration. This example highlights the importance of representing interactions in the conditioning variables rather than only marginal or coarse subgroup structure.

DGP 5: Extreme tail. Figure 13 shows the conditional distributions in the extreme-tail setting. Although the conditional mean remains zero, the conditional variance changes across the sign of $X _ { 2 }$ , producing a moderate but important change in distributional spread. DR-BART again closely follows the analytical distribution across all five conditioning points and preserves the increase in conditional dispersion for positive $X _ { 2 }$ . Several alternative methods introduce noticeable local shifts in the conditional mean or fail to reproduce the change in scale, while marginal permutation necessarily uses essentially the same unconditional replacement distribution throughout.

Taken together, these results show that the advantage of DR-BART is not tied to a single type of conditional dependence. It remains stable across independence, smooth location shifts, heteroskedasticity, discontinuous interactions, and sign-dependent changes in scale. This flexibility is particularly relevant for LUCID, since errors in the local replacement distribution directly propagate to errors in observation-level importance.

## E.10 Complete Pointwise Results

We provide the complete observation-level local variable importance for all methods and all five DGPs in Figures 14–18. Each panel compares the estimated LUCID with its analytical value across evaluation observations, with the corresponding mean squared error (MSE) reported above each feature. These figures complement the aggregate MSE comparisons in the main text by showing where estimation errors arise across the covariate space.

Note: MSE is reported uniformly across all methods for completeness; however, the values for SHAP, conditional SHAP, and LIME should not be interpreted as directly comparable estimation errors because these methods target diferent estimands from LUCID.

## E.11 Additional Posterior Credible Interval Results

We further evaluate the calibration of the DR-BART posterior LUCID intervals across multiple nominal coverage levels for observations where $X _ { 1 }$ is active in the prediction function. Table 8 reports the empirical coverage (C) and average interval width (W) for nominal levels of 50%, 80%, 90%, and 95%. Across the five DGPs, empirical coverage generally increases with the nominal level as expected, while interval widths increase accordingly. Coverage is close to or above nominal in most settings, except that DGP 3 shows mild undercoverage at the 95% nominal level, where the more extreme heteroskedastic observations remain challenging in finite samples. Figures 19–22 show the corresponding pointwise posterior credible intervals for $X _ { 1 }$ across all five DGPs. Observations are ordered by the observed value of $X _ { 1 }$ , and we randomly select only 1000 points to avoid a crowded plot. Each figure compares the posterior mean and credible interval with the analytical LUCID, highlighting observations for which the analytical value is covered or missed.

Table 8: Empirical coverage (C) and average width (W) of posterior LUCID credible intervals for observations where $X _ { 1 }$ is active in the prediction function across nominal levels.
<table><tr><td rowspan="2">DGP</td><td colspan="2">50%</td><td colspan="2">80%</td><td colspan="2">90%</td><td colspan="2">95%</td></tr><tr><td>C</td><td>W</td><td>C</td><td>W</td><td>C</td><td>W</td><td>C</td><td>W</td></tr><tr><td>DGP 1</td><td>0.686</td><td>0.311</td><td>0.948</td><td>0.596</td><td>0.984</td><td>0.766</td><td>0.994</td><td>0.910</td></tr><tr><td>DGP 2</td><td>0.519</td><td>0.063</td><td>0.828</td><td>0.121</td><td>0.917</td><td>0.155</td><td>0.958</td><td>0.184</td></tr><tr><td>DGP 3</td><td>0.640</td><td>0.382</td><td>0.871</td><td>0.734</td><td>0.913</td><td>0.944</td><td>0.929</td><td>1.123</td></tr><tr><td>DGP 4</td><td>0.606</td><td>0.393</td><td>0.872</td><td>0.753</td><td>0.932</td><td>0.969</td><td>0.956</td><td>1.152</td></tr><tr><td>DGP 5</td><td>0.676</td><td>0.387</td><td>0.929</td><td>0.742</td><td>0.971</td><td>0.956</td><td>0.985</td><td>1.138</td></tr></table>

![](images/fb4833279609c06fab6c78c711ff5fc771a65f66eb4c585939cd0d75c8ed0c19.jpg)  
Figure 12: Estimated conditional replacement distributions for $X _ { 1 }$ in the interacted-discontinuity DGP at five empirical quantiles of $X _ { 2 } ,$ with $X _ { 3 } = 1$

![](images/3cbcc52bd97205e0350d4ee50bfdc607eb4e06567f7d8143c8d1b3ded23895cb.jpg)  
Figure 13: Estimated conditional replacement distributions for $X _ { 1 }$ in the extreme-tail DGP at five empirical quantiles of $X _ { 2 }$

![](images/2a5f4863427fd561a862d0e51e53280b05e797edb052119cd8cdef297014e414.jpg)  
Figure 14: Complete pointwise LUCID results for the regime-dependent DGP. Estimated local variable importance values from all conditional sampling methods are compared with the corresponding analytical LUCID across evaluation observations.

Correlated

![](images/fffe9737b320f07e6259c5b9af81d7660b8be0e059ddcf4a0d4bf94715d7da98.jpg)  
Figure 15: Complete pointwise LUCID results for the strong conditional dependence DGP. Estimated local variable importance values from all conditional sampling methods are compared with the corresponding analytical LUCID across evaluation observations.

Analytical LUCID

Heteroskedastic

![](images/1ebdf272f24fd9b6593b7efe6ac7acbd54d27b0d8b9ee4e0caec6cda5ac57018.jpg)  
Figure 16: Complete pointwise LUCID results for the heteroskedastic conditional dependence DGP. Estimated local variable importance values from all conditional sampling methods are compared with the corresponding analytical LUCID across evaluation observations.

Interacted Discontinuity  
![](images/f63f40dcef75c9c8355cde4731d92d5c4765a28d83ba873ec32d088d74ace721.jpg)  
Figure 17: Complete pointwise LUCID results for the interaction conditional dependence DGP. Estimated local variable importance values from all conditional sampling methods are compared with the corresponding analytical LUCID across evaluation observations.

Extreme Tail

![](images/13bd1c2a8cd416df769c8adb6d63ceb5d1247e85e50e7ae8333778cb6244bf54.jpg)  
Figure 18: Complete pointwise LUCID results for the extreme-tail DGP. Estimated local variable importance values from all conditional sampling methods are compared with the corresponding analytical LUCID across evaluation observations.

X1: 50% posterior credible intervals and analytical LUCID  
![](images/ca1e05417fade513da76010e1c634a4195d32306c3414cf1cf605165e201a758.jpg)  
Figure 19: 50% posterior credible intervals. Pointwise posterior LUCID credible intervals for X<sub>1</sub> across the five DGPs at the 50% nominal level. Posterior means are compared with analytical LUCID values, with covered and missed analytical values indicated separately.

X1: 80% posterior credible intervals and analytical LUCID 80% interval Posterior mean Analytical: covered  
![](images/1db41876635b04b4c62d21e0b4426524dd54b272614f508d88034867cd0e26c2.jpg)  
Observation (ordered by observed $X _ { 1 } )$  
Figure 20: 80% posterior credible intervals. Pointwise posterior LUCID credible intervals for $X _ { 1 }$ across the five DGPs at the 80% nominal level. Posterior means are compared with analytical LUCID values, with covered and missed analytical values indicated separately.

X1: 90% posterior credible intervals and analytical LUCID 90% interval Posterior mean Analytical: covered  
![](images/81faa487e21419dfd0cbdf6ee1d5ebbc60c94aaf3993163ea1c2328c194a66f8.jpg)  
Figure 21: 90% posterior credible intervals. Pointwise posterior LUCID credible intervals for $X _ { 1 }$ across the five DGPs at the 90% nominal level. Posterior means are compared with analytical LUCID values, with covered and missed analytical values indicated separately.

X1: 95% posterior credible intervals and analytical LUCID  
![](images/d53ba4a6313d756a15547f7d451de9ef9b73aacc1a1a7156f8e201cec20d9677.jpg)  
Figure 22: 95% posterior credible intervals. Pointwise posterior LUCID credible intervals for $X _ { 1 }$ across the five DGPs at the 95% nominal level. Posterior means are compared with analytical LUCID values, with covered and missed analytical values indicated separately.

## E.12 Additional Results on the cs-PFI Simulation

To further evaluate LUCID estimation in higher-dimensional settings, we consider the groundtruth simulation introduced by Molnar et al. (2024) for evaluating conditional permutation feature importance. We retain the original scenario names—independent, linear, multiple linear dependencies, and non-linear—and vary the sample size and ambient dimension as in their experiment. Following their setting (I), we evaluate the true prediction function directly. Outcome noise is omitted because the LUCID calculation does not use the outcome. We only care about the given model.

The prediction function is

$$
g ( \pmb { x } ) = x _ { 1 } x _ { 2 } + \sum _ { j = 1 } ^ { 1 0 } x _ { j } ,\tag{E.32}
$$

where all features except $X _ { 1 }$ are independently distributed as $X _ { j } \sim \mathcal { N } ( 0 , 1 )$ . Feature $X _ { 1 }$ is generated according to

$$
X _ { 1 } = h ( X _ { - 1 } ) + \epsilon _ { x } ,\tag{E.33}
$$

with h and the distribution of $\epsilon _ { x }$ varying across the four scenarios. The original simulation was designed to range from independence to increasingly complex conditional dependence structures, including dependence on many covariates and discontinuous changes in both the conditional mean and variance (Molnar et al., 2024). In the first three scenarios, $\epsilon _ { x }$ is independent of $X _ { - 1 }$ and $\sigma _ { x } ( { \pmb X } _ { - 1 } )$ in the last scenario denotes the conditional variance, rather than the conditional standard deviation.

Table 9: Data-generating processes from the cs-PFI simulation and the corresponding analytical LUCID and CPFI for $X _ { 1 }$ under squared loss.
<table><tr><td>Setting</td><td>Conditional distribution of  $X _ { 1 }$ </td><td>Analytical LUCID</td><td>CPFI</td></tr><tr><td>Independent</td><td> $h ( X _ { - 1 } ) = 0 , \epsilon _ { x } \sim \mathcal { N } ( 0 , 1 )$ </td><td> $( x _ { 2 } + 1 ) ^ { 2 } ( 1 + x _ { 1 } ^ { 2 } )$ </td><td>4</td></tr><tr><td>Linear</td><td> $h ( X _ { - 1 } ) = X _ { 2 } , \epsilon _ { x } \sim \mathcal { N } ( 0 , 1 )$ </td><td> $( x _ { 2 } + 1 ) ^ { 2 } \{ 1 + ( x _ { 1 } - x _ { 2 } ) ^ { 2 } \}$ </td><td>4</td></tr><tr><td>Multiple linear dependencies</td><td> $h ( \boldsymbol { X } _ { - 1 } ) = \sum _ { j = 2 } ^ { 1 0 } X _ { j } , \epsilon _ { x } \sim \mathcal { N } ( 0 , 5 )$ </td><td> $( x _ { 2 } +$   $\begin{array} { r } { 1 ) ^ { 2 } \left\{ 5 + \left( x _ { 1 } - \sum _ { j = 2 } ^ { 1 0 } x _ { j } \right) ^ { 2 } \right\} } \end{array}$ </td><td>20</td></tr><tr><td>Non-linear</td><td> $h ( { \cal X } _ { - 1 } ) = 3 { \bf 1 } ( { \cal X } _ { 2 } > 0 ) - 3 { \bf 1 } ( { \cal X } _ { 2 } \leq 0 ) { \bf 1 } ( { \cal X } _ { 3 } > 0 ) ,$   $\epsilon _ { x } \mid X _ { - 1 } \sim { \mathcal { N } } ( 0 , \sigma _ { x } ( X ) ) ,$   $\sigma _ { x } ( X ) = \mathbf { 1 } ( X _ { 2 } > 0 )$   $+ 2 { \bf 1 } ( X _ { 2 } \le 0 ) { \bf 1 } ( X _ { 3 } > 0 ) ~ 1 ) ^ { 2 } \left[ \sigma _ { x } ( x ) + \{ x _ { 1 } - h ( x _ { - 1 } ) \} ^ { 2 } \right]$ </td><td> $( x _ { 2 } +$ </td><td>5.011</td></tr></table>

Analytical calculation. For all four scenarios, the dependence of the prediction function on $X _ { 1 }$ can be isolated as

$$
g ( x _ { 1 } ^ { \prime } , \pmb { x } _ { - 1 } ) - g ( x _ { 1 } , \pmb { x } _ { - 1 } ) = ( x _ { 2 } + 1 ) ( x _ { 1 } ^ { \prime } - x _ { 1 } ) .\tag{E.34}
$$

Let

$$
m ( { \pmb x } _ { - 1 } ) = \mathbb { E } [ { \pmb X } _ { 1 } \mid { \pmb X } _ { - 1 } = { \pmb x } _ { - 1 } ] , \qquad v ( { \pmb x } _ { - 1 } ) = \mathrm { V a r } ( { \pmb X } _ { 1 } \mid { \pmb X } _ { - 1 } = { \pmb x } _ { - 1 } ) .
$$

Under squared loss, the observation-level LUCID for $X _ { 1 }$ is therefore

$$
\begin{array} { r l } & { \mathrm { L U C I D } _ { i , 1 } = \mathbb E \left[ \{ g ( X _ { 1 } ^ { \prime } , \pmb { x } _ { i , - 1 } ) - g ( x _ { i , 1 } , \pmb { x } _ { i , - 1 } ) \} ^ { 2 } \mid \pmb { x } _ { i , - 1 } \right] } \\ & { \qquad = ( x _ { i , 2 } + 1 ) ^ { 2 } \mathbb E \left[ ( X _ { 1 } ^ { \prime } - x _ { i , 1 } ) ^ { 2 } \mid \pmb { x } _ { i , - 1 } \right] } \\ & { \qquad = ( x _ { i , 2 } + 1 ) ^ { 2 } \left[ v ( \pmb { x } _ { i , - 1 } ) + \{ x _ { i , 1 } - m ( \pmb { x } _ { i , - 1 } ) \} ^ { 2 } \right] . } \end{array}\tag{E.35}
$$

Averaging over observations gives the corresponding conditional permutation feature importance,

$$
\begin{array} { r l } {  { \mathrm { C P F I _ { 1 } } = \mathbb { E } [ \mathrm { L U C I D _ { 1 } } ] } \quad } & { } \\ & { = 2 \mathbb { E } [ ( X _ { 2 } + 1 ) ^ { 2 } v ( X _ { - 1 } ) ] , } \end{array}\tag{E.36}
$$

where the second equality follows because, conditional on $X _ { - 1 }$

$$
{ \mathbb E } \left[ \{ X _ { 1 } - m ( X _ { - 1 } ) \} ^ { 2 } \mid X _ { - 1 } \right] = v ( X _ { - 1 } ) .
$$

For the independent setting, $m ( { \pmb X } _ { - 1 } ) = 0$ and $v ( X _ { - 1 } ) = 1$ , giving

$$
\mathrm { L U C I D } _ { 1 } = ( X _ { 2 } + 1 ) ^ { 2 } ( 1 + X _ { 1 } ^ { 2 } ) , \qquad \mathrm { C P F I } _ { 1 } = 2 \mathbb { E } [ ( X _ { 2 } + 1 ) ^ { 2 } ] = 4 .
$$

For the linear setting, $m ( X _ { - 1 } ) = X _ { 2 }$ and $v ( X _ { - 1 } ) = 1$ , so

$$
\mathrm { { L U C I D } _ { 1 } = ( X _ { 2 } + 1 ) ^ { 2 } \{ 1 + ( X _ { 1 } - X _ { 2 } ) ^ { 2 } \} , \qquad \mathrm { { C P F I } _ { 1 } = 4 . } }
$$

For the multiple linear dependencies setting,

$$
m ( { \pmb X } _ { - 1 } ) = \sum _ { j = 2 } ^ { 1 0 } { \pmb X } _ { j } , \qquad v ( { \pmb X } _ { - 1 } ) = 5 ,
$$

and hence

$$
\mathrm { L U C I D _ { 1 } } = ( X _ { 2 } + 1 ) ^ { 2 } \left[ 5 + \left( X _ { 1 } - \sum _ { j = 2 } ^ { 1 0 } X _ { j } \right) ^ { 2 } \right] , \qquad \mathrm { C P F I _ { 1 } } = 2 0 .
$$

Finally, in the non-linear setting,

$$
m ( { \pmb X } _ { - 1 } ) = h ( { \pmb X } _ { - 1 } ) , \qquad v ( { \pmb X } _ { - 1 } ) = \sigma _ { x } ( { \pmb X } ) ,
$$

which yields

$$
\operatorname { L U C I D 1 } = ( X _ { 2 } + 1 ) ^ { 2 } \left[ \sigma _ { x } ( X ) + \left\{ X _ { 1 } - h ( X _ { - 1 } ) \right\} ^ { 2 } \right] .
$$

Since $X _ { 2 }$ and $X _ { 3 }$ are independent standard Gaussian variables,

$$
\begin{array} { r l } & { \mathrm { C P F I } _ { 1 } = 2 \mathbb { E } [ ( X _ { 2 } + 1 ) ^ { 2 } \sigma _ { x } ( X ) ] } \\ & { \qquad = 2 \left[ \mathbb { E } \{ ( X _ { 2 } + 1 ) ^ { 2 } \mathbf { 1 } ( X _ { 2 } > 0 ) \} + \frac { 2 + 5 } { 2 } \mathbb { E } \{ ( X _ { 2 } + 1 ) ^ { 2 } \mathbf { 1 } ( X _ { 2 } \leq 0 ) \} \right] } \\ & { \qquad = 2 \left[ 1 + \sqrt { \frac { 2 } { \pi } } + \frac { 7 } { 2 } \left( 1 - \sqrt { \frac { 2 } { \pi } } \right) \right] } \\ & { \qquad \approx 5 . 0 1 1 . } \end{array}\tag{E.37}
$$

Table 10: MSE between estimated and analytical LUCID for $X _ { 1 }$ in the cs-PFI simulation settings. Lower values indicate more accurate estimation. The lowest MSE in each setting is shown in bold.
<table><tr><td>Setting</td><td>DR-BART</td><td>ARF</td><td>cs-PFI</td><td>CMR</td><td>GCMR</td><td>CLIQUE</td><td>Perm.</td></tr><tr><td>Independent</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n = 3 0 0 , \ p = 1 0$ </td><td>3.30</td><td>13.23</td><td>1.29</td><td>21.94</td><td>15.68</td><td>0.13</td><td>0.95</td></tr><tr><td> $n = 3 0 0 , \ p = 9 0$ </td><td>5.03</td><td>4.04</td><td>0.31</td><td>17.31</td><td>10.32</td><td>0.26</td><td>0.25</td></tr><tr><td> $n = 3 0 0 0 , \ p = 1 0$ </td><td>1.21</td><td>15.65</td><td>0.22</td><td>10.89</td><td>16.05</td><td>0.014</td><td>0.21</td></tr><tr><td> $n = 3 0 0 0 , \ p = 9 0$ </td><td>1.21</td><td>7.44</td><td>0.10</td><td>9.53</td><td>8.23</td><td>0.004</td><td>0.10</td></tr><tr><td>Linear</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n = 3 0 0 , \ p = 1 0$ </td><td>26.52</td><td>170.37</td><td>73.22</td><td>60.85</td><td>45.93</td><td>302.60</td><td>367.92</td></tr><tr><td> $n = 3 0 0 , \ p = 9 0$ </td><td>30.00</td><td>206.58</td><td>24.65</td><td>21.86</td><td>30.50</td><td>201.73</td><td>197.66</td></tr><tr><td> $n = 3 0 0 0 , \ p = 1 0$ </td><td>5.59</td><td>31.87</td><td>7.19</td><td>18.80</td><td>16.46</td><td>304.57</td><td>312.82</td></tr><tr><td> $n = 3 0 0 0 , \ p = 9 0$ </td><td>2.51</td><td>437.41</td><td>5.66</td><td>12.80</td><td>16.21</td><td>437.32</td><td>443.73</td></tr><tr><td colspan="2">Multiple linear dependencies</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n = 3 0 0 , \ p = 1 0$ </td><td>239.59</td><td>2336.59</td><td>8575.39</td><td>1629.81</td><td>3037.15</td><td>7386.04</td><td>8688.08</td></tr><tr><td> $n = 3 0 0 , \ p = 9 0$ </td><td>848.80</td><td>4147.11</td><td>2562.16</td><td>1668.91</td><td>1712.87</td><td>3351.69</td><td>2931.48</td></tr><tr><td> $n = 3 0 0 0 , \ p = 1 0$ </td><td>125.38</td><td>1028.88</td><td>2742.75</td><td>639.26</td><td>741.14</td><td>6823.13</td><td>6856.28</td></tr><tr><td> $n = 3 0 0 0 , \ p = 9 0$ </td><td>139.18</td><td>8502.58</td><td>4116.04</td><td>1146.00</td><td>837.31</td><td>7948.26</td><td>7991.29</td></tr><tr><td>Non-linear</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $n = 3 0 0 , \ p = 1 0$ </td><td></td><td>41.871118.96</td><td>6.71</td><td>122.98</td><td>246.57</td><td>1832.60</td><td>1989.33</td></tr><tr><td> $n = 3 0 0 , \ p = 9 0$ </td><td>101.58</td><td>1619.07</td><td>6.26</td><td>66.22</td><td>71.33</td><td>1418.45</td><td>1429.19</td></tr><tr><td> $n = 3 0 0 0 , \ p = 1 0$ </td><td>3.86</td><td>196.79</td><td>0.74</td><td>59.03</td><td>80.88</td><td>1572.78</td><td>1588.26</td></tr><tr><td> $n = 3 0 0 0 , \ p = 9 0$ </td><td>3.97</td><td>1654.96</td><td>0.26</td><td>50.32</td><td>51.35</td><td>1897.84</td><td>1899.32</td></tr><tr><td>Mean rank</td><td>2.31</td><td>5.38</td><td>2.81</td><td>3.56</td><td>4.06</td><td>4.56</td><td>5.31</td></tr></table>

Results. Table 10 reports the MSE between estimated and analytical LUCID for $X _ { 1 }$ . We additionally report the mean rank across all combinations of sample size (n), dimension (p), and dependence structure. <sup>5</sup>

Across all 16 settings, DR-BART achieves the best overall mean rank of 2.31, followed by cs-PFI with a mean rank of 2.81. The relative performance varies substantially with the structure of the conditional distribution. In the linear setting, DR-BART is highly competitive and attains the lowest MSE in three of the four configurations. Its advantage is particularly pronounced in the multiple linear dependencies setting, where $X _ { 1 }$ depends jointly on nine covariates and DR-BART attains the lowest MSE in all four configurations. This setting is challenging for subgroup-based permutation methods because a tree must approximate several simultaneous linear dependencies through repeated partitioning; indeed, Molnar et al. (2024) similarly identify this setting as particularly dificult for cs-PFI.

The independent setting represents a diferent edge case. Because $X _ { 1 } ~ \bot ~ X _ { - 1 }$ , the correct conditional distribution is simply the marginal distribution of $X _ { 1 }$ , so no conditional density estimation is required. Methods that estimate a conditional distribution can therefore incur additional finitesample estimation error without gaining information, whereas marginal or near-marginal perturbation schemes can perform particularly well. This explains why the simplest methods obtain very small errors in this setting and why the independent case should not be interpreted as favoring a more

flexible conditional estimator.

In the non-linear setting, cs-PFI performs best in all four configurations, while DR-BART ranks second in three of the four configurations. This $\mathrm { D G P }$ is particularly favorable to the subgroup construction used by cs-PFI: both the conditional mean and variance of $X _ { 1 }$ change across only a small number of axis-aligned regions defined by the signs of $X _ { 2 }$ and $X _ { 3 }$ . A transformation tree can therefore represent the relevant conditional structure using only a few splits. In contrast, DR-BART estimates the full conditional density through a flexible mixture representation, which is substantially more general than is required for this piecewise structure and consequently pays a larger finite-sample estimation cost. Nevertheless, its error decreases sharply as the sample size increases. For example, when $p = 1 0$ , the MSE decreases from $4 1 . 8 7$ at $n = 3 0 0$ to 3.86 at $n = 3 0 0 0$ while for $p = 9 0$ it decreases from 101.58 to 3.97. Thus, although the finite-sample problem is substantially harder for DR-BART in this setting, the improvement with increasing sample size indicates that it is able to recover the underlying conditional structure with suficient data.

Another notable result is the poor performance of ARF in these higher-dimensional settings, where it is consistently among the worst-performing methods and often deteriorates markedly as the ambient dimension increases. This provides further empirical support for our motivation to estimate the conditional density directly: approaches such as ARF recover the conditional distribution through an estimated joint distribution of all covariates, a task that becomes increasingly challenging as the feature dimension grows (Izbicki and Lee, 2016).

Overall, these experiments emphasize that no conditional sampler uniformly dominates across all dependence structures. Subgroup permutation is particularly efective when the conditional distribution admits a simple tree partition, whereas direct conditional density estimation is substantially more competitive when dependence is smooth or distributed across many covariates. Importantly, LUCID itself is agnostic to this choice of conditional density estimator; these results therefore characterize the statistical trade-ofs of the sampling mechanism rather than a restriction of the LUCID framework.

![](images/9a2330f596336fd426b5620851aa091884652cf638d2683d908f3951d81b2f98.jpg)  
Figure 23: Complete individual LUCID results for the Bike Sharing data. For temperature, hour, working day, humidity, and season, panel (a) shows mean LUCID across hour of day separately for working and non-working days. Panels (b)–(d) compare the DR-BART conditional replacement distributions across season, working-day status, and hour periods with the corresponding marginal permutation distributions.

## F Additional Bike Sharing Results

## F.1 Complete Individual LUCID Results

We provide the complete individual LUCID results for all five features in Figure 23 as a complement to the main paper’s results. For each feature, the first column shows how its mean LUCID varies over hour of day on working and non-working days, while the remaining columns show the corresponding DR-BART conditional replacement distributions across season, working-day status, and hour periods, together with the marginal permutation distribution for comparison. The results reveal substantial heterogeneity for temperature, humidity, and season: their plausible replacement values shift across diferent covariate contexts, and their local importance varies accordingly. In contrast, the conditional replacement distributions of hour and working day are comparatively stable across the displayed contexts.

## F.2 Feature Dependence and the Interpretation of Individual LUCID

How do feature dependencies inform the interpretation of individual LUCID? Figure 24 summarizes feature dependencies alongside the estimated individual LUCIDs. We report pairwise correlations to describe linear associations and quantify each feature’s predictability from the remaining features as

$$
R _ { j } ^ { 2 } : = 1 - { \frac { \mathbb { E } [ \operatorname { V a r } ( X _ { j } \mid X _ { - j } ) ] } { \operatorname { V a r } ( X _ { j } ) } } .
$$

High predictability indicates that conditional resampling permits less variation on average relative to marginal resampling because information in this feature is already captured by the remaining features. Consequently, a small individual LUCID may reflect either limited model sensitivity to the feature or limited variation under conditional replacement. If the remaining features completely determine the feature of interest, then conditional replacement introduces no variation and LUCID is zero. Conversely, low predictability alone does not imply large LUCID, because the efect of replacement also depends on the model’s response: if the model does not use the feature of interest, its LUCID remains zero. Moreover, $R _ { j } ^ { 2 } = 0$ does not imply independence, since conditional variance or other distributional properties may still depend on the remaining features, as illustrated by the heteroskedastic synthetic example.

Pairwise correlations provide a complementary but weaker diagnostic. By the law of total variance,

$$
R _ { j } ^ { 2 } = { \frac { \operatorname { V a r } \{ \mathbb { E } ( X _ { j } \mid X _ { - j } ) \} } { \operatorname { V a r } ( X _ { j } ) } } ,
$$

which is the squared correlation ratio of $X _ { j }$ with respect to $X _ { - j }$ . Therefore, for any $k \neq j$

$$
\mathrm { C o r r } ( X _ { j } , X _ { k } ) ^ { 2 } \leq \operatorname* { s u p } _ { a } \mathrm { C o r r } ( X _ { j } , { \pmb a } ^ { \top } X _ { - j } ) ^ { 2 } \leq R _ { j } ^ { 2 } .
$$

Thus, pairwise correlation does not capture all forms of predictability, but a strong pairwise correlation is suficient evidence that the feature of interest may be highly predictable from the remaining covariates and that conditional importance should be interpreted accordingly. In Figure 24, temperature has particularly high predictability, while humidity and season are also meaningfully predictable from the remaining features. In contrast, hour and working day have very low predictability. These patterns help distinguish whether a small or large LUCID is primarily associated with restrictions in the conditional replacement distribution or with the model’s response to the feature.

Note: For binary and categorical features, we replace variance by Gini impurity, $\begin{array} { r } { G ( p ) = 1 - \sum _ { c } p _ { c } ^ { 2 } } \end{array}$ and compute predictability as $1 - \mathbb { E } \{ G ( p ( \cdot \mid X _ { - j } ) ) \} / G ( p )$ . This avoids dependence on arbitrary category codes and agrees with the variance-based measure for binary features. We estimate conditional uncertainty using 200 DR-BART draws per observation, with a finite-sample correction for Gini impurity. DR-BART samples continuous values from its posterior predictive density and discrete values from BART-based class probabilities; ARF samples from its conditional forest distribution; cs-PFI resamples within fitted tree leaves; CMR and GCMR resample residuals in the original and Gaussianized spaces, respectively, without enforcing categorical support. Permutation samples from the empirical marginal, while CLIQUE uses marginal quantiles or category-frequency grids.

## F.3 Predictability Diagnostics for Feature Groups

The main paper shows that group LUCID can complement individual importance when several features contain overlapping or shared information. Here, we provide two additional predictability

![](images/8e0953b6e038535fd57c8762d0b15536db817f9cbe1f5abc506ea9b0605afce1.jpg)  
Figure 24: Feature relationships and individual LUCID in the Bike Sharing data. The left panel summarizes pairwise correlations and each feature’s predictability $R _ { j } ^ { 2 }$ from the remaining covariates. The right panel shows observation-level LUCID values, colored by the observed feature value. Temperature is highly predictable from the remaining features, while hour and working day have very low predictability, providing additional context for interpreting their individual LUCID values.

diagnostics to help interpret a candidate feature group and assess whether its members meaningfully cover one another’s information. As defined above, let G denote a group of $d = | G |$ features, and let $X _ { - G }$ denote all features outside the group. The first metric is the group outside predictability,

$$
R _ { G } ^ { 2 } = 1 - { \frac { 1 } { d } } \sum _ { j \in G } { \frac { \operatorname { \mathbb { E } } [ \operatorname { V a r } ( X _ { j } \mid X _ { - G } ) ] } { \operatorname { V a r } ( X _ { j } ) } } ,\tag{F.1}
$$

which measures, under equal weighting, how well the features outside the group can explain the features within the group.

The second metric, denoted by $R _ { G , \mathrm { w i t h i n } } ^ { 2 }$ , measures how much the other features within the group further improve the prediction of each group member relative to using only features outside the group:

$$
R _ { G , \mathrm { w i t h i n } } ^ { 2 } = 1 - { \frac { 1 } { d } } \sum _ { j \in G } { \frac { \mathbb { E } [ \mathrm { V a r } ( X _ { j } \mid X _ { - j } ) ] } { \mathbb { E } [ \mathrm { V a r } ( X _ { j } \mid X _ { - G } ) ] } } .\tag{F.2}
$$

At the population level, both metrics lie in [0, 1] when their denominators are positive. Finitesample estimates need not satisfy these bounds exactly. A large $R _ { G } ^ { 2 }$ indicates that the group can already be predicted well from features outside the group, suggesting that the group may contribute limited information beyond the remaining covariates. In contrast, a small $R _ { G } ^ { 2 }$ indicates that the group retains substantial information that outside features cannot easily reconstruct. A large $R _ { G , \mathrm { w i t h i n } } ^ { 2 }$ indicates that on average across group members, the other members substantially reduce residual prediction variance, making joint perturbation particularly informative relative to separate individual perturbations. If $R _ { G , \mathrm { w i t h i n } } ^ { 2 }$ is very small, adding the other group members contributes little additional predictive information, which may suggest reconsidering whether those variables form a substantively meaningful group.

Table 11 shows substantial diferences among the candidate groups. The three-feature group consisting of season, humidity, and temperature has the lowest outside predictability $( R _ { G } ^ { 2 } = 0 . 0 9 5 )$ while maintaining relatively high within-group predictability $( R _ { G , \mathrm { w i t h i n } } ^ { 2 } = 0 . 4 1 0 )$ . This makes it a particularly natural candidate for group LUCID: the group is dificult to reconstruct from the remaining covariates, yet its members contain substantial overlapping information that can be revealed by joint perturbation. The season–temperature group shows a similar pattern, with low outside predictability (0.198) and the highest within-group predictability (0.450), also supporting a meaningful grouping interpretation.

Table 11: Group outside predictability $R _ { G } ^ { 2 }$ and within-group predictability $R _ { G , \mathrm { w i t h i n } } ^ { 2 }$ for candidate weather-related feature groups in the Bike Sharing data.
<table><tr><td>Group</td><td> $R _ { G } ^ { 2 }$ </td><td> $R _ { G , \mathrm { w i t h i n } } ^ { 2 }$ </td></tr><tr><td>Season + temperature</td><td>0.198</td><td>0.450</td></tr><tr><td>Temperature + humidity</td><td>0.482</td><td>0.190</td></tr><tr><td>Humidity + season</td><td>0.298</td><td>0.054</td></tr><tr><td>Season + humidity + temperature</td><td>0.095</td><td>0.410</td></tr></table>

In contrast, the temperature–humidity group has substantially higher outside predictability $( R _ { G } ^ { 2 } = 0 . 4 8 2 )$ , indicating that much of the group’s information can already be recovered from the remaining covariates, while its within-group predictability is more modest (0.190). The humidity– season group has moderate outside predictability (0.298) but very low within-group predictability (0.054), suggesting that these two features provide little additional predictive information about one another once the remaining covariates are known. Thus, among the groups considered here, the three-feature weather group and the season–temperature pair provide the strongest evidence for complementary group analysis, whereas the humidity–season pair is less naturally motivated by shared conditional information.

These metrics are intended as diagnostics rather than a unique grouping rule; substantive knowledge and the scientific question remain important when choosing feature groups. For broader discussions of feature grouping strategies, we refer readers to the literature on grouped variable importance (Gregorutti et al., 2015; Au et al., 2022; Chamma et al., 2024).

Together with Figure 7, these results show that a group’s LUCID is not fully reflected by the sum of its members’ individual LUCIDs, and that group importance itself varies across the remaining covariate context. Individual and group LUCID therefore provide complementary views of the learned model’s local predictive reliance. As shown throughout this section, the LUCID formulation and estimator accommodate both perspectives and support richer exploration of the fitted black-box model than marginal feature importance measures alone.

## F.4 Dispersion–Displacement Decomposition

We further examine the squared-loss decomposition of LUCID into its dispersion and displacement components in practice. In Figure 25, V denotes the dispersion term and D denotes the displacement term; observations with similar contributions from the two components appear near the balanced region. Averaged over all observations, the decomposition is nearly even, with $V = 5 1 . 3 \%$ and $D = 4 8 . 7 \%$ , consistent with Corollary C.2. Importantly, this near equality in the global average masks substantial observation-level heterogeneity in the mechanism driving local importance.

We use the hour feature as an example to show how to interpret the decomposition locally. For hour, which is only weakly predictable from the remaining features, conditional replacement leaves a relatively broad range of plausible hour values. The labeled examples in Figure 25 illustrate several qualitatively diferent regimes. At late-night observations such as example (1), the observed prediction can be far below the conditional replacement mean, producing a substantial displacement contribution. During the morning and evening commuting peaks on working days, as in example (2), the original model prediction is substantially higher than the conditional mean of the replacement predictions, so local importance is strongly displacement-driven. At similar hours on non-working days, the observed prediction is often closer to the conditional replacement mean, consistent with the less pronounced commuting pattern on non-working days and the remaining importance is driven more by dispersion across plausible replacements. Around midday, as illustrated by example (3), displacement is smaller, and variation across replacements contributes a larger share of the total LUCID. Thus, even for the same feature, a large LUCID can arise through diferent local mechanisms.

![](images/6b1cd16874a08d8fb7146c56a4116bbd92f450136a1a64e9e559b2979b447cb0.jpg)  
Figure 25: Dispersion–displacement decomposition of hour LUCID. Observation-level hour LUCID is decomposed into the dispersion term V and the displacement term D, the percentages indicate their respective shares of the mean LUCID across observations. Mean LUCID composition: V = 51.3%, D = 48.7% of the mean LUCID. Color indicates which component contributes more strongly, while the solid and dashed curves summarize working and non-working days, respectively.

## F.5 Conditional Replacement Distributions Across Methods

Finally, Figures 26–30 compare the conditional replacement distributions produced by the diferent sampling methods across season, working-day status, and hour periods. The goal here is not to identify a ground-truth conditional distribution, which is unavailable in the real data, but to illustrate how the diferent estimators behave under the same observed covariate contexts. DR-BART, ARF, cs-PFI, and GCMR generally preserve the support and qualitative structure of the observed features, whereas CMR can generate values outside the natural feature range because its residual-based construction is not intrinsically support-preserving; GCMR was designed in part to address this limitation for continuous features. Marginal permutation, by construction, does not adapt its replacement distribution to the conditioning context and therefore misses contextdependent distributional shifts. Diferences among the conditional methods are most visible for features such as temperature and humidity, whose replacement distributions vary appreciably across

![](images/4c599b3eb5b1cf76603948e661a90fe34e69d2b9c15fdee113123e6105ccde01.jpg)

![](images/8d9fbb7d2c268a2f345196e4539db669a08c982c2d3573ae69e786d78b974b0e.jpg)

(c) Hour CDE across hour period  
![](images/38e831c24315eced4252c87e5d226d75a2b99f0d2b57b4effb59e5aa7c380958.jpg)  
Figure 26: Conditional replacement distributions for hour. Replacement distributions produced by DR-BART, ARF, cs-PFI, CMR, GCMR, and marginal permutation across season, working-day status, and hour periods. Hour is weakly predictable from the remaining features, and its conditional replacement distribution is correspondingly broad and relatively stable across most contexts.

## season or hour period.

Note on visualization. For binary and categorical features, the violin shapes use smoothing only for display: the underlying samples from permutation, cs-PFI, DR-BART, and ARF remain on the valid categorical support. Each violin is normalized independently to the same maximum width, so violin widths are comparable only within a distribution and should not be interpreted as categorical probabilities across methods. Methods that generate small of-support continuous values, such as CMR and GCMR, are displayed without the discrete smoothing used for purely categorical samples. As a result, each local mode can appear visually narrower, so do not interpret the apparent width of individual bumps as lower categorical probability mass.

![](images/3ae1e3d6497ec7da1b91ad079b541172ddab7464858ba15dc9671ac1ae2681bc.jpg)

![](images/f8a37d4b798e6a45307d5402635e61066a67cebbf633fca0da7c8cfa1ad5cc1d.jpg)

![](images/1acd080fbde524fc4c744a2732c531c34c830d930cf72a5f4a61341b8221ab5e.jpg)  
Figure 27: Conditional replacement distributions for humidity. Replacement distributions across season, working-day status, and hour periods. Several conditional methods capture systematic changes in humidity across covariate contexts, particularly across hour periods, whereas marginal permutation uses the same unconditional distribution throughout. CMR also exhibits broader support than the naturally bounded humidity feature.

![](images/9e1f913e28a83e8ee349adf774d5802d1d84ea9ac0d6eddf5b468cd1fd449f94.jpg)

![](images/0203ee67c591c6280f9f42a32ef784832fdd6b441e803694debbd56aa16f22e5.jpg)

![](images/a54a40c5afa190a65388232f0d7c28b65510ce5e56e0f78a94dc1500ce574c84.jpg)  
Figure 28: Conditional replacement distributions for temperature. Replacement distributions across season, working-day status, and hour periods. The strongest distributional shifts occur across seasons, where conditional samplers adapt to the changing temperature distribution while marginal permutation remains fixed across contexts.

![](images/cc8b2bd5be4f45d479de88d39dc3b9f55c38ee2162dbd063dd89159a53c1e2d8.jpg)

![](images/164411574350eb8e037441c40cf69c51f6a5d8eeaab5b7c5ef94d82000629655.jpg)

![](images/8dffbac75a660870252a732247245be8b0821398acab5250b9c01f3d1daee998.jpg)  
Figure 29: Conditional replacement distributions for working day. Replacement distributions across season, working-day status, and hour periods. Most conditional methods preserve the binary structure of the feature, while CMR can place mass outside the natural {0, 1} support. The conditional distribution varies relatively little across the displayed contexts.

![](images/60aac62e256ead75cea22cf7a1533f632efc340fb940abe086fb57cd2bb452fe.jpg)

![](images/7d9863ca6aeac099b224579c82f2b2f55140e4f420d6d7f32a943a39704993a7.jpg)

![](images/0986e578e89a5f3ca054d57b7d6561bc3fd1ef22501dd861390985437cd8b486.jpg)  
Figure 30: Conditional replacement distributions for season. Replacement distributions across observed season, working-day status, and hour periods. Conditional methods preserve difering degrees of context dependence in the categorical replacement distribution, whereas marginal permutation does not condition on the observed context. CMR again produces values outside the natural categorical support.

## F.6 Runtime and Memory Scaling on OpenXAI Datasets

To further assess the computational feasibility of our DR-BART-based LUCID estimator beyond the five-feature Bike Sharing example, we benchmark runtime and peak memory usage on four tabular datasets from the OpenXAI (Agarwal et al., 2022) benchmark. These datasets provide complementary stress tests in both sample size and feature dimension: German contains the largest number of features with $p = 6 0$ , while GMSC contains more than $1 0 ^ { 5 }$ observations. Table 12 summarizes the datasets used in this experiment.

Table 12: OpenXAI benchmark datasets used in the scalability experiments.
<table><tr><td>Dataset</td><td>Samples (n)</td><td>Features (p)</td><td></td></tr><tr><td>Heart</td><td>3,658</td><td></td><td>15</td></tr><tr><td>German</td><td>1,000</td><td></td><td>60</td></tr><tr><td>HELOC</td><td>9,871</td><td></td><td>23</td></tr><tr><td>GMSC</td><td>102,209</td><td></td><td>10</td></tr></table>

Because LUCID computation is performed separately for each feature, we report runtime in two complementary ways. First, Table 13 reports the average runtime of one complete feature-specific task, obtained within each fold by dividing the total runtime across all features by the number of features. This normalizes for diferent feature dimensions and facilitates comparison across datasets. Second, Table 14 reports the total runtime per fold when all feature-specific jobs are executed serially. Each task includes any method-specific fitting and the computation of that method’s importance scores. For DR-BART, this includes conditional-sampler fitting and LUCID computation. All reported values are the mean and standard deviation across five folds. Feature-specific jobs are independent and can be executed concurrently, so the serial totals represent the computational cost in the absence of parallel resources, while actual wall-clock time can be substantially lower when multiple workers are available.

Table 13: Average runtime per feature-specific task in seconds (mean ± standard deviation across five folds). Each task includes conditional-sampler fitting and LUCID computation.
<table><tr><td>Method</td><td>Heart</td><td>German</td><td>HELOC</td><td>GMSC</td></tr><tr><td>Permutation</td><td> $8 . 8 1 \pm 1 . 1 6$ </td><td> $6 . 4 7 \pm 0 . 6 4$ </td><td> $1 5 . 6 7 \pm 2 . 2 1$ </td><td> $8 7 . 1 8 \pm 2 1 . 9 1$ </td></tr><tr><td>CMR</td><td> $5 . 8 9 \pm 0 . 0 3$ </td><td> $5 . 3 5 \pm 0 . 2 8$ </td><td> $1 4 . 9 9 \pm 3 . 3 7$ </td><td> $9 2 . 9 2 \pm 2 3 . 3 0$ </td></tr><tr><td>GCMR</td><td> $7 . 3 7 \pm 1 . 8 5$ </td><td> $6 . 8 4 \pm 0 . 1 4$ </td><td> $1 6 . 3 3 \pm 2 . 7 7$ </td><td> $1 0 6 . 1 8 \pm 2 4 . 7 9$ </td></tr><tr><td>local LOCO</td><td> $5 . 4 1 \pm 0 . 8 8$ </td><td> $4 . 9 7 \pm 0 . 1 5$ </td><td> $5 . 6 0 \pm 1 . 2 9$ </td><td> $4 . 7 0 \pm 0 . 1 8$ </td></tr><tr><td>CLIQUE</td><td> $7 . 6 5 \pm 1 . 6 8$ </td><td> $5 . 5 0 \pm 0 . 3 4$ </td><td> $1 4 . 6 1 \pm 2 . 8 7$ </td><td> $1 1 7 . 0 8 \pm 3 . 7 6$ </td></tr><tr><td>ARF</td><td> $4 3 . 8 4 \pm 5 . 8 8$ </td><td> $3 0 . 1 0 \pm 3 . 6 9$ </td><td> $2 8 7 . 2 0 \pm 2 5 . 1 1$ </td><td> $2 4 5 1 . 3 6 \pm 8 1 6 . 1 0$ </td></tr><tr><td> $\mathrm { c s - P F I }$ </td><td> $3 0 . 6 1 \pm 5 . 6 1$ </td><td> $9 . 7 0 \pm 0 . 1 2$ </td><td> $2 2 4 . 2 3 \pm 1 0 . 3 5$ </td><td> $1 0 0 3 6 . 0 2 \pm 5 3 0 . 4 5$ </td></tr><tr><td> $\mathrm { D R - B A R T }$ </td><td> $1 8 4 . 9 4 \pm 2 1 . 3 1$ </td><td> $3 6 . 9 6 \pm 0 . 2 9$ </td><td> $8 3 8 . 2 6 \pm 6 4 . 2 5$ </td><td> $9 0 1 5 . 1 4 \pm 1 4 1 9 . 1 6$ </td></tr></table>

Table 15 reports peak memory usage. For each fold, we take the largest peak memory among all feature-specific jobs, measured at 0.1-second intervals. Unlike runtime, peak memory is already a per-job quantity and therefore is not normalized by the number of features.

As expected, DR-BART is substantially more computationally demanding than simpler replacement schemes because it fits a flexible conditional density model separately for each feature.

Table 14: Serial-equivalent runtime per fold in minutes (mean ± standard deviation across five folds), obtained by summing all feature-specific jobs. This represents the runtime required when feature computations are not parallelized.
<table><tr><td>Method</td><td>Heart</td><td>German</td><td>HELOC</td><td>GMSC</td></tr><tr><td>Permutation</td><td> $2 . 2 0 \pm 0 . 2 9$ </td><td> $6 . 4 7 \pm 0 . 6 4$ </td><td> $6 . 0 1 \pm 0 . 8 5$ </td><td> $1 4 . 5 3 \pm 3 . 6 5$ </td></tr><tr><td>CMR</td><td> $1 . 4 7 \pm 0 . 0 1$ </td><td> $5 . 3 5 \pm 0 . 2 8$ </td><td> $5 . 7 5 \pm 1 . 2 9$ </td><td> $1 5 . 4 9 \pm 3 . 8 8$ </td></tr><tr><td>GCMR</td><td> $1 . 8 4 \pm 0 . 4 6$ </td><td> $6 . 8 4 \pm 0 . 1 4$ </td><td> $6 . 2 6 \pm 1 . 0 6$ </td><td> $1 7 . 7 0 \pm 4 . 1 3$ </td></tr><tr><td>local LOCO</td><td> $1 . 3 5 \pm 0 . 2 2$ </td><td> $4 . 9 7 \pm 0 . 1 5$ </td><td> $2 . 1 5 \pm 0 . 5 0$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td></tr><tr><td>CLIQUE</td><td> $1 . 9 1 \pm 0 . 4 2$ </td><td> $5 . 5 0 \pm 0 . 3 4$ </td><td> $5 . 6 0 \pm 1 . 1 0$ </td><td> $1 9 . 5 1 \pm 0 . 6 3$ </td></tr><tr><td>ARF</td><td> $1 0 . 9 6 \pm 1 . 4 7$ </td><td> $3 0 . 1 0 \pm 3 . 6 9$ </td><td> $1 1 0 . 0 9 \pm 9 . 6 3$ </td><td> $4 0 8 . 5 6 \pm 1 3 6 . 0 2$ </td></tr><tr><td>cs-PFI</td><td> $7 . 6 5 \pm 1 . 4 0$ </td><td> $9 . 7 0 \pm 0 . 1 2$ </td><td> $8 5 . 9 5 \pm 3 . 9 7$ </td><td> $1 6 7 2 . 6 7 \pm 8 8 . 4 1$ </td></tr><tr><td>DR-BART</td><td> $4 6 . 2 3 \pm 5 . 3 3$ </td><td> $3 6 . 9 6 \pm 0 . 2 9$ </td><td> $3 2 1 . 3 3 \pm 2 4 . 6 3$ </td><td> $1 5 0 2 . 5 2 \pm 2 3 6 . 5 3$ </td></tr></table>

Table 15: Peak memory usage in GiB (mean ± standard deviation across five folds). For each fold, the maximum peak memory among feature-specific jobs is reported.
<table><tr><td>Method</td><td>Heart</td><td>German</td><td>HELOC</td><td>GMSC</td></tr><tr><td>Permutation</td><td> $0 . 3 3 8 \pm 0 . 0 0 1$ </td><td> $0 . 3 2 9 \pm 0 . 0 0 0$ </td><td> $0 . 3 5 2 \pm 0 . 0 0 0$ </td><td> $0 . 3 6 5 \pm 0 . 0 0 4$ </td></tr><tr><td>CMR</td><td> $0 . 3 3 5 \pm 0 . 0 0 0$ </td><td> $0 . 3 2 7 \pm 0 . 0 0 2$ </td><td> $0 . 3 5 4 \pm 0 . 0 0 3$ </td><td> $0 . 3 7 1 \pm 0 . 0 0 2$ </td></tr><tr><td>GCMR</td><td> $0 . 3 3 9 \pm 0 . 0 0 3$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 1$ </td><td> $0 . 3 5 8 \pm 0 . 0 0 1$ </td><td> $0 . 3 7 5 \pm 0 . 0 0 5$ </td></tr><tr><td>local LOCO</td><td> $0 . 3 2 0 \pm 0 . 0 0 3$ </td><td> $0 . 3 2 0 \pm 0 . 0 0 2$ </td><td> $0 . 3 2 5 \pm 0 . 0 0 3$ </td><td> $0 . 3 4 1 \pm 0 . 0 0 0$ </td></tr><tr><td>CLIQUE</td><td> $0 . 3 3 6 \pm 0 . 0 0 3$ </td><td> $0 . 3 2 5 \pm 0 . 0 0 2$ </td><td> $0 . 3 5 1 \pm 0 . 0 0 4$ </td><td> $0 . 3 6 7 \pm 0 . 0 0 1$ </td></tr><tr><td>ARF</td><td> $0 . 9 9 6 \pm 0 . 0 0 6$ </td><td> $2 . 0 7 2 \pm 0 . 0 3 0$ </td><td> $1 . 7 0 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 5 \pm 0 . 0 1 0$ </td></tr><tr><td>cs-PFI</td><td> $0 . 7 5 8 \pm 0 . 0 2 1$ </td><td> $0 . 5 6 2 \pm 0 . 0 0 3$ </td><td> $2 . 2 2 6 \pm 0 . 0 9 4$ </td><td> $4 0 . 5 4 8 \pm 0 . 6 2 0$ </td></tr><tr><td>DR-BART</td><td> $1 . 2 4 5 \pm 0 . 0 0 1$ </td><td> $1 . 3 5 3 \pm 0 . 0 0 1$ </td><td> $0 . 8 0 8 \pm 0 . 0 1 3$ </td><td> $4 1 . 8 5 4 \pm 0 . 0 3 1$ </td></tr></table>

Nevertheless, the complete LUCID pipeline remains computationally feasible across all four benchmark datasets. In particular, we successfully run DR-BART on GMSC with 102,209 observations. The average feature-specific task requires approximately 9,015 seconds and the largest feature-specific job uses approximately 41.9 GiB of memory. If all ten feature-specific jobs are executed serially, one fold requires approximately 1,503 minutes, or about 25 hours of aggregate computation.

Remark 1. Local explanations are often required at prediction time. While training DR-BART itself requires expensive Markov Chain Monte Carlo (MCMC) iterations, the DR-BART model is fixed at prediction time and runtime for explanations depends only on time to sample posterior draws, counterfactuals for each posterior, and fitting/comparing the fixed predictive models. In contrast, other local explanation methods like LIME can require refitting new surrogate models for each new test observation, increasing explanation-inference time.