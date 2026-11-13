# Flow-CAD: Flow Matching for Contextual Anomaly Detection and Explanation in Tabular Data

Zhong Li<sup>1,2</sup>, Qi Huang<sup>2</sup>, Yuhang Wang<sup>3</sup>, Yang Cao<sup>1</sup>,

Thomas Bäck<sup>2</sup>, Niki van Stein<sup>2</sup>, Matthijs van Leeuwen<sup>2</sup>, Lincen Yang<sup>2,\*</sup>

<sup>1</sup>Great Bay University, Dongguan, China

<sup>2</sup>Leiden University, Leiden, The Netherlands

<sup>3</sup>KU Leuven, Leuven, Belgium

Corresponding author: Lincen Yang (l.yang@liacs.leidenuniv.nl)

Abstract—Contextual anomaly detection in tabular data aims to identify objects whose behavior is unexpected under their contextual conditions. In data-engineering pipelines, this task screens table rows whose measurements violate context-dependent integrity relationships. Most existing approaches rely on either shallow conditional models with limited distributional expressiveness or localized estimators that require per-context modeling, making it difficult to capture complex conditional relationships and scale to high-dimensional settings. We propose Flow-CAD, a new framework for contextual anomaly detection based on Flow Matching. Flow-CAD learns a single global and amortized conditional generative model that flexibly represents behavioral distributions conditioned on context, without assuming simple parametric forms or retraining local models. To enable robust and interpretable anomaly scoring, we introduce a scale normalized and truncated residual energy derived from the learned conditional flow, which stabilizes anomaly scores, mitigates domination by extreme deviations, and supports feature-wise attribution. We further provide theoretical insights showing that the proposed residual-based anomaly score is stable under velocity estimation error, and that its Monte Carlo estimator satisfies concentration bounds. Experiments on 20 benchmark datasets with 13 competitors demonstrate that Flow-CAD has the best average rank for PR AUC, ROC AUC, and P@n, highlighting its effectiveness and practicality for contextual anomaly detection in tabular data. We provide the source code at: https://github.com/ ZhongLIFR/FlowCAD.

## I. INTRODUCTION

Contextual anomaly detection aims to identify data instances whose behavior is abnormal when interpreted relative to their contextual conditions [1, 2]. Under this setting, features are typically partitioned into contextual variables (e.g., operating conditions, demographic attributes, or environmental factors) and behavioral variables (e.g., measurements, responses, or outcomes). An instance is considered anomalous not because it is globally rare, but because its behavior is unlikely given its context. This setting arises naturally in a wide range of applications, including industrial monitoring [3], environmental analysis [4], and security in smart grids [5], where identical behaviors may be normal in one context but anomalous in another. For example, a plant may record its operating mode alongside pressure and temperature readings in each row. Each reading can be typical for that mode on its own, while the combination is unusual. In database-backed analytics, screening for such rows can precede aggregation, monitoring, and model training [6, 7]. For a flagged row, Flow-CAD reports how much each behavioral column contributes to the anomaly score, helping analysts decide which measurements to inspect [8].

Despite steady progress, existing contextual anomaly detection methods continue to face several fundamental limitations when applied to complex tabular data. Early generic frameworks such as CAD [1] rely on parametric assumptions and handcrafted mapping functions to capture dependencies between contextual and behavioral variables, which restrict flexibility, scalability, and support for mixed feature types. A subsequent approach ROCOD [9] combines global regression models with local neighborhood statistics, but primarily characterizes conditional behavior through expectations, which can be inadequate when the conditional distribution is skewed, multimodal, or context-dependent. A recent method QCAD [2] improves robustness and interpretability by estimating conditional quantiles, yet remains fundamentally localized, relying on per-query neighborhood construction or local conditional estimation. This repeated local estimation adds scoring overhead (see Figure 8). Together, these limitations motivate a global and amortized approach that avoids repeated local modeling, captures complex context-dependent behavior, and supports efficient inference with interpretable anomaly scores.

Recent advances in deep generative modeling [10, 11] provide an opportunity to revisit contextual anomaly detection from a global and amortized perspective. In principle, conditional generative models are well suited for representing complex conditional distributions without relying on restrictive parametric assumptions. For continuous flows, likelihood evaluation typically requires numerical integration of an ordinary differential equation (ODE) and estimation of the velocity field’s divergence. Likelihood values can also be unreliable as a direct anomaly signal in complex distributions [12, 13]. As a result, existing generative approaches do not naturally satisfy the combined requirements of efficient inference, robust anomaly scoring, and interpretability in contextual anomaly detection. Flow Matching [14, 15] offers an alternative training paradigm for continuous generative models that avoids explicit likelihood evaluation and ODE-based density computation during training. By learning a conditional velocity field that transports samples from a simple base distribution to the data, Flow Matching retains the expressive power of conditional generative models while enabling path-wise, residual-based assessments of how well observed behaviors are explained under a given context. Motivated by these properties, we propose Flow-CAD, a new framework for contextual anomaly detection based on Flow Matching.

Specifically, Flow-CAD learns a single global and amortized conditional generative model that captures the joint distribution of behavioral variables conditioned on context. Instead of relying on explicit likelihood evaluation, Flow-CAD derives anomaly scores from residuals of the learned conditional flow, measuring how well the model explains an observed behavior given its context. To obtain robust and interpretable anomaly scores, we further introduce a scale-normalized and truncated residual energy that aggregates deviations across behavioral dimensions while limiting the influence of extreme values. This construction yields native, decomposable feature-level attributions, which identify the behavioral columns contributing to a flagged row’s score. In addition, we relate the ideal-field residual to conditional denoising, establish the stability of the normalized and truncated score under velocity estimation error, and analyze the variance and concentration of its nested Monte Carlo estimator.

Contributions. The main contributions of this work are as follows. (1) We introduce Flow-CAD, a contextual anomaly detection framework based on Flow Matching for screening database tables. A single model learns how the behavioral columns vary together given the context columns. (2) We propose a residual-based anomaly energy for conditional flows that scores rows in batches without computing likelihoods or refitting for each query. A data engineering pipeline can use these scores to flag records before aggregation or model training. (3) We develop a normalized and truncated aggregation scheme that yields robust scores and additive column contributions. For a flagged row, these contributions show which columns account for most of its score, helping analysts decide which values to inspect first. (4) We provide theoretical analyses and compare Flow-CAD with 13 baselines on 20 benchmark tables, using a fixed training budget for our model. In a separate test on held-out synthetic rows, we ask whether the model can detect a context mismatch that only becomes visible when three behavioral columns are considered together. We also compare runtime with QCAD on full tables and report batch scoring time and peak memory on synthetic inputs of up to one million rows.

## II. RELATED WORK

Contextual Anomaly Detection. Anomaly detection in tabular data has been extensively studied, with prior work commonly distinguishing between point, contextual, and collective anomalies [16, 17]. This work focuses on contextual anomaly detection, where an instance is considered anomalous if its behavior is unlikely given its contextual attributes. Early methods such as CAD [1] use probabilistic mixtures, while subsequent approaches explore probabilistic and correlationbased formulations such as ROCOD [9] and LoPAD [18]. More recent methods emphasize interpretability, with QCAD [2] modeling conditional distributions via quantile regression forests and deriving feature-level explanations from conditional deviations. RFOD [19] predicts each feature from the remaining features, and Normalcy Score [20] quantifies uncertainty for a scalar behavioral target. Local models and per-context estimation can incur repeated computation at scoring time.

Generative Models for Anomaly Detection. Generative approaches to anomaly detection often use low likelihood as a signal of abnormality. Classical density models and, more recently, deep generative approaches have been explored for tabular anomaly detection [10, 11, 21]. Normalizing flows further enable exact likelihood evaluation through invertible transformations [22], but their practical application is often constrained by architectural design and the cost of likelihood computation in high-dimensional settings. Flow Matching [14, 15, 23, 24] has recently emerged as a scalable alternative for training continuous normalizing flows without explicit likelihood or ODE-based density evaluation. Building on this paradigm, TCCM [25] uses a single contraction residual toward the origin for point anomaly detection. Flow Mismatching [26] aggregates residuals over Gaussian-to-image paths for visual defects, fitting on normal images. On contaminated tables, Konz et al. [27] compare TCCM and Forest-Flow, whose Deviation score also pools velocity residuals over time and noise. Flow CAD conditions the joint behavioral field on separate context columns and uses an additive score with median absolute deviation (MAD) scaling and clipping.

Explainability. Explainability is increasingly recognized as a critical requirement in anomaly detection, particularly for tabular data where decisions often have high practical impact. Recent surveys [28, 29] distinguish between post-hoc and intrinsic explanation methods, and highlight the importance of feature-level attribution in practical deployments. Flow-CAD contributes to this line of work by providing intrinsic and decomposable anomaly explanations derived directly from the coordinate terms of its residual score, without relying on posthoc explanation models.

## III. FLOW-CAD: CONDITIONAL FLOW-BASED CONTEXTUAL ANOMALY DETECTION

## A. Problem Setup and Method Overview

We study contextual anomaly detection in tabular data. Each instance is represented by a pair $( c , b )$ , where $\boldsymbol { c } \in \mathbb { R } ^ { P }$ denotes the encoded context vector and $\mathbf { \boldsymbol { b } } \in \mathbb { R } ^ { Q }$ denotes behavioral features. Given a table $\{ ( c _ { i } , b _ { i } ) \} _ { i = 1 } ^ { N } ,$ , our goal is to prioritize rows whose behavior conflicts with patterns expected under their context. We denote the corresponding conditional behavior distribution by $p ( \pmb { b } \mid \pmb { c } )$

Flow Matching as a modeling primitive. Flow Matching trains a continuous-time velocity field by regressing pathwise transport targets along prescribed interpolation paths [14, 24]. Its population optimum is the conditional mean target given the intermediate state, context, and time. The regression objective requires neither density evaluation nor ODE integration. Flow-CAD learns one conditional velocity field $f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t )$ for the behavior distribution $p ( \pmb { b } \mid \pmb { c } )$ and reuses the field across contexts.

![](images/febdca3ef2d9454b6966d9b406676a10d4843382c664dd059c6f4759e7013587.jpg)  
Fig. 1. Illustration of the training (left) and inference (right) phases of Flow-CAD. The inference panel writes $t ^ { ( \ell ) }$ for $t ^ { ( k , \ell ) }$ within Gaussian block k.

Method Overview. Figure 1 illustrates the overall pipeline of Flow-CAD, which consists of two phases: (1) Training Phase: Conditional Flow Matching (Section III-B), in which we train a context-conditioned neural velocity field via Flow Matching along a linear interpolation path between a Gaussian base sample and the observed behavioral vector. This yields a single global model that captures conditional behavior across all contexts; (2) Inference Phase: Residual-based contextual anomaly scoring (Section III-C), in which we quantify contextual abnormality by the discrepancy between the modelpredicted velocity and the true velocity induced by an observed instance, aggregated over multiple noise and time samples. To ensure robustness and interpretability, residuals are normalized using the median absolute deviation (MAD) and truncated to mitigate the dictator effect where a single extreme deviation dominates the score. It also yields additive output-coordinate terms for inspecting flagged rows.

## B. Phase 1: Training Detection Model using Flow Matching with Linear Interpolation

Let $b ~ \in ~ \mathbb { R } ^ { Q }$ denote behavioral features and $c ~ \in ~ \mathbb { R } ^ { P }$ denote encoded contextual features. We introduce a conditional velocity field $f _ { \theta } : \mathbb { R } ^ { Q } \times \mathbb { R } ^ { P } \times [ 0 , 1 ] \to \mathbb { R } ^ { Q } , \quad ( x _ { t } , c , t ) \mapsto \pi$ $f _ { \theta } ( \pmb { x } _ { t } , \pmb { c } , t )$ , parameterized by a neural network with learnable parameters θ. This velocity field is trained via Flow Matching on a simple linear path [23] between a base distribution and the data in the behavioral space.

Linear path. Let $p _ { 0 }$ be the density of $\mathcal { N } ( \mathbf { 0 } , I _ { Q } )$ , where $I _ { Q }$ is the Q-dimensional identity. For $z \sim p _ { 0 }$ and $t \in [ 0 , 1 ]$ , the path to b is $\pmb { x } _ { t } = ( 1 - t ) \pmb { z } + t \pmb { b }$ . For each sampled endpoint–noise pair, the pathwise target velocity is $\begin{array} { r } { { \pmb v } _ { t } = \frac { d { \pmb x } _ { t } } { d t } = { \pmb b } - z } \end{array}$ , which is independent of t.

Flow Matching objective. We train $f _ { \theta }$ so that it approximates this velocity along the path. We write the Flow Matching objective in squared-norm form as

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { ( \boldsymbol { b } , \boldsymbol { c } ) , \boldsymbol { z } , t } \left[ \big \| f _ { \boldsymbol { \theta } } ( \boldsymbol { x } _ { t } , \boldsymbol { c } , t ) - ( \boldsymbol { b } - \boldsymbol { z } ) \big \| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{1}
$$

We estimate this expectation over minibatches, drawing $z \sim p _ { 0 }$ and $t \sim \mathrm { U n i f } ( 0 , 1 )$ independently for each row. The implementation averages over behavioral coordinates as well, multiplying this objective by $1 / Q$ without changing its minimizer.

Context encoder and architecture. In practice, $f _ { \theta }$ is implemented as a context-conditional multi-layer perceptron (MLP): $c _ { \mathrm { e m b } } = g _ { \phi } ( \pmb { c } )$ , and $f _ { \theta } ( \mathbf { x } _ { t } , \mathbf { c } , t ) = h _ { \psi } \left( \left[ \mathbf { x } _ { t } , \ \mathbf { c } _ { \mathrm { e m b } } , \ t \right] \right)$ where $g _ { \phi }$ and $h _ { \psi }$ are MLPs, and $[ \cdot , \cdot , \cdot ]$ denotes concatenation. The embedding $\mathbf { \boldsymbol { c } } _ { \mathrm { e m b } } \in \mathbb { R } ^ { d }$ has width d. Categorical contextual features are one-hot encoded and then fed into $g _ { \phi }$ together with numerical contextual features, so that the model can handle mixed-type c in a unified way. The parameters $\theta = \left( \phi , \psi \right)$ are optimized jointly via Eq. (1) using Adam [30].

Conditional likelihood. Under the usual regularity conditions for a conditional continuous normalizing flow [14], its log-density is evaluated along the model ODE trajectory:

$$
\log p _ { \theta } ( \pmb { b } \mid \pmb { c } ) = \log p _ { 0 } ( \pmb { y } _ { 0 } ) - \int _ { 0 } ^ { 1 } \mathrm { d i v } _ { \pmb { y } } f _ { \theta } ( \pmb { y } _ { t } , \pmb { c } , t ) d t ,\tag{2}
$$

Here $\dot { \pmb y } _ { t } = f _ { \theta } ( \pmb { y } _ { t } , \pmb { c } , t )$ is integrated backward from $\mathbf { \Psi } _ { \mathbf { { y } } _ { 1 } } = \mathbf { \Psi } _ { b }$ This model ODE trajectory generally differs from the sampled straight interpolation paths. Evaluating (2) requires solving the ODE and integrating the divergence of $f _ { \theta }$ along its trajectory. Flow-CAD scores rows through batched velocityfield evaluations without computing the density.

C. Phase 2: Residual-Based Anomaly Scoring with MAD Normalization and Truncation

Conditional denoising interpretation. Flow Matching regresses $b - z$ on state, context, and time. The induced reconstruction is $m _ { \theta } ( { \pmb x } , { \pmb c } , t ) = { \pmb x } + ( 1 - t ) f _ { \theta } ( { \pmb x } , { \pmb c } , t )$ . Along the sampled path $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , its error relative to b is $( 1 - t )$ times the signed velocity residual. At the population optimum the reconstruction equals the posterior mean. The residual for an individual endpoint may remain nonzero. Proposition 1 relates field estimation error to our clipped conditional score.

Raw residuals. Given a fitted field $f _ { \theta }$ and an observation $( \boldsymbol { c } ^ { * } , \boldsymbol { b } ^ { * } )$ , draw independent $z ^ { ( k ) } \sim p _ { 0 }$ for $k = 1 , \dots , K _ { z }$ . For each k, draw fresh times $t ^ { ( k , \ell ) } \sim \operatorname { U n i f } ( 0 , 1 ) , \ell = 1 , \dots , K _ { t }$ All time draws are mutually independent and independent of the Gaussian draws; the same $\tilde { z ^ { ( k ) } }$ is reused only within its inner time loop. Set

$$
\begin{array} { c } { { { \pmb x } ^ { ( k , \ell ) } = ( 1 - t ^ { ( k , \ell ) } ) { \pmb z } ^ { ( k ) } + t ^ { ( k , \ell ) } { \pmb b } ^ { * } , } } \\ { { { \pmb v } ^ { ( k ) } = { \pmb b } ^ { * } - { \pmb z } ^ { ( k ) } . } } \end{array}
$$

Taking the absolute value coordinatewise gives

$$
\begin{array} { r } { \pmb { r } ^ { ( k , \ell ) } = \left| f _ { \theta } ( \pmb { x } ^ { ( k , \ell ) } , \pmb { c } ^ { * } , t ^ { ( k , \ell ) } ) - \pmb { v } ^ { ( k ) } \right| . } \end{array}
$$

Robust per-dimension scaling via MAD. Behavioral dimensions can exhibit very different magnitudes and intrinsic variability. Directly aggregating raw residuals would therefore bias the anomaly score toward features with larger natural scales. To account for this, we normalize residuals using a robust, per-dimension scale estimate based on the median absolute deviation (MAD). For each behavioral dimension $q \in \{ 1 , \ldots , Q \}$ , we compute me $\mathrm { d } _ { q } \ = \ \mathrm { m e d i a n } _ { i } \left( b _ { i q } \right)$ , and $\mathrm { M A D } _ { q } \ =$ median $_ i \left( | b _ { i q } - \mathrm { m e d } _ { q } | \right)$ , where $b _ { i q }$ denotes the value of the q-th behavioral feature of the i-th training instance. We floor the scale at $\varepsilon _ { \mathrm { M A D } } ~ = ~ 1 0 ^ { - 6 }$ and write $d _ { q } = \mathrm { m a x } ( \mathrm { M A D } _ { q } , \varepsilon _ { \mathrm { M A D } } )$ . The normalized residual magnitude is $\tilde { r } _ { q } ^ { ( k , \ell ) } = \dot { r _ { q } } ^ { ( k , \ell ) } / \dot { d } _ { q }$ . Section IV-E ablates this scaling.

Truncation to avoid dictator effect. If we simply sum the normalized residuals across behavioral dimensions, a single extremely deviant dimension could dominate the anomaly score (namely the dictator effect, also observed in [2]). To mitigate this, we truncate each normalized residual at a fixed threshold $\tau > 0 ; \hat { r } _ { q } ^ { ( k , \ell ) } = \operatorname* { m i n } \left( \tilde { r } _ { q } ^ { ( k , \ell ) } , \tau \right)$ , with $q = 1 , \ldots , Q$ . In our experiments we use $\tau \stackrel { \cdot } { = } 1 0$ (see Section IV-E for a sensitivity analysis on setting of this value), which can be interpreted as an upper bound on the number of “typical MAD units” that a single dimension is allowed to contribute. We then aggregate across dimensions as follows:

$$
s ^ { ( k , \ell ) } = \sum _ { q = 1 } ^ { Q } \hat { r } _ { q } ^ { ( k , \ell ) } .\tag{3}
$$

This scalar $s ^ { ( k , \ell ) }$ is the truncated, normalized residual energy for a single pair $( z ^ { ( k ) } , t ^ { ( k , \ell ) } )$

Monte Carlo averaging. The truncated residual energy $s ^ { ( k , \ell ) }$ depends on both the sampled base noise z and the interpolation time t. A single realization may therefore be noisy and sensitive to stochastic variation along the transport path. To obtain a stable anomaly score that reflects the expected mismatch between the learned conditional flow and the observed behavior, we estimate the expectation over $( z , t )$ by Monte Carlo averaging:

$$
s _ { \mathrm { r a w } } ( \pmb { b } ^ { * } \mid \pmb { c } ^ { * } ) = \frac { 1 } { K _ { z } K _ { t } } \sum _ { k = 1 } ^ { K _ { z } } \sum _ { \ell = 1 } ^ { K _ { t } } s ^ { ( k , \ell ) } .\tag{4}
$$

We use $K _ { z } = K _ { t } = 1 6 .$ , giving 256 field evaluations in 16 independent Gaussian-noise blocks. Proposition 2 establishes unbiasedness and quantifies the distinct effects of $K _ { z }$ and $K _ { t }$ on Monte Carlo variance. Section IV-E evaluates the samplingbudget trade-off empirically.

Ranking and column contributions. The raw anomaly scores $s _ { \mathrm { r a w } } ( b \mid c )$ do not have an intrinsic probabilistic scale and may vary across datasets and sampling configurations. We compute detection metrics from the raw scores. Percentile ranks preserve the within-table ordering. Because the anomaly energy in (4) is constructed from dimension-wise residuals, Flow-CAD naturally provides per-feature contributions: $s _ { q } ( b ^ { * } \mid c ^ { * } ) =$ $\textstyle \frac { 1 } { K _ { z } K _ { t } } \sum _ { k , \ell } \hat { r } _ { q } ^ { ( k , \ell ) }$ . The terms sum to the raw score in (4). Each output residual can depend on every behavioral input through the joint field. The score is decomposed by output coordinate.

## D. Theoretical Insights

A row’s anomaly score depends on the velocity field and the random draws used to compute it. Proposition 1 compares the expected scores from the fitted and optimal fields using the same MAD scales and truncation threshold. It bounds the change for an individual row and relates the average absolute change at a fixed context to the excess Flow Matching loss. With the fitted field and scoring settings fixed, Proposition 2 bounds the error from estimating the expected score with a finite number of noise and time draws.

With $d _ { q }$ from Section III-C and $0 < \tau < \infty .$ , define

$$
\rho ( { \pmb u } ) = \sum _ { q = 1 } ^ { Q } \mathrm { m i n } \{ | u _ { q } | / d _ { q } , \tau \} , \qquad \kappa = \left( \sum _ { q = 1 } ^ { Q } d _ { q } ^ { - 2 } \right) ^ { 1 / 2 } .\tag{5}
$$

For any field $f ,$ its expected raw score at a fixed instance is

$$
\begin{array} { r } { \mu _ { f } ( \pmb { b } \mid \pmb { c } ) = \operatorname { \mathbb { E } } _ { \pmb { Z } , T } \rho ( f ( \pmb { x } _ { T } , \pmb { c } , T ) - ( \pmb { b } - \pmb { Z } ) ) , } \\ { \pmb { x } _ { T } = ( 1 - T ) \pmb { Z } + T \pmb { b } , \qquad } \end{array}\tag{6}
$$

where $z \sim p _ { 0 }$ and $T \sim \mathrm { U n i f } ( 0 , 1 )$ are independent. For $f = f _ { \theta }$ , this is the expected raw score in (4).

Proposition 1 (Population optimality and stability of residual scoring). Fix a context c. Let $\begin{array} { r } { B \sim p ( \cdot \mid c ) , Z \sim p _ { 0 } } \end{array}$ , and $T \sim$ $\mathrm { U n i f } ( 0 , 1 )$ be independent, with E $\| B \| _ { 2 } ^ { 2 } < \infty$ . Here $p ( \cdot \mid c )$ is the conditional law of the training endpoints, which can include anomalies. Set $\pmb { X } _ { t } = ( 1 - t ) \pmb { Z } + t \pmb { B }$ and ${ \cal V } = B - Z$ Define the population risk and its optimal regression field by

$$
\begin{array} { r } { \mathcal { L } _ { c } ( f ) = \mathbb { E } \| f ( X _ { T } , \pmb { c } , T ) - V \| _ { 2 } ^ { 2 } , } \\ { f ^ { \star } ( \pmb { x } , \pmb { c } , t ) = \mathbb { E } [ \pmb { V } \mid X _ { T } = \pmb { x } , T = t ] . } \end{array}\tag{7}
$$

Let $f _ { \theta }$ be measurable with $\mathcal { L } _ { c } ( f _ { \theta } ) < \infty .$

(i) Population risk decomposition.

$$
\begin{array} { r l } & { \mathcal { L } _ { c } ( f _ { \theta } ) = \mathcal { L } _ { c } ( f ^ { \star } ) + \mathbb { E } \| f _ { \theta } ( X _ { T } , c , T ) - f ^ { \star } ( X _ { T } , c , T ) \| _ { 2 } ^ { 2 } , } \\ & { \mathcal { L } _ { c } ( f ^ { \star } ) = \mathbb { E } \ \mathrm { t r } { \mathrm { C o v } ( V \mid X _ { T } , T ) } . } \end{array}\tag{8}
$$

The expectations here include the random endpoint B. The conditional covariance can vary with the intermediate state and time.

(ii) Conditional denoising interpretation. For $0 \leq t < 1$ , let $m ^ { \star } ( { \pmb x } , { \pmb c } , t ) = \mathbb { E } [ { \pmb B } \mid { \pmb X } _ { T } = { \pmb x } , T = t ] .$ . At a fixed test endpoint $^ { b , }$ with $\begin{array} { r } { \pmb { x } _ { t } = ( 1 - t ) \pmb { z } + t \pmb { b } , } \end{array}$ the posterior version of $f ^ { \star }$ satisfies

$$
\begin{array} { l } { f ^ { \star } ( x _ { t } , c , t ) - ( b - z ) } \\ { \quad = \frac { m ^ { \star } ( x _ { t } , c , t ) - b } { 1 - t } . } \end{array}\tag{9}
$$

Thus $\mu _ { f } ,$ is the expectation of the dimension-wise scaled and truncated denoising residual in (9).

(iii) Stability of the implemented robust score. Define the pointwise velocity estimation error

$$
\begin{array} { r } { D _ { \theta } ( \pmb { b } \mid \pmb { c } ) = \mathbb { E } _ { Z , T } \Vert f _ { \theta } ( \pmb { x } _ { T } , \pmb { c } , T ) - f ^ { \star } ( \pmb { x } _ { T } , \pmb { c } , T ) \Vert _ { 2 } ^ { 2 } . } \end{array}\tag{10}
$$

Whenever this quantity is finite,

$$
| \mu _ { f _ { \theta } } ( b \mid c ) - \mu _ { f ^ { \star } } ( b \mid c ) | \leq \kappa \sqrt { D _ { \theta } ( b \mid c ) } .\tag{11}
$$

In particular, averaging over $\textstyle B \sim p ( \cdot | c )$ gives

$$
\begin{array} { r } { \mathbb { E } _ { B } | \mu _ { f _ { \theta } } ( B \mid c ) - \mu _ { f ^ { \star } } ( B \mid c ) | } \\ { \leq \kappa \sqrt { \mathscr { L } _ { c } ( f _ { \theta } ) - \mathscr { L } _ { c } ( f ^ { \star } ) } . } \end{array}\tag{12}
$$

From training loss to row scores. Part (i) splits the average training loss into the loss of the optimal field and the error from estimating it. For a fixed row, the cross term can remain, so the decomposition does not determine the row’s rank. Part (ii) explains why the optimal field can still leave a residual: the posterior mean can differ from the observed behavior.

Part (iii) bounds how much using the fitted field changes each row’s expected score. For rows i and $j$ scored with the same MAD scales, let $\Delta ^ { \star }$ be the expected score of i minus that of $j$ under the optimal field. The fitted field still gives i a higher expected score if

$$
\Delta ^ { \star } > \kappa ( \sqrt { D _ { i } } + \sqrt { D _ { j } } ) .
$$

The right-hand side adds the two bounds in (11), with velocity estimation errors $D _ { i }$ and $D _ { j }$ from (10). The average training loss does not determine this gap or how it changes with training duration or network size. In the one-dimensional Gaussian example in Section III-D1, even the optimal field gives higher expected scores to observations farther from the mean.

Proposition 2 (Nested Monte Carlo scoring). Fix (c, b), f<sub>θ</sub>, the scales $d _ { q } ,$ and τ before drawing the scoring samples. Write $s ( z , t ) = \rho ( f _ { \theta } ( x _ { t } , c , t ) - ( b - z ) ) , \mu = \mu _ { f _ { \theta } } ( b \mid c ) ,$ , and $H =$ $Q \tau .$ . For integers $K _ { z } , K _ { t } \ge 1$ , draw $Z _ { k }$ independently from p<sub>0</sub> and draw all $T _ { k \ell }$ independently from Unif(0, 1), independently of the Gaussian draws. Define

$$
A _ { k } = \frac { 1 } { K _ { t } } \sum _ { \ell = 1 } ^ { K _ { t } } s ( \pmb { Z } _ { k } , T _ { k \ell } ) , \qquad \widehat { \mu } = \frac { 1 } { K _ { z } } \sum _ { k = 1 } ^ { K _ { z } } A _ { k } .\tag{13}
$$

For the same draws in (4), ${ \widehat { \mu } } = s _ { \mathrm { r a w } } ( b \mid c )$ . Let

$$
\begin{array} { r l } & { \bar { s } ( z ) = \mathbb { E } _ { T } s ( z , T ) , } \\ & { \quad \sigma _ { z } ^ { 2 } = \mathrm { V a r } _ { Z } \bar { s } ( Z ) , } \\ & { \quad \sigma _ { t } ^ { 2 } = \mathbb { E } _ { Z } \mathrm { V a r } ( s ( Z , T ) \mid Z ) . } \end{array}\tag{14}
$$

The estimator is unbiased and has exact variance

$$
\mathbb { E } \widehat { \mu } = \mu , \qquad \mathrm { V a r } ( \widehat { \mu } ) = \frac { \sigma _ { z } ^ { 2 } } { K _ { z } } + \frac { \sigma _ { t } ^ { 2 } } { K _ { z } K _ { t } } .\tag{15}
$$

Since the block means $A _ { k }$ are independent and lie in $[ 0 , H ] ,$ for every $\epsilon > 0$

$$
\mathrm { P r } ( | \widehat { \mu } - \mu | \geq \epsilon ) \leq 2 \exp \left( - \frac { 2 K _ { z } \epsilon ^ { 2 } } { H ^ { 2 } } \right) .\tag{16}
$$

Equivalently, for $0 < \delta < 1$ , with probability at least $1 - \delta ,$

$$
\vert \widehat { \mu } - \mu \vert \leq H \sqrt { \frac { \log ( 2 / \delta ) } { 2 K _ { z } } } .\tag{17}
$$

A variance-sensitive bound, with $\nu _ { K _ { t } } = \sigma _ { z } ^ { 2 } + \sigma _ { t } ^ { 2 } / K _ { t } ,$ is

$$
\operatorname* { P r } ( | \widehat { \mu } - \mu | \geq \epsilon ) \leq 2 \exp \left( - \frac { K _ { z } \epsilon ^ { 2 } } { 2 \nu _ { K _ { t } } + 2 H \epsilon / 3 } \right) .\tag{18}
$$

When $\nu _ { K _ { t } } = 0 ,$ , the estimator equals $\mu$ almost surely.

Choosing the number of samples. For a fixed row, each Gaussian draw defines one path. Equation (15) separates variation across paths from variation across sampled times on the same path. Taking more time samples $( K _ { t } )$ gives a better average along each path, but leaves the variance term $\sigma _ { z } ^ { 2 } / K _ { z }$ unchanged. Drawing more paths $( K _ { z } )$ reduces both terms, so the estimate converges to the expected score in mean square even with $K _ { t }$ fixed.

The default $K _ { z } = K _ { t } = 1 6$ uses 16 independent paths and evaluates the field at 16 sampled times on each, for 256 evaluations per row. Section IV-E examines how the sampling budgets affect detection and the stability of review lists. Two rows with close expected scores can still switch places when rescored. The argument following (20) gives a sufficient gap to preserve their order with high probability.

1) Proof of Proposition 1: Step 1: Population orthogonality. Let $\mathcal { G } = \sigma (  { \boldsymbol { X } } _ { T } , T )$ at the fixed context. Conditional expectation gives E $\left[ { \pmb { V } } - { \pmb { f } } ^ { \star } ( { \pmb { X } } _ { T } , { \pmb { c } } , T ) \ | \ { \pmb { \mathcal { G } } } \right] = \mathbf { 0 }$ . Writing $U =$ $f _ { \theta } ( X _ { T } , c , T ) - f ^ { \star } ( X _ { T } , c , T )$ , which is G-measurable, yields

$$
\mathbb { E } \langle U , f ^ { \star } ( X _ { T } , c , T ) - V \rangle = 0 .
$$

All terms are integrable by the second-moment assumptions and conditional Jensen’s inequality. Expanding the squared residual

proves the first identity in (8). The remaining conditional squared norm has expectation

$$
\operatorname { \mathbb { E } } \| V - \operatorname { \mathbb { E } } [ V \mid { \mathcal { G } } ] \| _ { 2 } ^ { 2 } = \operatorname { \mathbb { E } } { \mathrm { ~ t r } } \operatorname { C o v } ( V \mid { \mathcal { G } } ) ,
$$

which proves the second identity and the optimality of $f ^ { \star } ~ [ 1 4 , 3 1 ]$ . This orthogonality averages over the random endpoint $B ;$ conditioning additionally on a fixed test endpoint would change the regression problem and does not justify the same decomposition.

Step 2: Posterior denoising identity. For $t ~ < ~ 1$ , the interpolation identity implies $V = ( B - X _ { t } ) / ( 1 - t )$ . Applying this identity at $T$ and conditioning on $( X _ { T } , T )$ gives

$$
f ^ { \star } ( x , c , t ) = \frac { m ^ { \star } ( x , c , t ) - x } { 1 - t } .
$$

For a fixed $( b , z )$ , one also has $b - z = ( b - x _ { t } ) / ( 1 - t )$ Subtracting proves (9). The Gaussian corruption has a strictly positive density for $t < 1$ , so posterior versions can be used along these fixed-endpoint paths almost surely. The singular endpoint $t = 1$ has zero probability under the time-sampling law and is not needed for this identity.

Step 3: Stability of the robust energy. The absolute value and the clipping map are nonexpansive. Hence, for any ${ \mathbf { } } u , v \in { \mathbf { \Gamma } }$ R<sup>Q</sup>,

$$
\begin{array} { l } { \displaystyle | \rho ( \pmb { u } ) - \rho ( \pmb { v } ) | \leq \sum _ { q = 1 } ^ { Q } \frac { | { \pmb u } _ { q } - { \pmb v } _ { q } | } { d _ { q } } } \\ { \leq \left( \sum _ { q = 1 } ^ { Q } d _ { q } ^ { - 2 } \right) ^ { 1 / 2 } \| \pmb { u } - \pmb { v } \| _ { 2 } = \kappa \| \pmb { u } - \pmb { v } \| _ { 2 } . } \end{array}\tag{19}
$$

At a fixed endpoint b, apply this inequality to the two fields residual vectors using the same $( Z , T )$ . The target $_ { b - Z }$ cancels. Taking expectations and applying Cauchy–Schwarz gives

$$
\begin{array} { r l } & { | \mu _ { f _ { \theta } } ( \pmb { b } \mid c ) - \mu _ { f ^ { \star } } ( \pmb { b } \mid c ) | } \\ & { \quad \leq { \kappa \mathbb { E } _ { Z , T } } \| f _ { \theta } ( \pmb { x } _ { T } , \pmb { c } , T ) - f ^ { \star } ( \pmb { x } _ { T } , \pmb { c } , T ) \| _ { 2 } } \\ & { \quad \leq { \kappa \sqrt { D _ { \theta } ( \pmb { b } \mid c ) } } . } \end{array}
$$

Finally, average over B and use Jensen’s inequality and Step 1:

$$
\begin{array} { r } { \mathbb { E } _ { B } | \mu _ { f _ { \theta } } - \mu _ { f ^ { \star } } | \le \kappa \sqrt { \mathbb { E } _ { B } { D } _ { \theta } ( B \mid c ) } \qquad } \\ { = \kappa \sqrt { \mathscr { L } _ { c } ( f _ { \theta } ) - \mathscr { L } _ { c } ( f ^ { \star } ) } . } \end{array}
$$

This proves (12). The argument requires no differentiability or ODE assumptions, and applies to the ReLU field used in our implementation. □

A nonzero residual at the optimal field. For a onedimensional reference distribution $B \ \sim \ { \mathcal { N } } ( 0 , 1 )$ and independent $Z \sim { \mathcal { N } } ( 0 , 1 )$ , let $a ( t ) = t ^ { 2 } + ( 1 - t ) ^ { 2 }$ . Suppressing the fixed context, $f ^ { \star } ( x , t ) = ( 2 t - 1 ) x / a ( t )$ and

$$
f ^ { \star } ( ( 1 - t ) z + t b , t ) - ( b - z ) = \frac { t z - ( 1 - t ) b } { a ( t ) } .
$$

Consequently, the unnormalized quadratic residual energy at a fixed b is

$$
\mathcal { R } ^ { \star } ( b ) = \int _ { 0 } ^ { 1 } \frac { t ^ { 2 } + ( 1 - t ) ^ { 2 } b ^ { 2 } } { a ( t ) ^ { 2 } } d t = \frac { \pi } { 4 } ( 1 + b ^ { 2 } ) .
$$

For the implemented score, fix $0 < t < 1$ and write $e _ { t }$ for the displayed residual evaluated at $z = Z$ . The distribution of $\left| e _ { t } \right|$ is that of $| G _ { t } |$ for $G _ { t } \sim \mathcal N ( m _ { t } , s _ { t } ^ { 2 } )$ , where $m _ { t } = ( 1 - t ) | b | / a ( t )$ and $s _ { t } = t / a ( t )$ . For $r > 0$ its CDF is

$$
\Phi ( ( r - m _ { t } ) / s _ { t } ) + \Phi ( ( r + m _ { t } ) / s _ { t } ) - 1 .
$$

Here $\Phi$ is the standard normal CDF. The displayed CDF decreases strictly with $m _ { t } > 0 ,$ since the standard normal density at $( r + m _ { t } ) / s _ { t }$ is smaller than at $( r - m _ { t } ) / s _ { t }$ . Using E min $\{ | e _ { t } | / d _ { 1 } , \tau \} ~ = ~ \int _ { 0 } ^ { \tau } \mathrm { P r } ( | e _ { t } | ~ > ~ d _ { 1 } u )$ du and averaging over t shows that the expected implemented score strictly increases with |b|. It therefore orders Gaussian tail observations by increasing negative log-likelihood in this one-dimensional setting even at $f ^ { \star }$

2) Proof of Proposition 2: Step 1: Unbiasedness under nested sampling. Each pair $( Z _ { k } , T _ { k \ell } )$ has the same marginal distribution as an independent $( Z , T )$ . By linearity,

$$
\mathbb { E } \widehat { \mu } = \frac { 1 } { K _ { z } K _ { t } } \sum _ { k , \ell } \mathbb { E } s ( Z _ { k } , T _ { k \ell } ) = \mu .
$$

This step does not require independence of all $K _ { z } K _ { t }$ residual terms.

Step 2: Exact variance. Conditional on $Z _ { k } .$ , the $K _ { t }$ terms within block k are independent. Therefore,

$$
\begin{array} { r l } & { \mathbb { E } [ A _ { k } \mid Z _ { k } ] = \hat { s } ( Z _ { k } ) , } \\ & { \mathrm { V a r } ( A _ { k } \mid Z _ { k } ) = \frac { \mathrm { V a r } \left( s ( Z _ { k } , T ) \mid Z _ { k } \right) } { K _ { t } } . } \end{array}
$$

The law of total variance gives $\mathrm { V a r } ( A _ { k } ) = \sigma _ { z } ^ { 2 } + \sigma _ { t } ^ { 2 } / K _ { t } =$ $\nu _ { K _ { t } }$ . The complete random blocks $( Z _ { k } , T _ { k 1 } , \ldots , T _ { k K _ { t } } )$ are independent and identically distributed, so the $A _ { k }$ are also independent and identically distributed. Dividing their variance by $K _ { z }$ proves (15). In contrast, for $\ell \neq m$ within one block,

$$
\mathrm { C o v } \left( s ( { \cal Z } _ { k } , { \cal T } _ { k \ell } ) , s ( { \cal Z } _ { k } , { \cal T } _ { k m } ) \right) = \sigma _ { z } ^ { 2 } .
$$

This covariance is generally nonzero.

Step 3: Finite-sample concentration. Since $0 \leq s ( z , t ) \leq$ H, every block average lies in [0, H]. Applying Hoeffding’s inequality [32] to the $K _ { z }$ independent block averages gives (16); solving its right-hand side for δ gives (17). For the variancesensitive form, let $W _ { k } = A _ { k } - \mu$ . Then $\mathbb { E } W _ { k } = 0 , | W _ { k } | \le H$ and $\mathbb { E } W _ { k } ^ { 2 } = \nu _ { K _ { t } }$ . For $0 < \lambda < 3 / H$ , the exponential series and $\mathbb { E } | W _ { k } | ^ { m } \le \nu _ { K _ { t } } H ^ { m - 2 }$ for m $\geq 2$ yield

$$
\log \mathbb { E } e ^ { \lambda W _ { k } } \leq \frac { \lambda ^ { 2 } \nu _ { K _ { t } } } { 2 ( 1 - \lambda H / 3 ) } .
$$

Here we used m! $\geq 2 3 ^ { m - 2 }$ and $\log ( 1 + x ) \leq x$ . Independence and the exponential Markov inequality, with $\lambda = \epsilon / ( \nu _ { K _ { t } } +$ $H \epsilon / 3 )$ when $\nu _ { K _ { t } } > 0$ , give

$$
\operatorname* { P r } ( \widehat { \mu } - \mu \geq \epsilon ) \leq \exp \left( - \frac { K _ { z } \epsilon ^ { 2 } } { 2 \nu _ { K _ { t } } + 2 H \epsilon / 3 } \right) .
$$

Apply the same argument $\mathrm { t o \ - } W _ { k }$ and add the two tails to obtain (18) [33]. If $\nu _ { K _ { t } } = 0$ , each block average is constant almost surely, which gives the stated degenerate case. □

Consequence for score ordering. Let $\mu _ { i }$ and $\widehat { \mu } _ { i }$ be the expected and estimated raw scores for row i. For N fixed rows with common $Q , \tau ,$ and $K _ { z }$ , the union bound gives, with probability at least 1 − δ for $0 < \delta < 1$

$$
\operatorname* { m a x } _ { 1 \le i \le N } | \widehat \mu _ { i } - \mu _ { i } | \le r _ { N } : = H \sqrt { \frac { \log ( 2 N / \delta ) } { 2 K _ { z } } } .\tag{20}
$$

Thus a pair with $\mu _ { i } - \mu _ { j } > 2 r _ { N }$ retains its strict order on this event. $\mathrm { A t } \ N = 2 0 , 0 0 0 , K _ { z } = 1 6$ , and $\delta = 0 . 0 5$ , this bound’s sufficient gap is 1.30H, exceeding the score range. We measure review-list stability under repeated scoring in Section IV-E5.

## IV. EXPERIMENTS

Our experiments are designed to answer the following research questions: RQ1. How well does Flow-CAD rank anomalous rows? RQ2. How does it compare with existing contextual and point detectors? RQ3. How do behavioral features contribute to individual anomaly scores? RQ4. How sensitive is it to scoring settings and context encoding?

## A. Experimental Setup

This subsection describes the common experimental protocol. We introduce the datasets and their context–behavior partitions, describe how the benchmark tables and anomaly labels are constructed, explain how the baseline methods are configured, and then give the fixed Flow-CAD settings and evaluation metrics.

Datasets and feature partitions. We evaluate all methods on 20 real-world tabular datasets summarized in Table I. We use the source tables released with QCAD [2]. We assume the partitions of features are given beforehand. Behavior is numerical. Context may include categorical columns.

For example, in the Abalone dataset, Sex, Length, Diameter, Height are treated as contextual features and the remaining attributes as behavioral features. In ForestFires, spatial and temporal variables (X, Y, month, day) define the context, and weather, fire-weather indices, and burned area form the behavior.

Benchmark construction. Following the QCAD benchmark [2], we remove incomplete rows, cap each input at 20,000 rows by seeded subsampling, and min–max scale the numerical columns designated by the benchmark. We sample the number of rows listed in Table I uniformly without replacement, excluding the first row as in the released generator. Each selected row receives an independent signed shift in every behavioral column, with magnitude drawn uniformly from [0.1, 0.5] and the sign chosen uniformly. The selected rows define the benchmark anomaly labels.

We use seven data seeds (43–49) for each dataset. All methods are evaluated on the same 140 benchmark tables, with runs paired by dataset and data seed.

Baselines. We compare Flow-CAD against 13 representative baselines, including four contextual anomaly detectors (QCAD [2], LoPAD [18], ROCOD [9], CAD [1]), and nine point anomaly detectors (Isolation Forest [34], LOF [35], k-NN [36], SOD [37], HBOS [38], LUNAR [39], DTE-Categorical, DTE-NonParametric [40], and TCCM [25]).

TABLE I  
RECONSTRUCTED BENCHMARK TABLES USED FOR FULL-TABLE EVALUATION. COLUMN COUNTS ARE REPORTED BEFORE ENCODING.
<table><tr><td colspan="3"></td><td>#Contextual columns</td><td>#Behavioral columns</td></tr><tr><td>Dataset</td><td>#Samples</td><td>#Anomalies (Rate)</td><td></td><td></td></tr><tr><td>Abalone</td><td>4177</td><td>100 (2.4%)</td><td>4 5</td><td>5 1</td></tr><tr><td>Airfoil</td><td>1503</td><td>70 (4.7%)</td><td>12</td><td>2</td></tr><tr><td>BodyFat</td><td>252 506</td><td>20 (7.9%) 40 (7.9%)</td><td>13</td><td>1</td></tr><tr><td>Boston</td><td>1030</td><td>50 (4.9%)</td><td>8</td><td>1</td></tr><tr><td>Concrete ElNino</td><td>20000</td><td>200 (1.0%)</td><td>6</td><td>5</td></tr><tr><td></td><td>768</td><td>50 (6.5%)</td><td>8</td><td>2</td></tr><tr><td>Energy FishWeight</td><td>159</td><td>15 (9.4%)</td><td>6</td><td>1</td></tr><tr><td>ForestFires</td><td>517</td><td>50 (9.7%)</td><td>4</td><td>9</td></tr><tr><td>GasEmission</td><td>7384</td><td>100 (1.4%)</td><td>8</td><td>3</td></tr><tr><td>HeartFailure</td><td>299</td><td>30 (10.0%)</td><td>6</td><td>6</td></tr><tr><td>Hepatitis</td><td>589</td><td>30 (5.1%)</td><td>3</td><td>10</td></tr><tr><td>IndianLiver</td><td>579</td><td>30 (5.2%)</td><td>3</td><td>8</td></tr><tr><td>Maintenance</td><td>11934</td><td>100 (0.8%)</td><td>15</td><td>3</td></tr><tr><td>Parkinson</td><td>5875</td><td>100 (1.7%)</td><td>20</td><td>2</td></tr><tr><td>PowerPlant</td><td>9568</td><td>100 (1.0%)</td><td>4</td><td>1</td></tr><tr><td></td><td>475</td><td>40 (8.4%)</td><td>7</td><td>6</td></tr><tr><td>QSRanking</td><td>557</td><td>50 (9.0%)</td><td>4</td><td>1</td></tr><tr><td>SynMachine</td><td>908</td><td>50 (5.5%)</td><td>6</td><td>1</td></tr><tr><td>Toxicity Yacht</td><td>308</td><td>30 (9.7%)</td><td>6</td><td>1</td></tr></table>

We keep each baseline’s settings fixed across runs on a given dataset. We reimplemented QCAD using the neighborhood sizes specified in its benchmark. LoPAD uses the Markov blankets from the QCAD release and fits CART models on 20 subsamples. ROCOD uses the benchmark distance cutoff and relies on its global predictor for rows with no neighbors. CAD fits a joint Gaussian mixture with five components and scores rows by their conditional density. We use the authors’ released code for DTE and TCCM. For the remaining methods, we use the settings from PyOD 2.0.3 [41], including LUNAR’s built-in validation with synthetic negatives.

Flow-CAD configuration. Flow-CAD one-hot encodes the designated categorical context columns. The resulting context vector is held fixed along the interpolation path, while interpolation and feature attribution are defined only for the numerical behavioral features. We use the fixed settings in Table II for every run. Each model is trained on a complete benchmark table for 1,000 epochs, and its final checkpoint is used to score all rows in that table. The seven runs pair data seeds 43–49 with training seeds 202–208 in ascending order. Each run uses a single CPU thread.

TABLE II  
FLOW-CAD CONFIGURATION FIXED ACROSS BENCHMARK DATASETS AND RUNS.
<table><tr><td>Component</td><td>Fixed value</td></tr><tr><td>Context encoder</td><td>2 layers of width 32</td></tr><tr><td>Velocity network</td><td>3 hidden layers of width 64, ReLU</td></tr><tr><td>Optimizer</td><td>Adam, learning rate 0.001</td></tr><tr><td>Batch size</td><td>128</td></tr><tr><td>Training epochs</td><td>1,000</td></tr><tr><td>Score aggregation</td><td> $L _ { 1 }$ </td></tr><tr><td>Residual truncation</td><td> $\tau = 1 0$ </td></tr><tr><td>Monte Carlo budget</td><td> $K _ { z } = K _ { t } = 1 6$ </td></tr></table>

Evaluation metrics. Anomaly labels are reserved for evaluation. For the main same-table benchmark, we report the area under the receiver operating characteristic curve (ROC AUC), the trapezoidal area under the precision–recall curve (PR AUC), and precision at n (P@n), with n equal to the number of labeled anomalies. We report mean and sample standard deviation across runs and average the per-dataset ranks over the 20 datasets.

## B. Detection Quality Across Benchmark Tables (RQ1)

Tables III–V report results on 20 datasets. We analyze the overall ranks and detection when anomalies are rare, then leave out one dataset at a time to test rank stability.

1) Ranks across datasets. Among the 14 methods, Flow-CAD has the best average rank for PR AUC (1.900), ROC AUC (2.050), and P@n (1.825). These metrics examine different parts of detection: ROC AUC summarizes the full score ordering, PR AUC focuses on correctly retrieving anomalies, and P@n measures precision among the top n rows. Thus, Flow-CAD performs well both across the full ranking and among the rows inspected first.

2) Detection with rare anomalies. Five datasets in Table I have anomaly rates below 2%. Across their 15 dataset–metric comparisons, Flow-CAD ranks among the top three methods in 13. It ranks first on all three metrics for Parkinson, and first for PR AUC and P@n on PowerPlant. These results show that its strong overall ranking is retained when only a small fraction of rows is anomalous.

3) Influence of individual datasets. Because an average rank can be driven by one especially favorable dataset, we recalculate it with one dataset left out at a time. Each calculation therefore averages the ranks over the other 19 datasets. Flow-CAD remains first for PR AUC and P@n in all 20 cases. For ROC AUC, it ranks first or ties for first in 16 cases and ranks second in the other four. Thus, the conclusions for PR AUC and P@n do not depend on any single table; the ROC AUC comparison is closer, but Flow-CAD never falls below second.

## C. Comparison with Existing Detectors (RQ2)

We compare Flow-CAD with contextual and point anomaly detectors, first within each group and then against the strongest competitor for each dataset and metric.

1) Contextual anomaly detectors. Among QCAD, CAD, ROCOD, LoPAD, and Flow-CAD, Flow-CAD has the best average rank for all three metrics. Its average ranks are 1.900, 2.050, and 1.825, compared with 2.300, 2.100, and 2.350 for QCAD. Flow-CAD has higher PR AUC and P@n than QCAD on 13 of the 20 datasets, while the two methods each lead on ten datasets for ROC AUC. Flow-CAD also leads CAD on at least 15 datasets for every metric, and leads ROCOD and LoPAD on all 20.

2) Point anomaly detectors. The best average ranks among the nine point detectors are 6.350 for PR AUC (HBOS), 5.800 for ROC AUC (LUNAR), and 6.025 for P@n (HBOS). Flow-CAD’s corresponding ranks are 1.900, 2.050, and 1.825. Across datasets, it leads every point detector in 14 PR AUC cases, 14 ROC AUC cases, and 16 P@n cases.

PR AUC ON 20 DATASETS. ENTRIES GIVE MEAN ± STANDARD DEVIATION (RANK). RANKS USE UNROUNDED MEANS AND AVERAGE NUMERICAL TIES (10<sup>−12</sup> TOLERANCE). EQUAL DISPLAYED MEANS CAN HAVE DIFFERENT RANKS. BOLD AND UNDERLINING MARK THE TOP TWO DISTINCT RANKS. EACH METHOD HAS 140 PAIRED CASES, WITH SEVEN BENCHMARK RUNS PER DATASET.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Flow-CAD (ours)</td><td rowspan="2">QCAD</td><td rowspan="2">LoPAD</td><td rowspan="2">ROCOD</td><td rowspan="2">CAD</td></tr><tr><td></td></tr><tr><td>Abalone</td><td>1.00±0.00 (1)</td><td>0.96±0.01 (3)</td><td>0.01±0.00 (14)</td><td>0.55±0.03 (10)</td><td>0.98±0.01 (2)</td></tr><tr><td>Airfoil</td><td>0.44±0.10 (4)</td><td>0.68±0.05 (1)</td><td>0.05±0.00 (14)</td><td>0.42±0.04 (5)</td><td>0.49±0.04 (2)</td></tr><tr><td>BodyFat</td><td>0.84±0.08 (1)</td><td>0.63±0.08 (4)</td><td>0.07±0.02 (14)</td><td>0.79±0.05 (3)</td><td>0.81±0.12 (2)</td></tr><tr><td>Boston</td><td>0.60±0.10 (2)</td><td>0.63±0.10 (1)</td><td>0.08±0.00 (13)</td><td>0.30±0.06 (4)</td><td>0.32±0.09 (3)</td></tr><tr><td>Concrete</td><td>0.67±0.14 (1)</td><td>0.60±0.09 (2)</td><td>0.09±0.02 (9)</td><td>0.31±0.06 (5)</td><td>0.30±0.06 (6)</td></tr><tr><td>ElNino</td><td>0.81±0.04 (3)</td><td>0.99±0.00 (1)</td><td>0.11±0.01 (8)</td><td>0.39±0.05 (7)</td><td>0.96±0.01 (2)</td></tr><tr><td>Energy</td><td>0.94±0.03 (1)</td><td>0.93±0.02 (2)</td><td>0.05±0.00 (14)</td><td>0.43±0.07 (7)</td><td>0.91±0.03 (3)</td></tr><tr><td>FishWeight</td><td>0.90±0.05 (1)</td><td>0.85±0.10 (2)</td><td>0.13±0.06 (8)</td><td>0.44±0.10 (4)</td><td>0.72±0.09 (3)</td></tr><tr><td>ForestFires</td><td>0.97±0.02 (4)</td><td>0.99±0.01 (2)</td><td>0.10±0.01 (14)</td><td>0.49±0.03 (10)</td><td>0.88±0.03 (6)</td></tr><tr><td>GasEmission</td><td>0.98±0.01 (2)</td><td>0.98±0.00 (1)</td><td>0.01±0.00 (14)</td><td>0.87±0.02 (3)</td><td>0.85±0.02 (4)</td></tr><tr><td>HeartFailure</td><td>0.97±0.01 (2)</td><td>0.91±0.03 (3)</td><td>0.05±0.00 (14)</td><td>0.54±0.03 (6)</td><td>0.77±0.09 (5)</td></tr><tr><td>Hepatitis</td><td>0.99±0.01 (2)</td><td>0.98±0.01 (3)</td><td>0.05±0.01 (14)</td><td>0.55±0.03 (8)</td><td>0.84±0.08 (5)</td></tr><tr><td>IndianLiver</td><td>0.97±0.01 (2)</td><td>0.90±0.04 (4)</td><td>0.05±0.00 (14)</td><td>0.50±0.06 (8)</td><td>0.79±0.12 (6)</td></tr><tr><td>Maintenance</td><td>0.76±0.08 (4)</td><td>0.95±0.02 (1)</td><td>0.01±0.00 (12)</td><td>0.50±0.00 (5)</td><td>0.87±0.04 (3)</td></tr><tr><td>Parkinson</td><td>0.98±0.01 (1)</td><td>0.92±0.02 (2)</td><td>0.02±0.00 (14)</td><td>0.60±0.06 (4)</td><td>0.75±0.04 (3)</td></tr><tr><td>PowerPlant</td><td>0.66±0.04 (1)</td><td>0.46±0.04 (4)</td><td>0.02±0.00 (10)</td><td>0.66±0.05 (2)</td><td>0.56±0.07 (3)</td></tr><tr><td>QSRanking</td><td>0.74±0.08 (3)</td><td>0.78±0.05 (1)</td><td>0.06±0.00 (14)</td><td>0.56±0.06 (4)</td><td>0.77±0.07 (2)</td></tr><tr><td>SynMachine</td><td>1.00±0.00 (1)</td><td>0.97±0.02 (3)</td><td>0.35±0.03 (13)</td><td>0.80±0.07 (7)</td><td>0.54±0.12 (11)</td></tr><tr><td>Toxicity</td><td>0.61±0.06 (1)</td><td>0.44±0.09 (4)</td><td>0.11±0.01 (8)</td><td>0.46±0.06 (3)</td><td>0.46±0.06 (2)</td></tr><tr><td>Yacht</td><td>0.92±0.06 (1)</td><td>0.89±0.04 (2)</td><td>0.53±0.01 (3)</td><td>0.29±0.06 (6)</td><td>0.25±0.07 (9)</td></tr><tr><td>Avg Rank</td><td>1.900</td><td>2.300</td><td>11.900</td><td>5.550</td><td>4.100</td></tr><tr><td>Dataset</td><td>IForest</td><td>LOF</td><td>k-NN</td><td>SOD</td><td>HBOS</td></tr><tr><td>Abalone</td><td>0.41±0.07 (11)</td><td>0.92±0.01 (6)</td><td>0.94±0.00 (5)</td><td>0.75±0.01 (9)</td><td>0.20±0.04 (13)</td></tr><tr><td>Airfoil</td><td>0.10±0.02 (8)</td><td>0.05±0.01 (11)</td><td>0.05±0.01 (12)</td><td>0.14±0.02 (7)</td><td>0.09±0.02 (9)</td></tr><tr><td>BodyFat</td><td>0.17±0.04 (7)</td><td>0.09±0.02 (12)</td><td>0.09±0.02 (13)</td><td>0.10±0.04 (8)</td><td>0.18±0.03 (6)</td></tr><tr><td>Boston</td><td>0.11±0.02 (8)</td><td>0.08±0.01 (11)</td><td>0.08±0.01 (12)</td><td>0.08±0.01 (9)</td><td>0.13±0.02 (6)</td></tr><tr><td>Concrete</td><td>0.08±0.02 (10)</td><td>0.06±0.01 (12)</td><td>0.06±0.02 (11)</td><td>0.06±0.01 (13)</td><td>0.26±0.06 (7)</td></tr><tr><td>ElNino</td><td>0.55±0.10 (5)</td><td>0.01±0.00 (14)</td><td>0.01±0.00 (13)</td><td>0.04±0.00 (9)</td><td>0.77±0.03 (4)</td></tr><tr><td>Energy</td><td>0.39±0.03 (8)</td><td>0.11±0.02 (12)</td><td>0.37±0.02 (9)</td><td>0.89±0.05 (4)</td><td>0.32±0.05 (11)</td></tr><tr><td>FishWeight</td><td>0.22±0.05 (6)</td><td>0.11±0.04 (14)</td><td>0.11±0.04 (12)</td><td>0.12±0.03 (11)</td><td>0.20±0.07 (7)</td></tr><tr><td>ForestFires</td><td>0.97±0.02 (3)</td><td>0.17±0.02 (13)</td><td>0.18±0.02 (12)</td><td>0.57±0.04 (9)</td><td>1.00±0.00 (1)</td></tr><tr><td>GasEmission</td><td>0.10±0.02 (7)</td><td>0.01±0.00 (13)</td><td>0.01±0.00 (12)</td><td>0.03±0.00 (8)</td><td>0.25±0.05 (6)</td></tr><tr><td>HeartFailure</td><td>0.86±0.02 (4)</td><td>0.13±0.03 (12)</td><td>0.17±0.05 (11)</td><td>0.37±0.07 (9)</td><td>0.98±0.01 (1)</td></tr><tr><td>Hepatitis</td><td>0.93±0.04 (4)</td><td>0.17±0.02 (12)</td><td>0.15±0.02 (13)</td><td>0.35±0.01 (10)</td><td>0.99±0.01 (1)</td></tr><tr><td>IndianLiver</td><td>0.94±0.02 (3)</td><td>0.11±0.02 (12)</td><td>0.17±0.04 (11)</td><td>0.45±0.05 (9)</td><td>0.99±0.01 (1)</td></tr><tr><td>Maintenance</td><td>0.02±0.00 (10)</td><td>0.01±0.00 (14)</td><td>0.01±0.00 (13)</td><td>0.02±0.00 (9)</td><td>0.23±0.08 (6)</td></tr><tr><td>Parkinson</td><td>0.02±0.00 (10)</td><td>0.02±0.00 (13)</td><td>0.02±0.00 (12)</td><td>0.04±0.00 (8)</td><td>0.02±0.00 (9)</td></tr><tr><td>PowerPlant</td><td>0.05±0.01 (9)</td><td>0.01±0.00 (14)</td><td>0.01±0.00 (12)</td><td>0.01±0.00 (11)</td><td>0.23±0.03 (7)</td></tr><tr><td>QSRanking</td><td>0.22±0.07 (7)</td><td>0.10±0.03 (10)</td><td>0.09±0.02 (12)</td><td>0.09±0.01 (13)</td><td>0.44±0.07 (5)</td></tr><tr><td>SynMachine</td><td>0.36±0.06 (12)</td><td>0.80±0.04 (8)</td><td>0.91±0.02 (4)</td><td>0.72±0.06 (9)</td><td>0.26±0.04 (14)</td></tr><tr><td>Toxicity Yacht</td><td>0.09±0.01 (10)</td><td>0.08±0.02 (11)</td><td>0.07±0.01 (14)</td><td>0.12±0.02 (7)</td><td>0.10±0.01 (9)</td></tr><tr><td></td><td>0.28±0.07 (7)</td><td>0.19±0.04 (13)</td><td>0.25±0.03 (10)</td><td>0.25±0.05 (8)</td><td>0.31±0.09 (4)</td></tr><tr><td>Avg Rank</td><td>7.450</td><td>11.850</td><td>11.150</td><td>9.000</td><td>6.350</td></tr><tr><td>Dataset</td><td>LUNAR</td><td>DTE-C</td><td></td><td>DTE-NP</td><td>TCCM</td></tr><tr><td>Abalone</td><td>0.92±0.03 (7)</td><td>0.92±0.01 (8)</td><td>0.94±0.00 (4)</td><td></td><td>0.24±0.11 (12)</td></tr><tr><td>Airfoil BodyFat</td><td>0.29±0.06 (6) 0.31±0.08 (5)</td><td>0.46±0.18 (3) 0.09±0.05 (11)</td><td>0.06±0.01 (10)</td><td></td><td>0.05±0.01 (13)</td></tr><tr><td>Boston</td><td>0.13±0.02 (7)</td><td>0.07±0.01 (14)</td><td>0.10±0.03 (9)</td><td></td><td>0.10±0.02 (10) 0.08±0.02 (10)</td></tr><tr><td>Concrete</td><td>0.12±0.06 (8)</td><td>0.52±0.00 (3)</td><td>0.16±0.03 (5) 0.48±0.03 (4)</td><td></td><td>0.05±0.01 (14)</td></tr><tr><td>ElNino</td><td>0.02±0.01 (10)</td><td>0.51±0.00 (6)</td><td>0.01±0.00 (11)</td><td></td><td>0.01±0.00 (12)</td></tr><tr><td>Energy</td><td>0.44±0.08 (6)</td><td>0.53±0.00 (5)</td><td>0.34±0.02 (10)</td><td></td><td>0.09±0.03 (13)</td></tr><tr><td>FishWeight</td><td>0.31±0.07 (5)</td><td>0.11±0.04 (13)</td><td>0.12±0.05 (10)</td><td></td><td>0.13±0.05 (9)</td></tr><tr><td>ForestFires</td><td>0.78±0.16 (8)</td><td>0.78±0.05 (7)</td><td>0.21±0.02 (11)</td><td></td><td>0.90±0.02 (5)</td></tr><tr><td>GasEmission</td><td>0.41±0.15 (5)</td><td>0.01±0.00 (11)</td><td>0.01±0.00 (10)</td><td></td><td>0.03±0.00 (9)</td></tr><tr><td>HeartFailure</td><td>0.41±0.04 (7)</td><td>0.12±0.02 (13)</td><td>0.20±0.06 (10)</td><td></td><td></td></tr><tr><td></td><td>0.53±0.09 (9)</td><td>0.84±0.04 (6)</td><td></td><td></td><td>0.40±0.11 (8)</td></tr><tr><td>Hepatitis</td><td>0.85±0.02 (5)</td><td>0.08±0.03 (13)</td><td>0.19±0.02 (11)</td><td></td><td>0.73±0.20 (7)</td></tr><tr><td>IndianLiver Maintenance</td><td>0.94±0.08 (2)</td><td>0.15±0.18 (7)</td><td>0.21±0.04 (10) 0.07±0.01 (8)</td><td></td><td>0.59±0.13 (7)</td></tr><tr><td>Parkinson</td><td>0.14±0.02 (6)</td><td>0.51±0.00 (5)</td><td>0.02±0.00 (11)</td><td></td><td>0.01±0.00 (11) 0.12±0.07 (7)</td></tr><tr><td>PowerPlant</td><td>0.43±0.16 (6) 0.17±0.03 (8)</td><td>0.43±0.19 (5)</td><td>0.01±0.00 (13)</td><td></td><td>0.15±0.11 (8)</td></tr></table>

3) Comparison with the strongest baseline on each table. The leading baseline changes with the dataset and metric. We therefore compare Flow-CAD with the best member of each group in every case. Across the 60 dataset–metric combinations, Flow-CAD exceeds the strongest contextual baseline in 34 cases and the strongest point detector in 44. Its average-rank advantage is thus supported by results across different datasets and metrics rather than by one favorable subset.

TABLE IV  
ROC AUC ON 20 DATASETS. NOTATION FOLLOWS TABLE III.
<table><tr><td>Dataset</td><td>Flow-CAD (ours)</td><td>QCAD</td><td>LoPAD</td><td>ROCOD</td><td>CAD</td></tr><tr><td>Abalone</td><td>1.00±0.00 (1)</td><td>1.00±0.00 (4)</td><td>0.29±0.03 (14)</td><td>0.98±0.00 (9)</td><td>0.99±0.01 (8)</td></tr><tr><td>Airfoil</td><td>0.81±0.05 (4)</td><td>0.92±0.02 (1)</td><td>0.50±0.03 (13)</td><td>0.79±0.02 (5)</td><td>0.82±0.02 (3)</td></tr><tr><td>BodyFat</td><td>0.96±0.03 (2)</td><td>0.93±0.04 (4)</td><td>0.49±0.03 (14)</td><td>0.94±0.03 (3)</td><td>0.98±0.01 (1)</td></tr><tr><td>Boston</td><td>0.89±0.04 (2)</td><td>0.93±0.02 (1)</td><td>0.47±0.02 (13)</td><td>0.82±0.04 (3)</td><td>0.78±0.05 (4)</td></tr><tr><td>Concrete</td><td>0.91±0.06 (2)</td><td>0.93±0.03 (1)</td><td>0.63±0.03 (7)</td><td>0.77±0.03 (3)</td><td>0.74±0.04 (4)</td></tr><tr><td>ElNino</td><td>0.99±0.00 (3)</td><td>1.00±0.00 (1)</td><td>0.91±0.01 (7)</td><td>0.98±0.00 (5)</td><td>1.00±0.00 (2)</td></tr><tr><td>Energy</td><td>0.99±0.01 (2)</td><td>0.99±0.00 (1)</td><td>0.38±0.03 (14)</td><td>0.82±0.04 (9)</td><td>0.98±0.01 (3)</td></tr><tr><td>FishWeight</td><td>0.97±0.03 (1)</td><td>0.96±0.03 (2)</td><td>0.52±0.05 (12)</td><td>0.80±0.07 (5)</td><td>0.91±0.05 (3)</td></tr><tr><td>ForestFires</td><td>1.00±0.00 (4)</td><td>1.00±0.00 (2)</td><td>0.51±0.04 (14)</td><td>0.87±0.02 (10)</td><td>0.99±0.00 (6)</td></tr><tr><td>GasEmission</td><td>1.00±0.00 (2)</td><td>1.00±0.00 (1)</td><td>0.46±0.01 (14)</td><td>1.00±0.00 (4)</td><td>1.00±0.00 (3)</td></tr><tr><td>HeartFailure</td><td>1.00±0.00 (2)</td><td>0.99±0.01 (3)</td><td>0.11±0.03 (14)</td><td>0.52±0.05 (13)</td><td>0.98±0.01 (5)</td></tr><tr><td>Hepatitis</td><td>1.00±0.00 (2)</td><td>1.00±0.00 (3)</td><td>0.50±0.07 (14)</td><td>0.98±0.00 (8)</td><td>0.99±0.01 (6)</td></tr><tr><td>IndianLiver</td><td>1.00±0.00 (2)</td><td>0.99±0.00 (4)</td><td>0.48±0.06 (14)</td><td>0.97±0.01 (7)</td><td>0.98±0.03 (6)</td></tr><tr><td>Maintenance</td><td>0.97±0.02 (3)</td><td>1.00±0.00 (2)</td><td>0.51±0.05 (11)</td><td>0.50±0.00 (13)</td><td>0.97±0.01 (4)</td></tr><tr><td>Parkinson</td><td>1.00±0.00 (1)</td><td>0.99±0.01 (2)</td><td>0.42±0.02 (14)</td><td>0.95±0.02 (3)</td><td>0.92±0.03 (5)</td></tr><tr><td>PowerPlant</td><td>0.98±0.00 (2)</td><td>0.98±0.01 (1)</td><td>0.46±0.01 (14)</td><td>0.97±0.01 (3)</td><td>0.91±0.02 (4)</td></tr><tr><td>QSRanking</td><td>0.94±0.03 (3)</td><td>0.96±0.01 (1)</td><td>0.37±0.03 (14)</td><td>0.88±0.03 (4)</td><td>0.95±0.02 (2)</td></tr><tr><td>SynMachine</td><td>1.00±0.00 (1)</td><td>0.98±0.01 (4)</td><td>0.79±0.04 (12)</td><td>0.90±0.03 (8)</td><td>0.75±0.09 (13)</td></tr><tr><td>Toxicity</td><td>0.90±0.02 (1)</td><td>0.88±0.03 (2)</td><td>0.58±0.04 (13)</td><td>0.85±0.04 (3)</td><td>0.84±0.04 (4)</td></tr><tr><td>Yacht</td><td>0.99±0.01 (1)</td><td>0.98±0.02 (2)</td><td>0.96±0.00 (3)</td><td>0.77±0.03 (7)</td><td>0.73±0.06 (10)</td></tr></table>

<table><tr><td>Avg Rank</td><td>2.050</td><td>2.100</td><td>12.250</td><td>6.250</td><td>4.800</td></tr><tr><td>Dataset</td><td>IForest</td><td>LOF</td><td>k-NN</td><td>SOD</td><td>HBOS</td></tr><tr><td>Abalone</td><td>0.98±0.00 (10)</td><td>1.00±0.00 (5)</td><td>1.00±0.00 (3)</td><td>0.96±0.01 (11)</td><td>0.92±0.01 (12)</td></tr><tr><td>Airfoil</td><td>0.71±0.04 (7)</td><td>0.52±0.04 (10)</td><td>0.51±0.03 (11)</td><td>0.78±0.03 (6)</td><td>0.65±0.02 (8)</td></tr><tr><td>BodyFat</td><td>0.75±0.05 (7)</td><td>0.52±0.06 (10)</td><td>0.52±0.05 (11)</td><td>0.54±0.06 (8)</td><td>0.76±0.05 (6)</td></tr><tr><td>Boston</td><td>0.59±0.04 (7)</td><td>0.50±0.04 (9)</td><td>0.48±0.04 (12)</td><td>0.50±0.04 (8)</td><td>0.67±0.04 (6)</td></tr><tr><td>Concrete</td><td>0.61±0.04 (8)</td><td>0.51±0.05 (10)</td><td>0.50±0.04 (12)</td><td>0.49±0.01 (14)</td><td>0.71±0.06 (5)</td></tr><tr><td>ElNino</td><td>0.97±0.01 (6)</td><td>0.50±0.01 (13)</td><td>0.51±0.02 (11)</td><td>0.90±0.00 (8)</td><td>0.99±0.00 (4)</td></tr><tr><td>Energy</td><td>0.90±0.02 (7)</td><td>0.56±0.03 (11)</td><td>0.94±0.00 (5)</td><td>0.98±0.01 (4)</td><td>0.79±0.03 (10)</td></tr><tr><td>FishWeight</td><td>0.77±0.06 (6)</td><td>0.49±0.11 (14)</td><td>0.51±0.11 (13)</td><td>0.56±0.10 (8)</td><td>0.67±0.05 (7)</td></tr><tr><td>ForestFires</td><td>1.00±0.00 (3)</td><td>0.70±0.03 (13)</td><td>0.74±0.04 (12)</td><td>0.96±0.01 (8)</td><td>1.00±0.00 (1)</td></tr><tr><td>GasEmission</td><td>0.93±0.01 (7)</td><td>0.50±0.02 (12)</td><td>0.51±0.03 (11)</td><td>0.77±0.01 (8)</td><td>0.97±0.00 (5)</td></tr><tr><td>HeartFailure</td><td>0.98±0.00 (4)</td><td>0.56±0.07 (11)</td><td>0.68±0.06 (10)</td><td>0.88±0.01 (7)</td><td>1.00±0.00 (1)</td></tr><tr><td>Hepatitis</td><td>1.00±0.00 (4)</td><td>0.86±0.02 (12)</td><td>0.85±0.02 (13)</td><td>0.96±0.00 (10)</td><td>1.00±0.00 (1)</td></tr><tr><td>IndianLiver</td><td>1.00±0.00 (3)</td><td>0.70±0.04 (12)</td><td>0.80±0.05 (11)</td><td>0.96±0.01 (8)</td><td>1.00±0.00 (1)</td></tr><tr><td>Maintenance</td><td>0.70±0.02 (7)</td><td>0.50±0.03 (14)</td><td>0.53±0.03 (9)</td><td>0.79±0.01 (5)</td><td>0.70±0.05 (6)</td></tr><tr><td>Parkinson</td><td>0.60±0.03 (9)</td><td>0.49±0.03 (13)</td><td>0.50±0.03 (12)</td><td>0.77±0.03 (6)</td><td>0.62±0.03 (8)</td></tr><tr><td>PowerPlant</td><td>0.80±0.02 (6)</td><td>0.50±0.03 (12)</td><td>0.50±0.02 (11)</td><td>0.63±0.02 (8)</td><td>0.71±0.04 (7)</td></tr><tr><td>QSRanking</td><td>0.73±0.05 (7)</td><td>0.48±0.07 (12)</td><td>0.48±0.04 (13)</td><td>0.52±0.04 (9)</td><td>0.86±0.03 (5)</td></tr><tr><td>SynMachine</td><td>0.85±0.03 (11)</td><td>0.92±0.03 (7)</td><td>0.99±0.00 (3)</td><td>0.85±0.03 (10)</td><td>0.68±0.03 (14)</td></tr><tr><td>Toxicity</td><td>0.69±0.03 (8)</td><td>0.60±0.05 (11)</td><td>0.59±0.03 (12)</td><td>0.72±0.04 (7)</td><td>0.69±0.03 (9)</td></tr><tr><td>Yacht</td><td>0.79±0.04 (6)</td><td>0.71±0.04 (11)</td><td>0.83±0.03 (5)</td><td>0.60±0.08 (14)</td><td>0.74±0.05 (9)</td></tr></table>

<table><tr><td>Avg Rank</td><td>6.650</td><td>11.100</td><td>10.000</td><td>8.350</td><td>6.250</td></tr><tr><td>Dataset</td><td>LUNAR</td><td>DTE-C</td><td>DTE-NP</td><td></td><td>TCCM</td></tr><tr><td>Abalone</td><td>1.00±0.00 (6)</td><td>1.00±0.00 (7)</td><td colspan="2">1.00±0.00 (2)</td><td>0.82±0.03 (13)</td></tr><tr><td>Airfoil</td><td>0.83±0.04 (2)</td><td>0.50±0.01 (14)</td><td colspan="2">0.53±0.03 (9)</td><td>0.51±0.05 (12)</td></tr><tr><td>BodyFat</td><td>0.86±0.05 (5)</td><td>0.50±0.09 (13)</td><td colspan="2">0.51±0.06 (12)</td><td>0.54±0.07 (9)</td></tr><tr><td>Boston</td><td>0.69±0.03 (5)</td><td>0.47±0.05 (14)</td><td colspan="2">0.48±0.04 (11)</td><td>0.48±0.07 (10)</td></tr><tr><td>Concrete</td><td>0.68±0.09 (6)</td><td>0.50±0.00 (13)</td><td colspan="2">0.51±0.03 (11)</td><td>0.51±0.05 (9)</td></tr><tr><td>ElNino</td><td>0.71±0.02 (9)</td><td>0.50±0.00 (14)</td><td colspan="2">0.52±0.02 (10)</td><td>0.51±0.02 (12)</td></tr><tr><td>Energy</td><td>0.85±0.02 (8)</td><td>0.50±0.00 (13)</td><td colspan="2">0.93±0.01 (6)</td><td>0.53±0.05 (12)</td></tr><tr><td>FishWeight</td><td>0.84±0.03 (4)</td><td>0.52±0.08 (11)</td><td colspan="2">0.52±0.09 (10)</td><td>0.55±0.09 (9)</td></tr><tr><td>ForestFires</td><td>0.94±0.09 (9)</td><td>0.97±0.01 (7)</td><td colspan="2">0.78±0.02 (11)</td><td>0.99±0.00 (5)</td></tr><tr><td>GasEmission</td><td>0.97±0.05 (6)</td><td>0.50±0.02 (13)</td><td colspan="2">0.52±0.03 (10)</td><td>0.70±0.06 (9)</td></tr><tr><td>HeartFailure</td><td>0.92±0.01 (6)</td><td>0.52±0.05 (12)</td><td colspan="2">0.71±0.06 (9)</td><td>0.80±0.06 (8)</td></tr><tr><td>Hepatitis</td><td>0.98±0.01 (9)</td><td>0.99±0.00 (5)</td><td colspan="2">0.89±0.01 (11)</td><td>0.98±0.02 (7)</td></tr><tr><td>IndianLiver</td><td>0.99±0.01 (5)</td><td>0.58±0.08 (13)</td><td colspan="2">0.87±0.03 (10)</td><td>0.96±0.03 (9)</td></tr><tr><td>Maintenance</td><td>1.00±0.00 (1)</td><td>0.51±0.03 (12)</td><td colspan="2">0.57±0.03 (8)</td><td>0.53±0.03 (10)</td></tr><tr><td>Parkinson</td><td>0.94±0.01 (4)</td><td>0.50±0.00 (11)</td><td colspan="2">0.53±0.03 (10)</td><td>0.71±0.07 (7)</td></tr><tr><td>PowerPlant</td><td>0.89±0.17 (5)</td><td>0.50±0.01 (13)</td><td colspan="2">0.51±0.02 (10)</td><td>0.59±0.13 (9)</td></tr><tr><td>QSRanking</td><td>0.77±0.04 (6)</td><td>0.51±0.05 (10)</td><td colspan="2">0.49±0.04 (11)</td><td>0.57±0.04 (8)</td></tr><tr><td>SynMachine</td><td>0.95±0.02 (6)</td><td>0.87±0.02 (9)</td><td colspan="2">1.00±0.00 (2)</td><td>0.98±0.01 (5)</td></tr><tr><td>Toxicity</td><td>0.76±0.03 (6)</td><td>0.79±0.03 (5)</td><td colspan="2">0.63±0.03 (10)</td><td>0.51±0.03 (14)</td></tr><tr><td>Yacht</td><td>0.76±0.04 (8)</td><td>0.70±0.05 (12)</td><td colspan="2">0.85±0.02 (4)</td><td>0.63±0.02 (13)</td></tr><tr><td>Avg Rank</td><td>5.800</td><td>11.050</td><td colspan="2">8.850</td><td>9.500</td></tr></table>

## D. Instance-Level Interpretability via

## Residual Decomposition (RQ3)

Flow-CAD assigns an anomaly score to each row and records how much each behavioral column contributes to that score. Figure 2 shows both quantities for one anomalous row from the Energy Efficiency dataset and a nearby normal row observed under a similar context. For this illustration, we train Flow-CAD on the QCAD benchmark table using the original Energy Efficiency configuration.

TABLE V  
P@n ON 20 DATASETS. NOTATION FOLLOWS TABLE III.
<table><tr><td>Dataset</td><td>Flow-CAD (ours)</td><td>QCAD</td><td>LoPAD</td><td>ROCOD</td><td>CAD</td></tr><tr><td>Abalone</td><td>0.97±0.02 (1)</td><td>0.91±0.01 (6)</td><td>0.00±0.00 (14)</td><td>0.57±0.01 (10)</td><td>0.94±0.01 (3)</td></tr><tr><td>Airfoil</td><td>0.43±0.07 (3)</td><td>0.67±0.03 (1)</td><td>0.03±0.02 (14)</td><td>0.41±0.04 (4)</td><td>0.45±0.04 (2)</td></tr><tr><td>BodyFat</td><td>0.74±0.11 (2)</td><td>0.61±0.06 (4)</td><td>0.06±0.06 (14)</td><td>0.70±0.06 (3)</td><td>0.77±0.08 (1)</td></tr><tr><td>Boston</td><td>0.60±0.09 (2)</td><td>0.64±0.07 (1)</td><td>0.08±0.01 (8.5)</td><td>0.36±0.04 (4)</td><td>0.39±0.08 (3)</td></tr><tr><td>Concrete</td><td>0.61±0.11 (1)</td><td>0.61±0.08 (2)</td><td>0.11±0.04 (7)</td><td>0.33±0.05 (3)</td><td>0.31±0.06 (4)</td></tr><tr><td>ElNino</td><td>0.73±0.04 (3)</td><td>0.94±0.01 (1)</td><td>0.12±0.03 (7)</td><td>0.40±0.05 (6)</td><td>0.91±0.02 (2)</td></tr><tr><td>Energy</td><td>0.88±0.04 (1)</td><td>0.83±0.03 (3)</td><td>0.00±0.01 (14)</td><td>0.43±0.05 (7)</td><td>0.85±0.04 (2)</td></tr><tr><td>FishWeight</td><td>0.83±0.08 (1)</td><td>0.76±0.11 (2)</td><td>0.12±0.10 (9)</td><td>0.41±0.07 (4)</td><td>0.64±0.11 (3)</td></tr><tr><td>ForestFires</td><td>0.91±0.05 (4)</td><td>0.95±0.03 (2)</td><td>0.10±0.03 (14)</td><td>0.40±0.03 (10)</td><td>0.88±0.02 (5)</td></tr><tr><td>GasEmission</td><td>0.92±0.01 (2)</td><td>0.94±0.01 (1)</td><td>0.00±0.00 (14)</td><td>0.84±0.03 (4)</td><td>0.86±0.02 (3)</td></tr><tr><td>HeartFailure</td><td>0.90±0.02 (2)</td><td>0.84±0.06 (3)</td><td>0.00±0.00 (14)</td><td>0.15±0.13 (11)</td><td>0.72±0.06 (5)</td></tr><tr><td>Hepatitis</td><td>0.97±0.01 (1)</td><td>0.91±0.03 (3)</td><td>0.05±0.03 (14)</td><td>0.65±0.05 (8)</td><td>0.81±0.04 (5)</td></tr><tr><td>IndianLiver</td><td>0.90±0.03 (2)</td><td>0.84±0.03 (4)</td><td>0.05±0.03 (14)</td><td>0.51±0.05 (8)</td><td>0.76±0.11 (6)</td></tr><tr><td>Maintenance</td><td>0.70±0.09 (4)</td><td>0.92±0.02 (1)</td><td>0.01±0.01 (8.5)</td><td>0.01±0.01 (12.5)</td><td>0.85±0.04 (3)</td></tr><tr><td>Parkinson</td><td>0.94±0.03 (1)</td><td>0.86±0.03 (2)</td><td>0.02±0.02 (7)</td><td>0.55±0.05 (4)</td><td>0.71±0.04 (3)</td></tr><tr><td>PowerPlant</td><td>0.71±0.04 (1)</td><td>0.49±0.04 (4)</td><td>0.01±0.01 (10)</td><td>0.69±0.04 (2)</td><td>0.53±0.06 (3)</td></tr><tr><td>QSRanking</td><td>0.69±0.09 (2.5)</td><td>0.71±0.05 (1)</td><td>0.02±0.02 (14)</td><td>0.52±0.07 (4)</td><td>0.69±0.09 (2.5)</td></tr><tr><td>SynMachine</td><td>1.00±0.01 (1)</td><td>0.96±0.02 (2)</td><td>0.41±0.03 (13)</td><td>0.74±0.08 (7)</td><td>0.48±0.11 (11)</td></tr><tr><td>Toxicity</td><td>0.57±0.07 (1)</td><td>0.47±0.04 (2)</td><td>0.17±0.04 (5)</td><td>0.47±0.06 (3)</td><td>0.46±0.06 (4)</td></tr><tr><td>Yacht</td><td>0.84±0.10 (1)</td><td>0.81±0.05 (2)</td><td>0.56±0.06 (3)</td><td>0.30±0.08 (5)</td><td>0.26±0.06 (9)</td></tr></table>

<table><tr><td>Avg Rank</td><td>1.825</td><td>2.350</td><td>10.900</td><td>5.975</td><td>3.975</td></tr><tr><td>Dataset</td><td>IForest</td><td>LOF</td><td>k-NN</td><td>SOD</td><td>HBOS</td></tr><tr><td>Abalone</td><td>0.43±0.08 (11)</td><td>0.92±0.02 (5)</td><td>0.94±0.01 (4)</td><td>0.74±0.01 (9)</td><td>0.21±0.06 (13)</td></tr><tr><td>Airfoil</td><td>0.12±0.04 (7.5)</td><td>0.04±0.03 (12.5)</td><td>0.06±0.03 (9)</td><td>0.13±0.03 (6)</td><td>0.12±0.01 (7.5)</td></tr><tr><td>BodyFat</td><td>0.19±0.07 (7)</td><td>0.09±0.09 (11.5)</td><td>0.11±0.06 (8)</td><td>0.08±0.06 (13)</td><td>0.25±0.06 (6)</td></tr><tr><td>Boston</td><td>0.11±0.04 (7)</td><td>0.08±0.05 (8.5)</td><td>0.06±0.04 (12)</td><td>0.06±0.04 (13)</td><td>0.15±0.06 (5)</td></tr><tr><td>Concrete</td><td>0.08±0.03 (8)</td><td>0.07±0.03 (11)</td><td>0.06±0.02 (13)</td><td>0.06±0.02 (12)</td><td>0.29±0.06 (5)</td></tr><tr><td>EINino</td><td>0.53±0.07 (5)</td><td>0.01±0.01 (11.5)</td><td>0.01±0.01 (11.5)</td><td>0.01±0.00 (14)</td><td>0.69±0.03 (4)</td></tr><tr><td>Energy</td><td>0.49±0.04 (5)</td><td>0.11±0.03 (10.5)</td><td>0.30±0.03 (9)</td><td>0.80±0.06 (4)</td><td>0.34±0.06 (8)</td></tr><tr><td>FishWeight</td><td>0.23±0.08 (7)</td><td>0.09±0.06 (13.5)</td><td>0.10±0.08 (11)</td><td>0.09±0.06 (13.5)</td><td>0.30±0.15 (6)</td></tr><tr><td>ForestFires</td><td>0.94±0.03 (3)</td><td>0.16±0.05 (12)</td><td>0.13±0.04 (13)</td><td>0.60±0.04 (9)</td><td>0.99±0.01 (1)</td></tr><tr><td>GasEmission</td><td>0.05±0.02 (7)</td><td>0.02±0.01 (10)</td><td>0.01±0.01 (13)</td><td>0.02±0.01 (10)</td><td>0.22±0.05 (6)</td></tr><tr><td>HeartFailure</td><td>0.80±0.03 (4)</td><td>0.12±0.06 (12)</td><td>0.18±0.06 (10)</td><td>0.32±0.06 (8)</td><td>0.93±0.03 (1)</td></tr><tr><td>Hepatitis</td><td>0.90±0.03 (4)</td><td>0.07±0.04 (13)</td><td>0.08±0.04 (12)</td><td>0.33±0.03 (10)</td><td>0.96±0.02 (2)</td></tr><tr><td>IndianLiver</td><td>0.89±0.03 (3)</td><td>0.07±0.03 (13)</td><td>0.20±0.10 (11)</td><td>0.41±0.08 (9)</td><td>0.95±0.02 (1)</td></tr><tr><td>Maintenance</td><td>0.02±0.01 (6)</td><td>0.00±0.00 (14)</td><td>0.01±0.00 (12.5)</td><td>0.01±0.01 (8.5)</td><td>0.36±0.09 (5)</td></tr><tr><td>Parkinson</td><td>0.02±0.01 (8.5)</td><td>0.01±0.02 (12)</td><td>0.01±0.01 (14)</td><td>0.02±0.01 (8.5)</td><td>0.01±0.01 (11)</td></tr><tr><td>PowerPlant</td><td>0.08±0.03 (8)</td><td>0.01±0.01 (11.5)</td><td>0.02±0.02 (9)</td><td>0.01±0.01 (11.5)</td><td>0.26±0.04 (6)</td></tr><tr><td>QSRanking</td><td>0.23±0.09 (6)</td><td>0.09±0.05 (8)</td><td>0.08±0.04 (10)</td><td>0.07±0.03 (13)</td><td>0.46±0.06 (5)</td></tr><tr><td>SynMachine</td><td>0.41±0.05 (12)</td><td>0.71±0.05 (8)</td><td>0.79±0.04 (5)</td><td>0.66±0.04 (9)</td><td>0.27±0.04 (14)</td></tr><tr><td>Toxicity</td><td>0.09±0.03 (11)</td><td>0.07±0.02 (12)</td><td>0.05±0.02 (13)</td><td>0.10±0.05 (9)</td><td>0.11±0.03 (8)</td></tr><tr><td>Yacht</td><td>0.32±0.07 (4)</td><td>0.20±0.05 (13)</td><td>0.23±0.05 (11.5)</td><td>0.26±0.06 (8)</td><td>0.29±0.09 (6)</td></tr></table>

<table><tr><td>Avg Rank</td><td>6.700</td><td>11.125</td><td>10.575</td><td>9.900</td><td>6.025</td></tr><tr><td>Dataset</td><td>LUNAR</td><td>DTE-C</td><td>DTE-NP</td><td></td><td>TCCM</td></tr><tr><td>Abalone</td><td>0.90±0.02 (7.5)</td><td>0.90±0.01 (7.5)</td><td>0.95±0.02 (2)</td><td></td><td>0.32±0.09 (12)</td></tr><tr><td>Airfoil</td><td>0.36±0.03 (5)</td><td>0.04±0.03 (11)</td><td>0.06±0.03 (10)</td><td></td><td>0.04±0.03 (12.5)</td></tr><tr><td>BodyFat</td><td>0.44±0.09 (5)</td><td>0.10±0.13 (9)</td><td>0.09±0.06 (10)</td><td></td><td>0.09±0.08 (11.5)</td></tr><tr><td>Boston</td><td>0.14±0.04 (6)</td><td>0.07±0.03 (10)</td><td>0.04±0.04 (14)</td><td></td><td>0.07±0.04 (11)</td></tr><tr><td>Concrete</td><td>0.12±0.08 (6)</td><td>0.08±0.02 (9)</td><td>0.07±0.02 (10)</td><td></td><td>0.05±0.02 (14)</td></tr><tr><td>ElNino</td><td>0.03±0.01 (8)</td><td>0.01±0.01 (10)</td><td>0.01±0.00 (9)</td><td></td><td>0.01±0.01 (13)</td></tr><tr><td>Energy</td><td>0.46±0.05 (6)</td><td>0.08±0.03 (13)</td><td>0.11±0.03 (10.5)</td><td></td><td>0.10±0.04 (12)</td></tr><tr><td>FishWeight</td><td>0.33±0.07 (5)</td><td>0.10±0.09 (11)</td><td>0.10±0.05 (11)</td><td></td><td>0.13±0.08 (8)</td></tr><tr><td>ForestFires</td><td>0.73±0.13 (8)</td><td>0.73±0.07 (7)</td><td>0.21±0.03 (11)</td><td></td><td>0.86±0.05 (6)</td></tr><tr><td>GasEmission</td><td>0.39±0.16 (5)</td><td>0.02±0.01 (12)</td><td>0.02±0.01 (10)</td><td></td><td>0.02±0.01 (8)</td></tr><tr><td>HeartFailure</td><td>0.46±0.06 (6)</td><td>0.10±0.04 (13)</td><td>0.22±0.06 (9)</td><td></td><td>0.41±0.12 (7)</td></tr><tr><td>Hepatitis</td><td>0.57±0.11 (9)</td><td>0.78±0.04 (6)</td><td>0.08±0.05 (11)</td><td></td><td>0.70±0.27 (7)</td></tr><tr><td>IndianLiver</td><td>0.82±0.04 (5)</td><td>0.11±0.06 (12)</td><td>0.31±0.05 (10)</td><td></td><td>0.55±0.11 (7)</td></tr><tr><td>Maintenance</td><td>0.87±0.09 (2)</td><td>0.01±0.01 (10.5)</td><td>0.01±0.01 (7)</td><td></td><td>0.01±0.01 (10.5)</td></tr><tr><td>Parkinson</td><td>0.11±0.03 (6)</td><td>0.02±0.01 (10)</td><td>0.01±0.01 (13)</td><td></td><td>0.23±0.14 (5)</td></tr><tr><td>PowerPlant</td><td>0.47±0.15 (5)</td><td>0.01±0.01 (13.5)</td><td>0.01±0.01 (13.5)</td><td></td><td>0.26±0.13 (7)</td></tr><tr><td>QSRanking</td><td>0.11±0.06 (7)</td><td>0.09±0.06 (9)</td><td>0.08±0.06 (11.5)</td><td></td><td>0.08±0.05 (11.5)</td></tr><tr><td>SynMachine</td><td>0.77±0.05 (6)</td><td>0.55±0.05 (10)</td><td>0.93±0.02 (3)</td><td></td><td>0.82±0.04 (4)</td></tr><tr><td>Toxicity</td><td>0.17±0.04 (6)</td><td>0.13±0.03 (7)</td><td>0.05±0.01 (14)</td><td></td><td>0.09±0.09 (10)</td></tr><tr><td>Yacht</td><td>0.23±0.10 (11.5)</td><td>0.24±0.08 (10)</td><td>0.28±0.06 (7)</td><td></td><td>0.19±0.05 (14)</td></tr><tr><td>Avg Rank</td><td>6.250</td><td>10.025</td><td>9.825</td><td></td><td>9.550</td></tr></table>

1) Why the anomalous row receives a high score. Figure 2(a) shows sampled points along the paths to the two observed rows. At each point, the arrow is the difference between the velocity predicted by Flow-CAD and the target velocity for that path. The anomalous row repeatedly produces longer and more dispersed arrows than the nearby normal row. Its high score therefore reflects large residuals across many Monte Carlo draws, rather than one unusual sample.

2) Which columns drive the score. Figure 2(b) reports the residual contribution of each behavioral column after MAD normalization and truncation. For the anomalous row, $Y _ { 1 }$ and $Y _ { 2 }$ contribute about 4.1 and 2.4, respectively, whereas both contributions are approximately 0.4 or less for the normal row. Because the contributions $s _ { q }$ sum to the raw score in (4), their relative sizes indicate which column should be inspected first. In this example, the larger contribution of $Y _ { 1 }$ directs attention to that column when the row is compared with observations from similar contexts.

![](images/14b230b3e1b12329bcef9ed51445d8d9afdad55ede1944a5313fb53d0a4879bd.jpg)  
Fig. 2. Illustrative raw-score decomposition for two Energy Efficiency instances with behavioral features $( Y _ { 1 } , Y _ { 2 } )$ . (a) Sampled intermediate states and signed velocity differences $\Delta { v } ;$ the score uses their coordinate-wise magnitudes $\pmb { r } = | \Delta \pmb { v } |$ . (b) Feature contributions estimated with a separate set of Monte Carlo draws.

## E. Ablation and Sensitivity Analysis (RQ4)

We conduct a series of ablation studies to assess the sensitivity of Flow-CAD to key design choices, including residual normalization, truncation, aggregation norms, and Monte Carlo sampling. Figures 3–7 vary one scoring choice at a time on the same tables and 1,000-epoch fits as the main benchmark. They report means and standard deviations across the seven benchmark runs, using MAD normalization, $\tau = 1 0 ,$ $L _ { 1 }$ aggregation, and $K _ { t } = K _ { z } = 1 6$ as the defaults. We also test whether repeated scoring flags the same rows and how the encoding of categorical context affects detection.

1) Ablation studies on residual normalization and truncation: We compare the default configuration (MAD normalization with truncation) against variants that remove either component. Normalization puts residuals from different behavioral columns on a comparable scale, while truncation limits the influence of any single residual. Across the 20 datasets, the complete score has the highest mean on all three metrics. Truncation provides the larger average gain: removing it lowers mean PR AUC by 3.3 percentage points and mean P@n by 1.5 points. MAD normalization gives smaller gains on average, but is important when residual scales differ substantially across columns. On HeartFailure and GasEmission, the complete score gives PR AUCs of 0.974 and 0.976, versus 0.893 and 0.941 without normalization and 0.738 and 0.794 without truncation.

2) Sensitivity to truncation threshold τ: Figure 4 varies the positive truncation threshold τ from 2 to 100 while keeping MAD normalization enabled. Across $\tau ~ \in ~ \{ 5 , 1 0 , 2 0 \}$ , the median per-dataset PR AUC range is only 0.015. At the default $\tau = 1 0$ , PR AUC is within one percentage point of the best of these three settings on 15 datasets and within two points on all 20. Thus, $\tau = 1 0$ lies in a broad stable region rather than depending on a narrow optimum. The no-truncation case is reported separately in Figure 3.

3) Sensitivity to residual aggregation norm: Figure 5 compares the default $L _ { 1 }$ norm with the $L _ { 2 }$ and $L _ { \infty }$ norms. Across the 20 datasets, $L _ { 1 }$ has the highest mean for all three metrics: 0.838 for PR AUC, 0.964 for ROC AUC, and 0.792 for $\mathrm { P @ } n$

![](images/e1a9750f69b251972252d0a223d431f5c3ce07de73613e32ee499e4355a26d7f.jpg)

![](images/37b1ed52c5e4aeec257551b3ea016572f961b1f23fe3da010689ed8c2bc35e49.jpg)

![](images/902e565983bc60dd87696b27762707163cfad5610ce119a23d8d39cf9ada4cda.jpg)  
Fig. 3. Ablation on residual normalization and truncation across all datasets. Panels show PR AUC, ROC AUC, and P@n (top to bottom). We compare the default configuration (MAD normalization with truncation) against variants without normalization and without truncation.

![](images/2cfd508b712c528ef1e428179fc3906672e3d2099a7ee3851e528f3a9befa0e9.jpg)  
Fig. 4. Sensitivity to truncation with MAD normalization. The threshold τ varies from 2 to 100.

The corresponding means are 0.829, 0.964, and 0.789 for $L _ { 2 } ,$ and 0.710, 0.927, and 0.636 for $L _ { \infty }$ . Moreover, $L _ { 1 }$ is best or tied on 13, 13, and 15 datasets, respectively, and is within one percentage point of the best norm in 52 of the 60 dataset–metric comparisons. We therefore retain $L _ { 1 }$ as the most reliable overall choice; it also preserves the additive column-wise decomposition used in Section III-C.

4) Sensitivity to Monte Carlo sampling budgets: Figures 6 and 7 show the effect of varying $K _ { t }$ (time draws per Gaussian draw) and $K _ { z }$ (Gaussian draws), respectively. We hold one budget at 16 while varying the other and reuse draws across settings. Residual normalization and truncation remain enabled.

![](images/7793546afc1541af74357a0561b8d28d543ee3075606a4364f99659b82a9be80.jpg)

![](images/c3e995bf74a265bc3bd1f8fbed8e5e95ec5a106f9af6ee393ff09af6afa77358.jpg)

![](images/7780deab7a06cfb315cc7f367eae26bcd11fd2cbfa57161e93926fa3e1d382e4.jpg)  
Fig. 5. Residual aggregation with $L _ { 1 } , L _ { 2 } ,$ and $L _ { \infty }$ norms. Panels show PR AUC, ROC AUC, and P@n (top to bottom).

![](images/0790e9442ee759c7d77873557e78f3808c6e53ba890b749d1a1982ab8a7d2e73.jpg)  
Fig. 6. Sensitivity to the number of time draws $K _ { t }$ per Gaussian draw with MAD normalization and truncation.

Both budgets matter most when they are small. Increasing $K _ { z }$ from 2 to 16 improves PR AUC on all 20 datasets, with a median gain of 5.3 percentage points. Increasing $K _ { t }$ over the same range also improves all 20, with a median gain of 1.0 point. The gains then taper: moving from 16 to 128 adds median gains of 0.5 points for $K _ { z }$ and 0.1 points for $K _ { t }$ The larger effect of $K _ { z }$ agrees with Proposition 2: additional Gaussian draws reduce both variance components, whereas additional time draws reduce only the inner component.

5) Stability of rows selectedfor review: In practice, repeated runs should flag largely the same rows for inspection. We test this by scoring each table twice with independent Monte Carlo draws and comparing the two top-5% lists. The experiment covers six datasets, each with three single-column variants, for a total of 18 fitted models. We fix $K _ { t } = 1 6$ and increase $K _ { z }$ from 16 to 64. The median overlap rises from 80.4% to 91.2%, and 17 of the 18 models show higher overlap. More draws therefore make row selection more stable.

![](images/76ec6fc42f26310a55bbf0fcff72ff68d744875f813f18d13e431f8236712afe.jpg)  
Fig. 7. Sensitivity to the number of Gaussian draws $K _ { z }$ with MAD normal ization and truncation.

6) Representing categorical context: Categorical values such as countries have no natural numerical order, yet scalar codes in [0, 1] impose one. We compare scalar and one-hot encodings on FishWeight and QSRanking using the same seven benchmark runs (data seeds 43–49) and 1,000 training epochs. One-hot encoding performs better on all three metrics in both datasets. On FishWeight, PR AUC, ROC AUC, and P@n rise from 0.859, 0.955, and 0.781 to 0.904, 0.973, and 0.829. On QSRanking, they rise from 0.693, 0.911, and 0.646 to 0.745, 0.936, and 0.693. QSRanking has 57 country categories, so one-hot encoding is the more natural choice.

## F. Controlled Sign-Parity Experiment

Results on real datasets do not show whether a detector has learned the joint relationship among behavioral columns or is relying on simpler one- or two-column patterns. We therefore construct a sign-parity task in which the context mismatch appears only in the joint pattern of three columns. Any one or two columns have the same distribution in both contexts, so success requires modeling all three together.

1) Test construction and evaluation. Each block contains all eight sign patterns for a fixed set of absolute values, and the parity of the negative signs determines the correct context. We train for 1,000 epochs on 800 rows, assigning the wrong context to 5% of them. The seven runs pair data seeds 43–49 with training seeds 202–208. We then score 800 new vectors under both the correct and wrong contexts. A pair earns one point when the wrong context receives a higher anomaly score and half a point for a tie. For each fitted flow, both contexts and all eight sign patterns in a test block share the same draws.

2) Comparison methods. The conditional-mean MLP tests whether mean prediction is sufficient; it uses the same encoder, hidden widths, initialization seed, and training MAD scales as Flow-CAD. Coordinatewise Flow fits one flow per column and adds their scores. Zero-context Flow keeps the joint model but removes context by setting its context inputs to zero.

3) Why separate column scores are insufficient. Fix one sign within a block. Across the four remaining patterns, each column takes the same values in both contexts. For any method that adds separate column scores, the four score gaps (wrong context minus correct context) must therefore sum to zero. If three gaps exceed the tie tolerance, the fourth must favor the correct context and count as a loss. Even with half credit for ties, paired accuracy cannot exceed 75%.

4) Results. Flow-CAD gives the wrong context a higher anomaly score for every test pair in all seven runs, achieving 100% paired accuracy (Table VI). The three alternatives remain near 50%. Flow-CAD gains 53.6 percentage points over the conditional-mean MLP (95% paired bootstrap interval across seeds: [50.0, 60.7]). The joint flow therefore detects a threecolumn mismatch that is not captured by mean prediction, separate column scores, or a joint model without context.

TABLE VI  
MEAN PAIRED ACCURACY (%) OVER SEVEN SIGN-PARITY RUNS.
<table><tr><td>Method</td><td>Paired accuracy</td></tr><tr><td>Flow-CAD</td><td>100.00</td></tr><tr><td>Conditional-mean MLP</td><td>46.43</td></tr><tr><td>Coordinatewise Flow</td><td>49.98</td></tr><tr><td>Zero-context Flow</td><td>50.00</td></tr></table>

## G. Runtime Comparison

We compare Flow-CAD and QCAD on tables of different sizes, using one Apple M4 CPU thread per run. Scoring time covers row scores and column contributions. Total time also covers data loading, fitting or setup, and metric computation.

Effect of table size. To isolate the effect of row count on runtime, we test four nested PowerPlant subsets from 1,000 to 8,000 rows at three seeds (12 paired runs), keeping the columns fixed. QCAD uses 500 neighbors; Flow-CAD trains for 1,000 epochs. Flow-CAD is faster in every run. At 8,000 rows, its median scoring and total times are 2.57 and 38.33 CPU seconds; QCAD takes 54.88 and 55.99 seconds, respectively.

Timing on the full benchmark. To see if the PowerPlant result holds elsewhere, we compare all 20 benchmarks (159 to 20,000 rows). Flow-CAD scores each table 4.0× to 164.1× faster and has lower total time on 16 of them. On ElNino, total time falls from 595 CPU seconds with QCAD to 69 with Flow-CAD. Flow-CAD reuses one fitted model across rows; QCAD builds a local neighborhood and estimates conditional quantiles for each row.

For each row, Flow-CAD evaluates K<sub>z</sub> Gaussian draws at $K _ { t }$ time points. With fixed network widths, scoring N rows with $P$ encoded context columns and Q behavioral columns requires $\mathcal { O } ( N K _ { z } K _ { t } ( P + Q ) )$ ) operations, while storing the draws requires $\mathcal { O } ( N K _ { z } ( Q + K _ { t } ) )$ memory.

Scaling to large tables. Since the benchmarks stop at 20,000 rows, we also test larger inputs by fitting Flow-CAD on 8,000 synthetic rows and scoring in 1,000-row batches with the training MAD scales. It scores one million rows with 64 encoded context and 10 behavioral columns in 564 CPU seconds. With 256 context and 128 behavioral columns, it scores 100,000 rows in 228 seconds. Peak memory is 322 and 364 MiB, respectively; batch scoring does not raise the peak reached during fitting.

![](images/d15a5193f7ffafb170dabbe6f76848d4a7afbaadc6a3957616e462fd2cfc2c0f.jpg)  
Fig. 8. Runtime comparison between Flow-CAD and QCAD on all 20 benchmark datasets (data seed 42, one Apple M4 CPU thread per run). Orange shows the time to compute row scores and feature contributions. Blue includes data loading and metric computation for both methods, plus the 1,000-epoch model fit for Flow-CAD and context encoding and Gower-neighborhood construction for QCAD. Flow-CAD is shown with solid bars and QCAD with striped bars. The vertical axis is logarithmic. Bar labels show total CPU time.

## V. CONCLUSION

We proposed Flow-CAD, a contextual anomaly detection framework for tabular data based on Flow Matching. Flow-CAD learns a single global and amortized model and employs a residual-based scoring mechanism that avoids explicit likelihood evaluation while providing native per-feature explanations. Experiments on 20 real-world datasets show that Flow-CAD has the best average rank for PR AUC, ROC AUC, and P@n. The ablations show stable performance around the default truncation and sampling settings, while the timing study shows faster scoring than QCAD on every benchmark table. These results demonstrate the effectiveness and practicality of Flow Matching for contextual anomaly detection. Future work will explore extensions to structured and temporal contexts.

## ACKNOWLEDGMENT

This publication is part of the project Digital Twin with project number P18-03 of the research programme TTW Perspective, which is (partly) financed by the Dutch Research Council (NWO).

We used GPT for language editing and for programming assistance with running experiments.

## REFERENCES

[1] X. Song, M. Wu, C. Jermaine, and S. Ranka, “Conditional anomaly detection,” IEEE Transactions on knowledge and Data Engineering, vol. 19, no. 5, pp. 631–645, 2007.

[2] Z. Li and M. Van Leeuwen, “Explainable contextual anomaly detection using quantile regression forests,” Data Mining and Knowledge Discovery, vol. 37, no. 6, pp. 2517–2563, 2023.

[3] H. Kayan, R. Heartfield, O. Rana, P. Burnap, and C. Perera, “Casper: Context-aware iot anomaly detection

system for industrial robotic arms,” ACM Transactions on Internet of Things, vol. 5, no. 3, pp. 1–36, 2024.

[4] M. A. Hayes and M. A. Capretz, “Contextual anomaly detection framework for big sensor data,” Journal of Big Data, vol. 2, no. 1, p. 2, 2015.

[5] A. M. Kosek, “Contextual anomaly detection for cyberphysical security in smart grids based on an artificial neural network model,” in 2016 joint workshop on cyberphysical security and resilience in smart grids (CPSR-SG). IEEE, 2016, pp. 1–6.

[6] X. Ding, H. Su, Z. Qian, W. Cui, S. Chen, Z. Liang, C. Wang, and H. Wang, “Cbaclean: A comprehensive system for recommending data cleaning solutions through cost-benefit analysis in data quality management,” in 2025 IEEE 41st International Conference on Data Engineering (ICDE), 2025.

[7] S. Siddiqi, R. Kern, and M. Boehm, “Saga: A scalable framework for optimizing data cleaning pipelines for machine learning applications,” Proceedings of the ACM on Management of Data, vol. 1, no. 3, 2023.

[8] S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, and A. Grafberger, “Automating large-scale data quality verification,” Proceedings of the VLDB Endowment, vol. 11, no. 12, pp. 1781–1794, 2018.

[9] J. Liang and S. Parthasarathy, “Robust contextual outlier detection: Where context meets sparsity,” in Proceedings of the 25th ACM international on conference on information and knowledge management, 2016, pp. 2167–2172.

[10] A. X. Wang, S. S. Chukova, C. R. Simpson, and B. P. Nguyen, “Challenges and opportunities of generative models on tabular data,” Applied Soft Computing, vol. 166, p. 112223, 2024.

[11] Z. Li, Q. Huang, L. Yang, J. Shi, Z. Yang, N. van Stein, T. Bäck, and M. van Leeuwen, “Diffusion and flow matching models for tabular data: A survey,” arXiv:2502.17119v2, May 2026. [Online]. Available: https://arxiv.org/abs/2502.17119v2

[12] H. Kamkari, B. L. Ross, J. C. Cresswell, A. L. Caterini, R. Krishnan, and G. Loaiza-Ganem, “A geometric explanation of the likelihood ood detection paradox,” in International Conference on Machine Learning. PMLR, 2024, pp. 22 908–22 935.

[13] E. Nalisnick, A. Matsukawa, Y. W. Teh, D. Gorur, and B. Lakshminarayanan, “Do deep generative models know what they don’t know?” in International Conference on Learning Representations, 2019.

[14] Y. Lipman, R. T. Chen, H. Ben-Hamu, M. Nickel, and M. Le, “Flow matching for generative modeling,” in 11th International Conference on Learning Representations, ICLR 2023, 2023.

[15] M. Albergo and E. Vanden-Eijnden, “Building normalizing flows with stochastic interpolants,” in ICLR 2023 Conference, 2023.

[16] V. Chandola, A. Banerjee, and V. Kumar, “Anomaly detection: A survey,” ACM computing surveys (CSUR), vol. 41, no. 3, pp. 1–58, 2009.

[17] S. Han, X. Hu, H. Huang, M. Jiang, and Y. Zhao, “ADBench: Anomaly detection benchmark,” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 32 142–32 159.

[18] S. Lu, L. Liu, J. Li, T. D. Le, and J. Liu, “Lopad: A local prediction approach to anomaly detection,” in Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 2020, pp. 660–673.

[19] Y. Ang, P. Yao, Y. Bao, Y. Feng, Q. Huang, A. K. H. Tung, and Z. Huang, “RFOD: Random forest-based outlier detection for mixed-type tabular data,” in 2026 IEEE 42nd International Conference on Data Engineering (ICDE), 2026, pp. 2192–2205.

[20] L. Bindini, L. Perini, S. Nistri, J. Davis, and P. Frasconi, “Dealing with uncertainty in contextual anomaly detection,” Transactions on Machine Learning Research, 2026. [Online]. Available: https://openreview. net/forum?id=yLoXQDNwwa

[21] G. Pang, C. Shen, L. Cao, and A. V. D. Hengel, “Deep learning for anomaly detection: A review,” ACM computing surveys (CSUR), vol. 54, no. 2, pp. 1–38, 2021.

[22] D. Rezende and S. Mohamed, “Variational inference with normalizing flows,” in International conference on machine learning. PMLR, 2015, pp. 1530–1538.

[23] X. Liu, C. Gong, and Q. Liu, “Flow straight and fast: Learning to generate and transfer data with rectified flow,” in The Eleventh International Conference on Learning Representations (ICLR), 2023.

[24] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio, “Improving and generalizing flow-based generative models with minibatch optimal transport,” Transactions on Machine Learning Research, pp. 1–34, 2024.

[25] Z. Li, Q. Huang, Y. Zhu, L. Yang, M. M. Amiri, N. van Stein, and M. van Leeuwen, “Scalable, explainable and provably robust anomaly detection with one-step flow matching,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[26] S. Chen, M. Moradi, K. Paynabar, and H. Yan, “Flow mismatching: Unsupervised anomaly detection via velocity discrepancies in flow matching models,” arXiv preprint arXiv:2605.23070, 2026.

[27] P. Konz, T. Medi, and M. Keuper, “Unsupervised anomaly detection using flow matching on tabular data,” arXiv preprint arXiv:2608.19801, 2026.

[28] V. Yepmo, G. Smits, and O. Pivert, “Anomaly explanation: A review,” Data & Knowledge Engineering, vol. 137, p. 101946, 2022.

[29] Z. Li, Y. Zhu, and M. Van Leeuwen, “A survey on explainable anomaly detection,” ACM Transactions on Knowledge Discovery from Data, vol. 18, no. 1, pp. 1–54, 2023.

[30] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[31] Y. S. Abu-Mostafa, M. Magdon-Ismail, and H.-T. Lin, Learning from data. AMLBook New York, 2012, vol. 4.

[32] W. Hoeffding, “Probability inequalities for sums of bounded random variables,” Journal of the American statistical association, vol. 58, no. 301, pp. 13–30, 1963.

[33] S. Boucheron, G. Lugosi, and P. Massart, Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 02 2013. [Online]. Available: https://doi.org/10.1093/acprof: oso/9780199535255.001.0001

[34] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation forest,” in 2008 eighth ieee international conference on data mining. IEEE, 2008, pp. 413–422.

[35] M. M. Breunig, H.-P. Kriegel, R. T. Ng, and J. Sander, “Lof: identifying density-based local outliers,” in Proceedings of the 2000 ACM SIGMOD international conference on Management of data, 2000, pp. 93–104.

[36] F. Angiulli and C. Pizzuti, “Fast outlier detection in high dimensional spaces,” in European conference on principles of data mining and knowledge discovery. Springer, 2002, pp. 15–27.

[37] H.-P. Kriegel, P. Kröger, E. Schubert, and A. Zimek, “Outlier detection in axis-parallel subspaces of high dimensional data,” in Pacific-asia conference on knowledge discovery and data mining. Springer, 2009, pp. 831–838.

[38] M. Goldstein and A. Dengel, “Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm,” KI-2012: poster and demo track, vol. 1, pp. 59–63, 2012.

[39] A. Goodge, B. Hooi, S.-K. Ng, and W. S. Ng, “Lunar: Unifying local outlier detection methods via graph neural networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, 2022, pp. 6737–6745.

[40] V. Livernoche, V. Jain, Y. Hezaveh, and S. Ravanbakhsh, “On diffusion modeling for anomaly detection,” in The Twelfth International Conference on Learning Representations, 2024.

[41] Y. Zhao, Z. Nasrullah, and Z. Li, “PyOD: A python toolbox for scalable outlier detection,” Journal of Machine Learning Research, vol. 20, no. 96, pp. 1–7, 2019.