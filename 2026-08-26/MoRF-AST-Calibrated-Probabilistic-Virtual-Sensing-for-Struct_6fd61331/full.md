# MoRF-AST: Calibrated Probabilistic Virtual Sensing for Structural Monitoring under Changing Operating Conditions

Wingho Feng<sup>1</sup>, Quanwang Li<sup>1</sup>, Ming Zhong<sup>2</sup>, Jingyu Yang<sup>2</sup>, and Chen Wang<sup>1,∗</sup>

<sup>1</sup>Department of Civil Engineering, Tsinghua University, Beijing 100084, China <sup>2</sup>Xiandai Investment Co., Ltd., Changsha 410004, China <sup>∗</sup>Corresponding author: chwang@tsinghua.edu.cn

August 2026

## Abstract

Probabilistic full-field reconstruction provides uncertainty-aware response evidence for structural reliability assessment, yet inference from sparse and noisy measurements remains underdetermined. Most existing methods overlook shifts between ofline training and operational distributions. Under such shifts, posterior intervals may become miscalibrated, causing the reported uncertainty to lose its probabilistic meaning. This study proposes Modal Residual Flow Matching with Context-Conditioned Afine Spread Transport (MoRF-AST) for calibrated structural virtual sensing under changing operating conditions. MoRF constructs an analytic Gaussian reference posterior in normalized modal coordinates and trains a conditional flow only on posterior-whitened residuals. At deployment, AST estimates response scale from historical measurements at installed sensors and uses gated, mean-preserving Bures–Wasserstein transport to adjust posterior spread. On a bridge-deck benchmark, MoRF achieves a posterior-mean normalized root-mean-square error (NRMSE) of 7.20%, compared with 16.1% and 17.9% for two direct conditional flows. Across eight shifted trafic domains, AST reduces MoRF’s cross-domain average coverage error from 0.0535 to 0.0236, a 55.9% reduction, while preserving posterior-mean accuracy. The same transport does not improve the tested alternatives in aggregate, showing that calibration gains require its direction to match the base posterior’s dispersion bias. MoRF-AST provides a data-eficient framework for probabilistic full-field reconstruction whose uncertainty remains interpretable under scale-dominated operational distribution shifts. More broadly, this work highlights the need to calibrate uncertainty under changing operational distributions, thereby supporting trustworthy probabilistic modeling and reliability-informed decision-making in civil and infrastructure engineering.

Keywords: Structural virtual sensing; Probabilistic full-field reconstruction; Conditional flow matching; Posterior calibration; Operational distribution shift; Bures–Wasserstein transport

## 1 Introduction

Large-scale engineering structures are instrumented at only a limited number of locations, whereas condition assessment, anomaly localization, and maintenance planning often require distributed displacements, strains, or internal forces. Access constraints, acquisition infrastructure, maintenance demands, and long-term reliability limit the spatial resolution that can be achieved by adding physical sensors. Virtual sensing therefore provides a practical link between sparse measurements and full-field structural response by reconstructing responses at uninstrumented locations [1, 2]. Deployment also exposes reconstruction models to operating conditions that evolve over time. Variations in loading, environmental conditions, and operating states can shift the structural response distribution away from the source domain [3]. Virtual sensing must therefore close the spatial information gap while quantifying how strongly current sparse measurements constrain the full field under changing operating conditions. The resulting response distribution can serve as one reliable source of screening evidence for deciding whether a detailed, computationally intensive reliability analysis is warranted.

Sparse and noisy measurements generally do not uniquely determine a high-dimensional response field. A single reconstruction therefore conceals the multiple full-field states that remain compatible with the current measurements. We cast this task as learning a conditional posterior in a reduced representation. The posterior mean provides a point reconstruction, while its spread and dependence describe the conditional uncertainty left by measurement sparsity and noise. A reliable probabilistic reconstruction should combine low reconstruction error with marginal intervals whose empirical coverage agrees with their nominal levels, thereby avoiding overconfidence from undercoverage and excessively conservative intervals from overcoverage [4, 5].

Two challenges are central to the deployment setting considered here. First, direct conditiona generative methods learn the full conditional state from finite paired sensor and full-field data. Under the retained observation model and known measurement noise, a Gaussian approximation already supplies an analytic posterior center and covariance. A direct conditional generator must nevertheless relearn this available geometry together with the non-Gaussian residual structure from finite paired data [6–8]. Second, posterior uncertainty depends on the source training distribution. When an ofline dataset is deliberately balanced across prescribed regimes, its sampling weights become part of the learned source distribution. Without deployment adaptation, the model continues to use the source prior or conditional relationship induced by that design. The regime frequencies and response scales encountered on site may difer, so the source posterior can produce intervals that are too wide or too narrow [9]. This issue afects most probabilistic deep learning models that claim to provide uncertainty intervals, yet it is often overlooked. Retraining the same supervised posterior model would require new target-condition full fields, whereas the only additional target-domain information considered here consists of historical measurements from the installed sensors. The source model must therefore be retained while its posterior uncertainty is adjusted using those histories.

To address these limitations, we propose Modal Residual Flow Matching with Context-Conditioned Afine Spread Transport, termed MoRF-AST, which integrates source-domain posterior learning with deployment-time uncertainty adjustment. MoRF combines an analytic Gaussian reference with residual learning to construct a base posterior from the current measurement. AST uses historical sensor measurements from the current operating condition to adjust posterior spread while preserving its mean. This separates event-specific reconstruction from operating-condition spread adjustment.

There are three main contributions. First, MoRF learns a conditional residual distribution around an analytic Gaussian posterior. This improves data eficiency and provides shared posterior coordinates under a fixed sensing setup. Second, AST adapts posterior spread using histories from the installed sensors. It combines efective-scale estimation, truncation-aware gating, and mean-preserving Bures transport without target full fields or generator retraining. Third, we identify and verify that spread transport improves calibration only when its contraction or expansion agrees with the base posterior’s dispersion bias. This explains why AST benefits MoRF but not the alternative base models across all shifted domains.

The structure of the paper is summarized as follows. Section 2 reviews probabilistic full-field reconstruction, conditional generative modeling, and deployment adaptation, and Section 3 presents MoRF-AST. Section 4 reports bridge-deck validation and calibration across operating domains, Section 5 examines design choices, ablations, deployment sensitivities, and recurrent operating histories, Section 6 discusses applicability and limitations, and Section 7 concludes the paper.

## 2 Related Works

This section reviews developments in structural virtual sensing and probabilistic full-field reconstruction, discusses generative posterior modeling for inverse problems, and then examines adaptation under operational distribution shift. This review identifies the remaining research gaps and outlines the objectives of the present study.

## 2.1 Structural virtual sensing and probabilistic full-field reconstruction

Structural virtual sensing uses a limited set of measured channels to reconstruct unmeasured displacements, accelerations, strains, and response fields at inaccessible or uninstrumented locations. Model-driven methods establish measurement-to-field relations through modal expansion, dynamic substructuring, reduced-order bases, and regularized inversion [2, 10, 11]. Mechanics-based model updating can jointly infer structural parameters, unknown loads, and unmeasured responses [12]. State-space estimators use Kalman filtering and smoothing to fuse heterogeneous measurements, including asynchronous data acquired at diferent rates [13]. Bayesian response expansion and Gaussian-process latent-force models further quantify predictive uncertainty by propagating uncertainties associated with loads, models, and measurements into conditional means, covariances, or credible intervals [1, 5].

Data-driven methods learn nonlinear maps from monitored channels to unmeasured responses. MLPs and DNNs support sensor selection and response regression, while CNNs exploit temporal and cross-channel dependence for missing-response reconstruction [14, 15]. GANs, VAEs, and conditional difusion models extend these mappings through generative response reconstruction, while latent-variable and difusion formulations can produce stochastic response ensembles beyond a single deterministic estimate [16–18]. Together, these developments broaden data-driven virtual sensing from point reconstruction toward stochastic response ensembles, complementing modeldriven approaches to probabilistic full-field reconstruction. MoRF-AST builds on this progression by coupling conditional full-field posterior reconstruction with a dedicated deployment-time mechanism for adapting posterior spread under changing operating conditions.

## 2.2 Generative posterior modeling for inverse problems

Generative methods model posteriors for imaging, flow-field recovery, subsurface characterization, inverse elasticity, and structural response reconstruction. Given paired observations and targets, VAEs, conditional GANs, conditional normalizing flows, difusion models, and flow-matching models can learn the conditional distribution directly and draw posterior samples for new observations [4, 6, 7, 19]. Specifically, difusion generates samples by progressively removing noise. Flow matching instead learns how samples move from a simple source to the target distribution and generates them by integrating an ODE [20–22]. Their well-defined dynamics and ability to represent highdimensional distributions have made both approaches increasingly prominent in inverse problems [8, 23–26].

Both families can instead combine an unconditional prior with measurements through inferencetime guidance. Difusion guidance modifies the reverse score or drift. DPS estimates a measurementbased correction from a denoised prediction [27]. Flow guidance corrects the ODE velocity or optimizes the initial noise [28]. These mechanisms are related. Because the observation model is introduced after training, the same prior can often serve diferent inverse problems when the required likelihood or physics information is available. Guided sampling has been applied to sparse dynamics, turbulent flows, PDE solutions, and structural response fields [29–34]. Approximate corrections may nevertheless miss the intended conditional posterior [35–37]. MoRF therefore follows direct conditional training but reduces the learning burden by fitting only the conditional residual velocity around an analytic Gaussian reference.

## 2.3 Operational shift and posterior-spread adaptation

Operational adaptation methods difer in the target-domain information they use after deployment. Supervised transfer updates predictors using sparse or periodic labels for selected target responses [38–40]. Unsupervised methods use unlabeled target sensor records to align inputs or representations [41–44]. These studies demonstrate adaptation without target response labels, with objectives centered mainly on point prediction or representation robustness. Full-field target responses would permit direct updating of field predictors or conditional distributions. They are unavailable in the present setting, where histories from the installed sensors are the only additional target-domain information for adaptation.

When reconstruction is probabilistic, adaptation must also address posterior spread and coverage. Existing work corrects amortized latent posteriors from individual out-of-distribution observations [45] or calibrates prediction sets under covariate shift [9, 46]. For Gaussian measures, Bures-Wasserstein geometry provides an established covariance metric and its associated quadratic-cost transport [47, 48]. Related adaptation methods use optimal-transport or Bures objectives to align sample, feature, or class-conditional distributions [49–51]. Unlike supervised full-field retraining or representation alignment, AST uses installed-sensor histories as its only additional target-domain information to adapt scale-dominated posterior spread while preserving the base-posterior mean and source-trained generator.

## 3 Methodology

## 3.1 Engineering Problem and Method Overview

We define probabilistic virtual sensing as the reconstruction of the conditional full-field response distribution $p ( X \mid y )$ from a limited number of noisy sensor measurements. For each event, $X \in \mathbb { R } ^ { P }$ denotes the multichannel physical response field with P field entries, and $y \in \mathbb { R } ^ { p }$ contains the $p$ scalar sensor readings available for reconstruction. To place response channels with diferent units and magnitudes on a common scale, the invertible operator $\mathcal { S } : \mathbb { R } ^ { P }  \mathbb { R } ^ { P }$ divides each field entry by the source-fixed scale of its channel. This gives the dimensionless field $z = S ( X )$ , while $S ^ { - 1 }$ restores physical units. The installed sensors observe selected components of $z ,$

$$
y = H z + \epsilon , \qquad \epsilon \sim { \mathcal { N } } ( 0 , R ) ,\tag{1}
$$

where the binary matrix $H \in \mathbb { R } ^ { p \times P }$ selects the installed sensor channels, $\epsilon \in \mathbb { R } ^ { p }$ is the measurement error, and $R \in \mathbb { R } ^ { p \times p }$ is its positive-definite covariance. Because y contains only sparse information about z, reconstruction is performed in a retained d-dimensional proper orthogonal decomposition (POD) subspace. The encoder $\mathcal { E } : \mathbb { R } ^ { P }  \mathbb { R } ^ { d }$ centers and projects z to the normalized modal coeficient $c = \mathcal { E } ( z )$ , where d is the number of retained modes. The afine decoder $\mathcal { D } : \mathbb { R } ^ { d }  \mathbb { R } ^ { P }$ maps a modal coeficient back to a physical response field, including the inverse scaling $S ^ { - 1 }$ . Given $y ,$ the reconstruction method generates posterior coeficient samples $c ^ { ( n ) }$ and decodes them as $X ^ { ( n ) } = { \mathcal { D } } ( c ^ { ( n ) } )$ , where n indexes the samples. Their collection represents $p ( X \mid y )$ within the retained POD subspace. In deployment, an operating-condition shift can miscalibrate the posterior spread even when the reconstructed field remains accurate. Without full-field reference data, this spread must be calibrated from the historical sensor context C, comprising measurements collected by the same sensors under the current operating condition.

We propose MoRF-AST, a probabilistic virtual sensing framework that combines Modal Residual Flow Matching (MoRF) with Context-Conditioned Afine Spread Transport (AST). As shown in Fig. 1, source full-field data establish the channel scaling, POD representation, Gaussian reference, and decoder (Section 3.2). They also train MoRF (Section 3.3) to learn the conditional residual distribution beyond the Gaussian reference. Given a current measurement y, the Gaussian update and MoRF produce a base posterior in modal space. AST (Section 3.4) uses C to assess whether the response scale has changed. It retains the base posterior when the scale remains consistent with the source domain and otherwise adjusts its spread through a symmetric Bures map. The final modal samples are decoded into physical response fields.

![](images/e942d0bdff69c4c48dcd314749eb023c0754fb49ffa3c1485e63b49055b07e39.jpg)  
Figure 1: Modal residual flow matching with context-conditioned afine spread transport

## 3.2 Gaussian Reference Model

The dimensionless source snapshots $z = S ( X )$ are centered and arranged as columns of a snapshot matrix. Their proper orthogonal decomposition (POD) is obtained by singular value decomposition [52]. Let $\bar { z } \in \mathbb { R } ^ { P }$ be the source mean, let $V \in \mathbb { R } ^ { P \times d }$ contain the retained orthonormal left singular vectors, and let $\boldsymbol { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { d } ) \succ 0$ contain the corresponding modal variances. The symbol $I _ { d }$ denotes the $d \times d$ identity matrix. The normalized modal encoder, scaled POD loading matrix $\Phi \in \mathbb { R } ^ { P \times d }$ , retained field, and physical decoder are

$$
\begin{array} { c c } { { c = \mathcal { E } ( z ) = \Lambda ^ { - 1 / 2 } V ^ { \top } ( z - \bar { z } ) , } } & { { \Phi = V \Lambda ^ { 1 / 2 } , } } \\ { { \widehat { z } ( c ) = \bar { z } + \Phi c , } } & { { \mathcal { D } ( c ) = S ^ { - 1 } ( \widehat { z } ( c ) ) . } } \end{array}\tag{2}
$$

The source coeficients have zero empirical mean and identity covariance under this normalization. The afine decoder reconstructs the retained dimensionless field and then restores physical units through $S ^ { - 1 }$

Applying the installed sensor selection to Eq. (2) gives the retained observation model

$$
A = H \Phi \in \mathbb { R } ^ { p \times d } , \qquad b = H \bar { z } \in \mathbb { R } ^ { p } , \qquad y = b + A c + \epsilon .\tag{3}
$$

Here A is the sensor-to-mode observation matrix and b is the source mean restricted to the installed sensors. Each column of A gives the sensor response of one normalized mode and therefore records how strongly that structural response pattern is observed.

For source event $i , z _ { i }$ and $c _ { i } = \mathcal { E } ( z _ { i } )$ denote the scaled field and modal coeficient, and $y _ { i } ^ { \mathrm { c l e a n } } = H z _ { i }$ is the noise-free sensor reading. Its sensor-space POD truncation residual $e _ { i }$ satisfies

$$
y _ { i } ^ { \mathrm { c l e a n } } = b + A c _ { i } + e _ { i } , \qquad e _ { i } = H [ z _ { i } - \widehat { z } ( c _ { i } ) ] .
$$

The Gaussian reference uses the retained observation model $y = b + A c + \epsilon$ . The truncation residual has no unique coeficient-space representation within the d retained modes. Section 3.4 therefore uses its source covariance only when deciding whether the operating context departs from the source reference.

A linear Gaussian model defines the analytic reference for this underdetermined reconstruction,

$$
c \sim \mathcal { N } ( 0 , I _ { d } ) , \qquad \epsilon \sim \mathcal { N } ( 0 , R ) .\tag{4}
$$

POD normalization gives the empirical second-order structure used by the prior, while the standardnormal joint law completes the Gaussian assumption. Using subscript G for this reference, the conditional law is exactly

$$
p _ { G } ( c \mid y ) = \mathcal { N } ( \mu _ { G } ( y ) , \Sigma _ { G } ) .
$$

Its covariance $\Sigma _ { G } \in \mathbb { R } ^ { d \times d }$ , lower Cholesky factor $L _ { G } \in \mathbb { R } ^ { d \times d }$ , and mean $\mu _ { G } ( y ) \in \mathbb { R } ^ { d }$ are

$$
\begin{array} { r } { \Sigma _ { G } = \left( I _ { d } + { A } ^ { \top } { R } ^ { - 1 } { A } \right) ^ { - 1 } = L _ { G } L _ { G } ^ { \top } , } \\ { \mu _ { G } ( y ) = \Sigma _ { G } { A } ^ { \top } { R } ^ { - 1 } ( y - b ) . \qquad } \end{array}\tag{5}
$$

For a fixed sensor arrangement and noise model, A and R are fixed. The covariance $\Sigma _ { G }$ and its factor $L _ { G }$ are therefore measurement-independent and can be computed once before deployment. Only the linear update $\mu _ { G } ( y )$ changes with the current measurement. This separation avoids a measurement-specific covariance inversion and gives MoRF a common posterior coordinate system across all source and operating events.

The departure from the analytic center is expressed in the whitened residual coordinate $r \in \mathbb { R } ^ { d }$

$$
r = L _ { G } ^ { - 1 } \left[ c - \mu _ { G } ( y ) \right] .\tag{6}
$$

This coordinate measures how far the modal state lies from the analytic estimate, with magnitude and modal correlation normalized by the reference posterior scale. Under the linear Gaussian reference, $r \mid y$ follows $\mathcal { N } ( 0 , I _ { d } )$ exactly. MoRF is trained on the observed source departures from this reference, as described next.

## 3.3 Modal Residual Flow Matching

## 3.3.1 Conditional Flow Matching

Flow matching parameterizes a transport from independent Gaussian draws to residual samples conditioned on the current sensor measurement [21, 22]. Let $\textit { h } \in \mathbb { R } ^ { d }$ denote the conditioning vector defined in $\operatorname { E q . } \ ( 9 )$ . A residual endpoint $r _ { 1 } \in \mathbb { R } ^ { d }$ follows the source conditional distribution $p _ { \mathrm { d a t a } } ( r \mid h )$ , while $r _ { 0 } \in \mathbb { R } ^ { d }$ is an independent standard Gaussian start. The interpolation coordinate $t \in [ 0 , 1 ]$ defines the straight path

$$
\begin{array} { r l } & { r _ { 0 } \sim \mathcal { N } ( 0 , I _ { d } ) , ~ r _ { 1 } \sim p _ { \mathrm { d a t a } } ( r \mid h ) , } \\ & { ~ t \sim \mathcal { U } ( 0 , 1 ) , ~ r _ { t } = ( 1 - t ) r _ { 0 } + t r _ { 1 } . } \end{array}\tag{7}
$$

The velocity of this path is the constant endpoint displacement $r _ { 1 } - r _ { 0 }$ . The neural velocity $v _ { \theta }$ , with trainable parameters $\theta ,$ predicts this displacement from the path location, interpolation coordinate, and condition. Its flow-matching loss is

$$
\mathcal { L } _ { \mathrm { F M } } ( \theta ) = \mathbb { E } _ { \boldsymbol { r } _ { 0 } \sim \mathcal { N } ( \boldsymbol { 0 } , \boldsymbol { I } _ { d } ) , \ t \sim \mathcal { U } ( 0 , 1 ) } \left[ \frac { 1 } { d } \left. \boldsymbol { v } _ { \theta } ( t , \boldsymbol { r } _ { t } , \boldsymbol { h } ) - ( \boldsymbol { r } _ { 1 } - \boldsymbol { r } _ { 0 } ) \right. _ { 2 } ^ { 2 } \right] .\tag{8}
$$

Minimizing Eq. (8) trains $v _ { \theta }$ to approximate the condition-dependent path velocity. At sampling, the resulting ordinary diferential equation

$$
\frac { \mathrm { d } \boldsymbol { r } } { \mathrm { d } t } = v _ { \boldsymbol { \theta } } ( t , \boldsymbol { r } , h )
$$

is integrated from $t = 0$ to $t = 1$ to generate residual samples from the learned transport.

## 3.3.2 Source Training

MoRF conditions the residual flow on the standardized analytic posterior mean

$$
h ( y ) = [ \mu _ { G } ( y ) - \bar { \mu } ] \oslash \sigma \in \mathbb { R } ^ { d } ,\tag{9}
$$

where $\oslash$ denotes componentwise division. The vector $\bar { \mu }$ is the componentwise training mean of $\mu _ { G } .$ and the positive vector σ contains the corresponding standard deviations. Both are computed once from a fixed noise-augmented version of the source training observations. For fixed A and $R ,$ the Gaussian conditional law depends on the measurement only through $\mu _ { G } ( y )$ . Thus $h ( y )$ retains this information while providing source-standardized values to the flow.

Each source event provides a clean sensor response $y _ { i } ^ { \mathrm { c l e a n } }$ and a modal coeficient $c _ { i }$ . During training, fresh measurement noise is drawn for every batch, and the resulting noisy reading determines both the analytic center and the condition. Equation (6) then gives the corresponding residual endpoint, which enters Eq. (8) to update $\theta .$ . This construction aligns training with the noisy measurements available after deployment. MoRF parameterizes $p _ { \theta } ( r \mid h ( y ) )$ to capture the residual bias, modal dependence, and non-Gaussian structure beyond the Gaussian reference. If the reference were exact, the residuals would already follow the standard-normal law and no additional correction would be needed.

## 3.3.3 Posterior Sampling

The current measurement y enters inference with acquisition noise already present. The frozen velocity transports standard Gaussian points according to the learned residual flow. Multiplication by the Cholesky factor then returns these samples to modal coordinates around the analytic center,

$$
c ^ { ( n ) } = \mu _ { G } ( y ) + L _ { G } r ^ { ( n ) } .
$$

Here $r ^ { ( n ) }$ is the terminal residual of trajectory n at $t = 1$ . Numerical integration with $N _ { t }$ steps produces this endpoint. These samples form the MoRF base posterior represented by $p _ { \theta } ( r \mid h ( y ) )$ . The identity branch decodes $X ^ { ( n ) } = { \mathcal { D } } ( c ^ { ( n ) } )$ , while active $\mathrm { A S T }$ modifies coeficient spread before decoding.

## 3.4 Context-Conditioned Afine Spread Transport

AST uses historical measurements collected at the same sensor locations under the current operating condition to determine whether the MoRF posterior spread should be retained, contracted, or expanded. Its context is $\mathcal { C } = \{ y _ { j } \} _ { j = 1 } ^ { N _ { c } }$ , where $N _ { c } \ge 1$ is the number of past sensor measurements and each $y _ { j } \in \mathbb { R } ^ { p }$ contains acquisition noise. From this context, AST estimates an operating scale shared by current measurements, applies an identity gate, and maps the deviations of the base coeficient samples when a scale change is detected. It requires no full-field data from the current operating condition and leaves the posterior mean unchanged.

## 3.4.1 Efective Operating Scale Estimation

AST represents an operating change by scaling the normalized source modal prior while retaining the source mean and POD basis,

$$
c \sim { \mathcal { N } } ( 0 , \alpha I _ { d } ) , \qquad \alpha > 0 .
$$

Under the installed sensor arrangement, the corresponding source signal covariance is

$$
S _ { y } = A A ^ { \top } \in \mathbb { R } ^ { p \times p } ,
$$

and its operating-scale counterpart is $\alpha S _ { y }$ . The scalar α therefore describes an overall change in response amplitude relative to the source reference.

Let $\mathcal { G } \subset \mathbb { R } _ { > 0 }$ be a logarithmic grid of candidate scales, and define the centered context measurement $\delta y _ { j } = y _ { j } - b$ . For a positive-semidefinite sensor-space signal covariance $\Gamma \in \mathbb { R } ^ { p \times p }$ , the mean-fixed Gaussian negative log-likelihood and its scale estimate are

$$
\begin{array} { l } { \displaystyle { J ( \alpha , \Gamma ) = \frac { 1 } { 2 N _ { c } } \sum _ { j = 1 } ^ { N _ { c } } \left[ \delta y _ { j } ^ { \top } ( R + \alpha \Gamma ) ^ { - 1 } \delta y _ { j } + \log \operatorname* { d e t } ( R + \alpha \Gamma ) \right] , } } \\ { \displaystyle { \widehat { \alpha } ( \Gamma ) = \underset { \alpha \in \mathcal { G } } { \mathrm { a r g } \operatorname* { m i n } } J ( \alpha , \Gamma ) . } } \end{array}\tag{10}
$$

The estimate is shared by measurements from the same operating condition. Because the context is centered about the source sensor mean $b ,$

$$
\begin{array} { r } { \mathbb { E } [ \delta y \delta y ^ { \top } ] = \mathrm { C o v } ( y ) + ( \mathbb { E } [ y ] - b ) ( \mathbb { E } [ y ] - b ) ^ { \top } . } \end{array}
$$

The fitted αb thus captures both covariance change and the contribution of an operating mean ofset.   
It is an efective response-amplitude scale rather than a pure covariance ratio.

## 3.4.2 Truncation-Aware Identity Decision

The source POD truncation residuals $e _ { i }$ defined in Section 3.2 quantify the response variance omitted from the retained modes at the sensor locations. From $N _ { \mathrm { c a l } }$ source calibration records, their covariance is estimated once as

$$
\bar { e } = \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c a l } } } e _ { i } , \quad \quad \Sigma _ { \mathrm { t r } } = \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c a l } } } ( e _ { i } - \bar { e } ) ( e _ { i } - \bar { e } ) ^ { \top } .
$$

Here $\Sigma _ { \mathrm { t r } } \in \mathbb { R } ^ { p \times p }$ is the sensor-space covariance of the POD truncation residual, while R is the measurement-error covariance.

AST evaluates the scale estimator in two forms. The gate estimate $\widehat { \alpha } _ { g }$ includes $\Sigma _ { \mathrm { t r } }$ when deciding whether transport is needed, while the transport estimate $\widehat { \alpha } _ { v }$ uses only the covariance represented by the retained modes. With a tolerance $\tau > 1$ , these estimates and the binary gate $g \in \{ 0 , 1 \}$ are

$$
\begin{array} { c c } { { { \widehat \alpha } _ { v } = { \widehat \alpha } ( S _ { y } ) , } } & { { { \widehat \alpha } _ { g } = { \widehat \alpha } ( S _ { y } + \Sigma _ { \mathrm { t r } } ) , } } \\ { { { } } } & { { { } g = { \bf 1 } \left( | \log { \widehat \alpha } _ { g } | > \log \tau \right) . } } \end{array}\tag{11}
$$

The interval $[ \tau ^ { - 1 } , \tau ]$ defines an identity band. Values within the band retain the MoRF base posterior, while values outside it activate transport. Including $\Sigma _ { \mathrm { t r } }$ in the gate prevents omitted source variability from being mistaken for an operating change. It is excluded from the transport because a general sensor-space truncation covariance has no unique representation within the retained coeficient space.

## 3.4.3 Mean-Preserving Afine Spread Transport

When transport is active, $\widehat { \alpha } _ { v }$ defines the scaled-reference posterior covariance

$$
\Sigma _ { v } = \left( \widehat { \alpha } _ { v } ^ { - 1 } I _ { d } + { A ^ { \top } R ^ { - 1 } A } \right) ^ { - 1 } .\tag{12}
$$

The symmetric Bures map [47] from the source reference covariance $\Sigma _ { G }$ to $\Sigma _ { v }$ is

$$
T _ { \mathrm { B } } = \Sigma _ { G } ^ { - 1 / 2 } \left( \Sigma _ { G } ^ { 1 / 2 } \Sigma _ { v } \Sigma _ { G } ^ { 1 / 2 } \right) ^ { 1 / 2 } \Sigma _ { G } ^ { - 1 / 2 } ,\tag{13}
$$

where every symmetric matrix square root is the principal square root. For positive-definite $\Sigma _ { G }$ and $\Sigma _ { v } ,$ this is the unique symmetric positive-definite map satisfying $T _ { \mathrm { B } } \Sigma _ { G } T _ { \mathrm { B } } ^ { \top } = \Sigma _ { v }$ . It is also the quadratic-cost $W _ { 2 } .$ -optimal transport between same-mean Gaussian distributions with these covariances.

Although $\widehat { \alpha } _ { v }$ is a scalar, conditioning makes the correction direction dependent. Writing

$$
\begin{array} { r } { \boldsymbol { A } ^ { \top } \boldsymbol { R } ^ { - 1 } \boldsymbol { A } = \boldsymbol { \Psi } \mathrm { d i a g } ( \rho _ { 1 } , \ldots , \rho _ { d } ) \boldsymbol { \Psi } ^ { \top } , } \end{array}
$$

the Bures variance multiplier along direction i is

$$
t _ { i } ^ { 2 } = \frac { \widehat { \alpha } _ { v } ( 1 + \rho _ { i } ) } { 1 + \widehat { \alpha } _ { v } \rho _ { i } } .
$$

The sign of the variance change is determined by $\widehat { \alpha } _ { v } - 1$ , while its magnitude decreases as the sensor information $\rho _ { i }$ increases. AST therefore acts most strongly along weakly observed directions and is attenuated where the measurements already constrain the response.

For a current measurement, let $N _ { s }$ be the number of MoRF base samples $c ^ { ( n ) }$ . Their empirical mean and the final modal samples are

$$
\begin{array} { c } { { \displaystyle \bar { c } = \frac { 1 } { N _ { s } } \sum _ { n = 1 } ^ { N _ { s } } c ^ { ( n ) } , } } \\ { { \tilde { c } ^ { ( n ) } = \left\{ \begin{array} { l l } { { \displaystyle \bar { c } + T _ { \mathrm { B } } \left( c ^ { ( n ) } - \bar { c } \right) , } } & { { g = 1 , } } \\ { { c ^ { ( n ) } , } } & { { g = 0 . } } \end{array} \right. } } \end{array}\tag{14}
$$

Because the map acts on deviations about ${ \bar { c } } ,$ it preserves the empirical coeficient mean and, through the afine decoder ${ \mathcal { D } } ,$ the posterior mean field. The final samples are $X ^ { ( n ) } = \mathcal { D } ( \tilde { c } ^ { ( n ) } )$

The construction replaces the Gaussian reference covariance while retaining the covariance structure learned by MoRF relative to that reference. Since $\Sigma _ { G }$ and $\Sigma _ { v }$ are both matrix functions of $A ^ { \top } R ^ { - 1 } A$ , they commute. For any empirical base covariance $C ,$

$$
\Sigma _ { v } ^ { - 1 / 2 } T _ { \mathrm { B } } C T _ { \mathrm { B } } ^ { \top } \Sigma _ { v } ^ { - 1 / 2 } = \Sigma _ { G } ^ { - 1 / 2 } C \Sigma _ { G } ^ { - 1 / 2 } .\tag{15}
$$

AST therefore changes the reference spread without discarding the reference-normalized covariance structure of the base posterior. Section 4.3 evaluates how this structure interacts with the transport direction and the base spread bias. The complete procedure is given in Algorithm 1.

Algorithm 1 MoRF-AST probabilistic full-field reconstruction   
Require: Current measurement y and historical context C   
Require: Trained MoRF model, $N _ { s }$ samples, and $N _ { t }$ ODE steps   
1: $\widehat { \alpha } _ { v }  \widehat { \alpha } ( S _ { y } ) , \widehat { \alpha } _ { g }  \widehat { \alpha } ( S _ { y } + \Sigma _ { \mathrm { t r } } )$ ▷ Scale estimation   
2: $r _ { 0 } ^ { ( n ) } \sim \mathcal { N } ( 0 , I _ { d } ) , n = 1 , . . . , N _ { s }$ ▷ Gaussian initialization   
3: $r ^ { ( n ) } \gets \mathrm { O D E S o l v e } ( v _ { \theta } , h ( y ) , r _ { 0 } ^ { ( n ) } , N _ { t } )$ ▷ MoRF sampling   
4: $c ^ { ( n ) } \gets \mu _ { G } ( y ) + L _ { G } r ^ { ( n ) }$ ▷ Base posterior   
5: $\begin{array} { r } { \bar { c } \gets N _ { s } ^ { - 1 } \sum _ { n = 1 } ^ { N _ { s } } c ^ { ( n ) } } \end{array}$ ▷ Ensemble center   
6: if |log $\widehat { \alpha } _ { g } | >$ log τ then ▷ Transport branch   
7: $\Sigma _ { v }  ( \widehat \alpha _ { v } ^ { - 1 } I _ { d } + A ^ { \top } R ^ { - 1 } A ) ^ { - 1 }$ ▷ Target covariance   
8: $T _ { \mathrm { B } } \gets \bar { \Sigma } _ { G } ^ { - 1 / 2 } ( \Sigma _ { G } ^ { 1 / 2 } \Sigma _ { v } \Sigma _ { G } ^ { 1 / 2 } ) ^ { 1 / 2 } \Sigma _ { G } ^ { - 1 / 2 }$ ▷ Bures map   
9: $\widetilde { c } ^ { ( n ) } \gets \bar { c } + T _ { \mathrm { B } } ( \bar { c } ^ { ( n ) } - \bar { c } )$ ▷ Active transport   
10: else ▷ Identity branch   
11: $\widetilde { c } ^ { ( n ) } \gets c ^ { ( n ) }$   
12: end if   
Ensure: $\{ X ^ { ( n ) } = { \mathcal { D } } ( { \widetilde { c } } ^ { ( n ) } ) \} _ { n = 1 } ^ { N _ { s } }$ ▷ Field decoding

## 4 Numerical Validation

## 4.1 Numerical Benchmark

## 4.1.1 Bridge Deck and Structural Model

We consider probabilistic full-field reconstruction of a bridge deck from sparse multichannel observations. Figure 2 shows the 32 m × 16 m, four-lane deck with five distributed longitudinal girders. On a 128 × 128 grid, its deflection satisfies [53]

$$
\left[ D _ { x } + \kappa _ { g } ( y ) \right] w _ { x x x x } + 2 \left( D _ { 1 } + 2 D _ { k } \right) w _ { x x y y } + D _ { y } w _ { y y y y } = q ( x , y ) .\tag{16}
$$

Here x and y denote the longitudinal and transverse coordinates, $w$ is the deflection, and $q$ is the wheel-load pressure. The coeficients $D _ { x }$ and $D _ { y }$ are the flexural rigidities, $D _ { 1 }$ and $D _ { k }$ are the coupling and twisting rigidities, and $\kappa _ { g }$ represents the girder contribution. Subscripts denote partial derivatives. The deck has simply supported ends and free side edges. Trafic realizations are generated by a seeded four-lane microscopic simulator with regime-specific distributions for lane flow, heavy-vehicle composition, speed, headway, and lateral position. Vehicle weights are allocated to axles and wheel pairs, and the on-deck wheel forces are rasterized as normalized Gaussian pressure patches to form $q ( x , y )$ , with each pressure field defining one event.

![](images/1d489bf501460b7adba4b9a0c88b243e5af2b0071575c48d7e49aca6b2844c18.jpg)  
Figure 2: Bridge benchmark for probabilistic full-field reconstruction  
Each event comprises seven response fields in the fixed order

$$
\left( w , \kappa _ { x } , \kappa _ { y } , \kappa _ { x y } , M _ { x } , M _ { y } , M _ { x y } \right) .\tag{17}
$$

The model reconstructs all seven fields, but the quantitative analysis focuses on $M _ { x } , M _ { y } ,$ and $M _ { x y } .$ The deflection field is comparatively easy to reconstruct, and its posterior intervals are already narrow, leaving little scope for AST to improve the spread. The moment fields provide a more demanding test of full-field uncertainty and can be used with the section properties to recover stresses.

Each channel is normalized as $z _ { j } = X _ { j } / s _ { j } ,$ , where $s _ { j }$ is the source-training standard deviation pooled over the 40 master locations. The fixed $d = 1 9 2$ source POD captures 99.9% of the variance and is used across all domains.

An $8 \times 6$ interior candidate grid is ordered by farthest-point sampling from the deck centre. The first 40 positions form the fixed master sensor layout. Each design with fewer sensors uses the first k positions in this ordering. The 40-location layout provides space-filling coverage while preserving the intended underdetermination.

The operational domains in Table 1 isolate changes in trafic loading statistics. The bridge model, boundary conditions, response normalization, POD representation, and observation design remain fixed. The source data S and E0 are independent draws from the same weighted mixture of nine trafic regimes. The models therefore learn from a broad source prior, while E0 provides an independent control over the full source envelope. N0, N1, and V1 to V6 draw from individual regime components to represent operating periods dominated by a particular trafic pattern.

Table 1: Operational domains and trafic regimes
<table><tr><td></td><td>Domain Dominant regime</td><td>Operating meaning</td></tr><tr><td>E0</td><td>Source mixture</td><td>Independent source-envelope control</td></tr><tr><td>NO</td><td>Restricted load</td><td>Heavy vehicles largely excluded</td></tr><tr><td>N1</td><td>Night</td><td>Low total flow with a non-negligible heavy-vehicle share</td></tr><tr><td>V1</td><td>Commuter peak</td><td>Dense, slow commuter traffic</td></tr><tr><td>V2</td><td>Platoon</td><td>Clustered convoy arrivals</td></tr><tr><td>V3</td><td>Work zone</td><td>Concentrated lane use with an elevated heavy-vehicle share</td></tr><tr><td>V4</td><td>Freight</td><td>High heavy-vehicle share</td></tr><tr><td>V5</td><td>Overload</td><td>Traffic concentrated toward the heaviest vehicle classes</td></tr><tr><td>V6</td><td>Congested</td><td>Slow, high-volume freight traffic</td></tr></table>

The source data comprise 8,000 training and 1,000 validation events. Each evaluation domain contains 1,500 sensor-only context events and 800 test events. Context and test identifiers are disjoint at the simulation-run level. Context fields are not retained. Test full-field responses are reserved for ofline scoring, held-out diagnostics, and the full-field visualization, and never enter model fitting or AST.

## 4.2 Experimental Design

## 4.2.1 Baselines

We consider three baselines: the analytic Gaussian reference, a Deep Ensemble [54], and Gaussian mixture residual regression (GMR) [55]. The Gaussian reference in Section 3.2 has no fitted network and tests whether learning non-Gaussian residuals improves on the analytic posterior. The Deep Ensemble is a standard supervised uncertainty baseline. Its ten independently trained members take the standardized Gaussian posterior mean as input, predict diagonal Gaussian residual distributions, and contribute equally weighted samples. GMR is a conventional conditional-density baseline. It conditions on the standardized query observation residual and fits a Gaussian mixture to selected high-variance residual coordinates, testing whether a mixture model can reproduce the residual distribution without a flow. This condition is separate from the historical context used by AST. MoRF and MoRF-AST are evaluated against these baselines.

## 4.2.2 Experimental Setup

The primary domain comparison changes only the trafic regime and uses the fixed 40-location layout. All methods share the source scaling, POD basis, observation operator, measurement covariance, decoder, evaluation events, and score implementation. Methods expressed in residual coordinates also share the Gaussian reference. Query observations use common noise realizations across methods. Reduced sensor designs use nested prefixes of the same master ordering, so sensor addition is the only change between consecutive layouts.

Target context observations are standardized using the source scales and kept separate from the test events. The source calibration set used to estimate $\Sigma _ { \mathrm { t r } }$ combines the source training and validation fields. The AST identity gate uses $\tau = 1 . 2 5$ in Eq. (11).

The directional coverage analysis pools each coverage cell over the 800 held-out events and 16,344 unobserved grid points. Confidence intervals are estimated from 2,000 paired bootstrap replicates that resample events as clusters and retain all fields, nominal levels, and spatial locations within each event. This construction preserves spatial dependence and pairs the base and adapted scores within every replicate.

MoRF uses fixed-step Heun integration with 100 steps. All methods use 500 posterior samples per observation.

## 4.2.3 Metrics

We evaluate posterior accuracy and calibration using the following three metrics. Scores are computed directly in physical coordinates for $M _ { x } , M _ { y }$ , and $M _ { x y } .$ . Let X be the true field, $\widehat { X }$ and ${ \widehat { X } } ^ { \prime }$ be independent posterior draws, $\overline { { \boldsymbol { X } } } = \mathbb { E } [ \widehat { \boldsymbol { X } } ]$ , and $X _ { \mathrm { r m s } }$ be the root-mean-square magnitude of the true field. Angular brackets denote averaging over test events and unobserved grid points. For a nominal level $a \in \mathcal { A } = \{ 0 . 1 , 0 . 2 , . . . , 0 . 9 \}$ , the empirical coverage $\widehat { C } _ { a }$ is the fraction of true values

(c) Scale estimates and gate decisions

contained in the corresponding central posterior interval. The three metrics are

$$
\mathrm { N R M S E } = \frac { \left. ( \overline { { X } } - X ) ^ { 2 } \right. ^ { 1 / 2 } } { X _ { \mathrm { r m s } } } ,\tag{18}
$$

$$
\mathrm { n C R P S } = \frac { 1 } { X _ { \mathrm { r m s } } } \left. \mathbb { E } | \widehat { X } - X | - \frac { 1 } { 2 } \mathbb { E } | \widehat { X } - \widehat { X } ^ { \prime } | \right. ,\tag{19}
$$

$$
\mathrm { A C E } = \frac { 1 } { 9 } \sum _ { a \in \mathcal { A } } | \widehat { C } _ { a } - a | .\tag{20}
$$

ACE averages the absolute diference between empirical and nominal coverage over the nine levels. Lower ACE indicates better marginal calibration, with zero denoting exact agreement. To distinguish undercoverage from overcoverage, the directional analysis also uses the signed coverage error

$$
\mathrm { S C E } _ { m , d } = \frac { 1 } { 3 | \mathcal { A } | } \sum _ { j \in \{ M _ { x } , M _ { y } , M _ { x y } \} } \sum _ { a \in \mathcal { A } } \left( \widehat { C } _ { m , d , j , a } - a \right) ,\tag{21}
$$

where $\widehat { C } _ { m , d , j , a }$ is the empirical central-interval coverage for model $m _ { \colon }$ domain $d ,$ moment field j, and nominal level a. Negative SCE denotes undercoverage, while positive SCE denotes overcoverage.

The CRPS expectation is evaluated with the fair finite-ensemble correction [56]. Each primary metric is first computed for each of the three moment fields and then averaged across fields. NRMSE and nCRPS are reported with percentage signs, whereas ACE is reported as a fraction. Empirical scores are reported to three significant figures throughout. E0 is reported separately, while averages over N0, N1, and V1 to V6 assign equal weight to each of the eight shifted domains.

## 4.3 Calibration Performance across Operational Domains

We compare MoRF, the Deep Ensemble, and GMR before and after AST across the operational domains using ACE and nCRPS, and record the corresponding scale estimates and gate decisions. The results are summarized in Fig. 3. The identity gate retains the base posterior in E0, V1, and V2, while Bures transport is activated in N0, N1, and V3 to V6. E0 therefore remains an independent source-envelope control and is reported separately from the shifted-domain average.

![](images/90bf7b56ff033dbf42c031b901bbe2209455ee33e731a10ef3420fa170c971c1.jpg)

![](images/7f47c669d474d0b018fee244ca117032088dda3912c258b603810a82dce51a8f.jpg)

![](images/6499a2a07d9c3b0d15cad5d8f255ee96a033c69c7517d8a03a488a4cef366bd1.jpg)  
Figure 3: Cross-domain posterior calibration and AST routing

Across the eight shifted domains, MoRF-AST reduces the mean ACE from 0.0535 to 0.0236, a relative reduction of 55.9%, and yields the lowest aggregate calibration error in the comparison. The corresponding mean NRMSE remains 7.72% by construction, while nCRPS changes only from 2.66% to 2.64%. All six activated domains improve, while the identity-routed domains retain their base values. The primary gain is better marginal coverage, while overall probabilistic accuracy changes only slightly.

The same Bures map does not improve the shifted-domain aggregate ACE of the Deep Ensemble or GMR. The Deep Ensemble mean ACE increases from 0.135 to 0.154, while the GMR value increases from 0.0841 to 0.102. Both alternatives benefit in some domains, but their losses elsewhere dominate the aggregate response. The estimated operating scale is therefore not suficient to determine whether transport will improve calibration. Its efect also depends on the spread bias already present in the base posterior.

Figure 4 resolves this dependence using the signed coverage error in Eq. (21). N0 and N1 require reference contraction, whereas V3 to V6 require reference expansion.

![](images/72273249ee280170db4c3c16bbe376d304bd7979a79097f39dcdf8f6398250b3.jpg)  
Figure 4: Signed coverage under active Bures transport

In the contraction domains, every base model overcovers and AST moves each signed error toward zero. In the expansion domains, MoRF undercovers and AST again moves the signed error toward zero in all four cases. The Deep Ensemble and GMR already overcover in these domains, so the same expansion moves them farther from nominal coverage. In every activated domain, the estimated paired-bootstrap probability that AST lowers ACE is at least 0.957 for MoRF. For the Deep Ensemble and GMR, the corresponding estimate is 0 in every expansion domain. Contraction is beneficial when the base posterior is too broad, while expansion is beneficial only when it acts on an underdispersed posterior.

This directional compatibility explains the contrasting aggregate results. The Bures map determines contraction or expansion from $\widehat { \alpha } _ { v }$ and the sensor geometry, independently of the signed coverage bias of the base model, and carries forward its reference-normalized covariance geometry. MoRF is the only tested model whose base bias is aligned with the required action in both transport regimes. This pattern is consistent with the full-dimensional conditional residual model capturing the principal normalized posterior structure and leaving an operating-scale spread mismatch for

AST to correct. The two alternative models enter the expansion domains with an existing positive coverage bias, which the reference expansion amplifies.

## 4.4 Full-Field Posterior Visualization

To examine the domain-level calibration result in physical space, we select the V5 test event with the largest absolute $M _ { x y }$ response and compare the truth, posterior means, reconstruction errors, peak-location marginals, and posterior standard-deviation fields. The resulting full-field visualization is shown in Fig. 5.

![](images/0bec242a7e13d7e60563d7abaf50deaf1cb649c18a7b10330de015b6841d0e63.jpg)  
Figure 5: Full-field reconstruction and posterior spread under overload trafic

MoRF and MoRF-AST give identical mean and error fields. Both resolve the sign and location of the principal high-response structure. At the peak, the unadapted MoRF 90% interval ends below the true response. AST expands the distribution about the same posterior center, bringing the truth inside the corresponding interval. The Deep Ensemble distribution is broader than the MoRF-AST distribution, although its center remains below the truth. The GMR distribution is narrower than MoRF-AST, and its center also remains below the truth. Interval width alone is therefore insuficient. The center and spread must be compatible with the event response. The MoRF standard-deviation field concentrates its largest local variation around the high-response region. AST increases the uncertainty scale while retaining this spatial organization. The event visualizes the cross-domain coverage correction in physical space and complements the aggregate test-set evidence.

## 5 Ablation, Sensitivity, and Operational Studies

## 5.1 Design Comparisons and Ablation Studies

## 5.1.1 Efect of Residual Formulation

To determine whether MoRF’s data eficiency arises from Gaussian conditioning alone or from learning a residual around the analytic reference, we compare it with two direct conditional flow models. Direct-y learns modal coeficients c from the sparse observation, while direct-µ learns c from the standardized Gaussian posterior mean. MoRF uses the same condition as the parameter-matched direct-µ model but learns the posterior-whitened residual r. All three models are trained on the same nested subsets of the 8,000 source events and evaluated on held-out E0 events.

The comparison across flow-training set sizes in Fig. 6 separates the contribution of the analytic reference from that of the residual flow. With 250 flow-training events, MoRF obtains 9.35% posterior-mean NRMSE, nearly identical to the 9.34% Gaussian reference, while both direct flows remain near 90.5%. The analytic update therefore anchors MoRF when full-field training data are scarce. At 8,000 events, MoRF reaches 7.20%, compared with 16.1% for direct-y and 17.9% for the parameter-matched direct-µ. The widening performance gap from the fixed reference, rather than a gap from the direct flows, identifies the additional structure learned by the residual model.

![](images/8566914ca12ca1d27e13016ac117c786dc78137d682db3a9f1b67b0edfd8a22a.jpg)

![](images/561b5fcf5e578dec5e27e3f42e1f62d45375ff179c2e421869e0300084bdbdd6.jpg)  
Figure 6: Training-data and integration-step sensitivity on the no-shift control

The integration step comparison used the full training set, fixed checkpoints, and common base samples. Increasing the integration budget from one to four Heun steps removes most of the integration-induced calibration error, while further refinement to 100 steps changes the result only slightly. Four steps therefore approach the 100-step result with one twenty-fifth as many integration steps. Together, these results show that the analytic reference makes MoRF data-eficient, while the residual flow reaches near-converged calibration with four Heun steps.

## 5.1.2 Guided Sampling Comparison

To assess the value of direct conditional posterior learning, we compare MoRF with an unconditional modal flow that incorporates the observation through ΠGDM-style likelihood guidance during sampling [57]. The two guidance cases use linearly and quadratically decaying guidance variances with the same 100-step integration budget. We evaluate source validation, where AST returns identity, and V5, where the complete MoRF-AST result includes active spread transport. The results are shown in Fig. 7.

![](images/116da40856f177b2c1f5699ad1f180768c922bbb074a622a620fee327aa94df0.jpg)

![](images/d44363d43530dff8ac50a49fe0da6577675aedb24b85ddfaff24fa7f35796c76.jpg)  
Figure 7: Direct posterior modeling and likelihood guidance across source validation and V5

On source validation, direct conditional MoRF gives the best mean reconstruction and calibration among the displayed flow constructions. In V5, the same conditional model retains its reconstruction advantage, while active AST restores calibration under the operating shift. These results support direct conditional MoRF with AST over likelihood-guided unconditional sampling under the same integration budget.

## 5.1.3 Ablation of Posterior Center and Spread Map

To isolate the efects of posterior centering and covariance transport, we apply ungated maps to the same MoRF base samples across all nine domains, including E0 (Fig. 8). Every spread-only map leaves NRMSE unchanged within numerical precision. Replacing that center with the target-scale Gaussian mean increases the nine-domain mean NRMSE from 7.65% to 10.1% and the mean ACE from 0.0495 to 0.171. Mean preservation is therefore necessary to retain the event-specific reconstruction learned by MoRF.

![](images/ccbc1ea7f0d2f8ca6f0ee5d6b4ef33aa47e6ab6c062bad007ff7cef7ca880a8b.jpg)  
Figure 8: Posterior-center and covariance-map ablations across operational domains

The symmetric Bures map attains a nine-domain mean ACE of 0.0225, compared with 0.0270 for the Cholesky map. It is positive definite, maps the Gaussian reference covariance exactly to its scaled counterpart, and minimizes quadratic displacement between the same-center Gaussian references.

Together, these ablations support mean-preserving Bures transport as the AST map and show that direct high-dimensional covariance estimation does not improve calibration with the available history.

## 5.1.4 Efect of Truncation Correction on Identity Gating

To test whether unresolved POD variance causes false transport, we compare the ordinary and truncation-aware scale estimates on repeated E0 subsets (Fig. 9). The truncation-aware estimate activates less often, while both estimates return identity for the full context. Accounting for unresolved sensor-space energy therefore protects the frozen posterior from unnecessary adaptation.

![](images/a8c37285b2a7d7d8b8ef93d78b9b0bceb1f4a60763804d96b5056b2c3aeff693.jpg)

![](images/0430da9d5a1f49518eb6e20cc2f86acfcbe0ad0dfd9ff1f762d6a5f334dc7103.jpg)

![](images/644938a77ccb94a2cd284773e250fa62637de91e0b5408b728df5b8f46ba6005.jpg)  
Figure 9: Truncation-aware identity gating under the no-shift control

Carrying the same truncation correction into the transported covariance raises aggregate ACE in most domains. The final design therefore assigns $\widehat { \alpha } _ { g }$ to the gate and $\widehat { \alpha } _ { v }$ to the retained-space spread map. This separation preserves a conservative identity decision with moderate histories without injecting unresolved POD energy into transport.

## 5.2 Sensitivity to Deployment Information and Measurement Assumptions

## 5.2.1 Sensitivity to Historical Context Size

To determine the history needed for stable scale estimation, we evaluate context budgets from 25 to 800 events using six independently assembled run clusters at each partial budget (Fig. 10). The full set of 1,500 events is evaluated once. Budgets are assembled from complete simulated trafic runs, so response snapshots within a run are not treated as independent operational realizations.

Even the smallest context budget improves the shifted-domain calibration relative to unadapted MoRF, although its repetition-to-repetition variability is appreciable. The eight-domain mean ACE is 0.0262 at the nominal 25-event budget, 0.0239 at 400 events, and 0.0236 with the full 1,500-event context. The result therefore stabilizes by the nominal 400-event budget, corresponding to a median of approximately 36 independent trafic runs and providing a practical operating point for scale estimation.

![](images/03e1b06c79c3723ec59f17b58a5a6e82bffe23e974cea4371dbcb244fabf6c5a.jpg)  
Figure 10: Historical-context sensitivity of MoRF-AST

## 5.2.2 Sensitivity to Sensor Coverage

To assess how sensor coverage afects reconstruction and calibration, we evaluate nested layouts with $k = 1 2 , 1 6 , 2 0 .$ , and 40 locations from the same farthest point ordering (Fig. 11). Every location supplies four channels, so the number of scalar observations is 4k for $d = 1 9 2$ retained coeficients. Source channel scales, the POD basis, the master noise realization, and a common scoring mask that excludes all 40 master sensor positions were fixed. Each sensor count used its own observation operator, Gaussian reference, and trained flow checkpoint. The nested layouts therefore change sensing information while retaining a common spatial design and evaluation region.

![](images/8b0606a4a1c4471f68d0009aca44506ba43c9f3f27495678e5ca543916f5f0e1.jpg)

![](images/d3d9b8e08e0dccf0236f630764af8af6509522f7e4d7061e380c9b3af2fe0bfc.jpg)  
Figure 11: Sensor-count sensitivity of MoRF and MoRF-AST

Increasing sensor coverage steadily lowers nCRPS because the query constrains more of the retained state. From $k = 1 2 \ \mathrm { t o } \ k = 4 0$ , the MoRF nCRPS decreases from 4.08% to 2.66%, while the MoRF-AST value decreases from 4.05% to 2.64%. At k = 12, AST reduces ACE from 0.0519 to 0.0298. At k = 40, it reduces ACE from 0.0535 to 0.0236, and the same ordering holds at $k = 1 6$ and $k = 2 0$ . Base MoRF calibration does not improve monotonically with sensor count. Additional query information alone therefore does not remove the operating-scale spread bias. Using common base samples within each layout confirms that AST improves calibration across all tested sensor counts through spread transport rather than posterior sampling variation.

## 5.2.3 Sensitivity to Observation Noise Misspecification

To assess robustness to noise misspecification, we hold the assumed covariance $R = 0 . 0 5 ^ { 2 } I$ fixed while scaling the true noise applied to both query and context observations (Fig. 12). The same clean events and seeded independent Gaussian draws are used for every true to assumed noise standard deviation ratio from 0.5 to 1.5. This construction isolates the imposed noise magnitude from event and noise realization changes.

![](images/997d70376359d40c91ed9fd8767aa5aa0def6b95daa95afa375ed9d48191c874.jpg)

![](images/24e89896463b0b0133d8bce5830939b14d91d20367a500a190a376d0f8e7d9d4.jpg)  
Figure 12: Observation-noise sensitivity of MoRF and MoRF-AST

Increasing the true noise raises nCRPS because the measurements carry less event-specific information. Calibration is best when the assumed and true noise levels agree and degrades when the model either overstates or understates measurement uncertainty. MoRF-AST nevertheless retains lower nCRPS and ACE than MoRF at every tested ratio. The adaptation advantage remains robust over true-to-assumed noise ratios of 0.5–1.5, while accurate noise characterization remains important for the scale fit.

## 5.3 Evaluation of Recurrent-State Context Aggregation

To support the practical use of AST, we make a preliminary comparison of several ways to select historical observations that represent the current operating condition. We construct 320-minute histories in which N1 and V6 alternate (Fig. 13). For each evaluation state, $n _ { \mathrm { o c c } } = 1 , 2 , 4 , 8 , 1 6$ , or 32 past occurrences correspond to individual dwell times from 160 to 5 minutes. Moving right in the figure therefore represents more frequent switching. Four seeded histories are evaluated at each setting. Simulator state labels select the final stable evaluation point and construct a privileged reference, but they are not supplied to the observation-based history models.

![](images/9cdd2449feefea9e942d4d1ea292218c1b61b40216de2dba9bf22f973ca42c5f.jpg)  
Figure 13: Context aggregation under recurrent operating states

Four ways of assembling history are compared with the same AST scale objective. Pooled history gives uniform weight to every available observation before the query. Recency weighting favors recent observations. The observation-only hidden semi-Markov model (HSMM) [58] uses 60-second bins of sensor root-mean-square response and availability, then weights each past frame by its posterior probability of sharing the anonymous query state. This weighting can recover noncontiguous but state-compatible periods. If the inferred segments are too short, it returns to uniform pooling. The state-labeled case uses known simulator states and serves only as a privileged reference. No strategy accesses observations after the query time.

For both recurrent states, HSMM weighting approaches the calibration obtained from a full state-specific context while remaining well below base MoRF. It improves on pooled history in every tested state-frequency combination and on recency weighting in nearly every combination. The only reversal is negligible relative to the variation across histories. These trends show that matching the current state is more important than treating all past observations equally.

The state-dependent response of pooled history shows that indiscriminate aggregation mixes relevant and irrelevant observations. Observation-based state weighting instead reuses noncontiguous periods that match the current operating state, which preserves a more coherent scale estimate as switching becomes frequent. The tested histories support this mechanism in the recurrent night and congested setting, while broader state structures require separate evaluation.

Overall, adaptation is governed primarily by the relevance of operating history: moderate contexts stabilize the scale estimate, denser sensing improves accuracy but does not remove spread bias, and observation-based state weighting preserves calibration when conditions recur.

## 6 Discussion

Across the numerical operating domains, changes in trafic composition and operating conditions were neither model inputs nor identified as individual parameters from sparse observations. Their efects propagated through the loading process and structural mapping and appeared in the second moment of sensor responses about the fixed source-domain mean b, $\mathbb { E } [ ( y - b ) ( y - b ) ^ { \mathsf { T } } ]$ . AST uses the inferred scale component to adjust posterior spread, capturing an overall response-scale shift rather than specific trafic parameters. Similar signatures may arise from wind loads on tall buildings, wave loads on ofshore platforms, or changes in rotating-machinery speed and load, suggesting applications of AST beyond trafic-induced response reconstruction.

With limited history, fitting a complex target-domain distribution is not necessarily preferable to estimating a scalar. For every layout with $p < d ,$ distinct modal covariances can yield the same sensor covariance, preventing identification of a general high-dimensional covariance. AST instead estimates an efective scale α, reducing unsupported degrees of freedom without claiming uniqueness or optimality. The Gaussian conditional covariance and Bures map combine this scalar with sensor geometry to produce direction-dependent modal adjustments, a construction supported by the covariance-map ablation in Fig. 8. AST thus transfers the observable scale change to the full-field posterior through the existing conditional structure.

AST modifies posterior spread without correcting an erroneous posterior center. The base model must retain the main conditional response structure, and the imposed expansion or contraction must match its dispersion bias, as shown in Fig. 4. AST is therefore not a general calibrator, and its role also diminishes as observations become more informative. As sensor information in every retained direction grows without bound, the posterior concentrates and the Bures map approaches the identity, provided that $\widehat { \alpha } _ { v }$ remains finite and the standardized residual second moment remains bounded. Probabilistic virtual sensing then approaches point estimation within the retained subspace, leaving little spread for AST to adjust. Conversely, when observations are insuficiently informative and the reconstruction must rely on the prior, AST can provide efective corrections along those directions.

MoRF-AST is best suited when the source representation and observation relation remain valid, the operating condition remains within source-domain coverage, and $R + \alpha S _ { y }$ adequately represents the mean-referenced operating sensor second moment. Because this moment also reflects operatingmean shifts and noise misspecification, αb is only an efective response scale. New response modes, structural changes, pronounced anisotropic shifts, or sensor faults may exceed both AST and the base model. They may also invalidate fixed-source assumptions shared by other data-driven models, making them broader out-of-distribution challenges. Although MoRF-AST is better calibrated than the evaluated baselines for marginal prediction intervals, this does not establish accurate spatially joint extremes, exceedance probabilities, or tail risk. Accordingly, the AST-adjusted posterior may trigger dedicated tail analysis rather than replace high-fidelity evaluation of extreme responses and tail risk.

## 7 Conclusion

This study proposes MoRF-AST for probabilistic full-field reconstruction from sparse and noisy measurements under changing operating conditions. Evaluation covers a bridge-deck benchmark across shifted trafic domains, with the main conclusions summarized below.

(1) MoRF combines an analytic Gaussian reference with flow matching in posterior-whitened residual coordinates. The reference focuses learning on residual structure beyond the Gaussian approximation. With 8,000 training events, MoRF achieves an NRMSE of 7.20%, less than half the corresponding NRMSEs of the two direct conditional flows.

(2) AST estimates an efective response scale from sensor-only operating history. It applies a truncation-aware identity decision and performs mean-preserving Bures transport when required. Across eight shifted domains, it reduces MoRF’s mean ACE from 0.0535 to 0.0236, and the identity gate correctly avoids unnecessary adaptation. Further ablation and sensitivity studies support the MoRF-AST design choices across the evaluated deployment settings.

(3) Transport improves calibration only when its direction matches the base posterior’s spread bias. MoRF’s bias aligns with the required action in both the contraction and expansion regimes, whereas the biases of the other baselines do not. MoRF thus gives AST a calibrated base posterior with correctable spread bias under domain shifts.

Overall, this study ofers design insights for difusion- and flow-based generative methods in engineering probabilistic virtual sensing. It also addresses posterior calibration under changing operating conditions, a significant problem that has received limited attention. By enabling observation-only calibration during deployment, MoRF-AST supports long-term probabilistic fullfield reconstruction and safer operation under changing conditions.

## CRediT authorship contribution statement

Wingho Feng: Data curation, Formal analysis, Methodology, Validation, Writing–original draft, Writing–review & editing, Visualization. Quanwang Li: Supervision, Project administration, Resources. Ming Zhong: Project administration, Resources. Jingyu Yang: Project administration, Resources. Chen Wang: Conceptualization, Methodology, Writing–review & editing.

## Funding

This work was supported by the National Key R&D Program of China (2025YFF0518604) and the Beijing Nova Program (202604841290).

## Data availability

The numerical benchmark dataset and source code are available at https://github.com/winghof eng/MoRF-AST.

## References

[1] Jyrki Kullaa. Bayesian virtual sensing in structural dynamics. Mechanical Systems and Signal Processing, 115:497–513, 2019. doi: 10.1016/j.ymssp.2018.06.010.

[2] Jixing Cao, Fanfu Bu, Jianze Wang, Chao Bao, Weiwei Chen, and Kaoshan Dai. Reconstruction of full-field dynamic responses for large-scale structures using optimal sensor placement. Journal of Sound and Vibration, 554:117693, 2023. doi: 10.1016/j.jsv.2023.117693.

[3] Hoon Sohn. Efects of environmental and operational variability on structural health monitoring. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 365(1851):539–560, 2007. doi: 10.1098/rsta.2006.1935.

[4] Kristian Gundersen, Anna Oleynik, Nello Blaser, and Guttorm Alendal. Semi-conditional variational auto-encoder for flow reconstruction and uncertainty quantification from limited observations. Physics of Fluids, 33(1):017119, 2021. doi: 10.1063/5.0025779.

[5] Joanna Zou, Eliz-Mari Lourens, and Alice Cicirello. Virtual sensing of subsoil strain response in monopile-based ofshore wind turbines via gaussian process latent force models. Mechanical Systems and Signal Processing, 200:110488, 2023. doi: 10.1016/j.ymssp.2023.110488.

[6] Govinda Anantha Padmanabha and Nicholas Zabaras. Solving inverse problems using conditional invertible neural networks. Journal of Computational Physics, 433:110194, 2021. doi: 10.1016/j.jcp.2021.110194.

[7] Sharmila Karumuri and Ilias Bilionis. Learning to solve bayesian inverse problems: An amortized variational inference approach using gaussian and flow guides. Journal of Computational Physics, 511:113117, 2024. doi: 10.1016/j.jcp.2024.113117.

[8] Agnimitra Dasgupta, Harisankar Ramaswamy, Javier Murgoitio-Esandi, Ken Y. Foo, Runze Li, Qifa Zhou, Brendan F. Kennedy, and Assad A. Oberai. Conditional score-based difusion models for solving inverse elasticity problems. Computer Methods in Applied Mechanics and Engineering, 433:117425, 2025. doi: 10.1016/j.cma.2024.117425.

[9] Hongxiang Qiu, Edgar Dobriban, and Eric Tchetgen Tchetgen. Prediction sets adaptive to unknown covariate shift. Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(5):1680–1705, 2023. doi: 10.1093/jrsssb/qkad069.

[10] Jyrki Kullaa. Virtual sensing of structural vibrations using dynamic substructuring. Mechanical Systems and Signal Processing, 79:203–224, 2016. doi: 10.1016/j.ymssp.2016.02.045.

[11] Haoyang Zhao, Chen Wang, Jiansheng Fan, and Ran Ding. Physics-aligned virtual sensing with AI-enhanced reduced-order modeling for structural digital twins. Advances in Engineering Software, 219:104217, 2026. doi: 10.1016/j.advengsoft.2026.104217.

[12] Mansureh-Sadat Nabiyan, Faramarz Khoshnoudian, Babak Moaveni, and Hamed Ebrahimian. Mechanics-based model updating for identification and virtual sensing of an ofshore wind turbine using sparse measurements. Structural Control and Health Monitoring, 28(2):e2647, 2021. doi: 10.1002/stc.2647.

[13] Zimo Zhu and Songye Zhu. Asynchronous Kalman filtering for dynamic response reconstruction by fusing multi-type sensor data with arbitrary sampling frequencies. Mechanical Systems and Signal Processing, 215:111395, 2024. doi: 10.1016/j.ymssp.2024.111395.

[14] Yuxin Pan, Carlos E. Ventura, and Teng Li. Sensor placement and seismic response reconstruction for structural health monitoring using a deep neural network. Bulletin of Earthquake Engineering, 20(9):4513–4532, 2022. doi: 10.1007/s10518-021-01266-y.

[15] Gao Fan, Jun Li, and Hong Hao. Dynamic response reconstruction for structural health monitoring using densely connected convolutional networks. Structural Health Monitoring, 20 (4):1373–1391, 2021. doi: 10.1177/1475921720916881.

[16] Gao Fan, Zhengyan He, and Jun Li. Structural dynamic response reconstruction using selfattention enhanced generative adversarial networks. Engineering Structures, 276:115334, 2023. doi: 10.1016/j.engstruct.2022.115334.

[17] Kah Hong Lee, Norhisham Bakhary, Khairul H. Padil, Jun Li, Francesca Brighenti, Davide Trapani, and Yon Kong Chen. A latent-representation-centric approach to structural response reconstruction under varying training data using Cauchy–Schwarz variational autoencoders. Structures, 89:112119, 2026. doi: 10.1016/j.istruc.2026.112119.

[18] Jiangpeng Shu, Hongchuan Yu, Gaoyang Liu, Yuanfeng Duan, Hao Hu, and He Zhang. DF-CDM: Conditional difusion model with data fusion for structural dynamic response reconstruction. Mechanical Systems and Signal Processing, 222:111783, 2025. doi: 10.1016/j.ymssp.2024.111783.

[19] Deep Ray, Harisankar Ramaswamy, Dhruv V. Patel, and Assad A. Oberai. The eficacy and generalizability of conditional GANs for posterior inference in physics-based inverse problems. Numerical Algebra, Control and Optimization, 14(1):167–196, 2024. doi: 10.3934/naco.2022038.

[20] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic diferential equations, 2021. URL https://openreview.net/forum?id=PxTIG12RRHS. International Conference on Learning Representations.

[21] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL https://openreview.net/forum?id=PqvMRD CJT9t. The Eleventh International Conference on Learning Representations.

[22] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, et al. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. URL https://openre view.net/forum?id=HgDwiZrpVq.

[23] Lorenzo Baldassari, Ali Siahkoohi, Josselin Garnier, Knut Sølna, and Maarten V. de Hoop. Conditional score-based difusion models for bayesian inference in infinite dimensions. In Advances in Neural Information Processing Systems, volume 36, pages 24262–24290. Curran Associates, Inc., 2023. doi: 10.52202/075280-1055.

[24] Tianyu Chen, Vansh Bansal, and James G. Scott. Conditional difusions for amortized neural posterior estimation. In Proceedings of the 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2377–2385. PMLR, 2025. URL https://proceedings.mlr.press/v258/chen25d.html.

[25] Jonas Bernhard Wildberger, Maximilian Dax, Simon Buchholz, Stephen R. Green, Jakob H. Macke, and Bernhard Schölkopf. Flow matching for scalable simulation-based inference. In Advances in Neural Information Processing Systems, volume 36, pages 16837–16864. Curran Associates, Inc., 2023. doi: 10.52202/075280-0737. URL https://proceedings.neurips.cc /paper\_files/paper/2023/hash/3663ae53ec078860bb0b9c6606e092a0-Abstract.html.

[26] Percy S. Zhai, So Won Jeong, and Veronika Ročková. Conditional flow matching for bayesian posterior inference, 2026. URL https://openreview.net/forum?id=x6J48heM3C. Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research 300.

[27] Hyungjin Chung, Jeongsol Kim, Michael Thompson McCann, Marc Louis Klasky, and Jong Chul Ye. Difusion posterior sampling for general noisy inverse problems, 2023. URL https: //openreview.net/forum?id=OnD9zGAGT0k. The Eleventh International Conference on Learning Representations.

[28] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-Flow: Diferentiating through flows for controlled generation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 3462–3483. PMLR, 2024. URL https://proceedings.mlr.press/v235/ben-hamu24a.html.

[29] Han Gao, Xu Han, Xiantao Fan, Luning Sun, Li-Ping Liu, Lian Duan, and Jian-Xun Wang. Bayesian conditional difusion models for versatile spatiotemporal turbulence generation. Computer Methods in Applied Mechanics and Engineering, 427:117023, 2024. doi: 10.1016/j.cma.2024.117023.

[30] Zeyu Li, Wang Han, Yue Zhang, Qingfei Fu, Jingxuan Li, Lizi Qin, Ruoyu Dong, Hao Sun, Yue Deng, and Lijun Yang. Learning spatiotemporal dynamics with a pretrained generative model. Nature Machine Intelligence, 6(12):1566–1579, 2024. doi: 10.1038/s42256-024-00938-z.

[31] Pan Du, Meet Hemant Parikh, Xiantao Fan, Xin-Yang Liu, and Jian-Xun Wang. Conditional neural field latent difusion model for generating spatiotemporal turbulence. Nature Communications, 15:10416, 2024. doi: 10.1038/s41467-024-54712-1.

[32] Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. DifusionPDE: Generative PDE-solving under partial observation. In Advances in Neural Information Processing Systems, volume 37, pages 130291–130323. Curran Associates, Inc., 2024. doi: 10.52202/079017-4140.

[33] Enze Jiang, Jishen Peng, Zheng Ma, and Xiong-Bin Yan. ODE-DPS: ODE-based difusion posterior sampling for linear inverse problems in partial diferential equation. Journal of Scientific Computing, 102(3):69, 2025. doi: 10.1007/s10915-025-02790-8.

[34] Wingho Feng, Quanwang Li, Chen Wang, and Jian-sheng Fan. Data fusion for full-range response reconstruction via difusion models. Advanced Engineering Informatics, 73:104499, 2026. doi: 10.1016/j.aei.2026.104499.

[35] Ruiqi Feng, Chenglei Yu, Wenhao Deng, Peiyan Hu, and Tailin Wu. On the guidance of flow matching. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 16993–17029. PMLR, 2025. URL https://proceedings.mlr.press/v267/feng25s.h tml.

[36] Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and Omer Deniz Akyildiz. Tweedie moment projected difusions for inverse problems. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.n et/forum?id=4unJi0qrTE. Featured Certification.

[37] Martin Zach, Youssef Haouchat, and Michael Unser. A statistical benchmark for difusionposterior-sampling algorithms, 2026. URL https://openreview.net/forum?id=zDI2G8t0of. The Fourteenth International Conference on Learning Representations.

[38] Petr Kadlec, Ratko Grbić, and Bogdan Gabrys. Review of adaptation mechanisms for datadriven soft sensors. Computers & Chemical Engineering, 35(1):1–24, 2011. doi: 10.1016/j.comp chemeng.2010.07.034.

[39] Yi Liu, Chao Yang, Kaixin Liu, Bocheng Chen, and Yuan Yao. Domain adaptation transfer learning soft sensor for product quality prediction. Chemometrics and Intelligent Laboratory Systems, 192:103813, 2019. doi: 10.1016/j.chemolab.2019.103813.

[40] Xiangrui Zhang, Chunyue Song, Jun Zhao, and Zuhua Xu. Deep gaussian mixture adaptive network for robust soft sensor modeling with a closed-loop calibration mechanism. Engineering Applications of Artificial Intelligence, 122:106124, 2023. doi: 10.1016/j.engappai.2023.106124.

[41] Paul Gardner, Xin Liu, and Keith Worden. On the application of domain adaptation in structural health monitoring. Mechanical Systems and Signal Processing, 138:106550, 2020. doi: 10.1016/j.ymssp.2019.106550.

[42] Hossein Shahabadi Farahani, Alireza Fatehi, Alireza Nadali, and Mahdi Aliyari Shoorehdeli. Domain adversarial neural network regression to design transferable soft sensor in a power plant. Computers in Industry, 132:103489, 2021. doi: 10.1016/j.compind.2021.103489.

[43] Xiangrui Zhang, Chunyue Song, Jun Zhao, and Xiaogang Deng. Domain adaptation mixture of gaussian processes for online soft sensor modeling of multimode processes when sensor degradation occurs. IEEE Transactions on Industrial Informatics, 18(7):4654–4664, 2022. doi: 10.1109/TII.2021.3120509.

[44] Yongjie Shi, Xianghua Ying, and Jinfa Yang. Deep unsupervised domain adaptation with time series sensor data: A survey. Sensors, 22(15):5507, 2022. doi: 10.3390/s22155507.

[45] Ali Siahkoohi, Gabrio Rizzuti, Rafael Orozco, and Felix J. Herrmann. Reliable amortized variational inference with physics-based latent distribution correction. Geophysics, 88(3): R297–R322, 2023. doi: 10.1190/geo2022-0472.1.

[46] Yachong Yang, Arun Kumar Kuchibhotla, and Eric Tchetgen Tchetgen. Doubly robust calibration of prediction sets under covariate shift. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(4):943–965, 2024. doi: 10.1093/jrsssb/qkae009.

[47] Matthias Gelbrich. On a formula for the l2 wasserstein metric between measures on euclidean and hilbert spaces. Mathematische Nachrichten, 147(1), 1990. doi: 10.1002/mana.19901470121.

[48] Rajendra Bhatia, Tanvi Jain, and Yongdo Lim. On the Bures–Wasserstein distance between positive definite matrices. Expositiones Mathematicae, 37(2):165–191, 2019. doi: 10.1016/j.ex math.2018.01.002.

[49] Nicolas Courty, Rémi Flamary, Devis Tuia, and Alain Rakotomamonjy. Optimal transport for domain adaptation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(9): 1853–1865, 2017. doi: 10.1109/TPAMI.2016.2615921.

[50] Chuan-Xian Ren, You-Wei Luo, and Dao-Qing Dai. BuresNet: Conditional bures metric for transferable representation learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4198–4213, 2023. doi: 10.1109/TPAMI.2022.3190645.

[51] Ran Wang, Fucheng Yan, Liang Yu, Changqing Shen, and Xiong Hu. Joint wasserstein distance matching under conditional probability distribution for cross-domain fault diagnosis of rotating machinery. Mechanical Systems and Signal Processing, 210:111121, 2024. doi: 10.1016/j.ymssp.2024.111121.

[52] Lawrence Sirovich. Turbulence and the dynamics of coherent structures. i. coherent structures. Quarterly of Applied Mathematics, 45(3), 1987. doi: 10.1090/qam/910462.

[53] Stephen P. Timoshenko and S. Woinowsky-Krieger. Theory of Plates and Shells. McGraw-Hill, New York, 2 edition, 1959.

[54] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information

Processing Systems, volume 30, pages 6402–6413. Curran Associates, 2017. URL https: //proceedings.neurips.cc/paper\_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5 fa85bce38-Abstract.html.

[55] Sylvain Calinon, Florent Guenter, and Aude Billard. On learning, representing, and generalizing a task in a humanoid robot. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 37 (2), 2007. doi: 10.1109/TSMCB.2006.886952.

[56] Christopher A. T. Ferro. Fair scores for ensemble forecasts. Quarterly Journal of the Royal Meteorological Society, 140(683), 2014. doi: 10.1002/qj.2270.

[57] Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided difusion models for inverse problems, 2023. URL https://openreview.net/forum?id=9\_gsMA8MRKQ. International Conference on Learning Representations.

[58] Shun-Zheng Yu. Hidden semi-markov models. Artificial Intelligence, 174(2), 2010. doi: 10.1016/j.artint.2009.11.011.