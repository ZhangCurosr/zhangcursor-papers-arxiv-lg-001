# TIME-WARPING ESTIMATION VIA STATIONARITY-BASED LEARNING OF THE DE-WARPED SIGNAL

Corentin Presvots Adrien Meynard ˆ

CNRS, ENS de Lyon, LPENSL, UMR 5672, Lyon, France

## ABSTRACT

Time-warping estimation is a fundamental problem in signal processing with applications in bioacoustics, radar, and biomedical analysis. This paper introduces a Time-Warping Estimation Trainable (TWET) model for estimating timewarping functions from a single observation. The proposed approach formulates time-warping estimation as a stationarization problem in the wavelet domain and leverages a hierarchical dilated convolutional architecture to estimate the time-warping functions. A differentiable stationarity criterion is introduced for end-to-end optimization. TWET is compared with existing approaches. Experimental results show improved deformation reconstruction accuracy together with significantly reduced computation time, making the framework compatible with low-latency applications.

Index Terms— Time-warping estimation, wavelet transform, non-stationary signals, convolutional neural networks, inverse problems.

## 1. INTRODUCTION

Time-warping phenomena arise in many real-world signals and constitute a major source of non-stationarity. Such deformations result from propagation effects (e.g., Doppler shifts), medium variability, or intrinsic physiological dynamics [1]. They are encountered in a wide range of applications, including bioacoustics, where they affect the temporal evolution of animal vocalizations [2], radar and sonar, where Dopplerinduced distortions reflect relative motion between emitter and receiver [3], and biomedical signals such as cardiac recordings, where physiological variability requires temporal alignment for reliable analysis [4]. Estimating these deformations is therefore a key problem, as it enables the recovery of intrinsic stationary structure and improves downstream tasks such as denoising [5] or classification [6].

To address this issue, the observed signal can be interpreted as a time-reparameterized version of a latent stationary process, where local variations along the time axis encode the deformation [7]. Estimating this time-warping function from a single observation remains a challenging inverse problem, and most existing approaches rely either on multiple realizations or on strong structural assumptions [8]. In this context, time–scale representations, in particular wavelet transforms, provide a natural framework, as time-warping induces interpretable distortions in the wavelet domain. Under suitable regularity assumptions, these effects can be approximated as local translations along the scale axis [7]. This property has motivated a variety of estimation strategies, including transport-based methods [9], energy distribution approaches [7], and maximum-likelihood formulations such as the Joint Estimation of Frequency, Amplitude and Spectrum (JEFAS) [10, 11].

Despite these advances, existing approaches still present several limitations. In particular, they may struggle to accurately estimate both slowly and rapidly varying time-warping functions, their performance depends heavily on carefully tuned hyperparameters, and their computational complexity often prevents their use in low-latency applications.

To address these limitations, a trainable scheme for timewarping estimation, referred to as TWET, is proposed. The method operates on time–scale representations and formulates time-warping estimation as a stationarization problem in the wavelet domain. It relies on a hierarchical dilated convolutional architecture to estimate the time-warping function from a single observation. A key component of the approach is a differentiable stationarity criterion, which provides a taskdriven objective function and enables end-to-end optimization. The formulation can be optimized on a signal-by-signal basis, as in existing signal-specific approaches, or trained on representative data to learn a deformation model for signals from the same application domain.

The remainder of this paper is organized as follows. Section 2 formalizes the inverse problem. Section 3 presents the proposed approach. Section 4 reports the experimental results, and Section 5 concludes the paper.

## 2. FRAMEWORK

This section introduces the time-warping estimation problem. Section 2.1 presents the deformation model, Section 2.2 describes its effect in the time-scale domain, and Section 2.3 reviews existing methods.

## 2.1. Modeling Time-Warped Signals

Let $y ( t )$ denote an observed real-valued continuous-time signal, defined for $t \in \mathbb { R }$ . The signal is assumed to result from a time-warped version of an underlying latent signal $x ( t )$ . The time-warping function, denoted $\gamma ( t )$ , is assumed to be continuously differentiable and strictly increasing. The resulting model is given by

$$
y ( t ) = x ( \gamma ( t ) ) .\tag{1}
$$

It is further assumed that $x ( t )$ is modeled as a realization of a zero-mean, wide-sense stationary stochastic process, i.e.,

$$
\mathbb { E } \left[ x \left( t \right) \right] = 0 , \quad \forall t , \in \mathbb { R }\tag{2}
$$

$$
\begin{array} { r } { \mathbb { E } \left[ x \left( t \right) x \left( t + \tau \right) \right] = \mathbb { E } \left[ x \left( 0 \right) x ( \tau ) \right] , \quad \forall t , \tau \in \mathbb { R } . } \end{array}\tag{3}
$$

The stationarity assumption implies that the statistical structure of $x ( t )$ is time-invariant, so that any observed nonstationarity observed in $y ( t )$ can be attributed solely to the deformation $\gamma ( t )$

The objective is therefore to estimate $\gamma ( t )$ from the sole observation of $y ( t )$ . Equivalently, this problem can be interpreted as a stationarization task, in which a deformation $\gamma ( t )$ is sought such that

$$
x ( t ) = y ( \gamma ^ { - 1 } ( t ) ) .\tag{4}
$$

Direct estimation of the deformation $\gamma ( t )$ in the time domain is challenging due to the unknown structure of the latent signal $x ( t )$ . To overcome this difficulty, the problem is reformulated in a time-scale representation, where time-warping can be expressed in a more structured form.

## 2.2. Time-Warping as Scale Translations

The estimation of time-warped signals can be carried out naturally in a time-scale representation. In particular, it has been shown that smooth time-warping induces a simple and structured transformation of the wavelet coefficients, which makes this domain especially suitable for time-warping function estimation.

To this end, the continuous wavelet transform is considered. Let $\psi ( t ) \in L ^ { 2 } ( \mathbb { R } )$ denote a mother wavelet. The wavelet coefficients of $y ( t )$ are defined as

$$
W _ { \mathrm { y } } ( s , u ) = q ^ { - \frac { s } { 2 } } \int _ { \mathbb R } y ( t ) \psi ^ { \ast } \left( \frac { t - u } { q ^ { s } } \right) \mathrm { d } t , \quad q > 1 ,\tag{5}
$$

where $s > 0$ denotes the scale parameter and $u \in \mathbb R$ the time variable.

Under suitable regularity assumptions on $\gamma ( t )$ and on the chosen wavelet, it has been shown in [7] that the wavelet coefficients of $y ( t )$ satisfy the following approximation.

Proposition 1. Assume that $x ( t )$ is a wide-sense stationary process and that $\gamma ( t ) \in \mathcal { C } ^ { 1 }$ with slowly varying derivative.

Then,for sufficiently regular wavelets, the wavelet transforms $o f x ( t )$ and $y ( t ) = x ( \gamma ( t ) )$ are related by

$$
W _ { \mathrm { y } } ( s , u ) \approx W _ { x } ( s + \delta ( u ) , \gamma ( u ) ) ,\tag{6}
$$

where

$$
\delta ( u ) = \log _ { q } { \big ( } \gamma ^ { \prime } ( u ) { \big ) } .\tag{7}
$$

The approximation error in (6) is controlled, among otherfactors, by $\| \gamma ^ { \prime \prime } \| _ { \infty } = \operatorname* { s u p } _ { t } | \gamma ^ { \prime \prime } ( t ) |$

Proof. Given in [7].

This result plays a central role in the analysis of nonstationary signals. Since the latent signal $x ( t )$ is assumed to be stationary, its wavelet transform is statistically invariant with respect to time. In contrast, time-warping manifests itself in the wavelet domain as a translation along the scale axis. Consequently, estimating and compensating for $\delta ( u )$ allows the restoration of stationarity in the time-scale representation. This property directly motivates a large class of estimation methods that aim at recovering the deformation from observed wavelet coefficients.

## 2.3. Existing Estimation Methods

Several methods in the literature exploit the approximation result of Proposition 1 to estimate the deformation function $\delta ( u )$ from the wavelet transform. Once $\delta ( u )$ is estimated, the derivative of the time-warping function is recovered as $\gamma ^ { \prime } ( u ) = q ^ { \delta ( u ) }$ , and $\gamma ( u )$ is obtained, up to an additive constant, by integration.

In [9], the authors exploit a local relationship between the partial derivatives of the scalogram (i.e., the squared modulus of the wavelet transform) to estimate the deformation. This leads to the estimator

$$
\widehat { \delta } ^ { \prime } ( u ) = \operatorname* { l i m } _ { s  0 } ( \frac { \partial _ { u } | W _ { \mathrm { y } } ( s , u ) | ^ { 2 } } { \partial _ { s } | W _ { \mathrm { y } } ( s , u ) | ^ { 2 } } ) .\tag{8}
$$

In practice, this limit is approximated using small scales. Nevertheless, the lack of signal energy at fine scales and the sensitivity of numerical differentiation to noise significantly limit the robustness of this approach.

An alternative approach, which is referred to as the Wavelet Scale Barycenter (WSB) method [7], relies on a barycentric aggregation across scales

$$
\widehat { \delta } \left( u \right) = \frac { \int _ { \mathbb { R } _ { + } } q ^ { s } \left| W _ { \mathbf { y } } ( s , u ) \right| ^ { 2 } \mathrm { d } s } { \int _ { \mathbb { R } _ { + } } \left| W _ { \mathbf { y } } ( s , u ) \right| ^ { 2 } \mathrm { d } s } .\tag{9}
$$

In contrast to derivative-based methods, WSB avoids explicit differentiation but does not explicitly account for temporal consistency of the wavelet energy distribution, which typically leads to high estimation variance.

More recently, maximum likelihood formulations have been introduced within the JEFAS framework [10]. Based on the approximation in (6), the likelihood of the wavelet coefficients is expressed in terms of both the deformation $\delta ( u )$ and the power spectrum of the latent stationary signal $x ( t )$ Estimation is performed iteratively by alternating between updating $\delta ( u )$ and the spectral parameters.

JEFAS relies strongly on the validity of the approximation in (6), which may deteriorate when $\gamma ^ { \prime } ( t )$ varies significantly within the support of the wavelet. To improve robustness to fast variations, JEFAS-S [11] adopts a Bayesian formulation. Nevertheless, its computational complexity becomes prohibitive for long signals (typically beyond 1024 samples), limiting its applicability in practice.

## 3. PROPOSED METHOD

A trainable scheme for time-warping estimation, referred to as TWET, is proposed to address limitations of existing approaches, including sensitivity to noise and rapidly varying deformations, as well as computational cost.

Section 3.1 describes the signal discretization and segmentation strategy, Section 3.2 formulates the inverse problem in the time-scale domain, and Section 3.3 presents the proposed neural architecture.

## 3.1. Discretization and Segmentation

The observed signal $y ( t )$ is sampled uniformly with sampling period $T _ { \mathrm { s } } = 1 / F _ { \mathrm { s } }$ , yielding $y _ { n } = y ( n T _ { \mathrm { s } } )$ , where $n \in \mathbb { Z }$

Time-warping estimation is performed on short-time segments of fixed length N to reduce latency and computational cost. The signal is thus partitioned as

$$
\pmb { y } _ { i } = \left( y _ { i N } , \ldots , y _ { ( i + 1 ) N - 1 } \right) ^ { T } , \quad i \in \mathbb { Z } .\tag{10}
$$

The parameter N controls the trade-off between temporal context and latency. The segment index is omitted in the remainder of the paper when no ambiguity arises.

The time-scale representation is computed using a discretized wavelet transform on a linear scale grid $s _ { m } , m \ =$ $0 , \ldots , M - 1$ , and a uniform time grid $u _ { n } ~ = ~ n T _ { \mathrm { s } } , ~ n ~ =$ $0 , \ldots , N - 1$ , with $q > 1$ . The coefficients are defined as

$$
W _ { \mathrm { y } , m , n } = W _ { \mathrm { y } } ( s _ { m } , u _ { n } ) ,\tag{11}
$$

forming a matrix $W _ { \mathrm { y } } \in \mathbb { C } ^ { M \times N }$

The same discrete notation is used for all estimated and reconstructed quantities, including the deformation derivative $\gamma ^ { \prime }$ , the scale-shift parameter $\delta ,$ the reconstructed signal x, and its associated wavelet transform $W _ { \textrm { x } }$

## 3.2. Inverse Problem Formulation

The objective is to estimate a deformation operator acting on the wavelet representation of the observed signal such that the resulting representation matches that of an underlying stationary process. According to Proposition 1, time-warping induces approximate translations along the scale axis in the time-scale domain.

Since the values of δ are not restricted to integers, a direct shift of discrete wavelet coefficients is not well-defined. The de-warping operator is therefore defined through interpolation along the scale axis. For a given deformation δ, the de-warped wavelet coefficient at location $( m , n )$ is defined as

$$
W _ { \mathrm { y } , m , n } ^ { ( \delta ) } = \mathcal { T } _ { s } ( W _ { \mathrm { y } } ( \cdot , u _ { n } ) ) \Big | _ { s = s _ { m } - \delta _ { n } } ,\tag{12}
$$

where $\mathcal { T } _ { s }$ denotes an interpolation operator along the scale axis. The wavelet representation is evaluated on the linear scale grid indexed by $m = 0 , \ldots , M - 1$ and on the discrete time samples $n = 0 , \ldots , N - 1$ . The resulting de-warped wavelet representation is denoted by $W _ { \mathrm { ~ v ~ } } ^ { ( \delta ) }$

Under correct compensation of the deformation, Proposition 1 implies that

$$
\begin{array} { r } { W _ { \mathrm { x } } \approx W _ { \mathrm { y } } ^ { ( \delta ) } , } \end{array}\tag{13}
$$

so that the de-warped coefficients are expected to exhibit stationarity along the time axis.

Following [12], a differentiable stationarity criterion is defined as the average temporal variance $\mathrm { V a r } _ { m } ( \cdot )$ of the dewarped coefficients along the time index n at each scale m

$$
\mathcal { T } ( W _ { \mathrm { y } } ^ { ( \delta ) } ) = \frac { 1 } { M } \sum _ { m = 0 } ^ { M - 1 } \mathrm { V a r } _ { m } \Big ( W _ { \mathrm { y } } ^ { ( \delta ) } \Big ) .\tag{14}
$$

This criterion measures the temporal variability of the wavelet coefficients at each scale. For a Wide-Sense Stationary (WSS) process, their second-order statistics are time-invariant. Therefore, (14) uses their temporal variance as a tractable surrogate for stationarity, rather than explicitly enforcing WSS or estimating the full autocorrelation function. The objective is to estimate the time-warping function that minimizes this variability. Only the modulus of the wavelet coefficients is retained, as motivated by Proposition 1, which provides an explicit relation for the wavelet modulus under time warping.

The deformation estimation is then formulated as the following regularized inverse problem

$$
\mathcal { L } ( \delta ) = \mathcal { T } ( W _ { \mathrm { y } } ^ { ( \delta ) } ) + \lambda \left. \nabla \delta \right. _ { 2 } ^ { 2 } , \quad \lambda > 0 ,\tag{15}
$$

where $\nabla \delta$ denotes the discrete temporal gradient of $\delta .$ The regularization term enforces temporal smoothness of the estimated deformation, consistent with the assumption that $\gamma ^ { \prime } ( t )$ varies slowly over time.

The optimal deformation is defined as

$$
{ \widehat { \pmb \delta } } = \arg \operatorname* { m i n } _ { \pmb { \delta } } { \mathcal { L } } ( \pmb { \delta } ) .\tag{16}
$$

Nevertheless, solving (16) is challenging due to the strong non-linearity of the warping operator and the ill-posed nature of the problem in the presence of noise. In addition, the resulting optimization landscape is highly non-convex, which makes classical numerical solvers sensitive to initialization and prone to suboptimal local minima.

These limitations motivate a reformulation of the estimation problem within a parametric and data-driven framework, where the deformation is predicted by a trainable model.

## 3.3. Proposed Time-Warping Estimation Trainable Model

To improve numerical stability and reduce sensitivity to amplitude variations, a channel-wise normalization is applied to the modulus of the wavelet coefficients $| W _ { \mathrm { y } } |$ . The input is standardized by subtracting its mean and dividing by its standard deviation, yielding the normalized representation $\mathbf { \boldsymbol { X } } ^ { ( 0 ) }$

![](images/af1298c178f962d892ba4238b52c565572ea24687b10ff3f773ddb78f5ba59ba.jpg)  
Fig. 1. Architectural overview of the proposed Time-Warping Estimation Trainable (TWET) model for estimating the timewarping derivative $\gamma ^ { \prime } .$

Fig. 1 illustrates the proposed hierarchical convolutional architecture for estimating the deformation from the timescale representation. The model processes $\mathbf { } X ^ { ( 0 ) } \in \mathbb { R } ^ { C \times M \times N }$ through a sequence of L convolutional blocks.

At layer $\ell ~ = ~ 0 , \ldots , L - 1$ , the feature map $\pmb { X } ^ { ( \ell ) } \in$ $\mathbb { R } ^ { C \times M \times \check { N } }$ is processed using C dilated 2D convolutions with kernel size $\left( k _ { m } , k _ { n } \right)$ , where $k _ { m }$ and $k _ { n }$ are odd integers, chosen small compared to M and N, respectively, and a dilation factor $2 ^ { \ell }$ . The output of the c-th convolution is denoted by $Z _ { c } ^ { ( \ell ) }$ , with components given by

$$
Z _ { c , m , n } ^ { ( \ell ) } = b _ { c } ^ { ( \ell ) } + \sum _ { k = 1 } ^ { C } \sum _ { ( i , j ) \in K } w _ { c , k , i , j } ^ { ( \ell ) } X _ { k , m + 2 ^ { \ell } i , n + 2 ^ { \ell } j } ^ { ( \ell ) } ,\tag{17}
$$

where $\begin{array} { r } { \mathcal { K } = \{ - \frac { k _ { m } - 1 } { 2 } , \ldots , \frac { k _ { m } - 1 } { 2 } \} \times \left\{ - \frac { k _ { n } - 1 } { 2 } , \ldots , \frac { k _ { n } - 1 } { 2 } \right\} } \end{array}$ and where $w _ { c , k , i , j } ^ { ( \ell ) }$ and $b _ { c } ^ { ( \ell ) }$ denote the learnable convolution weights and biases. The C feature maps are concatenated as

$$
\begin{array} { r } { Z ^ { ( \ell ) } = \left( Z _ { 1 } ^ { ( \ell ) } , \dots , Z _ { C } ^ { ( \ell ) } \right) , } \end{array}\tag{18}
$$

and the output of the $\ell + 1$ -th layer with $\ell = 0 , \ldots , L - 1$ , is given by the embedded in a residual structure

$$
\begin{array} { r } { \mathbf { \boldsymbol { X } } ^ { ( \ell + 1 ) } = \mathbf { \boldsymbol { X } } ^ { ( \ell ) } + \phi \left( \mathrm { L N } ^ { ( \ell ) } ( \mathbf { \boldsymbol { Z } } ^ { ( \ell ) } ) \right) , } \end{array}\tag{19}
$$

where $\phi ( \cdot )$ is a nonlinear activation function, namely the Gaussian Error Linear Unit (GELU) [13], and $\mathrm { L N } ^ { ( \ell ) }$ denotes layer normalization [14].

After L layers, a final convolution aggregates information across scales with kernel (M, 1) and dilation factor one

$$
\delta ( \Phi ) = \mathcal { C } \Big ( X ^ { ( L ) } , M , 1 , 1 \Big ) \in \mathbb { R } ^ { N } ,\tag{20}
$$

where Φ denotes the trainable parameters (see Fig. 1).

This output corresponds to an estimate of the log-derivative of the deformation, denoted $\log _ { q } \gamma ^ { \prime } ( t )$ in discrete form. To ensure identifiability and avoid trivial ambiguities such as global shifts, the output is constrained to have zero mean.

Given the estimated deformation, the de-warping operator defined in (12) is applied to reconstruct an approximation of the latent stationary representation $W _ { \textrm { x } }$ . The network parameters are then learned by minimizing

$$
\widehat { \Phi } = \arg \operatorname* { m i n } _ { \Phi } \mathcal { L } \left( \delta ( \Phi ) \right) ,\tag{21}
$$

where $\mathcal { L } ( \cdot )$ is defined in Section 3.2.

## 4. RESULTS

This section evaluates the proposed TWET framework<sup>1</sup>. TWET is evaluated on a synthetic dataset and compared with two reference approaches: WSB [7] and JEFAS $[ 1 0 ] ^ { 2 }$ described in Section 4.1. The method of Clerc and Mallat [9] is not considered due to its limited robustness to noise, while JEFAS-S [11] is excluded because of its high computational complexity for long signals. Since the ground-truth deformation derivative $\gamma ^ { \prime }$ is available, performance is evaluated by comparing the estimated deformation derivative $\widehat { \gamma } ^ { \prime }$ with $\gamma ^ { \prime } .$

bThe methods are then evaluated on real-world signals in Section 4.2, where only y is available and the framework is optimized on a signal-specific basis. Performance is assessed through the stationarity of the estimated de-warped signal x busing (14); with representative training data, the same formulation can instead learn a deformation model for signals from the same application domain.

## 4.1. Synthetic Experiments

For TWET training, 500 synthetic signals are generated for training, 50 for validation, and 50 for testing. All methods are evaluated on the same 50 test signals. For the proposed architecture, the convolution kernels are set to $k _ { m } = k _ { n } = 3 .$ the number of convolutional layers is fixed to $L = 4$ , and the number of filters per layer is set to $C = 1 6$

Each signal is generated from a stationary process x and a strictly positive time-warping derivative $\gamma ^ { \prime }$ . The deformation is obtained by temporally smoothing Gaussian white noise using a moving-average filter of random length $F _ { T } \in [ 1 , 1 0 0 ]$ followed by a shift and rescaling enforcing unit mean, strict positivity, and standard deviation $\sigma _ { \gamma } \in [ 0 , 0 . 5 ]$

The latent stationary signal x is generated as a sum of Gaussian narrowband processes obtained by filtering white noise with Hann-shaped spectra of random bandwidths and center frequencies uniformly distributed in $\left[ 0 , F _ { s } / 3 \right]$ . This construction produces signals exhibiting both spectral and amplitude modulations. The observed signal y is then obtained using the time-warping operator defined in (1).

All signals are sampled at $F _ { s } = 4 4 1 0 0$ Hz and segmented into $N = 4 4 1 0 0 – \mathrm { s a m p l e }$ (1 s) windows. The deformation is estimated locally, with N controlling the trade-off between temporal locality and latency: smaller N improves locality but may miss the full deformation trajectory, whereas larger $N$ provides a more global estimate at the cost of increased latency.

Table 1 reports the Mean Squared Error (MSE) and Mean Absolute Error (MAE) between the estimated deformation derivative $\widehat { \gamma } ^ { \prime }$ and the ground truth $\gamma ^ { \prime }$ , together with two stabtionarity indices

$$
r _ { T } = 1 0 0 T \left( W _ { \mathrm { y } } ^ { ( \delta ) } \right) , \qquad r _ { J } = \mathcal { I } \left( W _ { \mathrm { y } } ^ { ( \delta ) } \right) ,\tag{22}
$$

where $\tau ( \cdot )$ is a stationary metrics introduced in (14) and $\mathcal { I } ( \cdot )$ is the negative log-likelihood introduced in [10]. Lower values indicate improved stationarity after de-warping. Average runtimes over the 50 test signals are also reported in Table 1.

Time (s)  
Table 1. Performance over 50 synthetic signals: deformation error, stationarity indices, and runtime.
<table><tr><td>Method</td><td>TWET</td><td>JEFAS [10]</td><td>WSB [7]</td></tr><tr><td>MSE  $\overline { { ( \gamma ^ { \prime } - \widehat { \gamma } ^ { \prime } ) } }$ </td><td>0.0070</td><td>0.0424</td><td>0.0564</td></tr><tr><td>MAE  $( \gamma ^ { \prime } - \widehat { \gamma } ^ { \prime } )$ </td><td>0.0525</td><td>0.1076</td><td>0.1938</td></tr><tr><td> $r _ { T }$  (22)</td><td>4.279</td><td>6.460</td><td>7.111</td></tr><tr><td> $r _ { J }$  (22)</td><td>-308.9</td><td>-341.5</td><td>-286.2</td></tr><tr><td>Time per signal (s)</td><td>0.0008</td><td>13.5</td><td>1.7</td></tr></table>

TWET achieves the best MSE and MAE performance for the reconstruction of the deformation derivative $\widehat { \gamma } ^ { \prime }$ compared bwith existing approaches. Part of this improvement may be explained by the similarity between the training and test distributions. In contrast, JEFAS yields lower average reconstruction accuracy, as the method may fail to converge when the deformation derivative $\gamma ^ { \prime }$ varies rapidly. Nevertheless, JEFAS achieves better performance according to the stationarity criterion $r _ { J }$ . This behavior can be explained by the fact that JEFAS is explicitly designed to minimize the criterion $r _ { J }$ TWET also provides a substantial computational advantage, reaching real-time compatibility with an average runtime of 0.0008 s per signal. Although TWET results are obtained using batched GPU processing, both JEFAS and WSB remain significantly more computationally expensive.

## 4.2. Real-World Experiments

The proposed method is evaluated on real-world audio signals, including the sound of a car accelerating with gear shifts (Fig. 2), a recording of singing (Fig. 3), and the sound of wind blowing through a window (Fig. 4).

Since no ground-truth deformation is available, the comparison relies on qualitative analysis together with the stationarity measures defined in (22). TWET is trained directly on each signal, as the amount of available real-world data is currently insufficient to train a fully generalizable model. In this setting, the proposed framework can therefore be interpreted as a signal-adaptive parametric estimator.

Fig. 2–4 illustrate examples. Each figure shows the wavelet transform of the observed signal (top left), the TWET de-warped representation (top right), and the deformation derivatives estimated by WSB, JEFAS, and TWET (bottom).

![](images/111cf0f1f46bb5a35f9f3a95241950f9866362401665b3d1f27acd4c1237701b.jpg)

![](images/2a319ad6018d1ab2d6377e773d657002a1f7db3a1cd2b7a0144dea2c01d501e2.jpg)

![](images/1301d065551844fb3fd220e18b2a8a00f2c44b1523189739ca3d570ca8d219ba.jpg)  
Time (s)  
Fig. 2. Car acceleration. Top left: wavelet transform $W _ { \mathrm { y } }$ of the observed signal y. Top right: estimated de-warped representation obtained with TWET. Bottom: estimated deformation derivatives obtained with WSB, JEFAS, and TWET.

![](images/0ff57e2ba375a420c7b225a3129033db247134d406cabccfcf42bcd671581eab.jpg)

![](images/e715a832430be889f49810f67a26777f7f7b70c3100031a4257aafe68308b549.jpg)

![](images/6c4758c833dff25885b91acae6ed25641994e0fb828f03f93ee0ba43c9469ee9.jpg)  
Fig. 3. Singing voice recording: same layout as Fig. 2.

![](images/b290419925c318592f6e5bf0ba5a7d80d091fecff6efc01e86cb30c4090b37d1.jpg)

![](images/90c6fc39e2d0203874c5ab923a2bd51271369cf13608977042bc85e528554fa7.jpg)

![](images/1c9f1f4423c64b0ac6258b3360c6ba0297836e896639a8b72259b11bcab04236.jpg)  
Fig. 4. Wind blowing: same layout as Fig. 2.

Overall, TWET slightly outperforms JEFAS and significantly improves over WSB. In particular, WSB appears more sensitive to noise and amplitude variations, resulting in considerably noisier deformation estimates. Although TWET demonstrates strong empirical performance, its generalization capability remains dependent on the diversity and representativeness of the training data.

## 5. CONCLUSION

This paper introduced TWET, a trainable scheme for timewarping estimation based on time-scale representations and a differentiable stationarity criterion. The problem is formulated as an inverse problem in the wavelet domain and solved using a hierarchical dilated convolutional architecture that estimates time-warping functions from a single observation.

Experimental results on synthetic signals demonstrated that TWET achieves lower reconstruction errors and improved stationarity restoration than reference methods. In particular, the proposed framework remains effective for rapidly varying time-warping deformations, a regime where methods such as JEFAS tend to fail or produce unstable estimates. In addition, TWET significantly reduces computation time for deformation estimation, making it compatible with low-latency applications. Experiments on real-world signals further highlighted the robustness of the proposed method to noise and amplitude variations.

Future work includes joint estimation of amplitude modulation and time deformation, as well as incorporating multiple wavelet resolutions to better handle deformation dynamics ranging from slow to rapid variations.

## 6. REFERENCES

[1] M. B. Priestley, Non-linear and non-stationary time series analysis, Academic Press, 1988.

[2] C. Ioana and A. Quinquis, “On the use of timefrequency warping operators for analysis of marinemammal signals,” in 2004 IEEE Int. Conf. Acoust. Speech Signal Process., 2004, vol. 2, pp. ii–605.

[3] G. E. Smith, K. Woodbridge, and C. J. Baker, “Radar micro-Doppler signature classification using dynamic time warping,” IEEE Trans. Aerosp. Electron. Syst., vol. 46, no. 3, pp. 1078–1096, 2010.

[4] V. Tuzcu and S. Nas, “Dynamic time warping as a novel tool in pattern recognition of ECG changes in heart rhythm disturbances,” in IEEE Int. Conf. Syst. Man Cybern., 2005, vol. 1, pp. 182–186 Vol. 1.

[5] R. Souriau, J. Fontecave-Jallon, and B. Rivet, “Fetal ECG denoising using dynamic time warping template subtraction,” in 44th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC). IEEE, 2022, pp. 4978–4981.

[6] Y. Liu, Y.-A. Zhang, M. Zeng, and J. Zhao, “A novel distance measure based on dynamic time warping to improve time series classification,” Inf. Sci., vol. 656, pp. 119921, 2024.

[7] H. Omer and B. Torresani, “Time-frequency and time- ´ scale analysis of deformed stationary processes, with application to non-stationary sound modeling,” Appl. Comput. Harmon. Anal., vol. 43, no. 1, pp. 1 – 22, 2017.

[8] K.R. Chithra, A. Remesh, and M.S. Sinith, “Matched wavelets for musical signal processing using evolutionary algorithms,” Appl. Acoust., vol. 229, pp. 110385, 2025.

[9] M. Clerc and S. Mallat, “Estimating deformations of stationary processes,” Ann. Stat., vol. 31, no. 6, pp. 1772–1821, Dec. 2003.

[10] A. Meynard and B. Torresani, “Spectral Analysis for ´ Nonstationary Audio,” IEEE Trans. Audio Speech Lang. Process., vol. 26, no. 12, pp. 2371 – 2380, Dec. 2018.

[11] A. Meynard and B. Torresani, “Synthesis-based time-´ scale transforms for non-stationary signals,” Appl. Comput. Harmon. Anal., vol. 65, pp. 112–136, 2023.

[12] P. Borgnat, P. Flandrin, P. Honeine, C. Richard, and J. Xiao, “Testing stationarity with surrogates: A timefrequency approach,” IEEE Trans. Signal Process., vol. 58, no. 7, pp. 3459–3470, 2010.

[13] D. Hendrycks and K. Gimpel, “Gaussian error linear units (GELUs),” 2023.

[14] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normalization,” 2016.