# Bridging Digital Deception and Physical Fidelity: A Spatio-Temporal Attack on Cognitive Radars

Tongrun Xing, Yongjun Dai, Xuan Liao, Zhiwen Zhang, Tong Zhou, Longlong Zhang and Tian Jiang

Abstract—As cognitive radars evolve towards the end-toend processing of one-dimensional complex baseband sequences, deep networks face severe adversarial threats. However, existing attacks encounter dual limitations in practical electronic warfare: low cross-architecture transferability and severe vulnerability to physical channel distortions, which degrades their tactical reliability. To overcome these bottlenecks, we propose a physical prior-driven Segmented Spatio-Temporal Attack (SSTA). First, a global-local feature decoupling segmentation mechanism interrupts the adversarial gradients’ dependence on long-range sequence coherence, bridging the cross-architecture feature gap. Second, moving beyond static norm constraints devoid of physical semantics, we abstract channel distortions into differentiable spatio-temporal operators, integrating them into momentum iterations via an Expectation over Transformation (EoT) strategy. Comprehensive cross-evaluations across 9 heterogeneous models spanning CNNs, RNNs, and Transformers demonstrate that SSTA outperforms the state-of-the-art baseline by 10.36% in Average Black-Box Attack Success Rate. Furthermore, baseband equivalent simulations confirm that the generated waveforms survive extreme physical distortions and, after matched filtering, achieve zero main-lobe attenuation (0 dB) with noise floor lift strictly within Constant False Alarm Rate tolerances. SSTA successfully unifies digital-domain deception and physical-domain fidelity, providing crucial support for CEW security assessment.

Index Terms—Adversarial attack, black-box transferability, radar jamming recognition, cognitive electronic warfare (CEW), deep learning.

## I. INTRODUCTION

ITH the rapid development of digital array radar and Cognitive Electronic Warfare (CEW), deep learning   
technologies have been widely applied in the field of radar   
active jamming recognition [1]–[5]. Currently, end-to-end   
cognitive networks utilizing one-dimensional (1D) complex   
baseband sequences (I/Q dual-channel) as direct inputs are   
gradually becoming the mainstream development direction, as

they can effectively preserve intrinsic electromagnetic characteristics. However, while enhancing recognition accuracy, this input mechanism directly exposes deep cognitive models to the attack surface of minute electromagnetic perturbations, triggering security concerns regarding Adversarial Examples [6]–[8]. Unlike in the computer vision domain, signal detection in real radar systems highly relies on the physical radio frequency (RF) front-end and strict coherent processing chains (e.g., matched filtering) [9]. In electromagnetic engagements, if minute adversarial perturbations can be covertly superim posed onto the original jamming echoes and maintain their perturbation features after processing by high-gain radar receivers, they can induce high-confidence misclassifications by the cognitive network, subsequently leading to target misidentification or track loss [10]. This adversarial threat, operating at the physical layer of 1D RF signals and requiring penetration of physical processing chains, constitutes a critical security issue that urgently needs to be addressed in current CEW.

However, existing adversarial attack algorithms encounter two core challenges when faced with practical radar electronic warfare scenarios. The first is the feature transfer barrier across heterogeneous architectures. Under strict black-box conditions, attackers cannot access the internal parameters of the adversary’s radar models [11]–[13]. To mitigate overfitting to a single white-box model, existing state-of-the-art black-box attack algorithms have made extensive explorations. Specifically, these advanced strategies can be broadly categorized into gradient optimization methods based on momentum evolution [14]–[16], input transformation methods aimed at feature disruption [17]–[20], and deep representation tuning strategies [21]–[24]. Nevertheless, the underlying logic of these algorithms is primarily based on the visual priors of 2D images (such as translation invariance and spatial redundancy). When directly applied to 1D radar sequences with strict temporal dependencies and phase coupling, adversarial perturbations are highly prone to overfitting the source model’s local receptive fields and specific absolute positional encodings. Consequently, when transferring across heterogeneous defense networks (e.g., from local convolution-based CNNs to global self-attention-based Transformers [25]), the attack success rate decreases significantly. The second challenge is the lack of tactical equivalence under real physical RF constraints. Traditional adversarial perturbations are mostly generated in a static digital domain. Once they encounter clock synchronization drift, orthogonal phase rotation, or noise floor fluctuations of non-cooperative receivers in actual electromagnetic channels, their adversarial features are vulnerable to fading and thus become ineffective. Furthermore, some communication or speech adversarial methods relying on auditory masking effects or global static norm constraints [26], [27] struggle to maintain waveform stealthiness under the stringent pulse compression mechanisms of radars. These limitations in channel distortion resistance and coherent processing fidelity make it difficult for existing algorithms to maintain tactical equivalence in practical jamming scenarios.

To effectively bridge the cross-architecture feature gap and satisfy the physical constraints of radar baseband receiving chains, this paper departs from the traditional static optimization framework in the purely digital domain and proposes a physical prior-driven Segmented Spatio-Temporal Attack (SSTA). This method operates directly on 1D radar complex baseband sequences, aiming to enhance the black-box transfer rate while effectively accommodating the physical characteristics of radar coherent processing.

The main contributions of this paper can be summarized in the following three aspects:

1) A segmentation mechanism based on global-local feature decoupling is proposed to mitigate the crossarchitecture black-box transfer barrier. By independently slicing and locally optimizing the baseband sequence along the time axis, this mechanism effectively blocks the dependence of adversarial gradients on the longrange coherence of a single surrogate network. This strategy of extracting generalized features within discrete local spaces reduces the feature aggregation discrepancy between heterogeneous cognitive architectures, thereby significantly improving the cross-model blackbox transfer success rate.

2) A physical prior-driven spatio-temporal Expectation over Transformation (EoT) strategy is constructed to accommodate both model generalization and physical receiving chain compatibility. This paper abstracts channel distortions (clock drift, Jamming-to-Noise Ratio (JNR) fluctuations, and orthogonal phase rotations) into differentiable spatio-temporal joint operators and incorporates them into the momentum iteration trajectory of gradients. This strategy not only achieves smooth regularization of the loss landscape, further enhancing the generalization capability of the algorithm, but also endows the generated adversarial waveforms with the physical robustness to resist channel fading and adapt to matched filtering.

3) Comprehensive cross-evaluations across multiple heterogeneous models and equivalent simulations of radar baseband are conducted, achieving an effective integration of digital-domain deception and physical-domain fidelity. In 72 cross-evaluation paths comprising 9 heterogeneous models spanning CNNs, RNNs, and Transformers, SSTA achieves an increase of over 10% improvement in Average Black-Box Attack Success Rate compared to existing baseline algorithms. In addition, equivalent simulations based on radar physical receiving chains confirm that the generated adversarial waveforms can maintain a high effective survival rate under temporal, phase, and energy distortions. After passing through the radar matched filter, they cause no obvious attenuation of the main-lobe energy, and the noise floor lift is controlled within the Constant False Alarm Rate (CFAR) tolerance, demonstrating favorable application potential.

The remainder of this paper is organized as follows: Section II defines the radar baseband signals, physical distortion models, and the adversarial optimization objective with expectation over transformation. Section III details the derivation of the SSTA framework, explaining the mathematical principles of the feature decoupling segmentation mechanism and the physical spatio-temporal operators. Section IV conducts crossgeneralization evaluations based on a matrix of 9 heterogeneous models and verifies the tactical equivalence and physical robustness of the generated waveforms. Finally, Section V concludes the paper and discusses future perspectives.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

## A. 1D Baseband Signal and Coherent Deception Jamming Model

The receiving end of modern digital array radars maps the intercepted radio frequency electromagnetic waves into onedimensional complex baseband sequences through quadrature demodulation and analog-to-digital conversion. In this paper, the radar baseband signal is rigorously defined as a complex vector $\mathbf { x } \in \mathbb { C } ^ { 1 \times L }$ of length �, which comprises In-phase (I) and Quadrature (Q) dual-channel data, denoted as $\mathbf { x } = \mathbf { x } _ { \mathrm { I } } + \mathrm { j } \mathbf { x } _ { \mathrm { Q } }$

The essence of active coherent deception jamming is to intercept, modulate, and retransmit the radar’s transmitted signal to generate realistic false targets at the receiving end [28], [29]. Let the complex baseband sequence of the original radar transmitted waveform be $\mathbf { s } \in \mathbb { C } ^ { 1 \times L }$ . Typical coherent jamming (e.g., Dense False Target Jamming, DFTJ) is constructed by superimposing multiple discrete pulse segments along the time axis, each with an independent delay $\tau _ { i } ,$ amplitude $A _ { i } ,$ and phase shift $\phi _ { i }$ . Its discrete-time mathematical model can be expressed as:

$$
x [ n ] = \sum _ { i = 1 } ^ { K } A _ { i } s [ n - \tau _ { i } ] e ^ { \mathrm { j } \phi _ { i } }\tag{1}
$$

Unlike static images in the computer vision domain, target detection in radar systems highly depends on a strict coherent processing chain, the core component of which is pulse compression (matched filtering). Suppose the impulse response of the receiver’s matched filter is designed as the time-reversed and conjugated version of the transmitted signal, i.e., $h [ n ] ~ = ~ s ^ { * } \left[ - n \right] ~ \left[ 9 \right]$ . The output �[�] of the received jamming sequence x after matched filtering manifests as a discrete convolution:

$$
y [ n ] = \sum _ { k } x [ k ] h [ n - k ]\tag{2}
$$

Because the coherent jamming signal retains the pulse intraphase history of the original radar transmitted pulse, it can achieve coherent accumulation of energy during the convolution process, forming high-power false spikes. To quantify the tactical fidelity of adversarial perturbations within the physical processing chain, this paper introduces two core metrics: Peak

Attenuation $( \Delta P _ { \mathrm { p e a k } } )$ and Noise Floor Lift $( \Delta P _ { \mathrm { n o i s e } } )$ to evaluate the quality of the 1D Range Profile after pulse compression.

Let $y _ { \mathrm { c l e a n } } [ n ]$ and $y _ { \mathrm { a d v } } [ n ]$ denote the matched filtering output sequences of the original clean jamming and the jamming with added adversarial perturbations, respectively. Peak Attenuation is defined as the logarithmic difference of their global maximum peak powers:

$$
\Delta P _ { \mathrm { p e a k } } = 1 0 \log _ { 1 0 } { \frac { \operatorname* { m a x } _ { n } | y _ { \mathrm { c l e a n } } [ n ] | ^ { 2 } } { \operatorname* { m a x } _ { n } | y _ { \mathrm { a d v } } [ n ] | ^ { 2 } } } ~ ( \mathrm { d B } )\tag{3}
$$

This metric strictly quantifies the degree of energy loss inflicted by the adversarial perturbation on the core energy of the original deception jamming. Furthermore, Noise Floor Lift quantifies the extent to which the perturbation energy disperses into non-main-lobe regions after pulse compression. Let $\Omega _ { \mathrm { n o i s e } }$ be the noise floor interval in the range profile excluding the main lobe and near side lobes. This metric is defined as the growth ratio of the average power within this interval:

$$
\Delta P _ { \mathrm { n o i s e } } = 1 0 \log _ { 1 0 } \frac { \sum _ { n \in \Omega _ { \mathrm { n o i s e } } } | y _ { \mathrm { a d v } } [ n ] | ^ { 2 } } { \sum _ { n \in \Omega _ { \mathrm { n o i s e } } } | y _ { \mathrm { c l e a n } } [ n ] | ^ { 2 } } ~ ( \mathrm { d B } )\tag{4}
$$

When launching adversarial attacks, if unconstrained digital perturbations disrupt the physical envelope coherence of the original signal, it will lead to a sharp decrease in $\Delta P _ { \mathrm { p e a k } }$ and a drastic lift in $\Delta P _ { \mathrm { n o i s e } }$ after matched filtering. This not only significantly weakens the deceptive power of the false targets but also triggers abnormal alarms in the radar’s Constant False Alarm Rate detection mechanism due to the abnormally elevated wideband noise floor, thereby exposing the jamming intention and losing tactical equivalence.

## B. Electromagnetic Channel Distortion Model

Existing digital-domain adversarial attack algorithms are typically established on the assumption of ideal static data distributions, which essentially differs from real radar electronic warfare spaces. In non-cooperative electromagnetic engagements, when adversary reconnaissance receivers intercept signals and implement jamming, complex physical distortions are inevitably introduced. To endow adversarial examples with practical survivability during the generation phase, this paper abstracts three core types of channel fading, which are unavoidable in actual physical chains, into a mathematical operator set T.

First, due to clock asynchrony in non-cooperative systems and dynamic changes in spatial distance, the baseband sequences acquired by the receiving system experience random drift along the time axis. Let $\tau \in \mathbb { Z }$ be the discrete time delay offset; the clock synchronization drift model can be expressed as a circular shift of the sequence:

$$
T _ { \mathrm { t i m e } } ( \mathbf { x } ) \left[ n \right] = x \left[ ( n - \tau ) \mathrm { m o d } L \right]\tag{5}
$$

Second, minute deviations in the receiver’s local oscillator frequency or residual Doppler shifts caused by the relative motion of targets lead to energy coupling between the $\mathrm { I } / \mathrm { Q }$ dual channels of the baseband signal. Let $\phi \in [ - \pi , \pi )$ be the introduced random phase rotation angle; the orthogonal phase rotation model can be defined as:

$$
T _ { \mathrm { p h a s e } } ( \mathbf { x } ) [ n ] = x [ n ] e ^ { \mathrm { j } \phi }\tag{6}
$$

Finally, the real electromagnetic space is filled with complex background clutter and receiver thermal noise. Under a given JNR, the channel noise model can be formulated as superimposing independent identically distributed (i.i.d.) complex Gaussian white noise $\mathbf { w } \ \sim \ C N ( \mathbf { 0 } , \sigma ^ { 2 } \mathbf { I } )$ onto the original sequence:

$$
T _ { \mathrm { n o i s e } } ( \mathbf { x } ) = \mathbf { x } + \mathbf { w }\tag{7}
$$

Traditional adversarial perturbations rely heavily on the absolute spatiotemporal coordinates of the original input �. As highlighted by existing studies [10], [30], once subjected to the aforementioned channel distortions, including $T _ { \mathrm { t i m e } } , T _ { \mathrm { p h a s e } }$ and $T _ { \mathrm { n o i s e } }$ , during actual over-the-air (OTA) propagation, the fragile decision boundaries constructed along the direction of minute gradients become highly susceptible to failure.

## C. Threat Model and Adversarial Formulation

This paper strictly defines the adversarial scenario as a Black-box Attack. Under this threat model, the attacker possesses only the same physical signal transceiving capabilities as the adversary radar and can obtain the data structure and feature scheme of the input signals, but has no access to the internal architecture, neuron weights, and backpropagation gradients of the adversary’s cognitive radar defense system $f _ { \mathrm { t g t } }$ . The attacker can only generate adversarial perturbations � based on a locally constructed white-box surrogate model $f _ { \mathrm { s r c } }$ and rely on the generalization of its feature space to achieve cross-model transfer.

Considering the dynamic range limitations of the Digitalto-Analog Converter in the radar transmitter, and to prevent the perturbation from triggering the radar receiver’s abnormal alarm mechanisms due to excessive energy, this paper adopts a strict infinity norm constraint $( L _ { \infty }$ norm) to define the physical boundary of the perturbation. Specifically, the generated adversarial perturbation � must satisfy $\| \delta \| _ { \infty } \leq \epsilon .$ , where � is the minimal perturbation budget, ensuring that the perturbation is safely hidden within the radar system’s quantization noise floor and natural clutter.

Combining the coherent physical constraints of baseband signals (Section II-A) and the electromagnetic channel distortion models (Section II-B), this paper formulates the blackbox adversarial attack on 1D radar signals as a constrained optimization problem with physical EoT. Given the true label � and the surrogate model’s loss function $\mathcal { L } ( \cdot )$ , the global optimization objective of the proposed algorithm to solve for the optimal adversarial perturbation � is formulated as follows:

$$
\begin{array} { r l } { \arg \underset { \delta } { \operatorname* { m a x } } } & { \mathbb { E } _ { T \sim \mathcal { T } } \left[ \mathcal { L } ( f _ { \mathrm { s r c } } ( T ( \mathbf { x } + \delta ) ) , y ) \right] } \\ { \mathrm { s . t . } } & { \| \delta \| _ { \infty } \leq \epsilon } \end{array}\tag{8}
$$

where $T \sim \mathcal { T }$ represents the sampling operation from the true physical distortion distribution jointly composed of time delays, phase shifts, and channel noise. This objective function explicitly establishes from a mathematical foundation that:

![](images/90eb77306505fae5b0afdafef172ef15093b595cf08901e7aec32053583f50ab.jpg)  
Fig. 1. The overall framework of Segmented Spatio-Temporal Attack.

effective adversarial perturbations should not solely pursue loss maximization on a static input x, but must maintain their destructive attack capability even after experiencing various potential physical fading. This provides a rigorous theoretical basis for deriving the spatio-temporal joint operators and momentum iteration strategies based on physical priors within the SSTA framework in Section III.

## III. SEGMENTED SPATIO-TEMPORAL ATTACK (SSTA)

## A. Overall Framework of SSTA

To overcome the cross-architecture feature barriers and physical chain disconnections analyzed in Section II, this paper steps outside the purely data-driven optimization framework and proposes the SSTA. The core motivation of SSTA lies in deeply integrating the physical priors inherent in electromagnetic channels and radar hardware systems into the generation process of adversarial perturbations.

The overall operational mechanism of the SSTA algorithm is illustrated in Fig. 1, and its entire architecture consists of three mutually synergistic core mechanisms: 1) A globallocal feature decoupling segmentation mechanism that slices the original radar signal to forcefully interrupt the continuous propagation of adversarial gradients along the time axis. 2) Physical prior-driven spatio-temporal dual-domain operators that instantiate the electromagnetic distortions (synchronization delays, power fluctuations, and phase rotations) as differentiable forward operators. 3) A tactical-equivalent EoT cascading strategy that performs asymmetric random sampling in each iteration, confining the perturbation energy to the lowsensitivity regions of radar matched filtering.

These three mechanisms are deeply embedded into the momentum iterative optimization loop, ensuring that the generated adversarial examples exhibit outstanding black-box transferability while preserving the tactical effectiveness of the original deception jamming signals.

## B. Physical Prior-Driven Spatio-Temporal Operators

To accommodate the input format of deep neural networks, the complex baseband signal defined in Section II is mapped into a dual-channel real-valued tensor. Based on the electromagnetic channel distortion models established previously, this section instantiates them as differentiable spatio-temporal operators to be integrated into the adversarial gradient computation. For an independent local signal segment $\mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } \in \mathbb { R } ^ { 2 \times \dot { L } _ { \mathrm { s e g } } }$ decoupled by the segmentation mechanism (where $L _ { \mathrm { s e g } }$ is the sample length and the 2 channels represent In-phase and Quadrature components), the algorithm designs specific timedomain and spatial-domain operators.

1) Time-Domain Cyclic Shift Operator: To simulate the synchronization delay characteristics of real hardware systems and block the overfitting of adversarial perturbations to absolute position information, the time-domain cyclic shift operator $T _ { \mathrm { t i m e } } ( \cdot )$ is applied. The algorithm first sets a maximum shift amount $\tau _ { \mathrm { m a x } }$ based on the synchronization tolerance of typical radar receivers:

$$
\tau _ { \mathrm { m a x } } = \operatorname* { m a x } ( 1 , \lfloor \eta \cdot L _ { \mathrm { s e g } } \rfloor ) ,\tag{9}
$$

where � is the maximum time-shift ratio limiting factor (set to $\eta = 0 . 0 2$ by default). Subsequently, an integer shift step $\tau \sim$ $\mathcal { U } ( - \tau _ { \operatorname* { m a x } } , \tau _ { \operatorname* { m a x } } )$ is randomly sampled. To strictly maintain the non-mutation of total energy within the observation window, the operator performs a cyclic shift along the time dimension:

$$
T _ { \mathrm { t i m e } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) [ c , t ] = \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } [ c , ( t - \tau ) \mathrm { m o d } L _ { \mathrm { s e g } } ] ,\tag{10}
$$

where $c \in \{ 0 , 1 \}$ represents the spatial channel index and $t \in$ $[ 0 , L _ { \mathrm { s e g } } - 1 ]$ represents the time step index.

2) Spatial-Domain Amplitude-Phase Distortion Operators: To simulate the dynamic distortions in signal amplitude and phase induced by complex electromagnetic spaces, we construct the Gaussian noise injection operator $T _ { \mathrm { j n r } } ( \cdot )$ and the quadrature phase rotation operator $T _ { \mathrm { p h a s e } } ( \cdot )$

For amplitude distortions, the algorithm samples a logarithmic-scale perturbation value $\mathbf { J N R } \sim \mathcal { U } ( - 5 , 5 )$ to simulate channel power fluctuations of $\pm 5 \mathrm { d B }$ . Given the average signal power $P _ { \mathrm { s i g } }$ of the local segment, the required physical noise power is $P _ { \mathrm { n o i s e } } = P _ { \mathrm { s i g } } / 1 0 ^ { ( \mathrm { J N R } / 1 0 ) }$ . A zero-mean Gaussian white noise matrix $\begin{array} { r l r } { { \bf w } } & { { } \sim } & { N ( { \bf 0 } , \sigma _ { \mathrm { n } } ^ { 2 } { \bf I } ) } \end{array}$ is generated, where $\sigma _ { \mathrm { n } } = \sqrt { P _ { \mathrm { n o i s e } } / 2 }$ . The noise injection operator is expressed as:

$$
T _ { \mathrm { j n r } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) = \mathrm { C l i p } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } + \mathbf { w } , - 1 , 1 ) ,\tag{11}
$$

where the Clip(·) function ensures the amplitude does not exceed the hardware dynamic quantization range of [−1, 1].

For phase distortions, the algorithm randomly samples a rotation angle $\theta \ \sim \ \mathcal { U } [ 0 , 2 \pi )$ and performs a joint linear transformation on the I/Q channels using a rotation matrix:

$$
T _ { \mathrm { p h a s e } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) = \left[ \begin{array} { c c } { \cos \theta } & { - \sin \theta } \\ { \sin \theta } & { \cos \theta } \end{array} \right] \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } .\tag{12}
$$

## C. Global-Local Decoupling Segmentation Mechanism

To interrupt the unconstrained propagation of adversarial gradients across the entire time series and prevent the surrogate network from overfitting to specific global sequence structures (such as absolute positional encodings), SSTA introduces a global-local decoupling segmentation mechanism.

For a given 1D radar baseband signal $\mathbf { x } \in \mathbb { R } ^ { 2 \times L }$ , the algorithm randomly samples $N - 1$ split points $p _ { 1 } , p _ { 2 } , \dotsc , p _ { N - 1 }$ from the discrete interval $[ 1 , L - 1 ]$ . Combined with $p _ { 0 } = 0$ and $p _ { N } = L$ , the original signal is physically partitioned into � non-overlapping continuous sub-segments:

$$
\mathbf { x } = { \mathrm { C o n c a t } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( 1 ) } , \mathbf { x } _ { \mathrm { s e g } } ^ { ( 2 ) } , \ldots , \mathbf { x } _ { \mathrm { s e g } } ^ { ( N ) } ) ,\tag{13}
$$

where $\mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } = \mathbf { x } [ p _ { i - 1 } : p _ { i } ]$ , and Concat(·) is the concatenation operator along the time dimension.

Simultaneously applying extreme amplitude noise and phase rotation to a single local segment can easily lead to the complete loss of its original time-frequency semantics, violating the tactical equivalence constraints. Therefore, SSTA adopts an equiprobable mutually exclusive strategy for spatialdomain distortion processing. The single spatio-temporal joint perturbation process $T _ { \mathrm { S T } } ( \cdot )$ for each segment is strictly defined as a piecewise function:

$$
\begin{array} { r } { T _ { \mathrm { S T } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) = \left\{ \begin{array} { l l } { T _ { \mathrm { j n r } } \left( T _ { \mathrm { t i m e } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) \right) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 5 , } \\ { T _ { \mathrm { p h a s e } } \left( T _ { \mathrm { t i m e } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( i ) } ) \right) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 0 . 5 . } \end{array} \right. } \end{array}\tag{14}
$$

Finally, the algorithm concatenates all distorted segments strictly in their original temporal order to reconstruct the complete adversarial carrier:

$$
T _ { \mathrm { s e g } } ( \mathbf { x } ) = \mathrm { C o n c a t } \left( T _ { \mathrm { S T } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( 1 ) } ) , T _ { \mathrm { S T } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( 2 ) } ) , \dots , T _ { \mathrm { S T } } ( \mathbf { x } _ { \mathrm { s e g } } ^ { ( N ) } ) \right) .\tag{15}
$$

This time-slice-based segmented perturbation mechanism forces the surrogate network to extract adversarial features within fine-grained local high-dimensional manifolds, guaranteeing SSTA’s outstanding cross-model black-box transferability.

## D. Tactical-Equivalent EoT and Momentum Optimization

To solve the global expected optimization objective formulated in Section II, the algorithm employs the Monte Carlo method for approximation due to the intractability of analytical solutions in the continuous physical space $\mathcal { T }$ . In the �-th iteration, for the current adversarial example $\mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) }$ , the algorithm executes � independent segmented spatio-temporal transformations in parallel. The aggregated expected adversarial gradient $\mathbf { g } ^ { ( k ) }$ is calculated as follows:

$$
\mathbf { g } ^ { ( k ) } = \frac { 1 } { T } \sum _ { m = 1 } ^ { T } \nabla _ { \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } } J \left( f _ { \mathrm { s r c } } \left( T _ { \mathrm { s e g } , m } ( \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } ) \right) , y \right) .\tag{16}
$$

By sampling expected gradients within extreme physical boundaries, the generated adversarial perturbations are finely constrained to the low-sensitivity regions of traditional baseband processing, guaranteeing physical transparency to radar matched filtering.

Subsequently, to stabilize the update direction and escape local extrema, the algorithm utilizes the Momentum Iterative Method. Incorporating $L _ { 1 }$ norm gradient normalization, the momentum $\mathbf { m } ^ { ( k + 1 ) }$ update formula is:

$$
\mathbf { m } ^ { ( k + 1 ) } = { \boldsymbol { \mu } } \cdot \mathbf { m } ^ { ( k ) } + { \frac { \mathbf { g } ^ { ( k ) } } { \| \mathbf { g } ^ { ( k ) } \| _ { 1 } } } ,\tag{17}
$$

where $\mu$ is the momentum decay factor. Finally, the update direction is extracted using the sign function (sign) and superimposed onto the current adversarial example with step size $\alpha ,$ , followed by $L _ { \infty }$ norm clipping to ensure stealthiness and dynamic range compliance:

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { a d v } } ^ { ( k + 1 ) } = \mathrm { C l i p } _ { \mathbf { x } , \epsilon } \left( \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } + \alpha \cdot \mathrm { s i g n } ( \mathbf { m } ^ { ( k + 1 ) } ) \right) . } \end{array}\tag{18}
$$

Through � momentum iteration loops, SSTA outputs adversarial examples that possess both high black-box transferability and strict radar tactical equivalence.

Synthesizing the global-local feature decoupling segmentation mechanism, physical prior-driven operators, and tacticalequivalent EoT strategy, the complete execution procedure of SSTA is summarized in Algorithm 1.

## IV. EXPERIMENTS AND RESULTS ANALYSIS

This section details the experimental infrastructure, comparative baselines, and quantitative metrics. To address the complexity of radar electronic warfare scenarios and prevent dimension explosion and attribution confusion in subsequent cross-evaluations, a rigorous representative sampling and evaluation logic is established at the outset of the experimental design.

## A. Experimental Setup and Evaluation Metrics

This section details the experimental infrastructure, comparative baselines, and quantitative metrics. To address the complexity of radar electronic warfare scenarios and prevent dimension explosion and attribution confusion in subsequent cross-evaluations, a rigorous representative sampling and evaluation logic is established at the outset of the experimental design.

1) Dataset Construction: Because real military radar engagement data is highly sensitive and classified, this experiment constructs a radar jamming signal simulation dataset based on typical active jamming systems. The radar transmitted waveform employs a Linear Frequency Modulation (LFM) signal with a pulse width of 20 �s and a bandwidth of 10 MHz.

Algorithm 1: Momentum Iterative Radar Adversarial   
Attack Based on SSTA   
Input: Source model $f _ { \mathrm { s r c } } ,$ loss function $J ;$ Clean 1D   
baseband signal $\mathbf { x } \in \mathbb { R } ^ { 2 \times L }$ , true label $y ;$ Max   
perturbation bound $\epsilon ;$ Step size $\alpha ;$ Iterations $K ;$   
Momentum decay $\mu ;$ EoT samplings $T ;$   
Segments $N .$   
Output: Adversarial example $\mathbf { X } _ { \mathrm { a d v } } .$   
Initialize: $\mathbf { x } _ { \mathrm { a d v } } ^ { ( 0 ) } = \mathbf { x } , \mathbf { m } ^ { ( 0 ) } = \mathbf { 0 }$   
for $k = 0$ to $K - 1$ do   
$\mathbf { g } _ { \mathrm { a c c } } = \mathbf { 0 }$   
for $m = 1$ to $T$ do   
Randomly generate $N - 1$ split points to divide   
$\mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) }$ into $N$ local segments $\{ \mathbf { x } _ { \mathrm { s e g } , m } ^ { ( \mathrm { i } ) } , \hdots , \mathbf { x } _ { \mathrm { s e g } , m } ^ { ( N ) } \}$   
for $\hat { j } = 1$ to � do   
$\tilde { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( j ) } = T _ { \mathrm { t i m e } } ( \mathbf { x } _ { \mathrm { s e g } , m } ^ { ( j ) } )$   
$r \sim \mathcal { U } ( 0 , 1 )$   
if $r < 0 . 5$ then   
$\hat { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( j ) } = T _ { \mathrm { j n r } } ( \tilde { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( j ) } )$   
else   
$\hat { \mathbf { X } } _ { \mathrm { s e g } , m } ^ { ( j ) } = T _ { \mathrm { p h a s e } } ( \tilde { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( j ) } )$   
end   
end   
$T _ { \mathrm { s e g } , m } ( \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } ) = \mathrm { C o n c a t } ( \hat { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( 1 ) } , \dots , \hat { \mathbf { x } } _ { \mathrm { s e g } , m } ^ { ( N ) } )$   
${ \bf g } _ { m } = \nabla _ { \bf x _ { \mathrm { a d v } } } J \left( f _ { \mathrm { s r c } } \left( T _ { \mathrm { s e g } , m } ( \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } ) \right) , y \right)$   
$\mathbf { g } _ { \mathrm { a c c } } = \mathbf { g } _ { \mathrm { a c c } } + \mathbf { g } _ { m }$   
end   
$\mathbf { g } ^ { ( k ) } = \mathbf { g } _ { \mathrm { a c c } } / T$   
$\mathbf { \boldsymbol { \mathsf { m } } } ^ { ( k + 1 ) } = \boldsymbol { \mu } \cdot \mathbf { \boldsymbol { \mathsf { m } } } ^ { ( k ) } + \mathbf { g } ^ { ( k ) } / \| \mathbf { g } ^ { ( k ) } \| _ { 1 }$   
$\tilde { \mathbf { x } } _ { \mathrm { a d v } } = \mathbf { x } _ { \mathrm { a d v } } ^ { ( k ) } + \boldsymbol { \alpha } \cdot \mathrm { s i g n } ( \mathbf { m } ^ { ( k + 1 ) } )$   
$\begin{array} { r } { \mathbf { x } _ { \mathrm { a d v } } ^ { ( k + 1 ) } = \mathbf { C } \mathrm { l i p } _ { \mathbf { x } , \epsilon } ( \tilde { \mathbf { x } } _ { \mathrm { a d v } } ) } \end{array}$   
end   
return $\mathbf { x } _ { \mathrm { a d v } } ^ { ( K ) }$

The dataset comprises 10 typical jamming types (covering 6 types of single jamming and 4 types of composite jamming, detailed in Table I). The simulation dataset consists of 1,000 independent randomly generated samples for each type. All signals extract the complex baseband envelopes of the I/Q dual channels, with the sampling length truncated to 2000, and are Min-Max normalized to [−1, 1] to accommodate the physical dynamic range of the digital baseband.

2) Evaluation Model Pool and Representative Sampling: To investigate the generalization laws of adversarial perturbations across different underlying feature extraction mechanisms, this paper selects 9 highly representative heterogeneous networks as evaluation baselines. The model pool comprehensively covers three mainstream architectures: CNNs based on local receptive fields (ResNet18 [31], DenseNet121 [32], EfficientNet [33], SEResNet [34]), RNNs focusing on longrange hidden state accumulation (DeepLSTM [35], DeepGRU [36], CNN-BiLSTM [37]), and Transformers based on global self-attention (SwinTransformer [38], Conformer [39]). To manage the computational overhead of the Expectation over Transformation (EoT), we uniformly sample 100 samples from each jamming category to construct a standardized adversarial evaluation subset totaling 1,000 samples. On this subset, the defense systems exhibit extremely high classification accuracies (exceeding 98.00% across all models, with six of them achieving exactly 100%), providing a rigorous baseline for adversarial testing.

TABLE I  
SUMMARY OF JAMMING SIGNALS IN THE SIMULATION DATASET
<table><tr><td>Jamming Type</td><td>Key Parameter Range</td><td>Modulation Characteristics</td></tr><tr><td>DFTJ</td><td>False targets: 3–6,  $\mathrm { D e l a y } \colon 1 { - } 1 0 ~ \mu \mathrm { s }$ </td><td>Multi-pulse linear time shifting</td></tr><tr><td>ISRJ</td><td>Retransmissions: 1-4, Sampling period: 5–20 µs</td><td>Periodic spectral notches</td></tr><tr><td>RDJ</td><td>Lag time:  $1 - 1 0 ~ \mu \mathrm { s }$ </td><td>Constant frequency shift + time delay extension</td></tr><tr><td>SNJ</td><td>Duty cycle: 20%–50%, Delay:  $1 - 5 ~ \mu \mathrm { s }$ </td><td>Intermittent sampled noise retransmission</td></tr><tr><td>SPJ</td><td>Bandwidth: 20–40 MHz</td><td>Narrowband spectrum</td></tr><tr><td>SWJ</td><td>Sweep period: 40–80 µs, Bandwidth: 20 MHz</td><td>dwelling Time-varying sloped sawtooth spectrum</td></tr><tr><td>Composite Jamming</td><td>Superposition of 2–3 jamming parameters</td><td>Multi-modal time- frequency coupling</td></tr></table>

Faced with the potential spatial evaluation dimension explosion of a $9 \times 9$ transfer matrix, this paper adopts a rigorous representative sampling logic for subsequent mechanism and physical robustness analyses. Specifically, ResNet18 is identified as the Anchor Surrogate. As a benchmark in deep learning, its local receptive field characteristics provide a rigorous reference base for evaluating cross-architecture transfer. To maximize architectural differences and evaluation stringency, specific black-box targets are selected based on the principle of Orthogonal Inductive Biases: SwinTransformer and Conformer represent global attention paradigms, DeepGRU represents gated sequential memory mechanisms, whereas DenseNet and EfficientNet represent advanced dense topologies. This orthogonal combination can effectively examine the underlying generalization capability of adversarial perturbations. Furthermore, to avoid visual clutter in mechanism visualization and physical validation, a core evaluation subset is selected from the eight comparative baselines: the traditional momentum baseline MI-FGSM, the input transformation SIA, the strong baseline VMI-FGSM, and the proposed SSTA.

3) Evaluation Metrics and Implementation Details: The evaluation framework of this experiment is built upon the axiomatic logic of “fundamental attack success rate—adversarial space division—statistical derivation.” Let the standardized adversarial evaluation subset be $\mathcal { D } _ { \mathrm { t e s t } } = \{ ( \mathbf { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , and the adversarial samples generated by the source model $f _ { \mathrm { s r c } }$ be $\{ \mathbf { x } _ { \mathrm { a d v } , i } ^ { \mathrm { s r c } } \} _ { i = 1 } ^ { N }$ . For a given target model $f _ { \mathrm { t g t } } ,$ , the foundational Attack Success Rate (ASR) is defined as the proportion of samples causing misclassification:

$$
\mathrm { A S R } ( f _ { \mathrm { s r c } } \to f _ { \mathrm { t g t } } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left( \arg \operatorname* { m a x } f _ { \mathrm { t g t } } ( \mathbf { x } _ { \mathrm { a d v } , i } ^ { \mathrm { s r c } } ) \neq y _ { i } \right)\tag{19}
$$

where I(·) is the indicator function. Upon this cornerstone, according to the attack-defense boundary, the evaluation system is strictly divided into two core adversarial spaces:

1) White-Box ASR (WB-ASR): Constrained by $f _ { \mathrm { s r c } } = f _ { \mathrm { t g t } }$ reflecting the absolute attack strength against the source model’s features.

2) Black-Box ASR (BB-ASR): Constrained by $f _ { \mathrm { s r c } } \neq f _ { \mathrm { t g t } } ,$ representing the underlying transfer capability across heterogeneous network feature barriers.

Subsequent performance metrics are derived from these statistics, including Average WB-ASR (AWB-ASR), Average BB-ASR (ABB-ASR), Cross-Architecture BB-ASR, and Generalization Gap. Furthermore, addressing the distortion of radar signals in physical channels, an Effective ASR metric is specifically introduced in Section 4.5.

All gradient-based iterative attack algorithms follow identical underlying constraints: the infinity norm is strictly limited to $\epsilon \ = \ 0 . 0 5$ , the single iteration step size is $\alpha ~ = ~ 0 . 0 0 5$ iterations $K \ = \ 2 0 .$ , and momentum decay factor $\mu ~ = ~ 1 . 0$ For the proposed SSTA, unique hyperparameters are optimally configured based on sensitivity analysis: segment count $N = 3$ EoT physical sampling count $T = 1 0$ , and max delay ratio limit $\eta = 0 . 0 2$

## B. Comprehensive Attack Performance and Transferability

This section presents and analyzes the black-box transfer attack results of the proposed algorithm and the comparison baselines on 9 heterogeneous target models. By quantitatively analyzing the cross-transfer success rates of adversarial samples between heterogeneous cognitive systems, from two dimensions of the macroscopic black-box space’s overall performance and the microscopic cross-architecture transfer links, we explore the generalization rules of adversarial perturbations across the underlying feature extraction mechanisms.

1) Comprehensive Black-Box Space Performance: In the macroscopic evaluation dimension, based on the standardized adversarial evaluation subset of 1,000 samples established in Section IV-A, this experiment comprehensively records the transfer performance of adversarial samples generated by various attack algorithms on the full set of models against unknown black-box target models. Table II details the global black-box transfer attack success rates.

Statistical data indicate that the SSTA algorithm exhibits a significant advantage in comprehensive transfer capability. Its Average BB-ASR reaches 67.01%, achieving an absolute improvement of 10.36% compared to the best-performing baseline algorithm VMI-FGSM (56.65%). Our analysis reveals that traditional adversarial attack methods based on the visual priors of 2D images exhibit notable limitations when applied to 1D radar baseband signals. For instance, TIM and FTM yield Average BB-ASRs of only 30.33% and 37.13%, respectively. This is because the local temporal features and global phases of radar sequences possess strong physical coupling. Simple

1D spatial smoothing or feature tuning easily destroys the complex envelope coherent structure of the perturbed waveform, leading to severe source-model overfitting. In contrast, SSTA mitigates this by introducing spatio-temporal dualdomain physical operators aligned with receiver mechanisms, effectively enhancing the robustness of the perturbation under black-box transfer conditions.

To more intuitively visualize the distribution patterns of the global transfer performance, Fig. 2 presents the attack success rate heatmaps for MI-FGSM, VMI-FGSM, and SSTA under the $9 \times 9$ cross-evaluation matrix. In the color mapping of this experiment, darker tones represent higher attack success rates, while lighter tones indicate lower success rates. Visually, the dark areas of MI-FGSM are almost entirely concentrated along the diagonal, indicating its weak blackbox generalization capability; VMI-FGSM shows slightly deepened tones on some near-source models, but its overall performance across structurally distinct networks remains pale. Conversely, SSTA exhibits broad and uniform dark distributions in off-diagonal regions, confirming that the proposed algorithm effectively alleviates the reliance of adversarial perturbations on specific source network parameters. Notably, even when utilizing SwinTransformer (which relies heavily on global absolute positional encoding) as the source model— where traditional algorithms struggle to transfer because the gradients contain extensive specialized information targeting global correlations—SSTA still maintains a transfer success rate lower bound of 56.74%. This value even outperforms the migration performance of most traditional algorithms under optimal source model conditions, verifying the generalization stability of the algorithm independent of specific high-quality surrogate models.

2) Cross-Architecture Feature Barrier Breakthrough Analysis: Maximizing the architectural orthogonality (Orthogonal Inductive Biases) of cognitive defense systems is a core defensive measure to block adversarial generalization. To deeply investigate the breakthrough mechanism of SSTA across different feature aggregation paradigms, conditional averaging is performed on model pairs belonging to the same architecture based on the full evaluation matrix (Table II). Based on this calculation logic, Table III and Fig. 3 present the finegrained quantitative results and the cross-architecture transfer radar chart of the core comparative subset (including MI-FGSM, SIA, VMI-FGSM, and SSTA) on 6 cross-architecture backbone transfer links.

The geometric profile of the radar chart visually reflects the asymmetric blocking characteristics of deep learning architectural differences against adversarial energy. On paths where feature extraction preferences partially intersect (e.g., the transition from local convolutions to temporal memory accumulation in the CNN → RNN path, and from global attention to temporal memory in the Trans → RNN path), the core algorithms generally exhibit favorable transferability. Specifically, SSTA’s black-box transfer success rates reach 76.85% and 71.17%, respectively, indicating that introducing physical channel priors via Expectation over Transformation significantly broadens the feature coverage of adversarial examples within the high-dimensional manifold space.

TABLE II  
COMPREHENSIVE BLACK-BOX TRANSFER ATTACK SUCCESS RATES (%) ACROSS 9 HETEROGENEOUS MODELS
<table><tr><td>Algorithm</td><td>ResNet18</td><td>DenseNet121</td><td>EfficientNet</td><td>SEResNet</td><td>DeepLSTM</td><td>DeepGRU</td><td>CNN-BiLSTM</td><td>SwinTrans</td><td>Conformer</td><td>Average BB-ASR</td></tr><tr><td>FGSM [40]</td><td>34.89</td><td>18.29</td><td>45.71</td><td>41.34</td><td>51.46</td><td>46.20</td><td>36.06</td><td>38.62</td><td>51.81</td><td>40.49</td></tr><tr><td>MI-FGSM [14]</td><td>47.79</td><td>24.50</td><td>39.04</td><td>55.62</td><td>50.19</td><td>49.86</td><td>32.58</td><td>33.24</td><td>59.64</td><td>43.61</td></tr><tr><td>VMI-FGSM [16]</td><td>63.80</td><td>62.44</td><td>44.69</td><td>63.98</td><td>61.01</td><td>66.39</td><td>43.56</td><td>32.60</td><td>71.40</td><td>56.65</td></tr><tr><td>DIM [17]</td><td>49.54</td><td>22.81</td><td>50.02</td><td>55.95</td><td>43.33</td><td>53.81</td><td>24.98</td><td>33.62</td><td>65.25</td><td>44.37</td></tr><tr><td>TIM [18]</td><td>28.59</td><td>13.78</td><td>31.52</td><td>31.48</td><td>30.39</td><td>28.48</td><td>27.57</td><td>38.59</td><td>42.58</td><td>30.33</td></tr><tr><td>SIA [19]</td><td>53.69</td><td>48.84</td><td>51.78</td><td>57.92</td><td>53.01</td><td>62.60</td><td>33.10</td><td>35.60</td><td>52.55</td><td>49.90</td></tr><tr><td>ADMIX [20]</td><td>50.96</td><td>34.49</td><td>40.16</td><td>28.76</td><td>34.16</td><td>46.73</td><td>31.46</td><td>33.53</td><td>40.58</td><td>37.87</td></tr><tr><td>FTM [24]</td><td>49.56</td><td>6.61</td><td>45.04</td><td>50.02</td><td>28.03</td><td>39.05</td><td>20.62</td><td>37.01</td><td>58.21</td><td>37.13</td></tr><tr><td>SSTA (Ours)</td><td>76.92</td><td>73.31</td><td>66.55</td><td>80.01</td><td>60.26</td><td>65.81</td><td>50.25</td><td>56.74</td><td>73.19</td><td>67.01</td></tr></table>

![](images/410a6316bb735478f68a7de8e9408f007d5e6965e050378a07f933017d509a8e.jpg)

![](images/b4ed9a430bac785e098e743c3e9d2c77059782aba9f1265342c3d8f43687e488.jpg)

![](images/eeb94ae6af0ae12059ae9d61dc2894902e4ac3c5118a5de93c3aa611aa29952a.jpg)  
Fig. 2. Heatmap of transfer attack success rates across the 9 × 9 cross-evaluation matrix.

TABLE III  
TRANSFER ATTACK SUCCESS RATES (%) ACROSS DIFFERENT FEATURE EXTRACTION MECHANISMS
<table><tr><td>Transfer Path MI-FGSM SIA VMI-FGSM</td></tr><tr><td>SSTA CNN→RNN</td></tr><tr><td>47.41 61.66 63.34 76.85 20.30 25.85 33.55 52.73</td></tr><tr><td>CNN→Trans RNN→CNN 44.31 51.02 58.55 60.07</td></tr><tr><td>RNN→Trans 26.72 30.90 36.95 42.55</td></tr><tr><td>Trans→CNN 45.81 42.18 51.38 66.50</td></tr><tr><td>Trans→RNN 54.95 54.65 60.37 71.17</td></tr></table>

However, when the transfer link shifts to the chasm crossing from local receptive field architectures to global self-attention architectures (i.e., the CNN → Trans and RNN → Trans paths), traditional algorithms exhibit widespread limitations in attack performance due to their inability to overcome the underlying feature aggregation barriers. As illustrated in Fig. 3, the contour lines of the baseline algorithms shrink sharply inward when facing the two Transformer target axes. Notably, the transfer success rates of MI-FGSM and SIA on the CNN → Trans path drop to as low as 20.30% and 25.85%, respectively. Even the strong baseline VMI-FGSM, which incorporates variance tuning, can only maintain transfer levels of 33.55% and 36.95% on these two paths.

![](images/524bbbbefae7ad1bb490bab5114fa41f2cd5a7f51bfdf8593356f31c3fa22918.jpg)  
Fig. 3. Radar chart of cross-architecture transfer attack success rates on six core paths.

In stark contrast, SSTA’s performance envelope achieves a significant outward expansion on these two arduous paths, remarkably securing transfer success rates of 52.73% (CNN → Trans) and 42.55% (RNN → Trans) against the trend. This result validates the effectiveness of the global-local feature decoupling segmentation mechanism: by introducing random physical slice boundaries along the time axis, the algorithm forcefully interrupts the adversarial gradient propagation’s over-reliance on the long-range coherent features of a single network, avoiding the perturbation’s local overfitting to the source network’s absolute positional encodings, and thereby effectively penetrating the multi-head attention defense mechanisms of the black-box Transformer. Furthermore, the closed red curve of SSTA in the radar chart is relatively balanced across all axes, eliminating the asymmetric fluctuation characteristics of traditional algorithms that heavily depend on specific transfer directions. This fundamentally consolidates the tactical dominance of the perturbed waveforms under unknown orthogonal architecture defenses from the perspective of underlying feature optimization dynamics.

## C. Ablation Study and Parameter Sensitivity

This section aims to verify the necessity of the spatiotemporal dual-domain operators and their cascading strategy in SSTA through ablation experiments, and to deeply explore the underlying effects of key hyperparameters—such as the number of iterations, the number of segments, and the sampling frequency—on the model’s convergence and generalization performance.

1) Ablation Analysis of Spatio-Temporal Operators: To quantify the independent contributions of each physical operator to the model’s feature decoupling and generalization capabilities, this experiment constructs 4 independent algorithm variants on the standardized test subset, using the traditional MI-FGSM as the baseline algorithm (Baseline) and progressively stripping or replacing the core physical modules in SSTA. Specifically, variant V1 retains only the time-shift operator; V2 uses only the spatial-domain operators (i.e., equiprobably applying JNR noise or phase rotation); V3 employs a cascade of time shift and a single phase rotation operator; and V4 employs a cascade of time shift and a single JNR noise injection operator. Table IV records the precise quantitative evaluation metrics for each variant. Meanwhile, Fig. 4 plots an intuitive performance comparison bar-line chart, where the bars represent the average attack success rate and the red line represents the Generalization Gap (GAP).

TABLE IV  
PERFORMANCE COMPARISON OF DIFFERENT ABLATION VARIANTS
<table><tr><td>Variant</td><td>AWB-ASR (%)</td><td>ABB-ASR (%)</td><td>Gap (%)</td></tr><tr><td>Baseline (MI-FGSM)</td><td>93.11</td><td>43.89</td><td>49.22</td></tr><tr><td>V1: Time-Only</td><td>86.00</td><td>52.07</td><td>33.93</td></tr><tr><td>V2: Space-Only</td><td>91.44</td><td>65.19</td><td>26.25</td></tr><tr><td>V3: Time+Phase</td><td>82.56</td><td>57.22</td><td>25.34</td></tr><tr><td>V4: Time+JNR</td><td>75.44</td><td>58.69</td><td>16.75</td></tr><tr><td>SSTA (Full)</td><td>85.00</td><td>67.46</td><td>17.54</td></tr></table>

From the experimental results, it can be observed that the purely digital domain baseline algorithm (Baseline), which completely lacks physical priors, falls into severe white-box overfitting, with its Generalization Gap (GAP) reaching as high as 49.22%. After introducing single-domain physical operators (V1, V2), the overfitting phenomenon is alleviated to a certain extent, and the GAP narrows significantly. However, unrestrictedly enforcing the superposition of cross-domain operators (V3, V4) contrarily leads to the overload distortion of adversarial energy. For instance, the Average White-Box Success Rate (AWB-ASR) of V4 drops sharply to 75.44%, indicating that excessive joint transformations severely destroy the fundamental features of the original deception jamming, resulting in the degradation of the algorithm’s foundational blinding lethality. The SSTA algorithm successfully resolves this contradiction by ingeniously introducing an “equiprobable mutually exclusive” cascading strategy for spatial operators. While stabilizing the AWB-ASR at 85.00%, it pushes the Average BB-ASR to a peak of 67.46%. This result not only proves the indispensability of spatio-temporal dual-domain operators in resisting black-box defenses but also confirms the scientific rationality of the SSTA cascading strategy in balancing “underlying feature destruction” and “generalization space expansion”.

![](images/7bcddbfd8cfecf63909561a6a40ed0214afdb3ed369eb344a4666909a3f8a661.jpg)  
Fig. 4. Performance comparison trends of different ablation variants.

2) Sensitivity to Iteration Count and Convergence: The number of momentum iterations � determines the optimization depth of adversarial perturbations in the solution space. Fig. 5 tracks the evolutionary trajectories of the Average BB-ASR across 72 cross-evaluation paths with respect to the iteration number � ∈ [0, 20]. It can be observed that most traditional momentum-based algorithms tend to saturate within the interval of � = 14 to � = 16, subsequently entering a plateau or even exhibiting slight degradation (overfitting). In contrast, SSTA’s performance curve consistently overrides all comparative algorithms throughout the entire iteration cycle and maintains a steadily rising performance scaling trend even at � = 20. This indicates that the expectation over transformation guided by physical priors not only smooths the loss manifold but also effectively avoids gradient oscillation and stagnation at local minima, endowing the algorithm with stronger resilience against overfitting and deeper feature generalization potential.

3) Sensitivity to Segmentation Count and Sampling Frequency: The number of segments � in the global-local feature decoupling mechanism directly determines the blocking granularity of adversarial gradients along the time axis. As can be seen from the macroscopic data in Table V, when physical slicing is not performed $( N = 1 )$ , globally uniform distortions fail to break the model’s dependence on coherent features, resulting in an Average BB-ASR of merely 47.26%. As the number of segments increases to $N = 3 .$ , the Average BB-ASR experiences a leap. However, when $N \geq 5 .$ , the performance growth exhibits a clear diminishing marginal utility effect.

![](images/6ace3e72443c54e5c435deade00ec8a3532d83a3f18041d2d56778ffb0ea74e1.jpg)  
Fig. 5. Evolution curves of Average BB-ASR with respect to the iteration number �.

TABLE V  
ATTACK SUCCESS RATES (%) UNDER DIFFERENT SEGMENTATION COUNTS �
<table><tr><td>Segments (N)</td><td>1</td><td>2</td><td>3</td><td>5</td><td>7</td><td>10</td></tr><tr><td>AWB-ASR</td><td>64.33</td><td>81.22</td><td>85.00</td><td>85.78</td><td>87.00</td><td>87.56</td></tr><tr><td>ABB-ASR</td><td>47.26</td><td>64.31</td><td>67.46</td><td>68.68</td><td>69.61</td><td>69.71</td></tr></table>

![](images/9ce3686f2823549c6d286c056c5fbe167279b53dde2fa6ccff7dd4c68c1e0149.jpg)  
Fig. 6. Evolution dynamics of transfer success rates across different architectural paths with respect to segmentation count �.

The cross-architecture curves in Fig. 6 profoundly reveal the microscopic decoupling mechanism behind this phenomenon: when � changes from 1 to 3, the curves of all 6 cross-domain links (including the CNN → Trans and RNN → Trans paths, which possess the deepest feature chasms) exhibit extremely steep climbs. This demonstrates that local segmentation forcefully interrupts the overfitting of the Transformer’s multihead attention mechanism to long-range absolute positional encodings. Once crossing the $N = 3$ mark, the slopes of all curves rapidly flatten, indicating that the gradient decoupling advantage has been fully released. Considering that excessive slicing leads to signal fragmentation and destroys the physical coherence of actual radar pulse modulation, this experiment comprehensively balances generalization gains and waveform fidelity, strictly establishing the optimal segmentation count as $N = 3$

TABLE VI  
PERFORMANCE AND OVERHEAD UNDER DIFFERENT EOT SAMPLING COUNTS �
<table><tr><td>EoT Samplings (T)</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td>AWB-ASR (%)</td><td>77.67</td><td>82.00</td><td>85.00</td><td>84.33</td><td>85.44</td></tr><tr><td>ABB-ASR (%)</td><td>59.86</td><td>65.63</td><td>67.46</td><td>68.63</td><td>69.39</td></tr><tr><td>Total Time (s)</td><td>495.75</td><td>544.93</td><td>720.61</td><td>1158.04</td><td>1550.88</td></tr><tr><td>Avg Time (s)</td><td>0.55</td><td>0.61</td><td>0.80</td><td>1.29</td><td>1.72</td></tr></table>

On the other hand, the Expectation over Transformation sampling count � is the key to balancing generalization stability and computational overhead. Based on the aforementioned standardized test sample subset, Table VI records the Average BB-ASR and the average generation time per sample as the number of samplings varies. Fig. 7 provides a set of in-depth dual evaluation subplots: the left subplot 7(a) is a boxplot of the black-box transfer success rates (displaying variance and extreme cases), while the right subplot 7(b) shows the tradeoff curve between the generalization lower bound and time cost. Combined with the distribution evolution in Fig. 7(a), a single blind sampling $( T = 1 )$ leads to extremely unstable gradient update directions. This is visually reflected by the exceptionally wide interquartile range in the boxplot, with the worst-case transfer rate dropping close to 10%. As � increases to 10, the boxes converge significantly, effectively achieving variance reduction and solidly elevating the worst-case lower bound to above 20%. However, Fig. 7(b) reveals the computational bottleneck of unrestrictedly increasing �. Beyond $T ~ = ~ 1 0$ , the blue solid line representing the generalization lower bound exhibits a logarithmic deceleration trend (yielding marginal gains up to $T = 2 0 )$ , whereas the red dashed line representing the time cost surges almost linearly. As detailed in Table VI, forcibly increasing � from 10 to 20 doubles the average time cost per sample from 0.80 s to 1.72 s. In radar electronic warfare scenarios with stringent latency constraints, such a duplication in time cost is unacceptable. Therefore, synthesizing the intersecting trends of the curve slopes in Fig. 7(b) and practical engineering constraints, this paper identifies $T ~ = ~ 1 0$ as the optimal balance point between high tactical effectiveness and low computational overhead.

![](images/ef934b42c26530a2e0bb5b37a34a2043bc437c3d1a21ad6bc7b97a1fa7bd7359.jpg)  
(a) Variance Reduction in Transferability

![](images/db025011948b3f929f1e48c65d7bf8c1c925c694c7483dd1c3fc3b365e318f9b.jpg)  
(b) Trade-off: Robustness vs. Computation  
Fig. 7. Comprehensive evaluation of EoT sampling count �: (a) Boxplot of cross-model transfer rate variance; (b) Trade-off between worst-case transfer rate and time cost.

## D. Mechanistic Analysis of Transferability

The previous macroscopic evaluations have fully quantified the significant advantages of the SSTA algorithm when crossing heterogeneous cognitive defense systems. To explore the origin of this high transferability from the underlying mathematical principles of deep learning, this section will follow the representative sampling logic established in Section IV-A. By selecting specific source and target model paths, we deeply dissect the microscopic optimization dynamics mechanism of adversarial perturbations breaking through architectural barriers from two core dimensions: high-dimensional Feature Manifold and cross-model Gradient Alignment.

1) Evolution of High-Dimensional Feature Manifolds: The classification decision of deep neural networks essentially partitions hyperplanes in a high-dimensional latent space. For an adversarial example to be effective in a black-box model, the deep feature distribution it extracts within the target model must successfully cross the decision boundary of the original clean samples. To balance the clarity of high-dimensional spatial visualization and the statistical representativeness of categories, this experiment uniformly extracts 300 standardized samples containing all 10 jamming systems from the global test set as the analysis base by fixing the random seed. Based on this, t-SNE technology is utilized to perform dimensionality reduction visualization on the hidden features output by the penultimate layer of the target model. Fig. 8 extracts the feature manifold distributions of the baseline algorithm (MI-FGSM) and the proposed algorithm (SSTA) on three cross-architecture physical links (ResNet18 → SwinTransformer, ResNet18 → DeepGRU, and SwinTransformer → EficientNet). The scatter points represent clean samples (light-colored dots) and adversarial samples (dark-colored markers), respectively, while the dashed arrows precisely quantify the Centroid Shift Vector for each category.

Observing the high-dimensional manifold distributions in Fig. 8, because MI-FGSM relies heavily on the specific local receptive field of the source model, the adversarial sample features it generates stagnate at the margins of the clean sample clusters in the black-box latent space of the target model. The centroid shift vectors are extremely short, failing to substantively penetrate the discriminant boundary of the defense model. In contrast, SSTA generally induces significant directed centroid shifts, and after breaking away from the original distribution, it re-aggregates into highly compact, erroneous feature clusters.

A deeper analysis of the microscopic evolution trajectories of specific cross-domain links can further reveal the nonuniformity differences in adversarial generalization at the feature level. First, in the ResNet18 → SwinTransformer group with the most drastic architectural differences, the adversarial feature clusters of MI-FGSM almost completely overlap with the clean samples. Conversely, SSTA successfully induces the multi-class jamming features (e.g., RDJ, ISRJ, SNJ, and ISRJ+RDJ) to cross the stringent boundary of the global selfattention mechanism, achieving clear directed shifts. Furthermore, in the ResNet18 → DeepGRU and SwinTransformer → EficientNet links, a critical phenomenon of adversarial manifold degradation is revealed: although MI-FGSM compels some samples to escape the original cluster, a large number of adversarial samples remain highly entangled with the clean distribution, leading to a severe “intra-class dispersion” of the adversarial cluster in the latent space. This drastic intraclass divergence causes its arithmetic centroid shift vector to actually point to an empty manifold region lacking genuine sample support (i.e., an ineffective discriminant domain). In stark contrast, the adversarial samples generated by SSTA not only successfully detach from the original distribution but also maintain extremely high feature cohesion in the unknown black-box space, thereby forming a far more lethal and directed misleading clustering.

However, from a rigorous academic perspective, the above experiments also objectively expose the inherent physical limitations of the current additive adversarial mechanism. Observing Fig. 8, it can be found that for categories such as sweep jamming (SWJ), comb spectrum jamming (SPJ), and specific composite jamming (ISRJ+SWJ), the cluster shift magnitudes induced by both MI-FGSM and SSTA are relatively constrained. From the perspective of intrinsic signal analysis, these highly sparse spectrum or multi-domain strongly coupled jamming systems form extremely stable and isolated feature representation spaces in heterogeneous models. Under the strict infinity norm minimal perturbation constraint (� = 0.05), the limited additive perturbation energy struggles to cross their broad decision boundaries. This not only accurately depicts the efficiency bottleneck of existing minimal additive adversarial frameworks in coping with highly robust physically isolated features but also points out a clear evolutionary direction for exploring non-additive or large-scale structural adversarial attacks in the future.

![](images/7f1934b1f87eb5072e097271a49d489cb43b8cd8f2ed15cebb3f157787fd7159.jpg)  
Fig. 8. t-SNE dimensionality reduction visualization of hidden feature manifolds under different cross-architecture transfer paths

2) Cross-Model Gradient Alignment and Optimization Dynamics: The theoretical upper bound of the adversarial sample transfer success rate depends on the extent to which the optimization direction on the white-box source model can approximate the true descent direction of the black-box target model. To quantify the compatibility of this optimization dynamics, this paper introduces the Gradient Cosine Similarity metric across models. In specific implementation, to strictly eliminate the gradient noise interference caused by initial classification errors of the models, this experiment directionally screens 32 pure samples that can be correctly classified by the source model with 100% confidence as the optimization base. Based on this standard subset, Fig. 9 tracks the real-time alignment trajectories of the four core algorithms during a 20-step iteration cycle. This experiment uses ResNet18 as the unified benchmark source model to calculate the similarity between its iterative surrogate gradients and the true gradients of three black-box models: DenseNet, DeepGRU, and Conformer.

Analyzing the dynamic evolution curves in Fig. 9, in the early iteration stage $( K < 8 )$ , the gradient alignment degree of all algorithms exhibits a rapid upward trend. This indicates that in the initial stage of perturbation exploration, all algorithms can effectively capture the shallow decision boundaries shared by the source and target models. However, as the iterations deepen, the gradient similarities of the four algorithms all experience significant degradation. This is because, in the middle and late stages, the perturbations become deeply trapped in the high-frequency Non-robust Features specific to the source model, causing their update directions to gradually deviate from the target model’s true loss descent path, triggering feature-level “source-model overfitting.” Around $K \ = \ 1 1$ constrained by the hard boundary of the $L _ { \infty }$ norm truncation and the saturation effect of momentum accumulation, the exploration space of the adversarial perturbations peaks, and all four evolution curves gradually enter a dynamic plateau phase.

Throughout the entire plateau phase, the gradient cosine similarity of VMI-FGSM almost consistently overrides that of SSTA. However, cross-validating this with the terminal Black-Box Attack Success Rate (BB-ASR) at � = 20 (shown in the bottom right corner of Fig. 9), this experiment reveals a profound “Optimization Paradox”: although VMI-FGSM achieves optimal mathematical gradient alignment in the surrogate space, SSTA comprehensively surpasses it in actual blackbox transfer rates. Specifically, on the ResNet18 → DenseNet link, SSTA significantly leads VMI-FGSM with an absolute success rate of 96% to 78%; while on the most difficult link crossing to the global self-attention architecture (ResNet18 → Conformer), SSTA firmly holds a tactical deterrence baseline of 59%, far exceeding VMI-FGSM’s 42%.

![](images/e583eef54bbccdb1f7294daa4098000b6b587e64b6dc8e4fefb6b6829d778ed7.jpg)  
Fig. 9. Dynamic evolution trajectories of cross-model gradient cosine similarity during the iteration process.

Analyzing this anomalous phenomenon from the underlying mechanisms of optimization theory and the Loss Landscape precisely reveals the physical root of SSTA’s high transfer generalization capability. The variance tuning of VMI-FGSM is entirely confined to the static digital domain; its extremely high similarity means that the adversarial perturbations, at specific “static input coordinates,” precisely fall into the steep Sharp Minima basins where the source and target models coincidentally overlap. This stringent mathematical alignment severely lacks generalization tolerance and is extremely fragile when facing the architectural deformations of heterogeneous networks. In contrast, SSTA embeds actual time delays and amplitude-phase physical distortions as joint operators into the forward propagation, and its momentum gradient is the Expectation over Transformation (EoT) calculated under complex physical transformation distributions. This mechanism is equivalent to applying an Implicit Gradient Regularization based on physical priors to the loss extreme surface of the source model. Therefore, SSTA strategically sacrifices a portion of the “static absolute alignment precision” in exchange for the optimization trajectory evolving toward Flat Minima, which are smoother, broader, and possess exceptional structural robustness. It is exactly this smoothed solution, found under physical manifold constraints, that dynamically guarantees the extreme survivability of adversarial perturbations when penetrating unknown orthogonal defense architectures from the ground up.

## E. Physical Robustness and Effectiveness in EW Environments

In the aforementioned digital-domain evaluations, the SSTA algorithm has demonstrated outstanding generalization capability across heterogeneous cognitive models. However, in highly dynamic, real-world radar electronic warfare environments, a purely digital-domain adversarial attack does not constitute a complete tactical closed loop. Reconnaissance and jamming signals inevitably experience receiver clock drift, electromagnetic channel fading, and hard-core coherent processing at the radar front-end. To comprehensively verify the practical deployment potential of adversarial examples, this section shifts the evaluation perspective from tensor space back to physical electromagnetic space, deeply demonstrating the immunity and robustness of the SSTA algorithm against physical distortions and its tactical equivalence in the radar coherent processing chain.

1) Effective ASR under Physical Environment Distortions: Traditional Attack Success Rate (ASR) cannot distinguish whether a model’s prediction error is caused by the adversarial perturbation or solely by the physical environment distortion (e.g., severe noise blinding). To precisely quantify the true survival rate of adversarial perturbations under physical distortions, the Effective ASR metric is introduced. This metric utilizes a masking mechanism to calculate the subset of clean samples that the model still correctly predicts after noise injection at a specific physical distortion strength, and statistically evaluates the successful deception proportion of noisy adversarial samples only within this valid subset. Its rigorous mathematical definition is as follows:

$$
{ \mathrm { E f f e c t i v e \_ A S R } } = { \frac { \sum \mathbb { I } \left( F ( \mathbf { x } _ { \mathrm { c l e a n } } + \mathbf { n } ) = y \wedge F ( \mathbf { x } _ { \mathrm { a d v } } + \mathbf { n } ) \neq y \right) } { \sum \mathbb { I } \left( F ( \mathbf { x } _ { \mathrm { c l e a n } } + \mathbf { n } ) = y \right) } }\tag{20}
$$

where $\mathbf { X } _ { \mathrm { c l e a n } }$ is the clean baseband signal, $\mathbf { X } _ { \mathrm { a d v } }$ is the adversarial example, n denotes the specific injected physical environment distortion, � is the ground-truth label, �(·) represents the target cognitive model, and I(·) is the indicator function.

Based on this evaluation framework, this paper tested the robustness degradation curves of the core baselines and the SSTA algorithm on a unified source model (ResNet18) under three extreme physical distortions: clock drift (Δ�), static phase rotation (Δ�), and Additive White Gaussian Noise channels (AWGN). Regarding the experimental setup, this phase first strictly screened and generated 256 test bases with complete adversarial features in the digital domain. Subsequently, in the physical distortion simulation, the time offset range was set to [−30, 30] sampling points, the phase rotation range to [−�, �] radians, and the Jamming-to-Noise Ratio (JNR) degradation range from 35 dB to 5 dB. Each distortion dimension was uniformly divided into 31 fine-grained testing nodes to completely capture the performance evolution trajectory.

(c) Robustness to Channel Noise (AWGN)  
![](images/03756f63052dcc7c357df96db9fd7afbe27e9ceeef3ae073e4473ad25747745b.jpg)  
(a) Robustness to Clock Drift / Jitter

![](images/714056692a93f2ae0bd0538a7a05dc4295a20d53783222b92a4cefd86ef863ec.jpg)

![](images/01b6d62eb7d15876e5f8215c8914c6a566b68381a9253d82fa4e2779c3d16428.jpg)  
Fig. 10. Effective ASR degradation curves under extreme physical distortions: clock drift, phase rotation, and AWGN channels.

In the dynamic offset evaluation of the time and phase domains, constrained by the limitations of pure digital-domain optimization, the adversarial features of traditional algorithms are highly sensitive to absolute spatio-temporal coordinates. As shown in the distortion degradation curves in Fig. 10, when time shifts ranging from [−30, 30] sampling points or phase rotations of [−�, �] are introduced, the Effective ASR of the baseline algorithms exhibits severe oscillations and cliff-like degradation, dropping to approximately 72% in some intervals. Conversely, SSTA’s evolution trajectory presents a perfect horizontal straight line across the entire spatio-temporal distortion interval, with its Effective ASR consistently anchored firmly at 100%. This proves from the underlying mechanism that the spatio-temporal shift operators introduced based on Expectation over Transformation (EoT) successfully endow the adversarial perturbations with extremely strong translation and phase rotation invariance.

In the more stringent AWGN channel evaluation, as the JNR decreases, the target model’s recognition accuracy for clean samples (i.e., the theoretical upper-bound trajectory) inevitably declines accordingly. Observing the quantitative curves reveals that the perturbation energy generated by baseline algorithms rapidly collapses under the coverage of electromagnetic noise, and the decay rate of their effective success rate is much faster than the drop in the clean model’s accuracy. In contrast, SSTA’s performance curve presents a highly tactically valuable “upper-bound adherence” phenomenon: its evolution trajectory tightly and completely adheres to the theoretical upper-bound trajectory of the clean model’s accuracy down to extremely low SNR intervals. This highly overlapping decay dynamic characteristic declares that until the physical channel noise completely blinds the defense model, SSTA’s adversarial features will always maintain outstanding structural survivability.

2) Tactical Equivalence Validation in Coherent Processing Chains: Beyond resisting channel fading, adversarial perturbations must pass through the enemy radar’s baseband coherent processing (such as matched filtering) to ensure the target generation capability of the original active deception jamming, that is, to maintain “tactical equivalence.” This paper takes typical Dense False Target Jamming (DFTJ) as an example to conduct high-fidelity demonstration of physical waveforms from the core dimensions of joint time-frequency distribution and pulse compression.

![](images/dbd96e929215edde58872e307ced24553f646ac25e4f15ba9345f60ea00b2ea4.jpg)  
Fig. 11. Normalized time-domain envelope comparison between the clean DFTJ signal and the SSTA adversarial example.

![](images/9eb5cbe61d3b48de1c5f84598189a0a12d38c95fffbd64fb17e612b19cc043a0.jpg)

![](images/1e028bf9f0d4ffec8b5f208f184a1e1c5c5b2fdf98e37004ce1a45cacbed31e1.jpg)  
Fig. 12. STFT spectrogram comparison: (a) Clean DFTJ signal; (b) SSTA adversarial example.

In the time-domain envelope (Fig. 11) and Short-Time Fourier Transform (STFT) spectrogram (Fig. 12), benefiting from the strict minimal perturbation constraint $( \epsilon = 0 . 0 5 )$ , the physical distortions injected by SSTA are precisely limited within a minimal energy tolerance range. The adversarial examples do not cause macroscopic destruction to the temporal duty cycle of the original pulses, and the core false target energy ridges perfectly maintain the original linear frequency modulation slopes and delay intervals. The adversarial energy primarily diffuses into the background spectrum in a form similar to pseudo-random noise. This successfully misleads the cognitive network at the feature extraction level while presenting extremely high physical transparency to traditional radar time-frequency reconnaissance algorithms.

![](images/51bd3f3d6f34f7917a88881b45a3e558999099f33273b79355430d6171f98c55.jpg)  
Fig. 13. One-dimensional range profile comparison after matched filtering (pulse compression).

In the pulse compression stage, which determines the radar’s high-resolution range extraction, the one-dimensional Range Profile quantification metrics after matched filtering provide decisive physical evidence. Measured signal processing data reveals that the Peak Attenuation of the SSTA adversarial example is an absolute 0.000 dB, meaning that SSTA induces zero loss to the core main-lobe energy of the original deception waveform. Furthermore, the Noise Floor Lift after matched filtering is marginally increased by only +5.812 dB. In practical radar system detection theory, this minor noise floor elevation is completely within the system tolerance closed-loop of the Constant False Alarm Rate (CFAR) detection mechanism; it will by no means mask the main-lobe energy of the dense false targets, nor will it trigger unreasonable false alarm thresholds. This dual characteristic of “strong digital-domain blinding and physical-domain zero attenuation” fundamentally establishes the extremely high application value of the SSTA algorithm in real electromagnetic countermeasure scenarios.

## V. CONCLUSION AND FUTURE PERSPECTIVES

With the evolution of cognitive radar systems toward the end-to-end processing of one-dimensional complex baseband sequences, the underlying adversarial security vulnerabilities of deep networks have become increasingly prominent. Addressing the dual limitations of existing adversarial attack methods in cross-architecture transferability and physical receiving chain compatibility, this paper proposes a physical prior-driven Segmented Spatio-Temporal Attack (SSTA). By employing a segmentation mechanism based on global-local feature decoupling, SSTA effectively blocks the dependence of adversarial gradients on the long-range coherence of a single network, mitigating the problem of traditional adversarial perturbations overfitting to source models. In cross-evaluations involving 9 heterogeneous models spanning CNNs, RNNs, and

Transformers, the Average Black-Box Attack Success Rate (ABB-ASR) of SSTA is improved by over 10% compared to existing baseline algorithms, demonstrating the effectiveness of discrete local optimization in extracting highly generalizable adversarial features. Furthermore, by constructing spatio-temporal physical operators combined with the Expectation over Transformation (EoT) strategy, SSTA achieves an effective integration of adversarial iterations with actual channel distortions (delay and amplitude-phase fluctuations). Radar baseband equivalent simulation tests indicate that, after penetrating the radar matched filter, the generated adversarial waveforms cause no obvious attenuation to the main-lobe peak (0.000 dB), and the noise floor lift is strictly controlled within the Constant False Alarm Rate (CFAR) detection tolerance. This implies that while inducing high-confidence misclassifications in cognitive models, SSTA effectively preserves the coherent envelope structure of the original deception jamming, achieving the tactical unification of digital-domain deception and physical-domain fidelity.

Although the proposed SSTA framework effectively reveals the adversarial vulnerabilities faced by 1D radar baseband cognitive networks, certain limitations remain when transitioning to practical engineering deployment. On the one hand, the joint optimization mechanism based on expectation over transformation and multi-step momentum iterations introduces substantial computational overhead, making it difficult to fully satisfy the microsecond-level real-time response constraints in highly dynamic electronic warfare. On the other hand, the currently constructed spatio-temporal physical operators primarily focus on direct-path delays and basic amplitudephase distortions, lacking comprehensive coverage of higherorder environmental effects induced by high-speed platform maneuvers and complex terrains, such as Doppler dispersion and multipath fading. To address these shortcomings, future research will focus on two core dimensions: “lightweight realtime processing” and “physical closed-loop validation.” At the algorithmic level, we plan to explore forward inference networks based on deep generative architectures to replace the time-consuming iterative optimization process, aiming to achieve single-step real-time generation of adversarial waveforms and incorporate more comprehensive dynamic channel prior models. At the physical validation level, future work will be dedicated to deploying the adversarial generation framework onto real radio frequency hardware platforms, such as Software-Defined Radio (SDR), to strictly calibrate the RF survivability of adversarial examples in open Over-the-Air (OTA) channels. Additionally, extending the existing 1D temporal adversarial mechanisms to distributed cognitive radar networks, and investigating the blinding evolution laws of adversarial waveforms in multi-static collaborative detection and data fusion centers, will also serve as vital exploratory directions for deepening the security assessment of cognitive electronic warfare.

## REFERENCES

[1] Z. Wang, Z. Guo, G. Shu, and N. Li, “Radar jamming recognition: Models, methods, and prospects,” IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 18, pp. 3315– 3343, 2025.

[2] X. Yu, W. Wang, H. Wu, J. Bo, J. Zhang, and J. Meng, “Multi-domain feature fusion based radar deception jamming recognition method,” in 2023 IEEE 7th International Symposium on Electromagnetic Compatibility (ISEMC). Hangzhou, China: IEEE, Oct. 2023, pp. 1–4.

[3] T. Wang, Y. Cao, Z. Wu, Y. Xue, L. Zhang, and L. Yang, “Few-shot radar active deception jamming recognition: spatial-graph aggregated feature fusion based on transfer learning,” in IET International Radar Conference (IRC 2023), 2023.

[4] Z. Wu, J. Qian, M. Zhang, Y. Cao, T. Wang, and L. Yang, “Highconfidence sample augmentation based on label-guided denoising diffusion probabilistic model for active deception jamming recognition,” IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1–5, 2023.

[5] S. Wang, J. Du, W. Fan, and F. Zhou, “Ra-net: An effective radar jamming recognition method,” in IGARSS 2023 - 2023 IEEE International Geoscience and Remote Sensing Symposium. Pasadena, CA, USA: IEEE, Jul. 2023, pp. 6771–6774.

[6] C. Du, Y. Cong, L. Zhang, D. Guo, and S. Wei, “A practical deceptive jamming method based on vulnerable location awareness adversarial attack for radar hrrp target recognition,” IEEE Transactions on Information Forensics and Security, vol. 17, pp. 2410–2424, 2022.

[7] Z. Bao, S. Zhang, Y. Yang, S. Yang, and Y. Lin, “Mmof-agm: Multimodel adversarial attack for security assessment of neural networkbased radar signal recognition,” IEEE Transactions on Aerospace and Electronic Systems, vol. 62, pp. 7794–7806, 2026.

[8] J. Cui, J. Duan, W. Guo, C. Peng, and H. Li, “SAR-ESAE: Echo signalguided adversarial example generation method for synthetic aperture radar target detection,” Remote Sensing, vol. 17, no. 17, p. 3080, 2025.

[9] M. A. Richards, Fundamentals of Radar Signal Processing, 2nd ed. New York: McGraw-Hill Education, 2014.

[10] M. Sadeghi and E. G. Larsson, “Adversarial attacks on deep-learning based radio signal classification,” IEEE Wireless Communications Letters, vol. 8, no. 1, pp. 213–216, 2019.

[11] N. Papernot, P. McDaniel, and I. Goodfellow, “Transferability in machine learning: From phenomena to black-box attacks using adversarial samples,” arXiv preprint arXiv:1605.07277, 2016.

[12] Y. Liu, X. Chen, C. Liu, and D. Song, “Delving into transferable adversarial examples and black-box attacks,” in 5th International Conference on Learning Representations, ICLR 2017, 2017.

[13] X. Wang, Z. Ge, B. Liu, Z. Fang, F. Zhou, R. Zhang, S. Wang, and Y. Luo, “Devling into adversarial transferability on image classification: Review, benchmark, and evaluation,” arXiv preprint arXiv:2602.23117, 2026.

[14] Y. Dong, F. Liao, T. Pang, H. Su, J. Zhu, X. Hu, and J. Li, “Boosting adversarial attacks with momentum,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 9185–9193.

[15] J. Lin, C. Song, K. He, L. Wang, and J. E. Hopcroft, “Nesterov accelerated gradient and scale invariance for adversarial attacks,” in International Conference on Learning Representations (ICLR), 2020. [Online]. Available: https://openreview.net/forum?id=SJlHwkBYDH

[16] X. Wang and K. He, “Enhancing the transferability of adversarial attacks through variance tuning,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

[17] C. Xie, Z. Zhang, Y. Zhou, S. Bai, J. Wang, Z. Ren, and A. L. Yuille, “Improving transferability of adversarial examples with input diversity,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 2730–2739.

[18] Y. Dong, T. Pang, H. Su, and J. Zhu, “Evading defenses to transferable adversarial examples by translation-invariant attacks,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

[19] X. Wang, Z. Zhang, and J. Zhang, “Structure invariant transformation for better adversarial transferability,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[20] X. Wang, X. He, J. Wang, and K. He, “Admix: Enhancing the transferability of adversarial attacks,” in 18th IEEE/CVF International Conference on Computer Vision, ICCV 2021, 2021.

[21] A. Ganeshan, V. B. S, and V. B. Radhakrishnan, “Fda: Feature disruptive attack,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[22] Q. Huang, I. Katsman, Z. Gu, H. He, S. Belongie, and S.-N. Lim, “Enhancing adversarial example transferability with an intermediate level attack,” in 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 2019.

[23] Z. Wang, H. Guo, Z. Zhang, W. Liu, Z. Qin, and K. Ren, “Feature importance-aware transferable adversarial attacks,” in 18th IEEE/CVF International Conference on Computer Vision, ICCV 2021, 2021.

[24] M. Wang, J. Wang, B. Ma, and X. Luo, “Improving the transferability of adversarial examples through black-box feature attacks,” Neurocomputing, vol. 595, p. 127863, 2024.

[25] Z. Luo, Y. Cao, T.-S. Yeo, Y. Wang, and F. Wang, “Few-shot radar jamming recognition network via time-frequency self-attention and global knowledge distillation,” IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1–12, 2023.

[26] N. Carlini and D. Wagner, “Audio adversarial examples: Targeted attacks on speech-to-text,” in 2018 IEEE Security and Privacy Workshops (SPW), 2018.

[27] X. Yuan, Y. Chen, Y. Zhao, Y. Long, X. Liu, K. Chen, S. Zhang, H. Huang, X. Wang, and C. A. Gunter, “CommanderSong: A systematic approach for practical adversarial voice recognition,” in 27th USENIX Security Symposium (USENIX Security 18). USENIX Association, 2018, pp. 49–64.

[28] D. L. Adamy, EW 102: A Second Course in Electronic Warfare. Norwood, MA: Artech House, 2004.

[29] G. Zhao, Principles of Radar Countermeasures, 2nd ed. Xi’an, China: Xidian University Press, 2012, in Chinese.

[30] A. Athalye, L. Engstrom, A. Ilyas, and K. Kwok, “Synthesizing robust adversarial examples,” in International conference on machine learning. PMLR, 2018, pp. 284–293.

[31] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2016, pp. 770–778.

[32] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2017, pp. 4700–4708.

[33] M. Tan and Q. Le, “Efficientnet: Rethinking model scaling for convolutional neural networks,” in International conference on machine learning (ICML). PMLR, 2019, pp. 6105–6114.

[34] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR), 2018, pp. 7132–7141.

[35] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” Neural computation, vol. 9, no. 8, pp. 1735–1780, 1997.

[36] K. Cho, B. van Merriënboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder–decoder for statistical machine translation,” in Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2014, pp. 1724–1734.

[37] T. N. Sainath, O. Vinyals, A. Senior, and H. Sak, “Convolutional, long short-term memory, fully connected deep neural networks,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015, pp. 4580–4584.

[38] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” in Proceedings of the IEEE/CVF international conference on computer vision (ICCV), 2021, pp. 10 012–10 022.

[39] A. Gulati, J. Qin, C.-C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, and R. Pang, “Conformer: Convolutionaugmented Transformer for speech recognition,” in Interspeech 2020, 2020, pp. 5036–5040.

[40] I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial examples,” in 3rd International Conference on Learning Representations, ICLR 2015, 2015.