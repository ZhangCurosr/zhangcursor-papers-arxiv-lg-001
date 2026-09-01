# Liquid Gated Attention

Yiheng Jiang, Yuanbo Xu, and Yongjian Yang

Abstract—Real-world time series often exhibit irregular sampling and extended temporal horizons, requiring models to capture continuous-time dynamics across arbitrary time intervals without prohibitive scaling costs. Conventional discrete-time methods typically project observations onto uniformly spaced indices, collapsing variable time intervals into static positional steps and discarding temporal distances between observations. Conversely, solver-dependent continuous-time models preserve temporal structure but rely on sequential numerical integration, precluding parallelization. While solver-free approximations bypass this computational overhead, no parallel architecture explicitly couples observed time intervals with input-driven state modulation, limiting their ability to distinguish genuine temporal dynamics from observational noise. To bridge this gap, we propose Liquid Gated Attention (LGA), a solver-free parallel temporal operator. By parameterizing an input-driven gating mechanism with observed time intervals, LGA introduces a continuous-time inductive bias and formulates hidden state evolution as a fast-weight associative memory, enabling parallel computation across the temporal dimension. By leveraging matrix associativity in non-causal encoding and a prefix scan in causal encoding, LGA achieves linear temporal complexity with respect to sequence length in both modes. To stabilize long-horizon optimization, we employ a sequence-level normalization that bounds the cumulative temporal decay. Building on this operator, we instantiate LFormer, a modular backbone for continuous-time representation learning. Evaluations across six tasks and sixteen datasets, spanning up to 17,984 steps, demonstrate LFormer’s capabilities in long-range dependency modeling, fine-grained state tracking, and trajectory reconstruction from sparse and noisy observations. LFormer delivers competitive performance against state-of-the-art discrete-time and continuous-time baselines, alongside linear scaling efficiency confirmed by empirical benchmarks. These results illustrate how LGA embeds continuous-time dynamical principles into a parallel architecture, positioning LFormer as a scalable and versatile backbone for time series representation learning. Code is available at https: //anonymous.4open.science/r/Liquid-Gated-Attention-6B55.

Index Terms—Continuous-time Model, Linear Attention, Time Series Analysis, Numerical Stability

## I. INTRODUCTION

ONITORING real-world processes, such as physiological signaling [3], typically yields complex time series that exhibit irregular sampling and can span extended temporal horizons. These properties present significant challenges for temporal modeling. Specifically, irregular sampling requires models to characterize continuous-time dynamics across arbitrary observed time intervals [4], [5]. Simultaneously, extended sequence lengths demand capturing long-range dependencies without prohibitive computational costs [6], [7]. Furthermore, realistic observational constraints call for algorithms that extract the underlying continuous-time process from sparse and noisy measurements [8], [9].

Most existing approaches operate within the discrete-time paradigm that projects continuous temporal trajectories onto uniformly spaced indices [10]. Despite excelling at handling regularly sampled data, this paradigm collapses variable intervals into static positional steps and discards the temporal distances between observations, which are necessary for accurate continuous-time dynamics modeling [4], [5], [10]. Recurrent neural networks (RNNs) [11], [12] and Transformers [13] represent two dominant discrete-time models. Standard RNNs compress temporal dependencies into sequential hidden states. To accommodate irregular observations, they typically rely on heuristic preprocessing strategies, such as binning [14] and imputation [15], which inject spurious states into the timeline and distorts the underlying continuous dynamics [5]. Moreover, their recurrence precludes parallelization along the temporal dimension. Transformers eliminate this sequential constraint through parallel self-attention. Yet, even when augmented with explicit temporal encoding [16]–[18], they collapse time intervals into static similarity weights between discrete tokens, lacking a dedicated mathematical mechanism to model continuous-time dynamics between observations [10]. Furthermore, their quadratic complexity limits scalability to extended temporal horizons.

To preserve temporal structure, continuous-time approaches formulate the state evolution as a differential equation parameterized by a neural network. Neural ordinary differential equations pioneered this formulation [19]. Subsequent extensions, including neural controlled and rough differential equations, accommodate irregularly sampled time series through continuous path integration [20], [21]. Rooted in differential calculus, these methods explicitly incorporate temporal distances between observations [5] and can naturally smooth high-frequency noise [8]. However, their forward pass relies on numerical solvers for sequential integration. This solverdependent design introduces two bottlenecks. First, numerical solvers require multiple function evaluations per time step, and their sequential execution resists parallelization [22]. This computational overhead challenges these methods in largescale or real-time deployments. Second, truncation errors can accumulate over extended integration horizons [23], progressively degrading the fidelity of learned dynamics.

These bottlenecks motivate solver-free alternatives that retain continuous-time dynamics while bypassing iterative integration. To render continuous state evolution directly evaluable, three principal strategies have emerged: closed-form approximations [22], linear state-space formulations [24], and truncated algebraic path signatures [25]. Despite eliminating the solver-induced computational overhead, these methods commit continuous-time dynamics to fixed mathematical forms. These compromises introduce specific trade-offs documented in their respective literature, including potential gradient decay in closed-form recurrence [22], constrained expressivity for nonlinear dynamics in linear state transitions [6], and information loss from truncated signatures [25]. Furthermore, within this paradigm, no parallel architecture explicitly couples observed time intervals with input-driven state modulation. Without this linkage, they struggle to distinguish noise from genuine state transitions [5], [9], which limits their robustness under sparse and noisy conditions.

Across these three paradigms, a critical gap emerges: no single approach simultaneously incorporates observed time intervals, enables parallel computation, and maintains observational noise robustness. Discrete-time models discard the underlying temporal metric. Solver-dependent continuous-time methods retain temporal fidelity and noise robustness but sacrifice parallelism. Solver-free approaches recover parallel scalability yet lack an explicit temporal-state transition link, compromising robustness to sparse and noisy measurements.

In this work, we address this trilemma via Liquid Gated Attention (LGA), a solver-free temporal operator reconciling observed time intervals, parallel computation, and observational noise robustness across four design steps:

(1) Continuous-time Gating. LGA derives its gating mechanism from the closed-form structure of a one-dimensional liquid time-constant (1D LTC) equation [26]. The resulting liquid gate jointly encodes temporal decay and inputdriven modulation, providing a continuous-time inductive bias that attenuates high-frequency noise perturbations.

(2) Computational Efficiency. To avoid the computational overhead of numerical integration, we approximate the liquid gate via a learnable endpoint interpolation inspired by the trapezoidal rule. Fixing the interpolation weight at 0.5 recovers the classical trapezoidal rule with secondorder local accuracy, whereas the generalized learnable configuration serves as a trainable numerical surrogate. This step parallelizes the gating calculation while preserving the continuous-time inductive bias.

(3) Expressive Capability. To capture complex long-range dependencies, we elevate the scalar state in 1D LTCs to a matrix-valued associative memory. Adopting a fastweight programming framework [27], we translate the liquid gate into explicit modulation coefficients for attention features. This design preserves the continuoustime gating dynamics while yielding a parallel sequence operator that maintains linear temporal complexity with respect to sequence length in both non-causal and causal modes, the latter realized via a prefix scan.

(4) Numerical Safety. To stabilize long-horizon optimization, we introduce a sequence-level normalization that bounds the cumulative decay coefficients, preventing exponential underflow. This normalization operates as a regularized surrogate rather than an algebraically exact equivalent of the original LTC decay: it bounds the total decay budget while preserving the relative temporal distance allocation across the sequence.

Building upon the LGA operator, we instantiate LFormer, a modular backbone for continuous-time representation learning. LFormer constructs its deep temporal representations by interleaving decoupled multi-head LGA layers with channelmixing blocks, connected via residual connections with layer normalization. LFormer directly processes discrete observations, while inheriting the continuous-time inductive bias, noise robustness, linear temporal complexity, and training stability properties from the underlying LGA operator.

This work delivers three primary contributions:

• A parallel continuous-time temporal operator. LGA unifies the input-driven and time-interval-coupled liquid gate, a trapezoidal-rule-inspired numerical integral surrogate, fast-weight associative memory, and sequence-level normalization into a single solver-free formulation.

• A modular backbone for continuous-time representation learning. LFormer instantiates LGA in a decoupled multi-head form within a residual architecture. This design enables linear-complexity parallel sequence modeling directly over discrete observations, while inheriting LGA’s continuous-time inductive bias, noise robustness, and training stability properties.

• A comprehensive empirical validation across diverse temporal scenarios. Extensive evaluations across six distinct tasks and sixteen datasets demonstrate that LFormer achieves competitive performance compared to state-ofthe-art discrete and continuous baselines. These outcomes highlight its capabilities in long-range dependency modeling, fine-grained state tracking, and trajectory reconstruction from sparse and noisy observations, while maintaining linear scaling efficiency and empirical training stability over extended temporal horizons.

Ultimately, these results illustrate how LGA embeds continuous-time dynamical principles into a parallel architecture, positioning LFormer as a scalable and versatile backbone for time series representation learning.

## II. RELATED WORK

This section reviews prior literature along two axes. We first examine three temporal modeling paradigms distinguished by their mathematical treatment of time. We then discuss scalable parallelization mechanisms, focusing on fast-weight programmers and linear attention variants, which establish the computational foundation of the proposed architecture.

## A. Paradigms in Temporal Modeling

1) Discrete-time Models: This paradigm projects continuous temporal trajectories onto uniformly spaced discrete indices. Early approaches partition timeline into consecutive uniform bins [14], [28]–[30], and subsequently leverage imputation strategies [15], [31]–[33] to fill the resulting grids, yielding regularly sampled inputs on which standard architectures like RNNs [11], [12] and Transformers [13] achieve strong performance. However, these heuristic preprocessing strategies introduce pseudo-observations that can alter the underlying continuous dynamics [4], [34]. Recent methods shift this discrete-time paradigm to the sequence level by grouping adjacent time steps into rigid patches [35]–[39]. Although patching optimizes effective sequence lengths and mitigates the self-attention quadratic complexity, these rigid boundary partitions compromise fine-grained temporal structure. Alternatively, several approaches enrich token representations with explicit temporal encoding, including sinusoidal functions [16], [17], kernel-based methods [40], and learnable embeddings [18], [41]. Nevertheless, these encodings reduce time to an auxiliary feature attached to discrete tokens, rather than leveraging it as a continuous variable that could inform state transitions. Consequently, they offer limited capacity for capturing input-dependent continuous-time dynamics [10].

2) Solver-dependent Continuous-time Models: In contrast to discrete-time approaches, this paradigm parameterizes the derivative of continuous states via neural networks, solving the resulting differential equations through numerical methods [8]. This formulation affords a mathematically natural description of continuous-time dynamics. Pioneered by neural ordinary differential equations [19], this formulation underpins a broad lineage of architectures. Latent ODEs [5] extend this framework to generative modeling for irregularly sampled time series. Neural controlled differential equations [20] and neural rough differential equations [21] construct continuous input paths to drive continuous-time state evolution. Stochastic extensions [42] model uncertainty within these latent processes, while continuous-time attention variants [10], [43] integrate differential equations with the self-attention mechanism. Although mathematically principled, this solver-dependent paradigm incurs substantial computational overhead, particularly when integrating stiff equations [22]. Furthermore, sequential numerical integration inherently limits parallelization [44]. Moreover, truncation errors can accumulate over extended integration horizons [23], progressively degrading the fidelity of the learned continuous-time dynamics.

3) Solver-free Continuous-time Models: To circumvent the computational bottlenecks of numerical integration, this closely related paradigm transforms continuous-time dynamics into directly evaluable algebraic forms. Three principal strategies are prevalent. First, closed-form approximations solve specific differential equations analytically, as exemplified by closed-form continuous-time (CfC) models [22] for liquid time-constant networks [26]. Second, exact linear discretization rules from control theory govern state-space models [6], [24] and their stochastic variants [45]. Third, truncated algebraic path signatures from rough path theory encode continuous trajectories without invoking differential solvers [25]. Despite eliminating solver-induced overhead, these methods introduce specific compromises. As documented in their respective literature, these trade-offs include potential gradient decay in closed-form recurrences [22], constrained expressivity for highly nonlinear dynamics in linear transitions [6], and structural information loss due to signature truncation [25].

While each paradigm offers distinct advantages, none simultaneously satisfies the three requirements identified in Section I. This gap motivates a hybrid design that combines the parallel scalability of fast-weight attention with a continuoustime inductive bias derived from the LTC network—a synthesis absent from prior work and central to our proposed LGA.

## B. Parallelization via Fast Weights and Linear Attention

The self-attention mechanism underpinning Transformers [13] has become a dominant architecture in modern time series modeling [46]. While explicit pairwise interactions afford this mechanism exceptional expressive power, its quadratic complexity in sequence length precludes scaling to the extended horizons typical of complex dynamical systems.

To mitigate this bottleneck, linear attention [47] restructures the attention computation. By replacing the softmax operator with kernel feature maps, this formulation exploits matrix associativity to achieve linear complexity with respect to sequence length. Interpreting the attention state as a matrixvalued memory updated via outer products establishes a formal equivalence with fast-weight programmers [27], [48], providing a principled framework for parallelizing recurrent architectures. Subsequent advances have enriched this framework with data-dependent gating mechanisms, including forget gates in RetNet [49], gated linear attention [50], and the RWKV series [51]–[53]. Concurrently, state-space models such as Mamba [6], [54] and DeltaNet [55] achieve linear complexity through input-dependent transitions and hardware-aware associative scans. Despite their remarkable computational efficiency, these modern variants are predominantly designed for the discrete-time domain [56], where gating operators and state transitions are parameterized by integer step indices.

This historical focus leaves a key open question: can the parallelization principles of fast-weight attention be integrated with continuous-time, input-driven gating dynamics? Addressing this question requires embedding observed temporal distances into fast-weight memory, enabling the architecture to process irregularly sampled time series—a capability absent from existing parallel architectures. This requirement underpins the design of LGA.

## III. BACKGROUND: LIQUID TIME-CONSTANT GATING

Liquid time-constant networks [26] are continuous-time recurrent neural models. At time t, an LTC network determines the d-dimensional continuous state vector $\mathbf { s } ( t ) \in \mathbb { R } ^ { d }$ via the following initial value problem (IVP):

$$
\frac { \mathrm { d } \mathbf { s } ( t ) } { \mathrm { d } t } = - \Big [ \mathbf { w } _ { \tau } + f \big ( \mathbf { s } ( t ) , \mathbf { x } ( t ) \big ) \Big ] \odot \mathbf { s } ( t ) + \mathbf { b } _ { \tau } \odot f \big ( \mathbf { s } ( t ) , \mathbf { x } ( t ) \big ) ,\tag{1}
$$

where $\mathbf { x } ( t ) \in \mathbb { R } ^ { m }$ denotes the exogenous input to the system, and ${ \mathbf w } _ { \tau } , { \mathbf b } _ { \tau } \in \mathbb { R } ^ { d }$ represent the intrinsic inverse time-constant weight and bias, respectively. The function f(·) is a strictly positive, bounded, continuous, and monotonically increasing nonlinearity, while ⊙ denotes the element-wise product.

Discussion. The “liquid” property stems from the input-driven modulation of the system’s effective inverse time constants, defined as $\mathbf { w } _ { \tau } + f \big ( \mathbf { s } ( t ) , \mathbf { x } ( t ) \big )$ . This mechanism allows individual units to adjust their intrinsic time constants based on the incoming signal, enabling the network to adaptively characterize varying temporal dynamics.

To achieve analytical tractability, we first introduce the onedimensional LTC configuration which isolates a single scalar state unit $s ( t ) \in  { \mathbb { R } }$ by removing self-connections [22]. Driven by a scalar exogenous input $x ( t ) \in  { \mathbb { R } }$ , this continuous state follows a linear non-autonomous variant of Eq. (1):

$$
\frac { \mathrm { d } s ( t ) } { \mathrm { d } t } = - \big [ w _ { \tau } + f \big ( x ( t ) \big ) \big ] s ( t ) + b _ { \tau } f \big ( x ( t ) \big ) ,\tag{2}
$$

where $w _ { \tau } , b _ { \tau } \in \mathbb { R }$ are scalar parameters. To derive the solvable form, we follow the structural symmetry assumption established in [22], which introduces a balancing inverse timeconstant parameter into the bias compartment of Eq. (2):

$$
\frac { \mathrm { d } s ( t ) } { \mathrm { d } t } = - \big [ w _ { \tau } + f ( x ( t ) ) \big ] s ( t ) + \big [ w _ { \tau } + f ( x ( t ) ) \big ] b _ { \tau } .\tag{3}
$$

Under this symmetric condition, applying the linear ODE theory [57] yields the integral solution:

$$
s ( t ) = \left( s ( 0 ) - b _ { \tau } \right) \exp \left[ - w _ { \tau } t - \int _ { 0 } ^ { t } f \bigl ( x ( u ) \bigr ) \mathrm { d } u \right] + b _ { \tau } ,\tag{4}
$$

where $s ( 0 ) \in \mathbb { R }$ denotes the initial state boundary at $t = 0$ Section S1 in the Supplementary Material details the algebraic verification under this symmetric condition.

Let $t _ { n - 1 }$ and $t _ { n }$ denote two consecutive, irregularly spaced observation timestamps where $t _ { n - 1 } < t _ { n }$ . Defining the local temporal interval as $\delta _ { n } = t _ { n } - t _ { n - 1 }$ , evaluating the continuous integral solution in Eq. (4) across this boundary yields the following state transition:

$$
\frac { s ( t _ { n } ) - b _ { \tau } } { s ( t _ { n - 1 } ) - b _ { \tau } } = \exp \left[ - w _ { \tau } \delta _ { n } - \int _ { t _ { n - 1 } } ^ { t _ { n } } f \big ( \boldsymbol { x } ( u ) \big ) \mathrm { d } u \right] .\tag{5}
$$

Algebraic rearrangement of Eq. (5) produces a recurrent state update rule modulated by a continuous-time, input-driven gating mechanism:

$$
s ( t _ { n } ) = g ( t _ { n } ) s ( t _ { n - 1 } ) + \big [ 1 - g ( t _ { n } ) \big ] b _ { \tau } ,\tag{6}
$$

$$
g ( t _ { n } ) = \exp \left[ - w _ { \tau } \delta _ { n } - \int _ { t _ { n - 1 } } ^ { t _ { n } } f \big ( x ( u ) \big ) \mathrm { d } u \right] .\tag{7}
$$

Discussion. We refer to Eq. (7) as the liquid gate which structurally encapsulates two mathematical properties:

• Temporal Decay. The component $- w _ { \tau } \delta _ { n }$ characterizes the intrinsic continuous decay of the state trajectory over the observed temporal distance $\delta _ { n }$

• Input Adaptability. The integral $\begin{array} { r l } {  { - \int _ { t _ { n - 1 } } ^ { t _ { n } } f ( x ( u ) ) \mathrm { d } u } \quad } & { { } } \end{array}$ aggregates the cumulative modulation of exogenous inputs across the same localized time horizon.

The exponential operator maps these properties into a unified and bounded gate $g ( t _ { n } ) \in ( 0 , 1 )$ . Derived from LTC dynamics, this formulation endows the liquid gate with continuous-time inductive bias and noise-attenuation properties. This closedform relation provides the continuous-time, input-driven gating mechanism that underpins LGA Step (1).

## IV. METHODOLOGY: LIQUID GATED ATTENTION

In this section, we formulate Liquid Gated Attention, a solver-free parallel operator for continuous-time representation learning. Building upon the continuous-time inductive bias established in Section III, the subsequent derivation proceeds through three conceptual layers: (1) a learnable endpoint interpolation that approximates the liquid gate while circumventing numerical integration; (2) an expansion of the scalar state to a matrix-valued associative memory, enabling expressive sequence modeling; and (3) a sequence-level normalization that stabilizes optimization over extended sequences. Consequently, LGA functions as a continuous-time-inspired parallel operator rather than a numerically exact solver translating from the original LTC dynamics.

## A. Deriving the Recurrent Form of LGA

The RNN-like hidden state evolution in the 1D LTC system exposes two limitations: the computational overhead of the integral-based liquid gate and the limited capacity of the scalar hidden state. The following subsections elaborate on the solutions tailored to these two issues, respectively.

1) A Learnable Integral Approximation: We approximate the integral-based liquid gate in Eq. (7) via a learnable endpoint interpolation inspired by the trapezoidal rule. Given two consecutive input representations $\mathbf { x } _ { n - 1 } , \mathbf { x } _ { n } \in \mathbb { R } ^ { 1 \times d }$ , a shared nonlinearity first projects each token into a scalar variable:

$$
\begin{array} { r } { \tilde { x } _ { n - 1 } = \sigma \left( \mathbf { x } _ { n - 1 } \mathbf { w } _ { f } + b _ { f } \right) , } \\ { \tilde { x } _ { n } = \sigma \left( \mathbf { x } _ { n } \mathbf { w } _ { f } + b _ { f } \right) , ~ } \end{array}\tag{8}
$$

where $\mathbf { w } _ { f } \in \mathbb { R } ^ { d \times 1 }$ and $b _ { f } \in \mathbb { R }$ are learnable parameters, and $\sigma ( \cdot )$ denotes the sigmoid activation function.

We then approximate the integral via a learnable weighted average across the boundary endpoints:

$$
\int _ { t _ { n - 1 } } ^ { t _ { n } } \tilde { x } ( u ) \mathrm { d } u \approx \delta _ { n } \underbrace { [ \mu \tilde { x } _ { n - 1 } + ( 1 - \mu ) \tilde { x } _ { n } ] } _ { \mathrm { d e f i n e d ~ a s ~ } \tilde { x } _ { n } } = \delta _ { n } \overline { { x } } _ { n } ,\tag{9}
$$

where $\delta _ { n } = t _ { n } - t _ { n - 1 }$ denotes the time interval. We parameterize the interpolation coefficient as $\mu = \sigma ( \theta ) \in ( 0 , 1 )$ with learnable $\theta \in$ R to balance each endpoint’s contribution. When $\mu = 0 . 5$ , this interpolation recovers the classical trapezoidal rule with second-order local accuracy. In the general learnable case, it serves as a trainable numerical surrogate rather than a fixed quadrature rule. Section S2 of the Supplementary Material provides the proof.

For implementation simplicity, we merge the intrinsic inverse time-constant $w _ { \tau }$ and the input-dependent term into a single decay surrogate, simplifying Eq. (7) to the following efficient liquid gate:

$$
g _ { n } = \exp \left[ - \delta _ { n } \overline { { x } } _ { n } \right] .\tag{10}
$$

This formulation preserves the temporal decay and input adaptability functionalities of LTC while eliminating the need for numerical integration. This completes LGA Step (2) Computational Efficiency.

2) An Expressive Hidden State: We elevate the hidden state from a scalar to a matrix $\mathbf { S } _ { n } \in \mathbb { R } ^ { d \times d }$ that expands the memory storage capacity for long-range modeling. Furthermore, a dynamic, input-dependent associative memory term substitutes the static bias $b _ { \tau }$ in Eq. (6), improving model flexibility in tracking instantaneous feature variations. Following the linear attention framework [47], the input vector ${ \bf x } _ { n }$ is projected into key and value vectors $\mathbf { k } _ { n } , \mathbf { v } _ { n } \in \mathbb { R } ^ { 1 \times d } \mathrm { : }$

$$
\begin{array} { r l } & { \mathbf { k } _ { n } = \mathbf { x } _ { n } \mathbf { W } _ { k } / \sqrt { d } + \mathbf { b } _ { k } , } \\ & { \mathbf { v } _ { n } = \mathbf { x } _ { n } \mathbf { W } _ { v } + \mathbf { b } _ { v } , } \end{array}\tag{11}
$$

![](images/4d4ea9447905755cf5c117ac5cf261ea254478f36d922160f635b19684fe6a4f.jpg)  
Fig. 1: LGA hidden state update mechanics. The upper block depicts the recurrent state transition where the liquid gates modulate the combination of historical memory and associative update matrix. The lower pipeline details the liquid gate evaluation workflow incorporating nonlinear projection, learnable interpolation, temporal scaling, and exponential mapping.

where $\mathbf { W } _ { k } , \mathbf { W } _ { v } \ \in \ \mathbb { R } ^ { d \times d }$ and ${ \bf b } _ { k } , { \bf b } _ { v } \in \mathbb { R } ^ { 1 \times d }$ are learnable parameters, and $\sqrt { d }$ represents the scaling factor. As visualized in Fig. 1, the recurrent hidden state update rule in LGA is:

$$
\mathbf { S } _ { n } = g _ { n } \mathbf { S } _ { n - 1 } + ( 1 - g _ { n } ) \mathbf { k } _ { n } ^ { \top } \mathbf { v } _ { n } .\tag{12}
$$

Discussion. Eq. (12) implements the fast-weight programming scheme [27], where $\mathbf { S } _ { n }$ acts as an associative memory updated via vector outer products. This elevation expands the representational capacity into a d×d interaction matrix [52], [54], [58], fulfilling LGA Step (3) Expressive Capability.

The output retrieval in LGA handles the feature extraction and state readout from associative memory matrices. At time $t _ { n } .$ , the input representation ${ \bf x } _ { n }$ is mapped to the query vector $\mathbf { q } _ { n } \in \mathbb { R } ^ { 1 \times d }$ and the output gate $\mathbf { o } _ { n } \in \mathbb { R } ^ { 1 \times d }$ via:

$$
\mathbf { q } _ { n } = \mathbf { x } _ { n } \mathbf { W } _ { q } + \mathbf { b } _ { q } ,\tag{13}
$$

$$
\mathbf { o } _ { n } = \sigma \left( \mathbf { x } _ { n } \mathbf { W } _ { o } + \mathbf { b } _ { o } \right) ,\tag{14}
$$

where $\mathbf { W } _ { q } , \mathbf { W } _ { o } \in \mathbb { R } ^ { d \times d }$ and $\mathbf { b } _ { q } , \mathbf { b } _ { o } \in \mathbb { R } ^ { 1 \times d }$ are learnable weights and biases, respectively. The output gate then modulates the retrieved context within the hidden state matrix $\mathbf { S } _ { n }$ to produce the final output vector ${ \bf y } _ { n } \in \mathbb { R } ^ { 1 \times d }$ via:

$$
\mathbf { y } _ { n } = \mathbf { o } _ { n } \odot \left( \mathbf { q } _ { n } \mathbf { S } _ { n } \right) .\tag{15}
$$

The output gate allows the model to suppress irrelevant or noisy retrieved channels, improving robustness against highfrequency observational perturbations.

## B. Deriving the Parallel Former of LGA

Assuming a zero-initialized hidden state $\mathbf { S } _ { 0 } = \mathbf { 0 }$ , unrolling the recurrent update rule in Eq. (12) across the temporal horizon yields:

$$
\mathbf { S } _ { n } = \sum _ { i = 1 } ^ { n } \left[ \left( \prod _ { j = i + 1 } ^ { n } g _ { j } \right) \left( 1 - g _ { i } \right) \mathbf { k } _ { i } \right] ^ { \top } \mathbf { v } _ { i } .\tag{16}
$$

Substituting Eq. (16) into the output retrieval expression in Eq. (15) and factoring out the n-dependent terms from the temporal summation yields:

$$
{ \bf y } _ { n } = { \bf 0 } _ { n } \odot \left[ \left( \dot { g } _ { n } { \bf q } _ { n } \right) \sum _ { i = 1 } ^ { n } \left[ \left( 1 - g _ { i } \right) { \bf k } _ { i } \right] ^ { \top } \left( \frac { { \bf v } _ { i } } { \dot { g } _ { i } } \right) \right] ,\tag{17}
$$

where $\begin{array} { r } { \dot { g } _ { n } \ = \ \prod _ { i = 1 } ^ { n } g _ { i } \ \in \ \mathbb { R } } \end{array}$ represents the cumulative liquid gate product. Under zero initialization, Eq. (17) remains algebraically equivalent to the recurrent updates in Eq. (12). Temporal index alignment enables these liquid gates to directly scale token representations, transforming the continuous-time inductive bias into explicit modulation coefficients $g _ { n }$ and ${ \dot { g } } _ { n }$ for the query, key, and value vectors.

Building upon Eq. (17), exploiting matrix associativity within the fast-weight programming framework [27] can eliminate sequential dependencies in the recurrent state. This yields two linearly scaling parallel modes for causal and noncausal computation, depending on the temporal aggregation boundaries. Given the modulated features $\bar { \mathbf { Q } } , \dot { \mathbf { K } } , \bar { \mathbf { V } } \in \mathbb { R } ^ { n \times d }$ and the element-wise output gate $\textbf { O } \in \mathbb { R } ^ { n \times d }$ , the parallel matrix transformations in LGA that map these inputs to the output matrix $\mathbf { Y } \in \mathbb { R } ^ { n \times d }$ are instantiated as:

• Causal Mode. Online streaming and autoregressive generation tasks strictly prohibit the future information leakage. Leveraging a parallel prefix scan, the causal LGA operator is formulated as:

$$
\mathbf { Y } = \mathbf { O } \odot \left[ \dot { \mathbf { Q } } \circledast \mathrm { C u m S u m } \left( \dot { \mathbf { K } } \otimes \dot { \mathbf { V } } \right) \right] ,\tag{18}
$$

where $\otimes$ and ⊛ denote the row-wise outer product and vector-matrix multiplication, respectively. CumSum(·) computes the associative cumulative sum along the temporal dimension. This implementation maintains causal semantics while reducing both computational and memory complexity to $\mathcal { O } ( n d ^ { 2 } )$ .

• Non-causal Mode. For offline representation learning tasks where the strict causality is unnecessary, the temporal aggregation directly encompasses the full sequence. Replacing the prefix scan with a global temporal summation, LGA functions as a bidirectional encoder that

$$
\mathbf { Y } = \mathbf { O } \odot \left[ \dot { \mathbf { Q } } \left( \dot { \mathbf { K } } ^ { \top } \dot { \mathbf { V } } \right) \right] ,\tag{19}
$$

which prioritizes the matrix product between the key and value matrices to preserve the linear complexity $\mathcal { O } \left( n d ^ { 2 } \right)$

We summarize Eq. (18) and Eq. (19) into a unified form:

$$
\begin{array} { r } { \mathbf { Y } = \operatorname { L G A } ( \mathbf { X } , \pmb { \Delta } ) , } \end{array}\tag{20}
$$

since liquid gates, modulated features and output gate are calculated based on the input sequence representation matrix $\mathbf { X } \in \mathbb { R } ^ { n \times d }$ and the corresponding time interval vector $\pmb { \Delta } \in \mathbb { R } ^ { n \times 1 }$ . Figure 2 visualizes the LGA pipeline.

The base features Q, K, $\mathbf { V } \in \mathbb { R } ^ { n \times d }$ and the output gate O are obtained via linear projections of the input matrix X:

$$
\mathbf { Q } = \mathbf { X } \mathbf { W } _ { q } + \mathbf { b } _ { q } ,\tag{21}
$$

$$
\mathbf { K } = \mathbf { X } \mathbf { W } _ { k } / \sqrt { d } + \mathbf { b } _ { k } ,\tag{22}
$$

$$
\mathbf { V } = \mathbf { X } \mathbf { W } _ { v } + \mathbf { b } _ { v } ,\tag{23}
$$

$$
\mathbf { O } = \sigma \left( \mathbf { X } \mathbf { W } _ { o } + \mathbf { b } _ { o } \right) .\tag{24}
$$

![](images/69db9fe8371c3117353bcbdb738024b954892374467842bbacf6f362704ff311.jpg)  
Fig. 2: Parallel LGA pipeline. The input representation X is first projected and scaled by vectorized liquid gates to embed continuous-time gating dynamics, yielding modulated representations. The dual mode attention block then executes either a causal prefix scan or a global matrix multiplication with linear complexity. The output gate modulates the aggregated state to produce the final representation. ⊙, ⊛, and ⊗ denote the element-wise product, row-wise vectormatrix multiplication, and row-wise matrix outer product, respectively. CumSum(·) computes the cumulative sum along the temporal dimension.

The modulated features $\dot { \bf Q } , \dot { \bf K } , \dot { \bf V } \in \mathbb R ^ { n \times d }$ are obtained by scaling with the vectorized liquid gates $\mathbf { g } , \dot { \mathbf { g } } \in \mathbb { R } ^ { n \times 1 }$ :

$$
\begin{array} { r } { \dot { \bf Q } = { \bf Q } \odot \left( \dot { \bf g } { \bf 1 } _ { d } ^ { \top } \right) , } \end{array}\tag{25}
$$

$$
\begin{array} { r } { \dot { \mathbf { K } } = \mathbf { K } \odot \left[ \left( 1 - \mathbf { g } \right) \mathbf { 1 } _ { d } ^ { \top } \right] , } \end{array}\tag{26}
$$

$$
\begin{array} { r } { \dot { \bf V } = { \bf V } \odot \left( \dot { \bf g } ^ { - 1 } { \bf 1 } _ { d } ^ { \top } \right) . } \end{array}\tag{27}
$$

where the scalar 1 performs element-wise broadcasting and the vector $\mathbf { 1 } _ { d } ^ { \top } \in \mathbb { R } ^ { 1 \times d }$ denotes the channel-wise broadcasting.

The parallel computation of the liquid gate vector g is performed by applying the operations in Eqs. (8)—(10) simultaneously across the temporal dimension, as formalized in Algorithm 1. To evaluate the cumulative liquid gate ${ \dot { \mathbf { g } } } ,$ the sequential multiplication chain is mapped into log-space which reformulates the recurrent dependency into the cumulative summation, yielding

$$
\dot { \mathbf { g } } = \exp \left[ - \mathrm { C u m S u m } \left( \pmb { \Delta } \odot \overline { { \mathbf { x } } } \right) \right]\tag{28}
$$

where $\overline { { \mathbf { x } } } \in \mathbb { R } ^ { n \times 1 }$ represents the interpolated nonlinear vector.

1) Sequence-level Normalization: Due to the cumulative term g˙ , the vanilla parallel execution is prone to numerical instability over long sequences. Since the inputs are nonnegative, the unconstrained prefix sum in Eq. (28) grows monotonically with the sequence length, leading to exponential underflow and vanishing gradients.

To stabilize long-horizon optimization, we introduce a sequence-level normalization mechanism. Let $\mathbf { u } = \Delta \odot \overline { { \mathbf { x } } } \in$ $\mathbb { R } ^ { n \times 1 }$ . The scaling vector u is normalized by its summation along the sequence dimension:

Algorithm 1 Parallel Computation of Liquid Gates under   
Vanilla and Stabilized Modes   
Input: Input matrix $\mathbf { X } \in \mathbb { R } ^ { n \times d } .$ , time interval vector $\Delta \ \in$   
$\mathbb { R } ^ { n \times 1 }$ , weight vector $\mathbf { w } _ { f } \in \mathbb { R } ^ { d \times 1 }$ , bias scalar $b _ { f } \in \mathbb { R }$   
unconstrained parameter $\theta \in \mathbb { R } ,$ , stability constant $\epsilon > 0 ,$   
mode selector $\mathcal { M } \in \mathcal { E }$ {Vanilla, Stabilized}.   
Output: Local liquid gate vector $\mathbf { g } \in \mathbb { R } ^ { n \times \mathrm { 1 } }$ and cumulative   
liquid gate vector $\dot { \bf g } \in \mathbb { R } ^ { n \times 1 }$   
1: $\mu  \sigma ( \theta )$ ▷ Bounded interpolation coefficient   
2: x˜ $ \sigma ( \mathbf { X } \mathbf { w } _ { f } + b _ { f } )$ ▷ Nonlinear projection   
3: $\tilde { \mathbf { x } } _ { \mathrm { p r e v } } \gets [ 0 ; \tilde { x } _ { 1 } ; \dots ; \tilde { x } _ { n - 1 } ]$ ▷ Temporal shift   
4: x $ \mu \tilde { \mathbf { x } } _ { \mathrm { p r e v } } + ( 1 - \mu ) \tilde { \mathbf { x } }$ ▷ Interpolation   
5: u $ \Delta \odot \overline { { \mathbf { x } } }$ ▷ Base decay vector   
6: if M = Stabilized then   
7: $\hat { \mathbf { u } } \gets \mathbf { u } / \left( \sum _ { i = 1 } ^ { n } u _ { i } + \epsilon \right)$ ▷ Sequence-level normalization   
8: $\mathbf { g } \gets \exp ( - \hat { \mathbf { u } } )$ ▷ Stabilized local gate   
9: $\dot { \mathbf { g } } \gets \mathrm { e x p } ( - \mathrm { C u m S u m } ( \hat { \mathbf { u } } ) )$ ▷ Stabilized cumulative gate   
10: else   
11: $\mathbf { g }  \exp ( - \mathbf { u } )$ ▷ Vanilla local gate   
12: $\dot { \mathbf { g } } \gets \mathrm { e x p } ( - \mathrm { C u m S u m } ( \mathbf { u } ) )$ ▷ Vanilla cumulative gate   
13: end if   
14: return g, g˙

$$
\hat { \mathbf { u } } = \frac { \mathbf { u } } { \sum _ { i = 1 } ^ { n } u _ { i } + \epsilon } ,\tag{29}
$$

where $u _ { i } = \delta _ { i } { \overline { { x } } } _ { i }$ is the i-th element of u, and ϵ is a small constant added to prevent division by zero. Substituting u with uˆ bounds the cumulative sum within (0, 1]. Consequently, the exponential scaling coefficients remain constrained to $( e ^ { - 1 } , 1 ]$ as proven in Section S3 of the Supplementary Material. Thus, the stable local and cumulative liquid gates are formulated as:

$$
\mathbf { g } _ { \mathrm { s t a b l e } } = \exp \left( - \hat { \mathbf { u } } \right) ,\tag{30}
$$

$$
\dot { \bf g } _ { \mathrm { s t a b l e } } = \exp \left[ - \mathrm { C u m S u m } \left( \hat { \bf u } \right) \right] .\tag{31}
$$

Algorithm 1 formalizes the complete parallel evaluation flow of these liquid gate vectors, including the alternative numerical stabilization path.

Discussion. The normalized gate is not algebraically equivalent to the original LTC state transition. Instead, it operates as a sequence-level regularized surrogate that preserves relative temporal distance allocation while bounding the total decay budget. Moreover, this mechanism reduces the gradient decay from exponential to polynomial as the sequence length grows (see Section S3 of the Supplementary Material). This enables stable optimization over extended sequences, fulfilling Step (4) Numerical Safety. This design targets offline full-sequence representation learning; online streaming or causal inference requires a prefix-normalized variant to eliminate dependence on future observations.

2) A Decoupled Multi-Head Extension: To scale the continuous-time gating dynamics to high-dimensional spaces, we derive a decoupled multi-head LGA mechanism.

Specifically, the d-dimensional latent space is partitioned into H distinct subspaces, where each head $h \in \{ 1 , \ldots , H \}$ operates on a subspace of dimension $d _ { h } = d / H . \operatorname { L e t } \mathbf { W } ^ { ( h ) }$ $\bar { \mathbf { b } ^ { ( h ) } }$ , and $\mu ^ { ( h ) }$ denote the head-specific learnable projection weights, biases, and adaptive interpolation coefficients for the h-th subspace, respectively. Each partitioned head executes the linear projection, adaptive interpolation, and sequencelevel normalization procedures independently, yielding the decoupled subspace representations:

![](images/79d8bdc723a9b53b8f7385482e1ed51238bff775f8f75424110ff3f9663805b5.jpg)  
Fig. 3: LFormer architecture. The left block depicts the macro design, where L stacked Liquid Mixers connect a task-specific Embedder and Predictor. The right panels detail the MHLGA (bottom) and SwiGLU (upper) workflows.

$$
\mathbf { Y } ^ { ( h ) } = \mathrm { L G A } ^ { ( h ) } \left( \mathbf { X } , \pmb { \Delta } \right) , \quad h \in \{ 1 , \dots , H \} .\tag{32}
$$

Subsequently, the outputs from all H heads are concatenated along the feature dimension and linearly projected to construct the final multi-head representation:

$$
\begin{array} { r } { \mathbf { Y } = \mathrm { M H L G A } \left( \mathbf { X } , \pmb { \Delta } \right) = \mathrm { C o n c a t } \left( \mathbf { Y } ^ { ( 1 ) } , \ldots , \mathbf { Y } ^ { ( H ) } \right) \mathbf { W } _ { m , \pmb { \Delta } } } \end{array}\tag{33}
$$

where Concat (·) denotes the concatenation operator and $\mathbf { W } _ { m } \in \mathbb { R } ^ { d \times d }$ represents the terminal projection weight matrix. Discussion. Unlike standard multi-head attention that primarily captures heterogeneous subspace features [13], the key architectural advancement lies in the head-wise decoupling of continuous-time gating dynamics. Equipping each head with independent interpolation parameters and liquid gates enhances the model’s expressive capability at the system level, enabling collaborative modeling of complex temporal patterns.

## C. LFormer: An Instantiation of LGA

Building upon the LGA operator, we establish a modular backbone for continuous-time representation learning system, designated as LFormer. As illustrated in Fig. 3, the architecture employs a three-stage processing pipeline consisting of a taskspecific Embedder, L stacked Liquid Mixers, and a taskoriented Predictor.

1) Embedder: The Embedder projects raw input observations into a d-dimensional representation space. To accommodate variable-length sequences, we define a canonical sequence length n. The longer sequences are partitioned into non-overlapping sub-sequences of length n, while shorter sequences are extended to length n via a zero-padding scheme. The padding tokens are excluded from the gradient optimization. Formally, given a raw or pre-processed time series with m features $\mathbf { X } _ { \mathrm { r a w } } \in \mathbb { R } ^ { n \times m }$ , the Embedder operates via:

$$
\mathbf { X } = \mathrm { E m b e d } \left( \mathbf { X } _ { \mathrm { r a w } } \right) ,\tag{34}
$$

where Embed $( \cdot ) : \mathbb { R } ^ { m }  \mathbb { R } ^ { d }$ denotes a task-specific transformation mapping raw observations to the latent sequence representation matrix $\mathbf { X } \in \mathbb { R } ^ { n \times d }$

2) Liquid Mixer: The Liquid Mixer serves as the core block of LFormer, designed to refine latent representations through interleaved temporal and channel-wise transformations. The framework stacks L identical mixer layers, where the l-th mixer $( l \in \{ 1 , \ldots , L \} )$ processes the representations generated by its predecessor or the initialization from the Embedder.

Specifically, each mixer layer consists of a multi-head liquid gated attention (MHLGA) module and a swish-gated linear unit (SwiGLU), organized in a residual connection structure with pre-layer normalization (LN):

$$
\mathbf { Y } = \mathbf { X } + \mathrm { M H L G A } \left( \mathrm { L N } \left( \mathbf { X } \right) , \pmb { \Delta } \right) ,\tag{35}
$$

$$
\mathbf { Z } = \mathbf { Y } + \operatorname { S w i G L U } \left( \operatorname { L N } \left( \mathbf { Y } \right) \right) ,\tag{36}
$$

where $\textbf { Z } \in \ \mathbb { R } ^ { n \times d }$ denotes the output of the current mixer layer. The MHLGA component operates along the temporal sequence dimension to inject continuous-time inductive bias via decoupled liquid gates, whereas the SwiGLU block modulates cross-feature channel interactions to enhance non-linear expressivity. For any hidden representation $\mathbf { H } \in \mathbb { R } ^ { n \times d }$ , the operation in SwiGLU is:

$$
\mathrm { S w i G L U } \left( \mathbf { H } \right) = \left[ \mathrm { S i L U } \left( \mathbf { H } \mathbf { W } _ { 1 } \right) \odot \left( \mathbf { H } \mathbf { W } _ { 2 } \right) \right] \mathbf { W } _ { 3 } ,\tag{37}
$$

where $\mathbf { W } _ { 1 } , \mathbf { W } _ { 2 } \ \in \ \mathbb { R } ^ { d \times d _ { f f } }$ and $\mathbf { W } _ { 3 } ~ \in ~ \mathbb { R } ^ { d _ { f f } \times d }$ represent learnable weight matrices, and SiLU $( x ) = x \cdot \sigma \left( x \right)$ represents the sigmoid linear unit.

3) Predictor: The final output of the terminal Liquid Mixer layer, denoted as $\mathbf { Z } ^ { ( L ) } ~ \in ~ \hat { \mathbb { R } } ^ { n \times d }$ , serves as the high-level semantic representation of the input sequence. The Predictor maps this representation into the target task space:

$$
\mathbf { P } = \operatorname { P r e d } ( \mathbf { Z } ^ { ( L ) } ) ,\tag{38}
$$

where P represents the task-oriented target output (e.g., forecasted future trajectories or classification probabilities). Depending on downstream constraints, the operator Pred (·) can be instantiated as a linear projection head, a global pooling operator, or a multilayer perceptron (MLP).

## V. EXPERIMENTS AND DISCUSSION

This section empirically evaluates LFormer around three core research questions (RQs):

• RQ1 (Comparative Performance): How does LFormer perform against representative state-of-the-art discretetime and continuous-time baselines across diverse temporal tasks, regarding long-range dependency modeling, fine-grained state tracking, and trajectory reconstruction from sparse and noisy observations?

TABLE I: A Summary of 16 Datasets across 6 Tasks
<table><tr><td>Task</td><td>Dataset</td><td>Irregular</td><td>Length</td><td>Dimension</td><td>Target</td><td>Size</td></tr><tr><td rowspan="6">LTS-C</td><td>EC</td><td rowspan="6">No</td><td>1,751</td><td>3</td><td>4</td><td>524</td></tr><tr><td>EW</td><td>17,984</td><td>6</td><td>5</td><td>236</td></tr><tr><td>HB</td><td>405</td><td>61</td><td>2</td><td>409</td></tr><tr><td>MI</td><td>3,000</td><td>64</td><td>2</td><td>378</td></tr><tr><td>SCP1</td><td>896</td><td>6</td><td>2</td><td>561</td></tr><tr><td>SCP2</td><td>1,152</td><td>7</td><td>2</td><td>380</td></tr><tr><td rowspan="3">LTS-R</td><td>HR</td><td rowspan="3">No</td><td rowspan="3">4,000</td><td rowspan="3">2</td><td rowspan="3">1</td><td>7,949</td></tr><tr><td>RR</td><td>7,870</td></tr><tr><td>SpO2</td><td>7,949</td></tr><tr><td>PTS-C</td><td>HA</td><td>Yes</td><td>50</td><td>12</td><td>7</td><td>6,554</td></tr><tr><td>PTS-R</td><td>PDL</td><td>Yes</td><td>50</td><td>24 × 24</td><td>2</td><td>4,000</td></tr><tr><td rowspan="3">TS-I TS-E</td><td>2DSα</td><td rowspan="3">Yes</td><td>150</td><td></td><td></td><td></td></tr><tr><td>USH</td><td>730</td><td>2 5</td><td>2 5</td><td>300 1,192</td></tr><tr><td>PHY</td><td>72.16</td><td>37</td><td>37</td><td>8,000</td></tr></table>

Note: Irregular specifies the presence of irregular sampling; Dimension and Size denote the feature count and total instances, while Target indicates the number of classes or continuous values to be predicted. 2DS<sub>α</sub> refers to the 2D Spiral dataset variants with $\alpha \in \ \{ 0 . 0 2 , \bar { 0 } . 2 , 2 \}$ . Due to the variable length nature of PHY, its reported length reflects the sample mean.

• RQ2 (Ablation Analysis): How do the individual components of LFormer, including the continuous-time liquid gate dynamics, sequence-level normalization, and decoupled multi-head mechanism, contribute to its representational capacity and numerical stability?

• RQ3 (Computational Efficiency): Does LFormer achieve linear computational and memory complexity under both non-causal and causal settings while maintaining competitive accuracy over extended sequence horizons?

## A. Experimental Setup

1) Tasks and Datasets : We adopt a benchmark across six temporal modeling tasks spanning sixteen datasets to evaluate the three core capabilities defined in RQ1:

• Long-term Time Series Classification and Regression (LTS-C and LTS-R) evaluate the capacity to model longrange dependencies by mapping global trajectories to categorical labels or continuous scalars. The classification setup incorporates six datasets {EC, EW, HB, MI, SCP1, SCP2} from the UEA<sup>1</sup> archive [59] whereas the regression configuration utilizes three datasets {HR, RR, $\mathrm { S p O } _ { 2 } \}$ from the BIDMC<sup>2</sup> database [60] to infer patient vital signs from dense, 4,000-step physiological signals.

• Per-point Time Series Classification and Regression (PTS-C and PTS-R) assess the capability for finegrained state tracking by decoding latent states at arbitrary, irregularly sampled timestamps. For point-wise classification, the Human Activity (HA) dataset<sup>3</sup> [5] is employed to classify localized actions at each time point, using 50 irregular observations per sequence. For point-wise regression, the synthetic Pendulum (PDL) dataset [61] is adopted to infer underlying kinematic angles from noisy image sequences spanning 50 irregularly sampled frames [9].

• Time Series Interpolation and Extrapolation (TS-I and TS-E) evaluate trajectory reconstruction from sparse and noisy observations by recovering continuoustime dynamics under partial observation constraints. We adopt three datasets. The synthetic 2D spiral datasets $( \mathrm { 2 D } \mathrm { S } _ { \alpha } )$ [10] generate 150-step spiral trajectories with shape variance controlled by $\alpha \in \{ 2 , 0 . 2 , 0 . 0 2 \}$ ; models observe 30 randomly sampled points from the first half and reconstruct the full trajectory. The USHCN (USH) dataset<sup>4</sup> [62] comprises daily climate measurement (5 variables) from 1,192 weather stations across the United States over a four-year period. Following [9], we subsample 50% of the time points and randomly drop 20% of the remaining observations to obtain irregularly sampled sequences. The Physionet (PHY) dataset<sup>5</sup> [63] contains 8,000 multivariate clinical time series from intensive care unit patients, each recording 37 time-varying features over 48 hours. For both USH and PHY, we evaluate preprocessing, interpolation and extrapolation following the protocols in [45].

Table I summarizes the structural characteristics of these datasets. Section S4 of the Supplementary Material provides comprehensive preprocessing details, including irregular sampling protocols and train/validation/test partitioning schemas.

2) Baselines : We benchmark LFormer against state-of-theart baselines from three paradigms:

• Discrete-time Models. This paradigm encompasses architectures engineered for uniformly indexed sequences without explicit physical time tracking. Representatives include GRU [11] and Transformer [13]. Modern linear recurrent and discrete state-space models are incorporated, including LRU [64] and Mamba (S6) [6]. We also include baselines that augment discrete-time models with time-interval inputs or decay mechanisms: GRU-∆ [65], GRU-D [32], and RKN-∆<sub>t</sub> [61].

• Solver-dependent Continuous-time Models. This paradigm comprises architectures that parameterize latent state derivatives with neural networks and evaluate hidden trajectories via numerical differential solvers. This setup features vector-field equations ( NODE [19], R-ODE [19], O-ODE [5], LSDE [66]), continuous path integrals (NCDE [20], NRDE [21], LogNCDE [67]), solver-driven continuous-time attention ContiFormer [10], stochastic filtering CRU [9] and Bayesian extension GRU-ODE-B [68].

• Solver-free Continuous-time Models: This paradigm groups architectures that evaluate trajectory representations analytically or via exact linear discretization rules, bypassing iterative integration overhead. This setup incorporates closed-form liquid dynamics CfC [22], truncated algebraic path signatures RFormer [25], multi-time embedding kernels (mTAND [69], and time-varying continuous state-space formulations (S5 [70], ACSSM [45]).

Detailed description of all baselines is provided in Section S5 of the Supplementary Material.

TABLE II: Long-term Time Series Classification Accuracy (%, $\mathrm { m e a n \pm s t d } )$ on Six UEA Datasets
<table><tr><td rowspan="2">Dataset</td><td colspan="10">Method</td><td rowspan="2"> $\Delta _ { \uparrow }$ </td></tr><tr><td>LRU† [64]</td><td>S5† [70]</td><td>S6† [6]</td><td>Mamba† [6]</td><td>NCDE† [20]</td><td>NRDE† [21]</td><td>LogNCDE† [67]</td><td>Transformer† [13]</td><td>RFormer† [25]</td><td>LFormer (ours)</td></tr><tr><td>EC</td><td> $2 1 . 5 { \scriptstyle \pm 2 . 1 }$ </td><td> $2 4 . 1 _ { \pm 4 . 3 }$ </td><td> $2 6 . 4 _ { \pm 6 . 4 }$ </td><td> $2 7 . 9 _ { \pm 4 . 5 }$ </td><td> $2 9 . 9 _ { \pm 6 . 5 }$ </td><td> $2 5 . 3 { \scriptstyle \pm 1 . 8 }$ </td><td> $3 4 . 4 _ { \pm 6 . 4 }$ </td><td> $\underline { { 4 0 . 5 } } \pm 6 . 3$ </td><td> $3 4 . 7 _ { \pm 4 . 1 }$ </td><td> $4 5 . 3 _ { \pm 3 . 5 }$ </td><td> $+ 1 1 . 9 \%$ </td></tr><tr><td>EW</td><td> $8 7 . 8 { \scriptstyle \pm 2 . 8 }$ </td><td> $8 1 . 1 _ { \pm 3 . 7 }$ </td><td> $8 5 . 0 _ { \pm 1 6 . 1 }$ </td><td> $7 0 . 9 _ { \pm 1 5 . 8 }$ </td><td> $7 5 . 0 _ { \pm 3 . 9 }$ </td><td> $8 3 . 9 { \scriptstyle \pm 7 . 3 }$ </td><td> $8 5 . 6 _ { \pm 5 . 1 }$ </td><td>OOM</td><td> ${ \bf 9 0 . 3 _ { \pm 0 . 1 } }$ </td><td> $\underline { { 8 9 . 4 } } \pm 3 . 2$ </td><td> $- 1 . 0 \%$ </td></tr><tr><td>HB</td><td> $7 8 . 4 \pm 6 . 7$ </td><td> $7 7 . 7 \pm 5 . 5$ </td><td> $7 6 . 5 { \scriptstyle \pm 8 . 3 }$ </td><td> $7 6 . 2 { \scriptstyle \pm 3 . 8 }$ </td><td> $7 3 . 9 { \scriptstyle \pm 2 . 6 }$ </td><td> $7 2 . 9 { \pm } 4 . 8$ </td><td> $7 5 . 2 { \pm } 4 . 6 $ </td><td> $7 0 . 5 { \scriptstyle \pm 0 . 1 }$ </td><td> $7 2 . 5 { \scriptstyle \pm 0 . 1 }$ </td><td> ${ \bf 8 1 . 6 \pm } _ { 1 . 6 }$ </td><td>+4.1%</td></tr><tr><td>MI</td><td> $\overline { { 4 8 . 4 } } \pm 5 . 0$ </td><td> $4 7 . 7 \pm 5 . 5$ </td><td> $5 1 . 3 { \scriptstyle \pm 4 . 7 }$ </td><td> $4 7 . 7 \pm 4 . 5$ </td><td> $4 9 . 5 { \scriptstyle \pm 2 . 8 }$ </td><td> $4 7 . 0 { \scriptstyle \pm 5 . 7 }$ </td><td> $5 3 . 7 { \pm } 5 . 3 $ </td><td> $5 0 . 5 { \scriptstyle \pm 3 . 0 }$ </td><td> $5 5 . 8 { \pm } 6 . 6 $ </td><td> ${ \bf 6 5 . 3 \pm _ { 4 . 1 } }$ </td><td> $+ 1 7 . 0 \%$ </td></tr><tr><td>SCP1</td><td> $8 2 . 6 { \scriptstyle \pm 3 . 4 }$ </td><td> ${ \bf 8 9 . 9 2 . 4 . 6 }$ </td><td> $8 2 . 8 \pm 2 . 7$ </td><td> $8 0 . 7 \pm 1 . 4$ </td><td> $7 9 . 8 \pm 5 . 6$ </td><td> $8 0 . 9 { \pm 2 . 5 }$ </td><td> $8 3 . 1 \pm 2 . 8$ </td><td> $8 4 . 3 { \scriptstyle \pm 6 . 3 }$ </td><td> $8 1 . 2 \pm 2 . 8$ </td><td> $\underline { { 8 4 . 5 \pm 1 . 7 } }$ </td><td>-6.0%</td></tr><tr><td>SCP2</td><td> $5 1 . 2 _ { \pm 3 . 6 }$ </td><td> $5 0 . 5 { \scriptstyle \pm 2 . 6 }$ </td><td> $4 9 . 9 { \scriptstyle \pm 9 . 5 }$ </td><td> $4 8 . 2 _ { \pm 3 . 9 }$ </td><td> $5 3 . 0 _ { \pm 2 . 8 }$ </td><td> $\underline { { 5 3 . 7 } } \pm 6 . 9$ </td><td> $\underline { { 5 3 . 7 } } \pm 4 . 1$ </td><td> $4 9 . 1 \pm 2 . 5$ </td><td> $5 2 . 3 _ { \pm 3 . 7 }$ </td><td> ${ \bf 6 1 . 8 _ { \pm 3 . 0 } }$ </td><td> $+ 1 5 . 1 \%$ </td></tr><tr><td>Avg. Acc.↑</td><td>61.7</td><td>61.8</td><td>62.0</td><td>58.6</td><td>60.2</td><td>60.6</td><td>64.3</td><td>49.2</td><td>64.5</td><td>71.3</td><td> $+ 1 0 . 5 \%$ </td></tr><tr><td>Avg. Rank↓</td><td>5.67</td><td>5.92</td><td>5.50</td><td>7.92</td><td>6.67</td><td>7.08</td><td>3.92</td><td>6.50</td><td>4.50</td><td>1.33</td><td></td></tr></table>

Note: The best results are highlighted in bold, and the second best are underlined. ∆ denotes the relative performance gain of LFormer over the best competitor. “Avg. $\mathrm { A c c } . ^ { \prime \prime }$ denotes the average accuracy over six datasets, while “Avg. Rank” denotes the average ranking across all datasets (fractional ranking is used for ties; OOM is ranked last). The arrows ↑ and ↓ indicate whether a higher or lower value is better. OOM indicates Out-Of-Memory. Results marked with <sup>†</sup> are taken from [25] under identical dataset splits and evaluation metrics.

TABLE III: Long-term Time Series Regression MAE and RMSE $( \mathrm { m e a n \pm s t d } )$ on Three BIDMC Datasets
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td colspan="9">Method</td><td rowspan="2"> $\Delta _ { \uparrow }$ </td></tr><tr><td>GRU [11]</td><td>Mamba [6]</td><td>Transformer [13]</td><td>R-ODE [5]</td><td>NCDE [20]</td><td>NRDE [21]</td><td>CfC [22]</td><td>RFormer [25]</td><td>LFormer (ours)</td></tr><tr><td rowspan="2">HR</td><td> $\mathrm { { \mathbf { M A E } _ { \downarrow } } }$ </td><td> $9 . 6 4 _ { \pm 0 . 6 1 }$ </td><td> $\underline { { 1 . 9 7 } } \pm 0 . 3 9$ </td><td> $9 . 8 2 _ { \pm 0 . 1 9 }$ </td><td> $1 2 . 1 1 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $9 . 2 0 _ { \pm 0 . 0 8 }$ </td><td> $2 . 7 4 _ { \pm 0 . 1 3 }$ </td><td> $9 . 7 7 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $2 . 3 7 _ { \pm 0 . 1 7 }$ </td><td> ${ \bf 1 . 0 3 \mathrm { _ { \pm 0 . 0 4 } } }$ </td><td> $+ 4 7 . 7 2 \%$ </td></tr><tr><td>RMSE↓</td><td> $1 3 . 0 1 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $2 . 7 5 { \scriptstyle \pm 0 . 7 0 }$ </td><td> $1 3 . 2 2 \pm 0 . 1 6$ </td><td> $1 5 . 1 7 { \scriptstyle \pm 0 . 0 0 }$ </td><td> $1 1 . 4 1 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 . 4 5 _ { \pm 0 . 2 2 }$ </td><td> $1 3 . 1 2 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $3 . 0 9 { \scriptstyle \pm 0 . 2 3 }$ </td><td> ${ \bf 1 . } 5 2 \pm 0 . 0 7$ </td><td> $+ 4 4 . 7 3 \%$ </td></tr><tr><td rowspan="2">RR</td><td>MAE↓</td><td> $1 . 7 9 \pm _ { 0 . 3 6 }$ </td><td> $0 . 9 3 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $1 . 4 5 _ { \pm 0 . 2 2 }$ </td><td> $1 . 8 5 _ { \pm 0 . 0 0 }$ </td><td> $2 . 2 7 _ { \pm 0 . 0 3 }$ </td><td> $1 . 2 0 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $2 . 1 5 _ { \pm 0 . 1 2 }$ </td><td> $1 . 0 6 _ { \pm 0 . 0 4 }$ </td><td> $\mathbf { 0 . 7 4 _ { \pm 0 . 0 4 } }$ </td><td> $+ 2 0 . 4 3 \%$ </td></tr><tr><td>RMSE↓</td><td> $2 . 5 1 \pm _ { 0 . 4 7 }$ </td><td> $1 . 3 2 { \scriptstyle \pm 0 . 1 0 }$ </td><td>1.94±0.25</td><td> $2 . 3 2 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $2 . 8 4 _ { \pm 0 . 0 4 }$ </td><td> $1 . 5 1 \pm 0 . 0 6$ </td><td> $2 . 9 8 { \scriptstyle \pm 0 . 1 6 }$ </td><td>1.46±0.05</td><td> ${ \bf 0 . 9 7 } _ { \pm 0 . 0 2 }$ </td><td>+26.52%</td></tr><tr><td rowspan="2"> $\mathrm { S p O _ { 2 } }$ </td><td>MAE↓</td><td> $2 . 4 6 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 0 8 \pm 0 . 4 9$ </td><td> $2 . 4 5 _ { \pm 0 . 0 3 }$ </td><td> $2 . 7 1 \pm 0 . 0 0$ </td><td> $2 . 4 6 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $1 . 0 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 . 4 6 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $1 . 3 8 { \scriptstyle \pm 0 . 1 5 }$ </td><td> ${ \bf 0 . 2 9 } \pm 0 . 0 3$ </td><td>+71.57%</td></tr><tr><td> $\mathrm { R M S E } _ { \downarrow }$ </td><td> $3 . 2 8 _ { \pm 0 . 0 8 }$ </td><td> $1 . 4 1 _ { \pm 0 . 7 1 }$ </td><td> $3 . 2 8 _ { \pm 0 . 0 8 }$ </td><td> $3 . 4 1 _ { \pm 0 . 0 2 }$ </td><td> $2 . 8 3 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $1 . 2 8 \pm 0 . 1 3$ </td><td> $3 . 2 8 _ { \pm 0 . 0 8 }$ </td><td> $1 . 8 0 { \scriptstyle \pm 0 . 2 2 }$ </td><td> ${ \bf 0 . 4 2 _ { \pm 0 . 0 2 } }$ </td><td>+67.19%</td></tr></table>

Note: The best results are highlighted in bold, and the second best are underlined. ∆ denotes the relative performance gain of LFormer over the best competitor. The arrows $\uparrow$ and $\downarrow$ indicate whether a higher or lower value is better.

3) Evaluation Metrics : For classification tasks (LTS-C, PTS-C), we report Top-1 Accuracy. For LTS-R, we report Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE); for PTS-R, we report Mean Squared Error (MSE). For TS-I and TS-E, we report MAE and RMSE on $2 \mathrm { D } \mathrm { S } _ { \alpha } ,$ and MSE on PHY and USH.

## B. Temporal Modeling Capability Evaluation (RQ 1)

4) Implementation Details : We adopt a shared base configuration across the majority of benchmarks: $L \ = \ 2$ Liquid Mixers, model dimension $d = 6 4$ , SwiGLU expansion $d _ { f f } = 1 7 6$ , and H = 4 attention heads. This default setup is used for all LTS-R tasks and four of the six LTS-C datasets (EC, EW, HB, MI). Task-specific adaptations are applied where required by the data modality or objective: SCP1 and SCP2 use d = 128, $d _ { f f } = 3 5 2$ , H = 1, and $L = 1 ;$ PTS-R employs a convolutional embedder with $d = 1 2 8$ and $d _ { f f } = 3 5 2$ to process 24×24 pixel inputs; PTS-C stacks $L = 4$ layers for fine-grained per-point classification; TS-I/E on $2 \mathsf { D } S _ { \alpha }$ adopts d = 32, L = 1, and a continuous time embedding following [10]; USH and PHY use an MLP embedder with two hidden layers. Complete per-task specifications are provided in Section S6 of the Supplementary Material.

We evaluate LFormer against state-of-the-art baselines along three core temporal modeling dimensions as follows.

1) Long-range Dependency Modeling (LTS-C and LTS-R) : As summarized in Table II, LFormer achieves the highest average accuracy (71.3%) and the top average rank (1.33) on the LTS-C benchmark, delivering the top performance on four out of six UEA datasets. On EW and SCP1, it ranks second behind RFormer and S5, respectively, indicating that signature-based or linear state-space inductive biases can still offer competitive advantages in certain classification regimes. Notably, LFormer secures a top-two finish across all evaluated datasets, whereas no competing baseline achieves a top-two ranking on more than three. On the extended-horizon EW dataset (17,984 steps), LFormer retains robust optimization stability and computational tractability, whereas the canonical Transformer incurs out-of-memory (OOM) exceptions. This result empirically confirms the scalability benefit of LGA’s linear-complexity design.

The optimizer is Adam with a learning rate of $1 0 ^ { - 3 }$ (exceptions: $1 0 ^ { - 2 }$ for PTS-R, TS-I/E on $2 \mathsf { D } \mathsf { S } _ { \alpha } )$ . Early stopping is applied when the validation metric does not improve for 20 consecutive epochs.To promote fair comparison, we calibrate baselines via grid search with the identical early stop strategy, or directly report published results when the evaluation protocol exactly matches ours. All experiments are run with five fixed random seeds {42, 43, 44, 45, 46} and we report mean and standard deviation. All execution pipelines are deployed on a standardized workstation hardware environment equipped with an Intel Core i9-13900K CPU, 128 GB RAM, and a single NVIDIA RTX 4090 GPU.

The empirical advantage of LFormer extends to the LTS-R benchmark. As detailed in Table III, LFormer consistently outperforms all baselines across three evaluated BIDMC datasets in terms of both MAE and RMSE. Compared to the strongest baseline, LFormer achieves relative MAE reductions of 47.72% for HR, 20.43% for RR, and 71.57% for SpO<sub>2</sub>.

These evaluations address the first dimension of RQ1 concerning long-range dependency modeling. The prominent performance gains demonstrate the capacity of LGA’s matrixvalued associative memory to capture extended temporal dependencies, supported by the sequence-level normalization that stabilizes optimization over thousands of time steps in fullsequence encoding scenarios.

TABLE IV: Time Series Interpolation and Extrapolation MSE $( \% , \mathrm { m e a n \pm s t d } )$ on PHY and USH Datasets
<table><tr><td rowspan="2">Task Dataset Metric</td><td rowspan="2"></td><td rowspan="2"></td><td colspan="10">Method</td><td rowspan="2"> $\Delta _ { \uparrow }$ </td></tr><tr><td>mTAND‡ [69] RKN-∆t [61]</td><td></td><td> $\mathrm { { G R U } } { - } \Delta _ { t } ^ { \ddagger } \ [ 1 1 ]$ </td><td></td><td>GRU-D‡ [32] NODE [19]</td><td> $\mathrm { R - O D E ^ { \ddagger } \ } [ 5 ]$ </td><td></td><td></td><td>GRU-ODE-B‡ [68] CRU‡ [9] ACSSM‡ [45] LFormer (ours)</td><td></td></tr><tr><td>TS-I</td><td>PHY</td><td>MSE↓</td><td> $0 . 2 0 8 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $0 . 1 8 6 _ { \pm 0 . 0 3 0 }$ </td><td> $0 . 2 7 1 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $0 . 3 3 8 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 2 1 2 _ { \pm 0 . 0 2 7 }$ </td><td> $0 . 2 3 6 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 5 2 1 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $0 . 1 8 2 _ { \pm 0 . 0 9 1 }$ </td><td> $\underline { { 0 . 1 1 6 } } \pm 0 . 0 1 1$ </td><td> ${ \bf 0 . 0 4 0 _ { \pm 0 . 0 0 9 } }$ </td><td>+65.34%</td></tr><tr><td rowspan="3"></td><td>USH</td><td>MSE↓</td><td> $1 . 7 6 6 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 0 9 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 0 9 0 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 9 4 4 _ { \pm 0 . 0 1 1 }$ </td><td> $1 . 7 9 8 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 8 3 1 _ { \pm 0 . 0 0 8 }$ </td><td> $0 . 8 4 1 _ { \pm 0 . 1 4 2 }$ </td><td> $0 . 0 1 6 _ { \pm 0 . 0 0 6 }$ </td><td> $\underline { { 0 . 0 0 6 _ { \pm 0 . 0 0 1 } } }$ </td><td> $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $1 { + } 8 3 . 3 3 \%$ </td></tr><tr><td>PHY</td><td>MSE↓</td><td> $\mathbf { 0 . 3 4 0 _ { \pm 0 . 0 2 0 } }$ </td><td> $0 . 7 0 3 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 8 7 0 { \scriptstyle \pm 0 . 0 7 7 }$ </td><td> $0 . 8 7 3 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 7 2 5 { \scriptstyle \pm 0 . 0 7 2 }$ </td><td> $0 . 4 6 7 _ { \pm 0 . 0 0 6 }$ </td><td> $0 . 7 9 8 { \scriptstyle \pm 0 . 0 7 1 }$ </td><td> $0 . 6 2 9 { \scriptstyle \pm 0 . 0 9 3 }$ </td><td> $0 . 6 2 7 { \scriptstyle \pm 0 . 0 1 9 }$ </td><td> $\underline { { 0 . 3 7 2 } } \pm 0 . 0 0 2$ </td><td>-9.41%</td></tr><tr><td>USH</td><td>MSE↓</td><td> $2 . 3 6 0 { \scriptstyle \pm 0 . 0 3 8 }$ </td><td> $1 . 4 9 1 { \scriptstyle \pm 0 . 2 7 2 }$ </td><td> $2 . 0 8 1 { \scriptstyle \pm 0 . 0 5 4 }$ </td><td> $1 . 7 1 8 _ { \pm 0 . 0 1 5 }$ </td><td> $2 . 0 3 4 _ { \pm 0 . 0 0 5 }$ </td><td> $1 . 9 5 5 { \scriptstyle \pm 0 . 4 6 6 }$ </td><td> $5 . 4 3 7 _ { \pm 1 . 0 2 0 }$ </td><td> $1 . 2 7 3 { \scriptstyle \pm 0 . 0 6 6 }$ </td><td> $\underline { { 0 . 9 4 1 \pm 0 . 0 1 4 } }$ </td><td> ${ \bf 0 . 8 4 0 _ { \pm 0 . 0 1 8 } }$ </td><td>+10.73%</td></tr></table>

Note: The best results are highlighted in bold, and the second best are underlined. ∆ denotes the relative performance gain of LFormer over the best-performing baseline. The arrows ↑ and ↓ indicate whether a higher or lower value is better. Results marked with <sup>‡</sup> are taken from [45].

![](images/c90254e38394a6b8dbb297a4085e7a46b1238560610f6cad29afeae519e37940.jpg)

![](images/1e294616d53c6172e2333c2598e17e6e0c04cd55296d0fc0e1b1e73d2bea9961.jpg)  
(a) PTS-C Accuracy on the HA dataset  
(b) PTS-R MSE on the PDL dataset  
Fig. 4: Comparison of per-point time series classification and regression performance. Baseline results are reported from [45].

2) Fine-grained Dynamics Tracking (PTS-C and PTS-R) : Figure 4 reports the point-wise state tracking performance under the non-uniformly sampled benchmarks. LFormer delivers a classification accuracy of 92.9% on the HA dataset (PTS-C) and an MSE of 2.1‰ on the PDL dataset (PTS-R).

On the HA dataset, LFormer yields a modest 1.6% accuracy improvement over the state-of-the-art ACSSM, indicating that both methods achieve comparable performance when categorical decision boundaries dominate over high-fidelity continuous-time integration.

On the PDL dataset, which demands dense tracking of continuous pendulum angles from strongly perturbed image sequences, LFormer achieves a substantial 31.2% relative MSE reduction over ACSSM. Since both architectures employ the identical convolutional embedder, the pronounced performance gap isolates the algorithmic gain from visual feature extraction variations, attributing it to the underlying state transition mechanism. Crucially, our ablation analysis (Table VI) reveals that fixing $\mu = 0 . 5$ (recovering the standard trapezoidal rule) outperforms the learnable variant. This indicates that under sparse and noisy conditions, unconstrained interpolation can be sensitive to local noise fluctuations. Conversely, the symmetric smoothing prior functions as an inductive low-pass constraint that suppresses noise propagation. Consequently, these gains stem not from any single component, but from the interplay of LGA’s matrix memory, cumulative gate stabilization, and task-appropriate smoothing priors.

3) Trajectory Reconstruction (TS-I and TS-E) : We evaluate trajectory reconstruction on two scenarios: real-world irregular records (PHY and USH) and controlled synthetic spirals $( \mathrm { 2 D } \mathrm { S } _ { \alpha } )$ under varying signal-to-noise (SNR) ratios.

a) Real-world Irregular Data: As reported in Table IV, LFormer achieves substantial gains on TS-I for both PHY and USH, reducing MSE by 65.3% and 83.3% respectively over the strongest baseline (ACSSM). These margins indicate that coupling state transitions with observed time intervals provides an effective continuous-time inductive bias for interpolation. Within this non-causal setting, LGA leverages bidirectional temporal gaps to modulate the associative memory, circumventing the drift and error accumulation that solver-dependent methods can suffer under sparse conditions.

The TS-E results reveal an architectural boundary: LFormer reduces MSE by 10.73% over ACSSM on USH, yet trails mTAND by 9.41% on PHY. This divergence reflects that LGA’s recurrent memory excels on the smooth dynamics of USH, whereas mTAND’s query-based attention pooling is better suited to the non-stationary signals of PHY.

The ablation results support this: on USH, removing the decoupled multi-head gating or MHLGA incurs the largest performance drop (MSE rises to 0.890 and 0.884; Table VI). This highlights that head-wise specialization across distinct temporal scales is central to capturing the multi-frequency structure of climate variables.

In contrast, on PHY, all ablated variants stay within a narrow MSE range (0.370–0.379), with several configurations matching or slightly surpassing the full model (0.372). This insensitivity indicates that the extrapolation bottleneck of LFormer on PHY stems from a system-level constraint. When handling highly non-stationary signals, extrapolation demands acute sensitivity to localized boundary fluctuations. mTAND directly queries sparse past evidence at arbitrary future timestamps, avoiding the forward drift inherent in the recurrent state-transition of LGA.

TABLE V: Time Series Interpolation and Extrapolation MAE and RMSE (%, $\mathrm { m e a n \pm s t d } )$ on Three $2 \mathsf { D } S _ { \alpha }$ Datasets
<table><tr><td rowspan="2">Task</td><td rowspan="2">α</td><td rowspan="2">Metric</td><td colspan="7">Method</td><td rowspan="2"> $\Delta _ { \uparrow }$ </td></tr><tr><td> $\mathrm { T r a n s f o r m e r ^ { \ S } \ [ 1 3 ] }$ </td><td> $\mathrm { \Delta N O D E ^ { \ S } \ [ 1 9 ] }$ </td><td> $\mathrm { R - O D E ^ { \ S } \ [ 1 9 ] }$ </td><td> $\mathbf { O - O D E ^ { \ S } \ [ 5 ] }$ </td><td> $\mathrm { N C D E ^ { \ S } \ [ 2 0 ] }$ </td><td>ContiFormer§ [10]</td><td>LFormer (ours)</td></tr><tr><td rowspan="6">TS-I</td><td>2</td><td>MAE↓</td><td></td><td> $2 . 4 9 _ { \pm 0 . 3 0 }$ </td><td> $2 . 8 8 _ { \pm 0 . 5 4 }$ </td><td> $4 . 1 2 \pm _ { 1 . 0 0 }$ </td><td> $2 . 9 6 _ { \pm 0 . 2 3 }$ </td><td></td><td> ${ \bf 0 . 4 0 _ { \pm 0 . 0 6 } }$ </td><td>+2.44%</td></tr><tr><td></td><td>RMSE↓</td><td> $\begin{array} { c } { 2 . 3 3 { \pm } 0 . 6 0 } \\ { 3 . 7 5 { \pm } 0 . 5 0 } \end{array}$ </td><td> $2 . 1 0 _ { \pm 0 . 2 1 }$ </td><td> $3 . 0 4 _ { \pm 0 . 4 8 }$ </td><td> $4 . 4 9 _ { \pm 1 . 2 6 }$ </td><td> $2 . 9 2 _ { \pm 0 . 3 2 }$ </td><td> $\frac { 0 . 4 1 \pm 0 . 0 7 } { 0 . 4 5 \pm 0 . 1 1 }$ </td><td> ${ \bf 0 . 4 3 _ { \pm 0 . 0 9 } }$ </td><td>+4.44%</td></tr><tr><td>0.2</td><td> $\mathrm { { M A E } _ { \downarrow } }$ </td><td></td><td> $1 . 5 8 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $3 . 1 6 _ { \pm 0 . 4 3 }$ </td><td></td><td> $4 . 5 8 _ { \pm 0 . 9 3 }$ </td><td> $\underline { { 0 . 4 2 _ { \pm 0 . 0 1 } } }$ </td><td></td><td>+21.43%</td></tr><tr><td></td><td> $\mathrm { R M S E } _ { \downarrow } ^ { \mathsf { ^ { v } } }$ </td><td> $\begin{array} { c } { 1 . 7 5 _ { \pm 0 . 1 8 } } \\ { 2 . 3 0 _ { \pm 0 . 5 1 } } \end{array}$ </td><td> $1 . 8 8 _ { \pm 0 . 0 9 }$ </td><td> $3 . 3 4 _ { \pm 0 . 1 0 }$ </td><td> $\begin{array} { l } { 4 . 7 3 _ { \pm 1 . 2 1 } } \\ { 4 . 8 2 _ { \pm 1 . 9 7 } } \end{array}$ </td><td> $4 . 6 0 \pm _ { 1 . 0 8 }$ </td><td> $\underline { { 0 . 4 2 \pm 0 . 0 4 } }$ </td><td> $\begin{array} { c } { 0 . 3 3 _ { \pm 0 . 0 2 } } \\ { 0 . 3 4 _ { \pm 0 . 0 2 } } \end{array}$ </td><td>+19.05%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>0.02</td><td> $\mathrm { { M A E } _ { \downarrow } }$   $\mathrm { R M S E } _ { \downarrow }$ </td><td> $\begin{array} { c } { 1 . 4 2 _ { \pm 0 . 1 4 } } \\ { 1 . 3 7 _ { \pm 0 . 1 5 } } \end{array}$ </td><td> $\begin{array} { c } { { 1 . 9 0 { \scriptstyle \pm 0 . 1 7 } } } \\ { { 1 . 9 0 { \scriptstyle \pm 0 . 1 7 } } } \end{array}$ </td><td> $1 . 9 5 { \scriptstyle \pm 0 . 2 6 }$   $2 . 0 9 _ { \pm 0 . 2 2 }$ </td><td> $4 . 2 5 { \scriptstyle \pm 1 . 4 8 }$   $4 . 5 3 \pm _ { 1 . 3 4 }$ </td><td> $5 . 1 2 \pm 0 . 4 7$   $4 . 8 0 _ { \pm 0 . 5 0 }$ </td><td> $0 . 5 3 { \scriptstyle \pm 0 . 0 7 }$   $\underline { { 0 . 5 0 _ { \pm 0 . 0 6 } } }$ </td><td> $\begin{array} { c } { 0 . 3 3 { \scriptstyle \pm 0 . 0 9 } } \\ { 0 . 3 1 { \scriptstyle \pm 0 . 0 6 } } \end{array}$ </td><td>+37.74% +38.00%</td></tr><tr><td rowspan="6">TS-E</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>+23.19%</td></tr><tr><td></td><td> $\begin{array} { r } { \mathbf { M A E _ { \downarrow } } } \\ { \mathbf { R M S E _ { \downarrow } } } \end{array}$ </td><td> $\begin{array} { l } { 4 . 4 8 _ { \pm 0 . 8 2 } } \\ { 5 . 3 6 _ { \pm 0 . 9 7 } } \end{array}$ </td><td> $4 . 8 4 _ { \pm 0 . 4 3 }$ </td><td> $4 . 4 6 _ { \pm 0 . 6 0 }$ </td><td> $4 . 2 9 _ { \pm 1 . 3 4 }$ </td><td> $3 . 3 2 _ { \pm 0 . 8 5 }$ </td><td> $0 . 6 9 _ { \pm 0 . 0 7 }$ </td><td> $\begin{array} { c } { 0 . 5 3 _ { \pm 0 . 0 6 } } \\ { 0 . 5 2 _ { \pm 0 . 0 6 } } \end{array}$ </td><td></td></tr><tr><td></td><td></td><td></td><td> $6 . 0 2 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $5 . 7 0 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $4 . 6 8 _ { \pm 1 . 7 0 }$ </td><td> $3 . 4 4 _ { \pm 1 . 0 6 }$ </td><td> $\underline { { 0 . 7 5 } } \pm 0 . 0 7$ </td><td></td><td>+30.67%</td></tr><tr><td>0.2</td><td> $\mathrm { { M A E } _ { \downarrow } }$ </td><td> $2 . 7 9 \pm 1 . 1 1$ </td><td> $3 . 3 5 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $4 . 0 2 _ { \pm 0 . 5 9 }$ </td><td> $4 . 7 2 \pm 1 . 9 6$ </td><td> $3 . 3 7 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $\underline { { 0 . 6 9 \pm 0 . 0 9 } }$ </td><td> ${ \bf 0 . 4 4 } _ { \pm 0 . 0 4 }$ </td><td>+36.23%</td></tr><tr><td> $\mathrm { R M S E } _ { \downarrow }$ </td><td></td><td> $2 . 9 6 _ { \pm 1 . 4 6 }$ </td><td> $4 . 3 8 _ { \pm 0 . 3 5 }$ </td><td> $4 . 7 8 _ { \pm 0 . 1 5 }$ </td><td> $5 . 0 3 _ { \pm 1 . 9 7 }$ </td><td> $3 . 8 1 \pm { _ { 1 . 0 2 } }$ </td><td> $\underline { { 0 . 7 4 } } \pm 0 . 1 0$ </td><td> ${ \bf 0 . 4 5 _ { \pm 0 . 0 6 } }$ </td><td>+39.19%</td></tr><tr><td></td><td> $\mathrm { { M A E } _ { \downarrow } }$ </td><td></td><td> $1 . 4 9 { \scriptstyle \pm 0 . 1 2 }$ </td><td> $2 . 0 2 _ { \pm 0 . 0 7 }$ </td><td> $1 . 5 2 _ { \pm 0 . 0 9 }$ </td><td> $2 . 8 3 _ { \pm 0 . 8 8 }$ </td><td> $2 . 3 8 _ { \pm 0 . 0 4 }$ </td><td> $0 . 6 5 { \scriptstyle \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 3 4 _ { \pm 0 . 0 8 } }$ </td><td>+47.69%</td></tr><tr><td></td><td>0.02</td><td> $\mathrm { R M S E _ { \downarrow } }$ </td><td> $1 . 3 7 { \pm } 0 . 1 1$ </td><td> $2 . 0 7 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 5 9 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $2 . 6 9 _ { \pm 0 . 8 2 }$ </td><td> $2 . 2 4 _ { \pm 0 . 0 3 }$ </td><td> $0 . 6 4 \pm 0 . 1 0$ </td><td> ${ \bf 0 . 3 2 2 0 . 0 6 }$ </td><td>+50.00%</td></tr></table>

Note: The best results are highlighted in bold, and the second best are underlined. ∆ denotes the relative performance gain of LFormer over the best-performing baseline. The arrows ↑ and ↓ indicate whether a higher or lower value is better. Results marked with <sup>§</sup> are taken from [10] under identical evaluation protocols.

![](images/fea5811b64a41467cf9112600a71c6a16bb70bb46a1e0a849bc1a08a9d1b47d7.jpg)  
Fig. 5: Qualitative trajectory reconstruction comparison on the $2 \mathrm { D } \mathrm { S } _ { \alpha = 0 . 0 2 }$ benchmark. Panels (a)-(b) and panels (c)-(d) show standard Archimedean spirals and nonlinear asymptotic spirals, respectively. The initial temporal half contains 30 randomly sampled irregular observations (green dots) for interpolation, while the terminal half is left unobserved for extrapolation capacity. LFormer (red) closely tracks the continuous ground-truth trajectory (dashed green) during the unobserved extrapolation phase, whereas ContiFormer (blue) exhibits structural divergence.

This trade-off yields a concrete direction for future architectures: integrating LGA with explicit multi-scale timestamp query networks. Such a hybrid approach may effectively reconcile global trajectory consistency (evidenced by TS-E results on USH and real-world TS-I benchmarks) and acute localized temporal awareness in unobserved regimes.

to 0.02, the signal weakens and the trajectories become noisedominated. Under these low SNR conditions, solver-dependent methods can suffer from high-frequency overshooting when fitting sparse and noisy observations (as shown in Fig. 5).

b) Synthetic Irregular Trajectories: As detailed in Table V, LFormer achieves the best MAE and RMSE on all three $2 \mathsf { D } S _ { \alpha }$ variants in both TS-I and TS-E. The margin over ContiFormer, the strongest baseline, widens as α decreases: from marginal 2.44% (TS-I) MAE reductions at $\alpha = 2$ to pronounced 37.7% (TS-I) and 47.7% (TS-E) MAE reductions at $\alpha = 0 . 0 2$ . Note that both models employ the same continuous time embedding on this benchmark [10]; the widening gap therefore reflects differences in the underlying state transition mechanisms.

On the $2 \mathsf { D } S _ { \alpha }$ benchmark, α controls the geometric variance of generated trajectories under a constant observational noise. $\mathbf { A } \mathbf { t } { \boldsymbol { \alpha } } = \mathbf { 2 } $ , the signal dominates the noise, and all models can reliably recover the underlying trajectory. As α decreases

We attribute this to the bounded liquid gate: as a convex combination within (0, 1), the formulation limits the propagation of high-frequency noise across time steps. Rather than acting as a spectral filter, this formulation introduces an inductive smoothness prior under strong noise, consistent with the low-pass characteristics of exponential moving averages.

## C. Component Analysis

To evaluate the contributions of individual components in LFormer, we conduct an ablation study across all benchmark datasets. Taking the complete LFormer as the baseline (Full), variants are organized into three functional groups:

• Macroscopic Architecture evaluates structural configurations via three variants

– w/o MHLGA substitutes the MHLGA module with standard multi-head self-attention [13].

TABLE VI: Comprehensive Ablation Study across Real-world and Synthetic Benchmarks (mean<sub>±std</sub>)
<table><tr><td rowspan="2">Task</td><td rowspan="2">Dataset</td><td rowspan="2">Metric</td><td>Full</td><td colspan="3">Macroscopic Architecture</td><td colspan="3">Liquid Gate Dynamics</td><td>Stabilization</td></tr><tr><td>LFormer</td><td>w/o MHLGA</td><td>w/o SwiGLU</td><td>w/o Output Gate</td><td>w/o Decoupled µ</td><td>w/o Learnable µ</td><td>i w/o Input-aware g</td><td>w/o Normalized g</td></tr><tr><td rowspan="3">LTS-R</td><td>HR</td><td></td><td> $1 . 5 2 _ { \pm 0 . 0 7 }$ </td><td> $1 . 9 3 _ { \pm 0 . 2 0 }$ </td><td> $1 . 5 5 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $1 . 5 7 { \scriptstyle \pm 0 . 2 0 }$ </td><td> ${ \bf 1 . 5 0 _ { \pm 0 . 1 1 } }$ </td><td> $1 . 6 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 . 6 8 _ { \pm 0 . 1 0 }$ </td><td> $1 . 5 2 _ { \pm 0 . 0 6 }$ </td></tr><tr><td>RR</td><td>RMSE↓</td><td> ${ \bf 0 . 9 7 } _ { \pm 0 . 0 2 }$ </td><td> $1 . 1 6 \pm 0 . 0 4$ </td><td> $1 . 0 4 \pm 0 . 0 5$ </td><td> $1 . 0 6 \pm 0 . 0 9$ </td><td> $1 . 0 6 \pm 0 . 0 4$ </td><td> $1 . 0 5 { \scriptstyle \pm 0 . 0 7 }$ </td><td> $1 . 1 9 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $1 . 1 4 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td> $S p O 2$ </td><td></td><td> $0 . 4 2 _ { \pm 0 . 0 2 }$ </td><td> $0 . 4 8 { \scriptstyle \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 4 1 { \scriptstyle \pm 0 . 0 1 } }$ </td><td> $0 . 4 2 _ { \pm 0 . 0 3 }$ </td><td> $0 . 4 4 _ { \pm 0 . 0 3 }$ </td><td> $0 . 4 2 _ { \pm 0 . 0 1 }$ </td><td> $0 . 4 8 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 2 }$ </td></tr><tr><td rowspan="9">Average ∆ vs. Full Model</td><td rowspan="9">EC</td><td rowspan="9"></td><td></td><td> $- 2 0 . 2 8 \%$ </td><td> $- 2 . 2 7 \%$ </td><td> $- 4 . 1 9 \%$ </td><td>-4.24%</td><td> $- 4 . 5 0 \%$ </td><td> $- 1 5 . 8 3 \%$ </td><td>-8.22%</td></tr><tr><td> ${ \bf 4 5 . 3 0 } _ { \pm 3 . 5 0 }$ </td><td> $3 9 . 7 0 { \scriptstyle \pm 2 . 6 0 }$ </td><td> $4 1 . 8 0 _ { \pm 5 . 4 0 }$ </td><td>42.00±5.20</td><td> $4 0 . 3 0 { \scriptstyle \pm 4 . 4 0 }$ </td><td> $4 0 . 3 0 { \scriptstyle \pm 4 . 4 0 }$ </td><td> $4 1 . 0 0 { \scriptstyle \pm 4 . 1 0 }$ </td><td> $3 9 . 7 0 { \scriptstyle \pm 4 . 6 0 }$ </td></tr><tr><td> $\mathbf { 8 9 . 4 0 } 2 3 . 2 0$ </td><td> $8 3 . 3 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $8 7 . 2 0 { \scriptstyle \pm 2 . 2 0 }$ </td><td> $8 6 . 7 0 { \scriptstyle \pm 4 . 1 0 }$ </td><td> $8 8 . 9 0 { \scriptstyle \pm 4 . 6 0 }$ </td><td> $8 8 . 9 0 { \scriptstyle \pm 4 . 6 0 }$ </td><td> $8 8 . 9 0 { \scriptstyle \pm 1 . 8 0 }$ </td><td> $8 8 . 3 0 { \scriptstyle \pm 4 . 4 0 }$ </td></tr><tr><td> $\mathbf { A c c . } _ { \uparrow } \ ( \% )$   $8 1 . 6 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $8 0 . 3 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $8 0 . 0 0 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $8 1 . 0 0 { \scriptstyle \pm 2 . 1 0 }$ </td><td> $8 0 . 3 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $8 0 . 3 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $8 0 . 3 0 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $\mathbf { 8 2 . 3 0 } \pm \mathbf { 2 . 3 0 }$ </td></tr><tr><td> ${ \bf 6 5 . 3 0 _ { \pm 7 . 0 0 } }$ </td><td> $6 3 . 5 0 { \scriptstyle \pm 5 . 8 0 }$ </td><td> $6 1 . 4 0 _ { \pm 6 . 3 0 }$ </td><td> $5 9 . 6 0 _ { \pm 6 . 4 0 }$ </td><td> $6 0 . 4 0 { \scriptstyle \pm 6 . 0 0 }$ </td><td> $6 0 . 4 0 _ { \pm 6 . 0 0 }$ </td><td> $6 1 . 8 0 { \scriptstyle \pm 5 . 5 0 }$ </td><td> $6 0 . 0 0 { \scriptstyle \pm 4 . 1 0 }$ </td></tr><tr><td> $8 4 . 5 0 { \scriptstyle \pm 1 . 7 0 }$ </td><td> $\mathbf { 8 9 . 2 0 } 2 4 . 0 0$ </td><td> $8 4 . 9 0 { \scriptstyle \pm 3 . 7 0 }$ </td><td> $8 5 . 2 0 { \scriptstyle \pm 3 . 5 0 }$ </td><td> $8 4 . 0 0 { \scriptstyle \pm 3 . 8 0 }$ </td><td> $8 4 . 5 0 { \scriptstyle \pm 3 . 7 0 }$ </td><td> $8 6 . 4 0 { \scriptstyle \pm 3 . 5 0 }$ </td><td> $8 6 . 4 0 { \scriptstyle \pm 3 . 5 0 }$ </td></tr><tr><td> $6 1 . 8 0 { \scriptstyle \pm 3 . 0 0 }$ </td><td> $6 3 . 2 0 { \scriptstyle \pm 3 . 3 0 }$ </td><td> $6 2 . 8 0 { \scriptstyle \pm 2 . 6 0 }$ </td><td> ${ \bf 6 6 . 0 0 } \pm 2 . 4 0$ </td><td> $6 2 . 1 0 { \scriptstyle \pm 2 . 9 0 }$ </td><td> $6 1 . 8 0 { \scriptstyle \pm 3 . 4 0 }$ </td><td> $6 3 . 5 0 { \scriptstyle \pm 3 . 6 0 }$ </td><td> $6 1 . 8 0 { \scriptstyle \pm 3 . 4 0 }$ </td></tr><tr><td></td><td> $- 2 . 6 2 \%$ </td><td> $- 2 . 6 7 \%$ </td><td> $- 2 . 0 2 \%$ </td><td> $- 3 . 4 7 \%$ </td><td> $- 3 . 4 5 \%$ </td><td> $- 2 . 0 0 \%$ </td><td> $- 3 . 1 0 \%$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9"></td><td>PHY</td><td> $\mathrm { M S E _ { \downarrow } ~ } ( \% )$ </td><td> $0 . 0 4 0 _ { \pm 0 . 0 1 5 }$ </td><td> $0 . 0 7 0 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> ${ \bf 0 . 0 3 3 _ { \pm 0 . 0 0 5 } }$   $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 0 4 2 _ { \pm 0 . 0 0 3 }$   $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 0 4 4 _ { \pm 0 . 0 0 4 }$   $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 0 4 7 _ { \pm 0 . 0 1 8 }$   $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 0 4 7 _ { \pm 0 . 0 0 7 }$   $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 1 3 2 _ { \pm 0 . 0 5 3 }$   $\operatorname { i n f } \pm \operatorname { n a n }$ </td></tr><tr><td>USH</td><td></td><td> $\mathbf { 0 . 0 0 1 } _ { \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 4 8 { \scriptstyle \pm 0 . 0 7 }$ </td><td></td><td> $0 . 4 5 _ { \pm 0 . 0 7 }$ </td><td> ${ \bf 0 . 4 2 . _ { \pm 0 . 0 9 } }$ </td><td>0.46±0.10</td><td> $0 . 5 2 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 2 }$ </td></tr><tr><td> $2 \mathsf { D } S _ { \alpha = 2 }$   $2 \mathrm { D } S _ { \alpha = 0 . 2 }$ </td><td>RMSE↓ (%)</td><td> $0 . 4 3 { \scriptstyle \pm 0 . 0 9 }$   ${ \bf 0 . 3 4 _ { \pm 0 . 0 2 } }$ </td><td> $0 . 4 5 _ { \pm 0 . 0 9 }$ </td><td>0.50±0.19  $0 . 3 8 _ { \pm 0 . 0 5 }$ </td><td> $0 . 3 8 _ { \pm 0 . 0 4 }$ </td><td> $0 . 3 9 _ { \pm 0 . 0 7 }$ </td><td> $0 . 3 7 _ { \pm 0 . 0 5 }$ </td><td> $0 . 3 9 _ { \pm 0 . 1 0 }$ </td><td> $0 . 3 9 _ { \pm 0 . 0 7 }$ </td></tr><tr><td> $2 \mathrm { D } S _ { \alpha = 0 . 0 2 }$ </td><td></td><td> $0 . 3 1 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 4 2 _ { \pm 0 . 0 4 }$ </td><td>0.37±0.08</td><td> $0 . 4 0 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 3 1 { \scriptstyle \pm 0 . 0 4 }$ </td><td> ${ \bf 0 . 3 0 { \scriptstyle \pm 0 . 0 4 } }$ </td><td> $0 . 3 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 3 4 { \scriptstyle \pm 0 . 0 4 }$ </td></tr><tr><td>Average ∆ vs. Full Model</td><td></td><td></td><td> $- 4 1 . 2 2 \%$ </td><td> $- 1 1 . 7 5 \%$ </td><td> $- 1 1 . 4 1 \%$ </td><td>-5.43%</td><td> $- 6 . 1 8 \%$ </td><td> $- 1 2 . 3 0 \%$ </td><td> $- 3 2 . 8 8 \%$ </td></tr><tr><td>PHY</td><td></td><td></td><td> $0 . 3 7 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> ${ \bf 0 . 3 7 0 { \scriptstyle \pm 0 . 0 0 3 } }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">TS-E</td><td> $\mathrm { M S E _ { \downarrow } ~ } ( \% )$ </td><td> $0 . 3 7 2 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 8 8 4 _ { \pm 0 . 0 0 7 }$ </td><td> ${ \bf 0 . 8 3 5 _ { \pm 0 . 0 1 0 } }$ </td><td> $0 . 3 7 5 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 3 7 0 { \scriptstyle \pm 0 . 0 0 5 } }$   $0 . 8 9 0 _ { \pm 0 . 1 1 6 }$ </td><td> $0 . 3 7 4 { \scriptstyle \pm 0 . 0 0 6 }$   $0 . 8 3 7 _ { \pm 0 . 0 1 4 }$ </td><td> $0 . 3 7 3 { \scriptstyle \pm 0 . 0 0 4 }$   $0 . 8 4 2 _ { \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 3 7 0 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr><tr><td>USH</td><td> $0 . 8 4 0 _ { \pm 0 . 0 1 8 }$   $0 . 5 2 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 5 1 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 8 3 6 _ { \pm 0 . 0 0 5 }$   $0 . 5 5 { \scriptstyle \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 4 9 } _ { \pm 0 . 0 5 }$ </td><td> $0 . 5 2 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $0 . 5 7 { \scriptstyle \pm 0 . 0 8 }$ </td><td> $\operatorname { i n f } _ { \pm \operatorname { n a n } }$   $0 . 5 1 { \scriptstyle \pm 0 . 0 8 }$ </td></tr><tr><td> $2 \mathrm { D } S _ { \alpha = 2 }$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 4 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $0 . 4 7 _ { \pm 0 . 0 4 }$ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 0 7 }$ </td><td> ${ \bf 0 . 4 4 _ { \pm 0 . 0 7 } }$ </td><td> $0 . 4 6 { \scriptstyle \pm 0 . 0 6 }$ </td></tr><tr><td> $2 \mathrm { D } S _ { \alpha = 0 . 2 }$   $2 \mathrm { D } \mathrm { S } _ { \alpha = 0 . 0 2 }$ </td><td> $\mathrm { R M S E _ { \downarrow } ~ } ( \% )$   ${ \bf 0 . 3 2 _ { \pm 0 . 0 6 } }$ </td><td> $0 . 4 3 _ { \pm 0 . 0 1 }$ </td><td> $0 . 4 3 _ { \pm 0 . 1 0 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $0 . 3 3 _ { \pm 0 . 0 5 }$ </td><td> $0 . 3 3 { \scriptstyle \pm 0 . 0 4 }$ </td><td> $0 . 3 5 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $0 . 3 7 _ { \pm 0 . 0 5 }$ </td></tr><tr><td>Average ∆ vs. Full Model</td><td></td><td></td><td> $- 1 1 . 6 4 \%$ </td><td> $- 7 . 1 4 \%$ </td><td></td><td> $- 1 4 . 6 5 \%$ </td><td> $- 1 . 3 3 \%$ </td><td> $- 1 . 6 8 \%$ </td><td> $- 3 . 3 8 \%$ </td><td> $- 6 . 4 4 \%$ </td></tr><tr><td>PTS-C</td><td></td><td> $\mathbf { A c c . } _ { \uparrow } \ ( \% )$   ${ \bf 9 2 . 8 7 _ { \pm 0 . 1 6 } }$ </td><td> $8 9 . 9 4 _ { \pm 0 . 7 6 }$ </td><td></td><td> $9 0 . 8 7 _ { \pm 0 . 4 9 }$ </td><td> $9 1 . 7 2 _ { \pm 0 . 3 5 }$ </td><td> $9 1 . 8 3 _ { \pm 0 . 1 7 }$ </td><td> $9 1 . 8 1 _ { \pm 0 . 1 7 }$ </td><td> $9 1 . 8 0 _ { \pm 0 . 3 8 }$ </td><td> $2 0 . 4 9 _ { \pm 0 . 0 0 }$ </td></tr><tr><td>HA ∆ vs. Full Model</td><td></td><td></td><td> $- 3 . 1 5 \%$ </td><td> $- 2 . 1 5 \%$ </td><td></td><td></td><td> $- 1 . 1 2 \%$ </td><td> $- 1 . 1 4 \%$ </td><td> $- 1 . 1 5 \%$ </td><td> $\operatorname { C o l l a p s e }$ </td></tr><tr><td>PTS-R</td><td></td><td> $\mathrm { M S E } _ { \downarrow } ~ ( \mathcal { I } _ { o o } )$ </td><td></td><td></td><td> $\operatorname { i n f } \pm \operatorname { n a n }$ </td><td> $- 1 . 2 4 \%$   $\operatorname { i n f } \pm \operatorname { n a n }$ </td><td> $\operatorname { i n f } \pm \operatorname { n a n }$ </td><td> ${ \bf 1 . 5 6 { \scriptstyle \pm 0 . 3 6 } }$ </td><td> $2 . 8 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $\operatorname { i n f } \pm \operatorname { n a n }$ </td></tr><tr><td>PDL ∆ vs. Full Model</td><td></td><td></td><td> $2 . 0 5 _ { \pm 0 . 2 5 }$ </td><td> $\operatorname { i n f } \pm \operatorname { n a n }$ </td><td></td><td></td><td></td><td> $+ 2 3 . 9 0 \%$ </td><td> $- 3 9 . 0 2 \%$ </td><td></td></tr></table>

Note: 2DS denotes the Spiral 2D dataset evaluated under different α values. The arrows ↑ and ↓ indicate whether a higher or lower value is better. ∆ vs. Full Model denotes the average relative percentage difference in performance compared to the full LFormer model. $\mathbf { i n f \pm n a n }$ indicates the training process collapsed due to $\mathrm { ^ { * } N a N ^ { * } o r \Sigma ^ { \cdots } } \mathrm { ^ { - } }$ values. Collapse denotes performance degradation by orders of magnitude.

– w/o SwiGLU replaces the SwiGLU channel mixer with a standard Feed-Forward Network [13]. – w/o Output Gate removes the output gate.

• Liquid Gate Dynamics dissect continuous-time transition equations via three configurations:

macroscopic variants completely collapse, underscoring the necessity of the full architectural design.

– w/o Decoupled µ shares a global µ across all multihead subspaces instead of learning head-specific coefficients independently.

– w/o Learnable µ fixes the learnable interpolation weight as 0.5, recovering the trapezoidal rule.

– w/o Input-aware g replaces input-driven gating with a deterministic time-decay.

• Output gating supports extrapolation and sparseregime stability. Removing the output gate causes the most severe drop among macroscopic variants on TS-E (-14.65%) and leads to training collapse on PDL. Conversely, its impact remains moderate on classification and dense regression tasks (LTS-R: -4.19%; LTS-C: - 2.02%). This aligns with the output gate’s function as a nonlinear filter for retrieved memory. In sparse or extrapolative scenarios where retrieved states are unreliable, explicitly suppressing noisy channels is crucial; under dense sampling, the underlying liquid gating mechanism already provides sufficient implicit filtering.

• Stabilization assesses optimization stability via w/o Normalized g, which ablates the sequence-level normalization from the cumulative gate computation.

Empirical evaluation metrics for these structural configurations are compiled in Table VI, with specialized architectural analysis detailed below.

## 2) Liquid Gate Dynamics:

## 1) Macroscopic Architecture:

• Decoupled µ enables subspace diversity. w/o Decoupled µ degrades performance consistently across all tasks and causes training collapse on PTS-R. This highlights the importance of head-specific interpolation for capturing multi-scale temporal dynamics.

• MHLGA is critical for regression tasks under irregular sampling. On classification tasks, w/o MHLGA and w/o SwiGLU result in comparable, minor degradation (LTS-C: -2.62% vs. -2.67%; PTS-C: -3.15% vs. -2.15%). This suggests coarse-grained categorical decisions are relatively robust to the specific choices of temporal encoders or channel mixers. On regression tasks, this gap widens substantially: w/o MHLGA degrades performance far more severely than w/o SwiGLU on both LTS-R (- 20.28% vs. -2.27%) and TS-I (-41.22% vs. -11.75%) tasks. Furthermore, on the highly sparse PDL dataset, all • Learnable µ balances adaptability and stability. Fixing µ (w/o Learnable µ) degrades performance across most tasks, yet yields a 23.90% MSE improvement on PTS-R. This suggests that under highly irregular, sparse observations, unconstrained learned interpolation can overfit to local noise, whereas the symmetric trapezoidal prior suppresses error accumulation at the cost of local flexibility.

• Deterministic temporal decay fails across complex vector fields. Replacing input-driven modulation with a static time-decay (w/o Input-aware g) causes severe degradation on regression tasks (LTS-R: -15.83%; TS-I: -12.30%; PTS-R: -39.02%), whereas classification performance is largely unaffected (LTS-C: -2.00%; PTS-C: -1.15%). This indicates that input-aware gating is critical for distinguishing temporal structure from noise when modeling continuous-time dynamics.

![](images/f0d85f8845489f46a9a45b053a9070bae80d79de256a86ac49ebe56f7d5de7b1.jpg)  
(a) Peak GPU Memory Usage

![](images/73ac5ad2317c0bc0b446f0bed4f6ace9cc32fec0531cde33eb2d8f3107c7f42b.jpg)  
(b) Per-Step Latency  
Fig. 6: Empirical complexity scaling. (a) Peak GPU memory usage and (b) training step latency as functions of sequence length. OOM (Out-Of-Memory) indicates that execution exceeded available GPU memory; OOT (Out-Of-Time) indicates that a single step exceeded the measurement time limit.

![](images/2ee9299b49654216102887d450e1a05fa94b2715a34a6bb8bfee09ddd3d733bd.jpg)  
Fig. 7: Optimization dynamics during the stabilization ablation on the PTS-C benchmark. The left and right vertical axes track training loss convergence (solid curves) and out-of-sample test accuracy (dashed curves), respectively.

3) Numerical Stabilization: Removing sequence-level normalization degrades average performance across LTS-R (- 8.22%), LTS-C (-3.10%), TS-I (-32.88%), and TS-E (-6.44%), with individual exceptions on HB and PHY (TS-E) where it proves unnecessary or mildly beneficial. This omission causes training collapse on PTS-C, PTS-R, and on USH in both TS-I and TS-E. Figure 7 illustrates the training dynamics on PTS-C: w/o Normalized g suffers immediate gradient explosion and flatlines near the random-guessing level, while LFormer maintains stable optimization and smooth convergence.

The failure stems from the unconstrained cumulative sum in the liquid gate. Without normalization, the prefix sum over non-negative decay terms grows monotonically with sequence length, and the subsequent exponentiation amplifies this into numerical overflow or vanishing gradients. By bounding the cumulative gate, sequence-level normalization prevents these extremes. This mechanism is therefore essential for stable training on long or irregularly sampled sequences, while on shorter or regularly sampled tasks its role can be partially compensated by other components.

## D. Computational Efficiency (RQ3)

To address RQ3, we evaluate LFormer across three dimensions: (1) empirical complexity scaling with sequence length; (2) hardware utilization and inference throughput; and (3) the accuracy-efficiency Pareto trade-off.

1) Empirical Complexity Scaling: To quantify asymptotic complexity, we evaluate two variants of the parallel LGA operator: a causal variant $\mathrm { L F o r m e r _ { s c } }$ that leverages a prefix scan (Eq. (18)), and a non-causal bidirectional variant LFormer that uses full-sequence matrix multiplication (Eq. (19)). We benchmark both variants against the standard Transformer [13], Mamba [6], and R-ODE [5]. Synthetic trajectories are generated with a fixed batch size of 8 and a feature dimension of 12, while the sequence length scales exponentially from $2 ^ { 6 }$ to $2 ^ { 1 5 }$ steps. For fair comparison, we align hyperparameters across all models: hidden dimension 64, depth 2, and 4 attention heads for attention-based variants.

Fig. 6a reports the peak VRAM consumption, defined as the maximum GPU memory allocated during a complete training iteration. The Transformer exhibits quadratic memory growth and exceeds GPU capacity at $2 ^ { 1 3 }$ steps (OOM). Both $\mathrm { L F o r m e r _ { s c } }$ and $\mathrm { L F o r m e r _ { b i } }$ exhibit linear memory scaling across all tested lengths, where LFormer<sub>sc</sub> incurs higher memory overhead due to intermediate state maintenance during the prefix scan. Both LFormer variants show a linear spatial scaling profile, avoiding the quadratic bottleneck of global attention. Fig. 6b reports the execution latency, measured as the average time over 10 consecutive training forward-backward steps. Due to sequential numerical integration, R-ODE incurs substantial overhead, exceeding $1 0 ^ { 4 }$ ms at sequence length $2 ^ { 1 1 }$ . We mark this as out-of-time (OOT) and omit longer sequences. Both LFormer variants show a linear temporal scaling profile. Despite its native PyTorch implementation, LFormer closely matches Mamba which is accelerated by the hardware-optimized selective scan.

![](images/aa06c5d41ad90cf214bfee2fcc0fa0f87902cb5d4a5328410a1f9a43cfaf4d76.jpg)  
(a) Pareto Frontier on the MI Dataset

![](images/211e585c92e4ef5c3368b75903bf40a7009db1ea0ad34ee12431f875b7e33ccb.jpg)  
(b) Pareto Frontier on the EW Dataset  
Fig. 8: The efficiency-accuracy Pareto frontier on representative LTS-C datasets. The x-axis shows inference throughput (sequences per second) on a logarithmic scale; the y-axis shows classification accuracy.

TABLE VII: Parameter Count and Inference Throughput
<table><tr><td>Method</td><td>101↓</td><td>T↑</td><td>Δ↑</td></tr><tr><td>R-ODE [5]</td><td>34.56</td><td>2.46</td><td>1.0×</td></tr><tr><td>Transformer [13]</td><td>101.38</td><td>257.95</td><td>~105×</td></tr><tr><td>Mamba [6]</td><td>66.69</td><td>8,974.65</td><td>~3,648×</td></tr><tr><td>LFormer (ours)</td><td>103.86</td><td>4,652.90</td><td>~1,891×</td></tr></table>

Note: The best results are highlighted in bold, and the second best are underlined. The arrows ↑ and ↓ indicate whether a higher or lower value is better. |θ| denotes the number of trainable parameters (K), T represents the inference throughput (sequence per second), and ∆ indicates the speedup relative to the R-ODE baseline.

These empirical observations confirm that LFormer exhibits linear scaling complexity across spatial and temporal dimensions under both the non-causal and causal settings.

2) Throughput and Hardware Efficiency: We further evaluate the practical hardware efficiency of bidirectional LFormer by measuring parameter count and inference throughput under identical configurations. Synthetic sequences are generated with a batch size of 16, a feature dimension of 12, and a sequence length of 4, 096 steps.

Table VII summarizes the throughput and parameter alignment across the evaluated architectures. LFormer matches the parameter footprint of the standard Transformer while achieving substantially higher throughput, supporting the efficacy of its linear formulation. Although LFormer trails Mamba in absolute inference speed, this gap partly reflects Mamba’s use of hardware-optimized parallel scan kernels; LFormer relies on standard PyTorch operators without custom CUDA kernels. Consequently, these statistics highlight the substantial potential of LFormer for future operator-level optimization.

3) Accuracy vs. Efficiency Trade-off: To quantify the accuracy-efficiency trade-off, we map empirical Pareto frontiers on the MI and EW datasets. All baselines are instantiated using optimal configurations from their respective literature. For RFormer [25], we evaluate two configurations: an online variant that computes signatures during inference, and an offline variant that uses pre-computed cached signatures.

Fig. 8 plots the classification accuracy against logarithmic inference throughput. An ideal architecture occupies the topright quadrant, maximizing both accuracy and throughput. The visualization reveals several insights:

• Solver-dependent Models’ Bottlenecks: NCDE, NRDE, and LogNCDE occupy the lowest efficiency tier. Their reliance on numerical solvers necessitates iterative vectorfield evaluations per step, imposing a prohibitive temporal bottleneck that limits scalability.

• Transformer’s Scalability Limits: The standard Transformer achieves moderate throughput via parallelized attention. However, its quadratic complexity limits scalability over extended sequences, resulting in OOM exceptions on the EW dataset.

• Linear-Time RNNs’ Suboptimality: While LRU and S5 enhance efficiency via diagonalized recurrent matrices, they deliver lower performance, limiting their position on the accuracy-efficiency frontier.

• RFormer’s Real-time Constraint: The peak throughput of the offline RFormer variant is enhanced by precomputed signature transformations, which reduce the effective sequence lengths to 200 (MI) and 10 (EW). This preprocessing step assumes offline access to the full sequence, and the online variant shows lower throughput.

• LFormer’s Pareto Optimality: LFormer (bidirectional) traces the leading Pareto frontier on both datasets. It achieves the highest accuracy on MI and ranks second on EW dataset, with negligible throughput degradation compared to the fastest models.

These observations position LFormer as an efficient continuous-time framework that combines competitive temporal modeling capacities with high-throughput execution in non-causal full-sequence settings.

## VI. LIMITATIONS AND FUTURE WORK

The current LGA formulation has several limitations. First, the sequence-level normalization bounds the total decay budget for stable training, but it assumes offline full-sequence access; a prefix-normalized variant would be required for online or causal inference. Second, the learnable endpoint interpolation improves flexibility on most benchmarks, but can overfit under highly sparse and noisy conditions, as evidenced by the PTS-R task where a fixed $\mu \ = \ 0 . 5$ outperforms the learned variant. Third, the constrained extrapolation performance on non-stationary signals reveals the architectural boundary of LGA. These limitations suggest several directions for future work including streaming-compatible normalization, task-adaptive endpoint interpolation and integrating LGA with time-aware query networks.

## VII. CONCLUSION

This paper presented Liquid Gated Attention (LGA), a solver-free operator that embeds observed temporal intervals into gated linear attention. By unifying a continuous-time gating mechanism derived from liquid time-constant dynamics with fast-weight associative memory and a sequence-level normalization, LGA maintains linear spatio-temporal complexity and empirically stable training over extended horizons. Built upon LGA, LFormer provides a modular backbone for continuous-time representation learning. Across six tasks and sixteen benchmarks, LFormer delivers strong results across three core temporal modeling capabilities: long-range dependency modeling, fine-grained state tracking, and trajectory reconstruction from sparse and noisy observations.

Ultimately, LGA demonstrates how continuous-time dynamical principles can be embedded into parallel sequence architectures, positioning LFormer as an efficient backbone for time series representation learning.

## APPENDIX A

DERIVATION OF THE CLOSED-FORM INTEGRAL SOLUTION FOR 1D LTC

We solve the linear non-autonomous ODE that governs a decoupled scalar LTC state $s ( t ) \in  { \mathbb { R } }$ under a scalar exogenous input $x ( t ) \in  { \mathbb { R } }$

$$
\frac { \mathrm { d } s ( t ) } { \mathrm { d } t } = - \big [ w _ { \tau } + f \big ( x ( t ) \big ) \big ] s ( t ) + b _ { \tau } f \big ( x ( t ) \big ) ,\tag{39}
$$

where $w _ { \tau } , b _ { \tau } \in \mathbb { R }$ represent learnable scalar parameters.

Following the structural symmetry assumption established in [22], we introduce a balancing inverse time-constant term $w _ { \tau }$ into the bias compartment to symmetrize the algebraic structure:

$$
\frac { \mathrm { d } s ( t ) } { \mathrm { d } t } = - \big [ w _ { \tau } + f ( x ( t ) ) \big ] s ( t ) + \big [ w _ { \tau } + f ( x ( t ) ) \big ] b _ { \tau } .\tag{40}
$$

The effect of this structural balancing term on the solution has been analyzed in [22], which demonstrated its negligible impact on the approximation fidelity.

By defining the time-variant system rate coefficient as $p ( t ) \triangleq w _ { \tau } + f \big ( x ( t ) \big )$ , we map Eq. (40) onto the canonical form of a first-order linear non-homogeneous ODE:

$$
\frac { \mathrm { d } s ( t ) } { \mathrm { d } t } + p ( t ) s ( t ) = p ( t ) b _ { \tau } .\tag{41}
$$

To derive the analytical trajectory of Eq. (41), we define the standard non-autonomous integrating factor $\mu ( t )$ as:

$$
\mu ( t ) \triangleq \exp \left[ \int _ { 0 } ^ { t } p ( u ) \mathrm { d } u \right] .\tag{42}
$$

From the fundamental theorem of calculus, the derivative of the integrating factor is:

$$
{ \frac { \mathrm { d } \mu ( t ) } { \mathrm { d } t } } = \mu ( t ) { \frac { \mathrm { d } } { \mathrm { d } t } } \left[ \int _ { 0 } ^ { t } p ( u ) \mathrm { d } u \right] = \mu ( t ) p ( t ) .\tag{43}
$$

Multiplying both sides of the canonical ODE in Eq. (41) by $\mu ( t )$ yields:

$$
\mu ( t ) \frac { \mathrm { d } s ( t ) } { \mathrm { d } t } + \underbrace { \mu ( t ) p ( t ) s ( t ) } _ { s ( t ) \frac { \mathrm { d } \mu ( t ) } { \mathrm { d } t } } = \mu ( t ) p ( t ) b _ { \tau } .\tag{44}
$$

Recognizing the left-hand side as the exact derivative of the product $\mu ( t ) s ( t )$ , we condense the equation to:

$$
\frac { \mathrm { d } } { \mathrm { d } t } \big [ \mu ( t ) s ( t ) \big ] = \mu ( t ) p ( t ) b _ { \tau } .\tag{45}
$$

Performing definite integration on Eq. (45) from the initial boundary 0 to the target timestamp t, and noting that $\mu ( 0 ) =$ $\exp ( 0 ) = 1$ , we obtain:

$$
\mu ( t ) s ( t ) - s ( 0 ) = b _ { \tau } \int _ { 0 } ^ { t } \mu ( v ) p ( v ) \mathrm { d } v ,\tag{46}
$$

where $s ( 0 ) \in \mathbb { R }$ denotes the initial state condition.

Given that the exponential integrating factor $\mu ( t )$ is strictly positive, multiplying both sides by $\mu ^ { - 1 } ( t )$ isolates the continuous state trajectory, revealing a nested integral expression:

$$
\begin{array} { r l } & { s ( t ) = s ( 0 ) \mu ^ { - 1 } ( t ) } \\ & { \qquad + b _ { \tau } \displaystyle \int _ { 0 } ^ { t } \mu ^ { - 1 } ( t ) \mu ( v ) p ( v ) \mathrm { d } v } \\ & { = s ( 0 ) \exp \left[ - \displaystyle \int _ { 0 } ^ { t } p ( u ) \mathrm { d } u \right] } \\ & { \qquad + b _ { \tau } \displaystyle \underbrace { \int _ { 0 } ^ { t } \exp \left[ - \displaystyle \int _ { v } ^ { t } p ( u ) \mathrm { d } u \right] p ( v ) \mathrm { d } v } _ { \mathrm { N e s t e d ~ I n e r a l ~ T e m ~ } n ( t ) } . } \end{array}\tag{47}
$$

To simplify the nested integral $n ( t )$ , we apply integration by substitution. Define the auxiliary function a(v) as:

$$
a ( v ) \triangleq \int _ { v } ^ { t } p ( u ) \mathrm { d } u .\tag{48}
$$

Differentiating with respect to v gives $\mathrm { d } a ( v ) = - p ( v )$ dv. The integration limits transform as:

$$
a ( v ) = { \left\{ \begin{array} { l l } { 0 , } & { { \mathrm { w h e n ~ } } v = t , } \\ { \int _ { 0 } ^ { t } p ( u ) \mathrm { d } u , } & { { \mathrm { w h e n ~ } } v = 0 . } \end{array} \right. }\tag{49}
$$

Substituting these relations into n(t) yields:

$$
\begin{array} { r l r } & { } & { n ( t ) = \displaystyle \int _ { a ( 0 ) } ^ { 0 } \exp [ - a ] \big ( - \mathrm { d } a \big ) } \\ & { } & { = \displaystyle \int _ { 0 } ^ { a ( 0 ) } \exp [ - a ] \mathrm { d } a } \\ & { } & { = \Big [ - \exp [ - a ] \Big ] _ { 0 } ^ { a ( 0 ) } } \\ & { } & { = 1 - \exp \big [ - a ( 0 ) \big ] . } \end{array}\tag{50}
$$

Substituting the result of $\operatorname { E q . }$ (50) into $\operatorname { E q . }$ . (47) reduces the state trajectory to:

$$
s ( t ) = \left[ s ( 0 ) - b _ { \tau } \right] \exp \left[ - a ( 0 ) \right] + b _ { \tau } .\tag{51}
$$

Recalling that $\begin{array} { r } { a ( 0 ) = \int _ { 0 } ^ { t } p ( u ) } \end{array}$ du and $p ( t ) = w _ { \tau } + f \big ( x ( t ) \big )$ , we obtain the closed-form integral solution:

$$
s ( t ) = \left( s ( 0 ) - b _ { \tau } \right) \exp \left[ - w _ { \tau } t - \int _ { 0 } ^ { t } f \bigl ( x ( u ) \bigr ) \mathrm { d } u \right] + b _ { \tau } .\tag{52}
$$

## APPENDIX B ERROR ANALYSIS OF THE LEARNABLE ENDPOINT INTERPOLATION

To evaluate the local truncation error of the proposed interpolation, we first establish the requisite mathematical smoothness. Assuming the underlying trajectory x(u) is smooth, the integrand $\tilde { x } ( u ) = \sigma ( \mathbf { x } ( u ) \mathbf { w } _ { f } + b _ { f } ) \in ( 0 , 1 )$ is twice continuously differentiable, ${ \tilde { x } } \in C ^ { 2 } [ t _ { n - 1 } , t _ { n } ] .$ , over the localized time segment with step size $\delta _ { n } = t _ { n } - t _ { n - 1 }$

A Taylor expansion of the endpoint value $\tilde { x } ( t _ { n } )$ centered at $t _ { n - 1 }$ yields:

$$
\tilde { x } \left( t _ { n } \right) = \tilde { x } \left( t _ { n - 1 } \right) + \delta _ { n } \tilde { x } ^ { \prime } \left( t _ { n - 1 } \right) + \frac { \delta _ { n } ^ { 2 } } { 2 } \tilde { x } ^ { \prime \prime } \left( t _ { n - 1 } \right) + \mathcal { O } \left( \delta _ { n } ^ { 3 } \right) .\tag{53}
$$

The exact definite integral can be expressed via Taylor expansion as:

$$
\begin{array} { r l } & { \int _ { t _ { n - 1 } } ^ { t _ { n } } \tilde { x } ( u ) \mathrm { d } u = \delta _ { n } \tilde { x } \left( t _ { n - 1 } \right) } \\ & { \qquad + \frac { \delta _ { n } ^ { 2 } } { 2 } \tilde { x } ^ { \prime } \left( t _ { n - 1 } \right) } \\ & { \qquad + \frac { \delta _ { n } ^ { 3 } } { 6 } \tilde { x } ^ { \prime \prime } \left( t _ { n - 1 } \right) + \mathcal { O } \left( \delta _ { n } ^ { 4 } \right) . } \end{array}\tag{54}
$$

To bypass sequential numerical integration, LGA approximates this integral via a learnable endpoint interpolation, parameterized by a bounded coefficient $\mu = \sigma ( \theta ) \in ( 0 , 1 )$

$$
I _ { \mathrm { a p p r o x } } \triangleq \delta _ { n } \left[ \mu \tilde { x } \left( t _ { n - 1 } \right) + \left( 1 - \mu \right) \tilde { x } \left( t _ { n } \right) \right] .\tag{55}
$$

Substituting the expansion of $\tilde { x } ( t _ { n } )$ from Eq. (53) into the numerical surrogate in Eq. (55) yields:

$$
\begin{array} { r l } & { I _ { \mathrm { a p p r o x } } = \delta _ { n } \Big \{ \mu \tilde { x } \left( t _ { n - 1 } \right) } \\ & { \qquad + \left( 1 - \mu \right) \Big [ \tilde { x } \left( t _ { n - 1 } \right) + \delta _ { n } \tilde { x } ^ { \prime } \left( t _ { n - 1 } \right) } \\ & { \qquad + \frac { \delta _ { n } ^ { 2 } } { 2 } \tilde { x } ^ { \prime \prime } \left( t _ { n - 1 } \right) + \mathcal { O } \left( \delta _ { n } ^ { 3 } \right) \Big ] \Big \} } \\ & { \qquad = \delta _ { n } \tilde { x } \left( t _ { n - 1 } \right) + \delta _ { n } ^ { 2 } ( 1 - \mu ) \tilde { x } ^ { \prime } \left( t _ { n - 1 } \right) } \\ & { \qquad + \frac { \delta _ { n } ^ { 3 } } { 2 } ( 1 - \mu ) \tilde { x } ^ { \prime \prime } \left( t _ { n - 1 } \right) + \mathcal { O } \left( \delta _ { n } ^ { 4 } \right) . } \end{array}\tag{56}
$$

The local truncation error E is defined as the algebraic difference between the exact integral (Eq. (54)) and the numerical surrogate (Eq. (56)):

$$
\begin{array} { r l } { E = \displaystyle \int _ { t _ { n - 1 } } ^ { t _ { n } } \bar { x } ( u ) \mathrm { d } u - I _ { \mathrm { s p g r o x } } } \\ { = \displaystyle \left[ \frac { 1 } { 2 } - ( 1 - \mu ) \right] \delta _ { n } ^ { 2 } \bar { x } ^ { \prime } ( t _ { n - 1 } ) } \\ { + \displaystyle \left[ \frac { 1 } { 6 } - \frac { 1 - \mu } { 2 } \right] \delta _ { n } ^ { 3 } \bar { x } ^ { \prime \prime } ( t _ { n - 1 } ) + \mathcal { O } \left( \delta _ { n } ^ { 4 } \right) } \\ { = \displaystyle \left( \mu - \frac { 1 } { 2 } \right) \delta _ { n } ^ { 2 } \bar { x } ^ { \prime } ( t _ { n - 1 } ) } \\ { + \displaystyle \left( \frac { \mu } { 2 } - \frac { 1 } { 3 } \right) \delta _ { n } ^ { 3 } \bar { x } ^ { \prime \prime } ( t _ { n - 1 } ) + \mathcal { O } \left( \delta _ { n } ^ { 4 } \right) . } \end{array}\tag{57}
$$

Equation (57) characterizes the error under two distinct parameterization regimes:

General case $( \mu \neq 0 . 5 ) \colon$ The error is dominated by the $\delta _ { n } ^ { 2 }$ term, giving a local truncation error of $\mathcal { O } ( \delta _ { n } ^ { 2 } )$ . The magnitude of the leading error is proportional to $| \mu - 0 . 5 |$ so that moderate deviations from the midpoint do not induce unbounded error amplification. This property allows the model to adapt the interpolation to data-dependent temporal dynamics while preserving numerical stability.

• Symmetric case $( \mu = 0 . 5 ) \colon$ The coefficient of the leading $\delta _ { n } ^ { \dot { 2 } }$ term vanishes, and the error algebraically reduces to:

$$
E = - \frac { 1 } { 1 2 } \delta _ { n } ^ { 3 } \tilde { x } ^ { \prime \prime } \left( t _ { n - 1 } \right) + \mathcal { O } \left( \delta _ { n } ^ { 4 } \right) = \mathcal { O } \left( \delta _ { n } ^ { 3 } \right) .\tag{58}
$$

Under this condition, the interpolation recovers the classical trapezoidal rule with second-order local accuracy. For well-behaved integrands, accumulating this local error over a finite interval yields a global error of $\mathcal { O } ( \delta ^ { 2 } )$ .

## APPENDIX C NUMERICAL BOUNDS AND GRADIENT STABILITY OF SEQUENCE-LEVEL NORMALIZATION

Let $\mathbf { u } = \pmb { \Delta } \odot \overline { { \mathbf { x } } }$ denote the unnormalized decay vector, where $u _ { i } = \delta _ { i } { \overline { { x } } } _ { i }$ for $i \in \{ 1 , \ldots , n \}$ . In practice, the temporal intervals are non-negative and the interpolated trajectory satisfies $\overline { { x } } _ { i } \in ( 0 , 1 )$ . Consequently, the elements are non-negative: $u _ { i } \geq 0$ for all i. To prevent unbounded accumulation in the cumulative gate, we apply the sequence-level normalization:

$$
{ \hat { u } } _ { i } = { \frac { u _ { i } } { \sum _ { k = 1 } ^ { n } u _ { k } + \epsilon } } ,\tag{59}
$$

where $\epsilon > 0$ is a small stability constant that prevents division by zero. Since $u _ { i } \geq 0$ and $\epsilon > 0$ , we have $\hat { u } _ { i } \geq 0$ for all i.

Let $\textstyle s _ { j } \triangleq \sum _ { i = 1 } ^ { j } { \hat { u } } _ { i }$ denote the j-th element of the cumulative sum vector $\mathbf { s } = \mathrm { C u m S u m } ( \hat { \mathbf { u } } )$ . Since $\hat { u } _ { i } ~ \geq ~ 0$ , the sequence $\{ s _ { j } \} _ { j = 1 } ^ { n }$ is monotonically non-decreasing. The terminal value $s _ { n }$ is:

$$
s _ { n } = \sum _ { i = 1 } ^ { n } { \hat { u } } _ { i } = { \frac { \sum _ { i = 1 } ^ { n } u _ { i } } { \sum _ { k = 1 } ^ { n } u _ { k } + \epsilon } } .\tag{60}
$$

where the denominator strictly exceeds the numerator, yielding $s _ { n } \ < \ 1$ . Combined with the lower bound $s _ { j } \geq 0$ , we obtain $s _ { j } ~ \in ~ [ 0 , 1 )$ for all $j ~ \in ~ \{ 1 , \ldots , n \}$ . Thus, the stabilized cumulative gate g˙<sub>stable,</sub> $\mathbf { \Phi } _ { , j } ~ = ~ \exp ( - s _ { j } ) ~ \in ~ ( e ^ { - 1 } , 1 ]$ for all $j \in \{ 1 , \ldots , n \}$

Remark on temporal semantics. The normalized gate is not algebraically equivalent to the LTC continuous-time transition. Instead, it functions as a regularized surrogate that bounds the total decay budget while preserving the relative temporal allocation across the sequence. This design is suitable for offline full-sequence representation learning; for online streaming or causal inference, a prefix-normalized variant is required to avoid dependence on future observations.

Remark on gradient stability. Beyond bounding the forwardpass values, the sequence-level normalization alters the asymptotic behavior of the gradients during back propagation through time (BPTT). Consider the gradient of the cumulative gate g˙<sub>stable,</sub> $\mathbf { \delta } _ { , k } = \exp ( - s _ { k } )$ with respect to the unnormalized decay term $u _ { i }$ for $i \leq k$ . By the chain rule:

$$
{ \frac { \partial { \dot { g } } _ { \mathrm { s t a b l e } , k } } { \partial u _ { i } } } = - { \dot { g } } _ { \mathrm { s t a b l e } , k } \cdot { \frac { \partial s _ { k } } { \partial u _ { i } } } = - { \dot { g } } _ { \mathrm { s t a b l e } , k } \cdot \sum _ { j = 1 } ^ { k } { \frac { \partial { \hat { u } } _ { j } } { \partial u _ { i } } } .\tag{61}
$$

The partial derivative of the normalized element $\hat { u } _ { j }$ with respect to $u _ { i }$ is evaluated as:

$$
\frac { \partial \hat { u } _ { j } } { \partial u _ { i } } = \left\{ \begin{array} { l l } { \displaystyle \frac { \sum _ { \ell \neq i } u _ { \ell } + \epsilon } { ( \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon ) ^ { 2 } } , } & { j = i , } \\ { \displaystyle \frac { u _ { j } } { ( \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon ) ^ { 2 } } , } & { j \neq i . } \end{array} \right.\tag{62}
$$

Summing these derivatives over $j = 1 , \dots , k$ simplifies to:

$$
\sum _ { j = 1 } ^ { k } \frac { \partial \hat { u } _ { j } } { \partial u _ { i } } = \frac { \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon - \sum _ { j = 1 } ^ { k } u _ { j } } { ( \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon ) ^ { 2 } } \geq \frac { \epsilon } { ( \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon ) ^ { 2 } } > 0 ,\tag{63}
$$

where the inequality holds because $\textstyle \sum _ { j = 1 } ^ { k } u _ { j } \ \leq \ \sum _ { \ell = 1 } ^ { n }$ u<sub>ℓ</sub>. Substituting this bound back into the chain rule, we obtain the absolute gradient magnitude:

$$
\left. \frac { \partial \dot { g } _ { \mathrm { s t a b l e } , k } } { \partial u _ { i } } \right. = \dot { g } _ { \mathrm { s t a b l e } , k } \cdot \sum _ { j = 1 } ^ { k } \frac { \partial \hat { u } _ { j } } { \partial u _ { i } } \geq e ^ { - 1 } \cdot \frac { \epsilon } { ( \sum _ { \ell = 1 } ^ { n } u _ { \ell } + \epsilon ) ^ { 2 } } .\tag{64}
$$

This quantitative result reveals a crucial property. Let $S \triangleq$ $\textstyle \sum _ { \ell = 1 } ^ { n } u _ { \ell }$ denote the total sum of unnormalized decay terms along the sequence. While the gradient signal does inversely scale with the total sum of the sequence, the attenuation is strictly polynomial $\mathcal { O } ( S ^ { - 2 } )$ . Such polynomial scaling is manageable and easily mitigated by modern adaptive optimizers. In contrast, for the unnormalized variant, the gradient evaluates to $\begin{array} { r } { \frac { \partial { \dot { g } } _ { k } } { \partial u _ { i } } = - { \dot { g } } _ { k } } \end{array}$ , which decays exponentially $\mathcal { O } ( e ^ { - S } )$ as the cumulative sum grows. Thus, the normalization mathematically mitigates the exponential vanishing gradient problem, enabling robust optimization across extremely long temporal horizons.

## APPENDIX D

## DATASET DETAILS AND PREPROCESSING PROTOCOLS

## A. UEA Multivariate Time Series Archive

We select six benchmarks from the UEA multivariate time series archive [59]: EigenWorms (EW), EthanolConcentration (EC), Heartbeat (HB), MotorImagery (MI), SelfRegulation-SCP1 (SCP1), and SelfRegulationSCP2 (SCP2). Following the selection criteria in [67], we adopt benchmarks with more than 200 instances to ensure reliable evaluation. The selected benchmarks span a broad range of sequence lengths (405 to 17,984 steps), enabling assessment of long-range modeling capacity. Duplicate trajectories are removed following the preprocessing protocols in [21], [67]. The remaining sequences are randomly partitioned into training (70%), validation (15%), and testing (15%). Observation timestamps are normalized to [0, 1].

## B. BIDMC Physiological Database

We use three subsets from the BIDMC database [60]: heart rate (HR), respiratory rate (RR), and blood oxygen saturation $\left( \mathrm { S p O _ { 2 } } \right)$ . The task predicts a single continuous scalar (the vital sign value) from the full trajectory. The input consists of dual-channel physiological waveforms sampled at 125 Hz. A 32-second sliding window produces dense sequences spanning 4,000 time steps. Each sequence is labeled with the corresponding heart rate, respiratory rate, or blood oxygen saturation level. The remaining sequences are randomly partitioned into training (70%), validation (15%), and testing (15%). Observation timestamps are normalized to [0, 1].

## C. Human Activity Benchmark

The Human Activity (HA) benchmark [5] comprises 12- dimensional time-series recordings acquired from four wearable sensors across five subjects. The objective centers on dense, point-wise classification, mapping each temporal observation to one of seven distinct physical activity categories. Adhering to the protocol in [5], the raw trajectories are processed into 6,554 distinct sequence instances. While each canonical sequence contains 211 steps, a subset of 50 temporal observations is randomly sampled per instance to create irregularly sampled sequences. Timestamps are normalized by a scaling factor of 1/211, and the dataset is partitioned into 4,194 training, 1,049 validation, and 1,311 testing sequences.

## D. Pendulum Tracking Benchmark

The Pendulum (PDL) dataset [61] consists of sequences containing $2 4 \times 2 4$ pixel images, where the dense target at each temporal index corresponds to the angular coordinate (sin θ, cos θ). Consistent with [9], 4,000 sequences are synthesized. Each baseline trajectory spans 100 uniformly spaced steps, from which 50 frames are randomly sub-sampled to construct an irregular temporal grid. Spatially and temporally correlated noise is introduced to the pixel grids following [61]. Observation timestamps are scaled by a factor of 0.1 following [45]. The data configuration comprises 2,000 training, 1,000 validation, and 1,000 testing instances.

## E. Synthetic 2D Spiral Trajectories

Synthetic 2D spiral trajectories (2DS ) [10] are generated to evaluate trajectory interpolation (TS-I) and extrapolation (TS-E). Each individual trajectory is stochastically assigned

a clockwise or counter-clockwise orientation, governed by Archimedean parametric equations. The first geometric variation is formulated as:

$$
\begin{array} { r } { x ( t ) = ( a + b t ) \cos ( t ) , } \\ { y ( t ) = ( a + b t ) \sin ( t ) , } \end{array}\tag{65}
$$

and the alternative formulation is defined via:

$$
\begin{array} { l } { { \displaystyle x ( t ) = \left( a + \frac { 5 0 b } { e - t } \right) \cos ( e - t ) , } } \\ { { \displaystyle y ( t ) = \left( a + \frac { 5 0 b } { e - t } \right) \sin ( e - t ) , } } \end{array}\tag{66}
$$

where a and b represent latent structural parameters drawn from normal distributions: $a \sim \mathcal { N } ( 0 , \alpha )$ and $b \sim { \mathcal { N } } ( 0 . 3 , \alpha )$ Three distinct benchmarks are established by varying the variance scale parameter $\alpha \in \{ 2 , 0 . 2 , 0 . 0 2 \}$ . Additive Gaussian noise $\mathcal { N } ( 0 , 0 . 1 )$ is added to the training trajectories. Each complete trajectory contains 150 uniformly spaced time steps. To simulate partial observations with irregular sampling, the model receives 30 points randomly drawn from the first half of the trajectory $( t \in [ 1 , 7 5 ] )$ ). The task requires interpolating the unobserved positions within this interval and extrapolating the full trajectory over the entirely unobserved second half $( t \in [ 7 6 , 1 5 0 ] )$ . For each value of α, we generate 300 trajectories, partitioned into 200 training and 100 testing instances.

## F. USHCN Benchmark

The United States Historical Climatology Network (USH) dataset [62] comprises daily measurements from 1,218 weather stations across the United States, covering five variables: precipitation, snowfall, snow depth, and daily minimum and maximum temperature. Following the preprocessing pipeline of [45], we retain a subset of stations with continuous records spanning a four-year period starting from 1990. Stations whose observations end before 1994 or begin after 1990 are excluded, along with measurements flagged with unreliable quality markers. Each variable is standardized independently to zero mean and unit variance, and extreme values exceeding four standard deviations from the mean are treated as missing. To introduce irregular sampling, we randomly subsample 50% of the time points and independently drop 20% of the remaining observations, following the protocol of [45]. Only stations retaining at least 730 observations after subsampling are kept. The processed dataset contains 1,192 stations, which are randomly partitioned into training (60%), validation (20%), and testing (20%) sets at the station level. Observation timestamps are encoded as integer days elapsed since January 1, 1950, and normalized to [0, 1].

## G. PhysioNet Benchmark

The PhysioNet Challenge 2012 dataset (PHY) [63] contains 8,000 multivariate clinical time series collected from intensive care unit (ICU) patients during the first 48 hours after admission. Each record comprises 41 measurements, of which we discard 4 static features (age, gender, height, and ICU type) and retain the remaining 37 time-varying clinical variables, following the preprocessing protocol of [5]. Each variable is independently normalized to [0, 1] via min-max scaling, with statistics computed separately for each data split. The processed records are randomly partitioned into training (60%), validation (20%), and testing (20%) sets at the patient level.

## APPENDIX E DETAILED BASELINE SPECIFICATIONS

This section provides the detailed description for all baseline architectures used in the experimental evaluation.

## A. Discrete-Time Baselines

• GRU [11] models sequential dependencies through discrete gated recurrent units over uniform step grids. It compresses past information via interleaved reset and update gates, serving as a baseline for discrete recurrence. $\mathrm { G R U } - \Delta _ { t }$ is a variant of GRU which takes additional time intervals as input.

• Transformer [13] employs global dot-product selfattention mechanisms over static discrete tokens. It eliminates sequential dependencies via fully parallel matrix operations, evaluating standard discrete relational modeling without explicit temporal distance metrics.

• LRU [64] linearizes and diagonalizes recurrent layers into decoupled complex-valued hidden states to mitigate optimization bottlenecks over long sequences.

• Mamba (S6) [6] introduces input-dependent, selective linear state-space layers parallelized via a hardwareaware associative scan algorithm. In this setup, Mamba operates over uniform discrete steps, with selection step sizes driven by local token features rather than physical temporal distances.

• GRU-D [32] incorporates the missing mask and time intervals into the GRU architecture, enabling it to capture long-term temporal dependencies while leveraging informative missingness for improved prediction.

• RKN-∆<sub>t</sub> [61] factorizes the latent state representation to scalar Kalman updates. It employs local linear dynamics for state propagation and provides explicit uncertainty estimates alongside predictions.

## B. Solver-Dependent Continuous-Time Baselines

• NODE [19] parameterizes the latent state vector-field derivative via a neural network, mapping state trajectories through continuous integration solved by adaptive-step numerical ODE solvers. It utilizes the adjoint sensitivity method to achieve constant-memory backpropagation.

• The Latent ODE framework (comprising R-ODE [5] and O-ODE [5]) proposes a continuous-time generative framework for irregularly sampled time series. It utilizes neural ODE variants driven by either a recurrent recognition network (R-ODE) or an observation-aware continuous encoder (O-ODE) to infer latent state trajectories.

• LSDE [66] formulates variational Bayesian inference over continuous latent trajectories governed by stochastic differential equations. It performs path tracking over homogeneous vector fields through a simplified Kullback-Leibler divergence and parameterized continuous diffusion gradients.

• NCDE [20] extends the neural differential paradigm to controlled differential equations, addressing limitations related to initial conditions. It constructs continuous-time paths from irregularly sampled inputs via cubic spline interpolation to drive the hidden state evolution.

• NRDE [21] combines neural controlled dynamics with rough path theory to process high-frequency, irregularly sampled signals. It maps localized signal intervals into bounded algebraic log-signatures, compressing highfrequency variations into summary statistics to accelerate continuous-time scaling.

• LogNCDE [67] integrates the Log-ODE numerical approximation scheme into the neural rough path paradigm for stiff controlled differential equations, targeting longrange dependencies over extended integration horizons.

• ContiFormer [10] combines the continuous-time representation of neural ODEs with self-attention. It extends the attention formulation into the continuous-time domain to parameterize input-dependent vector fields.

• CRU [9] evolves hidden state according to a linear stochastic differential equation, yielding temporal continuity between hidden states and a gating mechanism that integrates noisy irregular observations.

• GRU-ODE-B [68] combines NODE with a Bayesian update network to handle observations that are irregular in both time and feature dimensions, encoding a continuity prior for the latent process.

## C. Solver-Free Continuous-Time Baselines

• CfC [22] derives a bounded, analytic closed-form integral solution to approximate the dynamics of liquid timeconstant networks. It bypasses the numerical integration overhead, enabling efficient continuous-time modeling.

• RFormer [25] enhances self-attention with multi-view algebraic path signatures. It constructs non-sequential continuous representations designed to bypass sequential integration overhead across long temporal trajectories.

• mTAND [69] leverages a continuous multi-time attention embedding kernel to map irregular, variable-length observation coordinates into fixed-dimensional continuoustime representations, preserving non-uniform interval metrics without tracking hidden derivatives.

• S5 [70] formulates structured linear state-space layers over non-uniform grids, driving state evolution analytically via bilinear (trapezoidal) transformations. It leverages a time-varying parallel associative scan to directly process physical time intervals under full parallelization.

• ACSSM [45] leverages multi-marginal Doob’s htransform mechanics and stochastic optimal control to construct a simulation-free latent dynamics framework. It integrates parallelized variational inference into a linear architecture, enabling parallel optimization over continuous trajectories.

## APPENDIX FDETAILED HYPERPARAMETER AND IMPLEMENTATIONCONFIGURATIONS

This section provides the complete architectural specifications and hyperparameter settings for LFormer.

Before detailing the network architectures, we adopt the following notational conventions. Linear $( d _ { \mathrm { i n } } , \ d _ { \mathrm { o u t } } )$ denotes a fully-connected layer with $d _ { \mathrm { i n } }$ input and $d _ { \mathrm { o u t } }$ output dimensions. Conv $( c _ { \mathrm { i n } } , \ c _ { \mathrm { o u t } } , \ k , \ s , \ p )$ denotes a 2D convolutional layer with $c _ { \mathrm { i n } }$ input channels, $c _ { \mathrm { o u t } }$ output channels, kernel size k, stride s, and padding p. MaxPool(k, s) denotes a 2D max-pooling operation with kernel size k and stride s.

Across all benchmarks, each Liquid Mixer layer consists of a multi-head liquid gated attention (MHLGA) module followed by a SwiGLU channel mixer, organized in a prenormalization residual structure. LFormer stacks L such layers, with hidden dimension d, H attention heads, and SwiGLU expansion dimension $d _ { f f }$ , denoted as L × [MHLGA(d, $H ) \  \ \mathtt { S w i G L U } ( d _ { f f } ) \ ]$

All tasks are optimized using Adam with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } =$ 0.999, no weight decay, and a batch size chosen to maximize GPU memory utilization. Early stopping is applied when the validation metric does not improve for 20 consecutive epochs. The learning rate for each benchmark is listed alongside its architecture below.

• LTS-R Benchmark:

– Embedder: Linear(2, 64) → ReLU() →   
LayerNorm()

– Liquid Mixer: 2 × [MHLGA(64, 4) →   
SwiGLU(176)]

– Predictor: Linear(64, 1)

– Learning Rate: 0.001

• LTS-C Benchmark: For EC, EW, HB, and MI, we adopt the standard configuration (d = 64, H = 4, L = 2). For SCP1 and SCP2, we use a larger hidden dimension (d = 128, $d _ { f f } = 3 5 2 )$ with a single attention head (H = 1) and a single Liquid Mixer layer (L = 1), as this configuration proved more effective for these two datasets.

– Embedder

```javascript
∗ Linear({EC: 3, EW: 6, HB: 61, MI:
64}, 64) → ReLU()
```

∗ Linear({SCP1: 6, SCP2: 7}, 128) →   
ReLU()

– Liquid Mixer

∗ {EC, EW, HB, MI}: 2 × [MHLGA(64,   
4) → SwiGLU(176)]

∗ {SCP1, SCP2}: 1 × [MHLGA(128, 1)   
→ SwiGLU(352)]

– Predictor

```csv
∗ {EC, EW, HB, MI}: Linear(64, {EC:
4, EW: 5, HB: 2, MI: 2})
```

∗ {SCP1, SCP2}: Linear(128, {SCP1:   
2, SCP2: 2})

– Learning Rate: 0.001

• PTS-R Benchmark: Due to the image-based input (24×24 pixels), the Pendulum benchmark employs a convolutional embedder and a larger hidden dimension (d = 128, d<sub>ff</sub> = 352).

– Embedder: Conv(24, 12, 5, 1, 2) → ReLU() → MaxPool(2, 2) → Conv(12, 12, 3, 2, 1) → ReLU() → MaxPool(2, 2) → Flatten() → Linear(108, 128) → ReLU()

– Liquid Mixer: 2 × [MHLGA(128, 4) → SwiGLU(352)]

– Predictor: Linear(128, 2)

– Learning Rate: 0.01

• PTS-C Benchmark: For the dense per-point classification objective on Human Activity, we increase the number of Liquid Mixer layers to L = 4 to capture fine-grained state transitions.

– Embedder: Linear(12, 64)→ReLU()→LayerNorm()→Linear(64, 64)

– Liquid Mixer: 4 × [MHLGA(64, 4) → SwiGLU(176)]

– Predictor: Linear(64, 7)

– Learning Rate: 0.001

## • TS-I and TS-E Benchmarks:

– Embedder:

∗ 2DS<sub>α</sub>: Linear(2, 32) + CTE(t)<sup>6</sup>

∗ USH: Linear(5 , 64) → ReLU() → LN() → 2 × [Linear(64, 64) → ReLU() → LN()] → Linear(64, 64)

∗ PHY: Linear(37, 64) → ReLU() → LN() → 2 × [Linear(64, 64) → ReLU() → LN()] → Linear(64, 64)

– Liquid Mixer:

∗ 2DS<sub>α</sub>: 1 × [MHLGA(32, 4) → SwiGLU(88)]

∗ USH and PHY: 2 × [MHLGA(64, 4) → SwiGLU(176)]

– Predictor:

∗ 2DS<sub>α</sub>: Linear(32, 2)

∗ USH: 3 × [Linear(64, 64)→ReLU()] → Linear(64, 5)

∗ PHY: 3 × [Linear(64, 64)→ReLU()] → Linear(64, 37)

– Learning Rate: 0.01

∗ 2DS<sub>α</sub>: 0.01

∗ USH and PHY: 0.001

## REFERENCES

[1] J. A. Quinn, C. K. I. Williams, and N. McIntosh, “Factorial switching linear dynamical systems applied to physiological condition monitoring,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 31, no. 9, pp. 1537–1551, 2009.

[2] Z. Li, R. Cai, T. Z. J. Fu, Z. Hao, and K. Zhang, “Transferable timeseries forecasting under causal conditional shift,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 46, no. 4, pp. 1932–1949, 2024.

<sup>6</sup>Following [10], we incorporate a continuous time embedding (CTE) into the embedder to provide an auxiliary temporal signal alongside the liquid gate mechanism. The continuous time embedding replaces the step indices in the standard sinusoidal positional encoding [13] with observed timestamps.

[3] A. Elnaggar, M. Heinzinger, C. Dallago, G. Rehawi, Y. Wang, L. Jones, T. Gibbs, T. Feher, C. Angerer, M. Steinegger, D. Bhowmik, and B. Rost, “Prottrans: Toward understanding the language of life through selfsupervised learning,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 10, pp. 7112–7127, 2022.

[4] S. N. Shukla and B. M. Marlin, “A survey on principles, models and methods for learning from irregularly sampled time series: From discretization to attention and invariance,” CoRR, vol. abs/2012.00168, 2020.

[5] Y. Rubanova, T. Q. Chen, and D. Duvenaud, “Latent ordinary differential equations for irregularly-sampled time series,” in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, 2019, pp. 5321–5331.

[6] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” CoRR, vol. abs/2312.00752, 2023.

[7] H. Zhou, S. Zhang, J. Peng, S. Zhang, J. Li, H. Xiong, and W. Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications ofArtificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, 2021, pp. 11 106–11 115.

[8] P. Kidger, “On neural differential equations,” arXiv preprint arXiv:2202.02435, 2022.

[9] M. Schirmer, M. Eltayeb, S. Lessmann, and M. Rudolph, “Modeling irregular time series with continuous recurrent units,” in International Conference on Machine Learning, ICML 2022, 17-23 July 2022, Baltimore, Maryland, USA, 2022, pp. 19 388–19 405.

[10] Y. Chen, K. Ren, Y. Wang, Y. Fang, W. Sun, and D. Li, “Contiformer: Continuous-time transformer for irregular time series modeling,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[11] K. Cho, B. van Merrienboer, C¸ . Gulc¸ehre, D. Bahdanau, F. Bougares,¨ H. Schwenk, and Y. Bengio, “Learning phrase representations using RNN encoder-decoder for statistical machine translation,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, 2014, pp. 1724–1734.

[12] S. Hochreiter and J. Schmidhuber, “Long long short-term memory,” Neural Comput., vol. 9, no. 8, pp. 1735–1780, 1997.

[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA, 2017, pp. 5998–6008.

[14] Z. C. Lipton, D. C. Kale, and R. C. Wetzel, “Directly modeling missing data in sequences with rnns: Improved classification of clinical time series,” in Proceedings of the 1st Machine Learning for Healthcare Conference, 2016, pp. 253–270.

[15] M. Smieja, L. Struski, J. Tabor, B. Zielinski, and P. Spurek, “Processing of missing data by neural networks,” in Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montreal,´ Canada, 2018, pp. 2724–2734.

[16] E. Wang, Y. Jiang, Y. Xu, L. Wang, and Y. Yang, “Spatial-temporal interval aware sequential POI recommendation,” in 38th IEEE International Conference on Data Engineering, ICDE 2022, Kuala Lumpur, Malaysia, May 9-13, 2022, 2022, pp. 2086–2098.

[17] Y. Jiang, Y. Yang, Y. Xu, and E. Wang, “Spatial-temporal interval aware individual future trajectory prediction,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 10, pp. 5374–5387, 2024.

[18] D. Xu, C. Ruan, E. Korpeoglu, S. Kumar, and K. Achan, “Self-attention with functional time representation learning,” in Advances in Neural Information Processing Systems, 2019, pp. 15 889–15 899.

[19] T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, “Neural ordinary differential equations,” in Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montreal,´ Canada, 2018, pp. 6572–6583.

[20] P. Kidger, J. Morrill, J. Foster, and T. J. Lyons, “Neural controlled differential equations for irregular time series,” in Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020.

[21] J. Morrill, C. Salvi, P. Kidger, and J. Foster, “Neural rough differential equations for long time series,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, 2021, pp. 7829–7838.

[22] R. M. Hasani, M. Lechner, A. Amini, L. Liebenwein, A. Ray, M. Tschaikowski, G. Teschl, and D. Rus, “Closed-form continuous-time neural networks,” Nat. Mac. Intell., vol. 4, no. 11, pp. 992–1003, 2022.

[23] I. Kuleshov, E. Romanenkova, V. A. Zhuzhel, G. Boeva, E. Vorsin, and A. Zaytsev, “DeNOTS: Stable deep neural ODEs for time series,” in The Fourteenth International Conference on Learning Representations, 2026.

[24] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences´ with structured state spaces,” in The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022.

[25] F. Moreno-Pino, A. Arroyo, H. Waldon, X. Dong, and A. Cartea, “Rough transformers: lightweight and continuous time series modelling through signature patching,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, 2024.

[26] R. M. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, “Liquid time-constant networks,” in Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, 2021, pp. 7657–7666.

[27] J. Schmidhuber, “Learning to control fast-weight memories: An alternative to dynamic recurrent networks,” Neural Comput., vol. 4, no. 1, pp. 131–139, 1992.

[28] M. C. Mozer, “Induction of multiscale temporal structure,” in Proceedings of the 5th International Conference on Neural Information Processing Systems, 1991, pp. 275–282.

[29] E. Choi, M. T. Bahadori, A. Schuetz, W. F. Stewart, and J. Sun, “Doctor ai: Predicting clinical events via recurrent neural networks,” in Proceedings of the 1st Machine Learning for Healthcare Conference, 2016, pp. 301–318.

[30] B. M. Marlin, D. C. Kale, R. G. Khemani, and R. C. Wetzel, “Unsupervised pattern discovery in electronic health care data using probabilistic clustering models,” in ACM International Health Informatics Symposium, 2012, pp. 389–398.

[31] R. Little and D. Rubin, Statistical Analysis with Missing Data, 2014.

[32] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent neural networks for multivariate time series with missing values,” Scientific Reports, vol. 8, no. 1, p. 6085, 2018.

[33] J. Yoon, J. Jordon, and M. van der Schaar, “GAIN: missing data imputation using generative adversarial nets,” in Proceedings of the 35th International Conference on Machine Learning, ICML 2018, July 10-15, 2018, Stockholmsmassan, Stockholm, Sweden¨ , 2018, pp. 5675–5684.

[34] M. Cheng, Z. Liu, X. Tao, Q. Liu, J. Zhang, T. Pan, S. Zhang, P. He, X. Zhang, D. Wang et al., “A comprehensive survey of time series forecasting: Concepts, challenges, and future directions,” Authorea Preprints, 2025.

[35] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in The Eleventh International Conference on Learning Representations, ICLR 2024, Kigali, Rwanda, May 1-5, 2024, 2023.

[36] Q. Huang, L. Shen, R. Zhang, J. Cheng, S. Ding, Z. Zhou, and Y. Wang, “Hdmixer: Hierarchical dependency with extendable patch for multivariate time series forecasting,” in Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, The Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2024, Vancouver, BC, Canada, February 20-27, 2024, 2024, pp. 12 608– 12 616.

[37] Y. Zhang, L. Ma, S. Pal, Y. Zhang, and M. Coates, “Multi-resolution time-series transformer for long-term forecasting,” in Proceedings ofThe 27th International Conference on Artificial Intelligence and Statistics, 2024, pp. 4222–4230.

[38] V. Ekambaram, A. Jati, N. Nguyen, P. Sinthong, and J. Kalagnanam, “Tsmixer: Lightweight mlp-mixer model for multivariate time series forecasting,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD 2023, Long Beach, CA, USA, August 6-10, 2023, 2023, pp. 459–469.

[39] D. Luo and X. Wang, “Moderntcn: A modern pure convolution structure for general time series analysis,” in The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, 2024.

[40] Y. Ma, Z. Liu, C. Zhuang, Y. Tan, Y. Dong, W. Zhong, and J. Gu, “Nonstationary time-aware kernelized attention for temporal event prediction,” in Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Washington, DC, USA, August 14-18, 2022, 2022, pp. 1224–1232.

[41] S. N. Shukla and B. M. Marlin, “Multi-time attention networks for irregularly sampled time series,” in 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021.

[42] X. Li, T. L. Wong, R. T. Q. Chen, and D. Duvenaud, “Scalable gradients for stochastic differential equations,” in The 23rd International Conference on Artificial Intelligence and Statistics, AISTATS 2020, 26- 28 August 2020, Online [Palermo, Sicily, Italy], ser. Proceedings of Machine Learning Research, S. Chiappa and R. Calandra, Eds. PMLR, 2020, pp. 3870–3882.

[43] J. Chien and Y. Chen, “Learning continuous-time dynamics with attention,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 2, pp. 1906–1918, 2023.

[44] Y. H. Lim, Q. Zhu, J. Selfridge, and M. F. Kasim, “Parallelizing nonlinear sequential models over the sequence length,” in The Twelfth International Conference on Learning Representations, 2024.

[45] B. Park, H. Lee, and J. Lee, “Amortized control of continuous state space feynman-kac model for irregular time series,” in The Thirteenth International Conference on Learning Representations, 2025.

[46] J. Zhao, F. Chu, L. Xie, Y. Che, Y. Wu, and A. F. Burke, “A survey of transformer networks for time series forecasting,” Comput. Sci. Rev., vol. 60, p. 100883, 2026.

[47] A. Katharopoulos, A. Vyas, N. Pappas, and F. Fleuret, “Transformers are rnns: Fast autoregressive transformers with linear attention,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, vol. 119, 2020, pp. 5156– 5165.

[48] I. Schlag, K. Irie, and J. Schmidhuber, “Linear transformers are secretly fast weight programmers,” in Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, 2021, pp. 9355–9366.

[49] Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, and F. Wei, “Retentive network: A successor to transformer for large language models,” CoRR, vol. abs/2307.08621, 2023.

[50] S. Yang, B. Wang, Y. Shen, R. Panda, and Y. Kim, “Gated linear attention transformers with hardware-efficient training,” in Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024, pp. 56 501–56 523.

[51] B. Peng, E. Alcaide, Q. Anthony, A. Albalak, S. Arcadinho, S. Biderman, H. Cao, X. Cheng, M. Chung, L. Derczynski, X. Du, M. Grella, K. K. GV, X. He, H. Hou, P. Kazienko, J. Kocon, J. Kong, J. L. Bartlomiej Koptyra Glen, Hayden Lau, K. S. I. Mantri, F. Mom, A. Saito, G. Song, X. Tang, J. S. Wind, S. Wozniak, Z. Zhang, Q. Zhou, J. Zhu, and R. Zhu, “RWKV: reinventing rnns for the transformer era,” in Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023, 2023, pp. 14 048–14 077.

[52] B. Peng, D. Goldstein, Q. Anthony, A. Albalak, E. Alcaide, S. Biderman, E. Cheah, X. D. triumphs, T. Ferdinan, H. Hou, P. Kazienko, K. K. GV, J. Kocon, B. Koptyra, S. Krishna, R. M. Jr., N. Muennighoff, F. Obeid, A. Saito, G. Song, H. Tu, S. Wozniak, R. Zhang, B. Zhao, Q. Zhao, P. Zhou, J. Zhu, and R. Zhu, “Eagle and finch: RWKV with matrixvalued states and dynamic recurrence,” CoRR, vol. abs/2404.05892, 2024.

[53] B. Peng, R. Zhang, D. Goldstein, H. H. Eric Alcaide Euphrat, Xingjian Du, J. Lin, J. Liu, J. Lu, W. Merrill, G. Song, K. Tan, S. Utpala, N. Wilce, J. S. Wind, T. Wu, D. Wuttke, and C. Zhou-Zheng, “RWKV-7 ”goose” with expressive dynamic state evolution,” CoRR, vol. abs/2503.14456, 2025.

[54] T. Dao and A. Gu, “Transformers are ssms: Generalized models and efficient algorithms through structured state space duality,” in Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024.

[55] S. Yang, B. Wang, Y. Zhang, Y. Shen, and Y. Kim, “Parallelizing linear transformers with the delta rule over sequence length,” in Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

[56] J. Zhang, R. Su, C. Liu, J. Wei, Z. Wang, P. Zhang, H. Wang, H. Jiang, H. Huang, C. Xiang et al., “Efficient attention methods: Hardwareefficient, sparse, compact, and linear attention.”

[57] L. Perko, Differential Equations and Dynamical Systems, 1991.

[58] M. Beck, K. Poppel, M. Spanring, A. Auer, O. Prudnikova, M. Kopp,¨ G. Klambauer, J. Brandstetter, and S. Hochreiter, “xlstm: Extended long short-term memory,” in Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024.

[59] A. J. Bagnall, J. Lines, A. Bostrom, J. Large, and E. J. Keogh, “The great time series classification bake off: a review and experimental evaluation of recent algorithmic advances,” Data Min. Knowl. Discov., vol. 31, no. 3, pp. 606–660, 2017.

[60] M. A. F. Pimentel, A. E. W. Johnson, P. Charlton, D. A. Birrenkott, P. J. Watkinson, L. Tarassenko, and D. A. Clifton, “Toward a robust estimation of respiratory rate from pulse oximeters,” IEEE Trans. Biomed. Eng., vol. 64, no. 8, pp. 1914–1923, 2017.

[61] P. Becker, H. Pandya, G. H. W. Gebhardt, C. Zhao, C. J. Taylor, and G. N. Gerhmann, “Recurrent kalman networks: Factorized inference in high-dimensional deep feature spaces,” in Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA, 2019, pp. 544–552.

[62] M. Menne, C. Williams, Jr., and R. Vose, “Long-term daily and monthly climate records from stations across the contiguous united states (u.s. historical climatology network),” 01 2016.

[63] I. Silva, G. B. Moody, D. J. Scott, L. A. Celi, and R. G. Mark, “Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012,” Computing in cardiology, vol. 39, pp. 245 – 248, 2012. [Online]. Available: https://api.semanticscholar.org/CorpusID:8678934

[64] A. Orvieto, S. L. Smith, A. Gu, A. Fernando, C. Gulcehre, R. Pascanu, and S. De, “Resurrecting recurrent neural networks for long sequences,” in Proceedings of the 40th International Conference on Machine Learning, 2023.

[65] J. Chung, C¸ . Gulc¸ehre, K. Cho, and Y. Bengio, “Empirical evaluation¨ of gated recurrent neural networks on sequence modeling,” CoRR, vol. abs/1412.3555, 2014.

[66] S. Zeng, F. Graf, and R. Kwitt, “Latent sdes on homogeneous spaces,” in Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023.

[67] B. Walker, A. D. McLeod, T. Qin, Y. Cheng, H. Li, and T. J. Lyons, “Log neural controlled differential equations: The lie brackets make A difference,” in Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, 2024, pp. 49 822– 49 844.

[68] E. D. Brouwer, J. Simm, A. Arany, and Y. Moreau, “Gru-ode-bayes: Continuous modeling of sporadically-observed time series,” in Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, H. M. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alche-Buc, E. B. Fox, and R. Garnett, Eds., 2019,´ pp. 7377–7388.

[69] S. N. Shukla and B. Marlin, “Multi-time attention networks for irregularly sampled time series,” in International Conference on Learning Representations, 2021.

[70] J. T. Smith, A. Warrington, and S. Linderman, “Simplified state space layers for sequence modeling,” in The Eleventh International Conference on Learning Representations, 2023.