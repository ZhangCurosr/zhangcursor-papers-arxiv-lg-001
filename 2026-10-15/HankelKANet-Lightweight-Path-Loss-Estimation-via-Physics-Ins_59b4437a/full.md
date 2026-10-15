# HankelKANet: Lightweight Path Loss Estimation via Physics-Inspired Hankel Convolution

Yue Ruan, Xiaoxiao Cheng, Yunan Zhu, Wenlong Huang, Meng Hua, Senior Member, IEEE

Abstract—Accurate path loss (PL) estimation is essential for wireless network planning and optimization. This paper proposes HankelKANet, a lightweight physics-inspired network for PL estimation that structurally embeds electromagnetic propagation characteristics into the neural architecture. Built on a single residual UNet backbone, the proposed model integrates urban layout features with a convolutional Kolmogorov-Arnold network (KAN). Specifically, the convolutional kernels are parameterized by the radial amplitudes of the zeroth- and first-order Hankel functions. This formulation acts as a physics-inspired nonlinear distance encoding, enabling the network to capture dominant attenuation characteristics directly without requiring additional physics-based loss terms. Experimental results demonstrate that HankelKANet achieves state-of-the-art accuracy and efficiency on both the simulated RadioMapSeer dataset and the real-world Huawei RSRPSet. Compared with the leading baseline, PEFNet, it reduces the parameter count by 78% and lowers the RMSE by 16.96% on the simulated data, while further exhibiting robust cross-dataset transferability and high data efficiency during target-domain adaptation. To interpret these empirical gains, we mathematically establish a parameter-conditional spatial gradient bound. Using discrete Dirichlet energy (DDE) and Jacobian sensitivity heatmaps, we empirically verify that this physical encoding intrinsically constrains the network’s spatial derivatives. This structural smoothness effectively eliminates non-physical spatial ripples, preventing overfitting to simulator artifacts and serving as the fundamental driver for robust realworld generalization.

Index Terms—path loss estimation, convolutional Kolmogorov-Arnold network, Hankel kernel, discrete Dirichlet energy, convolutional neural network.

## I. INTRODUCTION

P <sup>ATH</sup> <sup>Loss</sup> <sup>(PL)</sup> <sup>estimation</sup> <sup>is</sup> <sup>a</sup> <sup>fundamental</sup> <sup>task</sup> <sup>in</sup>wireless communications. Accurate and efficient PL pre- wireless communications. Accurate and efficient PL prediction plays a crucial role in the deployment and service quality assurance of 5G/6G systems, enabling critical functions such as network coverage planning, interference suppression, resource allocation, and performance optimization [1], [2], [3]. Consequently, developing effective and precise PL prediction schemes remains a key research priority in the field of wireless communications.

Traditional PL estimation methods, i.e., stochastic models such as 3GPP TR 36.873 [4] and TR 38.901 [5], which rely on empirical formulas derived from measurement data, are computationally efficient but lack adaptability to complex environments and offer limited accuracy. Deterministic models (e.g., ray tracing [6]) are capable of high-fidelity estimation by precisely simulating signal propagation paths; however, they incur extremely high computational complexity due to the modeling of numerous ray-environment interactions and cannot meet real-time requirements. Pure data-driven models, such as ANN [7] and RadioUNet [8], use deep learning to uncover data correlations. However, their reliance on training data distribution and lack of physical constraints remain inherent limitations, making estimation errors likely in unseen scenarios.

To overcome the limitations of the aforementioned approaches, physics-informed neural networks (PINNs), also referred to as physics-informed machine learning (PIML) <sup>2</sup>, offer an effective framework for integrating physical knowledge with data-driven learning [9], [10]. By embedding physical laws, such as partial differential equations (PDEs), into the loss function, PINNs guide the model training process under physical constraints, thereby balancing the flexibility of datadriven methods with the reliability of physics-based models [11], [12]. Building on this foundation, PEFNet [13] fuses the volume integral equation (VIE) of electromagnetic propagation with deep learning, employing a dual-UNet architecture to separately handle physics-driven and data-driven learning. This approach achieves a promising balance between generalization and efficiency, and serves as a representative hybrid-driven method.

Despite its merits, PEFNet’s complex architecture and hybrid loss function leave room for structural simplification and efficiency improvements. Recently, Kolmogorov-Arnold networks (KANs) [14] and their variants [15], [16] have offered a new way to embed physical priors by representing functions as linear combinations of learnable univariate functions. This structurally avoids the soft penalty terms required by PINNs. Notably, Hankel functions, the core components of the 2D free-space Green’s function for electromagnetic propagation [17], possess intrinsic nonlinearity. This makes them ideal learnable basis functions for a KAN-inspired, lightweight, and physics-aware design.

Motivated by this, we propose HankelKANet, an efficient architecture that directly embeds KAN-inspired learnable basis functions into convolutional kernels under the guidance of Hankel function priors.

The main contributions of this paper are summarized as follows:

• A Lightweight Physics-Inspired Architecture: We propose HankelKANet, a structurally streamlined model that processes a minimalist two-channel input (base station distance and building maps) via a single residual U-Net backbone equipped with a novel dual-convolution fusion module. This module seamlessly integrates standard spatial convolutions with Hankel-parameterized convolutions, intrinsically enforcing physical consistency without relying on complex hybrid loss functions.

• High Accuracy and Exceptional Efficiency: Evaluations on both the simulated RadioMapSeer and the realworld Huawei RSRPSet demonstrate that HankelKANet achieves state-of-the-art predictive performance. Compared to the leading baseline, PEFNet [13], our model reduces the parameter count by 78%, cuts training time by 26%, and improves inference speed by 22.2%, while simultaneously lowering the RMSE.

• Analytical Insights and Real-World Generalization: We derive a parameter-conditional spatial gradient bound to provide analytical insights into the model’s inherent stability. Supported by empirical validation via discrete Dirichlet energy (DDE) and Jacobian sensitivity heatmaps, we demonstrate that this structural bias inherently enforces spatial smoothness, thereby preventing overfitting to simulator artifacts and ensuring robust cross-domain generalizability in real-world scenarios.

The remainder of this paper is organized as follows. Section II reviews the methodological background. Section III presents the system model. Section IV details the proposed HankelKANet architecture. Section V reports the experimental results, including ablation and complexity studies. Section VI analyzes the model’s spatial smoothness and real-world generalization. Section VII concludes the paper.

## II. METHODOLOGY BACKGROUND

This section establishes the methodological background. We first review PEFNet to identify its architectural and optimization bottlenecks, motivating a structurally efficient alternative. We then introduce three core components: the Hankel-function representation of wave fields (physical prior), convolutional KANs (architectural framework), and geometric complexity. Together, they provide the theoretical and analytical foundation to ensure the model’s spatial smoothness, physical plausibility, and generalization.

## A. PEFNet Analysis

PEFNet [13] is a pioneering hybrid model for path loss estimation that integrates fundamental electromagnetic principles with deep learning. Its core innovation is incorporating the VIE via a two-stage, dual-UNet architecture. First, a knowledgedriven UNet processes a 4-channel input to estimate the total field under VIE constraints. Then, a data-driven UNet learns the residual against measured data. By infusing physical priors, PEFNet achieves a better balance of accuracy and efficiency than pure data-driven or deterministic models.

However, PEFNet’s design presents several challenges that limit its efficiency and broader applicability, directly motivating our work:

• Redundant Inputs: Using the real and imaginary parts of the incident field as separate channels requires complex numerical preprocessing. As these fields are highly correlated with the base station distance, this 4-channel redundancy increases computational overhead and the risk of overfitting.

• Cumbersome Architecture: The dual-UNet structure, with a maximum channel count of 1024 and approximately 118.38M parameters, incurs massive memory and computational costs, hindering deployment on resourceconstrained edge devices or large-scale real-time processing.

• Sensitive Hybrid Loss: The objective function $( { \mathcal { L } } \ =$ $\mathcal { L } _ { \mathrm { P H Y } } + \lambda \mathcal { L } _ { \mathrm { D A T } } )$ requires meticulous manual tuning of λ. Improper weighting often triggers gradient conflicts between physical constraints and data fitting, destabilizing convergence and degrading performance [18], [19], [20], [21].

To overcome these critical bottlenecks, Section IV introduces HankelKANet, a structurally regularized approach that inherently resolves the trade-off between physical fidelity, architectural efficiency, and optimization stability.

## B. Wave Field Representation via Hankel Functions

Under the 2D scalar wave approximation, the electromagnetic field $u ( { \pmb x } )$ at an observation point x, excited by a transmitter at $\pmb { x } _ { \mathrm { T x } }$ , is strictly governed by the following integral representation:

$$
u ( \pmb { x } ) = \underbrace { q _ { \mathrm { T x } } G _ { 0 } \left( \pmb { x } , \pmb { x } _ { \mathrm { T x } } \right) } _ { \mathrm { P r i m a r y ~ D i r e c t ~ F i e l d } } + \underbrace { \int _ { S } G _ { 0 } \left( \pmb { x } , \pmb { y } \right) J ( \pmb { y } ) \mathrm { d } \pmb { y } } _ { \mathrm { S e c o n d a r y ~ S c a t t e r e d ~ F i e l d } } ,\tag{1}
$$

where $G _ { 0 } ( \pmb { x } , \pmb { y } ) = \frac { \mathrm { i } } { 4 } H _ { 0 } ^ { ( 1 ) } \left( k _ { 0 } \| \pmb { x } - \pmb { y } \| _ { 2 } \right)$ is the free-space Green’s function with the background wavenumber $k _ { 0 }$ , and $J ( y )$ represents the equivalent induced sources over the scatterer boundaries S.

This decomposition explicitly highlights the fundamental role of Hankel functions in wave propagation [22]. The primary direct field (the first term) is dictated by the zerothorder Hankel function of the first kind, $H _ { 0 } ^ { ( 1 ) }$ . Physically, $H _ { 0 } ^ { ( 1 ) }$ models an isotropic, radially symmetric cylindrical wave, whose magnitude governs the fundamental geometric energy spreading in free space. Furthermore, the local spatial gradient of this primary wave is analytically characterized by the first-order mode $H _ { 1 } ^ { ( 1 ) }$ , based on the derivative relationship $\begin{array} { r } { \frac { \mathrm { d } } { \mathrm { d } z } H _ { 0 } ^ { ( 1 ) } ( z ) = - H _ { 1 } ^ { ( \mathrm { i } ) } ( z ) } \end{array}$ . Conversely, the integral term rep resenting the secondary scattered field accounts for complex phenomena such as scattering and diffraction from environmental boundaries. In classical analytical methods, evaluating this integral requires expanding the field into an infinite series of higher-order angular harmonics (multipole modes, $H _ { n } ^ { ( 1 ) }$ for $n \geq 2 )$ , which capture the intricate angular dependencies and directional fluctuations induced by environmental obstacles [23].

While Eq. (1) provides an exact boundary-value formulation, explicitly evaluating the full scattering integral and its corresponding high-order multipole expansions over electrically large urban layouts is computationally intractable. However, this rigorous physical decomposition successfully separates the unperturbed primary radial propagation from the complex secondary angular scattering. It naturally lays the theoretical foundation for extracting a highly structured, physicsinspired spatial prior for macroscopic channel modeling, as detailed in Section III-A.

## C. Kolmogorov-Arnold Network

Inspired by the Kolmogorov-Arnold representation theorem [24], Kolmogorov-Arnold Networks (KANs) replace the static linear weights of traditional neural networks with learnable univariate functions [14]. Instead of simple scalar multiplications, each connection in a KAN applies a non-linear function parameterized by a specific basis family. To construct these mappings, researchers have investigated a rich variety of bases, encompassing splines [25], frequency-domain representations like Fourier series [26] and wavelets [27], as well as radial [28] and polynomial [29] functions.

To efficiently process spatial data, Convolutional KANs extend this paradigm by embedding learnable univariate functions into local convolution kernels [30]. For an input feature map I and a $K \times K$ kernel, the convolution operation is defined as:

$$
( \pmb { I } * \Phi ) _ { i , j } = \sum _ { p = 1 } ^ { K } \sum _ { q = 1 } ^ { K } \varphi _ { p , q } ( \pmb { I } _ { i + p - 1 , j + q - 1 } ) ,\tag{2}
$$

where $\boldsymbol { \Phi } ~ = ~ \{ \varphi _ { p , q } \}$ represents a set of learnable univariate basis functions associated with each spatial position $( p , q )$ in the kernel.

By replacing fixed kernel weights with analytical basis evaluations, Convolutional KANs provide a flexible and expressive framework. More importantly for physics-aware tasks, this architecture allows for the direct embedding of specialized physical priors, such as electromagnetic wave propagation templates, into the fundamental convolution operation.

## D. Geometric Complexity and Jacobian Sensitivity

In the theoretical analysis of neural networks, spatial smoothness and generalization capabilities are closely tied to the network’s gradient with respect to its inputs. For a learned mapping $f _ { \theta } ,$ , its microscopic sensitivity to local input perturbations is characterized by the Jacobian matrix, $\mathbf { J } ( \mathbf { x } ) =$ $\nabla _ { \mathbf { x } } f _ { \pmb { \theta } } ( \mathbf { x } )$ . The Frobenius norm of this matrix, $\left\| \mathbf { J } ( \mathbf { x } ) \right\| _ { F }$ , quantifies the local rate of change at a specific spatial coordinate x. A bounded Jacobian norm indicates a stable and continuous local mapping, whereas excessively large values imply abrupt, high-frequency spatial fluctuations.

To evaluate the overall smoothness of a model from a macroscopic perspective, geometric analysis utilizes the Discrete Dirichlet Energy (DDE) [31]. The DDE quantifies the global geometric complexity by aggregating the squared local Jacobian norms over an entire dataset D:

$$
\langle f _ { \theta } , D \rangle = \frac { 1 } { | D | } \sum _ { \mathbf { x } \in D } \| \nabla _ { \mathbf { x } } f _ { \theta } ( \mathbf { x } ) \| _ { F } ^ { 2 } .\tag{3}
$$

Originating from harmonic function theory [32], [33], the DDE measures the average functional variation across the input distribution. Minimizing this energy encourages networks to find solutions with minimal intrinsic stretching, which has been theoretically linked to improved generalization [31]. Recently, this concept has also been adopted to analyze the functional properties and training dynamics of KAN architectures [16].

In the context of wireless communications, these mathematical properties carry direct physical significance. Realworld wave propagation, including macroscopic path loss and diffraction, exhibits inherent spatial coherence. Consequently, physically plausible radio maps naturally demand spatial continuity. A neural network that generates highly irregular spatial gradients, which mathematically manifest as localized Jacobian hotspots and an anomalously high global DDE, fundamentally violates electromagnetic principles. Such behavior suggests that the network is fitting non-physical spatial noise instead of the actual wave propagation laws.

## III. SYSTEM MODEL

This section formulates the map-based path loss estimation problem in urban wireless propagation scenarios. We first introduce the simplified large-scale channel model, followed by the map-based scenario representation and the learning objective.

## A. Large-Scale Channel and Path Loss Model

We now formalize the connection between the fundamental wave field representation in Section II-B and the macroscopic radio channel modeling. In wireless communications, considering a transmitter (Tx) at $\mathbf { p } _ { \mathrm { { T x } } } \in \mathbb { R } ^ { 3 }$ and a receiver (Rx) at $ { \mathbf { p } } \in \mathbb { R } ^ { 3 }$ , the baseband multi-path channel is typically represented as a superposition of L physical propagation paths:

$$
\mathcal { H } ( \mathbf { p } ; t , f ) = \sum _ { l = 1 } ^ { L } c _ { l } ( \mathbf { p } ) e ^ { j \psi _ { l } } e ^ { j 2 \pi \nu _ { l } t } e ^ { - j 2 \pi f \tau _ { l } } ,\tag{4}
$$

where $c _ { l } ( \mathbf { p } ) \in \mathbb { C } , \psi _ { l } , \tau _ { l }$ , and $\nu _ { l }$ denote the complex amplitude, carrier-induced phase, delay, and Doppler shift of the l-th path, respectively. Physically, the complex amplitude $c _ { l } ( \mathbf { p } )$ of each path is the macroscopic manifestation of the primary direct field and the secondary scattered field integrations defined in Eq. (1).

Since the carrier phase ψ<sub>l</sub> is highly sensitive to distance variations on the scale of a wavelength, {ψ<sub>l</sub>} can be modeled as i.i.d. uniform random variables on [0, 2π). By averaging out these local phase fluctuations, Doppler shifts, and smallscale spatial variability, the large-scale power gain becomes $\begin{array} { r } { \beta ( \mathbf { p } ) = \mathbb { E } [ | \mathcal { H } ( \mathbf { p } ) | ^ { 2 } ] = \sum _ { l = 1 } ^ { L } \mathbb { E } [ | c _ { l } ( \mathbf { p } ) | ^ { 2 } ] } \end{array}$ , as cross-terms vanish due to phase independence. The targeted large-scale PL is thus defined as:

$$
\operatorname { P L } ( \mathbf { p } ) = - 1 0 \log _ { 1 0 } \beta ( \mathbf { p } ) ,\tag{5}
$$

which inherently represents a spatial local average, strictly excluding small-scale fast fading by design.

To make urban path loss modeling computationally tractable at the city scale, we introduce two strategic simplifications in our network design based on the theoretical decomposition in

Eq. (1). First, we adopt a widely used 2D spatial formulation $( \pmb { x } \in \mathbb { R } ^ { 2 } )$ . This is justified by the practical observations that the Tx is mounted at a fixed height with Rx points on a horizontal user-height plane, and that urban obstacles (e.g., buildings) are well-approximated as vertically extruded prisms. However, even under this 2D approximation, explicitly evaluating the full boundary integration for secondary scattering remains computationally prohibitive. Therefore, instead of a brute-force numerical integration, our architecture relies on a physical decoupling strategy: we mathematically evaluate the Hankel functions on the absolute transmitter-to-receiver distance map $( \lVert \pmb { x } - \pmb { x } _ { \mathrm { T x } } \rVert )$ to isolate the envelope of the primary direct field as a global spatial prior. The complex secondary scattering and diffraction interactions are then left to be implicitly learned by a parallel network branch operating on the building layouts.

Building upon this strategy, we construct a low-order propagation prior specifically tailored for large-scale PL. As illustrated in Fig. 1, we approximate the spatial prior using exclusively the magnitudes of the zeroth- and first-order Hankel functions based on the following physical considerations:

• Macroscopic Energy Decay: The magnitude of the zeroth-order mode $( | H _ { 0 } ^ { ( 1 ) } | )$ directly models the baseline geometric spreading of the primary wave, establishing the fundamental distance-dependent decay envelope.

• Dominant Spatial Gradients: The magnitude of the first-order mode $( | H _ { 1 } ^ { ( 1 ) } | )$ captures the dominant radial gradients of the primary wave. Supplying these gradients as a spatial prior equips the neural network to effectively learn abrupt macroscopic power transitions, such as shadow boundaries and edge diffraction induced by the building layouts.

• Negligibility of Higher-Order Modes: Higher-order modes $( n \geq 2 )$ represent highly oscillatory multi-bounce scattering. While driving small-scale fading, these rapid fluctuations are intrinsically smoothed out by the local spatial averaging in Eq. (5), making their macroscopic impact negligible.

• Omission of the Phase Argument: Because PL focuses solely on the averaged power envelope rather than the coherent superposition of fields, the phase argument of the Hankel functions becomes redundant for describing large-scale fading.

By discarding the phase arguments and higher-order scattering terms, we extract a highly structured, physics-inspired, and computationally lightweight spatial prior. The rigorous empirical support for this theoretical simplification is extensively detailed in the ablation study in Section V-C2, which demonstrates that omitting the phase and higher-order modes does not degrade the large-scale PL prediction accuracy.

## B. Map-Based Scenario Representation

Building on the 2D formulation established above, the region of interest is discretized into a regular two-dimensional grid of size $H \times W$ . A grid cell $( x , y )$ corresponds to the physical location $\mathbf { r } _ { x , y }$ on the Rx-height plane, and the ground-truth path loss map is defined as $\mathbf { Y } \in \mathbb { R } ^ { H \times W }$ with $\mathbf { Y } ( x , y ) = P L ( \mathbf { r } _ { x , y } )$

To characterize the propagation environment, we construct two spatial matrices:

• Building Layout Map $( B \in \{ 0 , 1 \} ^ { H \times W } ) \colon$ A binary mask where $B ( x , y ) = 1$ indicates a building (blockage) and $B ( x , y ) = 0$ indicates free space.

• Distance-Prior Map $( R \in \mathbb { R } ^ { H \times W } ) \colon$ Encodes the geometric distance from the transmitter, defined as $R ( x , y ) =$ $\| \mathbf { r } _ { x , y } - \mathbf { r } _ { \mathrm { T x } } \| _ { 2 }$ . In practice, this map is min-max normalized into [0, 1].

The input representation of a single propagation scenario is the channel-wise concatenation of these two maps:

$$
\begin{array} { r } { X = \mathrm { C o n c a t } ( B , R ) \in \mathbb { R } ^ { 2 \times H \times W } . } \end{array}\tag{6}
$$

Thus, the path loss estimation is cast as a two-channel-toone-channel image translation problem, learning a mapping ${ \widehat { Y } } = F _ { \theta } ( X )$ parameterized by network weights θ.

## C. Learning Objective

Given a training dataset $\boldsymbol { \mathcal { D } } = \{ ( B _ { i } , R _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { N } ,$ the goal is to optimize the model parameters θ to accurately predict the path loss maps. Since path loss estimation is essentially a pixel-wise continuous regression problem, the mean squared error (MSE) is adopted as the training objective:

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { H W } \sum _ { x = 1 } ^ { H } \sum _ { y = 1 } ^ { W } \left( \widehat { Y } ( x , y ) - Y ( x , y ) \right) ^ { 2 } .\tag{7}
$$

Therefore, the final learning objective is

$$
\operatorname* { m i n } _ { \theta } \ \mathbb { E } _ { ( B , R , Y ) \sim \mathcal { D } } \mathcal { L } _ { \mathrm { M S E } } .\tag{8}
$$

## IV. PROPOSED METHOD

In this section, we present the architecture of the proposed HankelKANet that realizes the end-to-end mapping defined in Section III. The model is designed to combine environmental structural information and propagation-related priors within a compact encoder–decoder framework.

## A. Overall Architecture

HankelKANet adopts a UNet-based encoder–decoder backbone for end-to-end path loss prediction from the two-channel input $\{ B , R \}$ . This design is motivated by the nature of the task as an image-to-image regression problem, where both the input and output are two-dimensional spatial maps. The encoder progressively extracts hierarchical features and enlarges the receptive field, while the decoder restores spatial resolution and recovers fine-grained details through skip connections.

As illustrated in Fig. 2, the proposed network consists of two parallel components in the encoding stage: an environment branch for extracting scene-structure features from the building layout map, and a physics-prior branch for extracting propagation-aware features from the distance-prior map. The features from the two branches are fused at multiple scales and then passed to the decoder to generate the final path loss map. The entire model is built on a single residual UNet backbone with a maximum channel width of 512, resulting in a lightweight yet expressive architecture.

(a)  
![](images/4f610cf2e6bedd5f5ee2657ca57fc15de038eed6611ec92235fec2d835744ef5.jpg)

![](images/a2ace0500c8a2961888a5ca1458561874e2c0709af694c16134ebf242cc0dd37.jpg)  
(b)

![](images/5babe9c333a6695697594e8e51759f453554b2064934be3e058e16c093791d18.jpg)  
(c)

Fig. 1. Electromagnetic interpretation of the structured Hankel prior for large-scale path loss. (a) A typical urban scenario exhibiting sharp PL contrasts between dense buildings and open areas. (b) The zeroth- and first-order Hankel magnitudes capture macroscopic cylindrical spreading and fundamental spatial gradients, whereas highly oscillatory higher-order modes (n ≥ 2) correspond to fine-grained scattering that averages out in large-scale PL. (c) Visualization of the combined effect of the zeroth- and first-order Hankel magnitudes extracted as the spatial prior.  
![](images/186211f61632b6941f4eb0f54755cf708b6da13a13e00d8960219f360bbe1449.jpg)  
Fig. 2. Overall architecture of the proposed HankelKANet. The network processes a streamlined two-channel input comprising a base station distance map (R Map) and a building binary map. It is built upon a single residual UNet backbone with a maximum channel count of 512. The encoder-decoder structure utilizes skip connections to recover fine details in boundary and blockage regions, while the physical branches (PhyBlocks) embed distance-dependent propagation priors via 2D HankelKAN convolutions. The network finally outputs the predicted PL map.

## B. Dual-Branch Encoder Design

1) Environment Branch: The environment branch takes the building layout map B as input and extracts structural representations induced by the urban geometry. It is implemented by a hierarchical convolutional encoder that progressively downsamples the spatial resolution while increasing the channel dimension. In this way, the network captures both local building boundaries and broader scene-level contextual information. The intermediate features are preserved and forwarded to the decoder through skip connections for later detail recovery.

2) Physical Branch: The physical branch operationalizes the propagation prior established in Section III-A by explicitly evaluating the primary field envelope. It takes a dimensionless distance map $\widetilde { R } = R / L _ { \mathrm { r e f } } \in [ 0 , 1 ]$ as input, where R is the physical Tx-Rx distance and $L _ { \mathrm { r e f } }$ is the map’s diagonal length.

Crucially, to prevent the Hankel functions from diverging at their singular point (R = 0), we lower-bound the evaluated argument as $z = \operatorname* { m a x } \Bigl \{ \kappa \widetilde { R } , z _ { \mathrm { m i n } } \Bigr \}$ . Here, κ is a dimensionless learnable radial-frequency parameter, bounded in [0.1, 10.0] via a sigmoid mapping and initialized to π to ensure numerical stability. The threshold $z _ { \mathrm { m i n } } = \pi \times 1 0 ^ { - 3 }$ restricts evaluation to a physically negligible sub-pixel region $( \widetilde R ~ \ge ~ 1 0 ^ { - 3 } )$ preventing gradient explosion caused by the divergence of the Neumann functions during backpropagation without distorting macroscopic features.

This regularized map is then processed by the proposed Hankel-KAN convolutional layer. Given an input feature map X and a kernel of size $m \times n$ , the Conv-KAN operation is defined as:

$$
( \mathbf { X } * \varphi ) _ { i , j } \ = \ \sum _ { u = 1 } ^ { m } \sum _ { v = 1 } ^ { n } \varphi _ { u , v } ( \mathbf { X } _ { i + u - 1 , j + v - 1 } ) ,\tag{9}
$$

where each $\varphi _ { u , v } ( \cdot )$ is a learnable univariate function associated with the spatial position $( u , v )$ . Unlike standard CNNs that

apply simple scalar multiplications, Conv-KAN endows each kernel entry with an analytical, physics-inspired basis function.

Specifically, we parameterize each $\varphi _ { u , v } ( z )$ using a combination of the zeroth- and first-order Hankel responses:

$$
\varphi _ { u , v } ( z ) = w _ { 0 , u , v } \phi _ { 0 } ( z ) + w _ { 1 , u , v } \phi _ { 1 } ( z ) ,\tag{10}
$$

where $w _ { 0 , u , v } , w _ { 1 , u , v } ~ \in ~ \mathbb { R }$ are trainable modal weights that adaptively fuse the macroscopic energy decay and the dominant spatial gradients of the primary wave. The parameter κ (embedded within z) allows the network to dynamically adjust the fundamental spatial decay rate to varying propagation environments.

Following the magnitude-only envelope extraction established in Section III-A, the radial basis functions $\phi _ { \nu } ~ ( \nu ~ \in$ {0, 1}) are explicitly constructed as log-compressed magnitudes. This nonlinear mapping effectively compresses the massive dynamic range of the electromagnetic field:

$$
\phi _ { \nu } ( z ) = \mathrm { l o g } \Big ( 1 + | H _ { \nu } ^ { ( 1 ) } ( z ) | \Big ) = \mathrm { l o g } \Big ( 1 + \sqrt { J _ { \nu } ( z ) ^ { 2 } + Y _ { \nu } ( z ) ^ { 2 } } \Big )\tag{11}
$$

Consequently, the physical branch yields a structured spatial prior of the unperturbed primary field. These macroscopic features are subsequently fused with the environment branch to implicitly resolve complex secondary scattering interactions.

## C. Multi-Scale Feature Fusion and Decoder

The features extracted by the environment and physical branches are fused at four progressive scales. At the first scale, the building layout map $B$ and the distance-prior map $\widetilde { R }$ are processed by an initial environment block (EnvStem) and a physical block (PhyStem), respectively. The feature extraction at this initial scale utilizes 3×3 convolution kernels, producing the corresponding 32-channel feature maps $E _ { 1 }$ and $P _ { 1 }$ . These two feature maps are then concatenated and passed through a fusion layer, which can be written as

$$
F _ { 1 } = \mathcal { F } _ { 1 } ( [ E _ { 1 } ; P _ { 1 } ] ) , \qquad \mathcal { F } _ { 1 } ( \cdot ) = \mathrm { C o n v } _ { 1 \times 1 } ^ { ( 1 ) } ( \cdot ) ,\tag{12}
$$

where $[ \cdot ; \cdot ]$ denotes channel-wise concatenation. After this $1 \times 1$ fusion, a double-convolution block, consisting of a $3 \times 3$ convolution and a $5 \times 5$ convolution (each followed by batch normalization and SiLU activation), is applied to further transform the joint representation and expand the channel dimension to 64. The resulting feature $F _ { 1 }$ is then downsampled and forwarded to the next scale.

For scales $s \ = \ 2 , 3 , 4$ , the environmental features $E _ { s }$ are obtained by downsampling the fused features from the previous stage. Concurrently, the physical branch performs scale-matched average pooling on the original distance-prior map $\widetilde { R }$ and then applies Hankel-KAN convolutions (PhyBlock) to generate the propagation-aware features $P _ { s }$ . Specifically, the physical branch dynamically adapts its kernel sizes, using a $3 \times 3$ Hankel-KAN convolution at the second scale, and larger $5 \times 5$ Hankel-KAN convolutions at the third and fourth scales, producing 64-, 128-, and 256-channel feature maps, respectively. At each scale, the environmental and physical features are concatenated and fused through $\textbf { a } 1 \times 1$ convolution, followed again by the aforementioned double-convolution block (i.e., one $3 \times 3$ and one $5 \times 5$ convolution) to refine the representation and double the channel dimension. In this way, the fused features progressively encode both scene-structure information and propagation-aware priors across multiple resolutions.

The fused representations at different scales are not only passed to deeper encoder stages, but also delivered to the decoder through skip connections. The decoder is symmetric to the encoder: it progressively upsamples the bottleneck feature and concatenates it with the corresponding encoder outputs to recover spatial resolution and reconstruct fine-grained path loss structures, especially around blockage boundaries and transition regions. As the spatial resolution increases, the channel dimension is reduced accordingly. Finally, a $1 \times 1$ convolution maps the decoded features to a single-channel continuous-valued output, yielding the predicted path loss map.

## <sup>.</sup>D. End-to-End Training Strategy

By embedding propagation-related priors directly into the network architecture through Hankel-KAN convolutions, the proposed method can be trained end-to-end using only the MSE objective (Eq. 8). This design avoids the additional loss terms and manual coefficient tuning often required in hybrid formulations, while still enabling the model to capture both distance-dependent attenuation and environment-related spatial variations within a unified framework. The overall training procedure is summarized in Algorithm 1.

## V. EXPERIMENTAL VALIDATION

This section evaluates the predictive accuracy, architectural contributions, and computational cost of HankelKANet. RadioMapSeer [34] serves as the primary simulation benchmark, while Huawei RSRPSet [35] provides a complementary assessment of predictive performance on real-world measurements. All experiments in this section follow a within-dataset protocol, with models trained, selected, and tested using the respective partitions of each dataset. Synthetic-to-real transfer and cross-dataset generalization are examined separately in Section VI.

## A. Experimental Setup

1) Dataset Description: RadioMapSeer is a ray-tracing benchmark for urban PL prediction comprising 700 scenarios, each with 80 transmitter locations (inputs/outputs shown in Fig. 3). To ensure generalization, the dataset is split at the scenario level, holding out 8,000 maps for testing. PL values are normalized to [0, 1] during training and converted back to dB for evaluating valid outdoor pixels. All baselines use identical partitions and evaluation masks.

Huawei RSRPSet contains real-world received power measurements, capturing practical propagation effects and measurement noise often missed by simulations. Here, we specifically compare HankelKANet against the physics-guided PEFNet. Both models are trained from scratch under matched data budgets and evaluated on a common held-out test set, computing errors strictly at valid measurement locations.

![](images/96663ad21fdcd43af2904a80977d93b682eee378ea48f41540d999402aa30704.jpg)  
Fig. 3. Examples of input and output maps from RadioMapSeer. From left to right: building layout, distance prior, and pathloss map for two representative scenarios.

2) Evaluation Metrics: We evaluate prediction performance using mean absolute error (MAE), root mean square error (RMSE), and the coefficient of determination $( R ^ { 2 } )$ . Both MAE and RMSE are reported in dB after inverse normalization. Specifically, MAE reflects the average absolute deviation, RMSE imposes a heavier penalty on large prediction errors, and a higher $R ^ { 2 }$ signifies better overall agreement with the reference values.

For a single test path loss map, the MAE, RMSE, and $R ^ { 2 }$ are computed over its valid evaluation pixels as follows:

$$
\mathrm { M A E } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | \mathrm { P L } _ { i } - \hat { \mathrm { P L } _ { i } } | ,\tag{13a}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { m } \sum _ { i = 1 } ^ { m } ( \mathrm { P L } _ { i } - \hat { \mathrm { P L } } _ { i } ) ^ { 2 } } ,\tag{13b}
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { m } ( \mathrm { P L } _ { i } - \hat { \mathrm { P L } _ { i } } ) ^ { 2 } } { \sum _ { i = 1 } ^ { m } ( \mathrm { P L } _ { i } - \overline { { \mathrm { P L } } } ) ^ { 2 } } ,\tag{13c}
$$

where m is the number of valid evaluation pixels, $\mathrm { P L } _ { i }$ and $\hat { \mathrm { P L } _ { i } }$ denote the reference and predicted path loss values at pixel i, respectively, and $\begin{array} { r } { \overline { { \mathrm { P L } } } = \bar { \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \bar { \mathrm { P L } } _ { i } } \end{array}$ is the mean reference path loss over these pixels. The reported test-set results are obtained by averaging each metric across all test maps, with equal weight assigned to each map.

3) Training Settings and Reproducibility: All experiments are implemented in Python 3.12 and run on a single NVIDIA RTX 5880 Ada GPU with 48 GB memory. HankelKANet is optimized using AdamW with an initial learning rate of $4 \times 1 0 ^ { - 3 } .$ , a weight decay of $1 \times 1 0 ^ { - 4 }$ , cosine annealing, and a batch size of 32. The network is trained end-to-end using the MSE objective, without an additional physics-residual loss or manual weighting between data and physics losses. Training proceeds for up to 250 epochs, utilizing validationbased checkpoint selection and an early-stopping patience of 50 epochs.

To guarantee a strictly fair comparison, all baselines are reproduced within our unified evaluation pipeline to align perfectly with HankelKANet’s inputs and training budgets. Furthermore, all models are evaluated over five independent runs (seeds 0–4) using fixed dataset partitions, with results reported as mean ± standard deviation. Computational complexity is evaluated on the aforementioned GPU using a batch size of one, a $2 5 6 \times 2 5 6$ resolution, and FP32 precision. For reproducibility, source code and further implementation details are available at https://github.com/chengxx0317-lgtm/ HankelKANet.

## B. Comparison with Baselines

We compare HankelKANet with representative empirical, deterministic, and learning-based methods, including 3GPP TR 36.873 [4], ray tracing [6], decision tree regression (DTR) [36], k-nearest neighbors (KNN) [37], artificial neural networks (ANN) [7], RadioUNet [8], and PEFNet. Among these, PEFNet serves as the primary physics-guided baseline, allowing us to compare alternative ways of incorporating propagation information into a trainable prediction model.

Note that the ray-tracing baseline is evaluated as an independently executed solver rather than the ground-truth generator. For runtime comparisons of all learning-based models, we employ a standardized inference protocol on a single GPU with a 256×256 resolution and a batch size of 1. To isolate the pure network forward-pass latency, the reported times exclude data preprocessing and are averaged over 100 repetitions following 50 warm-up iterations.

1) Results on RadioMapSeer: Table I presents the quantitative comparison on the RadioMapSeer test set. Empirical and classical machine learning models (3GPP, DTR, KNN, ANN) exhibit limited accuracy, with RMSE values generally exceeding 10 dB, indicating their inability to capture fine-grained spatial variations. While the independent ray-tracing solver provides a much lower RMSE of 6.3250 dB, convolutional neural networks (RadioUNet, PEFNet) demonstrate superior performance by effectively learning spatial propagation priors from the training data.

Among all evaluated methods, HankelKANet achieves the highest $R ^ { 2 }$ (0.9849) and the lowest RMSE (1.8392 dB). Compared to the leading physics-guided baseline, PEFNet, our method reduces the RMSE by 16.96% (from 2.2148 to 1.8392 dB). Interestingly, its MAE (0.7980 dB) is marginally higher than that of PEFNet (0.7852 dB). Since RMSE penalizes large errors much more heavily than MAE, this specific performance signature, a significant drop in RMSE coupled with a comparable MAE, demonstrates that HankelKANet is highly effective at suppressing severe, catastrophic prediction errors (e.g., in deep shadow regions), rather than merely pursuing a uniform reduction in absolute error across easily predictable pixels.

Aggregate metrics can be dominated by easily predictable pixels (e.g., short-range LOS areas), masking prediction failures in heavily obstructed or distant regions that are critical for network planning. To rigorously evaluate model robustness, Table II provides a stratified analysis. To ensure objectivity, the stratification thresholds are strictly data-driven, derived from training-set percentiles for link distance (short ≤ 72.80 m, long $\geq 1 4 7 . 4 7 \mathrm { ~ m } )$ and building density $( \geq 3 0 . 8 0 \% )$ , while LOS/NLOS pixels are deterministically identified via geometric intersections.

Algorithm 1: Training procedure of HankelKANet   
Input : Training dataset $\overline { { \mathcal { D } _ { \mathrm { t r a i n } } } } ;$ validation dataset   
$\mathcal { D } _ { \mathrm { v a l } } ;$ Batch format: $( B , R , Y ) ;$ model $f _ { 0 } ;$   
number of training epochs $E ;$ learning rate $\eta .$   
Output: Best model parameters $\theta ^ { * }$   
1 Initialize model parameters $\theta$ and the optimizer;   
2 for epoch $ 1$ to $E$ do   
3 for each $( B , R , Y ) \in \mathcal { D } _ { t r a i n }$ do   
4 $E _ { 1 } \gets \mathrm { E n v S t e m } ( B ) ;$   
▷ Process building map B with EnvStem to   
get initial env features $E _ { 1 }$   
5 $P _ { 1 } \gets \mathrm { P h y S t e m } ( R ) ;$   
▷ Process distance map R with PhyStem to   
get initial phy features $P _ { 1 }$   
6 $F _ { 1 } \gets \mathrm { E n c } ( \mathrm { F u s e } ( E _ { 1 } , P _ { 1 } ) ) ;$   
▷ Fuse $E _ { 1 }$ and $P _ { 1 } ,$ then encode to get first   
fused features $F _ { 1 }$   
7 for $s \gets 2$ to $S$ do   
8 $E _ { s } \gets \mathrm { P o o l } ( F _ { s - 1 } ) ;$   
▷ Pool previous fused features $F _ { s - 1 } f o r$   
downsampling   
9 $P _ { s } \gets \mathrm { P h y B r a n c h } ( \mathrm { P o o l } ^ { ( s ) } ( R ) ) ;$   
▷ Multi-scale pool R, then phy branch   
to get $P _ { s }$   
10 $F _ { s } \gets \mathrm { E n c } ( \mathrm { F u s e } ( E _ { s } , P _ { s } ) ) ;$   
▷ Fuse $E _ { s }$ and $P _ { s } ,$ , then encode to get   
level s fused features $F _ { s }$   
11 $F _ { b } \gets \mathrm { B o t t l e n e c k } ( \mathrm { P o o l } ( F _ { S } ) ) ;$   
▷ Pool deepest fused features $F _ { S } ,$ then   
bottleneck to get $F _ { b }$   
12 $D _ { S } \gets \mathrm { D e c } ( \mathrm { C o n c a t } ( \mathrm { U p } ( F _ { b } ) , F _ { S } ) ) ;$   
▷ Upsample $F _ { b } ,$ concat with $F _ { S } ,$ then decode   
to get deepest decode features $D _ { S }$   
13 for $s  S$ to 2 do   
14 $D _ { s - 1 }  \mathrm { D e c } ( \mathrm { C o n c a t } ( \mathrm { U p } ( D _ { s } ) , F _ { s - 1 } ) ) ;$   
▷ Upsample current decode $D _ { s } ,$   
skip-connect with $F _ { s - 1 }$ , then decode   
to get $D _ { s - 1 }$   
15 $\hat { Y }  \mathrm { C o n v } _ { 1 \times 1 } ( D _ { 1 } ) ;$   
16 Compute loss $\mathcal { L } _ { \mathrm { M S E } }$ according to $\operatorname { E q . } 8 ;$   
17 Update θ using the optimizer based on $\mathcal { L } _ { \mathrm { M S E } } ;$   
18 Evaluate the model on $\mathcal { D } _ { \mathrm { v a l } } ;$   
19 if $\mathcal { L } _ { \nu a l } < \mathcal { L } _ { \nu a l } ^ { b e s t }$ then   
20 $\mathcal { L } _ { \mathrm { v a l } } ^ { \mathrm { b e s t } }  \mathcal { L } _ { \mathrm { v a l } } ;$   
21 $\theta ^ { * }  \theta ;$

As shown, HankelKANet maintains highly reliable estimates even in these adverse environments. While predictions in LOS and short-distance regions are remarkably accurate $\mathrm { ( R M S E } < 1 . 4 ~ \mathrm { d B ) }$ , the model demonstrates strong robustness in NLOS regions (RMSE 2.2506 dB) and high-density urban layouts (RMSE 2.2229 dB). Although long-range links remain the most challenging subset (3.1182 dB), these stratified results confirm that the low-order Hankel representation effectively captures complex composite propagation phenomena, preserving high accuracy even under severe obstruction.

2) Results on Huawei RSRPSet: To evaluate performance under real-world measured supervision, we deploy the models on the Huawei RSRPSet. In this section, we focus the comparison exclusively between HankelKANet and PEFNet— the strongest physics-guided baseline. Other empirical and classical machine learning methods yield substantially higher errors on this challenging real-world dataset and are therefore omitted for brevity.

Table III presents the within-dataset evaluation, comparing models trained from scratch under varying fractions of target data $( \eta )$ . HankelKANet consistently outperforms PEFNet in suppressing large prediction errors across all data regimes. Under extremely sparse supervision $( \eta ~ = ~ 1 \% )$ , our model reduces the RMSE by over 0.42 dB. With full supervision $( \eta =$ $1 0 0 \% ) .$ HankelKANet achieves the lowest RMSE (3.7926 dB) and the highest $R ^ { 2 }$ (0.9170). Notably, its MAE (1.7428 dB) is marginally higher than that of PEFNet (1.6865 dB). This specific performance signature, superior RMSE coupled with comparable MAE, is perfectly consistent with the observations on RadioMapSeer (Section V-B1). It reaffirms that the compact Hankel representation excels at mitigating severe outlier errors in noisy, measurement-based real-world data, rather than overfitting to uniform absolute deviations. Further analysis regarding cross-dataset generalization is deferred to Section VI.

## C. Ablation Studies on Architecture and Physical Priors

To systematically investigate the architectural components and the physical representation choices of HankelKANet, we conduct controlled evaluations on the RadioMapSeer benchmark.

1) Architectural Components: Table IV details the impact of removing or replacing key structural components.

Removing the physics-guided branch entirely produces a severe degradation, increasing the RMSE from 1.8392 dB to 9.5872 dB. This highlights the vital role of the transmitterdependent propagation prior. Similarly, removing the U-Net skip-connected module increases the RMSE to 4.0146 dB, confirming that the encoder-decoder feature fusion is crucial for recovering localized spatial details such as sharp shadowing transitions.

We also evaluate the specific parameterization of the physical branch. Replacing the Hankel amplitude basis with a standard B-spline basis increases the RMSE to 2.7958 dB, demonstrating that the Bessel-derived functions provide a more effective inductive bias for electromagnetic wave propagation. Furthermore, fixing the spatial scale parameter k instead of making it learnable increases the RMSE to 2.1594 dB. A trainable k allows the network to adaptively adjust the effective radial scale in the normalized feature domain, rather than being strictly bound to the physical free-space wavenumber.

TABLE I  
QUANTITATIVE PERFORMANCE COMPARISON OF DIFFERENT METHODS ON THE RADIOMAPSEER TEST SET.
<table><tr><td>Method</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE (dB)</td><td>RMSE (dB)</td></tr><tr><td>3GPP TR 36.873 [4]</td><td>0.2552</td><td>11.5416</td><td>14.7590</td></tr><tr><td>ray tracing [6]</td><td>0.8594</td><td>4.7591</td><td>6.3250</td></tr><tr><td>DTR [36]</td><td>0.4733 (± 0.0004)</td><td>8.7962 (± 0.0063)</td><td>12.1251 (± 0.0104)</td></tr><tr><td>KNN [37]</td><td>0.5808 (± 0.0002)</td><td>7.8994 (± 0.0095)</td><td>10.8649 (± 0.0053)</td></tr><tr><td>ANN [7]</td><td>0.6303 (± 0.0034)</td><td>7.4167 (± 0.0027)</td><td>10.1688 (± 0.0029)</td></tr><tr><td>RadioUNet [8]</td><td>0.9616 (± 0.0002)</td><td>1.4747 (± 0.0180)</td><td> $2 . 8 7 4 8 \ ( \pm 0 . 0 3 0 5 )$ </td></tr><tr><td>PEFNet [13]</td><td>0.9706 (± 0.0002)</td><td>0.7852 (± 0.0317)</td><td> $2 . 2 1 4 8 \ ( \pm 0 . 0 4 9 2 )$ </td></tr><tr><td>HankelKANet</td><td>0.9849 (± 0.0002)</td><td>0.7980 (± 0.0104)</td><td> $\mathbf { 1 . 8 3 9 2 \ ( \pm 0 . 0 3 0 0 ) }$ </td></tr></table>

TABLE II

STRATIFIED PERFORMANCE OF HANKELKANET WITH HANKEL ORDERS {0, 1} AGAINST RADIOMAPSEER REFERENCE MAPS.
<table><tr><td>Condition</td><td>Radio maps</td><td>Evaluated pixels</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE (dB)</td><td>RMSE (dB)</td></tr><tr><td>Overall outdoor test set</td><td>8,000</td><td>397,944,539</td><td>0.9849</td><td>0.7980</td><td>1.8392</td></tr><tr><td>LOS</td><td>8,000</td><td>125,925,669</td><td>0.9885</td><td>0.4614</td><td>1.3884</td></tr><tr><td>NLOS</td><td>8,000</td><td>272,018,870</td><td>0.9820</td><td>1.2583</td><td>2.2506</td></tr><tr><td>Short distance (≤ 72.80 m)</td><td>8,000</td><td>99,342,877</td><td>0.9910</td><td>0.3809</td><td>0.9827</td></tr><tr><td>Long distance (≥ 147.47 m)</td><td>8,000</td><td>99,980,657</td><td>0.9633</td><td>1.9792</td><td>3.1182</td></tr><tr><td>High building density (≥ 30.80%)</td><td>2,160</td><td>89,968,345</td><td>0.9715</td><td>1.1780</td><td>2.2229</td></tr></table>

TABLE III

WITHIN-DATASET PERFORMANCE COMPARISON ON THE HUAWEI RSRPSET (TRAINING FROM SCRATCH AND FULL REAL SUPERVISION).
<table><tr><td>Budget (η)</td><td>Method</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE (dB)</td><td>RMSE (dB)</td></tr><tr><td></td><td colspan="3">Training from Scratch</td></tr><tr><td>1%</td><td>PEFNet HankelKANet</td><td>0.6448 0.6896</td><td>4.7843 4.4375</td><td>7.8452 7.4218</td></tr><tr><td>5%</td><td>PEFNet</td><td>0.7620</td><td>3.5207</td><td>6.4211</td></tr><tr><td></td><td>HankelKANet PEFNet</td><td>0.7983 0.8340</td><td>3.2884 2.8129</td><td>5.9126 5.3628</td></tr><tr><td>10%</td><td>HankelKANet</td><td>0.8593</td><td>2.6792</td><td>4.9364</td></tr><tr><td rowspan="2">20%</td><td>PEFNet</td><td>0.8792</td><td>2.2736</td><td>4.5762</td></tr><tr><td>HankelKANet</td><td>0.8894</td><td>2.1846</td><td>4.3127</td></tr><tr><td></td><td>Full Real Supervision</td><td></td><td></td><td></td></tr><tr><td rowspan="2">100%</td><td></td><td></td><td></td><td></td></tr><tr><td>PEFNet HankelKANet</td><td>0.9117 0.9170</td><td>1.6865 1.7428</td><td>3.9124 3.7926</td></tr></table>

TABLE IV

ARCHITECTURAL COMPONENT ABLATION STUDY ON RADIOMAPSEER.
<table><tr><td>Ablation Variant</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>MAE (dB)</td><td>RMSE (dB)</td></tr><tr><td>w/o Physical Branch</td><td>0.6783</td><td>6.7468</td><td>9.5872</td></tr><tr><td>Hankel replaced by B-spline</td><td>0.8734</td><td>1.7899</td><td>2.7958</td></tr><tr><td>w/o skip-connected module</td><td>0.9445</td><td>2.1316</td><td>4.0146</td></tr><tr><td>with non-trainable k</td><td>0.8833</td><td>1.1926</td><td>2.1594</td></tr><tr><td>HankelKANet (Proposed)</td><td>0.9849</td><td>0.7980</td><td>1.8392</td></tr></table>

2) Hankel Orders and Angular Factors: To systematically determine the appropriate truncation orders and the necessity of explicit complex-field modeling, we compare varying Hankel configurations in Table V.

We evaluate subsets of radial orders I. Restricting the representation to a single mode (either {0} or {1}) degrades the RMSE noticeably (> 3.10 dB), indicating that both the primary radial decay and its rate-of-change baseline are highly beneficial. Conversely, adding higher-order modes (e.g., {0, 1, 2}) yields negligible accuracy gains (∆RMSE = −0.0016 dB) while increasing latency by 8.1%. Extending to {0, 1, 2, 3} actually degrades performance (1.8662 dB) due to parameter redundancy and optimization overhead on highfrequency oscillations. Thus, the joint low-order representation $\mathcal { T } = \{ 0 , 1 \}$ serves as a highly effective configuration.

We further examine whether complete Tx-centered angular factors (exp(inφ)) provide benefits over pure radial amplitudes. For each mode n, we expand the complete real and imaginary channels using Bessel functions combined with trigonometric angular terms. As shown in Table V, explicit angular coordinates yield marginal accuracy improvements $( \leq ~ 0 . 0 0 3$ dB for $| n | \leq 1 )$ but increase runtime by up to 28%. Because environmental blockage primarily originates from localized building edges rather than circular wavefronts centered at the transmitter, the parallel convolutional branch already extracts these angular variations efficiently. Furthermore, since the supervision target is scalar signal power rather than complex field data, explicit phase reconstruction is largely redundant.

Ultimately, these ablations empirically support that radial amplitudes with $\mathcal { T } = \{ 0 , 1 \}$ without explicit phase modeling offers a favorable trade-off between expressive fidelity, physical adherence, and computational efficiency.

## D. Complexity and Efficiency Analysis

To evaluate computational efficiency, we benchmark HankelKANet against representative learning-based baselines under strictly identical inference conditions, as detailed in Table VI. Compared to the primary physics-guided baseline, PEFNet (evaluated via an exact local reproduction to ensure matching conditions), HankelKANet reduces the parameter count by 78% (to 26.13 M) and FLOPs by 81.9% (to 91.58 G). Peak inference memory is also reduced by over 50% (from 720 MB to 340 MB), significantly lowering the barrier for deployment in resource-constrained edge environments.

Crucially, the end-to-end latency must fully account for the generation of the physical prior. Evaluating the Hankel functions online takes 0.285 s per map. Even including this pre-processing step, HankelKANet achieves a total inference latency of 1.110 s/map (0.901 maps/s). This remains strictly faster than both RadioUNet (1.396 s) and PEFNet (1.426 s) measured under the exact same unbatched protocol. Furthermore, HankelKANet reduces total training time on RadioMapSeer to 7.22 hours, compared to 9.77 hours for PEFNet. Given the nearly 3-hour per-map simulation cost of traditional ray tracing, HankelKANet offers an overwhelming efficiency advantage while simultaneously attaining superior predictive accuracy.

Ultimately, these evaluations confirm that HankelKANet attains superior predictive accuracy and an overwhelming efficiency advantage. Notably, this performance is validated within a well-defined operating regime targeting macroscopic large-scale path loss for 2D outdoor urban scenarios at sub-6 GHz frequencies. Given link distances up to ∼350 m, this macroscopic context naturally justifies the truncation to low-order Hankel modes. Having established the empirical capabilities and operational bounds of the proposed architecture, Section VI proceeds to analyze its learned spatial representations and cross-dataset generalization.

## VI. IMPACT OF SPATIAL SMOOTHNESS ON EMPIRICAL GENERALIZATION

While the previous section established HankelKANet’s superior accuracy and lightweight nature, this section uncovers the underlying mechanisms driving this performance. Our analysis serves two primary objectives: First, to explain the root of this excellence by tracing it to physics-inspired inductive biases that inherently enforce spatial smoothness; Second, to demonstrate how this physics-guided regularization prevents overfitting to simulator artifacts, thereby enhancing crossdataset generalization from simulated environments to realworld measurements.

## A. Hankel Operators as an Inductive Bias for Smoothness

Generic activation functions (e.g., ReLU) and standard KAN basis functions (e.g., B-splines) lack physics-inspired decay constraints. Consequently, as demonstrated in the literature on the spectral bias of neural networks [38], these unconstrained models are highly susceptible to fitting high-frequency noise, making them prone to fabricating non-physical spatial oscillations in unobserved regions. In contrast, the gradient of the Hankel basis function is intrinsically guided by the asymptotic envelope of cylindrical wave equations, naturally mitigating such artifacts.

Consider a single-layer Hankel-KAN mapping acting on a local spatial displacement vector $\mathbf { r } ^ { 3 }$ :

$$
\varphi ( \mathbf { r } ) = w _ { 0 } \phi _ { 0 } ( \kappa \| \mathbf { r } \| _ { 2 } ) + w _ { 1 } \phi _ { 1 } ( \kappa \| \mathbf { r } \| _ { 2 } ) ,\tag{14}
$$

where $\phi _ { \nu } ( z ) = \log ( 1 + | H _ { \nu } ^ { ( 1 ) } ( z ) | ) ( \nu \in \{ 0 , 1 \} )$ $w _ { 0 } , w _ { 1 } ~ \in$ R are trainable modal weights, and $\kappa \in \mathbb R ^ { + }$ is a learnable wavenumber.

Proposition 1 (Parameter-Conditional Spatial Gradient Bound): For any spatial position r in the propagation regime $( \kappa \| \mathbf { r } \| _ { 2 } \gg 1 )$ , the spatial gradient magnitude of $\varphi ( \mathbf { r } )$ is conditionally bounded by the modal parameters:

$$
\| \nabla _ { \mathbf { r } } \varphi ( \mathbf { r } ) \| _ { 2 } \leq \mathcal { C } \cdot \frac { \kappa ^ { 1 / 2 } } { \| \mathbf { r } \| _ { 2 } ^ { 1 / 2 } } \big ( | w _ { 0 } | + | w _ { 1 } | \big ) ,\tag{15}
$$

where ${ \mathcal { C } } = { \sqrt { 2 / \pi } }$ is a constant governed by the 2D cylindrical wave expansion.

Proof: Let $z ~ = ~ \kappa \| \mathbf { r } \| _ { 2 }$ . By the chain rule, $\nabla _ { \mathbf { r } } \varphi ( \mathbf { r } ) \ =$ $( d \varphi / d z ) ( \kappa \mathbf { r } ^ { \top } / \| \mathbf { r } \| _ { 2 } )$ , yielding $\| \nabla _ { \mathbf { r } } \varphi ( \mathbf { r } ) \| _ { 2 } = \kappa | d \varphi / d z |$ . For each basis $\phi _ { \nu } ( z ) = \log ( 1 + | H _ { \nu } ^ { ( 1 ) } ( z ) | )$ , using the contraction inequality and dynamic compression $( 1 + | H _ { \nu } ^ { ( 1 ) } ( z ) | ) ^ { - 1 } \leq 1$ we obtain: $| d \phi _ { \nu } / d z | \leq | d H _ { \nu } ^ { ( 1 ) } ( z ) / d z |$ . Using the Bessel recurrence relations $l H _ { 0 } ^ { \mathrm { ( i ) } } ( z ) / d z = - H _ { 1 } ^ { \mathrm { ( 1 ) } } ( z )$ and $d H _ { 1 } ^ { ( 1 ) } ( z ) / d z =$ $\begin{array} { r } { \frac { 1 } { 2 } [ H _ { 0 } ^ { ( 1 ) } ( z ) - H _ { 2 } ^ { ( 1 ) } ( z ) ] } \end{array}$ , applying the triangle inequality yields:

$$
\| \nabla _ { \mathbf { r } } \varphi ( \mathbf { r } ) \| _ { 2 } \leq \kappa \Big ( | w _ { 0 } | \big | H _ { 1 } ^ { ( 1 ) } \big | + \frac { 1 } { 2 } | w _ { 1 } | \big ( \big | H _ { 0 } ^ { ( 1 ) } \big | + \big | H _ { 2 } ^ { ( 1 ) } \big | \big ) \Big ) .\tag{16}
$$

Substituting the leading-order asymptotic decay $| H _ { \nu } ^ { ( 1 ) } ( z ) \rangle \sim$ $\sqrt { 2 / ( \pi z ) }$ for $\nu ~ \in ~ \{ 0 , 1 , 2 \}$ into Eq. (16) directly yields Eq. (15).

Remark (Physics-Inspired Structural Regularization): Eq. (15) reveals a fundamental property: the spatial gradient naturally decays at a rate of $\mathcal { O } ( { \| \mathbf { r } \| } _ { 2 } ^ { - 1 7 / 2 } )$ . Functioning as a physics-inspired distance encoding, this inherent asymptotic envelope acts as a structural inductive bias, preventing the network from fabricating high-frequency spatial noise (spurious gradients) in the far field.

## B. Empirical Verification of Spatial Smoothness

To validate whether the theoretical gradient bound (Proposition 1) effectively regularizes the learned neural mappings, we first track the global geometric complexity over the training epochs. For a fair comparison across architectures, we evaluate the dimension-normalized DDE as:

$$
G C _ { \mathrm { a v g } } = \frac { G C _ { \mathrm { s u m } } } { C \times H \times W } ,\tag{17}
$$

<sup>3</sup>In practical implementations, the input feature is the normalized radial distance $\widetilde { R } \propto \Vert \mathbf { r } \Vert _ { 2 } ^ { ^ { \mathbf { \sigma } } }$ . We formulate the analysis using spatial displacement r to rigorously evaluate the 2D gradient $\nabla _ { \mathbf { r } }$

TABLE V  
ABLATION OF PHYSICAL REPRESENTATION CHOICES (MEAN OVER 5 RUNS). I DENOTES THE SET OF UTILIZED HANKEL ORDERS.
<table><tr><td>Representation Choice</td><td>Orders (Z) / Mode n</td><td> $\scriptstyle \mathbf { R } ^ { 2 }$ </td><td>RMSE (dB)</td><td>FLOPs (G) Latency (s)</td><td></td></tr><tr><td colspan="6">Radial Envelope Only (Amplitude)</td></tr><tr><td>Zeroth-order only</td><td> ${ \mathcal { T } } = \{ 0 \}$ </td><td>0.8559</td><td>3.1025</td><td>91.48</td><td>0.95</td></tr><tr><td>First-order only</td><td> $\mathcal { T } = \{ 1 \}$ </td><td>0.8352</td><td>3.6856</td><td>91.48</td><td>0.97</td></tr><tr><td>Proposed Variant</td><td> $\mathcal { Z } = \{ 0 , 1 \}$ </td><td>0.9849</td><td>1.8392</td><td>91.58</td><td>1.11</td></tr><tr><td>Second-order added</td><td> $\mathcal { Z } = \{ 0 , 1 , 2 \}$ </td><td>0.9851</td><td>1.8376</td><td>91.66</td><td>1.20</td></tr><tr><td>Third-order added</td><td> $\mathcal { T } = \{ 0 , 1 , 2 , 3 \}$ </td><td>0.9845</td><td>1.8662</td><td>91.78</td><td>1.29</td></tr><tr><td colspan="6">Complete Angular Modes (Real &amp; Imaginary Expansion)</td></tr><tr><td>Complex-field variant</td><td> $| n | \leq 1$ </td><td>0.9852</td><td>1.8360</td><td>91.66</td><td>1.28</td></tr><tr><td>Complex-field variant</td><td> $| n | \leq 2$ </td><td>0.9850</td><td>1.8420</td><td>91.78</td><td>1.33</td></tr><tr><td>Complex-field variant</td><td> $| n | \leq 3$ </td><td>0.9843</td><td>1.8750</td><td>91.90</td><td>1.42</td></tr></table>

TABLE VI

COMPLEXITY COMPARISON EVALUATED UNDER IDENTICAL CONDITIONS (256 × 256, BATCH SIZE 1).
<table><tr><td>Method</td><td>Params (M)</td><td>FLOPs (G)</td><td>Size (MB)</td><td>Peak Mem (MB)</td><td>Lat. (s)</td><td>maps/s</td></tr><tr><td>RadioUNet</td><td>25.41</td><td>58.31</td><td>101.65</td><td>260</td><td>1.396</td><td>0.716</td></tr><tr><td>PEFNet</td><td>118.38</td><td>506.25</td><td>473.52</td><td>720</td><td>1.426</td><td>0.701</td></tr><tr><td>Ours (Net Only)</td><td>26.13</td><td>91.58</td><td>104.50</td><td>260</td><td>0.812</td><td>1.232</td></tr><tr><td>Hankel Encoding</td><td>0.0005</td><td> $\mathrm { N } / \mathrm { A } ^ { a }$ </td><td>0.002</td><td>120</td><td>0.285</td><td>3.509</td></tr><tr><td>HankelKANet</td><td>26.13</td><td> $\geq 9 1 . 5 8 ^ { a }$ </td><td>104.50</td><td>340</td><td>1.110</td><td>0.901</td></tr></table>

<sup>a</sup> Standard FLOP profilers only count basic neural network operations and cannot measure the mathematical complexity of special functions $( \mathrm { e . g . }$ , Bessel functions). Therefore, 91.58 G represents the FLOPs for the network backbone only. The actual computational cost of the Hankel encoding is instead transparently reflected in its wall-clock latency (0.285 s).

where $G C _ { \mathrm { s u m } }$ is the total DDE across the spatial domain, C is the number of input channels, and $H = W = 2 5 6$ are the spatial dimensions.

As shown in Fig. 4, training evolves through three distinct phases: fitting, transition, and diffusion [16]. Strikingly, while PEFNet’s global geometric complexity surges dramatically during the early stages before plateauing at a high level, HankelKANet maintains a consistently flat and stable $G C _ { \mathrm { a v g } }$ profile throughout. By the end of training, HankelKANet’s complexity remains orders of magnitude lower, indicating that it does not achieve high accuracy by memorizing irregular functional landscapes, but rather converges to an intrinsically smoother mapping.

This macroscopic stability is directly mirrored at the microscopic scene level. Fig. 5(a) visualizes the spatial Jacobian sensitivity norm, $\| \nabla _ { \mathbf { r } } \widehat { \mathrm { P L } } ( \mathbf { r } ) \| _ { 2 }$ , for a representative scene. While PEFNet inherently produces erratic gradient flares and local hotspots, HankelKANet exhibits smooth, radially decaying contours originating from the transmitter. To rigorously isolate these behaviors from the dominant radial path loss decay, we extract a 1D sensitivity profile along an iso-distance arc $A \to B \ ( \| \mathbf { r } \| _ { 2 } = \mathrm { c o n s t . } )$ traversing both LOS and shadowed NLOS regions (Fig. 5(b)).

Given the constant radial distance along this arc, the theoretical gradient in unobstructed free space should remain continuous and stable. While both models correctly capture the macroscopic step-changes at diffraction boundaries (buildings), their continuous-space behaviors diverge fundamentally. PEFNet exhibits pervasive jagged fluctuations across all regions, corroborating the macroscopic complexity surge seen in Fig. 4. Without intrinsic physical constraints, standard networks relying on piecewise activations (e.g., ReLU) lack explicit Lipschitz-like bounds and tend to overfit discrete training points, inevitably generating non-physical, Runge-like spatial ripples [39]. Conversely, HankelKANet maintains a remarkably smooth curve without spurious oscillations. This confirms that the embedded asymptotic envelope, functioning as a physics-inspired distance encoding, provides strong structural regularization. It inherently bounds the spatial gradient and naturally prevents the network from fabricating highfrequency spatial noise.

## C. From Spatial Smoothness to Real-World Generalization

The structural spatial smoothness discussed in the previous subsection naturally extends to superior cross-scenario and cross-dataset generalizability. Standard unconstrained networks tend to overfit the specific artifacts of the simulated source domain (as evidenced by their non-physical spatial ripples), which degrades their performance on distinct target domains. By leveraging the physics-inspired inductive bias (Proposition 1) to maintain bounded spatial gradients, HankelKANet is less prone to such overfitting, allowing it to extract domain-invariant wave propagation characteristics.

To empirically evaluate this cross-dataset adaptability, we transfer the models pre-trained on our simulated dataset to the real-world Huawei RSRPSet [35]. As summarized in Table VII, HankelKANet consistently outperforms PEFNet across all target data budgets (η). The most notable indicator is the zero-shot scenario $( \eta = 0 \% ) \mathrm { { ; } }$ : without seeing any field data, HankelKANet achieves an RMSE of 5.82 dB, surpassing PEFNet (6.43 dB) and even outperforming the scratch baseline trained with 5% real data (5.91 dB). This suggests that HankelKANet successfully transfers underlying physical decay characteristics rather than merely memorizing source-domain simulation features. Furthermore, during sparse adaptation, HankelKANet demonstrates superior data efficiency. Finetuning HankelKANet with only 5% of the target data (4.57 dB) yields performance comparable to PEFNet using nearly double the budget (10% data, 4.48 dB).

![](images/88268ee943c04f80db3e23e977dc6f89b8296f5b2decaae9f7f7a9a1beaea0b5.jpg)

![](images/0de8d91bdf26633e3d9ff1fc63451e58e062ec451e36c47585ee483db3df1fb5.jpg)

![](images/76029935eb1b2a8afc8cecff50cbc4c8541be7a453ec615c0451f46ae451970b.jpg)

![](images/d98f49ee154e251132ac68bc2692551e3232eee57d24881e7037ab55e448e4d1.jpg)  
Fig. 4. Evolution of the Dimension-Normalized Geometric Complexity $( G C _ { \mathrm { a v g } } )$ and prediction metrics over training epochs. Following [16], the shaded regions denote three optimization stages: fitting (capturing dominant large-scale structures), transition (refining finer details), and diffusion (stabilizing the learned mapping). Unlike PEFNet, which experiences a rapid initial surge in complexity before stabilizing at a highly elevated level, HankelKANet maintains a consistently low $G C _ { \mathrm { a v g } }$ throughout training. Ultimately, HankelKANet settles at a value several orders of magnitude lower than PEFNet. This massive gap indicates that HankelKANet converges to an intrinsically smoother mapping that naturally aligns with large-scale propagation physics.  
TABLE VII

(a)PEFNet  
![](images/d913ed5f3549fa03ad74495ac9acfdce50c6f6c790b2520599026995461d0b2f.jpg)  
(b) HankelKANet  
(a) 2D spatial heatmap of the Jacobian gradient norm

(b) 1D sensitivity profile along iso-distance arc $A  B$  
![](images/12e6a350f09a3261479195352072757dc2b51a288e936ec781bcb4b4e0122d15.jpg)  
Fig. 5. Microscopic Jacobian Sensitivity visualization. (a) 2D spatial heatmap of the Jacobian gradient norm. (b) 1D sensitivity profile extracted along an iso-distance arc traversing both Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) regions. While both models react to diffraction boundaries, PEFNet suffers from pervasive high-frequency fluctuations across all regions. In contrast, HankelKANet imposes physics-inspired smoothness, yielding a stable sensitivity curve that aligns with Proposition 1.

CROSS-DATASET GENERALIZATION VALIDATION: TRANSFER PERFORMANCE (RMSE IN DB) ON THE REAL-WORLD HUAWEI RSRPSET
<table><tr><td rowspan="2">Model</td><td rowspan="2">Pre-trained</td><td colspan="5">Target Domain Data Budget (η)</td></tr><tr><td>0% (Zero-shot)</td><td>1%</td><td>5%</td><td>10%</td><td>20%</td></tr><tr><td>Scratch Baseline</td><td>No</td><td>-</td><td>7.42</td><td>5.91</td><td>4.94</td><td>4.31</td></tr><tr><td>PEFNet</td><td>Yes</td><td>6.43</td><td>5.73</td><td>4.92</td><td>4.48</td><td>4.12</td></tr><tr><td>HankelKANet</td><td>Yes</td><td>5.82</td><td>5.14</td><td>4.57</td><td>4.23</td><td>3.98</td></tr></table>

Ultimately, these empirical results confirm that the structural spatial smoothness is not merely an aesthetic advantage, but a fundamental driver for robust real-world generalization and efficient target-domain adaptation.

## VII. CONCLUSION

This paper presented HankelKANet, a lightweight and physics-inspired neural network for path loss estimation. By integrating asymptotic electromagnetic decay characteristics to function as a physics-inspired distance encoding, the proposed architecture significantly streamlines the required input representations and the overall learning process. Experimental evaluations on both the simulated RadioMapSeer dataset and the real-world Huawei RSRPSet demonstrate that HankelKA-Net achieves state-of-the-art accuracy with minimal computational overhead. Beyond baseline performance, our analysis confirms that the embedded Hankel prior provides an effective structural regularization. This intrinsic spatial smoothness naturally prevents the network from overfitting to simulator artifacts, which serves as the fundamental driver for its robust cross-dataset generalization and high data efficiency during target-domain adaptation. Ultimately, HankelKANet offers an efficient, physics-guided, and highly practical solution for coverage planning and interference management in emerging 5G/6G networks.

## REFERENCES

[1] C. Phillips, D. Sicker, and D. Grunwald, “A survey of wireless path loss prediction and coverage mapping methods,” IEEE Communications Surveys & Tutorials, vol. 15, no. 1, pp. 255–270, 2012.

[2] I. Joseph, E. Oghu, and O. O. Roberts, “Path loss and models: A survey and future perspective for wireless communication networks,” International Journal ofAdvanced Networking and Applications, vol. 15, no. 02, pp. 5892–5907, 2023.

[3] Y. Ruan, P. Chen, Z. Yang, W. Zhao, F. Tang, Z. Liu, and N. Kato, “The divided-qubo-based quantum algorithm for optimal wireless link scheduling,” IEEE Transactions on Emerging Topics in Computing, 2025.

[4] 3GPP, “Study on 3d channel model for lte,” 3rd Generation Partnership Project (3GPP), Technical Report TR 36.873, 2015. [Online]. Available: https://portal.3gpp.org/desktopmodules/ Specifications/SpecificationDetails.aspx?specificationId=2574

[5] ——, “Study on channel model for 5g new radio (nr),” 3rd Generation Partnership Project (3GPP), Technical Report TR 38.901, 2018. [Online]. Available: https://portal.3gpp.org/desktopmodules/ Specifications/SpecificationDetails.aspx?specificationId=3173

[6] Z. Yun and M. F. Iskander, “Ray tracing for radio propagation modeling: Principles and applications,” IEEE access, vol. 3, pp. 1089–1100, 2015.

[7] J. Huang, C.-X. Wang, L. Bai, J. Sun, Y. Yang, J. Li, O. Tirkkonen, and M.-T. Zhou, “A big data enabled channel model for 5g wireless communication systems,” IEEE Transactions on Big Data, vol. 6, no. 2, pp. 211–222, 2018.

[8] R. Levie, C¸ . Yapar, G. Kutyniok, and G. Caire, “Radiounet: Fast radio map estimation with convolutional neural networks,” IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 4001–4015, 2021.

[9] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, “Physics-informed machine learning,” Nature Reviews Physics, vol. 3, no. 6, pp. 422–440, 2021.

[10] C. Meng, S. Griesemer, D. Cao, S. Seo, and Y. Liu, “When physics meets machine learning: A survey of physics-informed machine learning,” Machine Learning for Computational Science and Engineering, vol. 1, no. 1, p. 20, 2025.

[11] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” Journal of Computational physics, vol. 378, pp. 686–707, 2019.

[12] K. Luo, J. Zhao, Y. Wang, J. Li, J. Wen, J. Liang, H. Soekmadji, and S. Liao, “Physics-informed neural networks for pde problems: A comprehensive review,” Artificial Intelligence Review, vol. 58, no. 10, p. 323, 2025.

[13] F. Jiang, T. Li, X. Lv, H. Rui, and D. Jin, “Physics-informed neural networks for path loss estimation by solving electromagnetic integral equations,” IEEE Transactions on Wireless Communications, 2024.

[14] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljaciˇ c,´ T. Y. Hou, and M. Tegmark, “Kan: Kolmogorov-arnold networks,” arXiv preprint arXiv:2404.19756, 2024.

[15] J. D. Toscano, V. Oommen, A. J. Varghese, Z. Zou, N. Ahmadi Daryakenari, C. Wu, and G. E. Karniadakis, “From pinns to pikans: Recent advances in physics-informed machine learning,” Machine Learning for Computational Science and Engineering, vol. 1, no. 1, pp. 1–43, 2025.

[16] J. D. Toscano, L.-L. Wang, and G. E. Karniadakis, “Kkans: Kurkovakolmogorov-arnold networks and their learning dynamics,” Neural Networks, vol. 191, p. 107831, 2025.

[17] W. C. Chew, Waves and fields in inhomogenous media. John Wiley & Sons, 1999.

[18] S. Wang, Y. Teng, and P. Perdikaris, “Understanding and mitigating gradient flow pathologies in physics-informed neural networks,” SIAM Journal on Scientific Computing, vol. 43, no. 5, pp. A3055–A3081, 2021.

[19] F. M. Rohrhofer, S. Posch, C. Goßnitzer, and B. C. Geiger, “Data vs.¨ physics: The apparent pareto front of physics-informed neural networks,” IEEE Access, vol. 11, pp. 86 252–86 261, 2023.

[20] S. Wang, X. Yu, and P. Perdikaris, “When and why pinns fail to train: A neural tangent kernel perspective,” Journal of Computational Physics, vol. 449, p. 110768, 2022.

[21] R. Bischof and M. A. Kraus, “Multi-objective loss balancing for physicsinformed deep learning,” Computer Methods in Applied Mechanics and Engineering, vol. 439, p. 117914, 2025.

[22] D. Zhang, X.-C. Yuan, N. Ngo, and P. Shum, “Fast hankel transform and its application for studying the propagation of cylindrical electromagnetic fields,” Optics Express, vol. 10, no. 12, pp. 521–525, 2002.

[23] G. Gonzalez, Advanced electromagnetic wave propagation methods. CRC Press, 2021.

[24] A. N. Kolmogorov, “On the representations of continuous functions of many variables by superposition of continuous functions of one variable and addition,” in Dokl. Akad. Nauk USSR, vol. 114, 1957, pp. 953–956.

[25] H.-T. Ta, “Bsrbf-kan: A combination of b-splines and radial basis functions in kolmogorov-arnold networks,” in International Symposium on Information and Communication Technology. Springer, 2024, pp. 3–15.

[26] J. Xu, Z. Chen, J. Li, S. Yang, W. Wang, X. Hu, and E. Ngai, “Enhancing graph collaborative filtering with fourierkan feature transformation,” in Proceedings of the 34th ACM International Conference on Information and Knowledge Management, 2025, pp. 5376–5380.

[27] S. T. Seydi, Z. Bozorgasl, and H. Chen, “Unveiling the power of wavelets: A wavelet-based kolmogorov-arnold network for hyperspectral image classification,” arXiv preprint arXiv:2406.07869, 2024.

[28] Z. Chao, X. Liu, Z. Wu, and X. Li, “Rbf-kan: Radial basis functionkolmogorov-arnold networks,” IEEE Internet of Things Journal, 2026.

[29] S. T. Seydi, “Exploring the potential of polynomial basis functions in kolmogorov-arnold networks: A comparative study of different groups of polynomials,” arXiv preprint arXiv:2406.02583, 2024.

[30] A. D. Bodner, A. S. Tepsich, J. N. Spolski, and S. Pourteau, “Convolutional kolmogorov-arnold networks,” arXiv preprint arXiv:2406.13155, 2024.

[31] B. Dherin, M. Munn, M. Rosca, and D. Barrett, “Why neural networks find simple solutions: The many regularizers of geometric complexity,” Advances in Neural Information Processing Systems, vol. 35, pp. 2333– 2349, 2022.

[32] D. Mitrea, I. Mitrea, and M. Mitrea, “Geometric harmonic analysis,” Book Manuscript, 2020.

[33] M.-H. Yueh, “Theoretical foundation of the stretch energy minimization for area-preserving simplicial mappings,” SIAM Journal on Imaging Sciences, vol. 16, no. 3, pp. 1142–1176, 2023.

[34] “Radiomapseer dataset,” https://radiomapseer.github.io/, 2020.

[35] Y. Zheng, “Rsrpset: the dataset of the 16th cpgmcm,” 5, collected by Huawei for the 16th China Postgraduate Mathematical Contest in Modeling (CPGMCM), measured urban LTE RSRP path loss dataset. [Online]. Available: https://ieee-dataport.org/documents/rsrpset-dataset-16th-cpgmcm

[36] Y. Zhang, J. Wen, G. Yang, Z. He, and J. Wang, “Path loss prediction based on machine learning: Principle, method, and data expansion,” Applied Sciences, vol. 9, no. 9, p. 1908, 2019.

[37] I. R. Gomes, C. R. Gomes, H. S. Gomes, and G. P. d. S. Cavalcante, “Empirical radio propagation model for dtv applied to non-homogeneous paths and different climates using machine learning techniques,” PloS one, vol. 13, no. 3, p. e0194511, 2018.

[38] N. Rahaman, A. Baratin, D. Arpit, F. Draxler, M. Lin, F. Hamprecht, Y. Bengio, and A. Courville, “On the spectral bias of neural networks,” in International conference on machine learning. PMLR, 2019, pp. 5301–5310.

[39] H. Gouk, E. Frank, B. Pfahringer, and M. J. Cree, “Regularisation of neural networks by enforcing lipschitz continuity,” Machine Learning, vol. 110, no. 2, pp. 393–416, 2021.