# Architecture-Aware Mixed Precision for Replica-Exchange Training of Geometry-Aware Neural Operators with Uncertainty Quantification

Ujin Jeona, Michael Penwardenb, Wenqian Chenc, Guang Lind,\*

aDepartment of Electrical and Computer Engineering, Purdue University, West Lafayette, 47907, IN, USA bSandia National Laboratories, 1515 Eubank Blvd SE, Albuquerque, 87123, NM, USA cPacific Northwest National Laboratory, 3335 Innovation Blvd, Richland, 99354, WA, USA dDepartment of Mathematics and Mechanical Engineering, Purdue University, West Lafayette, 47907, IN, USA

## Abstract

Geometry-aware neural operators face substantial computational and memory demands when learning across complex physical domains. We introduce Architecture-Aware Mixed Precision (AA-MP), a replica-exchange training strategy that allocates numerical precision according to replica roles and architecture-specific operations. By retaining FP32 for refinement while selectively employing BF16 for exploration, AA-MP preserves predictive accuracy while improving training efficiency. Across fluid dynamics and three-dimensional structural mechanics benchmarks, AA-MP achieves comparable accuracy to full-precision training over an ensemble of runs, while reducing hot-replica peak GPU memory by 46.9% and training time by up to 25.8%. It also outperforms naive Automatic Mixed Precision (AMP) in memory efficiency and computational cost, with increasing memory advantages at larger model and geometry scales. Ablation studies show that retaining FP32 in the cold-replica is important for predictive accuracy, while additional FP32 operations in the exploratory replica provide limited benefits relative to their computational cost. Finally, conformal prediction for uncertainty calibration shows that AA-MP achieves near-nominal coverage with narrower prediction intervals than the alternatives. These results advance the scalability of geometry-aware scientific machine learning by reducing computational costs without compromising predictive accuracy or uncertainty quantification.

Keywords: Neural operator, Mixed precision, Replica exchange, Stochastic gradient Langevin dynamics, Uncertainty quantification

## 1. Introduction

Scientific machine learning (SciML) has become an important tool for accelerating simulation, inference, and prediction in computational physics. Methods such as physics-informed neural networks (PINNs), Deep Operator Networks (DeepONets), Fourier neural operators (FNOs), and related neural operator architectures have shown strong potential for learning parametric solution operators and constructing fast surrogate models for systems governed by partial differential equations (PDEs) [1, 2, 3, 4, 5]. These methods are especially useful in many-query settings, including design optimization, uncertainty quantification, inverse modeling, and real-time prediction, where repeated high-fidelity PDE solves are computationally expensive.

A central challenge is extending neural operators to complex and variable geometries. DeepONet supports evaluation at arbitrary spatial query locations, while conventional FNO formulations commonly use structured discretizations. Nevertheless, representing complex and varying geometries remains challenging and may require additional geometric encoding or architecture-specific treatment [6]. The Arbitrary Geometry-encoded Transformer (ArGEnT) addresses this challenge by representing geometry through point-cloud coordinates and signed-distance information and using attention to construct geometry-conditioned solution features [7]. In this work, we use the cross-attention

ArGEnT formulation, in which sampled query locations form the query representation while a separate geometry point cloud provides the key and value representations. ArGEnT is integrated as the geometry-aware trunk of a DeepONetstyle architecture, while the branch network encodes the non-geometric physical and operating parameters.

This geometric flexibility introduces substantial computational cost. Geometry-aware Transformer operators repeatedly process sampled query locations and geometry point clouds through projection, attention, and feedforward operations, making GPU memory consumption and training throughput important practical constraints. Mixedprecision computation provides a natural mechanism for reducing these costs and has been studied extensively in deep learning, numerical linear algebra, scientific computing, and, more recently, neural operator learning [8, 9, 10].

Existing mixed-precision strategies often assign numerical precision according to operation type, numerical sensitivity, model component, or backend execution policy. In this work, we additionally use the algorithmic role of each replica in replica-exchange optimization to determine its precision policy. The low-temperature replica is responsible for refinement and final model evaluation, whereas the high-temperature replica is used to promote broader exploration of the optimization landscape. This asymmetry motivates treating numerical precision as a replica-dependent computational resource rather than applying the same precision policy to both chains.

Accordingly, we propose an architecture-aware mixed-precision (AA-MP) strategy for replica-exchange training of geometry-aware neural operators. The low-temperature cold replica uses 32-bit floating-point (FP32) computation for refinement and model evaluation, whereas the high-temperature exploratory replica uses a manually controlled bfloat16 (BF16) model representation for its forward and backward computation. The hot-replica preconditioned reSGLD update retains FP32 master parameters and first- and second-moment states, while the DeepONet latent reduction and scalar prediction-error and exchange-energy reductions are also evaluated in FP32. Replica-exchange proposals couple this reduced-precision exploratory replica to the full-precision refinement replica. AA-MP therefore incorporates mixed precision directly into the exploration-refinement structure of replica exchange rather than applying a uniform precision policy to both replicas. Throughout this work, the notation A/B denotes the precision policies of the cold and hot replicas, respectively

The main contributions of this work are as follows.

• Replica-dependent mixed-precision training. We introduce a mixed-precision replica-exchange strategy that assigns numerical precision according to replica role. The low-temperature cold replica is maintained entirely in FP32 for refinement and evaluation, whereas the high-temperature exploratory replica executes the ArGEnT DeepONet forward and backward computation using the proposed architecture-aware mixed-precision mechanism.

• Explicit precision control for preconditioned reSGLD. We couple the BF16 exploratory model representation with FP32 master parameters and first- and second-moment optimizer states. The branch–trunk product is formed in BF16 and promoted to FP32 before the latent reduction, while the prediction error and exchangeenergy reductions are evaluated in FP32. This explicit precision map separates reduced-precision neuralnetwork computation from FP32 optimizer and reduction operations without relying on automatic mixedprecision (AMP) autocasting.

• Controlled precision-allocation and protection ablations. We evaluate the proposed FP32/AA-MP configuration over 10 independent seeds on the AirfRANS fluid-dynamics and jet-engine bracket structural-mechanics benchmarks, comparing it with FP32/FP32, FP32/AMP, and FP32/Naive-BF16. We further examine alternative cold/hot replica allocations and progressively increased FP32 protection within the exploratory replica to determine how precision placement affects predictive accuracy and computational efficiency.

• Computational efficiency and scaling. FP32/AA-MP reduces hot-phase peak allocated GPU memory by 46.9% on both benchmarks and hot-replica step time by 25.0% on AirfRANS and 25.8% on the bracket benchmark relative to FP32/FP32, while maintaining comparable predictive accuracy. Relative to FP32/AMP, the proposed configuration further reduces GPU memory by approximately 9.4% and hot-replica step time by 5.5% and 3.2% on AirfRANS and the bracket benchmark, respectively. Scaling experiments further show that AA-MP maintains lower GPU-memory requirements than standard AMP and the available FP32 reference as geometry point-cloud size and Transformer embedding dimension increase, with larger absolute savings at the highest tested workloads.

• Uncertainty quantification under different precision strategies. Using predictions from retained cold-replica states, we evaluate post-training uncertainty quantification on AirfRANS for FP32/FP32, FP32/AMP, FP32/AA-MP, and FP32/Naive-BF16. We apply conformal risk control and compare empirical case-averaged spatial coverage, prediction-interval width, and interval scores to assess uncertainty calibration and interval sharpness under the different precision strategies.

The remainder of the paper is organized as follows. Section 2 reviews mixed-precision computation, ArGEnT DeepONet, and replica-exchange SGLD. Section 3 presents the proposed AA-MP replica-exchange framework and its precision-aware preconditioned reSGLD implementation. Section 4 presents the benchmark results, precisionallocation ablations, computational-efficiency studies, and post-training conformal-UQ evaluation. Section 5 summarizes the main findings and discusses future directions.

## 2. Related Work and Background

This section reviews prior work on mixed-precision computation relevant to scientific machine learning and introduces the two technical components underlying the proposed methodology: ArGEnT for geometry-aware operator learning and replica-exchange SGLD for temperature-based exploration. Together, these components provide the architectural and optimization framework on which the proposed replica-dependent precision strategy is built

## 2.1. Mixed-Precision Training for Scientific Machine Learning

Mixed-precision computation has been widely used to reduce the memory and computational cost of machinelearning models. In deep learning, mixed-precision training was popularized through reduced-precision forward and backward operations, high-precision master weights, and loss scaling to avoid gradient underflow [8]. Related approaches have been developed for neural ordinary differential equations and mixed-precision quantization, where precision is assigned according to stability, sensitivity, entropy, or accuracy-aware criteria [11, 12, 13, 14]

Mixed precision also has a long history in numerical linear algebra and high-performance scientific computing. Classical mixed-precision iterative refinement performs expensive algebraic operations in reduced precision while using higher precision for residual correction, enabling low-precision hardware acceleration while recovering higheraccuracy solutions [15, 16, 17, 9]. Modern GPU implementations exploit tensor cores and other reduced-precision hardware to accelerate dense and sparse linear algebra, Krylov methods, eigensolvers, geostatistical modeling, weather prediction, and large-scale scientific simulations [18, 19, 20, 21, 22, 23, 24]. These studies demonstrate that numerical precision can serve as an algorithmic design parameter for balancing computational efficiency and numerical accuracy.

Mixed precision has also been investigated directly in scientific machine learning. Hayford et al. [25] studied mixed-precision training for physics-informed neural networks (PINNs) and DeepONets, demonstrating reductions in computational cost and memory usage while examining numerical difficulties associated with low-precision computation. Related work on mixed-precision neural operators has also investigated approximation guarantees and numerical precision effects [10].

These studies motivate the use of mixed precision in scientific machine learning, while leaving room to investigate how precision allocation can be tailored to specific neural operator architectures and optimization procedures. In this work, we investigate architecture-aware and replica-dependent precision allocation for ArGEnT-DeepONet training with preconditioned replica-exchange SGLD. The low-temperature refinement replica is retained in FP32, whereas the high-temperature exploratory replica uses an explicitly controlled BF16 model representation with FP32 master parameters, optimizer states, and selected numerical reductions. By directly comparing this configuration with standard BF16 AMP under the same replica-exchange training framework, we evaluate whether explicit precision control provides additional computational savings while maintaining competitive predictive accuracy.

## 2.2. Arbitrary Geometry-encoded Transformer for Operator Learning (ArGEnT)

The Arbitrary Geometry-encoded Transformer (ArGEnT) architecture was proposed in [7]. ArGEnT is a geometryaware neural operator architecture designed to learn solution operators on complex and varying physical domains. DeepONet supports evaluation at arbitrary spatial query locations, while conventional Fourier neural operator (FNO) formulations commonly use structured discretizations. Representing complex and varying geometries nevertheless requires suitable geometric inputs or architectural mechanisms. ArGEnT addresses this challenge by incorporating geometry through point-based representations and decoupling geometry encoding from query-point evaluation. In this work, ArGEnT serves as the baseline geometry-aware operator-learning model to which mixed-precision training is applied.

Let $x \in \mathbb { R } ^ { d }$ denote a spatial query point and let φ(x) denote the signed distance function (SDF) value associated with the geometry. We write the augmented query input as

$$
\tilde { x } = ( x , \phi ( x ) ) ,\tag{1}
$$

where additional pointwise features can be appended when available. The geometry is represented by a point cloud

$$
\Omega = \left\{ \tilde { x } _ { i } ^ { g } \right\} _ { i = 1 } ^ { M } , \qquad \tilde { x } _ { i } ^ { g } = \left( x _ { i } ^ { g } , \phi ( x _ { i } ^ { g } ) \right) ,\tag{2}
$$

where $x _ { i } ^ { g }$ is a geometry point and $\phi ( x _ { i } ^ { g } )$ is the corresponding SDF value. This representation allows the model to map complex domains without explicit finite-dimensional geometric parameterization.

In this work, we use the cross-attention ArGEnT formulation to evaluate the proposed mixed-precision training strategy.

In the cross-attention variant, the query points and geometry representation are decoupled. The embedded query features form the query matrix Q, whereas the embedded geometry point cloud Ω provides the key and value representations K and V. For a single attention head, let $Q \in \mathbb { R } ^ { N _ { q } \times d }$ and K $, V \in \mathbb { R } ^ { N _ { \mathrm { a u n } } \times d }$ , where $N _ { q }$ is the number of query points, $N _ { \mathrm { a t t n } }$ is the number of geometry tokens, and d is the head dimension. The Galerkin-type cross-attention operation is expressed as

$$
\mathrm { A t t n } ( \boldsymbol { Q } , \boldsymbol { K } , \boldsymbol { V } ) = \frac { 1 } { N _ { \mathrm { a t t n } } } \boldsymbol { Q } \left( \widetilde { \boldsymbol { K } } ^ { \top } \widetilde { \boldsymbol { V } } \right) ,\tag{3}
$$

where $\widetilde { K }$ and $\widetilde { V }$ denote the normalized key and value representations, respectively. The operation is softmax-free: the key-value aggregation is performed before multiplication by the query representation, without applying softmax normalization to $Q K ^ { \top }$ . The resulting cross-attention operator can be written abstractly as

$$
\mathcal { G } _ { \mathrm { C A } } : ( \tilde { x } , \Omega ) \mapsto f ( \tilde { x } , \Omega ) .\tag{4}
$$

Because the query points and geometry representation are supplied separately, the solution can be evaluated at query locations sampled independently of the geometry point cloud.

For physical systems whose solution depends both on geometry and on non-geometric parameters, ArGEnT is integrated into a DeepONet-style branch-trunk architecture. Let $\mu$ denote the non-geometric physical parameters, such as operating conditions or applied loads. For a problem with $d _ { \mathrm { o u t } }$ predicted physical fields and latent dimension $J ,$ the branch network produces feld-dependent coefficients

$$
c ( \mu ) = \left\{ c _ { \ell j } ( \mu ) \right\} _ { \ell = 1 , \dots , d _ { \mathrm { o u t } } , j = 1 , \dots , J } ,\tag{5}
$$

while the ArGEnT trunk produces geometry-conditioned latent features

$$
f ( \tilde { x } , \Omega ) = \left\{ f _ { \ell j } ( \tilde { x } , \Omega ) \right\} _ { \ell = 1 , \dots , d _ { \mathrm { o u t } } , j = 1 , \dots , J } .\tag{6}
$$

The predicted value of output field l is obtained by an elementwise branch-trunk product followed by reduction over the latent dimension,

$$
\hat { u } _ { \ell } ( \tilde { x } ; \Omega , \mu ) = \sum _ { j = 1 } ^ { J } c _ { \ell j } ( \mu ) f _ { \ell j } ( \tilde { x } , \Omega ) , \qquad \ell = 1 , \dots , d _ { \mathrm { o u t } } .\tag{7}
$$

The vector-valued prediction is therefore $\hat { \mathbf { u } } = ( \hat { u } _ { 1 } , \dots , \hat { u } _ { d _ { \mathrm { o u t } } } )$

The model is trained in a supervised operator-learning setting using high-fidelity simulation data. Given vectorvalued reference solutions $\mathbf { u } ( x _ { i } ; \Omega , \mu )$ at query points $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , the baseline supervised objective is the mean-squared error (MSE),

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \Vert \hat { \mathbf { u } } ( \tilde { x } _ { i } ; \Omega , \mu ) - \mathbf { u } ( x _ { i } ; \Omega , \mu ) \Vert _ { 2 } ^ { 2 } .\tag{8}
$$

## 2.3. Preconditioned Replica Exchange SGLD

Stochastic gradient Langevin dynamics (SGLD) was introduced as a scalable stochastic-gradient sampling method for Bayesian learning [26]. Let $\theta _ { k }$ denote the trainable parameters at iteration k and let $U _ { \mathcal { B } _ { k } } ( \boldsymbol { \theta } _ { k } )$ denote the stochastic training energy evaluated on minibatch $\mathcal { B } _ { k }$ . To motivate the optimization dynamics used in this work, consider the diagonally preconditioned update [27]

$$
\theta _ { k + 1 } = \theta _ { k } - \eta _ { k } G _ { k } \nabla U _ { \mathcal { B } _ { k } } ( \theta _ { k } ) + \sqrt { 2 \eta _ { k } T } G _ { k } ^ { 1 / 2 } \xi _ { k } , \qquad \xi _ { k } \sim N ( 0 , I ) ,\tag{9}
$$

where $\eta _ { k }$ is the step size, T is the Langevin temperature, and $G _ { k }$ is a positive diagonal preconditioner that rescales both the stochastic-gradient update and Langevin perturbation. Equation (9) provides a reference preconditioned Langevin update. The moment-based adaptive preconditioner and stochastic optimization updates used in this work are specified in Section 3; the reference expression is not taken to establish exact invariant-distribution sampling for the adaptive implementation.

Replica exchange, also known as parallel tempering or exchange Monte Carlo, improves exploration by evolving multiple chains at different temperatures and periodically proposing swaps between them [28]. For two chains with temperatures $T _ { c } < T _ { h }$ and states $\theta ^ { c }$ and $\theta ^ { h } ,$ , the exchange is accepted with probability

$$
\alpha = \operatorname* { m i n } \left\{ 1 , \exp \left[ \left( { \frac { 1 } { T _ { c } } } - { \frac { 1 } { T _ { h } } } \right) \left( U ( \theta ^ { c } ) - U ( \theta ^ { h } ) \right) \right] \right\} .\tag{10}
$$

Equation (10) gives the ideal replica-exchange acceptance probability based on the full training energy. The hightemperature replica promotes broader exploration of the loss landscape, whereas the low-temperature replica provides lower-temperature refinement. Replica-exchange stochastic-gradient methods extend this principle to stochasticgradient learning problems, where non-convexity and minibatch noise complicate the optimization landscape [29, 30]. In this work, replica exchange additionally provides the algorithmic structure on which replica-dependent numerical precision is assigned; the stochastic minibatch exchange criterion used in the actual implementation is introduced in Section 3.

## 3. Methodology

![](images/9c08558f8c7258c683aca595211c8f768c8ccdc7973f84c8eadf4510ed9a9285.jpg)  
Train with preconditioned replica-exchange SGLD (AA-MP hot replica; FP32 cold replica used for final prediction)

Figure 1: Overview of the proposed AA-MP ArGEnT-DeepONet training framework. (a) The DeepONet branch encodes non-geometric physical parameters, while the cross-attention ArGEnT trunk uses query features as queries and the geometry point cloud as keys and values. The exploratory hot replica uses a manually controlled BF16 live model, while the latent reduction, prediction-error and energy reductions, master parameters, and optimizer states are maintained in FP32. The cold replica executes the same network entirely in FP32. (b) Two replicas are evolved using preconditioned reSGLD at $T _ { c } < T _ { h } .$ The higher-temperature hot replica promotes broader exploration, whereas the FP32 cold replica performs low-temperature refinement and model evaluation. Replica-exchange proposals periodically exchange the parameter and associated optimizer states between the two replica slots.

We present a replica-dependent mixed-precision strategy for training ArGEnT-DeepONet models with preconditioned replica-exchange stochastic gradient Langevin dynamics (reSGLD). Rather than applying the same automatic mixed-precision policy to both replicas, the proposed method assigns numerical precision according to replica role. The low-temperature cold replica remains entirely in FP32 for refinement and model evaluation, whereas the hightemperature exploratory replica uses a manually controlled BF16 live model for its computationally intensive forward and backward operations. The hot-replica optimizer retains FP32 master parameters and first- and second-moment states, while the DeepONet latent reduction and scalar prediction-error and exchange-energy reductions are evaluated in FP32. This design couples numerical precision directly to the exploration-refinement structure of replica exchange.

## 3.1. Mixed Precision Replica Exchange

The proposed method incorporates mixed precision into the replica-exchange training strategy rather than treating reduced precision as a uniform acceleration mechanism. The cold chain is evaluated entirely in FP32 and acts as the low-temperature refinement and evaluation replica. The hot chain operates at a higher Langevin temperature and uses reduced-precision computation for the expensive ArGEnT-DeepONet forward and backward passes. Exchange proposals allow parameter states discovered by the exploratory replica to enter the cold replica while the cold slot subsequently continues its updates in FP32. The resulting strategy therefore assigns reduced precision to the exploratory replica while retaining full precision in the refinement replica.

In the present optimization setting, $\hat { U } _ { \mathcal { B } } ( \boldsymbol { \theta } )$ denotes the minibatch supervised training objective used by the implementation. The temperatures $T _ { c }$ and $T _ { h }$ are algorithmic hyperparameters that control the Langevin perturbation magnitude and the exchange rule

Let θ denote the trainable parameters of the ArGEnT-DeepONet operator. We evolve two replicas using preconditioned replica-exchange stochastic gradient Langevin dynamics (reSGLD) at Langevin temperatures $T _ { c } < T _ { h } ,$ where $T _ { c }$ and $T _ { h }$ denote the cold- and hot-replica temperatures, respectively. The cold chain is updated using full-precision evaluations. Its stochastic gradient is therefore written as

$$
g _ { \boldsymbol { k } } ^ { c } = \nabla \widehat { U } _ { 3 2 , \mathcal { B \boldsymbol { k } } } ( \boldsymbol { \theta } _ { \boldsymbol { k } } ^ { c } ) ,\tag{11}
$$

where $\widehat { U } _ { 3 2 , \mathcal { B } _ { k } }$ denotes the training objective on minibatch $\mathcal { B } _ { k }$ evaluated in full precision. For the hot chain, let $\theta _ { k } ^ { h }$ denote the FP32 master parameters and let

$$
\theta _ { k , \mathrm { l i v e } } ^ { h } = \mathrm { c a s t } _ { \mathrm { B F 1 6 } } ( \theta _ { k } ^ { h } )\tag{12}
$$

denote the synchronized BF16 live-model parameters. The hot-replica gradient used by the optimizer is

$$
g _ { k } ^ { h } = \mathrm { c a s t } _ { 3 2 } \left[ \mathrm { g r a d } _ { \mathrm { l i v e } } \widehat { U } _ { \mathrm { m p } , \mathcal { B } _ { k } } ( \theta _ { k , \mathrm { l i v e } } ^ { h } ) \right] ,\tag{13}
$$

where $\mathrm { g r a d } _ { \mathrm { l i v e } }$ denotes the gradient computed by automatic differentiation with respect to the BF16 live-model parameters. The resulting gradient is promoted to FP32 before updating the master parameters. This notation describes the implemented mixed-precision gradient computation rather than differentiation through the FP32-to-BF16 casting operation.

For each replica, the preconditioned reSGLD optimizer maintains FP32 first- and second-moment buffers. Let $r \in \{ c , h \}$ denote the cold or hot replica, and let $g _ { k } ^ { r }$ denote its stochastic gradient at update k. We index the optimizer updates by $k \geq 1$ and initialize the moment buffers as $m _ { 0 } ^ { r } = \nu _ { 0 } ^ { r } = 0$ . The first- and second-moment estimates are updated as

$$
m _ { k } ^ { r } = \beta _ { 1 } m _ { k - 1 } ^ { r } + ( 1 - \beta _ { 1 } ) g _ { k } ^ { r } ,\tag{14}
$$

$$
\nu _ { k } ^ { r } = \beta _ { 2 } \nu _ { k - 1 } ^ { r } + ( 1 - \beta _ { 2 } ) \left( g _ { k } ^ { r } \odot g _ { k } ^ { r } \right) ,\tag{15}
$$

where  denotes elementwise multiplication. The corresponding bias-corrected moment estimates are

$$
\widehat { m } _ { k } ^ { r } = \frac { m _ { k } ^ { r } } { 1 - \beta _ { 1 } ^ { k } } , \qquad \widehat { \nu } _ { k } ^ { r } = \frac { \nu _ { k } ^ { r } } { 1 - \beta _ { 2 } ^ { k } } .\tag{16}
$$

Using the second-moment estimate, we define the adaptive diagonal preconditioner as

$$
G _ { k } ^ { r } = \mathrm { d i a g } \left[ \operatorname* { m i n } \left( G _ { \mathrm { m a x } } , \frac { 1 } { \sqrt { \hat { \nu } _ { k } ^ { r } } + \epsilon } \right) \right] , \qquad r \in \{ c , h \} ,\tag{17}
$$

where the minimum is applied elementwise. The constant $G _ { \mathrm { m a x } }$ bounds the magnitude of the diagonal preconditioning coefficients and prevents excessively large updates in coordinates with very small estimated second moments.

Using the adaptive diagonal preconditioner in Eq. (17), we define the cold- and hot-replica updates as

$$
\theta _ { k + 1 } ^ { c } = \theta _ { k } ^ { c } - \eta _ { k } G _ { k } ^ { c } \widehat { m } _ { k } ^ { c } + \sqrt { 2 \eta _ { k } T _ { c } } \left( G _ { k } ^ { c } \right) ^ { 1 / 2 } \xi _ { k } ^ { c } , \qquad \xi _ { k } ^ { c } \sim N ( 0 , I ) ,\tag{18}
$$

$$
\theta _ { k + 1 } ^ { h } = \theta _ { k } ^ { h } - \eta _ { k } G _ { k } ^ { h } \widehat { m } _ { k } ^ { h } +  \sqrt { 2 \eta _ { k } T _ { h } } ( G _ { k } ^ { h } ) ^ { 1 / 2 } \xi _ { k } ^ { h } , \qquad \xi _ { k } ^ { h } \sim N ( 0 , I ) .\tag{19}
$$

Here the square root of the diagonal preconditioner acts elementwise. For the cold chain, $g _ { k } ^ { c }$ is obtained from an FP32 forward and backward pass. In AA-MP, the live hot-replica model parameters are stored in BF16 and the hot forward and backward passes are executed directly in the manually assigned BF16 representation without AMP autocasting. The resulting hot-replica gradients are promoted to FP32 for the preconditioned reSGLD update, whose master parameters and first- and second-moment states are maintained in FP32. After each optimizer step, the updated FP32 master parameters are cast and synchronized back to the BF16 live model. The precision used for the branch– trunk reduction and exchange-energy evaluation is described in Section 3.2 and summarized in Table 1.

Every $K _ { \mathrm { s w a p } }$ iterations after the burn-in period, we propose an exchange between the cold- and hot-replica states. The same Langevin temperatures $T _ { c }$ and $T _ { h }$ used in the preconditioned reSGLD dynamics are used in the exchange criterion.

Let $\widehat { U } _ { \mathcal { B } } ^ { c }$ and $\widehat { U } _ { \mathcal B } ^ { h }$ denote the minibatch estimates of the training energy obtained from the current cold- and hotreplica model representations, respectively. We define the stochastic exchange criterion as

$$
\widehat { \alpha } = \operatorname* { m i n } \left\{ 1 , \exp \left[ \left( \frac { 1 } { T _ { c } } - \frac { 1 } { T _ { h } } \right) \left( \widehat { U } _ { \mathcal { B } } ^ { c } ( \theta ^ { c } ) - \widehat { U } _ { \mathcal { B } } ^ { h } ( \theta ^ { h } ) \right) \right] \right\} .\tag{20}
$$

For each exchange proposal, the cold- and hot-replica energies are recomputed on the same current minibatch using the active numerical representation of each replica. The cold replica performs an FP32 forward evaluation, whereas the AA-MP hot replica performs its forward evaluation using the manually controlled BF16 model representation without AMP autocasting. For AA-MP, the branch-trunk product is formed in BF16 and subsequently promoted to FP32 before the latent reduction. The resulting prediction and the squared-error reduction used to form the minibatch energy are therefore evaluated in FP32. The scalar exchange criterion in Eq. (20) is then evaluated in FP32 from the current cold- and hot-replica minibatch energies. Thus, the computationally intensive hot-replica model evaluation remains in BF16 while the final DeepONet reduction and scalar exchange calculation are performed in FP32.

An exchange is accepted with probability α. Upon acceptance, the parameter state and the associated preconditioned reSGLD optimizer state are exchanged between the two replicas. Maintaining the optimizer state together with its parameter vector preserves the correspondence between the parameters and the adaptive preconditioner. After an accepted exchange, the cold replica continues its subsequent updates in FP32, while the hot-replica BF16 model representation is synchronized from its corresponding FP32 master parameters.

Equation (20) defines the stochastic exchange rule used in our implementation. Unlike the exact Metropolis criterion for replica exchange, this rule uses minibatch energy estimates obtained from replicas evaluated at different numerical precisions. Consequently, the acceptance probability is approximate and does not guarantee detailed balance with respect to the full-data Gibbs distributions. We therefore interpret replica exchange as a mechanism for coupling exploratory and refinement trajectories, rather than as an exact sampling procedure.

This formulation differs from enabling standard AMP for the hot replica. In the AMP baseline, model parameters remain stored in FP32 while selected operations are executed in BF16 according to PyTorch autocast rules. In contrast, AA-MP explicitly represents the live hot-replica model in BF16 and executes its forward and backward computations without autocast, while retaining FP32 master parameters and first- and second-moment states for the preconditioned reSGLD update. The cold replica remains entirely in FP32. AA-MP therefore differs from AMP both in the representation of the exploratory replica and in the explicit separation between reduced-precision model computation and FP32 optimizer state.

## 3.2. Architecture-Aware Mixed-Precision Policy

The proposed AA-MP strategy explicitly assigns numerical precision across the ArGEnT-DeepONet training pipeline rather than relying on automatic mixed-precision casting. The cold replica remains in FP32. In the exploratory hot replica, the live model parameters and major intermediate tensors use BF16, including those associated with the branch and trunk networks, embedding layers, Q/K/V projections, K/V normalization, rotary positional encoding (RoPE), kernel-attention operations, and feedforward layers. This precision policy specifies the tensor representations and explicitly controlled operations; backend kernels may perform internal accumulation at higher precision.

The DeepONet branch-trunk elementwise product is also formed in BF16 and is then promoted to FP32 before reduction over the latent dimension. The resulting prediction and prediction-error reduction are evaluated in FP32. During optimization, gradients produced by the BF16 live model are promoted to FP32, while the preconditioned reS-GLD optimizer maintains FP32 master parameters and first- and second-moment states. The scalar energy reduction used by the replica-exchange procedure is likewise evaluated in FP32. Table 1 summarizes this precision structure.

Table 1: Precision structure of the proposed AA-MP configuration. The cold replica remains entirely in FP32. The table summarizes the manually controlled precision policy of the exploratory hot replica.
<table><tr><td>Operation</td><td>Precision</td><td>Implementation role</td></tr><tr><td>Hot-replica live model parameters</td><td>BF16</td><td>Explicit reduced-precision model repre- sentation without autocast.</td></tr><tr><td>Branch/trunk embeddings and MLPs</td><td>BF16</td><td>Reduced-precision feature computation.</td></tr><tr><td>Q/K/V projections, K/V LayerNorm, and BF16 RoPE</td><td></td><td>Cross-attention and positional feature computation.</td></tr><tr><td>Kernel attention, output projection, and BF16 feedforward layers</td><td></td><td>Computationally intensive Transformer operations.</td></tr><tr><td>Branch-trunk elementwise product</td><td>BF16</td><td>DeepONet feature fusion before latent re- duction.</td></tr><tr><td>Latent reduction and prediction</td><td>FP32</td><td>The BF16 branch-trunk product is pro- moted before summation.</td></tr><tr><td>Hot-replica gradients</td><td>BF16 → FP32</td><td>Gradients are promoted before the opti- mizer update.</td></tr><tr><td>Master parameters and optimizer moments</td><td>FP32</td><td>Preconditioned reSGLD parameter and moment states.</td></tr><tr><td>Prediction-error and energy reductions</td><td>FP32</td><td>Final scalar loss and exchange-energy re- ductions.</td></tr></table>

## 4. Numerical Experiments

## 4.1. Experimental Setup

All primary replica-exchange experiments were conducted on a single NVIDIA H100 80 GB GPU using PyTorch 2.5.1.post303 with a CUDA 12.0 build. The models were trained for 40,000 iterations with a case batch size of 32, a query-point batch size of 3,000, a geometry point-cloud size of 3,000, and an ArGEnT embedding dimension of 128. The initial learning rate was $3 \times 1 0 ^ { - 4 }$

The experiments use the implemented preconditioned reSGLD optimizer with cold- and hot-replica Langevin temperatures $T _ { c } ~ = ~ 1 0 ^ { - 6 }$ and $T _ { h } = 1 0 ^ { - 5 }$ , respectively. The optimizer uses $\beta _ { 1 } = 0 . 9$ for the first-moment estimate. $\beta _ { 2 } = 0 . 9 9$ for the second-moment estimate, $\bar { \epsilon } = 1 0 ^ { - 8 }$ , and a maximum diagonal preconditioner value of $G _ { \mathrm { m a x } } = 5 0$

The learning rate is multiplied by 0.99 every 100 completed iterations. Replica-exchange proposals are evaluated every 100 completed iterations after a 100-iteration burn-in. The same Langevin temperatures $T _ { c }$ and $T _ { h }$ used in the replica updates are used in the exchange criterion.

The primary resource comparison considers three precision strategies: FP32/FP32, FP32/AMP, and FP32/AA-MP. FP32/FP32 uses full-precision computation for both replicas. In the proposed FP32/AA-MP configuration, the cold replica remains in FP32 while the live hot-replica model is represented and evaluated in BF16 without autocast; the hot preconditioned reSGLD master parameters and moment states remain in FP32, and selected output and energy reductions are performed in FP32. In the FP32/AMP baseline, the hot-replica parameters remain stored in FP32 while BF16 execution is selected through PyTorch autocast. FP32/Naive-BF16 is additionally included as a reduced-precision ablation in the predictive-accuracy, scaling, and post-training UQ evaluations. For the primary predictive-accuracy and computational-efficiency comparisons, each precision configuration is evaluated over 10 independent seeds. The experiments are designed to compare precision policies within the same replica-exchange training framework rather than to establish an optimization advantage of replica exchange over unrelated single-chain optimizers.

For the jet-engine bracket experiments, 5,052 cases are used for training and 1,263 cases for testing. For AirfRANS, the experiments use the dataset with 800 training cases and 200 test cases. The primary predictive-accuracy and computational-efficiency comparisons use the same 40,000-iteration replica-exchange protocol for both benchmarks. The architecture-level precision-protection ablation uses a separate 20,000-iteration screening protocol, while the post-training conformal-UQ study uses models trained for 100,000 iterations followed by an additional Langevin sampling phase. These distinct experimental protocols are described in their respective subsections.

At each test evaluation, 3,000 query points are sampled without replacement from each test case. The sampled point set is regenerated at each evaluation rather than held fixed across training iterations. Predictive accuracy is quantified using the per-case relative $\ell _ { 2 }$ error. For test case j and predicted field k, we define

$$
e _ { j , k } = \frac { \left( \sum _ { i = 1 } ^ { N _ { q } } \left| \widehat { u } _ { j , k } ( \mathbf { x } _ { i } ) - u _ { j , k } ( \mathbf { x } _ { i } ) \right| ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { i = 1 } ^ { N _ { q } } \left| u _ { j , k } ( \mathbf { x } _ { i } ) \right| ^ { 2 } \right) ^ { 1 / 2 } } ,\tag{21}
$$

where $N _ { q } = 3 0 0 0$ denotes the number of sampled spatial query points. Errors are averaged across test cases, and the reported mean and sample standard deviation are computed over the 10 independent training seeds. Each physical field is evaluated separately.

GPU memory and runtime are measured directly during training. Peak allocated GPU memory is obtained using torch.cuda.max\_memory\_allocated after resetting the CUDA peak-memory statistics, and cold- and hot-chain step times are measured using time.perf\_counter with CUDA synchronization before and after the timed GPU operations. The logged per-step timing statistic is formed from the most recent 100 iterations. For the hot-phase resource comparison reported below, each seed is summarized using the median over the final 50% of logged measurement windows, and the reported values are the mean ± sample standard deviation across 10 seeds. The memory quantity should therefore be interpreted as hot-phase peak allocated GPU memory with both replicas resident, rather than as the whole-job system peak. Total training time, when reported separately, denotes end-to-end wall-clock time from the beginning of training. Throughout the replica-exchange experiments, the notation A/B denotes the precision policies of the cold and hot replicas, respectively.

## 4.2. Benchmark Problems

## 4.2.1. AirfRANS

AirfRANS introduced in [31] is a high-fidelity computational fluid dynamic dataset for learning surrogate models of two-dimensional steady-state incompressible Reynolds-Averaged Navier-Stokes (RANS) solutions around airfoils in a subsonic regime. The dataset is designed for the aerodynamic shape-optimization setting, where the objective is to predict both the full flow fields and physically meaningful quantities such as lift and drag coefficients. Each simulation corresponds to an airfoil geometry, a Reynolds number, and an angle of attack (AOA)

The flow is modeled using the incompressible RANS equations. Denoting ensemble-averaged quantities by an overbar, the governing equations are written as

$$
\partial _ { j } ( \bar { u } _ { i } \bar { u } _ { j } ) = - \partial _ { i } \bar { p } + \partial _ { j } \left[ ( \nu + \nu _ { t } ) \left( \partial _ { j } \bar { u } _ { i } + \partial _ { i } \bar { u } _ { j } \right) \right] ,\tag{22}
$$

Table 2: Default configuration for the primary replica-exchange experiments. Unless otherwise stated, the same settings are used for the jet-engine bracket and AirfRANS benchmarks.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Training iterations</td><td>40,000</td></tr><tr><td>Case batch size</td><td>32</td></tr><tr><td>Query-point batch size</td><td>3,000</td></tr><tr><td>Geometry point-cloud size</td><td>3,000</td></tr><tr><td>Transformer embedding dimension</td><td>128</td></tr><tr><td>Initial learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate schedule</td><td>multiply by 0.99 every 100 steps</td></tr><tr><td>Cold-chain temperature  $T _ { c }$ </td><td> $1 0 ^ { - 6 }$ </td></tr><tr><td>Hot-chain temperature  $T _ { h }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Swap frequency</td><td>100 iterations</td></tr><tr><td>reSGLD burn-in</td><td>100 iterations</td></tr><tr><td>Evaluation frequency</td><td>100 iterations</td></tr><tr><td>Independent seeds</td><td>10</td></tr><tr><td>AA-MP hot precision</td><td>BF16 live model; FP32 master/moment states</td></tr><tr><td>AMP hot precision</td><td>BF16 autocast with FP32 model parameters</td></tr></table>

where ${ { \bar { u } } _ { i } }$ is the mean velocity field, $\bar { p } = { p _ { \mathrm { p h y s } } } / { \rho }$ denotes the reduced (kinematic) pressure, $\rho$ is the fluid density, v is the kinematic viscosity, and $\nu _ { t }$ is the turbulent kinematic viscosity. The turbulent viscosity is modeled using the k-ω SST turbulence model.

The airfoils are generated from the NACA 4-digit and 5-digit families. The 4-digit family is parameterized by a sequence MPXX, where M denotes the maximum ordinate of the camber line, $P$ its chordwise position, and XX the maximum thickness. The 5-digit case is parameterized by $L P Q X X$ , where L and P define the camber, $Q$ distinguishes single- and double-cambered airfoils, and XX again denotes the maximum thickness. The dataset contains 1000 simulations. The Reynolds number ranges from $2 \times 1 0 ^ { 6 }$ to $6 \times 1 0 ^ { 6 }$ , and the AOA ranges from $- 5 ^ { \circ }$ to $1 5 ^ { \circ } { }$

The simulations are generated using OpenFOAM with the steady-state RANS solver simpleFOAM, the SIM-PLEC algorithm, and the $k { \mathrm { - } } \omega \operatorname { S S T }$ turbulence model. Each airfoil is discretized using a C-grid mesh generated with blockMesh. The far-field boundary is placed at 200 chord lengths from the airfoil to reduce boundary-condition effects. To accurately resolve surface quantities such as wall shear stress, the boundary-layer mesh is refined so that the first cells near the airfoil surface have height $2 \mu \mathrm m$ , giving a worst-case $y ^ { + }$ of approximately 1. The resulting meshes contain approximately 250,000–300,000 cells.

The machine-learning task is to approximate the spatial fields

$$
\left( \bar { u } _ { x } , \bar { u } _ { y } , \bar { p } , \nu _ { t } \right) ,\tag{23}
$$

where ${ \bar { u } } _ { x }$ and $\bar { u } _ { y }$ are the mean velocity components, $\bar { p }$ is the reduced pressure, and $\nu _ { t }$ is the turbulent kinematic viscosity.

The experiments use the processed dataset generated by the repository's conversion pipeline from all 1,000 simulations. The conversion used for the reported runs retains the full spatial simulation data rather than applying the rectangular crop used in an earlier preprocessing notebook. The spatial coordinates and signed-distance information are retained as geometric inputs, while the flow variables and freestream operating parameters are standardized using statistics computed during construction of the processed dataset. In the current processed file, these normalization statistics were computed over the complete 1,000-case collection before the fixed 800/200 train-test partition was used by the training loader. Thus, test simulations do not enter the optimization batches, but their distribution contributes to the preprocessing statistics. During training and evaluation, the loader subsamples query points from the processed simulation fields to control computational cost.

The present study uses the AirfRANS full-data regime, consisting of 800 training simulations and 200 test simulations. The same fixed partition is used for all precision configurations. We evaluate the four predicted fields $( \bar { u } _ { x } , \bar { u } _ { y } , \bar { p } , \nu _ { t } )$ and use AirfRANS as the primary benchmark for examining predictive convergence, precision allocation, and computational efficiency under different numerical precision policies.

## 4.2.2. Jet-Engine Bracket

We use a three-dimensional jet-engine bracket dataset released with the Point-DeepONet study [32], built from geometries derived from the DeepJEB benchmark [33]. The processed dataset used in this work contains 6,315 simulation cases corresponding to 2,105 unique bracket geometries, with three loading conditions (vertical, horizontal. and diagonal) associated with each geometry. The predicted structural fields are the three displacement components $( u _ { x } , u _ { y } , u _ { z } )$ and the von Mises stress $\sigma _ { \mathrm { v m } }$

The Point-DeepONet source study [32] reports nonlinear static finite-element analyses performed with Altair OptiStruct using second-order tetrahedral elements and a Ti-6Al–4V elastic-plastic material model with linear isotropic hardening. We refer to this source for the simulation setup; the processed data used here contain the resulting pointwise structural fields and operating parameters rather than the original finite-element solver decks.

The 6,315 simulation cases are partitioned into 5,052 training and 1,263 test cases using a fixed 80/20 case-level split. This partition is constructed over geometry-loading-condition pairs rather than unique geometries. Of the 1,022 unique geometries represented in the test set, 1,004 (98.2%) also occur in training under another loading condition. Consequently, this benchmark measures generalization to held-out simulation cases and loading configurations, rather than generalization to unseen geometries. We use it to assess the consistency of the proposed precision strategy on a three-dimensional structural-mechanics problem.

## 4.3. Computational Efficiency

![](images/ff7285676ea387592087091a0da6137450f5f085c7216246fa2e7af9c2226c81.jpg)  
Figure 2: Hot-replica computational efficiency for AirfRANS and the jet-engine bracket benchmarks. (a) Hot-replica peak allocated GPU memory with both replicas resident. (b) Hot-replica execution time per training step. Bars show the mean over 10 seeds, with each seed summarized by the median over the final 50% of logged measurement windows; error bars indicate sample standard deviation. Percentage annotations denote reductions relative to FP32/FP32.

Figure 2 summarizes the computational effect of reducing the precision of the exploratory replica. On AirfRANS, FP32/AA-MP reduces the hot-phase peak allocated memory from 5.727 GB to 3.042 GB, a 46.9% reduction relative to FP32/FP32, while the hot-chain step time decreases from 0.0804 s to 0.0603 s, corresponding to a 25.0% reduction. The FP32/AMP baseline also reduces the corresponding memory and step-time costs, to 3.356 GB and 0.0638 s per step, respectively.

The jet-engine bracket benchmark exhibits a similar pattern. FP32/AA-MP reduces hot-phase peak allocated memory from 5.709 GB to 3.034 GB and hot-chain time from 0.0861 s to 0.0639 s per step, corresponding to reductions of 46.9% and 25.8%, respectively. FP32/AMP achieves 3.347 GB and 0.0660 s per hot-chain step. Across both benchmarks, AA-MP reduces hot-phase resource requirements relative to FP32/FP32 and provides additional savings over standard BF16 AMP. Relative to AMP, AA-MP reduces peak allocated GPU memory by approximately 9.4% on both benchmarks, while hot-replica step time decreases by approximately 5.5% on AirfRANS and 3.2% on the bracket benchmark.

Table 3: Hot-replica resource measurements corresponding to Fig. 2. Values are mean ± sample standard deviation across 10 seeds after per-seed aggregation.
<table><tr><td>Dataset</td><td></td><td>Configuration Peak alloc (GB)</td><td>Hot sec/step</td></tr><tr><td rowspan="3">AirfRANS</td><td>FP32/FP32</td><td> $5 . 7 2 7 \pm 0 . 0 0 0$ </td><td rowspan="3"> $0 . 0 8 0 4 \pm 0 . 0 0 2 4$   $0 . 0 6 3 8 \pm 0 . 0 0 1 6$ </td></tr><tr><td>FP32/AMP</td><td> $3 . 3 5 6 \pm 0 . 0 0 0$ </td></tr><tr><td>FP32/AA-MP</td><td> $\mathbf { 3 . 0 4 } 2 \pm \mathbf { 0 . 0 0 0 }$ </td></tr><tr><td rowspan="3">Bracket</td><td>FP32/FP32</td><td> $5 . 7 0 9 \pm 0 . 0 0 0$ </td><td> $0 . 0 8 6 1 \pm 0 . 0 0 1 8$ </td></tr><tr><td>FP32/AMP</td><td> $3 . 3 4 7 \pm 0 . 0 0 0$ </td><td> $0 . 0 6 6 0 \pm 0 . 0 0 2 7$ </td></tr><tr><td>FP32/AA-MP</td><td> ${ \bf 3 . 0 3 4 \pm 0 . 0 0 0 }$ </td><td> $\mathbf { 0 . 0 6 3 9 \pm 0 . 0 0 0 } 7$ </td></tr></table>

These measurements characterize the hot-chain phase rather than the whole-job system peak or end-to-end training speedup. Because the cold replica remains in FP32 and exchange evaluation, validation, data handling, and other training overheads are unaffected by the hot-replica precision policy, reductions in total wall-clock training time are necessarily smaller than the isolated hot-step reductions.

## 4.4. Predictive Accuracy Across Precision Strategies

## 4.4.1. AirfRANS Benchmark

AirfRANS provides the primary precision-sensitive benchmark in this study. Figure 3 compares the cold-replica test relative $\ell _ { 2 }$ error for FP32/FP32, FP32/AMP, FP32/Naive-BF16, and the proposed FP32/AA-MP configuration. The FP32/AA-MP trajectories remain close to the FP32/FP32 reference across the four predicted flow quantities, whereas the less-protected Naive-BF16 configuration exhibits a clearer degradation, particularly for the velocity and turbulent-viscosity fields. Pressure is less sensitive to the tested precision policies. These results indicate that reduced precision can be applied to the exploratory replica while retaining predictive accuracy close to the FP32 reference when the BF16 live model is combined with FP32 master parameters, optimizer states, and DeepONet output and energy reductions.

## 4.4.2. Jet-Engine Bracket Benchmark

Figure 4 provides a complementary evaluation on the jet-engine bracket benchmark. In contrast to AirfRANS, the four precision strategies produce nearly overlapping convergence trajectories for the three displacement components and von Mises stress. The bracket problem is therefore less discriminative with respect to the tested precision policies, but it provides evidence that the proposed strategy remains stable across a distinct three-dimensional structural-mechanics problem.

To complement the convergence trajectories, Table 4 reports the per-field test relative $\ell _ { 2 }$ errors aggregated over the same 10 seeds for each precision strategy. All four reSGLD precision configurations use the same model architecture, optimizer, learning-rate schedule, replica temperatures, batch sizes, data partition, and 40,000-update training budget; they differ only in the precision policy assigned to the hot replica.

The per-field error statistics are consistent with the convergence trajectories in Figures 3 and 4. Across both benchmarks, FP32/AA-MP achieves predictive accuracy comparable to FP32/FP32 and FP32/AMP. Although AA-MP attains lower mean errors for several individual fields, these differences are generally small relative to the variability across training seeds and are not interpreted as evidence of systematically improved predictive accuracy. In contrast, Naive-BF16 exhibits larger errors for several AirfRANS quantities. These results support the use of replica-dependent precision allocation to reduce computational resource requirements while maintaining comparable predictive performance.

![](images/198816665f6793176bef0f68fc688c0ebbdf0cd3688e409285bb84391d92d51e.jpg)  
Figure 3: AirfRANS reSGLD per-field test relative $\ell _ { 2 }$ error. Curves show the mean over 10 independent seeds, while shaded regions indicate empirically calibrated pointwise bands based on across-seed variability.

Table 4: Per-field test relative $\ell _ { 2 }$ errors across the evaluated precision strategies. For the AirfRANS and bracket reSGLD blocks, the cold replica remains in FP32 and the column header denotes the hot-replica precision policy. The bracket SGD block reports single-chain precision-policy controls in the corresponding column order; the cold/hot replica notation in the column headings applies only to the reSGLD results. Values are mean ± sample standard deviation over 10 seeds. Naive-BF16 is included as a reduced-precision ablation. Bold indicates the lowest mean error for each field.
<table><tr><td>Dataset</td><td>Field</td><td>FP32/FP32</td><td>FP32/AA-MP</td><td>FP32/AMP</td><td>FP32/Naive-BF16</td></tr><tr><td rowspan="4">AirfRANS (reSGLD)</td><td>u</td><td> $0 . 1 0 5 5 \pm 0 . 0 1 5 5$ </td><td> $\mathbf { 0 . 1 0 2 2 } \pm 0 . 0 2 1 0$ </td><td> $0 . 1 1 4 5 \pm 0 . 0 1 5 8$ </td><td> $0 . 1 2 5 0 \pm 0 . 0 1 7 9$ </td></tr><tr><td> $\nu$ </td><td> $0 . 0 7 6 5 \pm 0 . 0 0 6 9$ </td><td> $\mathbf { 0 . 0 7 3 6 \pm 0 . 0 0 7 8 }$ </td><td> $0 . 0 7 9 5 \pm 0 . 0 0 8 0$ </td><td> $0 . 0 8 6 5 \pm 0 . 0 1 0 4$ </td></tr><tr><td> $p$ </td><td> $\mathbf { 0 . 0 5 0 1 } \pm 0 . 0 0 1 1$ </td><td> $0 . 0 5 1 1 \pm 0 . 0 0 0 8$ </td><td> $0 . 0 5 1 8 \pm 0 . 0 0 1 4$ </td><td> $0 . 0 5 4 8 \pm 0 . 0 0 3 5$ </td></tr><tr><td> $\nu _ { t }$ </td><td> $\mathbf { 0 . 1 2 2 6 \pm 0 . 0 0 5 4 }$ </td><td> $0 . 1 2 2 9 \pm 0 . 0 0 3 2$ </td><td> $0 . 1 2 4 4 \pm 0 . 0 0 4 3$ </td><td> $0 . 1 3 6 3 \pm 0 . 0 1 0 2$ </td></tr><tr><td rowspan="4">Bracket (reSGLD)</td><td> $u _ { x }$ </td><td> $\mathbf { 0 . 1 5 4 3 \pm 0 . 0 0 7 3 }$ </td><td> $0 . 1 5 4 6 \pm 0 . 0 0 6 9$ </td><td> $0 . 1 5 5 3 \pm 0 . 0 0 6 4$ </td><td> $0 . 1 5 4 8 \pm 0 . 0 0 7 5$ </td></tr><tr><td> $u _ { y }$ </td><td> $0 . 1 9 5 4 \pm 0 . 0 0 5 5$ </td><td> $\mathbf { 0 . 1 9 4 5 \pm 0 . 0 0 4 3 }$ </td><td> $0 . 1 9 4 7 \pm 0 . 0 0 4 1$ </td><td> $0 . 1 9 5 9 \pm 0 . 0 0 6 0$ </td></tr><tr><td> $u _ { z }$ </td><td> $0 . 0 9 7 7 \pm 0 . 0 0 4 0$ </td><td> $\mathbf { 0 . 0 9 7 6 \pm 0 . 0 0 3 7 }$ </td><td> $0 . 0 9 8 4 \pm 0 . 0 0 3 6$ </td><td> $0 . 0 9 8 0 \pm 0 . 0 0 3 6$ </td></tr><tr><td> $\sigma _ { \nu m }$ </td><td> $\mathbf { 0 . 2 7 6 8 \pm 0 . 0 0 4 6 }$ </td><td> $0 . 2 7 7 0 \pm 0 . 0 0 3 4$ </td><td> $0 . 2 7 8 1 \pm 0 . 0 0 2 7$ </td><td> $0 . 2 7 7 7 \pm 0 . 0 0 4 2$ </td></tr><tr><td rowspan="4">Bracket (SGD)</td><td> $u _ { x }$ </td><td> $0 . 3 5 8 7 \pm 0 . 0 2 2 9$ </td><td> $\pm 0 . 3 5 5 4 \pm 0 . 0 2 3 2$ </td><td> $0 . 3 6 3 7 \pm 0 . 0 3 3 0$ </td><td> $1 . 0 0 4 6 \pm 0 . 0 0 8 4$ </td></tr><tr><td> $u _ { y }$ </td><td> $0 . 4 7 2 9 \pm 0 . 0 3 2 8$ </td><td> $0 . 4 7 4 6 \pm 0 . 0 3 6 3$ </td><td> $\mathbf { 0 . 4 7 1 4 } \pm 0 . 0 4 3 1$ </td><td> $1 . 0 0 3 0 \pm 0 . 0 0 5 4$ </td></tr><tr><td> $u _ { z }$ </td><td> $0 . 2 3 6 8 \pm 0 . 0 1 9 4$ </td><td> $\mathbf { 0 . 2 3 0 6 \pm 0 . 0 1 5 2 }$ </td><td> $0 . 2 3 6 8 \pm 0 . 0 2 5 9$ </td><td> $1 . 0 0 3 2 \pm 0 . 0 0 3 2$ </td></tr><tr><td> $\sigma _ { \nu m }$ </td><td> $0 . 5 5 2 0 \pm 0 . 0 2 5 7$ </td><td> $\mathbf { 0 . 5 5 0 2 } \pm 0 . 0 2 6 1$ </td><td> $0 . 5 5 2 9 \pm 0 . 0 3 1 8$ </td><td> $0 . 9 9 4 4 \pm 0 . 0 0 2 1$ </td></tr></table>

![](images/dc57b746105a9801be6689ded3a77f18cd9f34d331bf8f4232263b50cbe32236.jpg)  
Figure 4: Jet-engine bracket reSGLD cold-replica per-field test relative l2 error. Curves show the mean over 10 independent seeds, while shaded regions indicate empirically calibrated pointwise bands based on across-seed variability.

## 4.5. Qualitative Validation

Aggregate relative $\ell _ { 2 }$ errors quantify predictive accuracy but do not show where approximation errors occur in the physical domain. We therefore examine representative solution fields to assess whether AA-MP preserves the dominant spatial structures of the learned solutions and to identify localized error patterns.

## 4.5.1. AirfRANS

Figure 5 compares the AA-MP prediction with the reference solution and the corresponding pointwise percentage error for the streamwise velocity, transverse velocity, pressure, and turbulent viscosity. For each field, the percentage error is defined as $1 0 0 \widehat { \mu } _ { k } ( \mathbf { x } ) - u _ { k } ( \mathbf { x } ) | / \operatorname* { m a x } _ { j } | u _ { k } ( \mathbf { x } _ { j } ) |$ , where the maximum is taken over the full reference simulation case. The predicted fields reproduce the major flow structures surrounding the airfoil, while the error maps reveal the spatial regions in which the operator approximation is least accurate.

![](images/4825a07f9bc6b03b1c30c7943513500ba761e26ea60df3ae571dd4c7f473a07b.jpg)

![](images/d9875860ba57ec37ce0e28f9ae913bc0a22859c3040c2e1a78798562776a02f5.jpg)

![](images/30bd9657157432c7b87421b8861b7f7049ab6cb2bd915ebedee1b23c82cdd7a4.jpg)

![](images/a51679e1aba4459dcebf1b40a49bebc2dd56be904e5a9a1280d04082f73fd97c.jpg)

![](images/ff47758558ba2f277c5ff2584bc4106bfb7c11a1867473c5e9fccfe5a1af7477.jpg)

![](images/d7b8ab108806ac316bdf27de59f4fdcb0ce3cac7e581e157c9d8ad3e9ffcfb93.jpg)

![](images/967f88f1f91d2ee4a0126492dec5b76c22dd92e0256a7d697d4bea7188b42764.jpg)

![](images/87e493525e86ce2c8d0ef859f5030a8d5a73aa29f0bd680728a1a8f9a639c094.jpg)

![](images/c54337071d5be6ac040ab6d19f9874ca02be9e06f350ea2635bdd6a31d686160.jpg)

![](images/cb948c656c2fd0f8b3002fe03de8d58d002b201398a77f0fd0f6674228aea701.jpg)

![](images/c278077ba0b4bef0688199bd31b498d31a44ecd2fbfd7133a9a0933550378cc1.jpg)

![](images/9aac94faf53a78a1137461b105d5ab53526ab01f3be655d17bc6358de308ccd3.jpg)  
Figure 5: Illustrative AirfRANS solution fields obtained with AA-MP. For each predicted quantity, the columns show the prediction, reference solution, and pointwise percentage error normalized by the maximum absolute reference value for that field. Prediction and reference fields use identical color limits within each field.

## 4.5.2. Bracket

![](images/f94dee0bb2aaafc552bec604394aacad1180921404b01f2bd16cb1c19c119509.jpg)

![](images/c55c979506d96d7600f043590a82f374bfc84841a789f22c30fab5b1043d4140.jpg)

![](images/273bf5314dc49e0391cbe0e785fe49453ecedfd8ef926faae13dd2acb9ee7cbd.jpg)

![](images/bfc84e542d5451180e3b83917b8a5553cd15123b22ea8a76cf81eafd228bbcd7.jpg)

![](images/e5b50f9ff9f4b1791dea4ceab9d3253914735d64736939474718f1c004ea7af2.jpg)

![](images/4a045c87c055261bd0fbc87f090ae6175ba50807e5cb92c1a5f383e1d1712bf2.jpg)

![](images/c39b0271c9a4aa268d1dd9782683ecc3356db8e0feecca95ab6f5cd54dfe202d.jpg)

![](images/cb00343bfe09fd7a854b2f5dd082c48122f80657bd850d015fde883b6cea8df1.jpg)

![](images/1e1a0108d0910cecd77c0e9abe2f13d75db0a7aa993001196444b2a41e209135.jpg)

![](images/cbaa0f17ad3ad24eaa4a01059ca57b12ecfbaa7922a3567fb81dd189ee1b4407.jpg)

![](images/6397b135aabd46675886b5086e8e6da8d34941f60f04ec90f1f3d8b052021d48.jpg)

![](images/1b066f1d6ac5223b000e31cffdf9ab759875f7e495313cc74c330f2bdd722f6f.jpg)  
Figure 6: Illustrative jet-engine bracket solution fields obtained with AA-MP. For each predicted quantity, the columns show the prediction, reference solution, and pointwise percentage error normalized by the maximum absolute reference value for that field. Prediction and reference fields use identical color limits within each field.

Table 5: Per-field predictive accuracy for the architecture-level FP32-protection ablation on AirfRANS. All configurations use the same 20,000- iteration training protocol and 10 paired seeds. Values are mean per-case test relative l2 error, reported as mean ± sample standard deviation across seeds. The FP32-protection rows are cumulative in their displayed order: RoPE, K/V LayerNorm, the input-feature pathway, and Q/K/V projections. Each successive row retains the protections of the preceding row. Bold indicates the lowest mean error for each field.
<table><tr><td>Precision policy</td><td>u</td><td>ν</td><td> $p$ </td><td> $\nu _ { t }$ </td></tr><tr><td>AA-MP</td><td> $0 . 0 5 9 3 \pm 0 . 0 0 4 8$ </td><td> $0 . 0 6 8 4 \pm 0 . 0 0 4 6$ </td><td> $0 . 0 5 2 4 \pm 0 . 0 0 1 6$ </td><td> $0 . 0 9 8 8 \pm 0 . 0 0 3 2$ </td></tr><tr><td> $\mathbf { A A - M P + F P 3 2 R o P E }$ </td><td> $0 . 0 5 8 2 \pm 0 . 0 0 6 2$ </td><td> $\mathbf { 0 . 0 6 4 2 } \pm 0 . 0 0 5 6$ </td><td> $\mathbf { 0 . 0 5 0 0 \pm 0 . 0 0 1 8 }$ </td><td> $\mathbf { 0 . 0 9 3 6 \pm 0 . 0 0 2 6 }$ </td></tr><tr><td> $\mathbf { A } \mathbf { A } \mathbf { - } \mathbf { M } \mathbf { P } + \mathbf { F } \mathbf { P } 3 2$  LayerNorm</td><td> $0 . 0 5 9 6 \pm 0 . 0 0 7 2$ </td><td> $0 . 0 6 7 4 \pm 0 . 0 0 4 6$ </td><td> $0 . 0 5 1 2 \pm 0 . 0 0 1 2$ </td><td> $0 . 0 9 5 3 \pm 0 . 0 0 3 2$ </td></tr><tr><td>AA  $\mathbf { - M P + F P } 3 2$  input path</td><td> $0 . 0 5 9 8 \pm 0 . 0 0 8 5$ </td><td> $0 . 0 6 6 8 \pm 0 . 0 0 6 4$ </td><td> $0 . 0 5 1 0 \pm 0 . 0 0 2 5$ </td><td> $0 . 0 9 4 1 \pm 0 . 0 0 3 2$ </td></tr><tr><td> $\mathbf { A } \mathbf { A } \mathbf { - } \mathbf { M } \mathbf { P } + \mathbf { F } \mathbf { P } 3 2$  projections</td><td> $\mathbf { 0 . 0 5 8 1 } \pm 0 . 0 0 5 6$ </td><td> $0 . 0 6 6 9 \pm 0 . 0 0 5 5$ </td><td> $0 . 0 5 0 8 \pm 0 . 0 0 2 0$ </td><td> $0 . 0 9 5 9 \pm 0 . 0 0 3 0$ </td></tr><tr><td>FP32/FP32 control</td><td> $0 . 0 6 0 9 \pm 0 . 0 0 9 4$ </td><td> $0 . 0 6 8 1 \pm 0 . 0 0 8 9$ </td><td> $0 . 0 5 1 6 \pm 0 . 0 0 1 7$ </td><td> $0 . 0 9 5 2 \pm 0 . 0 0 2 9$ </td></tr></table>

## 4.6. Architecture-Level FP32 Protection Ablation

To examine whether retaining additional model components in FP32 improves predictive performance, we performed a progressive precision-protection ablation on the AirfRANS benchmark. Starting from the proposed AA-MP policy, FP32 computation was successively introduced into the coordinate/RoPE pathway, K/V normalization, the input-feature pathway, and the $\mathrm { Q } / \mathrm { K } / \mathrm { V }$ projections. All configurations use the same 20,000-iteration screening protocol and 10 paired seeds.

Table 6: Hot-replica computational cost for the AirfRANS FP32-protection ablation. For each seed, resource measurements are summarized by the median over the final 50% of logged measurement windows, using the same aggregation protocol as Table 3. Values are then averaged over 10 seeds. The FP32 protections are cumulative in the same order as Table 5. Memory and step-time reductions are relative to the FP32/FP32 control.
<table><tr><td>Precision policy</td><td>Peak alloc. (GB)</td><td>Mem. red.</td><td>Hot sec/step</td><td>Time red.</td></tr><tr><td>AA-MP</td><td>3.051</td><td>46.8%</td><td>0.0580</td><td>25.5%</td></tr><tr><td> $\mathbf { A A - M P + F P 3 2 R o P E }$ </td><td>3.785</td><td>34.0%</td><td>0.0655</td><td>15.8%</td></tr><tr><td> $\mathbf { A } \mathbf { A } \mathbf { - } \mathbf { M } \mathbf { P } + \mathbf { F } \mathbf { P } 3 2$  LayerNorm</td><td>4.153</td><td>27.6%</td><td>0.0685</td><td>12.0%</td></tr><tr><td> $\mathbf { A } \mathbf { A } \mathbf { - } \mathbf { M } \mathbf { P } + \mathbf { F } \mathbf { P } 3 2$  input path</td><td>4.151</td><td>27.6%</td><td>0.0690</td><td>11.4%</td></tr><tr><td> $\mathbf { A } \mathbf { A } \mathbf { - } \mathbf { M } \mathbf { P } + \mathbf { F } \mathbf { P } 3 2$  projections</td><td>4.357</td><td>24.0%</td><td>0.0728</td><td>6.5%</td></tr><tr><td>FP32/FP32 control</td><td>5.734</td><td>一</td><td>0.0778</td><td></td></tr></table>

Table 5 reveals a modest accuracy-efficiency tradeoff. Relative to AA-MP, retaining the RoPE pathway in FP32 reduces the mean test relative $\ell _ { 2 }$ error by 6.2% for v, 4.6% for $p ,$ and 5.2% for $\nu _ { t } .$ Paired tests across the 10 seeds show that these reductions remain statistically detectable after Holm correction $( p = 0 . 0 1 5 , 0 . 0 0 2$ , and 0.015, respectively). The corresponding change in u is smaller (–1.9%) and is not statistically detectable $( p = 0 . 4 0 9 )$

The accuracy improvement is accompanied by a substantial computational cost. As shown in Table $^ { 6 , }$ AA-MP reduces hot-replica peak allocated memory and hot-step time by 46.8% and 25.5%, respectively, relative to the FP32/FP32 control. Protecting the RoPE pathway reduces these savings to 34.0% and 15.8%. Equivalently, FP32 RoPE requires approximately 24% more hot-replica memory and 13% more hot-step time than AA-MP.

The additional protection stages do not yield a monotonic improvement in predictive accuracy. Among the protected configurations, FP32 RoPE gives the lowest mean errors for $\nu , p ,$ and $\nu _ { t } ,$ while further protecting LayerNorm, the input pathway, and the Q/K/V projections progressively reduces the computational advantage. At the most protected setting, the memory and step-time savings decrease to 24.0% and 6.5%, respectively.

We therefore retain AA-MP as the default configuration for the primary experiments because it provides the largest computational savings among the evaluated mixed-precision policies while maintaining competitive predictive accuracy. The FP32-RoPE configuration represents an alternative, more accuracy-oriented operating point when modest improvements in v, p, or $\nu _ { t }$ are prioritized.

The resource values in this ablation are obtained from the independent 20,000-iteration screening campaign and therefore differ slightly from the primary 40,000-iteration measurements in Table 3; both studies use the same per-seed resource-aggregation procedure.

## 4.7. Uncertainty Quantification Evaluation

We evaluate spatially resolved uncertainty quantification using cold-replica states obtained through a separate post-training Langevin sampling phase. Following 100,000 training iterations, each precision configuration undergoes 5,000 additional preconditioned reSGLD updates, yielding 50 retained cold-temperature states. These states provide predictive means and spatially varying uncertainty estimates, which are subsequently calibrated using conformal risk control. Conformal uncertainty quantification has also been investigated for DeepONet models [34].

Predictions from the retained cold-replica states vary across spatial locations and model parameter states. Their empirical dispersion provides a spatially varying uncertainty scale associated with the learned operator and the posttraining Langevin trajectories. This dispersion is subsequently adjusted through conformal calibration using observed prediction errors. The resulting prediction intervals therefore incorporate empirical predictive-error information in addition to the variability among retained states.

For field k ∈ {u, v, p, vt} and spatial point x, let $\mu _ { k } ( { \bf x } )$ and $s _ { k } ( \mathbf { x } )$ denote the mean and standard deviation of the retained-state predictions. We construct the symmetric conformal prediction interval

$$
C _ { k } ( { \mathbf x } ) = \left[ \mu _ { k } ( { \mathbf x } ) - H _ { k } ( { \mathbf x } ) , \mu _ { k } ( { \mathbf x } ) + H _ { k } ( { \mathbf x } ) \right] , \qquad H _ { k } ( { \mathbf x } ) = \widehat \lambda _ { k } \sqrt { s _ { k } ^ { 2 } ( { \mathbf x } ) + \epsilon _ { k } ^ { 2 } } .\tag{24}
$$

For spatial visualization and interval-sharpness comparison, we report the full conformal prediction-interval width,

$$
W _ { k } ( \mathbf { x } ) = 2 H _ { k } ( \mathbf { x } ) = 2 \widehat { \lambda } _ { k } \sqrt { s _ { k } ^ { 2 } ( \mathbf { x } ) + \epsilon _ { k } ^ { 2 } } .\tag{25}
$$

The scale floor $\epsilon _ { k }$ is estimated from 40 separate training-side cases, while the multiplier $\widehat { \lambda } _ { k }$ is selected using conformal risk control (CRC) [351 on 100 calibration cases. All four methods are evaluated on the same 98 heldout AirfRANS cases, with identical spatial coordinates and reference fields. CRC targets a nominal 5% mean spatial miscoverage across exchangeable AirfRANS cases, corresponding to a case-averaged coverage target of 95%. Interval construction uses consistent standardized output units, and the resulting intervals are converted to physical-field units before reporting widths and interval scores.

For spatial visualization, we select AirfRANS evaluation case 911 based on its proximity to the median physical characteristics within the majority airfoil-parameterization family, considering freestream velocity, angle of attack, thickness, and camber. The selection is independent of prediction errors and conformal coverage, and the same evaluation case is used for all four precision configurations.

Figure 8 illustrates the calibrated FP32/AA-MP intervals for case 911. For each flow field, the panels show the reference solution, predictive mean, full conformal prediction-interval width, and spatial coverage diagnostic. Corresponding visualizations for the other three precision configurations are provided in Appendix Appendix A (Figs. A.1– A.3). These maps complement the aggregate results in Table 7.

To evaluate interval sharpness together with miscoverage, we report the mean interval score [36]. Let yk(x) denote the corresponding truth value for field k. For the symmetric interval in Eq. (24), its pointwise value at nominal miscoverage $\alpha = 0 . 0 5$ is

$$
{ { S } _ { 0 . 0 5 , k } } ( { \bf { x } } ) = { { W } _ { k } } ( { \bf { x } } ) + 4 0 \left( { | y _ { k } ( { \bf { x } } ) - { \mu _ { k } } ( { \bf { x } } ) | - \frac { { { W } _ { k } } ( { \bf { x } } ) } { 2 } } \right) _ { + } ,\tag{26}
$$

where $( a ) _ { + } = \operatorname* { m a x } ( a , 0 )$ . The scores are averaged over nodes within each case and then equally over evaluation cases.   
Smaller scores indicate narrower intervals with an appropriate penalty for observations outside the interval.

As shown in Fig. 7, all four precision configurations achieve mean case-averaged spatial coverage near the nominal 95% target. FP32/AA-MP exhibits coverage closest to the target across all four fields. Table 7 further shows that AA-MP achieves the smallest mean full prediction-interval width and lowest mean interval score across the evaluated fields. However, its lower 5th-percentile per-case coverage indicates greater variability in coverage among individual flow configurations.

Figure 8 provides a spatial illustration for case 911. The coverage maps distinguish locations where the reference field lies inside the prediction interval from those where it falls above or below the interval bounds. The observed violations are localized, including regions around and downstream of the airfoil. The nominal coverage target concerns case-averaged spatial coverage, not uniform coverage at every node or for every individual case. Corresponding diagnostics for the other precision configurations appear in Figs. A.1, A.2, and A.3.

Table 7: Post-training AirfRANS conformal-UQ results for four precision configurations at nominal 95% case-averaged node coverage (98 evaluation cases). Coverage and 5th-percentile per-case coverage are percentages; mean full prediction-interval width and interval score use de-normalized dataset output units. Smaller interval widths indicate sharper predictions, while lower interval scores indicate a more favorable combined sharpness– coverage trade-off. Bold indicates coverage closest to the nominal target, the smallest interval width and score, or the highest 5th-percentile coverage, as appropriate.
<table><tr><td rowspan="2">Field</td><td colspan="5"></td></tr><tr><td>Method</td><td>Mean cov.</td><td>Mean full width</td><td>Interval score</td><td>5th pct.</td></tr><tr><td rowspan="5">u</td><td>FP32/FP32</td><td>96.24</td><td>14.570</td><td>16.880</td><td>91.73</td></tr><tr><td>FP32/AMP</td><td>95.94</td><td>23.300</td><td>26.400</td><td>91.59</td></tr><tr><td>FP32/AA-MP</td><td>94.92</td><td>3.328</td><td>4.593</td><td>82.36</td></tr><tr><td>FP32/Naive-BF16</td><td>95.95</td><td>28.564</td><td>31.537</td><td>93.45</td></tr><tr><td>FP32/FP32</td><td>96.24</td><td>14.370</td><td>16.690</td><td>92.80</td></tr><tr><td rowspan="4">ν</td><td>FP32/AMP</td><td></td><td></td><td></td><td>89.25</td></tr><tr><td>FP32/AA-MP</td><td>96.16 95.08</td><td>28.600</td><td>30.030 3.861</td><td></td></tr><tr><td>FP32/Naive-BF16</td><td>95.87</td><td>2.940 25.920</td><td>28.478</td><td>81.91</td></tr><tr><td></td><td></td><td></td><td></td><td>91.69</td></tr><tr><td rowspan="5">p</td><td>FP32/FP32</td><td>96.64</td><td>902.0</td><td>1155.0</td><td>92.52</td></tr><tr><td>FP32/AMP</td><td>96.17</td><td>2530.0</td><td>2634.0</td><td>92.09</td></tr><tr><td>FP32/AA-MP</td><td>95.24</td><td>190.6</td><td>292.3</td><td>83.48</td></tr><tr><td>FP32/Naive-BF16</td><td>96.08</td><td>2140.2</td><td>2338.6</td><td>91.93</td></tr><tr><td>FP32/FP32</td><td>95.97</td><td>0.0004360</td><td>0.0012770</td><td>92.66</td></tr><tr><td rowspan="5">Vt</td><td>FP32/AMP</td><td></td><td></td><td></td><td></td></tr><tr><td>FP32/AA-MP</td><td>95.54</td><td>0.0006592</td><td>0.0015200</td><td>92.59</td></tr><tr><td></td><td>95.48</td><td>0.0002104</td><td>0.0007496</td><td>88.42</td></tr><tr><td>FP32/Naive-BF16</td><td>96.12</td><td>0.0009412</td><td>0.0014996</td><td>91.01</td></tr></table>

![](images/4d3619704df5f350d970230a0cf8459ffde6a23026607faa95086ad4cbd55c8c.jpg)

![](images/7aa3a409d6d65a26f784bcf15fc4c9cff0ed1ae1f6272f4fbce7673513ebeeaa.jpg)

![](images/4d7b290f0995099793aa1f9ad315b67b248e8ec010416b29bb450a643b939b7e.jpg)

![](images/cd076b61cc30f4f6b46220dc67c6f425ba50025dfe3e59065b840980962b65e6.jpg)  
Figure 7: Empirical coverage of conformal prediction intervals across AirfRANS flow fields. Markers indicate the mean fraction of covered spatial nodes across 98 held-out CFD cases for each precision configuration. Horizontal error bars represent 95% bootstrap confidence intervals obtained by resampling evaluation cases. The dashed vertical line denotes the nominal 95% coverage target. All four configurations achieve mean coverage near the target, with FP32/AA-MP exhibiting coverage closest to 95% across the predicted fields.

![](images/8adb6186499cbde5bff5e5cd6250d14e41af6c92bd7ab377d2b2bc8af6070184.jpg)  
Figure 8: Spatial conformal uncertainty quantification for FP32/AA-MP on AirfRANS evaluation case 911. Rows show x-velocity u, y-velocity v, reduced pressure p, and turbulent viscosity vt. Columns show the reference solution, predictive mean, full conformal-calibrated prediction-interval width $W _ { k } = 2 H _ { k } .$ , and spatial interval-coverage diagnostic. Gray indicates locations covered by the interval, red indicates reference values above its upper bound, and blue indicates values below its lower bound.

The retained cold-state dispersion is interpreted as an empirical measure of parameter-dependent uncertainty, as the finite post-training Langevin trajectories have not been independently verified for posterior mixing.

## 4.8. Precision Allocation Across Replica Roles

We study whether reduced precision should be assigned to the cold replica, the hot replica, or both.

Table 8: Ablation configurations used to evaluate replica-dependent precision allocation. All preconditioned reSGLD configurations use identical hyperparameters.
<table><tr><td>Training configuration</td><td>Purpose</td></tr><tr><td>FP32/FP32</td><td>Full-precision replica-exchange baseline used to quantify the effect of low precision.</td></tr><tr><td>AA-MP/AA-MP</td><td>Evaluates the effect of applying reduced precision to both the cold and hot chain.</td></tr><tr><td>AA-MP/FP32</td><td>Evaluates the effect of applying low precision to the cold chain while retaining full precision in the hot chain.</td></tr><tr><td>FP32/AA-MP</td><td>Proposed replica-dependent precision configuration, retaining full precision for cold-chain refinement while using low precision for hot-chain exploration.</td></tr></table>

## 4.8.1. AirfRANS

AirfRANS provides the clearest evidence of precision-allocation sensitivity. Figure 9 shows that the FP32/AA-MP configuration closely follows the fully FP32 reference, whereas configurations that apply AA-MP to the cold replica exhibit a larger separation for the velocity and turbulent-viscosity fields. The pressure prediction is less sensitive to precision placement. This asymmetry supports retaining FP32 in the low-temperature refinement replica while assigning reduced precision to the higher-temperature exploratory replica.

![](images/005b9ef344b0b85d5c8d60d48a9cb8cd5b51332fc8d42c780c337f98b4212642.jpg)  
Figure 9: AirfRANS cold-replica test relative $\ell _ { 2 }$ error for four cold/hot precision allocations: FP32/FP32, AA-MP/FP32, AA-MP/AA-MP, and FP32/AA-MP. Curves show the mean over 10 independent seeds, while shaded regions indicate empirically calibrated pointwise bands based on across-seed variability. Configurations are denoted as cold-replica precision / hot-replica precision.

## 4.8.2. Jet-Engine Bracket

Figure 10 shows the corresponding result for the bracket benchmark. All four precision allocations produce nearly overlapping convergence trajectories for the reported structural fields. Although this benchmark does not strongly discriminate among the tested allocations, it confirms that FP32/AA-MP does not introduce a visible convergence penalty in this structural-mechanics problem.

![](images/071b25ec039d1d9dd7f741fb014c2ac6df5d8def448a92eee76821737649cf22.jpg)  
Figure 10: Jet-engine bracket cold-replica test relative $\ell _ { 2 }$ error for four cold/hot precision allocations: FP32/FP32, AA-MP/FP32, AA-MP/AA-MP, and FP32/AA-MP. Curves show the mean over 10 independent seeds, while shaded regions indicate empirically calibrated pointwise bands based on across-seed variability. Configurations are denoted as cold-replica precision / hot-replica precision.

Replica-exchange activity. Replica exchange remained active under both FP32/FP32 and FP32/AA-MP. Across 10 seeds, the mean swap-acceptance rate changed from 37.5 ± 10.4% for FP32/FP32 to 18.4 ± 2.8% for FP32/AA-MP on AirfRANS, and from $3 3 . 4 \pm 1 5 . 0 \% \mathrm { t o } 2 9 . 0 \pm 1 6 . 6 \%$ on the jet-engine bracket benchmark. These statistics are included only to confirm that exchange events remain active during training; the acceptance rate itself is not interpreted as a measure of predictive performance or sampling quality.

## 4.9. Computational Scaling

To examine how the computational benefit of AA-MP changes with geometry-processing workload, we vary the geometry point-cloud size while holding the remaining training configuration fixed. FP32/FP32, FP32/AMP, FP32/Naive-BF16, and FP32/AA-MP are compared at the same tested point-cloud sizes using the same reported resource metrics. Because the FP32 reference and reduced-precision measurements originate from different scaling workflows, these experiments are used to compare resource-scaling trends rather than absolute speedups.

Figure 11 shows GPU-memory and hot-replica step-time scaling as the geometry point-cloud size increases. AA-MP requires less allocated GPU memory than both FP32/FP32 and standard BF16 AMP across the tested sizes, with a widening absolute memory gap at larger point-cloud sizes. AA-MP also reduces the measured hot-replica step time relative to AMP across these configurations.

Figure 12 shows resource scaling as the Transformer embedding dimension increases from 128 to 512. Larger embeddings increase the memory required for hidden representations and intermediate activations. Across the tested dimensions, AA-MP requires less allocated GPU memory than both standard BF16 AMP and the available FP32 reference, with the absolute memory advantage increasing at larger embedding dimensions. The hot-step timing results are less uniform across reduced-precision configurations and do not establish a consistent timing advantage of AA-MP over AMP at every embedding dimension. Because the FP32 reference and reduced-precision results originate from different scaling workflows, these measurements are interpreted as resource-scaling trends rather than strictly matched speedup estimates.

![](images/21dc74443bc620e8099650f6cdcf28946560c1335d5d261b3ced871dadabcf07.jpg)

![](images/d1c37c96d37a9e5db81e6d360463180fc99efdf0e53f10ecd9c5c5afca94dab4.jpg)  
Figure 11: Geometry point-cloud-size resource scaling for FP32/FP32, FP32/AMP, FP32/Naive-BF16, and FP32/AA-MP. The geometry pointcloud size is varied from 5,000 to 25,000.

![](images/67979f2230cc4dcaf888c5106298fe43d43a03d6dce5d79381bb80367c03479b.jpg)

![](images/cfa7625e68456ce8eadc012f3a291f7c92a864866137b18c71eee0b28b4dd2e0.jpg)  
Figure 12: Embedding-dimension resource scaling for FP32/FP32, FP32/AMP, FP32/Naive-BF16, and FP32/AA-MP. The Transformer embedding dimension is varied from 128 to 512

## 5. Conclusion

This work proposed architecture-aware mixed precision (AA-MP) for replica-exchange training of geometryaware neural operators. By retaining FP32 computation in the cold-replica and employing controlled BF16 computation with selected FP32 states and reductions in the hot-replica, AA-MP assigns numerical precision according to replica roles and network architecture. Experiments on AirfRANS and jet-engine bracket benchmarks demonstrate that AA-MP reduces hot-phase GPU memory and step time relative to FP32 and standard AMP while maintaining comparable predictive accuracy. Scaling studies show a growing absolute GPU-memory advantage over AMP at larger tested workloads. Precision-allocation ablation and Naive-BF16 baseline further demonstrate that uncontrolled precision reduction degrades predictive accuracy, supporting the need for controlled mixed-precision computation. A separate conformal-UQ evaluation achieves mean spatial coverage near the nominal target with narrower mean prediction intervals, although case-level coverage varies. These findings indicate that explicit precision control can improve computational efficiency beyond standard AMP for the evaluated geometry-aware neural operators.

Future work will extend AA-MP implementation to physics-informed ArGEnT, where higher order derivatives required for PDE residual evaluation increase substantial computational costs, to investigate whether controlled mixed precision can improve computational efficiency while preserving predictive accuracy and physical consistency. Within this controlled mixed precision framework, We will further integrate uncertainty quantification with active learning to guide the selection of informative training samples, particularly for out-of-distribution geometries, and evaluate their effects on uncertainty reliability and data efficiency.

## Acknowledgement

This work was supported by Sandia National Laboratories under the Interlaboratory (IL) Laboratory Directed Research and Development (LDRD) Program. Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC (NTESS), a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy's National Nuclear Security Administration (DOE/NNSA) under contract DE-NA0003525. This written work is partially authored by an employee of NTESS. The employee, not NTESS, owns the right, title and interest in and to the written work and is responsible for its contents. Any subjective views or opinions that might be expressed in the written work do not necessarily represent the views of the U.S. Government. The publisher acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this written work or allow others to do so, for U.S. Government purposes. The DOE will provide public access to results of federally sponsored research in accordance with the DOE Public Access Plan. Guang Lin also acknowledges support from the National Science Foundation (DMS-2533878, DMS-2053746, DMS-2134209, ECCS-2328241, CBET-2347401 and OAC-2311848), and U.S. Department of Energy (DOE) Office of Science Advanced Scientific Computing Research program under the "Uncertainty Quantification for Multifidelity Operator Learning (MOLUcQ)" project (Project No. 81739), DE-SC0023161, the SciDAC LEADS Institute, and DOE–Fusion Energy Science, under grant number: DE-SC0024583.

## Appendix A. Additional AirfRANS Conformal UQ Visualizations

The following figures complement the FP32/AA-MP result in Fig. 8 by displaying the corresponding spatial uncertainty fields for the other precision configurations on the same held-out AirfRANS case 804. All figures use the same field arrangement: truth, predictive mean, calibrated 95% interval half-width, and interval-coverage diagnostic. Truth and prediction use common field-specific color limits across methods, and the half-width maps use the same field-specific scales to support direct visual comparison. The coverage categories indicate where the truth falls below, inside, or above the prediction-centered conformal interval. Aggregate 98-case results are reported in Table 7.

![](images/c3cf0d1d775772cb7a7ee5ebfe92665ecd9ac289fdcd8c10b52153fa2051035e.jpg)

![](images/63be04b4437a2af41fec40c39e7b136497fbbed01dd5e888bb7b166efc12b260.jpg)

![](images/5a4435342ed184a066854d734214309280bc781546e56fcf2476471b08f02b00.jpg)

![](images/caa327c92d59860d9e22f53137555bd90d0bcbd4de471d3aa1af39bb9a740438.jpg)

![](images/27c4f6c7c75ba635ecd7ba815272b746b989d328034aa756c1c65a5b6ad0fa8c.jpg)

![](images/5c6d6f2bf5bbf4b627ddaf1fb79ebe8ac91b7240bca6cdd1743102772f93bef5.jpg)

![](images/ada1c12bd1f25582db835c82e214846d254acc99c608f7f97ba7c3ea3bdcd57b.jpg)

![](images/485e55711359cbad605227be199a2832332cc8653217507797a13428599d6cef.jpg)

![](images/cd2fb8b0ce2e936e0dd0041bbfc3c44fdfeb94899be3f0d9dc1652ca6a7a7c7b.jpg)

![](images/482dde2c063d0369947b165995c44549a267f6ad5975fc3f05801b2f766999df.jpg)

![](images/1824b6124fb2c4e6a0d6d103bd62b62619b9a432c3f3ffbd05098b03842424a0.jpg)

![](images/b731662ac3f75ed8a349d5a62b2c401e7870f15ea7efa116f9a9d12d86417a3c.jpg)

![](images/67774af6109950dc930ee7165e5c84aa143fc2ef6b4e4d28ce4967719671859e.jpg)

![](images/db02ec9b685fe31b316a64433ed95f1276b2cd63fadff8406a192219ef47abec.jpg)

![](images/323df523f02f52a72709fa2ef0e9cb2c85a81b9bf7126418cb6d1ac0f2240d10.jpg)

![](images/044fa4180dde25516f44b98d24963f472fc195a7446586daa6ae044e02d7d600.jpg)  
Figure A.1: FP32/FP32: spatially resolved post-training conformal UQ for AirfRANS evaluation case 804. For $u , \nu , p ,$ and vt (rows), the columns display truth, predictive mean, calibrated 95% half-width, and interval coverage. Blue denotes truth below the interval, gray denotes truth inside and red denotes truth above.

![](images/61ab8545121d4f75b4b82327cb754c00a17db6d9eb20461d2b402312796cb9e8.jpg)

![](images/31bbec4340e34ff87a4bb3a4e6e97672823daee744779072747b1f25fb202f6c.jpg)

![](images/c43391316cd4a5874b58ff6b8d8b5d6f1824ab56880864807258e4d4957eaa30.jpg)

![](images/7fb7c5b7805e09ddd27de49f62374a572d51255bfd59475e22b92b0ef3ea26d8.jpg)

![](images/dfbbd2281461f651b5a3536848e33d040c74a88061073e4b613c1ad72fd3c51e.jpg)

![](images/a4094f552caed98254203dbf8715da8e0c9b1872e2f4ff999a2018e921f9c671.jpg)

![](images/c1043efbccb837b61f8abe91b2827e68b5c4bfbed1c0853409ef031bfc3da8c3.jpg)

![](images/f3124f53c301a4b7dd85be6c8e1cff1dfb1be043133144a265bdda2fdab703fb.jpg)

![](images/c22e9f4b37edc262a47be10c9fbef6a157165f17011935c7ccd869d89455481d.jpg)

![](images/cd3fbfcd16d7ae7460190dbacd475b0e6b04af6e2ad40ade9f4424ddc146fc8d.jpg)

![](images/67b8b43700dddbba0492122f9ec9e9503c350111f2bbc4afadf0574c57a3e76d.jpg)

![](images/78fd6b2cbb28479148e1349bc4b1e0a2775212c963fde6215d8ff03aa38082db.jpg)

![](images/2f230c7dd85d1851841792df2b831eb9dc31e3d241ff8e9d04d93da8eb1e4b9e.jpg)

![](images/bd3663e30403eb30d5d82e3649776c7c024667e4200a486020c8e021a63b81f4.jpg)

![](images/6a93241b8977a46ebc648e388efb685c9270266858cc36b688e9f105bf4b8111.jpg)

![](images/90f216cb4fda844274904365a9726acd76e0b4d18fd22fdb05d46cd7272de731.jpg)  
Figure A.2: FP32/AMP: spatially resolved post-training conformal UQ for the same AirfRANS evaluation case 804. Panel definitions and categorical coverage colors are identical to those of Fig. A.1.

![](images/9bec2bc3d55ab281c5103faca7232bb899d0a640f9cce508f968f3d753172ef6.jpg)

![](images/60e8e4ef1d0d8278930d3b828287314bec2b8fb3fc4b1c0f57f61bfe50a3c316.jpg)

![](images/01487cd8fad9293a6e44a373ebd9d8410f922e64a778b3119317fe93a524c2f9.jpg)

![](images/ead879ee2d2c15640cdb53266be644c48c1f6531b2800a7d6082a4af9cee4aff.jpg)

![](images/c955a165c51a4660f5f97777c98496554cf182b57404b8c372ddac8baf758887.jpg)

![](images/14b6e39f74d174cb24c2110cca5fc85b421f167b03fb431c00862fbf401efadb.jpg)

![](images/4dc5816289265013581cc95636bdc667fbc7469753dd34a9b83ebf613edd5300.jpg)

![](images/2dde2aaf104794168b349939a0577e9eebb2c5f3ed21685e93b2b5628e4a301f.jpg)

![](images/39fc92af5312f88ccfcb5052a476519cefdfa519832f7cab8d455b4737c0df4d.jpg)

![](images/cb1f45ee93cea3ef8c0b28cf305f7483f5505882aa842b15090aa4a392b58c73.jpg)

![](images/0a5972f6e9ae799a7204c06840bae5fd8e8eee8bc56c3da6c66202728897caf7.jpg)

![](images/f94286e09bbcc18356b4f0b5d818608b50f98659f8f6ec89169957e39384787d.jpg)

![](images/34f97f323277d04982aa02aa7d076058eeadea2c3ad31f0e8da00f06ffffd420.jpg)

![](images/c47adf1cbbbe734212b6663b01544088ac42ffbda27a8f0b5fe21964d957dddb.jpg)

![](images/c76dba433f3f92d450f47c0d53997d29013e59ec2a836bcf1b32d169c5fbcd56.jpg)

![](images/27b4dc6844f8dd253fc622b24a472dcb568f25b0fab6e0b742b8a6546ac7a6fe.jpg)  
Figure A.3: FP32/Naive-BF16: spatially resolved post-training conformal UQ for AirfRANS evaluation case 804. The panels show truth, predictive mean, calibrated 95% half-width, and categorical interval coverage. Aggregate calibration and interval-score results are reported in Table 7.

## References

[1] M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations, Journal of Computational Physics 378 (2019) 686–707.

[2] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, L. Yang, Physics-informed machine learning, Nature Reviews Physics 3 (6) (2021) 422–440

[3] L. Lu, P. Jin, G. E. Karniadakis, Learning nonlinear operators via deeponet based on the universal approximation theorem of operators, Nature Machine Intelligence 3 (3) (2021) 218–229.

[4] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, Fourier neural operator for parametric partial differential equations, International Conference on Learning Representations (2021).

[5] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhattacharya, A. Stuart, A. Anandkumar, Neural operator: Learning maps between function spaces with applications to pdes, Journal of Machine Learning Research 24 (89) (2023) 1–97.

[6] K. Shukla, V. Oommen, A. Peyvan, M. Penwarden, N. Plewacki, L. Bravo, A. Ghoshal, R. M. Kirby, G. E. Karniadakis, Deep neural operators as accurate surrogates for shape optimization, Engineering Applications of Artificial Intelligence 129 (2024) 107615.

[7] W. Chen, Y. Fu, M. Penwarden, P. Roy, P. Stinis, Argent: Arbitrary geometry-encoded transformer for operator learning (2026). arXiv:2602.11626.

[8] P. Micikevicius, S. Narang, J. Alben, G. Diamos, E. Elsen, D. Garcia, B. Ginsburg, M. Houston, O. Kuchaiev, G. Venkatesh, et al., Mixed precision training, 2017.

[9] A. Abdelfattah, H. Anzt, E. G. Boman, E. Carson, T. Cojean, J. Dongarra, A. Fox, M. Gates, N. J. Higham, X. S. Li, J. Loe, P. Luszczek, S. Pranesh, S. Rajamanickam, T. Ribizel, B. F. Smith, K. Swirydowicz, S. Thomas, S. Tomov, Y.-M. Tsai, U. M. Yang, A survey of numerical linear algebra methods utilizing mixed-precision arithmetic, The International Journal of High Performance Computing Applications 35 (4) (2021) 344–369.

[10] R. Tu, C. White, J. Kossaifi, B. Bonev, N. Kovachki, G. Pekhimenko, K. Azizzadenesheli, A. Anandkumar, Guaranteed approximation bounds for mixed-precision neural operators, in: International Conference on Learning Representations, 2024.

[11] E. Celledoni, B. Owren, L. Ruthotto, T. N. Yang, Mixed precision training of neural odes, arXiv preprint arXiv:2510.23498 (2026).

[12] D. Bablani, J. L. Mckinstry, S. K. Esser, R. Appuswamy, D. S. Modha, Efficient and effective methods for mixed precision neural network quantization for faster, energy-efficient inference, arXiv preprint arXiv:2301.13330 (2024).

[13] Z. Sun, C. Ge, J. Wang, M. Lin, H. Chen, H. Li, X. Sun, Entropy-driven mixed-precision quantization for deep network design, in: Advances in Neural Information Processing Systems, Vol. 35, 2022, pp. 21508–21520.

[14] T. Qin, Z. Li, J. Zhao, Y. Yan, Y. Du, Mixed precision quantization based on information entropy, Scientific Reports 15 (2025) 12974. doi:10.1038/s41598-025-91684-8.

[15] J. Langou, J. Langou, P. Luszczek, J. Kurzak, A. Buttari, J. Dongarra, Exploiting the performance of 32 bit floating point arithmetic in obtaining 64 bit accuracy, in: Proceedings of the 2006 ACM/IEEE Conference on Supercomputing, 2006, p. 50.

[16] J. Kurzak, J. Dongarra, Implementation of mixed precision in solving systems of linear equations on the cell processor, Concurrency and Computation: Practice and Experience 19 (10) (2007) 1371–1385.

[17] N. J. Higham, T. Mary, Mixed precision algorithms in numerical linear algebra, Acta Numerica 31 (2022) 347– 414.

[18] A. Haidar, S. Tomov, J. Dongarra, N. J. Higham, Harnessing gpu tensor cores for fast fp16 arithmetic to speed up mixed-precision iterative refinement solvers, in: Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis, 2018, pp. 1–11.

[19] A. Haidar, H. Bayraktar, S. Tomov, J. Dongarra, N. J. Higham, Mixed-precision iterative refinement using tensor cores on gpus to accelerate solution of linear systems, Proceedings of the Royal Society A: Mathematical. Physical and Engineering Sciences 476 (2243) (2020) 20200110.

[20] J. D. Hogg, J. A. Scott, A fast and robust mixed-precision solver for the solution of sparse symmetric linear systems, ACM Transactions on Mathematical Software 37 (2) (2010) 1–24.

[21] Y. Idomura, T. Ina, Y. Ali, T. Imamura, Acceleration of fusion plasma turbulence simulations using the mixedprecision communication-avoiding krylov method, 2020, pp. 1–13. doi:10.1109/SC41405.2020 .00097.

[22] S. Abdulah, Q. Cao, Y. Pei, G. Bosilca, J. Dongarra, M. G. Genton, D. E. Keyes, H. Ltaief, Y. Sun, Accelerating geostatistical modeling and prediction with mixed-precision computations: A high-productivity approach with parsec, IEEE Transactions on Parallel and Distributed Systems 33 (4) (2022) 964–976.

[23] C. M. Maynard, D. N. Walters, Mixed-precision arithmetic in the endgame dynamical core of the unified model, Computer Physics Communications 244 (2019) 69–75.

[24] I. Yamazaki, S. Tomov, J. Dongarra, Mixed-precision cholesky qr factorization and its case studies on multicore cpu with multiple gpus, SIAM Journal on Scientific Computing 37 (3) (2015) C307–C330

[25] J. Hayford, J. Goldman-Wetzler, E. Wang, L. Lu, Speeding up and reducing memory usage for scientific machine learning via mixed precision, Computer Methods in Applied Mechanics and Engineering 428 (2024) 117093 doi:10.1016/j.cma.2024.117093

[26] M. Welling, Y. W. Teh, Bayesian learning via stochastic gradient langevin dynamics, in: Proceedings of the 28th International Conference on Machine Learning, 2011, pp. 681–688.

[27] C. Li, C. Chen, D. Carlson, L. Carin, Preconditioned stochastic gradient langevin dynamics for deep neural networks, in: Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 30, 2016. doi: 10.1609/ aaai.v30i1.10200.

[28] K. Hukushima, K. Nemoto, Exchange monte carlo method and application to spin glass simulations, Journal of the Physical Society of Japan 65 (6) (1996) 1604–1608. doi:10.1143/JPSJ.65.1604.

[29] W. Deng, Q. Feng, L. Gao, F. Liang, G. Lin, Non-convex learning via replica exchange stochastic gradient mcmc, in: Proceedings of the 37th International Conference on Machine Learning, Vol. 119 of Proceedings of Machine Learning Research, PMLR, 2020, pp. 2474–2483.

[30] G. Li, G. Lin, Z. Zhang, Q. Zhou, Fast replica exchange stochastic gradient langevin dynamics, Statistical Papers 67 (2026). doi:10.1007/s00362-026-01896-x.

[31] F. Bonnet, J. A. Mazari, P. Cinnella, P. Gallinari, AirfRANS: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier-stokes solutions, in: Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https://arxiv.org/abs/2212.07564

[32] J. Park, N. Kang, Point-deeponet: Predicting nonlinear fields on non-parametric geometries under variable load conditions, Neural Networks 198 (2026) 108560. doi:10.1016/j.neunet .2026.108560. URL http://dx.doi.org/10.1016/j.neunet.2026.108560

[33] S. Hong, Y. Kwon, D. Shin, J. Park, N. Kang, Deepjeb: 3d deep learning-based synthetic jet engine bracket dataset (2024). arXiv:2406.09047. URL https://arxiv.org/abs/2406.09047

[34] C. Moya, A. Mollaali, Z. Zhang, L. Lu, G. Lin, Conformalized-DeepONet: A distribution-free framework for uncertainty quantification in deep operator networks, Physica D: Nonlinear Phenomena 471 (2025) 134418. doi:10.1016/j.physd.2024.134418.

[35] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster, Conformal risk control, in: International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html

[36] T. Gneiting, A. E. Raftery, Strictly proper scoring rules, prediction, and estimation, Journal of the American Statistical Association 102 (477) (2007) 359–378. doi:10.1198/016214506000001437.