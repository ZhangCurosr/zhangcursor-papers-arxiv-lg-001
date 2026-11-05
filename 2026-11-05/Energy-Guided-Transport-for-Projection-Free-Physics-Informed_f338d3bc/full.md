# Energy-Guided Transport for Projection-Free Physics-Informed Flow Matching

Giuliano Galadini Aurelio Uncini Danilo Comminiello

Department of Information Engineering, Electronics and Telecommunications

Sapienza University of Rome

Rome, Italy

{giuliano.galadini, aurelio.uncini, danilo.comminiello}@uniroma1.it

## Abstract

Deep generative models, particularly Flow Matching frameworks, have emerged as powerful tools for synthesizing complex physical systems. However, standard architectures remain purely statistically driven and agnostic to the underlying governing equations, often violating fundamental physical constraints during inference. Existing zero-shot approaches designed to enforce these constraints typically rely on rigid algebraic projections. While effective for sparse boundary conditions, applying such discrete geometric corrections to dense, full-domain PDEs may disrupt the probability path, potentially deviating from the learned manifold and introducing substantial computational overhead. To address these challenges, we introduce Energy-Guided Transport, a novel training-free framework that formulates non-linear physical constraints as a smooth potential energy landscape. Rather than interrupting the ODE integration, our method applies a continuous dissipative force directly to the intermediate generative states, steering them toward the physical manifold. By evaluating this force implicitly via reverse-mode automatic differentiation, our framework preserves the natural dynamics of the flow while maintaining an efficient O(N) algorithmic complexity. We extensively evaluate our framework across various 1D and 2D physical systems, including the chaotic Navier-Stokes equations. Our method achieves stable generation without structural divergence, significantly improves the stability-accuracy trade-off compared to state-of-the-art methods, and substantially accelerates inference with respect to exact projection strategies. Code available at: https://github.com/ispamm/ Projection-free-energy-guided-flow-matching.

## 1 Introduction

The accurate simulation of complex spatiotemporal dynamics has traditionally relied on computationally expensive numerical solvers. To bypass the curse of dimensionality inherent in high-dimensional PDEs, deep neural networks were initially introduced as direct physics solvers [1, 2]. Building on this foundation, deep generative models, ranging from Score-Based Diffusion [3, 4] to Flow Matching frameworks [5, 6], have recently emerged as scalable data-driven surrogates for modeling physical systems [7–10]. These models have achieved breakthrough successes in global weather forecasting and fluid dynamics [11–16].

By learning the intrinsic data manifold from high-fidelity solvers, these generative architectures offer unprecedented scalability, producing complex, high-dimensional physical states in a fraction of the time required by traditional simulators [8, 11, 12]. However, standard generative models remain purely statistical; they synthesize visually convincing fields but lack an inherent understanding of the governing physics. This purely statistical nature often leads to severe violations of conservation laws, boundary conditions, and internal dynamic consistencies [17]. Ultimately, violations of these macroscopic constraints severely degrade out-of-distribution generalization [18] and the reliability of foundation models for PDEs [19].

![](images/25945dcd193b31924a4dca29cfaede40b149cca66d0bc90b84f36741a7cd508f.jpg)  
(a) Explicit Hard Projections

![](images/701ed71178dfce29fc592a434c6565af620f5c05f48828a54aacaef8620ce899.jpg)  
(b) EGT-FFM (Ours)  
Figure 1: Conceptual comparison of generative probability flows. (a) Algebraic projections interrupt continuous integration, potentially displacing trajectories from the learned data manifold. (b) Our dissipative guidance smoothly steers the trajectory along the data manifold towards the physical equilibrium.

To bridge this gap, historical techniques attempted injecting physical priors into the generative process. Early approaches primarily focused on data-driven inverse problems and inpainting [20–22]. In these settings, models rely heavily on dense interior observations (e.g., sparse sensor data) to anchor the generative trajectory, reducing the physics problem to a conditioned interpolation task. While highly effective for data assimilation, these observation-dependent methods fail to act as rigorous, independent PDE solvers. Consequently, the field recently shifted toward Zero-Shot Physics Guidance, a much stricter paradigm. In this regime, a pre-trained base model is guided exclusively by the analytical PDE and its formal Boundary/Initial Conditions (BC/IC) during inference, without access to interior ground-truth data or computationally prohibitive retraining.

Despite the theoretical elegance of zero-shot inference, modern guidance architectures suffer from critical structural and computational limitations. Existing approaches often face trade-offs between numerical stability and computational scalability: gradient-based approximations can become unstable in non-linear regimes, while explicit algebraic projections (e.g., PCFM [23]), though effective for sparse boundary conditions, incur significant computational overhead and may perturb the generative trajectory when applied to dense, full-domain PDE constraints.

To address these limitations, we propose Energy-Guided Transport (EGT), a continuous guidance framework for physics-informed Functional Flow Matching. Rather than relying on unstable approximations or discrete geometric projections that intermittently halt the generative process, we formulate the physical PDE residual as a smooth potential energy landscape. Central to our approach are the Dissipation properties of Flow Matching, which proves that modulating the generative vector field with the gradient of this physical energy applies a Lyapunov-like dissipative force. As conceptualized in Figure 1, this allows our guidance to natively harmonize with the probability flow: it smoothly counteracts physical violations along the trajectory, steering samples toward the true physical manifold without the structural discontinuities or $\mathcal { O } \dot { ( N ^ { 2 } ) }$ memory bottlenecks of exact algebraic projections.

To empirically validate our theoretical claims, we conduct an exhaustive experimental campaign across four distinct physical systems of increasing non-linearity and dimensionality: the 1D Heat equation, 1D Reaction-Diffusion (RD), 1D Burgers’ equation, and the chaotic 2D Navier-Stokes equations. Most importantly, for each physical system, we rigorously evaluate our framework under two distinct regimes: partial physics guidance (enforcing only boundary and initial conditions) and a highly constrained full-PDE stress test. Furthermore, we provide extensive ablation studies investigating the hyperparameter dynamics of the dissipative force and the structural impact of generative priors, comparing standard White Noise against spatially correlated Gaussian Random Fields (Smooth Kernels).

Our primary contributions are summarized as follows:

• Theoretical Formulation of Dissipative Guidance: we establish the dissipation properties of Flow Matching, demonstrating via a Lagrangian perspective how a continuous dissipative force ensures the minimization of physical energy functionals. By steering the vector field natively rather than interrupting the ODE to compute discrete projections, our framework seamlessly preserves the foundational continuity of Flow Matching.

• Analysis of Projection-Based Guidance: we provide a rigorous empirical analysis exposing the structural limitations of explicit projection methods. We demonstrate that enforcing rigid algebraic constraints can disrupt the generative flow, highlighting the trade-off between exact boundary matching and internal physical coherence (the waterbed effect).

• Computational Scalability: by completely avoiding explicit Jacobian materialization and employing VJP-based dissipative guidance coupled with gradient norm clipping, our method circumvents the $\mathcal { O } ( N ^ { 2 } )$ memory bottlenecks, enabling scalable full-PDE guidance on 2D fluid dynamics.

• Favorable Speed–Stability–Accuracy Trade-Off: through a strict zero-shot evaluation protocol across four benchmark datasets, our method is capable of maintaining consistent numerical stability (100% success rate) while delivering competitive distributional fidelity and low physical residuals at up to two orders of magnitude faster inference speeds than explicit projection methods.

## 2 Related Work

Generative Models for Physical Systems. While early advancements in modeling complex physical dynamics relied heavily on Score-Based Diffusion Models [4], recent attention has shifted toward Flow Matching (FM) frameworks [5, 6, 24], drawing deep connections with computational optimal transport [25–27]. By formulating generation as an Ordinary Differential Equation (ODE) [28] and learning a continuous vector field, FM provides faster convergence and structurally smoother trajectories, seamlessly adapting even to complex geometries and Riemannian manifolds [29–32]. In the context of spatiotemporal physics, Functional Flow Matching (FFM) combines continuous normalizing flows with Neural Operators (e.g., Fourier Neural Operators [33], DeepONet [34], and general operator frameworks [35, 36]). Recently, the extension of these score-based and flow paradigms to functional spaces has gained significant traction [37, 38]. To overcome this purely statistical nature, the literature has historically turned to physics-informed training.

Physics-Informed Training. The historical paradigm for injecting physics into neural networks incorporates analytical residuals directly into the training objective (e.g., PINNs [39–41]). While highly successful for specific forward and inverse regression tasks, applying this paradigm to generative modeling fundamentally limits generalizability. Enforcing physics during training structurally couples the generative prior to specific boundary and initial conditions. Consequently, adapting the model to novel constraints requires computationally prohibitive retraining, rendering these approaches inflexible. Furthermore, directly optimizing physical residuals alongside data objectives during network training often suffers from severe gradient flow pathologies, where competing loss terms prevent convergence [42, 43]. This limitation has motivated the recent paradigm shift toward Zero-Shot guidance.

Zero-Shot Physics Guidance. Zero-shot inference methods attempt to correct generative trajectories using analytical PDE constraints without altering the pre-trained weights. Drawing inspiration from classifier-free [44] and universal guidance [45] in vision tasks, these approaches generally fall into three categories:

• Optimization-Based Guidance: Methods such as DiffusionPDE [46] and D-Flow [47] utilize gradients of physical residuals to shift the generative trajectory. For instance, D-Flow optimizes the initial noise to align the final state with the constraints. However, because diffusion backbones are heavily corrupted by noise, computing physical gradients directly on intermediate states is unstable. Consequently, these methods often rely on a “predictand-correct" mechanism using denoised terminal predictions $( \mathrm { e . g . }$ , via Tweedie’s formula), which can lead to overshooting in highly non-linear dynamics.

• Projection-Based Guidance: To avoid gradient-based instability, frameworks like Extrapolation, Correction, Interpolation (ECI) [48] and Physics-Constrained Flow Matching (PCFM) [23] apply explicit geometric corrections. ECI is a gradient-free approach that interleaves extrapolation, orthogonal projection, and interpolation to strictly enforce linear constraints. PCFM extends this concept to arbitrary nonlinear constraints using Gauss-Newton updates. Notably, while PCFM is highly memory-efficient $( { \mathcal { O } } ( n )$ complexity) for sparse constraints by solving a small $m \times m$ Schur complement system, applying its Jacobian-based projections to dense, full-domain PDE residuals incurs prohibitive computational costs. Furthermore, rigid geometric projections can sometimes force the intermediate states off the learned data manifold.

• Proximal-Based Guidance: To reconcile strict physical consistency with the pre-trained prior, recent frameworks like ProFlow [49] reframe the correction step as a proximal optimization problem. ProFlow alternates between a terminal optimization step, which anchors the prediction to the physically consistent set via proximal minimization, and an interpolation step that restores compatibility with the learned flow probability path. This implicitly balances trust in the generative prior against physical and data fidelity, avoiding both the intractability of full Jacobian inversions and the trajectory disruption caused by pure algebraic projections.

Unlike these paradigms, our Energy-Guided FFM avoids both unrolled terminal-state estimations and discrete intermediate projections. By applying a continuous dissipative force directly to the intermediate states, we natively harmonize physical consistency with the probability flow while maintaining strict algorithmic scalability.

## 3 Methodology

## 3.1 Problem Formulation

We consider physical systems governed by time-dependent Partial Differential Equations (PDEs) [50] defined over a bounded spatial domain $\dot { \Omega } \subset \mathbb { R } ^ { d }$ and a time interval [0, T]. Let $\bar { u } : \Omega \times [ 0 , T ]  \mathbb { R } ^ { n }$ denote the spatio-temporal solution field. A generalized physical system is strictly characterized by its internal dynamics, initial conditions (IC), and boundary conditions (BC):

$$
\partial _ { t } \boldsymbol { u } + \boldsymbol { \mathcal { N } } ( \boldsymbol { u } ; \theta ) = 0 \mathrm { i n } \Omega \times ( 0 , T ] , \quad \boldsymbol { u } ( \cdot , 0 ) = \boldsymbol { u } _ { 0 } \mathrm { i n } \Omega , \quad \boldsymbol { \mathcal { B } } ( \boldsymbol { u } ) = 0 \mathrm { o n } \partial \Omega \times [ 0 , T ] .\tag{1}
$$

where N is a spatial differential operator, θ denotes the physical parameters, and B is the boundary operator. We denote the manifold of all physically admissible functions that exactly satisfy Eq. (1) as $\boldsymbol { \mathcal { U } } _ { t r u e }$

In generative scientific machine learning, we assume access to a pre-trained base model (e.g., an unguided Flow Matching network) that approximates the empirical data distribution. However, due to finite data and neural approximations, this unguided model generates solutions from a broader, unconstrained support $\mathcal { U } _ { g e n }$ , frequently violating the strict structural properties of $\boldsymbol { \mathcal { U } } _ { t r u e }$

Our objective is to formulate a zero-shot inference mechanism that smoothly steers the generative trajectories toward the physically admissible manifold $\boldsymbol { \mathcal { U } } _ { t r u e }$ . To achieve this, we encapsulate the targeted physical laws into a general constraint operator $h ( u ) = 0$ . Depending on the downstream task, $h ( u )$ can represent partial constraints (e.g., IC/BC enforcement) or the highly non-linear, full microscopic PDE residual. A comprehensive theoretical taxonomy of these constraint operators and their energy landscapes is provided in Appendix B.

## 3.2 Functional Flow Matching Foundations

Our framework builds upon Functional Flow Matching (FFM) [38], which generalizes diffusion paradigms to infinite-dimensional Hilbert spaces ${ \mathcal { H } } \left( \mathrm { e . g . , } L ^ { 2 } ( \Omega ) \right)$ ). Let $p _ { 0 } ( u )$ be a tractable prior distribution over functions (e.g., Gaussian random fields) and $p _ { 1 } ( u )$ be the empirical target distribution of valid physical states. FFM learns a neural functional vector field $v _ { \theta } ( u , \bar { t } ) : \mathcal { H } \times \mathsf { \bar { \rho } } [ 0 , 1 ] \to \mathcal { H }$ to

Algorithm 1 Energy-Guided Transport for Functional Flow Matching   
Require: Pre-trained vector field v , Constraint $h ( u )$ , Energy $\begin{array} { r } { E ( u ) = \frac { 1 } { 2 } \| h ( u ) \| _ { 2 } ^ { 2 } } \end{array}$   
Require: Integration steps N, Max guidance $\gamma _ { m a x } ,$ Clip threshold c   
1: Sample initial state u<sub>0</sub> $\sim p _ { 0 } ( u )$ ▷ e.g., Smooth Gaussian Kernel   
2: Set step size $\Delta t = 1 / N$   
3: u ← u<sub>0</sub>   
4: for $i = 0$ to $N - 1$ do ▷ Continuous Guided Flow   
5: $t \gets i \cdot \Delta t$   
6: γ ← ComputeSchedule $( \gamma _ { m a x } , t )$ ▷ e.g., Parabolic or Sine   
7: u ← RequireGradient(u)   
8: $v _ { \mathrm { b a s e } } \gets v _ { \theta } ( u , t )$ ▷ Evaluate neural prior   
9: $g \gets \nabla _ { u } E ( u )$ ▷ Compute implicit VJP via reverse-mode AD   
10: $\tilde { g }  g$ · min $\left( 1 , { \frac { c } { \| g \| _ { 2 } } } \right)$ ▷ Apply gradient clipping   
11: u ← StopGradien $\mathrm { t } ( u ) + \Delta t ( v _ { \mathrm { b a s e } } - \gamma _ { t } \tilde { g } )$ ▷ Euler ODE update   
12: end for   
13: return Generated physical field u

parameterize the probability-flow ODE:

$$
\frac { d u } { d t } = v _ { \theta } ( u , t ) , \quad u ( 0 ) \sim p _ { 0 } ,\tag{2}
$$

which continuously transports probability mass such that the marginal density $p _ { t } ( u )$ interpolates between $p _ { 0 }$ and $p _ { 1 }$ . Utilizing an Optimal Transport (OT) formulation, the conditional vector field is defined along deterministic paths as $v _ { t } ( u | u _ { 1 } ) = u _ { 1 } - u _ { 0 }$

Operating directly in function space yields two structural advantages for PDE-constrained generation. First, it ensures resolution invariance: the learned field $v _ { \theta }$ can be evaluated at arbitrary spatial resolutions during inference, mitigating numerical discretization errors when computing the physical constraint $h ( u )$ . Second, intermediate states $u _ { t }$ evolve as continuous, structurally coherent fields rather than unstructured discrete noise arrays, enabling the stable evaluation of spatial differential operators at any integration timestep.

## 3.3 Energy-Guided Transport

Formulating physical dynamics through energy or Hamiltonian landscapes has shown great success in forward modeling [51, 52]. We extend this principle to generative guidance, leveraging implicit differentiable physics [53]. To embed physical knowledge directly into the probability flow without breaking its continuous-time trajectory, we define a physical energy functional $\begin{array} { r } { E ( u ) = \frac { 1 } { 2 } \| h ( u ) \| _ { 2 } ^ { 2 } } \end{array}$ where $\bar { h ( u ) } \in \mathbb { R } ^ { m }$ is the discretized non-linear PDE residual. We augment the baseline neural vector field $v _ { \theta } ( u , t )$ by injecting the functional gradient of this energy, effectively interpreting it as a time-annealed, Gibbs-like log-prior:

$$
\begin{array} { r } { \bar { v } _ { t } ( u ) = v _ { \theta } ( u , t ) - \gamma ( t ) \nabla _ { u } E ( u ) , \quad u ( 0 ) \sim p _ { 0 } } \end{array}\tag{3}
$$

where $\gamma ( t ) \geq 0$ is a scheduling function acting as the inverse temperature $1 / \tau ( t )$ . This thermodynamic cooling schedule allows the generative prior to dictate the macroscopic structure at high temperatures $( t \approx 0 )$ before strictly locking onto the physical manifold $( E ( u ) \bar { = } 0 )$ at low temperatures (t → 1).

Scalable Dissipative Force. Computing the guidance field $\nabla _ { u } E ( u )$ requires differentiating through complex, non-linear spatial operators. By the chain rule, $\nabla _ { \boldsymbol { u } } \dot { E } ( \boldsymbol { \dot { u } } ) \dot { = } J _ { h } ( \boldsymbol { u } ) ^ { T } h ( \boldsymbol { u } )$ , where $J _ { h } \in$ $\mathbb { R } ^ { m \times N }$ is the constraint Jacobian. Algebraic projection methods $( \mathrm { e . g } ^ { } )$ ., PCFM [23]) enforce constraints by explicitly materializing and solving systems involving $J _ { h } J _ { h } ^ { T }$ . While highly memory-efficient for sparse boundaries (where $m \ll N )$ , this geometric approach incurs an intractable $\mathcal { O } ( N ^ { 2 } )$ memory bottleneck when scaled to dense, full-domain PDE residuals $( m \approx N )$ . Our energy-based formulation naturally circumvents this limitation: rather than materializing $J _ { h }$ , the dissipative force is evaluated implicitly via standard reverse-mode automatic differentiation (Vector-Jacobian Products). This guarantees strict $\mathcal { O } ( N )$ space and time complexity, effortlessly scaling full-PDE guidance to high-dimensional fields without structural bottlenecks (see Appendix H for a complexity analysis). Furthermore, by relying exclusively on matrix-vector multiplications via VJPs rather than inverting local linear systems, EGT naturally handles ill-conditioned or rank-deficient Jacobians without numerical instability (producing zero gradients along rank-deficient directions). Unlike approximate Gauss-Newton or preconditioned iterative solvers that still require multiple inner-loop evaluations per step to approximate a projection, EGT embeds the physical correction directly into the transport dynamics via a single VJP evaluation.

![](images/8c2a8945bdfa3bca01c196860675d39cc5da918dfb2d03fbfa991f81558a9698.jpg)  
Figure 2: Qualitative evaluation on the 1D Burgers’ equation (Full-PDE Stress Test). Spatiotemporal velocity fields $u ( x , t )$ generated from the same initial conditions. Explicit algebraic projections (PCFM) can be sensitive to solver step sizes under non-linear shock dynamics, leading to numerical instabilities and high-frequency artifacts. In comparison, continuous Energy-Guided Transport provides smooth updates along the trajectory, accurately resolving sharp shockfronts while maintaining distributional fidelity.

Theoretical Guarantees. To show that Eq. (3) introduces a principled dissipative correction without arbitrarily perturbing the underlying data manifold, we establish the following propositions governing the continuous transport of probability mass (full analytical proofs are provided in Appendix A).

Proposition. (Continuous Energy Dissipation under Ideal Transport) Let $\mathbb { E } _ { p _ { t } } [ E ( u ) ]$ denote the expected physical energy over the marginal distribution at time t. The instantaneous variation of the physical error under the augmentedfield decomposes into the baseline variation minus a strictly non-negative dissipative term:

$$
\frac { d \mathbb { E } _ { t o t a l } } { d t } = \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E , v _ { \theta } \rangle \right] - \gamma ( t ) \mathbb { E } _ { p _ { t } } \left[ \| \nabla _ { u } E \| _ { 2 } ^ { 2 } \right]\tag{4}
$$

Consequently, for any $\gamma ( t ) \geq 0 ,$ , the continuous injection strictly dampens the physical violation relative to the baseline ODE trajectory, effectively acting as a Lyapunov-like dissipative mechanism for the physical constraint.

While Proposition 3.3 holds identically for the generative flow, in practice, the neural field $v _ { \theta }$ exhibits an approximation error $\delta ( u , t )$ relative to the optimal transport field. By bounding this error such that $\lVert \delta ( u , t ) \rVert \leq \epsilon _ { t }$ , we formulate the global descent guarantees for the practical implementation:

Corollary. (Error-Bounded Global Energy Descent) The true rate ofchange ofthe expectedphysical energy is bounded by the ideal transport variation and an error term proportional to the network’s inaccuracy. Integrating this instantaneous rate over the generation interval $t \in [ 0 , 1 ]$ yields the macroscopic physical variation:

$$
\mathbb { E } [ E ( u _ { 1 } ) ] - \mathbb { E } [ E ( u _ { 0 } ) ] \leq \Delta E _ { b a s e } - \int _ { 0 } ^ { 1 } \gamma ( t ) \mathbb { E } _ { p _ { t } } \left[ \| \nabla _ { u } E \| _ { 2 } ^ { 2 } \right] d t + { \mathcal { O } } ( { \bar { \epsilon } } )\tag{5}
$$

The instantaneous and cumulative forms of this bound are stated and proved as Corollaries 1 and 2 in Appendix A.

Scope of the Theoretical Guarantee. Importantly, Proposition 3.3 and Corollary 3.3 establish a continuous dissipative correction relative to the unguided flow under finite-step inference, rather than asymptotic convergence to $E ( u _ { 1 } ) = 0$ . Instantaneous energy decrease is guaranteed whenever the dissipative term $\gamma ( \dot { t } ) \lVert \nabla _ { u } E \rVert _ { 2 } ^ { 2 }$ dominates the baseline alignment $\langle \nabla _ { u } E , v _ { \theta } \rangle$ ; if the learned flow strongly opposes the physical gradient, the residual may temporarily increase. When machineprecision constraint satisfaction is strictly required, a lightweight terminal refinement can be applied at t = 1, motivating our Hybrid EGT variant (see Section 3.5).

![](images/d2c6abeb8fa7ff35cac6f391982bec2a16052bd29dcf4b0f480024981ed43d9b.jpg)

![](images/b71d02ec5941bbeadf8e63a6d9b087425b8837d3eadd7b670cf74d3c80667042.jpg)

![](images/b8ed634c0b94ef74f1e0e1940132889c55c4b671b14faa4037d4349e08c5b12b.jpg)

![](images/b1fcd97af002d4e7f91adba4a2ff28145331edc35ce20145f3d72fe7d28ff65c.jpg)

![](images/541021040b7e93659daa1a176930c0ab57995de1c46b6a6f31907d2dbc97d46b.jpg)  
Figure 3: Qualitative evaluation on 2D Navier-Stokes vorticity fields. While all guided frameworks attempt to satisfy the underlying physical constraints, (c) ECI and (d) DiffusionPDE tend to introduce noticeable structural blurring and dampen the vorticity intensity. In contrast, our proposed (b) EGT better preserves the macroscopic structural coherence and the sharp gradients of the flow, offering a closer approximation to the (a) Ground Truth dynamics.

## 3.4 Architectural Synergies: FFM and Smooth Priors

While Proposition 3.3 theoretically justifies continuous guidance across any probability flow, its practical success is deeply synergistic with the structural properties of Functional Flow Matching. Unlike diffusion models, where intermediate states are heavily corrupted by stochastic noise, FFM constructs a deterministic ODE path. Consequently, intermediate states $u _ { t }$ rapidly emerge as structurally coherent fields. This coherence allows for the stable, direct evaluation of spatial differential operators $( \mathrm { e . g . , } \nabla u _ { t } , \Delta u _ { t } )$ and the physical energy $E ( u _ { t } )$ along the generative trajectory, completely bypassing the need for terminal-state approximations.

Furthermore, this synergy can be naturally extended through the choice of the generative prior. While our dissipative guidance is highly effective under standard White Noise, utilizing a spatially correlated Gaussian Random Field (Smooth Kernel) ensures inherent spatial differentiability right from $t = 0$ By providing a well-behaved energy landscape from the very first integration step, this formulation allows for the stable evaluation of physical derivatives across the entire generative trajectory, fully unlocking the potential of continuous guidance without early-stage numerical truncation.

## 3.5 Practical Implementation and Sampling Algorithm

The complete generative inference process is summarized in Algorithm 1. To ensure stability across highly non-linear PDE landscapes without sacrificing the $\mathcal { O } ( N )$ computational complexity, our implementation incorporates three core mechanisms: ${ \check { L } } ^ { 2 }$ -norm gradient clipping, dynamic guidance scheduling, and first-order Euler integration.

We formally define two variants of our framework. Our primary, natively continuous approach is denoted as EGT. Additionally, we evaluate Hybrid EGT, which incorporates an optional post-hoc refinement phase (pure energy descent at $t = 1 )$ to strictly anchor the final sample to the physical manifold. Rather than replacing terminal refinement, pure EGT replaces expensive repeated projections along the flow, steering the trajectory into a favorable physical “basin of attraction” that serves as an optimal initialization for lightweight post-hoc refinement. Unless otherwise specified, our primary visualizations highlight pure EGT. This isolates our core hypothesis: continuous guidance during the flow is fundamentally superior to discrete algebraic corrections. Comprehensive implementation details, hyperparameter configurations, and the exact formulation of the guidance schedules are provided in Appendix G.

## 4 Experimental Results

To rigorously evaluate our Energy-Guided Transport, we benchmark against state-of-the-art physicsguided generative models: ProFlow [49], ECI [48], DiffusionPDE [46], D-Flow [47], and PCFM [23]. We test across four physical systems: 1D Heat, 1D Reaction-Diffusion (RD), 1D Burgers’, and 2D Navier-Stokes. To isolate their capabilities as pure PDE solvers and guarantee strict fairness, we adopt the zero-shot evaluation framework established by PCFM [23]. Specifically, all competing methods are denied access to interior ground-truth data and share the identical pre-trained FFM backbone and initial noise realizations. Performance is evaluated via Success Rate (SR) for numerical stability, Inference Speed, Sample MSE for distributional fidelity, and Physical Residual (Phys) MAE. Full formulations and extended metrics are deferred to Appendices C and J. Table 1 summarizes quantitative performance under a standard White Noise prior, while Table 2 details the corresponding performance under a spatially correlated Smooth Kernel prior. We evaluate all methods under two constraint regimes: Partial Guidance (Boundary/Initial Conditions only) and the Full-PDE Stress Test. Comprehensive visualizations are provided in our Github repository.

Table 1: Quantitative comparison across PDE datasets (White Noise Prior). Panel A shows performance with weak constraints (BC/IC only). Panel B shows the stress test constrained by the Full PDE. For each dataset, we report Success Rate (SR, %), average inference speed (Spd, sec/sample), Sample MSE (MSE), and physical residual (Phys). DNC (Did Not Complete) denotes trajectories that encountered numerical divergence or memory exhaustion (NaN/OOM). Best results are highlighted in bold, and second-best are underlined (excluding the unguided Vanilla baseline). Values marked with † fall below the discrete ground-truth stencil precision limit (stencil overfitting/waterbed effect). See Appendix I for an analysis of the Waterbed Effect.
<table><tr><td>Method</td><td>SR↑</td><td>Heat Spd↓</td><td>MSE↓</td><td>Phys↓</td><td>SR↑</td><td>Spd↓</td><td>RD MSE↓</td><td>Phys↓</td><td>SR↑</td><td>Spd↓</td><td>Burgers MSE↓</td><td>Phys↓</td><td>SR↑</td><td>Navier-Stokes Spd↓</td><td>MSE↓</td><td>Phys↓</td></tr><tr><td colspan="10">Panel A: Partial Physics Guidance (Boundary &amp; Initial Conditions)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla FFM</td><td>100</td><td>0.19</td><td>0.08</td><td>0.26</td><td>100</td><td>0.38</td><td>0.08</td><td>0.20</td><td>100</td><td>0.42</td><td>0.21</td><td>0.10</td><td>100</td><td>0.69</td><td>0.54</td><td></td><td>0.29</td></tr><tr><td>ProFlow</td><td>100</td><td>8.13</td><td>0.06</td><td>0.19</td><td>100</td><td>25.65</td><td>7.38e+05</td><td>1.16e+03</td><td>100</td><td>21.34</td><td>0.23</td><td></td><td>0.13</td><td>100</td><td>0.85</td><td>0.55</td><td>0.20</td></tr><tr><td>ECI</td><td>100</td><td>8.31</td><td>0.01</td><td>0.01</td><td>100</td><td>24.47</td><td>0.25</td><td>0.05</td><td>100</td><td>21.87</td><td>0.11</td><td></td><td>0.05</td><td>100</td><td>3.56</td><td>0.63</td><td>3.75e-05</td></tr><tr><td>DiffusionPDE</td><td>100</td><td>0.95</td><td>0.01</td><td>0.03</td><td>100</td><td>2.00</td><td>0.02</td><td>0.04</td><td>100</td><td>1.17</td><td>0.15</td><td></td><td>0.06</td><td>100</td><td>1.40</td><td>0.54</td><td>0.12</td></tr><tr><td>D-Flow</td><td>100</td><td>15.58</td><td>0.35</td><td>0.09</td><td>100</td><td>32.07</td><td>1.98e+02</td><td>0.74</td><td>100</td><td>39.62</td><td></td><td>0.31</td><td>0.08</td><td>14.2</td><td>DNC</td><td>DNC</td><td>DNC</td></tr><tr><td>PCFM</td><td>0.2</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>2.15</td><td>0.04</td><td>1.11e-08†</td><td>100</td><td>1.29</td><td></td><td>0.19</td><td>1.19e-08†</td><td>100</td><td>93.97</td><td>0.76</td><td>1.18e-10†</td></tr><tr><td>EGT (Ours)</td><td>100</td><td>0.37</td><td>0.02</td><td>0.05</td><td>100</td><td>1.13</td><td>0.05</td><td>0.07</td><td>100</td><td>0.48</td><td>0.21</td><td></td><td>0.05</td><td>100</td><td>0.71</td><td>0.53</td><td>0.01</td></tr><tr><td>Hybrid EGT (Ours)</td><td>100</td><td>0.40</td><td>0.02</td><td>0.02</td><td>100</td><td>1.24</td><td>0.05</td><td>8.51e-03</td><td>100</td><td>0.52</td><td>0.20</td><td></td><td>0.01</td><td>100</td><td>0.90</td><td>0.55</td><td>1.41e-04</td></tr><tr><td colspan="10">Panel B: Full PDE Guidance (Stress Test)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla FFM</td><td>100</td><td>0.14</td><td>0.08</td><td>6.24e-03</td><td>100</td><td>0.38</td><td>0.08</td><td>3.57e-03</td><td>100</td><td>0.27</td><td></td><td>0.20</td><td>5.13e-03</td><td>100</td><td>0.86</td><td>0.56</td><td>7.52e-03</td></tr><tr><td>ProFlow</td><td>0.4</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>25.05</td><td>5.00e+11</td><td>2.91e+05</td><td>100</td><td>18.68</td><td>0.24</td><td></td><td>6.27e-03</td><td>100</td><td>1.21</td><td>0.55</td><td>5.63e-03</td></tr><tr><td>ECI</td><td>100</td><td>10.28</td><td>0.01</td><td>1.85e-03</td><td>100</td><td>24.34</td><td>0.25</td><td>7.16e-05</td><td>100</td><td>18.68</td><td>0.12</td><td></td><td>1.95e-04</td><td>100</td><td>4.40</td><td>0.61</td><td>1.80e-03</td></tr><tr><td>DiffusionPDE</td><td>100</td><td>0.69</td><td>18.15</td><td>0.99</td><td>100</td><td>1.86</td><td>0.02</td><td>6.58e-04</td><td>100</td><td>1.14</td><td>0.15</td><td></td><td>9.46e-04</td><td>100</td><td>1.79</td><td>0.55</td><td>4.19e-03</td></tr><tr><td>D-Flow</td><td>100</td><td>18.06</td><td>23.67</td><td>0.03</td><td>100</td><td>31.44</td><td>0.17</td><td>6.58e-04</td><td>100</td><td>32.70</td><td>0.35</td><td></td><td>2.91e-03</td><td>1.6</td><td>DNC</td><td>DNC</td><td>DNC</td></tr><tr><td>PCFM</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>76.72</td><td>6.69e+16</td><td>1.34e+05</td><td>100</td><td>67.11</td><td>1.14e+17</td><td></td><td>4.76e+14</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td></tr><tr><td>EGT (Ours)</td><td>100</td><td>0.24</td><td>0.05</td><td>5.12e-03</td><td>100</td><td>0.89</td><td>0.05</td><td>1.18e-03</td><td>100</td><td>0.46</td><td>0.19</td><td></td><td>1.92e-03</td><td>100</td><td>0.94</td><td>0.55</td><td>1.63e-03</td></tr><tr><td>Hybrid EGT (Ours)</td><td>100</td><td>0.24</td><td>0.05</td><td>3.97e-03</td><td>100</td><td>0.93</td><td>0.05</td><td>1.95e-05</td><td>100</td><td>0.54</td><td>0.19</td><td></td><td>1.49e-04</td><td>100</td><td>1.02</td><td>0.55</td><td>2.89e-05</td></tr></table>

Partial Physics Guidance. Under relaxed boundary constraints, geometric projection methods (e.g., ECI, PCFM) achieve near-zero physical errors on specific equations (RD, Burgers). However, their stability and efficiency are inconsistent across domains: PCFM drops to a 0.2% SR on the Heat equation and exhibits substantial latency on 2D Navier-Stokes (∼ 94s per sample), while concurrently degrading the Sample MSE to 0.76 (compared to 0.54 for the Vanilla model). Similarly, terminal-state optimization via D-Flow struggles with high dimensionality, achieving only a 14.2% SR on Navier-Stokes. While other methods like DiffusionPDE and ProFlow maintain stability, they occasionally degrade the baseline distributional fidelity (e.g., DiffusionPDE increasing the Sample MSE to 0.74 on Navier-Stokes). In contrast, our continuous guidance maintains a 100% Success Rate across all environments, completing the 2D Navier-Stokes generation in just 0.71s while delivering competitive physical and visual fidelity.

Full PDE Guidance (The Stress Test). Constraining generation to satisfy the complete spatiotemporal PDE presents a challenging regime for guidance methods. Proximal formulations like ProFlow show sensitivity in this regime, yielding a 0.4% SR on the Heat equation and noticeable sample degradation on Reaction-Diffusion (Sample MSE reaching $5 . 0 \times 1 0 ^ { 1 1 } )$ . Gradient-based optimization approaches (DiffusionPDE, D-Flow) technically avoid crashing on 1D grids but severely distort the data manifold, causing the Sample MSE on the Heat equation to spike to 18.15 and 23.67, respectively. For exact projection methods like PCFM, explicitly materializing the Jacobian for full-domain constraints results in Out-of-Memory errors or immediate numerical divergence (0% SR) on both Heat and Navier-Stokes systems. More broadly, comparing operating regimes across Tables 1 and 2 reveals a clear structural trade-off: explicit projection methods (ECI, PCFM) excel in low-dimensional, projection-friendly regimes (e.g., 1D Heat or Burgers Sample MSE), but incur heavy computational costs (4.40s–18.68s for ECI) or numerical divergence when constraints become dense and non-linear. Conversely, our EGT framework, stabilized by Gradient Norm Clipping and dynamic scheduling, reliably scales to the Full-PDE regime while preserving sharp shockfronts and vortex structures without high-frequency artifacts or blurring, as visually confirmed on 1D Burgers (Figure 2) and 2D Navier–Stokes (Figure 3). EGT maintains a 100% Success Rate globally with strictly sub-second inference speeds. On the chaotic 2D Navier-Stokes system, it safely navigates the complex physical manifold in just 0.94s, reducing the physical residual to $1 . 6 3 \times 1 0 ^ { - 3 }$ (over 4× lower than the Vanilla unguided model) while preserving the baseline distributional fidelity (Sample MSE 0.55).

Table 2: Quantitative comparison across PDE datasets using the Smooth Kernel prior. $N o t e ^ { \dagger } .$ Evaluation on the 2D Navier-Stokes equations is omitted as the smooth kernel base distribution is currently not formulated for this specific environment. Values marked with <sup>†</sup> fall below the discrete ground-truth stencil precision limit (stencil overfitting/waterbed effect); see Appendix I. Best results are highlighted in bold, and second-best are underlined (excluding the unguided Vanilla).
<table><tr><td rowspan="2">Method</td><td colspan="4">Heat</td><td rowspan="2"></td><td colspan="2">RD</td><td rowspan="2"></td><td colspan="4">Burgers</td></tr><tr><td>SR↑</td><td> $\mathrm { S p d } \downarrow$ </td><td>MSE↓</td><td>SR↑</td><td>Spd↓</td><td>MSE↓</td><td> ${ \mathrm { P h y s } } \downarrow$ </td><td>SR↑  $\mathrm { S p d } \downarrow$ </td><td>MSE↓</td><td></td></tr><tr><td colspan="10">Panel A: Partial Physics Guidance (Boundary &amp; Initial Conditions)</td><td></td><td></td><td></td></tr><tr><td>Vanilla FFM</td><td>100</td><td>0.19</td><td>0.26</td><td>0.43</td><td>100</td><td>0.38</td><td>0.09</td><td>0.22</td><td>100</td><td>0.41</td><td>0.16</td><td>0.13</td></tr><tr><td>ProFlow</td><td>100</td><td>8.16</td><td>0.05</td><td>0.18</td><td>100</td><td>25.60</td><td>0.19</td><td>0.24</td><td>100</td><td>21.41</td><td>0.16</td><td>0.08</td></tr><tr><td>ECI</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>24.43</td><td>0.36</td><td>0.09</td><td>100</td><td>21.94</td><td>0.14</td><td>0.07</td></tr><tr><td>DiffusionPDE</td><td>100</td><td>0.95</td><td>0.04</td><td>0.06</td><td>100</td><td>1.98</td><td>9.37e-03</td><td>0.03</td><td>100</td><td>1.16</td><td>0.13</td><td>0.05</td></tr><tr><td>D-Flow</td><td>100</td><td>15.64</td><td>24.89</td><td>0.97</td><td>100</td><td>31.43</td><td>0.09</td><td>0.03</td><td>100</td><td>39.29</td><td>10.58</td><td>0.37</td></tr><tr><td>PCFM</td><td>1.1</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>2.11</td><td>0.09</td><td>1.11e-08†</td><td>100</td><td>1.29</td><td>0.22</td><td>1.23e-08†</td></tr><tr><td>EGT (Ours)</td><td>100</td><td>0.37</td><td>0.05</td><td>0.05</td><td>100</td><td>1.09</td><td>0.01</td><td>0.03</td><td>100</td><td>0.48</td><td>0.14</td><td>0.02</td></tr><tr><td>Hybrid EGT (Ours)</td><td>100</td><td>0.43</td><td>0.05</td><td>0.02</td><td>100</td><td>1.25</td><td>9.96e-03</td><td>8.48e-03</td><td>100</td><td>0.52</td><td>0.14</td><td>4.58e-03</td></tr><tr><td colspan="10">Panel B: Full PDE Guidance (Stress Test)</td><td></td><td></td><td></td></tr><tr><td>Vanilla FFM</td><td>100</td><td>0.20</td><td>0.25</td><td>0.17</td><td>100</td><td>0.37</td><td>0.09</td><td>3.63e-03</td><td>100</td><td>0.38</td><td>0.16</td><td>5.76e-03</td></tr><tr><td>ProFlow</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>25.10</td><td>0.19</td><td>4.04e-03</td><td>100</td><td>17.42</td><td>0.16</td><td>4.49e-03</td></tr><tr><td>ECI</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>24.40</td><td>0.36</td><td>5.24e-04</td><td>100</td><td>17.08</td><td>0.14</td><td>3.08e-04</td></tr><tr><td>DiffusionPDE</td><td>100</td><td>1.04</td><td>0.04</td><td>0.08</td><td>100</td><td>1.86</td><td>9.38e-03</td><td>2.35e-04</td><td>100</td><td>1.74</td><td>0.12</td><td>1.46e-03</td></tr><tr><td>D-Flow</td><td>100</td><td>16.79</td><td>17.62</td><td>0.66</td><td>100</td><td>31.31</td><td>1.53</td><td>1.60e-03</td><td>100</td><td>28.46</td><td>14.45</td><td>0.05</td></tr><tr><td>PCFM</td><td>0</td><td>DNC</td><td>DNC</td><td>DNC</td><td>100</td><td>77.71</td><td>2.14e+17</td><td>5.91e+04</td><td>100</td><td>50.87</td><td>2.56e+16</td><td>5.74e+13</td></tr><tr><td>EGT (Ours)</td><td>100</td><td>0.52</td><td>0.04</td><td>7.19e-03</td><td>100</td><td>0.89</td><td>9.80e-03</td><td>3.60e-04</td><td>100</td><td>0.95</td><td>0.13</td><td>1.46e-03</td></tr><tr><td>Hybrid EGT (Ours)</td><td>100</td><td>0.55</td><td>0.04</td><td>4.50e-03</td><td>100</td><td>0.95</td><td>9.79e-03</td><td>8.00e-05</td><td>100</td><td>1.06</td><td>0.13</td><td>1.85e-04</td></tr></table>

The Smooth Kernel Prior: A Test of Robustness. To evaluate model stability against prior shifts, we replace standard Gaussian noise with a spatially correlated Smooth Kernel prior (Table 2). Learning the transport map from correlated noise is empirically harder for the base network, leading to less accurate unguided trajectories. Under this more challenging baseline landscape, projection- and proximal-based methods show increased sensitivity: PCFM, ECI, and ProFlow do not complete (0% Success Rate) on the Full-PDE Heat equation. Furthermore, discrete corrections can compound early-trajectory inaccuracies, resulting in degraded sample quality (Sample MSE reaching $2 . 1 \dot { \times } 1 0 ^ { 1 7 }$ for PCFM on Reaction-Diffusion). Among the other compared guidance techniques, only DiffusionPDE maintains consistent stability across these 1D tasks, whereas D-Flow suffers from heavily degraded sample quality (Sample MSE spiking to 17.62 on the Heat equation). Our continuous framework demonstrates robust corrective power under these adverse conditions. In the Heat Full-PDE benchmark, Hybrid EGT effectively guides the less accurate baseline trajectory, reducing the physical residual by over an order of magnitude (from 0.17 down to 4.50e-3) and lowering the Sample MSE to 0.04. This confirms that continuous energy guidance provides substantial resilience against both complex physical constraints and variations in the generative prior.

## 4.1 Ablation Studies

To isolate the operational dynamics of our framework, we conduct targeted ablation studies. Figure 4 illustrates the sensitivity of Energy-Guided Transport (EGT) to the number of ODE integration steps (N) on the highly non-linear Full-PDE Burgers’ equation, evaluated across 256 samples for $N \in \{ 1 0 , \dot { 2 } 5 , 5 0 , 1 0 0 , 2 5 0 , 5 0 0 \}$ . Notably, the analysis reveals that EGT achieves its optimal stabilityaccuracy trade-off at a significantly lower discretization budget $( \mathrm { e . g . , } N \approx 5 0 )$ . For the primary evaluations in Section 4, we deliberately adopted a higher, suboptimal integration budget (i.e., N = 200) to maintain a standardized evaluation against projection-based baselines (e.g., PCFM), which generally require finer temporal discretizations for numerical stability. This finding underscores that, in practical deployments, our continuous guidance can be substantially accelerated while concurrently improving distributional fidelity (Sample MSE).

![](images/dcf7ff9b3781bc4c5ccd34f2a43d896112cb1162deaae623be757739d2a724f8.jpg)  
Figure 4: Impact of the number of ODE integration steps (N) on the 1D Burgers’ equation (Full-PDE guidance). The dual-axis plot tracks Physical Residual MAE (red) and Sample MSE (blue). EGT stabilizes at low step counts $( N \approx 5 0 )$ .

Accuracy–Stability–Speed Pareto Analysis. To further verify whether EGT improves the overall operational trade-off rather than representing a single static operating point, we perform a comprehensive Pareto analysis across guidance strengths $( \gamma )$ and scheduling profiles on the 1D Burgers and 1D Heat systems (detailed in Appendix K). The results confirm that EGT lies on a highly favorable accuracy–stability–speed frontier: it strictly dominates Vanilla FFM and DiffusionPDE by simultaneously improving physical accuracy and distributional stability at ∼ 0.23–0.26 s/sample. While ECI achieves lower absolute residuals in projection-friendly regimes, it incurs a 40–80× higher computational cost. Importantly, EGT occupies a compact region in the runtime dimension, allowing practitioners to freely navigate the trade-off between physical residual and Sample MSE by tuning γ or the scheduler at an essentially constant, sub-second inference time.

Isolating Continuous Transport vs. Terminal Refinement. To verify that the performance of Hybrid EGT stems from our continuous transport rather than the terminal optimization itself, we apply the exact same post-hoc refinement procedure to the baseline trajectories on 1D Burgers and 2D Navier–Stokes (Table 16 in Appendix K). On 2D Navier–Stokes, refining unconditional FFM samples only reduces the physical MAE from 7.50e-3 to 3.96e-3, whereas Pure EGT already reaches 1.60e-3 prior to any refinement and drops to 2.90e-5 when refined, confirming that continuous guidance steers trajectories into a markedly superior physical basin of attraction.

## 5 Conclusion

In this work, we introduced Energy-Guided Transport (EGT), a zero-shot inference framework for physics-informed Functional Flow Matching. By formulating physical constraints as a smooth poten tial energy landscape, our approach maintains the continuous-time probability flow without requiring discrete geometric interruptions. Our evaluations demonstrate that while explicit algebraic projections can face numerical instability and computational overhead in non-linear domains, continuous dissipative guidance provides a scalable and stable alternative. We achieved up to a 130× speedup over explicit projection techniques and a consistent 100% success rate across chaotic environments like 2D Navier-Stokes, achieving a highly favorable balance between physical consistency, computational efficiency, and distributional fidelity. Although evaluating the guidance field introduces a marginal per-step overhead compared to the vanilla unguided model, the framework’s strict $\mathcal { O } ( N )$ scalability ensures its applicability to arbitrary continuous domains.

Limitations and Future Work. While EGT provides a scalable alternative to explicit projections, several limitations define its operational scope. First, as a continuous dissipative mechanism rather than a hard algebraic constraint, EGT does not guarantee exact $E ( u _ { 1 } ) = 0$ convergence without optional terminal refinement, and relies on the pretrained FFM to provide a structurally reasonable transport prior. Second, excessively large guidance scales (γ) increase ODE stiffness $( \mathrm { e } . \mathrm { g } . , \gamma > 3 0 $ on 2D Navier–Stokes). Third, because our current energy functional minimizes pointwise $L ^ { 2 }$ PDE residuals, it does not explicitly constrain global macroscopic statistics such as shock gradient distributions or energy spectra (analyzed in Appendix J.6). Since EGT natively supports any differentiable objective and interacts with the domain geometry solely via $h ( u )$ , future work will incorporate spectral and conservation-law penalties into $E ( u )$ and leverage geometry-aware operators (e.g., GNO, SFNO) to scale EGT to unstructured 3D turbulence.

## References

[1] Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial differential equations using deep learning. Proceedings ofthe National Academy ofSciences, 2018.

[2] Justin Sirignano and Konstantinos Spiliopoulos. DGM: A deep learning algorithm for solving partial differential equations. Journal of Computational Physics, 2018.

[3] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, 2020.

[4] Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021.

[5] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[6] Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. Journal of Machine Learning Research, 2025.

[7] George Em Karniadakis, Ioannis G Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 2021.

[8] Dmitrii Kochkov, Jamie A Smith, Ayya Alieva, Qing Wang, Michael P Brenner, and Stephan Hoyer. Machine learning–accelerated computational fluid dynamics. Proceedings of the National Academy ofSciences, 2021.

[9] Johannes Brandstetter, Daniel E Worrall, and Max Welling. Message passing neural pde solvers. In International Conference on Learning Representations, 2022.

[10] Steven L Brunton, Bernd R Noack, and Petros Koumoutsakos. Machine learning for fluid mechanics. Annual Review ofFluid Mechanics, 2020.

[11] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate medium-range global weather forecasting with 3D neural networks. Nature, 2023.

[12] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. Science, 2023.

[13] Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, et al. Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214, 2022.

[14] C Jiang, R Wang, et al. Physics-informed machine learning: case studies for weather and climate modelling. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 2021.

[15] Rui Wang, Karthik Kashinath, Mustafa Mustafa, Adrian Albert, and Rose Yu. Towards physicsinformed deep learning for turbulent flow prediction. In Conference on Knowledge Discovery and Data Mining, 2020.

[16] Shengze Cai, Zhiping Mao, Zhicheng Wang, Minglang Yin, and George Em Karniadakis. Physics-informed neural networks (pinns) for fluid mechanics: A review. Acta Mechanica Sinica, 2021.

[17] Derek Hansen, Danielle C. Maddix, Shima Alizadeh, Gaurav Gupta, and Michael W. Mahoney. Learning physical models that can respect conservation laws. In International Conference on Machine Learning, 2023.

[18] S Chandra Mouli, Danielle C Maddix, Shima Alizadeh, Gaurav Gupta, Andrew Stuart, Michael W Mahoney, and Yuyang Wang. Using uncertainty quantification to characterize and improve out-of-domain learning for PDEs. In International Conference on Machine Learning, 2024.

[19] Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger Käppeli, Roberto Molinaro, Em-´ manuel De Bezenac, and Siddhartha Mishra. Poseidon: Efficient foundation models for PDEs. In Advances in Neural Information Processing Systems, 2024.

[20] Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, and Christian Etmann. Conditional image generation with score-based diffusion models. arXiv preprint arXiv:2111.13606, 2021.

[21] Yang Song, Conor Durkan, Iain Murray, and Stefano Ermon. Maximum likelihood training of score-based diffusion models. In Advances in Neural Information Processing Systems, 2021.

[22] Ruizhi Deng, Marcus A Brubaker, Greg Mori, and Andreas Lehrmann. Continuous latent process flows. In Advances in Neural Information Processing Systems, 2021.

[23] Utkarsh Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gomez-Bombarelli, and Christopher Vincent Rackauckas. Physics-constrained flow matching: Sampling generative models with hard constraints. In Advances in Neural Information Processing Systems, 2025.

[24] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

[25] Gabriel Peyré and Marco Cuturi. Computational optimal transport: With applications to data science. Now Foundations and Trends, 2019.

[26] Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024.

[27] Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, and Ricky T. Q. Chen. Multisample flow matching: Straightening flows with minibatch couplings. In International Conference on Machine Learning, 2023.

[28] Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In Advances in Neural Information Processing Systems, 2018.

[29] Ricky T. Q. Chen and Yaron Lipman. Flow matching on general geometries. In International Conference on Learning Representations, 2024.

[30] Emile Mathieu and Maximilian Nickel. Riemannian continuous normalizing flows. In Advances in Neural Information Processing Systems, 2020.

[31] Mevlana C Gemici, Danilo Rezende, and Shakir Mohamed. Normalizing flows on riemannian manifolds. arXiv preprint arXiv:1611.02304, 2016.

[32] Valentin De Bortoli, Emile Mathieu, Michael J. Hutchinson, James Thornton, Yee Whye Teh, and Arnaud Doucet. Riemannian score-based generative modelling. In Advances in Neural Information Processing Systems, 2022.

[33] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021.

[34] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. Nature Machine Intelligence, 2021.

[35] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 2023.

[36] Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM/IMS Journal ofData Science, 2024.

[37] Jae Hyun Lim, Nikola B Kovachki, Ricardo Baptista, Christopher Beckham, Kamyar Azizzadenesheli, Jean Kossaifi, Vikram Voleti, Jiaming Song, Karsten Kreis, Jan Kautz, et al. Score-based diffusion models in function space. Journal ofMachine Learning Research, 2025.

[38] Gavin Kerrigan, Giosue Migliorini, and Padhraic Smyth. Functional flow matching. In International Conference on Artificial Intelligence and Statistics, 2024.

[39] Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 2019.

[40] Maziar Raissi, Alireza Yazdani, and George Em Karniadakis. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science, 2020.

[41] Xiaowei Jin, Shengze Cai, Hui Li, and George Em Karniadakis. NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible navier-stokes equations. Journal ofComputational Physics, 2021.

[42] Aditi Krishnapriyan, Amir Gholami, Shandian Zhe, Robert Kirby, and Michael W Mahoney. Characterizing possible failure modes in physics-informed neural networks. In Advances in Neural Information Processing Systems, 2021.

[43] Sifan Wang, Yujun Teng, and Paris Perdikaris. Understanding and mitigating gradient flow pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing, 2021.

[44] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021.

[45] Arpit Bansal, Hong-Min Chu, Avi Schwarzschild, Soumyadip Sengupta, Micah Goldblum, Jonas Geiping, and Tom Goldstein. Universal guidance for diffusion models. In International Conference on Learning Representations, 2024.

[46] Jiahe Huang, Guandao Yang, Zichen Wang, and Jeong Joon Park. Diffusionpde: Generative PDE-solving under partial observation. In Advances in Neural Information Processing Systems, 2024.

[47] Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In International Conference on Machine Learning, 2024.

[48] Chaoran Cheng, Boran Han, Danielle C Maddix, Abdul Fatir Ansari, Andrew Stuart, Michael W Mahoney, and Bernie Wang. Gradient-free generation for hard-constrained systems. In International Conference on Learning Representations, 2025.

[49] Zichao Yu, Ming Li, Wenyi Zhang, Difan Zou, and Weiguo Gao. Proflow: Zero-shot physicsconsistent sampling via proximal flow guidance. arXiv preprint arXiv:2601.20227, 2026.

[50] Lawrence C Evans. Partial Differential Equations. American Mathematical Society, 2022.

[51] Samuel Greydanus, Misko Dzamba, and Jason Yosinski. Hamiltonian neural networks. In Advances in Neural Information Processing Systems, 2019.

[52] Jack Richter-Powell, Yaron Lipman, and Ricky TQ Chen. Neural conservation laws: A divergence-free perspective. In Advances in Neural Information Processing Systems, 2022.

[53] Philipp Holl, Nils Thuerey, and Vladlen Koltun. Learning to control pdes with differentiable physics. In International Conference on Learning Representations, 2020.

## A Proofs of Energy Dissipation and Error Bounds

In this Appendix, we provide the formal proofs for the Dissipation Proposition and its associated corollaries presented in Section 3.3. The derivations are formulated within the deterministic Flow Matching framework, utilizing a Lagrangian (pathwise) perspective to track the continuous probability mass transport in infinite-dimensional spaces.

## A.1 Proof of Proposition 1: Energy Dissipation (Ideal Case)

Let H be the infinite-dimensional Hilbert space representing the state space of the generated physical fields. Under the Functional Flow Matching (FFM) paradigm, the continuous temporal evolution of a generated state is governed by a deterministic ordinary differential equation (ODE) $\dot { u } _ { t } = \bar { v } _ { t } ( u _ { t } )$ . Let $\Phi _ { t } : \mathcal { H } \to \mathcal { H }$ denote the corresponding flow map, such that $u _ { t } = \Phi _ { t } ( u _ { 0 } )$ with the initial state drawn from the base prior $u _ { 0 } \sim p _ { 0 }$

In our continuous guidance framework, the augmented velocity field is defined as:

$$
\bar { v } _ { t } ( u ) = v _ { \theta } ( u , t ) - \gamma ( t ) \nabla _ { u } E ( u )\tag{6}
$$

where $\nabla _ { u } E ( u )$ denotes the Fréchet derivative (gradient) of the physical energy functional with respect to the field u.

We assume that the neural vector field $v _ { \theta }$ and the energy gradient $\nabla _ { u } E ( u )$ satisfy mild local Lipschitz conditions such that the ODE admits a unique global solution $u _ { t } \in \mathcal { H }$ almost everywhere. Furthermore, we assume sufficient regularity to permit the exchange of expectations and time derivatives via the Dominated Convergence Theorem.

We define the expected physical energy at time t over the marginal density $p _ { t } ( u )$ . Using the Law of the Unconscious Statistician, we express this expectation by pulling it back to the base prior $p _ { 0 }$ via the flow map:

$$
\mathbb { E } _ { p _ { t } } [ E ( u ) ] = \mathbb { E } _ { p _ { 0 } } [ E ( \Phi _ { t } ( u _ { 0 } ) ) ]\tag{7}
$$

Applying the functional chain rule in $\mathcal { H } ,$ , the derivative of the expected energy with respect to the generative time t is:

$$
\frac { d } { d t } \mathbb { E } _ { p _ { 0 } } [ E ( \Phi _ { t } ( u _ { 0 } ) ) ] = \mathbb { E } _ { p _ { 0 } } \left[ \left. \nabla _ { u } E ( \Phi _ { t } ( u _ { 0 } ) ) , \frac { d } { d t } \Phi _ { t } ( u _ { 0 } ) \right. _ { \mathcal { H } } \right]\tag{8}
$$

Substituting the ODE $\begin{array} { r } { \frac { d } { d t } \Phi _ { t } ( u _ { 0 } ) = \bar { v } _ { t } ( \Phi _ { t } ( u _ { 0 } ) ) } \end{array}$ and pushing the expectation forward back to the marginal distribution $p _ { t } ,$ we obtain the instantaneous variation of the expected energy:

$$
\frac { d } { d t } \mathbb { E } _ { p _ { t } } [ E ( u ) ] = \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , \bar { v } _ { t } ( u ) \rangle _ { \mathcal { H } } \right]\tag{9}
$$

Substituting the explicit formulation of our augmented vector field into the inner product:

$$
\frac { d } { d t } \mathbb { E } _ { p _ { t } } [ E ( u ) ] = \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , v _ { \theta } ( u , t ) - \gamma ( t ) \nabla _ { u } E ( u ) \rangle _ { \mathcal { H } } \right]\tag{10}
$$

$$
= \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , v _ { \theta } ( u , t ) \rangle _ { \mathcal { H } } \right] - \gamma ( t ) \mathbb { E } _ { p _ { t } } \left[ \| \nabla _ { u } E ( u ) \| _ { \mathcal { H } } ^ { 2 } \right]\tag{11}
$$

Let the first term be defined as $\begin{array} { r } { \frac { d \mathbb { E } _ { \mathrm { { b a s e } } } } { d t } \triangleq \mathbb { E } _ { p _ { t } } [ \langle \nabla _ { u } E ( u ) , v _ { \theta } ( u , t ) \rangle _ { \mathcal { H } } ] } \end{array}$ , representing the natural energy variation induced by the unguided model. Since $\gamma ( t ) \geq 0$ and the expected squared norm is strictly non-negative, the second term acts as a strictly dissipative force:

$$
\frac { d \mathbb { E } _ { \mathrm { t o t a l } } } { d t } \leq \frac { d \mathbb { E } _ { \mathrm { b a s e } } } { d t }\tag{12}
$$

This concludes the proof.

## A.2 Proof of Corollary 1: Error-Bounded Dissipation

In practice, the neural vector field $v _ { \theta } ( u , t )$ is an approximation of the ideal target field $v ^ { * } ( u , t )$ . We define the approximation error as $\delta ( u , t ) \dot { = } v _ { \theta } ( u , \bar { t } ) - v ^ { * } ( u , t )$ . In the context of Flow Matching, the

network is trained to minimize the expected mean squared error; thus, we bound the error in $L ^ { 2 } ( p _ { t } )$ norm: $\epsilon _ { t } ^ { 2 } \triangleq \mathbb { E } _ { p _ { t } } [ \| \delta ( u , t ) \| _ { \mathcal { H } } ^ { 2 } ]$

Substituting $v _ { \theta } ( u , t ) = v ^ { * } ( u , t ) + \delta ( u , t )$ into the expectation of the base variation defined in Proposition 1:

$$
\mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , v _ { \theta } ( u , t ) \rangle _ { \mathcal { H } } \right] = \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , v ^ { * } ( u , t ) \rangle _ { \mathcal { H } } \right] + \mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , \delta ( u , t ) \rangle _ { \mathcal { H } } \right]\tag{13}
$$

We now bound the spurious error term by applying the generalized Cauchy-Schwarz inequality for expectations:

$$
\mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , \delta ( u , t ) \rangle _ { \mathcal { H } } \right] \leq \sqrt { \mathbb { E } _ { p _ { t } } \left[ \| \nabla _ { u } E ( u ) \| _ { \mathcal { H } } ^ { 2 } \right] } \sqrt { \mathbb { E } _ { p _ { t } } \left[ \| \delta ( u , t ) \| _ { \mathcal { H } } ^ { 2 } \right] }\tag{14}
$$

Defining the expected gradient energy landscape as $V ( t ) \triangleq \mathbb { E } _ { p _ { t } } [ \| \nabla _ { u } E ( u ) \| _ { \mathcal { H } } ^ { 2 } ]$ , we obtain:

$$
\mathbb { E } _ { p _ { t } } \left[ \langle \nabla _ { u } E ( u ) , \delta ( u , t ) \rangle _ { \mathcal { H } } \right] \leq \sqrt { V ( t ) } \epsilon _ { t }\tag{15}
$$

Substituting this bound back into the full dissipation equation establishes Corollary 1:

$$
\frac { d \mathbb { E } _ { \mathrm { { t o t a l } } } } { d t } \leq \frac { d \mathbb { E } _ { v ^ { * } } } { d t } - \gamma ( t ) V ( t ) + \sqrt { V ( t ) } \epsilon _ { t }\tag{16}
$$

## A.3 Proof of Corollary 2: Global Energy Descent

To evaluate the macroscopic effect of the continuous guidance, we integrate the instantaneous rate of change derived in Corollary 1 over the entire generation process, from $t = 0 \mathrm { t o } t = 1$

$$
\int _ { 0 } ^ { 1 } \frac { d } { d t } \mathbb { E } _ { p _ { t } } [ E ( u ) ] d t \leq \int _ { 0 } ^ { 1 } \frac { d \mathbb { E } _ { v ^ { * } } } { d t } d t - \int _ { 0 } ^ { 1 } \gamma ( t ) V ( t ) d t + \int _ { 0 } ^ { 1 } \sqrt { V ( t ) } \epsilon _ { t } d t\tag{17}
$$

By the Fundamental Theorem of Calculus, the left-hand side evaluates exactly to the total change in expected physical energy:

$$
\mathbb { E } _ { p _ { 1 } } [ E ( u _ { 1 } ) ] - \mathbb { E } _ { p _ { 0 } } [ E ( u _ { 0 } ) ]\tag{18}
$$

Defining the cumulative natural energy variation under the ideal transport field as $\Delta E _ { \mathrm { b a s e } } ~ =$ $\begin{array} { r } { \int _ { 0 } ^ { 1 } \frac { d \mathbb { E } _ { v ^ { * } } } { d t } \bar { d t } . } \end{array}$ , and the exact cumulative neural approximation error as $\begin{array} { r } { \mathcal { E } _ { \mathrm { e r r } } = \int _ { 0 } ^ { 1 } \sqrt { V ( t ) } \epsilon _ { t } d t } \end{array}$ , we obtain Corollary 2:

$$
\mathbb { E } _ { p _ { 1 } } [ E ( u _ { 1 } ) ] - \mathbb { E } _ { p _ { 0 } } [ E ( u _ { 0 } ) ] \le \Delta E _ { \mathrm { b a s e } } - \int _ { 0 } ^ { 1 } \gamma ( t ) V ( t ) d t + \mathcal { E } _ { \mathrm { e r r } }\tag{19}
$$

This confirms that the cumulative dissipative force must be scaled via the scheduling parameter $\gamma ( t )$ to robustly mitigate the cumulative model error ${ \mathcal E } _ { \mathrm { e r r } } ,$ ensuring a net reduction in the overall PDE violation. Note that if the neural approximation error is uniformly bounded by $\operatorname { s u p } _ { t } \epsilon _ { t } \leq \epsilon .$ , the cumulative error scales as $\mathcal { O } ( \epsilon )$ . while the physical energy landscape $E ( u )$ may be highly non-convex for non-linear PDEs (as discussed in Appendix B), the local dissipativity guaranteed by the inner product $\langle \nabla _ { u } E , - \nabla _ { u } E \rangle \le 0$ holds universally. Unlike explicit algebraic projections that diverge in non-convex topologies, our continuous gradient injection safely ensures energy descent along the probability path without requiring global convexity.

## B Mathematical Taxonomy of Constraints

Integrating physical knowledge into zero-shot generative models requires defining a set of constraint functions $\bar { h ( u ) } = 0$ . As referenced in Section 3.1, we categorize these constraints based on their spatial scale, linearity, and the topological complexity of their resulting energy landscape $E ( u ) =$ $\textstyle { \frac { 1 } { 2 } } \| h ( u ) \| ^ { 2 }$ . Table 3 summarizes the mathematical properties of the constraints evaluated in this work.

1. Initial and Boundary Conditions (IC/BC): These constraints define the admissibility and uniqueness of the solution. While essential, they only constrain the field at $t = 0$ or on the spatial boundary ∂Ω, leaving the internal spatio-temporal evolution entirely to the learned prior of the generative model. Mathematically, they are typically formulated as linear constraints $( \mathrm { e . g . }$ , Dirichlet or standard Neumann conditions), though non-linear boundary physics can occur.

2. Macroscopic Constraints (Conservation Laws): Often expressed as global integrals over the domain, such as $\begin{array} { r } { \int _ { \Omega } u ( \mathbf { x } , t ) d \mathbf { x } = C } \end{array}$ , these constraints enforce the conservation of macroscopic invariants (e.g., total mass or momentum). Because the integral operator is a strictly linear functional mapping, its associated energy landscape is strictly convex. This guarantees that explicit algebraic projection methods can find a unique, global minimum efficiently. However, acting globally, they lack the spatial resolution required to correct local structural dynamics.

3. Full PDE Residual: This represents the most rigorous microscopic constraint, requiring the governing differential equation to hold locally at every coordinate $\mathbf { \rho } ( \mathbf { x } , t ) $ . The topological complexity of this constraint is entirely dictated by the nature of the spatial differential operator $\dot { \mathcal { N } } ( u )$ . For linear differential equations (e.g., the Heat equation), the resulting energy functional remains strictly convex. Conversely, for non-linear advective systems $( \mathrm { e . g . }$ , Burgers’ or Navier-Stokes equations), the residual forms a highly non-convex energy landscape. In these non-linear regimes, exact explicit projections become computationally intractable and structurally fragile, necessitating the continuous dissipative guidance proposed in our framework.

<table><tr><td>Constraint Type</td><td>Analytical Form  $h ( u ) = 0$ </td><td>Spatial Scope</td><td>Linearity / Landscape</td><td>Generative Impact</td></tr><tr><td>Initial/Boundary</td><td> $B ( u ) - g ( \mathbf { x } , t ) = 0$ </td><td>∂Ω ∪ {t = 0}</td><td>Typically Linear (Convex)</td><td>Anchors the domain; ensures uniqueness.</td></tr><tr><td>Macroscopic</td><td> $\begin{array} { r } { \int _ { \Omega } u d \mathbf { x } - C = 0 } \end{array}$ </td><td>Global (Ω)</td><td>Strictly Linear (Convex)</td><td>Enforces global invariants; blind to local topology.</td></tr><tr><td>Full PDE</td><td> $\partial _ { t } u + \mathcal { N } ( u ) = 0$ </td><td>Microscopic (∀x, t)</td><td>System-Dependent</td><td>Dictates true local dynamics; implicitly ensures global conservation.</td></tr></table>

Table 3: Theoretical taxonomy of physical constraints in generative modeling. The structural linearity of the physical operator dictates the convexity of the resulting energy landscape. This topological property drastically impacts the computational feasibility of applying exact algebraic corrections during the inference process.

## C Dataset Discretization and Numerical Generation

To ensure a fair and rigorous baseline comparison, all datasets used in this work follow the exact numerical generation protocols established by the Physics-Constrained Flow Matching (PCFM) frame work [23]. The datasets are mathematically identical to those in PCFM, ensuring that performance gains stem strictly from the generative methodology.

1. Heat Equation (1D): We consider the linear diffusion equation $\partial _ { t } u = \alpha \partial _ { x x } u$ . Instead of a numerical approximation, we generate samples using the exact analytical solution:

$$
u ( x , t ) = \sin ( x + \varphi ) e ^ { - \alpha t }\tag{20}
$$

where the diffusion coefficient is $\alpha \sim \mathcal { U } ( 1 . 0 , 5 . 0 )$ and the phase is $\varphi \sim \mathcal { U } ( 0 , \pi )$ . The domain is discretized on a $1 0 0 \times 1 0 0$ spatiotemporal grid $( x \in [ 0 , 2 \pi ] , t \in [ 0 , 1 ] )$ with periodic boundary conditions. This provides a high-fidelity ground truth with zero discretization error.

2. Reaction-Diffusion (1D): This system is governed by the non-linear equation $\partial _ { t } u - \nu \partial _ { x x } u -$ $\rho u ( 1 - u ) = 0$ , with parameters set to $\rho = 0 . 0 1$ and $\nu = 0 . 0 0 5$ . Data is generated via a semi-implicit finite difference scheme on a $1 2 8 \times 1 0 0$ grid over $x \in [ 0 , 1 ]$ . We construct 6,400 solutions by pairing 80 initial conditions with 80 boundary conditions.

3. Inviscid Burgers’ Equation (1D): We solve the inviscid form $\partial _ { t } u + u \partial _ { x } u = 0$ , which is highly prone to shock formation. To handle discontinuities, we employ a Godunov finite volume method. The initial conditions are smoothed step functions $u ( x , 0 ) = \bar { ( 1 + \exp ( ( x - p ) / \epsilon ) ) } ^ { - 1 }$ with steepness $\epsilon = 0 . 0 2$ and the center location sampled uniformly as $p \sim \mathcal { U } ( 0 . 2 , 0 . 8 )$ . The grid is $1 0 1 \times 1 0 1$ over $x , t \in [ 0 , 1 ]$

4. Navier-Stokes Equations (2D): We use the incompressible Navier-Stokes equations in vorticity form: $\partial _ { t } \omega + ( u \cdot \nabla ) \bar { \omega } = \nu \Delta \omega + f$ . We set the viscosity to $\nu = 1 0 ^ { - 3 }$ and use a Crank-Nicolson spectral solver on a $6 4 \times 6 4$ periodic grid, recording 50 temporal snapshots. The dataset comprises 10,000 simulations generated by sampling 100 random initial vorticities and 100 forcing phases.

## D Generative Kernels and Prior Distributions

The choice of the prior distribution $p _ { 0 } ( u )$ is critical in Functional Flow Matching, as it defines the starting point of the probability flow and dictates the initial geometry of the sample paths. While white noise is the standard setting for discrete diffusion models, embedding spatial smoothness directly into the functional prior is fundamental for the stable evaluation of high-order spatial physical operators. In our experiments, we compare two distinct classes of priors:

1. White Noise Prior: This represents the uncorrelated generative setting, where the initial state is sampled as independent and identically distributed (i.i.d.) Gaussian noise across all spatial grid points, effectively mimicking $u _ { 0 } \sim \mathcal { N } ( 0 , \bf { I } )$ . For the 2D Navier-Stokes equations, we exclusively utilize this prior due to the extreme dimensionality of the full spatio-temporal grid $( 6 4 \times 6 4 \times 5 0 )$ , which makes the instantiation and inversion of full dense covariance matrices computationally prohibitive for spatially correlated priors. Visualized in Figure 5a, it presents as high-frequency noise lacking any spatial structure.

2. Smooth Kernel Prior (Gaussian Random Fields): To investigate the impact of spatial correlation on continuous physical guidance, we employ Gaussian Random Fields (GRFs) as functional priors for the 1D equations. Unlike white noise, GRFs ensure that the initial state u<sub>0</sub> possesses a predefined degree of spatial correlation, guaranteeing spatial differentiability right from $t = 0$ . This property provides a well-behaved energy landscape for guiding derivatives $( \partial _ { x } , \partial _ { x x } )$

• Heat Equation: We utilize a Matérn covariance kernel $( \nu = 3 / 2 )$ with a characteristic correlation length $\ell = 0 . 2 .$ , promoting smoother and larger initial spatial features (Figure 5b).

• Reaction-Diffusion and Burgers’ Equations: We utilize a Squared Exponential kernel with a tighter correlation length $\ell = 0 . 0 5$ (Figure 5c), allowing for the representation of sharper initial functional shapes.

Figure 5 provides a qualitative 2D visualization of samples drawn from these functional prior distributions, clearly highlighting how the characteristic correlation length dictates the smoothness and frequency spectrum of the initial generative state.

![](images/18688f9acef5ab8f945ac64eda8ad568d40392f00094e36e1631073c87c0f7c0.jpg)  
b) Matérn Kernel Prior (Heat Eq, = 0.2)

![](images/0162977ef55ffed66ee386a9b755b3a03509c41b919ce542dcd97a24c29f1460.jpg)

![](images/8c098de6a5d83781982ac0fc9c24faa02fa6bc72d893ff06c701ba63e033036e.jpg)  
Figure 5: Qualitative 2D visual comparison of standard samples drawn from different generative functional prior distributions $p _ { 0 } ( u )$ , visualized as spatial maps over the domain $[ 0 , 1 ] \times [ 0 , 1 ]$ . Panel a) shows white noise, characterized by uncorrelated pixels that create local high-frequency content. Panels b) and c) show functional smooth kernel priors (Gaussian Random Fields), highlighting how the characteristic length scale ℓ dictates the typical size and smoothness of initial functional structures.

## E Architecture and Training Details

FNO Backbone. All generative models in this work parameterize the underlying time-dependent vector field $v _ { \theta } ( u _ { t } , t )$ using a Fourier Neural Operator (FNO) backbone. The FNO encoder takes as input the concatenation of the current state $u _ { t } ,$ a sinusoidal positional embedding for the spatial grid, and a Fourier time embedding to capture the non-stationary nature of the probability flow.

To maintain strict consistency with the baseline setups, we tailor the architecture capacity to the dimensionality of the problem:

• 1D Systems (Heat, Reaction-Diffusion, Burgers): The architecture consists of 4 Fourier layers preserving 32 modes across spatial and temporal dimensions, with 64 hidden channels and 256 projection channels.

• 2D Navier-Stokes: Due to the high-dimensional nature of the $6 4 \times 6 4 \times 5 0$ spatio-temporal grid, we utilize a 3D FNO equipped with 2 Fourier layers, preserving 16 modes per dimension, with 32 hidden channels and 256 projection channels.

Training Regimen. We pretrained 7 distinct Functional Flow Matching (FFM) models to comprehensively evaluate our continuous guidance against unconstrained baselines under different prior assumptions (White Noise vs. Smooth GRFs). All models were trained using the Adam optimizer $( \beta _ { 1 } = \mathsf { \bar { 0 } } . 9 , \beta _ { 2 } = 0 . 9 9 9$ , zero weight decay) with an initial learning rate of $3 \times \overline { { 1 0 } } ^ { - 4 }$ . We employed a Reduce-LROnPlateau scheduler with a decay factor of 0.5, a patience of 10 validation steps, and a minimum learning rate of $1 \times 1 0 ^ { - 4 }$

The batch sizes were set to 256 for the 1D environments and 24 for the 2D Navier-Stokes environment. Training was performed on clusters equipped with NVIDIA RTX 8000, A6000 and L40 GPUs. The 1D models converged after 20,000 iterations, while the Navier-Stokes model required 500,000 training steps to adequately capture the complex fluid dynamics. Table 4 summarizes the specific configurations for each of the 7 pretrained models.

Computational Bottleneck of Spatially Correlated Priors. While smooth GRFs were utilized for 1D systems, the 2D Navier-Stokes environment (64 × 64 spatial grid with 50 temporal snapshots) presents a prohibitive computational bottleneck. Instantiating a spatially correlated Gaussian prior requires computing and decomposing a dense covariance matrix of size $\dot { N } \times N$ , where N is the total number of grid points per sample. For Navier-Stokes, $N = 6 4 \times 6 4 \times 5 0 = 2 0 4$ , 800, resulting in a dense covariance matrix with over 4. $1 9 \times 1 0 ^ { 1 0 }$ elements. Even in single precision (FP32), storing this matrix for a single instance would require approximately 167 GB of VRAM, far exceeding the capacity of standard modern GPUs, thereby restricting our evaluation to the uncorrelated White Noise prior for this dataset.

Table 4: Architectural and optimization hyperparameters for the 7 pre-training environments.
<table><tr><td>PDE System</td><td>Prior Type</td><td>Layers</td><td>Modes</td><td>Channels</td><td>Batch Size</td><td>Steps</td></tr><tr><td>Heat 1D</td><td>White / Smooth</td><td>4</td><td>32</td><td>64</td><td>256</td><td>20k</td></tr><tr><td>RD 1D</td><td>White / Smooth</td><td>4</td><td>32</td><td>64</td><td>256</td><td>20k</td></tr><tr><td>Burgers 1D</td><td>White / Smooth</td><td>4</td><td>32</td><td>64</td><td>256</td><td>20k</td></tr><tr><td>Navier-Stokes 2D</td><td>White</td><td>2</td><td>16</td><td>32</td><td>24</td><td>500k</td></tr></table>

## F Baseline Descriptions and Implementation Details

To rigorously evaluate the efficacy of our proposed continuous dissipative guidance, we benchmarked it against five representative state-of-the-art methodologies for physical constraint enforcement and generative guidance. The theoretical and implementation details of each baseline are outlined below.

1. Physics-Constrained Flow Matching (PCFM): PCFM [23] enforces nonlinear physical constraints by applying an explicit algebraic projection at each solver step. The framework relies on a first-order Taylor expansion of the physical constraint functional $h ( \bar { u } )$ . Given the unconstrained terminal state prediction $\hat { u } _ { 1 } .$ , PCFM finds a corrected state u by solving the linearized system $h ( \hat { u } _ { 1 } ) + J ( u - \hat { u } _ { 1 } ) = 0$ , where $J = \nabla h ( { \hat { u } } _ { 1 } )$ is the constraint Jacobian. This maps the sample onto the tangent space of the constraint manifold via the Moore-Penrose pseudo-inverse. While theoretically exact for linear equations, this projection requires computing and applying $J ^ { \dagger }$ , leading to the severe $\mathcal { O } ( N ^ { 2 } )$ memory bottlenecks and computational overheads discussed in Section 4 during our full-PDE stress tests.

2. ProFlow (Proximal Flow Guidance): ProFlow [49] is a recent framework (introduced in Jan 2026) specifically designed to address the computational weight and numerical instability of exact projection-based methods like PCFM. Instead of solving a strict linearized system via matrix inversion, it utilizes K unrolled proximal gradient steps on the predicted terminal state $\hat { u } _ { 1 }$ to softly enforce physics consistency. Most importantly, it then applies a linear interpolation step to project the refined state back onto the probability path, attempting to avoid entirely disrupting the learned prior. Note: At the time of writing, the official source code for ProFlow was not publicly available; therefore, we provided an independent and faithful implementation based strictly on the algorithmic description and hyperparameter settings $( K = 3$ proximal steps) detailed in the original manuscript.

3. DiffusionPDE: DiffusionPDE [46] utilizes the gradient of a physical residual or observation loss to softly guide the generative process. Originally formulated for score-based diffusion models, it relies on obtaining a denoised estimate $\hat { u } _ { 0 }$ and computing the guidance gradient directly on it. In the context of Flow Matching, this baseline is adapted by computing the gradient of the physical energy functional directly on the deterministic prediction of the terminal state $\hat { u } _ { 1 }$ . Unlike our proposed dissipative guidance, which applies a continuous local force inherently coupled with the intermediate state $u _ { t } ,$ , this method relies entirely on the accuracy of the one-step prediction to anchor the physical correction.

4. Extrapolation, Correction, and Interpolation (ECI): ECI [48] is a gradient-free, zero-shot framework designed for hard-constrained generation. It iteratively alternates between three stages during flow sampling: extrapolating the intermediate state to a terminal prediction, applying an exact algebraic correction (e.g., orthogonal or oblique projection) to strictly satisfy the constraint, and interpolating the corrected state back to the flow probability path. While highly effective for simple linear or non-overlapping constraints where closed-form projections exist, its reliance on analytical corrections restricts its extensibility to dense, highly non-linear constraints, such as full-domain PDE residuals.

5. D-Flow: D-Flow [47] utilizes a flow-based generative prior coupled with an inner-loop optimization routine to satisfy physical constraints. Instead of modifying the intermediate sampling states, it attempts to optimize the initial noise $u _ { 0 }$ by backpropagating the constraint violation loss through the entire unrolled ODE trajectory.

## G Inference Implementation Details

To maintain stability across the highly non-convex PDE landscapes discussed in Section 3.5, our guided inference algorithm incorporates specific numerical mechanisms and carefully tuned hyperparameters to establish a fair evaluation against the baselines.

## G.1 Numerical Stabilization Mechanisms

Gradient Clipping The non-linear operators within the full PDE residual can produce gradients of highly variable magnitude, particularly in the presence of unrefined early-stage structures or sharp physical shocks. To prevent these gradients from dominating the neural vector field $v _ { \theta }$ and destabilizing the ODE solver, we employ ${ \\check { L } } ^ { 2 } .$ -norm gradient clipping. For a given clip threshold $c ,$ the gradient $g$ is scaled as $\tilde { g } = g \cdot \operatorname* { m i n } ( 1 , c / \| g \| _ { 2 } )$ . Unlike global gradient normalization, this strategy bounds the maximum corrective force while preserving the natural decay of the gradient magnitude as the sample converges toward the physical manifold.

Transient Guidance Scheduling $\mathbf { A } \mathbf { t } \ t \approx 0 ,$ , the state $u _ { t }$ is highly noisy, making the evaluation of spatial physical operators uninformative. Conversely, as $t  1$ , the formation of sharp physical features can lead to numerical instability in spatial PDE gradients. To address both extremes, we employ a dynamic scheduling function $\gamma ( t )$ that implements a transient guidance strategy. In our main experiments, we utilize a sine schedule $( \gamma ( t ) = \gamma _ { m a x } \sin ( \pi t )$ , as illustrated in Figure 6). This schedule explicitly zeroes out the physics penalty at $t = 0$ to allow the unguided neural prior to establish macroscopic structures, applies the strongest physical correction at $t = 0 . 5$ , and smoothly decays to zero as $t  1$ to prevent gradient explosion and preserve the generative fine-tuning. Empirically, the choice of scheduler significantly influences generative fidelity, particularly for complex PDE dynamics. A comprehensive ablation study comparing different scheduling profiles is provided in Appendix K.

![](images/e0601609fe0abd1ad3c796df5a38754b57bdf04be402568cae5de86501d93545.jpg)  
Figure 6: Comparison of guidance scheduling profiles $\gamma ( t )$ as a function of the probability flow time $t \in [ 0 , 1 ]$ . While traditional ’Late Guidance’ schedules (Linear, Parabolic, Cosine) monotonically increase to apply maximum physical correction at the terminal state, our proposed ’Transient’ Sine schedule peaks at $t = 0 . 5$ and decays to zero at $t = 1$ . This mitigates numerical instabilities caused by evaluating non-linear physical operators on sharp fluid structures during the final stages of the generative process.

Efficient ODE Integration While higher-order ODE solvers (e.g., Runge-Kutta) are commonly used in unguided flow matching, they require multiple evaluations of the vector field per time step. In an energy-guided framework, each evaluation necessitates a forward-backward pass through complex PDE operators to compute the implicit VJP $( \nabla _ { u } E ( u _ { t } ) )$ , making higher-order solvers computationally expensive. By leveraging the inherent smoothness of FFM alongside gradient clipping and transient scheduling, the augmented probability flow is sufficiently stabilized to be integrated using a first-order Euler method. This configuration ensures an $\mathcal { O } ( N )$ computational complexity, maintaining efficient inference times compared to projection-based baselines.

## G.2 Inference Hyperparameters and Baseline Tuning

To ensure a fair zero-shot evaluation, all sampling techniques are initialized with identical noise realizations. The foundational integration parameters are summarized in Table 5. The hyperparameters for PCFM, ECI, and D-Flow follow the configurations established by Utkarsh et al. [23]. ProFlow is configured according to its original manuscript [49], and DiffusionPDE [46] is tuned specifically to maintain numerical stability across the evaluated PDE systems.

Table 5: Shared integration hyperparameters across all physical systems. NFE denotes the Number of Function Evaluations (forward integration steps).
<table><tr><td>Shared Parameter</td><td>Heat (1D)</td><td>RD (1D)</td><td>Burgers (1D)</td><td>Navier-Stokes (2D)</td></tr><tr><td>ODE Solver Scheme</td><td>Forward Euler</td><td>Forward Euler</td><td>Forward Euler</td><td>Forward Euler</td></tr><tr><td>NFE (Steps)</td><td>100</td><td>200</td><td>200</td><td>100</td></tr><tr><td>Base Flow Prior</td><td>White / Smooth</td><td>White / Smooth</td><td>White / Smooth</td><td>White</td></tr></table>

Energy guided transport (Ours): The dissipative energy scale $\gamma$ follows a sine schedule. For 1D systems, we set $\gamma _ { m a x } = 7 5$ , while for the 2D Navier-Stokes equations, we reduce the scale to $\gamma _ { m a x } = 2 5$ to ensure stability. An optional post-hoc refinement phase is applied at t = 1 to further anchor the sample to the physical manifold. By default, this involves 100 gradient descent steps with a learning rate of $\eta = 0 . 1$ . For the “Heat Full $\mathrm { P D E } '$ experiment, we use 50 steps with $\eta = 0 . 0 1$ and for the “Heat Partial Guidance” experiment, we use 200 steps with $\eta = 0 . 1$ . The gradient norm clipping threshold is fixed at $c = 1 . 0$ across all experiments.

PCFM Specifics: We strictly adopt the official configurations from [23]. Specifically, guided interpolation $( \lambda = 1 . 0$ , step size 0.01, 20 refinement steps) is restricted exclusively to the linear Heat equation, whereas non-linear PDEs employ a single Gauss-Newton step per integration step without guided interpolation. To maximize stability, we retain the official implementation’s default Tikhonov regularization parameter $( \epsilon = 1 0 ^ { - 6 } )$ for inverting the Schur complement and execute projections in float64 precision. As noted independently by [49], explicit Jacobian-based projections can still exhibit numerical divergence when scaled to dense, non-linear full-PDE constraints.

Optimization-Based Baselines (DiffusionPDE & D-Flow): These baselines optimize a composite loss function $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { d a t a } } + \lambda _ { \mathrm { p h y s } } \mathcal { L } _ { \mathrm { p h y s } } .$ Following the evaluation protocol in [23], we set $\lambda _ { \mathrm { p h y s } } =$ 0.01. We observe that these methods are highly sensitive to this hyperparameter: larger values can lead to severe manifold degradation, while smaller values fail to adequately enforce the PDE constraint.

• DiffusionPDE: Uses a guidance step size of $\eta = 0 . 0 1$ . We adopt a more conservative step size than the original formulation to improve numerical stability and prevent divergence across the benchmarked PDEs.

• D-Flow: Performs $N _ { i t e r } = 2 0 ~ \mathrm { I }$ -BFGS inner optimization steps with a learning rate of $\eta = 1 . 0 .$

## Proximal and Projection Baselines (ECI & ProFlow):

• ECI: Configured with $n _ { m i x } = 5$ mixing updates per step and a prior resampling frequency of 5 steps, strictly mirroring the benchmark settings in [23].

• ProFlow: Employs $K = 3$ proximal steps with a time-dependent learning rate schedule starting from $\eta _ { 0 } = 0 . 0 1$ following [49]. When encountering numerical instabilities in our reimplementation, we escalated execution to float64 precision and performed a best-effort stabilization search while strictly preserving the original methodology.

## H Computational Complexity and Efficiency Analysis

A fundamental advantage of the proposed Energy-Guided Transport is its computational efficiency and structural scalability to high-dimensional domains. In this section, we formally analyze the time and memory complexity introduced by our physics-informed guidance mechanism, contrasting it with state-of-the-art baselines.

Let N denote the total number of spatial degrees of freedom (e.g., $N = H \times W$ for a 2D physical field).

## H.1 Theoretical Complexity: VJP vs. Explicit Jacobian

The primary computational bottleneck in physics-guided generation is the interaction with the constraint operator $h ( u )$

Memory Complexity: $\mathcal { O } ( N )$ To compute a hard projection, methods like PCFM must explicitly materialize the Jacobian matrix $J = \dot { \nabla } h ( u ) \in \mathbb { R } ^ { \dot { M } \times \check { N } }$ . For a full-PDE constraint on a 2D field of size $6 4 \times 6 4 , N = 4 0 9 6$ , resulting in a dense Jacobian with over 16 million entries per sample. Storing and inverting this matrix imposes an intractable $\mathcal { O } ( N ^ { 2 } )$ memory footprint, severely limiting its application in 3D systems or high-resolution 2D fluids. Conversely, our framework leverages reverse-mode automatic differentiation to compute implicit Vector-Jacobian Products (VJPs). By directly computing the gradient of the scalar energy $( \boldsymbol { \nabla } _ { \boldsymbol { u } } ^ { - } E ( { \boldsymbol { u } } _ { t } ) )$ , the memory requirement is strictly bounded by the storage of the intermediate forward activation tensors. Consequently, the spatial complexity of our physical guidance remains $\mathcal { O } ( N )$ , matching the theoretical minimum required to represent the state itself.

Time Complexity: $\mathcal { O } ( N )$ The computational overhead per ODE integration step in our method consists of evaluating the analytical PDE residual $h ( u _ { t } )$ (forward pass) and computing the VJP (backward pass). Assuming the differential operators are discretized via standard local finite differences, the forward evaluation scales linearly, $\mathcal O ( N )$ . According to the Baur-Strassen theorem of algorithmic differentiation, the time complexity of the reverse-mode backward pass is mathematically guaranteed to be bounded by a small constant factor $c \leq 5$ of the forward pass. Therefore, the total time complexity for the physical guidance step is strictly $\mathcal { O } ( N )$ . This linear scaling ensures that the physical guidance does not become the computational bottleneck: while the base generative backbone (e.g., Fourier Neural Operator) operates at O(N log N) due to Fast Fourier Transforms, our dissipative guidance preserves the overall asymptotic complexity of the framework.

## H.2 Empirical Computational Budget (NFE and Operator Evaluations)

In flow-based and diffusion-based generative models, the practical computational cost is typically measured by the Number of Function Evaluations (NFE) of the neural vector field $v _ { \theta } ,$ combined with the number of physical operator evaluations.

Baseline Bottlenecks Optimization and projection-based baselines inherently multiply the computational budget to satisfy constraints. For instance, D-Flow performs global optimization over the initial noise, requiring a forward and an adjoint-backward pass through the entire ODE for each of its 20 L-BFGS iterations. For a standard 100-step solver, this demands up to 4,000 NFE per generated sample. ECI multiplies its integration cost by repeating its mixing step $n _ { m i x } = 5$ times per ODE step. PCFM and ProFlow compute computationally demanding proximal or projection updates at every intermediate time step (e.g., ProFlow computes $K = 3$ proximal PDE steps per integration step, while PCFM evaluates explicit Jacobians alongside 20 refinement iterations per step).

Energy-Guided Transport Unlike the aforementioned baselines, our standard guided inference requires exactly 1 NFE and 1 VJP evaluation per integration step. The continuous dynamic provided by the transient scheduler smoothly steers the generation without requiring expensive inner-loop optimizations during the flow. For a 100-step Euler integration, this strictly bounds the budget to 100 NFE and 100 PDE gradient evaluations.

Hybrid Energy-Guided Transport When maximum physical adherence is required, our algorithm optionally incorporates a terminal refinement phase at $t = 1$ . Importantly, this refinement minimizes the physical energy strictly via gradient descent on the final tensor without querying the neural network. Therefore, performing R refinement steps (e.g., R = 50 or R = 200) adds 0 NFE to the overall budget. It incurs only R lightweight $\mathcal { O } ( N )$ evaluations of the explicit physical operator. This design strategically uncouples the heavy generative modeling from the physical fine-tuning, resulting in rigorous physical consistency, without compromising the learned data manifold, achieved at a fraction of the computational cost of competing methodologies.

## I The Ground Truth Floor, Stencil Overfitting, and the Waterbed Effect

To rigorously contextualize the physical residuals achieved by the generative models, we explicitly compute the evaluation metrics on the testing split of the Ground Truth (GT) simulated dataset. As shown in Table 6, the GT errors for Initial Conditions, Boundary Conditions, and Global Mass evaluate to exactly zero (up to machine precision limits). This is expected, as these macroscopic constraints are explicitly hard-coded or globally conserved during the numerical simulation process.

Conversely, the interior PDE residual and, notably, the Global Mass error in non-linear systems (such as Burgers and Reaction-Diffusion) evaluate to strictly positive values $( > 0 )$ This occurs because simulating continuous physical laws on a discrete grid inherently introduces numerical truncation errors and temporal drift $( \mathbf { e . g . } , \mathcal { O } ( \Delta x ^ { 2 } )$ for standard spatial finite difference stencils). These GT residuals serve as the theoretical numerical precision limit, or the “floor,” for our evaluation framework.

Overfitting the Evaluation Stencil (Discrete Floor Violations) An interesting phenomenon arises when analyzing optimization-heavy baselines: they occasionally report global scaled PDE or mass residuals that are strictly lower than the GT floor. As observed in the broader literature on Physics-Informed Neural Networks [42, 43], breaching the GT floor does not necessarily indicate the discovery of a structurally superior physical solution. Rather, it is often a manifestation of stencil overfitting.

When a generative baseline heavily relies on a specific discrete evaluation operator to guide its generation (e.g., via hard algebraic projections or continuous gradient penalties), the model treats the evaluation stencil itself as the absolute optimization objective. Most importantly, because the base generative prior is trained exclusively on the GT dataset, any sample whose discrete residual is forced significantly below the GT floor is, by definition, pushed outside the support of the training distribution. The model is essentially generating out-of-distribution samples that it has never seen, finding a local minimum where localized, high-frequency perturbations zero out the discrete arithmetic of the stencil while drifting from the true continuous physics.

The Waterbed Effect We term this topological trade-off the Waterbed Effect: artificially suppressing the scaled PDE error below the theoretical floor inevitably causes a complementary degradation in unconstrained metrics. While the generated samples generally maintain their macroscopic coherence, this subtle overfitting is empirically validated by two corresponding statistical signals in our evaluation:

1. An increase in Raw PDE Error: The unscaled spatial derivatives become abnormally large compared to the vanilla unguided model, revealing the presence of subtle high-frequency artifacts injected across the generated field.

2. A degradation of Sample MSE: Because the generated sample is pulled artificially off the true underlying data manifold to satisfy the discrete stencil, it results in a measurable worsening of the core distributional metrics.

Our EGT framework mitigates this effect structurally. By guiding the generative process smoothly and deferring explicit physics optimizations, the framework steers the generation toward the GT floor without over-constraining the features to the specific discrete stencil. This approach preserves the continuous physical manifold, successfully minimizing the physical residual without artificially inflating the complementary distributional metrics.

Table 6: Intrinsic physical residuals of the Ground Truth simulated datasets. These values represent the numerical precision limit (lower bound) dictated by grid discretization. To maintain strict consistency with our main evaluations, we report the global mean absolute error (MAE) computed using both the Partial and Full generative configurations. (Note: The Heat equation dataset is evaluated exclusively on Initial Conditions, hence the absence ofBC error).
<table><tr><td>Dataset</td><td>IC Error</td><td>BC Error</td><td>Mass Error</td><td>PDE Error (Scaled)</td><td>PDE Error (Raw)</td><td>Partial Phys (MAE)</td><td>Full Phys (MAE)</td></tr><tr><td>Heat Equation</td><td>0.00e0</td><td></td><td>1.32e-7</td><td>3.57e-5</td><td>9.04e-3</td><td>5.44e-8</td><td>3.53e-5</td></tr><tr><td>Reaction-Diff.</td><td>0.00e0</td><td>0.00e0</td><td>1.90e-3</td><td>3.07e-7</td><td>5.03e-3</td><td>8.30e-4</td><td>3.04e-7</td></tr><tr><td>Burgers&#x27; Eq.</td><td>0.00e0</td><td>5.99e-9</td><td>4.70e-3</td><td>6.24e-5</td><td>6.24e-1</td><td>1.56e-3</td><td>6.05e-5</td></tr><tr><td>Navier-Stokes</td><td>0.00e0</td><td>0.00e0</td><td>1.43e-7</td><td>6.98e-6</td><td>2.86e-2</td><td>1.66e-9</td><td>6.63e-6</td></tr></table>

## J Extended Results and Disaggregated Metrics

In this section, we provide the exhaustive evaluation metrics for all tested generative methods across our suite of PDE benchmarks. To facilitate a clear and comprehensive analysis, we consolidate the results into dedicated subsections for each physical system (Heat, Reaction-Diffusion, Burgers’, and Navier-Stokes).

Each physical system is evaluated under two distinct observation regimes:

• Partial Physics Guidance (Panel A): The generative models are constrained solely by partial observations, typically Initial Conditions (IC), Boundary Conditions (BC), and/or global integral quantities.

• Full PDE Guidance (Panel B): The generative models are subjected to a rigorous “stress test", where they are constrained by the full underlying differential equation across the entire spatiotemporal domain, alongside the boundary constraints.

Furthermore, we disaggregate the results based on the underlying generative prior: the standard Gaussian White Noise prior and the structured Smooth Gaussian Kernel prior.

## J.1 Evaluation Metrics Formulation

To rigorously assess the performance of the generative models, we track a diverse set of metrics encompassing computational stability, distributional fidelity, and physical consistency.

## Computational Stability and Efficiency:

• Success Rate (SR, %): Represents the percentage of samples successfully generated without encountering numerical divergence (e.g., exploding gradients resulting in NaN values) or Out-Of-Memory (OOM) errors. Trajectories that encountered numerical divergence or memory exhaustion (NaN/OOM) are denoted as DNC (Did Not Complete).

• Inference Speed (Spd, sec/sample): The average computational wall-clock time required to generate a single physical sample, amortized over the total batch size. Hardware Disclaimer: To ensure a strictly fair and unbiased comparison, all competing generative methods evaluated within a specific PDE benchmark were executed on the identical hardware (e.g., NVIDIA RTX 8000, A6000, L40 GPU). Consequently, while absolute inference times may vary across different physical systems due to hardware assignments and grid dimensionalities, the relative speed comparisons within each individual experiment are strictly standardized.

Distributional and Observational Fidelity: Following established practices in generative modeling for PDEs [23, 49], we evaluate the capability of the models to capture the true underlying data distribution. Let $U _ { p r e d } \in \mathbb { R } ^ { B \times N _ { x } \times N _ { t } }$ <sup>t</sup> and $U _ { t r u e } \mathbf { \bar { \Psi } } \in \mathbb { R } ^ { B \times N _ { x } \times N _ { t } }$ represent the batched generated and ground truth tensors.

• Mean MSE (MMSE): Measures the mean squared error between the sample mean of the generated batch and the ground truth batch: $\mathrm { M S E } ( \mu _ { p r e d } , \mu _ { t r u e } )$ . It indicates the accuracy of the estimated posterior mean.

• Standard Deviation MSE (SMSE): Measures the mean squared error between the sample standard deviations: $\mathbf { M S E } ( \sigma _ { p r e d } , \sigma _ { t r u e } )$ . It quantifies how well the method captures the predictive uncertainty.

• Sample-wise MSE (Sample MSE): The standard point-wise mean squared error evaluated directly between the predicted and true spatiotemporal fields.

Physical Consistency (Constraint Errors): To evaluate adherence to the governing physical laws, we compute absolute residuals for various constraints. All metrics are reported as Mean Absolute Errors (MAE) averaged over the generated samples.

• IC Error & BC Error: Measure the absolute deviation of the generated solution from the prescribed Initial Conditions (at t = 0) and boundary values (e.g., Dirichlet or Neumann fluxes).

• Mass Error & Conservation Laws (CL): Quantify the violation of global integral invariants. Mass explicitly refers to the conservation of mass over time (e.g., balancing total state sum with boundary fluxes and reaction sources), while CL refers to additional physical invariants or local multi-step flux balances (such as Godunov flux constraints in the Burgers equation).

• Raw PDE Error: The unscaled absolute error of the interior partial differential equation evaluated via discrete finite differences. As discussed in Appendix I, observing this metric is key to detect high-frequency artifacts and “stencil overfitting".

• Total Phys (MAE): This is the global Mean Absolute Error of the specific composite guidance function, $\mathcal { H } ( u )$ , employed during inference. It is important to notice that the composition of this metric depends on the evaluation regime:

– In the Partial Guidance task, the MAE is computed exclusively over the boundary constraints (IC, BC) and the integral mass/CL residuals.

– In the Full PDE Guidance task, the MAE concatenates the boundary constraints with the internal PDE residual. In this full configuration, the PDE residual is strictly scaled by the spatial grid discretization factor (e.g., ∆x<sup>2</sup>). This matches the exact H(u) loss formulation structurally required by optimization-based baselines to maintain numerical stability during gradient-based guidance.

## J.2 1D Heat Equation Results

Table 7 presents the comprehensive evaluation for the 1D Heat Equation. As the simplest linear PDE in our benchmark, this environment serves as a foundational test to measure the models’ capability to enforce basic spatial and temporal constraints before moving to non-linear dynamics.

Table 7: Comprehensive evaluation metrics for the 1D Heat Equation. The table compares zero-shot guidance performance across Partial (Boundary/IC) and Full-PDE tasks using both White Noise and Smooth Gaussian priors. ↓ indicates lower is better, ↑ indicates higher is better. All physical metrics are averaged over the test set. Instances with a success rate lower than 100% are marked as DNC. Best results are in bold, second-best are underlined (excluding Vanilla).
<table><tr><td>Metric</td><td>Vanilla</td><td>EGT (Ours)</td><td>Hybrid EGT (Ours)</td><td>PCFM</td><td>ProFlow</td><td>DiffPDE</td><td>ECI</td><td>D-Flow</td></tr><tr><td colspan="7">PART 1: PARTIAL/BOUNDARY GUIDANCE</td><td></td><td></td></tr><tr><td colspan="8">Prior: White Noise</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>0.2%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.354</td><td>0.399</td><td>0.483</td><td>DNC</td><td>0.402</td><td>0.318</td><td>0.473</td><td>5.676</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.256</td><td>0.051</td><td>0.019</td><td>DNC</td><td>0.189</td><td>0.026</td><td>0.013</td><td>0.085</td></tr><tr><td>IC Error (↓)</td><td>0.433</td><td>0.015</td><td>3.1e-5</td><td>DNC</td><td>0.270</td><td>4.3e-3</td><td>2.2e-20</td><td>0.011</td></tr><tr><td>Mass Error (↓)</td><td>1.8e-3</td><td>3.8e-3</td><td>0.033</td><td>DNC</td><td>1.6e-3</td><td>3.3e-3</td><td>0.027</td><td>0.152</td></tr><tr><td>MMSE (↓)</td><td>5.2e-3</td><td>1.1e-3</td><td>9.3e-4</td><td>DNC</td><td>4.9e-3</td><td>9.4e-4</td><td>3.3e-4</td><td>8.6e-3</td></tr><tr><td>SMSE (↓)</td><td>1.5e-2</td><td>6.1e-3</td><td>6.7e-3</td><td>DNC</td><td>5.0e-3</td><td>6.4e-3</td><td>5.3e-5</td><td>0.177</td></tr><tr><td>Sample MSE (↓)</td><td>0.083</td><td>0.015</td><td>0.017</td><td>DNC</td><td>0.062</td><td>0.013</td><td>0.014</td><td>0.348</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.19</td><td>0.37</td><td>0.46</td><td>DNC</td><td>8.12</td><td>0.94</td><td>8.30</td><td>15.58</td></tr><tr><td colspan="9">Prior: Smooth Kernel</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>1.1%</td><td>100%</td><td>100%</td><td>0%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>40.83</td><td>40.68</td><td>40.30</td><td>DNC</td><td>3.728</td><td>39.23</td><td>DNC</td><td>1.3e3</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.425</td><td>0.047</td><td>0.018</td><td>DNC</td><td>0.179</td><td>0.062</td><td>DNC</td><td>0.973</td></tr><tr><td>IC Error (↓)</td><td>0.984</td><td>0.010</td><td>2.7e-5</td><td>DNC</td><td>0.226</td><td>0.010</td><td>DNC</td><td>0.092</td></tr><tr><td>Mass Error (↓)</td><td>0.053</td><td>0.014</td><td>0.032</td><td>DNC</td><td>0.030</td><td>0.046</td><td>DNC</td><td>1.817</td></tr><tr><td>MMSE (↓)</td><td>1.3e-2</td><td>5.0e-4</td><td>3.7e-4</td><td>DNC</td><td>1.2e-2</td><td>2.6e-4</td><td>DNC</td><td>1.361</td></tr><tr><td>SMSE (↓)</td><td>3.4e-2</td><td>1.0e-2</td><td>1.0e-2</td><td>DNC</td><td>7.4e-3</td><td>6.2e-3</td><td>DNC</td><td>21.66</td></tr><tr><td>Sample MSE (↓)</td><td>0.256</td><td>0.049</td><td>0.049</td><td>DNC</td><td>0.052</td><td>0.043</td><td>DNC</td><td>24.88</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.19</td><td>0.37</td><td>0.43</td><td>DNC</td><td>8.16</td><td>0.94</td><td>DNC</td><td>15.63</td></tr><tr><td colspan="7">PART 2: FULL-PDE GUIDANCE</td><td></td><td></td></tr><tr><td colspan="7"></td><td></td><td></td><td></td></tr><tr><td>Success Rate (↑) Raw PDE Residual (↓)</td><td>100% 0.345</td><td>100% 0.530</td><td>100% 0.305</td><td>0%</td><td>0.4%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Physical Res. MAE (↓)</td><td>6.2e-3</td><td>5.1e-3</td><td></td><td>DNC</td><td>DNC DNC</td><td>0.025 0.985</td><td>0.472</td><td>7.875</td></tr><tr><td>IC Error (↓)</td><td>0.410</td><td>0.174</td><td>3.9e-3</td><td>DNC</td><td>DNC</td><td>18.05</td><td>1.8e-3</td><td>0.032</td></tr><tr><td>Mass Error (↓)</td><td>1.8e-3</td><td>0.017</td><td>0.156 0.012</td><td>DNC DNC</td><td>DNC</td><td>0.671</td><td>2.0e-20 0.026</td><td>28.12</td></tr><tr><td></td><td>4.9e-3</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.571</td></tr><tr><td>MMSE (↓) SMSE (↓)</td><td>1.4e-2</td><td>4.0e-3 0.015</td><td>4.0e-3 0.016</td><td>DNC</td><td>DNC DNC</td><td>0.137 16.81</td><td>1.0e-4</td><td>0.029</td></tr><tr><td>Sample MSE (↓)</td><td>0.078</td><td>0.046</td><td>0.049</td><td>DNC DNC</td><td>DNC</td><td>18.14</td><td>3.8e-5 0.014</td><td>22.11 23.66</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.13</td><td>0.23</td><td>0.27</td><td>DNC</td><td>DNC</td><td>0.68</td><td>10.28</td><td>18.06</td></tr><tr><td colspan="7">Prior: Smooth Kernel</td><td></td><td></td></tr><tr><td>Success Rate (↑)</td><td>100% 40.48</td><td>100%</td><td>100%</td><td>0%</td><td>0%</td><td>100%</td><td>0%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓) Physical Res. MAE (↓)</td><td>0.165</td><td>1.529 7.1e-3</td><td>0.914 4.5e-3</td><td>DNC DNC</td><td>DNC DNC</td><td>20.16 0.079</td><td>DNC DNC</td><td>1.6e2 0.657</td></tr><tr><td>IC Error (↓)</td><td>0.941</td><td>0.027</td><td>0.018</td><td>DNC</td><td>DNC</td><td>9.8e-3</td><td>DNC</td><td>0.117</td></tr><tr><td>Mass Error (↓)</td><td>0.052</td><td>0.066</td><td>0.044</td><td>DNC</td><td>DNC</td><td>0.024</td><td>DNC</td><td>3.298</td></tr><tr><td>MMSE (↓)</td><td>0.013</td><td>1.2e-3</td><td>1.2e-3</td><td>DNC</td><td>DNC</td><td>3.1e-4</td><td>DNC</td><td>0.620</td></tr><tr><td>SMSE (↓)</td><td>0.034</td><td>2.3e-3</td><td>2.1e-3</td><td>DNC</td><td>DNC</td><td>6.6e-3</td><td>DNC</td><td>15.47</td></tr><tr><td>Sample MSE (↓)</td><td>0.248</td><td>0.035</td><td>0.035</td><td>DNC</td><td>DNC</td><td>0.045</td><td>DNC</td><td>17.61</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.19</td><td>0.51</td><td>0.55</td><td>DNC</td><td>DNC</td><td>1.04</td><td>DNC</td><td>16.79</td></tr></table>

## J.2.1 Discussion

The empirical results for the 1D Heat Equation highlight the distinct operational characteristics of the evaluated methods, particularly regarding their sensitivity to the underlying generative prior and the complexity of the imposed constraints.

Performance under White Noise Prior Given the linear nature of the Heat Equation, the uncorrelated White Noise regime favors baseline methods equipped with exact algebraic correction mechanisms. Most notably, the ECI framework achieves state-of-the-art performance across almost all metrics in both Partial and Full-PDE tasks. It successfully enforces the Initial Condition to near machine precision $( 2 . 2 \times 1 0 ^ { - 2 0 } )$ while maintaining excellent distributional fidelity (Sample MSE 0.014), albeit at a higher computational cost (∼ 10 seconds per sample). DiffusionPDE demonstrates strong performance in the Partial task (Sample MSE 0.013), but its distributional fidelity degrades significantly under Full-PDE guidance, where the Sample MSE rises to 18.14 despite maintaining numerical stability. ProFlow operates reliably under Partial guidance (100% SR) but struggles to complete the Full-PDE stress test (0.4% SR). Finally, while D-Flow maintains a 100% success rate, it incurs significant computational overhead (∼ 18s) and struggles to preserve the underlying data distribution, yielding elevated Sample MSE and physical errors. PCFM encounters severe numerical instabilities in this environment, resulting in Did Not Complete (DNC) in all Heat experiments.

Prior Shift to Smooth Gaussian Kernels The operational landscape shifts considerably when transitioning to the spatially correlated Smooth Kernel prior. Methods reliant on hard algebraic projections struggle to adapt to this structured kernel: ECI drops to a 0% success rate, being unable to interpolate smoothly without diverging, and PCFM remains predominantly DNC. ProFlow continues to stably solve the Partial task but again fails to scale to the Full-PDE formulation (0% SR). While DiffusionPDE safely completes all trajectories (100% SR), its physical correction capability is limited in the Partial task (yielding a Raw PDE residual of 39.23, comparable to the unguided Vanilla model) and introduces distributional drift in the Full-PDE task.

Robustness and Efficiency of the Proposed Method The proposed EGT framework and its Hybrid variant demonstrate consistent robustness, maintaining a 100% success rate across all generative priors and constraint complexities. While EGT slightly trails ECI’s exact numerical matching in the simpler White Noise Partial task, it exhibits superior adaptability in the structured prior regime. Under the challenging Smooth Kernel Full-PDE task, Hybrid EGT effectively balances physical adherence and manifold preservation: it reduces the Raw PDE residual to 0.914 (down from the Vanilla baseline of 40.48) while maintaining a highly accurate Sample MSE of 0.035. Most importantly, this consistency is achieved with minimal computational overhead. Hybrid EGT requires only 0.55 seconds per sample, making it highly efficient compared to optimization-heavy baselines like D-Flow (∼ 16.7s) or ECI’s White Noise performance (∼ 10.2s).

## J.3 1D Burgers’ Equation Results

Table 8 presents the comprehensive evaluation for the 1D Burgers’ Equation. As a canonical nonlinear PDE that develops steep gradients and shock waves over time, this dataset introduces significant complexity, stress-testing the models’ ability to transport the generative flow accurately while avoiding numerical divergence in highly stiff regions.

## J.3.1 Discussion

The empirical results for the 1D Burgers’ Equation illustrate the varying scalability of guidance methods when subjected to non-linear physical dynamics and the formation of steep spatial gradients (shock waves).

Performance under White Noise Prior In the uncorrelated prior regime, optimization and proximal methods demonstrate distinct trade-offs. ECI achieves the best distributional fidelity (Sample MSE 0.110 in Partial, 0.116 in Full) and enforces Initial Conditions to machine precision, though it requires higher inference times $( \sim 1 8 - 2 1 \mathrm { s ) }$ . DiffusionPDE provides a highly efficient balance, lowering the Sample MSE compared to the Vanilla baseline while running in near real-time (∼ 1.1s). ProFlow successfully minimizes the Raw PDE Residual (0.949) more effectively than other methods, but doing so slightly increases the Sample MSE (0.239) and incurs significant computational overhead $( \sim 1 8 - 2 1 \mathrm { s } )$ . D-Flow avoids numerical divergence but consistently yields elevated physical errors and requires the longest inference times $( \sim 3 2 - 3 9 \mathrm { s } )$ . Notably, while PCFM minimizes boundary errors in the Partial task, its explicit Jacobian formulation struggles fundamentally with the non-linear shock dynamics under Full-PDE constraints, resulting in severe numerical overflow (Physical MAE $> 1 0 ^ { 1 4 } )$ .

Table 8: Comprehensive evaluation metrics for the 1D Burgers’ Equation. The table compares zero-shot guidance performance across Partial (Boundary/IC) and Full-PDE tasks using both White Noise and Smooth Gaussian priors. ↓ indicates lower is better, ↑ indicates higher is better. All physical metrics are averaged over the test set. Instances with a success rate lower than 100% are marked as DNC. <sup>†</sup> indicates values that artificially fall below the discrete ground-truth stencil precision limit (stencil overfitting/waterbed effect).
<table><tr><td>Metric</td><td>Vanilla</td><td>EGT (Ours)</td><td>Hybrid EGT (Ours)</td><td>PCFM</td><td>ProFlow</td><td>DiffPDE</td><td>ECI</td><td>D-Flow</td></tr><tr><td colspan="9">PART 1: PARTIAL/BOUNDARY GUIDANCE</td></tr><tr><td>Prior: White Noise</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100% 0.949</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.955</td><td>1.247</td><td>1.441 0.013</td><td>1.266 1.2e-8†</td><td>0.134</td><td>1.128 0.059</td><td>1.205 0.049</td><td>4.370 0.083</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.102</td><td>0.051</td><td>0.166</td><td>0.187</td><td>0.184</td><td>8.1e-3</td><td>6.0e-20</td><td>0.174</td></tr><tr><td>IC Error (↓)</td><td>0.175</td><td>0.167 7.0e-3</td><td>2.5e-5</td><td>5.5e-16†</td><td>0.098</td><td>6.2e-4</td><td>1.2e-4</td><td>0.160</td></tr><tr><td>BC Error (↓)</td><td>0.064</td><td>0.030</td><td>0.032</td><td>3.6e-8†</td><td>2.2e-3†</td><td>0.141</td><td>0.138</td><td>0.083</td></tr><tr><td>Mass Error (↓)</td><td>4.9e-3 0.077</td><td>0.107</td><td>0.083</td><td>0.066</td><td>0.172</td><td>0.132</td><td>0.140</td><td>0.090</td></tr><tr><td>CL Error (↓)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMSE (↓)</td><td>0.021</td><td>0.039</td><td>0.038</td><td>0.028</td><td>0.075</td><td>0.017</td><td>4.2e-3</td><td>0.023</td></tr><tr><td>SMSE (↓) Sample MSE (↓)</td><td>5.1e-3</td><td>5.7e-3 0.206</td><td>5.8e-3 0.205</td><td>6.2e-3 0.193</td><td>8.0e-3 0.235</td><td>4.9e-3</td><td>1.5e-3</td><td>0.104</td></tr><tr><td></td><td>0.205</td><td></td><td></td><td></td><td></td><td>0.155</td><td>0.110</td><td>0.309</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.41</td><td>0.48</td><td>0.52</td><td>1.29</td><td>21.34</td><td>1.17</td><td>21.87</td><td>39.61</td></tr><tr><td colspan="9">Prior: Smooth Kernel</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>1.543</td><td>1.684</td><td>1.722</td><td>1.856</td><td>1.052</td><td>1.733</td><td>1.512</td><td>3.6e2</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.127</td><td>0.021</td><td>4.6e-3</td><td>1.2e-8†</td><td>0.084</td><td>0.053</td><td>0.068</td><td>0.372</td></tr><tr><td>IC Error (↓)</td><td>0.130</td><td>0.131</td><td>0.131</td><td>0.206</td><td>0.156</td><td>0.019</td><td>2.5e-18†</td><td>14.673</td></tr><tr><td>BC Error (↓)</td><td>0.080</td><td>1.3e-3</td><td>4.6e-6</td><td>5.5e-16†</td><td>0.042</td><td>1.4e-3</td><td>2.9e-4</td><td>8.939</td></tr><tr><td>Mass Error (↓)</td><td>0.021</td><td>0.014</td><td>0.012</td><td>3.7e-8†</td><td>6.8e-3</td><td>0.106</td><td>0.188</td><td>0.532</td></tr><tr><td>CL Error (↓)</td><td>0.065</td><td>0.077</td><td>0.068</td><td>0.066</td><td>0.048</td><td>0.111</td><td>0.205</td><td>0.109</td></tr><tr><td>MMSE (↓)</td><td>4.4e-3</td><td>6.0e-3</td><td>5.9e-3</td><td>0.026</td><td>0.074</td><td>3.8e-3</td><td>0.057</td><td>0.037</td></tr><tr><td>SMSE (↓)</td><td>4.7e-3</td><td>4.9e-3</td><td>4.9e-3</td><td>0.015</td><td>0.065</td><td>4.9e-3</td><td>0.060</td><td>9.320</td></tr><tr><td>Sample MSE (↓)</td><td>0.16</td><td>0.137</td><td>0.137</td><td>0.223</td><td>0.164</td><td>0.126</td><td>0.142</td><td>10.584</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.41</td><td>0.48</td><td>0.52</td><td>1.28</td><td>21.41</td><td>1.16</td><td>21.94</td><td>39.29</td></tr><tr><td colspan="9">PART 2: FULL-PDE GUIDANCE</td></tr><tr><td colspan="9">Prior: White Noise</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.955</td><td>1.328</td><td>1.529</td><td>4.9e+18</td><td>0.947</td><td>1.131</td><td>1.221</td><td>5.116</td></tr><tr><td>Physical Res. MAE (↓)</td><td>5.1e-3</td><td>1.9e-3</td><td>1.5e-4</td><td>4.8e+14</td><td>6.3e-3</td><td>9.5e-4</td><td>2.0e-4</td><td>2.9e-3</td></tr><tr><td>IC Error (↓)</td><td>0.175</td><td>0.011</td><td>1.5e-9†</td><td>1.1e+15</td><td>0.184</td><td>7.6e-3</td><td>6.0e-20†</td><td>0.170</td></tr><tr><td>BC Error (↓)</td><td>0.062</td><td>6.6e-3</td><td>1.8e-9†</td><td>4.4e+5</td><td>0.101</td><td>6.2e-4</td><td>8.7e-5</td><td>0.095</td></tr><tr><td>Mass Error (↓)</td><td>4.9e-3</td><td>0.147</td><td>0.196</td><td>2.9e+14</td><td>3.2e-3</td><td>0.142</td><td>0.143</td><td>0.108</td></tr><tr><td>CL Error (↓)</td><td>0.074</td><td>0.156</td><td>0.200</td><td>3.0e+5</td><td>0.174</td><td>0.134</td><td>0.140</td><td>0.096</td></tr><tr><td>MMSE (↓)</td><td>0.021</td><td>0.033</td><td>0.033</td><td>1.3e+14</td><td>0.073</td><td>0.018</td><td>5.7e-3</td><td>0.022</td></tr><tr><td>SMSE (↓)</td><td>4.7e-3</td><td>4.8e-3</td><td>5.2e-3</td><td>1.1e+17</td><td>6.8e-3</td><td>4.5e-3</td><td>1.7e-3</td><td>0.140</td></tr><tr><td>Sample MSE (↓)</td><td>0.204</td><td>0.189</td><td>0.189</td><td>1.1e+17</td><td>0.239</td><td>0.153</td><td>0.116</td><td>0.346</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.26</td><td>0.45</td><td>0.54</td><td>67.11</td><td>18.68</td><td>1.14</td><td>18.67</td><td>32.69</td></tr><tr><td colspan="9">Prior: Smooth Kernel</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>1.538</td><td>1.819</td><td>1.900</td><td>5.9e+17</td><td>1.053</td><td>1.727</td><td>1.513</td><td>4.0e2</td></tr><tr><td>Physical Res. MAE (↓)</td><td>5.8e-3</td><td>1.5e-3</td><td>1.8e-4</td><td>5.7e+13</td><td>4.5e-3</td><td>1.5e-3</td><td>3.1e-4</td><td>0.048</td></tr><tr><td>IC Error (↓)</td><td>0.127</td><td>0.018</td><td>1.6e-9†</td><td>3.3e+15</td><td>0.153</td><td>0.018</td><td>2.5e-18†</td><td>16.369</td></tr><tr><td>BC Error (↓)</td><td>0.082</td><td>1.3e-3</td><td>3.4e-10†</td><td>9.1e+4</td><td>0.043</td><td>1.4e-3</td><td>2.9e-4</td><td>16.783</td></tr><tr><td>Mass Error (↓)</td><td>0.022</td><td>0.108</td><td>0.174</td><td>8.3e+14</td><td>7.2e-3</td><td>0.107</td><td>0.188</td><td>0.369</td></tr><tr><td>CL Error (↓)</td><td>0.065</td><td>0.125</td><td>0.184</td><td>2.7e+5</td><td>0.048</td><td>0.112</td><td>0.205</td><td>0.129</td></tr><tr><td>MMSE (↓)</td><td>4.0e-3</td><td>4.9e-3</td><td>4.9e-3</td><td>2.8e+13</td></table>

Prior Shift to Smooth Gaussian Kernels Transitioning to the spatially correlated prior exacerbates the challenges for certain baselines while favoring others. DiffusionPDE and ECI remain highly stable, effectively lowering the Sample MSE (0.124 and 0.142 in Full-PDE, respectively) compared to the Vanilla baseline (0.158). ProFlow maintains the lowest Raw PDE Residual (1.053) but remains computationally demanding. Conversely, D-Flow exhibits high sensitivity to the correlated prior, severely degrading the visual fidelity and causing its Sample MSE to spike above 10. PCFM continues to experience massive numerical overflow in the Full-PDE task. In the Partial task, PCFM artificially drives boundary errors below the Ground Truth numerical limit (stencil overfitting, marked with <sup>†</sup>), which structurally disconnects the flow from the learned manifold and degrades the Sample MSE (0.223) relative to Vanilla (0.160).

Robustness of the Proposed Method Our continuous frameworks (EGT and Hybrid EGT) robustly handle both the non-linear shocks of the Burgers’ equation and the complexities of the prior shift. EGT consistently improves the Sample MSE over the Vanilla baseline while maintaining sub-second inference speeds. Hybrid EGT proves highly effective under Full-PDE constraints, achieving the lowest overall Physical Residual MAE $( \mathrm { \bar { 1 } . 5 \check { \times } 1 0 ^ { - 4 } }$ for White Noise and $1 . 8 \times 1 0 ^ { - 4 }$ for Smooth Kernel) without relying on computationally expensive unrolled trajectory optimizations. Objectively, it must be noted that under the intense terminal refinement phase of Hybrid EGT in the Full-PDE task, the local IC and BC errors are driven below the numerical floor (marked with <sup>†</sup>). However, unlike strict algebraic projections, this localized boundary optimization does not negatively impact the global manifold, as Hybrid EGT still achieves a highly competitive Sample MSE (e.g., 0.128 under Smooth Kernel) at a fraction of the inference time (∼ 1.06s) required by ECI or ProFlow.

## J.4 1D Reaction-Diffusion Equation Results

Table 9 presents the comprehensive evaluation for the 1D Reaction-Diffusion (RD) Equation. By introducing a non-linear reaction source term that actively competes with spatial diffusion, this PDE dictates complex transient dynamics and localized pattern formations. Consequently, it rigorously tests the solvers’ ability to balance competing physical forces across the generative trajectory without falling into numerical instability.

## J.4.1 Discussion

The empirical results for the Reaction-Diffusion Equation heavily expose the structural limitations of rigid algebraic projections when faced with coupled non-linear PDE constraints over dense temporal grids.

Performance under White Noise Prior Under standard White Noise and simple Partial (Boundary/IC) constraints, most baseline methods maintain reasonable stability. However, scaling to the Full-PDE task exposes the scalability limits of explicit projection frameworks. Both PCFM and ProFlow encounter severe numerical instabilities along the flow; although they technically complete the generation without throwing NaN exceptions (100% Success Rate), their Raw PDE Residuals experience severe overflow, reaching $2 . 2 \times \mathrm { { 1 0 ^ { 9 } } }$ and $4 . 8 \times 1 0 ^ { 9 }$ , respectively. In contrast, DiffPDE and ECI handle the White Noise environment more reliably, maintaining sub-optimal but stable physical residuals without diverging.

Prior Shift and stencil overfitting The introduction of the spatially correlated Smooth Kernel prior further amplifies the divergence between continuous guidance and post-hoc approximations. The Waterbed Effect is starkly visible in PCFM during Partial tasks: it drives the physical and boundary residuals down to precision limits (marked with <sup>†</sup>), but this aggressive optimization of the discrete stencil severely degrades the underlying manifold, resulting in poor Sample MSE scores. Furthermore, the prior shift heavily impacts optimization-based methods on Full-PDE tasks. For example, ECI, which performed admirably under White Noise, struggles to interpolate the correlated noise, causing its Sample MSE to degrade to 0.365 under the Smooth Kernel Full-PDE task.

Table 9: Comprehensive evaluation metrics for the 1D Reaction-Diffusion Equation. The table compares zero-shot guidance performance across Partial (Boundary/IC) and Full-PDE tasks using both White Noise and Smooth Gaussian priors. ↓ indicates lower is better, ↑ indicates higher is better. All physical metrics are averaged over the test set. Instances with a success rate lower than 100% are marked as DNC.. <sup>†</sup> indicates values that artificially fall below the discrete ground-truth stencil precision limit (stencil overfitting/waterbed effect).
<table><tr><td>Metric</td><td>Vanilla</td><td>EGT (Ours)</td><td>Hybrid EGT (Ours)</td><td>PCFM</td><td>ProFlow</td><td>DiffPDE</td><td>ECI</td><td>D-Flow</td></tr><tr><td colspan="7">PART 1: PARTIAL/BOUNDARY GUIDANCE</td><td></td><td></td></tr><tr><td>Prior: White Noise</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.086</td><td>0.222</td><td>0.423</td><td>0.716</td><td>7.3e+3</td><td>0.184</td><td>1.185</td><td>4.258</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.200</td><td>0.075</td><td>8.5e-3</td><td>1.1e-8†</td><td>1.2e+3</td><td>0.043</td><td>0.054</td><td>0.739</td></tr><tr><td>IC Error (↓)</td><td>0.181</td><td>0.024</td><td>5.5e-6</td><td>1.4e-44</td><td>4.5e+5</td><td>7.8e-3</td><td>1.5e-20</td><td>2.0e+2</td></tr><tr><td>BC Error (↓)</td><td>6.9e-4</td><td>1.7e-3</td><td>0.032</td><td>0.154</td><td>7.8e+6</td><td>7.8e-4</td><td>0.224</td><td>3.040</td></tr><tr><td>Mass Error (↓)</td><td>4.5e-3</td><td>0.011</td><td>0.017</td><td>2.5e-8†</td><td>2.6e+3</td><td>0.016</td><td>0.124</td><td>1.077</td></tr><tr><td>CL Error (↓)</td><td>0.025</td><td>0.052</td><td>0.109</td><td>0.047</td><td>17.208</td><td>0.026</td><td>0.175</td><td>0.054</td></tr><tr><td>MMSE (↓)</td><td>2.0e-3</td><td>9.1e-3</td><td>8.9e-3</td><td>3.1e-3</td><td>3.3e+3</td><td>1.2e-3</td><td>0.041</td><td>0.245</td></tr><tr><td>SMSE (↓)</td><td>4.6e-3</td><td>5.0e-3</td><td>5.2e-3</td><td>6.7e-3</td><td>7.4e+5</td><td>5.1e-3</td><td>0.093</td><td>1.9e+2</td></tr><tr><td>Sample MSE (↓)</td><td>0.083</td><td>0.051</td><td>0.050</td><td>0.041</td><td>7.4e+5</td><td>0.023</td><td>0.246</td><td>2.0e+2</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.38</td><td>1.13</td><td>1.24</td><td>2.16</td><td>25.65</td><td>2.00</td><td>24.47</td><td>32.07</td></tr><tr><td>Prior: Smooth Kernel</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success Rate (↑) Raw PDE Residual (↓)</td><td>100% 1.274</td><td>100% 1.697</td><td>100% 1.460</td><td>100% 3.636</td><td>100% 2.604</td><td>100% 1.309</td><td>100%</td><td>100%</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.220</td><td>0.028</td><td>8.5e-3</td><td>1.1e-8†</td><td>0.239</td><td>0.027</td><td>8.682</td><td>10.993</td></tr><tr><td></td><td></td><td></td><td>4.6e-6</td><td>5.7e-44</td><td>0.219</td><td>4.6e-4</td><td>0.086</td><td>0.026</td></tr><tr><td>IC Error (↓) BC Error (↓)</td><td>0.182 0.018</td><td>1.1e-3 0.117</td><td>0.039</td><td>0.164</td><td>0.084</td><td>0.017</td><td>1.4e-18</td><td>5.2e-3</td></tr><tr><td>Mass Error (↓)</td><td></td><td></td><td></td><td>2.5e-8†</td><td></td><td></td><td>1.118</td><td>0.088</td></tr><tr><td></td><td>0.053</td><td>0.027</td><td>0.017</td><td></td><td>0.056</td><td>0.043</td><td>0.198</td><td>0.038</td></tr><tr><td>CL Error (↓)</td><td>0.031</td><td>0.036</td><td>0.053</td><td>0.100</td><td>0.041</td><td>0.041</td><td>0.091</td><td>0.071</td></tr><tr><td>MMSE (↓)</td><td>1.4e-3</td><td>1.1e-3</td><td>1.0e-3</td><td>3.5e-4</td><td>0.058</td><td>4.3e-4</td><td>0.044</td><td>1.7e-3</td></tr><tr><td>SMSE (↓)</td><td>3.8e-3</td><td>7.0e-4</td><td>6.8e-4</td><td>0.036</td><td>0.043</td><td>6.3e-4</td><td>0.179</td><td>0.015</td></tr><tr><td>Sample MSE (↓)</td><td>0.088</td><td>0.010</td><td>9.9e-3</td><td>0.087</td><td>0.191</td><td>9.3e-3</td><td>0.365</td><td>0.094</td></tr><tr><td>Speed (s/sample) (↓)</td><td></td><td>1.09</td><td>1.25</td><td>2.11</td><td>25.60</td><td>1.98</td><td></td><td></td></tr><tr><td></td><td>0.38</td><td></td><td></td><td></td><td></td><td></td><td>24.43</td><td>31.43</td></tr><tr><td colspan="9">PART 2: FULL-PDE GUIDANCE</td></tr><tr><td>Prior: White Noise</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.086</td><td>0.209</td><td>0.322</td><td>2.2e+9</td><td>4.8e+9</td><td>0.183</td><td>1.185</td><td>0.703</td></tr><tr><td>Physical Res. MAE (↓)</td><td>3.5e-3</td><td>1.2e-3</td><td>2.0e-5</td><td>1.3e+5</td><td>2.9e+5</td><td>6.5e-3</td><td>7.2e-5</td><td>6.6e-3</td></tr><tr><td>IC Error (↓)</td><td>0.181</td><td>0.021</td><td>1.1e-10</td><td>1.5e+5</td><td>5.0e+11</td><td>7.7e-3</td><td>1.5e-20</td><td>0.168</td></tr><tr><td>BC Error (↓)</td><td>6.9e-4</td><td>5.6e-4</td><td>5.6e-3</td><td>7.6e+18</td><td>3.0e+11</td><td>7.8e-4</td><td>0.224</td><td>0.085</td></tr><tr><td>Mass Error (↓)</td><td>4.5e-3</td><td>0.034</td><td>0.085</td><td>6.5e+14</td><td>2.5e+9</td><td>0.015</td><td>0.124</td><td>0.104</td></tr><tr><td>CL Error (↓)</td><td>0.025</td><td>0.056</td><td>0.106</td><td>1.0e+5</td><td>8.8e+2</td><td>0.026</td><td>0.175</td><td>0.045</td></tr><tr><td>MMSE (↓)</td><td>1.9e-3</td><td>8.0e-3</td><td>8.0e-3</td><td>8.6e+13</td><td>3.3e+9</td><td>1.2e-3</td><td>0.041</td><td>2.2e-3</td></tr><tr><td>SMSE (↓)</td><td>4.6e-3</td><td>5.2e-3</td><td>5.2e-3</td><td>6.7e+16</td><td>5.0e+11</td><td>5.0e-3</td><td>0.093</td><td>0.065</td></tr><tr><td>Sample MSE (↓)</td><td>0.083</td><td>0.049</td><td>0.049</td><td>6.7e+16</td><td>5.0e+11</td><td>0.023</td><td>0.246</td><td>0.169</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.376</td><td>0.891</td><td>0.928</td><td>76.720</td><td>25.055</td><td>1.860</td><td>24.343</td><td>31.443</td></tr><tr><td colspan="9">Prior: Smooth Kernel</td></tr><tr><td>Success Rate (↑) Raw PDE Residual (↓)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Physical Res. MAE (↓)</td><td>1.274 3.6e-3</td><td>1.335</td><td>1.324</td><td>9.8e+8</td><td>2.609</td><td>1.310</td><td>8.682</td><td>12.475</td></tr><tr><td>IC Error (↓)</td><td>0.182</td><td>3.6e-4</td><td>8.0e-5 9.9e-12</td><td>5.9e+4</td><td>4.0e-3</td><td>2.3e-4</td><td>5.2e-4</td><td>1.5e-3</td></tr><tr><td>BC Error (↓)</td><td></td><td>1.1e-3</td><td></td><td>1.7e+6</td><td>0.219</td><td>4.6e-4</td><td>1.4e-18</td><td>6.214</td></tr><tr><td></td><td>0.018</td><td>0.017</td><td>0.017</td><td>2.4e+19</td><td>0.088</td><td>0.018</td><td>1.118 0.198</td><td>3.369 0.153</td></tr><tr><td>Mass Error (↓) CL Error (↓)</td><td>0.053 0.031</td><td>0.052 0.036</td><td>0.047 0.052</td><td>2.1e+15 2.0e+5</td><td>0.082 0.041</td><td>0.051 0.041</td></table>

Robustness of the Proposed Method Our continuous guidance architectures (EGT and Hybrid EGT) stand out as the most robust and computationally efficient frameworks across all configurations. Hybrid EGT consistently excels in the challenging Full-PDE tasks, achieving the best Physical Residual MAE under the White Noise prior $( 2 . 0 \times \mathrm { 1 0 ^ { - 5 } } )$ and maintaining exceptional fidelity under the Smooth Kernel $( 8 . 0 \times 1 0 ^ { - 5 } )$ . Most importantly, our framework enforces these complex non-linear dynamics without resorting to stencil overfitting and without compromising the macroscopic visual coherence of the fields (maintaining top-tier Sample MSE). Beyond raw accuracy, EGT significantly outperforms optimization-based baselines in computational efficiency: it resolves the Full-PDE Smooth Kernel task in just ∼ 0.95 seconds per sample, bypassing the severe bottlenecks observed in ECI (∼ 24 seconds) and PCFM (∼ 77 seconds).

## J.5 2D Navier-Stokes Equations Results

Table 10 presents the comprehensive evaluation for the 2D Navier-Stokes (NS) Equations. As the only two-dimensional macroscopic fluid dynamics environment in our benchmark, this task requires enforcing complex vorticity structures, non-linear advection, and strict incompressibility over a dense spatial grid. This environment pushes the structural and computational limits of guided generative models.

Table 10: Comprehensive evaluation metrics for the 2D Navier-Stokes Equations. The table evaluates zero-shot guidance performance across Partial (Boundary/IC) and Full-PDE tasks using a White Noise prior. ↓ indicates lower is better, ↑ indicates higher is better. All physical metrics are averaged over the test set. Instances with a success rate lower than 100% are marked as DNC. <sup>†</sup> indicates values that artificially fall below the discrete ground-truth stencil precision limit (stencil overfitting/waterbed effect).
<table><tr><td>Metric</td><td>Vanilla</td><td>EGT (Ours)</td><td>Hybrid EGT (Ours)</td><td>PCFM</td><td>ProFlow</td><td>DiffPDE</td><td>ECI</td><td>D-Flow</td></tr><tr><td colspan="9">PART 1: PARTIAL/BOUNDARY GUIDANCE</td></tr><tr><td colspan="9">Prior: White Noise</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>14.2%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.063</td><td>0.060</td><td>0.060</td><td>0.075</td><td>0.057</td><td>0.070</td><td>0.071</td><td>DNC</td></tr><tr><td>Physical Res. MAE (↓)</td><td>0.291</td><td>0.010</td><td>1.4e-4</td><td>1.2e-10†</td><td>0.197</td><td>0.124</td><td>3.8e-5</td><td>DNC</td></tr><tr><td>IC Error (↓)</td><td>0.136</td><td>1.7e-4</td><td>2.4e-8</td><td>0.0</td><td>0.063</td><td>0.025</td><td>8.9e-17</td><td>DNC</td></tr><tr><td>BC Error (↓)</td><td>0.058</td><td>0.056</td><td>0.057</td><td>0.066</td><td>0.058</td><td>0.057</td><td>0.060</td><td>DNC</td></tr><tr><td>Mass Error (↓)</td><td>0.011</td><td>7.6e-3</td><td>6.2e-3</td><td>1.0e-8†</td><td>3.1e-3</td><td>0.039</td><td>3.2e-3</td><td>DNC</td></tr><tr><td>CL Error (↓)</td><td>0.012</td><td>7.6e-3</td><td>6.1e-3</td><td>1.0e-8</td><td>3.3e-3</td><td>0.038</td><td>3.1e-3</td><td>DNC</td></tr><tr><td>MMSE (↓)</td><td>0.018</td><td>0.018</td><td>0.018</td><td>0.037</td><td>0.020</td><td>0.036</td><td>0.116</td><td>DNC</td></tr><tr><td>SMSE (↓)</td><td>0.011</td><td>0.011</td><td>0.011</td><td>0.023</td><td>9.9e-3</td><td>0.011</td><td>0.011</td><td>DNC</td></tr><tr><td>Sample MSE (↓)</td><td>0.542</td><td>0.529</td><td>0.548</td><td>0.756</td><td>0.550</td><td>0.541</td><td>0.630</td><td>DNC</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.690</td><td>0.714</td><td>0.898</td><td>93.972</td><td>0.847</td><td>1.402</td><td>3.559</td><td>DNC</td></tr><tr><td colspan="9">PART 2: FULL-PDE GUIDANCE</td></tr><tr><td colspan="9">Prior: White Noise</td></tr><tr><td>Success Rate (↑)</td><td>100%</td><td>100%</td><td>100%</td><td>0%</td><td>100%</td><td>100%</td><td>100%</td><td>1.6%</td></tr><tr><td>Raw PDE Residual (↓)</td><td>0.062</td><td>0.080</td><td>0.124</td><td>DNC</td><td>0.057</td><td>0.069</td><td>0.070</td><td>DNC</td></tr><tr><td>Physical Res. MAE (↓)</td><td>7.5e-3</td><td>1.6e-3</td><td>2.9e-5</td><td>DNC</td><td>5.6e-3</td><td>4.2e-3</td><td>1.8e-3</td><td>DNC</td></tr><tr><td>IC Error (↓)</td><td>0.136</td><td>1.3e-3</td><td>4.9e-14</td><td>DNC</td><td>0.064</td><td>0.025</td><td>8.9e-17</td><td>DNC</td></tr><tr><td>BC Error (↓)</td><td>0.059</td><td>0.034</td><td>1.0e-6</td><td>DNC</td><td>0.057</td><td>0.057</td><td>0.059</td><td>DNC</td></tr><tr><td>Mass Error (↓)</td><td>9.4e-3</td><td>7.5e-3</td><td>6.4e-3</td><td>DNC</td><td>3.1e-3</td><td>0.035</td><td>3.2e-3</td><td>DNC</td></tr><tr><td>CL Error (↓)</td><td>9.9e-3</td><td>7.6e-3</td><td>6.2e-3</td><td>DNC</td><td>3.3e-3</td><td>0.035</td><td>3.1e-3</td><td>DNC</td></tr><tr><td>MMSE (↓)</td><td>0.019</td><td>0.018</td><td>0.018</td><td>DNC</td><td>0.022</td><td>0.032</td><td>0.098</td><td>DNC</td></tr><tr><td>SMSE (↓)</td><td>0.011</td><td>0.011</td><td>0.011</td><td>DNC</td><td>9.8e-3</td><td>0.010</td><td>0.010</td><td>DNC</td></tr><tr><td>Sample MSE (↓)</td><td>0.559</td><td>0.547</td><td>0.547</td><td>DNC</td><td>0.549</td><td>0.553</td><td>0.611</td><td>DNC</td></tr><tr><td>Speed (s/sample) (↓)</td><td>0.858</td><td>0.937</td><td>1.021</td><td>DNC</td><td>1.209</td><td>1.789</td><td>4.400</td><td>DNC</td></tr></table>

## J.5.1 Discussion

The empirical results for the Navier-Stokes equations decisively illustrate the scalability limit and structural fragilities of explicit projection methods when applied to high-dimensional physical systems.

Baseline Performance and Guidance Collapse Scaling the constraint complexity to a 2D grid exposes severe instabilities in baseline frameworks. D-Flow experiences significant guidance degradation, achieving a mere 14.2% Success Rate on Partial constraints and dropping to 1.6% on the Full-PDE task. More critically, PCFM experiences total numerical divergence under Full-PDE constraints (0% SR, yielding NaN metrics), indicating that its reliance on explicit Jacobian approximations is computationally and mathematically restrictive for complex macroscopic fluid flows. While methods like ProFlow, DiffPDE, and ECI technically complete the Full-PDE generation, they struggle to simultaneously satisfy the physics and preserve the underlying data distribution.

High-Dimensional stencil overfitting The Waterbed Effect is exceptionally prominent in this 2D setting. Under Partial constraints, PCFM algebraically forces the Initial Condition error to absolute zero (0.0) and the Physical Residual to precision limits $( 1 . 2 \times 1 0 ^ { - 1 0 }$ , marked with <sup>†</sup>). However, this aggressive structural constraint severely distorts the generative trajectory, heavily degrading the Sample MSE to 0.756 (markedly worse than the unguided Vanilla model at 0.542). ECI similarly exhibits stencil overfitting on the IC Error $( 8 . 9 \times 1 0 ^ { - 1 7 } )$ while suffering from visual and structural degradation (Sample MSE of 0.611 on the Full-PDE task).

Robustness of the Proposed Method Our Continuous Dissipative Guidance architectures (EGT and Hybrid EGT) successfully navigate the high-dimensional fluid manifold without succumbing to numerical divergence or stencil overfitting. On the highly constrained Full-PDE task, Hybrid EGT decisively outperforms all baselines, achieving a state-of-the-art Physical Residual MAE of $2 . 9 \times 1 0 ^ { - 5 }$ and driving boundary errors down to $1 . 0 \times 1 0 ^ { - 6 }$ . Crucially, our framework accomplishes this while perfectly preserving the structural integrity of the fluid (Sample MSE of 0.547). Beyond accuracy, EGT demonstrates unmatched scalability: it strictly satisfies the complex 2D Navier-Stokes physics in just ∼ 1.02 seconds per sample, completely bypassing the extreme computational bottlenecks of algebraic methods (e.g., PCFM requiring nearly 94 seconds per sample on Partial tasks).

## J.6 Macroscopic Physical Diagnostics

To evaluate how pointwise PDE residual minimization affects broader macroscopic statistics, we compute two additional holistic diagnostics on the generated samples from the main evaluation (White Noise, Full-PDE configuration): the 1D Wasserstein Distance between spatial gradient distributions (capturing shock statistics on 1D Burgers) and the Energy Spectrum $L ^ { 1 }$ Error (assessing spectral consistency on 2D Navier–Stokes).

Table 11: Macroscopic physical diagnostics on 1D Burgers and 2D Navier–Stokes (White Noise, Full-PDE).
<table><tr><td>Benchmark</td><td>Model</td><td>Diagnostic Metric</td><td> $\mathbf { M e a n } \pm \mathbf { S t d } \downarrow$ </td></tr><tr><td rowspan="5">Burgers 1D</td><td>Vanilla</td><td>Gradient Wasserstein Dist.</td><td> $5 . 8 5 5 \mathrm { e } { - 1 } \pm 2 . 5 7 7 \mathrm { e } { - 1 }$ </td></tr><tr><td>DiffusionPDE</td><td>Gradient Wasserstein Dist.</td><td> $4 . 4 4 1 \mathrm { e } . 1 \pm 2 . 2 3 9 \mathrm { e } . 1$ </td></tr><tr><td>ECI</td><td>Gradient Wasserstein Dist.</td><td> $3 . 8 1 9 \mathrm { e } \mathbf { - 1 } \pm 2 . 3 2 7 \mathrm { e } \mathbf { - 1 }$ </td></tr><tr><td>Ours (Pure EGT)</td><td>Gradient Wasserstein Dist.</td><td> $6 . 8 7 0 \mathrm { e } - 1 \pm 3 . 0 1 0 \mathrm { e } - 1$ </td></tr><tr><td>Ours (Hybrid EGT)</td><td>Gradient Wasserstein Dist.</td><td> $8 . 3 9 9 \mathrm { e } . 1 \pm 3 . 6 8 7 \mathrm { e } . 1$ </td></tr><tr><td rowspan="5">Navier-Stokes 2D</td><td>Vanilla</td><td>Energy Spectrum  $L ^ { 1 }$  Error</td><td> $9 . 6 3 0 \mathrm { e } { - 2 } \pm 1 3 . 0 6 \mathrm { e } { - 2 }$ </td></tr><tr><td>DiffusionPDE</td><td>Energy Spectrum  $L ^ { 1 }$  Error</td><td> $3 . 2 9 9 \mathrm { e } \mathrm { - } 1 \pm 2 . 7 2 2 \mathrm { e } \mathrm { - } 1$ </td></tr><tr><td>ECI</td><td>Energy Spectrum  $L ^ { 1 }$  Error</td><td> $\pm . 7 4 0 \mathrm { e } . 2 \pm 1 . 8 7 0 \mathrm { e } . 2$ </td></tr><tr><td>Ours (Pure EGT)</td><td>Energy Spectrum  $L ^ { 1 }$  Error</td><td> $9 . 8 4 0 \mathrm { e } { - 2 } \pm 1 3 . 0 9 \mathrm { e } { - 2 }$ </td></tr><tr><td>Ours (Hybrid EGT)</td><td>Energy Spectrum  $L ^ { 1 }$  Error</td><td> $\underline { { 8 . 5 7 0 \mathrm { e } \mathrm { - } 2 \pm 1 1 . 4 3 \mathrm { e } \mathrm { - } 2 } }$ </td></tr></table>

Discussion. Table 11 highlights an important distinction between local pointwise consistency and global statistical realism. Because the current EGT formulation explicitly minimizes the pointwise $L ^ { 2 }$ norm of the PDE, IC, and BC residuals, it substantially reduces local physical errors (and improves spectral consistency over DiffusionPDE on 2D Navier–Stokes) but does not directly optimize gradient distributions on 1D Burgers. Because EGT can accommodate any differentiable energy functional, augmenting $E ( u )$ with differentiable spectral or gradient-matching penalties represents a natural extension for jointly optimizing local and macroscopic realism.

## J.7 Statistical Significance and Seed-to-Seed Variability

To verify the statistical robustness of our evaluation, we first compare the empirical standard deviations computed across 100 and 1, 000 generated samples on the 1D Burgers and 2D Navier–Stokes benchmarks (White Noise, Full-PDE). As shown in Table 12, the standard deviations are virtually identical, confirming that 100 samples provide reliable variance estimates. We therefore evaluate seed-to-seed variability by generating 100 samples across 5 independent random seeds (Table 13, using the linear scheduler for EGT based on our scheduler ablation). EGT exhibits consistently low seed-to-seed variance across both systems, achieving a lower physical MAE (1.49e-3 vs. 1.81e-3) and more than 2× lower Sample MSE variance (1.42e-2 vs. 3.06e-2) than ECI on 2D Navier–Stokes.

Table 12: Variance convergence analysis comparing standard deviations across 100 and 1, 000 generated samples (White Noise, Full-PDE).
<table><tr><td>Model</td><td>Metric</td><td>Std NS (100)</td><td>Std NS (1000)</td><td>Std Burgers (100)</td><td>Std Burgers (1000)</td></tr><tr><td>Ours (EGT)</td><td>Sample MSE</td><td>2.292e-1</td><td>2.321e-1</td><td>9.750e-2</td><td>1.208e-1</td></tr><tr><td>Ours (EGT)</td><td>Physical MAE</td><td>3.000e-4</td><td>3.000e-4</td><td>5.000e-4</td><td>6.000e-4</td></tr><tr><td>ECI</td><td>Sample MSE</td><td>3.746e-1</td><td>4.020e-1</td><td>7.380e-2</td><td>9.500e-2</td></tr><tr><td>ECI</td><td>Physical MAE</td><td>6.000e-4</td><td>6.000e-4</td><td>1.000e-4</td><td>1.000e-4</td></tr><tr><td>DiffusionPDE</td><td>Sample MSE</td><td>2.448e-1</td><td>2.297e-1</td><td>9.700e-2</td><td>1.147e-1</td></tr><tr><td>DiffusionPDE</td><td>Physical MAE</td><td>6.000e-4</td><td>8.000e-4</td><td>4.000e-4</td><td>4.000e-4</td></tr><tr><td>Vanilla</td><td>Sample MSE</td><td>2.302e-1</td><td>2.339e-1</td><td>1.059e-1</td><td>1.329e-1</td></tr><tr><td>Vanilla</td><td>Physical MAE</td><td>1.000e-3</td><td>1.400e-3</td><td>2.100e-3</td><td>2.500e-3</td></tr></table>

Table 13: Seed-to-seed variability across 5 independent random seeds (100 samples per seed, White Noise, Full-PDE).
<table><tr><td>Model</td><td>Burgers (Sample MSE) ↓ Burgers (Phys MAE) ↓ NS (Sample MSE) ↓ NS (Phys MAE) ↓</td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td> $0 . 1 6 3 \pm 9 . 8 2 \mathrm { e } { - 3 }$ </td><td> $4 . 5 4 \mathrm { e } \mathrm { - } 3 \pm 2 . 1 8 \mathrm { e } \mathrm { - } 4$ </td><td> $0 . 5 4 1 \pm 1 . 4 3 \mathrm { e } { - 2 }$ </td><td> $6 . 7 8 \mathrm { e } { - 3 } \pm 9 . 7 6 \mathrm { e } { - 5 }$ </td></tr><tr><td>DiffusionPDE</td><td> $0 . 1 3 9 \pm 1 . 0 4 \mathrm { e } { - 2 }$ </td><td> $1 . 7 1 \mathrm { e } { - 3 } \pm 5 . 8 9 \mathrm { e } { - 5 }$ </td><td> $0 . 5 4 7 \pm 1 . 7 0 \mathrm { e } { - 2 }$ </td><td> $3 . 7 2 \mathrm { e } \mathrm { - } 3 \pm 7 . 4 0 \mathrm { e } \mathrm { - } 5$ </td></tr><tr><td>ECI</td><td> $\mathbf { 0 . 1 2 7 \pm 6 . 1 2 6 . 3 }$ </td><td> $\mathbf { 2 . 0 5 e { - 4 } } \pm \mathbf { 4 . 9 8 e { - 6 } }$ </td><td> $0 . 6 2 0 \pm 3 . 0 6 \mathrm { e } { - 2 }$ </td><td> $1 . 8 1 \mathrm { e } \mathrm { - } 3 \pm 4 . 5 2 \mathrm { e } \mathrm { - } 5$ </td></tr><tr><td>Ours (EGT)</td><td> $0 . 1 4 6 \pm 9 . 1 3 \mathrm { e } . 3$ </td><td> $\underline { { 6 . 0 4 \mathrm { e } } } . 4 \pm 1 . 2 0 \mathrm { e } . 5$ </td><td> $\mathbf { 0 . 5 2 8 \pm 1 . 4 2 e { - 2 } }$ </td><td> $\mathbf { 1 . 4 9 e { - 3 } \pm 2 . 5 0 e { - 5 } }$ </td></tr></table>

## K Ablation Studies

In this appendix, we provide an extensive sensitivity analysis of the core hyperparameters governing Continuous Dissipative Guidance. To evaluate the robustness of our framework across different dimensionalities, we conduct all ablation studies on the 1D Burgers’ (Full-PDE) and 2D Navier-Stokes (Full-PDE) datasets. These cases represent the challenging scenarios where the unguided model’s physics violation is most prominent. For consistency across all ablation trials, we utilize the standard White Noise prior $p _ { 0 }$ . All tests are evaluated on 256 samples.

## K.1 Sensitivity to Guidance Scale $\gamma$

The guidance scale γ controls the magnitude of the dissipative force injected into the generative vector field. Figure 7 illustrates the trade-off between physical adherence and distributional fidelity as a function of $\gamma$

Discussion. As shown in the dual-axis plots, increasing γ triggers a rapid dissipative behaviour of the physical residual. In 1D Burgers’, we observe that the physical MAE reaches a stable floor already at $\gamma \approx 2 5$ , with negligible impact on the Sample MSE. This confirms that our continuous guidance effectively steers trajectories toward the physical manifold without causing the “manifold collapse” commonly observed in rigid algebraic projection methods. For 2D Navier-Stokes, the energy landscape is significantly more complex; while EGT successfully navigates this space and improves physical fidelity up to $\gamma = 3 0$ , pushing the guidance scale beyond this threshold disrupts the generative trajectory and results in numerical instability. Importantly, selecting γ in a genuine zero-shot regime does not require access to ground-truth test solutions: since the physical energy E(u) depends exclusively on spatial differential operators evaluated on the generated samples, practitioners can simply perform a validation sweep on a small batch of initial noise, progressively increasing γ and selecting the largest value before numerical stiffness or gradient divergence (NaNs) occurs.

![](images/c1079bf28e121d44ea098c5e4980c1f7e797b2fe6de890ccc90526c624ff39fc.jpg)  
(a) Burgers’ Dual-Axis

![](images/b2650fee2a8c0510dec40ae647df5513ae78f669dc6ef885401678697143e0e1.jpg)  
(b) NS’ Dual-Axis  
Figure 7: Impact of the guidance scale γ. Both dual-axis plots simultaneously track the Physical Residual MAE (physics) and Sample MSE (fidelity) on 1D Burgers’ (a) and 2D Navier-Stokes (b).

## K.2 Number of Flow Matching Steps N

We investigate the performance and computational efficiency of our framework under different discretization budgets for the ODE solver, ranging from $N = 1 \dot { 0 } { } \mathrm { t o } N = 2 5 0$ generative steps (Figure 8).

![](images/6d9caaf4b7f736a3ee06cc357a22142372ac72a6fc687480e0e6b60532faef13.jpg)  
(a) Burgers’ Budget Ablation

![](images/229f55a77ed08d01b7883ebc5fed80e6bfd10f5b99e51a07373fa34c1c9ba5bc.jpg)  
(b) NS’ Budget Ablation  
Figure 8: Evaluation of ODE integration steps N. Both environments demonstrate that EGT stabilizes quickly. Note that for 2D Navier-Stokes (b), the y-axis for the Sample MSE is strictly logarithmic to accommodate the severe error variance at highly constrained integration budgets.

Discussion. Unlike standard diffusion or heavy explicit projection models that often require hundreds of steps for stability, our energy-guided FFM maintains rigorous physical consistency even at highly accelerated integration budgets $( N \approx 5 0 )$ . For instance, in the complex 2D Navier-Stokes setting (where the Sample MSE is plotted on a logarithmic y-axis to capture the extreme variance), lowering N below 25 leads to severe numerical divergence. However, at just $N = 5 0$ , the framework instantly stabilizes, achieving a Sample MSE perfectly aligned with the unguided baseline. While the physical residual further decreases with finer discretization, the relative gain plateaus around $N = 1 0 0$ . This scalability is a significant practical advantage for real-time physics simulators where sub-second inference is required.

## K.3 Effect of Guidance Scheduling

Finally, we compare different scheduling functions γ(t) for the temporal application of the dissipative force. We categorize these into a Constant baseline, “Late Guidance” schedules (Linear, Parabolic, Cosine), and a “Transient” schedule (Sine).

![](images/a6fa630b184aae791f7db9f7f25029b2404f7aa91bc5cb1f4b8ddd4eb21d7666.jpg)  
(a) Burgers’ Scheduler Ablation

![](images/6f38dd8dcfede931fe369f4efebf3075b5f0c83d4cbaf062f70e95e8b120ef6c.jpg)  
(b) NS’ Scheduler Ablation  
Figure 9: Impact of the temporal guidance scheduler γ(t). Both “Late Guidance” and “Transient” schedules systematically prevent trajectory collapse by allowing the base neural prior to establish macroscopic coherent structures before rigid physical constraints are fully applied.

Discussion. Our empirical analysis strongly validates the necessity of “Late Guidance” schedules (Linear, Parabolic, Cosine) and the “Transient” schedule (Sine). By allowing the unguided neural prior to freely establish the macroscopic structure of the field near pure noise (t ≈ 0) before smoothly ramping up the physical constraints, these dynamic schedules avoid interfering with the inherently high variance of the initial noise. Most notably, applying a Constant guidance schedule throughout the flow results in a complete divergence (0% Success Rate) on the 2D Navier-Stokes environment, and a noticeable degradation of Sample MSE in the Burgers’ equation. Consequently, smoothly scheduling the dissipative force acts as a necessary annealing process, leading to both optimal Sample MSE and lower final physical residuals.

## K.4 Pareto Frontier Analysis: Accuracy vs. Stability vs. Speed

To evaluate how EGT navigates the accuracy–stability–speed trade-off compared to projectionand gradient-based baselines, we compute the Pareto frontiers on the 1D Burgers and 1D Heat benchmarks (Full-PDE, N = 100) across guidance scales (γ) and schedulers. Figure 10 visualizes the four resulting trade-off curves, while Tables 14 and 15 report the exact numerical operating points.

Table 14: Guidance strength (γ) and scheduler Pareto ablation on the 1D Burgers system. EGT provides a controllable trade-off between physical accuracy and distributional fidelity while maintaining a constant sub-second runtime (0.23 s/sample).
<table><tr><td>Method (γ Ablation)</td><td>Speed (s) ↓</td><td>Phys MAE ↓</td><td>Sample MSE ↓</td><td>Method (Scheduler Ablation)</td><td>Speed (s) ↓</td><td>Phys MAE ↓</td><td>Sample MSE ↓</td></tr><tr><td>Vanilla</td><td>0.13</td><td>5.53e-3</td><td>2.17e-1</td><td>Vanilla</td><td>0.13</td><td>5.53e-3</td><td>2.17e-1</td></tr><tr><td>ECI</td><td>9.30</td><td>1.99e-4</td><td>1.66e-1</td><td>ECI</td><td>9.30</td><td>1.99e-4</td><td>1.66e-1</td></tr><tr><td>DiffusionPDE</td><td>0.63</td><td>1.27e-3</td><td>1.88e-1</td><td>DiffusionPDE</td><td>0.63</td><td>1.27e-3</td><td>1.88e-1</td></tr><tr><td>EGT (γ = 10)</td><td>0.23</td><td>3.47e-3</td><td>1.52e-1</td><td>EGT (Sine)</td><td>0.23</td><td>1.76e-3</td><td>1.44e-1</td></tr><tr><td>EGT (γ = 25)</td><td>0.23</td><td>1.54e-3</td><td>1.46e-1</td><td>EGT (Cosine)</td><td>0.23</td><td>6.11e-4</td><td>1.37e-1</td></tr><tr><td>EGT (γ = 50)</td><td>0.23</td><td>4.24e-4</td><td>1.41e-1</td><td>EGT (Linear)</td><td>0.23</td><td>6.21e-4</td><td>1.37e-1</td></tr><tr><td>EGT (γ = 75)</td><td>0.23</td><td>6.07e-4</td><td>1.39e-1</td><td>EGT (Constant)</td><td>0.23</td><td>8.56e-4</td><td>1.81e-1</td></tr><tr><td>EGT (γ = 100)</td><td>0.23</td><td>6.41e-4</td><td>1.38e-1</td><td>EGT (Parabolic)</td><td>0.23</td><td>6.07e-4</td><td>1.39e-1</td></tr><tr><td>EGT (γ = 150)</td><td>0.23</td><td>7.10e-4</td><td>1.37e-1</td><td></td><td></td><td></td><td></td></tr></table>

Discussion. The Pareto analysis highlights two key operational insights. First, the guidance scale γ smoothly controls the operating point of EGT in a problem-dependent manner: on Burgers, increasing γ continuously improves Sample MSE while the physical residual reaches its minimum around γ = 50; on Heat, both metrics monotonically improve across the evaluated range. Second, the

![](images/7f44c878bf0908604866b9450a10451479b5ac37955bdd46d84b8d8898dc0a84.jpg)  
(a) Burgers 1D (γ Ablation)

![](images/06d9f33b81b322d07e62b809148902cca43f4a4174835f545909b7171105ed0b.jpg)  
(b) Burgers 1D (Scheduler Ablation)

![](images/b1aa99e94cbe9ce21abb35924e6d624bc86a75e9e53bdce5e9601a2a77ec301a.jpg)  
(c) Heat 1D (γ Ablation)

![](images/dde3117f07f476d53bf741f0b5cc398ce4401a9f9dad8f6110e57397e997a64e.jpg)  
(d) Heat 1D (Scheduler Ablation)

Figure 10: Pareto frontiers of distributional stability (Sample MSE) versus physical accuracy (Physical MAE), with inference speed (sec/sample) encoded via the colormap. EGT consistently dominates Vanilla and DiffusionPDE while operating 40–80× faster than ECI.

Table 15: Guidance strength (γ) and scheduler Pareto ablation on the 1D Heat system.
<table><tr><td>Method (γ Ablation)</td><td>Speed (s) ↓</td><td>Phys MAE ↓</td><td>Sample MSE ↓</td><td>Method (Scheduler Ablation)</td><td>Speed (s) ↓</td><td>Phys MAE ↓</td><td>Sample MSE ↓</td></tr><tr><td>Vanilla</td><td>0.13</td><td>7.38e-3</td><td>1.12e-1</td><td>Vanilla</td><td>0.13</td><td>7.38e-3</td><td>1.12e-1</td></tr><tr><td>ECI</td><td>8.80</td><td>1.95e-3</td><td>2.80e-2</td><td>ECI</td><td>8.80</td><td>1.95e-3</td><td>2.80e-2</td></tr><tr><td>DiffusionPDE</td><td>0.63</td><td>6.90e-1</td><td>9.40e+1</td><td>DiffusionPDE</td><td>0.63</td><td>6.90e-1</td><td>9.40e+1</td></tr><tr><td> $\mathrm { E G T } \left( \gamma = 1 0 \right)$ </td><td>0.26</td><td>4.67e-3</td><td>6.56e-2</td><td>EGT (Sine)</td><td>0.27</td><td>3.91e-3</td><td>4.82e-2</td></tr><tr><td> $\mathrm { E G T } \left( \gamma = 2 5 \right)$ </td><td>0.26</td><td>3.88e-3</td><td>4.98e-2</td><td>EGT (Cosine)</td><td>0.27</td><td>2.30e-2</td><td>6.14e-2</td></tr><tr><td> $\mathrm { E G T } \left( \gamma = 5 0 \right)$ </td><td>0.26</td><td>3.56e-3</td><td>4.25e-2</td><td>EGT (Linear)</td><td>0.27</td><td>2.21e-2</td><td>5.78e-2</td></tr><tr><td> $\mathrm { E G T } \left(  \gamma = 7 5 \right)$ </td><td>0.26</td><td>3.41e-3</td><td>3.95e-2</td><td>EGT (Constant)</td><td>0.27</td><td>2.17e-2</td><td>3.40e-2</td></tr><tr><td> $\mathrm { E G T } \left( \gamma = 1 0 0 \right)$ </td><td>0.26</td><td>3.31e-3</td><td>3.76e-2</td><td>EGT (Parabolic)</td><td>0.27</td><td>2.14e-2</td><td>6.69e-2</td></tr><tr><td> $\mathrm { E G T } \left( \gamma = 1 5 0 \right)$ </td><td>0.26</td><td>3.19e-3</td><td>3.52e-2</td><td></td><td></td><td></td><td></td></tr></table>

optimal temporal schedule is problem-dependent: cosine, linear, and parabolic schedules yield the strongest overall balance on Burgers, whereas the transient sine schedule achieves by far the lowest physical residual on Heat. Across all configurations, EGT reduces inference cost by one to two orders of magnitude compared to explicit projections (ECI) while avoiding any computational penalty when tuning hyperparameters.

## K.5 Contribution of Continuous Guidance vs. Post-Hoc Refinement

To isolate the distinct contribution of our continuous transport mechanism (Pure EGT) from the optional terminal energy descent (Hybrid EGT), we apply the exact same post-hoc optimization procedure (identical learning rate and number of refinement steps) to the terminal states generated by Vanilla FFM, DiffusionPDE, and ECI under the White Noise Full-PDE setting $( \gamma = 7 5$ for Burgers, $\gamma = 2 5$ for Navier–Stokes).

<table><tr><td>Model</td><td>Burgers (Sample MSE) ↓ Burgers (Phys MAE) ↓ NS (Sample MSE) ↓ NS (Phys MAE) ↓</td><td></td><td></td><td></td></tr><tr><td>Vanilla (Pure FFM)</td><td> $2 . 0 4 \mathrm { e } . 1 \pm 1 . 3 3 \mathrm { e } . 1$ </td><td> $5 . 1 0 \mathrm { e } . 3 \pm 2 . 5 0 \mathrm { e } . 3$ </td><td> $5 . 6 0 \mathrm { e } . 1 \pm 2 . 3 4 \mathrm { e } . 1$ </td><td> $7 . 5 0 \mathrm { e } { - 3 } \pm 1 . 4 0 \mathrm { e } { - 3 }$ </td></tr><tr><td>Vanilla (Refined)</td><td> $2 . 0 1 \mathrm { e } . 1 \pm 1 . 3 1 \mathrm { e } . 1$ </td><td> $1 . 5 4 \mathrm { e } \mathrm { - } 4 \pm 1 . 0 0 \mathrm { e } \mathrm { - } 4$ </td><td> $5 . 4 0 \mathrm { e } \mathrm { - } 1 \pm 2 . 3 4 \mathrm { e } \mathrm { - } 1$ </td><td> $3 . 9 6 \mathrm { e } { - 3 } \pm 1 . 4 2 \mathrm { e } { - 3 }$ </td></tr><tr><td>DiffusionPDE (Refined)</td><td> $1 . 5 3 \mathrm { e } \mathrm { - } 1 \pm 1 . 1 5 \mathrm { e } \mathrm { - } 1$ </td><td> $1 . 1 6 \mathrm { e } { - 4 } \pm 2 . 2 2 \mathrm { e } { - 5 }$ </td><td> $5 . 4 1 \mathrm { e } . 1 \pm 2 . 3 0 \mathrm { e } . 1$ </td><td> $4 . 4 2 \mathrm { e } . 4 \pm 5 . 4 9 \mathrm { e } . 4$ </td></tr><tr><td>ECI (Refined)</td><td> $1 . 1 6 \mathrm { e } - 1 \pm 9 . 5 0 \mathrm { e } - 2$ </td><td> $1 . 1 8 \mathrm { e } { - 4 } \pm 1 . 8 0 \mathrm { e } { - 5 }$ </td><td> $6 . 2 0 \mathrm { { e } - 1 } \pm 4 . 0 2 \mathrm { { e } - 1 }$ </td><td> $3 . 1 6 \mathrm { e } { - 5 } \pm 1 . 1 7 \mathrm { e } { - 5 }$ </td></tr><tr><td>Ours (Pure EGT)</td><td> $1 . 8 9 \mathrm { e } \mathrm { - } 1 \pm 1 . 2 1 \mathrm { e } \mathrm { - } 1$ </td><td> $1 . 9 0 \mathrm { e } { - 3 } \pm 6 . 0 0 \mathrm { e } { - 4 }$ </td><td> $5 . 4 7 \mathrm { e } . 1 \pm 2 . 3 2 \mathrm { e } . 1$ </td><td> $1 . 6 0 \mathrm { e } { - 3 } \pm 3 . 0 0 \mathrm { e } { - 4 }$ </td></tr><tr><td>Ours (Hybrid EGT)</td><td> $1 . 8 9 \mathrm { e } \mathrm { - } 1 \pm 1 . 1 8 \mathrm { e } \mathrm { - } 1$ </td><td> $1 . 5 0 \mathrm { e } { - 4 } \pm 2 . 9 0 \mathrm { e } { - 5 }$ </td><td> $5 . 4 7 \mathrm { e } . 1 \pm 2 . 3 2 \mathrm { e } . 1$ </td><td> $2 . 9 0 \mathrm { e } { - 5 } \pm 7 . 0 0 \mathrm { e } { - 6 }$ </td></tr></table>

Table 16: Ablation study applying identical post-hoc terminal refinement across baselines on 1D Burgers and 2D Navier–Stokes (White Noise, Full-PDE).

Discussion. Table 16 demonstrates that post-hoc refinement alone cannot replace continuous guidance during the flow. While terminal optimization reduces residuals across all methods on 1D Burgers, the non-linear 2D Navier–Stokes system exposes a clear gap: refining unconditional samples (Vanilla Refined) plateaus at a physical MAE of 3.96e-3, whereas Pure EGT already achieves 1.60e-3 before any refinement. Applying the identical terminal optimization to EGT (Hybrid EGT) drives the error down to 2.90e-5. Although ECI + Refinement reaches a comparable residual (3.16e-5), it relies on repeated projections that are nearly 5× slower and degrade Sample MSE (0.620 vs. 0.547).

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction accurately summarize the paper’s primary contributions: the Energy-Guided Transport framework, its theoretical foundations, and its empirical advantages over projection-based baselines.

Guidelines:

• The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.

• The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.

• The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.

• It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: Limitations regarding the sensitivity to the guidance scale γ in highdimensional settings and the reliance on discrete stencils for energy evaluation are discussed in the Conclusion and Appendix J.

Guidelines:

• The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.

• The authors are encouraged to create a separate “Limitations” section in their paper.

• The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.

• The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.

• The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.

• The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.

• If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.

• While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: All mathematical assumptions required for the derivation of the dissipative force and the stability analysis are explicitly stated in Appendix A.

Guidelines:

• The answer [N/A] means that the paper does not include theoretical results.

• All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.

• All assumptions should be clearly stated or referenced in the statement of any theorems.

• The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.

• Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.

• Theorems and Lemmas that the proof relies upon should be properly referenced.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: We fully disclose all necessary information to reproduce our experiments. The model architectures, continuous guidance hyperparameters, ODE integration budgets, and evaluation metrics are meticulously detailed in Appendices E, G, and J. Additionally, we provide a GitHub repository containing the complete source code to seamlessly reproduce all baseline and proposed framework results.

## Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.

• If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.

• Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.

• While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

(a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.

(b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.

(c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).

(d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: We provide a link to an GitHub repository containing the implementation, dataset generation scripts, and instructions for reproduction in the introduction.

Guidelines:

• The answer [N/A] means that paper does not include experiments requiring code.

• Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.

• While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).

• The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.

• The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.

• The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.

• At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).

• Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: Detailed training regimes, FNO architectures, and optimization hyperparameters are documented in Appendices E and G.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.

• The full details can be provided either with the code, in appendix, or as supplemental material.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: We report Sample MSE, MMSE, and SMSE across all benchmarks, and provide a dedicated statistical significance and multi-seed variability analysis (across 5 independent random seeds) in Appendix J.7.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.

• The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

• The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)

• The assumptions made should be given (e.g., Normally distributed errors).

• It should be clear whether the error bar is the standard deviation or the standard error of the mean.

• It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.

• For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).

• If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes]

Justification: Hardware specifications (NVIDIA RTX 8000, A6000, L40) and average inference times per sample are provided in Appendices E and J.

Guidelines:

• The answer [N/A] means that the paper does not include experiments.

• The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.

• The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.

• The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: This research focuses on fundamental computational physics and does not have foreseeable direct negative societal impacts.

Guidelines:

• The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.

• If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.

• The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

## Answer: [N/A]

Justification: Our study uses synthetic physical datasets (Heat, Burgers, Navier-Stokes) and does not involve human subjects or data derived from them.

Guidelines:

• The answer [N/A] means that there is no societal impact of the work performed.

• If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

• Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.

• The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.

• The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.

• If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: This research focuses on solving standard continuous physical systems (e.g., Navier-Stokes, Burgers’ equations) using synthetic data, which poses no high risks for societal misuse or dual-use.

Guidelines:

• The answer [N/A] means that the paper poses no such risks.

• Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.

• Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.

• We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: All existing assets (e.g., baseline model architectures, PDE simulation environments) used or adapted in our experiments are properly cited in the main text and Appendices, respecting their original open-source licenses.

Guidelines:

• The answer [N/A] means that the paper does not use existing assets.

• The authors should cite the original paper that produced the code package or dataset.

• The authors should state which version of the asset is used and, if possible, include a URL.

• The name of the license (e.g., CC-BY 4.0) should be included for each asset.

• For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

• If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.

• For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.

• If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We introduce a new codebase for the Energy-Guided Transport framework. It is thoroughly documented and provided via an GitHub repository linked in the submission, complete with instructions for data generation and reproducibility.

Guidelines:

• The answer [N/A] means that the paper does not release new assets.

• Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.

• The paper should discuss whether and how consent was obtained from people whose asset is used.

• At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: This research does not involve crowdsourcing or human subjects.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.

• According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

## Answer: [N/A]

Justification: This research does not involve human subjects, making IRB approval nonapplicable.

Guidelines:

• The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.

• Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

• We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.

• For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [N/A]

Justification: LLMs were used for writing assistance, structural editing, and formatting purposes.

## Guidelines:

• The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.

• Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.