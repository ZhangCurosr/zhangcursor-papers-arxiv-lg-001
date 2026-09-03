# Learning the Constitutive Behavior of Materials via Neural Operators and Causal Attention: Case Studies in Plasticity and Damage

Rishabh Arora<sup>1,∗</sup>, Lisa Scheunemann<sup>1</sup>, Tim Brepols<sup>1</sup>, Shahed Rezaei<sup>2,∗</sup>

<sup>1</sup>Institute of Applied Mechanics,

RWTH Aachen University, Mies-van-der-Rohe-Str. 1, D-52074 Aachen, Germany

<sup>2</sup>ACCESS e.V., Intzestr. 5, D-52072 Aachen, Germany

\* corresponding authors: rishabh.arora@ifam.rwth-aachen.de, s.rezaei@access-technology.de

## Abstract

Classical constitutive modeling of path-dependent inelastic materials relies on internal state variables whose evolution equations must be postulated based on domain knowledge and calibrated against experimental data. However, in many practical settings, the relevant internal variables are typically not measurable in experiments, and the constitutive response must be inferred entirely from measured strain-stress data without any prior knowledge of the material’s internal state. We propose a data-driven constitutive modeling framework based on the concept of a material operator, which treats a deforming material as a functional mapping from its entire strain history to the corresponding stress response. In contrast to traditional autoregressive or recurrent formulations, the model is trained directly on full loading paths as function-to-function mappings, predicting complete stress trajectories in a single parallel forward pass. Temporal path dependence is enforced through a causally masked attention mechanism embedded within the operator, which restricts the model’s attention to past material states while preserving computational parallelizability. Spectral convolutions provide discretization-invariant representations in the frequency domain, while causal attention captures highly adaptive, non-local history dependence. Furthermore, sinusoidal activation functions are used to resolve the strong nonlinear transitions inherent in inelastic regimes. The framework is evaluated across multidimensional, rate-independent material models exhibiting complex phenomena, with an emphasis on nonlinear plasticity and ductile damage accumulation. The results demonstrate accurate and robust predictions of irreversible deformation mechanisms while simultaneously achieving resolution invariance and excellent parallel eficiency.

Keywords: Constitutive relations, Nonlinear material behavior, Path-dependent material models, Neural operators, Plasticity, Damage, Data-driven modeling

## Nomenclature

<table><tr><td>Abbreviation</td><td>Meaning</td></tr><tr><td>ANN</td><td>Artificial Neural Network</td></tr><tr><td>CANN</td><td>Constitutive Artificial Neural Network</td></tr><tr><td>CP</td><td>Crystal Plasticity</td></tr><tr><td>FCNN</td><td>Fully Connected Neural Network</td></tr><tr><td>FEM</td><td>Finite Element Method</td></tr><tr><td>FFNN</td><td>Feed Forward Neural Network</td></tr><tr><td>FFT</td><td>Fast Fourier Transform</td></tr><tr><td>FNO</td><td>Fourier Neural Operator</td></tr><tr><td>GP</td><td>Gaussian-Process</td></tr><tr><td>GRU</td><td>Gated Recurrent Unit</td></tr><tr><td>ISV</td><td>Internal State Variable</td></tr><tr><td>KKT</td><td>Karush-Kuhn-Tucker</td></tr><tr><td>LN</td><td>Layer Normalization</td></tr><tr><td>LSTM</td><td>Long Short-Term Memory</td></tr><tr><td>MLP</td><td>Multilayer Perceptron</td></tr><tr><td>MSE</td><td>Mean Squared Error</td></tr><tr><td>PDE</td><td>Partial Differential Equation</td></tr><tr><td>PINN</td><td>Physics-Informed Neural Network</td></tr><tr><td>RNN</td><td>Recurrent Neural Network</td></tr><tr><td>RVE</td><td>Representative Volume Element</td></tr><tr><td>TANN</td><td>Thermodynamics-based Artificial Neural Network</td></tr><tr><td>SIREN</td><td>Sinusoidal Representation Network</td></tr><tr><td>TPE</td><td>Tree-structured Parzen Estimator</td></tr></table>

## 1. Introduction

Constitutive models lie at the heart of every computational solid mechanics simulation. They define the relationship between deformation and stress that is evaluated at every integration point during every time step, and the fidelity of a finite element analysis is therefore strongly dependent on the fidelity of the constitutive law it employs. For path-dependent inelastic materials that exhibit elastoplasticity, damage, and coupled plastic–damage behavior, formulating such a law remains a central challenge in computational solid mechanics. The defining feature of these materials is memory, which means that the stress at any instant depends not only on the current strain but also on the entire prior deformation history. This history dependence cannot be captured by a single algebraic stress–strain relationship and is usually encoded through internal state variables whose evolution traces the irreversible processes within the material [1, 2, 3]. Classical phenomenological models describe such behavior through these internal state variables, governed by evolution equations together with formulation of suitable yield and damage surfaces [4]. This framework is powerful and physically transparent, but constructing a specific model is demanding. The task becomes considerably harder when several internal mechanisms are active simultaneously. Plastic hardening, damage growth, and their mutual coupling interact in ways that are dificult even for an experienced modeler to anticipate. Damage can, for example, degrade the efective stifness that drives plastic flow, while plastic deformation in turn drives the energy release that fuels damage. The two mechanisms may evolve together under loading paths involving arbitrary sequences of loading, unloading, and reversal [5].

These dificulties have motivated data-driven constitutive modeling, which aims to learn the stress–strain response directly from experimental or simulation data without prescribing the functional form of the constitutive law in advance. The use of neural networks to represent material behavior dates back to the pioneering work of Ghaboussi et al. [6, 7], who first proposed learning constitutive relations directly from data with multilayer perceptron (MLP), exploiting the universal approximation property of feedforward networks [8]. This idea was quickly taken up across a range of material classes and, in the years since, has grown into a substantial and internationally distributed body of work. Furukawa and Yagawa [9] developed an implicit neural viscoplastic model in which the network predicts the rates of the viscoplastic strain and the internal variables, extending the approach to rate-dependent behavior. Fernández et al. [10] applied artificial neural networks as a direct data-driven surrogate for grain-boundary interface mechanics, learning the constitutive behavior of grain boundaries from data rather than deriving it from a prescribed physical model. Zhang and Mohr [11] demonstrated the capability of neural networks to efectively capture path-dependent material behavior by modeling von Mises plasticity with isotropic hardening. To capture loading history without an explicit window, Huang et al. [12] developed a machine learning-driven material modeling framework in which plasticity is represented through a Proper Orthogonal Decomposition, with the accumulated absolute strain serving as the history variable driving the reduced-order plastic response. Fernández et al. [13] combined material theory with data-driven calibration to construct anisotropic hyperelastic constitutive models for finite deformations, applying the approach to cubic lattice metamaterials. Mianroodi et al. [14] trained a convolutional network to map an image of a nanoporous microstructure directly to its elasticity tensor computed via molecular statics.

Utilizing convolutional neural networks, Frankel et al. [15] successfully predicted the spatiotemporal evolution of the stress field based on the initial microstructural features and applied external loading conditions. To accelerate computationally expensive crystal plasticity finite element simulations, Ali et al. [16] developed Feed Forward Neural Network (FFNN)-based surrogate models focused on capturing the homogenized response of the Representative Volume Element (RVE). Further work in this field is reviewed in detail in [17, 18, 19, 20].

These contributions established the central premise of the field, that a suficiently expressive network can encode a stress–strain relationship without a prescribed functional form, and formed the basis for much subsequent work. Despite their influence, these models shared several limitations that have shaped the research directions since. Their reliance on a fixed window of preceding time steps ties the learned response to the specific temporal sampling of the training data, which leads to accuracy degradation when the model is used at a diferent time step size.

A fundamentally diferent strategy is the model-free data-driven computing paradigm, which dispenses with a constitutive model altogether rather than learning one via a neural network. Kirchdoerfer and Ortiz [21] and Eggersmann et al. [22] introduced this approach for linear elastic and inelastic material, respectively. The study examined materials with memory (using the full deformation history), diferential materials (using only a short recent history), and history variables (using ad-hoc auxiliary variables). Each representation involves a natural trade-of that continues to motivate further development. The full-history representation ofers the most complete description of memory but increases the dataset’s dimensionality, favoring short-tomoderate horizon problems, while the history-variable representation is more scalable but leaves the question open of how such variables might be systematically selected or optimized.

A substantial body of work has since sought to restore physical consistency for inelastic materials by embedding known structure directly into the learned model. One direction enforces the laws of thermodynamics by construction (TANN by Masi et al. [23, 24]), so that the predicted response is admissible by design rather than only during training. Another line of work by Vlassis and Sun [25] instead builds thermodynamic consistency into a thermodynamics-informed neural network. This approach uses deep network predictions to evolve interpretable components like the yield surface and plastic flow rule, which depend on internal variables. To solve material-specific diferential equations, Haghighat et al. [26] detailed an alternative methodology that uses Physics-Informed Neural Network (PINN) [27] as its foundational framework. By training a neural network to predict the Cholesky factor of the tangent stifness matrix, Xu et al. [28] established an alternative method for stress prediction. Ultimately, this construction improves numerical stability in finite element simulations by weakly enforcing strain energy convexity. A data-driven, multiscale approach utilizing physics-constrained neural networks was introduced by Kalina et al. [29] to model finite strain hyperelasticity. Related research by Klein et al. [30] also explores the development of constitutive models using polyconvex neural networks. Rezaei et al. [31] introduced COMM-PINN, a physics-informed neural network that instantly resolves nonlinear, path-dependent constitutive equations of materials and satisfies thermodynamic constraints during the training time without relying on initial training data. The work by Rosenkranz et al. [32] presents a physics-augmented neural network model for nonlinear viscoelastic materials that is thermodynamically consistent by construction (built on free energy and dissipation potentials via generalized standard materials). Aldakheel et al. [33] embedded the governing equations of brittle and ductile fracture in elastic-viscoplastic materials directly into a feedforward network’s architecture, enforcing thermodynamic consistency by construction. As’ad and Farhat [34] proposed a mechanics-informed training procedure that constrains the learned model to satisfy dynamic and material stability, while Roy et al. [35] embedded the governing equations of von Mises plasticity directly into a physics-informed network for plane-strain boundary value problems.

A second direction seeks not a black-box surrogate but an interpretable method. For example, sparse-regression approaches over a library of candidate terms discover the governing equations directly [36], an idea recently extended by pairing multiple sparse-regression algorithms with model-selection criteria to automate constitutive model discovery [37]. The Constitutive Artificial Neural Network (CANN) framework of Linka et al. [38] and Linka and Kuhl [39] structures the network so that its architecture embeds kinematic and thermodynamic constraints and its weights carry direct physical meaning, turning model selection into automated model dis covery. These ideas have since been extended to inelastic behavior [40] and to anisotropic finitestrain viscoelasticity [41]. While these physics-structured models are accurate, interpretable, and data-eficient, they come with a trade-of. They require reintroducing the exact domain assumptions that purely data-driven methods set out to avoid, such as a specific free-energy form, a chosen set of invariants, or an assumed family of internal variables. When internal state variables are discovered automatically, through the encoder-decoder construction stated in the work of Masi and Stefanou [42], the user must still prescribe an underlying thermodynamic potential and dissipation format. These approaches are therefore most efective when the underlying physical structure is at least partially known. Modeling path-dependent behavior accurately when no such structure is prescribed, when the constitutive response, including its internal memory, must be inferred solely from measured strain-stress histories, remains a challenging and largely open problem, and is the setting addressed in this work.

Parallel to the development of physics-embedded approaches, a longstanding purely datadriven strategy for path-dependent materials has relied on sequence-modeling architectures to capture history dependence without explicit internal state variables. FFNN with explicit history windows [6, 43] maps a finite window of past strain–stress pairs to the next stress increment in an autoregressive fashion: at every time step, the network first predicts the new stress $\sigma _ { n + 1 }$ based on the most recent k strain–stress observations and the new strain increment, the prediction is then fed back into the history bufer, and the procedure is repeated. Recurrent neural network (RNN) and its gated variants avoid an explicit input window by encoding loading memory in an evolving hidden state $h _ { t }$ , which is updated sequentially one step at a time as the strain trajectory is consumed; the gating mechanism of Cho et al. [44] in particular lets the network learn what to retain and what to discard from this hidden state at every step, rather than carrying the full history forward unfiltered. Gorji et al. [45] established this paradigm for path-dependent plasticity by training gated recurrent unit networks to reproduce the response of an anisotropic plasticity model with homogeneous anisotropic hardening under complex, non-monotonic plane-stress loading. They also showed that the resulting surrogate generalizes to loading paths not seen during training. Chen [46] applied the same recurrent structure to viscoelastic materials, training on synthetic strain-stress sequences generated from a classical viscoelastic constitutive law and showing that the RNN captures the rate-dependent response without a prescribed Prony-series or relaxation-spectrum form, generalizing from simple linear and step strain inputs to unseen quadratic loading paths. Mozafar et al. [47] showed that recurrent neural networks, trained on representative volume element simulations, can predict path-dependent plastic response directly from strain histories without postulating an explicit yield surface. Bonatti et al. [48] scaled the recurrent approach to full crystal plasticity, training an RNN surrogate directly on fast-Fourier-transform-based crystal plasticity simulations to replace the costly micromechanical solve at each macroscopic integration point. Maia et al. [49] took a hybrid route, embedding a classical constitutive model inside the recurrent cell itself so that the network’s hidden state coincides with genuine physical internal variables, combining the flexibility of a data-driven surrogate with the interpretability of a mechanistic one. Both paradigms thus process the loading path temporally: the strain trajectory is presented as a stream of values, and path dependence is realized either through an explicit recursion over a bufer of recent observations (FFNN) or through an implicit recursion over a hidden state (RNN). Operator-based extensions such as the history-aware neural operator in recent studies by Guo et al. [50] retain the autoregressive structure of FFNN-style surrogates while replacing the underlying network with a Fourier Neural Operator (FNO), inheriting some discretization invariance from the spectral backbone but still operating in the next-step-prediction mode. A consequence of this temporal processing is that the model representation is tied to the temporal discretization of the training data: a network trained on loading paths sampled at one temporal resolution typically degrades when evaluated at coarser or finer resolutions, violating the principle that a rate-independent constitutive response should depend on the loading path itself rather than on how it is sampled. This loss of self-consistency has been documented for standard RNN architectures in the work by Bonatti and Mohr [51]. Achieving discretization invariance without sacrificing predictive accuracy on complex loading paths is the central methodological objective of the present work.

We propose a constitutive modeling framework that departs from both the autoregressive and the sequential-recurrent paradigms by treating the entire loading path as a single function input. Given a discretely sampled strain trajectory $\boldsymbol { \varepsilon } ~ = ~ \left( \varepsilon _ { 0 } , \varepsilon _ { 1 } , \ldots , \varepsilon _ { n } \right)$ , the proposed operator produces the full stress trajectory $\pmb { \sigma } = ( \pmb { \sigma } _ { 0 } , \pmb { \sigma } _ { 1 } , \dots , \pmb { \sigma } _ { n } )$ in a single parallel forward pass. The temporal index acts as a spatial-like coordinate over which the operator performs spectral convolution and self-attention. Within each block, a strict causal mask restricts every attention layer so that its output at a given temporal position depends on past and present strain values. The spectral convolution that follows operates globally over the temporal axis to provide resolution-invariant mixing of these causally-attended features. Figure 1 presents the three paradigms: autoregressive FFNN, sequential RNN-GRU (Gated Recurrent Unit), and the proposed sequence-to-sequence operator, highlighting how the input is presented and how path dependence is realized in each case during inference time.

![](images/ce56387b87d081ce4e3f43dd367e4d2b9f329d22caaa232f58a84f7069bba270.jpg)  
Figure 1: Three paradigms for data-driven path-dependent constitutive modeling at inference time (a) Autoregressive FFNN (b) Sequential RNN-GRU (c) Proposed material operator

Neural operators generalize neural networks to map between function spaces [52, 53]. For time-dependent PDEs, they have most often been applied autoregressively in time. The operator advances the full spatial field by one increment and is rolled forward step by step over the temporal horizon [54, 55, 52, 56]. Each such evaluation is a complete function-to-function map over space but not in time. We depart from this step-wise view. Rather than treating time as a stream advanced increment by increment, we treat the entire loading path as a single functionvalued input and learn a direct mapping to the corresponding stress trajectory. The FNO implements this through spectral convolution in the frequency domain, retaining a finite set of modes [52]. This formulation enables discretization-invariant behavior, since the learned filters are tied to physical wavenumbers instead of discrete time indices. As a result, the same model can be evaluated on loading paths sampled at diferent temporal resolutions without retraining. The main focus of our work is how it stands apart from earlier neural operators. In the recent history-aware neural operator for constitutive modeling [50], the system still moves forward step by step by predicting one increment from the previous one. This step-wise recursion makes those models sensitive to the temporal step size. The proposed material operator is instead applied once to the entire loading path, mapping the complete strain trajectory to the full stress trajectory in a single parallel pass without any rollout. This setup lets us compare, on a common footing, the three ways of representing path dependence that structure this paper as

shown in Figure 1.

To resolve the sharp features of inelastic response, the architecture combines the spectral structure of the FNO [52] with the sinusoidal representation networks (SIREN) of Sitzmann et al. [57], whose periodic activations of the form sin $( \omega _ { 0 } W x + b )$ are particularly well suited to representing the sharp transitions characteristic of yield onset, elastic-to-plastic switching, and damage initiation. Standard non-periodic activations instead exhibit a spectral bias toward smooth, low-frequency signals [58]. The choice of $\omega _ { 0 }$ controls the spanned frequency spectrum, enabling sharp transitions to be captured without an excessive number of layers or parameters [57]. Finally, the self-attention mechanism of Vaswani et al. [59] is adopted, with strict causal masking. This provides adaptive, content-dependent weighting of past values (i.e. strain) without compromising the directional flow of mechanical history.

We develop an FNO-SIREN architecture that integrates spectral operator learning with sinusoidal representation networks for data-driven constitutive modeling of rate-independent path-dependent materials, and we further enhance it with a self-attention mechanism for history weighting. We demonstrate that the proposed architecture achieves strong path-family generalization, in that a model trained exclusively on Gaussian-Process loading paths accurately predicts the stress response on unseen style distribution zig-zag and sinusoidal paths. We validate the framework on three benchmark problems of increasing complexity. These consist of a one-dimensional elastoplasticity material model with nonlinear isotropic hardening, a one-dimensional coupled plastic-damage material model, and a two-dimensional elastoplasticity material model.

## 2. Thermodynamics-based material modeling

For rate-independent inelastic solids under isothermal conditions, the constitutive response is conventionally formulated within the framework of irreversible thermodynamics with internal state variables [1, 2]. Let $\varepsilon ( t )$ denote the strain at time t and $\mathbf z ( t ) = ( \mathbf z _ { 1 } , \ldots , \mathbf z _ { \mathbf n } )$ a collection of Internal State Variables (ISVs) encoding the irreversible material memory representing plastic strain, hardening parameters, or damage variables. The Helmholtz free energy is postulated as $\boldsymbol { \varPsi } = \boldsymbol { \varPsi } ( \boldsymbol { \varepsilon } , \mathbf { z } )$ , from which the Cauchy stress and thermodynamic conjugate forces follow as

$$
{ \pmb { \sigma } } = \frac { \partial \psi } { \partial \epsilon } , \qquad { \bf Y } = - \frac { \partial \psi } { \partial { \bf z } } .\tag{1}
$$

The remaining part of the Clausius–Duhem inequality ${ \mathcal { D } } = \mathbf { Y } \cdot { \dot { \mathbf { z } } } \geq 0$ constrains the evolution of the ISVs, typically prescribed through a rate equation $\dot { { \mathbf z } } = { \mathbf f } ( { \mathbf Y } , \pmb { \sigma } , { \mathbf z } )$ together with admissibility conditions, for example, for rate-independent plasticity and damage. The resulting constitutive law ${ \pmb \sigma } ( t ) = { \pmb \sigma } ( \varepsilon ( t ) , { \bf z } ( t ) )$ is implicitly path-dependent. The stress at any instant depends on the entire prior strain history through the evolution of z.

In the remainder of this paper we evaluate the proposed data-driven framework on three benchmark constitutive models of increasing complexity. The first is a one-dimensional elastoplastic model with nonlinear isotropic hardening, which serves as the baseline scalar setting in which all three model architectures are compared. The second is a one-dimensional coupled damage-plasticity model, which couples plastic flow with progressive isotropic damage and tests whether the architectural advantages observed in pure plasticity carry over to a regime with two simultaneous irreversible processes. The third is a two-dimensional plane-strain elastoplastic model, which generalizes the first benchmark to a multi-component setting where the surrogate must simultaneously predict three stress components from three strain components and learn the coupling between normal and shear directions.

## 2.1. One-dimensional elastoplastic material model

We consider a one-dimensional elastoplastic model with nonlinear isotropic hardening of saturation type. The total strain is additively decomposed as $\varepsilon = \varepsilon ^ { e } + \varepsilon ^ { p }$ , with the stress given by $\sigma = E \varepsilon ^ { e }$ and the yield function

$$
\phi ( \sigma , \xi ^ { p } ) = | \sigma | - \big ( \sigma _ { y } + h _ { 1 } ( 1 - e ^ { - h _ { 2 } \xi ^ { p } } ) \big ) ,
$$

where $\xi ^ { p }$ is the accumulated plastic strain and the hardening term saturates exponentially with rate $h _ { 2 }$ and asymptotic amplitude $h _ { 1 }$ .The associated flow rule is given by $\dot { \varepsilon } ^ { p } = \dot { \gamma } \mathrm { s i g n } ( \sigma )$ and $\dot { \xi } ^ { p } = \dot { \gamma } ;$ where the plastic multiplier $\dot { \gamma }$ satisfies the Karush–Kuhn–Tucker (KKT) conditions $\dot { \gamma } \ge 0 , \phi \le 0 , \dot { \gamma } \phi = 0$ . The constants used throughout this section are $E = 3 . 0 \ \mathrm { M P a } , \sigma _ { y } = 0 . 6$ MPa, $h _ { 1 } = 0 . 4 ~ \mathrm { M P a }$ , and $h _ { 2 } ~ = ~ 1 0 . 0$ . Reference stress histories are generated by a returnmapping algorithm [2]. The algorithm is stated in the work of Rezaei et al. [31].

## 2.2. One-dimensional coupled damage–plasticity material model

To probe a setting in which two irreversible mechanisms evolve simultaneously, we adopt a one-dimensional coupled isotropic two-surface damage–plasticity model defined in the work of Brepols et al. [60, 61]. We consider here a purely local formulation of damage, i.e., without a gradient-extension. The model introduces a scalar isotropic damage variable $D \in [ 0 , 1 )$ that enters the efective material stifness through the degradation function

$$
f ( D ) = 1 - D ,
$$

such that the stifness is ultimately degraded by $f ( D ) ^ { 2 }$ , and the Cauchy stress is expressed in terms of the elastic strain $\varepsilon ^ { e } = \varepsilon - \varepsilon ^ { p }$ as

$$
\sigma = f ( D ) ^ { 2 } E \varepsilon ^ { e } .
$$

The squared damage factor reflects the dual reduction of efective load-bearing area in both the stress–strain relation and the conjugate thermodynamic forces, consistent with the energyequivalence interpretation in continuum damage mechanics. In addition to $\varepsilon ^ { p }$ and $D ,$ which are themselves internal variables governing plastic and damage evolution, two further internal state variables track the hardening response of the two irreversible mechanisms: $\xi ^ { p }$ for isotropic plastic hardening, evolving in the efective configuration, and $\xi ^ { d }$ for damage hardening.

The plastic yield function is

$$
F _ { p } ( \sigma , \xi ^ { p } , D ) = | \sigma | - f ( D ) \sigma _ { 0 } - f ( D ) ^ { 2 } h _ { p } \xi ^ { p } ,
$$

where $\sigma _ { 0 }$ is the initial yield stress and $h _ { p }$ represents the linear plastic hardening modulus. The associated flow rule is $\dot { \varepsilon } ^ { p } = \dot { \gamma } _ { p } \mathrm { s i g n } ( \sigma )$ , and the plastic hardening variable evolves as $\dot { \xi } ^ { p } = \dot { \gamma } _ { p } / f ( D )$ ， so that hardening progresses faster as the material softens through damage.

Damage growth is governed by the elastic energy release rate

$$
Y ( \varepsilon ^ { e } , \xi ^ { p } , D ) = f ( D ) \big ( E ( \varepsilon ^ { e } ) ^ { 2 } + h _ { p } ( \xi ^ { p } ) ^ { 2 } \big ) ,
$$

together with the damage activation criterion

$$
F _ { d } ( Y , \xi ^ { d } ) = Y - \big ( Y _ { 0 } + r _ { d } \xi ^ { d } \big ) ,
$$

where $Y _ { 0 }$ is the threshold for damage initiation and $r _ { d }$ is the linear damage hardening modulus. The damage variable evolves associatively as $\dot { D } = \dot { \gamma } _ { d } .$ , while the damage hardening variable follows the non-associative rule $\begin{array} { r } { \dot { \xi } ^ { d } = \dot { \gamma } _ { d } \big ( 1 + \frac { s _ { d } } { r _ { d } } q _ { d } \big ) } \end{array}$ with $q _ { d } = r _ { d } \xi ^ { d }$ . Upon backward-Euler time discretization, these evolution equations are integrated over each time increment $\varDelta t$ in terms of the incremental multiplier $\varDelta \gamma _ { d } = \dot { \gamma } \varDelta t \geq 0$ , yielding the closed-form algorithmic update

$$
\varDelta D = \varDelta \gamma _ { d } , \qquad \varDelta \xi ^ { d } = \frac { \varDelta \gamma _ { d } } { 1 + s _ { d } \varDelta \gamma _ { d } } ,
$$

which is the form used in the local Newton solve described below, with $s _ { d }$ saturating the damage hardening rate at large increments. Both plastic and damage multipliers satisfy independent Karush–Kuhn–Tucker conditions, stated here in discrete form as $F _ { p } \le 0 , \ : \varDelta \gamma _ { p } \ge 0 , \ : \varDelta \gamma _ { p } F _ { p } = 0$ and $F _ { d } \le 0 , \varDelta \gamma _ { d } \ge 0 , \varDelta \gamma _ { d } F _ { d } = 0$

The constitutive response is computed by a stress update that branches into four cases depending on the activity of the plastic and damage surfaces: a purely elastic case, an elastoplasticonly case, an elastic-damage-only case, and a coupled case in which both $F _ { p } = 0$ and $F _ { d } = 0$ The coupled case requires the simultaneous solution of two consistency equations in the two multipliers $\varDelta \gamma _ { p }$ and $\varDelta \gamma _ { d }$ , which is performed by a Newton iteration on the $2 \times 2$ residual system; further details can be found in [60, 61]. Parameter values used in the present study are $E = 3 . 0$ MPa, σ<sub>0</sub> = 0.6 MPa, h<sub>p</sub> = 0.4 MPa, Y<sub>0</sub> = 0.15 MPa, r<sub>d</sub> = 0.5 MPa, $s _ { d } = 0 . 0 5$

## 2.3. Two-dimensional plane-strain elastoplastic material model

We finally extend the evaluation to a two-dimensional plane-strain setting, where the surrogate must simultaneously predict three stress components from three strain components throughout the loading history. For the multi-axial extension, we adopt classical $J _ { 2 }$ (von Mises) plasticity, in which yielding is governed by the second invariant of the deviatoric stress tensor s,

$$
J _ { 2 } = { \frac { 1 } { 2 } } { \bf s } : { \bf s } ,
$$

rather than by the normal and shear stress components individually. The associated von Mises equivalent stress is $q \ = \ { \sqrt { 3 J _ { 2 } } }$ . The added dificulty is structural rather than just dimensional: the trained model must learn the coupling between normal and shear components through the deviatoric structure of the $J _ { 2 }$ flow rule.

We consider rate-independent $J _ { 2 }$ plasticity with nonlinear isotropic hardening of the same saturation form used in Section 2.1, generalized to two-dimensional plane strain. The total strain tensor is decomposed additively into elastic and plastic parts, $\varepsilon = \varepsilon ^ { e } + \varepsilon ^ { p }$ , and the Cauchy stress follows Hooke’s law:

$$
{ \pmb \sigma } = \lambda \mathrm { t r } ( { \pmb \varepsilon } ^ { e } ) { \bf I } + 2 \mu { \pmb \varepsilon } ^ { e } ,
$$

with Lamé parameters $\lambda = E \nu / [ ( 1 + \nu ) ( 1 - 2 \nu ) ]$ and $\mu = { E } / [ 2 ( 1 + \nu ) ]$ . The out-of-plane normal strain is constrained by plane-strain kinematics, $\varepsilon _ { z z } ^ { e } = - ( \varepsilon _ { x x } ^ { p } + \varepsilon _ { y y } ^ { p } )$ , so the out-of-plane normal stress $\sigma _ { z z }$ is non-zero and contributes to the deviatoric response. The deviatoric stress is $\begin{array} { r } { { \bf s } = { \pmb \sigma } - \frac { 1 } { 3 } \mathrm { t r } ( { \pmb \sigma } ) { \bf I } } \end{array}$ , and the equivalent von Mises stress is

$$
q ( \mathbf { s } ) = { \sqrt { \frac { 3 } { 2 } \mathbf { s } : \mathbf { s } } } .
$$

Plastic yielding is governed by the $J _ { 2 }$ yield function

$$
\phi ( q , \xi ^ { p } ) = q - \big ( \sigma _ { y } + h _ { 1 } ( 1 - e ^ { - h _ { 2 } \xi ^ { p } } ) \big ) ,
$$

with $\xi ^ { p }$ being the accumulated plastic strain. The associated flow rule defines the plastic strain rate as $\dot { \varepsilon } ^ { p } = \dot { \gamma }$ n and the equivalent plastic strain rate as $\dot { \xi } ^ { p } = \dot { \gamma }$ . In this formulation, the flow direction is given by $\mathbf { n } = { \frac { 3 } { 2 } } \mathbf { s } / q$ , and the non-negative plastic multiplier $( \dot { \gamma } \geq 0 )$ enforces the Karush–Kuhn–Tucker conditions, namely $\phi \leq 0$ and ${ \dot { \gamma } } \phi = 0$ . The material constants are inherited from Section 2.1 $( E = 3 . 0$ MPa, $\sigma _ { y } = 0 . 6$ MPa, $h _ { 1 } = 0 . 4$ MPa, $h _ { 2 } = 1 0 . 0 )$ and the plane-strain setting is closed by $\nu = 0 . 3$ . Reference responses are computed by a return-mapping algorithm with a Newton iteration on the consistency residual, following the standard procedure (see also [2]).

The surrogate operates on three in-plane components: the strain history is represented as $\boldsymbol \varepsilon ( t ) = ( \varepsilon _ { x x } , \varepsilon _ { y y } , \varepsilon _ { x y } )$ and the stress history as $\pmb { \sigma } ( t ) = ( \sigma _ { x x } , \sigma _ { y y } , \sigma _ { x y } )$ , so the input and output of the operator are both functions taking values in $\mathbb { R } ^ { 3 }$ over the temporal axis.

## 3. Data-driven constitutive modeling of path-dependent materials

This section develops the proposed framework for data-driven constitutive modeling of pathdependent materials that are rate-independent. We review three paradigms, which are FFNN with explicit history windows, RNN-GRU, and, as the proposed alternative, a sequence-tosequence operator built on a FNO backbone with sinusoidal representation layers and causal self-attention.

The objective of data-driven constitutive modeling is to learn implicit history dependence directly from observed strain–stress data, without committing to a specific functional form. In continuous-time form, the target mapping is a causal operator

$$
\mathcal { G } : \pmb { \varepsilon } ( \cdot ) \big | _ { [ 0 , t ] } \ \longmapsto \ \pmb { \sigma } ( t ) ,\tag{2}
$$

which maps the entire strain history up to time t to the current stress, with the causality requirement that $\pmb { \sigma } ( t )$ should not depend on $\varepsilon ( \tau )$ for $\tau > t$ . In practice, measurements are available at discrete instants $t _ { 0 } < t _ { 1 } < . . . < t _ { n }$ , and the discrete counterpart maps the full sampled strain trajectory $\left( \varepsilon _ { 0 } , \varepsilon _ { 1 } , \ldots , \varepsilon _ { n } \right)$ to the full sampled stress trajectory $( \pmb { \sigma } _ { 0 } , \pmb { \sigma } _ { 1 } , \dots , \pmb { \sigma } _ { n } )$

The three neural-network paradigms reviewed and developed in the remainder of this section difer in how they approximate this causal operator. Each makes a diferent structural choice about (i) what is presented to the network as input, (ii) how path dependence is enforced, and (iii) how predictions are produced at inference. Figure 1 contrasts these choices schematically.

## 3.1. Sequence-to-sequence operator: a diferent way to present the loading path

The conventional approach to path-dependent surrogate modeling is to process the loading path (strain trajectory) as a stream of values, consumed one step at a time, and stress predictions are produced sequentially. The proposed framework departs from this convention. Instead of streaming the strain trajectory, we treat the entire discretely sampled loading path as a single function-valued input and learn an operator that produces the entire stress trajectory in one parallel forward pass:

$$
\pmb { \sigma } = \pmb { \mathcal { G } } ( \pmb { \varepsilon } ) , \qquad \pmb { \varepsilon } = ( \pmb { \varepsilon } _ { 0 } , \ldots , \pmb { \varepsilon } _ { n } ) , \quad \pmb { \sigma } = ( \pmb { \sigma } _ { 0 } , \ldots , \pmb { \sigma } _ { n } )\tag{3}
$$

The temporal index plays the role of a spatial-like coordinate over which the operator acts; the loading path is, in efect, lifted into a one-dimensional function-space representation. Path dependence is the requirement that $\sigma _ { i }$ depend only on $\varepsilon _ { 0 } , \ldots , \varepsilon _ { i } .$ . It is no longer realized through iteration over a bufer or through a hidden-state recurrence. The proposed architecture is discussed further.

## 3.1.1. Fourier Neural Operator backbone

In the proposed architecture, each hidden operator layer $\{ \mathcal { L } _ { \ell } \} _ { \ell = 1 } ^ { L }$ takes the form (Li et al. [52]):

$$
\begin{array} { r } { \mathscr { L } _ { \ell } \mathbf { v } _ { l - 1 } ( x ) = \rho \bigl ( W _ { \ell } \mathbf { v } _ { \ell - 1 } ( x ) + ( K _ { \ell } \mathbf { v } _ { \ell - 1 } ) ( x ) \bigr ) , } \end{array}\tag{4}
$$

where L is the total number of Fourier layers between the projection and lifting layers, $W _ { \ell }$ is a pointwise linear map, $\kappa _ { \ell }$ is the integral kernel operator, and $\rho$ is a nonlinear activation function applied pointwise (meaning it acts independently on each component of the vector). We implement $\displaystyle \kappa _ { \ell }$ as a global convolution in the frequency domain (Li et al. [52]):

$$
\begin{array} { r } { ( \mathcal { K } _ { \ell } \mathbf { v } _ { \ell - 1 } ) ( x ) = \mathcal { F } ^ { - 1 } \big [ { R } _ { \ell } \cdot \mathcal { F } [ \mathbf { v } _ { \ell - 1 } ] \big ] ( x ) , } \end{array}\tag{5}
$$

where $\mathcal { F }$ and ${ \mathcal { F } } ^ { - 1 }$ denote the (fast) Fourier transform and its inverse, and $R _ { \ell }$ is a learnable complex-valued spectral filter retaining only the lowest M Fourier modes. Because $R _ { \ell }$ indexes physical wavenumbers rather than discrete grid points, the same trained filter applies consistently across coarse and fine discretizations of the temporal axis, provided the spectral content of the signal is adequately resolved [53]. This is the discretization invariance we require of our constitutive framework.

For the present application, the input function is the strain trajectory ε, the domain $\mathcal { D }$ is the temporal index axis, and the output is the stress trajectory $\sigma .$ The FNO layer captures global temporal correlations within the loading path through its spectral convolution; the next two subsections explain our operator capacity to represent sharp transitions and to enforce causality.

## 3.1.2. Sinusoidal representation layers

To represent the sharp transitions characteristic of yield onset, elastic-to-plastic switching, and damage initiation, we employ sinusoidal representation layers (SIREN) [57], in which every fully connected layer uses a sine activation:

$$
\mathbf { h } _ { \ell } = \sin \bigl ( \omega _ { 0 } U _ { \ell } \mathbf { h } _ { \ell - 1 } + b _ { \ell } \bigr ) ,\tag{6}
$$

with $U _ { \ell }$ as the weight matrix, and $\omega _ { 0 }$ a frequency hyperparameter and the weights initialized to preserve the variance of activations and gradients across layers.

We employ SIREN layers in three roles in the proposed architecture: (i) as the lifting operator that maps the scalar strain input $\varepsilon ( t )$ into the wider latent feature space, (ii) as the activation function within the Fourier layers (FNO-SIREN) applied after combining the spectral and local linear transformations, and (iii) as the projection operator  that maps the latent features back to the stress output $\sigma ( t )$ . In these roles, SIREN acts pointwise along the temporal axis.

## 3.1.3. Causal self-attention and the FNO-SIREN block

While the spectral convolution captures global temporal correlations and the SIREN layers sharpen local features, both treat every position in the loading trajectory equally. In particular, the spectral convolution is non-causal. The Fourier transform couples each output position to both past and future strain values, so the predicted stress at a given instant is allowed to depend on strain values that occur later in the loading path. This violates a basic physical requirement of a path-dependent constitutive law, namely that the stress at any instant may depend only on the prior strain history and not on future deformation. Moreover, even among the admissible past positions, their relevance for predicting the current stress is highly non-uniform. Positions near loading reversals, yield transitions, or damage onset typically carry far more information than positions within purely elastic response. To address both points, we incorporate a multi-head self-attention module [59] with a strict causal mask, which adaptively reweights the temporal axis while preventing any position from attending future values, as shown in Figure 2 and 3.

For an input sequence $\mathbf { X } \in \mathbb { R } ^ { N \times d }$ , where N is the sequence length (the number of discretized time steps along the loading trajectory) and d is the feature dimension at that stage of the network, the attention block computes queries, keys, and values

$$
{ \bf Q } = { \bf X } W _ { Q } , \qquad { \bf K } = { \bf X } W _ { K } , \qquad { \bf V } = { \bf X } W _ { V } ,\tag{7}
$$

where $W _ { Q } , W _ { K } \in \mathbb { R } ^ { d \times d _ { k } }$ and $W _ { V } \in \mathbb { R } ^ { d \times d _ { v } }$ are learnable projection matrices, $d _ { k }$ is the dimension of the query and key vectors, and $d _ { v }$ is the dimension of the value vectors (and hence of each output row of the attention block). The block then returns

$$
\mathrm { A t t n } ( \mathbf { X } ) = \mathrm { s o f t m a x } \left( \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { k } } } + \mathbf { M } \right) \mathbf { V } ,\tag{8}
$$

where softmax denotes the function that maps a real-valued vector $\mathbf { z } \in \mathbb { R } ^ { n }$ to a probability distribution via softmax $( \mathbf { z } ) _ { j } = \exp ( z _ { j } ) / \sum _ { k = 1 } ^ { n } \exp ( z _ { k } )$ , and is applied row-wise here, turning each row of $\frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d _ { k } } } + \mathbf { M }$ into a probability distribution over time steps, so that each output position is a weighted average of the value vectors in V. The factor $\sqrt { d _ { k } }$ rescales the query-key dot products, which otherwise grow in magnitude with $d _ { k }$ and push the softmax into a regime with vanishingly small gradients. The causal mask $\mathbf { M } \in \mathbb { R } ^ { N \times N }$ has entries

$$
M _ { i j } = \left\{ { \begin{array} { l l } { 0 , } & { j \leq i , } \\ { - \infty , } & { j > i , } \end{array} } \right.\tag{9}
$$

![](images/cf73f51553aa438334525520d4c3f7db9e60aaa534df264c4bd987f8e27038f3.jpg)  
Figure 2: FNO-SIREN+Attention

The masked entries vanish under the softmax and the attention output at position i is a convex combination of value vectors at positions $j \le i$ only. Multiple heads operate in parallel and their outputs are concatenated through a linear projection. The mask is what enforces causality at the architectural level. The causal self-attention layer operates on hidden feature representations of the strain history, not on stress. It is combined with the pointwise SIREN layers (which act independently at each position) and with the spectral convolution, which is applied after the masked attention (Figure 2).

![](images/06a1f0d642a9d946e2dd2d733360936883e92ef534e602a4376743fc6649d03d.jpg)  
Figure 3: Causal Attention Mechanism

Each operator block in the proposed architecture combines causal attention, spectral convolution, and a SIREN feature transform with residual connections:

$$
\begin{array} { r } { \mathbf { v }  \mathbf { v } + \mathrm { A t t n } \big ( \mathrm { L N } ( \mathbf { v } ) \big ) , \qquad \mathbf { v }  \sin \big ( \mathcal { K } _ { \ell } \mathbf { v } + W _ { \ell } \mathbf { v } \big ) , } \end{array}\tag{10}
$$

where LN is layer normalization, $\displaystyle \kappa _ { \ell }$ is the spectral convolution , $W _ { \ell }$ is a pointwise SIREN linear map, and the sine activation provides the final nonlinearity of the block. Stacking L such blocks between a SIREN lifting operator  and a SIREN-plus-linear projection $\mathcal { Q }$ produces the full FNO-SIREN operator $\mathcal { G }$ . To reduce the risk of overfitting, meaning the model memorizing the training data instead of learning the underlying operator, we apply dropout during training. Dropout works by randomly setting a fraction of the intermediate values in the network to zero at each training step, with the specific values chosen anew every time. Dropout is applied inside each attention block and after each spectral activation for regularization.

A minor implementation detail worth noting is that strain-to-stress sequences are zeroanchored: the first entry of every training trajectory satisfies $\varepsilon _ { 0 } = \mathbf { 0 }$ and $\pmb { \sigma } _ { 0 } = \mathbf { 0 }$ , so the output of the network is post-processed by subtracting $\pmb { \sigma } _ { 0 }$ from every position. This ensures the zerostress boundary condition is satisfied exactly regardless of any residual bias in the projection head.

## 3.1.4. Training protocol

Loss: The network is trained with the mean squared error over the full predicted stress trajectory:

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { N _ { \mathrm { t r } } } \sum _ { j = 1 } ^ { N _ { \mathrm { t r } } } \frac { 1 } { N _ { j } } \sum _ { n = 0 } ^ { N _ { j } - 1 } \big ( \hat { \pmb { \sigma } } _ { n } ^ { ( j ) } - \pmb { \sigma } _ { n } ^ { ( j ) } \big ) ^ { 2 } ,\tag{11}
$$

where $N _ { \mathrm { t r } }$ denotes the number of training trajectories (paths) in the dataset, $N _ { j }$ is the number of time steps in the j-th trajectory, and $\hat { \pmb { \sigma } } ^ { ( j ) } = \mathcal { G } ( \pmb { \varepsilon } ^ { ( j ) } )$ is the prediction for the j-th training trajectory, obtained by mapping the entire strain trajectory $\varepsilon ^ { ( j ) }$ to the entire predicted stress trajectory $\hat { \pmb { \sigma } } ^ { ( j ) }$ in a single forward pass.

Note that no autoregressive unrolling is needed, i.e., the model does not generate the trajectory step-by-step by recursively feeding back its own previous predictions. Strain and stress components are standardized to zero mean and unit variance using the training-set statistics, and predictions are denormalized before evaluation.

Optimization: All models are trained with AdamW [62] with the learning rate, batch size, and weight decay determined by hyperparameter search. Early stopping with a patience parameter is used to avoid overfitting during training. The exact optimized hyperparameters used in our final training phase are reported. Table 1 reports the tuned configurations for the two one-dimensional material models and the two-dimensional model.

<table><tr><td rowspan=1 colspan=1>Hyperparameter</td><td rowspan=1 colspan=1>1D Elastoplasticmodel (seeSection 2.1)</td><td rowspan=1 colspan=1>1D CoupledDamage-Plasticity(see Section 2.2)</td><td rowspan=1 colspan=1>2D Elastoplastic(see Section 2.3)</td></tr><tr><td rowspan=1 colspan=1>Channel width</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>96</td><td rowspan=1 colspan=1>48</td></tr><tr><td rowspan=1 colspan=1>Fourier modes perSpectralConv</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=1>Number of layers</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>SIREN activationfrequency scaling $\left( \omega _ { 0 } \right)$ </td><td rowspan=1 colspan=1>20.75</td><td rowspan=1 colspan=1>19.95</td><td rowspan=1 colspan=1>9.37</td></tr><tr><td rowspan=1 colspan=1>Dropout rate</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.03</td><td rowspan=1 colspan=1>0.0000825</td></tr><tr><td rowspan=1 colspan=1>Number of attentionheads</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>AdamW</td><td rowspan=1 colspan=1>AdamW</td></tr><tr><td rowspan=1 colspan=1>Learning rate</td><td rowspan=1 colspan=1> $\overline { { 2 . 0 3 \times 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 3 . 7 0 \times 1 0 ^ { - 4 } } }$ </td><td rowspan=1 colspan=1> $\overline { { 8 . 2 5 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Weight decay</td><td rowspan=1 colspan=1> $2 . 9 3 \times 1 0 ^ { - 4 }$ </td><td rowspan=1 colspan=1> $5 . 7 4 \times 1 0 ^ { - 5 }$ </td><td rowspan=1 colspan=1> $\overline { { 4 . 9 6 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td rowspan=1 colspan=1>Batch size</td><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>Loss function</td><td rowspan=1 colspan=1>Mean Squared Error</td><td rowspan=1 colspan=1>Mean Squared Error</td><td rowspan=1 colspan=1>Mean Squared Error</td></tr><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>Optuna (TPE)</td><td rowspan=1 colspan=1>Optuna (TPE)</td><td rowspan=1 colspan=1>Optuna (TPE)</td></tr><tr><td rowspan=1 colspan=1>Total tuning trials</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>Maximum epochsPatience</td><td rowspan=1 colspan=1>10,000 / 2,000</td><td rowspan=1 colspan=1>10,000 2,000</td><td rowspan=1 colspan=1>10,000 / 3,000</td></tr></table>

Table 1: Optimized hyperparameters for the 1D and 2D material models described in Section 2.1-2.3.

Hyperparameter selection: The architecture exposes several hyperparameters — the number of retained Fourier modes M, the latent channel width w, the depth L of operator blocks, the SIREN frequency $\omega _ { 0 } ,$ the number of attention heads H, the dropout rate, and the optimizer settings (learning rate, batch size, weight decay). These are tuned jointly via Bayesian optimization with the Tree-structured Parzen Estimator algorithm [63] as implemented in Optuna [64], with median pruning to terminate unpromising trials early. Width and number of heads are drawn jointly from a predefined set of compatible pairs rather than independently. Each configuration is evaluated by its best validation loss over a reduced training budget; the best configuration identified by the search is then retrained at full budget and used for all reported results.

Inference: The full strain trajectory of length n is presented to the model as a single tensor of shape $( B , N , C )$ , where B is the batch size, N is the number of time steps, and C is the number of strain components $( C = 1$ for the 1D case, $C = 3$ for 2D). The operator maps this to a stress tensor of identical shape $( B , N , C )$ in a single forward pass. There is no sliding window, no recurrent loop, and no prediction-feedback. The trajectory length n at inference need not equal the training trajectory length. The operator can be evaluated on the same physical loading path sampled at any resolution, a property that is demonstrated in the discretization-sensitivity study.

## 3.2. Autoregressive Feed Forward Neural Networks with explicit history windows

An FFNN realizes the discrete mapping by retaining only a finite recent window of past strain–stress observations and predicting the next stress step from this window together with the upcoming strain increment. Following Noor et al. [43] and related studies, the network defines a parametric map

$$
\sigma _ { n + 1 } = \mathrm { F F N N } _ { \theta } \big ( \varepsilon _ { n - W + 1 } , \sigma _ { n - W + 1 } , \dots , \varepsilon _ { n } , \sigma _ { n } , \varepsilon _ { n + 1 } \big ) ,\tag{12}
$$

where W is the window length and θ collects all network parameters. The choice of W is a hyperparameter; in our numerical experiments we report results for $W \in \{ 1 , 5 , 1 0 \}$ , denoted $\operatorname { F F N N } ( W \ = \ 1 )$ $\operatorname { F F N N } ( W \mathrm { ~ = ~ 5 ) }$ , and $\mathrm { F F N N } ( W = 1 0 )$ , to expose the sensitivity of FFNN performance to this choice.

![](images/f86f2974b802fe57cc801b9d36b61f73bdfb4370c83fb57139b8a1126c0e87c7.jpg)  
Figure 4: Autoregressive Inference: Feed Forward Neural Network Architecture

Crucially, the FFNN operates autoregressively at inference. To produce a full stress trajectory of length $n ,$ the network is evaluated $n - W + 1$ times in sequence, and after each evaluation the predicted stress $\sigma _ { n + 1 }$ is fed back into the input window in place of the (unobserved) true stress. This introduces two well-known dificulties. First, errors accumulate along the trajectory because each step’s prediction conditions every subsequent step as shown in Figure 4. Second, the input dimension scales linearly with $W$ , making long-memory representations expensive and prone to overfitting; conversely, too small a window cannot capture the relevant material memory.

## 3.2.1. Training protocol

Loss: The FFNN is trained using the mean squared error over single-step predictions, utilizing overlapping historical windows extracted from the full trajectories:

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { N _ { \mathrm { p a i r s } } } \sum _ { k = 1 } ^ { N _ { \mathrm { p a i r s } } } \big ( \hat { \pmb { \sigma } } _ { n + 1 } ^ { ( k ) } - \pmb { \sigma } _ { n + 1 } ^ { ( k ) } \big ) ^ { 2 } ,\tag{13}
$$

where $N _ { p a i r s }$ are total numbers of discrete points in all trajectories/paths given for training, $\begin{array} { r } { \hat { \pmb { \sigma } } _ { n + 1 } ^ { ( k ) } ( \theta ) \dot { = } f _ { \theta } ( \pmb { \varepsilon } _ { n - W + 1 : n } ^ { ( k ) } , \pmb { \sigma } _ { n - W + 1 : n } ^ { ( k ) } , \pmb { \varepsilon } _ { n + 1 } ^ { ( k ) } ) } \end{array}$ is the prediction for the next stress state given a sliding window of size W. Unlike the sequence-to-sequence FNO model, the FFNN relies on historical stresses during training. The model is trained exclusively on fully populated history windows starting from time step W.

Optimization: The models are optimized using AdamW (Loshchilov and Hutter [62]) with a fixed learning rate of $1 0 ^ { - 3 }$ , a weight decay of $1 0 ^ { - 4 }$ , and a batch size of 512. Training proceeds for a maximum of 10,000 epochs, utilizing early stopping with a patience of 500 epochs evaluated on the validation set’s mean squared error to prevent overfitting.

Hyperparameter selection: In contrast to the Bayesian optimization employed for the operator learning approach, the FFNN utilizes a fixed multi-layer perceptron architecture. For example, the hyperparameters chosen for 1D elastoplastic material of Section 2.1 were four hidden layers with sizes 128, 128, 128, 64 with ReLU activations. The primary structural hyperparameter investigated is the sliding window size W, which dictates the temporal memory capacity of the model. Specifically, models with $W \in \{ 1 , 5 , 1 0 \}$ are evaluated independently to quantify the efect of historical context length on path-dependent predictive accuracy. Consequently, the input feature dimension is $2 W + 1$ , comprising 2W past strain and stress values, plus the current strain for which the corresponding stress is being evaluated.

Inference: During deployment, the FFNN operates in a strictly autoregressive manner, a fundamental departure from the non-autoregressive FNO. The model must feed its own past predictions back into the input bufer to step forward in time. The autoregressive unrolling takes the form

$$
\begin{array} { r } { \hat { \pmb { \sigma } } _ { n + 1 } = f _ { \theta } \big ( \pmb { \varepsilon } _ { n - W + 1 : n } , \hat { \pmb { \sigma } } _ { n - W + 1 : n } , \pmb { \varepsilon } _ { n + 1 } \big ) , } \end{array}\tag{14}
$$

where $\hat { \pmb { \sigma } }$ denotes the model’s predictions. Both the sequence strain and sequence stress rolling bufers are initialized to zero to properly enforce the physical initial condition of the material prior to loading.

## 3.3. Recurrent neural networks with hidden-state memory

RNN and its gated variants, in particular GRU and LSTM (Long Short-Term Memory) cells, provide an alternative representation in which the loading memory is encoded in an evolving hidden state $\mathbf { h _ { n } }$ rather than in an explicit input window. Conceptually the hidden state plays the role of a learned, nominal counterpart to the physical internal state variables. It carries the loading memory forward in time without requiring the user to specify what that memory should contain as shown in Figure 5. RNN-based surrogates have been applied to a broad range of inelastic problems including plasticity (Mozafar et al. [47], Gorji et al. [45]). Like the FFNN, the RNN processes the loading path sequentially. A trajectory of length n requires N sequential network evaluations.

The principal structural limitation of RNN-based surrogates in this context is discretization dependence. Because the hidden state is updated through a per-step nonlinear recurrence, the trajectory of $\mathbf { h } _ { n }$ depends on the training resolution of a given loading path. Refining the temporal sampling, even when the underlying continuous strain history is unchanged, produces a diferent sequence of hidden states and hence a diferent stress prediction. This behavior violates the self-consistency expected of a rate-independent constitutive law.

![](images/5d47581b9985ccb3eb5776560a865e13846d047a457c48d68d776fd6af6c07a8.jpg)  
Figure 5: RNN architecture variant with GRU cell

## 3.3.1. Training protocol

Loss: The RNN is trained using the mean squared error evaluated over the entirety of the predicted stress trajectory:

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { N _ { \mathrm { t r } } } \sum _ { j = 1 } ^ { N _ { \mathrm { t r } } } \frac { 1 } { N _ { j } } \sum _ { n = 0 } ^ { N _ { j } - 1 } \left( \hat { \pmb { \sigma } } _ { n } ^ { ( j ) } - \pmb { \sigma } _ { n } ^ { ( j ) } \right) ^ { 2 } ,\tag{15}
$$

where $\hat { \pmb { \sigma } } _ { n } ^ { ( j ) }$ is the stress predicted at timestep n. The model processes the strain history sequentially, taking only the single instantaneous strain tensor $\varepsilon _ { n }$ as input at each step. The path-dependency is captured entirely by the internal hidden state $\boldsymbol { h } _ { n }$ , which evolves continuously via the Gated Recurrent Unit (GRU) operations and is never reset between timesteps.

Optimization: The network is trained using the AdamW optimizer [62] with a fixed learning rate of $1 0 ^ { - 3 }$ , a weight decay of $1 0 ^ { - 4 }$ , and a batch size of 32. The training process runs for a maximum of 500 epochs, utilizing early stopping with a patience of 200 epochs monitored on the validation set’s mean squared error to prevent overfitting.

Hyperparameter selection: Instead of conducting a Bayesian hyperparameter search, this baseline adheres to the exact architectural specifications proposed by Gorji et al. [45]. The GRU model is constructed with a fixed configuration: $N _ { \mathrm { G R U } } = 2$ stacked GRU layers, each containing $N _ { \mathrm { N P U } } ~ = ~ 1 0 0$ hidden neurons. The final hidden state is passed through a Fully Connected Neural Network (FCNN) featuring $N _ { \mathrm { H L } } = 1$ hidden layer with $N _ { \mathrm { { N P L } } } = 1 0 0$ neurons to map the latent memory representation $h _ { n }$ to the instantaneous stress state $\sigma _ { n } .$

Inference: The sequence of strains is processed in a single forward sequential pass. At each discrete timestep n, the single strain value $\varepsilon _ { n }$ updates the persistent hidden state $\pmb { h } _ { n - 1 }  \pmb { h } _ { n }$ and the shared FCNN maps this updated state to $\scriptstyle { \hat { \pmb { \sigma } } } _ { n }$ . Unlike the FFNN approach, there is no sliding window of historical data to manage, and unlike typical sequence generation models, there is no feedback loop of prior predictions into the input stream. The temporal memory is strictly encapsulated within the continuous evolution of the GRU hidden state.

## 4. Results

This section evaluates the proposed FNO-SIREN with causal attention and compares it against two families of baselines: feedforward networks with sliding history windows of length $W \in \{ 1 , 5 , 1 0 \}$ , and an RNN-GRU surrogate. The evaluation is done for three material models: a one-dimensional rate-independent elastoplastic model with nonlinear isotropic hardening, a one-dimensional rate-independent coupled damage-plasticity model and a two-dimensional rate-independent elastoplastic model with a nonlinear isotropic hardening (see Sections 2.1-2.3, respectively). We probe the central methodological claim of this work, discretization invariance, by evaluating the trained models on physically identical loading paths sampled at temporal resolutions ranging from $N = 5 0$ (the training resolution) up to $N = 1 0 0 0$

## 4.1. One-dimensional elastoplastic material model

## 4.1.1. Datasets

We generate $N _ { \mathrm { t r } } ~ = ~ 9 0 0 0$ Gaussian-Process (GP) loading paths for training and another $N _ { \mathrm { v a l } } = 1 0 0 0$ for validation, all uniformly sampled at $N = 5 0$ steps over a fixed time interval. Each path is drawn from a zero-mean GP with squared-exponential covariance whose length scale ℓ is sampled uniformly from [0.005, 0.05], shifted so that $\varepsilon _ { 0 } = 0$ , and rescaled so that the peak strain is drawn uniformly from [ 1.0, 1.0]. This produces a wide distribution of smooth, randomly oscillating loading paths that exercise repeated yielding, unloading, and reloading and the corresponding stress paths were also generated based on the material model defined in Section 2.1. In Figure 6, some training strain paths together with the corresponding stress paths are shown.

![](images/faf6433447855e45ed08fa31b0f5801af9635a719dd196a9ac7ec1748ffa4bb6.jpg)

![](images/88d0f0d9c2e367688bd3a727e0924260a73bdcacb1c3fef33a6c03b8cdcb539b.jpg)  
Figure 6: Training distribution of strain paths (blue, left) and corresponding stress responses (orange, right). The bold solid curves highlight representative strain–stress pairs, while the faint curves are some remaining trajectories in the dataset.

Three families of loading test paths are used to evaluate generalization, each containing 100 trajectories at $N = 5 0$

• Gaussian-Process paths: Freshly drawn GP paths, in-distribution with the training set, used as a sanity check.

• Zig-zag paths: piecewise-linear paths defined by seven equally spaced key points whose interior values are sampled uniformly from [ 1, 1] as shown in Figure 7.

• Sinusoidal paths: $\varepsilon ( t ) = a | \sin ( 2 \pi f t ) |$ with frequency f uniform on [1, 2.5] and amplitude a uniform on [0.1, 1.0]. These paths contain multi-cycle, oscillatory loading–unloading sequences as shown in Figure 7.

In Figure 7, the images in the second row show testing paths used to evaluate the models when a single strain loading is sampled at diferent resolutions, in order to assess discretization invariance. Further details on how these plots were used for the discretization study are provided in Section 4.1.3.

![](images/9d0a9a0d1dc08246c5a54af1b3d6ae8a6450ee10c1a694680cafde8ea815475e.jpg)

![](images/c3c0335f2309fc624b0e3b42f0b79b540ca06db80d09acf970740254b58f08c2.jpg)

![](images/5ab6e3f9d3ccc659ae2515cee8e8c566cd6064969852986259652d7f7d5843cb.jpg)

![](images/29a21210102f0aba0cb3e11033c547a78611898c2a720614dd360f8aa54c962e.jpg)  
Figure 7: Testing distribution and discretization study. Top row shows sample strain paths from the two testing distributions, zig-zag (left) and sinusoidal (right), with bold and dashed curves marking representative examples among the faint background paths. Bottom row shows the same paths resampled at diferent resolutions (N = 50 to N = 1000) against a high resolution reference (N = 5000).

## 4.1.2. Representative predictions

To illustrate the qualitative behavior of the three model families before presenting aggregate statistics, we examine representative stress-strain hysteresis loops under two evaluation conditions.

In each case we compare the FFNN with window size 5, the RNN-GRU surrogate, and the proposed FNO-SIREN with causal attention operator. The two conditions are (i) a Gaussian-Process path evaluated at the training resolution N = 50 as shown in Figure 8a, which serves as a sanity check; (ii) a complex, variable-amplitude cyclic loading path (evaluated at N = 1000) as shown in Figure 9a designed to simultaneously test path-family generalization and resolution invariance.

![](images/a95b56f95768be5f717ffd77e57ef46eb53774ef0e56c981158413a9d87aeacd.jpg)  
(a) Strain loading path at training resolution

![](images/2b8230119f968ea1cbf2dcde2c223ccd36647480095d8b5b152aec528ae5a703.jpg)  
(b) FFNN(W = 5)

![](images/4d76a2b5ee6998637f24d2582a609e8c27cec8c6981f7b908635da5ee64cb190.jpg)  
(c) RNN-GRU

![](images/a2d9fd86a6fee13df47415be8657eab9867970c61d35448faa96aedd01dd3f5e.jpg)  
(d) FNO-SIREN+Attention  
Figure 8: In-distribution (Gaussian-Process) loading path at temporal resolution N = 50

![](images/f17565846dbdb2ececdd6733bb0d2c7f58326572a6479ed410fb9fe29e66115f.jpg)  
(a) Strain loading path at temporal resolution N = 1000

![](images/b5015e17872087888a534c38596fd0ea36cd660bacadb19c043423401c990ceb.jpg)  
(b) FFNN(W = 5)

![](images/c1b3f59fe1c44e926db203f461c673479cda5e92854e735eb83272a7a05079f1.jpg)  
(c) RNN-GRU

![](images/984cfec190acc4480998e6cc530c1d154908338a92994516ab2fb3c7d5e15dd8.jpg)  
(d) FNO-SIREN+Attention  
Figure 9: Complex loading path at temporal resolution N = 1000

Figures 8b, 8c, and 8d show that, at the training resolution, all models predict quite accurately on paths similar to the Gaussian training paths. Reference solutions are obtained from the material model calculated numerically. Model accuracy is evaluated using the relative L2 error, defined in the formula below.

$$
e _ { L _ { 2 } } = \frac { \Vert \hat { y } - y \Vert _ { 2 } } { \Vert y \Vert _ { 2 } } = \sqrt { \frac { \sum _ { i = 1 } ^ { N } \left( \hat { y } _ { i } - y _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } y _ { i } ^ { 2 } } }\tag{16}
$$

where $\hat { y } _ { i }$ is the predicted stress at point $i , y _ { i }$ is the corresponding reference value from the numerical material model, and N is the number of points along the loading path. All models achieve a relative error below 1% with respect to the reference solution.

However, when evaluated at a higher resolution with more complex loading paths, as shown in Figures 9b, 9c, and 9d, the FFNN and RNN-GRU models exhibit an abrupt increase in error, ranging from 6–9%, while the proposed operator architecture, FNO-SIREN+Attention, maintains a substantially lower error of 1.4%. To further investigate this behavior, we conducted a statistical study to assess discretization invariance.

## 4.1.3. Discretization Study

To probe discretization invariance, we evaluate six trained models on the canonical zig-zag test set (100 paths), with each path sampled at multiple temporal resolutions, as illustrated in Figure 7. Specifically, each of the 100 paths is resampled at N  50, 100, 150, 200, 250, 300, 400, 500, 800, 1000 time steps, where N denotes the number of steps. All models are trained at N = 50, and the underlying physical loading paths are identical across resolutions, difering only in temporal discretization. Per-model results across resolutions are reported in Figure 10, and the six curves are combined in Figure 11 for direct comparison.

![](images/586e1b85d764b9516eb8b1aace6b3aac27e0c321a7601b649b8a842fc361fbf2.jpg)  
(a) Feed Forward Neural Network with window size 1

![](images/1f24f966d4fac10123a416d1abf7fb1557a139d707dc91f1a48c2e1ed863537e.jpg)  
(b) Feed Forward Neural Network with window size 5

![](images/9e41645e84a26fdc7cc129ed2a7061c719aaf5d1b3d323dc406c025daa7ac391.jpg)  
(c) Feed Forward Neural Network with window size 10

![](images/e72fa6d793a482547f31b9080d20c8fd45bb2f77bf4a685c0b5482faa29d8320.jpg)  
(d) RNN-GRU

![](images/a8b4b03095b091584b64b901296d1fe554efc66ef5a783c484d016d1e0eea37b.jpg)  
(e) FNO-SIREN

![](images/c3c1036cba82fc9e2557eec6bfe21066db4b81ad319208dc927fbc0b5d0a4665.jpg)  
(f) FNO-SIREN+Attention  
Figure 10: Discretization study for every individual model architecture

As shown in the graphs in Figure 10, the Feed Forward Neural Network with diferent window sizes and RNN-GRU baselines exhibit a clear trend, both the mean relative $L _ { 2 }$ error and its standard deviation grow with sampling resolution. This behavior reflects a known structural limitation of RNN and FFNN based surrogates as the learned mapping is tied to the step size seen during training, and prediction errors accumulate over an increasing number of autoregressive evaluations as the path is refined. Mechanically, this contradicts the defining property of a rate-independent constitutive law that the stress response should depend on the loading path itself, not on the training resolution.

![](images/00b7138dffbb2497304c7af7fbc615a1a5880d1d6015b252b6bd583a2ccd0631.jpg)  
Figure 11: Discretization Study - Comparison Plot

In contrast, the FNO-SIREN model mean error and standard deviation remain essentially constant (flat) throughout the entire resolution range. This is the expected consequence of operator learning. The spectral filters are indexed by physical wavenumbers rather than sample indices, so the same trained operator applies consistently to fine discretizations. Adding causal self-attention on top of the FNO-SIREN backbone further reduces the error at every resolution and significantly lowers the standard deviation compared to other model architectures, as shown in Figures 10 and 11. This makes the model more robust while preserving discretization invariance. The attention-augmented model achieves a lower error at $N = 5 0$ and maintains essentially the same error at N = 1000, twenty times the training resolution. This is empirical evidence that the network has learned the underlying continuous strain-to-stress mapping rather than memorizing a step-size-specific approximation. We performed the same discretization study on 100 sinusoidal loading paths and observed the same overall trend.

## 4.2. One-dimensional coupled damage-plasticity material model

## 4.2.1. Datasets

The dataset construction follows exactly the protocol of Section 4.1.1. Training and validation paths are drawn from the same Gaussian-Process distribution at N = 50 steps as shown in

Figure 12, with identical length-scale and amplitude ranges; the three test families (Gaussian-Process, zig-zag, and sinusoidal, 100 trajectories each) and the canonical zig-zag set used for the discretization study are generated with the same procedures. The reference stress responses are computed by a return-mapping algorithm using Newton iterations; all other dataset parameters are unchanged. This deliberate choice allows the comparison to isolate the efect of the underlying constitutive law from any variation in the loading-path statistics.

![](images/a3ddda8361bffa959d02fbdad4f66f240e1e9d7c46ff404f119af406312d13a8.jpg)

![](images/59e839cae37e7850dae06ee7e86ec193fbecb7361a5eaa6a29a2cc075877ee70.jpg)  
Figure 12: Training distribution: strain paths with corresponding stress paths

## 4.2.2. Representative predictions

To illustrate the qualitative behavior of the three model families for the damage-plasticity material model before presenting aggregate statistics, we examine representative stress-strain hysteresis loops under two progressively harder evaluation conditions. In each case we compare the FFNN with window size 10, the RNN-GRU surrogate, and the proposed FNO-SIREN with causal attention operator. The two evaluation conditions are (i) a Gaussian-Process path evaluated at the training resolution N = 50, which serves as a sanity check as shown in Figure 13a; (ii) a complex, variable-amplitude cyclic loading path (evaluated at $N = 1 0 0 0 )$ as shown in Figure 14a designed to simultaneously test path-family generalization and resolution invariance. All models are trained exclusively at N = 50 on GP paths and receive no further training between conditions.

![](images/f531af5f21142b5148d3b39ee29a0a10cfad8aa1e9276b0721101ef8f2b20849.jpg)  
(a) Strain loading path at training resolution

![](images/cc27e67544c16ef0bad95ac7655b5338c492e46e7dbab16689c924a6905e0eb3.jpg)  
(b) FFNN(W = 10)

![](images/77772a9add818e4a7a90f4791220c1bc1381c3b5beed1ea83383712aff6d83db.jpg)  
(c) RNN-GRU

![](images/a2ccb5ee4c149dc507e6b1ff8fe8ee9a28ae971a7183467acb7004ab44f45d75.jpg)  
(d) FNO-SIREN+Attention

Figure 13: In-distribution (Gaussian-Process) loading path at temporal resolution $\mathrm { N } = 5 0$  
![](images/7b2c3bdca11cf3899404731f0704b34b9fd87ef347dbc09cba30b642f75a086a.jpg)  
(a) Strain loading path at temporal resolution $\mathrm { N } = 1 0 0 0$

![](images/3ed741c8a90b2d80172f399c5df190896caba7aa34ec6a9b8f51113663948b2f.jpg)  
(b) FFNN(W = 10)

![](images/39530d3a4c07062d232dd84ca41dcae59540ba64928864852cbfde1e39e6c849.jpg)  
(c) RNN-GRU

![](images/bf4f91067b491b89bdee0c248e9c8d588843ed16fbfce82ba84202ceb0bed542.jpg)  
(d) FNO-SIREN+Attention  
Figure 14: Complex loading path at temporal resolution N = 1000

Figures 13b, 13c, and 13d show that, at the training resolution, all models predict quite similarly on paths similar to the Gaussian training paths. Reference solutions are obtained from the numerically calculated material model. Model accuracy is evaluated using the relative L2 error. All models had relative error with respect to the reference solution in the range of 5-7%.

However, when evaluated at a higher resolution with more complex loading paths, as shown in Figures 14b, 14c, and 14d, the FFNN exhibits an abrupt increase in error of about 91%, while the proposed operator architecture, FNO-SIREN+Attention, had the substantially lowest error of 4.34%. To further investigate this behavior, we conducted a statistical study to assess discretization invariance.

## 4.2.3. Discretization Study

We repeat the discretization-invariance evaluation, with the surrogates now trained on the coupled damage-plastic model. The 100 canonical zig-zag test paths are evaluated at the same temporal resolutions $N \in \{ 5 0 , 1 0 0 , 1 5 0 , 2 0 0 , 2 5 0 , 3 0 0 , 4 0 0 , 5 0 0 , 8 0 0 , 1 0 0 0 \}$ , and all six model architectures are again trained exclusively at N = 50. Figure 15 reports the mean relative $L _ { 2 }$ error as a function of N for all six models. To make the comparison among the more competitive architectures visible, Figure 15 reproduces the same data with the three FFNN variants omitted, isolating the RNN-GRU baseline, the FNO-SIREN model, and the proposed FNO-SIREN with causal attention.

![](images/e3a0643d6fb079483639af11ac531a96a1d467532e31090cec3b855f9e587820.jpg)  
(a)

![](images/0e5dbea8ca980132f81618e78702c320107d40b1e6910f93bf79bc7a87bab4d3.jpg)  
(b)  
Figure 15: Discretization study on the coupled plastic-damage model

The qualitative picture is identical to that observed for pure elastoplasticity. FFNN accuracy degrades sharply as N grows beyond the training resolution, with one exception: the FFNN with window size 1, whose error plateaus after a certain point as resolution increases. However, this error was already above 80%, an anomalously high and somewhat erratic result that was not investigated further. The step-wise mapping is bound to the training-time discretization, and autoregressive rollout amplifies the mismatch as the path is refined. Adding damage to the constitutive law makes the autoregressive drift even more visible as the network must simultaneously track plastic flow and stifness degradation. The RNN-GRU surrogate performs better than the FFNN family but exhibits a similar, milder upward drift in error with N. The hiddenstate recurrence, trained at N = 50, is evaluated through finer-stepped trajectories at a pace that no longer matches the training distribution, and prediction errors accumulate accordingly.

In contrast, both operator-based models remain essentially flat across the full resolution range. The FNO-SIREN backbone alone is already substantially less sensitive to N than any of the step-wise baselines. This confirms that the discretization-invariance property is valid for both the purely elastoplastic case and the more complex case in which damage and plasticity play a role. Adding causal self-attention on top of the FNO-SIREN backbone yields the lowest error at every resolution and the smallest standard deviation across test trajectories, providing the most robust surrogate among the six. This confirms that the architectural advantages, i.e. the spectral parameterization in physical wavenumbers, the SIREN representation of sharp transitions, and the adaptive causal attention, transfer cleanly to the coupled plastic-damage setting and are not due to the simplicity of the purely elastoplastic model.

## 4.3. 2D plane strain elastoplastic material model

This section extends the evaluation to a two-dimensional plane-strain setting, where the surrogate must simultaneously predict three stress components from three strain components throughout the loading history. The added dificulty is that the trained model must learn the coupling between normal and shear components. Consequently, instead of a one-dimensional strain input, the input now consists of three strain components, forming an input tensor with three channels, as opposed to a single channel previously.

## 4.3.1. Datasets

The training set is constructed to expose the surrogate to a broad span of loading modes. Three families of Gaussian-Process strain paths are combined:

• Multiaxial strain condition GP (10000 paths): every channel/dimension is an independently drawn Gaussian-Process path with range from [ 1, 1], anchored to ε(0) = 0. These paths exercise all three components simultaneously.

• Uniaxial strain condition GP (2000 paths): only a single randomly chosen channel/dimension is active; the other two channels/dimensions are identically zero. These paths exercise pure normal-only or pure shear-only loading.

• Biaxial strain condition GP (2000 paths): two randomly chosen channels/dimensions are active, the third is zero. These paths exercise two-component combinations.

The combined training pool of 14000 paths is split 90/10 into training and validation. All paths use N = 50 sampling points and start at the undeformed state.

## 4.3.2. Representative predictions

To illustrate the qualitative behavior of the RNN and proposed material operator for the 2D elastoplastic material model before presenting aggregate statistics, we examine representative stress-strain hysteresis loops under two evaluation conditions. In each case we compare the RNN-GRU surrogate, and the proposed FNO-SIREN with causal attention operator. The two conditions are (i) a Gaussian-Process biaxial loading path evaluated at the training resolution $N ~ = ~ 5 0$ as shown in Figure 16a, which serves as a sanity check; (ii) a complex, variableamplitude cyclic biaxial loading path (evaluated at $N = 1 0 0 0 )$ as shown in Figure 17a, designed to simultaneously test path-family generalization and resolution invariance. All models are trained exclusively at $N = 5 0$ on GP paths and receive no further training between conditions.

![](images/2887ded7333a264202ffebf7702f845aee0673bc1554a08ba4c6a470904a88c1.jpg)

![](images/96c8f0affbfdd3c9808014e8709340b1eaec732c1f5f01f9fef023629fe18833.jpg)  
(a) Strain loading path at training resolution

![](images/1ae10b0a18424805b2c618ddce5bb4426346f5e0eb533ef095828151b6e4ffec.jpg)  
Figure 16: In-distribution(Gaussian-Process) loading path at temporal resolution N = 50

![](images/25d495eb42b62377fa53615db313fd81daf0a3214773e511dc4c1eb4fbf84c65.jpg)

![](images/e5301e6498d8c8eec213ae6a70137904330479ba356d8832bc2c690d3a3ac8f5.jpg)  
(a) Strain loading path at temporal resolution N = 1000

![](images/0507a389f63c7acb4c035c76abe0d43c55f38e2e654fd6777eed2421259fd1b4.jpg)  
Figure 17: Complex loading path at temporal resolution N = 1000

Figure 16 shows that, at the training resolution, the RNN-GRU model and the proposed operator model predict quite accurately on paths similar to the Gaussian training paths. Reference solutions are obtained from the numerically calculated material model. Model accuracy is evaluated using the relative L2 error. All models had relative error with respect to the reference solution in the range of 0-2%.

However, when evaluated at a higher resolution with more complex loading paths, as shown in Figure 17, the RNN-GRU model shows an increase in error of up to 13.5% for some components, while the proposed operator architecture, FNO-SIREN+Attention, had substantially lower error in the range of 0.8%-3.0%. To further investigate this behavior, we conducted a statistical study to assess discretization invariance.

## 4.3.3. Discretization study

We repeat the discretization-invariance test, now on the 100 canonical zig-zag loading paths consisting of uniaxial, biaxial, and multiaxial loading. The model is trained only at N = 50 and evaluated at finer resolutions. Figure 18 reports the mean relative $L _ { 2 }$ error and its standard deviation band for RNN-GRU, FNO-SIREN and FNO-SIREN+Attention models .

![](images/beb47883a244dcce9c33ff83df995255054bd6b5cc9274284cd864123d826c2e.jpg)  
(a) RNN-GRU

![](images/0c80799026a74ff7ef5c25044933d896bdd981e5698361da57b10f302c4ae8df.jpg)  
(b) FNO-SIREN

![](images/2ef38e8f64de25bae2ebf50b7cb2e30f70509a1ad4f2f56e90bd7eea1484917c.jpg)  
(c) FNO-SIREN+Attention

![](images/a632303ee1670d54141be9e340c9deea2bf5d63adadd62e7d0d556d1ab5b3422.jpg)  
(d) Overall mean error for all architectures  
Figure 18: Discretization study comparing RNN-GRU, FNO-SIREN, and FNO-SIREN with Attention.

For FNO-SIREN and FNO-SIREN+Attention, the overall mean error remains almost constant (flat) across the entire range of sampling resolutions. Each type of loading (uniaxial, biaxial or multiaxial) curve exhibits the same (constant error) flat behavior, all predicted with comparable or better accuracy regardless of the temporal sampling density. The standarddeviation band remains narrow throughout, indicating that the discretization invariance is not the result of averaging over a heterogeneous distribution of per-trajectory errors but a structural property that holds path-by-path. These results confirm the successful generalization of the architectural ingredients identified in previous sections. Specifically, spectral parameterization indexed by physical wavenumbers, SIREN representation of sharp transitions, and adaptive causal self-attention transition seamlessly from a scalar one-dimensional to the two-dimensional case. The operator learned at N = 50 delivers predictions of comparable quality at N = 1000 across loadings (uniaxial, biaxial, or multiaxial).

## 5. Accuracy and computational eficiency of surrogate models

We compare all six surrogate models in terms of predictive accuracy and computational cost. Accuracy is reported at both the training resolution N = 50 and the highest evaluated resolution N = 1000, so that the table captures both the in-distribution performance and the resolution-generalization behavior simultaneously. Sections 4.1.2, 4.2.2, and 4.3.2 present the predicted stress–strain responses for representative loading paths under the three evaluation conditions. Table 2 summarizes the mean relative $L _ { 2 }$ error over 100 zig-zag test paths for the 1D elastoplasticity benchmark, together with training and per-trajectory inference times. Data generation is shared across all models; this cost is excluded from the per-model training times reported below.

<table><tr><td rowspan=1 colspan=1>ModelArchitectures</td><td rowspan=1 colspan=1>Mean Errorat trainingresolution (%)</td><td rowspan=1 colspan=1>Mean Errorat N = 1000resolution (%)</td><td rowspan=1 colspan=1>Training Time(min)</td><td rowspan=1 colspan=1>Inferencetime persignal (msec)</td></tr><tr><td rowspan=1 colspan=1>FFNN (W=1)</td><td rowspan=1 colspan=1>1.51</td><td rowspan=1 colspan=1>13.27</td><td rowspan=1 colspan=1>14.6</td><td rowspan=1 colspan=1>12.882</td></tr><tr><td rowspan=1 colspan=1>FFNN(W=5)</td><td rowspan=1 colspan=1>1.29</td><td rowspan=1 colspan=1>13.83</td><td rowspan=1 colspan=1>10.8</td><td rowspan=1 colspan=1>14.935</td></tr><tr><td rowspan=1 colspan=1>FFNN(W=10)</td><td rowspan=1 colspan=1>1.2</td><td rowspan=1 colspan=1>9.87</td><td rowspan=1 colspan=1>24.1</td><td rowspan=1 colspan=1>14.891</td></tr><tr><td rowspan=1 colspan=1>RNN-GRU</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>12.41</td><td rowspan=1 colspan=1>17.28</td><td rowspan=1 colspan=1>18.629</td></tr><tr><td rowspan=1 colspan=1>FNO-SIREN</td><td rowspan=1 colspan=1>4.28</td><td rowspan=1 colspan=1>7.49</td><td rowspan=1 colspan=1>22.36</td><td rowspan=1 colspan=1>1.487</td></tr><tr><td rowspan=1 colspan=1>FNO-SIREN+Attention</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>2.01</td><td rowspan=1 colspan=1>103.45</td><td rowspan=1 colspan=1>2.617</td></tr></table>

Table 2: Performance comparison between various model architectures and the traditional return mapping algorithm (baseline inference time: 51.156 ms) for the 1D elastoplasticity material model. All metrics were benchmarked on an NVIDIA GeForce RTX 4090 GPU.

The results summarized in Table 2 reveal several important trends regarding the trade-ofs between accuracy, generalization, and computational cost across the six surrogate architectures evaluated. For each metric, the best and second-best results are highlighted in boldface and underlined, respectively.

Across the six surrogate architectures, RNN-GRU achieves the best accuracy at the training resolution (0.32%), reflecting its natural suitability for capturing the path-dependent, historysensitive nature of elastoplastic loading. However, this advantage does not carry over to the higher evaluation resolution (N = 1000), where RNN-GRU’s error rises sharply to 12.41%, similar to the degradation seen in the FFNN variants (9.87–13.83%). This indicates that purely feed-forward and recurrent architectures, while efective within their training discretization, lack the inductive bias needed to generalize across resolutions. In contrast, the FNO-based models generalize considerably better as FNO-SIREN leads to 7.49% error at N = 1000. The FNO-SIREN+Attention model achieves the best overall balance, with 0.75% error at training resolution and only 2.01% at N = 1000 which shows an order of magnitude better generalization than the FFNN and RNN-GRU models, consistent with the resolution-invariance properties of Fourier-based operator learning enhanced by attention.

These accuracy gains come with trade-ofs in computational cost. The FNO-SIREN+Attention surrogate requires by far the longest training time (103.45 min, roughly 4–9 the other models), reflecting the added complexity of combining spectral convolutions with attention. Despite this, its inference time (2.617 msec/signal) is much faster than the FFNN and RNN-GRU models (12.8–18.6 msec/signal), whose sequential or dense computations are less eficient at inference. Overall, FNO-SIREN+Attention is the preferred choice when resolution-robust predictions and fast inference are required, justifying its higher one-time training cost for deployment scenarios involving varying discretizations. While not applied to the 1D coupled plastic-damage model, FNO-SIREN+Attention model should yield equal or greater speed-ups than the elastoplastic case due to its reduced layer and Fourier mode requirements (Table 1). Instead, we extended the study to the multidimensional case. Results are presented in Table 3. The observed trends remain consistent with those reported above.

<table><tr><td rowspan=1 colspan=1>ModelArchitectures</td><td rowspan=1 colspan=1>Mean Errorat trainingresolution (%)</td><td rowspan=1 colspan=1>Mean Errorat N = 1000resolution (%)</td><td rowspan=1 colspan=1>Training Time(min)</td><td rowspan=1 colspan=1>Inferencetime persignal (msec)</td></tr><tr><td rowspan=1 colspan=1>FFNN(W=1)</td><td rowspan=1 colspan=1>1.201</td><td rowspan=1 colspan=1>12.502</td><td rowspan=1 colspan=1>29.82</td><td rowspan=1 colspan=1>12.267</td></tr><tr><td rowspan=1 colspan=1>FFNN(W=5)</td><td rowspan=1 colspan=1>1.453</td><td rowspan=1 colspan=1>15.671</td><td rowspan=1 colspan=1>49.15</td><td rowspan=1 colspan=1>16.897</td></tr><tr><td rowspan=1 colspan=1>FFNN(W=10)</td><td rowspan=1 colspan=1>1.425</td><td rowspan=1 colspan=1>18.273</td><td rowspan=1 colspan=1>38.82</td><td rowspan=1 colspan=1>16.988</td></tr><tr><td rowspan=1 colspan=1>RNN-GRU</td><td rowspan=1 colspan=1>1.211</td><td rowspan=1 colspan=1>12.215</td><td rowspan=1 colspan=1>14.38</td><td rowspan=1 colspan=1>18.503</td></tr><tr><td rowspan=1 colspan=1>FNO-SIREN</td><td rowspan=1 colspan=1>6.463</td><td rowspan=1 colspan=1>7.364</td><td rowspan=1 colspan=1>197.42</td><td rowspan=1 colspan=1>1.537</td></tr><tr><td rowspan=1 colspan=1>FNO-SIREN+Attention</td><td rowspan=1 colspan=1>1.654</td><td rowspan=1 colspan=1>2.717</td><td rowspan=1 colspan=1>273.40</td><td rowspan=1 colspan=1>2.827</td></tr></table>

Table 3: Performance comparison between various model architectures and the traditional return mapping algorithm (baseline inference time: 30.470 ms) for the 2D elastoplasticity material model. All metrics were benchmarked on an NVIDIA GeForce RTX 4090 GPU.

Figure 19 evaluates the extent to which the trained surrogate models are thermodynamically consistent with the Newton-Raphson reference, by measuring the discrepancy between their implied cumulative mechanical work over time. For a stress history $\pmb { \sigma } ( t )$ produced along an applied strain path $\varepsilon ( t )$ , the cumulative mechanical work done by the stress up to time t is

$$
E ( t ) = \int _ { 0 } ^ { t } { \pmb \sigma } : \dot { \varepsilon } d { \tau } .\tag{17}
$$

The local Clausius–Duhem inequality states that the mechanical dissipation $D = { \pmb \sigma } : \dot { \pmb \varepsilon } - \dot { \psi }$ must be non-negative at every instant, where $\psi$ is the Helmholtz free energy. Integrating this pointwise statement from 0 to t gives the time-integrated form:

$$
\int _ { 0 } ^ { t } D \mathrm { d } \tau = \int _ { 0 } ^ { t } { \Bigl ( } \sigma : { \dot { \varepsilon } } - { \dot { \psi } } { \Bigr ) } \mathrm { d } \tau \geq 0 \quad  \quad D ( t ) = E ( t ) - \psi ( t ) \geq 0\tag{18}
$$

Evaluating $\psi ( t )$ in general requires knowledge of the plastic hardening and damage contributions to the free energy, in addition to the elastic part. However, the surrogate models are trained purely on strain-to-stress data, without access to any constitutive parameters or internal variables, so $\psi ( t )$ itself is not directly accessible. We therefore make the reasonable assumptions that the initial state of the material coincides with its natural, virgin state, ${ \mathrm { i . e . , ~ } } \psi ( 0 ) = 0$ $D ( t = 0 ) = 0$ and that the Helmholtz free energy is non-negative for all time, $\psi ( t ) \geq 0$

![](images/6e8dc8953ff09c38f011867d5eeda753431814cb77c2a07771bd9971337b4e92.jpg)  
(a) In-distribution loading path at higher resolution

![](images/d775633344ebe9f2757cb22b35be1490c625ba5e1fd672489f459f523178f32e.jpg)

![](images/ad32a8fb01a9963ba5f77f4d22b68df1d7688c0d103b8267d5700c09f1dcfa19.jpg)  
(c) Mechanical work for 1D elastoplasticity material model

(b) Complex loading path at higher resolution  
![](images/90bbb4c5f822b9aa7d54408cfac45c3e80119f69a6289517747cbcbdfc175a9b.jpg)

![](images/c580b49523dd91ccb1c981c6581fc5e005e0f9f30393b4cd01d126c42450924d.jpg)  
(e) Mechanical work for 1D coupled damage-plasticity model

(d) Mechanical work for 1D elastoplasticity material model  
![](images/9ac934962cb9e1bd8042c72e2c67347cab29d79f146735c21e0e245ee878d5b6.jpg)  
(f) Mechanical work for 1D coupled damage-plasticity model  
Figure 19: Mechanical work diference between surrogate models and reference models

Under these assumptions, $E ( t )$ can serve as a model-agnostic, thermodynamically admissible quantity that is checkable directly from the predicted stress–strain response. Since $D ( t ) \ \geq$

0 and $\psi ( t ) \geq 0$ , it follows that $E ( t ) = D ( t ) + \psi ( t ) \geq 0$ holds for every material for any thermodynamically admissible process. A surrogate whose predicted $E ( t )$ deviates from this violates the second law of thermodynamics.

We initially attempted to exploit this same conservativeness during training, imposing $E ( t ) \geq 0$ directly as a penalty on the network’s predictions. This did not produce a significant change in the predictive behavior of the surrogates. Because the reference solution satisfies $E ( t ) \geq 0$ exactly by construction as shown in Figure 19, we quantify each surrogate’s departure from thermodynamic consistency:

$$
\int _ { 0 } ^ { t } \left| \Delta E \right| d \tau = \int _ { 0 } ^ { t } \left| E _ { \mathrm { p r e d } } ( \tau ) - E _ { \mathrm { r e f } } ( \tau ) \right| d \tau\tag{19}
$$

At higher resolution, all three evaluated models show a comparatively small accumulated error, with the FNO-SIREN+Attention model showing the smallest accumulated error, followed by RNN-GRU and then FFNN. This indicates that FNO-SIREN+Attention’s stress predictions track the reference’s energy accurately under high resolution and complex paths.

## 6. Potential extension to FE solver

A question for any data-driven constitutive surrogate is whether it can be embedded in a finite element solver, where the constitutive law is queried incrementally. The internal variables that a classical model would carry between steps are, in our case, never predicted or calculated at all. Since the proposed operator has no explicit internal state, it is not obvious a priori that it can be used in this incremental setting without retraining or redesign.

To test this, we take a single loading path and query the trained operator at increasing horizons, corresponding to the path truncated at $t = 0 . 5 \mathrm { s } , 1 . 0 \mathrm { s } , \ldots , 5 . 0 \mathrm { s }$ . At each horizon, the model is given only the strain history up to that point and predicts the stress over that same interval, as an FE solver would present a growing loading history to the constitutive law as the simulation advances. Figure 20 shows the input strain path together with five representative horizons against the reference return-mapping solution for the 1D elastoplastic material model (Section 2.1); the remaining horizons follow the same trend and are omitted for space.

Two observations follow directly from these results. First, the model produces an accurate stress response at every horizon, with the relative error remaining between 0.33% and 1.06% across all ten windows and showing no upward trend as the horizon lengthens. Second, and more significant for FE deployment, each prediction is an independent forward pass rather than an extension of the previous one. It is recomputed from scratch from the full strain history up to that point. If predictions at later horizons were built on top of earlier ones, any small inaccuracy at t = 0.5s would compound as the horizon grows, and this doesn’t happen in our case as shown in Figure 20. This points to a practical strategy for coupling the operator to an FE solver. At each global time step, the solver would supply the strain history recorded so far at a given integration point, the operator would return the full stress history over that interval, and only the stress at the current time step would be retained for the equilibrium iteration, with the rest of the returned trajectory discarded. Because each call is independent, this requires no internal state to be tracked by the solver between steps beyond the strain history itself. The results in Figure 20 indicate that this repeated-evaluation strategy would remain accurate throughout the simulation, since the operator’s accuracy at a given horizon does not depend on how many times it has been queried at earlier horizons.

![](images/3802439503f12d99fe9f2374f6561030553ca075ef7375ac65e7cd24b37e9bdb.jpg)  
(a) Input strain

![](images/1851e58e5ffd60a78ae8553b0875fe477d7c46c1e101601b5391c7c6b46d752c.jpg)  
(b) Stress field till t1 (0.5 s)

![](images/db6edf80583e86802bb749a053a276453f447b904ce17e608d5f102fccf28c11.jpg)  
(c) Stress field till t4 (2 s)

![](images/9faacad6b40e11034874b68485e232c7b0bca89745275db37df3d709cbe8bb5f.jpg)

![](images/5d705f1a22f18dad87b34e062429a191b3e87791eed0ea8e54c480190faca60c.jpg)  
(e) Stress field till t8 (4 s)

(d) Stress field till t5 (2.5 s)  
![](images/1b06ffbcdca24047171b2c0b4373b61ac3b94bd7f7c8f038350850856c30ad63.jpg)  
(f) Stress field till t10 (5 s)  
Figure 20: No error accumulation across independently evaluated horizons

## 7. Conclusion and Outlook

This work addressed a structural limitation that has long constrained data-driven constitutive modeling of rate-independent path-dependent materials, which is the tying of the learned representation to the temporal discretization of the training data. Step-wise surrogates, such as feedforward networks with sliding history windows and recurrent networks with hidden-state memory, process the loading path one increment at a time, and their predictions degrade when the test loading path is sampled at a resolution diferent from the training resolution. For a rate-independent constitutive law, whose response should depend only on the loading path and not on how it is sampled, this is a mechanical inconsistency.

We proposed a sequence-to-sequence operator that departs from both autoregressive and sequential-recurrent paradigms. Instead of consuming the loading path incrementally, the proposed model presents the entire discretely sampled strain trajectory as a single function input and produces the entire stress trajectory in one parallel forward pass. Path dependence is architecturally encouraged by a strict causal mask inside every self-attention layer. The architecture combines three complementary ingredients: an FNO backbone for resolution-invariant spectral convolution, sinusoidal representation layers (SIREN) for the high-fidelity recovery of sharp transitions characteristic of yield onset, unloading, and damage initiation, and causal multi-head self-attention for adaptive weighting of the strain history.

We evaluated the framework using three constitutive models of increasing complexity. We began with one-dimensional elastoplasticity featuring nonlinear isotropic hardening, moved on to one-dimensional coupled damage-plasticity, and concluded with two-dimensional plane-strain $J _ { 2 }$ elastoplasticity. Across all three benchmarks the proposed model achieved quite low prediction error and the smallest sensitivity to the temporal discretization, retaining essentially flat error curves across a twenty-fold change in sampling resolution from $N = 5 0$ (the training resolution) up to $N = 1 0 0 0$ The operator achieved its best performance when the internal variables were highly coupled, as demonstrated in the coupled damage-plasticity model. In this scenario, the proposed framework showed the largest advantage over the baseline data-driven approaches. Furthermore, the core architectural features enabling this robustness transferred without modification to both the coupled damage-plasticity and multi-component plane-strain settings. In the latter case, the operator successfully learned the coupling between normal and shear components alongside the diferential evolution of the plastic variables. In terms of computational eficiency, the proposed operator also delivered substantially faster inference than competing data-driven approaches, despite its added architectural complexity. This speed advantage, given a one-time training cost per material model, makes the framework particularly well suited for deployment scenarios requiring repeated or large-scale evaluation.

The observed discretization invariance has a simple interpretation. The model has learned the continuous strain-to-stress mapping itself, not an approximation tied to one step size. This is exactly the behavior a rate-independent constitutive law requires, and in the proposed framework it follows directly from the architecture, without any additional constraint or penalty during training.

The present study is limited to rate-independent materials, for which the response depends on the loading path but not on the loading rate. Extending the framework to viscoelastic and viscoplastic materials requires the temporal axis to carry physical-time information rather than to act purely as a sample index, which in turn requires the spectral filters to be parameterized in terms of true frequency rather than dimensionless wavenumber. This extension is conceptually compatible with the proposed architecture and would broaden its applicability to a much wider class of inelastic materials. The current architecture already encourages causality in the material operator through the causal attention mask. Building on this, a natural next step is to addi tionally enforce thermodynamic admissibility. An additional extension can be to condition the operator on a microstructural descriptor. The resulting joint space-time operator would predict the full stress response under arbitrary loading from a heterogeneous microstructure, providing a unified surrogate for both the constitutive behavior and the microstructural variability.

Together, these future directions aim to use the current discretization-invariant surrogate to produce a physics-consistent and microstructure-aware operator, fully integrated within finite element frameworks for complex inelastic materials.

## Data Availability

All codes used for this study are openly available in folax at Material-Operator.

## Acknowledgements

R. Arora, T. Brepols, and S. Rezaei thankfully acknowledge the funding of the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – Project number 561202254. L. Scheunemann gratefully acknowledges the funding of project SPP 2489 (project number 562153065) by the DFG. L. Scheunemann and T. Brepols gratefully acknowledge the funding of project CRC/TRR 339 (subproject B05, project number 453596084) by the DFG. During the preparation of this work, the authors used Claude (Opus 4.8, Anthropic) to assist with drafting and editing portions of the manuscript text. After using this tool/service, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Funding Information

This work was supported by the Deutsche Forschungsgemeinschaft (561202254, 562153065, 453596084).

## Conflict of Interest Statement

The authors declare no conflict of interest.

## Appendix A. Discretization study on sinusoidal loading paths

We repeat the procedure of Sections 4.1.3 and 4.2.3 on the 100 canonical sinusoidal test paths, evaluating all six models (trained at N = 50) at resolutions N 50, 100, 150, 200, 250, 300, 400, 500, 800, 1000 . Figures 21, 22 and 23 report the mean relative $L _ { 2 }$ error as a function of temporal resolution. The overall picture is consistent with the zig-zag study: the FFNN(W = 1), FFNN(W = 5), FFNN(W = 10), and RNN-GRU models all degrade steadily as resolution increases. The FNO-SIREN model starts from a higher baseline error but remains comparatively flat across the full resolution range, confirming that the discretization-invariance of the spectral backbone holds independently of the attention mechanism. Adding attention improves accuracy substantially without sacrificing this discretization-invariant behavior.

![](images/8b8d4a4b0913106844b236cf8ebfaf44e6dcf1f68eadc9716fdb41df387ae91f.jpg)  
Figure 21: Discretization study on sinusoidal loading paths, 1D elastoplastic material model.

![](images/6c734d89a946a6752634718837bca0171a26ca5dda71c184c1d9b6386b38768b.jpg)  
Figure 22: Discretization study on sinusoidal loading paths, 1D coupled damage-plasticity model.

![](images/a9b1b9254e9af0bfdaa6638612569c21b4626fd464e98a1a6c54409c41716c54.jpg)  
Figure 23: Discretization study on sinusoidal loading paths, 1D coupled damage-plasticity model.

## References

[1] Bernard D. Coleman and Morton E. Gurtin. Thermodynamics with internal state variables. The Journal of Chemical Physics, 47(2):597–613, 07 1967. ISSN 0021-9606. doi: 10.1063/ 1.1711937. URL https://doi.org/10.1063/1.1711937.

[2] J.C. Simo and T.J.R. Hughes. Computational Inelasticity. Interdisciplinary Applied Mathematics. Springer New York, 2006. ISBN 9780387227634. URL https://books.google. de/books?id=EILbBwAAQBAJ.

[3] J. Lubliner. Plasticity Theory. Dover books on engineering. Dover Publications, 2008. ISBN 9780486462905. URL https://books.google.de/books?id=MkK-BLbHtcAC.

[4] Eduardo de Souza Neto, D. Perić, and D.R.J. Owen. Computational Methods for Plasticity: Theory and Applications. 12 2008. ISBN 9780470694527. doi: 10.1002/9780470694626.

[5] Jean Lemaitre. Coupled elasto-plasticity and damage constitutive equations. Computer Methods in Applied Mechanics and Engineering, 51(1):31–49, 1985. ISSN 0045-7825. doi: https://doi.org/10.1016/0045-7825(85)90026-X. URL https://www.sciencedirect.com/ science/article/pii/004578258590026X.

[6] Jamshid Ghaboussi, James H. Garrett, and X. Wu. Knowledge-based modeling of material behavior with neural networks. Journal of Engineering Mechanics-asce, 117:132–153, 1992. URL https://api.semanticscholar.org/CorpusID:110978241.

[7] Jamshid Ghaboussi, Xiping Wu, and Gintaris Kaklauskas. Neural network material modelling. Statyba, 5:250–257, 07 2012. doi: 10.1080/13921525.1999.10531472.

[8] Kurt Hornik, Maxwell Stinchcombe, and Halbert White. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. ISSN 0893-6080. doi: https://doi.org/10.1016/0893-6080(89)90020-8. URL https://www.sciencedirect.com/ science/article/pii/0893608089900208.

[9] Tomonari Furukawa and Genki Yagawa. Implicit constitutive modelling for viscoplasticity using neural networks. International Journal for Numerical Methods in Engineering, 43 (2):195–219, 1998. doi: https://doi.org/10.1002/(SICI)1097-0207(19980930)43:2<195:: AID-NME418>3.0.CO;2-6. URL https://onlinelibrary.wiley.com/doi/abs/10. 1002/%28SICI%291097-0207%2819980930%2943%3A2%3C195%3A%3AAID-NME418%3E3.0. CO%3B2-6.

[10] Mauricio Fernández, Shahed Rezaei, Jaber Mianroodi, Felix Fritzen, and Stefanie Reese. Application of artificial neural networks for the prediction of interface mechanics: a study on grain boundary constitutive behavior. Advanced Modeling and Simulation in Engineering Sciences, 7, 01 2020. doi: 10.1186/s40323-019-0138-7.

[11] Annan Zhang and Dirk Mohr. Using neural networks to represent von mises plasticity with isotropic hardening. International Journal of Plasticity, 132:102732, 2020. ISSN 0749-6419. doi: https://doi.org/10.1016/j.ijplas.2020.102732. URL https://www.sciencedirect. com/science/article/pii/S0749641919307119.

[12] Dengpeng Huang, Jan Niklas Fuhg, Christian Weißenfels, and Peter Wriggers. A machine learning based plasticity model using proper orthogonal decomposition. Computer Methods in Applied Mechanics and Engineering, 365:113008, 2020. ISSN 0045-7825. doi: https: //doi.org/10.1016/j.cma.2020.113008. URL https://www.sciencedirect.com/science/ article/pii/S0045782520301924.

[13] Mauricio Fernández, Mostafa Jamshidian, Thomas Böhlke, Kristian Kersting, and Oliver Weeger. Anisotropic hyperelastic constitutive models for finite deformations combining material theory and data-driven approaches with application to cubic lattice metamaterials. Computational Mechanics, 67(2):653–677, 2021. doi: 10.1007/s00466-020-01954-7.

[14] Jaber Mianroodi, Shahed Rezaei, Nima Siboni, Bai-Xiang Xu, and Dierk Raabe. Lossless multi-scale constitutive elastic relations with artificial intelligence. npj Computational Materials, 8:67, 04 2022. doi: 10.1038/s41524-022-00753-3.

[15] Ari Frankel, Kousuke Tachida, and Reese Jones. Prediction of the evolution of the stress field of polycrystals undergoing elastic-plastic deformation with a hybrid neural network model. Machine Learning: Science and Technology, 1(3), 07 2020. ISSN ISSN 2632-2153. doi: 10.1088/2632-2153/ab9299. URL https://www.osti.gov/biblio/1638756.

[16] Usman Ali, Waqas Muhammad, Abhijit Brahme, Oxana Skiba, and Kaan Inal. Application of artificial neural networks in micromechanics for polycrystalline metals. International Journal of Plasticity, 120:205–219, 2019. ISSN 0749-6419. doi: https://doi.org/ 10.1016/j.ijplas.2019.05.001. URL https://www.sciencedirect.com/science/article/ pii/S0749641918307290.

[17] Jan Fuhg, Govinda Padmanabha, Nikolaos Bouklas, Bahador Bahmani, Waiching Sun, Nick Vlassis, Moritz Flaschel, Pietro Carrara, and Laura De Lorenzis. A review on data-driven constitutive laws for solids, 05 2024.

[18] Arif Hussain, Amir Hosein Sakhaei, and Mahmoud Shafiee. Machine learning-based constitutive modelling for material non-linearity: A review. Mechanics of Advanced Materials and Structures, 33:1–19, 12 2024. doi: 10.1080/15376494.2024.2439557.

[19] Johannes Dornheim, Lukas, and MorandDirk Helm. Neural networks for constitutive modeling – from universal function approximators to advanced models and the integration of physics. preprint, 2023.

[20] Max Rosenkranz, Karl Alexander Kalina, Jörg Brummund, and Markus Kästner. A comparative study on diferent neural network architectures to model inelasticity, 03 2023.

[21] T. Kirchdoerfer and M. Ortiz. Data-driven computational mechanics. Computer Methods in Applied Mechanics and Engineering, 304:81–101, 2016. ISSN 0045-7825. doi: https: //doi.org/10.1016/j.cma.2016.02.001. URL https://www.sciencedirect.com/science/ article/pii/S0045782516300238.

[22] R. Eggersmann, T. Kirchdoerfer, S. Reese, L. Stainier, and M. Ortiz. Model-free datadriven inelasticity. Computer Methods in Applied Mechanics and Engineering, 350:81–99, 2019. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2019.02.016. URL https: //www.sciencedirect.com/science/article/pii/S0045782519300878.

[23] Filippo Masi, Ioannis Stefanou, Paolo Vannucci, and Victor Mafi-Berthier. Thermodynamics-based artificial neural networks for constitutive modeling. Journal of the Mechanics and Physics of Solids, 147:104277, 2021. ISSN 0022-5096. doi: https://doi.org/10.1016/j.jmps.2020.104277. URL https://www.sciencedirect.com/ science/article/pii/S0022509620304841.

[24] Filippo Masi, Ioannis Stefanou, Paolo Vannucci, and Victor Mafi-Berthier. Material Modeling via Thermodynamics-Based Artificial Neural Networks, pages 308–329. 06 2021. ISBN 978-3-030-77956-6. doi: 10.1007/978-3-030-77957-3\_16.

[25] Nikolaos N. Vlassis and WaiChing Sun. Sobolev training of thermodynamic-informed neural networks for interpretable elasto-plasticity models with level set hardening. Computer Methods in Applied Mechanics and Engineering, 377:113695, 2021. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2021.113695. URL https://www.sciencedirect. com/science/article/pii/S0045782521000311.

[26] Ehsan Haghighat, Sahar Abouali, and Reza Vaziri. Constitutive model characterization and discovery using physics-informed deep learning. Engineering Applications of Artificial Intelligence, 120:105828, 2023. ISSN 0952-1976. doi: https://doi.org/10.1016/j. engappai.2023.105828. URL https://www.sciencedirect.com/science/article/pii/ S095219762300012X.

[27] M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational Physics, 378:686–707, 2019. ISSN 0021- 9991. doi: https://doi.org/10.1016/j.jcp.2018.10.045. URL https://www.sciencedirect. com/science/article/pii/S0021999118307125.

[28] Kailai Xu, Daniel Z. Huang, and Eric Darve. Learning constitutive relations using symmetric positive definite neural networks. Journal of Computational Physics, 428: 110072, 2021. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2020.110072. URL https://www.sciencedirect.com/science/article/pii/S0021999120308469.

[29] Karl A. Kalina, Lennart Linden, Jörg Brummund, and Markus Kästner. FE<sup>ANN</sup>: an eficient data-driven multiscale approach based on physics-constrained neural networks and automated data mining. Computational Mechanics, 71(5):827–851, February 2023. ISSN 1432-0924. doi: 10.1007/s00466-022-02260-0. URL http://dx.doi.org/10.1007/ s00466-022-02260-0.

[30] Dominik K. Klein, Mauricio Fernández, Robert J. Martin, Patrizio Nef, and Oliver Weeger. Polyconvex anisotropic hyperelasticity with neural networks. Journal of the Mechanics and Physics of Solids, 159:104703, 2022. ISSN 0022-5096. doi: https://doi.org/ 10.1016/j.jmps.2021.104703. URL https://www.sciencedirect.com/science/article/ pii/S0022509621003215.

[31] Shahed Rezaei, Ahmad Moeineddin, and Ali Rajaei Harandi. Learning solutions of thermodynamics-based nonlinear constitutive material models using physics-informed neural networks. Computational Mechanics, 74:1–34, 01 2024. doi: 10.1007/ s00466-023-02435-3.

[32] Max Rosenkranz, Karl Alexander Kalina, Jörg Brummund, Waiching Sun, and Markus Kästner. Viscoelasticty with physics-augmented neural networks: model formulation and training methods without prescribed internal variables. Computational Mechanics, 74: 1279–1301, 05 2024. doi: 10.1007/s00466-024-02477-1.

[33] Fadi Aldakheel, Elsayed Elsayed, Yousef Heider, and Oliver Weeger. Physics-based machine learning for computational fracture mechanics. Machine Learning for Computational Science and Engineering, 1, 04 2025. doi: 10.1007/s44379-025-00019-x.

[34] Faisal As’ad and Charbel Farhat. A mechanics-informed deep learning framework for datadriven nonlinear viscoelasticity. Computer Methods in Applied Mechanics and Engineering, 417, 09 2023. doi: 10.1016/j.cma.2023.116463.

[35] Arunabha M. Roy, Rikhi Bose, Veera Sundararaghavan, and Raymundo Arróyave. Deep learning-accelerated computational framework based on physics informed neural network for the solution of linear elasticity. Neural Networks, 162:472–489, 2023. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2023.03.014. URL https://www.sciencedirect. com/science/article/pii/S0893608023001314.

[36] Steven Brunton, Joshua Proctor, and J. Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113:3932–3937, 03 2016. doi: 10.1073/pnas.1517384113.

[37] Jorge–Humberto Urrea–Quintero, David Anton, Laura De Lorenzis, and Henning Wessels. Automated constitutive model discovery by pairing sparse regression algorithms with model selection criteria. Computer Methods in Applied Mechanics and Engineering, 449:118551, 2026. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2025.118551. URL https: //www.sciencedirect.com/science/article/pii/S0045782525008230.

[38] K. Linka, M. Hillgärtner, K. P. Abdolazizi, R. C. Aydin, M. Itskov, and C. J. Cyron. Constitutive artificial neural networks: A fast and general approach to predictive datadriven constitutive modeling by deep learning. Journal of Computational Physics, 429: 110010, 2021. doi: 10.1016/j.jcp.2020.110010.

[39] Kevin Linka and Ellen Kuhl. A new family of constitutive artificial neural networks towards automated model discovery. Computer Methods in Applied Mechanics and Engineering, 403:115731, 01 2023. doi: 10.1016/j.cma.2022.115731.

[40] Hagen Holthusen, Lukas Lamm, Tim Brepols, Stefanie Reese, and Ellen Kuhl. Theory and implementation of inelastic constitutive artificial neural networks. Computer Methods in Applied Mechanics and Engineering, 428:117063, 2024. ISSN 0045-7825. doi: https: //doi.org/10.1016/j.cma.2024.117063. URL https://www.sciencedirect.com/science/ article/pii/S0045782524003190.

[41] Kian P. Abdolazizi, Kevin Linka, and Christian J. Cyron. Viscoelastic constitutive artificial neural networks (vcanns) – a framework for data-driven anisotropic nonlinear finite viscoelasticity. Journal of Computational Physics, 499:112704, 2024. ISSN 0021-9991. doi: https://doi.org/10.1016/j.jcp.2023.112704. URL https://www.sciencedirect.com/ science/article/pii/S0021999123007994.

[42] Filippo Masi and Ioannis Stefanou. Evolution tann and the identification of internal variables and evolution equations in solid mechanics. Journal of the Mechanics and Physics of Solids, 174:105245, 2023. doi: 10.1016/j.jmps.2023.105245.

[43] Toiba Noor, Soban Lone, G. Ramana, and Rajdip Nayek. A recursive bayesian neural network for constitutive modeling of sands under monotonic loading, 01 2025.

[44] Kyunghyun Cho, Bart Merrienboer, Dzmitry Bahdanau, Holger Schwenk, Caglar Gulcehre, and Fethi Bougares. Learning phrase representations using rnn encoder-decoder for statistical machine translation. 06 2014. doi: 10.3115/v1/D14-1179.

[45] Maysam Gorji, Mojtaba Mozafar, Julian Heidenreich, Jian Cao, and Dirk Mohr. On the potential of recurrent neural networks for modeling path dependent plasticity. Journal of the Mechanics and Physics ofSolids, 143:103972, 05 2020. doi: 10.1016/j.jmps.2020.103972.

[46] Guang Chen. Recurrent neural networks (rnns) learn the constitutive law of viscoelasticity. Computational Mechanics, 67, 03 2021. doi: 10.1007/s00466-021-01981-y.

[47] Mojtaba Mozafar, Ramin Bostanabad, Wenjin Chen, K. Ehmann, Jingjiao Cao, and M. Bessa. Deep learning predicts path-dependent plasticity. Proceedings of the National Academy of Sciences, 116:26414–26420, 12 2019. doi: 10.1073/pnas.1911815116.

[48] Colin Bonatti, Bekim Berisha, and Dirk Mohr. From cp-ft to cp-rnn: Recurrent neural network surrogate model of crystal plasticity. International Journal of Plasticity, 158: 103430, 09 2022. doi: 10.1016/j.ijplas.2022.103430.

[49] M.A. Maia, I.B.C.M. Rocha, P. Kerfriden, and F.P. van der Meer. Physically recurrent neural networks for path-dependent heterogeneous materials: Embedding constitutive models in a data-driven surrogate. Computer Methods in Applied Mechanics and Engineering, 407:115934, 2023. ISSN 0045-7825. doi: https://doi.org/10.1016/j.cma.2023.115934. URL https://www.sciencedirect.com/science/article/pii/S0045782523000579.

[50] Binyao Guo, Zihan Lin, and QiZhi He. History-aware neural operator: Robust datadriven constitutive modeling of path-dependent materials. Computer Methods in Applied Mechanics and Engineering, 447:118358, 2025. ISSN 0045-7825. doi: https://doi.org/ 10.1016/j.cma.2025.118358. URL https://www.sciencedirect.com/science/article/ pii/S0045782525006309.

[51] Colin Bonatti and Dirk Mohr. On the importance of self-consistency in recurrent neural network models representing elasto-plastic solids. Journal of the Mechanics and Physics of Solids, 158:104697, 2022. ISSN 0022-5096. doi: https://doi.org/ 10.1016/j.jmps.2021.104697. URL https://www.sciencedirect.com/science/article/ pii/S0022509621003161.

[52] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations, 2021. URL https://arxiv.org/abs/2010.08895.

[53] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: learning maps between function

spaces with applications to pdes. J. Mach. Learn. Res., 24(1), January 2023. ISSN 1532- 4435.

[54] Yusuke Yamazaki, Ali Harandi, Mayu Muramatsu, Alexandre Viardin, Markus Apel, Tim Brepols, Stefanie Reese, and Shahed Rezaei. A finite element-based physics-informed operator learning framework for spatiotemporal partial diferential equations on arbitrary domains. Engineering with Computers, 41(1):1–29, August 2024. ISSN 1435-5663. doi: 10.1007/s00366-024-02033-8. URL http://dx.doi.org/10.1007/s00366-024-02033-8.

[55] Reza Najian Asl, Yusuke Yamazaki, Kianoosh Taghikhani, Mayu Muramatsu, Markus Apel, and Shahed Rezaei. A physics-informed meta-learning framework for the continuous solution of parametric pdes on arbitrary geometries, 04 2025.

[56] Gege Wen, Zongyi Li, Kamyar Azizzadenesheli, Anima Anandkumar, and Sally Benson. U-fno – an enhanced fourier neural operator-based deep-learning model for multiphase flow, 09 2021.

[57] Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions, 2020. URL https://arxiv.org/abs/2006.09661.

[58] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred A. Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks, 2019. URL https://arxiv.org/abs/1806.08734.

[59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

[60] Tim Brepols, Stephan Wulfinghof, and Stefanie Reese. Gradient-extended two-surface damage-plasticity: Micromorphic formulation and numerical aspects. International Journal of Plasticity, 97:64–106, 2017. ISSN 0749-6419. doi: https://doi.org/10. 1016/j.ijplas.2017.05.010. URL https://www.sciencedirect.com/science/article/ pii/S0749641916303461.

[61] Tim Brepols, Stephan Wulfinghof, and Stefanie Reese. A Micromorphic Damage-Plasticity Model to Counteract Mesh Dependence in Finite Element Simulations Involving Material Softening, pages 235–255. Springer International Publishing, Cham, 2018. ISBN 978- 3-319-65463-8. doi: 10.1007/978-3-319-65463-8\_12. URL https://doi.org/10.1007/ 978-3-319-65463-8\_12.

[62] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/1711.05101.

[63] James Bergstra, Rémi Bardenet, Yoshua Bengio, and Balázs Kégl. Algorithms for hyperparameter optimization. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 24. Curran

Associates, Inc., 2011. URL https://proceedings.neurips.cc/paper\_files/paper/ 2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf.

[64] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A next-generation hyperparameter optimization framework, 2019. URL https: //arxiv.org/abs/1907.10902.