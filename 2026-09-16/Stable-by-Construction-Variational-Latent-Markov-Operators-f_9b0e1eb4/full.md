# Stable by Construction: Variational Latent Markov Operators for Long-Horizon PDE Prediction

Junyi Liao<sup>†</sup>, Johann Guilleminot<sup>‡</sup>, Vahid Tarokh<sup>†</sup>

<sup>†</sup>Department of Electrical and Computer Engineering, Duke University

<sup>‡</sup>Department of Mechanical Engineering and Materials Science, Duke University

{junyi.liao, johann.guilleminot, vahid.tarokh}@duke.edu

## Abstract

Neural PDE solvers provide eficient surrogates for time-dependent physical systems, but autoregressive prediction over long horizons remains challenging because local errors can induce distribution shift and accumulate under recursive deployment. We develop a variational approach to this problem by introducing latent Markov dynamics in which physical states are represented by latent distributions and evolved through probabilistic transitions. The framework is formulated directly on function spaces and specialized to functional Gaussian models, where structured latent perturbations induce a spectral geometry and variational transition alignment regularizes the learned dynamics. We further analyze how these mechanisms afect autoregressive error propagation, providing a theoretical connection between variational training and long-horizon prediction. We instantiate the framework as the Variational Autoencoding Markov Operator (VAMO), which combines spatially resolved latent fields, structured Gaussian perturbations, and a neural operator transition. Empirically, we demonstrate the efectiveness of VAMO on several fluiddynamics benchmarks with prediction horizons extending substantially beyond those represented during training, where it consistently reduces error accumulation and improves rollout stability over several deterministic and noise-injection baselines. Overall, these results highlight variational modeling as a complementary approach to robust long-horizon neural PDE dynamics.

## 1 Introduction

Partial diferential equations (PDEs) provide a fundamental mathematical formulation of physical systems across science and engineering, but repeated numerical simulation can be costly when solutions are required across many initial conditions, coeficients, or forcing terms. This has motivated data-driven PDE solvers, ranging from physics-informed neural networks (Raissi et al., 2019; Pang et al., 2019) to operator-learning methods such as DeepONet, Fourier neural operator (FNO), OFormer, and PiT (Lu et al., 2021; Li et al., 2021; Li et al., 2022; Chen and Wu, 2024). Operatorlearning methods seek to approximate mappings between function spaces across families of PDE instances, enabling rapid surrogate evaluation after training (Kovachki et al., 2023). Depending on the underlying architecture, neural operators can further support properties such as transfer across spatial discretizations, nonlocal modeling of spatial interactions, or adaptation to irregular computational domains (Li et al., 2021; Kovachki et al., 2023; Li et al., 2023).

Despite these advances, long-horizon prediction of time-dependent PDEs remains challenging, particularly when a learned local evolution operator is deployed recursively beyond the temporal regime represented during training. Autoregressive models are typically trained on ground-truth one-step transitions but recursively consume their own predictions at inference, so small local errors can shift the rollout away from the training distribution and accumulate over time (Sanchez Gonzalez et al., 2020; Brandstetter et al., 2022; Lippe et al., 2023). This dificulty is further exacerbated when training trajectories cover only a limited temporal horizon, while deployment requires evolving the system substantially beyond that horizon (Yin et al., 2023; Micha lowska et al., 2024; Diab and Al Kobaisi, 2025). Existing remedies take several forms. Recurrent or sequence-based temporal architectures augment neural operators with learned temporal dynamics to capture longer-range dependencies (Li et al., 2022; Micha lowska et al., 2024). Multi-step or curriculum-based training more directly mitigates the mismatch between teacher-forced training and autoregressive deployment by progressively exposing the model to longer rollouts or modelgenerated states (Takamoto et al., 2023; Li et al., 2022; Hagnberger et al., 2025). A diferent strategy avoids recursive prediction over a prescribed interval by learning a direct space-time solution map (Li et al., 2021; Wang et al., 2021). Although these strategies mitigate long-horizon prediction errors in some settings, their efectiveness can remain closely tied to the training regime and temporal range represented in the data. Recurrent or sequence-based architectures better capture tempora dependencies but can still sufer from error accumulation under the distribution shift induced by recursive deployment; curriculum or multi-step training reduces the train-test mismatch but may remain tied to the rollout horizons encountered during training; and direct space-time prediction avoids recursion over a prescribed interval but does not naturally extrapolate beyond it. Thus, reliable generalization to substantially longer temporal horizons remains challenging.

At the same time, most neural PDE solvers and operator-learning methods formulate the learned solution map or temporal evolution deterministically, while probabilistic approaches remain comparatively less explored. Existing work includes Bayesian operator learning and uncertainty quantification (Yang et al., 2022; Lin et al., 2023; Garg and Chakraborty, 2023), generative operator models (Rahman et al., 2022), and more recent difusion-based approaches for PDE modeling and temporal prediction (Lippe et al., 2023; Serrano et al., 2024b). More closely related to our formulation, Variational Autoencoding Neural Operators (VANO) extend variational autoencoders to functional data (Seidman et al., 2023), while variational autoencoders have also been developed directly on function spaces with well-posed infinite-dimensional objectives (Bunker et al., 2025). These variational formulations, however, primarily concern static function representations or input-output operator mappings rather than recursively evolving latent dynamics. A variational formulation is particularly appealing in the dynamic setting: representing each physical state by a distribution in latent space allows the learned transition to be trained over neighborhoods of the observed trajectory, while distributional alignment between predicted and encoded next states provides an additional mechanism for regularizing the latent evolution. These properties suggest that variational modeling may ofer a principled way to improve robustness to the perturbations encountered during autoregressive rollout. However, how such probabilistic latent dynamics should be formulated for PDE evolution, and how their variational structure relates to long-horizon deterministic prediction, remain insuficiently understood.

In this work, we study variational modeling of time-dependent PDE dynamics for long-horizon autoregressive prediction. We formulate PDE evolution through probabilistic latent Markov dynamics, where physical states are encoded into latent distributions, propagated through a stochastic transition model, and decoded back to physical space. Stochasticity is used during training to regularize the latent evolution, while inference follows deterministic mean dynamics. We instantiate this framework as the Variational Autoencoding Markov Operator (VAMO), which combines spatially resolved latent fields, structured Gaussian perturbations, and a neural-operator transition. Our main contributions are:

• Variational latent dynamics. We develop a function-space variational formulation for learning temporal PDE evolution through latent Markov dynamics.

• Rollout analysis. We characterize how latent perturbations and variational transition alignment enter the error propagation of deterministic autoregressive rollouts.

• Long-horizon evaluation. We instantiate the framework as VAMO and evaluate it on three fluid-dynamics benchmarks, where it improves long-horizon rollout stability over deterministic latent, direct neural-operator, and noise-injection baselines.

Overall, our study positions variational modeling as a complementary approach to improving the robustness of learned PDE dynamics, rather than merely as a mechanism for probabilistic prediction. By connecting variational training with autoregressive evolution, we aim to broaden the role of probabilistic methods in neural PDE solvers toward long-horizon dynamical modeling.

## 1.1 Related Work

Neural PDE solvers and operator learning. Neural networks have been widely used for PDE approximation through either physics-informed objectives or data-driven surrogate modeling. Physics-informed neural networks (PINNs) enforce governing equations and boundary or initial conditions through the training loss (Raissi et al., 2019; Pang et al., 2019), while operator-learning methods learn mappings between function spaces across families of PDE instances (Lu et al., 2022; Hoop et al., 2022; Kovachki et al., 2023). Representative approaches include DeepONet (Lu et al., 2021), model-reduction-based operator learning (Bhattacharya et al., 2021), graph-based and general neural operators (Li et al., 2020a,b; Kovachki et al., 2023), and the Fourier neura operator (FNO) (Li et al., 2021). Subsequent extensions consider multiple-input operators, deeper architectures, nonlinear manifold representations, physics-informed training, alternative spectral representations, and complex geometries (Jin et al., 2022; Rahman et al., 2023; Seidman et al., 2022; Li et al., 2024; Gupta et al., 2021; Li et al., 2023). Another prominent line develops attentionbased operator architectures, including the Galerkin Transformer (Cao, 2021), LOCA (Kissas et al., 2022), OFormer (Li et al., 2022), GNOT (Hao et al., 2023), ONO (Xiao et al., 2023), PiT (Chen and Wu, 2024), Transolver (Wu et al., 2024), and LNO (Wang and Wang, 2024). Our work builds on this operator-learning paradigm and focuses on stable temporal evolution.

Learning physical dynamics. A related line of work focuses explicitly on learning the temporal evolution of physical systems. Autoregressive simulators commonly learn a local update rule and repeatedly apply it to advance the physical state, including graph-based physical simulators, mesh-based models, neural PDE solvers, and convolutional encoder-decoder surrogates (Sanchez-Gonzalez et al., 2020; Pfaf et al., 2020; Brandstetter et al., 2022; Stachenfeld et al., 2021; Geneva and Zabaras, 2020). Since repeated prediction can amplify local errors and shift the model away from the state distribution encountered during training, prior works have explored perturbing training states, multi-step or curriculum training, adaptive temporal stepping, and explicitly stabilized learned dynamics to improve rollout robustness (Sanchez-Gonzalez et al., 2020; Li et al., 2022; Hagnberger et al., 2025; Wu et al., 2026; Linot et al., 2023; Lippe et al., 2023). Another prominent direction evolves the system in a learned latent space, using recurrent models, continuous latent dynamics, Koopman-inspired representations, or learned latent operators (Wiewel et al., 2019; Lusch et al., 2018; Morton et al., 2018; Pan and Duraisamy, 2020; Yin et al., 2023; Kontolat et al., 2024). Along this direction, Geneva and Zabaras (2022) further combines Koopman-informed embeddings with Transformers for temporal prediction, while OFormer (Li et al., 2022) performs recurrent time-marching on spatially resolved latent representations. More recent approaches such as LNO and CALM-PDE similarly model time-dependent PDEs through compressed latent representations (Wang and Wang, 2024; Hagnberger et al., 2025). Our work is most closely related to this latent-dynamics perspective, but difers in formulating the latent evolution variationally.

Probabilistic and variational modeling. Probabilistic PDE surrogates have been studied primarily for uncertainty quantification. Early work includes Bayesian convolutional encoder-decoder models for stochastic PDEs and physics-constrained generative surrogates (Zhu and Zabaras, 2018; Zhu et al., 2019). Similar ideas have subsequently been extended to operator learning through Bayesian, probabilistic, and variational neural-operator formulations (Yang et al., 2022; Lin et al., 2023; Garg and Chakraborty, 2023; B¨ulte et al., 2025). ore closely related to our formulation, several generative approaches model distributions over function-valued data. Generative Adversarial Neural Operators (GANO) learn distributions directly on function spaces using neural operators (Rahman et al., 2022). Variational Autoencoding Neural Operators (VANO) extend variational autoencoders to functional data (Seidman et al., 2023), while autoencoders have also been formulated directly on function spaces (Bunker et al., 2025). Probabilistic learning of physical systems has additionally been developed directly at the field level, including information-field-theoretic approaches for uncertainty-aware modeling (Alberts and Bilionis, 2023). Difusion-based methods provide another probabilistic generative approach to PDE modeling: difusion models have been formulated as probabilistic neural operators (Haitsiukevich et al., 2024), while Physics-Informed Difusion Models incorporate PDE constraints into difusion training (Bastek et al., 2025). For temporal prediction, PDE-Refiner uses difusion-inspired denoising for long autoregressive rollouts (Lippe et al., 2023), and AROMA employs difusion-based latent dynamics (Serrano et al., 2024a,b).

## 1.2 Roadmap

The remainder of the paper is organized as follows. §2 introduces the time-dependent PDE learning problem, the long-horizon autoregressive prediction setting, and the variational autoencoder preliminaries used throughout the paper. §3 develops variational latent dynamics directly on function spaces, first establishing the abstract variational formulation, then specializing to functional Gaussian models and analyzing the propagation of errors under deterministic mean rollout. §4 introduces the Variational Autoencoding Markov Operator (VAMO), including its spatially resolved architecture, structured Gaussian latent perturbations, and training objective. §5 evaluates the resulting framework on long-horizon fluid-dynamics benchmarks.

Notation. We use calligraphic letters such as $\mathcal { U }$ and $\mathcal { Z }$ for physical and latent state spaces, and bold uppercase letters U and Z for the corresponding function-valued states. Lowercase bold letters u and z denote their finite-dimensional discretizations when needed. For a measurable space $x ,$ $\Delta ( \mathcal { X } )$ denotes the set of probability measures on $\mathcal { X }$ . For two probability measures $P$ and $Q ,$ we write $P \ll Q$ for absolute continuity, $D _ { \mathrm { K L } } ( P \Vert Q )$ for the Kullback-Leibler divergence, and $\mathsf { N } ( m , K )$ for a Gaussian measure with mean m and covariance operator $K .$ . For a linear operator $K$ , let $\| K \| _ { \mathrm { o p } }$ and $\operatorname { T r } ( K )$ denote its operator norm and trace, respectively. Expectations are written as $\mathbb { E } ,$ with subscripts indicating the corresponding distribution when needed.

## 2 Problem Setting and Preliminaries

We first introduce the time-dependent PDE learning problem considered throughout this work and formalize the autoregressive prediction setting. We then briefly review the variational autoencoder framework that motivates the latent probabilistic formulation developed in $\ S 3$

## 2.1 PDE Evolution and Flow Maps

We consider a class of time-dependent partial diferential equations (PDEs) defined on a spatial domain $\Omega \subset \mathbb { R } ^ { d _ { x } }$ . Let

$$
\mathbf { U } ( t ) : \Omega  \mathbb { R } ^ { c }
$$

denote the physical state at time t, where c is the number of physical channels. We abstract the governing dynamics, conditional on fixed problem-specific quantities, as an autonomous evolution equation

$$
\partial _ { t } \mathbf { U } ( t ) = \mathcal { F } ( \mathbf { U } ( t ) ) ,\tag{2.1}
$$

where $\mathcal { F }$ is a generally nonlinear diferential operator. In concrete PDEs, $\mathcal { F }$ may depend on spatial derivatives of U, physical parameters, forcing fields, boundary conditions, or other problem-specific quantities. When these quantities vary across trajectories, we treat them as additional conditioning inputs and suppress them from the notation for simplicity.

For a fixed time increment $h > 0$ , the PDE induces a solution operator, or flow map,

$$
S _ { h } : { \mathbf { U } } ( t ) \mapsto { \mathbf { U } } ( t + h ) .\tag{2.2}
$$

Given discrete times $t _ { n } = n h$ , we write $\mathbf { U } _ { n } : = \mathbf { U } ( t _ { n } )$ , so that the exact discrete-time evolution satisfies

$$
\mathbf { U } _ { n + 1 } = S _ { h } ( \mathbf { U } _ { n } ) .\tag{2.3}
$$

Thus, learning the temporal PDE dynamics can be viewed as learning an approximation to the local flow map $S _ { h }$ from observed trajectories.

Autoregressive rollout. Let the training data consist of trajectories

$$
\mathcal { D } = \left\{ \left( \mathbf { U } _ { 0 } ^ { ( i ) } , \mathbf { U } _ { 1 } ^ { ( i ) } , \ldots , \mathbf { U } _ { N _ { \mathrm { t r a i n } } } ^ { ( i ) } \right) \right\} _ { i = 1 } ^ { N _ { \mathrm { d a t a } } } ,
$$

possibly together with the problem-dependent conditions suppressed above. A local data-driven solver learns the transition ${ \bf U } _ { n } \mapsto { \bf U } _ { n + 1 }$ from adjacent states in these trajectories. Denoting the learned local flow map by $\widehat { S } _ { h }$ , prediction from an initial state $\mathbf { U } _ { 0 }$ is performed autoregressively as

$$
\widehat { \mathbf { U } } _ { 0 } = \mathbf { U } _ { 0 } , \qquad \widehat { \mathbf { U } } _ { n } = \widehat { S } _ { h } ( \widehat { \mathbf { U } } _ { n - 1 } ) , \qquad n = 1 , 2 , \cdots , N _ { \mathrm { t e s t } } .\tag{2.4}
$$

This creates an important distinction between training and deployment. During supervised training on adjacent state pairs, $\widehat { S } _ { h }$ is evaluated on ground-truth states ${ \mathbf { U } } _ { n }$ , whereas during autoregressive rollout it is repeatedly evaluated on its own predictions $\widehat { \mathbf { U } } _ { n }$ . Consequently, local prediction errors can move the rollout away from the state distribution represented in the training trajectories and propagate through subsequent predictions.

Our primary interest is temporal extrapolation, where the available training trajectories cover only a finite horizon while prediction extends substantially beyond it,

$$
N _ { \mathrm { t e s t } } > N _ { \mathrm { t r a i n } } .
$$

This setting has also been studied in recent work on long-time PDE forecasting and temporal extrapolation (Yin et al., 2023; Micha lowska et al., 2024; Diab and Al Kobaisi, 2025). In our experiments, the training and test initial conditions are sampled from the same problem-specific distributions, so the principal extrapolation occurs along the temporal direction. The goal is therefore not only to approximate the local flow map accurately, but also to maintain stable predictions under repeated autoregressive application.

## 2.2 Variational Autoencoders

We briefly review the variational autoencoder (VAE) framework (Kingma and Welling, 2014; Seidman et al., 2023) in a form that will be used throughout the paper. Let U denote the data space and Z the latent space. We denote a data sample by $\mathbf { U } \in \mathcal { U }$ and its latent representation by $\mathbf { Z } \in { \mathcal { Z } }$ A variational autoencoder consists of two main probabilistic components. The encoder $Q _ { \phi }$ defines a conditional distribution

$$
\mathcal { U } \ni \mathbf { U } \mapsto Q _ { \phi } ( \mathrm { d } \mathbf { Z } \mid \mathbf { U } ) \in \Delta ( \mathcal { Z } )
$$

over latent representations $\mathbf { Z } \in { \mathcal { Z } }$ given an observation $\mathbf { U } \in { \mathcal { U } } .$ . Similarly, the decoder $P _ { \psi }$ is a conditional kernel

$$
\mathcal { Z } \ni \mathbf { Z } \mapsto P _ { \psi } ( \mathrm { d } \mathbf { U } \mid \mathbf { Z } ) \in \Delta ( \mathcal { U } )
$$

over observations conditioned on the latent representation. Here $\phi \in \Phi$ and $\psi \in \Psi$ denote the encoder and decoder parameters, respectively.

The generative model is completed by prescribing a reference latent distribution $\mathsf { P } _ { \mathcal { Z } } \in \Delta ( \mathcal { Z } )$ . A latent sample is first drawn according to $\mathbf { Z } \sim \mathsf { P } _ { \mathcal { Z } }$ , and the corresponding observation is generated from $\mathbf { U } \sim P _ { \psi } ( \cdot \mid \mathbf { Z } )$ . The encoder $Q _ { \phi } ( \cdot \mid \mathbf { U } )$ provides a tractable variational approximation to the generally intractable posterior distribution of Z conditioned on U.

To train the model, one maximizes a variational lower bound on the data likelihood. Let $\mathsf { W } _ { \mathcal { U } }$ be a reference measure on $u ,$ and suppose that the decoder likelihood is dominated by $\mathsf { W } _ { \mathcal { U } }$ . Under the usual absolute-continuity conditions, the evidence lower bound (ELBO) takes the form

$$
\log \frac { \mathrm { d } P _ { \psi } ^ { \mathcal { U } } } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) \geq \mathbb { E } _ { \mathbf { Z } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) \right] - D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } ) \parallel \mathsf { P } _ { \mathcal { Z } } \right) ,\tag{2.5}
$$

where

$$
P _ { \psi } ^ { \mathcal { U } } ( \mathrm { d } \mathbf { U } ) = \int _ { \mathcal { Z } } P _ { \psi } ( \mathrm { d } \mathbf { U } \mid \mathbf { Z } ) \mathsf { P } _ { \mathcal { Z } } ( \mathrm { d } \mathbf { Z } )
$$

is the marginal distribution induced by the generative model.

The two terms in (2.5) play complementary roles. The first encourages latent samples produced by the encoder to retain suficient information for reconstruction through the decoder, while the KL term regularizes the encoded distribution toward the prescribed latent prior. Thus, a variational autoencoder combines probabilistic autoencoding with distributional regularization of the latent representation.

While the standard VAE formulation above regularizes the latent representation against a fixed reference distribution $\mathsf { P } _ { \mathcal { Z } } ,$ corresponding to a static latent prior, variational latent-variable models have also been extended to sequential and dynamical settings through learned latent dynamics, including state-space models (Krishnan et al., 2015, 2017; Rangapuram et al., 2018), latent ordinary diferential equations (Rubanova et al., 2019; Yildiz et al., 2019), and latent stochastic diferential equations (Ha et al., 2018; Tzen and Raginsky, 2019; Hasan et al., 2021). Motivated by these perspectives, we replace the static latent prior of a standard VAE with a conditional transition kernel

$$
P _ { \theta } ( d \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) ,
$$

so that variational regularization is imposed on the evolution of the latent state. Our focus difers from these finite-dimensional latent-dynamics models in developing the resulting latent Markov formulation directly on function spaces for time-dependent PDE evolution. In $\ S 3 .$ , we give a detailed construction of this framework.

## 3 Variational Latent Dynamics on Function Spaces

In this section, we develop the theoretical foundation of variational latent dynamics for timedependent PDEs. We first formulate the model abstractly on physical and latent function spaces and characterize the variational objectives governing latent representation and transition alignment in §3.1. Then in §3.2, we specialize to functional Gaussian distributions, where Cameron-Martin theory provides tractable likelihood representations and the covariance structure induces a spectral geometry on latent perturbations and residuals. Finally, in §3.3, we analyze long-horizon error propagation under deterministic mean rollout, highlighting two complementary mechanisms of variational training: noise injection promotes local stability, while variational alignment provides direct control of latent transition errors and their accumulation over the rollout horizon.

## 3.1 Abstract Variational Latent Dynamics

We formulate the latent dynamics directly at the level of function spaces (Bogachev, 1998), before introducing a particular Gaussian parameterization or spatial discretization. Let U denote the space of physical PDE states and $\mathcal { Z }$ the corresponding latent state space.

Assumption 3.1 (State spaces). The physical state space $\mathcal { U }$ and latent state space $\mathcal { Z }$ are real separable Banach spaces, equipped with their Borel σ-algebras.

This setting is suficiently general to include many state spaces arising in PDE modeling (Stuart, 2010; Dashti and Stuart, 2013), while providing the standard measurable structure required to define conditional probability distributions on U and $\mathcal { Z } .$ . In particular, separability ensures that these spaces have the regularity needed for the probabilistic constructions below.

Built on this setting, we study a variational latent formulation of the PDE flow map. Instead of directly learning a deterministic propagator in the physical space, we introduce a latent representation $\mathbf { Z } _ { n } \in \mathcal { Z }$ for each physical state ${ \mathbf { U } } _ { n } \in { \mathcal { U } }$ and model the one-step evolution through the latent pathway

$$
\mathbf { U } _ { n } \xrightarrow { \mathrm { e n c o d e } } \mathbf { Z } _ { n } \xrightarrow { \mathrm { e v o l v e } } \mathbf { Z } _ { n + 1 } \xrightarrow { \mathrm { d e c o d e } } \mathbf { U } _ { n + 1 } .
$$

Along this pathway, an encoder maps the physical field to a latent representation, a latent transition model propagates this representation in time, and a decoder maps the propagated latent state back to the physical space. This formulation provides a probabilistic description of the learned dynamics while allowing the evolution to be regularized in a structured latent space.

Concretely, for the pathway above, we introduce encoder, transition and decoder kernels, parameterized by $\phi \in \Phi , \theta \in \Theta$ and $\psi \in \Psi$ , respectively:

(encoder)

(transition)

(decoder)

$$
\begin{array} { r l } & { \mathcal { U } \ni \mathbf { U } _ { n } \mapsto Q _ { \phi } ( \mathrm { d } \mathbf { Z } _ { n } \left| \mathbf { U } _ { n } \right) \in \Delta \left( \mathcal { Z } \right) , } \\ & { \mathcal { Z } \ni \mathbf { Z } _ { n } \mapsto P _ { \theta } ( \mathrm { d } \mathbf { Z } _ { n + 1 } \left| \mathbf { Z } _ { n } \right) \in \Delta ( \mathcal { Z } ) , } \\ & { \mathcal { Z } \ni \mathbf { Z } _ { n } \mapsto P _ { \psi } ( \mathrm { d } \mathbf { U } _ { n + 1 } \left| \mathbf { Z } _ { n + 1 } \right) \in \Delta ( \mathcal { U } ) . } \end{array}
$$

Together, these components define a probabilistic approximation of the one-step PDE flow map and yield a variational latent Markov model for time-dependent PDEs.

The formulation is independent of the particular physical variables used to represent the underlying PDE state. For example, ${ \mathbf { U } } _ { n }$ may denote the vorticity field in incompressible Navier-Stokes or the density, velocity, and pressure fields in compressible Euler. The same latent probabilistic framework therefore applies across diferent PDE systems through the corresponding physical-state space and data distribution.

Before specializing the model to functional Gaussian kernels, we first establish the variational foundation of the latent Markov formulation at the level of general function-space probability measures. We begin by deriving lower bounds on the one-step predictive likelihood, including a conditional functional ELBO that motivates the reconstruction and latent transition consistency terms. We then study the dynamics term at the population level and characterize the latent Markov transition targeted by its minimization. These results do not rely on any particular parameterization and therefore apply to the general formulation above.

Proposition 3.2 (Functional lower bounds). Let U and Z be Banach spaces. Let $\mathsf { W } _ { \mathcal { U } }$ be a fixed reference measure on U. For a consecutive pair $( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) \in \mathcal { U } \times \mathcal { U }$ , suppose that the encoder, latent transition, and decoder are given by Markov kernels

$$
\begin{array} { r } { Q _ { \phi } ( \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) \in \Delta ( \mathcal { Z } ) , \quad P _ { \theta } ( \mathrm { d } \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) \in \Delta ( \mathcal { Z } ) , \quad P _ { \psi } ( \mathrm { d } \mathbf { U } _ { n + 1 } \mid \mathbf { Z } _ { n + 1 } ) \in \Delta ( \mathcal { U } ) . } \end{array}
$$

Assume that the decoder kernel is absolutely continuous with respect to $\mathsf { W } _ { \mathcal { U } } \mathrm { : }$

$$
P _ { \psi } ( \cdot \mid \mathbf { Z } ) \ll \mathsf { W } _ { \mathcal { U } } .
$$

Then the following lower bounds hold.

(a) (Predictive decoder likelihood bound).

$$
\log \frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot | \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { M } } ( \mathbf { U } _ { n + 1 } ) \geq \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim P _ { \theta } ( \cdot | \mathbf { Z } _ { n } ) , \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { U } _ { n } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { M } } ( \mathbf { U } _ { n + 1 } ) \right] .\tag{3.1}
$$

(b) (Conditional functional ELBO). Assume that the relevant likelihood densities exist and that

$$
Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) \ll P _ { \theta } ( \cdot | \mathbf { Z } _ { n } )
$$

for $Q _ { \phi } ( \cdot | \mathbf { U } _ { n } )$ -almost every $\mathbf { Z } _ { n }$ . Then

$$
\begin{array} { r l } & { \log \frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \geq \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] } \\ & { \qquad - \mathbb { E } _ { \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) \parallel P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ) \right) \right] . } \end{array}
$$

Remark 3.3. Equivalently, the first bound gives the predictive negative log-likelihood objective

$$
\mathcal { L } _ { \mathrm { N L L } } ^ { \mathrm { p r e d } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) = \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim P _ { \theta } ( \cdot | \mathbf { Z } _ { n } ) , \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { U } _ { n } ) } \left[ - \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] .\tag{3.2}
$$

Similarly, the negative conditional ELBO is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E L B O } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) = - \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] } \\ { + \mathbb { E } _ { \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) \parallel P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ) \right) \right] . } \end{array}\tag{3.3}
$$

Notably, the ELBO can be understood in the extended-real sense. If $Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } )$ is not absolutely continuous with respect to $P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } )$ , the corresponding KL divergence is infinite and the lower bound becomes trivial. A finite, nontrivial ELBO therefore requires

$$
Q _ { \phi } ( \cdot \mid { \bf U } _ { n + 1 } ) \ll P _ { \theta } ( \cdot \mid { \bf Z } _ { n } )
$$

for $Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } )$ -almost every $\mathbf { Z } _ { n }$

The conditional ELBO in (3.3) separates two complementary learning objectives. The reconstruction term

$$
- \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) } \big [ \log P _ { \psi } ( \mathbf { U } _ { n + 1 } \mid \mathbf { Z } _ { n + 1 } ) \big ] ,
$$

encourages latent samples drawn from $Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } )$ to retain the information needed to reconstruct the physical state ${ \bf U } _ { n + 1 }$ . It therefore promotes consistency between the encoder and decoder and ensures that the latent variables remain informative representations of the underlying PDE states. The second term,

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) \parallel P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ) \right) \right] , } \end{array}
$$

enforces consistency of the latent dynamics. Given a latent state $\mathbf { Z } _ { n }$ encoded from ${ \mathbf { U } } _ { n }$ , the transition kernel $P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } )$ predicts a distribution over the next latent state. The KL divergence aligns this prediction with the posterior distribution obtained by encoding the true next state ${ \bf U } _ { n + 1 }$ . Thus, the variational objective couples two complementary requirements: physical reconstruction and latent transition alignment. The preceding ELBO gives a sample-level variational objective. We next characterize its latent dynamics term at the population level and identify the transition kernel targeted by its minimization.

Proposition 3.4 (Population decomposition of the ELBO dynamics term). Let $\rho$ be a probability measure on consecutive PDE states $( \mathbf { U } , \mathbf { U } ^ { + } ) \in \mathcal { U } \times \mathcal { U }$ , and let $Q _ { \phi } ( \cdot \mid \mathbf { U } )$ be an encoder kernel. For a transition kernel P from $\mathcal { Z }$ to $\mathcal { Z } ,$ define the population dynamics loss

$$
\begin{array} { r } { \mathcal { R } ( P ) = \mathbb { E } _ { ( \mathbf { U } , \mathbf { U } ^ { + } ) \sim \rho } \mathbb { E } _ { \mathbf { Z } \sim Q _ { \phi } ( \cdot | \mathbf { U } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot | \mathbf { U } ^ { + } ) \| P ( \cdot | \mathbf { Z } ) \right) \right] . } \end{array}
$$

Let $\Gamma _ { \phi }$ be the joint law of $( { \bf U } , { \bf U } ^ { + } , { \bf z } , { \bf Z } ^ { + } )$ induced by

$$
( { \bf U } , { \bf U } ^ { + } ) \sim \rho , \qquad { \bf Z } \sim Q _ { \phi } ( \cdot \mid { \bf U } ) , \qquad { \bf Z } ^ { + } \sim Q _ { \phi } ( \cdot \mid { \bf U } ^ { + } ) ,
$$

and let $\Pi _ { \phi } ( \cdot \mid \mathbf { Z } )$ be the conditional law of $\mathbf { Z } ^ { + }$ given Z under $\Gamma _ { \phi }$ . Then

$$
\begin{array} { r } { \mathcal { R } ( P ) = \mathbb { E } _ { ( \mathbf { Z } , \mathbf { U } ^ { + } ) \sim \Gamma _ { \phi } } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot | \mathbf { U } ^ { + } ) \| \Pi _ { \phi } ( \cdot | \mathbf { Z } ) \right) \right] + \mathbb { E } _ { \mathbf { Z } \sim \Gamma _ { \phi } } \left[ D _ { \mathrm { K L } } \left( \Pi _ { \phi } ( \cdot | \mathbf { Z } ) \| P ( \cdot | \mathbf { Z } ) \right) \right] . } \end{array}\tag{3.4}
$$

Consequently, among all Markov kernels $P ,$ the population dynamics loss is minimized by

$$
P ^ { \star } ( \cdot | \mathbf { Z } ) = \Pi _ { \phi } ( \cdot | \mathbf { Z } )
$$

for $\Gamma _ { \phi } .$ -almost every Z.

Proposition 3.4 separates the population dynamics loss into an encoder-induced term and a transition-model mismatch:

• The first term in $( 3 . 4 )$ depends only on the encoder and the distribution of consecutive physical states. It quantifies the dispersion of the encoded next-state posterior $Q _ { \phi } ( \cdot \mid { \bf U } ^ { + } )$ around the aggregated conditional latent transition $\Pi _ { \phi } ( \cdot \mid \mathbf { Z } )$ and is therefore irreducible when optimizing over the transition kernel $P .$

• The second term is the only component that depends on P. It measures the discrepancy between the learned Markov transition $P ( \mathbf { \cdot } \mid \mathbf { Z } )$ and the encoder-induced conditional transition $\Pi _ { \phi } ( \cdot \mathrm { ~ \bf ~ \vert ~ Z ) ~ }$ . Hence, for a fixed encoder, minimizing the population dynamics objective is equivalent to matching the learned transition to $\Pi _ { \phi }$ almost surely under the latent marginal distribution. In particular, if the model class $\{ P _ { \theta } \} _ { \theta \in \Theta }$ contains $\Pi _ { \phi }$ , any global population minimizer satisfies

$$
P _ { \theta ^ { \star } } ( \cdot \mid \mathbf { Z } ) = \Pi _ { \phi } ( \cdot \mid \mathbf { Z } )
$$

up to $\Gamma _ { \phi } .$ -null sets. Therefore, rather than identifying the original PDE generator $\mathcal { F }$ directly, the ELBO dynamics term encourages the model to learn the Markov transition induced by the encoder on latent representations of true PDE trajectories.

The preceding results characterize the variational latent dynamics without imposing a specific form on the underlying kernels. We now specialize this framework to functional Gaussian kernels and study the covariance-induced geometry of the latent dynamics through their Cameron-Martin spaces and spectral structure.

## 3.2 Functional Gaussian Latent Markov Model

We now specialize the abstract variational formulation to Gaussian distributions on function spaces (Bogachev, 1998; Kuo, 2006; Da Prato and Zabczyk, 2014; Bunker et al., 2025). This specialization serves two purposes. First, the Cameron-Martin theorem converts the Radon-Nikodym derivatives in the functional variational objective into tractable reconstruction and transition terms. Second, the covariance operators induce a geometry on the latent space that determines how perturbations and transition residuals are weighted across diferent functional directions.

We briefly review the necessary background on Gaussian measures and Cameron-Martin spaces in Appendix $\ S \mathrm { A . 1 }$ . Recall that a functional Gaussian measure $\mathsf { N } ( m , K )$ is characterized by its mean element m and covariance operator K. Accordingly, we consider the functional Gaussian parameterization

$$
\begin{array} { r l } { \mathrm { ( e n c o d e r ) } } & { \quad Q _ { \phi } ( \cdot  { | \textbf { U } } ) =  { \mathsf { N } } \big ( m _ { \phi } (  { \mathbf { U } } ) , K _ { \phi } (  { \mathbf { U } } ) \big ) , } \\ { \mathrm { ( t r a n s i t i o n ) } } & { \quad P _ { \theta } ( \cdot  { | \textbf { Z } } ) =  { \mathsf { N } } \big ( T _ { \theta } (  { \mathbf { Z } } ) , \alpha _ { \theta } (  { \mathbf { Z } } ) ^ { 2 } K \big ) , } \\ { \mathrm { ( d e c o d e r ) } } & { \quad P _ { \psi } (  { \mathbf { \cdot } }  { | \textbf { Z } } ) =  { \mathsf { N } } \big ( D _ { \psi } (  { \mathbf { Z } } ) , K _ { \psi } \big ) . } \end{array}
$$

Here $m _ { \phi }$ denotes the encoder mean, $T _ { \theta }$ the mean latent transition, and $D _ { \psi }$ the decoder mean. The state-dependent covariance $K _ { \phi } ( \mathbf { U } )$ describes uncertainty in the encoded representation, while K specifies the spatial structure of the latent transition noise and $\alpha _ { \theta } ( { \bf Z } ) > 0$ controls its amplitude. The decoder covariance $K _ { \psi }$ is fixed.

These Gaussian distributions are defined on the underlying function spaces rather than on a particular spatial discretization. The finite-dimensional parameterizations used in computation are introduced later in §4.2. Here we focus on the function-space structure induced by the corresponding Gaussian measures.

Cameron-Martin representation of the likelihood. The likelihood terms in the functional variational bounds of Proposition 3.2 are expressed through Radon-Nikodym derivatives with respect to reference measures. For Gaussian kernels, the Cameron-Martin theorem (Stroock, 2010) gives an explicit representation of these derivatives in terms of the geometry induced by the covariance operator. We first record this consequence for the decoder likelihood.

Proposition 3.5 (Cameron-Martin form of the decoder likelihood). Let U be a separable Banach space and let $\mathsf { W } _ { \mathcal { U } }$ be a centered Gaussian measure on U. Denote its Cameron-Martin space by $\mathcal { H } _ { \mathcal { U } }$ continuously embedded in U. Suppose the decoder kernel is given by the Cameron-Martin shift

$$
P _ { \psi } ( \mathbf { d U } \mid \mathbf { Z } ) = \mathsf { W } _ { \mathcal { U } } ^ { D _ { \psi } ( \mathbf { Z } ) } ( \mathbf { d U } ) ,
$$

where $D _ { \psi } ( \mathbf { Z } ) \in \mathcal { H } _ { \mathcal { U } }$ . Then

$$
P _ { \psi } ( \cdot \mid \mathbf { Z } ) \ll \mathsf { W } _ { \mathcal { U } } ,
$$

and the Radon-Nikodym derivative is

$$
\frac { \mathrm { d } P _ { \psi } ( \cdot  { | { \bf \delta Z } } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) = \exp \left( \langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle ^ { \sim } - \frac { 1 } { 2 } \| D _ { \psi } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } \right) ,\tag{3.5}
$$

where $\langle \cdot , \cdot \rangle ^ { \sim }$ denotes the Paley-Wiener map associated with $\mathsf { W } _ { \mathcal { U } }$ . Consequently, the decoder negative log-likelihood with respect to $\mathsf { W } _ { \mathcal { U } }$ is

$$
- \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) = \frac { 1 } { 2 } \| D _ { \psi } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } - \langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle ^ { \sim } .\tag{3.6}
$$

In particular, if $\mathbf { U } \in \mathcal { H } _ { \mathcal { U } }$ , then

$$
\langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle \tilde { \mathbf { \Psi } } = \langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle _ { \mathcal { H } _ { M } } ,
$$

and therefore

$$
- \log \frac { \mathrm { d } P _ { \psi } ( \cdot  { | \textbf { Z } ) } } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } (  { \mathbf { U } } ) = \frac { 1 } { 2 } \|  { \mathbf { U } } - D _ { \psi } (  { \mathbf { Z } } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } - \frac { 1 } { 2 } \|  { \mathbf { U } } \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } .\tag{3.7}
$$

Thus, up to a term depending only on the observed field U, the Gaussian decoder negative loglikelihood is equivalent to the squared Cameron-Martin reconstruction error.

Remark 3.6 (Deterministic decoder mean). The probabilistic decoder is parameterized through the deterministic mean map $D _ { \psi }$ , while the observation uncertainty is represented by the fixed Gaussian reference measure. Proposition 3.5 therefore connects the functional Gaussian likelihood to the deterministic decoder used in computation. In particular, after finite-dimensional discretization with isotropic covariance, the Cameron-Martin reconstruction term reduces to the usual squared Euclidean reconstruction loss.

Spectral geometry of the latent transition. The same Gaussian structure also determines how latent transition errors are measured. We now assume that $\mathcal { Z }$ is a separable Hilbert space and consider a positive, self-adjoint, trace-class covariance operator, which admits a spectral decomposition and the corresponding Karhunen-Lo\`eve representation for Gaussian random elements (Da Prato and Zabczyk, 2014; Bogachev, 1998). The resulting Cameron-Martin norm reveals how the covariance spectrum assigns diferent weights to diferent latent directions.

Proposition 3.7 (Spectral geometry of the latent transition). Let Z be a separable Hilbert space and let linear operator $K : \mathcal { Z } \to \mathcal { Z }$ be self-adjoint, positive-definite, and trace-class. Let $\{ ( \lambda _ { i } , \mathbf { E } _ { i } ) \} _ { i \geq 1 }$ be an eigensystem of $K ,$ , where $\{ { \bf E } _ { i } \} _ { i \ge 1 }$ is an orthonormal basis of ${ \mathcal { Z } } ,$ eigenvalues $\lambda _ { i } > 0 ,$ , and $\textstyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } < \infty$ . Fix

$$
\mathsf { W } _ { \mathcal { Z } } = \mathsf { N } ( 0 , K )
$$

and consider the latent transition

$$
P _ { \theta } ( \cdot \mid \mathbf { Z } ) = \mathsf { N } \big ( T _ { \theta } ( \mathbf { Z } ) , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K \big ) ,
$$

where $T _ { \theta } ( \mathbf { Z } ) \in \mathcal { H } _ { K }$ and $\alpha _ { \theta } ( { \bf Z } ) > 0$ . Then:

(a) The Cameron-Martin space associated with $\mathsf { W } _ { \mathcal { Z } }$ is

$$
\mathcal { H } _ { K } = \left\{ { \bf R } = \sum _ { i = 1 } ^ { \infty } r _ { i } { \bf E } _ { i } : \sum _ { i = 1 } ^ { \infty } \frac { | r _ { i } | ^ { 2 } } { \lambda _ { i } } < \infty \right\} ,
$$

with norm

$$
\| { \bf R } \| _ { \mathcal { H } _ { K } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \frac { | r _ { i } | ^ { 2 } } { \lambda _ { i } } .\tag{3.8}
$$

(b) The transition admits the Karhunen–Lo\`eve representation

$$
\mathbf { Z } ^ { + } = T _ { \theta } ( \mathbf { Z } ) + \alpha _ { \theta } ( \mathbf { Z } ) \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i } , \qquad \xi _ { i } \overset { \mathrm { i . i . d . } } { \sim } \mathsf { N } ( 0 , 1 ) ,\tag{3.9}
$$

where the series converges in $L ^ { 2 } ( \Omega ; \mathcal { Z } )$ and almost surely in Z. Moreover, for ${ \bf Z } ^ { + } \in \mathcal { H } _ { K }$

$$
- \log \frac { \mathrm { d } P _ { \theta } ( \cdot \mid \mathbf { Z } ) } { \mathrm { d } \mathsf { N } ( 0 , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K ) } ( \mathbf { Z } ^ { + } ) = \frac { 1 } { 2 \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } } \| \mathbf { Z } ^ { + } - T _ { \theta } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { K } } ^ { 2 } - \frac { 1 } { 2 \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } } \| \mathbf { Z } ^ { + } \| _ { \mathcal { H } _ { K } } ^ { 2 } .\tag{3.10}
$$

Remark 3.8 (Spectral-coordinate interpretation). Equivalently, writing

$$
T _ { \boldsymbol { \theta } } ( { \mathbf { Z } } ) = \sum _ { i = 1 } ^ { \infty } t _ { i } ( { \mathbf { Z } } ) { \mathbf { E } } _ { i } , \qquad { \mathbf { Z } } ^ { + } = \sum _ { i = 1 } ^ { \infty } z _ { i } ^ { + } { \mathbf { E } } _ { i } ,
$$

the representation (3.9) gives

$$
\begin{array} { r l r } { z _ { i } ^ { + } = t _ { i } ( { \bf Z } ) + \alpha _ { \theta } ( { \bf Z } ) \sqrt { \lambda _ { i } } \xi _ { i } , } & { { } } & { \mathrm { V a r } ( z _ { i } ^ { + } \mid { \bf Z } ) = \alpha _ { \theta } ( { \bf Z } ) ^ { 2 } \lambda _ { i } . } \end{array}
$$

Hence the covariance spectrum simultaneously determines the directions and scales of the injected transition noise and the geometry of the corresponding Cameron-Martin residual. Directions with smaller $\lambda _ { i }$ receive less stochastic variation and a larger weight $1 / \lambda _ { i }$ in the residual norm, whereas directions with larger $\lambda _ { i }$ are perturbed more strongly and penalized less.

A particularly relevant specialization arises when the latent space is a periodic function space and the covariance operator is diagonal in the Fourier basis, as for Laplacian-resolvent covariances (Stuart, 2010; Dashti and Stuart, 2013). In this setting, the spectral weighting induced by the Cameron-Martin norm admits a direct Sobolev interpretation.

Corollary 3.9 (Sobolev geometry of latent residuals). Under the setting of Proposition 3.7, suppose

$$
\mathcal { Z } = L ^ { 2 } ( \mathbb { T } ^ { d _ { \boldsymbol { x } } } ; \mathbb { R } ^ { d _ { \boldsymbol { z } } } )
$$

and

$$
K = ( I - \ell ^ { 2 } \Delta ) ^ { - s } , \qquad \ell > 0 , \qquad s > \frac { d _ { x } } { 2 } ,
$$

where the Laplacian acts componentwise. With the Fourier basis

$$
\phi _ { m } ( x ) = e ^ { 2 \pi i m \cdot x } , \qquad m \in \mathbb { Z } ^ { d _ { x } } ,
$$

the covariance eigenvalues are

$$
\lambda _ { m } = \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } .
$$

Consequently,

$$
\big \| \mathbf Z ^ { + } - T _ { \theta } ( \mathbf Z ) \big \| _ { \mathcal { H } _ { K } } ^ { 2 } = \sum _ { m \in \mathbb Z ^ { d _ { \boldsymbol { x } } } } \sum _ { a = 1 } ^ { d _ { \boldsymbol { z } } } \big ( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \big ) ^ { s } \Big | \widehat { Z } _ { a } ^ { + } ( m ) - \widehat { T _ { \theta } ( \mathbf Z ) } _ { a } ( m ) \Big | ^ { 2 } .\tag{3.11}
$$

Thus the Cameron-Martin space $\mathcal { H } _ { K }$ is $H ^ { s } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { R } ^ { d _ { z } } )$ as a set, with an equivalent length-scaleweighted Sobolev norm. In particular, higher-frequency latent residuals receive increasingly large weight as the covariance spectrum decays.

## 3.3 Error Propagation in Variational Latent Dynamics

We now study how the variational training objective afects the stability of long-horizon latent rollout. We first make explicit the one-step objective analyzed below. Let $\mu _ { n }$ denote the joint distribution of consecutive physical states $\left( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } \right)$ , and consider the Gaussian encoder and transition

$$
\widetilde { \mathbf { Z } } _ { n } = m _ { \phi } ( \mathbf { U } _ { n } ) + \boldsymbol { \Xi } _ { n } , \qquad \boldsymbol { \Xi } _ { n } \sim \mathsf { N } ( 0 , K _ { \phi } ( \mathbf { U } _ { n } ) ) ,
$$

and

$$
\begin{array} { r } { \widetilde { \mathbf { Z } } _ { n } ^ { + } = T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) + \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \mathbf { G } _ { n } , \qquad \mathbf { G } _ { n } \sim \mathsf { N } ( 0 , K ) . } \end{array}
$$

For notational convenience, define the decoder negative log-likelihood

$$
\ell _ { \psi } ( \mathbf { U } ; \mathbf { Z } ) : = - \log \frac { \mathrm { d } P _ { \psi } ( \mathbf { \cdot } \mid \mathbf { Z } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) .
$$

For a consecutive pair $\left( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } \right)$ , define

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e d } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) : = \mathbb { E } _ { \Xi _ { n } , \mathbf { G } _ { n } } \left[ \ell _ { \psi } ( \mathbf { U } _ { n + 1 } ; \widetilde { \mathbf { Z } } _ { n } ^ { + } ) \right] , } \\ & { \mathcal { L } _ { \mathrm { K L } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) : = \mathbb { E } _ { \widetilde { \mathbf { Z } } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { U } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) \| P _ { \theta } ( \cdot | \widetilde { \mathbf { Z } } _ { n } ) \right) \right] , } \\ & { \qquad \mathcal { L } _ { \mathrm { r e c } } ( \mathbf { U } _ { n } ) : = \mathbb { E } _ { \widetilde { \mathbf { Z } } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { U } _ { n } ) } \left[ \ell _ { \psi } ( \mathbf { U } _ { n } ; \widetilde { \mathbf { Z } } _ { n } ) \right] . } \end{array}
$$

The population training objective takes the form

$$
\begin{array} { r } { \mathcal { L } ^ { ( n ) } ( \phi , \theta , \psi ) = \mathbb { E } _ { ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) \sim \mu _ { n } } \left[ \mathcal { L } _ { \mathrm { p r e d } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) + \beta \mathcal { L } _ { \mathrm { K L } } ( \mathbf { U } _ { n } , \mathbf { U } _ { n + 1 } ) + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } ( \mathbf { U } _ { n + 1 } ) \right] , } \end{array}\tag{3.12}
$$

with the empirical objective obtained by averaging over observed one-step pairs.

By Proposition 3.5, whenever the corresponding physical state belongs to $\mathcal { H } _ { \mathcal { U } }$

$$
\ell _ { \psi } ( \mathbf { U } ; \mathbf { Z } ) = \frac { 1 } { 2 } \| \mathbf { U } - D _ { \psi } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } + c ( \mathbf { U } ) ,
$$

where c(U) is independent of the model parameters. Hence the decoder likelihood terms are equivalent, for optimization purposes, to squared Cameron-Martin reconstruction errors, up to additive data-dependent constants.

The objective (3.12) can be viewed as a weighted combination of the two variational bounds introduced in §3.1. The predictive term corresponds to the negative decoder-likelihood bound in Proposition 3.2(a), while the reconstruction and KL terms correspond to the two components of the negative conditional ELBO in Proposition 3.2(b). The coeficients $\beta$ and $\lambda _ { \mathrm { { r e c } } }$ allow for relative weighting of these terms, including the scaling induced by the fixed decoder covariance.

For convenience, our subsequent rollout analysis is stated instead in the ambient physical-state norm $\| \cdot \| u$ . The two geometries are compatible because the Cameron-Martin space is continuously embedded in $\mathcal { U } \mathrm { : }$

$$
\mathcal { H } _ { \mathcal { U } } \hookrightarrow \mathcal { U } , \qquad \| \mathbf { v } \| _ { \mathcal { U } } \le C _ { \mathcal { U } } \| \mathbf { v } \| _ { \mathcal { H } _ { \mathcal { U } } } , \quad \mathbf { v } \in \mathcal { H } _ { \mathcal { U } } ,
$$

for some $C _ { U } > 0$ . Thus, whenever the relevant residual lies in $\mathcal { H } _ { \mathcal { U } }$ , likelihood control also yields control in the ambient norm used for rollout errors. After finite-dimensional discretization, the isotropic decoder covariance used in our implementation reduces these likelihood terms to the squared Euclidean losses described in $\ S 4 . 3$

Deterministic rollout. At inference time, we deploy the deterministic mean dynamics. Given an initial physical state $\mathbf { U } _ { 0 }$ , the rollout is initialized and propagated according to

$$
\widehat { \mathbf { Z } } _ { 0 } = m _ { \phi } ( \mathbf { U } _ { 0 } ) , \quad \widehat { \mathbf { Z } } _ { n } = T _ { \theta } ( \widehat { \mathbf { Z } } _ { n - 1 } ) , \quad \widehat { \mathbf { U } } _ { n } = D _ { \psi } ( \widehat { \mathbf { Z } } _ { n } ) , \qquad n = 1 , 2 , \cdots , T / \Delta t .\tag{3.13}
$$

Hence the stochasticity in (3.12) is used during training, whereas the deployed solver follows the deterministic mean latent dynamics. Our analysis below explains how this stochastic training objective can nevertheless control the error accumulated by the deterministic rollout.

We identify two complementary mechanisms. First, sampling from the encoder and transition distributions replaces pointwise fitting by training over Gaussian neighborhoods in latent space, providing a noise-injection regularization efect. Second, the variational KL term directly controls the mismatch between the learned transition and the encoded distribution of the true next state. We study these two mechanisms separately below.

## 3.3.1 Noise Injection and Local Stability

For the analysis below, let $( \mathbf { U } _ { 0 } ^ { \star } , \ldots , \mathbf { U } _ { N } ^ { \star } ) \sim \mu$ be a reference trajectory, and define its encoder-mean latent representation by

$$
\begin{array} { r } { \mathbf { Z } _ { n } ^ { \star } : = m _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) . } \end{array}
$$

Specializing the stochastic pathway above to this trajectory, let

$$
\begin{array} { r } { \widetilde { \mathbf Z } _ { n } = \mathbf Z _ { n } ^ { \star } + \boldsymbol { \Xi } _ { n } , \qquad \widetilde { \mathbf Z } _ { n } ^ { + } = T _ { \theta } ( \widetilde { \mathbf Z } _ { n } ) + \alpha _ { \theta } ( \widetilde { \mathbf Z } _ { n } ) G _ { n } , } \end{array}
$$

where

$$
\Xi _ { n } \sim \mathsf { N } ( 0 , K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) ) , \qquad G _ { n } \sim \mathsf { N } ( 0 , K ) ,
$$

are conditionally independent given the reference trajectory.

We first quantify the local sensitivity probed by stochastic sampling in the predictive pathway. Noise injection during training has long been associated with regularization of the learned mapping (Bishop, 1995), and has also been used in learned physical simulators to improve robustness under autoregressive rollout (Sanchez-Gonzalez et al., 2020). In our setting, the stochastic objective evaluates the transition and decoder over Gaussian neighborhoods of the reference latent trajectory. To characterize this efect, define the noisy one-step prediction error

$$
\begin{array} { r } { \epsilon _ { n } : = \left( \mathbb { E } \left[ \left\| D _ { \psi } ( \widetilde { \mathbf { Z } } _ { n } ^ { + } ) - \mathbf { U } _ { n + 1 } ^ { \star } \right\| _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } , } \end{array}\tag{3.14}
$$

the encoder-noise sensitivity of the latent transition

$$
\eta _ { n } : = \left( \mathbb { E } \left[ \left. T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) - T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) \right. _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } ,\tag{3.15}
$$

and the transition-noise sensitivity of the decoder

$$
\rho _ { n } : = \left( \mathbb { E } \left[ \left\| D _ { \psi } ( \widetilde { \mathbf { Z } } _ { n } ^ { + } ) - D _ { \psi } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \right\| _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } .\tag{3.16}
$$

Here the expectations are taken jointly over the reference trajectory and the corresponding Gaussian perturbations. The quantity $\eta _ { n }$ directly measures the sensitivity of the latent transition to perturbations of its encoded input, whereas $\rho _ { n }$ measures the sensitivity of the decoder to perturbations around the predicted latent state.

The following result makes this local-stability interpretation explicit by bounding the two sensitivity terms in terms of the noise magnitude and the regularity of the learned maps.

Proposition 3.10 (Gaussian sensitivity bounds). Let the setting and notation of §3.3.1 hold. Assume that

$$
\begin{array} { r } { \mathrm { T r } \big ( K _ { \phi } ( { \bf U } ) \big ) \le \varsigma , \qquad \alpha _ { \theta } ( { \bf Z } ) \le \bar { \alpha } } \end{array}
$$

on the relevant regions, and suppose that $T _ { \theta } : \mathcal { Z }  \mathcal { Z }$ and $D _ { \psi } : \mathcal { Z } \to \mathcal { U }$ are Lipschitz with constants $\Lambda _ { T }$ and $\Lambda _ { D }$ , respectively. Then

$$
\eta _ { n } \leq \Lambda _ { T } \sqrt \varsigma , \qquad \rho _ { n } \leq \Lambda _ { D } \bar { \alpha } \sqrt { \mathrm { T r } ( K ) } .\tag{3.17}
$$

Suppose, in addition, that $T _ { \theta }$ is Fr´echet diferentiable on the relevant encoder-mean neighborhoods and satisfies the uniform second-order remainder bound

$$
\| T _ { \theta } ( \mathbf { z } + \mathbf { h } ) - T _ { \theta } ( \mathbf { z } ) - J _ { T _ { \theta } } ( \mathbf { z } ) \mathbf { h } \| _ { \mathcal { Z } } \leq \frac { M _ { T } } { 2 } \| \mathbf { h } \| _ { \mathcal { Z } } ^ { 2 } .
$$

Then

$$
\eta _ { n } \leq \left( \mathbb { E } \left[ \| J _ { T _ { \theta } } ( \mathbf { Z } _ { n } ^ { \star } ) \| _ { \mathrm { o p } } ^ { 2 } \right] \varsigma \right) ^ { 1 / 2 } + \frac { \sqrt { 3 } M _ { T } } { 2 } \varsigma .\tag{3.18}
$$

Likewise, suppose that $D _ { \psi }$ is Fr´echet diferentiable on the relevant transition-noise neighborhoods and satisfies

$$
\left\| D _ { \psi } ( \mathbf { z } + \mathbf { h } ) - D _ { \psi } ( \mathbf { z } ) - J _ { D _ { \psi } } ( \mathbf { z } ) \mathbf { h } \right\| _ { \mathcal { U } } \leq \frac { M _ { D } } { 2 } \| \mathbf { h } \| _ { \mathcal { Z } } ^ { 2 } .
$$

Then

$$
\rho _ { n } \leq \overline { { \alpha } } \left( \mathbb { E } \left[ \| J _ { D _ { \psi } } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \| _ { \mathrm { o p } } ^ { 2 } \right] \right) ^ { 1 / 2 } \sqrt { \mathrm { T r } K } + \frac { \sqrt { 3 } M _ { D } \overline { { \alpha } } ^ { 2 } } { 2 } \mathrm { T r } K .\tag{3.19}
$$

Proposition 3.10 gives two complementary views of the sensitivity induced by Gaussian perturbations. The Lipschitz bounds in (3.17) provide global control in terms of the overall noise magnitude, whereas the local estimates (3.18)-(3.19) show that, for small perturbations, the sensitivities are governed primarily by the local Fr´echet derivatives of the transition and decoder. In particular, $\eta _ { n }$ measures the response of $T _ { \theta }$ to encoder perturbations, while $\rho _ { n }$ measures the response of $D _ { \psi }$ to perturbations around the predicted latent state. The higher-order terms vanish with the corresponding noise scales.

We can interpret how stochastic training acts on these sensitivity terms from an optimization perspective. The predictive objective evaluates

$$
D _ { \psi } \left( T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } + \Xi _ { n } ) + \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \mathbf { G } _ { n } \right)
$$

over Gaussian neighborhoods of the nominal latent trajectory rather than only at their centers. Large local variations of $T _ { \theta }$ in the encoder-noise directions or of $D _ { \psi }$ in the transition-noise directions can therefore increase the noisy prediction error. Through the reparameterized samples, back-propagation updates the model using errors evaluated throughout these neighborhoods. Thus, although the sensitivity terms are not explicitly penalized, noise-injection training implicitly encourages local stability in the regions and directions explored by the Gaussian perturbations.

## 3.3.2 Variational Alignment and Rollout Control

The previous subsection studies how stochastic sampling regularizes the local sensitivity of the learned transition and decoder. We now turn to the variational KL term. At the distributional level, Proposition 3.4 shows that this term aligns the learned latent transition with the encoderinduced dynamics of true PDE trajectories. Under the Gaussian parameterization, this probabilistic alignment further yields quantitative control of the corresponding latent means. The following lemma makes this connection precise by bounding the discrepancy between the transition mean and the mean of a target latent distribution in terms of their KL divergence.

Lemma 3.11 (KL control of latent means). Let

$$
P _ { \theta } ( \cdot \mid \mathbf { Z } ) = \mathsf { N } \left( T _ { \theta } ( \mathbf { Z } ) , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K \right) ,
$$

where K is positive, self-adjoint, and trace-class on Z. Let $Q$ be any probability measure on $\mathcal { Z }$ with finite second moment and mean

$$
m _ { Q } : = \int _ { \mathcal Z } y \mathrm { d } Q ( y ) .
$$

Assume that

$$
D _ { \mathrm { K L } } \left( Q \parallel P _ { \theta } ( \cdot \mid { \bf Z } ) \right) < \infty .
$$

Then

$$
\begin{array} { r } { \| m _ { Q } - T _ { \theta } ( \mathbf { Z } ) \| _ { \mathcal { Z } } \leq \alpha _ { \theta } ( \mathbf { Z } ) \sqrt { 2 \| K \| _ { \mathrm { o p } } D _ { \mathrm { K L } } \left( Q \parallel P _ { \theta } ( \cdot | \mathbf { Z } ) \right) } . } \end{array}
$$

Lemma 3.11 converts the dynamic KL divergence into quantitative control of the discrepancy between latent means. In particular, when the transition distribution is Gaussian, a small KL divergence between the encoded next-state distribution and the predicted transition distribution implies that their mean states must also be close. This is the key step that connects the probabilistic alignment enforced by the variational objective to the deterministic mean dynamics used at inference. The proof of Lemma 3.11 relies on a Gaussian transportation inequality (Riedel, 2017) and is deferred to Appendix §B.4.

We now introduce the remaining population quantities needed for the rollout analysis. Motivated by Lemma 3.11, for the random reference trajectory, define the dynamic KL error

$$
\begin{array} { r } { \kappa _ { n } : = \left( \mathbb { E } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ^ { \star } ) \parallel P _ { \theta } ( \cdot \mid \widetilde { \mathbf { Z } } _ { n } ) \right) \right] \right) ^ { 1 / 2 } . } \end{array}
$$

Thus, $\kappa _ { n }$ measures, at the population level, how well the learned transition distribution from the perturbed current latent state matches the encoder distribution of the true next physical state. It is the direct population counterpart of the dynamic KL term appearing in the training objective.

In addition, we do not assume exact reconstruction of the physical state from its encoder mean, and therefore define

$$
\delta _ { n } : = \left( \mathbb { E } \left[ \| D _ { \psi } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } .
$$

The quantity $\delta _ { n }$ measures the discrepancy introduced when the encoder-mean latent state is decoded back to the physical space. Here the expectations are taken over the reference trajectory and, where applicable, the encoder perturbation.

Together with the sensitivity quantities studied in §3.3.1, we have the necessary ingredients for analyzing deterministic autoregressive rollout. The following proposition characterizes how the corresponding one-step discrepancies propagate and accumulate over the rollout horizon.

Proposition 3.12 (Population KL-aware rollout control with predictive refinement). Let the setting and notation of §3.3.1 hold. Assume that

$$
0 < \alpha _ { \theta } ( \mathbf { Z } ) \leq \bar { \alpha }
$$

on the relevant latent region, and that $T _ { \theta }$ and $D _ { \psi }$ are Lipschitz with constants $\Lambda _ { T }$ and $\Lambda _ { D }$ , respectively. Then the following hold.

(a) The population latent one-step defect satisfies

$$
\left( \mathbb { E } \left[ \left. T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { Z } _ { n + 1 } ^ { \star } \right. _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } \leq \eta _ { n } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n } .\tag{3.20}
$$

(b) For the deterministic mean rollout (3.13),

$$
\left( \mathbb { E } \left[ \Vert \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \Vert _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } \leq \Lambda _ { D } \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { T } ^ { n - 1 - j } \left( \eta _ { j } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { j } \right) + \delta _ { n } .\tag{3.21}
$$

(c) Incorporating the noisy predictive error gives the refined bound

$$
\begin{array} { r l r } {  { ( \mathbb { E } [ \| \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } ] ) ^ { 1 / 2 } \leq \Lambda _ { D } \sum _ { j = 0 } ^ { n - 2 } \Lambda _ { T } ^ { n - 1 - j } ( \eta _ { j } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { j } ) + \Lambda _ { D } \eta _ { n - 1 } } } \\ & { } & { + \operatorname* { m i n } \{ \Lambda _ { D } \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n - 1 } + \delta _ { n } , \epsilon _ { n - 1 } + \rho _ { n - 1 } \} , } \end{array}\tag{3.22}
$$

for every $n \geq 1$ , where the sum is understood as zero when $n = 1$

Remark 3.13 (Interpretation of the rollout bound). The terms appearing in Proposition 3.12 are controlled by diferent components of the training objective. The dynamic KL error $\kappa _ { n }$ is the population counterpart of the latent consistency term ${ \mathcal { L } } _ { \mathrm { K L } }$ , while the reconstruction error $\delta _ { n }$ is associated with the encoder-decoder reconstruction objective. The noisy prediction error $\epsilon _ { n }$ is controlled through $\mathcal { L } _ { \mathrm { p r e d } } .$ , and the sensitivity quantities $\eta _ { n }$ and $\rho _ { n }$ are studied in Proposition 3.10. As discussed in §3.3.1, these sensitivity terms are not explicitly optimized, but stochastic training evaluates the model over Gaussian neighborhoods and thereby implicitly encourages local stability in the corresponding perturbation directions.

Part (b) makes explicit how the two terms in the conditional ELBO (3.3) enter deterministic rollout. The static reconstruction term is reflected by the reconstruction error $\delta _ { n }$ , which measures the discrepancy between the true physical state and the decoder applied to its encoder-mean representation. The dynamic KL term is reflected by $\kappa _ { n }$ , which controls the mismatch between the predicted transition mean and the encoder mean of the true next state. The additional sensitivity term $\eta _ { n }$ accounts for the fact that the KL objective is evaluated at a perturbed encoder input, whereas deterministic rollout propagates the encoder mean. These one-step latent discrepancies are accumulated through the mean transition with geometric factors $\Lambda _ { T } ^ { n - 1 - j }$ , while $\delta _ { n }$ enters when the propagated latent state is decoded back to the physical space. Thus, the two principal terms of the conditional ELBO reappear naturally in the rollout error bound, linking variational training to deterministic long-horizon accuracy.

Part (c) additionally incorporates the predictive objective into the same deterministic rollout analysis. At the final prediction step, the physical error admits two complementary upper bounds: one obtained from the KL-controlled latent mean discrepancy together with reconstruction,

$$
\Lambda _ { D } \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n - 1 } + \delta _ { n } ,
$$

and the other from the noisy predictive error together with transition-noise sensitivity,

$$
\epsilon _ { n - 1 } + \rho _ { n - 1 } .
$$

Taking the minimum simply selects the tighter analytical estimate for the same deterministic prediction and therefore sharpens the physical-space rollout bound. In particular, this refinement reflects the complementary roles of the variational and predictive components of the training objective.

Finally, we discuss the error amplification in rollouts and characterize the growth rate of the rollout error bound more explicitly. Suppose that the one-step error and sensitivity terms remain uniformly bounded over the rollout horizon:

$$
\eta _ { j } \le \bar { \eta } , \qquad \kappa _ { j } \le \bar { \kappa } , \qquad \delta _ { j } \le \bar { \delta } .
$$

Then Proposition 3.12 (b) implies

$$
\begin{array} { r } { \left( \mathbb { E } \| \widehat { \mathbf { U } } _ { n } - { \mathbf { U } } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right) ^ { 1 / 2 } \leq \left\{ \begin{array} { l l } { \displaystyle \Lambda _ { D } \left( \bar { \eta } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \bar { \kappa } \right) \frac { 1 - \Lambda _ { T } ^ { n } } { 1 - \Lambda _ { T } } + \bar { \delta } , } & { \displaystyle \Lambda _ { T } < 1 , } \\ { \displaystyle n \Lambda _ { D } \left( \bar { \eta } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \bar { \kappa } \right) + \bar { \delta } , } & { \displaystyle \Lambda _ { T } = 1 , } \\ { \displaystyle \Lambda _ { D } \left( \bar { \eta } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \bar { \kappa } \right) \frac { \Lambda _ { T } ^ { n } - 1 } { \Lambda _ { T } - 1 } + \bar { \delta } , } & { \displaystyle \Lambda _ { T } > 1 . } \end{array} \right. } \end{array}
$$

Therefore, under uniformly controlled error and sensitivity terms, the bound remains uniformly bounded for $\Lambda _ { T } < 1$ , grows at most linearly for $\Lambda _ { T } = 1$ , and permits geometric growth for $\Lambda _ { T } > 1$ The result thus separates two ingredients of long-horizon accuracy: the magnitude of the local one-step discrepancies and their amplification through the learned latent transition.

While our main analysis applies deterministic mean-map rollout at inference, Appendix §B.5 gives a complementary high-probability characterization of stochastic rollouts, showing that their latent deviations from the mean-map trajectory remain within a two-sided Gaussian envelope.

Generic autoregressive amplification. To distinguish the generic efect of autoregressive error accumulation from the mechanisms specific to variational latent dynamics, we consider a generic deterministic predictor $F _ { \vartheta } : \mathcal { U }  \mathcal { U }$ , which is deployed recursively as

$$
\widehat { \mathbf { U } } _ { 0 } = \mathbf { U } _ { 0 } ^ { \star } , \qquad \widehat { \mathbf { U } } _ { n + 1 } = F _ { \vartheta } ( \widehat { \mathbf { U } } _ { n } ) .
$$

Define its population one-step prediction error by

$$
\epsilon _ { n } ^ { \mathrm { d i r } } : = \big ( \mathbb { E } \left[ \| F _ { \vartheta } ( \mathbf { U } _ { n } ^ { \star } ) - \mathbf { U } _ { n + 1 } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right] \big ) ^ { 1 / 2 } .
$$

The following result isolates how these one-step errors accumulate solely through repeated application of the learned predictor.

Proposition 3.14 (Generic autoregressive error amplification). Suppose that $F _ { \vartheta }$ is Lipschitz on the relevant region with constant $\Lambda _ { F }$ . Then

$$
\left( \mathbb { E } \left[ \Vert \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \Vert _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } \leq \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { F } ^ { n - 1 - j } \epsilon _ { j } ^ { \mathrm { d i r } } .\tag{3.23}
$$

Proposition 3.14 applies to deterministic autoregressive predictors such as the direct neuraloperator baseline considered in our subsequent experiments. In particular, (3.23) shows that geometric error amplification is a generic feature of autoregressive prediction rather than a consequence of the variational formulation. Although the variational latent model evolves autoregressively in latent rather than physical space, Proposition 3.12 exhibits the same amplification mechanism through powers of $\Lambda _ { T }$ . Thus, the preceding analysis characterizes how the variational training controls the one-step quantities that are subsequently amplified during latent rollout.

The sensitivity analysis in §3.3.1 suggests a complementary role for stochastic training in the autoregressive amplification mechanism. Both the generic deterministic bound and the variational rollout bound contain geometric amplification factors like $\Lambda _ { F }$ and $\Lambda _ { T }$ , governed respectively by the stability of the learned one-step maps. In the variational latent formulation, encoder perturbations expose the transition map to neighborhoods of the encoded trajectory, while transition perturbations probe the decoder around predicted latent states. Proposition 3.10 shows that these sensitivities are locally governed by the Fr´echet derivatives of the corresponding maps. Sharp local fluctuations in rollout-relevant neighborhoods can therefore increase the noisy predictive loss and are implicitly discouraged through reparameterized back-propagation. In this way, variationa training can regularize the efective local expansion encountered along the rollout trajectory and thereby mitigate geometric error amplification over long horizons.

## 4 Variational Autoencoding Markov Operator

The variational latent dynamics developed in $\ S 3$ are formulated directly at the function-space level and are therefore independent of a particular spatial discretization or neural architecture. In this section, we introduce the Variational Autoencoding Markov Operator (VAMO), a discretized neuraloperator realization of this framework for time-dependent PDEs. VAMO realizes the encodertransition-decoder structure of §3.1 using spatially resolved physical and latent fields. Figure 1 summarizes the variational training procedure and the corresponding mean latent rollout. We first construct its deterministic backbone from a resolution-preserving encoder and decoder together with a residual neural-operator transition in §4.1. We then augment this backbone with structured Gaussian latent perturbations in §4.2 and formulate the resulting training objective in §4.3.

## 4.1 Spatially Resolved Architecture

To obtain a computational realization of the function-space framework, we first discretize the physical and latent fields on a finite spatial grid and specify the deterministic mean maps underlying VAMO. This subsection focuses on the encoder mean, latent mean transition, and decoder mean that form the deterministic backbone of the model; the Gaussian covariance structure and stochastic sampling used for variational training are introduced separately in $\ S 4 . 2$

(a) Variational training  
![](images/b2fc9f2e7ec7e9a85fe4b46c0e9d7c65bdd46d2ef9c7a04b84c355881ce4df37.jpg)  
(b) Mean latent rollout

![](images/1412cd593e046afe4e7c78cee667044260e1376414cca47f356289c0549ebdb2.jpg)

Figure 1: Architecture and deployment of VAMO. (a) Variational training on an adjacent reference pair $\left( \mathbf { u } _ { n } , \mathbf { u } _ { n + 1 } \right)$ . The encoder samples $\mathbf { z } _ { n }$ from $q _ { n } = Q _ { \phi } ( \cdot \mid \mathbf { u } _ { n } )$ , and the latent transition samples ${ \bf z } _ { n } ^ { + }$ from $p _ { n } = P _ { \theta } ( \cdot \mid \mathbf { z } _ { n } )$ . A shared decoder produces the reconstruction $\widetilde { \mathbf { u } } _ { n } = D _ { \psi } ( \mathbf { z } _ { n } )$ and prediction $\widehat { \mathbf { u } } _ { n + 1 } = D _ { \psi } ( \mathbf { z } _ { n } ^ { + } )$ . Structured Gaussian perturbations act at both latent sampling stages; the KL term aligns the transition distribution with the encoded next-state distribution $q _ { n + 1 } = Q _ { \phi } ( \cdot \mid { \bf u } _ { n + 1 } )$ Dashed lines indicate loss comparisons, with expectations as defined in §4.3. (b) Mean latent rollout encodes the initial condition once, repeatedly applies $T _ { \theta }$ , and decodes the evolved latent states through step N without sampling or re-encoding physical predictions. Circles denote physical states; rounded rectangles denote spatially resolved latent representations and their distributions.

Let $\{ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot , x _ { N _ { \mathrm { r e s } } } \} \subset \Omega$ denote a spatial grid on the domain Ω with $N _ { \mathrm { r e s } }$ discretization points. Evaluating a physical field with $C _ { u }$ channels and its latent counterpart with $C _ { z }$ channels on this grid gives

$$
\mathbf { u } _ { n } \in \mathbb { R } ^ { C _ { u } \times N _ { \mathrm { r e s } } } , \quad \mathbf { z } _ { n } \in \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } } .
$$

Thus, spatial discretization converts the function-space variables ${ \mathbf { U } } _ { n } \in { \mathcal { U } }$ and $\mathbf { Z } _ { n } \in { \mathcal { Z } }$ into finitedimensional arrays while preserving their spatial organization. In particular, the latent representation remains a field over the computational grid rather than being collapsed into a single latent vector. Under this discretization, the encoder, transition, and decoder mean maps from §3.1 are represented by

$$
m _ { \phi } : \mathbb { R } ^ { C _ { u } \times N _ { \mathrm { r e s } } }  \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } } , \quad T _ { \theta } : \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } }  \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } } , \quad \mathrm { a n d } \quad D _ { \psi } : \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } }  \mathbb { R } ^ { C _ { u } \times N _ { \mathrm { r e s } } } .
$$

The encoder $m _ { \phi }$ first lifts the physical state to a spatially resolved latent representation, which is subsequently evolved by $T _ { \theta }$ and mapped back to the physical variables through $D _ { \psi }$ . We implement the encoder and decoder using resolution-preserving convolutional residual networks (He et al., 2016), while the latent dynamics are modeled by a Fourier neural operator (Li et al., 2021). Specifically, the mean transition takes the residual form

$$
\begin{array} { r } { T _ { \theta } ( \mathbf { z } ) = \mathbf { z } + \mathcal { G } _ { \theta } ^ { \mathrm { F N O } } ( \mathbf { z } ) , } \end{array}\tag{4.1}
$$

where $\mathcal { G } _ { \theta } ^ { \mathrm { F N O } }$ is an FNO acting on the latent feature field. The residual parameterization represents one-step evolution through an increment of the current latent state, consistent with the timediscretized PDE flow map, while the Fourier layers provide nonlocal interactions across the spatial domain. The three mean maps therefore define the deterministic backbone

$$
\mathbf { u } _ { n } \xrightarrow { m _ { \phi } } \mathbf { z } _ { n } \xrightarrow { T _ { \theta } } \mathbf { z } _ { n + 1 } \xrightarrow { D _ { \psi } } \mathbf { u } _ { n + 1 } .
$$

VAMO augments this backbone with state-dependent Gaussian perturbations of the encoded and propagated latent fields. We next describe how the corresponding covariance structures are constructed on the discretized grid.

## 4.2 Structured Gaussian Latent Perturbations

The deterministic backbone in §4.1 specifies the mean evolution of the latent state. To realize the variational formulation, VAMO augments this evolution with Gaussian perturbations whose covariance preserves the spatial structure of the latent field. Rather than injecting independent noise at individual grid points, we generate correlated perturbations through fixed spectral covariance operators and modulate their amplitudes according to the current state.

Spectral covariance structure. We first describe the covariance model shared by the encoder and transition perturbations. On a periodic spatial grid, let $K _ { \ell , s }$ denote the discrete analogue of the Laplacian-resolvent covariance

$$
K _ { \ell , s } = ( I - \ell ^ { 2 } \Delta ) ^ { - s } ,
$$

which is diagonal in the Fourier basis. Its square root therefore acts as

$$
\widehat { K _ { \ell , s } ^ { 1 / 2 } \pmb { \xi } } ( k ) = \big ( 1 + \ell ^ { 2 } | k | ^ { 2 } \big ) ^ { - s / 2 } \widehat { \pmb { \xi } } ( k ) ,\tag{4.2}
$$

where $\boldsymbol { \xi }$ is a standard Gaussian field and k denotes the discrete spatial frequency. Thus, $\ell > 0$ controls the correlation length of the perturbation, while $s > 0$ determines the spectral decay at high frequencies. This construction is the finite-dimensional counterpart of the Laplacian-resolvent Gaussian covariance and Sobolev geometry discussed in §3.2.

Complementary encoder and transition perturbations. For the Gaussian kernels, we use separate covariance operators

$$
K _ { \mathrm { e n c } } = K _ { \ell _ { \mathrm { e n c } } , s } , \quad K _ { \mathrm { t r } } = K _ { \ell _ { \mathrm { t r } } , s }
$$

for the encoder and latent transition, with correlation lengths chosen independently. In practice, we use $\ell _ { \mathrm { e n c } } < \ell _ { \mathrm { t r } }$ and combine these covariance structures with diferent state-dependent modulations.

1. State-dependent encoder distribution. In addition to the encoder mean $m _ { \phi } ( { \mathbf { u } } )$ , the encoder network produces a pointwise log-variance modulation field

$$
\boldsymbol { \ell } _ { \phi } ( { \mathbf { u } } ) \in \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } } .
$$

We define the corresponding standard-deviation multiplier

$$
\sigma _ { \phi } ( { \bf u } ) : = \exp \left( \frac { 1 } { 2 } \ell _ { \phi } ( { \bf u } ) \right) ,
$$

and let $S _ { \phi } ( \mathbf { u } )$ denote pointwise multiplication by $\sigma _ { \phi } ( \mathbf { u } )$ . A latent state is generated via the reparameterization

$$
\begin{array} { r } { \mathbf { z } = m _ { \phi } ( \mathbf { u } ) + S _ { \phi } ( \mathbf { u } ) K _ { \mathrm { e n c } } ^ { 1 / 2 } \pmb { \xi } , \qquad \pmb { \xi } \sim \mathsf { N } ( 0 , I ) , } \end{array}\tag{4.3}
$$

or equivalently,

$$
\mathbf { z } \sim \mathsf { N } \big ( m _ { \phi } ( \mathbf { u } ) , K _ { \phi } ( \mathbf { u } ) \big ) , \quad K _ { \phi } ( \mathbf { u } ) = S _ { \phi } ( \mathbf { u } ) K _ { \mathrm { e n c } } S _ { \phi } ^ { * } ( \mathbf { u } ) .
$$

Hence, the fixed covariance operator $K _ { \mathrm { e n c } }$ specifies the underlying spatial correlation structure, while the learned multiplier $S _ { \phi } ( \mathbf { u } )$ adapts the perturbation magnitude across latent channels, spatial locations, and input states.

2. Stochastic latent transition. The transition distribution uses the same structured-noise principle, but modulates the fixed covariance $K _ { \mathrm { t r } }$ through a scalar state-dependent amplitude. Given a latent state z, we sample

$$
\begin{array} { r } { { \bf z } ^ { + } = T _ { \theta } ( { \bf z } ) + \alpha _ { \theta } ( { \bf z } ) { \cal K } _ { \mathrm { t r } } ^ { 1 / 2 } \boldsymbol { \xi } ^ { \mathrm { t r } } , \qquad { \boldsymbol { \xi } } ^ { \mathrm { t r } } \sim \mathsf { N } ( 0 , I ) , } \end{array}\tag{4.4}
$$

which defines

$$
P _ { \theta } ( \cdot \mid \mathbf { z } ) = \mathsf { N } \big ( T _ { \theta } ( \mathbf { z } ) , \alpha _ { \theta } ( \mathbf { z } ) ^ { 2 } K _ { \mathrm { t r } } \big ) .\tag{4.5}
$$

The amplitude $\alpha _ { \theta } ( { \mathbf { z } } ) > 0$ is predicted by a lightweight network head and adapts the overall perturbation magnitude to the current latent state. The precise numerical parameterization of $\alpha _ { \theta }$ and $\ell _ { \phi }$ is deferred to Appendix §D.

This construction yields an intentional local-global asymmetry between the two stochastic components. The encoder combines the relatively small correlation length $\ell _ { \mathrm { e n c } }$ with a pointwise, spatially heterogeneous multiplier, thereby probing robustness to localized and fine-scale variations in the encoded physical state. In contrast, the transition instead combines the larger correlation length $\ell _ { \mathrm { t r } }$ with a single state-dependent amplitude applied globally across the latent field, producing smoother and more spatially coherent perturbations around the evolved latent state. Thus, the encoder probes spatially heterogeneous local uncertainty, whereas the transition probes coherent variations of the predicted latent dynamics. Together, these complementary perturbation mechanisms realize the stochastic training efects analyzed in §3.3.1.

Decoder likelihood. Finally, we equip the discretized decoder with an isotropic Gaussian likelihood,

$$
P _ { \psi } ( \cdot \mid \mathbf { z } ) = \mathsf { N } \big ( D _ { \psi } ( \mathbf { z } ) , \sigma _ { u } ^ { 2 } I \big ) ,\tag{4.6}
$$

where $\sigma _ { u } > 0$ is fixed. This is the finite-dimensional specialization of the Gaussian decoder considered in Proposition 3.5. In the present setting, the Cameron-Martin space is the entire discretized physical space and its norm is a scaled Euclidean norm:

$$
\| \mathbf { v } \| _ { \mathcal { H } _ { \sigma _ { u } ^ { 2 } I } } ^ { 2 } = \frac { 1 } { \sigma _ { u } ^ { 2 } } \| \mathbf { v } \| _ { 2 } ^ { 2 } .
$$

Consequently, up to an additive constant, the decoder negative log-likelihood is

$$
- \log P _ { \psi } ( \boldsymbol { \mathbf { u } } \mid \boldsymbol { \mathbf { z } } ) = \frac { 1 } { 2 \sigma _ { u } ^ { 2 } } \| \boldsymbol { \mathbf { u } } - D _ { \psi } ( \boldsymbol { \mathbf { z } } ) \| _ { 2 } ^ { 2 } + \mathrm { c o n s t . }
$$

Therefore, the Cameron-Martin reconstruction geometry of the function-space formulation reduces exactly, up to a fixed scaling, to the squared Euclidean norm loss used in the discretized model.

Unlike the encoder and transition distributions, we do not sample from the decoder likelihood. Given a latent state $\mathbf { z } ,$ prediction uses $D _ { \psi } ( \mathbf { z } )$ directly, which is both the mean and the maximumlikelihood output of the isotropic Gaussian decoder. Hence the stochasticity used for variational training is confined to the latent representation and transition rather than being propagated into the physical prediction.

The fixed likelihood scale $\sigma _ { u } ~ > ~ 0$ determines the relative weighting between reconstruction error and latent KL regularization in the ELBO. Indeed, multiplying the negative ELBO by $2 \sigma _ { u } ^ { 2 }$ converts the likelihood term to an unweighted squared Euclidean loss while rescaling the KL term. Accordingly, rather than treating $\sigma _ { u }$ as an additional tunable uncertainty parameter, we fix it and absorb this relative scaling into the KL coeficient $\beta$ in the training objective introduced below.

## 4.3 Training Objective

Having specified the discretized encoder, transition, and decoder distributions, we now formulate the training objective on observed PDE trajectories as the finite-dimensional counterpart of the variational objective studied in §3.3. Let $\{ { \mathbf { u } } _ { n } \} _ { n = 0 } ^ { N }$ denote a trajectory from the training set. For each adjacent pair $\left( \mathbf { u } _ { n } , \mathbf { u } _ { n + 1 } \right)$ , we draw

$$
\mathbf { z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { u } _ { n } ) , \qquad \mathbf { z } _ { n } ^ { + } \sim P _ { \theta } ( \cdot \mid \mathbf { z } _ { n } ) ,
$$

using the reparameterizations in (4.3) and (4.4). The resulting one-step objective contains three components:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } } ^ { ( n ) } : = \mathbb { E } _ { \mathbf { z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { u } _ { n } ) } \mathbb { E } _ { \mathbf { z } _ { n } ^ { + } \sim P _ { \theta } ( \cdot | \mathbf { z } _ { n } ) } \left[ \left| \left| \mathbf { u } _ { n + 1 } - D _ { \psi } ( \mathbf { z } _ { n } ^ { + } ) \right| \right| _ { 2 } ^ { 2 } \right] , } \end{array}\tag{4.7}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K L } } ^ { ( n ) } : = \mathbb { E } _ { \mathbf { z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { u } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { u } _ { n + 1 } ) \parallel P _ { \theta } ( \cdot \mid \mathbf { z } _ { n } ) \right) \right] , } \end{array}\tag{4.8}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { r e c } } ^ { ( n ) } : = \mathbb { E } _ { \mathbf { z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { u } _ { n } ) } \left[ \| \mathbf { u } _ { n } - D _ { \psi } ( \mathbf { z } _ { n } ) \| _ { 2 } ^ { 2 } \right] . } \end{array}\tag{4.9}
$$

We optimize the trajectory-averaged objective

$$
\mathcal { L } ( \theta , \phi , \psi ) = \frac { 1 } { N } \sum _ { n = 0 } ^ { N - 1 } \left( \mathcal { L } _ { \mathrm { p r e d } } ^ { ( n ) } + \beta \mathcal { L } _ { \mathrm { K L } } ^ { ( n ) } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } ^ { ( n ) } \right) ,\tag{4.10}
$$

where the expectations are estimated by Monte Carlo samples through the reparameterized encoder and transition distributions (Kingma and Welling, 2014).

The three terms correspond directly to the mechanisms analyzed in §3.3, with their computational roles illustrated in Figure 1(a). The predictive term trains the full stochastic pathway $\mathbf { u } _ { n }  \mathbf { z } _ { n }  \mathbf { z } _ { n } ^ { + }  \widehat { \mathbf { u } } _ { n + 1 }$ , the KL term aligns the predicted latent transition with the distribution obtained by encoding the true next state, and the reconstruction term maintains an informative latent representation of the physical field.

Relation to the function-space objective. Equation (4.10) is a finite-dimensional realization of a weighted combination of the predictive negative log-likelihood and the negative conditional ELBO. To make this relation explicit, consider

$$
\lambda _ { \mathrm { p r e d } } \mathcal { L } _ { \mathrm { p r e d } } ^ { \mathrm { N L L } } + \lambda _ { \mathrm { E L B O } } \left( \mathcal { L } _ { \mathrm { K L } } + \mathcal { L } _ { \mathrm { r e c } } ^ { \mathrm { N L L } } \right) .
$$

Under the isotropic decoder likelihood (4.6), both likelihood terms reduce to squared Euclidean losses with the factor $( 2 \sigma _ { u } ^ { 2 } ) ^ { - 1 }$ . Normalizing the coeficient of the predictive Euclidean loss to one therefore gives, up to additive constants,

$$
\mathcal { L } _ { \mathrm { p r e d } } + \underbrace { 2 \sigma _ { u } ^ { 2 } \frac { \lambda _ { \mathrm { E L B O } } } { \lambda _ { \mathrm { p r e d } } } } _ { \beta } \mathcal { L } _ { \mathrm { K L } } + \underbrace { \frac { \lambda _ { \mathrm { E L B O } } } { \lambda _ { \mathrm { p r e d } } } } _ { \lambda _ { \mathrm { r e c } } } \mathcal { L } _ { \mathrm { r e c } } .
$$

Hence, up to an overall rescaling, the relative weighting of the predictive bound and conditional ELBO in Proposition 3.2, together with the decoder likelihood scale, is represented equivalently by the two coeficients $\beta$ and $\lambda _ { \mathrm { { r e c } } }$ in (4.10).

Notably, the conditional ELBO in Proposition 3.2 naturally contains next-state reconstruction, whereas (4.9) uses the current state. Define the step-n reconstruction term

$$
\mathscr { R } _ { n } : = \mathbb { E } _ { \mathbf { z } _ { n } \sim Q _ { \phi } ( \cdot | \mathbf { u } _ { n } ) } \left[ - \log P _ { \psi } ( \mathbf { u } _ { n } \mid \mathbf { z } _ { n } ) \right] .
$$

Over a complete trajectory, the ELBO reconstruction terms sum to

$$
\sum _ { n = 0 } ^ { N - 1 } \mathcal { R } _ { n + 1 } = \sum _ { n = 1 } ^ { N } \mathcal { R } _ { n } ,
$$

whereas the implemented current-state reconstruction sums to $\textstyle \sum _ { n = 0 } ^ { N - 1 } \mathcal { R } _ { n }$ . The two therefore difer only through the boundary contributions $\mathcal { R } _ { 0 }$ and $\mathcal { R } _ { N }$ . We use the current-state form because it reuses the latent sample $\mathbf { z } _ { n }$ already drawn for the transition step, avoiding an additional reparameterized sample from $Q _ { \phi } ( \cdot \mid \mathbf { u } _ { n + 1 } )$ for reconstruction. This provides a modest computational eficiency gain during training while leaving all interior reconstruction contributions unchanged.

Mean latent rollout. At inference, VAMO follows the mean latent dynamics illustrated in Figure 1(b). Given an initial condition $\mathbf { u } _ { 0 }$ , we initialize $\widehat { \mathbf { z } } _ { 0 } = m _ { \phi } ( \mathbf { u } _ { 0 } )$ and recursively compute

$$
\widehat { \mathbf { z } } _ { n + 1 } = T _ { \theta } ( \widehat { \mathbf { z } } _ { n } ) , \qquad \widehat { \mathbf { u } } _ { n + 1 } = D _ { \psi } ( \widehat { \mathbf { z } } _ { n + 1 } ) .
$$

The initial condition is encoded once, and subsequent evolution proceeds entirely in latent space, with Gaussian perturbations disabled and decoded predictions used only as physical outputs.

## 5 Numerical Experiments

In this section, we evaluate VAMO on three fluid-dynamics benchmarks with distinct dynamical characteristics: the one-dimensional compressible Euler equations, two-dimensional compressible fluid dynamics, and forced two-dimensional incompressible Navier–Stokes equations. Together, these problems cover inviscid wave propagation, coupled compressible flow, and long-time vortex dynamics under diferent levels of physical dissipation. Our experiments are designed to answer three main questions:

• Can variational latent dynamics improve the accuracy and stability of long-horizon autoregressive PDE prediction?

• Are the improvements attributable to the structured variational formulation, rather than latent compression or generic noise injection alone?

• Does VAMO better preserve physically relevant spatial and spectral structures throughout the rollout?

For all benchmarks, the model is trained using trajectories restricted to a finite training horizon and evaluated by autoregressive rollout over a substantially longer test horizon. During inference, the model is initialized from the exact initial condition and receives no additional ground-truth states. We report errors accumulated over the entire test rollout. This evaluation therefore measures both prediction accuracy within the temporal regime represented during training and stability beyond the training horizon.

## 5.1 Benchmarks and Baselines

## 5.1.1 Benchmarks Problems

We consider three representative fluid-dynamics systems, summarized in Table 1. We provide a brief description here and defer the complete governing equations, data-generation procedures, discretization schemes, and dataset configurations to Appendix $\ S \mathrm { C }$

1D compressible Euler equations. The one-dimensional Euler equations describe inviscid compressible flow through conservation of mass, momentum, and total energy. We represent the physical state using the primitive variables

$$
\mathbf { u } ( t , x ) = \big ( \rho ( t , x ) , v ( t , x ) , p ( t , x ) \big ) , \quad t \in [ 0 , T ] , x \in \mathbb { T } _ { L } .
$$

where $\rho , \ v ,$ and $p$ denote density, velocity, and pressure, respectively. Since the system contains no explicit viscous dissipation, prediction errors in wave speed and phase can accumulate over time, while nonlinear evolution may generate sharp gradients and discontinuous structures. This benchmark therefore tests long-horizon propagation of compressible waves in a controlled onedimensional setting.

2D compressible fluid dynamics. We consider a two-dimensional viscous compressible-flow benchmark with a physical configuration similar to the CFD setting studied in PDEBench (Takamoto et al., 2022). The state is represented by

$$
\mathbf { u } ( t , \mathbf { x } ) = { \big ( } \rho ( t , \mathbf { x } ) , v _ { x } ( t , \mathbf { x } ) , v _ { y } ( t , \mathbf { x } ) , p ( t , \mathbf { x } ) { \big ) } , \quad t \in [ 0 , T ] , \ \mathbf { x } \in \mathbb { T } ^ { 2 } .
$$

Table 1: Summary of the fluid-dynamics benchmarks. The three systems span one-dimensional inviscid compressible waves, two-dimensional viscous compressible flow, and forced incompressible vorticity dynamics.
<table><tr><td>Benchmark</td><td>PDE type</td><td>Fields</td><td>Main challenge</td></tr><tr><td>1D Euler</td><td>Compressible, inviscid</td><td> $( \rho , v , p )$ </td><td>Wave propagation, phase error, shocks</td></tr><tr><td>2D CFD</td><td>Compressible, viscous</td><td> $( \rho , \mathbf { v } , p )$ </td><td>Coupled multi-field dynamics, conservation</td></tr><tr><td>2D Navier-Stokes</td><td>Incompressible, viscous</td><td>ω</td><td>Nonlocal transport and long-time stability</td></tr></table>

This system couples density, pressure, and the two velocity components through nonlinear conservation laws and viscous transport. It provides a challenging multi-field benchmark in which local prediction errors can propagate across physical channels and spatial locations.

2D incompressible Navier-Stokes equations. We study the forced two-dimensional incompressible Navier-Stokes equations in vorticity form on the unit torus (Li et al., 2022). The learned state is the scalar vorticity field

$$
\omega ( t , { \mathbf x } ) , \quad t \in [ 0 , T ] , { \mathbf x } \in \mathbb { T } ^ { 2 } .
$$

while the corresponding velocity field is recovered nonlocally through the Biot-Savart operator. We consider several viscosity levels and two families of external forcing. Under weak viscosity, small errors in the predicted vorticity can induce nonlocal velocity errors that feed back into future advection. This benchmark is therefore particularly suited to evaluating long-time rollout stability and preservation of spatial and spectral flow structure.

## 5.1.2 Compared Methods

We compare VAMO with three baselines that isolate the efects of direct physical-space evolution, generic noise injection, and deterministic latent dynamics. All models are trained as one-step predictors and evaluated through autoregressive rollout.

FNO. We use the Fourier neural operator (Li et al., 2021) as a direct physical-space baseline. Unlike the trajectory-to-trajectory setting considered in the original FNO experiments, our model maps the current physical snapshot to the next snapshot and is applied recursively over the test horizon.

FNO+Noise. This baseline uses the same autoregressive FNO architecture, but injects Gaussian perturbations into the lifted feature field during training. It tests whether generic featurespace noise regularization alone can improve long-horizon stability.

FNO-AE. This model combines a deterministic autoencoder with a residual FNO latent transition. The initial physical state is encoded once, the latent field is evolved recursively, and the resulting latent states are decoded into physical predictions. It therefore isolates deterministic latent evolution from the variational training used by VAMO.

FNO-VAMO. This denotes the complete proposed model described in §4. FNO-AE and FNO-VAMO use closely matched latent architectures, allowing their comparison to isolate the efect of the variational formulation.

Table 2: Experimental role of each model comparison.
<table><tr><td>Comparison</td><td>Question addressed</td></tr><tr><td>FNO-VAMO vs. FNO</td><td>Does the complete framework improve long-horizon prediction?</td></tr><tr><td>FNO-VAMO vs. FNO-AE</td><td>Is variational training necessary beyond latent modeling?</td></tr><tr><td>FNO-VAMO vs. FNO+Noise </td><td>Is structured latent stochasticity better than generic noise?</td></tr><tr><td>FNO-AE vs. FNO</td><td>Does the deterministic latent architecture alone improve stability?</td></tr></table>

The roles of the four comparisons are summarized in Table 2. For a fair comparison, the models use matched FNO backbones whenever applicable, including the number of Fourier layers, retained modes, and hidden width. All methods are trained using the same trajectories, data splits, optimization budget, and model-selection protocol. Complete architectural and optimization details are provided in Appendix §D.

Evaluation protocol. All models are trained as one-step predictors and evaluated by autoregressive rollout from the exact initial condition, without teacher forcing or ground-truth reinitialization. Errors are computed over the full test horizon and averaged across test trajectories. Relative $L ^ { 2 }$ error is the primary metric across all benchmarks. We additionally report relative $L ^ { 1 }$ error for the 1D Euler equations, relative $H ^ { 1 }$ error for the 2D benchmarks, and errors in enstrophy, palinstrophy, and the isotropic energy spectrum for incompressible Navier-Stokes. Complete metric definitions, including the benchmark-specific temporal aggregation and numerical implementation, are provided in Appendix §C.

## 5.2 Main Results

We first compare the long-horizon rollout performance of VAMO and the three baselines across the fluid-dynamics benchmarks. For each problem, all models are trained and evaluated using the same trajectories and autoregressive protocol described in §5.1. The tables report errors accumulated over the full test horizon. We focus here on the overall quantitative comparison and defer the temporal evolution of rollout errors, qualitative predictions, and additional physical diagnostics to the subsequent sections.

## 5.2.1 1D Compressible Euler Equations

We evaluate the models on periodic 1D Euler trajectories with domain lengths $L \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ and corresponding spatial resolutions 1024, 2048, 3072, and 4096, keeping the grid spacing fixed across settings. Initial conditions are generated from normalized periodic Gaussian random fields with physical correlation length $\ell = 1$ . Models are trained on trajectories over [0, 1.5] and rolled out autoregressively to $T _ { \mathrm { t e s t } } = 1 0$ with snapshot interval $\Delta t = 0 . 1$ . Complete data-generation and solver details are provided in Appendix §C.1.1. We report full-rollout relative $L ^ { 1 }$ and $L ^ { 2 }$ errors for density, velocity, and pressure in Table 3.

As shown in Table 3, FNO-VAMO achieves the lowest channel-averaged relative $L ^ { 2 }$ and $L ^ { 1 }$ errors for all four domain lengths. Relative to the strongest baseline in each setting (FNO-AE for $L = 5 , 1 0 , 1 5$ and FNO+Noise for $L = 2 0 )$ , FNO-VAMO reduces the channel-mean relative $L ^ { 2 }$ error by approximately 21.5%, 25.8%, 19.0%, and 46.2% for $L = 5 , 1 0 , 1 5 .$ , and 20, respectively. The improvement is consistent across density, pressure, and velocity, with velocity remaining the most challenging physical channel.

Table 3: Results on the 1D Euler equations with diferent domain lengths L. We report rolloutaggregated relative $L ^ { 1 }$ and $L ^ { 2 }$ errors for density $\rho ,$ pressure $p ,$ velocity $u ,$ and their channel average. Lower is better.
<table><tr><td colspan="2">L Method</td><td colspan="2">Density  $\rho$ </td><td colspan="2">Pressure p</td><td colspan="2">Velocity u</td><td colspan="2">Channel Mean</td></tr><tr><td colspan="2"></td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td>Rel. L¹</td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . L ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td>Rel.  $L ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td>Rel.  $L ^ { 1 }$ </td></tr><tr><td rowspan="4">5</td><td>FNO</td><td>0.0194</td><td>0.00883</td><td>0.0249</td><td>0.00951</td><td>0.1409</td><td>0.0596</td><td>0.0617</td><td>0.0260</td></tr><tr><td>FNO+Noise</td><td>0.0200</td><td>0.00962</td><td>0.0259</td><td>0.0109</td><td>0.1488</td><td>0.0727</td><td>0.0649</td><td>0.0311</td></tr><tr><td>FNO-AE</td><td>0.0161</td><td>0.00737</td><td>0.0208</td><td>0.00873</td><td>0.1195</td><td>0.0553</td><td>0.0521</td><td>0.0238</td></tr><tr><td>FNO-VAMO</td><td>0.0132</td><td>0.00704</td><td>0.0168</td><td>0.00756</td><td>0.0926</td><td>0.0515</td><td>0.0409</td><td>0.0220</td></tr><tr><td rowspan="4">10</td><td>FNO</td><td>0.0206</td><td>0.00801</td><td>0.0277</td><td>0.00942</td><td>0.1325</td><td>0.0512</td><td>0.0602</td><td>0.0229</td></tr><tr><td>FNO+Noise</td><td>0.0212</td><td>0.00835</td><td>0.0286</td><td>0.00986</td><td>0.1370</td><td>0.0540</td><td>0.0623</td><td>0.0241</td></tr><tr><td>FNO-AE</td><td>0.0181</td><td>0.00709</td><td>0.0245</td><td>0.00913</td><td>0.1263</td><td>0.0553</td><td>0.0563</td><td>0.0238</td></tr><tr><td>FNO-VAMO</td><td>0.0141</td><td>0.00625</td><td>0.0189</td><td>0.00701</td><td>0.0924</td><td>0.0450</td><td>0.0418</td><td>0.0194</td></tr><tr><td rowspan="4">15</td><td>FNO</td><td>0.0266</td><td>0.00919</td><td>0.0369</td><td>0.0113</td><td>0.1656</td><td>0.0646</td><td>0.0763</td><td>0.0284</td></tr><tr><td>FNO+Noise</td><td>0.0263</td><td>0.00918</td><td>0.0366</td><td>0.0113</td><td>0.1642</td><td>0.0649</td><td>0.0757</td><td>0.0285</td></tr><tr><td>FNO-AE</td><td>0.0214</td><td>0.00854</td><td>0.0290</td><td>0.0102</td><td>0.1300</td><td>0.0571</td><td>0.0601</td><td>0.0253</td></tr><tr><td>FNO-VAMO</td><td>0.0173</td><td>0.00733</td><td>0.0232</td><td>0.00844</td><td>0.1056</td><td>0.0475</td><td>0.0487</td><td>0.0211</td></tr><tr><td rowspan="4">20</td><td>FNO</td><td>0.0380</td><td>0.0127</td><td>0.0516</td><td>0.0159</td><td>0.2227</td><td>0.0855</td><td>0.1099</td><td>0.0380</td></tr><tr><td>FNO+Noise</td><td>0.0378</td><td>0.0126</td><td>0.0511</td><td>0.0158</td><td>0.2208</td><td>0.0851</td><td>0.1032</td><td>0.0378</td></tr><tr><td>FNO-AE</td><td>0.5394</td><td>0.2259</td><td>0.5434</td><td>0.2261</td><td>2.1236</td><td>1.0253</td><td>1.0688</td><td>0.4924</td></tr><tr><td>FNO-VAMO</td><td>0.0211</td><td>0.00993</td><td>0.0255</td><td>0.00955</td><td>0.1200</td><td>0.0641</td><td>0.0555</td><td>0.0279</td></tr></table>

The baseline comparison also distinguishes variational latent training from simpler alternatives. Generic noise injection provides little consistent improvement over the direct FNO, while the deterministic latent model performs competitively for $L \ \leq \ 1 5$ but becomes unstable at $L \ = \ 2 0$ In contrast, FNO-VAMO maintains a channel-mean relative $L ^ { 2 }$ error below 0.056 across all four domain lengths, despite the larger physical domain and increasingly long-range wave interactions. These results suggest that the variational formulation improves the robustness of recursively evolved latent dynamics rather than merely providing a better one-step approximation.

## 5.2.2 2D Compressible Fluid Dynamics

We evaluate the models on 2D compressible Navier-Stokes trajectories on the periodic unit torus $\mathbb { T } ^ { 2 } = \mathbb { R } ^ { 2 } / \mathbb { Z } ^ { 2 }$ in a low-viscosity, low-Mach-number regime with $\mu = \zeta = 1 0 ^ { - 8 }$ and $M = 0 . 1$ . Initial conditions are sampled from periodic Gaussian random fields. Models are trained on $t \in [ 0 , 1 ]$ and evaluated by autoregressive rollout to $T _ { \mathrm { t e s t } } = 1 0$ with snapshot interva $\Delta t = 0 . 1$ . The lowviscosity regime is chosen to preserve nontrivial dynamics over the long test horizon; complete data-generation and numerical details are provided in Appendix §C.2.1.

As shown in Table 4, FNO-VAMO substantially outperforms all baselines across every physical component and both error metrics. Relative to the strongest baseline (FNO-AE), it reduces the channel-averaged relative $L ^ { 2 }$ error from 0.2526 to 0.0565 (−77.6%) and the relative $H ^ { 1 }$ error from 2.3394 to 0.4044 (−82.7%). The gains are consistent across density, pressure, and velocity, with particularly large improvements for the velocity field, which is the most challenging component for the baseline models.

The baseline comparisons also clarify the source of the improvement. Generic noise injection does not improve upon the direct FNO, while the deterministic latent model lowers the $L ^ { 2 }$ error but remains much less accurate in $H ^ { 1 }$ , indicating poor preservation of spatial derivatives despite improved field-level accuracy. In contrast, FNO-VAMO reduces both metrics simultaneously, suggesting better preservation of the spatial structure of the coupled compressible dynamics. Figure 2 further shows that this advantage is reflected across individual test trajectories: FNO-VAMO exhibits lower typical errors, tighter distributions, and substantially fewer large-error outliers, especially in relative $H ^ { 1 }$ . Thus, its improvement reflects more reliable long-horizon prediction across the test distribution rather than gains on only a small subset of trajectories.

## 5.2.3 2D Incompressible Navier-Stokes Equations

We consider the forced 2D incompressible Navier-Stokes equations in vorticity form on the periodic unit torus $\mathbb { T } ^ { 2 } = \mathbb { R } ^ { 2 } / \mathbb { Z } ^ { 2 }$ , with viscosities $\nu \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ following the standard FNO benchmark setting (Li et al., 2021). We extend this setting by independently sampling a time-independent forcing field for each trajectory from either a Gaussian random field (GRF) or a sparse mixture of sinusoidal Fourier modes (Wave). For each viscosity, a single model is trained jointly on 1,600 trajectories, with 800 from each forcing family, and evaluated separately on the two forcing types. The training horizons are [0, 10], [0, 12], and [0, 8] for $\nu = 1 0 ^ { - 3 } , 1 0 ^ { - 4 }$ , and $1 0 ^ { - 5 }$ , respectively, with corresponding test horizons $T = 5 0$ , 30, and 20. Complete data-generation and numerical details are provided in Appendix §C.3.1.

As shown in Table 5, FNO-VAMO achieves the lowest error across all 6 viscosity-forcing settings and all five evaluation metrics. The gains are particularly strong in the more challenging low-viscosity regimes. Relative to the strongest baseline in each setting, FNO-VAMO reduces the relative $L ^ { 2 }$ error by approximately 65–87% for $\nu = 1 0 ^ { - 4 }$ and 51–72% for $\nu = 1 0 ^ { - 5 }$ , with corresponding relative $H ^ { 1 }$ reductions of about 85–94% and 61–71%, respectively.

The improvements extend beyond field-level accuracy. FNO-VAMO also consistently yields lower enstrophy, palinstrophy, and spectral errors, indicating better preservation of both overall vorticity magnitude and small-scale spatial structure. Generic noise injection is not consistently beneficial, while the deterministic latent baseline becomes highly unstable in several settings, especially at $\nu = 1 0 ^ { - 3 }$ and $\nu = 1 0 ^ { - 4 }$ . These comparisons suggest that neither latent evolution nor unstructured noise alone is suficient for reliable long-horizon prediction.

Figure 3 further shows that the advantage of FNO-VAMO is reflected not only in lower average error, but also in the distribution across individual trajectories. In the two more challenging regimes $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ , FNO-VAMO generally achieves comparable or lower median errors while exhibiting narrower spreads and substantially fewer large-error outliers. This indicates that the variational formulation improves not only average accuracy but also the reliability of long-horizon

Table 4: Results on 2D compressible fluid dynamics. We report rollout-aggregated relative $L ^ { 2 }$ and $H ^ { 1 }$ errors for density $\rho ,$ pressure $p ,$ velocity $\mathbf { v } = ( v _ { x } , v _ { y } )$ , and their channel average. Lower is better.
<table><tr><td rowspan="2">Method</td><td colspan="2">Density  $\rho$ </td><td colspan="2">Pressure  $p$ </td><td colspan="2">Velocity v</td><td colspan="2">Channel Mean</td></tr><tr><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l . } H ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td></tr><tr><td>FNO</td><td>0.1248</td><td>2.2440</td><td>0.1674</td><td>3.2094</td><td>0.8345</td><td>2.7868</td><td>0.3756</td><td>2.7467</td></tr><tr><td>FNO+Noise</td><td>0.1422</td><td>2.3479</td><td>0.2073</td><td>3.8650</td><td>0.9055</td><td>3.0393</td><td>0.4183</td><td>3.0841</td></tr><tr><td>FNO-AE</td><td>0.0972</td><td>3.3152</td><td>0.0331</td><td>1.7665</td><td>0.6276</td><td>1.9364</td><td>0.2526</td><td>2.3394</td></tr><tr><td>FNO-VAMO</td><td>0.0290</td><td>0.6375</td><td>0.0090</td><td>0.2612</td><td>0.1316</td><td>0.3145</td><td>0.0565</td><td>0.4044</td></tr></table>

![](images/559d2aff113b6a94c8cea3a1ef517c31cfbfe609e4a35bef93ce5b36d801ca01.jpg)  
Figure 2: Distribution of per-trajectory full-rollout relative $L ^ { 2 }$ and $H ^ { 1 }$ errors for the twodimensional compressible-flow benchmark. Results are shown separately for density, pressure, velocity, and their channel mean. The logarithmic scale highlights the heavy upper tails of the baseline methods and the tighter error distribution achieved by FNO-VAMO.

Table 5: Results on the 2D incompressible Navier-Stokes equations. We evaluate six test settings with viscosity $\nu \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ and forcing type in {GRF, Wave}. The training horizons are [0, 10], [0, 12], and [0, 8] for $\nu = 1 0 ^ { - 3 } , 1 0 ^ { - 4 }$ , and $1 0 ^ { - 5 }$ , respectively, while the corresponding test horizons are [0, 50], [0, 30], and [0, 20]. We report rollout-aggregated errors. Lower is better.
<table><tr><td>Data Setting</td><td>Forcing</td><td>Method</td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td><td>E-error</td><td>P-error</td><td>Spec. error</td></tr><tr><td rowspan="5"> $\nu = 1 0 ^ { - 3 }$   $T _ { \mathrm { t r a i n } } = 1 0$   $T _ { \mathrm { t e s t } } = 5 0$ </td><td rowspan="5">GRF</td><td>FNO</td><td>0.1453</td><td>0.8015</td><td>0.1151</td><td>3.784</td><td>0.0239</td></tr><tr><td>FNO+Noise</td><td>0.0375</td><td>0.0985</td><td>0.0196</td><td>0.1773</td><td>0.0206</td></tr><tr><td>FNO-AE</td><td>22.31</td><td>292.0</td><td>2447</td><td> $4 . 3 5 8 \times 1 0 ^ { 5 }$ </td><td>452.0</td></tr><tr><td>FNO-VAMO</td><td>0.0203</td><td>0.0264</td><td>0.0119</td><td>0.0123</td><td>0.0138</td></tr><tr><td>FNO</td><td>0.0698</td><td>0.1769</td><td>0.0430</td><td>0.4039</td><td>0.0116</td></tr><tr><td rowspan="5">GRF  $\nu = 1 0 ^ { - 4 }$ </td><td rowspan="5">WAVE</td><td>FNO+Noise FNO-AE</td><td>0.0686 5.623</td><td>0.1148 55.84</td><td>0.0406 565.0</td><td>0.1778  $5 . 7 5 3 \times 1 0 ^ { 4 }$ </td><td>0.0241 124.3</td></tr><tr><td>FNO-VAMO</td><td>0.0556</td><td>0.0687</td><td>0.00808</td><td>0.00940</td><td>0.0106</td></tr><tr><td>FNO</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FNO+Noise</td><td>0.4068 0.2987</td><td>3.109</td><td>0.7338</td><td>45.84</td><td>0.0456</td></tr><tr><td>FNO-AE</td><td>2.691</td><td>2.003 31.93</td><td>0.5866 23.19</td><td>35.15 3608</td><td>0.0569 0.8164</td></tr><tr><td rowspan="5"> $T _ { \mathrm { t r a i n } } = 1 2$   $T _ { \mathrm { t e s t } } = 3 0$ </td><td></td><td>FNO-VAMO</td><td>0.0403</td><td>0.1139</td><td>0.0139</td><td>0.0512</td><td>0.0146</td></tr><tr><td rowspan="4">WAVE</td><td>FNO</td><td>0.3570</td><td>1.613</td><td>0.6580</td><td>14.51</td><td>0.0497</td></tr><tr><td>FNO+Noise</td><td>0.3407</td><td>1.449</td><td>0.0666</td><td>15.73</td><td>0.0499</td></tr><tr><td>FNO-AE</td><td>1.188</td><td>10.19</td><td>7.213</td><td>580.3</td><td>0.4303</td></tr><tr><td>FNO-VAMO</td><td>0.1198</td><td>0.2212</td><td>0.0386</td><td>0.1008</td><td>0.0390</td></tr><tr><td rowspan="6"> $\nu = 1 0 ^ { - 5 }$   $T _ { \mathrm { t r a i n } } = 8$   $T _ { \mathrm { t e s t } } = 2 0$ </td><td rowspan="4">GRF</td><td>FNO</td><td>0.2987</td><td></td><td></td><td></td><td></td><td>0.0382</td></tr><tr><td>FNO+Noise</td><td></td><td>2.262</td><td>0.4862</td><td>32.90</td><td></td><td></td></tr><tr><td></td><td>0.4089</td><td>3.061</td><td>1.179</td><td>75.58</td><td></td><td>0.0909</td></tr><tr><td>FNO-AE FNO-VAMO</td><td>0.5991</td><td>1.335</td><td>0.2788</td><td>1.354</td><td></td><td>0.1551</td></tr><tr><td rowspan="4">WAVE</td><td></td><td>0.0841</td><td>0.3858</td><td>0.0211</td><td>0.2415</td><td></td><td>0.00929</td></tr><tr><td>FNO FNO+Noise</td><td>0.2993</td><td>1.969</td><td>0.3519</td><td></td><td>19.14</td><td>0.0314</td></tr><tr><td>FNO-AE</td><td>0.3450 0.6719</td><td>2.254 1.158</td><td>0.5831 0.2098</td><td></td><td>31.22</td><td>0.0617</td></tr><tr><td>FNO-VAMO</td><td>0.1469</td><td>0.4509</td><td>0.0460</td><td></td><td>0.5878 0.2734</td><td>0.2518 0.0247</td></tr></table>

![](images/bd8da111700d103a74bdff5c1ce4febfb2c4b95547591ecdcd427991278bbe5b.jpg)  
Figure 3: Distribution of per-trajectory full-rollout relative $L ^ { 2 }$ and $H ^ { 1 }$ errors for the 2D incompressible Navier-Stokes benchmark. Results are shown separately for GRF and Wave forcing. The logarithmic scale highlights both the typical error and the heavy upper tails associated with unstable rollouts. VAMO generally reduces the median error and suppresses large-error outliers.

rollout across trajectories.

## 5.3 Long-Horizon Error Growth

The trajectory-aggregated results in §5.2 do not show how prediction errors evolve during autoregressive rollout. We therefore report per-snapshot errors for representative 1D Euler settings with L = 5 and L = 20 (Figure 4), the 2D compressible-flow benchmark (Figure 5), and the two more challenging 2D incompressible Navier–Stokes regimes, $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ , under both forcing families (Figures 6–7). Results for more benchmarks are provided in Appendix §E.2. In each figure, the shaded region denotes the temporal horizon represented during training.

Across the three PDE benchmarks, the main advantage of VAMO appears in the later stages of rollout. Its error can be comparable to, or occasionally higher than, that of the direct baselines at early in-distribution snapshots, particularly in easier settings such as the 1D Euler equation with $L = 5$ . However, the baseline errors generally grow more rapidly as the rollout proceeds, whereas VAMO exhibits substantially slower error accumulation. Consequently, VAMO achieves lower late-time error and better full-rollout accuracy.

![](images/e94e4c2a500bb96389edf30ad779267d47c8e5aaafd2c9e75b80e9980e53f992.jpg)  
(a) Domain length L = 5.

![](images/47831f8daf77a52b09faa759f4c68ff7d03e40b155c53ba7a07af542341a356b.jpg)  
(b) Domain length L = 20.

Figure 4: Error trend for representative 1D Euler Benchmarks.  
![](images/c4ea5a6eb83fcfcf79705a02ebb2e86d40549816d0669a910fd8e3cb3c3916a1.jpg)  
Figure 5: Error trend for 2D CFD Benchmark.

![](images/59d17440d4a6859b5e74df38819bbfe31e287295bc41e2f566eab5043743d1fb.jpg)  
(b) Wave forcing type.  
Figure 6: Error trend for 2D incompressible NS Benchmark with $\nu = 1 0 ^ { - 4 }$

![](images/a68632d3618c93a88153bacd85a106b26b050914e32ea2bde945cf32b8461736.jpg)  
(b) Wave forcing type.  
Figure 7: Error trend for 2D incompressible NS Benchmark with $\nu = 1 0 ^ { - 5 }$

This distinction becomes especially clear in the more challenging settings. For Euler with $L = 2 0$ , the deterministic latent model remains competitive initially but becomes unstable during the later rollout, while VAMO maintains controlled $L ^ { 1 }$ and $L ^ { 2 }$ errors. On the 2D compressibleflow benchmark, the $L ^ { 2 }$ and $H ^ { 1 }$ errors of the baselines grow rapidly after the training horizon, whereas VAMO remains substantially more stable. A similar pattern is observed in the low-viscosity incompressible Navier–Stokes regimes $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ under both forcing families, where VAMO consistently limits the growth of field-level errors and physically relevant quantities such as enstrophy and palinstrophy. These results indicate that the primary benefit of the variationa formulation lies in improving long-horizon stability rather than uniformly reducing the error at every individual prediction step.

![](images/76b2edd856b9aa7ec219cac3b05753bd091516d53f6a25c8db60d7dffdb5775a.jpg)  
(a) Viscosity $\nu = 1 0 ^ { - 4 }$ , GRF forcing.

![](images/6ec1e2c8d173191f28fcdece4004527588c7c4a2b894023de85a6626a9d30474.jpg)  
(b) Viscosity $\nu = 1 0 ^ { - 4 }$ , Wave forcing.  
Figure 8: Physical diagnostics for the 2D incompressible Navier-Stokes benchmark. Each panel compares enstrophy and palinstrophy along 20 ground-truth and predicted trajectories. Solid translucent curves denote ground truth, and dashed curves denote predictions.

![](images/388b7533829ad31508fc687ad64d88748746c540fcb7b7bdc09b3aa94e6567f5.jpg)  
(c) Viscosity $\nu = 1 0 ^ { - 5 } ,$ , GRF forcing.

![](images/d45729cf68097294a471c8bb5598bae66f5904a3a446cb6b1d79b28dba3f4b23.jpg)  
(d) Viscosity $\nu = 1 0 ^ { - 5 }$ , Wave forcing.  
Figure 8: Physical diagnostics for the 2D incompressible Navier-Stokes benchmark (continued).

## 5.4 Physical-Statistic Stability

We further examine whether the predicted Navier-Stokes trajectories preserve physically relevant flow statistics. Figure 8 compares the enstrophy and palinstrophy of representative ground-truth and predicted trajectories for the two more challenging regimes, $\nu = 1 0 ^ { - 4 } \mathrm { a n d } \nu = 1 0 ^ { - 5 }$ , under both forcing families; results for $\nu = 1 0 ^ { - 3 }$ are deferred to Appendix §E.2. High enstrophy corresponds to large overall vorticity magnitude, while high palinstrophy reflects strong vorticity gradients and therefore finer-scale spatial structure.

Across the four challenging settings, the baseline failures are strongly trajectory-dependent. The largest deviations and occasional explosions occur primarily on trajectories with high ground-truth enstrophy or palinstrophy. In these cases, small prediction errors can generate excessive vorticity magnitude or spurious small-scale structure, which is reflected by increasingly inflated enstrophy and, more prominently, palinstrophy. FNO-AE is the least stable, with predicted statistics growing by several orders of magnitude in some trajectories. FNO and FNO+Noise are more stable overall, but still tend to overestimate enstrophy and especially palinstrophy on the more demanding trajectories. In contrast, FNO-VAMO tracks the scale and temporal evolution of these statistics substantially more closely and avoids the pronounced late-time inflation observed in the baselines.

![](images/3b5f71518f8e3aeadf7708c8e41bd461eba44ed335a84153107a9575a67fc5bf.jpg)  
(a) $\nu = 1 0 ^ { - 4 } ,$ , Wave forcing, Sample 26.

![](images/e8ef25b317d4b325b05c1ece1748c026fd7261fe54e0d3c69af122f12da76a0d.jpg)  
(b) $\nu = 1 0 ^ { - 5 }$ , Wave forcing, Sample 7.  
Figure 9: Representative autoregressive rollouts for the 2D incompressible Navier–Stokes equations with $\nu \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ . Each panel shows the initial condition, the trajectory-specific forcing, the ground-truth vorticity, and predictions from the four evaluated methods at selected times. All predicted snapshots use the corresponding ground-truth color scale.

The qualitative rollouts in Figure 9 provide a complementary view of the same late-time instability. The baseline predictions remain close to the reference during the early rollout but progressively develop oscillatory and small-scale artifacts beyond the training horizon, with the deterministic latent model deteriorating most severely. FNO-VAMO instead preserves the dominant vorticity structures over the long rollout. Together with the aggregate errors in Table 5, these diagnostics show that the advantage of VAMO extends beyond lower field error to improved control of spurious small-scale growth during autoregressive prediction.

## 5.5 Ablation Studies

To isolate the contributions of the main variational components, we conduct ablations on the two more challenging Navier–Stokes regimes, $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ . We consider three variants of our VAMO framework: (i) setting the KL weight to $\beta = 0$ while retaining both stochastic components, (ii) removing encoder noise, and (iii) removing transition noise.

In each case, the remaining architecture and training settings are kept unchanged. As in the main experiments, a single model is trained jointly on GRF- and Wave-forced trajectories for each viscosity and evaluated separately on the two forcing families. The aggregate results are reported in Table 6, while Figure 10 shows how the corresponding $L ^ { 2 }$ and $H ^ { 1 }$ errors evolve throughout the rollout. The temporal trends reinforce the aggregate comparison: removing the KL term leads to rapid error growth, removing encoder noise produces a clear but less severe degradation, and removing transition noise remains close to the full model.

The empirical efects of these ablations can be interpreted through the population rollout analysis in Proposition 3.12. In particular, the refined bound (3.22) gives

$$
\begin{array} { r l r } {  { ( \mathbb { E } [ \| \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } ] ) ^ { 1 / 2 } \leq \Lambda _ { D } \sum _ { j = 0 } ^ { n - 2 } \Lambda _ { T } ^ { n - 1 - j } ( \eta _ { j } + \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { j } ) + \Lambda _ { D } \eta _ { n - 1 } } } \\ & { } & { + \operatorname* { m i n } \{ \Lambda _ { D } \bar { \alpha } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n - 1 } + \delta _ { n } , \ \epsilon _ { n - 1 } + \rho _ { n - 1 } \} , } \end{array}\tag{5.1}
$$

where $\kappa _ { j }$ measures the latent KL mismatch, $\eta _ { j }$ measures the sensitivity of the decoded transition to encoder-side perturbations, and $\rho _ { j }$ measures sensitivity to transition-side perturbations. The bound highlights an important asymmetry: $\kappa _ { j }$ and $\eta _ { j }$ enter throughout the recursively accumulated rollout history and are weighted by the geometric factors $\Lambda _ { T } ^ { n - 1 - j }$ , whereas $\rho _ { n - 1 }$ appears only in the terminal predictive refinement. This provides a useful lens for interpreting the markedly diferent efects of the three ablations:

• KL consistency. Removing the KL term produces by far the largest degradation in Table 6. Across both viscosities and forcing families, the $\beta = 0$ model becomes dramatically less accurate in field-level and physical metrics, with relative $L ^ { 2 }$ , enstrophy, and spectral errors increasing by orders of magnitude. This behavior is consistent with the role of $\kappa _ { j }$ in (5.1): the KL-controlled latent mismatch enters at every previous rollout step and is accumulated with the geometric factors $\Lambda _ { T } ^ { n - 1 - j }$ . Thus, without explicit alignment between the predicted latent transition and the encoded next-state distribution, latent transition errors can be repeatedly amplified throughout autoregressive evolution.

(a) Viscosity $\nu = 1 0 ^ { - 4 } ,$  
![](images/23d551ea65017f688455c3d26d891355f27e8c9552e30cb63f57b6bdb98c8aac.jpg)

![](images/6669b9d11e7f18c3729a753a7a8c90fe13789a90906a7df87ee9a51b9bfc8bc5.jpg)  
(b) Viscosity $\nu = 1 0 ^ { - 5 }$  
Figure 10: Per-snapshot ablation results on the two-dimensional incompressible Navier-Stokes benchmark for (a) $\nu \ : = \ : 1 0 ^ { - 4 }$ and (b) $\nu \ : = \ : 1 0 ^ { - 5 }$ . We report relative $L ^ { 2 }$ and $H ^ { 1 }$ errors under both GRF and Wave forcing. The shaded region denotes the training horizon. Removing the KL term causes rapid error growth, removing encoder noise leads to a clear but less severe degradation, while removing transition noise remains close to the full VAMO model throughout the rollout.

Table 6: Ablation of the latent KL-consistency term on the two-dimensional incompressible Navier– Stokes benchmark. We compare full VAMO with a variant that sets the KL weight to $\beta = 0$ while retaining the same architecture, stochastic encoder and transition, and all other training settings. Results are reported for $\nu \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ under GRF and Wave forcing. Lower is better.
<table><tr><td>Viscosity</td><td>Forcing</td><td>Method</td><td>Rel.  $L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td><td> $\scriptstyle { { \mathcal { E } } \mathrm { - e r r o r } }$ </td><td>P-error</td><td>Spec. error</td></tr><tr><td rowspan="7"> $\nu = 1 0 ^ { - 4 }$ </td><td rowspan="5">GRF</td><td>VAMO (β = 0)</td><td>7.017</td><td>2.087</td><td>92.33</td><td>3.727</td><td>4.185</td></tr><tr><td>VAMO  $\mathrm { w / o }$  encoder noise</td><td>0.1373</td><td>1.018</td><td>0.0472</td><td>3.755</td><td>0.0257</td></tr><tr><td>VAMO  $\mathrm { w / o }$  transition noise</td><td>0.0366</td><td>0.1086</td><td>0.00977</td><td>0.0573</td><td>0.0102</td></tr><tr><td>Full VAMO</td><td>0.0403</td><td>0.1139</td><td>0.0139</td><td>0.0512</td><td>0.0146</td></tr><tr><td>VAMO (β = 0) WAVE</td><td>6.381</td><td>1.548</td><td>82.16</td><td>1.425</td><td>7.769</td></tr><tr><td>VAMO  $\mathrm { w / o }$  encoder noise</td><td>0.1827</td><td>0.4276</td><td>0.0390</td><td>0.3667</td><td>0.0492</td></tr><tr><td rowspan="6">GRF</td><td>VAMO  $\mathrm { w / o }$  transition noise</td><td>0.1181</td><td>0.2128</td><td>0.0349</td><td>0.1026</td><td>0.0303</td></tr><tr><td>Full VAMO</td><td>0.1198</td><td>0.2212</td><td>0.0386</td><td>0.1008</td><td>0.0390</td></tr><tr><td>VAMO (β = 0) VAMO  $\mathrm { w / o }$ </td><td>17.54</td><td>2.098</td><td>614.8</td><td>3.145</td><td>28.56</td></tr><tr><td>encoder noise VAMO transition noise</td><td>0.1105</td><td>0.5422</td><td>0.0241</td><td>0.4616</td><td>0.0164</td></tr><tr><td> $\mathrm { w / o }$  Full VAMO</td><td>0.0844</td><td>0.3870</td><td>0.0224</td><td>0.2371</td><td>0.0103</td></tr><tr><td></td><td>0.0841</td><td>0.3858</td><td>0.0211</td><td>0.2415</td><td>0.00929</td></tr><tr><td rowspan="4"> $\nu = 1 0 ^ { - 5 }$  WAVE</td><td>VAMO  $( \beta = 0 )$ </td><td>14.55</td><td>1.573</td><td>424.7</td><td>1.329</td><td>41.27</td></tr><tr><td>VAMO  $\mathrm { w / o }$  encoder noise</td><td>0.1743</td><td>0.5676</td><td>0.0405</td><td>0.3532</td><td>0.0268</td></tr><tr><td>VAMO  $\mathrm { w / o }$  transition noise</td><td>0.1523</td><td>0.4555</td><td>0.0488</td><td>0.2820</td><td>0.0271</td></tr><tr><td>Full VAMO</td><td>0.1469</td><td>0.4509</td><td>0.0460</td><td>0.2734</td><td>0.0247</td></tr></table>

• Encoder noise. Removing encoder noise also consistently worsens performance, although less severely than removing the KL term. The degradation is particularly visible in derivativesensitive and physical-statistic errors. This observation agrees with the role of the encoderneighborhood sensitivity $\eta _ { j }$ , which, like $\kappa _ { j }$ , appears throughout the geometrically weighted rollout sum. Encoder-side perturbations expose the learned transition to a neighborhood of the encoded trajectory during training, thereby encouraging reduced sensitivity to deviations from the nominal latent state. The ablation suggests that this neighborhood regularization is important precisely because such input-side sensitivity can be propagated and amplified over many subsequent rollout steps.

• Transition noise. In contrast, removing transition noise afects the performance of VAMO only marginally. The ablated model remains close to full VAMO and is slightly better on several metrics for $\nu = 1 0 ^ { - 4 }$ , while full VAMO is generally competitive or marginally better for $\nu = 1 0 ^ { - 5 }$ . This observation is consistent with the refined rollout bound: unlike $\kappa _ { j }$ and $\eta _ { j }$ the transition-noise sensitivity $\rho _ { n } .$ <sub>−1</sub> appears only in the terminal predictive refinement and does not enter the geometrically amplified rollout history. Thus, transition noise appears to act primarily as a secondary local regularizer whose contribution is more dependent on the particular dynamical regime.

Taken together, the ablations reveal a clear empirical hierarchy among the variational components. Latent KL consistency is essential for stable evolution, encoder-side neighborhood regularization provides a second substantial source of robustness, and transition noise has a smaller, problem-dependent efect. This hierarchy closely mirrors the structure of the rollout analysis: quantities governing latent consistency and input-side sensitivity enter the recursively amplified part of the error bound, whereas transition-output sensitivity enters only the terminal predictive refinement. While the ablations modify the perturbation distributions and therefore should not be viewed as direct consequences of the bound, their behavior provides empirical support for the mechanism identified by the theory.

## 6 Conclusion and Outlook

In this work, we developed a variational framework for learning time-dependent PDE dynamics with an emphasis on long-horizon autoregressive prediction. The proposed formulation represents physical states through latent distributions and models their evolution with a probabilistic Markov transition, while retaining deterministic mean dynamics at inference. At the theoretical level, we formulated the model directly on function spaces, characterized the geometry induced by functional Gaussian distributions, and connected stochastic latent perturbations and variational transition alignment to the propagation of deterministic rollout errors. We instantiated this framework as the Variational Autoencoding Markov Operator (VAMO), which combines spatially resolved latent representations, structured Gaussian perturbations, and a neural-operator transition. Across the compressible Euler, compressible-flow, and incompressible Navier–Stokes benchmarks, VAMO consistently exhibited slower error accumulation and improved long-horizon stability compared with direct, deterministic-latent, and generic noise-injection baselines.

Limitations. This work also leaves several limitations. Our theoretical analysis is intended primarily to expose the mechanisms through which the variational objective enters autoregressive error propagation; the resulting bounds rely on regularity assumptions and are not expected to be quantitatively tight for general nonlinear PDE dynamics. For our empirical study, the current realization uses an FNO-based latent transition and spectrally structured Gaussian perturbations on regular grids, and our experiments focus on fluid-dynamics systems with a fixed local prediction interval. The results therefore do not establish that the same architecture or stochastic parameterization is universally preferable across other PDE classes, geometries, discretizations, or temporal training regimes.

Future directions. Several extensions of this work are natural. First, the present objective is formulated from local transition pairs. A multi-step variational objective could instead be derived by introducing latent distributions over several consecutive transitions, providing a principled basis for combining variational dynamics with multi-step or curriculum-based training. Such a formulation may further reduce the mismatch between training and long autoregressive deployment. Second, VAMO is not tied to the FNO transition used in our experiments: the latent evolution operator could be replaced by geometry-aware or attention-based architectures such as Geo-FNO, OFormer, or PiT, allowing the variational formulation to be studied together with spatial-resolution transfer, irregular geometries, and more flexible discretizations. More broadly, the probabilistic latent dynamics developed here provide a starting point for studying how distributional modeling and deterministic long-horizon prediction can be combined in neural PDE solvers.

More broadly, the variational latent-dynamics framework developed here provides a starting point for studying how probabilistic modeling can support reliable deterministic evolution in neural PDE solvers and operator learning.

## References

Alberts, A. and Bilionis, I. (2023). Physics-informed information field theory for modeling physical systems with uncertainty quantification. Journal of Computational Physics, 486 112100.

Bastek, J.-H., Sun, W. and Kochmann, D. (2025). Physics-informed difusion models. In International Conference on Learning Representations, vol. 2025.

Bhattacharya, K., Hosseini, B., Kovachki, N. B. and Stuart, A. M. (2021). Model reduction and neural networks for parametric PDEs. The SMAI journal of computational mathematics, 7 121– 157.

Bishop, C. M. (1995). Training with noise is equivalent to Tikhonov regularization. Neural computation, 7 108–116.

Bogachev, V. I. (1998). Gaussian measures. 62, American Mathematical Soc.

Brandstetter, J., Worrall, D. and Welling, M. (2022). Message passing neural PDE solvers. arXiv preprint arXiv:2202.03376.

B¨ulte, C., Scholl, P. and Kutyniok, G. (2025). Probabilistic neural operators for functional uncertainty quantification. Transactions on Machine Learning Research.

Bunker, J., Girolami, M., Lambley, H., Stuart, A. M. and Sullivan, T. J. (2025). Autoencoders in function space. Journal of Machine Learning Research, 26 1–54.

Canuto, C., Hussaini, M. Y., Quarteroni, A. and Zang, T. A. (1988). Spectral methods in fluid dynamics. Springer.

Cao, S. (2021). Choose a Transformer: Fourier or Galerkin. Advances in neural information processing systems, 34 24924–24940.

Chen, J. and Wu, K. (2024). Positional knowledge is all you need: Position-induced Transformer (PiT) for operator learning. arXiv preprint arXiv:2405.09285.

Crank, J. and Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial diferential equations of the heat-conduction type. In Mathematical proceedings of the Cambridge philosophical society, vol. 43. Cambridge University Press.

Da Prato, G. and Zabczyk, J. (2014). Stochastic equations in infinite dimensions. Cambridge university press.

Dashti, M. and Stuart, A. M. (2013). The Bayesian approach to inverse problems. arXiv preprint arXiv:1302.6989.

Diab, W. and Al Kobaisi, M. (2025). Temporal neural operator for modeling time-dependent physical phenomena. Scientific Reports, 15 32791.

Garg, S. and Chakraborty, S. (2023). VB-DeepONet: A Bayesian operator learning framework for uncertainty quantification. Engineering Applications of Artificial Intelligence, 118 105685.

Geneva, N. and Zabaras, N. (2020). Modeling the dynamics of PDE systems with physicsconstrained deep auto-regressive networks. Journal of Computational Physics, 403 109056.

Geneva, N. and Zabaras, N. (2022). Transformers for modeling physical systems. Neural Networks, 146 272–289.

Gupta, G., Xiao, X. and Bogdan, P. (2021). Multiwavelet-based operator learning for diferential equations. Advances in neural information processing systems, 34 24048–24062.

Ha, J.-S., Park, Y.-J., Chae, H.-J., Park, S.-S. and Choi, H.-L. (2018). Adaptive path-integral autoencoders: Representation learning and planning for dynamical systems. Advances in Neural Information Processing Systems, 31.

Hagnberger, J., Musekamp, D. and Niepert, M. (2025). CALM-PDE: Continuous and adaptive convolutions for latent space modeling of time-dependent PDEs. Advances in Neural Information Processing Systems, 39 160431–160489.

Haitsiukevich, K., Poyraz, O., Marttinen, P. and Ilin, A. (2024). Difusion models as probabilistic neural operators for recovering unobserved states of dynamical systems. In 2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP). IEEE.

Hao, Z., Wang, Z., Su, H., Ying, C., Dong, Y., Liu, S., Cheng, Z., Song, J. and Zhu, J. (2023). GNOT: A general neural operator Transformer for operator learning. In International Conference on Machine Learning. PMLR.

Hasan, A., Pereira, J. M., Farsiu, S. and Tarokh, V. (2021). Identifying latent stochastic diferential equations. IEEE Transactions on Signal Processing, 70 89–104.

He, K., Zhang, X., Ren, S. and Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition.

Hoop, M. V. d., Huang, D. Z., Qian, E. and Stuart, A. M. (2022). The cost-accuracy trade-of in operator learning with neural networks. Journal of Machine Learning, 1 299–341.

Hsu, D., Kakade, S. and Zhang, T. (2012). A tail inequality for quadratic forms of subgaussian random vectors. Electronic Communications in Probability, 17 1–6.

Jin, P., Meng, S. and Lu, L. (2022). MIONet: Learning multiple-input operators via tensor product. SIAM Journal on Scientific Computing, 44 A3490–A3514.

Kingma, D. P. and Welling, M. (2014). Auto-encoding variational Bayes. In International Conference on Learning Representations.

Kissas, G., Seidman, J. H., Guilhoto, L. F., Preciado, V. M., Pappas, G. J. and Perdikaris, P. (2022). Learning operators with coupled attention. Journal of Machine Learning Research, 23 1–63.

Kontolati, K., Goswami, S., Em Karniadakis, G. and Shields, M. D. (2024). Learning nonlinear operators in latent spaces for real-time predictions of complex dynamics in physical systems. Nature Communications, 15 5101.

Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A. and Anandkumar, A. (2023). Neural operator: Learning maps between function spaces with applications to PDEs. Journal of Machine Learning Research, 24 1–97.

Krishnan, R., Shalit, U. and Sontag, D. (2017). Structured inference networks for nonlinear state space models. In Proceedings of the AAAI conference on artificial intelligence, vol. 31.

Krishnan, R. G., Shalit, U. and Sontag, D. (2015). Deep Kalman filters. arXiv preprint arXiv:1511.05121.

Kuo, H.-H. (2006). Gaussian measures in Banach spaces. In Gaussian measures in banach spaces. Springer, 1–109.

Li, Z., Huang, D. Z., Liu, B. and Anandkumar, A. (2023). Fourier neural operator with learned deformations for PDEs on general geometries. Journal of Machine Learning Research, 24 1–26.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A. and Anandkumar, A. (2020a). Neural operator: Graph kernel network for partial diferential equations. arXiv preprint arXiv:2003.03485.

Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Stuart, A., Bhattacharya, K. and Anandkumar, A. (2020b). Multipole graph neural operator for parametric partial diferential equations. Advances in neural information processing systems, 33 6755–6766.

Li, Z., Kovachki, N. B., Azizzadenesheli, K., liu, B., Bhattacharya, K., Stuart, A. and Anandkumar, A. (2021). Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations.

Li, Z., Meidani, K. and Farimani, A. B. (2022). Transformer for partial diferential equations operator learning. arXiv preprint arXiv:2205.13671.

Li, Z., Zheng, H., Kovachki, N., Jin, D., Chen, H., Liu, B., Azizzadenesheli, K. and Anandkumar, A. (2024). Physics-informed neural operator for learning partial diferential equations. ACM/IMS Journal of Data Science, 1 1–27.

Lin, G., Moya, C. and Zhang, Z. (2023). B-DeepONet: An enhanced Bayesian DeepONet for solving noisy parametric PDEs using accelerated replica exchange SGLD. Journal of Computational Physics, 473 111713.

Linot, A. J., Burby, J. W., Tang, Q., Balaprakash, P., Graham, M. D. and Maulik, R. (2023). Stabilized neural ordinary diferential equations for long-time forecasting of dynamical systems. Journal of Computational Physics, 474 111838.

Lippe, P., Veeling, B., Perdikaris, P., Turner, R. and Brandstetter, J. (2023). PDE-refiner: Achieving accurate long rollouts with neural PDE solvers. Advances in Neural Information Processing Systems, 36 67398–67433.

Lu, L., Jin, P., Pang, G., Zhang, Z. and Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature machine intelligence, 3 218–229.

Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z. and Karniadakis, G. E. (2022). A comprehensive and fair comparison of two neural operators (with practical extensions) based on fair data. Computer Methods in Applied Mechanics and Engineering, 393 114778.

Lusch, B., Kutz, J. N. and Brunton, S. L. (2018). Deep learning for universal linear embeddings of nonlinear dynamics. Nature communications, 9 4950.

Micha lowska, K., Goswami, S., Karniadakis, G. E. and Riemer-Sørensen, S. (2024). Neural operator learning for long-time integration in dynamical systems with recurrent neural networks. In 2024 International joint conference on neural networks (IJCNN). IEEE.

Morton, J., Jameson, A., Kochenderfer, M. J. and Witherden, F. (2018). Deep dynamical modeling and control of unsteady fluid flows. Advances in neural information processing systems, 31.

Orszag, S. A. (1971). Numerical simulation of incompressible flows within simple boundaries. I. Galerkin (spectral) representations. Studies in applied mathematics, 50 293–327.

Otto, F. and Villani, C. (2000). Generalization of an inequality by Talagrand and links with the logarithmic Sobolev inequality. Journal of Functional Analysis, 173 361–400.

Pan, S. and Duraisamy, K. (2020). Physics-informed probabilistic learning of linear embeddings of nonlinear dynamics with guaranteed stability. SIAM Journal on Applied Dynamical Systems, 19 480–509.

Pang, G., Lu, L. and Karniadakis, G. E. (2019). fPINNs: Fractional physics-informed neural networks. SIAM Journal on Scientific Computing, 41 A2603–A2626.

Pfaf, T., Fortunato, M., Sanchez-Gonzalez, A. and Battaglia, P. W. (2020). Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409.

Rahman, M. A., Florez, M. A., Anandkumar, A., Ross, Z. E. and Azizzadenesheli, K. (2022). Generative adversarial neural operators. Transactions on Machine Learning Research.

Rahman, M. A., Ross, Z. E. and Azizzadenesheli, K. (2023). U-NO: U-shaped neural operators. Transactions on Machine Learning Research.

Raissi, M., Perdikaris, P. and Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial diferential equations. Journal of Computational physics, 378 686–707.

Rangapuram, S. S., Seeger, M. W., Gasthaus, J., Stella, L., Wang, Y. and Januschowski, T. (2018). Deep state space models for time series forecasting. Advances in neural information processing systems, 31.

Riedel, S. (2017). Transportation–cost inequalities for difusions driven by Gaussian processes. Electronic Journal of Probability, 22 1 – 26.

Rubanova, Y., Chen, R. T. and Duvenaud, D. K. (2019). Latent ordinary diferential equations for irregularly-sampled time series. Advances in neural information processing systems, 32.

Rusanov, V. V. (1962). The calculation of the interaction of non-stationary shock waves and obstacles. USSR Computational Mathematics and Mathematical Physics, 1 304–320.

Sanchez-Gonzalez, A., Godwin, J., Pfaf, T., Ying, R., Leskovec, J. and Battaglia, P. (2020). Learning to simulate complex physics with graph networks. In International conference on machine learning. PMLR.

Seidman, J., Kissas, G., Perdikaris, P. and Pappas, G. J. (2022). NOMAD: Nonlinear manifold decoders for operator learning. Advances in Neural Information Processing Systems, 35 5601– 5613.

Seidman, J. H., Kissas, G., Pappas, G. J. and Perdikaris, P. (2023). Variational autoencoding neural operators. arXiv preprint arXiv:2302.10351.

Serrano, L., Vittaut, J.-N. and Gallinari, P. (2024a). Latent difusion Transformer with local neural field as PDE surrogate model. In ICLR 2024 Workshop on AI4DiferentialEquations In Science.

Serrano, L., Wang, T. X., Le Naour, E., Vittaut, J.-N. and Gallinari, P. (2024b). AROMA: Preserving spatial structure for latent PDE modeling with local neural fields. Advances in Neural Information Processing Systems, 37 13489–13521.

Shu, C.-W. and Osher, S. (1988). Eficient implementation of essentially non-oscillatory shockcapturing schemes. Journal of computational physics, 77 439–471.

Stachenfeld, K., Fielding, D. B., Kochkov, D., Cranmer, M., Pfaf, T., Godwin, J., Cui, C., Ho, S., Battaglia, P. and Sanchez-Gonzalez, A. (2021). Learned coarse models for eficient turbulence simulation. arXiv preprint arXiv:2112.15275.

Stroock, D. W. (2010). Probability theory: an analytic view. Cambridge university press.

Stuart, A. M. (2010). Inverse problems: a Bayesian perspective. Acta numerica, 19 451–559.

Takamoto, M., Alesiani, F. and Niepert, M. (2023). Learning neural PDE solvers with parameterguided channel attention. In International Conference on Machine Learning. PMLR.

Takamoto, M., Praditia, T., Leiteritz, R., MacKinlay, D., Alesiani, F., Pfl¨uger, D. and Niepert, M. (2022). PDEBench: An extensive benchmark for scientific machine learning. Advances in neural information processing systems, 35 1596–1611.

Talagrand, M. (1996). Transportation cost for Gaussian and other product measures. Geometric & Functional Analysis GAFA, 6 587–600.

Toro, E. F., Spruce, M. and Speares, W. (1994). Restoration of the contact surface in the HLL-Riemann solver. Shock waves, 4 25–34.

Tzen, B. and Raginsky, M. (2019). Neural stochastic diferential equations: Deep latent Gaussian models in the difusion limit. arXiv preprint arXiv:1905.09883.

Van Leer, B. (1979). Towards the ultimate conservative diference scheme. v. a second-order sequel to Godunov’s method. Journal of computational Physics, 32 101–136.

Wang, S., Wang, H. and Perdikaris, P. (2021). Learning the solution operator of parametric partial diferential equations with physics-informed DeepONets. Science advances, 7 eabi8605.

Wang, T. and Wang, C. (2024). Latent neural operator for solving forward and inverse PDE problems. Advances in Neural Information Processing Systems, 37 33085–33107.

Wiewel, S., Becher, M. and Thuerey, N. (2019). Latent space physics: Towards learning the temporal evolution of fluid flow. In Computer graphics forum, vol. 38. Wiley Online Library.

Wu, H., Luo, H., Wang, H., Wang, J. and Long, M. (2024). Transolver: A fast Transformer solver for PDEs on general geometries. In International Conference on Machine Learning.

Wu, Z., Wang, S., Zhang, S., He, S., Zhu, M., Jiao, A., Lu, L. and van Dijk, D. (2026). TANTE: Time-adaptive operator learning via neural Taylor expansion. Journal of Computational Physics, 562 115041.

Xiao, Z., Hao, Z., Lin, B., Deng, Z. and Su, H. (2023). Improved operator learning by orthogonal attention. arXiv preprint arXiv:2310.12487.

Yang, Y., Kissas, G. and Perdikaris, P. (2022). Scalable uncertainty quantification for deep operator networks using randomized priors. Computer Methods in Applied Mechanics and Engineering, 399 115399.

Yildiz, C., Heinonen, M. and Lahdesmaki, H. (2019). ODE2VAE: Deep generative second order ODEs with Bayesian neural networks. Advances in Neural Information Processing Systems, 32.

Yin, Y., Kirchmeyer, M., Franceschi, J.-Y., Rakotomamonjy, A. and Gallinari, P. (2023). Continuous PDE dynamics forecasting with implicit neural representations. In The Eleventh International Conference on Learning Representations.

Zhu, Y. and Zabaras, N. (2018). Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification. Journal of Computational Physics, 366 415–447.

Zhu, Y., Zabaras, N., Koutsourelakis, P.-S. and Perdikaris, P. (2019). Physics-constrained deep learning for high-dimensional surrogate modeling and uncertainty quantification without labeled data. Journal of computational physics, 394 56–81.

## A Gaussian Measures and Cameron-Martin Spaces

## A.1 Gaussian Measures on Banach Spaces

We briefly review several basic notions for Gaussian measures on Banach spaces (Bogachev, 1998; Kuo, 2006). Let X be a real separable Banach space equipped with its Borel σ-algebra, and let $\mathcal { X } ^ { \ast }$ denote its dual space. A Borel probability measure $\mu$ on $\mathcal { X }$ is called a Gaussian measure if, for every continuous linear functional $f \in \mathcal { X } ^ { * }$ , the pushforward

$$
f _ { \sharp } \mu = \mu \circ f ^ { - 1 }
$$

is a Gaussian probability measure on $\mathbb { R }$ , possibly degenerate. Equivalently, if a random element $u \sim \mu ,$ then $f ( u )$ is a real-valued Gaussian random variable for every $f \in \mathcal { X } ^ { * }$

The mean of $\mu$ is defined weakly through its action on linear functionals. For each $f \in \mathcal { X } ^ { * }$ , one writes

$$
m _ { \mu } ( f ) : = \int _ { \mathcal { X } } f ( u ) \mathrm { d } \mu ( u ) .
$$

This defines a linear functional on $\mathcal { X } ^ { \ast }$ , hence an element of the bidual $\boldsymbol { \mathcal { X } ^ { * * } }$ . If there exists an element $m \in { \mathcal { X } }$ such that

$$
f ( m ) = \int _ { \mathcal { X } } f ( x ) \mathrm { d } \mu ( x ) , \qquad \forall f \in \mathcal { X } ^ { * } ,
$$

then $m$ is called the mean element of $\mu .$ In this case we write $m = \mathbb { E } _ { X \sim \mu } [ X ]$ in the weak sense. For Gaussian measures on separable Banach spaces, the mean can typically be identified with an element of the underlying Banach space $\mathcal { X }$

The covariance of $\mu$ is also naturally defined through dual pairings. Assuming that $\mu$ has mean m $\in \mathcal { X }$ , its covariance bilinear form is

$$
C _ { \mu } ( f , g ) : = \int _ { \mathcal X } \bigl ( f ( u ) - f ( m ) \bigr ) \bigl ( g ( u ) - g ( m ) \bigr ) \mathrm { d } \mu ( u ) , \qquad f , g \in \mathcal X ^ { * } .
$$

Equivalently, $C _ { \mu } ( f , g ) = \mathrm { C o v } _ { X \sim \mu } ( f ( X ) , g ( X ) )$ . Thus $C _ { \mu } : \mathcal { X } ^ { * } \times \mathcal { X } ^ { * }  \mathbb { R }$ is a positive semidefinite bilinear form. In particular,

$$
C _ { \mu } ( f , f ) = \operatorname { V a r } _ { \mu } ( f ( u ) ) \geq 0 .
$$

The covariance bilinear form induces a covariance operator. Abstractly, one may define a mapping $K _ { \mu } : \mathcal { X } ^ { * } \to \mathcal { X } ^ { * * }$ by

$$
( K _ { \mu } f ) ( g ) : = C _ { \mu } ( f , g ) , \qquad f , g \in \mathcal { X } ^ { * } .
$$

When $K _ { \mu } f$ can be identified with an element of $\mathcal { X }$ , we regard the covariance operator as a mapping $K _ { \mu } : { \mathcal { X } } ^ { * } \to { \mathcal { X } }$ satisfying

$$
g ( K _ { \mu } f ) = C _ { \mu } ( f , g ) , \qquad f , g \in \mathcal { X } ^ { * } .
$$

This is the natural Banach-space analogue of the finite-dimensional covariance matrix. Indeed, if $\chi = \mathbb { R } ^ { d }$ and $\mu = \Nu ( m , \Sigma )$ , then each $f \in { \mathcal { X } } ^ { * }$ has the form $f ( u ) = a ^ { \top } u$ for some $a \in \mathbb { R } ^ { d }$ , and

$$
C _ { \mu } ( \boldsymbol { f } , \boldsymbol { g } ) = \boldsymbol { a } ^ { \intercal } \Sigma \boldsymbol { b } , \qquad \boldsymbol { g } ( \boldsymbol { u } ) = \boldsymbol { b } ^ { \intercal } \boldsymbol { u } .
$$

In this case the covariance operator is simply multiplication by the covariance matrix $\Sigma .$

## A.2 Abstract Wiener Space and the Cameron-Martin Theorem

We next review the Cameron-Martin space associated with a Gaussian measure on a Banach space.   
A convenient framework is provided by the notion of an abstract Wiener space (Stroock, 2010).

Definition A.1 (Abstract Wiener space). Let H be a real separable Hilbert space and let $\mathcal { X }$ be a real separable Banach space. Suppose that H is continuously and densely embedded into X through an injective linear map

$$
\iota : \mathcal { H } \hookrightarrow \mathcal { X } .
$$

The triple $( { \mathcal { H } } , { \mathcal { X } } , \mu )$ is called an abstract Wiener space if $\mu$ is a centered Gaussian measure on $\mathcal { X }$ whose covariance structure is induced by the embedding ι in the following sense: for every $f \in \mathcal { X } ^ { * }$ the pushforward $f _ { \sharp } \mu$ is a centered real Gaussian measure satisfying

$$
\int _ { \mathcal { X } } \boldsymbol { f } ( \boldsymbol { u } ) ^ { 2 } \mathrm { d } \mu ( \boldsymbol { u } ) = \| \boldsymbol { \iota } ^ { * } \boldsymbol { f } \| _ { \mathcal { H } } ^ { 2 } ,
$$

where $\iota ^ { * } : \mathcal { X } ^ { * }  \mathcal { H }$ is the adjoint map defined by

$$
\langle \iota ^ { * } f , h \rangle _ { \mathcal { H } } = f ( \iota h ) , \qquad f \in \mathcal { X } ^ { * } , \ h \in \mathcal { H } .
$$

The Hilbert space $\mathcal { H }$ is called the Cameron-Martin space of $\mu .$

Equivalently, the covariance operator $K _ { \mu } : { \mathcal { X } } ^ { * } \to { \mathcal { X } }$ satisfies

$$
K _ { \mu } = \iota \iota ^ { * }
$$

in the weak sense that

$$
f ( K _ { \mu } g ) = \langle \iota ^ { * } f , \iota ^ { * } g \rangle _ { \mathcal { H } } , \qquad f , g \in \mathcal { X } ^ { * } .
$$

Thus the Cameron-Martin inner product determines the covariance structure of all continuous linear observations of the Gaussian random element.

We now define the Paley-Wiener map associated with the abstract Wiener space. For any $f , g \in { \mathcal { X } } ^ { * }$ , we have

$$
\mathbb { E } _ { X \sim \mu } \left[ f ( X ) g ( X ) \right] = \langle \iota ^ { * } f , \iota ^ { * } g \rangle \varkappa .
$$

Hence the assignment $\iota ^ { * } f \mapsto f ( \cdot )$ is an isometry from the subspace $\iota ^ { * } ( { \mathcal { X } } ^ { * } ) \subset { \mathcal { H } }$ into $L ^ { 2 } ( \mathcal { X } , \mu )$ Since $\iota ^ { * } ( { \mathcal { X } } ^ { * } )$ is dense in H for an abstract Wiener space, this isometry extends uniquely to all of H. This extension is called the Paley-Wiener map. For $h \in \mathcal H$ , we denote its image by

$$
\langle h , \cdot \rangle \tilde { \mathbf { \Gamma } } \in L ^ { 2 } ( \mathcal { X } , \mu ) .
$$

Thus $\langle h , u \rangle ^ { \sim }$ is a centered Gaussian random variable satisfying

$$
\mathbb { E } _ { \mu } \left[ \langle h , \cdot \rangle ^ { \sim } \langle k , \cdot \rangle ^ { \sim } \right] = \langle h , k \rangle \varkappa , \qquad h , k \in \mathcal { H } .
$$

In particular, for $f \in { \mathcal { X } } ^ { * }$ 2

$$
\langle \iota ^ { * } f , x \rangle ^ { \sim } = f ( x ) , \qquad \mu \mathrm { - a . s . }
$$

This identity is the main point of the construction: the Paley-Wiener map extends ordinary linear observations $f ( x )$ from directions of the form $\iota ^ { * } f$ to arbitrary Cameron-Martin directions $h \in \mathcal H$

The key property of the Cameron-Martin space is that it characterizes the directions along which the Gaussian measure can be translated without changing its measure class. For $h \in { \mathcal { X } }$ define the translated measure

$$
\mu _ { h } ( A ) : = \mu ( A - h ) , \qquad A \in { \mathcal { B } } ( { \mathcal { X } } ) .
$$

Equivalently, if $u \sim \mu .$ , then $u + h \sim \mu _ { h }$

Theorem A.2 (Cameron-Martin theorem). Let $( { \mathcal { H } } , { \mathcal { X } } , \mu )$ be an abstract Wiener space. For $h \in \mathcal { X }$ let $\mu _ { h }$ be the translated measure defined by $\mu _ { h } = \mu ( \cdot - h )$ . Then $\mu _ { h } \ll \mu$ if and only if $h \in \iota ( { \mathcal { H } } )$ Identifying H with its image $\iota ( { \mathcal { H } } ) \subset { \mathcal { X } }$ , if $h \in \mathcal H$ , then

$$
\frac { \mathrm { d } \mu _ { h } } { \mathrm { d } \mu } ( u ) = \exp \left( \langle h , u \rangle ^ { \sim } - \frac { 1 } { 2 } \| h \| _ { \mathcal { H } } ^ { 2 } \right) .
$$

If $h \not \in \iota ( \mathcal { H } )$ , then $\mu _ { h } \perp \mu .$

Here the notation $\langle h , u \rangle ^ { \sim }$ should not be confused with the inner product $\langle h , u \rangle _ { \mathcal { H } }$ in the Hilbert space. In infinite dimensions, a typical sample $u \sim \mu$ does not belong to ${ \mathcal { H } } ,$ so $\langle h , u \rangle _ { \mathcal { H } }$ is generally not defined. The Paley-Wiener variable $\langle h , u \rangle ^ { \sim }$ is the rigorous substitute for this formal pairing.

Construction from a Gaussian measure. Conversely, the preceding abstract Wiener-space structure can be constructed from a centered Gaussian measure on a separable Banach space. Let $\mu$ be a centered Gaussian measure on X with covariance operator $K _ { \mu } : { \mathcal { X } } ^ { * } \to { \mathcal { X } }$ , and let

$$
\mathcal G _ { \mu } : = \overline { { \mathscr { X } ^ { * } } } ^ { L ^ { 2 } ( \mathscr { X } , \mu ) }
$$

denote the closure of the continuous linear functionals in $L ^ { 2 } ( \mathcal { X } , \mu )$ . For $f \in { \mathcal { G } } _ { \mu }$ , define $R _ { \mu } f \in { \mathcal { X } }$ weakly by

$$
R _ { \mu } f : = \int _ { \mathcal { X } } x f ( x ) \mathrm { d } \mu ( x ) ,
$$

meaning that

$$
g ( R _ { \mu } f ) = \int _ { \mathcal { X } } g ( x ) f ( x ) \mathrm { d } \mu ( x ) , \qquad \forall g \in \mathcal { X } ^ { * } .
$$

For $f \in { \mathcal { X } } ^ { * }$ , one has

$$
R _ { \mu } f = K _ { \mu } f .
$$

The corresponding Cameron–Martin space is

$$
\begin{array} { r } { \mathcal { H } _ { \mu } : = R _ { \mu } ( \mathcal G _ { \mu } ) \subset \mathcal X , } \end{array}
$$

equipped with the inner product induced from $L ^ { 2 } ( \mathcal { X } , \mu )$

$$
\langle R _ { \mu } f , R _ { \mu } g \rangle _ { \mathcal { H } _ { \mu } } : = \int _ { \mathcal { X } } f ( x ) g ( x ) \mathrm { d } \mu ( x ) .
$$

With the canonical embedding $\iota : { \mathcal { H } } _ { \mu } \hookrightarrow { \mathcal { X } }$ , the triple $( { \mathcal { H } } _ { \mu } , { \mathcal { X } } , \mu )$ is an abstract Wiener space of the form described above. Moreover, the inverse map

$$
R _ { \mu } ^ { - 1 } : { \mathcal { H } } _ { \mu } \to { \mathcal { G } } _ { \mu } \subset L ^ { 2 } ( { \mathcal { X } } , { \mu } )
$$

coincides with the Paley-Wiener map:

$$
\langle h , \cdot \rangle \widetilde { \bf \Phi } = R _ { \mu } ^ { - 1 } h .
$$

In particular, for $h = K _ { \mu } f$ with $f \in \mathcal { X } ^ { * }$ 2

$$
\langle h , x \rangle ^ { \sim } = \langle K _ { \mu } f , x \rangle ^ { \sim } = f ( x ) , \qquad \mu { \mathrm { - a . s . } }
$$

This identity explains why the Paley-Wiener map provides the rigorous counterpart of the formal pairing $\langle h , x \rangle _ { \mathcal { H } _ { \mu } }$ . In infinite dimensions, a typical sample $x \sim \mu$ need not belong to ${ \mathcal { H } } _ { \mu } ,$ so this Hilbert-space inner product is generally not defined.

Specializing Theorem A.2 to the construction above gives the following form of Cameron-Martin theorem, which is used later in our analysis.

Theorem A.3 (Cameron-Martin theorem for Gaussian measures). Let $\mu$ be a centered Gaussian measure on a real, separable Banach space $\mathcal { X } _ { : }$ , and let ${ \mathcal { H } } _ { \mu }$ be its Cameron-Martin space. For $h \in \mathcal { X }$ the translated measure $\mu _ { h } \ll \mu$ if and only if $h \in { \mathcal { H } } _ { \mu }$ . Moreover, if $h \in { \mathcal { H } } _ { \mu }$ , then

$$
\frac { \mathrm { d } \mu _ { h } } { \mathrm { d } \mu } ( u ) = \exp \left( \langle h , u \rangle ^ { \sim } - \frac { 1 } { 2 } \| h \| _ { \mathcal { H } _ { \mu } } ^ { 2 } \right) .
$$

If h $, \notin \mathcal { H } _ { \mu }$ , then $\mu _ { h } \perp \mu .$

For a non-centered Gaussian measure with mean m $\in \mathcal { X }$ and the same covariance structure, the Cameron-Martin space is unchanged. In that case, for $h \in { \mathcal { H } } _ { \mu }$

$$
\frac { \mathrm { d } { \mathsf { N } } ( m + h , K _ { \mu } ) } { \mathrm { d } { \mathsf { N } } ( m , K _ { \mu } ) } ( u ) = \exp \left( \langle h , u - m \rangle ^ { \sim } - \frac { 1 } { 2 } \| h \| _ { \mathcal { H } _ { \mu } } ^ { 2 } \right) .
$$

Thus the Cameron-Martin norm is the infinite-dimensional analogue of the Mahalanobis norm. In particular, shifts outside ${ \mathcal { H } } _ { \mu }$ produce singular Gaussian measures, while shifts inside ${ \mathcal { H } } _ { \mu }$ preserve absolute continuity.

## B Proofs and Auxiliary Results

This appendix collects the proofs and auxiliary results supporting the theoretical developments of §3. The organization follows the progression of the main text: variational bounds, functional Gaussian geometry, local sensitivity under Gaussian perturbations, KL-aware deterministic rollout, and finally the high-probability stochastic extension.

## B.1 Variational Bounds and Population Alignment

We first provide the proofs of the abstract variational results in §3.1. We begin with the one-step variational bounds and then turn to the population decomposition of the latent dynamics term.

Proof of Proposition 3.2. (a) Define the latent predictive kernel

$$
K _ { \theta , \phi } ( \mathrm { d } \mathbf { Z } _ { n + 1 } , \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) = P _ { \theta } ( \mathrm { d } \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) Q _ { \phi } ( \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) .
$$

Then

$$
P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) = \int _ { \mathcal { Z } \times \mathcal { Z } } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) K _ { \theta , \phi } ( \mathrm { d } \mathbf { Z } _ { n + 1 } , \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) ,
$$

and W<sub>U</sub>-almost every ${ \bf U } _ { n + 1 }$ , we have

$$
\frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) = \int _ { \mathbb { Z } \times \mathcal { Z } } \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) K _ { \theta , \phi } ( \mathrm { d } \mathbf { Z } _ { n + 1 } , \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) .
$$

By Jensen’s inequality,

$$
\begin{array} { r l } & { \log \frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot | \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \geq \int _ { \mathcal { Z } \times \mathcal { Z } } \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) K _ { \theta , \phi } ( \mathrm { d } \mathbf { Z } _ { n + 1 } , \mathrm { d } \mathbf { Z } _ { n } | \mathbf { U } _ { n } ) } \\ & { \qquad = \mathbb { E } _ { \mathbf { Z } _ { n + 1 } , \mathbf { Z } _ { n } \sim K _ { \theta , \phi } ( \cdot | \mathbf { U } _ { n } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] . } \end{array}
$$

Expanding $K _ { \theta , \phi }$ gives (3.1).

(b) For fixed $\mathbf { Z } _ { n } \in { \mathcal { Z } }$ , define the latent-conditioned predictive distribution

$$
P _ { \theta , \psi } ( \mathrm { d } \mathbf { U } _ { n + 1 } \mid \mathbf { Z } _ { n } ) = \int _ { \mathcal { Z } } P _ { \psi } ( \mathrm { d } \mathbf { U } _ { n + 1 } \mid \mathbf { Z } _ { n + 1 } ) P _ { \theta } ( \mathrm { d } \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) .
$$

Then we have the decomposition

$$
\frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) = \int _ { \mathcal { Z } } \frac { \mathrm { d } P _ { \theta , \psi } ( \cdot \mid \mathbf { Z } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathrm { d } \mathbf { U } _ { n + 1 } ) Q _ { \phi } ( \mathrm { d } \mathbf { Z } _ { n } \mid \mathbf { U } _ { n } ) , \quad \mathrm { f o r ~ \mathsf { W } _ { \mathcal { U } } \mathrm { - a . e . } ~ } \mathbf { U } _ { n + 1 } .\tag{B.1}
$$

We choose a reference measure Λ on Z such that both $Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } )$ and $P _ { \theta } ( \cdot | \mathbf { Z } _ { n } )$ are absolutely continuous with respect to Λ, for instance, $\begin{array} { r } { \Lambda = Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) / 2 + P _ { \theta } ( \cdot | \mathbf { Z } _ { n } ) / 2 } \end{array}$ , and define the likelihood densities

$$
p _ { \theta } = \frac { \mathrm { d } P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ) } { \mathrm { d } \Lambda } , \quad q _ { \phi } = \frac { \mathrm { d } Q _ { \theta } ( \cdot \mid \mathbf { U } _ { n + 1 } ) } { \mathrm { d } \Lambda } .
$$

Then we factorize $P _ { \theta , \psi }$ and apply Jensen’s inequality to obtain

$$
\begin{array} { r l } { \log \frac { \mathrm { d } P _ { \theta , \psi } ( \cdot \mid \mathbf { Z } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) = \log \displaystyle \int _ { \mathcal { Z } } \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \ p _ { \theta } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) \Lambda ( \mathrm { d } \mathbf { Z } _ { n + 1 } ) } & { } \\ { = \log \displaystyle \int _ { \mathcal { Z } } \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \frac { p _ { \theta } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) } { q _ { \phi } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { U } _ { n + 1 } ) } q _ { \phi } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { U } _ { n + 1 } ) \Lambda ( \mathrm { d } \mathbf { Z } _ { n + 1 } ) } & { } \\ { \geq \displaystyle \int _ { \mathcal { Z } } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) - \log \frac { q _ { \phi } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { U } _ { n + 1 } ) } { p _ { \theta } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { Z } _ { n } ) } \right] q _ { \phi } ( \mathbf { Z } _ { n + 1 } \mid \mathbf { U } _ { n + 1 } ) \Lambda ( \mathrm { d } \mathbf { Z } _ { n + 1 } ) , } & { } \end{array}
$$

with the convention $q _ { \phi } ( { \bf Z } _ { n + 1 } \mid { \bf U } _ { n + 1 } ) \log q _ { \phi } ( { \bf Z } _ { n + 1 } \mid { \bf U } _ { n + 1 } ) = 0$ outside the support of $Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } )$ For $Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) \ll P _ { \theta } ( \cdot | \mathbf { Z } _ { n } )$ , the above result is equivalent to

$$
\begin{array} { r l } & { \log \frac { \mathrm { d } P _ { \theta , \psi } ( \cdot | \mathbf { Z } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \geq \displaystyle \int _ { \mathbb { Z } } [ \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) - \log \frac { \mathrm { d } Q _ { \phi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { P _ { \theta } ( \cdot | \mathbf { Z } _ { n } ) } ( \mathbf { Z } _ { n + 1 } ) ] Q _ { \phi } ( \mathrm { d } \mathbf { Z } _ { n + 1 } | \mathbf { \Delta } \mathbf { U } _ { n + 1 } ) } \\ & { \qquad = \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) } [ \log \frac { \mathrm { d } P _ { \psi } ( \cdot | \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) ] - D _ { \mathrm { K L } } ( Q _ { \phi } ( \cdot | \mathbf { U } _ { n + 1 } ) ) \| P _ { \theta } ( \cdot | \mathbf { Z } _ { n } ) ) . } \end{array}
$$

Again we use Jensen’s inequality to (B.1):

$$
\log \frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \ge \mathbb { E } _ { \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } ) } \left[ \log \frac { \mathrm { d } P _ { \theta , \psi } ( \cdot \mid \mathbf { Z } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] .
$$

Combining the two inequalities gives

$$
\begin{array} { r l } & { \log \frac { \mathrm { d } P _ { \theta , \psi , \phi } ( \cdot \mid \mathbf { U } _ { n } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \geq \mathbb { E } _ { \mathbf { Z } _ { n + 1 } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) } \left[ \log \frac { \mathrm { d } P _ { \psi } ( \cdot \mid \mathbf { Z } _ { n + 1 } ) } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } _ { n + 1 } ) \right] } \\ & { \qquad - \mathbb { E } _ { \mathbf { Z } _ { n } \sim Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n } ) } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ) \mid \mid P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ) \right) \right] , } \end{array}
$$

which proves the claim.

We next lift the dynamic KL term from the one-step objective to the population level and identify the transition kernel targeted by its minimization.

Proof of Proposition $\ 3 . 4 \cdot$ As in the proof of Proposition 3.2, we choose a common dominating measure on $\mathcal { Z }$ as reference and, with a slight abuse of notation, write $Q _ { \phi } ( { \bf Z } ^ { + } \mid { \bf U } ^ { + } ) , P ( { \bf Z } ^ { + } \mid { \bf Z } )$ , and $\Pi _ { \phi } ( \mathbf { Z } ^ { + } \mid \mathbf { Z } )$ for the corresponding densities with respect to the dominating measure. Expanding the KL divergence in $\mathcal { R } ( P )$ gives

$$
\begin{array} { r l } & { \mathcal { R } ( P ) = \mathbb { E } _ { ( \mathbf { U } , \mathbf { U } ^ { + } ) \sim \rho } \mathbb { E } _ { \mathbf { Z } \sim Q _ { \phi } ( \cdot \vert \mathbf { \Delta U } ) } \mathbb { E } _ { \mathbf { Z } ^ { + } \sim Q _ { \phi } ( \cdot \vert \mathbf { \Delta U } ^ { + } ) } \left[ \log \frac { \mathrm { d } Q _ { \phi } ( \cdot \vert \mathbf { \Delta U } ^ { + } ) } { \mathrm { d } P ( \cdot \vert \mathbf { \Delta Z } ) } ( \mathbf { Z } ^ { + } ) \right] } \\ & { \qquad = \mathbb { E } _ { ( \mathbf { Z } , \mathbf { Z } ^ { + } , \mathbf { U } ^ { + } ) \sim \Gamma _ { \phi } } \left[ \log \frac { \mathrm { d } Q _ { \phi } ( \cdot \vert \mathbf { \Delta U } ^ { + } ) } { \mathrm { d } \Pi _ { \phi } ( \cdot \vert \mathbf { Z } ) } ( \mathbf { Z } ^ { + } ) \right] + \mathbb { E } _ { ( \mathbf { Z } , \mathbf { Z } ^ { + } ) \sim \Gamma _ { \phi } } \left[ \log \frac { \mathrm { d } \Pi _ { \phi } ( \cdot \vert \mathbf { Z } ) } { \mathrm { d } P ( \cdot \vert \mathbf { Z } ) } ( \mathbf { Z } ^ { + } ) \right] . } \end{array}
$$

The first term depends only on the encoder and the data distribution, and is therefore independent of $P .$ For the second term, disintegrate $\Gamma _ { \phi }$ as

$$
\Gamma _ { \phi } (  { \mathrm { d } } \mathbf { Z } ,  { \mathrm { d } } \mathbf { Z } ^ { + } ,  { \mathrm { d } } \mathbf { U } ^ { + } ) = \Gamma _ { \phi } (  { \mathrm { d } } \mathbf { Z } ,  { \mathrm { d } } \mathbf { U } ^ { + } ) Q _ { \phi } (  { \mathrm { d } } \mathbf { Z } ^ { + } | \mathbf { U } ^ { + } )
$$

Then

$$
\begin{array} { r l } & { \mathbb { E } _ { ( \mathbf { Z } , \mathbf { Z } ^ { + } , \mathbf { U } ^ { + } ) \sim \Gamma _ { \phi } } \left[ \log \frac { \mathrm { d } Q _ { \phi } ( \cdot \mid \mathbf { U } ^ { + } ) } { \mathrm { d } \Pi _ { \phi } ( \cdot \mid \mathbf { Z } ) } ( \mathbf { Z } ^ { + } ) \right] } \\ & { \quad = \displaystyle \int _ { \mathcal { U } \times \mathcal { U } } \int _ { \mathcal { Z } } D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } ^ { + } ) \parallel \Pi _ { \phi } ( \cdot \mid \mathbf { Z } ) \right) Q _ { \phi } ( \mathrm { d } \mathbf { Z } \mid \mathbf { U } ) \rho ( \mathrm { d } \mathbf { U } , \mathrm { d } \mathbf { U } ^ { + } ) . } \end{array}\tag{B.2}
$$

For the second term, disintegrate $\Gamma _ { \phi }$ as

$$
\Gamma _ { \phi } ( \mathrm { d } \mathbf { Z } , \mathrm { d } \mathbf { Z } ^ { + } ) = \Pi _ { \phi } ( \mathrm { d } \mathbf { Z } ^ { + } \mid \mathbf { Z } ) \Gamma _ { \phi } ( \mathrm { d } \mathbf { Z } ) .
$$

Then

$$
\mathbb { E } _ { ( \mathbf { Z } , \mathbf { Z } ^ { + } ) \sim \Gamma _ { \phi } } \left[ \log \frac { \mathrm { d } \Pi _ { \phi } ( \cdot \mid \mathbf { Z } ) } { \mathrm { d } P ( \cdot \mid \mathbf { Z } ) } ( \mathbf { Z } ^ { + } ) \right] = \int _ { \mathcal { Z } } D _ { \mathrm { K L } } \left( \Pi _ { \phi } ( \cdot \mid \mathbf { Z } ) \parallel P ( \cdot \mid \mathbf { Z } ) \right) \Gamma _ { \phi } ( \mathrm { d } \mathbf { Z } ) ,\tag{B.3}
$$

Combining the identities (B.2) and (B.3) proves (3.4).

The first term in (3.4) is independent of P. The second term is nonnegative and equals zero if and only if $P ( \cdot | \mathbf { Z } ) = \Pi _ { \phi } ( \cdot | \mathbf { Z } )$ for $\Gamma _ { \phi } \mathrm { - a l m o s t }$ every Z. Therefore $P ^ { \star } = \Pi _ { \phi }$ is the population minimizer over all Markov kernels. □

We now specialize the abstract formulation to functional Gaussian kernels and their associated Cameron-Martin and spectral geometry.

## B.2 Gaussian Likelihood and Spectral Geometry

This subsection collects the arguments underlying the functional Gaussian specialization in §3.2. We first establish the Cameron-Martin representation of the decoder likelihood in the physical state space. We then turn to the latent Hilbert space, where a trace-class covariance admits a spectral representation that determines both the Gaussian perturbations and the associated Cameron-Martin geometry. Finally, we specialize this general construction to the Laplacian-resolvent covariance on the torus, which yields the Sobolev interpretation used in Corollary 3.9.

Proof of Proposition 3.5. Since $D _ { \psi } ( \mathbf { Z } ) \in \mathcal { H } _ { \mathcal { U } }$ , the Cameron-Martin theorem [Theorem $\mathrm { A . 3 } ]$ implies that the shifted Gaussian measure $\mathsf { W } _ { \mathcal { U } } ^ { D _ { \psi } ( \mathbf { Z } ) }$ is absolutely continuous with respect to $\mathsf { W } _ { \mathcal { U } }$ and satisfies

$$
\frac { \mathrm { d } \mathsf { W } _ { \mathcal { U } } ^ { D _ { \psi } ( \mathbf { Z } ) } } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } ( \mathbf { U } ) = \exp \left( \langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle ^ { \sim } - \frac { 1 } { 2 } \| D _ { \psi } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } \right) .
$$

This proves (3.5). Taking negative logarithms gives (3.6). If $\mathbf { U } \in \mathcal { H } _ { \mathcal { U } }$ , then the Paley-Wiener map agrees with the Cameron-Martin inner product:

$$
\langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle \tilde { \mathbf { \Psi } } = \langle D _ { \psi } ( \mathbf { Z } ) , \mathbf { U } \rangle _ { \mathcal { H } _ { M } } .
$$

Hence

$$
\begin{array} { r } { - \log \frac { \mathrm { d } P _ { \psi } ( \cdot  { | \textbf { Z } ) } } { \mathrm { d } \mathsf { W } _ { \mathcal { U } } } (  { \mathbf { U } } ) = \frac { 1 } { 2 } \| D _ { \psi } (  { \mathbf { Z } } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } - \langle D _ { \psi } (  { \mathbf { Z } } ) ,  { \mathbf { U } } \rangle _ { \mathcal { H } _ { \mathcal { U } } } } \\ { = \displaystyle \frac { 1 } { 2 } \|  { \mathbf { U } } - D _ { \psi } (  { \mathbf { Z } } ) \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } - \frac { 1 } { 2 } \|  { \mathbf { U } } \| _ { \mathcal { H } _ { \mathcal { U } } } ^ { 2 } , } \end{array}
$$

which proves (3.7).

We next turn from the decoder likelihood on the physical state space to the Gaussian geometry of the latent transition. We first present a standard auxiliary result that constructs a Gaussian random element from a positive trace-class covariance and characterizes its Cameron-Martin space in spectral coordinates (Da Prato and Zabczyk, 2014; Bogachev, 1998).

Lemma B.1 (Gaussian measure induced by a trace-class covariance). Let Z be a separable Hilbert space and let $K : \mathcal { Z } \to \mathcal { Z }$ be a positive, self-adjoint, trace-class operator. Let $\{ ( \lambda _ { i } , \mathbf { E } _ { i } ) \} _ { i \geq 1 }$ be an eigensystem of K, where $\{ { \bf E } _ { i } \} _ { i \ge 1 }$ is an orthonormal basis of $\mathcal { Z } , \lambda _ { i } > 0$ , and

$$
\operatorname { T r } ( K ) = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } < \infty .
$$

Let $\{ \xi _ { i } \} _ { i \geq 1 }$ be i.i.d. standard Gaussian random variables. Then the series

$$
\mathbf { G } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i }
$$

converges in $L ^ { 2 } ( \Omega ; \mathcal { Z } )$ and almost surely in $\mathcal { Z } .$ Moreover, G is a centered Z-valued Gaussian random element with covariance operator K, i.e.,

$$
\mathbf { G } \sim \mathsf { N } ( 0 , K ) .
$$

The Cameron-Martin space of $\mathsf { N } ( 0 , K )$ is

$$
\mathcal { H } _ { K } = \left\{ { \bf h } = \sum _ { i = 1 } ^ { \infty } h _ { i } { \bf E } _ { i } : \sum _ { i = 1 } ^ { \infty } \frac { | h _ { i } | ^ { 2 } } { \lambda _ { i } } < \infty \right\} ,
$$

equipped with the inner product

$$
\langle { \bf h } , { \bf g } \rangle _ { \mathcal { H } _ { K } } = \sum _ { i = 1 } ^ { \infty } \frac { h _ { i } g _ { i } } { \lambda _ { i } } ,
$$

and norm

$$
\| { \bf h } \| _ { \mathcal { H } _ { K } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \frac { | h _ { i } | ^ { 2 } } { \lambda _ { i } } .
$$

Equivalently,

$$
\mathcal { H } _ { K } = \mathrm { R a n g e } ( K ^ { 1 / 2 } ) , \qquad \| \mathbf { h } \| _ { \mathcal { H } _ { K } } = \| K ^ { - 1 / 2 } \mathbf { h } \| _ { \mathcal { Z } } ,
$$

where $K ^ { - 1 / 2 }$ is understood on ${ \mathrm { R a n g e } } ( K ^ { 1 / 2 } )$

Proof. Step I: Convergence. For each positive integer N, we define

$$
\mathbf { G } _ { N } = \sum _ { i = 1 } ^ { N } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i } .
$$

For $M > N$ , by orthonormality of $\{ { \bf E } _ { i } \} _ { i \ge 1 }$ and independence of $\{ \xi _ { i } \} _ { i \geq 1 }$

$$
\mathbb { E } \| \mathbf { G } _ { M } - \mathbf { G } _ { N } \| _ { \mathcal { Z } } ^ { 2 } = \mathbb { E } \left\| \sum _ { i = N + 1 } ^ { M } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i } \right\| _ { \mathcal { Z } } ^ { 2 } = \sum _ { i = N + 1 } ^ { M } \lambda _ { i } .
$$

Since $\Sigma _ { i = 1 } ^ { \infty } \lambda _ { i } < \infty$ , the sequence $\{ { \bf G } _ { N } \} _ { N \ge 1 }$ is Cauchy in $L ^ { 2 } ( \Omega ; \mathcal { Z } )$ . Therefore it converges in $L ^ { 2 } ( \Omega ; \mathcal { Z } )$ to some Z-valued random element G. In particular,

$$
\mathbb { E } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 2 } = \operatorname* { l i m } _ { N \to \infty } \mathbb { E } \| \mathbf { G } _ { N } \| _ { \mathcal { Z } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } = \operatorname { T r } ( K ) < \infty .
$$

Moreover, $\{ { \bf G } _ { N } \} _ { N \ge 1 }$ is an $L ^ { 2 } .$ -bounded martingale with respect to the filtration

$$
\mathcal { F } _ { N } = \sigma ( \xi _ { 1 } , \ldots , \xi _ { N } ) .
$$

Indeed, for $M > N$ 2

$$
\mathbb { E } [ { \bf G } _ { M } \mid \mathcal { F } _ { N } ] = { \bf G } _ { N } ,
$$

and

$$
\operatorname* { s u p } _ { N \geq 1 } \mathbb { E } \| \mathbf { G } _ { N } \| _ { \mathcal { Z } } ^ { 2 } = \operatorname* { s u p } _ { N \geq 1 } \sum _ { i = 1 } ^ { N } \lambda _ { i } \leq \sum _ { i = 1 } ^ { \infty } \lambda _ { i } < \infty .
$$

Therefore, by the Hilbert-space martingale convergence theorem, $\mathbf { G } _ { N }$ converges almost surely and in $L ^ { 2 } ( \Omega ; \mathcal { Z } )$ to a Z-valued random element. The $L ^ { 2 }$ limit is unique, so the almost sure limit is the same random element G.

Step II: Verify the Karhunen-Lo\`eve expansion. We next verify that the random element G has covariance operator K. For any $\mathbf { f } \in { \mathcal { Z } }$ ，

$$
\langle \mathbf { G } , \mathbf { f } \rangle _ { \mathcal { Z } } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \langle \mathbf { E } _ { i } , \mathbf { f } \rangle _ { \mathcal { Z } } ,
$$

where the convergence holds in $L ^ { 2 } ( \Omega )$ . Hence $\langle \mathbf { G } , \mathbf { f } \rangle _ { \mathcal { Z } }$ is a centered Gaussian random variable. Therefore G is a centered Gaussian random element. Moreover, for any $\mathbf { f } , \mathbf { g } \in { \mathcal { Z } }$ 2

$$
\begin{array} { r l } & { \mathbb { E } \left[ \langle \mathbf { G } , \mathbf { f } \rangle _ { \mathcal { Z } } \langle \mathbf { G } , \mathbf { g } \rangle _ { \mathcal { Z } } \right] = \displaystyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } \langle \mathbf { E } _ { i } , \mathbf { f } \rangle _ { \mathcal { Z } } \langle \mathbf { E } _ { i } , \mathbf { g } \rangle _ { \mathcal { Z } } } \\ & { \quad \quad \quad = \left. \displaystyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } \langle \mathbf { f } , \mathbf { E } _ { i } \rangle _ { \mathcal { Z } } \mathbf { E } _ { i } , \mathbf { g } \right. _ { \mathcal { Z } } = \langle K \mathbf { f } , \mathbf { g } \rangle _ { \mathcal { Z } } . } \end{array}
$$

Thus the covariance operator of G is K, and $\mathbf { G } \sim \mathsf { N } ( 0 , K )$

Step III: Identify the Cameron-Martin space. Since K is positive and self-adjoint,

$$
K ^ { 1 / 2 } \mathbf { f } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \langle \mathbf { f } , \mathbf { E } _ { i } \rangle _ { \mathcal { Z } } \mathbf { E } _ { i } .
$$

Therefore h $\in \mathrm { R a n g e } ( K ^ { 1 / 2 } )$ if and only if there exists $\textstyle \mathbf { f } = \sum _ { i = 1 } ^ { \infty } f _ { i } \mathbf { E } _ { i } \in \mathcal { Z }$ such that

$$
\mathbf { h } = K ^ { 1 / 2 } \mathbf { f } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } f _ { i } \mathbf { E } _ { i } .
$$

Writing $h _ { i } = \sqrt { \lambda _ { i } } f _ { i }$ , this is equivalent to

$$
\sum _ { i = 1 } ^ { \infty } { \frac { | h _ { i } | ^ { 2 } } { \lambda _ { i } } } = \sum _ { i = 1 } ^ { \infty } | f _ { i } | ^ { 2 } < \infty .
$$

Hence

$$
\mathcal { H } _ { K } = \mathrm { R a n g e } ( K ^ { 1 / 2 } ) = \left\{ { \bf h } = \sum _ { i = 1 } ^ { \infty } h _ { i } { \bf E } _ { i } : \sum _ { i = 1 } ^ { \infty } \frac { | h _ { i } | ^ { 2 } } { \lambda _ { i } } < \infty \right\} .
$$

The Cameron-Martin inner product is the pullback inner product through $K ^ { 1 / 2 }$ :

$$
\langle { \bf h } , { \bf g } \rangle _ { \mathcal { H } _ { K } } = \langle K ^ { - 1 / 2 } { \bf h } , K ^ { - 1 / 2 } { \bf g } \rangle _ { \mathcal { Z } } = \sum _ { i = 1 } ^ { \infty } \frac { h _ { i } g _ { i } } { \lambda _ { i } } .
$$

This proves the claim.

The preceding lemma provides the two ingredients needed for Proposition 3.7: it identifies the Cameron-Martin geometry induced by K and gives the Karhunen-Lo\`eve representation of a Gaussian perturbation with covariance K. Applying these facts to the scaled and shifted transition measure $\mathsf { N } ( T _ { \theta } ( \mathbf { Z } ) , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K )$ yields the spectral form of the latent transition.

Proof of Proposition 3.7. Part (a) follows directly from Lemma B.1. Indeed, since $K$ is positive, self-adjoint, and trace-class, the centered Gaussian measure $\mathsf { W } _ { \mathcal { Z } } = \mathsf { N } ( 0 , K )$ is well-defined on $\mathcal { Z }$ and its Cameron-Martin space is

$$
\mathcal { H } _ { K } = \mathrm { R a n g e } ( K ^ { 1 / 2 } ) = \left\{ { \bf R } = \sum _ { i = 1 } ^ { \infty } r _ { i } { \bf E } _ { i } : \sum _ { i = 1 } ^ { \infty } \frac { | r _ { i } | ^ { 2 } } { \lambda _ { i } } < \infty \right\} ,
$$

with norm

$$
\| { \bf R } \| _ { \mathcal { H } _ { K } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \frac { | r _ { i } | ^ { 2 } } { \lambda _ { i } } .
$$

We now prove part (b). Fix Z and write

$$
a = \alpha _ { \theta } ( { \bf Z } ) , \qquad m = T _ { \theta } ( { \bf Z } ) .
$$

Then

$$
P _ { \theta } ( \cdot \mid { \bf Z } ) = { \sf N } ( m , a ^ { 2 } K ) .
$$

The Cameron-Martin space of $\mathsf { N } ( 0 , a ^ { 2 } K )$ is the same set $\mathcal { H } _ { K }$ , but with scaled norm

$$
\| \mathbf { h } \| _ { \mathcal { H } _ { a ^ { 2 } K } } ^ { 2 } = \frac { 1 } { a ^ { 2 } } \| \mathbf { h } \| _ { \mathcal { H } _ { K } } ^ { 2 } .
$$

Since $m \in \mathcal { H } _ { K }$ , the Cameron-Martin theorem implies that $\mathsf { N } ( m , a ^ { 2 } K )$ is absolutely continuous with respect to $\mathsf { N } ( 0 , a ^ { 2 } K )$ , and for $\mathbf { Z } ^ { + } \in \mathcal { H } _ { K }$ ,

$$
\log \frac { \mathrm { d } \mathsf { N } ( m , a ^ { 2 } K ) } { \mathrm { d } \mathsf { N } ( 0 , a ^ { 2 } K ) } ( \mathbf { Z } ^ { + } ) = \frac { 1 } { a ^ { 2 } } \langle \mathbf { Z } ^ { + } , m \rangle _ { \mathcal { H } _ { K } } - \frac { 1 } { 2 a ^ { 2 } } \| m \| _ { \mathcal { H } _ { K } } ^ { 2 } .
$$

Therefore

$$
\begin{array} { r l r } { \mathrm { ~ } } & { } & { - \log \frac { \mathrm { d } \mathsf { N } ( m , a ^ { 2 } K ) } { \mathrm { d } \mathsf { N } ( 0 , a ^ { 2 } K ) } ( \mathbf { Z } ^ { + } ) = \frac { 1 } { 2 a ^ { 2 } } \| m \| _ { \mathcal { H } _ { K } } ^ { 2 } - \frac { 1 } { a ^ { 2 } } \langle \mathbf { Z } ^ { + } , m \rangle _ { \mathcal { H } _ { K } } } \\ & { } & { = \frac { 1 } { 2 a ^ { 2 } } \| \mathbf { Z } ^ { + } - m \| _ { \mathcal { H } _ { K } } ^ { 2 } - \frac { 1 } { 2 a ^ { 2 } } \| \mathbf { Z } ^ { + } \| _ { \mathcal { H } _ { K } } ^ { 2 } . } \end{array}
$$

Substituting back $a = \alpha _ { \theta } ( \mathbf { Z } )$ and $m = T _ { \theta } ( \mathbf { Z } )$ yields

$$
- \log \frac { \mathrm { d } P _ { \theta } ( \cdot \mid \mathbf { Z } ) } { \mathrm { d } \mathsf { N } ( 0 , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K ) } ( \mathbf { Z } ^ { + } ) = \frac { 1 } { 2 \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } } \| \mathbf { Z } ^ { + } - T _ { \theta } ( \mathbf { Z } ) \| _ { \mathcal { H } _ { K } } ^ { 2 } - \frac { 1 } { 2 \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } } \| \mathbf { Z } ^ { + } \| _ { \mathcal { H } _ { K } } ^ { 2 } .
$$

Finally, by Lemma B.1, the random series

$$
\sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i }
$$

defines a $\mathcal { Z } _ { - }$ valued Gaussian random element with law $\mathsf { N } ( 0 , K )$ . Hence

$$
\alpha _ { \theta } ( { \bf Z } ) \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } { \bf E } _ { i } \sim \mathsf { N } ( 0 , \alpha _ { \theta } ( { \bf Z } ) ^ { 2 } K ) .
$$

Adding the mean $T _ { \theta } ( \mathbf { Z } )$ gives

$$
\mathbf { Z } ^ { + } = T _ { \theta } ( \mathbf { Z } ) + \alpha _ { \theta } ( \mathbf { Z } ) \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i } \sim \mathsf { N } ( T _ { \theta } ( \mathbf { Z } ) , \alpha _ { \theta } ( \mathbf { Z } ) ^ { 2 } K ) ,
$$

which is exactly the transition kernel $P _ { \theta } ( \cdot \mid \mathbf { Z } )$ . The spectral-coordinate statement in the remark follows by taking inner products with $\mathbf { E } _ { i }$ □

Proposition 3.7 applies to an arbitrary positive, self-adjoint, trace-class covariance operator. For the spatially structured Gaussian perturbations considered in this work, a particularly useful choice is the Laplacian-resolvent family

$$
K _ { \ell , s } = ( I - \ell ^ { 2 } \Delta ) ^ { - s }
$$

on a periodic domain. The next lemma makes the corresponding Fourier structure explicit. In particular, it identifies the covariance spectrum, the condition under which $K _ { \ell , s }$ is trace-class on $L ^ { 2 }$ , and the resulting Cameron-Martin space. These properties will allow the abstract spectral geometry above to be interpreted as a Sobolev geometry.

Lemma B.2 (Laplacian-resolvent covariance on the torus). Let

$$
\mathcal { Z } = L ^ { 2 } ( \mathbb { T } ^ { d _ { \boldsymbol { x } } } ; \mathbb { R } ^ { d _ { \boldsymbol { z } } } ) ,
$$

and let

$$
K \ell , s : = ( I - \ell ^ { 2 } \Delta ) ^ { - s } , \qquad \ell > 0 , \quad s > 0 ,
$$

where the Laplacian acts componentwise. Let

$$
\mathcal { Z } _ { \mathbb { C } } = L ^ { 2 } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { C } ^ { d _ { z } } )
$$

be the complexification of $\mathcal { Z }$ . For $m \in \mathbb { Z } ^ { d _ { x } }$ , define

$$
\phi _ { m } ( x ) = e ^ { 2 \pi i m \cdot x } ,
$$

and, for the standard basis $\{ { \bf e } _ { a } \} _ { a = 1 } ^ { d _ { z } }$ of $\mathbb { R } ^ { d _ { z } }$ , set

$$
{ \bf E } _ { m , a } ( x ) = \phi _ { m } ( x ) { \bf e } _ { a } .
$$

Then the following hold.

(a) The family $\{ \mathbf { E } _ { m , j } \} _ { m \in \mathbb { Z } ^ { d _ { x } } , 1 \leq j \leq d _ { z } }$ is an orthonormal eigenbasis of $\mathcal { Z } _ { \mathbb { C } }$ for $K _ { \ell , s }$ , with

$$
K _ { \ell , s } \mathbf { E } _ { m , j } = \lambda _ { m } \mathbf { E } _ { m , j } , \qquad \lambda _ { m } = \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } .\tag{B.4}
$$

Equivalently, $K _ { \ell , s }$ is the Fourier multiplier with symbol $\lambda _ { m }$

(b) The operator $K _ { \ell , s }$ is self-adjoint, positive definite, and compact. Moreover, it is trace-class if and only if

$$
s > \frac { d _ { x } } { 2 } .
$$

In this case,

$$
\mathrm { T r } ( K _ { \ell , s } ) = d _ { z } \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } .\tag{B.5}
$$

(c) If $s > d _ { x } / 2$ , then ${ \sf N } ( 0 , K _ { \ell , s } )$ defines a Gaussian measure on ${ \mathcal { Z } } ,$ , and its Cameron-Martin space is

$$
\mathcal { H } _ { K _ { \ell , s } } = \mathrm { R a n g e } ( K _ { \ell , s } ^ { 1 / 2 } ) ,
$$

with norm

$$
\| \mathbf { h } \| _ { \mathcal { H } _ { K _ { \ell , s } } } ^ { 2 } = \sum _ { m \in \mathbb { Z } ^ { d _ { \boldsymbol { x } } } } \sum _ { j = 1 } ^ { d _ { z } } \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { s } | \widehat { h } _ { j } ( m ) | ^ { 2 } .\tag{B.6}
$$

Consequently,

$$
\mathcal { H } _ { K _ { \ell , s } } = H ^ { s } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { R } ^ { d _ { z } } )
$$

as sets, with equivalent norms.

Proof. We first recall the spectral decomposition of the periodic Laplacian. For each $m \in \mathbb { Z } ^ { d _ { x } }$

$$
\phi _ { m } ( x ) = e ^ { 2 \pi i m \cdot x }
$$

solves the eigenvalue problem

$$
- \Delta \phi _ { m } = 4 \pi ^ { 2 } | m | ^ { 2 } \phi _ { m } .
$$

Since $\{ \phi _ { m } \} _ { m \in \mathbb { Z } ^ { d _ { x } } }$ is an orthonormal basis of $L ^ { 2 } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { C } )$ , the vector-valued functions

$$
\begin{array} { r } { \mathbf { E } _ { m , j } ( x ) = \phi _ { m } ( x ) e _ { j } , \qquad m \in \mathbb { Z } ^ { d _ { x } } , \ j = 1 , \dots , d _ { z } , } \end{array}
$$

form an orthonormal basis of $L ^ { 2 } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { C } ^ { d _ { z } } )$ , and hence also give the usual complex Fourier representation of $L ^ { 2 } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { R } ^ { d _ { z } } )$ subject to the standard conjugate-symmetry constraint on real-valued fields. Because the Laplacian acts componentwise, we have

$$
\Delta \mathbf { E } _ { m , j } = ( \Delta \phi _ { m } ) \mathbf { e } _ { j } = - 4 \pi ^ { 2 } | m | ^ { 2 } \mathbf { E } _ { m , j } .
$$

Therefore,

$$
( I - \ell ^ { 2 } \Delta ) \mathbf { E } _ { m , j } = \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) \mathbf { E } _ { m , j } .
$$

Applying the functional calculus for the positive self-adjoint operator $I - \ell ^ { 2 } \Delta$ , we obtain

$$
K \mathbf { E } _ { m , j } = { ( I - \ell ^ { 2 } \Delta ) ^ { - s } } \mathbf { E } _ { m , j } = { \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } } \mathbf { E } _ { m , j } .
$$

Thus $\mathbf { E } _ { m , j }$ is an eigenfunction of K with eigenvalue

$$
\lambda _ { m } = \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } .
$$

It follows immediately that for any Fourier expansion

$$
Z ( x ) = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \widehat { Z } ( m ) \phi _ { m } ( x ) , \qquad \widehat { Z } ( m ) \in \mathbb { C } ^ { d _ { z } } ,
$$

the action of K is given by

$$
K Z = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \lambda _ { m } \widehat { Z } ( m ) \phi _ { m } ( x ) .
$$

Hence K is a Fourier multiplier with multiplier $\lambda _ { m }$ . Since all eigenvalues are positive and converge to zero, $K _ { \ell , s }$ is positive, self-adjoint, and compact. It is trace-class if and only if

$$
\sum _ { m \in \mathbb { Z } ^ { d _ { x } } } ( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } ) ^ { - s } < \infty .
$$

For large |m|,

$$
( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } ) ^ { - s } \asymp | m | ^ { - 2 s } ,
$$

and the corresponding lattice sum converges if and only if $2 s > d _ { x }$ . Accounting for the $d _ { z }$ latent channels gives (B.5).

Finally, when $s > d _ { x } / 2$ , the trace-class property ensures that ${ \sf N } ( 0 , K _ { \ell , s } )$ is a well-defined Gaussian measure on Z. By the spectral characterization of Cameron-Martin spaces [Lemma B.1],

$$
\mathcal { H } _ { K _ { \ell , s } } = \mathrm { R a n g e } ( K _ { \ell , s } ^ { 1 / 2 } ) ,
$$

and

$$
\| \mathbf { h } \| _ { \mathcal { H } _ { K _ { \ell , s } } } ^ { 2 } = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \sum _ { j = 1 } ^ { d _ { z } } \lambda _ { m } ^ { - 1 } | \widehat { h } _ { j } ( m ) | ^ { 2 } = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \sum _ { j = 1 } ^ { d _ { z } } \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { s } | \widehat { h } _ { j } ( m ) | ^ { 2 } ,
$$

which proves (B.6). Recall the usual Fourier definition of the Sobolev norm on the torus:

$$
\| \mathbf { h } \| _ { H ^ { s } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { R } ^ { d _ { z } } ) } ^ { 2 } = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \sum _ { j = 1 } ^ { d _ { z } } \left( 1 + 4 \pi ^ { 2 } | m | ^ { 2 } \right) ^ { s } | \widehat { h } _ { j } ( m ) | ^ { 2 } .
$$

Since $\ell > 0$ is fixed,

$$
( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } ) ^ { s } \asymp ( 1 + 4 \pi ^ { 2 } | m | ^ { 2 } ) ^ { s } .
$$

Hence this norm is equivalent to the standard $H ^ { s } ( \mathbb { T } ^ { d _ { x } } ; \mathbb { R } ^ { d _ { z } } )$ norm.

Corollary 3.9 now follows by applying this Cameron-Martin geometry to the transition residual ${ \bf Z } ^ { + } - T _ { \theta } ( { \bf Z } )$ appearing in Proposition 3.7.

Proof of Corollary 3.9. By Lemma B.2, the eigenvalues of $K = ( I - \ell ^ { 2 } \Delta ) ^ { - s }$ in the Fourier basis are

$$
\lambda _ { m } = \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { - s } .
$$

Proposition 3.7 therefore gives

$$
\begin{array} { l } { { \displaystyle { \left\| { \bf Z } ^ { + } - T _ { \theta } ( { \bf Z } ) \right\| _ { \mathcal { H } _ { K } } ^ { 2 } = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \sum _ { a = 1 } ^ { d _ { z } } \lambda _ { m } ^ { - 1 } \left| \widehat { Z } _ { a } ^ { + } ( m ) - \widehat { T _ { \theta } ( { \bf Z } ) } _ { a } ( m ) \right| ^ { 2 } } } } \\ { { \displaystyle ~ = \sum _ { m \in \mathbb { Z } ^ { d _ { x } } } \sum _ { a = 1 } ^ { d _ { z } } \left( 1 + 4 \pi ^ { 2 } \ell ^ { 2 } | m | ^ { 2 } \right) ^ { s } \left| \widehat { Z } _ { a } ^ { + } ( m ) - \widehat { T _ { \theta } ( { \bf Z } ) } _ { a } ( m ) \right| ^ { 2 } . } } \end{array}
$$

The equivalence with the standard $H ^ { s }$ norm follows from Lemma B.2.

## B.3 Gaussian Perturbations and Local Sensitivity

We next establish the auxiliary estimates used in the local-sensitivity analysis of §3.3.1. We first collect Gaussian moment bounds and then use them to control the efect of Gaussian perturbations under a second-order local expansion.

Lemma B.3 (Moment estimates). Let Z be a separable Hilbert space, and let $K : \mathcal { Z } \to \mathcal { Z }$ be positive, self-adjoint, and trace-class operator. Assume the random element $\mathbf { G } \sim \mathsf { N } ( 0 , K )$ . Then

(a) $\mathbb { E } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 4 } \leq 3 ( \operatorname { T r } K ) ^ { 2 }$

(b) $\mathbb { E } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 8 } \leq 1 0 5 ( \operatorname { T r } K ) ^ { 4 }$

Proof. Let $\{ ( \lambda _ { i } , \mathbf { E } _ { i } ) \} _ { i \geq 1 }$ be an eigensystem of K. By Karhunen-Lo\`eve expansion,

$$
\mathbf { G } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } \mathbf { E } _ { i } , \quad \xi _ { i } \overset { \mathrm { i . i . d . } } { \sim } \mathsf { N } ( 0 , 1 ) .
$$

Thus $\begin{array} { r } { \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } \xi _ { i } ^ { 2 } } \end{array}$ , and the cumulant generating function of $\| \mathbf G \| _ { \mathcal Z } ^ { 2 }$ is given by

$$
\begin{array} { r l } & { \log \mathbb { E } \left[ e ^ { t \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 2 } } \right] = \displaystyle \sum _ { i = 1 } ^ { \infty } \log \mathbb { E } \left[ e ^ { t \lambda _ { i } \xi _ { i } ^ { 2 } } \right] = - \frac { 1 } { 2 } \displaystyle \sum _ { i = 1 } ^ { \infty } \log ( 1 - 2 t \lambda _ { i } ) , } \\ & { \quad \quad \quad = - \frac { 1 } { 2 } \displaystyle \sum _ { i = 1 } ^ { \infty } \sum _ { n = 1 } ^ { \infty } \frac { ( 2 t \lambda _ { i } ) ^ { n } } { n } = \displaystyle \sum _ { n = 1 } ^ { \infty } \frac { 2 ^ { n - 1 } } { n } \left( \displaystyle \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { n } \right) t ^ { n } , \quad t < \frac { 1 } { 2 } . } \end{array}
$$

Matching the coeficients in

$$
\log \mathbb { E } \left[ e ^ { t \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 2 } } \right] = \sum _ { n = 1 } ^ { \infty } \frac { \kappa _ { n } } { n ! } t ^ { n } ,
$$

we obtain the cumulants of $\| \mathbf G \| _ { \mathcal { Z } } ^ { 2 }$

$$
\kappa _ { n } = 2 ^ { n - 1 } ( n - 1 ) ! \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { n } = 2 ^ { n - 1 } ( n - 1 ) ! \operatorname { T r } ( K ^ { n } ) , \quad n = 1 , 2 , \cdots .\tag{B.7}
$$

For fixed $n \in \mathbb { N }$ , let $\Pi ( n )$ denote the set of partitions of $\{ 1 , \cdots , n \}$ . We use the moment-cumulant formula for $\| \mathbf G \| _ { \mathcal { Z } } ^ { 2 }$

$$
\mathbb { E } \Vert \mathbf { G } \Vert _ { \mathcal { Z } } ^ { 2 n } = \sum _ { \pi \in \Pi ( n ) } \prod _ { V \in \pi } \kappa _ { | V | } .
$$

Then

$$
\begin{array} { r } { \mathbb { E } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 4 } = \kappa _ { 2 } + \kappa _ { 1 } ^ { 2 } = 2 \operatorname { T r } ( K ^ { 2 } ) + ( \operatorname { T r } K ) ^ { 2 } . } \end{array}
$$

and

$$
\begin{array} { r l } & { \mathbb { E } \left\| \mathbf { G } \right\| _ { \mathcal { Z } } ^ { 8 } = \kappa _ { 4 } + 4 \kappa _ { 1 } \kappa _ { 3 } + 3 \kappa _ { 2 } ^ { 2 } + 6 \kappa _ { 1 } ^ { 2 } \kappa _ { 2 } + \kappa _ { 1 } ^ { 4 } } \\ & { \qquad = 4 8 \mathrm { T r } ( K ^ { 4 } ) + 3 2 \mathrm { T r } ( K ) \mathrm { T r } ( K ^ { 3 } ) + 1 2 \mathrm { T r } ( K ^ { 2 } ) ^ { 2 } + 1 2 ( \mathrm { T r } K ) ^ { 2 } \mathrm { T r } ( K ^ { 2 } ) + \mathrm { T r } ( K ) ^ { 4 } . } \end{array}
$$

Finally, since K is positive trace-class, we have

$$
\operatorname { T r } ( K ^ { n } ) = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { n } \leq \left( \sum _ { i = 1 } ^ { \infty } \lambda _ { i } \right) ^ { n } = ( \operatorname { T r } K ) ^ { n } .
$$

Combining the last three displays gives the desired estimates.

These moment estimates control the higher-order remainder terms generated by Gaussian perturbations. The following lemma combines them with a local Fr´echet expansion to obtain a general perturbation estimate.

Lemma B.4 (Local sensitivity under Gaussian perturbations). Let $\mathcal { Z }$ and $\mathcal { U }$ be separable Hilbert spaces, and let $K : \mathcal { Z } \to \mathcal { Z }$ be positive, self-adjoint, and trace-class operator. Let $\mathbf { G } \sim \mathsf { N } ( 0 , K )$ . Fix $\mathbf { z } \in { \mathcal { Z } }$ and $a > 0$ . Suppose that $F : { \mathcal { Z } } \to { \mathcal { U } }$ is Fr´echet diferentiable at $\mathbf { z } ,$ with Fr´echet derivative

$$
J _ { F } ( \mathbf { z } ) \in \mathfrak { B } ( \mathcal { Z } , \mathcal { U } ) ,
$$

and assume that, on the relevant neighborhood of $\mathbf { z } ,$ the second-order remainder satisfies

$$
\| F ( \mathbf { z } + \mathbf { h } ) - F ( \mathbf { z } ) - J _ { F } ( \mathbf { z } ) \mathbf { h } \| _ { \mathcal { U } } \leq \frac { M } { 2 } \| \mathbf { h } \| _ { \mathcal { Z } } ^ { 2 } .
$$

Then

$$
\mathbb { E } \left[ \| F ( \mathbf { z } + a \mathbf { G } ) - F ( \mathbf { z } ) \| _ { \mathcal { U } } ^ { 2 } \right] \leq a ^ { 2 } \| J \| _ { \mathrm { o p } } ^ { 2 } \operatorname { T r } K + \frac { 3 } { 4 } M ^ { 2 } a ^ { 4 } ( \operatorname { T r } K ) ^ { 2 } .
$$

Proof. For brevity, write $J = J _ { F } ( { \bf z } )$ . By the assumed second-order expansion, for every h in the relevant neighborhood,

$$
F ( \mathbf { z } + \mathbf { h } ) - F ( \mathbf { z } ) = J \mathbf { h } + R ( \mathbf { h } ) ,
$$

where

$$
\| R ( \mathbf { h } ) \| _ { \mathcal { U } } \leq \frac { M } { 2 } \| \mathbf { h } \| _ { \mathcal { Z } } ^ { 2 } .
$$

Taking $\mathbf { h } = a \mathbf { G }$ , we have

$$
F ( \mathbf { z } + a \mathbf { G } ) - F ( \mathbf { z } ) = a J \mathbf { G } + R ( a \mathbf { G } ) .
$$

Therefore,

$$
\begin{array} { r } { \mathbb { E } \left[ \| F ( \mathbf { z } + a \mathbf { G } ) - F ( \mathbf { z } ) \| _ { \boldsymbol { u } } ^ { 2 } \right] = \mathbb { E } \left[ \| a J \mathbf { G } + R ( a \mathbf { G } ) \| _ { \boldsymbol { u } } ^ { 2 } \right] \leq 2 a ^ { 2 } \mathbb { E } \| J \mathbf { G } \| _ { \boldsymbol { u } } ^ { 2 } + 2 \mathbb { E } \| R ( a \mathbf { G } ) \| _ { \boldsymbol { u } } ^ { 2 } . } \end{array}
$$

We first compute the linear term. Since $\mathbf { G } \sim \mathsf { N } ( 0 , K )$ and $J$ is bounded linear, $J \mathbf G$ is a Gaussian random element in $\mathcal { U }$ with covariance operator $J K J ^ { * }$ . Hence

$$
\begin{array} { r } { \mathbb { E } \| J \mathbf { G } \| _ { \mathcal { U } } ^ { 2 } = \| J \| _ { \mathrm { o p } } ^ { 2 } \mathbb { E } \| \mathbf { G } \| _ { \mathcal { U } } ^ { 2 } \leq \| J \| _ { \mathrm { o p } } ^ { 2 } \mathrm { T r } ( K ) . } \end{array}
$$

We next bound the remainder term. From the second-order remainder assumption,

$$
\| R ( a \mathbf { G } ) \| _ { \mathcal { U } } \leq \frac { M } { 2 } a ^ { 2 } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 2 } .
$$

Therefore

$$
\mathbb { E } \| R ( a \mathbf { G } ) \| _ { \mathcal { U } } ^ { 2 } \leq \frac { M ^ { 2 } } { 4 } a ^ { 4 } \mathbb { E } \| \mathbf { G } \| _ { \mathcal { Z } } ^ { 4 } .
$$

Combining the remainder estimate and the moment estimate in Lemma B.3 yields

$$
\begin{array} { r } { \mathbb { E } \left[ \| F ( \mathbf { z } + a \mathbf { G } ) - F ( \mathbf { z } ) \| _ { \mathcal { U } } ^ { 2 } \right] \leq a ^ { 2 } \| J \| _ { \mathrm { o p } } ^ { 2 } \operatorname { T r } ( K ) + \frac { 3 } { 4 } M ^ { 2 } a ^ { 4 } ( \operatorname { T r } K ) ^ { 2 } . } \end{array}
$$

which is the desired result.

We now apply this perturbation estimate to the latent transition and decoder to prove Proposition 3.10.

Proof of Proposition 3.10. We first prove the Lipschitz robustness estimates. By the Lipschitz continuity of $T _ { \theta }$ on the relevant regions,

$$
\| T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } + \Xi _ { n } ) - T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) \| _ { \mathcal { Z } } \leq \Lambda _ { T } \| \Xi _ { n } \| _ { \mathcal { Z } } .
$$

Conditionally on $\mathbf { U } _ { n } ^ { \star }$ , one samples $\Xi _ { n } \sim \mathsf { N } \big ( 0 , K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) \big )$ . Then we have

$$
\mathbb { E } [ \| \Xi _ { n } \| _ { \mathcal { Z } } ^ { 2 } \big | \mathbf { U } _ { n } ^ { \star } \big ] = \operatorname { T r } K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) .
$$

Taking expectation over the reference trajectory therefore gives

$$
\eta _ { n } ^ { 2 } \le \Lambda _ { T } ^ { 2 } \mathbb { E } \left[ \operatorname { T r } K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) \right] \leq \Lambda _ { T } ^ { 2 } \varsigma .
$$

Similarly,

$$
\| D _ { \psi } ( \widetilde { \mathbf { Z } } _ { n } ^ { + } ) - D _ { \psi } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \| _ { \mathcal { U } } \leq \Lambda _ { D } \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \| \mathbf { G } _ { n } \| _ { \mathcal { Z } } .
$$

Using $\alpha _ { \theta } \leq \bar { \alpha }$ and $\mathbb { E } \| \mathbf { G } _ { n } \| _ { \mathcal { Z } } ^ { 2 } = \mathrm { T r } ( K )$ yields

$$
\rho _ { n } \leq \Lambda _ { D } \bar { \alpha } \sqrt { \mathrm { T r } ( K ) } .
$$

This proves (3.17).

Now we derive the local estimate for $\rho _ { n }$ . By Lemma B.4,

$$
\begin{array} { r l } {  { \mathbb { E } _ { { \mathbf { G } } _ { n } \sim \mathsf { N } ( 0 , K ) } \| D _ { \psi } \big ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) + \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) { \mathbf { G } } _ { n } \big ) - D _ { \psi } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \| _ { \mathcal { U } } ^ { 2 } } } \\ & { \le \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ^ { 2 } \big \| J _ { D _ { \psi } } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \big \| _ { \mathrm { o p } } ^ { 2 } \mathrm { T r } K + \frac { 3 } { 4 } M _ { D } ^ { 2 } \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ^ { 4 } ( \mathrm { T r } K ) ^ { 2 } . } \end{array}
$$

Taking expectation over the reference trajectory and Gaussian perturbation $\Xi _ { n }$ , and using the fact $\alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \leq \overline { { \alpha } }$ , we have

$$
\rho _ { n } \leq \overline { { \alpha } } \left( \mathbb { E } \left[ \| J _ { D _ { \psi } } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \| _ { \mathrm { o p } } ^ { 2 } \right] \right) ^ { 1 / 2 } \sqrt { \mathrm { T r } K } + \frac { \sqrt { 3 } M _ { D } \overline { { \alpha } } ^ { 2 } } { 2 } \mathrm { T r } K ,
$$

which is the desired bound. Similarly, applying Lemma B.4 conditionally on $\mathbf { U } _ { n } ^ { \star }$ , with $a = 1$ , and ${ \cal K } = { \cal K } _ { \phi } ( { \bf U } _ { n } ^ { \star } )$ , and using Tr $K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) \leq \varsigma$ pointwise before taking any expectation,

$$
\mathbb { E } \left[ \left. T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) - T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) \right. _ { \mathcal { Z } } ^ { 2 } \middle | \mathbf { U } _ { n } ^ { \star } \right] \leq \Vert J _ { T _ { \theta } } ( \mathbf { Z } _ { n } ^ { \star } ) \Vert _ { \mathrm { o p } } ^ { 2 } \varsigma + \frac { 3 } { 4 } M _ { T } ^ { 2 } \varsigma ^ { 2 } .
$$

Taking expectation over the reference trajectory and applying ${ \sqrt { x + y } } \leq { \sqrt { x } } + { \sqrt { y } }$ for any $x , y \geq 0$ gives

$$
\eta _ { n } \leq \left( \mathbb { E } \left[ \| J _ { T _ { \theta } } ( \mathbf { Z } _ { n } ^ { \star } ) \| _ { \mathrm { o p } } ^ { 2 } \right] \varsigma \right) ^ { 1 / 2 } + \frac { \sqrt { 3 } M _ { T } } { 2 } \varsigma ,
$$

which is (3.18).

We next turn from the local efect of Gaussian perturbations to the control provided by variational transition alignment over autoregressive rollout.

## B.4 Variational Alignment and Deterministic Rollout

We now prove the KL-aware rollout results of §3.3.2. The key step is to convert the dynamic KL divergence into control of latent mean discrepancies. We begin with a Gaussian transportation inequality and its specialization to the Hilbert-space setting used for the latent dynamics.

We first recall a transportation-cost inequality for Gaussian measures. The quadratic transportation inequality for Gaussian measures originates from the classical work of Talagrand (1996) and was later connected to logarithmic Sobolev inequalities and related functional inequalities (Otto and Villani, 2000). For the infinite-dimensional setting considered here, we use the Banach-space formulation of Riedel (2017, Corollary 1.3).

Theorem B.5 (Riedel’s Gaussian transportation inequality). Let $( B , { \mathcal { H } } , \gamma )$ be a Gaussian Banach space, where B is a separable Banach space, γ is a centered Gaussian measure on B, and H is its Cameron–Martin space. Then, for every probability measure $\nu \in \mathcal { P } ( B )$

$$
\operatorname* { i n f } _ { \pi \in \Pi ( \nu , \gamma ) } \int _ { B \times B } \| x - y \| _ { B } ^ { 2 } \mathrm { d } \pi ( x , y ) \leq 2 \sigma _ { \gamma } ^ { 2 } D _ { \mathrm { K L } } ( \nu \| \gamma ) ,
$$

where

$$
\sigma _ { \gamma } ^ { 2 } : = \operatorname* { s u p } _ { \ell \in B ^ { * } , \ \| \ell \| _ { B ^ { * } } \leq 1 } \int _ { B } \ell ( x ) ^ { 2 } { \mathrm { d } } \gamma ( x ) < \infty .
$$

Equivalently,

$$
W _ { 2 } ^ { 2 } ( \nu , \gamma ) \leq 2 \sigma _ { \gamma } ^ { 2 } D _ { \mathrm { K L } } ( \nu \parallel \gamma ) ,
$$

where $W _ { 2 }$ is computed with respect to the Banach norm $\| \cdot \| _ { B }$

For Gaussian measures on the latent Hilbert space, the weak variance in this inequality reduces to the operator norm of the covariance.

Lemma B.6 (Hilbert-space specialization). Let $\mathcal { Z }$ be a separable Hilbert space and let

$$
\mu = \Nu ( m , C ) ,
$$

where $m \in { \mathcal { Z } }$ and $C : \mathcal { Z } \to \mathcal { Z }$ is positive, self-adjoint, and trace-class. Then, for every $\nu \in \mathcal { P } _ { 2 } ( \mathcal { Z } )$ ,

$$
W _ { 2 } ^ { 2 } ( \nu , \mu ) \leq 2 \| C \| _ { \mathrm { o p } } D _ { \mathrm { K L } } ( \nu \| \mu ) .
$$

Here $W _ { 2 }$ is computed with respect to the Hilbert norm $\| \cdot \| _ { \mathcal { Z } }$

Combining this transportation bound with the fact that Wasserstein distance controls diferences of means gives the KL mean-control result used in the main text.

Proof. It sufices to consider the centered case $m = 0$ , since translation preserves both Wasserstein distance and relative entropy. Let

$$
\gamma = \mathsf { N } ( 0 , C ) .
$$

By Theorem B.5, it remains to identify the weak variance $\sigma _ { \gamma } ^ { 2 } .$

Since Z is a Hilbert space, every continuous linear functional $\ell \in \mathcal { Z } ^ { * }$ has the form

$$
\ell _ { h } ( x ) = \langle h , x \rangle _ { \mathcal { Z } }
$$

for a unique $h \in { \mathcal { Z } } .$ , with

$$
\| \ell _ { h } \| _ { \mathcal { Z } ^ { * } } = \| h \| _ { \mathcal { Z } } .
$$

If $X \sim \mathsf { N } ( 0 , C )$ , then

$$
\int _ { \mathcal { Z } } \ell _ { h } ( x ) ^ { 2 } \mathrm { d } \gamma ( x ) = \mathbb { E } \langle h , X \rangle _ { \mathcal { Z } } ^ { 2 } = \langle C h , h \rangle _ { \mathcal { Z } } .
$$

Therefore

$$
\sigma _ { \gamma } ^ { 2 } = \operatorname* { s u p } _ { \ell \in \mathcal { Z } ^ { * } , \| \ell \| _ { \mathcal { Z } ^ { * } } \leq 1 } \int _ { \mathcal { Z } } \ell ( x ) ^ { 2 } \mathrm { d } \gamma ( x ) \operatorname* { s u p } _ { \| h \| _ { \mathcal { Z } } \leq 1 } \langle C h , h \rangle _ { \mathcal { Z } } .
$$

Since C is positive and self-adjoint,

$$
\operatorname* { s u p } _ { \| h \| _ { \mathcal { Z } } \leq 1 } \langle C h , h \rangle _ { \mathcal { Z } } = \| C \| _ { \mathrm { o p } } .
$$

Thus

$$
\sigma _ { \gamma } ^ { 2 } = \| { \cal C } \| _ { \mathrm { o p } } .
$$

Substituting this into Theorem B.5 gives

$$
W _ { 2 } ^ { 2 } ( \nu , \mathsf { N } ( 0 , C ) ) \leq 2 \| C \| _ { \mathrm { o p } } D _ { \mathrm { K L } } ( \nu \| \mathsf { N } ( 0 , C ) ) .
$$

The translated case $\mu = \Nu ( m , C )$ follows by applying the same result to the translated measures.

Proof of Lemma 3.11. Let

$$
P _ { { \bf z } } : = P _ { \theta } ( { \bf \cdot } | { \bf z } ) = { \sf N } ( T _ { \theta } ( { \bf z } ) , \alpha _ { \theta } ( { \bf z } ) ^ { 2 } { \cal K } ) .
$$

By Lemma B.6,

$$
W _ { 2 } ^ { 2 } ( Q , P _ { \mathbf { z } } ) \leq 2 \| \alpha _ { \theta } ( \mathbf { z } ) ^ { 2 } K \| _ { \mathrm { o p } } D _ { \mathrm { K L } } ( Q \parallel P _ { \mathbf { z } } ) .
$$

Since

$$
\| { \boldsymbol { \alpha } } _ { \theta } ( \mathbf { z } ) ^ { 2 } K \| _ { \mathrm { o p } } = { \boldsymbol { \alpha } } _ { \theta } ( \mathbf { z } ) ^ { 2 } \| K \| _ { \mathrm { o p } } ,
$$

we have

$$
W _ { 2 } ( Q , P _ { \mathbf { z } } ) \leq \alpha _ { \theta } ( \mathbf { z } ) { \sqrt { 2 \| K \| _ { \mathrm { o p } } D _ { \mathrm { K L } } ( Q \| P _ { \mathbf { z } } ) } } .
$$

It remains to observe that Wasserstein distance controls the distance between means. Indeed, for any coupling $( { \bf Y } , { \bf Y ^ { \prime } } )$ of Q and $P _ { \mathbf { z } }$

$$
\begin{array} { r } { \| m _ { Q } - T _ { \theta } ( \mathbf { z } ) \| _ { \mathcal { Z } } = \left\| \mathbb { E } \mathbf { Y } - \mathbb { E } \mathbf { Y } ^ { \prime } \right\| _ { \mathcal { Z } } \leq \mathbb { E } \| \mathbf { Y } - \mathbf { Y } ^ { \prime } \| _ { \mathcal { Z } } \leq \left( \mathbb { E } \| \mathbf { Y } - \mathbf { Y } ^ { \prime } \| _ { \mathcal { Z } } ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}
$$

Taking the infimum over all couplings gives

$$
\| m _ { Q } - T _ { \theta } ( \mathbf { z } ) \| _ { \mathcal { Z } } \leq W _ { 2 } ( Q , P _ { \mathbf { z } } ) .
$$

Combining the two estimates proves the claim.

We now apply Lemma 3.11 along the encoded reference trajectory and propagate the resulting one-step latent control through the deterministic mean rollout. Let

$$
\{ \mathbf { U } _ { n } ^ { \star } \} _ { n = 0 } ^ { N } \subset \mathcal { U }
$$

be a reference physical trajectory and recall our notations

$$
\begin{array} { r } { \mathbf { Z } _ { n } ^ { \star } : = m _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) , \qquad \bar { \mathbf { U } } _ { n } : = D _ { \psi } ( \mathbf { Z } _ { n } ^ { \star } ) , \qquad \delta _ { n } : = \| \bar { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } . } \end{array}
$$

For each $n = 0 , \ldots , N - 1$ , let

$$
\Xi _ { n } \sim \mathsf { N } ( 0 , K _ { \phi } ( \mathbf { U } _ { n } ^ { \star } ) ) .
$$

The dynamic KL quantity is

$$
\kappa _ { n } ^ { 2 } : = \mathbb { E } _ { \Xi _ { n } } \left[ D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ^ { \star } ) \parallel P _ { \theta } ( \cdot \mid \mathbf { Z } _ { n } ^ { \star } + \Xi _ { n } ) \right) \right] ,
$$

and the latent encoder-noise sensitivity of the transition map is

$$
\eta _ { n } : = \left( \mathbb { E } _ { \Xi _ { n } } \left[ \| T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } + \Xi _ { n } ) - T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) \| _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } .
$$

Proof of Proposition 3.12. Step I. We first bound the latent one-step defect

$$
\| T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { Z } _ { n + 1 } ^ { \star } \| _ { \mathcal { Z } } .
$$

For each realization of $\Xi _ { n } .$ , apply Lemma 3.11 with

$$
Q = Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ^ { \star } ) , \qquad \mathbf { Z } = \mathbf { Z } _ { n } ^ { \star } + \Xi _ { n } .
$$

Since the mean of $Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ^ { \star } )$ is

$$
m _ { \phi } ( { \bf U } _ { n + 1 } ^ { \star } ) = { \bf Z } _ { n + 1 } ^ { \star } ,
$$

we obtain

$$
\begin{array} { r } { \big \| \mathbf { Z } _ { n + 1 } ^ { \star } - T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \big \| _ { \mathcal { Z } } \leq \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \sqrt { 2 \| K \| _ { \mathrm { o p } } D _ { \mathrm { K L } } \left( Q _ { \phi } ( \cdot \mid \mathbf { U } _ { n + 1 } ^ { \star } ) \| P _ { \theta } ( \cdot \mid \widetilde { \mathbf { Z } } _ { n } ) \right) } . } \end{array}\tag{B.8}
$$

Using $\alpha _ { \theta } \leq \bar { \alpha }$ , squaring and taking expectation over both the reference trajectory and $\Xi _ { n }$ gives

$$
\left( \mathbb { E } \left[ \left. \mathbf { Z } _ { n + 1 } ^ { \star } - T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \right. _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } \leq \overline { { \alpha } } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n } .\tag{B.9}
$$

Now write

$$
T _ { \boldsymbol { \theta } } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { Z } _ { n + 1 } ^ { \star } = \big [ T _ { \boldsymbol { \theta } } ( \mathbf { Z } _ { n } ^ { \star } ) - T _ { \boldsymbol { \theta } } ( \widetilde { \mathbf { Z } } _ { n } ) \big ] + \big [ T _ { \boldsymbol { \theta } } ( \widetilde { \mathbf { Z } } _ { n } ) - \mathbf { Z } _ { n + 1 } ^ { \star } \big ] .
$$

Taking the $L ^ { 2 }$ norm over the joint distribution and applying Minkowski’s inequality together with (B.9) yields

$$
\left( \mathbb { E } \left[ \left. T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { Z } _ { n + 1 } ^ { \star } \right. _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } \leq \eta _ { n } + \overline { { \alpha } } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n } ,
$$

which proves (3.20).

Step II. We next derive the latent rollout bound. Define

$$
\begin{array} { r } { e _ { n } : = \left( \mathbb { E } \left[ \| \widehat { \mathbf { Z } } _ { n } - \mathbf { Z } _ { n } ^ { \star } \| _ { \mathcal { Z } } ^ { 2 } \right] \right) ^ { 1 / 2 } . } \end{array}
$$

Note that $e _ { 0 } = 0$ since the rollout is initialized at the encoder mean. For every realization of the reference trajectory,

$$
\begin{array} { r l } & { \| \widehat { { \mathbf Z } } _ { n + 1 } - { \mathbf Z } _ { n + 1 } ^ { \star } \| { \boldsymbol z } = \| T _ { \theta } ( \widehat { { \mathbf Z } } _ { n } ) - { \mathbf Z } _ { n + 1 } ^ { \star } \| { \boldsymbol z } } \\ & { \qquad \leq \| T _ { \theta } ( \widehat { { \mathbf Z } } _ { n } ) - T _ { \theta } ( { \mathbf Z } _ { n } ^ { \star } ) \| { \boldsymbol z } + \| T _ { \theta } ( { \mathbf Z } _ { n } ^ { \star } ) - { \mathbf Z } _ { n + 1 } ^ { \star } \| { \boldsymbol z } . } \end{array}
$$

Taking the population $L ^ { 2 }$ norm and using Minkowski’s inequality, Lipschitz continuity of $T _ { \theta }$ , and (3.20), we obtain

$$
e _ { n + 1 } \leq \Lambda _ { T } e _ { n } + \eta _ { n } + \overline { { \alpha } } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { n } .
$$

Iterating this recursion from $e _ { 0 } = 0$ gives

$$
e _ { n } \leq \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { T } ^ { n - 1 - j } \left( \eta _ { j } + \overline { { \alpha } } \sqrt { 2 \| K \| _ { \mathrm { o p } } } \kappa _ { j } \right) .\tag{B.10}
$$

For the physical rollout, use $\widehat { \mathbf { U } } _ { n } = D _ { \psi } ( \widehat { \mathbf { Z } } _ { n } )$ and insert the decoded reference state $D _ { \psi } ( \mathbf { Z } _ { n } ^ { \star } )$ :

$$
\| \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } \leq \| D _ { \psi } ( \widehat { \mathbf { Z } } _ { n } ) - D _ { \psi } ( \mathbf { Z } _ { n } ^ { \star } ) \| _ { \mathcal { U } } + \| D _ { \psi } ( \mathbf { Z } _ { n } ^ { \star } ) - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } .
$$

Taking the population $L ^ { 2 }$ norm and using Lipschitz continuity of $D _ { \psi }$ yields

$$
\left( \mathbb { E } \Vert \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \Vert _ { \mathcal { U } } ^ { 2 } \right) ^ { 1 / 2 } \leq \Lambda _ { D } e _ { n } + \delta _ { n } .
$$

Substituting (B.10) proves (3.21).

Step III. It remains to incorporate the predictive loss. Define the clean decoded one-step error

$$
q _ { n } : = \left( \mathbb { E } \left[ \left. D _ { \psi } ( T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) ) - \mathbf { U } _ { n + 1 } ^ { \star } \right. _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } .
$$

For every realization of the reference trajectory and the Gaussian perturbations,

$$
\begin{array} { r l } & { \left\| D _ { \psi } ( T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) ) - \mathbf { U } _ { n + 1 } ^ { \star } \right\| _ { \mathcal { U } } } \\ & { \quad \leq \left\| D _ { \psi } ( T _ { \theta } ( \mathbf { Z } _ { n } ^ { \star } ) ) - D _ { \psi } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) \right\| _ { \mathcal { U } } + \left\| D _ { \psi } ( T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) ) - D _ { \psi } ( \widetilde { \mathbf { Z } } _ { n } ^ { + } ) \right\| _ { \mathcal { U } } + \left\| D _ { \psi } ( \widetilde { \mathbf { Z } } _ { n } ^ { + } ) - \mathbf { U } _ { n + 1 } ^ { \star } \right\| _ { \mathcal { U } } . } \end{array}
$$

Taking the joint $L ^ { 2 }$ norm and applying Minkowski’s inequality gives

$$
q _ { n } \leq \Lambda _ { D } \eta _ { n } + \rho _ { n } + \epsilon _ { n } .\tag{B.11}
$$

We now obtain a second bound for the physical rollout error at time n. Since

$$
\widehat { \bf U } _ { n } = D _ { \psi } ( T _ { \theta } ( \widehat { \bf Z } _ { n - 1 } ) ) ,
$$

by Minkowski’s inequality and Lipschitz continuity of $D _ { \psi }$ and $T _ { \theta } .$ , we have

$$
\begin{array} { r l } & { \left( \mathbb { E } \| \widehat { \mathbf { U } } _ { n } - { \mathbf { U } } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right) ^ { 1 / 2 } \leq \left( \mathbb { E } \| D _ { \psi } ( T _ { \theta } ( \widehat { \mathbf { Z } } _ { n } ) ) - D _ { \psi } ( T _ { \theta } ( { \mathbf { Z } } _ { n } ^ { \star } ) ) \| _ { \mathcal { U } } ^ { 2 } \right) ^ { 1 / 2 } + \left( \mathbb { E } \left[ \| D _ { \psi } ( T _ { \theta } ( { \mathbf { Z } } _ { n } ^ { \star } ) ) - { \mathbf { U } } _ { n + 1 } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } } \\ & { \qquad \leq \Lambda _ { D } \Lambda _ { T } c _ { n - 1 } + q _ { n - 1 } . } \end{array}
$$

Using (B.10) at time n − 1 and (B.11),

$$
\left( \mathbb { E } \| \widehat { \mathbf { U } } _ { n } - { \mathbf { U } } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right) ^ { 1 / 2 } \leq \Lambda _ { D } \sum _ { j = 0 } ^ { n - 2 } \Lambda _ { T } ^ { n - 1 - j } \left( \eta _ { j } + \bar { \alpha } \sqrt { 2 \| K \| _ { \infty } } \kappa _ { j } \right) + \Lambda _ { D } \eta _ { n - 1 } + \epsilon _ { n - 1 } + \rho _ { n - 1 } .\tag{B.12}
$$

Both (B.12) and (3.21) are valid upper bounds for the same population rollout error. Taking the smaller of their final terms yields (3.22). □

For comparison, we conclude with the generic autoregressive amplification bound (3.23) used to separate the efect of recursive deployment from mechanisms specific to variational latent dynamics.

Proof of Proposition $\it 3 . 1 4$ . Define the population rollout error

$$
\begin{array} { r } { e _ { n } : = \left( \mathbb { E } \left[ \| \widehat { \mathbf { U } } _ { n } - \mathbf { U } _ { n } ^ { \star } \| _ { \mathcal { U } } ^ { 2 } \right] \right) ^ { 1 / 2 } . } \end{array}
$$

Since $\widehat { \mathbf { U } } _ { 0 } = \mathbf { U } _ { 0 } ^ { \star }$ , we have $e _ { 0 } = 0$ . For every $n \geq 0$

$$
\begin{array} { r l } & { \| \widehat { \mathbf { U } } _ { n + 1 } - \mathbf { U } _ { n + 1 } ^ { \star } \| { \boldsymbol { u } } = \| F _ { \vartheta } ( \widehat { \mathbf { U } } _ { n } ) - \mathbf { U } _ { n + 1 } ^ { \star } \| { \boldsymbol { u } } } \\ & { \qquad \leq \| F _ { \vartheta } ( \widehat { \mathbf { U } } _ { n } ) - F _ { \vartheta } ( \mathbf { U } _ { n } ^ { \star } ) \| { \boldsymbol { u } } + \| F _ { \vartheta } ( \mathbf { U } _ { n } ^ { \star } ) - \mathbf { U } _ { n + 1 } ^ { \star } \| { \boldsymbol { u } } . } \end{array}
$$

Taking the population $L ^ { 2 }$ norm and applying Minkowski’s inequality together with the Lipschitz continuity of $F _ { \vartheta }$ gives

$$
e _ { n + 1 } \leq \Lambda _ { F } e _ { n } + \epsilon _ { n } ^ { \mathrm { d i r } } .
$$

Iterating this recursion from $e _ { 0 } = 0$ yields

$$
e _ { n } \leq \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { F } ^ { n - 1 - j } \epsilon _ { j } ^ { \mathrm { d i r } } ,
$$

which proves (3.23).

The preceding results concern the deterministic mean-map rollout used at inference. The next subsection gives a complementary high-probability analysis of stochastic transition rollouts around this mean trajectory.

## B.5 High-Probability Stochastic Envelope

The main analysis in §3.3 concerns the deterministic mean-map rollout used at inference. Here we give a complementary high-probability characterization of stochastic rollouts obtained by sampling from the learned transition kernel. We first establish the required Gaussian concentration bounds and then propagate them through the latent dynamics. The following result is similar to the conclusion of Hsu et al. (2012).

Lemma B.7 (Two-sided Gaussian norm concentration in a Hilbert space). Let $\mathcal { Z }$ be a separable Hilbert space and let

$$
\mathbf { Z } \sim \mathsf { N } ( 0 , C ) ,
$$

where $C : \mathcal { Z } \to \mathcal { Z }$ is positive, self-adjoint, and trace-class. Then, for every $t \geq 0$

$$
\begin{array} { r } { \mathbb { P } \Big ( \| \mathbf Z \| _ { \mathcal { Z } } ^ { 2 } \geq \operatorname { T r } ( C ) + 2 \sqrt { \operatorname { T r } ( C ^ { 2 } ) t } + 2 \| C \| _ { \mathrm { o p } } t \Big ) \leq e ^ { - t } , } \end{array}\tag{B.13}
$$

and

$$
\begin{array} { r } { \mathbb { P } \Big ( \| \mathbf Z \| _ { \mathcal { Z } } ^ { 2 } \leq \operatorname { T r } ( C ) - 2 \sqrt { \operatorname { T r } ( C ^ { 2 } ) t } \Big ) \leq e ^ { - t } . } \end{array}\tag{B.14}
$$

Consequently, for $q \in ( 0 , 1 / 2 )$ , with probability at least $1 - 2 q$

$$
r _ { C } ^ { - } ( q ) \leq \lVert \mathbf { Z } \rVert _ { \mathcal { Z } } \leq r _ { C } ^ { + } ( q ) ,
$$

where

$$
r _ { C } ^ { - } ( q ) : = \left[ \mathrm { T r } ( C ) - 2 \sqrt { \mathrm { T r } ( C ^ { 2 } ) \log ( 1 / q ) } \right] _ { + } ^ { 1 / 2 } ,\tag{B.15}
$$

$$
r _ { C } ^ { + } ( q ) : = \left[ \mathrm { T r } ( C ) + 2 \sqrt { \mathrm { T r } ( C ^ { 2 } ) \log ( 1 / q ) } + 2 \| C \| _ { \mathrm { o p } } \log ( 1 / q ) \right] ^ { 1 / 2 } .\tag{B.16}
$$

Moreover, let $P : \mathcal { Z } \to \mathcal { Z }$ be any orthogonal projection satisfying rank $( I - P ) \leq 1$ . Define

$$
r _ { C } ^ { \perp } ( q ) : = \Big [ \mathrm { T r } ( C ) - \| C \| _ { \mathrm { o p } } - 2 \sqrt { \mathrm { T r } ( C ^ { 2 } ) \log ( 1 / q ) } \Big ] _ { + } ^ { 1 / 2 } .\tag{B.17}
$$

Then

$$
\mathbb { P } \Big ( \| P \mathbf { Z } \| _ { \mathcal { Z } } < r _ { C } ^ { \perp } ( q ) \Big ) \leq q .\tag{B.18}
$$

Proof. Since $C$ is positive, self-adjoint, and trace-class, there exists an orthonormal basis $\{ E _ { i } \} _ { i \ge 1 }$ of $\mathcal { Z }$ and eigenvalues $\lambda _ { i } \geq 0$ such that

$$
C E _ { i } = \lambda _ { i } E _ { i } , \qquad \sum _ { i = 1 } ^ { \infty } \lambda _ { i } = \mathrm { T r } ( C ) < \infty .
$$

By the Karhunen-Lo\`eve expansion,

$$
{ \bf Z } = \sum _ { i = 1 } ^ { \infty } \sqrt { \lambda _ { i } } \xi _ { i } E _ { i } , \xi _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } { \mathsf { N } } ( 0 , 1 ) ,
$$

and hence

$$
\| \mathbf { Z } \| _ { \mathcal { Z } } ^ { 2 } = \sum _ { i = 1 } ^ { \infty } \lambda _ { i } \xi _ { i } ^ { 2 } \qquad \mathrm { a . s . }
$$

Set

$$
S : = \| \mathbf { Z } \| _ { \mathcal { Z } } ^ { 2 } , \qquad \mu : = \operatorname { T r } ( C ) , \qquad v : = \operatorname { T r } ( C ^ { 2 } ) , \qquad L : = \| C \| _ { \mathrm { o p } } .
$$

If $v = 0$ , then $C = 0$ and the claims are immediate, so assume $v > 0$

(i) For the upper tail, for every $0 \leq s < 1 / ( 2 L )$ ,

$$
\begin{array} { r l r } {  { \log \mathbb { E } e ^ { s ( S - \mu ) } = \sum _ { i = 1 } ^ { \infty } [ - s \lambda _ { i } - \frac { 1 } { 2 } \log ( 1 - 2 s \lambda _ { i } ) ] } } \\ & { } & { \leq \sum _ { i = 1 } ^ { \infty } \frac { s ^ { 2 } \lambda _ { i } ^ { 2 } } { 1 - 2 s \lambda _ { i } } \leq \frac { s ^ { 2 } v } { 1 - 2 s L } , ~ } \end{array}
$$

where we used

$$
- { \frac { 1 } { 2 } } \log ( 1 - 2 x ) - x \leq { \frac { x ^ { 2 } } { 1 - 2 x } } , \qquad 0 \leq x < { \frac { 1 } { 2 } } .
$$

Chernof’s inequality therefore gives

$$
\mathbb { P } ( S - \mu \geq a ) \leq \exp \left( - s a + \frac { s ^ { 2 } v } { 1 - 2 s L } \right) .
$$

Taking

$$
a = 2 \sqrt { v t } + 2 L t , \qquad s = \frac { \sqrt { t } } { \sqrt { v } + 2 L \sqrt { t } } ,
$$

yields

$$
\mathbb { P } \left( S - \mu \geq 2 { \sqrt { v t } } + 2 L t \right) \leq e ^ { - t } ,
$$

which proves (B.13).

(ii) For the lower tail, for every $s \geq 0 .$

$$
\log \mathbb { E } e ^ { - s ( S - \mu ) } = \sum _ { i = 1 } ^ { \infty } \left[ s \lambda _ { i } - \frac { 1 } { 2 } \log ( 1 + 2 s \lambda _ { i } ) \right] \leq s ^ { 2 } \sum _ { i = 1 } ^ { \infty } \lambda _ { i } ^ { 2 } = s ^ { 2 } v ,
$$

where we used

$$
\log ( 1 + 2 x ) \geq 2 x - 2 x ^ { 2 } , \qquad x \geq 0 .
$$

Thus, for $a > 0$

$$
\begin{array} { r } { \mathbb { P } ( S - \mu \le - a ) \le \exp ( - s a + s ^ { 2 } v ) . } \end{array}
$$

Choosing

$$
a = 2 \sqrt { v t } , \qquad s = \sqrt { \frac { t } { v } } ,
$$

gives

$$
\mathbb { P } \left( S - \mu \leq - 2 \sqrt { v t } \right) \leq e ^ { - t } ,
$$

which proves (B.14). Taking $t = \log ( 1 / q )$ and applying a union bound gives the two-sided interval.

(iii) It remains to prove (B.18). Since P is bounded and linear,

$$
P \mathbf { Z } \sim \mathsf { N } ( 0 , P C P ) .
$$

Because rank $( I - P ) \leq 1$ ，

$$
\mathrm { T r } ( P C P ) \geq \mathrm { T r } ( C ) - \| C \| _ { \mathrm { o p } } ,
$$

while

$$
\mathrm { T r } \big ( ( P C P ) ^ { 2 } \big ) = \| P C P \| _ { \mathrm { H S } } ^ { 2 } \leq \| C \| _ { \mathrm { H S } } ^ { 2 } = \mathrm { T r } ( C ^ { 2 } ) .
$$

Applying (B.14) to PZ with $t = \log ( 1 / q )$ therefore yields

$$
\mathbb { P } \Big ( \| P \mathbf { Z } \| _ { \mathcal { Z } } < r _ { C } ^ { \perp } ( q ) \Big ) \leq q ,
$$

which concludes the proof.

The preceding lemma controls both the magnitude of each Gaussian transition perturbation and the component that remains after projecting out one potentially canceling direction. We now use these estimates to obtain a two-sided latent envelope around the deterministic mean-map rollout.

Proposition B.8 (Two-sided stochastic envelope around the mean-map rollout). Let U be the physical state space and let $\mathcal { Z }$ be a separable Hilbert space. Let $K : \mathcal { Z } \to \mathcal { Z }$ be positive, selfadjoint, and trace-class. Given an initial physical state $\mathbf { U } _ { 0 } \in \mathcal { U } .$ , define the mean-map latent rollout by

$$
\begin{array} { r } { \widehat { \mathbf { Z } } _ { 0 } = m _ { \phi } ( \mathbf { U } _ { 0 } ) , \qquad \widehat { \mathbf { Z } } _ { n + 1 } = T _ { \theta } ( \widehat { \mathbf { Z } } _ { n } ) , \qquad n = 0 , \dots , N - 1 , } \end{array}\tag{B.19}
$$

and the corresponding physical rollout by

$$
\widehat { \mathbf { U } } _ { 0 } = \mathbf { U } _ { 0 } , \qquad \widehat { \mathbf { U } } _ { n } = D _ { \psi } ( \widehat { \mathbf { Z } } _ { n } ) , \qquad n = 1 , \ldots , N .
$$

Starting from the same physical initial condition, consider the stochastic latent rollout

$$
\begin{array} { r } { \widetilde { \mathbf { Z } } _ { 0 } = m _ { \phi } ( \mathbf { U } _ { 0 } ) , \qquad \widetilde { \mathbf { Z } } _ { n + 1 } = T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) + \boldsymbol { \alpha } _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \mathbf { G } _ { n } , \qquad \mathbf { G } _ { n } \overset { \mathrm { i . i . d . } } { \sim } \mathsf { N } ( 0 , K ) , } \end{array}\tag{B.20}
$$

with physical states

$$
\widetilde { \bf U } _ { 0 } = { \bf U } _ { 0 } , \qquad \widetilde { \bf U } _ { n } = D _ { \psi } ( \widetilde { \bf Z } _ { n } ) , \qquad n = 1 , \ldots , N .
$$

Assume that $T _ { \theta }$ is Λ<sub>T</sub>-Lipschitz and $D _ { \psi }$ is $\Lambda _ { D } .$ -Lipschitz on the relevant latent regions, and that

$$
0 < \underline { { \alpha } } \le \alpha _ { \theta } ( \mathbf { Z } ) \le \bar { \alpha }
$$

throughout the stochastic rollout tube. Then the following hold.

(a) One-step conditional spread. For every $n = 0 , \ldots , N - 1$ and $q \in ( 0 , 1 / 2 )$ , conditionally on $\widetilde { \mathbf { Z } } _ { n }$ , with probability at least $1 - 2 q$

$$
\underline { { \alpha } } r _ { K } ^ { - } ( q ) \leq \left\| \widetilde { \mathbf Z } _ { n + 1 } - T _ { \theta } ( \widetilde { \mathbf Z } _ { n } ) \right\| _ { \mathcal { Z } } \leq \bar { \alpha } r _ { K } ^ { + } ( q ) .\tag{B.21}
$$

(b) Two-sided latent envelope and physical upper envelope. For any $\gamma \in ( 0 , 1 )$ , with probability at least $1 - \gamma _ { ; }$ , simultaneously for all $n = 1 , \ldots , N$ , the latent ensemble deviation satisfies

$$
\underline { { \alpha } } r _ { K } ^ { \perp } \Big ( \frac { \gamma } { 2 N } \Big ) \leq \| \widetilde { \mathbf Z } _ { n } - \widehat { \mathbf Z } _ { n } \| _ { \mathcal { Z } } \leq \bar { \alpha } r _ { K } ^ { + } \Big ( \frac { \gamma } { 2 N } \Big ) \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { T } ^ { n - 1 - j } .\tag{B.22}
$$

Consequently,

$$
\left. \widetilde { \mathbf { U } } _ { n } - \widehat { \mathbf { U } } _ { n } \right. _ { \mathcal { U } } \leq \Lambda _ { D } \bar { \alpha } r _ { K } ^ { + } \Big ( \frac { \gamma } { 2 N } \Big ) \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { T } ^ { n - 1 - j } .\tag{B.23}
$$

The latent lower envelope is nontrivial whenever

$$
\mathrm { T r } ( K ) - \| K \| _ { \mathrm { o p } } > 2 \sqrt { \mathrm { T r } ( K ^ { 2 } ) \log \biggl ( \frac { 2 N } { \gamma } \biggr ) } .\tag{B.24}
$$

Proof. For part $\mathrm { ( a ) }$ , condition on $\widetilde { \mathbf { Z } } _ { n }$ . Since ${ \bf G } _ { n }$ is independent of the preceding transition noises,

$$
\begin{array} { r } { \widetilde { \mathbf { Z } } _ { n + 1 } - T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) = \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \mathbf { G } _ { n } . } \end{array}
$$

The amplitude $\alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } )$ is fixed under this conditioning. Applying Lemma B.7 to $\mathbf { G } _ { n } \sim \mathsf { N } ( 0 , K )$ gives

$$
\begin{array} { r } { \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) r _ { K } ^ { - } ( q ) \leq \left\| \widetilde { \mathbf { Z } } _ { n + 1 } - T _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \right\| _ { \mathcal { Z } } \leq \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) r _ { K } ^ { + } ( q ) . } \end{array}
$$

Since $\underline { { { \alpha } } } \le \alpha _ { \theta } ( \mathbf { Z } ) \le \bar { \alpha }$ , we obtain (B.21).

For part (b), define the latent ensemble deviation

$$
\mathbf { E } _ { n } : = { \widetilde { \mathbf { Z } } } _ { n } - { \widehat { \mathbf { Z } } } _ { n } .
$$

Let

$$
\mathscr { F } _ { n } : = \sigma ( \mathbf { G } _ { 0 } , \ldots , \mathbf { G } _ { n - 1 } ) , \qquad \mathscr { F } _ { 0 } : = \{ \emptyset , \Omega \} .
$$

Then $\widetilde { \mathbf { Z } } _ { n }$ and $\mathbf { E } _ { n }$ are ${ \mathcal { F } } _ { n }$ -measurable, while ${ \bf G } _ { n }$ is independent of ${ \mathcal { F } } _ { n }$ . Since the two rollouts are initialized from the same encoder mean,

$$
\mathbf { E } _ { 0 } = 0 .
$$

Subtracting (B.19) from (B.20) gives

$$
\begin{array} { r } { { \bf E } _ { n + 1 } = \underbrace { T _ { \theta } ( \widetilde { \bf Z } _ { n } ) - T _ { \theta } ( \widehat { \bf Z } _ { n } ) } _ { { \bf D } _ { n } } + \alpha _ { \theta } ( \widetilde { \bf Z } _ { n } ) { \bf G } _ { n } , } \end{array}\tag{B.25}
$$

where ${ \bf D } _ { n }$ is ${ \mathcal { F } } _ { n }$ -measurable.

For the upper envelope, Lipschitz continuity of $T _ { \theta }$ and $\alpha _ { \theta } \leq \bar { \alpha }$ give

$$
\| \mathbf { E } _ { n + 1 } \| _ { \mathcal { Z } } \leq \Lambda _ { T } \| \mathbf { E } _ { n } \| _ { \mathcal { Z } } + \bar { \alpha } \| \mathbf { G } _ { n } \| _ { \mathcal { Z } } .
$$

Iterating from ${ \bf E } _ { 0 } = 0$ yields

$$
\| \mathbf { E } _ { n } \| _ { \mathcal { Z } } \leq \bar { \alpha } \sum _ { j = 0 } ^ { n - 1 } \Lambda _ { T } ^ { n - 1 - j } \| \mathbf { G } _ { j } \| _ { \mathcal { Z } } .\tag{B.26}
$$

By Lemma B.7 and a union bound, with probability at least $1 - \gamma / 2$

$$
\| { \bf G } _ { j } \| _ { \mathcal { Z } } \le r _ { K } ^ { + } \biggl ( \frac { \gamma } { 2 N } \biggr ) , \qquad j = 0 , \ldots , N - 1 .
$$

Substitution into (B.26) proves the upper bound in (B.22) simultaneously for all $n \leq N$

For the lower envelope, fix $n \in \{ 0 , \ldots , N - 1 \}$ and let $P _ { n }$ be the orthogonal projection onto $\{ \mathbf { D } _ { n } \} ^ { \perp }$ , with $P _ { n } = I$ when ${ \bf D } _ { n } = 0$ . Then $P _ { n }$ is ${ \mathcal { F } } _ { n }$ -measurable, rank $( I - P _ { n } ) \leq 1$ , and $P _ { n } { \bf D } _ { n } = 0$ Applying $P _ { n }$ to (B.25) gives

$$
P _ { n } \mathbf { E } _ { n + 1 } = \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) P _ { n } \mathbf { G } _ { n } .
$$

Hence

$$
\begin{array} { r } { \| \mathbf { E } _ { n + 1 } \| _ { \mathcal { Z } } \geq \| P _ { n } \mathbf { E } _ { n + 1 } \| _ { \mathcal { Z } } = \alpha _ { \theta } ( \widetilde { \mathbf { Z } } _ { n } ) \| P _ { n } \mathbf { G } _ { n } \| _ { \mathcal { Z } } \geq \underline { { \alpha } } \| P _ { n } \mathbf { G } _ { n } \| _ { \mathcal { Z } } . } \end{array}
$$

Conditionally on $\mathcal { F } _ { n }$ , the projection $P _ { n }$ is fixed and $\mathbf { G } _ { n } \sim \mathsf { N } ( 0 , K )$ remains independent of ${ \mathcal { F } } _ { n }$ Therefore,

$$
\mathbb { P } \bigg ( \| P _ { n } \mathbf { G } _ { n } \| \boldsymbol { z } < r _ { K } ^ { \perp } \Big ( \frac { \gamma } { 2 N } \Big ) \Big | \mathcal { F } _ { n } \bigg ) \leq \frac { \gamma } { 2 N }
$$

by (B.18). Taking expectations and applying a union bound over $n = 0 , \ldots , N - 1$ shows that, with probability at least $1 - \gamma / 2$

$$
\| \mathbf { E } _ { n } \| _ { \mathcal { Z } } \ge \underline { { \alpha } } r _ { K } ^ { \perp } \biggl ( \frac { \gamma } { 2 N } \biggr ) , \qquad n = 1 , \ldots , N .
$$

Intersecting the upper- and lower-envelope events gives probability at least $1 - \gamma$ and proves (B.22).

Finally, for every $n \geq 1$ , Lipschitz continuity of the decoder gives

$$
\begin{array} { r l } & { \left\| \widetilde { { \mathbf U } } _ { n } - \widehat { { \mathbf U } } _ { n } \right\| _ { \mathcal { U } } = \left\| D _ { \psi } ( \widetilde { { \mathbf Z } } _ { n } ) - D _ { \psi } ( \widehat { { \mathbf Z } } _ { n } ) \right\| _ { \mathcal { U } } } \\ & { \qquad \leq \Lambda _ { D } \| { \mathbf E } _ { n } \| _ { \mathcal { Z } } , } \end{array}
$$

and (B.23) follows from the latent upper envelope. The condition (B.24) is precisely the condition under which $r _ { K } ^ { \perp } ( \gamma / ( 2 N ) ) > 0$ □

Relation to the deterministic rollout analysis. The stochastic-envelope result complements the deterministic rollout analysis in §3.3. Proposition 3.12 characterizes the error of the mean-map rollout relative to the reference PDE trajectory, with the latent transition mismatch controlled by the variational KL term and the encoder-noise sensitivity. In contrast, Proposition B.8 conditions on the initial physical state and characterizes the spread induced by sampling from the learned Gaussian transition around this mean-map trajectory. Thus, the two results respectively describe the location and the stochastic spread of the learned latent dynamics.

Notably, both bounds are governed by the same transition-stability factor $\Lambda _ { T }$ . In the deterministic analysis, powers of $\Lambda _ { T }$ amplify local transition discrepancies relative to the reference trajectory; in the stochastic analysis, they amplify the Gaussian perturbations around the mean-map rollout. The upper envelope therefore shows that stochastic spread remains controlled whenever the mean transition is suficiently stable, while the lower latent envelope shows that, when the transition covariance has suficient variance outside any single direction, sampled rollouts do not collapse onto the mean-map trajectory. These results characterize the spread generated by the learned transition distribution, but do not imply that this spread is calibrated to the true prediction error.

## C Benchmark Details

This appendix provides the detailed specifications of the three fluid-dynamics benchmarks used in §5, including their governing equations, data generation, numerical solvers, and evaluation metrics. Across all benchmarks, the main extrapolation occurs along the temporal direction, with test rollouts extending substantially beyond the training horizon.

## C.1 1D compressible Euler equation

We consider the 1D compressible Euler equations, a canonical inviscid model for compressible fluid dynamics. The system describes the evolution of density, momentum, and total energy through conservation laws. In conservative form, the equation is

$$
\partial _ { t } \mathbf { U } + \partial _ { x } \mathbf { F } ( \mathbf { U } ) = 0 , \qquad x \in \mathbb { T } _ { L } ,\tag{C.1}
$$

where the conserved variables and flux are given by

$$
\mathbf { U } = \left( \rho \atop \rho u \right) , \qquad \mathbf { F } ( \mathbf { U } ) = \left( \begin{array} { c c } { \rho u } \\ { \rho u ^ { 2 } + p } \\ { u ( E + p ) } \end{array} \right) ,\tag{C.2}
$$

where $\rho ( t , x )$ denotes the fluid density, $u ( t , x )$ denotes the velocity, $p ( t , x )$ denotes the pressure, and $E ( t , x )$ denotes the total energy density. We close the system with the ideal gas equation of state

$$
p = \left( \gamma - 1 \right) \left( E - \frac { 1 } { 2 } \rho u ^ { 2 } \right) ,
$$

where $\gamma = 1$ .4 is the constant adiabatic index. Equivalently, the system can be written as

$$
\left\{ \begin{array} { l l } { \partial _ { t } \rho + \partial _ { x } ( \rho u ) } & { = 0 , } \\ { \partial _ { t } ( \rho u ) + \partial _ { x } ( \rho u ^ { 2 } + p ) } & { = 0 , } \\ { \partial _ { t } E + \partial _ { x } \big ( u ( E + p ) \big ) } & { = 0 . } \end{array} \right.
$$

The Euler system contains no explicit viscous or thermal difusion. Therefore, wave-like structures are not smoothed by physical dissipation, and the dynamics can develop sharp gradients, contact discontinuities, rarefaction waves, and shocks.

In our experiments, we use smooth periodic initial conditions on $\mathbb { T } _ { L } = \mathbb { R } / L \mathbb { Z }$ and represent the state by the primitive variables

$$
\mathbf { V } ( t , x ) = \bigl ( \rho ( t , x ) , u ( t , x ) , p ( t , x ) \bigr ) , \quad t \in [ 0 , T ] , \ x \in  { \mathbb { T } } _ { L } .\tag{C.3}
$$

The learning task is to approximate the time-h solution map

$$
S _ { h } : \left( \rho ( t ) , u ( t ) , p ( t ) \right) \mapsto \left( \rho ( t + h ) , u ( t + h ) , p ( t + h ) \right) ,\tag{C.4}
$$

which is then rolled out autoregressively. This benchmark provides a compact test of deterministic compressible wave dynamics. Compared with dissipative 1D equations such as Burgers’ equation, the absence of viscosity makes long-horizon prediction more sensitive to phase errors and accumulated wave-interaction errors. At the same time, the one-dimensional setting avoids the full cost of multidimensional compressible flow, making it useful as a controlled inviscid-fluid benchmark.

Evaluation metrics. For the Euler benchmark, we report rollout-aggregated relative $L ^ { 1 }$ and $L ^ { 2 }$ errors separately for density, velocity, and pressure, together with their channel average. Let $u _ { i } ^ { n }$ and $\widehat { u } _ { i } ^ { n }$ denote the reference and predicted fields for trajectory i at rollout step n. The relative errors are computed as

$$
\mathrm { R e l . } L ^ { 1 } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \sum _ { n = 1 } ^ { T } \Vert \widehat { u } _ { i } ^ { n } - u _ { i } ^ { n } \Vert _ { L ^ { 1 } } } { \sum _ { n = 1 } ^ { T } \Vert u _ { i } ^ { n } \Vert _ { L ^ { 1 } } } , \quad \mathrm { R e l . } L ^ { 2 } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \left( \sum _ { n = 1 } ^ { T } \Vert \widehat { u } _ { i } ^ { n } - u _ { i } ^ { n } \Vert _ { L ^ { 2 } } ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { n = 1 } ^ { T } \Vert u _ { i } ^ { n } \Vert _ { L ^ { 2 } } ^ { 2 } \right) ^ { 1 / 2 } } .
$$

The $L ^ { 2 }$ metric emphasizes larger-amplitude pointwise errors, while $L ^ { 1 }$ is less dominated by a small number of large local discrepancies and is therefore useful when sharp fronts or discontinuous structures are present. We do not report $H ^ { 1 }$ error for Euler because the inviscid dynamics may develop shocks and contact discontinuities, for which derivative-based errors become highly sensitive to grid resolution, numerical shock thickness, and small spatial shifts of the discontinuity.

## C.1.1 Data Generation for the 1D Euler Equations

Initial conditions. For each trajectory, the density, pressure, and velocity fields are initialized from independent periodic Gaussian random fields on $\mathbb { T } _ { L }$ . At first, independent periodic random fields $g _ { \rho } , g _ { p } , g _ { u }$ are generated spectrally with power

$$
S _ { k } = \exp \left( - 2 \pi ^ { 2 } \ell _ { \mathrm { G R F } } ^ { 2 } \frac { k ^ { 2 } } { L ^ { 2 } } \right) , \qquad \ell _ { \mathrm { G R F } } = 1 ,
$$

with the zero mode removed. Each realization is subsequently centered and normalized to unit root-mean-square amplitude:

$$
g  \frac { g - \frac { 1 } { n _ { x } } \sum _ { j = 1 } ^ { n _ { x } } g ( x _ { j } ) } { [ \frac { 1 } { n _ { x } } \sum _ { j = 1 } ^ { n _ { x } } \Big ( g ( x _ { j } ) - \frac { 1 } { n _ { x } } \sum _ { m = 1 } ^ { n _ { x } } g ( x _ { m } ) \Big ) ^ { 2 } ] ^ { 1 / 2 } + 1 0 ^ { - 8 } } .
$$

Then the primitive conditions are constructed as

$$
\rho ( 0 , x ) = \rho _ { 0 } \left( 1 + a _ { \rho } g _ { \rho } ( x ) \right) , \quad p ( 0 , x ) = p _ { 0 } \left( 1 + a _ { p } g _ { p } ( x ) \right) , \quad u ( 0 , x ) = M c _ { 0 } g _ { u } ( x ) ,
$$

where

$$
\rho _ { 0 } = p _ { 0 } = 1 , \qquad a _ { \rho } = 0 . 1 5 , \qquad a _ { p } = 0 . 1 0 , \qquad c _ { 0 } = \sqrt { \frac { \gamma p _ { 0 } } { \rho _ { 0 } } } ,
$$

and the trajectory-level Mach amplitude is sampled as $M \sim \mathrm { U n i f o r m } ( 0 . 0 5 , 0 . 3 5 )$ . The physical correlation length $\ell _ { \mathrm { G R F } } = 1$ is fixed across all domain lengths, so increasing L enlarges the domain while preserving the local scale of the initial perturbations.

Spatial and temporal discretization. We use

<table><tr><td>L</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td>nx</td><td>1024</td><td>2048</td><td>3072</td><td>4096</td></tr></table>

so that the grid spacing is fixed at $\Delta x = 5 / 1 0 2 4$ across all settings. No spatial downsampling is applied. Trajectories are recorded every $\Delta t _ { \mathrm { d a t a } } = 0 . 1$ . Training and validation trajectories cover $t \in [ 0 , 1 . 5 ]$ , while the long-horizon test trajectories extend to $T _ { \mathrm { t e s t } } = 1 0$

Numerical solver. We solve the Euler equations using a first-order finite-volume scheme with the local Lax-Friedrichs (Rusanov) flux (Rusanov, 1962) and periodic boundary conditions. The internal time step is chosen adaptively using a CFL number of 0.45, with a maximum solver step of $1 0 ^ { - 3 }$ . Density and pressure are positivity-clipped when necessary to avoid invalid numerical states. Numerical integration is performed in double precision and the trajectory data are stored in single precision.

Dataset splits. For each domain length, we generate 1200 training trajectories and 300 validation trajectories over $t \in [ 0 , 1 . 5 ]$ , together with an independently sampled test set of 200 trajectories over $t \in [ 0 , 1 0 ]$ . Training, validation, and test initial conditions are drawn from the same GRF-based distribution. Thus, the primary distribution shift considered in this benchmark is the substantially longer temporal horizon at test time rather than a change in the initial-condition distribution.

## C.2 2D compressible fluid dynamics

We consider the 2D compressible Navier-Stokes equations following the Compressible Fluid Dynamics (CFD) setting used in PDEBench (Takamoto et al., 2022). The system describes the evolution of density, velocity, pressure, and total energy for a viscous compressible fluid.

Specifically, let $\rho ( t , x )$ denote the mass density, $\mathbf { v } ( t , x ) = ( v _ { 1 } ( t , x ) , v _ { 2 } ( t , x ) )$ the velocity field, $p ( t , x )$ the gas pressure, and $\epsilon = p / ( \Gamma - 1 )$ the internal energy density, where $\Gamma = 5 / 3$ is the heat capacity ratio. The governing equations are

$$
\left\{ \begin{array} { l } { \displaystyle \partial _ { t } \rho + \nabla \cdot ( \rho \mathbf v ) = 0 , } \\ { \displaystyle \rho \left( \partial _ { t } \mathbf v + \mathbf v \cdot \nabla \mathbf v \right) = - \nabla p + \eta \Delta \mathbf v + \left( \zeta + \frac { \eta } { 3 } \right) \nabla ( \nabla \cdot \mathbf v ) , } \\ { \displaystyle \partial _ { t } \left( \epsilon + \frac 1 2 \rho | \mathbf v | ^ { 2 } \right) + \nabla \cdot \left[ \left( \epsilon + p + \frac 1 2 \rho | \mathbf v | ^ { 2 } \right) \mathbf v - \mathbf v \cdot \sigma ^ { \prime } \right] = 0 . } \end{array} \right.\tag{C.5}
$$

Here $\eta$ and $\zeta$ denote the shear and bulk viscosity coeficients, respectively, and $\sigma ^ { \prime }$ is the viscous stress tensor. Because viscosity governs the dissipation of flow structures, larger values of η and $\zeta$ cause the random initial perturbations to decay more rapidly, with the density, pressure, and velocity fields eventually approaching nearly spatially uniform states over the long simulation horizon. To retain nontrivial dynamics beyond the early stage of the trajectory, we therefore focus on a low-viscosity, low-Mach-number regime,

$$
\mu = \zeta = 1 0 ^ { - 8 } , \qquad M = 0 . 1 .
$$

In this regime, spatial structures and coupled density-pressure-velocity interactions remain dynamically active well beyond the training interval rather than rapidly relaxing to a near-stationary state. The fixed Mach number $M = 0 . 1$ maintains a stable low-Mach regime while avoiding strong shocks that could make the comparison overly sensitive to numerical-solver artifacts.

We consider the equations on a two-dimensional periodic domain and represent the learned state using the primitive variables

$$
\mathbf { U } ( t , x ) = \big ( \rho ( t , x ) , v _ { 1 } ( t , x ) , v _ { 2 } ( t , x ) , p ( t , x ) \big ) .
$$

Under this setting, the learning task is to approximate the time-h solution map

$$
S _ { h } : \big ( \rho ( t ) , v _ { 1 } ( t ) , v _ { 2 } ( t ) , p ( t ) \big ) \mapsto \big ( \rho ( t + h ) , v _ { 1 } ( t + h ) , v _ { 2 } ( t + h ) , p ( t + h ) \big ) ,
$$

which is then applied autoregressively over the prediction horizon. The benchmark therefore tests long-horizon prediction of coupled multi-field dynamics, where errors in density, velocity, and pressure can interact and propagate through both nonlinear transport and viscous efects.

Evaluation metrics. For the two-dimensional compressible-flow benchmark, we report rolloutaggregated relative $L ^ { 2 }$ and $H ^ { 1 }$ errors for density, pressure, and velocity, together with their channel average. For $\mathcal { X } \in \{ L ^ { 2 } , H ^ { 1 } \}$ , the metric is

$$
\mathrm { R e l - } \mathcal { X } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \left( \sum _ { n = 1 } ^ { T } \| \widehat { u } _ { i } ^ { n } - u _ { i } ^ { n } \| _ { \mathcal { X } } ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { n = 1 } ^ { T } \| u _ { i } ^ { n } \| _ { \mathcal { X } } ^ { 2 } \right) ^ { 1 / 2 } } ,
$$

with

$$
\| u \| _ { H ^ { 1 } } ^ { 2 } = \| u \| _ { L ^ { 2 } } ^ { 2 } + \| \nabla u \| _ { L ^ { 2 } } ^ { 2 } .
$$

For the velocity field $\mathbf { v } = ( v _ { x } , v _ { y } )$ , the component norms are combined as

$$
\| \mathbf { v } \| _ { \mathcal { X } } ^ { 2 } = \| v _ { x } \| _ { \mathcal { X } } ^ { 2 } + \| v _ { y } \| _ { \mathcal { X } } ^ { 2 } .
$$

The relative $L ^ { 2 }$ error measures overall field accuracy, whereas the $H ^ { 1 }$ error additionally penalizes errors in spatial derivatives and is therefore more sensitive to the preservation of interfaces, gradients, and fine spatial structure. Unlike the inviscid Euler benchmark, the viscous compressible-flow trajectories remain suficiently regular for the derivative-sensitive $H ^ { 1 }$ metric to be meaningful.

## C.2.1 Data Generation for 2D Compressible Fluid Dynamics

Initial conditions. For each trajectory, density, pressure, and the two velocity components are initialized from independent periodic Gaussian random fields on $\mathbb { T } ^ { 2 }$ . A correlation length

$$
\ell \sim \mathrm { U n i f o r m } ( 0 . 0 5 , 0 . 1 5 )
$$

is sampled once per trajectory and shared across the four physical channels. Conditional on $\ell ,$ independent random fields $f _ { \rho } , f _ { v _ { x } } , f _ { v _ { y } } , f _ { p }$ are generated spectrally:

$$
\widehat { f } ( \mathbf { k } ) = S _ { \ell } ( \mathbf { k } ) ^ { 1 / 2 } \xi _ { \mathbf { k } } , ~ S _ { \ell } ( \mathbf { k } ) = \exp \left( - 2 \pi ^ { 2 } \ell ^ { 2 } | \mathbf { k } | ^ { 2 } \right) , ~ \mathbf { k } = ( k _ { x } , k _ { y } ) \in \mathbb { Z } ^ { 2 } \backslash \{ ( 0 , 0 ) \} .
$$

with the zero Fourier mode removed, ${ \widehat { f } } ( 0 , 0 ) = 0$ . This construction corresponds to a centered stationary Gaussian random field with radial a basis function kernel. Each realization is subsequently normalized to zero spatial mean and unit empirical variance.

After normalization, the primitive initial conditions are generated from

$$
\begin{array} { r l } & { \rho ( 0 , \mathbf { x } ) = \rho _ { 0 } + 0 . 1 f _ { \rho } ( \mathbf { x } ) , \quad p ( 0 , \mathbf { x } ) = p _ { 0 } \left( 1 + 0 . 0 5 f _ { p } ( \mathbf { x } ) \right) , } \\ & { v _ { x } ( 0 , \mathbf { x } ) = M c _ { 0 } f _ { v _ { x } } ( \mathbf { x } ) , \quad v _ { y } ( 0 , \mathbf { x } ) = M c _ { 0 } f _ { v _ { y } } ( \mathbf { x } ) , } \end{array}
$$

where the background density $\rho _ { 0 } = 1$ , the background pressure $p _ { 0 } = 1 / \Gamma$ , the sound velocity $c _ { 0 } = \sqrt { \Gamma p _ { 0 } / \rho _ { 0 } } = 1$ , and the Mach number $M = 0 . 1$ . To avoid invalid numerical states, density and pressure are positivity-clipped by $1 0 ^ { - 6 }$ when necessary.

Numerical solver. The equations are solved on a uniform $2 5 6 \times 2 5 6$ periodic grid. Inviscid fluxes are computed using an HLLC approximate Riemann solver (Toro et al., 1994) with secondorder MUSCL reconstruction (Van Leer, 1979) and a minmod slope limiter. Viscous and heatconduction terms are discretized using second-order centered finite diferences, and time integration is performed using a two-stage strong-stability-preserving Runge-Kutta method (Shu and Osher, 1988). The internal time step is selected adaptively using an acoustic CFL number of 0.45 together with the corresponding difusive stability restriction. Density and pressure positivity floors are applied after each Runge-Kutta stage. Numerical evolution is performed in double precision. The resulting trajectories are spectrally downsampled to $6 4 \times 6 4$ , which is the resolution used for all learning experiments.

Temporal sampling and dataset splits. Snapshots are recorded every $\Delta t _ { \mathrm { d a t a } } = 0 . 1$ . Training and validation trajectories cover $t \in [ 0 , 1 ]$ and contain 10 one-step transitions. We generate 1200 training trajectories and 300 validation trajectories. The independently sampled test set contains 200 trajectories over $t \in [ 0 , 1 0 ]$ , corresponding to 100 autoregressive prediction steps. Training, validation, and test initial conditions follow the same distribution, so the primary extrapolation in this benchmark is along the temporal direction.

## C.3 2D Incompressible Navier-Stokes equation

We consider the 2D incompressible Navier-Stokes equations on the periodic domain $\mathbb { T } ^ { 2 }$ . In velocity form,

$$
\partial _ { t } \mathbf { u } + \mathbf { u } \cdot \nabla \mathbf { u } = - \nabla p + \nu \Delta \mathbf { u } + \mathbf { f } _ { u } , \qquad \nabla \cdot \mathbf { u } = 0 ,\tag{C.6}
$$

where ${ \bf u } ( t , x ) = ( u _ { 1 } ( t , x ) , u _ { 2 } ( t , x ) ) \in \mathbb { R } ^ { 2 }$ is the velocity field, $p ( t , x )$ is the pressure, $\nu > 0$ is the kinematic viscosity, and $\mathbf { f } _ { u }$ is an external body force. Following Li et al. (2021, 2022); Chen and Wu (2024), we work with the scalar vorticity

$$
\boldsymbol { \omega } = \nabla \times \mathbf { u } = \partial _ { x } u _ { 2 } - \partial _ { y } u _ { 1 } .
$$

Taking the curl of (C.6) eliminates the pressure and gives

$$
\partial _ { t } \boldsymbol { \omega } + \mathbf { u } \cdot \nabla \boldsymbol { \omega } = \nu \Delta \boldsymbol { \omega } + f , \qquad f = \nabla \times \mathbf { f } _ { u } ,\tag{C.7}
$$

where the velocity is recovered nonlocally from vorticity through the stream function:

$$
{ \bf u } = \nabla ^ { \perp } \psi , \qquad - \Delta \psi = \omega , \qquad \nabla ^ { \perp } \psi = ( \partial _ { y } \psi , - \partial _ { x } \psi ) .\tag{C.8}
$$

In our experiments, the forcing field is time-independent, so the dynamics take the form

$$
\partial _ { t } \omega + \mathbf { u } \cdot \nabla \omega = \nu \Delta \omega + f , \qquad \mathbf { u } = \nabla ^ { \perp } ( - \Delta ) ^ { - 1 } \omega .\tag{C.9}
$$

The learning task is to approximate the conditioned time-∆t solution map

$$
S _ { \Delta t } : ( \omega ( t ) , f ) \mapsto \omega ( t + \Delta t ) ,\tag{C.10}
$$

and to apply the learned map autoregressively over the test horizon.

This benchmark is challenging because the predicted vorticity determines the velocity field that transports future vorticity. Thus, small of-manifold errors in ω can induce nonlocal velocity errors, which then feed back into the advection term $\mathbf { u } \cdot \nabla \omega$ . Under weak viscosity and persistent forcing, these errors may accumulate over long horizons and generate localized high-frequency artifacts. We therefore evaluate not only pointwise rollout accuracy, but also spectral and derivative-sensitive diagnostics such as enstrophy and palinstrophy.

Evaluation metrics. We report rollout-aggregated relative $L ^ { 2 }$ and $H ^ { 1 }$ errors of the vorticity field using the same definitions as in the two-dimensional compressible-flow benchmark. The $L ^ { 2 }$ error measures field-level accuracy, while the $H ^ { 1 }$ error is additionally sensitive to spatial gradients.

We further evaluate enstrophy and palinstrophy,

$$
\mathsf { E n s } ( \omega ) = \frac { 1 } { 2 } \| \omega \| _ { L ^ { 2 } } ^ { 2 } , \qquad \mathsf { P a l } ( \omega ) = \frac { 1 } { 2 } \| \nabla \omega \| _ { L ^ { 2 } } ^ { 2 } ,
$$

which respectively measure the overall vorticity magnitude and the strength of vorticity gradients. For either scalar diagnostic $\mathsf { q } \in \{ \mathsf { E n s } , \mathsf { P a l } \}$ , we compute

$$
\mathrm { R e l E r r } _ { \mathsf { q } } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \left( \sum _ { n = 1 } ^ { T } | \mathsf { q } ( \widehat { \omega } _ { i } ^ { n } ) - \mathsf { q } ( { \omega } _ { i } ^ { n } ) | ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { n = 1 } ^ { T } | \mathsf { q } ( { \omega } _ { i } ^ { n } ) | ^ { 2 } \right) ^ { 1 / 2 } } .
$$

Finally, we compare the isotropic energy spectra of the predicted and reference flows. Let $E _ { i } ^ { n } ( k )$ and $\widehat { E } _ { i } ^ { n } ( k )$ denote the radially aggregated spectral energies at wavenumber shell k. The spectral error is

$$
\mathrm { S p e c E r r } = \frac { 1 } { N _ { \mathrm { t e s t } } } \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \frac { \left( \sum _ { n = 1 } ^ { T } \sum _ { \boldsymbol { k } } | \widehat { E } _ { i } ^ { n } ( \boldsymbol { k } ) - E _ { i } ^ { n } ( \boldsymbol { k } ) | ^ { 2 } \right) ^ { 1 / 2 } } { \left( \sum _ { n = 1 } ^ { T } \sum _ { \boldsymbol { k } } | E _ { i } ^ { n } ( \boldsymbol { k } ) | ^ { 2 } \right) ^ { 1 / 2 } } .
$$

Together, these metrics assess field accuracy, spatial regularity, physically relevant flow statistics, and the distribution of energy across spatial scales.

Table 7: Forcing distributions used for the two-dimensional incompressible Navier-Stokes datasets. Amplitude ranges refer to the target $L ^ { \infty }$ norm of each sampled forcing field.
<table><tr><td>Viscosity</td><td>Forcing</td><td>Length scale</td><td>e Number of modes Amplitude range</td><td></td></tr><tr><td>10-3</td><td>GRF</td><td>[0.05, 0.25]</td><td></td><td>[0.10, 0.30]</td></tr><tr><td>10-3</td><td>WAVE</td><td></td><td>1-4</td><td>[0.10,0.30]</td></tr><tr><td> $1 0 ^ { - 4 } , 1 0 ^ { - 5 }$ </td><td>GRF</td><td>[0.05, 0.15]</td><td></td><td>[0.08,0.20]</td></tr><tr><td> $1 0 ^ { - 4 } , 1 0 ^ { - 5 }$ </td><td>WAVE</td><td></td><td>1-4</td><td>[0.08, 0.16]</td></tr></table>

## C.3.1 Data Generation for 2D Incompressible Navier-Stokes Benchmark

Initial-vorticity distribution. For all viscosities and forcing families, the initial vorticity is sampled on the mean-zero subspace as

$$
\omega _ { 0 } \sim \mathsf { N } \left( 0 , 7 ^ { 3 } ( - \Delta + 4 9 I ) ^ { - 5 / 2 } \right) .
$$

This distribution has the same spectral decay as the Gaussian prior used in Li et al. (2021), with a larger overall amplitude to reduce the mismatch between the weak initial vorticity fields and the higher-amplitude states produced later under persistent forcing. The zero Fourier mode is removed to enforce zero spatial mean.

Trajectory-dependent forcing fields. Unlike the original FNO benchmark in Li et al. (2021), which uses a fixed forcing function, we independently sample one time-independent forcing field for each trajectory. We consider two forcing families.

• For the GRF family, the forcing is sampled from a periodic squared-exponential Gaussian random field with spectral power

$$
S _ { \ell } ( { \mathbf k } ) \propto \exp \left( - 2 \pi ^ { 2 } \ell ^ { 2 } | { \mathbf k } | ^ { 2 } \right) ,
$$

where the correlation length ℓ is sampled independently for each trajectory. The zero mode is removed and each realization is rescaled to a randomly sampled target $L ^ { \infty }$ amplitude.

• For the Wave family, the forcing is a sparse mixture of periodic Fourier modes,

$$
f ( \mathbf { x } ) = \sum _ { j = 1 } ^ { J } a _ { j } \cos \left( 2 \pi \mathbf { k } _ { j } \cdot \mathbf { x } + \phi _ { j } \right) ,
$$

with $J \in \{ 1 , 2 , 3 , 4 \}$ . The wavevectors have nonzero integer components with maximum magnitude three, the phases are sampled uniformly from [0, 2π), and the coeficients decay proportionally to $| \mathbf { k } _ { j } | ^ { - 2 }$ . Each realization is projected to zero mean and rescaled to a randomly sampled target $L ^ { \infty }$ amplitude.

The forcing distributions are summarized in Table 7. For the high-viscosity regime $\nu = 1 0 ^ { - 3 }$ stronger viscous dissipation causes the trajectories to approach a slowly varying regime more rapidly. We therefore use stronger and more spatially correlated forcing to retain nontrivial dynamics over the long simulation horizon.

Numerical solver. Trajectories are generated on a $2 5 6 \times 2 5 6$ periodic grid using a pseudospectral method (Orszag, 1971; Canuto et al., 1988). The velocity and vorticity gradients are computed spectrally, while the nonlinear advection term is evaluated in physical space and transformed back to Fourier space. A two-thirds spectral mask is applied to de-alias the nonlinear term.

Difusion is advanced using a Crank-Nicolson discretization (Crank and Nicolson, 1947), while advection and forcing are treated explicitly. For an internal time step $\delta t$ , the Fourier-space update is

$$
\left( 1 + \frac { \delta t \nu } { 2 } | \mathbf { k } | ^ { 2 } \right) \widehat { \omega } _ { \mathbf { k } } ^ { m + 1 } = \left( 1 - \frac { \delta t \nu } { 2 } | \mathbf { k } | ^ { 2 } \right) \widehat { \omega } _ { \mathbf { k } } ^ { m } - \delta t \widehat { \mathbf { v } ^ { m } \cdot \nabla \omega ^ { m } } _ { \mathbf { k } } + \delta t \widehat { f } _ { \mathbf { k } } .
$$

We use $\delta t = 1 0 ^ { - 4 }$ , following the standard FNO Navier-Stokes benchmark (Li et al., 2021). The resulting advective CFL number remains below 0.03 across all settings. Snapshots are recorded every one physical time unit, and the zero vorticity mode is removed after each step. The simulated vorticity and forcing fields are then spectrally truncated from $2 5 6 \times 2 5 6$ to $6 4 \times 6 4$ for learning.

Training and evaluation datasets. For each viscosity regime, a single model is trained jointly on both forcing families. The training set contains 1,600 trajectories, equally divided between GRF and Wave forcing, and the validation set contains 400 trajectories with the same split. At test time, the model is evaluated separately on the two forcing families.

The training horizons are $T _ { \mathrm { t r a i n } } ~ = ~ 1 0 , 1 2$ and 8 for $\nu = 1 0 ^ { - 3 } , 1 0 ^ { - 4 }$ and $1 0 ^ { - 5 }$ , respectively, while the corresponding test horizons are $T _ { \mathrm { t e s t } } = 5 0 , 3 0$ and 20. Because snapshots are stored at unit intervals, these correspond directly to 10, 12, and 8 training transitions and 50, 30, and 20 autoregressive test steps. Initial-vorticity and forcing fields are sampled independently across the training, validation, and test sets from the same viscosity-dependent distributions. Thus, the primary extrapolation is again along the temporal direction.

## D Implementation Details

This appendix gives the architectural and optimization details for the models compared in $\ S 5$ . All models are implemented in PyTorch and trained as one-step predictors on adjacent snapshots. At test time, all methods are rolled out autoregressively from the exact initial condition, using deterministic mean predictions.

## D.1 Common Data Handling and Model Selection

For each benchmark, the stored trajectories are converted into one-step training pairs

$$
\big ( \mathbf { u } _ { n } , \mathbf { f } , \mathbf { u } _ { n + 1 } \big ) ,
$$

where f is the trajectory-dependent forcing field when present and a zero-valued compatibility channel for the unforced compressible benchmarks. The same one-step pairs, trajectory validation loader, and train/validation split are used for all compared methods on a fixed benchmark.

All models are optimized with AdamW, learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , and gradient clipping with maximum norm 1. We also use a StepLR scheduler with decay factor 0.5 every 100 epochs. All models are trained for 500 epochs and batch size 8. Validation is performed after each epoch. The checkpoint used for reporting is selected by the validation autoregressive rollout relative $L ^ { 2 }$ error when a trajectory validation loader is available; otherwise it is selected by the one-step validation loss.

For multichannel compressible states, the one-step prediction and reconstruction losses use channel-weighted mean-squared error, the channel weights are set to the inverse empirical variance of each physical channel on the training trajectories and normalized to have unit mean. This prevents high-variance channels from dominating the optimization objective. For the scalar vorticity benchmark no channel reweighting is needed.

Rollout safeguard. The reported autoregressive evaluations use deterministic model outputs. During validation and model selection, we optionally clip a single-step increment in $L ^ { \infty }$ norm to avoid numerical overflow from already-diverged checkpoints. The clip thresholds used in the reported runs are reported in Table 8.

## D.2 FNO Backbone

The direct FNO baseline (Li et al., 2021) maps the current physical state to the next physical state,

$$
\widehat { \mathbf { u } } _ { n + 1 } = \mathbf { u } _ { n } + \mathcal { F } _ { \theta } ( \mathbf { u } _ { n } , \mathbf { f } , \mathbf { x } ) ,
$$

where x denotes the periodic grid-coordinate channels. The residual form is used in all reported experiments. The model first lifts the concatenated input channels by a $1 \times 1$ convolution, applies a stack of Fourier layers with pointwise local convolutions and GELU activations, and projects back to the physical channels by two $1 \times 1$ convolutions. All FNO runs use width 64 and six Fourier layers. The 1D Euler experiments retain 32 one-dimensional Fourier modes, while the 2D compressible-flow and incompressible Navier-Stokes experiments retain $1 6 \times 1 6$ modes.

## D.3 Deterministic Latent Baseline

FNO-AE uses the same spatial latent representation as VAMO but removes all stochastic components. The encoder is a resolution-preserving residual convolutional network

$$
E _ { \phi } : { \mathbf { u } } _ { n } \mapsto { \mathbf { z } } _ { n } \in \mathbb { R } ^ { C _ { z } \times N _ { \mathrm { r e s } } } ,
$$

with four residual blocks and hidden width 64. We use $C _ { z } = 3 2$ latent channels for the 1D Euler benchmark and $C _ { z } = 1 6$ for both the 2D compressible fluid dynamics and incompressible Navier-Stokes benchmarks. For multichannel states, the encoder input contains both the physical fields and centered finite-diference gradient features; for scalar vorticity, the same construction is applied channelwise. The decoder has the mirrored residual-convolutional structure and output dimension equal to the number of physical state channels.

The deterministic latent transition is a residual FNO,

$$
{ \bf z } _ { n + 1 } = { \bf z } _ { n } + \mathcal { G } _ { \boldsymbol { \theta } } ( { \bf z } _ { n } , { \bf f } , { \bf x } ) ,
$$

using the same retained modes and layer counts as the corresponding VAMO latent transition. The deterministic latent objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { A E } } = \mathcal { L } _ { \mathrm { s t e p } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } , } \end{array}
$$

where $\mathcal { L } _ { \mathrm { s t e p } }$ is the one-step prediction loss and ${ \mathcal { L } } _ { \mathrm { r e c } }$ reconstructs the current state from the encoded latent. The deterministic latent runs use $\lambda _ { \mathrm { { r e c } } } = 1$

## D.4 VAMO Architecture and Structured Noise

VAMO uses the same encoder, decoder, and residual latent FNO mean transition as FNO-AE, but the encoder outputs both a mean field and a log-variance field:

$$
( \mu _ { \phi } ( \mathbf { u } ) , \ell _ { \phi } ( \mathbf { u } ) ) = \mathrm { E n c } _ { \phi } ( \mathbf { u } ) .
$$

During training, the posterior log-variance is clamped to $[ - 8 , 2 ]$ before sampling. A posterior latent sample is generated by

$$
\mathbf { z } _ { n } = \mu _ { \phi } ( \mathbf { u } _ { n } ) + \exp \left( \frac { 1 } { 2 } \ell _ { \phi } ( \mathbf { u } _ { n } ) \right) \odot \varepsilon _ { \mathrm { e n c } } ,
$$

where $\varepsilon _ { \mathrm { e n c } }$ is a spectrally filtered standard Gaussian field. This filtering is implemented by FFT as

$$
{ \widehat \varepsilon } ( \mathbf { k } ) = \left( 1 + \ell ^ { 2 } | 2 \pi \mathbf { k } | ^ { 2 } \right) ^ { - s / 2 } { \widehat \xi } ( \mathbf { k } ) ,
$$

with orthonormal FFT normalization. This is the discrete implementation of the Laplacianresolvent covariance described in §4.2. The spectral decay exponent is $s = 2$ in all VAMO runs.

The latent prior mean is the residual FNO transition

$$
\begin{array} { r } { \mu _ { p } = T _ { \boldsymbol { \theta } } ( \mathbf { z } _ { n } ) , } \end{array}
$$

and the transition noise amplitude is a scalar predicted separately for each sample. The amplitude head consists of two convolutional layers with GELU activations, global spatial averaging, and a final $1 \times 1$ convolution. In the multichannel implementation used for the reported Euler, compressibleflow, and incompressible Navier-Stokes experiments, the final scalar is passed through a softplus and then shifted and clipped:

$$
\alpha _ { \theta } ( \mathbf { z } , \mathbf { f } ) = \operatorname* { m i n } \{ \alpha _ { \mathrm { m a x } } , \alpha _ { \mathrm { m i n } } + \mathrm { s o f t p l u s } ( a _ { \theta } ( \mathbf { z } , \mathbf { f } ) ) \} .
$$

The softplus component is initialized to 0.10. Thus the initial efective amplitude before clipping is approximately 0.1001 for Euler and CFD2D, where $\alpha _ { \mathrm { m i n } } = 1 0 ^ { - 4 }$ , and 0.11 for NS2D, where $\alpha _ { \mathrm { m i n } } = 1 0 ^ { - 2 }$

The stochastic transition sample is

$$
{ \bf z } _ { n + 1 } = \mu _ { p } + \alpha _ { \theta } ( { \bf z } _ { n } , { \bf f } ) \varepsilon _ { \mathrm { t r } } ,
$$

where $\varepsilon _ { \mathrm { t r } }$ is generated by the same spectral filter with the transition correlation length. During training, the CFD2D and NS2D VAMO samplers add a spectral variance floor $1 0 ^ { - 2 }$ before taking the square root of the filter variance. Inference uses the deterministic mean only: the initial state is encoded as $\mu _ { \phi } ( \mathbf { u } _ { 0 } )$ , the latent is advanced by $T _ { \theta } ,$ and each latent state is decoded by $D _ { \psi }$

The VAMO training loss is

$$
\mathcal { L } _ { \mathrm { V A M O } } = \mathcal { L } _ { \mathrm { s t e p } } + \beta _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } + \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } .
$$

The prediction and reconstruction terms are the same channel-weighted MSE terms used for the deterministic latent baseline. The KL term compares the encoded posterior at the next state with the predicted latent prior. In the numerical implementation this KL is evaluated using the encoder’s pointwise diagonal posterior variance and the transition’s scalar per-sample variance $\alpha _ { \theta } ^ { 2 } ;$ the spectral covariance is used for reparameterized sampling of the stochastic latent states.

Table 8: Main implementation hyperparameters from the experiment configurations.
<table><tr><td>Setting</td><td>1D Euler  $L \in \{ 5 , 1 0 , 1 5 \}$ </td><td>1D Euler  $L = 2 0$ </td><td>2D CFD  $\nu = \eta = 1 0 ^ { - 8 }$ </td><td> $\nu = 1 0 ^ { - 3 }$ </td><td>2D Navier-Stokes 2D Navier-Stokes  $\nu \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ </td></tr><tr><td># Training trajectories</td><td>1200</td><td>1200</td><td>1200</td><td>1600</td><td>1600</td></tr><tr><td># Validation trajectories</td><td>300</td><td>300</td><td>300</td><td>400</td><td>400</td></tr><tr><td>Resolution  $N _ { \mathrm { r e s } }$ </td><td>{1024, 2048, 3072}</td><td>4096</td><td> $6 4 \times 6 4$ </td><td> $6 4 \times 6 4$ </td><td> $6 4 \times 6 4$ </td></tr><tr><td>Batch size</td><td>8</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td># Epochs</td><td>500</td><td>500</td><td>500</td><td>500</td><td>500</td></tr><tr><td>Training transitions  $N _ { \mathrm { t r a i n } }$ </td><td>15</td><td>15</td><td>10</td><td>10</td><td>{12, 8}</td></tr><tr><td>FNO width</td><td>64</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td># FNO layers</td><td>6</td><td>6</td><td>6</td><td>6</td><td>6</td></tr><tr><td>Fourier modes</td><td>32</td><td>32</td><td>16 × 16</td><td>16 × 16</td><td>16 × 16</td></tr><tr><td># Latent channels  $C _ { z }$ </td><td>32</td><td>32</td><td>16</td><td>16</td><td>16</td></tr><tr><td># Encoder/decoder blocks</td><td>4/4</td><td>4/4</td><td>4/4</td><td>4/4</td><td>4/4</td></tr><tr><td>Encoder filter  $( \ell _ { \mathrm { e n c } } , s )$ </td><td>(0.01, 2)</td><td>(0.01, 2)</td><td>(0.01, 2)</td><td>(0.01,2)</td><td>(0.01, 2)</td></tr><tr><td>Transition filter  $( \ell _ { \mathrm { t r } } , s )$ </td><td>(0.05, 2)</td><td>(0.05, 2)</td><td>(0.1,2)</td><td>(0.05, 2)</td><td>(0.05, 2)</td></tr><tr><td>Amplitude range  $[ \alpha _ { \mathrm { m i n } } , \alpha _ { \mathrm { m a x } } ]$ </td><td>[10−4, 5]</td><td>[10−4, 5]</td><td>[10−4, 5]</td><td> $[ 1 0 ^ { - 2 } , 5 ]$ </td><td>[10−2, 5]</td></tr><tr><td>Loss weights  $( \beta _ { \mathrm { K L } } , \lambda _ { \mathrm { r e c } } )$ </td><td> $( 1 0 ^ { - 3 } , 1 )$ </td><td> $( 1 0 ^ { - 2 } , 1 )$ </td><td> $( 1 0 ^ { - 4 } , 1 )$ </td><td> $( 1 0 ^ { - 3 } , 1 )$ </td><td>(10−2, 1)</td></tr><tr><td>FNO+Noise lift std.</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.1</td><td>0.1</td></tr><tr><td>FNO+Noise lift length</td><td>0.05</td><td>0.05</td><td>0.1</td><td>0.05</td><td>0.05</td></tr><tr><td>Rollout increment clip</td><td>5</td><td>5</td><td>10</td><td>10</td><td>10</td></tr></table>

## D.5 FNO+Noise Baseline and Model Complexity

FNO+Noise uses the same architecture, optimizer, data loaders, and model selection rule as the direct FNO baseline. The only change is that during training we add spectrally filtered Gaussian noise to the lifted FNO feature field before the Fourier layers:

$$
h  h + \sigma _ { \mathrm { l i f t } } \varepsilon _ { \mathrm { l i f t } } ,
$$

where $\varepsilon _ { \mathrm { l i f t } }$ is sampled with the same FFT filter family as above and spectral decay $s = 2$ . The experiments use $\sigma _ { \mathrm { l i f t } } = 0 . 0 2$ for Euler1D and CFD2D, and $\sigma _ { \mathrm { l i f t } } = 0 . 1$ for NS2D, which are com parable to the VAMO counterparts. The corresponding lift-noise correlation lengths are reported in Table 8. This noise is disabled during validation and test rollout. The baseline therefore tests generic feature-space noise injection without a latent posterior, learned transition uncertainty, or KL alignment term.

Model size. For completeness, Table 9 reports the number of trainable parameters for the models used in each benchmark. The FNO+Noise baseline has exactly the same trainable architecture as the direct FNO baseline; its additional stochastic perturbation introduces no learnable parameters. Similarly, FNO-AE and VAMO share the same encoder-transition-decoder backbone. The additional parameters in VAMO arise only from the encoder variance output and the lightweight transition amplitude head.

Table 9: Number of trainable parameters for the evaluated models. FNO+Noise has the same architecture and parameter count as FNO. Counts are independent of the spatial resolution because all models are fully convolutional or neural-operator based.
<table><tr><td>Benchmark</td><td>FNO</td><td>FNO-AE</td><td>FNO-VAMO</td></tr><tr><td>1D Euler</td><td>1.607M</td><td>1.945M</td><td>1.979M</td></tr><tr><td>2D compressible fluid dynamics</td><td>25.200M</td><td>25.801M</td><td>25.817M</td></tr><tr><td>2D incompressible Navier-Stokes</td><td>25.200M</td><td>25.796M</td><td>25.811M</td></tr></table>

## E Additional Experimental Results

## E.1 Comparison with Autoregressive Curriculum Training

Autoregressive curriculum training has been used to reduce the discrepancy between teacher-forced training and autoregressive deployment by progressively exposing a model to longer rollouts generated from its own predictions (Li et al., 2022; Hagnberger et al., 2025). To assess whether a deterministic baseline trained with explicit multistep rollouts can match VAMO’s long-horizon performance, we construct a curriculum-trained FNO baseline for the two more challenging Navier-Stokes regimes, $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ . The comparison between standard and curriculum-trained FNO isolates the efect of the training strategy, as the underlying architecture and evaluation protocol remain unchanged.

Curriculum setting. Specifically, let K denote the current curriculum length. Starting from the ground-truth initial state, the model is applied autoregressively for the first K steps, with each prediction fed back as the input to the next step. For the remaining steps in the training trajectory, the model uses ground-truth inputs as in standard teacher forcing. Thus, increasing K gradually shifts the objective from one-step prediction toward full autoregressive rollout training. The curriculum begins with $K = 1$ , which is maintained for the first 100 epochs. The rollout length is then increased linearly until it reaches the full training horizon, namely $K = 1 2$ for $\nu = 1 0 ^ { - 4 }$ and $K = 8$ for $\nu = 1 0 ^ { - 5 }$ , at the end of 500 training epochs.

The curriculum baseline uses the same optimizer settings and FNO architecture as the standard FNO baseline, with width 64, 6 Fourier layers, and $1 6 \times 1 6$ retained Fourier modes. At test time, all methods are evaluated identically: starting from the initial condition, each model is rolled out autoregressively and without intermediate ground-truth correction until $T _ { \mathrm { t e s t } }$ . Thus, the comparison isolates the efect of the curriculum training strategy while keeping the model architecture and evaluation protocol unchanged.

Table 10 reports relative $L ^ { 2 }$ and $H ^ { 1 }$ errors over both the training horizon $[ 0 , T _ { \mathrm { t r a i n } } ]$ and the full rollout horizon $[ 0 , T _ { \mathrm { t e s t } } ]$ . We restrict this comparison to these two metrics because its purpose is to assess predictive accuracy and spatial regularity under diferent training strategies; the additional physical diagnostics are studied separately in the main experiments.

As shown in Table 10, curriculum training substantially improves the FNO baseline over the full rollout horizon, reducing its relative $L ^ { 2 }$ errors by 36.6%–58.4% and its relative $H ^ { 1 }$ errors by 65.0%– 68.4% across the four evaluation settings. Nevertheless, VAMO achieves the lowest full-horizon errors in all cases, further reducing the $L ^ { 2 }$ and $H ^ { 1 }$ errors of FNO-Curriculum by 22.6%–76.2% and $3 1 . 6 \% { - } 8 8 . 4 \%$ , respectively. VAMO also exhibits consistently smaller standard deviations over the full rollout horizon, indicating substantially lower variability across test trajectories and more reliable long-horizon predictions. Notably, for $\nu = 1 0 ^ { - 5 }$ , the improved full-horizon performance of FNO-Curriculum is accompanied by larger training-horizon errors than standard FNO under both forcing families. This suggests an accuracy-stability tradeof rather than a uniform improvement across the temporal regimes represented during training and deployment.

Table 10: Comparison with autoregressive curriculum training on the two more challenging Navier-Stokes regimes. Errors are evaluated over the training horizon $[ 0 , T _ { \mathrm { t r a i n } } ]$ and the full rollout horizon $[ 0 , T _ { \mathrm { t e s t } } ]$ . All entries report mean ± standard deviation over 200 test trajectories.
<table><tr><td colspan="4">Evaluation setting</td><td colspan="2">Training horizon</td><td colspan="2">Full horizon</td></tr><tr><td>Viscosity</td><td>Forcing</td><td>Method</td><td>Training</td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td><td> $\mathrm { R e l } . L ^ { 2 }$ </td><td> $\mathrm { R e l } . H ^ { 1 }$ </td></tr><tr><td rowspan="6"> $\nu = 1 0 ^ { - 4 }$ </td><td></td><td>FNO</td><td>Standard</td><td> $0 . 0 1 2 3 _ { \pm 0 . 0 3 4 1 }$ </td><td> $0 . 1 0 1 3 _ { \pm 0 . 3 2 4 5 }$ </td><td> $0 . 4 0 6 8 _ { \pm 0 . 5 0 9 4 }$ </td><td> $3 . 1 0 9 4 _ { \pm 4 . 0 2 0 4 }$ </td></tr><tr><td>GRF</td><td>FNO</td><td>Curriculum</td><td> $0 . 0 0 9 3 { \scriptstyle \pm 0 . 0 0 5 3 }$ </td><td> $0 . 0 5 1 3 { \scriptstyle \pm 0 . 0 3 5 7 }$ </td><td> $0 . 1 6 9 3 { \scriptstyle \pm 0 . 1 9 8 1 }$ </td><td> $0 . 9 8 1 2 _ { \pm 1 . 2 5 0 3 }$ </td></tr><tr><td></td><td>VAMO</td><td>Standard</td><td> $\mathbf { 0 . 0 0 5 7 \bot 0 . 0 0 1 7 }$ </td><td> $\mathbf { 0 . 0 2 1 8 _ { \pm 0 . 0 0 7 0 } }$ </td><td> $\mathbf { 0 . 0 4 0 3 _ { \pm 0 . 0 5 6 2 } }$ </td><td> $\mathbf { 0 . 1 1 3 9 _ { \pm 0 . 1 5 4 7 } }$ </td></tr><tr><td></td><td>FNO</td><td>Standard</td><td> $0 . 0 1 1 8 { \scriptstyle \pm 0 . 0 3 6 3 }$ </td><td> $0 . 0 8 2 8 { \scriptstyle \pm 0 . 2 9 9 0 }$ </td><td> $0 . 3 5 7 0 { \scriptstyle \pm 0 . 6 1 0 2 }$ </td><td> $1 . 6 1 3 1 { \scriptstyle \pm 2 . 7 7 7 8 }$ </td></tr><tr><td>WAVE</td><td>FNO</td><td>Curriculum</td><td> $0 . 0 1 0 9 _ { \pm 0 . 0 1 6 2 }$ </td><td> $0 . 0 4 7 0 { \scriptstyle \pm 0 . 0 7 8 6 }$ </td><td> $0 . 2 0 9 6 { \scriptstyle \pm 0 . 2 9 9 1 }$ </td><td> $0 . 5 6 4 1 { \scriptstyle \pm 0 . 8 0 8 7 }$ </td></tr><tr><td></td><td>VAMO</td><td>Standard</td><td> $\mathbf { 0 . 0 0 5 8 _ { \pm 0 . 0 0 3 1 } }$ </td><td> $\mathbf { 0 . 0 1 8 2 _ { \pm 0 . 0 1 1 0 } }$ </td><td> $\mathbf { 0 . 1 1 9 8 _ { \pm 0 . 1 7 8 5 } }$ </td><td> $\mathbf { 0 . 2 2 1 2 _ { \pm 0 . 2 6 8 0 } }$ </td></tr><tr><td rowspan="6"> $\nu = 1 0 ^ { - 5 }$ </td><td rowspan="3">GRF</td><td>FNO</td><td>Standard</td><td> $\mathbf { 0 . 0 1 3 2 } _ { \pm 0 . 0 0 5 0 }$ </td><td> $\mathbf { 0 . 1 1 8 6 _ { \pm 0 . 0 5 0 2 } }$ </td><td> $0 . 2 9 8 7 { \scriptstyle \pm 0 . 4 3 3 0 }$ </td><td> $2 . 2 6 2 1 { \scriptstyle \pm 3 . 6 5 8 5 }$ </td></tr><tr><td>FNO</td><td>Curriculum</td><td> $0 . 0 1 9 7 { \scriptstyle \pm 0 . 0 0 4 4 }$ </td><td> $0 . 1 6 6 8 _ { \pm 0 . 0 4 2 1 }$ </td><td> $0 . 1 5 0 3 { \scriptstyle \pm 0 . 2 7 1 9 }$ </td><td> $0 . 7 8 6 7 { \scriptstyle \pm 0 . 7 6 2 7 }$ </td></tr><tr><td>VAMO</td><td>Standard</td><td> $0 . 0 1 3 9 { \scriptstyle \pm 0 . 0 0 1 8 }$ </td><td> $0 . 1 2 3 8 _ { \pm 0 . 0 2 3 0 }$ </td><td> $\mathbf { 0 . 0 8 4 1 _ { \pm 0 . 0 5 6 0 } }$ </td><td> $\mathbf { 0 . 3 8 5 8 _ { \pm 0 . 1 5 6 1 } }$ </td></tr><tr><td rowspan="3">WAVE</td><td>FNO</td><td>Standard</td><td> $\mathbf { 0 . 0 1 2 2 _ { \pm 0 . 0 0 5 9 } }$ </td><td> $\mathbf { 0 . 0 9 9 8 _ { \pm 0 . 0 4 7 2 } }$ </td><td> $0 . 2 9 9 3 { \scriptstyle \pm 0 . 4 0 8 3 }$ </td><td> $1 . 9 6 8 5 { \scriptstyle \pm 3 . 1 2 1 9 }$ </td></tr><tr><td>FNO</td><td>Curriculum</td><td> $0 . 0 1 8 4 { \scriptstyle \pm 0 . 0 0 5 2 }$ </td><td> $0 . 1 4 5 3 { \scriptstyle \pm 0 . 0 5 2 4 }$ </td><td> $0 . 1 8 9 9 { \scriptstyle \pm 0 . 1 4 5 6 }$ </td><td> $0 . 6 5 9 6 { \scriptstyle \pm 0 . 4 3 9 0 }$ </td></tr><tr><td>VAMO</td><td>Standard</td><td> $0 . 0 1 2 6 { \scriptstyle \pm 0 . 0 0 2 4 }$ </td><td> $0 . 1 0 4 1 { \scriptstyle \pm 0 . 0 2 9 4 }$ </td><td> $\mathbf { 0 . 1 4 6 9 } _ { \pm 0 . 1 1 2 7 }$ </td><td> $\mathbf { 0 . 4 5 0 9 _ { \pm 0 . 1 9 8 1 } }$ </td></tr></table>

Overall, autoregressive curriculum training considerably improves the deterministic FNO baseline by exposing it to model-generated states, but it does not close the gap to VAMO in any of the four full-horizon comparisons. Notably, VAMO attains stronger long-horizon performance using only a local variational objective, whereas FNO-Curriculum explicitly optimizes progressively longer autoregressive rollouts up to the full training horizon. Nevertheless, the two mechanisms are potentially complementary. A promising extension is to derive a rollout-aware multistep variational objective for VAMO and incorporate a rollout-length curriculum into its prediction terms. Such an extension could further reduce the mismatch between local training and long-horizon autoregressive deployment while retaining structured variational regularization of the latent dynamics.

## E.2 Additional Quantitative Results

We provide additional quantitative results complementing the aggregate comparisons and temporal error analysis in §5. In particular, we report per-trajectory error distributions for the 1D Euler benchmark and the $\nu = 1 0 ^ { - 3 }$ incompressible Navier-Stokes setting, together with the remaining temporal error trends and physical diagnostics omitted from the main text for brevity.

1D Euler equations. Figures 11–12 show the distributions of per-trajectory full-rollout relative $L ^ { 2 }$ and $L ^ { 1 }$ errors for all four domain lengths. Across the evaluated settings, VAMO generally exhibits lower typical errors and reduced upper tails, with the clearest separation appearing in the more challenging large-domain regime. For $L = 2 0$ , in particular, the deterministic latent baseline develops a pronounced heavy tail associated with unstable rollouts, whereas the VAMO error distribution remains substantially more concentrated. These results complement the aggregate errors in Table 3 by showing that the long-horizon improvement is reflected across the test distribution rather than being driven by a small number of trajectories.

Figure 13 further reports the temporal evolution of the Euler rollout errors for the additional domain-length settings. The same pattern observed in §5.3 persists: the methods can remain comparable during the early rollout, while their behavior separates as prediction proceeds beyond the training horizon. VAMO exhibits slower late-time error growth, consistent with the full-rollout results in Table 3 and the representative settings shown in the main text.

2D incompressible Navier-Stokes equations. We additionally report results for the more dissipative $\nu = 1 0 ^ { - 3 }$ regime, which is omitted from the detailed temporal analysis in the main text. Figure 14 shows the per-trajectory full-rollout relative $L ^ { 2 }$ and $H ^ { 1 }$ error distributions under both GRF and Wave forcing. VAMO maintains low typical errors and comparatively tight distributions for both forcing families. The deterministic latent baseline is substantially less stable and exhibits pronounced heavy-tailed errors, while the direct and noise-injection baselines remain more stable but generally less accurate than VAMO.

The corresponding temporal errors are shown in Figure 15. Under both forcing families, VAMO limits the growth of field-level, derivative-sensitive, and physical-statistic errors throughout the long rollout. The advantage is particularly clear relative to the deterministic latent model, whose errors grow rapidly after the training horizon. Compared with the lower-viscosity regimes studied in the main text, the direct FNO and FNO+Noise baselines are more competitive at $\nu = 1 0 ^ { - 3 }$ , consistent with the stronger physical dissipation in this setting.

Finally, Figure 16 compares the enstrophy and palinstrophy trajectories for representative $\nu =$ $1 0 ^ { - 3 }$ test samples. The deterministic latent baseline frequently develops severe late-time inflation in both statistics, while the direct baselines exhibit smaller but still visible deviations on some trajectories. VAMO more closely tracks the scale and temporal evolution of the reference statistics across both forcing families. Together with the results for $\nu = 1 0 ^ { - 4 }$ and $\nu = 1 0 ^ { - 5 }$ in $\ S 5 . 4$ , these observations show that the improved physical-statistic stability of VAMO persists across the full range of viscosities considered in our experiments.

## E.3 Additional Qualitative Results

We provide additional representative autoregressive rollouts from the test sets to complement the quantitative results in $\ S 5$ . The visualizations illustrate how the accumulated rollout errors manifest in the predicted physical fields and highlight the qualitative diferences among the evaluated models.

1D Euler equations. Figures 17–20 show representative trajectories for the four domain lengths $L \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ . At early rollout times, all methods generally reproduce the large-scale wave structure of the reference solution. As the rollout proceeds, diferences become increasingly visible around sharp fronts and interacting waves. The deterministic latent baseline is particularly susceptible to overshoots and oscillatory artifacts, with the degradation becoming more pronounced for the larger domains. The direct FNO and FNO+Noise baselines remain more stable but accumulate visible phase and amplitude errors at later times. VAMO remains more closely aligned with the reference density, velocity, and pressure profiles over the full prediction horizon, consistent with the error trends reported in §5.3.

(a) 1D Euler per-trajectory rollout errors, domain length L = 5.  
![](images/258e3b88b5b6eb4c4ec71e74eaba0e1f985aff82b020dc43cf88b8d2fe4b2000.jpg)

(b) 1D Euler per-trajectory rollout errors, domain length L = 10.  
![](images/246a8fd86ed001a1297f0e4145a09d854c6d350b3557621c4845a5cf1a31e8d1.jpg)  
Figure 11: Distribution of per-trajectory aggregate relative $L ^ { 2 }$ and $L ^ { 1 }$ errors for the 1D Euler benchmark. The logarithmic scale highlights both the typical error and the heavy upper tails associated with unstable rollouts.

(a) 1D Euler per-trajectory rollout errors, domain length L = 15.  
![](images/f31adb3e1a70459ee2853dd89e352bc5f5f686c2a0096a8c07cdb9fa51f35fc6.jpg)

(b) 1D Euler per-trajectory rollout errors, domain length L = 20.  
![](images/fc98976c48359a9fdf3781e43aa2a810b414b720f88aab2a3c6f32f25cb32d62.jpg)  
Figure 12: Distribution of per-trajectory aggregate relative $L ^ { 2 }$ and $L ^ { 1 }$ errors for the 1D Euler benchmark. The logarithmic scale highlights both the typical error and the heavy upper tails associated with unstable rollouts.

![](images/48675b7a762e17160133d57a7f738d5934d2c4d8cf8ea0cd78b7c8d1b17c54ff.jpg)  
(a) Domain length L = 10.

![](images/63c843e27b24ce66c7a24de9f37c2cf4e13060eac6f76f596aa84de74d0b6739.jpg)  
(b) Domain length L = 20.  
Figure 13: Error trend for 1D Euler Benchmarks with $L \in \{ 1 0 , 1 5 \}$

![](images/e2d88b60a845bb7f087cc373ca90e5d8cddfcce971a4c4f653febba4e8187b1e.jpg)  
Figure 14: Distribution of per-trajectory full-rollout relative $L ^ { 2 }$ and $H ^ { 1 }$ errors for the 2D incompressible Navier-Stokes benchmark, $\nu = 1 0 ^ { - 3 }$ . Results are shown separately for GRF and Wave forcing. The logarithmic scale highlights both the typical error and the heavy upper tails associated with unstable rollouts.

2D compressible fluid dynamics. Figure 21 presents a representative rollout of the 2D compressible fluid dynamics benchmark. The diferences among the methods are modest near the beginning of the trajectory but become clearer beyond the training horizon. The baseline predictions progressively develop larger spatial errors and fine-scale artifacts, particularly in the velocity fields, whereas VAMO better preserves the coherent spatial organization of the density, velocity, and pressure fields throughout the rollout. These qualitative diferences agree with the substantially lower full-horizon $L ^ { 2 }$ and $H ^ { 1 }$ errors reported in Table 4.

2D incompressible Navier–Stokes equations. Figures 22–24 provide representative Navier-Stokes rollouts for $\nu \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ under both GRF and Wave forcing. In each panel, all predictions are shown using the corresponding ground-truth color scale, allowing deviations in vorticity magnitude and spatial structure to be compared directly. The baseline methods generally track the reference dynamics during the early rollout but can develop increasingly pronounced smallscale and oscillatory artifacts at later times. This behavior is most severe for the deterministic latent baseline, which becomes unstable in several examples. The direct FNO and FNO+Noise models remain more controlled but still exhibit substantial late-time deviations in the more challenging trajectories.

In contrast, VAMO more consistently preserves the dominant vorticity structures and avoids the severe high-frequency artifacts observed in the unstable baseline rollouts. The qualitative behavior is consistent across viscosity levels and forcing families and complements the aggregate field errors, temporal error trends, and physical-statistic diagnostics reported in the main text. Together, these examples provide a spatial view of the central empirical finding: the primary advantage of VAMO emerges during prolonged autoregressive evolution, where the variational formulation helps limit the accumulation of unphysical structure.

![](images/f9220933c2b614f413aed7022a9cea01c9f56b6bac80b8460269d8abe41b1d28.jpg)  
(a) GRF forcing type.

![](images/e728639bc98b4304dacd8bc31699ddf09e93cfda422f87b06ba437fae7fdb548.jpg)  
(b) Wave forcing type.  
Figure 15: Error trend for 2D incompressible NS Benchmark with $\nu = 1 0 ^ { - 3 }$

![](images/5c4a15fc177145e1b89088a549d6916c003976a9397ad8a38ea353447fc89ebb.jpg)  
(a) Viscosity $\nu = 1 0 ^ { - 3 } ,$ GRF forcing.

![](images/d714cb327792b20c84b456175226c41dba6d60ab4c284daa267bc56079428301.jpg)  
(b) Viscosity $\nu = 1 0 ^ { - 3 } ,$ , Wave forcing.  
Figure 16: Physical diagnostics for the 2D incompressible Navier-Stokes benchmark. Each panel compares enstrophy and palinstrophy along 16 ground-truth and predicted trajectories. Solid translucent curves denote ground truth, dashed curves denote predictions, and the shaded region marks the training horizon.

![](images/87d30ad8e9c03c00e6d66a959ce5ddfa97d27241117b7853330fbad815de006a.jpg)  
Figure 17: Representative autoregressive rollout for the 1D Euler equations with $L = 5$ . Rows show density $\rho ,$ velocity $u ,$ and pressure $p ,$ while columns correspond to selected rollout times. The solid gray curve denotes the ground truth, and the colored dashed curves denote predictions from the evaluated models. All methods remain accurate at early times, whereas the deterministic latent baseline develops increasingly pronounced overshoots and oscillatory errors around sharp wave structures during the later rollout. VAMO remains closely aligned with the reference trajectory over the full prediction horizon.

![](images/f70a350b605e062b29ee5a0e02493fa5afed43b09788dc65198af40f33d0a496.jpg)  
Figure 18: Representative autoregressive rollout for the 1D Euler equations with $L = 1 0 .$ Rows show density $\rho ,$ velocity $u ,$ and pressure $p ,$ while columns correspond to selected rollout times. The solid gray curve denotes the ground truth, and the colored dashed curves denote predictions from the evaluated models. All methods remain accurate at early times, whereas the deterministic latent baseline develops increasingly pronounced overshoots and oscillatory errors around sharp wave structures during the later rollout. VAMO remains closely aligned with the reference trajectory over the full prediction horizon.

![](images/1de2df1f8e86279334733d8f61ebc72de13ee3e2b8d9ec96336a5dea7d0ec24c.jpg)  
Figure 19: Representative autoregressive rollout for the 1D Euler equations with $L = 1 5$ . Rows show density $\rho ,$ velocity $u ,$ and pressure $p ,$ while columns correspond to selected rollout times. The solid gray curve denotes the ground truth, and the colored dashed curves denote predictions from the evaluated models. All methods remain accurate at early times, whereas the deterministic latent baseline develops increasingly pronounced overshoots and oscillatory errors around sharp wave structures during the later rollout. VAMO remains closely aligned with the reference trajectory over the full prediction horizon.

![](images/cb0233c7989fefbde9b8f2c22672c09c5ed182c927f140ae34678542e9ed720e.jpg)  
Figure 20: Representative autoregressive rollout for the 1D Euler equations with $L = 2 0 .$ Rows show density $\rho ,$ velocity $u ,$ and pressure $p ,$ while columns correspond to selected rollout times. The solid gray curve denotes the ground truth, and the colored dashed curves denote predictions from the evaluated models. All methods remain accurate at early times, whereas the deterministic latent baseline develops increasingly pronounced overshoots and oscillatory errors around sharp wave structures during the later rollout. VAMO remains closely aligned with the reference trajectory over the full prediction horizon.

![](images/3ff1c2cf7a660d91f835ea0a1bd2ff6f4c49f6db0a69d9403fe3a6be83d44836.jpg)  
Figure 21: Representative autoregressive rollout for the 2D compressible fluid-dynamics benchmark. Rows compare the ground truth with predictions from FNO, FNO+Noise, FNO-AE, and FNO-VAMO for density, the two velocity components, and pressure.

![](images/5c34d17e1405ee685136433079a7731fd2709b7ea3c0c40192e974d387887057.jpg)  
(a) $\nu = 1 0 ^ { - 3 }$ , GRF forcing, Sample 177.

![](images/34788f8788979cc487ce62e099087612c10904841a7cec92fda0e1d1d96994c4.jpg)  
(b) $\nu = 1 0 ^ { - 3 }$ , Wave forcing, Sample 70.  
Figure 22: Representative autoregressive rollouts for the 2D incompressible Navier-Stokes equations with $\nu = 1 0 ^ { - 3 }$ . Each panel shows the initial condition, the trajectory-specific forcing, the groundtruth vorticity, and predictions from the four evaluated methods at selected times. All predicted snapshots use the corresponding ground-truth color scale.

![](images/0af3db2cb86518ec35ff78005a09fea086ed382d648bcde9ffa708e9ef63f59a.jpg)  
(a) $\nu = 1 0 ^ { - 4 }$ , GRF forcing, Sample 0.

![](images/4f8e2d87ec8f431ec88664a51312cd754e431d10eeb343c4a953395ec4daf12e.jpg)  
(b) $\nu = 1 0 ^ { - 4 }$ , Wave forcing, Sample 136.  
Figure 23: Representative autoregressive rollouts for the 2D incompressible Navier-Stokes equations with $\nu = 1 0 ^ { - 4 }$ . Each panel shows the initial condition, the trajectory-specific forcing, the groundtruth vorticity, and predictions from the four evaluated methods at selected times. All predicted snapshots use the corresponding ground-truth color scale.

![](images/aa6ff389da47beb20cc0b6b88100c5c41f394694c30e35d84e10bec725fbb122.jpg)  
(a) $\nu = 1 0 ^ { - 5 }$ , GRF forcing, Sample 0.

![](images/9198b6c4029fc4908acd92b495cc766231b1af98f09cf29a809a79fbd03bbe39.jpg)  
(b) $\nu = 1 0 ^ { - 5 }$ , Wave forcing, Sample 136.  
Figure 24: Representative autoregressive rollouts for the 2D incompressible Navier–Stokes equations with $\nu = 1 0 ^ { - 5 }$ . Each panel shows the initial condition, the trajectory-specific forcing, the groundtruth vorticity, and predictions from the four evaluated methods at selected times. All predicted snapshots use the corresponding ground-truth color scale.