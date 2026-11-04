# Macrocanonical Generator Networks: Data-Efficient Neural Surrogates for Amortized Physics Simulation

Niall Jeffrey Dept. of Physics & King’s Institute for Artificial Intelligence King’s College London niall.jeffrey@kcl.ac.uk

Benjamin D. Wandelt Dept. of Physics and Astronomy & Dept. of Applied Mathematics and Statistics Johns Hopkins University wandelt@jhu.edu

## Abstract

Many scientific simulations are computationally expensive, limiting their use in simulation-based inference, uncertainty quantification, and decision making. We introduce macrocanonical generator networks, a framework for learning fast, dataefficient neural surrogates for amortized simulation of stationary physical processes by matching multiscale statistics of a target process. Inspired by microcanonical maximum-entropy synthesis, which matches feature statistics per sample by innerloop optimization, our approach instead trains a neural generator to satisfy the same constraints in expectation, preserving physically realistic sample-to-sample fluctuations. We prove that gradient-descent training induces preconditioned gradientdescent dynamics in sample space that inherit the symmetry-preservation properties of the microcanonical gradient descent of Bruna and Mallat, amortizing its persample inner-loop optimization into a single training run after which every sample is generated in one forward pass. We further show that residual generator parameterizations initialized near the identity, trained with a small learning rate, preserve entropy locally during early optimization and mitigate mode collapse. Experiments on cosmology and fluid simulations show that macrocanonical generators recover realistic samples from extremely small training sets, in some cases a single training example, while generating new realizations more than $1 0 ^ { 6 }$ times faster per sample than the reference simulator and more than $1 0 ^ { 3 }$ times faster than microcanonical gradient descent. The trained generators can be reused for downstream pretraining, including simulation-based inference, providing a data-efficient route to multi-fidelity scientific machine learning workflows.

## 1 Introduction

Scientific simulations are essential tools for understanding complex physical systems. However, high-fidelity simulators are often computationally expensive, sometimes requiring days to generate a single realization because each sample typically requires an iterative solve or stochastic rollout. This limits their use in tasks such as simulation-based inference [2, 7], uncertainty quantification, and decision making, where large numbers of simulations are required. Examples include cosmologica simulation [33, 17], tsunami models [3], wind farm simulation [18], and nuclear fusion models [14].

Recent advances in deep generative modelling have demonstrated that neural networks can learn complex data distributions. Methods such as generative adversarial networks [9], diffusion models [13], and normalizing flows [31] have achieved remarkable success in image generation and related domains. Earlier work such as Generative Moment Matching Networks [21] trains generators by matching the empirical data distribution using the maximum mean discrepancy (MMD). However, these methods typically rely on large training datasets and are not designed for settings in which only a small number of simulator realizations are available, a regime that is common in scientific computing.

![](images/6865634f0bd04b96d06abe1d3e800cd636699459dba9f1cd5a08ab50712a2743.jpg)  
Figure 1: Macrocanonical generator networks (centre) turn a small set of expensive simulator outputs (left; turbulence and cosmology targets, Section 5) into large ensembles of new realizations matching the chosen statistical features (e.g. wavelet phase harmonics) of the targets, unlocking downstream scientific machine-learning tasks (right) that are intractable with the expensive simulator alone.

Other moment-matching approaches, such as MMD-GAN [20], the method of simulated moments [25] and turbulence closures [28], as well as neural operators (which address the different task of predicting a simulator’s output for given boundary conditions), likewise require many simulations or observations, whereas we match a fixed multiscale moment vector (features) that a single stationary realization can be sufficient to estimate.

Many physical processes of interest—turbulent flows, cosmological fields, multifractal processes— are stationary: their statistical properties are invariant under spatial translations. This stationarity is a powerful inductive bias. As shown by Bruna and Mallat [4], the translation symmetry of a stationary process means that a single realization of dimension d provides O(d) translated copies of local structures, enabling accurate estimation of statistical moments from as few as a single example.

In such systems, knowledge about the process is naturally expressed through statistical constraints. Multiscale statistics derived from wavelet transforms, scattering transforms [1, 5], or wavelet phase harmonics [23, 35] can capture important non-Gaussian structure in complex signals. Throughout, “multiscale” refers to spatial scales rather than temporal, and all signals considered are equal-time snapshots. Classical texture synthesis methods exploit this structure by generating samples that match these statistics [29]. In these approaches, samples are synthesized individually by optimizing pixel values until the target statistics are satisfied. This procedure can be interpreted as a form of microcanonical maximum-entropy sampling [4, 35], where the probability distribution is conditioned on the empirical statistics of the observed signal. Related recent work extends moment-constrained generation with diffusion-based samplers that guide moments along stochastic trajectories toward maximum-entropy models [19]. These approaches remain per-sample iterative at generation time.

In this work we introduce macrocanonical generator networks, a neural framework that generalizes this idea. Instead of synthesizing each realization independently through optimization, we train a neural generator that produces samples whose statistics match those of the target process in expectation. The generator therefore learns a global transport map that produces samples satisfying the desired statistical constraints. Moreover, enforcing constraints in expectation rather than per sample is a better match to real physical processes, in which individual realizations fluctuate around their defining statistics (Section 2.2).

This shift from per-sample optimization to learning a global generator has important consequences. Once trained, the generator acts as a fast surrogate simulator capable of producing large numbers of realizations in one forward pass per sample, effectively amortizing the expense of iterative simulation and iterative moment-guided sampling. In this sense, our objective is complementary to trajectorybased approaches: rather than improving a per-sample iterative sampler, we learn a reusable transport map for high-throughput workflows. Because the training objective depends on statistical constraints rather than individual examples, and because stationarity provides abundant constraints from few realizations, the method can operate effectively in extremely data-limited regimes. The resulting generators can be used to pretrain neural networks for downstream tasks, enabling new approaches to simulation-based inference [2, 7], multi-fidelity modelling [12], and the training of scientific foundation models [27, 24, 34].

Beyond the practical advantages, the macrocanonical formulation provides a conceptual bridge between classical statistical synthesis methods and modern deep generative modelling. In particular, we show that training such generators induces a preconditioned gradient descent dynamics in sample space, closely related to the optimization procedure used in microcanonical synthesis [4, 35]. Two design choices are key: (i) residual generator parameterizations initialized near the identity, and (ii) sufficiently low learning rates. Together, these help the generator preserve entropy locally during early optimization and mitigate collapse to degenerate outputs. Without these inductive biases, the generator can collapse toward a constant mapping $F ( x ) \approx { \bar { y } } ^ { * }$ , effectively giving the same output for every input, which is a situation of near-minimal entropy.

Table 1: Macrocanonical generator networks (ours) compared with prevailing deep generative methods and with classical microcanonical synthesis. Properties marked “exact, by construction” follow from architectural equivariance and the loss; properties marked “approximate (learned)” hold only insofar as the training set is large and diverse enough to make them statistically apparent.
<table><tr><td></td><td>Macrocanonical (ours)</td><td>GAN</td><td>Diffusion model</td><td>Microcanonical synthesis</td></tr><tr><td>Training samples needed</td><td>≥ 1 realization</td><td>large dataset</td><td>large dataset</td><td>≥ 1 realization</td></tr><tr><td>Inference cost per sample</td><td>1 forward pass</td><td>1 forward pass</td><td>many (~ 102-3) steps</td><td>inner-loop optimization</td></tr><tr><td>Symmetries of moment constraints</td><td>exact, by construction</td><td>approximate (learned)</td><td>approximate (learned)</td><td>exact, by construction</td></tr><tr><td>Per-realization fluctuations</td><td>yes (constraints in expectation)</td><td>yes</td><td>yes</td><td>no (Dirac on φ*)</td></tr><tr><td>Training signal Distributional flexibility</td><td>feature-moment matching limited to constrained moments</td><td>adversarial unrestricted</td><td>denoising score matching unrestricted</td><td>per-sample feature matching limited to constrained moments</td></tr></table>

The main contributions of this work are therefore:

(i) We introduce macrocanonical generator networks, a generative framework that learns neural transport maps producing samples whose feature statistics match those of a target process while replacing per-sample iterative simulation or sampling with amortized one-pass generation after training.

(ii) We establish a connection between neural generator training and microcanonical synthesis, showing that generator training induces preconditioned gradient descent dynamics.

(iii) We analyze the entropy behavior of residual generators and show that identity-initialized residual parameterizations with low learning rates help to preserve local volume and maintain diversity in generated samples, while generators without these inductive biases suffer from mode collapse.

(iv) We demonstrate that macrocanonical generators can learn effective surrogate simulators in extremely data-limited regimes, including cases with a single training example, leveraging the stationarity of the target process as a key inductive bias and making many-query analyses with surrogate simulators computationally practical.

(v) We show how such generators can be used for pretraining in simulation-based inference pipelines, enabling new multi-fidelity workflows and providing a potential foundation for training scientific foundation models.

Together, these results suggest that macrocanonical generator networks provide a new framework for constructing fast neural surrogate simulators of stationary physical processes, bridging ideas from statistical physics, classical synthesis methods, and modern deep generative modelling.

Table 1 compares macrocanonical generators with the most widely used deep generative model classes and classical microcanonical synthesis. The framework trades distributional generality for three properties that are important in the data-limited scientific regime: (i) training from a handful, or even a single, realization; (ii) by-construction preservation of every symmetry respected by the moment constraints, rather than approximate symmetry learned from data; and (iii) amortized one-pass generation, like a GAN and unlike a diffusion model.

## 2 Microcanonical and Macrocanonical Generation

We first formalize the distinction between microcanonical synthesis methods and the macrocanonical generator formulation introduced in this work.

## 2.1 Microcanonical synthesis

Let ϕ(x) denote a differentiable vector of feature statistics extracted from a signal x. Given a reference signal with statistics $\phi ^ { * }$ , microcanonical synthesis generates samples by minimizing

$$
L ( x ) = \frac { 1 } { 2 } \| \phi ( x ) - \phi ^ { * } \| ^ { 2 } .\tag{1}
$$

Starting from an initialization $x ^ { ( 0 ) }$ drawn from a stationary Gaussian distribution with periodic boundary conditions [4, 35], samples are obtained by gradient descent in signal space

$$
\boldsymbol { x } ^ { ( t + 1 ) } = \boldsymbol { x } ^ { ( t ) } - \eta \nabla _ { \boldsymbol { x } } L ( \boldsymbol { x } ^ { ( t ) } ) .\tag{2}
$$

Each synthesized sample is therefore produced by an independent optimization procedure that progressively adjusts the signal until the desired feature constraints are satisfied. When the feature statistics $\phi$ are shift-equivariant (as is the case for wavelet-based statistics), the gradient descent preserves the stationarity of the initial distribution [4]. The Gaussian initialization provides maximum entropy subject to a mean and variance constraint, and convergence to the microcanonical set is guaranteed when the step size lies below the inverse Lipschitz constant of the gradient [4]. The resulting distribution is not in general maximum entropy on the microcanonical set, but inherits symmetries of the initial measure including stationarity (Theorem 3.4 and Corollary 3.5 of Bruna and Mallat 4).

## 2.2 Macrocanonical generators

The terminology reflects a physical distinction. A microcanonical formulation requires each realization to satisfy $\phi = \phi ^ { * }$ exactly, eliminating fluctuations. This is generally unphysical, because real physical systems are generated by ensembles defined only through expectation values. In the macrocanonical formulation the ensemble fixes $\phi ^ { * } = \langle \phi \rangle$ , so individual realizations fluctuate around the constraint, as observed for spectra and analogous features in quantum field configurations, cosmological fields, turbulence, and quantum many-body energy spectra.

Instead of optimizing each sample independently, we introduce a neural generator $y = F _ { \theta } ( x )$ , where $x \sim p _ { 0 }$ is drawn from a simple input distribution such as a stationary Gaussian field with periodic boundary conditions. The generator is trained so that feature constraints are satisfied in expectation:

$$
\mathbb { E } _ { { x } \sim { p } _ { 0 } } [ \phi ( F _ { \theta } ( x ) ) ] \approx { \phi } ^ { * } .\tag{3}
$$

This corresponds to a macrocanonical formulation $[ 1 6 , 6 ] \colon$ moment constraints are enforced on the ensemble of generated samples rather than on each individual sample. The trained network therefore learns a global transport map $F _ { \theta } : x \mapsto y$ that produces samples satisfying the same statistical constraints that would otherwise be enforced by per-sample optimization. Crucially, once training is complete, generating a new sample requires only a single forward pass through the network, whereas microcanonical synthesis requires a full iterative optimization for every sample. The cost of learning the transport map is thus amortized over all subsequent samples.

Why few realizations still work: The claim that $\phi ^ { * }$ can be estimated from very few realizations, or even a single realization, is a consequence of stationarity together with the shift-invariant, spatially averaged form of the feature statistics [4]. Each wavelet phase harmonic coefficient is an average over translates, $\begin{array} { r } { \phi _ { k } ( x ) = \frac { 1 } { d } \sum _ { u } f _ { k } ( \tilde { T _ { u } } \dot { x } ) } \end{array}$ , of a local functional $f _ { k }$ , where $T _ { u }$ denotes translation by u on a periodic grid of d sites. For a stationary field the variance of this average is $\begin{array} { r } { \frac { 1 } { d } \sum _ { u } \mathrm { C o v } \big ( \dot { f } _ { k } ( X ) , f _ { k } \dot { ( T _ { u } X ) } \big ) } \end{array}$ , which is $O ( 1 / d )$ if the spatial correlations of $f _ { k }$ decay sufficiently rapidly. A single realization then behaves like an average over $O ( d )$ weakly correlated copies, and n independent realizations reduce the variance to $O ( \bar { 1 } / ( n d ) )$ , so field size and the number of realizations play the same statistical role [4].

## 2.3 Residual generator parameterizations

We consider generator architectures that are initialized near the identity map [11]. In particular we study residual parameterizations of the form

$$
F _ { \theta } ( x ) = x + G _ { \theta } ( x )\tag{4}
$$

and

$$
F _ { \theta } ( x ) = G _ { \theta } ( x ) + H _ { \theta } ( x ) \odot x ,\tag{5}
$$

where $G _ { \theta }$ and $H _ { \theta }$ are convolutional networks. The choice of convolutional architecture is motivated by the analogy with microcanonical synthesis: in microcanonical gradient descent, shift-equivariant features ϕ ensure that the gradient transport preserves stationarity [4, 35]. Similarly, convolutional $G _ { \theta }$ and $H _ { \theta }$ make the generator $F _ { \theta }$ translation-equivariant, relating the macrocanonical generator to this symmetry-preserving structure. These parameterizations are initialized so that

$$
G _ { \theta _ { 0 } } ( x ) = 0 , \quad H _ { \theta _ { 0 } } ( x ) = { \bf 1 } .\tag{6}
$$

Under this initialization the generator initially acts as the identity mapping: $F _ { \theta _ { 0 } } ( x ) = x$ . If the input x is drawn from a stationary distribution, the generated samples are therefore also stationary at initialization. During early stages of optimization with a low learning rate, the learned transport remains close to the identity, which plays an important role in preserving both the entropy and the stationarity of the generated distribution, as we analyze in the following sections.

## 3 Training Dynamics of Macrocanonical Generators

We now analyze how gradient-based training of the generator affects the generated samples. Let $y = F _ { \theta } ( x )$ for $x \sim p _ { 0 }$ , and define the macrocanonical training objective

$$
L ( \theta ) = \frac { 1 } { 2 } \left\| \mathbb { E } _ { x \sim p _ { 0 } } [ \phi ( F _ { \theta } ( x ) ) ] - \phi ^ { * } \right\| ^ { 2 } .\tag{7}
$$

To understand the effect of this update in sample space, we linearize the generator around the current parameters.

Theorem 1 (Generator training induces preconditioned gradient descent). Assume (i) the generator $F _ { \theta }$ is twice continuously differentiable with respect to its parameters θ, (ii) the feature map ϕ is differentiable with respect to its input, (iii) the learning rate η is sufficiently small that the secondorder remainder below is negligible relative to the first-order term, and (iv) the required expectations and derivatives exist and differentiation may be interchanged with expectation.

Let $L ( \theta )$ denote the macrocanonical objective defined above. Write m $( \theta ) = \mathbb { E } _ { x ^ { \prime \prime } \sim p _ { 0 } } [ \phi ( F _ { \theta } ( x ^ { \prime \prime } ) ) ]$ for the meanfeature vector, so $\begin{array} { r l } { L ( \theta ) = \frac { 1 } { 2 } \| m ( \theta ) - \phi ^ { * } \| ^ { 2 } } \end{array}$ . For each input $x ^ { \prime } { } _ { \mathrm { { ; } } }$ , define a per-sample objective

$$
{ \widetilde L } _ { x ^ { \prime } } ( y ) = \big \langle \phi ( y ) , m ( \theta ^ { ( t ) } ) - \phi ^ { * } \big \rangle .\tag{8}
$$

Then, for a fixed input sample x, the corresponding change in the generated sample satisfies, to first order in the parameter update,

$$
y ^ { ( t + 1 ) } ( x ) \approx y ^ { ( t ) } ( x ) - \eta \mathbb { E } _ { x ^ { \prime } \sim p _ { 0 } } \left[ K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } ) \nabla _ { y } \widetilde { L } _ { x ^ { \prime } } \right] ,\tag{9}
$$

where $y ^ { ( t ) } ( x ) = F _ { \theta ^ { ( t ) } } ( x )$ , the approximation error is $O ( \| \Delta \theta \| ^ { 2 } ) = O ( \eta ^ { 2 } )$ by assumption (i),

$$
K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } ) = J _ { \theta ^ { ( t ) } } ( x ) J _ { \theta ^ { ( t ) } } ( x ^ { \prime } ) ^ { \top } , \qquad J _ { \theta ^ { ( t ) } } ( x ) = \frac { \partial F _ { \theta ^ { ( t ) } } ( x ) } { \partial \theta } .\tag{10}
$$

$\nabla _ { y } \widetilde { L } _ { x ^ { \prime } }$ denotes the gradient of the per-sample objective evaluated at the current output $y = F _ { \theta ^ { ( t ) } } ( x ^ { \prime } )$

$$
\nabla _ { y } \widetilde { L } _ { x ^ { \prime } } = \left( \nabla _ { y } \phi ( F _ { \theta ^ { ( t ) } } ( x ^ { \prime } ) ) \right) ^ { \top } \left( m ( \theta ^ { ( t ) } ) - \phi ^ { * } \right) .\tag{11}
$$

Thus generator training induces a preconditioned gradient descent in sample space, governed by the kernel $K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } )$ .

Kernel interpretation. The full sample-space update is coupled across inputs through the crosskernel $K _ { \theta } ( x , \mathbf { \bar { \psi } } ^ { \prime } ) = J _ { \theta } ( x ) J _ { \theta } ( x ^ { \prime } ) ^ { \top }$ . The update can therefore be written

$$
\boldsymbol { y } ^ { ( t + 1 ) } = \boldsymbol { y } ^ { ( t ) } - \eta K _ { \theta } \nabla _ { y } \widetilde { \boldsymbol { L } } ,\tag{12}
$$

where $K _ { \theta }$ acts as an integral operator on the gradient field

$$
( K _ { \theta } g ) ( x ) = \mathbb { E } _ { x ^ { \prime } \sim p _ { 0 } } [ K _ { \theta } ( x , x ^ { \prime } ) g ( x ^ { \prime } ) ] .\tag{13}
$$

Generator training therefore induces a kernel-preconditioned gradient descent in sample space. The kernel couples updates across inputs through the generator Jacobian, so that the evolution of each sample depends on the sample-space gradients evaluated at other inputs $x ^ { \prime }$ . We note that for a batch containing the single input x, the update of the output at that same input reduces to a pointwise preconditioned step with the self-kernel $K _ { \theta } ( x , x ) = \mathbf { \bar { \cal J } } _ { \theta } ( x ) \cal J _ { \theta } ( x ) ^ { \top }$ , analogous to the Gauss–Newton or neural tangent kernel preconditioner [15].

The theorem does not assume training on a single (noise) input x: the update $\Delta \theta = - \eta \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ uses the full macrocanonical objective, with the expectation over $x ^ { \prime }$ inside. The theorem gives the change in the output at arbitrary input x. Since all outputs share $\theta ,$ one update changes every sample.

Proofsketch. The argument has two ingredients. First, by assumptions (i) and (iii) the parameter update $\Delta \theta = - \eta \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ is small, so we may linearize the generator around $\boldsymbol { \theta } ^ { ( t ) }$ ,

$$
F _ { \theta ^ { ( t ) } + \Delta \theta } ( x ) \approx F _ { \theta ^ { ( t ) } } ( x ) + J _ { \theta ^ { ( t ) } } ( x ) \Delta \theta , \qquad J _ { \theta ^ { ( t ) } } ( x ) = \frac { \partial F _ { \theta ^ { ( t ) } } ( x ) } { \partial \theta } ,\tag{14}
$$

which converts the parameter update into a sample-space update $y ^ { ( t + 1 ) } ( x ) ~ \approx ~ y ^ { ( t ) } ( x ) ~ -$ $\eta J _ { \theta ^ { ( t ) } } ( x ) \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ . Second, the chain rule applied to $\begin{array} { r } { L ( \theta ) = \frac { 1 } { 2 } \| m ( \theta ) - \phi ^ { * } \| ^ { 2 } } \end{array}$ with $m ( \theta ) =$ $\mathbb { E } _ { x ^ { \prime } \sim p _ { 0 } } [ \phi ( F _ { \theta } ( x ^ { \prime } ) ) ]$ , together with the regularity assumptions (ii) and $( \mathrm { i v } )$ , expresses

$$
\nabla _ { \boldsymbol { \theta } } L ( { \boldsymbol { \theta } } ^ { ( t ) } ) = \mathbb { E } _ { { \boldsymbol { x } } ^ { \prime } \sim { \boldsymbol { p } } _ { 0 } } \big [ J _ { { \boldsymbol { \theta } } ^ { ( t ) } } ( { \boldsymbol { x } } ^ { \prime } ) ^ { \top } \nabla _ { \boldsymbol { y } } \widetilde L _ { { \boldsymbol { x } } ^ { \prime } } \big ] ,\tag{15}
$$

with $\nabla _ { y } \widetilde { L } _ { x ^ { \prime } }$ as defined in the theorem statement. Substituting this expectation into the linearized sample update and identifying $K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } ) = J _ { \theta ^ { ( t ) } } ( x ) J _ { \theta ^ { ( t ) } } ( x ^ { \prime } ) ^ { \top }$ gives the stated kernel-preconditioned form. A complete derivation, with the chain-rule manipulations written out step by step, is given in Appendix A. □

Proposition 1 (Symmetry preservation under equivariant generators). Let G be a group acting on the sample space by measurable transformations. Let $X \sim p _ { 0 }$ and define $Y = F _ { \theta } ( \check { X } )$ . Assume that: (1) p<sub>0</sub> is G-invariant, i.e. for all $g \in { \mathcal { G } } ,$ , the random variable gX has the same distribution as $X ,$ ; (2) the generator is G-equivariant, i.e. $F _ { \theta } ( g x ) = g F _ { \theta } ( x )$ for all $g \in { \mathcal { G } }$

Then Y is G-invariant: for all $g \in { \mathcal { G } }$ , the random variables gY and Y have the same distribution.

Proof. Fix $g \in { \mathcal { G } }$ . Since $p _ { 0 }$ is G-invariant: $g X { \overset { d } { = } } X$

Since $g X { \overset { d } { = } } X$ and $F _ { \theta }$ is measurable,

$$
F _ { \theta } ( g X ) \overset { d } { = } F _ { \theta } ( X ) = Y .\tag{16}
$$

By equivariance of the generator,

$$
F _ { \theta } ( g X ) = g F _ { \theta } ( X ) = g Y , { \mathrm { ~ t h e r e f o r e ~ } } g Y \stackrel { d } { = } Y .\tag{17}
$$

Since this holds for every $g \in { \mathcal { G } }$ , the random variable Y is G-invariant.

Corollary 1 (Stationarity of macrocanonical samples). Let G be the group of periodic translations on the grid. $I f p _ { 0 }$ is stationary and $F _ { \theta }$ is translation-equivariant (as enforced by a convolutional architecture with periodic boundary conditions), then $\grave { p _ { \theta } } = \mathcal { L } ( F _ { \theta } ( X ) )$ is stationary.

Proof. Take $\mathcal { G }$ to be the group of periodic translations on the grid. By assumption, $p _ { 0 }$ is invariant under these translations, and $\scriptstyle { \dot { F } } _ { \theta }$ is equivariant under the same action. Proposition 1 therefore implies that $F _ { \theta } ( X )$ is stationary. □

Corollary 2 (Symmetry inheritance and amortization of microcanonical synthesis). Assume the conditions of Theorem 1, let $\mathcal { G }$ be the group ofperiodic grid translations acting by permutation matrices $^ { g , }$ and assume ϕ is shift-invariant $( \phi ( g y ) ~ = ~ \phi ( y )$ for all $g ~ \in ~ \mathcal { G } )$ , F is translationequivariant $( F _ { \theta } ( g x ) = g F _ { \theta } ( x ) ; e . g .$ , a convolutional network with periodic boundary conditions), and $p _ { 0 }$ is stationary. Then, for every $g \in { \mathcal { G } }$ and all inputs $x , x ^ { \prime } \colon ( i )$ the per-sample gradient is equivariant, $\nabla _ { y } \widetilde { L } _ { g x ^ { \prime } } = g \nabla _ { y } \widetilde { L } _ { x ^ { \prime } } ; ( i i )$ the kernel intertwines the translation action, $K _ { \theta ^ { ( t ) } } ( g x , g x ^ { \prime } ) =$ $g K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } ) g ^ { - 1 } ,$ ; and (iii) consequently, the one-step update map $U ^ { ( t ) } : y ^ { ( t ) } ( \cdot ) \mapsto y ^ { ( t + 1 ) } ( \cdot )$ of

(a)  
(b)  
![](images/d249104787f9f66a5ed9e32fdd399d98a8a10443cb9320735b8205f4dba8aa61.jpg)

(c)  
![](images/fe6631b1821fa68a2401b0fcf387454a2a604aa97287d0786b1391af0c2e8d09.jpg)

![](images/1dac4ee17f498e6c5fb6b3e80dbfb22be1d9fa77aff8e6124ef0758e43a01ba5.jpg)  
Figure 2: Simulation-based inference of the dark-energy fraction $\Omega _ { \mathrm { D E } }$ from 2-D density images (Section 5). (a, b) A validation example before and after the observational effects (i.e. masking). (c) Posterior after fine-tuning on N “real” simulations. (d) Validation NLL against N (inc. standard deviation with 3 seeds) using cosmologies in the $\Omega _ { \mathrm { D E } }$ training range. The surrogate-pretrained model improves on the prior before any “real” simulation is used and consistently outperforms the baseline.

Theorem 1 commutes with translations, $U ^ { ( t ) } ( g \cdot ) = g U ^ { ( t ) } ( \cdot )$ , and the generated distribution of $\cdot _ { y } ( t )$ is stationary at every training step t.

Stationarity ofthe generated distribution infact holds exactly,for every parameter value, by Corollary 1; the content of (i)–(iii) is that the first-order training dynamics of Theorem 1 respect the same symmetry. Property (iii) is the same symmetry-preservation guarantee that Bruna and Mallat [4] establish for microcanonical pixel-space gradient descent (their Theorem 3.4), transferred to the training dynamics of the macrocanonical generator. After a single training run the generator amortizes the per-sample inner-loop optimization required by microcanonical synthesis [4, 35]: every subsequent sample is produced in oneforward pass while inheriting the stationarity-preservation guarantee of property (iii).

## Proof. See Appendix B.

Together, Theorem 1 and Corollary 2 provide the macrocanonical analogue of symmetry-preserving microcanonical dynamics. Theorem 1 describes the local sample-space geometry induced by training under the macrocanonical objective, in the regime of low learning rate where the generator remains well approximated by its local linearization. Proposition 1 and Corollary 1 show that equivariant generators preserve symmetries of the input distribution, including stationarity.

When $K _ { \theta } ( x , x )$ is close to the identity, the local induced dynamics resemble the pixel-space gradient descent used in microcanonical synthesis [4, 35].

This connection reveals the importance of the learning rate $\eta \colon$ the linearization underlying Theorem 1 is valid when $\eta$ is sufficiently small. This mirrors the requirement in microcanonical gradient descent that the step size be sufficiently small to ensure stable convergence. In the macrocanonical setting, a low learning rate typically ensures that each training step produces a small perturbation of the transport map, helping preserve entropy and maintain the intended stationarity (in practice, numerically) of the generated distribution.

## 4 Entropy and Volume Distortion

We now examine how the generator transformation affects the diversity and stationarity of generated samples. Let $x \sim p _ { 0 }$ denote the input distribution and $y = F _ { \theta } ( x )$ the generated samples.

For a differentiable mapping $F _ { \theta }$ , the input Jacobian $\mathcal { T } _ { \theta } ( x ) = \partial F _ { \theta } ( x ) / \partial x$ describes how local volumes are distorted by the transformation. We use $\mathcal { I }$ for the input Jacobian to distinguish it from the parameter Jacobian $\dot { J } _ { \theta } ( x ) = \partial F _ { \theta } ( x ) / \partial \theta$ of Section 3. When $\bar { F } _ { \theta }$ is a diffeomorphism, the change of variables formula relates the differential entropy of the output to that of the input:

$$
\mathcal { H } ( Y ) = \mathcal { H } ( X ) + \mathbb { E } _ { x \sim p _ { 0 } } \big [ \log \big | \operatorname* { d e t } \mathcal { I } _ { \theta } ( x ) \big | \big ] ,\tag{18}
$$

where $\mathcal { H } ( \cdot )$ denotes differential entropy. Even when $F _ { \theta }$ is not globally invertible, the log absolute determinant of the Jacobian remains a useful measure of local volume change: regions where

| det $\mathcal { I } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) | \approx 1$ are approximately volume-preserving, while $\vert \operatorname* { d e t } { \mathcal { T } } _ { \theta } ( x ) \vert  0$ indicates that the mapping is collapsing a neighborhood to a lower-dimensional set.

For residual generators $F _ { \theta } ( x ) = x + G _ { \theta } ( x )$ , the input Jacobian takes the form

$$
\mathcal { I } _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) = \boldsymbol { I } + \frac { \partial G _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) } { \partial \boldsymbol { x } } .\tag{19}
$$

Because the generator is initialized with $G _ { \theta _ { 0 } } ( x ) = 0 .$ , at initialization $\mathcal { I } _ { \theta _ { 0 } } ( x ) = I$ and therefore det $\mathcal { I } _ { \theta _ { 0 } } ( x ) = 1$ . The entropy of the generated distribution equals that of the input: $\mathcal { H } ( Y ) = \mathcal { H } ( X )$ During early training with a low learning rate, the residual branch $G _ { \theta } ( x )$ can remain small, so the Jacobian tends to stay close to the identity and the mapping approximately preserves volume in input space.

This behavior helps maintain high entropy in the generated distribution and reduces the risk of mode collapse. In contrast, a generator without these inductive biases can rapidly learn a mapping that contracts input space, driving | det $\mathcal { T } _ { \theta } ( x ) |  0$ and producing a degenerate distribution. In the extreme case, training can drive the generator toward a near-constant mapping $F _ { \theta } ( x ) \approx y ^ { * }$ for all x, in which the generator outputs essentially the same image for every input and the generated distribution has near-minimal entropy.

If the fixed output satisfies the constraints, $\phi ( y ^ { * } ) = \phi ^ { * } \left( \mathrm { i . e . , } y ^ { * } \right.$ is a microcanonical solution), the constant mapping $x \mapsto y ^ { * }$ attains zero training loss and is therefore a global minimizer. Avoiding collapse is therefore a property of the training dynamics (identity initialization, residual parameterization and a small learning rate) rather than of the loss; empirically, the trained generator matches the across-sample variance of held-out simulations (Figure 4), whereas removing these inductive biases produces collapse (Figure 3).

A further consequence concerns stationarity. As noted in Section 2.3, the convolutional structure of $G _ { \theta }$ and $H _ { \theta }$ makes the generator translation-equivariant, mirroring the role of shift-equivariant features in microcanonical gradient descent where stationarity is preserved [4, 35]. Together with the residual parameterization and low learning rate, which keep the transport close to the identity, these design choices help preserve the stationarity of the generated field throughout training. Without these inductive biases, the generator can exhibit severe collapse, reducing entropy and sample diversity even when constrained moments are matched; in practical architectures this may yield apparent or effective stationarity violations in finite samples, even though exact equivariance would preserve stationarity in theory. This failure mode is particularly damaging because stationarity is the symmetry that enables estimation of statistics from a single realization in data-limited settings. Viewed through this lens, identity initialization and low learning rates act as an implicit regularization through the training dynamics, discouraging rapid collapse while feature constraints are being enforced.

To summarize, residual parameterizations initialized near the identity therefore provide two interrelated structural biases: (i) local volume preservation, maintaining entropy and avoiding mode collapse; and (ii) gradual departure from the identity, keeping training in the small-update regime of Theorem 1, in which the induced dynamics retain the per-sample transport structure of microcanonical synthesis. Combined with the convolutional architecture, which encodes translation equivariance, these biases help maintain both the diversity and the stationarity of the generated distribution. In practice, this avoids rapid mode collapse.

We find that both of these structural biases are needed. Removing either the residual shortcut or the small learning rate produces collapse (Figure 3), with detailed sweeps in Appendix G.

## 5 Experiments

We test macrocanonical generators on the axes that matter for scientific workflows: extreme data limits, faithful per-realization variability of summary statistics, wall-clock feasibility relative to reference simulators and iterative synthesis, and cold-start amortized inference when simulation budgets are inadequate. The two periodic stationary targets are a single 128×128 magnetohydrodynamic (MHD) turbulence snapshot $( N _ { \mathrm { t r a i n } } = 1 )$ and up to 100 fiducial Quijote cosmic-web slices (Appendix C); Figure 1 shows targets and syntheses. MHD turbulence produces stationary, strongly non-Gaussian fields with structure coupled across scales. The cosmic web is the filamentary matter distribution formed by gravitational collapse, whose physical structures depend on the cosmological parameters. In both cases, the simulators are often too expensive to generate the required training set sizes.

![](images/3daa8740a337cd619dfb79ad33230dc9391ceb0754365a8dd90df43b258bb256.jpg)  
Figure 3: Inductive-bias ablation on the MHD (top) and cosmic-web (bottom) benchmarks; one target slice and one synthesis from each variant after 100 optimizer steps. The three failure modes correspond to removing the small step $( \eta = 1 0 ^ { - 2 } )$ , the residual shortcut, and (additionally) the circular-padded convolutions; zero-pad means the U-Net is no longer translation-equivariant, so Corollary 1 no longer guarantees stationarity (producing boundary-correlated structure).

Setup. We instantiate (4)–(5) with a four-stage circular-padded U-Net, so $F _ { \theta }$ is translationequivariant on the torus (Corollary 1) and begins at the identity. Features ϕ are wavelet phase harmonics [23, 35]; $\phi ^ { * }$ is the empirical mean of ϕ over the available training fields; inputs $x \sim p _ { 0 }$ are stationary Gaussian fields with power spectrum matched to the training pool, so second-order structure is correct before higher-order moments shape $F _ { \theta } .$ . We optimize with Adam at $\eta = 1 0 ^ { - 4 }$ and global gradient-norm clipping. Architecture, WPH settings and optimizer schedules are detailed in Appendices E and H.

Effect of training-set size. Trained on 1, 10 or 100 fields from a separate suite of MHD turbulence simulations, scored against held-out runs (Appendix F), the generator produces new fields rather than copies even from a single training field, and its statistics approach those of held-out simulations within about ten fields, as expected from Section 2.2. On the cosmic-web benchmark, a single training field already reproduces the held-out sample-to-sample scatter.

Comparison with generative baselines. On the same data, WGAN-GP [10], flow matching [22] and a DDPM [13] memorize the training field when only one is available, even with translation augmentation [8], and close the gap only as many more fields are added (Appendix F). Microcanonical synthesis [4, 35] removes sample-to-sample fluctuations and is significantly slower per sample.

Per-realization feature distributions. Matching $\phi ^ { * }$ in expectation does not, by itself, guarantee that each realization exhibits realistic scatter in every summary channel. Yet simulation-based inference (SBI) fits conditional densities by exploiting precisely those fluctuations: posteriors become miscalibrated if every surrogate has near-identical statistics. Training on $\phi ^ { * }$ averaged over 100 Quijote slices without supervising per-realization variance, we draw 100 synthetic fields and compare each WPH coefficient against 100 held-out simulator slices. Coefficient-wise Kolmogorov–Smirnov tests give mean p-value $\langle p \rangle = 0 . 5 2 ,$ , consistent with draws from the same marginals $( \mathrm { A } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { }$ pendix I, Figure 4), in line with Section 4. Microcanonical synthesis [4, 35] instead forces $\phi = \phi ^ { * }$ per realization, collapsing to a Dirac mass and removing real likelihood contributions from the posterior density.

Surrogate simulation speed. Cosmological inference ensembles routinely spend O(hours)– O(days) of cluster time per nonlinear N-body realization [17], and Quijote-class simulations inherit the same bottleneck. After a single offline fit, our generator emits batches of fresh $1 2 8 \times 1 2 8$ fields at O(ms) per sample on one NVIDIA A100—more than six orders of magnitude higher throughput than reference simulators at comparable fidelity, and more than $4 \times 1 0 ^ { 3 }$ times faster than iterative microcanonical gradient-descent synthesis [4, 35] (which additionally destroys per-realization fluctuations, as noted above). Once trained, the surrogate amortizes across arbitrarily many downstream queries— ensemble forecasting, uncertainty loops, iterative refinement—workflows otherwise inaccessible. Per-iteration training cost and exact wall-clock numbers are tabulated in Appendix D.

Pretraining for simulation-based inference. We infer $\Omega _ { \mathrm { D E } } = 1 - \Omega _ { m }$ from $1 2 8 \times 1 2 8$ matterdensity projections in the Quijote Latin-hypercube suite. Ω<sub>DE</sub> is the dark-energy fraction of the cosmic energy-density, which causes the accelerated expansion of the Universe, the effect of which is imprinted on the matter density structures (e.g. the morphology of filaments and halos). Posteriors on $\Omega _ { \mathrm { D E } }$ from such projected density images are a simplified version of the inference problem faced by galaxy surveys. Each cosmology requires a full N-body simulation (about 800 CPU core-hours per Quijote realization [32]), so conventional neural SBI hits an obstacle: even thirty cosmologies provide insufficient paired supervision for an image encoder without auxiliary simulators. We fit one macrocanonical generator per training cosmology from a single clean projection and synthesize a much larger labelled pool against which a small ResNet encoder coupled to a Gaussian neural density estimator can be pretrained, before fine-tuning on the available real simulations. Training the same architecture from scratch on identical simulation subsets remains worse on average with all 672 real fields than the pretrained model with none. Synthetic pretraining improves the validation NLL by 0.76 (from −0.93 to −1.69) before any genuine simulation is used for training (Figure 2). This shows that macrocanonical surrogates can unblock amortized scientific pipelines whose limiting factor is simulator throughput rather than model capacity. Appendix D gives further details.

Inductive-bias ablation. Finally we isolate the architectural biases advocated in Sections 2.3– 4: identity residual initialization so Jacobian spectra begin at volume-preserving values, learning rates mild enough for Theorem 1’s linearization, and circular convolutions enforcing translation equivariance so Corollary 1 applies on the torus. Removing each ingredient (training at $\eta = 1 0 ^ { - 2 }$ deleting residual shortcuts, or swapping zero padding and circular padding) while freezing datasets and iteration budgets reproduces the predicted failure modes: collapsed transports, rapid entropy loss, or stationary-boundary artifacts (Figure 3).

## 6 Discussion and Limitations

We have introduced macrocanonical generator networks, an amortized neural surrogate for stationary physical processes that matches multiscale feature statistics in expectation, together with the entropy and dynamics-based analysis (Sections 3–4) that identifies identity-initialized residual parameteriza tions and small learning rates as the inductive biases responsible for diversity-preserving training. On the cosmological dark-energy inference task of Section 5 (a real-world scientific task whose limiting factor is N-body simulator throughput) pretraining with the macrocanonical-generated surrogates makes simulation-based inference of dark energy from matter-density image data tractable in a regime where the available real-simulation budget alone is insufficient.

Limitation discussion:

• The method depends on the choice of feature statistics. If the chosen statistics do not capture the relevant structure of the target process, generated samples may fail to reproduce important properties.

• The framework is designed for stationary processes with periodic boundary conditions, where translation invariance provides the symmetry group that enables estimation from few realizations. For non-stationary processes the method cannot directly exploit this symmetry, and the data efficiency advantages may not apply. An interesting direction for future work is to pretrain generators on stationary processes (e.g., periodic fluid simulations) and then fine-tune on nonstationary simulations, which we have not yet tested. The periodic-boundary requirement can be relaxed, for example by appropriately masking or padding the region near the boundary (naive non-periodic application of the method instead produces the boundary artifacts in Figure 3).

• While the generator often appears empirically to produce samples consistent with maximumentropy distributions under the imposed constraints, and we provide theoretical justification for this, we cannot currently provide a general theoretical guarantee of convergence to such a distribution (similarly, neither do [4, 35]), nor of avoiding collapse (Section 4).

## Acknowledgments and Disclosure of Funding

NJ thanks Benjamin Joachimi and Lorne Whiteway for useful discussions. The authors acknowledge support from the Simons Collaboration on “Learning the Universe”. NJ acknowledges support from the King’s AI+ Academic Fellowship at King’s College London.

## References

[1] Erwan Allys, Théo Marchand, Jean-François Cardoso, Francisco Villaescusa-Navarro, Shirley Ho, and Stéphane Mallat. New interpretable statistics for large-scale structure analysis and generation. Physical Review D, 102(10):103506, 2020.

[2] Justin Alsing, Tom Charnock, Stephen Feeney, and Benjamin Wandelt. Fast likelihood-free cosmology with neural density estimators and active learning. Monthly Notices of the Royal Astronomical Society, 2019. doi: 10.1093/mnras/stz1960.

[3] Jörn Behrens and Frédéric Dias. New computational methods in tsunami science. Philosophical Transactions ofthe Royal Society A, 373(2053):20140382, 2015.

[4] Joan Bruna and Stéphane Mallat. Multiscale sparse microcanonical models. Mathematical Statistics and Learning, 1:257–315, 2018.

[5] Sihao Cheng and Brice Ménard. Weak lensing scattering transform: dark energy and neutrino mass sensitivity. Monthly Notices ofthe Royal Astronomical Society, 507(1):1012–1020, 2021.

[6] Thomas M Cover and Joy A Thomas. Elements of Information Theory. Wiley-Interscience, 2006.

[7] Kyle Cranmer, Johann Brehmer, and Gilles Louppe. The frontier of simulation-based inference. Proceedings ofthe National Academy ofSciences, 117(48):30055–30062, 2020.

[8] Alessandro Favero, Antonio Sclocchi, Francesco Cagnetta, Pascal Frossard, and Matthieu Wyart. How compositional generalization and creativity improve as diffusion models are trained. In Proceedings of the 42nd International Conference on Machine Learning, 2025.

[9] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems, pages 2672–2680, 2014.

[10] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of Wasserstein GANs. In Advances in Neural Information Processing Systems, volume 30, 2017.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[12] Yuga Hikida, Ayush Bharti, Niall Jeffrey, and François-Xavier Briol. Multilevel neural simulation-based inference. Advances in Neural Information Processing Systems, 38, 2025.

[13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.

[14] Mathias Hoppe, Ola Embreus, and Tünde Fülöp. DREAM: A fluid-kinetic framework for tokamak disruption runaway electron simulations. Computer Physics Communications, 268: 108098, 2021.

[15] Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, 2018.

[16] Edwin T Jaynes. Information theory and statistical mechanics. Physical Review, 106(4):620, 1957.

[17] Niall Jeffrey et al. Dark energy survey year 3 results: likelihood-free, simulation-based wCDM inference with neural compression of weak-lensing map statistics. Monthly Notices of the Royal Astronomical Society, 536(2):1303–1322, 2025.

[18] Andrew Kirby, François-Xavier Briol, Thomas D Dunstan, and Takafumi Nishino. Data-driven modelling of turbine wake interactions and flow resistance in large wind farms. Wind Energy, 26(9):875–1011, 2023.

[19] Etienne Lempereur, Nathanael Cuvelle-Magar, Florentin Coeurdoux, Stephane Mallat, and Eric Vanden-Eijnden. MGD: Moment guided diffusion for maximum entropy generation. arXiv preprint arXiv:2602.17211, 2026.

[20] Chun-Liang Li, Wei-Cheng Chang, Yu Cheng, Yiming Yang, and Barnabás Póczos. MMD GAN: Towards deeper understanding of moment matching network. In Advances in Neural Information Processing Systems, volume 30, 2017.

[21] Yujia Li, Kevin Swersky, and Richard Zemel. Generative moment matching networks. In International Conference on Machine Learning (ICML), 2015.

[22] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

[23] Stéphane Mallat, Sixin Zhang, and Gaspar Rochette. Phase harmonic correlations and convolutional neural networks. Information and Inference: A Journal ofthe IMA, 9(3):721–747, 2020.

[24] Michael McCabe, Payel Mukhopadhyay, Tanya Marwah, Bruno Regaldo-Saint Blancard, Francois Rozet, Cristiana Diaconu, Lucas Meyer, Kaze W. K. Wong, Hadi Sotoudeh, Alberto Bietti, et al. Walrus: A cross-domain foundation model for continuum dynamics. arXiv preprint arXiv:2511.15684, 2025.

[25] Daniel McFadden. A method of simulated moments for estimation of discrete response models without numerical integration. Econometrica, 57(5):995–1026, 1989.

[26] Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel, Fruzsina J. Agocs, Miguel Beneitez, Marsha Berger, Blakesley Burkhart, Keaton Burns, Stuart B. Dalziel, Drummond B. Fielding, Daniel Fortunato, Jared A. Goldberg, Keiya Hirashima, Yan-Fei Jiang, Rich R. Kerswell, Suryanarayana Maddu, Jonah Miller, Payel Mukhopadhyay, Stefan S. Nixon, Jeff Shen, Romain Watteaux, Bruno Régaldo-Saint Blancard, François Rozet, Liam H. Parker, Miles Cranmer, and Shirley Ho. The Well: a Large-Scale Collection of Diverse Physics Simulations for Machine Learning. arXiv e-prints, art. arXiv:2412.00568, November 2024. doi: 10.48550/arXiv.2412. 00568.

[27] Liam Parker, Francois Lanusse, Jeff Shen, Ollie Liu, Tom Hehir, Leopoldo Sarra, Lucas Meyer, Micah Bowles, Sebastian Wagner-Carena, Helen Qu, Siavash Golkar, Alberto Bietti, et al. AION-1: Omnimodal foundation model for astronomical sciences. arXiv preprint arXiv:2510.17960, 2025.

[28] Stephen B. Pope. Turbulent Flows. Cambridge University Press, 2000.

[29] Javier Portilla and Eero P Simoncelli. A parametric texture model based on joint statistics of complex wavelet coefficients. International Journal ofComputer Vision, 40(1):49–70, 2000.

[30] B. Regaldo-Saint Blancard, F. Levrier, E. Allys, E. Bellomi, and F. Boulanger. Statistical description of dust polarized emission from the diffuse interstellar medium: A rwst approach. Astronomy & Astrophysics, 642:A217, October 2020. ISSN 1432-0746. doi: 10.1051/0004-6361/ 202038044. URL http://dx.doi.org/10.1051/0004-6361/202038044.

[31] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International Conference on Machine Learning, pages 1530–1538, 2015.

[32] Francisco Villaescusa-Navarro, ChangHoon Hahn, Elena Massara, Arka Banerjee, Andrij M. Delgado, Doogesh Kodi Ramanah, Tom Charnock, Elena Giusarma, Yin Li, Erwan Allys, et al. The Quijote simulations. The Astrophysical Journal Supplement Series, 250(1):2, 2020.

[33] Francisco Villaescusa-Navarro et al. The CAMELS project: Public data release. The Astrophysical Journal Supplement Series, 265(2):54, 2023.

[34] Florian Wiesner, Matthias Wessling, and Stephen Baek. Towards a physics foundation model. arXiv preprint arXiv:2509.13805, 2025.

[35] Sixin Zhang and Stéphane Mallat. Maximum entropy models from phase harmonic covariances. Applied and Computational Harmonic Analysis, 53:199–230, 2021.

## A Proof of Theorem 1

We give the complete derivation of the kernel-preconditioned sample-space update stated in Theorem 1; a high-level argument is sketched in the main text

Proof of Theorem 1. Let

$$
m ( \theta ) = \mathbb { E } _ { { x ^ { \prime } } \sim p _ { 0 } } { \big [ } \phi ( F _ { \theta } ( x ^ { \prime } ) ) { \big ] } , \qquad L ( \theta ) = { \frac { 1 } { 2 } } \| m ( \theta ) - \phi ^ { * } \| ^ { 2 } .\tag{20}
$$

A gradient descent step gives

$$
\boldsymbol { \theta } ^ { ( t + 1 ) } = \boldsymbol { \theta } ^ { ( t ) } - \eta \nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ^ { ( t ) } ) .\tag{21}
$$

Because the learning rate is assumed to be sufficiently small, the parameter update is small, so for a fixed input x we may linearize the generator around $\theta ^ { ( t ) }$ , with remainder $O ( \| \Delta \theta \| ^ { 2 } )$ by assumption (i):

$$
F _ { \theta ^ { ( t ) } + \Delta \theta } ( x ) \approx F _ { \theta ^ { ( t ) } } ( x ) + J _ { \theta ^ { ( t ) } } ( x ) \Delta \theta ,\tag{22}
$$

where

$$
J _ { \theta ^ { ( t ) } } ( x ) = \frac { \partial F _ { \theta ^ { ( t ) } } ( x ) } { \partial \theta } .\tag{23}
$$

Writing $y ^ { ( t ) } ( x ) = F _ { \theta ^ { ( t ) } } ( x )$ and $\Delta \theta = - \eta \nabla _ { \theta } L ( \theta ^ { ( t ) } )$ , this gives

$$
y ^ { ( t + 1 ) } ( x ) \approx y ^ { ( t ) } ( x ) - \eta J _ { \theta ^ { ( t ) } } ( x ) \nabla _ { \theta } L ( \theta ^ { ( t ) } ) .\tag{24}
$$

It remains to compute $\nabla _ { \boldsymbol { \theta } } L$ . By the chain rule,

$$
\nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) = \left( \frac { \partial m ( \boldsymbol { \theta } ) } { \partial \boldsymbol { \theta } } \right) ^ { \top } \left( m ( \boldsymbol { \theta } ) - \boldsymbol { \phi } ^ { * } \right) .\tag{25}
$$

Assuming differentiation may be interchanged with expectation,

$$
\frac { \partial m ( \theta ) } { \partial \theta } = \mathbb { E } _ { x ^ { \prime } \sim p _ { 0 } } \left[ \frac { \partial \phi ( F _ { \theta } ( x ^ { \prime } ) ) } { \partial \theta } \right] .\tag{26}
$$

For each sample $x ^ { \prime } { . }$ , the chain rule gives

$$
\frac { \partial \phi ( F _ { \theta } ( x ^ { \prime } ) ) } { \partial \theta } = \nabla _ { y } \phi ( F _ { \theta } ( x ^ { \prime } ) ) J _ { \theta } ( x ^ { \prime } ) .\tag{27}
$$

Define

$$
\nabla _ { y } \widetilde { L } _ { x ^ { \prime } } = \left( \nabla _ { y } \phi ( F _ { \theta } ( x ^ { \prime } ) ) \right) ^ { \top } \left( m ( \theta ) - \phi ^ { * } \right) .\tag{28}
$$

Then

$$
\nabla _ { \boldsymbol { \theta } } L ( \boldsymbol { \theta } ) = \mathbb { E } _ { \boldsymbol { x } ^ { \prime } \sim p _ { 0 } } \left[ J _ { \boldsymbol { \theta } } ( \boldsymbol { x } ^ { \prime } ) ^ { \top } \nabla _ { \boldsymbol { y } } \widetilde L _ { \boldsymbol { x } ^ { \prime } } \right] .\tag{29}
$$

Substituting into the linearized sample update yields

$$
\begin{array} { r } { y ^ { ( t + 1 ) } ( x ) \approx y ^ { ( t ) } ( x ) - \eta \mathbb { E } _ { x ^ { \prime } \sim p _ { 0 } } \left[ J _ { \theta ^ { ( t ) } } ( x ) J _ { \theta ^ { ( t ) } } ( x ^ { \prime } ) ^ { \top } \nabla _ { y } \widetilde { L } _ { x ^ { \prime } } \right] . } \end{array}\tag{30}
$$

Defining

$$
K _ { \theta ^ { ( t ) } } ( x , x ^ { \prime } ) = J _ { \theta ^ { ( t ) } } ( x ) J _ { \theta ^ { ( t ) } } ( x ^ { \prime } ) ^ { \top }\tag{31}
$$

gives the stated result.

## B Proof of Corollary 2

Proof of Corollary 2. By assumption ϕ is shift-invariant, so for any translation g and input $z , \phi ( g z ) =$ $\phi ( z )$ , and the residual $\dot { \phi } ( F _ { \theta } ( x ^ { \prime } ) ) - \dot { \phi } ^ { * }$ is invariant under $x ^ { \prime } \mapsto g x ^ { \prime }$ . Periodic translations act by permutation matrices, so $g ^ { \top } = g ^ { - 1 }$ . Differentiating $\phi ( g y ) ~ = ~ \phi ( y ) ~$ in y and using translation equivariance of $F _ { \theta }$ gives $\nabla _ { y } \widetilde { L } _ { g x ^ { \prime } } = g \nabla _ { y } \widetilde { L } _ { x ^ { \prime } }$ . Translation equivariance of $F _ { \theta }$ also gives $J _ { \theta } ( g x ) =$ $g J _ { \theta } ( x )$ , hence $K _ { \theta } ( g x , g x ^ { \prime } ) = g K _ { \theta } ( x , x ^ { \prime } ) g ^ { - 1 }$ . The integrand of the per-sample update of Theorem 1 therefore satisfies

$$
K _ { \theta } ( g x , g x ^ { \prime } ) \nabla _ { y } \widetilde { L } _ { g x ^ { \prime } } = g \big ( K _ { \theta } ( x , x ^ { \prime } ) \nabla _ { y } \widetilde { L } _ { x ^ { \prime } } \big ) .\tag{32}
$$

Since $p _ { 0 }$ is invariant under $^ { g , }$ the change of variables $x ^ { \prime } \mapsto g x ^ { \prime }$ leaves the expectation unchanged, so the full update map $y ^ { ( t ) } ( x ) \mapsto y ^ { ( { \bar { t } } + 1 ) } ( x )$ is shift-equivariant. Stationarity of the generated distribution of $y ^ { ( t ) }$ at every training step t then follows from Proposition 1 applied to $F _ { \theta ^ { ( t ) } }$ , which is translation-equivariant for every ${ \theta } ^ { ( \overline { { t } } ) }$ □

## C Quijote N-body simulations and slice preprocessing

The cosmic-web experiments of Section 5 use 2-D slices extracted from the Quijote suite of N-body simulations [32], a publicly released ensemble of $\mathcal { O } ( 1 0 ^ { 4 } )$ dark-matter-only simulations spanning a wide range of cosmological parameters and designed to support large-scale-structure cosmology and machine-learning applications. The SBI experiment of Section 5 uses the same suite but draws cosmologies from the Quijote Latin-hypercube subset rather than the single-cosmology fiducial pool described below; its preprocessing differs only in extracting non-overlapping mean-projections of voxel thickness 4 along z to produce 32 images per cosmology.

Simulation setup. Each Quijote run evolves a set of dark-matter particles inside a periodic cubic box from a high-redshift initial condition — generated by second-order Lagrangian perturbation theory from a linear matter power spectrum — down to redshift $z = 0$ using a particle-based gravity solver. After the gravitational evolution, the particles are deposited onto a regular three-dimensional density grid using a mass-assignment scheme. For our experiments we use the fiducial cosmology, in which the cosmological parameters are held fixed at values consistent with current cosmic-microwavebackground and large-scale-structure constraints; this single-cosmology subset provides a large pool of statistically independent realizations of the matter density field, all governed by the same underlying physics.

Slice extraction. From each three-dimensional density snapshot we take a 2-D slice through the volume at z = 0, producing a square map of the projected matter density. We take 100 such slices in total, drawn from the fiducial Quijote pool, and treat them as independent realizations of the same stationary 2-D random field with periodic boundary conditions inherited from the simulation box.

Preprocessing. The raw surface-density values span several orders of magnitude due to the highly non-Gaussian, filamentary structure of the cosmic web. To compress this dynamic range we apply $x \mapsto \log _ { 1 0 } ( x + 1 )$ to the surface density of each slice. We then downsample to a $1 2 8 \times 1 2 8$ pixel grid by spatial mean-pooling. The pixel values are subsequently normalized to zero mean and unit variance using the empirical mean and standard deviation computed over the entire pool of slices, and finally rescaled by a constant factor of $1 / 8$ so that the resulting fields lie comfortably inside the dynamic range of the WPH operator. The resulting slices are stationary, periodic on the simulation grid, and exhibit the characteristic non-Gaussian filamentary structure that motivates the use of higher-order summary statistics such as wavelet phase harmonic coefficients.

## D Throughput and simulation-based inference protocol

Reference N-body cost. A single N-body simulation of the kind used in modern weak-lensing cosmology, for example the Gower Street suite produced for recent Dark Energy Survey analyses [17], typically requires O(hours to days) of compute per realization, even when distributed across hundreds of cores. The Quijote suite required more than 35 million CPU core-hours for its 44,100 simulations, about 800 core-hours per realization [32], so even 30 Latin-hypercube cosmologies constitute an aggressive pool for amortized inference.

Macrocanonical inference throughput. Once trained, the macrocanonical generator produces a batch of 200 new 128×128 realizations in one forward pass that completes in well under a second on a single NVIDIA A100 GPU—a few milliseconds per field and more than six orders of magnitude higher throughput than a single Gower-Street-class N-body realization.

Training wall-clock. Training uses two A100 GPUs at batch size 200, costing roughly one minute per 25 iterations. The simulation-based inference configurations use 500 iterations (∼20 minutes); the higher-fidelity runs for Section 5 use 5000 iterations (a few hours).

Microcanonical synthesis baseline. Microcanonical gradient-descent synthesis [4, 35] solves a fresh inner loop per sample and takes approximately 20 seconds per 128×128 field on the same A100—more than 4× $1 0 ^ { 3 }$ slower per sample than the amortized generator—while pinning $\phi = \phi ^ { * }$ exactly (Section 5), so it cannot supply the fluctuations downstream inference requires.

Amortization. One-off generator training is amortized across all later queries, making simulationbased inference, uncertainty quantification, and sensitivity analysis feasible where repeated N-body runs are not.

SBI data splits and corruption. We draw 32 non-overlapping $1 2 8 \times 1 2 8$ projections from each of the first 30 Quijote Latin-hypercube simulations, split by cosmology into 24 training cosmologies (768 training fields) and 6 validation cosmologies (192 validation fields). The same fixed instrumentfootprint corruption (heteroskedastic noise with per-pixel standard-deviation map, jagged border, centred-circle mask) is applied to every real and synthetic image. We use a small ResNet embedding $f _ { \psi }$ and a single-component Gaussian estimator $\dot { p ( } \Omega _ { \mathrm { D E } } \mid \operatorname { i m a g e } ) = N ( \mu ( f _ { \psi } ) , \sigma ( f _ { \psi } ) ^ { 2 } )$ ), with Adam at $5 \times 1 0 ^ { - 5 }$ for fine-tuning and $5 \times 1 0 ^ { - 4 }$ for from-scratch training, plus patience-based early stopping.

Pretraining and fine-tuning. We train one macrocanonical generator per training cosmology from a single $1 2 8 \times 1 2 8$ WPH target, draw 1000 synthetic fields per cosmology, yielding 24,000 synthetic training fields compared with 768 real training tiles $( 2 4 \times 3 2$ non-overlapping projections). We pretrain the encoder and Gaussian neural density estimator on synthetics, then fine-tune on real subsets of size $N \in \{ 1 , 1 0 , 1 0 0 , 3 0 0 , 6 7 2 \}$ (96 of the 768 training tiles, four projections per training cosmology, are held out for early stopping of all models). An identical architecture trained from scratch on the same real subsets is the baseline.

Likelihood results. Validation NLL is averaged over the four validation cosmologies whose $\Omega _ { \mathrm { D E } }$ lies inside the training range [0.525, 0.887] (the other two require extrapolation) and over three seeds. Training from scratch stays at the empirical prior (−0.93) up to N=10 and reaches $- 1 . 4 \pm 0 . 3$ at $N \geq 3 0 0 ;$ pretraining alone gives $- 1 . { \overset { - } { 6 } } 9 \pm 0 . 0 3$ before any genuine simulation is used for training, and fine-tuning reaches −1.76 at N=672 (Figure 2).

Synthetic ensemble cost. After ∼20 minutes of generator training per cosmology on two A100s, each network emits ∼200 fields per second. The resulting 24,000-tile synthetic training corpus would correspond to 750 additional Quijote realizations, roughly $6 \times 1 0 ^ { 5 }$ CPU core-hours, for comparably dense cosmology–image supervision.

## E Experimental setup details

This appendix gives the architecture, initialization, and optimizer details summarized in Section 5.

Code availability. A reference implementation of the macrocanonical generator, with training and sampling scripts for both benchmarks of Section 5, is available at https://github.com/ NiallJeffrey/MacrocanonicalGeneratorNetwork.

Generator architectures. We instantiate the two residual parameterizations introduced in Section 2.2. The additive form of eq. (4) uses a single convolutional U-Net for $G _ { \theta }$ . The multiplicative form of eq. (5) uses two independent U-Nets of identical architecture for $G _ { \theta }$ and $H _ { \theta }$ . Both networks share the same backbone: a four-stage encoder–decoder with channel widths $6 4  1 0 2 4 , 3 { \times } 3$ convolutions, batch normalization, and ReLU activations. Because every target process in our experiments is periodic, all convolutions use circular padding and all bilinear upsamples are wrapped in a circular pad/interpolate/crop block, so the network is exactly translation-equivariant on the periodic grid (Corollary 1). Convolutional weights are initialized from $\mathcal { N } ( 0 , 1 0 ^ { - \frac { 2 } { 4 } } )$ so that $G _ { \theta _ { 0 } } ( x ) \approx 0 ;$ for the multiplicative parameterization, an additive constant is appended to the forward map of $H _ { \theta }$ so that $H _ { \theta _ { 0 } } ( x ) \approx \mathbf { 1 }$ . Both parameterizations therefore begin training at the identity, $F _ { \theta _ { 0 } } ( x ) = x$ , as required by the analysis of Sections 3 and 4. Activation checkpointing is used in the U-Net body to reduce peak memory at the cost of an additional forward pass during the backward, allowing batch size 200 at $1 2 8 \times 1 2 8$ resolution on a single GPU.

Feature statistics. We use wavelet phase harmonic (WPH) coefficients [23, 35] as the feature map ϕ, computed under periodic boundary conditions. The hyperparameters $( J , L , \Delta \ell , \Delta n )$ and the phase-harmonic exponent list are chosen per dataset; the per-benchmark values used in this work are summarized in Appendix H, Table 6. The training target $\phi ^ { * }$ is the empirical mean of the WPH coefficients over the available training realizations; in the single-realization regime this reduces to the WPH coefficients of that one signal.

MHD benchmark data. The single 128×128 MHD turbulence snapshot of Section 5 is derived from a 512×512 map of simulated dust emission from the magnetohydrodynamic simulation of the diffuse interstellar medium used by Regaldo-Saint Blancard et al. [30]. The map is stored as a complex-valued array; we take its modulus, apply a $\log _ { 1 0 }$ transform, mean-pool to 128×128, subtract a smooth large-scale component (a Gaussian filter of width 8 pixels with periodic wrapping), standardize to zero mean and unit variance, and rescale by $1 / 1 6$

Input distribution. $p _ { 0 }$ is a stationary Gaussian random field with periodic boundary conditions, generated by FFT from the radially averaged power spectrum estimated from the training data. This matches the second-order statistics of the target process while leaving all higher-order structure to be learned by $F _ { \theta }$

Optimization. We minimize the macrocanonical loss with Adam at base learning rate $1 0 ^ { - 4 }$ and global gradient-norm clipping at 1.0, optionally preceded by a short linear warmup and followed by cosine annealing over the configured number of iterations. The generator and the WPH operator are placed on separate GPUs to accommodate their respective memory footprints; cross-device autograd is handled transparently by PyTorch.

## F MHD turbulence suite for the controlled comparisons

The training-set-size sweep and the baseline comparison of Section 5, and the sweeps of Appendix G, require many independent realizations of one stationary process: to train at $N _ { \mathrm { t r a i n } } = 1 0$ and 100, and to score every method against held-out fields that no method has seen. The single MHD snapshot of the main benchmark cannot supply these, so for these experiments we use the MHD\_64 dataset of The Well [26]: three-dimensional isothermal, compressible MHD turbulence without self-gravity, simulated at $2 5 6 ^ { 3 }$ with periodic boundaries and released at $6 4 ^ { 3 }$ after an ideal low-pass filter, with independent runs for each combination of sonic and Alfvénic Mach number. We use the single combination $\mathcal { M } _ { s } = 2 , \mathcal { M } _ { A } = 0 . 7$ , so that all fields are realizations of one stationary process, and take two-dimensional 64×64 slices of the density at fixed z. The low-pass filter can produce small non-positive density values, which we clamp to the smallest positive value before taking $\log _ { 1 0 } .$ Training fields are drawn from the first of the 8 runs in the released training file, subsampled every 8 slices in z and every 5 snapshots in time to reduce correlation; the 100 held-out evaluation fields are drawn in the same way from the remaining 7 runs. The held-out fields are therefore independent simulations, whereas the 10 or 100 training fields are correlated slices of a single run. Fields are standardized with the mean and standard deviation of the training fields only and divided by 8; the WPH configuration is $J = 4 , L = 4 , \Delta n = 1 , \Delta \ell = 2$ with the default phase-harmonic exponents. Every method in Tables 2 and 3 is trained and scored on exactly these fields.

The macrocanonical generator uses the multiplicative parameterization (5) with the optimizer settings of Appendix E, trained for 1500 iterations at batch size 64 and without augmentation. The learned baselines are trained for 20,000 iterations at batch size 32: WGAN-GP [10] with five critic steps per generator step and gradient-penalty weight 10; flow matching [22] and the DDPM [13] with the same 5.3M-parameter circular-padded U-Net, sampled with 100 Euler steps and 250 DDIM steps respectively. During DDPM sampling we clamp the predicted clean field to the range of the training data (its 0.1st to 99.9th percentiles, widened by 10%); without this safeguard the sampler diverges on these fields. Translation augmentation, where used, applies a random cyclic shift to the training field at every step. Microcanonical synthesis runs 400 steps of pixel-space gradient descent per sample.

Metrics. The WPH distance is the root-mean-square difference, over the normalized WPH coefficients, between the mean feature vector of the generated samples and that of the held-out fields. KS mean-p is the mean, over WPH coefficients, of the two-sample Kolmogorov–Smirnov p-value comparing the across-sample distribution of that coefficient for generated and held-out fields: it is near 0.5 when the per-realization distributions match and near 0 when the scatter is missing or shifted. The nearest-neighbour (NN) ratio is the root-mean-square distance from a sample to its nearest training field, minimized over cyclic translations, divided by the same quantity for held-out fields; values much less than 1 indicate memorization, including memorization of shifted copies. Numbers in parentheses are held-out fields scored against themselves, which is the level a faithfu surrogate should reach.

Table 2: Effect of training-set size (64×64 MHD suite of Appendix F, scored against 100 held-out fields; parentheses: held-out fields scored against themselves).
<table><tr><td> $N _ { \mathrm { t r a i n } }$ </td><td>WPH distance</td><td>KS mean-p</td><td>NN ratio</td></tr><tr><td>1</td><td>0.20 (0.053)</td><td>0.08 (0.54)</td><td>0.89 (1.00)</td></tr><tr><td>10</td><td>0.075 (0.056)</td><td>0.27 (0.47)</td><td>0.98 (0.96)</td></tr><tr><td>100</td><td>0.086 (0.037)</td><td>0.31 (0.50)</td><td>0.99 (1.01)</td></tr></table>

Table 3: Baseline comparison on the 64×64 MHD suite of Appendix F: all methods trained on the same fields and scored against 100 held-out fields. NN ratios much less than 1 mean memorization; held-out fields scored against themselves give 1.00, 0.96 and 1.01 at $N _ { \mathrm { t r a i n } } = 1$ , 10 and 100. Sampling time is per field on one A100.
<table><tr><td rowspan="2">Method</td><td colspan="3">NN ratio at  $N _ { \mathrm { t r a i n } }$ </td><td rowspan="2">ms/sample</td><td rowspan="2">Outcome at  $N _ { \mathrm { t r a i n } } = 1$ </td></tr><tr><td>1</td><td>10</td><td>100</td></tr><tr><td>Macrocanonical generator (ours)</td><td>0.89</td><td>0.98</td><td>0.99</td><td>0.9</td><td>new fields</td></tr><tr><td>Microcanonical synthesis</td><td>0.87</td><td></td><td></td><td>9200</td><td>statistics pinned (no fluctuations)</td></tr><tr><td>DDPM</td><td>0.32</td><td>0.58</td><td>0.86</td><td>107</td><td>memorizes training field</td></tr><tr><td>WGAN-GP</td><td>0.010</td><td>0.33</td><td>0.84</td><td>0.09</td><td>memorizes training field</td></tr><tr><td>WGAN-GP + translation aug.</td><td>0.24</td><td></td><td></td><td>0.10</td><td>memorizes shifts of it</td></tr><tr><td>Flow matching</td><td>0.054</td><td>0.24</td><td>0.66</td><td>36</td><td>memorizes training field</td></tr><tr><td>Flow matching + translation aug.</td><td>0.052</td><td></td><td></td><td>36</td><td>memorizes training field</td></tr></table>

The baselines improve as more independent fields become available, which is the expected signature of fitting the empirical data distribution: diffusion models in particular generalize only above a dataset-size threshold and reproduce their training data below it [8]. By $\breve { N } _ { \mathrm { t r a i n } } = 1 0 0$ , WGAN-GP matches the mean feature vector at least as well as our generator (WPH distance 0.058 against our 0.086; held-out 0.037), whereas the DDPM still has a WPH distance of 0.14 and a KS mean-p of 0.007 (held-out 0.50), with a per-coefficient scatter of only 0.28 of the held-out level (ours: 0.67). The advantage of the macrocanonical generator is therefore confined to the regime in which fields are scarce, which is the regime that motivates the method.

Single training field at the paper’s resolution. The $N _ { \mathrm { t r a i n } } = 1$ row of Table 2 is limited by the 64×64 field size rather than by the training budget: increasing the budget from 1500 to 5000 iterations leaves the WPH distance (0.20 in both cases) and the NN ratio (0.89 against 0.87) unchanged.

Repeating the single-field protocol on the 128×128 Quijote benchmark (one training slice, 100 generator samples scored against 100 held-out slices) gives a per-coefficient scatter ratio, the mean over WPH coefficients of the generated to the held-out across-sample standard deviation, of 1.14 (held-out slices scored against themselves: 1.05), an NN ratio of 1.00 and a pixel-marginal KS distance of 0.015 (0.014). This is the result referred to in Section 5: the widths of the per-coefficient distributions are reproduced from a single 128×128 field, although their positions are not all matched at this training-set size (WPH distance 0.052 against 0.021 for held-out slices, KS mean-p 0.15 against 0.56), as on the MHD suite.

## G Learning-rate, initialization-scale and input-distribution sweeps

Two hyperparameters govern whether training remains in the near-identity regime analyzed in Sections 3 and 4: the learning rate η and the standard deviation σ of the initial residual weights. We sweep each in turn on the 64×64 MHD suite of Appendix F $( N _ { \mathrm { t r a i n } } = 1 0 0$ , 600 iterations at batch size 32, multiplicative parameterization, all other settings as in Appendix E) and record three diagnostics on a fixed probe batch every 10 iterations. The sample diversity is the per-pixel standard deviation across the generated batch divided by that of the input batch, so a value near 1 means the generator has not contracted the input distribution. The per-step output change, $\| F _ { \theta ^ { ( t + 1 ) } } ( x ) - F _ { \theta ^ { ( t ) } } ( x ) \| / \| F _ { \theta ^ { ( t ) } } ( x ) \|$ is the quantity that the linearization of Theorem 1 requires to be small. The input-Jacobian deviation, $\lVert \mathcal { T } _ { \theta } ( x ) v - \dot { v } \rVert / \lVert v \rVert$ for a random direction v computed by finite differences, measures how far the map is from the identity and tracks the volume-preservation argument of Section 4.

Learning rate. For $\eta \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ the median per-step output change grows as 0.03, 0.085, 0.14 and 0.46 (Table 4). At the paper’s η = 10<sup>−4</sup> the input-Jacobian deviation grows gradually from 0 to 0.63 over training, and the sample diversity never falls below 0.93 and ends at 0.98, the input level. $\mathrm { A t } \eta = 1 0 ^ { - 2 }$ the first updates take the map far from the identity (input-Jacobian deviation above 300 within the first 20 iterations, diversity falling to 0.70); this is the learning rate at which Figure 3 shows collapse after 100 steps. On this small problem the run later recovers its diversity, but it ends with a larger constraint error than $\eta = 1 0 ^ { - 3 }$ . The two smallest learning rates have not reached the constraints after 600 iterations, which is why the paper’s runs use 1500 to 5000 iterations at $\eta = 1 0 ^ { - 4 }$

Table 4: Learning-rate sweep (64×64 MHD, 600 iterations, all other settings fixed). The per-step change is the median relative output change on a fixed probe batch; the linearization of Theorem 1 requires it to be small. The input-Jacobian deviation and diversity are given as the extreme value over training and, in parentheses, the value at the end of training; the loss is the mean over the last 50 iterations.
<table><tr><td>η</td><td>Median per-step change</td><td>Input-Jacobian deviation, max (final)</td><td>Diversity, min (final)</td><td>Loss</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>0.03</td><td>0.40 (0.39)</td><td>0.94 (1.02)</td><td>2.0</td></tr><tr><td> $1 0 ^ { - 4 }$  (paper)</td><td>0.085</td><td>0.65 (0.63)</td><td>0.93 (0.98)</td><td>1.2</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>0.14</td><td>0.70 (0.69)</td><td>0.83 (0.99)</td><td>0.76</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>0.46</td><td>322 (0.90)</td><td>0.70 (0.98)</td><td>0.83</td></tr></table>

Initialization scale. The paper draws the initial residual weights from ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with $\sigma = 1 0 ^ { - 4 } ;$ σ controls how far the generator starts from the identity. We sweep $\sigma \in \{ 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 3 \times$ $1 0 ^ { - 2 } , 1 0 ^ { - 1 } \}$ at $\eta = \bar { 1 } 0 ^ { - 4 }$ (Table 5). Measured on the training inputs, the initial departure from the identity $\| \ b { \dot { F } } ( \ b { x } ) - \ b { x } \| / \| \ b { x } \|$ runs from below $1 0 ^ { - 1 1 }$ to 4.5, so the sweep spans the range from the identity to a map far from it. Training reaches the constraints from all six starting points: the mean loss over the final 50 iterations ranges from 0.36 to 0.85, against a batch-to-batch scatter of 0.23–0.38 within each run, with the largest $\sigma$ the worst, and the sample diversity at the end of training lies between 0.90 and 0.96 of the input’s. What changes is the trained map itself: its input-Jacobian deviation is 0.6–0.9 for $\sigma \le 1 0 ^ { - 3 }$ but 2.1, 4.9 and 11.1 at $\sigma = 1 0 ^ { - 2 } , 3 \times \mathrm { \dot { ^ 2 } 1 0 ^ { - 2 } }$ and $1 0 ^ { - 1 }$ , so a larger initialization leaves the trained map further from a perturbation of the identity, and the diversity is lowest at the largest σ.

Table 5: Initialization-scale sweep $( 6 4 \times 6 4 ~ \mathrm { M H D }$ $\eta = 1 0 ^ { - 4 }$ , 600 iterations, all other settings fixed). The loss is the mean over the last 50 iterations; the diversity and input-Jacobian deviation are measured at the end of training.
<table><tr><td>σ</td><td> $\| F ( x ) - x \| / \| x \|$  at initialization</td><td>Input-Jacobian deviation</td><td>Diversity</td><td>Loss</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td> $< 1 0 ^ { - 1 1 }$ </td><td>0.81</td><td>0.96</td><td>0.51</td></tr><tr><td> $1 0 ^ { - 4 } \ ( \mathrm { p a p e r } )$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td><td>0.61</td><td>0.94</td><td>0.70</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>0.046</td><td>0.86</td><td>0.96</td><td>0.44</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>0.44</td><td>2.1</td><td>0.93</td><td>0.36</td></tr><tr><td> $3 \times 1 0 ^ { - 2 }$ </td><td>1.3</td><td>4.9</td><td>0.91</td><td>0.49</td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td>4.5</td><td>11.1</td><td>0.90</td><td>0.85</td></tr></table>

Relation to the ablation of Figure 3. The initialization sweep complements the ablation rather than repeating it. In the ablation the residual shortcut is removed altogether and the transport collapses; with the shortcut in place the input still reaches the output even from a poor initialization, which is why the initialization scale has a wide margin around the paper’s value. The learning rate is the setting that decides whether training stays in the regime described by Theorem 1.

Input distribution. The $\mathrm { p a p e r } \mathrm { \bf s } p _ { 0 }$ is a stationary Gaussian field with the power spectrum of the training pool (Appendix E). To test the sensitivity to this choice, we repeated the $N _ { \mathrm { t r a i n } } = 1 0 0$ training on the same MHD suite with an input that carries no power-spectrum information, white noise of the same variance, which is also stationary, so Corollary 1 still applies; each input was run twice, for 1000 and 1200 iterations. Training is stable with either input and the sample variability is unchanged: relative to held-out fields, the per-pixel diversity is 1.02 with the matched input and 1.06 with white noise (held-out halves against each other: 1.02), the mean pairwise distance between samples is 0.99 and 1.02 (1.02), and the per-coefficient scatter ratio is 0.66 and 0.67, scale by scale. The results degrade measurably, however, in the highest-order statistic we measure: the relative error in the number of peaks above 1, 2 and 3σ of the held-out pixel distribution is 1.84 and 3.56 with white noise, against 0.43 in both spectrum-matched runs. The remaining held-out statistics are less sensitive on this small problem: the power-spectrum error is smaller with white noise (0.12 against 0.18–0.19), as is the pixel-marginal KS distance (0.08–0.12 against 0.13–0.14), and the mean-feature error differs by less than the run-to-run variation (0.18 against 0.13 in one run, 0.13 against 0.17 in the other). Matching the second-order structure of the target lets the generator spend its capacity on the higher-order structure rather than on building the correlations itself, which is where the white-noise input falls short, and we expect the gap to widen for steeper spectra, larger fields or shorter training. We therefore recommend the spectrum-matched input used throughout the paper.

## H Wavelet phase harmonic summary statistics

We use wavelet phase harmonic (WPH) coefficients [23, 35] as the feature map $\phi$ in every experiment, computed by the pywph library with periodic boundary conditions $( \mathtt { p b c } \ = \ \mathtt { T r u e } )$

WPH coefficients are constructed from the wavelet transform $\{ W _ { j , \ell } f \}$ of the field f at J dyadic scales and $L$ orientations, augmented with phase harmonics $W _ { j , \ell } \dot { f } \mapsto | W _ { j , \ell } f | e ^ { i p \arg W _ { j , \ell } f }$ for an integer-valued exponent set $p ,$ and form translation-invariant second-order cross-moments across nearby scales (separated by up to $\Delta j$ octaves), orientations (∆ℓ rotations), and spatial translations $( \Delta n$ pixel shifts); see [23] for the full vocabulary. This descriptor captures phase coupling that pure power-spectrum or wavelet-scattering moments do not represent, including filamentary structure, ridges, and small-scale alignment.

Per-benchmark configuration. For both stationary benchmarks of Section 5 we use the same J = 5-level Morlet wavelet bank with $L = 4$ orientations and the pywph default scale and orientation separations $\Delta j = J - 1 = 4$ and $\Delta \ell = 2$ , computed at the native field resolution $M = 1 2 8$ . The two benchmarks differ only in the spatial-shift order $\Delta n$ used in the cross-moments and in the set of phase-harmonic exponents (Table 6).

For the cosmological cosmic web (Quijote) target we set $\Delta n = 0$ (cross-moments at the same position only) and pass an empty phase-harmonic exponent list $( { \tt p } _ { - } 1 \tt { i s t } = \emptyset )$ to pywph’s load\_model, retaining the wavelet-scattering-like subset of the WPH family. This choice is sufficient to capture the second-order non-Gaussian phase structure of the Quijote slices while keeping the coefficient vector compact.

For the MHD turbulent fluid simulation target we set $\Delta n = 1$ (adding a nearest-neighbour spatial shift to each cross-moment) and use pywph’s default phase-harmonic exponent list, yielding a richer WPH descriptor than the cosmic-web setting; we found this expanded coefficient vector to be necessary in order to match $\phi ^ { * }$ in the single-snapshot regime.

In both cases we use the library’s automatic per-coefficient normalization $( \mathrm { n o r m } ~ = ~ \mathsf { ^ { \prime } a u t o } ^ { , } )$ , which rescales each coefficient by its empirical magnitude on the first call so that the contribution of every scale/orientation/exponent triple to the macrocanonical loss is comparable. The first invocation of the operator on the training data primes this normalization cache; all subsequent calls during training reuse it.

Table 6: WPH summary-statistic configuration per benchmark; symbols match [23].
<table><tr><td>Benchmark</td><td>J</td><td>L</td><td> $\Delta j$ </td><td> $\Delta \ell$ </td><td> $\Delta n$ </td><td> $p { \mathrm { - } } { \mathrm { l i s t } }$ </td></tr><tr><td>Cosmological cosmic web (Quijote)</td><td>5</td><td>4</td><td>4</td><td>2</td><td>0</td><td>0</td></tr><tr><td>MHD turbulent fluid simulation</td><td>5</td><td>4</td><td>4</td><td>2</td><td>1</td><td>pywph default</td></tr></table>

## I Per-coefficient WPH distribution test

We back up the claim of Section 5 with a visual and a quantitative comparison. After training the macrocanonical generator on 100 Quijote cosmic-web slices against the empirical mean WPH vector $\phi ^ { * }$ alone—so the generator never sees per-realization fluctuations during training—we draw 100 generator samples and compare the across-sample distribution of every WPH coefficient to that of 100 held-out Quijote slices the network has never seen. Figure 4 shows the comparison for four randomly chosen coefficients: the generator’s per-realization distribution closely matches that of the held-out simulator pool in both mean and width, even though only the mean was supplied as a training target. A complementary quantitative test runs a two-sample Kolmogorov–Smirnov (KS) test independently for every WPH coefficient; averaged over the full coefficient vector this yields a mean p-value of $\langle p \rangle = 0 . 5 2$ , and the empirical distribution of p-values across coefficients is consistent with the uniform distribution on [0, 1] expected under the null that both samples are drawn from the same underlying distribution. This empirically confirms the entropy-preservation analysis of Section 4: scatter that is never directly supervised is nonetheless inherited from the input distribution and preserved by the residual identity-initialized generator. Microcanonical synthesis [4, 35], by construction, produces $\phi ( x _ { i } ) ~ = ~ \phi ^ { * }$ for every sample and would therefore collapse each of the panels in Figure 4 to a Dirac mass at $\phi ^ { * }$ , eliminating the per-feature variance that downstream simulation-based inference (Section 5) needs in order to learn a non-degenerate posterior.

![](images/adaab3c6938527aef7d7f7f39b42a9a352afa827a188c62b01b0f87ad91e56e0.jpg)

![](images/a5a355c6ad9d90e2132639cdba227e1b3cf9ab00344c3cf0ebbe60bad9a61acb.jpg)

![](images/31e01734188479f9345fd1a1756c73941762fc2e93747212426af0d6e819ba82.jpg)

![](images/d8988f89bfda0c0b21897da41c3afe51d3f862674f478e348535f3900b8eb416.jpg)  
Figure 4: Across-sample distributions of four randomly chosen WPH coefficients from 100 macrocanonical generator outputs (blue) and 100 held-out Quijote simulator slices (orange). Microcanonical synthesis [4, 35] would instead collapse each panel to a Dirac mass at $\phi ^ { * }$