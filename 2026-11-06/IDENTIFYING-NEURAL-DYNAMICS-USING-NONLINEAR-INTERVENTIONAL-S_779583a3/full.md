# IDENTIFYING NEURAL DYNAMICS USING NONLINEAR INTERVENTIONAL STATE SPACE MODELS

Hanna M. Tolle   
Department of Computing   
Imperial College London   
London, UK   
hmt23@ic.ac.uk

Carles Balsells-Rodas Department of Computing Imperial College London London, UK cb221@ic.ac.uk

Yixin Wang Department of Statistics University of Michigan Ann Arbor, USA yixinw@umich.edu

Pedro A. M. Mediano<sup>∗</sup> Department of Computing Imperial College London London, UK p.mediano@imperial.ac.uk

Amin Nejatbakhsh<sup>∗</sup> Center for Neural Science New York University New York, USA mn2986@nyu.edu

## ABSTRACT

State-space models (SSMs) recover low-dimensional latent dynamics from high dimensional observations, but latents are generally not identifiable from observational data alone. Interventional state-space models (iSSMs) use perturbations to break observational symmetries and identify parameters up to permutation, scaling, and shift. However, prior work restricts identifiability to linear latent dy namics and piecewise-linear emissions, excluding many behaviors characteristic of neural circuits. We extend iSSMs to a substantially broader model class: we prove that (i) affine identifiability holds under general injective emission functions and Gaussian latent transitions, and (ii) with interventions, identifiability up to permutation, scaling, and shift is preserved under general nonlinear latent dynamics. We validate our theory on simulated FitzHugh-Nagumo dynamics, and then demonstrate two applications: adjudicating between competing models of decision-making; and predicting responses to unseen micro-stimulation in macaque and human neural recordings, with proof-of-concept generalization to a held-out stimulation site in the latter. Finally, simulations yield practical guidance for designing stimulation protocols that improve identifiability.

## 1 INTRODUCTION

In biology, and especially neuroscience, experimental interventions are central to answering causal questions. Neuroscientists routinely perturb circuits using pharmacology (Bolz et al., 1989), electrophysiology (Cohen & Newsome, 2004; Histed et al., 2013), photostimulation (Fenno et al., 2011; Adesnik & Abdeladim, 2021), and more recently brain-machine interfaces (Motiwala et al., 2026).

Classical perturbation experiments establish causal involvement – e.g., by silencing a brain region and measuring behavioral deficits. Many core questions in systems neuroscience, however, require mechanistic explanations that specify not only which variables are involved, but also how they interact to generate observed dynamics. Such explanations could resolve longstanding debates about circuit motifs of working memory (Javadzadeh et al., 2024; Daie et al., 2023; Inagaki et al., 2019), e.g., or dynamical regimes underlying perceptual decision-making (Galgali et al., 2023).

A standard approach to such mechanistic questions is to fit state-space models (SSMs) to neural recordings. SSMs posit a low-dimensional latent state with explicit dynamics, and an emission map linking latents to observed activity. Without identifiability, however, distinct mechanisms can produce identical observations, so a fitted SSM cannot generally be viewed as the true causal mechanism (Qian et al., 2024; Maheswaranathan et al., 2019; Lipshutz et al., 2024; Galgali et al., 2023).

(a)  
![](images/122eb1270a46faa6a871435bc3d5aef9e7279411ef995c797f322701450c8277.jpg)  
Figure 1: Overview: ingredients for identifying latent dynamics in state-space models. (a) iSSM leverages do-interventions to identify latent dynamics up to permutation, coordinate-wise scaling, and shift. (b) Two independent failure modes prevent SSMs from identifying the true dynamics $g ^ { * }$ : identification error $\epsilon _ { 1 } ,$ resolved by interventions, and hypothesis class mismatch $\epsilon _ { 2 } ,$ resolved by a nonlinear model class. (c) Latent trajectory recovery across model class (rows) and data regime (columns): only a nonlinear model trained on interventional data recovers the true trajectory (gray).

Recent work by Nejatbakhsh & Wang (2025) addressed this gap with interventional state-space models (iSSMs), a framework for fitting SSMs to interventional data with theoretical guarantees of unique recovery up to benign symmetries. Although prior work has fitted SSMs to perturbation data, conventional SSMs remain subject to observational symmetries that permit spurious mechanisms. iSSM breaks those symmetries by fixing targeted latents to external inputs via do-interventions (Fig. 1a), severing causal dependencies from past latents and enabling identification up to permutation, coordinate-wise scaling, and shift (Ahuja et al. (2023) for a review of interventional inference).

While iSSM marked an important step, its guarantees only apply when the true system lies within the model’s hypothesis class. Its identifiability results cover linear latent dynamics with piecewiselinear emission functions – restrictive for neural circuits, which commonly exhibit nonlinear dynamics such as thresholding, bistability, and limit cycles. Specifically, we distinguish two inference failure modes (Fig. 1b): (1) identification error $\epsilon _ { 1 }$ from observational symmetries, resolved by interventional data; and (2) hypothesis class mismatch $\epsilon _ { 2 }$ from model misspecification, requiring a sufficiently expressive nonlinear model class. Fig. 1c illustrates both errors on a bistable-well system: without interventions, even a nonlinear model fails to recover transitions between attractors (ϵ ); with interventions, a linear model still misspecifies the true dynamics (ϵ ).

In this work, we generalize iSSM to nonlinear latent dynamics and general injective emission functions, addressing the principal open limitation of prior theory, and mitigating both inference failure modes – ϵ<sub>1</sub> through interventions and $\epsilon _ { 2 }$ through a nonlinear hypothesis class. We validate the framework in simulations, demonstrate applications to macaque and human neural recordings under microstimulation, and distill practical guidance for experimental design to improve identifiability.

## 2 RELATED WORK

This paper builds on the literature on identifiability in time series models. Broadly, this literature can be divided into theoretical guarantees obtained with or without interventional data.

Identifiability in time series models without interventions. One line of work establishes identifiability purely from the statistical structure of latent processes. Early nonlinear ICA results exploited changing source distributions across time segments (Hyvarinen & Morioka, 2016) or latent discrete Markov states (Hälvä & Hyvarinen, 2020; Hälvä et al., 2021), but treated latents as non-interacting; Yao et al. (2022) later allowed general nonlinear time-delayed coupling, identifying latents up to permutation and component-wise transformations when transition distributions vary sufficiently with the past. In a parallel line on switching dynamical systems, Balsells-Rodas et al. (2024) identified coupled latents with regime-switching nonlinear Gaussian transitions up to an affine map under piecewise-linear emissions, and Balsells-Rodas et al. (2026) extended this to multi-lag dependencies and refined the map to permutation, scaling, and shift when noise variances differ across regimes. Across these observational approaches, identifiability relies on non-interacting latents or transition noise that is non-Gaussian or state-/regime-dependent; for coupled dynamics with additive Gaussian noise of constant variance – the standard SSM assumption – temporal structure alone cannot resolve rotational symmetries (Yao et al., 2022; Lippe et al., 2022).

Identifiability in time series models via interventions. A second line obtains identifiability from interventions or exogenous inputs acting on latents. CITRIS (Lippe et al., 2022) and iCITRIS (Lippe et al., 2023) identify latent causal variables with general nonlinear stochastic dynamics from intervened sequences, but only their intervention-dependent parts, up to component-wise transformations, and only when intervention targets are known. Zhang & Xie (2025) identify latents up to scaling under deterministic linear or control-affine dynamics in a known controllable canonical form with sparse input matrices. Interventional state-space models (iSSMs) (Nejatbakhsh & Wang, 2025) make the target map learnable and identify full latents up to permutation, scaling, and shift, yet only for linear latent dynamics and piecewise-linear emissions inherited from Balsells-Rodas et al. (2024). The above approaches to interventional guarantees for latent dynamics therefore either require known intervention targets or restrict the dynamics to linear or control-affine forms and – in the case of iSSM – piecewise-linear emissions.

Building on this literature, we relax iSSM’s piecewise-linear emission and linear-dynamics restrictions while retaining permutation-scale-shift identifiability under hard latent interventions.

## 3 METHODS

## 3.1 GENERAL INTERVENTIONAL STATE SPACE MODELS

We consider a general class of interventional state-space models (iSSMs) that capture latent dynamical systems under interventions. Let $\pmb { x } _ { t } \in \mathbb { R } ^ { D }$ denote latent variables and $\boldsymbol { y } _ { t } \in \dot { \mathbb { R } } ^ { N }$ the observations at time t. The model is defined as:

$$
{ \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } \sim P ( { \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } \mid f _ { \theta } ( { \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } ) ) ,\tag{1}
$$

$$
\pmb { x } _ { t + 1 } = \pmb { m } _ { t } \odot g _ { \phi } ( \pmb { x } _ { t } ) + \pmb { B } \pmb { u } _ { t } + \pmb { \varepsilon } _ { t } ,\tag{2}
$$

where $f _ { \theta }$ is an injective emission function, $g _ { \phi }$ is the latent transition function, $\boldsymbol { u } _ { t } \in \mathbb { R } ^ { M }$ denotes interventional inputs, and the process noise $\varepsilon _ { t }$ has finite first moments and is independent of $( \pmb { x } _ { t } , \pmb { u } _ { t } )$ Specifically, we use Gaussian process noise $\varepsilon _ { t } \sim \mathcal { N } ( 0 , Q )$ with positive-definite covariance $Q$ The matrix $\boldsymbol { B } \in \mathbb { R } ^ { D \times M }$ maps interventions to latent variables. The inputs $\mathbf { \pmb { u } } _ { t }$ are exogenous and observed, i.e., independent of the latent states and of the process noise. The binary mask $\mathbf { \psi } _ { m _ { t } } \in$ $\{ 0 , 1 \} ^ { D }$ encodes which latent variables are not intervened on at time t, i.e., it is 1 where $( B \pmb { u } _ { t } ) _ { i } = 0$ and otherwise 0. This masking mechanism implements hard interventions in latent space, where intervened variables are disconnected from their causal parents in the past. While we use a linear input map $\mathbf { \nabla } _ { B u _ { t } }$ in all our experiments, we note that our theory extends to arbitrary deterministic input maps $h ( { \boldsymbol { \mathbf { \mathit { u } } } } _ { t } )$ , with $m _ { t , j } = \mathbb { I } \{ h _ { j } ( { \pmb u } _ { t } ) = 0 \}$ , under a mild support condition on $h \left( \mathrm { A p p . } \mathrm { A . } 5 \right)$ .

The above formulation reduces to the original iSSM when $g _ { \phi } ( \pmb { x } _ { t } ) = A \pmb { x } _ { t }$ is linear and the emission function satisfies additional structural constraints.

## 3.2 THEORETICAL RESULTS

We provide sufficient conditions for identifiability of the latent dynamics in the proposed nonlinear iSSM. The goal is to recover the latent transition and emission functions up to permutation, scaling, and shift. The main challenge is establishing this under nonlinear latent dynamics and general injective emissions.

Our approach follows the original iSSM proof and proceeds in three steps: (1) identify the distribution of noiseless latent representations from observed data; (2) identify latent variables and transition dynamics up to an affine transformation; and (3) use interventional structure to further restrict the equivalence class to permutation, scaling, and shift. Step (1) follows the observationmodel argument of prior work. We extend Steps (2) and (3) to general injective emission functions and nonlinear latent dynamics.

We begin by identifying the distribution of the noiseless latent representations from the observed data. We denote the noiseless signal as $\bar { \pmb y } _ { t } = f _ { \theta } ( { \pmb x } _ { t } )$ . Under the following standard regularity condition, this distribution can be recovered from the observed data.

Assumption 1 (Bounded completeness). The observation kernel is bounded complete: for any finite signed measure $\mu$ on $\mathbb { R } ^ { N }$

$$
\int P ( y _ { t } \in E \mid \bar { y } _ { t } = z ) d \mu ( z ) = 0 \quad \mathrm { f o r ~ e v e r y ~ m e a s u r a b l e ~ } E \subseteq \mathbb { R } ^ { N } \Rightarrow \mu = 0 .
$$

This condition is satisfied by many commonly used observation models, including Gaussian additive noise and Poisson rate observations.

Proposition 1 (Recovery of noiseless observations (Nejatbakhsh & Wang, 2025)). Under Assumption $^ { l , }$ the distribution of $\{ \bar { y } _ { t } \} _ { t = 1 } ^ { T }$ is uniquely identifiablefrom the observed distribution of $\{ y _ { t } \} _ { t = 1 } ^ { T } .$

Proposition 1 reduces identifiability to recovering latent variables from noiseless observations. This is needed because Theorem 2 requires noiseless observations $\left\{ \hat y _ { t } \right\}$ , but experimental neural data is usually noisy.

For Step (2), affine identifiability, prior work establishes this under injective piecewise-linear emissions (Nejatbakhsh & Wang, 2025). We show the same conclusion for general injective emissions under Gaussian latent transitions and mild regularity.

Theorem 2 (Affine identifiability). Consider the distribution of $\{ \bar { y } _ { t } \} _ { t = 1 } ^ { T }$ , where $\bar { \pmb y } _ { t } = f _ { \theta } ( \pmb x _ { t } )$ . Assume thefollowing conditions hold:

(A1) Observation model and regularity. The emission function $f _ { \theta }$ is injective and continuously differentiable, with continuously differentiable inverse on its image. The latent transition function $g _ { \phi }$ is continuously differentiable.

(A2) Gaussian latent transitions. The latent dynamics are Gaussian and follow a first-order Markov process:

$$
\pmb { x } _ { t } \mid \pmb { x } _ { t - 1 } \sim \mathcal { N } ( g _ { \phi } ( \pmb { x } _ { t - 1 } ) , \pmb { Q } ) , \quad \pmb { Q } \succ 0 .
$$

(A3) Local non-degeneracy of dynamics. There exists $\pmb { x } ^ { * } \in \mathbb { R } ^ { D }$ such that $J _ { g _ { \phi } } ( { \pmb x } ^ { * } )$ is full rank.

Let $( f _ { \hat { \theta } } , g _ { \hat { \phi } } , \hat { Q } )$ be another model parametrisation under assumptions $( A I ) – ( A 3 )$ that induces the same distribution on $\{ \bar { y } _ { t } \} _ { t = 1 } ^ { T }$ , where $\bar { \pmb y } _ { t } = f _ { \hat { \theta } } ( \hat { \pmb x } _ { t } )$ . Then there exists an invertible affine transformation $\psi ( { \pmb x } ) = M { \pmb x } + { \pmb x }$ such that $\hat { \pmb x } _ { t } = \psi ( { \pmb x } _ { t } )$ for all $t ,$ where $M \in \mathbb { R } ^ { D \times D }$ is invertible and $\boldsymbol { c } \in \mathbb { R } ^ { D }$

The proof of Theorem 2 is in App. A.3.

We now show that the intervention-based reduction from affine to block identifiability extends to the general nonlinear iSSM, even without assuming faithfulness as required by the original iSSM proof (Nejatbakhsh & Wang, 2025).

Theorem 3 (Block identifiability under interventions). Assume that observationally equivalent parameterizations are related by a common invertible affine map $\hat { \pmb { x } } _ { t } = \pmb { M } \pmb { x } _ { t } + \pmb { c }$ , as established, $e . g .$ by Theorem 2. Suppose additionally that the following conditions hold:

(B1) Regularity and local non-degeneracy of dynamics. The transition functions $g _ { \phi }$ and $g _ { \hat { \phi } }$ are continuously differentiable, and there exists $\pmb { x } ^ { * } \in \mathbb { R } ^ { D }$ such that $J _ { { g } _ { \phi } } ( { \pmb x } ^ { * } )$ is invertible.

(B2) Full state support across regimes. For each intervention regime, including the interventionfree regime, the distribution of the current latent state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ has support $\mathbb { R } ^ { \tilde { D } }$

Then the affine transformation $\psi ( { \pmb x } ) = M { \pmb x } + { \pmb x }$ is block-diagonal up to permutation, i.e., each estimated latent coordinate depends only on a subset of true latent coordinates corresponding to an intervention block.

Assumption 2 (Unordered pairs condition). A set of interventions satisfies the unordered pairs condition on the latents ${ \mathrm { i f } } ,$ for any unordered pair of latent coordinates $( i , j )$ with $i \neq j ,$ , there exists an intervention environment in which exactly one of the two coordinates is intervened on.

Corollary 4 (Permutation, scaling, and shift identifiability). Under the conditions of Theorem 3, assume further that the unordered pairs condition holds. Then the affine transformation $\psi ( { \pmb x } ) =$ $M x + c$ reduces to a permutation, scaling, and shift of the latent coordinates. That is, M is a scaled permutation matrix, and the latent variables are identifiable up to permutation, scaling, and shift.

Permutation-scale-shift identifiability is desirable because it enables prediction of intervention combinations never seen in training. Additionally, Assumption 2 implies that experimenters can partially control latent disentanglement by adding intervention channels, which increases the chances of latent pairs for which the unordered-pairs condition holds.

## 3.3 EVALUATION METRICS

Reconstruction accuracy. The encoder infers latents from the full observed sequence and decodes them to observations. Reconstruction accuracy is the Pearson correlation between predicted and true observations on held-out trials. Prior work argues that reconstruction-based metrics poorly reflect capture of underlying dynamics (Versteeg et al., 2025; Lampinen et al., 2026; Sedler et al., 2023); we therefore report two additional metrics.

Rollout accuracy. We take only the initial latent state $\scriptstyle { \mathbf { { \mathit { x } } } } _ { 0 }$ from the encoder and roll out the learned dynamics open-loop without further observations. On neural recordings, the encoder sees only prestimulation observations when inferring $\pmb { x } _ { 0 } \left( \mathrm { A p p . } \mathrm { G . } 4 \right)$ . Rollout accuracy is the Pearson correlation between predicted and true observations over the horizon on held-out trials.

Latent accuracy. When ground-truth latents are available, we apply the same open-loop rollout in latent space. Since iSSM identifies latents only up to permutation, coordinate-wise scaling, and shift (Corollary 4), we align inferred latents to ground truth before evaluation. We use the Hungarian algorithm to find the optimal permutation matching columns of inferred and true B matrices, then fit per-dimension affine transforms (scaling and shift) from true to inferred latents by ordinary least squares on test trajectories. Latent accuracy is the Pearson correlation between aligned predicted and true latent trajectories on held-out trials.

## 4 RESULTS

We first validate our theoretical results on simulated FitzHugh-Nagumo dynamics with known ground truth (Sec. 4.1). We then demonstrate the framework’s practical value across increasingly realistic use cases: adjudicating between competing models of decision-making under idealized conditions (Sec. 4.2); and predicting responses to unseen micro-stimulation in macaque dlPFC and human intracranial recordings, with proof-of-concept generalization to a held-out stimulation site in the human data (Secs. 4.3–4.4). Finally, we return to controlled simulations to ask how future experiments can improve identifiability through the design of their stimulation protocols (Sec. 4.5).

## 4.1 IDENTIFYING FITZHUGH-NAGUMO DYNAMICS

The FitzHugh-Nagumo (FHN) system is a canonical nonlinear model of neuronal excitability with thresholding and stable limit-cycle dynamics (FitzHugh, 1961; ichi Nagumo et al., 1962). Because these dynamics lie outside the linear iSSM hypothesis class, FHN provides a suitable test of whether nonlinear iSSM can identify nonlinear latent dynamics from observations.

We simulated FHN dynamics

$$
\frac { d v } { d t } = v - \frac { v ^ { 3 } } { 3 } - w + I _ { \mathrm { e x t } } , \qquad \tau \frac { d w } { d t } = v + a - b w ,\tag{3}
$$

with $a = 0 . 7 , b = 0 . 8 , \tau = 1 2 . 5 ,$ and $I _ { \mathrm { e x t } } = 0 . 5 ( D = 2$ latent dimensions). Latents were mapped to $N = 4$ observations through a fixed random injective projection and pointwise tanh nonlinearity. Training used $K = 2 0 0$ trials of length $T = 5 0 \mathrm { s } \mathrm { a t } \Delta t = 0 . 1 \mathrm { s } .$ , with hard interventions arriving as a Poisson process at rate $\lambda = 0 . 2 { \mathrm { s } } ^ { - 1 }$ and amplitudes in $[ - 2 , 2 ]$ . We trained 10 seeds per model class and evaluated on 100 held-out trajectories with novel interventions; full details are in App. D.

(a)  
![](images/ac88cfcb9f96e9c4900c07f441c6d0d57973f659d0b41ea67715898f7e95db74.jpg)

![](images/aa9b14e0a5c24ccbd3e08e666ae178fe24d23cbef1317660f9cf6f82c6e84a5a.jpg)  
(d)

![](images/384d93701187df4d6537d5b1d75a1b92b6d6bd3ea662159cd514076c30b93bfc.jpg)

(e)  
![](images/81bf25f689d22d32526a22ee3cbfbfda3bc0922e4f3eaf00a7648d837b00a181.jpg)

![](images/021b4b64b7796e6c9f805f7258637cf3714ecc56a84ed32366543b8e9fded47e.jpg)  
Figure 2: Nonlinear iSSM identifies FitzHugh-Nagumo latent dynamics. (a,b) Open-loop rollout and aligned latent trajectories on a test trial: true (black), linear iSSM (blue), nonlinear iSSM (magenta); $\mathrm { m e a n } \pm \mathrm { s . d }$ . across 10 seeds. Gray lines mark interventions. (c) Latent flow fields from the seed with highest Jacobian similarity to ground truth. Gray/black: true flow (arrows) and trajectory (line); blue/magenta: inferred linear/nonlinear flow and trajectories. (d) Median inferred input map B after permutation-scale alignment. (e) Latent-dimension sweep over $D \in \{ 1 , 2 , 3 , 4 \}$ shows that model selection picks out the true dimension $( D = 2 )$

The nonlinear iSSM substantially outperforms the linear iSSM on both rollout and latent accuracy (Fig. 2a,b; both $p = 0 . 0 0 3 9 )$ . Consistent with this performance gap, nonlinear models recover the characteristic relaxation-type limit cycle whereas linear models produce an elliptic spiral attractor (Fig. 2c); both recover the input map B (Fig. 2d). Finally, a latent-dimension sweep over $D \in$ $\{ 1 , 2 , 3 , 4 \}$ selects the true dimension $( D = 2 ; \operatorname { F i g } . 2 { \mathrm { e } } )$ . We replicate this analysis on the Van der Pol oscillator in App. E. Overall, these results support Corollary 4: nonlinear iSSM identifies the latent dynamics up to permutation, scaling, and shift, while linear iSSM cannot because the true system lies outside its hypothesis class.

## 4.2 IDENTIFYING LATENT DYNAMICS OF DECISION-MAKING MODELS

We next consider a scientific use case under idealized conditions: adjudicating between three competing models of decision making. These posit (i) input-driven stable decay toward a point attractor, (ii) perfect integration along a line attractor, or (iii) autonomous nonlinear integration in a bistable well. Despite their distinct recurrent mechanisms, these systems can generate observationally indistinguishable trajectories (Galgali et al., 2023, Fig. 1).

We simulated the three systems (D = 2),

$$
{ \dot { \pmb x } } = \mathrm { d i a g } ( a _ { 1 } , a _ { 2 } ) { \pmb x } \qquad \mathrm { ( p o i n t ~ a t t r a c t o r ) } ,\tag{4}
$$

$$
{ \dot { \pmb { x } } } = { \pmb { A } } { \pmb { x } } \qquad \mathrm { ( l i n e ~ a t t r a c t o r ) } ,\tag{5}
$$

$$
\dot { x } _ { 1 } = a x _ { 1 } ^ { 3 } + b x _ { 1 } + c _ { \mathrm { o h } } , \qquad \dot { x } _ { 2 } = c x _ { 2 } + c _ { \mathrm { o h } }
$$

$$
( { \mathrm { b i s t a b l e ~ w e l l } } ) ,\tag{6}
$$

with $( a _ { 1 } , a _ { 2 } ) = ( - 0 . 5 , - 1 . 0 ) , A = \mathrm { R o t } ( \gamma ) { R } { \Lambda } R ^ { - 1 } ( \lambda _ { 1 } = - 1 . 0 , \gamma = 4 5 ^ { \circ } )$ , and $( a , b , c , c _ { \mathrm { o h } } ) =$ $( - 0 . 6 , 2 . 0 , - 1 . 0 , 0 . 1 )$ . Following the protocol in Sec. 4.1, latents were mapped to $N = 1 0$ observations; training used $K = 4 0$ trials of length $T = 2 0 \mathrm { s }$ at $\Delta t = 0 . 1 \mathrm { s }$ , with hard interventions at rate $\lambda = \mathrm { { 0 } } . 7 5 \mathrm { { s ^ { - 1 } } }$ and amplitudes in $[ - 3 , 3 ]$ on $M = 2$ input channels. We trained 10 seeds per model class and evaluated on 100 held-out trajectories; full details are in App. F.

![](images/6fe8374ff78f3ae7cb5b0875837fb9ee3332b74e6fa97de6fc61ad9549e674a4.jpg)  
Figure 3: Nonlinear iSSM identifies a broader class of decision-making dynamics. Latent flow fields and trajectories for linear (left, blue) and nonlinear (right, magenta) iSSMs trained on interventional data from point-, line-, and bistable-well systems. Ground truth: gray arrows and black lines. Panels show best seed by Jacobian similarity; check marks indicate correct qualitative recovery.

As expected, the linear iSSM recovers the point- and line-attractor dynamics, which lie within its hypothesis class (Fig. 3, left). However for the bistable-well system, its learned flow collapses the two wells into a single fixed point and fails to reproduce trajectories on both sides of the separatrix. In contrast, the nonlinear iSSM recovers the flow fields and trajectories of all three systems (Fig. 3, right); and on the bistable system, it achieves significantly higher latent accuracy $( p = 0 . 0 0 4 4 )$

Interventional data can disambiguate competing decision models, but only within the assumed dynamics class. Nonlinear iSSM extends that class to include bistable attractor dynamics that linear iSSM inherently cannot represent. We next show that this also matters for real neural recordings.

## 4.3 PREDICTING MACAQUE DLPFC RESPONSES TO MICRO-STIMULATION

We next show how iSSM can be used to predict neural responses to unseen micro-stimulation of known intervention sites. We applied linear and nonlinear iSSMs to electrophysiological recordings from macaque dlPFC (Nejatbakhsh et al., 2023), pooling data from 14 sessions of the same monkey comprising 686 control and 685 interventional trials across $N = 9 6$ electrodes and $M = 1 2$ stimulation sites $( D = 1 2 ; \mathrm { F i g } . 4 \mathrm { a } )$ . Training details are in App. G.

We separate evaluation by trial type – control versus stimulation – to distinguish modelling of spontaneous dynamics from generalisation to unseen interventions. The nonlinear iSSM significantly outperforms the linear iSSM on reconstruction accuracy on both control and stimulation trials (Fig. 4c). On rollout accuracy, the largest gap appears under stimulation (Fig. 4d).

(a)  
![](images/578ca1b764c8ba38529c3e42cc3a0f2ee595921663e6424360ace9fdb9b3ca71.jpg)

(b)  
![](images/11455ee0ba0be66e1bc533c142898c1a1a420e14dc33bdf68bc09e0924cc757d.jpg)

(c)  
![](images/5f05db567b600bf86e0a7fef25925dfa8db39e7d06819d53a92d00cc33841d1b.jpg)

(d)  
![](images/8e9a8b832dfe01e82aba3c3cdb6a69bef8f9822d5c069f855ada9f701b715c92.jpg)  
Figure 4: Nonlinear iSSM outperforms linear iSSM on macaque prefrontal cortex recordings. (a,b) Example reconstructions and rollouts: observations (black), linear iSSM (blue), nonlinear iSSM (magenta); mean $\pm \ \mathrm { s . d . }$ across seeds. Gray lines mark interventions. (c,d) Reconstruction and rollout accuracy on held-out control and stimulation trials. Asterisks: $p < 0 . 0 5$

These results suggest that dlPFC dynamics lie outside the linear iSSM hypothesis class. The rollout gap under stimulation, but not control, implies this matters most for predicting evoked responses.

## 4.4 PREDICTING HUMAN IEEG RESPONSES TO SINGLE-PULSE STIMULATION

We apply linear and nonlinear iSSMs to intracranial EEG from one epilepsy patient recorded during single-pulse electrical stimulation (Huang et al., 2023) $( N = 9 9$ channels, $M = 5 0$ sites, $K = 4 2 3$ trials, $\bar { D } = 2 5 ; \mathrm { F i g . } 5 \mathrm { a } )$ , extending the macaque analysis to a second species and recording modality. Test trials fall into two groups: 20 from sites seen during training; and 10 at a site, clinically labelled irritative zone, fully withheld from training. To enable predicting stimulation response at this site, never seen during training, we map each site’s input column from electrode coordinates via a siteembedding network (App. H).

(a)  
![](images/6b4dfc0c2e0f95b3994ff6fd188d782fb3bf9bcd825c9b12cc8b81e196ef510b.jpg)

(b)  
![](images/69d4f4a4793765c80bf55077b4679d1ee82cc694525df308b74c3e09c5078d94.jpg)

(c)  
![](images/5b17a0ad8d0bc6f6716467a9b621d6e3cd926e9d6d27346375320e1518bdcf4d.jpg)

![](images/b96d14f93281cce3a7d4decca23f6e67f38b1a9cc115e0736fba2183cc359005.jpg)  
Figure 5: Nonlinear iSSM predicts human iEEG stimulation responses more accurately, including at a held-out site. (a) Axial view: recording contacts, stimulation sites, annotated epileptogenic contacts, and the held-out site. (b) Rollouts for the held-out site trial: observations (black), linear iSSM (blue), nonlinear iSSM (magenta); mean $\pm \ : \mathrm { s . d }$ . across seeds. Grey line: stimulation pulse. (c,d) Reconstruction and rollout accuracy on held-in and held-out site trials. Asterisks: $p < 0 . 0 5$

On held-out trials from trained stimulation sites, both models reconstruct equally well, but the nonlinear iSSM predicts rollouts significantly better $( p \ = \ 0 . 0 0 0 4 ; \ \mathrm { F i g } . \ 5 \mathrm { b - d } )$ . At the held-out site, the nonlinear iSSM reaches rollout accuracy $0 . 7 1 \pm 0 . 0 2$ versus $0 . 3 7 \pm 0 . 0 2$ for the linear iSSM $( p = 0 . 0 0 2 ;$ mean $\pm \ : \mathrm { s . e . m }$ . across held-out trials), with a small reconstruction advantage.

Note that our theory constrains the input map only on observed stimulation sites (Remark 1); thus, the held-out-site evaluation tests coordinate-based interpolation and is included here only as a proof of concept. What makes this empirical generalization test possible is that the held-out site lies 3.5 mm from the nearest held-in site, and recorded responses provide ground truth for validation.

## 4.5 DESIGNING INTERVENTIONS AND EVALUATING LATENT RECOVERY

Our theory offers practical guidance on how future stimulation experiments can be designed to maximize identifiability: the theorems are population statements requiring the observation law to be matched over the latent support, so identification holds only where training trajectories visit. Thus, latent accuracy should improve with intervention number and diversity. We test this on the bistable well system (details in App. I), and replicate on two other systems (App. I.2).

First, we sweep the Poisson stimulation rate over $\lambda \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 \} \mathrm { s } ^ { - 1 }$ , generating 0 to ∼600 interventions for training (Fig. 6a). Training nonlinear iSSM on each condition shows latent accuracy increases monotonically with the number of interventions from near chance to near ceiling. Next, we compare stimulation protocols, crossing (constant or diverse) amplitude with (regular or Poisson) timing at a matched total number of pulses. Amplitude diversity has the larger main effect on latent accuracy (Cohen’s $d = 1 . 8 6 , p \stackrel { - } { = } 5 . 7 \times 1 0 ^ { - 6 } )$ , although irregular timing also helps $( d = 0 . 7 1 , p = 1 . 4 \times \mathrm { \dot { 1 } 0 ^ { - 5 } }$ ; Fig. 6b). The fixed-amplitude, regular-timing protocol performs worst.

As ground-truth latents are typically unavailable in real data, we ask which observable metric best reflects latent recovery. Pooling all seeds and conditions from Fig. 6a–b, rollout accuracy tracks latent accuracy closely (points scatter around the identity line), whereas the more commonly used reconstruction accuracy stays near ceiling even when latent recovery fails (Fig. 6c).

(a)  
![](images/34041d91380497a1de0a9753d96d0bfc2f6db524490a08e630811746ff054b6a.jpg)

(b)  
![](images/24240d6e18d00c5087607d9aee24bbc3285bdcb5cd4600202a3af9573fd3a34a.jpg)

(c)  
![](images/5991e4d5e2cb66911e5f83548ed61a189338112a4c145224a633d4a2746619e6.jpg)  
Figure 6: Numerous and diverse interventions improve latent recovery. (a) Latent accuracy improves with more training interventions. (b) Diverse amplitudes and irregular timing yield higher latent accuracy; boxes: 10 seeds, $^ { * * * } p \ < \ 0 . 0 0 1$ main effect. (c) Latent accuracy versus rollout (black) and reconstruction (gray) accuracy; gray dashed line: identity.

## 5 DISCUSSION

We generalize iSSM identifiability guarantees to injective emissions and nonlinear latent dynamics. We demonstrate two applications – adjudicating between decision-making models, and predicting stimulus response in real neural recordings – which speak to central neuroscience objectives.

Implications for neuroscience. A core objective in systems neuroscience is to identify dynamic regimes of cognition: e.g., prior work used dynamical inference on neural recordings to characterize evidence accumulation for decision-making (Latimer et al., 2015; Luo et al., 2025). Yet, conclusions remain contested (Zoltowski et al., 2019; Steinemann et al., 2024); in part because equally good fits can imply different dynamics, which led Genkin & Engel (2020) to propose an empirical model selection approach that prioritizes consistency of inferred dynamics. Our framework offers a complementary route that combines formal identifiability guarantees with considerable expressiveness.

A second objective, with direct clinical relevance, is predicting neural response to stimulation – e.g., to localize seizure onset zones in epilepsy or to optimize deep brain stimulation targets. Wholebrain biophysical models pursue this via virtual stimulations (Hashemi et al., 2023; Deco & Kringelbach, 2014). However, they are strongly hypothesis-driven, commonly fitting only a few parameters to data (Deco et al., 2008; 2018), and current state-of-the-art models achieve comparatively poor fits (Herzog et al., 2024). iSSM offers a data-driven alternative, and we show that it predicts stimulus response in both macaque and human intracranial recordings, and enables response prediction at unseen stimulation sites. Notably, an exploratory analysis suggests that virtual stimulations, including at several untrained sites, may discriminate healthy from epileptogenic tissue (App. H.4); though this extends beyond our identifiability guarantees (see Limitations).

Limitations. Theorem 3 assumes hard interventions. If microstimulation is not perfectly hard, latent and rollout accuracy may degrade due to misspecification. Nevertheless, toy-model simulations show partial robustness as interventions soften (App. J; Fig. 12); and our theory requires only stochastically hard interventions rather than a deterministic clamp to a known value. Established techniques such as dynamic clamps (Prinz et al., 2004) or optogenetic silencing (Wiegert et al., 2017) plausibly operate within this regime. Mapping experimental techniques onto hard/soft regimes is an important avenue for future work.

For stimulation-response prediction, identifiability holds only over the observed support of latent inputs (Remark 1). Predicting response at unseen sites is therefore extrapolation that our theory does not guarantee. Additionally, we analysed data from only one patient; and the 99 shaft contacts are unlikely to sufficiently sample the full hemisphere. Reliable prediction at distant sites likely requires denser coverage and validation across multiple patients.

Stimulation artefacts, which corrupt signal at stimulated channels, pose a further practical challenge. In the human intracranial recordings, we mitigate this by interpolating and masking stimulated recording sites (App. H.1), and we extend our identifiability theory to cover fitting under such observation masking (App. A.6).

Progressing towards improved identifiability in real data. Accurate identification of something as complex as neural dynamics needs adequate data. Typical interventional datasets, however, include only a few channels, fixed amplitudes, regularly timed pulses, and limited repetitions. Our toy-model simulations clearly show that more diverse and numerous stimulations substantially improve latent recovery. We hope that these results motivate richer interventional protocols to help advance towards identifying causal mechanisms of neural dynamics.

## AI USE STATEMENT

In this work, we used generative AI tools (Claude Code and ChatGPT) to assist in the writing of proofs (with presentation and proof checking; see below), provide feedback on research methodology and experiments, interpret results, and via coding assistance: implement methods, generate synthetic data, clean and reformat datasets. We have not used generative AI tools to formulate mathematical claims, provide critical ingredients for proving mathematical claims, help develop theoretical models or conceptual frameworks, or propose or refine hypotheses, and translation and qualitative and thematic data analysis are not applicable to this work. Additionally, we used generative AI tools to create or edit software code, summarize existing literature, draft parts of the paper (such as this section) and edit it for readability. We have reviewed all AI-assisted work. All theoretical results and proofs were developed by the authors; AI was used to discuss proofs in related work, check our proofs for errors, and improve their clarity in the writeup. Experiments and methods were designed by the authors; AI gave feedback and assisted with coding, including implementing the preprocessing pipeline for the human iEEG data according to the pipeline published by the original authors (Huang et al., 2023). All AI-assisted code was reviewed and tested by the authors. Results were interpreted by the authors, with AI used for discussion. AI drafted the ethics, reproducibility and AI use statements; the main text was written by the authors, with AI used for polishing, streamlining and typo elimination. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work uses only previously published neural recordings; no new data were collected. The human iEEG data are the publicly available, de-identified OpenNeuro dataset ds004457 (Huang et al., 2023), recorded under Mayo Clinic Institutional Review Board approval with written informed consent from each patient, as reported by the original authors. The macaque recordings were previously described by Nejatbakhsh et al. (2023). The authors declare no conflicts of interest.

## REPRODUCIBILITY STATEMENT

Code for all experiments is available at https://anonymous.4open.science/r/ issm-code-E235/. Assumptions for our identifiability results are stated in Sec. 3.2, with complete proofs in App. A. For each experiment, the appendix details data generation or preprocessing, model architectures, training procedures and computational resources (App. D–I); statistical tests are described in App. C. The human iEEG data are publicly available on OpenNeuro (ds004457).

## AUTHOR CONTRIBUTIONS

Conceptualization, H.M.T. and A.N.; Data curation, A.N.; Formal analysis, H.M.T.; Investigation, H.M.T.; Methodology, H.M.T. and C.B.-R. and Y.W.; Software, H.M.T. and A.N.; Supervision, P.A.M.M. and A.N.; Validation, C.B.-R. and Y.W. and P.A.M.M.; Visualization, H.M.T. and C.B.- R.; Writing – original draft, H.M.T. and A.N.; Writing – review & editing, H.M.T. and C.B.-R. and Y.W. and P.A.M.M. and A.N.

## ACKNOWLEDGMENTS

H.M.T. is supported by the Doctoral Teaching Scholarship of the Department of Computing, Imperial College London. Y.W is supported in part by funding from the Office of Naval Research under grant N00014-23-1-2590, the Department of Energy under grant DE-SC0026902, the National Science Foundation under grant DMS-2310831, DMS-2428059, IIS-2435696, IIS-2440954, Two Sigma Investments LP, Science of Trustworthy AI program at Schmidt Sciences, and LG Management Development Institute AI Research.

## REFERENCES

Hillel Adesnik and Lamiae Abdeladim. Probing neural codes with two-photon holographic optogenetics. Nature neuroscience, 24(10):1356–1366, 2021.

Kartik Ahuja, Divyat Mahajan, Yixin Wang, and Yoshua Bengio. Interventional causal representation learning. In International conference on machine learning, pp. 372–407. PMLR, 2023.

Carles Balsells-Rodas, Yixin Wang, and Yingzhen Li. On the identifiability of switching dynamical systems. In International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/ 2305.15925.

Carles Balsells-Rodas, Toshiko Matsui, Pedro AM Mediano, Yixin Wang, and Yingzhen Li. On the identifiability of regime-switching models with multi-lag dependencies. arXiv preprint arXiv:2601.03325, 2026.

Jürgen Bolz, Charles D Gilbert, and Torsten N Wiesel. Pharmacological analysis of cortical circuitry. Trends in Neurosciences, 12(8):292–296, 1989.

Marlene R Cohen and William T Newsome. What electrical microstimulation has revealed about the neural basis of cognition. Current opinion in neurobiology, 14(2):169–177, 2004.

Kayvon Daie, Lorenzo Fontolan, Shaul Druckmann, and Karel Svoboda. Feedforward amplification in recurrent networks underlies paradoxical neural coding. bioRxiv, 2023. URL https://doi.org/10.1101/ 2023.08.04.552026.

Gustavo Deco and Morten L. Kringelbach. Great expectations: Using whole-brain computational connectomics for understanding neuropsychiatric disorders. Neuron, 84:892–905, 2014. URL https://doi.org/10. 1016/j.neuron.2014.08.034.

Gustavo Deco, Viktor Jirsa, Peter A. Robinson, Michael Breakspear, and Karl J. Friston. The dynamic brain: From spiking neurons to neural masses and cortical fields. PLoS Computational Biology, 4, 2008. URL https://doi.org/10.1371/journal.pcbi.1000092.

Gustavo Deco, Josephine Cruzat, Joana Cabral, Gitte Moos Knudsen, Robin L. Carhart-Harris, Peter C. Whybrow, Nikos K. Logothetis, and Morten L. Kringelbach. Whole-brain multimodal neuroimaging model using serotonin receptor maps explains non-linear functional effects of lsd. Current biology : CB, 28 19: 3065–3074.e6, 2018. URL https://doi.org/10.1016/j.cub.2018.07.083.

Lief Fenno, Ofer Yizhar, and Karl Deisseroth. The development and application of optogenetics. Annual review of neuroscience, 34:389–412, 2011. URL https://doi.org/10.1146/ annurev-neuro-061010-113817.

Richard FitzHugh. Impulses and physiological states in theoretical models of nerve membrane. Biophysical journal, 1:445–66, 1961. URL https://doi.org/10.1016/S0006-3495%2861%2986902-6.

Aniruddh R Galgali, Maneesh Sahani, and Valerio Mante. Residual dynamics resolves recurrent contributions to neural computation. Nature Neuroscience, 26(2):326–338, 2023.

Mikhail Genkin and Tatiana A. Engel. Moving beyond generalization to accurate interpretation of flexible models. Nature machine intelligence, 2:674–683, 2020. URL https://doi.org/10.1038/ s42256-020-00242-6.

Hermanni Hälvä and Aapo Hyvarinen. Hidden markov nonlinear ica: Unsupervised learning from nonstationary time series. In Conference on Uncertainty in Artificial Intelligence, pp. 939–948. PMLR, 2020.

Hermanni Hälvä, Sylvain Le Corff, Luc Lehéricy, Jonathan So, Yongjie Zhu, Elisabeth Gassiat, and Aapo Hyvarinen. Disentangling identifiable features from noisy data with structured nonlinear ica. Advances in Neural Information Processing Systems, 34:1624–1633, 2021.

Meysam Hashemi, Anirudh Nihalani Vattikonda, Jayant Jha, Viktor Sip, Marmaduke M. Woodman, Fabrice Bartolomei, and Viktor Jirsa. Amortized bayesian inference on generative dynamical network models of epilepsy using deep neural density estimators. Neural networks : the official journal of the International Neural Network Society, 163:178–194, 2023. URL https://doi.org/10.1016/j.neunet.2023. 03.040.

Rubén Herzog, Pedro A. M. Mediano, Fernando E. Rosas, Andrea I. Luppi, Yonatan Sanz-Perl, Enzo Tagliazucchi, Morten L. Kringelbach, Rodrigo Cofré, and Gustavo Deco. Neural mass modeling for the masses: Democratizing access to whole-brain biophysical modeling with fastdmf. Network Neuroscience, 8:1590– 1612, 2024. URL https://doi.org/10.1162/netn\_a\_00410.

Mark H Histed, Amy M Ni, and John HR Maunsell. Insights into cortical mechanisms of behavior from microstimulation experiments. Progress in neurobiology, 103:115–130, 2013.

Harvey Huang, Nicholas M. Gregg, Gabriela Ojeda Valencia, Benjamin H. Brinkmann, Brian Nils Lundstrom, Gregory A. Worrell, Kai J. Miller, and Dora Hermes. Electrical stimulation of temporal and limbic circuitry produces distinct responses in human ventral temporal cortex. The Journal ofNeuroscience, 43:4434 – 4447, 2023. URL https://doi.org/10.1523/JNEUROSCI.1325-22.2023.

Aapo Hyvarinen and Hiroshi Morioka. Unsupervised feature extraction by time-contrastive learning and nonlinear ica. Advances in neural information processing systems, 29, 2016.

Jin ichi Nagumo, Suguru Arimoto, and Shuji Yoshizawa. An active pulse transmission line simulating nerve axon. Proceedings of the IRE, 50:2061–2070, 1962. URL https://doi.org/10.1109/JRPROC. 1962.288235.

Hidehiko K Inagaki, Lorenzo Fontolan, Sandro Romani, and Karel Svoboda. Discrete attractor dynamics underlies persistent activity in the frontal cortex. Nature, 566(7743):212–217, 2019.

Mitra Javadzadeh, Marine Schimel, Sonja B. Hofer, Yashar Ahmadian, and Guillaume Hennequin. Dynamic consensus-building between neocortical areas via long-range connections. bioRxiv, 2024. URL https: //doi.org/10.1101/2024.11.27.625691.

Andrew Kyle Lampinen, Stephanie CY Chan, Yuxuan Li, and Katherine Hermann. Representation biases: variance is not always a good proxy for importance. eneuro, 13(3):ENEURO–0461, 2026.

Kenneth W Latimer, Jacob L Yates, Miriam LR Meister, Alexander C Huk, and Jonathan W Pillow. Single-trial spike trains in parietal cortex reveal discrete steps during decision-making. Science, 349(6244):184–187, 2015.

Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M Asano, Taco Cohen, and Stratis Gavves. Citris: Causal identifiability from temporal intervened sequences. In International Conference on Machine Learning, pp. 13557–13603. PMLR, 2022.

Phillip Lippe, Sara Magliacane, Sindy Löwe, Yuki M Asano, Taco Cohen, and Efstratios Gavves. Causal representation learning for instantaneous and temporal effects in interactive systems. In International Conference on Learning Representations, 2023.

David Lipshutz, Amin Nejatbakhsh, and Alex H Williams. Disentangling recurrent neural dynamics with stochastic representational geometry. In ICLR 2024 Workshop on Representational Alignment, 2024.

Thomas Zhihao Luo, Timothy Doyeon Kim, Diksha Gupta, Adrian G Bondy, Charles D Kopec, Verity A Elliott, Brian DePasquale, and Carlos D Brody. Transitions in dynamical regime and neural mode during perceptual decisions. Nature, 646(8087):1156–1166, 2025.

Niru Maheswaranathan, Alex Williams, Matthew Golub, Surya Ganguli, and David Sussillo. Universality and individuality in neural dynamics across large populations of recurrent networks. Advances in neural information processing systems, 32, 2019.

Asma Motiwala, Joana Soldado-Magraner, Aaron P. Batista, Matthew A. Smith, and Byron M. Yu. Braincomputer interfaces as a causal probe for scientific inquiry. Trends in cognitive sciences, 30(1):40 – 53, 2026. URL https://doi.org/10.1016/j.tics.2025.06.017.

Amin Nejatbakhsh and Yixin Wang. Identifying neural dynamics using interventional state space models. In International Conference on Machine Learning, 2025. URL https://openreview.net/forum? id=n7qKt6gjl9.

Amin Nejatbakhsh, Francesco Fumarola, Saleh Esteki, Taro Toyoizumi, Roozbeh Kiani, and Luca Mazzucato. Predicting the effect of micro-stimulation on macaque prefrontal activity based on spontaneous circuit dynamics. Physical review research, 5, 2023. URL https://doi.org/10.1103/physrevresearch. 5.043211.

Astrid A. Prinz, L. F. Abbott, and Eve Marder. The dynamic clamp comes of age. Trends in neurosciences, 27 (4):218–24, 2004. URL https://doi.org/10.1016/J.TINS.2004.02.004.

William Qian, Jacob A Zavatone-Veth, Benjamin S Ruben, and Cengiz Pehlevan. Partial observation can induce mechanistic mismatches in data-constrained models of neural dynamics. Advances in Neural Information Processing Systems, 37:67467–67510, 2024.

Andrew R. Sedler, Chris VerSteeg, and Chethan Pandarinath. Expressive architectures enhance interpretability of dynamics-based neural population models. Neurons, behavior, data analysis, and theory, 2023. URL https://arxiv.org/abs/2212.03771.

Natalie A. Steinemann, Gabriel M. Stine, Eric M. Trautmann, Ariel Zylberberg, Daniel M. Wolpert, and Michael N. Shadlen. Direct observation of the neural computations underlying a single decision. eLife, 12, 2024. URL https://doi.org/10.7554/eLife.90859.3.

Christopher Versteeg, Jonathan D McCart, Mitchell Ostrow, David M Zoltowski, Clayton B Washington, Laura Driscoll, Olivier Codol, Jonathan A Michaels, Scott W Linderman, David Sussillo, et al. Computationthrough-dynamics benchmark: Simulated datasets and quality metrics for dynamical models of neural activity. bioRxiv, 2025.

J. Simon Wiegert, Mathias Mahn, Matthias Prigge, Yoav Printz, and Ofer Yizhar. Silencing neurons: Tools, applications, and experimental constraints. Neuron, 95:504 – 529, 2017. URL https://doi.org/10. 1016/j.neuron.2017.06.050.

Weiran Yao, Guangyi Chen, and Kun Zhang. Temporally disentangled representation learning. Advances in Neural Information Processing Systems, 35:26492–26503, 2022.

Congxi Zhang and Yongchun Xie. Identifiable representation and model learning for latent dynamic systems. Space: Science & Technology, 5:0267, 2025.

David M. Zoltowski, Kenneth W. Latimer, Jacob L. Yates, Alexander C. Huk, and Jonathan W. Pillow. Discrete stepping and nonlinear ramping dynamics underlie spiking responses of lip neurons during decision-making. Neuron, 102, 2019. URL http://dx.doi.org/10.1016/j.neuron.2019.04.031.

## A IDENTIFIABILITY PROOFS

## A.1 SETUP

We consider the interventional state space model

$$
{ \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } \sim P ( { \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } \mid f _ { \theta } ( { \mathbf { \psi } } _ { { \mathbf { \psi } } _ { t } } ) ) ,\tag{7}
$$

$$
\pmb { x } _ { t + 1 } = \pmb { m } _ { t } \odot g _ { \phi } ( \pmb { x } _ { t } ) + \pmb { B } \pmb { u } _ { t } + \pmb { \varepsilon } _ { t } ,\tag{8}
$$

where $\pmb { x } _ { t } \in \mathbb { R } ^ { D } , \pmb { y } _ { t } \in \mathbb { R } ^ { N } , \pmb { m } _ { t } \in \{ 0 , 1 \} ^ { D }$ with $m _ { t , j } : = \mathbb { I } \{ ( B \pmb { u } _ { t } ) _ { j } ~ = ~ 0 \}$ is the intervention mask, and $\odot$ denotes elementwise multiplication. The inputs $\mathbf { \pmb { u } } _ { t }$ are exogenous and observed, i.e., independent of the latent states and of the process noise. The process noise $\varepsilon _ { t }$ has finite first moments and is jointly independent of the current latent state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ and intervention ${ \pmb u } _ { t } , \mathrm { i . e . , } \varepsilon _ { t } \perp \perp \left( { \pmb x } _ { t } , { \pmb u } _ { t } \right)$

Throughout, we compare two parameterizations from the same iSSM model class that induce identical observation distributions within each considered intervention regime, including the interventionfree regime ${ \pmb u } = 0$ . We denote the second parameterization by hats:

$$
{ \mathbf { } } { \mathbf { } } y _ { t } \sim P ( { \mathbf { } } { \mathbf { } } y _ { t } \mid f _ { \hat { \theta } } ( \hat { { \mathbf { x } } } _ { t } ) ) ,\tag{9}
$$

$$
\hat { \pmb { x } } _ { t + 1 } = \hat { m } _ { t } \odot g _ { \hat { \phi } } ( \hat { \pmb { x } } _ { t } ) + \hat { \pmb { B } } \pmb { u } _ { t } + \hat { \varepsilon } _ { t } .\tag{10}
$$

Both parameterizations share the same observation kernel $P ( \pmb { y } _ { t } \mid \cdot )$ ; that is, we compare candidate latent dynamics and decoders under a common measurement map, which is given by the recording process. Also note that observations are conditionally independent across time given the noiseless signals.

Notation. We use P for probability laws and conditional distributions, and $p$ for densities where they exist. No density is assumed for the observations or noiseless signals in Step 1. Events E and $E _ { t }$ are measurable subsets of the observation space.

## A.2 STEP 1: IDENTIFICATION OF THE NOISELESS REPRESENTATION

We follow the observation-model argument of Nejatbakhsh & Wang (2025), allowing noiseless distributions without densities and establishing identification of the noiseless transitions.

Define the noiseless signal $\bar { \pmb y } _ { t } : = f _ { \theta } ( \pmb x _ { t } )$ . Write

$$
K _ { t } ( E _ { t } \mid z _ { t } ) : = P ( \pmb { y } _ { t } \in E _ { t } \mid \bar { \pmb { y } } _ { t } = z _ { t } )\tag{11}
$$

for the observation kernel, which is shared by both parameterizations. Here $z _ { t } \in \mathbb { R } ^ { N }$ is a possible value of the noiseless signal ${ \bar { \mathbf { y } } } _ { t } .$ . We use $E _ { t }$ for measurable events in the observation space and $F _ { t }$ for measurable events in the noiseless-signal space.

Noiseless marginals. For every measurable observation event $E _ { \mathrm { { : } } }$

$$
{ \cal P } ( { \pmb y } _ { t } \in E ) = \int K _ { t } ( E \mid z ) P _ { { \bar { \pmb y } } _ { t } } ( d z ) .\tag{12}
$$

If two candidate parameterizations induce the same observed distribution $P _ { \pmb { y } _ { t } } = \hat { P } _ { \pmb { y } _ { t } }$ , we have

$$
\int K _ { t } ( E \mid z ) \left( P _ { { \bar { y } } _ { t } } - { \hat { P } } _ { { \bar { y } } _ { t } } \right) ( d z ) = 0 \qquad { \mathrm { f o r ~ e v e r y ~ } } E .\tag{13}
$$

By Assumption 1, $P _ { \bar { \pmb { y } } _ { t } } = \hat { P } _ { \bar { \pmb { y } } _ { t } }$ , so the noiseless marginal is identified.

Noiseless transitions. We first identify the joint law of two consecutive noiseless signals; their conditional distribution then follows. For convenience, we label the two time points as 1, 2, but they can be any two adjacent times.

Conditional independence of the observations given the noiseless signals yields

$$
P ( y _ { 1 } \in E _ { 1 } , y _ { 2 } \in E _ { 2 } ) = \int K _ { 1 } ( E _ { 1 } \mid z _ { 1 } ) K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) P _ { { \bar { y } } _ { 1 } , { \bar { y } } _ { 2 } } ( d z _ { 1 } , d z _ { 2 } ) .\tag{14}
$$

Let

$$
\mu : = P _ { \bar { y } _ { 1 } , \bar { y } _ { 2 } } - \hat { P } _ { \bar { y } _ { 1 } , \bar { y } _ { 2 } }\tag{15}
$$

be the finite signed measure given by the difference of the two candidate noiseless joint distributions. Since the candidates induce the same observed joint law, subtracting their expressions above gives

$$
\int K _ { 1 } ( E _ { 1 } \mid z _ { 1 } ) K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) \mu ( d z _ { 1 } , d z _ { 2 } ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } E _ { 1 } , E _ { 2 } .\tag{16}
$$

We want to show that this implies $\mu = 0 ;$ ; and we will do this by applying Assumption 1 once to each coordinate.

First, fix an observation event $E _ { 2 }$ and define a signed measure on the first noiseless coordinate by

$$
\nu _ { E _ { 2 } } ( F _ { 1 } ) : = \int _ { F _ { 1 } \times \mathbb { R } ^ { N } } K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) \mu ( d z _ { 1 } , d z _ { 2 } ) .\tag{17}
$$

This measure is finite because $\mu$ is finite and $0 \leq K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) \leq 1$ . Substituting into Eq. (16) gives

$$
\int K _ { 1 } ( E _ { 1 } \mid z _ { 1 } ) \nu _ { E _ { 2 } } ( d z _ { 1 } ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } E _ { 1 } .\tag{18}
$$

By Assumption 1, $\nu _ { E _ { 2 } } = 0$ . Since $E _ { 2 }$ was arbitrary, this gives

$$
\int _ { F _ { 1 } \times \mathbb { R } ^ { N } } K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) \mu ( d z _ { 1 } , d z _ { 2 } ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } F _ { 1 } , E _ { 2 } .\tag{19}
$$

Next, we fix a noiseless event $F _ { 1 }$ and define a finite signed measure on the second coordinate by

$$
\rho _ { F _ { 1 } } ( F _ { 2 } ) : = \mu ( F _ { 1 } \times F _ { 2 } ) .\tag{20}
$$

Eq. (19) then becomes

$$
\int K _ { 2 } ( E _ { 2 } \mid z _ { 2 } ) \rho _ { F _ { 1 } } ( d z _ { 2 } ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ } E _ { 2 } .\tag{21}
$$

A second application of Assumption 1 gives $\rho _ { F _ { 1 } } = 0$ . Since $F _ { 1 }$ was arbitrary,

$$
\mu ( F _ { 1 } \times F _ { 2 } ) = 0 \qquad \mathrm { f o r e v e r y } \ F _ { 1 } , F _ { 2 } .\tag{22}
$$

Thus the candidate joint distributions agree on all measurable rectangles, which determine a probability law on $\mathbb { R } ^ { N } \times \mathbf { \bar { \mathbb { R } } } ^ { N }$ . Hence $\mu = 0$ , and the noiseless joint distribution is identified. Uniqueness of conditional distributions then identifies $P ( \bar { y } _ { 2 } \mid \bar { y } _ { 1 } )$ up to a set of conditioning values of probability zero under the common marginal $P _ { \bar { y } _ { 1 } }$ . Because the two time points were arbitrary, this identifies $P ( \bar { \pmb y } _ { t } \mid \bar { \pmb y } _ { t - 1 } )$ for every adjacent pair.

Noiseless trajectories. The same argument applies successively to every coordinate of a finite trajectory. Let

$$
\mu _ { T } : = P _ { { \bar { y } } _ { 1 : T } } - { \hat { P } } _ { { \bar { y } } _ { 1 : T } }\tag{23}
$$

be the difference of the candidate noiseless trajectory laws. Equality of the full observed trajectory laws and conditional independence of the observations give

$$
\int \prod _ { t = 1 } ^ { T } K _ { t } ( E _ { t } \mid z _ { t } ) \mu _ { T } ( d z _ { 1 : T } ) = 0 \qquad \mathrm { f o r e v e r y } \ E _ { 1 } , \ldots , E _ { T } .\tag{24}
$$

Fixing $E _ { 2 } , \ldots , E _ { T }$ , define a finite signed measure on the first coordinate by integrating the remaining kernel factors against $\mu _ { T }$ , as in the definition of $\nu _ { E _ { 2 } }$ above. Completeness of $K _ { 1 }$ implies that this measure is zero on every noiseless event $F _ { 1 }$ . Next, fix $F _ { 1 }$ and $E _ { 3 } , \ldots , E _ { T }$ and apply completeness of $K _ { 2 }$ to the corresponding signed measure on the second coordinate. Continuing in this way through all $T$ coordinates gives

$$
\mu _ { T } ( F _ { 1 } \times \cdot \cdot \cdot \times F _ { T } ) = 0 \qquad \mathrm { f o r e v e r y } \ F _ { 1 } , \ldots , F _ { T } .\tag{25}
$$

The candidate distributions therefore agree on all measurable rectangles and hence on the entire noiseless trajectory distribution. This proves Proposition 1.

## A.3 STEP 2: AFFINE IDENTIFIABILITY WITH SMOOTH INJECTIVE DECODERS

We now provide an alternative route to affine identifiability that replaces the piecewise linearity assumption of Nejatbakhsh & Wang (2025) by smoothness, injectivity, and Gaussian latent transitions.

Assumptions. We adopt the assumptions of Theorem 2. In particular:

(A1) The decoders are injective and continuously differentiable, with continuously differentiable inverses, and the latent transition function $g _ { \phi }$ is continuously differentiable,

(A2) the latent dynamics follow a Gaussian first-order Markov process with positive definite covariance, and

(A3) there exists $\pmb { x } ^ { * } \in \mathbb { R } ^ { D }$ such that ${ \cal J } _ { g _ { \phi } } ( { \pmb x } ^ { * } )$ is full rank.

Proof. Step 1 shows that observationally equivalent parameterizations induce the same noiseless transition distributions, almost everywhere under the common conditioning marginal, i.e.,

$$
P ( { \bar { y } } _ { t } \mid { \bar { y } } _ { t - 1 } ) = { \hat { P } } ( { \bar { y } } _ { t } \mid { \bar { y } } _ { t - 1 } )\tag{26}
$$

almost surely. Given injectivity of the decoder, we may pull back this equality to latent coordinates using $\pmb { x } _ { t } = \dot { { f } } _ { \theta } ^ { - 1 } ( \bar { \pmb { y } } _ { t } )$ and $\hat { \pmb { x } } _ { t } = \dot { f } _ { \hat { \pmb { \theta } } } ^ { - 1 } ( \bar { \pmb { y } } _ { t } )$ :

$$
\operatorname { v o l } J _ { f _ { \theta } ^ { - 1 } } ( { \bar { y } } _ { t } ) p ( x _ { t } \mid x _ { t - 1 } ) = \operatorname { v o l } J _ { f _ { \hat { \theta } } ^ { - 1 } } ( { \bar { y } } _ { t } ) { \hat { p } } ( { \hat { x } } _ { t } \mid { \hat { x } } _ { t - 1 } ) .\tag{27}
$$

Invertibility of the decoders ensures that a mapping $\psi ( \pmb { x } _ { t } ) = \hat { \pmb { x } } _ { t }$ exists, where $\psi = f _ { \hat { \theta } } ^ { - 1 } \circ f _ { \theta }$ . Using this definition and taking logarithms yields

$$
\log \operatorname { v o l } J _ { f _ { \theta } ^ { - 1 } } ( { \bar { y } } _ { t } ) + \log p ( x _ { t } \mid x _ { t - 1 } ) = \log \operatorname { v o l } J _ { f _ { \hat { \sigma } } ^ { - 1 } } ( { \bar { y } } _ { t } ) + \log { \hat { p } } ( \psi ( x _ { t } ) \mid \psi ( x _ { t - 1 } ) ) .\tag{28}
$$

Now apply the mixed derivative $\nabla _ { \pmb { x } _ { t } } \nabla _ { \pmb { x } _ { t - 1 } }$ . The Jacobian-volume terms vanish because they depend on $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ but not on ${ \mathbf { \mathcal { x } } } _ { t - 1 }$ . Hence

$$
\nabla _ { \pmb { x } _ { t } } \nabla _ { \pmb { x } _ { t - 1 } } \log p ( \pmb { x } _ { t } \mid \pmb { x } _ { t - 1 } ) = \nabla _ { \pmb { x } _ { t } } \nabla _ { \pmb { x } _ { t - 1 } } \log \hat { p } ( \psi ( \pmb { x } _ { t } ) \mid \psi ( \pmb { x } _ { t - 1 } ) ) .\tag{29}
$$

For Gaussian latent transitions,

$$
\log p ( { \pmb x } _ { t } \mid { \pmb x } _ { t - 1 } ) = - \frac { 1 } { 2 } ( { \pmb x } _ { t } - g _ { \phi } ( { \pmb x } _ { t - 1 } ) ) ^ { \top } { \pmb Q } ^ { - 1 } ( { \pmb x } _ { t } - g _ { \phi } ( { \pmb x } _ { t - 1 } ) ) + \mathrm { c o n s t } ,\tag{30}
$$

where the constant does not depend jointly on $\mathbf { \Phi } ( \pmb { x } _ { t } , \pmb { x } _ { t - 1 } )$ . Therefore,

$$
\nabla _ { \pmb { x } _ { t } } \nabla _ { \pmb { x } _ { t - 1 } } \log p ( \pmb { x } _ { t } \mid \pmb { x } _ { t - 1 } ) = \pmb { Q } ^ { - 1 } \pmb { J } _ { \pmb { g } _ { \phi } } ( \pmb { x } _ { t - 1 } ) .\tag{31}
$$

Analogously, for the hatted parameterization, the chain rule gives

$$
\nabla _ { \pmb { x } _ { t } } \nabla _ { \pmb { x } _ { t - 1 } } \log \hat { p } ( \psi ( \pmb { x } _ { t } ) \mid \psi ( \pmb { x } _ { t - 1 } ) ) = \pmb { J } _ { \psi } ( \pmb { x } _ { t } ) ^ { \top } \hat { \pmb { Q } } ^ { - 1 } \pmb { J } _ { \pmb { g } _ { \hat { \phi } } } ( \psi ( \pmb { x } _ { t - 1 } ) ) \pmb { J } _ { \psi } ( \pmb { x } _ { t - 1 } ) .\tag{32}
$$

Substituting into (29) yields

$$
\begin{array} { r } { Q ^ { - 1 } J _ { g _ { \phi } } ( x _ { t - 1 } ) = J _ { \psi } ( x _ { t } ) ^ { \top } \hat { Q } ^ { - 1 } J _ { g _ { \hat { \phi } } } ( \psi ( x _ { t - 1 } ) ) J _ { \psi } ( x _ { t - 1 } ) . } \end{array}\tag{33}
$$

By Assumption $( \mathbf { A } 3 ) , J _ { g _ { \phi } } ( \pmb { x } ^ { * } )$ is full rank at some $\pmb { x } ^ { * } \in \mathbb { R } ^ { D }$ . Since $g _ { \phi }$ is continuously differentiable by Assumption $( \mathbf { A } 1 )$ , its Jacobian is continuous. Hence there exists a nonempty open neighbourhood $\boldsymbol { B } _ { \boldsymbol { x } ^ { * } }$ of $\mathbf { \nabla } _ { \mathbf { \mathcal { X } } } \ast \mathbf { \ v { x } }$ on which $J _ { g _ { \phi } }$ is invertible. Since all matrices in (33) are of size $\bar { D } \times \bar { D }$ , this implies that each factor on the right-hand side must also be full rank on that set. In particular, $J _ { \psi } ( \pmb { x } _ { t } ) , \hat { Q }$ $J _ { g _ { \hat { \phi } } } ( \psi ( \pmb { x } _ { t - 1 } ) )$ ), and $\bar { J } _ { \psi } ( \pmb { x } _ { t - 1 } )$ ) are invertible for any $\pmb { x } _ { t - 1 } \in B _ { \pmb { x } ^ { * } }$ , and we may therefore rearrange the expression as

$$
\pmb { J } _ { \psi } ( \pmb { x } _ { t } ) ^ { \top } = \pmb { Q } ^ { - 1 } \pmb { J } _ { g _ { \phi } } ( \pmb { x } _ { t - 1 } ) \pmb { J } _ { \psi } ( \pmb { x } _ { t - 1 } ) ^ { - 1 } \pmb { J } _ { g _ { \hat { \phi } } } ( \psi ( \pmb { x } _ { t - 1 } ) ) ^ { - 1 } \pmb { \hat { Q } } .\tag{34}
$$

By Assumption (A2) and for any ${ \pmb x } _ { t - 1 } ~ \in ~ \boldsymbol { B } _ { { \pmb x } ^ { * } }$ , the conditional distribution of ${ \mathbf { } } x _ { t } | { \mathbf { } } x _ { t - 1 }$ has full support for $t > 2$ . Therefore, the above equation holds for any $\pmb { x } _ { t } \in \mathbb { R } ^ { D }$ . Since the right-hand side depends on $\mathbf { \nabla } _ { \mathbf { x } _ { t - 1 } }$ but not on $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , this implies $\textbf J _ { \psi } ( \pmb x _ { t } )$ is constant with respect to $\mathbf { \nabla } _ { \mathbf { x } _ { t } } .$ , and therefore, ψ is affine:

$$
\pmb { \hat { x } } _ { t } = \psi ( \pmb { x } _ { t } ) = \pmb { M } \pmb { x } _ { t } + \pmb { c } ,\tag{35}
$$

which proves affine identifiability. Note that the decoders, and consequently $\psi = f _ { \hat { \theta } } ^ { - 1 } \circ f _ { \theta } ,$ do not depend on the intervention. The above affine map is therefore shared across regimes with and without interventions. □

## A.4 STEP 3: REFINEMENT UNDER INTERVENTIONS

We now show how hard interventions refine affine identifiability (Theorem 2) to block-wise permutation-scaling-shift identifiability.

Notation. For an index set $\mathcal { T } \subseteq \{ 1 , \dots , D \} , - \mathcal { T }$ denotes its complement. We write $M _ { \mathcal { I ^ { \prime } } , \mathcal { I } }$ for the submatrix with rows indexed by ${ \mathcal { J } } ^ { \prime }$ and columns indexed by $\mathcal { I }$

Assumptions. Step 3 assumes a common invertible affine correspondence $\hat { \pmb { x } } _ { t } = { \pmb { M } } { \pmb { x } } _ { t } + { \pmb { c } } .$ , regularity and local non-degeneracy of dynamics (Assumption (B1)), and full state support across intervention regimes (Assumption (B2)).

Statement. For a fixed intervention input u, let $\mathcal { I }$ and $\mathcal { T } ^ { \prime }$ be the target sets in the true and candidate parameterizations, respectively. We show that $| { \mathcal { I } } ^ { \prime } | = | { \mathcal { I } } |$ and

$$
M _ { { \mathcal { I } } ^ { \prime } , - { \mathcal { I } } } = 0 , \qquad M _ { - { \mathcal { I } } ^ { \prime } , { \mathcal { I } } } = 0 .\tag{36}
$$

Thus, after aligning the coordinates, M is block diagonal with invertible blocks, which means that the candidate coordinates in $\mathcal { T } ^ { \prime }$ depend only on the true coordinates in $\mathcal { I } . \mathrm { ~ H ~ } | \mathcal { I } | = 1$ , the intervened latent is identifiable up to permutation, scaling, and shift.

Proof. Let $\pmb { b } = \mathbb { E } [ \pmb { \varepsilon } _ { t } \ \vert \ \pmb { x } _ { t } = \pmb { x } , \pmb { u } _ { t } = \pmb { u } ]$ and $\hat { \pmb { b } } = \mathbb { E } [ \hat { \pmb { \varepsilon } } _ { t } \ | \ \hat { \pmb { x } } _ { t } = \hat { \pmb { x } } , \pmb { u } _ { t } = \pmb { u } ]$ be the conditional process noise means for the true and candidate parameterizations, respectively. In the interventionfree regime ${ \pmb u } = 0$ , comparing conditional means under affine identifiability gives

$$
g _ { \hat { \phi } } ( M x + c ) + \hat { b } = M g _ { \phi } ( \pmb { x } ) + \pmb { c } + M \pmb { b } .\tag{37}
$$

The above equality initially holds almost everywhere under the intervention-free state distribution. By Assumption (B2) and Assumption (B1), the latent state distribution has full support and both sides are continuous, making the equality hold everywhere in x $\in \mathbb { R } ^ { D }$

Since the noise means are constant in x and the intervention $( \mathbf { A p p . A . l } )$ , differentiating with respect to x yields

$$
J _ { g _ { \hat { \phi } } } ( M \mathbf { x } + \mathbf { c } ) M = M J _ { g _ { \phi } } ( \mathbf { x } ) .\tag{38}
$$

For a fixed intervention input u, define $D _ { u } : = \mathrm { d i a g } ( m ( { \boldsymbol { u } } ) )$ and $\hat { D } _ { \pmb { u } } : = \mathrm { d i a g } ( \hat { m } ( \pmb { u } ) )$ ). Comparing conditional means under this input gives

$$
M D _ { u } g _ { \phi } ( x ) + M B u + c + M b = \hat { D } _ { u } g _ { \hat { \phi } } ( M x + c ) + \hat { B } u + \hat { b } .\tag{39}
$$

Again, full-support and continuity assumptions ensure this holds everywhere in $\pmb { x } \in \mathbb { R } ^ { D }$ . Differentiating with respect to x yields

$$
M D _ { u } J _ { g _ { \phi } } ( { \pmb x } ) = \hat { D } _ { \pmb u } J _ { g _ { \hat { \phi } } } ( M { \pmb x } + { \pmb c } ) M .\tag{40}
$$

Substituting Eq. (38) into the right-hand side gives

$$
M D _ { u } J _ { g _ { \phi } } ( { \pmb x } ) = \hat { D } _ { \pmb { u } } M J _ { g _ { \phi } } ( { \pmb x } )\tag{41}
$$

By Assumption (B1), $J _ { g _ { \phi } } ( { \pmb x } ^ { * } )$ is invertible at some $\pmb { x } ^ { * } \in \mathbb { R } ^ { D }$ , thus we get

$$
M D _ { u } = \hat { D } _ { u } M .\tag{42}
$$

Entrywise, this means $( m _ { j } ( { \pmb u } ) - \hat { m } _ { i } ( { \pmb u } ) ) M _ { i j } = 0$ . Hence, whenever candidate coordinate i and true coordinate j have different intervention statuses, we have $M _ { i j } = 0$ . Consequently,

$$
M _ { { \mathcal { I } } ^ { \prime } , - { \mathcal { I } } } = 0 , \qquad M _ { - { \mathcal { I } } ^ { \prime } , { \mathcal { I } } } = 0 .\tag{43}
$$

Moreover, Eq. (42) gives $\hat { D } _ { u } = M D _ { u } M ^ { - 1 }$ , so the masks have equal rank and $| { \mathcal { I } } ^ { \prime } | = | { \mathcal { I } } |$ . After aligning the target sets, the diagonal blocks of M are therefore square and invertible, since M is invertible. □

From block to full identifiability. Under the unordered pairs condition (Assumption 2), the above argument applies to every pair of latent coordinates, forcing M to be diagonal up to permutation. Hence $M = \pmb { \Lambda } \pmb { \Pi }$ , where Π is a permutation matrix and Λ is diagonal and invertible, yielding identifiability up to permutation, scaling, and shift as shown by Nejatbakhsh & Wang (2025).

## A.5 NONLINEAR INPUT MAPS

Corollary 4 also holds when the linear input map $\mathbf { \nabla } _ { B u _ { t } }$ is replaced by an arbitrary deterministic map $h ( { \boldsymbol { \mathbf { \mathit { u } } } } _ { t } )$ , provided there exists an intervention-free regime on the support of u.

Proposition 5 (Nonlinear input maps). Replace $\mathbf { \nabla } _ { B u _ { t } }$ in Eq. (2) by an arbitrary deterministic map $h : { \dot { \mathbb { R } } } ^ { M } \to \mathbb { R } ^ { \dot { D } }$ , and let $m _ { t , j } : = \mathbb { I } \{ h _ { j } ( { \boldsymbol { \boldsymbol { u } } } _ { t } ) = 0 \}$ . Assume the conditions of Corollary 4, with Assumption 2 applied to the coordinates that receive nonzero input. If there exists an interventionfree regime ${ \pmb u } ^ { 0 } \in \mathrm { s u p p } ( { \pmb u } )$ such that $h ( { \pmb u } ^ { 0 } ) = 0$ and $\hat { h } ( { \pmb u } ^ { 0 } ) = 0 ;$ , then the latent variables remain identifiable up to permutation, scaling, and shift.

ProofofProposition 5. Here $m _ { t , j } = \mathbb { I } \{ h _ { j } ( { \boldsymbol { \boldsymbol { u } } } _ { t } ) = 0 \}$ . We verify that the three-step proof of Corollary 4 remains valid when replacing Bu by $h ( \pmb { u } )$ with one additional assumption on the input maps h and $\hat { h } ,$ , needed for steps 2 and 3.

Step 1 is unchanged because it does not involve the input map.

Step 2 requires an intervention-free regime for both parameterizations because the transition Jacobian $\mathrm { d i a g } ( { \pmb { m } } _ { t } ) { J } _ { { g } _ { \phi } } ( { \pmb { x } } _ { t } )$ is singular during active interventions. By assumption, $h ( { \pmb u } ^ { 0 } ) = \hat { h } ( { \pmb u } ^ { 0 } ) = 0 .$ so both masks equal 1 at $\mathbf { } \mathbf { } u _ { t } = \mathbf { \Gamma } u ^ { 0 }$ . With this, the latent transitions become ${ \pmb x } _ { t + 1 } = g _ { \phi } ( { \pmb x } _ { t } ) + { \pmb \varepsilon } _ { t }$ and $\hat { \pmb x } _ { t + 1 } = g _ { \hat { \phi } } ( \hat { \pmb x } _ { t } ) + \hat { \pmb \varepsilon } _ { t }$ , and Theorem 2 can be applied unchanged. The resulting affine map $\psi ( { \pmb x } ) = M { \pmb x } + { \pmb x }$ is independent of the intervention and therefore holds both at $\mathbf { \boldsymbol { u } } ^ { 0 }$ and under active interventions.

Step $^ 3$ uses $\mathbf { \Delta } _ { \pmb { u } } ^ { 0 }$ as the intervention-free regime for both parameterizations. For any fixed input $^ { \mathbf { \delta u , } }$ , the terms $h ( \pmb { u } )$ and $\hat { h } ( { \boldsymbol u } )$ , just like Bu and $\hat { B } \boldsymbol { u } .$ , disappear upon differentiation with respect to x. The remainder of the proof in App. A.4 is unchanged. □

Remark 1. Note that the data constrains h only on supp $( { \pmb u } )$ . If the observed interventions comprise finitely many channel-amplitude combinations, one-hot encoding these combinations allows a linear input map with one column per combination to represent any h exactly on the observed support.

## A.6 OBSERVATION MASKING

In the human iEEG study, we mask out observation channels that correspond to active stimulation sites on each trial to exclude saturation artefacts. We show here that the identifiability guarantees above remain valid in this setting with modest additional conditions.

Definition 1 (Exogenous observation masking). Let ${ \pmb o } \in \{ 0 , 1 \} ^ { N }$ denote a binary mask indicating which observation coordinates are retained. On trial k, a mask $\mathbf { \xi } _ { o _ { k } }$ applies. The mask is known and determined only by observed exogenous variables; the observation noise is independent of these variables and the latent states taken jointly. Write $\mathcal { O } ( o ) : = \{ n : o _ { n } = 1 \}$ for the indices of the observed coordinates and let $S _ { o } : \bar { \mathbb { R } } ^ { N } \overset { \cdot } {  } \mathbb { R } ^ { | \mathcal { O } ( o ) | }$ denote projection onto those coordinates; we write $S _ { A }$ analogously for projection onto an index set $\mathcal { A } \subseteq \bar { \{ 1 , \ldots , N \} }$ . Finally, define the masked noiseless signal as $\bar { \pmb { y } } _ { t , \mathcal { O } ( \pmb { o } ) } : = S _ { o } f _ { \theta } ( \pmb { x } _ { t } )$ , and masked observations as $\pmb { y } _ { t , \mathcal { O } ( \pmb { o } ) } : = S _ { o } \pmb { y } _ { t }$

We verify that the three-step proof of Corollary 4 remains valid under Definition 1. For each step, we state what the masking changes, give sufficient conditions under which the step still holds, and discuss plausibility in the human iEEG analysis.

Step 1: bounded completeness under observation masking. Step 1 (App. A.2) requires bounded completeness (Assumption 1) for the observation model. Under masking, the analogous requirement is that $P ( \pmb { y } _ { t , \mathcal { O } ( o ) } \mid \bar { \pmb { y } } _ { t , \mathcal { O } ( o ) } )$ is bounded complete in the sense of Assumption 1. Below we provide a sufficient condition that applies to the human iEEG analysis.

Proposition 6 (Bounded completeness under observation masking). Suppose that:

(i) observations are generated by additive noise $\pmb { y } _ { t } = \bar { \pmb { y } } _ { t } + \pmb { \eta } _ { t }$ , where $\bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } } \bar { \mathbf { \nabla } \mathbf { \nabla } } \bar { \mathbf { \nabla } \mathbf { \nabla } } \bar { \mathbf { \nabla } \mathbf { \nabla } } \bar { \mathbf { \nabla } \mathbf { \nabla } } \bar \bar { \mathbf { \nabla } \mathbf { \nabla } \nabla } \bar \bar { \mathbf { \nabla } \mathbf { \nabla } \nabla \nabla \mathbf { \nabla } \nabla \nabla \bar } \bar \bar  \mathbf { \nabla \nabla } \mathbf { \nabla } \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla \nabla $ is the noiseless signal and $\eta _ { t } \perp \perp \bar { y } _ { t } ,$

(ii) the noise $\pmb { \eta } _ { t }$ admits a density $p _ { \eta }$ on $\mathbb { R } ^ { N }$ ; and

(iii) the characteristicfunction $\varphi _ { \eta } o f \eta _ { t }$ satisfies $\varphi _ { \pmb { \eta } } ( \pmb { s } ) \neq 0$ for all $\boldsymbol { s } \in \mathbb { R } ^ { N }$

Then the conditional distribution $P ( \pmb { y } _ { t } \mid \bar { \pmb { y } } _ { t } )$ satisfies Assumption 1. The same conclusion holds under any mask satisfying Definition 1.

Proof. We first prove the claim for the unmasked case, and later we show that the argument also holds under masking.

By (i) and (ii), the conditional density of ${ \mathbf { } } _ { \pmb { y } _ { t } }$ given $\bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar { \mathbf { \nabla } } } \bar { \mathbf { \nabla } } _ { \bar \mathbf { \nabla } }$ is $p _ { \eta } ( \pmb { y } _ { t } - \bar { \pmb { y } } _ { t } )$ . Let $\mu$ be a finite signed measure satisfying

$$
\int P ( { \pmb y } _ { t } \in E \mid { \bar { \pmb y } } _ { t } = z ) d \mu ( z ) = 0 \qquad \mathrm { f o r ~ e v e r y ~ m e a s u r a b l e ~ } E .
$$

Using the conditional density and interchanging the integrals gives $\begin{array} { r } { \int _ { E } ( p _ { \pmb { \eta } } * \mu ) ( \pmb { y } ) d \pmb { y } = 0 } \end{array}$ for every such $E .$ Since $p _ { \eta } \in L ^ { 1 }$ and $\mu$ is finite, the convolution is integrable, so

$$
( p _ { \eta } * \mu ) ( { \pmb y } ) = \int p _ { \eta } ( { \pmb y } - { \pmb z } ) d \mu ( { \pmb z } ) = 0 \quad \mathrm { f o r ~ L e b e s g u e - a l m o s t ~ e v e r y ~ } { \pmb y } .\tag{44}
$$

Taking Fourier transforms in $^ { \mathbf { \psi } } { \mathbf { * } } ^ { \prime } { \mathbf { * } } ^ { \mathrm { ~ } }$ , with $\begin{array} { r } { \mathcal { F } \mu ( s ) : = \int e ^ { i s ^ { \intercal } z } d \mu ( z ) } \end{array}$ , gives

$$
{ \mathcal { F } } \mu ( s ) \varphi _ { \pmb { \eta } } ( s ) = 0 \quad { \mathrm { f o r ~ a l l ~ } } s .\tag{45}
$$

By (iii), $\varphi _ { \pmb { \eta } } ( \pmb { s } ) \neq 0$ for all $s ,$ thus $\begin{array} { r } { \mathcal { F } \mu ( \pmb { s } ) = 0 } \end{array}$ for all $s .$ Uniqueness of the Fourier–Stieltjes transform of finite signed measures then implies $\mu = 0$ , as required.

Now we show that the conditions are preserved under masking. By Definition 1, the mask is exogenous and fixed, so the additive noise model and independence of (i) are preserved:

$$
y _ { t , \mathcal { O } ( o ) } = S _ { o } \bar { y } _ { t } + S _ { o } \eta _ { t } \quad \mathrm { a n d } \quad \eta _ { t , \mathcal { O } ( o ) } \perp \perp \bar { y } _ { t , \mathcal { O } ( o ) } .\tag{46}
$$

Condition (ii) is preserved because the marginal of a density is a density, i.e., integrating $p _ { \eta }$ over the masked-out coordinates yields a density $p _ { \eta _ { o } }$ of $S _ { o } \eta _ { t }$ on $\mathbb { R } ^ { | \mathcal { O } ( o ) | }$

Finally, the characteristic function of the marginal noise satisfies

$$
\varphi _ { \eta _ { o } } ( s _ { o } ) = \varphi _ { \eta } \big ( S _ { o } ^ { \top } s _ { o } \big ) ,\tag{47}
$$

so a non-vanishing $\varphi _ { \eta }$ implies $\varphi _ { \eta _ { o } } ( s _ { o } ) \neq 0$ for all $s _ { o }$ . Hence, the above convolution argument shown for the unmasked case also applies to the masked case. □

Remark 2. In the human iEEG analysis, the mask is fully determined by whichever stimulation site is active during that trial, so Definition 1 applies. Additionally, we assume Gaussian observation noise, $y _ { t } = \bar { y } _ { t } + \eta _ { t }$ with $\pmb { \eta } _ { t } \sim \mathcal { N } ( 0 , \pmb { \Sigma } _ { \eta } )$ and $\Sigma _ { \eta }$ positive definite. Gaussian noise is additive and its characteristic function never vanishes. The non-singular covariance ensures that η<sub>t</sub> admits a density on $\mathbb { R } ^ { N }$ . Thus, Proposition 6 implies that Step 1 holds under masking in this application.

Step 2: affine identifiability under observation masking. Observation masking may break Assumption (A1) of Step 2 (App. A.3), which requires the emission function to be injective and continuously differentiable, with continuously differentiable inverse on its image: projecting to $\mathcal { O } ( o )$ can break injectivity even when the full decoder $f _ { \theta }$ is injective. Below we give sufficient condition under which affine identifiability is preserved, then justify their plausibility in our setting.

Proposition 7 (Affine identifiability under observation masking). Assume that all conditions of Theorem 2 hold, and that for every mask the data contain transitions without active interventions $( m _ { t } = 1 )$ . Assume additionally that:

(i) for every mask o, the restricted decoders $S _ { o } \circ f _ { \theta }$ and $S _ { o } \circ f _ { \hat { \theta } }$ are injective and continuously differentiable, with continuously differentiable inverses on their images;

(ii) for every pair ofmasks o, o<sup>′</sup>, the restricted decoders $S _ { \mathcal { O } ( o ) \cap \mathcal { O } ( o ^ { \prime } ) } \circ f _ { \theta }$ and $S _ { \mathcal { O } ( o ) \cap \mathcal { O } ( o ^ { \prime } ) } \circ f _ { \hat { \theta } }$ are injective.

Then affine identifiability is preserved under observation masking in the sense ofDefinition 1.

Proof. Fix two masks o and ${ \pmb o } ^ { \prime } .$ By (i), we can apply Step 2 separately under each mask, using its transitions without active interventions, yielding affine maps $\psi _ { o } ( x ) = M _ { o } x + c _ { o }$ and $\psi _ { o ^ { \prime } } ( { \pmb x } ) =$ $M _ { o ^ { \prime } } { \pmb x } + { \pmb c } _ { o ^ { \prime } }$ such that

$$
S _ { o } f _ { \theta } ( { \pmb x } ) = S _ { o } f _ { \hat { \theta } } ( \psi _ { o } ( { \pmb x } ) ) , \qquad S _ { o ^ { \prime } } f _ { \theta } ( { \pmb x } ) = S _ { o ^ { \prime } } f _ { \hat { \theta } } ( \psi _ { o ^ { \prime } } ( { \pmb x } ) ) .\tag{48}
$$

Restricting both identities to $\mathcal { O } ( o ) \cap \mathcal { O } ( o ^ { \prime } )$ gives

$$
\begin{array} { r } { S _ { \mathcal { O } ( \bullet ) \cap \mathcal { O } ( \sigma ^ { \prime } ) } f _ { \theta } ( \boldsymbol { x } ) = S _ { \mathcal { O } ( \sigma ) \cap \mathcal { O } ( \sigma ^ { \prime } ) } f _ { \hat { \theta } } ( \psi _ { o } ( \boldsymbol { x } ) ) = S _ { \mathcal { O } ( \sigma ) \cap \mathcal { O } ( \sigma ^ { \prime } ) } f _ { \hat { \theta } } ( \psi _ { o ^ { \prime } } ( \boldsymbol { x } ) ) . } \end{array}\tag{49}
$$

By (ii), injectivity of $S _ { \mathcal { O } ( \pmb { o } ) \cap \mathcal { O } ( \pmb { o } ^ { \prime } ) } \circ f _ { \hat { \theta } }$ implies $\psi _ { o } ( x ) = \psi _ { o ^ { \prime } } ( x )$ for all $\pmb { x } \in \mathbb { R } ^ { D }$ , which is the latent support under both masks by Assumption $( \mathbf { A } 2 )$ ). Hence $M _ { o } = M _ { o ^ { \prime } }$ and $c _ { o } = c _ { o ^ { \prime } }$ . Since o and $\mathbf { o } ^ { \prime }$ were arbitrary, a single affine map ${ \hat { \pmb x } } = { \pmb M } { \pmb x } + { \pmb c }$ holds across all mask patterns. □

Remark 3. In the human iEEG analysis, each mask removes only the two channels at the active stimulation site (at most four under pairwise overlap, out of $N = 9 9$ outputs). For $S _ { o } \circ f _ { \theta }$ to fail injectivity, distinct latents x $\neq \pmb { x } ^ { \prime }$ would have to agree on all unmasked channels while differing only on masked ones. That is, the ∼4 masked coordinates would have to contain unique information needed to distinguish these latents that is absent from the remaining ∼95 channels. A similar argument applies to the continuously differentiable inverse in (i): it requires the Jacobian $S _ { o } J _ { f _ { \theta } } ( { \pmb x } )$ to keep full rank $D$ at every x, which fails only if, at some x, a small change of the latent state along some direction affects the masked channels alone. The masked channels are the ones closest to the stimulated tissue; it is therefore not unthinkable that they contain identifying information that other channels lack. However, other contacts on the same electrode shaft remain observed, and stimulation also evokes responses at distant channels. We therefore consider such unique information in the few masked channels unlikely and treat conditions (i) and (ii) as a mild robustness assumption. Finally, each trial contains a single stimulation pulse, so all other time steps are free of active interventions, as required by Proposition 7.

Step 3: refinement under interventions. Step 3 (App. A.4) concerns only the latent transition under hard interventions and does not involve the observation model. What is relevant in this context is that Step 3 combines constraints obtained from different intervention channels to restrict a single affine transformation M. The conclusion of Proposition 7 that the affine maps obtained under different observation masks must coincide, is therefore essential. However, once Steps 1 and 2 establish a common affine map ${ \hat { \pmb x } } = { \pmb M } { \pmb x } + { \pmb c }$ across masks, Step 3 remains valid and requires no additional assumptions.

## B VARIATIONAL INFERENCE

We perform parameter learning and latent state inference using variational inference. Given observations $\mathbf { \boldsymbol { \mathbf { \mathit { y } } } } _ { 1 : T }$ and interventions ${ \pmb u } _ { 1 : T }$ , our goal is to infer the posterior distribution over latent trajectories $p _ { \Theta } ( \pmb { x } _ { 1 : T } \mid \pmb { y } _ { 1 : T } , \pmb { u } _ { 1 : T } )$ while learning model parameters $\Theta = ( \theta , \phi , B , Q )$

Since exact inference is intractable due to nonlinearities in both the transition and emission models, we introduce an approximate posterior ${ \bf q } _ { \Phi } ( { \pmb x } _ { 1 : T } \mid { \pmb y } _ { 1 : T } , { \pmb u } _ { 1 : T } )$ and optimize the evidence lower bound (ELBO):

$$
\begin{array} { r } {  { \mathcal { L } } ( \Phi , \Theta ) =  { \mathbb { E } } _ { q _ { \Phi } } \left[ \log p _ { \Theta } ( y _ { 1 : T } , x _ { 1 : T } \mid u _ { 1 : T } ) - \log q _ { \Phi } ( x _ { 1 : T } \mid y _ { 1 : T } , u _ { 1 : T } ) \right] . } \end{array}\tag{50}
$$

We adopt amortized variational inference and parameterize the approximate posterior using a recurrent neural network. Specifically, we use an LSTM-based recognition model to produce the mean and variance of a Gaussian approximate posterior over latent states at each time step. Latent variables are reparameterized as $\pmb { x } _ { t } = \pmb { \mu } _ { t } + \pmb { \sigma } _ { t } \odot \pmb { \xi } _ { t }$ , where $\xi _ { t } \sim \mathcal { N } ( 0 , I )$

To match the interventional structure from Eq. (2), we apply the mask ${ \mathbf { } } m _ { t }$ to the approximate posterior mean during training:

$$
\pmb { \mu } _ { t }  \pmb { m } _ { t } \odot \pmb { \mu } _ { t } + \pmb { B } \pmb { u } _ { t } .\tag{51}
$$

The model parameters and variational parameters are jointly optimized using stochastic gradient descent on the negative ELBO.

## C STATISTICAL ANALYSIS

Unless stated otherwise, each model is trained with 10 random seeds. Reported means are accompanied by the standard error of the mean (s.e.m.). When reporting per-trial performance in real-data experiments, we first average predictions across seeds for each held-out trial, then report the mean and s.e.m. across trials within each evaluation tier (e.g., control vs. stimulation). Otherwise, reported accuracy metrics are mean ± s.e.m. across seeds. Model comparisons use a two-sided paired Wilcoxon signed-rank test on seed-level scores $( n = 1 0 )$ . Where applicable, $p \mathrm { - }$ -values are Benjamini-Hochberg FDR-corrected for multiple comparisons.

## D FITZHUGH-NAGUMO EXPERIMENT DETAILS

We provide full details for the simulation experiments in Sec. 4.1.

## D.1 DATA GENERATION

We simulate the two-dimensional FHN system

$$
\begin{array} { r } { \dot { v } = v - \frac { v ^ { 3 } } { 3 } - w + I _ { \mathrm { e x t } } , \qquad \dot { w } = \frac { v + a - b w } { \tau } , } \end{array}\tag{52}
$$

with parameters $a { = } 0 . 7 , b { = } 0 . 8 , \tau { = } 1 2 . 5 , I _ { \mathrm { e x t } } { = } 0 . 5$ (oscillatory regime; true latent dimension $D { = } 2 )$ Trajectories have length $T { = } 5 0 \mathrm { s a t } \Delta t { = } 0 . 1 \mathrm { s } ,$ preceded by a 5-second burn-in that is discarded before training. We use 200 training trajectories (batch size 25) and 10 validation trajectories. Initial conditions are sampled as $\pmb { x } _ { 0 } \tilde { \mathbf { \alpha } } \sim \dot { \mathcal { N } } ( 0 , \sigma _ { 0 } ^ { 2 } \pmb { I } _ { D } )$ with $\sigma _ { 0 } { = } 0 . 0 5$ before the burn-in. Latent dynamics include additive Gaussian process noise with $\sigma _ { x } { = } 0 . 0 5 ;$ observations are noiseless $( \sigma _ { y } = 0 )$ .

Observation model. Observations are obtained via ${ \pmb y } _ { t } = \operatorname { t a n h } ( { \pmb C } { \pmb x } _ { t } )$ , where $C \in \mathbb { R } ^ { N \times D }$ is a fixed random matrix with orthonormal columns $( N { = } 4$ observations).

Interventions. Interventions are delivered on $M { = } 4$ input channels via random (Poisson) pulse stimulation: pulses arrive as a Poisson process at rate $\lambda { = } 0 . 2 0 \mathrm { s } ^ { - 1 }$ , each assigned to a uniformly random input channel, with pulse duration 0.25 s, amplitude drawn uniformly from $[ - 2 , 2 ]$ , and additive Gaussian noise $\sigma { = } 0 . 2 5$ . Of the training trajectories, 10% are non-interventional (control trials) with no stimulation applied.

## D.2 MODEL ARCHITECTURE

Linear and nonlinear variants share the same emission and recognition architectures and differ only in the dynamics parameterization.

Dynamics. The nonlinear model uses $g _ { \phi } ( \pmb { x } _ { t } ) = \mathrm { M L P } ( \pmb { x } _ { t } )$ , implemented as a two-hidden-layer network with 8 units per layer and GeLU activations. The linear model uses $g _ { \phi } ( \pmb { x } _ { t } ) = A \pmb { x } _ { t }$ , with $\pmb { A } \in \mathbb { R } ^ { D \times D }$ learned. Both share a learned input matrix $B \in \mathbb { R } ^ { D \times M }$ , initialized to the identity (zero-padded as needed) and subject to the B-sparsity mechanisms below.

B-sparsity mechanisms. Two mechanisms enforce sparsity on B. First, at each forward pass we treat entries with $| B _ { i j } | < \tau _ { B }$ as inactive when computing the hard-intervention mask $( \tau _ { B } \mathrm { = } 1 0 ^ { - 3 }$ fixed for all experiments), so tiny numerical residuals do not trigger spurious interventions. Second, during training we hard-threshold B elementwise after each optimizer step (outside backpropagation): entries with $| B _ { i j } | < \tau _ { t }$ are set to zero, where $\tau _ { t }$ increases linearly from 0 to $\tau _ { \mathrm { m a x } } { = } 0 . 1 0$ over the first $p _ { \tau } { = } 0 . 1 0$ fraction of the maximum number of epochs and is fixed thereafter. We use hard thresholding rather than an $\ell _ { 1 }$ penalty so that surviving entries are not shrunk toward zero while small residual entries are set to exact zero; dense support on B also violates the unordered-pairs condition needed for Corollary 4. The warmup avoids premature pruning early in training. On FHN (nonlinear iSSM, 10 seeds), disabling the schedule $( \tau _ { \operatorname* { m a x } } { = } 0 )$ reduced latent accuracy from $0 . 9 7 \pm 0 . 0 2$ to $0 . 7 4 \pm 0 . 0 4$ and degraded B recovery (paired Wilcoxon signed-rank test: both $\scriptstyle p = 0 . 0 0 4 )$ ; although a milder threshold $\tau _ { \mathrm { m a x } } { = } 0 . 0 5$ already matched the default model’s performance.

Emission. The emission model factorises into a linear readout and a nonlinear residual correction:

$$
f _ { \theta } ( \pmb { x } _ { t } ) = W _ { \mathrm { s k i p } } \pmb { x } _ { t } + \mathrm { M L P } _ { \theta } ( \pmb { x } _ { t } ) ,\tag{53}
$$

where $W _ { \mathrm { s k i p } } \in \mathbb { R } ^ { N \times D }$ is a learned linear readout and the nonlinear branch projects $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ to $H _ { \mathrm { e m } } { = } 3 2$ dimensions (linear $+ \mathrm { G e L U } )$ , passes it through $n _ { \mathrm { b l o c k s } } { = } 2$ pre-norm residual blocks (LayerNorm → Linear → GeLU → Linear with skip connection), and projects back to N. The nonlinear branch is initialised with near-zero output weights so the model starts approximately linear.

Recognition. The approximate posterior $p _ { \Phi } ( { \pmb x } _ { 1 : T } \ \mid \ { \pmb y } _ { 1 : T } , { \pmb u } _ { 1 : T } )$ is parameterized by a sharedbackbone LSTM: inputs $( \pmb { y } _ { t } , \pmb { u } _ { t } )$ are linearly projected to $H _ { \mathrm { r e c } } { = } 3 2$ dimensions (LayerNorm + GeLU), processed by a single-layer LSTM, and decoded by two separate heads for the posterior mean and log-variance.

## D.3 TRAINING

Models are trained to maximize the ELBO (standard $\beta { = } 1$ VAE objective) using Adam with learning rate $3 \times 1 0 ^ { - 3 }$ , gradient clipping (max norm 1.0), and $\ell _ { 2 }$ weight decay $1 0 ^ { - 3 }$ . The KL term is linearly warmed up over the first $p _ { \beta } { = } 0 . 2 0$ fraction of epochs; the training-time B-sparsity warmup follows the schedule above. Early stopping monitors rollout MAE on the validation set with patience 50 epochs, up to a maximum of 1000 epochs.

## D.4 EVALUATION PROTOCOL

All toy-model results in Secs. 4.1, 4.2, and 4.5 use the same evaluation protocol. After training, we score each seed on 100 newly simulated test trajectories from the true dynamical system. Using the training $\Delta t$ and burn-in $T _ { \mathrm { b u r n } } .$ , we simulate $T _ { \mathrm { b u r n } } + T _ { \mathrm { e n c o d e } } + T$ seconds per trial $( T _ { \mathrm { e n c o d e } } { = } 5 \mathrm { s }$ ${ \cal T } { = } 1 0 0 \mathrm { s } )$ and discard the burn-in.

(a)  
![](images/d23598b42aa83abad7c32782926e7279f23c01f5beb84443b51e6e6aa6d53aba.jpg)

![](images/6826c0d2e51b18f101a8dc0ae1bf210aef5d9bd1ffca4d5b02b98b19233528e6.jpg)

(d)  
![](images/83e1cce73011826f1a7bdf2ae77fcb95dd94087dc6e1c80d021c4eddff340821.jpg)

![](images/2463168df157a71c5ace3e56f4ac2b2a09c1396db1438486e585db6625a113be.jpg)

(e)  
![](images/13f34d10ece19d0255b0d90ca4d5652ac70d8e0d713711c506ac19739bc9a597.jpg)  
Figure 7: Nonlinear iSSM identifies Van Der Pol oscillator latent dynamics. (a,b) Open-loop rollout and aligned latent trajectories on a test trial: true (black), linear iSSM (blue), nonlinear iSSM (magenta); mean $\pm \ : s . 0 .$ across 10 seeds. Gray lines mark interventions. (c) Latent flow fields from the seed with highest Jacobian similarity to ground truth. Gray/black: true flow (arrows) and trajectory (line); blue/magenta: inferred linear/nonlinear flow and trajectories. (d) Median inferred input map B after permutation-scale alignment. (e) Latent-dimension sweep over $D \in \{ 1 , 2 , 3 , 4 \}$ shows that model selection picks out the true dimension $( D = 2 )$

Test interventions. Every test trial receives Poisson pulse stimulation. Pulse duration, input noise, and the amplitude interval are inherited from the training stimulator, with the amplitude range scaled by 1.2 to probe interventions stronger than those seen during training. The 100 test trials are distributed approximately evenly over $\bar { 7 }$ stimulation rates uniformly spaced in $[ 0 . 2 , 0 . 8 ] \mathrm { s } ^ { - 1 }$

Trial structure and scoring. Reconstruction, rollout, and latent accuracy (Sec. 3.3) are computed on the 100 s evaluation segment (after the encode window). We compute one Pearson correlation per trial then average over trials within each seed. Correlations use only non-interventional timesteps where all input channels are zero. Trials on which the model produces non-finite outputs (e.g. a diverged rollout) score zero.

Alignment. For latent accuracy we permutation-align inferred latents by Hungarian matching on the inferred and true input matrices $B ,$ then fit a per-dimension affine map (scale and shift) from the driven full-trial posterior at non-interventional timesteps.

## D.5 COMPUTATIONAL RESOURCES

All FHN experiments were run on CPU with 2 GB RAM. Per seed, training takes approximately 30-45 min (variation due to early stopping).

## E VAN DER POL OSCILLATOR EXPERIMENTS

The Van der Pol (VdP) oscillator with its square-like relaxation limit cycle has qualitatively different attractor geometry compared to FHN, but it also lies outside the linear iSSM hypothesis class.

We simulated VdP dynamics

$$
{ \dot { x } } _ { 1 } = x _ { 2 } ,
$$

$$
\dot { x } _ { 2 } = \mu ( 1 - x _ { 1 } ^ { 2 } ) x _ { 2 } - x _ { 1 } ,\tag{54}
$$

with $\mu \ = \ 2 \ ( D \ = \ 2 )$ latent dimensions). Model architecture and training follow Apps. D.2 and D.3. Data generation follows the same protocol as App. D.1, with the following deviations: $\Delta t { = } 0 . 0 5 \mathrm { s } ;$ process noise $\sigma _ { x } { = } 0 . 0 0 2 ;$ observation nonlinearity arsinh instead of tanh; stimulation rate $\lambda { = } 0 . 4 5 \mathrm { ^ { - 1 } s ^ { - 1 } }$ ; recognition hidden dimension $H _ { \mathrm { r e c } } { = } 6 4$ ; learning rate $2 \times 1 0 ^ { - 3 }$ ; maximum epochs 2000; and B-sparsity warmup fraction $p _ { \tau } { = } 0 . 0 5$

Observation nonlinearity. We use arsinh rather than tanh as the pointwise observation nonlinearity. The VdP oscillator at $\mu = 2$ produces oscillations with a substantially larger amplitude than FHN, and the random injective projection C can amplify this further. At such value ranges, tanh saturates and produces near-zero gradients at the extremes, creating an optimization challenge that is unrelated to identifiability. The arsinh nonlinearity is unbounded (growing as $\log ( 2 | x | )$ for large |x|), preserves injectivity, and avoids gradient saturation.

The VdP experiments replicate the FHN trends: The nonlinear iSSM again outperforms its linear counterpart (Fig. 7), achieving rollout accuracy of $r = 0 . 9 4 \pm 0 . 0 1$ s.e.m. versus $0 . 5 8 \pm 0 . 0 4$ and latent accuracy of $r = 0 . 9 3 \pm 0 . 0 1$ versus $0 . 5 5 \pm 0 . 0 3 ( \mathrm { F i g . \ 7 a , b } ;$ Wilcoxon signed-rank test: both $p = 0 . 0 0 2 0 )$ . The best nonlinear model recovers the characteristic square-like VdP attractor, whereas the best linear model produces a spiral attractor (Fig. 7c). Nonlinear iSSM also more faithfully recovers B (Fig. 7d). Model selection picks out the true dimension $( D = 2 )$ for both models (Fig. 7e).

## F DECISION-MAKING MODEL EXPERIMENT DETAILS

We provide full details for the simulation experiments in Sec. 4.2.

## F.1 DYNAMICAL SYSTEMS

We simulate three two-dimensional systems $\left( D = 2 \right)$ as follows.

Bistable well (nonlinear integration).

$$
\dot { x } _ { 1 } = a x _ { 1 } ^ { 3 } + b x _ { 1 } + c _ { \mathrm { o h } } , \qquad \dot { x } _ { 2 } = c x _ { 2 } + c _ { \mathrm { o h } } ,\tag{55}
$$

with $a = - 0 . 6 , b { = } 2 . 0 , c { = } - 1 . 0$ , and $c _ { \mathrm { o h } } { = } 0 . 1$ . Evidence is integrated via an unstable saddle separating two stable choice states.

## Line attractor (perfect integration).

$$
{ \dot { \pmb { x } } } = { \pmb { A } } { \pmb { x } } ,\tag{56}
$$

where ${ \bf \delta A } = { \mathrm { R o t } } ( \gamma ) R \Lambda R ^ { - 1 }$ has one marginally stable direction (eigenvalue 0) and one stable direction (eigenvalue $\approx - 1 . 2 1$ for the values below). Here $\pmb { \Lambda } = \mathrm { d i a g } \overline { { ( 0 , \lambda _ { 1 } ) } } , \pmb { R } = [ \pmb { r } _ { 0 } , \pmb { l } _ { 0 } ] ^ { \top }$ has rows $\boldsymbol { r } _ { 0 } ^ { \top }$ and $l _ { 0 } ^ { \top }$ , and Rot(γ) is a $2 \times 2$ rotation matrix. We use $l _ { 0 } = [ 1 , 1 ] ^ { \top } , r _ { 0 } = [ 1 , 0 ] ^ { \top }$ (each normalized), $\stackrel { \sim } { \lambda _ { 1 } } = - 1 . 0$ , and $\gamma { = } 4 5 ^ { \circ }$

Point attractor (input-driven dynamics).

$$
\begin{array} { r } { \dot { \pmb x } = \mathrm { d i a g } ( a _ { 1 } , a _ { 2 } ) \pmb x , } \end{array}\tag{57}
$$

with $a _ { 1 } = - \ : 0 . 5$ and $a _ { 2 } { = } - 1 . 0$ . Both latent dimensions decay to a fixed point at the origin, so decision-related activity is inherited from inputs rather than generated locally.

## F.2 DATA GENERATION

The three systems share the same observation and intervention protocol. Trajectories have length $\begin{array} { r } { T { = } 2 0 \mathrm { s } \mathrm { a t } \dot { \Delta } t { = } 0 . } \end{array}$ 1 s with no burn-in. We generate 40 training, 10 validation, and 100 test trajectories (batch size 20). All systems use D=2 latent dimensions, $N { = } 1 0$ observations, and $M { = } 2$ input channels. Latent dynamics include additive Gaussian process noise with $\sigma _ { x } { = } 0 . 1$ ; observations are noiseless $( \sigma _ { y } = 0 )$

Observation model. Observations are obtained via ${ \pmb y } _ { t } = \operatorname { t a n h } ( { \pmb C } { \pmb x } _ { t } )$ , where $C \in \mathbb { R } ^ { N \times D }$ is a fixed random matrix with orthonormal columns.

Interventions. Interventions are delivered on M=2 input channels via random pulse stimulation: pulses arrive as a Poisson process at rate $\lambda { = } 0 . 7 5 \mathrm { s } ^ { - 1 }$ , each assigned to a uniformly random input channel, with pulse duration 0.25 s, amplitude drawn uniformly from [−3, 3], and additive Gaussian noise $\sigma { = } 0 . 2 5$ . All training trajectories contain interventions.

## F.3 TRAINING

We fit linear and nonlinear iSSMs to interventional data from each system. Model architecture and training follow Apps. D.2 and D.3, with learning rate $5 \times 1 0 ^ { - 3 }$ and batch size 20. Linear and nonlinear variants differ only in the dynamics parameterization. We train 10 random seeds per model class and evaluate on held-out trajectories with novel interventions.

## F.4 COMPUTATIONAL RESOURCES

All decision-making experiments were run on CPU with 2 GB RAM. Per seed, training takes approximately 15–30 minutes (variation due to early stopping).

## G MACAQUE DLPFC DATASET EXPERIMENT DETAILS

## G.1 DATA

We use spike-count recordings from macaque prefrontal cortex (area 8Ar) collected during restingstate micro-stimulation experiments (Nejatbakhsh et al., 2023). Data were recorded at $f _ { s } { = } 3 0$ Hz across N=96 channels. Trials are extracted by windowing ±1 s around each stimulation event (stimulation trials) or around randomly sampled time points matched in count (control trials), giving T=60 bins per trial at $\Delta t { = } 3 3$ ms. Both conditions are used during training.

Input construction. The stimulation input $\pmb { u } _ { t } \in \mathbb { R } ^ { M }$ is a binary indicator that is one at the stimulated electrode’s channel during stimulation and zero elsewhere. We compress u to only the $M { = } 1 2$ electrodes that were ever stimulated across all sessions, so that each entry of u has a direct causal interpretation. Control trials have $\mathbf { \nabla } { \mathbf { } } u _ { t } = 0$ throughout. Train, validation, and test sets are split with stratification by trial type (stimulation vs. control) to ensure both conditions are represented in each split; 10 trials are reserved for validation and 60 for testing.

## G.2 MODEL ARCHITECTURE

The linear and nonlinear models use the same implementation as the simulation experiments (App. D.2), with the dataset-specific modifications below. They differ only in their dynamics parameterization.

Likelihood. Because observations are spike counts, we use a Poisson likelihood: $p ( \pmb { y } _ { t } \mid \pmb { x } _ { t } ) =$ Poisson $\left( \exp ( f _ { \theta } ( \pmb { x } _ { t } ) ) \right)$ applied independently per channel, where $f _ { \theta } ( \pmb { x } _ { t } )$ outputs log-rates.

Dynamics. The nonlinear model uses $g _ { \phi } ( \pmb { x } _ { t } ) = \mathrm { M L P } ( \pmb { x } _ { t } )$ , implemented as a three-hidden-layer network with 64 units per layer and GeLU activations. The linear model uses $g _ { \phi } ( { \pmb x } _ { t } ) = { \pmb A } { \pmb x } _ { t } ,$ with $\pmb { A } \in \mathbb { R } ^ { D \times D }$ learned. Both share a learned input matrix $B \in \mathbb { R } ^ { D \times M }$ , initialized to an identity-like matrix with small perturbations and subject to the B-sparsity mechanisms described in App. D.2. We sweep the latent dimension over $\bar { D \in \{ 5 , 1 2 , 1 5 , 2 \bar { 0 } \} }$ and report D=12 in the main text. The rollout advantage of the nonlinear iSSM holds at every D (Fig. 8).

Emission. We use the residual-MLP emission described in App. D.2, with hidden dimension $H _ { \mathrm { e m } } { = } 3 2$ , two residual blocks, and dropout probability 0.1.

Recognition. The approximate posterior $q _ { \Phi } ( { \pmb x } _ { 1 : T } \ \mid \ { \pmb y } _ { 1 : T } , { \pmb u } _ { 1 : T } )$ is parameterized by a bidirectional shared-backbone LSTM with hidden dimension $H _ { \mathrm { r e c } } { = } 3 2$ , input dropout probability 0.1, and separate linear heads for the posterior mean and log-variance. Because the recognition network is applied offline to the full trial, bidirectionality is appropriate and improves posterior quality. During training, each neuron-level observation is independently masked with probability 0.1.

![](images/b1d8dd95fd1374b18bff54cd985c753208ae1c761ebe554c766958706a098dd4.jpg)

(b)  
![](images/3aa3b949989db63388404e736a57e2681a20aaddccfd7b17216f3d7d9ab59b7a.jpg)  
Figure 8: Accuracy versus latent dimension D. (a) Reconstruction and (b) rollout accuracy for linear (blue) and nonlinear (magenta) iSSMs; mean ± s.e.m. across 10 seeds. Asterisks mark significant differences $( p < 0 . 0 5$ , FDR-corrected).

## G.3 TRAINING

We use the same training loop and ELBO objective as in App. D.3. We use Adam with learning rate $3 \times 1 0 ^ { - 3 }$ , batch size $2 5 , \ell _ { 2 }$ weight decay $1 0 ^ { - 2 }$ , and gradient clipping at maximum norm 1.0. The KL term is warmed up linearly over the first $p _ { \beta } { = } 0 . 2 5$ fraction of epochs, while the trainingtime B-sparsity threshold increases to $\tau _ { \mathrm { m a x } } { = } 0 . 0 5$ over the first $p _ { \tau } { = } 0 . 5 0$ fraction. Early stopping monitors rollout MAE on the validation set with patience 100 epochs, up to a maximum of 3000 epochs. These settings are shared by the linear and nonlinear models.

## G.4 EVALUATION

Reconstruction and rollout accuracy are computed on held-out test trials. Reconstruction encodes the full trial with the bidirectional recognition network (App. G.2). For rollouts, the encoder sees only pre-stimulation observations and inputs: a fixed window of 31 timesteps (≈ 1.0 s at 30 Hz), set to the earliest stimulation onset in the test set and shared across all trials including controls with $\mathbf { \boldsymbol { u } } _ { t } ~ = ~ 0$ throughout. $\scriptstyle { \pmb x } _ { 0 }$ is taken from the start of that window and the dynamics are integrated open-loop with the known test inputs over the full trial. Rollout accuracy is the Pearson correlation between predicted and true observations, with the same channel masks as reconstruction.

## G.5 UNORDERED PAIRS ESTIMATE

Permutation-scale-shift identifiability (Corollary 4) requires the unordered pairs condition (Assumption 2). To assess the degree to which the learned intervention design implements this condition, we read the τ<sub>B</sub>-thresholded support of the trained B matrix $( \tau _ { B } { = } 1 0 ^ { - 3 }$ ; App. D.2). Let $G _ { d k } = \mathbb { I } [ | B _ { d k } | \ge \tau _ { B } ]$ indicate whether an intervention via channel k drives latent coordinate d. Channel k separates pair $( i , j )$ iff it drives exactly one of the two coordinates, i.e. $G _ { i \cdot } \neq G _ { j }$ . We report the fraction of separately intervenable pairs as:

$$
\mathsf { u p \_ f r a c } = \frac { \# \{ ( i , j ) , i < j : G _ { i } \neq G _ { j } . \} } { \binom { D } { 2 } } ,\tag{58}
$$

computed per seed and summarised as mean $\pm \ : \mathrm { s . e . m }$ . across seeds.

For the main nonlinear macaque model $( D = 1 2 )$ , up $\mathrm { f r a c } = 9 3 . 3 \pm 1 . 3 \%$ (mean ± s.e.m. across 10 seeds). Note that this estimate is not a validation of the unordered pairs condition, as we have no ground-truth latent interventions.

## G.6 COMPUTATIONAL RESOURCES

All experiments were run on CPU with 8 GB RAM. Training a single model run takes approximately 3–6 hours depending on early stopping; both the linear and nonlinear iSSM were trained with the same budget.

## H HUMAN IEEG DATASET EXPERIMENT DETAILS

## H.1 DATA

We use the human iEEG single-pulse electrical stimulation dataset of Huang et al. (2023) (Open-Neuro ds004457) for subject sub-2. Continuous SEEG was recorded at $f _ { s } { = } 2 0 4 8$ Hz across 170 channels; biphasic single pulses were delivered to bipolar contact pairs at 4 or 6 mA. Following the event filter of Huang et al. (2023) (status=good), the preprocessed release contains 529 pulses at 60 bipolar stimulation sites.

Preprocessing. We replicate the signal-processing pipeline of Huang et al. (2023). For each pulse we extract a [−1, +2] s window and apply, in order: (i) stimulation-artefact interpolation over [−2, +15] ms (PCHIP spline fitted to surrounding clean samples), (ii) common-average reference over good non-stimulated channels, (iii) 60 Hz line-noise removal by spectrum fitting, and (iv) baseline correction (median over [−0.5, −0.05] s).

Channel and trial selection. We drop clinically bad channels, channels without anatomical location, white-matter contacts, and white-matter stimulation sites (and trials at removed sites), leaving N=99 recording channels, M=50 stimulation sites, and 437 trials. Observations are decimated to 256 Hz with an anti-aliasing filter, cropped to [−50, +200] ms around the pulse (T=64 bins at ∆t=3.9 ms), and standardized per channel.

Input construction. The stimulation input $\pmb { u } _ { t } \in \mathbb { R } ^ { M }$ is a single-sample impulse at pulse onset in the coordinate of the stimulated site, with amplitude equal to the delivered current (rescaled so the maximum maps to one) and zero elsewhere.

Masking and artefact trials. We exclude the two contacts of the stimulated pair from the likelihood for the whole trial (they saturate around the pulse). App. A.6 shows that identifiability is preserved under this masking, given modest additional conditions. The same whole-trial stimcontact masks are used when scoring validation and test trials; masked channels are excluded from per-channel standardization and zero-filled before the recognition network. Additionally, 14 trials flagged by a joint variance and template-correlation criterion as having residual onset artefacts on any kept channel beyond the active stimulation pair are dropped, leaving 423 trials.

Splits. Every trial is a single pulse at one site, so we stratify by stimulation site. We hold out 10 trials for validation and 30 for testing, and train on the remaining 383. The test set has two tiers: a held-out-site tier with all 10 trials stimulating the right-hippocampal pair RC3-RC4 (clinically labelled irritative zone), excluded from training and validation; and a held-in tier with the remaining 20 test trials from sites seen during training.

## H.2 MODEL ARCHITECTURE

The models follow the macaque setup (App. G.2) with the modifications below; linear and nonlinear variants differ only in their dynamics parameterization.

Likelihood. Observations are continuous voltages, so we use a multivariate Gaussian likelihood $p ( \pmb { y } _ { t } \ | \ \mathbf { \delta x } _ { t } ) = \mathcal { N } ( f _ { \theta } ( \mathbf { x } _ { t } ) , L L ^ { \top } )$ with a learned full covariance: L is lower triangular with perchannel scales on the diagonal and learned off-diagonal entries below.

Dynamics. As in App. $_ { \mathrm { G . 2 , } }$ except for the B parameterization (see below); we sweep $D \in$ {15, 20, 25, 30} and report $D { = } 2 5$ in the main text. The rollout advantage of the nonlinear iSSM holds at every D (Fig. 9).

![](images/6a61cb128a2b881db3cf4995aa208f564108fbc69fcf3338d44e8ced0511d5e6.jpg)

(b)  
![](images/57149528bdd683dfb50821a12895a9e9b8ab0e396284c292b03ae9c3a0e86d25.jpg)  
Figure 9: Accuracy versus latent dimension D. (a) Reconstruction and (b) rollout accuracy for linear (blue) and nonlinear (magenta) iSSMs; mean ± s.e.m. across 10 seeds. Asterisks mark significant differences $( p < 0 . 0 5$ , FDR-corrected).

Site-embedding intervention matrix. Unlike the macaque models, B is not a free parameter: columns are predicted from electrode coordinates via a site-embedding network. Let $s _ { m } \in$ $\mathbb { R } ^ { 3 }$ be the midpoint of bipolar site m in the electrode coordinates provided with the dataset (electrodes.tsv). We center and isotropically scale the coordinates (root-mean-square deviation over all sites and axes) and map them through a shared MLP:

$$
b _ { m } = \mathrm { M L P } _ { \kappa } \bigg ( \frac { s _ { m } - \bar { s } } { \sigma } \bigg ) \in \mathbb { R } ^ { D } , \qquad B = [ b _ { 1 } , \ldots , b _ { M } ] \in \mathbb { R } ^ { D \times M } .\tag{59}
$$

${ \mathrm { M L P } } _ { \kappa }$ has one hidden layer of 32 units with GeLU activation; the output layer is initialized with Gaussian weights scaled by 0.1. This parameterization of B by coordinates lets us predict responses at sites that were never stimulated during training, and, more speculatively, at sites that were never stimulated at all (App. H.4). We note that this goes beyond our identifiability results, which constrain the input map only on the support of the observed inputs (Remark 1). Generalisation to unseen sites instead relies on the smoothness of the coordinate parameterisation, which we evaluate empirically.

Emission. We use the residual-MLP emission of App. D.2 with hidden dimension $H _ { \mathrm { e m } } { = } 1 2 8 .$ three residual blocks, and dropout probability 0.1.

Recognition. As in App. G.2, with $H _ { \mathrm { r e c } }$ =64 and input dropout probability 0.2.

## H.3 TRAINING

Training follows App. G.3, with learning rate $3 \times 1 0 ^ { - 3 }$ , batch size $^ { 3 2 , }$ and no training-time Bsparsity warmup (since B is computed from coordinates; the forward-pass threshold $\tau _ { B }$ still applies). We train 10 random seeds per latent dimension. For the nonlinear model reported in the main text $( D = 2 5 )$ , the unordered-pairs estimate (App. G.5) is up\_frac $= 4 4 . 3 \pm 3 . 3 \%$ (mean ± s.e.m. across 10 seeds). Evaluation follows App. G.4, with a fixed encoder window of 12 pre-stimulation timesteps (≈ 47 ms at ∆t=3.9 ms).

## H.4 VIRTUAL STIMULATION OF NEVER-STIMULATED SITES

We synthesize 77 candidate bipolar pairs from adjacent recorded contacts on the same lead (33 never stimulated in the data), annotate each with the most severe clinical label of its two contacts, and group them into epileptogenic $( n { = } 2 6 ;$ seizure-onset, early-propagation, or irritative zone) and healthy $\left( n { = } 5 1 \right)$ pairs. Labels enter only at analysis time. For each pair we obtain a virtual column b from its midpoint coordinates and run two rollouts from each of 100 initial states (sampled from encoded latents at pulse onset across trials): one with a unit impulse along b at the first step, one autonomous. Then we difference the two to remove shared baseline drift. Both trajectories are decoded and we take $\| \Delta \pmb { y } ( t ) \|$ ∥, averaged over initial states over a 50-step (195 ms) window. From this curve we report the peak (response amplitude) and the mean from 100 ms to window end (response persistence). We repeat this analysis for each of the ten random seeds from model training and aggregate the resulting readouts before testing. Epileptogenic and healthy pairs are compared with Mann-Whitney U tests, with Benjamini-Hochberg correction over the two readouts and two models.

Under the nonlinear iSSM, epileptogenic pairs produce responses that are both larger (Cohen’s $d =$ 1.08, $p < 0 . 0 0 0 1 )$ and more sustained $( d = 0 . 7 5 , p = 0 . 0 0 6 0 )$ than healthy pairs, matching the clinical expectation that seizure-onset tissue is hyperexcitable and hyperconnected; the linear iSSM shows neither effect (Fig. 10).

(a)  
![](images/0aac965a1f9d6d6b9c821813424ccd8af790babead784c0fd0e0f42a861b503f.jpg)

(b)  
![](images/93eb89c4450f4ef8f40dd5f16a9ed42d142732c7da73ffaf3f36d3b2b9549e28.jpg)  
Figure 10: Virtual stimulation of clinically annotated sites. (a) Peak and (b) sustained networkwide response $\| \Delta \boldsymbol { y } \|$ to virtual stimulation of healthy $( n = 5 1 )$ and epileptogenic $( n = 2 6 )$ bipolar pairs, under the linear and the nonlinear iSSM. Mann-Whitney U tests, FDR-corrected over the two readouts and two models; ∗ ∗ ∗: $p < 0 . 0 0 1$

Please note that we treat this as a pilot only. Input columns for many virtually stimulated sites are far from any sites actually stimulated in the dataset (Fig. 5a), and outside the scope of our identifiability results (Remark 1). Also, the present analysis uses only one subject and without independent replication.

## H.5 COMPUTATIONAL RESOURCES

All experiments were run on CPU, with 4 cores and 6 GB RAM per run for ∼2–4 hours.

## I STIMULATION PROTOCOL EXPERIMENT DETAILS

We provide full details for the stimulation protocol design experiments in Sec. 4.5. Both analyses use the bistable well of App. F.1 and vary one property of the intervention set at a time, holding the model class fixed.

## I.1 DIVERSITY OF INTERVENTIONS

Data generation. Data generation follows App. F.2; only stimulator timing and amplitude parameters are varied, in a $2 \times 2$ factorial crossing pulse timing (regular or Poisson) with pulse amplitude (constant or diverse). Under Poisson timing, pulses arrive as a Poisson process at rate $\lambda { = } 0 . { \overset { } { 7 } } 5 \mathrm { s } ^ { - 1 }$ each assigned to a uniformly random input channel. Under regular timing, pulses cycle round-robin over the input channels with a fixed rest interval $d _ { \mathrm { r e s t } } = 1 / \lambda - d _ { \mathrm { s t i m } } = 1 . 0 8 3 \mathrm { s }$ , so the two timings deliver the same expected number of pulses and differ only in their placement. Under diverse amplitudes, each pulse amplitude is drawn independently from $\mathcal { U } [ - 3 , \bar { 3 } ]$ . Under constant amplitudes, a single amplitude is drawn once per run, with magnitude $\mathcal { U } [ 1 . \dot { 5 } , 3 ]$ and random sign, and used for every pulse in the run; amplitude and sign only vary across seeds.

Training. We train nonlinear iSSM only. Model architecture and training follow $\operatorname { A p p s } .$ . D.2 and D.3, with the settings of App. F.3. We train 10 random seeds for each of the four cells.

Point attractor

Computational resources. All runs were performed on CPU (2 cores, 2 GB RAM).

## I.2 REPLICATION ON POINT AND LINE ATTRACTORS

We repeat the intervention-count sweep and the amplitude-timing factorial of Sec. 4.5 on the point and line attractor systems of App. F.1, with the same training protocol as above. The qualitative trends match the bistable well (Fig. 6): latent accuracy rises with more interventions, diverse amplitudes and irregular timing improve recovery, and rollout accuracy tracks latent accuracy more closely than reconstruction does (Fig. 11).

![](images/6c700c0345627bddcb9a3328f01475d641eeb890a2301f29dabacde923d2b533.jpg)

(b)  
![](images/2e6e2ff9f4c17bd854bdb35e78240d271e23395c74254246ee0a4cc08b49f733.jpg)

(c)  
![](images/2efd6b2dc8fbec7b0430dee8b9f94c3fc60c391ae672e8be7c7c908700972333.jpg)

![](images/38f9c91593672c5612b0028bcf6536e886957ce2a3d21e23c58e3b542ff5b53a.jpg)

(e)  
![](images/016d2c5f63e4a36d6e28527e75c611c76c838f78fc893e00ea5a32df996da301.jpg)

(f)  
![](images/4b3c17f915aaa4912a8715d9f1588e70c4c4275262b5c5ea4299427e3b528f90.jpg)  
Figure 11: Intervention-design trends replicate on point and line attractors. Top row (a-c): point attractor; bottom row (d-f): line attractor. Panel layout matches Fig. 6: (a,d) latent accuracy versus intervention count; (b,e) amplitude-timing factorial (10 seeds); (c,f) latent accuracy versus rollout (black) and reconstruction (gray); dashed line: identity.

## J INTERVENTION HARDNESS

Theorem 3 assumes hard interventions that decouple driven latents from the past state $\left( \operatorname { E q . } \left( 2 \right) \right)$ We view intervention hardness as a gradual property that depends on the particular intervention technique. We therefore empirically test the robustness of latent identification to gradual violation of this assumption.

We sweep the data-generating intervention from a purely additive drive $( \alpha = 0 )$ to a perfectly hard clamp $( \alpha = 1 )$ in the bistable well system. We compared the iSSM against a standard SSM and an input-conditioned variant whose nonlinear drift takes $\mathbf { \Delta } \mathbf { u } _ { t }$ as an argument – the most expressive of the three, as its hypothesis class contains every α regime.

On the bistable well (Fig. 12a), latent accuracy reveals two patterns. First, iSSM identification improves gradually, not stepwise, as interventions harden; though it is poor under purely additive drives $( \alpha = 0 )$ when the hardness assumption is completely violated. Second, across all α, only iSSM ever recovers the true latents near-perfectly: at $\alpha = 1$ it reaches 0.98, significantly above either baseline measured in its own ideal regime $( p = 0 . 0 0 2 )$ ). We replicate this analysis on FitzHugh-Nagumo dynamics (Fig. 12b), finding qualitatively similar results.

(a) Bistable well  
![](images/3cfd37560303e2799728f426198f6681734cccbc55bb40f2325081c15632dd21.jpg)  
Intervention hardness (®)

(b) FitzHugh-Nagumo  
![](images/4c8890dc284bd115a34ed983d39528bfdc1a1e1387cb37216621b6625ff06261.jpg)  
Intervention hardness (®)  
Figure 12: Latent accuracy under softened interventions and model comparison. Sweep over the intervention hardness parameter α of the data-generating process, from a purely additive drive $( \alpha = 0 )$ to a hard clamp $( \alpha = 1 )$ , comparing nonlinear iSSM to two baselines. (a) Bistable well (App. J.1). (b) FitzHugh-Nagumo replication (App. J.2). Asterisks indicate α-regimes where iSSM significantly outperforms both baselines $( p < 0 . 0 5 )$

## J.1 BISTABLE WELL

Data generation. Data generation follows App. F.2, with 40 training and 10 validation trajectories (batch size 20), and with the hardness of the data-generating intervention swept. Let ${ \pmb x } _ { t + 1 } ^ { \mathrm { f r e e } } = g ^ { * } ( { \pmb x } _ { t } ) + { \pmb \varepsilon } _ { t }$ denote the state that the system would have reached without stimulation, where $g ^ { * }$ is the true one-step transition map. At every driven latent entry we blend the clamp with these free dynamics,

$$
\pmb { x } _ { t + 1 } = \pmb { m } _ { t } \odot \pmb { x } _ { t + 1 } ^ { \mathrm { f r e e } } + ( \pmb { 1 } - \pmb { m } _ { t } ) \odot \big ( \pmb { B } \pmb { u } _ { t } + ( 1 - \alpha ) \pmb { x } _ { t + 1 } ^ { \mathrm { f r e e } } \big ) ,\tag{60}
$$

with the intervention mask $m _ { t , j } = \mathbb { I } \{ ( B \pmb { u } _ { t } ) _ { j } = 0 \}$ of Eq. (2); undriven entries are unaffected. At $\alpha { = } 1$ the driven entry is overwritten by $\mathbf { \nabla } _ { B u _ { t } }$ , recovering the hard intervention that Eq. (2) assumes; at α=0 the input is added on top of the free dynamics and the driven latent retains its dependence on the past state. We sweep $\alpha \in \{ 1 , 0 . 6 7 , 0 . 3 3 , 0 \}$ . All other data properties are held at the values of App. F.2.

Model variants. The three models differ only in how the input $\mathbf { \Delta } \mathbf { u } _ { t }$ enters the latent transition; emission, recognition, likelihood and initial-condition modules are identical across arms. Writing ${ \mathrm { M L P } } _ { \phi }$ for the drift network of App. D.2, the transitions are

$$
\mathrm { n o n l i n e a r \ i S S M : } \qquad \quad \pmb { x } _ { t + 1 } = \pmb { m } _ { t } \odot \mathrm { M L P } _ { \phi } ( \pmb { x } _ { t } ) + \pmb { B } \pmb { u } _ { t } + \varepsilon _ { t } ,\tag{61}
$$

$$
\mathrm { s t a n d a r d ~ S S M : } \qquad \quad \mathbf { x } _ { t + 1 } = \mathrm { M L P } _ { \phi } ( \mathbf { x } _ { t } ) + B \mathbf { u } _ { t } + \varepsilon _ { t } ,\tag{62}
$$

$$
\mathrm { i n p u t - c o n d i t i o n e d S S M : }
$$

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \mathrm { M L P } _ { \phi } ( \left[ \pmb { x } _ { t } , \pmb { u } _ { t } \right] ) + \pmb { \varepsilon } _ { t } , } \end{array}\tag{63}
$$

$$
\mathrm { w i t h } m _ { t , j } = \mathbb { I } \{ ( B \boldsymbol { } \boldsymbol { u } _ { t } ) _ { j } = 0 \} \mathrm { a s ~ a b o v e . }
$$

Nonlinear iSSM is misspecified at $\alpha \neq 1 ;$ ; standard SSM is misspecified at $\alpha \neq 0 ;$ the inputconditioned SSM is never misspecified.

Training. Model architecture and training follow Apps. D.2 and D.3, with the settings of App. F.3.   
We train 10 random seeds for each (model, α) condition.

Evaluation. Evaluation follows App. D.4, with one exception: since the input-conditioned SSM has no B, we index-align latents from latent trajectories rather than B columns for all models in this experiment.

Computational resources. All runs were performed on CPU (4 cores, 2 GB RAM for training and 4 GB for evaluation), with per-seed training cost comparable to App. F.4.

## J.2 FITZHUGH-NAGUMO REPLICATION

Data generation. Data generation follows App. D.1, with the intervention hardness blended as in Eq. (60) and swept over $\bar { \alpha ^ { \prime } } \in \{ 1 , 0 . 6 7 , 0 . 3 3 , 0 \}$ . All other data properties match App. D.1.

Model variants, training, and evaluation. We use the same three model classes (Eqs. (61)–(63)), training protocol (App. D.3), and evaluation protocol (App. D.4, with index alignment for the inputconditioned SSM) as in App. J.1. We train 10 random seeds for each (model, α) condition.

Computational resources. All runs were performed on CPU (4 cores, 2 GB RAM for training and 4 GB for evaluation), with per-seed training cost comparable to App. D.5.