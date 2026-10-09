# Beyond Distributional Fidelity: Causal-Penalized Diffusion for Synthetic Tabular Data

Lan Tao<sup>1</sup> lantao@ucla.edu

Yongxian He <sup>2</sup> yoh017@ucsd.edu

Shirong Xu<sup>3</sup> shirongxu5566@xmu.edu.cn

Yidong Ouyang<sup>1</sup> yidongouyang@g.ucla.edu

Guang Cheng<sup>1</sup> guangcheng@ucla.edu

<sup>1</sup> Department of Statistics & Data Science, University of California, Los Angeles <sup>2</sup> Herbert Wertheim School of Public Health and Human Longevity Science, University of California, San Diego

<sup>3</sup> Wang Yanan Institute for Studies in Economics, Xiamen University

## Abstract

Synthetic tabular generators are commonly optimized for distributional fidelity, but statistical similarity alone does not guarantee preservation of causal effects. In this paper, we study whether causal fidelity can be improved directly within a fully generative tabular model. Causal Fidelity is defined with respect to a target estimand as the discrepancy between inferential distributions obtained from real and synthetic data, and theoretical results show that high statistical fidelity does not generally imply high causal fidelity. We then propose a causal-fidelity-aware training framework which adds a causal discrepancy penalty to the generative objective. The framework is instantiated with a causal-penalized TabDDPM and optimized using an on-policy score-function estimator. We further establish conditions under which causal regularization improves expected causal fidelity. Experiments across diverse treatment-effect simulations and two benchmark datasets evaluate the ability of our method to improve causal fidelity while preserving competitive statistical fidelity.

## 1 Introduction

Synthetic tabular data have become increasingly important for data sharing, privacy protection, data augmentation, and downstream machine learning. Recent generative models, including generative adversarial networks, variational autoencoders, and diffusion-based models, have substantially improved the realism of synthetic tabular data. Accordingly, the quality of synthetic data is commonly assessed through distributional fidelity, measuring how closely the generated data reproduce the statistical properties of the original population (Jordon et al., 2022). Existing evaluation criteria range from feature-wise and pairwise discrepancies, to multivariate distributional distances (Herurkar et al., 2025; Stoian et al., 2025; Hernandez et al., 2025). While these criteria provide useful information about statistical similarity, they do not necessarily determine whether synthetic data preserve the information required for downstream causal inference.

Recent work has begun to examine whether synthetic tabular data preserve treatment-effect information beyond conventional statistical fidelity (Amad et al., 2026; Xu, 2026). To motivate this question empirically, we conduct a preliminary comparison of representative tabular generators. As shown in Figure 1, their ability to preserve the average treatment effect (ATE) posterior varies substantially across model families. CTGAN exhibits severe distortion of the ATE posterior, TVAE reduces but does not eliminate the discrepancy, and TabDDPM provides the strongest baseline while still showing a systematic shift from the reference posterior. These results suggest that causal information is not reliably preserved by standard tabular generators and motivate incorporating causal fidelity directly into the generative objective.

![](images/7cdd9d8af91f0a01d769842aa6b17e345d21e2b0fa0a23403f8bf732cdb1bd6e.jpg)  
(a) CTGAN  
(b) TVAE  
(c) TabDDPM  
Figure 1: ATE posterior preservation for CTGAN, TVAE, and TabDDPM. For each generator, 10 independent models are trained on the same reference dataset and 10 synthetic datasets are sampled from each model, yielding 100 synthetic datasets per generator. CausalPFN is used to infer the ATE posterior from both the reference and synthetic datasets. Red denotes the reference posterior and blue denotes the synthetic-data posteriors.

Existing approaches mainly address this issue through causal-aware evaluation and structured generation (Amad et al., 2026), or hybrid data-generation strategies (Xu, 2026). In contrast, we ask whether causal fidelity can be improved directly within the training objective of a fully generative tabular model.

To this end, we first formalize causal fidelity with respect to a target causal estimand as the discrepancy between inferential distributions obtained from real and synthetic data. We then propose a causal-fidelity-aware training framework that augments the standard generative objective with a causal discrepancy penalty. Because this discrepancy need not be differentiable with respect to the generator parameters, we optimize it using a score-function estimator, enabling a broad class of causal inference operators to be incorporated without requiring end-to-end differentiability.

Contributions. Our main contributions are summarized as follows:

• We formalize causal fidelity as the discrepancy between inferential distributions of a target causal estimand from real and synthetic data, and theoretically show that high distributional fidelity does not generally guarantee ATE preservation.

• We propose a causal-fidelity-aware training framework, which adds a causal discrepancy penalty to the generative objective, and instantiate it as an on-policy causal-penalized Tab-DDPM using a score-function estimator.

• We evaluate the proposed method across diverse treatment-effect simulations and two realworld benchmarks, showing that causal regularization improves causal fidelity while maintaining competitive distributional fidelity.

## 2 Related Works

In the following, we review three strands of literature most relevant to our study: (1) generative models for synthetic tabular data, (2) synthetic data for causal inference, and (3) causal effect estimation.

## 2.1 Generative Models for Synthetic Tabular Data

A wide range of deep generative models has been developed for synthetic tabular data. Earlier work includes GAN-based models such as CTGAN (Xu et al., 2019) and TableGAN (Park et al., 2018), together with VAE-based approaches such as TVAE (Xu et al., 2019) and RTVAE (Akrami et al.,

2020). Autoregressive language-model-based generators have also been explored for tabular synthesis, such as GReaT (Borisov et al., 2022). More recently, diffusion-based models have emerged as a competitive paradigm for tabular synthesis, including TabDDPM (Kotelnikov et al., 2023) and TabSyn (Zhang et al., 2024).

Another line of work incorporates task-specific feedback into tabular generation. ReTabSyn (Lin et al., 2026) introduces a reinforcement-learning-based tabular synthesis framework to preserve feature correlations and predictive signals, whereas our method directly adds a causal discrepancy penalty to the generative objective to improve causal fidelity.

## 2.2 Synthetic Data for Causal Inference

Recent work has begun to examine whether synthetic tabular data preserve treatment-effect information beyond conventional statistical fidelity. Amad et al. (2026) are among the first to systematically study synthetic tabular data for downstream treatment-effect analysis, and propose STEAM, a causal-aware framework for generating synthetic data in medical treatment-effect settings. Xu (2026) demonstrates that high distributional fidelity can coexist with distorted ATE estimates, and propose a hybrid synthesis framework which separates covariate generation from treatment and outcome modeling. CausalWrap (Asiaee et al., 2026) applies a model-agnostic post-hoc correction using differentiable causal constraint functionals to improve structural and treatment-effect fidelity. In contrast, our work uses a score-function estimator, allowing non-differentiable causal discrepancy measures to be incorporated directly into generator training.

## 2.3 Causal Effect Estimation

Classical methods for causal effect estimation include outcome regression or g-computation (Robins, 1986), inverse probability weighting (IPW) (Horvitz and Thompson, 1952), augmented IPW (AIPW) (Robins et al., 1994), and targeted maximum likelihood estimation (TMLE) (Van Der Laan and Rubin, 2006). Bayesian tree-based approaches further provide posterior uncertainty over treatment effects, including BART for causal inference (Hill, 2011) and Bayesian Causal Forests (Hahn et al., 2020). More recently, amortized causal inference and causal foundation models have enabled fast, uncertainty-aware inference over causal quantities, including CausalPFN (Balazadeh et al., 2025), Do-PFN (Robertson et al., 2026), and CausalFM (Ma et al., 2026). Such methods are well aligned with our causal-fidelity-aware training framework, which measures fidelity through discrepancies between inferential distributions from real and synthetic data. In our experiments, we instantiate the causal inference operator using CausalPFN.

## 3 Problem Setup and Modeling

## 3.1 Preliminaries

We adopt the potential outcomes framework for causal inference (Imbens and Rubin, 2015). Let $X \in { \mathcal { X } }$ denote covariates, $W \in \{ 0 , 1 \}$ a binary treatment, and $Y \in \mathcal { V }$ the observed outcome. For each unit, define potential outcomes $\dot { Y } ( 0 )$ and ${ \bf \bar { \cal Y } } ( 1 )$ corresponding to the outcomes under control and treatment, respectively. However, only one of these is observed for each unit, yielding the fundamental problem of causal inference. The individual treatment effect is defined as $\tau _ { i } = Y _ { i } ( 1 ) - Y _ { i } ( 0 )$ which is generally unidentifiable. Consequently, inference focuses on population-level quantities. For example, the Average Treatment $E f f e c t ( A T )$

$$
\tau : = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) ] ,
$$

and the Conditional Average Treatment Effect (CATE),

$$
\tau ( x ) : = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ] .
$$

Identification from observational data relies on standard assumptions: (i) consistency $Y = Y ( W )$ (ii) unconfoundedness $( Y ( 0 ) , Y ( 1 ) ) \perp W \mid X$ , and (iii) overlap $0 < \mathbb { P } ( W = 1 \mid \overline { { \boldsymbol X } } = \boldsymbol x ) < \dot { 1 }$ Under these assumptions, the ATE is identified from the observed-data distribution through

$$
\tau = \operatorname { \mathbb { E } } _ { X } \left[ \operatorname { \mathbb { E } } ( Y \mid W = 1 , X ) - \operatorname { \mathbb { E } } ( Y \mid W = 0 , X ) \right] .
$$

## 3.2 Problem Statement

Let $D _ { \mathrm { r e a l } } = \{ ( X _ { i } , W _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ denote an i.i.d. dataset drawn from an unknown distribution $\mathcal { P }$ over covariates $X \in { \mathcal { X } }$ , treatment $\dot { W } \in \{ 0 , 1 \}$ , and outcome $Y \in \mathcal { V }$ . We consider a generative model $G _ { \theta }$ parameterized by θ, which is trained on $D _ { \mathrm { r e a l } }$ to learn an approximation $\hat { \mathcal { P } } _ { \theta }$ of the data-generating distribution. A synthetic dataset of size $n _ { \mathrm { s y n } }$ is generated as follows

$$
D _ { \mathrm { s y n } } = \{ ( \widetilde X _ { i } , \widetilde W _ { i } , \widetilde Y _ { i } ) \} _ { i = 1 } ^ { n _ { \mathrm { s y n } } } , \qquad ( \widetilde X _ { i } , \widetilde W _ { i } , \widetilde Y _ { i } ) \stackrel { \mathrm { i . i . d . } } { \sim } \widehat { \mathcal P } _ { \theta } .
$$

To characterize the causal information contained in a dataset, let A denote a fixed causal inference operator that maps a dataset D to an inferential distribution over the target causal estimand:

$$
\boldsymbol { \mathcal { A } } ( \boldsymbol { D } ) = \widehat { p } _ { \boldsymbol { D } } ^ { A } ( \tau ) .
$$

We define Causal Fidelity with respect to the target estimand τ as the extent to which synthetic data preserve the inferential information about τ contained in the real data. Specifically, let

$$
\widehat { p } _ { \mathrm { r e a l } } ( \tau ) = \mathcal { A } ( D _ { \mathrm { r e a l } } )
$$

and

$$
\widehat { p } _ { \mathrm { s y n } } ( \tau ) = \mathcal { A } ( D _ { \mathrm { s y n } } ) .
$$

For a distributional discrepancy $d ( \cdot , \cdot )$ , such as the Wasserstein distance, we define causal discrepancy of a synthetic dataset as

$$
\begin{array} { r } { \mathcal { D } _ { \tau } ( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } ) = d ( \widehat { p } _ { \mathrm { r e a l } } ( \tau ) , \widehat { p } _ { \mathrm { s y n } } ( \tau ) ) . } \end{array}
$$

A smaller discrepancy corresponds to higher causal fidelity.

We quantify the overall causal fidelity loss of $G _ { \theta }$ by the expected discrepancy across synthetic datasets:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) = \mathbb { E } _ { D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } } } \left[ \mathcal { D } _ { \tau } ( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } ) \right] } \\ & { \quad \quad \quad = \mathbb { E } _ { D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } } } \left[ d ( \widehat { p } _ { \mathrm { r e a l } } ( \tau ) , \widehat { p } _ { \mathrm { s y n } } ( \tau ) ) \right] } \end{array}\tag{1}
$$

## 3.3 Causal-Fidelity-Aware Training

Standard generative training seeks to minimize a generative loss ${ \mathcal { L } } _ { \mathrm { g e n } } ( \theta )$ that encourages ${ \widehat { \mathcal { P } } } _ { \theta }$ to approximate the observational distribution. However, this objective does not explicitly require the generated data to preserve the target causal estimand. We therefore consider the causal-regularized objective

$$
\operatorname* { m i n } _ { \theta } \quad { \mathcal { L } } _ { \mathrm { g e n } } ( \theta ) + \lambda { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta ) ,
$$

where $\lambda \geq 0$ controls the tradeoff between conventional generative fidelity and causal fidelity.

A direct optimization of $\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta )$ is challenging, because the causal discrepancy is a dataset-level quantity evaluated only after sampling a synthetic dataset from $G _ { \theta }$ and applying a causal inference operator ${ \mathcal { A } } .$ In general, A may involve estimation procedures that are not differentiable with respect to the generator parameters. Moreover, the discrete sampling process prevents gradients from being propagated directly through the generated dataset. Thus, $\nabla _ { \theta } \bar { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta )$ cannot generally be obtained through standard backpropagation.

For notational convenience, let $p _ { \theta } ( D )$ denote the probability density or mass assigned by the generator to a synthetic dataset D. Then the causal fidelity loss can be written as

$$
\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) = \mathbb { E } _ { D _ { \mathrm { s y n } } \sim p _ { \theta } } \left[ \mathcal { D } _ { \tau } ( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } ) \right] .\tag{2}
$$

Although $\mathcal { D } _ { \tau }$ need not be differentiable with respect to θ, the score-function identity yields

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) = \mathbb { E } _ { D _ { \mathrm { s y n } } \sim p _ { \theta } } \Big [ \mathcal { D } _ { \tau } \big ( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } \big ) \cdot \nabla _ { \theta } \log p _ { \theta } ( D _ { \mathrm { s y n } } ) \Big ] .\tag{3}
$$

Thus, the causal discrepancy acts as a dataset-level weight on the generator score. Synthetic datasets with larger causal discrepancy contribute more strongly to the causal gradient, encouraging the

generator, in aggregate, to shift probability mass away from regions associated with poor causal fidelity.

In practice, at training iteration $k ,$ let $\theta _ { k }$ denote the current generator parameters. We draw m synthetic datasets

$$
D _ { \mathrm { s y n } } ^ { ( j ) } \sim p _ { \theta _ { k } } , \qquad j = 1 , \dots , m ,
$$

and compute their causal discrepancies

$$
\mathcal { D } _ { j } : = \mathcal { D } _ { \tau } \left( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } ^ { ( j ) } \right) = d \Big ( \widehat { p } _ { \mathrm { r e a l } } ( \tau ) , \widehat { p } _ { D _ { \mathrm { s y n } } ^ { ( j ) } } ( \tau ) \Big ) .
$$

The score-function gradient can then be approximated by

$$
\left. \nabla _ { \theta } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) \right. _ { \theta = \theta _ { k } } \approx \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathcal { D } _ { j } \nabla _ { \theta } \log p _ { \theta } \left( D _ { \mathrm { s y n } } ^ { ( j ) } \right) \bigg \vert _ { \theta = \theta _ { k } } .\tag{4}
$$

## 3.4 Algorithm

We instantiate our framework with TabDDPM, which shows stronger treatment-effect preservation than GAN- and VAE-based generators in our motivating experiments (Figure 1). However, its causal fidelity becomes less reliable in higher-dimensional and more complex causal settings (Appendix A.2). This motivates adding explicit causal regularization to the TabDDPM training objective.

For diffusion-based generators, directly working with the marginal likelihood of the terminal synthetic dataset can be inconvenient because it requires integrating over the intermediate reversediffusion states. Rather than approximating this marginal likelihood, we augment the sampling space to include the full reverse-diffusion trajectory. Let $Z \sim P _ { \theta }$ denote the collection of stochastic reverse trajectories generated by the model, and let $D _ { \mathrm { s y n } } = h ( Z )$ denote the corresponding terminal synthetic dataset. Since $D _ { \mathrm { s y n } }$ is a deterministic function of the sampled trajectory, the expected causal loss can be equivalently written as

$$
\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) = \mathbb { E } _ { Z \sim P _ { \theta } } \left[ \mathcal { D } _ { \tau } \big ( D _ { \mathrm { r e a l } } , h ( Z ) \big ) \right] .\tag{5}
$$

This reformulation does not change the objective; it simply expresses the same expectation over the augmented trajectory space rather than over the terminal-data marginal. Applying the score-function identity to $P _ { \theta } ( Z )$ then yields

$$
\nabla _ { \theta } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) = \mathbb { E } _ { Z \sim P _ { \theta } } \Big [ \mathcal { D } _ { \tau } \big ( D _ { \mathrm { r e a l } } , h ( Z ) \big ) \cdot \nabla _ { \theta } \log P _ { \theta } ( Z ) \Big ] .\tag{6}
$$

For a reverse diffusion process, $P _ { \theta } ( Z )$ factorizes over the reverse transitions, so its score can be computed from the transition probabilities along the sampled trajectory.

At each causal-fidelity update, the current generator samples m reverse-diffusion trajectories $Z _ { j } \sim$ $P _ { \theta }$ , producing synthetic datasets $D _ { j } = h ( \breve { Z } _ { j } )$ . Each dataset is assigned a causal discrepancy $\mathcal { D } _ { j } ^ { \top }$ by comparing its inferred ATE distribution with that of the reference data. The causal gradient is then estimated as a discrepancy-weighted average of the trajectory scores,

$$
g _ { \mathrm { c a u s a l } } = \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { S t o p G r a d } ( \mathscr { D } _ { j } - b ) \nabla _ { \theta } \log P _ { \theta } ( Z _ { j } ) ,\tag{7}
$$

where b is a moving-average baseline used to reduce variance.

Algorithm 1 summarizes the proposed training procedure, which combines the standard TabDDPM diffusion gradient with periodic on-policy causal-fidelity updates.

## Algorithm 1 On-Policy Causal-Penalized TabDDPM

Require: $D _ { \mathrm { r e a l } }$ , generator $G _ { \theta }$ , causal operator ${ \mathcal { A } } ,$ penalty weight $\lambda ,$ , number of synthetic datasets   
$m ,$ iterations ${ \bar { K } } .$ , baseline decay $\rho ,$ gradient cap κ, learning rates $\{ \eta _ { k } \} _ { k = 1 } ^ { K }$   
Ensure: Trained $G _ { \theta }$   
$1 \colon \widehat { p } _ { \mathrm { r e a l } } ( \tau ) \gets \mathcal { A } ( D _ { \mathrm { r e a l } } )$

2: Initialize θ and $b \gets 0$   
3: for $k = 1 , \ldots , K$ do   
4: g<sub>gen</sub> $ \nabla _ { \theta } \mathcal { L } _ { \mathrm { g e n } } ( \theta )$   
5: if causal update then   
6: for $j = 1 , \ldots , m$ do   
7: $\begin{array} { r } { Z _ { j } \sim P _ { \theta } , D _ { j } = h ( Z _ { j } ) } \end{array}$   
8: $\widehat { p _ { j } } ( \tau )  A ( \dot { D _ { j } } )$   
9: $\mathbf { \mathcal { D } } _ { j }  W _ { 1 } ( \hat { p } _ { \mathrm { r e a l } } ( \tau ) , \widehat { p } _ { j } ( \tau ) )$   
10: $s _ { j } \gets \nabla _ { \theta } \log P _ { \theta } ( Z _ { j } )$   
11: end for   
12: $\begin{array} { r } { g _ { \mathrm { c a u s a l } }  \frac { \lambda } { m } \sum _ { j = 1 } ^ { m } \mathrm { S t o p G r a d } ( \mathcal { D } _ { j } - b ) s _ { j } } \end{array}$   
13: $\begin{array} { r } { b  ( 1 - \rho ) b + \frac { \rho } { m } \sum _ { j = 1 } ^ { m } \mathcal { D } _ { j } } \end{array}$   
14: $\begin{array} { r } { \alpha  \operatorname* { m i n } \biggl ( 1 , \kappa \frac { \| g _ { \mathrm { g e n } } \| _ { 2 } } { \| g _ { \mathrm { c a u s a l } } \| _ { 2 } + \varepsilon } \biggr ) } \end{array}$   
15: $g  g _ { \mathrm { g e n } } + \alpha g _ { c }$ ausal   
16: else   
17: $\smash { g  g _ { \mathrm { g e n } } }$   
18: end if   
19: $\theta  \theta - \eta _ { k } g$   
20: end for   
21: return $G _ { \theta }$

## 4 Theory

In this section, we examine the relationship between joint distributional fidelity and ATE preservation. Under the problem setup, we show that joint distributional proximity alone does not uniformly control ATE error. We then establish a bound on the expected causal discrepancy of regularized solutions and a sufficient condition for strict improvement.

## 4.1 Joint Distributional Fidelity and the ATE

For a synthetic distribution ${ \widehat { \mathcal { P } } } _ { \theta }$ , we define its population ATE as

$$
\widetilde { \tau } _ { \theta } = \mathbb { E } _ { \widetilde { X } } \big [ \mathbb { E } ( \widetilde { Y } \mid \widetilde { W } = 1 , \widetilde { X } ) - \mathbb { E } ( \widetilde { Y } \mid \widetilde { W } = 0 , \widetilde { X } ) \big ] ,
$$

where all expectations are taken under ${ \widehat { \mathcal { P } } } _ { \theta }$ . The total variation distance is denoted by $d _ { \mathrm { T V } } ( \mathcal { P } , \widehat { \mathcal { P } } _ { \theta } ) =$ sup $_ B \mathinner { | { \mathcal { P } ( B ) - \widehat { \mathcal { P } } _ { \theta } ( B ) } }$ |, where the supremum is over measurable sets B.

Theorem 1 Fix $\Delta > 0$ and $\sigma > 0 .$ . For each $\epsilon \in ( 0 , 1 )$ , consider the observational and generated distributions given by

$$
\begin{array} { l l } { \mathcal { P } : } & { X \sim \mathcal { N } ( 0 , 1 ) , \quad W \sim \mathrm { B e r n o u l l i } ( \epsilon ) , } \\ & { Y = U , \quad U \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , } \\ { \widehat { \mathcal { P } } _ { \theta } : } & { \widetilde { X } \sim \mathcal { N } ( 0 , 1 ) , \quad \widetilde { W } \sim \mathrm { B e r n o u l l i } ( \epsilon ) , } \\ & { \widetilde { Y } = \Delta \widetilde { W } + \widetilde { U } , \quad \widetilde { U } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) . } \end{array}
$$

Within each distribution, the covariate, treatment, and noise are mutually independent. The dependence of P and θ on ϵ is suppressed in the notation. Both distributions admit potential outcomes satisfying consistency, unconfoundedness, and the pointwise overlap condition in the problem setup. However,

$$
\left| \widetilde { \tau } _ { \theta } - \tau \right| = \Delta a n d d _ { \mathrm { T V } } ( \mathcal { P } , \widehat { \mathcal { P } } _ { \theta } ) \leq \epsilon ,\tag{8}
$$

$A s \epsilon \to 0 ,$ the total variation distance converges to zero, whereas the ATE discrepancy remains $\Delta .$

The construction isolates the impact of rare treatment assignments on joint distributional discrepancies. Since the real and generated distributions differ only in the treated group, their contribution to the joint TV distance is weighted by the treatment probability ϵ. In contrast, the ATE depends directly on the treated conditional mean, so the ATE discrepancy can remain fixed even as the joint TV distance approaches zero. Thus, Theorem 1 rules out any uniform ATE error bound that vanishes with this joint distributional discrepancy under pointwise overlap alone. At the same time, exact equality of the joint distributions still implies equality of the corresponding identified ATEs.

## 4.2 Effect of Causal Regularization

Fix the reference dataset $D _ { \mathrm { r e a l } } .$ , the inference operator ${ \mathcal { A } } ,$ the synthetic sample size $n _ { \mathrm { s y n } } ,$ and a nonempty generator class $\{ G _ { \theta } \ | \ \theta \in \Theta \}$ . We set $d = W _ { 1 }$ , as in Algorithm 1, and retain the losses ${ \mathcal { L } } _ { \mathrm { g e n } }$ and $\mathcal { L } _ { \mathrm { c a u s a l } }$ defined in the problem setup. Assume that the inferred ATE distributions have finite first moments and that both losses are finite on $\Theta ,$ with

$$
{ \mathcal { L } } _ { \mathrm { g e n } } ^ { \star } = \operatorname* { i n f } _ { \theta \in \Theta } { \mathcal { L } } _ { \mathrm { g e n } } ( \theta ) > - \infty .
$$

Unless specified otherwise, expectations are taken over $D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } }$ conditional on $D _ { \mathrm { r e a l } }$

Theorem 2 (Causal regularization) Let $\lambda > 0$ and $\delta _ { \mathrm { o p t } } \geq 0 .$ . Suppose $\theta _ { \lambda } \in \Theta$ satisfies the $a p \cdot$ proximate optimality condition

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) + \lambda \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \leq \operatorname* { i n f } _ { \theta \in \Theta } \bigl \{ \mathcal { L } _ { \mathrm { g e n } } ( \theta ) + \lambda \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) \bigr \} + \delta _ { \mathrm { o p t } } .\tag{9}
$$

Then, for every $\bar { \theta } \in \Theta$

$$
\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \leq \mathcal { L } _ { \mathrm { c a u s a l } } ( \bar { \theta } ) + \frac { \mathcal { L } _ { \mathrm { g e n } } ( \bar { \theta } ) - \mathcal { L } _ { \mathrm { g e n } } ^ { \star } + \delta _ { \mathrm { o p t } } } { \lambda } .\tag{10}
$$

If an unpenalized minimizer $\theta _ { \mathrm { g e n } } \in$ arg min<sub>θ∈Θ</sub> ${ \mathcal { L } } _ { \mathrm { g e n } } ( \theta )$ exists, then

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \le \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } ) + \delta _ { \mathrm { o p t } } / \lambda . } \end{array}
$$

For any such $\theta _ { \mathrm { g e n } } , i f c$ comparator $\bar { \theta } \in \Theta$ satisfies ${ \mathcal { L } } _ { \mathrm { c a u s a l } } ( { \bar { \theta } } ) < { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } )$ , then $\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) <$ $\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } )$ whenever

$$
\lambda > \frac { { \mathcal { L } } _ { \mathrm { g e n } } ( \bar { \theta } ) - { \mathcal { L } } _ { \mathrm { g e n } } ^ { \star } + \delta _ { \mathrm { o p t } } } { { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } ) - { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \bar { \theta } ) } .\tag{11}
$$

For exact global minimizers $\theta _ { \lambda _ { i } } \in \Theta$ of $\mathcal { L } _ { \mathrm { g e n } } + \lambda _ { i } \mathcal { L } _ { \mathrm { c a u s a l } }$ at $0 \leq \lambda _ { 1 } < \lambda _ { 2 }$ , respectively,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) \leq \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) , } \end{array}\tag{12}
$$

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 2 } } ) \geq \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 1 } } ) .\tag{13}
$$

Exact minimization of the regularized objective yields an expected causal discrepancy no greater than that of an unpenalized global minimizer. Equation (11) provides a sufficient condition for strict improvement. Equation (10) further relates the causal loss to that of any comparator, with the comparator’s excess generative loss and the optimization error scaled by $1 / \dot { \lambda }$ . Along exact global optima, the causal loss is nonincreasing in λ, while the generative loss is nondecreasing.

Example (Gaussian model with an omitted intercept). Consider the Gaussian models

$$
\begin{array} { c c } { { \mathcal { P } : } } & { { Y = \beta _ { 0 } + \tau W + U , } } \\ { { \widehat { \mathcal { P } } _ { \theta } : } } & { { \widetilde { Y } = \theta \widetilde { W } + \widetilde { U } , } } \end{array}
$$

where $\beta _ { 0 } \neq 0 , \tau , \theta \in \mathbb { R }$ , and $W , \widetilde { W } \sim \mathrm { B e r n o u l l i } ( 1 / 2 )$ . The noise terms follow ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ for fixed $\sigma > 0$ and are independent of their respective treatment indicators. Covariates have the same independent distribution under both models. With ${ \mathcal { L } } _ { \mathrm { g e n } } ( \theta ) = D _ { \mathrm { K L } } ( { \mathcal { P } } \| { \widehat { \mathcal { P } } } _ { \theta } )$ , the unpenalized minimizer is $\theta _ { \mathrm { g e n } } = \tau + \beta _ { 0 }$ , while the generated ATE is $\widetilde { \tau } _ { \theta } \ = \ \theta .$ . Consider the population penalty $W _ { 1 } ( \delta _ { \tau } , \delta _ { \widetilde { \tau } _ { \theta } } \breve { ) } = | \theta - \tau |$ , where $\delta _ { a }$ denotes a point mass at a. For $\lambda \geq 0 .$ , the unique minimizer of $\mathcal { L } _ { \mathrm { g e n } } ( \theta ) + \lambda \vert \theta - \tau \vert \mathrm { i }$ s

$$
\theta _ { \lambda } = \tau + \mathrm { s g n } ( \beta _ { 0 } ) \big ( | \beta _ { 0 } | - 2 \lambda \sigma ^ { 2 } \big ) _ { + } ,\tag{14}
$$

where $( a ) _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . Hence

$$
| \widetilde { \tau } _ { \theta _ { \lambda } } - \tau | = \left( | \beta _ { 0 } | - 2 \lambda \sigma ^ { 2 } \right) _ { + } .
$$

Thus every $\lambda > 0$ strictly reduces the ATE error relative to the unpenalized solution. The generated ATE equals τ whenever $\lambda \ge | \beta _ { 0 } | / ( 2 \sigma ^ { 2 } )$ . Appendix B.4 provides the derivation and relates this population penalty to its finite-sample counterpart.

Implications for ATE estimation. Assume τ is finite. Define the means of the inferred ATE distributions as

$$
\begin{array} { r } { \widehat { \tau } _ { \mathrm { r e a l } } = \mathbb { E } _ { t \sim \widehat { p } _ { \mathrm { r e a l } } } [ t ] , \qquad \widehat { \tau } _ { \mathrm { s y n } } = \mathbb { E } _ { t \sim \widehat { p } _ { \mathrm { s y n } } } [ t ] . } \end{array}
$$

The estimator $\widehat { \tau } _ { \mathrm { s y n } }$ depends on the sampled dataset $D _ { \mathrm { s y n } } .$ while $\widetilde { \tau } _ { \theta }$ is the population ATE under ${ \widehat { \mathcal { P } } } _ { \theta }$ Define the absolute reference estimation error as $e _ { \mathrm { r e a l } } = | \widehat { \tau } _ { \mathrm { r e a l } } - \tau |$ . Then the expected absolute estimation error satisfies

$$
\begin{array} { r } { \mathbb { E } \big | \widehat { \tau } _ { \mathrm { s y n } } - \tau \big | \leq e _ { \mathrm { r e a l } } + \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) . } \end{array}\tag{15}
$$

Combining Equations (15) and (10) gives an estimation error bound for the regularized generator. The bound separates the reference estimation error from the inferential discrepancy introduced by synthesis. Reducing $\mathcal { L } _ { \mathrm { c a u s a l } }$ tightens this bound without implying a monotone reduction in the actual estimation error. Appendix B.3 gives a corresponding bound for $| \widetilde { \tau } _ { \theta } - \tau |$ that also accounts for estimation error under the generated distribution.

## 5 Experiments

In this section, we present the experimental results. First, we conduct simulation studies to evaluate whether incorporating a causal-fidelity penalty improves the preservation of causal effects in synthetic data. Second, we compare the proposed causal-penalized TabDDPM with several baseline tabular data generators on two real-world datasets, evaluating both statistical fidelity and causal fidelity.

## 5.1 Simulation Study

Experimental Setup. We compared baseline TabDDPM and causal-penalized TabDDPM in three selected scenarios: (i) linear outcome models with a constant treatment effect, (ii) nonlinear outcome models with a constant treatment effect, and (iii) heterogeneous treatment-effect models.

Each scenario included 25 continuous and five categorical covariates, randomized binary treatment, and a continuous outcome with Gaussian noise. Both methods were trained on 2, 000 observations for 20, 000 iterations using the same architecture and optimization settings. The causal penalty matched synthetic-data and reference-data ATE posteriors inferred by CausalPFN. Each fitted model generated 20 synthetic datasets of 3, 000 observations.

Evaluation Metrics. We evaluate both distributional and causal fidelity of each synthetic dataset. Distributional fidelity is measured by mean Wasserstein-1 distance for continuous covariates, mean Jensen–Shannon distance for categorical covariates, total variation distance for treatment, Wasserstein-1 distance for outcome, and correlation discrepancy. Causal fidelity is assessed by comparing the ATE posteriors inferred from the reference and synthetic datasets using Wasserstein-1 distance, absolute difference in posterior means, and RMSE relative to the ground-truth effect value.

Results. As shown in Table 1, causal-penalized TabDDPM consistently improves causal fidelity across all three simulation settings compared with vanilla TabDDPM. In the linear setting, it also improves all distributional-fidelity metrics. In the nonlinear and heterogeneous settings, the gains in causal fidelity come with modest degradation in some distributional metrics. These results demonstrate that the proposed causal penalty effectively improves preservation of treatment effects across increasingly complex causal mechanisms.

Table 1: Comparison of vanilla TabDDPM and causal-penalized TabDDPM across three simulation settings: linear constant treatment effect $( \tau = - 2 )$ , nonlinear constant treatment effect $( \tau = 1 )$ , and heterogeneous treatment effects $( \tau = 1 . 5 5 )$
<table><tr><td rowspan="2">Method</td><td rowspan="2"></td><td colspan="5">Distributional fidelity</td><td colspan="3">Causal fidelity</td></tr><tr><td>Num.  $\mathbf { W } _ { 1 } \downarrow$ </td><td>Cat. JS ↓</td><td>T TVD ↓</td><td> $\textbf { Y W } _ { 1 } \downarrow$ </td><td>Corr. L2 ↓</td><td>Post.  $\mathbf { W } _ { 1 } \downarrow$ </td><td>Mean Gap ↓</td><td>ATE RMSE ↓</td></tr><tr><td rowspan="2">Linear</td><td>TabDDPM</td><td> $0 . 8 7 7 { \scriptstyle \pm 0 . 0 2 1 }$ </td><td> $0 . 0 6 5 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $0 . 0 1 6 _ { \pm 0 . 0 0 7 }$ </td><td> $1 . 8 9 9 _ { \pm 0 . 0 4 8 }$ </td><td></td><td> $0 . 3 3 6 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td> $0 . 3 3 6 { \scriptstyle \pm 0 . 0 8 6 }$ </td><td>0.302</td></tr><tr><td>Causal-TabDDPM</td><td> $\mathbf { 0 . 7 3 7 { \scriptstyle \pm 0 . 0 1 7 } }$ </td><td> $\mathbf { 0 . 0 4 4 _ { \pm 0 . 0 0 3 } }$ </td><td> $\mathbf { 0 . 0 0 6 _ { \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 1 . 5 9 6 _ { \pm 0 . 0 4 2 } }$ </td><td> $\begin{array} { c } { 5 . 1 1 5 _ { \pm 0 . 1 5 9 } } \\ { 4 . 9 6 7 _ { \pm 0 . 1 4 0 } } \end{array}$ </td><td>0.123±0.093</td><td>0.122±0.094</td><td>0.179</td></tr><tr><td rowspan="2">Nonlinear</td><td>TabDDPM</td><td></td><td> $\begin{array} { c } { 0 . 0 4 3 _ { \pm 0 . 0 0 2 } } \\ { 0 . 0 4 0 _ { \pm 0 . 0 0 4 } } \end{array}$ </td><td></td><td> $\mathbf { 2 . 1 2 0 _ { \pm 0 . 0 6 1 } }$ </td><td></td><td> $\begin{array} { c } { 0 . 2 1 3 _ { \pm 0 . 1 0 7 } } \\ { 0 . 0 6 6 _ { \pm 0 . 0 6 0 } } \end{array}$ </td><td> $\begin{array} { c } { 0 . 2 1 2 _ { \pm 0 . 1 0 8 } } \\ { 0 . 0 6 1 _ { \pm 0 . 0 6 3 } } \end{array}$ </td><td>0.233</td></tr><tr><td>Causal-TabDDPM</td><td> $\begin{array} { c } { 0 . 6 8 8 _ { \pm 0 . 0 1 6 } } \\ { 0 . 7 5 9 _ { \pm 0 . 0 2 0 } } \end{array}$ </td><td></td><td> $\begin{array} { c } { 0 . 0 6 8 _ { \pm 0 . 0 1 0 } } \\ { 0 . 0 8 6 _ { \pm 0 . 0 0 9 } } \end{array}$ </td><td> $2 . 2 5 4 _ { \pm 0 . 0 7 1 }$ </td><td> $\begin{array} { c } { { \mathbf { 3 . 6 0 4 _ { \pm 0 . 0 9 5 } } } } \\ { { 6 . 4 0 4 _ { \pm 0 . 1 3 5 } } } \end{array}$ </td><td></td><td></td><td>0.086</td></tr><tr><td rowspan="2">Heterogeneous</td><td>TabDDPM</td><td> $\mathbf { 0 . 3 7 2 _ { \pm 0 . 0 1 2 } }$ </td><td></td><td> $\mathbf { 0 . 0 1 3 _ { \pm 0 . 0 0 8 } }$ </td><td> $\mathbf { 0 . 7 8 0 _ { \pm 0 . 0 3 5 } }$ </td><td></td><td> $0 . 1 2 5 _ { \pm 0 . 0 6 8 }$ </td><td> $\begin{array} { c } { 0 . 1 2 4 _ { \pm 0 . 0 6 9 } } \\ { 0 . 0 6 0 _ { \pm 0 . 0 4 9 } } \end{array}$ </td><td>0.136</td></tr><tr><td>Causal-TabDDPM</td><td> $0 . 3 9 2 _ { \pm 0 . 0 1 4 }$ </td><td> $\begin{array} { c } { \mathbf { 0 . 0 2 4 _ { \pm 0 . 0 0 2 } } } \\ { 0 . 0 3 3 _ { \pm 0 . 0 0 3 } } \end{array}$ </td><td>0.018±0.009</td><td> $0 . 8 3 0 _ { \pm 0 . 0 3 8 }$ </td><td> $\begin{array} { c } { { \mathbf { 2 . 6 2 9 _ { \pm 0 . 1 2 5 } } } } \\ { { \mathrm { 3 . 0 7 0 _ { \pm 0 . 1 3 2 } } } } \end{array}$ </td><td> $\mathbf { 0 . 0 6 1 _ { \pm 0 . 0 4 9 } }$ </td><td></td><td>0.081</td></tr></table>

![](images/0e260b8a6c3d4407550fff6e2ca3bbee36fa3487b1776124b358c181a19af243.jpg)

![](images/882e54101e37f5b13a6e8c7a01329692f7251b1547c83de2d8ea40256d5660ec.jpg)  
Figure 2: ATE posterior distributions estimated from real and synthetic data in the linear simulation setting with true ATE τ = −2. Synthetic datasets are generated using vanilla TabDDPM (left) and the proposed causal-penalized TabDDPM (right). The causal-fidelity penalty shifts the synthetic data posterior toward the posterior estimated from the real data and the true ATE.

## 5.2 Real Application

We further evaluate causal-penalized TabDDPM and other tabular data generators on two causal benchmark datasets, IHDP and IST, which preserve realistic covariate structures and treatmentassignment mechanisms.

Datasets. We use two datasets that complement the controlled simulations from different perspectives. IHDP provides a standard semi-synthetic benchmark with known treatment effects, while IST allows us to evaluate the method using data from a large randomized clinical trial.

• IHDP dataset: The Infant Health and Development Program (IHDP) is a widely used semi-synthetic benchmark for causal inference (Hill, 2011). It contains 747 subjects with 25 baseline covariates describing characteristics of the children and their mothers. The treatment corresponds to an intensive early-childhood intervention that included specialist home visits and high-quality child care. The outcome is measured using later cognitive test scores. In the benchmark dataset, the covariates and treatment assignments come from the original study, while the potential outcomes are simulated.

• IST dataset: The International Stroke Trial (IST) is a large randomized controlled trial of antithrombotic treatment for acute ischaemic stroke (International Stroke Trial Collaborative Group, 1997). We use randomized aspirin assignment (RXASP) as the binary treatment and recurrent ischaemic stroke within 14 days (DRSISC) as the outcome. Following the IST analysis in Gruber et al. (2024), we retain the same set of 24 baseline analysis variables. These cover demographic characteristics, stroke presentation and severity, prior medication use, imaging findings, and other clinical information. After preprocessing, the final analysis dataset contains 19,408 patients.

Baselines and Metrics. We compare our proposed method with four representative tabular data generators: (1) CTGAN (Xu et al., 2019), a conditional GAN designed for mixed-type tabular data; (2) TVAE (Xu et al., 2019), a variational autoencoder that models the joint distribution through a latent representation; (3) TabSyn (Zhang et al., 2024), a diffusion-based method that maps heterogeneous tabular features into a continuous latent space before performing diffusion; and (4) TabD-DPM (Kotelnikov et al., 2023), a diffusion model designed directly for heterogeneous tabular data. We evaluate all methods in terms of both distributional and causal fidelity using the same metrics as in the simulation study.

Results. The results on IHDP and IST are reported in Table 2. Relative to vanilla TabDDPM, causal regularization reduces the ATE posterior Wasserstein distance and posterior mean gap by 27.3% and 30.4% on IHDP, and by 22.7% and 24.5% on IST.

These improvements are accompanied by relatively small changes in conventional distributional fidelity. On IHDP, Causal-TabDDPM ranks second across all distributional metrics and substantially improves the causal metrics over vanilla TabDDPM, approaching the performance of TabSyn. On IST, it remains competitive on most distributional metrics and achieves the lowest ATE posterior

Table 2: Comparison of synthetic tabular data generators on the IHDP and IST datasets with respect to distributional and causal fidelity. The best and second-best results for each metric are highlighted in blue and orange, respectively.
<table><tr><td></td><td colspan="5">Distributional fidelity</td><td colspan="2">Causal fidelity</td></tr><tr><td>Method</td><td> $\mathrm { N u m . \ W D \downarrow }$ </td><td> $\mathrm { C a t . } ~ \mathrm { J S } \downarrow$ </td><td>T TVD ↓</td><td> $\mathrm { ~ Y ~ } W _ { 1 } \downarrow$ </td><td> $\mathrm { C o r r . } \ L _ { 2 } \ \downarrow$ </td><td> $\mathrm { P o s t . } ~ W _ { 1 } \downarrow$ </td><td> $\mathbf { M e a n G a p \downarrow }$ </td></tr><tr><td>IHDP dataset</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CTGAN</td><td> $0 . 0 9 2 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 2 3 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 5 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 9 3 1 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $3 . 0 5 6 _ { \pm 0 . 0 1 4 }$ </td><td> $3 . 9 2 8 _ { \pm 0 . 1 3 1 }$ </td><td> $3 . 9 2 8 _ { \pm 0 . 1 3 1 }$ </td></tr><tr><td>TVAE</td><td> $0 . 0 3 5 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 1 5 9 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 1 0 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 4 6 4 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $3 . 0 6 0 _ { \pm 0 . 1 0 1 }$ </td><td> $0 . 1 1 0 { \scriptstyle \pm 0 . 0 6 5 }$ </td><td> $0 . 1 0 7 { \scriptstyle \pm 0 . 0 6 9 }$ </td></tr><tr><td>TabSyn</td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 2 8 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 2 9 0 { \scriptstyle \pm 0 . 0 2 7 }$ </td><td> $0 . 4 8 6 { \scriptstyle \pm 0 . 0 2 5 }$ </td><td> $\mathbf { 0 . 0 5 6 { \scriptstyle \pm 0 . 0 2 2 } }$ </td><td> $\mathbf { 0 . 0 4 3 { \scriptstyle \pm 0 . 0 3 0 } }$ </td></tr><tr><td>TabDDPM</td><td> $\mathbf { 0 . 0 1 0 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 1 3 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 0 . 1 1 8 _ { \pm 0 . 0 2 7 } }$ </td><td> $\mathbf { 0 . 4 3 0 _ { \pm 0 . 0 5 1 } }$ </td><td> $0 . 1 3 3 { \scriptstyle \pm 0 . 0 5 9 }$ </td><td> $0 . 1 3 1 { \scriptstyle \pm 0 . 0 6 0 }$ </td></tr><tr><td>Causal-TabDDPM</td><td> $\mathbf { 0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 0 1 2 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 4 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 1 2 6 _ { \pm 0 . 0 2 9 }$ </td><td> $0 . 4 6 6 _ { \pm 0 . 0 3 9 }$ </td><td> $0 . 0 9 7 { \scriptstyle \pm 0 . 0 4 8 }$ </td><td> $0 . 0 9 2 _ { \pm 0 . 0 5 3 }$ </td></tr><tr><td>IST dataset</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CTGAN</td><td> $0 . 0 2 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 0 6 2 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 3 9 _ { \pm 0 . 0 0 7 }$ </td><td> $0 . 0 1 3 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $0 . 0 1 2 _ { \pm 0 . 0 0 6 }$ </td><td> $0 . 0 1 2 _ { \pm 0 . 0 0 6 }$ </td></tr><tr><td>TVAE</td><td> $0 . 0 2 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 4 1 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 7 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 0 2 2 _ { \pm 0 . 0 0 0 }$ </td><td> $1 . 3 9 2 _ { \pm 0 . 1 2 3 }$ </td><td> $0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td>0.011±0.000</td></tr><tr><td>TabSyn</td><td> $0 . 0 0 5 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 0 1 1 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 0 0 7 { \scriptstyle \pm 0 . 0 0 5 } }$ </td><td> $\mathbf { 0 . 0 0 2 _ { \pm 0 . 0 0 2 } }$ </td><td> $0 . 2 6 8 _ { \pm 0 . 0 2 0 }$ </td><td> $0 . 0 0 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 0 5 _ { \pm 0 . 0 0 2 }$ </td></tr><tr><td>TabDDPM</td><td> $\mathbf { 0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 0 1 6 _ { \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 0 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 3 0 1 { \scriptstyle \pm 0 . 0 2 4 }$ </td><td> $0 . 0 0 6 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 6 _ { \pm 0 . 0 0 1 }$ </td></tr><tr><td>Causal-TabDDPM</td><td> $\mathbf { 0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 1 } }$ </td><td> $0 . 0 1 6 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 0 9 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 0 0 3 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 3 5 9 { \scriptstyle \pm 0 . 0 4 2 }$ </td><td> $\mathbf { 0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 2 } }$ </td><td> $\mathbf { 0 . 0 0 4 { \scriptstyle \pm 0 . 0 0 2 } }$ </td></tr></table>

## 6 Conclusions

Wasserstein distance and posterior mean gap. Overall, causal regularization consistently improves causal fidelity over vanilla TabDDPM while maintaining strong distributional fidelity across both datasets. Additional visualizations are provided in Appendix A.4.2.

We formalize causal fidelity through discrepancies between inferential distributions of a target causal estimand, and show that strong distributional fidelity does not generally guarantee ATE preservation. We introduce a causal-fidelity-aware training framework that augments the generative objective with a causal discrepancy penalty, and instantiate it as a causal-penalized TabDDPM using an on-policy score-function estimator. Across simulation settings and benchmarks, the proposed method improved causal fidelity while largely preserving distributional fidelity. Future work may extend the framework to broader causal estimands, alternative causal inference operators, and other classes of tabular generative models.

## AI Use Statement

The research ideas, methodology, experimental design, and conclusions were developed by the authors. OpenAI Codex was used to assist with code development and deployment, and ChatGPT was used for manuscript editing, language polishing, and refinement of parts of the theoretical analysis. All AI-assisted outputs were reviewed and verified by the authors. The authors take full responsibil ity for the content, claims, experimental designs, codes, and results in this submission.

## References

Akrami, H., Aydore, S., Leahy, R. M., and Joshi, A. A. (2020). Robust variational autoencoder for tabular data with beta divergence. arXiv preprint arXiv:2006.08204.

Amad, H., Qian, Z., Frauen, D., Piskorz, J., Feuerriegel, S., and van der Schaar, M. (2026). Improving the generation and evaluation of synthetic data for downstream medical causal inference. Advances in Neural Information Processing Systems, 38:30435–30478.

Asiaee, A., Liang, Z. J., and Yan, C. (2026). Causalwrap: Model-agnostic causal constraint wrappers for tabular synthetic data. arXiv preprint arXiv:2603.02015.

Balazadeh, V., Kamkari, H., Thomas, V., Li, B., Ma, J., Cresswell, J. C., and Krishnan, R. G. (2025). Causalpfn: Amortized causal effect estimation via in-context learning. arXiv preprint arXiv:2506.07918.

Borisov, V., Seßler, K., Leemann, T., Pawelczyk, M., and Kasneci, G. (2022). Language models are realistic tabular data generators. arXiv preprint arXiv:2210.06280.

Gruber, S., Phillips, R. V., Lee, H., Ho, M., Concato, J., and van der Laan, M. J. (2024). Targeted learning: Toward a future informed by real-world evidence. Statistics in Biopharmaceutical Research, 16(1):11–25.

Hahn, P. R., Murray, J. S., and Carvalho, C. M. (2020). Bayesian regression tree models for causal inference: Regularization, confounding, and heterogeneous effects (with discussion). Bayesian Analysis, 15(3):965–1056.

Hernandez, M., Osorio-Marulanda, P. A., Catalina, M., Loinaz, L., Epelde, G., and Aginako, N. (2025). Comprehensive evaluation framework for synthetic tabular data in health: fidelity, utility and privacy analysis of generative models with and without privacy guarantees. Frontiers in Digital Health, 7:1576290.

Herurkar, D., Ali, A., and Dengel, A. (2025). Evaluating generative models for tabular data: Novel metrics and benchmarking. arXiv preprint arXiv:2504.20900.

Hill, J. L. (2011). Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20(1):217–240.

Horvitz, D. G. and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. Journal of the American statistical Association, 47(260):663–685.

Imbens, G. W. and Rubin, D. B. (2015). Causal inference in statistics, social, and biomedical sciences. New York, 517.

International Stroke Trial Collaborative Group (1997). The international stroke trial (ist): A randomised trial of aspirin, subcutaneous heparin, both, or neither among 19,435 patients with acute ischaemic stroke. The Lancet, 349(9065):1569–1581.

Jordon, J., Szpruch, L., Houssiau, F., Bottarelli, M., Cherubin, G., Maple, C., Cohen, S. N., and Weller, A. (2022). Synthetic data–what, why and how? arXiv preprint arXiv:2205.03257.

Kotelnikov, A., Baranchuk, D., Rubachev, I., and Babenko, A. (2023). Tabddpm: Modelling tabular data with diffusion models. In International conference on machine learning, pages 17564– 17579. PMLR.

Lin, X., Kim, S., Li, Z., DeSoto, Z., Fleming, C., and Cheng, G. (2026). Retabsyn: Realistic tabular data synthesis via reinforcement learning. arXiv preprint arXiv:2603.10823.

Ma, Y., Frauen, D., Javurek, E., and Feuerriegel, S. (2026). Foundation models for causal inference via prior-data fitted networks. In International Conference on Learning Representations, volume 2026, pages 79065–79098.

Park, N., Mohammadi, M., Gorde, K., Jajodia, S., Park, H., and Kim, Y. (2018). Data synthesis based on generative adversarial networks. arXiv preprint arXiv:1806.03384.

Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F., and Scholkopf, B. (2026). Do-pfn: In-¨ context learning for causal effect estimation. Advances in Neural Information Processing Systems, 38:174811–174848.

Robins, J. (1986). A new approach to causal inference in mortality studies with a sustained exposure period—application to control of the healthy worker survivor effect. Mathematical modelling, 7(9-12):1393–1512.

Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. Journal of the American statistical Association, 89(427):846– 866.

Stoian, M. C., Giunchiglia, E., and Lukasiewicz, T. (2025). A survey on tabular data generation: Utility, alignment, fidelity, privacy, and beyond. arXiv preprint arXiv:2503.05954.

Van Der Laan, M. J. and Rubin, D. (2006). Targeted maximum likelihood learning.

Xu, L., Skoularidou, M., Cuesta-Infante, A., and Veeramachaneni, K. (2019). Modeling tabular data using conditional gan. Advances in neural information processing systems, 32.

Xu, Y. (2026). Generative synthetic data for causal inference: Pitfalls, remedies, and opportunities. arXiv preprint arXiv:2604.23904.

Zhang, H., Zhang, J., Shen, Z., Srinivasan, B., Qin, X., Faloutsos, C., Rangwala, H., and Karypis, G. (2024). Mixed-type tabular data synthesis with score-based diffusion in latent space. In International Conference on Learning Representations, volume 2024, pages 52829–52857.

## A Experiments Details

## A.1 Comparison of Baseline Tabular Generative Models

To examine how standard tabular generators differ in their ability to preserve causal information, we compare CTGAN, TVAE, and TabDDPM under the same relatively simple causal data-generating setting. This comparison serves as a preliminary study before considering more challenging datagenerating mechanisms and motivates our focus on TabDDPM in the subsequent experiments.

## A.1.1 Experimental Setup

We use Setting 1 with $d = 3 0$ , consisting of 25 continuous and five categorical covariates, randomized treatment $T \sim$ Bernoulli(0.5), and a constant treatment effect. We consider $\tau \in \{ - 2 , 1 , 5 \}$ with Gaussian outcome noise $\sigma = 1$ . Each reference dataset contains 3,000 observations, split into 2,000 training, 500 validation, and 500 test samples. The data-generating seed is fixed at 42.

For each generator, we perform 10 independent training runs and generate 10 synthetic datasets of 3,000 observations from each fitted model, yielding 100 synthetic datasets per method. Training seeds are varied across fitted models, and distinct sampling seeds are used for each synthetic dataset. Causal fidelity is evaluated using CausalPFN with 8,000 posterior draws, applied consistently to the reference training data and all synthetic datasets.

The final generator configurations are selected under Setting 1 with $\tau = 1$ CTGAN uses 300 epochs, a batch size of 500, 128-dimensional embeddings, generator and discriminator dimensions of (256, 256), and learning rates of $1 0 ^ { - 4 }$ TVAE uses 300 epochs, a batch size of 500, 128- dimensional embeddings, and encoder and decoder dimensions of (256, 256); its learning rate is left at the package default. TabDDPM is trained for 50,000 steps with a batch size of 2,000, learning rate $3 \times 1 0 ^ { - 4 }$ , two hidden layers of size 256, and 1,000 diffusion steps with a cosine schedule. Numerical variables are quantile-normalized before TabDDPM training. CTGAN and TVAE configurations are selected using validation-set distributional fidelity, while the TabDDPM configuration is selected based on the generative training loss.

## A.1.2 Results

Table 3 reports conventional distributional-fidelity metrics under Setting 1 with $\tau = 1$ . TabDDPM achieves the lowest discrepancy across all five reported metrics, while TVAE improves substantially over CTGAN on the numerical covariates and outcome but performs worse on categorical fidelity and correlation structure.

Table 3: Distributional fidelity of CTGAN, TVAE, and TabDDPM under Setting 1 with $\tau = 1$ . Each entry reports the mean and standard deviation across 100 synthetic datasets (10 fitted models $\times 1 0$ datasets).
<table><tr><td>Method</td><td>Num. WD↓</td><td>Cat. JS ↓</td><td> $\mathrm { C o r r . } \ L _ { 2 } \ L$ </td><td> $\textsf { Y } W _ { 1 } \downarrow$ </td><td> $\mathrm { \Delta T V D \downarrow }$ </td></tr><tr><td>CTGAN</td><td> $0 . 0 6 7 _ { \pm 0 . 0 0 9 }$ </td><td> $0 . 0 4 3 _ { \pm 0 . 0 1 0 }$ </td><td> $2 . 0 3 0 _ { \pm 0 . 0 5 8 }$ </td><td> $0 . 8 2 4 _ { \pm 0 . 2 2 2 }$ </td><td> $0 . 0 4 3 _ { \pm 0 . 0 3 3 }$ </td></tr><tr><td>TVAE</td><td> $0 . 0 1 6 _ { \pm 0 . 0 0 1 }$ </td><td> $0 . 0 4 8 _ { \pm 0 . 0 0 7 }$ </td><td> $2 . 5 3 0 _ { \pm 0 . 1 7 6 }$ </td><td> $0 . 1 9 4 _ { \pm 0 . 0 9 4 }$ </td><td> $0 . 0 1 6 _ { \pm 0 . 0 1 3 }$ </td></tr><tr><td>TabDDPM</td><td> $\mathbf { 0 . 0 0 7 _ { \pm 0 . 0 0 1 } }$ </td><td> $\mathbf { 0 . 0 1 7 _ { \pm 0 . 0 0 4 } }$ </td><td> $\mathbf { 1 . 0 8 3 _ { \pm 0 . 2 2 1 } }$ </td><td> $\mathbf { 0 . 1 0 4 _ { \pm 0 . 0 2 8 } }$ </td><td> $\mathbf { 0 . 0 0 9 _ { \pm 0 . 0 0 7 } }$ </td></tr></table>

Distributional fidelity alone does not directly assess whether the treatment-effect information relevant to downstream causal inference is preserved. We therefore additionally compare the ATE posteriors inferred from the reference and synthetic datasets.

Figure 1 shows the ATE posteriors under Setting 1 with $\tau = 1$ . CTGAN exhibits a large shift from the reference posterior and substantial variation across synthetic datasets. TVAE reduces this discrepancy, while TabDDPM produces posteriors that are much more closely aligned with the reference posterior.

Table 4 summarizes the posterior mean discrepancies across the three treatment-effect magnitudes. We report

$$
| \mathbb { E } [ \tau \mid D _ { \mathrm { s y n } } ] - \mathbb { E } [ \tau \mid D _ { \mathrm { r e f } } ] | .
$$

Table 4: Absolute discrepancy between the synthetic- and reference-derived posterior mean ATEs under Setting 1.
<table><tr><td>True τ</td><td>CTGAN</td><td>TVAE</td><td>TabDDPM</td></tr><tr><td>-2</td><td>2.102</td><td>0.952</td><td>0.109</td></tr><tr><td>1</td><td>0.896</td><td>0.328</td><td>0.085</td></tr><tr><td>5</td><td>4.806</td><td>0.715</td><td>0.199</td></tr></table>

TabDDPM yields the smallest discrepancy for all three values of τ, with TVAE showing intermediate performance and CTGAN substantially larger deviations.

Overall, TabDDPM provides the strongest baseline in this setting in terms of both conventional distributional fidelity and preservation of the ATE posterior. We therefore focus on TabDDPM in the subsequent experiments and examine whether this behavior remains reliable under more challenging data-generating mechanisms.

## A.2 Causal Fidelity of Vanilla TabDDPM

We next examine whether the strong baseline performance of vanilla TabDDPM remains consistent across a broader range of data-generating regimes and dimensions.

## A.2.1 Experimental Setup

We use the same three families of simulation mechanisms described in Appendix A.3.1, covering linear outcomes, nonlinear outcomes, and heterogeneous treatment effects. Here, we vary the covariate dimension over $d \in \{ 1 0 , 2 0 , 3 0 \}$ to examine the stability of vanilla TabDDPM across problem sizes and data-generating regimes.

The main experiments use mixed continuous and categorical covariates. The $d = 1 0 , d = 2 0$ and $d = 3 0$ settings contain 8 continuous and 2 categorical variables, 16 continuous and 4 categorical variables, and 25 continuous and 5 categorical variables, respectively. Treatment remains randomized as $T \sim \mathrm { B e r n o u l l i } ( 0 . 5 )$ , and the outcome noise level is fixed at $\sigma = 1$ . For each configuration, we train 10 independent TabDDPM models and generate 10 synthetic datasets from each fitted model.

The TabDDPM configuration varies with dimension. For $d = 3 0$ , the model is trained for 50,000 steps with a batch size of 2,000, learning rate $3 \times 1 0 ^ { - 4 }$ , hidden-layer dimensions (256, 256), and 1,000 diffusion steps. For $d = 2 0$ , we use 30,000 training steps with hidden-layer dimensions (512, 256); for $d = 1 0$ , we use 30,000 steps with hidden-layer dimensions (256, 256). Both retain the same batch size, learning rate, and number of diffusion steps as the $d = 3 0$ configuration. Each model is trained on 2,000 observations and generates synthetic datasets of size 3,000.

Causal fidelity is evaluated using CausalPFN-derived ATE posteriors. For $d = 3 0$ , we use direct ATE inference with 8,000 posterior draws. For $d = 1 0$ and $d = 2 0$ , we use an 80/20 split for CATE-based inference and aggregate the estimated CATEs to obtain the ATE, using 2,000 posterior draws. Figure 4 additionally includes one $d = 1 0$ all-continuous example generated under the corresponding all-continuous configuration.

## A.2.2 Results

Figure 3 summarizes the causal-fidelity discrepancy across the mixed-covariate simulation regimes. Each cell reports

$$
\frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left| \mathbb { E } \Big [ \tau \mid D _ { \mathrm { s y n } } ^ { ( k ) } \Big ] - \mathbb { E } \Big [ \tau \mid D _ { \mathrm { r e f } } ^ { ( k ) } \Big ] \right| ,
$$

where $K$ denotes the number of treatment-effect configurations evaluated within the corresponding regime. Setting 3A contains a single subgroup-treatment-effect configuration and is therefore represented by its corresponding discrepancy.

![](images/068c375d30c7d64b998f7d02e8294d8e92f85107107f00cf643315f2b7b00778.jpg)  
Figure 3: Causal-fidelity discrepancy of vanilla TabDDPM across simulation regimes and dimensions. Each cell shows the mean absolute synthetic–reference ATE posterior mean gap, averaged over treatment-effect configurations.

![](images/4bc7210a0a6f2cac309d5753c61c37dd8f568fa6538aae68783b503ac1abc457.jpg)  
(a) d=10, Setting 2: sinusoidal, tau=1

![](images/d677f9575dbd3a632fa68c69374b11af5c1a1db386390e50696f8280cd8c8bdc.jpg)  
(b) d=30, Setting 1: linear, tau=1

![](images/7031d27dc205d761e038786be74537d19a66d8d0f217c851ef5574cc96ce8ed0.jpg)  
(c) d=30, Setting 2: interaction, tau=5  
Figure 4: Representative ATE posterior comparisons for vanilla TabDDPM. Red and blue denote reference and synthetic posteriors; dashed lines indicate the corresponding posterior means and the true ATE.

The discrepancy varies across both dimensions and data-generating mechanisms. Larger gaps occur in several higher-dimensional settings, but the pattern is not uniformly monotonic in either dimension or model complexity.

Figure 4 illustrates this variation with three representative cases. In the $d = 1 0$ sinusoidal setting with $\tau = 1$ , the synthetic and reference ATE posteriors are closely aligned. In the $d = 3 0$ linear setting with $\tau = 1$ , the synthetic posteriors remain close to the reference but show a visible shift toward larger ATE values. The $d = 3 0$ interaction setting with $\tau = 5$ exhibits a substantially larger and more systematic displacement from the reference posterior.

Overall, vanilla TabDDPM preserves causal information well in some settings but does not do so uniformly across data-generating regimes. This variability motivates incorporating causal fidelity directly into the generative objective rather than relying on distributional fidelity alone.

## A.3 Simulation Study

## A.3.1 Simulation Data-Generating Mechanisms

Across all settings, the outcome is generated from continuous covariates $X = ( X _ { 1 } , \ldots , X _ { 2 5 } ) ^ { \top }$ five categorical covariates $C _ { 1 } , \ldots , \bar { C _ { 5 } }$ , a randomized binary treatment T, and Gaussian noise ϵ. For Settings 1 and 2, the outcome model is

$$
\boldsymbol { Y } = \boldsymbol { f } ( \boldsymbol { X } ) + \sum _ { j = 1 } ^ { 5 } \boldsymbol { \alpha } _ { j , \boldsymbol { C } _ { j } } + \tau \boldsymbol { T } + \epsilon ,
$$

where $\alpha _ { j , C _ { j } }$ denotes the level-specific effect of the jth categorical covariate.

Setting 1: Linear outcome model. The baseline outcome function is linear,

$$
f _ { \mathrm { l i n } } ( X ) = \beta ^ { \top } X ,
$$

with

$$
\beta = ( 1 , \ 0 . 5 , \ - 0 . 5 , \ 0 . 3 , \ - 0 . 3 , \underbrace { 0 . 2 , . . . , 0 . 2 } _ { 2 0 } ) ^ { \top } .
$$

Setting 2: Nonlinear outcome models. We consider two nonlinear baseline outcome functions:

$$
\begin{array} { r l } & { f _ { \mathrm { s i n } } ( X ) = \mathrm { s i n } ( X _ { 1 } ) + 0 . 5 X _ { 2 } + 0 . 3 X _ { 3 } , } \\ & { f _ { \mathrm { q u a d } } ( X ) = \beta ^ { \top } X + 0 . 5 X _ { 1 } ^ { 2 } . } \end{array}
$$

Setting 3: Heterogeneous treatment effects. To introduce subgroup-specific treatment-effect heterogeneity, we replace the first observed categorical covariate by

$$
G \sim \mathrm { { B e r n o u l l i } } ( 0 . 5 ) ,
$$

and define the treatment effect as

$$
\tau ( G ) = { \left\{ \begin{array} { l l } { 3 , } & { G = 1 , } \\ { 0 . 1 , } & { G = 0 . } \end{array} \right. }
$$

The outcome is generated as

$$
Y = f _ { \mathrm { s i n } } ( X ) + \alpha _ { 1 , \widetilde { C } _ { 1 } } + \sum _ { j = 2 } ^ { 5 } \alpha _ { j , C _ { j } } + \tau ( G ) T + \epsilon ,
$$

where

$$
f _ { \sin } ( X ) = \sin ( X _ { 1 } ) + 0 . 5 X _ { 2 } + 0 . 3 X _ { 3 } .
$$

Here, $\widetilde { C } _ { 1 }$ is an auxiliary three-level categorical variable used only to provide an additive categorical contribution to the outcome. Each category coefficient is drawn from $\mathcal { N } ( 0 , 0 . 5 ^ { 2 } )$ and then held fixed across observations. Thus, $G$ controls treatment-effect heterogeneity, while $\widetilde { C } _ { 1 }$ contributes additional outcome variation. Since $G \sim$ Bernoulli(0.5), the population ATE is

$$
\mathbb { E } [ \tau ( G ) ] = 0 . 5 \times 3 + 0 . 5 \times 0 . 1 = 1 . 5 5 .
$$

## A.3.2 Implementation Details

Both vanilla and causal-penalized TabDDPM use Gaussian diffusion for numerical variables and multinomial diffusion for categorical variables, with 1, 000 diffusion steps. The denoising network uses a 128-dimensional time embedding and two hidden layers of 256 units with ReLU activations and no dropout. Models are trained for 20, 000 updates using AdamW with batch size 2, 000, an initial learning rate of $3 \times 1 0 ^ { - 4 }$ with linear decay, and zero weight decay.

For causal-penalized TabDDPM, the causal discrepancy is the Wasserstein-1 distance between the CausalPFN ATE posteriors inferred from the reference and synthetic datasets. After 10,000 warmup updates, causal updates are performed every 100 iterations using m = 10 synthetic tables of 2,000 observations, penalty weight λ = 1, and moving-baseline rate $\rho = 0 . 2$

For each method and simulation scenario, we fit one model using training seed 0 and evaluate the final model on 20 synthetic datasets of 3,000 observations each, using matched sampling seeds across methods.

## A.3.3 Additional Results

To further assess robustness, we compare vanilla TabDDPM and Causal-TabDDPM across multiple causal outcome models and treatment-effect values. As shown in Table 5, Causal-TabDDPM generally achieves lower MAE, RMSE, and posterior Wasserstein-1 distance across most of the considered settings, indicating more accurate preservation of the target ATE. The gains are particularly pronounced in settings where vanilla TabDDPM exhibits larger treatment-effect discrepancies.

Table 5: Causal fidelity comparison of vanilla TabDDPM and Causal-TabDDPM across multiple simulation settings and treatment-effect values. The better result within each setting is shown in bold.
<table><tr><td>Setting</td><td>True ATE</td><td>Method</td><td>ATE mean ± SD</td><td>MAE↓</td><td>RMSE↓</td><td>Post.  $\mathbf { W } _ { 1 } \downarrow$ </td></tr><tr><td>Linear</td><td>-2</td><td>TabDDPM Causal-TabDDPM</td><td> $- 1 . 7 0 9 9 \pm 0 . 0 8 5 5$   $\mathbf { - 2 . 1 1 6 5 \pm 0 . 1 3 8 }$ </td><td>0.2901 0.1454</td><td>0.3018 0.1786</td><td>0.3361 0.1233</td></tr><tr><td>Linear</td><td>1</td><td>TabDDPM Causal-TabDDPM</td><td> $\mathbf { 0 . 9 7 2 9 \pm 0 . 0 6 1 4 }$   $0 . 9 5 1 7 \pm 0 . 0 6 1 8$ </td><td>0.0499 0.0609</td><td>0.0657 0.0772</td><td>0.0532 0.0546</td></tr><tr><td>Sin mix</td><td>1</td><td>TabDDPM Causal-TabDDPM</td><td> $1 . 2 0 0 5 \pm 0 . 0 7 3 8$  0.9728 ± 0.0680</td><td>0.2005 0.0578</td><td>0.2130 0.0716</td><td>0.2370 0.0575</td></tr><tr><td>Sin mix</td><td>5</td><td>TabDDPM Causal-TabDDPM</td><td>4.7801 ± 0.1043 4.8066 ± 0.2010</td><td>0.2233 0.2474</td><td>0.2422 0.2753</td><td>0.1930 0.2296</td></tr><tr><td>Quadratic</td><td>1</td><td>TabDDPM Causal-TabDDPM</td><td>1.2084 ± 0.1077 1.0108 ± 0.0877</td><td>0.2084 0.0600</td><td>0.2334 0.0862</td><td>0.2127 0.0655</td></tr><tr><td>Quadratic</td><td>5</td><td>TabDDPM Causal-TabDDPM</td><td> $5 . 1 3 5 4 \pm 0 . 1 1 1 7$   $\mathbf { 5 . 0 3 8 2 } \pm 0 . 1 7 6 0$ </td><td>0.1443 0.1350</td><td>0.1738 0.1758</td><td>0.1487 0.1413</td></tr><tr><td>Heterogeneous</td><td>1.55</td><td>TabDDPM Causal-TabDDPM</td><td> $1 . 6 6 7 7 \pm 0 . 0 6 9 3$  1.4978 ± 0.0635</td><td>0.1181 0.0641</td><td>0.1357 0.0810</td><td>0.1247 0.0608</td></tr></table>

## A.4 Real Application

## A.4.1 Implementation Details

For both IHDP and IST, each generator is fitted once and used to produce 20 synthetic datasets, each containing 3,000 samples. We use a fixed training seed of 0 and sampling seeds from 10,000 to 10,019. The same train–validation–test split is used across methods, with a split seed of 43. Causal fidelity is evaluated using CausalPFN with 8,000 posterior draws and a fixed inference seed of 314159. For all methods, the reference ATE posterior is estimated from the training split, and the same inference procedure is applied to every synthetic dataset.

CTGAN and TVAE are implemented using SDV 1.18.0, while TabSyn follows the official ICLR 2024 implementation. Their hyperparameters are selected separately on IHDP and IST using validation-set distributional fidelity, without using the causal-fidelity metrics for model selection. The final configurations are summarized in Table 6. For CTGAN and TVAE, the main tuning dimensions include network size, batch size, learning rate or latent dimension, and training length. For TabSyn, we tune the VAE regularization strength, batch size, and diffusion model width. Final TabSyn models are trained for 4,000 VAE epochs and 10,001 diffusion epochs, with 50 sampling steps.

Table 6: Final generator configurations used in the real-application experiments. Only the main tuned parameters are shown.
<table><tr><td>Method</td><td>Dataset Key parameters</td><td></td></tr><tr><td>CTGAN</td><td>IHDP</td><td> $\mathrm { l r } = 1 0 ^ { - 4 } , \mathrm { b a t c h } = 5 0 0 , \mathrm { e p o c h s } = 3 0 0$ </td></tr><tr><td>CTGAN</td><td>IST</td><td> $\mathrm { l r } = 1 0 ^ { - 4 } , \mathrm { b a t c h } = 2 0 0 , \mathrm { e p o c h s } = 3 0 0$ </td></tr><tr><td>TVAE</td><td>IHDP</td><td>dims=(128,128), batch=200, epochs=300</td></tr><tr><td>TVAE</td><td>IST</td><td>dims=(256,256), batch=500, epochs=300</td></tr><tr><td>TabSyn</td><td>IHDP</td><td> $\beta _ { \mathrm { m a x } } = 0 . 0 5 , \mathrm { b a t c h } { = } 4 0 9 6$ </td></tr><tr><td>TabSyn</td><td>IST</td><td> $\beta _ { \mathrm { m a x } } = 0 . 0 1 , \mathrm { b a t c h } { = } 1 0 2 4$ </td></tr><tr><td>TabDDPM</td><td>IHDP</td><td>layers=[512,1024,1024,1024,1024,512], diffusion steps=1000</td></tr><tr><td>TabDDPM</td><td>IST</td><td>layers=[256,128], diffusion steps=100</td></tr></table>

Vanilla TabDDPM and Causal-TabDDPM use the same generator architecture, diffusion settings, optimization schedule, training budget, initialization seed, and evaluation protocol within each dataset. The IHDP models are trained for 20,000 steps with a batch size of 498, 1,000 diffusion steps, and a learning rate of $1 0 ^ { - 3 } ;$ the corresponding IST settings use a batch size of 4,096 and 100 diffusion steps. Causal-TabDDPM adds on-policy causal regularization with $\lambda = 1$ , using $m = 1 0$ synthetic trajectories per causal update and a moving-average baseline with $\rho = 0 . 2$ . Causal updates begin after 10,000 training steps and are applied every 100 steps. The discrepancy is the Wasserstein-1 distance between the ATE posteriors inferred from the reference and synthetic datasets. The causal gradient is capped relative to the generative gradient, with maximum ratios of 1.1 for IHDP and 1.0 for IST.

![](images/4ec44c321d659711f393117056bdbf5f6e751d963e6876a1b908369aa91bec47.jpg)

![](images/1fbcd354eeb5b61acb09b33c4ede84f433bae7de11234566ecaf8ad326f319ef.jpg)

![](images/4fab8fd14a8b8b10dc3440f010e31783ab87c56fbb953cb8d0ee5d20907d4ba6.jpg)  
(c)

![](images/a8afd55a7785d9feacb7dd1391b7e394322a1e0153a6a48a6317979a4bc46687.jpg)  
(d)  
Figure 5: Detailed comparison of vanilla TabDDPM and Causal-TabDDPM on IHDP. Panels (a)-(b) show feature-wise distributional discrepancies, panel (c) compares ATE posteriors, and panel (d) reports posterior Wasserstein distance and absolute posterior mean gap across synthetic datasets.

## A.4.2 Additional Results

Figures 5 and 6 provide a more detailed comparison between vanilla TabDDPM and Causal-TabDDPM. On IHDP, causal regularization generally leads to slightly larger feature-level discrepancies, particularly for the numerical covariates. The differences are small, however, and are not uniform across all categorical variables. In contrast, the causal metrics improve more consistently: the synthetic ATE posteriors move toward the reference posterior, and both the posterior Wasserstein distance and posterior mean gap decrease.

The pattern on IST is somewhat different. Feature-level distributional fidelity remains close between the two models, with some covariates improving under causal regularization and others showing small increases in discrepancy. This suggests that the effect on distributional fidelity is not uniformly negative. The causal improvement is again more systematic. The synthetic ATE posteriors shift toward the reference posterior, and the distributions of both causal discrepancy measures move downward relative to vanilla TabDDPM. These visualizations complement the aggregate results in Table 2 by showing that the gains in causal fidelity do not arise from a uniform deterioration in marginal data fidelity.

![](images/45adfeb7d0da47a77f52844786f1f84c43e058c3aee58a75dd6dac1fb9e57414.jpg)  
(a)

![](images/a3d0b85687c71d856d4e97ef4ef0d1ef47b2d45f12dbaebdffd61bffd1c61276.jpg)

![](images/523eb20bcd09e78ea8108d045d2aa2a1096dcf5eb31d275a8d09ad9fb0398ed8.jpg)  
(c)

![](images/fa7274c10314d5f5b3e6da16965910ab52954bf04125b89c2f9b982a59b90059.jpg)  
(d)  
Figure 6: Detailed comparison of vanilla TabDDPM and Causal-TabDDPM on IST. Panels (a)-(b) show feature-wise distributional discrepancies, panel (c) compares ATE posteriors, and panel (d) reports posterior Wasserstein distance and absolute posterior mean gap across synthetic datasets.

## A.5 Computing Infrastructure

All experiments were conducted on a workstation equipped with an NVIDIA GeForce RTX 4090 GPU. Model training, synthetic-data generation, and causal fidelity evaluation were performed using this GPU environment.

## B Proofs of the Theoretical Results

## B.1 Proof of Theorem 1

Identification and ATEs. Fix $\epsilon \in ( 0 , 1 )$ . Under $\mathcal { P } _ { \cdot }$ , choose potential outcomes $Y ( 0 ) = Y ( 1 ) = U ;$ under ${ \widehat { \mathcal { P } } } _ { \theta }$ choose $\widetilde { Y } ( 0 ) = \widetilde { U }$ and $\widetilde { Y } ( 1 ) = \widetilde { U } + \Delta$ . Consistency holds in both cases. Each pair of potential outcomes is independent of its treatment indicator conditional on its covariate. Moreover, $\mathbb { P } ( W = 1 \ | \ X ) = \epsilon \in ( 0 , 1 )$ under $\mathcal { P } _ { \cdot }$ , and $\mathbb { P } ( \widetilde { W } = 1 \ | \ \widetilde { X } ) = \epsilon$ under ${ \widehat { \mathcal { P } } } _ { \theta }$ . The identification assumptions therefore hold, with $\tau = 0$ and $\widetilde { \tau } _ { \theta } = \Delta$

Total variation and KL divergence. The joint distributions of $( X , W )$ and $( \widetilde { X } , \widetilde { W } )$ coincide, as do the conditional outcome distributions in the control arm. In the treatment arm, the real and generated outcome laws are ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ and ${ \mathcal { N } } ( \Delta , \sigma ^ { 2 } )$ , respectively. Let $\phi _ { \sigma }$ be the density of ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ and Φ the standard normal distribution function. Integrating out the common covariate density and summing over treatment status yields

$$
\begin{array} { r l r } {  { d _ { \mathrm { T V } } ( \mathcal { P } , \widehat { \mathcal { P } } _ { \theta } ) = \frac { \epsilon } { 2 } \int _ { \mathbb { R } }  \phi _ { \sigma } ( y ) - \phi _ { \sigma } ( y - \Delta )  d y } } \\ & { } & { = \epsilon [ 2 \Phi ( \frac { \Delta } { 2 \sigma } ) - 1 ] \leq \epsilon . \quad } \end{array}
$$

The second equality follows because the Gaussian densities cross at $y = \Delta / 2$ . For KL divergence, the common covariate and treatment factors cancel in the density ratio, giving

$$
D _ { \mathrm { K L } } ( \mathcal { P } \Vert \widehat { \mathcal { P } } _ { \theta } ) = \epsilon \mathbb { E } _ { Y \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) } \left[ \frac { ( Y - \Delta ) ^ { 2 } - Y ^ { 2 } } { 2 \sigma ^ { 2 } } \right] = \frac { \epsilon \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } ,
$$

$$
D _ { \mathrm { K L } } ( \widehat { \mathcal { P } } _ { \theta } \Vert \mathcal { P } ) = \epsilon \mathbb { E } _ { \widetilde { Y } \sim \mathcal { N } ( \Delta , \sigma ^ { 2 } ) } \left[ \frac { \widetilde { Y } ^ { 2 } - ( \widetilde { Y } - \Delta ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right] = \frac { \epsilon \Delta ^ { 2 } } { 2 \sigma ^ { 2 } } .
$$

Wasserstein distance. Couple the laws by setting $\widetilde { X } = X , \widetilde { W } = W$ , and $\widetilde { U } = U$ . The resulting Euclidean transport cost is $| \Delta W |$ , so $W _ { 1 } ( \mathcal { P } , \widehat { \mathcal { P } } _ { \theta } ) \leq \epsilon \Delta$ . Conversely, every coupling of $( X , W , Y ) \sim$ $\mathcal { P }$ and $( \widetilde { X } , \widetilde { W } , \bar { \widetilde { Y } } ) \sim \widehat { \mathcal { P } } _ { \theta }$ satisfies

$$
\begin{array} { r } { \mathbb { E } \big \| ( X , W , Y ) - ( \widetilde { X } , \widetilde { W } , \widetilde { Y } ) \big \| _ { 2 } \ge \mathbb { E } | Y - \widetilde { Y } | \ge | \mathbb { E } Y - \mathbb { E } \widetilde { Y } | = \epsilon \Delta . } \end{array}
$$

Taking the infimum over couplings completes the proof.

Connection to inferred ATE distributions. Fix $\epsilon \in ( 0 , 1 )$ and consider $D _ { \mathrm { r e a l } } \sim \mathcal { P } ^ { n }$ and $D _ { \mathrm { s y n } } \sim$ $\widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } }$ . Recall that $\widehat { p } _ { \mathrm { r e a l } } = \mathcal { A } ( D _ { \mathrm { r e a l } } )$ and $\widehat { p } _ { \mathrm { s y n } } = \mathcal { A } ( D _ { \mathrm { s y n } } )$ . Denote a point mass at a by $\delta _ { a } .$ . The triangle inequality and its reverse form imply

$$
| W _ { 1 } ( \widehat { p } _ { \mathrm { r e a l } } , \widehat { p } _ { \mathrm { s y n } } ) - | \widetilde { \tau } _ { \boldsymbol { \theta } } - \tau | | \leq W _ { 1 } ( \widehat { p } _ { \mathrm { r e a l } } , \delta _ { \tau } ) + W _ { 1 } ( \widehat { p } _ { \mathrm { s y n } } , \delta _ { \widetilde { \tau } _ { \boldsymbol { \theta } } } ) ,
$$

because $W _ { 1 } ( \delta _ { \tau } , \delta _ { \tilde { \tau } _ { \theta } } ) = | \widetilde { \tau } _ { \theta } - \tau | = \Delta$ . If both terms on the right converge to zero in probability as $n , n _ { \mathrm { s y n } }  \infty$ , then $\dot { \mathcal { D } } _ { \tau } ( D _ { \mathrm { r e a l } } , \dot { D } _ { \mathrm { s y n } } )$ with $d = W _ { 1 }$ converges in probability to $\Delta$ . The order of limits is essential. For equal, fixed sample sizes $n _ { \mathrm { s y n } } = n$

$$
d _ { \mathrm { T V } } ( \mathcal { P } ^ { n } , \widehat { \mathcal { P } } _ { \theta } ^ { n } ) \leq n d _ { \mathrm { T V } } ( \mathcal { P } , \widehat { \mathcal { P } } _ { \theta } ) \leq n \epsilon .
$$

Consequently, taking $\epsilon \downarrow 0$ at fixed sample size does not establish separation of the inferred ATE distributions.

## B.2 Proof of Theorem 2

Comparator bound and strict improvement. By Equation (9), any $\bar { \theta } \in \Theta$ satisfies

$$
\begin{array} { r } { { \mathcal { L } } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) + \lambda { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \leq { \mathcal { L } } _ { \mathrm { g e n } } ( \bar { \theta } ) + \lambda { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \bar { \theta } ) + \delta _ { \mathrm { o p t } } . } \end{array}
$$

Rearranging and using $\mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) \ge \mathcal { L } _ { \mathrm { g e n } } ^ { \star }$ gives

$$
\begin{array} { r l } & { \lambda \big \{ \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) - \mathcal { L } _ { \mathrm { c a u s a l } } ( \bar { \theta } ) \big \} \leq \mathcal { L } _ { \mathrm { g e n } } ( \bar { \theta } ) - \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) + \delta _ { \mathrm { o p t } } } \\ & { \qquad \leq \mathcal { L } _ { \mathrm { g e n } } ( \bar { \theta } ) - \mathcal { L } _ { \mathrm { g e n } } ^ { \star } + \delta _ { \mathrm { o p t } } . } \end{array}
$$

Division by $\lambda > 0$ proves Equation (10). If $\theta _ { \mathrm { g e n } }$ minimizes $\mathcal { L } _ { \mathrm { g e n } }$ , substituting ${ \bar { \theta } } = \theta _ { \mathrm { g e n } }$ gives

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \le \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } ) + \delta _ { \mathrm { o p t } } / \lambda . } \end{array}
$$

For a comparator with ${ \mathcal { L } } _ { \mathrm { c a u s a l } } ( { \bar { \theta } } ) < { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } )$ , Equation (11) makes the right-hand side of Equation (10) strictly smaller than $\mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } )$ , proving strict improvement.

Monotonicity and the generative tradeoff. For $0 \leq \lambda _ { 1 } < \lambda _ { 2 }$ , exact optimality at each weight gives

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 1 } } ) + \lambda _ { 1 } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) \leq \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 2 } } ) + \lambda _ { 1 } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) , } \\ & { \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 2 } } ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) \leq \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 1 } } ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) . } \end{array}
$$

Adding these inequalities yields

$$
\begin{array} { r } { ( \lambda _ { 2 } - \lambda _ { 1 } ) \big \{ \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) - \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) \big \} \leq 0 . } \end{array}
$$

Since $\lambda _ { 2 } > \lambda _ { 1 }$ , this proves Equation (12). The same optimality inequalities imply

$$
\begin{array} { r l } & { 0 \leq \lambda _ { 1 } \big \{ { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) - { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) \big \} } \\ & { \leq { \mathcal { L } } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 2 } } ) - { \mathcal { L } } _ { \mathrm { g e n } } ( \theta _ { \lambda _ { 1 } } ) } \\ & { \leq \lambda _ { 2 } \big \{ { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 1 } } ) - { \mathcal { L } } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda _ { 2 } } ) \big \} . } \end{array}
$$

This also proves Equation (13). In particular, comparing an exact penalized optimum with $\theta _ { \mathrm { g e n } }$ yields

$$
0 \leq \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) - \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \mathrm { g e n } } ) \leq \lambda \big \{ \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \mathrm { g e n } } ) - \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta _ { \lambda } ) \big \} .
$$

These inequalities require neither convexity nor correct specification of the generator class. □

## B.3 From Causal Discrepancy to ATE Error

For any two probability laws $\mu$ and ν on the real line with finite first moments, every coupling π satisfies

$$
\left| \int u \mu ( d u ) - \int v \nu ( d v ) \right| = \left| \int ( u - v ) \pi ( d u , d v ) \right| \leq \int | u - v | \pi ( d u , d v ) .
$$

Taking the infimum over couplings shows that $W _ { 1 } ( \mu , \nu )$ bounds the absolute difference between their means. Applying this inequality to the inferred ATE distributions gives

$$
| \widehat { \tau } _ { \mathrm { s y n } } - \widehat { \tau } _ { \mathrm { r e a l } } | \leq W _ { 1 } ( \widehat { p } _ { \mathrm { s y n } } , \widehat { p } _ { \mathrm { r e a l } } ) = \mathcal { D } _ { \tau } ( D _ { \mathrm { r e a l } } , D _ { \mathrm { s y n } } ) .
$$

The triangle inequality followed by expectation over $D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } }$ therefore proves Equation (15), with $e _ { \mathrm { r e a l } } = | \widehat { \tau } _ { \mathrm { r e a l } } - \tau |$ |. For a penalized solution and any $\bar { \theta } \in \Theta .$ , Theorem 2 gives

$$
\mathbb { E } _ { D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta _ { \lambda } } ^ { n _ { \mathrm { s y n } } } } \left| \widehat { \tau } _ { \mathrm { s y n } } - \tau \right| \leq e _ { \mathrm { r e a l } } + \mathcal { L } _ { \mathrm { c a u s a l } } ( \bar { \theta } ) + \frac { \mathcal { L } _ { \mathrm { g e n } } ( \bar { \theta } ) - \mathcal { L } _ { \mathrm { g e n } } ^ { \star } + \delta _ { \mathrm { o p t } } } { \lambda } .
$$

To bound the population ATE discrepancy, suppose the generated law ${ \widehat { \mathcal { P } } } _ { \theta }$ satisfies the identification conditions and has finite population ATE $\widetilde { \tau } _ { \theta }$ . For the fixed synthetic sample size $n _ { \mathrm { s y n } }$ , define the expected absolute estimation error under this law by

$$
e _ { \mathrm { s y n } } ( \theta ) : = \mathbb { E } _ { D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } } } \left| \widehat { \tau } _ { \mathrm { s y n } } - \widetilde { \tau } _ { \theta } \right| .
$$

The triangle inequality and expectation over $D _ { \mathrm { s y n } }$ yield

$$
| \widetilde { \tau } _ { \theta } - \tau | \leq e _ { \mathrm { s y n } } ( \theta ) + \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) + e _ { \mathrm { r e a l } } .\tag{16}
$$

Equation (16) bounds the population ATE discrepancy by three terms: synthetic estimation error, causal discrepancy, and reference estimation error. In particular, if $e _ { \mathrm { s y n } } ( \theta ) \leq \bar { e } _ { \mathrm { s y n } }$ uniformly over the compared generators, then $\bar { e } _ { \mathrm { s y n } } + e _ { \mathrm { r e a l } } + \mathcal { L } _ { \mathrm { c a u s a l } } \bar { ( \theta ) }$ is a common valid upper bound. Along exact penalized optima, this common upper bound is nonincreasing in λ. This conclusion concerns the error bound; the actual population ATE discrepancy need not be monotone in λ.

## B.4 Proof of the Gaussian Example

Under P, take $Y ( 0 ) = \beta _ { 0 } + U$ and $\begin{array} { r } { Y ( 1 ) = \beta _ { 0 } + { \tau } + { U } ; } \end{array}$ ; under ${ \widehat { \mathcal { P } } } _ { \theta } .$ , take $\widetilde { Y } ( 0 ) = \widetilde { U }$ and $\widetilde { Y } ( 1 ) = \theta { + } \widetilde { U }$ Randomized treatment with probability $1 / 2$ ensures identification, so the respective ATEs are τ and $\widetilde { \tau } _ { \theta } = \theta$ . The covariate and treatment distributions coincide under the two laws. The KL divergence between normal distributions with common variance $\sigma ^ { 2 }$ is their squared mean difference divided by $2 \sigma ^ { 2 }$ . Conditioning on treatment therefore gives

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta ) = \frac { \beta _ { 0 } ^ { 2 } } { 4 \sigma ^ { 2 } } + \frac { ( \theta - \tau - \beta _ { 0 } ) ^ { 2 } } { 4 \sigma ^ { 2 } } , \qquad \theta _ { \mathrm { g e n } } = \tau + \beta _ { 0 } .
$$

The generator’s fixed zero intercept forces the unpenalized fit to absorb $\beta _ { 0 }$ into its treatment coefficient.

Since $W _ { 1 } ( \delta _ { \tau } , \delta _ { \theta } ) = | \theta - \tau |$ , setting $v = \theta - \tau$ reduces the population regularized objective, up to an additive constant, to

$$
{ \frac { ( v - \beta _ { 0 } ) ^ { 2 } } { 4 \sigma ^ { 2 } } } + \lambda | v | .
$$

This function is strictly convex. Its unique minimizer satisfies the subgradient condition

$$
0 \in \frac { v - \beta _ { 0 } } { 2 \sigma ^ { 2 } } + \lambda \partial | v | .
$$

For $v \neq 0 ,$ solving this condition gives $v = \beta _ { 0 } - 2 \lambda \sigma ^ { 2 } \operatorname { s g n } ( v )$ ; the solution is $v = 0$ exactly when $| \beta _ { 0 } | \le 2 \lambda \sigma ^ { 2 }$ . Together these cases yield Equation (14) and the stated ATE error. Since $\beta _ { 0 } \neq 0 ,$ , this error is strictly smaller than $| \beta _ { 0 } |$ for every $\lambda > 0$ . The corresponding increase in generative loss is also explicit:

$$
\mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \lambda } ) - \mathcal { L } _ { \mathrm { g e n } } ( \theta _ { \mathrm { g e n } } ) = \frac { \operatorname* { m i n } \{ | \beta _ { 0 } | , 2 \lambda \sigma ^ { 2 } \} ^ { 2 } } { 4 \sigma ^ { 2 } } .
$$

Thus causal fidelity improves through a quantifiable tradeoff in joint distributional fit.

Relation to the finite-sample penalty. The example uses a population analogue of the Wasserstein penalty. For each fixed θ, the reverse triangle inequality and expectation over $D _ { \mathrm { s y n } }$ give

$$
| \mathcal { L } _ { \mathrm { c a u s a l } } ( \theta ) - | \theta - \tau | | \leq W _ { 1 } ( \widehat { p } _ { \mathrm { r e a l } } , \delta _ { \tau } ) + \mathbb { E } _ { D _ { \mathrm { s y n } } \sim \widehat { \mathcal { P } } _ { \theta } ^ { n _ { \mathrm { s y n } } } } W _ { 1 } ( \widehat { p } _ { \mathrm { s y n } } , \delta _ { \theta } ) .
$$

The population penalty is recovered when both terms on the right vanish, requiring convergence in mean on the synthetic side. The closed-form solution therefore describes the population illustration, rather than the original finite-sample optimization problem.