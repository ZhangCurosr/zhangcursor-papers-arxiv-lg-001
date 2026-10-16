# Conditional Mean Flows: Few-Step Densities for Simulation-Based Inference

Jonas Arruda University of Bonn

Jan Hasenauer University of Bonn

Stefan T. Radev Rensselaer Polytechnic Institute

## Abstract

Flow matching has emerged as a powerful generative family for simulation-based inference (SBI). However, sampling and density evaluation require numerical integration at inference time. This makes posterior sampling in neural posterior estimation costly. More importantly, it can make repeated likelihood evaluation prohibitively expensive and impractical for Markov chain Monte Carlo (MCMC) sampling, which may require thousands of repeated density evaluations. Here, we develop conditional mean flows for posterior and likelihood approximation, combining few-step sampling with tractable density evaluation. We show that the mean flow objective identifies conditional flow maps and that independent masking enables direct estimation of marginal likelihoods. We evaluate both posterior and likelihood estimation across several tasks and show that conditional mean flows retain the flexibility of flow-based density estimators while significantly reducing inference time. In particular, single-step conditional mean flow density evaluation yields estimation error comparable to 200-step flow matching while enabling about 200-fold faster downstream MCMC sampling.

## 1 INTRODUCTION

Mechanistic simulators are used to describe processes ranging from cognitive science (Ratclif et al., 2016) to disease modeling (Ye et al., 2025). Yet, while practically important, their likelihoods are often unavailable in closed form. Simulation-based inference (SBI; Sisson et al., 2018; Cranmer et al., 2020; Deistler et al., 2025; Arruda et al., 2026) addresses this challenge by using simulated parameter-observation pairs (θ, y) to infer unknown parameters, and amortized methods reuse the trained network across datasets (Bürkner et al., 2023).

Two popular forms of SBI are neural posterior estimation (NPE) and neural likelihood estimation (NLE). NPE learns a posterior model $q _ { \psi } ( \pmb { \theta } \mid \textbf { y } )$ which can sample from the approximate posterior directly. NLE, instead, learns a likelihood model $q _ { \psi } ( \mathbf { y } \mid \theta )$ which requires Markov chain Monte Carlo (MCMC) to sample from the posterior. NPE amortizes over datasets but is usually tied to the prior and data format used in training, whereas NLE needs MCMC for every dataset, but allows the prior to be changed after training and composes exchangeable likelihoods across any number of i.i.d. observations (Cranmer et al., 2020).

Many SBI methods demand neural estimators that are both flexible and inexpensive to evaluate. In NLE, MCMC requires thousands of likelihood evaluations per posterior, making each evaluation a practical bottleneck. Further, hierarchical approaches (Arruda et al., 2024) or self-consistency objectives that improve NPE (Schmitt et al., 2024a; Mishra et al., 2026) and enable model comparison (Kucharský and Bürkner, 2026) rely on density evaluation during training or inference.

So far, normalizing flows have been a standard choice in these applications because they evaluate densities in a single pass (Papamakarios et al., 2021; Dinh et al., 2017; Durkan et al., 2019). However, their transformations must remain invertible with tractable Jacobians, which restricts how inductive biases can be built into the architecture (Chen et al., 2025). Autoregressive flows based on causal transformers (e.g., TARFlow and STARFlow) relax this restriction (Zhai et al., 2025; Gu et al., 2025), but impose a fixed variable ordering and require sequential inversion for sampling. Moreover, tractable joint densities do not generally imply tractable marginals, which are required with missing observations (Yalavarthi et al., 2024).

Flow matching and difusion models instead learn a velocity or score field with an unconstrained neural network (Lipman et al., 2023; Liu et al., 2022; Wildberger et al., 2023; Sharrock et al., 2024; Arruda et al., 2026), and difusion transformers can flexibly represent arbitrary conditionals and marginals (Gloeckler et al.,

2024). Yet, their densities require solving an ordinary diferential equation (ODE), which can take hundreds of network evaluations and makes repeated likelihood calls during training or within MCMC costly. Few-step generative models reduce the sampling cost, but consistency models, for instance, provide no tractable density (Song et al., 2023; Schmitt et al., 2024b). Thus, current SBI approaches either restrict the network architecture or make every density evaluation expensive.

To overcome this limitation, we present conditional mean flows for SBI, which learn the average velocity between two times, so that one network evaluation defines a finite-time map based on mean flows (Geng et al., 2025, 2026). We first evaluate direct posterior sampling and density estimation, then focus on learned likelihoods for compositional inference, missing observations, and gradient-based and delayed acceptance MCMC. Concretely, we develop and showcase

1. a conditional mean flow formulation for posterior and likelihood approximation, with exact stepwise Jacobian determinants, and a proof that the mean flow objective sufices to identify conditional flow maps (Section 3.1, Proposition 1);

2. marginal density estimation for missing targets with a masked transformer backbone, and a proof that the masked objective targets the correct marginal likelihood (Section 3.3, Proposition 2);

3. and a benchmark against normalizing flows and regular flow matching on four tasks covering both NLE and NPE, including a single-trial likelihood comparison in which one mean flow step matches the error of 200-step flow matching with two orders of magnitude faster sampling (Section 4);

## 2 BACKGROUND AND RELATED WORK

Simulation-based inference. For conditionally independent observations ${ \bf y } _ { n } ,$ simulation-based inference targets $\begin{array} { r } { p ( \pmb { \theta } ~ \vert ~ \mathbf { y } _ { \mathrm { { o b s } } } ) \propto p ( \pmb { \theta } ) \prod _ { n = 1 } ^ { N } p ( \mathbf { y } _ { n } | \mathbf { \theta } \pmb { \theta } ) } \end{array}$ when the likelihood is unavailable. Classical approximate Bayesian computation compares simulated and observed data directly, but becomes less efective as dimension increases (Sisson et al., 2018). Neural posterior estimation (NPE) fits $q _ { \psi } ( \pmb { \theta } \mid \mathbf { y } )$ to simulated pairs and draws approximate posterior samples without running MCMC for each dataset. Neural likelihood estimation (NLE), instead, fits a conditional density to prior-predictive pairs and uses it within MCMC (for an overview, see Cranmer et al., 2020). A perobservation estimator can be reused across dataset sizes, making single-trial likelihoods useful for models with exchangeable trials (Boelts et al., 2022), and spline coupling flows are the standard tractable-density baseline (Durkan et al., 2019).

Flow matching and difusion-based inference. Flow matching learns a velocity field $\mathbf { v } ( \mathbf { z } _ { t } , t )$ that transports a Gaussian distribution to the target along a prescribed interpolation (Lipman et al., 2023; Liu et al., 2022). It has been applied to both NPE and NLE, alongside score-based generative models (for a review, see Arruda et al., 2026). Sampling requires ODE integration, and density evaluation additionally integrates the velocity divergence through the continuous changeof-variables formula (Chen et al., 2018), so repeated integration can dominate the cost of NLE.

Few-step generative models. Consistency models learn transitions from intermediate noisy states toward data, enabling generation in one or a few steps (Song et al., 2023). They also support fast posterior sampling in SBI (Schmitt et al., 2024b; Arruda et al., 2026), although the standard formulation does not provide a tractable posterior density. Flow maps learn finitetime ODE transitions between arbitrary time pairs, and mean flows parameterize these maps through average velocities and can be trained without distillation (Geng et al., 2025, 2026). Rehman et al. (2026) combined mean flows with a cycle loss that encourages invertibility to obtain few-step densities for Boltzmann generators.

MCMC with learned and multi-fidelity likelihoods. The no-U-turn sampler (NUTS; Hofman and Gelman, 2014; Betancourt, 2017) uses density and gradient evaluations to construct proposals, and probabilistic programming frameworks allow learned likelihoods to enter these calculations as model components (Carpenter et al., 2017). Delayed acceptance MCMC uses a cheap approximation to screen proposals while leaving the fine target invariant (Christen and Fox, 2005). Moreover, its multilevel variant organizes several approximations (Lykkegaard et al., 2023), and related approaches in SBI use simulators at diferent resolutions (Prescott and Baker, 2021; Tatsuoka et al., 2026).

## 3 METHOD

A note on notation. We write z for the density target and x for the conditioning variables. In posterior estimation, the network learns $q _ { \psi } ( \pmb { \theta } \mid \mathbf { y } )$ with parameters $\mathbf { z } \equiv \pmb { \theta }$ as the target and data $\mathbf x \equiv \mathbf y$ as the condition. In likelihood estimation, it learns $q _ { \psi } ( \mathbf { y } \mid \theta )$ with $\mathbf { z } \equiv \mathbf { y }$ and $\mathbf { x } \equiv { \pmb \theta }$ . The same formulation applies to either direction by choosing the target and condition

![](images/d425816c80de548ca799fc0c745c3c4699b92312f7c87c9346ecc04d2b71d65d.jpg)  
Figure 1: Learned two-moons posteriors as samplers and as densities. Top row: 5,000 draws per method for the canonical observation $\mathbf { y } = ( 0 , 0 )$ ; bottom row: the learned log-density on a 512×512 grid with the exact 95% highest-density region dashed. (A) A normalizing flow; (B) a consistency model with S=32 steps, which has no tractable density; (C) a mean flow with S=1. (D) The same mean flow with S=32 and (E) a separately trained flow matching model integrated with 32 Euler steps. (F) The velocity head of the mean flow and (G) the flow matching model, both integrated with the adaptive Tsit5 ODE solver.

accordingly.

## 3.1 Flow-Based Density Evaluation

Flow matching vs. normalizing flows. We consider flows that transport the target distribution $p ( \mathbf { z }$ x) to a spherical Gaussian base distribution. Time runs from data to noise: $t = 0$ is the clean target and $t = 1$ is pure noise, along the flow matching interpolation ${ \bf z } _ { t } = ( 1 - t ) { \bf z } _ { 0 } + t { \bf \epsilon }$ with $\mathbf { \epsilon } \in \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Flow matching for SBI learns the marginal velocity $\mathbf { v } _ { \psi } ( \mathbf { z } _ { t } , t , \mathbf { x } )$ by regressing the conditional velocity $\epsilon - { \bf z } _ { 0 }$ . For sampling, it draws the initial condition ${ \bf z } _ { 1 } \sim \mathcal { N } ( { \bf 0 } , { \bf I } )$ and then integrates $\mathrm { d } { \mathbf { z } } _ { t } = { \mathbf { v } } _ { \psi }$ dt from t=1 to t=0 (Wildberger et al., 2023). The density follows from the continuous change of variables,

$$
\log q _ { \psi } ( \mathbf { z } _ { 0 } \mid \mathbf { x } ) = \log \mathcal { N } ( \mathbf { z } _ { 1 } ; \mathbf { 0 } , \mathbf { I } ) + \int _ { 0 } ^ { 1 } \nabla \cdot { \mathbf { v } } _ { \psi } ( \mathbf { z } _ { t } , t , \mathbf { x } ) \mathrm { d } t\tag{1}
$$

which can be integrated with any ODE solver (Arruda et al., 2026). Normalizing flows instead learn an invertible map $f _ { \psi } ( \cdot ; \mathbf { x } )$ from the target to the base distribution and evaluate

$$
\begin{array} { r l } & { \log q _ { \psi } ( \mathbf { z } \mid \mathbf { x } ) = \log \mathcal { N } \big ( f _ { \psi } ( \mathbf { z } ; \mathbf { x } ) ; \mathbf { 0 } , \mathbf { I } \big ) } \\ & { \qquad + \log \big | \operatorname* { d e t } \nabla _ { \mathbf { z } } f _ { \psi } ( \mathbf { z } ; \mathbf { x } ) \big | } \end{array}\tag{2}
$$

in a single pass (Papamakarios et al., 2021). To keep the determinant tractable, $f _ { \psi }$ is composed of invertible layers with triangular Jacobians (Dinh et al., 2017; Durkan et al., 2019), which avoids the integral in (1) but restricts the neural network architecture (Figure 1).

Average velocity and the flow map. Extending Geng et al. (2025) to conditional models, we train a network to predict the average velocity

$$
\mathbf { u } _ { \psi } ( \mathbf { z } , s , t , \mathbf { x } ) \approx \frac { 1 } { t - s } \int _ { s } ^ { t } \mathbf { v } ( \mathbf { z } _ { \tau } , \tau , \mathbf { x } ) \mathrm { d } \tau ,\tag{3}
$$

the mean of the instantaneous field along the ODE trajectory started at ${ \bf z } _ { s } = { \bf z }$ conditioned on x. Since x is constant along the trajectory, the conditional velocity defines one ODE per condition. One network evaluation then applies the finite-time flow map

$$
\Phi _ { \psi } ( \mathbf { z } ; s , t , \mathbf { x } ) = \mathbf { z } + ( t - s ) \mathbf { u } _ { \psi } ( \mathbf { z } , s , t , \mathbf { x } ) ,\tag{4}
$$

with the boundary case $\mathbf { u } _ { \psi } ( \mathbf { z } , t , t , \mathbf { x } )$ corresponding to the instantaneous velocity v.

Training with a self-consistent tangent and a cycle loss. Diferentiating $\begin{array} { r } { ( t - s ) { \bf u } = { \bf \breve { \Gamma } } _ { s } ^ { t } } \end{array}$ v dτ along the trajectory gives the MeanFlow identity (Geng et al., 2025),

$$
\begin{array} { r } { { \bf v } ( { \bf z } _ { s } , s , { \bf x } ) = { \bf u } _ { \psi } ( { \bf z } _ { s } , s , t , { \bf x } ) - ( t - s ) \frac { \mathrm { d } } { \mathrm { d } s } { \bf u } _ { \psi } ( { \bf z } _ { s } , s , t , { \bf x } ) , } \end{array}\tag{5}
$$

where $\mathrm { d } / \mathrm { d } s = \mathbf { v } \cdot \nabla _ { \mathbf { z } } + \partial _ { s }$ is the total derivative along $( \dot { \bf z } _ { s } , \dot { s } , \dot { t } ) = ( { \bf v } , 1 , 0 )$ at fixed x. Following Geng et al. (2026), we use the right-hand side as a prediction of the instantaneous velocity,

$$
\begin{array} { r } { \mathbf { V } _ { \psi } = \mathbf { u } _ { \psi } ( \mathbf { z } _ { s } , s , t , \mathbf { x } ) - ( t - s ) \operatorname { s g } \big [ \frac { \mathrm { d } } { \mathrm { d } s } \mathbf { u } _ { \psi } ( \mathbf { z } _ { s } , s , t , \mathbf { x } ) \big ] , } \end{array}\tag{6}
$$

where sg[·] denotes the stop-gradient and the total derivative is computed by one Jacobian-vector product with tangent $( \mathrm { s g } [ \hat { \mathbf { v } } _ { \psi } ] , 1 , 0 )$ . Since the marginal velocity v is unknown, the tangent $\hat { \mathbf { v } } _ { \psi } ( \mathbf { z } _ { s } , s , \mathbf { x } )$ is predicted by an auxiliary velocity head, which shares all but the last two layers with $\mathbf { u } _ { \psi }$ . With ${ \bf z } _ { s } = ( 1 - s ) { \bf z } _ { 0 } + s \epsilon$ , both outputs regress onto the conditional velocity,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F } } ( \psi ) = \mathbb { E } \Big [ w \left| \left| \mathbf { V } _ { \psi } - \left( \epsilon - \mathbf { z } _ { 0 } \right) \right| \right| ^ { 2 } + \frac { 1 } { d } \big \| \hat { \mathbf { v } } _ { \psi } - \left( \epsilon - \mathbf { z } _ { 0 } \right) \big \| ^ { 2 } \Big ] , } \end{array}\tag{7}
$$

with the adaptive weight $w = \mathrm { s g } \left\lceil ( \| \mathbf { V } _ { \psi } - ( \epsilon - \mathbf { z } _ { 0 } ) \| ^ { 2 } + \right.$ $c ) ^ { - p } \rceil$ of Geng et al. (2025) and d the dimension of z (all hyperparameters are defined in Appendix C). The expectation is taken over simulations $( { \bf z } _ { 0 } , { \bf x } )$ , noise $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and independent times s, t drawn from a power-law distribution. Hence, times are sampled more often closer to the target, similar to Wildberger et al. (2023). For some of the pairs we set $t { = } s .$ , where $\mathbf { V } _ { \psi } = \mathbf { u } _ { \psi } ( \mathbf { z } _ { s } , s , s , \mathbf { x } )$ and (7) reduces to flow matching. Besides stabilizing training, the velocity head allows us to use the trained mean flow as a flow matching model. To encourage invertibility of the flow map, we add the cycle loss of Rehman et al. (2026), which penalizes the discrepancy between a state and its image after mapping $s \to t$ and back,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { c y c } } ( \psi ) = \mathbb { E } \Big [ \frac { 1 } { d } \big \| \mathbf { z } _ { s } - \boldsymbol { \Phi } _ { \psi } \big ( \boldsymbol { \Phi } _ { \psi } ( \mathbf { z } _ { s } ; s , t , \mathbf { x } ) ; t , s , \mathbf { x } \big ) \big \| ^ { 2 } \Big ] , } \end{array}\tag{8}
$$

and minimize $\mathcal { L } _ { \mathrm { M F } } + \lambda \mathcal { L } _ { \mathrm { c y c } }$ with the cycle weight $\lambda > 0 .$ We further replace some x in (8) by the condition of another simulation, to encourage $\Phi _ { \psi }$ to be invertible for every condition and to make the map robust to any MCMC proposal.

Exact densities from composed steps. We fix a schedule $0 = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { S } = 1$ and let $G _ { i } ( \mathbf { z } ) =$ $\Phi _ { \psi } ( { \bf z } ; t _ { i - 1 } , t _ { i } , { \bf x } )$ be the step maps toward noise. With ${ \bf z } ^ { ( 0 ) } = { \bf z }$ and $\mathbf { z } ^ { ( i ) } = G _ { i } ( \mathbf { z } ^ { ( i - \bar { 1 } ) } )$ , the change-of-variables expression is

$$
\begin{array} { l } { \log q _ { \psi } ^ { ( S ) } ( \mathbf { z } \mid \mathbf { x } ) = \log \mathcal { N } \big ( \mathbf { z } ^ { ( S ) } ; \mathbf { 0 } , \mathbf { I } \big ) } \\ { \displaystyle \qquad + \sum _ { i = 1 } ^ { S } \log \big | \operatorname* { d e t } \nabla _ { \mathbf { z } } G _ { i } \big ( \mathbf { z } ^ { ( i - 1 ) } \big ) \big | , } \end{array}\tag{9}
$$

where each Jacobian can be computed via autodif. If the composed map $T = G _ { S } \circ \cdot \cdot \cdot \circ G _ { 1 }$ is a difeomorphism of $\mathbb { R } ^ { d }$ , the chain rule gives det $\begin{array} { r } { \nabla _ { \mathbf { z } } T = \prod _ { i } \operatorname* { d e t } \nabla _ { \mathbf { z } } G _ { i } } \end{array}$ so (9) is the pullback of the base density through T. We place the nodes by an EDM schedule (Karras et al., 2022), $t _ { i } = ( \sigma _ { i } - \sigma _ { \operatorname* { m i n } } ) / ( 1 - \sigma _ { \operatorname* { m i n } } )$ with $\begin{array} { r } { \sigma _ { i } = \big ( \sigma _ { \mathrm { m i n } } ^ { 1 / \rho } + \frac { i } { S } ( 1 - \sigma _ { \mathrm { m i n } } ^ { 1 / \rho } ) \big ) ^ { \rho } , } \end{array}$ , which concentrates the steps near the data for $\rho > 1$ and has been shown to outperform other schedules for few-step density evaluation (Rehman et al., 2026).

Conditioning. Although the mean flow only receives x as an additional input and regresses the same target $\mathbf { \epsilon } \mathbf { \epsilon } \mathbf { - } \mathbf { z } _ { 0 }$ as an unconditional model, we show that it learns the correct conditional flow map. Similar to conditional difusion models (Vincent, 2011; Batzolis et al., 2021; Arruda et al., 2026), the condition only restricts which targets $\mathbf { z } _ { 0 }$ contribute to the regression.

Proposition 1 (Conditional mean flows). Under the assumptions of Appendix $A . 1 ,$ every population minimizer of the unweighted loss ${ \mathcal { L } } _ { \mathrm { M F } }$ satisfies $\mathbf { u } _ { \psi } = \mathbf { u } ^ { * }$ for $p ( \mathbf { x } )$ -almost every $\mathbf { x } ,$ where $\mathbf { u } ^ { * }$ is the average velocity (3) of $\mathbf { v } ^ { * } ( \mathbf { z } , s , \mathbf { x } ) = \mathbb { E } [ \boldsymbol { \epsilon } - \mathbf { z } _ { 0 } \mid \mathbf { z } _ { s } = \mathbf { z } , \mathbf { x } ]$ . Hence, (9) equals log $p ( \mathbf { z } \mid \mathbf { x } ) ~ f o r$ every schedule.

The proof is given in Appendix A.1.

## 3.2 Sampling and Posterior Inference

For sampling, we draw $\mathbf { z } ^ { ( S ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ and apply the flow map backward along the same schedule, $\mathbf { z } ^ { ( i - 1 ) } =$ $\Phi _ { \psi } ( \mathbf { z } ^ { ( i ) } ; t _ { i } , t _ { i - 1 } , \mathbf { x } )$ for $i = S , \ldots , 1$ , so that $S$ steps yield a sample $\mathbf { z } ^ { ( 0 ) }$ . In the posterior direction, these are parameter draws conditional on the observed data; in the likelihood direction, they are observation draws conditional on the parameters. These samples follow the density in (9) only if the backward maps invert the forward maps $G _ { i } ,$ , which the cycle loss encourages.

In the likelihood direction, the joint log-posterior of N i.i.d. observations is proportional to log $p ( \pmb \theta ) +$ $\textstyle \sum _ { n = 1 } ^ { N }$ log $q _ { \psi } ^ { ( S ) } ( \mathbf { y } _ { n } \mid \pmb { \theta } )$ , which is diferentiable in θ for any fixed S with nonsingular step Jacobians. We expose it to PyMC as a distribution (Salvatier et al., 2016) and sample with nutpie’s NUTS on the JAX backend (Seyboldt et al., 2026).

Additionally, we can construct a two-level sampler as a trained mean flow provides a coarse likelihood $\stackrel { - } { q } _ { \psi } ^ { S _ { c } }$ with few steps and a fine likelihood $q _ { \psi } ^ { \mathrm { f } }$ by integrating its velocity head with an accurate ODE solver (1). This defines a coarse and a fine posterior,

$$
\begin{array} { r l } & { \pi _ { \mathrm { c } } ( \pmb { \theta } ) \propto p ( \pmb { \theta } ) \prod _ { n = 1 } ^ { N } q _ { \psi } ^ { S _ { c } } ( \mathbf { y } _ { n } \mid \pmb { \theta } ) , } \\ & { \pi _ { \mathrm { f } } ( \pmb { \theta } ) \propto p ( \pmb { \theta } ) \prod _ { n = 1 } ^ { N } q _ { \psi } ^ { \mathrm { f } } ( \mathbf { y } _ { n } \mid \pmb { \theta } ) , } \end{array}\tag{10}
$$

which we combine in two-level delayed acceptance MCMC (Christen and Fox, 2005; Lykkegaard et al., 2023). From the current state $\theta ,$ a subchain of J random-walk Metropolis steps targeting $\pi _ { \mathrm { c } }$ yields a proposal $\pmb { \theta } ^ { \prime }$ , which is accepted with probability

$$
\alpha ( \pmb \theta , \pmb \theta ^ { \prime } ) = \operatorname* { m i n } \left\{ 1 , \frac { \pi _ { \mathrm { f } } \left( \pmb \theta ^ { \prime } \right) \pi _ { \mathrm { c } } ( \pmb \theta ) } { \pi _ { \mathrm { f } } \left( \pmb \theta \right) \pi _ { \mathrm { c } } ( \pmb \theta ^ { \prime } ) } \right\} .\tag{11}
$$

Since the subchain is reversible with respect to $\pi _ { \mathrm { c } } ,$ this correction makes $\pi _ { \mathrm { f } }$ the invariant distribution (Lykkegaard et al., 2023), while the fine likelihood is evaluated only once per subchain instead of at every proposal.

## 3.3 Masked Transformers for Marginal Densities

Inspired by the Simformer (Gloeckler et al., 2024), we use masked attention to learn marginal densities $p ( \mathbf { z } _ { \mathcal { O } } \mid \mathbf { x } )$ for subsets O of the target coordinates. Our transformer tokenizes each coordinate of z and x and receives $( s , s - t )$ through adaptive layer normalization (Peebles and Xie, 2023). The mask excludes omitted target tokens, so outputs depend on z only through z<sub>O</sub>; the mean-flow and cycle losses are restricted to O.

Proposition 2 (Marginalization by masking). Under the assumptions of Appendix A.2, every population minimizer of the unweighted masked loss satisfies $\mathbf { u } _ { \psi , \mathcal { O } } = \mathbf { u } _ { \mathcal { O } } ^ { * }$ for every training mask O and $p ( \mathbf { x } )$ almost every $\mathbf { x } ,$ where $\mathbf { u } _ { O } ^ { * }$ is the average velocity of $\mathbf { v } _ { \mathcal { O } } ^ { * } ( \mathbf { z } _ { \mathcal { O } } , s , \mathbf { x } ) = \mathbb { E } [ \epsilon _ { \mathcal { O } } - \mathbf { z } _ { 0 , \mathcal { O } } \ | \ \mathbf { z } _ { s , \mathcal { O } } = \mathbf { z } _ { \mathcal { O } } , \mathbf { x } ]$ . Hence, (9) in |O| dimensions equals $\log p ( \mathbf { z } _ { \mathcal { O } } \mid \mathbf { x } )$ for every schedule.

The proof is given in Appendix A.2. Choosing $\mathbf { z } = \pmb { \theta }$ gives marginal posterior densities $p ( \pmb { \theta } _ { \mathcal { O } } \mid \mathbf { y } )$ ; choosing z = y gives marginal likelihoods $p ( \mathbf { y } _ { \mathcal { O } } \mid \mathbf { \theta } )$ . We evaluate the latter for missing observations in Section 4.3.

## 4 EXPERIMENTS

We evaluate direct posterior approximation on two moons and likelihood approximation on the three further benchmarks, where the learned densities enter MCMC. The benchmarks show the following:

• Two moons: few steps give accurate samples, so sampling has only a minor speed-accuracy tradeof, and densities improve with more steps (Section 4.1).

• Cognitive model: exchangeable likelihoods yield precision comparable to 200-step flow matching with about 200× faster MCMC (Section 4.2).

• Epidemic renewal: a masked transformer marginalizes missing observations and is more accurate than a normalizing flow (Section 4.3).

• g-and-k: at a shared budget, more draws beat more steps, and delayed acceptance MCMC is as accurate as NUTS without gradients (Section 4.4).

We compare samples with exact references using the classifier two-sample test (C2ST; Lopez-Paz and Oquab, 2017), and densities through the Hellinger distance $( \mathrm { A p \mathrm { - } }$ pendix B). Reference-free evaluations use normalized root mean squared error (NRMSE), efective sample size (ESS) and predictive checks, and the calibration for every trained estimator is assessed using tests of accuracy with random points (TARP; Lemos et al., 2023) prior to any other evaluation (Table S1). Tasks, architectures and generative models are described in Appendix B to E.

## 4.1 Few Steps Sufice for High Posterior Sample and Density Quality

Setup and baseline. The two moons task has a two-dimensional parameter and observation, a bimodal crescent posterior, and an exact likelihood and posterior sampler (Greenberg et al., 2019; Lueckmann et al., 2021). All methods learn the posterior direction and are sampled directly with 5,000 draws per dataset, which we score against 32,000 exact reference draws. We compare a conditional mean flow at step counts $S \in$ $\{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ with three baselines: a conditional normalizing flow, a conditional consistency model as the established few-step method (Song et al., 2023; Schmitt et al., 2024b), and a separately trained conditional flow matching model integrated with S Euler steps or an adaptive Tsit5 solver (Tsitouras, 2011). We integrate the velocity head of the mean flow with the same Euler or adaptive scheme, which repurposes the mean flow as a flow matching model. For both the consistency model and the mean flow, we tuned the sampling schedule for each step count on held-out observations, which are disjoint from the evaluation datasets.

Sample quality. The conditional mean flow reaches a similar C2ST at every step count (Figure 1 and Figure S1A), so a single step sufices for sampling and already outperforms the normalizing flow (0.523 against 0.543). In contrast, the consistency model needs at least 4 steps to approach this accuracy, and both flow matching variants need at least 32 Euler steps. The adaptive solver further improves the accuracy, but requires more than 300 solver steps. Hence, the mean flow with S=1 draws 5,000 samples about 23 times faster than flow matching with 32 Euler steps and 140 times faster than with the adaptive solver (Figure S2). It is also about 6 times faster than the normalizing flow and only reaches its sampling time at S=8.

Since the velocity head of the mean flow and the separately trained model perform similarly under both solvers, mean flow training does not seem to degrade the learned velocity field. The consistency model is also sensitive to its discretization exponent, whose candidates span a C2ST from 0.54 to 0.98 at S=4, whereas the EDM schedule of the mean flow changes the C2ST by at most 0.015 (Table S3).

Density quality. In contrast to sampling, we find that additional mean flow steps improve the density, and the Hellinger distance to the exact posterior decreases with S (Figure S1B and Table S4). With $S { = } 1$ the mean flow places mass between the two modes (Figure 1C), but it still outperforms the normalizing flow. Euler integration of either velocity field remains less accurate than the mean flow with $S { \ge } 2$ up to 32 steps. The adaptive solver is most accurate, with the velocity head of the mean flow ahead of the separately trained flow matching model.

![](images/dfad3a3933c1cbb119fb715c0686263ed9a1644260f3a48ad8e9392543b059e4.jpg)

![](images/333c437b47a770d2004f75d9db80c1c9a99720a138aed0488fce432f39cb18f8.jpg)

![](images/d0efd6b23e288665f06659fc22523805d34fa05cfc327baa3982fad841539971.jpg)  
Figure 2: Composing one learned trial likelihood over N trials of a cognitive model. (A) NRMSE of the composed posterior to the generating parameter. (B) Worst posterior predictive discrepancy across choice proportion, conditional response-time quantiles and timeout rate. (C) Posterior contraction relative to the prior. Lines show the mean over 10 independent datasets, bands the sample standard deviation.

Moreover, the mean flow density is close to normalized, whereas the Euler densities overshoot unit mass by many orders of magnitude at S=1 and remain far from normalized at S=32 (Table S4). We attribute this to the Euler scheme approximating both the trajectory and the divergence integral in (1), whereas the mean flow computes the exact Jacobian determinant of each step. The normalizing flow is exactly normalized, but its density shows fold artifacts and places mass outside the prior support (Figure 1A). Flow matching models with adaptive solvers provide the most accurate densities, but more than 300 steps per evaluation make them too expensive for repeated density evaluation.

## 4.2 Single-Step Likelihoods Yield Error Comparable to 200-Step Flow Matching

Setup and baseline. The cognitive model describes two-alternative decisions as a Wiener first-passage process with drift $\nu ,$ boundary separation a and nondecision time $t _ { 0 }$ (Ratclif et al., 2016), and is a standard benchmark for learned single-trial likelihoods (Fengler et al., 2021). We train one conditional mean flow on the signed response time $y = ( 2 c - 1 ) t$ , with choice $c \in \{ 0 , 1 \}$ and response time $t ,$ and compose the learned trial likelihood into joint posteriors over $N \in \{ 4 , 1 6 , 6 4 , 2 5 6 \}$ trials, which we sample with NUTS. Since a response cannot be faster than $t _ { 0 } ,$ , we truncate the prior of $t _ { 0 }$ at the fastest observed response time of each run. We compare mean flow step counts $S \in \{ 1 , 2 , 4 , 8 \}$ with a flow matching model integrated

with 200 Euler steps.

Results. With S=1, the NRMSE and posterior contraction of the mean flow are comparable to the flow matching baseline at every N (Figure 2, Table S5 and Figure S3), while sampling is 182 to 216× faster (Figure S2D). Additional steps increase the cost and the number of convergence failures without clearly reducing the parameter error (Figure S4). From all MCMC runs, 8.5% failed to converge, and all failed runs used $S \geq 4$ (Table S5). We attribute this to the fact that the mean flow is trained for few-step sampling and evaluation only, and hence the intermediate maps might be poorly trained and additional steps, unlike in the discretization of an ODE, need not reduce the approximation error.

However, for every method the coverage of the 95% central interval decreases and the worst posterior predictive discrepancy increases with N (Figure 2B). Composing many approximate likelihoods accumulates their errors while the posterior contracts, which makes even the 200-step baseline overconfident. With S=1, the additional discretization error further lowers the coverage at N=256, whereas with S=2 coverage and predictive discrepancy are comparable to the baseline.

## 4.3 Masked Transformer Backbone Reduces Error under Missing Observations

Setup and baseline. Since conditional mean flows do not constrain the network architecture, we can tailor the backbone to observations with missing entries. We consider a stochastic renewal process, a standard forward model for estimating time-varying reproduction numbers $R _ { t }$ (Fraser, 2007; Cori et al., 2013; Gostic et al., 2020), in which the incidence is driven by a generation-interval force (Appendix E.3 and Figure S5). The parameter is the 128-dimensional coeficient vector z with log $R = L \mathbf { z } $ , where L is the Cholesky factor of a squared-exponential kernel (Rasmussen and Williams, 2005), such that the prior is N(0, I). We withhold 30% of the observed steps completely at random and redraw this pattern for each dataset. Under this mechanism, inference only requires the marginal likelihood p(y<sub>O</sub> | z) of the observed entries O (Little and Rubin, 2019). We parameterize the mean flow with a masked transformer trained with missing coordinates and a causal mask over time. In contrast, the normalizing flow has a fixed input width, so we train and evaluate it with a value of −1 in the missing entries (Figure S5C). Both models are trained on a low budget of 8,192 simulations, and the mean flow is evaluated with S=1.

![](images/259809ea9cedaaef6eb073e24392e8c5d7a11f7b023a0acef4c13c889c1ae7da.jpg)  
Figure 3: Posterior predictive simulations of the renewal model on one dataset. Observations $\tilde { y }$ simulated from the learned likelihood at posterior draws of (A) our conditional mean flow with the masked transformer and (B) the normalizing flow with sentinel values; median with 50 and 90% bands against the observed data (black). The gray strip is the seeded window, and the striped columns mark the missing time points.

Table 1: Likelihoods with missing observations. Mean (sample standard deviation) over 10 datasets. $\tilde { y } _ { t }$ is simulated with the learned likelihood.
<table><tr><td></td><td>Mean flow</td><td>Normalizing flow</td></tr><tr><td>NRMSE</td><td>0.187 (0.008)</td><td>0.199 (0.009)</td></tr><tr><td>RMSE  $\tilde { y } _ { t }$ </td><td>0.055 (0.007)</td><td>0.159 (0.036)</td></tr><tr><td>RMSE yt</td><td>0.060 (0.013)</td><td>0.125 (0.031)</td></tr><tr><td>ESS</td><td>4,234 (247)</td><td>2,700 (629)</td></tr><tr><td>Leapfrog steps</td><td>121 (4)</td><td>480 (57)</td></tr><tr><td>Wall time (s)</td><td>36,146 (6,157)</td><td>1,624 (285)</td></tr></table>

Posterior accuracy and sampling cost. Both methods converge on every dataset without divergent transitions. The conditional mean flow with the masked transformer has a lower NRMSE to the generating coefficients than the normalizing flow, its posterior draws reproduce the observed $y _ { t }$ more closely under the learned likelihood and the simulator (Table 1), and its ESS per network evaluation is six times higher. However, each evaluation of the transformer is computationally more costly, such that the normalizing flow finishes sampling 22× sooner, despite requiring more leapfrog steps inside NUTS.

Posterior predictive. Simulations from the learned likelihoods at posterior draws reveal the problem of sentinel values (Figure 3). The predictive bands of the mean flow contract around the observed data at every observed time point. In contrast, the normalizing flow frequently generates the sentinel value −1 also at observed time points, which is reflected in a higher predictive RMSE (Table 1).

In summary, the flexible architecture of mean flows allows a masked transformer to marginalize missing observations. This yields a lower posterior error than a normalizing flow with sentinel values, but at a higher cost per density evaluation.

## 4.4 More Posterior Draws Are Better Than More Flow Steps

Setup and baseline. The g-and-k distribution is a standard benchmark with an intractable likelihood, defined through its quantile function (Rayner and MacGillivray, 2002; Drovandi and Pettitt, 2011). We use a bivariate version with a Gaussian copula and parameters $( g , k , \rho )$ , and datasets of N=50 observations (Appendix E.2). The conditional mean flow learns the likelihood of a single observation, and we sample the posterior with NUTS. To assess whether a computational budget should be spent on more mean flow steps or on more posterior draws, we fix the product of requested draws per chain and network evaluations per density (Appendix D). Under this budget, we compare mean flow step counts $S \in \{ 1 , 2 , 4 , 8 \}$ with a normalizing flow. In addition, we evaluate delayed acceptance MCMC, which screens proposals with the mean flow at $S _ { c }$ steps and accepts them against a 100-step Tsit5 evaluation of the velocity head of the same network (Section 3.2 and Appendix D). A separately trained flow matching model integrated with 200 Euler steps serves as a baseline outside the budget, with 28 times

![](images/c3e1667b717ac97db60f1932438960c1455aba084d44b9938f5faba3dd22555b.jpg)

![](images/1d31e0f3ea6068bf7254c9d71a25a8d335475ed1b58ef8a5a55552b7ab5190a2.jpg)

![](images/b3414c78543c7452916fc47577f667adfae9194168bdb26ab1a5d01bcb7aa765.jpg)  
Two-level DA (fine: mean flow as FM, S = 100) Flow matching, Euler S = 200, off budget  
Figure 4: Budget-matched posterior accuracy on g-and-k. (A) C2ST against the exact posterior over mean flow steps S, at the shared budget of Appendix D. (B) Minimum bulk ESS per 1,000 network evaluations. (C) Draws kept per chain under the shared budget. Lines show the mean over 10 independent datasets, bands the standard deviation; the delayed acceptance MCMC line is indexed by its coarse step count, and horizontal lines mark the normalizing flow and the flow matching model with 200 Euler steps outside the budget.

more network evaluations.

Steps against draws. All mean flow settings converge on every dataset and reach a similar accuracy, with the lowest mean C2ST at S=4, although these diferences are smaller than the spread across datasets (Figure 4A, Table S6). Fewer steps permit longer chains, and the minimum bulk ESS per network evaluation drops by an order of magnitude from S=1 to S=8 (Figure 4B–C). Hence, beyond S=4, a more accurate likelihood does not compensate for fewer posterior draws, and in practice, we recommend S=2 for this benchmark. Under the same budget, the normalizing flow has a higher C2ST than every mean flow setting and requires more than four times the wall time of the mean flow (S=2; Figure S2B). Regular flow matching also has a higher C2ST despite requiring more evaluations, and the same mean flow is about 50× faster.

Delayed acceptance MCMC. Screening proposals with a coarse mean flow and accepting them against the fine evaluation of the same network gives an accuracy comparable to NUTS (Table S6). With the coarse acceptance rate adapted to 0.3, the fine level accepts 80 to 85% of the evaluated proposals, rising with S<sub>c</sub>. Delayed acceptance MCMC is about 26–37 times faster than the flow matching baseline, but slower than NUTS with the mean flow likelihoods and yields a lower ESS.

In summary, a few mean flow steps with more posterior draws give a lower posterior error than both the normalizing flow and the flow matching baseline, at lower wall time. Delayed acceptance MCMC with the same network gives comparable accuracy without gradients.

## 5 DISCUSSION

Neural density estimators for SBI have so far traded flexibility for fast density evaluation. Conditional mean flows relax this trade-of by allowing the step count to be chosen after training. In low-dimensional benchmarks, one or two steps matched the accuracy of flow matching while reducing MCMC time by up to two orders of magnitude. Few steps also yielded accurate posterior samples, and under a fixed MCMC budget, more draws were more valuable than more steps.

Limitations. The main limitation is the cost of density evaluation in high dimensions. Each step requires its full Jacobian and a log-determinant, whose cost grows cubically with the dimension, and diferentiating the log-determinant for NUTS involves second-order derivatives. Together with the higher cost of each transformer evaluation, this made MCMC in the renewal application slower than with the normalizing flow despite using fewer network evaluations. In addition, the cycle loss encourages but does not guarantee invertibility, so the learned densities may not exactly be normalized. Finally, the coverage of every method, including flow matching with 200 steps, decreased at large trial counts, consistent with error accumulation.

Outlook. Beyond MCMC, fast mean flow densities can support self-consistency objectives (Schmitt et al., 2024a) and, together with prior and likelihood evaluations, evidence estimation (Kucharský et al., 2026). Since these methods also rely on repeated density evaluation, our work could make flexible flow-based estimators practical in any SBI setting that requires densities.

## References

T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama. Optuna: A next-generation hyperparameter optimization framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, pages 2623–2631, New York, NY, USA, 7 2019. Association for Computing Machinery. doi: 10.1145/3292500.3330701.

J. Arruda, Y. Schälte, C. Peiter, O. Teplytska, U. Jaehde, and J. Hasenauer. An amortized approach to non-linear mixed-efects modeling based on neural posterior estimation. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 1865–1901. PMLR, 7 2024. URL https://proceedings.mlr. press/v235/arruda24a.html.

J. Arruda, N. Bracher, U. Köthe, J. Hasenauer, and S. T. Radev. Difusion models in simulation-based inference: A tutorial review. Transactions on Machine Learning Research, 2026. URL https://openreview.net/forum?id=86Q261z005.

G. Batzolis, J. Stanczuk, C.-B. Schönlieb, and C. Etmann. Conditional image generation with score-based difusion models. arXiv preprint arXiv:2111.13606, 2021. doi: 10.48550/arxiv.2111.13606.

M. Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434, 2017. doi: 10.48550/arxiv.1701.02434.

J. Boelts, J.-M. Lueckmann, R. Gao, and J. H. Macke. Flexible and eficient simulation-based inference for models of decision-making. Elife, 11:e77220, 2022. doi: 10.7554/elife.77220.

J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, Y. Katariya, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs, 2018. URL http://github.com/jax-ml/jax.

P.-C. Bürkner, M. Scholz, and S. T. Radev. Some models are useful, but how do we know which ones? towards a unified Bayesian model taxonomy. Statistics Surveys, 17:216–310, 2023. doi: 10.1214/23-ss145.

B. Carpenter, A. Gelman, M. D. Hofman, D. Lee, B. Goodrich, M. Betancourt, M. Brubaker, J. Guo, P. Li, and A. Riddell. Stan: A probabilistic programming language. Journal of statistical

software, 76(1):1–32, 2017. doi: 10.18637/jss.v076.i01.

R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud. Neural ordinary diferential equations. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pages 6572–6583, Red Hook, NY, USA, 2018. Curran Associates Inc. URL https: //dl.acm.org/doi/10.5555/3327757.3327764.

T. Chen, V. Bansal, and J. G. Scott. Conditional difusions for amortized neural posterior estimation. In Y. Li, S. Mandt, S. Agrawal, and E. Khan, editors, Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 2377–2385. PMLR, 2025. URL https: //proceedings.mlr.press/v258/chen25d.html.

F. Chollet et al. Keras. https://keras.io, 2015.

J. A. Christen and C. Fox. Markov chain Monte Carlo using an approximation. Journal of Computational and Graphical Statistics, 14(4):795–810, 2005. doi: 10.1198/106186005x76983.

A. Cori, N. M. Ferguson, C. Fraser, and S. Cauchemez. A new framework and software to estimate time-varying reproduction numbers during epidemics. American journal of epidemiology, 178 (9):1505–1512, 2013. doi: 10.1093/aje/kwt133.

K. Cranmer, J. Brehmer, and G. Louppe. The frontier of simulation-based inference. Proceedings of the National Academy of Sciences, 117(48):30055–30062, 5 2020. doi: 10.1073/pnas.1912789117.

M. Deistler, J. Boelts, P. Steinbach, G. Moss, T. Moreau, M. Gloeckler, P. L. C. Rodrigues, J. Linhart, J. K. Lappalainen, B. K. Miller, P. J. Gonçalves, J.-M. Lueckmann, C. Schröder, and J. H. Macke. Simulation-Based Inference: A Practical Guide. arXiv preprint arXiv:2508.12939, 2025. doi: 10.48550/arxiv.2508.12939.

L. Dinh, J. Sohl-Dickstein, and S. Bengio. Density estimation using real NVP. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=HkpbnH9lx.

C. C. Drovandi and A. N. Pettitt. Likelihood-free Bayesian estimation of multivariate quantile distributions. Computational Statistics & Data Analysis, 55(9):2541–2556, 2011. doi: 10.1016/j.csda.2011.03.019.

C. Durkan, A. Bekasov, I. Murray, and G. Papamakarios. Neural spline flows. In Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran

Associates Inc., 2019. URL https://proceedings. neurips.cc/paper\_files/paper/2019/file/7ac 71d433f282034e088473244df8c02-Paper.pdf.

A. Fengler, L. N. Govindarajan, T. Chen, and M. J. Frank. Likelihood approximation networks (lans) for fast inference of simulation models in cognitive neuroscience. eLife, 10:e65074, 2021. doi: 10.7554/elife.65074.

C. Fraser. Estimating individual and household reproduction numbers in an emerging epidemic. PLoS one, 2(8):e758, 2007. doi: 10.1371/journal.pone.0000758.

Z. Geng, M. Deng, X. Bai, Z. Kolter, and K. He. Mean flows for one-step generative modeling. In Advances in Neural Information Processing Systems 38, pages 84032–84054, 2025. doi: 10.52202/085713-2534.

Z. Geng, Y. Lu, Z. Wu, E. Shechtman, J. Z. Kolter, and K. He. Improved mean flows: On the challenges of fastforward generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 30467–30476, 6 2026. URL https://openaccess.thecvf.com/co ntent/CVPR2026/html/Geng\_Improved\_Mean\_Flo ws\_On\_the\_Challenges\_of\_Fastforward\_Genera tive\_Models\_CVPR\_2026\_paper.html.

M. Gloeckler, M. Deistler, C. Weilbach, F. Wood, and J. H. Macke. All-in-one simulation-based inference. In Proceedings of the 41st International Conference on Machine Learning, ICML’24, pages 15735–15766. JMLR.org, 2024. URL https: //dl.acm.org/doi/10.5555/3692070.3692700.

K. M. Gostic, L. McGough, E. B. Baskerville, S. Abbott, K. Joshi, C. Tedijanto, R. Kahn, R. Niehus, J. A. Hay, P. M. De Salazar, et al. Practical considerations for measuring the efective reproductive number, R<sub>t</sub>. PLoS computational biology, 16(12):e1008409, 2020. doi: 10.1371/journal.pcbi.1008409.

D. Greenberg, M. Nonnenmacher, and J. Macke. Automatic posterior transformation for likelihood-free inference. In International conference on machine learning, pages 2404–2414. PMLR, 2019. URL https://proceedings.mlr.press/v97/gree nberg19a.html.

J. Gu, T. Chen, D. Berthelot, H. Zheng, Y. Wang, R. ZHANG, L. Dinh, M. Á. Bautista, J. M. Susskind, and S. Zhai. STARFlow: Scaling latent normalizing flows for high-resolution image synthesis. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, pages 134122–134158, 2025. doi: 10.52202/085713-4036.

M. D. Hofman and A. Gelman. The No-U-turn sampler: Adaptively setting path lengths in

Hamiltonian Monte Carlo. Journal of Machine Learning Research, 15(1):1593–1623, 2014. doi: 10.5555/2627435.2638586.

T. Karras, M. Aittala, S. Laine, and T. Aila. Elucidating the design space of difusion-based generative models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. doi: 10.52202/068431-1926.

Š. Kucharský and P.-C. Bürkner. Amortized Bayesian mixture models. Statistics and Computing, 36(4), 6 2026. doi: 10.1007/s11222-026-10911-y.

Š. Kucharský, A. Mishra, D. Habermann, S. T. Radev, and P.-C. Bürkner. Improving the accuracy of amortized model comparison with self-consistency. arXiv preprint arXiv:2508.20614, 2026. doi: 10.48550/arxiv.2508.20614.

L. Kühmichel, J. M. Huang, V. Pratz, J. Arruda, H. Olischläger, D. Habermann, S. Kucharsky, L. Elsemüller, A. Mishra, N. Bracher, et al. BayesFlow 2.0: Multi-backend amortized Bayesian inference in python. arXiv preprint arXiv:2602.07098, 3 2026. doi: 10.48550/arxiv.2602.07098.

P. Lemos, A. Coogan, Y. Hezaveh, and L. Perreault-Levasseur. Sampling-based accuracy testing of posterior estimators for general inference. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19256–19273. PMLR, 7 2023. URL https: //proceedings.mlr.press/v202/lemos23a.html.

Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=PqvMRDCJT9t.

R. J. Little and D. B. Rubin. Statistical analysis with missing data. Wiley Series in Probability and Statistics. John Wiley & Sons, 2019. doi: 10.1002/9781119482260.

X. Liu, C. Gong, and Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=XVjTT1nw5z.

D. Lopez-Paz and M. Oquab. Revisiting classifier two-sample tests. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=SJkXfE5xx.

C. Lu and Y. Song. Simplifying, stabilizing and scaling continuous-time consistency models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=LyJi5ugyJx.

J.-M. Lueckmann, J. Boelts, D. Greenberg, P. Goncalves, and J. Macke. Benchmarking simulation-based inference. In International conference on artificial intelligence and statistics, pages 343–351. PMLR, 2021. URL http://procee dings.mlr.press/v130/lueckmann21a.html.

M. B. Lykkegaard, T. J. Dodwell, C. Fox, G. Mingas, and R. Scheichl. Multilevel delayed acceptance MCMC. SIAM/ASA Journal on Uncertainty Quantification, 11(1):1–30, 2023. doi: 10.1137/22m1476770.

O. A. Martin, O. Abril-Pla, J. Deklerk, S. D. Axen, C. Carroll, A. Hartikainen, and A. Vehtari. ArviZ: a modular and flexible library for exploratory analysis of Bayesian models. Journal of Open Source Software, 11(119):9889, 2026. doi: 10.21105/joss.09889.

A. Mishra, D. Habermann, M. Schmitt, S. Radev, and P.-C. Bürkner. Robust amortized Bayesian inference with self-consistency losses on unlabeled data. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 9374–9405, 2026. URL https://proceedings.iclr.cc/paper\_files/pa per/2026/hash/0fd3d8093b3ba73d19b393a1326f dba7-Abstract-Conference.html.

G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021. URL http://jmlr.org/papers/v22/19-1028.html.

W. Peebles and S. Xie. Scalable difusion models with transformers. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 4172–4182, 2023. URL https: //doi.org/10.1109/ICCV51070.2023.00387.

E. Perez, F. Strub, H. De Vries, V. Dumoulin, and A. Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 4 2018. doi: 10.1609/aaai.v32i1.11671.

T. P. Prescott and R. E. Baker. Multifidelity approximate Bayesian computation with sequential Monte Carlo parameter sampling. SIAM/ASA Journal on Uncertainty Quantification, 9(2): 788–817, 1 2021. doi: 10.1137/20m1316160.

S. T. Radev, M. Schmitt, L. Schumacher, L. Elsemüller, V. Pratz, Y. Schälte, U. Köthe, and P.-C. Bürkner. BayesFlow: Amortized Bayesian workflows with neural networks. Journal of Open Source Software, 8(89):5702, 2023. doi: 10.21105/joss.05702.

C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. The MIT Press, 11 2005. doi: 10.7551/mitpress/3206.001.0001.

R. Ratclif, P. L. Smith, S. D. Brown, and G. McKoon. Difusion decision model: Current issues and history. Trends in cognitive sciences, 20(4):260–281, 2016. doi: 10.1016/j.tics.2016.01.007.

G. D. Rayner and H. L. MacGillivray. Numerical maximum likelihood estimation for the g-and-k and generalized g-and-h distributions. Statistics and computing, 12(1):57–75, 2002. doi: 10.1023/a:1013120305780.

D. Rehman, T. Akhound-Sadegh, A. Gazizov, Y. Bengio, and A. Tong. FALCON: Few-step accurate likelihoods for continuous flows. In C. Vondrick, B. Hariharan, C. Rafel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 70449–70473, 2026. URL https://proceedings.iclr.cc/paper\_files/pa per/2026/file/72bcad2c3fe7625fc233fabe84a6 d7ef-Paper-Conference.pdf.

J. Salvatier, T. V. Wiecki, and C. Fonnesbeck. Probabilistic programming in Python using PyMC3. PeerJ Computer Science, 2016:e55, 2016. doi: 10.7717/peerj-cs.55.

M. Schmitt, D. R. Ivanova, D. Habermann, U. Koethe, P.-C. Bürkner, and S. T. Radev. Leveraging self-consistency for data-eficient amortized Bayesian inference. In R. Salakhutdinov, Z. Kolter, K. Heller, A. Weller, N. Oliver, J. Scarlett, and F. Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 43723–43741. PMLR, 7 2024a. URL https://proceedings.mlr.press/v235/schmitt 24a.html.

M. Schmitt, V. Pratz, U. Köthe, P.-C. Bürkner, and S. T. Radev. Consistency models for scalable and fast simulation-based inference. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, pages 126908–126945, Red Hook, NY, USA, 2024b. Curran Associates Inc. doi: 10.52202/079017-4030.

A. Seyboldt, E. L. Carlson, and B. Carpenter. Preconditioning Hamiltonian Monte Carlo by minimizing Fisher divergence. arXiv preprint

arXiv:2603.18845, 2026. doi: 10.48550/arxiv.2603.18845.

L. Sharrock, J. Simons, S. Liu, and M. Beaumont. Sequential neural score estimation: likelihood-free inference with conditional score based difusion models. In Proceedings of the 41st International Conference on Machine Learning, ICML’24, pages 44565–44602. JMLR.org, 2024. URL https: //dl.acm.org/doi/10.5555/3692070.3693884.

S. A. Sisson, Y. Fan, and M. Beaumont. Handbook of approximate Bayesian computation. Chapman and Hall/CRC, 2018. doi: 10.1201/9781315117195.

Y. Song, P. Dhariwal, M. Chen, and I. Sutskever. Consistency models. In Proceedings of the 40th International Conference on Machine Learning, ICML’23, pages 32211–32252. JMLR.org, 2023. URL https: //dl.acm.org/doi/10.5555/3618408.3619743.

C. Tatsuoka, M. Yang, D. Xiu, and G. Zhang. Bifidelity parameter estimation using conditional difusion models. SIAM/ASA Journal on Uncertainty Quantification, 14(3):743–778, 7 2026. doi: 10.1137/25m1734567.

C. Tsitouras. Runge–kutta pairs of order 5(4) satisfying only the first column simplifying assumption. Computers & mathematics with applications, 62(2):770–775, 2011. doi: 10.1016/j.camwa.2011.06.002.

A. Vehtari, A. Gelman, D. Simpson, B. Carpenter, and P.-C. Bürkner. Rank-normalization, folding, and localization: An improved R<sup>ˆ</sup> for assessing convergence of MCMC (with discussion). Bayesian analysis, 16(2):667–718, 2021. doi: 10.1214/20-ba1221.

P. Vincent. A connection between score matching and denoising autoencoders. Neural computation, 23(7): 1661–1674, 2011. doi: 10.1162/neco\_a\_00142.

Z. Wang, J. Hasenauer, and Y. Schälte. Missing data in amortized simulation-based neural posterior estimation. PLoS Computational Biology, 20(6): e1012184, 2024. doi: 10.1371/journal.pcbi.1012184.

J. Wildberger, M. Dax, S. Buchholz, S. R. Green, J. H. Macke, and B. Schölkopf. Flow matching for scalable simulation-based inference. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, pages 16837–16864, Red Hook, NY, USA, 2023. Curran Associates Inc. doi: 10.52202/075280-0737.

V. K. Yalavarthi, R. Scholz, C. Kloetergens, K. Madhusudhanan, S. Born, and L. Schmidt-Thieme. Marginalization consistent probabilistic forecasting of irregular time series via

mixture of separable flows. arXiv preprint arXiv:2406.07246, 2024. doi: 10.48550/arxiv.2406.07246.

Y. Ye, A. Pandey, C. Bawden, D. M. Sumsuzzman, R. Rajput, A. Shoukat, B. H. Singer, S. M. Moghadas, and A. P. Galvani. Integrating artificial intelligence with mechanistic epidemiological modeling: a scoping review of opportunities and challenges. Nature Communications, 16(1):581, 2025. doi: 10.1038/s41467-024-55461-x.

S. Zhai, R. Zhang, P. Nakkiran, D. Berthelot, J. Gu, H. Zheng, T. Chen, M. Á. Bautista, N. Jaitly, and J. M. Susskind. Normalizing flows are capable generative models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=2uheUFcFsM.

## A PROOFS

## A.1 Conditional Mean Flows

Conditioning reduces the main population objective to an unconditional mean flow objective for each conditional target, as in conditional difusion and flow matching (Vincent, 2011; Batzolis et al., 2021; Arruda et al., 2026). We write $\Psi _ { s \to t } ^ { \mathbf { x } }$ for the ODE flow of $\mathbf { v } ^ { * }$ and $\mathbf { u } ^ { * }$ for its average velocity, defined by $\Psi _ { s \to t } ^ { \mathbf { x } } ( \mathbf { z } ) = \mathbf { z } + ( t - s ) \mathbf { u } ^ { * } ( \mathbf { z } , s , t , \mathbf { x } )$ with diagonal value $\mathbf { u } ^ { * } ( \mathbf { z } , s , s , \mathbf { x } ) = \mathbf { v } ^ { * } ( \mathbf { z } , s , \mathbf { x } )$

## Assumptions.

(A1) Noise. $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ is independent of $( { \bf z } _ { 0 } , { \bf x } )$ , and $\mathbb { E } \| \mathbf { z } _ { 0 } \| ^ { 2 } < \infty$

(A2) Time pairs. $( s , t )$ is independent of $( \mathbf { z } _ { 0 } , \mathbf { x } , \epsilon )$ , and its distribution has an absolutely continuous component whose density is strictly positive on $( 0 , 1 ) ^ { 2 }$

(A3) Regularity. For $p ( \mathbf { x } )$ -almost every x, the conditional target $p ( \mathbf { z } _ { 0 } \mid \mathbf { x } )$ has a density, and $\mathbf { v } ^ { * } ( \cdot , \cdot , \mathbf { x } )$ is continuously diferentiable in state and time and globally Lipschitz in the state, uniformly over $[ 0 , 1 ]$

(A4) Realizability. The admissible pairs $\left( \mathbf { u } _ { \psi } , \hat { \mathbf { v } } _ { \psi } \right)$ are jointly measurable in x, continuously diferentiable in state and time, and have finite population loss, and they contain the exact pair $( \mathbf { u } ^ { * } , \mathbf { v } ^ { * } )$ simultaneously for al conditions.

(A3) gives a global continuously diferentiable flow through both endpoints. For targets with compact support or discontinuous densities, such as the uniform priors in two moons, $\mathbf { v } ^ { * }$ is in general not Lipschitz near $t { = } 0$ . For such targets, the argument below applies on [δ, 1] for every $\delta > 0$ and, for schedules starting at $t _ { 0 } = \delta$ , yields the density of the slightly smoothed target ${ \bf z } _ { \delta } = ( 1 - \delta ) { \bf z } _ { 0 } + \delta \epsilon$ , which converges to the target as $\delta  0$ . In practice, we do not truncate the schedule and keep $t _ { 0 } = 0$ , since $s = 0$ is almost surely never sampled, and the network’s value at $s = 0$ is the continuous extension of the fitted values at small s.

Proposition 1 (Conditional mean flows). Assume $( A 1 ) { - } ( A 4 )$ , and let $\mathcal { L } _ { \mathrm { M F } }$ be unweighted (w=1). Then every population minimizer satisfies, for p(x)-almost every x,

$$
\begin{array} { r } { \hat { \mathbf { v } } _ { \psi } ( \mathbf { z } , s , \mathbf { x } ) = \mathbf { v } ^ { * } ( \mathbf { z } , s , \mathbf { x } ) \quad a n d \quad \mathbf { u } _ { \psi } ( \mathbf { z } , s , t , \mathbf { x } ) = \mathbf { u } ^ { * } ( \mathbf { z } , s , t , \mathbf { x } ) \qquad f o r ~ a l l \ \mathbf { z } \in \mathbb { R } ^ { d } , ~ s , t \in [ 0 , 1 ] . } \end{array}
$$

Consequently, for every schedule $0 = t _ { 0 } < \cdot \cdot \cdot < t _ { S } = 1$ , the composed map T equals $\Psi _ { 0 \to 1 } ^ { \mathbf { x } }$ , a difeomorphism that transports $p ( \mathbf { z } _ { 0 } \mid \mathbf { x } )$ to $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , and (9) equals log $p ( \mathbf { z } \mid \mathbf { x } )$

Proof of Proposition 1. Let ${ \bf Y } = { \bf \epsilon } { \bf \epsilon } - { \bf z } _ { 0 }$ and m $\mathbf { \xi } = \mathbf { v } ^ { * } ( \mathbf { z } _ { s } , s , \mathbf { x } )$ . By (A1) and (A2), E $[ \mathbf { Y } \mid \mathbf { z } _ { s } , s , t , \mathbf { x } ] = \mathbf { m }$ , and (A1) ensures that all terms are finite. Both predictions in the main loss are functions of these inputs, so the conditional-expectation decomposition gives

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M F } } = \big ( 1 + \frac { 1 } { d } \big ) \mathbb { E } \| { \bf Y } - { \bf m } \| ^ { 2 } + \mathbb { E } \| { \bf V } _ { \psi } - { \bf m } \| ^ { 2 } + \frac { 1 } { d } \mathbb { E } \| \hat { \bf v } _ { \psi } - { \bf m } \| ^ { 2 } . } \end{array}
$$

By (A4) and the MeanFlow identity, the exact pair $\mathbf { ( u ^ { * } , v ^ { * } ) }$ makes the last two terms zero. Every minimizer therefore satisfies $\hat { \mathbf { v } } _ { \psi } = \mathbf { V } _ { \psi } = \mathbf { v } ^ { * }$ almost everywhere under the training distribution. For almost every condition, the Gaussian interpolation has full support at interior times. Together with the time-pair density and continuity, this extends the equalities to all states and times.

Fix such a condition and a trajectory $\mathbf { z } _ { \tau } = \Psi _ { s  \tau } ^ { \mathbf { x } } ( \mathbf { z } )$ . Since $\hat { \mathbf { v } } _ { \psi } = \mathbf { v } ^ { * }$ , the total derivative in $\mathbf { V } _ { \psi }$ is taken along this trajectory, and for fixed t,

$$
\frac { \mathrm { d } } { \mathrm { d } \tau } \Phi _ { \psi } ( \mathbf { z } _ { \tau } ; \tau , t , \mathbf { x } ) = \mathbf { v } ^ { * } ( \mathbf { z } _ { \tau } , \tau , \mathbf { x } ) - \mathbf { V } _ { \psi } ( \mathbf { z } _ { \tau } , \tau , t , \mathbf { x } ) = \mathbf { 0 } .
$$

Since $\Phi _ { \psi } ( \mathbf { z } _ { t } ; t , t , \mathbf { x } ) = \mathbf { z } _ { t }$ , we obtain $\Phi _ { \psi } = \Psi ^ { \bf x }$ and hence $\mathbf { u } _ { \psi } = \mathbf { u } ^ { * }$ , with the diagonal identity following by continuity. Since $\mathbf { z } _ { 1 } = \epsilon .$ , the flow of $\mathbf { v } ^ { * }$ transports $p ( \mathbf { z } _ { 0 } \mid \mathbf { x } ) \mathrm { t o } \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ under (A3) (Lipman et al., 2023; Liu et al., 2022). Flow composition gives $T = \Psi _ { 0 \to 1 } ^ { \mathbf { x } }$ for every schedule, and change of variables gives (9). □

The exact flow map satisfies $\Psi _ { t \to s } ^ { \mathbf { x } } \circ \Psi _ { s \to t } ^ { \mathbf { x } } =$ id for every condition, including swapped ones, so the exact pair also attains the minimum ${ \mathcal { L } } _ { \mathrm { c y c } } = 0 .$ . Hence, adding $\lambda \mathcal { L } _ { \mathrm { c y c } }$ does not change the set of population minimizers. For $p > 0$ , the weight w depends on the regression target Y through the residual, and the decomposition above no longer applies. The adaptive weight thus acts as a robust loss that down-weights large residuals (Geng et al., 2025), and in general changes the learned velocity, but empirically seems to stabilize training (see Table S2).

## A.2 Marginalization by Masking

We show that the masked objective of Section 3.3 targets the marginal distribution of the retained target coordinates. For a set $O \subseteq \{ 1 , \ldots , d \}$ , we write $\mathbf { z } _ { \mathcal { O } }$ for the retained coordinates of z, and $\mathbf { u } _ { \psi , \mathcal { O } }$ and $\hat { \mathbf { v } } _ { \psi , \mathcal { O } }$ for the outputs of the network on O.

Proposition 2 (Marginalization by masking). Let the mask $O \subseteq \{ 1 , \ldots , d \}$ be nonempty with law π, independent $o f \left( \mathbf { z } _ { 0 } , \mathbf { x } , \mathbf { \epsilon } , s , t \right)$ , let the network outputs on O depend on the target only through $\mathbf { z } _ { \mathcal { O } }$ , and let the unweighted loss be restricted to the coordinates in O. Assume $\ ( A 1 ) - ( A { \it 3 } )$ for each marginal target $\mathbf { z } _ { 0 , \mathcal { O } }$ with $\pi ( \mathcal { O } ) > 0$ , and $( A \ 4 )$ simultaneously for all such masks. Then every population minimizer satisfies, for every O with $\pi ( \mathcal { O } ) > 0$ and $p ( \mathbf { x } )$ -almost every x, $\hat { \mathbf { v } } _ { \psi , \mathcal { O } } = \mathbf { v } _ { \mathcal { O } } ^ { * }$ and $\mathbf { u } _ { \psi , \mathcal { O } } = \mathbf { u } _ { \mathcal { O } } ^ { * }$ , where $\mathbf { u } _ { O } ^ { * }$ is the average velocity of

$$
\begin{array} { r } { { \bf v } _ { \mathcal { O } } ^ { * } ( { \bf z } _ { \mathcal { O } } , s , { \bf x } ) = \mathbb { E } [ \epsilon _ { \mathcal { O } } - { \bf z } _ { 0 , \mathcal { O } } \mid { \bf z } _ { s , \mathcal { O } } = { \bf z } _ { \mathcal { O } } , { \bf x } ] . } \end{array}
$$

Consequently, (9), with the base density and Jacobians evaluated in |O| dimensions, equals log $p ( \mathbf { z } _ { \mathcal { O } } \mid \mathbf { x } )$ for every schedule.

Proof of Proposition 2. For a fixed mask, independence leaves the simulation law unchanged, and the coordinatewise interpolation is

$$
\begin{array} { r } { \mathbf { z } _ { s , \mathcal { O } } = ( 1 - s ) \mathbf { z } _ { 0 , \mathcal { O } } + s \epsilon _ { \mathcal { O } } , \qquad \epsilon _ { \mathcal { O } } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { | \mathcal { O } | } ) . } \end{array}
$$

The masked main loss is therefore a positively weighted sum of the conditional mean flow objectives for the marginal targets. By the simultaneous realizability in (A4), the exact pairs attain every summand’s lower bound so every minimizer attains each one. Proposition 1 applied to $\mathbf { z } _ { 0 , \mathcal { O } }$ gives the result. □

Additional attention restrictions are covered only when they preserve the ability to represent $\mathbf { v } _ { O } ^ { * }$ and its average velocity. The independence of the synthetic training mask holds by construction and does not impose a missingness mechanism on the observed data. The marginal density uses only the observed coordinates. For $p > 0 ,$ , the residual and hence w depend only on the retained coordinates, so the reduction to the marginal targets still holds, but each summand then has the weighted minimizer discussed in Appendix A.1.

## B EVALUATION METRICS

All experiments use 10 independent datasets, and we report the mean and standard deviation of the following metrics over datasets.

Classifier two-sample test (C2ST). The C2ST compares two distributions through a classification task (Lopez-Paz and Oquab, 2017). If the approximate and the reference posterior coincide, no classifier can tell their draws apart better than chance, whereas any diference between them can be exploited by a suficiently flexible classifier. Hence, the held-out accuracy aˆ of a trained classifier measures how distinguishable the two distributions are: $\hat { a } = 0 . 5$ indicates indistinguishable and $\hat { a } = 1$ perfectly separable draws. We label approximate draws with 1 and reference draws with 0, subsample both sets without replacement to the same size, which is fixed within each experiment across all methods and step counts, standardize the pooled draws, and train a classifier $c ( \pmb { \theta } ) \in ( 0 , 1 )$ with the binary cross-entropy. The classifier is an MLP with two residual hidden layers of width $2 \left. { \lceil \log _ { 2 } \left( \mathrm { \bar { 1 } 0 } D \right) \right\rceil }$ for D-dimensional draws, Mish activations, dropout of 0.1 and a sigmoid output unit. We train it with AdamW (learning rate $5 \times 1 0 ^ { - 4 }$ with linear warmup and cosine decay, weight decay $5 \times 1 0 ^ { - 3 }$ , gradient clipping at norm 1.5) and batch size 128 for at most 100 epochs, with early stopping (patience 5) on 20% of the training data. The accuracy of the classifier $c ( \pmb \theta )$ on held-out draws is averaged over 5-fold cross-validation.

Grid density metrics. On two moons, we evaluate the exact posterior $p$ and each learned density q on a 512×512 midpoint grid over the prior support with cell area $\Delta A .$ . We report the total mass $\begin{array} { r } { M = \sum _ { j } q _ { j } \Delta A } \end{array}$ of the learned density and compare the shapes of both densities after normalizing them to cell probabilities $\tilde { p } _ { j }$ and $\tilde { q } _ { j }$ on the grid,

$$
\mathrm { K L } = \sum _ { j } \tilde { p } _ { j } \log \frac { \tilde { p } _ { j } } { \tilde { q } _ { j } } , \qquad H = \biggl ( \frac { 1 } { 2 } \sum _ { j } \big ( \sqrt { \tilde { p } _ { j } } - \sqrt { \tilde { q } _ { j } } \big ) ^ { 2 } \biggr ) ^ { 1 / 2 } .
$$

The Hellinger distance $H \in [ 0 , 1 ]$ remains finite when the learned density vanishes where the exact posterior has mass, whereas the Kullback-Leibler (KL) divergence becomes infinite.

Normalized RMSE and coverage. For posterior draws $\hat { \pmb { \theta } } ^ { ( l ) } , l = 1 , \dots , L$ , and the generating parameter $\pmb { \theta } ^ { * }$ of dimension D, we compute

$$
\mathrm { N R M S E } = \Bigg ( \frac { 1 } { L D } \sum _ { l = 1 } ^ { L } \sum _ { j = 1 } ^ { D } \Big ( \frac { \hat { \theta } _ { j } ^ { ( l ) } - \theta _ { j } ^ { * } } { r _ { j } } \Big ) ^ { 2 } \Bigg ) ^ { 1 / 2 } ,
$$

where $r _ { j }$ is the prior range of parameter $j .$ Coverage is the fraction of parameters whose generating value lies within the central 95% credible interval.

Posterior contraction. We define posterior contraction as the relative reduction in variance from prior to posterior,

$$
\mathrm { C o n t r a c t i o n } = 1 - \frac { 1 } { D } \sum _ { j = 1 } ^ { D } \frac { \mathrm { V a r } _ { \mathrm { p o s t } } ( \theta _ { j } ) } { \mathrm { V a r } _ { \mathrm { p r i o r } } ( \theta _ { j } ) } ,
$$

which measures how much the data reduce the prior uncertainty, but not whether the posterior is correct.

TARP. TARP estimates the expected coverage of credible regions around random reference points (Lemos et al., 2023) directly from the trained estimator, without MCMC or reference posteriors. We report the area between the TARP curve and the diagonal (ATC; 0 is calibrated) and the p-value of the Kolmogorov-Smirnov test, using 2,000 validation pairs split into 10 folds of 200 pairs with 200 draws each (Table S1). On two moons, TARP assesses the learned posterior; for the other experiments, it assesses whether the learned likelihood generates calibrated data.

Table S1: TARP calibration of every trained estimator. Area between the TARP curve and the diagonal (ATC; 0 is calibrated, positive is overdispersed), mean (standard deviation) over 10 folds of 200 prior-predictive validation pairs (2,000 in total) with 200 draws each. Two moons calibrates the posterior; g-and-k, the renewal model and the DDM calibrate the likelihood. † marks methods whose Kolmogorov-Smirnov p-value is below 0.01; the ATC closest to zero per experiment is in bold.
<table><tr><td>Experiment</td><td>Method</td><td colspan="2">ATC</td></tr><tr><td rowspan="5">Two moons</td><td>Normalizing flow</td><td>-0.008 (0.019)</td><td></td></tr><tr><td>Mean flow,  $\scriptstyle \mathbf { \bar {  { S } } } = 1 \ / \ . \mathbf { \bar {  { S } } } = 2 \ / \ S = 3 2$ </td><td></td><td>-0.001 (0.010) / +0.007 (0.011) / +0.008 (0.011)</td></tr><tr><td>Consistency model, S=1 / S=2 / S=32</td><td></td><td>+0.102 (0.011)† / +0.113 (0.011) / +0.001 (0.014)</td></tr><tr><td>Mean flow (v-head), Euler S=1 / S=2 / S=32</td><td></td><td>−0.179 (0.013)† / −0.134 (0.015) / −0.007 (0.012)</td></tr><tr><td>Flow matching, Euler S=1 / S=32 Mean flow (v-head) / flow matching, adaptive</td><td>−0.176 (0.011)† / −0.004 (0.014)</td><td></td></tr><tr><td rowspan="3">g-and-k</td><td></td><td>–0.007 (0.012)</td><td>/ +0.000 (0.011)</td></tr><tr><td>Mean flow, S=1 / S=8 Normalizing flow</td><td>-0.004 (0.012)</td><td>/ −0.009 (0.011)</td></tr><tr><td>Flow matching, Euler S=200</td><td>+0.001 (0.011) +0.004 (0.007)</td><td></td></tr><tr><td rowspan="2">Renewal</td><td>Mean flow, masked transformer</td><td>-0.025 (0.013)</td><td></td></tr><tr><td>Normalizing flow, sentinel</td><td>+0.130 (0.011)†</td><td></td></tr><tr><td rowspan="2">DDM</td><td>Mean flow, S=1 / S=8</td><td></td><td></td></tr><tr><td>Flow matching, Euler S=200</td><td>-0.003 (0.009) –0.003 (0.016)</td><td>/ -0.006 (0.013)</td></tr></table>

Posterior predictive checks. For the DDM, we simulate a replicate dataset for each of 200 posterior draws and compare the observed choice proportion, the conditional response-time quantiles (10, 30, 50, 70 and 90%) and the timeout rate T with their replicate distribution. For each summary, we compute the discrepancy $z = \left( T ( \mathbf { y } _ { \mathrm { o b s } } ) - \bar { T } _ { \mathrm { r e p } } \right) / \mathrm { s d } ( T _ { \mathrm { r e p } } )$ and the predictive p-value $P ( T ( \mathbf { y } _ { \mathrm { r e p } } ) \geq T ( \mathbf { y } _ { \mathrm { o b s } } ) )$ , and report the largest |z| across summaries.

Convergence diagnostics. We compute $\mathrm { s p l i t } - { \hat { R } }$ and bulk ESS following Vehtari et al. (2021). A run passes our convergence criteria if max $\hat { R } \leq 1 . 0 5$ , the minimum bulk ESS is at least 100 and at least 500 draws are kept. Reported means and standard deviations include runs that fail these criteria.

## C ARCHITECTURES, TRAINING AND HYPERPARAMETER SEARCH

All networks and diagnostics are implemented in BayesFlow 2.0.14 (Radev et al., 2023; Kühmichel et al., 2026), where we added the conditional mean flow as an additional inference network.

## Architectures.

• Mean flow: An MLP with 5 hidden layers of width 256. The condition x is embedded by a linear layer and concatenated with the target z at the input, and, together with a Fourier embedding with scale 0.3 of the time pair $( s , s - t )$ , it modulates every residual block (Perez et al., 2018). The auxiliary velocity head shares all but the last 2 layers.

• Masked transformer: A transformer (Peebles and Xie, 2023) with 5 blocks of width 128 and four attention heads. It tokenizes each observation and parameter and receives the time pair $( s , s - t )$ through adaptive layer normalization. During training, each observation of each simulation is dropped independently with probability 0.3, redrawn for every batch. Dropped tokens are excluded from attention and loss. A causal mask additionally restricts each observation token to parameter and observation tokens up to its own time step. The velocity head branches from the last 2 blocks and uses the same masks. Each observation and parameter token receives a learned embedding of its coordinate index, which for observations is the time step, and a Fourier embedding is used only for the time pair.

• Consistency model: A continuous-time consistency model (Lu and Song, 2025) with an MLP of 6 hidden layers of width 256.

• Flow matching: An MLP with the same depth and width as the mean flow, without the velocity head.

• Normalizing flow: A spline coupling flow (Durkan et al., 2019) with 5 coupling layers, each with 2 hidden layers of width 256, and with 2 coupling layers, each with 2 hidden layers of width 128, for the renewal application.

Training. We trained all networks ofline on standardized parameters and observations with a batch size of 256, using AdamW with weight decay $5 \times 1 0 ^ { - 3 }$ , gradient clipping at norm 1.5, and a learning rate warmed up linearly over the first 5% of steps from $5 \times 1 0 ^ { - 5 }$ to $5 \times 1 0 ^ { - 4 }$ and then decayed to zero with a cosine schedule. We used 50,176 simulations for two moons and the DDM, 100,000 for g-and-k, and 8,192 for the renewal model to compare architectures under limited data, with 1,024 further simulations for validation. The conditional mean flow, the consistency model, and the flow matching model were trained for 1,000 epochs, and the normalizing flow for 200 epochs with early stopping (patience 10).

Hyperparameter search. We selected the training hyperparameters of the conditional mean flow with a search in Optuna (Akiba et al., 2019) on two benchmarks: two moons and a three-dimensional variant of the Gaussian linear task with uniform prior. Table S2 lists the search space and the selected values, which all experiments use. The training times are $s = u _ { s } ^ { 1 + \alpha }$ and $t = u _ { t } ^ { 1 + \alpha }$ with $u _ { s } , u _ { t } \sim \mathcal { U } ( 0 , 1 )$ , and for 0 unshared layers the velocity head reads the velocity from the boundary case $\mathbf { u } _ { \psi } ( \mathbf { z } , s , s , \mathbf { x } )$ . Each trial trained one network per benchmark for 250 epochs of 100 freshly simulated batches of size 256 and evaluated it with S=4 steps on 256 held-out datasets with 500 posterior draws each. The score of a benchmark was the sum of an accuracy term, the calibration error, and the grid Hellinger distance to the exact posterior. The accuracy term is 2 (C2ST − 0.5) on two moons and the NRMSE on the Gaussian task, and the calibration error is the median absolute diference between the nominal and empirical coverage of central credible intervals at 20 levels. The C2ST was averaged over 15 and the Hellinger distance over 3 of the held-out datasets. The search minimized the sum of both benchmark scores with the multivariate tree-structured Parzen estimator of Optuna (Akiba et al., 2019) on 200 trials.

Table S2: Hyperparameters of the conditional mean flow. Search space of the Optuna search and the selected configuration used in all experiments.
<table><tr><td>Hyperparameter</td><td>Search space</td><td>Selected</td></tr><tr><td>Cycle weight λ (log)</td><td>[1, 100]</td><td>2.5</td></tr><tr><td>Probability of replacing the condition in  $\mathcal { L } _ { \mathrm { c y c } }$ </td><td>[0,0.8]</td><td>0.6</td></tr><tr><td>Fraction of time pairs with t=s</td><td>[0.2,0.8]</td><td>0.7</td></tr><tr><td>Exponent p of the adaptive weight</td><td>{0, 0.25, 0.5, 0.75, 1}</td><td>0.25</td></tr><tr><td>Offset c of the adaptive weight (log)</td><td> $[ 1 0 ^ { - 5 } , 1 ]$ </td><td>0.1</td></tr><tr><td>Scale of the Fourier time embedding (log)</td><td>[0.02, 30]</td><td>0.3</td></tr><tr><td>Exponent α of the training times</td><td>[−0.5,4]</td><td>0.5</td></tr><tr><td>Final layers not shared by the velocity head</td><td> $\left\{ 0 , \ldots , \bar { 5 } \right\}$ </td><td>2</td></tr><tr><td>Adaptive weight on the velocity head</td><td>{yes, no}</td><td>no</td></tr></table>

Sampling schedule. For two moons, we tuned the sampling schedule of the consistency model and the mean flow separately for each step count, using the C2ST on 2,000 draws for 3 held-out observations. For the consistency model, we selected the discretization exponent from $\rho \in \{ 1 , 2 , 3 , 3 . 5 , 5 , 7 , 1 0 \}$ . For the mean flow, we optimized the EDM schedule $( \rho , \sigma _ { \mathrm { m i n } } )$ with 7 Optuna trials over $\rho \in [ 1 , 1 0 ]$ and $\sigma _ { \mathrm { m i n } } \in [ 1 0 ^ { - 6 } , 1 0 ^ { - 1 } ]$ (log-uniform), the same number of candidates as for the consistency model. Table S3 lists the selected schedules. For S=1, the schedule is fixed. For all other experiments, we set $\rho { = } 7$ and $\sigma _ { \mathrm { m i n } } { = } 0 . 0 0 1$

Table S3: Sampling schedules selected on two moons. Discretization exponent $\rho$ of the consistency model and EDM schedule $( \rho , \sigma _ { \mathrm { m i n } } )$ of the conditional mean flow for each step count S, tuned on held-out observations; at S=1, the schedule has no free parameter. To show how much the selection matters, we report the tuning C2ST (3 held-out observations, 2,000 draws each) of the selected and of the worst of the 7 candidates.
<table><tr><td></td><td></td><td>S=2</td><td>S=4</td><td>S=8</td><td>S=16</td><td>S=32</td></tr><tr><td>Consistency model</td><td>ρ</td><td>1</td><td>2</td><td>3</td><td>3.5</td><td>3.5</td></tr><tr><td></td><td>C2ST, selected C2ST, worst</td><td>0.918 0.973</td><td>0.542 0.975</td><td>0.555 0.946</td><td>0.539 0.700</td><td>0.535 0.666</td></tr><tr><td>Mean flow</td><td></td><td>4.54</td><td>3.82</td><td>4.12</td><td>8.34</td><td>7.42</td></tr><tr><td></td><td>ρ</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\sigma _ { \mathrm { m i n } }$ </td><td>1.1×10-4</td><td> $4 . 9 \times 1 0 ^ { - 2 }$ </td><td> $5 . 7 \times 1 0 ^ { - 5 }$ </td><td> $7 . 0 \times 1 0 ^ { - 2 }$ </td><td> $2 . 0 { \times } 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>C2ST, selected</td><td>0.502</td><td>0.504</td><td>0.509</td><td>0.505</td><td>0.507</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>C2ST, worst</td><td>0.513</td><td>0.519</td><td>0.519</td><td>0.517</td><td>0.521</td></tr></table>

Code. The code and the configurations of all experiments are available under the MIT license at https: //anonymous.4open.science/r/MeanFlow-AISTAST. The code is written in Python 3.14, and all networks are Keras 3.15.1 models (Chollet et al., 2015) on the JAX 0.11.1 backend (Bradbury et al., 2018), both under the Apache 2.0 license. Besides BayesFlow (MIT), we build on PyMC 5.28.5 (Apache 2.0) (Salvatier et al., 2016) and nutpie 0.16.11 (MIT) (Seyboldt et al., 2026) for MCMC, ArviZ 0.23.4 (Apache 2.0) (Martin et al., 2026) for the convergence diagnostics, and Optuna 4.9.0 (MIT) (Akiba et al., 2019) for the hyperparameter searches.

Compute environment. We ran all analyses on a computing cluster using eight CPU cores and one NVIDIA A100 GPU with 40 GB of VRAM. The computing cluster uses an AMD EPYC 7F72 CPU with a core clock speed of up to 3.2 GHz and 50 GB of RAM.

## D MCMC SAMPLERS

NUTS. We sample with the NUTS implementation of nutpie (Seyboldt et al., 2026) through PyMC (Salvatier et al., 2016) on the JAX backend (Bradbury et al., 2018), with 4 chains, a diagonal mass matrix, a maximum tree depth of 10 and the default target acceptance rate of nutpie. We use the default of 400 warmup iterations, except for g-and-k, where the warmup is 25% of the requested draws of each method.

Budget on g-and-k. A method whose density costs K network evaluations requests 8,000/K draws per chain, so that the product of requested draws and network evaluations per density is constant across methods. The normalizing flow counts as $K { = } 1$ . For delayed acceptance MCMC, the requested iterations are $9 6 , 0 0 0 / ( J S _ { c } + 1 0 0 )$ per chain, which accounts for the $J { = } 5$ coarse evaluations and one fine evaluation per iteration. The budget is set on the requested draws and ignores the number of leapfrog steps per draw, which difers between methods, so the realized number of network evaluations varies. We count one network evaluation per forward pass of the network for one pair $( \mathbf { z } , \mathbf { x } ) { \mathrm { : } }$ : a normalizing flow costs one evaluation per density, a mean flow with S steps costs $S ,$ a velocity field integrated with S Euler steps costs S, and a fixed-step Tsit5 integration costs 7 per step. Jacobians, divergences and gradients are obtained by automatic diferentiation on top of these passes and are not counted; their cost is reflected in the wall times.

Delayed acceptance MCMC. We implement two-level delayed acceptance MCMC (Christen and Fox, 2005) with the subchains of Lykkegaard et al. (2023). For any coarse posterior $\pi _ { \mathrm { c } } ,$ , its invariant distribution is the fine posterior π<sub>f</sub> (Section 3.2). Both levels use the same trained network: the coarse level evaluates the mean flow with $S _ { c } \in \{ 1 , 2 , 4 , 8 \}$ steps, and the fine level integrates its velocity head with 100 Tsit5 steps of fixed size and the exact divergence in (1). Each iteration runs a subchain of $J { = } 5$ random-walk Metropolis steps on $\pi _ { \mathrm { c } } ,$ following Lykkegaard et al. (2023), and proposes its end point to the fine level with the acceptance probability (11). The random walk draws Gaussian increments with a step size per parameter, initialized at 10% of the prior width, and a global scale. A proposal outside the prior’s support is rejected without evaluating the network. If the subchain does not move, the proposal equals the current state and is accepted without evaluating the fine likelihood. Adaptation takes 25% of the requested iterations and is discarded. During adaptation, a Robbins-Monro update steers the global scale towards a coarse acceptance rate of 0.3, and in its second half, the step sizes are updated to the marginal standard deviations of the states visited in that half, after which the proposal is fixed. We run 4 chains started from prior draws.

## E PER-EXPERIMENT DETAILS

## E.1 Two Moons

The prior is $\pmb \theta \sim \mathcal { U } ( - 1 , 1 ) ^ { 2 }$ , and the forward model is

$$
\begin{array} { l l } { { a \sim \mathcal { U } ( - { \pi } / 2 , { \pi } / 2 ) , } } & { { r \sim \mathcal { N } ( 0 . 1 , 0 . 0 1 ^ { 2 } ) , } } \\ { { \ } } & { { { \bf y } = \big ( r \cos a + 0 . 2 5 , r \sin a \big ) + \big ( - | \theta _ { 1 } + \theta _ { 2 } | / \sqrt { 2 } , \ : ( - \theta _ { 1 } + \theta _ { 2 } ) / \sqrt { 2 } \big ) , } } \end{array}
$$

whose fold makes the posterior invariant under $\left( \theta _ { 1 } , \theta _ { 2 } \right) \mapsto \left( - \theta _ { 2 } , - \theta _ { 1 } \right)$ , which gives two crescent-shaped modes (Greenberg et al., 2019). Since the density is available in closed form and the forward model can be inverted, we draw 32,000 exact i.i.d. reference samples per observation. The first dataset uses the canonical observation $\mathbf { y } = ( 0 , 0 )$ , and the other nine use prior predictive draws.

## E.2 g-and-k

The g-and-k distribution is defined by its quantile function in normal-score space (Rayner and MacGillivray, 2002),

$$
W ( z ; g , k ) = \left( 1 + 0 . 8 \operatorname { t a n h } ( g z / 2 ) \right) ( 1 + z ^ { 2 } ) ^ { k } z , \qquad z \sim \mathcal { N } ( 0 , 1 ) ,
$$

and has no closed-form density. Since W is strictly increasing for $k \geq 0$ , we invert it by bisection, which gives the exact log-density by the change of variables. We couple two coordinates with a Gaussian copula of correlation $\rho$ in normal-score space, such that both marginals are g-and-k distributed. The priors are $g \sim \mathcal { U } ( - 1 , 1 ) , k \sim \mathcal { U } ( 0 , 0 . 5 )$ and $\rho \sim \mathcal { U } ( - 0 . 9 , 0 . 9 )$ , and each dataset consists of $N { = } 5 0$ observations simulated from a parameter drawn from the prior. We obtain 32,000 reference draws per dataset from an adaptive random-walk Metropolis ensemble on the exact log-posterior, started around the generating parameter with a jitter of 2% of the prior width (8 chains, 5,000 adaptation iterations, 40,000 draws thinned by 10, $\hat { R } \leq 1 . 0 5$ and $\mathrm { E S S } \geq 4 0 0 )$

## E.3 Epidemic Renewal

On a grid of $T = 1 2 8$ steps, of which the first $n _ { 0 } = 5$ are seeded at incidence $I _ { 1 } = \cdot \cdot \cdot = I _ { n _ { 0 } } = 1 0$ and not observed, the renewal model is, for $t > n _ { 0 }$

$$
\begin{array} { r } { \Lambda _ { t } = \sum _ { s = 1 } ^ { 1 5 } w _ { s } I _ { t - s } , \qquad I _ { t } = R _ { t } \Lambda _ { t } \exp \bigl ( \sigma _ { p } \varepsilon _ { t } - \frac { 1 } { 2 } \sigma _ { p } ^ { 2 } \bigr ) , \qquad y _ { t } = \log I _ { t } + \sigma _ { o } \eta _ { t } , } \end{array}
$$

with $\varepsilon _ { t } , \eta _ { t } \sim \mathcal { N } ( 0 , 1 ) , \sigma _ { p } = \sigma _ { o } = 0 . 0 5$ , and w the Gamma density with mean 5.0 and standard deviation 2.5 (Fraser, 2007; Cori et al., 2013). This is the renewal equation of Cori et al. (2013) with a discretized generation interval truncated at 15 steps. The observation is $( y _ { n _ { 0 } + 1 } , \dots , y _ { T } )$ , so $d = 1 2 3$ . The parameter is the whitened coeficient vector $\mathbf { z } \in \mathbb { R } ^ { 1 2 8 }$ with log $R = L \mathbf { z }$ , where $L L ^ { \top }$ is a squared-exponential kernel with lengthscale 2.5, marginal standard deviation 0.4 and a $1 0 ^ { - 6 }$ ridge (Rasmussen and Williams, 2005). Hence, the prior is exactly $\mathbf { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Since L is lower triangular, $z _ { s }$ enters log $R _ { t }$ only for $t \geq s ,$ and since the recursion runs forward in time, y does not depend on $z _ { s }$ for $s > t .$ . The causal attention mask of the masked transformer (Appendix C) encodes exactly this structure. Observations are standardized separately at each time step.

We withhold each of the observed steps independently with probability 0.3 and draw a new pattern for every dataset. Since this mechanism is independent of the process and the parameters, the data are missing completely at random, and the posterior only requires the marginal likelihood $p ( \mathbf { y } _ { \mathcal { O } } \mid \mathbf { z } )$ of the observed entries O (Little and Rubin, 2019). The masked transformer trains on complete simulations with masks drawn independently per simulation and observation at the same rate and thereby learns this marginal for every pattern. The normalizing flow instead models vectors with the sentinel −1 in the missing entries according to Wang et al. (2024).

## E.4 Cognitive Model

A trial is a Wiener process with drift ν that starts at $\beta a$ (with fixed relative starting point $\beta = 0 . 5 )$ and terminates at the first hit of 0 or a. The response time adds a non-decision time $t _ { 0 } .$ , and trials time out after 10 s. The priors are $\nu \sim \mathcal { U } ( - 3 , 3 ) , a \sim \mathcal { U } ( 0 . 8 , 2 )$ and $t _ { 0 } \sim \mathcal { U } ( 0 . 1 , 0 . 4 )$ , and we simulate with Euler-Maruyama at $\mathrm { d } t = 1 0 ^ { - 4 }$ The network observes the signed response time $y = ( 2 c - 1 ) t .$ , with $c = 1$ for the upper boundary, which is bijective with the pair of response time and choice. Since the non-decision time cannot exceed the fastest observed response time, we sample it below this bound, which tightens as trials are added. Within each dataset, trial counts are nested, and every setting keeps 4 × 1,000 NUTS draws after the default warmup. For the first trial count $N { = } 4 ,$ , the chains start at the center of the prior support with the default jitter of PyMC. The chains for the next trial count start at the posterior median of the preceding trial count, clipped to the current support of $t _ { 0 } .$ with independent jitter per chain.

## F ADDITIONAL FIGURES AND TABLES

Table S4: Density metrics on two moons. Total mass, KL divergence and Hellinger distance of each learned posterior density against the exact one, on a 512×512 grid over the prior; mean (standard deviation) over 10 independent datasets. The best value per column is in bold, for the mass the one closest to 1.
<table><tr><td>Method</td><td>Steps</td><td>Mass</td><td>KL</td><td>Hellinger</td></tr><tr><td>Normalizing flow</td><td>S=1</td><td>0.977 (0.055)</td><td>0.446 (0.073)</td><td>0.383 (0.023)</td></tr><tr><td>Mean flow</td><td>S=1</td><td>1.066 (0.017)</td><td>0.347 (0.137)</td><td>0.322 (0.052)</td></tr><tr><td>Mean flow</td><td>S=2</td><td>1.045 (0.030)</td><td>0.179 (0.077)</td><td>0.230 (0.042)</td></tr><tr><td>Mean flow</td><td>S=8</td><td>1.085 (0.093)</td><td>0.165 (0.053)</td><td>0.219 (0.050)</td></tr><tr><td>Mean flow</td><td>S=32</td><td>1.003 (0.025)</td><td>0.160 (0.052)</td><td>0.169 (0.027)</td></tr><tr><td>Mean flow (v-head)</td><td>Euler, S=1</td><td> $7 . 2 { \times } 1 0 ^ { 3 9 } \ ( 2 . 3 { \times } 1 0 ^ { 4 0 } )$ </td><td>73.2 (19.0)</td><td>1.000 (0.000)</td></tr><tr><td>Mean flow (v-head)</td><td>Euler, S=8</td><td>57.3 (45.5)</td><td>4.10 (1.20)</td><td>0.902 (0.076)</td></tr><tr><td>Mean flow (v-head)</td><td>Euler, S=32</td><td>1.783 (0.077)</td><td>0.385 (0.033)</td><td>0.363 (0.020)</td></tr><tr><td>Mean flow (v-head)</td><td>adaptive, S=311</td><td>0.993 (0.016)</td><td>0.031 (0.013)</td><td>0.087 (0.017)</td></tr><tr><td>Flow matching</td><td>Euler, S=1</td><td> $5 . 5 { \times } 1 0 ^ { 2 3 } \ ( 1 . 7 { \times } 1 0 ^ { 2 4 } )$ </td><td>37.8 (9.5)</td><td></td></tr><tr><td>Flow matching</td><td>Euler, S=32</td><td>1.825 (0.073)</td><td>0.271 (0.021)</td><td>1.000 (0.001)</td></tr><tr><td>Flow matching</td><td>adaptive, S=302</td><td>0.991 (0.023)</td><td>0.048 (0.019)</td><td>0.304 (0.017) 0.117 (0.016)</td></tr></table>

![](images/9604a0f1d70a7a940b0e34274628156d058780bb38a6f1afa449eeb8cad38b15.jpg)

![](images/eb71349125da5e4790ac0378ca8ffd6ce8767aba072d1abeac936af63ef18315.jpg)  
Figure S1: Sample and density quality against step count on two moons. (A) C2ST against the reference posterior over mean flow steps S (0.5 indicates indistinguishable samples). (B) Hellinger distance of the learned density from the exact posterior on a 512×512 grid. Lines show the mean over 10 independent datasets, bands the standard deviation.

Table S5: Composing the DDM trial likelihood, per method and trial count. NRMSE to the generating parameter, coverage of the 95% central interval, minimum bulk ESS, divergent transitions, and worst predictive discrepancy; mean (standard deviation) over 10 independent datasets.
<table><tr><td>Method</td><td>N</td><td>NRMSE</td><td>Cov. 95%</td><td>Min ESS</td><td> $\mathbf { D i v . }$ </td><td></td><td>maxk | |zk|</td></tr><tr><td>Mean flow S=1</td><td>4</td><td>0.268 (0.079)</td><td>0.90 (0.16)</td><td></td><td>1,396 (312)</td><td>1.0 (1.9)</td><td>0.73 (0.14)</td></tr><tr><td>Mean flow S=1</td><td>16</td><td>0.161 (0.033)</td><td>0.90 (0.16)</td><td>1,174</td><td>(378)</td><td>6.6 (11.5)</td><td>0.90 (0.30)</td></tr><tr><td>Mean flow S=1</td><td>64</td><td>0.102 (0.028)</td><td>0.90 (0.16)</td><td>1,205</td><td>(186)</td><td>2.4 (4.0)</td><td>1.13 (0.47)</td></tr><tr><td>Mean flow S=1</td><td>256</td><td>0.070 (0.019)</td><td>0.60 (0.31)</td><td>1,146</td><td>(335)</td><td>7.8 (21.7)</td><td>2.15 (1.03)</td></tr><tr><td>Mean flow S=2</td><td>256</td><td>0.061 (0.016)</td><td>0.73 (0.21)</td><td>1,094</td><td>(335)</td><td>0.6 (1.6)</td><td>1.93 (0.77)</td></tr><tr><td>Mean flow S=4</td><td>256</td><td>0.066 (0.018)</td><td>0.77</td><td>(0.27) 503</td><td>(620)</td><td>50.7 (50.1)</td><td>2.02 (0.77)</td></tr><tr><td>Mean flow S=8</td><td>256</td><td>0.064 (0.023)</td><td>0.70</td><td>(0.29) 513</td><td>(504)</td><td>31.8 (29.0)</td><td>1.99 (0.83)</td></tr><tr><td>Flow matching, Euler S=200</td><td>4</td><td>0.267 (0.081)</td><td></td><td>0.90 (0.16)</td><td>1,279 (228)</td><td>0.3 (0.7)</td><td>0.72 (0.13)</td></tr><tr><td>Flow matching, Euler S=200</td><td>16</td><td>0.151 (0.032)</td><td>0.90</td><td>(0.16)</td><td>1,285 (270)</td><td>0.1 (0.3)</td><td>0.88 (0.22)</td></tr><tr><td>Flow matching, Euler S=200</td><td>64</td><td>0.099 (0.031)</td><td>0.87</td><td>(0.23)</td><td>1,127 (228)</td><td>0.0 (0.0)</td><td>1.03 (0.57)</td></tr><tr><td>Flow matching, Euler S=200</td><td>256</td><td>0.067 (0.033)</td><td></td><td>0.70 (0.33)</td><td>1,275 (329)</td><td>0.1 (0.3)</td><td>1.79 (0.57)</td></tr></table>

Table S6: Budget-matched MCMC on g-and-k. Every method except the flow matching baseline shares the same budget of requested draws times network evaluations per density. The realized totals in forward passes are 0.9 to $2 . 0 \times 1 0 ^ { 7 }$ for the NUTS arms and 1.0 to $1 . 4 \times 1 0 ^ { 8 }$ for delayed acceptance MCMC. Mean (standard deviation) over 10 independent datasets; the best minimum ESS and C2ST are in bold.
<table><tr><td>Method</td><td>Steps</td><td>Samples</td><td>Min ESS</td><td>C2ST</td></tr><tr><td>Mean flow S=1</td><td>1</td><td>32,000</td><td>10,498 (1,552)</td><td>0.604 (0.049)</td></tr><tr><td>Mean flow S=2</td><td>2</td><td>16,000</td><td>4,856 (939)</td><td>0.587 (0.046)</td></tr><tr><td>Mean flow S=4</td><td>4</td><td>8,000</td><td>2,436 (708)</td><td>0.577 (0.039)</td></tr><tr><td>Mean flow S=8</td><td>8</td><td>4,000</td><td>1,143 (476)</td><td>0.599 (0.030)</td></tr><tr><td>Normalizing flow</td><td>1</td><td>32,000</td><td>8,023 (3,169)</td><td>0.679 (0.078)</td></tr><tr><td>Flow matching, Euler S=200</td><td>200</td><td>4,000</td><td>1,270 (331)</td><td>0.614 (0.048)</td></tr><tr><td>Delayed acceptance MCMC,  $S _ { c } { = } 1$ </td><td>100</td><td>3,656</td><td>642 (246)</td><td>0.592 (0.031)</td></tr><tr><td>Delayed acceptance MCMC,  $S _ { c } { = } 2$ </td><td>100</td><td>3,492</td><td>649 (200)</td><td>0.594 (0.030)</td></tr><tr><td>Delayed acceptance MCMC,  $S _ { c } { = } 4$ </td><td>100</td><td>3,200</td><td>581 (133)</td><td>0.595 (0.033)</td></tr><tr><td>Delayed acceptance MCMC,  $S _ { c } { = } 8$ </td><td>100</td><td>2,744</td><td>544 (157)</td><td>0.599 (0.033)</td></tr></table>

![](images/7605898cd001034f90e35dbb887959a11dfccdb1fc0dbc21415ee4aaf10ca55c.jpg)

![](images/838b68eb4831a5f252914f6ff04ab21ead91a176031e786ef96bb0ea533824fc.jpg)

![](images/3ae485afbfa03e47cbd953896161001b57d4071db21298be2f62a1358c6b07d4.jpg)  
Wall time (s)

![](images/7c7affa1733b3e79040412f756de56192b6f66095b71e13c0e059fc878a977d5.jpg)  
Figure S2: Wall time of every method across experiments. Warm wall times by method (log scale), mean over 10 datasets, for (A) two-moons sampling (5,000 draws), (B) g-and-k MCMC under the shared budget, with the flow matching baseline outside it, (C) renewal MCMC at equal draws and (D) DDM MCMC per trial count.

![](images/7f381dc6fbd92748ca66df734b097cebb46eae4072d6358eaac15204b1192ea3.jpg)  
B

![](images/3319dcc5706d4260a900e7a72635858249a32c5d4c0b813c9a466ddf048038fa.jpg)

C  
![](images/f0a4e3e400a9bbb14c44ca5acc1f7c15f01b2541f77db35c0312aae04b553a7b.jpg)

![](images/5a3c08a4ffd58fcf8d477fd6c8c1903a6a15c106fcc4a75d7e921a83921c7a7e.jpg)

![](images/f62ae990de80a3a634455a68b82187c4b85b01bafbc53ab3f75eb61fcedd74c6.jpg)

![](images/4d4999e259824af4e8d944caf39b933cf8b085a05f40dbc1ebbd3c8e3626991a.jpg)

![](images/833725474227e54b76c4165da31a58c7ac163cd9aa0dec38e43c824758c8875b.jpg)

![](images/3bf2f9261877c33df723c46f24671863a3a5829922efd7adfdc13d7ea18ab2c0.jpg)

![](images/d8989a67e4c3a5d7eace31442a2c7aafea9cf4c8d37ce8e4ac3c512a22571923.jpg)

![](images/35bbde6a61f21f268aeb1792f91dbb076a0fe066c4ddeb85bdd923b6b42072c8.jpg)

![](images/889cbe68721c652c95073877767205256bb41fd9545a06390e4e008e3936b334.jpg)

![](images/57f82e5d18fcaaa40ac3c24316d1c199523f91601816dd1f1bc9731fb228b4e4.jpg)

![](images/4fd8a1df50612d6eab7325ee5bb11f451b0a683fbb15771bc1249015ba9eda16.jpg)

![](images/acc6b6f5767325bc8d72dcd1e898cc7d0f5795c4cfa97e3cfba84782e2b11961.jpg)

![](images/68fc6ec06977824a640c608943cc05218d55da0060edf7919862bbdf1c57f084.jpg)

![](images/f69dbe047067be28b19c17704d3d62a37c2d3d321d48b5cc034ebe0db6ffb928.jpg)  
Figure S3: Composed DDM posteriors on one dataset, over the trial count. Rows are $N \in \{ 4 , 1 6 , 6 4 , 2 5 6 \}$ columns are the marginals of (A) the drift ν, (B) the boundary separation a and (C) the non-decision time $t _ { 0 }$ for $S { = } 1 , S { = } 8$ and the 200-step baseline, with the generating parameter dashed and the sampled $t _ { 0 }$ bound dotted. (D) Posterior predictive CDF of the signed response time (bands: mean ± one standard deviation over 200 replicates) against the observed trials. Shown for one of 10 datasets.

![](images/29f631cbc9f48557b861efd0fed511e938ffd3b306f3ad5c89d2e370f9e43d73.jpg)

![](images/6915b6cd56a5bc070538804ab816bdbd80c607c51a90720e46e6ce8f4a88dc4b.jpg)

![](images/aa60b2dde57cdb6939d281cf1884cdf077a54a329e9259994ddd9e8c17db87aa.jpg)  
Figure S4: Efect of step count on DDM inference. A NRMSE over mean flow steps S, one line per trial count N, with the flow matching baseline with 200 Euler steps dashed. B Step count selected within each dataset by NRMSE among converged settings, over the trial count N. C Minimum bulk ESS over S, with the baseline dash-dotted. Lines show the mean over 10 independent datasets, bands the standard deviation.

![](images/6e75f23bd71b36ba92e1bbbc0433b0aa314ccd4a2b6ab004e2bad6ad5e0169c9.jpg)

![](images/9f26dc57aedb127eb0258b44288ba642a66e74fe88b599e6b65cc6ffb919bd30.jpg)

![](images/1a20bd2e313227dffdac939fbfef01550aa272eebacbb7ee25c7a418d6766024.jpg)  
Figure S5: Renewal-model observations and what each method is shown of them. (A) The observation: four log-incidence curves at one parameter, with the latent log $I _ { t }$ of the first behind them and the seeded steps shaded, giving 123 observed values on a 128-step grid. (B) The reproduction number that generated them, against the R=1 line and the prior’s 90% interval, for comparison with the generating trajectory. (C) The same curve under one 30% missingness pattern, as each method receives it: the observed coordinates alone for the masked transformer, against the vector with sentinel values for the normalizing flow.