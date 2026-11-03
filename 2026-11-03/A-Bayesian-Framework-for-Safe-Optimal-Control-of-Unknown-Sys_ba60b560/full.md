# A Bayesian Framework for Safe Optimal Control of Unknown Systems from Input–Output Data <sup>⋆</sup>

Robert Lefringhausen, Annalena Daniels <sup>1</sup>, Michael Fink <sup>1</sup>, Sandra Hirche

School of Computation, Information and Technology, Technical University of Munich, Germany

## Abstract

As control engineering is applied to increasingly complex systems, learning-based control, which uses empirical data to reduce the reliance on first-principles models, has emerged as a promising paradigm. However, such approaches must cope with nonlinear and stochastic dynamics, the absence of state measurements, and scarce data, while ensuring safe operation under constraints, which requires rigorous uncertainty quantification. Whereas existing methods treat these challenges in isolation, this article proposes a Bayesian framework that jointly addresses them. A prior in state-space form is updated from input–output data using particle Markov chain Monte Carlo methods, yielding a representation of joint uncertainty over dynamics and latent states. Building on this representation, a validation procedure is developed that provides probabilistic guarantees for constraint satisfaction under arbitrary causal control laws. For control synthesis, a formulation penalizing the conditional value-at-risk of the constraint function is proposed, together with an adaptive calibration mechanism that adjusts the performance–robustness trade-off to achieve a desired robustness level. The probability of successful certification is characterized theoretically. The framework is extended to a receding-horizon setting, where posterior updates from new measurements allow the controller to progressively reduce uncertainty and improve performance. The approach is demonstrated on the optimal control of a vertical farm.

Key words: Data-based control, Safe learning-based control, Bayesian system identification, Uncertainty quantification, Optimization under uncertainties, Randomized methods

## 1 Introduction

Learning-based optimal control aims to enable the highperformance operation of systems with unknown dynamics by learning models from empirical data. This paradigm is particularly relevant in applications where pure firstprinciples modeling is infeasible or prohibitively expensive, such as soft robotic actuators, systems involving human interaction, and biological processes. In real-world applications, several challenges arise jointly that fundamentally shape the learning-based control problem:

(1) Nonlinear and stochastic dynamics: Real-world systems are inherently nonlinear and affected by both process noise, which alters the system trajectory, and measurement noise, which only corrupts observations.

(2) Absence of state measurements: In many applications, the system state is latent, i.e., it cannot be measured directly due to limited sensing capabilities. Consequently, methods must operate solely on input–output data.

(3) Scarce data: In engineering applications, data are often scarce due to costly and time-consuming experiments, making the incorporation of structural knowledge from first principles essential for data efficiency and generalization.

(4) Rigorous safety guarantees: In safety-critical applications, constraint satisfaction must be ensured with high confidence despite the challenges posed by (1)–(3). This necessitates the rigorous quantification of both the irreducible aleatoric uncertainty introduced by process and measurement noise and the reducible epistemic uncertainty about the dynamics and latent states that arises from limited data.

Despite the practical importance of these challenges, existing approaches address individual aspects only, while their combined treatment remains largely unresolved. In the following, we briefly review existing approaches and their limitations with respect to these challenges.

Data-driven control methods rooted in behavioral systems theory [4] formulate optimal control problems directly from input–output data without requiring measurements of the underlying system state. These approaches have primarily been developed for linear time-invariant systems, and although extensions to stochastic and nonlinear settings have been proposed, their guarantees rely on specific system classes or local approximations.

Nonlinear input–output models, such as nonlinear autoregressive exogenous (NARX) models, learn a mapping from past inputs and outputs to future system behavior. Bayesian formulations, for instance Gaussian process (GP) NARX models [21], systematically quantify both epistemic and aleatoric uncertainty through a posterior distribution over this mapping, potentially enabling control with probabilistic safety guarantees. However, input–output models lack an internal state and therefore cannot distinguish process noise from measurement noise, treating all stochastic variability as a single lumped disturbance. This can lead to biased and increasingly inaccurate multi-step predictions over the horizons relevant for optimal control [28]. Moreover, prior knowledge from first principles, such as conservation laws or kinematic relations, is naturally expressed in terms of state variables and cannot be straightforwardly encoded in input–output mappings.

State-space models are a compelling foundation for learningbased optimal control, as they can describe nonlinear dynamics, inherently separate process and measurement noise, and allow the direct integration of prior knowledge from first principles. In their Bayesian formulation, for instance as GP state-space models, they further provide a principled framework for quantifying both epistemic and aleatoric uncertainty, which several works have exploited for robust learning-based control [25]. These approaches, however, rely on full state measurements, which are unavailable in many applications. Without them, learning the dynamics no longer reduces to fitting a function to observed state transitions, and predictions over a planning horizon depend on a current state that is itself unknown. The unknown dynamics and the latent states must therefore be estimated jointly. While several works provide point estimates of both [27,26], quantifying the remaining uncertainty is substantially more challenging because the uncertainties of the dynamics and the latent states are coupled and their joint posterior is generally intractable. To the best of our knowledge, no existing control method for nonlinear systems accounts for this joint uncertainty and translates it into certifiable probabilistic guarantees, except for our preliminary work [17].

This article closes this gap by proposing a framework for the optimal control of nonlinear systems with unknown dynamics from input–output data that addresses all four challenges. To retain the advantages of Bayesian state-space models without requiring full state measurements, we use particle Markov chain Monte Carlo (PMCMC) methods [2] to sample from the intractable joint posterior over dynamics and latent states. Together with the noise model, these samples provide a representation of both epistemic and aleatoric uncertainty. While PMCMC-based inference for nonlinear state-space models has been studied in the context of system identification [13,30,34], we exploit this representation for control, with the following main contributions:

(i) We develop a validation procedure that provides nonasymptotic probabilistic guarantees for constraint satisfaction under arbitrary causal control laws with respect to the posterior predictive distribution over dynamics, latent states, and future disturbances.

(ii) We propose a synthesis–validation procedure that computes a certified input sequence by combining the validation procedure in (i) with risk-sensitive optimal control using a conditional value-at-risk (CVaR) penalty. The weight of this penalty trades off performance and robustness, and an adaptive calibration mechanism adjusts it until the desired robustness level is certified. We further analyze its theoretical properties, linking the risk weight to the risk of constraint violation and quantifying the probability of successful certification.

(iii) We extend the synthesis–validation procedure in (ii) to a receding-horizon setting in which the posterior is updated between control cycles as new measurements become available, enabling the controller to progressively reduce uncertainty and improve performance. By exploiting the gradual evolution of the posterior, each cycle can be performed more efficiently than repeating inference and synthesis from scratch, while preserving the per-cycle probabilistic guarantees.

This article extends our preliminary work [17], which proposes PMCMC-based open-loop optimal control with robustness guarantees from non-convex scenario theory [7]. The present work replaces this conservative and computationally demanding formulation with a synthesis–validation framework in which the robustness level is an explicit design choice, extends it to a receding-horizon setting, and demonstrates it in a comprehensive case study.

Outline After the problem formulation in Section 2 and a brief review of PMCMC methods in Section 3, Section 4 develops the validation procedure and its probabilistic guarantees. Section 5 presents the CVaR-penalized control synthesis with adaptive calibration and its analysis, and Section 6 extends the framework to a receding-horizon setting. Section 7 demonstrates the approach on the optimal control of a vertical farm, and Section 8 concludes the article.

Notation Lowercase/uppercase bold symbols denote vectors/matrices, respectively. R denotes the set of real numbers, and N<sup>0</sup> and N denote the sets of natural numbers with and without zero, respectively. For indices $b \leq c , \mathbf { a } _ { b : c }$ is the shorthand notation for $( \pmb { a } _ { b } , \ldots , \pmb { a } _ { c } )$ . P(·) denotes the probability of an event, E[·] the expected value, and $p ( \cdot )$ the probability density function.

## 2 Problem Formulation

Consider a nonlinear discrete-time system

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } = \pmb { f } _ { \pmb { \theta } } \left( \pmb { x } _ { t } , \pmb { u } _ { t } , \pmb { v } _ { t } \right) } \\ { \pmb { y } _ { t } = \pmb { g } _ { \pmb { \theta } } \left( \pmb { x } _ { t } , \pmb { u } _ { t } \right) + \pmb { w } _ { t } , } \end{array}\tag{1a}
$$

(1b)

where $\pmb { x } _ { t } \in \mathbb { R } ^ { n _ { x } }$ denotes the state, $\mathbf { \boldsymbol { u } } _ { t } \in \mathbb { R } ^ { n _ { u } }$ the input, and $\textstyle { y _ { t } } \in \mathbb { R } ^ { n _ { y } }$ the output at time $t \in \mathbb { N } ^ { 0 }$ . The state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ is not directly observed, and only the output y<sub>t</sub> is available for feedback and learning. The vector $\bar { \pmb { \theta } _ { } } \in \mathbb { R } ^ { n _ { \theta } }$ collects all unknown model parameters. The state-transition function $f _ { \theta } ( \cdot )$ and the observation function $g _ { \theta } ( \cdot )$ are assumed to be measurable and to belong to a known parametric model class characterized by θ. The process noise ${ \mathbf { } } v _ { t }$ and measurement noise ${ \pmb w } _ { t }$ follow the distributions $\nu _ { \theta }$ and $\mathcal { W } _ { \theta }$ , respectively, whose characteristics (e.g., mean and covariance) may depend on $\pmb \theta$ and whose probability density functions are denoted by $p _ { \mathcal { V } } ( \pmb { v } _ { t } \ | \ \pmb { \theta } )$ and $p _ { \mathcal { W } } ( \pmb { w } _ { t } \ | \ \pmb { \theta } )$ . Conditional on $\theta ,$ they are independent and identically distributed (i.i.d.) over time, mutually independent, and independent of the initial state $\scriptstyle { \pmb x } _ { 0 }$ . The assumption of a parametric model class is motivated by the fact that, in many physical systems, prior structural knowledge derived from first principles enables the formulation of expressive yet interpretable models. The same source of knowledge can be leveraged to specify prior distributions over the unknown parameters and the initial state, which motivates the following assumption.

Assumption 1 (Prior knowledge) Prior distributions $p ( \pmb \theta )$ and $p ( \pmb { x } _ { 0 } \ \mid \ \pmb { \theta } )$ for the model parameters and the initial state are available. The unknown system parameters and the initial state are realizations drawn from these priors.

In addition to the prior, a dataset $\mathbb { D } = \left( \boldsymbol { \mathsf { u } } _ { 0 : t _ { 0 } - 1 } , \boldsymbol { \mathsf { y } } _ { 0 : t _ { 0 } - 1 } \right)$ of past input–output measurements collected up to time $t _ { 0 } - 1$ where $t _ { 0 } ~ \in ~ \mathbb { N }$ , is available. Given a prediction horizon $H \in \mathbb { N }$ and a cost functional

$$
J \left( { { \pmb u } _ { t _ { 0 } : t _ { 0 } + H } } , { \pmb x } _ { t _ { 0 } : t _ { 0 } + H } , { \pmb y } _ { t _ { 0 } : t _ { 0 } + H } \right) ,\tag{2}
$$

the objective is to determine an input sequence $\pmb { u } _ { t _ { 0 } : t _ { 0 } + H }$ that minimizes the cost subject to two types of constraints. Input constraints, such as actuator limits, restrict the inputs to a set $\mathbb { U } \subset \mathbb { R } ^ { n _ { \tau } }$ <sup>u</sup> and can be enforced directly. Constraints on states and outputs of the form

$$
h \left( { \pmb u } _ { t _ { 0 } : t _ { 0 } + H } , { \pmb x } _ { t _ { 0 } : t _ { 0 } + H } , { \pmb y } _ { t _ { 0 } : t _ { 0 } + H } \right) \le 0\tag{3}
$$

depend on the uncertain dynamics and cannot be guaranteed deterministically: since both parameters and disturbances may have unbounded support, this would require robustness with respect to all possible realizations of the uncertainty, which is typically infeasible or overly conservative. Instead, the objective is to ensure their satisfaction with high probability, i.e.,

$$
\mathbb { P } \left[ h \left( \pmb { u } _ { t _ { 0 } : t _ { 0 } + H } , \pmb { x } _ { t _ { 0 } : t _ { 0 } + H } , \pmb { y } _ { t _ { 0 } : t _ { 0 } + H } \right) \leq 0 \right] \geq 1 - \alpha ,\tag{4}
$$

where $\alpha ~ \in ~ ( 0 , 1 )$ denotes the allowable risk level and the probability is taken with respect to the joint uncertainty over parameters, latent states, and future disturbances that remains given the available data D. Here, $h ( \cdot )$ is assumed measurable and, without loss of generality, scalarvalued, since multiple constraints can be aggregated via $h = \operatorname* { m a x } ( h _ { 1 } , \ldots , h _ { m } )$

The formulation (1)–(4) captures all four challenges identified in Section 1 within a single problem statement, whose joint treatment is the focus of this article.

## 3 Particle Markov Chain Monte Carlo Methods

In order to solve the optimal control problem (OCP) (1)–(4), it is essential to reduce uncertainty about the system by incorporating the information contained in the observed data D. If decisions were based solely on the prior distribution, the uncertainty over future state trajectories obtained by propagating the prior over $( \pmb \theta , \pmb x _ { 0 } )$ through the stochastic dynamics would typically be excessively large, leading either to infeasibility of the probabilistic constraints or to overly conservative solutions. Within the Bayesian framework, this reduction is achieved by inferring the posterior distribution $p ( \pmb \theta , \pmb x _ { 0 : t _ { 0 } - 1 } \ | \ \mathbb D )$ , which combines prior knowledge with the information provided by the measurements. For nonlinear state-space models with latent states, this posterior is analytically intractable, since its normalizing constant is an integral over the parameters and all latent state trajectories that generally admits no closed-form solution.

Approximate inference methods address this intractability in different ways. Variational approaches (e.g., [10]) approximate the posterior within a chosen parametric family, but since the resulting approximation error is generally hard to quantify, they do not readily support rigorous probabilistic guarantees. We therefore employ PMCMC methods [2], which embed sequential Monte Carlo algorithms, also known as particle filters, in Markov chain Monte Carlo (MCMC) samplers. Given the model parameters, a particle filter approximates the conditional distribution of latent state trajectories using weighted particles and provides an unbiased likelihood estimate, thereby enabling the construction of valid MCMC kernels targeting the exact joint posterior. The two main variants for joint inference of parameters and latent states are briefly reviewed below.

Particle marginal Metropolis–Hastings (PMMH) At each iteration, a candidate parameter $\scriptstyle { \overleftarrow { \theta } } ^ { \prime }$ is drawn from a proposal distribution $q ( \theta ^ { \prime } \ \bar { | } \ \theta ^ { ( i ) } )$ , and a particle filter with $N$ particles computes an unbiased estimate $\hat { p } ( \mathbb { D } \mid \pmb { \theta } ^ { \prime } )$ of the likelihood $p ( \mathbb { D } \mid \pmb { \theta } ^ { \prime } ) = p ( \pmb { y } _ { 0 : t _ { 0 } - 1 } \mid \pmb { u } _ { 0 : t _ { 0 } - 1 } , \pmb { \theta } ^ { \prime } )$ . The candidate is accepted with probability

$$
\rho = \operatorname* { m i n } \left( 1 , \frac { \hat { p } ( \mathbb { D } \mid \pmb { \theta } ^ { \prime } ) p ( \pmb { \theta } ^ { \prime } ) q ( \pmb { \theta } ^ { ( i ) } \mid \pmb { \theta } ^ { \prime } ) } { \hat { p } ( \mathbb { D } \mid \pmb { \theta } ^ { ( i ) } ) p ( \pmb { \theta } ^ { ( i ) } ) q ( \pmb { \theta } ^ { \prime } \mid \pmb { \theta } ^ { ( i ) } ) } \right) ,\tag{5}
$$

in which case $\pmb { \theta } ^ { ( i + 1 ) } = \pmb { \theta } ^ { \prime }$ and a state trajectory $\pmb { x } _ { 0 : t _ { 0 } - 1 } ^ { ( i + 1 ) }$ is sampled from the corresponding particle system. Otherwise, the current parameter, trajectory, and likelihood estimate are retained. The efficiency of PMMH depends on how well the proposal matches the posterior and on the number of particles N, which controls the variance of the log-likelihood estimate. If this variance is high, an unusually large estimate can cause subsequent candidates to be rejected over many iterations. Both can be adapted by incorporating the data gradually and using the samples of each stage to tune the next: the proposal covariance is set to their empirical covariance, and N is chosen to achieve a target variance of the log-likelihood estimate [12]. Within each stage, the proposal scale can additionally be adjusted by a Robbins–Monro recursion that drives the acceptance rate toward a target value [3].

Particle Gibbs (PG) Instead of estimating the likelihood, PG alternates between drawing a new state trajectory with a conditional particle filter, which retains the previous trajectory as a reference particle, and sampling the parameters from $p ( \pmb \theta \mid \pmb x _ { 0 : t _ { 0 } - 1 } , \mathbb D )$ . This avoids the accept–reject step of PMMH and can improve mixing, but requires that this conditional distribution can be sampled directly, which is often achieved by choosing a conjugate prior [30]. For long data records, path degeneracy in the conditional particle filter can make consecutive trajectories highly correlated, which ancestor sampling [18] mitigates even with few particles.

Under mild regularity conditions, both PMMH and PG admit $p ( \pmb \theta , \pmb x _ { 0 : t _ { 0 } - 1 } \mid \mathbb D )$ as their invariant distribution and generate asymptotically exact samples for any finite number of particles [2], provided that any adaptation of the proposal is stopped or vanishes asymptotically [3]. In practice, early samples are influenced by the initialization and are discarded as burn-in. Although no rigorous guarantees exist for the required length $K _ { \mathrm { b } }$ , its choice is guided by empirical convergence diagnostics. Moreover, successive samples are correlated, whereas independent samples are required in the following. Two remedies are common: thinning, whereby only every dth sample is retained with d chosen from the empirical autocorrelation function such that the remaining serial dependence is negligible, or running an independent chain for each retained sample, which yields independence by construction at the cost of a separate burn-in per chain. In either case, the retained draws can be treated as independent samples from the posterior, as formalized in the following assumption.

Assumption 2 (Independent posterior samples) The employed PMCMCprocedure provides K independent samples $\{ ( \pmb { \theta } ^ { [ k ] } , \pmb { x } _ { 0 : t _ { 0 } - 1 } ^ { [ k ] } ) \} _ { k = 1 } ^ { K } f r o m \ p ( \pmb { \theta } , \pmb { x } _ { 0 : t _ { 0 } - 1 } \ | \ \mathbb { D } )$

Each sample represents a realization of the unknown dynamics and latent states that is consistent with the observed data and prior knowledge. These samples serve as the basis for the control framework developed in the following sections.

Algorithm 1 Scenario generation   
Input: Dataset $\mathbb { D } = ( \boldsymbol { \mathfrak { u } } _ { 0 : t _ { 0 } - 1 } , \boldsymbol { \mathfrak { y } } _ { 0 : t _ { 0 } - 1 } ) ,$   
model $\{ f _ { \pmb \theta } ( \cdot ) , \pmb \{ g _ { \pmb \theta } ( \cdot ) , \} _ { \pmb \theta } , \mathcal { W } _ { \pmb \theta } \} ,$   
priors $p ( \pmb \theta )$ and $p ( { \pmb x } _ { 0 } \mid { \pmb \theta } ) ,$   
horizon $H ,$ number of scenarios K   
Output: Scenarios $\{ \pmb { \xi } ^ { [ k ] } \} _ { k = 1 } ^ { K } ,$ where   
ξ<sup>[k]</sup> = $( \pmb { \theta } ^ { [ k ] } , \pmb { x } _ { t _ { 0 } } ^ { [ k ] ^ { \ast } } , \pmb { v } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \pmb { w } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } )$   
1: for $k = 1$ to K do   
2: Draw $( \pmb { \theta } ^ { [ k ] } , \pmb { x } _ { 0 : t _ { 0 } - 1 } ^ { [ k ] } ) \sim p ( \pmb { \theta } , \pmb { x } _ { 0 : t _ { 0 } - 1 } \mid \mathbb { D } )$   
using PMCMC   
3: Sample ${ \pmb v } _ { t _ { 0 } - 1 : t _ { 0 } + H } ^ { [ k ] }$ with $\underline { { v _ { t } ^ { [ k ] } } } \sim \mathcal { V } _ { \pmb { \theta } ^ { [ k ] } }$   
for $t = t _ { 0 } - 1 , \ldots , t _ { 0 } + H$   
4: Sample ${ \pmb w } _ { t _ { 0 } : t _ { 0 } + H } ^ { \lfloor { k } \rfloor }$ with ${ \pmb w } _ { t } ^ { [ k ] } \sim \mathcal { W } _ { \pmb { \theta } ^ { [ k ] } }$   
for $\bar { t = { t _ { 0 } } , \ldots , { t _ { 0 } } + H }$   
5: $\pmb { x } _ { t _ { 0 } } ^ { [ k ] }  \pmb { f } _ { \pmb { \theta } ^ { [ k ] } } ( \pmb { x } _ { t _ { 0 } - 1 } ^ { [ k ] } , \pmb { u } _ { t _ { 0 } - 1 } , \pmb { v } _ { t _ { 0 } - 1 } ^ { [ k ] } )$   
6: Store scenario $\xi ^ { [ k ] }$   
7: end for

The framework is not tied to a specific PMCMC method, provided that the sampler satisfies Assumption 2.

## 4 Validation of Control Laws

For control purposes, and particularly in safety-critical applications, it is essential to assess how the uncertainty remaining after incorporating the data affects the robustness of a control law. We therefore consider the probability that the trajectories generated by a given control law satisfy the constraints (3) under the posterior predictive distribution induced by $p ( \pmb \theta , \pmb x _ { 0 : t _ { 0 } - 1 } \mid \mathbb D )$ and the disturbance model. Since this probability cannot be computed in closed form, we resort to randomized methods, which assess such probabilities from sampled uncertainty realizations with rigorous statistical guarantees [31,1]. These methods, however, assume that the underlying distribution can be sampled directly, which is not the case here. The PMCMC procedure of Section 3 closes this gap: combined with forward propagation to the current state and sampling of future disturbances, its posterior samples yield realizations of the posterior predictive distribution, on which we build a validation procedure with finite-sample guarantees.

For the remainder of this section, we consider measurable causal control laws ${ \pmb u } _ { t } = { \pmb \pi } ( { \pmb y } _ { 0 : t - 1 } , { \pmb u } _ { 0 : t - 1 } , t )$ taking values in U, which include fixed input sequences and output feedback laws. Due to the parametric state-space representation, future states and outputs are uniquely determined by the inputs together with the parameters θ, the current state $\mathbf { \mathcal { x } } _ { t _ { 0 } }$ and the noise realizations. A sample $\pmb { \xi } ^ { [ k ] }$ of these exogenous quantities from $p ( \pmb \theta , \pmb x _ { t _ { 0 } } , \pmb v _ { t _ { 0 } : t _ { 0 } + H } , \pmb w _ { t _ { 0 } : t _ { 0 } + H } \ | \ \mathbb D )$ , referred to as a scenario, therefore fully determines a future trajectory for any given control law. Algorithm 1 generates such scenarios by drawing parameters and past state trajectories via PMCMC, sampling noise realizations from the parameterized distributions, and propagating the state forward to $t _ { 0 }$

Algorithm 2 Validation of a control law   
Input: Validation scenarios $\{ \pmb { \xi } ^ { [ k ] } \} _ { k \in \mathcal { S } _ { \mathrm { v a l } } } ,$   
control law $\pmb { u } _ { t } = \pi ( \pmb { y } _ { 0 : t - 1 } , \pmb { u } _ { 0 : t - 1 } , t ) .$   
horizon H   
Output: Number of scenarios with constraint violations V   
1: $\bar { V }  0$   
2: for $k \in S _ { \mathrm { v a l } }$ do   
3: $( \pmb { u } _ { 0 : t _ { 0 } - 1 } ^ { [ k ] } , \pmb { y } _ { 0 : t _ { 0 } - 1 } ^ { [ k ] } )  ( \pmb { u } _ { 0 : t _ { 0 } - 1 } , \pmb { y } _ { 0 : t _ { 0 } - 1 } )$   
4: for $t = t _ { 0 }$ to $\dot { t } _ { 0 } + H$ do   
5: ${ \pmb u } _ { t } ^ { [ k ] }  \pi ( { \pmb y } _ { 0 : t - 1 } ^ { [ k ] } , { \pmb u } _ { 0 : t - 1 } ^ { [ k ] } , t )$   
6: $\pmb { x } _ { t + 1 } ^ { [ k ] }  \pmb { f _ { \pmb { \theta } ^ { [ k ] } } } ( \pmb { x } _ { t } ^ { [ k ] } , \pmb { u } _ { t } ^ { [ k ] } , \pmb { v } _ { t } ^ { [ k ] } )$   
7: $\pmb { y } _ { t _ { - } } ^ { [ k ] }  \pmb { g } _ { \pmb { \theta } ^ { [ k ] } } ( \pmb { x } _ { t } ^ { [ k ] } , \pmb { u } _ { t } ^ { [ k ] } ) + \pmb { w } _ { t } ^ { [ k ] }$   
8: end for   
9: if $\underset { \mathbf { \textsc { h } } _ { f } } { \mathbf { \textsc { m } } } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \pmb { x } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \pmb { y } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } ) > 0$ then   
10: ${ \dot { V } }  \dot { V } + 1$   
11: end if   
12: end for

Statistical validation requires the scenarios to be independent of the control law, which we formalize as follows.

Assumption 3 (Independence of validation scenarios) The control law $\mathbf { } u _ { t } = \pi ( \cdot )$ is independent of the validation scenarios used to assess it, that is, it is neither designed nor adapted based on these scenarios.

This assumption is not restrictive in the present setting. Although the dataset D is fixed, Algorithm 1 can generate arbitrarily many independent scenarios, so that separate scenarios can be used for synthesis and validation without reducing the information available for controller design. Let $K _ { \mathrm { v a l } }$ denote the number of validation scenarios and $ { S _ { \mathrm { v a l } } }$ their index set.

To assess the robustness of a control law, the closed-loop system is simulated over the horizon H for each validation scenario $\pmb { \xi } ^ { [ k ] } , k \in S _ { \mathrm { v a l } }$ , and it is checked whether the constraints (3) are satisfied (Algorithm 2). The constraint value

$$
Z ^ { [ k ] } = h \left( \boldsymbol { u } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \boldsymbol { x } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \boldsymbol { y } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } \right)\tag{6}
$$

indicates a violation whenever $Z ^ { [ k ] } > 0 .$ , and the total number of violations over the validation set is

$$
V = \sum _ { k \in \mathcal { S } _ { \mathrm { v a l } } } { \bf 1 } \left\{ Z ^ { [ k ] } > 0 \right\} ,\tag{7}
$$

where 1{·} denotes the indicator function. Based on this count, the following theorem provides a finite-sample guarantee for constraint satisfaction.

Theorem 4 (Robustness guarantee) Suppose that Assumptions 1, 2, and 3 hold. Let V denote the number of constraint violations observed among the $K _ { \mathrm { v a l } }$ validation scenarios, let $1 - \beta$ with $\beta \in ( 0 , 1 )$ be a chosen confidence level, and let $\nu : = \mathbb { P } \left( Z > 0 \mid \mathbb { D } , \pi \right)$ denote the violation probability under the posterior predictive distribution. If $\mathbf { \bar { \psi } } V < K _ { \mathrm { v a l } } , l e t \delta \in ( 0 , 1 \bar { ) }$ be the unique solution of

$$
\sum _ { i = 0 } ^ { V } { \binom { K _ { \mathrm { v a l } } } { i } } \delta ^ { i } ( 1 - \delta ) ^ { K _ { \mathrm { v a l } } - i } = \beta ;\tag{8}
$$

$i f V = K _ { \mathrm { v a l } } , s e t \delta = 1$ . Then, with confidence at least $1 - \beta ,$ it holds that $\nu \leq \delta ,$ , i.e., the constraints (3) are satisfied under $\pi ( \cdot )$ withposteriorpredictiveprobability at least $1 - \delta$

PROOF. By Assumption 1, the true system is a realization of the probabilistic model, so that the posterior predictive distribution describes its possible future trajectories given D. By Assumptions 2 and 3, the validation scenarios are independent draws from this distribution and independent of $\pi ( \cdot )$ . Conditional on D and $\pi ,$ , the violation indicators are therefore i.i.d. Bernoulli with parameter $\nu ,$ and $V \sim \mathrm { B i n o m i a l } ( K _ { \mathrm { v a l } } , \nu )$ . For this binomial model, the Clopper–Pearson construction [9] yields the upper $( 1 - \beta )$ confidence bound δ on ν given by (8) for $V < \bar { K } _ { \mathrm { v a l } }$ , while $\nu \leq \delta = 1$ holds trivially for $V = K _ { \mathrm { v a l } }$ ■

For a fixed empirical violation rate $V / K _ { \mathrm { v a l } } ,$ the guarantee $1 - \delta$ strengthens as the number of validation scenarios $K _ { \mathrm { v a l } }$ grows, and for a fixed $K _ { \mathrm { v a l } }$ , fewer violations yield a stronger guarantee [9]. The confidence parameter $\beta$ controls the statistical reliability of the bound: if the validation procedure were repeated many times, the bound $1 - \delta$ would hold in at least ${ \bf a } \mathrm { ~ 1 ~ - ~ } \beta$ fraction of these repetitions. Since δ is a Beta-distribution quantile that changes only slowly in $\beta ,$ the confidence parameter can be chosen very small $( \mathrm { e . g . , } \beta = 1 0 ^ { - 3 } )$ without significantly weakening the guarantee [32]. If several candidates are validated, $\beta$ must be allocated across attempts to account for multiple testing [14].

## 5 Risk-Sensitive Optimal Control

The validation procedure of the previous section certifies a given control law but does not construct one. This section develops a systematic method that computes and certifies an input sequence for the OCP (1)–(4). As in sequential randomized methods [31,1], a candidate is synthesized on $K _ { \mathrm { t r a i n } }$ training scenarios with index set $\boldsymbol { S } _ { \mathrm { t r a i n } }$ and then assessed on an independent validation set of size $K _ { \mathrm { v a l } }$ . Section 5.1 formulates a synthesis problem whose conservativeness is adjusted by a single risk weight, Section 5.2 describes how this weight is calibrated across repeated synthesis–validation attempts, and Section 5.3 analyzes the resulting procedure.

## 5.1 CVaR-Penalized Scenario Program

Given the training scenarios $S _ { \mathrm { t r a i n } } .$ , we seek a candidate input sequence that approximates the solution of the stochastic OCP (1)–(4). To this end, we formulate a scenario-based optimization problem in which the training scenarios are treated as deterministic realizations of the uncertainty. A natural approach would be to enforce the constraints for all scenarios, but this is overly conservative, since a few lowprobability scenarios can severely restrict the feasible set. Explicitly selecting which scenarios may violate the constraints, on the other hand, leads to an expensive combinatorial problem. Instead, we penalize tail risk through the CVaR, which allows the conservativeness to be adjusted continuously without changing the structure of the problem.

The CVaR quantifies the expected value of the upper tail of a distribution. The constraint value $Z = h ( \cdot )$ and the cost J are random variables due to the uncertain dynamics and disturbances, assumed to be integrable with respect to the posterior predictive distribution for every admissible input sequence. Minimizing the CVaR of $Z$ tends to reduce both the magnitude and the frequency of violations, and since fewer violations yield a tighter bound in Theorem $^ { 4 , }$ , this improves the certifiable robustness of the synthesized input sequence. For a tail level $\tau \in ( 0 , 1 )$ , the CVaR is defined as

$$
\operatorname { C V a R } _ { \tau } ( Z ) = \operatorname* { m i n } _ { \eta \in \mathbb { R } } \left( \eta + \frac { 1 } { \tau } \mathbb { E } \left[ ( Z - \eta ) _ { + } \right] \right) ,\tag{9}
$$

where $( a ) _ { + } = \operatorname* { m a x } ( a , 0 )$ and the smallest minimizer is the value-at-risk (VaR)

$$
\operatorname { V a R } _ { \tau } ( Z ) = \operatorname* { i n f } \{ z \in \mathbb { R } : \mathbb { P } ( Z > z ) \leq \tau \} ,\tag{10}
$$

i.e., the $( 1 - \tau )$ -quantile of Z [24]. We set $\tau = \alpha$ by default, which aligns the penalized tail with the desired robustness level. Combining the expected cost with a penalty on the CVaR yields the risk-sensitive objective

$$
\operatorname* { m i n } _ { \substack { { \boldsymbol u } _ { t _ { 0 } : t _ { 0 } + H } } } ( 1 - \omega ) \mathbb { E } [ J ] \ + \ \omega \mathrm { C V a R } _ { \tau } ( Z ) ,\tag{11}
$$

where $\mathcal { U } = \mathbb { U } ^ { H + 1 }$ denotes the set of admissible input sequences. The risk weight $\omega \in [ 0 , 1 ]$ balances nominal performance and robustness: at $\omega = 0$ , the CVaR term is inactive and (11) reduces to the risk-neutral problem of minimizing the expected cost, whereas at $\omega = 1$ the cost is disregarded and only the tail risk is minimized.

Since the expectations in (11) are generally intractable, we replace them by sample averages over the training scenarios, each of which induces realizations of the cost $\boldsymbol { J } ^ { [ k ] }$ and the constraint value $Z ^ { [ k ] }$ . This yields the scenario-based optimization problem

$$
\begin{array} { r l r } {  { \operatorname* { m i n } _ { u _ { t _ { 0 } : t _ { 0 } + H } \in \mathcal { U } , \eta } \frac { 1 - \omega } { K _ { \mathrm { t r a i n } } } \sum _ { k \in S _ { \mathrm { t r a i n } } } J ^ { [ k ] } } } \\ & { } & { + \omega ( \eta + \frac { 1 } { \tau K _ { \mathrm { t r a i n } } } \sum _ { k \in \mathcal { S } _ { \mathrm { t r a i n } } } ( Z ^ { [ k ] } - \eta ) _ { + } ) } \end{array}\tag{12a}
$$

$$
\mathrm { s . t . } \quad \forall k \in \mathcal { S } _ { \mathrm { t r a i n } } , \quad \forall t \in \left\{ t _ { 0 } , \ldots , t _ { 0 } + H \right\}
$$

$$
\begin{array} { r } { \pmb { x } _ { t + 1 } ^ { [ k ] } = \pmb { f } _ { \pmb { \theta } ^ { [ k ] } } \left( \pmb { x } _ { t } ^ { [ k ] } , \pmb { u } _ { t } , \pmb { v } _ { t } ^ { [ k ] } \right) , } \end{array}\tag{12b}
$$

$$
{ \pmb y } _ { t } ^ { [ k ] } = { \pmb g } _ { \pmb { \theta } ^ { [ k ] } } \left( { \pmb x } _ { t } ^ { [ k ] } , { \pmb u } _ { t } \right) + { \pmb w } _ { t } ^ { [ k ] } ,\tag{12c}
$$

$$
J ^ { [ k ] } = J \left( u _ { t _ { 0 } : t _ { 0 } + H } , \pmb { x } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \pmb { y } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } \right) ;
$$

$$
Z ^ { [ k ] } = h \left( \boldsymbol { u } _ { t _ { 0 } : t _ { 0 } + H } , \boldsymbol { x } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , \boldsymbol { y } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } \right)\tag{12d}
$$

(12e)

Since the only hard constraints apart from the dynamics are the input constraints, while violations of (3) enter via the CVaR penalty, problem (12) is feasible for any ω. We assume that a solution can be computed by a suitable numerical optimization method [23], which may require additional regularity of the dynamics, cost, and constraint functions. Since the problem is generally non-convex, global optimality cannot be guaranteed. The certificate, however, does not rely on optimality, and locally optimal solutions are often sufficient in practice. Importantly, the training and validation set sizes can be chosen independently: while $K _ { \mathrm { t r a i n } }$ is limited by the computational cost of solving (12), validation only requires forward simulation, so that $K _ { \mathrm { v a l } }$ can be chosen substantially larger, yielding a tighter bound in Theorem 4.

## 5.2 Adaptive Calibration of the Risk Weight

Since the appropriate risk weight ω is not known a priori, it is calibrated in a synthesis–validation loop, illustrated in Figure 1. In attempt j, the scenario OCP (12) is solved for a given risk weight $\omega _ { j } .$ , and the resulting candidate input sequence $\hat { \pmb { u } } _ { t _ { 0 } : t _ { 0 } + H } ^ { ( j ) }$ is evaluated on an independent validation set, yielding the bound $\delta _ { j }$ of Theorem 4. We call the candidate certified if $\delta _ { j } \leq \alpha$ . The search terminates once a certified candidate is found whose bound lies within a specified tolerance of $\alpha .$ . Otherwise, ω is adjusted and the synthesis step is repeated. The training scenarios are reused across attempts to keep the approximation of the objective consistent, while a fresh validation set is generated at each attempt to maintain independence (Assumption 3).

Each attempt, however, constitutes a separate statistical test, and repeated validation inflates the probability of a spurious certification. To maintain the overall confidence level $1 - \beta _ { ; }$ at most $n _ { \mathrm { m a x } }$ attempts are permitted and the confidence parameter is split uniformly as $\beta _ { \mathrm { a t t } } = \beta / n _ { \mathrm { m a x } }$ [14], which weakens the individual bounds only marginally (cf. Section 4). This limited budget requires a schedule that explores the range of ω efficiently and reaches the most conservative choice $\omega = 1$ by the final attempt.

## 5.2.1 Response model

To select the risk weight efficiently, we characterize how the probability ν that a scenario violates the constraints (cf. Theorem 4) depends on it. For $\omega < 1$ , dividing (11) by $1 - \omega$

![](images/c5320099aa6a3e83bf8f19a3e5b9009fdcbb3b65da96c86e44c819f20412fa63.jpg)  
Fig. 1. Overview of the proposed approach. Training and validation scenarios are generated from PMCMC posterior samples. In attempt $j ,$ a candidate input sequence is synthesized from the training scenarios with risk weight $\omega _ { j }$ and evaluated on a fresh validation set, yielding a high-confidence bound $\delta _ { j }$ on its violation probability. The risk weight is updated until a candidate is certified within the acceptance band or the attempt budget is exhausted. Colors indicate scenario generation (blue, Section 3 and Algorithm 1), validation (green, Section 4), and synthesis and calibration (orange, Section 5).

yields the equivalent objective

$$
\operatorname* { m i n } _ { { \substack { u _ { t _ { 0 } : t _ { 0 } + H } \in \mathcal { U } } } } ~ \mathbb { E } [ J ] + \lambda \mathrm { C V a R } _ { \tau } ( Z ) , \qquad \lambda = \frac { \omega } { 1 - \omega } ,\tag{13}
$$

so that the trade-off is governed by the relative penalty weight $\lambda \in [ 0 , \infty )$ , which provides a natural parameterization for the calibration. The violation probability ν typically exhibits a saturating response: for small $\lambda ,$ the CVaR term has little influence on the solution, whereas for large $\lambda ,$ further increasing its weight produces diminishing changes and ν may approach a floor induced by the remaining uncertainty. The calibration therefore targets the transition region in between, where we locally linearize log ν in log λ, which corresponds to a power law $\nu \approx c \lambda ^ { s }$ and yields

$$
\log \nu \approx a + s \log \lambda , \qquad s = \frac { \mathrm { d } \log \nu } { \mathrm { d } \log \lambda } < 0 .\tag{14}
$$

The local elasticity s is assumed to be negative, since a larger weight on the tail risk tends to shift the solution toward fewer violations (cf. Section 5.3).

Since s is unknown, it is estimated from the preceding attempts, each of which yields an empirical violation rate $\begin{array} { r } { \hat { \nu } _ { j } = { V } _ { j } / { K } _ { \mathrm { v a l } } . } \end{array}$ , a noisy estimate of $\nu _ { j }$ . To balance locality against noise averaging, we use exponentially weighted least squares in (log λ, log ˆν) and exclude saturated attempts with $V _ { j } \in \{ 0 , \mathrm { \bar { \cal K } _ { v a l } } \}$ , which carry no slope information. Until sufficiently many attempts are available, an initial value $s _ { \mathrm { i n i t } } < 0$ is used, and $\hat { s } _ { j }$ is bounded from above by $s _ { \operatorname* { m a x } } < 0$ to preserve a decreasing response under sampling noise.

## 5.2.2 Weight update

The next weight is chosen to reach a target violation probability $\nu _ { \mathrm { t a r } }$ . Since the bound δ is strictly increasing in the violation count $V ,$ certification at level $1 - \alpha$ is equivalent to $V _ { j } \leq V _ { \alpha } ^ { * }$ , where

$$
V _ { \alpha } ^ { * } = \operatorname* { m a x } \left. v \in \mathbb { N } ^ { 0 } : \delta _ { v } \leq \alpha \right.\tag{15}
$$

is the largest violation count that still permits certification, and $\delta _ { v }$ follows from (8) with $V = { \bar { v } }$ and $\beta = \beta _ { \mathrm { a t t } } .$ We assume that $K _ { \mathrm { v a l } }$ is large enough that $V _ { \alpha } ^ { * } \geq 1$ , since otherwise certification would require zero violations and the target would degenerate. Targeting the corresponding rate $V _ { \alpha } ^ { * } / K _ { \mathrm { v a l } }$ directly would place $\nu _ { \mathrm { t a r } }$ exactly at the certification boundary, where sampling variability in the validation set alone would lead to frequent rejections. We therefore set $\nu _ { \mathrm { t a r } } = ( 1 - m ) V _ { \alpha } ^ { * } / K _ { \mathrm { v a l } }$ with a relative margin $m \in [ 0 , 1 )$ Together with $\hat { s } _ { j }$ and the current outcome, (14) then yields the step

$$
\Delta _ { j } = \frac { \log \nu _ { \mathrm { t a r } } - \log \hat { \nu } _ { j } } { \hat { s } _ { j } } , \quad \log \lambda _ { j + 1 } = \log \lambda _ { j } + \Delta _ { j } ,\tag{16}
$$

which increases λ when the observed rate exceeds the target and decreases it otherwise, with a magnitude that adapts to the local sensitivity. Since (14) is only a local approximation with a noisy slope estimate, $| \Delta _ { j } |$ is bounded in practice, which also covers the case $V _ { j } = 0$ , and $\lambda _ { j + 1 }$ is restricted to lie between the largest risk weight that failed to certify and the smallest one that certified with more margin than required. These choices affect only the number of attempts, not what is certified.

Since the update (16) ignores the remaining budget, an inaccurate slope estimate may exhaust all attempts without certification, even if a certifiable candidate exists at a larger risk weight. As a conservative certified solution is preferable to none, a recovery phase is entered over the final $n _ { \mathrm { r e c } }$ attempts if no candidate has certified yet. With $r _ { j } = n _ { \mathrm { m a x } } - j$ remaining attempts, the recovery proposal

$$
\omega _ { j + 1 } ^ { \mathrm { r e c } } = \omega _ { j } + \frac { 1 - \omega _ { j } } { r _ { j } }\tag{17}
$$

advances ω in equal steps and reaches $\omega = 1$ , which corresponds to $\lambda = \infty$ , exactly at the final attempt. During re-

covery, the more conservative of the two proposals is used,

$$
\omega _ { j + 1 } = \operatorname* { m a x } \left\{ \frac { \lambda _ { j + 1 } } { 1 + \lambda _ { j + 1 } } , \omega _ { j + 1 } ^ { \mathrm { r e c } } \right\} ,\tag{18}
$$

and once a candidate has certified, recovery is disabled.

## 5.2.3 Termination

Every certified candidate is admissible, regardless of its risk weight. However, a bound $\delta _ { j }$ well below α typically indicates an unnecessarily large λ at the expense of nominal performance. The search therefore terminates once the bound lies within the acceptance band

$$
\alpha - \alpha _ { \mathrm { t o l } } ~ \le ~ \delta _ { j } ~ \le ~ \alpha\tag{19}
$$

with tolerance $\alpha _ { \mathrm { t o l } } ~ \in ~ ( 0 , \alpha ]$ , or once the attempt budget is exhausted. The admissible candidate with the smallest empirical cost

$$
\hat { J } = \frac { 1 } { K _ { \mathrm { v a l } } } \sum _ { k \in \cal S _ { \mathrm { v a l } } } J \big ( { \pmb u } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , { \pmb x } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } , { \pmb y } _ { t _ { 0 } : t _ { 0 } + H } ^ { [ k ] } \big ) ,\tag{20}
$$

evaluated along the trajectories simulated in Algorithm $^ { 2 , }$ is then returned as $\boldsymbol { \mathbf { \mathit { u } } } _ { t _ { 0 } : t _ { 0 } + H } ^ { * }$ together with its bound $\delta ^ { * }$ No further correction is required, since the union bound underlying $\beta _ { \mathrm { a t t } }$ covers all attempts simultaneously, so the certificate holds regardless of which admissible candidate is selected. If no candidate has certified despite recovery, the candidate with the smallest bound is returned instead, with $\delta ^ { * } = \operatorname* { m i n } _ { j } \delta _ { j } > \alpha$ and a failure flag.

## 5.3 Analysis

The validity of the certificate returned by the synthesis– validation procedure follows directly from the results of Section 4. Each attempt is evaluated on a fresh, independent validation set, so that Theorem 4 applies with confidence parameter $\beta _ { \mathrm { a t t } }$ , and a union bound over all $n _ { \mathrm { m a x } }$ attempts yields confidence at least $1 - \beta$ for the returned bound $\delta ^ { * }$ This guarantee holds for any candidate, regardless of how it was obtained. In particular, it remains valid if the sampleaverage approximation is inaccurate, the synthesized solution is only locally or approximately optimal, or the local model (14) is a poor fit.

It remains to analyze whether the procedure finds a certified candidate. To this end, we show that the risk weight orders the synthesized solutions by tail risk, that a negative tail risk bounds the violation probability strictly below the target level, and that this yields a lower bound on the certification probability. To study these relations independently of finite-sample effects in the training scenarios, we analyze the population-level objective (11), whose minimizers are defined directly with respect to the posterior predictive distribution. In the following, we write $\pmb { u } = \pmb { u } _ { t _ { 0 } : t _ { 0 } + H } \in \mathcal { U }$ for brevity and assume that this objective attains a finite minimum for every $\omega \in [ 0 , 1 ]$ , which holds, e.g., if U is compact and $\mathbb { E } [ J ]$ and $\mathrm { C V a R } _ { \alpha } ( \bar { Z } )$ are continuous on U.

Lemma 5 (Monotonicity) For each ω $\in \ [ 0 , 1 ] ,$ , let $\pmb { u } ^ { * } ( \omega )$ be any global minimizer over U of the population objective (11) with $\tau = \alpha$ . Then, for all $0 \leq \omega _ { 1 } < \omega _ { 2 } \leq 1 _ { \cdot }$ , it holds that

$$
\mathrm { C V a R } _ { \alpha } \left( Z ( \pmb { u } ^ { * } ( \omega _ { 2 } ) ) \right) \le \mathrm { C V a R } _ { \alpha } \left( Z ( \pmb { u } ^ { * } ( \omega _ { 1 } ) ) \right) ,\tag{21}
$$

$$
\mathbb { E } \left[ J ( \pmb { u } ^ { * } ( \omega _ { 2 } ) ) \right] \geq \mathbb { E } \left[ J ( \pmb { u } ^ { * } ( \omega _ { 1 } ) ) \right] .\tag{22}
$$

PROOF. The result follows from standard properties of weighted-sum scalarizations in bi-objective optimization; see, e.g., [6, Section 4.7.5]. Define $\bar { C } ( \boldsymbol { u } ) = \bar { \mathbb { E } } [ J ( \boldsymbol { u } ) ]$ and $R ( { \pmb u } ) \ = \ \mathrm { C V a R } _ { \alpha } ( Z ( { \pmb u } ) )$ , and consider first $\omega _ { 2 } ~ < ~ 1$ . For $i \in \{ 1 , 2 \}$ , let $\mathbf { } \mathbf { } u _ { i } = \mathbf { } u ^ { * } ( \omega _ { i } )$ and $\lambda _ { i } = \omega _ { i } / ( 1 - \omega _ { i } )$ . Since $\mathbf { \delta } \mathbf { u } _ { i }$ minimizes (13) with $\lambda = \lambda _ { i }$

$$
C ( \mathbf { \boldsymbol { u } } _ { 2 } ) + \lambda _ { 2 } R ( \mathbf { \boldsymbol { u } } _ { 2 } ) \leq C ( \mathbf { \boldsymbol { u } } _ { 1 } ) + \lambda _ { 2 } R ( \mathbf { \boldsymbol { u } } _ { 1 } ) ,\tag{23}
$$

$$
C ( \mathbf { \boldsymbol { u } } _ { 1 } ) + \lambda _ { 1 } R ( \mathbf { \boldsymbol { u } } _ { 1 } ) \leq C ( \mathbf { \boldsymbol { u } } _ { 2 } ) + \lambda _ { 1 } R ( \mathbf { \boldsymbol { u } } _ { 2 } ) .\tag{24}
$$

Adding the inequalities yields $( \lambda _ { 2 } - \lambda _ { 1 } ) ( R ( { \pmb u } _ { 2 } ) - R ( { \pmb u } _ { 1 } ) ) \leq$ 0. Since $\omega \mapsto \lambda$ is strictly increasing on $[ 0 , 1 ) , \lambda _ { 2 } > \lambda _ { 1 }$ and hence $R ( { \mathbf { u } } _ { 2 } ) \leq R ( { \mathbf { u } } _ { 1 } )$ . For $\omega _ { 2 } = 1$ , the objective (11) reduces to R alone, so that ${ \cal R } ( u _ { 2 } ) \le { \cal R } ( u )$ for every $\mathbf { \pmb { u } } \in \mathcal { U } .$ which extends (21) to the entire interval [0, 1]. In both cases, rearranging (24), which relies only on the optimality of ${ \mathbf { } } ^ { \mathbf { } } \mathbf { 1 } \mathbf { . }$ and using $\lambda _ { 1 } \geq 0$ together with (21) gives $C ( { \bf \dot { u } } _ { 1 } ) { - } \dot { C } ( { \bf \dot { u } } _ { 2 } ) \le$ $\begin{array} { r }  \lambda _ { 1 } ( R ( \mathbf { \check { u } } _ { 2 } ) - R ( \mathbf { \check { u } } _ { 1 } \boldsymbol { \bar { ) } ) \leq 0 . } \end{array}$ , which proves (22). ■

Lemma 5 establishes the qualitative premise of the calibration loop: increasing ω never yields a global solution with a larger CVaR of the constraint value or a smaller expected cost. Since certification depends on the violation probability rather than on the CVaR, we connect the two under the following mild regularity condition.

Assumption 6 (Uniform lower-tail bound) There exists $M > 0$ such that

$$
\operatorname { V a R } _ { \alpha } \left( Z ( \pmb { u } ) \right) \ge - M\tag{25}
$$

for every $\mathbf { \pmb { u } } \in \mathcal { U } ,$ , with $\mathrm { V a R } _ { \alpha }$ as defined in (10).

Assumption 6 is implied by a uniform lower bound on the constraint value itself, which holds whenever the admissible trajectories remain within a bounded operating region and the constraint function is continuous, as is common for physical systems. The constant M enters only the analysis and is neither required nor estimated by the procedure.

Lemma 7 (Violation-probability margin) Let Assumption 6 hold and let $\textbf { \em u } \in \boldsymbol { U }$ $I f \mathrm { C V a R } _ { \alpha } ( Z ( \pmb { u } ) ) \le - \kappa$ for some $\kappa > 0 ,$ , then

$$
\mathbb { P } \left( Z ( \boldsymbol { \mathsf { u } } ) > 0 \right) \le \alpha \left( 1 - \frac { \kappa } { M } \right) .\tag{26}
$$

PROOF. Write $Z = Z ( { \boldsymbol { \mathbf { u } } } )$ , let $\nu = \mathbb { P } ( Z > 0 )$ , and recall the quantile representation [24]

$$
\operatorname { C V a R } _ { \alpha } ( Z ) = \frac { 1 } { \alpha } \int _ { 0 } ^ { \alpha } \operatorname { V a R } _ { q } ( Z ) \mathrm { d } q .\tag{27}
$$

Since $\operatorname { V a R } _ { q } ( Z )$ as defined in (10) is non-increasing in $q , \mathbf { A s } -$ sumption 6 implies $\mathrm { V a R } _ { q } ( Z ) \geq - M$ for all $q \leq \alpha$ . Moreover, for every $q < \nu , \mathbb { P } ( \mathcal { \hat { Z } } > 0 ) > q$ implies $\begin{array} { r } { \hat { \mathrm { V a R } } _ { q } ( Z ) \geq 0 . } \end{array}$ It follows that $\nu < \alpha .$ , as otherwise the latter bound would hold almost everywhere on the integration range and yield $\mathrm { C V a R } _ { \alpha } ( Z ) \geq \dot { 0 }$ , contradicting $\begin{array} { r } { \mathrm { C } \breve { \mathrm { V a R } } _ { \alpha } ( Z ) \ \breve { \leq } \ - \kappa \ \breve { < } \ 0 , } \end{array}$ Splitting the quantile integral at ν then gives

$$
\operatorname { C V a R } _ { \alpha } ( Z ) \geq { \frac { 1 } { \alpha } } \left( \int _ { 0 } ^ { \nu } 0 \mathrm { d } q + \int _ { \nu } ^ { \alpha } - M \mathrm { d } q \right) = - { \frac { \alpha - \nu } { \alpha } } M .
$$

Combining this with $\mathrm { C V a R } _ { \alpha } ( Z ) \leq - \kappa$ yields (26).

(28)

That a non-positive CVaR implies satisfaction of the corresponding chance constraint is well known [24]. Lemma 7 quantifies this implication: once the CVaR is driven below zero, the bound on the violation probability decreases linearly with the margin κ. To apply this along the family of global solutions, let $\mathcal { U } ^ { * } ( \omega )$ denote the set of global minimizers of (11) over U at ω and define the guaranteed CVaR margin

$$
\kappa ( \omega ) = - \operatorname* { s u p } _ { \boldsymbol { u } \in \mathcal { U } ^ { * } ( \omega ) } \operatorname { C V a R } _ { \alpha } \left( Z ( \boldsymbol { u } ) \right) .\tag{29}
$$

Every global minimizer at $\omega$ then has violation probability at most

$$
\bar { \nu } ( \omega ) = \left\{ \begin{array} { l l } { \displaystyle \alpha \left( 1 - \frac { \kappa ( \omega ) } { M } \right) , } & { \displaystyle \kappa ( \omega ) \geq 0 , } \\ { \displaystyle 1 , } & { \displaystyle \kappa ( \omega ) < 0 , } \end{array} \right.\tag{30}
$$

where the first case follows from Lemma 7 for $\kappa ( \omega ) > 0$ and from the classical implication for $\kappa ( \omega ) = 0 .$ , while for $\kappa ( \omega ) < 0$ only the trivial bound holds. An independent validation attempt therefore certifies with probability at least

$$
p _ { \mathrm { c e r t } } ( \omega ) = \sum _ { v = 0 } ^ { V _ { \alpha } ^ { * } } { \binom { K _ { \mathrm { v a l } } } { v } } \bar { \nu } ( \omega ) ^ { v } \left( 1 - \bar { \nu } ( \omega ) \right) ^ { K _ { \mathrm { v a l } } - v } .\tag{31}
$$

Since Lemma 5 holds for every choice of global minimizers, $\kappa ( \omega )$ is non-decreasing and hence $\bar { \nu } ( \omega )$ non-increasing in ω. $\mathbf { A } \mathbf { s } \left( 3 1 \right)$ is the binomial cumulative distribution function with success probability $\bar { \nu } ( \omega )$ evaluated at $V _ { \alpha } ^ { * }$ , which decreases in the success probability, $p _ { \mathrm { c e r t } } ( \omega )$ is non-decreasing in ω. This establishes the mechanism underlying the calibration: the larger the risk weight $\omega ,$ the higher the guaranteed probability that the resulting candidate is certified.

The procedure exploits this in two ways. As long as no candidate has certified, the observed violation rate lies above the target, so that the update (16) raises the risk weight, at a pace governed by the local model (14). The recovery phase, in turn, enforces progress toward $\omega = 1$ independently of the accuracy of this model. The following theorem quantifies the resulting probability of successful certification.

Theorem 8 (Certification probability) Suppose Assumptions 1–6 hold, and that each synthesis step returns a global minimizer of the population objective (11) with $\tau \ = \ \alpha .$ Then, with respect to the validation scenarios generated during calibration, the synthesis–validation procedure of Section 5.2 returns a certified input sequence with probability at least

$$
1 - \prod _ { k = 1 } ^ { n _ { \mathrm { r e c } } } \left[ 1 - p _ { \mathrm { c e r t } } \left( \frac { k } { n _ { \mathrm { r e c } } } \right) \right] \geq p _ { \mathrm { c e r t } } ( 1 ) .\tag{32}
$$

PROOF. Since the risk weights depend on the preceding validation outcomes, we restrict attention to the recovery phase, which is conservative, as certification at an earlier attempt can only increase the overall success probability. Let $\omega _ { \mathrm { e } }$ denote the risk weight at the entry to recovery and let $\widetilde { \omega } _ { k }$ denote the risk weight of its kth subsequent attempt. Conditional on no previous certification, the recovery phase remains active, and (17) together with (18) yields

$$
\widetilde { \omega } _ { k } \geq \omega _ { \mathrm { e } } + \frac { k } { n _ { \mathrm { r e c } } } \left( 1 - \omega _ { \mathrm { e } } \right) \geq \frac { k } { n _ { \mathrm { r e c } } } , \quad k = 1 , \ldots , n _ { \mathrm { r e c } } ,\tag{33}
$$

where the second inequality removes the dependence on the preceding attempts. Now condition on any history under which no earlier attempt has certified. This history determines both $\widetilde { \omega } _ { k }$ and the synthesized candidate. Since the validation scenarios are independent of that candidate by Assumption 3, the conditional probability of certification is at least $p _ { \mathrm { c e r t } } ( \widetilde { \omega } _ { k } ) \geq p _ { \mathrm { c e r t } } ( k / n _ { \mathrm { r e c } } )$ , uniformly over all such histories, by (33) and the monotonicity of $p _ { \mathrm { c e r t } } .$ Applying these bounds successively yields the first inequality in (32). The second follows by retaining only the factor $k = n _ { \mathrm { r e c } } ,$ since all factors lie in [0, 1]. ■

The probability in Theorem 8 concerns whether the calibration finds a candidate that passes validation. It is distinct from the confidence level $1 - \beta ,$ , which concerns the correctness of the returned bound $\delta ^ { * }$ . The lower bound (32) is informative whenever the solution at $\omega = 1$ , which minimizes the CVaR of the constraint value, attains a non-positive CVaR, and it strengthens as this value decreases. If even this solution has a positive CVaR, the bound becomes vacuous, which reflects the control problem rather than the calibration: the latter can identify an appropriate risk weight but cannot create robustness that the problem does not admit. Since the CVaR condition is only sufficient, however, a positive CVaR at $\omega = 1$ does not imply infeasibility of the chance constraint, and the procedure may nevertheless certify a candidate.

Two idealizations of the synthesis step remain. First, it solves the scenario OCP (12) instead of minimizing the population objective (11). Since the CVaR (9) is a minimum of an expectation over η, the scenario OCP is a standard sampleaverage approximation in the joint variable $( \pmb { u } , \eta )$ , so that, under mild regularity conditions, its objective converges uniformly to the population objective as $K _ { \mathrm { t r a i n } }  \infty$ [29]. Hence, for sufficiently large training sets, the synthesized solution attains a strict CVaR margin whenever the population problem at $\omega = 1$ admits one, which yields a corresponding positive lower bound on the certification probability. Second, the theorem assumes that global minimizers are obtained, which cannot be guaranteed for the non-convex problems considered here. Since Lemma 7 applies to any input sequence, a local minimizer with CVaR margin κ still certifies with probability at least $p _ { \mathrm { c e r t } }$ evaluated at that margin; only the ordering of Lemma 5 is lost. As established at the beginning of this section, neither idealization affects the validity of the certificate.

## 6 Receding-Horizon Implementation

The synthesis–validation procedure of Section 5.2 computes a certified input sequence for the entire horizon from the data available at time $t _ { 0 }$ . The measurements collected during execution, however, provide additional information about the system that a fixed input sequence cannot exploit, so that later inputs are chosen more conservatively than necessary. We therefore embed the procedure in a receding-horizon scheme and discuss how the resulting repeated inference and optimization can be implemented efficiently.

Let $t _ { i } = t _ { 0 } + i L , i \in \mathbb { N } ^ { 0 }$ , denote the start of the ith control cycle, where $L \leq H$ is the number of inputs applied per cycle, and let $\mathbb { D } _ { i } = ( \boldsymbol { \mathsf { u } } _ { 0 : t _ { i } - 1 } , \boldsymbol { \mathsf { y } } _ { 0 : t _ { i } - 1 } )$ denote the data available at that time. In each cycle, the synthesis–validation procedure is executed with $\mathbb { D } _ { i }$ in place of D, the first L inputs of the certified sequence are applied, and the next cycle starts from the extended dataset $\mathbb { D } _ { i + 1 }$

Remark 9 (Closed-loop guarantees) The guarantees of Theorem 4 apply to the planned trajectory of each control cycle and are re-established after every replanning step. They do not, however, constitute a joint guarantee over the complete closed-loop trajectory. This interpretation is standard in stochastic model predictive control, where probabilistic guarantees are inherently tied to the current planning problem [22]. If a joint guarantee over a fixed number of replanning steps is required, the risk level α and the confidence parameter $\beta$ can be allocated across these steps.

The computational cost of a control cycle is dominated by posterior sampling and by the synthesis of candidates, which requires solving the scenario OCP (12) in each attempt. Since each cycle adds only L new measurements to the accumulated dataset, the posterior, and with it the scenario

OCP, typically changes only gradually between cycles, particularly once the dataset is large. Both computations are therefore warm-started from the previous cycle, and within a cycle, results of previous attempts are reused.

Posterior sampling Across cycles, samples from the previous posterior provide useful initializations for sampling from the updated posterior. This favors running one chain per retained sample over thinning a single long chain (cf. Section 3), since each chain can be continued from its previous sample with $\mathbb { D } _ { i + 1 }$ as target, whereas a single thinned chain would carry over only one such starting point. For PMMH, the particle filter is rerun for the current parameter to obtain an updated likelihood estimate, and the proposal and the number of particles are adapted as described in Section $_ { 3 ; }$ for PG, the reference trajectory is extended over the L new time steps by forward simulation. A short burn-in of $K _ { \mathrm { b } } ^ { \prime } \ll K _ { \mathrm { b } }$ iterations then typically suffices, and a full burn-in is needed only in the initial cycle. Within a cycle, the posterior is unchanged, but each further attempt requires a new validation set that is independent of the previous ones (Assumption 3). Since the number of attempts typically lies below $n _ { \mathrm { m a x } } ,$ these sets are generated on demand by advancing the $K _ { \mathrm { v a l } }$ validation chains for another $K _ { \mathrm { b } } ^ { \prime }$ iterations whenever an attempt does not terminate the procedure. This amounts to thinning across attempts, with $\mathbf { \dot { K } _ { b } ^ { \prime } }$ chosen large enough that successive validation sets can be treated as independent. Apart from synchronization for adapting the PMMH proposal and number of particles, all $K _ { \mathrm { t r a i n } } + K _ { \mathrm { v a l } }$ chains evolve independently and can be run in parallel.

Remark 10 (Relation to sequential inference) Sequential inference methods process new measurements incrementally and may thus appear natural for the online posterior update, but they do not provide independent posterior samples in the sense of Assumption 2. Sequential Monte Carlo methods for joint state–parameter inference, such as $S M C ^ { 2 } ~ I { \boldsymbol { \& } } J ;$ , represent the posterior by a weighted particle population whose members are mutually dependent through resampling. Forward-filtering approaches [5] additionally never regenerate past states in light of later observations, so that path degeneracy causes their approximation of the parameter posterior to deteriorate as the dataset grows.

Synthesis A good initialization substantially reduces the time required to solve the scenario OCP (12). Between control cycles, the previously returned input sequence is shifted forward by L steps, padded, e.g., by repeating its final element, and used to initialize the optimization method. If state and output trajectories are also treated as decision variables, they are initialized by forward simulation of the shifted input sequence, since they depend on the newly sampled scenarios. The calibration is initialized with the risk weight of the previously returned candidate as $\omega _ { 1 } ,$ clipped below one to keep λ finite, and the most recent slope estimate as s<sub>init</sub>, so that the first attempt typically lands close to the acceptance band (19) and fewer attempts are needed. Within a cycle, consecutive attempts differ only in the risk weight, so the previous solution serves as the initial guess. Because the certificate holds for any candidate, these warm starts affect only the computational effort, not the per-cycle guarantees.

![](images/5a413750fdf40c070bfb4697df601e19f63451466b830db3382bcb125d64c48d.jpg)  
Fig. 2. Schematic of the vertical farm. The inputs (temperature, drought level, lighting, and CO<sub>2</sub> concentration) act on the crop, whose state ${ \bf { \sigma } } _ { { \bf { { x } } } _ { t } }$ evolves according to the SIMPLE model with uncertain parameters θ. Cameras provide noisy measurements y<sub>t</sub> of the biomass, while the remaining states are latent (AI-assisted, Claude Opus 5.5).

## 7 Numerical Example: Optimal Control of a Vertical Farm

In this section, we demonstrate the proposed framework in simulation on the optimal control of a vertical farm, where crops are grown in stacked layers inside closed buildings and temperature, irrigation, lighting, and $\mathrm { C O } _ { 2 }$ concentration can be regulated independently of the weather (see Figure 2). Since maintaining this controlled environment is costly, adapting it to the changing needs of the crop through optimal control is particularly promising [11]. We consider a problem in which these inputs are set on a daily basis to steer crop growth over a fixed cultivation period, with the goal of achieving a target yield at minimum resource cost. Meeting this target is economically important, as production may be tied to delivery commitments made before the cultivation cycle begins. Falling short of the target can lead to direct financial losses, whereas using more resources than necessary reduces the profit margin. This problem is a well-suited test case, since crop growth is described by nonlinear parametric state-space models derived from plant physiology, but the model parameters vary across cultivars and even individual plants, while only some of the physiological states can be measured, and only with considerable noise. Reaching the target yield with a prescribed probability therefore requires input plans that rigorously account for the resulting uncertainty about future crop growth. The full model specification, parameters, settings, preliminary analyses, and code to reproduce the experiments are available at https://github.com/ TUM-ITR/PosteriorScenarioPlanning.jl.

## 7.1 Crop Model and Prior

We consider a vertical farm cultivating tomato, whose growth is described by the SIMPLE crop model [35] in the discrete-time state-space formulation of [11] with a sampling time of one day, which matches the slow growth dynamics. The model captures the nonlinear response of crop growth to the environmental conditions and has previously been employed for optimal control in vertical farming. The state

$$
\mathbf { \Delta } \mathbf { x } _ { t } = \left[ m _ { \mathrm { B } , t } \mathrm { ~ } T _ { \mathrm { c } , t } \mathrm { ~ } I _ { 5 0 \mathrm { B } , t } \right] ^ { \top }\tag{34}
$$

contains the above-ground biomass $m _ { \mathrm { B } , t }$ in kg m $^ { - 2 } ,$ , the cumulative temperature $T _ { \mathrm { c } , t } ,$ which governs the developmental stage of the crop, and the senescence variable $I _ { \mathrm { 5 0 B } , t } ,$ which governs the onset of senescence. The input

$$
\mathbf { \boldsymbol { \mathsf { u } } } _ { t } = \left[ T _ { t } ~ D _ { t } ~ R _ { t } ~ C _ { t } \right] ^ { \top }\tag{35}
$$

comprises the daily mean air temperature $T _ { t }$ , the drought level $D _ { t } \in [ 0 , 1 ]$ quantifying the relative reduction in water supply, the daily radiation $R _ { t } ,$ and the $\mathrm { C O } _ { 2 }$ concentration $C _ { t }$ . Since the inputs affect not only the current growth rate but also the timing of senescence and thus the future growth potential, planning over the entire cultivation period is required. The dynamics are augmented with additive Gaussian process noise $\mathbf { } v _ { t } \sim \mathcal { N } ( \mathbf { 0 } , Q )$ , where $Q$ is diagonal with standard deviations of approximately 5 % of the average daily state increments under nominal operating conditions. The states are clipped at zero to preserve nonnegativity. Only the biomass can be measured, for example using image-based methods [19], whereas $T _ { \mathrm { c } , t }$ and $I _ { \mathrm { 5 0 B } , t }$ remain latent. The measurement model is

$$
y _ { t } = \left[ 1 \mathrm { ~ 0 ~ } 0 \right] x _ { t } + w _ { t } ,\tag{36}
$$

where $w _ { t } \sim \mathcal { N } ( 0 , \sigma _ { w } ^ { 2 } )$ , with $\sigma _ { w }$ set to approximately $5 \%$ of the biomass after 50 days under nominal operating conditions. For simplicity of exposition, both noise covariances are assumed known, although they could be included in θ and estimated jointly.

Parameter values of the SIMPLE model have been published for a range of crops and cultivars [35], but the cultivar at hand may not be among them, and even published values may not transfer exactly due to natural biological variability and differing growing conditions. We therefore model as uncertain the cultivar-specific parameters, which differ between varieties of the same crop, and the species-specific parameters, shared by all varieties of a crop, that most strongly influence the terminal yield according to a local sensitivity analysis. All remaining parameters are fixed at their published tomato values. The resulting parameter vector $\pmb { \theta } ^ { \mathrm { ~ \tiny ~ \in ~ } \mathbb { R } ^ { 6 } }$ governs, among other aspects, the temperature response, the conversion of radiation into biomass, and the response to drought. Since only two tomato cultivars are covered, we construct independent Gaussian priors, truncated at zero, from the sample mean and standard deviation of their values and those of five legume cultivars with similar growth characteristics, which yields a relatively broad prior. For the initial state, the cumulative temperature is zero at sowing, the initial biomass is assigned a lognormal prior with a small mean, and the prior for the initial senescence variable is constructed in the same way as for θ.

## 7.2 Control Objective and Constraints

The goal is to minimize the total production cost until harvest, which takes place $t _ { \mathrm { h } } = 1 0 0$ days after sowing. At planning time $t _ { 0 } .$ , the inputs for the remaining $H = t _ { \mathrm { h } } - t _ { 0 }$ days before harvest are optimized, since the input on the day of harvest no longer affects the yield. The objective function comprises the costs of heating, irrigation, lighting, and CO enrichment, which are linear in the inputs and based on representative prices, together with a small quadratic regularization around a nominal operating point u¯ to improve numerical conditioning. The inputs are subject to box constraints representing the operating limits of the farm, with temperature and $\mathrm { C O } _ { 2 }$ concentration bounded below by their ambient values.

At harvest, a minimum yield of $Y _ { \mathrm { m i n } } = 8 \mathrm { k g m ^ { - 2 } }$ must be achieved. Since the resources needed to reach this target are uncertain, the input plan must trade off production cost against the risk of missing it. As only a fraction $H I = 0 . 6 8$ of the above-ground biomass is harvestable, the target corresponds to a minimum terminal biomass of $1 1 . 8 \mathrm { { k g } m ^ { - 2 } }$ and the requirement is expressed as

$$
\mathbb { P } \left[ h ( \pmb { x } _ { t _ { \mathrm { h } } } ) = \frac { Y _ { \mathrm { m i n } } } { H I } - m _ { \mathrm { B } , t _ { \mathrm { h } } } \leq 0 \right] \geq 1 - \alpha ,\tag{37}
$$

where α specifies the risk of a yield shortfall that the operator is willing to accept. For this case study, we set $\alpha = 5 \%$

## 7.3 Settings and Baselines

To assess the probabilistic guarantee empirically, we perform a Monte Carlo study with 200 runs. In each run, the true parameters and initial state of the crop are drawn from the prior, and the farm is operated during the first $t _ { 0 } = 7 5$ days with inputs that vary randomly around the nominal operating point u¯. These inputs and the resulting biomass measurements form the dataset D.

Posterior samples are generated by PMMH with a bootstrap particle filter, and the proposal and the number of particles are adapted as described in Section 3. Each of the $K _ { \mathrm { t r a i n } } = 4 0 0$ training and $K _ { \mathrm { v a l } } = 5 0 0 0$ validation scenarios is obtained from a separate chain after burn-in, and fresh validation sets are generated on demand as described in Section 6. The calibration uses $\tau = \alpha , \beta = 1 0 ^ { - 3 } , n _ { \mathrm { m a x } } = 8$ $n _ { \mathrm { r e c } } = 3 .$ , and $\alpha _ { \mathrm { t o l } } = 1 . 5 \%$ , while ω and $s _ { \mathrm { i n i t } }$ are chosen based on a preliminary study, which also confirms that the power law (14) is a good local approximation. The scenario OCPs are implemented in JuMP [20] and solved with IPOPT [33] using the HSL linear solvers [15]. Since IPOPT requires twice continuously differentiable problem functions, all maximum and minimum operators in the crop model, including the clipping of the states at zero, are replaced by smooth approximations.

We compare the proposed framework with two baselines and a hindsight oracle, all evaluated on the same realizations of the true parameters, initial states, and noise sequences. The certainty-equivalent (CE) planner uses the same data as the proposed method but ignores the remaining uncertainty. It predicts crop growth deterministically without process noise, based on the posterior mean of the parameters, estimated from the PMMH samples, and a particle-filter estimate of the current state conditioned on these parameters. The yield requirement is then enforced on this prediction. The priorbased planner, in contrast, accounts for the uncertainty but ignores the measurements. It applies the same synthesis– validation procedure to scenarios drawn from the prior and propagated to the planning time under the applied inputs, which yields an analogous guarantee with respect to the prior predictive distribution. Finally, the hindsight oracle knows the true parameters, the current state, and the future noise realizations, and computes the cheapest input sequence that reaches the target, providing a cost reference for each run.

## 7.4 Open-Loop Planning

We first evaluate the framework in an open-loop setting, in which all methods plan at day $t _ { 0 }$ for the remaining $H = 2 5$ days until harvest. Figure 3 shows a representative run. The initial attempt predicts a mean biomass below the target, and the calibration raises the risk weight so that the second attempt certifies that the violation probability is at most 4.44 %. The certified plan reaches the target on the true system. The CE plan predicts that the target is reached but misses it, whereas the prior-based planner fails to certify any candidate, since even its most risk-averse candidate only attains a bound of 8.26 %, above $\alpha = 5 \%$

Table 1 summarizes the results over all 200 runs. In 8 runs, even the oracle with perfect knowledge finds no feasible plan within the input constraints. Since the oracle problem is solved only locally, this does not strictly imply that the target is unattainable. Overall, the proposed method certifies a plan in 155 runs, with a median certified violation bound of 4.0 %. In 90 % of these runs, the bound lies within the acceptance band. None of the 155 certified plans violates the yield constraint on the true system. Based on these independent evaluations alone, the violation probability among certified plans has a 95 % upper confidence bound of $1 . 9 \% ,$ below the prescribed $\alpha = 5 \%$ . In the remaining 45 runs, including all 8 without a feasible oracle plan, no candidate is certified even at maximal risk aversion, presumably because the remaining uncertainty is too large to reach the target with sufficient probability. This is the intended behavior rather than a failure: the method signals before execution that the required robustness cannot be certified, allowing the operator to react, e.g., by adjusting the target, instead of applying a plan with unknown risk. The two baselines illustrate why both learning and uncertainty quantification are required. The prior-based planner, which does not learn from the measurements, certifies no plan, as the prior uncertainty over the parameters and current state is too large. The CE planner, which neglects the remaining uncertainty, misses the target in 119 of 191 runs (62.3 %), far above $\alpha = 5 \%$

Table 1  
![](images/786d4a7b2efe0f5d73650d372bd02243c1ae6312dce83b2ecc34dd17f8a4ca48.jpg)  
Fig. 3. Biomass trajectories for a representative Monte Carlo run. Solid lines show the trajectories realized on the true system, dashed lines the predictions, and the shaded area the 5–95 % interval of the posterior predictive distribution under the certified plan. The dotted line shows the predicted mean of the initial attempt. The triangle marks the terminal biomass required to meet the yield target. The certified plan reaches the target, whereas the certainty-equivalent (CE) plan misses it despite predicting otherwise.

The price of the guarantee is a median cost 24 % higher than that of the oracle, since the certified plan must reach the target with high probability across the possible crop realizations consistent with the data, whereas the oracle plans only for the true one. Across all 200 runs, the synthesis– validation procedure takes on average 108 min on 8 cores of an AMD EPYC 7402 processor, of which 99 min are spent on PMCMC sampling, including the generation of fresh validation sets, and 9 min on solving the scenario OCPs. Since the PMCMC chains can be run largely in parallel, the sampling time could be reduced further by allocating more cores. Even with this modest allocation, the total computation time of less than two hours is well below the one-day interval between successive control inputs.

## 7.5 Shrinking-Horizon Planning

We next evaluate the receding-horizon implementation of Section 6 in the same setting. Starting at $t _ { 0 } = 7 5$ , the plan is updated every $L = 5$ days, resulting in 5 control cycles until harvest. As the yield requirement refers to the fixed harvest date, the scheme is applied in its shrinking-horizon form with $H _ { i } = t _ { \mathrm { h } } - t _ { i }$ . We restrict the evaluation to the

Open-loop results over 200 Monte Carlo runs. Plans are counted if certified (proposed and prior-based) or feasible (CE and oracle). Violations are reported among runs with a plan, and costs are medians of the per-run cost ratios to the oracle over the 155 runs in which the proposed, CE, and oracle planners all find a plan.
<table><tr><td>Method</td><td>Plan found</td><td>Violations</td><td> $J / J _ { \mathrm { o r a c l e } }$ </td></tr><tr><td>Proposed</td><td> $1 5 5 / 2 0 0$ </td><td>0/155</td><td>1.24</td></tr><tr><td>Prior-based</td><td>0/200</td><td></td><td>一</td></tr><tr><td>CE</td><td>191/200</td><td>119/191</td><td>1.00</td></tr><tr><td>Oracle</td><td>192/200</td><td>一</td><td>1</td></tr></table>

![](images/4419f462234f7fc7362a4a91ac8b74297ecd4f31c7c23ba62f2ec58b6210653f.jpg)  
Pred. interval (t = 95)

Fig. 4. Biomass trajectories for the run of Figure 3 under shrinking-horizon planning. Solid lines show the trajectories realized on the true system. The shaded area shows the 5–95 % interval of the posterior predictive biomass under the certified plan at $t = 7 5 ,$ while the error bar shows the corresponding interval of the terminal biomass under the certified plan at $t = 9 5$ . The triangle marks the terminal biomass required to meet the yield target. Replanning brings the terminal biomass closer to the target than the open-loop plan, whereas the certainty-equivalent (CE) planner misses the target despite replanning.

155 runs certified at $t _ { 0 } ,$ as in the remaining runs the method already signals before execution that the required robustness cannot be certified. If certification fails in a later cycle, the candidate with the smallest bound is applied until the next cycle. The CE planner is likewise replanned every L days and, if replanning fails, continues with the remaining portion of its last feasible plan.

Figure 4 shows the same run as Figure 3. As new measurements become available, the predictive interval of the terminal biomass narrows considerably from $t _ { 0 }$ to the final control cycle. The proposed planner exploits the reduced uncertainty to adapt its inputs, so that the realized yield exceeds the target by only 2 %, compared with 14 % for the openloop plan, which must hedge against the greater uncertainty at $t _ { 0 } .$ The CE planner, in contrast, initially plans without any margin for uncertainty. By day 90, the measurements reveal that the crop has fallen behind its predicted growth trajectory, and the planner fails to find an input sequence that reaches the target within the input constraints. Consequently, the target is missed despite replanning.

Shrinking-horizon results over the 155 runs certified at $t _ { 0 } , \mathbf { A }$ plan is counted in each cycle if certified (proposed) or feasible (CE). Violations are reported over all runs, including those in which no new plan is found in a later cycle. Costs are medians of the perrun cost ratios to the oracle over all 155 runs.
<table><tr><td>Method</td><td>Plan in all cycles</td><td>Violations</td><td> $J / J _ { \mathrm { o r a c l e } }$ </td></tr><tr><td>Proposed</td><td>153/155</td><td>0/155</td><td>1.14</td></tr><tr><td>CE</td><td>143/155</td><td>113/155</td><td>1.00</td></tr></table>

Table 2 summarizes the results. Of the 155 runs certified at $t _ { 0 } ,$ all subsequent control cycles are certified in 153 runs, and none of the 155 runs violates the yield constraint. Compared with the open-loop results of Section 7.4, replanning reduces the median cost ratio to the oracle from 1.24 to 1.14. The CE planner, in contrast, still misses the target in 113 of 155 runs despite replanning.

Over the 155 evaluated runs, the first cycle, which coincides with the open-loop planning step, takes on average 89 min (compared with 108 min over all 200 runs including those without a certified plan), whereas subsequent cycles take 73 min, a reduction of 17 %. The mean number of synthesis– validation attempts, each requiring a fresh validation set, decreases by 9 %. Despite the growing dataset, the sampling time decreases by 15 %, from 83 min to 71 min, i.e., by more than the number of attempts, which indicates that warmstarting the PMCMC chains reduces the sampling effort per cycle. The time spent solving the scenario OCPs decreases from about 6 min to about 3 min, reflecting the warm-started optimization but also the shorter planning horizons.

Overall, the case study shows that uncertainty quantification and learning play complementary roles in the proposed framework. Explicitly accounting for the remaining uncertainty is necessary to satisfy the constraints reliably and enables their probabilistic certification. Learning from data, both before planning and through posterior updates during operation, makes such certification attainable without excessive conservatism and reduces the cost of robustness over time. The computational effort remains well below the oneday interval between successive control inputs.

## 8 Conclusion

This article presents a Bayesian framework for the optimal control of nonlinear systems with unknown dynamics from input–output data, providing probabilistic guarantees for constraint satisfaction. The framework combines PMCMC-based posterior inference with a validation procedure that certifies constraint satisfaction and a CVaRpenalized control synthesis whose risk weight is calibrated until the desired robustness level is certified. In a recedinghorizon setting, the posterior is updated as new measurements become available, allowing the controller to progressively reduce uncertainty and improve performance. A case study on the optimal control of a vertical farm confirms the guarantees empirically and illustrates how learning from data and replanning reduce the cost of robustness. Future work includes extending the framework to continuous-time models with infrequent measurements, building on MCMC methods with embedded numerical integration [16], and synthesizing output-feedback policies, whose certification is already supported by the proposed validation procedure. The synthesis of such policies is particularly challenging because control decisions must rely on noisy outputs rather than direct access to the latent state.

## Declaration of Generative AI and AI-Assisted Technologies in the Manuscript Preparation Process

During the preparation of this work the authors used Claude (Anthropic) and ChatGPT (OpenAI), in various versions between 2025 and 2026, to improve the wording and structure of the manuscript, to discuss technical ideas, to assist in programming the simulation, and to assist in creating Figure 2. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] Teodoro Alamo, Roberto Tempo, Amalia Luque, and Daniel R Ramirez. Randomized methods for design of uncertain systems: Sample complexity and sequential algorithms. Automatica, 52:160– 172, 2015.

[2] Christophe Andrieu, Arnaud Doucet, and Roman Holenstein. Particle Markov chain Monte Carlo methods. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72(3):269–342, 2010.

[3] Christophe Andrieu and Johannes Thoms. A tutorial on adaptive MCMC. Statistics and Computing, 18(4):343–373, 2008.

[4] Julian Berberich and Frank Allgower. An overview of systems-¨ theoretic guarantees in data-driven model predictive control. Annual Review of Control, Robotics, and Autonomous Systems, 8(1):77–100, 2025.

[5] Karl Berntorp. Online Bayesian inference and learning of Gaussianprocess state–space models. Automatica, 129:109613, 2021.

[6] Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

[7] Marco C Campi, Algo Care, and Simone Garatti. The scenario\` approach: A tool at the service of data-driven decision making. Annual Reviews in Control, 52:1–17, 2021.

[8] Nicolas Chopin, Pierre E Jacob, and Omiros Papaspiliopoulos. SMC<sup>2</sup>: an efficient algorithm for sequential analysis of state space models. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(3):397–426, 2013.

[9] Charles J Clopper and Egon S Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4):404–413, 1934.

[10] Jarrad Courts, Adrian G Wills, Thomas B Schon, and Brett Ninness.¨ Variational system identification for nonlinear state-space models. Automatica, 147:110687, 2023.

[11] Annalena Daniels, Michael Fink, Marion Leibold, Dirk Wollherr, and Senthold Asseng. Optimal control for indoor vertical farms based on crop growth. IFAC-PapersOnLine, 56(2):9887–9893, 2023.

[12] Arnaud Doucet, Michael K Pitt, George Deligiannidis, and Robert Kohn. Efficient implementation of Markov chain Monte Carlo when using an unbiased likelihood estimator. Biometrika, 102(2):295–313, 2015.

[13] Roger Frigola, Fredrik Lindsten, Thomas B Schon, and Carl E¨ Rasmussen. Bayesian inference and learning in Gaussian process state-space models with particle MCMC. In Advances in Neural Information Processing Systems, volume 26, 2013.

[14] Yosef Hochberg and Ajit C Tamhane. Multiple Comparison Procedures. John Wiley & Sons, Inc., 1987.

[15] HSL. A collection of Fortran codes for large scale scientific computation. http://www.hsl.rl.ac.uk/, 2026.

[16] Robert Lefringhausen, Theodor Springer, and Sandra Hirche. Learning dynamics from infrequent output measurements for uncertainty-aware optimal control. arXiv preprint arXiv:2512.08013, 2025.

[17] Robert Lefringhausen, Supitsana Srithasan, Armin Lederer, and Sandra Hirche. Learning-based optimal control with performance guarantees for unknown systems with latent states. In 2024 European Control Conference (ECC), pages 90–97. IEEE, 2024.

[18] Fredrik Lindsten, Michael I Jordan, and Thomas B Schon. Particle¨ Gibbs with ancestor sampling. Journal of Machine Learning Research, 15(1):2145–2184, 2014.

[19] Muxin Lu, Zhi Yao, Dongyu Wang, Demin Xu, Zhi Wang, Xinyu Gu, Jiahang Shen, Cheng Peng, Tiangang Lu, Jia Guo, et al. Nondestructive monitoring and modeling of greenhouse tomato growth based on multi-sources of RGB, multispectral, and thermal infrared images. Smart Agricultural Technology, page 102171, 2026.

[20] Miles Lubin, Oscar Dowson, Joaquim Dias Garcia, Joey Huchette, Benoˆıt Legat, and Juan Pablo Vielma. JuMP 1.0: Recent improvements to a modeling language for mathematical optimization. Mathematical Programming Computation, 15(3):581–589, 2023.

[21] Michael Maiworm, Daniel Limon, and Rolf Findeisen. Online learning-based model predictive control with Gaussian process models and stability guarantees. International Journal of Robust and Nonlinear Control, 31(18):8785–8812, 2021.

[22] Ali Mesbah. Stochastic model predictive control: An overview and perspectives for future research. IEEE Control Systems Magazine, 36(6):30–44, 2016.

[23] Jorge Nocedal and Stephen J Wright. Numerical Optimization. Springer, second edition, 2006.

[24] R Tyrrell Rockafellar and Stanislav Uryasev. Conditional value-atrisk for general loss distributions. Journal of Banking & Finance, 26(7):1443–1471, 2002.

[25] Anna Scampicchio, Elena Arcari, Amon Lahr, and Melanie N Zeilinger. Gaussian processes for dynamics learning in model predictive control. Annual Reviews in Control, 60:101034, 2025.

[26] Julian D Schiller and Matthias A Muller. Nonlinear moving horizon¨ estimation for robust state and parameter estimation. Automatica, 185:112790, 2026.

[27] Thomas B Schon, Adrian Wills, and Brett Ninness. System¨ identification of nonlinear state-space models. Automatica, 47(1):39– 49, 2011.

[28] Johan Schoukens and Lennart Ljung. Nonlinear system identification: A user-oriented road map. IEEE Control Systems Magazine, 39(6):28–99, 2019.

[29] Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczynski. Lectures on Stochastic Programming: Modeling and Theory. SIAM, 2021.

[30] Andreas Svensson and Thomas B Schon. A flexible state–space¨ model for learning nonlinear dynamical systems. Automatica, 80:189–199, 2017.

[31] Roberto Tempo, Fabrizio Dabbene, and Giuseppe Calafiore. Randomized Algorithms for Analysis and Control of Uncertain Systems. Springer, second edition, 2013.

[32] Mans Thulin. The cost of using exact confidence intervals for a˚ binomial proportion. Electronic Journal of Statistics, 8(1):817–840, 2014.

[33] Andreas Wachter and Lorenz T Biegler. On the implementation of¨ an interior-point filter line-search algorithm for large-scale nonlinear programming. Mathematical Programming, 106:25–57, 2006.

[34] Anna Wigren, Johan Wagberg, Fredrik Lindsten, Adrian G Wills, and˚ Thomas B Schon. Nonlinear system identification: learning while¨ respecting physical models using a sequential Monte Carlo method. IEEE Control Systems Magazine, 42(1):75–102, 2022.

[35] Chuang Zhao, Bing Liu, Liujun Xiao, Gerrit Hoogenboom, Kenneth J Boote, Belay T Kassie, Willingthon Pavan, Vakhtang Shelia, Kwang Soo Kim, Ixchel M Hernandez-Ochoa, et al. A SIMPLE crop model. European Journal of Agronomy, 104:97–106, 2019.