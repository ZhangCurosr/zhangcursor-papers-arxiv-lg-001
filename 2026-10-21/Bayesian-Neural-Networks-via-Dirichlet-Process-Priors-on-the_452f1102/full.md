# Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

Sumit Vashishtha

vash.sumit@gmail.com

Independent Researcher

Reviewed on OpenReview: https: //openreview. net/forum?id=VrWIwB8g3Z

## Abstract

Bayesian neural networks (BNNs) ofer a principled framework for quantifying epistemic uncertainty. However, standard weight-space priors yield complex, high-dimensional neural network posteriors that are computationally prohibitive to sample from, and popular approximate inference schemes often fail to capture epistemic uncertainty efectively—a failure mode particularly acute in reinforcement learning (RL) (Osband et al., 2018). To resolve this challenge, we introduce a new class of BNNs that quantify uncertainty by placing Bayesian nonparametric Dirichlet Process (DP) priors directly on the underlying data-generating distribution rather than on network parameters. Building on this foundation, we develop DP-DQN, a reinforcement learning algorithm that leverages DP-BNNs to maintain posterior distributions over value functions and perform eficient Thompson sampling. By grounding randomized value functions in Bayesian nonparametrics, DP-DQN unifies data-driven posterior reweighting (via the Bayesian Bootstrap) with generative prior hypotheses injected directly through the DP base measure, enabling decisive “deep exploration” with a single neural network without the overhead of multi-network ensembles.

## 1 Introduction

In reinforcement learning (RL), maximizing cumulative return requires navigating the fundamental explorationexploitation dilemma: balancing current best estimates (exploitation) with exploratory actions that resolve epistemic uncertainty. Solving this trade-of eficiently is challenging because delayed rewards require actions to be coordinated across extended multi-step decision sequences—a property known as deep exploration (Osband et al., 2016; 2019a; Oren et al., 2025)—rather than perturbed independently at each individual state.

Thompson Sampling (Thompson, 1933; Kaufmann et al., 2012; Russo et al., 2018; Durand et al., 2018; Riquelme et al., 2018; Maillard, 2019) provides an elegant approach that tackles the exploration-exploitation dilemma by maintaining a posterior distribution over models and selecting actions in proportion to the probability that they are optimal. While these models are conventionally drawn from parametric distribution families (e.g., exponential families), Thompson sampling can in principle be applied to any statistical mode admitting tractable posterior sampling (Riquelme et al., 2018). A natural choice for such statistical models is Bayesian Neural Networks (BNNs), owing to their universal function approximation and rich representation learning capabilities. In contrast to standard neural networks that provide only point estimates, conventional BNNs specify a prior distribution over parameters θ. Given n training observations $z _ { 1 : n } ,$ one infers a posterior distribution over θ, which in turn captures epistemic uncertainty in function predictions.

In deep RL, one can represent value functions via BNNs and drive exploration via Thompson sampling over value-function predictions. However, exact posterior inference over neural network weights is notoriously intractable, prompting extensive research into approximate Bayesian methods. These range from Monte-Carlo (MC) dropout (Gal & Ghahramani, 2016) and variational inference (Blundell et al., 2015) to stochastic mini-batch Markov Chain Monte Carlo (Welling & Teh, 2011). Unfortunately, deep RL algorithms built on these weight-space approximate Bayesian methods struggle on hard-exploration tasks because they fail to capture function-space epistemic uncertainty faithfully (Osband et al., 2018). This deficiency stems largely from the dificulty of designing meaningful priors directly in parameter space (Arbel et al., 2023): while practitioners frequently possess domain knowledge regarding the data-generating process (e.g., transition dynamics and reward scales), mapping such insight into high-dimensional, isotropic parameter priors on over-parameterized neural weight vectors $\theta \in \mathbb { R } ^ { d } \ ( d \gg 1 )$ is rarely straightforward and often causes posterior uncertainty collapse.

To resolve these limitations, we introduce Bayesian Neural Networks via Dirichlet Process priors (DP-BNNs), a paradigm that quantifies epistemic uncertainty by placing a nonparametric DP prior directly on the underlying data-generating distribution rather than on network parameters. Under this formulation, posterior inference over the data distribution decomposes into two complementary components (Ferguson, 1973; Vashishtha & Maillard, 2025; Vashishtha, 2026): an empirical reweighting of observed data (the Bayesian Bootstrap) and an explicit prior base measure. In reinforcement learning, optimizing an action-value network on random measures sampled from this DP posterior induces a principled functional posterior distribution over value functions. Crucially, whereas existing randomized value function approaches require training an ensemble of multiple neural networks to represent functional uncertainty (Osband et al., 2016; 2018), our approach achieves temporally coordinated deep exploration with a single neural network by freezing the value function sampled for each episode across the entire rollout.

We summarize our primary contributions as follows:

• DP-BNN Formulation and Sampling: We introduce Bayesian Neural Networks via Dirichlet Process priors (DP-BNNs), a framework that quantifies epistemic uncertainty by placing a nonparametric DP prior on the data-generating distribution rather than on network weights. Using the Sethuraman stick-breaking construction, we develop a tractable posterior sampling algorithm for deep architectures and validate its epistemic uncertainty quantification on spatial 2D regression.

• Deep Reinforcement Learning via DP-BNNs (DP-DQN): We instantiate DP-BNNs for randomized value functions in deep Q-learning, establishing the DP-DQN algorithm. By optimizing an action-value DP-BNN on random measures sampled from the DP posterior via episodic boundary warmstarts, DP-DQN realizes principled Thompson sampling with exact multi-step temporal consistency using a single neural network.

• Comprehensive Empirical Benchmarking: We evaluate DP-BNNs across diverse decisionmaking benchmarks, demonstrating that they resolve hard exploration in DeepSea across extensive horizons evaluated up to $N = 5 0 ( 2 ^ { 5 0 } \approx \mathrm { i } 0 ^ { 1 5 }$ trajectory search space with empirical $\mathcal { O } ( N ^ { 1 . 4 5 } )$ sample complexity), RiverSwim-6, deceptive N-Chain, continuous MountainCar-v0, and contextual Wheel Bandit, matching or outperforming multi-network ensembles while strictly maintaining a single-network footprint.

The remainder of this paper is organized as follows. Section 2 surveys related work. Section 3 reviews Dirichlet Processes and their stick-breaking construction. Section 4 introduces Bayesian Neural Networks via Dirichlet Process priors (DP-BNNs) and details their posterior sampling algorithm. Section 5 presents DP-DQN, applying DP-BNNs to randomized value functions in deep reinforcement learning, and evaluates its performance on challenging exploration benchmarks. Finally, Section 6 concludes the paper.

## 2 Related Works

Randomized Value Functions and Exploration in Deep RL. Our work connects directly to deep reinforcement learning algorithms that explore via randomized value functions, replacing heuristic ϵ-greedy exploration with principled posterior sampling (Thompson Sampling) (Thompson, 1933; Kaufmann et al., 2012; Ghavamzadeh et al., 2015; Osband et al., 2016; Russo et al., 2018). In deep RL, the most direct approach to Thompson sampling is to maintain a Bayesian neural network over action-values. However, exact posterior inference over network weights is intractable (MacKay, 2003), and mini-batch MCMC schemes scale poorly to online RL training regimes (Welling & Teh, 2011). Consequently, practitioners have relied on approximate inference methods such as Monte-Carlo dropout (Gal & Ghahramani, 2016) and mean-field variational inference (Blundell et al., 2015). Unfortunately, these weight-space approximations struggle to maintain functional uncertainty in unvisited regions, leading to exploration failure in challenging environments (Osband et al., 2018).

Bootstrap Ensembles and Prior Mechanisms. To approximate posterior sampling over function space without exact weight posteriors, Lakshminarayanan et al. (2017) proposed Deep Ensembles, relying on random parameter initializations across independently trained networks. In sequential decision-making, Osband et al. (2016) introduced Bootstrapped DQN, which approximates posterior sampling using an ensemble of K neural networks trained on nonparametrically resampled data (see also Kveton et al. 2019 for bootstrapping exploration in bandits). In sparse-reward environments, however, empirical bootstrapping alone provides limited guidance in regions devoid of data: because all ensemble heads are trained on unrewarded transitions, their value estimates tend to align across unvisited states. To address this and sustain directed exploration, Osband et al. (2018) introduced Bootstrapped DQN with Randomized Prior Functions (BootDQN-RP), which augments each trainable head $f _ { \theta _ { k } }$ with a fixed, randomly initialized prior network $p _ { k }$ . While highly efective in practice, this formulation substantially increases computational and memory demands, as it requires training and maintaining an ensemble of K distinct neural networks alongside their corresponding prior networks. In another approach, Bayesian Deep Q-Networks (BDQN) (Azizzadenesheli et al., 2018) applies Bayesian linear regression to the final layer representations; however, its capability on deep-exploration benchmarks has remained largely unexplored in the literature, and our empirical evaluations on DeepSea show that BDQN does not resolve the task. Theoretically, restricting Bayesian inference strictly to the final linear readout cannot account for epistemic uncertainty in the underlying feature representations on unvisited states (see Appendix D.3 for a detailed theoretical analysis).

Our Approach: Dirichlet Process Priors on Data Distributions. Rather than placing priors on network parameters (Blundell et al., 2015) or relying on multi-network ensembles with additive priors (Osband et al., 2018), our approach formulates epistemic uncertainty nonparametrically on the data-generating distribution. Building upon DP posterior sampling techniques developed in Vashishtha & Maillard (2025) and Vashishtha (2026), DP-BNNs connect Rubin’s classical Bayesian Bootstrap (Rubin, 1981)—recovered in the non-informative limit α → 0—with an informative prior base measure $F _ { 0 }$ . Crucially, whereas empirical bootstrapping can only resample observed data, the base measure populates unvisited regions of the domain, sustaining non-vanishing epistemic uncertainty where experience is absent. This enables directed exploration with a single neural network while avoiding the parameter and memory footprint of multi-network ensembles.

## 3 Background on Dirichlet Processes

This section reviews essential foundational concepts regarding Dirichlet Processes, conjugate posteriors, the Bayesian Bootstrap, and stick-breaking representations that underpin our DP-BNN framework; for comprehensive treatises on Bayesian nonparametric statistics, see Ghosal & van der Vaart (2017) and Castillo (2024).

Dirichlet distribution. The Dirichlet distribution is a multivariate generalization of the Beta distribution. We denote the Dirichlet distribution parameterized by $\left( \alpha _ { 1 } , \ldots , \alpha _ { n } \right)$ on the probability simplex $\Delta ^ { n - 1 } = \{ \mathbf { w } \in$ $\begin{array} { r } { [ 0 , 1 ] ^ { n } : \sum _ { i = 1 } ^ { n } w _ { i } = 1 \} } \end{array}$ as $\mathrm { D i r } ( \alpha _ { 1 } , \ldots , \alpha _ { n } )$ , with density $\begin{array} { r } { { p } ( \mathbf { w } ) \propto \prod _ { i = 1 } ^ { \dot { n } } w _ { i } ^ { \alpha _ { i } - 1 } } \end{array}$

Dirichlet Processes. In the Bayesian formalism, an unknown object of interest is treated as a random variable endowed with a prior distribution. When the unknown object is itself a probability measure on a measurable space $( \Theta , B ( \Theta ) ) \ ( \mathrm { e . g . , \ a }$ Polish space endowed with its Borel σ-algebra), defining a prior over infinite-dimensional distributions while satisfying unit-measure normalization is challenging. Ferguson (1973) introduced the Dirichlet Process (DP)—a distribution over probability measures that induces finitedimensional Dirichlet distributions on partitions. Concretely, a random probability measure G is Dirichlet Process distributed with base probability measure $F _ { 0 }$ and concentration parameter $\alpha \in \mathbb { R } ^ { + }$ , denoted

$G \sim \mathrm { D P } ( \alpha , F _ { 0 } )$ , if for every finite measurable partition $( A _ { 1 } , \ldots , A _ { r } )$ of Θ:

$$
( G ( A _ { 1 } ) , \dots , G ( A _ { r } ) ) \sim \operatorname { D i r } ( \alpha F _ { 0 } ( A _ { 1 } ) , \dots , \alpha F _ { 0 } ( A _ { r } ) ) .\tag{1}
$$

Conjugacy and Posteriors. A particularly useful property of the DP is its conjugacy: if observations $X _ { 1 } , \ldots , X _ { n } \mid G \stackrel { \mathrm { i . i . d . } } { \sim } G$ are drawn from an unknown distribution with prio $G \sim \mathrm { D P } ( \alpha , F _ { 0 } )$ , the posterior distribution given $\mathbf { X } _ { n } = \left( X _ { 1 } , \ldots , X _ { n } \right)$ is itself a DP (Ferguson, 1973):

$$
{ \cal G } \mid { \bf X } _ { n } \sim \mathrm { D P } ( \alpha _ { n } , \overline { { F } } _ { n } ) , \quad \mathrm { w h e r e } \quad \alpha _ { n } = \alpha + n , \quad \overline { { F } } _ { n } = \frac { n } { \alpha + n } F _ { n } + \frac { \alpha } { \alpha + n } F _ { 0 } ,\tag{2}
$$

where $\begin{array} { r } { F _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } } \end{array}$ denotes the empirical measure of the observations (Ghosal, 2010).

Notice how the posterior base measure ${ \overline { { F } } } _ { n }$ in Equation (2) naturally balances empirical evidence $F _ { n }$ with prior structure $F _ { 0 }$ . As sample size $n \to \infty$ , the empirical data dominates; conversely, when data is scarce $( n  0 )$ the prior $F _ { 0 }$ governs. In our setting, $F _ { 0 }$ serves as the foundational mechanism through which prior inductive biases enter the posterior; more broadly, recent advances in Bayesian nonparametrics demonstrate that random probability measures can also be elicited directly from prior information on meaningful functionals (Gafi et al., 2025).

Bayesian Bootstrap. The Bayesian Bootstrap (Rubin, 1981) provides a Bayesian analogue to Efron’s classical bootstrap (Efron & Tibshirani, 1994), and has been leveraged for randomized exploration in multi armed bandits (Kveton et al., 2019). Crucially, the Bayesian Bootstrap arises as the non-informative limiting case of a DP posterior as the prior concentration vanishes, $\alpha  0$ (Ghosal & van der Vaart, 2017):

$$
\mathrm { B B } _ { n } : = \operatorname* { l i m } _ { \alpha \to 0 } \mathrm { D P } \left( \alpha + n , \frac { n } { \alpha + n } F _ { n } + \frac { \alpha } { \alpha + n } F _ { 0 } \right) = \sum _ { i = 1 } ^ { n } W _ { i } \delta _ { X _ { i } } ,\tag{3}
$$

where $\mathbf { W } _ { n } = ( W _ { 1 } , \ldots , W _ { n } ) \sim \mathrm { D i r } ( 1 , \ldots , 1 )$ and $\delta _ { X _ { i } }$ denotes a Dirac measure at observation $X _ { i }$ . The posterior mean of any linear functional under the Bayesian Bootstrap evaluates simply as $\begin{array} { r } { \mu ( \mathrm { B B } _ { n } ) = \sum _ { i = 1 } ^ { n } W _ { i } X _ { i } = } \end{array}$ $\langle \mathbf { W } _ { n } , \mathbf { X } _ { n } \rangle$ ; for a comprehensive survey on the distributional properties of means and linear functionals of random probability measures, see Lijoi & Prünster (2009).

Stick-Breaking Construction. While Equation (2) specifies the posterior distribution of $G ,$ practical computation requires a constructive method to draw sample measures from a DP. Sethuraman (1994) established that almost surely discrete random probability measures from $\mathrm { D P } ( \alpha , F _ { 0 } )$ admit an explicit stick-breaking representation:

$$
Q ( \cdot ) = \sum _ { i = 1 } ^ { \infty } q _ { i } \delta _ { Z _ { i } } ( \cdot ) , \quad Z _ { i } \overset { \mathrm { i i d } } { \sim } F _ { 0 } ,\tag{4}
$$

where weights $q _ { i }$ are generated by successively breaking pieces from a unit stick:

$$
q _ { 1 } = V _ { 1 } , \quad q _ { i } = V _ { i } \prod _ { j = 1 } ^ { i - 1 } ( 1 - V _ { j } ) \quad ( i \geq 2 ) ,\tag{5}
$$

$$
V _ { i } \stackrel { \mathrm { i i d } } { \sim } \mathrm { B e t a } ( 1 , \alpha ) , \quad Z _ { i } \stackrel { \mathrm { i i d } } { \sim } F _ { 0 } .\tag{6}
$$

In practice, the infinite series is truncated at a finite level $T$ by setting $V _ { T } = 1$ , which ensures the remaining stick length $\begin{array} { r } { q _ { T } = \prod _ { i = 1 } ^ { T - 1 } ( 1 - V _ { j } ) = 1 - \sum _ { i = 1 } ^ { T - 1 } } \end{array}$ q<sub>i</sub> is fully assigned to $Z _ { T }$ , guaranteeing $\textstyle \sum _ { i = 1 } ^ { T } q _ { i } = 1$ (Ishwaran & James, 2001; Muliere & Tardella, 1998). Truncation bounds and approximation guarantees are detailed in Appendix A.

Iterative Sampling from the DP Posterior. To sample an explicit realization from the posterior $\mathrm { D P } ( \alpha _ { n } , \overline { { F } } _ { n } )$ , we leverage the sequential stick-breaking recursion of Dirichlet Processes. Conditioning a DP on a single observation $X _ { i }$ satisfies the recursive distributional relation (Ferguson, 1973; Ghosal & van der Vaart, 2017):

$$
\begin{array} { r } { Q _ { i } ( \cdot ) \stackrel { d } { = } V _ { i } \delta _ { X _ { i } } + ( 1 - V _ { i } ) Q _ { i - 1 } ( \cdot ) , \quad V _ { i } \sim \mathrm { B e t a } ( 1 , \alpha + i ) , } \end{array}\tag{7}
$$

where $\circeq$ denotes equality in distribution. Building upon this relation, Vashishtha & Maillard (2025) derived an explicit multi-step stick-breaking representation for sampling realized probability measures directly from the $\mathrm { D P }$ posterior: initializing with a prior draw $Q _ { 0 } \sim \mathrm { D P } ( \alpha , F _ { 0 } )$ via stick-breaking and unrolling recursion (7) across all n observations yields

$$
Q _ { n } \stackrel { d } { = } V _ { n } \delta _ { X _ { n } } + \sum _ { i = 1 } ^ { n - 1 } \left[ V _ { i } \prod _ { j = i + 1 } ^ { n } ( 1 - V _ { j } ) \right] \delta _ { X _ { i } } + \left[ \prod _ { i = 1 } ^ { n } ( 1 - V _ { i } ) \right] Q _ { 0 } .\tag{8}
$$

The scalar coeficient $\begin{array} { r } { W _ { \mathrm { p r i o r } } \triangleq \prod _ { i = 1 } ^ { n } ( 1 - V _ { i } ) } \end{array}$ represents the total posterior probability mass retained by the prior measure $Q _ { 0 }$ . Under the sequential Beta stick-breaking construction where $1 - V _ { i } \sim \mathrm { B e t a } ( \alpha + i , 1 )$ ), this remaining prior weight follows the exact closed-form distribution:

$$
W _ { \mathrm { p r i o r } } = \prod _ { i = 1 } ^ { n } ( 1 - V _ { i } ) \sim \mathrm { B e t a } ( \alpha + 1 , n ) ,\tag{9}
$$

with expected prior mass $\begin{array} { r } { \mathbb { E } [ W _ { \mathrm { p r i o r } } ] = \frac { \alpha + 1 } { \alpha + 1 + n } } \end{array}$ . This term cleanly governs the balance between empirical evidence and prior belief: when data is absent $( n = 0 ) , W _ { \mathrm { p r i o r } } = 1$ and the posterior recovers the prior $Q _ { 0 } ;$ as observations accumulate $( n  \infty )$ $W _ { \mathrm { p r i o r } }  0$ and empirical data dominates.

## 4 Bayesian Neural Networks via DP Priors

In this section, we present a nonparametric methodology for sampling neural-network models from their posteriors to capture epistemic uncertainty in function predictions. The core idea is that predictive uncertainty originates from incomplete knowledge of the true data-generating distribution. Consider a neural network parameterized by $\theta$ trained under loss $\mathcal { L } _ { \boldsymbol { \theta } } ( \cdot )$ . If the true data-generating distribution $G ^ { * } ( \cdot )$ were known (generating observations $\{ z _ { i } = ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , with inputs $x _ { i }$ and targets $y _ { i } )$ , optimal parameters $\theta ^ { * }$ would be obtained by minimizing the expected risk:

$$
\theta ^ { * } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { z \sim G ^ { * } } [ \mathcal { L } _ { \theta } ( z ) ] = \arg \operatorname* { m i n } _ { \theta } \int \mathcal { L } _ { \theta } ( z ) d G ^ { * } ( z ) .\tag{10}
$$

In practice, however, the learner only observes a finite training set, leaving $G ^ { * }$ unknown. Epistemic uncertainty over $\theta ,$ and consequently over network predictions, is therefore a direct consequence of uncertainty over $G ^ { * }$

To quantify this uncertainty, we place a Dirichlet Process prior directly on the unknown data distribution: $G \sim \mathrm { D P } ( \alpha , F _ { 0 } )$ , where the base measure $F _ { 0 }$ encapsulates prior domain structure and support (Lemma 1). As training observations accumulate $( n  \infty )$ , posterior consistency ensures that the posterior over $G$ concentrates weakly around the true distribution $G ^ { * }$ (Ferguson, 1973; Ghosal, 2010; Castillo, 2024). In finite-sample regimes, however—and particularly in regions where data is sparse or absent—the posterior does not collapse to an overconfident point estimate; instead, it retains non-vanishing epistemic variance governed by $F _ { 0 }$ . Whereas weight-space variational BNNs frequently underestimate uncertainty in voids or between separated data clusters (Foong et al., 2019), placing the nonparametric prior directly in data space preserves functional uncertainty wherever data has not yet been observed.

Given a finite training dataset $\mathcal { D } _ { n } = \{ z _ { i } \} _ { i = 1 } ^ { n }$ , the posterior over the data-generating distribution is $G ^ { * } ~ |$ ${ \mathcal { D } } _ { n } \sim \mathrm { D P } ( \alpha _ { n } , { \overline { { F } } } _ { n } )$ with posterior indices defined in Equation (2). Applying the stick-breaking construction (Equations $\left( 5 \right) - \left( 6 \right) )$ and the iterative posterior recursion (Equation (8)) (Vashishtha & Maillard, 2025), any random posterior measure $P _ { j } \sim \mathrm { D P } ( \alpha _ { n } , \overline { { F } } _ { n } )$ truncated at T prior atoms takes the explicit finite discrete form:

$$
P _ { j } ( \cdot ) = \sum _ { i = 1 } ^ { n + T } p _ { i } ^ { \prime } \delta _ { z _ { i } } ( \cdot ) ,\tag{11}
$$

![](images/1a50205bf6d1d75bbe984cd11db20b9b6ec6f73bc7245ce2b86d991f11798faf.jpg)

![](images/f7f4fa383f39c581f021c56f01a2dc7a191dc251ef074a44851d21e5a68f1cf6.jpg)  
Figure 1: Epistemic uncertainty on 2D annular regression. (A) Annular training data $( r \in [ 1 . 3 , 2 . 4 5 ] )$ with unobserved central cavity $( r \leq 1 . 0 )$ . (B–D) BALD epistemic uncertainty heatmaps (nats) (Houlsby et al., 2011) on a shared [0.0, 1.8] scale for Deep Ensembles (B), BootDQN with randomized priors (C), and DP-BNN (D). (E) Void vs. data support BALD contrast across evaluated methods.

```latex
Algorithm 1 Posterior Sampling of Bayesian Neural Networks via Dirichlet Process Priors (DP-BNN)
1: Input: Training observations $\mathcal { D } _ { n } = \{ z _ { i } \} _ { i = 1 } ^ { n }$ , DP prior parameters $( \alpha , F _ { 0 } )$ , truncation level T, sample
count J, loss function $\mathcal { L } _ { \boldsymbol { \theta } } ( \cdot )$
2: Output: Posterior model samples $\{ \theta ^ { ( j ) } \} _ { j = 1 } ^ { J }$
3: for $j = 1 , \dots , J$ do
4: Sample synthetic prior pseudo-observations: $\{ \underline { { z } } _ { n + t } \} _ { t = 1 } ^ { T } \stackrel { \mathrm { i . i . d . } } { \sim } F _ { 0 }$
5: Sample stick-breaking weights $\{ p _ { i } ^ { \prime } \} _ { i = 1 } ^ { n + T } \in \Delta ^ { n + T - 1 }$ via Equations (11) and (8)
6: Form random discrete posterior measure: $\begin{array} { r } { P _ { j } = \sum _ { i = 1 } ^ { n + T } p _ { i } ^ { \prime } \delta _ { z _ { i } } } \end{array}$
7: Initialize network parameters $\theta ^ { ( j ) }$
8: Optimize parameters to convergence: $\theta ^ { ( j ) }  \mathrm { a r g }$ min<sub>θ</sub> $\begin{array} { r } { \sum _ { i = 1 } ^ { n + T } p _ { i } ^ { \prime } \mathcal L _ { \theta } ( z _ { i } ) } \end{array}$ via Eq. (12)
9: end for
10: return $\{ \theta ^ { ( j ) } \} _ { j = 1 } ^ { J }$
```

where $\{ z _ { i } \} _ { i = 1 } ^ { n }$ are the observed training points and $\{ z _ { n + t } \} _ { t = 1 } ^ { T } \stackrel { \mathrm { i i d } } { \sim } F _ { 0 }$ are prior pseudo-observations sampled from the base measure. The weights $\{ p _ { i } ^ { \prime } \} _ { i = 1 } ^ { n + T }$ satisfy $\begin{array} { r } { \sum _ { i = 1 } ^ { n + T } p _ { i } ^ { \prime } = 1 } \end{array}$ , where the empirical weights $\{ p _ { i } ^ { \prime } \} _ { i = 1 } ^ { n }$ are obtained from Equation (8) and the prior weights $\{ p _ { n + t } ^ { \prime } \} _ { t = 1 } ^ { T }$ are generated by scaling stick-breaking weights from Equation (5) by the remaining prior factor $\begin{array} { r } { W _ { \mathrm { p r i o r } } = \dot { \prod } _ { k = 1 } ^ { n } ( \bar { 1 } - V _ { k } ) \ ( \mathrm { E q . ~ } ( 9 ) ) } \end{array}$

Substituting this discrete measure into the risk functional in Equation (10), sampling a neural-network model $\theta ^ { ( j ) }$ from the posterior reduces directly to a weighted empirical loss:

$$
\theta ^ { ( j ) } = \arg \operatorname* { m i n } _ { \theta } \sum _ { i = 1 } ^ { n + T } p _ { i } ^ { \prime } \mathcal { L } _ { \theta } ( z _ { i } ) .\tag{12}
$$

The complete procedure is summarized in Algorithm 1. In order to compute variations in predictions and confidence intervals, one computes predictions across the sampled models $\{ \theta ^ { ( j ) } \} _ { j = 1 } ^ { J }$ . Notice that as $\alpha  0 .$ $P _ { j }$ recovers the Bayesian Bootstrap (Equation (3)), placing probability mass solely on observed data points $\{ \bar { z } _ { i } \} _ { i = 1 } ^ { n }$ . When $\alpha > 0$ , the DP base measure $F _ { 0 }$ incorporates prior domain structure and ensures non-vanishing epistemic uncertainty in unexplored regions of the input space.

We evaluate the epistemic uncertainty of DP-BNNs on a 2D regression benchmark over an annular manifold where 2D spatial inputs $x = ( x _ { 1 } , x _ { 2 } ) \in [ - 2 . 5 , 2 . 5 ] ^ { 2 }$ have radial distance $r = \| x \| _ { 2 } \in [ 1 . 3 , 2 . 4 5 ]$ with noisy scalar target observations $y = \cos ( 1 . 5 \pi r ) + \epsilon \in [ - 1 , 1 ] \ ( \sigma _ { \mathrm { a l e a t o r i c } } = 0 . 3 0 )$ , leaving an unobserved central cavity void for $r \leq 1 . 0$ (Figure 1A). Because there are no training observations within the interior cavity, a reliable Bayesian model is fundamentally expected to exhibit high epistemic uncertainty (low predictive confidence) throughout this unobserved void, while contracting with high confidence on the data-rich annular ring. We quantify epistemic uncertainty via Bayesian Active Learning by Disagreement (BALD) (Houlsby et al., 2011) across $J = 1 0$ sampled models (Appendix B.2). Crucially, unlike deep ensembles (Lakshminarayanan et al., 2017) that rely on disparate parameter initializations to generate variance, DP-BNN induces epistemic diversity non-parametrically in function space via synthetic pseudo-observations from a product base measure $F _ { 0 } ( x , y ) = F _ { 0 x } ( x ) \times F _ { 0 y } ( y )$ . Following Lemma 1, we choose the support of $F _ { 0 x }$ as the uniform distribution over the domain $[ - 2 . 5 , 2 . 5 ] ^ { 2 }$ to ensure non-empty coverage of unobserved voids, paired with dispersed prior targets $F _ { 0 y } = \mathcal { N } ( 0 , 1 . 5 ^ { 2 } ) ~ ( T = 5 0$ atoms, concentration parameter $\alpha = 1 0 . 0 )$ . As shown in Figure 1B–E, DP-BNN reflects this expected epistemic behavior, forming an isotropic uncertainty region in the void (1.41 nats vs. 0.18 nats on data support; 8.0× contrast ratio). In comparison, standard baselines exhibit lower contrast between the void and data support: BootDQN with randomized priors (Osband et al., 2018) yields 0.54 vs. 0.14 nats (3.8×), MC Dropout (Gal & Ghahramani, 2016) yields 0.36 vs. 0.16 nats (2.2×), Deep Ensembles (Lakshminarayanan et al., 2017) yields 0.29 vs. 0.06 nats (4.8×), and variational inference (Bayes By Backprop, BBB) (Blundell et al., 2015) produces near-uniform uncertainty across the domain (0.07 vs. 0.09 nats). Because each model $\theta ^ { ( j ) }$ minimizes the loss under an independently drawn measure $P _ { j }$ , the training cost of DP-BNN is on par with standard ensemble baselines (wall-clock training times for $J = 1 0$ models: DP-BNN [5.9 s], BootDQN+priors [5.4 s], Deep Ensembles [6.9 s], BBB [2.1 s], MC Dropout [0.7 s]).

While quantifying predictive variance in this benchmark required training multiple models $( J = 1 0 )$ , Thompson sampling in sequential decision-making requires only a single sampled model per episode (Thompson, 1933; Russo et al., 2018). Because DP-BNNs represent uncertainty via the DP posterior over data distributions, training a single neural network on a draw from this posterior directly provides the required sample. Consequently, an agent can perform Thompson sampling with a single neural network, eliminating the heuristic multi-network ensembles required by previous methods. In the next section, we apply this framework to deep reinforcement learning.

## 5 Reinforcement Learning with DP-BNNs

We now extend the DP-BNN framework to reinforcement learning, developing DP-DQN: a deep Q-learning algorithm that achieves directed, temporally consistent exploration via posterior sampling over action-value functions.

## 5.1 Problem Formulation and Exploration Bottlenecks

Markov Decision Processes. A Markov Decision Process (MDP) (Puterman, 1990) is formalized by the tuple $M = ( S , \mathcal { A } , R , P , \gamma )$ , where S is the state space, A is the discrete action space, $R : S \times \mathcal { A }  \mathbb { R }$ is the reward function, $\textstyle P ( \cdot \mid s , a )$ is the transition probability distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. An agent aims to identify an optimal policy $\pi ^ { * }$ maximizing expected cumulative discounted return, characterized by the unique optimal action-value function $Q ^ { * } ( s , a )$ satisfying the Bellman optimality equation:

$$
Q ^ { * } ( s , a ) = R ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \mid s , a ) } \left[ \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } Q ^ { * } ( s ^ { \prime } , a ^ { \prime } ) \right] .\tag{13}
$$

Deep Q-Learning and Exploration Bottlenecks. In complex or continuous state spaces, tabular representation is infeasible. Value-based deep reinforcement learning methods such as Deep Q-Networks (DQN) and Double DQN (Mnih et al., 2015; van Hasselt et al., 2016; Bellemare et al., 2017) approximate $Q ^ { * } ( s , a )$ with a deep neural network $Q ( s , a ; \theta )$ parameterized by θ. Parameters are optimized by minimizing the mean-squared Temporal Diference (TD) regression loss over transition tuples $z = ( s , a , r , s ^ { \prime } , d )$ sampled from an experience replay bufer B:

$$
\mathcal { L } _ { \mathrm { D Q N } } ( \theta ; \theta ^ { - } ) = \mathbb { E } _ { z \sim B } \left[ \ell _ { \mathrm { T D } } ( z ; \theta , \theta ^ { - } ) \right] , \quad \mathrm { w h e r e } \quad \ell _ { \mathrm { T D } } ( z ; \theta , \theta ^ { - } ) = \left( r + ( 1 - d ) \gamma \operatorname* { m a x } _ { a ^ { \prime } \in A } Q ( s ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) - Q ( s , a ; \theta ) \right) ^ { 2 } ,\tag{14}
$$

where $\theta ^ { - }$ denotes the parameters of a tracking target network. In the framework of statistical learning theory, Eq. (14) is the empirical risk minimizer of the expected TD loss $\mathbb { E } _ { z \sim G ^ { * } } [ \ell _ { \mathrm { T D } } ( z ; \theta , \theta ^ { - } ) ]$ under the empirical measure $\begin{array} { r } { \mathbb { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { z _ { i } } } \end{array}$ of the replay bufer, corresponding to neural Fitted Q-Iteration (Ernst et al., 2005; Riedmiller, 2005; Mnih et al., 2015).

Standard DQN relies on heuristic ϵ-greedy dithering for exploration. However, independent action perturbations at each step lack temporal consistency, requiring an exponential sample complexity $\Omega ( | A | ^ { H } )$ to discover delayed rewards over horizon H (Osband et al., 2018; 2019a). Principled deep exploration instead requires Posterior Sampling for Reinforcement Learning (PSRL) (Dearden et al., 1998; Strens, 2000; Osband et al., 2013; Ghavamzadeh et al., 2015), where an agent samples an action-value hypothesis from an epistemic posterior and acts greedily with respect to it for the entire episode rollout. However, existing deep exploration approaches either maintain an ensemble of K separate neural networks (e.g., Bootstrapped DQN; Osband et al. 2016), which incurs a prohibitive K-fold memory and computational footprint, train parameterized epistemic hypermodels (Dwaracherla et al., 2020), inject heuristic parameter noise (NoisyNet; Fortunato et al. 2018), or rely on parametric weight-space Bayesian approximations (e.g., Bayes By Backprop, MC Dropout) that sufer from prior generalization collapse and fail to sustain epistemic uncertainty across deep horizons (Foong et al., 2019; Osband et al., 2018). DP-BNNs resolve these dual bottlenecks—computational scaling overhead and epistemic uncertainty collapse—by placing a DP prior directly on the data-generating distribution of transition experience G<sup>∗</sup>, inducing rich function-space epistemic diversity nonparametrically with a single neural network (instantiated here as DP-DQN).

## 5.2 DP-DQN: Posterior Sampling for Randomized Value Functions

While standard DQN minimizes empirical TD loss over observed transitions in the replay bufer (Eq. (14)), the DP-BNN framework quantifies epistemic uncertainty by placing a DP prior directly on the underlying transition-experience distribution:

$$
G ^ { * } \sim \mathrm { D P } ( \alpha , F _ { 0 } ) ,\tag{15}
$$

where the base measure $F _ { 0 }$ generates synthetic transition tuples $\tilde { z } = ( \tilde { s } , \tilde { a } , \tilde { r } , \tilde { s } ^ { \prime } , \tilde { d } )$ via factorized marginals $F _ { 0 } ( s , a , r , s ^ { \prime } , d ) = F _ { 0 s } ( s ) \times F _ { 0 a } ( a ) \times F _ { 0 r } ( r ) \times F _ { 0 s ^ { \prime } } ( s ^ { \prime } ) \times F _ { 0 d } ( d )$ , with continuing transitions $( \tilde { d } = 0 )$ . Following Lemma 1, the base measure maintains full topological support over state-action spaces $( \operatorname { s u p p } ( F _ { 0 s } ) \supseteq { \mathcal { S } } $ supp(F<sub>0a</sub>) = A). Because data collection in reinforcement learning is policy-dependent, the reward prior $F _ { 0 r }$ is specified to sustain exploratory variance in unvisited regions (Appendix B).

Given n observed transitions in the replay bufer $\mathcal { B } = \{ z _ { i } = ( s _ { i } , a _ { i } , r _ { i } , s _ { i } ^ { \prime } , d _ { i } ) \} _ { i = 1 } ^ { n }$ , Ferguson’s conjugacy (Ferguson, 1973) yields the posterior distribution over the data-generating distribution:

$$
G ^ { * } \mid \mathcal { B } \sim \mathrm { D P } \left( \alpha + n , \frac { n } { \alpha + n } \mathbb { P } _ { n } + \frac { \alpha } { \alpha + n } F _ { 0 } \right) , \quad \mathrm { w h e r e ~ } \mathbb { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { z _ { i } } .\tag{16}
$$

Under this formulation, posterior sampling over value functions corresponds to drawing a discrete random measure $P _ { k } \ \sim \ G ^ { * } \ | \ B$ for each episode k and optimizing the action-value network under $P _ { k }$ $\theta _ { k } = \arg \operatorname* { m i n } _ { \theta } \mathbb { E } _ { z \sim P _ { k } } [ \ell _ { \mathrm { T D } } ( z ; \theta , \theta ^ { - } ) ]$

Full-Bufer Conditioning and Prior Swamping. However, directly realizing this posterior draw over the full replay bufer B introduces two practical challenges:

1. Computational overhead: Evaluating stick-breaking representations and backpropagating across $| B | \sim 1 0 ^ { 5 }$ stored transitions at every episode boundary is computationally prohibitive.

2. Prior swamping: In sparse-reward exploration, unguided initial rollouts inevitably accumulate thousands of unrewarded transitions. If the sample size n in Eq. (16) scales with the raw bufer size $( n \sim | B | \sim 1 0 ^ { 5 } )$ the relative weight of the prior base measure $\textstyle { \frac { \alpha } { \alpha + n } } \to 0$ rapidly diminishes. The posterior contracts toward empirical resampling over observed transitions—recovering the Bayesian Bootstrap limit (α → 0) (Rubin, 1981)—which diminishes exploratory drive early in training (Figure 2a; Appendix C).

Composite DP-Bellman Loss. To approximate this posterior without full-bufer computation, DP-DQN evaluates an empirical mini-batch $\overline { { \{ z _ { i } \} _ { i = 1 } ^ { B } } } \sim B$ alongside T synthetic prior atoms $\{ \tilde { z } _ { j } \} _ { j = 1 } ^ { T } \sim F _ { 0 } .$ forming the discrete composite measure $\begin{array} { r } { \begin{array} { r } { P _ { k } = \sum _ { i = 1 } ^ { B } w _ { i } ^ { \mathrm { e m p } } \delta _ { z _ { i } } + W _ { \mathrm { p r i o r } } \sum _ { j = 1 } ^ { T } q _ { j } ^ { ( 0 ) } \delta _ { \tilde { z } _ { j } } } \end{array} } \end{array}$ . Minimizing the expected temporal-diference error $\mathbb { E } _ { z \sim P _ { k } } \big [ \ell _ { \mathrm { T D } } ( z ; \theta , \theta ^ { - } ) \big ]$ yields the composite DP-Bellman loss:

![](images/c8db2c522fbc3df3c33012675db007b0723b36ef9caaecf09ae8397c3679e22c.jpg)

(b) Scalability Across Horizon N [10, 50]  
![](images/3d7741d4014b18f255c90bda9d79f1f9fad0e2116a6e579fe7cfcc28609a4d33.jpg)  
Figure 2: Empirical exploration and scaling on DeepSea (Osband et al., 2019a). (a) Cumulative regret on DeepSea-20 (10 random seeds, shaded bands denote ±1 standard deviation) comparing DP-DQN against BootDQN-RP, BDQN, and DP-DQN in the Bayesian bootstrap limit $( \alpha  0 ;$ Section 3). (b) Empirical sample complexity across 30 grid sizes up to $N = 5 0$ (6 random seeds per size, error bars denote ±1 standard deviation), showing empirical $\mathcal { O } ( N ^ { 1 . 4 5 } )$ scaling for DP-DQN compared to $\mathcal { O } ( N ^ { 3 } )$ for BootDQN-RP.

$$
\mathcal { L } _ { \mathrm { D P } } ( \theta ) = \sum _ { i = 1 } ^ { B } w _ { i } ^ { \mathrm { e m p } } \big ( y _ { i } - Q ( s _ { i } , a _ { i } ; \theta ) \big ) ^ { 2 } + W _ { \mathrm { p r i o r } } \sum _ { j = 1 } ^ { T } q _ { j } ^ { ( 0 ) } \big ( \tilde { y } _ { j } - Q ( \tilde { s } _ { j } , \tilde { a } _ { j } ; \theta ) \big ) ^ { 2 } ,\tag{17}
$$

where $\begin{array} { r } { y _ { i } = r _ { i } + ( 1 - d _ { i } ) \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s _ { i } ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) } \end{array}$ and $\begin{array} { r l } & { \tilde { y } _ { j } = \tilde { r } _ { j } + ( 1 - \tilde { d } _ { j } ) \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( \tilde { s } _ { j } ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) } \end{array}$ denote empirical and prior Bellman targets, and $\theta ^ { - }$ tracks target network parameters via Polyak averaging $\theta ^ { - }  ( 1 - \tau ) \theta ^ { - } +$ $\tau \theta _ { k }$ (Lillicrap et al., 2015). Layer Normalization (Ba et al., 2016) is applied across hidden layers to ensure numerical stability across heterogeneous stick-breaking weights.

Information Calibration via Bellman Surprise. Scaling the prior weight $W _ { \mathrm { p r i o r } } \sim \mathrm { B e t a } ( \alpha + 1 , n )$ by the raw bufer size would cause the prior mass to vanish in early unrewarded rollouts (prior swamping). To prevent this, we modulate the prior mass by the cumulative information content of observed transitions rather than raw sample count. Setting $W _ { \mathrm { p r i o r } } \sim \mathrm { B e t a } ( \alpha + 1 , N _ { \mathrm { s t a t } } )$ , the efective sample counter $N _ { \mathrm { s t a t } }$ accumulates sample weight proportional to normalized Bellman surprise:

$$
N _ { \mathrm { s t a t } }  N _ { \mathrm { s t a t } } + \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m i n } ( 1 , \frac { | \delta _ { i } | } { \sigma _ { \mathrm { T D } } } ) , \quad \delta _ { i } = y _ { i } - Q ( s _ { i } , a _ { i } ; \theta ) ,\tag{18}
$$

where $\sigma _ { \mathrm { T D } }$ denotes the running standard deviation of Bellman errors. Transitions with near-zero Bellman error represent well-modeled dynamics that contribute negligible information, maintaining non-vanishing prior weight until novel reward signals are uncovered (Appendix C).

Episodic Boundary Warmstart and Rollout Freezing. To preserve multi-step temporal consistency, DP-DQN eliminates intra-episode parameter updates, amortizing optimization into $W = \lceil H / 2 \rceil$ stochastic warmstart gradient steps at episode boundaries. At the start of episode $k ,$ prior atoms $\{ \tilde { z } _ { j } \} _ { j = 1 } ^ { T } \sim F _ { 0 }$ are drawn once to fix the episodic exploration hypothesis. Across steps $w = 1 , \ldots , W$ , fresh empirical mini-batches $\mathcal { D } _ { w } \sim \mathcal { B }$ are sampled to optimize θ on ${ \mathcal { L } } _ { \mathrm { D P } } ( \theta )$ , which approximately draws a value function from the functional posterior. In accordance with PSRL (Strens, 2000; Osband et al., 2013), the updated parameters $\theta _ { k } \gets \theta$ are frozen throughout the episode rollout, executing greedy actions $a _ { t } = \mathrm { a r g }$ ma $\mathrm { x } _ { a } Q ( s _ { t } , a ; \theta _ { k } )$ to guarantee temporally coordinated exploration with a single neural network (Algorithm 2).

Algorithm 2 DQN with Thompson Sampling-Based Exploration via DP-BNNs   
1: Input: Action-value network $Q ( s , a ; \theta )$ , target network parameters $\theta ^ { - }$ , DP prior parameters $( \alpha , F _ { 0 } )$ , truncation   
level T, replay bufer B, mini-batch size $B ,$ warmstart steps W, learning rate $\eta ,$ discount factor $\gamma \in [ 0 , 1 )$ , episode   
horizon H, Polyak rate τ   
2: Initialize efective information counter $N _ { \mathrm { s t a t } } \gets 0$   
3: for episode $k = 1 , 2 , \ldots$ . do   
4: Stage 1: Value Function Posterior Sampling via DP-BNNs (Warmstart)   
5: Sample synthetic prior transitions: $\tilde { z } _ { j } = ( \tilde { s } _ { j } , \tilde { a } _ { j } , \tilde { r } _ { j } , \tilde { s } _ { j } ^ { \prime } , \tilde { d } _ { j } ) \overset { \mathrm { i i d } } { \sim }$ F<sub>0</sub> for $j = 1 , \dots , T$   
6: Compute prior stick weights $q _ { j } ^ { ( 0 ) }$ via Eq. (5) for $j = 1 , \dots , T$   
7: for step $w = 1 , \ldots , W$ do   
8: Sample empirical mini-batch $\mathcal { D } _ { w } = \{ z _ { i } = ( s _ { i } , a _ { i } , r _ { i } , s _ { i } ^ { \prime } , d _ { i } ) \} _ { i = 1 } ^ { B } \sim \mathcal { B }$   
9: Sample prior factor $W _ { \mathrm { p r i o r } } \sim$ Beta $( \alpha + 1 , N _ { \mathrm { s t a t } } )$ and empirical weights $w _ { i } ^ { \mathrm { e m p } }$ via $\operatorname { E q . } \left( 8 \right)$   
10: Compute Bellman targets: $y _ { i } = r _ { i } + ( 1 - d _ { i } ) \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s _ { i } ^ { \prime } , a ^ { \prime } ; \theta ^ { - } )$ and $\begin{array} { r } { \tilde { y } _ { j } = \tilde { r } _ { j } + ( 1 - \tilde { d } _ { j } ) \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( \tilde { s } _ { j } ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) } \end{array}$   
11: Update online network: $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { D P } } ( \theta )$ (via Eq. (17))   
12: Calibrate information counter $N _ { \mathrm { s t a t } }$ via TD surprise (Eq. (18))   
13: end for   
14: Freeze episode policy network: $\theta _ { k } \gets \theta$   
15: Stage 2: Temporally Coordinated Rollout under Sampled Hypothesis   
16: Observe initial state $s _ { 0 }$ from environment   
17: for $t = 0 , \ldots , H - 1$ do   
18: Select action greedily under sampled model: $a _ { t } = \arg \operatorname* { m a x } _ { a } Q ( s _ { t } , a ; \theta _ { k } )$   
19: Execute $a _ { t } ,$ , observe reward $r _ { t } ,$ next state $s _ { t + 1 } .$ , and terminal flag $d _ { t }$   
20: Store transition $z _ { t } = ( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } , d _ { t } )$ into replay bufer B   
21: if d is true then break   
22: end if   
23: end for   
24: Synchronize target network via Polyak tracking: $\theta ^ { - }  ( 1 - \tau ) \theta ^ { - } + \tau \theta _ { k }$   
25: end for

## 5.3 Empirical Evaluation on Hard-Exploration Benchmarks

To evaluate whether DP-BNN posteriors provide efective randomized value functions with a single neural network, we examine DP-DQN across the scalable DeepSea benchmark (Osband et al., 2019a), the canonical diagnostic domain for deep exploration. Specifically, we evaluate: (i) whether a single neural network equipped with a DP posterior matches the sample eficiency of multi-network ensembles on canonical DeepSea-20; (ii) how empirical sample complexity scales as grid size expands up to N = 50; and (iii) how the computational and memory footprint of DP-DQN compares with multi-network ensemble baselines.

The DeepSea Exploration Benchmark. DeepSea requires an agent to execute N consecutive coordinated actions across an episode of length N to collect a solitary terminal reward $( r = + 1 . 0 ) $ at $( N - 1 , N - 1 )$ . It challenges exploration through three distinct mechanisms: (i) an unknown action permutation per row masks the optimal direction; (ii) a continuous step cost $\mathrm { o f \mathrm { ~ - 0 . 0 1 / } } N$ penalizes exploration along the optimal path and biases myopic policies toward the unrewarded boundary; and (iii) any suboptimal step irreversibly routes the agent into zero-reward absorbing states. Consequently, dithering exploration strategies (such as ϵ-greedy) require an expected $\Omega ( 2 ^ { N } )$ episodes to discover the goal (Osband et al., 2016; 2019a)—scaling from $2 ^ { 2 0 } \approx 1 0 ^ { 6 }$ episodes at N = 20 to $2 ^ { 5 0 } \approx 1 0 ^ { 1 5 }$ episodes at $N = 5 0$ . We benchmark on canonical DeepSea-20 (Figure 2a) and evaluate scaling behavior across 30 grid sizes up to N = 50 (Figure 2b).

Base Measure for DP-DQN. To ensure exploration is driven without privileged environment knowledge, DP-DQN specifies an uninformative product base measure over transitions, $F _ { 0 } ( s , a , r , s ^ { \prime } , d ) = F _ { 0 s } ( s ) \times F _ { 0 a } ( a ) \times$ $F _ { 0 r } ( r ) \times F _ { 0 s ^ { \prime } } ( s ^ { \prime } ) \times F _ { 0 d } ( d )$ . State and next-state components are sampled uniformly over the $N ^ { 2 }$ canonical one-hot vectors $( \tilde { s } , \tilde { s } ^ { \prime } \sim$ Uniform $( S _ { \mathrm { o n e - h o t } } \subset \{ 0 , 1 \} ^ { N ^ { 2 } } ) )$ , actions uniformly a˜ ∼ Uniform({0, 1}), and continuing flags $\tilde { d } = 0$ , encoding zero information about transition graph topology or goal coordinates. For rewards, setting $F _ { 0 r } = \mathcal { U } ( 0 , R _ { \mathrm { m a x } } )$ (with $R _ { \mathrm { m a x } } = 1 . 0 )$ naturally induces stochastic optimism (Osband & Van Roy, 2017; Agrawal & Jia, 2017), sustaining positive expected value on unexplored states to overcome continuous step costs without artificial reward shaping. We set concentration parameter $\alpha = 3 . 0$ and truncation leve $T = 5 0 ;$ complete hyperparameter specifications, support conditions, and base measure ablations are detailed in Table 2 and Appendices B and D.

DeepSea-20 Exploration Performance. Figure 2a presents comparative results on DeepSea-20 across 10 independent random seeds per algorithm over 10,000 episodes (mean cumulative regret with shaded ±1 standard deviation). All agents employ an action-value architecture with a single hidden layer of 20 units (MLP-20), matching the network capacity in Osband et al. (2018). For DP-DQN, Layer Normalization (Ba et al., 2016) is included to stabilize gradient variance across heterogeneous stick-breaking weights, with concentration parameter $\alpha = 3 . 0$ and truncation level $T = 5 0$ (Table 2). Equipped with the uninformative uniform base measure, DP-DQN achieves a 100% solve rate (10/10 seeds), discovering the terminal goal at episode $3 9 7 \pm 1 5 4$ and reaching policy convergence in $6 0 2 \pm 1 0 4$ episodes, with cumulative regret plateauing at $4 1 8 \pm 9 4$ . In comparison, Bootstrapped DQN with randomized priors (BootDQN-RP, 20 ensemble heads + 20 prior networks) (Osband et al., 2018) requires $5 , 2 1 0 \pm 6 0 4$ episodes with 4,922 ± 582 regret. Notably, DP-DQN consolidates into the optimal policy within ≈ 200 episodes after initial discovery $( 3 9 7 \pm 1 5 4 \to 6 0 2 \pm 1 0 4 )$ Because the high Bellman surprise of the newly discovered goal sharply increments the information counter $N _ { \mathrm { s t a t } }$ , the prior factor $W _ { \mathrm { p r i o r } }$ is rapidly attenuated in favor of empirical Bellman targets (Eq. (17)), stabilizing greedy execution without the inter-head variance or conflicting prior biases that can impede policy consolidation in heuristic ensembles. Conversely, standard DQN, BDQN (Azizzadenesheli et al., 2018), and DP-DQN in the Bayesian bootstrap limit $( \alpha  0 ;$ Section 3, Eq. (3)) fail to discover the goal across all seeds $( 0 / 1 0$ seeds solved), incurring linear regret $\left( > 9 , 9 0 0 \right)$ . In this Bayesian bootstrap limit where the prior concentration parameter vanishes $( \alpha \to 0 )$ , the agent samples purely from empirical replay transitions without synthetic prior augmentation $( W _ { \mathrm { p r i o r } } = 0 )$ . Because early trajectories observe only zero-reward transitions, pure empirical resampling provides no exploratory optimism toward unvisited states, confirming that the prior base measure $F _ { 0 }$ is essential for driving deep exploration (Appendices C and D.3).

Scalability Across DeepSea Grid Size $N \in [ 1 0 , 5 0 ]$ . To evaluate scalability as the exploration horizon expands, we benchmarked DP-DQN across 30 grid sizes up to $N = 5 0$ (6 independent random seeds per size, error bars denote ±1 standard deviation; Figure 2b and Table 1). To accommodate the expanding state space $( | S | = 2 , 5 0 0 )$ and trajectory space $( 2 ^ { 5 0 } \approx 1 \bar { 0 } ^ { 1 5 }$ paths), the action-value network uses two hidden layers of 64 units each (64 × 64 MLP with Layer Normalization). Following standard benchmark convention (Osband et al., 2019a), an environment run is deemed solved when the agent achieves consistent policy convergence, operationalized as maintaining a rolling mean episode return $> 0 . 9$ over 20 consecutive episodes (relative to the optimal return $1 . 0 - 0 . 0 1 = 0 . 9 9 )$ , corresponding to the point where cumulative regret permanently flatlines. DP-DQN consistently solves DeepSea-50 in $1 , 9 2 5 \pm 1 8 4$ episodes (Table 1). Across grid sizes from $N = 1 0$ to $N = 5 0$ , empirical sample complexity scales as $\mathcal { O } ( N ^ { 1 . 4 5 } )$ , outperforming the empirical $\mathcal { O } ( N ^ { 3 } )$ scaling of BootDQN-RP (Osband et al., 2018) and remaining well below the $\Omega ( 2 ^ { N } )$ lower bound of dithering heuristics.

Computational and Memory Eficiency. A foundational advantage of DP-DQN is that it eliminates the severe computational overhead of multi-network ensembles. While BootDQN-RP $( K = 2 0 )$ must instantiate, store, and train K separate action-value networks alongside K prior networks and target networks $( \mathcal { O } ( K d )$ parameters and optimizer bufers), DP-DQN operates on a single active neural network $( \mathcal { O } ( d )$ parameters), matching the parameter footprint of standard DQN. The non-parametric mechanism introduces two computational overheads relative to standard DQN: (i) sampling Dirichlet process stick-breaking weights requires $\mathcal { O } ( T )$ scalar operations once per episode for prior atoms and $\mathcal { O } ( B )$ per warmstart step for empirical mini-batches, totaling an asymptotic stick-breaking complexity of only $\mathcal { O } ( T + W \cdot B )$ scalar operations per episode $( \mathcal { O } ( T + W B ) \ll W ( B + T ) d _ { \{ }$ , negligible compared to network tensor arithmetic); and (ii) evaluating the composite loss $\left( \mathrm { E q . ~ } ( 1 7 ) \right)$ across the T prior transitions in addition to the empirical mini-batch of size B scales gradient backpropagation by a factor of $1 + T / B$ . Profiling on canonical DeepSea-20 on a standardized CPU core confirms this eficiency: DP-DQN incurs an amortized execution latency of 0.43 ms/step (comparable to standard $\mathrm { D Q N ^ { \prime } s \ 0 . 3 8 m s / s t e p }$ , and 2.4× faster than BootDQN-RP’s 1.05 ms/step). Combined with superior sample eficiency (602 ± 104 vs. $5 , 2 1 0 \pm 6 0 4$ episodes), DP-DQN converges in only 5.16 s of total CPU time compared to 103.26 s for BootDQN-RP—achieving a 20.0× wall-clock speedup while using 20× less memory (Table 4, Appendix G).

Generality Across Diverse Decision Tasks. To evaluate whether the exploratory advantages of DP-BNNs generalize beyond DeepSea, we benchmarked the framework across diverse decision-making tasks (Table 1 and Appendix E). Across deceptive sequential corridors (N-Chain; Strens 2000), environmental stochasticity and reward traps (RiverSwim-6; Strehl & Littman 2008), continuous control under gravity constraints (MountainCar-v0; Sutton & Barto 1998), and deep contextual bandits (Wheel Bandit; Riquelme et al. 2018; Figure 6), DP-BNN posteriors consistently sustain directed exploration, matching or exceeding competitive multi-network baselines while operating with a single network. Complete experimental protocols and tabular comparisons are detailed in Appendix E.

## 6 Conclusions and Future Work

We introduced Dirichlet Process Bayesian Neural Networks (DP-BNNs), which quantify epistemic uncertainty by placing a nonparametric prior on the data-generating distribution rather than on network weights. We illustrated the application of DP-BNNs to deep reinforcement learning by developing DP-DQN for randomized value functions. By drawing an action-value hypothesis for each episode rollout, DP-DQN realizes temporally consistent Thompson sampling with a single neural network, resolving hard exploration challenges while avoiding the memory and computational overhead of multi-network ensembles.

Several promising research directions emerge from this paradigm. On the theoretical side, establishing formal finite-sample regret bounds for DP-DQN that explicitly capture the dependence on the DP prior hyperparameters—namely the concentration parameter α, the base measure F , and the truncation level T—presents a vital avenue of inquiry to rigorously characterize how prior mass and truncation accuracy govern exploration eficiency and asymptotic convergence (Osband & Van Roy, 2017). More broadly, designing structured base measures to incorporate domain-specific inductive biases presents an exciting future direction (Klissarov & Machado, 2023; Machado et al., 2023), which is notoriously dificult to achieve in weight-space Bayesian neural networks (Wilson & Izmailov, 2020). Finally, extending DP-BNN posteriors to continuous actor-critic architectures holds strong practical promise to eliminate the multi-critic ensembles commonly required to mitigate value overestimation or estimate uncertainty (Haarnoja et al., 2018), enabling principled, sample-eficient continuous control with a single neural network.

## Acknowledgments

The author is grateful to the Action Editor and the anonymous reviewers for their insightful and constructive reviews, which significantly improved the paper. The author warmly thanks Dr. Igor Prünster, Dr. Pierre Alquier, and Dr. Odalric-Ambrym Maillard for encouraging comments.

## Reproducibility Statement

The source code, environment wrappers, configuration files, reproduction scripts, and pedagogical Python notebooks are available at https://github.com/sumilations/ Dirichlet-Process-Bayesian-Neural-networks. Experimental hyperparameters, architectures, and evaluation protocols are detailed in Appendix E (Table 2), while mathematical derivations and algorithmic analyses are provided in Appendices A–D.

## References

Shipra Agrawal and Navin Goyal. Analysis of Thompson sampling for the multi-armed bandit problem. In Conference on Learning Theory (COLT), pp. 39–1. JMLR Workshop and Conference Proceedings, 2012.

Shipra Agrawal and Randy Jia. Optimistic posterior sampling for reinforcement learning: worst-case regret bounds. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

Julyan Arbel, Konstantinos Pitas, Mariia Vladimirova, and Vincent Fortuin. A primer on Bayesian neural networks: Review and debates. arXiv preprint arXiv:2309.16314, 2023.

Peter Auer, Thomas Jaksch, and Ronald Ortner. Near-optimal regret bounds for reinforcement learning. Advances in neural information processing systems, 21, 2008.

Kamyar Azizzadenesheli, Emma Brunskill, and Animashree Anandkumar. Eficient exploration through Bayesian deep Q-networks. In 2018 Information Theory and Applications Workshop (ITA), pp. 1–9. IEEE, 2018.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geofrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pp. 449–458. PMLR, 2017.

Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. Weight uncertainty in neural network. In International conference on machine learning, pp. 1613–1622. PMLR, 2015.

Ronen I Brafman and Moshe Tennenholtz. R-max-a general polynomial time algorithm for near-optimal reinforcement learning. Journal of Machine Learning Research, 3(Oct):213–231, 2002.

Ismaël Castillo. Bayesian Nonparametric Statistics: École d’Été de Probabilités de Saint-Flour LI–2023, volume 2358 of Lecture Notes in Mathematics. Springer, 2024.

Richard Dearden, Nir Friedman, and David Andre. Bayesian Q-learning. In AAAI/IAAI, pp. 761–768, 1998.

Audrey Durand, Odalric-Ambrym Maillard, and Joelle Pineau. Streaming kernel regression with provably adaptive mean, variance, and regularization. In International Conference on Artificial Intelligence and Statistics (AISTATS), pp. 1849–1857, 2018.

Vikranth Dwaracherla, Kavosh Asadi, and Ian Osband. Hypermodels for exploration. In International Conference on Learning Representations, 2020.

Bradley Efron and Robert J Tibshirani. An introduction to the bootstrap. Chapman and Hall/CRC, 1994.

Damien Ernst, Pierre Geurts, and Louis Wehenkel. Tree-based batch mode reinforcement learning. Journal of Machine Learning Research, 6(Apr):503–556, 2005.

Thomas S. Ferguson. A Bayesian analysis of some nonparametric problems. The Annals of Statistics, 1(2): 209–230, 1973.

Andrew YK Foong, Yingzhen Li, José Miguel Hernández-Lobato, and Richard E Turner. “In-between” uncertainty in Bayesian neural networks. In ICML Workshop on Uncertainty and Robustness in Deep Learning, 2019.

Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Ian Osband, Alex Graves, Vlad Mnih, Alistair Muldal, Thomas Degris, Shane Legg, Charles Blundell, and Jan Leike. Noisy networks for exploration. In International Conference on Learning Representations, 2018.

Francesco Gafi, Antonio Lijoi, and Igor Prünster. Random probability measures with fixed mean distributions. The Annals of Applied Probability, 35(4):2239–2261, 2025.

Yarin Gal and Zoubin Ghahramani. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.

Mohammad Ghavamzadeh, Shie Mannor, Joelle Pineau, and Aviv Tamar. Bayesian reinforcement learning: A survey. Foundations and Trends® in Machine Learning, 8(5-6):359–483, 2015.

Subhashis Ghosal. The Dirichlet process, related priors and posterior asymptotics. In Nils Lid Hjort, Chris Holmes, Peter Müller, and Stephen G. Walker (eds.), Bayesian Nonparametrics, pp. 35–79. Cambridge University Press, 2010.

Subhashis Ghosal and Aad W van der Vaart. Fundamentals of Nonparametric Bayesian Inference, volume 44. Cambridge University Press, 2017.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Of-policy maximum entropy deep reinforcement learning with a stochastic actor. In International Conference on Machine Learning (ICML), pp. 1861–1870, 2018.

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011.

Hemant Ishwaran and Lancelot F James. Gibbs sampling methods for stick-breaking priors. Journal of the American statistical Association, 96(453):161–173, 2001.

Emilie Kaufmann, Nathaniel Korda, and Rémi Munos. Thompson sampling: An asymptotically optimal finite-time analysis. In International Conference on Algorithmic Learning Theory (ALT), pp. 199–213. Springer, 2012.

Martin Klissarov and Marlos C. Machado. Deep laplacian-based options for temporally-extended exploration. In International Conference on Machine Learning (ICML), pp. 17067–17088. PMLR, 2023.

Branislav Kveton, Csaba Szepesvari, Sharan Vaswani, Zheng Wen, Tor Lattimore, and Mohammad Ghavamzadeh. Garbage in, reward out: Bootstrapping exploration in multi-armed bandits. In International Conference on Machine Learning, pp. 3601–3610. PMLR, 2019.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Antonio Lijoi and Igor Prünster. Distributional properties of means of random probability measures. Statistics Surveys, 3:47–95, 2009.

Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.

Marlos C. Machado, André Barreto, Doina Precup, and Michael Bowling. Temporal abstraction in reinforcement learning with the successor representation. Journal of Machine Learning Research, 24(80):1–69, 2023.

David JC MacKay. Information theory, inference and learning algorithms. Cambridge university press, 2003.

Odalric-Ambrym Maillard. Mathematics of statistical sequential decision making. HDR dissertation, Université de Lille, 2019.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.

Pietro Muliere and Luca Tardella. Approximating distributions of random functionals of Ferguson–Dirichlet priors. Canadian Journal of Statistics, 26(2):283–297, 1998.

Yaniv Oren, Viliam Vadocz, Matthijs T. J. Spaan, and Wendelin Boehmer. Epistemic Monte Carlo tree search. In International Conference on Learning Representations (ICLR), 2025.

Ian Osband and Benjamin Van Roy. Why is posterior sampling better than optimism for reinforcement learning? In International conference on machine learning, pp. 2701–2710. PMLR, 2017.

Ian Osband, Daniel Russo, and Benjamin Van Roy. (more) eficient reinforcement learning via posterior sampling. Advances in Neural Information Processing Systems, 26, 2013.

Ian Osband, Charles Blundell, Alexander Pritzel, and Benjamin Van Roy. Deep exploration via bootstrapped DQN. Advances in neural information processing systems, 29, 2016.

Ian Osband, John Aslanides, and Albin Cassirer. Randomized prior functions for deep reinforcement learning. Advances in neural information processing systems, 31, 2018.

Ian Osband, Yotam Doron, Matteo Hessel, John Aslanides, Eren Sezener, Andre Saraiva, Katrina McKinney, Tor Lattimore, Csaba Szepesvari, Satinder Singh, et al. Behaviour suite for reinforcement learning. arXiv preprint arXiv:1908.03568, 2019a.

Ian Osband, Benjamin Van Roy, Daniel J Russo, and Zheng Wen. Deep exploration via randomized value functions. Journal of Machine Learning Research, 20(124):1–62, 2019b.

Martin L Puterman. Markov decision processes. Handbooks in operations research and management science, 2:331–434, 1990.

Martin Riedmiller. Neural fitted Q iteration–first experiences with a data eficient neural reinforcement learning method. In European Conference on Machine Learning, pp. 317–328. Springer, 2005.

Carlos Riquelme, George Tucker, and Jasper Snoek. Deep Bayesian bandits showdown: An empirical comparison of Bayesian deep networks for Thompson sampling. arXiv preprint arXiv:1802.09127, 2018.

Kathryn Roeder. Density estimation with confidence sets exemplified by superclusters and voids in the galaxies. Journal of the American Statistical Association, 85(411):617–624, 1990.

Donald B. Rubin. The Bayesian Bootstrap. The Annals of Statistics, 9(1):130–134, 1981.

Daniel J Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen, et al. A tutorial on Thompson sampling. Foundations and Trends® in Machine Learning, 11(1):1–96, 2018.

Jayaram Sethuraman. A constructive definition of Dirichlet priors. Statistica Sinica, 4(2):639–650, 1994.

Alexander L. Strehl and Michael L. Littman. An analysis of model-based interval estimation for Markov decision processes. Journal of Computer and System Sciences, 74(8):1309–1331, 2008.

Malcolm Strens. A Bayesian approach to reinforcement learning. In Proceedings of the Seventeenth International Conference on Machine Learning (ICML), pp. 943–950, 2000.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.

William R Thompson. On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. Biometrika, 25(3-4):285–294, 1933.

Hado van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double Q-learning. In AAAI Conference on Artificial Intelligence, volume 30, pp. 2094–2100, 2016.

Sumit Vashishtha. Bayesian Reinforcement Learning with Dirichlet Processes. PhD thesis, Inria & Université de Lille, 2026.

Sumit Vashishtha and Odalric-Ambrym Maillard. Leveraging priors on distribution functions for multi-arm bandits. Reinforcement Learning Journal, 6:1600–1623, 2025.

Max Welling and Yee W Teh. Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pp. 681–688, 2011.

Andrew Gordon Wilson and Pavel Izmailov. Bayesian deep learning and a probabilistic perspective of generalization. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pp. 4697– 4708, 2020.

# Supplementary Material: Technical Appendix

Bayesian Neural Networks via Dirichlet Process Priors on the Data-Generating Distribution for Randomized Value Functions in Deep Reinforcement Learning

Organization of the Appendix. The supplementary appendix is organized as follows:

• Appendix A: Truncated stick-breaking representation, truncation bounds, the 99.9% residual mass rule, and galaxy velocity density estimation.

• Appendix B: Base measure topological support conditions, 2D annular manifold benchmark formulation, and closed-form BALD derivations.

• Appendix C: Dynamic stochastic mini-batch warmstart mechanics, prevention of prior attenuation, and Bellman credit propagation depth.

• Appendix D: Stochastic optimism and deep exploration analysis, contrasting DP-BNNs against randomized prior ensembles (BootDQN-RP) and neural-linear approximations (BDQN), with empirical base measure ablations.

• Appendix E: Consolidated empirical benchmark suite (Table 1) and hyperparameter protocols (Table 2) covering DeepSea, RiverSwim-6, N-Chain, and Wheel Bandit.

• Appendix F: Continuous state-space RL exploration on MountainCar-v0 with a model-free Brownian difusive base measure.

• Appendix G: Computational and memory complexity comparison across value-based deep RL paradigms (Table 3).

## A Truncated Stick-Breaking Representation and Hyperparameter Selection

The finite stick-breaking representation of DP priors (Equations (5)–(6)) and the posterior random measure construction (Vashishtha & Maillard, 2025) provide an exact, constructive generative mechanism for sampling random probability measures (Sethuraman, 1994; Ishwaran & James, 2001). Rather than relying on heuristic or arbitrary cutof levels, we determine the stick-breaking truncation level T (the number of prior pseudoobservations) for the prior measure $\mathrm { D P } ( \alpha , F _ { 0 } )$ using a principled 99.9% expected residual probability mass rule.

Under Sethuraman’s stick-breaking construction, prior probability weights are generated via $\begin{array} { r } { \pi _ { k } = V _ { k } \prod _ { l < k } ( 1 - } \end{array}$ $V _ { l } )$ with $V _ { l } \stackrel { \mathrm { i . i . d . } } { \sim } \mathrm { B e t a } ( 1 , \alpha )$ . When the series is truncated at T atoms, the residual probability mass unallocated to the truncated support is $\begin{array} { r } { R _ { T } = \sum _ { k = T + 1 } ^ { \infty } \pi _ { k } = \prod _ { k = 1 } ^ { T } ( 1 - V _ { k } ) } \end{array}$ . Since $\begin{array} { r } { \mathbb { E } [ 1 - V _ { k } ] = \frac { \alpha } { \alpha + 1 } } \end{array}$ , mutual independence of $\{ V _ { k } \} _ { k = 1 } ^ { T }$ yields the exact closed-form expected residual mass:

$$
\mathbb { E } [ R _ { T } ] = \prod _ { k = 1 } ^ { T } \mathbb { E } [ 1 - V _ { k } ] = \left( { \frac { \alpha } { \alpha + 1 } } \right) ^ { T } .\tag{19}
$$

To ensure that the finite truncation captures at least 99.9% of the total prior probability mass $( \mathbb { E } [ R _ { T } ] \leq 1 0 ^ { - 3 } )$ we require:

$$
\mathbb { E } [ R _ { T } ] = \left( { \frac { \alpha } { \alpha + 1 } } \right) ^ { T } \leq 1 0 ^ { - 3 } \implies T \geq { \frac { \ln ( 1 0 0 0 ) } { \ln \left( 1 + { \frac { 1 } { \alpha } } \right) } } \approx { \frac { 6 . 9 0 7 8 } { \ln \left( 1 + { \frac { 1 } { \alpha } } \right) } } .\tag{20}
$$

Equation (20) provides an explicit, quantitative guideline for configuring T across our benchmark suite:

Reinforcement Learning Tasks $( \alpha = 3 . 0 )$ . In DeepSea and N-Chain, evaluating Equation (20) gives $\begin{array} { r } { T \geq \frac { 6 . 9 0 7 8 } { \ln ( 4 / 3 ) } \approx 2 4 . 0 1 } \end{array}$ . Hence, any truncation level $T \geq 2 5$ guarantees $> 9 9 . 9 \%$ prior mass retention. In our standard implementation, setting $T = 5 0$ captures over 99.9999% of the prior mass $( \mathbb { E } [ R _ { T } ] = ( 0 . 7 5 ) ^ { 5 0 }$ ≈ $5 . 6 6 \times 1 0 ^ { - 7 } < 1 0 ^ { - 6 } )$ , completely eliminating truncation distortion while incurring negligible computational overhead.

Unit Concentration Baseline $( \alpha = 1 . 0 )$ . Under standard uninformative Dirichlet priors $( \alpha = 1 . 0 )$ Equation (20) yields $\begin{array} { r } { T \geq \frac { 6 . 9 0 7 8 } { \ln ( 2 ) } \approx \dot { 9 . 9 6 } } \end{array}$ , showing that just 10 atoms sufice to capture 99.9% of the prior mass.

Dispersed Regression Benchmark $( \alpha = 1 0 . 0 )$ . In the 2D annular manifold benchmark (Appendix B.2), where high concentration $\alpha = 1 0 . 0$ is deliberately selected to produce dispersed functional hypotheses in the cavity void, Equation (20) yields $\begin{array} { r } { T \ge \frac { 6 . 9 0 7 8 } { \ln ( 1 . 1 0 ) } \approx 7 2 . 5 } \end{array}$ . Setting T = 50 captures over 99.1% of the total prior mass $( \mathbb { E } [ R _ { T } ] = ( 1 0 / 1 1 ) ^ { 5 0 } \approx 0 . 0 0 8 5 )$

Empirical Density Estimation on the Galaxy Velocity Dataset. To validate our stick-breaking posterior sampler empirically, we evaluate density estimation on the canonical galaxy velocity benchmark (Roeder, 1990). The measurements are velocities at which galaxies in the Corona-Borealis region move away from our galaxy. When galaxies are clustered, the velocity density is multimodal, as evidenced in the empirical CDF in Figure 3 where mass concentrates in distinct interior modes. Starting with a $\mathrm { D P } ( \alpha = 2 , F _ { 0 } = \mathcal { N } ( 1 0 , 1 ) )$ prior, the spread of distributions sampled from the DP posterior provides a well-calibrated confidence band of the density estimated through the DP.

![](images/14daff99c79f318e1978b442e88b87d3082aec5a9fb142307dd3f78cc461b810.jpg)  
Figure 3: DP prior and posterior density estimates on the galaxy velocity dataset.

## B Prior Base Measure Design and Epistemic Uncertainty Formulation

In this section, we formalize the topological support conditions required for the DP prior base measure $F _ { 0 }$ and detail its concrete realization for epistemic uncertainty quantification in spatial regression.

## B.1 Base Measure Support Condition

In the initial exploration phase of learning, whether in supervised regression or reinforcement learning, observations are sparse or absent across entire regions of the input space. Value predictions and exploratory hypotheses are therefore driven almost entirely by the synthetic pseudo-observations generated through the DP prior base measure $F _ { 0 }$ . Because early exploratory actions have delayed, irreversible consequences in deep MDPs, the structure and domain support of $F _ { 0 }$ are critical.

A natural and general formulation is to specify a factorized product base measure over transition tuples $z = ( s , a , r , s ^ { \prime } , d ) \colon$

$$
F _ { 0 } ( s , a , r , s ^ { \prime } , d ) = F _ { 0 s } ( s ) \times F _ { 0 a } ( a ) \times F _ { 0 r } ( r ) \times F _ { 0 s ^ { \prime } } ( s ^ { \prime } ) \times F _ { 0 d } ( d ) .\tag{21}
$$

The specification of state, action, and terminal components $F _ { 0 s } , \ F _ { 0 a }$ , and $F _ { 0 d }$ must strictly respect the physical domain and support of the underlying Markov Decision Process (with $d = 0$ for non-terminal continuing transitions). This requirement is mathematically dictated by the topological support properties of $\mathrm { D P s } \mathrm { : }$

Lemma 1 (Support of Dirichlet Processes (Ghosal, 2010)). In the weak topology, the topological support of the $D P \mathrm { D P } ( \alpha , F _ { 0 } )$ is the set of all probability measures $P ^ { \star }$ whose supports are contained within the support of the base measure:

$$
\operatorname { s u p p } ( \operatorname { D P } ( \alpha , F _ { 0 } ) ) = \{ P ^ { \star } \in { \mathcal { M } } _ { 1 } ( { \mathcal { X } } ) : \operatorname { s u p p } ( P ^ { \star } ) \subseteq \operatorname { s u p p } ( F _ { 0 } ) \} .\tag{22}
$$

Lemma 1 establishes that the DP posterior can place non-zero probability mass in an exploratory region if and only if that region is contained within the support of the base measure $F _ { 0 }$ . In practice, the factorized components are configured as follows:

Action Space Prior $( F _ { 0 a } )$ . For discrete action spaces with $| { \cal A } |$ actions, $F _ { 0 a }$ is chosen as a uniform categorical distribution over $\{ 1 , \ldots , | A | \}$ . For continuous action spaces, $F _ { 0 a }$ is selected to span the admissible action bounds $( \mathrm { e . g . , U n i f o r m } [ a _ { \mathrm { m i n } } , a _ { \mathrm { m a x } } ] )$ .

State Space and Transition Priors $( F _ { 0 s } , F _ { 0 s ^ { \prime } } )$ . Chosen to encapsulate the admissible state space $s$ (e.g., uniform over coordinate bounds or standardized feature bounding boxes).

Reward Prior $\left( F _ { 0 r } \right)$ . Crucially, unlike supervised regression where observations are static, in reinforcement learning data collection is actively dictated by the agent’s policy: a pessimistic reward prior discourages exploration, preventing the agent from ever gathering experience in unvisited states. Therefore, $F _ { 0 r }$ is designed not to replicate environment step penalties (which are learned exclusively through the empirical replay stream $\mathbb { P } _ { n } )$ , but to induce stochastic optimism (Osband & Van Roy, 2017): either via the uniform prior $F _ { 0 r } = \mathcal { U } ( 0 , R _ { \mathrm { m a x } } )$ on non-negative returns, or via zero-mean Gaussians $\mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } )$ where extreme-value order statistics across T prior atoms $( \mathbb { E } [ \operatorname* { m a x } _ { j = 1 } ^ { T } \tilde { r } _ { j } ] \approx \sigma _ { 0 } \sqrt { 2 \ln T } > 0 )$ naturally stimulate directed exploration without requiring knowledge of $R _ { \mathrm { m a x } }$ or goal locations $( \mathrm { A p p e n d i x ~ D } )$

We now demonstrate how these foundational principles—the 99.9% mass truncation rule and the topological support condition—are concretely instantiated in static spatial regression. Appendix B.2 details the geometry, factorized base measure design, and information-theoretic BALD epistemic uncertainty derivations for the 2D annular benchmark introduced in Section 4.

## B.2 Prior Base Measure and BALD Metric for the 2D Regression Benchmark

In this section, we provide the full mathematical formulation, prior base measure design, and hyperparameter specifications for the 2D epistemic uncertainty benchmark presented in Section 4 and Figure 1.

Annular Manifold Geometry. Training data points $\{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { n } ~ ( n = 3 0 0 )$ are sampled uniformly on an annulus with 2D spatial coordinates $x = ( x _ { 1 } , x _ { 2 } ) \in [ - 2 . 5 , 2 . 5 ] ^ { 2 }$ and radial distance $r = \| x \| _ { 2 } \in [ 1 . 3 , 2 . 4 5 ]$ The scalar targets $y \in [ - 1 , 1 ]$ are generated via $y = \cos ( 1 . 5 \pi r ) + \epsilon$ with additive Gaussian observation noise $\epsilon \sim \mathcal { N } ( 0 , \sigma _ { \mathrm { a l e a t o r i c } } ^ { 2 } )$ where $\sigma _ { \mathrm { a l e a t o r i c } } = 0 . 3 0$ . The central circular region $r \leq 1 . 0$ is completely void of training data (Figure 1A). A well-calibrated epistemic uncertainty estimator must detect this out-of-distribution cavity by predicting high uncertainty in the void $\mathcal { V } = \{ x \in [ - 2 . 5 , 2 . 5 ] ^ { 2 } : \| x \| _ { 2 } \leq 1 . 0 \}$ and low uncertainty on the data support $\mathcal { D } = \{ x \in [ - 2 . 5 , 2 . 5 ] ^ { 2 } : 1 . 3 \leq \| x \| _ { 2 } \leq 2 . 4 5 \}$

Bayesian Active Learning by Disagreement (BALD). Epistemic uncertainty is quantified informationtheoretically using BALD (Houlsby et al., 2011), which measures the mutual information between the mode output y and the model parameters θ conditioned on the dataset D:

$$
\operatorname { B A L D } ( x ) \triangleq I ( y , \theta \mid x , \mathcal { D } ) = H ( y \mid x , \mathcal { D } ) - \mathbb { E } _ { \theta \sim p ( \theta \mid \mathcal { D } ) } [ H ( y \mid x , \theta ) ] .\tag{23}
$$

Intuitively, BALD isolates the information that learning the true model parameters θ would provide about the prediction at test input $x ,$ cleanly separating epistemic uncertainty from inherent aleatoric observation noise $H ( y \mid x , \theta )$ . Under a Gaussian predictive distribution $\mathcal { N } ( f _ { \theta } ( x ) , \dot { \sigma } _ { \mathrm { a l e a t o r i c } } ^ { 2 } )$ , the total predictive entropy across an ensemble of J posterior models $\{ \theta ^ { ( j ) } \} _ { j = 1 } ^ { J }$ is given by:

$$
H ( y \mid x , \mathcal { D } ) \approx \frac { 1 } { 2 } \ln ( 2 \pi e ( \sigma _ { \mathrm { e p i s t e m i c } } ^ { 2 } ( x ) + \sigma _ { \mathrm { a l e a t o r i c } } ^ { 2 } ) ) ,\tag{24}
$$

where $\begin{array} { r } { \sigma _ { \mathrm { e p i s t e m i c } } ^ { 2 } ( x ) \ : = \ : \frac { 1 } { J } \sum _ { j = 1 } ^ { J } ( f _ { \theta ^ { ( j ) } } ( x ) - \bar { f } ( x ) ) ^ { 2 } } \end{array}$ is the ensemble variance, with mean prediction ${ \bar { f } } ( x ) =$ $\begin{array} { r } { \frac { 1 } { J } \sum _ { j = 1 } ^ { J } f _ { \theta ^ { ( j ) } } ( x ) } \end{array}$ . The expected observation noise entropy is:

$$
\mathbb { E } _ { \theta } [ H ( y \mid x , \theta ) ] = \frac { 1 } { 2 } \ln ( 2 \pi e \sigma _ { \mathrm { a l e a t o r i c } } ^ { 2 } ) .\tag{25}
$$

Subtracting the two terms yields the closed-form expression for BALD in nats:

$$
\mathrm { B A L D } ( x ) = \frac { 1 } { 2 } \ln \left( 1 + \frac { \sigma _ { \mathrm { e p i s t e m i c } } ^ { 2 } ( x ) } { \sigma _ { \mathrm { a l e a t o r i c } } ^ { 2 } } \right) .\tag{26}
$$

This metric evaluates strictly to zero when all posterior models agree $( \sigma _ { \mathrm { e p i s t e m i c } } ^ { 2 } = 0 )$ , and scales logarithmically as posterior disagreement increases.

Architecture and Training Hyperparameters. All evaluated neural network models share an identical multi-layer perceptron architecture comprising 3 hidden layers with 64 units per layer and ReLU activations, without Layer Normalization, mapping $\mathbb { R } ^ { 2 } \to \mathbb { R }$ . Training is conducted using the Adam optimizer with a learning rate of $1 0 ^ { - 3 }$ and batch size of 64 for 200 epochs across all models. For DP-BNN, posterior measure draws $P _ { j } \sim \mathrm { D P } ( \alpha _ { n } , \bar { F } _ { n } )$ are drawn with $J = 1 0$ independent posterior samples.

Base Measure Construction and Prior Rationale. In accordance with Lemma 1, the support of the DP prior must encapsulate the entire domain of interest to allow posterior mass in regions lacking empirical data. We specify a factorized product base measure $F _ { 0 } ( x , y ) = F _ { 0 x } ( x ) \times F _ { 0 y } ( y )$ consisting of: (i) a continuous spatial input prior $F _ { 0 x } = \mathcal { U } ( [ - 2 . 5 , 2 . 5 ] ^ { 2 } )$ , ensuring uniform prior coverage across both the observed annulus and the unobserved cavity void, and (ii) a zero-mean Gaussian target prior $F _ { 0 y } = \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } )$ with dispersed target variance $( \sigma _ { 0 } = 1 . 5 \gg \sigma _ { \mathrm { a l e a t o r i c } } = 0 . 3 0 )$ . We truncate the prior component at $T = 5 0$ synthetic pseudo-observations with concentration parameter $\alpha = 1 0 . 0$ . Because the prior atoms sample diverse target values in the void with non-trivial Dirichlet weights, each posterior measure draw $P _ { j }$ pulls the network towards a distinct hypothesis in the void, naturally producing the sharp epistemic bubble observed in Figure 1.

## C Dynamic Stochastic Mini-Batch Warmstart and Algorithmic Dynamics

In this section, we provide a comprehensive analysis of the stochastic mini-batch warmstart mechanism introduced in Section 5 and Algorithm 2, detailing why conditioning directly on the full replay bufer fails in reinforcement learning and how dynamic stick-breaking updates resolve this fundamental challenge.

## C.1 Pathologies of Full-Bufer Conditioning and Static Freezing

In classical Bayesian nonparametric regression, one is given a static dataset $\mathcal { D } = \{ z _ { i } \} _ { i = 1 } ^ { n }$ and conditions the DP prior $\mathrm { D P } ( \alpha , F _ { 0 } )$ ) directly on all n observations, yielding the posterior:

$$
P \mid { \mathcal { D } } \sim \mathrm { D P } \left( \alpha + n , { \frac { n } { \alpha + n } } \mathbb { P } _ { n } + { \frac { \alpha } { \alpha + n } } F _ { 0 } \right) ,\tag{27}
$$

where $\begin{array} { r } { \mathbb { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { z _ { i } } } \end{array}$ is the empirical distribution of observations. In reinforcement learning, however, the replay bufer B fundamentally violates the statistical assumptions of supervised learning:

Data Redundancy and Non-Uniformity. Early RL rollouts are overwhelmingly dominated by repetitive transitions in absorbing, suboptimal regions of the state space (e.g., falling into bottom-left corners in DeepSea or cycling through uninformative states). The replay bufer rapidly accumulates tens of thousands of near-identical transitions $( M = | \boldsymbol { B } | \sim 1 0 ^ { 5 } - 1 0 ^ { 6 } )$ .

Prior Swamping and Epistemic Starvation. In standard full-bufer DP conditioning, the relative weight assigned to the prior base measure scales as:

$$
\omega _ { \mathrm { p r i o r } } = \frac { \alpha } { \alpha + M } \stackrel { M \gg \alpha } { - - \alpha } 0 .\tag{28}
$$

While asymptotic concentration $( \omega _ { \mathrm { p r i o r } }  0 )$ is desirable in stationary supervised learning, in explorationsensitive RL it causes epistemic starvation: the prior base measure $F _ { 0 } .$ , which is specifically designed to inject optimism into unexplored states, is overwhelmed by the sheer volume of redundant failure data. As a consequence, the agent ceases to explore and prematurely collapses into pessimistic, greedy exploitation of suboptimal trajectories.

Computational and Memory Bottlenecks. Constructing truncated stick-breaking weights $p _ { 1 } , \dots , p _ { M }$ and performing backpropagation through a deep neural network over $M \sim 1 0 ^ { 5 }$ samples at every single episode imposes an intolerable computational and memory burden.

Conversely, sampling a single mini-batch $\mathcal { D } \subset B$ of size $B \ll M$ and freezing it across all gradient steps creates severe empirical overfitting: the value function optimizes exclusively against an isolated empirical slice, ignoring the rest of the bufer and failing to propagate Bellman credit across multi-step trajectory chains.

## C.2 Resolution via Dynamic Stochastic Mini-Batch Warmstart

Algorithm 2 overcomes the twin pathologies of computational overhead and prior swamping by performing stochastic optimization over dynamic, freshly sampled stick-breaking measures at episode boundaries.

Under the DP posterior, each sampled random measure decomposes into an empirical component over observed replay transitions and a prior component drawn from $F _ { 0 }$ . Rather than performing computationally prohibitive full-bufer optimization $( | B | \sim 1 0 ^ { 5 } )$ , the $W = \lceil H / 2 \rceil$ warmstart steps use mini-batch stochastic gradient descent to approximate the empirical component, while modulating prior salience online via Bellman temporal-diference surprise:

Episodic Prior Generation. At the beginning of episode $k ,$ the agent samples synthetic prior transitions $\tilde { \mathcal { D } } _ { 0 } = \{ ( \tilde { s } _ { j } , \tilde { a } _ { j } , \tilde { r } _ { j } , \tilde { s } _ { j } ^ { \prime } , \tilde { d } _ { j } ) \} _ { j = 1 } ^ { T } \stackrel { \mathrm { i i d } } { \sim } F _ { 0 }$ and computes prior stick weights $\begin{array} { r } { q _ { j } ^ { ( 0 ) } = V _ { j } ^ { \prime } \prod _ { l < j } ( 1 - V _ { l } ^ { \prime } ) } \end{array}$ with $V _ { j } ^ { \prime } \sim$ Beta(1, α) for $j = \mathrm { i } , \dots , \bar { T }$ . Because the warmstart phase is an iterative procedure approximating a single posterior draw, sampling the prior pseudo-atoms and stick weights once per episode instantiates a coherent, frozen Thompson sampling value hypothesis for the upcoming rollout.

Information Calibration via Bellman TD Surprise. In sparse-reward exploration, unguided initial rollouts inevitably accumulate thousands of redundant, unrewarded transitions. Standard Bayesian updating scaling with the raw bufer size $( n = | B | \sim 1 0 ^ { 5 } )$ would cause the prior weight to collapse prematurely $( \omega _ { \mathrm { p r i o r } }  0 )$ . From an information-theoretic standpoint, however, repeating predictable failure trajectories yields zero novel information about the environment. To prevent prior swamping, DP-DQN tracks an efective statistical sample count $N _ { \mathrm { s t a t } }$ calibrated online by the Bellman TD surprise:

$$
N _ { \mathrm { s t a t } }  N _ { \mathrm { s t a t } } + \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \operatorname* { m i n } ( 1 , \frac { | \delta _ { i } | } { \sigma _ { \mathrm { T D } } } ) , \quad \mathrm { w h e r e } \quad \delta _ { i } = y _ { i } - Q ( s _ { i } , a _ { i } ; \theta ) ,\tag{29}
$$

and $\sigma _ { \mathrm { T D } }$ denotes the running standard deviation of Bellman errors. Transitions with near-zero Bellman error $\left( \left| \delta _ { i } \right| \approx 0 \right)$ represent well-modeled dynamics that contribute negligible information, incrementing $N _ { \mathrm { s t a t } }$ by ≈ 0. Only when the agent encounters statistically novel transitions $\left( | \delta _ { i } | \geq \sigma _ { \mathrm { T D } } \right)$ does $N _ { \mathrm { s t a t } }$ increment toward 1.0.

Composite Stochastic Warmstart Updates. Across $w = 1 , \ldots , W$ warmstart steps, the online network is optimized against the episodic hypothesis using dynamic empirical mini-batches. At each step w, a fresh empirical mini-batch $\mathcal { D } _ { w } = \{ ( s _ { i } , a _ { i } , r _ { i } , s _ { i } ^ { \prime } , d _ { i } ) \} _ { i = 1 } ^ { B } \sim \mathcal { B }$ is sampled uniformly at random from the replay bufer. The total prior probability mass assigned to the synthetic base transitions is sampled as:

$$
W _ { \mathrm { p r i o r } } \sim \mathrm { B e t a } ( \alpha + 1 , N _ { \mathrm { s t a t } } ) ,\tag{30}
$$

yielding expected prior weight $\begin{array} { r } { \mathbb { E } [ W _ { \mathrm { p r i o r } } ] = \frac { \alpha + 1 } { \alpha + 1 + N _ { \mathrm { s t a t } } } } \end{array}$ . The remaining empirical probability mass $( 1 - W _ { \mathrm { p r i o r } } )$ is partitioned across the B mini-batch transitions via the multi-step stick-breaking recursion $\left( \mathrm { E q . ~ } \left( 8 \right) \right)$

$$
w _ { i } ^ { \mathrm { e m p } } = ( 1 - W _ { \mathrm { p r i o r } } ) \frac { \tilde { w } _ { i } } { \sum _ { k = 1 } ^ { B } \tilde { w } _ { k } } , \quad \mathrm { w h e r e } \quad \tilde { w } _ { i } = V _ { i } \prod _ { k = i + 1 } ^ { B } ( 1 - V _ { k } ) , \quad V _ { i } \sim \mathrm { B e t a } ( 1 , \alpha + i ) ,\tag{31}
$$

with $\tilde { w } _ { B } = V _ { B }$ , forming the discrete composite stick-breaking measure:

$$
P _ { w } = \sum _ { i = 1 } ^ { B } w _ { i } ^ { \mathrm { e m p } } \delta _ { z _ { i } } + W _ { \mathrm { p r i o r } } \sum _ { j = 1 } ^ { T } q _ { j } ^ { ( 0 ) } \delta _ { \tilde { z } _ { j } } .\tag{32}
$$

Minimizing the composite DP-Bellman loss ${ \mathcal { L } } _ { \mathrm { D P } } ( \theta )$ (Eq. (17)) via stochastic gradient descent updates online parameters θ, while $N _ { \mathrm { s t a t } }$ is calibrated online via Eq. (29).

Rollout with Frozen Policy. Upon completing the warmstart phase, the policy parameters are frozen for the rollout $( \theta _ { k } \gets \theta )$ , and the agent acts greedily $a _ { t } = \arg \operatorname* { m a x } _ { a } Q ( s _ { t } , a ; \theta _ { k } )$ across the entire rollout $t = 0 , \ldots , H - 1$ . Freezing the parameters enforces multi-step temporal consistency across the trajectory.

Theoretical Guarantees of the Warmstart Scheme. This dynamic procedure provides four critical theoretical properties:

• Preservation of Coherent Episodic Hypotheses: Generating prior pseudo-atoms $\{ \tilde { z } _ { j } \} _ { j = 1 } ^ { T }$ and prior stick weights $q _ { j } ^ { ( 0 ) }$ once before the warmstart ensures the network fits a single, coherent value hypothesis throughout all W optimization steps, preventing intra-episode prior drift.

• Anti-Swamping and Persistent Stochastic Optimism: Because $W _ { \mathrm { p r i o r } } \sim$ Beta $( \alpha + 1 , N _ { \mathrm { s t a t } } )$ , during unrewarded exploration rollouts where $N _ { \mathrm { s t a t } }$ remains small, the expected prior salience $\begin{array} { r } { \mathbb { E } [ W _ { \mathrm { p r i o r } } ] = \frac { \alpha + 1 } { \alpha + 1 + N _ { \mathrm { s t a t } } } = } \end{array}$ $\mathcal { O } ( 1 )$ remains permanently non-vanishing. The agent sustains consistent epistemic exploration drive regardless of the total replay bufer capacity $| B | \sim 1 0 ^ { 5 }$

• Unbiased $F u l l - B u f f e r$ Integration via Monte Carlo SGD: Because mini-batches $\mathcal { D } _ { w }$ are sampled uniformly from $B ,$ the expected empirical gradient is an unbiased Monte Carlo estimate over the full replay history B. The sequence of $W = \lceil H / 2 \rceil$ gradient updates propagates Bellman credit backward across the entire trajectory length without overfitting to any single static slice.

• Direct Closed-Form Stick-Breaking vs. MCMC: Unlike Markov Chain Monte Carlo (MCMC) methods that require iterative Markov chain transitions, burn-in periods, and acceptance-rejection evaluations, stickbreaking generates exact, independent random measures in closed form with negligible scalar computation $\mathcal { O } ( T + W \cdot B )$

## C.3 Bellman Error Propagation Depth and Choice of Warmstart Budget $W = \lceil H / 2 \rceil$

In episodic reinforcement learning over horizon H, temporal diference methods update value estimates via Bellman targets $\begin{array} { r } { y _ { i } = r _ { i } + ( 1 - d _ { i } ) \gamma \operatorname* { m a x } _ { a ^ { \prime } } Q ( s _ { i } ^ { \prime } , a ^ { \prime } ; \theta ^ { - } ) } \end{array}$ . Consequently, a single gradient step on a mini-batch propagates reward credit backward along an empirical trajectory chain by at most one temporal transition. In standard online Q-learning (such as Bootstrapped DQN; Osband et al. 2018), updates are executed every two environment steps, performing $\lfloor H / 2 \rfloor$ backpropagation updates distributed throughout the rollout. To ensure strict algorithmic fairness and computational parity, DP-DQN amortizes these updates into $W = \lceil H / 2 \rceil$ warmstart gradient steps at the episode boundary. Crucially, choosing $W \ge H / 2$ provides the exact gradient depth necessary for Bellman error signals to propagate backward across the full trajectory length $H _ { \ast }$ , while freezing $\theta _ { k }$ during the subsequent rollout guarantees complete multi-step temporal consistency—the foundational requirement of pure Thompson sampling.

## D Stochastic Optimism and Deep Exploration: DP-BNN vs. Ensembles vs. BDQN

A foundational question in Bayesian reinforcement learning is whether exploration requires explicit, deterministic optimism (such as heuristic upper confidence bounds Q + cσ or engineered reward bonuses) or whether it arises naturally from posterior sampling (Osband et al., 2013; Osband & Van Roy, 2017). In this section, we formalize the theory of stochastic optimism, contrast the exploratory mechanisms of randomized prior ensembles (BootDQN-RP) and neural-linear Bayesian Deep Q-Networks (BDQN) against DP-BNNs, and provide empirical ablations across prior base measure geometries.

## D.1 The Principle of Stochastic Optimism in Deep Exploration

To make the distinction between exploration paradigms mathematically precise, we define both deterministic and stochastic optimism:

Definition 1 (Deterministic Optimism (Brafman & Tennenholtz, 2002; Auer et al., 2008)). A value-function estimator $\widetilde { Q } : { \cal S } \times { \cal A }  \mathbb { R }$ satisfies deterministic optimism if, for every state-action pair $( s , a ) \in \mathcal S \times \mathcal A$ , the estimate dominates the optimal action-value function:

$$
\widetilde { Q } ( s , a ) \geq Q ^ { * } ( s , a ) w i t h \ p r o b a b i l i t y \ 1 \ ( o r \ w i t h \ h i g h \ p r o b a b i l i t y \ \geq 1 - \delta ) .\tag{33}
$$

Deterministic optimism is commonly realized by adding an explicit upper-confidence bound $\widetilde Q ( s , a ) = \widehat Q ( s , a ) +$ $\beta \sigma ( s , a ) \ ( e . g .$ ., UCRL2) or initializing unvisited values to the theoretical maximum ${ \frac { R _ { \mathrm { m a x } } } { 1 - \gamma } } \ ( e . g . , \ \dot { R } { - } M \dot { A } \dot { X } )$

While deterministic optimism guarantees polynomial regret in small, tabular MDPs, it enforces an exhaustive exploration pattern: the agent is required to visit every suboptimal branch until its upper confidence bound drops below the true value of optimal trajectories, leading to severe sample ineficiencies in deep horizons (Osband & Van Roy, 2017). Randomized posterior sampling resolves this bottleneck via stochastic optimism:

Definition 2 (Stochastic Optimism in Posterior Sampling (Osband et al., 2013; Osband & Van Roy, 2017; Osband et al., 2019b)). Let $\mathcal { P } ( \cdot \mid B )$ denote the posterior distribution over action-value functions conditioned on the empirical replay history B. The posterior distribution $\mathcal { P } ( \cdot \mid B )$ satisfies stochastic optimism with parameter $p _ { 0 } \in ( 0 , 1 ]$ if, for any unvisited state $s \in S$ , a sampled value function $\tilde { Q } \sim \mathcal { P } ( \cdot | \ B )$ dominates the optimal value function with strictly positive probability:

$$
\mathbb { P } _ { \tilde { Q } \sim \mathcal { P } ( \cdot | \mathcal { B } ) } \left( \operatorname* { m a x } _ { a \in \mathcal { A } } \tilde { Q } ( s , a ) \ge V ^ { * } ( s ) \right) \ge p _ { 0 } > 0 .\tag{34}
$$

Unlike local dithering (ϵ-greedy), an action-value function sampled from a functional posterior is temporally consistent across the entire episode horizon H. When a sample satisfies max ${ } _ { a } \tilde { Q } ( s , a ) \geq V ^ { * } ( s )$ along an unexplored path, the agent commits to following that path for the full duration of the rollout. Furthermore, under the DP posterior mixture, the weight allocated to the prior base measure $F _ { 0 }$ contracts as $\frac { \alpha } { \alpha + | \boldsymbol { B } | }$ Consequently, unexplored transitions maintain high prior variance (sustaining stochastic optimism), while frequently visited transitions collapse to their true empirical targets.

Practical Construction: Knowledge of $R _ { \mathrm { m a x } }$ Sufices for Stochastic Optimism. In standard reinforcement learning theory (Brafman & Tennenholtz, 2002; Strehl & Littman, 2008; Osband et al., 2013), the maximum achievable reward $R _ { \mathrm { m a x } }$ (or the reward support $[ 0 , R _ { \mathrm { m a x } } ] )$ is universally assumed to be known or normalized. Crucially, to construct a stochastically optimistic base measure that solves deep exploration across horizons evaluated up to $N = 5 0$ , the practitioner requires no knowledge of transition graphs, optimal trajectories, scrambled action mappings, or goal locations; knowledge of the scalar bound $R _ { \mathrm { m a x } }$ alone is completely suficient. The standard uninformative prior over bounded rewards in $[ 0 , R _ { \mathrm { m a x } } ]$ is simply the uniform distribution $\mathcal { U } ( 0 , R _ { \mathrm { m a x } } )$ . This guarantees that unexplored state-action pairs maintain prior variance with positive expectation $\mathbb { E } [ r ] = R _ { \operatorname* { m a x } } / 2 > 0$ , naturally satisfying stochastic optimism $\begin{array} { r } { ( \mathbb { P } ( \operatorname* { m a x } _ { a } \tilde { Q } ( s , a ) \ge V ^ { * } ( s ) ) > 0 ) } \end{array}$ (Agrawal & Goyal, 2012; Agrawal & Jia, 2017) without requiring handengineered intrinsic bonuses, count-based density estimators, or environment-specific heuristics.

## D.2 Attaining Stochastic Optimism with Zero-Mean Normal Priors via Order Statistics

Remarkably, even when the base measure possesses an entirely zero-mean symmetric distribution like the Standard Normal $\mathcal { N } ( 0 , \sigma ^ { 2 } ) ~ ( \mathbb { E } [ r ] = 0 )$ , stochastic optimism can be naturally realized through extreme value order statistics in multi-sample posterior evaluation. If an agent evaluates M candidate functional samples $\{ X _ { 1 } , \ldots , X _ { M } \} \stackrel { \mathrm { i . i . d . } } { \sim } { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ , the expected maximum among these samples—the top order statistic $X _ { ( M ) } = \operatorname* { m a x } _ { 1 \leq i \leq M } X _ { \ast }$ <sub>i</sub>—is strictly positive and scales as ${ \mathbb E } [ X _ { ( M ) } ] \approx \sigma \sqrt { 2 \ln M }$ . Through Monte Carlo simulation $\left( 1 0 ^ { 6 } \mathrm { \ t r i a l s } \right)$ , we observe the exact upward shift in efective prior optimism:
<table><tr><td>Candidate Samples (M)</td><td> $M = 1$ </td><td> $M = 2$ </td><td> $M = 4$ </td><td> $M = 8$ </td><td> $M = 1 0$ </td><td> $M = 2 0$ </td><td> $M = 5 0$ </td></tr><tr><td>Simulated  $\mathbb { E } [ \operatorname* { m a x } _ { i } X _ { i } ]$ </td><td>0.00σ</td><td>+0.56σ</td><td>+1.03σ</td><td> $+ 1 . 4 2 \sigma$ </td><td> $+ \mathbf { 1 . 5 4 } \sigma$ </td><td> $+ 1 . 8 7 \sigma$ </td><td> $+ 2 . 2 5 \sigma$ </td></tr><tr><td>Empirical Std. Dev.</td><td>1.00σ</td><td>0.83σ</td><td>0.70σ</td><td>0.61σ</td><td>0.59σ</td><td>0.53σ</td><td>0.46σ</td></tr><tr><td>Theoretical √2 ln M</td><td>0.00</td><td>1.18</td><td>1.67</td><td>2.04</td><td>2.15</td><td>2.45</td><td>2.80</td></tr></table>

As shown in the table, drawing $M \ = \ 1 0$ candidate functional evaluations in Multi-Sample DP-DQN automatically shifts the efective exploration frontier upward by +1.54 σ. Consequently, multi-sample posterior sampling inherently converts symmetric, zero-mean functional variance into efective stochastic optimism without requiring manual reward inflation or multiple physical ensemble networks.

To empirically validate this principle on a deep exploration task, we benchmarked this order-statistic base measure directly on DeepSea-20 across random seeds using a standardized single MLP-20. Under a plain zero-mean Standard Normal prior $( M = 1 , \mathbb { E } [ r ] = 0 . 0 )$ , the agent collapses into boundary failure, achieving a 0% solve rate (never discovering the terminal goal). In sharp contrast, simply setting $M = 1 0$ (evaluating the top order statistic $r = \mathrm { m a x } _ { 1 \leq i \leq 1 0 } X _ { i }$ from $X _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 ) )$ achieves a 100% solve rate across all seeds $( 6 1 2 . 0 { \pm } 1 5 3 . 2 $ episodes, first discovered at Episode $5 5 2 . 0 { \pm } 1 5 4 . 5 )$ . This demonstrates that even when individual prior draws are strictly zero-mean Gaussians, order-statistic multi-sampling naturally generates the requisite +1.54σ stochastic optimism to overcome continuous step costs and solve DeepSea-20 without modifying the single-network architecture.

## D.3 Comparative Exploration Mechanisms: Ensembles vs. BDQN vs. DP-BNN

How Bootstrapped DQN with Randomized Priors Accomplishes Stochastic Optimism. Bootstrapped DQN with Randomized Prior Functions (BootDQN-RP) (Osband et al., 2018) achieves stochastic optimism by decomposing each ensemble head $k \in \{ 1 , \ldots , K \}$ additively:

$$
Q _ { k } ( s , a ; \theta _ { k } ) = f ( s , a ; \theta _ { k } ) + \beta \cdot p ( s , a ; \tilde { \theta } _ { k } ) ,\tag{35}
$$

where $f ( \cdot ; \theta _ { k } )$ is a trainable neural network, $p ( \cdot ; \tilde { \theta } _ { k } )$ is a fixed, non-trainable randomized prior network with parameters $\smash { \widetilde { \theta } _ { k } \sim \mathcal { N } ( 0 , \sigma _ { p } ^ { 2 } I ) }$ , and $\beta > 0$ scales prior influence. On unvisited states where $f \approx 0 ,$ predictions are dominated by $\beta p _ { k }$ . Across K heads, extreme value properties of independent Gaussians guarantee E[max<sub>k</sub> $\beta p _ { k } ] \approx \beta \sigma _ { \mathrm { f u n c } } \sqrt { 2 \ln K } > 0 ,$ ensuring at least one head is optimistic. On visited states, empirical loss drives $f  Q ^ { * } - \beta p _ { k }$ , canceling the prior on data support. However, BootDQN-RP requires maintaining K physical neural networks and target networks $( { \mathcal { O } } ( K )$ compute and memory), and small discrete ensembles can sufer from bootstrap head collapse on deceptive tasks if all K heads prematurely sample suboptimal paths.

Why Neural-Linear Bayesian Deep Q-Networks (BDQN) Fail. While parameter-space Bayesian neural networks—such as Mean-Field Variational Inference (Blundell et al., 2015), Laplace approximations (MacKay, 2003), and MC-Dropout (Gal & Ghahramani, 2016; Riquelme et al., 2018)—broadly struggle in deep Q-learning due to well-documented variance collapse in unobserved regions (Foong et al., 2019) and non-stationary Bellman targets, the canonical single-network architecture for Thompson sampling is the neural-linear Bayesian Deep Q-Network (BDQN) (Azizzadenesheli et al., 2018). BDQN splits the action-value function into a deterministic feature extractor $\phi _ { \psi } ( s ) \in \mathbb { R } ^ { p }$ followed by Bayesian linear regression over finallayer weights $w _ { a } \mid \boldsymbol { \mathcal { B } } \sim \mathcal { N } ( \mu _ { a } , \sigma ^ { 2 } \Sigma _ { a } )$ with precision $\begin{array} { r } { \Sigma _ { a } ^ { - 1 } = \Sigma _ { 0 } ^ { - 1 } + \frac { 1 } { \sigma ^ { 2 } } \sum _ { i : a _ { i } = a } \phi _ { \psi } ( s _ { i } ) \phi _ { \psi } ( s _ { i } ) ^ { \top } } \end{array}$ . Action values are sampled via $\tilde { Q } ( s , a ) = \tilde { w } _ { a } ^ { \top } \phi _ { \psi } ( s )$

Despite its lightweight design, BDQN sufers from linear regret on deep exploration tasks (incurring ${ 9 , 9 5 9 \pm 5 }$ regret on DeepSea-20, identical to vanilla DQN; Figure 2a). This limitation stems from three structural factors:

1. Deep Feature Collapse on Unvisited States. Because Bayesian uncertainty is confined strictly to the linear readout layer, the feature backbone $\phi _ { \psi } ( s )$ is trained deterministically on sparse, unrewarded failure data. It maps unexplored states to degenerate representations $\big ( \phi _ { \psi } \big ( s _ { \mathrm { u n v i s i t e d } } \big ) \approx \mathbf { 0 } \big )$ , causing predictive variance $\sigma ^ { 2 } ( s , a ) = \phi _ { \psi } ( s ) ^ { \top } \Sigma _ { a } \phi _ { \psi } ( s )  0$ to extinguish across the unvisited frontier regardless of the final-layer prior.

2. Non-Stationary Covariates Corrupt Posterior Updates. Exact Bayesian regression requires stationary inputs. In BDQN, $\phi _ { \psi _ { t } } ( s )$ changes with every gradient step, meaning the precision matrix $\textstyle \sum _ { a } ^ { - 1 }$ accumulates stale outer products evaluated under obsolete parameter snapshots $\psi _ { \tau } \neq \psi _ { t }$ , invalidating posterior calibration.

3. Subspace Projection and Epistemic Shrinkage. Redundant rollouts in absorbing early states inflate eigenvalue mass along dominant feature directions. Because deep layers share weights, unexplored states project onto these empirical directions, causing $\Sigma _ { a }$ to aggressively shrink predictive variance and pulling sampled values toward pessimistic empirical means $( \tilde { Q } \approx \mu _ { a } ^ { \top } \phi \leq 0 )$ . This violates stochastic optimism $( \mathbb { P } ( \tilde { Q } \geq Q ^ { * } )  0 )$ and halts exploration.

How DP-BNNs Overcome BDQN and Ensemble Methods. DP-BNNs overcome the structural limitations of neural-linear BDQN by placing the DP prior directly on the data-generating distribution in experience space $( s , a , r , s ^ { \prime } , d )$ , rather than on a linear final layer or parametric weights. The entire deep neural network is optimized end-to-end against the nonparametrically sampled measure $P _ { w } .$ Prior pseudo-transitions $\{ \tilde { z } _ { j } \} _ { j = 1 } ^ { T } \sim F _ { 0 }$ provide global, non-linear function-space exploration across the full depth of the network, preventing feature collapse. By freezing the sampled value hypothesis $\theta _ { k }$ for each rollout, DP-DQN guarantees multi-step temporal consistency across the entire trajectory with a single unified neural network, achieving the deep exploration of an ensemble at the lightweight computational cost of standard DQN.

## D.4 Empirical Investigation Across Prior Base Measures

To empirically verify this principle and confirm that DP-DQN does not rely on hand-engineered optimistic goal priors, we conducted a systematic ablation across three distinct reward base measure distributions on both canonical RiverSwim-6 $( N = 6 , H = 5 0 )$ and DeepSea-10 $( N = 1 0 )$ using identical single MLP-20 architectures:

Uniform Prior $( \mathcal { U } ( 0 , 1 ) )$ . Draws reward atoms uniformly across the support [0, 1] for all state-action pairs without encoding any target location coordinates (prior mean $\mu = + 0 . 5 )$

Zero-Mean Standard Normal Prior $( \mathcal { N } ( 0 , 1 ) )$ . Uninformative prior with zero expected return everywhere $( \mu = 0 . 0 , \sigma = 1 . 0 )$ , providing zero target bias or privileged goal information.

Pessimistic Prior $( \mathcal { N } ( - 1 . 0 , 1 . 0 ) )$ . Assumes all unexplored transitions yield negative reward on average $( \mu = - 1 . 0 , \sigma = 1 . 0 )$ , testing the behavior under severe risk aversion.

As illustrated in Figure 4(a), on canonical RiverSwim-6, both the zero-mean standard normal prior $( \mu = 0 . 0 )$ and the uniform prior $( \mu = + 0 . 5 )$ achieve a 100% solve rate, reaching the upstream jackpot in $2 7 . 4 \pm 8 . 1$ and $3 3 . 4 \pm 1 4 . 8$ episodes respectively, and accumulating over 450 cumulative return. Crucially, neither prior contains any inductive bias favoring the rightward jackpot. Exploration succeeds because prior variance $( \sigma > 0 )$ generates stochastically optimistic value samples that propagate backward along the chain via Bellman updates. Conversely, under an explicitly pessimistic prior $( \mu = - 1 . 0 )$ , the agent believes unexplored upstream transitions yield negative rewards $( \mathbb { E } [ r ] = - 1 . 0 )$ . Because the local downstream trap yields $+ 0 . 0 5 > - 1 . 0$ the agent evaluates the known downstream trap as strictly superior to the unexplored river, becoming permanently trapped in 80% of seeds.

![](images/62da1a6621ad13bfc2933e4c4ec78a5b533c0ea38f35f2f5c7f31c4186b5edd4.jpg)

![](images/7a2bd3d63d55723cbceb9bc7abee2247894e5d46ce8f6508d9670bf778bb78c7.jpg)  
Figure 4: Stochastic optimism across base measures (MLP-20 with LayerNorm). (a) Cumulative return on RiverSwim-6: uniform $\mathcal { U } ( 0 , 1 )$ and standard normal $\mathcal { N } ( 0 , 1 )$ solve the task across 100% of seeds, while a pessimistic prior $\mathcal { N } ( - 1 , 1 )$ traps 80% of runs. (b) Cumulative regret on DeepSea-10: optimistic and zero-mean priors discover the goal without privileged target coordinates, whereas a pessimistic prior causes boundary collapse and linear regret.

Figure 4(b) illustrates the critical distinction between zero-mean perturbations and principled stochastic optimism on DeepSea. In DeepSea, traversing the optimal diagonal incurs an immediate per-step penalty of $- 0 . 0 1 / N$ at every step. On small horizons $( N = 1 0 )$ , zero-mean standard normal priors $\mathcal { N } ( 0 , 1 )$ can occasionally discover the terminal goal (Episode 276). However, as the horizon scales to $N \geq 2 0$ , zero-mean random fluctuations cannot sustain 20 to 50 consecutive coordinated steps against the continuous step drag, collapsing into boundary failure.

In sharp contrast, the uniform prior $\mathcal { U } ( 0 , R _ { \mathrm { m a x } } )$ provides the strictly necessary stochastic optimism: because its expected reward is $\mathbb { E } [ r ] = R _ { \operatorname* { m a x } } / 2 = + 0 . 5 \gg - 0 . 0 1 / N$ , unexplored transitions maintain suficient positive value to decisively overcome the local step penalty, enabling $\mathrm { D P - D Q N }$ to scale seamlessly from $N = 1 0$ all the way to $N = 5 0$ . Conversely, under an explicitly pessimistic prior $( \mu = - 1 . 0 )$ , the agent fears the step penalty, immediately cascades to the left boundary, and sufers linear regret across all episodes.

## E Comprehensive Empirical Benchmark Suite and Implementation Details

To evaluate the generality, scalability, and robustness of DP-BNNs across diverse decision-making tasks, we benchmark DP-DQN against competitive baselines across diverse testbeds: deceptive exploration in N-Chain, deep exploration in canonical RiverSwim-6, and contextual bandit exploration on the Wheel Bandit benchmark (Riquelme et al., 2018) (with the large-scale DeepSea scaling suite presented in the main text, Section 5 and Figure 2). A consolidated summary is presented in Table 1.

Table 1: Consolidated empirical evaluation across diverse decision-making and exploration benchmarks (mean ± standard error across seeds). All DP-BNN results are obtained using a single neural network (single MLP-20 with LayerNorm for DeepSea-20, RiverSwim-6, and Deceptive N-Chain; 2-layer $6 4 \times 6 4$ MLP with LayerNorm for DeepSea-50 and MountainCar-v0; 3-layer MLP-64 for contextual bandits).
<table><tr><td>Benchmark</td><td>Metric</td><td>Vanilla / Baseline</td><td>BootDQN-RP / PSRL</td><td>DP-BNN (Ours)</td></tr><tr><td>DeepSea N = 20</td><td>Episodes to Solve</td><td> $\mathrm { > 2 0 , 0 0 0 ~ ( D Q N ) }$ </td><td> $5 , 2 1 0 \pm 6 0 4$ </td><td> ${ \bf 6 0 2 \pm 1 0 4 }$ </td></tr><tr><td>DeepSea  $N = 5 0 \ ( 6 4 \times 6 4 \ \mathrm { M L P } )$ </td><td>Episodes to Solve</td><td> $> 1 0 ^ { 1 5 } ~ \mathrm { ( D Q N ) }$ </td><td> $> 2 0 { , } 0 0 0$ </td><td> $\mathbf { 1 , 9 2 5 \pm 1 8 4 }$ </td></tr><tr><td>RiverSwim-6  $( \mathcal { N } ( 0 , 1 ) \ \mathrm { P r i o r } )$ </td><td>Episodes to Solve</td><td> $4 4 . 0 \pm 1 5 . 2 \ \mathrm { ( D Q N ) }$ </td><td> $7 . 0 \pm 1 . 4 ~ \mathrm { ( P S R L ) }$ </td><td> ${ \bf 2 7 . 4 \pm 8 . 1 }$ </td></tr><tr><td>Deceptive N-Chain  $( N = 1 0 )$ </td><td>Cum. Regret</td><td> $1 , 4 2 0 . 5 \pm 1 1 2 . 0 \ \mathrm { ( D Q N ) }$ </td><td> $6 8 1 . 4 \pm 9 2 . 6$ </td><td> ${ \bf 1 5 3 . 2 \pm 1 8 . 4 }$ </td></tr><tr><td>Wheel Bandit  $( \delta = 0 . 9 )$ </td><td>1.5k-Step Regret</td><td> $6 , 8 9 4 . 6 \pm 1 , 5 1 4 . 6 ~ ( \epsilon \mathrm { G r . } )$ </td><td> $4 { , } 3 0 9 { . } 4 \pm 8 2 9 { . } 2$ </td><td> $\mathbf { 2 , 8 0 3 . 3 \pm 1 8 9 . 6 }$ </td></tr><tr><td>MountainCar-v0 (Continuous)</td><td>Episodes to Solve</td><td> $8 5 \pm 1 2 ~ \mathrm { ( D Q N ) }$ </td><td> $1 5 \pm 3 \ \mathrm { ( B o o t D Q N ) }$ </td><td> ${ \bf 1 6 . 8 \pm 7 . 4 }$ </td></tr></table>

![](images/e35361932b44c3c4a40e8521624463972504e676a02d33c379a002551c276166.jpg)  
Figure 5: Architectural schematics and Markov Decision Process (MDP) graphs for the canonical deep exploration benchmarks evaluated in Appendix E. (a) DeepSea (N × N DAG): An agent traverses from (0, 0) across N stages. Taking the designated rightward action at every row reaches the jackpot goal at $( N - 1 , N - 1 )$ with reward +1.0 (paying a step penalty −0.01/N), whereas any leftward step permanently traps the agent in zero-reward branches. An unguided random policy discovers the goal with probability $2 ^ { - N }$ $( 9 . 5 \times 1 0 ^ { - 7 }$ for N = 20; $8 . 8 \times 1 0 ^ { - 1 6 }$ for $N = 5 0 )$ , requiring temporally consistent value function posterior samples. (b) RiverSwim-6: A 6-state chain where swimming downstream (a<sub>0</sub>) is deterministic and trapped by an immediate sub-optimal reward $r = 0 . 0 5$ at s<sub>0</sub>, while swimming upstream (a<sub>1</sub>) fights against a strong current $( P ( s + 1 \mid s , a _ { 1 } ) = 0 . 6 ,$ , P(s | s, a<sub>1</sub>) = 0.3, $P ( s - 1 \mid s , a _ { 1 } ) = 0 . 1 )$ to reach the distal goal at $s _ { 5 } ~ ( r = 1 . 0 )$ (c) Deceptive N-Chain: A sequential corridor where choosing a<sub>0</sub> at $s _ { 0 }$ yields an immediate lure reward $( r = 0 . 0 1 )$ that prematurely terminates the rollout. Only persistent, multi-step posterior exploration traverses the silent zero-reward corridor $\left( s _ { 1 } \to \cdots \to s _ { N - 1 } \right)$ to unlock the terminal jackpot $( r = 1 . 0 )$

DeepSea Scaling Protocol (N × N Exploration DAG). The empirical scaling and sample complexity results for DeepSea (N = 10 to $N = 5 0 )$ are presented in the main text (Section 5, Figure 2) and illustrated in Figure 5a. Formally, DeepSea is defined as a directed acyclic graph (DAG) MDP with state space $S = \{ ( h , c ) : 0 \leq h < N , 0 \leq c \leq h \}$ representing vertical depth h and horizontal position $^ { c , }$ with initial state $s _ { 0 } = ( 0 , 0 )$ and episode length N. At each time step h, the agent chooses between two discrete actions $\mathcal { A } = \{ 0 , 1 \}$ . Under the standard bsuite protocol (Osband et al., 2019a), an unknown row-specific action mapping hides the true rightward direction behind an independent random permutation per row, preventing heuristic action bias. Stepping rightward advances the agent to $( h + 1 , c + 1 )$ and incurs a minor deterministic step penalty of $- 0 . 0 1 / N ;$ ; stepping leftward transitions to $( h + 1 , \operatorname* { m a x } ( 0 , c - 1 ) )$ with zero penalty. The environment provides a single positive terminal reward $r = + 1 . 0$ located exclusively at the bottom-right corner $( N - 1 , N - 1 )$ , which can only be reached if the agent takes the designated rightward action at every single step along the trajectory. Any single leftward deviation irreversibly commits the agent to zero-reward terminal leaves. Consequently, a uniform random policy discovers the goal with probability exactly $( 1 / 2 ) ^ { N } = 2 ^ { - N }$ $( 9 . 5 \times 1 0 ^ { - 7 }$ for $N = 2 0$ , and $8 . 8 \times 1 0 ^ { - 1 6 }$ for $N = 5 0 )$ , rendering myopic dithering methods (such as ϵ-greedy and standard DQN) asymptotically incapable of exploration $( \mathcal { O } ( 2 ^ { N } )$ sample complexity). A run is recorded as solved at the earliest episode where the rolling average return over 20 consecutive episodes exceeds 0.9 (matching optimal return $1 . 0 - 0 . 0 1 = 0 . 9 9 )$ , after which cumulative regret flatlines. In contrast, DP-DQN’s non-parametric DP prior generates temporally correlated, optimistic value functions that drive directed exploration straight to the optimal terminal goal, solving DeepSea-20 in $6 0 2 \pm 1 0 4$ episodes and DeepSea-50 in $1 , 9 2 5 \pm 1 8 4$ episodes (Table 1).

Deep Exploration on Canonical RiverSwim-6 (Uninformative Prior). The RiverSwim environment (Strehl & Littman, 2008; Osband et al., 2013) tests an agent’s ability to overcome an immediate sub-optimal reward trap in the presence of strong environmental stochasticity (Figure 5b). The MDP consists of $N = 6$ linearly arranged states $S = \{ 0 , 1 , 2 , 3 , 4 , 5 \}$ , starting at the downstream state $s _ { 0 } = 0$ with horizon $H = 5 0$ . The agent has two actions $\mathcal { A } = \{ a _ { 0 } , a _ { 1 } \}$ , where $a _ { 0 }$ corresponds to swimming downstream (with the current) and $a _ { 1 }$ corresponds to swimming upstream (against the current). Downstream swimming is deterministic: taking $a _ { 0 }$ at state $s > 0$ transitions to $s - 1$ with probability 1.0, while taking $a _ { 0 }$ at $s _ { 0 }$ stays at $s _ { 0 }$ and yields an immediate deterministic reward $r ( 0 , a _ { 0 } ) = 0 . 0 5$ . In contrast, swimming upstream against the current is highly stochastic: from any intermediate state $s \in \{ 1 , 2 , 3 , 4 \}$ , taking $a _ { 1 }$ successfully advances to $s + 1$ with probability 0.6, remains at s with probability $0 . 3 .$ , and slips downstream to $s - 1$ with probability 0.1. $\mathrm { A t } \ s _ { 0 } , \ a _ { 1 }$ transitions to $s _ { 1 }$ with probability 0.6 and remains at $s _ { 0 }$ with probability 0.4. At the upstream terminal state $s _ { 5 }$ , swimming right succeeds with probability 0.6 and yields the maximal terminal reward $r ( 5 , a _ { 1 } ) = 1 . 0$ (falling back to $s _ { 4 }$ with probability 0.4). The minor reward at $s _ { 0 }$ acts as an exploration trap: because $r = 0 . 0 5$ is immediately accessible, standard ϵ-greedy exploration quickly reinforces the myopic policy of idling at s<sub>0</sub> and fails to reach $s _ { 5 }$ . For DP-DQN, the factorized base measure $F _ { 0 } ( s , a , r , s ^ { \prime } , d ) = F _ { 0 s } ( s ) \times F _ { 0 a } ( a ) \times F _ { 0 r } ( r ) \times F _ { 0 s ^ { \prime } } ( s ^ { \prime } ) \times F _ { 0 d } ( d )$ samples discrete states uniformly over the chain $\tilde { s } , \tilde { s } ^ { \prime } \sim \mathrm { U n i f o r m } ( \{ 0 , \dots , 5 \} )$ , actions a˜ ∼ Uniform $\left( \left\{ a _ { 0 } , a _ { 1 } \right\} \right)$ , prior rewards from an uninformative standard normal distribution $\tilde { r } \sim \mathcal { N } ( 0 , 1 . 0 )$ everywhere (providing zero target bias or privileged goal information), and continuing terminal flag $d = 0$ . Driven purely by functional posterior variance via stick-breaking draws, single-network DP-DQN (single MLP-20 with LayerNorm) solves RiverSwim-6 across 100% of random seeds in only $2 7 . 4 \pm 8 . 1$ episodes, achieving a cumulative return of $4 7 4 . 2 \pm 6 8 . 2$ (Table 1).

Deceptive Exploration on N-Chain. In the deceptive N-Chain benchmark $( N = 1 0 ,$ episode length $N ;$ Figure 5c), an agent must traverse a sequential corridor $ { \mathcal { S } } = \{ 0 , 1 , \ldots , N - 1 \}$ from start state $s _ { 0 } = 0$ . At state $s _ { 0 } .$ , choosing action $a _ { 0 }$ yields an immediate minor lure reward $( r = 0 . 0 1 )$ but prematurely terminates the episode (or resets to $s _ { 0 } )$ . Choosing action $a _ { 1 }$ advances to state $s _ { 1 }$ with zero intermediate reward $( r = 0 )$ . At intermediate states $s \in \{ 1 , \ldots , N - 2 \}$ , choosing $a _ { 1 }$ advances to $s + 1 \ ( r = 0 )$ , while choosing $a _ { 0 }$ slips backward to $s - 1 \ ( r = 0 )$ . Only upon successfully traversing all intermediate states to reach $s N - 1$ does choosing $a _ { 1 }$ yield the maximal terminal reward $( r = 1 . 0 )$ . Standard exploration algorithms sufer severe deception because the immediate reward at $s _ { 0 }$ reinforces the greedy policy of exiting immediately. Furthermore, ensemble methods such as Bootstrapped DQN with randomized priors (Osband et al., 2018) can sufer from bootstrap head collapse when all ensemble heads prematurely fixate on the sub-optimal left branch. In contrast, DP-DQN equips the single network with an exploratory DP base measure on transitions: states are sampled uniformly $\tilde { s } \sim$ Uniform $( \{ 0 , \ldots , N - 1 \} )$ , actions a˜ ∼ Uniform $\left( \left\{ a _ { 0 } , a _ { 1 } \right\} \right)$ , and next states $\tilde { s } ^ { \prime } = \operatorname* { m i n } ( \tilde { s } + 1 , N - 1 )$ for $\tilde { a } = 1$ (and max $( \tilde { s } - 1 , 0 )$ for $\tilde { a } = 0 )$ , with prior rewards $\tilde { r } \sim \mathcal { N } ( 1 . 0 , 0 . 2 ^ { 2 } )$ for $a = 1$ and $\mathcal { N } ( 0 . 0 , 0 . 1 ^ { 2 } )$ for $a = 0$ . This maintains global functional support across value functions and discovers the optimal rightward policy on Episode 1 across all seeds (100% discovery rate), reducing cumulative regret by 77.5% relative to BootDQN-RP $( 1 5 3 . 2 \pm 1 8 . 4 $ vs. $6 8 1 . 4 \pm 9 2 . 6 ;$ Table 1).

Deep Contextual Bandits on Wheel Bandit $( \delta = 0 . 9 )$ : Geometry and Exploration Trap. We benchmarked DP-BNN on the canonical exploration challenge of the Deep Bayesian Bandit suite (Riquelme et al., 2018): the Wheel Bandit $( d = 2 , K = 5 .$ exploration parameter $\delta = 0 . 9 ;$ Figure 6a). In this benchmark, context vectors $\boldsymbol { x } = ( x _ { 1 } , x _ { 2 } )$ are sampled uniformly from the 2D unit disk $\mathcal { X } = \{ x \in \mathbb { R } ^ { 2 } : \| x \| _ { 2 } \leq 1 \}$ . The environment provides a safe base arm $( a = 0 )$ that yields a deterministic reward of $r = 1 . 0$ across the entire domain. The remaining four arms $a \in \{ 1 , 2 , 3 , 4 \}$ correspond to the four geometric quadrants of the plane $\left( Q _ { 1 } \ \mathrm { t o } \ Q _ { 4 } \right)$ . If a context falls in the interior disk $\| x \| _ { 2 } \leq \delta ,$ arms 1..4 yield zero reward $( r = 0 . 0 )$ . However, if the context falls in the outer rim $\| { \boldsymbol x } \| _ { 2 } > \delta$ and lies in quadrant $Q _ { a } ,$ , pulling arm a yields a high-payof reward of $r = 5 0 . 0$ (while the other three arms yield 0.0).

The Wheel Bandit represents an extreme exploration bottleneck because the safe interior disk occupies $\delta ^ { 2 } = 0 . 9 ^ { 2 } =$ 81% of the total context probability measure, whereas the high-reward outer rim $\mathcal { X } _ { \mathrm { r a r e } } = \{ \boldsymbol { x } :$ $\| x \| _ { 2 } > 0 . 9 \}$ appears only $1 - \delta ^ { 2 } = 1 9 \%$ of the time (≈ 4.75% probability mass per optimal quadrant). Because 81% of early interactions occur in the interior where arms 1..4 yield zero return, standard exploration schemes (such as ϵ-greedy, Deep Ensembles, and Neural-Linear) quickly observe that arm 0 is superior to arms 1..4 in the bulk of the domain. Lacking principled functional uncertainty in rare regions, their predictive variance over arms 1..4 collapses; consequently, when a rare context on the boundary $\Vert { x } \Vert _ { 2 } > 0 . 9$ finally arrives, the agent fails to explore and greedily pulls arm $0 \ ( r = 1 . 0 )$ , incurring catastrophic linear regret $\Omega ( T )$ (Figure 6b).

To solve this task, DP-BNN specifies a factorized DP base measure $F _ { 0 } ( x , a , r ) = F _ { 0 x } ( x ) \times F _ { 0 a } ( a ) \times F _ { 0 r } ( r )$ where context vectors are sampled uniformly from the unit disk $\tilde { x } = \left( \sqrt { u } \cos \phi , \sqrt { u } \sin \phi \right) \left( u \sim \mathcal { U } ( 0 , 1 ) , \phi \sim \mathcal { U } ( 0 , 2 \pi ) \right)$ , arm choices are uniform $\tilde { a } \sim$ Uniform({0, 1, 2, 3, 4}), and prior rewards follow an optimistic Gaussian distribution $\tilde { r } \sim \mathcal { N } ( 1 . 0 , 0 . 5 ^ { 2 } )$ (Table 2). By pairing empirical replay reweighting with synthetic prior pseudo-observations across the full unit disk, DP-BNN sustains functional epistemic variance on the rare outer boundary without collapsing into the interior trap. As shown in Figure 6b, single-network DP-BNN achieves the lowest cumulative regret across all evaluated methods $( 2 , 8 0 3 . 3 \pm 1 8 9 . 6 $ over 1,500 steps across 5 seeds), substantially outperforming Bootstrapped NN with randomized priors $( 4 , 3 0 9 . 4 \pm 8 2 9 . 2 )$ , ϵ-greedy $( 6 , 8 9 4 . 6 \pm 1 , 5 1 4 . 6 )$ , Deep Ensembles $( 7 , 9 9 0 . 0 \pm 1 , 6 8 9 . 7 )$ , and Neural-Linear $( 9 , 5 9 2 . 3 \pm 8 6 6 . 2 )$ . Crucially, as shown in Figure 6c, DP-BNN achieves this performance using a single neural network (1× parameter memory), whereas Bootstrapped NN with randomized priors requires 20 total networks (10 active + 10 prior networks, 20× memory) and Deep Ensembles requires 5 networks (5× memory).

![](images/a229dd9cac8ac75b2dddb18d53f5f7d2368264d9693b5f096dff8819575f3078.jpg)

![](images/74b296675ed1ff1113f09da45d9f9ef246e29be4b8b7b8d5250a072cebb97ed4.jpg)

![](images/59ab2aa477851b50325167ffc0d7eccc44e76e4654491e5c384253d591bd5f70.jpg)  
Figure 6: Deep Bayesian Bandit Showdown on the Wheel Bandit (δ = 0.9, K = 5 arms) across 1,500 environment steps (evaluated over 5 independent random seeds). (a) Wheel Bandit Geometry and Exploration Trap: Contexts $x \in \mathbb { R } ^ { 2 }$ are sampled uniformly from the unit disk. The safe interior $( \| x \| _ { 2 } \leq \delta = 0 . 9$ , grey) accounts for 81% of all contexts, where base arm 0 yields $r = 1 . 0$ while arms 1..4 yield $r = 0 . 0$ . The high-payof optimal arms $( r = 5 0 . 0 )$ reside exclusively on the rare outer rim $( \Vert x \Vert _ { 2 } > 0 . 9 $ , 19% total probability, ≈ 4.75% per quadrant). (b) Mean Cumulative Regret $R ( t )$ with ±1 Standard Error Bands: DP-BNN rapidly discovers the outer quadrants and sustains high optimal action selection, achieving the lowest cumulative regret $( 2 , 8 0 3 . 3 \pm 1 8 9 . 6 )$ and outperforming Bootstrapped NN + Priors $( 4 , 3 0 9 . 4 \pm 8 2 9 . 2 )$ and ϵ-greedy $( 6 , 8 9 4 . 6 \pm 1 , 5 1 4 . 6 )$ . (c) Final Regret and Network Footprint: DP-BNN operates with a single neural network (1× memory footprint), whereas Bootstrapped NN + Priors requires 20 total networks (20× memory) and Deep Ensembles requires 5 networks (5× memory).

Absence of Warmstart Steps in Contextual Bandits $( H = 1 )$ . Unlike sequential reinforcement learning where the warmstart budget $W = \lceil H / 2 \rceil$ is required to propagate delayed temporal-diference credit across multi-step trajectory chains and freeze policies for rollout consistency (Appendix C), contextual bandits possess horizon $H = 1$ and discount factor $\gamma = 0$ . Because each interaction is an instantaneous single-step decision without state transitions $\left( s ^ { \prime } \right.$ does not exist), there are no recursive Bellman error chains to propagate. Following standard contextual bandit benchmarks (Riquelme et al., 2018), learning operates as supervised regression from context-action pairs $( x , a )$ to reward r: the network updates periodically on the bufer plus T prior atoms from $F _ { 0 } ,$ and Thompson sampling is performed per context $x _ { t } .$ , obviating episodic warmstart steps $\left( W = - \mathrm { ~ i n ~ T a b l e ~ 2 } \right)$

Unified Hyperparameter Protocols Across All Benchmarks. Table 2 consolidates the complete structural and optimization hyperparameters across all empirical evaluations in this work, ensuring full algorithmic transparency and reproducibility.

## F Continuous State-Space RL Exploration on MountainCar-v0

While the benchmarks in Appendix E evaluate tabular representations, discrete grids, and continuous contextual bandits, many physical control problems operate in continuous state spaces with non-linear dynamical constraints. To evaluate DP-DQN in continuous state-space exploration, we benchmark on the canonical MountainCar-v0 task (Sutton & Barto, 1998).

Table 2: Consolidated experimental specifications and hyperparameters for DP-BNN and DP-DQN across all empirical benchmarks $( W = -$ for static regression and contextual bandits because H = 1 with no sequential Bellman propagation).
<table><tr><td>Hyperparameter</td><td>2D Annular Cavity</td><td>DeepSea-20</td><td>DeepSea-50</td><td>RiverSwim-6</td><td>Deceptive N-Chain</td><td>MountainCar-v0</td><td>Wheel Bandit</td></tr><tr><td>State Space S</td><td> $[ - 2 . 5 , 2 . 5 ] ^ { 2 }$ </td><td>One-Hot R400</td><td>One-Hot R2500</td><td>One-Hot R6</td><td>One-Hot R10</td><td>Continuous R²</td><td>Unit Disk R²</td></tr><tr><td>Action Space A</td><td>(Regression)</td><td>Discrete (2)</td><td>Discrete (2)</td><td>Discrete (2)</td><td>Discrete (2)</td><td>Discrete (3)</td><td>Discrete (5)</td></tr><tr><td>Horizon H</td><td></td><td>20</td><td>50</td><td>50</td><td>10</td><td>200</td><td>1</td></tr><tr><td>Network Architecture</td><td>3-layer MLP-64</td><td>Single MLP-20</td><td>2-layer MLP-64</td><td>Single MLP-20</td><td>Single MLP-20</td><td>2-layer MLP-64</td><td>3-layer MLP-64</td></tr><tr><td>Layer Normalization</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Learning Rate η</td><td>10−3</td><td>10−3</td><td>10−3</td><td>10−3</td><td>10−3</td><td>10−3</td><td>10−3</td></tr><tr><td>Mini-Batch Size B</td><td>64</td><td>64</td><td>64</td><td>32</td><td>32</td><td>32</td><td>64</td></tr><tr><td>Replay Buffer Size |B|</td><td></td><td>100,000</td><td>100,000</td><td>10,000</td><td>10,000</td><td>50,000</td><td>10,000</td></tr><tr><td>DP Concentration α Prior Truncation Level T</td><td>10.0</td><td>3.0</td><td>3.0 50</td><td>3.0</td><td>3.0</td><td>3.0</td><td>1.0</td></tr><tr><td>Warmstart Steps W</td><td>50</td><td>50</td><td>25</td><td>50</td><td>50</td><td>50</td><td>20</td></tr><tr><td>Polyak Target Rate τ</td><td></td><td>10 0.05</td><td>0.05</td><td>25 0.05</td><td>5 0.05</td><td>100 0.01</td><td></td></tr><tr><td></td><td></td><td></td><td>U(0, 1.0)</td><td></td><td>N(1.0, 0.22)</td><td></td><td></td></tr><tr><td>Prior Base Reward  $F _ { 0 r }$ </td><td>N(0,1.52)</td><td>U(0, 1.0)</td><td></td><td>N(0, 1.0)</td><td></td><td>u(0, 1.0)</td><td>N(1.0,0.52)</td></tr></table>

Task Formulation and Exploration Challenge. MountainCar-v0 features a two-dimensional continuous state space $\mathcal { S } = [ - 1 . 2 , 0 . 6 ] \times [ - 0 . 0 7 , 0 . 0 7 ]$ representing horizontal car position x and velocity v, and a threeaction discrete space $\mathcal { A } = \{ 0 , 1 , 2 \}$ corresponding to pushing left $( a = 0 )$ , coasting (a = 1), and pushing right $( a = 2 )$ . The agent is initialized at the bottom of a sinusoidal valley $( x \in [ - 0 . 6 , - 0 . 4 ] , v = 0 )$ and must reach the hilltop goal flag $( x \ge 0 . 5 )$ within horizon $H = 2 0 0$ . The engine is underpowered: accelerating directly forward cannot overcome the gravitational force of the slope, requiring the agent to discover a coordinated swinging strategy that builds momentum by moving left before driving up the right hill. The reward is strictly negative at every step $( r _ { t } = - 1 . 0 )$ with termination upon reaching the flag $( r _ { \mathrm { t e r m } } = 0 . 0 )$ . Under local dithering (ϵ-greedy), exploration is exceedingly slow because random actions cancel out momentum, trapping the car in the bottom of the valley.

Model-Free Brownian Difusive Base Measure (Zero Dynamics Knowledge). To establish that DP-DQN does not rely on domain-specific diferential equations or privileged physics knowledge, we specify a model-free difusive base measure $F _ { 0 } = F _ { 0 s } \times F _ { 0 a } \times F _ { 0 r } \times F _ { 0 s ^ { \prime } } \times F _ { 0 d }$ . States $\tilde { s } = ( \tilde { x } , \tilde { v } )$ are sampled uniformly over domain support $[ - 1 . 2 , 0 . 6 ] \times [ - 0 . 0 7 , 0 . 0 7 ]$ , actions $\tilde { a } \sim \mathrm { U n i f o r m } ( \{ 0 , 1 , 2 \} )$ , and prior rewards follow a uniform distribution $F _ { 0 r } = \mathcal { U } ( 0 , 1 . 0 )$ with continuing flag $\tilde { d } = 0$ . Next-state transitions are generated via continuous Brownian difusion without physical equations of motion: $\tilde { s } ^ { \prime } = \mathrm { c l i p } ( \tilde { s } + \Delta , S )$ , where $\Delta \sim \mathcal { N } ( 0 , \mathrm { d i a g } ( \sigma _ { x } ^ { 2 } , \sigma _ { v } ^ { 2 } ) )$ with difusion scales $\sigma _ { x } = 0 . 0 4$ and $\sigma _ { v } = 0 . 0 0 7$ , chosen according to the canonical bounding-box heuristic as $2 \% - 5 \%$ of each state coordinate’s domain span (2.2% of position span 1.80 and 5.0% of velocity span 0.14, matching the characteristic 1-step displacement scale). Driven purely by functional posterior variance via stick-breaking draws, single-network DP-DQN (2-layer 64 × 64 MLP with LayerNorm) solves MountainCar-v0 across all random seeds in $1 6 . 8 \pm 7 . 4$ episodes (reaching the goal as early as Episode 4), outperforming vanilla DQN (85 ± 12 episodes) and matching 20-head Bootstrapped DQN (15 ± 3 episodes) while delivering a 3.8× wall-clock training speedup (12.4 s vs. 47.1 s for 100 episodes) due to single-network parameter eficiency.

## G Computational and Memory Complexity Analysis

In this section, we analyze the asymptotic computational, memory, and scaling complexity of DP-DQN in comparison with multi-network ensemble architectures (Bootstrapped DQN with randomized priors; Osband et al. 2018), parametric Bayesian value-function methods (BDQN; Azizzadenesheli et al. 2018), and standard deep Q-learning (Mnih et al., 2015). A comparative summary of complexity orders is presented in Table 3.

Parameter Footprint and Memory Scaling. Let $d = | \theta |$ denote the parameter count of a single action-value neural network. Multi-network ensemble methods such as BootDQN-RP maintain an ensemble of K distinct neural networks $( K = 2 0$ in standard DeepSea benchmarks). For each ensemble member $k \in \{ 1 , \ldots , K \}$ , the algorithm instantiates an active trainable network $f _ { \theta _ { k } }$ , a non-trainable randomized prior network $p _ { \tilde { \theta } _ { k } }$ , and a tracking target network $f _ { \theta _ { k } ^ { - } }$ , yielding 3K · d stored parameters $( \mathscr { O } ( K \cdot d )$ asymptotic scaling). In stark contrast, DP-DQN represents Bayesian epistemic uncertainty nonparametrically directly in experience space via the DP base measure $F _ { 0 }$ . Consequently, DP-DQN operates on a single active neural network θ and a single target network $\theta ^ { - }$ , requiring only 2d stored weights (O(d) asymptotic scaling), achieving a strict K-fold reduction in active trainable parameters and a 1.5K-fold (30× for K = 20) reduction in total stored model footprint. Furthermore, first-order adaptive optimizers (e.g., Adam) maintain running first- and second-moment momentum accumulators for every trainable parameter, requiring 2d auxiliary state bufers for DP-DQN versus 2K · d bufers for BootDQN-RP, eliminating memory thrashing and cache overhead during training updates.

Table 3: Computational and memory complexity comparison across value-based deep RL paradigms for an action-value network with $d = | \theta |$ parameters, episode horizon H, replay batch size $B ,$ ensemble size $K ,$ and prior truncation level T.
<table><tr><td>Complexity Dimension</td><td>Vanilla DQN</td><td>Parametric BDQN</td><td>BootDQN-RP (K networks)</td><td>DP-DQN (Ours)</td></tr><tr><td>Active Neural Networks</td><td>1</td><td>1</td><td>K</td><td>1</td></tr><tr><td>Total Stored Parameters</td><td>2d</td><td> $2 d + | S _ { \mathrm { f e a t } } | ^ { 2 }$ </td><td>3K · d</td><td>2d</td></tr><tr><td>Parameter Memory Asymptotics</td><td>O(d)</td><td> $\mathcal { O } ( d + \dot { | } S _ { \mathrm { f e a t } } | ^ { 2 } )$ </td><td>O(K · d)</td><td>O(d)</td></tr><tr><td>Optimizer State Buffers (Adam)</td><td>2d</td><td>2d</td><td>2K· d</td><td>2d</td></tr><tr><td>Action Selection Latency / Step</td><td> $\mathcal { O } ( 1 \cdot d )$ </td><td> $\mathcal { O } ( d + \vert \mathcal { A } \vert \cdot \vert S _ { \mathrm { f e a t } } \vert )$ </td><td>O(K · d)</td><td>O(1 · d)</td></tr><tr><td>Training FLOPs / Episode</td><td> $\begin{array} { r } { \mathcal { O } \left( \frac { H } { \Delta } \cdot B \cdot d \right) } \end{array}$ </td><td> $\begin{array} { r } { { \mathcal O } \left( \frac { \dot { H } } { \Delta } \cdot B \cdot d \right) } \end{array}$ </td><td> $\begin{array} { r } { \mathcal { O } \left( \frac { H } { \Delta } \cdot K \cdot B \cdot d \right) } \end{array}$ </td><td>O (W · (B + T) · d)</td></tr><tr><td>Posterior Sampling Overhead / Episode</td><td>None</td><td> $\dot { \mathcal { O } } ( | \vec { S } _ { \mathrm { f e a t } } | ^ { 3 } )$ </td><td>None</td><td> $\mathcal { O } ( \mathbf { \dot { T } } + \mathbf { W } \cdot \mathbf { B } )$  scalars</td></tr><tr><td>Temporal Consistency Mechanism</td><td>None (dithering)</td><td>Final-layer sampling</td><td>Episode head freezing</td><td>Episode network freezing</td></tr><tr><td>Sample Complexity in Horizon H</td><td> $\Omega ( | A | ^ { H } )$ </td><td> $\bar { \Omega ( | \mathcal { A } | ^ { H } ) }$ </td><td>O(poly(H))</td><td>O(poly(H))</td></tr></table>

Table 4: Empirical computational complexity, parameter volume, and runtime profiling on DeepSea-20 across algorithms on a standardized single CPU core.

<table><tr><td>Metric</td><td>Vanilla DQN</td><td>BDQN</td><td>BootDQN-RP</td><td>DP-DQN (W = 10)</td><td>DP-DQN (W = 4)</td></tr><tr><td>Active Networks</td><td>1</td><td>1</td><td>20</td><td>1</td><td>1</td></tr><tr><td>Total Networks (incl. Prior/Target)</td><td>2</td><td>2</td><td>60</td><td>2</td><td>2</td></tr><tr><td>Trainable Parameters (d)</td><td>8,102</td><td>8,142</td><td>162,040</td><td>8,102</td><td>8,102</td></tr><tr><td>Total Stored Parameters</td><td>16,204</td><td>16,284</td><td>486,120</td><td>16,204</td><td>16,204</td></tr><tr><td>Optimizer Instances (sets of 2d)</td><td>1</td><td>1</td><td>20</td><td>1</td><td>1</td></tr><tr><td>Intra-Episode Backprop Steps</td><td>10</td><td>10</td><td>200</td><td>0</td><td>0</td></tr><tr><td>Training FLOPs / Episode</td><td>1,280 · d</td><td>1,300 · d</td><td>25,600 · d</td><td>2,880 · d</td><td>1,152 · d</td></tr><tr><td>Pure Action Inference Latency</td><td>25.8 µs</td><td>69.7 µs</td><td>107.0 µs</td><td>28.8 µs</td><td>28.6 µs</td></tr><tr><td>End-to-End Latency (ms/step)</td><td>0.381</td><td>0.651</td><td>1.049</td><td>0.429</td><td>0.204</td></tr><tr><td>Throughput (steps / second)</td><td>2,625</td><td>1,536</td><td>953</td><td>2,331</td><td>4,902</td></tr><tr><td>Episodes to 100% Convergence</td><td>Never</td><td>Never</td><td>5,210 ± 604</td><td>602 ± 104</td><td>645 ± 112</td></tr><tr><td>Total Compute to Solve (Wallclock)</td><td>∞</td><td>∞</td><td>103.26 s</td><td>5.16 s</td><td>2.63 s</td></tr></table>

Training FLOPs Scaling and Sampling Overhead per Episode. Consider an episodic MDP with horizon H. In standard online deep Q-learning and BootDQN-RP, stochastic gradient updates are executed every $\Delta$ environment steps (typically $\Delta = 2 )$ . For an ensemble of K networks evaluated over empirical mini-batches of size B, each network executes a backpropagation pass requiring $C _ { \mathrm { b a c k p r o p } } \cdot B$ · d floating-point operations (where $C _ { \mathrm { b a c k p r o p } } \approx 2$ for standard feedforward architectures). Over the entire episode of H steps, the cumulative training compute scales as:

$$
{ \mathrm { F L O P s } } _ { \mathrm { B o o t D Q N - R P } } = \left( { \frac { H } { \Delta } } \right) \cdot K \cdot C _ { \mathrm { b a c k p r o p } } \cdot B \cdot d = { \mathcal { O } } ( H \cdot K \cdot B \cdot d ) .\tag{36}
$$

In contrast, DP-DQN freezes the active policy network $\theta _ { k }$ during the rollout to maintain complete multi-step temporal consistency, completely bypassing intra-episode backpropagation. All network optimization is amortized into $W = \lceil H / 2 \rceil$ warmstart gradient updates at episode boundaries evaluated over composite mini-batches of size $B + T$ (empirical transitions plus synthetic prior atoms). In addition to network backpropagation, DP-DQN executes $\mathcal { O } ( T )$ scalar operations to sample prior stick weights once per episode, and $\mathcal { O } ( B )$ scalar operations to sample empirical weights at each warmstart step, totaling $\mathcal { O } ( T + W \cdot B )$ scalar operations per episode. Because $T + W \cdot B \ll W \cdot ( B + T ) \cdot d \ ( \mathrm { e . g . } , \approx 6 5 0$ scalar stick-breaking operations versus ≈ $2 . 3 \times 1 0 ^ { 7 }$ cumulative training backpropagation FLOPs per episode on DeepSea-20), this stick-breaking sampling overhead is negligible relative to tensor arithmetic. The per-episode training FLOPs evaluate to:

$$
\mathrm { F L O P s } _ { \mathrm { D P \mathrm { - } D Q N } } = W \cdot C _ { \mathrm { b a c k p r o p } } \cdot ( B + T ) \cdot d = \mathcal { O } \left( W \cdot ( B + T ) \cdot d \right) .\tag{37}
$$

Recalling that $W = \lceil H / 2 \rceil \approx H / \Delta$ , the computational ratio of DP-DQN relative to standard vanilla DQN evaluates to:

$$
\frac { \mathrm { F L O P s } _ { \mathrm { D P - D Q N } } } { \mathrm { F L O P s } _ { \mathrm { V a n i l l a - D Q N } } } = \frac { W \cdot C _ { \mathrm { b a c k p r o p } } \cdot ( B + T ) \cdot d } { ( H / \Delta ) \cdot C _ { \mathrm { b a c k p r o p } } \cdot B \cdot d } \approx 1 + \frac { T } { B } .\tag{38}
$$

Because the prior truncation level T is moderate $( T \in [ 2 5 , 5 0 ]$ via the $9 9 . 9 \%$ mass rule) relative to typical batch sizes $B \in [ 6 4 , 1 2 8 ]$ , this ratio represents only a modest 1.2× to 1.8× factor in per-episode backpropagation compute over standard DQN, in exchange for upgrading sample complexity from exponential $\Omega ( | A | ^ { H } )$ to polynomial $\mathcal { O } ( \mathrm { p o l y } ( H ) )$ . The per-step execution latencies reported in Table 4 are measured by amortizing total episode wall-clock time over horizon H.

Simultaneously, the computational ratio of DP-DQN relative to BootDQN-RP evaluates to:

$$
{ \frac { \mathrm { F L O P s } _ { \mathrm { D P - D Q N } } } { \mathrm { F L O P s } _ { \mathrm { B o o t D Q N - R P } } } } = { \frac { W \cdot ( B + T ) \cdot d } { ( H / \Delta ) \cdot K \cdot B \cdot d } } \approx { \frac { B + T } { K \cdot B } } = { \frac { 1 } { K } } \left( 1 + { \frac { T } { B } } \right) .\tag{39}
$$

In our canonical configurations $( K = 2 0 , B = 6 4 , T = 5 0 )$ , this ratio evaluates to $\textstyle \frac { 1 } { 2 0 } ( 1 + 5 0 / 6 4 ) \approx 0 . 0 8 9$ yielding an asymptotic $\approx 1 1 \times$ reduction in per-episode backpropagation operations.

Action Inference Complexity per Step. At each environment step t, the agent selects an exploratory action. For BootDQN-RP, evaluating the value function requires either querying all K heads to compute posterior voting statistics or indexing the active head and adding the forward evaluation of its randomized prior network, requiring $\mathcal { O } ( K \cdot d )$ operations. In contrast, DP-DQN requires only a single forward pass through the frozen network $\theta _ { k } \colon a _ { t } = \arg \operatorname* { m a x } _ { a } Q ( s _ { t } , a ; \theta _ { k } )$ , scaling strictly as $\mathcal { O } ( 1 \cdot d ) \ ( \mathcal { O } ( 1 )$ network latency).

Exploration Complexity: Posterior Sampling vs. Local Dithering. In MDPs requiring deep multistep exploration over horizon H, exploration eficiency is fundamentally governed by the temporal consistency of the agent’s policy. Local dithering schemes (such as ϵ-greedy in DQN or Gaussian action noise in continuous control) make independent exploratory perturbations at each step. In environments where optimal credit requires an extended sequence of coordinated actions (such as traversing H steps along the diagonal in DeepSea), the probability that independent dithering selects the optimal path by chance decays exponentially as $( 1 / | { \dot { A } } | ) ^ { H }$ . Consequently, any local dithering heuristic incurs an exponential sample complexity lower bound:

$$
{ \mathrm { S a m p l e ~ C o m p l e x i t y } } _ { \mathrm { D i t h e r i n g } } = \Omega \left( | A | ^ { H } \right) .\tag{40}
$$

In stark contrast, by freezing an action-value function sampled from a functional posterior satisfying stochastic optimism $( \mathbb { P } ( Q \geq Q ^ { * } ) \geq p _ { \operatorname* { m i n } } > 0 )$ (Agrawal & Goyal, 2012; Agrawal & Jia, 2017; Osband et al., 2013), posterior sampling achieves provably polynomial sample complexity $\mathcal { O } ( \mathrm { p o l y } ( H , | S | , | A | ) )$ in tabular MDPs. DP-DQN preserves this polynomial deep-exploration capability nonparametrically with a single neural network, bypassing both the exponential sample wall of dithering heuristics and the K-fold compute penalty of physical network ensembles.