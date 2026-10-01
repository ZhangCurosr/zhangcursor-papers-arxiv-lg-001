# Fast Regularized Policy Mirror Descent with One-Step TD Updates

Qipei Chen<sup>1</sup>, Wenye Li<sup>2</sup>, Yule Sun<sup>2</sup>, and Ke Wei<sup>2</sup>

<sup>1</sup>School of Mathematical Sciences, Fudan University <sup>2</sup>School of Data Science, Fudan University

October 1, 2026

## Abstract

Policy mirror descent (PMD) enjoys fast convergence in regularized Markov decision processes (MDPs), but existing guarantees often rely on exact or increasingly accurate policy evaluation. We analyze PMD coupled with a persistent critic advanced by one temporal-diference (TD) update. For finite discounted MDPs, we establish global linear convergence in value for exact coordinate-wise Bellman updates, with any positive constant actor stepsize and arbitrary finite critic initialization. The proof combines a resolvent-based auxiliary distribution with a decaying Bellman-violation correction and a potential weighted by inverse coordinate weights. We then study stochastic TD–PMD with general strongly convex mirror maps under a single of-policy Markov trajectory. With suitably chosen constant stepsizes and a finite-batch TD update, the method achieves an expected value gap of ϵ after $\begin{array} { r } { \widetilde { \mathcal O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon } \right) } \end{array}$ transitions. The stochastic analysis relies on the trajectory-wise Lipschitz continuity of the regularizer, derived from uniform bounds on vertex Bregman divergences, together with a visitation-weighted resolvent estimate for signed critic-error propagation that yields an inverse-linear dependence on behavior coverage $\widetilde { \sigma } _ { b }$ . In contrast to many prior guarantees for regularized policy optimization, our sample-complexity guarantee holds without trajectory resets, generative-model access, or nested policy-evaluation loops. Numerical results are consistent with the theoretical convergence analysis.

## 1 Introduction

Regularization is a fundamental design principle in reinforcement learning (RL). In large-scale problems with complex dynamics, maintaining suficient randomness in the policy iterates helps sustain exploration and reduces the risk of committing too early to a suboptimal policy [Husain et al., 2021]. A variety of policy regularizers have been studied in RL, including those based on Shannon entropy, Kullback–Leibler divergence, the squared $\ell _ { 2 }$ norm, and Tsallis entropies [Geist et al., 2019; Lan, 2023; Chow et al., 2018; Lee et al., 2018; 2019]. Regularizers can also promote sparse or structured policies and incorporate information from a reference policy. Therefore, regularized formulations are very useful for robust, cost-constrained, and safety-aware decision making [Husain et al., 2021; Lan, 2023; Ying et al., 2022].

A general optimization framework for regularized RL is the regularized form of policy mirror descent (PMD). In this formulation, PMD improves the policy through a statewise Bregman-proximal step and, depending on the choice of mirror geometry, includes softmax natural policy gradient (NPG) and projected Q-ascent (PQA) as representative instances corresponding to KL and Euclidean geometries, respectively [Kakade, 2001; Xiao, 2022; Lan, 2023]. Entropy-regularized NPG and regularized PMD enjoy global geometric convergence [Cen et al., 2022; Lan, 2023]. The framework has since been extended in several directions. Generalized PMD (GPMD) accommodates broad classes of convex, possibly nonsmooth regularizers and controlled evaluation and update errors [Zhan et al., 2023]. Homotopic, lookahead, implicit-exploration, policy-convergence, and strongly-polynomial variants further extend PMD in terms of convergence guarantees and computational properties [Li et al., 2024; Protopapas and Barakat, 2024; Li and Lan, 2025; Li and Wei, 2026; Ju and Lan, 2026]. For strongly convex mirror maps, regularized stochastic PMD can attain an $\widetilde { \mathcal { O } } ( \epsilon ^ { - 1 } )$ dependence on the target accuracy [Lan, 2023]. In this paper, we focus on regularized PMD. For the algorithmic and theoretical developments of unregularized PMD or its specific instances, see Xiao [2022]; Johnson et al. [2023]; Agarwal et al. [2021]; Liu et al. [2024; 2025a]; Lin and Zhang [2022]; Khodadadian et al. [2021]; Alfano et al. [2023]; Alfano and Rebeschini [2022]; Chelu and Precup [2024]; Feng et al. [2024] and references therein.

Existing analyses of regularized PMD, however, generally rely on exact or suficiently accurate evaluation of the current policy before each policy improvement step. For example, exact analyses assume access to the current policy’s action values [Cen et al., 2022; Lan, 2023; Johnson et al., 2023], while inexact analyses assume an evaluation oracle that returns action-value estimates with controlled error [Zhan et al., 2023]. Under generative-model access, samples are used at each policy iteration to estimate the current action values to a prescribed accuracy [Lan, 2023; Li et al., 2024; Johnson et al., 2023]; under Markovian sampling, a temporal-diference or Monte Carlo evaluator is run for a fixed current policy before the next policy improvement step [Lan, 2023; Li and Lan, 2025]. Thus, across these settings, the critic is required to track the action-value function of the current policy suficiently accurately before the next actor update is performed.

In contrast, temporal-diference (TD) learning provides a natural way to update the critic incrementally from its current estimate, without solving the Bellman equation for each newly updated policy to a prescribed accuracy before the next policy update [Sutton, 1988; Sutton and Barto, 2018]. A related idea appears in regularized modified policy iteration [Geist et al., 2019], where only finitely many Bellman evaluation steps are performed between successive policy improvements. With a single evaluation step, the resulting recursion is closely related to one-step TD–PMD. More recently, one-step TD–PMD has been analyzed for unregularized MDPs in the exact and generative-model settings [Liu et al., 2025b], and subsequently under online Markov data [Li et al., 2025]. In the exact setting, under a one-sided Bellman initialization, a monotonicity property of TD–PMD with constant stepsizes is established and used to derive last-iterate sublinear convergence, and a shifting argument then extends the result to arbitrary initializations [Liu et al., 2025b]. Regularized settings have also been studied with single-loop methods that do not require each intermediate policy to be evaluated to a prescribed accuracy. For entropy-regularized zero-sum Markov games, a single-loop actor–critic method uses optimistic NPG for the policy step and updates the value estimate at each iteration [Cen et al., 2023]. Another related approach for regularized MDPs is value mirror descent (VMD), which takes a value-iteration approach: it computes a mirror-based policy update from the current value estimate and then applies a Bellman value update under the resulting policy [Jia and Lan, 2026]. In the exact setting, linear convergence of VMD is established with an increasing, epoch-dependent stepsize schedule for one-sided Bellman initialization. The proof also uses a monotonicity property of the value iterates, similar to the analysis developed in Liu et al. [2025b].

Despite these developments, for standard regularized PMD with only one TD update per policy step, a fast convergence rate in the exact setting remains unavailable, while the corresponding convergence theory under Markov data is still lacking. Existing one-step TD–PMD results focus on unregularized MDPs, covering the exact, generative-model, and online Markov-data settings [Liu et al., 2025b; Li et al., 2025]. In regularized settings, the single-loop method of Cen et al. [2023] is developed specifically for the entropy regularizer corresponding to NPG, and its discounted analysis is full-information, requires suficiently small actor stepsizes, and uses a full-support evaluation distribution. Value mirror descent (VMD) [Jia and Lan, 2026] considers regularized MDPs from a diferent algorithmic perspective: it replaces the classical PMD recursion with an epochwise value-iteration scheme and achieves linear convergence in the exact setting with increasing, epoch-dependent policy stepsizes under a one-sided Bellman initialization. Its stochastic variant, SVMD, further relies on generative-model samples to construct increasingly accurate estimates of the transition kernel and performs variance-reduced Bellman updates on the estimated models, making it model-based rather than model-free TD. Thus, a general convergence theory for standard regularized PMD with one-step TD updates is still missing. The online setting is further complicated by the need to track a changing target from a single of-policy Markov trajectory. This gap motivates the central question of this paper:

Can the fast convergence of standard regularized PMD be retained when the critic is updated by only one TD step, even along a single of-policy Markov trajectory?

We answer this question afirmatively. The main contributions of this paper are summarized as follows. See Table 1 for a comparison with existing regularized PMD results.

• Linear convergence for regularized TD–PMD with coordinate-wise critic updates. We study regularized TD–PMD with coordinate-wise Bellman updates, allowing any fixed diagonal update matrix W satisfying $0 < W \leq I .$ . For a general convex mirror map, we establish its global linear convergence in value. When the mirror map is strongly convex, policy convergence is further obtained. The results allow arbitrary finite critic initialization, any admissible initial policy, every positive constant actor stepsize, and any state distribution used to evaluate performance, including distributions without full support. The analysis relies on a decaying Bellman-violation correction for arbitrary critic initialization, a change-of-distribution argument that avoids full-support requirements, and a potential function with inverse coordinate weights for nonuniform critic updates.

• Stochastic convergence under a single of-policy Markov trajectory. With a constant actor stepsize and a finite-batch TD update per policy step, regularized TD–PMD with a strongly convex mirror map achieves an expected value gap of at most ϵ after

$$
\mathcal { \widetilde { O } } \bigg ( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon } \bigg )
$$

observed transitions, where $\widetilde { \sigma } _ { b } > 0$ is the smallest stationary state–action visitation probability under the behavior policy. The method uses neither trajectory resets nor generative-model access and requires no nested policy-evaluation loop. A key step in the analysis is to establish trajectory-wise Lipschitz continuity of the regularizer from uniform bounds on vertex Bregman divergences, which in turn controls the policy drift and the variation of $Q _ { \tau } ^ { \pi _ { k } }$ . We then use a signed critic-error decomposition and geometric output weighting to handle the resulting moving-target and Markov-sampling errors.

Table 1: Comparison of regularized PMD results. Here, $\widetilde { \sigma } _ { b }$ is the smallest stationary visitation probability over state–action pairs; under near-uniform coverage, $\widetilde { \sigma } _ { b } = \Theta ( ( | S | | \mathcal { A } | ) ^ { - 1 } )$ .
<table><tr><td></td><td>Algorithm</td><td>Iteration complexity</td><td>Stochastic access</td><td>Sample complexity</td></tr><tr><td>Cen et al. [2022]</td><td>NPG</td><td> $\mathcal { O } \Big ( \log _ { ( 1 - \eta \tau ) ^ { - 1 } } ( \epsilon ^ { - 1 } ) \Big )$ </td><td></td><td></td></tr><tr><td>Cen et al. [2023]</td><td>TD-NPG</td><td> $\mathcal { O } \biggl ( \log _ { [ 1 - \frac { ( 1 - \gamma ) \eta \tau } { 4 } ] ^ { - 1 } } ( \epsilon ^ { - 1 } ) \biggr ) _ { \sqrt { \epsilon } }$ </td><td></td><td></td></tr><tr><td>Zhan et al. [2023]</td><td>PMD</td><td> $\begin{array} { r } { \mathcal { O } \left( \log _ { [ 1 - \frac { \eta \tau } { 1 + \eta \tau } ( 1 - \gamma ) ] ^ { - 1 } } ( \epsilon ^ { - 1 } ) \right) } \end{array}$ </td><td></td><td></td></tr><tr><td>Lan [2023]</td><td>PMD</td><td> $\mathcal { O } \Big ( \log _ { \gamma ^ { - 1 } } ( \epsilon ^ { - 1 } ) \Big )$ </td><td>Generative model</td><td> $\widetilde { \mathcal { O } } \left( \frac { | \mathcal { S } | | \mathcal { A } | } { ( 1 - \gamma ) ^ { 5 } \epsilon } \right)$ </td></tr><tr><td>Jia and Lan [2026]</td><td>VMD</td><td> $\mathcal { O } \big ( ( 1 - \gamma ) ^ { - 1 } \log _ { 2 } ( [ ( 1 - \gamma ) \epsilon ] ^ { - 1 } ) \big )$ </td><td>Generative model</td><td> $\tilde { \mathcal { O } } \Big ( \frac { | \boldsymbol { \mathsf { \sigma } } | | \boldsymbol { \mathcal { A } } | } { ( 1 - \gamma ) ^ { 5 } \epsilon } \Big )$ </td></tr><tr><td>This work</td><td>TD-PMD</td><td> $\mathcal { O } \left( \log _ { [ 1 - \underline { { w } } \operatorname* { m i n } \{ 1 - \gamma , \frac { \eta \tau } { 1 + \eta \tau } \} ] ^ { - 1 } } ( \epsilon ^ { - 1 } ) \right)$ </td><td>Off-policy Markov data</td><td> $\begin{array} { r } { \widetilde { \mathcal { O } } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon } \right) } \end{array}$ </td></tr></table>

The rest of the paper is organized as follows. Section 2 introduces the problem setup and necessary preliminaries. Section 3 studies regularized TD–PMD with deterministic coordinate-wise Bellman updates. Section 4 extends the analysis to of-policy Markov data. Section 5 presents numerical results that are consistent with the theoretical convergence analysis, and Section 6 concludes the paper with directions for future work. The proofs of the main results and technical lemmas are deferred to the appendices.

## 2 Problem Setting and Preliminaries

## 2.1 Regularized MDPs and Bellman preliminaries

Model and objective. Consider a finite discounted Markov decision process (MDP) $\mathcal { M } = ( \mathcal { S } , \mathcal { A } , P , r , \gamma )$ where $s$ and $\mathcal { A }$ are finite state and action spaces, $\textstyle P ( \cdot \mid s , a )$ is the transition kernel, $r ( s , a )$ is the immediate reward, and $\gamma \in [ 0 , 1 )$ is the discount factor. Letting $\Delta ( \mathcal { A } )$ be the probability simplex over ${ \mathcal { A } } ,$ the space of stationary Markov policies is

$$
\Pi : = \{ \pi : \pi ( \cdot \mid s ) \in \Delta ( A ) , \ s \in { \mathcal { S } } \} .
$$

Under a policy π, a trajectory evolves according to $a _ { t } \sim \pi ( \cdot \mid s _ { t } )$ and $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } )$

Assumption 2.1 (Reward and regularization parameters). For simplicity, assume $0 \leq r ( s , a ) \leq 1$ for every $( s , a ) \in \mathcal S \times \mathcal A$ . Unless otherwise specified, we also assume $\tau > 0$ throughout the paper.

Let $h : \mathbb { R } ^ { | A | }  \mathbb { R } \cup \{ + \infty \}$ be a convex policy regularizer and write $h ^ { \pi } ( s ) : = h ( \pi ( \cdot \mid s ) )$ . For a policy $\pi ,$ its regularized state- and action-value functions are

$$
V _ { \tau } ^ { \pi } ( s ) : = \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau h ^ { \pi } ( s _ { t } ) \big ) \Bigg | s _ { 0 } = s \right] ,
$$

$$
Q _ { \tau } ^ { \pi } ( s , a ) : = \mathbb { E } \left[ r ( s _ { 0 } , a _ { 0 } ) + \sum _ { t = 1 } ^ { \infty } \gamma ^ { t } \big ( r ( s _ { t } , a _ { t } ) - \tau h ^ { \pi } ( s _ { t } ) \big ) \Bigg | \left( s _ { 0 } , a _ { 0 } \right) = ( s , a ) \right] .
$$

It follows directly from the definitions that

$$
V _ { \tau } ^ { \pi } ( s ) = \mathbb { E } _ { a \sim \pi ( \cdot \vert s ) } [ Q _ { \tau } ^ { \pi } ( s , a ) ] - \tau h ^ { \pi } ( s ) , \qquad Q _ { \tau } ^ { \pi } ( s , a ) = r ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) } [ V _ { \tau } ^ { \pi } ( s ^ { \prime } ) ] .
$$

For $\mu \in \Delta ( \mathcal { S } )$ , we abbreviate $V _ { \tau } ^ { \pi } ( \mu ) : = \mathbb { E } _ { s \sim \mu } [ V _ { \tau } ^ { \pi } ( s ) ]$

Bellman representation. The Bellman operators acting on state- and action-value vectors are

$$
\begin{array} { r l } & { ~ \mathcal { T } _ { \tau } ^ { \pi } V ( s ) : = \mathbb { E } _ { \mathbf { \lambda } _ { a } \sim \pi ( \cdot \vert s ) } ~ [ r ( s , a ) + \gamma V ( s ^ { \prime } ) - \tau h ^ { \pi } ( s ) ] , } \\ & { ~ \mathcal { F } _ { \tau } ^ { \pi } Q ( s , a ) : = \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) } [ r ( s , a ) + \gamma Q ( s ^ { \prime } , a ^ { \prime } ) - \tau \gamma h ^ { \pi } ( s ^ { \prime } ) ] . } \\ & { ~ \mathcal { F } _ { \tau } ^ { \pi } Q ( s , a ) : = \mathbb { E } _ { s ^ { \prime } \sim \pi ( \cdot \vert s ^ { \prime } ) } [ r ( s , a ) + \gamma Q ( s ^ { \prime } , a ^ { \prime } ) - \tau \gamma h ^ { \pi } ( s ^ { \prime } ) ] . } \end{array}
$$

Equivalently,

$$
\begin{array} { r } { \mathcal { T } _ { \tau } ^ { \pi } V = r ^ { \pi } + \gamma P _ { s } ^ { \pi } V - \tau h _ { s } ^ { \pi } , } \end{array}
$$

$$
\mathcal { F } _ { \tau } ^ { \pi } Q = r + \gamma P _ { s \times \mathcal { A } } ^ { \pi } Q - \tau \gamma h _ { s \times \mathcal { A } } ^ { \pi } ,
$$

where $\begin{array} { r } { r ^ { \pi } ( s ) : = \sum _ { a } \pi ( a \mid s ) r ( s , a ) , h _ { s } ^ { \pi } ( s ) : = h ^ { \pi } ( s ) , h _ { s \times . A } ^ { \pi } ( s , a ) : = \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) h ^ { \pi } ( s ^ { \prime } ) } \end{array}$ . The state and state–action transition matrices are defined by

$$
{ \cal P } _ { s } ^ { \pi } ( s , s ^ { \prime } ) : = \sum _ { a \in \mathcal { A } } \pi ( a \mid s ) { \cal P } ( s ^ { \prime } \mid s , a ) ,
$$

$$
{ \cal P } _ { s \times { \cal A } } ^ { \pi } ( ( s , a ) , ( s ^ { \prime } , a ^ { \prime } ) ) : = { \cal P } ( s ^ { \prime } \mid s , a ) \pi ( a ^ { \prime } \mid s ^ { \prime } ) .
$$

The following standard properties, which can be verified directly, identify policy evaluation with a contractive fixed-point problem.

Lemma 2.1 (Bellman fixed points and contractions). For any $\pi \in \Pi$ , the regularized values are the unique fixed points

$$
\begin{array} { r } { T _ { \tau } ^ { \pi } V _ { \tau } ^ { \pi } = V _ { \tau } ^ { \pi } , \qquad \mathcal { F } _ { \tau } ^ { \pi } Q _ { \tau } ^ { \pi } = Q _ { \tau } ^ { \pi } . } \end{array}
$$

Moreover, $\mathcal { T } _ { \tau } ^ { \pi }$ and $\mathcal { F } _ { \tau } ^ { \pi }$ are monotone and are γ-contractions in the supremum norm: for all conformable $V , V ^ { \prime }$ and $Q , Q ^ { \prime }$ ，

$$
\| T _ { \tau } ^ { \pi } V - T _ { \tau } ^ { \pi } V ^ { \prime } \| _ { \infty } \leq \gamma \| V - V ^ { \prime } \| _ { \infty } , \qquad \| \mathcal { F } _ { \tau } ^ { \pi } Q - \mathcal { F } _ { \tau } ^ { \pi } Q ^ { \prime } \| _ { \infty } \leq \gamma \| Q - Q ^ { \prime } \| _ { \infty } .
$$

The optimal Bellman operators are

$$
\begin{array} { r l } & { \mathcal { T } _ { \tau } V ( s ) : = \displaystyle \operatorname* { m a x } _ { p \in \Delta ( A ) } \left\{ \sum _ { a } p ( a ) \left[ r ( s , a ) + \gamma \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) } V ( s ^ { \prime } ) \right] - \tau h ( p ) \right\} , } \\ & { \mathcal { F } _ { \tau } Q ( s , a ) : = { r ( s , a ) } + \gamma \mathbb { E } _ { s ^ { \prime } \sim P ( \cdot \vert s , a ) } \left[ \displaystyle \operatorname* { m a x } _ { p \in \Delta ( A ) } \left\{ \langle p , Q ( s ^ { \prime } , \cdot ) \rangle - \tau h ( p ) \right\} \right] . } \end{array}
$$

Under the mirror-map conditions below, an optimal stationary policy exists for the finite discounted MDP [Puterman, 1994; Geist et al., 2019]. We fix one such policy $\pi _ { \tau } ^ { * }$ and write

$$
V _ { \tau } ^ { * } ( s ) : = V _ { \tau } ^ { \pi _ { \tau } ^ { * } } ( s ) = \operatorname* { m a x } _ { \pi \in \Pi } V _ { \tau } ^ { \pi } ( s ) , \qquad Q _ { \tau } ^ { * } ( s , a ) : = Q _ { \tau } ^ { \pi _ { \tau } ^ { * } } ( s , a ) = \operatorname* { m a x } _ { \pi \in \Pi } Q _ { \tau } ^ { \pi } ( s , a ) .
$$

Distributional notation and policy comparison. A stationary distribution of $P _ { s } ^ { \pi }$ , denoted $\nu ^ { \pi }$ , is any distribution satisfying $( \nu ^ { \pi } ) ^ { \top } P _ { s } ^ { \pi } \bar { = } ( \nu ^ { \pi } ) ^ { \top }$ . The associated stationary state–action distribution is $\sigma ^ { \pi } ( s , a ) : =$ $\nu ^ { \pi } ( s ) \pi ( a \mid s )$ , which satisfies $\mathsf { \bar { \Pi } } ( \sigma ^ { \pi } ) ^ { \dagger } P _ { s \times { \cal A } } ^ { \pi } = ( \sigma ^ { \pi } ) ^ { \top }$ . For an initial distribution $\mu \in \Delta ( \mathcal { S } )$ , the discounted state-visitation measure is

$$
d _ { \mu } ^ { \pi } ( s ) : = ( 1 - \gamma ) \mathbb { E } \left[ \sum _ { \mathfrak { t } = 0 } ^ { \infty } \gamma ^ { t } \mathbb { 1 } \left\{ s _ { t } = s \right\} \bigg | s _ { 0 } \sim \mu \right] .
$$

For the fixed optimal policy $\pi _ { \tau } ^ { * } .$ write $d _ { \mu } ^ { \ast } : = d _ { \mu } ^ { \pi _ { \tau } ^ { * } }$ , and let $\nu ^ { * }$ denote any stationary distribution of $P _ { s } ^ { \pi _ { \tau } ^ { * } }$ We do not assume that $\nu ^ { * }$ is unique or has full support.

The following performance diference lemma converts a policy comparison into an average one-step Bellman advantage.

Lemma 2.2 (Performance diference lemma [Kakade and Langford, 2002]). For any $\pi , \pi ^ { \prime } \in \Pi$ and $\mu \in \Delta ( \mathcal { S } )$

$$
\begin{array} { l } { { V _ { \tau } ^ { \pi } ( \mu ) - V _ { \tau } ^ { \pi ^ { \prime } } ( \mu ) = \displaystyle \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mu } ^ { \pi } } \bigl [ T _ { \tau } ^ { \pi } V _ { \tau } ^ { \pi ^ { \prime } } ( s ) - V _ { \tau } ^ { \pi ^ { \prime } } ( s ) \bigr ] } } \\ { { \displaystyle \quad \quad = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mu } ^ { \pi } } \Bigl [ \langle \pi ( \cdot \mid s ) - \pi ^ { \prime } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi ^ { \prime } } ( s , \cdot ) \rangle - \tau \bigl ( h ^ { \pi } ( s ) - h ^ { \pi ^ { \prime } } ( s ) \bigr ) \Bigr ] . } } \end{array}
$$

## 2.2 Mirror geometry and policy mirror descent

Mirror map and standing conditions. The following condition is standard for mirror maps.

Assumption 2.2 (Mirror map). The function $h : \mathbb { R } ^ { | A | }  \mathbb { R } \cup \{ + \infty \}$ is proper, closed, and convex, with domh $, \supseteq [ 0 , 1 ] ^ { | { \cal A } | }$ . It is diferentiable on intdomh and essentially smooth: if $p _ { n } \in$ intdomh and $p _ { n } \to p \in$ bd domh, then $\| \nabla h ( p _ { n } ) \| _ { \infty } \to \infty$

Essential smoothness ensures that a Bregman-proximal update initialized in rint(domh) remains in this set. This boundary condition is automatically satisfied when domh $= \mathbb { R } ^ { | \boldsymbol { A } | }$ , since the domain has no boundary, as in the quadratic mirror map used by PQA. The negative-entropy mirror map $\begin{array} { r } { h ( p ) = \sum _ { a } p ( a ) \log p ( a ) } \end{array}$ on $[ 0 , \infty ) ^ { | { \cal A } | }$ is also essentially smooth because 1 + log $p ( a ) \to - \infty$ as $p ( a ) \downarrow 0$

Assumption 2.2 implies that h is bounded on the simplex. Indeed, since h is closed, it attains a finite minimum on the compact set $\Delta ( \mathcal { A } )$ , while convexity gives

$$
h ( p ) \leq \sum _ { a } p ( a ) h ( e _ { a } ) \leq \operatorname* { m a x } _ { a } h ( e _ { a } ) ,
$$

where $e _ { a }$ is the a-th vertex of the simplex. Hence, throughout the paper, we set

$$
H _ { h } : = \operatorname* { m a x } _ { p \in \Delta ( \mathcal { A } ) } \left| h ( p ) \right| < \infty .
$$

It follows immediately that all regularized value functions admit uniform bounds.

Lemma 2.3 (Bounded regularized values). Under Assumptions 2.1 and 2.2, for every $\pi \in \Pi$ 2

$$
| V _ { \tau } ^ { \pi } ( s ) | \leq \frac { 1 + \tau H _ { h } } { 1 - \gamma } , \qquad | Q _ { \tau } ^ { \pi } ( s , a ) | \leq \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } .
$$

Policy-distance guarantees and the stochastic analysis additionally use the following condition.

Assumption 2.3 (Strong convexity). The mirror map h is λ-strongly convex on $\Delta ( \mathcal { A } )$ with respect to the $\ell _ { 1 } .$ -norm for some $\lambda > 0$

Bregman geometry and the PMD update. For $q \in$ rint(domh), let

$$
D _ { h } ( p \| q ) : = h ( p ) - h ( q ) - \langle \nabla h ( q ) , p - q \rangle
$$

be the Bregman divergence generated by h. We use the statewise and distribution-averaged notation

$$
D _ { \pi ^ { \prime } } ^ { \pi } ( s ) : = D _ { h } ( \pi ( \cdot \mid s ) \parallel \pi ^ { \prime } ( \cdot \mid s ) ) , \qquad D _ { \pi ^ { \prime } } ^ { \pi } ( \mu ) : = \mathbb { E } _ { s \sim \mu } [ D _ { \pi ^ { \prime } } ^ { \pi } ( s ) ] .
$$

Assumption 2.4 (Policy initialization). Every PMD scheme considered below is initialized with a target policy satisfying

$$
\pi _ { 0 } ( \cdot \mid s ) \in \operatorname { r i n t } ( \operatorname { d o m } h ) \cap \Delta ( { \cal A } ) , \qquad s \in { \mathcal { S } } .
$$

For quadratic geometry, Assumption 2.4 is automatic because $\mathrm { d o m } h = \mathbb { R } ^ { | \boldsymbol { A } | }$ . For negative entropy, it requires a full-support initial target policy. Given a critic $Q _ { \tau } ^ { k }$ and a policy stepsize $\eta > 0 ,$ , PMD updates every state according to

$$
\pi _ { k + 1 } ( \cdot \mid s ) \in \arg \operatorname* { m a x } _ { p \in \Delta ( A ) } \left\{ \langle p , Q _ { \tau } ^ { k } ( s , \cdot ) \rangle - \tau h ( p ) - \frac { 1 } { \eta } D _ { h } ( p \parallel \pi _ { k } ( \cdot \mid s ) ) \right\} .\tag{1}
$$

Note that a maximizer exists under Assumption 2.2. Moreover, when Assumption 2.3 holds, the maximizer is unique, and otherwise $\pi _ { k + 1 } ( \cdot \mid s )$ denotes any maximizer. With negative entropy, $D _ { h }$ is the Kullback–Leibler divergence and the update recovers the geometry of entropy-regularized natural policy gradient [Kakade, 2001; Cen et al., 2022]; with a quadratic $h ,$ it is a Euclidean proximal policy update.

Classical PMD typically constructs $Q _ { \tau } ^ { k }$ by exact or controlled-inexact policy evaluation. In this paper, $Q _ { \tau } ^ { k }$ is instead a TD critic and need not equal $Q _ { \tau } ^ { \pi _ { k } }$ . The critic therefore evolves together with the policy sequence, rather than serving as an exact evaluator of each current policy. The main technical tool is the three-point inequality associated with the statewise PMD subproblem.

Lemma 2.4 (Three-point inequality [Xiao, 2022; Lan, 2023; Jia and Lan, 2026]). Suppose Assumption 2.2 holds and $\pi _ { k } ( \cdot \mid s ) \in \operatorname { r i n t } ( \operatorname { d o m } h ) \cap \Delta ( A )$ . Then the update (1) satisfies $\pi _ { k + 1 } ( \cdot \mid s ) \in \operatorname { r i n t } ( \operatorname { d o m } h ) \cap \Delta ( A )$ and, for every $p \in \Delta ( \mathcal { A } )$ ，

$$
\begin{array} { r } { \langle \pi _ { k + 1 } ( \cdot \ \vert \ \boldsymbol { s } ) - p , Q _ { \tau } ^ { k } ( \boldsymbol { s } , \cdot ) \rangle - \tau \left( h ^ { \pi _ { k + 1 } } ( \boldsymbol { s } ) - h ( p ) \right) \geq \eta ^ { - 1 } \Big [ D _ { \pi _ { k } } ^ { \pi _ { k + 1 } } ( \boldsymbol { s } ) + ( 1 + \eta \tau ) D _ { \pi _ { k + 1 } } ^ { p } ( \boldsymbol { s } ) - D _ { \pi _ { k } } ^ { p } ( \boldsymbol { s } ) \Big ] , } \end{array}
$$

where $D _ { \pi } ^ { p } ( s ) : = D _ { h } ( p \| \pi ( \cdot \mid s ) )$ .

## 3 Exact TD–PMD with Coordinate-Wise Bellman Updates

Regularized TD–PMD with coordinate-wise critic updates is given in Algorithm 1. At each iteration, after the policy update, the critic is advanced by a single Bellman update rather than being evaluated to its fixed point. The weight matrix W allows diferent update magnitudes across state–action coordinates and satisfies

$$
W : = \mathrm { d i a g } ( w ( s , a ) : ( s , a ) \in \mathcal { S } \times \mathcal { A } ) , \qquad 0 < \underline { { w } } : = \operatorname* { m i n } _ { s , a } w ( s , a ) \leq \overline { { w } } : = \operatorname* { m a x } _ { s , a } w ( s , a ) \leq 1 .\tag{2}
$$

It is evident that $W = I$ corresponds to the full one-step Bellman update, whereas a general diagonal W allows coordinate-wise partial updates. Moreover, since $W$ is invertible, for every fixed policy $\pi _ { \mathrm { : } }$

$$
Q = Q + W ( { \mathcal { F } } _ { \tau } ^ { \pi } Q - Q ) \quad \Longleftrightarrow \quad { \mathcal { F } } _ { \tau } ^ { \pi } Q = Q .\tag{3}
$$

Thus coordinate-wise weighting changes the dynamics and the rate, but not the policy-evaluation fixed point $Q _ { \tau } ^ { \pi }$ or the optimal fixed point $Q _ { \tau } ^ { * }$

Algorithm 1 Exact Regularized TD–PMD with Coordinate-Wise Bellman Updates   
Input: Iterations $K ,$ , initial finite critic $Q _ { \tau } ^ { 0 }$ , policy $\pi _ { 0 } ,$ , policy stepsize $\eta > 0 ,$ and diagonal update matrix W   
satisfying (2).   
for $k = 0 , 1 , \ldots , K - 1$ do   
(Policy update) For every $s \in S ,$ set   
$\pi _ { k + 1 } ( \cdot \mid s ) \in \underset { p \in \Delta ( { \cal A } ) } { \arg \operatorname* { m a x } } \left\{ \langle p , Q _ { \tau } ^ { k } ( s , \cdot ) \rangle - \tau h ( p ) - \eta ^ { - 1 } D _ { h } ( p \parallel \pi _ { k } ( \cdot \mid s ) ) \right\} .$   
(Critic update) Set   
$\begin{array} { r } { Q _ { \tau } ^ { k + 1 } = Q _ { \tau } ^ { k } + W \left( \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \right) . } \end{array}$   
end for   
Output: Last-iterate policy $\pi _ { K }$ and critic $Q _ { \tau } ^ { K }$

The value-convergence guarantee of Algorithm 1 is stated in Theorem 3.4, followed by the action-value, policy, and unregularized corollaries. To prepare for the proof of Theorem 3.4, we first isolate three technical lemmas. Lemma 3.1 handles an arbitrary initial critic, Lemma 3.2 allows the error to be evaluated under an arbitrary state distribution, and Lemma 3.3 combines the critic and policy errors into a single potential recursion. Iterating this recursion gives the value bound in Theorem 3.4 under the basic mirror-map condition. The Bellman representation then gives the action-value bound in Corollary 3.5, while the additional strong convexity condition converts the Bregman term into the policy-distance bound in Corollary 3.6. Throughout the rest of this section, assume that Assumptions 2.2 and 2.4 hold, W satisfies $( 2 )$ , and $Q _ { \tau } ^ { 0 }$ is finite.

The following Bellman violation quantity plays an important role in the analysis:

$$
\delta _ { k } : = \frac { \| W [ Q _ { \tau } ^ { k } - \mathcal { F } _ { \tau } ^ { \pi _ { k } } Q _ { \tau } ^ { k } ] + \| _ { \infty } } { \underline { { w } } ( 1 - \gamma ) } , \qquad k \geq 0 .\tag{4}
$$

It measures how far the current critic is from being a Bellman subsolution for $\pi _ { k }$ that satisfies $Q _ { \tau } ^ { k } \le { \mathcal { F } } _ { \tau } ^ { \pi _ { k } } Q _ { \tau } ^ { k }$ corresponding to $\delta _ { k } = 0$ . Lemma 3.1 shows that $\delta _ { k }$ decays linearly and can be used to characterize the diference between $Q _ { \tau } ^ { k }$ and $Q _ { \tau } ^ { \pi _ { k } }$

Lemma 3.1 (Bellman violation and critic comparison). For every $k \geq 0$

$$
0 \leq \delta _ { k + 1 } \leq [ 1 - \underline { { w } } ( 1 - \gamma ) ] \delta _ { k } , \qquad Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \leq Q _ { \tau } ^ { \pi _ { k } } \leq Q _ { \tau } ^ { * } .\tag{5}
$$

Moreover, $Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \leq Q _ { \tau } ^ { \pi _ { k + 1 } }$

In order to handle a general $\mu \in \Delta ( \mathcal { S } )$ , we need to introduce an auxiliary distribution that is tied to $\mu { : }$

$$
( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } : = \left( 1 - \frac { \gamma } { \xi } \right) \sum _ { t = 0 } ^ { \infty } \left( \frac { \gamma } { \xi } \right) ^ { t } \mu ^ { \top } ( P _ { s } ^ { \pi _ { \tau } ^ { * } } ) ^ { t } ,
$$

where $\xi \in ( \gamma , 1 )$ is fixed. Define

$$
\gamma _ { \mu , \xi } : = \xi - ( \xi - \gamma ) \operatorname* { m i n } _ { s : \nu _ { \mu , \xi } ^ { * } ( s ) > 0 } \frac { \mu ( s ) } { \nu _ { \mu , \xi } ^ { * } ( s ) } .
$$

Then $\gamma _ { \mu , \xi } \in [ \gamma , \xi ]$ . The next lemma gives the relations between $\mu$ and $\nu _ { \mu , \xi } ^ { * } .$ The first relation shows exactly how one transition under $\pi _ { \tau } ^ { * }$ links $\nu _ { \mu , \xi } ^ { * }$ back to itself and to $\mu ,$ and this is the identity that replaces stationarity. The scalar $\gamma _ { \mu , \xi }$ records the loss incurred when comparing their weights, as quantified by the second relation.

Lemma 3.2 (Change of distribution). For every $s \in S$ , the preceding definitions satisfy

$$
\gamma ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } P _ { s } ^ { \pi _ { \tau } ^ { * } } = \xi ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } - ( \xi - \gamma ) \mu ^ { \top } , \qquad ( \xi - \gamma ) \mu ( s ) \ge ( \xi - \gamma _ { \mu , \xi } ) \nu _ { \mu , \xi } ^ { * } ( s )\tag{6}
$$

Define the potential function

$$
\mathcal { L } _ { k } : = \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \big ( Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) + \delta _ { k } \big ) + \operatorname* { m a x } \{ \gamma _ { \mu , \xi } ( \eta ^ { - 1 } + \tau ) , \eta ^ { - 1 } \} D _ { \pi _ { k } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) ,\tag{7}
$$

and, for brevity, set

$$
\rho : = 1 - \underline { { w } } \cdot \operatorname* { m i n } \Biggl \{ 1 - \gamma _ { \mu , \xi } , ~ \frac { \eta \tau } { 1 + \eta \tau } \Biggr \} .\tag{8}
$$

Lemma 3.3 (Potential recursion). For every $k \geq 0$

$$
\mathcal { L } _ { k + 1 } \leq \rho \mathcal { L } _ { k } + ( 1 - \gamma ) \left( 1 - \underline { { w } } \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \right) \delta _ { k } .\tag{9}
$$

For $k \geq 0$ , define

$$
\mathcal { C } _ { k } : = \rho ^ { k } \mathcal { L } _ { 0 } + ( 1 - \gamma ) \left( 1 - \underline { { w } } \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \right) \sum _ { j = 0 } ^ { k - 1 } \rho ^ { k - 1 - j } \delta _ { j } .\tag{10}
$$

Furthermore, for any finite set X and distributions $\zeta , \nu \in \Delta ( \mathcal { X } )$ satisfying supp $( \zeta ) \subseteq \operatorname { s u p p } ( \nu )$ , define

$$
\left\| \frac { \zeta } { \nu } \right\| _ { \infty } : = \operatorname* { m a x } _ { x \in \mathrm { s u p p } ( \nu ) } \frac { \zeta ( x ) } { \nu ( x ) } .\tag{11}
$$

We are now ready to present the main result of this section.

Theorem 3.4 (Value convergence under arbitrary evaluation distributions). Suppose Assumptions 2.1, 2.2, and $\it 2 . 4$ hold, let W satisfy (2), and let $Q _ { \tau } ^ { 0 }$ be finite. Fix $\mu \in \Delta ( \mathcal { S } )$ and $\xi \in ( \gamma , 1 )$ . For every $k \geq 0$

$$
0 \leq V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \mu ) \leq \left\| { \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } } \right\| _ { \infty } { \mathcal { C } } _ { k } .\tag{12}
$$

In particular,

$$
0 \leq V _ { \tau } ^ { \ast } ( \nu ^ { \ast } ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \nu ^ { \ast } ) \leq \mathcal { C } _ { k } .
$$

Here $\mathcal { C } _ { k }$ is evaluated with $\mu = \nu ^ { * }$ , so that $\nu _ { \mu , \xi } ^ { * } = \nu ^ { * }$ and $\gamma _ { \mu , \xi } = \gamma$

Structure and decay of $\mathcal { C } _ { k }$ . The bound $\mathcal { C } _ { k }$ consists of the initial-error term $\rho ^ { k } \mathcal { L } _ { 0 }$ and a term accumulating the Bellman violations. The coeficient of the latter term measures variation in the coordinate weights under $\nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s )$ , as seen from the identity

$$
1 - \underline { { w } } \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } = \sum _ { s , a } \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) \left( 1 - \frac { w } { w ( s , a ) } \right) .
$$

This coeficient is nonnegative and vanishes when all coordinate weights are equal, in which case $\mathcal { C } _ { k } = \rho ^ { k } \mathcal { L } _ { 0 }$ For general coordinate weights, Lemma 3.1 yields $\delta _ { j } \leq [ 1 - \underline { { w } } ( 1 - \gamma ) ] ^ { j } \delta _ { 0 }$ . Since $1 - \underline { w } ( 1 - \gamma ) \le \rho < 1$ , it follows that

$$
\mathcal { C } _ { k } = \left\{ { O } ( \rho ^ { k } ) , \begin{array} { l l } { \mathrm { i f ~ } 1 - \underline { { w } } ( 1 - \gamma ) < \rho , } \\ { O ( ( k + 1 ) \rho ^ { k } ) , } & { \mathrm { i f ~ } 1 - \underline { { w } } ( 1 - \gamma ) = \rho . } \end{array}  \right.
$$

Thus the bound enjoys linear convergence for every finite initial critic.

For $W = I ,$ the contraction factor reduces to

$$
\rho = \operatorname* { m a x } \{ \gamma _ { \mu , \xi } , ( 1 + \eta \tau ) ^ { - 1 } \} .
$$

Moreover, the Bellman-violation term vanishes, so $\mathcal { C } _ { k } = \rho ^ { k } \mathcal { L } _ { 0 } . \mathrm { ~ H } , \mu = \nu ^ { * }$ , where $\nu ^ { * }$ is stationary under $\pi _ { \tau } ^ { * }$ then $\gamma _ { \mu , \xi } = \gamma$ , and the density-ratio factor equals one. The value bound therefore simplifies to

$$
0 \leq V _ { \tau } ^ { * } ( \nu ^ { * } ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \nu ^ { * } ) \leq \rho ^ { k } \mathcal { L } _ { 0 } , \qquad \rho = \operatorname* { m a x } \left\{ \gamma , ( 1 + \eta \tau ) ^ { - 1 } \right\} .
$$

For $1 + \eta \tau \geq \gamma ^ { - 1 }$ , this factor equals $\gamma ,$ matching the rate in the strongly convex PMD theorem of Lan [2023]. For smaller stepsizes, the factor is governed by $( 1 + \eta \tau ) ^ { - 1 }$ . Thus the bound covers every $\eta > 0$ . For comparison, Zhan et al. [2023] obtain the linear-convergence factor $1 - \eta \tau ( 1 - \gamma ) / ( 1 + \eta \tau ) \geq \operatorname* { m a x } \{ \gamma , ( 1 + \eta \tau ) ^ { - 1 } \}$ .

Choice of $\xi \cdot$ First note that the density-ratio factor in Theorem 3.4 is bounded even when $\mu$ lacks full support. Indeed, by the definition of $\nu _ { \mu , \xi } ^ { * }$ , we have $\nu _ { \mu , \xi } ^ { * } ( s ) \geq ( 1 - \gamma / \xi ) \mu ( s )$ for every state, and hence

$$
\left. \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } \right. _ { \infty } \leq \frac { \xi } { \xi - \gamma } < \infty .\tag{13}
$$

Moreover, the bounds display a finite-time tradeof. The inequality $\gamma _ { \mu , \xi } \leq \xi \ \mathrm { g i }$ ves

$$
\rho \leq 1 - { \underline { { w } } } \cdot \operatorname* { m i n } \left\{ 1 - \xi , \ { \frac { \eta \tau } { 1 + \eta \tau } } \right\} .
$$

Choosing $\xi$ closer to $\gamma$ decreases the critic term in this upper bound. On the other hand, the universal density-ratio bound $\xi / ( \xi - \gamma )$ decreases as $\xi$ increases and diverges as $\xi \downarrow \gamma$ . The midpoint choice $\xi = ( 1 + \gamma ) / 2$ gives the simple bounds

$$
\gamma _ { \mu , \xi } \leq \frac { 1 + \gamma } { 2 } , \qquad \left\| \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } \right\| _ { \infty } \leq \frac { 1 + \gamma } { 1 - \gamma } .
$$

Coordinate weighting can improve convergence. Although $W = I$ yields the smallest contraction factor in the theoretical bound, nonuniform coordinate weights can nevertheless lead to faster convergence on certain MDPs. The following example illustrates this. Consider a two-state, two-action MDP with $\boldsymbol { S } = \{ s _ { 1 } , s _ { 2 } \}$ and $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } \}$ . Both actions move deterministically to the other state, and, for $i = 1 , 2$ , let

$$
\begin{array} { r } { r ( s _ { i } , a _ { 1 } ) = \frac 1 2 , \qquad r ( s _ { i } , a _ { 2 } ) = 0 , \qquad \gamma = \frac 1 2 , \qquad \tau = \eta = 1 , \qquad h ( p ) = \frac 1 2 \| p \| _ { 2 } ^ { 2 } . } \end{array}
$$

For a policy $\pi ,$ write $p ( s _ { i } ) = \pi ( a _ { 1 } \mid s _ { i } )$ . The regularized one-step reward at $s _ { i }$ is

$$
\frac { p ( s _ { i } ) } { 2 } - \frac { 1 } { 2 } \big [ p ( s _ { i } ) ^ { 2 } + ( 1 - p ( s _ { i } ) ) ^ { 2 } \big ] = \frac { 1 } { 1 6 } - \bigg ( p ( s _ { i } ) - \frac { 3 } { 4 } \bigg ) ^ { 2 } .
$$

Since transitions do not depend on the action, the optimal policy is $\pi _ { \tau } ^ { * } ( \cdot \mid s _ { i } ) = ( 3 / 4 , 1 / 4 )$ at both states. For evaluation distribution $\mu = ( 1 / 2 , 1 / 2 )$ , the value gap is $\begin{array} { r } { V _ { \tau } ^ { \ast } ( \mu ) - V _ { \tau } ^ { \pi } ( \mu ) = \sum _ { i = 1 } ^ { 2 } \left( p ( s _ { i } ) - \frac { 3 } { 4 } \right) ^ { 2 } } \end{array}$ . Let $\pi _ { k }$ and $\pi _ { k } ^ { \prime }$ denote the policy sequences generated by Algorithm 1 with

$$
W = \mathrm { d i a g } ( 3 / 4 , 3 / 4 , 1 , 1 ) , \qquad W ^ { \prime } = I ,\tag{14}
$$

respectively. Write $p _ { k } ( s _ { i } ) = \pi _ { k } ( a _ { 1 } \mid s _ { i } )$ and $p _ { k } ^ { \prime } ( s _ { i } ) = \pi _ { k } ^ { \prime } ( a _ { 1 } \mid s _ { i } )$ for their respective probabilities of choosing $a _ { 1 }$ at $s _ { i }$ . Both runs use the same initialization:

$$
\begin{array} { c c } { Q _ { \tau } ^ { 0 } = \binom { 3 / 4 } { 1 } \ } \end{array} , \qquad p _ { 0 } ( s _ { i } ) = p _ { 0 } ^ { \prime } ( s _ { i } ) = \frac { 1 } { 2 } , \quad i = 1 , 2 .\tag{15}
$$

Both actions at each state lead to the same next state, so their Bellman targets difer only in the immediate reward. At each state, the two actions have equal update weights in each run, so subtracting the critic updates gives

$$
\begin{array} { r } { Q _ { \tau } ^ { k + 1 } ( s _ { i } , a _ { 1 } ) - Q _ { \tau } ^ { k + 1 } ( s _ { i } , a _ { 2 } ) = \big ( 1 - w ( s _ { i } , a _ { 1 } ) \big ) \big [ Q _ { \tau } ^ { k } ( s _ { i } , a _ { 1 } ) - Q _ { \tau } ^ { k } ( s _ { i } , a _ { 2 } ) \big ] + \frac { 1 } { 2 } w ( s _ { i } , a _ { 1 } ) . } \end{array}\tag{16}
$$

The policy update along these trajectories is

$$
\begin{array} { r } { p _ { k + 1 } ( s _ { i } ) = \frac { 1 } { 2 } p _ { k } ( s _ { i } ) + \frac { 1 } { 4 } + \frac { 1 } { 4 } \big [ Q _ { \tau } ^ { k } ( s _ { i } , a _ { 1 } ) - Q _ { \tau } ^ { k } ( s _ { i } , a _ { 2 } ) \big ] . } \end{array}
$$

Applying (16) with (14) and (15), we obtain, for every $k \geq 1$ 2

$$
\begin{array} { r } { p _ { k } ( s _ { 1 } ) = \frac { 3 } { 4 } - \frac { 1 } { 4 } 4 ^ { - k } , \qquad p _ { k } ^ { \prime } ( s _ { 1 } ) = \frac { 3 } { 4 } - \frac { 1 } { 8 } 2 ^ { - k } , \qquad p _ { k } ( s _ { 2 } ) = p _ { k } ^ { \prime } ( s _ { 2 } ) = \frac { 3 } { 4 } . } \end{array}\tag{17}
$$

Substituting (17) into the value gap gives, for every $k \geq 1$

$$
\begin{array} { r l r } { V _ { \tau } ^ { \ast } ( \mu ) - V _ { \tau } ^ { \pi _ { k } } ( \mu ) = \frac { 1 } { 1 6 } 1 6 ^ { - k } , } & { { } } & { V _ { \tau } ^ { \ast } ( \mu ) - V _ { \tau } ^ { \pi _ { k } ^ { \prime } } ( \mu ) = \frac { 1 } { 6 4 } 4 ^ { - k } . } \end{array}
$$

Thus W gives a strictly smaller value gap than $W ^ { \prime }$ for every $k \geq 2$ and improves its exact exponential factor from $1 / 4$ to $1 / 1 6$

Fix $( s , a )$ and set $\mu = P ( \cdot \mid s , a )$ . By the Bellman equations, one has

$$
Q _ { \tau } ^ { \ast } ( s , a ) - Q _ { \tau } ^ { \pi _ { k + 1 } } ( s , a ) = \gamma \big [ V _ { \tau } ^ { \ast } ( \mu ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \mu ) \big ] .
$$

Applying Theorem 3.4 establishes the convergence of the action values.

Corollary 3.5 (Action-value convergence). Suppose Assumptions 2.1, 2.2, and 2.4 hold, let W satisfy (2), and let $Q _ { \tau } ^ { 0 }$ be finite. Fix $( s , a ) \in S \times A , \xi \in ( \gamma , 1 )$ , and set $\mu = P ( \cdot \mid s , a )$ . With $\nu _ { \mu , \xi } ^ { \ast } , \gamma _ { \mu , \xi } ,$ , and $\mathcal { C } _ { k }$ evaluated at this choice of $\mu ,$ , we obtain

$$
0 \leq Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { \pi _ { k + 1 } } ( s , a ) \leq \gamma \left\| \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } \right\| _ { \infty } \mathcal { C } _ { k } .\tag{18}
$$

If we further assume the strong convexity of $h$ (Assumption 2.3), then by the definition of $\mathcal { L } _ { k }$

$$
\frac { \lambda } 2 \mathbb { E } _ { s \sim \nu _ { \mu , \xi } ^ { * } } \big [ \| \pi _ { k } ( \cdot \mid s ) - \pi _ { \tau } ^ { * } ( \cdot \mid s ) \| _ { 1 } ^ { 2 } \big ] \leq D _ { \pi _ { k } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) \leq \frac { \eta } { \operatorname* { m a x } \{ \gamma _ { \mu , \xi } ( 1 + \eta \tau ) , 1 \} } \mathcal { L } _ { k } .
$$

Using $\mu ( s ) \leq \| \mu / \nu _ { \mu , \xi } ^ { * } \| _ { \infty } \nu _ { \mu , \xi } ^ { * } ( s )$ for every state transfers this estimate to the expectation under $\mu .$ . The proof of Theorem 3.4 establishes the bound $\mathcal { L } _ { k } \leq \mathcal { C } _ { k }$ in (52), which immediately yields the policy convergence of Algorithm 1.

Corollary 3.6 (Policy convergence under strong convexity). Fix $\mu \in \Delta ( \mathcal { S } )$ and $\xi \in ( \gamma , 1 )$ . Suppose in addition that Assumption 2.3 holds. Then, for every $k \geq 0$

$$
\mathbb { E } _ { s \sim \mu } \left[ \| \pi _ { k } ( \cdot \mid s ) - \pi _ { \tau } ^ { * } ( \cdot \mid s ) \| _ { 1 } ^ { 2 } \right] \leq \frac { 2 \eta } { \lambda \operatorname* { m a x } \{ \gamma _ { \mu , \xi } ( 1 + \eta \tau ) , 1 \} } \left\| \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } \right\| _ { \infty } \mathcal { C } _ { k } .\tag{19}
$$

The argument used to prove Theorem 3.4 also applies when $\tau = 0 .$ , and yields two types of convergence: a constant stepsize gives an averaged sublinear rate, whereas geometrically increasing stepsizes give a last-iterate linear rate. This agrees with the behavior established for unregularized one-step TD–PMD by Liu et al. [2025b]. For clarity, the proof of the following result is also included in Appendix A.

Corollary 3.7 (Unregularized exact TD–PMD). Consider the variable-stepsize variant of Algorithm 1 with $\tau = 0$ and $W = I$ , where the policy stepsize at iteration k is $\eta _ { k } > 0 .$ , and omit the subscript τ. Fix an unregularized optimal policy $\pi ^ { * }$ and a stationary distribution $\nu ^ { * } ~ o f ~ P _ { s } ^ { \pi ^ { * } }$ , and let $\delta _ { k }$ be defined by (4) with $\tau = 0$ and $W = I$ $I f \eta _ { k } \equiv \eta > 0$ , then, for every $K \geq 1$

$$
\frac { 1 } { K } \sum _ { k = 0 } ^ { K - 1 } \left[ V ^ { * } ( \nu ^ { * } ) - V ^ { \pi _ { k + 1 } } ( \nu ^ { * } ) \right] \leq \frac { \sum _ { s , a } \nu ^ { * } ( s ) \pi ^ { * } ( a \mid s ) \big ( Q ^ { * } ( s , a ) - Q ^ { 0 } ( s , a ) + \delta _ { 0 } \big ) + \eta ^ { - 1 } D _ { \pi _ { 0 } } ^ { \pi ^ { * } } ( \nu ^ { * } ) } { ( 1 - \gamma ) K } ,\tag{20}
$$

where $V ^ { \pi }$ and $V ^ { * }$ denote the unregularized value function $o f \pi$ and optimal value function, respectively. If instead the positive stepsizes satisfy $\eta _ { k + 1 } \ge \eta _ { k } / \gamma$ for every $k \geq 0$ , then the last iterate satisfies

$$
V ^ { * } ( \nu ^ { * } ) - V ^ { \pi _ { k + 1 } } ( \nu ^ { * } ) \leq \gamma ^ { k } \left\{ \sum _ { s , a } \nu ^ { * } ( s ) \pi ^ { * } ( a \mid s ) \big ( Q ^ { * } ( s , a ) - Q ^ { 0 } ( s , a ) + \delta _ { 0 } \big ) + \eta _ { 0 } ^ { - 1 } D _ { \pi _ { 0 } } ^ { \pi ^ { * } } ( \nu ^ { * } ) \right\} .\tag{21}
$$

## 4 Regularized TD–PMD under Of-Policy Markov Data

In this section, we consider the sample complexity of regularized TD–PMD under of-policy Markov data. The method is summarized in Algorithm 2 and maintains a target policy $\pi _ { k }$ and a critic $Q _ { \tau } ^ { k }$ , while all data are generated by a fixed behavior policy $\pi _ { b }$ . At iteration k, after updating the target policy from $\pi _ { k }$ to $\pi _ { k + 1 }$ the algorithm collects a batch of $B _ { k }$ consecutive transitions

$$
\tau _ { k } = \{ ( s _ { t } ^ { k } , a _ { t } ^ { k } , r _ { t } ^ { k } , s _ { t + 1 } ^ { k } ) \} _ { t = 0 } ^ { B _ { k } - 1 }
$$

along the behavior trajectory, where $a _ { t } ^ { k } \sim \pi _ { b } ( \cdot \mid s _ { t } ^ { k } )$ and $s _ { t + 1 } ^ { k } \sim P ( \cdot \mid s _ { t } ^ { k } , a _ { t } ^ { k } )$ . The trajectory is not reset between iterations: the terminal state of the current batch becomes the initial state of the next one, $s _ { 0 } ^ { k + 1 } = s _ { B _ { k } } ^ { k }$ For each sampled transition, we form a coordinate-wise TD increment whose target is

$$
r _ { t } ^ { k } + \gamma \mathbb { E } _ { a \sim \pi _ { k + 1 } ( \cdot | s _ { t + 1 } ^ { k } ) } \big [ Q _ { \tau } ^ { k } ( s _ { t + 1 } ^ { k } , a ) \big ] - \tau \gamma h ^ { \pi _ { k + 1 } } ( s _ { t + 1 } ^ { k } ) .
$$

The resulting TD increments are aggregated using deterministic weights $\{ c _ { t } ^ { k } \} _ { t = 0 } ^ { B _ { k } - 1 }$ , and their weighted average is used to perform one critic update with stepsize $\alpha _ { k }$

Assumption 4.1 (Behavior-policy coverage and ergodicity). The behavior policy has full support: $\pi _ { b } ( a \mid$ $s ) > 0$ for every $( s , a ) \in \mathcal S \times \mathcal A$ . The transition matrix $P _ { s } ^ { \pi _ { b } }$ is irreducible and aperiodic.

Under Assumption 4.1, the behavior chain has a unique stationary distribution $\nu ^ { \pi _ { b } } \in \Delta ( \mathcal { S } )$ with strictly positive entries [Levin and Peres, 2017, Corollary 1.17 and Proposition 1.19], satisfying

$$
( \nu ^ { \pi _ { b } } ) ^ { \top } P _ { s } ^ { \pi _ { b } } = ( \nu ^ { \pi _ { b } } ) ^ { \top } , \qquad \nu ^ { \pi _ { b } } ( s ) > 0 \quad ( s \in \mathcal { S } ) .\tag{23}
$$

Moreover, by Levin and Peres [2017, Theorem 4.9], there exist constants $m _ { b } > 0$ and $\kappa _ { b } \in ( 0 , 1 )$ such that

$$
\begin{array} { r } { d _ { \mathrm { T V } } \Big ( \big ( P _ { s } ^ { \pi _ { b } } \big ) ^ { t } ( s , \cdot ) , \nu ^ { \pi _ { b } } \Big ) \leq m _ { b } \kappa _ { b } ^ { t } , \qquad s \in \mathcal { S } , \quad t \geq 0 . } \end{array}\tag{24}
$$

We also assume the following standard conditions on the initialization, critic stepsizes, batch lengths, and within-batch averaging weights. Note that the restriction $Q _ { \tau } ^ { 0 } = 0$ in the assumption is made only to simplify the displayed constants.

Algorithm 2 Finite-Batch Of-Policy Expected TD–PMD   
Input: Iterations $K ,$ , initial action-value vector $Q _ { \tau } ^ { 0 } = 0 .$ , initial policy π<sub>0</sub>, critic stepsizes $\{ \alpha _ { k } \}$ , constant policy   
stepsize $\eta > 0 ,$ , initial state $s _ { 0 } ,$ , batch sizes $\{ B _ { k } \}$ , and averaging weights $\{ c _ { t } ^ { k } \}$   
Set $s _ { 0 } ^ { 0 } = s _ { 0 }$   
for $k = 0 , 1 , \ldots , K - 1$ do   
(Policy update) Update the target policy by   
$\forall s \in S : \quad \pi _ { k + 1 } ( \cdot | s ) = \underset { p \in \Delta ( { \mathcal A } ) } { \arg \operatorname* { m a x } } \left\{ \left. p , Q _ { \tau } ^ { k } ( s , \cdot ) \right. - \tau h ( p ) - \frac { 1 } { \eta } D _ { \pi _ { k } } ^ { p } ( s ) \right\} .$   
(Sampling) Starting from $s _ { 0 } ^ { k } ,$ collect the consecutive batch $\tau _ { k }$ under the behavior policy $\pi ,$   
$\tau _ { k } = \{ \left( s _ { t } ^ { k } , a _ { t } ^ { k } , r _ { t } ^ { k } , s _ { t + 1 } ^ { k } \right) \} _ { t = 0 } ^ { B _ { k } - 1 } , \quad \mathrm { w h e r e ~ } r _ { t } ^ { k } = r ( s _ { t } ^ { k } , a _ { t } ^ { k } ) , a _ { t } ^ { k } \sim \pi _ { b } ( \cdot | s _ { t } ^ { k } ) , s _ { t + 1 } ^ { k } \sim P ( \cdot | s _ { t } ^ { k } , a _ { t } ^ { k } ) ,$   
and set $s _ { 0 } ^ { k + 1 } = s _ { B _ { k } } ^ { k }$   
(Critic update) Construct the expected TD error,   
$\begin{array} { r } { \delta _ { t } ^ { k } ( s , a ) : = \Im [ ( s _ { t } ^ { k } , a _ { t } ^ { k } ) = ( s , a ) ] \cdot \left[ r _ { t } ^ { k } + \gamma \mathbb { E } _ { a \sim \pi _ { k + 1 } ( \cdot \vert s _ { t + 1 } ^ { k } ) } \left[ Q _ { \tau } ^ { k } ( s _ { t + 1 } ^ { k } , a ) \right] - \tau \gamma h ^ { \pi _ { k + 1 } } ( s _ { t + 1 } ^ { k } ) - Q _ { \tau } ^ { k } ( s _ { t } ^ { k } , a _ { t } ^ { k } ) \right] } \end{array}$   
$\bar { \delta } _ { k } ( s , a ) : = \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \cdot \delta _ { t } ^ { k } ( s , a ) .$   
Update the critic by   
$Q _ { \tau } ^ { k + 1 } ( s , a ) = Q _ { \tau } ^ { k } ( s , a ) + \alpha _ { k } \cdot \bar { \delta } _ { k } ( s , a ) .$   
end for   
Independently sample $\widehat { K } \in \{ 0 , \ldots , K - 1 \}$ according to   
$\mathbb { P } ( \widehat { K } = k ) = \frac { ( 1 - \rho ) \rho ^ { K - k - 1 } } { 1 - \rho ^ { K } } , \qquad k = 0 , \dots , K - 1 , \quad \rho = ( 1 + \eta \tau ) ^ { - 1 } .$ (22)   
Output $\pi _ { \widehat { K } } .$

Assumption 4.2 (Algorithmic conditions). The critic initialization and parameters satisfy $Q _ { \tau } ^ { 0 } = 0 , 0 <$ $\alpha _ { k } = \alpha \leq 1$ , and $B _ { k } \geq 1$ . The weights $\{ c _ { t } ^ { k } \} _ { t = 0 } ^ { \dot { B } _ { k } - 1 }$ are deterministic and satisfy $c _ { t } ^ { k } \geq 0$ and $\begin{array} { r } { \sum _ { t = 0 } ^ { B _ { k } - 1 } \dot { c } _ { t } ^ { k } = 1 } \end{array}$

To proceed, define the following quantities based on the stationary distribution $\nu ^ { \pi _ { b } }$ :

$$
\begin{array} { r l } & { \sigma ^ { \pi _ { b } } ( s , a ) : = \nu ^ { \pi _ { b } } ( s ) \pi _ { b } ( a \mid s ) , \qquad \Sigma _ { b } : = \mathrm { d i a g } ( \sigma ^ { \pi _ { b } } ( s , a ) : s , a ) , } \\ & { \qquad \widetilde { \sigma } _ { b } : = \displaystyle \operatorname* { m i n } _ { s , a } \sigma ^ { \pi _ { b } } ( s , a ) > 0 , \qquad \overline { { \sigma } } _ { b } : = \displaystyle \operatorname* { m a x } _ { s , a } \sigma ^ { \pi _ { b } } ( s , a ) . } \end{array}\tag{25}
$$

If each sampled TD increment is replaced by its expectation under the stationary behavior distribution, it is not hard to see that the critic update reduces to the coordinate-wise update in Algorithm 1 with $W = \alpha \Sigma _ { b }$ where the coordinate weights are determined by the stationary state–action visitation probabilities. Thus Theorem 3.4 immediately yields a noise-free value-convergence benchmark.

Corollary 4.1 (Noise-free behavior-weighted value convergence). Suppose the assumptions of Theorem $\it 3 . 4$ and Assumption 4.1 hold. Fix $\xi \in ( \gamma , 1 )$ , and let $\boldsymbol { W } = \alpha \boldsymbol { \Sigma } _ { b }$ with $0 < \alpha \overline { { \sigma } } _ { b } \le 1$ . Consider the stationary distribution $\nu ^ { * }$ of $P _ { s } ^ { \pi _ { \tau } ^ { * } }$ for simplicity. Since $\nu _ { \nu ^ { * } , \xi } ^ { * } = \nu ^ { * }$ and $\gamma _ { \nu ^ { * } , \xi } = \gamma _ { \mathrm { \scriptscriptstyle 3 } }$ , a direct application of Theorem $\it 3 . 4$ gives

$$
0 \leq V _ { \tau } ^ { \ast } ( \nu ^ { \ast } ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \nu ^ { \ast } ) \leq \mathcal { C } _ { k } ,
$$

$$
\mathcal { C } _ { k }
$$

$$
\mu = \nu ^ { * }
$$

$$
\boldsymbol { W } = \alpha \boldsymbol { \Sigma } _ { b }
$$

$$
\underline { w } = \alpha \widetilde \sigma _ { b }
$$

$$
\overline { { w } } = \alpha \overline { { \sigma } } _ { b }
$$

Next, we are going to show that the stochastic variant (i.e., Algorithm 2) attains an expected value gap of at most ϵ after $\begin{array} { r } { \widetilde { \mathcal O } \left( \frac { 1 } { ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon } \right) } \end{array}$ observed transitions. The analysis requires bounds on the additional errors caused by finite batches, temporal dependence, and nonstationary batch initialization. As in the previous section, we first present the technical results used in the proof of the main result. The first result shows that the critic iterates are uniformly bounded.

Lemma 4.2 (Bounded critic iterates). Under Assumption 4.2, for every $k \geq 0$

$$
| Q _ { \tau } ^ { k } ( s , a ) | \leq \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } , \qquad ( s , a ) \in \mathcal { S } \times \mathcal { A } .
$$

Because the target policy changes from one batch to the next, we also need to control the variation of the policy iterates and the exact action-value targets along the PMD trajectory. The following lemma derives this trajectory-local regularity from the Bregman geometry of the iterates, without imposing a global Lipschitz condition on the regularizer or its gradient.

Lemma 4.3 (Trajectory regularity along the PMD iterates). Define

$$
L _ { h } : = H _ { h } + \frac { 1 } { 2 } \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { s \in S , a \in \mathcal { A } } D _ { h } ( e _ { a } \parallel \pi _ { 0 } ( \cdot \mid s ) ) , \frac { 2 ( 1 + \tau H _ { h } ) } { \tau ( 1 - \gamma ) } \right\} .\tag{26}
$$

For every $k \geq 0$ and $s \in S$

$$
\begin{array} { r } { \left| h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) \right| \leq L _ { h } \left\| \pi _ { k + 1 } ( \cdot  { \left| \begin{array} { l } { s } \end{array} \right) } - \pi _ { k } ( \cdot  { \left| \begin{array} { l } { s } \end{array} \right) } \right\| _ { 1 } . } \end{array}\tag{27}
$$

Moreover,

$$
\| \pi _ { k + 1 } - \pi _ { k } \| _ { 1 , \infty } \leq \frac { \eta } { \lambda } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) ,\tag{28}
$$

where $\begin{array} { r } { \| \pi - \pi ^ { \prime } \| _ { 1 , \infty } : = \operatorname* { m a x } _ { s \in \mathcal { S } } \| \pi ( \cdot \mid s ) - \pi ^ { \prime } ( \cdot \mid s ) \| _ { 1 } } \end{array}$ , and

$$
\| Q _ { \tau } ^ { \pi _ { k + 1 } } - Q _ { \tau } ^ { \pi _ { k } } \| _ { \infty } \leq \frac { \gamma } { 1 - \gamma } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) \| \pi _ { k + 1 } - \pi _ { k } \| _ { 1 , \infty } .\tag{29}
$$

We also need to separate the deterministic benchmark from the error caused by Markov sampling. Identify $\delta _ { t } ^ { k }$ and $\bar { \delta } _ { k }$ with vectors in $\mathbb { R } ^ { | \boldsymbol { s } | | \boldsymbol { A } | }$ and define the centered errors

$$
\omega _ { t } ^ { k } : = \delta _ { t } ^ { k } - \Sigma _ { b } \bigl ( \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \bigr ) , \qquad \bar { \omega } _ { k } : = \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \omega _ { t } ^ { k } .\tag{30}
$$

The critic recursion is therefore given by

$$
Q _ { \tau } ^ { k + 1 } = Q _ { \tau } ^ { k } + \alpha _ { k } \Sigma _ { b } \big ( \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \big ) + \alpha _ { k } \bar { \omega } _ { k } .\tag{31}
$$

Let

$$
\mathcal { H } _ { k } : = \sigma \big ( s _ { 0 } ^ { 0 } , \big \{ \big ( s _ { t } ^ { \ell } , a _ { t } ^ { \ell } , r _ { t } ^ { \ell } , s _ { t + 1 } ^ { \ell } \big ) : 0 \leq \ell < k , 0 \leq t < B _ { \ell } \big \} \big )\tag{32}
$$

be the information available before the k-th batch is sampled. In particular, $s _ { 0 } ^ { k } , \ Q _ { \tau } ^ { k }$ , and $\pi _ { k + 1 }$ are $\mathcal { H } _ { k } .$ measurable. The conditional sampling-bias can be bounded as follows.

Lemma 4.4 (Conditional bias of the stochastic critic error). Under Assumption 4.2, for every $k \geq 0$ and $( s , a ) \in S \times A$ 2

$$
\left| \mathbb { E } [ \bar { \omega } _ { k } ( s , a ) \mid \mathcal { H } _ { k } ] \right| \leq \frac { 2 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \kappa _ { b } ^ { t } .\tag{33}
$$

Equivalently, the same bound holds for $\| \mathbb { E } [ { \bar { \omega } } _ { k } \mid { \mathcal { H } } _ { k } ] \| _ { \infty }$ , and consequently it also holds for $\left| \mathbb { E } [ { \bar { \omega } } _ { k } ( s , a ) ] \right.$ |.

The factor $\kappa _ { b } ^ { t }$ measures how much dependence on the initial state remains after t steps of the behavior chain, while $c _ { t } ^ { k }$ records how much weight the critic assigns to that sample. Their weighted sum therefore quantifies the conditional bias of one batch. The bias is reduced when the weighting scheme places suficient mass on later, better-mixed samples. The exponential weights used in Theorem 4.5 make this dependence decay geometrically with the batch length.

Theorem 4.5 (Expected value-gap bound). Suppose Assumptions 2.1, 2.2, 2.3, 2.4, 4.1, and $4 . 2$ hold. Fix $\vartheta \in [ 0 , \kappa _ { b } )$ and run Algorithm 2 with $\alpha _ { k } = \alpha , B _ { k } = B$ , and

$$
c _ { t } ^ { k } : = \frac { \vartheta ^ { B - t - 1 } } { \sum _ { \ell = 0 } ^ { B - 1 } \vartheta ^ { \ell } } , \qquad t = 0 , \dots , B - 1 ,\tag{34}
$$

where $0 ^ { 0 } : = 1$ . Let $\rho : = ( 1 + \eta \tau ) ^ { - 1 } \in ( 0 , 1 )$ and assume $\begin{array} { r } { 0 < \eta \leq \frac { \alpha ( 1 - \gamma ) \widetilde \sigma _ { b } } { \tau [ 2 - \alpha ( 1 - \gamma ) \widetilde \sigma _ { b } ] } } \end{array}$ . Then, for every integer $K \geq \lceil \log ( 2 ) / \log ( 1 / \rho ) \rceil$ , the output (22) satisfies

$$
\mathbb { E } \big [ V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi _ { \widehat { \kappa } } } ( \mu ) \big ] \leq \frac { C _ { 1 } } { \rho ^ { - K } - 1 } + C _ { 2 } \eta + C _ { 3 } \frac { \kappa _ { b } ^ { B } } { \kappa _ { b } - \vartheta } .\tag{35}
$$

where $\begin{array} { r l } { C _ { 1 } : = \frac { \tau D _ { v _ { 0 } } ^ { \pi _ { 0 } ^ { * } } ( a _ { \mu } ^ { * } ) } { 1 - \gamma } + \frac { 4 \eta \tau ( 1 + \tau \gamma H _ { h } ) } { \alpha ( 1 - \gamma ) ^ { 3 } \overline { { \eta } } _ { b } } , C _ { 2 } : = \frac { 4 \tau ( 1 + \tau H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } } + \frac { 2 ( 1 + \tau \gamma H _ { h } ) ^ { 2 } } { \lambda ( 1 - \gamma ) ^ { 4 } } + \frac { 2 ( 1 + \tau \gamma H _ { h } ) \left[ 1 + \tau \gamma H _ { h } + \tau ( 1 - \gamma ) L _ { h } \right] } { \lambda \alpha ( 1 - \gamma ) ^ { 4 } \overline { { \eta } } _ { b } } } & { + } \end{array}$ $\begin{array} { r } { \frac { 2 \gamma \left[ 1 + \tau \gamma H _ { h } + \tau ( 1 - \gamma ) L _ { h } \right] \left[ 9 ( 1 + \tau \gamma H _ { h } ) + \tau ( 1 - \gamma ) L _ { h } \right] } { \lambda \alpha ( 1 - \gamma ) ^ { 5 } \widetilde \sigma _ { b } } , \ a n d \ C _ { 3 } : = \frac { 4 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ^ { 3 } \widetilde \sigma _ { b } } . } \end{array}$

Corollary 4.6 (ϵ-accuracy and sample complexity). Assume that the conditions of Theorem 4.5 hold. For any $0 < \epsilon \leq 1$ , choose

$$
\eta _ { \epsilon } : = \frac { \epsilon \lambda \alpha ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } } { 1 2 \left[ 1 + \lambda \tau ( 1 + \tau H _ { h } ) \right] \left[ 1 + \tau \gamma H _ { h } + \tau ( 1 - \gamma ) L _ { h } \right] \left[ 9 ( 1 + \tau \gamma H _ { h } ) + \tau ( 1 - \gamma ) L _ { h } \right] } .\tag{36}
$$

Set

$$
K _ { \epsilon } : = \lceil \frac { 2 } { \eta _ { \epsilon } \tau } \log ( 1 + \operatorname* { m a x } \Biggl \{ 1 , \frac { 3 } { \epsilon } [ \frac { \tau D _ { \pi _ { 0 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) } { 1 - \gamma } + \frac { 4 ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } } ] \} ) \rceil\tag{37}
$$

and

$$
B _ { \epsilon } : = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { \log \left( \operatorname* { m a x } \left\{ 1 , \frac { 1 2 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { \epsilon ( 1 - \gamma ) ^ { 3 } \widetilde { \sigma } _ { b } ( \kappa _ { b } - \vartheta ) } \right\} \right) } { \log ( 1 / \kappa _ { b } ) } \right\rceil \right\} .\tag{38}
$$

With $\eta = \eta _ { \epsilon } , K = K _ { \epsilon } ,$ and $B = B _ { \epsilon }$ , the total number of observed transitions satisfies $N _ { \epsilon } : = K _ { \epsilon } B _ { \epsilon } =$ $\widetilde { \mathcal { O } } \big ( [ ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon ] ^ { - 1 } \big )$ , where $\alpha , \tau , \lambda , H _ { h }$ , ma $\mathfrak { c } _ { s , a } D _ { h } ( e _ { a } \parallel \pi _ { 0 } ( \cdot \mid s ) )$ , $m _ { b } , \kappa _ { b }$ , and ϑ are treated as fixed constants. These transitions sufice to ensure

$$
\mathbb { E } \big [ V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi _ { \widehat { K } } } ( \mu ) \big ] \leq \epsilon .\tag{39}
$$

The terms $C _ { 1 } / ( \rho ^ { - K } - 1 ) , C _ { 2 } \eta$ , and $C _ { 3 } \kappa _ { b } ^ { B } / ( \kappa _ { b } - \vartheta )$ in (35) are, respectively, the decaying optimization error, the constant-stepsize tracking bias, and the finite-batch Markov mixing bias. Increasing K reduces the first term, while decreasing η reduces the tracking bias but slows the contraction $\rho = ( 1 + \eta \tau ) ^ { - 1 }$ . Increasing B reduces the mixing bias exponentially as $\kappa _ { b } ^ { B }$ . The choice $\vartheta = 0$ uses only the last sample in each batch; every fixed $\vartheta \in [ 0 , \kappa _ { b } )$ retains this exponential decay in B, with the additional factor $( \kappa _ { b } - \vartheta ) ^ { - 1 }$ . The critic stepsize α and the coverage $\widetilde { \sigma } _ { b }$ determine how quickly the behavior-weighted critic tracks its target. Corollary 4.6 balances these efects by taking $\eta = \Theta ( \epsilon ) , K = \widetilde { \mathcal { O } } ( 1 / \epsilon )$ , and $B = \mathcal { O } ( \log ( 1 / \epsilon ) )$ while the remaining problem parameters are fixed. Under near-uniform behavior coverage, $\widetilde { \sigma } _ { b } = \Theta \big ( ( | S | | \mathcal { A } | ) ^ { - 1 } \big )$ . Thus, Corollary 4.6 yields the sample complexity $\begin{array} { r } { \widetilde { \mathcal { O } } \left( \frac { | \mathcal { S } | | \mathcal { A } | } { \epsilon ( 1 - \gamma ) ^ { 5 } } \right) } \end{array}$ . This matches the leading-order dependence on $| S | , | A | , 1 - \gamma$ , and ϵ of strongly convex SVMD [Jia and Lan, 2026], while our result is established using only a single of-policy Markov trajectory, without trajectory resets or generative-model access.

## 5 Numerical Experiments

We use a randomly generated MDP with $| \boldsymbol { S } | = 5 0$ and $| { \mathcal { A } } | = 1 0$ to illustrate the behavior of exact coordinate wise TD–PMD and its finite-batch stochastic counterpart under of-policy Markov data. Each reward $r ( s , a )$ is drawn independently from Unif[0, 1]. For each state–action pair $( s , a )$ , we generate $\textstyle P ( \cdot \mid s , a )$ by drawing i.i.d. samples from Unif[0, 1] across the next states, followed by normalization. Both the initial policy $\pi _ { 0 }$ and the evaluation distribution $\mu$ are uniform. The behavior policy is generated independently as $\pi _ { b } ( a \mid s ) \propto u _ { s , a ; }$ where $u _ { s , a } \sim \mathrm { U n i f } [ 0 . 5 , 1 . 5 ]$ . We consider two standard regularizers:

$$
h _ { \mathrm { N P G } } ( p ) = \sum _ { a \in \mathcal { A } } p ( a ) \log p ( a ) , \qquad h _ { \mathrm { P Q A } } ( p ) = \frac { 1 } { 2 } \| p \| _ { 2 } ^ { 2 } ,\tag{40}
$$

For each regularizer $h ,$ the optimal policy $\pi _ { \tau , h } ^ { * }$ and value function $V _ { \tau , h } ^ { * }$ are computed via regularized value iteration with the termination condition $\| V ^ { k + 1 } - V ^ { k } \| _ { \infty } \leq 1 0 ^ { - 1 3 }$ . We then measure the value gap and the mean squared $\ell _ { 1 }$ policy error of exact coordinate-wise TD–PMD relative to the corresponding optimum:

$$
\begin{array} { r l r l } & { \mathcal { E } _ { V } ( \pi ) : = V _ { \tau , h } ^ { * } ( \mu ) - V _ { \tau , h } ^ { \pi } ( \mu ) , } & & { \mathcal { E } _ { \pi } ( \pi ) : = \mathbb { E } _ { s \sim \mu } \left[ \| \pi ( \cdot \mid s ) - \pi _ { \tau , h } ^ { * } ( \cdot \mid s ) \| _ { 1 } ^ { 2 } \right] . } \end{array}\tag{41}
$$

## 5.1 Exact coordinate-wise update experiments

The exact experiment illustrates convergence from two critic initializations under the coordinate-wise updates. We use $\gamma = 0 . 9 5 , \tau = 0 . 1$ , and $\eta = 0 . 5$ , and run each experiment for 1000 iterations with fixed weights $W = \alpha \Sigma _ { b }$ and $\alpha = 2 5 0$ . For the generated environment and behavior policy, $\widetilde { \sigma } _ { b } = 9 . 5 6 5 \times 1 0 ^ { - 4 }$ and $\overline { { \sigma } } _ { b } = 3 . 4 3 6 \times 1 0 ^ { - 3 }$ , giving $w ( s , a ) = \alpha \sigma ^ { \pi _ { b } } ( s , a ) \in [ 0 . 2 3 9 , 0 . 8 5 9 ]$

The solid curves use $Q _ { \tau } ^ { 0 } = 0$ , which satisfies the one-sided Bellman condition $\mathcal { F } _ { \tau } ^ { \pi _ { 0 } } Q _ { \tau } ^ { 0 } - Q _ { \tau } ^ { 0 } \geq 0$ in this instance. For the dashed curves, a bounded random critic is shifted by the same constant across all state–action pairs so that $\mathcal { F } _ { \tau } ^ { \pi _ { 0 } } Q _ { \tau } ^ { 0 } - Q _ { \tau } ^ { 0 } \leq - { \bf 1 }$ . The left panel reports $\mathcal { E } _ { V } ( \pi _ { k } )$ , and the right panel reports $\mathcal { E } _ { \pi } ( \pi _ { k } )$

![](images/ae3ece6223903a7497658f7877346745a2f8a518f5fbc9f053da6aef7ce5ce52.jpg)

![](images/1fd866612ebf2084e47176bfbbb28057d3d73713ecc0932bfc00d4198be4097e.jpg)  
Figure 1: Exact coordinate-wise TD–PMD with diferent critic initializations. Solid curves correspond to initialization that satisfies the one-sided Bellman condition $\mathcal { F } _ { \tau } ^ { \pi _ { 0 } } Q _ { \tau } ^ { 0 } - Q _ { \tau } ^ { 0 } \geq 0$ while dashed curves correspond to general initialization.

It can be observed that both initializations lead to approximately geometric decay of the value gap and policy error after an initial transient. The initialization that violates the one-sided Bellman condition leads to an initial increase in error before the subsequent geometric decay. These observations are consistent with Theorem 3.4 and Corollary 3.6, which control the value gap and policy error, respectively, for arbitrary finite critic initializations. It is also worth noting that the one-sided Bellman condition does not guarantee a monotonic decrease in the policy error, as illustrated in Figure 1 (right).

## 5.2 Finite-batch stochastic experiment

The stochastic experiment uses the same problem setup as the exact experiment. We use sampled transitions with $Q _ { \tau } ^ { 0 } = 0$ and set $\gamma = 0 . 5 , \tau = 0 . 7 , \eta = 4 \times 1 0 ^ { - 7 } , \alpha = 1$ , and $B = 1 0$ . The batch weights are given by (34) with $\vartheta = 0 . 1$ . For $x \in \{ V , \pi \}$ , each run reports the conditional average

$$
\widehat { \xi } _ { x } ( K ) = \frac { \sum _ { k = 0 } ^ { K - 1 } \rho ^ { - k } \mathcal { E } _ { x } ( \pi _ { k } ) } { \sum _ { k = 0 } ^ { K - 1 } \rho ^ { - k } } , \qquad \rho : = ( 1 + \eta \tau ) ^ { - 1 } .\tag{42}
$$

For each regularizer, we perform five independent runs for $K = 5 \times 1 0 ^ { 7 }$ iterations $( 5 \times 1 0 ^ { 8 }$ transitions per run). The error are evaluated and accumulated at every iteration and the resulting weighted metrics are recorded every $1 0 ^ { 5 }$ iterations. Figure 2 reports the five-run mean together with the range of the five trajectories.

![](images/3007f258f95d76b02a1334debe66f7187f652b7c01270014942ac747103882c8.jpg)

![](images/7d0e70079aa32d4b71182a279c390cf8a22a729685e75c66ec13e398ff5c058f.jpg)  
Figure 2: Finite-batch stochastic TD–PMD over five independent Markov trajectories.

The value metric shows an initial approximately exponential decrease, followed by slower improvement, which is qualitatively consistent with the three-term bound (35) of Theorem 4.5. The policy metric exhibits a similar empirical pattern. The five independent runs remain close to each other throughout the experiment.

## 6 Conclusion and Future Directions

The present paper develops a convergence theory for regularized TD–PMD with a persistent critic updated through one-step TD recursions. In the exact setting, we establish global linear convergence for coordinate-wise Bellman updates under general convex mirror geometry. We then extend the analysis to a single of-policy Markov trajectory and show that, under strong convexity, finite-batch TD–PMD attains an expected value gap of at most ϵ with $\widetilde { \mathcal { O } } ( \epsilon ^ { - 1 } )$ observed transitions. Numerical experiments illustrate the predicted behavior in both the exact and Markov-sampling settings.

Several questions remain open. First, it is interesting to analyze an online variant that performs one critic update and one policy update for every observed transition. Such a result would require controlling the interaction between Markov dependence, critic noise, and policy drift without relying on within-batch mixing. Second, one may consider a stochastic variant in which the conditional expectation over actions in the TD target is replaced by a single action sampled from the current target policy. In this case, the resulting additional sampling noise needs to be controlled jointly with the Markov dependence and the drift of the target policy. It would also be interesting to develop stochastic TD–PMD schemes with more flexible output rules and stepsize schedules, as well as parameter choices that require less prior knowledge of the behavior-chain mixing and coverage properties.

## References

Alekh Agarwal, Sham M. Kakade, Jason D. Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22 (98):1–76, 2021.

Carlo Alfano and Patrick Rebeschini. Linear convergence for natural policy gradient with log-linear policy parametrization. arXiv:2209.15382, 2022.

Carlo Alfano, Rui Yuan, and Patrick Rebeschini. A novel framework for policy mirror descent with general parameterization and linear convergence. In Advances in Neural Information Processing Systems, 2023.

Shicong Cen, Chen Cheng, Yuxin Chen, Yuting Wei, and Yuejie Chi. Fast global convergence of natural policy gradient methods with entropy regularization. Operations Research, 70(4):2563–2578, 2022.

Shicong Cen, Yuejie Chi, Simon S Du, and Lin Xiao. Faster last-iterate convergence of policy optimization in zero-sum markov games. In International Conference on Learning Representations, 2023.

Veronica Chelu and Doina Precup. Functional acceleration for policy mirror descent. arXiv preprint arXiv:2407.16602, 2024.

Yinlam Chow, Ofir Nachum, and Mohammad Ghavamzadeh. Path consistency learning in Tsallis entropy regularized MDPs. In Proceedings of the 35th International Conference on Machine Learning, volume 80, pages 979–988, 2018.

Jie Feng, Ke Wei, and Jinchi Chen. Global convergence of natural policy gradient with hessian-aided momentum variance reduction. Journal of Scientific Computing, 101(2):44, 2024.

Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In Proceedings of the 36th International Conference on Machine Learning, pages 2160–2169, 2019.

Hisham Husain, Kamil Ciosek, and Ryota Tomioka. Regularized policies are reward robust. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, volume 130, pages 64–72, 2021.

Zhichao Jia and Guanghui Lan. Value mirror descent for reinforcement learning. arXiv preprint arXiv:2604.06039, 2026.

Emmeran Johnson, Ciara Pike-Burke, and Patrick Rebeschini. Optimal convergence rate for exact policy mirror descent in discounted markov decision processes. In Advances in Neural Information Processing Systems, 2023.

Caleb Ju and Guanghui Lan. Strongly-polynomial time and validation analysis of policy gradient methods. Mathematical Programming, pages 1–45, 2026.

Sham Kakade. A natural policy gradient. In Advances in Neural Information Processing Systems, pages 1531–1538, 2001.

Sham M. Kakade and John Langford. Approximately optimal approximate reinforcement learning. In International Conference on Machine Learning, pages 267–274, 2002.

Sajad Khodadadian, Prakirt Raj Jhunjhunwala, Sushil Mahavir Varma, and Siva Theja Maguluri. On the linear convergence of natural policy gradient algorithm. In IEEE Conference on Decision and Control, pages 3794–3799, 2021.

Guanghui Lan. Policy mirror descent for reinforcement learning: Linear convergence, new sampling complexity, and generalized problem classes. Mathematical Programming, 198(1):1059–1106, 2023.

Kyungjae Lee, Sungjoon Choi, and Songhwai Oh. Sparse Markov decision processes with causal sparse Tsallis entropy regularization for reinforcement learning. IEEE Robotics and Automation Letters, 3(3):1466–1473 2018.

Kyungjae Lee, Sungyub Kim, Sungbin Lim, Sungjoon Choi, and Songhwai Oh. Tsallis reinforcement learning: A unified framework for maximum entropy reinforcement learning. arXiv preprint arXiv:1902.00137, 2019. URL https://arxiv.org/abs/1902.00137.

David A Levin and Yuval Peres. Markov chains and mixing times, volume 107. American Mathematical Soc., 2017.

Wenye Li and Ke Wei. On the policy convergence of policy mirror descent methods. arXiv preprint arXiv:2607.11626, 2026.

Wenye Li, Hongxu Chen, Jiacai Liu, and Ke Wei. Policy mirror descent with temporal diference learning: sample complexity under online Markov data. arxiv preprint arXiv:2512.24056, 2025.

Yan Li and Guanghui Lan. Policy mirror descent inherently explores action space. SIAM Journal on Optimization, 35(1):116–156, 2025.

Yan Li, Guanghui Lan, and Tuo Zhao. Homotopic policy mirror descent: Policy convergence, algorithmic regularization, and improved sample complexity. Mathematical Programming, 207:457–513, 2024.

Dachao Lin and Zhihua Zhang. On the convergence of policy in unregularized policy mirror descent. arxiv preprint arXiv:2205.08176, 2022.

Jiacai Liu, Wenye Li, and Ke Wei. Elementary analysis of policy gradient methods. arxiv:2404.03372, 2024.

Jiacai Liu, Wenye Li, Dachao Lin, Ke Wei, and Zhihua Zhang. On the convergence of projected policy gradient for any constant step sizes. Journal of Machine Learning Research, 26:1–35, 2025a.

Jiacai Liu, Wenye Li, and Ke Wei. On the convergence of policy mirror descent with temporal diference evaluation. arXiv preprint arXiv:2509.18822, 2025b.

Kimon Protopapas and Anas Barakat. Policy mirror descent with lookahead. Advances in Neural Information Processing Systems, 37:26443–26481, 2024.

Martin L Puterman. Markov decision processes: discrete stochastic dynamic programming. Wiley Series in Probability and Statistics, 1994.

Richard S. Sutton. Learning to predict by the methods of temporal diferences. Machine Learning, 3:9–44, 1988.

Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, Cambridge, 2018.

Lin Xiao. On the convergence rates of policy gradient methods. Journal of Machine Learning Research, 23 (282):1–36, 2022.

Donghao Ying, Yuhao Ding, and Javad Lavaei. A dual approach to constrained markov decision processes with entropy regularization. In Proceedings of the 25th International Conference on Artificial Intelligence and Statistics, volume 151, pages 1887–1909, 2022.

Wenhao Zhan, Shicong Cen, Baihe Huang, Yuxin Chen, Jason D. Lee, and Yuejie Chi. Policy mirror descent for regularized reinforcement learning: A generalized framework with linear convergence. SIAM Journal on Optimization, 33(2):1061–1091, 2023.

## A Proofs for Section 3

## A.1 Proof of Lemma 3.1

By setting $p = \pi _ { k } ( \cdot \mid s )$ , Lemma 2.4 implies the following actor-improvement inequality

$$
\begin{array} { r } { \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } \geq \mathcal { F } _ { \tau } ^ { \pi _ { k } } Q _ { \tau } ^ { k } . } \end{array}\tag{43}
$$

Let $R _ { k } : = \mathcal { F } _ { \tau } ^ { \pi _ { k } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k }$ and $P _ { k + 1 } : = P _ { s \times A } ^ { \pi _ { k + 1 } }$ , and set $\widehat { R } _ { k } : = \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k }$ . By (43), there holds $\widehat { R } _ { k } \geq R _ { k }$ Additionally, the matrix $I - W + \gamma P _ { k + 1 } W$ is nonnegative since $0 < W \le I$ . Leveraging the afine identity $\mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q ^ { \prime } - \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q = \gamma P _ { k + 1 } ( Q ^ { \prime } - Q )$ and the critic update $Q _ { \tau } ^ { k + 1 } - Q _ { \tau } ^ { k } = W \widehat { R } _ { k }$ , we obtain that

$$
\begin{array} { r l } & { R _ { k + 1 } = \widehat { R } _ { k } - ( Q _ { \tau } ^ { k + 1 } - Q _ { \tau } ^ { k } ) + \gamma P _ { k + 1 } ( Q _ { \tau } ^ { k + 1 } - Q _ { \tau } ^ { k } ) } \\ & { \qquad = \widehat { R } _ { k } - W \widehat { R } _ { k } + \gamma P _ { k + 1 } W \widehat { R } _ { k } } \\ & { \qquad = ( I - W + \gamma P _ { k + 1 } W ) \widehat { R } _ { k } } \\ & { \qquad \geq ( I - W + \gamma P _ { k + 1 } W ) R _ { k } . } \end{array}\tag{44}
$$

Taking positive parts after multiplying by −1,

$$
[ - R _ { k + 1 } ] _ { + } \leq ( I - W + \gamma P _ { k + 1 } W ) [ - R _ { k } ] _ { + } .\tag{45}
$$

Since W is diagonal, we have $W ( I - W + \gamma P _ { k + 1 } W ) = ( I - W + \gamma W P _ { k + 1 } ) W$ , and the row sum of $I { - } W { + } \gamma W P _ { k + 1 }$ corresponding to coordinate (s, a) is $1 - w ( s , a ) ( 1 - \gamma ) \leq 1 - \underline { w } ( 1 - \gamma )$ . Therefore

$$
\begin{array} { r } { \| W [ - R _ { k + 1 } ] _ { + } \| _ { \infty } \leq [ 1 - \underline { { w } } ( 1 - \gamma ) ] \| W [ - R _ { k } ] _ { + } \| _ { \infty } , } \end{array}
$$

which proves $\delta _ { k + 1 } \leq [ 1 - \underline { w } ( 1 - \gamma ) ] \delta _ { k }$

We next prove the critic comparisons in (5). By the definition of $\delta _ { k } .$ , for every $( s , a )$

$$
w ( s , a ) [ - R _ { k } ( s , a ) ] _ { + } \leq \| W [ - R _ { k } ] _ { + } \| _ { \infty } = \underline { { w } } ( 1 - \gamma ) \delta _ { k } ,
$$

and hence

$$
\lbrack - R _ { k } ( s , a ) \rbrack _ { + } \leq \frac { w } { w ( s , a ) } ( 1 - \gamma ) \delta _ { k } \leq ( 1 - \gamma ) \delta _ { k } .
$$

Thus $R \kappa \geq - ( 1 - \gamma ) \delta _ { k } \mathbf { 1 }$ . By the definition of the Bellman operator,

$$
\mathcal { F } _ { \tau } ^ { \pi _ { k } } \big ( Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \big ) - \big ( Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \big ) = R _ { k } + ( 1 - \gamma ) \delta _ { k } \mathbf { 1 } \geq 0 .
$$

Iterating the monotone γ-contraction operator $\mathcal { F } _ { \tau } ^ { \pi _ { k } }$ to its fixed point proves $Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \leq Q _ { \tau } ^ { \pi _ { k } }$ , while $Q _ { \tau } ^ { \pi k } \leq Q _ { \tau } ^ { * }$ follows from the optimality of $Q _ { \tau } ^ { * }$ . Finally, (43) implies

$$
\mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } \big ( Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \big ) - \big ( Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \big ) = \widehat { R } _ { k } + ( 1 - \gamma ) \delta _ { k } \mathbf { 1 } \geq R _ { k } + ( 1 - \gamma ) \delta _ { k } \mathbf { 1 } \geq 0 .
$$

Iterating $\mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } }$ to its fixed point gives the inequality $Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \leq Q _ { \tau } ^ { \pi _ { k + 1 } }$ <sup>1</sup>, which completes the proof.

## A.2 Proof of Lemma 3.2

Since $\gamma / \xi \in ( 0 , 1 )$ , the distribution $\nu _ { \mu , \xi } ^ { * }$ is a convex combination of probability distributions. Shifting its series by one index gives

$$
\frac { \gamma } { \xi } ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } P _ { s } ^ { \pi _ { \tau } ^ { * } } = ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } - \left( 1 - \frac { \gamma } { \xi } \right) \mu ^ { \top } .
$$

Multiplying by $\xi$ proves the first identity in (6).

Let $m : = \mathrm { m i n } _ { s : \nu _ { \mu , \xi } ^ { * } ( s ) > 0 } \mu ( s ) / \nu _ { \mu , \xi } ^ { * } ( s ) \geq 0$ . Noting that the support of $\mu$ is contained in that of $\nu _ { \mu , \xi } ^ { * }$ , there holds

$$
1 = \sum _ { s : \nu _ { \mu , \xi } ^ { * } ( s ) > 0 } \nu _ { \mu , \xi } ^ { * } ( s ) \frac { \mu ( s ) } { \nu _ { \mu , \xi } ^ { * } ( s ) } \geq m \sum _ { s : \nu _ { \mu , \xi } ^ { * } ( s ) > 0 } \nu _ { \mu , \xi } ^ { * } ( s ) = m ,
$$

so $m \leq 1$ . Since $\gamma _ { \mu , \xi } = \xi - ( \xi - \gamma ) m$ , we have $( \xi - \gamma ) \mu ( s ) \geq ( \xi - \gamma ) m \nu _ { \mu , \xi } ^ { * } ( s ) = ( \xi - \gamma _ { \mu , \xi } ) \nu _ { \mu , \xi } ^ { * } ( s )$ for every state. This also proves $\gamma _ { \mu , \xi } \in [ \gamma , \xi ]$

## A.3 Proof of Lemma 3.3

For brevity, define the first term of (7) by

$$
\mathcal { B } _ { k } : = \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \big ( Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) + \delta _ { k } \big ) .
$$

Lemma 3.1 implies $B _ { k } \geq 0$ . We first derive a critic inequality for an arbitrary fixed $\beta \in [ \gamma _ { \mu , \xi } , 1 )$ and specify its value at the end of the proof.

Step 1: Policy estimate. Define the value estimate $V _ { \tau } ^ { k + 1 } ( s ) \ : = \ \langle \pi _ { k + 1 } ( \cdot \ | \ s ) , Q _ { \tau } ^ { k } ( s , \cdot ) \rangle - \tau h ^ { \pi _ { k + 1 } } ( s ) $ Expanding the two values statewise, applying Lemma 2.4 with $p = \pi _ { \tau } ^ { * } ( \cdot \mid s )$ , and immediately dropping the nonnegative term $D _ { \pi _ { k } } ^ { \pi _ { k + 1 } } \left( s \right)$ gives

$$
\begin{array} { r } { V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { * } ) + ( \eta ^ { - 1 } + \tau ) D _ { \pi _ { k + 1 } ^ { * } } ^ { \pi ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) \leq \mathbb { E } _ { \mathrm { ~ } s \sim \nu _ { \mu , \xi } ^ { * } } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] + \eta ^ { - 1 } D _ { \pi _ { k } ^ { * } } ^ { \pi ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) . } \end{array}\tag{46}
$$

Step 2: Distribution comparison. By Lemma 3.1 we have $Q _ { \tau } ^ { k } - \delta _ { k } \mathbf { 1 } \leq Q _ { \tau } ^ { \pi _ { k + 1 } }$ . Hence, by the definition of $\bar { V } _ { \tau } ^ { k + 1 }$ ,

$$
\begin{array} { r l } { \forall s \in S : } & { V _ { \tau } ^ { k + 1 } ( s ) - \delta _ { k } = \left. \pi _ { k + 1 } ( \cdot \ \vert \ s ) , Q _ { \tau } ^ { k } ( s , \cdot ) - \delta _ { k } \mathbf { 1 } \right. - \tau h ^ { \pi _ { k + 1 } } ( s ) } \\ & { \qquad \leq \left. \pi _ { k + 1 } ( \cdot \ \vert \ s ) , Q _ { \tau } ^ { \pi _ { k + 1 } } ( s , \cdot ) \right. - \tau h ^ { \pi _ { k + 1 } } ( s ) = V _ { \tau } ^ { \pi _ { k + 1 } } ( s ) \leq V _ { \tau } ^ { * } ( s ) . } \end{array}
$$

Thus $V _ { \tau } ^ { * } - V _ { \tau } ^ { k + 1 } + \delta _ { k } \mathbf { 1 } \geq 0$ . Together with the second componentwise relation in (6) and the fact that $( \xi - \gamma ) \dot { \mu } ( s ) \geq ( \xi - \gamma _ { \mu , \xi } ) \nu _ { \mu , \xi } ^ { * } ( s ) \geq ( \xi - \beta ) \nu _ { \mu , \xi } ^ { * } ( s )$ , this gives

$$
( \xi - \gamma ) \left[ V _ { \tau } ^ { \ast } ( \mu ) - V _ { \tau } ^ { k + 1 } ( \mu ) \right] \geq ( \xi - \beta ) \left[ V _ { \tau } ^ { \ast } ( \nu _ { \mu , \xi } ^ { \ast } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { \ast } ) \right] - ( \beta - \gamma ) \delta _ { k } .\tag{47}
$$

By the Bellman equation,

$$
\forall ( s , a ) \in \mathcal { S } \times \mathcal { A } : \quad Q _ { \tau } ^ { * } ( s , a ) - \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } ( s , a ) = \gamma \sum _ { s ^ { \prime } } P ( s ^ { \prime } \mid s , a ) \big [ V _ { \tau } ^ { * } ( s ^ { \prime } ) - V _ { \tau } ^ { k + 1 } ( s ^ { \prime } ) \big ] .
$$

Taking expectation with respect to $\nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s )$ and then applying the first identity in (6), we obtain

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { \Phi } _ { a \sim v _ { \pi } ^ { * } ( s ) } } \left[ \mathcal { F } _ { \tau } ^ { \pi _ { k } + 1 } Q _ { \tau } ^ { k } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] = \mathbb { E } _ { \mathbf { \Phi } _ { s \sim v _ { \pi } ^ { * } ( s ) } } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] - \gamma ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } P _ { s } ^ { \pi _ { \tau } ^ { * } } ( V _ { \tau } ^ { * } - V _ { \tau } ^ { k + 1 } ) } \\ & { \mathrm { ~ a s . e . } _ { \tau + \tau } ^ { - 1 } | \mathcal { S } | } \\ & { \mathrm { ~ a s . e . r . } _ { \tau ( s ) } ^ { * } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] - \xi \big [ V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { * } ) \big ] + ( \xi - \gamma ) \left[ V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { k + 1 } ( \mu ) \right] } \\ &  \mathrm { ~ a s . e . r . } _ { \tau ( s ) } ^ { * } \left[ \mathcal { S } _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] - \xi \big [ V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { * } ) \big ] + ( \xi - \beta ) \left[ V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { * } ) \right] - ( \beta - \gamma ) \delta _ { k } .  \end{array}
$$

The last inequality follows from (47). Substituting (46) yields

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathbf { \Phi } _ { s \sim \pi _ { \tau } ^ { * } \left( 1 \right) s } } \left[ \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] \geq ( 1 - \beta ) \mathbb { E } _ { \mathbf { \Phi } _ { s \sim \pi _ { \tau } ^ { * } \left( 1 \right) s } } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] } \\ & { \qquad \quad \quad \quad \quad \quad \quad \alpha \sim \pi _ { \tau } ^ { * } ( \cdot | s ) } \\ & { \qquad \quad \quad \quad - \beta \eta ^ { - 1 } D _ { \pi _ { k } } ^ { \pi _ { k } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) + \beta ( \eta ^ { - 1 } + \tau ) D _ { \pi _ { k + 1 } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) - ( \beta - \gamma ) \delta _ { k } . } \end{array}\tag{48}
$$

Step 3: Critic update. The inverse weight in $\boldsymbol { B } _ { k }$ cancels the corresponding coordinate weight $w ( s , a )$ in the critic update. Thus

$$
\mathcal { B } _ { k + 1 } = \mathcal { B } _ { k } - \mathbb { E } _ { \underset { a \sim \pi _ { \tau } ^ { * } ( \cdot | s ) } { s \sim \nu _ { \pi _ { \tau } ^ { * } ( \cdot | s ) } ^ { * } } } \left[ \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] + ( \delta _ { k + 1 } - \delta _ { k } ) \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } .\tag{49}
$$

Since every shifted critic coordinate is nonnegative and $w ( s , a ) \geq \underline { w }$

$$
\begin{array} { r l } & { 0 \leq \underline { w } \mathcal { B } _ { k } \leq \mathbb { E } _ { \mathbf { \phi } _ { s \sim \nu _ { \mu , \xi } ^ { * } } } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) \right] + \delta _ { k } . } \end{array}\tag{50}
$$

Substituting (48) into (49), using $\delta _ { k + 1 } - \delta _ { k } \leq - \underline { w } ( 1 - \gamma ) \delta _ { k }$ , and then applying (50), we obtain

$$
\begin{array} { r l } & { \mathcal { B } _ { k + 1 } \leq [ 1 - \underline { { w } } ( 1 - \beta ) ] \mathcal { B } _ { k } + \beta \eta ^ { - 1 } D _ { \pi _ { k } } ^ { \pi _ { r } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) - \beta ( \eta ^ { - 1 } + \tau ) D _ { \pi _ { k + 1 } } ^ { \pi _ { r } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) } \\ & { \qquad + ( 1 - \gamma ) \left( 1 - \underline { { w } } \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \right) \delta _ { k } . } \end{array}\tag{51}
$$

Now choosing

$$
\beta = \operatorname* { m a x } \{ \gamma _ { \mu , \xi } , ( 1 + \eta \tau ) ^ { - 1 } \} \in [ \gamma _ { \mu , \xi } , 1 ) ,
$$

we obtain $\beta ( \eta ^ { - 1 } + \tau ) = \operatorname* { m a x } \{ \eta ^ { - 1 } , \gamma _ { \mu , \xi } ( \eta ^ { - 1 } + \tau ) \}$ and $1 - \underline { w } ( 1 - \beta ) = \rho .$ . Thus the potential in (7) is $\mathcal { L } _ { k } = \mathcal { B } _ { k } + \beta ( \eta ^ { - 1 } + \tau ) D _ { \pi _ { k } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } )$ . Adding the Bregman component of $\mathcal { L } _ { k + 1 }$ to both sides of (51), the definition of $\mathcal { L } _ { k }$ and the fact that $( 1 + \dot { \eta } \tau ) ^ { - 1 } \leq \rho$ gives that

$$
\mathcal { L } _ { k + 1 } \leq \rho \mathcal { L } _ { k } + ( 1 - \gamma ) \left( 1 - \underline { { w } } \sum _ { s , a } \frac { \nu _ { \mu , \xi } ^ { * } ( s ) \pi _ { \tau } ^ { * } ( a \mid s ) } { w ( s , a ) } \right) \delta _ { k } ,
$$

which completes the proof.

## A.4 Proof of Theorem 3.4

By Lemma 3.1 there holds $\delta _ { k } \leq [ 1 - \underline { w } ( 1 - \gamma ) ] ^ { k } \delta _ { 0 }$ . Iterating (9) and using the definition of $\mathcal { C } _ { k }$ yields

$$
\mathcal { L } _ { k } \leq \mathcal { C } _ { k } .\tag{52}
$$

The first term of the resolvent series shows that $\nu _ { \mu , \xi } ^ { * }$ assigns positive mass wherever $\mu ( s ) > 0$ . Hence, for every nonnegative $f : S  \mathbb { R }$ , there holds

$$
\mu ^ { \top } f \leq \left. \frac { \mu } { \nu _ { \mu , \xi } ^ { * } } \right. _ { \infty } ( \nu _ { \mu , \xi } ^ { * } ) ^ { \top } f .
$$

Write the first term of $\mathcal { L } _ { k }$ as $B _ { k }$ . The policy estimate (46) and the comparison $V _ { \tau } ^ { \pi k + 1 } \geq V _ { \tau } ^ { k + 1 } - \delta _ { k } \mathbf { 1 }$ imply that

$$
\begin{array} { r l } & { V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { \pi _ { k + 1 } } ( \nu _ { \mu , \xi } ^ { * } ) \leq V _ { \tau } ^ { * } ( \nu _ { \mu , \xi } ^ { * } ) - V _ { \tau } ^ { k + 1 } ( \nu _ { \mu , \xi } ^ { * } ) + \delta _ { k } } \\ & { \qquad \overset { ( a ) } { \leq } \mathbb { E } _ { \mathbf { \mu } _ { s \sim \nu _ { \mu , \xi } ^ { * } } } \left[ Q _ { \tau } ^ { * } ( s , a ) - Q _ { \tau } ^ { k } ( s , a ) + \delta _ { k } \right] + \eta ^ { - 1 } D _ { \pi _ { k } ^ { * } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) } \\ & { \qquad \overset { ( b ) } { \leq } \overline { { w } } \mathcal { B } _ { k } + \eta ^ { - 1 } D _ { \pi _ { k } ^ { * } } ^ { \pi _ { \tau } ^ { * } } ( \nu _ { \mu , \xi } ^ { * } ) } \\ & { \qquad \overset { ( c ) } { \leq } \mathcal { L } _ { k } . } \end{array}\tag{53}
$$

Here (a) follows from (46) after dropping the nonnegative Bregman term, (b) uses $Q _ { \tau } ^ { * } - Q _ { \tau } ^ { k } + \delta _ { k } \mathbf { 1 } \geq 0$ and $w ( s , a ) \leq \overline { { w } }$ , and (c) follows directly from $\overline { { w } } \leq 1$ and the definition of $\mathcal { L } _ { k }$ . Since the value gap is pointwise nonnegative, we apply the preceding inequality for nonnegative $f$ with $f = V _ { \tau } ^ { * } - V _ { \tau } ^ { \pi _ { k + 1 } }$ . Combining the resulting bound with (53) and (52) proves (12). Moreover, stationarity gives $\nu _ { \nu ^ { * } , \xi } ^ { * } = \nu ^ { * }$ for every $\xi \in ( \gamma , 1 )$ which implies $\gamma _ { \nu ^ { * } , \xi } = \gamma$ and $\| \nu ^ { * } / \nu _ { \nu ^ { * } , \xi } ^ { * } \| _ { \infty } = 1$ . Substituting these identities into the definition of $\mathcal { C } _ { k }$ gives the stated value bound and contraction factor.

## A.5 Proof of Corollary 3.7

Let

$$
B _ { k } : = \sum _ { s , a } \nu ^ { * } ( s ) \pi ^ { * } ( a \mid s ) \bigl ( Q ^ { * } ( s , a ) - Q ^ { k } ( s , a ) + \delta _ { k } \bigr ) \geq 0 .
$$

Taking $\mu = \nu ^ { * } , \beta = \gamma , \tau = 0$ , and $W = I$ in the first inequality of (51), and replacing η by $\eta _ { k } > 0 .$ , yields

$$
B _ { k + 1 } + \frac { \gamma } { \eta _ { k } } D _ { \pi _ { k + 1 } } ^ { \pi ^ { * } } ( \nu ^ { * } ) \leq \gamma \left[ B _ { k } + \frac { 1 } { \eta _ { k } } D _ { \pi _ { k } } ^ { \pi ^ { * } } ( \nu ^ { * } ) \right] .\tag{54}
$$

The same three-point argument, together with $V ^ { \pi _ { k + 1 } } \geq V ^ { k + 1 } - \delta _ { k } \mathbf { 1 }$ , yields

$$
\begin{array} { r } { V ^ { * } ( \nu ^ { * } ) - V ^ { \pi _ { k + 1 } } ( \nu ^ { * } ) + \eta _ { k } ^ { - 1 } D _ { \pi _ { k + 1 } } ^ { \pi ^ { * } } ( \nu ^ { * } ) \leq B _ { k } + \eta _ { k } ^ { - 1 } D _ { \pi _ { k } } ^ { \pi ^ { * } } ( \nu ^ { * } ) . } \end{array}\tag{55}
$$

When $\eta _ { k } = \eta$ , summing (54) from $k = 0$ to $K - 1$ provides the bound for $\textstyle \sum _ { k = 0 } ^ { K - 1 } B _ { k }$ . Furthermore, summing (55) and then telescoping the Bregman terms gives (20).

$$
\begin{array} { r } { \Phi _ { k } : = B _ { k } + \eta _ { k } ^ { - 1 } D _ { \pi _ { k } } ^ { \pi ^ { * } } ( \nu ^ { * } ) } \end{array}
$$

$$
\eta _ { k + 1 } ^ { - 1 } \leq \gamma \eta _ { k } ^ { - 1 }
$$

$$
\Phi _ { k + 1 } \leq \gamma \Phi _ { k }
$$

$$
V ^ { * } ( \nu ^ { * } ) - V ^ { \pi _ { k + 1 } } ( \nu ^ { * } ) \le \Phi _ { k }
$$

$$
\Phi _ { k } \le \gamma ^ { k } \Phi _ { 0 }
$$

## B Proofs for Section 4

## B.1 Proof of Lemma 4.2

For each sample, define its TD target by

$$
Y _ { t } ^ { k } : = r _ { t } ^ { k } + \gamma \mathbb { E } _ { a ^ { \prime } \sim \pi _ { k + 1 } ( \cdot | s _ { t + 1 } ^ { k } ) } \left[ Q _ { \tau } ^ { k } ( s _ { t + 1 } ^ { k } , a ^ { \prime } ) \right] - \tau \gamma h ^ { \pi _ { k + 1 } } ( s _ { t + 1 } ^ { k } ) .
$$

For each $( s , a )$ , let

$$
\widehat { \sigma } _ { k } ( s , a ) : = \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \mathbb { 1 } [ ( s _ { t } ^ { k } , a _ { t } ^ { k } ) = ( s , a ) ] \in [ 0 , 1 ] .
$$

The normalized weighted target is then defined as

$$
\widehat { y } _ { k } ( s , a ) : = \left\{ \begin{array} { l l } { \displaystyle \frac { 1 } { \widehat { \sigma } _ { k } ( s , a ) } \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \mathbb { 1 } [ ( s _ { t } ^ { k } , a _ { t } ^ { k } ) = ( s , a ) ] Y _ { t } ^ { k } , } & { \widehat { \sigma } _ { k } ( s , a ) > 0 , } \\ { 0 , } & { \widehat { \sigma } _ { k } ( s , a ) = 0 . } \end{array} \right.
$$

With this notation, the coordinate critic update becomes

$$
\begin{array} { r } { Q _ { \tau } ^ { k + 1 } ( s , a ) = \bigl ( 1 - \alpha _ { k } \widehat { \sigma } _ { k } ( s , a ) \bigr ) Q _ { \tau } ^ { k } ( s , a ) + \alpha _ { k } \widehat { \sigma } _ { k } ( s , a ) \widehat { y } _ { k } ( s , a ) . } \end{array}
$$

If $\| Q _ { \tau } ^ { k } \| _ { \infty } \leq ( 1 + \tau \gamma H _ { h } ) / ( 1 - \gamma )$ , Assumptions 2.1 and 2.2 give

$$
| \widehat { y } _ { k } ( s , a ) | \leq 1 + \gamma \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau \gamma H _ { h } = \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } .
$$

Thus $Q _ { \tau } ^ { k + 1 } ( s , a )$ is a convex combination of two points in $[ - ( 1 + \tau \gamma H _ { h } ) / ( 1 - \gamma ) , ( 1 + \tau \gamma H _ { h } ) / ( 1 - \gamma ) ]$ . Induction from $Q _ { \tau } ^ { 0 } = 0$ proves the claim.

## B.2 Proof of Lemma 4.3

Step 1: Bounded vertex divergences. Fix $k \geq 0 , s \in \mathcal { S }$ , and $a \in { \mathcal { A } }$ . By Lemma 2.4 with $\boldsymbol { p } = \boldsymbol { e } _ { a }$

$$
\begin{array} { r l } & { \langle \pi _ { k + 1 } ( \cdot \vert ~ s ) - e _ { a } , Q _ { \tau } ^ { k } ( s , \cdot ) \rangle - \tau \left[ h ^ { \pi _ { k + 1 } } ( s ) - h ( e _ { a } ) \right] } \\ & { \qquad \geq \eta ^ { - 1 } [ D _ { h } ( \pi _ { k + 1 } ( \cdot \vert ~ s ) \| \pi _ { k } ( \cdot \vert ~ s ) ) + ( 1 + \eta \tau ) D _ { h } ( e _ { a } \| \pi _ { k + 1 } ( \cdot \vert ~ s ) ) - D _ { h } ( e _ { a } \| \pi _ { k } ( \cdot \vert ~ s ) ) ] . } \end{array}
$$

Using Lemma 4.2 to bound the critic,

$$
( 1 + \eta \tau ) D _ { h } ( e _ { a } \| \pi _ { k + 1 } ( \cdot \mid s ) ) \leq D _ { h } ( e _ { a } \| \pi _ { k } ( \cdot \mid s ) ) + \frac { 2 \eta ( 1 + \tau H _ { h } ) } { 1 - \gamma } ,\tag{56}
$$

where we drop the nonnegative divergence term $D _ { h } ( \pi _ { k + 1 } ( \cdot | s ) \| \pi _ { k } ( \cdot | s ) )$ and use the bounds $\| \pi _ { k + 1 } ( \cdot \mid s ) - e _ { a } \| _ { 1 } \leq$ 2 and $| h | \leq H _ { h }$ on $\Delta ( \mathcal { A } )$ . Equivalently,

$$
D _ { h } ( e _ { a } \| \pi _ { k + 1 } ( \cdot \mid s ) ) \le \frac { D _ { h } ( e _ { a } \| \pi _ { k } ( \cdot \mid s ) ) } { 1 + \eta \tau } + \left( 1 - \frac { 1 } { 1 + \eta \tau } \right) \frac { 2 ( 1 + \tau H _ { h } ) } { \tau ( 1 - \gamma ) } .
$$

The right-hand side is a convex combination of the preceding divergence and the displayed constant. By induction we have

$$
D _ { h } ( e _ { a } \parallel \pi _ { k } ( \cdot \mid s ) ) \le \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { s ^ { \prime } \in \mathcal { S } , b \in \mathcal { A } } D _ { h } ( e _ { b } \parallel \pi _ { 0 } ( \cdot \mid s ^ { \prime } ) ) , \frac { 2 ( 1 + \tau H _ { h } ) } { \tau ( 1 - \gamma ) } \right\} .
$$

The initial divergences are finite because Assumption 2.2 ensures $e _ { a } \in$ dom $h ,$ and Assumption 2.4 ensures $\pi _ { 0 } ( \cdot \mid s ) \in \operatorname { r i n t } ( \operatorname { d o m } h )$

Step 2: Trajectory-wise Lipschitz continuity. For any $a , b \in { \mathcal { A } }$ we have

$$
\nabla _ { a } h ( \pi _ { k } ( \cdot \mid s ) ) - \nabla _ { b } h ( \pi _ { k } ( \cdot \mid s ) ) = h ( e _ { a } ) - h ( e _ { b } ) - D _ { h } ( e _ { a } \parallel \pi _ { k } ( \cdot \mid s ) ) + D _ { h } ( e _ { b } \parallel \pi _ { k } ( \cdot \mid s ) ) .
$$

The vertex divergences are nonnegative and satisfy the preceding bound as $| h ( e _ { a } ) - h ( e _ { b } ) | \leq 2 H _ { h }$ . Thus,

$$
\operatorname* { m a x } _ { \alpha \in \mathcal { A } } \nabla _ { \alpha } h ( \pi _ { k } ( \cdot \mid s ) ) - \operatorname* { m i n } _ { \alpha \in \mathcal { A } } \nabla _ { \alpha } h ( \pi _ { k } ( \cdot \mid s ) ) \leq 2 H _ { h } + \operatorname* { m a x } \left\{ \operatorname* { m a x } _ { s ^ { \prime } \in \mathcal { S } , c \in \mathcal { A } } D _ { h } ( e _ { c } \| \pi _ { 0 } ( \cdot \mid s ^ { \prime } ) ) , \frac { 2 ( 1 + \tau H _ { h } ) } { \tau ( 1 - \gamma ) } \right\} = 2 L _ { h } .\tag{57}
$$

For $p \in \{ \pi _ { k } ( \cdot \mid s ) , \pi _ { k + 1 } ( \cdot \mid s ) \}$ , let $\begin{array} { r } { c _ { p } : = \frac { 1 } { 2 } [ \operatorname* { m a x } _ { a \in \mathcal { A } } \nabla _ { a } h ( p ) + \operatorname* { m i n } _ { a \in \mathcal { A } } \nabla _ { a } h ( p ) ] } \end{array}$ . Leveraging the convexity of h and (57) yields

$$
h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) \leq \left. \nabla h ( \pi _ { k + 1 } ( \cdot \mid s ) ) - c _ { \pi _ { k + 1 } ( \cdot \mid s ) } \mathbf { 1 } , \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) \right. \leq L _ { h } \| \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) \| _ { 1 } .
$$

Interchanging the two policies gives the reverse inequality and proves (27).

Step 3: Policy increments. By setting $p = \pi _ { k } ( \cdot \mid s )$ , Lemma 2.4 implies that

$$
\begin{array} { r l } & { \big \langle \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { k } ( s , \cdot ) \big \rangle - \tau [ h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) ] } \\ & { \qquad \geq \eta ^ { - 1 } [ D _ { h } ( \pi _ { k + 1 } ( \cdot \mid s ) \| \pi _ { k } ( \cdot \mid s ) ) + ( 1 + \eta \tau ) D _ { h } ( \pi _ { k } ( \cdot \mid s ) \| \pi _ { k + 1 } ( \cdot \mid s ) ) ] . } \end{array}
$$

Using Assumption 2.3, together with Hölder’s inequality, Lemma 4.2, and (27), we obtain

$$
\lambda \| \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) \| _ { 1 } ^ { 2 } \leq \eta \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) \| \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) \| _ { 1 } ,
$$

from which (28) follows immediately.

Step 4: Action-value variation. For every state $s \in S$ , the Bellman representation gives

$$
\begin{array} { r l } & { V _ { \tau } ^ { \pi _ { k + 1 } } ( s ) - V _ { \tau } ^ { \pi _ { k } } ( s ) = \langle \pi _ { k + 1 } ( \cdot  { \mid } s ) , Q _ { \tau } ^ { \pi _ { k + 1 } } ( s , \cdot ) - Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \rangle } \\ & { \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } + \langle \pi _ { k + 1 } ( \cdot  { \mid } s ) - \pi _ { k } ( \cdot  { \mid } s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \rangle - \tau \left[ h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) \right] . } \end{array}\tag{58}
$$

By Hölder’s inequality, Lemma 2.3, and (27),

$$
\| V _ { \tau } ^ { \pi _ { k + 1 } } - V _ { \tau } ^ { \pi _ { k } } \| _ { \infty } \leq \| Q _ { \tau } ^ { \pi _ { k + 1 } } - Q _ { \tau } ^ { \pi _ { k } } \| _ { \infty } + \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) \| \pi _ { k + 1 } - \pi _ { k } \| _ { 1 , \infty } .\tag{59}
$$

Additionally, the Bellman equations give

$$
\begin{array} { r } { \| Q _ { \tau } ^ { \pi _ { k + 1 } } - Q _ { \tau } ^ { \pi _ { k } } \| _ { \infty } \leq \gamma \| V _ { \tau } ^ { \pi _ { k + 1 } } - V _ { \tau } ^ { \pi _ { k } } \| _ { \infty } . } \end{array}\tag{60}
$$

Combining the last two inequalities proves (29) by rearranging terms.

## B.3 Proof of Lemma 4.4

Fix $k , ( s , a )$ , and condition on $\mathcal { H } _ { k }$ . By the Markov property and the definition of the regularized TD sample,

$$
\begin{array} { r } { { \mathbb E } \big [ \delta _ { t } ^ { k } ( s , a ) \mid { \mathcal H } _ { k } \big ] = { \mathbb P } ( s _ { t } ^ { k } = s \mid s _ { 0 } ^ { k } ) \pi _ { b } ( a \mid s ) \left[ \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \right] ( s , a ) . } \end{array}\tag{61}
$$

The corresponding stationary mean of the sampled increment is

$$
\nu ^ { \pi _ { b } } ( s ) \pi _ { b } ( a \mid s ) \left[ { \mathcal F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \right] ( s , a ) = \left[ \Sigma _ { b } \left( { \mathcal F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \right) \right] ( s , a ) .
$$

Hence the definition of $\omega _ { t } ^ { k }$ yields the exact identity

$$
\mathbb { E } \left[ \omega _ { t } ^ { k } ( s , a ) \mathbf { \omega } \middle | \mathbf { \mathcal { H } } _ { k } \right] = \left[ \mathbb { P } ( s _ { t } ^ { k } = s \mathbf { \omega } \middle | s _ { 0 } ^ { k } ) - \nu ^ { \pi _ { b } } ( s ) \right] \pi _ { b } ( a \mathbf { \omega } | s ) \left[ \mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } - Q _ { \tau } ^ { k } \right] ( s , a ) .\tag{62}
$$

Lemma 4.2 and Assumptions 2.1–2.2 imply that both $Q _ { \tau } ^ { k } ( s , a )$ and $\mathcal { F } _ { \tau } ^ { \pi _ { k + 1 } } Q _ { \tau } ^ { k } ( s , a )$ belong to $[ - ( 1 + \tau \gamma H _ { h } ) / ( 1 -$ $\gamma ) , ( 1 + \tau \gamma H _ { h } ) / ( 1 - \gamma ) ]$ . Therefore, the mixing estimate in (24) gives

$$
\left| \mathbb { E } \left[ \omega _ { t } ^ { k } ( s , a ) \mathbf { \Delta } | \mathbf { \mathcal { H } } _ { k } \right] \right| \leq \frac { 2 ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } d _ { \mathrm { T V } } \big ( \mathbb { P } ( s _ { t } ^ { k } = \cdot \mathbf { \Delta } | \mathbf { \Phi } _ { 0 } ^ { k } ) , \nu ^ { \pi _ { b } } \big ) \leq \frac { 2 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } \kappa _ { b } ^ { t } .\tag{63}
$$

Finally, the nonnegativity of the batch weights and $\begin{array} { r } { \bar { \boldsymbol \omega } _ { k } = \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \boldsymbol \omega _ { t } ^ { k } } \end{array}$ imply

$$
\| \mathbb { E } [ \bar { \omega } _ { k } \mid \mathcal { H } _ { k } ] \| _ { \infty } \leq \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \left\| \mathbb { E } \big [ \omega _ { t } ^ { k } \mid \mathcal { H } _ { k } \big ] \right\| _ { \infty } \leq \frac { 2 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } \sum _ { t = 0 } ^ { B _ { k } - 1 } c _ { t } ^ { k } \kappa _ { b } ^ { t } ,\tag{64}
$$

which yields (33).

## B.4 Proof of Theorem 4.5

We begin by reducing the performance guarantee to a weighted sum of signed critic errors. For $s \in S$ , define

$$
B _ { k } ^ { ( k ) } ( s ) : = \left. \pi _ { \tau } ^ { \ast } ( \cdot  { | } s ) - \pi _ { k } ( \cdot  { | } s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) - Q _ { \tau } ^ { k } ( s , \cdot ) \right. .\tag{65}
$$

Lemma B.1 (Weighted value-gap reduction). For every $K \geq 1$ ，

$$
\begin{array} { r l } { \mathbb { E } \left[ V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi } \overline { { \kappa } } ( \mu ) \right] \leq \displaystyle \frac { \tau D _ { \pi _ { 0 } ^ { * } } ^ { \pi } ( d _ { \mu } ^ { * } ) } { ( 1 - \gamma ) ( \rho ^ { - K } - 1 ) } + \displaystyle \frac { 2 ( \rho ^ { - 1 } - 1 ) \rho ^ { - ( K - 1 ) } ( 1 + \tau H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } ( \rho ^ { - K } - 1 ) } + \displaystyle \frac { 2 \eta ( 1 + \tau \gamma H _ { h } ) ^ { 2 } } { \lambda ( 1 - \gamma ) ^ { 4 } } } & { } \\ { + \displaystyle \frac { \rho ^ { - 1 } - 1 } { ( 1 - \gamma ) ( \rho ^ { - K } - 1 ) } \displaystyle \sum _ { k = 0 } ^ { K - 1 } \rho ^ { - k } \mathbb { E } \left[ \mathbb { E } _ { s \sim d _ { \mu } ^ { * } } \left[ B _ { k } ^ { ( k ) } ( s ) \right] \right] . } & { } \end{array}\tag{66}
$$

Proof. Fix $k \geq 0$ and condition on $\mathcal { H } _ { k }$ . Applying Lemma 2.2 to $( \pi _ { \tau } ^ { * } , \pi _ { k } )$ and then isolating the signed critic error gives

$$
\begin{array} { l } { { V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi _ { k } } ( \mu ) = \displaystyle \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mu } ^ { * } } \left[ \langle \pi _ { \tau } ^ { * } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \rangle - \tau \Big ( h ^ { \pi _ { \tau } ^ { * } } ( s ) - h ^ { \pi _ { k } } ( s ) \Big ) \right] } } \\ { { \displaystyle \quad \quad = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { \mu } ^ { * } } \left[ \langle \pi _ { \tau } ^ { * } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { k } ( s , \cdot ) \rangle - \tau \Big ( h ^ { \pi _ { \tau } ^ { * } } ( s ) - h ^ { \pi _ { k } } ( s ) \Big ) + B _ { k } ^ { ( k ) } ( s ) \right] . } } \end{array}\tag{67}
$$

Lemma 2.4 with $p = \pi _ { \tau } ^ { * } ( \cdot \mid s )$ yields

$$
\begin{array} { r } { \eta \Big [ \Big \langle \pi _ { \tau } ^ { * } ( \cdot \ | \ s ) - \pi _ { k + 1 } ( \cdot \ | \ s ) , Q _ { \tau } ^ { k } ( s , \cdot ) \Big \rangle - \tau \Big ( h ^ { \pi _ { \tau } ^ { * } } ( s ) - h ^ { \pi _ { k + 1 } } ( s ) \Big ) \Big ] \leq D _ { \pi _ { k } ^ { * } } ^ { \pi _ { \tau } ^ { * } } ( s ) - \rho ^ { - 1 } D _ { \pi _ { k + 1 } ^ { * } } ^ { \pi _ { \tau } ^ { * } } ( s ) - D _ { \pi _ { k } } ^ { \pi _ { k + 1 } } ( s ) . } \end{array}\tag{68}
$$

By Young’s inequality,

$$
\eta \left. \pi _ { k + 1 } ( \cdot \ | \ s ) - \pi _ { k } ( \cdot \ | \ s ) \right. _ { 1 } \left. Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right. _ { \infty } - \frac \lambda 2 \left. \pi _ { k + 1 } ( \cdot \ | \ s ) - \pi _ { k } ( \cdot \ | \ s ) \right. _ { 1 } ^ { 2 } \leq \frac { \eta ^ { 2 } } { 2 \lambda } \left. Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right. _ { \infty } ^ { 2 } .
$$

Combining this inequality with (68),

$$
\left. \pi _ { \tau } ^ { * } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { k } ( s , \cdot ) \right. - \tau \Big ( h ^ { \pi _ { \tau } ^ { * } } ( s ) - h ^ { \pi _ { k } } ( s ) \Big )\tag{69}
$$

$$
\leq \frac { D _ { \tau _ { k } ^ { \tau } } ^ { \frac { n } { \tau } } ( s ) - \rho ^ { - 1 } D _ { \tau _ { k + 1 } ^ { \tau } } ^ { \frac { \tau } { \tau } } ( s ) } { \eta } + \frac { \eta } { 2 \lambda } \left. Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right. _ { \infty } ^ { 2 } + \left. \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \right. - \tau ( h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) ) .\tag{70}
$$

On the other hand, Lemma 2.4 with $p = \pi _ { k } ( \cdot \mid s )$ and the strong convexity of h imply that

$$
\langle \pi _ { k + 1 } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \rangle - \tau ( h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) ) + \frac { \eta } { 2 \lambda } \left. Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right. _ { \infty } ^ { 2 } \ge 0 .\tag{71}
$$

By restoring the state argument and leveraging the componentwise inequality $d _ { d _ { \mu } ^ { * } } ^ { \pi _ { k + 1 } } \geq ( 1 - \gamma ) d _ { \mu } ^ { * }$ , combining the non-negativity in (71) with Lemma 2.2 for $\left( \pi _ { k + 1 } , \pi _ { k } \right)$ with initial distribution $d _ { \mu } ^ { \ast }$ yields

$$
\begin{array} { r l } & { \mathbb { E } _ { s \sim d _ { \mu } ^ { * } } \left[ \langle \pi _ { k + 1 } ( \cdot \vert s ) - \pi _ { k } ( \cdot \vert s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) \rangle - \tau ( h ^ { \pi _ { k + 1 } } ( s ) - h ^ { \pi _ { k } } ( s ) ) \right] \leq V _ { \tau } ^ { \pi _ { k + 1 } } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { k } } ( d _ { \mu } ^ { * } ) } \\ & { \qquad + \frac { \gamma \eta } { 2 \lambda ( 1 - \gamma ) } \left\| Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right\| _ { \infty } ^ { 2 } . } \end{array}\tag{72}
$$

Combining (67), (70), and (72), we obtain

$$
\begin{array} { l } { { V _ { \tau } ^ { * } ( \mu ) - V _ { \tau } ^ { \pi _ { k } } ( \mu ) \leq \frac { D _ { \pi _ { k } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) - \rho ^ { - 1 } D _ { \pi _ { k + 1 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) } { \eta ( 1 - \gamma ) } + \frac { V _ { \tau } ^ { \pi _ { k + 1 } } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { k } } ( d _ { \mu } ^ { * } ) } { 1 - \gamma } } } \\ { { + \frac { \eta } { 2 \lambda ( 1 - \gamma ) ^ { 2 } } \| Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \| _ { \infty } ^ { 2 } + \frac { \mathbb { E } _ { s \sim d _ { \mu } ^ { * } } [ B _ { k } ^ { ( k ) } ( s ) ] } { 1 - \gamma } . } } \end{array}\tag{73}
$$

Moreover, Lemmas 2.3 and 4.2 imply that

$$
\left. Q _ { \tau } ^ { k } - Q _ { \tau } ^ { \pi _ { k } } \right. _ { \infty } \leq \frac { 2 ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } .\tag{74}
$$

Multiply (73) by $\rho ^ { - k }$ , sum over $k = 0 , \ldots , K - 1$ , and take expectation. The Bregman terms telescope as

$$
\sum _ { k = 0 } ^ { K - 1 } \rho ^ { - k } \left( D _ { \pi _ { k } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) - \rho ^ { - 1 } D _ { \pi _ { k + 1 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) \right) \leq D _ { \pi _ { 0 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) .
$$

Since $0 \leq V _ { \tau } ^ { * } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { k } } ( d _ { \mu } ^ { * } ) \leq 2 ( 1 + \tau H _ { h } ) / ( 1 - \gamma )$ , summation by parts gives

$$
\begin{array} { r } { \displaystyle \sum _ { k = 0 } ^ { K - 1 } \rho ^ { - k } \left( V _ { \tau } ^ { \pi _ { k + 1 } } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { k } } ( d _ { \mu } ^ { * } ) \right) \leq V _ { \tau } ^ { * } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { 0 } } ( d _ { \mu } ^ { * } ) + \displaystyle \sum _ { k = 1 } ^ { K - 1 } ( \rho ^ { - k } - \rho ^ { - ( k - 1 ) } ) \left( V _ { \tau } ^ { * } ( d _ { \mu } ^ { * } ) - V _ { \tau } ^ { \pi _ { k } } ( d _ { \mu } ^ { * } ) \right) } \\ { \leq \frac { 2 ( 1 + \tau H _ { h } ) } { 1 - \gamma } \left[ 1 + \displaystyle \sum _ { k = 1 } ^ { K - 1 } ( \rho ^ { - k } - \rho ^ { - ( k - 1 ) } ) \right] = \frac { 2 \rho ^ { - ( K - 1 ) } ( 1 + \tau H _ { h } ) } { 1 - \gamma } . } \end{array}\tag{75}
$$

Finally, multiplying the resulting inequality by $( \rho ^ { - 1 } - 1 ) / ( \rho ^ { - K } - 1 )$ , using $\rho ^ { - 1 } - 1 = \eta \tau .$ , (74), and the definition of $\widehat { K }$ proves (66). □

We now decompose the last term in (66), following the notation of Li et al. [2025, Lemma 3.9]. Stack $\{ \pi ( \cdot \mid s ) \} _ { s \in { \mathcal { S } } }$ into a vector $\pi \in \mathbb { R } ^ { | S | | A | }$ . For each $s \in S$ , let $E _ { s } \in \mathbb { R } ^ { | \mathcal { A } | \times | \mathcal { S } | | \mathcal { A } | }$ select the coordinates associated with s, and set

$$
E _ { s } Q = Q ( s , \cdot ) , \qquad E _ { s } \pi = \pi ( \cdot \mid s ) , \qquad J _ { s } : = E _ { s } ^ { \top } E _ { s } .\tag{76}
$$

Suppose $\alpha _ { k } = \alpha \in ( 0 , 1 ]$ for every $k \geq 0$ . For a constant critic stepsize α and a policy π, define

$$
A ^ { \pi } : = I - \alpha \Sigma _ { b } \left( I - \gamma P _ { s \times { \cal A } } ^ { \pi } \right) .\tag{77}
$$

Since $Q _ { \tau } ^ { \pi _ { j } } = \mathcal { F } _ { \tau } ^ { \pi _ { j } } Q _ { \tau } ^ { \pi _ { j } }$ and $\mathcal { F } _ { \tau } ^ { \pi _ { j } }$ has a linear part $\gamma P _ { s \times A } ^ { \pi _ { j } } .$ , it follows from (31) that

$$
Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { j } = A ^ { \pi _ { j } } \left( Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { j - 1 } \right) - \alpha \bar { \omega } _ { j - 1 } , \qquad j \geq 1 .\tag{78}
$$

Fix $k \geq 0$ and $s \in S$ . For $0 \leq j \leq k$ , define

$$
B _ { j } ^ { ( k ) } ( s ) : = \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { j } \right. , \qquad ( A ^ { \pi _ { j } } ) ^ { 0 } : = I .\tag{79}
$$

For $1 \le j \le k$ , define

$$
\begin{array} { r } { C _ { j } ^ { ( k ) } ( s ) : = \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j + 1 } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { \pi _ { j - 1 } } \right. , } \end{array}\tag{80}
$$

$$
\begin{array} { r } { D _ { j } ^ { ( k ) } ( s ) : = \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j + 1 } \right] ^ { \top } J _ { s } ( \pi _ { j - 1 } - \pi _ { j } ) , Q _ { \tau } ^ { \pi _ { j - 1 } } - Q _ { \tau } ^ { j - 1 } \right. , } \end{array}\tag{81}
$$

$$
E _ { j } ^ { ( k ) } ( s ) : = \left. \left( \left[ ( A ^ { \pi _ { j } } ) ^ { k - j + 1 } \right] ^ { \top } - \left[ ( A ^ { \pi _ { j - 1 } } ) ^ { k - j + 1 } \right] ^ { \top } \right) J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j - 1 } ) , Q _ { \tau } ^ { \pi _ { j - 1 } } - Q _ { \tau } ^ { j - 1 } \right. ,\tag{82}
$$

$$
F _ { j } ^ { ( k ) } ( s ) : = - \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , \alpha \bar { \omega } _ { j - 1 } \right. .\tag{83}
$$

Here $B _ { i } ^ { ( k ) }$ is the propagated signed critic error, with $B _ { 0 } ^ { ( k ) }$ representing the initial remainder. The terms $C _ { j } ^ { ( k ) }$ $D _ { j } ^ { ( k ) } , \check { E } _ { j } ^ { ( k ) }$ , and $F _ { j } ^ { ( k ) }$ capture, respectively, the true-Q drift, policy drift, operator drift, and Markov-noise contribution.

## Lemma B.2 (Five-term decomposition). One has

$$
\langle \pi _ { \tau } ^ { * } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) - Q _ { \tau } ^ { k } ( s , \cdot ) \rangle = { \cal B } _ { 0 } ^ { ( k ) } ( s ) + \sum _ { j = 1 } ^ { k } \left( C _ { j } ^ { ( k ) } ( s ) + D _ { j } ^ { ( k ) } ( s ) + E _ { j } ^ { ( k ) } ( s ) + F _ { j } ^ { ( k ) } ( s ) \right) .\tag{84}
$$

Consequently,

$$
\mathbb { E } \left[ B _ { k } ^ { ( k ) } ( s ) \right] \leq \mathbb { E } \left[ \left| B _ { 0 } ^ { ( k ) } ( s ) \right| \right] + \sum _ { j = 1 } ^ { k } \mathbb { E } \left[ \left| C _ { j } ^ { ( k ) } ( s ) \right| + \left| D _ { j } ^ { ( k ) } ( s ) \right| \right] + \left| \mathbb { E } \left[ \sum _ { j = 1 } ^ { k } E _ { j } ^ { ( k ) } ( s ) \right] \right| + \left| \mathbb { E } \left[ \sum _ { j = 1 } ^ { k } F _ { j } ^ { ( k ) } ( s ) \right] \right| .\tag{85}
$$

Proof. By (76) and $\left( A ^ { \pi _ { k } } \right) ^ { 0 } = I$

$$
\left. \pi _ { \tau } ^ { * } ( \cdot \mid s ) - \pi _ { k } ( \cdot \mid s ) , Q _ { \tau } ^ { \pi _ { k } } ( s , \cdot ) - Q _ { \tau } ^ { k } ( s , \cdot ) \right. = B _ { k } ^ { ( k ) } ( s ) = B _ { 0 } ^ { ( k ) } ( s ) + \sum _ { j = 1 } ^ { k } \left( B _ { j } ^ { ( k ) } ( s ) - B _ { j - 1 } ^ { ( k ) } ( s ) \right) .\tag{86}
$$

For $1 \le j \le k$ , using (78) we obtain

$$
\begin{array} { r l } & { B _ { j } ^ { ( k ) } ( s ) = \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j + 1 } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { j - 1 } \right. - \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , \alpha \bar { \omega } _ { j - 1 } \right. } \\ & { \qquad = B _ { j - 1 } ^ { ( k ) } ( s ) + C _ { j } ^ { ( k ) } ( s ) + D _ { j } ^ { ( k ) } ( s ) + E _ { j } ^ { ( k ) } ( s ) + F _ { j } ^ { ( k ) } ( s ) . } \end{array}\tag{87}
$$

Substituting (87) into (86) proves (84). Taking expectations and bounding the aggregate operator drift and Markov bias in absolute value proves (85). □

To proceed, let $q _ { b } : = 1 - \alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } \in ( 0 , 1 )$ . The definition of $A ^ { \pi _ { j } }$ gives

$$
\| \big ( A ^ { \pi _ { j } } \big ) ^ { \ell } x \| _ { \infty } \leq q _ { b } ^ { \ell } \| x \| _ { \infty } , \qquad \sum _ { \ell = 1 } ^ { \infty } q _ { b } ^ { \ell } \leq \frac { 1 } { \alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } } .
$$

Lemma B.3 (Bound for the true-Q drift). For every $k \geq 1$ and $s \in S$

$$
\sum _ { j = 1 } ^ { k } \left| C _ { j } ^ { ( k ) } ( s ) \right| \leq \frac { 2 \gamma \eta } { \lambda \alpha ( 1 - \gamma ) ^ { 2 } \widetilde \sigma _ { b } } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) ^ { 2 } .\tag{88}
$$

Proof. The $\ell _ { 1 } - \ell _ { \infty }$ duality gives that

$$
\begin{array} { r } { \left| C _ { j } ^ { ( k ) } ( s ) \right| \leq \| J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) \| _ { 1 } \left\| ( A ^ { \pi _ { j } } ) ^ { k - j + 1 } ( Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { \pi _ { j - 1 } } ) \right\| _ { \infty } \leq 2 q _ { b } ^ { k - j + 1 } \| Q _ { \tau } ^ { \pi _ { j } } - Q _ { \tau } ^ { \pi _ { j - 1 } } \| _ { \infty } . } \end{array}\tag{89}
$$

Here $\lVert J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) \rVert _ { 1 } \leq 2$ , and the last inequality uses the contraction of $A ^ { \pi _ { j } }$ . Lemma 4.3 then implies

$$
\left| C _ { j } ^ { ( k ) } ( s ) \right| \le \frac { 2 \gamma \eta } { \lambda ( 1 - \gamma ) } q _ { b } ^ { k - j + 1 } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) ^ { 2 } .\tag{90}
$$

Summing over $j$ and using $\begin{array} { r } { \sum _ { j = 1 } ^ { k } q _ { b } ^ { k - j + 1 } \leq [ \alpha ( 1 - \gamma ) \widetilde \sigma _ { b } ] ^ { - 1 } } \end{array}$ proves (88).

Lemma B.4 (Bound for the policy drift). For every $k \geq 1$ and $s \in S$

$$
\sum _ { j = 1 } ^ { k } \left| D _ { j } ^ { ( k ) } ( s ) \right| \le \frac { 2 \eta ( 1 + \tau \gamma H _ { h } ) } { \lambda \alpha ( 1 - \gamma ) ^ { 2 } \widetilde { \sigma } _ { b } } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) .\tag{91}
$$

Proof. By the $\ell _ { 1 } - \ell _ { \infty }$ duality and the contraction of $A ^ { \pi _ { j } }$

$$
\begin{array} { r l } & { \left| D _ { j } ^ { ( k ) } ( s ) \right| \leq \| J _ { s } \big ( \pi _ { j - 1 } - \pi _ { j } \big ) \| _ { 1 } \left\| \big ( A ^ { \pi _ { j } } \big ) ^ { k - j + 1 } \big ( Q _ { \tau } ^ { \pi _ { j - 1 } } - Q _ { \tau } ^ { j - 1 } \big ) \right\| _ { \infty } } \\ & { \qquad \leq q _ { b } ^ { k - j + 1 } \| \pi _ { j } - \pi _ { j - 1 } \| _ { 1 , \infty } \left\| Q _ { \tau } ^ { \pi _ { j - 1 } } - Q _ { \tau } ^ { j - 1 } \right\| _ { \infty } . } \end{array}\tag{92}
$$

Applying (28) and (74) yields

$$
\left| D _ { j } ^ { ( k ) } ( s ) \right| \leq \frac { 2 \eta ( 1 + \tau \gamma H _ { h } ) } { \lambda ( 1 - \gamma ) } q _ { b } ^ { k - j + 1 } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) .\tag{93}
$$

Summing this series as in the preceding proof gives (91).

Lemma B.5 (Exponentially weighted operator drift). Suppose $\rho ^ { - 1 } q _ { b } < 1$ . Then, for every $K \geq 2$ and $s \in S$

$$
\left| \mathbb { E } \left[ \sum _ { j = 1 } ^ { \widehat { K } } E _ { j } ^ { ( \widehat { K } ) } ( s ) \right] \right| \leq \frac { 4 \alpha \gamma \widetilde { \sigma } _ { b } \eta ( 1 + \tau \gamma H _ { h } ) } { \lambda ( 1 - \gamma ) ( 1 - \rho ^ { - 1 } q _ { b } ) ^ { 2 } } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) .\tag{94}
$$

Proof. For $N \geq 1$ , let $\begin{array} { r } { S _ { j , N } : = \sum _ { n = 1 } ^ { N } \rho ^ { - n } \big [ ( A ^ { \pi _ { j } } ) ^ { n } - ( A ^ { \pi _ { j - 1 } } ) ^ { n } \big ] } \end{array}$ . Recall the distribution of $\widehat { K }$ . Exchanging the finite $k -$ and j-sums and setting $n = k - j + 1$ give

$$
\mathbb E \left[ \sum _ { j = 1 } ^ { \widehat K } E _ { j } ^ { ( \widehat K ) } ( s ) \right] = \frac { \rho ^ { - 1 } - 1 } { \rho ^ { - K } - 1 } \sum _ { j = 1 } ^ { K - 1 } \rho ^ { - ( j - 1 ) } \mathbb E \left[ \langle S _ { j , K - j } ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j - 1 } ) , Q _ { \tau } ^ { \pi _ { j - 1 } } - Q _ { \tau } ^ { j - 1 } \rangle \right] .\tag{95}
$$

Thus it remains to bound $\| S _ { j , N } \| _ { \infty }$ uniformly in $N .$ . To this end, we first verify the following estimate whenever $\rho ^ { - 1 } q _ { b } < 1$ 2

$$
\left\| \left( I - \rho ^ { - 1 } A ^ { \pi } \right) ^ { - 1 } \alpha \Sigma _ { b } \right\| _ { \infty } \leq \frac { \alpha \widetilde { \sigma } _ { b } } { 1 - \rho ^ { - 1 } q _ { b } } .\tag{96}
$$

With $c : = \alpha \widetilde { \sigma } _ { b } / ( 1 - \rho ^ { - 1 } q _ { b } )$ , the identity $A ^ { \pi _ { j } } \mathbf { 1 } = \mathbf { 1 } - \alpha ( 1 - \gamma ) \Sigma _ { b } \mathbf { 1 }$ implies $( I - \rho ^ { - 1 } A ^ { \pi _ { j } } ) c { \bf 1 } \geq \alpha \Sigma _ { b } { \bf 1 }$ . Since $A ^ { \pi _ { j } }$ is nonnegative and $\rho ^ { - 1 } q _ { b } < 1$ , comparison with its convergent Neumann series proves (96). It also implies, for every $L \geq 0$

$$
\left. \sum _ { \ell = 0 } ^ { L } \left( \rho ^ { - 1 } A ^ { \pi _ { j } } \right) ^ { \ell } \alpha \Sigma _ { b } \right. _ { \infty } \leq \frac { \alpha \widetilde { \sigma } _ { b } } { 1 - \rho ^ { - 1 } q _ { b } } .
$$

Using $A ^ { \pi _ { j } } - A ^ { \pi _ { j - 1 } } = \alpha \gamma \Sigma _ { b } ( P _ { s \times A } ^ { \pi _ { j } } - P _ { s \times A } ^ { \pi _ { j - 1 } } )$ and the power-diference identity,

$$
\begin{array} { l } { { S _ { j , N } = \displaystyle \sum _ { n = 1 } ^ { N } \rho ^ { - n } \sum _ { m = 0 } ^ { n - 1 } ( A ^ { \pi _ { j } } ) ^ { n - 1 - m } ( A ^ { \pi _ { j } } - A ^ { \pi _ { j - 1 } } ) ( A ^ { \pi _ { j - 1 } } ) ^ { m } } } \\ { { \displaystyle \quad = \rho ^ { - 1 } \gamma \sum _ { m = 0 } ^ { N - 1 } \left[ \sum _ { \ell = 0 } ^ { N - 1 - m } ( \rho ^ { - 1 } A ^ { \pi _ { j } } ) ^ { \ell } \alpha \Sigma _ { b } \right] \left( P _ { s \times A } ^ { \pi _ { j } } - P _ { s \times A } ^ { \pi _ { j - 1 } } \right) ( \rho ^ { - 1 } A ^ { \pi _ { j - 1 } } ) ^ { m } . } } \end{array}\tag{97}
$$

The second equality substitutes the expression for $A ^ { \pi _ { j } } - A ^ { \pi _ { j - 1 } }$ , sets $\ell = n - 1 - m$ , and exchanges the two finite sums. For every vector $x ,$

$$
\begin{array} { r } { \big \| \big ( P _ { s \times A } ^ { \pi _ { j } } - P _ { s \times A } ^ { \pi _ { j - 1 } } \big ) x \big \| _ { \infty } \leq \| \pi _ { j } - \pi _ { j - 1 } \| _ { 1 , \infty } \| x \| _ { \infty } , \qquad \| \big ( \rho ^ { - 1 } A ^ { \pi _ { j - 1 } } \big ) ^ { m } \| _ { \infty } \leq ( \rho ^ { - 1 } q _ { b } ) ^ { m } . } \end{array}
$$

Therefore, applying (96) to the first bracket in (97) and summing $\begin{array} { r } { \sum _ { m \ge 0 } ( \rho ^ { - 1 } q _ { b } ) ^ { m } = ( 1 - \rho ^ { - 1 } q _ { b } ) ^ { - 1 } } \end{array}$ yield

$$
\| S _ { j , N } \| _ { \infty } \leq \frac { \rho ^ { - 1 } \alpha \gamma \widetilde { \sigma } _ { b } } { ( 1 - \rho ^ { - 1 } q _ { b } ) ^ { 2 } } \| \pi _ { j } - \pi _ { j - 1 } \| _ { 1 , \infty } .\tag{98}
$$

Finally, as $\lVert J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j - 1 } ) \rVert _ { 1 } \leq 2$ , (74) implies that

$$
\begin{array} { r l } & { \left| \mathbb E \left[ \displaystyle \sum _ { j = 1 } ^ { \widehat K } E _ { j } ^ { ( \widehat K ) } ( s ) \right] \right| \leq \frac { 4 ( \rho ^ { - 1 } - 1 ) ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ( \rho ^ { - K } - 1 ) } \displaystyle \sum _ { j = 1 } ^ { K - 1 } \rho ^ { - ( j - 1 ) } \mathbb E \left[ \| S _ { j , K - j } \| _ { \infty } \right] } \\ & { \qquad \leq \frac { 4 \alpha \gamma \widetilde \sigma _ { b } \eta ( 1 + \tau \gamma H _ { h } ) } { \lambda ( 1 - \gamma ) ( 1 - \rho ^ { - 1 } q _ { b } ) ^ { 2 } } \left( \frac { 1 + \tau \gamma H _ { h } } { 1 - \gamma } + \tau L _ { h } \right) , } \end{array}
$$

where we leverage (28), (98), and the fact that $\textstyle ( \rho ^ { - 1 } - 1 ) \sum _ { j = 1 } ^ { K - 1 } \rho ^ { - ( j - 1 ) } / ( \rho ^ { - K } - 1 ) \leq \rho$ in the last inequality.

For the weights in (34), the condition $0 \leq \vartheta < \kappa _ { b }$ and summation of the resulting powers give

$$
\sum _ { t = 0 } ^ { B - 1 } c _ { t } ^ { k } \kappa _ { b } ^ { t } = \frac { \kappa _ { b } ^ { B } - \vartheta ^ { B } } { \left( \kappa _ { b } - \vartheta \right) \sum _ { \ell = 0 } ^ { B - 1 } \vartheta ^ { \ell } } \leq \frac { \kappa _ { b } ^ { B } } { \kappa _ { b } - \vartheta } .\tag{99}
$$

Lemma B.6 (Bound for the Markov-noise term). For every $k \geq 1$ and $s \in S$

$$
\left\| \mathbb { E } \left[ \sum _ { j = 1 } ^ { k } F _ { j } ^ { ( k ) } ( s ) \right] \right\| \leq \frac { 4 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } \widetilde { \sigma } _ { b } } \frac { \kappa _ { b } ^ { B } } { \kappa _ { b } - \vartheta } .\tag{100}
$$

Proof. Since $A ^ { \pi _ { j } } , \pi _ { j }$ , and the initial state of batch $j - 1$ are $\mathcal { H } _ { j - 1 }$ -measurable, conditioning the definition of $F _ { j } ^ { ( k ) } ( s )$ on $\mathcal { H } _ { j - 1 }$ yields

$$
\begin{array} { r l } & { \left| \mathbb { E } \left[ F _ { j } ^ { ( k ) } ( s ) \right] \right| = \alpha \left| \mathbb { E } \left[ \left. \left[ ( A ^ { \pi _ { j } } ) ^ { k - j } \right] ^ { \top } J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) , \mathbb { E } [ \bar { \omega } _ { j - 1 } \mid \mathcal { H } _ { j - 1 } ] \right. \right] \right| } \\ & { \phantom { \left| \mathbb { E } \left[ F _ { j } ^ { ( k ) } ( s ) \right] \right| } \leq 2 \alpha q _ { b } ^ { k - j } \mathbb { E } \left[ \| \mathbb { E } [ \bar { \omega } _ { j - 1 } \mid \mathcal { H } _ { j - 1 } ] \| _ { \infty } \right] } \\ & { \phantom { \left| \mathbb { E } \left[ F _ { j } ^ { ( k ) } ( s ) \right] \right| } \leq \frac { 4 \alpha m _ { b } ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } q _ { b } ^ { k - j } \frac { \kappa _ { b } ^ { B } } { \kappa _ { b } - \vartheta } , } \end{array}\tag{101}
$$

where we use $\lVert J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { j } ) \rVert _ { 1 } \leq 2$ and the contraction of $A ^ { \pi _ { j } }$ in the first inequality, and the second inequality is due to Lemma 4.4 together with (99). Finally, linearity of expectation and the triangle inequality give

$$
\left. \mathbb { E } \left[ \sum _ { j = 1 } ^ { k } F _ { j } ^ { ( k ) } ( s ) \right] \right. \leq \sum _ { j = 1 } ^ { k } \left. \mathbb { E } [ F _ { j } ^ { ( k ) } ( s ) ] \right. \leq \frac { 4 m _ { b } ( 1 + \tau \gamma H _ { h } ) } { 1 - \gamma } \frac { \kappa _ { b } ^ { B } } { \kappa _ { b } - \vartheta } \alpha \sum _ { \ell = 0 } ^ { k - 1 } q _ { b } ^ { \ell } .
$$

Since $\begin{array} { r } { \chi \sum _ { \ell = 0 } ^ { k - 1 } q _ { b } ^ { \ell } \le [ ( 1 - \gamma ) \widetilde \sigma _ { b } ] ^ { - 1 } } \end{array}$ , (100) is proved.

We now substitute the preceding five bounds above into (66). First, the initial propagation term satisfies, uniformly in $s \in S$ ,

$$
\begin{array} { r } { \left| B _ { 0 } ^ { ( k ) } ( s ) \right| \leq 2 q _ { b } ^ { k } \left\| Q _ { \tau } ^ { \pi _ { 0 } } - Q _ { \tau } ^ { 0 } \right\| _ { \infty } . } \end{array}\tag{102}
$$

Indeed, $\lVert J _ { s } ( \pi _ { \tau } ^ { * } - \pi _ { 0 } ) \rVert _ { 1 } \leq 2$ and $\| \left( A ^ { \pi _ { 0 } } \right) ^ { k } x \| _ { \infty } \leq q _ { b } ^ { k } \| x \| _ { \infty }$ . Hence, whenever $\rho ^ { - 1 } q _ { b } < 1$ ，

$$
\mathbb { E } \left[ \left| B _ { 0 } ^ { ( \widehat { K } ) } ( s ) \right| \right] \leq \frac { 2 ( \rho ^ { - 1 } - 1 ) \left[ 1 - ( \rho ^ { - 1 } q _ { b } ) ^ { K } \right] } { ( \rho ^ { - K } - 1 ) ( 1 - \rho ^ { - 1 } q _ { b } ) } \left\| Q _ { \tau } ^ { \pi _ { 0 } } - Q _ { \tau } ^ { 0 } \right\| _ { \infty } .\tag{103}
$$

Proof of Theorem 4.5. Since $\rho = ( 1 + \eta \tau ) ^ { - 1 } \in ( 0 , 1 )$

$$
\rho ^ { - K } \geq 2 \iff K \geq \frac { \log 2 } { \log ( 1 / \rho ) } .
$$

Thus, the stated integer lower bound on $K$ ensures $\rho ^ { - K } \geq 2$ . Writing $a : = \alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } .$ , the stepsize bound in the theorem gives $1 - \rho = \eta \tau / ( 1 + \eta \tau ) \le a / 2$ and hence the common stability margin

$$
1 - \rho ^ { - 1 } q _ { b } = \rho ^ { - 1 } \big [ a - ( 1 - \rho ) \big ] \geq \frac { 1 } { 2 } \rho ^ { - 1 } \alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } > 0 .
$$

Thus, Lemma B.5 applies. Moreover, Lemmas 2.3 and 4.2, together with $Q _ { \tau } ^ { 0 } = 0 ;$ give $\| Q _ { \tau } ^ { \pi _ { 0 } } - Q _ { \tau } ^ { 0 } \| _ { \infty } \leq$ $( 1 + \tau \gamma H _ { h } ) / ( 1 - \gamma )$ . Consequently, (103) bounds the contribution of $B _ { 0 }$ by the second summand in $C _ { 1 }$ . The condition $\rho ^ { - K } \geq 2$ also gives

$$
\frac { 2 ( \rho ^ { - 1 } - 1 ) \rho ^ { - ( K - 1 ) } ( 1 + \tau H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } ( \rho ^ { - K } - 1 ) } \leq \frac { 4 \eta \tau ( 1 + \tau H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } } .
$$

Substituting Lemmas B.3, B.4, B.5, and B.6 into (85) and then into (66), the C- and E-terms combine as

$$
\frac { 2 \gamma \eta \left[ 1 + \tau \gamma H _ { h } + \tau ( 1 - \gamma ) L _ { h } \right] \left[ 9 ( 1 + \tau \gamma H _ { h } ) + \tau ( 1 - \gamma ) L _ { h } \right] } { \lambda \alpha ( 1 - \gamma ) ^ { 5 } \widetilde \sigma _ { b } } .
$$

Collecting the remaining terms proves (35).

## B.5 Proof of Corollary 4.6

Constant-stepsize bias. For this proof only, set $A : = 1 + \tau \gamma H _ { h } , R : = \tau ( 1 - \gamma ) L _ { h }$ , and $G : = ( A + R ) ( 9 A + R )$ Since $A \geq 1$ and $R \geq 0$ , we have $G \geq 9 A ^ { 2 } \geq 9$ . Additionally, the constant $C _ { 2 }$ is at most

$$
\frac { 4 \big [ 1 + \lambda \tau ( 1 + \tau H _ { h } ) \big ] G } { \lambda \alpha ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } }
$$

as $\alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } < 1$ and $\gamma \leq 1$ . Indeed, after division by $G / [ \lambda \alpha ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } ]$ , the first three summands are bounded by $4 \lambda \tau ( 1 + \tau H _ { h } ) , 1$ , and 1, respectively, while the last is bounded by 2. Thus, (36) ensures that the second term in (35) does not exceed $\epsilon / 3$

Stability and optimization error. Writing $a : = \alpha ( 1 - \gamma ) \widetilde { \sigma } _ { b } \in ( 0 , 1 )$ within this proof, there holds

$$
\frac { \eta _ { \epsilon } \tau ( 2 - a ) } { a } \leq \frac { 2 \lambda \tau } { 1 2 \bigl [ 1 + \lambda \tau ( 1 + \tau H _ { h } ) \bigr ] G } < 1 .
$$

Hence, the stepsize condition in Theorem 4.5 holds and $0 < \eta _ { \epsilon } \tau < 1$ , which implies $\log ( 1 + \eta _ { \epsilon } \tau ) \ge \eta _ { \epsilon } \tau / 2$ Furthermore,

$$
\frac { \tau D _ { \pi _ { 0 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) } { 1 - \gamma } + \frac { 4 \eta _ { \epsilon } \tau ( 1 + \tau \gamma H _ { h } ) } { \alpha ( 1 - \gamma ) ^ { 3 } \widetilde \sigma _ { b } } \leq \frac { \tau D _ { \pi _ { 0 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) } { 1 - \gamma } + \frac { 4 ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } } .
$$

Consequently, (37) yields

$$
( 1 + \eta _ { \epsilon } \tau ) ^ { K _ { \epsilon } } - 1 \geq \operatorname* { m a x } \biggr \{ 1 , \frac { 3 } { \epsilon } \left[ \frac { \tau D _ { \pi _ { 0 } } ^ { \pi _ { \tau } ^ { * } } ( d _ { \mu } ^ { * } ) } { 1 - \gamma } + \frac { 4 ( 1 + \tau \gamma H _ { h } ) } { ( 1 - \gamma ) ^ { 2 } } \right] \biggr \} .
$$

In particular, $( 1 + \eta _ { \epsilon } \tau ) ^ { K _ { \epsilon } } \geq 2$ , and the first term in (35) does not exceed $\epsilon / 3$

Mixing error and sample count. Finally, (38) gives

$$
\frac { \kappa _ { b } ^ { B _ { \epsilon } } } { \kappa _ { b } - \vartheta } \leq \frac { \epsilon ( 1 - \gamma ) ^ { 3 } \widetilde { \sigma } _ { b } } { 1 2 m _ { b } ( 1 + \tau \gamma H _ { h } ) } ,
$$

so the third term does not exceed $\epsilon / 3$ . Summing the three bounds proves (39). Moreover, the definition of $L _ { h }$ gives

$$
R = \tau ( 1 - \gamma ) H _ { h } + \operatorname* { m a x } \left\{ \frac { \tau ( 1 - \gamma ) } { 2 } \operatorname* { m a x } _ { s \in \mathcal { S } , a \in A } D _ { h } ( e _ { a } \parallel \pi _ { 0 } ( \cdot \mid s ) ) , A + \tau ( 1 - \gamma ) H _ { h } \right\} ,
$$

so $A , R ,$ , and $G$ are bounded when $\tau , H _ { h }$ , and $\operatorname* { m a x } _ { s , a } D _ { h } ( e _ { a } \parallel \pi _ { 0 } ( \cdot \textit { \textbf { \em s } } ) )$ are fixed. Hence (36)–(38) give $K _ { \epsilon } = \widetilde { \mathcal { O } } \big ( [ ( 1 - \gamma ) ^ { 5 } \widetilde { \sigma } _ { b } \epsilon ] ^ { - 1 } \big )$ and $B _ { \epsilon } = \widetilde { \mathcal { O } } ( 1 )$ under the stated convention, and then the claimed bound for $N _ { \epsilon } = K _ { \epsilon } B _ { \epsilon }$ follows.