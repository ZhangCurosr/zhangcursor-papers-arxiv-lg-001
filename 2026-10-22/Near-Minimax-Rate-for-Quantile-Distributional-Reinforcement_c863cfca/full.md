# Near Minimax Rate for Quantile Distributional Reinforcement Learning

Zijie Cheng<sup>∗</sup>

Zhihua Zhang<sup>†</sup>

October 22, 2026

## Abstract

In this paper, we study quantile distributional policy evaluation, where the return distribution under a given policy is approximated by the quantile Bellman fixed point $\pmb { \eta } _ { m }$ . We investigate the statistical accuracy of estimating $\pmb { \eta } _ { m }$ across quantile resolutions under Wasserstein distances. Assuming access to a generative model, we construct a quantile fixed point estimator $\eta _ { m } ^ { ( n ) }$ from an empirical Markov decision process and establish non-asymptotic error bounds under Wasserstein distances. When reward densities are uniformly bounded away from zero, we obtain a root-n high-probability bound simultaneously over all resolutions m and all Wasserstein orders $1 \leqslant p \leqslant \infty$ . When reward densities vanish at the support endpoints, we show that continuity and endpoint monotonicity sufice to preserve the same resolution-independent rate for $1 \leqslant p \leqslant 2$ . For $p > 2$ , we establish a root-n bound with coeficient $m ^ { 1 / 2 - 1 / p }$ , together with a slower, resolutionindependent $n ^ { - 1 / p }$ bound for every finite $p > 2$ . We further obtain resolution-independent $W _ { \infty }$ consistency governed by the continuity modulus of reward quantile functions.

To investigate the sharpness of these bounds, we characterize the exact local asymptotic minimax risk at each fixed resolution as the expected Wasserstein-coordinate norm of the eficient Gaussian limit and prove that our estimator attains it. We then examine the resolution dependence of this optimal local risk. A regular one-state instance has resolution-independent local risk constants, whereas a single endpoint-vanishing two-state instance exhibits growth of order $m ^ { 1 / 2 - 1 / p }$ , up to logarithmic factors along a resolution subsequence, for every $p > 2 .$

## Contents

1 Introduction 3   
1.1 Our Contributions 4   
1.2 Related Works 6   
2 Preliminaries 7   
2.1 Problem Setup 7   
2.2 Metrics on the Space of Measures 7   
2.3 Distributional Bellman Operator and Quantile Projection Operator 8   
2.4 Main Assumptions 10   
3 Main Results 12   
3.1 The Quantile Fixed Point Estimator 12   
3.2 Uniformly Positive Reward Densities 13   
3.3 Endpoint-Vanishing Reward Densities 14   
3.4 Local Asymptotic Minimaxity 16   
3.5 Fixed Instances and Sharp Resolution Dependence 18   
4 Proof Outlines 19   
4.1 Analysis under the Uniform Lower Density Condition 19   
4.2 Analysis under Endpoint Vanishing 20   
4.3 Larger Wasserstein Orders . 25   
4.4 Resolution-Independent $W _ { \infty }$ Consistency . 25   
4.5 Analysis of the Local Asymptotic Minimax Risk 27   
4.6 Resolution Dependence in the Fixed Instances . 30   
5 Conclusions 33   
A Proofs of Supporting Results in Section 4 38   
A.1 Proof of Lemma 4.1 38   
A.2 Proofs of Lemmas 4.2–4.5 40   
A.3 Proof of Lemma 4.6 46   
A.4 Proofs of Lemmas 4.7–4.10 47   
A.5 Proofs of Lemmas 4.11–4.12 51   
B Technical Lemmas 55   
B.1 Basic Geometry and Paired Empirical Laws 55   
B.2 Density-Weighted Empirical-CDF Bounds 63   
B.3 Deterministic Quantile Inversion 67   
B.4 Local Geometry of the Quantile Bellman Map 71   
B.5 Technical Lemmas for the Local Statistical Experiment . 76   
B.6 Construction and Localization of the Hard Instance . 79

## 1 Introduction

Reinforcement learning has achieved substantial progress across a wide range of domains, including game-playing [27, 33], robotics [14], and large language models [19, 18]. In the classical formulation of reinforcement learning based on the reward hypothesis [29, 30], policy performance is evaluated through expected returns (i.e., the expected discounted cumulative reward). While this criterion provides a principled objective for sequential decision-making, it ignores the distributional characteristics of returns, such as variability, tail risk, and higher-order distributional information. Such considerations are important in many real-world applications. In healthcare, for instance, one is concerned not only with average outcomes but also with adverse or long-tail efects of dynamic treatment regimes, as poor outcomes may have severe consequences for patients [15]. In financial decision-making, investors must balance return and risk, since higher expected returns are typically associated with greater uncertainty [10].

Distributional reinforcement learning [17, 1] generalizes the classical expected-return framework by characterizing the full distribution of return rather than only its expectation. This distributional perspective provides a richer description of uncertainty in the performance of learning agents, capturing variability arising from both the stochastic nature of environments and the randomness induced by the agent’s policy. However, return distributions are generally infinite-dimensional objects, and hence cannot be represented or learned exactly. This motivates finite-dimensional approximations of return distributions, among which categorical and quantile-based representations are two prominent choices. In particular, a widely-used family of algorithms for distributional reinforcement learning is based on the notion of quantile representations, an approach that originated with Dabney et al. [9]. By approximating the return distribution through a finite collection of quantiles, quantile representations provide a flexible and computationally tractable approximation while preserving key distributional information. As a result, quantile-based approaches have been widely adopted in practice and have become a standard paradigm in modern distributional reinforcement learning, underpinning methods such as quantile regression deep Q-network (QR-DQN) [9], implicit quantile network (IQN) [8], and non-crossing quantile regression method [37]. This paradigm has also been successfully applied in a range of settings, including high-altitude balloon guidance [2], robotic manipulation [4], and benchmark simulated domains such as the Arcade Learning Environment [9, 8].

The statistical analysis of a quantile representation must account for both the sample size and the number of quantiles. Increasing the number of atoms improves the representation of a return law, but it also introduces more parameters and moves the outermost quantile levels closer to zero and one. A small density at such a level can amplify the estimation error of an individual coordinate. In a Wasserstein distance, however, each coordinate has probability mass $1 / m$ . Whether this averaging compensates for the instability of extreme quantiles is not determined by a coordinatewise analysis.

Cheng et al. [7] developed a model-based analysis of the quantile Bellman fixed point, including a non-asymptotic supremum $W _ { \infty }$ bound with coeficient of order $\sqrt { m }$ up to logarithmic factors, fixed-resolution asymptotic normality and eficiency, and a limiting covariance analysis as the quantile resolution increases. These results leave a more specific question: which dependence on the resolution is intrinsic to estimation, and which dependence comes from the choice of loss or from bounding all quantile coordinates by the worst one? In this paper, we answer this question by studying estimation of the projected fixed point under the full family of Wasserstein distances.

The role of the density assumptions is central. A positive lower density bound, familiar from quantile estimation, rules out local degeneracy in the reward law. We also consider a complementary regime in which the reward density vanishes at the support endpoints and is monotone near them. This second regime imposes no quantitative rate of boundary decay. We show that the statistical cost of removing the lower density bound depends on the Wasserstein order: resolution-independent root-n estimation survives through $W _ { 2 }$ , whereas higher orders can have an unavoidable polynomial dependence on m in their first-order statistical constant.

From a theoretical perspective, a quantile Bellman fixed point is not a collection of ordinary sample quantiles. Each continuation quantile changes the Bellman target distribution at other states, and the target must then be projected back onto the midpoint quantile grid. Although the projected Bellman map is globally contractive in supremum $W _ { \infty }$ metric, that property alone does not preserve the averaging that makes an m-free supremum $W _ { 2 }$ bound possible. The proof must control the empirical perturbation, invert it on the actual finite grid, and propagate it through the nonlinear fixed-point equation without adding a discretization remainder.

## 1.1 Our Contributions

In this paper, we focus on the problem of distributional policy evaluation with quantile parametrization, where the goal is to estimate the return distribution of a given policy using a finite-dimensional quantile representation. We consider a γ-discounted infinite-horizon Markov decision process (MDP) in the tabular setting, where the state space S and the action space A are finite. Given a policy π and quantile level $m ,$ let $\eta _ { m }$ denote the unique fixed point of the projected distributional Bellman equation $\pmb { \eta } = \pmb { \Pi } _ { m } \pmb { \mathcal { T } } ^ { \pi } \pmb { \eta }$ . Here, $\mathbf { I I } _ { m }$ represents the quantile projection operator and $\mathcal { T } ^ { \pi }$ represents the distributional Bellman operator.

Our goal is to estimate $\eta _ { m }$ when the underlying MDP is unknown. We assume that both the distribution of the random reward and the transition probability of the MDP are unknown. Given an ofline dataset consisting of $n | { \cal S } | | { \cal A } |$ samples collected from a generative model, we construct the empirical joint distribution $\widehat { Q } _ { s , a } ^ { ( n ) }$ of the reward and next state for each $( s , a ) \in S \times \mathcal { A }$ Following the certainty-equivalence principle [28, 31], we estimate $\eta _ { m }$ by the fixed point $\eta _ { m } ^ { ( n ) }$ of the empirical projected distributional Bellman operator $\mathbf { I I } _ { m } \hat { T } _ { n } ^ { \pi }$ , where $\widehat { T } _ { n } ^ { \pi }$ is constructed from $\{ \widehat { Q } _ { s , a } ^ { ( n ) } : ( s , a ) \in { \mathcal { S } } \times { \mathcal { A } } \}$ . We refer to $\pmb { \eta } _ { m } ^ { ( n ) }$ as the quantile fixed point estimator.

Our first contribution is a family of non-asymptotic bounds under supremum p-Wasserstein metrics. Under the assumption that the reward density is uniformly lower bounded, Theorem 3.1 gives a $O ( n ^ { - 1 / 2 } )$ bound simultaneously over every m and every $1 \leqslant p \leqslant \infty$ . Under the assumption that the reward density vanishes and are monotone at the boundary, Theorem 3.2 gives the same rate for $1 \leqslant p \leqslant 2$ . For $p > 2$ , Corollary 3.1 gives the rate $O ( m ^ { 1 / 2 - 1 / p } n ^ { - 1 / 2 } )$ and an alternative $O ( n ^ { - 1 / p } )$ bound independent of m for every finite $p > 2$ . Theorem 3.3 complements these rate statements with an m-free $W _ { \infty }$ consistency bound governed by the modulus of continuity of the reward distribution.

Our second contribution is an exact local statistical benchmark. At each fixed level of quantiles $m _ { \ i }$ , Theorem 3.4 identifies the local asymptotic minimax risk and shows that the quantile fixed point estimator attains it. Thus a Gaussian limit distribution characterizes the optimal first-order asymptotic risk of the problem.

Our third contribution is to determine when the resolution dependence can be attained. Proposition 3.1 gives a fully specified MDP with reward density uniformly away from zero, for which the local asymptotic minimax risk constants are bounded above and away from zero uniformly over m and p. Theorem 3.5 constructs an MDP with monotone and vanishing reward density at the boundary for which the local asymptotic minimax risk constants grow as $m ^ { 1 / 2 - 1 / p }$ up to logarithmic factors along a sequence of m for every $p > 2$ . Thus the polynomial resolution exponent in Corollary 3.1 cannot be reduced in general.

The main technical challenge is that the extreme quantiles can become increasingly unstable under vanishing reward densities at the boundary as m grows. Moreover, the projected Bellman operator is not globally contractive in supremum $W _ { p }$ metric for $p < \infty$ . To overcome these dificulties, we show that the averaging efect of the supremum $W _ { 2 }$ metric is strong enough to yield an $m -$ independent error bound and we prove a local $W _ { 2 }$ contraction of the projected Bellman operator. The local minimax result is obtained separately by a uniform linearization over shrinking Hellinger neighborhoods.

## 1.2 Related Works

Distributional reinforcement learning. Our operator framework uses distributional Bellman operators and finite-dimensional representations [1, 3]. Quantile representations underlie quantile regression methods [9], implicit quantile networks [8], fully parameterized quantile functions [35], and non-crossing quantile regression [37]. For categorical representations, Rowland et al. [23] analyzed convergence of categorical temporal-diference learning. Peng et al. [20] studied its statistical eficiency, and Peng et al. [21] established finite-sample results with linear function approximation. For quantile temporal-diference learning, Rowland et al. [25] established convergence, and Rowland et al. [24] studied its statistical benefits for expected-value estimation. Our object is instead a model-based empirical quantile Bellman fixed point, and our focus is the dependence of its statistical error on the quantile resolution.

Statistical analysis of reinforcement learning. Statistical inference in reinforcement learning has been developed for value functions and of-policy evaluation, including high-confidence procedures [32, 13], bootstrap methods [11], and inference in infinite-horizon settings [26]. Distributional and risk-sensitive evaluation has been studied through return-CDF estimation [6, 12] and distributional of-policy methods [34, 22]. In the generative-model setting, Zhang et al. [36] developed estimation and inference for unprojected return distributions. The companion work of Cheng et al. [7] supplies the regularity framework and canonical gradient for quantile Bellman fixed points. The present paper addresses resolution-uniform Wasserstein estimation and exact local expected-distance risk. In particular, our local model is the same general joint-law model used for eficiency there, rather than a model restricted to independent reward and successor-state perturbations.

The remainder of the paper is organized as follows. Section 2 introduces the MDP, the distributional metrics, the Bellman and projection operators, and the assumptions. Section 3 presents the estimator and all main results. Section 4 gives their proofs, with the main intermediate steps stated as lemmas. Section 5 concludes. Appendix A proves the lemmas from Section 4; Appendix B collects their technical ingredients.

## 2 Preliminaries

In this section, we introduce the necessary background for our work. We review the Markov decision process in Section 2.1, followed by metrics on the space of measures in Section 2.2. We introduce distributional Bellman operator and quantile projection operator in Section 2.3. Finally, in Section 2.4, we present the assumptions under which our main theoretical results are established.

## 2.1 Problem Setup

We consider a discounted Markov decision process (MDP) specified by the tuple $\mathcal { M } = \langle \boldsymbol { S } , \mathcal { A } , \mathcal { P } _ { R } , \boldsymbol { P } , \gamma \rangle$ 2 where S and A are finite state space and finite action space respectively, $\mathcal { P } _ { R } \colon \mathcal { S } \times \mathcal { A } \to \Delta ( [ 0 , 1 ] )$ is the distribution of rewards, $P \colon S \times A \to \Delta ( S )$ is the transition probability, and $\gamma \in ( 0 , 1 )$ is the discount factor. Here $\Delta ( \cdot )$ denotes the set of probability distributions over some set.

For a fixed policy $\pi \colon S \to \Delta ( { \mathcal { A } } )$ and an initial state $S _ { 0 } ~ = ~ s ~ \in ~ \mathcal { S }$ , a random trajectory $\{ ( S _ { t } , A _ { t } , R _ { t } ) \} _ { t = 0 } ^ { \infty }$ can be sampled from the MDP using the following procedure:

$$
\begin{array} { r l } & { \quad A _ { t } \mid S _ { t } \sim \pi ( \cdot \mid S _ { t } ) , } \\ & { \quad \quad R _ { t } \mid ( S _ { t } , A _ { t } ) \sim { \mathcal { P } } _ { R } ( \cdot \mid S _ { t } , A _ { t } ) , } \\ & { \quad \quad S _ { t + 1 } \mid ( S _ { t } , A _ { t } ) \sim P ( \cdot \mid S _ { t } , A _ { t } ) . } \end{array}
$$

Here the reward and next state are conditionally independent given $( S _ { t } , A _ { t } )$ , as in the model-based formulation.

The return of such a trajectory starting from state s is defined as the random variable

$$
G ^ { \pi } ( s ) : = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } ,
$$

which is bounded almost surely by $[ 0 , ( 1 - \gamma ) ^ { - 1 } ]$ . The value function $V ^ { \pi } ( s )$ is defined by the expected return $\mathbb { E } [ G ^ { \pi } ( s ) ]$ . We further denote by $\eta ^ { \pi } ( s ) \in \Delta ( [ 0 , ( 1 - \gamma ) ^ { - 1 } ] )$ the distribution of $G ^ { \pi } ( s )$ and denote $\pmb { \eta } ^ { \pi } = ( \eta ^ { \pi } ( s ) ) _ { s \in { \cal S } }$ for the collection of return distributions across all states.

## 2.2 Metrics on the Space of Measures

Denote the space of all probability distributions on $\mathbb { R }$ as $\mathcal { P }$ . For $\mu \in \mathcal { P }$ , the cumulative distribution function is defined as $F _ { \mu } ( x ) = \mu ( - \infty , x ]$ , and the quantile function is defined as $F _ { \mu } ^ { - 1 } ( \tau ) = \operatorname* { i n f } \{ x \colon F _ { \mu } ( x ) \geqslant \tau \}$ for $0 < \tau < 1$ . For $1 \leqslant p < \infty$ and $\mu , \nu \in { \mathcal { P } }$ , the p-Wasserstein metric

between $\mu$ and $\nu$ is defined as

$$
W _ { p } ( \mu , \nu ) = \left( \int _ { 0 } ^ { 1 } \left| F _ { \mu } ^ { - 1 } ( t ) - F _ { \nu } ^ { - 1 } ( t ) \right| ^ { p } \mathrm { d } t \right) ^ { 1 / p } ,
$$

and the $\infty \mathrm { - W }$ asserstein metric is defined as

$$
W _ { \infty } ( \mu , \nu ) = \underset { 0 < t < 1 } { \mathrm { e s s } } \left| F _ { \mu } ^ { - 1 } ( t ) - F _ { \nu } ^ { - 1 } ( t ) \right| .
$$

Suppose $\mu$ and $\nu$ have cumulative distribution functions $F _ { \mu }$ and $F _ { \nu }$ , respectively. In the case of $p = 1$ we have

$$
W _ { 1 } ( \mu , \nu ) = \int _ { \mathbb { R } } | F _ { \mu } ( x ) - F _ { \nu } ( x ) | \mathrm { d } x .
$$

The Kolmogorov–Smirnov metric (KS metric) is defined as

$$
\operatorname { K S } ( \mu , \nu ) = \operatorname* { s u p } _ { x \in \mathbb { R } } \left| F _ { \mu } ( x ) - F _ { \nu } ( x ) \right| .
$$

Moreover, for any extended metric $d \colon \mathcal { P } \times \mathcal { P }  [ 0 , \infty ]$ , we can define its supremum extension $\bar { d } \colon \mathcal { P } ^ { S } \times \mathcal { P } ^ { S }  [ 0 , \infty ]$ as

$$
\bar { d } ( \eta , \eta ^ { \prime } ) = \operatorname* { s u p } _ { s \in { \mathcal { S } } } d ( \eta ( s ) , \eta ^ { \prime } ( s ) ) ,
$$

which is an extended metric on $\mathcal { P } ^ { S }$

## 2.3 Distributional Bellman Operator and Quantile Projection Operator

A fundamental property of the value function is that it satisfies the Bellman equation. Letting $V ^ { \pi } : = ( V ^ { \pi } ( s ) ) _ { s \in { \cal S } }$ , we have for any $s \in \mathcal S$ ,

$$
\begin{array} { r l } & { V ^ { \pi } ( s ) = [ T ^ { \pi } V ^ { \pi } ] ( s ) } \\ & { \qquad : = { \mathbb E } _ { A \sim \pi ( \cdot \vert s \rangle , R \sim \mathcal { P } _ { R } ( \cdot \vert s , A ) } [ R ] + \gamma { \mathbb E } _ { A \sim \pi ( \cdot \vert s \rangle , S ^ { \prime } \sim P ( \cdot \vert s , A ) } [ V ^ { \pi } ( S ^ { \prime } ) ]  } \\ & { \qquad = \displaystyle \sum _ { a \in \mathcal { A } } \pi ( a \mid s ) \displaystyle \int _ { 0 } ^ { 1 } r \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) + \gamma \displaystyle \sum _ { a \in \mathcal { A } , s ^ { \prime } \in S } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) . } \end{array}\tag{1}
$$

The operator $T ^ { \pi } \colon { \mathbb { R } ^ { S } }  { \mathbb { R } ^ { S } }$ is referred to as the Bellman operator, and Equation (1) characterizes $V ^ { \pi }$ as its unique fixed point.

An analogous relationship holds for the return distributions $\eta ^ { \pi }$ , known as the distributional

Bellman equation. That is, for each $s \in \mathcal S$

$$
\begin{array} { r l } & { \eta ^ { \pi } ( s ) = \left[ \mathcal { T } ^ { \pi } \eta ^ { \pi } \right] ( s ) } \\ & { \qquad : = {  { \mathbb E } } _ { A \sim \pi ( \cdot | s ) , R \sim \mathcal { P } _ { R } ( \cdot | s , A ) , S ^ { \prime } \sim P ( \cdot | s , A ) } \left[ ( b _ { R , \gamma } ) _ { \# } \eta ^ { \pi } ( S ^ { \prime } ) \right] } \\ & { \qquad = \displaystyle \sum _ { a \in \mathcal { A } , s ^ { \prime } \in S } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) \displaystyle \int _ { 0 } ^ { 1 } ( b _ { r , \gamma } ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) . } \end{array}
$$

Here $b _ { r , \gamma } \colon \mathbb { R } \to  { \mathbb { R } }$ denotes the afine map $b _ { r , \gamma } ( x ) \ = \ r + \gamma x$ , and $g _ { \# } \mu$ is the pushforward of a measure $\mu$ under $^ { g , }$ defined by $g _ { \# } \mu ( B ) ~ = ~ \mu ( g ^ { - 1 } ( B ) )$ for all Borel sets $B$ . The integral $\int _ { 0 } ^ { 1 } { ( b _ { r , \gamma } ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r } \mid s , a )$ is defined in the sense that for any Borel set $B ,$

$$
\left[ \int _ { 0 } ^ { 1 } \left( b _ { r , \gamma } \right) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) \right] ( B ) = \int _ { 0 } ^ { 1 } \left[ ( b _ { r , \gamma } ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \right] ( B ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) .
$$

Recall that $\mathcal { P }$ is the space of all probability measures on $\mathbb { R } .$ , and $\mathcal { T } ^ { \pi } \colon \mathcal { P } ^ { S } \to \mathcal { P } ^ { S }$ is referred to as the distributional Bellman operator, of which the fixed point is the return distribution $\eta ^ { \pi }$

Because the exact distribution $\eta ^ { \pi }$ is infinite-dimensional and cannot be computed exactly, we approximate it using a quantile-parameterized distribution. The space of all quantile-parametrized probability distributions is defined as

$$
\mathcal { P } _ { m } : = \left\{ \nu _ { x } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { x _ { i } } \mid x = ( x _ { 1 } , \hdots , x _ { m } ) ^ { \top } \in \mathbb { R } ^ { m } , x _ { 1 } \leqslant \hdots \leqslant x _ { m } \right\} ,
$$

which is a mixture of Dirac measures and $m \in \mathbb { N }$ . We define the quantile projection operator $\Pi _ { m } \colon { \mathcal { P } } \to { \mathcal { P } } _ { m }$ as

$$
\Pi _ { m } \nu = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { F _ { \nu } ^ { - 1 } ( \tau _ { i } ) } ,
$$

where $\begin{array} { r } { \tau _ { i } = \frac { 2 i - 1 } { 2 m } } \end{array}$ . We lift $\Pi _ { m }$ to the product space $\mathcal { P } ^ { S }$ by defining $( \pmb { \Pi } _ { m } \pmb { \eta } ) ( s ) : = \Pi _ { m } \eta ( s )$ for any $\pmb { \eta } = ( \eta ( s ) ) _ { s \in \mathcal { S } } \in \mathcal { P } ^ { S }$ . The properties of $\mathcal { T } ^ { \pi }$ and $\Pi _ { m }$ are summarized in the following proposition.

Proposition 2.1 (3). The following statements hold:

$\mathcal { T } ^ { \pi }$ is γ-contractive under the $\overline { { W } } _ { p }$ metric for every $p \in [ 1 , \infty ]$ , namely for every η, $\pmb { \eta } ^ { \prime } \in \mathcal { P } ^ { S }$

$$
\overline { { { W } } } _ { p } ( { \cal T } ^ { \pi } \eta , { \cal T } ^ { \pi } \eta ^ { \prime } ) \leqslant \gamma \overline { { { W } } } _ { p } ( \eta , \eta ^ { \prime } ) ;
$$

$\mathbf { I I } _ { m }$ is non-expansive under the $\overline { W } _ { \infty }$ metric, namely $\overline { { { W } } } _ { \infty } ( \Pi _ { m } \eta , \Pi _ { m } \eta ^ { \prime } ) \leqslant \overline { { { W } } } _ { \infty } ( \eta , \eta ^ { \prime } )$ for every

$$
\eta , \eta ^ { \prime } \in \mathcal { P } ^ { S }
$$

It immediately follows from Proposition 2.1 that the quantile projected Bellman operator $\Pi _ { m } \mathcal { T } ^ { \pi }$ is a γ-contraction on the complete space of ordered m-atomic distributions supported in $[ 0 , ( 1 - \gamma ) ^ { - 1 } ]$ 2 equipped with $\overline { { W } } _ { \infty }$ . Hence, the quantile projected Bellman equation $\pmb { \eta } = \pmb { \Pi } _ { m } \pmb { \mathcal { T } } ^ { \pi } \pmb { \eta }$ admits a unique solution $\eta _ { m }$ . We define the quantile parameter $\pmb { \theta } _ { m } \in \mathbb { R } ^ { S \times [ m ] }$ by

$$
\eta _ { m } ( s ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta _ { m } ( s , i ) } ,
$$

with $\theta _ { m } ( s , 1 ) \leqslant \cdots \leqslant \theta _ { m } ( s , m )$ for every $s \in \mathcal S$

## 2.4 Main Assumptions

For every $( s , a ) \in \mathcal { S } \times \mathcal { A }$ , we make the following assumption about $\mathcal { P } _ { R } ( \cdot \mid s , a )$ , retaining the numbering of Cheng et al. [7].

Assumption 1 (Positive reward densities). For any $s \in \mathcal S$ and $a \in \mathcal { A } , \mathcal { P } _ { R } ( \cdot \mid s , a )$ is supported on r0, 1s and has a Lebesgue density $p _ { s , a }$ . Moreover, $p _ { s , a }$ is continuous on r0, 1s, and there is a positive constant $C _ { 0 }$ such that $0 < p _ { s , a } ( x ) \leqslant C _ { 0 }$ for every $x \in ( 0 , 1 )$ .

The following proposition shows that this assumption gives a well-behaved return density. Its statement is the same as in the companion paper; a proof is included in Appendix B.1.1.

Proposition 2.2. For every $s \in { \mathcal { S } } , \eta ^ { \pi } ( s )$ has a continuous Lebesgue density $p _ { \eta ^ { \pi } ( s ) }$ upper bounded by C<sub>0</sub> on $[ 0 , ( 1 - \gamma ) ^ { - 1 } ]$ . Moreover, $p _ { \eta ^ { \pi } ( s ) } ( x ) > 0$ for every $x \in ( 0 , ( 1 - \gamma ) ^ { - 1 } )$ , and $p _ { \eta ^ { \pi } ( s ) } ( 0 ) =$ $p _ { \eta ^ { \pi } ( s ) } ( ( 1 - \gamma ) ^ { - 1 } ) = 0$

We also impose a non-boundary condition on the population quantile parameters.

Assumption 2 (No boundary coincidence). For every $m \geqslant 1$ and $( s , i ) , ( s ^ { \prime } , j ) \in S \times [ m ]$

$$
\theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) \notin \{ 0 , 1 \} .
$$

This condition excludes configurations in which a Bellman-shifted quantile coincides with an endpoint of the reward support. Together with Assumption 1, it gives positive target density at each population midpoint quantile and diferentiability of the fixed-point CDF equation near the true parameter. We suppose that assumptions 1 and 2 hold throughout the paper.

For the non-asymptotic analysis we impose one of two alternative conditions. The first requires a positive lower reward density, as in the usual nondegenerate quantile-estimation setting.

Assumption 3 (Uniform lower density). There is a constant $c _ { 0 } > 0$ such that $p _ { s , a } ( x ) \geq c _ { 0 }$ for every $s \in { \mathcal { S } } , a \in { \mathcal { A } }$ , and $x \in [ 0 , 1 ]$

The second allows the reward density to vanish at the support endpoints, replacing the lower bound by a qualitative shape condition.

Assumption 4 (Regular endpoint vanishing). For every $s \in { \mathcal { S } }$ and $a \in \mathcal { A } , p _ { s , a } ( 0 ) = p _ { s , a } ( 1 ) = 0$ There is a common $\kappa > 0$ such that each $p _ { s , a }$ is nondecreasing on $[ 0 , \kappa ]$ and nonincreasing on $[ 1 - \kappa , 1 ]$

The endpoint monotonicity excludes oscillatory boundary behaviour but places no quantitative restriction on the speed at which the density vanishes. Beta densities with both shape parameters greater than one are examples. The two assumptions describe complementary density regimes.

Under Assumption 4, zero extension makes each reward density continuous on $\mathbb { R } .$ . Since there are finitely many state–action pairs, these extensions have the common modulus

$$
\varpi _ { R } ( h ) : = \operatorname* { m a x } _ { s , a } \ \operatorname* { s u p } _ { x , y \in \mathbb { R } } \ | p _ { s , a } ( x ) - p _ { s , a } ( y ) | , \qquad \operatorname* { l i m } _ { h \downarrow 0 } \varpi _ { R } ( h ) = 0 .\tag{2}
$$

The assumptions above apply to the fixed MDP at which the finite-sample bounds are evaluated. In the local asymptotic minimax analysis, they apply only to the centre of the local neighbourhood.

Notation and constants. Write $B = ( 1 - \gamma ) ^ { - 1 }$ and $I = [ 0 , B ]$ . Bare $C , c , C ^ { \prime } , \ldots$ . denote positive universal numerical proof constants, unless a symbol has explicitly been introduced as a hypothesis parameter. Symbols $C _ { \mathcal { M } } , c _ { \mathcal { M } } , C _ { j } ( \mathcal { M } )$ denote finite instance-dependent constants, with the fixed policy included in the instance. They may depend on the reward laws, their continuity moduli, positive transition probabilities, the discount, and the finite state–action spaces, but not on $m , n , \delta ,$ or p unless such dependence is displayed. Constants such as $C _ { m , c , Q _ { 0 } }$ display their additional fixedresolution and local-radius dependence. When only $| S | , | A |$ , and B are involved, those dependencies are shown explicitly. Deterministic inversion constants are labelled by their geometry parameters. Repeated constant labels in diferent results need not have the same value unless explicitly inherited. Construction constants used for the single hard instance are distinguished from constants uniform over a model class.

## 3 Main Results

In this section, we establish non-asymptotic Wasserstein guarantees for the quantile fixed point estimator and identify their local statistical limits. We first define the paired empirical-joint estimator, then study its resolution dependence under the two density regimes. Finally, we characterize the exact local asymptotic minimax risk and exhibit fixed MDPs that attain the relevant resolution orders.

## 3.1 The Quantile Fixed Point Estimator

We assume access to a generative model for the unknown MDP. For each $( s , a ) \in S \times \mathcal { A }$ , we observe

$$
X _ { k , ( s , a ) } = ( R _ { k } ^ { ( s , a ) } , S _ { k } ^ { \prime ( s , a ) } ) { \overset { \mathrm { i i d } } { \sim } } Q _ { 0 , s , a } , \qquad k \in [ n ] , \qquad Q _ { 0 , s , a } = { \mathcal { P } } _ { R } ( \cdot \mid s , a ) \otimes P ( \cdot \mid s , a ) .
$$

The samples are independent for diferent $( s , a )$ . Thus n is the number of observations per pair, and the total sample size is $n | { \cal S } | | { \cal A } |$

For any collection of joint reward–next-state laws $Q = ( Q _ { s , a } ) _ { s , a }$ , define

$$
[ { \cal T } _ { Q } ^ { \pi } \eta ] ( s ) = \sum _ { a } \pi ( a \mid s ) \int ( b _ { r , \gamma } ) _ { \# } \eta ( s ^ { \prime } ) Q _ { s , a } ( \mathrm { d } ( r , s ^ { \prime } ) ) .\tag{3}
$$

At $Q = Q _ { 0 }$ , this is the population operator $\mathcal { T } ^ { \pi }$ from Section 2. The empirical joint laws and empirical operator are

$$
\widehat { Q } _ { s , a } ^ { ( n ) } = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \delta _ { ( R _ { k } ^ { ( s , a ) } , S _ { k } ^ { \prime ( s , a ) } ) } , \qquad \widehat { T } _ { n } ^ { \pi } = \mathcal { T } _ { \widehat { Q } ^ { ( n ) } } ^ { \pi } .\tag{4}
$$

We define the quantile fixed point estimator by

$$
\eta _ { m } ^ { ( n ) } = \Pi _ { m } { \widehat T } _ { n } ^ { \pi } \eta _ { m } ^ { ( n ) } , \qquad \eta _ { m } ^ { ( n ) } ( s ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta _ { m } ^ { ( n ) } ( s , i ) } ,\tag{5}
$$

with the coordinates ordered within each state. In particular, the empirical joint law in (4) is not replaced by the product of its empirical marginals. The definition remains meaningful even though the empirical law is discrete. The projected empirical operator has a unique fixed point, obtainable by iterating quantile dynamic programming; the proof of its contraction and measurability is given in Lemma B.1.

Now we define some notations used in subsequent analysis. Let $\Theta _ { m }$ be the closed set of arrays

in $I ^ { S \times [ m ] }$ ordered within each state. For $\pmb { \theta } \in \Theta _ { m } .$ write

$$
\eta _ { \theta } ( s ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta ( s , i ) } , \qquad \eta _ { \theta } = ( \eta _ { \theta } ( s ) ) _ { s \in \mathcal { S } } .
$$

For a joint law $Q ,$ denote the ordered parameter of the fixed point of $\Pi _ { m } \mathcal { T } _ { Q } ^ { \pi }$ by $\pmb { \theta } _ { m } ( Q )$ and its distribution vector by $\pmb { \eta } _ { m } ( Q )$ . We abbreviate the centre quantities by $\pmb { \theta } _ { m } = \pmb { \theta } _ { m } ( \boldsymbol { Q } _ { 0 } )$ and $\eta _ { m } = \eta _ { m } ( Q _ { 0 } )$ . For later use, set

$$
\begin{array} { r l } & { F _ { Q , s , \theta } ( x ) = F _ { [ \mathcal { T } _ { Q } ^ { \pi } \eta _ { \theta } ] ( s ) } ( x ) , \qquad F _ { s , \theta } = F _ { Q _ { 0 } , s , \theta } , } \\ & { \quad \widehat { F } _ { s , \theta } ^ { ( n ) } ( x ) = F _ { [ \widehat { \mathcal { T } } _ { n } ^ { \pi } \eta _ { \theta } ] ( s ) } ( x ) , \qquad F _ { m , s } = F _ { s , \theta _ { m } } . } \end{array}\tag{6}
$$

The population and empirical maps on quantile parameters are

$$
[ \Phi _ { m } ( \pmb { \theta } ) ] _ { s , i } = F _ { s , \pmb { \theta } } ^ { - 1 } ( \tau _ { i } ) , \qquad [ \widehat { \Phi } _ { m , n } ( \pmb { \theta } ) ] _ { s , i } = ( \widehat { F } _ { s , \pmb { \theta } } ^ { ( n ) } ) ^ { - 1 } ( \tau _ { i } ) .\tag{7}
$$

Here and throughout, inverse CDFs are lower generalized inverses.

For $\boldsymbol { z } \in \mathbb { R } ^ { \boldsymbol { S \times } [ m ] }$ , define

$$
\ell _ { m , p } ( z ) = \operatorname* { m a x } _ { s } \left( \frac { 1 } { m } \sum _ { i = 1 } ^ { m } | z ( s , i ) | ^ { p } \right) ^ { 1 / p } \quad ( 1 \leqslant p < \infty ) , \qquad \ell _ { m , \infty } ( z ) = \operatorname* { m a x } _ { s , i } | z ( s , i ) | .
$$

For ordered arrays, the one-dimensional quantile formula gives the exact identity

$$
\overline { { W } } _ { p } ( \eta _ { \theta } , \eta _ { \vartheta } ) = \ell _ { m , p } ( \theta - \vartheta ) , \qquad 1 \leqslant p \leqslant \infty .\tag{8}
$$

Our loss is $\overline { { W } } _ { p } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } )$ , so approximation of $\eta ^ { \pi }$ is not part of the estimation target. We call a bound resolution-independent, or m-free, when its constants and sample-size requirements do not depend on m and it contains no additive discretization remainder. The bounds below are stronger than separate fixed-m statements: they hold on one event for every resolution constructed from the same data.

## 3.2 Uniformly Positive Reward Densities

We first consider Assumption 3. The next theorem shows that increasing the quantile resolution does not worsen the root-n estimation order, even in the strongest Wasserstein metric.

Theorem 3.1. Under the standing assumptions and Assumption 3, there is an instance-dependent constant $0 < C _ { 1 } ( \mathcal { M } ) < \infty$ such that for every $\delta \in ( 0 , 1 )$ and every $n \geqslant 1$ ，

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { \infty } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant C _ { 1 } ( \mathcal { M } ) \sqrt { \frac { \log ( 4 | S | / \delta ) } { n } } \right\} \geqslant 1 - \delta .\tag{9}
$$

On the same event, the same bound holds simultaneously for every $1 \leqslant p \leqslant \infty$ , since $\overline { { W } } _ { p } \leqslant \overline { { W } } _ { \infty }$ Moreover,

$$
\mathbb { E } \operatorname* { s u p } _ { m \geq 1 } \overline { { W } } _ { p } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant \frac { C _ { 2 } ( \mathcal { M } ) } { \sqrt { n } } , \qquad 1 \leqslant p \leqslant \infty ,
$$

for an instance-dependent constant $C _ { 2 } ( \mathcal { M } ) < \infty$

The proof separates reward-distribution error from transition-weight error before using the global supremum-norm contraction. This distinction avoids replacing all quantile sensitivities by the sensitivity of the outermost coordinate. The complete proof is in Section 4.1.

## 3.3 Endpoint-Vanishing Reward Densities

Assumption 4 removes the lower density bound and allows arbitrary boundary decay subject to monotonicity. The root-n rate remains independent of m under $\overline { { W } } _ { 2 }$ metric.

Theorem 3.2. Under the standing assumptions and Assumption $^ { 4 , }$ there are instance-dependent constants $C _ { 1 } ( \mathcal { M } ) , C _ { 2 } ( \mathcal { M } ) > 0$ such that, for every n ě 1 and $\delta \in ( 0 , 1 )$

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant C _ { 1 } ( \mathcal { M } ) \sqrt { \frac { \log ( 4 | S | / \delta ) } { n } } \right\} \geqslant 1 - \delta .\tag{10}
$$

On this event the same bound holds simultaneously for every m $\geqslant ~ 1$ and every $1 \leqslant p \leqslant 2$ Consequently,

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \right] \leqslant \frac { C _ { 2 } ( \mathcal { M } ) } { \sqrt { n } } .\tag{11}
$$

The same constant also satisfies

$$
\mathbb { E } \left[ \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \right\} ^ { 2 } \right] \leqslant \frac { C _ { 2 } ( \mathcal { M } ) ^ { 2 } } { n } .\tag{12}
$$

The constants are independent of $m , n , \delta ,$ , and $p$

The population map $\mathbf { \Pi } \pi ^ { \pi }$ need not be globally contractive under the $\overline { { W } } _ { 2 }$ metric. We prove

that it is contractive on a common neighborhood of $\eta ^ { \pi }$ for all suficiently large $m ;$ the finitely many smaller resolutions are handled separately. Section 4.2 gives the proof.

We have the following corollary from Theorem 3.2.

Corollary 3.1. Under the assumptions of Theorem 3.2, for every $n \geqslant 1$ and $\delta \in ( 0 , 1 )$ , on the event in (10), which has probability at least $1 - \delta$ , the following bounds hold simultaneously for every m $\geqslant 1$ and every $2 \leqslant p \leqslant \infty$

$$
\overline { { W } } _ { p } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant C _ { 1 } ( \mathcal { M } ) m ^ { 1 / 2 - 1 / p } \sqrt { \frac { \log ( 4 | \pmb { S } | / \delta ) } { n } }\tag{13}
$$

$$
\operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { p } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant C _ { 1 } ( \mathcal { M } ) ^ { 2 / p } ( 1 - \gamma ) ^ { - ( 1 - 2 / p ) } \left( \frac { \log ( 4 | S | / \delta ) } { n } \right) ^ { 1 / p }\tag{14}
$$

where $1 / \infty = 0$ and $C _ { 1 } ( \mathcal { M } )$ is the constant defined in Theorem 3.2. At $p = \infty , ( 1 4 )$ is understood as the support bound $\begin{array} { r } { \operatorname* { s u p } _ { m } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant B } \end{array}$

However, Equation (14) in Corollary 3.1 does not give consistency in $W _ { \infty }$ . Nevertheless, Assumption 4 still yields an m-free $W _ { \infty }$ consistency bound, at a rate determined by the continuity modulus of the reward quantile functions. Write $F _ { s , a }$ for the CDF of $\mathcal { P } _ { R } ( \cdot \mid s , a )$ , extend its inverse to the endpoints by $F _ { s , a } ^ { - 1 } ( 0 ) = 0$ and $F _ { s , a } ^ { - 1 } ( 1 ) = 1$ , and define

$$
\omega _ { R } ( h ) : = \operatorname* { m a x } _ { s , a } \operatorname* { s u p } _ { \boldsymbol { u } , \boldsymbol { v } \in [ 0 , 1 ] \atop | \boldsymbol { u - v } | \leqslant h } \left| F _ { s , a } ^ { - 1 } ( \boldsymbol { u } ) - F _ { s , a } ^ { - 1 } ( \boldsymbol { v } ) \right| , \qquad 0 \leqslant h \leqslant 1 .\tag{15}
$$

By Assumption 1, each reward CDF is continuous and strictly increasing on r0, 1s, so $\omega _ { R } ( h ) \to 0$ as $h \downarrow 0$

Theorem 3.3. Under the standing assumptions and Assumption $^ { 4 , }$ there is an instance-dependent constant $K _ { 3 } ( \mathcal { M } ) \in ( 0 , \infty )$ such that, for every $n \geqslant 1$ and $\delta \in ( 0 , 1 )$ , with

$$
h _ { n , \delta } : = \operatorname* { m i n } \left\{ 1 , \frac { 1 6 } { p _ { * } } \sqrt { \frac { \log ( 4 | \mathcal { S } | / \delta ) } { n } } \right\} ,\tag{16}
$$

where

$$
p _ { * } = \operatorname* { m i n } _ { ( s , a , s ^ { \prime } ) : P ( s ^ { \prime } | s , a ) > 0 } P ( s ^ { \prime } \mid s , a ) ,
$$

we have

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { \infty } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant K _ { 3 } ( \mathcal { M } ) \big [ \omega _ { R } ( h _ { n , \delta } ) + h _ { n , \delta } \big ] \right\} \geqslant 1 - \delta .\tag{17}
$$

Theorem 3.3 shows that, even when the reward density vanishes at the endpoints, the empirical quantile fixed point remains uniformly consistent over all resolutions $m$ in the $\overline { W } _ { \infty }$ metric. The convergence rate is governed by the modulus of continuity of the reward quantile functions, evaluated at the statistical scale $h _ { n , \delta }$

## 3.4 Local Asymptotic Minimaxity

The preceding statements are upper bounds for a specified estimator. To assess their sharpness, we identify the optimal first-order expected loss near a fixed MDP. We use the general statistical model

$$
\mathcal { Q } = \left\{ Q = ( Q _ { s , a } ) _ { ( s , a ) \in S \times A } : Q _ { s , a } \in \Delta ( [ 0 , 1 ] \times S ) , \quad Q _ { s , a } \ll \lambda \otimes \# S \right\} ,\tag{18}
$$

where λ is Lebesgue measure on r0, 1s and $\# S$ is counting measure on ${ \mathcal { S } } .$ This is the model used for eficiency in Cheng et al. [7, Theorem 3.3]. Under $Q ,$ the sampling experiment is $\mathbb { P } _ { Q } ^ { ( n ) } = \bigotimes _ { s , a } Q _ { s , a } ^ { \otimes n }$ and the target is $\pmb { \eta } _ { m } ( Q )$ defined in Section 3.1.

The local decision problem. Define the block Hellinger distance by

$$
d _ { H } ( Q , Q _ { 0 } ) ^ { 2 } = \sum _ { s , a } \int \big ( \sqrt { \mathrm { d } Q _ { s , a } } - \sqrt { \mathrm { d } Q _ { 0 , s , a } } \big ) ^ { 2 } , \qquad \mathcal { N } _ { n } ( c ; Q _ { 0 } ) = \{ Q \in Q : d _ { H } ( Q , Q _ { 0 } ) \leqslant c / \sqrt { n } \} .\tag{19}
$$

At fixed $m _ { \colon }$ , the standardized local minimax risk is

$$
r _ { n , m , p } ( c ; Q _ { 0 } ) = \operatorname* { i n f } _ { \hat { \eta } _ { n } } \operatorname* { s u p } _ { Q \in \mathcal { N } _ { n } ( c ; Q _ { 0 } ) } \mathbb { E } _ { Q } \big [ { \sqrt { n } } \overline { { W } } _ { p } \big ( \hat { \eta } _ { n } , \eta _ { m } ( Q ) \big ) \big ] .\tag{20}
$$

The infimum ranges over all possibly randomized measurable estimators taking values in $\Delta ( I ) ^ { s }$ ; they may depend on the fixed centre, radius, and resolution. Define

$$
\mathbb { R } _ { m , p } ^ { \mathrm { l o c } } ( Q _ { 0 } ) = \operatorname* { l i m } _ { c  \infty } \operatorname* { l i m } _ { n  \infty } \boldsymbol { \mathrm { f } } r _ { n , m , p } ( c ; Q _ { 0 } ) , \qquad \overline { { \mathbb { R } } } _ { m , p } ^ { \mathrm { l o c } } ( Q _ { 0 } ) = \operatorname* { l i m } _ { c  \infty } \operatorname* { l i m } _ { n  \infty } \operatorname* { s u p } _ { r _ { n , m , p } ( c ; Q _ { 0 } ) } .\tag{21}
$$

The outer limits exist by monotonicity in c.

The eficient Gaussian loss. In the following definitions all population quantities are evaluated at $Q _ { 0 }$ , and $d = | S | m$ . Write

$$
y _ { m , s , i } ( r , s ^ { \prime } ) = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbb { 1 } \{ r + \gamma \theta _ { m } ( s ^ { \prime } , j ) \leqslant \theta _ { m } ( s , i ) \} .
$$

For a sampling block $b = ( s , a )$ , define a vector $Y _ { m , b } \in \mathbb { R } ^ { d }$ by

$$
[ Y _ { m , ( s , a ) } ( r , s ^ { \prime } ) ] _ { u , i } = \mathbb { 1 } \{ u = s \} \pi ( a \mid s ) y _ { m , s , i } ( r , s ^ { \prime } ) , \qquad \xi _ { m , b } = Y _ { m , b } - \mathbb { E } _ { Q _ { 0 , b } } Y _ { m , b } .
$$

Let $H _ { Q } ( \pmb \theta )$ be the vector of population Bellman-target CDFs evaluated at the corresponding coordinates of θ. Explicitly,

$$
[ \pmb { H } _ { Q } ( \pmb { \theta } ) ] _ { s , i } = F _ { Q , s , \pmb { \theta } } ( \pmb { \theta } ( s , i ) ) , \qquad [ \pmb { H } _ { n } ( \pmb { \theta } ) ] _ { s , i } = \widehat { F } _ { s , \pmb { \theta } } ^ { ( n ) } ( \pmb { \theta } ( s , i ) ) .
$$

Thus $\pmb { H } _ { Q _ { 0 } } ( \pmb { \theta } _ { m } ) = \pmb { \tau } _ { m }$ , where $[ \tau _ { m } ] _ { s , i } = \tau _ { i }$ . Writing $f _ { m , s }$ for the Bellman-target density at the centre, its Jacobian is

$$
G _ { m } = \nabla _ { \theta } H _ { Q _ { 0 } } ( \theta _ { m } ) = D _ { m } - \gamma W _ { m } = D _ { m } ( I - \gamma K _ { m } ) , \qquad K _ { m } = D _ { m } ^ { - 1 } W _ { m } ,\tag{22}
$$

where

$$
\begin{array} { l } { { { [ } W _ { m } ] _ { ( s , i ) , ( s ^ { \prime } , j ) } = \displaystyle \frac { 1 } { m } \sum _ { a } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) p _ { s , a } \bigl ( \theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) \bigr ) , } } \\ { { { } } } \\ { { { [ } D _ { m } ] _ { ( s , i ) , ( s , i ) } = f _ { m , s } ( \theta _ { m } ( s , i ) ) = \displaystyle \sum _ { s ^ { \prime } , j } [ W _ { m } ] _ { ( s , i ) , ( s ^ { \prime } , j ) } . } } \end{array}
$$

The diagonal is positive, $K _ { m } \geqslant 0$ , and $K _ { m } \mathbf { 1 } = \mathbf { 1 }$ , so $G _ { m }$ is invertible. Define

$$
\begin{array} { r } { \pmb { \Sigma } _ { m } = \displaystyle { \sum _ { b } ^ { } } \mathbb { E } _ { Q _ { 0 , b } } [ \pmb { \xi } _ { m , b } \pmb { \xi } _ { m , b } ^ { \top } ] , \qquad V _ { m } = \pmb { G } _ { m } ^ { - 1 } \pmb { \Sigma } _ { m } \pmb { G } _ { m } ^ { - \top } . } \end{array}\tag{23}
$$

Equivalently the canonical gradient has block components

$$
\phi _ { m , b } = - \pmb { G } _ { m } ^ { - 1 } \pmb { \xi } _ { m , b } , \qquad \pmb { V } _ { m } = \sum _ { b } \mathbb { E } _ { \pmb { Q } _ { 0 , b } } [ \phi _ { m , b } \phi _ { m , b } ^ { \top } ] .\tag{24}
$$

These are the matrices and influence functions of Cheng et al. [7, Theorem 3.3 and Lemma 4.5]. In the model (18), the tangent space is $\textcircled { + } _ { b } L _ { 0 } ^ { 2 } ( Q _ { 0 , b } )$ . Using the norm $\ell _ { m , p }$ defined in Section 3.1, define the instance-dependent complexity by

$$
\mathfrak { C } _ { m , p } ( Q _ { 0 } ) : = \mathbb { E } \ell _ { m , p } ( Z _ { m } ) , \qquad Z _ { m } \sim \mathcal { N } ( 0 , V _ { m } ) .\tag{25}
$$

The covariance may be singular; neither the definition nor the theorem requires a positive-definite covariance.

Theorem 3.4. Fix $m \geqslant 1 , 1 \leqslant p \leqslant \infty$ , and a centre $Q _ { 0 }$ satisfying the standing assumptions. Then

$$
\underline { { \mathcal { R } } } _ { m , p } ^ { \mathrm { l o c } } ( Q _ { 0 } ) = \overline { { \mathcal { R } } } _ { m , p } ^ { \mathrm { l o c } } ( Q _ { 0 } ) = \mathfrak { C } _ { m , p } ( Q _ { 0 } ) .\tag{26}
$$

The paired empirical-joint fixed-point estimator attains this constant. More precisely, for every fixed $c < \infty$ ，

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { s u p } _ { Q \in \mathcal { N } _ { n } ( c ; Q _ { 0 } ) } \left. \mathbb { E } _ { Q } \Big [ \sqrt { n } \overline { { W } } _ { p } \big ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ( Q ) \big ) \Big ] - \mathfrak { C } _ { m , p } ( Q _ { 0 } ) \right. = 0 .\tag{27}
$$

The alternatives in these suprema are all joint laws in the Hellinger ball, not merely a selected parametric family.

Thus the optimal local first-order expected-distance scale is ${ \mathfrak { C } } _ { m , p } ( Q _ { 0 } ) / { \sqrt { n } }$ . The proof in Section 4.5 combines uniform linearization and uniform integrability on the full balls with a finitedimensional least-favourable joint-law submodel and a Gaussian decision argument. A centre-wise CLT alone would not imply (27) or the all-estimator assertion.

## 3.5 Fixed Instances and Sharp Resolution Dependence

The local limit in Theorem 3.4 is taken at each fixed m. We now compare its coeficient across resolutions. The following one-state example exhibits a resolution-independent local risk.

Proposition 3.1. Consider one state, one action, a self-transition, $R \sim \mathrm { U n i f } [ 0 , 1 ]$ , and $\gamma = e ^ { - 1 }$ This fixed MDP satisfies Assumptions 1–3 for every $m ,$ including the non-boundary condition. For every m $\geqslant 1 , 1 \leqslant p \leqslant \infty$ , we have ${ \mathfrak { C } } _ { m , p } = 1$

Together with Theorem 3.4, this instance shows that the root-n order in Theorem 3.1 cannot be improved in general. The next result gives the contrasting endpoint-vanishing instance.

Theorem 3.5. There is one fixed two-state, one-action MDP $Q _ { * }$ with $\gamma \ = \ 1 / 2$ , transitions $P ( t \mid s ) = P ( t \mid t ) = 1$ , and rewards satisfying Assumptions 1, ${ \mathcal { Q } } ,$ and $^ { 4 , }$ together with a sequence $m _ { j } \to \infty$ , such that

$$
\frac { c _ { * } m _ { j } ^ { 1 / 2 - 1 / p } } { ( \log m _ { j } ) ^ { K _ { * } } } \leqslant \mathfrak { C } _ { m _ { j } , p } ( Q _ { * } ) \leqslant C _ { * } m _ { j } ^ { 1 / 2 - 1 / p } , \qquad 2 < p \leqslant \infty ,\tag{28}
$$

for all suficiently large $j$ . Here $c _ { * } , C _ { * } , K _ { * }$ are constants determined by the $M D P Q _ { * }$ , which are independent of $j , m _ { j } , n$ , and $p .$ For this same instance $\mathfrak { C } _ { m , p } ( Q _ { * } ) = 1$ for $1 \leqslant p \leqslant 2$

This result shows that for $p > 2$ , the polynomial resolution dependence $m ^ { 1 / 2 - 1 / p }$ in Corollary 3.1 cannot be improved in general, up to logarithmic factors. Meanwhile, the same instance has a nonzero, resolution-independent local coeficient for $p \leqslant 2$

## 4 Proof Outlines

We give the proofs of the results in Section 3.

## 4.1 Analysis under the Uniform Lower Density Condition

The proof of Theorem 3.1 consists of two steps. We first establish a concentration inequality for the empirical Bellman operator evaluated at the population fixed point $\eta _ { m }$ . This yields a uniform bound on the perturbation of the corresponding quantile functions. We then combine this estimate with the γ-contraction property of $\Pi _ { m } \widehat { T } _ { n } ^ { \pi }$ under the $\overline { { W } } _ { \infty }$ metric to obtain a fixed-point perturbation bound, which implies Theorem 3.1. The key ingredient is the following concentration result, the proof of which is given in Appendix A.1.

Lemma 4.1. Under the standing assumptions and Assumption 3, define

$$
\Delta _ { n } : = \operatorname* { s u p } _ { m \geq 1 } \overline { { W } } _ { \infty } \left( \mathbf { \Pi } \Pi _ { m } \hat { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } , \mathbf { \Pi } \Pi _ { m } \mathcal { T } ^ { \pi } \pmb { \eta } _ { m } \right) .
$$

There is an instance-dependent constant $K _ { \mathcal { M } } > 0$ such that, for every n $\geqslant 1$ and $\delta \in ( 0 , 1 )$

$$
\mathbb { P } \left[ \Delta _ { n } > K _ { \mathcal { M } } \sqrt { \frac { \log ( 4 | S | / \delta ) } { n } } \right] \leqslant \delta .
$$

Proof of Theorem 3.1. Since $\Pi _ { m } \widehat { T } _ { n } ^ { \pi }$ is a γ-contraction under the $\overline { W } _ { \infty }$ metric for every m, we have

$$
\begin{array} { r l } & { \left. \theta _ { m } ^ { ( n ) } - \theta _ { m } \right. _ { \infty } \leqslant \overline { { W } } _ { \infty } \left( \mathbf { { I } } _ { m } \widehat { T } _ { n } ^ { \pi } \eta _ { m } ^ { ( n ) } , \mathbf { { I } } _ { m } \widehat { T } _ { n } ^ { \pi } \eta _ { m } \right) + \overline { { W } } _ { \infty } \left( \mathbf { { I } } _ { m } \widehat { T } _ { n } ^ { \pi } \eta _ { m } , \mathbf { { I } } _ { m } T ^ { \pi } \eta _ { m } \right) } \\ & { \qquad \leqslant \gamma \left. \theta _ { m } ^ { ( n ) } - \theta _ { m } \right. _ { \infty } + \Delta _ { n } . } \end{array}
$$

Rearranging and using (8) gives

$$
\operatorname* { s u p } _ { m \geq 1 } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant \frac { \Delta _ { n } } { 1 - \gamma } .
$$

Apply Lemma 4.1, absorbing $( 1 - \gamma ) ^ { - 1 }$ into the instance-dependent constants, to obtain (9) and the expectation bound. Finally, $\overline { { W } } _ { p } \leqslant \overline { { W } } _ { \infty }$ for every 1 $\leqslant p \leqslant \infty$ , so the same event and expectation bound apply to all orders. □

## 4.2 Analysis under Endpoint Vanishing

When the reward densities vanish at the endpoints, extreme quantiles can be highly unstable because the Bellman-target density there may be arbitrarily small. Nevertheless, the $\overline { { W } } _ { 2 }$ loss averages the squared errors over all quantile coordinates, and our goal is to show that this averaging prevents the overall error from deteriorating with the resolution.

We first show that both the population and empirical fixed points remain close to the true return distribution. In this region, the empirical Bellman CDF fluctuates less where the target density is small, and this compensates for the increased sensitivity of the corresponding quantiles. As a result, the discrepancy between the empirical and population quantile Bellman maps has an m-independent average squared quantile error. We then show that the population quantile Bellman map is locally contractive in this averaged quadratic metric, uniformly for all suficiently large m. The finitely many smaller resolutions are treated separately.

In this subsection, write $\| z \| _ { 2 , m } = \ell _ { m , 2 } ( z )$ . Under Assumption 4, the density of the Bellman target at an input θ is

$$
f _ { s , \theta } ( x ) = \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) } { m } p _ { s , a } ( x - \gamma \theta ( s ^ { \prime } , j ) ) ,\tag{29}
$$

where each reward density is extended by zero. Define the neighborhood

$$
\mathcal { U } _ { m } ( \rho ) = \{ \pmb { \theta } \in \Theta _ { m } : \overline { { W } } _ { 1 } ( \eta \pmb { \theta } , \pmb { \eta } ^ { \pi } ) \leqslant \rho \} .\tag{30}
$$

For localization, introduce the joint sub-CDFs and their empirical counterparts,

$$
J _ { s , a , s ^ { \prime } } ( x ) = P ( s ^ { \prime } \mid s , a ) F _ { s , a } ( x ) ,
$$

$$
\widehat { J } _ { s , a , s ^ { \prime } } ( x ) = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } \mathbb { 1 } \{ S _ { k } ^ { \prime ( s , a ) } = s ^ { \prime } , ~ R _ { k } ^ { ( s , a ) } \leqslant x \} ,
$$

and set

$$
D _ { n } = \operatorname* { m a x } _ { s } \sum _ { a } \pi ( a \mid s ) \sum _ { s ^ { \prime } } \| \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } \| _ { \infty } .\tag{31}
$$

The following lemma, proved in Appendix A.2, shows that this random variable controls the empirical Bellman CDF for every resolution and every input.

Lemma 4.2. Under Assumption 4, fix $0 < \ell < \mathrm { m i n } \{ \kappa , B / 2 \}$ . There are instance-dependent $\rho , c _ { \mathcal { M } , \ell } , \varepsilon _ { 0 } > 0$ and $M _ { 1 } < \infty$ with the following properties. For every m $\geqslant 1$ and $\pmb { \theta } \in \mathcal { U } _ { m } ( \rho )$ , the continuous density $f _ { s , \theta }$ satisfies

$$
\operatorname* { i n f } _ { s \in S } \operatorname* { i n f } _ { x \in [ \ell , B - \ell ] } f _ { s , \theta } ( x ) \geqslant c _ { \mathcal { M } , \ell } .
$$

Moreover, on $\{ D _ { n } \leqslant \varepsilon _ { 0 } \}$ R

$$
\pmb { \theta } _ { m } , \pmb { \theta } _ { m } ^ { ( n ) } \in \mathcal { U } _ { m } ( \rho ) \qquad \mathrm { ~ f o r ~ } e v e r y ~ m \geqslant M _ { 1 } .
$$

Finally, for each fixed $M < \infty$ , there exists an instance-dependent $C _ { M , M } < \infty$ such that

$$
\operatorname* { m a x } _ { 1 \leqslant m \leqslant M } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant C _ { M , \mathcal { M } } D _ { n } .
$$

The next lemma supplies the fixed-point stability needed for the $W _ { 2 }$ argument and it is proved in Appendix A.2. Unlike the global $W _ { \infty }$ contraction, it uses the density geometry of the common neighborhood.

Lemma 4.3. Under the standing assumptions and Assumption $^ { 4 , }$ there exist instance-dependent $\rho > 0 , M _ { 0 } < \infty$ , and $\lambda _ { 2 } < 1$ such that for all m $\geqslant M _ { 0 }$ and all θ, $\vartheta \in \mathcal { U } _ { m } ( \rho )$

$$
\begin{array} { r } { \| \Phi _ { m } ( \pmb { \theta } ) - \Phi _ { m } ( \pmb { \vartheta } ) \| _ { 2 , m } \leqslant \lambda _ { 2 } \| \pmb { \theta } - \pmb { \vartheta } \| _ { 2 , m } . } \end{array}
$$

The neighbourhood may be chosen together with Lemma $4 . 2 .$

To convert a uniform CDF bound into a bound on the average squared error of the quantiles, we use the following deterministic result, which is proved in Appendix A.2.

Lemma 4.4. Let F be a CDF supported in r0, Bs with a continuous density f satisfying $f ( 0 ) =$ $f ( B ) = 0$ . For some $0 < \ell < B / 2$ and $c _ { \mathrm { i n t } } , C _ { 0 } > 0$ , assume that $f$ is nondecreasing on $[ 0 , \ell ]$ nonincreasing on $[ B - \ell , B ]$ , at least $c _ { \mathrm { i n t } }$ on $[ \ell , B - \ell ]$ , and at most $C _ { 0 }$ everywhere. Let G be any CDF supported on r0, Bs. $I f z , \varepsilon \geqslant 0$ and

$$
| G ( x ) - F ( x ) | \leqslant z { \sqrt { f ( x ) } } + \varepsilon \qquad { f o r \ a l l \ x \in [ 0 , B ] } ,\tag{32}
$$

then

$$
{ \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } \left| { \cal G } ^ { - 1 } ( \tau _ { i } ) - { \cal F } ^ { - 1 } ( \tau _ { i } ) \right| ^ { 2 } \leqslant C _ { \mathrm { a d d } } ( z ^ { 2 } + \varepsilon ) \qquad f o r \ e v e r y \ m \geqslant 1 .\tag{33}
$$

Here $C _ { \mathrm { a d d } } = C _ { \mathrm { a d d } } ( B , \ell , c _ { \mathrm { i n t } } , C _ { 0 } )$ depends only on the displayed deterministic geometry parameters.

The preceding inversion result reduces the remaining task to controlling the empirical Bellman CDF uniformly over all resolutions and all inputs in the localized neighborhood. The next lemma provides such a high-probability event in the precise density-adaptive form required by Lemma 4.4. The proof is given in Appendix A.2.

Lemma 4.5. Under Assumption $^ { 4 , }$ choose ρ and $\mathcal { U } _ { m } ( \rho )$ from Lemmas $4 . 2$ and $4 . 3 .$ There exist an instance-dependent constant $K _ { \mathcal { M } } < \infty$ and a constant $c _ { \mathcal { M } } ^ { * } > 0$ such that, for every $1 \leqslant t \leqslant c _ { \mathcal { M } } ^ { * } n$ there is an event $\mathcal { E } _ { n , t }$ satisfying

$$
\mathbb { P } ( \mathcal { E } _ { n , t } ) \geq 1 - 4 | S | e ^ { - t } .
$$

On $\mathcal { E } _ { n , t }$

$$
{ \cal D } _ { n } \leqslant \varepsilon _ { 0 } , \qquad { \cal D } _ { n } \leqslant \sqrt { 2 } | { \cal S } | \sqrt { \frac { t } { n } } ,
$$

and, simultaneously for every m ě 1, $\pmb \theta \in \mathcal { U } _ { m } ( \rho ) , s \in \mathcal { S } ,$ , and $x \in [ 0 , B ]$

$$
\left| \widehat { F } _ { s , \theta } ^ { ( n ) } ( x ) - F _ { s , \theta } ( x ) \right| \leqslant K _ { \mathcal { M } } \left\{ \sqrt { \frac { t f _ { s , \theta } ( x ) } { n } } + \frac { t } { n } \right\} .\tag{34}
$$

We now combine the preceding lemmas.

Proof of Theorem 3.2. Let

$$
V _ { n } : = \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) , \qquad M _ { * } : = \operatorname* { m a x } \{ M _ { 0 } , M _ { 1 } \} ,
$$

where $M _ { 0 } , M _ { 1 }$ are the fixed-model thresholds from Lemmas 4.3 and 4.2.

We first prove the exponential tail. Suppose $1 \leqslant t \leqslant c _ { \mathcal { M } } ^ { * } n$ and work on $\mathcal { E } _ { n , t }$ from Lemma 4.5. For every $m \geqslant M _ { * }$ , both fixed points belong to $\mathcal { U } _ { m } ( \rho )$ . Apply (34) at the data-dependent input $\pmb { \theta } _ { m } ^ { ( n ) }$ , and then apply Lemma 4.4 with

$$
z = K _ { \mathcal { M } } \sqrt { t / n } , \qquad \varepsilon = K _ { \mathcal { M } } t / n .
$$

Set

$$
K _ { 1 } ( \mathcal { M } ) : = \sqrt { C _ { \mathrm { a d d } } ( K _ { \mathcal { M } } ^ { 2 } + K _ { \mathcal { M } } ) } ,
$$

where $C _ { \mathrm { a d d } }$ uses the common geometry parameters $B , \ell , c _ { { \mathcal { M } } , \ell } , C _ { 0 }$ from Lemma 4.2. Since $z ^ { 2 } + \varepsilon =$ $( K _ { \mathcal { M } } ^ { 2 } + K _ { \mathcal { M } } ) t / n$ , the empirical fixed-point relation gives

$$
\mathopen { } \mathclose \bgroup \left\| \theta _ { m } ^ { ( n ) } - \Phi _ { m } ( \theta _ { m } ^ { ( n ) } ) \aftergroup \egroup \right\| _ { 2 , m } \leqslant K _ { 1 } ( \mathcal { M } ) \sqrt { t / n } .\tag{35}
$$

Since $\theta _ { m } = \Phi _ { m } ( \theta _ { m } )$ , Lemma 4.3 yields

$$
\begin{array} { r l } & { \left\| \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { m } \right\| _ { 2 , m } \leqslant K _ { 1 } ( \mathcal { M } ) \sqrt { t / n } + \left\| \Phi _ { m } ( \pmb { \theta } _ { m } ^ { ( n ) } ) - \Phi _ { m } ( \pmb { \theta } _ { m } ) \right\| _ { 2 , m } } \\ & { \qquad \leqslant K _ { 1 } ( \mathcal { M } ) \sqrt { t / n } + \lambda _ { 2 } \left\| \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { m } \right\| _ { 2 , m } . } \end{array}
$$

Therefore, simultaneously for every $m \geqslant M _ { * }$ ，

$$
\overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant \frac { K _ { 1 } ( \mathcal { M } ) } { 1 - \lambda _ { 2 } } \sqrt { t / n } .
$$

For each remaining resolution $1 \leqslant m < M _ { * }$ , Lemma 4.2 and the bound $D _ { n } \leqslant \sqrt { 2 } \left| \boldsymbol { S } \right| \sqrt { t / n }$ on $\mathcal { E } _ { n , t }$ give

$$
\overline { { W } } _ { 2 } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant K _ { 2 } ( \mathcal { M } ) \sqrt { t / n } , \qquad K _ { 2 } ( \mathcal { M } ) : = \sqrt { 2 } \left| \mathcal { S } \right| C _ { M _ { \ast } , \mathcal { M } } ,
$$

where $C _ { M _ { * } , \mathcal { M } }$ is the finite-resolution constant in that lemma. Define

$$
K _ { 0 } ( \mathcal { M } ) : = \operatorname* { m a x } \left\{ \frac { K _ { 1 } ( \mathcal { M } ) } { 1 - \lambda _ { 2 } } , \ K _ { 2 } ( \mathcal { M } ) , \frac { B } { \sqrt { c _ { \mathcal { M } } ^ { * } } } \right\} .
$$

The event in Lemma 4.5 controls the failure probability, so

$$
\mathbb { P } \left\{ V _ { n } > K _ { 0 } ( \mathcal { M } ) \sqrt { t / n } \right\} \leqslant 4 | \mathcal { S } | e ^ { - t } , \qquad 1 \leqslant t \leqslant c _ { \mathcal { M } } ^ { \ast } n .\tag{36}
$$

For $t > c _ { { \mathcal { M } } } ^ { * } n$ , the choice $K _ { 0 } ( \mathcal { M } ) \sqrt { c _ { \mathcal { M } } ^ { * } } \geqslant B$ and the support bound $V _ { n } \leqslant B$ make the event on the left of (36) empty. Thus (36) holds for every $t \geqslant 1$ , including when $c _ { \mathcal { M } } ^ { * } n < 1$

Set $C _ { 1 } ( \mathcal { M } ) = K _ { 0 } ( \mathcal { M } )$ . For any $\delta \in ( 0 , 1 )$ , take

$$
t = \log { \frac { 4 | { \mathcal { S } } | } { \delta } } \geqslant 1 .
$$

Then (36) gives

$$
\mathbb { P } \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant C _ { 1 } ( \mathcal { M } ) \sqrt { \frac { \log ( 4 | S | / \delta ) } { n } } \right\} \geqslant 1 - \delta ,
$$

which is (10). On this same event, $\overline { { W } } _ { p } \leqslant \overline { { W } } _ { 2 }$ for every $1 \leqslant p \leqslant 2$ , proving the simultaneous assertion for those orders. The construction uses the original paired empirical-joint estimator, with no sample splitting and no extra assumption on the endpoint decay rate.

Finally, the mean-square statement is a direct consequence of the exponential tail. Define

$$
Y _ { n } : = \frac { n V _ { n } ^ { 2 } } { K _ { 0 } ( \mathcal { M } ) ^ { 2 } } .
$$

By (36), for every $y \geqslant 1$

$$
\mathbb { P } ( Y _ { n } > y ) \leqslant 4 | S | e ^ { - y } .
$$

Hence, by the tail-integration identity for nonnegative random variables,

$$
\begin{array} { l } { \displaystyle \mathbb { E } Y _ { n } = \int _ { 0 } ^ { \infty } \mathbb { P } ( Y _ { n } > y ) \mathrm { d } y } \\ { \displaystyle \qquad \leqslant 1 + 4 | S | \int _ { 1 } ^ { \infty } e ^ { - y } \mathrm { d } y = 1 + \frac { 4 | S | } { e } . } \end{array}
$$

Choose

$$
C _ { 2 } ( \mathcal { M } ) : = K _ { 0 } ( \mathcal { M } ) \sqrt { 1 + \frac { 4 | \mathcal { S } | } { e } } .
$$

Then

$$
\mathbb { E } \left[ \left\{ \operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \right\} ^ { 2 } \right] \leqslant \frac { C _ { 2 } ( \mathcal { M } ) ^ { 2 } } { n } .\tag{37}
$$

This proves (12); Cauchy–Schwarz then gives (11) with the same $C _ { 2 } ( \mathcal { M } )$

## 4.3 Larger Wasserstein Orders

The two extensions above $W _ { 2 }$ use diferent deterministic norm comparisons. Neither changes the data event.

Proof of Corollary 3.1. Set

$$
a _ { n , \delta } : = C _ { 1 } ( \mathcal { M } ) \sqrt { \frac { \log ( 4 | \mathcal { S } | / \delta ) } { n } }
$$

and work on the event in (10), so that $\begin{array} { r } { \operatorname* { s u p } _ { m } \overline { { W } } _ { 2 } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant a _ { n , \delta } } \end{array}$ . For any m-vector d and $2 \leqslant p < \infty$

$$
\left( \frac { 1 } { m } \sum _ { i } | d _ { i } | ^ { p } \right) ^ { 1 / p } \leqslant m ^ { 1 / 2 - 1 / p } \left( \frac { 1 } { m } \sum _ { i } | d _ { i } | ^ { 2 } \right) ^ { 1 / 2 } .
$$

For $p = \infty$ 2 $\begin{array} { r } { \operatorname* { m a x } _ { i } | d _ { i } | \leqslant \sqrt { m } ( m ^ { - 1 } \sum _ { i } | d _ { i } | ^ { 2 } ) ^ { 1 / 2 } } \end{array}$ . Apply these inequalities to the ordered coordinate diferences in each state and take the state maximum. Identity (8) proves (13) simultaneously for all $m , p$ on the same event, which has probability at least $1 - \delta$

For Equation (14), we note that for any two laws $\mu , \nu$ supported in r0, Bs and finite $p > 2$ , their quantile diference has absolute value at most B. Hence

$$
W _ { p } ( \mu , \nu ) ^ { p } = \int _ { 0 } ^ { 1 } | F _ { \mu } ^ { - 1 } ( u ) - F _ { \nu } ^ { - 1 } ( u ) | ^ { p } \mathrm { d } u \leqslant B ^ { p - 2 } W _ { 2 } ( \mu , \nu ) ^ { 2 } .
$$

Apply this inequality statewise and take the maximum over states and resolutions on the event in (10). Taking pth roots gives

$$
\operatorname* { s u p } _ { m } \overline { { W } } _ { p } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant B ^ { 1 - 2 / p } a _ { n , \delta } ^ { 2 / p } = C _ { 1 } ( \mathcal { M } ) ^ { 2 / p } ( 1 - \gamma ) ^ { - ( 1 - 2 / p ) } \left\{ \frac { \log ( 4 | \mathcal { S } | / \delta ) } { n } \right\} ^ { 1 / p } .
$$

This proves (14) for every finite $p > 2$ on the same event. The case $p = 2$ is (10), and the case $p = \infty$ is the support bound $\begin{array} { r } { \operatorname* { s u p } _ { m } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant B } \end{array}$ □

## 4.4 Resolution-Independent $W _ { \infty }$ Consistency

The preceding interpolation argument stops at finite $p .$ For $W _ { \infty }$ we instead preserve the onedimensional reward-quantile modulus through one Bellman update and then use the global supremumnorm contraction.

Lemma 4.6 (One-step $W _ { \infty }$ perturbation under endpoint vanishing). Under the standing assumptions and Assumption $^ { 4 , }$ let $d _ { n } , r _ { n }$ be the primitive errors in Appendix B.1.5. On the probability-one

transition-support event, there is an instance-dependent $K _ { \mathcal { M } } < \infty$ such that, simultaneously for every $m \geqslant 1$ 2

$$
\begin{array} { r } { \big \| \widehat { \Phi } _ { m , n } ( \pmb \theta _ { m } ) - \Phi _ { m } ( \pmb \theta _ { m } ) \big \| _ { \infty } \leqslant K _ { \mathcal { M } } \big [ \omega _ { R } ( d _ { n } ) + r _ { n } \big ] . } \end{array}\tag{38}
$$

The proof is given in Appendix A.3. Its reward part uses generalized-inverse bracketing and the modulus (15); its transition part uses the relative mixture-weight bound together with the uniform midpoint logit stability in Lemma B.16.

Proof of Theorem 3.3. Write $\mathcal { T } = \{ ( s , a , s ^ { \prime } ) : P ( s ^ { \prime } \mid s , a ) > 0 \}$ as in Appendix B.1.5. By Lemma B.5, with probability at least $1 - \delta$

$$
d _ { n } + r _ { n } \leqslant t _ { n , \delta } : = \frac { 5 } { p _ { * } } \sqrt { \frac { \log ( 2 | \mathcal { T } | / \delta ) } { 2 n } } .\tag{39}
$$

Denote the constant in Lemma 4.6 by $K _ { \mathrm { s t e p } }$ . For each $m ,$ the population and empirical parameter maps are both γ-contractions in the coordinatewise supremum norm. Using their fixed-point identities and that lemma,

$$
\begin{array} { r l } & { \| \theta _ { m } ^ { ( n ) } - \theta _ { m } \| _ { \infty } \leqslant \| \widehat { \Phi } _ { m , n } ( \theta _ { m } ^ { ( n ) } ) - \widehat { \Phi } _ { m , n } ( \theta _ { m } ) \| _ { \infty } + \| \widehat { \Phi } _ { m , n } ( \theta _ { m } ) - \Phi _ { m } ( \theta _ { m } ) \| _ { \infty } } \\ & { \qquad \leqslant \gamma \| \theta _ { m } ^ { ( n ) } - \theta _ { m } \| _ { \infty } + K _ { \mathrm { s t e p } } [ \omega _ { R } ( d _ { n } ) + r _ { n } ] . } \end{array}
$$

Hence, pathwise on the support event,

$$
\operatorname* { s u p } _ { m \geqslant 1 } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant \frac { K _ { \mathrm { s t e p } } } { 1 - \gamma } [ \omega _ { R } ( d _ { n } ) + r _ { n } ] .\tag{40}
$$

To retain the radius in (16), define the explicit constants

$$
L _ { \mathrm { t a b } } : = \operatorname* { m a x } \left\{ 1 , \frac { \log ( 2 | \mathcal { Z } | ) } { \log ( 4 | S | ) } \right\} , \qquad J _ { \mathrm { m o d } } : = \left\lceil \operatorname* { m a x } \left\{ 1 , \frac { 5 } { 1 6 \sqrt { 2 } } \sqrt { L _ { \mathrm { t a b } } } \right\} \right\rceil .
$$

For every $\delta \in ( 0 , 1 )$ ,

$$
\log ( 2 | \mathcal { T } | / \delta ) \leqslant L _ { \mathrm { t a b } } \log ( 4 | \mathcal { S } | / \delta ) .
$$

Consequently, on (39),

$$
d _ { n } + r _ { n } \leqslant t _ { n , \delta } \leqslant J _ { \mathrm { m o d } } g _ { n , \delta } , \qquad g _ { n , \delta } : = \frac { 1 6 } { p _ { * } } \sqrt { \frac { \log ( 4 | S | / \delta ) } { n } } .
$$

For any positive integer J and $0 < h \leqslant 1$ , subdividing a probability interval into at most J pieces of length at most h gives

$$
\omega _ { R } ( \operatorname* { m i n } \{ 1 , J h \} ) \leqslant J \omega _ { R } ( h ) .
$$

If $g _ { n , \delta } < 1$ , then $h _ { n , \delta } = g _ { n , \delta } , d _ { n } \leqslant \operatorname* { m i n } \{ 1 , J _ { \mathrm { m o d } } h _ { n , \delta } \}$ , and $r _ { n } \leqslant J _ { \mathrm { m o d } } h _ { n , \delta }$ . Thus (40) is at most

$$
\frac { J _ { \mathrm { m o d } } K _ { \mathrm { s t e p } } } { 1 - \gamma } [ \omega _ { R } ( h _ { n , \delta } ) + h _ { n , \delta } ] .
$$

If $g _ { n , \delta } \geqslant 1$ , then $h _ { n , \delta } = 1$ and the support bound gives

$$
\operatorname* { s u p } _ { m } \overline { { W } } _ { \infty } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) \leqslant B = \frac { B } { 2 } [ \omega _ { R } ( 1 ) + 1 ] ,
$$

where $\omega _ { R } ( 1 ) = 1$ because every reward law has support r0, 1s. Therefore the explicit choice

$$
K _ { 3 } ( \mathcal { M } ) : = \operatorname* { m a x } \left\{ \frac { J _ { \mathrm { m o d } } K _ { \mathrm { s t e p } } } { 1 - \gamma } , \frac { B } { 2 } \right\}
$$

proves (17) for every n and δ, with exactly the radius (16).

Finally, for any fixed $\delta \in ( 0 , 1 ) , h _ { n , \delta } \to 0$ and (15) gives $\omega _ { R } ( h _ { n , \delta } )  0$ . Thus the displayed supremum converges to zero in probability. □

## 4.5 Analysis of the Local Asymptotic Minimax Risk

Fix m and a centre $Q _ { 0 }$ for this subsection. The proof has two parts: uniform attainment on the full Hellinger ball, and a lower bound using a finite-dimensional least-favourable experiment. We first reduce the distribution-valued loss to its exact coordinate form.

Lemma 4.7 (A loss-reducing decision map). For $\mu \in \Delta ( I )$ define

$$
a _ { i } ( \mu ) = m \int _ { ( i - 1 ) / m } ^ { i / m } F _ { \mu } ^ { - 1 } ( u ) \mathrm { d } u , \qquad A _ { m } \mu = \frac { 1 } { m } \sum _ { i } \delta _ { a _ { i } ( \mu ) } .
$$

The values $a _ { i } ( \mu )$ are ordered. For every ordered $\theta _ { 1 } , \ldots , \theta _ { m }$ and every $1 \leqslant p \leqslant \infty$

$$
W _ { p } \left( A _ { m } \mu , \frac { 1 } { m } \sum _ { i } \delta _ { \theta _ { i } } \right) \leqslant W _ { p } \left( \mu , \frac { 1 } { m } \sum _ { i } \delta _ { \theta _ { i } } \right) .
$$

The map is measurable and applies statewise to random decisions. Clipping an arbitrary candidate law to I first cannot increase its distance to a target supported on I, so the restriction to that support

entails no loss of generality.

The proof is given in Appendix A.4.

The averaging map in Lemma 4.7 is used only to compare decision rules. It does not replace the midpoint projection defining the estimator. The next lemma provides the uniform approximation needed for expected loss, rather than only a CLT at $Q _ { 0 }$

Lemma 4.8 (Linearization on the entire local ball). Fix m ě 1 and a centre $Q _ { 0 }$ satisfying the standing assumptions. For every fixed $c < \infty$ , uniformly over $Q \in { \mathcal { N } } _ { n } ( c ; Q _ { 0 } )$ ，

$$
\sqrt { n } \{ \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { m } ( Q ) \} = - \pmb { G } _ { m } ^ { - 1 } \frac { 1 } { \sqrt { n } } \sum _ { k = 1 } ^ { n } \sum _ { b } \{ \pmb { Y } _ { m , b } ( X _ { k , b } ) - \mathbb { E } _ { Q _ { b } } \pmb { Y } _ { m , b } \} + o _ { \mathbb { P } _ { Q } } ( 1 ) .\tag{41}
$$

The meaning of the remainder is that its probability of exceeding any fixed positive number tends to zero after taking the supremum over the local ball. The law of the left-hand side converges to ${ \mathsf { N } } ( 0 , V _ { m } )$ uniformly over that ball in the bounded-Lipschitz metric. Its $\ell _ { m , p } { - } r i s k$ converges uniformly to $\mathfrak { C } _ { m , p } ( Q _ { 0 } )$

The proof is given in Appendix A.4.

The lower bound uses the same canonical gradient. The following lemma records a leastfavourable submodel inside the full joint-law class, together with the local likelihood and target expansions.

Lemma 4.9 (Gradient and finite-dimensional least-favourable model). Let $\mathbb { H } = \bigoplus _ { b } L _ { 0 } ^ { 2 } ( Q _ { 0 , b } )$ with its sum-of-blocks inner product. For every bounded score $h = ( h _ { b } ) _ { b } \in \mathbb { H }$ , the path

$$
\mathrm { d } Q _ { \epsilon , b } = \left( 1 + \epsilon h _ { b } \right) \mathrm { d } Q _ { 0 , b }
$$

is in $\mathcal { Q }$ for suficiently small $| \epsilon |$ , and

$$
\dot { \theta } _ { m } [ h ] = - G _ { m } ^ { - 1 } \sum _ { b } \mathbb { E } _ { Q _ { 0 , b } } [ \pmb { Y } _ { m , b } h _ { b } ] = \sum _ { b } \mathbb { E } _ { Q _ { 0 , b } } [ \phi _ { m , b } h _ { b } ] .\tag{42}
$$

Let $r = { \mathrm { r a n k } } ( V _ { m } )$ $I f r > 0$ , there is a bounded r-dimensional joint-law submodel with information matrix ${ { I } _ { r } }$ at zero and target derivative $\pmb { A } \in \mathbb { R } ^ { d \times r }$ such that

$$
\mathbf { \boldsymbol { A } } \mathbf { \boldsymbol { A } } ^ { \top } = \mathbf { \boldsymbol { V } } _ { m } .
$$

The submodel can be parametrized by $\pmb { t } \in \mathbb { R } ^ { r }$ near zero so that, uniformly on fixed compact sets $o f t ,$

$$
\begin{array} { r l r } { \log \frac { \mathrm { d } \mathbb { P } _ { Q _ { t \sqrt { n } } } ^ { ( n ) } } { \mathrm { d } \mathbb { P } _ { Q _ { 0 } } ^ { ( n ) } } = t ^ { \top } \Delta _ { n } - \frac { 1 } { 2 } \| t \| _ { 2 } ^ { 2 } + o _ { \mathbb { P } _ { Q _ { 0 } } } ( 1 ) , } & { \qquad } & { \Delta _ { n } \overset { d } { \to } \mathbb { N } ( 0 , I _ { r } ) , } \\ { \sqrt { n } \{ \theta _ { m } ( Q _ { t / \sqrt { n } } ) - \theta _ { m } ( Q _ { 0 } ) \} = A t + o ( 1 ) , } & { } & { n d _ { H } ( Q _ { t / \sqrt { n } } , Q _ { 0 } ) ^ { 2 } = \frac { 1 } { 4 } \| t \| _ { 2 } ^ { 2 } + o ( 1 ) . } \end{array}
$$

The proof is given in Appendix A.4.

We will transfer a finite-prior lower bound from the Gaussian limit experiment. The required Gaussian decision fact is stated explicitly to specify both the loss and the class of decision rules.

Lemma 4.10 (Gaussian location risk). Observe $Y = t + Z$ with $Z \sim \mathsf { N } ( 0 , I _ { r } )$ and $t \in \mathbb { R } ^ { r }$ , and estimate $A t \in \mathbb { R } ^ { d }$ under any norm L on $\mathbb { R } ^ { d }$ . Then

$$
\operatorname* { i n f } _ { \delta } \operatorname* { s u p } _ { t } \mathbb { E } _ { t } L ( \delta ( Y ) - \pmb { A } t ) = \mathbb { E } L ( \pmb { A } Z ) .
$$

Furthermore, for every $\varepsilon > 0$ there is a finitely supported prior on a bounded set of t whose Bayes risk is at least E $L ( A Z ) - \varepsilon$

The proof is given in Appendix A.4.

Proof of Theorem $\ 3 . 4 \cdot$ The upper bound follows at once from Lemma 4.8: for every fixed $^ { c , }$ the paired estimator has worst local standardized risk tending to $\mathfrak { C } _ { m , p } ( Q _ { 0 } )$ . Thus both limiting minimax risks are at most that constant.

For the lower bound, apply Lemma 4.7 and allow, in the resulting vector decision problem, all actions in $\mathbb { R } ^ { d }$ . This relaxation can only reduce the infimum. Use the least-favourable submodel of Lemma 4.9. If $r = 0$ , the claimed lower bound is zero and is immediate; suppose $r > 0$

Choose a finite prior with parameter points $t _ { 1 } , \ldots , t _ { k }$ as in Lemma 4.10, with $L = \ell _ { m , p }$ and Bayes risk at least $\mathfrak { C } _ { m , p } ( Q _ { 0 } ) - \varepsilon$ . For suficiently large fixed $^ { c , }$ all alternatives $Q _ { t _ { j } / \sqrt { n } }$ lie in $\mathcal { N } _ { n } ( c ; Q _ { 0 } )$ for every suficiently large n. Let

$$
a _ { n , j } = \sqrt { n } \{ \theta _ { m } ( Q _ { t _ { j } / \sqrt { n } } ) - \theta _ { m } ( Q _ { 0 } ) \} \longrightarrow A t _ { j } , \qquad L _ { n , j } = \frac { \mathrm { d } \mathbb { P } _ { Q _ { t _ { j } / \sqrt { n } } } ^ { ( n ) } } { \mathrm { d } \mathbb { P } _ { Q _ { 0 } } ^ { ( n ) } } .
$$

The finite likelihood-ratio vector converges under $Q _ { 0 }$ by the LAN expansion in Lemma 4.9 to the likelihood-ratio vector in the Gaussian experiment $Y \sim \mathsf { N } ( t , \pmb { I } _ { r } )$ . Each coordinate of the limit has

expectation one; each $L _ { n , j }$ also has expectation one. Therefore the nonnegative finite sums of likelihood ratios are uniformly integrable: for each such sum $S _ { n } \Rightarrow S$ , its expectations converge to ES, while $\mathbb { E } ( S _ { n } \wedge L ) \to \mathbb { E } ( S \wedge L )$ for fixed L. Subtraction and $L \to \infty$ control the positive tails.

The local target vectors $a _ { n , j }$ lie in a fixed norm ball. As in the proof of Lemma 4.10, decisions outside twice that ball can be replaced by zero, so it sufices to minimize a bounded loss over a compact action set. If $w _ { j }$ are the prior weights, the finite-experiment Bayes risk is

$$
\mathbb { E } _ { Q _ { 0 } } \operatorname* { i n f } _ { a } \sum _ { j = 1 } ^ { k } w _ { j } L _ { n , j } \ell _ { m , p } ( a - a _ { n , j } ) .
$$

The infimum can be taken over a fixed compact ball containing all required actions for large n. A countable dense subset gives the same infimum, making it measurable. Continuity of the loss on this compact set makes the infimum continuous in the nonnegative likelihood-ratio vector and in the target vectors; it is bounded by $\begin{array} { r } { C _ { m , p , Q _ { 0 } , \Pi } \sum _ { j } w _ { j } L _ { n , j } } \end{array}$ for a finite constant depending only on the fixed finite prior Π, $m , p ,$ and $Q _ { 0 }$ . Randomization cannot improve this conditional infimum. Convergence in distribution and uniform integrability therefore show that these Bayes risks converge to the Gaussian finite-prior Bayes risk. The local worst-case risk is at least this Bayes risk. Consequently

$$
\operatorname* { l i m } _ { n } \operatorname* { i n f } r _ { n , m , p } ( c ; Q _ { 0 } ) \geqslant \mathfrak { C } _ { m , p } ( Q _ { 0 } ) - \varepsilon
$$

for all suficiently large c. Let $c \to \infty$ and then $\varepsilon \downarrow 0$ . This proves the lower bound and the equality of both limits. □

The argument does not identify the whole infinite-dimensional Hellinger ball with a finitedimensional Gaussian experiment. The least-favourable submodel is used only for the lower bound; Lemma 4.8 handles all alternatives for attainment, including dependent laws and new transition cells.

## 4.6 Resolution Dependence in the Fixed Instances

We first record the positive lower constant needed in both examples. Its proof uses one bounded reward-score direction for all resolutions.

Lemma 4.11 (A common reward-score direction). Under the standing assumptions there is $c _ { Q _ { 0 } } > 0$

such that

$$
\operatorname* { i n f } _ { m \geq 1 } \mathfrak { C } _ { m , 1 } ( Q _ { 0 } ) \geqslant c _ { Q _ { 0 } } .
$$

The proof is given in Appendix A.5.

Proof of Proposition 3.1. Assumptions 1 and 3 are immediate. We verify Assumption 2 for every m. Fix m and write $\theta _ { i } = \theta _ { m } ( i )$ . Let

$$
A _ { i } = \{ j : 0 < \theta _ { i } - \gamma \theta _ { j } < 1 \} , \qquad a _ { i } = | A _ { i } | , \qquad b _ { i } = \# \{ j : \theta _ { i } - \gamma \theta _ { j } \geqslant 1 \} .
$$

The uniform-reward CDF equation is

$$
a _ { i } \theta _ { i } - \gamma \sum _ { j \in A _ { i } } \theta _ { j } = m \tau _ { i } - b _ { i } .\tag{43}
$$

If $a _ { i } = 0$ , its CDF value would be $b _ { i } / m$ , which cannot equal $( i - 1 / 2 ) / m$ . Thus $a _ { i } \geqslant 1$ . For the fixed active sets define $C _ { i j } = \mathbb { 1 } \{ j \in A _ { i } \}$ and

$$
M ( z ) = \mathrm { d i a g } ( a _ { i } ) - z C .
$$

For $z = \gamma \in ( 0 , 1 )$ it is invertible because the row sums of $\mathrm { d i a g } ( a _ { i } ) ^ { - 1 } C$ equal one. It follows from (43) that each $\theta _ { i }$ is the value at $z = \gamma$ of a rational function $\Theta _ { i } ( z )$ with rational coeficients. Also det $M ( 0 ) \neq 0$ and

$$
\Theta _ { i } ( 0 ) = \frac { i - 1 / 2 - b _ { i } } { a _ { i } } \not \in \{ 0 , 1 \} .
$$

If a prohibited equality $\theta _ { i } - \gamma \theta _ { j } = e , e \in \{ 0 , 1 \}$ , held, clearing denominators in $\Theta _ { i } ( z ) - z \Theta _ { j } ( z ) - e$ would produce a polynomial with rational coeficients vanishing at $\gamma .$ It is not the zero polynomial because its value at zero is nonzero. This contradicts the transcendence of $\gamma = e ^ { - 1 }$ . The argument applies to every pair and every resolution.

For the lower bound, Lemma 4.11 and $\ell _ { m , p } \geqslant \ell _ { m , 1 }$ give a positive constant independent of $m , p .$ For the upper bound, the centre case of Theorem 3.4 and the expectation bound in Theorem 3.1 imply, at each fixed $m$

$$
\mathfrak { C } _ { m , \infty } ( Q _ { 0 } ) = \operatorname* { l i m } _ { n \to \infty } \mathbb { E } _ { Q _ { 0 } } \bigl [ \sqrt { n } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \bigr ] \leqslant C _ { Q _ { 0 } } .
$$

Since $\ell _ { m , p } \leqslant \ell _ { m , \infty }$ , this is also an upper bound for every $p .$ The MRP is fully specified, so the

two comparison constants are numerical and independent of $m , p .$ . For $m = 1$ the parameter is $( 1 / 2 ) / ( 1 - \gamma )$ , and its influence function for one reward observation is

$$
- \frac { 1 ! \{ R \leqslant 1 / 2 \} - 1 / 2 } { 1 - \gamma } .
$$

Therefore $V _ { 1 } = 1 / [ 4 ( 1 - \gamma ) ^ { 2 } ]$ and $\mathfrak { C } _ { 1 , p } = 1 / [ ( 1 - \gamma ) \sqrt { 2 \pi } ]$ for every $p .$

For the endpoint-vanishing example, the dificult part is to make one extreme coordinate have a large eficient variance at infinitely many resolutions, while retaining one fixed regular MRP. We isolate this construction in the next lemma.

Lemma 4.12 (A fixed instance with a large extreme-coordinate variance). There exist one fixed two-state, one-action $M R P Q { \mathrm { : } }$ with $\gamma = 1 / 2$ and $P ( t \mid s ) = P ( t \mid t ) = 1$ , satisfying Assumptions 1, 2, and $^ { 4 , }$ a sequence $m _ { j } \to \infty$ , and fixed-instance constants $c _ { * } , K _ { * } > 0$ such that

$$
[ V _ { m _ { j } } ] _ { ( s , 1 ) , ( s , 1 ) } \geqslant \frac { c _ { * } m _ { j } } { ( \log m _ { j } ) ^ { K _ { * } } }
$$

for all suficiently large $j$ . The constants and the MRP do not depend on $j , m _ { j } , n$ , or the Wasserstein order.

The proof is given in Appendix A.5.

Proof of Theorem 3.5. Choose $Q _ { * }$ and $m _ { j }$ from Lemma 4.12, and let $Z _ { m } \sim \mathsf { N } ( 0 , V _ { m } )$ . For every $2 < p \leqslant \infty$ , the coordinate ps, 1q gives

$$
\begin{array} { r l r } {  { \mathfrak { C } _ { m _ { j } , p } ( Q _ { * } ) \geqslant m _ { j } ^ { - 1 / p } \mathbb { E } | Z _ { m _ { j } } ( s , 1 ) | } } \\ & { } & { = \sqrt { \frac { 2 } { \pi } } m _ { j } ^ { - 1 / p } \sqrt { [ V _ { m _ { j } } ] _ { ( s , 1 ) , ( s , 1 ) } } \geqslant \frac { c _ { * } ^ { \prime } m _ { j } ^ { 1 / 2 - 1 / p } } { ( \log m _ { j } ) ^ { K _ { * } ^ { \prime } / 2 } } , } \end{array}
$$

where the renamed constants depend only on the fixed instance. This proves the lower inequality after renaming the logarithmic exponent.

For the upper inequality, fix m first. The centre CLT in Lemma 4.8, applied to nonnegative bounded truncations of $\ell _ { m , 2 } ^ { 2 }$ , and the second-moment bound (12) in Theorem 3.2 imply

$$
\mathbb { E } \ell _ { m , 2 } ( Z _ { m } ) ^ { 2 } \leqslant \operatorname* { l i m } _ { n  \infty } \Pi \mathbb { E } _ { Q _ { * } } \overline { { W } } _ { 2 } ( \eta _ { m } ^ { ( n ) } , \eta _ { m } ) ^ { 2 } \leqslant C _ { 2 } ( Q _ { * } ) ^ { 2 } .
$$

The constant is the same for all m because the finite-sample moment bound was resolutionindependent. Jensen’s inequality and the normalized-norm comparison used in Corollary 3.1 now give

$$
\mathfrak { C } _ { m , p } ( Q _ { * } ) \leqslant m ^ { 1 / 2 - 1 / p } \mathbb { E } \ell _ { m , 2 } ( Z _ { m } ) \leqslant C _ { 2 } ( Q _ { * } ) m ^ { 1 / 2 - 1 / p } , \qquad 2 < p \leqslant \infty .
$$

Thus the upper-bound constant can be chosen as $C _ { * } : = C _ { 2 } ( Q _ { * } )$ . For $1 \leqslant p \leqslant 2$ , use $\ell _ { m , p } \leqslant \ell _ { m , 2 }$ for the upper bound and Lemma 4.11 for the positive lower bound. This proves the stated $\Theta ( 1 )$ order in that regime. Finally, Theorem 3.4 identifies the Gaussian constants with the all-estimator local minimax risks. Every limit used above is taken at a fixed resolution before comparing diferent m. □

## 5 Conclusions

In this paper, we have investigated the statistical accuracy of model-based quantile distributional policy evaluation as the quantile resolution increases. Although a finer quantile representation introduces more parameters and increasingly extreme quantile levels, it does not necessarily lead to greater estimation error. Under uniformly positive reward densities, root-n estimation remains possible uniformly over all resolutions in every Wasserstein distance. Even when reward densities vanish at the support endpoints, this resolution-independent rate persists through $W _ { 2 }$ without any quantitative restriction on boundary decay. In contrast, higher-order Wasserstein distances are more sensitive to the estimation errors of extreme quantiles, leading to a potentially stronger dependence on the resolution.

Beyond these non-asymptotic guarantees, we have characterized the exact local asymptotic minimax risk at each fixed resolution and established its attainability by the quantile fixed point estimator. This characterization provides a statistical benchmark for examining the resolution dependence of the estimation error. Our fixed-instance analysis shows that the optimal local risk can remain bounded across resolutions, while under endpoint-vanishing reward densities it can also exhibit the polynomial growth predicted by our non-asymptotic bounds, up to logarithmic factors. Thus, the resolution dependence of the estimator reflects an intrinsic statistical dificulty at a fixed MDP.

## References

[1] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pages 449–458. PMLR, 2017.

[2] M. G. Bellemare, S. Candido, P. S. Castro, J. Gong, M. C. Machado, S. Moitra, S. S. Ponda, and Z. Wang. Autonomous navigation of stratospheric balloons using reinforcement learning. Nature, 588(7836):77–82, 2020.

[3] M. G. Bellemare, W. Dabney, and M. Rowland. Distributional Reinforcement Learning. MIT Press, 2023. http://www.distributional-rl.org.

[4] C. Bodnar, A. Li, K. Hausman, P. Pastor, and M. Kalakrishnan. Quantile qt-opt for risk-aware vision-based robotic grasping. In Robotics: Science and Systems XVI, 2020.

[5] S. Boucheron, G. Lugosi, and P. Massart. Concentration Inequalities: A Nonasymptotic Theory of Independence. Oxford University Press, 02 2013.

[6] Y. Chandak, S. Niekum, B. da Silva, E. Learned-Miller, E. Brunskill, and P. S. Thomas. Universal of-policy evaluation. Advances in Neural Information Processing Systems, 34: 27475–27490, 2021.

[7] Z. Cheng, Y. Peng, and Z. Zhang. Statistical eficiency and inference of quantile distributional reinforcement learning. arXiv preprint arXiv:2607.08444, 2026.

[8] W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning, pages 1096–1105. PMLR, 2018.

[9] W. Dabney, M. Rowland, M. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1): 2892–2901, 2018. doi: 10.1609/aaai.v32i1.11791.

[10] E. Ghysels, P. Santa-Clara, and R. Valkanov. There is a risk-return trade-of after all. Journal of financial economics, 76(3):509–548, 2005.

[11] B. Hao, X. Ji, Y. Duan, H. Lu, C. Szepesvari, and M. Wang. Bootstrapping fitted q-evaluation for of-policy inference. In International Conference on Machine Learning, pages 4074–4084. PMLR, 2021.

[12] A. Huang, L. Leqi, Z. Lipton, and K. Azizzadenesheli. Of-policy risk assessment for markov decision processes. In International Conference on Artificial Intelligence and Statistics, pages 5022–5050. PMLR, 2022.

[13] N. Jiang and L. Li. Doubly robust of-policy value evaluation for reinforcement learning. In International Conference on Machine Learning, pages 652–661. PMLR, 2016.

[14] J. Kober, J. A. Bagnell, and J. Peters. Reinforcement learning in robotics: A survey. The International Journal of Robotics Research, 32(11):1238–1274, 2013.

[15] P. W. Lavori and R. Dawson. Dynamic treatment regimes: practical design considerations. Clinical trials, 1(1):9–20, 2004.

[16] P. Massart. The tight constant in the dvoretzky-kiefer-wolfowitz inequality. The annals of Probability, pages 1269–1283, 1990.

[17] T. Morimura, M. Sugiyama, H. Kashima, H. Hachiya, and T. Tanaka. Nonparametric return distribution approximation for reinforcement learning. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pages 799–806, 2010.

[18] OpenAI. Gpt-4 technical report, 2023.

[19] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155, 2022.

[20] Y. Peng, L. Zhang, and Z. Zhang. Statistical eficiency of distributional temporal diference learning. Advances in Neural Information Processing Systems, 37:24724–24761, 2024.

[21] Y. Peng, K. Jin, L. Zhang, and Z. Zhang. A finite sample analysis of distributional td learning with linear function approximation. arXiv preprint arXiv:2502.14172, 2025.

[22] Z. Qi, C. Bai, Z. Wang, and L. Wang. Distributional of-policy evaluation in reinforcement learning. Journal of the American Statistical Association, 120(551):1517–1530, 2025.

[23] M. Rowland, M. Bellemare, W. Dabney, R. Munos, and Y. W. Teh. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 29–37. PMLR, 2018.

[24] M. Rowland, Y. Tang, C. Lyle, R. Munos, M. G. Bellemare, and W. Dabney. The statistical benefits of quantile temporal-diference learning for value estimation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 29210–29231. PMLR, 2023. URL https://proceedings.mlr.press/v202/ rowland23a.html.

[25] M. Rowland, R. Munos, M. G. Azar, Y. Tang, G. Ostrovski, A. Harutyunyan, K. Tuyls, M. G. Bellemare, and W. Dabney. An analysis of quantile temporal-diference learning. Journal of Machine Learning Research, 25(163):1–47, 2024. URL https://jmlr.org/papers/v25/ 23-0154.html.

[26] C. Shi, S. Zhang, W. Lu, and R. Song. Statistical inference of the value function for reinforcement learning in infinite-horizon settings. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(3):765–793, 2022.

[27] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M. Lanctot, L. Sifre, D. Kumaran, T. Graepel, et al. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419):1140–1144, 2018.

[28] H. A. Simon. Dynamic programming under uncertainty with a quadratic criterion function. Econometrica, Journal of the Econometric Society, pages 74–81, 1956.

[29] R. S. Sutton. The reward hypothesis, 2004. URL http://incompleteideas.net/rlai.cs. ualberta.ca/RLAI/rewardhypothesis.html.

[30] R. S. Sutton and A. G. Barto. Reinforcement learning: An introduction. MIT press, 2018.

[31] H. Theil. A note on certainty equivalence in dynamic planning. Econometrica: Journal of the Econometric Society, pages 346–349, 1957.

[32] P. Thomas, G. Theocharous, and M. Ghavamzadeh. High-confidence of-policy evaluation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2015.

[33] O. Vinyals, I. Babuschkin, W. M. Czarnecki, M. Mathieu, A. Dudzik, J. Chung, D. H. Choi, R. Powell, T. Ewalds, P. Georgiev, et al. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575(7782):350–354, 2019.

[34] R. Wu, M. Uehara, and W. Sun. Distributional ofline policy evaluation with predictive error guarantees. arXiv preprint arXiv:2302.09456, 2023.

[35] D. Yang, L. Zhao, Z. Lin, T. Qin, J. Bian, and T.-Y. Liu. Fully parameterized quantile function for distributional reinforcement learning. Advances in neural information processing systems, 32, 2019.

[36] L. Zhang, Y. Peng, J. Liang, W. Yang, and Z. Zhang. Estimation and inference in distributional reinforcement learning. The Annals of Statistics, 53(5):1987–2011, 2025.

[37] F. Zhou, J. Wang, and X. Feng. Non-crossing quantile regression for distributional reinforcement learning. Advances in Neural Information Processing Systems, 33, 2020.

## A Proofs of Supporting Results in Section 4

We prove the intermediate lemmas stated in Section 4, in the same order as they are used there. The proofs below assemble the technical results from Appendix B; the final proofs of the main theorems and corollaries remain in Section 4.

## A.1 Proof of Lemma 4.1

Proof of Lemma 4.1. Use the stratification and primitive errors $d _ { n } , r _ { n }$ from Appendix B.1.5. This is an exact rewriting of the paired empirical law, not a change of estimator. Define the hybrid target $\tilde { \tau } _ { n } ^ { \pi } \eta _ { m }$ by using the empirical transition frequencies and the population reward laws. By Lemma B.6, the reward law in each nonempty stratum can be coupled to its population law with discrepancy at most $d _ { n } / c _ { 0 }$ . Use the same action, successor state and continuation atom in the two Bellman targets. This gives

$$
\overline { { W } } _ { \infty } ( \widehat { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } , \widetilde { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } ) \leqslant d _ { n } / c _ { 0 } .\tag{44}
$$

Empty strata have zero empirical weight and do not afect this coupling.

It remains to compare the hybrid and population transition weights. Work on the probability-one event that a population-zero transition cell is never observed. Fix a state s and abbreviate the corresponding population and hybrid CDFs by $\boldsymbol { F } = F _ { m , s }$ and ${ \tilde { F } } .$ If $0 < r _ { n } \leqslant 1 / 8$ , Lemma B.7 gives

$$
| { \widetilde { F } } ( x ) - F ( x ) | \leqslant r _ { n } \operatorname* { m i n } \{ F ( x ) , 1 - F ( x ) \} .\tag{45}
$$

Fix $u = \tau _ { i }$ and put $q = F ^ { - 1 } ( u )$ . Set

$$
d _ { u } = 4 r _ { n } u ( 1 - u ) , \qquad u _ { - } = u - d _ { u } , \qquad u _ { + } = u + d _ { u } .
$$

Both levels lie in p0, 1q. The function $a ( v ) = \operatorname* { m i n } \{ v , 1 - v \}$ is 1-Lipschitz and satisfies $a ( u ) \leqslant 2 u ( 1 - u )$ Hence $( 1 - r _ { n } ) d _ { u } > r _ { n } a ( u )$ , and therefore

$$
u _ { - } + r _ { n } a ( u _ { - } ) < u , \qquad u _ { + } - r _ { n } a ( u _ { + } ) > u .
$$

Continuity gives $F ( F ^ { - 1 } ( u _ { \pm } ) ) = u _ { \pm }$ . Substituting these points into (45) yields the strict bracket

$$
\widetilde F ( F ^ { - 1 } ( u _ { - } ) ) < u < \widetilde F ( F ^ { - 1 } ( u _ { + } ) ) , \qquad F ^ { - 1 } ( u _ { - } ) \leqslant \widetilde F ^ { - 1 } ( u ) \leqslant F ^ { - 1 } ( u _ { + } ) .
$$

By Lemma B.1 in Cheng et al. [7], there is an instance-dependent $c _ { \mathcal { M } } > 0$ , independent of $m ,$ , such that

$$
| F ( q + h ) - u | \geqslant c _ { \mathcal { M } } u ( 1 - u ) | h | , \qquad 0 < | h | \leqslant B .\tag{46}
$$

Apply this inequality at $h = F ^ { - 1 } ( u _ { \pm } ) - q$ . The zero-displacement cases are immediate; otherwise it gives

$$
| F ^ { - 1 } ( u _ { \pm } ) - q | \leqslant \frac { | u _ { \pm } - u | } { c _ { \mathcal { M } } u ( 1 - u ) } = \frac { 4 r _ { n } } { c _ { \mathcal { M } } } .
$$

Consequently the same bound holds for $| \widetilde F ^ { - 1 } ( \tau _ { i } ) - F ^ { - 1 } ( \tau _ { i } ) |$ . Taking the maximum over states and coordinates proves

$$
\overline { { { W } } } _ { \infty } ( \mathbf { H } _ { m } \widetilde { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } , \pmb { \Pi } _ { m } \mathcal { T } ^ { \pi } \pmb { \eta } _ { m } ) \leqslant C _ { \mathcal { M } } r _ { n } .\tag{47}
$$

For $r _ { n } > 1 / 8$ , use the support bound $B \leqslant 8 B r _ { n }$ instead; for $r _ { n } = 0$ the two mixtures coincide. Thus (47) holds for every $r _ { n } .$ with one instance-dependent constant for all $m$

Projection is nonexpansive in $\overline { { W } } _ { \infty } ,$ so (44) and (47) imply the pathwise bound

$$
\Delta _ { n } \leqslant C _ { \mathcal { M } } ( d _ { n } + r _ { n } ) .
$$

To put the confidence factor in the form stated in the lemma, define

$$
L _ { \mathcal { T } } : = \operatorname* { m a x } \left\{ 1 , \frac { \log ( 2 | \mathcal { T } | ) } { \log ( 4 | S | ) } \right\} .
$$

For every $\delta \in ( 0 , 1 )$ ,

$$
\log \frac { 2 | \mathcal { T } | } { \delta } \leqslant L \tau \log \frac { 4 | \mathcal { S } | } { \delta } .
$$

Lemma B.5 therefore proves the asserted probability bound with

$$
K _ { \mathcal { M } } = \frac { 5 C _ { \mathcal { M } } } { p _ { \ast } } \sqrt { \frac { L _ { \mathcal { T } } } { 2 } } .
$$

Its expectation estimate also gives

$$
\mathbb { E } \Delta _ { n } \leqslant \frac { 5 C _ { \mathcal { M } } } { p _ { * } } \sqrt { \frac { 1 + \log ( 2 | \mathcal { T } | ) } { 2 n } } ,
$$

which supplies the expectation bound used in Theorem 3.1. No union over m is used. □

## A.2 Proofs of Lemmas 4.2–4.5

Proof of Lemma $4 . 2 .$ The boundary monotonicity and common interior lower bound follow from Lemma B.14; that lemma also proves convexity. The target-density formula (29) gives $0 \leqslant f _ { s , \pmb { \theta } } \leqslant C _ { 0 }$ and, using the zero endpoint values of each reward density and $0 \leqslant \theta ( s ^ { \prime } , j ) \leqslant B _ { \mathrm { ~ \scriptsize ~ : ~ } }$ , gives $f _ { s , \theta } ( 0 ) =$ $f _ { s , \theta } ( B ) = 0$

For localization, Lemma B.2 gives $\| \widehat { F } _ { s , \theta } ^ { ( n ) } - F _ { s , \theta } \| _ { \infty } \leqslant D _ { n }$ pathwise for all inputs. At the empirical fixed point, Lemma B.3 therefore yields

$$
\overline { { W } } _ { 1 } ( \pmb { \eta } _ { m } ^ { ( n ) } , T ^ { \pi } \pmb { \eta } _ { m } ^ { ( n ) } ) \leqslant B \left( D _ { n } + \frac { 1 } { 2 m } \right) .
$$

Use $\eta ^ { \pi } = \mathcal { T } ^ { \pi } \eta ^ { \pi }$ and the $\overline { { W } } _ { 1 }$ contraction to obtain

$$
\overline { { W } } _ { 1 } ( \eta _ { m } ^ { ( n ) } , \eta ^ { \pi } ) \leqslant \frac { B } { 1 - \gamma } \left( D _ { n } + \frac { 1 } { 2 m } \right) .
$$

For the population projected fixed point, Lemma B.1 gives

$$
\overline { { { W } } } _ { 1 } ( \pmb { \eta } _ { m } , \pmb { \eta } ^ { \pi } ) \leqslant \frac { B } { 2 m ( 1 - \gamma ) } .
$$

Choose $M _ { 1 }$ so that $B / [ 2 M _ { 1 } ( 1 - \gamma ) ] \leqslant \rho / 2$ , and choose $\varepsilon _ { 0 }$ so that $B \varepsilon _ { 0 } / ( 1 - \gamma ) \leqslant \rho / 2$ . These choices prove the simultaneous localization. The finitely many coarse-resolution bound is Lemma B.4. All thresholds depend only on the fixed instance and the chosen $\ell .$ □

Proof of Lemma $4 . 3 .$ Choose $\ell > 0$ and $\zeta > 0$ so that

$$
L = 1 + 4 C _ { 0 } \ell + \zeta , \qquad \gamma \sqrt { L } < 1 .
$$

Choose $\rho$ from Lemma B.14 and $M _ { 0 }$ from Lemma B.17 for these parameters. Along the line $\pmb { \theta } _ { t } = ( 1 - t ) \pmb { \theta } + t \pmb { \vartheta }$ , convexity keeps $\theta _ { t }$ in $\mathcal { U } _ { m } ( \rho )$ . Put $q _ { s , i } ( t ) = [ \Phi _ { m } ( \pmb { \theta } _ { t } ) ] _ { s , i }$

Lemma B.11 gives $f _ { s , \pmb { \theta } _ { t } } ( q _ { s , i } ( t ) ) > 0$ for every t. The zero-extended reward densities are continuous by Assumptions 1 and $4 ;$ their CDFs are therefore $C ^ { 1 }$ on R, including at support endpoints. Thus $( t , x ) \mapsto F _ { s , \pmb { \theta } _ { t } } ( x )$ is $C ^ { 1 }$ . The implicit function theorem applied to

$$
F _ { s , \pmb { \theta } _ { t } } ( q _ { s , i } ( t ) ) = \tau _ { i }
$$

shows that $q _ { s , i }$ is $C ^ { 1 }$ on $( 0 , 1 )$ , with one-sided extensions at the endpoints. In particular, no nonboundary condition is required along the interpolation path. Diferentiating the displayed equation gives

$$
\dot { q } _ { s , i } ( t ) = \gamma \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) } { m } \frac { p _ { s , a } ( q _ { s , i } ( t ) - \gamma \theta _ { t } ( s ^ { \prime } , j ) ) } { f _ { s , \theta _ { t } } ( q _ { s , i } ( t ) ) } ( \vartheta ( s ^ { \prime } , j ) - \theta ( s ^ { \prime } , j ) ) .
$$

The coeficient matrix after dividing by $\gamma$ is nonnegative and each row sums to one. Hence Jensen’s inequality gives

$$
\begin{array} { c } { \displaystyle \frac { 1 } { m } \sum _ { i } | \dot { q } _ { s , i } ( t ) | ^ { 2 } \leqslant \gamma ^ { 2 } \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) } { m } | \vartheta ( s ^ { \prime } , j ) - \theta ( s ^ { \prime } , j ) | ^ { 2 } } \\ { \displaystyle \qquad \times \left[ \frac { 1 } { m } \sum _ { i } \frac { p _ { s , a } ( q _ { s , i } ( t ) - \gamma \theta _ { t } ( s ^ { \prime } , j ) ) } { f _ { s , \theta _ { t } } ( q _ { s , i } ( t ) ) } \right] . } \end{array}
$$

By Lemma B.17, the bracket is at most

$$
L : = 1 + 4 C _ { 0 } \ell + \zeta .
$$

Thus

$$
\| { \dot { q } } ( t ) \| _ { 2 , m } \leqslant \gamma { \sqrt { L } } \ \| \vartheta - \theta \| _ { 2 , m } .
$$

The choice at the start gives $\lambda _ { 2 } = \gamma \sqrt { L } < 1$ . The path is absolutely continuous (also a consequence of the global supremum-norm contraction), so

$$
\Phi _ { m } ( \pmb { \vartheta } ) - \Phi _ { m } ( \pmb { \theta } ) = \int _ { 0 } ^ { 1 } \dot { q } ( t ) \mathrm { d } t .
$$

Minkowski’s integral inequality, followed by the maximum over states, now gives

$$
\begin{array} { r l r } & { } & { \| \Phi _ { m } ( \pmb { \vartheta } ) - \Phi _ { m } ( \pmb { \theta } ) \| _ { 2 , m } \leqslant { \displaystyle \int _ { 0 } ^ { 1 } } \| \dot { q } ( t ) \| _ { 2 , m } \mathrm { d } t } \\ & { } & { \leqslant \lambda _ { 2 } \| \pmb { \vartheta } - \pmb { \theta } \| _ { 2 , m } . } \end{array}
$$

Proof of Lemma $4 { \cdot } 4$ . Write $q = F ^ { - 1 }$ and $\widehat { q } = G ^ { - 1 }$ . If $\varepsilon \geqslant 1 / 4$ , the common support gives

$$
\frac { 1 } { m } \sum _ { i } | \widehat { q } ( \tau _ { i } ) - q ( \tau _ { i } ) | ^ { 2 } \leqslant B ^ { 2 } \leqslant 4 B ^ { 2 } \varepsilon .
$$

We henceforth suppose $0 \leqslant \varepsilon < 1 / 4$ . If $z = \varepsilon = 0$ , the two CDFs coincide and the assertion is immediate. Lemma B.12 also covers $z = 0$ by taking its four density envelopes to be zero.

A continuous-distribution estimate. For $\varepsilon < u < 1 - \varepsilon .$ , put $v _ { \pm } = u \pm \varepsilon$ and $r ( u ) = q ( v _ { + } ) - q ( v _ { - } )$ Lemma B.12 bounds the squared displacement by $C _ { \mathrm { g e o m } }$ times $r ( u ) ^ { 2 } + z ^ { 2 }$ and four shifted nonnegative envelopes whose total integral is at most $1 4 \ell z ^ { 2 }$ . Shifting their integration variables does not increase this total. Moreover, $r ( u ) \leqslant B$ and

$$
\begin{array} { r l } & { \displaystyle \int _ { \varepsilon } ^ { 1 - \varepsilon } r ( u ) \mathrm { d } u = \int _ { 2 \varepsilon } ^ { 1 } q ( v ) \mathrm { d } v - \int _ { 0 } ^ { 1 - 2 \varepsilon } q ( v ) \mathrm { d } v } \\ & { \qquad \displaystyle = \int _ { 1 - 2 \varepsilon } ^ { 1 } q ( v ) \mathrm { d } v - \int _ { 0 } ^ { 2 \varepsilon } q ( v ) \mathrm { d } v \leqslant 2 B \varepsilon . } \end{array}
$$

Thus $\int r ( u ) ^ { 2 }$ du $\leqslant 2 B ^ { 2 } \varepsilon$ . The omitted endpoint intervals have total length $2 \varepsilon$ and squared displacement at most $B ^ { 2 }$ . Consequently,

$$
W _ { 2 } ( G , F ) ^ { 2 } \leqslant C _ { \mathrm { g e o m } } ( z ^ { 2 } + \varepsilon ) ,\tag{48}
$$

after increasing the constant depending only on $B , \ell , c _ { \mathrm { i n t } } , C _ { 0 }$ . This is an intermediate estimate, not yet the midpoint-grid conclusion.

Shift smaller than one grid cell: $m \varepsilon < 1 / 4$ . Every $\tau _ { i } \pm \varepsilon$ lies in the inner half of its original probability cell. Lemma B.13 bounds the discrete average of each shifted monotone envelope by four times its integral, and gives

$$
\frac { 1 } { m } \sum _ { i } r ( \tau _ { i } ) ^ { 2 } \leqslant B ( 1 6 \ell + 2 / c _ { \mathrm { i n t } } ) \varepsilon .
$$

Average the pointwise inequality (78) over the midpoint grid. The four envelope terms contribute at most $C _ { \mathrm { g e o m } } z ^ { 2 }$ , the displayed rank-shift bound contributes at most $C _ { \mathrm { g e o m } } \varepsilon$ , and the constant $z ^ { 2 }$ term stays $z ^ { 2 }$ . This proves (33) in this regime, including $\varepsilon = 0$

Shift at least one quarter-cell: $m \varepsilon \geqslant 1 / 4$ . For any law $\mu$ supported in $[ 0 , B ]$ , the midpoint quantization estimate in Lemma B.1 and the support diameter imply

$$
W _ { 2 } ( \Pi _ { m } \mu , \mu ) ^ { 2 } \leqslant B W _ { 1 } ( \Pi _ { m } \mu , \mu ) \leqslant \frac { B ^ { 2 } } { 2 m } .
$$

Apply the triangle inequality through G and F, square, and use (48):

$$
\begin{array} { l } { \displaystyle { W _ { 2 } ( \Pi _ { m } G , \Pi _ { m } F ) ^ { 2 } \leqslant 3 W _ { 2 } ( G , F ) ^ { 2 } + \frac { 3 B ^ { 2 } } { m } } } \\ { \displaystyle { \leqslant C _ { \mathrm { g e o m } } ( z ^ { 2 } + \varepsilon ) , } } \end{array}
$$

since $1 / m \leqslant 4 \varepsilon$ . Both projected laws have ordered equal weights, so the left side is exactly $\begin{array} { r } { m ^ { - 1 } \sum _ { i } | G ^ { - 1 } ( \tau _ { i } ) - F ^ { - 1 } ( \tau _ { i } ) | ^ { 2 } } \end{array}$ . Choose $C _ { \mathrm { a d d } }$ to dominate the constants in these cases, including $4 B ^ { 2 }$ Its dependence is only on the displayed geometry. The temporary quantization term is used solely where it is dominated by ε and is absent from the final conclusion. □

Proof of Lemma $4 . 5 .$ Use $\tau , p _ { * } , r _ { n }$ and the stratum counts from Appendix B.1.5. We first obtain the displayed numerical constant for $D _ { n }$ . Write

$$
Z _ { s , a , s ^ { \prime } } : = \| \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } \| _ { \infty } , \qquad D _ { n , s } : = \sum _ { a } \pi ( a \mid s ) \sum _ { s ^ { \prime } } Z _ { s , a , s ^ { \prime } } ,
$$

so that $D _ { n } = \operatorname* { m a x } _ { s } D _ { n , s }$ . The scalar DKW argument in Lemma B.2 gives

$$
\begin{array} { r } { \mathbb { P } ( Z _ { s , a , s ^ { \prime } } > v ) \leqslant 2 e ^ { - 2 n v ^ { 2 } } , \qquad v > 0 . } \end{array}
$$

Integrating this tail yields

$$
\begin{array} { r l r } & { } & { \displaystyle \mathbb { E } \exp \left( \frac { n } { 2 } Z _ { s , a , s ^ { \prime } } ^ { 2 } \right) = 1 + \frac { n } { 2 } \int _ { 0 } ^ { \infty } e ^ { n v / 2 } \mathbb { P } ( Z _ { s , a , s ^ { \prime } } ^ { 2 } > v ) \mathrm { d } v } \\ & { } & { \displaystyle \leqslant 1 + n \int _ { 0 } ^ { \infty } e ^ { - 3 n v / 2 } \mathrm { d } v = \frac { 5 } { 3 } . } \end{array}
$$

For each fixed s, the weights $\lambda _ { a , s ^ { \prime } } : = \pi ( a \mid s ) / | S |$ sum to one. Applying Jensen’s inequality first to the square and then to the exponential gives

$$
\begin{array} { r l r } {  { \mathbb { E } \exp ( \frac { n D _ { n , s } ^ { 2 } } { 2 | { \mathcal S } | ^ { 2 } } ) \leqslant \mathbb { E } \exp ( \frac { n } { 2 } \sum _ { a , s ^ { \prime } } \lambda _ { a , s ^ { \prime } } Z _ { s , a , s ^ { \prime } } ^ { 2 } ) } } \\ & { } & { \displaystyle \leqslant \sum _ { a , s ^ { \prime } } \lambda _ { a , s ^ { \prime } } \mathbb { E } \exp ( \frac { n } { 2 } Z _ { s , a , s ^ { \prime } } ^ { 2 } ) \leqslant \frac { 5 } { 3 } . } \end{array}
$$

No independence between the sub-CDF errors is required. Markov’s inequality and a union bound

over states show that the event

$$
{ \mathcal { D } } _ { n , t } : = \left\{ D _ { n } \leqslant { \sqrt { 2 } } | S | { \sqrt { t / n } } \right\}
$$

satisfies

$$
\mathbb { P } ( \mathcal { D } _ { n , t } ^ { c } ) \leqslant \frac { 5 } { 3 } | { \cal S } | e ^ { - t } .\tag{49}
$$

We allocate the remaining failure probability to the transition labels and conditional reward CDFs. Define

$$
a _ { \mathcal { Z } } : = 1 + \log \frac { 3 | \mathcal { Z } | } { | S | } , \qquad u : = t + \log \frac { 3 | \mathcal { Z } | } { | S | } .
$$

Since $\vert \mathcal { T } \vert \geqslant \vert \mathcal { S } \vert$ and $t \geqslant 1$ , we have $1 \leqslant u \leqslant a _ { \mathcal { T } } t$ . Hoefding’s inequality gives an event $\mathcal { R } _ { n , t }$ depending only on the successor-state labels, on which

$$
\operatorname* { m a x } _ { ( s , a , s ^ { \prime } ) \in \mathbb { Z } } | \widehat { P } ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , a ) | \leqslant \sqrt { \frac { u } { 2 n } } , \qquad r _ { n } \leqslant \frac { 1 } { p _ { * } } \sqrt { \frac { a _ { \mathbb { Z } } t } { 2 n } } ,
$$

and

$$
\mathbb { P } ( \mathcal { R } _ { n , t } ^ { c } ) \leqslant 2 | \mathbb { Z } | e ^ { - u } .\tag{50}
$$

Choose

$$
c _ { \mathcal { M } } ^ { \ast } : = \operatorname* { m i n } \left\{ \frac { \varepsilon _ { 0 } ^ { 2 } } { 2 | S | ^ { 2 } } , \frac { p _ { \ast } ^ { 2 } } { 2 a _ { \mathcal { T } } } \right\} > 0 .\tag{51}
$$

For 1 $\leqslant t \leqslant c _ { \mathcal { M } } ^ { \ast } n$ , the event $\mathcal { D } _ { n , t }$ ensures $D _ { n } \leqslant \varepsilon _ { 0 }$ , while $\mathcal { R } _ { n , t }$ ensures $r _ { n } \leqslant 1 / 2$ . In particular, on $\mathcal { R } _ { n , t }$ every positive-probability stratum has $N _ { s , a , s ^ { \prime } } \geqslant n p _ { * } / 2$ and $\widehat { P } ( s ^ { \prime } \mid s , a ) \leqslant ( 3 / 2 ) P ( s ^ { \prime } \mid s , a )$

Let $C _ { R } : = \operatorname* { m a x } _ { s , a } C _ { F _ { s , a } }$ , where $C _ { F _ { s , a } }$ is the constant in Lemma B.10 for the reward CDF $F _ { s , a }$ Condition only on the full array of successor-state labels. For every label realization in $\mathcal { R } _ { n , t } ,$ , the rewards in each stratum are independent with CDF $F _ { s , a }$ . Apply that lemma with confidence parameter u in each stratum. Define $B _ { n , t }$ by requiring its bound, with sample size $N _ { s , a , s ^ { \prime } ; }$ , in every nonempty positive-probability stratum; empty strata impose no condition. Then

$$
\begin{array} { r } { \mathbb { P } ( \mathcal { R } _ { n , t } \cap \mathcal { B } _ { n , t } ^ { c } ) \leqslant | \mathcal { I } | e ^ { - u } . } \end{array}\tag{52}
$$

On $\mathcal { R } _ { n , t } \cap B _ { n , t }$ , substituting $N _ { s , a , s ^ { \prime } } \geqslant n p _ { * } / 2$ and u ď a<sub>I</sub>t gives

$$
| \widehat { F } _ { s , a , s ^ { \prime } } ( \boldsymbol { r } ) - F _ { s , a } ( \boldsymbol { r } ) | \leqslant C _ { R } \left\{ \sqrt { \frac { 2 a _ { \mathcal { I } } t p _ { s , a } ( \boldsymbol { r } ) } { n p _ { \ast } } } + \frac { 2 a _ { \mathcal { T } } t } { n p _ { \ast } } \right\} .\tag{53}
$$

The inequality holds simultaneously for all positive-probability strata and all $r \in [ 0 , 1 ]$ . Outside that interval the CDF discrepancy is zero, so it also holds for every real r with the zero-extended density. The conditioning here does not involve $\mathcal { D } _ { n , t }$ , which depends on rewards.

Intersect $\mathcal { D } _ { n , t } , \mathcal { R } _ { n , t } , \mathcal { B } _ { n , t }$ and the probability-one transition-support event, and call the result $\mathcal { E } _ { n , t }$ . Equations (49), (50), and (52) give

$$
\begin{array} { r l r } & { } & { \mathbb { P } ( \mathscr { E } _ { n , t } ^ { c } ) \leqslant \displaystyle \frac { 5 } { 3 } | { \cal S } | e ^ { - t } + 3 | { \cal T } | e ^ { - u } } \\ & { } & { \quad = \displaystyle \frac { 8 } { 3 } | { \cal S } | e ^ { - t } \leqslant 4 | { \cal S } | e ^ { - t } . } \end{array}
$$

It remains to verify the Bellman-target bound on this same event.

Insert (53) into the reward part of the exact stratum mixture (70). Its additive contribution is at most $2 a _ { \mathcal { T } } C _ { R } t / ( n p _ { * } )$ , since the empirical mixture weights sum to one. For the square-root contribution, Jensen’s inequality gives

$$
\begin{array} { r l r } {  { \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) \widehat { P } ( s ^ { \prime } \mid s , a ) } { m } \sqrt { p _ { s , a } ( x - \gamma \theta ( s ^ { \prime } , j ) ) } } } \\ & { } & { \leqslant \{ \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) \widehat { P } ( s ^ { \prime } \mid s , a ) } { m } p _ { s , a } ( x - \gamma \theta ( s ^ { \prime } , j ) ) \} ^ { 1 / 2 } \leqslant \sqrt { \frac { 3 } { 2 } f _ { s , \theta } ( x ) } . } \end{array}
$$

Thus the reward contribution is at most

$$
C _ { R } \sqrt { \frac { 3 a _ { \mathcal { T } } } { p _ { * } } } \sqrt { \frac { t f _ { s , \theta } ( x ) } { n } } + \frac { 2 a _ { \mathcal { T } } C _ { R } } { p _ { * } } \frac { t } { n } .
$$

Write $c _ { \mathrm { i n t } } = c _ { \mathcal { M } , \ell }$ for the common interior density lower bound and set

$$
C _ { \mathrm { c d f } } : = \operatorname* { m a x } \{ B , c _ { \mathrm { i n t } } ^ { - 1 } \} .
$$

The proof of Lemma B.15 gives

$$
\operatorname* { m i n } \{ F _ { s , \theta } ( x ) , 1 - F _ { s , \theta } ( x ) \} \leqslant C _ { \mathrm { c d f } } \sqrt { C _ { 0 } } \sqrt { f _ { s , \theta } ( x ) }
$$

uniformly for $\pmb \theta \in \mathcal { U } _ { m } ( \rho )$ . Consequently, Lemma B.7 bounds the transition contribution by

$$
r _ { n } \operatorname* { m i n } \{ F _ { s , \theta } ( x ) , 1 - F _ { s , \theta } ( x ) \} \leqslant \frac { C _ { \mathrm { c d f } } } { p _ { * } } \sqrt { \frac { a _ { \mathcal { Z } } C _ { 0 } } { 2 } } \sqrt { \frac { t f _ { s , \theta } ( x ) } { n } } .
$$

Combining the two contributions proves (34) with the explicit choice

$$
K _ { \mathcal { M } } : = \operatorname* { m a x } \left\{ C _ { R } \sqrt { \frac { 3 a _ { \mathcal { T } } } { p _ { * } } } + \frac { C _ { \mathrm { c d f } } } { p _ { * } } \sqrt { \frac { a _ { \mathcal { T } } C _ { 0 } } { 2 } } , \frac { 2 a _ { \mathcal { T } } C _ { R } } { p _ { * } } \right\} .
$$

Both $K _ { \mathcal { M } }$ and the range constant in (51) are independent of $m , n , t .$ . The reward-CDF event was constructed before substituting any shifts or parameter arrays, so no union bound over resolutions, quantile coordinates, or inputs is needed. In particular, the result applies at the data-dependent empirical fixed points whenever they belong to the common neighborhood. □

## A.3 Proof of Lemma 4.6

Proof of Lemma $4 . 6 .$ Use the exact stratum decomposition and primitive errors $d _ { n } , r _ { n }$ from $\mathrm { A p \mathrm { - } }$ pendix B.1.5. Introduce the hybrid target $\tilde { \tau } _ { n } ^ { \pi } \eta _ { m }$ that uses the empirical transition frequencies but the population reward laws.

For every nonempty stratum, generalized-inverse bracketing gives

$$
W _ { \infty } ( \widehat { F } _ { s , a , s ^ { \prime } } , F _ { s , a } ) \leqslant \omega _ { R } ( d _ { n } ) .
$$

Indeed, $\| \widehat { F } _ { s , a , s ^ { \prime } } - F _ { s , a } \| _ { \infty } \leqslant d _ { n }$ implies

$$
F _ { s , a } ^ { - 1 } ( ( u - d _ { n } ) \vee 0 ) \leqslant \widehat { F } _ { s , a , s ^ { \prime } } ^ { - 1 } ( u ) \leqslant F _ { s , a } ^ { - 1 } ( ( u + d _ { n } ) \wedge 1 ) ,
$$

and the two outer quantiles difer from $F _ { s , a } ^ { - 1 } ( u )$ by at most $\omega _ { R } ( d _ { n } )$ . Empty strata carry zero empirical transition weight. Coupling the reward variables by these quantile couplings while keeping the action, successor state and continuation atom fixed therefore yields

$$
\begin{array} { r } { \overline { { W } } _ { \infty } ( \widehat { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } , \widetilde { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } ) \leqslant \omega _ { R } ( d _ { n } ) . } \end{array}\tag{54}
$$

Projection does not increase $\overline { { W } } _ { \infty }$

It remains to compare the hybrid and population transition weights. Fix a state and write $F = F _ { m , s }$ for the population Bellman-target CDF and $\tilde { F }$ for its hybrid version. On the probabilityone support event, Lemma B.7 gives

$$
| { \widetilde { F } } ( x ) - F ( x ) | \leqslant r _ { n } \operatorname* { m i n } \{ F ( x ) , 1 - F ( x ) \} .\tag{55}
$$

For $0 < r _ { n } \leqslant r _ { 0 }$ , where $r _ { 0 } > 0$ is a suficiently small instance-dependent constant, fix $u = \tau _ { i }$ and set

$$
d _ { u } = 4 r _ { n } u ( 1 - u ) , \qquad u _ { - } = u - d _ { u } , \qquad u _ { + } = u + d _ { u } .
$$

As in the proof of Lemma 4.1, the 1-Lipschitz property of v ÞÑ min $\{ v , 1 - v \}$ and (55) imply

$$
F ^ { - 1 } ( u _ { - } ) \leqslant \widetilde { F } ^ { - 1 } ( u ) \leqslant F ^ { - 1 } ( u _ { + } ) .
$$

Moreover, throughout the interval between u and $u \pm$ , both v and $1 - v$ remain within fixed factors of u and $1 - u$ . Hence

$$
| \operatorname { l o g i t } ( u _ { \pm } ) - \operatorname { l o g i t } ( u ) | \leqslant C r _ { n } .
$$

Choose $r _ { 0 }$ small enough that this logit distance lies in the neighbourhood of Lemma B.16. That lemma gives

$$
| F ^ { - 1 } ( u _ { \pm } ) - F ^ { - 1 } ( u ) | \leqslant K _ { \mathcal { M } } r _ { n } ,
$$

and therefore

$$
\overline { { W } } _ { \infty } ( \mathbf { H } _ { m } \widetilde { T } _ { n } ^ { \pi } \pmb { \eta } _ { m } , \pmb { \Pi } _ { m } \mathcal { T } ^ { \pi } \pmb { \eta } _ { m } ) \leqslant K _ { \mathcal { M } } r _ { n }\tag{56}
$$

uniformly in m. If $r _ { n } > r _ { 0 }$ , the common support gives the same form after increasing $K _ { \mathcal { M } }$ , since $B \leqslant ( B / r _ { 0 } ) r _ { n } ;$ if $r _ { n } = 0$ the two targets coincide.

Combining (54) and (56), and using the triangle inequality after projection, proves (38).

## A.4 Proofs of Lemmas 4.7–4.10

Proof of Lemma $4 . 7 .$ Monotonicity of the quantile function gives ordering. For $p < \infty$ , Jensen’s inequality on each probability cell gives

$$
\frac { 1 } { m } | a _ { i } ( \mu ) - \theta _ { i } | ^ { p } \leqslant \int _ { ( i - 1 ) / m } ^ { i / m } | F _ { \mu } ^ { - 1 } ( u ) - \theta _ { i } | ^ { p } \mathrm { d } u .
$$

Summing proves the claim. For $p = \infty$ , the absolute value of a cell average is bounded by the essential supremum on that cell. The same inequalities survive the maximum over states. Quantile integrals on a bounded interval are measurable in the underlying probability measure. Conditioning on any auxiliary randomization proves the randomized version. □

Proof of Lemma 4.8. All population targets under $Q \in \mathcal { Q }$ have continuous CDFs, since the reward

coordinate has a Lebesgue density. Thus

$$
\pmb { H } _ { Q } ( \pmb { \theta } _ { m } ( Q ) ) = \pmb { \tau } _ { m } .
$$

For the empirical target, the lower-inverse convention gives

$$
H _ { n } ( \theta _ { m } ^ { ( n ) } ) = \tau _ { m } + r _ { n } , \qquad 0 \leqslant r _ { n } ( s , i ) \leqslant | { \cal S } | / n \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

To verify the bound even at a data-dependent parameter, fix $( s , a , s ^ { \prime } , j )$ . The rewards in the block $( s , a )$ are almost surely distinct, so at a given target value at most one observation with next state $s ^ { \prime }$ can contribute for this fixed $j .$ Summing over $s ^ { \prime } , j .$ , multiplying by $1 / ( n m )$ and averaging over a gives the stated bound. For deterministic transitions it improves to $1 / n$

Let $\pmb { \theta } _ { Q } = \pmb { \theta } _ { m } ( Q )$ and $\pmb { \theta } _ { 0 } = \pmb { \theta } _ { m } ( Q _ { 0 } )$ . Lemma B.18 gives $\sqrt { n } \| \pmb { \theta } _ { Q } - \pmb { \theta } _ { 0 } \| _ { \infty } \leqslant L _ { m , Q _ { 0 } } c$ and, for each $r \geqslant 1$ 2

$$
\operatorname* { s u p } _ { Q \in {  { N _ { n } } } ( c ; Q _ { 0 } ) } \mathbb { P } _ { Q } \{ \sqrt { n } \| \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { 0 } \| _ { \infty } > M \} \leqslant C _ { m , c , r , Q _ { 0 } } M ^ { - r } .
$$

Fix $M > L _ { m , Q _ { 0 } } c .$ . All expansions that follow are first made on this event’s complement and then M is allowed to increase.

At the centre, Assumption 2 and continuity of the reward densities imply

$$
\begin{array} { r } { H _ { Q _ { 0 } } ( \theta _ { 0 } + u ) = \tau _ { m } + G _ { m } u + R _ { 0 } ( u ) , \qquad \| R _ { 0 } ( u ) \| _ { \infty } \leqslant \| u \| _ { \infty } \omega _ { m } ( \| u \| _ { \infty } ) , } \end{array}
$$

where $\omega _ { m } ( t )  0 \mathrm { \ a s \ } t \downarrow 0$ . Subtracting the two estimating equations gives the exact decomposition

$$
\begin{array} { r l } { G _ { m } \sqrt { n } \big ( \theta _ { m } ^ { ( n ) } - \theta _ { Q } \big ) = } & { - \sqrt { n } \big ( H _ { n } - H _ { Q } \big ) ( \theta _ { 0 } ) + \sqrt { n } r _ { n } } \\ & { \phantom { = } - \sqrt { n } \{ ( H _ { n } - H _ { Q } ) ( \theta _ { m } ^ { ( n ) } ) - ( H _ { n } - H _ { Q } ) ( \theta _ { 0 } ) \} } \\ & { \phantom { = } - \sqrt { n } \{ ( H _ { Q } - H _ { Q _ { 0 } } ) ( \theta _ { m } ^ { ( n ) } ) - ( H _ { Q } - H _ { Q _ { 0 } } ) ( \theta _ { Q } ) \} } \\ & { \phantom { = } - \sqrt { n } \{ R _ { 0 } ( \theta _ { m } ^ { ( n ) } - \theta _ { 0 } ) - R _ { 0 } ( \theta _ { Q } - \theta _ { 0 } ) \} . } \end{array}
$$

On the radius- $M / \sqrt { n }$ event the last line is at most $2 M \omega _ { m } ( M / \sqrt { n } )$ in norm. Lemma B.19, with $\delta = M / { \sqrt { n } }$ and $d \leqslant c / { \sqrt { n } }$ , bounds the expected norm of the empirical increment by

$$
C _ { m , Q _ { 0 } } \sqrt { M / \sqrt { n } + c ^ { 2 } / n } ;
$$

Markov’s inequality makes it uniformly negligible. Its population counterpart is bounded deterministically by

$$
2 C _ { m , Q _ { 0 } } c \sqrt { M / \sqrt { n } + c ^ { 2 } / n } .
$$

Finally ${ \sqrt { n } } \| r _ { n } \| _ { \infty } \leqslant | S | / { \sqrt { n } }$ . Thus all remainders vanish for fixed M. Taking lim $\mathrm { s u p } _ { n }$ and then $M \to \infty$ removes the localization event uniformly in $Q$ . The leading empirical term is precisely the sum in (41).

For every sequence $Q _ { n } \in \mathcal { N } _ { n } ( c ; Q _ { 0 } )$ , the leading sum is a triangular array of independent, uniformly bounded, centred random vectors of fixed dimension. Its covariance tends to $\Sigma _ { m }$ , because max<sub>b</sub> $\mathrm { T V } ( Q _ { n , b } , Q _ { 0 , b } )  0$ . The Lindeberg central limit theorem yields the asserted Gaussian limit. The argument for every sequence implies uniform convergence: any failure of uniformity would select a contradicting sequence $Q _ { n }$

Finally, Lemma B.18 bounds a moment strictly larger than one of the scaled error, uniformly over the ball. Since $\ell _ { m , p } ( z ) \leqslant \| z \| _ { \infty }$ , it supplies uniform integrability for the loss, including $p = \infty$ For clarity, for any $r > 1$ and $L > 0$ the contribution of losses exceeding L is at most $C _ { m , c , r , Q _ { 0 } } L ^ { 1 - r }$ uniformly in $n , Q$ . The truncated norm min $\{ \ell _ { m , p } , L \}$ is bounded and Lipschitz. First pass to its uniform Gaussian limit and then let $L  \infty$ . This proves the asserted uniform risk convergence, including the expected-distance rather than merely weak-convergence assertion. □

Proof of Lemma $4 . 9 .$ Normalization holds because $\mathbb { E } h _ { b } = 0 ;$ boundedness ensures positivity for small $\epsilon .$ The target equation changes at fixed input by $\epsilon \sum _ { b } \mathbb { E } [ Y _ { m , b } h _ { b } ]$ . Lemma B.18 applied to population laws gives a parameter displacement at most $C _ { m , Q _ { 0 } , h } | \epsilon |$ for a finite constant depending only on $m , Q _ { 0 }$ and the fixed bounded score h. The score contribution at the displaced input difers by at most $C _ { m , Q _ { 0 } , h } \epsilon ^ { 2 }$ after increasing that constant, since the centre has a bounded reward density and $h$ is bounded. Taylor expansion of the centre equation gives (42). Bounded mean-zero scores are dense in H, so the displayed $\phi _ { m }$ is the canonical gradient in the full tangent space. Each of its components is itself bounded at fixed $m$

Let $h ^ { 1 } , \ldots , h ^ { r }$ be an orthonormal basis, in $\mathbb { H } .$ , for the span of the d scalar coordinate gradients. A finite linear combination of bounded gradients is bounded, so every $h _ { b } ^ { k }$ is bounded. Put

$$
\mathrm { d } Q _ { t , b } = \left( 1 + \sum _ { k = 1 } ^ { r } t _ { k } h _ { b } ^ { k } \right) \mathrm { d } Q _ { 0 , b }\tag{57}
$$

for t in a suficiently small neighbourhood of zero. Define

$$
A _ { u , i ; k } = \sum _ { b } \mathbb { E } _ { Q _ { 0 , b } } [ \phi _ { m , b } ( u , i ) h _ { b } ^ { k } ] .
$$

Expansion of the coordinate gradients in their orthonormal basis gives $A A ^ { \top } = V _ { m }$ . Direct diferentiation of the density shows that the information matrix is $\scriptstyle { { \cal I } _ { r } } .$ , and (42) gives the target derivative A.

For completeness, with

$$
\Delta _ { n , k } = \frac { 1 } { \sqrt { n } } \sum _ { j = 1 } ^ { n } \sum _ { b } h _ { b } ^ { k } ( X _ { j , b } ) ,
$$

the bounded-score Taylor expansion of $\log ( 1 + x )$ and the law of large numbers $\mathrm { g i v e }$ , uniformly on every fixed compact set of $\mathbf { \delta } _ { t , \tiny { \begin{array} { r l } \end{array} } } t ,$

$$
\log \frac { \mathrm { d } \mathbb { P } _ { Q _ { t / \sqrt { n } } } ^ { ( n ) } } { \mathrm { d } \mathbb { P } _ { Q _ { 0 } } ^ { ( n ) } } = t ^ { \top } \Delta _ { n } - \frac { 1 } { 2 } \| t \| _ { 2 } ^ { 2 } + o _ { \mathbb { P } _ { Q _ { 0 } } } ( 1 ) , \qquad \Delta _ { n } \overset { d } { \to } \mathsf { N } ( 0 , I _ { r } ) .\tag{58}
$$

Also

$$
\begin{array} { r } { \sqrt { n } \{ \theta _ { m } ( Q _ { t / \sqrt { n } } ) - \theta _ { m } ( Q _ { 0 } ) \} = A t + o ( 1 ) , \quad n d _ { H } ( Q _ { t / \sqrt { n } } , Q _ { 0 } ) ^ { 2 } = \frac { 1 } { 4 } \| t \| _ { 2 } ^ { 2 } + o ( 1 ) . } \end{array}
$$

These expansions are uniform on compact sets. The scores in (57) are arbitrary joint functions of $( R , S ^ { \prime } )$ ; no additive restriction is present. □

Proof of Lemma $ 4 . 1 0 .$ The decision $\delta ( Y ) = A Y$ has constant risk $R _ { G } : = \mathbb { E } L ( A Z )$ . For a lower bound give t the prior ${ \sf N } ( 0 , \tau ^ { 2 } I _ { r } )$ . Its conditional law given Y has mean vY and covariance $v { \cal I } _ { r }$ where $v = \tau ^ { 2 } / ( 1 + \tau ^ { 2 } )$ . A symmetric random vector U and convex even loss satisfy

$$
\begin{array} { r } { \mathbb { E } L ( U - a ) = \frac { 1 } { 2 } \mathbb { E } \{ L ( U - a ) + L ( U + a ) \} \geqslant \mathbb { E } L ( U ) . } \end{array}
$$

Thus the posterior mean of At minimizes posterior norm loss, and the Bayes risk is $\sqrt { v } R _ { G }$ , tending to $R _ { G } { \mathrm { ~ a s ~ } } \tau \to \infty$

We justify finite-prior approximation, which will be used below. For a prior supported on $\| t \| _ { 2 } \leqslant R _ { \ast }$ , all targets have norm at most $M _ { R } = R \| A \| _ { 2 \to L }$ . Any action with norm greater than $2 M _ { R }$ can be replaced by zero without increasing its loss at any target in this set. Therefore actions may be restricted to the compact bal $L ( a ) \leqslant 2 M _ { R }$ , and the loss is bounded by $3 M _ { R }$

Fix τ and truncate the Gaussian prior to this parameter ball. Let $p _ { R } = \mathbb { P } ( \| t \| _ { 2 } \leqslant R )$ and let $B _ { R }$ and $B _ { \tau }$ denote the truncated and full Gaussian-prior Bayes risks. An optimal truncated-prior decision can be chosen with norm at most $2 M _ { R }$ . Evaluating that decision under the full prior gives

$$
B _ { \tau } \leqslant p _ { R } B _ { R } + 2 M _ { R } \mathbb { P } ( \| t \| _ { 2 } > R ) + \mathbb { E } [ L ( A t ) \mathbb { 1 } \{ \| t \| _ { 2 } > R \} ] .
$$

The last two terms tend to zero and $p _ { R } \to 1$ , so lim inf $\dot { \phantom { } _ { R } } \mathrm { \mathop { : } } _ { { \cal R } } \mathrm { \mathop {  } { \infty } } B _ { R } \mathrm { \\mathop { \geq } { \cal B } } _ { \tau }$ . Thus compactly supported priors have Bayes risks arbitrarily close from below to the full Gaussian-prior value. Finally quantize a compact prior on a finite mesh. Gaussian shift laws are continuous in total variation as their means vary, and $L ( A t - a )$ is Lipschitz in t, uniformly over the restricted action ball. These two facts show that Bayes risks converge under mesh refinement. First take τ large, then R large, and then a fine mesh. □

## A.5 Proofs of Lemmas 4.11–4.12

Proof of Lemma 4.11. At the fixed product-form centre, let $\mu _ { s , a } = \mathbb { E } R ^ { s , a }$ and choose the same bounded score direction for all resolutions,

$$
h _ { s , a } ( r , s ^ { \prime } ) = r - \mu _ { s , a } .
$$

It belongs to the full tangent space and has a fixed finite positive squared norm

$$
\| h \| _ { \mathbb H } ^ { 2 } = \sum _ { s , a } \mathsf { V a r } ( R ^ { s , a } ) .
$$

Define, on r0, 1s,

$$
b _ { s , a } ( \boldsymbol { x } ) = - \int _ { 0 } ^ { \boldsymbol { x } } ( r - \mu _ { s , a } ) p _ { s , a } ( r ) \mathrm { d } r ,
$$

and extend it by zero outside. It is nonnegative, strictly positive on p0, 1q, and globally Lipschitz with constant at most $C _ { 0 }$ . Indeed, $\begin{array} { r } { \int _ { 0 } ^ { x } ( r - \mu ) p ( r ) \mathrm { d } r = F ( x ) ( \mathbb { E } [ R \mid R \leqslant x ] - \mu ) < 0 } \end{array}$ for $0 < x < 1$ since both sides of x have positive probability.

Put

$$
b _ { m , s } ( \boldsymbol { x } ) = \sum _ { a , s ^ { \prime } } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) \int b _ { s , a } ( \boldsymbol { x } - \gamma \boldsymbol { y } ) \eta _ { m } ( s ^ { \prime } ) ( \mathrm { d } \boldsymbol { y } ) , \qquad \boldsymbol { u } _ { m } ( s , i ) = b _ { m , s } ( \theta _ { m } ( s , i ) ) .
$$

The derivative formula (42) reads

$$
\begin{array} { r } { \dot { \pmb \theta } _ { m } [ h ] = ( { \pmb I } - \gamma { \pmb K } _ { m } ) ^ { - 1 } { \pmb D } _ { m } ^ { - 1 } { \pmb u } _ { m } \geqslant { \pmb D } _ { m } ^ { - 1 } { \pmb u } _ { m } \geqslant 0 , } \end{array}\tag{59}
$$

where the inequalities are coordinatewise and follow from the nonnegative Neumann series. In fact every $u _ { m } ( s , i ) > 0$ : positivity of the target density implies that at least one positively weighted reward argument lies strictly inside p0, 1q, where $b _ { s , a } > 0$

Replace $\eta _ { m }$ by $\eta ^ { \pi }$ in the definition of $b _ { m , s }$ and call the result $b _ { \infty , s }$ . Each $b _ { \infty , s }$ is continuous and positive on $( 0 , B )$ . For any x in this interval, the set of continuation values y for which $x - \gamma y \in ( 0 , 1 )$ contains a nonempty open interval in $( 0 , B )$ , and every relevant continuation return law has full support there. Lipschitz continuity and the deterministic approximation in Lemma B.1 imply

$$
\operatorname* { s u p } _ { s , x } | b _ { m , s } ( x ) - b _ { \infty , s } ( x ) | \leqslant C _ { Q _ { 0 } } / m .
$$

Fix one state $s _ { 0 }$ and choose $0 < \ell < B / 4$ such that $\pmb { \eta } ^ { \pi } ( s _ { 0 } ) ( [ 2 \ell , B - 2 \ell ] ) > 0$ . Then, for all large m, a fixed fraction $\alpha > 0$ of the coordinates $\theta _ { m } ( s _ { 0 } , i )$ belong to $[ \ell , B - \ell ]$ . To see this, couple the projected and true laws optimally in $W _ { 1 }$ : if $X \in [ 2 \ell , B - 2 \ell ]$ and $Y \notin [ \ell , B - \ell ]$ , then $\left| X - Y \right| \geqslant \ell$ Markov’s inequality under that coupling gives

$$
\frac { 1 } { m } \# \{ i : \theta _ { m } ( s _ { 0 } , i ) \in [ \ell , B - \ell ] \} \geqslant \eta ^ { \pi } ( s _ { 0 } ) ( [ 2 \ell , B - 2 \ell ] ) - \frac { W _ { 1 } ( \eta _ { m } ( s _ { 0 } ) , \eta ^ { \pi } ( s _ { 0 } ) ) } { \ell } .
$$

The right side is eventually bounded below by a fixed positive $\alpha$

On that compact interval, min<sub>s</sub> $b _ { \infty , s } > 0$ , so $u _ { m } ( s _ { 0 } , i ) \geqslant \beta > 0$ at those coordinates, for all large m. Since every entry of $D _ { m }$ is at most $C _ { 0 } .$ , equation (59) implies

$$
L _ { s _ { 0 } } ( \dot { \theta } _ { m } [ h ] ) \geqslant \alpha \beta / C _ { 0 } , \qquad L _ { s } ( z ) : = \frac { 1 } { m } \sum _ { i } z ( s , i ) .
$$

The same expression is strictly positive at each of the finitely many remaining resolutions, so its infimum over all m is a positive constant $a _ { Q _ { 0 } }$

By the canonical-gradient identity and Cauchy–Schwarz,

$$
a _ { Q _ { 0 } } \leqslant \left| \sum _ { b } \mathbb { E } _ { Q _ { 0 , b } } [ L _ { s _ { 0 } } ( \phi _ { m , b } ) h _ { b } ] \right| \leqslant \sqrt { \mathsf { V a r } ( L _ { s _ { 0 } } ( Z _ { m } ) ) } \| h \| _ { \mathbb { H } } .
$$

Finally $| L _ { s _ { 0 } } ( z ) | \leqslant \ell _ { m , 1 } ( z )$ , and the absolute first moment of a centred Gaussian is $\scriptstyle { \sqrt { 2 / \pi } }$ times its standard deviation. Hence

$$
\mathfrak { C } _ { m , 1 } ( Q _ { 0 } ) \geqslant \mathbb { E } | L _ { s _ { 0 } } ( Z _ { m } ) | \geqslant \sqrt { \frac { 2 } { \pi } } \frac { a _ { Q _ { 0 } } } { \| h \| _ { \mathbb { H } } } > 0 \qquad ( m \geqslant 1 ) .
$$

Proof of Lemma $4 . 1 2 .$ Take the shelf density from Lemma B.20 and start the downstream reward with the Betap2, 2q density. Lemma B.21 supplies one pair of perturbed densities satisfying Assumptions 1, 2, and 4 for every resolution. The source density still satisfies (95)–(96), and the downstream density remains within fixed positive constant factors of $6 r ( 1 - r )$ . Fix this pair once and for all, and denote the resulting M $\mathrm { R P }$ by $Q _ { * }$ . In this proof, bare $c , C , C ^ { \prime } , K$ are constants tied to that single fixed instance and independent of $j , m , n , p$

Let ν be the downstream return law and $\nu _ { m }$ the downstream projected fixed point. Lemma B.22 supplies the resolution sequence $m _ { j }$ and the first source-state quantile

$$
q _ { j } = \theta _ { m _ { j } } ( s , 1 ) \in [ x _ { j } / 2 , x _ { j } ] , \qquad \log m _ { j } = e ^ { j ^ { 2 } } + O ( j ^ { 2 } ) .
$$

All probability laws here and below are the final perturbed laws. We now compute a source-block contribution to the eficient variance.

For fixed $j ,$ define

$$
g _ { j } ( r ) : = \nu _ { m _ { j } } \left( \left[ 0 , \frac { q _ { j } - r } { \gamma } \right] \right) .
$$

For $R _ { s }$ drawn from the fixed source reward law,

$$
\mathbb { E } g _ { j } ( R _ { s } ) = \frac { 1 } { 2 m _ { j } } .
$$

If $r \in [ x _ { j } / 4 , 3 x _ { j } / 8 ]$ , (103) gives

$$
g _ { j } ( r ) \geqslant \nu _ { m _ { j } } ( [ 0 , x _ { j } / 4 ] ) .
$$

By (97) and (98), using the elementary implication

$$
F _ { \mu } ( x ) \geqslant F _ { \nu } ( x - h ) - \frac { W _ { 1 } ( \mu , \nu ) } { h } ,
$$

with $x = x _ { j } / 4$ and $h = x _ { j } / 8$ , we obtain

$$
\nu _ { m _ { j } } ( [ 0 , x _ { j } / 4 ] ) \geqslant \nu ( [ 0 , x _ { j } / 8 ] ) - \frac { C } { m _ { j } x _ { j } } .
$$

Here $1 / ( m _ { j } x _ { j } ) \leqslant C a _ { j } / x _ { j } \leqslant C \epsilon _ { j }$ , whereas the first term is at least $e ^ { - C j ^ { 2 } }$ . Since $\epsilon _ { j } = e ^ { - e ^ { j ^ { 2 } } }$ , the subtraction is negligible, and after changing $C _ { i }$

$$
\nu _ { m _ { j } } ( [ 0 , x _ { j } / 4 ] ) \geqslant e ^ { - C j ^ { 2 } }
$$

for all suficiently large $j .$ Since the reward interval $[ x _ { j } / 4 , 3 x _ { j } / 8 ]$ has probability at least $c \epsilon _ { j } x _ { j }$ and $( \mathbb { E } g _ { j } ) ^ { 2 } = O ( \epsilon _ { j } ^ { 2 } x _ { j } ^ { 2 } ) = o ( \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } } )$

$$
\mathsf { V a r } ( g _ { j } ( R _ { s } ) ) \geqslant c \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } } .\tag{60}
$$

The target density at $q _ { j }$ is bounded above by

$$
f _ { m _ { j } , s } ( q _ { j } ) = \mathbb { E } p _ { s } ( q _ { j } - \gamma Z _ { m _ { j } } ) \leqslant C \epsilon _ { j } .\tag{61}
$$

At this deterministic-transition centre the equations at t depend only on the state-t coordinates, while the equation for ps, iq depends on its own coordinate and the state-t vector. Consequently the source-block component of the canonical gradient in (24) is exactly

$$
[ \phi _ { m _ { j } , ( s ) } ( R _ { s } , t ) ] _ { s , 1 } = - \frac { g _ { j } ( R _ { s } ) - 1 / ( 2 m _ { j } ) } { f _ { m _ { j } , s } ( q _ { j } ) } .
$$

The covariance $V _ { m _ { j } }$ is a sum over independent sampling blocks. The state-t contribution is nonnegative on the diagonal and cannot cancel this source contribution. Thus, defining $\sigma _ { j } ^ { 2 } =$ $[ V _ { m _ { j } } ] _ { ( s , 1 ) , ( s , 1 ) }$ , we have

$$
\begin{array} { c } { { \sigma _ { j } ^ { 2 } \geqslant \displaystyle \frac { \mathsf { V a r } ( g _ { j } ( R _ { s } ) ) } { f _ { m _ { j } , s } ( q _ { j } ) ^ { 2 } } } } \\ { { \geqslant c \displaystyle \frac { x _ { j } } { \epsilon _ { j } } e ^ { - C j ^ { 2 } } . } } \end{array}\tag{62}
$$

From (99),

$$
m _ { j } \leqslant \frac { C e ^ { C j ^ { 2 } } } { \epsilon _ { j } x _ { j } } .
$$

Combining with (62), and absorbing $x _ { j } ^ { 2 } = 8 ^ { - 2 j }$ into the exponential, gives

$$
\sigma _ { j } ^ { 2 } \geqslant c m _ { j } e ^ { - C ^ { \prime } j ^ { 2 } } .
$$

Equation (100) therefore implies

$$
\sigma _ { j } ^ { 2 } \geqslant \frac { c m _ { j } } { ( \log m _ { j } ) ^ { K } }\tag{63}
$$

for a fixed finite $K .$

The last inequality is precisely the asserted diagonal covariance bound. The MRP and density perturbations were fixed before selecting $m _ { j }$ , so no part of the centre depends on the resolution or sample size. □

## B Technical Lemmas

## B.1 Basic Geometry and Paired Empirical Laws

This subsection supplies the background propositions from Section 2 and the elementary ingredients used in Appendix A. The scalar empirical-process bounds below depend on the table sizes only where those sizes are displayed explicitly.

## B.1.1 Bellman contraction and return densities

Proof of Proposition 2.1. For two continuation laws, couple the action, reward, and successor state identically, and couple the two continuation values optimally in the chosen Wasserstein distance conditional on the successor state. Multiplication by $\gamma$ contracts each displacement by that factor. Taking the state maximum proves the contraction of $\mathcal { T } ^ { \pi }$ . For $\Pi _ { m }$ , each pair of selected midpoint quantiles difers by at most the $W _ { \infty }$ distance of the input laws. Taking the maximum over the selected quantiles and states proves nonexpansiveness. These are the contraction properties recorded in Bellemare et al. [3]. □

Proof of Proposition 2.2. Conditioning on the first action and successor state, and then on a continuation return $Y ,$ shows that the return law has density

$$
p _ { \eta ^ { \pi } ( s ) } ( x ) = \sum _ { a , s ^ { \prime } } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) \int p _ { s , a } ( x - \gamma y ) \eta ^ { \pi } ( s ^ { \prime } ) ( \mathrm { d } y ) .
$$

This density is at most $C _ { 0 }$ . Every open interval in $( 0 , B )$ has positive return probability: choose a finite path with positive policy and transition probabilities, require the rewards along it to lie in suficiently small intervals around a suitable common point of $( 0 , 1 )$ , and choose the path length so that the remaining discounted tail is smaller than the desired interval. Assumption 1 makes the finite reward event have positive probability. Thus every continuation return law has full support r0, Bs.

For each $x \in ( 0 , B )$ , the set of $y \in ( 0 , B )$ satisfying $x - \gamma y \in ( 0 , 1 )$ contains a nonempty open interval. Positivity of the reward density and full continuation support therefore make the displayed density positive. To prove continuity, every continuation law already has a density bounded by $C _ { 0 }$ . For a zero-extended reward density $p ,$ the diference of its convolution terms at $x + h$ and x is bounded by

$$
\frac { C _ { 0 } } { \gamma } \int _ { \mathbb { R } } \left| p ( r + h ) - p ( r ) \right| \mathrm { d } r ,
$$

which tends to zero by translation continuity in $L ^ { 1 }$ . Finally, at $x = 0$ or $B .$ , the integrand can be nonzero only at an endpoint of a continuation support, a probability-zero event. This proves the zero endpoint values and the proposition. □

## B.1.2 Fixed-point geometry and midpoint approximation

Lemma B.1 (Contraction, approximation, and target geometry). Under Assumption 1, $\mathcal { T } ^ { \pi }$ is a γ-contraction in every $\overline { { W } } _ { p } ,$ , and $\Phi _ { m } , { \hat { \Phi } } _ { m , n }$ are γ-contractions in the parameter supremum norm. Their fixed points exist and are unique; the empirical fixed point is measurable in the data. Moreover,

$$
\overline { { { W } } } _ { 1 } ( \pmb { \eta } _ { m } , \pmb { \eta } ^ { \pi } ) \leqslant \frac { B } { 2 m ( 1 - \gamma ) } .\tag{64}
$$

For each fixed $\ell \in ( 0 , B / 2 )$ , under Assumption 3 or $^ { 4 , }$ the densities of $[ \mathcal { T } ^ { \pi } \eta _ { m } ] ( s )$ have a common positive lower bound on $[ \ell , B - \ell ]$ for all suficiently large $m$

Proof. Couple the action, reward and next state identically, and use an optimal coupling of the continuation distributions at that next state. This proves the Wasserstein contraction. For ordered arrays, couple the quantile index identically; the target values difer by at most $\gamma \| \pmb \theta - \pmb \vartheta \| _ { \infty }$ Monotonicity of generalized inverses transfers this bound to every midpoint quantile. The parameter space $\Theta _ { m }$ is complete and invariant, since $0 \leqslant R + \gamma \theta ( s ^ { \prime } , j ) \leqslant 1 + \gamma B = B$ . Banach’s theorem applies to both parameter maps. The $W _ { 1 }$ contraction on the complete space $\Delta ( I ) ^ { s }$ also identifies the unique unprojected fixed point with the return law. Coupling the same $( R , S ^ { \prime } )$ proves these contraction statements for arbitrary joint laws as well; no conditional independence is needed for contraction.

For measurability, start empirical parameter iteration at a deterministic array in $\Theta _ { m } .$ . Each update is a finite weighted order statistic of measurable support locations and is therefore measurable, even when some locations coincide. Geometric convergence makes its limit measurable. All resolutions are countable, so their error suprema are measurable. The population map $Q \mapsto \pmb { \theta } _ { m } ( Q )$ is likewise Borel: fixed-input CDF evaluations are measurable, a lower inverse can be expressed through rational thresholds, and the fixed point is the limit of measurable iterations.

For any distribution $\mu$ on $I ,$ its quantile function is nondecreasing. On each grid cell, the integral of the absolute diference from its midpoint value is at most half the cell length times the oscillation on that cell. Summing gives

$$
W _ { 1 } ( \Pi _ { m } \mu , \mu ) \leqslant B / ( 2 m ) .
$$

Apply this at $\mu = [ \tau ^ { \pi } \eta _ { m } ] ( s )$ and use the $W _ { 1 }$ contraction to obtain (64).

It remains to transfer interior positivity to projected inputs. If ν has density at most $C _ { 0 }$ , then

$$
\mathrm { K S } ( \mu , \nu ) \leqslant { \sqrt { 2 C _ { 0 } W _ { 1 } ( \mu , \nu ) } } .
$$

Indeed a CDF discrepancy of size d forces an area of at least $d ^ { 2 } / ( 2 C _ { 0 } )$ under the absolute CDF diference. A zero-extended reward density, continuous on r0, 1s, can be uniformly approximated within $\beta$ by a bounded-variation function, retaining any endpoint jumps at their original locations. Write its total variation as $V _ { \beta }$ . Integration against a diference of probability measures then gives, uniformly in the shift $x ,$

$$
\left| \int p _ { s , a } ( x - \gamma y ) \mathrm { d } ( \mu - \nu ) ( y ) \right| \leqslant 2 \beta + 2 V _ { \beta } \mathrm { K S } ( \mu , \nu ) .
$$

The factor two accommodates one-sided values at possible jumps. Combine this estimate with (64), first choosing $\beta$ and then $m$ . The target densities converge uniformly to the return densities, which are positive on the specified compact interval by Proposition 2.2. □

## B.1.3 Uniform joint-sub-CDF control

Lemma B.2 (Uniform joint-sub-CDF control). Let $J _ { s , a , s ^ { \prime } } , \widehat { J } _ { s , a , s ^ { \prime } }$ and $D _ { n }$ be as in (31). Let $N _ { \mathrm { t a b } } : = | S | ^ { 2 } | A |$ . Then, for every $t > 0$ ，

$$
\mathbb { P } ( D _ { n } > t ) \leqslant 2 N _ { \mathrm { t a b } } \exp \left( - \frac { 2 n t ^ { 2 } } { \vert \boldsymbol { S } \vert ^ { 2 } } \right) ,\tag{65}
$$

and

$$
\mathbb { E } D _ { n } ^ { 2 } \leqslant \frac { | { \cal S } | ^ { 2 } } { 2 n } \left\{ 1 + \log ( 2 N _ { \mathrm { t a b } } ) \right\} .\tag{66}
$$

Thus these concentration constants depend only on $| S |$ and $| { \cal A } |$ and not on the instance beyond the finite spaces. Pathwise, for every m, every ordered array θ, and every state s,

$$
\| \widehat { F } _ { s , \theta } ^ { ( n ) } - F _ { s , \theta } \| _ { \infty } \leqslant D _ { n } .
$$

Proof. For fixed $( s , a , s ^ { \prime } )$ encode a paired observation by $Y = R$ when $S ^ { \prime } = s ^ { \prime }$ and $Y = 2$ otherwise. On r0, 1s the ordinary CDF of Y equals ${ { J } _ { s , a , s ^ { \prime } } } ;$ outside this interval the sub-CDF error is either zero or its value at one. DKW [16] therefore controls each sub-CDF error by $2 e ^ { - 2 n t ^ { 2 } }$ . If

$$
M _ { n } : = \operatorname* { m a x } _ { s , a , s ^ { \prime } } \| \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } \| _ { \infty } ,
$$

then $D _ { n } \leqslant | S | M _ { n }$ because $\textstyle \sum _ { a } \pi ( a \mid s ) = 1$ . A union bound over the $N _ { \mathrm { t a b } } = | S | ^ { 2 } | \mathcal { A } |$ triples gives (65). Integrating $\mathbb { P } ( M _ { n } > t ) \leqslant \operatorname* { m i n } \{ 1 , 2 N _ { \mathrm { t a b } } e ^ { - 2 n t ^ { 2 } } \}$ yields $\mathbb { E } M _ { n } ^ { 2 } \leqslant \{ 1 + \log ( 2 N _ { \mathrm { t a b } } ) \} / ( 2 n )$ , hence (66). Finally,

$$
\widehat { F } _ { s , \theta } ^ { ( n ) } ( x ) - F _ { s , \theta } ( x ) = \sum _ { a , s ^ { \prime } } \frac { \pi ( a \mid s ) } { m } \sum _ { j } ( \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } ) ( x - \gamma \theta ( s ^ { \prime } , j ) ) .
$$

Take absolute values and use the definition of $D _ { n }$ . This bound is uniform even at data-dependent inputs. □

## B.1.4 Quantile graphs and finitely many resolutions

Lemma B.3 (Quantile graphs and the localization grid). Let F be continuous and let G be any CDF, with $\| G - F \| _ { \infty } \leqslant e . \ I f y = G ^ { - 1 } ( u ) , 0 < u < 1$ , then $| F ( y ) - u | \leqslant e$ . More generally the same conclusion holds whenever

$$
G ( y - ) \leqslant u \leqslant G ( y ) .
$$

If $x _ { 1 } \leqslant \cdots \leqslant x _ { m }$ and $| F ( x _ { i } ) - \tau _ { i } | \leqslant e ,$ , then

$$
\mathrm { K S } \left( \frac { 1 } { m } \sum _ { i } \delta _ { x _ { i } } , F \right) \leqslant e + \frac { 1 } { 2 m } .
$$

If both laws are supported on an interval of length B, their $W _ { 1 }$ distance is at most $B ( e + 1 / ( 2 m ) )$

Proof. The lower inverse has $G ( y - ) \leqslant u \leqslant G ( y )$ . Take a left limit in the uniform CDF bound and use continuity of F to obtain $u - e \leqslant F ( y ) = F ( y - ) \leqslant u + e$ . For the grid statement, let k be the number of coordinates at most $x ,$ counting repetitions. If $0 < k < m$ , then $x _ { k } \leqslant x < x _ { k + 1 }$ and

$$
( k - 1 / 2 ) / m - e \leqslant F ( x ) \leqslant ( k + 1 / 2 ) / m + e .
$$

The empirical CDF equals $k / m$ . For $k = 0 \ { \mathrm { o r } } \ k = m$ use the first or last grid coordinate and the bounds $0 \leqslant F \leqslant 1$ . Integration of the CDF diference proves the $W _ { 1 }$ statement. In particular no empirical jump-size or $1 / n$ rounding bound is needed in this localization argument. □

Lemma B.4 (Stability at finitely many resolutions). Suppose each selected population midpoint quantile has a positive continuous target density in a neighborhood. For every fixed $M < \infty$ , there is an instance-dependent $C _ { M , M } < \infty$ such that

$$
\operatorname* { m a x } _ { 1 \leqslant m \leqslant M } \overline { { W } } _ { \infty } ( \pmb { \eta } _ { m } ^ { ( n ) } , \pmb { \eta } _ { m } ) \leqslant C _ { M , \mathcal { M } } D _ { n } .
$$

Under the standing Assumptions $\ : 1 - 2 , \ :$ the required positivity holds.

Proof. A continuous target CDF can have zero density at its lower quantile only if this point is an endpoint of one of the finitely many shifted reward-support intervals: if it were in the interior of an active interval, Assumption 1 would give positive density; if it were in an open gap, it could not be the lower inverse. Such an endpoint would violate Assumption 2. At the selected quantile all reward arguments avoid their endpoints, so the density is also continuous nearby.

For the finitely many selected quantiles choose instance-dependent neighborhoods with common lower density $c _ { M , \mathscr { M } } > 0$ and probability margins $d _ { M , M } > 0$ . When $D _ { n } \leqslant d _ { M , M }$ , generalized-inverse bracketing and Lemma B.2 give

$$
\| \widehat { \Phi } _ { m , n } ( \pmb { \theta } _ { m } ) - \Phi _ { m } ( \pmb { \theta } _ { m } ) \| _ { \infty } \leqslant D _ { n } / c _ { M , \mathcal { M } } .
$$

The empirical contraction gives the fixed-point bound with factor $( 1 - \gamma ) ^ { - 1 }$ . If $D _ { n } > d _ { M , M }$ , use the support bound $B \leqslant ( B / d _ { M , \mathcal { M } } ) D _ { n }$ □

## B.1.5 Stratifying the paired empirical law

Let

$$
\mathcal { T } : = \{ ( s , a , s ^ { \prime } ) : P ( s ^ { \prime } \mid s , a ) > 0 \} , \qquad p _ { * } : = \operatorname* { m i n } _ { ( s , a , s ^ { \prime } ) \in \mathcal { T } } P ( s ^ { \prime } \mid s , a ) > 0 .
$$

For $( s , a , s ^ { \prime } ) \in \mathcal { T }$ , define

$$
N _ { s , a , s ^ { \prime } } : = \sum _ { k = 1 } ^ { n } \mathbb { 1 } \{ S _ { k } ^ { \prime , s , a } = s ^ { \prime } \} , \qquad \widehat { P } ( s ^ { \prime } \mid s , a ) = \frac { N _ { s , a , s ^ { \prime } } } { n } .
$$

Whenever $N _ { s , a , s ^ { \prime } } > 0 .$ , let $\widehat { F } _ { s , a , s ^ { \prime } }$ be the empirical CDF of those rewards $R _ { k } ^ { s , a }$ whose paired next state is $s ^ { \prime }$ . Conditional on $N _ { s , a , s ^ { \prime } }$ , these rewards are i.i.d. from the same population reward CDF $F _ { s , a }$ because reward and next state are independent conditional on $( s , a )$

Define

$$
d _ { n } : = \operatorname* { m a x } _ { ( s , a , s ^ { \prime } ) \in \mathcal { I } } \left. \widehat { F } _ { s , a , s ^ { \prime } } - F _ { s , a } \right. _ { \infty } , \qquad r _ { n } : = \operatorname* { m a x } _ { ( s , a , s ^ { \prime } ) \in \mathcal { I } } \left. \frac { \widehat { P } ( s ^ { \prime } \mid s , a ) } { P ( s ^ { \prime } \mid s , a ) } - 1 \right. .
$$

If $N _ { s , a , s ^ { \prime } } = 0$ , define $\widehat { F } _ { s , a , s ^ { \prime } }$ to be the CDF of a point mass at zero. This convention defines $d _ { n }$ on the whole sample space. Write

$$
\mathcal { C } _ { n } : = \{ r _ { n } \leqslant 1 / 2 \} .
$$

On ${ \mathcal { C } } _ { n }$ every positive-probability stratum is nonempty.

Lemma B.5 (Uniform concentration of the two primitive errors). For every $\delta \in ( 0 , 1 )$ and every $n \geqslant 1$

$$
\mathbb { P } \left( d _ { n } + r _ { n } \leqslant \frac { 5 } { p _ { * } } \sqrt { \frac { \log ( 2 | \mathcal { T } | / \delta ) } { 2 n } } \right) \geqslant 1 - \delta .\tag{67}
$$

Moreover,

$$
\mathbb { E } ( d _ { n } + r _ { n } ) \leqslant \frac { 5 } { p _ { * } } \sqrt { \frac { 1 + \log ( 2 | \mathcal { T } | ) } { 2 n } } .\tag{68}
$$

Proof. Let

$$
M _ { n } ^ { + } : = \operatorname* { m a x } _ { ( s , a , s ^ { \prime } ) \in \mathcal { T } } \| \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } \| _ { \infty } .
$$

For every positive-probability stratum, evaluation at $x = 1$ gives

$$
| \widehat { P } ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , a ) | \leqslant M _ { n } ^ { + } , \qquad r _ { n } \leqslant M _ { n } ^ { + } / p _ { * } .
$$

If $M _ { n } ^ { + } \leqslant p _ { * } / 2$ , every such stratum is nonempty and $\hat { P } ( s ^ { \prime } \mid s , a ) \geqslant P ( s ^ { \prime } \mid s , a ) / 2$ . Using $J _ { s , a , s ^ { \prime } } =$ $P ( s ^ { \prime } \mid s , a ) F _ { s , a }$ and $\widehat { F } _ { s , a , s ^ { \prime } } = { \widehat { J } } _ { s , a , s ^ { \prime } } / { \widehat { P } } ( s ^ { \prime } \mid s , a )$ , we obtain

$$
\begin{array} { r l } & { \| \widehat { F } _ { s , a , s ^ { \prime } } - F _ { s , a } \| _ { \infty } \leqslant \frac { \| \widehat { J } _ { s , a , s ^ { \prime } } - J _ { s , a , s ^ { \prime } } \| _ { \infty } + | \widehat { P } ( s ^ { \prime } \mid s , a ) - P ( s ^ { \prime } \mid s , a ) | } { \widehat { P } ( s ^ { \prime } \mid s , a ) } } \\ & { \qquad \leqslant \frac { 4 M _ { n } ^ { + } } { P ( s ^ { \prime } \mid s , a ) } \leqslant \frac { 4 M _ { n } ^ { + } } { p _ { * } } . } \end{array}
$$

If $M _ { n } ^ { + } > p _ { * } / 2$ , the bound $d _ { n } \leqslant 1 \leqslant 2 M _ { n } ^ { + } / p _ { * }$ holds, including for empty strata under the stated convention. Consequently, on the whole sample space,

$$
d _ { n } + r _ { n } \leqslant \frac { 5 M _ { n } ^ { + } } { p _ { * } } .\tag{69}
$$

The sub-CDF DKW argument in Lemma B.2, with a union bound only over $\mathcal { T } ,$ gives

$$
\begin{array} { r } { \mathbb { P } ( M _ { n } ^ { + } > x ) \leqslant 2 | \mathcal { T } | e ^ { - 2 n x ^ { 2 } } , \qquad x > 0 . } \end{array}
$$

Combining (69) with this tail bound at $x = \sqrt { \log ( 2 | \mathcal { I } | / \delta ) / ( 2 n ) }$ proves (67). For the expectation bound, integrating the minimum of one and the tail bound yields

$$
\mathbb { E } [ ( M _ { n } ^ { + } ) ^ { 2 } ] \leqslant \frac { 1 + \log ( 2 | \mathcal { T } | ) } { 2 n } .
$$

Finally, (69) and Cauchy–Schwarz imply (68).

On the probability-one event that a population-zero transition cell is never sampled, the empirical Bellman CDF has the exact representation

$$
\widehat { F } _ { s , \theta } ^ { ( n ) } ( x ) = \sum _ { a , s ^ { \prime } , j } \frac { \pi ( a \mid s ) \widehat { P } ( s ^ { \prime } \mid s , a ) } { m } \widehat { F } _ { s , a , s ^ { \prime } } ( x - \gamma \theta ( s ^ { \prime } , j ) ) .\tag{70}
$$

Every term with an empty stratum has zero empirical transition weight. Conditional on the full array of successor-state labels, the reward observations in each nonempty stratum are independent with CDF $F _ { s , a }$ at the product-form centre. Conditioning here is on labels, not on an event that also depends on rewards.

## B.1.6 Reward inversion and relative mixture weights

Lemma B.6 (CDF error to reward $W _ { \infty }$ error under Assumption 3). Suppose F has density bounded below by $c _ { 0 }$ on r0, 1s and G is any CDF supported on r0, 1s. If

$$
\| F - G \| _ { \infty } \leqslant d ,
$$

then

$$
W _ { \infty } ( F , G ) \leqslant \frac { d } { c _ { 0 } } .
$$

Proof. Let $q = F ^ { - 1 }$ and $\widehat { q } = G ^ { - 1 }$ . Generalized-inverse bracketing gives

$$
q ( ( u - d ) \lor 0 ) \leqslant { \widehat { q } } ( u ) \leqslant q ( ( u + d ) \land 1 ) .
$$

Since $F ( y ) - F ( x ) \geqslant c _ { 0 } ( y - x )$ for $0 \leqslant x \leqslant y \leqslant 1$ , the population quantile function is $1 / c _ { 0 } \mathrm { - L i p s c h i t z }$ in probability level, including the endpoint truncations. Hence

$$
\operatorname* { s u p } _ { u \in ( 0 , 1 ) } | \widehat { q } ( u ) - q ( u ) | \leqslant d / c _ { 0 } .
$$

□

Lemma B.7 (Relative mixture-weight error). Let

$$
F = \sum _ { k } w _ { k } { \cal F } _ { k } , \qquad { \widetilde F } = \sum _ { k } { \widetilde w } _ { k } { \cal F } _ { k } ,
$$

where $w _ { k } , \widetilde { w } _ { k } \geqslant 0$ , both weight vectors sum to one, and

$$
\widetilde { w } _ { k } = 0 \quad w h e n e v e r \ : w _ { k } = 0 .
$$

Assume also that

$$
| \widetilde { w } _ { k } / w _ { k } - 1 | \leqslant r < 1
$$

whenever $w _ { k } > 0$ . Then

$$
\left| \widetilde { F } ( x ) - F ( x ) \right| \leqslant r \operatorname* { m i n } \{ F ( x ) , 1 - F ( x ) \}\tag{71}
$$

for every x.

Proof. The direct bound is

$$
\left| \widetilde { F } - F \right| \leqslant \sum _ { k } \left| \widetilde { w } _ { k } - w _ { k } \right| F _ { k } \leqslant r \sum _ { k } w _ { k } F _ { k } = r F .
$$

Since $\begin{array} { r } { \sum _ { k } ( \widetilde { w } _ { k } - w _ { k } ) = 0 } \end{array}$ , the same argument applied to $1 - F _ { k }$ gives

$$
\left| \widetilde F - F \right| \leqslant r ( 1 - F ) .
$$

## B.2 Density-Weighted Empirical-CDF Bounds

Only reward distributions are considered in this subsection. The resulting bounds are subsequently inserted into the exact stratum mixture (70). The untruncated expected supremum is an auxiliary empirical-process estimate; it is not a mean-square estimate for the Bellman fixed point.

## B.2.1 A decreasing-weight maximal inequality

Lemma B.8 (Decreasing-weight Doob inequality). Let $( M _ { u } ) _ { 0 \leqslant u \leqslant T }$ be a square-integrable càdlàg martingale with $M _ { 0 } = 0$ , and let $w : ( 0 , T ] \to [ 0 , \infty )$ be continuous and nonincreasing. Then, allowing either side to be infinite,

$$
\mathbb { E } \operatorname* { s u p } _ { 0 < u \leqslant T } w ( u ) M _ { u } ^ { 2 } \leqslant 4 \int _ { ( 0 , T ] } w ( u ) \mathrm { d } \mathbb { E } M _ { u } ^ { 2 } .
$$

Proof. First suppose w is bounded, extended at zero by its right limit, and put $\begin{array} { r } { M _ { u } ^ { * } = \operatorname* { s u p } _ { 0 \leqslant v \leqslant u } | M _ { v } | } \end{array}$ For $u \leqslant T$

$$
w ( u ) = w ( T ) + \int _ { ( u , T ] } ( - \mathrm { d } w ( v ) ) .
$$

Since $M _ { u } ^ { 2 } \leqslant ( M _ { v } ^ { * } ) ^ { 2 }$ for $v \geqslant u$ and ´dw is a positive measure,

$$
\operatorname* { s u p } _ { u \leqslant T } w ( u ) M _ { u } ^ { 2 } \leqslant w ( T ) ( M _ { T } ^ { * } ) ^ { 2 } + \int _ { ( 0 , T ] } ( M _ { v } ^ { * } ) ^ { 2 } ( - \mathrm { d } w ( v ) ) .
$$

Tonelli’s theorem and Doob’s $L ^ { 2 }$ inequality give an expectation bound by

$$
4 \left\{ w ( T ) a ( T ) + \int _ { ( 0 , T ] } a ( v ) ( - \mathrm { d } w ( v ) ) \right\} , \qquad a ( v ) = \mathbb { E } M _ { v } ^ { 2 } .
$$

The function a is nondecreasing and $a ( 0 ) = 0$ . Stieltjes integration by parts, using continuity of w, makes the expression in braces equal to $\int _ { ( 0 , T ] }$ w da. For general w, apply this bounded-weight argument to $w _ { K } = w \wedge K ;$ its initial boundary term is $w _ { K } ( 0 ) a ( 0 ) = 0$ . Monotone convergence on both sides as $K  \infty$ proves the claim. The process $\sqrt { w } M$ is not assumed to be a martingale or a submartingale. □

## B.2.2 The expected density-weighted empirical error

Lemma B.9 (Weighted empirical-CDF maximal inequality). Let F be a distribution on $[ 0 , 1 ]$ with density f. Suppose f is continuous, positive on p0, 1q, nondecreasing on $[ 0 , \kappa ]$ , nonincreasing on $[ 1 - \kappa , 1 ]$ , and bounded away from zero on every compact subinterval of $( 0 , 1 )$ . For N i.i.d. samples with empirical $C D F F _ { N }$ , define

$$
Z _ { N } ( F ) : = \operatorname* { s u p } _ { x : f ( x ) > 0 } { \frac { | F _ { N } ( x ) - F ( x ) | } { \sqrt { f ( x ) } } } .
$$

Then

$$
\mathbb { E } Z _ { N } ( F ) ^ { 2 } \leqslant \frac { C ( F ) } { N } .\tag{72}
$$

For a finite collection $\mathcal { F }$ of such reward laws, the maximum of the corresponding $Z _ { N }$ ’s satisfies the same bound with a constant $C ( \mathcal { F } )$ depending only on that fixed collection.

Proof. By the probability integral transform it sufices to study the empirical CDF $U _ { N }$ of iid uniform variables $U _ { 1 } , \dots , U _ { N }$ and the weight $f ( q ( u ) )$ , where $q = F ^ { - 1 }$ . Choose $0 < \kappa _ { 0 } < \operatorname* { m i n } \{ \kappa , 1 / 2 \}$ and set $T = F ( \kappa _ { 0 } ) < 1$ . With the filtration generated by all indicators ${ \mid } \{ U _ { k } \leqslant v \}$ for $v \leqslant u ,$ , define

$$
M _ { u } = \frac { U _ { N } ( u ) - u } { 1 - u } , \qquad 0 \leqslant u \leqslant T .
$$

This is a martingale: conditional on $U _ { k } \ > \ v$ , the remaining variable is uniform on $( v , 1 )$ , and conditional on $U _ { k } \leqslant v$ its indicator remains one. Direct calculation gives

$$
\mathbb { E } M _ { u } ^ { 2 } = \frac { u } { N ( 1 - u ) } , \qquad \mathrm { d } \mathbb { E } M _ { u } ^ { 2 } = \frac { \mathrm { d } u } { N ( 1 - u ) ^ { 2 } } .
$$

On the left boundary, the weight $w ( u ) = ( 1 - u ) ^ { 2 } / f ( q ( u ) )$ is nonincreasing because $f ( q ( u ) )$ is

nondecreasing. Lemma B.8 therefore gives

$$
\begin{array} { r l r } & { } & { \mathbb { E } \underset { 0 < u \leqslant T } { \operatorname* { s u p } } \frac { | U _ { N } ( u ) - u | ^ { 2 } } { f ( q ( u ) ) } \leqslant 4 \displaystyle \int _ { 0 } ^ { T } \frac { ( 1 - u ) ^ { 2 } } { f ( q ( u ) ) } \frac { \mathrm { d } u } { N ( 1 - u ) ^ { 2 } } } \\ & { } & { \quad \quad \quad \quad = \displaystyle \frac { 4 } { N } \displaystyle \int _ { 0 } ^ { T } \frac { \mathrm { d } u } { f ( q ( u ) ) } = \frac { 4 \kappa _ { 0 } } { N } . } \end{array}
$$

The last equality is the change of variables $u = F ( x )$ on the active reward support. Reflection gives the same bound for the right boundary. On $\left[ \kappa _ { 0 } , 1 - \kappa _ { 0 } \right]$ , let $c _ { F }$ be the positive minimum of $f .$ DKW gives

$$
\mathbb { E } \operatorname* { s u p } _ { x } | F _ { N } ( x ) - F ( x ) | ^ { 2 } \leqslant \frac { 1 } { N } ,
$$

for example by integrating $\begin{array} { r } { \mathbb { P } ( \operatorname* { s u p } _ { x } | F _ { N } - F | > a ) \leqslant 2 e ^ { - 2 N a ^ { 2 } } } \end{array}$ . Divide by $c _ { F }$ on that middle interval. The maximum over the three regions is bounded by their sum, proving (72). For a finite collection of laws, bound the squared maximum by the sum of the individual squared maxima; the constants then add. No independence between these maxima is needed.

## B.2.3 Exponential confidence by truncating the density weight

Lemma B.10 (A truncated density-weighted Bernstein envelope). Under the assumptions $o f$ Lemma $B . 9 ,$ there is a finite constant $C _ { F }$ such that, for every integer $N \geqslant 1$ and every $t \geqslant 1$ , with probability at least $1 - e ^ { - t }$ 2

$$
| F _ { N } ( x ) - F ( x ) | \leqslant C _ { F } \left\{ \sqrt { \frac { t f ( x ) } { N } } + \frac { t } { N } \right\} \qquad f o r \ e v e r y \ x \in [ 0 , 1 ] .\tag{73}
$$

The constant depends on $F _ { \cdot }$ , but not on N or t.

Proof. Let $\tau = t / N$ and put

$$
S _ { \tau } = \operatorname* { s u p } _ { x \in [ 0 , 1 ] } { \frac { | F _ { N } ( x ) - F ( x ) | } { \sqrt { f ( x ) \vee \tau } } } .
$$

By Lemma B.9 and Jensen’s inequality,

$$
\begin{array} { r } { \mathbb { E } S _ { \tau } \leqslant \left( \mathbb { E } Z _ { N } ( F ) ^ { 2 } \right) ^ { 1 / 2 } \leqslant C _ { F } N ^ { - 1 / 2 } . } \end{array}\tag{74}
$$

At an endpoint of zero density the empirical and population CDFs agree almost surely, so these points do not afect this comparison.

Boundary monotonicity gives $F ( x ) / f ( x )$ ď x near zero and $( 1 - F ( x ) ) / f ( x ) \leqslant 1 - x$ near one. On a fixed middle interval the density has a positive lower bound. Consequently,

$$
\sigma _ { F } ^ { 2 } : = \operatorname* { s u p } _ { 0 < x < 1 } { \frac { F ( x ) ( 1 - F ( x ) ) } { f ( x ) } } < \infty .\tag{75}
$$

Consider the centered, signed class

$$
\mathcal { H } _ { \tau } = \left\{ r \longmapsto \frac { \pm ( \mathbb { 1 } \{ r \leqslant x \} - F ( x ) ) } { \sqrt { f ( x ) \vee \tau } } : x \in [ 0 , 1 ] \right\} .
$$

Writing $P _ { N }$ for empirical averaging, the signed class satisfies $\mathbb { E } h ( R ) = 0$ for every $h \in { \mathcal { H } } .$ <sub>τ</sub> and

$$
S _ { \tau } = \operatorname* { s u p } _ { h \in \mathcal { H } _ { \tau } } P _ { N } h .
$$

Every function has absolute value at most $b = \tau ^ { - 1 / 2 }$ and variance at most $\sigma _ { F } ^ { 2 }$ . For each $x < 1$ , take rational thresholds $x _ { k } \downarrow x$ . Right continuity of $F _ { N }$ and $F ,$ and continuity and positivity of $\sqrt { f \vee \tau }$ 2 imply convergence of the corresponding normalized errors. Thus restricting to rational thresholds and the endpoint 1 leaves the supremum unchanged and makes the index set countable. Thus the usual countable-class form of Bousquet’s inequality applies; see, for example, Boucheron et al. [5]. Writing the inequality for the average rather than the sum gives

$$
\mathbb { P } \left\{ S _ { \tau } > \mathbb { E } S _ { \tau } + \sqrt { \frac { 2 ( \sigma _ { F } ^ { 2 } + 2 b \mathbb { E } S _ { \tau } ) t } { N } } + \frac { b t } { 3 N } \right\} \leqslant e ^ { - t } .
$$

The class is centred and signed, its envelope is $b ,$ its variance is at most $\sigma _ { F } ^ { 2 }$ , and its expected supremum is finite by (74). Bounding the cross term with $2 { \sqrt { x y } } \leqslant x + y$ yields

$$
\mathbb { P } \left\{ S _ { \tau } > 2 \mathbb { E } S _ { \tau } + C \left( \sqrt { \frac { \sigma _ { F } ^ { 2 } t } { N } } + \frac { b t } { N } \right) \right\} \leqslant e ^ { - t } ,\tag{76}
$$

where $C$ is universal. This form follows from the usual variance proxy $N \sigma _ { F } ^ { 2 } + 2 b N \mathbb { E } S _ { \imath }$ <sub>τ</sub> by absorbing the square-root cross term into $\mathbb { E } S _ { \tau }$ and $b t / N$

Since $t \geqslant 1$ and $b t / N = \sqrt { t / N }$ , equations $( 7 4 ) ‐ ( 7 6 )$ show that $S _ { \tau } \leqslant C _ { F } \sqrt { t / N }$ outside an event of probability $e ^ { - t }$ . On this event, multiplying by the truncated denominator gives, for every $x ,$

$$
| F _ { N } ( x ) - F ( x ) | \leqslant C _ { F } \sqrt { \frac { t } { N } } \sqrt { f ( x ) \vee \frac { t } { N } } \leqslant C _ { F } \left( \sqrt { \frac { t f ( x ) } { N } } + \frac { t } { N } \right) .
$$

This proves (73). It does not assert a sub-Gaussian tail for $Z _ { N } ( F )$ with its untruncated denominator. □

## B.3 Deterministic Quantile Inversion

This subsection proves the scalar ingredients used in the proof of Lemma 4.4. The density may have an initial or terminal interval on which it is zero. All inverse-density integrals below are over its active support. No diferentiability of the density and no regularity of the perturbed CDF G are required beyond the hypotheses stated in that lemma.

## B.3.1 Active support and quantile graphs

Lemma B.11 (Active support and graph conventions). Suppose f is a continuous probability density on $[ 0 , B ]$ , nondecreasing on $[ 0 , \ell ]$ , nonincreasing on $[ \boldsymbol { B } - \boldsymbol { \ell } , \boldsymbol { B } ]$ , and at least $c _ { \mathrm { i n t } } > 0$ on $[ \ell , B - \ell ]$ , where $0 < \ell < B / 2$ . There are $0 \leqslant a < b \leqslant B$ such that $f > 0$ on $( a , b )$ and $f = 0$ outside ra, bs. For $q = F ^ { - 1 }$

$$
q \in C ^ { 1 } ( ( 0 , 1 ) ) , \qquad q ^ { \prime } ( u ) = \frac { 1 } { f ( q ( u ) ) } , \qquad \int _ { 0 } ^ { 1 } q ^ { \prime } ( u ) \mathrm { d } u = b - a \leqslant B .
$$

For any CDF G, a graph selection y at level u satisfies $G ( y - ) \leqslant u \leqslant G ( y )$ , and hence $G ( x ) \leqslant u$ for $x < y$ and $G ( x ) \geqslant$ u for $x > y$ . If G is supported in $[ 0 , B ]$ , reflection gives

$$
G ^ { \mathrm { r e f } } ( x ) = 1 - G \big ( ( B - x ) - \big ) , \qquad G ^ { \mathrm { r e f } } ( ( B - y ) - ) \leqslant 1 - u \leqslant G ^ { \mathrm { r e f } } ( B - y ) .
$$

A continuous bound on $| G - F |$ also bounds $| G ( \cdots ) - F |$ , and therefore transfers to the reflected $C D F s$

Proof. A nonnegative nondecreasing function has an initial zero interval and is positive thereafter if it is positive at ℓ. The analogous right-end statement and the interior lower bound join the positive regions into $( a , b )$ . On every compact subinterval of $( a , b )$ the density has a positive minimum. The inverse-function theorem gives the stated derivative; integrate it first over $[ v , 1 - v ]$ and then let $v \downarrow 0$ to obtain $b - a .$ . The graph inequalities follow from monotonicity of G. The reflection formula follows from the left and right limits of a CDF. Finally, take left limits in the continuous error envelope to obtain the last assertion. In particular, reflection need not identify a lower inverse with a lower inverse: the graph inequalities apply to either endpoint of a flat probability level. □

## B.3.2 One-sided envelopes at shifted probability levels

Lemma B.12 (Shifted inverse-CDF envelopes). Let $F , f , G , z , \varepsilon$ satisfy the hypotheses of Lemma $4 . 4 ,$ with $0 \leqslant \varepsilon < 1 / 4$ . Put $q = F ^ { - 1 }$ and $\widehat { q } = G ^ { - 1 }$ . There are nonnegative left envelopes $H _ { L , + } , H _ { L , - }$ and right envelopes $H _ { R , + } , H _ { R , - } \ o n \ ( 0 , 1 )$ . The left envelopes are nonincreasing, the right envelopes are nondecreasing, and

$$
\int _ { 0 } ^ { 1 } ( H _ { L , + } + H _ { L , - } + H _ { R , + } + H _ { R , - } ) ( v ) { \mathrm { d } } v \leqslant 1 4 \ell z ^ { 2 } .\tag{77}
$$

For $\varepsilon < u < 1 - \varepsilon$ , set $v _ { + } = u \pm \varepsilon$ and $r ( u ) = q ( v _ { + } ) - q ( v _ { - } )$ . Then

$$
\begin{array} { r l r } & { } & { | \widehat { q } ( u ) - q ( u ) | ^ { 2 } \leqslant C _ { \mathrm { g e o m } } \big [ r ( u ) ^ { 2 } + z ^ { 2 } + H _ { L , + } ( v _ { + } ) + H _ { L , - } ( v _ { - } ) } \\ & { } & \\ & { } & { \qquad + H _ { R , + } ( v _ { + } ) + H _ { R , - } ( v _ { - } ) \big ] , \quad \quad } \end{array}\tag{78}
$$

where $C _ { \mathrm { g e o m } }$ depends only on $B , \ell , c _ { \mathrm { i n t } } , C _ { 0 }$ . The same bound holds for any graph selection of G at level u.

Proof. Let $g ( v ) = 1 / f ( q ( v ) )$ for $0 < v < 1$ . First suppose $z > 0$ . On the left define

$$
H _ { L , + } ( v ) = 4 z ^ { 2 } g ( v ) \mathbb { 1 } \{ v < F ( \ell ) \} .
$$

It is nonincreasing, and the change of variables $v = F ( x )$ on the active left support gives

$$
\int _ { 0 } ^ { 1 } H _ { L , + } ( v ) \mathrm { d } v = 4 z ^ { 2 } \int _ { 0 } ^ { F ( \ell ) } g ( v ) \mathrm { d } v \leqslant 4 \ell z ^ { 2 } .\tag{79}
$$

For the other displacement direction set $A _ { z } ( x ) = F ( x ) + z { \sqrt { f ( x ) } }$ on $[ 0 , \ell ]$ , and let $a _ { z } ( v )$ be its generalized inverse clipped at ℓ. The map $A _ { z }$ is continuous and nondecreasing; it is strictly increasing on the active left support. Define

$$
H _ { L , - } ( v ) = \operatorname* { m i n } \left\{ \ell ^ { 2 } , \frac { z ^ { 2 } } { f ( a _ { z } ( v ) ) } \right\} \mathbb { 1 } \{ 0 < v \leqslant A _ { z } ( \ell ) \} .
$$

This envelope is nonincreasing, including the terminal truncation. On an initial zero-density interval, $\mathrm { d } A _ { z } = 0 ;$ define the bounded integrand there by its limiting value. Stieltjes change of variables, followed by $\mathrm { d } A _ { z } = f \mathrm { d } x + z \mathrm { d } \sqrt { f }$ , gives

$$
\int _ { 0 } ^ { 1 } H _ { L , - } ( v ) { \mathrm { d } } v \leqslant \int _ { 0 } ^ { \ell } \operatorname* { m i n } \left\{ \ell ^ { 2 } , { \frac { z ^ { 2 } } { f ( x ) } } \right\} { \mathrm { d } } { \big ( } F ( x ) + z { \sqrt { f ( x ) } } { \big ) }
$$

$$
\leqslant \ell z ^ { 2 } + z \int _ { 0 } ^ { \infty } \operatorname* { m i n } \left\{ \ell ^ { 2 } , { \frac { z ^ { 2 } } { y ^ { 2 } } } \right\} { \mathrm { d } } y = 3 \ell z ^ { 2 } .\tag{80}
$$

The last integral is $2 \ell z$ , obtained by splitting at $y = z / \ell$ . Apply these constructions to $f ^ { \mathrm { r e f } } ( x ) =$ $f ( B - x )$ , and define

$$
H _ { R , + } ( v ) = H _ { L , - } ^ { \mathrm { r e f } } ( 1 - v ) , \qquad H _ { R , - } ( v ) = H _ { L , + } ^ { \mathrm { r e f } } ( 1 - v ) .
$$

They are nondecreasing and have integral bounds $3 \ell z ^ { 2 }$ and $4 \ell z ^ { 2 }$ . This proves $( 7 7 )$

We verify the displacement bound without assuming that G is continuous. Fix a graph selection y with $G ( y - ) \leqslant u \leqslant G ( y )$ . For $x < y , G ( x ) \leqslant u$ , whereas $G ( y ) \geqslant u$ . The CDF envelope therefore implies

$$
F ( x ) - v _ { + } \leqslant z { \sqrt { f ( x ) } } \quad ( x < y ) , \qquad F ( y ) + z { \sqrt { f ( y ) } } \geqslant v _ { - } .\tag{81}
$$

Taking a left limit in the first inequality also gives $F ( y ) - v _ { + } \leqslant z { \sqrt { f ( y ) } }$ . The zero endpoint values of f and $\varepsilon < u < 1 - \varepsilon$ exclude any nontrivial displacement into an initial or terminal interval of zero density.

Decompose $x \in [ 0 , B ]$ as the sum of three nondecreasing, 1-Lipschitz clipped parts,

$$
L ( x ) = x \wedge \ell , \quad J ( x ) = \left( ( x - \ell ) \vee 0 \right) \wedge ( B - 2 \ell ) , \quad R ( x ) = \left( x - B + \ell \right) \vee 0 , \qquad x = L ( x ) + J ( x ) + R ( x ) .
$$

If $q ( v _ { + } ) < y \land \ell ,$ let $b = F ( x ) - v _ { + } > 0$ for a point between these two locations. Then $x = q ( v _ { + } + b )$ and (81) gives

$$
g ( v _ { + } + b ) \leqslant z ^ { 2 } / b ^ { 2 } .
$$

Boundary monotonicity also gives $g ( v _ { + } + b ) \leqslant g ( v _ { + } )$ . Thus, writing $y _ { L } = y \land \ell .$

$$
\begin{array} { r l r } {  { L ( y ) - L ( q ( v _ { + } ) ) = \int _ { 0 } ^ { F ( y _ { L } ) - v _ { + } } g ( v _ { + } + b ) \mathrm { d } b } } \\ & { } & { \leqslant \int _ { 0 } ^ { \infty } \operatorname* { m i n } \{ g ( v _ { + } ) , z ^ { 2 } / b ^ { 2 } \} \mathrm { d } b = 2 z \sqrt { g ( v _ { + } ) } . } \end{array}
$$

This bounds the positive left-clipped displacement by $\sqrt { H _ { L , + } ( v _ { + } ) }$ ; if the displacement is zero or negative, no such bound is needed.

For the other direction, suppose $L ( y ) < L ( q ( v _ { - } ) )$ . Then $y < \ell$ and (81) implies $A _ { z } ( y ) \geqslant v _ { - }$ . In

particular $v _ { - } \leqslant A _ { z } ( \ell )$ and $y \geqslant a = a _ { z } ( v _ { - } )$ . Because $A _ { z } ( a ) = v _ { - }$ , monotonicity of f on $[ a , \ell ]$ gives

$$
f ( a ) \{ L ( q ( v _ { - } ) ) - a \} \leqslant F ( L ( q ( v _ { - } ) ) ) - F ( a ) \leqslant v _ { - } - F ( a ) = z { \sqrt { f ( a ) } } .
$$

The displacement is also at most ℓ. It is consequently bounded by $\sqrt { H _ { L , - } ( v _ { - } ) }$ . Replacing $q ( v _ { \pm } )$ by $q ( u )$ costs at most $r ( u )$ in either clipped comparison, so

$$
| L ( y ) - L ( q ( u ) ) | \leqslant r ( u ) + \sqrt { H _ { L , + } ( v _ { + } ) } + \sqrt { H _ { L , - } ( v _ { - } ) } .
$$

The graph-reflection identities of Lemma B.11 give the same assertion for R, with $H _ { R , + } ( v _ { + } )$ and $H _ { R , - } ( v _ { - } )$ . These identities justify reflection even at flat probability levels of $G .$

For the middle part, a displacement past $q ( v _ { + } )$ or before $q ( v _ { - } )$ has CDF mass at most $z \sqrt { C _ { 0 } }$ by (81) and its left limit. Every unit of middle-interval length carries density at least $c _ { \mathrm { i n t } }$ . Hence

$$
| J ( y ) - J ( q ( u ) ) | \leqslant r ( u ) + z { \sqrt { C _ { 0 } } } / { c _ { \mathrm { i n t } } } .
$$

Add the three clipped bounds and square, using the finite-sum Cauchy–Schwarz inequality. This proves (78) with a constant depending only on the stated geometry. ${ \mathrm { I f ~ } } z = 0 , \ ( 8 1 )$ brackets y between $q ( v _ { - } )$ and $q ( v _ { + } )$ by continuity and strict increase of F on its active support. All four envelopes may then be set to zero, proving that case as well. □

## B.3.3 From shifted probability levels to midpoint sums

Lemma B.13 (Shifted-grid and rank-shift bounds). Let $F , f , q$ have the geometry in Lemma $4 . 4$ $I f m \varepsilon < 1 / 4$ , then for every nonnegative monotone function H on $( 0 , 1 )$

$$
{ \frac { 1 } { m } } \sum _ { i = 1 } ^ { m } H ( \tau _ { i } \pm \varepsilon ) \leqslant 4 \int _ { 0 } ^ { 1 } H ( v ) \mathrm { d } v .
$$

With $r ( u ) = q ( u + \varepsilon ) - q ( u - \varepsilon )$

$$
\frac { 1 } { m } \sum _ { i } r ( \tau _ { i } ) \leqslant ( 1 6 \ell + 2 / c _ { \mathrm { i n t } } ) \varepsilon , \qquad \frac { 1 } { m } \sum _ { i } r ( \tau _ { i } ) ^ { 2 } \leqslant B ( 1 6 \ell + 2 / c _ { \mathrm { i n t } } ) \varepsilon .\tag{82}
$$

Proof. The shifted levels $w _ { i } = \tau _ { i } \pm \varepsilon$ are at distance at least $1 / ( 4 m )$ from both endpoints of their

original cells. If H is nonincreasing, then

$$
\frac { 1 } { m } H ( w _ { i } ) \leqslant 4 \int _ { ( i - 1 ) / m } ^ { w _ { i } } H ( v ) \mathrm { d } v .
$$

Summing over the disjoint cells gives the first assertion. For nondecreasing H, integrate from $w _ { i }$ to $i / m$ instead.

Put $g ( v ) = q ^ { \prime } ( v ) = 1 / f ( q ( v ) )$ . The rank shift has the integral representation

$$
r ( \tau _ { i } ) = \int _ { \tau _ { i } - \varepsilon } ^ { \tau _ { i } + \varepsilon } g ( v ) \mathrm { d } v .
$$

The portion with $q ( v ) \in [ \ell , B - \ell ]$ contributes at most $2 \varepsilon / c _ { \mathrm { i n t } }$ to the grid average. To treat the left boundary, let $a _ { i } = ( i - 1 ) / m$ and $b _ { i } = \tau _ { i } - \varepsilon$ . We have

$$
b _ { i } - a _ { i } = \frac { 1 } { 2 m } - \varepsilon \geqslant \frac { 1 } { 4 m } .
$$

If $q ( v ) < \ell$ for a point $v \in [ b _ { i } , \tau _ { i } + \varepsilon ]$ , then $q ( w ) \leqslant q ( v ) < \ell$ for every $w \in \left( a _ { i } , b _ { i } \right)$ . Left-boundary monotonicity gives $g ( v ) \leqslant g ( w )$ , and hence

$$
g ( v ) \leqslant 4 m \int _ { a _ { i } } ^ { b _ { i } } g ( w ) \mathrm { d } w .
$$

Integrating over the relevant v-interval of length at most 2ε and multiplying by $1 / m$ bounds its contribution by $8 \varepsilon \int _ { a _ { i } } ^ { b _ { i } } g .$ The intervals $( a _ { i } , b _ { i } )$ are disjoint, and each contributing interval lies entirely in the left-boundary quantile region. Thus the total left contribution is at most

$$
8 \varepsilon \int _ { \{ q ( w ) < \ell \} } g ( w ) \mathrm { d } w \leqslant 8 \ell \varepsilon .
$$

Reflection gives the same bound at the right boundary. Adding the middle part proves the first inequality in (82). The second follows from $0 \leqslant r \leqslant B$ , so that $r ^ { 2 } \leqslant B r$ . For $\varepsilon = 0$ , both rank-shift terms vanish. □

## B.4 Local Geometry of the Quantile Bellman Map

Assumption 4 is in force throughout this subsection. We establish the geometric ingredients of Lemmas 4.2 and 4.3. The neighbourhood is defined by (30); it is centred at the unprojected return law, not at an m-dependent support endpoint.

## B.4.1 A common positive density in the interior

Lemma B.14 (Common interior density lower bound). For each fixed $0 < \ell <$ min $\{ \kappa , 1 / 2 , B / 2 \}$ ， there exist $\rho > 0$ and $c _ { \mathcal { M } , \ell } > 0$ , both instance-dependent for the fixed ℓ, such that for every m and every $\pmb \theta \in \mathcal { U } _ { m } ( \rho )$

$$
\operatorname* { i n f } _ { s \in S } \operatorname* { i n f } _ { x \in [ \ell , B - \ell ] } f _ { s , \theta } ( x ) \geqslant c _ { \mathcal { M } , \ell } .
$$

Moreover, $\mathcal { U } _ { m } ( \rho )$ is convex under coordinatewise linear interpolation.

Proof. We first record the boundary shape. For $x \in [ 0 , \ell ]$ and $\theta ( s ^ { \prime } , j ) \geqslant 0$ , each shifted reward argument satisfies $x - \gamma \theta ( s ^ { \prime } , j ) \leqslant \ell$ . The zero-extended reward density is nondecreasing on $( - \infty , \ell ]$ Similarly, if $x \in [ B - \ell , B ]$ and $\theta ( s ^ { \prime } , j ) \leqslant B$ , then $x - \gamma \theta ( s ^ { \prime } , j ) \geqslant 1 - \ell ,$ where the zero extension is nonincreasing. Summing the nonnegative mixture weights in (29) gives

$$
\begin{array} { r } { f _ { s , \theta } ( y ) \leqslant f _ { s , \theta } ( x ) \quad ( 0 \leqslant y \leqslant x \leqslant \ell ) , \qquad f _ { s , \theta } ( y ) \leqslant f _ { s , \theta } ( x ) \quad ( B - \ell \leqslant x \leqslant y \leqslant B ) . } \end{array}\tag{83}
$$

These are boundary strips of the ambient interval; the actual support can be a proper subinterval.

By Proposition 2.2,

$$
c _ { \mathrm { r e t } } ( \mathcal { M } , \ell ) : = \operatorname* { m i n } _ { s } \operatorname* { i n f } _ { x \in [ \ell , B - \ell ] } p _ { \eta ^ { \pi } ( s ) } ( x ) > 0 .
$$

For each fixed successor state $s ^ { \prime } ,$ , choose an optimal $W _ { 1 }$ coupling $\Lambda _ { s ^ { \prime } }$ of $\eta _ { \theta } ( s ^ { \prime } )$ and $\eta ^ { \pi } ( s ^ { \prime } )$ , and let $( Y , Y ^ { \prime } )$ be its coordinate random variables. All expectations and probabilities in the next display are with respect to this coupling. For $d > 0$ 2

$$
\begin{array} { r } { \left| \mathbb { E } _ { \Lambda _ { s ^ { \prime } } { p } _ { s , a } } ( x - \gamma Y ) - \mathbb { E } _ { \Lambda _ { s ^ { \prime } } { p } _ { s , a } } ( x - \gamma Y ^ { \prime } ) \right| \leqslant \varpi _ { R } ( d ) + 2 C _ { 0 } \mathbb { P } _ { \Lambda _ { s ^ { \prime } } } ( \gamma | Y - Y ^ { \prime } | > d ) } \\ { \leqslant \varpi _ { R } ( d ) + \frac { 2 C _ { 0 } \gamma } { d } W _ { 1 } ( \eta _ { \theta } ( s ^ { \prime } ) , \eta ^ { \pi } ( s ^ { \prime } ) ) . } \end{array}
$$

The first term bounds the diference on $\{ \gamma | Y - Y ^ { \prime } | \leqslant d \}$ by uniform continuity; the second uses the density upper bound and then Markov’s inequality. The two expectations individually depend only on their indicated marginal laws.

Average this inequality over $( a , s ^ { \prime } )$ with weights $\pi ( a \mid s ) P ( s ^ { \prime } \mid s , a )$ . They sum to one, and $\eta ^ { \pi } = \tau ^ { \pi } \eta ^ { \pi }$ . Hence, for every $\pmb \theta \in \mathcal { U } _ { m } ( \rho )$

$$
| f _ { s , \theta } ( x ) - p _ { \eta ^ { \pi } ( s ) } ( x ) | \leqslant \infty _ { R } ( d ) + \frac { 2 C _ { 0 } \gamma \rho } { d } ,
$$

uniformly in $m , s , x .$ . Choose d so that $\varpi _ { R } ( d ) \leqslant c _ { \mathrm { r e t } } ( { \cal M } , \ell ) / 4$ , then choose $\rho$ so that $2 C _ { 0 } \gamma \rho / d \leqslant$ $c _ { \mathrm { r e t } } ( \mathcal { M } , \ell ) / 4$ . The claimed lower bound holds with $c _ { \mathcal { M } , \ell } = c _ { \mathrm { r e t } } ( \mathcal { M } , \ell ) / 2$

Finally, for ordered arrays, coordinatewise interpolation also interpolates their quantile functions on each probability cell. Thus, for $\pmb { \theta } _ { t } = ( 1 - t ) \pmb { \theta } + t \pmb { \vartheta }$

$$
\begin{array} { l } { { \displaystyle W _ { 1 } ( \eta _ { \theta _ { t } } ( s ) , \eta ^ { \pi } ( s ) ) = \int _ { 0 } ^ { 1 } \left| ( 1 - t ) F _ { \eta _ { \theta } ( s ) } ^ { - 1 } ( u ) + t F _ { \eta _ { \vartheta } ( s ) } ^ { - 1 } ( u ) - F _ { \eta ^ { \pi } ( s ) } ^ { - 1 } ( u ) \right| \mathrm { d } u } } \\ { { \displaystyle \leqslant ( 1 - t ) W _ { 1 } ( \eta _ { \theta } ( s ) , \eta ^ { \pi } ( s ) ) + t W _ { 1 } ( \eta _ { \vartheta } ( s ) , \eta ^ { \pi } ( s ) ) . } } \end{array}
$$

Take the state maximum. If both endpoints are in $\mathcal { U } _ { m } ( \rho )$ , the right side is at most $\rho ,$ proving convexity.

## B.4.2 A CDF–density comparison for transition perturbations

Lemma B.15 (CDF–density envelope). There is an instance-dependent $K _ { \mathcal { M } } < \infty$ such that, uniformly in $m \geqslant 1 , \pmb { \theta } \in \mathcal { U } _ { m } ( \rho )$ , s, and $x \in [ 0 , B ]$

$$
\operatorname* { m i n } \{ F _ { s , \theta } ( x ) , 1 - F _ { s , \theta } ( x ) \} \leqslant K _ { \mathcal { M } } f _ { s , \theta } ( x ) .\tag{84}
$$

In particular, since $f _ { s , \theta } \leqslant C _ { 0 }$ 2

$$
\operatorname* { m i n } \{ F _ { s , \theta } ( x ) , 1 - F _ { s , \theta } ( x ) \} \leqslant K _ { \mathcal { M } } \sqrt { C _ { 0 } } \sqrt { f _ { s , \theta } ( x ) } .\tag{85}
$$

Proof. On the left boundary, (83) gives

$$
F _ { s , \theta } ( x ) = \int _ { 0 } ^ { x } f _ { s , \theta } ( y ) \mathrm { d } y \leqslant x f _ { s , \theta } ( x ) \leqslant B f _ { s , \theta } ( x ) .
$$

The reflected argument gives $1 - F _ { s , \theta } ( x ) \leqslant ( B - x ) f _ { s , \theta } ( x )$ on the right boundary. These inequalities remain valid when an initial or terminal interval has zero density. On the interior, Lemma B.14 gives min $( F , 1 - F ) \leqslant 1 \leqslant c _ { \mathcal { M } , \ell } ^ { - 1 } f .$ . Choose $K _ { \mathcal { M } } = \operatorname* { m a x } \{ B , c _ { \mathcal { M } , \ell } ^ { - 1 } \}$ to prove (84); the second inequality follows from $f \leqslant \sqrt { C _ { 0 } } \sqrt { f }$ □

## B.4.3 The midpoint column-average estimate

The continuous version of the density ratio below has integral at most one. The issue is to transfer that fact to the finite midpoint grid uniformly over inputs, including extreme quantiles.

Lemma B.16 (Uniform midpoint logit stability under Assumption 4). Under the standing assumptions and Assumption $^ { 4 , }$ there exist instance-dependent constants $K _ { \mathcal { M } } < \infty$ and $r _ { \mathcal { M } } > 0$ such that, for every m ě 1, every state $s ,$ every $i \in [ m ]$ , and every $u \in ( 0 , 1 )$ satisfying

$$
| \mathrm { l o g i t } ( u ) - \mathrm { l o g i t } ( \tau _ { i } ) | \leqslant r _ { \mathcal { M } } ,
$$

we have

$$
| F _ { m , s } ^ { - 1 } ( u ) - F _ { m , s } ^ { - 1 } ( \tau _ { i } ) | \leqslant K _ { \mathcal { M } } | \mathrm { l o g i t } ( u ) - \mathrm { l o g i t } ( \tau _ { i } ) | .
$$

Proof. For all suficiently large $m ,$ Lemma B.1 puts $\pmb { \theta } _ { m }$ in the common neighbourhood $\mathcal { U } _ { m } ( \rho )$ , and Lemma B.14 gives a fixed lower bound ${ c . } _ { { \cal M } , \ell }$ for $f _ { m , s }$ on $[ \ell , B - \ell ]$ . The boundary monotonicity in (83) yields, on the active left boundary,

$$
F _ { m , s } ( x ) = \int _ { a _ { m , s } } ^ { x } f _ { m , s } ( y ) \mathrm { d } y \leqslant ( x - a _ { m , s } ) f _ { m , s } ( x ) \leqslant B f _ { m , s } ( x ) ,
$$

where $a _ { m , s }$ is the left endpoint of the active support. Thus, writing $q = F _ { m , s } ^ { - 1 } .$

$$
\frac { u ( 1 - u ) } { f _ { m , s } ( q ( u ) ) } \leqslant B
$$

for left-boundary probability levels. Reflection gives the same bound at the right boundary, and on the interior the ratio is at most $1 / ( 4 c _ { { \cal M } , \ell } )$ . Lemma B.11 shows that the active support is an interval and that $q$ is locally absolutely continuous there. Consequently

$$
\frac { \mathrm { d } } { \mathrm { d } \log \mathrm { i t } ( u ) } q ( u ) = \frac { u ( 1 - u ) } { f _ { m , s } ( q ( u ) ) }
$$

almost everywhere and is bounded by one instance-dependent constant. Integration in logit probability proves the displayed estimate for all suficiently large m.

Only finitely many smaller resolutions remain. By Assumption 2 and the argument of Lemma B.4, each selected midpoint quantile has positive continuous target density in a neighbourhood. Hence its inverse CDF is locally Lipschitz in probability level, and therefore locally Lipschitz in logit

probability. Taking the minimum of the finitely many logit neighbourhood radii and the maximum of their constants completes the proof. □

Lemma B.17 (Column-average estimate). For every $\zeta > 0$ , after choosing $\ell > 0$ suficiently small, there exists an instance-dependent threshold $M _ { 0 } = M _ { 0 } ( M , \ell , \zeta ) < \infty$ such that for all $m \geqslant M _ { 0 }$ , every $\pmb { \theta } \in \mathcal { U } _ { m } ( \rho )$ , every $s , a$ , and every shift $b \in [ 0 , B - 1 ]$ , if

$$
q _ { i } : = F _ { s , \theta } ^ { - 1 } ( \tau _ { i } ) ,
$$

then

$$
\frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { p _ { s , a } ( q _ { i } - b ) } { f _ { s , \theta } ( q _ { i } ) } \leqslant 1 + 4 C _ { 0 } \ell + \zeta .\tag{86}
$$

Proof. Write $q = F _ { s , \theta } ^ { - 1 }$ and split the sum into $q _ { i } < \ell , \ell \leqslant q _ { i } \leqslant B - \ell ,$ , and $q _ { i } > B - \ell .$ On the left, set $a _ { i } = ( i - 1 ) / m$ . For $a _ { i } < v \leqslant \tau _ { i }$ and $q _ { i } < \ell ,$ boundary monotonicity gives

$$
\frac { 1 } { f _ { s , \theta } ( q _ { i } ) } \leqslant \frac { 1 } { f _ { s , \theta } ( q ( \boldsymbol { v } ) ) } .
$$

Integrating on this half-cell, summing, and using Lemma B.11 gives

$$
\frac { 1 } { m } \sum _ { q _ { i } < \ell } \frac { 1 } { f _ { s , \theta } ( q _ { i } ) } \leqslant 2 \int _ { 0 } ^ { F _ { s , \theta } ( \ell ) } \frac { \mathrm { d } v } { f _ { s , \theta } ( q ( v ) ) } \leqslant 2 \ell .
$$

Multiplication by $p _ { s , a } \leqslant C _ { 0 }$ bounds the left-boundary contribution by $2 C _ { 0 } \ell .$ . Reflection gives the same bound on the right.

For the interior, abbreviate $c = c _ { \mathcal { M } , \ell }$ and define

$$
r ( x ) = \frac { p _ { s , a } ( x - b ) } { f _ { s , \theta } ( x ) } , \quad x \in [ \ell , B - \ell ] , \qquad h ( u ) = r ( q ( u ) ) \mathbb { 1 } \{ \ell \leqslant q ( u ) \leqslant B - \ell \} .
$$

The abbreviation c in this proof is the displayed density parameter, not a universal constant. The mixture density and each zero-extended reward density have modulus bounded by $\varpi R$ . Since $f _ { s , \theta } \geqslant c$ on the interior,

$$
| r ( x ) - r ( y ) | \leqslant \left( { \frac { 1 } { c } } + { \frac { C _ { 0 } } { c ^ { 2 } } } \right) \varpi _ { R } ( | x - y | ) .
$$

Also $| q ( u ) - q ( v ) | \leqslant | u - v | / c$ whenever the two quantiles lie in the interior interval. The continuous

ratio mass is

$$
\int _ { 0 } ^ { 1 } h ( u ) \mathrm { d } u = \int _ { \ell } ^ { B - \ell } p _ { s , a } ( x - b ) \mathrm { d } x \leqslant 1 .
$$

On a probability cell whose interior does not contain either cut level $F _ { s , \theta } ( \ell )$ or $F _ { s , \theta } ( B - \ell )$ , it follows that, for almost every u in that cell,

$$
| h ( \tau _ { i } ) - h ( u ) | \leqslant \left( \frac { 1 } { c } + \frac { C _ { 0 } } { c ^ { 2 } } \right) \varpi _ { R } \left( \frac { 1 } { m c } \right) .
$$

There are at most two exceptional cells. Their midpoint contributions are each at most $C _ { 0 } / ( m c )$ 2 since $h \leqslant C _ { 0 } / c$ . If a cut level is a cell endpoint, it does not afect the integral or require an additional exceptional cell. Consequently,

$$
\begin{array} { r l r } {  { \frac { 1 } { m } \sum _ { i : \ell \leqslant q _ { i } \leqslant B - \ell } \frac { p _ { s , a } ( q _ { i } - b ) } { f _ { s , \theta } ( q _ { i } ) } = \frac { 1 } { m } \sum _ { i } h ( \tau _ { i } ) } } \\ & { } & { \leqslant 1 + ( \frac { 1 } { c } + \frac { C _ { 0 } } { c ^ { 2 } } ) \varpi _ { R } ( \frac { 1 } { m c } ) + \frac { 2 C _ { 0 } } { m c } . } \end{array}
$$

The last two terms tend to zero uniformly in the stated inputs and shifts. Choose $M _ { 0 }$ so that they are at most $\zeta$ for $m \geqslant M _ { 0 }$ . Adding the two boundary estimates proves (86).

□

## B.5 Technical Lemmas for the Local Statistical Experiment

Throughout this subsection m is fixed and $Q _ { 0 }$ satisfies the standing assumptions. Constants display any dependence on the fixed resolution, centre, and local radius. All alternatives range over the full class (18); no independence-based stratification is used here.

## B.5.1 Empirical and Hellinger control for general joint laws

For $Q \in \mathcal { Q } .$ , set

$$
\begin{array} { l } { { \displaystyle J _ { Q , s , a , s ^ { \prime } } ( x ) = Q _ { s , a } ( ( - \infty , x ] \times \{ s ^ { \prime } \} ) , \qquad D _ { n } ( Q ) = \operatorname* { m a x } _ { s } \sum _ { a } \pi ( a \mid s ) \sum _ { s ^ { \prime } } \mid \hat { J } _ { s , a , s ^ { \prime } } - J _ { Q , s , a , s ^ { \prime } } \mid _ { \infty } . } } \end{array}
$$

The proof of Lemma B.2 uses only a scalar encoding of a paired observation, not conditional independence. Hence, uniformly over all $Q \in \mathcal { Q }$ , with $N _ { \mathrm { t a b } } = | S | ^ { 2 } | \mathcal { A } |$ 2

$$
\mathbb { P } _ { Q } \{ D _ { n } ( Q ) > t \} \leqslant 2 N _ { \mathrm { t a b } } \exp \left( - \frac { 2 n t ^ { 2 } } { | S | ^ { 2 } } \right) ,\tag{87}
$$

and, for every $r \geqslant 1$

$$
\operatorname* { s u p } _ { Q } \mathbb { E } _ { Q } [ ( \sqrt { n } D _ { n } ( Q ) ) ^ { r } ] \leqslant C _ { r } | S | ^ { r } \{ 1 + \log ( 2 N _ { \mathrm { t a b } } ) \} ^ { r / 2 } ,\tag{88}
$$

where $C _ { r }$ is a universal numerical constant depending only on r. Thus these empirical-process constants depend on the experiment only through |S| and $| { \cal A } |$ . For later use, if $H ( P , P _ { 0 } ) ^ { 2 } =$ $\int ( \sqrt { \mathrm { d } P } - \sqrt { \mathrm { d } P _ { 0 } } ) ^ { 2 }$ , then

$$
\mathrm { T V } ( P , P _ { 0 } ) \leqslant H ( P , P _ { 0 } ) , \qquad P ( A ) \leqslant 2 P _ { 0 } ( A ) + 2 H ( P , P _ { 0 } ) ^ { 2 } .\tag{89}
$$

The second inequality follows by the triangle inequality for the $L ^ { 2 } ( A )$ norms of the square-root densities. Both statements allow P to put mass on a $P _ { \mathrm { 0 } ^ { - } \mathrm { n u l l } }$ set.

## B.5.2 Fixed-resolution stability and moments

Lemma B.18 (Fixed-resolution stability and local moments). There is $L _ { m , Q _ { 0 } } < \infty$ , depending only on the fixed resolution and centre, such that, with $d = d _ { H } ( Q , Q _ { 0 } )$

$$
\| \pmb { \theta } _ { m } ( Q ) - \pmb { \theta } _ { m } ( Q _ { 0 } ) \| _ { \infty } \leqslant L _ { m , Q _ { 0 } } d ,\tag{90}
$$

$$
\| \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { m } ( Q _ { 0 } ) \| _ { \infty } \leqslant L _ { m , Q _ { 0 } } ( D _ { n } ( Q ) + d ) .\tag{91}
$$

In particular, for fixed $c < \infty$ and $r \geqslant 1$ 1,

$$
\operatorname* { s u p } _ { n \geqslant 1 } \operatorname* { s u p } _ { Q \in { \cal N } _ { n } ( c ; Q _ { 0 } ) } \mathbb { E } _ { Q } \left[ \left\{ \sqrt { n } \| \pmb { \theta } _ { m } ^ { ( n ) } - \pmb { \theta } _ { m } ( Q ) \| _ { \infty } \right\} ^ { r } \right] \leqslant C _ { m , c , r , Q _ { 0 } } < \infty .\tag{92}
$$

Proof. At the fixed centre, every selected target quantile has a neighbourhood with a positive continuous density, by the argument of Lemma B.4. There are finitely many coordinates. Choose a common spatial radius $b _ { m , Q _ { 0 } } > 0$ and density lower bound $a _ { m , Q _ { 0 } } ~ > ~ 0$ around their centre quantiles, with all intervals contained in I. If $e \leqslant a _ { m , Q _ { 0 } } b _ { m , Q _ { 0 } } / 2 .$ , graph inversion and the inequalities $F ( q + h ) \geqslant u + a _ { m , Q _ { 0 } } h$ and $F ( q - h ) \leqslant u - a _ { m , Q _ { 0 } } h$ give a displacement at most $e / a _ { m , Q _ { 0 } }$ (use a limiting argument for endpoint equality). For $e \mathrm { ~ > ~ } a _ { m , Q _ { 0 } } b _ { m , Q _ { 0 } } / 2$ , the support bound gives $B \leqslant 2 B e / ( a _ { m , Q _ { 0 } } b _ { m , Q _ { 0 } } )$ . Hence for the fixed input $\theta _ { m } ( Q _ { 0 } )$ , a target-CDF perturbation of supremum size e moves every selected lower inverse by at most $C _ { m , Q _ { 0 } } e$ . For small e this is inverse bracketing within these neighbourhoods; for larger e the bound follows after increasing $C _ { m , Q _ { 0 } }$ , since all quantiles

lie in I.

A Bellman target CDF is an expectation of a function between zero and one. Thus (89) gives

$$
\operatorname* { s u p } _ { s , x , \theta } | F _ { Q , s , \theta } ( x ) - F _ { Q _ { 0 } , s , \theta } ( x ) | \leqslant d .
$$

The empirical target difers from the target under $Q$ by at most $D _ { n } ( Q )$ , uniformly in all inputs. Apply the inverse bound at $\pmb { \theta } _ { m } ( Q _ { 0 } )$ and then the supremum-norm γ-contraction of the perturbed population or empirical parameter map. This proves (90)–(91), absorbing $( 1 - \gamma ) ^ { - 1 }$ into $L _ { m , Q _ { 0 } }$ . The triangle inequality bounds the centred empirical error by $L _ { m , Q _ { 0 } } ( D _ { n } ( Q ) + 2 d )$ . Equations (87)–(88) and ${ \sqrt { n } } d \leqslant c$ prove (92). □

The alternative densities need not be uniformly bounded and their quantile maps need not be diferentiable. Stability follows from regularity at the centre and Hellinger closeness to it.

## B.5.3 Small-interval empirical and population increments

Lemma B.19 (Small-interval empirical and population increments). Let $\theta _ { 0 } = \theta _ { m } ( Q _ { 0 } )$ and $H _ { n }$ be the empirical CDF equation vector. For $0 < \delta \leqslant 1$ , uniformly over $Q \in \mathcal { Q }$ with $d = d _ { H } ( Q , Q _ { 0 } )$ ,

$$
\mathbb { E } _ { Q } \operatorname* { s u p } _ { \| \theta - \theta _ { 0 } \| _ { \infty } \leqslant \delta } \sqrt { n } \| ( H _ { n } - H _ { Q } ) ( \theta ) - ( H _ { n } - H _ { Q } ) ( \theta _ { 0 } ) \| _ { \infty } \leqslant C _ { m , Q _ { 0 } } \sqrt { \delta + d ^ { 2 } } ,\tag{93}
$$

$$
\operatorname* { s u p } _ { \| \theta - \theta _ { 0 } \| _ { \infty } \leqslant \delta } \| ( H _ { Q } - H _ { Q _ { 0 } } ) ( \theta ) - ( H _ { Q } - H _ { Q _ { 0 } } ) ( \theta _ { 0 } ) \| _ { \infty } \leqslant C _ { m , Q _ { 0 } } d \sqrt { \delta + d ^ { 2 } } .\tag{94}
$$

Suprema can be restricted to ordered arrays in the parameter domain.

Proof. Every summand in a CDF equation has the form $\mathbb { 1 } \{ S ^ { \prime } = s ^ { \prime } , R \leqslant x \}$ , with threshold $x = \theta ( u , i ) - \gamma \theta ( s ^ { \prime } , j )$ . Changing the parameter by at most $\delta$ changes this threshold by at most $( 1 + \gamma ) \delta$ . There are finitely many centre thresholds at fixed $m$ . For a corresponding band $A _ { \delta }$ , the centre density bound gives $Q _ { 0 , b } ( A _ { \delta } ) \leqslant C _ { m , Q _ { 0 } } \delta$ . By (89),

$$
Q _ { b } ( A _ { \delta } ) \leqslant C _ { m , Q _ { 0 } } \delta + 2 d ^ { 2 } .
$$

Here is an elementary maximal bound for the nested indicators within one such band A. Write $q = Q _ { b } ( A )$ and let $N _ { A }$ be its sample count. Conditional on $N _ { A }$ , the observations in A are iid from the conditional law. When $N _ { A } > 0$ , the expected supremum of their conditional empirical-CDF error is at most $C / \sqrt { N _ { A } }$ by DKW, where C is universal. When $N _ { A } = 0$ , the corresponding empirical conditional term is defined to be zero. To see the decomposition, for the nested sets $B _ { x } \subseteq A$ write

$$
( Q _ { n , b } - Q _ { b } ) ( B _ { x } ) = \frac { N _ { A } } { n } \{ F _ { N _ { A } , A } ( x ) - F _ { A } ( x ) \} + \left( \frac { N _ { A } } { n } - q \right) F _ { A } ( x ) .
$$

An increment based at an interior threshold is the diference of two such nested terms, which changes only an absolute constant. The expected error of the count fraction is at most $\sqrt { q / n }$ . Multiplying the former bound by $N _ { A } / n$ , taking expectations and using $\mathbb { E } \sqrt { N _ { A } } \leqslant \sqrt { n q }$ shows

$$
\mathbb { E } _ { Q } \operatorname* { s u p } _ { x \mathrm { ~ i n ~ t h e ~ b a n d } } | ( Q _ { n , b } - Q _ { b } ) ( \mathbb { 1 } _ { \{ S ^ { \prime } = s ^ { \prime } , R \leqslant x \} } - \mathbb { 1 } _ { \{ S ^ { \prime } = s ^ { \prime } , R \leqslant x _ { 0 } \} } ) | \leqslant C \sqrt { q / n } .
$$

The case $q = 0$ is immediate. Scalar suprema can be taken over rational thresholds with one-sided limits; they are measurable. The parameter supremum is dominated by the finite sum of these scalar envelopes, so all ensuing probability bounds use measurable majorants. If necessary the expectation of the original parameter supremum is read as an outer expectation. Summing over the finitely many thresholds and blocks proves (93).

For any function u with $| u | \leqslant 1$ supported on A, Cauchy–Schwarz applied to the square-root densities gives

$$
\left| \int u \mathrm { d } ( Q _ { b } - Q _ { 0 , b } ) \right| \leqslant H ( Q _ { b } , Q _ { 0 , b } ) \sqrt { 2 \{ Q _ { b } ( A ) + Q _ { 0 , b } ( A ) \} } .
$$

Apply this to each indicator diference, and sum, to obtain (94).

## B.6 Construction and Localization of the Hard Instance

This subsection supplies the technical construction used in Appendix A.5. The MRP has two states $\{ s , t \}$ , one action, $P ( t \mid s ) = P ( t \mid t ) = 1$ , and $\gamma = 1 / 2$ . Put

$$
x _ { j } = 8 ^ { - j } , \qquad \epsilon _ { j } = \exp ( - e ^ { j ^ { 2 } } ) .
$$

In this subsection, bare $c , C , C ^ { \prime } , \ldots$ . denote positive constants tied to this one fixed construction, not universal constants over the model class. They may change from line to line but are independent of $j , m , n , p$ . Any additional dependence, such as on a fixed small-ball multiplier, is stated explicitly.

## B.6.1 A reward density with infinitely many low-density intervals

Lemma B.20 (Strictly monotone shelf construction). There is a $C ^ { 1 }$ density $p _ { s }$ satisfying Assumptions 1 and $^ { 4 , }$ , with strictly positive derivative on a left endpoint neighborhood and strictly negative derivative on a right endpoint neighborhood, such that, for fixed construction constants $c , C > 0$ and all suficiently large j,

$$
\begin{array} { r } { c \epsilon _ { j } \leqslant p _ { s } ( r ) \leqslant C \epsilon _ { j } , \qquad r \in [ x _ { j } / 4 , x _ { j } ] , } \end{array}\tag{95}
$$

and

$$
p _ { s } ( r ) \leqslant C \epsilon _ { j } , \qquad 0 \leqslant r \leqslant x _ { j } .\tag{96}
$$

Proof. On $[ x _ { j } / 4 , x _ { j } ]$ start with the linear segment

$$
p ^ { 0 } ( r ) = \epsilon _ { j } + \frac { 4 \epsilon _ { j } } { 3 x _ { j } } ( r - x _ { j } / 4 ) ,
$$

whose values range from $\epsilon _ { j }$ to $2 \epsilon _ { j }$ and whose derivative is $d _ { j } = 4 \epsilon _ { j } / ( 3 x _ { j } ) > 0$ . The gap $[ x _ { j } / 8 , x _ { j } / 4 ]$ joins height $2 \epsilon _ { j + 1 }$ and derivative $d _ { j + 1 }$ to height $\epsilon _ { j }$ and derivative $d _ { j }$ . Let its length be $L _ { j } = x _ { j } / 8$ and required height increase be $A _ { j } = \epsilon _ { j } - 2 \epsilon _ { j + 1 } > 0$

Here is an explicit way to choose a positive derivative on the gap. Take a fixed small $\zeta \in ( 0 , 1 / 1 0 )$ On the initial and final subintervals of length $\zeta L _ { j }$ , smoothly interpolate from the respective endpoint derivatives to the positive level $\epsilon _ { j + 1 } / x _ { j }$ . Keep that level on the remaining middle interval. The integral $I _ { j }$ of this preliminary derivative is less than $A _ { j } / 2$ for all large $j ,$ by choosing $\zeta$ suficiently small; all its values are at most $C \epsilon _ { j } / x _ { j }$ . Add a nonnegative smooth bump supported in the middle, with integral $A _ { j } - I _ { j }$ . A rescaled fixed bump has maximum at most $C A _ { j } / L _ { j } \leqslant C \epsilon _ { j } / x _ { j }$ . The resulting derivative is strictly positive, matches both endpoint derivatives, and has exactly the required integral. Its primitive joins the two shelf segments in $C ^ { 1 }$ . This construction is needed only for suficiently large $j ;$ finitely many remaining pieces can be joined separately.

Thus the combined function is $C ^ { 1 }$ , strictly increasing, and has derivative at most $C \epsilon _ { j } / x _ { j }$ on $[ x _ { j } / 8 , x _ { j } ]$ . Both its value and derivative tend to zero at zero. Extend it over the rest of $( 0 , 1 )$ as a positive $C ^ { 1 }$ function, with negative derivative near one and value zero at one, and normalize its integral. Normalization changes all comparisons only by one fixed constant. On each shelf the density is comparable to $\epsilon _ { j } ,$ and monotonicity gives the upper bound on $[ 0 , x _ { j } ]$ . The derivative is globally bounded, so the density is Lipschitz. □

## B.6.2 Enforcing all non-boundary conditions

Lemma B.21 (A regular perturbation preserving the lower bound). The shelf density from Lemma B.20 and the Betap2, 2q density can be perturbed so that Assumption 2 holds for every m and every pair of coordinates, while preserving the shelf comparisons and constant-factor comparability of $p _ { t }$ to the Betap2, 2q density. The perturbed densities still satisfy strict endpoint monotonicity and Lipschitz continuity.

Proof. First note that for this deterministic-transition model every population target is an equally weighted mixture of m shifted continuous reward laws with positive density in each open support interval. If a target quantile $q$ had zero density, every summand $p _ { u } ( q - \gamma \theta _ { m } ( t , j ) )$ would vanish. Positivity on p0, 1q forces each shifted argument outside that open interval, so every corresponding reward CDF value is either zero or one. The target CDF at $q$ would therefore be $k / m$ for an integer k. This cannot equal $( i - 1 / 2 ) / m$ . Thus all midpoint target densities are positive, independently of any endpoint coincidence between individual shifted arguments. Since each reward density is continuous and zero at its endpoints, its zero-extended CDF is $C ^ { 1 }$ . The fixed-point equations are therefore continuously diferentiable at every fixed m.

Here is a precise perturbation space. Start from the $C ^ { 1 }$ shelf density and the Betap2, 2q density, denoting them by $p _ { s } ^ { 0 } , p _ { t } ^ { 0 }$ . Choose disjoint endpoint neighborhoods $E = ( 0 , \kappa ) \cup ( 1 - \kappa , 1 )$ on which the derivatives have the prescribed strict signs. For pairs of $C ^ { 1 }$ functions $h = \left( h _ { s } , h _ { t } \right)$ with $h _ { u } ( 0 ) = h _ { u } ( 1 ) = 0$ and $\int h _ { u } = 0$ , use the norm

$$
\| h \| _ { * } : = \operatorname* { m a x } _ { u \in \{ s , t \} } \left( \| h _ { u } \| _ { C ^ { 1 } } + \operatorname* { s u p } _ { 0 < r < 1 } \frac { | h _ { u } ( r ) | } { p _ { u } ^ { 0 } ( r ) } + \operatorname* { s u p } _ { r \in E } \frac { | h _ { u } ^ { \prime } ( r ) | } { | ( p _ { u } ^ { 0 } ) ^ { \prime } ( r ) | } \right) .
$$

The finite-norm subspace is complete: a Cauchy sequence converges in $C ^ { 1 }$ , and its weighted quotients converge uniformly to the corresponding quotients of the limit. In an open ball $\| h \| _ { * } < 1 / 4$ , the densities $p _ { u } ^ { 0 } + h _ { u }$ retain positivity, endpoint derivative signs, and all constant-factor comparisons. This ball is a Baire space.

For fixed $m$ the parameter map is diferentiable in any smooth compactly supported density direction. Its fixed-point derivative is well defined because

$$
D \Phi _ { m } = \gamma K _ { m } , \qquad K _ { m } \geqslant 0 , \quad K _ { m } { \bf 1 } = { \bf 1 } ,
$$

so $I - \gamma K _ { m }$ is invertible. At any pair of densities in the ball, consider the finite set of active reward arguments $\theta _ { m } ( u , i ) - \gamma \theta _ { m } ( t , j ) , u = s , t ,$ that lie in p0, 1q. Choose a smooth compactly supported CDF perturbation $H _ { u }$ taking the value $- p _ { u }$ at all these arguments. Such interpolation is possible by disjoint smooth bumps around the distinct arguments. At arguments outside p0, 1q or at the endpoints both $H _ { u }$ and the zero-extended $p _ { u }$ are zero. Set the density perturbation to $H _ { u } ^ { \prime }$ . It integrates to zero, is smooth and compactly supported in p0, 1q, and has finite $\| \cdot \| _ { * }$ norm. A suficiently small perturbation stays in the ball.

Implicit diferentiation of all fixed-point quantile equations in this direction gives

$$
\begin{array} { r } { \dot { \theta } _ { m } = ( 1 - \gamma ) ^ { - 1 } \mathbf { 1 } . } \end{array}
$$

Indeed the direct CDF perturbation at each selected quantile is minus its density, namely $- [ D _ { m } \mathbf { 1 } ] _ { u , i }$ The derivative equation is consequently

$$
\begin{array} { r } { D _ { m } ( { I } - \gamma { K } _ { m } ) \dot { \pmb { \theta } } _ { m } = D _ { m } { \mathbf 1 } . } \end{array}
$$

Because $K _ { m } \mathbf { 1 } = \mathbf { 1 }$ , the displayed common-shift solution follows. This also verifies the sign and the discount factor without relying on an ambiguous row index. Consequently, for any $u , u ^ { \prime } \in \{ s , t \}$ and coordinates $i , j$

$$
\frac { \mathrm { d } } { \mathrm { d } \epsilon } \big ( \theta _ { m } ( u , i ) - \gamma \theta _ { m } ( u ^ { \prime } , j ) \big ) = 1 .
$$

Thus each forbidden equality to zero or one defines a closed set with empty interior in the perturbation ball. Closedness follows from local continuity of fixed points, using their positive quantile densities and the supremum-norm contraction. There are countably many equalities over all $m , u , u ^ { \prime } , i , j$ . The Baire category theorem supplies a point outside their union. At that point Assumption 2 holds, and all prescribed density comparisons hold by construction. □

## B.6.3 Downstream small balls and the extreme quantile subsequence

Lemma B.22 (Localization of an extreme midpoint quantile). Choose the final perturbed densities from Lemma B.21. Let ν be the true return law at state $t ,$ and $\nu _ { m }$ its population quantile fixed point. Define

$$
H ( x ) = \mathbb { P } ( R _ { s } + \gamma Z \leqslant x ) , \quad H _ { m } ( x ) = \mathbb { P } ( R _ { s } + \gamma Z _ { m } \leqslant x ) , \qquad Z \sim \nu , \quad Z _ { m } \sim \nu _ { m } ,
$$

with $R _ { s }$ independent of the continuation value. Set $a _ { j } = H ( { 3 x _ { j } } / { 4 } )$ and choose an integer $m _ { j }$ nearest to $( 2 a _ { j } ) ^ { - 1 }$ . For all suficiently large $j$

$$
c \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } } \leqslant a _ { j } \leqslant C \epsilon _ { j } x _ { j } , \qquad \log m _ { j } = e ^ { j ^ { 2 } } + O ( j ^ { 2 } ) , \qquad q _ { j } : = H _ { m _ { j } } ^ { - 1 } ( 1 / ( 2 m _ { j } ) ) \in [ x _ { j } / 2 , x _ { j } ] .
$$

Also $W _ { 1 } ( \nu _ { m } , \nu ) \leqslant C / m$ . For every fixed $c _ { 1 } > 0$ , a constant depending only on the constructed instance and $c _ { 1 }$ gives $\nu ( [ 0 , c _ { 1 } x _ { j } ] ) \geqslant e ^ { - C j ^ { 2 } }$ for all suficiently large $j$

Proof. For every fixed $c _ { 1 } > 0$ there is a finite $C ,$ depending also on $c _ { 1 }$ , such that

$$
\nu ( [ 0 , c _ { 1 } x _ { j } ] ) \geqslant e ^ { - C j ^ { 2 } } .\tag{97}
$$

For all suficiently large $j ,$ , choose $K _ { j } \ = \ \lceil \log _ { 2 } ( 4 / ( c _ { 1 } x _ { j } ) ) \rceil \ = \ O ( j )$ , so that $\textstyle \sum _ { k \geq K _ { j } } 2 ^ { - k } \leqslant c _ { 1 } x _ { j } / 2$ Require the first $K _ { j }$ rewards to be at most $c _ { 2 } x _ { j }$ , with $c _ { 2 } \leqslant c _ { 1 } / 4$ . Their discounted sum is then at most $2 c _ { 2 } x _ { j } \leqslant c _ { 1 } x _ { j } / 2$ . Since $p _ { t } ( r ) = r$ near zero, each such event has probability at least $c _ { 3 } x _ { j } ^ { 2 }$ Independence gives

$$
( c _ { 3 } x _ { j } ^ { 2 } ) ^ { K _ { j } } \geqslant e ^ { - C j ^ { 2 } } .
$$

Let $\nu _ { m }$ be the population quantile fixed point at state t. The deterministic midpoint approximation and the $W _ { 1 }$ contraction of $\mathcal { T } ^ { \pi }$ give

$$
W _ { 1 } ( \nu _ { m } , \nu ) \leqslant C / m .\tag{98}
$$

Selecting the resolution. Define

$$
H ( x ) : = \mathbb { P } ( R _ { s } + \gamma Z \leqslant x ) , \qquad H _ { m } ( x ) : = \mathbb { P } ( R _ { s } + \gamma Z _ { m } \leqslant x ) , \quad Z _ { m } \sim \nu _ { m } .
$$

Let $y _ { j } = 3 x _ { j } / 4$ and $a _ { j } : = H ( y _ { j } )$ . By (95) and (97),

$$
c \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } } \leqslant a _ { j } \leqslant C \epsilon _ { j } x _ { j } .\tag{99}
$$

Choose $m _ { j }$ to be an integer nearest to $( 2 a _ { j } ) ^ { - 1 }$ . Then

$$
\log m _ { j } = e ^ { j ^ { 2 } } + O ( j ^ { 2 } ) .\tag{100}
$$

For $x \leqslant x _ { j }$ , the map $z \mapsto F _ { R _ { s } } ( x - \gamma z )$ has Lipschitz constant at most $C \epsilon _ { j }$ by (96). Hence (98) gives

$$
\operatorname* { s u p } _ { x \leqslant x _ { j } } | H _ { m } ( x ) - H ( x ) | \leqslant C \epsilon _ { j } / m .\tag{101}
$$

Moreover, throughout $x \in [ x _ { j } / 2 , x _ { j } ]$ , the target density obeys

$$
H ^ { \prime } ( x ) \geqslant c \epsilon _ { j } e ^ { - C j ^ { 2 } } ,\tag{102}
$$

because on $Z \leqslant x _ { j } / 2$ one has $x - \gamma Z \in [ x _ { j } / 4 , x _ { j } ]$ for every $x \in [ x _ { j } / 2 , x _ { j } ]$ . For all large $j ,$ , nearestinteger rounding gives $m _ { j } = a _ { j } ^ { - 1 }$ and

$$
\left| \frac { 1 } { 2 m _ { j } } - a _ { j } \right| \leqslant C a _ { j } ^ { 2 } .
$$

The CDF increase from $x _ { j } / 2$ to $y _ { j } ~ = ~ 3 x _ { j } / 4$ , and from $y _ { j }$ to $x _ { j }$ , is at least $c \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } }$ . Both perturbations are negligible relative to that margin: using $a _ { j } \leqslant C \epsilon _ { j } x _ { j }$

$$
\frac { a _ { j } ^ { 2 } + \epsilon _ { j } / m _ { j } } { \epsilon _ { j } x _ { j } e ^ { - C j ^ { 2 } } } \leqslant C \epsilon _ { j } ( 1 + x _ { j } ) e ^ { C j ^ { 2 } } \longrightarrow 0 .
$$

Hence $H _ { m _ { j } } ( x _ { j } / 2 ) < 1 / ( 2 m _ { j } ) < H _ { m _ { j } } ( x _ { j } )$ eventually, which gives

$$
q _ { j } : = H _ { m _ { j } } ^ { - 1 } \left( \frac { 1 } { 2 m _ { j } } \right) \in [ x _ { j } / 2 , x _ { j } ] .\tag{103}
$$

This is the first population midpoint quantile at state $s .$