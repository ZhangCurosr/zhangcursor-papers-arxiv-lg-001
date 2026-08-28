# A Finite Sample Analysis for Quantile Temporal Diference Learning in Distributional Reinforcement Learning

Zijie Cheng<sup>∗</sup> Xiang Li<sup>†</sup> Yang Peng<sup>‡</sup> Zhihua Zhang<sup>§</sup>

August 28, 2026

## Abstract

We establish a global finite-sample guarantee for synchronous quantile temporal-diference learning (QTD) in tabular distributional reinforcement learning. The proof separates two stability mechanisms. A global comparison argument, based on the order monotonicity of reward cumulative distribution functions and the $W _ { \infty }$ contraction of the distributional Bellman operator, brings an arbitrarily initialized iterate into a local neighborhood. Inside that neighborhood, we linearize the QTD mean field. Its Jacobian is a nonsingular M-matrix, and the associated positive semigroup permits a variance-sensitive martingale analysis. For stepsizes $\alpha _ { t } = c ( t + 1 ) ^ { - a }$ with $a \in ( 1 / 2 , 1 )$ , the leading last-iterate fluctuation is of order $\widetilde { O } ( T ^ { - a / 2 } / \sqrt { 1 - \gamma } )$ and has no polynomial dependence on the number of quantiles. The deterministic transient and the required burn-in can still depend on the smallest Bellman-target density, which is of order $m ^ { - 1 }$ in the worst case. The result therefore distinguishes sharply between the local stochastic fluctuation and the global sample complexity.

## 1 Introduction

Distributional reinforcement learning (DRL) models the law of the random return rather than only its expectation [7, 1]. Since a return distribution is generally infinite-dimensional, practical methods use finite-dimensional representations. Quantile representations are especially prominent: quantile temporal-diference learning (QTD) underlies algorithms such as QR-DQN and implicit quantile networks [6, 5].

The finite-sample analysis of QTD is subtle even in a tabular policy-evaluation problem with a generative model. The update contains indicator functions and is therefore non-smooth at the sample level. Its coordinates are coupled through a projected distributional Bellman target, and the density at an extreme quantile can decrease with the number of quantiles. A proof that treats the stochastic update as an arbitrary bounded perturbation consequently loses both the cancellation of martingale noise and the local variance of the quantile score.

This paper studies synchronous QTD with polynomially decreasing stepsizes $\alpha _ { t } = c ( t + 1 ) ^ { - a }$ $a ~ \in ~ ( 1 / 2 , 1 )$ . We prove a high-probability bound for the raw last iterate from an arbitrary initialization in the natural parameter domain. The main result separates a deterministic transient from a local stochastic term. Up to logarithmic factors, the latter is

$$
\sqrt { \frac { \alpha _ { T } } { c _ { \mathcal M } ( 1 - \gamma ) } } ,
$$

with no polynomial dependence on the number m of quantiles. By contrast, the transient depends on the smallest Bellman-target density and can require an m-dependent burn-in. This distinction prevents an m-free local fluctuation statement from being mistaken for an m-uniform global sample complexity result.

## 1.1 Proof Strategy and Contributions

Our argument has two stages.

First, we use a global comparison mechanism. A smooth approximation to the supremum norm tracks the largest signed coordinate error. Bellman discounting creates an inward displacement for such a coordinate; monotonicity of reward cumulative distribution functions and a quantitative identifiability bound turn that displacement into a uniform drift. A stopped martingale argument on a late time block then shows that the iterate enters a prescribed local neighborhood with high probability. This stage uses no linearization and allows an arbitrary initialization.

Second, we exploit the exact local geometry. If h is the QTD mean field, then its Jacobian at the projected Bellman fixed point has the form

$$
\begin{array} { r } { \pmb { G } _ { m } = \pmb { D } _ { m } - \gamma \pmb { B } _ { m } = \pmb { D } _ { m } \big ( \pmb { I } - \gamma \pmb { K } _ { m } \big ) , } \end{array}
$$

where $D _ { m }$ is the diagonal matrix of Bellman-target densities and $K _ { m }$ is row-stochastic. Hence $G _ { m }$

is a nonsingular M-matrix and its discrete semigroup is positive and substochastic. Positivity yields a discrete hazard identity. Combining that identity with

$$
\begin{array} { r } { \mathrm { V a r } ( \xi _ { s , i } \mid \mathcal { F } _ { t } ) \lesssim \tau _ { i } ( 1 - \tau _ { i } ) \lesssim v _ { m } d _ { s , i } } \end{array}
$$

telescopes the predictable variance of the propagated martingale. This variance–drift matching removes the artificial inverse-density factor from the leading stochastic term. A stopped nonlinear bootstrap controls the quadratic Taylor remainder and completes the local capture argument.

The proof is therefore hybrid. Monotonicity and $W _ { \infty }$ contraction provide global stability, while Jacobian linearization provides the sharp local fluctuation. Treating the Jacobian merely as a generic Hurwitz matrix would discard the positive-system structure and can reintroduce dimension or conditioning losses.

## 1.2 Related Work

Foundational work on DRL established distributional Bellman operators and categorical or quantile approximations [1, 10, 6]. For quantile representations, Rowland et al. [11] proved asymptotic convergence of QTD, while Rowland et al. [12] studied the statistical role of quantile approximations for expected-return estimation. Finite-sample analyses are more developed for categorical distributional methods [8, 9]. Our focus is the non-smooth, nonlinear QTD recursion and, in particular, a finite-time last-iterate bound that retains the score variance.

The local part of our proof is related to linear stochastic approximation and martingale concentration, but it uses more than generic stability. The M-matrix factorization links density, Bellman contraction, and noise variance coordinate by coordinate. This structure also connects the finite-sample calculation to local limit and inference theory for QTD [3], while the present paper addresses a distinct global localization problem.

The remainder of the paper is organized as follows. Section 2 introduces the model and assumptions. Section 3 states the finite-sample result. Section 4 develops the two-stage proof, and the appendices provide the omitted arguments and technical lemmas.

## 2 Preliminaries

In this section, we introduce the necessary background. We review the Markov decision process in Section 2.1, followed by metrics on spaces of probability measures in Section 2.2. Section 2.3

introduces the distributional Bellman and quantile projection operators, and Section 2.4 presents synchronous QTD. We state our main assumptions and preliminary lemmas in Sections 2.5 and 2.6.

## 2.1 Problem Setup

We consider a discounted Markov decision process (MDP) specified by the tuple $\mathcal { M } = \langle \boldsymbol { S } , \mathcal { A } , \mathcal { P } _ { R } , \boldsymbol { P } , \gamma \rangle$ 2 where $s$ and $\mathcal { A }$ are finite state space and finite action space respectively, $\mathcal { P } _ { R } \colon \mathcal { S } \times \mathcal { A } \to \Delta ( [ 0 , 1 ] )$ is the distribution of rewards, $P \colon S \times A \to \Delta ( S )$ is the transition probabili $\mathrm { t y , }$ and $\gamma \in ( 0 , 1 )$ is the discount factor. Here $\Delta ( \cdot )$ denotes the set of probability distributions over some set.

For a fixed policy $\pi \colon S \to \Delta ( { \mathcal { A } } )$ and an initial state $S _ { 0 } ~ = ~ s ~ \in ~ \mathcal { S } .$ , a random trajectory $\{ ( S _ { t } , A _ { t } , R _ { t } ) \} _ { t = 0 } ^ { \infty }$ can be sampled from the MDP using the following procedure:

$$
\begin{array} { r l } & { \quad A _ { t } \mid S _ { t } \sim \pi ( \cdot \mid S _ { t } ) , } \\ & { \quad \quad R _ { t } \mid ( S _ { t } , A _ { t } ) \sim { \mathcal { P } } _ { R } ( \cdot \mid S _ { t } , A _ { t } ) , } \\ & { \quad \quad S _ { t + 1 } \mid ( S _ { t } , A _ { t } ) \sim P ( \cdot \mid S _ { t } , A _ { t } ) . } \end{array}
$$

The return of such a trajectory starting from state s is defined as the random variable

$$
G ^ { \pi } ( s ) : = \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { t } ,
$$

which is bounded almost surely by $[ 0 , ( 1 - \gamma ) ^ { - 1 } ]$ . The value function $V ^ { \pi } ( s )$ is defined by the expected return $\mathbb { E } [ G ^ { \pi } ( s ) ]$ . We further denote by $\eta ^ { \pi } ( s ) \in \Delta ( [ 0 , ( 1 - \gamma ) ^ { - 1 } ] )$ the distribution of $G ^ { \pi } ( s )$ , and denote $\pmb { \eta } ^ { \pi } = ( \eta ^ { \pi } ( s ) ) _ { s \in { \mathcal { S } } }$ for the collection of return distributions across all states.

## 2.2 Metrics on the Space of Measures

Denote the space of all probability distributions on $\mathbb { R }$ as $\mathcal { P }$ . For $\mu \in \mathcal { P }$ , the cumulative distribution function is defined as $F _ { \mu } ( x ) = \mu ( - \infty , x ]$ , and the quantile function is defined as $F _ { \mu } ^ { - 1 } ( \tau ) = \operatorname* { i n f } \{ x \colon F _ { \mu } ( x ) \geq \tau \}$ . For $1 \leq p < \infty$ and $\mu , \nu \in { \mathcal { P } }$ , the p-Wasserstein metric between $\mu$ and ν is defined as

$$
W _ { p } ( \mu , \nu ) = \left( \int _ { 0 } ^ { 1 } \left| F _ { \mu } ^ { - 1 } ( t ) - F _ { \nu } ^ { - 1 } ( t ) \right| ^ { p } \mathrm { d } t \right) ^ { 1 / p } ,
$$

and the ∞-Wasserstein metric is defined as

$$
W _ { \infty } ( \mu , \nu ) = \operatorname * { s u p } _ { t \in ( 0 , 1 ) } \left| F _ { \mu } ^ { - 1 } ( t ) - F _ { \nu } ^ { - 1 } ( t ) \right| .
$$

Moreover, for any extended metric d: $\mathcal { P } \times \mathcal { P }  [ 0 , \infty ]$ , we can define its supremum extension $\bar { d } \colon \mathcal { P } ^ { S } \times \mathcal { P } ^ { S }  [ 0 , \infty ]$ as

$$
\bar { d } ( \eta , \eta ^ { \prime } ) = \operatorname* { s u p } _ { s \in { \mathcal { S } } } d ( \eta ( s ) , \eta ^ { \prime } ( s ) ) ,
$$

which is an extended metric on $\mathcal { P } ^ { S }$

## 2.3 Distributional Bellman Operator and Quantile Projection Operator

A fundamental property of the value function is that it satisfies the Bellman equation. Letting $V ^ { \pi } : = ( V ^ { \pi } ( s ) ) _ { s \in { \cal S } }$ , we have for any $s \in { \mathcal { S } }$

$$
\begin{array} { r l } & { V ^ { \pi } ( s ) = \left[ T ^ { \pi } V ^ { \pi } \right] ( s ) } \\ & { \qquad : = \mathbb { E } _ { A \sim \pi ( \cdot \vert s ) , R \sim \mathcal { P } _ { R } ( \cdot \vert s , A ) } [ R ] + \gamma \mathbb { E } _ { A \sim \pi ( \cdot \vert s ) , S ^ { \prime } \sim P ( \cdot \vert s , A ) } [ V ^ { \pi } ( S ^ { \prime } ) ] } \\ & { \qquad = \displaystyle \sum _ { a \in \mathcal { A } } \pi ( a \mid s ) \int _ { 0 } ^ { 1 } r \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) + \gamma \sum _ { a \in \mathcal { A } , s ^ { \prime } \in S } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) V ^ { \pi } ( s ^ { \prime } ) . } \end{array}\tag{1}
$$

The operator $T ^ { \pi } \colon { \mathbb { R } ^ { S } }  { \mathbb { R } ^ { S } }$ is referred to as the Bellman operator, and Equation (1) characterizes $V ^ { \pi }$ as its unique fixed point.

An analogous relationship holds for the return distributions $\eta ^ { \pi }$ , known as the distributional Bellman equation. That is, for each $s \in { \mathcal { S } }$

$$
\begin{array} { r l } & { \eta ^ { \pi } ( s ) = \left[ \mathcal { T } ^ { \pi } \eta ^ { \pi } \right] ( s ) } \\ & { \qquad : = { \mathbb E } _ { A \sim \pi ( \cdot \vert s ) , R \sim \mathcal { P } _ { R } ( \cdot \vert s , A ) , S ^ { \prime } \sim P ( \cdot \vert s , A ) } \left[ ( b _ { R , \gamma } ) _ { \# } \eta ^ { \pi } ( S ^ { \prime } ) \right] } \\ & { \qquad = \displaystyle \sum _ { a \in \mathcal { A } , s ^ { \prime } \in S } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) \int _ { 0 } ^ { 1 } ( b _ { r , \gamma } ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) . } \end{array}
$$

Here $b _ { r , \gamma } \colon \mathbb { R } \to  { \mathbb { R } }$ denotes the afine map $b _ { r , \gamma } ( x ) = r + \gamma x$ , and $g _ { \# } \mu$ is the pushforward of a measure $\mu$ under $g _ { \colon }$ , defined by $g _ { \# } \mu ( B ) ~ = ~ \mu ( g ^ { - 1 } ( B ) )$ for all Borel sets B. The integral $\begin{array} { r l } { \int _ { 0 } ^ { 1 } \big ( b _ { r , \gamma } \big ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) } & { { } } \end{array}$ is defined in the sense that for any Borel set $B .$

$$
\left[ \int _ { 0 } ^ { 1 } \left( b _ { r , \gamma } \right) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) \right] ( B ) = \int _ { 0 } ^ { 1 } \left[ ( b _ { r , \gamma } ) _ { \# } \eta ^ { \pi } ( s ^ { \prime } ) \right] ( B ) \mathcal { P } _ { R } ( \mathrm { d } r \mid s , a ) .
$$

Recall that $\mathcal { P }$ is the space of all probability measures on $\mathbb { R } .$ . The map $\mathcal { T } ^ { \pi } \colon \mathcal { P } ^ { S } \to \mathcal { P } ^ { S }$ is the distributional Bellman operator, whose fixed point is the return-distribution vector $\eta ^ { \pi }$

Because the exact distribution $\eta ^ { \pi }$ is infinite-dimensional and cannot be computed exactly, we

approximate it using a quantile-parameterized distribution. The space of all quantile-parametrized probability distributions is defined as

$$
\mathcal { P } _ { m } : = \left\{ \nu _ { x } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { x _ { i } } \mid x = ( x _ { 1 } , \ldots , x _ { m } ) ^ { \top } \in \mathbb { R } ^ { m } , x _ { 1 } \leq \ldots \leq x _ { m } \right\} ,
$$

which is a mixture of Dirac measures and $m \in \mathbb { N }$ . We define the quantile projection operator $\Pi _ { m } \colon { \mathcal { P } } \to { \mathcal { P } } _ { m }$ as

$$
\Pi _ { m } \nu = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { F _ { \nu } ^ { - 1 } ( \tau _ { i } ) } ,
$$

where $\begin{array} { r } { \tau _ { i } = \frac { 2 i - 1 } { 2 m } } \end{array}$ . We lift $\Pi _ { m }$ to the product space $\mathcal { P } ^ { S }$ by defining $( \pmb { \Pi } _ { m } \pmb { \eta } ) ( s ) : = \Pi _ { m } \eta ( s )$ for any $\pmb { \eta } = ( \eta ( s ) ) _ { s \in \mathcal { S } } \in \mathcal { P } ^ { S }$ . The properties of $\mathcal { T } ^ { \pi }$ and $\mathbf { I I } _ { m }$ are summarized in the following proposition.

Proposition 2.1. [2, Proposition 4.15, Lemma 5.25] The following statements hold:

$\mathcal { T } ^ { \pi }$ is $\gamma$ -contractive under the $W _ { p }$ metric for every $p \in [ 1 , \infty ]$ , namely for every η, $\pmb { \eta } ^ { \prime } \in \mathcal { P } ^ { S }$

$$
\bar { W } _ { p } ( T ^ { \pi } \eta , T ^ { \pi } \eta ^ { \prime } ) \le \gamma \bar { W } _ { p } ( \eta , \eta ^ { \prime } ) ;
$$

$\mathbf { I I } _ { m }$ is non-expansive under the $W _ { \infty }$ metric, namely $\bar { W } _ { \infty } ( \Pi _ { m } \eta , \Pi _ { m } \eta ^ { \prime } ) \leq \bar { W } _ { \infty } ( \eta , \eta ^ { \prime } )$ for every $\eta , \eta ^ { \prime } \in \mathcal { P } ^ { S }$

It immediately follows from Proposition 2.1 that the quantile projected Bellman operator $\Pi _ { m } \mathcal { T } ^ { \pi }$ is a γ-contraction in the Polish space $( \mathcal { P } ^ { S } , \bar { W } _ { \infty } )$ . Hence, the quantile projected Bellman equation $\pmb { \eta } = \pmb { \Pi } _ { m } \tau ^ { \pi } \pmb { \eta }$ admits a unique solution $\eta _ { m }$ . Denote $[ m ] = \{ 1 , \dots , m \}$ , and we define the quantile parameter $\pmb { \theta } _ { m } \in \mathbb { R } ^ { S \times [ m ] }$ by

$$
\eta _ { m } ( s ) = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta _ { m } ( s , i ) } ,
$$

with $\theta _ { m } ( s , 1 ) \leq \dots \leq \theta _ { m } ( s , m )$ for every $s \in { \mathcal { S } }$ . For the stochastic recursion, the coordinates need not remain ordered. We therefore use

$$
\eta _ { \theta } ( s ) : = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \delta _ { \theta ( s , i ) } , \qquad \theta \in \mathbb { R } ^ { S \times [ m ] } ,\tag{2}
$$

without imposing an ordering on $\pmb \theta .$

## 2.4 Quantile Temporal Diference Learning

We introduce synchronous quantile temporal diference learning (QTD) in this section. At every iteration t, for every $s \in { \mathcal { S } }$ , we sample

$$
\begin{array} { r } { a ^ { ( t , s ) } \sim \pi ( \cdot \mid s ) , \qquad \left( r ^ { ( t , s ) } , s ^ { \prime ( t , s ) } \right) \sim Q _ { s , a ^ { ( t , s ) } } } \end{array}
$$

from a generative model and update

$$
\theta ^ { ( t ) } ( s , i ) = \theta ^ { ( t - 1 ) } ( s , i ) + \alpha _ { t - 1 } \left[ \tau _ { i } - \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { 1 } \left\{ r ^ { ( t , s ) } + \gamma \theta ^ { ( t - 1 ) } \left( s ^ { \prime ( t , s ) } , j \right) < \theta ^ { ( t - 1 ) } ( s , i ) \right\} \right] ,\tag{3}
$$

where we define

$$
Q _ { s , a } = { \mathcal { P } } _ { R } ( \cdot \mid s , a ) \otimes P ( \cdot \mid s , a ) .
$$

In this paper, we use step sizes $\alpha _ { t } = c ( t + 1 ) ^ { - a }$ with $a \in ( 1 / 2 , 1 )$

## 2.5 Main Assumptions

We first impose regularity conditions on the reward distributions.

Assumption 1. For every $( s , a ) \in \mathcal { S } \times \mathcal { A }$ , the reward distribution $\mathcal { P } _ { R } ( \cdot \mid s , a )$ is supported on $[ 0 , 1 ]$ and admits a Lebesgue density $p _ { s , a }$ . There exist constants $C _ { 0 } \geq 1$ and $L \ge 0$ such that $0 < p _ { s , a } ( x ) \leq C _ { 0 }$ on (0, 1) and $p _ { s , a }$ is L-Lipschitz on (0, 1).

In addition, we assume the endpoint regularity conditions: either

(i) for any $s \in \mathcal S , \ a \in \mathcal A$ , there exists a positive constant $c _ { 0 }$ such that $p _ { s , a } ( x ) \ge c _ { 0 }$ for any $x \in ( 0 , 1 )$ ; or

(ii) for any $s \in S , a \in A$ , the zero extension of $p _ { s , a }$ to R is L-Lipschitz, $p _ { s , a } ( 0 ) = p _ { s , a } ( 1 ) = 0$ , and there exists $\kappa > 0$ such that $p _ { s , a }$ is increasing on $[ 0 , \kappa ]$ and decreasing on $[ 1 - \kappa , 1 ]$

The first condition assumes that the reward densities are uniformly bounded away from zero, a common assumption in quantile estimation. The second condition excludes highly oscillatory boundary behavior and ensures that the density approaches zero regularly. Typical examples include beta distributions with shape parameters $\alpha , \beta \geq 2$

Moreover, we make an assumption on the locations of $\pmb { \theta } _ { m }$

Assumption 2. For every $s , s ^ { \prime } \in \mathcal { S }$ satisfying

$$
P ^ { \pi } ( s ^ { \prime } \mid s ) : = \sum _ { a \in \cal { A } } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) > 0 ,
$$

we have

$$
\theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) \notin \{ 0 , 1 \}
$$

for every $i , j \in [ m ]$

This is a technical condition that excludes boundary-degenerate configurations in which Bellmanshifted quantile locations coincide with the endpoints of the reward support. We suppose Assumptions 1 and 2 hold throughout the paper.

## 2.6 Preliminary Lemmas

We first introduce a lemma from Cheng et al. [4].

Lemma 2.1. For every $m \in \mathbb { N }$ and $s \in { \mathcal { S } }$ , we have

$$
\operatorname* { i n f } _ { \substack { i \in [ m ] , 0 < | u | \leq ( 1 - \gamma ) ^ { - 1 } } } \left| \frac { F _ { ( \mathcal { T } ^ { \pi } \eta _ { m } ) ( s ) } \left( F _ { ( \mathcal { T } ^ { \pi } \eta _ { m } ) ( s ) } ^ { - 1 } ( \tau _ { i } ) + u \right) - \tau _ { i } } { \tau _ { i } ( 1 - \tau _ { i } ) u } \right| \geq c _ { \mathcal { M } } > 0 ,
$$

where $c _ { \mathcal { M } }$ is a constant that does not depend on m.

For later quantitative analysis, define the boundary identifiability margin

$$
\Delta _ { m } : = \operatorname* { m i n } _ { \substack { P ^ { \pi } ( s ^ { \prime } | s ) > 0 , i , j \in [ m ] } } \operatorname* { m i n } \left\{ \left| \theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) \right| , \left| \theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) - 1 \right| \right\} .
$$

By Assumption 2, we have $\Delta _ { m } > 0$

For $( s , i ) \in \mathcal { S } \times [ m ]$ , define

$$
d _ { s , i } : = p _ { ( \mathcal T ^ { \pi } \eta _ { m } ) ( s ) } \mathopen { } \mathclose \bgroup \left( \theta _ { m } \mathopen { } \mathclose \bgroup \left( s , i \aftergroup \egroup \right) \aftergroup \egroup \right) , \quad \underline { { d } } _ { m } : = \operatorname* { m i n } d _ { s , i } ,
$$

$$
v _ { m } : = \operatorname* { m a x } _ { s , i } \frac { \tau _ { i } ( 1 - \tau _ { i } ) } { d _ { s , i } } , \qquad \lambda _ { m } : = ( 1 - \gamma ) \underline { { d _ { m } } } .\tag{4}
$$

Taking $u \to 0$ in Lemma 2.1 gives

$$
d _ { s , i } \geq c _ { \mathcal { M } } \tau _ { i } ( 1 - \tau _ { i } ) , \qquad d _ { m } \geq \frac { c _ { \mathcal { M } } ( 2 m - 1 ) } { 4 m ^ { 2 } } , \qquad v _ { m } \leq \frac { 1 } { c _ { \mathcal { M } } } .\tag{5}
$$

Define

$$
r _ { \mathrm { o u t } } : = \left\{ \begin{array} { c c } { \displaystyle \operatorname* { m i n } \Biggl \{ \frac { 1 } { 1 - \gamma } , \frac { \Delta _ { m } } { 2 ( 1 + \gamma ) } , \frac { d _ { m } } { 2 L ( 1 + \gamma ) } , } & \\ { \displaystyle \frac { 2 m - 1 } { 8 m ^ { 2 } C _ { 0 } ( 1 + \gamma ) } , \frac { \lambda _ { m } } { 8 L ( 1 + \gamma ) ^ { 2 } } \Biggr \} , } & { \mathrm { i n ~ c a s e ~ ( i ) } , } \\ { \displaystyle \operatorname* { m i n } \Biggl \{ \frac { 1 } { 1 - \gamma } , \frac { d _ { m } } { 2 L ( 1 + \gamma ) } , \frac { 2 m - 1 } { 8 m ^ { 2 } C _ { 0 } ( 1 + \gamma ) } , } & \\ { \displaystyle \frac { \lambda _ { m } } { 8 L ( 1 + \gamma ) ^ { 2 } } \Biggr \} , } & { \mathrm { i n ~ c a s e ~ ( i i ) } . } \end{array} \right.\tag{6}
$$

Fractions with denominator $L = 0$ are interpreted $\mathrm { a s } + \infty$

The following lemma provides the local density lower bound used in our non-asymptotic analysis.

Lemma 2.2. For every θ satisfying $\lVert \pmb \theta - \pmb \theta _ { m } \rVert _ { \infty } \leq r _ { \mathrm { o u t } }$ and $z \ s a t i s f y i n g \ | z - \theta _ { m } ( s , i ) | \leq r _ { \mathrm { o u t } }$ , we have

$$
p _ { ( { \mathcal T } ^ { \pi } \eta _ { \theta } ) ( s ) } ( z ) \ge \frac { d _ { s , i } } { 2 } \ge \frac { c _ { { \mathcal M } } } { 2 } \tau _ { i } ( 1 - \tau _ { i } ) .\tag{7}
$$

## 3 Main Results

In this section, we establish a global non-asymptotic convergence guarantee for synchronous QTD. The result allows an arbitrary initialization in the natural parameter domain. Its local stochastic term retains the variance of the quantile score, rather than treating the martingale noise as an arbitrary bounded perturbation.

For $T \geq 8$ and $\delta \in ( 0 , 1 )$ , let

$$
\ell _ { T } ( \delta ) : = \log \frac { 1 6 | \mathcal { S } | m T ^ { 2 } } { \delta } , \qquad \varepsilon _ { T } ( \delta ) : = \sqrt { \frac { c v _ { m } \ell _ { T } ( \delta ) } { ( 1 - \gamma ) T ^ { a } } } + \frac { c \ell _ { T } ( \delta ) } { T ^ { a } } .\tag{8}
$$

The first term is the variance contribution and the second is the bounded-increment contribution in Freedman’s inequality.

Theorem 3.1. Suppose $\pmb { \theta } ^ { ( 0 ) } \in [ 0 , ( 1 - \gamma ) ^ { - 1 } ] ^ { S \times [ m ] }$ and $c C _ { 0 } \leq 1$ . For every $\delta \in ( 0 , 1 )$ and $T \ge T _ { 0 } ( \delta )$ where the explicit threshold $T _ { 0 } ( \delta )$ is given in Equation (34), with probability at least $1 - \delta$

$$
\Big \| \pmb { \theta } ^ { ( T ) } - \pmb { \theta } _ { m } \Big \| _ { \infty } \lesssim r _ { \mathrm { o u t } } \exp \left\{ - \frac { c \lambda _ { m } T ^ { 1 - a } } { C } \right\} + \varepsilon _ { T } ( \delta ) + \frac { L ( 1 + \gamma ) ^ { 2 } } { \lambda _ { m } } \varepsilon _ { T } ^ { 2 } ( \delta ) ,\tag{9}
$$

where $C$ is a universal constant. In particular, the definition of $T _ { 0 } ( \delta )$ ensures that the last term is

absorbed by $\varepsilon _ { T } ( \delta )$ , and hence

$$
\Big \| \pmb { \theta } ^ { ( T ) } - \pmb { \theta } _ { m } \Big \| _ { \infty } \lesssim r _ { \mathrm { o u t } } \exp \left\{ - \frac { c \lambda _ { m } T ^ { 1 - a } } { C } \right\} + \sqrt { \frac { c v _ { m } \ell _ { T } ( \delta ) } { ( 1 - \gamma ) T ^ { a } } } + \frac { c \ell _ { T } ( \delta ) } { T ^ { a } } .\tag{10}
$$

Theorem 3.1 separates the global transient from the local statistical fluctuation. By Equation (5),

$$
\lambda _ { m } \geq \frac { c _ { \mathcal { M } } ( 1 - \gamma ) ( 2 m - 1 ) } { 4 m ^ { 2 } } , \qquad v _ { m } \leq \frac { 1 } { c _ { \mathcal { M } } } .\tag{11}
$$

Consequently, the transient in Equation (10) is, in the worst case,

$$
\exp \left\{ - \frac { c c _ { \mathcal { M } } ( 1 - \gamma ) T ^ { 1 - a } } { C m } \right\} ,
$$

whereas the leading stochastic term is

$$
\sqrt { \frac { c \log ( | \mathcal { S } | m T / \delta ) } { c \mathcal { M } ( 1 - \gamma ) T ^ { a } } } .\tag{12}
$$

Thus, after the iterate has entered the local regime, the leading last-iterate fluctuation has no polynomial dependence on the number of quantiles. This improvement is not obtained by replacing Azuma’s inequality with Freedman’s inequality alone. It follows from matching the coordinatewise noise variance $\tau _ { i } ( 1 - \tau _ { i } )$ with the Bellman-target density $d _ { s , i }$ in the positive linearized dynamics.

The conclusion is not uniform in m from time zero. Both the entrance time and the deterministic transient depend on $\underline { d } _ { m } ^ { - 1 }$ , which can be of order $m .$ . Moreover, before the quadratic remainder in Equation (9) can be absorbed, one needs, up to logarithmic factors,

$$
T ^ { a } \gtrsim \frac { c m ^ { 2 } L ^ { 2 } ( 1 + \gamma ) ^ { 4 } } { c _ { \mathcal { M } } ^ { 3 } ( 1 - \gamma ) ^ { 3 } } .\tag{13}
$$

Accordingly, Equation (12) describes the leading stochastic fluctuation after an m-dependent burn-in, rather than an m-independent global sample complexity.

The proof uses two diferent stability mechanisms. The global entrance argument relies on the order monotonicity of reward CDFs and the $W _ { \infty }$ contraction of the distributional Bellman operator. Inside the local region, we instead linearize the mean field at $\pmb { \theta } _ { m }$ . The Jacobian is a nonsingular M-matrix, so its discrete semigroup is positive and substochastic. This positivity permits a variance-sensitive martingale argument that is unavailable in a generic Hurwitz-matrix

analysis.

## 4 Proof Outlines

The proof of Theorem 3.1 has a global comparison stage and a local linearization stage. The first stage uses monotonicity and Bellman contraction to obtain a late entrance into a neighborhood of $\pmb { \theta } _ { m }$ . The second stage uses the Jacobian of the QTD mean field, but crucially retains its positive M-matrix structure and the conditional variance of the quantile score.

For $t \geq 1$ , let $\mathcal { Z } _ { t }$ denote the full synchronous sample batch

$$
{ \mathcal { Z } } _ { t } : = \left\{ { \big ( } a ^ { ( t , s ) } , r ^ { ( t , s ) } , s ^ { \prime ( t , s ) } { \big ) } : s \in { \mathcal { S } } \right\} ,
$$

and set $\mathcal { F } _ { 0 } : = \sigma ( \pmb { \theta } ^ { ( 0 ) } )$ and $\mathcal { F } _ { t } : = \sigma ( \pmb { \theta } ^ { ( 0 ) } , \mathcal { Z } _ { 1 } , \ldots , \mathcal { Z } _ { t } )$ for $t \geq 1$ . Define

$$
h _ { s , i } ( \theta ) : = \frac { 1 } { m } \sum _ { a \in A , s ^ { \prime } \in S } \sum _ { j = 1 } ^ { m } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) F _ { s , a } \big ( \theta ( s , i ) - \gamma \theta ( s ^ { \prime } , j ) \big ) - \tau _ { i } ,\tag{14}
$$

$$
\widehat { h } _ { s , i } ^ { ( t + 1 ) } ( \pmb \theta ) : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { 1 } \left\{ r ^ { ( t + 1 , s ) } + \gamma \theta { ( s ^ { \prime ( t + 1 , s ) } , j ) } < \theta ( s , i ) \right\} - \tau _ { i } ,\tag{15}
$$

where $F _ { s , a }$ is the reward CDF. With

$$
\pmb \xi ^ { ( t + 1 ) } : = \widehat { \pmb h } ^ { ( t + 1 ) } ( \pmb \theta ^ { ( t ) } ) - \pmb h ( \pmb \theta ^ { ( t ) } ) ,\tag{16}
$$

the recursion becomes

$$
\pmb { \theta } ^ { ( t + 1 ) } = \pmb { \theta } ^ { ( t ) } - \alpha _ { t } \left\{ \pmb { h } ( \pmb { \theta } ^ { ( t ) } ) + \pmb { \xi } ^ { ( t + 1 ) } \right\} , \qquad \mathbb { E } \left[ \pmb { \xi } ^ { ( t + 1 ) } \ \vert \ \mathcal { F } _ { t } \right] = \mathbf { 0 } .\tag{17}
$$

## 4.1 Local Jacobian and Variance–Drift Matching

For $( s , i ) , ( s ^ { \prime } , j ) \in \mathcal { S } \times [ m ]$ , define

$$
\begin{array} { c } { { ( { \pmb B } _ { m } ) _ { ( s , i ) , ( s ^ { \prime } , j ) } : = \displaystyle \frac { 1 } { m } \sum _ { a \in \cal A } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) p _ { s , a } \big ( \theta _ { m } ( s , i ) - \gamma \theta _ { m } ( s ^ { \prime } , j ) \big ) , } } \\ { { { } } } \\ { { { \pmb D } _ { m } : = \mathrm { d i a g } \big ( ( d _ { s , i } ) _ { s \in { \pmb S } , i \in [ m ] } \big ) , \qquad { \pmb G } _ { m } : = { \pmb D } _ { m } - \gamma { \pmb B } _ { m } . } } \end{array}\tag{18}
$$

The following lemma records the structure that makes the sharp local bound possible.

Lemma 4.1. The QTD mean field is continuously diferentiable on the $r _ { \mathrm { o u t } }$ -neighborhood of $\pmb { \theta } _ { m }$ , and

$$
\nabla { \pmb h } \big ( \pmb \theta _ { m } \big ) = { \pmb G } _ { m } , \qquad { \pmb B } _ { m } { \pmb 1 } = D _ { m } { \pmb 1 } .\tag{19}
$$

Consequently, $G _ { m }$ is a nonsingular M-matrix and $- G _ { m }$ is Hurwitz. I $f 0 \leq \alpha C _ { 0 } \leq 1$ , then

$$
\begin{array} { r } { I - \alpha G _ { m } \geq \mathbf { 0 } , \qquad \left\| I - \alpha G _ { m } \right\| _ { \infty } \leq 1 - \alpha \lambda _ { m } . } \end{array}\tag{20}
$$

Moreover, for $\pmb { e } = \pmb { \theta } - \pmb { \theta } _ { m }$ with $\| e \| _ { \infty } \leq r _ { \mathrm { o u t } }$

$$
h ( \theta ) = G _ { m } e + R ( e ) , \qquad \left\| R ( e ) \right\| _ { \infty } \leq \frac { L ( 1 + \gamma ) ^ { 2 } } { 2 } \left\| e \right\| _ { \infty } ^ { 2 } .\tag{21}
$$

Finally, as long as $\left\| \theta ^ { ( t ) } - \theta _ { m } \right\| _ { \infty } \leq r _ { \mathrm { o u t } }$ , for every nonnegative vector ${ \pmb v } _ { ; }$

$$
\mathbb { E } \left[ \left( \pmb { v } ^ { \top } \pmb { \xi } ^ { ( t + 1 ) } \right) ^ { 2 } | \mathcal { F } _ { t } \right] \leq 2 v _ { m } ( \pmb { v } ^ { \top } \pmb { 1 } ) ( \pmb { v } ^ { \top } \pmb { D } _ { m } \pmb { 1 } ) .\tag{22}
$$

Equation (22) is specific to the quantile score. Each coordinate of the random CDF is in [0, 1]. Its conditional mean difers from $\tau _ { i }$ by at most $C _ { 0 } ( 1 + \gamma ) r _ { \mathrm { o u t } }$ , and the definition of $r _ { \mathrm { o u t } }$ therefore gives

$$
\mathbb { E } \left[ ( \xi _ { s , i } ^ { ( t + 1 ) } ) ^ { 2 } \mid \mathcal { F } _ { t } \right] \leq 2 \tau _ { i } ( 1 - \tau _ { i } ) \leq 2 v _ { m } d _ { s , i } .
$$

The proof of Lemma 4.1 is given in Appendix B.

## 4.2 Stage I: A Late Entrance into the Local Region

Fix $T \geq 8$ and write

$$
u : = \left\lfloor { \frac { T } { 4 } } \right\rfloor , \qquad v : = \left\lfloor { \frac { T } { 2 } } \right\rfloor .\tag{23}
$$

We restart the entrance argument at time u. This is important: once the iterate reaches the local region, all subsequent step sizes are of order $T ^ { - a }$ , so localization does not require choosing the global stepsize constant c as a function of m or $\delta .$

Define the late entrance time

$$
\tau _ { T } : = \operatorname* { i n f } \left\{ t \in \{ u , \dots , v \} : \left\| \pmb { \theta } ^ { ( t ) } - \pmb { \theta } _ { m } \right\| _ { \infty } \leq \frac { r _ { \mathrm { o u t } } } { 2 } \right\} ,\tag{24}
$$

with the convention inf $\mathcal { D } = \infty$

Let

$$
\begin{array} { r l } & { c _ { \mathrm { g } } : = \frac { c _ { \mathcal { M } } \tau _ { 1 } ( 1 - \tau _ { 1 } ) ( 1 - \gamma ) r _ { \mathrm { o u t } } } { 8 } , } \\ & { \beta : = \frac { 4 } { ( 1 - \gamma ) r _ { \mathrm { o u t } } } \log \left[ \frac { 2 | S | m ( 1 + 2 c _ { \mathrm { g } } ) } { c _ { \mathrm { g } } } \right] , } \\ & { D _ { \mathrm { g } } : = \frac { 1 + c } { 1 - \gamma } - \frac { r _ { \mathrm { o u t } } } { 2 } + \frac { \log ( 2 | S | m ) } { \beta } . } \end{array}\tag{25}
$$

For integers $r < q .$ , denote

$$
A _ { r , q } : = \sum _ { t = r } ^ { q - 1 } \alpha _ { t } , \qquad B _ { r , q } : = \sum _ { t = r } ^ { q - 1 } \alpha _ { t } ^ { 2 } .\tag{26}
$$

Lemma 4.2. For every $T \geq 8$

$$
\mathbb { P } ( \tau _ { T } = \infty ) \leq \exp \left\{ - \frac { \Big ( c _ { \mathrm { g } } A _ { u , v } - D _ { \mathrm { g } } - \frac { \beta } { 2 } B _ { u , v } \Big ) _ { + } ^ { 2 } } { 2 B _ { u , v } } \right\} .\tag{27}
$$

The proof uses the smooth maximum

$$
\Phi _ { \beta } ( e ) : = \frac { 1 } { \beta } \log \sum _ { s \in \mathcal { S } , i \in [ m ] } \left( e ^ { \beta e _ { s , i } } + e ^ { - \beta e _ { s , i } } \right) .
$$

For signed coordinates close to the maximum error, Bellman discounting creates an inward shift of order $( 1 - \gamma ) r _ { \mathrm { o u t } }$ . Lemma 2.1 and monotonicity of the reward CDFs turn this shift into a drift of at least $2 c _ { \mathrm { g } }$ . The choice of $\beta$ makes the total softmax weight of all other coordinates small enough to retain drift $c _ { \mathrm { g } }$ . A stopped Azuma–Hoefding argument over the block [u, v] gives Equation (27). The full proof is given in Appendix A.

## 4.3 Stage II: Variance-Sensitive Local Concentration

For $n \leq k$ define the deterministic linearized propagator

$$
\Phi _ { n , k } : = \prod _ { t = n } ^ { k - 1 } ( I - \alpha _ { t } G _ { m } ) , \qquad \Phi _ { k , k } : = I .\tag{28}
$$

All factors commute, and Equation (20) gives

$$
\begin{array} { r } { \Phi _ { n , k } \geq \mathbf { 0 } , \qquad \left\| \Phi _ { n , k } \right\| _ { \infty } \leq \exp \left\{ - \lambda _ { m } A _ { n , k } \right\} . } \end{array}\tag{29}
$$

Lemma 4.3. There exists a universal constant $C _ { \star }$ with the following property. Let n be an

(F<sub>t</sub>)-stopping time satisfying $u \leq n \leq v$ and

$$
\left. \theta ^ { ( n ) } - \theta _ { m } \right. _ { \infty } \leq \frac { r _ { \mathrm { o u t } } } { 2 } .
$$

If

$$
C _ { \star } \varepsilon _ { T } ( \delta ) \leq \operatorname* { m i n } \left\{ r _ { \mathrm { o u t } } , \frac { \lambda _ { m } } { L ( 1 + \gamma ) ^ { 2 } } \right\} ,\tag{30}
$$

then, conditionally on ${ \mathcal { F } } _ { n }$ , with probability at least $1 - \delta / 2$ , the iterates remain in the $r _ { \mathrm { o u t } }$ -neighborhood through time T and

$$
\Big \| \pmb { \theta } ^ { ( T ) } - \pmb { \theta } _ { m } \Big \| _ { \infty } \lesssim r _ { \mathrm { o u t } } \exp \left\{ - \frac { \lambda _ { m } A _ { v , T } } { C } \right\} + \varepsilon _ { T } ( \delta ) + \frac { L ( 1 + \gamma ) ^ { 2 } } { \lambda _ { m } } \varepsilon _ { T } ^ { 2 } ( \delta ) .\tag{31}
$$

The key calculation behind this lemma is a discrete hazard identity. For a row vector

$$
\begin{array} { r } { \pmb { v } _ { t } ^ { \top } : = \pmb { e } _ { q } ^ { \top } \pmb { \Phi } _ { t + 1 , k } , } \end{array}
$$

positivity gives $\mathbf { \sigma } _ { { v } _ { t } } \geq \mathbf { 0 }$ and $\pmb { v } _ { t } ^ { \top } \pmb { 1 } \leq 1$ , while

$$
\begin{array} { r } { { \pmb v } _ { t } ^ { \top } { \bf 1 } - { \pmb v } _ { t } ^ { \top } ( { \pmb I } - \alpha _ { t } { \pmb G } _ { m } ) { \bf 1 } = ( 1 - \gamma ) \alpha _ { t } { \pmb v } _ { t } ^ { \top } { \pmb D } _ { m } { \bf 1 } . } \end{array}\tag{32}
$$

Combining Equations (22) and (32) yields

$$
\sum _ { t = n } ^ { k - 1 } \alpha _ { t } ^ { 2 } \mathbb { E } \left[ \left( \pmb { v } _ { t } ^ { \top } \pmb { \xi } ^ { ( t + 1 ) } \right) ^ { 2 } \mid \mathcal { F } _ { t } \right] \lesssim \frac { v _ { m } \alpha _ { n } } { 1 - \gamma } .\tag{33}
$$

Freedman’s inequality therefore produces a stochastic term of order $\sqrt { { v } _ { m } \alpha _ { n } / ( 1 - \gamma ) }$ , with no factor $\underline { { d } } _ { m } ^ { - 1 / 2 }$ . A stopped-process argument and a deterministic noise-removal bootstrap absorb the quadratic remainder in Equation (21). Details are given in Appendix A.

## 4.4 Proof of Theorem 3.1

Let $T _ { 0 } ( \delta )$ be the smallest integer $T \geq 8$ such that, with u and v defined in Equation (23),

$$
\begin{array} { r l r } & { \displaystyle { c _ { \mathrm { g } } A _ { u , v } \geq 2 D _ { \mathrm { g } } + \beta B _ { u , v } } , } & \\ & { \displaystyle { \frac { c _ { \mathrm { g } } ^ { 2 } A _ { u , v } ^ { 2 } } { 8 B _ { u , v } } \geq \log \frac { 2 } { \delta } , } } & \\ & { \displaystyle { C _ { \star } \varepsilon _ { T } ( \delta ) \leq \operatorname* { m i n } \left\{ r _ { \mathrm { o u t } } , \frac { \lambda _ { m } } { L ( 1 + \gamma ) ^ { 2 } } \right\} . } } & \end{array}\tag{34}
$$

When $L = 0$ , the last fraction is interpreted as $+ \infty$ . Since $a \in ( 1 / 2 , 1 )$ , these conditions hold for all suficiently large $T .$ . In particular, the block estimates $A _ { u , v } \gtrsim c T ^ { 1 - a }$ and $B _ { u , v } \lesssim c ^ { 2 } T ^ { 1 - 2 a }$ show that a suficient threshold, up to logarithmic factors, is

$$
\begin{array} { c } { \displaystyle { T _ { 0 } ( \delta ) \lesssim \left( \frac { D _ { \mathrm { g } } } { c c _ { \mathrm { g } } } \right) ^ { \frac { 1 } { 1 - a } } + \frac 1 { c _ { \mathrm { g } } ^ { 2 } } \log \frac 2 \delta + \left( \frac { \beta c } { c _ { \mathrm { g } } } \right) ^ { \frac 1 a } } } \\ { \displaystyle { + \left\{ \frac { c v _ { m } } { 1 - \gamma } \operatorname* { m a x } \left( \frac 1 { r _ { \mathrm { o u t } } ^ { 2 } } , \frac { L ^ { 2 } ( 1 + \gamma ) ^ { 4 } } { \lambda _ { m } ^ { 2 } } \right) \right\} ^ { \frac 1 a } , } } \end{array}\tag{35}
$$

where logarithmic factors in the displayed parameters are suppressed.

The first two inequalities in Equation (34) and Lemma 4.2 give

$$
\mathbb { P } ( \tau _ { T } = \infty ) \leq \frac { \delta } { 2 } .
$$

On $\{ \tau _ { T } < \infty \}$ , the random time $\tau _ { T }$ lies in $[ u , v ]$ and the iterate at $\tau _ { T }$ lies in the half-radius ball. Applying Lemma 4.3 conditionally on $\mathcal { F } _ { \tau _ { T } }$ and then summing over the disjoint events $\{ \tau _ { T } = t \} , u \le t \le v ,$ , shows that Equation (31) fails with probability at most $\delta / 2$ . A union bound proves Equation (9). The last inequality in Equation (34) absorbs its quadratic term and yields Equation (10).

## 5 Conclusions

We established a global high-probability bound for the raw last iterate of synchronous QTD. The analysis separates global localization from local fluctuation. Monotonicity of reward cumulative distribution functions and distributional Bellman contraction yield a finite-time entrance into a local region. Within that region, the QTD Jacobian is a nonsingular M-matrix. Its positive semigroup, combined with the variance of the quantile score, gives a leading stochastic term with no polynomial dependence on the number of quantiles and with the sharp square-root dependence on the efective Bellman horizon.

The distinction between fluctuation and burn-in is essential. The smallest Bellman-target density can be of order $m ^ { - 1 }$ , so the deterministic transient, the localization radius, and the sample size needed to absorb the nonlinear remainder can still depend on m. The theorem therefore does not claim a uniformly m-free global sample complexity.

Several extensions remain open. It would be useful to obtain matching lower bounds for the global burn-in, to study asynchronous and Markovian sampling, and to determine which parts of the positive-semigroup argument survive under function approximation. Another direction is to compare raw last iterates with Polyak–Ruppert averages at finite time: averaging improves the rate in T but can introduce an additional inverse-Jacobian factor and hence a diferent dependence on extreme-quantile densities.

## References

[1] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement learning. In International conference on machine learning, pages 449–458. PMLR, 2017.

[2] M. G. Bellemare, W. Dabney, and M. Rowland. Distributional Reinforcement Learning. MIT Press, 2023. http://www.distributional-rl.org.

[3] Z. Cheng, Y. Peng, and Z. Zhang. Online inference for quantile temporal diference learning in distributional reinforcement learning. arXiv preprint arXiv:2608.12973, 2026.

[4] Z. Cheng, Y. Peng, and Z. Zhang. Statistical eficiency and inference of quantile distributional reinforcement learning. arXiv preprint arXiv:2607.08444, 2026.

[5] W. Dabney, G. Ostrovski, D. Silver, and R. Munos. Implicit quantile networks for distributional reinforcement learning. In International conference on machine learning, pages 1096–1105. PMLR, 2018.

[6] W. Dabney, M. Rowland, M. Bellemare, and R. Munos. Distributional reinforcement learning with quantile regression. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

[7] T. Morimura, M. Sugiyama, H. Kashima, H. Hachiya, and T. Tanaka. Nonparametric return distribution approximation for reinforcement learning. In Proceedings of the 27th International Conference on Machine Learning (ICML-10), pages 799–806, 2010.

[8] Y. Peng, L. Zhang, and Z. Zhang. Statistical eficiency of distributional temporal diference learning. Advances in Neural Information Processing Systems, 37:24724–24761, 2024.

[9] Y. Peng, K. Jin, L. Zhang, and Z. Zhang. A finite sample analysis of distributional td learning with linear function approximation. arXiv preprint arXiv:2502.14172, 2025.

[10] M. Rowland, M. Bellemare, W. Dabney, R. Munos, and Y. W. Teh. An analysis of categorical distributional reinforcement learning. In International Conference on Artificial Intelligence and Statistics, pages 29–37. PMLR, 2018.

[11] M. Rowland, R. Munos, M. G. Azar, Y. Tang, G. Ostrovski, A. Harutyunyan, K. Tuyls, M. G. Bellemare, and W. Dabney. An analysis of quantile temporal-diference learning. arXiv preprint arXiv:2301.04462, 2023.

[12] M. Rowland, Y. Tang, C. Lyle, R. Munos, M. G. Bellemare, and W. Dabney. The statistical benefits of quantile temporal-diference learning for value estimation. In International Conference on Machine Learning, pages 29210–29231. PMLR, 2023.

## A Omitted Proofs in Section 4

## A.1 Proof of Lemma 4.2

Proof. Set $e ^ { ( t ) } : = \pmb { \theta } ^ { ( t ) } - \pmb { \theta } _ { m }$ and consider

$$
\Phi _ { \beta } ( e ) : = \frac { 1 } { \beta } \log \sum _ { s \in \mathcal { S } , i \in [ m ] } \left( e ^ { \beta e _ { s , i } } + e ^ { - \beta e _ { s , i } } \right) .\tag{36}
$$

Lemma B.2 gives

$$
\| e \| _ { \infty } \leq \Phi _ { \beta } ( e ) \leq \| e \| _ { \infty } + \frac { \log ( 2 | S | m ) } { \beta } ,\tag{37}
$$

$$
\begin{array} { r } { \big \| \nabla \Phi _ { \beta } ( e ) \big \| _ { 1 } \leq 1 , \qquad \pmb { u } ^ { \top } \nabla ^ { 2 } \Phi _ { \beta } ( e ) \pmb { u } \leq \beta \left\| \pmb { u } \right\| _ { \infty } ^ { 2 } , } \end{array}\tag{38}
$$

and, whenever $\lVert \pmb \theta - \pmb \theta _ { m } \rVert _ { \infty } \geq r _ { \mathrm { o u t } } / 2$

$$
\langle \nabla \Phi _ { \beta } ( \pmb \theta - \pmb \theta _ { m } ) , \pmb h ( \pmb \theta ) \rangle \geq c _ { \mathrm { g } } .\tag{39}
$$

For $t \geq u ,$ define

$$
Z _ { t + 1 } : = - \left. \nabla \Phi _ { \beta } ( e ^ { ( t ) } ) , \pmb { \xi } ^ { ( t + 1 ) } \right. .
$$

Then $\mathbb { E } [ Z _ { t + 1 } \mid \mathcal { F } _ { t } ] = 0$ and $| Z _ { t + 1 } | \le 1$ . Indeed, every coordinate of $\pmb { \xi } ^ { ( t + 1 ) }$ is the diference of two numbers in [0, 1]. Moreover,

$$
\begin{array} { r } { \pmb { h } ( \pmb { \theta } ^ { ( t ) } ) + \pmb { \xi } ^ { ( t + 1 ) } = \pmb { \widehat { h } } ^ { ( t + 1 ) } ( \pmb { \theta } ^ { ( t ) } ) , \qquad \left\| \widehat { \pmb { h } } ^ { ( t + 1 ) } ( \pmb { \theta } ^ { ( t ) } ) \right\| _ { \infty } \leq 1 . } \end{array}
$$

Thus, on $\{ t < \tau _ { T } \}$ , Taylor’s theorem and Equations (17)–(39) imply

$$
\Phi _ { \beta } ( e ^ { ( t + 1 ) } ) \leq \Phi _ { \beta } ( e ^ { ( t ) } ) - c _ { \mathrm { g } } \alpha _ { t } + \alpha _ { t } Z _ { t + 1 } + \frac \beta 2 \alpha _ { t } ^ { 2 } .\tag{40}
$$

The stopped process

$$
M _ { k } : = \sum _ { t = u } ^ { k - 1 } \alpha _ { t } \mathbf { 1 } \big \{ t < \tau _ { T } \big \} Z _ { t + 1 } , \qquad u \leq k \leq v ,
$$

is a martingale with increments bounded by $\alpha _ { t }$ . Lemma B.1 and Equation (37) give

$$
\Phi _ { \beta } ( e ^ { ( u ) } ) \leq \frac { 1 + c } { 1 - \gamma } + \frac { \log ( 2 | S | m ) } { \beta } .
$$

On $\left\{ \tau _ { T } = \infty \right\}$ , summing Equation (40) from u to $v - 1$ and using $\Phi _ { \beta } ( e ^ { ( v ) } ) > r _ { \mathrm { o u t } } / 2$ yields

$$
M _ { v } \ge c _ { \mathrm { g } } A _ { u , v } - D _ { \mathrm { g } } - \frac \beta 2 B _ { u , v } .
$$

The Azuma–Hoefding inequality gives, for every $x \geq 0$

$$
\mathbb { P } ( M _ { v } \ge x ) \le \exp \left\{ - \frac { x ^ { 2 } } { 2 B _ { u , v } } \right\} .
$$

Combining the preceding two displays proves Equation (27).

## A.2 Proof of Lemma 4.3

Proof. Condition on ${ \mathcal { F } } _ { n }$ and write

$$
e ^ { ( t ) } : = \theta ^ { ( t ) } - \theta _ { m } , \qquad \sigma : = \operatorname* { i n f } \left\{ t \geq n : \left\| e ^ { ( t ) } \right\| _ { \infty } > r _ { \mathrm { o u t } } \right\} .
$$

For $t < \sigma ,$ Lemma 4.1 and Equation (17) give

$$
\pmb { e } ^ { ( t + 1 ) } = ( \pmb { I } - \alpha _ { t } \pmb { G } _ { m } ) \pmb { e } ^ { ( t ) } - \alpha _ { t } \pmb { \xi } ^ { ( t + 1 ) } - \alpha _ { t } \pmb { R } ( \pmb { e } ^ { ( t ) } ) .\tag{41}
$$

Consequently, for every $n < k \leq \sigma$

$$
\pmb { e } ^ { ( k ) } = \Phi _ { n , k } \pmb { e } ^ { ( n ) } - \sum _ { t = n } ^ { k - 1 } \alpha _ { t } \Phi _ { t + 1 , k } \pmb { \xi } ^ { ( t + 1 ) } - \sum _ { t = n } ^ { k - 1 } \alpha _ { t } \Phi _ { t + 1 , k } \pmb { R } ( \pmb { e } ^ { ( t ) } ) .\tag{42}
$$

Apply Lemma B.3 to the stopped diferences ${ \bf 1 } \{ t < \sigma \} \pmb { \xi } ^ { ( t + 1 ) }$ . Since $n \geq u$ and the stepsizes are nonincreasing, with conditional probability at least $1 - \delta / 2$

$$
\operatorname* { m a x } _ { n \leq r < k \leq T } \left. \sum _ { t = r } ^ { k - 1 } \alpha _ { t } \Phi _ { t + 1 , k } \mathbf { 1 } \{ t < \sigma \} \boldsymbol { \xi } ^ { ( t + 1 ) } \right. _ { \infty } \lesssim \varepsilon _ { T } ( \delta ) .\tag{43}
$$

The factor $T ^ { 2 }$ in $\ell _ { T } ( \delta )$ accounts for the possible pairs $( r , k )$

We work on the event in Equation (43). By Equation (21), the nonlinear term in Equation (42) satisfies

$$
\left\| R ( e ^ { ( t ) } ) \right\| _ { \infty } \leq L _ { h } \left\| e ^ { ( t ) } \right\| _ { \infty } ^ { 2 } , \qquad L _ { h } : = { \frac { L ( 1 + \gamma ) ^ { 2 } } { 2 } } .
$$

Lemma B.4, applied with

$$
P _ { t } = I - \alpha _ { t } G _ { m } , \qquad \zeta ^ { ( t + 1 ) } = - \alpha _ { t } { \bf 1 } \{ t < \sigma \} \xi ^ { ( t + 1 ) } ,
$$

$\mu = \lambda _ { m } , r _ { 0 } = r _ { \mathrm { o u t } }$ , and $b \lesssim \varepsilon _ { T } ( \delta )$ , now gives

$$
\begin{array} { r l } & { \displaystyle \operatorname* { m a x } _ { n \leq k \leq T \wedge \sigma } \left\| e ^ { ( k ) } \right\| _ { \infty } < r _ { \mathrm { o u t } } , } \\ & { \qquad \quad \left\| e ^ { ( T \wedge \sigma ) } \right\| _ { \infty } \lesssim r _ { \mathrm { o u t } } \exp \left\{ - \frac { \lambda _ { m } A _ { v , T } } { C } \right\} + \varepsilon _ { T } ( \delta ) + \frac { L _ { h } } { \lambda _ { m } } \varepsilon _ { T } ^ { 2 } ( \delta ) . } \end{array}\tag{44}
$$

Here we used $n \leq v .$ , so $A _ { n , T } \geq A _ { v , T }$ . The radius condition in Equation (6) gives $L _ { h } r _ { \mathrm { o u t } } / \lambda _ { m } \leq 1 / 1 6$ while Equation (30) supplies the remaining small-noise conditions of Lemma B.4.

If $\sigma \leq T$ , the first line of Equation (44), evaluated at $k = \sigma .$ , contradicts the definition of $\sigma .$ Hence $\sigma > T$ , and the second line is precisely Equation (31). Since Equation (43) holds with conditional probability at least $1 - \delta / 2$ , the result follows. □

## B Technical Lemmas

## B.1 Local Density Stability

Proof of Lemma 2.2. For every $z \in \mathbb { R }$ and $\pmb \theta \in \mathbb R ^ { S \times [ m ] }$ 2

$$
p _ { ( \mathcal T ^ { \pi } \eta _ { \theta } ) ( s ) } ( z ) = \frac { 1 } { m } \sum _ { a \in A , s ^ { \prime } \in \mathcal S } \sum _ { j = 1 } ^ { m } \pi ( a \mid s ) P ( s ^ { \prime } \mid s , a ) p _ { s , a } ( z - \gamma \theta ( s ^ { \prime } , j ) ) .\tag{45}
$$

If $\left\| \pmb { \theta } - \pmb { \theta } _ { m } \right\| _ { \infty } \leq r _ { \mathrm { o u t } }$ and $| z - \theta _ { m } ( s , i ) | \leq r _ { \mathrm { o u t } }$ , then

$$
\begin{array} { r } { | z - \gamma \theta ( s ^ { \prime } , j ) - \theta _ { m } ( s , i ) + \gamma \theta _ { m } ( s ^ { \prime } , j ) | \vphantom { s ^ { 2 } } } \\ { \leq ( 1 + \gamma ) r _ { \mathrm { o u t } } . } \end{array}\tag{46}
$$

Under condition (i) of Assumption 1, the $\Delta _ { m }$ term in Equation (6) ensures that the two arguments in Equation (46) belong to the same connected component of $\mathbb { R } \setminus \{ 0 , 1 \}$ . The density is L-Lipschitz on the interior component and is zero on the two exterior components. Under condition (ii), its zero

extension is L-Lipschitz on all of R. Thus, in both cases,

$$
\begin{array} { c l c r } { \displaystyle { \Big | p _ { ( \mathcal T ^ { \pi } \eta _ { \theta } ) ( s ) } ( z ) - p _ { ( \mathcal T ^ { \pi } \eta _ { m } ) ( s ) } \big ( \theta _ { m } ( s , i ) \big ) \big | } } \\ { \displaystyle { \le L ( 1 + \gamma ) r _ { \mathrm { o u t } } \le \frac { d _ { m } } { 2 } \le \frac { d _ { s , i } } { 2 } . } } \end{array}
$$

The definition of $d _ { s , i }$ and Equation (5) now give Equation (7).

## B.2 Global Comparison Lemmas

Lemma B.1. Let

$$
B : = \frac { 1 } { 1 - \gamma } , \qquad \Delta : = \frac { c } { 1 - \gamma } .
$$

If $\pmb { \theta } ^ { ( 0 ) } \in [ 0 , B ] ^ { S \times [ m ] }$ , then, almost surely,

$$
\begin{array} { r } { \pmb { \theta } ^ { ( t ) } \in [ - \Delta , B + \Delta ] ^ { \mathcal { S } \times [ m ] } \qquad f o r \ e v e r y \ t \geq 0 . } \end{array}\tag{47}
$$

Consequently,

$$
\left. \theta ^ { ( t ) } - \theta _ { m } \right. _ { \infty } \leq \frac { 1 + c } { 1 - \gamma } \qquad f o r \ e v e r y \ t \geq 0 .\tag{48}
$$

Proof. We prove Equation (47) by induction. Suppose all coordinates of ${ \pmb \theta } ^ { ( t ) }$ belong to $[ - \Delta , B + \Delta ]$ If $\theta ^ { ( t ) } ( s , i ) < - \gamma \Delta$ , then every sampled target is at least $- \gamma \Delta$ , so the update of this coordinate is nonnegative. Otherwise, since $\alpha _ { t } \leq c ,$

$$
\theta ^ { ( t + 1 ) } ( s , i ) \geq - \gamma \Delta - c = - \Delta .
$$

Similarly, every sampled target is at most $1 + \gamma ( B + \Delta ) = B + \gamma \Delta . \mathrm { ~ I f ~ } \theta ^ { ( t ) } ( s , i ) > B + \gamma \Delta$ , its update is nonpositive; otherwise

$$
\theta ^ { ( t + 1 ) } ( s , i ) \leq B + \gamma \Delta + c = B + \Delta .
$$

This proves the invariant box. Since $\pmb { \theta } _ { m } \in [ 0 , B ] ^ { S \times [ m ] }$ , Equation (48) follows.

Lemma B.2. Let $c _ { \mathrm { g } }$ and β be defined in Equation (25), and let $\Phi _ { \beta }$ be defined in Equation (36). For every e, $\pmb { u } \in \mathbb { R } ^ { S \times [ m ] }$ 2

$$
\| e \| _ { \infty } \leq \Phi _ { \beta } ( e ) \leq \| e \| _ { \infty } + \frac { \log ( 2 | S | m ) } { \beta } ,\tag{49}
$$

and

$$
\begin{array} { r } { \big \| \nabla \Phi _ { \beta } ( e ) \big \| _ { 1 } \leq 1 , \qquad \pmb { u } ^ { \top } \nabla ^ { 2 } \Phi _ { \beta } ( e ) \pmb { u } \leq \beta \left\| \pmb { u } \right\| _ { \infty } ^ { 2 } . } \end{array}\tag{50}
$$

Moreover, i $\dot { \mathbf { \zeta } } e = \pmb { \theta } - \pmb { \theta } _ { m }$ and $\| e \| _ { \infty } \ge r _ { \mathrm { o u t } } / 2$ , then

$$
\langle \nabla \Phi _ { \beta } ( e ) , h ( \pmb \theta ) \rangle \geq c _ { \mathrm { g } } .\tag{51}
$$

Proof. Put $V : = \| e \| _ { \infty }$ . The log-sum-exp bounds in Equation (49) are immediate. For $q = ( s , i ) \in$ $s \times [ m ]$ and $\sigma \in \{ - 1 , 1 \}$ , define

$$
w _ { q , \sigma } ( e ) : = \frac { \exp ( \beta \sigma e _ { q } ) } { \sum _ { r \in \mathcal { S } \times [ m ] } \{ \exp ( \beta e _ { r } ) + \exp ( - \beta e _ { r } ) \} } .
$$

If X is the random signed basis vector that equals $\sigma e _ { q }$ with probability $w _ { q , \sigma } ( e )$ , then

$$
\nabla ^ { 2 } \Phi _ { \beta } ( e ) = \beta \operatorname { C o v } ( X ) .
$$

This identity proves Equation (50).

It remains to prove the drift bound. Consider a signed coordinate with

$$
\sigma e _ { s , i } \geq V - \frac { ( 1 - \gamma ) r _ { \mathrm { { o u t } } } } { 4 } .\tag{52}
$$

If $\sigma = 1$ , then for every $( s ^ { \prime } , j )$

$$
e _ { s , i } - \gamma e _ { s ^ { \prime } , j } \geq ( 1 - \gamma ) V - \frac { ( 1 - \gamma ) r _ { \mathrm { o u t } } } 4 \geq \frac { ( 1 - \gamma ) r _ { \mathrm { o u t } } } 4 .
$$

Monotonicity of every reward CDF and Lemma 2.1 therefore give

$$
\begin{array} { l } { \displaystyle h _ { s , i } ( \pmb \theta ) \geq F _ { ( \mathcal T ^ { \pi } \eta _ { m } ) ( s ) } \left( \theta _ { m } ( s , i ) + \frac { \left( 1 - \gamma \right) r _ { \mathrm { o u t } } } { 4 } \right) - \tau _ { i } } \\ { \displaystyle \geq \frac { c _ { \mathcal M } \tau _ { i } ( 1 - \tau _ { i } ) ( 1 - \gamma ) r _ { \mathrm { o u t } } } { 4 } \geq 2 c _ { \mathrm { g } } . } \end{array}
$$

The same argument with all inequalities reversed shows that $- h _ { s , i } ( \pmb { \theta } ) \geq 2 c _ { \mathrm { g } }$ when $\sigma = - 1$

The total softmax weight of signed coordinates that do not satisfy Equation (52) is at most

$$
2 | S | m \exp \left\{ - \frac { \beta ( 1 - \gamma ) r _ { \mathrm { o u t } } } { 4 } \right\} = \frac { c _ { \mathrm { g } } } { 1 + 2 c _ { \mathrm { g } } } .
$$

Since $| h _ { s , i } ( \pmb { \theta } ) | \leq 1$ , averaging the signed drift with the weights $w _ { q , \sigma }$ proves Equation (51). □

## B.3 Local Linearization Lemmas

Proof of Lemma $4 . 1 .$ The no-boundary condition and Equation (6) ensure that, throughout the local ball, every density argument either stays in (0, 1) or stays in one of the two exterior components. Under condition (ii), the zero extension is Lipschitz across the endpoints as well. Diferentiating Equation (14) at $\pmb { \theta } _ { m }$ consequently gives

$$
\nabla h ( \pmb \theta _ { m } ) = D _ { m } - \gamma B _ { m } = \pmb G _ { m } .
$$

Summing a row of $B _ { m }$ over $( s ^ { \prime } , j )$ and using Equation (45) gives $B _ { m } \mathbf { 1 } = D _ { m } \mathbf { 1 }$

Let $\pmb { K } _ { m } : = \pmb { D } _ { m } ^ { - 1 } \pmb { B } _ { m }$ . Then ${ \cal K } _ { m } \ge 0$ and $K _ { m } \mathbf { 1 } = \mathbf { 1 }$ , while

$$
G _ { m } = D _ { m } ( I - \gamma K _ { m } ) , \qquad G _ { m } ^ { - 1 } = \sum _ { r = 0 } ^ { \infty } \gamma ^ { r } K _ { m } ^ { r } D _ { m } ^ { - 1 } \geq \mathbf { 0 } .\tag{53}
$$

Thus $G _ { m }$ is a nonsingular M-matrix and $- G _ { m }$ is Hurwitz. Moreover, $d _ { s , i } \leq C _ { 0 }$ , so $0 \leq \alpha C _ { 0 } \leq 1$ implies $\pmb { I } - \alpha \pmb { G } _ { m } \ge \mathbf { 0 }$ . Its row sums are

$$
\begin{array} { r } { ( I - \alpha G _ { m } ) \mathbf { 1 } = \mathbf { 1 } - \alpha \big ( 1 - \gamma \big ) D _ { m } \mathbf { 1 } \leq ( 1 - \alpha \lambda _ { m } ) \mathbf { 1 } , } \end{array}
$$

which proves Equation (20).

For a density argument at the fixed point, let $u = e _ { s , i } - \gamma e _ { s ^ { \prime } , j }$ . The L-Lipschitz property gives

$$
| F _ { s , a } ( x + u ) - F _ { s , a } ( x ) - p _ { s , a } ( x ) u | \leq \frac { L } { 2 } u ^ { 2 } .
$$

Averaging this inequality and using $\left| u \right| \leq \left( 1 + \gamma \right) \left\| e \right\| _ { \infty }$ proves Equation (21).

Finally, let

$$
Y _ { s , i } ^ { ( t + 1 ) } : = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \mathbf { 1 } \left\{ r ^ { ( t + 1 , s ) } + \gamma \theta ^ { ( t ) } \big ( s ^ { \prime ( t + 1 , s ) } , j \big ) < \theta ^ { ( t ) } ( s , i ) \right\}
$$

and $q _ { s , i } ^ { ( t ) } : = \mathbb { E } [ Y _ { s , i } ^ { ( t + 1 ) } \mid \mathcal { F } _ { t } ]$ . Since $Y _ { s , i } ^ { ( t + 1 ) } \in [ 0 , 1 ]$

$$
\mathbb { E } [ ( \xi _ { s , i } ^ { ( t + 1 ) } ) ^ { 2 } \mid \mathcal { F } _ { t } ] \leq q _ { s , i } ^ { ( t ) } ( 1 - q _ { s , i } ^ { ( t ) } ) .
$$

The target CDF has density at most $C _ { 0 }$ , and hence

$$
\left| q _ { s , i } ^ { ( t ) } - \tau _ { i } \right| \leq C _ { 0 } ( 1 + \gamma ) \left\| \pmb { \theta } ^ { ( t ) } - \pmb { \theta } _ { m } \right\| _ { \infty } \leq \frac { 1 } { 2 } \tau _ { i } ( 1 - \tau _ { i } ) ,
$$

where the last inequality follows from Equation (6). Consequently,

$$
\mathbb { E } [ ( \xi _ { s , i } ^ { ( t + 1 ) } ) ^ { 2 } \mid \mathcal { F } _ { t } ] \leq 2 \tau _ { i } ( 1 - \tau _ { i } ) \leq 2 v _ { m } d _ { s , i } .\tag{54}
$$

For every $\mathbf { \nabla } v \geq \mathbf { 0 }$ , weighted Cauchy–Schwarz gives

$$
( \pmb { v } ^ { \top } \pmb { \xi } ^ { ( t + 1 ) } ) ^ { 2 } \leq ( \pmb { v } ^ { \top } \pmb { 1 } ) \sum _ { s , i } v _ { s , i } ( \pmb { \xi } _ { s , i } ^ { ( t + 1 ) } ) ^ { 2 } .
$$

Taking conditional expectations and applying Equation (54) proves Equation (22).

Lemma B.3. Let n be an $( \mathcal { F } _ { t } )$ -stopping time with $u \leq n \leq v ,$ and let $\sigma \geq n$ be a stopping time such that $\left\| \theta ^ { ( t ) } - \theta _ { m } \right\| _ { \infty } \leq r _ { \mathrm { o u t } }$ on $\{ t < \sigma \}$ . For every $\eta \in ( 0 , 1 )$ , conditionally on $\mathcal { F } _ { n }$ , with probability at least $1 - \eta$

$$
\begin{array} { r l r } & { } & { \underset { q \in \cal S \times [ m ] } { \operatorname* { m a x } } \underset { n \leq r < k \leq T } { \operatorname* { m a x } } \left| \sum _ { t = r } ^ { k - 1 } \alpha _ { t } e _ { q } ^ { \top } \Phi _ { t + 1 , k } \mathbf { 1 } \{ t < \sigma \} \pmb { \xi } ^ { ( t + 1 ) } \right| } \\ & { } & { \leq 2 \sqrt { \frac { v _ { m } \alpha _ { n } } { 1 - \gamma } } \log \frac { 2 | \cal S | m T ^ { 2 } } { \eta } + \frac { \alpha _ { n } } { 3 } \log \frac { 2 | \cal S | m T ^ { 2 } } { \eta } . } \end{array}\tag{55}
$$

Proof. Fix $q , r ,$ k and put

$$
\pmb { v } _ { t } ^ { \top } : = \pmb { e } _ { q } ^ { \top } \pmb { \Phi } _ { t + 1 , k } , \qquad r \leq t < k .
$$

By Equation (20), $\mathbf { \nabla } \mathbf { \mathbf { } } v _ { t } \geq \mathbf { 0 }$ and $s _ { t } : = v _ { t } ^ { \top } \mathbf { 1 } \leq 1$ . If

$$
s _ { t - 1 } : = { \pmb v } _ { t } ^ { \top } ( { \pmb I } - \alpha _ { t } { \pmb G } _ { m } ) { \bf 1 } ,
$$

then

$$
s _ { t } - s _ { t - 1 } = ( 1 - \gamma ) \alpha _ { t } \pmb { v } _ { t } ^ { \top } D _ { m } \pmb { 1 } \geq 0 .\tag{56}
$$

Equations (22) and (56), together with the monotonicity of the stepsizes, imply

$$
\begin{array} { r l r } {  { \sum _ { t = r } ^ { k - 1 } \alpha _ { t } ^ { 2 } \mathbb { E } [ ( \pmb { v } _ { t } ^ { \top } \mathbf { 1 } \{ t < \sigma \} \pmb { \xi } ^ { ( t + 1 ) } ) ^ { 2 } \mid \mathcal { F } _ { t } ] } } \\ & { } & { \leq \frac { 2 v _ { m } \alpha _ { r } } { 1 - \gamma } \displaystyle \sum _ { t = r } ^ { k - 1 } s _ { t } ( s _ { t } - s _ { t - 1 } ) \leq \frac { 2 v _ { m } \alpha _ { r } } { 1 - \gamma } . } \end{array}\tag{57}
$$

The absolute value of every martingale increment is at most $\alpha _ { t } \pmb { v } _ { t } ^ { \top } \pmb { 1 } \leq \alpha _ { r }$ . Freedman’s inequality therefore gives the right-hand side of Equation (55) for a fixed triple $( q , r , k )$ . A union bound over

at most $| S | m T ^ { 2 }$ triples, and $\alpha _ { r } \leq \alpha _ { n }$ , completes the proof.

□

Lemma B.4. Let $\mu > 0 , r _ { 0 } > 0 , L _ { h } \geq 0$ , and let $\{ \alpha _ { t } \} _ { t = n } ^ { N - 1 }$ be nonincreasing with $0 \leq \mu \alpha _ { t } \leq 1$ Suppose matrices $P _ { t }$ satisfy

$$
P _ { t } \geq 0 , \qquad \| P _ { t } \| _ { \infty } \leq 1 - \mu \alpha _ { t } .
$$

Set $\Phi _ { k , k } : = I$ and, for $r < k$ , write

$$
\Phi _ { r , k } : = P _ { k - 1 } \cdot \cdot \cdot P _ { r } .
$$

Suppose, up to the first exit of $e ^ { ( t ) }$ from the $r _ { 0 }  – b a l l$

$$
\begin{array} { r } { \pmb { e } ^ { ( t + 1 ) } = \pmb { P } _ { t } \pmb { e } ^ { ( t ) } + \pmb { \zeta } ^ { ( t + 1 ) } - \alpha _ { t } \pmb { R } ( \pmb { e } ^ { ( t ) } ) , \qquad \| \pmb { R } ( \pmb { e } ) \| _ { \infty } \leq L _ { h } \| \pmb { e } \| _ { \infty } ^ { 2 } , } \end{array}\tag{58}
$$

and, for every $n \leq r < k \leq N$ ，

$$
\left. \sum _ { t = r } ^ { k - 1 } \Phi _ { t + 1 , k } \zeta ^ { ( t + 1 ) } \right. _ { \infty } \leq b .\tag{59}
$$

If

$$
\left. e ^ { ( n ) } \right. _ { \infty } \leq \frac { r _ { 0 } } { 2 } , \qquad \frac { L _ { h } r _ { 0 } } { \mu } \leq \frac { 1 } { 1 6 } , \qquad b \leq \operatorname* { m i n } \left\{ \frac { r _ { 0 } } { 1 6 } , \frac { \mu } { 6 4 L _ { h } } \right\} ,\tag{60}
$$

where the second fraction is +∞ if $L _ { h } = 0$ , then no exit occurs through time $N$ , and

$$
\left\| e ^ { ( N ) } \right\| _ { \infty } \leq C \left[ r _ { 0 } \exp \left\{ - \frac { \mu } { C } \sum _ { t = n } ^ { N - 1 } \alpha _ { t } \right\} + b + \frac { L _ { h } } { \mu } b ^ { 2 } \right]\tag{61}
$$

for a universal constant $C .$

Proof. The bootstrap is easiest to see after removing the linear response to the perturbations. Let

$$
\sigma : = \operatorname* { i n f } \left\{ k \in \{ n , \dots , N \} : \left\| e ^ { ( k ) } \right\| _ { \infty } > r _ { 0 } \right\} ,
$$

with the convention inf $\emptyset = N + 1$

Step 1: Remove the linear perturbation. For $n \leq k \leq N$ , define

$$
{ \pmb w } ^ { ( n ) } : = { \bf 0 } , \qquad { \pmb w } ^ { ( k ) } : = \sum _ { t = n } ^ { k - 1 } \pmb { \Phi } _ { t + 1 , k } \pmb { \zeta } ^ { ( t + 1 ) } .\tag{62}
$$

Taking $r = n$ in Equation (59) gives

$$
\begin{array} { r } { \left\| \boldsymbol { w } ^ { ( k ) } \right\| _ { \infty } \leq b . } \end{array}\tag{63}
$$

The chosen order of the propagator implies

$$
\begin{array} { r } { \pmb { w } ^ { ( t + 1 ) } = \pmb { P } _ { t } \pmb { w } ^ { ( t ) } + \pmb { \zeta } ^ { ( t + 1 ) } . } \end{array}
$$

Therefore, if

$$
\begin{array} { r } { z ^ { ( t ) } : = e ^ { ( t ) } - w ^ { ( t ) } , \qquad y _ { t } : = \left. z ^ { ( t ) } \right. _ { \infty } , } \end{array}
$$

then the perturbations cancel exactly. For every $t < \sigma$ , Equations (58) and (63) give

$$
\begin{array} { r } { y _ { t + 1 } \leq ( 1 - \mu \alpha _ { t } ) y _ { t } + \alpha _ { t } \left. \pmb { R } ( e ^ { ( t ) } ) \right. _ { \infty } } \\ { \leq ( 1 - \mu \alpha _ { t } ) y _ { t } + L _ { h } \alpha _ { t } ( y _ { t } + b ) ^ { 2 } . } \end{array}\tag{64}
$$

Step 2: Close the local bootstrap. We claim that $y _ { k } \le r _ { 0 } / 2$ for every $n \leq k \leq N \land \sigma$ . Suppose $y _ { t } \le r _ { 0 } / 2$ . Then

$$
\begin{array} { l } { { L _ { h } ( y _ { t } + b ) ^ { 2 } \le 2 L _ { h } y _ { t } ^ { 2 } + 2 L _ { h } b ^ { 2 } } } \\ { { \qquad \le L _ { h } r _ { 0 } y _ { t } + 2 L _ { h } b ^ { 2 } \le \displaystyle \frac { \mu } { 1 6 } y _ { t } + 2 L _ { h } b ^ { 2 } . } } \end{array}
$$

Moreover, if $L _ { h } > 0$ , the two bounds on b in Equation (60) imply

$$
2 L _ { h } b ^ { 2 } \leq \frac { \mu b } { 3 2 } \leq \frac { \mu r _ { 0 } } { 5 1 2 } ,\tag{65}
$$

while the same bound is immediate when $L _ { h } = 0$ . Because $0 \leq \mu \alpha _ { t } \leq 1$ , the coeficient $1 - 1 5 \mu \alpha _ { t } / 1 6$ is nonnegative. Thus, Equation (64) and $y _ { t } \le r _ { 0 } / 2$ yield

$$
\begin{array} { l } { { y _ { t + 1 } \leq \left( 1 - \displaystyle \frac { 1 5 } { 1 6 } \mu \alpha _ { t } \right) y _ { t } + 2 L _ { h } \alpha _ { t } b ^ { 2 } } } \\ { { \mathrm { } \leq \left( 1 - \displaystyle \frac { 1 5 } { 1 6 } \mu \alpha _ { t } \right) \displaystyle \frac { r _ { 0 } } { 2 } + \displaystyle \frac { \mu \alpha _ { t } r _ { 0 } } { 5 1 2 } } } \\ { { \mathrm { } = \displaystyle \frac { r _ { 0 } } { 2 } - \displaystyle \frac { 2 3 9 } { 5 1 2 } \mu \alpha _ { t } r _ { 0 } \leq \displaystyle \frac { r _ { 0 } } { 2 } . } } \end{array}\tag{66}
$$

Because $y _ { n } = \left\| e ^ { ( n ) } \right\| _ { \infty } \leq r _ { 0 } / 2$ , induction proves the claim.

Combining the claim with Equation (63) shows that, for every $n \leq k \leq N \land \sigma ,$

$$
\left\| e ^ { ( k ) } \right\| _ { \infty } \leq y _ { k } + \left\| w ^ { ( k ) } \right\| _ { \infty } \leq \frac { r _ { 0 } } { 2 } + b \leq \frac { 9 r _ { 0 } } { 1 6 } < r _ { 0 } .
$$

If $\sigma \leq N$ , this estimate at $k = \sigma$ contradicts the definition of $\sigma ;$ the induction step leading to $k = \sigma$

is valid because it only uses the recursion at $t = \sigma - 1 < \sigma$ . Thus, $\sigma = N + 1$ , so no exit occurs before time N.

Step 3: Iterate the scalar contraction. Now the first inequality in Equation (66) holds on the whole interval. The scalar kernel satisfies

$$
\begin{array} { r l r } {  { \sum _ { t = n } ^ { N - 1 } \alpha _ { t } \prod _ { \ell = t + 1 } ^ { N - 1 } ( 1 - \frac { 1 5 } { 1 6 } \mu \alpha _ { \ell } ) } } \\ & { } & { \quad = \frac { 1 - \prod _ { t = n } ^ { N - 1 } \big ( 1 - \frac { 1 5 } { 1 6 } \mu \alpha _ { t } \big ) } { ( 1 5 / 1 6 ) \mu } \le \frac { 1 6 } { 1 5 \mu } . } \end{array}
$$

Iterating the scalar recursion and using this identity gives

$$
\begin{array} { l } { \displaystyle { y _ { N } \leq y _ { n } \prod _ { t = n } ^ { N - 1 } \left( 1 - \frac { 1 5 } { 1 6 } \mu \alpha _ { t } \right) } } \\ { \displaystyle { \phantom { \frac { N - 1 } { 2 } } + 2 L _ { h } b ^ { 2 } \sum _ { t = n } ^ { N - 1 } \alpha _ { t } \prod _ { \ell = t + 1 } ^ { N - 1 } \left( 1 - \frac { 1 5 } { 1 6 } \mu \alpha _ { \ell } \right) } } \\ { \displaystyle { \phantom { \frac { N - 1 } { 2 } } \leq \frac { r _ { 0 } } { 2 } \exp \left\{ - \frac { 1 5 \mu } { 1 6 } \sum _ { t = n } ^ { N - 1 } \alpha _ { t } \right\} + \frac { 3 2 L _ { h } } { 1 5 \mu } b ^ { 2 } } . } \end{array}
$$

Finally, Equation (63) gives

$$
\left. e ^ { ( N ) } \right. _ { \infty } \leq \frac { r _ { 0 } } { 2 } \exp \left\{ - \frac { 1 5 \mu } { 1 6 } \sum _ { t = n } ^ { N - 1 } \alpha _ { t } \right\} + b + \frac { 3 2 L _ { h } } { 1 5 \mu } b ^ { 2 } ,
$$

which is stronger than Equation (61).