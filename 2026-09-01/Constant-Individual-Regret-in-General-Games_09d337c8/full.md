# Constant Individual Regret in General Games

Mingyang Liu<sup>1</sup>, Gabriele Farina<sup>1</sup>, Asuman Ozdaglar<sup>1</sup>

<sup>1</sup> LIDS, EECS, Massachusetts Institute of Technology

<sup>1</sup> {liumy19,gfarina,asuman}@mit.edu

## Abstract

Uncoupled no-regret dynamics provide a decentralized route to equilibrium, but prior guarantees for individual regret retain a polylogarithmic dependence on the horizon. We remove this dependence for every finite N-player normal-form game under full-information feedback. We introduce ECHO-OFTRL: optimistic followthe-regularized-leader (OFTRL) equipped with an EMA cascade for high-order optimism (ECHO), where EMA denotes exponential moving average. The algorithm is deterministic and fully uncoupled. If m<sub>max</sub> denotes the largest action-set size, then, simultaneously for every horizon T ≥ 1, it guarantees that each of the N players in the game incurs regret upper bounded by $O ( \mathrm { p o l y } ( N , \log m _ { \mathrm { m a x } } ) )$ . Our algorithm leverages a new form of optimism inspired by modern filter design.

## 1 Introduction

Uncoupled no-regret dynamics model decentralized learning: each player updates from its own full payoff vector without knowing the other players’ objectives (Hart and Mas-Colell, 2000; Greenwald and Jafari, 2003). If every player has small external regret, empirical play approaches the set of coarse correlated equilibria (CCE) (Moulin and Vial, 1978; Cesa-Bianchi and Lugosi, 2006). Regret-minimization methods also underlie several landmark systems for solving large imperfect-information games (Bowling et al., 2015; Moravˇcík et al., 2017; Brown and Sandholm, 2018, 2019).

Traditional online-learning analysis treats the payoff sequence as an arbitrary, possibly adversarial, exogenous process. Against such a sequence, T regret can be easily shown to be unavoidable (Hazan, 2016). In self-play, however, the payoff sequences are coupled through a fixed game and the learners’ updates. While nonstationary, the smooth update of strategies makes the dynamic landscape faced by each player in the game slightly predictable. Optimistic methods seek to exploit this predictability to accelerate learning and convergence to equilibrium. Optimism has also been studied more broadly as a route to acceleration in optimization and game solving (Wang and Abernethy, 2018; Mertikopoulos et al., 2019; Farina et al., 2019; Piliouras et al., 2022a). Fast rates under robustness and nonparametric structure have also been studied (Foster et al., 2016; Daskalakis and Golowich, 2022).

The quest for accelerated learning dynamics has driven a long progression of faster rates.

Predictable-sequence methods brought optimism to online learning (Chiang et al., 2012; Rakhlin and Sridharan, 2013). The RVU framework (Syrgkanis et al., 2015) then gave $T ^ { 1 / 4 }$ individual regret and, more crucially, exposed a key mechanism of analysis that has since been a key mainstay. Unfortunately, that analysis alone is capable of guaranteeing constant sum of regrets, but not constant individual regret, as negative regrets can hide another player’s large positive regret.

Later, Chen and Peng (2020) leveraged the RVU bound, but was able to break through the $T ^ { 1 / 4 }$ barrier in the special case of two-player games. Breaking past the polynomial dependence in $T ,$ the high-order, discrete-time frequency analysis of Optimistic Hedge of Daskalakis, Fishelson, and Golowich (2021) reduced individual regret to $O ( \log ^ { 4 } T )$ . This approach was subsequently extended to certain structured polyhedral games (Farina et al., 2022b). Lifted OFTRL then obtained O(log T) regret over general convex games by turning ordinary regret into a nonnegative lifted regret (Farina et al., 2022a). Dynamic learning-rate control matched this logarithmic horizon dependence while improving the action dependence to polylogarithmic (Soleymani et al., 2025b). This mechanism proved extremely general and robust across different choices of the convex action set and regularizer (Soleymani et al., 2025a). Related work also obtains logarithmic swap regret in multiplayer games (Anagnostides et al., 2022). Finally, Clairvoyant MWU (Piliouras et al., 2022b) obtained a constant regret bound from a fixed-point update that evaluates payoff vectors at the strategy profile being computed. This guarantee is not directly comparable to those above: the method requires unrolling joint fixed-point computations, and its regret bound applies only to a sparse subsequence rather than the full history of play.

We summarize the comparable results in Table 1, where N denotes the number of players and $m : = m _ { \mathrm { m a x } }$ denotes the largest number of actions of any player. All bounds are specialized to utilities in [0, 1]; the OFTRL/OOMD row uses entropy regularization. This history points to a natural question:

Can uncoupled regularized learning keep every player’s positive regret uniformly bounded in time?

We show that it can for full-information self-play in finite normal-form games, resolving the horizon-dependence question in this setting. We also remark that because our dynamics are based on a (lifted) version of Hedge, it is possible to kernelize them, making our method applicable beyond normal-form games to a number of important combinatorial settings, including extensiveform games (Farina et al., 2022b).

## 1.1 Contributions

The main contribution of the paper is to show the following.

Theorem 1.1 (informal main theorem). For every N-player finite normal-form game with utilities in [0, 1] and full payoff-vector feedback, there exist deterministic, uncoupled learning dynamics such that,

Table 1: Progress toward horizon-independent individual regret via uncoupled learning dynamics.
<table><tr><td>Method</td><td>Individual regret</td><td>Main technique(s)</td></tr><tr><td>OFTRL/OOMD (Syrgkanis et al., 2015)</td><td> $O \big ( \sqrt { N } \log m T ^ { 1 / 4 } \big )$ </td><td>One-step optimism and RVU bounds</td></tr><tr><td>Optimistic Hedge (Chen and Peng, 2020)</td><td> $O \big ( \log ^ { 5 / 6 } m T ^ { 1 / 6 } \big )$   $( \mathrm { T w o - p l a y e r s ~ o n l y } )$ </td><td>Coupled strategy/payoff path-length bounds</td></tr><tr><td>Optimistic Hedge (Daskalakis et al., 2021)</td><td> $O \big ( N \log m \log ^ { 4 } T \big )$ </td><td>Analysis leverages high-order discrete differences</td></tr><tr><td>LRL-OFTRL (Farina et al., 2022a)</td><td> $O ( N m \log T )$ </td><td>Lifting, nonnegative RVU bounds, and log regularization</td></tr><tr><td>DLRC-OMWU / Cautious Optimism (Soleymani et al., 2025b,a)</td><td> $O \big ( N \log ^ { 2 } m \log T \big )$ </td><td>Dynamic pacing, intrinsic Lipschitzness</td></tr><tr><td>ECHO-OFTRL (This paper)</td><td> $O \left( N ^ { 2 1 } \log ^ { 4 } m \right)$ </td><td>Transfer-function-designed EMA residual and nonnegative RVU bound</td></tr></table>

when used by all players, the individual regret of the generic player i satisfies

$$
\mathrm { R e g } _ { i } ( T ) \leq \sum _ { j = 1 } ^ { N } [ \mathrm { R e g } _ { j } ( T ) ] _ { + } \leq C ( N + 1 ) ^ { 2 1 } \left( 1 + \log ( m _ { \operatorname* { m a x } } + 1 ) \right) ^ { 4 }
$$

simultaneously for every horizon $T \geq 1$ , where C is a universal constant.

The player dependence is explicitly polynomial and the bound is independent of $T ;$ we made no attempt to optimize either displayed exponent.

Technique-wise, our main methodological contribution is a signal-processing construction of a new, stable high-order predictor. Standard optimistic learning uses the preceding payoff vector to predict the current one, so the prediction error is the one-step change in payoffs. Repeating this idea N times would give the raw difference $( I - \mathsf { L } ) ^ { N }$ , where L is the one-round delay: for a payoff sequence g and every integer $t , ( \mathsf { L } \pmb { g } ) ^ { ( t ) } = \pmb { g } ^ { ( t - 1 ) }$ . The coefficients of the raw difference have total absolute value $2 ^ { \dot { N } } .$ , so this direct construction can amplify oscillations exponentially. We instead set

$$
\begin{array} { l } { \displaystyle { \rho : = 1 - \frac { 1 } { N } , \qquad \delta : = \frac { I - \mathsf { L } } { I - \rho \mathsf { L } } , } } \\ { \displaystyle { \widehat { \pmb { g } } : = ( I - \delta ^ { N } ) \pmb { g } , \qquad \pmb { g } - \widehat { \pmb { g } } = \delta ^ { N } \pmb { g } . } } \end{array}
$$

The filter $I - \delta = ( 1 - \rho ) \mathsf { L } ( I - \rho \mathsf { L } ) ^ { - 1 }$ is a one-pole EMA of past values. This preconditioning reduces the worst-case frequency gain of the N-fold difference from $2 ^ { N }$ to $( 2 / ( 1 + \rho ) ) ^ { N } \leq e ,$ while the time-domain filter bounds used in the proof remain polynomial in N.

In short, high-order optimism lets us bound the cumulative prediction-error term by a horizonindependent constant plus strategy movement; the negative movement term absorbs the latter, yielding constant regret.

## 2 Preliminaries

For every positive integer n, let $[ n ] : = \{ 1 , \dots , n \}$ . For a vector $z \in \mathbb { R } ^ { n }$ and $a \in [ n ]$ , write $z ( a )$ for its $a ^ { \mathrm { t h } }$ coordinate. For $1 \ \leq \ p \ < \ \infty ,$ define $\Vert z \Vert _ { p } : = \ \left( \sum _ { a = 1 } ^ { n } | z ( a ) | ^ { p } \right) ^ { 1 / p }$ , and define $\| z \| _ { \infty } : = \operatorname* { m a x } _ { a \in [ n ] } | z ( a ) |$ . For a scalar $r \in \mathbb { R } ,$ , define $[ r ] _ { + } : =$ max $\{ r , 0 \}$ . The $( n - 1 )$ -dimensional probability simplex is

$$
\Delta ^ { n } : = \left\{ x \in [ 0 , 1 ] ^ { n } : \sum _ { a = 1 } ^ { n } x ( a ) = 1 \right\} .
$$

We use $\langle \cdot , \cdot \rangle$ for the standard inner product and 1 for the all-ones vector of the appropriate dimension. For a tuple $\pmb { x } : = ( \pmb { x } _ { i } ) _ { i } ,$ write $\pmb { x } _ { - i } : = ( \pmb { x } _ { j } ) _ { j \neq i }$ . Superscripts in parentheses index time, so $z ^ { ( t ) }$ denotes the value of a sequence z at round t. For a differentiable map $\Phi : E  F$ between finite-dimensional Euclidean spaces, $\pmb { x } \in E ,$ and $\pmb { \xi } \in E , J _ { \Phi } ( \pmb { x } ) : E  F$ denotes its Jacobian at $^ { \mathbf { \delta x } , }$ and $\begin{array} { r } { J _ { \Phi } ( \pmb { x } ) [ \pmb { \xi } ] : = \left. \frac { \mathrm { d } } { \mathrm { d } r } \Phi ( \pmb { x } + r \pmb { \xi } ) \right| _ { r = 0 } } \end{array}$ is its directional derivative along ξ. If $\begin{array} { r } { E = \Pi _ { j = 1 } ^ { k } E _ { j } , } \end{array}$ , then, for $j \in [ k ] .$ , write $\partial _ { j } \Phi$ for the derivative with respect to the $j ^ { \mathrm { t h } }$ block. For scalar-valued $f , \nabla f$ denotes its gradient, so $J _ { f } ( { \pmb x } ) [ { \pmb \xi } ] = \langle \nabla f ( { \pmb x } ) , { \pmb \xi } \rangle$ , and $\nabla ^ { 2 } f$ denotes its Hessian.

## 2.1 Finite normal-form games and regret

There are $N \geq 2$ players. Player i has action set $\left[ m _ { i } \right]$ , where $m _ { i } \geq 2 .$ , and chooses a mixed strategy $\pmb { x } _ { i } \in \Delta ^ { m _ { i } }$ . Define the largest action-set size by

$$
\begin{array} { r } { m _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { i \in [ N ] } m _ { i } . } \end{array}
$$

Player $i ^ { \prime } \mathrm { s }$ utility function $\begin{array} { r } { \mathcal { U } _ { i } \colon \prod _ { i = 1 } ^ { N } \Delta ^ { m _ { j } } \to [ 0 , 1 ] } \end{array}$ is the multilinear extension of a pure-profile payoff function taking values in [0, 1]. Define the expected payoff vector by

$$
\mathcal { U } _ { i } ( \pmb { x } _ { - i } ) : = \nabla _ { \pmb { x } _ { i } } \mathcal { U } _ { i } ( \pmb { x } ) \in [ 0 , 1 ] ^ { m _ { i } } .
$$

It is multiaffine in the opponents’ strategies and does not depend on $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$

For a horizon $T \geq 1$ , at every round $t \in [ T ]$ , the players choose $\begin{array} { r } { \pmb { x } ^ { ( t ) } \in \prod _ { j = 1 } ^ { N } \Delta ^ { m _ { j } } } \end{array}$ simultaneously and player i observes the full vector ${ \pmb u } _ { i } ^ { ( t ) } : = \mathcal { U } _ { i } ( { \pmb x } _ { - i } ^ { ( t ) } )$ . Its external regret is

$$
\begin{array} { r l } { \mathrm { R e g } _ { i } ( T ) : = \underset { { \pmb x } _ { i } \in \Delta ^ { m _ { i } } } { \operatorname* { m a x } } \underset { t = 1 } { \overset { T } { \sum } } \left[ \mathcal { U } _ { i } ( { \pmb x } _ { i } , { \pmb x } _ { - i } ^ { ( t ) } ) - \mathcal { U } _ { i } ( { \pmb x } ^ { ( t ) } ) \right] } & { } \\ { = \underset { { \pmb x } _ { i } \in \Delta ^ { m _ { i } } } { \operatorname* { m a x } } \underset { t = 1 } { \overset { T } { \sum } } \left. { \pmb u } _ { i } ^ { ( t ) } , { \pmb x } _ { i } - { \pmb x } _ { i } ^ { ( t ) } \right. . } \end{array}
$$

The equality follows from multilinearity in player $i ^ { \prime } \mathrm { s }$ strategy.

## 3 Lifted optimistic dynamics

## 3.1 Lifted regret

Following Farina et al. (2022a), for every player $i \in [ N ]$ , define

$$
\widetilde { \Delta } _ { i } : = \left\{ \left( \lambda , \pmb { y } \right) : 0 \leq \lambda \leq 1 , \pmb { y } \in \lambda \Delta ^ { m _ { i } } \right\} .
$$

For every $i \in [ N ]$ and $( \lambda , y ) \in \widetilde { \Delta } _ { i }$ with $\lambda > 0$ , the point $\left( \lambda , y \right)$ plays $\pmb { x } : = \pmb { y } / \lambda$ . For every player $i \in [ N ]$ , define $\mathcal { G } _ { i }$ by

$$
\begin{array} { r } { \mathcal { G } _ { i } ( \pmb { x } ) : = \mathcal { U } _ { i } ( \pmb { x } _ { - i } ) - \langle \mathcal { U } _ { i } ( \pmb { x } _ { - i } ) , \pmb { x } _ { i } \rangle \mathbf { 1 } . } \end{array}
$$

For every player $i \in [ N ]$ , define $\pmb { g } _ { i } = \left( \pmb { g } _ { i } ^ { ( s ) } \right) _ { s \in \mathbb { Z } }$ by

$$
\begin{array} { r } { \pmb { g } _ { i } ^ { ( s ) } : = \left\{ \begin{array} { l l } { \mathcal { G } _ { i } ( \pmb { x } ^ { ( s ) } ) , } & { s \geq 1 , } \\ { \mathbf { 0 } , } & { s \leq 0 , } \end{array} \right. \quad s \in \mathbb { Z } . } \end{array}
$$

For every $i \in [ N ]$ and integer $s \geq 1$

$$
\begin{array} { c c } { g _ { i } ^ { ( s ) } = \pmb { u } _ { i } ^ { ( s ) } - \left. \pmb { u } _ { i } ^ { ( s ) } , \pmb { x } _ { i } ^ { ( s ) } \right. \mathbf { 1 } , \qquad \left\| \pmb { g } _ { i } ^ { ( s ) } \right\| _ { \infty } \leq 1 , } \\ { \left. \pmb { g } _ { i } ^ { ( s ) } , \pmb { x } _ { i } ^ { ( s ) } \right. = 0 . } \end{array}
$$

For every $i \in [ N ]$ and round $t \geq 1$ , write $\pmb { \mu } _ { i } ^ { ( t ) } = \left( \lambda _ { i } ^ { ( t ) } , \lambda _ { i } ^ { ( t ) } \pmb { x } _ { i } ^ { ( t ) } \right) \in \widetilde { \Delta } _ { i }$ for player i’s lifted strategy at round t. For every $i \in [ N ]$ and $T \geq 1$ , write each comparator in $\widetilde { \Delta } _ { i }$ as $\left( \widehat { \lambda } _ { i } , \widehat { \lambda } _ { i } \widehat { \pmb { x } } _ { i } \right)$ , where $\widehat { \lambda } _ { i } \in [ 0 , 1 ]$ and $\widehat { \pmb x } _ { i } \in \Delta ^ { m _ { i } }$ . Since $\left. \pmb { g } _ { i } ^ { ( t ) } , \lambda _ { i } ^ { ( t ) } \pmb { x } _ { i } ^ { ( t ) } \right. = \lambda _ { i } ^ { ( t ) } \left. \pmb { g } _ { i } ^ { ( t ) } , \pmb { x } _ { i } ^ { ( t ) } \right. = 0$ for every $t \geq 1 ,$

$$
\operatorname* { m a x } _ { ( \widetilde { \lambda } _ { i } , \widehat { \lambda } _ { i } \widehat { \mathbf { x } } _ { i } ) \in \widetilde { \Delta } _ { i } } \sum _ { t = 1 } ^ { T } \left. g _ { i } ^ { ( t ) } , \widehat { \lambda } _ { i } \widehat { \mathbf { x } } _ { i } - \lambda _ { i } ^ { ( t ) } \pmb { x } _ { i } ^ { ( t ) } \right. = \operatorname* { m a x } _ { 0 \leq \widehat { \lambda } _ { i } \leq 1 } \widehat { \lambda } _ { i } \mathrm { R e } \mathbf { g } _ { i } ( T ) = [ \mathrm { R e } \mathbf { g } _ { i } ( T ) ] _ { + } .\tag{3.1}
$$

## 3.2 The square-root entropy response

For every integer $m > 0$ and $\pmb { x } \in \Delta ^ { m } , \Gamma _ { m }$ bounds $\operatorname { V a r } _ { a \sim x } [ - \log x ( a ) ]$ . Formally,

$$
\Gamma _ { m } : = ( \log m ) ^ { 2 } + 2 \log m + 2 , \qquad \Xi _ { m } : = 1 0 ^ { 4 } ( 1 + \Gamma _ { m } ) .
$$

For every integer $m \geq 2$ and $\pmb { x } \in \Delta ^ { m }$ , define $\begin{array} { r } { \psi _ { m } ( \pmb { x } ) : = \sum _ { a = 1 } ^ { m } x ( a ) \log x ( a ) } \end{array}$ , where 0 log $0 : = 0$ . For every integer $m \geq 2 ,$ , define

$$
\widetilde { \Delta } ^ { m } : = \left\{ ( \lambda , \pmb { y } ) : 0 \leq \lambda \leq 1 , \ \pmb { y } \in \lambda \Delta ^ { m } \right\} .
$$

For every integer $m \geq 2 ,$ , define $\widetilde { \psi } _ { m }$ on $\widetilde { \Delta } ^ { m }$ by

$$
\widetilde { \psi } _ { m } ( \lambda , y ) : = - \sqrt { 1 - \lambda } + \sqrt { \lambda } \left[ \psi _ { m } \left( y / \lambda \right) - 2 \Gamma _ { m } - 3 \right] , \qquad \lambda > 0 .
$$

Set $\widetilde { \psi } _ { m } ( 0 , \mathbf { 0 } ) : = - 1$ for every integer $m \geq 2$ . For every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , define

$$
\begin{array} { r } { Q _ { m } ( \pmb { \theta } ) : = \underset { ( \lambda , \pmb { y } ) \in \widetilde { \Delta } ^ { m } } { \mathrm { a r g m a x } } \left\{ \langle \pmb { \theta } , \pmb { y } \rangle - \widetilde { \psi } _ { m } ( \lambda , \pmb { y } ) \right\} . } \end{array}
$$

The next lemma establishes some basic properties of $Q _ { m } ( \pmb \theta )$

Lemma 3.1. For every integer $m \geq 2 ,$ the function $\widetilde { \psi } _ { m }$ is finite and convex on $\widetilde { \Delta } ^ { m }$ and strictly convex in its relative interior. For every integer m $\geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the maximizer defining $Q _ { m } ( \pmb \theta )$ is unique and belongs to relint $\widetilde { \Delta } ^ { m }$ . More explicitly, for every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the function on $( 0 , 1 )$ given by

$$
\lambda ^ { \prime } \longmapsto { \frac { 1 } { \sqrt { 1 - \lambda ^ { \prime } } } } - { \frac { 2 \Gamma _ { m } + 3 + \log \left( \sum _ { a = 1 } ^ { m } \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) \right) } { \sqrt { \lambda ^ { \prime } } } } - { \frac { \sum _ { a = 1 } ^ { m } \theta ( a ) \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) } { \sum _ { a = 1 } ^ { m } \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) } }
$$

is strictly increasing, tends to $- \infty$ at the left endpoint, and tends to $+ \infty$ at the right endpoint. Let $\lambda \in ( 0 , 1 )$ denote its unique zero. Then

$$
Q _ { m } ( \pmb \theta ) = \left( \lambda , \frac { \lambda \left( \exp \left( \sqrt \lambda \theta ( a ) \right) \right) _ { a \in [ m ] } } { \sum _ { b = 1 } ^ { m } \exp \left( \sqrt \lambda \theta ( b ) \right) } \right) .
$$

Moreover, for every integer $m \geq 2 ,$

$$
\operatorname* { s u p } _ { \pmb { \mu } \in \widetilde { \Delta } ^ { m } } \widetilde { \psi } _ { m } ( \pmb { \mu } ) - \operatorname* { i n f } _ { \pmb { \mu } \in \widetilde { \Delta } ^ { m } } \widetilde { \psi } _ { m } ( \pmb { \mu } ) \leq 2 \Gamma _ { m } + \log m + 4 .
$$

The proof is deferred to Section B.

Remark 3.2. Fix an integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ . By Lemma 3.1, binary search on the sign of the strictly increasing function in the lemma localizes its unique zero λ and the formula in the lemma determines $Q _ { m } ( \pmb \theta )$

For every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , write

$$
\big ( \lambda _ { m } ( \pmb \theta ) , \pmb y _ { m } ( \pmb \theta ) \big ) : = Q _ { m } ( \pmb \theta ) , \qquad \pmb x _ { m } ( \pmb \theta ) : = \pmb y _ { m } ( \pmb \theta ) / \lambda _ { m } ( \pmb \theta ) .
$$

For every integer $m \geq 2$ and $\mu , \mu ^ { \prime } \in$ relint $\widetilde { \Delta } ^ { m }$ , define

$$
\begin{array} { r l } & { D _ { \widetilde { \psi } _ { m } } ( \mu ^ { \prime } , \pmb { \mu } ) : = \widetilde { \psi } _ { m } ( \pmb { \mu } ^ { \prime } ) - \widetilde { \psi } _ { m } ( \pmb { \mu } ) - \left. \nabla \widetilde { \psi } _ { m } ( \pmb { \mu } ) , \pmb { \mu } ^ { \prime } - \pmb { \mu } \right. , } \\ & { D _ { \widetilde { \psi } _ { m } } ^ { \mathrm { s y m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) : = D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) + D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } , \pmb { \mu } ^ { \prime } ) . } \end{array}
$$

For every $i \in [ N ]$ , abbreviate

$$
\begin{array} { r l r l r l } & { \psi _ { i } : = \psi _ { m _ { i } } , } & & { \widetilde { \psi } _ { i } : = \widetilde { \psi } _ { m _ { i } } , } & & { Q _ { i } : = Q _ { m _ { i } } , } \\ & { \Gamma _ { i } : = \Gamma _ { m _ { i } } , } & & { \Xi _ { i } : = \Xi _ { m _ { i } } . } \end{array}
$$

Set $\Xi _ { \mathrm { m a x } } : = \mathrm { m a x } _ { i \in [ N ] } \Xi _ { i }$ . For every $i \in [ N ]$ and $\pmb \theta \in \mathbb { R } ^ { m _ { i } }$ , write $Q _ { i } ( \pmb \theta ) = \big ( \lambda _ { m _ { i } } ( \pmb \theta ) , \lambda _ { m _ { i } } ( \pmb \theta ) \pmb x _ { m _ { i } } ( \pmb \theta ) \big )$ .

## 3.3 ECHO-OFTRL

We call the algorithm below ECHO-OFTRL, where ECHO stands for EMA cascade for high-order optimism and EMA denotes exponential moving average.

Fix a common learning rate $\eta > 0 .$ . On every space of bounded bi-infinite sequences in a finite-dimensional normed space, let I be the identity operator and let L be the backward shift, defined for every sequence z in that space and integer t by $( \mathsf { L } z ) ^ { ( t ) } : = z ^ { ( t - 1 ) }$ . Set

$$
\begin{array} { l } { \rho : = 1 - \displaystyle \frac 1 N , \qquad \mathsf { D } : = I - { \mathsf { L } } , \qquad \mathsf { A } : = ( I - \rho { \mathsf { L } } ) ^ { - 1 } = \displaystyle \sum _ { k = 0 } ^ { \infty } \rho ^ { k } { \mathsf { L } } ^ { k } , } \\ { \delta : = \mathsf { A D } = \displaystyle \frac { I - { \mathsf { L } } } { I - \rho { \mathsf { L } } } . } \end{array}\tag{3.2}
$$

Since A and D commute, $\delta = \mathsf { D } \mathsf { A }$ . Thus, $\delta$ takes the one-step difference of the exponentially weighted sequence generated by A.

For every sequence z that vanishes at nonpositive times and every integer $t \geq 1$

$$
\left( ( I - \delta ) z \right) ^ { ( t ) } = ( 1 - \rho ) \sum _ { k = 1 } ^ { \infty } \rho ^ { k - 1 } z ^ { ( t - k ) } .
$$

For every $i \in [ N ]$ , define

$$
\begin{array} { r l } & { \widehat { \pmb { g } } _ { i } : = ( I - \delta ^ { N } ) \pmb { g } _ { i } , } \\ & { \pmb { e } _ { i } : = \pmb { g } _ { i } - \widehat { \pmb { g } } _ { i } = \delta ^ { N } \pmb { g } _ { i } . } \end{array}\tag{3.3}
$$

Thus, for every $i \in [ N ]$ , the prediction error $e _ { i }$ is the stable $N ^ { \mathrm { t h } }$ -order filtered difference $\delta ^ { N } \pmb { g } _ { i }$ Since $\begin{array} { r } { I - \delta ^ { N } = \left( I - \delta \right) \sum _ { h = 1 } ^ { N } \delta ^ { h - 1 } } \end{array}$ , the preceding formula for $I - \delta$ shows that $\widehat { \pmb { g } } _ { i } ^ { ( t ) }$ depends only on feedback from rounds before t. The following EMA cascade computes this predictor recursively.

Every player $i \in [ N ]$ maintains the sequences $( \mathbf { e m a } _ { i , h } ^ { ( t ) } ) _ { t \geq 1 }$ in $\mathbb { R } ^ { m _ { i } }$ for $h \in [ N ]$ , initialized by ema $\mathbf { \Delta } _ { i , h } ^ { ( 1 ) } : = \mathbf { 0 }$ for every $h \in [ N ]$ . For every $i \in [ N ]$ and integer $t \geq 1$ , define

$$
S _ { i } ^ { ( t - 1 ) } : = \sum _ { s = 1 } ^ { t - 1 } \pmb { g } _ { i } ^ { ( s ) } .
$$

For every $i \in [ N ]$ and round $t \geq 1$ , player i forms $\widehat { \pmb { g } } _ { i } ^ { ( t ) }$ and plays

(3.4)

$$
\pmb { \mu } _ { i } ^ { ( t ) } : = \boldsymbol { Q } _ { i } \left( \eta \left( \pmb { S } _ { i } ^ { ( t - 1 ) } + \widehat { \pmb { g } } _ { i } ^ { ( t ) } \right) \right) , \qquad \pmb { \mu } _ { i } ^ { ( t ) } = \left( \lambda _ { i } ^ { ( t ) } , \lambda _ { i } ^ { ( t ) } \pmb { x } _ { i } ^ { ( t ) } \right) .\tag{3.5}
$$

For every $i \in [ N ]$ and $t \geq 1$ , after observing ${ \pmb u } _ { i } ^ { ( t ) } = \mathcal { U } _ { i } ( { \pmb x } _ { - i } ^ { ( t ) } )$ and computing ${ \bf g } _ { i } ^ { ( t ) }$ , player i updates all N vectors simultaneously by

$$
\mathbf { e m a } _ { i , h } ^ { ( t + 1 ) } : = \rho \mathbf { e m } \mathbf { a } _ { i , h } ^ { ( t ) } + ( 1 - \rho ) \left( g _ { i } ^ { ( t ) } - \sum _ { \ell = 1 } ^ { h - 1 } \mathbf { e m } \mathbf { a } _ { i , \ell } ^ { ( t ) } \right) , \qquad h \in [ N ] .\tag{3.6}
$$

(3.2) gives $( I - \rho \mathsf { L } ) ( I - \delta ) = ( 1 - \rho ) \mathsf { L }$ . Hence, for every bounded bi-infinite sequence z in a finitedimensional normed space and every integer $t \geq 0 , ( ( I - \delta ) z ) ^ { ( t + 1 ) } = \rho \left( ( I - \delta ) z \right) ^ { ( t ) } + ( 1 - \rho ) z ^ { ( t ) }$ Fix $i , h \in [ N ]$ and an integer $t \geq 1$ . If ema $\mathbf { \Phi } _ { i , \ell } ^ { ( t ) } = \left( ( I - \delta ) \delta ^ { \ell - 1 } \mathbf { g } _ { i } \right) ^ { ( t ) }$ for every $\ell < h ,$ then

$$
g _ { i } ^ { \left( t \right) } - \sum _ { \ell = 1 } ^ { h - 1 } \mathbf { e m } \mathbf { a } _ { i , \ell } ^ { \left( t \right) } = \left( \left( I - \left( I - \delta \right) \sum _ { \ell = 1 } ^ { h - 1 } \delta ^ { \ell - 1 } \right) g _ { i } \right) ^ { \left( t \right) } = \left( \delta ^ { h - 1 } g _ { i } \right) ^ { \left( t \right) } ,
$$

where the last equality is the geometric-series identity. The zero initial states and (3.6) therefore imply, by induction over $h \in \left[ N \right]$ , that ema $\mathbf { \Phi } _ { i , h } ^ { ( t ) } \ : = \ : \left( ( I - \delta ) \delta ^ { h - 1 } \mathbf { g } _ { i } \right) ^ { ( t ) }$ for every integer $t \geq 1$ Summing over $h \in [ N ]$ and using $\begin{array} { r } { \left( I - \delta \right) \sum _ { h = 1 } ^ { N } \delta ^ { h - 1 } = I - \delta ^ { N } } \end{array}$ verifies (3.4).

For every $i \in [ N ]$ , extend $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and $\lambda _ { i }$ by $\pmb { x } _ { i } ^ { ( t ) } : = \pmb { x } _ { m _ { i } } ( \mathbf { 0 } )$ and $\lambda _ { i } ^ { ( t ) } : = \lambda _ { m _ { i } } ( \mathbf { 0 } )$ for $t \leq 0 ,$ , and write $\pmb { x } _ { i } : = ( \pmb { x } _ { i } ^ { ( t ) } ) _ { t \in \mathbb { Z } }$ and $\lambda _ { i } : = ( \lambda _ { i } ^ { ( t ) } ) _ { t \in \mathbb { Z } }$ . Thus, for every $i \in [ N ]$ and $t \geq 1 ,$ , player i forms $\widehat { \pmb { g } } _ { i } ^ { ( t ) }$ from past feedback, computes and plays $\mathbf { \boldsymbol { x } } _ { i } ^ { ( t ) }$ , observes $\mathbf { \boldsymbol { \mathsf { u } } } _ { i } ^ { ( t ) }$ , computes ${ \bf g } _ { i } ^ { ( t ) }$ , evaluates $e _ { i } ^ { ( t ) }$ , updates the cascade by (3.6), and sets $\begin{array} { r } { \bar { \mathbf { \mathbf { S } } } _ { i } ^ { ( t ) } : = \dot { \mathbf { S } } _ { i } ^ { ( t - 1 ) } + \mathbf { g } _ { i } ^ { ( t ) } } \end{array}$ . For every $i \in [ N ]$ , the cascade uses $O ( N m _ { i } )$ memory and arithmetic per round. For every $i \in [ N ]$ and $t \geq 1$ , Lemma 3.1 gives $0 < \lambda _ { i } ^ { ( t ) } < 1 .$ so $\mathbf { \boldsymbol { x } } _ { i } ^ { ( t ) }$ is well defined.

For every $i \in [ N ]$ and $t \geq 1$ , fixing $\lambda _ { i } ^ { ( t ) }$ in the lifted optimistic-FTRL objective gives, for every $a \in [ m _ { i } ]$

$$
x _ { i } ^ { ( t ) } ( a ) = \frac { \exp \left( \eta \sqrt { \lambda _ { i } ^ { ( t ) } } [ S _ { i } ^ { ( t - 1 ) } + \widehat { \pmb { g } } _ { i } ^ { ( t ) } ] ( a ) \right) } { \sum _ { b = 1 } ^ { m _ { i } } \exp \left( \eta \sqrt { \lambda _ { i } ^ { ( t ) } } [ S _ { i } ^ { ( t - 1 ) } + \widehat { \pmb { g } } _ { i } ^ { ( t ) } ] ( b ) \right) } .
$$

Thus, for every $i \in [ N ]$ and $t \geq 1 ,  { \mathbf { x } } _ { i } ^ { ( t ) }$ is the Hedge strategy with learning rate $\eta \sqrt { \lambda _ { i } ^ { ( t ) } }$

For every $i \in [ N ]$ and $t \geq 1 _ { \ast }$ , Algorithm 1 computes $e _ { i } ^ { ( t ) }$ in (3.3) with one loop over $h \in [ N ]$

Algorithm 1 ECHO-OFTRL for player i   
Require: Common learning rate $\eta > 0$ and $\rho = 1 - 1 / N$   
1: Initialize ${ S _ { i } ^ { ( 0 ) }  \bf 0 }$ and ema $\mathbf { \Delta } _ { i , h } ^ { ( 1 ) } \gets \mathbf { 0 }$ for every $h \in [ N ]$   
2: for $t = 1 , 2 , \ldots$ do   
3: Set $\widehat { \pmb { g } } _ { i } ^ { ( t ) }  \Sigma _ { h = 1 } ^ { N }$ ema<sub>i,h</sub> (t)   
4: Compute $\mu _ { i } ^ { ( t ) }$ and play $\mathbf { \boldsymbol { x } } _ { i } ^ { ( t ) }$ using (3.5)   
5: Observe ${ \bf \nabla } { \bf { u } } _ { i } ^ { ( t ) } .$ , compute $\mathbf { \mathcal { g } } _ { i } ^ { ( \dot { t } ) } \gets \mathbf { \mathcal { u } } _ { i } ^ { ( \check { t } ) } - \left. { \mathbf { \mathcal { u } } _ { i } ^ { ( t ) } } , { \mathbf { \mathcal { x } } _ { i } ^ { ( t ) } } \right.$ 1, and set $e _ { i , 0 } ^ { ( t ) } \gets g _ { i } ^ { ( t ) }$   
6: for $h = 1 , \ldots , N$ do   
7: Set ema $\mathbf { \Lambda } _ { i , h } ^ { ( t + 1 ) } \gets \rho \mathbf { e } \mathbf { m } \mathbf { a } _ { i , h } ^ { ( t ) } + ( 1 - \rho ) \mathbf { e } _ { i , h - 1 } ^ { ( t ) }$   
8: Set $\boldsymbol { e } _ { i , h } ^ { ( t ) } \gets \boldsymbol { e } _ { i , h - 1 } ^ { ( t ) } - \mathbf { e m a } _ { i , h } ^ { ( t ) }$   
9: Set $e _ { i } ^ { ( t ) } \gets e _ { i , N } ^ { ( t ) }$ and ${ \pmb S } _ { i } ^ { ( t ) }  { \bf \overset { \ } { { \cal S } } } _ { i } ^ { ( t - 1 ) } + { \pmb g } _ { i } ^ { ( t ) }$

## 3.4 Main result

Theorem 3.3. In every N-player finite normal-form game with utilities in [0, 1], suppose every player $i \in [ N ]$ runs ECHO-OFTRL as specified in Algorithm 1 with $Q _ { i }$ and the constant learning rate

$$
\eta : = \frac { 2 ^ { - 9 4 } } { N ^ { 2 0 } \Xi _ { \operatorname * { m a x } } } .
$$

Then, simultaneously for every horizon $T \geq 1$

$$
\sum _ { i = 1 } ^ { N } [ \mathrm { R e g } _ { i } ( T ) ] _ { + } \leq 2 ^ { 1 1 2 } N ^ { 2 1 } \left( 1 + \log \left( m _ { \operatorname* { m a x } } + 1 \right) \right) ^ { 4 } .
$$

A proof sketch is in Section 4 and the formal proof is in Section A.

## 4 Proof sketch

The formal proof of Theorem 3.3 appears in Section $\operatorname { A } ;$ the subsequent appendices prove its geometric and filter estimates. This section outlines the main steps of the argument; $O \left( \cdot \right)$ suppresses only universal numerical factors.

## 4.1 The optimistic-FTRL bound

Fix a horizon $T \geq 1$ . For every player $i \in [ N ]$ , set $\mu _ { i } ^ { ( 0 ) } : = Q _ { i } ( \mathbf { 0 } )$ and define

$$
\mathcal { P } _ { i , T } : = \sum _ { t = 1 } ^ { T } D _ { \widetilde { \psi } _ { i } } ( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } ) , \qquad \mathcal { P } _ { T } ^ { \Xi } : = \sum _ { i = 1 } ^ { N } \Xi _ { i } \mathcal { P } _ { i , T } .
$$

The optimistic-FTRL calculation in (A.19), together with (3.3), gives

$$
\sum _ { i = 1 } ^ { N } [ \mathrm { R e g } _ { i } ( T ) ] _ { + } \leq \frac { 4 } { \eta } \sum _ { i = 1 } ^ { N } \Gamma _ { i } + 8 0 \eta \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \Xi _ { i } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left\| \left( \delta ^ { N } g _ { i } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } - \frac { 1 } { 2 \eta } \sum _ { i = 1 } ^ { N } \mathcal { P } _ { i , T } .
$$

It therefore remains to prove the variation estimate

$$
\sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left. \left( \delta ^ { N } g _ { i } \right) ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq O \left( N ^ { 1 6 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) \right) .
$$

The weights $( \eta \Xi _ { i } ) _ { i \in [ N ] }$ are inserted only after this estimate, when the resulting sum is absorbed by $- \sum _ { i } \mathcal { P } _ { i , T } / ( 2 \eta )$ .

## 4.2 A recursive bound

For a positive weight sequence $w = \Big ( w ^ { ( t ) } \Big ) _ { t \leq T } , \mathsf { a }$ player $i \in [ N ]$ , and an integer $1 \leq h \leq N ,$ define

$$
\mathcal { E } _ { i } ^ { h } ( w ) : = \sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| \left( \delta ^ { h } \pmb { x } _ { i } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 2 } .
$$

By Lemma $\mathrm { A . } 9 ,$ for every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ and positive weight sequence $w =$ $\left( w ^ { ( t ) } \right) _ { t < T }$ satisfying $w ^ { ( t ) } \leq 1$ for every $t \in [ T ]$

$$
\sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| \left( \delta ^ { h } g _ { i } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } \leq 8 N \sum _ { j = 1 } ^ { N } \mathcal { E } _ { j } ^ { h } ( w ) + O \left( N ^ { 1 3 } + \eta ^ { 2 } \Xi _ { \operatorname* { m a x } } ^ { 2 } N ^ { 2 7 } \mathcal { P } _ { T } ^ { \Xi } \right) .
$$

For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ and positive weight sequence $w = \Big ( w ^ { ( t ) } \Big ) _ { t \leq T }$ satisfying $w ^ { ( t ) } \leq \sqrt { \lambda _ { i } ^ { ( t ) } }$ for every $t \in [ T ]$ , Lemma $\mathrm { A . 7 }$ gives

$$
\mathcal { E } _ { i } ^ { h } ( w ) \leq O \left( N ^ { 1 2 } \Xi _ { i } \mathcal { P } _ { i , T } \right) .\tag{4.1}
$$

For every $r \in [ N ] , S \subseteq [ N ] \setminus \{ r \}$ , and $t \in \mathbb { Z } ,$ set

$$
w _ { r , S } ^ { ( t ) } : = \left( \lambda _ { r } ^ { ( t ) } \right) ^ { 3 / 2 } \prod _ { k \in S } \lambda _ { k } ^ { ( t ) } .\tag{4.2}
$$

Fix $r , j \in [ N ]$ and $S \subseteq [ N ] \setminus \{ r \}$ , and set $h = N - | S | . \operatorname { I f } j \in S \cup \{ r \}$ , then $w _ { r , S } ^ { ( t ) } \leq \sqrt { \lambda _ { j } ^ { ( t ) } }$ for every $t \in [ T ]$ , so (4.1) applies. $\operatorname { I f } j \notin S \cup \{ r \}$ , then $h \geq 2$ and

$$
\mathsf { A } \delta ^ { h - 1 } ( I - \mathsf { D } \delta ^ { N } ) = \left( \mathsf { A } - \delta ^ { N + 1 } \right) \delta ^ { h - 1 } .
$$

The proof of (A.29) verifies the hypothesis of Lemma A.3 for every weight in (4.2), in particular $w _ { r , S \cup \{ j \} }$ . Combining this with Lemmas $\mathrm { A } . 5 , \mathrm { A } . 8$ and A.9 gives

$$
\mathcal { E } _ { j } ^ { N - | S | } ( w _ { r , S } ) \leq O \left( N ^ { 1 3 } \eta ^ { 2 } \Xi _ { \operatorname* { m a x } } ^ { 2 } \sum _ { k = 1 } ^ { N } \mathcal { E } _ { k } ^ { N - | S \cup \{ j \} | } \left( w _ { r , S \cup \{ j \} } \right) + N ^ { 1 2 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) \right) .
$$

The constants are specified in (A.29).

## 4.3 Iterating the bound

For every $r \in [ N ]$ , begin with $S = \varnothing ,$ for which $w _ { r , \emptyset } ^ { ( t ) } = ( \lambda _ { r } ^ { ( t ) } ) ^ { 3 / 2 }$ for every $t \in \mathbb { Z }$ . Consider a current term $\mathcal { E } _ { i } ^ { N - | S | } ( w _ { r , S } )$ , where $r , j \in [ N ]$ and $S \subseteq [ N ] \setminus \{ r \} . \mathrm { ~ I f ~ } j \notin S \cup \{ r \}$ , the preceding estimate replaces S by $S \cup \{ j \}$ and lowers the order by one; otherwise, (4.1) applies. Thus at most $N - 1$ additions precede a term to which (4.1) applies.

Combining the resulting estimates gives

$$
\sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left. \left( \delta ^ { N } \mathbf { g } _ { i } \right) ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq \mathrm { p o l y } ( N ) \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .
$$

For the learning rate in Theorem 3.3, the preceding estimate gives

$$
8 0 \eta \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \Xi _ { i } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left\| \left( \delta ^ { N } g _ { i } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } - \frac { 1 } { 2 \eta } \sum _ { i = 1 } ^ { N } \mathcal { P } _ { i , T } \leq \mathrm { p o l y } ( N ) .
$$

## Acknowledgments

Initial versions of the proofs were developed with the assistance of ChatGPT 5.5 and ChatGPT 5.6 Sol. The authors subsequently streamlined and revised the proofs substantially.

## References

Ioannis Anagnostides, Gabriele Farina, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Tuomas Sandholm. Uncoupled learning dynamics with O(log T) swap regret in multiplayer games. In Conference on Neural Information Processing Systems (NeurIPS), 2022.

Michael Bowling, Neil Burch, Michael Johanson, and Oskari Tammelin. Heads-up Limit Hold’em poker is solved. Science, 347(6218):145–149, 2015.

Noam Brown and Tuomas Sandholm. Superhuman ai for heads-up no-limit poker: Libratus beats top professionals. Science, 359(6374):418–424, 2018.

Noam Brown and Tuomas Sandholm. Superhuman ai for multiplayer poker. Science, 365(6456): 885–890, 2019.

Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university press, 2006.

Xi Chen and Binghui Peng. Hedging in games: Faster convergence of external and swap regrets. In Conference on Neural Information Processing Systems (NeurIPS), 2020.

Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Conference on Learning Theory (COLT), 2012.

Constantinos Daskalakis and Noah Golowich. Fast rates for nonparametric online learning: from realizability to learning in games. In Annual ACM Symposium on Theory of Computing (STOC), 2022.

Constantinos Daskalakis, Maxwell Fishelson, and Noah Golowich. Near-optimal no-regret learning in general games. In Conference on Neural Information Processing Systems (NeurIPS), 2021.

Gabriele Farina, Christian Kroer, Noam Brown, and Tuomas Sandholm. Stable-predictive optimistic counterfactual regret minimization. In International Conference on Machine Learning (ICML), 2019.

Gabriele Farina, Ioannis Anagnostides, Haipeng Luo, Chung-Wei Lee, Christian Kroer, and Tuomas Sandholm. Near-optimal no-regret learning dynamics for general convex games. In Conference on Neural Information Processing Systems (NeurIPS), 2022a.

Gabriele Farina, Chung-Wei Lee, Haipeng Luo, and Christian Kroer. Kernelized multiplicative weights for 0/1-polyhedral games: Bridging the gap between learning in extensive-form and normal-form games. In International Conference on Machine Learning, pages 6337–6357. PMLR, 2022b.

Dylan J Foster, Zhiyuan Li, Thodoris Lykouris, Karthik Sridharan, and Eva Tardos. Learning in games: Robustness of fast convergence. In Conference on Neural Information Processing Systems (NeurIPS), 2016.

Amy Greenwald and Amir Jafari. A general class of no-regret learning algorithms and gametheoretic equilibria. In Learning Theory and Kernel Machines: 16th Annual Conference on Learning Theory and 7th Kernel Workshop (COLT/Kernel), 2003.

Sergiu Hart and Andreu Mas-Colell. A simple adaptive procedure leading to correlated equilibrium. Econometrica, 68(5):1127–1150, 2000.

Elad Hazan. Introduction to online convex optimization. Foundations and Trends® in Optimization, 2(3-4):157–325, 2016.

Panayotis Mertikopoulos, Bruno Lecouat, Houssam Zenati, Chuan-Sheng Foo, Vijay Chandrasekhar, and Georgios Piliouras. Optimistic mirror descent in saddle-point problems: Going the extra (gradient) mile. In International Conference on Learning Representations (ICLR), 2019.

Matej Moravˇcík, Martin Schmid, Neil Burch, Viliam Lisý, Dustin Morrill, Nolan Bard, Trevor Davis, Kevin Waugh, Michael Johanson, and Michael Bowling. DeepStack: Expert-level artificial intelligence in heads-up no-limit poker. Science, 356(6337):508–513, 2017.

Hervé Moulin and J-P Vial. Strategically zero-sum games: the class of games whose completely mixed equilibria cannot be improved upon. International Journal of Game Theory, 7(3-4):201–221, 1978.

Georgios Piliouras, Lillian Ratliff, Ryann Sim, and Stratis Skoulakis. Fast convergence of optimistic gradient ascent in network zero-sum extensive form games. arXiv preprint arXiv:2207.08426, 2022a.

Georgios Piliouras, Ryann Sim, and Stratis Skoulakis. Beyond time-average convergence: Nearoptimal uncoupled online learning via clairvoyant multiplicative weights update. In Conference on Neural Information Processing Systems (NeurIPS), 2022b.

Alexander Rakhlin and Karthik Sridharan. Optimization, learning, and games with predictable sequences. In Conference on Neural Information Processing Systems (NeurIPS), 2013.

Ashkan Soleymani, Georgios Piliouras, and Gabriele Farina. Cautious optimism: A metaalgorithm for near-constant regret in general games. In ACM Conference on Economics and Computation (EC), 2025a.

Ashkan Soleymani, Georgios Piliouras, and Gabriele Farina. Faster rates for no-regret learning in general games via cautious optimism. In Annual ACM Symposium on Theory of Computing (STOC), 2025b.

Vasilis Syrgkanis, Alekh Agarwal, Haipeng Luo, and Robert E Schapire. Fast convergence of regularized learning in games. In Conference on Neural Information Processing Systems (NeurIPS), 2015.

Jun-Kun Wang and Jacob D Abernethy. Acceleration through optimistic no-regret dynamics. In Conference on Neural Information Processing Systems (NeurIPS), 2018.

## A Proof of Theorem 3.3

Set

$$
\gamma : = \eta \Xi _ { \mathrm { m a x } } .\tag{A.1}
$$

For every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , write $J _ { m } ^ { x } ( \pmb { \theta } ) : = J _ { x _ { m } } ( \pmb { \theta } )$ whenever the response is differentiable. For every integer $m \geq 2$ and linear map $J \colon  { \mathbb { R } ^ { m } } \to  { \mathbb { R } ^ { m } }$ , write

$$
\| J \| _ { \infty \to 1 } : = \operatorname* { s u p } _ { \| z \| _ { \infty } \leq 1 } \| J z \| _ { 1 } .
$$

Lemma A.1. For every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the maximizer defining $Q _ { m } ( \pmb \theta )$ is unique and belongs to relint $\widetilde { \Delta } ^ { m }$ . For every integer m $\geq 2 ,$ , the map $Q _ { m }$ is continuously differentiable and $J _ { m } ^ { x }$ is differentiable. For every integer $m \geq 2 ,$ score $\pmb \theta \in \mathbb { R } ^ { m }$ , and direction $\pmb { \xi } \in \mathbb { R } ^ { m }$

$$
\| J _ { m } ^ { x } ( \pmb { \theta } ) \| _ { \infty  1 } \leq 4 \Xi _ { m } \sqrt { \lambda _ { m } ( \pmb { \theta } ) } ,\tag{A.2}
$$

$$
\frac { \mathrm { d } } { \mathrm { d } r } \boldsymbol { J } _ { m } ^ { x } ( \pmb { \theta } + r \pmb { \xi } )  _ { r = 0 }  _ { \infty  1 } \leq \Xi _ { m } \sqrt {  \xi , \frac { \mathrm { d } } { \mathrm { d } r } \boldsymbol { y } _ { m } ( \pmb { \theta } + r \pmb { \xi } )  _ { r = 0 } }  .\tag{A.3}
$$

For every integer m $\geq 2$ and $\pmb { \theta } , \pmb { e } \in \mathbb { R } ^ { m }$ , set $\theta ^ { \prime } : = \theta + e , \mu : = Q _ { m } ( \theta ) , \mu ^ { \prime } : = Q _ { m } ( \theta ^ { \prime } )$ , and $\widehat { \lambda } : =$ max $\{ \lambda _ { m } ( \pmb \theta ) , \lambda _ { m } ( \pmb \theta ^ { \prime } ) \} .  I f \Xi _ { m } \| e \| _ { \infty } \leq 1 / 1 0 0 .$ , then

$$
\left| \log \frac { \lambda _ { m } ( \pmb \theta ^ { \prime } ) } { \lambda _ { m } ( \pmb \theta ) } \right| \le \Xi _ { m } \left\| e \right\| _ { \infty } ,\tag{A.4}
$$

$$
\begin{array} { r } { \left. e , \pmb { y } _ { m } \big ( \pmb { \theta } ^ { \prime } \big ) - \pmb { y } _ { m } \big ( \pmb { \theta } \big ) \right. \leq 2 0 \sqrt { 2 } \Xi _ { m } \widehat { \lambda } ^ { 3 / 2 } \left\| e \right\| _ { \infty } ^ { 2 } , } \end{array}\tag{A.5}
$$

$$
\begin{array} { r } { \sqrt { \hat { \lambda } } \left\| \mathbf { x } _ { m } ( \pmb { \theta } ^ { \prime } ) - \mathbf { x } _ { m } ( \pmb { \theta } ) \right\| _ { 1 } ^ { 2 } \leq 1 6 \sqrt { 2 } \Xi _ { m } \operatorname* { m i n } \left\{ D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) , D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } , \pmb { \mu } ^ { \prime } ) \right\} , } \end{array}\tag{A.6}
$$

$$
D _ { \widetilde { \psi } _ { m } } ^ { \mathrm { s y m } } ( \mu ^ { \prime } , \mu ) \le 4 \operatorname* { m i n } \left\{ D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) , D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } , \pmb { \mu } ^ { \prime } ) \right\} .\tag{A.7}
$$

The proof is deferred to Section B.2.

For a filter $\begin{array} { r } { \mathsf { K } = \sum _ { s > 0 } k _ { s } \mathsf { L } ^ { s } } \end{array}$ , write

$$
{ \sf K } ( \varsigma ) = \sum _ { s \geq 0 } k _ { s } \varsigma ^ { s } , \qquad { \mathrm { k e r } } ( { \sf K } ) = ( k _ { s } ) _ { s \geq 0 } .
$$

Lemma A.2. Set $\begin{array} { r } { \mathsf { K } _ { \star } : = \mathsf { A } - \delta ^ { N + 1 } } \end{array}$ , and, for every integer $1 \leq h \leq N + 1$ , set $\mathsf { K } _ { h } : = \mathsf { A } \delta ^ { h - 1 }$ . For every integer $1 \leq h \leq N + 1$ , the filters in (3.2) satisfy

$$
\delta = \mathsf { A D } , \qquad \mathsf { K } _ { h } \mathsf { D } = \delta ^ { h } ,\tag{A.8}
$$

$$
{ \sf K } _ { h } \big ( { I } - { \mathsf { D } } \delta ^ { N } \big ) = { \sf K } _ { \star } \delta ^ { h - 1 } .\tag{A.9}
$$

For every integer $1 \leq h \leq N + 1 , \mathsf { K } \in \{ \delta ^ { h } , \mathsf { K } _ { h } , \mathsf { K } _ { \star } \} , a n d \mathbb { O } \leq \epsilon \leq 1 / ( 8 N ) , i f \mathsf { k e r } ( \mathsf { K } ) = ( k _ { s } ) _ { s \geq 0 } .$ , then

$$
\sum _ { s \geq 0 } e ^ { \epsilon s / 2 } ( s + 1 ) ^ { 4 } \left. k _ { s } \right. \leq 2 ^ { 2 8 } N ^ { 6 } .\tag{A.10}
$$

For every integer $1 \leq h \leq N + 1$ and $\mathsf { K } \in \mathsf { \{ \delta ^ { h } , K _ { h } , K _ { \star } \} }$ , the $\ell _ { 1 }$ norm and the first four absolute moments of ker(K) are at most $2 ^ { 2 8 } N ^ { 6 }$ . For every integer $1 \leq h \leq N + 1$ , the coefficients of $\delta ^ { h }$ satisfy $\begin{array} { r } { \sum _ { s \geq 0 } \ker ( \delta ^ { h } ) _ { s } = 0 } \end{array}$

For every integer $1 \leq h \leq N + 1$ and ${ \sf K } \in \left\{ \delta ^ { h } , { \sf K } _ { h } , { \sf K } _ { \star } \right\}$ , consider in any normed space two inputs to K that agree at positive times and are equal, respectively, to a fixed vector φ and 0 at every nonpositive time. If their output difference is $\tau ,$ , then

$$
\sum _ { t = 1 } ^ { \infty } \left\| \tau ^ { ( t ) } \right\| ^ { 2 } \leq 2 ^ { 5 9 } N ^ { 1 3 } \left\| \varphi \right\| ^ { 2 } .\tag{A.11}
$$

Lemma A.3. Fix $T \geq 1$ and $\epsilon \geq 0$ . Let z take values in a normed space, with $z ^ { ( t ) } = \mathbf { 0 } \ : f o r \ : t \leq 0 ,$ , and let $\begin{array} { r } { \mathsf { K } = \sum _ { s > 0 } k _ { s } \mathsf { L } ^ { s } } \end{array}$ satisfy $\begin{array} { r } { \sum _ { s \ge 0 } e ^ { \epsilon s / 2 } \left| k _ { s } \right| < \infty } \end{array}$ . Suppose positive weights $w ^ { ( t ) }$ are defined for every integer $t \leq T$ and satisfy

$$
w ^ { ( t ) } \leq e ^ { \epsilon s } w ^ { ( t - s ) } \qquad ( t \in [ T ] , s \geq 0 ) .
$$

Then

$$
\sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| ( \mathsf { K } z ) ^ { ( t ) } \right\| ^ { 2 } \leq \left( \sum _ { s \geq 0 } e ^ { \epsilon s / 2 } \left| k _ { s } \right| \right) ^ { 2 } \sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| z ^ { ( t ) } \right\| ^ { 2 } .
$$

Lemma A.4. For every player $i \in [ N ]$ and time $t \geq 1$

$$
\begin{array} { r } { \left\| e _ { i } ^ { ( t ) } \right\| _ { \infty } \leq 3 \cdot 2 ^ { 2 8 } N ^ { 6 } , \qquad \left\| \left( \left( I - \mathsf { D } \delta ^ { N } \right) g _ { i } \right) ^ { ( t ) } \right\| _ { \infty } \leq 3 \cdot 2 ^ { 2 8 } N ^ { 6 } . } \end{array}\tag{A.12}
$$

$I f 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ , then, for every player $i \in [ N ]$ and time $t \geq 1$ , the two scores in (3.5) at times t and $t + 1$ satisfy the hypothesis of Lemma A.1, and

$$
\left| \log \frac { \lambda _ { i } ^ { ( t + 1 ) } } { \lambda _ { i } ^ { ( t ) } } \right| \le 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } .\tag{A.13}
$$

For every player $i \in [ N ]$ and time $t \geq 1$ , define $\pmb { \theta } _ { i } ^ { ( t ) }$ , and for every player $i \in [ N ]$ , define $\mathbf { \ b { d } } _ { i }$ by

$$
\pmb { \theta } _ { i } ^ { ( t ) } : = \eta \left( S _ { i } ^ { ( t - 1 ) } + \widehat { \pmb { g } } _ { i } ^ { ( t ) } \right) , \qquad \pmb { d } _ { i } : = \big ( I - \mathsf { D } \delta ^ { N } \big ) \pmb { g } _ { i } .
$$

For every player $i \in [ N ]$ and integer $t \leq 0$ , set $\pmb { \theta } _ { i } ^ { ( t ) } = \mathbf { 0 }$ . For every player $i \in [ N ]$ , both $\pmb { \theta } _ { i } ^ { ( 1 ) } - \pmb { \theta } _ { i } ^ { ( 0 ) }$ and $d _ { i } ^ { ( 1 ) }$ vanish. For every $i \in [ N ]$ and integer $t \geq 2 ,$

$$
\pmb { \theta } _ { i } ^ { ( t ) } - \pmb { \theta } _ { i } ^ { ( t - 1 ) } = \eta \left( \pmb { g } _ { i } ^ { ( t - 1 ) } + \widehat { \pmb { g } } _ { i } ^ { ( t ) } - \widehat { \pmb { g } } _ { i } ^ { ( t - 1 ) } \right) = \eta \left( \widehat { \pmb { g } } _ { i } ^ { ( t ) } + \pmb { e } _ { i } ^ { ( t - 1 ) } \right) = \eta \pmb { d } _ { i } ^ { ( t ) } .
$$

For every player $i \in [ N ]$ and time $t \geq 1$ , define

$$
\overline { { \pmb { J } } } _ { i } ^ { ( t ) } : = \eta \int _ { 0 } ^ { 1 } \pmb { J } _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t - 1 ) } + \alpha \eta \pmb { d } _ { i } ^ { ( t ) } \right) \mathrm { d } \alpha .
$$

For every $i \in [ N ]$ and integer $t \leq 0 ,$ , set $\overline { { J } } _ { i } ^ { ( t ) } = \eta J _ { m _ { i } } ^ { x } ( \mathbf { 0 } )$ . The fundamental theorem of calculus $\mathrm { g i v e s , }$ for every $i \in [ N ]$ and integer t,

$$
\left( \mathsf { D } \mathbf { x } _ { i } \right) ^ { ( t ) } = \overline { { \mathbf { J } } } _ { i } ^ { ( t ) } \mathbf { d } _ { i } ^ { ( t ) } .\tag{A.14}
$$

Lemma A.5. Suppose 3 · $2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i \in [ N ]$ and $t \geq 1$

$$
\begin{array} { r } { \left. \overline { { \boldsymbol { J } } } _ { i } ^ { ( t ) } \right. _ { \infty \to 1 } \le 4 \sqrt { 2 } \eta \Xi _ { i } \sqrt { \lambda _ { i } ^ { ( t ) } } \le 4 \sqrt { 2 } \gamma \sqrt { \lambda _ { i } ^ { ( t ) } } , } \end{array}\tag{A.15}
$$

$$
\sqrt { \lambda _ { i } ^ { ( t ) } } \left. \left( \mathsf { D } \pmb { x } _ { i } \right) ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 1 6 \sqrt { 2 } \Xi _ { i } D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) ,\tag{A.16}
$$

$$
\begin{array} { r } { \left\| { ( \mathsf { D } \pmb { x } _ { i } ) } ^ { ( t ) } \right\| _ { 1 } \le 3 \sqrt { 2 } \cdot 2 ^ { 3 0 } \eta \Xi _ { i } N ^ { 6 } \sqrt { \lambda _ { i } ^ { ( t ) } } \le 3 \sqrt { 2 } \cdot 2 ^ { 3 0 } \gamma N ^ { 6 } \sqrt { \lambda _ { i } ^ { ( t ) } } . } \end{array}\tag{A.17}
$$

For every $i \in [ N ]$ and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \left( \mathsf { D } \overline { { \mathbf { J } } } _ { i } \right) ^ { ( t ) } \right. _ { \infty \to 1 } ^ { 2 } \leq 1 6 \eta ^ { 2 } \Xi _ { i } ^ { 2 } \mathcal { P } _ { i , T } .\tag{A.18}
$$

The proofs of these four lemmas are deferred to Section C.

## A.1 The optimistic-FTRL path inequality

Fix $T \geq 1 , i \in [ N ]$ , and a comparator $\widehat { \pmb { \mu } } _ { i } = ( \widehat { \lambda } _ { i } , \widehat { \pmb { y } } _ { i } ) \in \widetilde { \Delta } _ { i }$ . Use $\mathcal { P } _ { i , T }$ from Section 4 and suppress the player index i below. Set $\mathbf { \epsilon } _ { e } ( 0 ) : = \mathbf { 0 }$ . For $0 \leq t \leq T$ , write

$$
\pmb { \mu } ^ { ( t ) } = ( \lambda ^ { ( t ) } , \pmb { y } ^ { ( t ) } ) : = Q \left( \eta \left( \pmb { S } ^ { ( t ) } - \pmb { e } ^ { ( t ) } \right) \right) ,
$$

and set $\pmb { \mu } ^ { ( T + 1 ) } = ( \lambda ^ { ( T + 1 ) } , \pmb { y } ^ { ( T + 1 ) } ) : = { \cal Q } \left( \eta \pmb { S } ^ { ( T ) } \right)$

The increments of the scores defining $\pmb { \mu } ^ { ( 0 ) } , \ldots , \pmb { \mu } ^ { ( T + 1 ) }$ , divided by η, are ${ \pmb g } ^ { ( t ) } - ( { \pmb e } ^ { ( t ) } - { \pmb e } ^ { ( t - 1 ) } )$ for $t \in [ T ]$ and $e ^ { ( T ) }$ for $t = T + 1$ . Reindexing the terms involving e and applying the standard regularized-leader telescope give

$$
\begin{array} { r l } & { \displaystyle \sum _ { t = 1 } ^ { T } \left. g ^ { ( t ) } , \widehat { y } - y ^ { ( t ) } \right. = \displaystyle \sum _ { t = 1 } ^ { T } \left. g ^ { ( t ) } - \left( e ^ { ( t - 1 ) } \right) , \widehat { y } - y ^ { ( t ) } \right. + \left. e ^ { ( T ) } , \widehat { y } - y ^ { ( T + 1 ) } \right. } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \left. e ^ { ( t ) } , y ^ { ( t + 1 ) } - y ^ { ( t ) } \right. } \\ & { \displaystyle \leq \frac { \widetilde { \psi } ( \widehat { \mu } ) - \widetilde { \psi } ( { \mu } ^ { ( 0 ) } ) } { \eta } - \frac { 1 } { \eta } \displaystyle \sum _ { t = 1 } ^ { T + 1 } D _ { \widetilde { \psi } } \left( { \mu } ^ { ( t ) } , { \mu } ^ { ( t - 1 ) } \right) + \displaystyle \sum _ { t = 1 } ^ { T } \left. e ^ { ( t ) } , y ^ { ( t + 1 ) } - y ^ { ( t ) } \right. . } \end{array}
$$

Suppose $2 \eta \Xi _ { i } \left\| e ^ { ( t ) } \right\| _ { \infty } \leq 1 / 1 0 0$ for every $t \in [ T ]$ . For each $t \in [ T ]$ , set $\widetilde { \pmb \mu } ^ { ( t ) } = ( \widetilde \lambda ^ { ( t ) } , \widetilde { \pmb y } ^ { ( t ) } ) : =$ $Q \left( \eta \left( S ^ { ( t ) } + e ^ { \ddot { ( t ) } } \right) \right)$ . For every $t \in [ T ] , ( \mathrm { A } . 4 )$ and (A.5) and exp $( 3 / 2 0 0 ) \leq { \sqrt { 2 } }$ give

$$
D _ { \widetilde { \psi } } \left( \pmb { \mu } ^ { ( t ) } , \widetilde { \pmb { \mu } } ^ { ( t ) } \right) \le D _ { \widetilde { \psi } } ^ { \mathrm { s y m } } \left( \widetilde { \pmb { \mu } } ^ { ( t ) } , \pmb { \mu } ^ { ( t ) } \right)
$$

$$
\begin{array} { r l } & { = \left. 2 \eta e ^ { ( t ) } , \widetilde { \pmb { y } } ^ { ( t ) } - \pmb { y } ^ { ( t ) } \right. } \\ & { \le 2 0 \sqrt { 2 } \Xi _ { i } \left( \sqrt { 2 } \left( \lambda ^ { ( t ) } \right) ^ { 3 / 2 } \right) \left( 2 \eta \right) ^ { 2 } \left\| e ^ { ( t ) } \right\| _ { \infty } ^ { 2 } } \\ & { = 1 6 0 \eta ^ { 2 } \Xi _ { i } \left( \lambda ^ { ( t ) } \right) ^ { 3 / 2 } \left\| e ^ { ( t ) } \right\| _ { \infty } ^ { 2 } . } \end{array}
$$

For every $t \in [ T ]$ , the Bregman three-point identity therefore gives

$$
\begin{array} { r l } & { 2 \eta \left. e ^ { ( t ) } , y ^ { ( t + 1 ) } - y ^ { ( t ) } \right. = D _ { \widetilde { \psi } } \left( \mu ^ { ( t + 1 ) } , \mu ^ { ( t ) } \right) + D _ { \widetilde { \psi } } \left( \mu ^ { ( t ) } , \widetilde { \mu } ^ { ( t ) } \right) - D _ { \widetilde { \psi } } \left( \mu ^ { ( t + 1 ) } , \widetilde { \mu } ^ { ( t ) } \right) } \\ & { \qquad \le D _ { \widetilde { \psi } } \left( \mu ^ { ( t + 1 ) } , \mu ^ { ( t ) } \right) + 1 6 0 \eta ^ { 2 } \Xi _ { i } \left( \lambda ^ { ( t ) } \right) ^ { 3 / 2 } \left. e ^ { ( t ) } \right. _ { \infty } ^ { 2 } . } \end{array}
$$

Summing over $t \in [ T ]$ and using $\pmb { \mu } ^ { ( 1 ) } = \pmb { \mu } ^ { ( 0 ) } = Q \left( \mathbf { 0 } \right)$

$$
- \frac { 1 } { \eta } \sum _ { t = 1 } ^ { T + 1 } D _ { \tilde { \psi } } \left( \mu ^ { ( t ) } , \mu ^ { ( t - 1 ) } \right) + \frac { 1 } { 2 \eta } \sum _ { t = 1 } ^ { T } D _ { \tilde { \psi } } \left( \mu ^ { ( t + 1 ) } , \mu ^ { ( t ) } \right) = - \frac { 1 } { 2 \eta } \sum _ { t = 2 } ^ { T + 1 } D _ { \tilde { \psi } } \left( \mu ^ { ( t ) } , \mu ^ { ( t - 1 ) } \right) \le - \frac { \mathcal { P } _ { i , T } } { 2 \eta } .
$$

Using the preceding bounds and the range estimate $4 \Gamma _ { i }$ implied by Lemma 3.1, we maximize over the comparator, sum over players, and apply (3.1) to obtain

$$
\sum _ { i = 1 } ^ { N } [ \mathrm { R e g } _ { i } ( T ) ] _ { + } \leq \frac { 4 } { \eta } \sum _ { i = 1 } ^ { N } \Gamma _ { i } + 8 0 \eta \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \Xi _ { i } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left\| e _ { i } ^ { ( t ) } \right\| _ { \infty } ^ { 2 } - \frac { 1 } { 2 \eta } \sum _ { i = 1 } ^ { N } \mathcal { P } _ { i , T } .\tag{A.19}
$$

## A.2 Weighted variation bound

Proposition A.6. If

$$
\gamma \leq { \frac { 2 ^ { - 9 4 } } { N ^ { 2 0 } } } ,\tag{A.20}
$$

then, for every $T \geq 1$

$$
\sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left. \left( \delta ^ { N } g _ { i } \right) ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq 2 ^ { 2 0 4 } N ^ { 1 6 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.21}
$$

Proof. Use the extensions $( \pmb { g } _ { i } , \pmb { x } _ { i } , \lambda _ { i } ) _ { i \in [ N ] }$ from Section 3, together with $\overline { { J } } _ { i } ^ { ( t ) } = \eta J _ { m _ { i } } ^ { x } ( \mathbf { 0 } )$ for every $i \in [ N ]$ and $t \leq 0$ . For every integer $1 \overset { \cdot } { \le } h \le N ,$ , (A.9) gives

$$
{ \sf K } _ { h } \left( I - { \sf D } \delta ^ { N } \right) = { \sf K } _ { \star } \delta ^ { h - 1 } .
$$

Fix $r , j \in [ N ]$ and $S \subseteq [ N ] \setminus \{ r \}$ . Since $| S | \le N - 1$

$$
{ \frac { 3 } { 2 } } + \left| S \right| \leq N + { \frac { 1 } { 2 } } .
$$

Hence (A.13) and (A.20) and $N \geq 2$ give, for every $t \in [ T ]$

$$
\begin{array} { r l r } {  {  \log \frac { w _ { r , S } ^ { ( t + 1 ) } } { w _ { r , S } ^ { ( t ) } }  \le 3 \cdot 2 ^ { 2 8 } ( \frac 3 2 + | S | ) \gamma N ^ { 6 } } } \\ & { } & { \le \displaystyle \frac 9 2 2 ^ { 2 8 } \gamma N ^ { 7 } \le \frac 9 2 2 ^ { - 6 6 } N ^ { - 1 3 } \le \frac 1 { 8 N } . } \end{array}
$$

The same rate condition also gives

$$
\begin{array} { c } { { 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 3 \cdot 2 ^ { - 6 6 } N ^ { - 1 4 } \leq \displaystyle \frac { 1 } { 1 0 0 } , } } \\ { { 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 7 } \leq 3 \cdot 2 ^ { - 6 6 } N ^ { - 1 3 } \leq \displaystyle \frac { 1 } { 4 } , } } \\ { { \gamma \leq N ^ { - 2 0 } . } } \end{array}
$$

These inequalities verify the numerical hypotheses of Lemmas $\mathrm { A . 7 }$ to A.9. For every $i \in [ N ] ,$ $\lambda _ { i } ^ { ( 1 ) } = \lambda _ { m _ { i } } \mathbf { \bar { ( } 0 ) } = \lambda _ { i } ^ { ( 0 ) }$ . Telescoping the logarithmic bound and using $\lambda _ { i } ^ { ( \tau ) } = \lambda _ { i } ^ { ( 0 ) }$ for every $i \in [ N ]$ and integer $\tau \leq 0$ gives, for every $t \in [ T ]$ and $s \geq 0 ,$

$$
w _ { r , S } ^ { ( t ) } \leq e ^ { s / ( 8 N ) } w _ { r , S } ^ { ( t - s ) } .
$$

The kernel bound in (A.10) therefore verifies the remaining hypothesis of Lemma ${ \mathrm { A } } . 3$ with $w = w _ { r , S }$ and $\mathsf { K } \in \{ \mathsf { K } _ { \star } , \mathsf { K } _ { h } \}$ for every integer $1 \leq h \leq N$

Lemma A.7. Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ and $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 7 } \leq 1 / 4$ . For every $j \in \ [ N ]$ , integer $1 \leq h \leq N .$ , and horizon $T \geq 1 .$

$$
\sum _ { t = 1 } ^ { T } \sqrt { \lambda _ { j } ^ { ( t ) } } \left. \left( \delta ^ { h } \pmb { x } _ { j } \right) ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 2 ^ { 6 0 } \sqrt { 2 } N ^ { 1 2 } \Xi _ { j } \mathcal { P } _ { j , T } .
$$

The proof is deferred to Section C.2.

There are two cases. If $j \in S \cup \{ r \}$ , then, for every $t \in [ T ] , w _ { r , S } ^ { ( t ) } \leq \sqrt { \lambda _ { j } ^ { ( t ) } }$ , so Lemma $\mathrm { A . 7 }$ gives

$$
\mathcal { E } _ { j } ^ { N - | S | } ( w _ { r , S } ) \leq 2 ^ { 6 1 } N ^ { 1 2 } \mathcal { P } _ { T } ^ { \Xi } .\tag{A.22}
$$

For every player $i \in [ N ]$ , regard ${ \overline { { \mathbf { J } } } } _ { i }$ as pointwise multiplication on score sequences. For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and score sequence $z ,$ define

$$
[ \mathsf { K } _ { h } , \overline { { \boldsymbol { J } } } _ { i } ] \boldsymbol { z } : = \mathsf { K } _ { h } ( \overline { { \boldsymbol { J } } } _ { i } \boldsymbol { z } ) - \overline { { \boldsymbol { J } } } _ { i } ( \mathsf { K } _ { h } \boldsymbol { z } ) .
$$

For every $i \in [ N ]$ and integer $1 \leq h \leq N ,$ , define

$$
\begin{array} { r } { \pmb { c } _ { i , h } : = [ \mathsf { K } _ { h } , \overline { { \pmb { J } } } _ { i } ] \pmb { d } _ { i } . } \end{array}
$$

Lemma A.8. Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i \in [ N ]$ , integer $1 \leq h \leq N .$ , and time $t \geq 1$

$$
\left( \delta ^ { h } \mathbf { x } _ { i } \right) ^ { ( t ) } = \overline { { \pmb { J } } } _ { i } ^ { ( t ) } \left( \mathsf { K } _ { \star } \delta ^ { h - 1 } \pmb { g } _ { i } \right) ^ { ( t ) } + \pmb { c } _ { i , h } ^ { ( t ) } .\tag{A.23}
$$

For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. c _ { i , h } ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 9 \cdot 2 ^ { 1 1 6 } \eta ^ { 2 } \Xi _ { i } ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } \leq 9 \cdot 2 ^ { 1 1 6 } \gamma ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } .\tag{A.24}
$$

The proof is deferred to Section C.2.

Suppose instead that j $\not \in S \cup \{ r \}$ , and set $h = N - | S |$ . Then $| S | \le N - 2 , s o h \ge 2$ . By Lemma A.8 and (A.23),

$$
\mathcal { E } _ { j } ^ { N - | S | } ( w _ { r , S } ) \leq 2 \sum _ { t = 1 } ^ { T } w _ { r , S } ^ { ( t ) } \left\| \overline { { \boldsymbol { J } } } _ { j } ^ { ( t ) } \left( \mathsf { K } _ { \star } \delta ^ { h - 1 } \pmb { g } _ { j } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 2 } + 2 \sum _ { t = 1 } ^ { T } w _ { r , S } ^ { ( t ) } \left\| \boldsymbol { c } _ { j , h } ^ { ( t ) } \right\| _ { 1 } ^ { 2 } .\tag{A.25}
$$

Lemma A.9. Let

$$
C _ { \mathrm { m u l t i } } : = 2 \operatorname* { m a x } \left\{ 2 ^ { 5 9 } , 9 \sqrt { 2 } \cdot 2 ^ { 1 2 1 } \right\} .
$$

Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i , j \in [ N ]$ and profile $\textstyle { \pmb { x } } \in \prod _ { k = 1 } ^ { N } \Delta ^ { m _ { k } }$

$$
\operatorname* { s u p } _ { \pmb { \xi } \in \mathbb { R } ^ { m _ { j } } , \ \lVert \pmb { \xi } \rVert _ { 1 } \leq 1 } \left. \partial _ { j } \mathcal { G } _ { i } ( \pmb { x } ) [ \pmb { \xi } ] \right. _ { \infty } \leq 2 .
$$

For every $i \in [ N ]$ and integer $1 \leq h \leq N .$ , there is a sequence $\chi _ { i , h }$ such that, for every $t \geq 1$

$$
\left( \delta ^ { h } \pmb { g } _ { i } \right) ^ { ( t ) } = \sum _ { j = 1 } ^ { N } \partial _ { j } \mathcal { G } _ { i } \left( \pmb { x } ^ { ( t ) } \right) \left( \delta ^ { h } \pmb { x } _ { j } \right) ^ { ( t ) } + \chi _ { i , h } ^ { ( t ) } .\tag{A.26}
$$

For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \boldsymbol { x } _ { i , h } ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq C _ { \mathrm { m u l t i } } \left( N ^ { 1 3 } + \gamma ^ { 2 } N ^ { 2 7 } \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.27}
$$

$I f \gamma \leq N ^ { - 2 0 }$ , then, for every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \boldsymbol { x } _ { i , h } ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq C _ { \mathrm { m u l t i } } N ^ { 1 3 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.28}
$$

The proof is deferred to Section C.3.

Since $C _ { \mathrm { m u l t i } } = 2$ max $\left\{ 2 ^ { 5 9 } , 9 { \sqrt { 2 } } \cdot 2 ^ { 1 2 1 } \right\} , 9 { \sqrt { 2 } } < 1 6$ , and $9 < 1 6 ,$

$$
C _ { \mathrm { m u l t i } } < 2 ^ { 1 2 8 } , \qquad 9 \cdot 2 ^ { 1 1 6 } < 2 ^ { 1 2 0 } .
$$

By Lemma A.5 and (A.15), $\left\| \overline { { J } } _ { j } ^ { ( t ) } \right\| _ { \infty \to 1 } \le 8 \gamma \sqrt { \lambda _ { j } ^ { ( t ) } }$ for every $t \geq 1$ . Since $j \notin S \cup \{ r \}$ , for every integer $t \leq T$

$$
\begin{array} { r l } & { w _ { r , S } ^ { ( t ) } \lambda _ { j } ^ { ( t ) } = w _ { r , S \cup \{ j \} } ^ { ( t ) } , } \\ & { \quad h - 1 = N - | S | - 1 = N - | S \cup \{ j \} | . } \end{array}
$$

The weight bound above, with $S \cup \{ j \}$ in place of $S ,$ verifies the hypotheses of Lemma A.3 for $w = w _ { r , S \cup \{ j \} }$ . Applying that lemma with $\mathsf { K } = \mathsf { K } _ { \star }$ <sub>⋆</sub>, followed by (A.26) at order $h - 1$ , gives

$$
\begin{array} { l } { \displaystyle \sum _ { t = 1 } ^ { T } w _ { r , S } ^ { ( t ) } \left\| \overline { { J } } _ { j } ^ { ( t ) } \left( \mathsf { K } _ { \star } \delta ^ { h - 1 } g _ { j } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 2 } } \\ { \displaystyle \leq 6 4 \cdot 2 ^ { 5 6 } \gamma ^ { 2 } N ^ { 1 2 } \displaystyle \sum _ { t = 1 } ^ { T } w _ { r , S } ^ { ( t ) } \lambda _ { j } ^ { ( t ) } \left\| \left( \delta ^ { h - 1 } g _ { j } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } } \\ { \displaystyle \leq 5 1 2 \cdot 2 ^ { 5 6 } \gamma ^ { 2 } N ^ { 1 3 } \displaystyle \sum _ { k = 1 } ^ { N } \mathcal { E } _ { k } ^ { N - | S \cup \{ j \} | } \left( w _ { r , S \cup \{ j \} } \right) + 1 2 8 \cdot 2 ^ { 5 6 } C _ { \mathrm { m u l t i } } \gamma ^ { 2 } N ^ { 2 5 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) . } \end{array}
$$

Here (A.28) applies because $w _ { r , S } ^ { ( t ) } \lambda _ { j } ^ { ( t ) } \leq 1$ for every $t \in [ T ]$ . Moreover, (A.24) gives

$$
\sum _ { t = 1 } ^ { T } w _ { r , S } ^ { ( t ) } \left. \pmb { c } _ { j , h } ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 9 \cdot 2 ^ { 1 1 6 } \gamma ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { j , T } .
$$

Substituting the last two displays into (A.25) and using $\gamma \leq N ^ { - 2 0 }$ gives

$$
\mathcal { E } _ { j } ^ { N - | S | } ( w _ { r , S } ) \leq 2 ^ { 2 0 0 } N ^ { 1 3 } \gamma ^ { 2 } \sum _ { k = 1 } ^ { N } \mathcal { E } _ { k } ^ { N - | S \cup \{ j \} | } \left( w _ { r , S \cup \{ j \} } \right) + 2 ^ { 2 0 0 } N ^ { 1 2 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.29}
$$

Since $r , j ,$ and S were arbitrary, the preceding two case bounds hold throughout their stated ranges. For every integer $0 \leq s \leq N - 1$ , define

$$
M _ { s } : = \operatorname* { m a x } \left\{ \mathcal { E } _ { i } ^ { N - s } ( w _ { r , S } ) : r , i \in [ N ] , S \subseteq [ N ] \setminus \{ r \} , | S | = s \right\} .\tag{A.30}
$$

When $s = N - 1$ , every feasible pair $( r , S )$ in (A.30) satisfies $S = [ N ] \setminus \{ r \}$ . Hence $j \in S \cup \{ r \}$ for every $j \in [ N ]$ , and (A.22) applies at order one. For every integer $0 \leq s < N - 1$ , (A.22) and (A.29) give

$$
M _ { s } \leq \operatorname* { m a x } \left\{ 2 ^ { 2 0 0 } N ^ { 1 2 } \mathcal { P } _ { T } ^ { \Xi } , 2 ^ { 2 0 0 } N ^ { 1 4 } \gamma ^ { 2 } M _ { s + 1 } + 2 ^ { 2 0 0 } N ^ { 1 2 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) \right\} .
$$

Moreover,

$$
2 ^ { 2 0 0 } N ^ { 1 4 } \gamma ^ { 2 } \leq 2 ^ { 1 2 } N ^ { - 2 6 } \leq 2 ^ { - 1 4 } \leq \frac { 1 } { 2 } .
$$

Backward induction thus proves, for every integer $0 \leq s \leq N - 1$

$$
\begin{array} { r } { M _ { s } \leq 2 ^ { 2 0 1 } N ^ { 1 2 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) . } \end{array}\tag{A.31}
$$

![](images/a9c1f7b3fa25bd8d08fb0a9eb0d41e0820e57e3e28436b57729d1fcad6b5f356.jpg)

Finally, apply (A.26) at order N and use (A.31) with $s = 0$ . For every $r \in [ N ]$

$$
\sum _ { t = 1 } ^ { T } \left( \lambda _ { r } ^ { ( t ) } \right) ^ { 3 / 2 } \left\| \left( \delta ^ { N } g _ { r } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } \leq 8 N \sum _ { j = 1 } ^ { N } \mathcal { E } _ { j } ^ { N } ( w _ { r , \emptyset } ) + 2 C _ { \mathrm { m u l t i } } N ^ { 1 3 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .
$$

Summing over $r ,$ using $C _ { \mathrm { m u l t i } } < 2 ^ { 1 2 8 }$ , and applying (A.31) prove

$$
\begin{array} { r l } { \displaystyle \sum _ { r = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left( \lambda _ { r } ^ { ( t ) } \right) ^ { 3 / 2 } \left\| \left( \delta ^ { N } g _ { r } \right) ^ { ( t ) } \right\| _ { \infty } ^ { 2 } \leq \left( 2 ^ { 2 0 4 } N ^ { 1 5 } + 2 ^ { 1 2 9 } N ^ { 1 4 } \right) \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) } & { } \\ { \leq 2 ^ { 2 0 4 } N ^ { 1 6 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) , } \end{array}
$$

as claimed.

Proof of Theorem 3.3. Fix $T \geq 1$ and set $\eta = 2 ^ { - 9 4 } / \left( N ^ { 2 0 } \Xi _ { \mathrm { m a x } } \right)$ . Then (A.20) holds, and

$$
6 \cdot 2 ^ { 2 8 } \eta \Xi _ { \mathrm { m a x } } N ^ { 6 } = 6 \cdot 2 ^ { - 6 6 } N ^ { - 1 4 } \le { \frac { 1 } { 1 0 0 } } .
$$

Thus, by Lemma $\begin{array} { r } { \mathbb { A } . 4 , \ : 2 \eta \Xi _ { i } \left\| e _ { i } ^ { ( t ) } \right\| _ { \infty } \leq 1 / 1 0 0 } \end{array}$ for every $i \in [ N ]$ and $t \in [ T ]$ , as required in the derivation of (A.19). By (A.21) and (3.3),

$$
8 0 \eta \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \Xi _ { i } \left( \lambda _ { i } ^ { ( t ) } \right) ^ { 3 / 2 } \left. \boldsymbol { e } _ { i } ^ { ( t ) } \right. _ { \infty } ^ { 2 } \le 8 0 \cdot 2 ^ { 1 1 0 } N ^ { - 4 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) ,
$$

$$
\frac { 1 } { 2 \eta } \sum _ { i = 1 } ^ { N } \mathcal { P } _ { i , T } \geq 2 ^ { 9 3 } N ^ { 2 0 } \mathcal { P } _ { T } ^ { \Xi } .
$$

Since $8 0 \cdot 2 ^ { 1 7 } \leq 2 ^ { 2 4 } \leq N ^ { 2 4 }$ , the coefficient of $\mathcal { P } _ { T } ^ { \Xi }$ after substitution into (A.19) is nonpositive. Hence

$$
\sum _ { i = 1 } ^ { N } [ \mathrm { R e g } _ { i } ( T ) ] _ { + } \leq 4 \cdot 2 ^ { 9 4 } N ^ { 2 0 } \Xi _ { \mathrm { m a x } } \sum _ { i = 1 } ^ { N } \Gamma _ { i } + 8 0 \cdot 2 ^ { 1 1 0 } N ^ { - 4 } .
$$

The definitions of $\Gamma _ { i } , \Xi _ { i } ,$ and $m _ { \mathrm { m a x } }$ give

$$
\Xi _ { \operatorname* { m a x } } \sum _ { i = 1 } ^ { N } \Gamma _ { i } \leq 4 \cdot 1 0 ^ { 4 } N \left( 1 + \log \left( m _ { \operatorname* { m a x } } + 1 \right) \right) ^ { 4 } .
$$

Consequently,

$$
\begin{array} { l } { { \displaystyle \sum _ { i = 1 } ^ { N } [ \mathrm { R e } { } \mathrm { g } _ { i } ( T ) ] _ { + } \leq 5 \cdot 2 ^ { 1 0 9 } N ^ { 2 1 } \left( 1 + \log \left( m _ { \mathrm { m a x } } + 1 \right) \right) ^ { 4 } + 8 0 \cdot 2 ^ { 1 1 0 } N ^ { - 4 } } \ ~ } \\ { { \displaystyle \leq 7 \cdot 2 ^ { 1 0 9 } N ^ { 2 1 } \left( 1 + \log \left( m _ { \mathrm { m a x } } + 1 \right) \right) ^ { 4 } } \ ~ } \\ { { \displaystyle \leq 2 ^ { 1 1 2 } N ^ { 2 1 } \left( 1 + \log \left( m _ { \mathrm { m a x } } + 1 \right) \right) ^ { 4 } } . } \end{array}
$$

## B Geometry of $Q _ { m }$

Fix an integer $m \geq 2$ . Recall $\psi _ { m }$ and $\Gamma _ { m }$ from Section 3:

$$
\psi _ { m } ( x ) = \sum _ { a = 1 } ^ { m } x ( a ) \log x ( a ) , \qquad \Gamma _ { m } = ( \log m ) ^ { 2 } + 2 \log m + 2 .
$$

Throughout this section, every dummy action index ranges over $[ m ] ;$ in particular, $\begin{array} { r } { \sum _ { a } : = \sum _ { a \in [ m ] } } \end{array}$

## B.1 Convexity of $\widetilde { \psi } _ { m }$

For an interior $\pmb { x } \in \Delta ^ { m }$ , define

$$
W _ { m } ( { \pmb x } ) : = \sum _ { a } { x ( a ) \left( \log { x ( a ) } - \psi _ { m } ( { \pmb x } ) \right) ^ { 2 } } .
$$

Let $a \sim x$ and set $\varepsilon : = - \log x ( a )$ . For every $t \geq 0 ,$

$$
\operatorname* { P r } ( \varepsilon \geq t ) = \sum _ { a : x ( a ) \leq e ^ { - t } } x ( a ) \leq \operatorname* { m i n } \left\{ 1 , m e ^ { - t } \right\} ,
$$

so the tail-integral formula gives

$$
\begin{array} { r l } & { \mathbb { E } \left[ \varepsilon ^ { 2 } \right] \leq ( \log m ) ^ { 2 } + 2 \log m + 2 = \Gamma _ { m } , } \\ & { \mathbb { E } \left[ \varepsilon ^ { 4 } \right] \leq ( \log m ) ^ { 4 } + 4 ( \log m ) ^ { 3 } + 1 2 ( \log m ) ^ { 2 } + 2 4 \log m + 2 4 \leq 4 \Gamma _ { m } ^ { 2 } . } \end{array}
$$

The last inequality follows by expansion and holds for $m \geq 2$ . Thus

$$
W _ { m } ( { \pmb x } ) = \mathrm { V a r } ( \varepsilon ) \leq \Gamma _ { m } ,\tag{B.1}
$$

$$
\sum _ { a } x ( a ) \left| \log x ( a ) - \psi _ { m } ( x ) \right| ^ { 4 } = \operatorname { \mathbb { E } } \left[ \left| \varepsilon - \operatorname { \mathbb { E } } \left[ \varepsilon \right] \right| ^ { 4 } \right] \leq 8 \left( \operatorname { \mathbb { E } } \left[ \varepsilon ^ { 4 } \right] + \left( \operatorname { \mathbb { E } } \left[ \varepsilon \right] \right) ^ { 4 } \right) \leq 6 4 \Gamma _ { m } ^ { 2 } .\tag{B.2}
$$

For every $z \in \mathbb { R } ^ { m }$ satisfying $\mathbf { 1 } ^ { \top } z = 0$

$$
\begin{array} { l } { \displaystyle \langle \nabla \psi _ { m } ( { \pmb x } ) , { \pmb z } \rangle = \sum _ { a } \left( \log { x ( a ) } - \psi _ { m } ( { \pmb x } ) \right) z ( a ) . } \\ { \displaystyle \langle \nabla \psi _ { m } ( { \pmb x } ) , { \pmb z } \rangle ^ { 2 } \le \Gamma _ { m } \sum _ { a } \frac { z ( a ) ^ { 2 } } { x ( a ) } . } \end{array}\tag{B.3}
$$

The last inequality follows from weighted Cauchy–Schwarz and (B.1).

For $\lambda > 0$ and $\begin{array} { r } { \pmb { y } = \lambda \pmb { x } , } \end{array}$ recall $\widetilde { \psi } _ { m }$ from Section 3:

$$
\widetilde { \psi } _ { m } ( \lambda , \pmb { y } ) = - \sqrt { 1 - \lambda } + \sqrt { \lambda } \left[ \psi _ { m } ( \pmb { y } / \lambda ) - 2 \Gamma _ { m } - 3 \right]
$$

on $\widetilde { \Delta } ^ { m }$ , with $\widetilde { \psi } _ { m } ( 0 , \mathbf { 0 } ) = - 1$

Lemma 3.1. For every integer $m \geq 2$ , the function $\widetilde { \psi } _ { m }$ is finite and convex on $\widetilde { \Delta } ^ { m }$ and strictly convex in its relative interior. For every integer m $\geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the maximizer defining $Q _ { m } ( \pmb \theta )$ is unique and belongs to relint $\widetilde { \Delta } ^ { m }$ . More explicitly, for every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the function on $( 0 , 1 )$ given by

$$
\lambda ^ { \prime } \longmapsto { \frac { 1 } { \sqrt { 1 - \lambda ^ { \prime } } } } - { \frac { 2 \Gamma _ { m } + 3 + \log \left( \sum _ { a = 1 } ^ { m } \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) \right) } { \sqrt { \lambda ^ { \prime } } } } - { \frac { \sum _ { a = 1 } ^ { m } \theta ( a ) \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) } { \sum _ { a = 1 } ^ { m } \exp \left( { \sqrt { \lambda ^ { \prime } } } \theta ( a ) \right) } }
$$

is strictly increasing, tends to −∞ at the left endpoint, and tends $t o + \infty$ at the right endpoint. $L e t$ $\lambda \in ( 0 , 1 )$ denote its unique zero. Then

$$
Q _ { m } ( \pmb \theta ) = \left( \lambda , \frac { \lambda \left( \exp \left( \sqrt \lambda \theta ( a ) \right) \right) _ { a \in [ m ] } } { \sum _ { b = 1 } ^ { m } \exp \left( \sqrt \lambda \theta ( b ) \right) } \right) .
$$

Moreover, for every integer $m \geq 2 ,$

$$
\operatorname* { s u p } _ { \pmb { \mu } \in \widetilde { \Delta } ^ { m } } \widetilde { \psi } _ { m } ( \pmb { \mu } ) - \operatorname* { i n f } _ { \pmb { \mu } \in \widetilde { \Delta } ^ { m } } \widetilde { \psi } _ { m } ( \pmb { \mu } ) \leq 2 \Gamma _ { m } + \log m + 4 .
$$

Proof. Fix an interior point ${ \pmb \mu } = ( \lambda , \lambda { \pmb x } )$ and write a tangent vector as $( s , s x + \lambda z )$ , where $s \in$ R and $\mathbf { 1 } ^ { \top } z = 0$ . Define

$$
\begin{array} { l } { v : = \left( x ( a ) \left( \log x ( a ) - \psi _ { m } ( \pmb { x } ) \right) \right) _ { a \in [ m ] } , } \\ { \beta : = 2 \Gamma _ { m } + 3 - \psi _ { m } ( \pmb { x } ) - W _ { m } ( \pmb { x } ) , \qquad \omega _ { \lambda } : = \left( \displaystyle \frac { \lambda } { 1 - \lambda } \right) ^ { 3 / 2 } . } \end{array}
$$

Since $\mathbf { 1 } ^ { \top } z = 0$

$$
\langle \nabla \psi _ { m } ( { \pmb x } ) , { \pmb z } \rangle = \sum _ { a } \frac { v ( a ) { \pmb z } ( a ) } { { \pmb x } ( a ) } , \qquad W _ { m } ( { \pmb x } ) = \sum _ { a } \frac { v ( a ) ^ { 2 } } { { \pmb x } ( a ) } .
$$

Direct differentiation gives

$$
\begin{array} { r l } & { \quad ( s , s \pm \lambda z ) ^ { \top } \nabla ^ { 2 } \widetilde { \psi } _ { m } ( \pmb { \mu } ) \left( s , s \pmb { x } + \lambda z \right) } \\ & { = \sqrt { \lambda } \displaystyle \sum _ { a } \frac { z ( a ) ^ { 2 } } { x ( a ) } - \lambda ^ { - 1 / 2 } s \displaystyle \sum _ { a } \frac { v ( a ) z ( a ) } { x ( a ) } + \frac { 2 \Gamma _ { m } + 3 - \psi _ { m } ( \pmb { x } ) + \omega _ { \lambda } } { 4 \lambda ^ { 3 / 2 } } s ^ { 2 } } \\ & { = \sqrt { \lambda } \displaystyle \sum _ { a } \frac { \left[ z ( a ) - ( s / ( 2 \lambda ) ) v ( a ) \right] ^ { 2 } } { x ( a ) } + \frac { 2 \Gamma _ { m } + 3 - \psi _ { m } ( \pmb { x } ) - W _ { m } ( \pmb { x } ) + \omega _ { \lambda } } { 4 \lambda ^ { 3 / 2 } } s ^ { 2 } . } \end{array}\tag{B.4}
$$

By (B.1) and the bounds $\psi _ { m } ( \pmb { x } ) \leq 0$ and $- \psi _ { m } ( \pmb { x } ) \leq \log m \leq \sqrt { \Gamma _ { m } } ,$

$$
\Gamma _ { m } + 3 \leq \beta \leq 3 ( 1 + \Gamma _ { m } ) .\tag{B.5}
$$

By (B.3) and Young’s inequality,

$$
\lambda ^ { - 1 / 2 } \left| s \right| \left| \sum _ { a } \frac { v ( a ) z ( a ) } { x ( a ) } \right| \leq \frac { 1 } { 2 } \sqrt { \lambda } \sum _ { a } \frac { z ( a ) ^ { 2 } } { x ( a ) } + \frac { \Gamma _ { m } } { 2 \lambda ^ { 3 / 2 } } s ^ { 2 } .
$$

Substitution gives

$$
( s , s { \pmb x } + \lambda z ) ^ { \top } \nabla ^ { 2 } \widetilde \psi _ { m } ( { \pmb \mu } ) ( s , s { \pmb x } + \lambda z ) \geq \frac { 1 } { 2 } \sqrt { \lambda } \sum _ { a } \frac { z ( a ) ^ { 2 } } { x ( a ) } + \frac { 3 s ^ { 2 } } { 4 \lambda ^ { 3 / 2 } } + \frac { s ^ { 2 } } { 4 ( 1 - \lambda ) ^ { 3 / 2 } } .\tag{B.6}
$$

Continuity at $\lambda \ = \ 0 ,$ convexity on the closed domain, and the range bound follow from − log $m \leq \psi _ { m } ( \pmb { x } ) \leq 0$ . The one-sided λ-derivatives of $\widetilde { \psi } _ { m }$ diverge at $\lambda \in \{ 0 , 1 \}$ , while $\partial _ { a } \psi _ { m } ( { \pmb x } ) =$ $1 + \log x ( a ) \to - \infty$ as $x ( a ) \downarrow 0$ . Hence $Q _ { m } ( \pmb \theta )$ is interior for every $\pmb \theta \in \mathbb { R } ^ { m } ;$ strict convexity gives uniqueness. Fix $\pmb \theta \in \mathbb { R } ^ { m }$ and $\lambda ^ { \prime } \in ( 0 , 1 )$ . Entropy conjugacy gives

$$
\operatorname* { m a x } _ { \boldsymbol { x } \in \Delta ^ { m } } \Big \{ \lambda ^ { \prime } \langle \pmb { \theta } , \boldsymbol { x } \rangle - \sqrt { \lambda ^ { \prime } } \psi _ { m } ( \pmb { x } ) \Big \} = \sqrt { \lambda ^ { \prime } } \log \left( \sum _ { a } \mathrm { e x p } \left( \sqrt { \lambda ^ { \prime } } \theta ( a ) \right) \right) ,
$$

and its unique maximizer is

$$
\frac { \Bigl ( \exp { \Bigl ( \sqrt { \lambda ^ { \prime } } \theta ( a ) \Bigr ) } \Bigr ) _ { a \in [ m ] } } { \sum _ { a } \exp { \Bigl ( \sqrt { \lambda ^ { \prime } } \theta ( a ) \Bigr ) } } .
$$

Thus, after optimizing over x, the objective defining $Q _ { m } ( \pmb \theta )$ as a function of $\lambda ^ { \prime }$ is

$$
\sqrt { 1 - \lambda ^ { \prime } } + \sqrt { \lambda ^ { \prime } } \left[ 2 \Gamma _ { m } + 3 + \log \left( \sum _ { a } \exp \left( \sqrt { \lambda ^ { \prime } } \theta ( a ) \right) \right) \right] .
$$

Its derivative is

$$
- \frac { 1 } { 2 \sqrt { 1 - \lambda ^ { \prime } } } + \frac { 2 \Gamma _ { m } + 3 + \log \left( \sum _ { a } \exp \left( \sqrt { \lambda ^ { \prime } } \theta ( a ) \right) \right) } { 2 \sqrt { \lambda ^ { \prime } } } + \frac { 1 } { 2 } \frac { \sum _ { a } \theta ( a ) \exp \left( \sqrt { \lambda ^ { \prime } } \theta ( a ) \right) } { \sum _ { a } \exp \left( \sqrt { \lambda ^ { \prime } } \theta ( a ) \right) } .
$$

Fix distinct $\lambda _ { 0 } , \lambda _ { 1 } \in ( 0 , 1 )$ . For every $j \in \{ 0 , 1 \}$ , let $\pmb { y } _ { j } \in \lambda _ { j } \Delta ^ { m }$ maximize $\langle \pmb \theta , \pmb y \rangle - \widetilde { \psi } _ { m } ( \lambda _ { j } , \pmb y )$ . For every $t \in ( 0 , 1 )$ ,

$$
( 1 - t ) { \pmb y } _ { 0 } + t { \pmb y } _ { 1 } \in \left( ( 1 - t ) \lambda _ { 0 } + t \lambda _ { 1 } \right) \Delta ^ { m } .
$$

Strict concavity of the lifted objective therefore implies strict concavity of the scalar objective, so its derivative is strictly decreasing. The function in the lemma is −2 times this derivative and is therefore strictly increasing. The derivative tends to +∞ as $\lambda ^ { \prime } \downarrow$ 0 and to −∞ as $\lambda ^ { \prime } \uparrow 1 .$ which gives the two limits in the lemma. Its unique zero and the conditional maximizer give the formula for $Q _ { m } ( \pmb \theta )$ . □

## B.2 Bounds for $Q _ { m }$

Lemma A.1. For every integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , the maximizer defining $Q _ { m } ( \pmb \theta )$ is unique and belongs to relint $\widetilde { \Delta } ^ { m }$ . For every integer $m \geq 2 ,$ the map $Q _ { m }$ is continuously differentiable and $J _ { m } ^ { x }$ is differentiable. For every integer $m \geq 2 ,$ , score $\pmb \theta \in \mathbb { R } ^ { m }$ , and direction $\pmb { \xi } \in \mathbb { R } ^ { m }$

$$
\| J _ { m } ^ { x } ( \pmb { \theta } ) \| _ { \infty  1 } \leq 4 \Xi _ { m } \sqrt { \lambda _ { m } ( \pmb { \theta } ) } ,\tag{A.2}
$$

$$
\frac { \mathrm { d } } { \mathrm { d } r } \boldsymbol { J } _ { m } ^ { x } ( \pmb { \theta } + r \pmb { \xi } )  _ { r = 0 }  _ { \infty \to 1 } \le \Xi _ { m } \sqrt {  \xi , \frac { \mathrm { d } } { \mathrm { d } r } \boldsymbol { y } _ { m } ( \pmb { \theta } + r \pmb { \xi } )  _ { r = 0 }  } .\tag{A.3}
$$

For every integer m $\geq 2$ and $\pmb { \theta } , \pmb { e } \in \mathbb { R } ^ { m }$ , set $\theta ^ { \prime } : = \theta + e , \mu : = Q _ { m } ( \theta ) , \mu ^ { \prime } : = Q _ { m } ( \theta ^ { \prime } )$ , and $\widehat { \lambda } : =$ max $\{ \lambda _ { m } ( \pmb \theta ) , \lambda _ { m } ( \pmb \theta ^ { \prime } ) \} .  I f \Xi _ { m } \| e \| _ { \infty } \leq 1 / 1 0 0 .$ , then

$$
\left| \log \frac { \lambda _ { m } ( \pmb \theta ^ { \prime } ) } { \lambda _ { m } ( \pmb \theta ) } \right| \le \Xi _ { m } \left\| e \right\| _ { \infty } ,\tag{A.4}
$$

$$
\begin{array} { r } { \left. e , \pmb { y } _ { m } ( \pmb { \theta } ^ { \prime } ) - \pmb { y } _ { m } ( \pmb { \theta } ) \right. \leq 2 0 \sqrt { 2 } \Xi _ { m } \widehat { \lambda } ^ { 3 / 2 } \left\| e \right\| _ { \infty } ^ { 2 } , } \end{array}\tag{A.5}
$$

$$
\begin{array} { r } { \sqrt { \hat { \lambda } } \left\| \mathbf { x } _ { m } ( \pmb { \theta } ^ { \prime } ) - \mathbf { x } _ { m } ( \pmb { \theta } ) \right\| _ { 1 } ^ { 2 } \leq 1 6 \sqrt { 2 } \Xi _ { m } \operatorname* { m i n } \left\{ D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) , D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } , \pmb { \mu } ^ { \prime } ) \right\} , } \end{array}\tag{A.6}
$$

$$
D _ { \widetilde { \psi } _ { m } } ^ { \mathrm { s y m } } ( \mu ^ { \prime } , \mu ) \le 4 \operatorname* { m i n } \left\{ D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } ^ { \prime } , \pmb { \mu } ) , D _ { \widetilde { \psi } _ { m } } ( \pmb { \mu } , \pmb { \mu } ^ { \prime } ) \right\} .\tag{A.7}
$$

Proof. Fix an integer $m \geq 2$ and $\pmb \theta \in \mathbb { R } ^ { m }$ , and write

$$
Q _ { m } ( \pmb \theta ) = ( \lambda , \lambda \pmb x ) .
$$

By Lemma 3.1, the maximizer defining $Q _ { m } ( \pmb \theta )$ is unique and belongs to relint $\widetilde { \Delta } ^ { m }$ . The Hessian in (B.4) is positive definite, so the implicit-function theorem makes $Q _ { m }$ and ${ \pmb x } _ { m }$ smooth.

$$
\begin{array} { r l } & { B : = \mathrm { d i a g } ( \pmb { x } ) - \pmb { x } \pmb { x } ^ { \top } , \qquad \imath : = \left( \log \pmb { x } ( a ) - \psi _ { m } ( \pmb { x } ) \right) _ { a \in [ m ] } , } \\ & { \pmb { v } : = \left( \pmb { x } ( a ) \imath ( a ) \right) _ { a \in [ m ] } , \qquad \pmb { w } : = \pmb { x } + \frac { 1 } { 2 } \pmb { v } , } \\ & { \beta : = 2 \Gamma _ { m } + 3 - \psi _ { m } ( \pmb { x } ) - W _ { m } ( \pmb { x } ) , \qquad \omega _ { \lambda } : = \left( \frac { \lambda } { 1 - \lambda } \right) ^ { 3 / 2 } . } \end{array}
$$

For $e \in \mathbb { R } ^ { m }$ , write $\begin{array} { r } { \frac { \textrm { d } } { \textrm { d } r } Q _ { m } ( \pmb { \theta } + r \pmb { e } ) \Big | _ { r = 0 } = ( s , s \pmb { x } + \lambda z ) } \end{array}$ . The linearized first-order condition gives, for every $\pmb { \xi } \in \mathbb { R } ^ { m }$ satisfying $\mathbf { 1 } ^ { \top } \pmb { \xi } = 0$

$$
\begin{array} { r } { \sqrt { \lambda } \displaystyle \sum _ { a } \frac { [ z ( a ) - ( s / ( 2 \lambda ) ) v ( a ) ] \xi ( a ) } { x ( a ) } = \lambda \left. e , \xi \right. , } \\ { \displaystyle \frac { \beta + \omega _ { \lambda } } { 4 \lambda ^ { 3 / 2 } } s = \left. e , w \right. . } \end{array}
$$

Therefore,

$$
s = \frac { 4 \lambda ^ { 3 / 2 } } { \beta + \omega _ { \lambda } } \left. e , { \pmb w } \right. , \qquad { \pmb z } = \sqrt { \lambda } { \pmb B } e + \frac { s } { 2 \lambda } { \pmb v } .
$$

Consequently, $J _ { m } ^ { x } ( \pmb \theta )$ and $J _ { y _ { m } } ( \pmb { \theta } )$ are

$$
\begin{array} { c } { { J _ { m } ^ { x } ( \pmb { \theta } ) = \sqrt \lambda \left[ \pmb { B } + \frac { 2 } { \beta + \omega _ { \lambda } } \pmb { v } \pmb { w } ^ { \top } \right] , } } \\ { { J _ { y _ { m } } ( \pmb { \theta } ) = \lambda ^ { 3 / 2 } \left[ \pmb { B } + \frac { 4 } { \beta + \omega _ { \lambda } } \pmb { w } \pmb { w } ^ { \top } \right] . } } \end{array}\tag{B.7}
$$

By (B.1) and (B.5),

$$
\| v \| _ { 1 } \leq \sqrt { \Gamma _ { m } } , \qquad \| w \| _ { 1 } \leq 1 + \frac 1 2 \sqrt { \Gamma _ { m } } , \qquad \| B \| _ { \infty \to 1 } \leq 2 .
$$

Consequently,

$$
\| J _ { m } ^ { x } ( \pmb { \theta } ) \| _ { \infty  1 } \leq 5 \sqrt { \lambda } ,\tag{B.8}
$$

$$
\boldsymbol { e } ^ { \top } J _ { y _ { m } } ( \pmb { \theta } ) \boldsymbol { e } \leq 1 0 \lambda ^ { 3 / 2 } \left\| \boldsymbol { e } \right\| _ { \infty } ^ { 2 } ,\tag{B.9}
$$

$$
\frac { | s | } { \lambda } \leq 6 \left\| e \right\| _ { \infty } .\tag{B.10}
$$

(B.8) proves (A.2); integrating (B.10) along $\pmb \theta + t e , t \in [ 0 , 1 ]$ , proves (A.4).

Let ϑ : $[ 0 , 1 ] \to \mathbb { R } ^ { m }$ be differentiable and set

$$
\pmb \mu ( r ) : = Q _ { m } ( \pmb \vartheta ( r ) ) = ( \lambda ( r ) , \lambda ( r ) \pmb x ( r ) ) .
$$

Fix $r \in [ 0 , 1 ]$ , suppress the argument $r ,$ and set

$$
s : = \dot { \lambda } , \qquad \tau : = s / \lambda , \qquad \mathcal { E } : = \sum _ { a } \frac { \dot { x } ( a ) ^ { 2 } } { x ( a ) } ,
$$

$$
\mathcal { H } : = \dot { \pmb { \mu } } ^ { \top } \nabla ^ { 2 } \widetilde { \psi } _ { m } ( \pmb { \mu } ) \dot { \pmb { \mu } } .
$$

Differentiating $B , \psi _ { m } , v , W _ { m } ,$ , and $\beta$ with respect to r gives

$$
\begin{array} { r l r } { \displaystyle } & { \dot { B } = \mathrm { d i a g } ( \dot { x } ) - \dot { x } x ^ { \top } - x \dot { x } ^ { \top } , } & \\ { \displaystyle } & { \dot { \psi } _ { m } = \sum _ { a } \iota ( a ) \dot { x } ( a ) , } & \\ { \displaystyle } & { \dot { v } ( a ) = \dot { x } ( a ) ( 1 + \iota ( a ) ) - x ( a ) \dot { \psi } _ { m } , \qquad a \in [ m ] , } & \\ { \dot { W } _ { m } = \sum _ { a } \dot { x } ( a ) ( \iota ( a ) ^ { 2 } + 2 \iota ( a ) ) , } & { \dot { \beta } = - \dot { \psi } _ { m } - \dot { W } _ { m } . } & \end{array}
$$

Weighted Cauchy–Schwarz and (B.1) and (B.2) yield

$$
\| \dot { \boldsymbol { B } } \| _ { \infty \to 1 } \le 3 \sqrt { \mathcal { E } } , \qquad \| \dot { \boldsymbol { v } } \| _ { 1 } + \| \dot { \boldsymbol { w } } \| _ { 1 } \le 6 \sqrt { 1 + \Gamma _ { m } } \sqrt { \mathcal { E } } , \qquad \big | \dot { \boldsymbol { \beta } } \big | \le 1 3 ( 1 + \Gamma _ { m } ) \sqrt { \mathcal { E } } .\tag{B.11}
$$

The two scalar estimates

$$
\frac { \sqrt { \lambda } \omega _ { \lambda } \big ( \lambda ^ { - 1 } + ( 1 - \lambda ) ^ { - 1 } \big ) } { ( \beta + \omega _ { \lambda } ) ^ { 2 } } \leq 3 \sqrt { \lambda ^ { - 3 / 2 } + ( 1 - \lambda ) ^ { - 3 / 2 } } ,\tag{B.12}
$$

$$
\frac { \lambda ^ { 3 / 2 } \omega _ { \lambda } \big ( \lambda ^ { - 1 } + ( 1 - \lambda ) ^ { - 1 } \big ) } { ( \beta + \omega _ { \lambda } ) ^ { 2 } } \leq \sqrt { 2 }\tag{B.13}
$$

follow by splitting at $\lambda = 1 / 2$

The derivative of (B.7) is

$$
\pmb { j } _ { m } ^ { x } = \frac { 1 } { 2 } \pmb { \tau } \pmb { J } _ { m } ^ { x } + \sqrt { \lambda } \left[ \dot { \pmb { B } } + \frac { 2 ( \dot { \pmb { v } } \pmb { w } ^ { \top } + \pmb { v } \dot { \pmb { w } } ^ { \top } ) } { \beta + \omega _ { \lambda } } - \frac { 2 \pmb { v } \pmb { w } ^ { \top } ( \dot { \beta } + \dot { \omega } _ { \lambda } ) } { ( \beta + \omega _ { \lambda } ) ^ { 2 } } \right] ,
$$

$$
\dot { \omega } _ { \lambda } = \frac { 3 } { 2 } \omega _ { \lambda } s \left( \lambda ^ { - 1 } + ( 1 - \lambda ) ^ { - 1 } \right) .
$$

Substituting (B.8), (B.11) and (B.12) and the bounds on $v , w , \beta$ gives

$$
\begin{array} { r } { \| \dot { \pmb J _ { m } ^ { x } } \| _ { \infty  1 } \leq 2 0 0 ( 1 + \Gamma _ { m } ) [ \sqrt { \lambda } \sqrt { \pmb { \mathcal { E } } } + \sqrt { \lambda } | \tau | + | s | \sqrt { \lambda ^ { - 3 / 2 } + ( 1 - \lambda ) ^ { - 3 / 2 } } ] . } \end{array}
$$

By (B.6),

$$
\sqrt { \lambda } \sqrt { \mathcal { E } } \leq \sqrt { 2 } \sqrt { \mathcal { H } } , \qquad \sqrt { \lambda } \left| \tau \right| \leq \frac { 2 } { \sqrt { 3 } } \sqrt { \mathcal { H } } , \qquad \left| s \right| \sqrt { \lambda ^ { - 3 / 2 } + ( 1 - \lambda ) ^ { - 3 / 2 } } \leq 2 \sqrt { \mathcal { H } } .
$$

Hence

$$
\begin{array} { r } { \| \pmb { j } _ { m } ^ { x } \| _ { \infty  1 } \leq 1 0 ^ { 3 } ( 1 + \Gamma _ { m } ) \sqrt { \mathcal { H } } \leq \Xi _ { m } \sqrt { \mathcal { H } } . } \end{array}\tag{B.14}
$$

For an arbitrary $e \in \mathbb { R } ^ { m }$ , specialize the preceding path to $\pmb { \vartheta } ( r ) = \pmb { \theta } + r \pmb { e } , r \in [ 0 , 1 ]$ $\mathrm { A t } \ r = 0 ,$ , the linearized response optimality condition gives

$$
\mathcal { H } = \left. e , \frac { \mathrm { d } } { \mathrm { d } r } \pmb { y } _ { m } ( \pmb { \theta } + r e ) \bigg | _ { r = 0 } \right. .
$$

Applying (B.14) at $r = 0$ proves (A.3).

Fix $e \in \mathbb { R } ^ { m }$ . Along $Q _ { m } ( \pmb { \theta } + t e ) , t \in [ 0 , 1 ]$ , let overdots denote derivatives with respect to $t ,$ set $s : = \dot { \lambda }$ and $\tau : = s / \lambda$ , and use $\lambda , x , B , \iota , v , w , \beta ,$ and $\omega _ { \lambda }$ for their values at t. Define

$$
\begin{array} { l } { { \displaystyle e ^ { \circ } : = e - \left. e , x \right. \mathbf { 1 } , \qquad \varpi : = \left. e , { \pmb w } \right. , \qquad } } \\ { { \displaystyle \kappa : = \frac { 4 \lambda ^ { 3 / 2 } } { \beta + \omega _ { \lambda } } , \qquad T : = \lambda ^ { 3 / 2 } \sum _ { a } x ( a ) e ^ { \circ } ( a ) ^ { 2 } , \qquad } } \\ { { \displaystyle \mathcal { S } : = \kappa \varpi ^ { 2 } . } } \end{array}
$$

Applying the formulas for s and z above with $z = { \dot { x } }$ at $Q _ { m } ( \pmb { \theta } + t \pmb { e } )$ gives, for every $a \in [ m ]$

$$
\frac { \dot { x } ( a ) } { x ( a ) } = \sqrt { \lambda } e ^ { \circ } ( a ) + \frac { 1 } { 2 } \tau \iota ( a ) , \qquad s = \kappa \varpi , \qquad e ^ { \top } J _ { y _ { m } } ( \theta + t e ) e = \mathcal { T } + \mathcal { S } .
$$

Differentiating ${ \mathcal { T } } , \beta , \varpi , \kappa ,$ and S with respect to t gives

(B.15)

$$
\begin{array} { l } { \displaystyle \mathcal { \dot { T } } = \frac { 3 } { 2 } \tau \mathcal { T } + \lambda ^ { 2 } \sum _ { a } x ( a ) e ^ { \circ } ( a ) ^ { 3 } + \frac { 1 } { 2 } \tau \lambda ^ { 3 / 2 } \sum _ { a } x ( a ) e ^ { \circ } ( a ) ^ { 2 } \iota ( a ) , } \\ { \displaystyle \dot { \beta } = - \sum _ { a } x ( a ) ( \iota ( a ) ^ { 2 } + 3 \iota ( a ) ) \left( \sqrt { \lambda } e ^ { \circ } ( a ) + \frac { 1 } { 2 } \tau \iota ( a ) \right) , } \\ { \displaystyle \dot { \omega } = \frac { 3 } { 2 } \sum _ { a } x ( a ) e ^ { \circ } ( a ) \left( \sqrt { \lambda } e ^ { \circ } ( a ) + \frac { 1 } { 2 } \tau \iota ( a ) \right) + \frac { 1 } { 2 } \sum _ { a } x ( a ) e ^ { \circ } ( a ) \iota ( a ) \left( \sqrt { \lambda } e ^ { \circ } ( a ) + \frac { 1 } { 2 } \tau \iota ( a ) \right) , } \\ { \displaystyle \frac { \dot { \kappa } } { \kappa } = \frac { 3 } { 2 } \tau - \frac { \dot { \beta } + \dot { \omega } _ { \lambda } } { \beta + \omega _ { \lambda } } , \qquad \dot { S } = \frac { \dot { \kappa } } { \kappa } S + 2 \kappa \varpi \dot { \omega } . } \end{array}\tag{B.16}
$$

Using $| e ^ { \circ } ( a ) | \leq 2 \| e \| _ { \infty }$ for every $a \in [ m ]$ , (B.1), (B.2) and (B.13), and weighted Cauchy–Schwarz gives

$$
\begin{array} { c } { { \displaystyle \left| \tau \right| \leq 6 \left\| e \right\| _ { \infty } , \qquad \frac { \left| \dot { \kappa } \right| } { \kappa } \leq 1 3 0 \sqrt { 1 + \Gamma _ { m } } \left\| e \right\| _ { \infty } , } } \\ { { \sqrt { \kappa } \left| \dot { \varpi } \right| \leq 8 \left\| e \right\| _ { \infty } \left( \sqrt { \mathscr { T } } + \sqrt { \mathscr { S } } \right) . } } \end{array}
$$

Substitution in (B.15) and (B.16) gives

$$
\begin{array} { r l } & { | \dot { T } | \leq 1 2 \left\| e \right\| _ { \infty } ( T + \mathcal { S } ) , } \\ & { | \dot { S } | \leq 1 6 0 ( 1 + \Gamma _ { m } ) \left\| e \right\| _ { \infty } ( T + \mathcal { S } ) . } \end{array}
$$

Therefore, whenever $\mathcal { T } + \mathcal { S } > 0 .$

$$
\left| \frac { \mathrm { d } } { \mathrm { d } t } \log ( \mathcal { T } + \mathcal { S } ) \right| \leq 2 0 0 ( 1 + \Gamma _ { m } ) \left\| e \right\| _ { \infty } \leq \Xi _ { m } \left\| e \right\| _ { \infty } .\tag{B.17}
$$

Suppose $\Xi _ { m } \left\| e \right\| _ { \infty } \leq 1 / 1 0 0 _ { \ L }$ , and set

$$
\mu _ { t } : = Q _ { m } ( \pmb { \theta } + t e ) , \qquad t \in [ 0 , 1 ] , \qquad \mathcal { H } ( t ) : = e ^ { \top } J _ { y _ { m } } ( \pmb { \theta } + t e ) e .
$$

If $e \neq { \mathbf { 0 } } ,$ , positive definiteness and (B.17) imply $\begin{array} { r } { \operatorname* { m a x } _ { t \in [ 0 , 1 ] } \mathcal { H } ( t ) \leq 2 \operatorname* { m i n } _ { t \in [ 0 , 1 ] } \mathcal { H } ( t ) } \end{array}$ . Fenchel duality and Taylor’s integral formula give

$$
\begin{array} { r l } { D _ { \widetilde { \psi } _ { m } } ( \mu _ { 1 } , \mu _ { 0 } ) = \displaystyle \int _ { 0 } ^ { 1 } t \mathcal { H } ( t ) \mathrm { d } t , } & { \quad D _ { \widetilde { \psi } _ { m } } ( \mu _ { 0 } , \mu _ { 1 } ) = \displaystyle \int _ { 0 } ^ { 1 } ( 1 - t ) \mathcal { H } ( t ) \mathrm { d } t , } \\ { D _ { \widetilde { \psi } _ { m } } ^ { \mathrm { s y m } } ( \mu _ { 1 } , \mu _ { 0 } ) = \displaystyle \int _ { 0 } ^ { 1 } \mathcal { H } ( t ) \mathrm { d } t . } \end{array}
$$

Thus each directed divergence is at least one quarter of the symmetric divergence, proving (A.7). By (B.10), $\lambda _ { m } ( \pmb \theta + t e ) \leq 2 \lambda _ { m } ( \pmb \theta + t ^ { \prime } e )$ for every $t , t ^ { \prime } \in [ 0 , 1 ]$ . Integrating (B.9) therefore gives

$$
\begin{array} { r } { \langle e , y _ { m } ( \pmb { \theta } + e ) - y _ { m } ( \pmb { \theta } ) \rangle \leq 2 0 \sqrt { 2 } \hat { \lambda } ^ { 3 / 2 } \left. \pmb { e } \right. _ { \infty } ^ { 2 } , } \end{array}
$$

which is stronger than (A.5).

Finally, write ${ \pmb \mu } _ { \ell } = ( \lambda _ { \ell } , \lambda _ { \ell } { \pmb x } _ { \ell } )$ for $\ell \in \{ 0 , 1 \}$ . Direct differentiation gives

$$
\nu ( t ) = ( 1 - t ) \mu _ { 0 } + t \mu _ { 1 } = ( \lambda ( t ) , \lambda ( t ) x ( t ) ) , \qquad t \in [ 0 , 1 ] , \qquad \dot { x } ( t ) = \frac { \lambda _ { 0 } \lambda _ { 1 } } { \lambda ( t ) ^ { 2 } } \left( x _ { 1 } - x _ { 0 } \right) .
$$

Since $\lambda _ { \ell } \leq 2 \lambda _ { 1 - \ell }$ for $\ell \in \{ 0 , 1 \} , \lambda ( t ) \geq { \widehat { \lambda } } / 2$ and $\lambda _ { 0 } \lambda _ { 1 } / \lambda ( t ) ^ { 2 } \geq 1 / 2$ . Hence (B.6) and weighted Cauchy–Schwarz give

$$
\pmb { \dot { \nu } } ( t ) ^ { \top } \nabla ^ { 2 } \widetilde { \psi } _ { m } ( \pmb { \nu } ( t ) ) \pmb { \dot { \nu } } ( t ) \geq \frac { 1 } { 8 \sqrt { 2 } } \sqrt { \hat { \lambda } } \left. \pmb { x } _ { 1 } - \pmb { x } _ { 0 } \right. _ { 1 } ^ { 2 } .
$$

Taylor’s integral formula in both orientations yields

$$
\operatorname* { m i n } \Big \{ D _ { \widetilde { \psi } _ { m } } ( \mu _ { 1 } , \mu _ { 0 } ) , D _ { \widetilde { \psi } _ { m } } ( \mu _ { 0 } , \mu _ { 1 } ) \Big \} \geq \frac { 1 } { 1 6 \sqrt { 2 } } \sqrt { \widehat { \lambda } } \| \pmb { x } _ { 1 } - \pmb { x } _ { 0 } \| _ { 1 } ^ { 2 } .
$$

Since $\Xi _ { m } \geq 1$ , this proves (A.6).

## C EMA estimates for multilinear games

Use $\rho$ and $\gamma$ from (A.1) and (3.2).

## C.1 EMA kernels and weighted convolution

Lemma A.2. Set $\begin{array} { r } { \mathsf { K } _ { \star } : = \mathsf { A } - \delta ^ { N + 1 } } \end{array}$ , and, for every integer $1 \leq h \leq N + 1$ , set $\mathsf { K } _ { h } : = \mathsf { A } \delta ^ { h - 1 }$ . For every integer $1 \leq h \leq N + 1$ , the filters in (3.2) satisfy

$$
\delta = \mathsf { A D } , \qquad \mathsf { K } _ { h } \mathsf { D } = \delta ^ { h } ,\tag{A.8}
$$

$$
{ \sf K } _ { h } \big ( { I } - { \mathsf { D } } \delta ^ { N } \big ) = { \sf K } _ { \star } \delta ^ { h - 1 } .\tag{A.9}
$$

For every integer $1 \leq h \leq N + 1 , \mathsf { K } \in \{ \delta ^ { h } , \mathsf { K } _ { h } , \mathsf { K } _ { \star } \} , a n d \mathbb { O } \leq \epsilon \leq 1 / ( 8 N ) , i f \mathsf { k e r } ( \mathsf { K } ) = ( k _ { s } ) _ { s \geq 0 } .$ , then

$$
\sum _ { s \geq 0 } e ^ { \epsilon s / 2 } ( s + 1 ) ^ { 4 } \left. k _ { s } \right. \leq 2 ^ { 2 8 } N ^ { 6 } .\tag{A.10}
$$

For every integer $1 \leq h \leq N + 1$ and $\mathsf { K } \in \mathsf { \{ \delta ^ { h } , K _ { h } , K _ { \star } \} }$ , the $\ell _ { 1 }$ norm and the first four absolute moments of ker(K) are at most $2 ^ { 2 8 } N ^ { 6 }$ . For every integer $1 \leq h \leq N + 1$ , the coefficients of $\delta ^ { h }$ satisfy $\begin{array} { r } { \sum _ { s > 0 } \ker ( \delta ^ { h } ) _ { s } = 0 } \end{array}$

For every integer $1 \leq h \leq N + 1$ and ${ \sf K } \in \left\{ \delta ^ { h } , { \sf K } _ { h } , { \sf K } _ { \star } \right\}$ , consider in any normed space two inputs to K that agree at positive times and are equal, respectively, to a fixed vector φ and 0 at every nonpositive time. If their output difference is $\tau ,$ then

$$
\sum _ { t = 1 } ^ { \infty } \left\| \tau ^ { ( t ) } \right\| ^ { 2 } \leq 2 ^ { 5 9 } N ^ { 1 3 } \left\| \varphi \right\| ^ { 2 } .\tag{A.11}
$$

Proof. The transfer functions are

$$
\mathsf { A } ( \varsigma ) = \frac { 1 } { 1 - \rho \varsigma } , \qquad \delta ( \varsigma ) = \frac { 1 - \varsigma } { 1 - \rho \varsigma } , \qquad \mathsf { K } _ { h } ( \varsigma ) = \frac { ( 1 - \varsigma ) ^ { h - 1 } } { ( 1 - \rho \varsigma ) ^ { h } } .
$$

Thus $\delta = \mathsf { A D }$ and ${ \sf K } _ { h } { \sf D } = \delta ^ { h }$ . Since these operators commute,

$$
\mathsf { K } _ { h } ( I - \mathsf { D } \delta ^ { N } ) = \mathsf { A } \delta ^ { h - 1 } - \delta ^ { N + h } = \left( \mathsf { A } - \delta ^ { N + 1 } \right) \delta ^ { h - 1 } = \mathsf { K } _ { \star } \delta ^ { h - 1 } .
$$

Set $R = 1 + 1 / ( 4 N )$ . Then $\rho R < 1$ and

$$
\rho R ^ { 2 } = 1 - \frac { 1 } { 2 N } - \frac { 7 } { 1 6 N ^ { 2 } } - \frac { 1 } { 1 6 N ^ { 3 } } < 1 .
$$

For $\vartheta \in \mathbb { R }$ and $\varsigma = R e ^ { \mathrm { i } \vartheta }$

$$
\lvert \delta ( \varsigma ) \rvert ^ { 2 } = \frac { 1 + R ^ { 2 } - 2 R \cos \vartheta } { 1 + \rho ^ { 2 } R ^ { 2 } - 2 \rho R \cos \vartheta } ,
$$

$$
\frac { \mathrm { d } } { \mathrm { d } ( \cos \vartheta ) } \left| \delta ( \varsigma ) \right| ^ { 2 } = \frac { 2 R ( 1 - \rho ) \bigl ( \rho R ^ { 2 } - 1 \bigr ) } { \bigl ( 1 + \rho ^ { 2 } R ^ { 2 } - 2 \rho R \cos \vartheta \bigr ) ^ { 2 } } \leq 0 .
$$

The maximum on $| \varsigma | = R$ is therefore attained at $\varsigma = - R$ . Since $R ( 4 - 3 \rho ) \leq 3 { \mathrm { f o r } } N \geq 2 ,$

$$
\operatorname* { s u p } _ { | \varsigma | = R } | \delta ( \varsigma ) | = \frac { 1 + R } { 1 + \rho R } = 1 + \frac { ( 1 - \rho ) R } { 1 + \rho R } \leq 1 + \frac { 3 } { 4 N } .
$$

For $h \le N + 1 \le 3 N / 2$ , this gives

$$
\displaystyle { \operatorname* { s u p } _ { | \varsigma | = R } \left| \delta ( \varsigma ) \right| ^ { h } \le e ^ { 9 / 8 } < 4 . }
$$

Moreover, $1 - \rho R \geq 3 / ( 4 N )$ , and hence

$$
\underset { | \varsigma | = R } { \operatorname* { s u p } } | \mathsf { K } _ { h } ( \varsigma ) | \le \frac { 4 } { 1 - \rho R } \le 6 N ,
$$

$$
\operatorname* { s u p } _ { | \varsigma | = R } | \mathsf { K } _ { \star } ( \varsigma ) | \le \frac { 1 } { 1 - \rho R } + 4 \le 6 N .
$$

The same upper bound 6N applies to $\delta ^ { h } .$ . Cauchy’s coefficient estimate therefore gives

$$
\vert k _ { s } \vert \le 6 N R ^ { - s } , \qquad s \ge 0 .
$$

Fix $0 \le \epsilon \le 1 / ( 8 N )$ . The inequality lo $\mathsf { y } ( 1 + y ) \ge y / ( 1 + y )$ for $y \geq 0$ gives

$$
\log R - \frac { \epsilon } { 2 } \geq \frac { 1 } { 4 N + 1 } - \frac { 1 } { 1 6 N } \geq \frac { 1 } { 8 N } .
$$

Thus $e ^ { \epsilon / 2 } / R \le e ^ { - 1 / ( 8 N ) }$ . Since $1 - e ^ { - y } \ge y / 2$ for $0 \leq y \leq 1$ , we have $1 - e ^ { \epsilon / 2 } / R \geq 1 / ( 1 6 N )$ Consequently,

$$
\begin{array} { r l r } {  { \sum _ { s \geq 0 } e ^ { \epsilon s / 2 } ( s + 1 ) ^ { 4 } | k _ { s } | \leq 6 N \sum _ { s \geq 0 } ( s + 1 ) ^ { 4 } ( \frac { e ^ { \epsilon / 2 } } { R } ) ^ { s } } } \\ & { } & { \leq \frac { 1 4 4 N } { ( 1 - e ^ { \epsilon / 2 } / R ) ^ { 5 } } \leq 1 4 4 \cdot 1 6 ^ { 5 } N ^ { 6 } \leq 2 ^ { 2 8 } N ^ { 6 } , } \end{array}
$$

where, for every $q \ \in \ [ 0 , 1 ) , \ \sum _ { s > 0 } ( s + 1 ) ^ { 4 } q ^ { s } \ = \ ( 1 + 1 1 q + 1 1 q ^ { 2 } + q ^ { 3 } ) / ( 1 - q ) ^ { 5 }$ . The absolute convergence permits evaluation at $\varsigma = 1$ , so the coefficients of $\delta ^ { h }$ sum to $\delta ( 1 ) ^ { h } = 0$

For the last claim, the output difference at time $t \geq 1$ is

$$
\tau ^ { ( t ) } = \varphi \sum _ { s \geq t } k _ { s } .
$$

Apply (A.10) with $\epsilon = 1 / ( 8 N )$ . For every $t \geq 1$

$$
\left\| { \tau ^ { ( t ) } } \right\| \le e ^ { - t / ( 1 6 N ) } \left\| \varphi \right\| \sum _ { s \ge 0 } e ^ { s / ( 1 6 N ) } \left| k _ { s } \right| \le 2 ^ { 2 8 } N ^ { 6 } e ^ { - t / ( 1 6 N ) } \left\| \varphi \right\| .
$$

Since $e ^ { 1 / ( 8 N ) } - 1 \ge 1 / ( 8 N )$ , summing the squares gives

$$
\sum _ { t = 1 } ^ { \infty } \left\| \pmb { \tau } ^ { ( t ) } \right\| ^ { 2 } \leq \frac { 2 ^ { 5 6 } N ^ { 1 2 } } { e ^ { 1 / ( 8 N ) } - 1 } \left\| \varphi \right\| ^ { 2 } \leq 2 ^ { 5 9 } N ^ { 1 3 } \left\| \varphi \right\| ^ { 2 } .
$$

Lemma A.3. Fix $T \geq 1$ and $\epsilon \geq 0$ . Let z take values in a normed space, with $z ^ { ( t ) } = 0$ for $t \leq 0 ,$ , and let $\begin{array} { r } { \mathsf { K } = \sum _ { s > 0 } k _ { s } \mathsf { L } ^ { s } } \end{array}$ satisfy $\begin{array} { r } { \sum _ { s \ge 0 } e ^ { \epsilon s / 2 } \left| k _ { s } \right| < \infty } \end{array}$ . Suppose positive weights $w ^ { ( t ) }$ are defined for every integer $t \leq T$ and satisfy

$$
w ^ { ( t ) } \leq e ^ { \epsilon s } w ^ { ( t - s ) } \qquad ( t \in [ T ] , s \geq 0 ) .
$$

Then

$$
\sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| ( \mathsf { K } z ) ^ { ( t ) } \right\| ^ { 2 } \leq \left( \sum _ { s \geq 0 } e ^ { \epsilon s / 2 } \left| k _ { s } \right| \right) ^ { 2 } \sum _ { t = 1 } ^ { T } w ^ { ( t ) } \left\| z ^ { ( t ) } \right\| ^ { 2 } .
$$

Proof. The hypothesis on w gives

$$
\sqrt { w ^ { ( t ) } } \left\| ( { \mathsf K } z ) ^ { ( t ) } \right\| \le \sum _ { s \ge 0 } e ^ { \epsilon s / 2 } \left| k _ { s } \right| \sqrt { w ^ { ( t - s ) } } \left\| z ^ { ( t - s ) } \right\| .
$$

Extend the scalar sequence on the right by zero outside [T]. The $\ell _ { 1 } * \ell _ { 2 } \to \ell _ { 2 }$ Young inequality gives the conclusion. □

## C.2 Score and response movement

Lemma A.4. For every player $i \in [ N ]$ and time $t \geq 1$

$$
\begin{array} { r } { \left\| e _ { i } ^ { ( t ) } \right\| _ { \infty } \leq 3 \cdot 2 ^ { 2 8 } N ^ { 6 } , \qquad \left\| \left( \left( I - \mathsf { D } \delta ^ { N } \right) g _ { i } \right) ^ { ( t ) } \right\| _ { \infty } \leq 3 \cdot 2 ^ { 2 8 } N ^ { 6 } . } \end{array}\tag{A.12}
$$

$I f 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ , then, for every player $i \in [ N ]$ and time $t \geq 1$ , the two scores in (3.5) at times t and $t + 1$ satisfy the hypothesis of Lemma $A . 1 ,$ , and

$$
\left| \log \frac { \lambda _ { i } ^ { ( t + 1 ) } } { \lambda _ { i } ^ { ( t ) } } \right| \le 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } .\tag{A.13}
$$

Proof. For every $i \in [ N ]$ and $t \geq 1 , \left\| g _ { i } ^ { ( t ) } \right\| _ { \infty } \leq 1$ , and $\pmb { g } _ { i } ^ { ( s ) } = \pmb { 0 }$ for every integer $s \leq 0$ . Since $e _ { i } = \delta ^ { N } g _ { i }$ , Lemma $\mathrm { A } . 2$ gives the first bound. Moreover,

$$
\begin{array} { r } { \left\| \left( \left( I - \mathsf { D } \delta ^ { N } \right) \pmb { g } _ { i } \right) ^ { ( t ) } \right\| _ { \infty } = \left\| \widehat { \pmb { g } } _ { i } ^ { ( t ) } + \pmb { e } _ { i } ^ { ( t - 1 ) } \right\| _ { \infty } \leq 1 + 2 ^ { 2 9 } N ^ { 6 } \leq 3 \cdot 2 ^ { 2 8 } N ^ { 6 } . } \end{array}
$$

Fix $i \in [ N ]$ and $t \geq 1$ . Then

$$
\eta ^ { - 1 } \left( \pmb { \theta } _ { i } ^ { ( t + 1 ) } - \pmb { \theta } _ { i } ^ { ( t ) } \right) = \sum _ { s = 1 } ^ { t } \pmb { g } _ { i } ^ { ( s ) } + \pmb { \widehat { g } } _ { i } ^ { ( t + 1 ) } - \sum _ { s = 1 } ^ { t - 1 } \pmb { g } _ { i } ^ { ( s ) } - \pmb { \widehat { g } } _ { i } ^ { ( t ) }
$$

$$
\begin{array} { r l } & { = \left( ( I - \mathsf { D } \delta ^ { N } ) \pmb { g } _ { i } \right) ^ { ( t + 1 ) } } \\ & { = \widehat { \pmb { g } } _ { i } ^ { ( t + 1 ) } + \pmb { e } _ { i } ^ { ( t ) } . } \end{array}
$$

Applying the preceding bound at time t + 1 and multiplying by $\eta \Xi _ { i }$ verifies the hypothesis of Lemma A.1. Applying (A.4) proves (A.13). □

Lemma A.5. Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i \in [ N ]$ and $t \geq 1$

$$
\begin{array} { r } { \left. \overline { { \boldsymbol { J } } } _ { i } ^ { ( t ) } \right. _ { \infty \to 1 } \le 4 \sqrt { 2 } \eta \Xi _ { i } \sqrt { \lambda _ { i } ^ { ( t ) } } \le 4 \sqrt { 2 } \gamma \sqrt { \lambda _ { i } ^ { ( t ) } } , } \end{array}\tag{A.15}
$$

$$
\sqrt { \lambda _ { i } ^ { ( t ) } } \left. \left( \mathsf { D } \pmb { x } _ { i } \right) ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 1 6 \sqrt { 2 } \Xi _ { i } D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) ,\tag{A.16}
$$

$$
\begin{array} { r } { \left\| { ( \mathsf { D } \pmb { x } _ { i } ) } ^ { ( t ) } \right\| _ { 1 } \le 3 \sqrt { 2 } \cdot 2 ^ { 3 0 } \eta \Xi _ { i } N ^ { 6 } \sqrt { \lambda _ { i } ^ { ( t ) } } \le 3 \sqrt { 2 } \cdot 2 ^ { 3 0 } \gamma N ^ { 6 } \sqrt { \lambda _ { i } ^ { ( t ) } } . } \end{array}\tag{A.17}
$$

For every $i \in [ N ]$ and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \left( \mathsf { D } \overline { { \mathbf { J } } } _ { i } \right) ^ { ( t ) } \right. _ { \infty \to 1 } ^ { 2 } \leq 1 6 \eta ^ { 2 } \Xi _ { i } ^ { 2 } \mathcal { P } _ { i , T } .\tag{A.18}
$$

Proof. Fix $i \in [ N ]$ and $t \geq 1$ . For every $\alpha \in [ 0 , 1 ]$

$$
\pmb { \theta } _ { i } ^ { ( t - 1 ) } + \alpha \eta \pmb { d } _ { i } ^ { ( t ) } - \pmb { \theta } _ { i } ^ { ( t ) } = - ( 1 - \alpha ) \eta \pmb { d } _ { i } ^ { ( t ) } ,
$$

and (A.1) and (A.12) give

$$
\Xi _ { i } \left\| ( 1 - \alpha ) \eta \pmb { d } _ { i } ^ { ( t ) } \right\| _ { \infty } \leq \left( 1 - \alpha \right) 3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq \frac { 1 } { 1 0 0 } .
$$

Therefore, Lemma A.1 and (A.4) gives, for every $\alpha \in [ 0 , 1 ]$

$$
\lambda _ { m _ { i } } \left( \pmb \theta _ { i } ^ { ( t - 1 ) } + \alpha \eta \pmb d _ { i } ^ { ( t ) } \right) \le e ^ { 1 / 1 0 0 } \lambda _ { i } ^ { ( t ) } \le 2 \lambda _ { i } ^ { ( t ) } .
$$

Integrating (A.2) over $\alpha \in [ 0 , 1 ]$ and using $\eta \Xi _ { i } \leq \gamma$ proves (A.15).

Since $\pmb { \theta } _ { i } ^ { ( t ) } - \pmb { \theta } _ { i } ^ { ( t - 1 ) } = \eta \pmb { d } _ { i } ^ { ( t ) }$ , (A.6) gives

$$
\sqrt { \lambda _ { i } ^ { ( t ) } } \left\| \left( \mathsf { D } \pmb { x } _ { i } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 2 } \leq 1 6 \sqrt { 2 } \Xi _ { i } D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) .
$$

This is (A.16). The identity in (A.14), (A.15), and (A.12) give

$$
\left\| \left( \mathsf { D } \pmb { x } _ { i } \right) ^ { ( t ) } \right\| _ { 1 } \leq 3 \sqrt { 2 } \cdot 2 ^ { 3 0 } \eta \Xi _ { i } N ^ { 6 } \sqrt { \lambda _ { i } ^ { ( t ) } } ,
$$

which proves (A.17).

Since t was arbitrary, this proves the first three assertions. It remains to prove (A.18). Fix $t \geq 1$ The fundamental theorem of calculus gives

$$
\overline { { \pmb { J } } } _ { i } ^ { ( t ) } - \eta \pmb { J } _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t - 1 ) } \right) = \eta \int _ { 0 } ^ { 1 } ( 1 - \alpha ) \frac { \mathrm { d } } { \mathrm { d } \alpha } \pmb { J } _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t - 1 ) } + \alpha \eta \pmb { d } _ { i } ^ { ( t ) } \right) \mathrm { d } \alpha ,
$$

$$
\eta \boldsymbol { J } _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t ) } \right) - \overline { { \boldsymbol { J } } } _ { i } ^ { ( t ) } = \eta \int _ { 0 } ^ { 1 } \alpha \frac { \mathrm { d } } { \mathrm { d } \alpha } \boldsymbol { J } _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t - 1 ) } + \alpha \eta \boldsymbol { d } _ { i } ^ { ( t ) } \right) \mathrm { d } \alpha .
$$

By (A.3), the Cauchy–Schwarz inequality, and $( \mathrm { A } . 7 )$

$$
\begin{array} { r } { \operatorname* { m a x } \left\{ \left\| \overline { { J } } _ { i } ^ { ( t ) } - \eta J _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t - 1 ) } \right) \right\| _ { \infty \to 1 } ^ { 2 } , \left\| \overline { { J } } _ { i } ^ { ( t ) } - \eta J _ { m _ { i } } ^ { x } \left( \pmb { \theta } _ { i } ^ { ( t ) } \right) \right\| _ { \infty \to 1 } ^ { 2 } \right\} \le \eta ^ { 2 } \Xi _ { i } ^ { 2 } D _ { \widetilde { \psi } _ { i } } ^ { \mathrm { s y m } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) } \\ { \le 4 \eta ^ { 2 } \Xi _ { i } ^ { 2 } D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) . } \end{array}
$$

Since t was arbitrary, the preceding bound holds for every $t \geq 1$ . Since $\overline { { J } } _ { i } ^ { ( 1 ) } = \overline { { J } } _ { i } ^ { ( 0 ) } = \eta J _ { m _ { i } } ^ { x } ( \mathbf { 0 } )$ for every $t \geq 2$ the preceding bound gives

$$
\left\| \left( \mathsf { D } \overline { { \boldsymbol { J } } } _ { i } \right) ^ { ( t ) } \right\| _ { \infty \to 1 } ^ { 2 } \le 8 \eta ^ { 2 } \Xi _ { i } ^ { 2 } \left[ D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t ) } , \pmb { \mu } _ { i } ^ { ( t - 1 ) } \right) + D _ { \widetilde { \psi } _ { i } } \left( \pmb { \mu } _ { i } ^ { ( t - 1 ) } , \pmb { \mu } _ { i } ^ { ( t - 2 ) } \right) \right] .
$$

Fix $T \geq 1$ . Summing over $t \in [ T ]$ proves (A.18).

Lemma A.7. Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ and $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 7 } \leq 1 / 4$ . For every $j \in \ [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \sqrt { \lambda _ { j } ^ { ( t ) } } \left. \left( \delta ^ { h } \pmb { x } _ { j } \right) ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 2 ^ { 6 0 } \sqrt { 2 } N ^ { 1 2 } \Xi _ { j } \mathcal { P } _ { j , T } .
$$

Proof. Extend $\lambda _ { j }$ constantly to all nonpositive integer times. By (A.13), for $t \in [ T ]$ and $s \geq 0$

$$
\left| \log \frac { \sqrt { \lambda _ { j } ^ { ( t ) } } } { \sqrt { \lambda _ { j } ^ { ( t - s ) } } } \right| \le 3 \cdot 2 ^ { 2 7 } \gamma N ^ { 6 } s .
$$

The rate condition gives $3 \cdot 2 ^ { 2 7 } \gamma N ^ { 6 } s \le s / ( 8 N )$ . Since $\delta ^ { h } = \mathsf { K } _ { h } \mathsf { D }$ and $\mathsf { D } \pmb { x } _ { j }$ vanishes at every nonpositive integer time, Lemmas $\mathrm { A } . 2$ and A.3 gives

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \sqrt { \lambda _ { j } ^ { ( t ) } } \| ( \delta ^ { h } { \pmb x } _ { j } ) ^ { ( t ) } \| _ { 1 } ^ { 2 } \leq 2 ^ { 5 6 } N ^ { 1 2 } \sum _ { t = 1 } ^ { T } \sqrt { \lambda _ { j } ^ { ( t ) } } \| ( { \tt D } { \pmb x } _ { j } ) ^ { ( t ) } \| _ { 1 } ^ { 2 } } } \\ & { } & { \leq 2 ^ { 6 0 } \sqrt { 2 } N ^ { 1 2 } { \Xi } _ { j } \mathcal { P } _ { j , T } , } \end{array}
$$

where the last inequality follows by summing (A.16) over $t \in [ T ]$

Lemma A.8. Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i \in [ N ]$ , integer $1 \leq h \leq N .$ , and time $t \geq 1$

$$
\left( \delta ^ { h } \mathbf { x } _ { i } \right) ^ { ( t ) } = \overline { { \pmb { J } } } _ { i } ^ { ( t ) } \left( \mathsf { K } _ { \star } \delta ^ { h - 1 } \pmb { g } _ { i } \right) ^ { ( t ) } + \pmb { c } _ { i , h } ^ { ( t ) } .\tag{A.23}
$$

For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. c _ { i , h } ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 9 \cdot 2 ^ { 1 1 6 } \eta ^ { 2 } \Xi _ { i } ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } \leq 9 \cdot 2 ^ { 1 1 6 } \gamma ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } .\tag{A.24}
$$

Proof. Apply $\mathsf { K } _ { h }$ to (A.14), commute it past ${ \overline { { \mathbf { J } } } } _ { i } ,$ and use (A.8) and (A.9) to obtain (A.23).

Write $k _ { s } = \ker ( \mathsf { K } _ { h } ) _ { s }$ . By the definition of $^ { c _ { i , h . } }$ , for every $t \geq 1$

$$
\begin{array} { r } { c _ { i , h } ^ { \left( t \right) } = \displaystyle \sum _ { s \geq 1 } k _ { s } \left( \overline { { \mathbf { J } } } _ { i } ^ { \left( t - s \right) } - \overline { { \mathbf { J } } } _ { i } ^ { \left( t \right) } \right) \mathbf { d } _ { i } ^ { \left( t - s \right) } } \\ { = - \displaystyle \sum _ { s > 1 } k _ { s } \sum _ { r = 0 } ^ { s - 1 } \left( \mathsf { D } \overline { { \mathbf { J } } } _ { i } \right) ^ { \left( t - r \right) } \mathbf { d } _ { i } ^ { \left( t - s \right) } . } \end{array}
$$

For every $t \in [ T ] , s \geq 1$ , and integer $0 \leq r < s , ( \mathrm { A } . 1 2 )$ gives

$$
\begin{array} { r } { \left\| \left( \mathsf { D } \overline { { \boldsymbol { J } } } _ { i } \right) ^ { ( t - r ) } \pmb { d } _ { i } ^ { ( t - s ) } \right\| _ { 1 } \le 3 \cdot 2 ^ { 2 8 } N ^ { 6 } \left\| \left( \mathsf { D } \overline { { \boldsymbol { J } } } _ { i } \right) ^ { ( t - r ) } \right\| _ { \infty \to 1 } . } \end{array}
$$

Weighted Cauchy–Schwarz over the pairs $\left( s , r \right)$ , followed by the time shifts $\alpha = t - r ,$ gives

$$
\begin{array} { r l } { \displaystyle \sum _ { t = 1 } ^ { T } \left\| c _ { i , h } ^ { ( t ) } \right\| _ { 1 } ^ { 2 } \leq 9 \cdot 2 ^ { 5 6 } N ^ { 1 2 } \left( \displaystyle \sum _ { s \geq 1 } s \left| k _ { s } \right| \right) \displaystyle \sum _ { s \geq 1 } \left| k _ { s } \right| \displaystyle \sum _ { r = 0 } ^ { s - 1 } \displaystyle \sum _ { t = 1 } ^ { T } \left\| \left( \mathsf { D } \overline { { J } } _ { i } \right) ^ { ( t - r ) } \right\| _ { \infty \to 1 } ^ { 2 } } & { } \\ { \leq 9 \cdot 2 ^ { 5 6 } N ^ { 1 2 } \left( \displaystyle \sum _ { s \geq 1 } s \left| k _ { s } \right| \right) ^ { 2 } \displaystyle \sum _ { \alpha = 1 } ^ { T } \left\| \left( \mathsf { D } \overline { { J } } _ { i } \right) ^ { ( \alpha ) } \right\| _ { \infty \to 1 } ^ { 2 } } & { } \\ { \leq 9 \cdot 2 ^ { 1 1 6 } \eta ^ { 2 } \Xi _ { i } ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } . } & { } \end{array}
$$

The second inequality uses $\overline { { J } } _ { i } ^ { ( \tau ) } = \eta J _ { m _ { i } } ^ { x } ( \mathbf { 0 } )$ for every integer $\tau \leq 0 ,$ , and the last uses (A.10) and (A.18). By the definition of $^ { c _ { i , h } , }$ , the preceding display proves the first inequality in (A.24). The inequality $\eta \Xi _ { i } \leq \gamma$ gives

$$
\sum _ { t = 1 } ^ { T } \left. \pmb { c } _ { i , h } ^ { ( t ) } \right. _ { 1 } ^ { 2 } \leq 9 \cdot 2 ^ { 1 1 6 } \gamma ^ { 2 } N ^ { 2 4 } \mathcal { P } _ { i , T } .
$$

This proves the second inequality.

## C.3 Filtered action and payoff differences

Lemma A.9. Let

$$
C _ { \mathrm { m u l t i } } : = 2 \operatorname* { m a x } \left\{ 2 ^ { 5 9 } , 9 \sqrt { 2 } \cdot 2 ^ { 1 2 1 } \right\} .
$$

Suppose $3 \cdot 2 ^ { 2 8 } \gamma N ^ { 6 } \leq 1 / 1 0 0$ . For every $i , j \in [ N ]$ and profile $\textstyle { \pmb { x } } \in \prod _ { k = 1 } ^ { N } \Delta ^ { m _ { k } }$

$$
\operatorname* { s u p } _ { \pmb { \xi } \in \mathbb { R } ^ { m _ { j } } , \ \lVert \pmb { \xi } \rVert _ { 1 } \leq 1 } \left. \partial _ { j } \mathcal { G } _ { i } ( \pmb { x } ) [ \pmb { \xi } ] \right. _ { \infty } \leq 2 .
$$

For every $i \in [ N ]$ and integer $1 \leq h \leq N .$ , there is a sequence $\chi _ { i , h }$ such that, for every $t \geq 1$

$$
\left( \delta ^ { h } \pmb { g } _ { i } \right) ^ { ( t ) } = \sum _ { j = 1 } ^ { N } \partial _ { j } \mathcal { G } _ { i } \left( \pmb { x } ^ { ( t ) } \right) \left( \delta ^ { h } \pmb { x } _ { j } \right) ^ { ( t ) } + \chi _ { i , h } ^ { ( t ) } .\tag{A.26}
$$

For every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \boldsymbol { x } _ { i , h } ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq C _ { \mathrm { m u l t i } } \left( N ^ { 1 3 } + \gamma ^ { 2 } N ^ { 2 7 } \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.27}
$$

$I f \gamma \leq N ^ { - 2 0 }$ , then, for every $i \in [ N ]$ , integer $1 \leq h \leq N ,$ , and horizon $T \geq 1$

$$
\sum _ { t = 1 } ^ { T } \left. \boldsymbol { x } _ { i , h } ^ { ( t ) } \right. _ { \infty } ^ { 2 } \leq C _ { \mathrm { m u l t i } } N ^ { 1 3 } \left( 1 + \mathcal { P } _ { T } ^ { \Xi } \right) .\tag{A.28}
$$

Proof. For every $i \in [ N ]$ , integer $1 \leq q \leq N ,$ , pairwise distinct $j _ { 1 } , \ldots , j _ { q } \in [ N ]$ , profile ${ \textbf { \em x } } \in$ $\Pi _ { k = 1 } ^ { N } \Delta ^ { m _ { k } }$ , and directions $z _ { j _ { \ell } } \in \mathbb { R } ^ { m _ { j _ { \ell } } } , \ell \in [ q ]$ , multilinearity and the payoff range [0, 1] give

$$
\Big \| \partial _ { j _ { 1 } } \cdot \cdot \cdot \partial _ { j _ { q } } \mathscr { G } _ { i } ( { \pmb x } ) [ z _ { j _ { 1 } } , \ldots , z _ { j _ { q } } ] \Big \| _ { \infty } \leq 2 \prod _ { \ell = 1 } ^ { q } \| z _ { j _ { \ell } } \| _ { 1 } .
$$

Indeed, for every output coordinate, expanding $\mathcal { U } _ { i }$ over pure profiles bounds the absolute value of the mixed directional derivative of each term in $\mathcal { G } _ { i } ( \pmb { x } ) = \mathcal { U } _ { i } ( \pmb { x } _ { - i } ) - \langle \mathcal { U } _ { i } ( \pmb { x } _ { - i } ) , \pmb { x } _ { i } \rangle$ 1 by $\textstyle \prod _ { \ell = 1 } ^ { q } \left\| z _ { j _ { \ell } } \right\| _ { 1 }$ . This proves the first inequality in the statement by taking $q = 1$

Fix $i \in [ N ]$ and an integer $1 \leq h \leq N$ . Use $( { \pmb x } _ { j } ) _ { j \in [ N ] }$ with the extension from Section 3, and set $\widetilde { \pmb { g } } _ { i } ^ { ( t ) } = \mathcal { G } _ { i } ( \pmb { x } ^ { ( t ) } )$ for every integer t. Write $k _ { h , s } = \ker ( \delta ^ { h } ) _ { s }$ for $s \geq 0$ . Since $\Sigma _ { s \geq 0 } k _ { h , s } = 0$ , for every integer t and $s \geq 0$ define the Taylor remainder by

$$
\chi _ { i , t , s } ^ { \mathrm { T a y } } : = \mathcal { G } _ { i } \left( { \pmb x } ^ { ( t - s ) } \right) - \mathcal { G } _ { i } \left( { \pmb x } ^ { ( t ) } \right) - \sum _ { j = 1 } ^ { N } \partial _ { j } \mathcal { G } _ { i } \left( { \pmb x } ^ { ( t ) } \right) \left( { \pmb x } _ { j } ^ { ( t - s ) } - { \pmb x } _ { j } ^ { ( t ) } \right) .
$$

Taylor’s formula gives

$$
\left( \delta ^ { h } \widetilde { \pmb { g } } _ { i } \right) ^ { ( t ) } = \sum _ { j = 1 } ^ { N } \partial _ { j } \mathcal { G } _ { i } \left( \pmb { x } ^ { ( t ) } \right) \left( \delta ^ { h } \pmb { x } _ { j } \right) ^ { ( t ) } + \sum _ { s \ge 0 } k _ { h , s } \pmb { \chi } _ { i , t , s } ^ { \mathrm { \tiny { \mathrm { T a y } } } } .
$$

For every integer t, nonnegative integer s, distinct $j , k \in [ N ]$ , and $\alpha \in [ 0 , 1 ]$ , the profile ${ \pmb x } ^ { ( t ) } +$ $\alpha \left( \pmb { x } ^ { ( t - s ) } - \pmb { x } ^ { ( t ) } \right)$ belongs to $\Pi _ { \ell = 1 } ^ { N } \Delta ^ { m _ { \ell } }$ . The bound with $q = 2$ gives

$$
\begin{array} { r l } & { \quad \left\| \partial _ { j } \partial _ { k } \mathcal { G } _ { i } \left( { \pmb x } ^ { ( t ) } + \alpha \left( { \pmb x } ^ { ( t - s ) } - { \pmb x } ^ { ( t ) } \right) \right) \left[ { \pmb x } _ { j } ^ { ( t - s ) } - { \pmb x } _ { j } ^ { ( t ) } , { \pmb x } _ { k } ^ { ( t - s ) } - { \pmb x } _ { k } ^ { ( t ) } \right] \right\| _ { \infty } } \\ & { \le 2 \left\| { \pmb x } _ { j } ^ { ( t - s ) } - { \pmb x } _ { j } ^ { ( t ) } \right\| _ { 1 } \left\| { \pmb x } _ { k } ^ { ( t - s ) } - { \pmb x } _ { k } ^ { ( t ) } \right\| _ { 1 } . } \end{array}
$$

Since $\partial _ { j } ^ { 2 } { \mathcal { G } } _ { i } = 0$ , the integral Taylor remainder and $\begin{array} { r } { \int _ { 0 } ^ { 1 } 2 ( 1 - \alpha ) \mathrm { d } \alpha = 1 } \end{array}$ yield

$$
\left\| \boldsymbol { \chi } _ { i , t , s } ^ { \mathrm { T a y } } \right\| _ { \infty } \leq \left( \sum _ { j = 1 } ^ { N } \left\| \boldsymbol { x } _ { j } ^ { ( t - s ) } - \boldsymbol { x } _ { j } ^ { ( t ) } \right\| _ { 1 } \right) ^ { 2 } .
$$

Since $\pmb { g } _ { i } ^ { ( s ) } = \pmb { 0 }$ for every integer $s \leq 0 ,$ , for every $t \geq 1$

$$
\left( \delta ^ { h } \mathbf { g } _ { i } \right) ^ { ( t ) } = \left( \delta ^ { h } \widetilde { \mathbf { g } } _ { i } \right) ^ { ( t ) } - \mathcal { G } _ { i } \left( \mathbf { \boldsymbol { x } } ^ { ( 0 ) } \right) \sum _ { s \geq t } k _ { h , s } .
$$

Therefore, for every $t \geq 1$ , set

$$
\mathcal { X } _ { i , h } ^ { ( t ) } : = \sum _ { s \geq 0 } k _ { h , s } \mathcal { X } _ { i , t , s } ^ { \mathrm { { T a y } } } - \mathcal { G } _ { i } \left( { \pmb x } ^ { ( 0 ) } \right) \sum _ { s \geq t } k _ { h , s } .
$$

Fix $T \geq 1$ . Weighted Cauchy–Schwarz and Hölder’s inequality give

$$
\sum _ { t = 1 } ^ { T } \left\| \sum _ { s \geq 0 } k _ { h , s } \chi _ { i , t , s } ^ { \mathrm { T a y } } \right\| _ { \infty } ^ { 2 } \leq N ^ { 3 } \left( \sum _ { s \geq 0 } \left| k _ { h , s } \right| \right) \sum _ { s \geq 0 } \left| k _ { h , s } \right| \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left\| x _ { j } ^ { ( t - s ) } - x _ { j } ^ { ( t ) } \right\| _ { 1 } ^ { 4 } .
$$

For every $j \in [ N ]$ , integer $t ,$ and integer $s \geq 1$ , telescoping and Hölder’s inequality yield

$$
\left\| \pmb { x } _ { j } ^ { ( t - s ) } - \pmb { x } _ { j } ^ { ( t ) } \right\| _ { 1 } ^ { 4 } \leq s ^ { 3 } \sum _ { r = 0 } ^ { s - 1 } \left\| \left( \mathbb { D } \pmb { x } _ { j } \right) ^ { ( t - r ) } \right\| _ { 1 } ^ { 4 } .
$$

For every $j \in [ N ]$ and integer $s \geq 1 .$ , the extension of $\boldsymbol { \mathscr { x } } _ { j }$ from Section 3 gives

$$
\sum _ { t = 1 } ^ { T } \sum _ { r = 0 } ^ { s - 1 } \left. \left( \mathsf { D } \mathbf { x } _ { j } \right) ^ { ( t - r ) } \right. _ { 1 } ^ { 4 } \leq s \sum _ { t = 1 } ^ { T } \left. \left( \mathsf { D } \mathbf { x } _ { j } \right) ^ { ( t ) } \right. _ { 1 } ^ { 4 } .
$$

Substitution and (A.10) give

$$
\sum _ { t = 1 } ^ { T } \left\| \sum _ { s \geq 0 } k _ { h , s } \chi _ { i , t , s } ^ { \mathrm { T a y } } \right\| _ { \infty } ^ { 2 } \leq 2 ^ { 5 6 } N ^ { 1 5 } \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left\| \left( \mathsf { D } \pmb { x } _ { j } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 4 } .
$$

By (A.16) and (A.17),

$$
\begin{array} { r l } & { \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \left\| \left( \mathsf { D } \pmb { x } _ { j } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 4 } \leq 9 \cdot 2 ^ { 6 1 } \gamma ^ { 2 } N ^ { 1 2 } \displaystyle \sum _ { j = 1 } ^ { N } \sum _ { t = 1 } ^ { T } \lambda _ { j } ^ { ( t ) } \left\| \left( \mathsf { D } \pmb { x } _ { j } \right) ^ { ( t ) } \right\| _ { 1 } ^ { 2 } } \\ & { \quad \quad \quad \leq 9 \sqrt { 2 } \cdot 2 ^ { 6 5 } \gamma ^ { 2 } N ^ { 1 2 } \mathcal { P } _ { T } ^ { \Xi } . } \end{array}
$$

The second inequality uses $\lambda _ { j } ^ { ( t ) } \leq \sqrt { \lambda _ { j } ^ { ( t ) } }$ , followed by (A.16). Substitution gives

$$
\sum _ { t = 1 } ^ { T } \left\| \sum _ { s \geq 0 } k _ { h , s } \chi _ { i , t , s } ^ { \mathrm { T a y } } \right\| _ { \infty } ^ { 2 } \leq 9 \sqrt { 2 } \cdot 2 ^ { 1 2 1 } \gamma ^ { 2 } N ^ { 2 7 } \mathcal { P } _ { T } ^ { \Xi } .
$$

Since $\left\| \mathcal { G } _ { i } ( \pmb { x } ^ { ( 0 ) } ) \right\| _ { \infty } \leq 1 ,$ , (A.11) gives

$$
\sum _ { t = 1 } ^ { T } \left. \mathcal { G } _ { i } \left( \pmb { x } ^ { ( 0 ) } \right) \sum _ { s \geq t } k _ { h , s } \right. _ { \infty } ^ { 2 } \leq 2 ^ { 5 9 } N ^ { 1 3 } .
$$

By the definition of $\chi _ { i , h } ^ { ( t ) }$ , for every $t \geq 1$

$$
\left\| \boldsymbol { \chi } _ { i , h } ^ { ( t ) } \right\| _ { \infty } ^ { 2 } \leq 2 \left\| \sum _ { s \geq 0 } \boldsymbol { k } _ { h , s } \boldsymbol { \chi } _ { i , t , s } ^ { \mathrm { T a y } } \right\| _ { \infty } ^ { 2 } + 2 \left\| \mathcal { G } _ { i } \left( \boldsymbol { x } ^ { ( 0 ) } \right) \sum _ { s \geq t } \boldsymbol { k } _ { h , s } \right\| _ { \infty } ^ { 2 } .
$$

Summing this inequality and using the preceding two estimates proves (A.27). Finally, $\gamma \leq N ^ { - 2 0 }$ implies

$$
\gamma ^ { 2 } N ^ { 2 7 } \le N ^ { - 1 3 } \le N ^ { 1 3 } ,
$$

which proves (A.28).