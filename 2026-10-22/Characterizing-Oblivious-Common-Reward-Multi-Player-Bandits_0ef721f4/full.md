# Characterizing Oblivious Common-Reward Multi-Player Bandits: Universal Stationarity, Coordination Traps, and Certified Global Rates

Mingxiao Li, Wenhai Lai, and Hongyu Wang

## Abstract

This paper presents a theoretical analysis of the oblivious common-reward multi-player multi-armed bandit (OC-MPMAB), a minimal multi-player bandit model in which multiple players select arms simultaneously and observe only a shared reward. Crucially, the players are fully oblivious: they do not observe the actions of other players and do not even know whether any other players exist. The contributions of this work are threefold. First, we prove that the independent gradient-bandit updates applied by the players jointly form an unbiased stochastic gradient estimator of a hidden productpolicy objective. This property then yields a universal finitetime first-order stationarity guarantee for every instance. Second, we show that stationarity cannot, in general, be strengthened to global optimality. To illustrate this, we identify a product-policy trap obstruction that rules out the guarantee of global optimality. Third, we show that global optimality can be recovered under structural assumptions on the reward table. To make global optimality achievable, we introduce finite-table certificates that separately characterize product-gradient dominance and self-bounding stochastic noise. We then establish certificate-conditioned bounds on the product-policy gap and pseudo-regret. Consequently, the standard O(1/t) productpolicy gap and O(log T) pseudo-regret rates hold when the certificates are uniform.

## Introduction

Multi-armed bandits provide a fundamental framework for sequential decision making under partial feedback. Classical algorithms, such as UCB and Thompson sampling, achieve strong regret guarantees for single-player stochastic bandits (Thompson 1933; Lai and Robbins 1985; Auer, Cesa-Bianchi, and Fischer 2002; Agrawal and Goyal 2012; Lattimore and Szepesvari 2020). Recently, stochastic gradient methods provide an alternative approach. For example, likelihood-ratio estimators, REINFORCE, and gradient bandit algorithms update softmax policies using sampled actions and rewards (Robbins and Monro 1951; Glynn 1990; Williams 1992; Sutton and Barto 2018). For single-player bandits, these methods can achieve global optimality under self-bounding noise and weak exploration conditions (Me et al. 2023).

These single-player guarantees rely on the assumption that each arm has a fixed reward distribution. This assumption may fail in modular AI systems, where separate controllers can adapt prompts, retrieval, tool usage, and verification based on the same end-to-end score. Due to blackbox interfaces, access restrictions, or separate ownership, the decisions and even the existence of other adaptive components may remain hidden. Consequently, a controller may appear to face a standard bandit problem, while the value of each local arm changes as the hidden controllers adapt. This hidden coupling motivates a minimal model that characterizes what can be achieved from shared rewards alone. We formalize this setting as the oblivious common-reward multi-player multi-armed bandit (OC-MPMAB). In an OC-MPMAB, each player selects a local arm, the environment evaluates the joint action, and all players receive the same reward. Crucially, each player observes only its selected arm and the shared reward. In the setting studied here, each player applies the standard gradient-bandit update independently. From a centralized perspective, however, the independent softmax policies induce a product distribution over joint actions. As a main contribution of this work, we show that the local updates jointly constitute an unbiased stochastic gradient estimator of the expected-reward objective induced by this hidden product policy.

Moreover, the model considered in this work difers fundamentally from both equilibrium selection in common-interest games and payof-based learning under a specified game model. Existing analyses typically assume a known game form, observable joint actions, or prescribed revision dynamics. In contrast, players in OC-MPMAB have access to none of this information. Therefore, the theoretical convergence results of these models cannot be directly applied to OC-MPMAB. In this paper, we revisit and re-establish the theoretical convergence guarantees of OC-MPMAB. Specifically, we show that finite-time stationarity is guaranteed for every finite normalized reward table, whereas global optimality cannot be guaranteed in general because the productpolicy landscape may contain attracting suboptimal traps. Nevertheless, global bounds on the product-policy gap and pseudo-regret can be recovered when ofline certificates for the reward table rule out such traps and control stochastic sampling noise. Accordingly, the positive and negative results characterize fundamental properties of the hidden product-policy landscape.

In summary, this paper provides a comprehensive theoretical analysis of the OC-MPMAB model. The contributions

of this work are as follows:

• We prove that the independent local gradient-bandit updates in OC-MPMAB jointly constitute an unbiased stochastic gradient estimator of the centralized expectedreward objective over product policies, even though no player observes either the joint action or this objective. This identity then yields a universal finite-time stationarity guarantee.

• We show that finite-time stationarity does not imply global optimality for OC-MPMAB: strict trap minors create suboptimal attractors. Separately, on a deterministic block trap, the stochastic update converges from the default initialization to a suboptimal block with constant probability under suficiently small diminishing stepsizes.

• We formulate guarantees of global optimality as a finite reward-table certification problem. Specifically, we introduce separate certificates that rule out product-gradient traps and control stochastic sampling noise. These certificates yield bounds on the product-policy gap and pseudoregret for both sparse coordination reward tables and locally dominant dense reward tables.

Notation. The bold lowercase symbol $\pmb { a } = ( A _ { 1 } , \ldots , A _ { N } )$ denotes a joint action, where $A _ { i } \in { \mathcal { A } } _ { i }$ is the local arm of player i; $\mathbf { \delta } \mathbf { a } _ { - i }$ denotes the profile of all players except $i ,$ and $\begin{array} { r } { \mathcal { A } _ { - i } = \prod _ { i \neq i } \mathcal { A } _ { j } } \end{array}$ . A policy tuple $\pi = ( \pi _ { 1 } , \ldots , \pi _ { N } )$ induces $\begin{array} { r } { \pi ( \mathbf { a } ) \ : = \ : \dot { \prod _ { i } } \pi _ { i } ( A _ { i } ) } \end{array}$ , while π<sub>H</sub> denotes the product policy parameterized by the preference collection H. For vectors, ${ \bar { \| } } \cdot \| _ { 2 }$ is the Euclidean norm; for matrices, $\| \cdot \| _ { 2 }$ and $\| \cdot \| _ { F }$ are the spectral and Frobenius norms, respectively.

## OC-MPMAB Model

## System Model

An oblivious common-reward multi-player multi-armed bandit (OC-MPMAB) consists of a finite player set $[ N ] =$ $\{ 1 , 2 , \ldots , N \}$ , where each player $i \in [ N ]$ ] is associated with a finite arm set $A _ { i }$ of cardinality $K _ { i } .$ . At each round $t ,$ all players simultaneously select local arms. Denote by $A _ { i , t }$ the local arm selected by player i. These local arms then form the joint action

$$
\begin{array} { r } { \pmb { a } _ { t } = ( A _ { 1 , t } , \ldots , A _ { N , t } ) \in \mathcal { A } , \mathcal { A } = \mathcal { A } _ { 1 } \times \cdots \times \mathcal { A } _ { N } . } \end{array}\tag{1}
$$

For every joint action $\mathbf { \alpha } _ { \mathbf { { a } } } \in \mathcal { A }$ , the environment specifies a stochastic reward distribution. Let $\mathcal { F } _ { t }$ denote the centralized history before arm selection at round t. After the joint action $\mathbf { } \mathbf { a } _ { t }$ is selected, the environment generates a scalar reward $R _ { t }$ according to the reward distribution associated with $\mathbf { } \mathbf { a } _ { t }$ Following the standard convention for stochastic bandits, we assume that rewards are normalized:

$$
R _ { t } \in [ 0 , 1 ] \quad { \mathrm { f o r ~ a l l } } t .\tag{2}
$$

We assume this normalized feedback model throughout the paper. Moreover, we define the mean reward of the joint action $\mathbf { \pmb { a } } _ { t }$ as

$$
r ( \pmb { a } _ { t } ) = \mathbb { E } [ R _ { t } \ | \ \mathcal { F } _ { t } , \pmb { a } _ { t } ] \in [ 0 , 1 ] .\tag{3}
$$

Therefore, the conditional mean reward depends only on the current joint action through a fixed finite reward table and is independent of the previous history.

In OC-MPMABs, oblivious means that each player is unaware of both the existence and the actions of the other players. Specifically, player i observes only the sampled local arm and the reward received in each round. Consequently, each player behaves as a standard single-agent player and executes a conventional gradient bandit algorithm. More importantly, the reward is common: all players observe the same scalar reward $R _ { t }$ at round t, and there are no player-specific rewards. We remark that the joint action space A, the common reward function $r ,$ and the product-policy structure introduced below are used solely for centralized analysis. None of this information is available to any individual player. The next subsection introduces the local gradient bandit update for each player.

## Gradient Bandit Algorithm for Each Player

Because a player has no information about other players, the player applies the standard gradient bandit update using only the sampled local arm and the common reward. Specifically, player i maintains a numerical preference variable $H _ { i , t } ( x ) \in$ R for every local arm $x \in \bar { A } _ { i \cdot } \operatorname { A }$ larger value of $H _ { i , t } ( x )$ assigns a higher sampling probability to arm x in round t. At the beginning of each round, player i converts these preferences into the softmax distribution

$$
\pi _ { i , t } ( x ) = \frac { \exp ( H _ { i , t } ( x ) ) } { \sum _ { y \in A _ { i } } \exp ( H _ { i , t } ( y ) ) } , \ x \in A _ { i } .\tag{4}
$$

The player then samples a local arm according to

$$
A _ { i , t } = x \mathrm { w i t h p r o b a b i l i t y } \pi _ { i , t } ( x ) .\tag{5}
$$

After all players have chosen their arms, the environment observes the joint action ${ \pmb a } _ { t } = ( A _ { 1 , t } , A _ { 2 , t } , . . . , A _ { N , t } )$ and broadcasts the same reward sample $R _ { t }$ to every player.

It remains to specify the update of each $H _ { i , t } ( x )$ . By convention, every $H _ { i , 1 } ( x )$ is initialized to zero. Each player then updates the local preference variables as

$$
H _ { i , t + 1 } ( x ) = H _ { i , t } ( x ) + \gamma _ { t } g _ { i , t } ( x ) , x \in \mathcal { A } _ { i } ,\tag{6}
$$

where $\gamma _ { t } \geq 0$ is a deterministic stepsize and

$$
\begin{array} { r l } & { g _ { i , t } ( x ) = R _ { t } \big ( \mathbf { 1 } \{ A _ { i , t } = x \} - \pi _ { i , t } ( x ) \big ) } \\ & { \qquad = \left\{ \begin{array} { l l } { R _ { t } ( 1 - \pi _ { i , t } ( x ) ) , } & { \mathrm { i f ~ } A _ { i , t } = x , } \\ { - R _ { t } \pi _ { i , t } ( x ) , } & { \mathrm { i f ~ } x \neq A _ { i , t } . } \end{array} \right. } \end{array}\tag{7}
$$

Note that the update does not use the number of players, the actions of other players, joint-action samples, messages, or the reward table.

We now examine the above process from a centralized perspective. Let $\begin{array} { r } { d = \sum _ { i = 1 } ^ { N } K _ { i } } \end{array}$ and let $\mathbf { H } = ( H _ { i } ( x ) : i \in$ $[ N ] , x \in \mathcal { A } _ { i } ) \in \mathbb { R } ^ { d }$ denote the vector of all preference variables. We suppress the round index and write each $H _ { i } ( x )$ as a generic preference variable; the value generated by the update in round t is denoted by $H _ { i , t } ( x )$ . Moreover, we write $\mathbf { g } _ { t } = ( g _ { i , t } ( x ) : i \in [ N ] , x \in \mathcal { A } _ { i } )$ for the vector obtained by concatenating all local update coordinates at round t. Given H, the players independently sample their local arms according to the softmax probabilities in (4). Therefore, the probability assigned to a joint action ${ \pmb a } = ( A _ { 1 } , A _ { 2 } , \ldots , A _ { N } )$ is

$$
\pi _ { \mathbf { H } } ( \pmb { a } ) = \prod _ { i = 1 } ^ { N } \pi _ { i } ( A _ { i } ) .\tag{8}
$$

Treating H as the optimization variable, we define the expected reward as

$$
J ( \mathbf { H } ) = \mathbb { E } _ { a \sim \pi _ { \mathbf { H } } } [ r ( a ) ] = \sum _ { a \in \mathcal { A } } \pi _ { \mathbf { H } } ( a ) r ( a ) .\tag{9}
$$

Although the players update independently, their coupled updates estimate the gradient of J. Because finite softmax policies can approach every pure profile, the relevant optimum is

$$
\operatorname* { s u p } _ { \mathbf { H } \in \mathbb { R } ^ { d } } J ( \mathbf { H } ) = J ^ { \star } ,\tag{10}
$$

where $J ^ { \star } = \operatorname* { m a x } _ { \scriptstyle { a \in { \mathcal { A } } } } r ( a )$ . No player observes this objective or the product policy in (8); both are used only for centralized analysis.

## Unbiased Product-Policy Gradient

Unlike in a single-player bandit, the observed reward depends on the full joint action rather than on one local arm. We first show that, despite this coupling, each local update is an unbiased estimator of the corresponding gradient coordinate of J(H). The concatenated updates therefore implement stochastic gradient ascent on the objective in (10).

Proposition 1. For every round $t \geq 1$ , player $i \in [ N ]$ , and arm $x \in A _ { i } ,$ , the gradient bandit update $g _ { i , t } ( x )$ in (7) is an unbiased estimator of the partial derivative of the expected reward with respect to the preference variablefor arm x:

$$
\mathbb { E } \left[ g _ { i , t } ( x ) \mid \mathcal { F } _ { t } \right] = \left. \frac { \partial J ( \mathbf { H } ) } { \partial H _ { i } ( x ) } \right| _ { \mathbf { H } = \mathbf { H } _ { t } } ,\tag{11}
$$

where the conditional expectation is taken over the sampled arms and the reward in round t.

Proofsketch. Here and throughout, $\begin{array} { r l } { \nabla J ( \mathbf { H } ) } & { { } = } \end{array}$ $( \partial J ( \mathbf H ) / \partial H _ { i } ( x ) : i \in [ N ] , x \in \mathcal A _ { i } \bar { ) }$ denotes the gradient vector of J with respect to the concatenated preference vector H. Notably, within the product policy, the only term that depends on ${ \dot { H } } _ { i } ( x )$ is $\pi _ { i } ( A _ { i } )$ , whose softmax derivative is given by

$$
\frac { \partial \pi _ { i } ( A _ { i } ) } { \partial H _ { i } ( x ) } = \pi _ { i } ( A _ { i } ) \big ( { \bf 1 } \{ A _ { i } = x \} - \pi _ { i } ( x ) \big ) .\tag{12}
$$

Substituting this identity into $\partial J ( \mathbf { H } ) / \partial H _ { i } ( x )$ and using the identity $\mathbb { E } [ \mathbf { \bar { \boldsymbol { R } } } _ { t } \ | \ \mathcal { F } _ { t } , \mathbf { \boldsymbol { a } } _ { t } ] = r ( \mathbf { \boldsymbol { a } } _ { t } )$ yields the same summation as the conditional expectation of $g _ { i , t } ( x )$ . The complete derivation is provided in the supplementary material. □

Efect of baselines. We remark that adding an actionindependent baseline in the update (7) will not change the claim in Proposition 1. Specifically, if $b _ { t }$ is $\mathcal { F } _ { t } .$ -measurable, its score contribution has zero conditional mean because ${ \mathbb E } [ \mathbf { 1 } \{ A _ { i , t } = x \} - \pi _ { i , t } ( x ) \mid { \mathcal F } _ { t } ] = 0 ;$ therefore, replacing $R _ { t }$ by $R _ { t } - b _ { t }$ in (7) preserves Proposition 1.

## Finite-Time Stationarity

Proposition 1 identifies thejoint update as stochastic gradient ascent on (10). Without additional reward-table structure, a natural unconditional guarantee is finite-time first-order stationarity. We first establish the smoothness of J(H).

Lemma 1. The objective J(H) is globally L-smooth. Specifically, if

$$
d = \sum _ { i = 1 } ^ { N } K _ { i } ,\tag{13}
$$

then a valid smoothness constant is $L = 2 d .$ Equivalently, for all preference vectors H and H<sup>′</sup>, we have

$$
J ( \mathbf { H } ^ { \prime } ) \geq J ( \mathbf { H } ) + \langle \nabla J ( \mathbf { H } ) , \mathbf { H } ^ { \prime } - \mathbf { H } \rangle - \frac { L } { 2 } \| \mathbf { H } ^ { \prime } - \mathbf { H } \| _ { 2 } ^ { 2 } .\tag{14}
$$

Proof. For coordinates $u \ = \ ( i , x )$ and $\begin{array} { r } { \boldsymbol { v } \ = \ ( j , y ) } \end{array}$ , let $S _ { u } ( \dot { \bf a } ) = { \bf 1 } \{ A _ { i } = x \} - \pi _ { i } ( x ) $ Proposition 1 gives $\partial J / \partial H _ { i } ( x ) \ \stackrel { \cdot } { = } \ \mathbb { E } [ r ( \pmb { a } ) \dot { S } _ { u } ( \pmb { a } ) ]$ . The Hessian entry indexed $\dot { \mathrm { \bf ~ b y ~ } } ( v , u ) \mathrm { i s } \dot { \mathbb { E } } [ r \dot { S } _ { u } \ddot { S } _ { v } ] + \mathbb { E } [ r \partial { S } _ { u } / \partial H _ { j } ( y ) ]$ , where $\partial S _ { u } / \partial \bar { H } _ { j } ( y ) ~ = ~ - { \bf 1 } \bar { \{ i ~ = ~ j \} } \pi _ { i } ( x \dot { ) } ( { \bf 1 } \{ x ~ \dot { = ~ } \bar { y } \} ~ - ~ \pi _ { i } ( y ) )$ Both terms have absolute value at most 1, so every Hessian entry is bounded by 2. Since the Hessian is $d \times d ,$ $\| \nabla ^ { 2 } J ( \mathbf { H } ) \| _ { 2 } \leq \| \nabla ^ { 2 } J ( \mathbf { H } ) \| _ { F } \leq 2 d$ , and the stated smoothness inequality follows. □

Combining Proposition 1 with Lemma 1 yields the following finite-time stationarity guarantee for OC-MPMABs.

Theorem 1 (Finite-time stationarity). Let

$$
G ^ { 2 } = 2 N , \Gamma _ { T } = \sum _ { t = 1 } ^ { T } \gamma _ { t } .\tag{15}
$$

Assume that $\gamma _ { t } \geq 0 .$ for all t and $\Gamma _ { T } > 0 .$ . Then

$$
\frac { \sum _ { t = 1 } ^ { T } \gamma _ { t } \mathbb { E } \left[ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \right] } { \Gamma _ { T } } \leq \frac { J ^ { \star } - J ( \mathbf { H } _ { 1 } ) } { \Gamma _ { T } } + \frac { L G ^ { 2 } } { 2 } \frac { \sum _ { t = 1 } ^ { T } \gamma _ { t } ^ { 2 } } { \Gamma _ { T } } .
$$

Consequently,

(16)

$$
\operatorname* { m i n } _ { 1 \leq t \leq T } \mathbb { E } \big [ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \big ] \leq \frac { J ^ { \star } - J ( \mathbf { H } _ { 1 } ) } { \Gamma _ { T } } + \frac { L G ^ { 2 } } { 2 } \frac { \sum _ { t = 1 } ^ { T } \gamma _ { t } ^ { 2 } } { \Gamma _ { T } } .\tag{17}
$$

In particular, $i f \gamma _ { t } = \eta > 0$ for all t, then

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \big ] \leq \frac { J ^ { \star } - J ( \mathbf { H } _ { 1 } ) } { \eta T } + \frac { L \eta G ^ { 2 } } { 2 } .\tag{18}
$$

Further, let $\Delta _ { 1 } = J ^ { \star } - J ( { \bf H } _ { 1 } ) . \ I f \Delta _ { 1 } > 0 ,$ , choosing $\eta =$ $\sqrt { \frac { 2 \Delta _ { 1 } } { L G ^ { 2 } T } } y i e l d s$

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \big ] \leq \sqrt { \frac { 2 L G ^ { 2 } \Delta _ { 1 } } { T } } .\tag{19}
$$

$I f \Delta _ { 1 } = 0 _ { \cdot }$ , the left-hand side is zero. Since $G ^ { 2 } = 2 N$ and $0 \leq \Delta _ { 1 } \leq 1$ , the upper bound is at most

$$
2 { \sqrt { \frac { L N } { T } } } = 2 { \sqrt { \frac { 2 d N } { T } } } .\tag{20}
$$

Proof sketch. Conditioning on the history $\mathcal { F } _ { t }$ , Proposition 1 implies that E[g<sub>t</sub> $\mid \mathcal { F } _ { t } \} = \breve { \nabla } J ( \mathbf { H } _ { t } )$ . The centered score vector for each player has squared norm at most 2. Consequently, $\mathbb { E } [ \| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } ~ \big | ~ \dot { \mathcal { F } } _ { t } ] ^ { * } \le G ^ { 2 } = \dot { 2 } N$ . Applying Lemma 1 to the update $\mathbf { H } _ { t + 1 } = \mathbf { H } _ { t } + \gamma _ { t } \mathbf { g } _ { t }$ yields

$$
\mathbb { E } [ J ( \mathbf { H } _ { t + 1 } ) - J ( \mathbf { H } _ { t } ) \mid \mathcal { F } _ { t } ] \geq \gamma _ { t } \Vert \nabla J ( \mathbf { H } _ { t } ) \Vert _ { 2 } ^ { 2 } - \frac { L \gamma _ { t } ^ { 2 } G ^ { 2 } } { 2 } .\tag{21}
$$

Summing this inequality over t, using $J ( \mathbf { H } ) ~ \leq ~ J ^ { \star }$ , and dividing by $\Gamma _ { T }$ yields the stationarity bound. The constantstep-size rate follows by optimizing over η. The complete proof is provided in the supplementary material. □

Theorem 1 holds without additional structural assumptions: it requires only finite action spaces, the normalization $R _ { t } \in [ 0 , 1 ]$ , and the unbiased product-policy gradient identity. In particular, it does not require $J ( { \bar { \mathbf { H } } } )$ to be concave, the optimal joint action to be unique, or the players to satisfy any coordination structure. The result provides finite-time control of the average and minimum expected gradient norm; it does not assert convergence of the iterates or global optimality. The next section explains why this distinction is unavoidable in general.

## Verifiable Structural Global Convergence

Beyond unconditional stationarity, the results in this section separate two further layers: structural obstructions to global optimality and ofline certificates that exclude those obstructions while controlling stochastic noise.

## Why Stationarity Does Not Imply Optimality

First-order stationarity is insuficient for global optimality because, under the deterministic product-gradient flow $\dot { \mathbf H } = \nabla J ( \mathbf H )$ , a globally beneficial move may require several players to change arms jointly while every unilateral gradient points away from that move. The following two-player, threearm example makes this coordination obstruction explicit:

$$
\begin{array} { r } { \overrightarrow { A _ { 1 } } = 0 \vert \begin{array} { c c c } { A _ { 2 } } & { = 0 } & { A _ { 2 } } & { = 1 \quad A _ { 2 } = 2 } \\ { 0 . 9 } & { 0 . 9 } & { 0 } \\ { 0 . 9 } & { 0 . 9 } & { 0 } \\ { A _ { 1 } } & { = 2 } & { 0 } \end{array} } \end{array}\tag{22}
$$

Although the joint action ${ \pmb a } ^ { \star } = ( 2 , 2 )$ is globally optimal, the block $\{ 0 , \check { 1 } \} ^ { 2 }$ is locally attracting under this deterministic product-gradient flow. Letting $q _ { i } = \pi _ { i } ( 2 )$ , the objective depends on H only through $( q _ { 1 } , q _ { 2 } )$ and satisfies

$$
J ( \mathbf { H } ) = 0 . 9 ( 1 - q _ { 1 } ) ( 1 - q _ { 2 } ) + q _ { 1 } q _ { 2 } .\tag{23}
$$

In contrast to the single-player bandit setting, the local advantage of arm 2 is averaged with respect to the current policy of the other player and may therefore point away from the globally optimal profile. The following proposition formalizes this efect by identifying an invariant basin of attraction that contains the default initialization.

Proposition 2 (Block-trap basin). For the reward table in (22), the region

$$
q _ { 1 } < \frac 9 { 1 9 } , q _ { 2 } < \frac 9 { 1 9 }\tag{24}
$$

is forward-invariant under this flow, and every trajectory initialized within this region is attracted to the suboptimal $b l o c k \{ 0 , 1 \} ^ { 2 }$ . The default uniform initialization satisfies $q _ { 1 } = q _ { 2 } = 1 / 3$ and thus lies within this basin ofattraction.

Proofsketch. Because the two suboptimal arms of each player constitute a block with reward 0.9, the softmax gradient component that increases the probability assigned to the optimal arm of player 1 has the sign of $1 . 9 \dot { q } _ { 2 } - \dot { 0 . 9 }$ , and the corresponding component for player 2 follows by symmetry. Consequently, $q _ { 1 } = q _ { 2 } = 9 / 1 9$ is the stationary point that separates the two regimes in the reduced coordinate system. If both probabilities are below $9 / 1 9 _ { ; }$ , both gradient components are negative. Therefore, the flow decreases both optimal-arm probabilities and cannot escape the lower-left region. The complete derivation is provided in the supplementary material. □

Proposition 2 shows that Theorem 1 does not imply global optimality. The obstruction is structural: the globally optimal joint action consists of one local arm per player, and any such arm may have a negative product-policy advantage unless the other players already coordinate with it. The following proposition shows that this limitation persists beyond the deterministic basin analysis.

Proposition 3 (Trap-minor obstruction). Suppose that a deterministic two-player reward table contains a $2 \times 2$ minor with a corner $( x _ { 0 } , y _ { 0 } )$ whose reward is below J<sup>⋆</sup> but strictly exceeds both adjacent entries. If this corner also strictly dominates every unilateral deviation outside the minor, then its pure product policy has an open basin of attraction B. For every compact set $K \subset B$ and every fixed α $\begin{array} { r } { \iota > 0 , } \end{array}$ there exist $C _ { K } < \infty$ and $t _ { K } ( \alpha ) < \infty$ such that, for any finitepreference initialization that induces a product policy in K, the update in (6) with $\gamma _ { t } = \alpha / ( t + t _ { 0 } )$ converges to the trap with probability at least

$$
\left[ 1 - \frac { C _ { K } \alpha ^ { 2 } } { t _ { 0 } } \right] _ { + }\tag{25}
$$

for every $t _ { 0 } \geq t _ { K } ( \alpha )$ , where $[ z ] _ { + } = \operatorname* { m a x } \{ z , 0 \}$

The supplementary material provides complete proofs and the corresponding asymptotic lower bound on the expected gap. Together, these results constitute the negative component of the framework: the obstruction is a reusable structural pattern rather than a simulation artifact. They also identify precisely why the single-player softmax-gradient mechanism fails in the multi-player setting. Given the current policies of the other players, a local component of a globally optimal joint action may have a lower expected reward than a suboptimal local arm until the other players coordinate with that component. Consequently, finite action spaces, normalized rewards, and the oblivious gradient-bandit update are insufficient, on their own, to guarantee global optimality.

Default-initialization block trap. The tied block in Proposition 2 is not directly covered by the strict trap-minor result in Proposition 3. Nevertheless, for deterministic rewards, the supplementary material proves that there exist constants $C _ { 0 } < \infty$ and $\alpha _ { 0 } > 0$ , with $\dot { C } _ { 0 } \alpha _ { 0 } ^ { 2 } \le 1 / 2$ , such that, for every $\alpha \in ( 0 , \alpha _ { 0 } ]$ , the update initialized with zero preferences and run with $\overset { \cdot } { \gamma _ { t } } = \alpha / ( t + 1 )$ ) satisfies

$$
\operatorname* { P r } \{ \pi _ { i , t } ( 2 )  0 , \ : i = 1 , 2 \} \geq [ 1 - C _ { 0 } \alpha ^ { 2 } ] _ { + } \geq \frac { 1 } { 2 } .\tag{26}
$$

Thus, the stochastic update converges to the suboptimal block with a probability that is lower-bounded independently of the time horizon.

## Two Structural Certificates

To recover guarantees of global optimality without modifying the gradient update rule, we restrict our analysis to reward tables that satisfy two properties within a region containing the trajectory. First, every suboptimal product policy must have a gradient whose magnitude is suficient to control its optimality gap, thereby excluding the traps identified above. Second, the update variance must decrease with the gradient magnitude so that sampling noise does not dominate the resulting ascent. We formalize these requirements through product-gradient-dominance and selfbounding-noise certificates. Together, these certificates yield bounds on the product-policy gap and pseudo-regret under the specified conditions.

Fix an optimal joint action $\pmb { a } ^ { \star } = ( A _ { 1 } ^ { \star } , \ldots , A _ { N } ^ { \star } )$ . For a product policy $\pi = ( \pi _ { 1 } , \ldots , \pi _ { N } )$ , define

$$
J ( \pi ) = \sum _ { \pm \in { \mathcal { A } } } \left( \prod _ { i = 1 } ^ { N } \pi _ { i } ( A _ { i } ) \right) r ( \pmb { a } ) ,\tag{27}
$$

$$
Q _ { i } ^ { \pi } ( x ) = \sum _ { \substack { \pmb { a } _ { - i } \in \pmb { A } _ { - i } } } \left( \prod _ { j \neq i } \pi _ { j } ( A _ { j } ) \right) r ( x , \pmb { a } _ { - i } ) .\tag{28}
$$

Under the product-softmax parameterization, the preference gradient has coordinates

$$
\begin{array} { r } { \mathcal G _ { i , x } ( \pi ) = \pi _ { i } ( x ) \big ( Q _ { i } ^ { \pi } ( x ) - J ( \pi ) \big ) , } \end{array}\tag{29}
$$

and Proposition 1 gives $\mathcal { G } ( \pi _ { \mathbf { H } } ) = \nabla J ( \mathbf { H } )$ . For the baselinefree update, define the second-moment envelope

$$
\mathcal { V } ( \pi ) = \sum _ { \pm \in \mathcal { A } } \left( \prod _ { i = 1 } ^ { N } \pi _ { i } ( A _ { i } ) \right) r ( \pmb { a } ) \sum _ { i = 1 } ^ { N } \| \mathbf { e } _ { A _ { i } } - \pi _ { i } \| _ { 2 } ^ { 2 } ,\tag{30}
$$

where $\mathbf { e } _ { A _ { i } } \in \mathbb { R } ^ { K _ { i } }$ denotes the unit vector corresponding to $A _ { i }$ . Since $R _ { t } ^ { 2 } \leq R _ { t }$ , the conditional second moment of the update satisfies $\mathbb { E } [ \| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } ] \le \mathcal { V } ( \pi _ { \mathbf { H } _ { t } } )$ . For $c \in ( 0 , 1 ]$ define

$$
\begin{array} { r l } & { \Pi _ { \star } ( c ) = \{ \pi : \pi _ { i } ( A _ { i } ^ { \star } ) \geq c , \forall i \} , } \\ & { \Pi _ { \mathrm { a l l } } ( c ) = \{ \pi : \pi _ { i } ( x ) \geq c , \forall i , x \} . } \end{array}\tag{31}
$$

The all-arm region is considered only when it is nonempty.

Product-gradient-dominance certificate. The deterministic certificate is

$$
\lambda _ { \star } ( c ) = \operatorname* { i n f } _ { \pi \in \Pi _ { \star } ( c ) } \frac { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } } { ( J ^ { \star } - J ( \pi ) ) ^ { 2 } } .\tag{32}
$$

Positive $\lambda _ { \star } ( c )$ rules out suboptimal stationary points in Π<sub>⋆</sub>(c) by requiring the gradient magnitude to control the optimality gap.

Self-bounding-noise certificate. The stochastic certificate is

$$
M _ { \mathrm { a l l } } ( c ) = \operatorname* { s u p } _ { \pi \in \Pi _ { \mathrm { a l l } } ( c ) } \frac { \mathcal { V } ( \pi ) } { \Vert \mathcal { G } ( \pi ) \Vert _ { 2 } ^ { 2 } } .\tag{33}
$$

Set $M _ { \mathrm { a l l } } ( c ) = \infty { \mathrm { ~ i f ~ } } \mathcal { V } > 0$ at any zero-gradient policy, set it to 0 if the displayed index set is empty, and otherwise use the displayed supremum. Thus, finite $\dot { M } _ { \mathrm { a l l } } ( c )$ gives $\mathcal { V } ( \pi ) \le M _ { \mathrm { a l l } } \dot { ( c ) } \| \dot { \mathcal { G } } ( \pi ) \| _ { 2 } ^ { 2 }$ throughout $\Pi _ { \mathrm { a l l } } ( c )$ . Both certificates are ofline reward-table properties rather than assumptions fitted to a trajectory. Requiring these controls over the entire policy space may be unnecessarily restrictive. If a deterministic region $s$ is certified to contain the iterates, it sufices to use the localized constants

$$
\begin{array} { l } { \displaystyle \lambda _ { S } = \operatorname* { i n f } _ { \pi \in S _ { \epsilon } } \frac { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } } { ( J ^ { \star } - J ( \pi ) ) ^ { 2 } } , } \\ { \displaystyle M _ { S } = \operatorname* { s u p } _ { \pi \in S _ { \epsilon } } \frac { \mathcal { V } ( \pi ) } { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } } . } \end{array}\tag{34}
$$

Use the same zero-gradient and empty-supremum conventions for $M _ { S }$ . These constants sufice whenever $s$ contains the trajectory. The next subsection applies the all-arm certificates to a finite-horizon region.

## Structural Global Rate and Regret

When both certificates hold on a region containing the trajectory, product-gradient dominance converts every positive optimality gap into a quantitative ascent signal, while selfbounding noise ensures that the stochastic second-moment penalty remains proportional to this signal. With a suficiently small stepsize, smoothness then guarantees a decreasing expected gap and establishes pseudo-regret bounds; uniform certificates further imply that the expected value of the product policy converges to $J ^ { \star }$ . We first construct a certified finite-horizon region that contains every iterate.

Lemma 2 (Finite-horizon exploration certificate). Fix a horizon T, and let

$$
\Gamma _ { T } = \sum _ { t = 1 } ^ { T } \gamma _ { t } , K _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i \in [ N ] } K _ { i } , c _ { T } = \frac { e ^ { - 2 \Gamma _ { T } } } { K _ { \operatorname* { m a x } } } .\tag{35}
$$

If $H _ { i , 1 } ( x ) = 0$ for all i and x, then, for every $t \leq T + 1$ every player i, and every local arm $x \in A _ { i }$

$$
\pi _ { i , t } ( x ) \geq c _ { T } .\tag{36}
$$

Proof. Since $R _ { t } ~ \in ~ [ 0 , 1 ]$ and the centered score lies in $[ - 1 , \bar { 1 } ] , | g _ { i , t } ( x ) | \leq 1$ . Hence, for $t \leq T + 1 , | H _ { i , t } ( x ) | \leq$ $\smash { \sum _ { s = 1 } ^ { t - 1 } \gamma _ { s } \le \Gamma _ { T } }$ . The softmax formula then gives $\pi _ { i , t } ( x ) \geq$ $\overline { { e ^ { - } } } \mathrm { \check { r } } _ { T } \dot { \mathbf { \eta } } / ( K _ { i } e ^ { \Gamma _ { T } } ) \geq e ^ { - 2 \Gamma _ { T } } / K _ { \operatorname* { m a x } } = c _ { T }$ □

Lemma 2 therefore places the entire finite-horizon trajectory in $\Pi _ { \mathrm { a l l } } ( c _ { T } ) \subseteq \mathbf { \dot { \Pi } } \mathbf { \epsilon } _ { \star } ( c _ { T } )$ , where both certificates can be evaluated. To state the resulting performance guarantee, define the product-policy gap by

$$
D _ { t } = J ^ { \star } - J ( \pi _ { \mathbf { H } _ { t } } ) , \quad d _ { t } = \mathbb { E } [ D _ { t } ] ,\tag{37}
$$

and define the cumulative mean-reward regret random variable, whose expectation is the expected pseudo-regret, as

$$
\operatorname { R e g } ( T ) = \sum _ { t = 1 } ^ { T } \bigl ( J ^ { \star } - r ( \mathbf { a } _ { t } ) \bigr ) .\tag{38}
$$

On this certified region, $\lambda _ { \star }$ converts gradient progress into gap reduction, while $M _ { \mathrm { a l l } }$ controls stochastic update noise. These two controls yield the following finite-horizon guarantee.

Theorem 2 (Certificate-conditioned finite-horizon rate and pseudo-regret). Fix a time horizon T. Starting from $\mathbf { \bar { \boldsymbol { H } } } _ { i , 1 } ( \boldsymbol { x } ) = \mathrm { ~ 0 ~ }$ for all i and x, run the update in (6) using the step sizes $\{ \gamma _ { t } \} _ { t = 1 } ^ { T } ,$ , and define

$$
c _ { T } = \frac { e ^ { - 2 \Gamma _ { T } } } { K _ { \operatorname* { m a x } } } , \quad \lambda _ { T } = \lambda _ { \star } ( c _ { T } ) , \quad M _ { T } = M _ { \mathrm { a l l } } ( c _ { T } ) .\tag{39}
$$

Assume that $\lambda _ { T } ~ > ~ 0 , 0 ~ < ~ M _ { T } ~ < ~ \infty ,$ , and $0 ~ \leq ~ \gamma _ { t } ~ \leq$ $\begin{array} { r } { \frac { 1 } { L M _ { T } } , t = 1 , . . . , T } \end{array}$ . We further define

$$
\rho _ { t } = \gamma _ { t } \left( 1 - \frac { L M _ { T } \gamma _ { t } } { 2 } \right) .\tag{40}
$$

If $D _ { 1 } > 0 ,$ then, for every $t \leq T + 1$

$$
d _ { t } \leq \frac { 1 } { D _ { 1 } ^ { - 1 } + \lambda _ { T } \sum _ { s = 1 } ^ { t - 1 } \rho _ { s } } .\tag{41}
$$

Moreover,

$$
\mathbb { E } [ \mathrm { R e g } ( T ) ] \leq \sum _ { t = 1 } ^ { T } { \frac { 1 } { D _ { 1 } ^ { - 1 } + \lambda _ { T } \sum _ { s = 1 } ^ { t - 1 } \rho _ { s } } } .\tag{42}
$$

The complete proof is given in the supplementary material. Theorem 2 is an ofline finite-horizon guarantee: choose T and a stepsize schedule, certify $( \lambda _ { T } , M _ { T } ^ { \mathrm { - } } )$ on the induced region, and verify the stepsize condition. For a constant stepsize $\gamma _ { t } = \eta$ with $0 < \eta \dot { \leq } 1 / ( L M _ { T } )$ , let

$$
\beta _ { T } = \lambda _ { T } \eta \left( 1 - \frac { L M _ { T } \eta } { 2 } \right) .\tag{43}
$$

Then

$$
d _ { t } \leq \frac { 1 } { D _ { 1 } ^ { - 1 } + \beta _ { T } ( t - 1 ) }\tag{44}
$$

and

$$
\mathbb { E } [ \mathrm { R e g } ( T ) ] \leq D _ { 1 } + \frac { 1 } { \beta _ { T } } \log \bigl ( 1 + \beta _ { T } D _ { 1 } ( T - 1 ) \bigr ) .\tag{45}
$$

Thus, uniform certificates yield $\begin{array} { r l r } { d _ { t } } & { { } = } & { O ( 1 / t ) } \end{array}$ and $\mathbb { E } [ \mathrm { R e g } ( T ) ] = O ( \log T )$ ; otherwise, the guarantee remains horizon dependent. The next subsection identifies sparse and dense reward classes with analytic certificates.

## Explicit Verifiable Classes

To make Theorem 2 operational, we derive analytic certificates for two nontrivial classes of reward tables: sparse coordination tables admit a trajectory-invariant region that yields uniform rates, whereas dense locally dominant tables admit explicit finite-horizon certificates.

Corollary 1 (Optimally supported common rewards). Suppose there is a unique optimal joint action

$$
\pmb { a } ^ { \star } = ( A _ { 1 } ^ { \star } , \ldots , A _ { N } ^ { \star } )\tag{46}
$$

such that

$$
r ( { \pmb a } ^ { \star } ) = J ^ { \star } > 0 , \qquad r ( { \pmb a } ) = 0 \quad \forall { \pmb a } \neq { \pmb a } ^ { \star } .\tag{47}
$$

Let $c _ { \star } = 1 / K _ { \operatorname* { m a x } } .$ . For the trajectory of (6) initialized with $H _ { i , 1 } ( x ) = \mathrm { \dot { 0 } } ,$

$$
\pi _ { i , t } ( A _ { i } ^ { \star } ) \geq { \frac { 1 } { K _ { i } } } \quad f o r a l l i a n d t .\tag{48}
$$

Let $S _ { \mathrm { s p } } = \{ \pi : \pi _ { i } ( A _ { i } ^ { \star } ) \geq 1 / K _ { i } , \forall i \}$ and $M _ { \mathrm { s p } } = M _ { S _ { \mathrm { s p } } } .$ Every iterate lies in $ { S _ { \mathrm { s p } } }$ and assigns probability at least $c _ { \star } ^ { \bar { N } }$ to $\mathbf { \pmb { a } } ^ { \star }$ , yielding

$$
\lambda _ { \star } ( c _ { \star } ) \geq \frac { c _ { \star } ^ { 2 N } } { N ^ { 2 } } , \qquad M _ { \mathrm { s p } } \leq \frac { 1 } { J ^ { \star } c _ { \star } ^ { N } } .\tag{49}
$$

Consequently, for any constant stepsize

$$
0 < \eta \leq \frac { J ^ { \star } c _ { \star } ^ { N } } { L } ,\tag{50}
$$

update (6) satisfies $\begin{array} { r c l } { d _ { t } } & { = } & { { \cal O } ( 1 / t ) } \end{array}$ and E[Reg(T)] = ${ \bar { O } } ( \log T )$ .

The complete proof is given in the supplementary material. This corollary relies on the sparsity of the reward table. The following proposition removes this requirement for dense reward tables in which the optimal local arms dominate all alternatives under every opponent profile, thereby excluding the coordination reversal identified in Proposition 2.

Proposition 4 (Local dominance certificates). Suppose the optimal joint action $\pmb { a } ^ { \star } = ( A _ { 1 } ^ { \star } , \ldots , A _ { N } ^ { \star } )$ satisfies uniform local dominance: for some $\delta > 0 ,$ every player i, arm x ̸= $A _ { i } ^ { \star }$ , and opponent profile a<sub>−i</sub> obey $\cdot ( A _ { i } ^ { \star } , \bar { \bf { a } } _ { - i } ) - r ( x , { \bf { a } } _ { - i } ) \geq$ δ. Then, for every $\dot { c } \in ( 0 , 1 ] , \ \dot { \lambda } _ { \star } \dot { ( c ) } \geq \ ( \dot { c } \delta / N ) ^ { 2 } . \ J \stackrel { . } { c } \leq$ $1 / K _ { \operatorname* { m a x } }$ and at least one player has a non-optimal arm, define $\begin{array} { r } { \mu ( c ) = 1 - \prod _ { i } ( 1 - \bar { ( K _ { i } - 1 ) } c ) } \end{array}$ . Then

$$
M _ { \mathrm { a l l } } ( c ) \leq \frac { 2 N ^ { 3 } } { c ^ { 2 } \delta ^ { 4 } \mu ( c ) ^ { 2 } } .\tag{51}
$$

Thus every locally dominant dense table has explicit finitehorizon certificatesfor Theorem 2.

A complete proof of this proposition is provided in the supplementary material. Note that Proposition 4 covers dense, nonadditive reward tables. For example, if the normalized reward is $\begin{array} { r } { r ( { \mathbf a } ) = N ^ { - 1 } \sum _ { i } u _ { i } ( A _ { i } ) + \dot { \epsilon } h ( { \mathbf a } ) } \end{array}$ , each $u _ { i }$ has an optimal-arm gap of at least $\Delta$ , and all one-coordinate changes in h are bounded in absolute value by B, then $\Delta / N > | \check { \epsilon } | B$ yields a local-dominance margin $\delta = \Delta / N - | \epsilon | B$ . Thus, Corollary 1 and Proposition 4 avoid the block trap through complementary mechanisms: joint reinforcement in sparse tables and persistent local advantage in dense tables. Figure 1 illustrates both mechanisms, together with the trap that arises under oblivious feedback.

![](images/04905a715cfb87223db5f9b60b26af97296a5af25b58bfb4a0dd73931496d208.jpg)

![](images/0139e7f11f5e615eeec4bf85e2990c948427251b8eb69a404266ec9b7adac3c7.jpg)

![](images/4eb2ab3b0381e33bcb7dfe4aea91470db36778c89ee29d2915c795b2e27065c6.jpg)  
Figure 1: Sparse coordination (left), block trap (center), and AI routing (right); bands show 95% CIs.

## Mechanism Diagnostics

The diagnostics illustrate the theoretical mechanisms under the fully oblivious protocol rather than provide a comprehensive benchmark. The left panel of Figure 1 shows a decreasing product-policy gap for the sparse coordination class and its invariant support geometry. Because the practical stepsize exceeds the conservative range in Corollary 1, this trajectory diagnoses the mechanism rather than verifies the certified constants. The center panel shows the negative case: from uniform initialization, the optimal-profile probability remains small while the suboptimal block dominates, consistent with Proposition 2. The right panel presents a dense, nonadditive AI-pipeline case with an analytic localdominance certificate; its four modules observe only one terminal score. Supplementary diagnostics show that certificate estimates are associated with terminal gaps and that the trap basin expands as the suboptimal block reward approaches the optimum.

## Related Work

Multi-player and cooperative bandits. Multi-player bandits study collisions, decentralized allocation, and limited communication (Anantharam, Varaiya, and Walrand 1987; Liu and Zhao 2010; Anandkumar et al. 2011; Kalathil, Nayyar, and Jain 2014; Rosenski, Shamir, and Szlak 2016; Besson and Kaufmann 2018; Boursier and Perchet 2019; Bistritz and Leshem 2018). Cooperative and combinatorial bandits use communication, aggregation, centralization, or component feedback (Bistritz and Bambos 2020; Gai, Krishnamachari, and Jain 2012; Chen, Wang, and Yuan 2013; Kveton et al. 2015). Unlike these settings, OC-MPMAB provides neither coordination signals nor awareness ofother learners; we characterize the hidden product-policy geometry induced by independent updates driven solely by a shared scalar reward.

Common-interest games. Product-policy traps are related to common-payof and potential games (Monderer and Shapley 1996; Marden et al. 2009), learning automata and payof-based dynamics (Narendra and Thathachar 1989; Leslie and Collins 2005; Mertikopoulos and Sandholm 2016), and independent multi-agent reinforcement learning coordination failures (Claus and Boutilier 1998; Lauer and

Riedmiller 2000; Littman 1994; Hu and Wellman 2003; Shoham, Powers, and Grenager 2007; Busoniu, Babuska, and De Schutter 2008). Unlike equilibrium-selection analyses with a known game form or prescribed dynamics, we characterize stationarity, traps, and finite-table certificates when learners observe neither joint actions, player identities, nor the game model.

Policy gradients and gradient bandits. Likelihood-ratio and policy-gradient identities are classical (Glynn 1990; Williams 1992; Sutton et al. 1999; Baxter and Bartlett 2001; Konda and Tsitsiklis 2003); baselines and variance reduction are well studied (Greensmith, Bartlett, and Baxter 2004; Mei et al. 2022). Softmax policy-gradient theory mostly treats single-agent or centralized policies (Mei et al. 2020b,a; Li et al. 2021; Agarwal et al. 2021; Mei et al. 2023). Unlike the single-player case, hidden product coupling can make the component A⋆ disadvantageous because Qπ(A⋆) averages over the policies of other players. We characterize the resulting traps and verifiable conditions for global rates, rather than deriving another score-function identity.

## Conclusion

This paper presents a theoretical analysis of OC-MPMAB, a minimal multi-player bandit model in which fully oblivious players observe only their selected arms and a shared reward. Our findings are threefold. First, independent gradient-bandit updates jointly form an unbiased stochastic gradient estimator of the hidden product-policy objective, which guarantees universal finite-time convergence to first-order stationary points. Second, first-order stationarity cannot generally be strengthened to global optimality: product-policy traps can induce suboptimal attractors, and a deterministic block trap can lead to failure with constant probability from the default initialization under suficiently small diminishing stepsizes. Third, structural properties of the reward table can recover global optimality guarantees. Specifically, finite-table certificates for product-gradient dominance and self-bounding stochastic noise provide certificate-dependent bounds on the product-policy gap and pseudo-regret; uniform certificates further achieve an O(1/t) gap and O(log T) pseudo-regret.

## References

Agarwal, A.; Kakade, S. M.; Lee, J. D.; and Mahajan, G. 2021. On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift. Journal of Machine Learning Research, 22(98): 1–76.

Agrawal, S.; and Goyal, N. 2012. Analysis of Thompson Sampling for the Multi-Armed Bandit Problem. In Conference on Learning Theory, 39.1–39.26.

Anandkumar, A.; Michael, N.; Tang, A. K.; and Swami, A. 2011. Distributed Algorithms for Learning and Cognitive Medium Access with Logarithmic Regret. IEEE Journal on Selected Areas in Communications, 29(4): 731–745.

Anantharam, V.; Varaiya, P.; and Walrand, J. 1987. Asymptotically Eficient Allocation Rules for the Multiarmed Bandit Problem with Multiple Plays-Part I: I.I.D. Rewards. IEEE Transactions on Automatic Control, 32(11): 968–976.

Auer, P.; Cesa-Bianchi, N.; and Fischer, P. 2002. Finite-Time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47(2–3): 235–256.

Baxter, J.; and Bartlett, P. L. 2001. Infinite-Horizon Policy-Gradient Estimation. Journal of Artificial Intelligence Research, 15: 319–350.

Besson, L.; and Kaufmann, E. 2018. Multi-Player Bandits Revisited. In Algorithmic Learning Theory, 56–92.

Bistritz, I.; and Bambos, N. 2020. Cooperative Multi-Player Bandit Optimization. In Advances in Neural Information Processing Systems, volume 33, 11518–11528.

Bistritz, I.; and Leshem, A. 2018. Distributed Multi-Player Bandits: A Game of Thrones Approach. In Advances in Neural Information Processing Systems, 7222–7232.

Boursier, E.; and Perchet, V. 2019. SIC-MMAB: Synchronisation Involves Communication in Multiplayer Multi-Armed Bandits. In Advances in Neural Information Processing Systems, 12048–12057.

Busoniu, L.; Babuska, R.; and De Schutter, B. 2008. A Comprehensive Survey of Multiagent Reinforcement Learning. IEEE Transactions on Systems, Man, and Cybernetics, Part C (Applications and Reviews), 38(2): 156–172.

Chen, W.; Wang, Y.; and Yuan, Y. 2013. Combinatorial Multi-Armed Bandit: General Framework and Applications. In International Conference on Machine Learning, 151–159.

Claus, C.; and Boutilier, C. 1998. The Dynamics of Reinforcement Learning in Cooperative Multiagent Systems. In AAAI/IAAI, 746–752.

Gai, Y.; Krishnamachari, B.; and Jain, R. 2012. Combinatorial Network Optimization with Unknown Variables: Multi-Armed Bandits with Linear Rewards. IEEE/ACM Transactions on Networking, 20(5): 1466–1478.

Glynn, P. W. 1990. Likelihood Ratio Gradient Estimation for Stochastic Systems. Communications of the ACM, 33(10): 75–84.

Greensmith, E.; Bartlett, P. L.; and Baxter, J. 2004. Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning. Journal of Machine Learning Research, 5: 1471–1530.

Hu, J.; and Wellman, M. P. 2003. Nash Q-Learning for General-Sum Stochastic Games. Journal ofMachine Learning Research, 4: 1039–1069.

Kalathil, D.; Nayyar, N.; and Jain, R. 2014. Decentralized Learning for Multiplayer Multiarmed Bandits. IEEE Transactions on Information Theory, 60(4): 2331–2345.

Konda, V. R.; and Tsitsiklis, J. N. 2003. On Actor-Critic Algorithms. SIAM Journal on Control and Optimization, 42(4): 1143–1166.

Kveton, B.; Wen, Z.; Ashkan, A.; and Szepesvari, C. 2015. Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits. In International Conference on Artificial Intelli gence and Statistics, 535–543.

Lai, T. L.; and Robbins, H. 1985. Asymptotically Eficient Adaptive Allocation Rules. Advances in Applied Mathematics, 6(1): 4–22.

Lattimore, T.; and Szepesvari, C. 2020. Bandit Algorithms. Cambridge University Press.

Lauer, M.; and Riedmiller, M. 2000. An Algorithm for Distributed Reinforcement Learning in Cooperative Multi-Agent Systems. In International Conference on Machine Learning, 535–542.

Leslie, D. S.; and Collins, E. J. 2005. Individual Q-Learning in Normal Form Games. SIAM Journal on Control and Optimization, 44(2): 495–514.

Li, G.; Wei, Y.; Chi, Y.; Gu, Y.; and Chen, Y. 2021. Softmax Policy Gradient Methods Can Take Exponential Time to Converge. In Conference on Learning Theory, 3107–3110.

Littman, M. L. 1994. Markov Games as a Framework for Multi-Agent Reinforcement Learning. In International Conference on Machine Learning, 157–163.

Liu, K.; and Zhao, Q. 2010. Distributed Learning in Multi-Armed Bandit with Multiple Players. IEEE Transactions on Signal Processing, 58(11): 5667–5681.

Marden, J. R.; Young, H. P.; Arslan, G.; and Shamma, J. S. 2009. Payof-Based Dynamics for Multiplayer Weakly Acyclic Games. SIAM Journal on Control and Optimization, 48(1): 373–396.

Mei, J.; Chung, W.; Thomas, V.; Dai, B.; Szepesvari, C.; and Schuurmans, D. 2022. The Role of Baselines in Policy Gradient Optimization. In Advances in Neural Information Processing Systems.

Mei, J.; Xiao, C.; Dai, B.; Li, L.; Szepesvari, C.; and Schuurmans, D. 2020a. Escaping the Gravitational Pull of Softmax. In Advances in Neural Information Processing Systems, vol ume 33, 21130–21140.

Mei, J.; Xiao, C.; Szepesvari, C.; and Schuurmans, D. 2020b. On the Global Convergence Rates of Softmax Policy Gradient Methods. In International Conference on Machine Learning, 6820–6829.

Mei, J.; Zhong, Z.; Dai, B.; Agarwal, A.; Szepesvari, C.; and Schuurmans, D. 2023. Stochastic Gradient Succeeds for Bandits. In International Conference on Machine Learning, 24314–24345.

Mertikopoulos, P.; and Sandholm, W. H. 2016. Learning in Games via Reinforcement and Regularization. Mathematics ofOperations Research, 41(4): 1297–1324.

Monderer, D.; and Shapley, L. S. 1996. Potential Games. Games and Economic Behavior, 14(1): 124–143.

Narendra, K. S.; and Thathachar, M. A. L. 1989. Learning Automata: An Introduction. Prentice Hall.

Robbins, H.; and Monro, S. 1951. A Stochastic Approximation Method. Annals of Mathematical Statistics, 22(3): 400–407.

Rosenski, J.; Shamir, O.; and Szlak, L. 2016. Multi-Player Bandits: A Musical Chairs Approach. In International Conference on Machine Learning, 155–163.

Shoham, Y.; Powers, R.; and Grenager, T. 2007. If Multi-Agent Learning Is the Answer, What Is the Question? Artificial Intelligence, 171(7): 365–377.

Sutton, R. S.; and Barto, A. G. 2018. Reinforcement Learning: An Introduction. MIT Press, 2 edition.

Sutton, R. S.; McAllester, D.; Singh, S.; and Mansour, Y. 1999. Policy Gradient Methods for Reinforcement Learning with Function Approximation. In Advances in Neural Information Processing Systems, volume 12, 1057–1063.

Thompson, W. R. 1933. On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples. Biometrika, 25(3/4): 285–294.

Williams, R. J. 1992. Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 8(3–4): 229–256.

# Supplementary Material for Characterizing Oblivious Common-Reward Multi-Player Bandits: Universal Stationarity, Coordination Traps, and Certified Global Rates

Mingxiao Li, Wenhai Lai, and Hongyu Wang

## Omitted Proofs

Unless redefined locally, all model, policy, and certificate notation follows the main text.

## Proof of Main-Text Proposition 1

Proof. Fix an arbitrary player i and arm $x \in A _ { i }$ , and condition on $\mathcal { F } _ { t }$ . Because only the local factor $\pi _ { i } ( A _ { i } )$ depends on $H _ { i } ( x )$

$$
\frac { \partial J ( \mathbf { H } ) } { \partial H _ { i } ( x ) } = \sum _ { \pmb { a \in A } } r ( \pmb { a } ) \left( \prod _ { j \neq i } \pi _ { j } ( A _ { j } ) \right) \frac { \partial \pi _ { i } ( A _ { i } ) } { \partial H _ { i } ( x ) } .\tag{1}
$$

Let $Z _ { i }$ denote the softmax denominator. We then have

$$
\pi _ { i } ( A _ { i } ) = \frac { \exp ( H _ { i } ( A _ { i } ) ) } { Z _ { i } } ,\tag{2}
$$

and the quotient rule yields

$$
\frac { \partial \pi _ { i } ( A _ { i } ) } { \partial H _ { i } ( x ) } = \pi _ { i } ( A _ { i } ) \big ( { \bf 1 } \{ A _ { i } = x \} - \pi _ { i } ( x ) \big ) .\tag{3}
$$

Thus

$$
\frac { \partial J ( \mathbf { H } ) } { \partial H _ { i } ( x ) } = \sum _ { \pmb { a } \in \mathcal { A } } \pi _ { \mathbf { H } } ( \pmb { a } ) r ( \pmb { a } ) \big ( \mathbf { 1 } \{ A _ { i } = x \} - \pi _ { i } ( x ) \big ) .\tag{4}
$$

Meanwhile,

$$
\begin{array} { r l } & { \mathbb { E } [ g _ { i , t } ( x ) \mid \mathcal { F } _ { t } ] = \mathbb { E } \big [ R _ { t } \big ( \mathbf { 1 } \{ A _ { i , t } = x \} - \pi _ { i , t } ( x ) \big ) \big ] } \\ & { \qquad = \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \mathbf { H } _ { t } } ( a ) r ( a ) \big ( \mathbf { 1 } \{ A _ { i } = x \} - \pi _ { i , t } ( x ) \big ) , } \end{array}\tag{5}
$$

which is precisely the derivative evaluated at $\mathbf { H } = \mathbf { H } _ { t }$ □

## Proof of Main-Text Theorem 1

Proof. Let $\mathcal { F } _ { t }$ denote the history before round $t ,$ and write

$$
\begin{array} { r } { \mathbf { g } _ { t } = \left( g _ { i , t } ( x ) : i \in [ N ] , x \in \mathcal { A } _ { i } \right) \in \mathbb { R } ^ { d } . } \end{array}\tag{6}
$$

Here $d = \textstyle \sum _ { i } K _ { i }$ , and we set $\Delta _ { 1 } = J ^ { \star } { - } J ( \mathbf { H } _ { 1 } )$ . Conditional on $\mathcal { F } _ { t }$ , Main-Text Proposition 1 gives

$$
\mathbb { E } [ { \mathbf { g } } _ { t } \mid { \mathcal { F } } _ { t } ] = \nabla J ( { \mathbf { H } } _ { t } ) .\tag{7}
$$

For each player, the centered score vector

$$
s _ { i , t } ( x ) = \mathbf { 1 } \{ A _ { i , t } = x \} - \pi _ { i , t } ( x )\tag{8}
$$

satisfies $\| s _ { i , t } \| _ { 2 } ^ { 2 } \leq 2 .$ Since $R _ { t } \in [ 0 , 1 ]$ 2

$$
\| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { N } R _ { t } ^ { 2 } \| s _ { i , t } \| _ { 2 } ^ { 2 } \leq 2 N = G ^ { 2 } .\tag{9}
$$

Applying Main-Text Lemma 1 to $\mathbf { H } _ { t + 1 } = \mathbf { H } _ { t } + \gamma _ { t } \mathbf { g } _ { t }$ yields

$$
J ( \mathbf { H } _ { t + 1 } ) \geq J ( \mathbf { H } _ { t } ) + \gamma _ { t } \langle \nabla J ( \mathbf { H } _ { t } ) , \mathbf { g } _ { t } \rangle - \frac { L } { 2 } \gamma _ { t } ^ { 2 } \Vert \mathbf { g } _ { t } \Vert _ { 2 } ^ { 2 } .\tag{10}
$$

Taking conditional expectation yields

$$
\mathbb { E } [ J ( \mathbf { H } _ { t + 1 } ) \mid \mathcal { F } _ { t } ] \geq J ( \mathbf { H } _ { t } ) + \gamma _ { t } \Vert \nabla J ( \mathbf { H } _ { t } ) \Vert _ { 2 } ^ { 2 } - \frac { L } { 2 } \gamma _ { t } ^ { 2 } G ^ { 2 } .\tag{11}
$$

Rearranging, taking total expectations, and summing over $t = 1 , \ldots , \bar { T }$ yields

$$
\begin{array} { r l r } {  { \sum _ { t = 1 } ^ { T } \gamma _ { t } \mathbb { E } \big [ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \big ] } \le \mathbb { E } \big [ J ( \mathbf { H } _ { T + 1 } ) \big ] - J ( \mathbf { H } _ { 1 } ) }  \\ & { } & { \qquad + \frac { L G ^ { 2 } } { 2 } \displaystyle \sum _ { t = 1 } ^ { T } \gamma _ { t } ^ { 2 } . } \end{array}\tag{12}
$$

Since $J ( \mathbf { H } ) ~ \le ~ J ^ { \star }$ , dividing by $\Gamma _ { T }$ proves the weightedaverage bound. The minimum bound follows because the minimum is no greater than the weighted average. For the constant stepsize $\gamma _ { t } = \eta$ , substitution gives

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \big [ \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } \big ] \leq \frac { \Delta _ { 1 } } { \eta T } + \frac { L \eta G ^ { 2 } } { 2 } .\tag{13}
$$

When $\Delta _ { 1 } > 0$ , the stated choice of η balances the two terms. If $\Delta _ { 1 } = 0 $ , the full support of the softmax policy implies that every joint action has mean reward $J ^ { \star }$ , so $\nabla J$ is identically zero. Finally, $G ^ { 2 } = 2 N , L = 2 d ,$ and $0 \leq \Delta _ { 1 } \leq 1$ □

## Proof of Main-Text Proposition 2

Proof. For the two-player, three-arm block-trap table in Main-Text Proposition 2, let

$$
q _ { i } = \pi _ { i } ( 2 ) , \qquad i = 1 , 2 .\tag{14}
$$

The objective depends on the preferences only through $\left( q _ { 1 } , q _ { 2 } \right)$ . Define its reduced form by

$$
J _ { \mathrm { t r a p } } ( q _ { 1 } , q _ { 2 } ) : = J ( { \bf H } ) = 0 . 9 ( 1 - q _ { 1 } ) ( 1 - q _ { 2 } ) + q _ { 1 } q _ { 2 } .\tag{15}
$$

The two non-optimal arms of each player form a reward-0.9 block. Thus, only the total probability $q _ { i }$ assigned to the optimal arm matters in this two-dimensional reduction. For player 1, the conditional value of arm 2 is $q _ { 2 } .$ , whereas each arm in the non-optimal block has conditional value $0 . 9 ( 1 -$ $q _ { 2 } )$ . Therefore, the product-gradient direction that increases the probability of the optimal arm of player 1 relative to the non-optimal block has the sign of

$$
q _ { 2 } - 0 . 9 ( 1 - q _ { 2 } ) = 1 . 9 q _ { 2 } - 0 . 9 .\tag{16}
$$

Similarly, the corresponding direction for player 2 has the sign of $1 . 9 q _ { 1 } - 0 . 9$ . Hence, the separating interior stationary point in the reduced $( q _ { 1 } , q _ { 2 } )$ coordinates is

$$
q _ { 1 } = q _ { 2 } = { \frac { 0 . 9 } { 1 . 9 } } = { \frac { 9 } { 1 9 } } .\tag{17}
$$

In the full product-softmax parameterization, this point corresponds to a stationary set because the reward table is invariant to how each player distributes the remaining probability $1 - q _ { i }$ between arms 0 and 1. At any policy in this stationary set,

$$
J = 0 . 9 \left( { \frac { 1 0 } { 1 9 } } \right) ^ { 2 } + \left( { \frac { 9 } { 1 9 } } \right) ^ { 2 } = { \frac { 1 7 1 } { 3 6 1 } } < 1 = J ^ { \star } .\tag{18}
$$

Thus, product-gradient dominance fails for every $c \in$ $( 0 , 9 / 1 { \bar { 9 } } ]$

For completeness, the exact policy-space flow makes the boundary behavior explicit. Under $\dot { \mathbf H } = \nabla J ( \mathbf H )$ , the softmax chain rule yields, for $j \neq i$

$$
\dot { q } _ { i } = q _ { i } ^ { 2 } \left( ( 1 - q _ { i } ) ^ { 2 } + \pi _ { i } ( 0 ) ^ { 2 } + \pi _ { i } ( 1 ) ^ { 2 } \right) ( 1 . 9 q _ { j } - 0 . 9 ) .\tag{19}
$$

Thus, if $q _ { 1 } < 9 / 1 9$ and $q _ { 2 } < 9 / 1 9$ , both optimal-arm probabilities decrease, and the lower-left region is forward invariant. Each $q _ { i }$ therefore has a limit. If either limit were positive, the displayed derivative would remain bounded above by a negative constant because the other probability never increases, yielding a contradiction. Hence, $q _ { 1 } , q _ { 2 } \to 0$ . The default initialization satisfies $q _ { 1 } = q _ { 2 } = 1 / 3 < 9 / 1 9$ and therefore lies in this attracting basin. □

## Proof of Main-Text Proposition 3

Proof. Supplementary Proposition 3 establishes an open attracting basin for a strict trap minor, while Supplementary Proposition 4 establishes stochastic lock-in on compact subsets of that basin. Together, these results prove Main-Text Proposition 3. □

## Proof of the Main-Text Default-Initialization Block-Trap Statement

Proof. The probability bound under default initialization and the corresponding asymptotic lower bound on the expected gap follow from Supplementary Proposition 1. □

## Proof of Main-Text Theorem 2

Proof. By Main-Text Lemma 2, $\pi _ { \mathbf { H } _ { t } } ~ \in ~ \Pi _ { \mathrm { a l l } } ( c _ { T } )$ for all $t \leq T + 1$ , and hence $\pi _ { \mathbf { H } _ { t } } ~ \in ~ \Pi _ { \star } ( c _ { T } )$ . When $D _ { t } ~ > ~ 0 .$ the definition of $\lambda _ { T } = \lambda _ { \star } ( c _ { T } )$ gives

$$
\| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } = \| \mathcal { G } ( \pi _ { \mathbf { H } _ { t } } ) \| _ { 2 } ^ { 2 } \geq \lambda _ { T } D _ { t } ^ { 2 } .\tag{20}
$$

The second-moment envelope, together with $\begin{array} { r l } { M _ { T } } & { { } = } \end{array}$ $M _ { \mathrm { a l l } } ( c _ { T } )$ , gives

$$
\mathbb { E } [ \| \mathbf { g } _ { t } \| _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } ] \le M _ { T } \| \nabla J ( \mathbf { H } _ { t } ) \| _ { 2 } ^ { 2 } .\tag{21}
$$

Applying Main-Text Lemma 1 and using the conditional unbiasedness from Main-Text Proposition 1 and the definition of $\rho _ { t } ,$ , we obtain

$$
\begin{array} { r l } & { \mathbb { E } [ J ( \mathbf { H } _ { t + 1 } ) - J ( \mathbf { H } _ { t } ) \mid \mathcal { F } _ { t } ] \geq \gamma _ { t } \Vert \nabla J ( \mathbf { H } _ { t } ) \Vert _ { 2 } ^ { 2 } } \\ & { \qquad - \displaystyle \frac { L } { 2 } \gamma _ { t } ^ { 2 } \mathbb { E } [ \Vert \mathbf { g } _ { t } \Vert _ { 2 } ^ { 2 } \mid \mathcal { F } _ { t } ] } \\ & { \qquad \geq \rho _ { t } \Vert \nabla J ( \mathbf { H } _ { t } ) \Vert _ { 2 } ^ { 2 } \geq \rho _ { t } \lambda _ { T } D _ { t } ^ { 2 } . } \end{array}\tag{22}
$$

Thus

$$
\mathbb { E } [ D _ { t + 1 } \mid \mathcal { F } _ { t } ] \leq D _ { t } - \rho _ { t } \lambda _ { T } D _ { t } ^ { 2 } .\tag{23}
$$

Taking expectations and applying Jensen’s inequality gives

$$
d _ { t + 1 } \leq d _ { t } - \rho _ { t } \lambda _ { T } d _ { t } ^ { 2 } .\tag{24}
$$

If $d _ { t + 1 } = 0 ,$ the reciprocal bound follows immediately. Otherwise, the sequence $\{ d _ { t } \}$ is nonincreasing, so

$$
\frac { 1 } { d _ { t + 1 } } - \frac { 1 } { d _ { t } } \geq \frac { d _ { t } - d _ { t + 1 } } { d _ { t } ^ { 2 } } \geq \rho _ { t } \lambda _ { T } .\tag{25}
$$

Because the zero-preference initialization is deterministic, $d _ { 1 } = D _ { 1 }$ . Summing from 1 to $t - 1$ proves the gap bound. For the pseudo-regret,

$$
\mathbb { E } [ J ^ { \star } - r ( \pmb { a } _ { t } ) \ | \ \mathcal { F } _ { t } ] = J ^ { \star } - J ( \pi _ { \mathbf { H } _ { t } } ) = D _ { t } .\tag{26}
$$

Taking expectations and summing yields

$$
\mathbb { E } [ \mathrm { R e g } ( T ) ] = \sum _ { t = 1 } ^ { T } d _ { t } ,\tag{27}
$$

and substituting the gap bound proves the displayed pseudoregret bound. □

## Proof of Main-Text Corollary 1

Proof. Because rewards lie in [0, 1], the condition $r ( { \pmb a } ) = 0$ implies that $R _ { t } = 0$ almost surely whenever $\mathbf { a } _ { t } \neq a ^ { \star }$ . No preference changes in such rounds. If $\mathbf { \boldsymbol { a } } _ { t } = \mathbf { \boldsymbol { a } } ^ { \star }$ , then, for any $x \neq A _ { i } ^ { \star }$

$$
g _ { i , t } \big ( A _ { i } ^ { \star } \big ) - g _ { i , t } \big ( x \big ) = R _ { t } \big ( 1 - \pi _ { i , t } \big ( A _ { i } ^ { \star } \big ) + \pi _ { i , t } ( x ) \big ) \geq 0 .\tag{28}
$$

Thus, $H _ { i , t } ( A _ { i } ^ { \star } ) - H _ { i , t } ( x )$ never decreases from its initial value 0, which implies $\pi _ { i , t } ( A _ { i } ^ { \star } ) \geq 1 / K _ { i }$

Fix any $\pi \in \Pi _ { \star } ( c _ { \star } )$ , and let

$$
p _ { i } = \pi _ { i } ( A _ { i } ^ { \star } ) , \qquad p _ { \star } = \prod _ { i = 1 } ^ { N } p _ { i } .\tag{29}
$$

Then $J ( \pi ) = J ^ { \star } p .$ <sub>⋆</sub> and, writing $D ( \pi ) = J ^ { \star } { - } J ( \pi ) , D ( \pi ) =$ $J ^ { \star } ( 1 - p _ { \star } )$ . The optimal-arm gradient coordinate is

$$
\mathcal { G } _ { i , A _ { i } ^ { \star } } ( \pi ) = J ^ { \star } p _ { \star } ( 1 - p _ { i } ) .\tag{30}
$$

Since $\begin{array} { r } { 1 - p _ { \star } \le \sum _ { i } ( 1 - p _ { i } ) } \end{array}$ , there exists a player such that

$$
1 - p _ { i } \ge \frac { D ( \pi ) } { J ^ { \star } N } .\tag{31}
$$

Using $p _ { \star } \geq c _ { \star } ^ { N }$

$$
\| \mathcal { G } ( \pi ) \| _ { 2 } \ge J ^ { \star } p _ { \star } ( 1 - p _ { i } ) \ge \frac { c _ { \star } ^ { N } } { N } D ( \pi ) ,\tag{32}
$$

which proves the lower bound on $\lambda _ { \star } ( c _ { \star } )$

For the noise certificate on $\begin{array} { r } {  { \boldsymbol { S } } _ { \mathrm { s p } } , } \end{array}$

$$
\mathcal { G } _ { i } ( \pi ) = J ^ { \star } p _ { \star } ( \mathbf { e } _ { A _ { i } ^ { \star } } - \pi _ { i } ) ,\tag{33}
$$

so

$$
\| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } = ( J ^ { \star } ) ^ { 2 } p _ { \star } ^ { 2 } \sum _ { i } \| \mathbf { e } _ { A _ { i } ^ { \star } } - \boldsymbol { \pi } _ { i } \| _ { 2 } ^ { 2 } .\tag{34}
$$

The same sparse reward structure gives

$$
\mathcal { V } ( \pi ) = J ^ { \star } p _ { \star } \sum _ { i } \| \mathbf { e } _ { A _ { i } ^ { \star } } - \boldsymbol { \pi } _ { i } \| _ { 2 } ^ { 2 } .\tag{35}
$$

Whenever $\| \mathcal { G } ( \pi ) \| _ { 2 } > 0$

$$
\frac { \mathcal { V } ( \pi ) } { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } } = \frac { 1 } { J ^ { \star } p _ { \star } } \leq \frac { 1 } { J ^ { \star } c _ { \star } ^ { N } } .\tag{36}
$$

If $\mathcal { G } ( \pi ) = 0$ on $ { \mathcal { S } } _ { \mathrm { s p } }$ , the displayed gradient formula forces every $\pi _ { i }$ to be concentrated on $A _ { i } ^ { \star }$ , and hence $\nu ( \pi ) = 0$ Thus the zero-gradient convention for $M _ { \mathrm { s p } }$ is also satisfied. The rate statement follows by applying the same one-step recursion as in Main-Text Theorem 2 within this invariant region, using constants that are independent of the horizon. 口

## Proof of Main-Text Proposition 4

Proof. Let $D ( \pi ) = J ^ { \star } { - } J ( \pi )$ and $q _ { i } = 1 { - } \pi _ { i } ( A _ { i } ^ { \star } )$ . Because the rewards are normalized and $\mathbf { \delta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \beta } _  \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathrm { \bf ~ \textit ~ { ~ a ~ } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha } \mathbf { \alpha \alpha } }$ is optimal,

$$
D ( \pi ) = J ^ { \star } - J ( \pi ) \leq \operatorname* { P r } _ { a \sim \pi } \{ a \neq a ^ { \star } \} \leq \sum _ { i = 1 } ^ { N } q _ { i } .\tag{37}
$$

Thus, some player i satisfies $q _ { i } \geq D ( \pi ) / N$ . For this player,

$$
J ( \pi ) = \sum _ { x \in \mathcal { A } _ { i } } \pi _ { i } ( x ) Q _ { i } ^ { \pi } ( x ) ,\tag{38}
$$

and the local-dominance condition implies

$$
Q _ { i } ^ { \pi } ( A _ { i } ^ { \star } ) - J ( \pi ) \geq \delta q _ { i } .\tag{39}
$$

If $\pi \in \Pi _ { \star } ( c )$ , then

$$
\mathcal { G } _ { i , A _ { i } ^ { \star } } ( \pi ) = \pi _ { i } ( A _ { i } ^ { \star } ) \big ( Q _ { i } ^ { \pi } ( A _ { i } ^ { \star } ) - J ( \pi ) \big ) \geq c \delta q _ { i } \geq \frac { c \delta } { N } D ( \pi ) .\tag{40}
$$

The claim follows by taking the Euclidean norm.

It remains to bound the baseline-free noise ratio on $\Pi _ { \mathrm { a l l } } ( c )$ For any joint action $\mathbf { \pmb { a } } \neq \mathbf { \pmb { a } } ^ { \star }$ , change each non-optimal coordinate of a to its optimal arm one at a time. Each such change

increases the reward by at least δ. Hence, $J ^ { \star } - r ( a ) \geq \delta$ whenever $\textbf { \em a } \neq a ^ { \star }$ . Therefore, for any product policy in $\Pi _ { \mathrm { a l l } } ( c )$

$$
D ( \pi ) = J ^ { \star } - J ( \pi ) \geq \delta \operatorname* { P r } _ { a \sim \pi } \{ a \neq a ^ { \star } \} .\tag{41}
$$

Since $\pi _ { i } ( x ) \geq c$ for every local arm,

$$
\begin{array} { r l } { \underset { a \sim \pi } { \operatorname* { P r } } \left\{ a \neq a ^ { \star } \right\} = 1 - \underset { i } { \prod _ { i } } \pi _ { i } ( A _ { i } ^ { \star } ) } & { } \\ { \geq 1 - \underset { i } { \prod } \big ( 1 - ( K _ { i } - 1 ) c \big ) = \mu ( c ) . } \end{array}\tag{42}
$$

Thus, $D ( \pi ) \geq \delta \mu ( c )$ on $\Pi _ { \mathrm { a l l } } ( c )$ . Combining this lower bound with the product-gradient dominance bound gives

$$
\| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } \geq \left( \frac { c \delta } { N } \right) ^ { 2 } D ( \pi ) ^ { 2 } \geq \frac { c ^ { 2 } \delta ^ { 4 } \mu ( c ) ^ { 2 } } { N ^ { 2 } } .\tag{43}
$$

Finally, because $R _ { t } \in [ 0 , 1 ]$ and $\| \mathbf { e } _ { A _ { i } } - \pi _ { i } \| _ { 2 } ^ { 2 } \leq 2$

$$
\begin{array} { r } { \mathcal { V } ( \pi ) \leq 2 N . } \end{array}\tag{44}
$$

The displayed bound on $M _ { \mathrm { a l l } } ( c )$ follows.

## Supplementary Theoretical Results

This section presents additional theoretical results that strengthen or extend the main-text analysis.

## Constant-Probability Stochastic Failure for the Block Trap

Main-Text Proposition 2 establishes a deterministic productgradient basin for the block-trap table. The next result provides a stochastic counterpart under the same zero-preference initialization: with a standard diminishing stepsize schedule and a suficiently small initial scale, the failure probability of the baseline-free stochastic process is bounded below by a positive constant that is independent of the horizon. This result complements the finite-horizon constant-stepsize simulations, which illustrate the basin geometry but do not constitute an almost-sure convergence theorem.

Supplementary Lemma 1 (Policy-space lock-in). Let $( \mathcal { F } _ { t } )$ be a filtration and let $X _ { t }$ be an $\dot { \mathcal { F } } _ { t } .$ -adapted stochastic approximation on a compact set $\mathcal { X } \subset \mathbb { R } ^ { m }$

$$
X _ { t + 1 } = X _ { t } + \gamma _ { t } \big ( F ( X _ { t } ) + U _ { t + 1 } \big ) + \xi _ { t + 1 } ,\tag{45}
$$

where F is Lipschitz, $U _ { t + 1 } i s \mathcal { F } _ { t + 1 }$ -measurable with $\mathbb { E } [ U _ { t + 1 } \mid$ $\mathcal { F } _ { t } ] = 0$ and $\| U _ { t + 1 } \| _ { 2 } \leq B ,$ , and the F<sub>t+1</sub>-measurable remainder satisfies $\| \xi _ { t + 1 } ^ { \cdot } \| _ { 2 } \leq B _ { r } \gamma _ { t } ^ { 2 }$ . Assume that $\{ \gamma _ { t } \}$ is deterministic and nonnegative, $\textstyle \sum _ { t } \gamma _ { t } = \infty ,$ , and $\begin{array} { r } { \sum _ { t } \gamma _ { t } ^ { 2 } < \infty . } \end{array}$ If T is a compact asymptotically stable attractor of $\dot { X } =$ ${ \dot { F } } ( X )$ ) and K is a compact subset ofits basin, then constants $a _ { K } , b _ { K } > 0$ and $C _ { K } <$ ∞ exist such that

$$
\operatorname* { P r } \{ \operatorname { d i s t } ( X _ { t } , { \mathcal { T } } ) \to 0 \} \geq \left[ 1 - C _ { K } \sum _ { t = 1 } ^ { \infty } \gamma _ { t } ^ { 2 } \right] _ { + }\tag{46}
$$

where $[ z ] _ { + } = \operatorname* { m a x } \{ z , 0 \}$ . This bound holds whenever $X _ { 1 } \in$ $K ,$ ma $\mathfrak { r } _ { t } \gamma _ { t } \le a _ { K }$ , and $\begin{array} { r } { \sum _ { t } \gamma _ { t } ^ { 2 } \le b _ { K } } \end{array}$

Proof. Let $\Phi _ { s }$ denote the flow of ${ \dot { X } } = F ( X )$ . Asymptotic stability provides nested neighborhoods

$$
\mathcal { T } \subset U _ { 0 } , \qquad \overline { { U } } _ { 0 } \subset U _ { 1 } , \qquad \overline { { U } } _ { 1 } \subset U _ { 2 } , \qquad \overline { { U } } _ { 2 } \subset B ( \mathcal { T } ) ,\tag{47}
$$

where $U _ { 2 }$ can be chosen as a positively invariant attracting neighborhood of $\tau$ . Compactness of K and attraction imply that there is a finite $s _ { K }$ such that $\Phi _ { s _ { K } } ( K ) \subset U _ { 0 }$ . The flow tube

$$
\mathcal { C } _ { K } = \{ \Phi _ { s } ( x ) : x \in K , 0 \leq s \leq s _ { K } \}\tag{48}
$$

is compact and lies inside the basin. Hence, the tube and the nested neighborhood boundaries have a positive robustness margin $\varepsilon _ { K }$

Consider the afine interpolation of the recursion in ODE time. On every interval of bounded ODE length, the Lipschitz property and Gronwall’s inequality bound its distance from the corresponding flow segment by a constant times the oscillation of the cumulative perturbation on that interval, plus the Euler and remainder errors. The total nonmartingale error is at most

$$
C _ { F } \sum _ { t } \gamma _ { t } ^ { 2 } + B _ { r } \sum _ { t } \gamma _ { t } ^ { 2 } .\tag{49}
$$

After reducing $a _ { K }$ and $b _ { K }$ , this quantity is below $\varepsilon _ { K } / 4$ Thus, if every martingale increment over a flow-time window is below $\varepsilon _ { K } / 2 .$ , the interpolation follows $\mathcal { C } _ { K }$ into $U _ { 1 }$ and remains in $U _ { 2 }$

Let

$$
\mathsf { M } _ { n } = \sum _ { t = 1 } ^ { n } \gamma _ { t } U _ { t + 1 } .\tag{50}
$$

The martingale-diference property, boundedness, and $\sum _ { t } \gamma _ { t } ^ { 2 } <$ ∞ imply that $\mathsf { M } _ { n }$ converges almost surely and in $L ^ { 2 }$ to a limit $\mathsf { M } _ { \infty }$ , with

$$
\mathbb { E } \| \mathsf { M } _ { \infty } \| _ { 2 } ^ { 2 } \leq B ^ { 2 } \sum _ { t = 1 } ^ { \infty } \gamma _ { t } ^ { 2 } .\tag{51}
$$

Doob’s maximal inequality therefore yields

$$
\mathrm { P r } \bigg \{ \operatorname* { s u p } _ { n \geq 1 } \| \mathsf { M } _ { n } \| _ { 2 } > \frac { \varepsilon _ { K } } { 4 } \bigg \} \leq \frac { 6 4 B ^ { 2 } } { \varepsilon _ { K } ^ { 2 } } \sum _ { t = 1 } ^ { \infty } \gamma _ { t } ^ { 2 } .\tag{52}
$$

On the complementary event, every martingale increment satisfies

$$
\| \mathsf { M } _ { n } - \mathsf { M } _ { m } \| _ { 2 } \leq \frac { \varepsilon _ { K } } { 2 } , \qquad m , n \geq 0 ,\tag{53}
$$

so the preceding robustness argument applies to every flowtime window. Moreover, the martingale tail oscillations vanish because $\mathsf { M } _ { n }$ converges, and the remainder and Euler errors are summable. The interpolation is therefore an asymptotic pseudotrajectory that enters and remains in $U _ { 2 }$ . The standard attractor result for asymptotic pseudotrajectories then gives dist $( X _ { t } , \mathcal { T } )  0$ (Benaïm 1999). Absorbing the displayed maximal-inequality coeficient into $C _ { K }$ proves the claim. □

Supplementary Proposition 1 (Constant-probability stochastic trapping). Consider the baseline-free independent gradient-bandit update on the block-trap table from Main-Text Proposition 2, initialized with zero preferences and using deterministic rewards. Let

$$
\gamma _ { t } = \frac { \alpha } { t + t _ { 0 } } , \qquad \alpha > 0 ,\tag{54}
$$

where $t _ { 0 }$ is suficiently large. Then, there exists a constant $p _ { \mathrm { t r a p } } > 0$ , independent ofthe horizon, such that, with probability at least p<sub>trap</sub>,

$$
q _ { i , t } = \pi _ { i , t } ( 2 ) < { \frac { 9 } { 1 9 } } f o r a l l t , \qquad q _ { i , t } \to 0 f o r i = 1 , 2 .\tag{55}
$$

On this event, the optimal-profile probability $\pi _ { 1 , t } ( 2 ) \pi _ { 2 , t } ( 2 )$ converges to 0, the objective converges to the suboptimal block value 0.9, and the product-policy gap converges to 0.1. Denote this gap by $D _ { t } = 1 { - } J _ { \mathrm { t r a p } } ( q _ { 1 , t } , q _ { 2 , t } )$ . More precisely, there exists a finite constant C, depending only on the table and the zero-preference initialization, such that, for every fixed $\alpha > 0$ and all suficiently large $t _ { 0 } ,$

$$
p _ { \mathrm { t r a p } } \geq \left[ 1 - \frac { C \alpha ^ { 2 } } { t _ { 0 } } \right] _ { + } .\tag{56}
$$

In particular, increasing t<sub>0</sub> if necessary gives $p _ { \mathrm { t r a p } } \geq 1 / 2 .$ The default initialization alsofails under the natural schedule $\gamma _ { t } = \alpha / ( t + 1 )$ : there exist constants $\alpha _ { 0 } > 0$ and $\begin{array} { r } { C _ { 0 } < \infty , } \end{array}$ depending only on the block-trap table and chosen such that $\bar { C _ { 0 } } \alpha _ { 0 } ^ { 2 } \leq \bar { 1 } / 2 ,$ , for which every $\alpha \in ( 0 , \alpha _ { 0 } ]$ satisfies

$$
\operatorname* { P r } \{ q _ { i , t } \to 0 f o r i = 1 , 2 \} \geq \left[ 1 - C _ { 0 } \alpha ^ { 2 } \right] _ { + } .\tag{57}
$$

Consequently,

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbb { E } } [ D _ { t } ] \geq 0 . 1 \left[ 1 - C _ { 0 } \alpha ^ { 2 } \right] _ { + } .\tag{58}
$$

Proof. Let $X _ { t } = ( \pi _ { 1 , t } , \pi _ { 2 , t } )$ denote the product policy. The exact preference update induces the following multiplicative policy recursion:

$$
\pi _ { i , t + 1 } ( x ) = \frac { \pi _ { i , t } ( x ) \exp ( \gamma _ { t } g _ { i , t } ( x ) ) } { \sum _ { y } \pi _ { i , t } ( y ) \exp ( \gamma _ { t } g _ { i , t } ( y ) ) } .\tag{59}
$$

Because $| g _ { i , t } ( x ) | \leq 1$ , a uniform Taylor expansion on the compact product simplex yields

$$
X _ { t + 1 } = X _ { t } + \gamma _ { t } \big ( F ( X _ { t } ) + U _ { t + 1 } \big ) + \xi _ { t + 1 } ,\tag{60}
$$

where $U _ { t + 1 }$ is a bounded martingale diference, $\| \xi _ { t + 1 } \| _ { 2 } \leq$ $B _ { r } \gamma _ { t } ^ { 2 }$ for a table-dependent constant $B _ { r }$ , and the deterministic policy flow induced by $\dot { \mathbf H } = \nabla J ( \mathbf H )$ has coordinates

$$
F _ { i , x } ( \pi ) = \pi _ { i } ( x ) \left( \mathcal { G } _ { i , x } ( \pi ) - \sum _ { y } \pi _ { i } ( y ) \mathcal { G } _ { i , y } ( \pi ) \right) .\tag{61}
$$

The field $F$ is polynomial in the policy coordinates and therefore Lipschitz on the compact product simplex.

Let

$$
T = \{ ( \pi _ { 1 } , \pi _ { 2 } ) : \pi _ { 1 } ( 2 ) = \pi _ { 2 } ( 2 ) = 0 \} .\tag{62}
$$

The exact reduced policy flow derived above is

$$
\begin{array} { l } { { \dot { q } _ { i } = q _ { i } ^ { 2 } \left( ( 1 - q _ { i } ) ^ { 2 } + \pi _ { i } ( 0 ) ^ { 2 } + \pi _ { i } ( 1 ) ^ { 2 } \right) } } \\ { { \qquad \times \left( 1 . 9 q _ { j } - 0 . 9 \right) , \qquad j \neq i . } } \end{array}\tag{63}
$$

Consequently, T is a compact asymptotically stable attractor, and the open set $q _ { 1 } , q _ { 2 } < 9 / 1 9$ is contained in its basin. In particular, the singleton consisting of the zero-preference policy $q _ { 1 } = q _ { 2 } = 1 / 3$ is a compact subset of this basin.

For $\gamma _ { t } = \alpha / ( t + t _ { 0 } )$

$$
\operatorname* { m a x } _ { t } \gamma _ { t } \leq \frac { \alpha } { t _ { 0 } } , \qquad \sum _ { t = 1 } ^ { \infty } \gamma _ { t } ^ { 2 } \leq \frac { \alpha ^ { 2 } } { t _ { 0 } } .\tag{64}
$$

Supplementary Lemma 1 therefore gives

$$
\operatorname* { P r } \{ q _ { 1 , t } , q _ { 2 , t } \to 0 \} \geq \left[ 1 - { \frac { C \alpha ^ { 2 } } { t _ { 0 } } } \right] _ { + }\tag{65}
$$

for all suficiently large $t _ { 0 } .$ . On the robust event used by the lemma, the interpolation remains in $q _ { 1 } , q _ { 2 } < 9 / 1 9$ , so these inequalities hold at every iterate. For the natural schedule $\gamma _ { t } = \alpha / ( t + 1 )$ , we instead have max<sub>t</sub> $\gamma _ { t } \le \alpha / 2$ and $\textstyle \sum _ { t } \gamma _ { t } ^ { 2 } \leq \alpha ^ { 2 }$ . The same lemma yields constants $\alpha _ { 0 } > 0$ and $C _ { 0 } < \infty$ such that

$$
\operatorname* { P r } \{ q _ { 1 , t } , q _ { 2 , t } \to 0 \} \geq \left[ 1 - C _ { 0 } \alpha ^ { 2 } \right] _ { + } \geq \frac { 1 } { 2 }\tag{66}
$$

for every $\alpha \in ( 0 , \alpha _ { 0 } ]$ , after reducing $\alpha _ { 0 }$ if necessary. On this event, $D _ { t } \to 0 . 1$ . Since $D _ { t } \geq 0 .$ , Fatou’s lemma yields

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbb { E } } [ D _ { t } ] \geq \mathbb { E } [ \operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } D _ { t } ] \geq 0 . 1 \left[ 1 - C _ { 0 } \alpha ^ { 2 } \right] _ { + } .\tag{67}
$$

Finite-horizon witness. The preceding proposition concerns asymptotic behavior and uses diminishing stepsizes. A simpler explicit finite-horizon event is also available for any fixed horizon and constant stepsize. Fix $T \geq 1$ and $\eta > 0 ,$ and run the baseline-free independent gradient-bandit update on the same block-trap table from zero preferences. Let $E _ { T }$ denote the event that both players sample arm 0 in every round $t \leq T$ . This event lies within the suboptimal block. Conditional on $E _ { t - 1 }$ , symmetry ensures that the preferences of arms 1 and 2 remain equal for each player, while the preference of arm 0 is no smaller than either of them. Hence, on each prefix of the event, every player samples arm 0 with probability at least $1 / 3$ in every round. Therefore,

$$
\operatorname* { P r } ( E _ { T } ) \geq ( 1 / 9 ) ^ { T } .\tag{68}
$$

On $E _ { T }$ , the equality between the preferences of arms 1 and 2 implies $\pi _ { i , t } ( 1 ) = \pi _ { i , t } ( 2 )$ for every i and $t \leq T + 1$ Because arm 0 is sampled repeatedly and receives the positive block reward, its preference never falls below the common preference of arms 1 and 2. Thus, $\pi _ { i , T + 1 } ( 2 ) \leq 1 / 3$ . The optimal-profile probability is the product of the two optimalarm probabilities and is therefore at most $1 / 9$ . Finally, for any $q _ { 1 } , q _ { 2 } \leq 1 / 3$

$$
J _ { \mathrm { t r a p } } ( q _ { 1 } , q _ { 2 } ) = 0 . 9 ( 1 - q _ { 1 } ) ( 1 - q _ { 2 } ) + q _ { 1 } q _ { 2 } \le 0 . 9 ,\tag{69}
$$

and hence $J ^ { \star } - J _ { \mathrm { t r a p } } ( q _ { 1 } , q _ { 2 } ) \geq 0 . 1$

## Strict Trap Structure Beyond One Matrix

The following results establish the trap-minor obstruction in Main-Text Proposition 3 beyond a single numerical table. The classification characterizes strict traps, the embedding result extends them to larger tables, and the lock-in result transfers deterministic attraction to the sampled update. These statements are intentionally restricted to strict traps: degenerate ties can create neutral continua, whereas the separate block trap result above addresses the tied example under default initialization.

Supplementary Proposition 2 (Complete strict-trap classification for $2 \times 2$ tables). Consider any deterministic twoplayer, two-arm common-reward table

$$
{ \frac { 1 } { \ 0 \ { \left| \begin{array} { l l } { 0 } & { 1 } \\ { a } & { b } \end{array} \right| } } } .\tag{70}
$$

Under deterministic product-gradient flow, a strict, attracting pure-product trap with a suboptimal value exists if and only if the table contains a suboptimal strict coordinate-wise local maximum. Equivalently, at least one of the following strict conditions holds:

$$
\begin{array} { r } { a > \operatorname* { m a x } \{ b , c \} \ : a n d \ : a < \operatorname* { m a x } \{ a , b , c , d \} , } \\ { b > \operatorname* { m a x } \{ a , d \} \ : a n d \ : b < \operatorname* { m a x } \{ a , b , c , d \} , } \\ { c > \operatorname* { m a x } \{ a , d \} \ : a n d \ : c < \operatorname* { m a x } \{ a , b , c , d \} , } \\ { d > \operatorname* { m a x } \{ b , c \} \ : a n d \ : d < \operatorname* { m a x } \{ a , b , c , d \} . } \end{array}\tag{71}
$$

Proof. Let $p = \pi _ { 1 } ( 0 )$ and $q = \pi _ { 2 } ( 0 )$ . The product objective is

$$
J ( p , q ) = p q a + p ( 1 - q ) b + ( 1 - p ) q c + ( 1 - p ) ( 1 - q ) d .\tag{72}
$$

The sign of $\dot { p }$ equals the sign of the local advantage of arm 0 over arm 1 for player 1,

$$
F _ { 1 } ( q ) = q ( a - c ) + ( 1 - q ) ( b - d ) ,\tag{73}
$$

and the sign of $\dot { q }$ equals the sign of the local advantage of arm 0 over arm 1 for player 2,

$$
F _ { 2 } ( p ) = p ( a - b ) + ( 1 - p ) ( c - d ) .\tag{74}
$$

Thus, the corner $( p , q ) = ( 1 , 1 )$ , which corresponds to payof a, is strictly attracting if and only if $F _ { 1 } ( 1 ) > 0$ and $F _ { 2 } ( 1 ) >$ 0, namely, if $a > c$ and $a > b$ . It is a spurious trap exactly when a is also below the global table maximum. Applying the same inward-sign calculation at the other three corners yields the four displayed alternatives.

It remains to rule out other strictly attracting productpolicy equilibria. $\operatorname { I f } a - b - c + d \neq 0$ , any interior stationary point is the intersection of the two afine indiference lines. The of-diagonal entries in the linearization of the $( p , q )$ flow have a strictly positive product, so the point is a saddle. If $a - b - c + d = 0 .$ , the indiference conditions are constant or define non-isolated neutral sets. Hence, no additional strictly attracting interior trap exists. A boundary equilibrium that is not a strict corner requires a tie in the local advantage of one player and is therefore not strictly attracting. This proves the equivalence. □

Supplementary Proposition 3 (Trap-minor embedding for finite-arm tables). Consider any deterministic two-player common-reward table withfinite arm sets. Suppose that there are arms $x _ { 0 } , x _ { 1 } \in \mathcal { A } _ { 1 }$ and $y _ { 0 } , y _ { 1 } \in \mathcal { A } _ { 2 }$ such that the $2 \times 2$ minor on these arms has a strict suboptimal trap at $( x _ { 0 } , y _ { 0 } ) .$

$$
\begin{array} { r l } & { r ( x _ { 0 } , y _ { 0 } ) > r ( x _ { 1 } , y _ { 0 } ) , } \\ & { r ( x _ { 0 } , y _ { 0 } ) > r ( x _ { 0 } , y _ { 1 } ) , } \\ & { r ( x _ { 0 } , y _ { 0 } ) < J ^ { \star } . } \end{array}\tag{75}
$$

Assume further that the trap corner is strict with respect to all outside unilateral deviations,

$$
\begin{array} { r l } & { r ( x _ { 0 } , y _ { 0 } ) > r ( x , y _ { 0 } ) \quad \forall x \neq x _ { 0 } , } \\ & { r ( x _ { 0 } , y _ { 0 } ) > r ( x _ { 0 } , y ) \quad \forall y \neq y _ { 0 } . } \end{array}\tag{76}
$$

Then, $( x _ { 0 } , y _ { 0 } )$ is a strict, attracting pure-product trapfor the full product-gradientflow. In particular, there exists a neighborhood ofproduct policies concentrated on $( x _ { 0 } , y _ { 0 } )$ whose deterministicflow moves toward that suboptimal corner.

Proof. For player 1, when player 2 assigns probability close to one to $y _ { 0 } .$ , every alternative arm $x \neq x _ { 0 }$ has a conditional value strictly below that of $x _ { 0 }$ . This follows because the displayed inequalities are strict and the conditional values vary continuously with the opponent policy. Thus, there exists a neighborhood of $y _ { 0 }$ in which all preference diferences $H _ { 1 } ( x _ { 0 } ) \stackrel { - } { - } H _ { 1 } ( x )$ have positive drift. The same argument applies to player 2 around $x _ { 0 }$ . Therefore, a product-policy neighborhood of the corner has inward drift for every unilateral deviation away from $( x _ { 0 } , y _ { 0 } )$ . Since $r ( x _ { 0 } , y _ { 0 } ) < J ^ { \star }$ the attracting corner is suboptimal. The $2 \times 2$ minor conditions identify the local coordination-trap pattern, while the outside-deviation inequalities ensure that the minor remains attracting in the full table. □

Supplementary Proposition 4 (Stochastic lock-in for strict trap minors). Consider a deterministic finite two-player common-reward table satisfying the trap-minor conditions of Supplementary Proposition 3. Let B denote the basin of the resulting suboptimal pure-product attractor. Fix $\alpha > 0 ,$ and run the baseline-free independent gradient-bandit update with stepsizes

$$
\gamma _ { t } = \frac { \alpha } { t + t _ { 0 } } .\tag{77}
$$

For every compact set $K \subset B ,$ , there exist constants $C _ { K } < \infty$ and $t _ { K } < \infty$ , where $t _ { K }$ may depend on α, such that the following statement holds:for everyfinite-preference initialization whose induced product policy lies in K and every $t _ { 0 } ~ \geq ~ t _ { K }$ , the stochastic product policies converge to the suboptimal trap with probability at least

$$
\left[ 1 - \frac { C _ { K } \alpha ^ { 2 } } { t _ { 0 } } \right] _ { + } .\tag{78}
$$

Let

$$
\Delta _ { \mathrm { t r a p } } = J ^ { \star } - r ( x _ { 0 } , y _ { 0 } ) > 0 .\tag{79}
$$

Then the expected product-policy gap also satisfies

$$
\operatorname* { l i m } _ { t \to \infty } \operatorname* { i n f } _ { \mathbb { E } } [ J ^ { \star } - J ( \pi _ { t } ) ] \geq \Delta _ { \mathrm { t r a p } } \left[ 1 - { \frac { C _ { K } \alpha ^ { 2 } } { t _ { 0 } } } \right] _ { + } .\tag{80}
$$

Proof. The strict unilateral inequalities in Supplementary Proposition 3 make the trap corner an asymptotically stable attractor of the deterministic policy flow. The multiplicative softmax expansion used in the proof of Supplementary Proposition 1 expresses the stochastic product-policy recursion in the form required by Supplementary Lemma 1, with a bounded martingale diference and a uniform $O ( \gamma _ { t } ^ { 2 } )$ remainder. Finally,

$$
\operatorname* { m a x } _ { t } \gamma _ { t } \leq \frac { \alpha } { t _ { 0 } } , \qquad \sum _ { t = 1 } ^ { \infty } \gamma _ { t } ^ { 2 } \leq \frac { \alpha ^ { 2 } } { t _ { 0 } } .\tag{81}
$$

Applying the lemma to $K \subset B$ proves the stated bound after increasing $t _ { K }$ , if necessary. On the lock-in event, $J ( \pi _ { t } ) $ $r ( x _ { 0 } , y _ { 0 } )$ , whereas $J ^ { \star } - \dot { \boldsymbol J } ( \pi _ { t } ) \ge 0$ on every sample path. Fatou’s lemma therefore gives the displayed expected-gap lower bound. □

Supplementary Proposition 5 (A block-trap family). Let each player $i ~ \in ~ \{ 1 , 2 \}$ have one optimal arm $o _ { i }$ and a nonempty block $B _ { i }$ of $m _ { i }$ non-optimal arms. Consider the deterministic rewards

$$
\begin{array} { l r } { { r ( o _ { 1 } , o _ { 2 } ) = 1 , } } \\ { { r ( b _ { 1 } , b _ { 2 } ) = \rho , } } \\ { { r ( o _ { 1 } , b _ { 2 } ) = r ( b _ { 1 } , o _ { 2 } ) = 0 } } \end{array}\tag{82}
$$

for all $b _ { i } \in B _ { i }$ , where $\rho \in ( 0 , 1 )$ . If

$$
\frac { 1 } { m _ { i } + 1 } < \frac { \rho } { 1 + \rho } f o r i = 1 , 2 ,\tag{83}
$$

then the default uniform initialization lies in an invariant bad basin ofthe deterministic product-gradientflow. Within this basin, theflow moves both optimal-arm probabilities toward zero. Moreover, under the same diminishing-stepsize conditions as Supplementary Proposition 1, the baselinefree independent gradient-bandit dynamics have a horizonindependent constant lower bound on the probability ofcon vergence to the suboptimal block $B _ { 1 } \times B _ { 2 }$

Proof. Let $q _ { i } = \pi _ { i } ( o _ { i } )$ . For every table in this family, define the reduced objective by

$$
J _ { \rho } ( q _ { 1 } , q _ { 2 } ) : = J ( { \bf H } ) = \rho ( 1 - q _ { 1 } ) ( 1 - q _ { 2 } ) + q _ { 1 } q _ { 2 } .\tag{84}
$$

The conditional value of $o _ { 1 }$ is q<sub>2</sub>, whereas every arm in $B _ { 1 }$ has conditional value $\rho ( 1 - q _ { 2 } )$ . Hence, the direction that increases the probability of the optimal arm of player 1 relative to the block has the sign of

$$
q _ { 2 } - \rho ( 1 - q _ { 2 } ) = ( 1 + \rho ) q _ { 2 } - \rho .\tag{85}
$$

The expression for player 2 is symmetric. Therefore, the separator is

$$
q _ { 1 } = q _ { 2 } = \frac { \rho } { 1 + \rho } .\tag{86}
$$

The displayed condition states precisely that the uniform initialization $q _ { i } = 1 / ( m _ { i } + 1 )$ lies below this separator for both players. Below the separator, both optimal-arm directions are negative; thus, the lower-left region is invariant and attracts the flow toward $q _ { 1 } = q _ { 2 } = 0$ . The stochastic statement follows by applying Supplementary Lemma 1 to this attracting block face. 口

Where single-player and game-learning intuitions fail. The analysis of a single-player stochastic gradient bandit relies on the fact that the advantage of an optimal arm is determined by that arm relative to the current scalar objective. In particular, the conditional value of the optimal arm cannot deteriorate merely because another learner has not yet coordinated with it. In an OC-MPMAB, however, player i compares $Q _ { i } ^ { \pi } ( A _ { i } ^ { \star } )$ with $J ( \pi )$ , and $Q _ { i } ^ { \pi } ( A _ { i } ^ { \star } )$ averages over the current product policy of the other players. In the block trap,

$$
\begin{array} { r } { Q _ { 1 } ^ { \pi } ( 2 ) - Q _ { 1 } ^ { \pi } ( 0 ) = q _ { 2 } - 0 . 9 ( 1 - q _ { 2 } ) = 1 . 9 q _ { 2 } - 0 . 9 . } \end{array}\tag{87}
$$

At uniform initialization, this quantity is negative. Thus, the same softmax gradient mechanism that drives a single player toward a globally best arm can drive an oblivious player away from its component of the globally optimal joint action when the other player has not yet assigned suficient probability to the matching component. This is precisely the product-policy obstruction ruled out by the certificate $\lambda _ { \star } ( c ) > 0$

This setting also difers from standard common-interest game dynamics. A common-interest game typically provides each learner with a payoffunction indexed by a known action profile or analyzes a specified payof-based revision protocol over a known game form. In an OC-MPMAB, a player observes neither the action profile nor the game form; the player observes only a scalar sample after choosing a local arm. The product-policy objective and the quantities $Q _ { i } ^ { \pi } ( x )$ are therefore analytical objects rather than data available to the learner. The framework in the main text determines which conclusions survive this loss of information: unbiased stationarity survives, global optimality does not, and certificates identify reward tables for which the missing information is not harmful.

## Dense Additive-Interaction Extension

Supplementary Corollary 1 (Additive rewards with bounded interactions). Suppose that each player has an optimal local arm $A _ { i } ^ { \star }$ , and

$$
r ( \pmb { a } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } u _ { i } ( A _ { i } ) + \epsilon h ( \pmb { a } ) , \qquad 0 \leq r ( \pmb { a } ) \leq 1 ,\tag{88}
$$

where $u _ { i } ( A _ { i } ^ { \star } ) - u _ { i } ( x ) \geq \Delta$ for every i and $x \neq A _ { i } ^ { \star }$ , and $| h ( { \pmb a } ) - \dot { h } ( \dot { \pmb a } ^ { \prime } ) | \leq B$ whenever a and a<sup>′</sup> difer in one coordinate. If

$$
\delta = \frac { \Delta } { N } - | \epsilon | B > 0 ,\tag{89}
$$

then the local-dominance condition holds with this value of δ. Consequently,

$$
\lambda _ { \star } ( c ) \geq \left( \frac { c } { N } \left( \frac { \Delta } { N } - \lvert \epsilon \rvert B \right) \right) ^ { 2 } .\tag{90}
$$

Moreover, the explicit bound on $M _ { \mathrm { a l l } } ( c )$ in Main-Text Proposition 4 also applies with this value ofδ.

Proof. For any player i, any $x \neq A _ { i } ^ { \star }$ , and any fixed $\mathbf { \delta } \mathbf { a } _ { - i }$ ,

$$
\begin{array} { l } { r ( A _ { i } ^ { \star } , \pmb { a } _ { - i } ) - r ( x , \pmb { a } _ { - i } ) = \displaystyle \frac { u _ { i } ( A _ { i } ^ { \star } ) - u _ { i } ( x ) } { N } } \\ { \qquad + \epsilon \big ( h ( A _ { i } ^ { \star } , \pmb { a } _ { - i } ) - h ( x , \pmb { a } _ { - i } ) \big ) } \\ { \geq \displaystyle \frac { \Delta } { N } - | \epsilon | B = \delta . } \end{array}\tag{91}
$$

The result then follows from Main-Text Proposition 4.

## Experiments

## Experiment Setup

The experiments examine the product-policy geometry identified by the theory. Unless otherwise stated, all experiments use the same baseline-free independent gradient-bandit update analyzed in the paper. The primary trajectories use 200 runs and 5000 steps, with $\eta ~ = ~ 0 . 2 0$ for sparse coordination and the block trap and $\eta = 0 . 1 5$ for the additive and AI-pipeline instances. When all arm sets have the same cardinality, we write $K = K _ { i }$ for every player. The trajectory plots report means over independent runs with 95% confidence bands computed as the mean ±1.96 standard errors.

Implementation environment. The archived results were regenerated without a GPU on Ubuntu 24.04.4 LTS using two Intel Xeon Gold 5318Y CPUs, 251 GiB of memory, Python 3.13.9, and Matplotlib 3.10.6. The simulation code otherwise uses only the Python standard library, and the complete default suite takes approximately eight minutes on this machine.

Certificate computation. For a finite reward table and a chosen optimal joint action $\mathbf { \pmb { a } } ^ { \star }$ , the structural certificates are ofline properties of the product-policy landscape. First, choose a support threshold c. The deterministic no-trap ratio is evaluated over Π<sub>⋆</sub>(c), whereas the conservative baselinefree noise ratio is evaluated over $\Pi _ { \mathrm { a l l } } ( c )$ , unless a sharper invariant region is known. For each product policy, compute $J ( \pi )$ , the local conditional values $Q _ { i } ^ { \bar { \pi } } ( x )$ , the product-policy gradient $\mathcal { G } ( \pi )$ , the gap $D ( \pi ) = J ^ { \star } - \dot { J ( \pi ) }$ , and the secondmoment quantity $\mathcal { V } ( \pi )$ . The certificate values are

$$
\operatorname* { i n f } _ { D ( \pi ) > 0 } \frac { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } } { D ( \pi ) ^ { 2 } } , \qquad \operatorname* { s u p } _ { \| \mathcal { G } ( \pi ) \| _ { 2 } > 0 } \frac { \mathcal { V } ( \pi ) } { \| \mathcal { G } ( \pi ) \| _ { 2 } ^ { 2 } }\tag{92}
$$

over the selected region. As in the main text, a finite noise certificate also requires $\nu ( \pi ) = 0$ at every zero-gradient policy in the selected region; its value is set to 0 when no positive-gradient policy remains and $\nu$ vanishes throughout the region.

We implement a fixed-depth checker for small binary-arm reward tables. After relabeling the local arms so that $\mathbf { 0 } ^ { \star } =$ $( 0 , \ldots , 0 )$ , we parameterize each product policy by $p _ { i } ~ =$ $\pi _ { i } ( 0 )$ , partition $[ 0 , 1 ] ^ { N }$ into dyadic boxes B at a prescribed depth, and intersect these boxes with the selected certificate region. On each box, rational interval arithmetic provides the certified enclosures

$$
\begin{array} { c } { { \underline { { { D } } } ( B ) \leq D ( \pi ) \leq \overline { { { D } } } ( B ) , } } \\ { { \underline { { { G } } } ^ { 2 } ( B ) \leq \| \mathcal G ( \pi ) \| _ { 2 } ^ { 2 } , } } \\ { { \mathcal V ( \pi ) \leq \overline { { { V } } } ( B ) . } } \end{array}\tag{93}
$$

For any box with $\overline { { { D } } } ( B ) > 0$ , the value ${ \underline { { G ^ { 2 } } } } ( B ) / { \overline { { D } } } ( B ) ^ { 2 }$ is a valid lower bound on the deterministic ratio. The implementation combines these interval values with a separately established analytic lower bound $\lambda _ { \mathrm { a n } } ;$ it does not infer the positivity of $\lambda _ { \star } ( \bar { c } )$ from boxes that intersect the zero-gap set. For the noise ratio, every box that intersects $\Pi _ { \mathrm { a l l } } ( c )$ must satisfy ${ \underline { { G } } } ^ { 2 } ( B ) > 0 .$ , in which case $\overline { { V } } ( B ) / \underline { G } ^ { 2 } ( B )$ is a valid upper bound. If any such box intersects the zero-gradient set, the checker returns unknown. The current implementation neither adaptively subdivides unresolved boxes nor returns a trap witness.

The checker is executable because $J , Q _ { i } ^ { \pi } ( x ) , { \mathcal { G } } ,$ and V are finite sums and products of rational reward entries and policy coordinates. It is therefore a hybrid analytic/interval procedure: the positivity of the deterministic certificate is established analytically, while fixed-depth rational intervals independently certify the all-arm noise bound. Every reported pair $( \underline { { \lambda } } , \overline { { M } } )$ is sound; cases with insuficient interval resolution are reported as unknown. Analytic certificates remain preferable when available: Main-Text Corollary 1 certifies sparse coordination, and Main-Text Proposition 4 provides both certificates for every locally dominant dense table. The AI-pipeline routing instance below is dense and nonadditive but locally dominant with $\delta = 0 . 1 1$ . Grid and random-policy evaluations remain diagnostics rather than certificates.

Table 1 reports the results of the fixed-depth checker at depth four. Its dense interaction row uses the normalized nonadditive table

$$
\begin{array} { c } { { \displaystyle r ( { \pmb a } ) = \frac { 1 } { 1 0 } \bigg ( 2 + 2 \sum _ { i = 1 } ^ { 3 } { \bf 1 } \{ A _ { i } = 0 \} } } \\ { { + { \bf 1 } \{ A _ { 1 } = A _ { 2 } = 0 \} } } \\ { { + { \bf 1 } \{ A _ { 2 } = A _ { 3 } = 0 \} \bigg ) , } } \end{array}\tag{94}
$$

which is locally dominant with $\delta = 0 . 2$ . The displayed values are conservative certified bounds.

The horizon-dependent constants in Main-Text Theorem 2 should be applied through an ofline accept/reject workflow rather than used as an adaptive online rule. First, choose a horizon T and a candidate stepsize schedule, compute $c _ { T } =$ $e ^ { - 2 \Gamma _ { T } } / K _ { \operatorname* { m a x } }$ or a sharper invariant region, and then obtain certified constants $( \lambda , \bar { M } )$ on that region analytically or with the hybrid checker. The final step is to verify that $\gamma _ { t } ~ \leq$ $1 / ( L \dot { M } )$ for every $t \leq T$ . If this condition fails, reduce the stepsize or certify a sharper invariant region and repeat the procedure. Thus, the condition is circular only if the certificate is treated as unknown; under an ofline check, it is a finite suficient-condition test for a proposed run.

All three rows use the finite-horizon regions Π (0.25) and $\Pi _ { \mathrm { a l l } } ( 0 . 2 5 )$ . For sparse coordination, the first value is the analytic bound $c ^ { 2 N } / N ^ { 2 }$ at $c = 0 . 2 5$ , and the second is the fixed-depth interval bound on $M _ { \mathrm { a l l } } ( 0 . 2 5 )$ . These values difer from the invariant-region constants in Main-Text Corollary 1: for $N = 3 , K = 2$ , and $J ^ { \star } = 1$ , that corollary instead uses $c _ { \star } = 0 . 5$ and gives $\lambda _ { \star } ( c _ { \star } ) \geq 1 / 5 7 6$ and $M _ { \mathrm { s p } } \leq 8$ . For the dense additive row, writing $p _ { i } = \pi _ { i } ( 0 )$ gives $D ( \pi ) =$ $\textstyle { \frac { 1 } { 3 } } \sum _ { i } ( 1 - p _ { i } )$ ; selecting an index with $1 - { \dot { p _ { i } } } \geq D ( \pi )$ yields the instance-specific analytic bound $\lambda _ { \star } ( 0 . 2 5 ) \geq 0 . 2 5 ^ { 2 } / 3 ^ { 2 } =$ $1 / 1 4 4$ . The dense interaction row uses the local-dominance bound with $\delta = 0 . 2$ , which gives 1/3600. The three values of M are rational interval bounds over the corresponding all-arm region.

<table><tr><td>Instance</td><td>Region C</td><td> $\Delta$ </td><td>M</td></tr><tr><td>Sparse</td><td></td><td></td><td></td></tr><tr><td> $N = 3 , K = 2$ </td><td>0.25</td><td> $2 . 7 1 \times 1 0 ^ { - 5 }$ </td><td>148.76</td></tr><tr><td>Dense additive  $N = 3 , K = 2$ </td><td>0.25</td><td> $6 . 9 4 \times 1 0 ^ { - 3 }$ </td><td>147.70</td></tr><tr><td>Dense interaction</td><td></td><td></td><td></td></tr><tr><td> $N = 3 , K = 2$ </td><td>0.25</td><td> $2 . 7 8 \times 1 0 ^ { - 4 }$ </td><td>346.23</td></tr></table>

Table 1: Hybrid analytic and interval certificates for binaryarm tables. The checker combines analytic lower bounds for $\lambda _ { \star } ( c )$ with fixed-depth rational interval upper bounds for $M _ { \mathrm { a l l } } ( c )$
<table><tr><td>Reference</td><td>Online information</td><td>Interpretation</td></tr><tr><td>Independent</td><td>own arm + common re- ward</td><td>matched learner</td></tr><tr><td>update Moving-</td><td>own arm + common re-</td><td>matched ablation</td></tr><tr><td>baseline variant</td><td>ward</td><td></td></tr><tr><td>Checker</td><td>offline reward table</td><td>offline certificate or diagnostic check</td></tr><tr><td>Centralized</td><td>joint actions or policy</td><td>upper-info reference</td></tr><tr><td>learner Comm. bandit</td><td>messages or rewards</td><td>different protocol</td></tr></table>

Table 2: Information-model reference. Centralized or communicating methods can serve as upper-information references, but they are not protocol-matched baselines for fully oblivious learners.

Information model and ablations. Table 2 records which rows use the same online information as the independent gradient-bandit update and which require joint actions, communication, or aggregation channels. To separate the product-policy mechanism from variance reduction, we also evaluate an action-independent moving reward baseline initialized at $b _ { 1 } ~ = ~ 0 .$ . Round t uses $R _ { t } \mathrm { ~ - ~ } b _ { t }$ , followed by $b _ { t + 1 } ~ = ~ b _ { t } + 0 . 0 5 ( R _ { t } - b _ { t } )$ . Because $b _ { t }$ is fixed before the current actions are sampled, the baseline term has zero conditional mean and changes neither the product-gradient drift nor the deterministic trap geometry. This variant uses the same online information as the baseline-free update: the sampled local arm and the common reward. The stepsize and baseline ablation uses $\eta \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , \hat { 0 } . 3 0 \}$ , 80 runs, and 3000 steps. The structural pseudo-regret theorem is stated for the baseline-free update because a moving baseline requires a self-bounding certificate for the residual second moment along the joint policy–baseline process.

We also include a small upper-information reference for the block trap. Both rows use 4 runs, 1500 steps, and $\eta = 0 . 2 0$ . The centralized row uses a single softmax distribution over the nine joint actions and therefore observes and parameterizes the joint action, which is unavailable to oblivious learners. This row is not a fair protocol-matched baseline; it is included only to indicate the performance cost of the product-policy information restriction.

<table><tr><td>Reference</td><td>Final gap</td><td> $\pi ( a ^ { \star } )$ </td><td>Avg. pseudo-regret</td></tr><tr><td>Independent</td><td></td><td></td><td></td></tr><tr><td>default Centralized</td><td>0.1037</td><td> $4 . 7 7 \times 1 0 ^ { - 6 }$ </td><td>0.1274</td></tr><tr><td>joint-action</td><td>0.0270</td><td>0.7443</td><td>0.0595</td></tr></table>

Table 3: Upper-information reference on the block trap. The centralized learner has access to joint-action information and is not protocol matched to the independent update.

Instances. The sparse coordination instance uses the reward class and invariant support geometry of Main-Text Corollary 1. The optimal joint action is $\pmb { a } ^ { \star } = ( 0 , \ldots , 0 )$ each player has two arms, and the deterministic reward is

$$
r ( a ) = 1 \{ a = a ^ { \star } \} .\tag{95}
$$

We vary the number of players over $N \in \{ 3 , 5 , 8 \}$ . The practical choice $\eta = 0 . 2 0$ exceeds the corollary’s conservative suficient limits 1/96, 1/640, and 1/8192, respectively. These trajectories are therefore mechanism diagnostics and do not empirically verify the certified rate constants.

The dense additive instance uses $N = 6$ players and $K _ { i } =$ 4 arms per player. Arm 0 is optimal for every player, and the deterministic mean reward is

$$
r ( \pmb { a } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \pmb { 1 } \{ A _ { i } = 0 \} .\tag{96}
$$

The observed reward follows a Bernoulli distribution with this mean.

The AI-pipeline routing instance is a stylized dense OC-MPMAB with $N = 4$ modules: prompt template, retrieval policy, tool choice, and verification mode. Each module chooses one of three local configurations and observes only the final task-success reward. Arm 0 is the preferred local configuration. The mean reward is

$$
\begin{array} { l } { { \displaystyle r ( { \pmb a } ) = \frac { 1 } { 4 } \sum _ { i = 1 } ^ { 4 } u ( A _ { i } ) - 0 . 0 1 5 \sum _ { ( i , j ) \in E } { \bf 1 } \{ A _ { i } \not = A _ { j } \} } , } \\ { { \displaystyle u = ( 1 , 0 . 4 , 0 . 1 ) } , } \end{array}\tag{97}
$$

where $E = \{ ( 1 , 2 ) , ( 2 , 3 ) , ( 3 , 4 ) , ( 1 , 4 ) \}$ encodes mild compatibility penalties between adjacent pipeline choices. Although this table is dense and nonadditive, changing any one module from a nonzero arm to arm 0 improves the reward by at least $0 . 1 5 - 2 ( 0 . 0 1 5 ) = 0 . 1 2$ . We use the conservative certificate value $\delta = 0 . 1 1$

The coordinate block trap is the two-player, three-arm reward matrix used in Main-Text Proposition 2:

$$
\begin{array} { r } { \overbrace { A _ { 1 } = 0 } ^ { \boldsymbol { \ A } _ { 2 } = 0 } \quad \overbrace { 0 . 9 \qquad 0 . 9 \qquad 0 } ^ { A _ { 2 } = 1 } } \\ { A _ { 1 } = 1 \qquad 0 \qquad 0 . 9 \qquad 0 } \\ { A _ { 1 } = 2 \qquad 0 \qquad 0 \qquad 1 } \end{array} .\tag{98}
$$

For the opt-biased trajectory, $H _ { i , 1 } ( 2 ) \textrm { = 2 }$ and the other preferences are zero. For the phase diagram, we scan the two gaps $H _ { i , 1 } ( 2 ) - H _ { i , 1 } ( 0 ) = \hat { H _ { i , 1 } } ( 2 ) - \check { H } _ { i , 1 } ( 1 ) , i = 1 , 2 ,$ , over $\mathbf { a 3 1 } \times \mathbf { 3 1 }$ grid on $[ - 4 , 4 ] ^ { 2 }$ , using 20 runs per grid point, 2000 steps, and $\eta = 0 . { \dot { 2 } } 0$

For the random finite-table stress test, the rewards $r ( a )$ are sampled independently from the uniform distribution on [0, 1], the best joint action is selected as $\mathbf { \pmb { a } } ^ { \star }$ , and the ratio diagnostics are evaluated with $c = 0 . 0 5$ by combining a coarse grid with random product policies. The construction enforces only $\pi _ { i } ( A _ { i } ^ { \star } ) \geq c ,$ so every sampled policy lies in $\Pi _ { \star } ( c ) ;$ it does not enforce the all-arm constraints of $\mathrm { \bar { I I } _ { a l l } } ( c )$ Consequently, $\widehat { M } _ { \mathrm { r s } } ( c )$ is a diagnostic ratio over the sampled policies, not an estimate of the formal certificate $M _ { \mathrm { a l l } } ( { \boldsymbol { c } } )$

The plotted $N = 3 , K = 2$ test uses 64 tables, 16 runs, 3000 learning steps, $\eta = 0 . 1 5$ , 8000 diagnostic policy samples per table, and an 11-point grid for each optimal-arm probability. The multi-size suite uses $\eta = 0 . 1 5 ,$ , 2000 learning steps, 1500 diagnostic policy samples, a 9-point grid per optimal-arm probability, and 24 random starts per table. Each initial preference is sampled uniformly from [−1, 1], and rewards during learning are Bernoulli samples with the table entry as their mean.

To assess trap robustness, we vary the suboptimal coordinated reward in the family

$$
\begin{array} { r } { \overrightarrow { A } _ { 1 } = 0 \left| \begin{array} { c c c } { A _ { 2 } = 0 } & { A _ { 2 } = 1 } & { A _ { 2 } = 2 } \\ { \rho } & { \rho } & { 0 } \\ { \rho } & { \rho } & { 0 } \\ { 0 } & { 0 } & { 1 } \end{array} \right| } \end{array}\tag{99}
$$

with $\rho \in \{ 0 . 6 0 , 0 . 7 5 , 0 . 9 0 , 0 . 9 5 \}$ . For each $\rho ,$ we scan a $2 1 \times 2 1$ grid on $[ - 4 , 4 ] ^ { 2 }$ , use 4 runs per grid point, and run 1500 steps with $\eta = 0 . 2 0$

## Results

Protocol-matched baseline and stepsize ablations. Table 4 reports the best final product-policy gap over the stepsize grid for each variant. The moving baseline leaves the sparse gap unchanged at 0.0011 and reduces the dense additive gap from 0.0133 to 0.0054, consistent with efective variance reduction in a benign landscape. On the block trap, however, the gap changes from 0.0922 to 0.0963 and remains bounded away from zero. Thus, the baseline can improve finite-sample optimization without changing the mean product-policy geometry that creates the trap.

Random finite-table stress test. We denote the randomsearch diagnostics by $\widehat { \lambda } _ { \mathrm { { r s } } } ( c )$ and $\widehat { M } _ { \mathrm { r s } } ( c )$ . The latter is not an estimate of $M _ { \mathrm { a l l } } ( c )$ because the diagnostic search does not enforce $\Pi _ { \mathrm { a l l } } ( c )$ . Table 5 reports the multi-size stress suite. Across all three settings, the highest quartile of $\widehat { \lambda } _ { \mathrm { { r s } } } ( c )$ has a lower mean terminal gap than the lowest quartile. This experiment is an ofline stress test of the certificate heuristic rather than a comparison with algorithms that use stronger information.

<table><tr><td>Instance</td><td>No baseline η</td><td>Gap</td><td>Moving baseline η</td><td>Gap</td></tr><tr><td>Sparse</td><td>0.30</td><td>0.0011</td><td>0.30</td><td>0.0011</td></tr><tr><td>Additive</td><td>0.20</td><td>0.0133</td><td>0.30</td><td>0.0054</td></tr><tr><td>Block trap</td><td>0.30</td><td>0.0922</td><td>0.30</td><td>0.0963</td></tr></table>

Table 4: Protocol-matched stepsize and baseline ablation. Both variants use only the sampled local arm and the common reward.

![](images/d219a9cce2bbdc213714a6c921697cde00fd630128bd065a2b8b2075e5bb982f.jpg)

![](images/10f2ae801244acaec707f50456336f5752c0d11d4cc400d36fbe5eeb4db9f0a1.jpg)  
Figure 1: Coordinate block trap. Left: trajectories from default and opt-biased initializations. Right: final optimal-profile probability over the initial preference gaps.

<table><tr><td>Setting</td><td>Tables</td><td>All gap</td><td> $\operatorname { L o w - } \widehat { \lambda } \operatorname { g a p }$ </td><td> $\mathrm { H i g h } { - } \widehat { \lambda }$  gap</td></tr><tr><td> $N = 3 , K = 2$ </td><td>128</td><td>0.0413</td><td>0.0561</td><td>0.0260</td></tr><tr><td> $N = 4 , K = 2$ </td><td>64</td><td>0.0561</td><td>0.0584</td><td>0.0485</td></tr><tr><td> $N = 3 , K = 3$ </td><td>64</td><td>0.0701</td><td>0.0903</td><td>0.0573</td></tr></table>

Table 5: Random finite-table stress suite; low and high denote the lowest and highest $\widehat { \lambda } _ { \mathrm { { r s } } } ( c )$ quartiles.  
![](images/85354a9c7ac997db1258ad963bfc29ff8d91464c185d58a026a8992956b3d9b9.jpg)

![](images/df96d16bac86dcab2e57b27ebe78fe77734b2aaa034a598be2e352a8e6c1be05.jpg)  
Figure 2: Random-table stress: terminal gap versus $\widehat { \lambda } _ { \mathrm { { r s } } } ( c )$ (top; color indicates $\widehat { M } _ { \mathrm { r s } } ( c ) )$ and quartile means with ±1.96 standard errors (bottom).

Optimally supported coordination. Figure 3 shows the product-policy gap and average pseudo-regret. In this instance, the probability assigned to the optimal joint action is exactly one minus the gap. The rewards and invariant support match the structural class of Main-Text Corollary 1, but the practical stepsize exceeds its suficient range. The plots therefore illustrate the predicted geometry rather than certify the finite-time constants.

![](images/a775ed6e8e555d09f3ea8f66ff7fac7bd61b79d979552efffbd6a72e5079206f.jpg)

![](images/adaa489195353a3739cf51ded013e7ce383f2fff2d13e0deed2eb58bf67ed18a.jpg)  
Figure 3: Optimally supported coordination: product-policy gap (top) and average pseudo-regret (bottom).

Benign additive common reward. This dense instance has a benign no-trap geometry: every player has a local move that improves the common reward independently of the other players. The top panel of Figure 4 shows the product-policy gap, average pseudo-regret, and probability assigned to the optimaljoint action. The experiment illustrates that the structural perspective also applies to dense reward tables beyond the explicit sparse class.

Stylized AI-pipeline routing. The AI-pipeline instance is not intended to benchmark a deployed system; instead, it is a compact finite-table case that matches the motivating information pattern. Four pipeline modules independently adapt their local choices using one shared task-success score, and no module observes the choices of the others. The bottom panel of Figure 4 shows that the independent update improves both the product-policy gap and the probability assigned to the best pipeline configuration. The certificate for this instance is analytic rather than diagnostic: although the instance is dense and nonadditive, local dominance gives $\lambda _ { \star } ( c ) \geq ( c \delta / 4 ) ^ { 2 }$ and the all-arm noise bound from Main-Text Proposition 4.

![](images/78c47f9869b0a98cd6b2589f0fae9d23a704fc46fc53f77f083c3fbc17ff12e9.jpg)

![](images/72c79741fbcae490078c1306a759a0a9c65f8d3d3aafb250f34ab084c9b04d6a.jpg)  
Figure 4: Dense structured instances: additive reward (top) and AI-pipeline routing (bottom).

Coordinate block trap. The globally optimal action is (2, 2), but the block $\{ 0 , 1 \} ^ { 2 }$ is coordinate-wise attractive in the deterministic product-gradient landscape. The left panel of Figure 1 compares the default uniform initialization with an opt-biased initialization. Under the default initialization, the finite-horizon stochastic runs remain concentrated on the suboptimal block and assign low probability to (2, 2), consistent with the deterministic basin analysis. Supplementary Proposition 1 complements these constant-stepsize diagnostics by establishing constant-probability trapping for a diminishing-stepsize process; the right panel instead estimates the deterministic basin of attraction. Table 6 shows that this basin grows as the suboptimal coordinated reward approaches the global optimum, supporting the interpretation that the failure mode is an efect of the product-policy landscape rather than a numerical artifact of a single reward table.

<table><tr><td> $\rho$ </td><td>Trap basin</td><td>Mean  $\pi ( a ^ { \star } )$ </td><td>Mean objective</td></tr><tr><td>0.60</td><td>0.575</td><td>0.423</td><td>0.766</td></tr><tr><td>0.75</td><td>0.603</td><td>0.396</td><td>0.846</td></tr><tr><td>0.90</td><td>0.634</td><td>0.364</td><td>0.933</td></tr><tr><td>0.95</td><td>0.644</td><td>0.354</td><td>0.964</td></tr></table>

Table 6: Trap robustness versus $\rho ;$ “trapped” means final π $\mathbf { \partial } \cdot ( \mathbf { a } ^ { \star } ) < 0 . 5$

## References

Benaïm, M. 1999. Dynamics of Stochastic Approximation Algorithms. In Séminaire de Probabilités XXXIII, volume 1709 of Lecture Notes in Mathematics, 1–68. Berlin, Heidelberg: Springer.