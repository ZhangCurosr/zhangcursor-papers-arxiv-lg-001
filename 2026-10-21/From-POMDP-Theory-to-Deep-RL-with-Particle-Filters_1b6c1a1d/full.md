# From POMDP Theory to Deep RL with Particle Filters

Kevin Tan Wharton Statistics and Data Science University of Pennsylvania kevtan@wharton.upenn.edu

Ziping Xu School of Data and Information Sciences University of North Carolina at Chapel Hill zipingxu@unc.edu

## Abstract

Model-based reinforcement learning in partially observable environments requires inferring hidden states, learning latent dynamics, and planning under the resulting belief. Practical algorithms often solve one aspect of the above pipeline while relying on approximations for the others that are often poorly understood, while existing POMDP theory typically analyzes computationally ineficient theoretical algorithms far from modern deep RL practice. To fill this gap, we propose a practical end-to-end algorithm that approximately solves the hidden state filtering problem via sequential Monte Carlo (SMC), updates a latent dynamics model online, and performs approximate online planning for d steps before bootstrapping leaves with a state-value critic (a QMDP approximation). We analyze each approximation and prove an end-to-end regret bound decomposing the error into model estimation error, particle filter error, Monte Carlo rollout error, and a structural bias term induced by the QMDP approximation controllable by the planning depth that we further analyze in various special cases. The resulting guarantee makes explicit how statistical sample size, particle budget, rollout budget, and planning depth afect performance. We instantiate our algorithm in an MPC-based deep RL framework, demonstrating its practicality on a variety of environments. On a POMDP that satisfies our assumptions, each term of the bound decays at its predicted rate. Our results provide a practical and theoretically grounded route for bridging theory and practice within online decision-making in POMDPs.

## 1 Introduction

Partially observable Markov decision processes (POMDPs) provide a standard framework for sequential decision making under latent state uncertainty [19]. They are a natural model for problems in robotics, healthcare, recommendation, and other settings [17, 33] where the learner must act using only indirect and noisy observations of the underlying system. In principle, optimal control in a POMDP can be reduced to planning in the belief-state MDP, where the belief is the posterior distribution over latent states given the observation history [4, 19]. This makes model-based RL especially natural in POMDPs: to act under a belief, the learner must already learn a model rich enough to perform filtering, and this same model can be used for planning.

However, the belief state reduction does not remove the main computational and learning dificulties. An online model-based decision maker must simultaneously (i) learn the unknown model, (ii) maintain a posterior over latent states online, and (iii) plan over this continuous belief space. These three tasks–learning, inference, and planning–interact in a way that makes exact online control in POMDPs intractable in both computation and sample complexity in general [32] without special structure, motivating a large literature on approximate methods for each component individually.

Existing methods typically make one component of the above POMDP pipeline tractable while leaving the others either computationally expensive or theoretically uncontrolled. For instance, on the planning side, Monte Carlo tree search has led to scalable online POMDP solvers such as Partially Observable Monte Carlo Planning (POMCP), sparse tree search, and progressive widening methods [26, 34, 36]. However, these are primarily planning algorithms under a known generative model, and do not natively provide end-to-end learning guarantees for unknown latent dynamics. A more aggressive approximation is QMDP, which replaces the belief-MDP continuation value by the fully observed MDP value averaged under the current belief [27]. QMDP is computationally eficient but can introduce substantial bias. Existing theory gives limited guidance on when this bias is smal or how it can be mitigated. Regarding inference, particle filters, a.k.a. sequential Monte Carlo (SMC), provide a flexible approximation to Bayesian filtering in nonlinear non-Gaussian systems [8, 38], but SMC theory for POMDPs [25, 26] has not been fully integrated with approximate online planning and model learning. On learning, recent theoretical work has made significant progress on sample-eficient model estimation for POMDPs. [28, 29] develop Optimistic MLE as a generic model-based algorithm with polynomial sample complexity for a broad class of POMDPs. These results establish statistical feasibility under exact or abstract planning oracles. They do not specify how to perform approximate filtering or planning with the learned model in a computationall practical way.

A separate practical line of work uses recurrent policies or diferentiable planning modules. Recurrent model-free deep RL methods, such as GRU or LSTM-based PPO and Q-learning variants, can perform well empirically under partial observability, but their hidden states are dificult to interpret as beliefs and are hard to analyze theoretically. QMDP-net takes a more structured approach by embedding approximate Bayesian filtering and QMDP planning into a diferentiable architecture [22]. This is closer in spirit to our goal: combine model learning, belief state inference, and planning in a way that resembles modern deep learning practice, while maintaining provable guarantees. However, the approximation error of learned variational or diferentiable filtering is dificult to control. Our view is that using SMC as the filtering primitive yields a more principled bridge: it is compatible with learned latent dynamics and has well-developed stability and approximation theory [8, 38].

In summary, existing work advances planning, inference, or learning separately, but no current method provides a unified pipeline that combines all three with both computational practicality and end-to-end theoretical guarantees. This leaves open the central question

Can we design a practical model-based deep RL pipeline for POMDPs that still admits end-to-end theoretical guarantees?

## 1.1 Our Contributions

This paper develops a practical and analyzable route from POMDP theory to model-based deep RL. We answer the central question by developing a model-based learning-inference-planning pipeline for POMDPs called PF-MCTQ that is compatible with deep RL practice. Our contributions are:

1. A practical inference-learning-planning pipeline. We propose an online algorithm that learns a latent dynamics model, tracks the current belief with particles, and plans under this belief. Each is modular and can be instantiated with standard deep RL primitives.

2. A controlled alternative to full belief-tree planning. Instead of treating the particle belief as the planning state, we plan particle-wise and average action-values of the fully observable MDP. This gives a practical middle ground between expensive belief-tree search and cheap but potentially biased QMDP. Our analysis shows that the bias decays geometrically after d steps of belief-MDP lookahead. For cheaper state-level planners, we show that QMDP is exact for the belief-linear component of the value, with error only from an uncertainty-sensitive residual, clarifying when the approximation is small.

![](images/864d97ef5974cdc68f44f0c7b285566f6155a1e8ff5b7feee2d9de548786c8bc.jpg)

![](images/d04ea7defb7354c5319354babd4d86911d404e5def41a445b29898959fc82866.jpg)  
Figure 1: Particle beliefs for the unobserved velocity of a double integrator LQR over the last 1000 steps. Mean, 10th and 90th quantiles visualized. Beliefs track the state when the system is stable.

3. An end-to-end regret guarantee. We prove a regret bound that decomposes performance loss into model estimation error, particle filtering error, Monte Carlo rollout error, and QMDP structural bias. The bound makes explicit how sample size, particle budget, rollout budget, and planning depth jointly control performance.

4. A bridge to deep RL practice. Compared with theoretical POMDP learning algorithms that rely on exact POMDP solvers or posterior sampling over parameters, our approach uses MPC-style online planning, approximate filtering, and critic bootstrapping. These are standard primitives in model-based deep RL, while remaining explicit enough to support theoretical analysis. We instantiate our algorithm in an MPC-based deep RL implementation and evaluate it on various POMDP environments, demonstrating its practicality. On a POMDP satisfying our assumptions, each term of the bound decays at its predicted rate (Section 5).

Overall, the paper argues that particle filtering can serve as a practical bridge between POMDP theory and deep RL. It gives a belief representation that is explicit enough to analyze, flexible enough for nonlinear latent dynamics, and compatible with MPC-style planning and learned critics. This yields a route to online POMDP learning that is closer to modern model-based RL than exact belief-space planning, while providing a transparent theoretical account that lets users trace each source of approximation error and make informed design choices.

Section 2 introduces the online learning problem and the regret criterion in the belief-state MDP. Section 3 presents the PF-MCTQ algorithm, describing the MCTQ planning subroutine and the full online pipeline. Section 4 develops the error analysis: we decompose the regret into four traceable terms, analyze the structural QMDP bias and its variance interpretation, bound each error source, and combine them into the end-to-end regret guarantee. We conclude with case studies on LQG, contextual MDPs, and smooth-value POMDPs that illustrate when the QMDP approximation is benign.

## 2 Problem Formulation

We study online reinforcement learning in a partially observable environment with unknown dynamics. Since the latent state is hidden, the natural state variable for control is the belief, the posterior distribution of the latent state given the history. We first introduce the POMDP model and its belief-state dynamics. Figure 1 shows a particle belief for a partially observed double integrator (Section 5).

Parameterized POMDP model. For each parameter $\theta \in \Theta$ , let $\mathcal { M } _ { \theta } = ( S , A , \mathcal { O } , P _ { \theta } , Z _ { \theta } , r , \gamma , b _ { 0 } )$ be an infinite-horizon discounted POMDP, where S is the latent state space, A is the action space, and O is the observation space. The controlled state transition kernel is $P _ { \theta } ( \cdot \mid s , a ) \in \Delta ( S )$ , the observation kernel is $Z _ { \theta } ( \cdot \mid s ^ { \prime } ) \in \Delta ( \mathcal { O } )$ , the reward function is $r : \mathcal { S } \times \mathcal { A }  [ 0 , R _ { \mathrm { m a x } } ]$ , the discount factor is $\gamma \in ( 0 , 1 )$ , and $b _ { 0 } \in \Delta ( { \cal S } )$ is the initial belief. The true environment is specified by an unknown true parameter $\theta ^ { \star }$ , and we write $\mathcal { M } ^ { \star } : = \mathcal { M } _ { \theta } ,$ ⋆ . Here, for a measurable space $x , \Delta ( { \mathcal { X } } )$ denotes the set of probability measures on $x ,$ and beliefs $b \in \Delta ( \mathcal { S } )$ are written as measures $b ( \mathrm { d } s )$

Generative process and belief states. At each time $t = 0 , 1 , 2 , . . . ,$ the latent state, observation, and reward evolve according to

$$
S _ { t + 1 } \sim P _ { \theta ^ { \star } } ( \cdot \mid S _ { t } , A _ { t } ) , \qquad O _ { t + 1 } \sim Z _ { \theta ^ { \star } } ( \cdot \mid S _ { t + 1 } ) , \qquad R _ { t } = r ( S _ { t } , A _ { t } ) .\tag{1}
$$

The learner does not observe $S _ { t }$ . Instead, decisions are based on the entire history up to time t denoted by $H _ { t } : = ( O _ { 1 : t } , A _ { 0 : t - 1 } , R _ { 0 : t - 1 } )$ . Here $O _ { t }$ is available before choosing $A _ { t }$ . A possibly randomized policy is a sequence $\pi = \{ \pi _ { t } \} _ { t \geq 0 }$ such that $\pi _ { t } ( \cdot \mid H _ { t } ) \in \Delta ( \mathcal { A } )$

Belief states and Bayes filtering. Fix a candidate parameter θ. The corresponding belief state at time t is the posterior distribution $b _ { t } ^ { \theta } ( \cdot ) : = \mathbb { P } _ { \theta } ( S _ { t } \in \cdot \mid H _ { t } )$ . When $\theta = \theta ^ { \star }$ , we write $b _ { t } : = b _ { t } ^ { \theta ^ { \star } }$ for the true belief. Given a belief $b \in \Delta ( \mathcal { S } )$ and action $a \in { \mathcal { A } }$ , define the one-step predictive state distribution and the corresponding predictive observation law respectively by

$$
\tilde { b } _ { \theta } ( \cdot \mid b , a ) : = \int _ { S } P _ { \theta } ( \cdot \mid s , a ) b ( \mathrm { d } s ) , \qquad p _ { \theta } ( \cdot \mid b , a ) : = \int _ { S } Z _ { \theta } ( \cdot \mid s ^ { \prime } ) \tilde { b } _ { \theta } ( \mathrm { d } s ^ { \prime } \mid b , a ) .\tag{2}
$$

After observing $o \in \mathcal { O }$ , the posterior is updated through the Bayes filter (which we later approximate)

$$
\tau _ { \theta } ( b , a , o ) ( \mathrm { d } s ) : = \frac { Z _ { \theta } ( o \mid s ) \tilde { b } _ { \theta } ( \mathrm { d } s \mid b , a ) } { \int _ { \mathcal { S } } Z _ { \theta } ( o \mid s ^ { \prime } ) \tilde { b } _ { \theta } ( \mathrm { d } s ^ { \prime } \mid b , a ) } .\tag{3}
$$

Belief-MDP value functions. The reward associated with belief b and action a is $r ( b , a ) : =$ $\mathbb { E } _ { s \sim b } [ r ( s , a ) ]$ . We assume throughout that $\mathcal { A }$ is finite and that $| r ( s , a ) | \leq R _ { \mathrm { m a x } }$ for all $( s , a ) \in \mathcal { S } \times \mathcal { A }$

For each $\theta ,$ the POMDP induces a fully observed MDP on $\Delta ( S )$ . For any bounded measurable function $V : \Delta ( S )  \mathbb { R }$ , define the belief-MDP Bellman operator

$$
( \mathcal T _ { \theta } V ) ( b ) : = \operatorname* { m a x } _ { a \in \mathcal A } \left\{ r ( b , a ) + \gamma \int _ { \mathcal O } V \big ( \tau _ { \theta } ( b , a , o ) \big ) p _ { \theta } ( o \mid b , a ) \mathrm d o \right\} .\tag{4}
$$

Let $V _ { \theta } ^ { \star , B }$ denote the unique fixed point of $\mathcal { T } _ { \theta }$ . The optimal belief-state action-value function is

$$
Q _ { \theta } ^ { \star , B } ( b , a ) : = r ( b , a ) + \gamma \int _ { \mathcal { O } } V _ { \theta } ^ { \star , B } \big ( \tau _ { \theta } ( b , a , o ) \big ) p _ { \theta } ( o \mid b , a ) \mathrm { d } o , \mathrm { ~ a n d ~ } V _ { \theta } ^ { \star , B } ( b ) = \operatorname* { m a x } _ { a \in \mathcal { A } } Q _ { \theta } ^ { \star , B } ( b , a ) .\tag{5}
$$

More generally, for a policy $\pi$ acting on histories, let $V _ { \theta } ^ { \pi , B } ( b )$ be its discounted value in the belief-MDP from initial belief b, and let $Q _ { \theta } ^ { \star , S }$ and $V _ { \theta } ^ { \star , S }$ denote the value functions of the fully observable MDP.

## 3 The PF-MCTQ Algorithm

We present PF-MCTQ, a model-based online learning algorithm that integrates model learning, belief inference, and planning into a modular pipeline. We divide time into doubling epochs $E _ { k } = [ 2 ^ { k - 1 } , 2 ^ { k } )$ $k = 1 , . . . , \lceil \log _ { 2 } T \rceil$ , using the standard doubling trick [3]. At the start of each epoch, the algorithm (i) updates the MLE $\theta _ { k }$ using all past data and freezes it for the epoch, (ii) maintains a particle approximation $b _ { t } ^ { J }$ of the current belief via sequential predict-reweight-resample, and (iii) plans under ${ \bar { \theta } _ { k } }$ and $b _ { t } ^ { J }$ using finite-depth Monte-Carlo belief-tree search with QMDP leaf evaluation. The doubling-epoch structure keeps the model from updating too often.

Algorithm 1 Pf-Mctq: Particle filtering with Monte Carlo tree QMDP planning. Mctq (Algo  
rithm 2) takes the particle belief $b _ { t } ^ { J }$ and returns an estimate $\widehat { Q } _ { t } ( b _ { t } ^ { J } , a )$ of $Q _ { \bar { \theta } _ { k } , d } ( b _ { t } ^ { J } , a )$ for every root   
action $a \in A .$   
Require: particles $\overline { { J ; } }$ depth $d ;$ rollouts ${ \overline { { N } } } ;$ horizon $\overline { { T ; } }$ tolerance $\beta ;$ model class $\Theta ;$ exploration   
parameter $\eta > 0 .$   
1: Sample $s _ { 1 } ^ { ( 1 ) } , \ldots , s _ { J } ^ { ( 1 ) } \sim b _ { 0 } .$   
2: Set epochs $E _ { k } = [ 2 ^ { k - 1 } , 2 ^ { k } )$ for $k = 1 , \dots , K : = \lceil \log _ { 2 } T \rceil$ , with $\varepsilon _ { k } : = \eta / \sqrt { | E _ { k } | }$   
3: for epoch $k = 1 , 2 , \dots , K$ do   
4: $\bar { \theta } _ { k } \gets$ Model $\mathrm { { . U P D A T E } } ( H _ { T _ { k } } )$ via equation $6 ;$ compute $\widehat { Q } _ { \bar { \theta } _ { k } } ^ { \mathrm { M D P } } \approx Q _ { \bar { \theta } _ { k } } ^ { \star , S }$   
5: for $t \in E _ { k }$ do   
6: $\{ \widehat { Q } _ { t } ( b _ { t } ^ { \check { J } } , a ) \} _ { a \in \mathcal { A } }  \mathrm { M c T Q } \big ( \bar { \theta } _ { k } , b _ { t } ^ { J } , d , N , \widehat { Q } _ { \bar { \theta } _ { k } } ^ { \mathrm { M D P } } \big )$ ▷ Algorithm 2   
7: Play a<sub>t</sub> = arg max<sub>a</sub> $\widehat { Q } _ { t } ( b _ { t } ^ { J } , a )$ w.p. $1 - \varepsilon _ { k } ;$ else $a _ { t } \sim \operatorname { U n i f } ( \mathcal { A } )$   
8: Observe $\left( o _ { t + 1 } , r _ { t } \right)$   
9: Propagate, reweight, and resample particles via equation 7 under ${ \bar { \theta } _ { k } }$   
10: end for   
11: end for

Learning oracle. Let $\mathcal { L } _ { t } ( \boldsymbol { \theta } )$ be the log-likelihood of the observed history $H _ { t }$ under model θ. At the start of epoch $E _ { k }$ , ModelUpdate returns a β-approximate maximizer on all data up to $T _ { k }$ :

$$
\bar { \theta } _ { k } \in \Bigl \{ \theta \in \Theta : \mathcal { L } _ { T _ { k } } ( \theta ) \geq \operatorname* { m a x } _ { \theta ^ { \prime } \in \Theta } \mathcal { L } _ { T _ { k } } ( \theta ^ { \prime } ) - \beta \Bigr \} ,\tag{6}
$$

for a tolerance $\beta \geq 0$ . The model $\bar { \theta } _ { k }$ is frozen throughout epoch $E _ { k }$ . When $\beta = 0$ this is exact MLE, but we accommodate a β-suboptimal solver. In practice, many recent gradient-based methods for the MLE in partially observed Markov process (POMP) models apply here [5, 7, 18, 30, 37].

Filtering update. After playing $a _ { t }$ and observing $o _ { t + 1 } .$ , particles are propagated and reweighted under the current epoch’s model ${ \bar { \theta } } _ { k ( t ) } ;$

$$
\begin{array} { r } { \hat { s } _ { j } ^ { ( t + 1 ) } \sim P _ { \bar { \theta } _ { k ( t ) } } ( \cdot \mid s _ { j } ^ { ( t ) } , a _ { t } ) , \qquad w _ { j } ^ { ( t + 1 ) } \propto Z _ { \bar { \theta } _ { k ( t ) } } ( o _ { t + 1 } \mid \hat { s } _ { j } ^ { ( t + 1 ) } ) , } \end{array}\tag{7}
$$

followed by multinomial resampling $s _ { j } ^ { ( t + 1 ) } \sim$ Categorica $( w _ { 1 } ^ { ( t + 1 ) } , \dots , w _ { J } ^ { ( t + 1 ) } )$ for $j = 1 , \dots , J$ . This is the bootstrap sampling-importance-resampling (SIR) filter; any SMC scheme satisfying the $L ^ { p }$ error bound of Appendix D.3 can be substituted.

Approximate planning. For the current belief, MCTQ runs Monte Carlo tree search on the belief tree for the first d steps and applies the QMDP approximation at the leaves. This gives a middle ground between expensive belief-tree search and the cheap but potentially biased QMDP approximation.

For a belief $b \in \Delta ( \mathcal { S } )$ , define the QMDP leaf value and the depth-d belief tree value as

$$
\bar { V } _ { \theta } ^ { \star , S } ( b ) : = \operatorname* { m a x } _ { a \in \mathcal { A } } \int _ { \mathcal { S } } Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) , \quad \mathrm { ~ a n d ~ } \quad \bar { V } _ { \theta , d } ^ { \star , S } : = { \mathcal T } _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S } , \quad \quad d \geq 0 ,\tag{8}
$$

where $Q _ { \theta } ^ { \star , S }$ is the optimal action-value function of the fully observable MDP induced by $\theta ,$ and $\mathcal { T } _ { \theta }$ is the Bellman operator of the belief-MDP from equation 4. The corresponding action-value function is

$$
Q _ { \theta , d } ( b , a ) : = r ( b , a ) + \gamma \int _ { \mathcal { O } } \bar { V } _ { \theta , d - 1 } ^ { \star , S } \big ( \tau _ { \theta } ( b , a , o ) \big ) p _ { \theta } ( o \mid b , a ) \mathrm { d } o , \qquad d \geq 1 .\tag{9}
$$

When $\begin{array} { r } { d = 0 , \bar { V } _ { \theta , 0 } ^ { \star , S } ( b ) = \operatorname* { m a x } _ { a } \int Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) } \end{array}$ and MCTQ reduces exactly to QMDP. As $d \to \infty$ $\bar { V } _ { \theta , d } ^ { \star , S } \to V _ { \theta } ^ { \star , B }$ (see Proposition 2 in Appendix D.1 for the proof).

Particle belief and MCTS approximation. In practice, the current belief is represented by a particle approximation $\begin{array} { r } { b _ { t } ^ { J } = \frac { 1 } { J } \sum _ { j = 1 } ^ { \bar { J } } \delta _ { s _ { t } ^ { ( j ) } } } \end{array}$ . Starting from $b _ { t } ^ { J }$ , MCTQ builds a search tree of depth d with N Monte Carlo simulations per root action, inspired by the POMCP framework [34]. Each simulation samples a particle from $b _ { t } ^ { J }$ , descends to depth d using uniform random action selection at internal nodes, expands nodes as it passes through them, and records its state in the particle set $B ( v )$ of every node v it visits. Each depth-d leaf is evaluated by the QMDP value of its pooled particles, max<sub>a</sub>′ $\begin{array} { r } { \dot { | \boldsymbol { B } ( v ) | ^ { - 1 } } \sum _ { s \in { \boldsymbol { B } ( v ) } } \boldsymbol { Q } _ { \boldsymbol { \theta } } ^ { \star , S } ( s , a ^ { \prime } ) } \end{array}$ , and a bottom-up dynamic program then computes $\widehat { Q } _ { t } ( \bar { b } _ { t } ^ { J } , a )$ for each root action. We use uniform selection for the analysis and do not claim it is the best rule in practice. The analysis only needs the internal rule to ignore the current tree’s statistics and to give every action probability at least $\pi _ { \operatorname* { m i n } } > 0$ , so a frozen learned proposal also qualifies, but UCB1 does not (Appendix B).

Observation discretization. With continuous O, every simulated observation is distinct, so tree nodes below the root are never revisited and a depth-d tree is not well defined. MCTQ therefore indexes children by a binned observation $\phi ( o )$ , where $\phi : \mathcal { O } \to \bar { \mathcal { O } }$ has $M : = | \bar { \mathcal { O } } |$ bins of diameter diam(ϕ). For finite $\mathcal { O }$ we take $\phi = \mathrm { i d }$ and $M = | \mathcal { O } |$ , and for $d = 0$ no binning is needed. The resulting bias and its trade-of against N are made explicit in Theorem 1.

## 4 Error Analysis and Regret Guarantee

We measure performance by the expected cumulative regret defined as:

$$
\mathrm { R e g } _ { T } : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \mathrm { r e g r e t } _ { t } \right] : = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \bigl ( V _ { \theta ^ { \star } } ^ { \star , B } ( b _ { t } ) - V _ { \theta ^ { \star } } ^ { \pi _ { t } , B } ( b _ { t } ) \bigr ) \right] ,\tag{10}
$$

where $\pi _ { t }$ is the policy deployed from round t, and the expectation is over the trajectory and algorithmic randomness. A key contribution is an end-to-end regret bound for PF-MCTQ, enabled by the following novel regret decomposition. Let $k ( t )$ be the epoch containing round t, with frozen model ${ \bar { \theta } } _ { k ( t ) }$

$$
\begin{array} { r } { \mathrm { r e g r e t } _ { t } = \underbrace { V _ { \theta ^ { \star } } ^ { \star , B } ( b _ { t } ) - V _ { \widetilde { \theta } _ { k } ( t ) } ^ { \star , B } ( b _ { t } ) } _ { \mathrm { ( A ) ~ o p t i m a l - v a l u e ~ m o d e l ~ e r o r ~ } } + \underbrace { V _ { \widetilde { \theta } _ { k } ( t ) } ^ { \star , B } ( b _ { t } ) - V _ { \widetilde { \theta } _ { k } ( t ) } ^ { \pi _ { t } , B } ( b _ { t } ) } _ { \mathrm { ( B ) ~ p l a n n i n g + ~ s t a t e - i d ~ e r r o r ~ } } + \underbrace { V _ { \widetilde { \theta } _ { k } ( t ) } ^ { \pi _ { t } , B } ( b _ { t } ) - V _ { \theta ^ { \star } } ^ { \pi _ { t } , B } ( b _ { t } ) } _ { \mathrm { ( C ) ~ f i x e d - p o l i c y ~ m o d e l ~ e r r o r ~ } } , } \end{array}\tag{11}
$$

Terms (A) and (C) capture model estimation error: the gap between ${ \bar { \theta } } _ { k ( t ) }$ and $\theta ^ { \star }$ . Term (B) captures planning approximation under the learned model, which is further decomposed into three sources:

$$
\begin{array} { r } { \mathrm { ( B ) } ~ \le ~ \underbrace { \big | V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B } ( b _ { t } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B } ( b _ { t } ^ { J } ) \big | } _ { \mathrm { ( B 1 ) ~ b e l i e f ~ a p p r o x . } } + \underbrace { V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B } ( b _ { t } ^ { J } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi _ { t } , B } ( b _ { t } ^ { J } ) } _ { \mathrm { ( B 2 ) ~ p l a n n i n g ~ s u b o p t i m a l i t y } } + \underbrace { \big | V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi _ { t } , B } ( b _ { t } ^ { J } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi _ { t } , B } ( b _ { t } ) \big | } _ { \mathrm { ( B 1 ^ { \prime } ) ~ b e l i e f ~ a p p r o x . } } . } \end{array}\tag{12}
$$

Terms (B1) and (B1<sup>′</sup>) evaluate values at the particle belief $b _ { t } ^ { J }$ instead of the posterior $b _ { t }$ , each bounded by ${ \cal O } ( 1 / J ^ { 1 / ( 2 \kappa _ { \mathrm { p f } } ) } )$ via particle-filter concentration. The planning gap (B2) splits into Monte Carlo error from MCTS rollouts $( O ( 1 / { \sqrt { N } } ) )$ and structural error from QMDP bootstrapping at depth d.

We distinguish the belief-MDP optimum $V _ { \theta } ^ { \star , B }$ , the deterministic planning target $\bar { V } _ { \theta , d } ^ { \star , S } = T _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S }$ of equation 8, and the randomized MCTQ estimate $\widehat { Q } _ { t } ( b _ { t } ^ { J } , a )$ from pooled particle sets. The target difers from the optimum by the structural error, which decays geometrically in d and involves no particles. The estimate difers from the target by the Monte Carlo error, which grows with $d ,$ so for fixed N the bound is U-shaped in d. Each component enters only through its guarantee (a β-approximate MLE, an SMC filter with an $L ^ { p }$ bound, a planner with a Monte Carlo bound), so the decomposition extends to, $\mathrm { e . g . }$ , POMCP-style planners or variational filters.

## 4.1 Assumptions and main result

Modern SMC analyses [6, 9, 21] impose rather unavoidable but stringent conditions on the transition kernel and observation density. We inherit these in the form of the following Doeblin and bounded observation density assumption (Assumption 1) on the latent dynamics. This choice aligns with our aim to provide an end-to-end regret bound that decomposes error across the learning–inference–planning pipeline, rather than to develop optimal exploration methods for POMDP model estimation, as Assumption 1 ensures that simple exploration strategies like ε-greedy have suficient coverage. At the same time, the Doeblin condition resembles commonplace strong mixing assumptions within the RL literature [13, 23, 39, 40] and the α-revealing condition commonplace in the POMDP literature plays a similar role to the bounded observation density condition [28, 29].

Assumption 1 (Uniform mixing and bounded weights).

1. Uniform Doeblin condition. There exist $\varepsilon _ { \operatorname* { m i x } } > 0$ and a probability measure $\nu \in \Delta ( \mathcal { S } )$ such that $P _ { \theta } ( \cdot \mid s , a ) \ge \varepsilon _ { \mathrm { m i x } } \nu ( \cdot )$ for all $( \theta , s , a ) \in \Theta \times \mathcal { S } \times \mathcal { A }$

2. Bounded observation density. There exist $0 < g _ { \mathrm { m i n } } \le g _ { \mathrm { m a x } } < \infty$ such that $g _ { \mathrm { m i n } } \leq Z _ { \theta } ( o \mid s ) \leq$ g<sub>max</sub> for all $( \theta , s , o ) \in \Theta \times \mathcal { S } \times \mathcal { O }$

Let $\mathcal { N } _ { \Theta } ( \varepsilon )$ be the ε-covering number of Θ, the minimum number of $\ell _ { 1 }$ -balls of radius ε (in trajectory likelihood) needed to cover Θ. Define the efective model complexity $\beta _ { T } : = \beta + \log \left[ \Lambda _ { \Theta } ( 1 / T ) T / \delta \right]$ where $\beta$ is the MLE tolerance from equation 6 and $\delta \in ( 0 , 1 )$ is a confidence parameter.

Theorem 1 (End-to-end regret of Pf-Mctq). Suppose Assumption 1 holds. Algorithm 1, run with exploration parameter $\eta \asymp T ^ { 1 / 4 } ( \beta _ { T } / \log T ) ^ { 1 / 4 }$ and observation binning $\phi$ with M bins of diameter $\mathrm { d i a m } ( \phi ) = \omega _ { Z } ^ { - 1 } \big ( ( 1 - \gamma ) g _ { \mathrm { m i n } } / \big ( 2 \gamma g _ { \mathrm { m a x } } V _ { \mathrm { m a x } } \sqrt { N } \big ) \big ) \ ( \phi = \mathrm { i d }$ and $M = | \mathcal { O } |$ when O is finite), and, $i f$ $d \geq 2$ , N above the threshold of Remark 3, and $\kappa _ { \mathrm { p f } } : = \operatorname* { m a x } \left( 1 , \log \left( 2 g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } \right) / \log ( 1 / \gamma ) \right)$ satisfies

$$
\mathrm { R e g } _ { T } \leq \underbrace { \widetilde { O } \biggl ( \frac { C _ { \mathrm { e s t } } R _ { \mathrm { m a x } } } { ( 1 - \gamma ) ^ { 3 / 2 } } T ^ { 3 / 4 } \biggr ) } _ { m o d e l + e x p l o r a t i o n + f i t e r } + \underbrace { \frac { C _ { \mathrm { p f } } R _ { \mathrm { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \cdot \frac { T } { J ^ { 1 / ( 2 \kappa _ { \mathrm { p f } } ) } } } _ { p a r i c l e a p r o x . } + \underbrace { \frac { C _ { \mathrm { m c } } R _ { \mathrm { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \cdot \frac { T } { \sqrt { N } } } _ { M o n t c ~ C a r l o } + \underbrace { \frac { 2 \gamma ^ { d + 1 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { t = 1 } ^ { T } \Xi _ { \tilde { \theta } _ { k ( t ) } } } _ { Q M D P ~ s t r u c t u r l ~ b i a s } ,
$$

Here $\widetilde O$ hides factors polylogarithmic in $T , \omega _ { Z }$ is the modulus of continuity of the observation likelihood ratio (Appendix D.2), $M = | \bar { \mathcal { O } } |$ is the number of observation bins, and $\Xi _ { \bar { \theta } _ { k ( t ) } }$ is the QMDP structural bias of equation 13. The coeficients are

$$
C _ { \mathrm { e s t } } = \mathrm { P o l y } ( | A | , g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } } , \epsilon _ { \mathrm { m i x } } ^ { - 1 } , \log ( 1 / \gamma ) , \log \mathcal { N } _ { \Theta } ( 1 / T ) , \beta ) , \quad C _ { \mathrm { m c } } = O ( d \gamma ^ { d } | A | ^ { ( d + 3 ) / 2 } M ^ { d / 2 } ) ,
$$

$$
C _ { \mathsf { p f } } = { \widetilde { O } } { \big ( } V _ { \operatorname* { m a x } } ^ { 2 } g _ { \operatorname* { m a x } } / ( \varepsilon _ { \operatorname* { m i x } } ^ { 3 } g _ { \operatorname* { m i n } } ) \cdot ( \log J ) ^ { 3 / 2 } { \sqrt { \log ( \mathcal { N } _ { \Theta } ( 1 / T ) \left| A \right| ) } } { \big ) } { \pmod { \mathsf { s e e ~ e q u a t i o n ~ 5 7 } } } .
$$

Reading the bound. Theorem 1 bounds cumulative regret with four major terms. The first term captures model estimation error, exploration cost, and filter mismatch between the true posterior $b _ { t } ^ { \theta ^ { \star } }$ and the particle belief $b _ { t } ^ { \bar { \theta } _ { 1 : K } }$ . The ${ \widetilde O } ( T ^ { 3 / 4 } )$ rate arises from the coverage transfer needed to bound the optimal-value model error (term (A) in equation 11) via ε-greedy exploration. Terms 2, 3, and 4 capture the total approximate planning errors that may scale linearly in T. They can be diminished by increasing the number of particles J, number of Monte Carlo rollouts N, and the planning depth d. The particle error decays as $J ^ { - 1 / ( 2 \kappa _ { \mathrm { p f } } ) }$ at cost $O ( J )$ , the Monte Carlo error as $N ^ { - 1 / 2 }$ at cost $O ( N )$ , and the structural error as $\gamma ^ { d }$ at a tree size of up to $( | { \cal { A } } | { \cal { M } } ) ^ { d }$ nodes. The bound therefore implies a sample-vs.-computational eficiency trade-of through the marginal return of each budget, but a concrete allocation requires knowledge of $\kappa _ { \mathrm { p f } }$ and $\Xi _ { \theta }$ and is not claimed to be optimal. Section 5 gives an empirical budgeting rule on an instance where the assumptions hold. With continuous observations, the binning keeps the discretization bias at the Monte Carlo order $O ( V _ { \mathrm { m a x } } / \sqrt { N } )$ , but M grows as $N ^ { d _ { o } / 2 }$ for $\mathcal { O } \subseteq \mathbb { R } ^ { d _ { o } }$ , so $C _ { \mathrm { m c } }$ grows exponentially in $d \cdot d _ { o }$ . Balancing bin width against N instead gives the rate $\widetilde { O } ( N ^ { - 1 / ( d d _ { o } + 2 ) } )$ (Remark 4). This cost is absent only at $d = 0$ , the regime used in our deep RL experiments.

Technical challenges. Bounding each source in equation 12 under online learning raises three dificulties. First, controlling (B1) requires working with a test-function class defined directly over $s ,$ rather than only the parametric family $\{ Q _ { \theta } ^ { \star , B } \} _ { \theta }$ . This is because the belief-space value functions may lack the regularity needed for uniform particle-filter error control. In fact, the planning-induced class over multiple steps is efectively uncountable over the possible model parameters, so we control it through truncation. The required truncation level depends on the desired particle approximation accuracy: as the particle error target decreases, the tail must be made smaller, which enlarges the truncated function class. This creates a tradeof between the complexity of the truncated class and the particle ensemble size, leading to a slower $\left( 1 / J ^ { 1 / 2 \kappa _ { \mathrm { p f } } } \right)$ , essentially nonparametric, rate.

Second, the filtering analysis must handle data generated online by a changing policy. Classical particle-filter stability assumes a fixed generative model, whereas here the filter receives actions from a non-stationary, improving policy. We extend time-uniform $L ^ { p }$ bounds [9] to this setting by exploiting the Doeblin condition uniformly over actions. Third, Doeblin coverage applies to states but not actions, so we ensure action coverage via ε-greedy exploration. This lets us transfer MLE concentration across policies, controlling terms (A) and (C) and the Monte Carlo error in (B2).

Remark 1 (Relation to POMDP hardness). POMDP sample complexity may in the worst case scale exponentially in the efective horizon $1 / ( 1 - \gamma )$ . Our bound is polynomial in $1 / ( 1 - \gamma )$ for two reasons. First, under the Doeblin condition the Bayes filter forgets its initialization at a geometric rate $\rho = 1 - \varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } / g _ { \mathrm { m a x } } < 1$ uniformly over actions, so errors add up over time rather than compound. Second, bounded observation densities play the role of the α-revealing condition [28]: they bound importance weights, so data transfer across policies without an $( | \mathcal { A } | / \varepsilon ) ^ { H }$ blow-up, and particle weights stay non-degenerate. This does not circumvent the lower bound: $\varepsilon _ { \mathrm { m i x } } ^ { - 1 }$ and $g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } }$ can be exponential in the efective horizon for hard instances, and $C _ { \mathrm { e s t } }$ and $C _ { \mathrm { p f } }$ depend polynomially on them. Appendix F illustrates failures when $g _ { \mathrm { m i n } } = 0$ and when $\Xi _ { \theta }$ is large.

Remark 2 (Bypassing exploration with ofline coverage). The ${ \widetilde O } ( T ^ { 3 / 4 } )$ rate is driven by the coverage transfer for term $( A )$ , which requires ε-greedy exploration to ensure model accuracy at the optimal policy. If instead the agent has access to $N _ { \mathrm { o f f } }$ ofline trajectories from a behavior policy $\pi _ { b }$ with single-policy concentrability $c _ { \mathrm { o f f } } ^ { \star }$ (the worst-case ratio of trajectory $\ell _ { 1 }$ distances between $\pi ^ { \star }$ and $\pi _ { b } )$ then greedy exploitation $( \varepsilon _ { t } = 0 )$ achieves regret $\widetilde { \cal O } ( \sqrt { T } / ( 1 - \gamma ) ) + { \cal O } ( c _ { \mathrm { o f f } } ^ { \star } T / ( \sqrt { N _ { \mathrm { o f f } } } ( 1 - \gamma ) ) )$ plus computational terms. When $N _ { \mathrm { o f f } } \geq ( c _ { \mathrm { o f f } } ^ { \star } ) ^ { 2 } T$ , the total regret is $\overset { \triangledown } { \partial } ( \sqrt { T } )$ without any additional online exploration. The formal statement and proof are in Appendix D.7.

## 4.2 Understanding the structural bias

It remains to discuss the QMDP approximation error, which is governed by the Jensen gap

$$
\Xi _ { \theta } : = \operatorname* { s u p } _ { b \in \Delta ( S ) } \left( \int _ { S } V _ { \theta } ^ { \star , S } ( s ) b ( \mathrm { d } s ) - \operatorname* { m a x } _ { a \in \mathcal { A } } \int _ { S } Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) \right) ,\tag{13}
$$

which measures the value lost by choosing a single action for all states (QMDP planning) rather than the state-dependent optimal action. We show (Proposition 2 in Appendix D.1) that the depth-d MCTQ structural error satisfies $\begin{array} { r } { \lVert \bar { V } _ { \theta , d } ^ { \star , S } - V _ { \theta } ^ { \star , B } \rVert _ { \infty } \leq \frac { \gamma ^ { d + 1 } } { 1 - \gamma } \Xi _ { \theta } } \end{array}$ , decaying geometrically with depth d.

The quantity $\Xi _ { \theta }$ has an interpretation via the posterior variance of value-relevant features. Let $g : S  \mathbb { R } ^ { k }$ be a measurable embedding, and write $m _ { g } ( b ) : = \mathbb { E } _ { S \sim b } [ g ( S ) ]$ and $\Sigma _ { g } ( b ) : = \operatorname { C o v } _ { S \sim b } ( g ( S ) )$

Assumption 2 (Value-relevant embedding). There exist twice continuously diferentiable maps $F _ { \theta } ^ { a } : \mathbb { R } ^ { k }  \mathbb { R }$ for each $a \in { \mathcal { A } }$ such that $Q _ { \theta } ^ { \star , S } ( s , a ) = F _ { \theta } ^ { a } ( g ( s ) )$ for all $( s , a ) \in \mathcal S \times \mathcal A$ , with curvature bounds, where $F _ { \theta } ( x ) : = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( x )$ and $V _ { \theta } ^ { \star , S } ( s ) = F _ { \theta } ( g ( s ) )$ , of

$$
\operatorname* { s u p } _ { a \in \mathcal { A } } \operatorname* { s u p } _ { x \in \mathrm { c o n v } ( g ( S ) ) } \left\| \nabla ^ { 2 } F _ { \theta } ^ { a } ( x ) \right\| _ { \mathrm { o p } } \leq L _ { \theta } \qquad a n d \qquad \operatorname* { s u p } _ { x \in \mathrm { c o n v } ( g ( S ) ) } \left\| \nabla ^ { 2 } F _ { \theta } ( x ) \right\| _ { \mathrm { o p } } \leq L _ { \theta } ,
$$

where $\| \cdot \| _ { \mathrm { o p } }$ is the spectral norm.

In practice, the condition holds when either (i) a single action is optimal throughout $\mathrm { c o n v } ( g ( S ) )$ or (ii) $F _ { \theta }$ is a single smooth function of s (as in linear-quadratic-Gaussian (LQG) systems, where it is a quadratic form and both conditions hold with the same $L _ { \theta } )$

Proposition 1 (Curvature bound on $\Xi _ { \theta } )$ . Under Assumption 2, for every belief $b \in \Delta ( \mathcal { S } )$

$$
\Xi _ { \theta } ( b ) : = \widetilde { V } _ { \theta } ^ { \star } ( b ) - \bar { V } _ { \theta } ^ { \star , S } ( b ) \ \leq \ L _ { \theta } \operatorname { t r } \bigl ( \Sigma _ { g } ( b ) \bigr ) ,\tag{14}
$$

where $\begin{array} { r } { \widetilde { V } _ { \theta } ^ { \star } ( b ) : = \int _ { S } V _ { \theta } ^ { \star , S } ( s ) b ( \mathrm { d } s ) } \end{array}$ is the linear QMDP surrogate. In particular, $\Xi _ { \theta } = \operatorname* { s u p } _ { b } \Xi _ { \theta } ( b )$

The proof is in Appendix D.8. That QMDP is accurate once the belief concentrates is expected; Proposition 1 is sharper in two ways. First, only the posterior covariance of the value-relevant features g matters; uncertainty in directions to which the value is insensitive does not enter, so the full belief need not concentrate. Second, the curvature $L _ { \theta }$ converts this covariance into a value gap, which enters Theorem 1 as $\gamma ^ { d + 1 } { \Xi } _ { \theta } / ( 1 - \gamma )$ and trades of against planning depth. In discounted LQG, for instance, the Kalman covariance converges to a strictly positive steady state and never vanishes, yet the QMDP policy is belief-optimal, and $\Xi _ { \theta } = 0$ when the control does not enter the dynamics (Appendix D.9).

Appendix D.9 instantiates the structural analysis on linear-quadratic Gaussian systems, where $\Xi _ { \theta }$ has a closed form in terms of the Riccati solution and the steady-state Kalman covariance, and on contextual MDPs with identifiable latent context, where $\Xi _ { \theta , t } \leq 2 V _ { \operatorname* { m a x } } \rho _ { t }$ for a context-misidentification probability $\rho _ { t } ,$ , so eventual identification becomes the rate condition $\begin{array} { r } { \sum _ { t } \rho _ { t } = o ( \sqrt { T } ) } \end{array}$

## 5 Experiments

Our experiments test whether particle filtering and model-based planning can be used as a practical layer on top of modern deep reinforcement learning under partial observability. Here, our goal is not to obtain state-of-the-art asymptotic performance. Instead, we aim to evaluate whether explicit particle beliefs can improve low-sample decision making with modest additional overhead. Table 1 states which experiments fall under the theory. The deep RL experiments demonstrate that the pipeline is practical but are not evidence for the constants of Theorem 1.

(a) Minimal +, zero -  
![](images/f1be0b65117311adfeca65f53cf8b5f6326b2783858bca498f984ee22e62015e.jpg)

(b) Minimal +, minimal -  
![](images/bcada4ebafa33ac4492eaecacf7d91235fd2f94946374a527d8d198bfaab4fcb.jpg)

![](images/1b5a1bcdf81167125bf972f30fd1dd56b5e52ac8e8cc0fa19b6654b380fb886d.jpg)

![](images/3b535305d180a9b76a1713c3673e4305279b651d2b463e34a09cab9785d1572d.jpg)  
Figure 2: Cumulative reward above the baseline rollout in four HeartSteps scenarios varying the positive intervention efect and the delayed negative measurement efect or proxy $O _ { t }$ informativeness. Mean and 2.5%/97.5% quantiles over five seeds.

Table 1: Which experiments satisfy the assumptions of Theorem 1. $\checkmark \colon$ holds; ×: fails; ∼: approx.
<table><tr><td>Environment</td><td>Doeblin</td><td> $g _ { \mathrm { m i n } } > 0$ </td><td>Finite A</td><td>Bounded r</td><td>Role</td></tr><tr><td>Mixing Tiger (Sec. 5)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>theory-valid</td></tr><tr><td>Active sensing (App. F)</td><td>V</td><td>2</td><td>√</td><td>√</td><td>structural diagnostic</td></tr><tr><td>Box search (App. F)</td><td>X</td><td>X</td><td>√</td><td>√</td><td> $g _ { \mathrm { m i n } } = 0$  by design</td></tr><tr><td>HeartSteps</td><td>2</td><td>2</td><td>√</td><td>√</td><td>not claimed</td></tr><tr><td>Double-integrator LQR</td><td>√</td><td>X</td><td>X</td><td>X</td><td>filtering diagnostic</td></tr><tr><td>MuJoCo</td><td>X</td><td>X</td><td>X</td><td>X</td><td>heuristic demonstration</td></tr></table>

Empirical validation of the regret decomposition. On a Doeblin-mixing Tiger [19] that satisfies Assumption 1 exactly $( \varepsilon _ { \mathrm { m i x } } = 0 . 0 5 , 0 . 1 5 \leq Z \leq 0 . 8 5 , \gamma = 0 . 8 ;$ Appendix E), varying one budget at a time shows each term decaying at its predicted rate (Table 2). The smallest tested (J, N, d) putting every isolated term below 0.10, 0.05, and 0.025 is (16, 16, 6), (64, 64, 9), and (128, 256, 12).

Deep RL experiments. We instantiate PF-MCTQ using the pypomp [1] package for particlefiltering primitives with MOP [37] or VSMC [30] gradient estimators. This allows highly optimized JAX particle filters on GPU. We learn a neural dynamics model online, maintain a particle approximation to the latent state posterior, and use an actor and a critic trained with recurrent SAC-style updates [31] in the same vein as [16]. In the experiments we use belief-tree depth $d = 0 \colon$ each particle is rolled out at the state level under a learned actor and bootstrapped with a learned critic, rather than planning over a tree of particle beliefs. With continuous actions and observations a depth-d tree requires binning at a cost of up to $( | { \cal { A } } | { \cal { M } } ) ^ { d }$ nodes, so this is the main practical gap from the theory. Rollout planning corresponds to the QMDP case d = 0, whereas deeper belief planning reduces the structural bias. Appendix G runs the analyzed MCTQ at $d > 0$ on finite POMDPs. Algorithm 4 gives the practical algorithm. All experiments took about three days on two RTX 3090 GPUs. Code is available at https://github.com/pypomp/pf\_control.

Double-integrator LQR. We first study a partially observed double-integrator LQR. The state contains position and velocity, but the observation reveals only position. Thus the latent velocity is therefore not identifiable from the current observation alone. Diferent velocities induce the same observation and can only be distinguished using history. This environment isolates the filtering problem. The posterior diagnostics in Figure 1 show that particle beliefs can track the latent state without access to memory, while failures occur when the system is unstable. The GRU backbone improves this behavior relative to a memoryless MLP, suggesting that particle filtering and recurrent state representations are complementary. More details and figures can be found in Appendix C.1.

Table 2: Fitted vs. predicted rates on mixing Tiger; Theorem 1 gives slower $J ^ { - 1 / ( 2 \kappa _ { \mathrm { p f } } ) } , T ^ { 3 / 4 }$
<table><tr><td>Term</td><td>Error at endpoints</td><td>Fitted</td><td>Predicted</td></tr><tr><td>Particle filtering,  $J = 8  5 1 2$ </td><td> $0 . 1 4 4  0 . 0 1 2$ </td><td> $J ^ { - 0 . 5 9 }$ </td><td> $J ^ { - 1 / 2 }$ </td></tr><tr><td>Monte Carlo, N = 4 → 512</td><td>0.171 → 0.015 (RMSE)</td><td> $N ^ { - 0 . 5 0 }$ </td><td> $N ^ { - 1 / 2 }$ </td></tr><tr><td>Structural, d = 0 → 12</td><td> $0 . 3 1 4  0 . 0 2 1 6$ </td><td> $0 . 8 0 \ \mathrm { p e r }$  level</td><td> $\gamma = 0 . 8 0$ </td></tr><tr><td>Estimation (online MLE),  $\textstyle \sum _ { t \leq T } \varepsilon _ { \mathrm { o p t } } ( t )$ </td><td></td><td> $T ^ { 0 . 5 7 }$ </td><td> $\sqrt { T }$ </td></tr></table>

![](images/654dbcb9c42a7324f529366c82c97a8e0a3421e1fa45cca8a064652b6d97de8a.jpg)  
Figure 3: Partial-observation MuJoCo learning curves in the low-sample regime: smoothed means with 95% Student-t bands over five seeds (Ant: three seeds for corrected PF-MCTQ; the submitted configuration, dashed, shows its two seeds as thin lines). Ant PF-MCTQ shows the 10-episode rolling training return; others, per-episode returns. Dotted: the submitted 3k-step cutof; none extrapolated.

HeartSteps. We next evaluate the HeartSteps benchmark of [12] as an actively observable MDP with a latent reward state, where the learner can choose to measure the state at a cost. PF-MCTQ is run in the never-measure regime. It never directly measures the latent reward and must infer it from passive observations. The strongest benchmark is RLSVI-SMC/never-measure, which is specifically tailored to the linear-Gaussian HeartSteps structure and uses hand-specified linear features. Consequently, RLSVI-SMC performs best in most scenarios. PF-MCTQ typically improves over the baseline rollout but trails RLSVI-SMC because it must learn a deep model, actor, and critic from data. Dyna-ATMQ is not directly comparable as a no-measure method. It actively measures the latent reward roughly half the time, which gives it an informational advantage, but this advantage can be ofset when measurement has delayed negative efects on future engagement. Details are in Appendix C.2.

MuJoCo partial-observation control. Finally, we evaluate Ant, HalfCheetah, Hopper, and Walker2d with position or velocity removed from the observation. We compare PF-MCTQ to Recurrent SAC and GRU-PPO baselines from the recurrent model-free POMDP benchmarking literature [31]. These are strong comparators that learn a recurrent history representation directly from reward. Figure 3 shows that PF-MCTQ is competitive in the ultra-low-sample regime. On Ant, PF-MCTQ learns a high-return policy within the first few thousand steps under both ablations. We no longer stop Ant early. When run to 49.2k steps, the submitted configuration’s policy collapses (dashed). After restoring online model updates and ordinary weight decay, a controlled reduction of the policy update-to-data ratio (one update per ten steps) is decisive and prevents this (Appendix H). Over the prespecified 40k–49.2k window, corrected PF-MCTQ averages $8 9 3 \pm 1 7 7$ and $8 2 2 \pm 2 6 8$ (position/velocity hidden; three seeds, 95% Student-t) against −396 and −518 for the submitted configuration. On Hopper and Walker2d, PF-MCTQ is broadly competitive with Recurrent SAC and usually stronger than GRU-PPO early in training. HalfCheetah is more mixed: PF-MCTQ performs well when velocity is hidden but struggles when position is hidden, suggesting that model bias and rollout error can dominate when the omitted coordinates are crucial for short-horizon prediction.

On Ant, corrected PF-MCTQ needs about 30% of recurrent SAC’s training time per step plus a one-time 86.3 s model warm start, while GRU-PPO is faster still (Table 15). Overall, these results show that particle filtering and model-based planning can be implemented on top of existing deep RL algorithms and can yield sample-eficiency gains in partially observed control. The results should not be read as a claim of asymptotic dominance over recurrent model-free RL. Rather, they show that the learning–filtering–planning pipeline can be made operational in standard deep RL benchmarks.

## 6 Discussion

We presented an online model-based POMDP pipeline combining particle filtering, latent-dynamics learning, and approximate planning with an end-to-end regret bound. We decompose regret into model estimation, particle approximation, Monte Carlo planning, and QMDP structural errors, clarifying how statistical and computational budgets afect performance. The framework bridges POMDP theory and model-based deep RL. Experiments with a simplified rollout-based version show that it can be made operational and competitive in low-sample partially observed control.

Limitations. Our analysis relies on the Doeblin mixing condition and bounded observation densities, which ensure passive state coverage and obviate the need for strategic exploration. These assumptions fail in many POMDPs. In item search with exact observations, the observation density is not bounded below, and a particle set of size J misses the true location among K with probability $( 1 - 1 / K ) ^ { J }$ . In Light–Dark-type problems, information must be actively gathered, so $\Xi _ { \theta }$ is large and shallow planning never gathers it (see Appendix F). Second, the bounded reward assumption excludes settings such as LQG. Third, the deep RL experiments use d = 0 with a learned critic and actor, which is the QMDP case of our algorithm, and our guarantees apply to non-adaptive action selection inside the tree but not to UCT-style selection. Natural extensions are strategic exploration to relax the dynamics assumptions, unbounded rewards, and guarantees for adaptive tree search.

## Acknowledgments

Ziping Xu was supported by the National Science Foundation under award DMS-2625746.

## References

[1] Aaron Abkemeier, Jun Chen, Kevin Tan, Jesse Wheeler, and Edward Ionides. pypomp: Partially observed Markov processes in Python, version 0.4.4.10. https://github.com/pypomp/pypomp, 2026.

[2] Ameya Anjarlekar, S. Rasoul Etesami, and R. Srikant. Scalable policy-based RL algorithms for POMDPs. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[3] Lilian Besson and Emilie Kaufmann. What doubling tricks can and can’t do for multi-armed bandits. arXiv preprint arXiv:1803.06971, 2018.

[4] Anthony R Cassandra, Leslie Pack Kaelbling, and Michael L Littman. Acting optimally in partially observable stochastic domains. In Proceedings of the Twelfth National Conference on Artificial Intelligence (AAAI), pages 1023–1028, 1994.

[5] Yuan Chen, Mathieu Gerber, Christophe Andrieu, and Randal Douc. Self-organizing state-space models with artificial dynamics. Journal of the Royal Statistical Society Series B: Statistical Methodology, page qkaf069, 2025.

[6] Nicolas Chopin and Omiros Papaspiliopoulos. An Introduction to Sequential Monte Carlo. Springer Series in Statistics. Springer, 2020.

[7] Adrien Corenflos, James Thornton, George Deligiannidis, and Arnaud Doucet. Diferentiable particle filtering via entropy-regularized optimal transport. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 2100–2111. PMLR, 18–24 Jul 2021.

[8] Dan Crisan and Arnaud Doucet. A survey of convergence results on particle filtering methods for practitioners. IEEE Transactions on signal processing, 50(3):736–746, 2002.

[9] Pierre Del Moral. Feynman-Kac Formulae: Genealogical and Interacting Particle Systems with Applications. Springer, 2004.

[10] Randal Douc, Eric Moulines, and Jimmy Olsson. Long-term stability of sequential Monte Carlo methods under verifiable conditions. Annals of Applied Probability, 24(5):1767–1802, 2014.

[11] Simon S Du, Wei Hu, Zhiyuan Li, Ruoqi Shen, Zhao Song, and Jiajun Wu. When is particle filtering eficient for planning in partially observed linear dynamical systems? In Uncertainty in artificial intelligence, pages 728–737. PMLR, 2021.

[12] Daiqi Gao, Ziping Xu, Aseel Rawashdeh, Predrag Klasnja, and Susan Murphy. Active measuring in reinforcement learning with delayed negative efects. In The 29th International Conference on Artificial Intelligence and Statistics, 2025.

[13] Mudit Gaur, Amrit Singh Bedi, Di Wang, and Vaneet Aggarwal. Closing the gap: achieving global convergence (last iterate) of actor-critic under markovian sampling with neural network parametrization. In Proceedings of the 41st International Conference on Machine Learning, pages 15153–15179, 2024.

[14] Noah Golowich, Ankur Moitra, and Dhruv Rohatgi. Planning in observable pomdps in quasipolynomial time. arXiv preprint arXiv:2201.04735, 2022.

[15] Noah Golowich, Ankur Moitra, and Dhruv Rohatgi. Planning and learning in partially observable systems via filter stability. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 349–362, 2023.

[16] Chao Han, Stefanos Ioannou, Luca Manneschi, T. J. Hayward, Michael Mangan, Aditya Gilra, and Eleni Vasilaki. Neural ode and sde models for adaptation and planning in model-based reinforcement learning. arXiv preprint arXiv:2603.23245, 2026.

[17] Milos Hauskrecht and Hamish Fraser. Planning treatment of ischemic heart disease with partially observable markov decision processes. Artificial Intelligence in Medicine, 18(3):221–244, 2000.

[18] Edward L. Ionides, Dao Nguyen, Yves Atchadé, Stilian Stoev, and Aaron A. King. Inference for dynamic and latent variable models via iterated, perturbed Bayes maps. Proceedings of the National Academy of Sciences, 112(3):719–724, 2015.

[19] Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

[20] Ali Devran Kara and Serdar Yüksel. Convergence of finite memory q learning for pomdps and near optimality of learned policies under filter stability. Mathematics of Operations Research, 48(4):2066–2093, 2023.

[21] Joona Karjalainen, Anthony Lee, Sumeetpal S. Singh, and Matti Vihola. On the forgetting of particle filters. arXiv preprint arXiv:2309.08517, 2026.

[22] Peter Karkus, David Hsu, and Wee Sun Lee. QMDP-Net: Deep learning for planning under partial observability. In Advances in Neural Information Processing Systems, volume 30, 2017.

[23] Chinmaya Kausik, Kevin Tan, and Ambuj Tewari. Learning mixtures of Markov chains and MDPs. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 15970–16017. PMLR, 23–29 Jul 2023.

[24] Chinmaya Kausik, Kevin Tan, and Ambuj Tewari. Learning mixtures of markov chains and mdps. In International Conference on Machine Learning, pages 15970–16017. PMLR, 2023.

[25] Michael H Lim, Tyler J Becker, Mykel J Kochenderfer, Claire J Tomlin, and Zachary N Sunberg. Optimality guarantees for particle belief approximation of pomdps. Journal of Artificial Intelligence Research, 77:1591–1636, 2023.

[26] Michael H Lim, Claire J Tomlin, and Zachary N Sunberg. Sparse tree search optimality guarantees in pomdps with continuous observation spaces. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence, pages 4135–4142, 2021.

[27] Michael L Littman, Anthony R Cassandra, and Leslie Pack Kaelbling. Learning policies for partially observable environments: Scaling up. In Machine learning proceedings 1995, pages 362–370. Elsevier, 1995.

[28] Qinghua Liu, Alan Chung, Csaba Szepesvári, and Chi Jin. When is partially observable reinforcement learning not scary? In Conference on Learning Theory, pages 5175–5220. PMLR, 2022.

[29] Qinghua Liu, Praneeth Netrapalli, Csaba Szepesvari, and Chi Jin. Optimistic MLE: A generic model-based algorithm for partially observable sequential decision making. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing, pages 363–376, 2023.

[30] Christian Naesseth, Scott Linderman, Rajesh Ranganath, and David Blei. Variational sequential monte carlo. In International conference on artificial intelligence and statistics, pages 968–977. PMLR, 2018.

[31] Tianwei Ni, Benjamin Eysenbach, and Ruslan Salakhutdinov. Recurrent model-free RL is a strong baseline for many pomdps. CoRR, abs/2110.05038, 2021.

[32] Christos H Papadimitriou and John N Tsitsiklis. The complexity of markov decision processes. Mathematics of operations research, 12(3):441–450, 1987.

[33] Joelle Pineau, Geof Gordon, Sebastian Thrun, et al. Point-based value iteration: An anytime algorithm for POMDPs. In Ijcai, volume 3, pages 1025–1032, 2003.

[34] David Silver and Joel Veness. Monte-carlo planning in large pomdps. Advances in neural information processing systems, 23, 2010.

[35] Yuda Song, Yifei Zhou, Ayush Sekhari, J Andrew Bagnell, Akshay Krishnamurthy, and Wen Sun. Hybrid rl: Using both ofline and online data can make rl eficient. In International Conference on Learning Representations, 2023.

[36] Zachary Sunberg and Mykel Kochenderfer. Online algorithms for pomdps with continuous state, action, and observation spaces. In Proceedings of the International Conference on Automated Planning and Scheduling, volume 28, pages 259–263, 2018.

[37] Kevin Tan, Giles Hooker, and Edward L. Ionides. Accelerated inference for partially observed markov processes using automatic diferentiation. arXiv preprint arXiv:2407.03085, 2024.

[38] Nick Whiteley. Stability properties of some particle filters. The Annals of Applied Probability, pages 2500–2537, 2013.

[39] Yue Frank Wu, Weitong Zhang, Pan Xu, and Quanquan Gu. A finite-time analysis of two timescale actor-critic methods. Advances in Neural Information Processing Systems, 33:17617–17628, 2020.

[40] Tengyu Xu, Zhe Wang, and Yingbin Liang. Improving sample complexity bounds for (natural) actor-critic algorithms. Advances in Neural Information Processing Systems, 33:4358–4369, 2020.

## A Further Related Work

Our work is related to research on approximate planning in POMDPs, particle-based belief representations, and inference in latent-state models.

Planning in POMDPs. POMDP planning can be formulated as dynamic programming over the belief state [19], but exact planning is dificult because the belief space is continuous and must be updated online. Practical methods therefore rely on approximations. Sampling-based planners such as POMCP combine Monte-Carlo tree search with particle-based belief updates and scale to large problems without explicit belief-stateenumeration [26, 34], while progressive-widening methods extend this approach to continuous spaces [36]. Another common approximation is QMDP, which replaces belief-space continuation values with fully observed MDP values averaged under the current belief [4, 27]. Our method is connected to this line of work through combining finite-depth MCTS with a QMDP-style bootstrap, but difers in that we study this approximation in an online learning setting and explicitly quantify the resulting structural error. POMCP, POWSS, and sparse PFT are online planners in that they replan from the current belief at every step. However, they are analyzed with the model given to the planner, so their analyses are not online learning results. The same holds for our planner when it is analyzed at a frozen θ. Their guarantees concern a single planning call at an exactly known $\theta ,$ , with no estimation error and no horizon $T _ { i }$ , so they do not directly yield a regret bound when the planner is called with a re-estimated ${ \bar { \theta } _ { k } }$ . Because the planner enters our decomposition only through $\epsilon _ { \mathrm { m c } } ( t , d )$ , one could substitute such a planner by re-deriving that term (and term (B1), which a particle-belief MDP absorbs) under ${ \bar { \theta } _ { k } }$ . Planning on the particle-belief MDP also requires $O ( J )$ generative calls per node, compared with $O ( 1 )$ for a state-level simulation.

Particle-based inference under partial observability. A related line of work uses particle filtering to approximate the belief state. Classical particle filtering theory provides convergence and stability guarantees for approximate Bayes filtering [8, 38]. [25] studied the particle-belief MDP directly and established optimality guarantees relative to the true belief-state MDP. More recent work has studied planning with particle methods in structured partially observed systems [11] as well as joint state and parameter inference in state-space models [5]. Our paper connects these directions in a diferent way: rather than planning over propagated particle beliefs, we use particles to represent the current belief, perform state-level lookahead from those particles, and combine this with online model learning. This leads to an end-to-end regret analysis that separates estimation error, particle approximation error, Monte-Carlo planning error, and QMDP-style approximation bias.

Model-free approaches. Model-free alternatives instead avoid explicit filtering by giving the learner a short horizon lookback or memory [20, 31]. The latter are often efective in practice, especially with deep recurrent policies, but their hidden states are dificult to interpret as beliefs, and remain challenging to analyze. Existing sample complexity guarantees for the former often contain caveats, such as only quasipolynomial rather than polynomial sample complexity [14, 15], or non-standard assumptions [2]. We will focus on the model-based approach.

## B MCTQ Planning Subroutine

We describe the full MCTQ (Monte Carlo Tree search with QMDP leaves) planning subroutine referenced in Algorithm 1. MCTQ estimates the depth-d belief-tree action-value $Q _ { \theta , d } ( b ^ { J } , a )$ defined in equation 9 by Monte Carlo tree search on the particle belief. It is inspired by the POMCP framework [34], with three departures: internal actions are drawn uniformly at random rather than by UCB1, every simulation descends to the full depth $d ,$ and values are computed by a bottom-up dynamic programming (DP) backup after all simulations complete.

Overview. Given the current particle belief $\begin{array} { r } { b ^ { J } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \delta _ { s _ { j } } } \end{array}$ , model $\theta ,$ search depth $d ,$ simulation budget N per root action, and a precomputed fully observable Q-function $Q _ { \theta } ^ { \star , S } ~ \mathrm { ( e . g . }$ ., from value iteration), MCTQ estimates $Q _ { \theta , d } ( b ^ { J } , a )$ for each root action a ∈ A separately (round-robin), so that every root action receives exactly N simulations. For each root action $^ { a , }$ each simulation samples a particle $s _ { j }$ from $b ^ { J }$ , takes the fixed root action a to obtain a child state $s ^ { \prime } \sim P _ { \theta } ( \cdot \mid s _ { j } , a )$ and observation $o \sim Z _ { \theta } ( \cdot \mid s ^ { \prime } )$ , and then descends the search tree $T _ { a }$ from the child node $( a , o )$ down to depth $d ,$ selecting actions uniformly at random at each internal node. Nodes are created the first time a simulation passes through them, and every node on the path records the simulated state. After all

N simulations complete, a bottom-up DP backup computes value estimates: at each depth-d leaf, the QMDP value of its accumulated particles; at each internal node, $\hat { V } ( v ) = \operatorname* { m a x } _ { a ^ { \prime } } \hat { Q } ( v , a ^ { \prime } )$ , where $\hat { Q } ( v , a ^ { \prime } )$ is the empirical mean of $r + \gamma \hat { V } ( \mathrm { c h i l d } )$ over the simulations that selected $a ^ { \prime }$ at v. The root estimate ${ \widehat { Q } } ( b ^ { J } , a )$ is the same empirical mean at the root.

Like POMCP [34], MCTQ indexes its search trees by action-observation histories $h = ( a _ { 1 } , o _ { 1 } , \ldots , a _ { k } , o _ { k } )$ rather than by explicit beliefs, and represents the belief at each node implicitly by the collection of states simulated through that node. The round-robin over root actions is needed because the regret analysis (Lemma 4) requires uniformly good estimates ${ \widehat { Q } } ( b ^ { J } , a )$ for all $a \in { \mathcal { A } }$ . The analysis (Proposition 3) uses two properties of the internal action-selection rule π, i.e., the rule that picks actions at nodes below the root within one planning call:

(S1) Non-adaptivity. π may depend on the node’s history, on $\theta ,$ on the root particle set $b ^ { J }$ , and on fresh randomness, but not on any statistic accumulated by the current tree.

(S2) Full support. $\pi ( a ^ { \prime } \mid \cdot ) \geq \pi _ { \operatorname* { m i n } } > 0$ for every $a ^ { \prime } \in { \mathcal { A } }$

Uniform selection satisfies both with $\pi _ { \mathrm { m i n } } = 1 / \vert \mathcal { A } \vert$ . So does a learned proposal that is frozen during the planning call and mixed with the uniform distribution. UCB1 violates (S1), because its choice at a node depends on the visit counts and value estimates accumulated by the current tree. We therefore use uniform selection for the analysis and do not claim it is the best rule in practice. Together with the absence of an early return (every simulation descends to depth $d )$ , (S1) makes the path of each simulation a function of its own random seed alone (Lemma 5). The simulation seeds are i.i.d. given $b ^ { J }$ , but the simulation returns are not, because the leaf evaluator pools particles across simulations and the $\mathrm { D P }$ backup couples them. The analysis therefore treats ${ \widehat { Q } } ( b ^ { J } , a )$ as a bounded-diference function of the i.i.d. seeds rather than as an average of independent returns.

Tree structure and belief particles. Each search tree $T _ { a }$ (one per root action a) has the particle belief $b ^ { J }$ at its root (depth 0), the root action fixed to $^ { a , }$ internal nodes at depths $1 , \ldots , d - 1$ , and leaves at depth $d .$ Each node v is identified with the action-observation history that leads to it from the root, so a given history always refers to the same node. A child of node v under action $a ^ { \prime }$ and observation o is denoted $\mathrm { C } _ { \mathrm { H I L D } } ( v , a ^ { \prime } , o )$ ; it is created on demand when first visited.

Each node v maintains a particle set $B ( v )$ : the collection of all states that have been simulated through v across all simulations. The empirical distribution $\begin{array} { r } { \hat { b } _ { v } : = \frac { 1 } { | B ( v ) | } \sum _ { s \in B ( v ) } \delta _ { s } } \end{array}$ is the particle approximation of the belief at v, and $N ( v ) : = | B ( v ) |$ is the number of simulations that have visited v. In addition, each internal node stores:

$N ( v , a ^ { \prime } )$ : the number of simulations that selected action $a ^ { \prime }$ at $v ;$

$\bar { r } ( v , a ^ { \prime } )$ : the running mean of the immediate rewards $r ( s , a ^ { \prime } )$ over those simulations.

These statistics are accumulated during the simulation phase and used in the post-simulation DP backup.

QMDP leaf evaluation. At depth-d leaf nodes (remaining depth $\ell = 0 )$ , the QMDP value of the accumulated belief is

$$
\bar { V } _ { \theta } ^ { \star , S } ( B ( v ) ) : = \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } \frac { 1 } { | B ( v ) | } \sum _ { s \in B ( v ) } Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } ) .\tag{15}
$$

This takes the max outside the average over states, choosing a single action for all particles, which coincides with the QMDP approximation $\bar { V } _ { \theta } ^ { \star , S }$ from equation $8 .$ This is distinct from the perstate evaluation $\frac { 1 } { | B ( v ) | } \sum _ { s }$ <sub>s</sub> max<sub>a</sub> $, Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } )$ , which would compute the linear surrogate $\widetilde { V } _ { \boldsymbol { \theta } } ^ { \star }$ instead.

Using equation 15 ensures the algorithm matches the theoretical target $\bar { V } _ { \theta , d } ^ { \star , S } = T _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S }$ analyzed in Section 4.

Simulation procedure. For each root action $^ { a , }$ each of the N simulations proceeds as follows. A particle $s _ { j }$ is sampled uniformly from $b ^ { J }$ , and the root transition is simulated: $s ^ { \prime } \sim P _ { \theta } ( \cdot \mid s _ { j } , a )$ $o \sim Z _ { \theta } ( \cdot \mid s ^ { \prime } )$ , with immediate reward $r ( s _ { j } , a )$ . Starting from the child node $\mathrm { C H I L D } ( \mathrm { r o o t } , a , o )$ with remaining depth $\ell = d - 1$ , the Simulate subroutine (Algorithm 3) descends the tree recursively:

1. Record. Add the current state s to $B ( v )$

2. QMDP leaf $( \ell = 0 )$ . Stop. The leaf is evaluated by equation 15 on its final particle set once all simulations are complete.

3. Expansion (new internal node). If v is not yet expanded, initialize $N ( v , a ^ { \prime } ) \gets 0$ and $\bar { r } ( v , a ^ { \prime } ) \gets 0$ for all $a ^ { \prime } \in { \mathcal { A } }$ , and continue. There is no early return, so a simulation that creates a node continues to descend.

4. Uniform selection. Select

$$
a ^ { \prime } \sim \operatorname { U n i f } ( A ) .\tag{16}
$$

5. Simulate and recurse. Draw $s ^ { \prime \prime } \sim P _ { \theta } ( \cdot \mid s , a ^ { \prime } )$ and $o ^ { \prime } \sim Z _ { \theta } ( \cdot \mid s ^ { \prime \prime } )$ , update $N ( v , a ^ { \prime } )$ and $\bar { r } ( v , a ^ { \prime } )$ with the reward $r ( s , a ^ { \prime } )$ , and recurse into $\mathrm { C } _ { \mathrm { H I L D } } ( v , a ^ { \prime } , o ^ { \prime } )$ with remaining depth $\ell - 1$

Every simulation therefore visits exactly one node at each depth $1 , \ldots , d .$ . After all N simulations for root action $^ { a , }$ the DP backup (Phase 2 of Algorithm 2) computes, from the leaves upward,

$$
\begin{array} { r l } & { \hat { V } ( v ) = \bar { V } _ { \boldsymbol { \theta } } ^ { \star , S } ( B ( v ) ) } \\ & { \hat { Q } ( v , a ^ { \prime } ) = \bar { r } ( v , a ^ { \prime } ) + \gamma \displaystyle \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } \frac { N ( w ) } { N ( v , a ^ { \prime } ) } \hat { V } ( w ) \quad \mathrm { ( i n t e r n a l ~ } v , N ( v , a ^ { \prime } ) > 0 ) , } \\ & { \hat { V } ( v ) = \displaystyle \operatorname* { m a x } _ { a ^ { \prime } : N ( v , a ^ { \prime } ) > 0 } \hat { Q } ( v , a ^ { \prime } ) \qquad \quad \mathrm { ( i n t e r n a l ~ } v ) , } \end{array}\tag{17}
$$

where $\mathrm { C h } ( v , a ^ { \prime } )$ is the set of children of v under action $a ^ { \prime }$ . Equivalently, $\hat { Q } ( v , a ^ { \prime } )$ is the average of $r ( s , a ^ { \prime } ) + \gamma \hat { V } ( \mathrm { c h i l d } )$ over the simulations that selected $a ^ { \prime }$ at v. The root estimate is

$$
\widehat { Q } ( b ^ { J } , a ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ r ( s _ { j _ { n } } , a ) + \gamma \hat { V } ( v _ { n , 1 } ) \bigr ] ,
$$

where s is the particle sampled by simulation n and $s _ { j _ { n } }$ $v _ { n , 1 }$ is the depth-1 node it enters. When $d = 0$ no tree is built and $\begin{array} { r } { \widehat { Q } ( b ^ { J } , a ) = \frac { 1 } { J } \sum _ { i = 1 } ^ { J } Q _ { \theta } ^ { \star , S } ( s _ { j } , a ) } \end{array}$

The full procedure is given in Algorithms 2 and 3.

Convergence. As $N  \infty$ , the following two properties hold. First, by Lemma 5 the events “simulation n reaches v and selects $a ^ { \prime \prime } { } ^ { * }$ are i.i.d. across n with positive probability for every reachable node v and every $a ^ { \prime } { \mathrm { . } }$ , so every such node and action is visited infinitely often. Second, the states entering a node v are i.i.d. draws from the (binned) conditional belief $b _ { v }$ at $v ,$ so $\hat { b } _ { v }  b _ { v }$ by the law of large numbers. At each depth-d leaf, $\bar { V } _ { \theta } ^ { \star , \dot { S } } ( B ( v ) ) \stackrel { \prime } { { }  } \bar { V } _ { \theta } ^ { \star , S } ( b _ { v } )$ , and the DP backup then gives, level by level, $\hat { Q } ( v , a ^ { \prime } )  Q _ { \theta , \ell } ^ { \phi } ( b _ { v } , a ^ { \prime } )$ and $\begin{array} { r } { \hat { V } ( v )  \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta , \ell } ^ { \phi } ( b _ { v } , a ^ { \prime } ) = \bar { V } _ { \theta , \ell } ^ { \star , S } ( b _ { v } ) } \end{array}$ , where ℓ is the remaining depth at v. Since the root action is fixed and states are sampled from $b ^ { J }$ , the backed-up estimate $\widehat { Q } ( b ^ { J } , a ) \to Q _ { \theta , d } ^ { \phi } ( b ^ { J } , a )$ for each $a \in { \mathcal { A } }$ . Proposition 3 quantifies the rate.

Algorithm 2 Mctq: Monte Carlo tree QMDP planning (uniform exploration, full-depth simulations,   
DP backup)   
Require: model $\theta ;$ particle belief $\begin{array} { r } { b ^ { J } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \delta _ { s _ { j } } ; } \end{array}$ depth $d ;$ simulations per action $N ;$ MDP   
Q-function $Q _ { \theta } ^ { \star , S }$   
1: if $d = 0$ then ▷ No tree: exact QMDP   
2: return $\textstyle \left\{ { \frac { 1 } { J } } \sum _ { j = 1 } ^ { J } Q _ { \theta } ^ { \star , S } ( s _ { j } , a ) \right\} _ { a \in { \mathcal { A } } }$   
3: end if   
4: for each action $a \in { \mathcal { A } }$ do ▷ Round-robin: N simulations per root action   
5: Initialize empty search tree $T _ { a }$   
6: for $n = 1 , \ldots , N$ do ▷ Phase 1: tree construction   
7: Sample $j _ { n } \sim \operatorname { U n i f } ( [ J ] )$   
8: Sample $s ^ { \prime } \sim P _ { \boldsymbol \theta } ( \cdot \mid s _ { j _ { n } } , a ) , o \sim Z _ { \boldsymbol \theta } ( \cdot \mid s ^ { \prime } )$   
9: $v _ { n , 1 }  \mathrm { C H I L D } ( \mathrm { r o o t } ( T _ { a } ) , a , o )$ ▷ Child of root for $( a , o )$   
10: $\mathrm { S I M U L A T E } ( s ^ { \prime } , d - 1 , \theta , v _ { n , 1 } )$ ▷ Alg. 3   
11: end for   
12: ▷ Phase 2: DP backup equation 17   
13: $\begin{array} { r } { \hat { V } ( v )  \operatorname* { m a x } _ { a ^ { \prime } \in \mathcal { A } } \frac { 1 } { | B ( v ) | } \sum _ { s \in B ( v ) } Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } ) } \end{array}$ at each leaf v of $T _ { a }$ ▷ Pooled QMDP leaf   
14: for each internal node v of $T _ { a } ,$ from depth $d - 1$ up to depth 1 do   
15: $\begin{array} { r } { \hat { Q } ( v , a ^ { \prime } ) \gets \bar { r } ( v , a ^ { \prime } ) + \gamma \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } \frac { \mathbf { \dot { N } } ( w ) } { N ( v , a ^ { \prime } ) } \hat { V } ( w ) } \end{array}$ for each $a ^ { \prime }$ with $N ( v , a ^ { \prime } ) > 0$   
16: $\hat { V } ( v ) \gets \operatorname* { m a x } _ { a ^ { \prime } : N ( v , a ^ { \prime } ) > 0 } \hat { Q } ( v , a ^ { \prime } )$   
17: end for   
18: $\begin{array} { r } { \widehat { Q } ( b ^ { J } , a ) \gets \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ r ( s _ { j _ { n } } , a ) + \gamma \hat { V } ( v _ { n , 1 } ) \bigr ] } \end{array}$   
19: end for   
20: return $\{ \widehat { Q } ( b ^ { J } , a ) \} _ { a \in \mathcal { A } }$

When $d = 0 ,$ no tree search is performed and the planner returns max $\begin{array} { r } { \mathbf { \Sigma } _ { \cdot a } \frac { 1 } { J } \sum _ { j } Q _ { \theta } ^ { \star , S } ( s _ { j } , a ) } \end{array}$ , which is the QMDP value $\bar { V } _ { \theta } ^ { \star , S } ( b ^ { J } )$ . When $d \to \infty$ , the tree-search value converges to $Q _ { \theta } ^ { \star , B } ( b ^ { J } , a )$

Computational cost. For each root action, each of the N simulations traverses a path of exactly d nodes in the tree, requiring $O ( d )$ generative model calls and $O ( 1 )$ work for uniform action sampling at each internal node. The post-simulation DP backup adds $O ( | A | )$ work per node but is performed once, not per simulation. The QMDP leaf evaluation equation 15 can be maintained incrementally by storing running sums $\begin{array} { r } { \sum _ { s \in B ( v ) } Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } ) } \end{array}$ for each action $a ^ { \prime }$ , reducing the per-simulation leaf cost to $O ( | A | )$ . The total cost per planning step is $| { \mathcal { A } } | N d$ generative-model calls across all root actions. The additional |A| factor in the backup involves only arithmetic operations and no model calls.

Relation to the error analysis. The Monte Carlo error $\begin{array} { r } { \epsilon _ { \mathrm { m c } } ( t , d ) = \operatorname* { s u p } _ { a } | \widehat { Q } _ { t } ( b _ { t } ^ { J } , a ) - Q _ { \theta , d } ( b _ { t } ^ { J } , a ) | } \end{array}$ arises from the finite simulation budget N per root action. The simulation seeds are i.i.d. given $b _ { t } ^ { J }$ and under (S1) with no early return each simulation’s path is a function of its own seed (Lemma 5). The estimate $\widehat { Q } _ { t } ( b _ { t } ^ { J } , a )$ is then a bounded-diference function of the seeds, so the Efron–Stein and McDiarmid inequalities control its fluctuation at rate $O ( \Lambda _ { d } / \sqrt { N } )$ , while the DP-backup bias vanishes at rate $O ( \Phi _ { d } / \sqrt { N } )$ (Proposition 3). The simulation returns themselves are not independent, and the argument does not require them to be. The proof is written for uniform selection. For a rule satisfying (S1)–(S2), the same argument applies with |A| replaced by $1 / \pi _ { \operatorname* { m i n } }$ wherever it bounds the fraction of a node’s visits that select a given action.

Algorithm 3 Simulate: Recursive uniform-exploration descent to full depth   
Require: state s; remaining depth $\ell ;$ model $\theta ;$ tree node v   
1: $B ( v )  B ( v ) \cup \{ s \}$ ▷ Record state; $N ( v ) = | B ( v ) |$   
2: if ℓ = 0 then ▷ Leaf: evaluated in Phase 2 of Alg. 2   
3: return   
4: end if   
5: if v is not yet expanded then ▷ Expansion: initialize and continue (no early return)   
6: Expand v: $N ( v , a ^ { \prime } ) \gets 0 , \bar { r } ( v , a ^ { \prime } ) \gets 0$ for all $a ^ { \prime } \in { \mathcal { A } }$   
7: end if   
8: Sample $a ^ { \prime } \sim \operatorname { U n i f } ( \mathcal { A } )$ ▷ Uniform, reads no tree statistic   
9: Sample $s ^ { \prime } \sim P _ { \boldsymbol \theta } ( \cdot \mid s , a ^ { \prime } ) , o \sim Z _ { \boldsymbol \theta } ( \cdot \mid s ^ { \prime } )$   
10: $\begin{array} { r } { N ( v , a ^ { \prime } ) \gets N ( v , a ^ { \prime } ) + 1 ; \bar { r } ( v , a ^ { \prime } ) \gets \bar { r } ( v , a ^ { \prime } ) + \frac { r ( s , a ^ { \prime } ) - \bar { r } ( v , a ^ { \prime } ) } { N ( v , a ^ { \prime } ) } } \end{array}$ ▷ Incremental mean   
11: $\mathrm { S I M U L A T E } \big ( s ^ { \prime } , \ell - 1 , \theta , \mathrm { C H I L D } \big ( v , a ^ { \prime } , o \big ) \big )$

## C Figures and Additional Details for Experiments

```latex
Algorithm 4 Practical particle filtering with state-level Monte-Carlo planning
Require: number of state particles $J ;$ rollout length $H _ { \mathrm { r o l l } } ;$ rollouts per action $N ;$ horizon $\overline { { T ; } }$ initial
parameter ${ \bar { \theta } } _ { 1 } ;$ model update oracle ModelUpdate(·).
1: Initialize: model dataset ${ \mathcal { D } } _ { m } ,$ policy dataset ${ \mathcal { D } } _ { p } .$
2: Sample $s _ { 1 } ^ { ( 1 ) } , \ldots , s _ { J } ^ { ( 1 ) } \sim b _ { 0 }$ and set $\begin{array} { r } { b _ { 1 } ^ { J } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \delta _ { s _ { j } ^ { ( 1 ) } } } \end{array}$
3: for $t = 1 , 2 , \dots , T$ do
4: if model update then
5: $\mathcal { D } _ { m } = \mathcal { D } _ { m } \cup \mathcal { D } _ { p } .$
6: Update ${ \bar { \theta } } _ { t }$ via updating the model on $\mathcal { D } _ { m }$ with gradient descent.
7: $\begin{array} { r } { \mathcal { D } _ { p } = \emptyset . } \end{array}$
8: else
9: $\bar { \theta } _ { t }  \bar { \theta } _ { t - 1 } .$
10: end if
11: for each particle $j = 1 , \dots , J$ and each action $a \in { \mathcal { A } }$ do
12: $\widehat { Q } _ { t } ( s _ { j } ^ { ( t ) } , a ) \gets$ StateMCTS $\big ( s _ { j } ^ { ( t ) } , a , \bar { \theta } _ { t } , H _ { \mathrm { r o l l } } , N , \widehat { Q } _ { \bar { \theta } _ { t } } ^ { \mathrm { M D P } } , \widehat { \pi } _ { \bar { \theta } _ { t } } ^ { \mathrm { M D P } } \big )$ , where actions are sampled
by ${ \widehat \pi } _ { \bar { \theta } _ { t } } ^ { \mathrm { M D P } }$ and not the greedy policy w.r.t. $\widehat { Q } _ { t }$ . Collect $\widehat { a } _ { j } ^ { ( t ) }$ , the best action sampled for particle j.
13: end for
14: $\begin{array} { r } { \widehat { Q } _ { t } ( b _ { t } ^ { J } , a ) \gets \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \widehat { Q } _ { t } ( s _ { j } ^ { ( t ) } , a ) } \end{array}$ for all $a \in A .$
15: Play $a _ { t } \in \arg \operatorname* { m a x } _ { j \in [ J ] } \widehat { Q } _ { t } ( b _ { t } ^ { J } , \widehat { a } _ { j } ^ { ( t ) } )$ . Alternatively, sample $\tilde { s } ^ { ( t ) } \sim b _ { t } ^ { J }$ and play $a _ { t } \sim \widehat \pi _ { \bar { \theta } _ { t } } ^ { \mathrm { M D P } } ( \cdot | \tilde { s } ^ { ( t ) } )$
16: Observe $\left( o _ { t + 1 } , r _ { t } \right)$ and append to $\mathcal { D } _ { p }$
17: for $j = 1 , \dots , J$ do
18: Sample prediction $\hat { s } _ { j } ^ { ( t + 1 ) } \sim P _ { \bar { \theta } _ { t } } ( \cdot \mid s _ { j } ^ { ( t ) } , a _ { t } )$
19: Set weight $w _ { j } ^ { ( t + 1 ) } \propto \overline { { Z _ { \bar { \theta } _ { t } } ( o _ { t + 1 } \mid \hat { s } _ { j } ^ { ( t + 1 ) } ) } }$
20: end for
21: Resample $s _ { 1 } ^ { ( t + 1 ) } , \dots , s _ { J } ^ { ( t + 1 ) }$ ∼ Categorical $( w _ { 1 } ^ { ( t + 1 ) } , \dots , w _ { J } ^ { ( t + 1 ) } )$
22: Set $\begin{array} { r } { b _ { t + 1 } ^ { J } = \frac { 1 } { J } \sum _ { j = 1 } ^ { J } \delta _ { s _ { j } ^ { ( t + 1 ) } } } \end{array}$
23: Update MDP value estimate $\widehat { Q } _ { \widehat { \theta } _ { t } } ^ { \mathrm { M D P } } ( s , a )$ and policy ${ \widehat \pi } _ { \widehat { \theta } _ { t } } ^ { \mathrm { M D P } }$ for $( S , \mathcal { A } , P _ { \bar { \theta } _ { t } } , r , \gamma )$ on ${ \mathcal { D } } _ { p } .$
24: end for
```

## C.1 Double Integrator LQR

We use a partially observed double-integrator LQR system. The latent state is

$$
x _ { t } = \left[ { p _ { t } { \right] } _ { } , }
$$

where $p _ { t }$ is position and $v _ { t }$ is velocity. The scalar control $u _ { t } \in [ - 3 , 3 ]$ is acceleration. With time step $\Delta t = 0 . 0 5$ , the dynamics are

$$
x _ { t + 1 } = A x _ { t } + B u _ { t } + w _ { t } , \qquad A = { \left[ \begin{array} { l l } { 1 } & { \Delta t } \\ { 0 } & { 1 } \end{array} \right] } , \qquad B = { \left[ \begin{array} { l l } { { \frac { 1 } { 2 } } \Delta t ^ { 2 } } \\ { \Delta t } \end{array} \right] } ,
$$

with process noise

$$
w _ { t } \sim \mathcal { N } ( 0 , \sigma _ { w } ^ { 2 } I _ { 2 } ) , \qquad \sigma _ { w } = 0 . 0 1 .
$$

The agent observes only the position coordinate:

$$
o _ { t } = C x _ { t } + \epsilon _ { t } , \qquad C = \left[ 1 \quad 0 \right] , \qquad \epsilon _ { t } \sim { \mathcal N } ( 0 , \sigma _ { o } ^ { 2 } ) , \qquad \sigma _ { o } = 0 . 0 5 .
$$

Thus the velocity $v _ { t }$ is latent. It is not recoverable from the current observation alone because the observation map is not injective:

$$
\operatorname { r a n k } ( C ) = 1 < 2 .
$$

Equivalently, for any fixed position p and any two distinct velocities $v \ne v ^ { \prime }$

$$
C \left[ _ { v } ^ { p } \right] = p = C \left[ _ { v ^ { \prime } } ^ { p } \right] .
$$

Therefore, even in the noiseless observation case, a single current observation cannot distinguish states that difer only in velocity. With observation noise, the ambiguity is strictly larger. Recovering information about $v _ { t }$ requires temporal history, e.g. past observations and actions, rather than $o _ { t }$ alone.

The reward is the negative quadratic LQR cost,

$$
r _ { t } = - \left( x _ { t } ^ { \top } Q x _ { t } + u _ { t } ^ { \top } R u _ { t } \right) , \qquad Q = I _ { 2 } , \qquad R = 0 . 1 I _ { 1 } .
$$

The discounted objective is

$$
J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T - 1 } \gamma ^ { t } r _ { t } \right] , \qquad \gamma = 0 . 9 , \qquad T = 2 0 0 .
$$

In our experiments, the initial state is sampled as

$$
x _ { 0 } \sim { \mathcal { N } } ( 0 , I _ { 2 } ) ,
$$

episodes terminate after $T = 2 0 0$ steps or when

$$
\| x _ { t } \| _ { 2 } > 2 0 .
$$

We use 32 particles for belief tracking, rollout length 1, planner discount 0.9, and a maximum of 16 candidate actions.

![](images/351cb52fe0b024a6de61492c79ee59dbdb111d5ee652a1038f19a6551eb3260c.jpg)  
Figure 4: Snapshot of belief state at select training checkpoints. State recovery improves over time.

![](images/d830ad910cd3f94a44ad4893fbc78f8f3331cb38d5e5dfc715cd7b4f5925af7d.jpg)

![](images/b51cd4ef3dec6ebf0be11f2b537c3216e8a231036283601338476c4ccdfde0c6.jpg)

![](images/fe2832875ddf1f9ff24ea566294844959f82b2f2718309721bb822cea3a7ecdb.jpg)

![](images/ed119ac2ac2d14985a2989779d08333580b24761deddf4df019f195358a82b20.jpg)

![](images/72bc1f1370e1398dbe3a075cb89cf4f33a625362fc03ca23d4151aa46bfde112.jpg)

![](images/8651ce0cfdd7f69589b4f2bf7326c5aae70142d9e587c148932ada8d621775c8.jpg)

![](images/23337c130c42437c29f0f88d041238ec8a0d37afaaa719f2848ca0d926bb662d.jpg)

![](images/fad27cb0cef4d11fddcf0f367dd83ca32df4423e48b8e164972587e29971f6aa.jpg)

![](images/963aa135fc0f0e0fd52a4f1885159863309431442fcf29c31dd27a367baa2a59.jpg)

![](images/115c21a2cecde0aecb5393b91c681933a32466241ec38462cb35bfa6f637f45a.jpg)  
Figure 5: Posterior panel over five seeds for the double integrator LQR, over the last 1000 steps. The 10th quantile and 90th quantile are visualized. In this setting, only the magnitude, but not the direction, latent state is identifiable from the current observations alone. With no access to memory, the particle belief state still exhibits the ability to tracking the belief state, as seen in seeds 2 and 4, but fails when the state rapidly diverges. This suggests that the particle belief can be used in tandem with memory for latent state recovery..

![](images/50b41c24e619cab7ad2ef08f10bb05d0cde41ec2b124da641a05577a3f51b127.jpg)

![](images/6e906ff529914fe37f30a6261abe63fa2cc953ddb959196856745cb36114fbb5.jpg)

![](images/9c30e2d40ef5f824d5704536b90e4751836c767b10a4d3957017f191933f3f63.jpg)

![](images/9fc0864ddc02a5fc557c9b06e035e6b69040747f17bec5ddda99f23ff5fbbabe.jpg)

![](images/620edef647960a12d022c1c6e7994858eff576ec99895869e3bba1a78790823e.jpg)

![](images/6bdd6bca6e1aca62e4a8677e559f618e87dd4fbc7190ee2f2d4277b54a428802.jpg)

![](images/0408d2749ed6f2b2afe0589524c45e7dac0a021fcd32d551044f630b3e4dfa24.jpg)

![](images/5b29f1d17c4a744654ad5bdd80ca82c8186a0c17f91a7e22f49e50c1d9d6cd7c.jpg)

![](images/586d29bc00b31ac89a556542e2039eb99dd36beee103aa1c242fffa8ff7e7f28.jpg)

![](images/a8ea9079874970cb9358681e933b2ba2c3cf540e1f2d61bb96c768350c4e2402.jpg)  
Figure 6: Belief mean panel over five seeds for LQR. The MLP backbone diverges, while the GRU backbone converges.

## C.2 Heartsteps

HeartSteps as an AOMDP. We follow the formulation of [12] in formulating HeartSteps as an actively observable Markov decision process (AOMDP). At each day t, the agent first chooses a measurement action $I _ { t , 1 } \in \{ 0 , 1 \}$ , where $I _ { t , 1 } = 1$ corresponds to sending a survey that reveals the latent reward/commitment state. The agent then chooses a control action $A _ { t , 2 } \in \{ 0 , 1 \}$ , where $A _ { t , 2 } = 1$ corresponds to sending a walking-suggestion intervention.

Let $R _ { t }$ denote the user’s latent commitment to being active on day $t , E _ { t }$ denote engagement or burden, $C _ { t , 2 }$ denote context before the intervention decision, $M _ { t , 2 }$ denote the proximal physical activity outcome, and $O _ { t }$ denote a passive proxy emission of the latent reward. Following the AOMDP indexing, the latent state is

$$
U _ { t + 1 } : = R _ { t } ,
$$

and the observed state after the control action is

$$
Z _ { t + 1 } : = ( M _ { t , 2 } , E _ { t } ) .
$$

The reward is the latent reward itself,

$$
r ( Z _ { t + 1 } , U _ { t + 1 } ) = R _ { t } .
$$

The measurement-stage and control-stage states are

$$
S _ { t , 1 } ^ { I } = ( E _ { t - 1 } , R _ { t - 1 } ) , \qquad S _ { t , 2 } ^ { A } = ( E _ { t - 1 } , R _ { t - 1 } , C _ { t , 2 } , I _ { t , 1 } ) .
$$

Before choosing $I _ { t , 1 }$ , the agent observes $E _ { t - 1 }$ and the passive emission $O _ { t - 1 }$ , but not $R _ { t - 1 }$ . If $I _ { t , 1 } = 1$ the agent observes $R _ { t - 1 }$ before choosing $A _ { t , 2 } ;$ if $I _ { t , 1 } = 0 , R _ { t - 1 }$ remains latent.

The HeartSteps transition model is

$$
\begin{array} { c } { C _ { t , 2 } = { \theta } _ { 0 } ^ { C } + \varepsilon _ { t , 2 } ^ { C } , } \\ { M _ { t , 2 } = { \theta } _ { 0 } ^ { M } + { \theta } _ { 1 } ^ { M } E _ { t - 1 } + { \theta } _ { 2 } ^ { M } R _ { t - 1 } + { \theta } _ { 3 } ^ { M } C _ { t , 2 } + A _ { t , 2 } \left( { \theta } _ { 4 } ^ { M } + { \theta } _ { 5 } ^ { M } E _ { t - 1 } + { \theta } _ { 6 } ^ { M } R _ { t - 1 } + { \theta } _ { 7 } ^ { M } C _ { t , 2 } \right) + \varepsilon _ { t , 2 } ^ { M } , } \\ { E _ { t } = { \theta } _ { 0 } ^ { E } + { \theta } _ { 1 } ^ { E } E _ { t - 1 } + { \theta } _ { A } ^ { E } A _ { t , 2 } + { \theta } _ { A } ^ { E } A _ { t , 2 } E _ { t - 1 } + { \theta } _ { T } ^ { E } I _ { t , 1 } + { \theta } _ { R } ^ { E } I _ { t , 1 } E _ { t - 1 } + \varepsilon _ { t } ^ { E } , } \\ { R _ { t } = { \theta } _ { 0 } ^ { R } + { \theta } _ { 1 } ^ { R } M _ { t , 2 } + { \theta } _ { 2 } ^ { R } E _ { t } + { \theta } _ { 3 } ^ { R } R _ { t - 1 } + \varepsilon _ { t } ^ { R } , } \end{array}
$$

and the passive emission model is

$$
O _ { t } = \theta _ { 0 } ^ { O } + \theta _ { 1 } ^ { O } R _ { t } + \varepsilon _ { t } ^ { O } .
$$

The noise terms are mean-zero, with variances $\sigma _ { C } ^ { 2 } , \sigma _ { M } ^ { 2 } , \sigma _ { E } ^ { 2 } , \sigma _ { R } ^ { 2 } , \sigma _ { O } ^ { 2 }$

The latent reward is not recoverable from the current passive observation alone. Indeed, when $I _ { t , 1 } = 0$ , the posterior over $R _ { t - 1 }$ satisfies

$$
p ( R _ { t - 1 } = r \mid E _ { t - 1 } , O _ { t - 1 } , H _ { t , 1 } ) \propto p ( O _ { t - 1 } \mid R _ { t - 1 } = r ) p ( R _ { t - 1 } = r \mid E _ { t - 1 } , H _ { t , 1 } ) ,
$$

where

$$
O _ { t - 1 } = \theta _ { 0 } ^ { O } + \theta _ { 1 } ^ { O } R _ { t - 1 } + \varepsilon _ { t - 1 } ^ { O } .
$$

Thus, unless the emission noise is degenerate and the prior is already a point mass, the posterior is not a Dirac measure. In particular, for $\sigma _ { O } ^ { 2 } > 0$ , multiple values of $R _ { t - 1 }$ have positive likelihood for the same observed $O _ { t - 1 }$ . Therefore the latent reward cannot be identified from the current passive observation alone. By contrast, when $I _ { t , 1 } = 1$ , the survey reveals $R _ { t - 1 }$ and collapses the posterior:

$$
\begin{array} { r } { p ( R _ { t - 1 } \mid H _ { t , 2 } ) = \delta _ { R _ { t - 1 } } . } \end{array}
$$

The AOMDP objective is

$$
\pi ^ { \star } \in \arg \operatorname* { m a x } _ { \pi } \mathbb { E } _ { \pi } \left[ \sum _ { t = 1 } ^ { H } \gamma ^ { t - 1 } R _ { t } \right] .
$$

Implementation settings. In our HeartSteps experiments, the encoded simulator state is

$$
x _ { t } = ( R _ { t } , E _ { t } , t / H , C _ { t } , M _ { t } , O _ { t } ) \in \mathbb { R } ^ { 6 } .
$$

The observed coordinates are

$$
( E _ { t } , t / H , C _ { t } , M _ { t } , O _ { t } ) ,
$$

and the only latent coordinate is $R _ { t }$ . We use horizon

$$
H = 2 7 0 , \qquad \gamma = 0 . 9 ,
$$

binary control actions $A _ { t , 2 } \in \{ 0 , 1 \}$ , and binary measurement actions $I _ { t , 1 } \in \{ 0 , 1 \}$ in the AOMDP formulation. The benchmark contains 42 user-specific environments.

For PF-MCTQ, our default settings use

$$
J = 1 6
$$

particles, rollout length 1, planner discount 0.9, and at most 16 candidate actions. The model and policy are trained online over 27 epochs with 10 environment steps per epoch, for a total of 270 environment steps.

Benchmarks. RLSVI-SMC is the never-measure benchmark in our experiments. It fixes the measurement action to zero,

$$
I _ { t , 1 } = 0 \qquad \mathrm { f o r ~ a l l } ~ t ,
$$

so the latent reward is not directly observed after initialization. Instead, the algorithm maintains an SMC approximation to the posterior belief over the latent reward,

$$
\widehat { b } _ { t } ^ { R } = \sum _ { j = 1 } ^ { J } w _ { t } ^ { ( j ) } \delta _ { \widehat { R } _ { t } ^ { ( j ) } } ,
$$

and chooses the control action using randomized least-squares value iteration with a linear valuefunction approximation,

$$
\begin{array} { r } { Q _ { A } ( b _ { t , 2 } , a ) \approx \phi _ { A } ( b _ { t , 2 } , a ) ^ { \top } \beta _ { A } . } \end{array}
$$

Thus RLSVI-SMC is a strong and intentionally tailored benchmark for this linear-function-approximation setting: the belief update uses the specified linear-Gaussian working model, and the control rule uses hand-specified linear features of the belief state, context, action, and action interactions.

PF-MCTQ is also evaluated under the never-measure information pattern,

$$
I _ { t , 1 } = 0 ,
$$

so it has the same lack of direct latent-reward measurements as RLSVI-SMC. However, PF-MCTQ is less specialized to the linear benchmark structure: it must learn a dynamics model, actor, and critic, parameterized by neural networks, from the observed data. Consequently, PF-MCTQ incurs an additional estimation cost relative to RLSVI-SMC, which is already matched to the linear basis and Bayesian regression structure of the benchmark.

Dyna-ATMQ difers from both no-measure methods. It is an active-measurement baseline that maintains a tabular belief over a discretized state space, learns a Dirichlet transition model and

Q-table, and separately decides whether to measure by comparing the estimated value of measurement to a measurement cost. Its control action is selected from the belief-weighted Q-values,

$$
A _ { t , 2 } \in \arg \operatorname* { m a x } _ { a } \sum _ { s } b _ { t } ( s ) Q ( s , a ) ,
$$

while its measurement decision can reveal the latent reward and collapse the posterior uncertainty. In our experiments Dyna-ATMQ measures the environment around half the time on average, giving it an informational advantage over both RLSVI-SMC/never-measure and PF-MCTQ, which do not measure the environment.

![](images/a302c984e2899c47f8fef229777d48a5458d2e85a3d6ba0626ff7f75a168ebc6.jpg)

![](images/87eaf86ba0115376e20d944dc0f5a8a375cbae29d287ecbd7fe2d4a70c4b4237.jpg)

![](images/537c58c5ae602acb5ff3399c9a8aab2890f0c8e894e80999f78e29efec4b8589.jpg)

![](images/30ee14236ee3b0eda6352b8918e262f755f2c61131a33ca47d92d76102502110.jpg)

![](images/7ec5a2b32acadaee23457c2397183b89bbf01dd8036e23e88efc318780a155b9.jpg)

![](images/e80b025da1c1917c7a8914fbc5de98472695e314248791b2b4bda7799e0cff2d.jpg)

![](images/694cc484eb63fea23b02aa07072fb6834db0f48e72867d4cbd8990efa1d4eab4.jpg)

![](images/6eaea9af0a3bf070ad318e822910ba6cb06d86301a519f09044491d269590182.jpg)

![](images/b3ff39503778bd458fd33b6d82a28f4983328bbde47610d7f27d97ba214842f9.jpg)

![](images/3233601ac547b76308f2c0efeb9b3644aa1ff636c37350c1cd00f1718d7716d8.jpg)

![](images/bcd22db265fb708ad2e1e112c746644c1e3d611dd5e43dbd70298f20d6c1ca66.jpg)

![](images/f2283ca12942bb7218f1f67a51e7dc2176622b36e674491987849c9621209695.jpg)  
Figure 7: Cumulative reward above the baseline rollout across the twelve HeartSteps scenarios. Panels vary the positive intervention efect (rows: minimal, small, medium) and the delayed negative measurement efect or proxy informativeness (columns: zero negative efect, minimal negative efect, small negative efect, and uninformative passive proxy $O _ { t } )$ . Solid lines show the mean across runs and shaded bands show 2.5% and 97.5% quantiles. The never-measure RLSVI-SMC benchmark performs best in most scenarios because it is specifically matched to the linear-Gaussian HeartSteps structure and is given access to hand-specified linear features. PF-MCTQ also never measures, but must learn its model, actor, and critic from data using neural approximators, so it typically improves over baseline while trailing the tailored linear benchmark. Dyna-ATMQ often underperforms when measurement has delayed negative efects, especially when the positive intervention efect is weak, because it measures frequently and therefore gains latent-state information at the cost of worsening future engagement. Its relative performance improves when the positive efect is larger or when passive proxy information is unavailable and so direct measurements become more valuable.

![](images/52194793b13348422774a0029145c337ea6196bb040742f9bdd5c7e043030c6f.jpg)  
Figure 8: Belief over the latent reward at three training checkpoints in HeartSteps. Rows show PF-MCTQ, the never-measure RLSVI-SMC benchmark, and Dyna-ATMQ; columns show days 27, 135, and 243. Each panel displays posterior particles, a kernel density estimate, the posterior interquartile range, and the posterior mean. The red vertical line marks the true latent reward. We see that Dyna-ATMQ having access to measurements of the environment gives it an informational advantage over the other two algorithms. On the other hand, the correct linear specification that RLSVI-SMC enjoys allows it to create a posterior that is close to Gaussian. PF-MCTQ, which has to estimate a dynamics model, actor, and critic, parameterized by neural networks, from the observed data, consequently has a much more expressive and multimodal posterior.

![](images/629c2ee07686d004dddfea32db6b1b1d6592aa2b3023d07f5e135bc5b87a4034.jpg)  
Figure 9: Latent-reward belief trajectories for HeartSteps scenario (a), shown separately by random seed. Columns compare PF-MCTQ, the never-measure RLSVI-SMC benchmark, and Dyna-ATMQ; rows show five independent seeds. Each panel plots the posterior median, interquartile range, and 10–90% interval for the latent reward, with the dashed red curve showing the true latent reward. All three methods track the latent reward at the level needed for control, but they express uncertainty diferently. Never-measure produces comparatively narrow posteriors because it is tailored to the linear-Gaussian HeartSteps model and uses a conjugate SMC/RLSVI update. PF-MCTQ has wider and more variable posteriors because it must learn the transition model and value functions from data while never directly measuring the latent reward. Dyna-ATMQ also shows broad posterior support, but its belief remains centered near the true trajectory because it actively measures the latent reward on many days, partially collapsing uncertainty despite the coarser tabular approximation.

## C.3 Hyperparameters

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment</td><td>LQR double integrator</td></tr><tr><td>Observed indices</td><td>[0]</td></tr><tr><td>Warmup steps</td><td>256</td></tr><tr><td>Initial model / policy updates</td><td>100 /100</td></tr><tr><td>Epochs / steps per epoch</td><td>40 / 256</td></tr><tr><td>Model updates per epoch Particles</td><td>10</td></tr><tr><td>Model subsequence length</td><td>32</td></tr><tr><td>Network hidden dimensions</td><td>64</td></tr><tr><td>Recurrent lookback</td><td>0</td></tr><tr><td>Rollout length / discount</td><td>5</td></tr><tr><td>SAC actor / critic learning rate</td><td> $1 ~ / ~ 0 . 9$ </td></tr><tr><td>SAC batch size / entropy coefficient</td><td> $3 \times 1 0 ^ { - 4 } / 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td> $3 2 / 0 . 0 1$ </td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 5 } ~ / ~ 6 4$ </td></tr><tr><td>Particle filter mode  $/ \alpha$ </td><td> $\mathrm { M O P ~ / ~ } 0 . 8$ </td></tr><tr><td>ESS threshold / rejuvenation std.</td><td> $0 . 5 ~ / ~ 0 . 1$ </td></tr><tr><td>Environment noise: transition / observation 0.01 / 0.05</td><td></td></tr><tr><td>Action bound / horizon</td><td>3.0 / 200</td></tr></table>

Table 3: Key hyperparameters for LQR.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment</td><td>HeartSteps</td></tr><tr><td>User id / horizon Noise mode / observation noise</td><td>0 / 270 sequential / 0.0</td></tr><tr><td>Warmup steps Initial model / policy updates</td><td>1024 10 / 30</td></tr><tr><td>Epochs / steps per epoch Model updates per epoch Particles</td><td>27 / 10 10</td></tr><tr><td>GRU backbone / recurrent BPTT</td><td>16 false / false</td></tr><tr><td>Hidden size / preprocessor / postprocessor Rollout length / discount</td><td>32 / 32 / 32</td></tr><tr><td>SAC actor / critic learning rate</td><td> $1 ~ / ~ 0 . 9$   $1 0 ^ { - 3 } / 1 0 ^ { - 3 }$ </td></tr><tr><td>SAC batch size / entropy coefficient</td><td></td></tr><tr><td></td><td> $6 4 \mathrm { ~ / ~ } 0 . 0 0 3$ </td></tr><tr><td>Automatic entropy tuning</td><td>false</td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 2 5 6$ </td></tr><tr><td>Particle filter mode / α</td><td></td></tr><tr><td></td><td> $\mathrm { M O P ~ / ~ } 0 . 8$ </td></tr><tr><td>ESS threshold / rejuvenation std.</td><td>0.5 / 0.001</td></tr></table>

Table 4: Key hyperparameters for HeartSteps.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment</td><td>HalfCheetah</td></tr><tr><td>Observation variants</td><td>no position, no velocity</td></tr><tr><td>Observation noise / discount</td><td>0.1 / 0.99</td></tr><tr><td>Warmup steps</td><td>1000</td></tr><tr><td>Initial model / policy updates</td><td>1000 / 10</td></tr><tr><td>Epochs / steps per epoch</td><td>50 / 1000</td></tr><tr><td>Model updates per epoch</td><td>100</td></tr><tr><td>Early-stop patience / min updates</td><td>20/20 no-position; 5/5 no-velocity</td></tr><tr><td>Particles Model subsequence length</td><td>16 32</td></tr><tr><td>GRU backbone / BPTT size</td><td>true / 32</td></tr><tr><td>Hidden dimensions</td><td>[256, 256]</td></tr><tr><td>Rollout length / discount</td><td>10 / 0.99</td></tr><tr><td>SAC actor / critic learning rate</td><td> $3 \times 1 0 ^ { - 4 } / 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>SAC batch size / entropy coefficient 256 / 0.2</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 2 5 6$ </td></tr><tr><td>Particle filter mode / α</td><td>MOP / 0.8</td></tr></table>

Table 6: Key hyperparameters for MuJoCo HalfCheetah.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment</td><td>Ant</td></tr><tr><td>Observation variants</td><td>no position, no velocity</td></tr><tr><td>Observation noise / discount</td><td>0.1 / 0.99</td></tr><tr><td>Warmup steps</td><td>1000</td></tr><tr><td>Initial model / policy updates</td><td>2000 / 10</td></tr><tr><td>Epochs / steps per epoch</td><td>164 / 300 (submitted: 10 / 300)</td></tr><tr><td>Model updates per epoch</td><td>up to 10, early stopping (submitted: 0)</td></tr><tr><td>Policy update every</td><td>10 env. steps (submitted: 1)</td></tr><tr><td>AdamW weight decay (actor, critic, model) 10−4 (submitted: 0.1)</td><td></td></tr><tr><td>Particles Model subsequence length</td><td>16</td></tr><tr><td>GRU backbone / BPTT size</td><td>32</td></tr><tr><td></td><td>true / 32</td></tr><tr><td>Hidden dimensions</td><td>[256, 256]</td></tr><tr><td>Rollout length / discount</td><td>10 / 0.99</td></tr><tr><td>SAC actor / critic learning rate SAC batch size / entropy coefficient</td><td> $3 \times 1 0 ^ { - 4 } / 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td> $2 5 6 ~ / ~ 0 . 2$ </td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 2 5 6$ </td></tr><tr><td>Particle filter mode / α</td><td> $\mathrm { M O P ~ / ~ } 0 . 8$ </td></tr><tr><td>ESS threshold / rejuvenation std.</td><td> $0 . 5 ~ / ~ 0 . 1$ </td></tr></table>

Table 5: Key hyperparameters for MuJoCo Ant, using the corrected configuration of Figure 3. Values from the submitted version are given in parentheses where they difer.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment Observation variants Observation noise / discount</td><td>Hopper no position, no velocity</td></tr><tr><td>Warmup steps Initial model / policy updates Epochs / steps per epoch</td><td>0.1 / 0.99 1000 1000 /10 50 / 1000</td></tr><tr><td>Model updates per epoch Particles</td><td>100 16</td></tr><tr><td>Model subsequence length GRU backbone / BPTT size Hidden dimensions</td><td>32 true / 32</td></tr><tr><td></td><td>[256,256]</td></tr><tr><td></td><td></td></tr><tr><td>Rollout length / discount</td><td>10 / 0.99</td></tr><tr><td>SAC actor / critic learning rate</td><td></td></tr><tr><td></td><td> $3 \times 1 0 ^ { - 4 } / 3 \times 1 0 ^ { - 4 }$  256/0.2</td></tr><tr><td>SAC batch size / entropy coefficient</td><td></td></tr><tr><td></td><td></td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 2 5 6$ </td></tr><tr><td>Particle filter mode / α</td><td></td></tr><tr><td></td><td>MOP / 0.8</td></tr><tr><td>ESS threshold / rejuvenation std.</td><td>0.5 / 0.1</td></tr></table>

Table 7: Key hyperparameters for MuJoCo Hopper.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Environment</td><td>Walker2d</td></tr><tr><td>Observation variants</td><td>no position, no velocity</td></tr><tr><td>Observation noise / discount</td><td>0.1 / 0.99</td></tr><tr><td>Warmup steps</td><td>1000</td></tr><tr><td>Initial model / policy updates</td><td>1000 / 10</td></tr><tr><td>Epochs / steps per epoch</td><td>50 / 1000</td></tr><tr><td>Model updates per epoch</td><td>100</td></tr><tr><td>Early-stop patience / min updates</td><td>20/20 no-position; 5/5 no-velocity</td></tr><tr><td>Particles</td><td>16</td></tr><tr><td>Model subsequence length</td><td>32</td></tr><tr><td>GRU backbone / BPTT size</td><td>true / 32</td></tr><tr><td>Hidden dimensions</td><td>[256, 256]</td></tr><tr><td>Rollout length / discount</td><td>10 / 0.99</td></tr><tr><td>SAC actor / critic learning rate</td><td> $3 \times 1 0 ^ { - 4 } / 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>SAC batch size / entropy coefficient 256 / 0.2</td><td></td></tr><tr><td>Model learning rate / batch size</td><td> $3 \times 1 0 ^ { - 4 } ~ / ~ 2 5 6$ </td></tr><tr><td>Particle filter mode / α</td><td>MOP /0.8</td></tr></table>

Table 8: Key hyperparameters for MuJoCo Walker2d.

## D Detailed Error Analysis and Proofs

Throughout, $| r ( s , a ) | \leq R _ { \mathrm { m a x } }$ and $V _ { \operatorname* { m a x } } : = R _ { \operatorname* { m a x } } / ( 1 - \gamma )$ . We define the per-round error quantities:

$$
\epsilon _ { \mathrm { m c } } ( t , d ) : = \operatorname* { s u p } _ { a \in \mathcal { A } } \bigl | \widehat { Q } _ { t } ( b _ { t } ^ { J } , a ) - Q _ { \theta , d } ( b _ { t } ^ { J } , a ) \bigr | ,
$$

$$
\epsilon _ { \mathrm { s t r u c t } } ( t , d ) : = \operatorname* { s u p } _ { a \in \mathcal { A } } \bigl | Q _ { \theta } ^ { \star , B } ( b _ { t } ^ { J } , a ) - Q _ { \theta , d } ( b _ { t } ^ { J } , a ) \bigr | .
$$

## D.1 Structural Error of QMDP Bootstrapping

Define the linear QMDP surrogate

$$
\widetilde { V } _ { \theta } ^ { \star } ( b ) : = \int _ { S } V _ { \theta } ^ { \star , S } ( s ) b ( \mathrm { d } s ) .\tag{18}
$$

By Jensen’s inequality, $\widetilde { V } _ { \theta } ^ { \star } ( b ) \geq \bar { V } _ { \theta } ^ { \star , S } ( b )$ and the gap $\Xi _ { \theta } : = \operatorname* { s u p } _ { b } [ \widetilde { V } _ { \theta } ^ { \star } ( b ) - \bar { V } _ { \theta } ^ { \star , S } ( b ) ] \geq 0$

Proposition 2 (Structural error of MCTQ). For every $\theta \in \Theta$ and $d \geq 0$

$$
\| \bar { V } _ { \theta , d } ^ { \star , S } - V _ { \theta } ^ { \star , B } \| _ { \infty } \leq \frac { \gamma ^ { d + 1 } } { 1 - \gamma } \Xi _ { \theta } .\tag{19}
$$

In particular, $\begin{array} { r } { \epsilon _ { \mathrm { s t r u c t } } ( u , d ) \leq \frac { \gamma ^ { d + 1 } } { 1 - \gamma } \Xi _ { \bar { \theta } _ { u } } } \end{array}$

Proof of Proposition 2. Write

$$
\begin{array} { r l } & { Q _ { \theta } ^ { \star , B } ( b , a ) = r ( b , a ) + \gamma \displaystyle \int _ { \mathcal { O } } V _ { \theta } ^ { \star , B } \left( \tau _ { \theta } ( b , a , o ) \right) p _ { \theta } ( o \mid b , a ) \mathrm { d } o , } \\ & { Q _ { \theta , 0 } ( b , a ) = r ( b , a ) + \gamma \displaystyle \int _ { S } V _ { \theta } ^ { \star , S } \left( s ^ { \prime } \right) \tilde { b } _ { \theta } ( \mathrm { d } s ^ { \prime } \mid b , a ) = r ( b , a ) + \gamma \widetilde { V } _ { \theta } ^ { \star } ( \tilde { b } _ { \theta } ( \cdot \mid b , a ) ) . } \end{array}
$$

By Bayes consistency and linearity of $\widetilde { V } _ { \boldsymbol { \theta } } ^ { \star }$ :

$$
\widetilde { V } _ { \theta } ^ { \star } ( \tilde { b } _ { \theta } ( \cdot \mid b , a ) ) = \int _ { \mathcal { O } } \widetilde { V } _ { \theta } ^ { \star } ( \tau _ { \theta } ( b , a , o ) ) p _ { \theta } ( o \mid b , a ) \mathrm { d } o .
$$

Subtracting and using $\widetilde { V } _ { \theta } ^ { \star } = \bar { V } _ { \theta } ^ { \star , S } + ( \widetilde { V } _ { \theta } ^ { \star } - \bar { V } _ { \theta } ^ { \star , S } )$ with $0 \leq \widetilde { V } _ { \theta } ^ { \star } - \bar { V } _ { \theta } ^ { \star , S } \leq \Xi _ { \theta }$

$$
\| Q _ { \theta } ^ { \star , B } - Q _ { \theta , 0 } \| _ { \infty } \leq \gamma \| V _ { \theta } ^ { \star , B } - \bar { V } _ { \theta } ^ { \star , S } \| _ { \infty } + \gamma \Xi _ { \theta } .
$$

Since $\begin{array} { r } { \| V _ { \theta } ^ { \star , B } - \bar { V } _ { \theta } ^ { \star , S } \| _ { \infty } = \operatorname* { s u p } _ { b } | \operatorname* { m a x } _ { a } Q _ { \theta } ^ { \star , B } ( b , a ) - \operatorname* { m a x } _ { a } Q _ { \theta , 0 } ( b , a ) | \leq \| Q _ { \theta } ^ { \star , B } - Q _ { \theta , 0 } \| _ { \infty } } \end{array}$ (using | max $f -$ max $g | \leq$ max $\left| f - g \right| )$ , rearranging gives $\begin{array} { r } { \| V _ { \theta } ^ { \star , B } - \bar { V } _ { \theta } ^ { \star , S } \| _ { \infty } \leq \frac { \gamma } { 1 - \gamma } \Xi _ { \theta } } \end{array}$ (the $d = 0 { \mathrm { ~ c a s e ) } }$ . For general $d \ge 0 \colon \bar { V } _ { \theta , d } ^ { \star , S } = \mathcal { T } _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S }$ and $V _ { \theta } ^ { \star , B } = \mathcal { T } _ { \theta } V _ { \theta } ^ { \star , B }$ (fixed point), so

$$
\| \bar { V } _ { \theta , d } ^ { \star , S } - V _ { \theta } ^ { \star , B } \| _ { \infty } = \| \mathcal { T } _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S } - \mathcal { T } _ { \theta } ^ { d } V _ { \theta } ^ { \star , B } \| _ { \infty } \leq \gamma ^ { d } \| \bar { V } _ { \theta } ^ { \star , S } - V _ { \theta } ^ { \star , B } \| _ { \infty } \leq \frac { \gamma ^ { d + 1 } } { 1 - \gamma } \Xi _ { \theta } .
$$

## D.2 Planning and State-Identification Error

Definition 1 (Q-belief distance). For $b , b ^ { \prime } \in \Delta ( \mathcal { S } )$ , define $\begin{array} { r } { d _ { Q } ( b , b ^ { \prime } ) : = \operatorname* { s u p } _ { \theta \in \Theta , a \in \mathcal { A } } | Q _ { \theta } ^ { \star , B } ( b , a ) - } \end{array}$ $Q _ { \theta } ^ { \star , B } ( b ^ { \prime } , a ) |$

Lemma 1 $( Q ^ { \star , B } .$ -diference via test functions). Fix $\theta \in \Theta , a \in { \mathcal { A } }$ , and truncation depth $H \geq 1$ Recall the Bayesian update $\tau ( b , a , o ) \in \Delta ( \mathcal { S } )$ , the observation marginal $\begin{array} { r } { p ( o \mid b , a ) = \int Z _ { \theta } ( o \mid s ^ { \prime } ) P _ { \theta } ( s ^ { \prime } \mid } \end{array}$ $s , a ) b ( d s ) d s ^ { \prime }$ , and define

$$
\psi _ { o , a } ( s ) : = \int Z _ { \theta } ( o \mid s ^ { \prime } ) P _ { \theta } ( s ^ { \prime } \mid s , a ) d s ^ { \prime } , \qquad \tilde { f } _ { o , a } ( s ) : = \int f ( s ^ { \prime } ) Z _ { \theta } ( o \mid s ^ { \prime } ) P _ { \theta } ( s ^ { \prime } \mid s , a ) d s ^ { \prime } ,\tag{20}
$$

so that $p ( o \mid b , a ) = b ( \psi _ { o , a } )$ and $\tau ( b , a , o ) ( f ) = b ( \tilde { f } _ { o , a } ) / b ( \psi _ { o , a } )$

Under Assumption 1, for all $b , b ^ { \prime } \in \Delta ( { \cal S } )$

$$
| Q _ { \theta } ^ { \star , B } ( b , a ) - Q _ { \theta } ^ { \star , B } ( b ^ { \prime } , a ) | \ \leq \ \sum _ { \ell = 0 } ^ { H - 1 } \gamma ^ { \ell } C _ { \mathrm { B U } } ^ { \ell } \operatorname* { s u p } _ { f \in \mathcal { F } _ { \ell } ( \theta , a ) } | b ( f ) - b ^ { \prime } ( f ) | \ + \ 2 \gamma ^ { H } V _ { \mathrm { m a x } } ,\tag{21}
$$

where $C _ { \mathrm { B U } } : = 2 g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } } ^ { 2 }$ and ${ \mathcal { F } } _ { \ell } ( \theta , a )$ is a class of functions $\mathcal { S }  [ - g _ { \mathrm { m a x } } ^ { \ell } V _ { \mathrm { m a x } } , g _ { \mathrm { m a x } } ^ { \ell } V _ { \mathrm { m a x } } ]$ with $\mathcal { N } _ { \mathrm { [ ] } } ( \mathcal { F } _ { \ell } , \epsilon ) \leq | A | ^ { \ell } \mathcal { N } _ { \Theta } ( \epsilon / ( \ell V _ { \mathrm { m a x } } ) )$ , defined recursively:

$$
\bullet \ { \mathcal { F } } _ { 0 } ( \theta , a ) = \{ r ( \cdot , a ^ { \prime } ) : a ^ { \prime } \in { \mathcal { A } } \} \cup \{ Q _ { \theta } ^ { \star , S } ( \cdot , a ^ { \prime } ) : a ^ { \prime } \in { \mathcal { A } } \} ,
$$

$$
\bullet \ \mathcal { F } _ { \ell + 1 } ( \theta , a ) = \left\{ \widetilde { f } _ { o , a ^ { \prime \prime } } : \ f \in \mathcal { F } _ { \ell } ( \theta , a ^ { \prime \prime } ) , \ a ^ { \prime \prime } \in A \right\} \cup \left\{ \psi _ { o , a ^ { \prime \prime } } : a ^ { \prime \prime } \in A \right\} .
$$

Proof. We prove by induction on h: for any $\begin{array} { r } { \Phi ( b ) = \mathbb { E } _ { \theta } ^ { \pi } [ \sum _ { u = 0 } ^ { h - 1 } \gamma ^ { u } r _ { u } \mid b _ { 0 } = b ] } \end{array}$ with $h \leq H , | \Phi ( b ) - \Phi ( b ^ { \prime } ) |$ admits the decomposition equation 21. Taking $\Phi = Q _ { \theta } ^ { \star , B } ( \cdot , a )$ with $h = H$ and bounding the tail by $2 \gamma ^ { H } V _ { \mathrm { m a x } }$ yields the result.

Base case $\begin{array} { r } { ( h = 1 ) . \ | \Phi ( b ) - \Phi ( b ^ { \prime } ) | = | \operatorname* { s u p } _ { a } [ b ( r _ { a } ) - b ^ { \prime } ( r _ { a } ) ] | \leq \operatorname* { s u p } _ { f \in \mathcal { F } _ { 0 } } | b ( f ) - b ^ { \prime } ( f ) | . } \end{array}$

Inductive step $( h  h + 1 )$ . For a fixed action a, write

$$
\Phi ( b ) = b ( r _ { a } ) + \gamma \int _ { \mathcal { O } } \Phi _ { h - 1 } \bigl ( \tau ( b , a , o ) \bigr ) p ( o \mid b , a ) d o ,
$$

where $\Phi _ { h - 1 }$ is the $( h - 1 )$ -step continuation. The continuation diference decomposes as:

$$
\begin{array} { r l } & { \bigg | \int \Phi _ { h - 1 } ( \tau ( b , a , o ) ) p ( o | b , a ) d o - \int \Phi _ { h - 1 } ( \tau ( b ^ { \prime } , a , o ) ) p ( o | b ^ { \prime } , a ) d o \bigg | } \\ & { \leq \underbrace { | | \Phi _ { h - 1 } | | _ { \infty } \displaystyle \int | ( b - b ^ { \prime } ) ( \psi _ { o , a } ) | d o } _ { \mathrm { ( I ) ~ o b s . ~ d i s t r i b u t i o n ~ s h i f t } } + \underbrace { \int | \Phi _ { h - 1 } ( \tau ( b , a , o ) ) - \Phi _ { h - 1 } ( \tau ( b ^ { \prime } , a , o ) ) | p ( o | b ^ { \prime } , a ) d o } _ { \mathrm { ( I I ) ~ b e l i e f ~ p r o p a g a t i o n } } , } \end{array}\tag{22}
$$

using $p ( o \mid b , a ) - p ( o \mid b ^ { \prime } , a ) = ( b - b ^ { \prime } ) ( \psi _ { o , a } )$

Term (I) contributes $\psi _ { o , a } \in \mathcal { F } _ { \ell + 1 }$

Term (II) requires bounding $\vert \tau ( b , a , o ) ( f ) - \tau ( b ^ { \prime } , a , o ) ( f ) \vert$ for $\| f \| _ { \infty } \leq 1$ . Since $\tau ( b , a , o ) ( f ) =$ $b ( \tilde { f } _ { o , a } ) / b ( \psi _ { o , a } )$ , the quotient rule gives

$$
\frac { b ( \tilde { f } _ { o , a } ) } { b ( \psi _ { o , a } ) } - \frac { b ^ { \prime } ( \tilde { f } _ { o , a } ) } { b ^ { \prime } ( \psi _ { o , a } ) } = \frac { ( b - b ^ { \prime } ) ( \tilde { f } _ { o , a } ) } { b ( \psi _ { o , a } ) } - \frac { b ^ { \prime } ( \tilde { f } _ { o , a } ) } { b ( \psi _ { o , a } ) b ^ { \prime } ( \psi _ { o , a } ) } ( b - b ^ { \prime } ) ( \psi _ { o , a } ) .\tag{23}
$$

Since $b ( \psi _ { o , a } ) , b ^ { \prime } ( \psi _ { o , a } ) \geq g _ { \mathrm { m i n } }$ and $| b ^ { \prime } ( { \tilde { f } } _ { o , a } ) | \leq g _ { \mathrm { m a x } }$ :

$$
| \tau ( b , a , o ) ( f ) - \tau ( b ^ { \prime } , a , o ) ( f ) | \leq \frac { | ( b - b ^ { \prime } ) ( \tilde { f } _ { o , a } ) | } { g _ { \mathrm { m i n } } } + \frac { g _ { \mathrm { m a x } } } { g _ { \mathrm { m i n } } ^ { 2 } } | ( b - b ^ { \prime } ) ( \psi _ { o , a } ) | .\tag{24}
$$

Both $\tilde { f } _ { o , a }$ and $\psi _ { o , a }$ are in $\mathcal { F } _ { \ell + 1 }$ , with per-update amplification

$$
C _ { \mathrm { B U } } = \frac { 1 } { g _ { \mathrm { m i n } } } + \frac { g _ { \mathrm { m a x } } } { g _ { \mathrm { m i n } } ^ { 2 } } \ \leq \ \frac { 2 g _ { \mathrm { m a x } } } { g _ { \mathrm { m i n } } ^ { 2 } } ,
$$

By the inductive hypothesis, $| \Phi _ { h - 1 } ( \tau ( b , a , o ) ) - \Phi _ { h - 1 } ( \tau ( b ^ { \prime } , a , o ) ) |$ decomposes into evaluations of $\tau ( b , a , o )$ vs. $\tau ( b ^ { \prime } , a , o )$ on $\mathcal { F } _ { \ell } .$ . Substituting equation 24 reduces each to evaluations of b vs. $b ^ { \prime }$ on $\mathcal { F } _ { \ell + 1 }$ , with one factor of C<sub>BU</sub> per level. Taking ${ \mathrm { s u p } } _ { a }$ over the action set and iterating across H levels yields equation 21. □

Lemma 2 $( Q ^ { \star , B }$ is Lipschitz in TV). For any $\theta \in \Theta , a \in { \mathcal { A } }$ , and b, $b ^ { \prime } \in \Delta ( \mathcal { S } )$

$$
| Q _ { \theta } ^ { \star , B } ( b , a ) - Q _ { \theta } ^ { \star , B } ( b ^ { \prime } , a ) | \ \leq \ 2 V _ { \operatorname* { m a x } } \| b - b ^ { \prime } \| _ { \mathrm { T V } } .
$$

Proof. $| r ( b , a ) - r ( b ^ { \prime } , a ) | ~ \leq ~ 2 R _ { \operatorname* { m a x } } \| b - b ^ { \prime } \| _ { \mathrm { T V } }$ . The trajectory law under b and b<sup>′</sup> difers in TV by at most $\| b - b ^ { \prime } \| _ { \mathrm { T V } }$ (data processing), so the $V _ { \mathrm { m a x } ^ { - } } \mathrm { b o u n d e d }$ continuation difers by at most $2 V _ { \operatorname* { m a x } } \Vert b - b ^ { \prime } \Vert _ { \mathrm { T V } }$ . Using $R _ { \mathrm { m a x } } + \gamma V _ { \mathrm { m a x } } = V _ { \mathrm { m a x } }$ gives the result. □

Lemma 3 (Performance decomposition). Fix $\theta \in \Theta$ . Let $\pi _ { t }$ denote the stationary policy induced by the planner with frozen parameters (model $\theta _ { k ( t ) }$ , belief $b _ { t } ^ { J } )$ at time $t ,$ and let $\bar { \delta _ { u } } : = \bar { V _ { \theta } ^ { \star , B } } ( b _ { u } ) -$ $Q _ { \theta } ^ { \star , B } ( b _ { u } , A _ { u } )$ where $A _ { u } \sim \pi _ { t } ( \cdot \mid H _ { u } )$ . Then

$$
V _ { \theta } ^ { \star , B } ( b _ { t } ) - V _ { \theta } ^ { \pi _ { t } , B } ( b _ { t } ) \ \leq \ \mathbb { E } \left[ \sum _ { u = t } ^ { \infty } \gamma ^ { u - t } \delta _ { u } \ \Big | \ H _ { t } \right] .
$$

Proof. Unroll the Bellman recursion $V _ { \theta } ^ { \star , B } ( b _ { u } ) - V _ { \theta } ^ { \pi _ { t } , B } ( b _ { u } ) = \delta _ { u } + \gamma \mathbb { E } [ V _ { \theta } ^ { \star , B } ( b _ { u + 1 } ) - V _ { \theta } ^ { \pi _ { t } , B } ( b _ { u + 1 } ) \mid H _ { u } ]$ from $u = t { \mathrm { ~ t o ~ } } u = t + n .$ , using $| V _ { \theta } ^ { \star , B } - V _ { \theta } ^ { \pi _ { t } , B } | \leq \overline { { 2 } } V _ { \mathrm { m a x } }$ as the tail bound, and send $n  \infty . \qquad \bigsqcup$

Lemma 4 (Per-step bound on $\delta _ { u } )$ . For every u and any $d \geq 0$

$$
\delta _ { u } \ \leq \ 2 d _ { Q } ( b _ { u } , b _ { u } ^ { J } ) + 2 \epsilon _ { \mathrm { m c } } ( u , d ) + 2 \epsilon _ { \mathrm { s t r u c t } } ( u , d ) .
$$

Proof. Let $a _ { u } ^ { \star } \in \arg$ ma $\mathrm { x } _ { a } Q _ { \theta } ^ { \star , B } ( b _ { u } , a )$ and $\hat { a } _ { u } \in \arg \operatorname* { m a x } _ { a } \widehat { Q } _ { u } ( b _ { u } ^ { J } , a )$ . Then

$$
\begin{array} { r l } & { \delta _ { u } = Q _ { \theta } ^ { \star , B } ( b _ { u } , a _ { u } ^ { \star } ) - Q _ { \theta } ^ { \star , B } ( b _ { u } , \hat { a } _ { u } ) } \\ & { \quad \leq \underbrace { | Q _ { \theta } ^ { \star , B } ( b _ { u } , a _ { u } ^ { \star } ) - Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , a _ { u } ^ { \star } ) | + | Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , \hat { a } _ { u } ) - Q _ { \theta } ^ { \star , B } ( b _ { u } , \hat { a } _ { u } ) | } _ { \leq 2 d _ { Q } ( b _ { u } , b _ { u } ^ { J } ) ~ \mathrm { ~ b y ~ D e f i n i t i o n ~ 1 ~ } } } \\ & { \quad \quad + Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , a _ { u } ^ { \star } ) - Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , \hat { a } _ { u } ) . } \end{array}
$$

For the last line, add and subtract $Q _ { \theta , d }$ and $\widehat { Q } _ { u }$ , using optimality of $\hat { a } _ { u }$ for $\widehat { Q } _ { u }$

$$
Q _ { \theta , d } ( b _ { u } ^ { J } , a _ { u } ^ { \star } ) \leq \widehat { Q } _ { u } ( b _ { u } ^ { J } , a _ { u } ^ { \star } ) + \epsilon _ { \mathrm { { m c } } } \leq \widehat { Q } _ { u } ( b _ { u } ^ { J } , \widehat { a } _ { u } ) + \epsilon _ { \mathrm { { m c } } } \leq Q _ { \theta , d } ( b _ { u } ^ { J } , \widehat { a } _ { u } ) + 2 \epsilon _ { \mathrm { { m c } } } ,
$$

so $Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , a _ { u } ^ { \star } ) - Q _ { \theta } ^ { \star , B } ( b _ { u } ^ { J } , \hat { a } _ { u } ) \leq 2 \epsilon _ { \mathrm { s t r u c t } } + 2 \epsilon _ { \mathrm { m c } }$

Theorem 2 (Round-t planning + state-identification bound). For each $t \geq 1$ and any depth $d ,$

$$
V _ { \tilde { \theta } _ { k ( t ) } } ^ { \star , B } ( b _ { t } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi _ { t } , B } ( b _ { t } ) \ \leq \ \mathbb { E } \left[ \sum _ { u = t } ^ { \infty } \gamma ^ { u - t } \big ( 2 d _ { Q } ( b _ { u } , b _ { u } ^ { J } ) + 2 \epsilon _ { \mathrm { m c } } ( u , d ) + 2 \epsilon _ { \mathrm { s t r u c t } } ( u , d ) \big ) \ \Big | \ H _ { t } \right] .
$$

Proof. Substitute the per-step bound of Lemma 4 into Lemma 3.

Observation discretization. For continuous ${ \mathcal { O } } ,$ the tree never revisits observation-children (each sampled o is unique), so every node below the root holds a single particle. At $d = 1$ each leaf value equation 15 then reduces to max $\mathfrak { c } _ { a ^ { \prime } } Q _ { \theta } ^ { \star , S } ( s ^ { \prime } , a ^ { \prime } )$ at a single state, which targets the linear surrogate rather than $Q _ { \theta , 1 }$ , and for $d \geq 2$ the tree cannot build depth beyond 1. We therefore discretize the observations. Let $\phi : \mathcal { O } \to \bar { \mathcal { O } }$ with $| \bar { \mathcal { O } } | = M$ and bin diameter diam $\begin{array} { r } { ( \phi ) : = \operatorname* { m a x } _ { \bar { o } } \operatorname* { s u p } _ { o , o ^ { \prime } \in \phi ^ { - 1 } ( \bar { o } ) } \| o - o ^ { \prime } \| } \end{array}$ Algorithms 2 and 3 are modified so that $\mathrm { C } _ { \mathrm { H I L D } } ( v , a ^ { \prime } , \phi ( o ^ { \prime } ) )$ replaces $\mathrm { C } _ { \mathrm { H I L D } } ( v , a ^ { \prime } , o ^ { \prime } )$ . The bin-averaged posterior is $\begin{array} { r } { \tau _ { \theta } ^ { \phi } ( b , a , \bar { o } ) : = \int _ { \phi ^ { - 1 } ( \bar { o } ) } \tau _ { \theta } ( b , a , o ) p _ { \theta } ( o \mid b , a ) \mathrm { d } o \big / \int _ { \phi ^ { - 1 } ( \bar { o } ) } p _ { \theta } ( o \mid b , a ) \mathrm { d } o } \end{array}$ , and we write $Q _ { \theta , d } ^ { \phi }$ for the target equation 9 with $\tau _ { \theta } ( b , a , o )$ replaced by $\tau _ { \theta } ^ { \phi } ( b , a , \phi ( o ) )$ at every level. When $| \mathcal { O } | < \infty$ , take $\phi = \mathrm { i d } , M = | \mathcal { O } |$ , diam(ϕ) = 0. When $d = 0$ , no tree is built and no discretization is needed.

Seeds and the estimator. Fix a round t and a root action $^ { a , }$ and write $\boldsymbol { \theta } : = \bar { \boldsymbol { \theta } } _ { k ( t ) }$ and $b ^ { J } : = b _ { t } ^ { J }$ All probabilities and expectations below are conditional on $b _ { t } ^ { J }$ , and we take $d \geq 1$ . Simulation $n \in [ N ]$ is driven by a seed

$$
\begin{array} { r } { \xi _ { n } : = \big ( j _ { n } , u _ { n , 0 } , ( a _ { n , \ell } ^ { \prime } , u _ { n , \ell } ) _ { \ell = 1 } ^ { d - 1 } \big ) , } \end{array}\tag{25}
$$

where $j _ { n } \sim \operatorname { U n i f } ( [ J ] )$ is the sampled particle index, $a _ { n , \ell } ^ { \prime } \sim \mathrm { U n i f } ( \mathcal { A } )$ is the action drawn at depth $\ell ,$ and $u _ { n , \ell }$ is the noise that generates the next state and observation at depth ℓ from $P _ { \theta }$ and $Z _ { \theta }$ . The seeds $\xi _ { 1 } , \dots , \xi _ { N }$ are i.i.d. Running Algorithm 2 on them produces the tree $T _ { a }$ , and we write the root estimate as a deterministic function of all seeds,

$$
\widehat { Q } ( b ^ { J } , a ) = f _ { a } ( \xi _ { 1 } , \ldots , \xi _ { N } ) : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \bigl [ r ( s _ { j _ { n } } , a ) + \gamma \hat { V } ( v _ { n , 1 } ) \bigr ] ,\tag{26}
$$

where $v _ { n , \ell }$ is the depth-ℓ node on the path of simulation n and $\hat { V }$ is the DP backup equation 17. The function $f _ { a }$ is not an average of independent terms, since $\hat { V } ( v _ { n , 1 } )$ depends on every simulation that passes through $v _ { n , 1 }$ . For $S \subseteq [ N ]$ , let $f _ { a } ^ { ( - S ) }$ be the same expression computed from the seeds $( \xi _ { m } ) _ { m \notin S }$ alone, with the sum running over n $\notin S$ and the normalizer $1 / N$ retained.

Balancedness. For $\rho \geq 1$ , let $\mathcal { E } _ { \rho }$ be the event that every internal node v of $T _ { a }$ satisfies

$$
N ( v ) \leq \rho \left| { \cal A } \right| N ( v , a ^ { \prime } ) \qquad { \mathrm { f o r ~ a l l ~ } } a ^ { \prime } \in { \mathcal { A } } ,\tag{27}
$$

so that every action receives at least a $1 / ( \rho | \mathcal { A } | )$ fraction of the visits of every internal node. Define the depth factor

$$
\Lambda _ { d } : = \sum _ { \ell = 0 } ^ { d - 1 } ( \rho \gamma | \boldsymbol { \cal { A } } | ) ^ { \ell } ,\tag{28}
$$

so that $\Lambda _ { 0 } = 0$ and $\Lambda _ { 1 } = 1$ . Let $p _ { \rho } : = \operatorname* { m a x } _ { a \in \mathcal { A } } \operatorname* { P r } ( \mathcal { E } _ { \rho } ^ { c } \mid b _ { t } ^ { J } )$ , where the event refers to the tree $T _ { a }$ , and let

$$
q _ { \operatorname* { m i n } } : = \operatorname* { i n f } _ { \substack { s \in S , a ^ { \prime \prime } \in \mathcal { A } , \bar { \sigma } \in \bar { \mathcal { O } } } } \int _ { S } Z _ { \theta } \big ( \phi ^ { - 1 } ( \bar { o } ) \mid s ^ { \prime } \big ) P _ { \theta } ( \mathrm { d } s ^ { \prime } \mid s , a ^ { \prime \prime } )
$$

be the smallest one-step bin probability. Under the bounded observation density of Assumption 1, $q _ { \mathrm { m i n } } \ge g _ { \mathrm { m i n } } \operatorname* { m i n } _ { \bar { o } } \lambda ( \phi ^ { - 1 } ( \bar { o } ) )$ , where λ is the reference measure of $Z _ { \theta } ;$ for finite O with the counting measure, $q _ { \mathrm { m i n } } \ge g _ { \mathrm { m i n } }$

Proposition 3 (Monte Carlo error). Run Algorithm 2 with uniform internal selection equation 16, N simulations per root action, observation discretization $\phi \ ( | \bar { \mathcal { O } } | = M )$ , and returns in $[ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ]$ and let $\rho \geq 1$ . Let

$$
\mathrm { b i a s } ( N , d ) : = \operatorname* { s u p } _ { a } \bigl | \mathbb { E } [ \widehat { Q } ( b ^ { J } , a ) \mid b _ { t } ^ { J } ] - Q _ { \bar { \theta } _ { k ( t ) } , d } ( b ^ { J } , a ) \bigr |\tag{29}
$$

and $B _ { d } : = 2 V _ { \operatorname* { m a x } } \big ( 1 + 2 ( 1 + \gamma ) \Lambda _ { d } \big )$ . For every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta - | \mathcal { A } | p _ { \rho }$ conditionally on $b _ { t } ^ { J }$

$$
\epsilon _ { \mathrm { m c } } ( t , d ) \ \leq \ \underbrace { 2 ( 1 + \gamma ) V _ { \mathrm { m a x } } \Lambda _ { d } \sqrt { \frac { 2 \log ( 2 | A | / \delta ) } { N } } } _ { f u c t u a t i o n } + \ \mathrm { b i a s } ( N , d ) \ + \ B _ { d } p _ { \rho } ,\tag{30}
$$

and, in second moment,

$$
{ \mathbb E } \big [ \epsilon _ { \mathrm { m c } } ( t , d ) ^ { 2 } \big | b _ { t } ^ { J } \big ] \ \leq \ \frac { 8 ( 1 + \gamma ) ^ { 2 } | A | V _ { \operatorname* { m a x } } ^ { 2 } \Lambda _ { d } ^ { 2 } } { N } \ + \ 2 | A | \mathrm { b i a s } ( N , d ) ^ { 2 } \ + \ 3 | A | B _ { d } ^ { 2 } p _ { \rho } .\tag{31}
$$

The bias satisfies

$$
\begin{array} { r l r } { \mathrm { b i a s } ( N , d ) } & { \leq } & { \underbrace { \frac { c _ { 0 } \Phi _ { d } V _ { \mathrm { m a x } } \sqrt { | \cal { A } | } } { \sqrt { N } } } _ { D P \ b a c k u p \ e s t i m a t i o n \ e r r o r } + \underbrace { \frac { 2 \gamma ( 1 - \gamma ^ { d } ) } { 1 - \gamma } \cdot \frac { g _ { \mathrm { m a x } } } { g _ { \mathrm { m i n } } } V _ { \mathrm { m a x } } \omega _ { Z } ( \mathrm { d i a m } ( \phi ) ) } _ { d i s c r e t i z a t i o n \ e r r o r } , } \end{array}\tag{32}
$$

where $c _ { 0 } > 0$ is a constant independent of N, M and $\delta ,$

$$
\Phi _ { d } : = \sum _ { \ell = 1 } ^ { d } ( \gamma \sqrt { | { \cal A } | M } ) ^ { \ell } ,\tag{33}
$$

and $\begin{array} { r } { \omega _ { Z } ( \delta ) : = \operatorname* { s u p } _ { \theta , s } \operatorname* { s u p } _ { \| o - o ^ { \prime } \| < \delta } | Z _ { \theta } ( o \mid s ) / Z _ { \theta } ( o ^ { \prime } \mid s ) - 1 | . \ I f d \le 1 } \end{array}$ , then $p _ { \rho } = 0$ $H d \ge 2$ $q _ { \mathrm { m i n } } > 0$ and $| \mathcal { A } | M \ge 2$ , then $p _ { 2 } \ \leq \ 2 | A | ( | A | M ) ^ { d } \exp \bigl ( - N ( q _ { \mathrm { m i n } } / | A | ) ^ { d - 1 } / 3 2 \bigr )$ . When $d = 0 , \widehat { Q } ( b ^ { J } , a ) =$ $\begin{array} { r l } { \frac { 1 } { J } \sum _ { j } Q _ { \theta } ^ { \star , S } ( s _ { j } , a ) } \end{array}$ is exact and every term vanishes $( \Lambda _ { 0 } = \Phi _ { 0 } = 0 )$ . When $\phi = \mathrm { i d } \ ( M = | \mathcal { O } |$ diam $\left( \phi \right) = 0 )$ , the discretization error vanishes.

No stationarity assumption on the rewards at internal nodes is required, and the simulation returns need not be independent. The proof uses only that the seeds are i.i.d., that the path of each simulation depends on its own seed alone, and that the estimator has bounded diferences on $\mathcal { E } _ { \rho }$ . It rests on the following four results; throughout, we fix a root action a and $d \geq 1$

Lemma 5 (Path decoupling). There is a deterministic map ν such that, whatever the other seeds are, the path of simulation m (its states, actions, binned observations, and nodes $v _ { m , 1 } , \ldots , v _ { m , d } )$ equals $\nu ( \xi _ { m } )$

Proof. We induct on depth. Simulation m starts from $s _ { j _ { m } }$ with the root action $a ,$ and $u _ { m , 0 }$ generates its next state and observation, so $v _ { m , 1 } = \left( a , \phi ( o ) \right)$ is a function of $\xi _ { m }$ . Suppose the state $s _ { m , \ell }$ and the node $v _ { m , \ell }$ at depth $\ell < d$ are functions of $\xi _ { m }$ . Whether or not an earlier simulation expanded $v _ { m , \ell }$ , Algorithm 3 only initializes statistics and continues, and it selects the action $a _ { m , \ell } ^ { \prime } ,$ a component of $\xi _ { m }$ that does not depend on any tree statistic. The next state and observation are generated from $( s _ { m , \ell } , a _ { m , \ell } ^ { \prime } , u _ { m , \ell } )$ , and $v _ { m , \ell + 1 }$ is the history of $v _ { m , \ell }$ extended by $a _ { m , \ell } ^ { \prime }$ and the binned observation. Hence $s _ { m , \ell + 1 }$ and $v _ { m , \ell + 1 }$ are functions of $\xi _ { m }$ , and the simulation stops at depth d regardless of the other seeds. □

Both hypotheses of the lemma are needed. Under UCB1, the action at $v _ { m , \ell }$ depends on the statistics left by simulations $1 , \ldots , m - 1$ . Under a POMCP-style early return (“if v is not yet expanded, expand it and return”), simulation m stops at the first node that no earlier simulation created, so its depth depends on $\xi _ { 1 } , \ldots , \xi _ { m - 1 }$

Corollary 1 (Localized influence). Let $S \subseteq [ N ]$ and let $P _ { S }$ be the set of nodes on the paths $\nu ( \xi _ { m } )$ $m \in S$ . Every node $v \not \in P _ { S }$ has the same particle set $B ( v )$ , counts $N ( v )$ and $N ( v , a ^ { \prime } )$ , mean rewards $\bar { r } ( v , a ^ { \prime } )$ , and $D P$ value $\hat { V } ( v )$ in the runs on $( \xi _ { m } ) _ { m \in [ N ] }$ and on $( \xi _ { m } ) _ { m \notin S }$ . In particular, replacing one seed $\xi _ { n }$ by $\xi _ { n } ^ { \prime }$ changes these quantities only at nodes on the two root-to-leaf paths $\nu ( \xi _ { n } )$ and $\nu ( \xi _ { n } ^ { \prime } )$

Proof. By Lemma 5, in either run the simulations through v are those m with $v \in \nu ( \xi _ { m } )$ among the seeds present. For v $\notin P _ { S }$ no $m \in S$ is among them, so $B ( v ) , N ( v ) , N ( v , a ^ { \prime } )$ and $\bar { r } ( v , a ^ { \prime } )$ coincide. $\mathrm { A }$ descendant of v lies on a path in $P _ { S }$ only if v does, so the whole subtree of v avoids $P _ { S }$ , and backward induction on depth with equation 17 gives the same $\hat { V } ( v )$ . For the last claim, compare both runs with the run on $( \xi _ { m } ) _ { m \neq n }$ □

Lemma 6 (Bounded diference). On $\mathcal { E } _ { \rho ; }$ , for every $S \subseteq [ N ]$

$$
\big | f _ { a } ( \xi _ { 1 } , \dots , \xi _ { N } ) - f _ { a } ^ { ( - S ) } \big | \le | S | \cdot \frac { 2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } \Lambda _ { d } } { N } .\tag{34}
$$

Consequently, if two seed vectors $\xi , \xi ^ { \prime } \ d i f f e r$ in k coordinates and both of their runs lie in $\mathcal { E } _ { \rho } ,$ then $| f _ { a } ( \xi ) - f _ { a } ( \xi ^ { \prime } ) | \le k c ,$ , where

$$
c : = \frac { 4 ( 1 + \gamma ) V _ { \mathrm { m a x } } \Lambda _ { d } } { N } .\tag{35}
$$

Proof. Write $K ( v ) : = N ( v )$ in the full run, $k ( v )$ for the number of simulations $m \in S$ whose path passes through $v ,$ and $\Delta ( v ) : = | \hat { V } ( v ) - \hat { V } ^ { ( - S ) } ( v ) |$ , where $\hat { V } ^ { ( - S ) }$ is the DP value in the reduced run. If $k ( v ) = K ( v )$ , node v is absent from the reduced run, and we set $\hat { V } ^ { ( - S ) } ( v ) : = 0$ . By Corollary 1, $\Delta ( v ) = 0$ unless $k ( v ) \geq 1$ . Define the weighted influence $I ( v ) : = ( K ( v ) - k ( v ) ) \Delta ( v )$ and the sequence

$$
y _ { d } : = 1 , \qquad y _ { \ell } : = \rho | { \cal { A } } | ( 1 + \gamma y _ { \ell + 1 } ) \quad ( 1 \leq \ell \leq d - 1 ) .
$$

We show by backward induction on depth that every node v at depth $\ell \in [ 1 , d ]$ satisfies

$$
I ( v ) \leq 2 V _ { \mathrm { m a x } } y _ { \ell } k ( v ) .\tag{36}
$$

Leaves $( \ell = d ) . { \mathrm { ~ I f ~ } } k ( v ) = K ( v )$ , then $I ( v ) = 0$ . Otherwise, for each $a ^ { \prime } ,$ removing $k ( v )$ of the $K ( v )$ states in $B ( v )$ moves the average of $Q _ { \theta } ^ { \star , S } ( \cdot , a ^ { \prime } )$ by

$$
\frac { k ( v ) } { K ( v ) } \left| \mathrm { m e a n ~ o v e r ~ r e m o v e d ~ s t a t e s - m e a n ~ o v e r ~ r e m a i n i n g ~ s t a t e s } \right| \leq \frac { 2 V _ { \operatorname* { m a x } } k ( v ) } { K ( v ) } .
$$

The maximum over $a ^ { \prime }$ in equation 15 is 1-Lipschitz in the sup norm, so $\Delta ( v ) \leq 2 V _ { \mathrm { m a x } } k ( v ) / K ( v )$ and $I ( v ) \leq 2 V _ { \mathrm { m a x } } k ( v )$

Internal nodes $( 1 \leq \ell \leq d - 1 )$ . Fix $a ^ { \prime } .$ , and write $K ^ { \prime } : = N ( v , a ^ { \prime } )$ and $k ^ { \prime }$ for the number of simulations in S that select $a ^ { \prime }$ at v. If $k ^ { \prime } = 0$ , then $\hat { Q } ( v , a ^ { \prime } )$ is unchanged, since no simulation of S passes through a child of v under $a ^ { \prime }$ . If $1 \leq k ^ { \prime } < K ^ { \prime }$ , let Σ and $\Sigma ^ { \prime }$ be the sums of $r ( s , a ^ { \prime } ) + \gamma \hat { V } ( \mathrm { c h i l d } )$ over the simulations that select $a ^ { \prime }$ at v in the full and in the reduced run. Each removed simulation contributes a term of absolute value at most $V _ { \mathrm { m a x } }$ , because $| r | \leq ( 1 - \gamma ) V _ { \mathrm { m a x } }$ and $| \hat { V } | \le V _ { \mathrm { m a x } }$ . Each of the $K ( w ) - k ( w )$ remaining simulations entering a child w changes by $\gamma \Delta ( w )$ . Hence

$$
\left| \Sigma - \Sigma ^ { \prime } \right| \le k ^ { \prime } V _ { \mathrm { m a x } } + \gamma \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } I ( w ) .
$$

Since $\hat { Q } ( v , a ^ { \prime } ) = \Sigma / K ^ { \prime }$ , the reduced-run value is $\Sigma ^ { \prime } / ( K ^ { \prime } - k ^ { \prime } )$ , and $| \Sigma ^ { \prime } | \le ( K ^ { \prime } - k ^ { \prime } ) V _ { \mathrm { m a x } }$

$$
\Big | \frac { \Sigma } { K ^ { \prime } } - \frac { \Sigma ^ { \prime } } { K ^ { \prime } - k ^ { \prime } } \Big | \le \frac { | \Sigma - \Sigma ^ { \prime } | } { K ^ { \prime } } + \frac { k ^ { \prime } | \Sigma ^ { \prime } | } { K ^ { \prime } ( K ^ { \prime } - k ^ { \prime } ) } \le \frac { 1 } { K ^ { \prime } } \Big ( 2 k ^ { \prime } V _ { \mathrm { m a x } } + \gamma \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } I ( w ) \Big ) .
$$

If $k ^ { \prime } = K ^ { \prime }$ , action $a ^ { \prime }$ is unvisited in the reduced run and drops out of the maximum in equation 17. A maximum of values in $[ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ]$ changes by at most the largest change of its retained arguments, or by at most $2 V _ { \mathrm { m a x } } = 2 k ^ { \prime } V _ { \mathrm { m a x } } / K ^ { \prime }$ if its maximizer drops out. Writing $k _ { a ^ { \prime } } ^ { \prime }$ for the $k ^ { \prime }$ of action $a ^ { \prime } ,$ , in every case

$$
\Delta ( v ) \leq \operatorname* { m a x } _ { a ^ { \prime } \in A } \frac { 1 } { N ( v , a ^ { \prime } ) } \Bigl ( 2 k _ { a ^ { \prime } } ^ { \prime } V _ { \operatorname* { m a x } } + \gamma \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } I ( w ) \Bigr ) .
$$

Multiplying by $K ( v )$ , using equation 27, and bounding the maximum by the sum,

$$
I ( v ) \leq K ( v ) \Delta ( v ) \leq \rho | A | \sum _ { a ^ { \prime } \in \mathcal { A } } \Bigl ( 2 k _ { a ^ { \prime } } ^ { \prime } V _ { \operatorname* { m a x } } + \gamma \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } I ( w ) \Bigr ) .
$$

By the induction hypothesis equation 36 at depth $\ell + 1$ and $\begin{array} { r } { \sum _ { a ^ { \prime } } k _ { a ^ { \prime } } ^ { \prime } = \sum _ { a ^ { \prime } } \sum _ { w \in \mathrm { C h } ( v , a ^ { \prime } ) } k ( w ) = k ( v ) } \end{array}$

$$
\begin{array} { r } { I ( v ) \leq \rho | A | \left( 2 V _ { \operatorname* { m a x } } + 2 \gamma V _ { \operatorname* { m a x } } y _ { \ell + 1 } \right) k ( v ) = 2 V _ { \operatorname* { m a x } } y _ { \ell } k ( v ) . } \end{array}
$$

Root. Each simulation enters exactly one depth-1 node, and the normalizer $1 / N$ is the same in both runs, so

$$
\left| f _ { a } - f _ { a } ^ { ( - S ) } \right| \leq \frac { 1 } { N } \sum _ { m \in S } \left| r ( s _ { j _ { m } } , a ) + \gamma \hat { V } ( v _ { m , 1 } ) \right| + \frac { \gamma } { N } \sum _ { w : \mathrm { d e p t h } ~ 1 } I ( w ) \leq \frac { | S | } { N } V _ { \operatorname* { m a x } } \left( 1 + 2 \gamma y _ { 1 } \right) .
$$

Unrolling the recursion,

$$
y _ { 1 } = \rho | \boldsymbol { A } | \sum _ { i = 0 } ^ { d - 2 } ( \rho \gamma | \boldsymbol { A } | ) ^ { i } + ( \rho \gamma | \boldsymbol { A } | ) ^ { d - 1 } .
$$

By equation 28 and $( \rho \gamma | \mathcal { A } | ) ^ { d - 1 } \leq \Lambda _ { d }$

$$
1 + 2 \gamma y _ { 1 } = 2 \Lambda _ { d } - 1 + 2 \gamma ( \rho \gamma | \boldsymbol { A } | ) ^ { d - 1 } \leq 2 ( 1 + \gamma ) \Lambda _ { d } .
$$

This proves equation 34. For the second claim, let S be the set of coordinates where $\xi$ and $\xi ^ { \prime }$ difer. Both runs have the same reduced run on $( \xi _ { m } ) _ { m \notin S }$ , so equation 34 applied to each and the triangle inequality give $| f _ { a } ( \xi ) - f _ { a } ( \xi ^ { \prime } ) | \le 2 k \cdot 2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } ^ { \prime } \Lambda _ { d } / N = k c$ □

Lemma 7 (Fluctuation up to a bad event). Let $p : = \operatorname* { P r } ( \mathcal { E } _ { \rho } ^ { c } \mid b _ { t } ^ { J } )$ for the tree $T _ { a } ,$ , and let $B _ { d } : =$ $2 V _ { \mathrm { m a x } } + c N$ with c from equation 35. For every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta - p ,$

$$
\left| f _ { a } - \mathbb { E } f _ { a } \right| \leq 2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } \Lambda _ { d } \sqrt { \frac { 2 \log ( 2 / \delta ) } { N } } + B _ { d } p ,
$$

and for every $q \in [ - V _ { \operatorname* { m a x } } , V _ { \operatorname* { m a x } } ]$ 7

$$
\mathbb { E } \big [ ( f _ { a } - q ) ^ { 2 } \big ] \leq \frac { 8 ( 1 + \gamma ) ^ { 2 } V _ { \operatorname* { m a x } } ^ { 2 } \Lambda _ { d } ^ { 2 } } { N } + 2 ( \mathbb { E } f _ { a } - q ) ^ { 2 } + 3 B _ { d } ^ { 2 } p .
$$

Proof. Let $G$ be the set of seed vectors whose run lies in $\mathcal { E } _ { \rho }$ . If $\operatorname* { P r } ( G ) = 0$ , both claims are trivial, so assume otherwise. Let H denote the Hamming distance and define the extension

$$
g ( \xi ) : = \operatorname* { i n f } _ { \eta \in G } \bigl \{ f _ { a } ( \eta ) + c H ( \xi , \eta ) \bigr \} .
$$

First, $g = f _ { a }$ on $G \mathrm { : }$ for $\xi \in G ,$ , Lemma 6 gives $f _ { a } ( \eta ) + c H ( \xi , \eta ) \ge f _ { a } ( \xi )$ for every $\eta \in G$ , with equality at $\eta = \xi$ . Second, g changes by at most c when one coordinate of ξ changes, because each $H ( \cdot , \eta )$ does. Third, $- V _ { \operatorname* { m a x } } \leq g \leq V _ { \operatorname* { m a x } } + c N$ , so $| f _ { a } - g | \le B _ { d }$ everywhere and

$$
| \mathbb { E } f _ { a } - \mathbb { E } g | \leq \mathbb { E } \big [ | f _ { a } - g | \mathbf { 1 } _ { G ^ { c } } \big ] \leq B _ { d } p .
$$

The seeds are independent, so McDiarmid’s bounded-diferences inequality applies to $g \colon$

$$
\operatorname* { P r } \ \Bigl ( | g - \mathbb { E } g | \ge s \Bigr ) \le 2 \exp \biggl ( - \frac { 2 s ^ { 2 } } { N c ^ { 2 } } \biggr ) .
$$

Setting the right-hand side to $\delta$ gives $s = c \sqrt { N \log ( 2 / \delta ) / 2 } = 2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } \Lambda _ { d } \sqrt { 2 \log ( 2 / \delta ) / N }$ . On $G \cap \{ | g - \mathbb { E } g | < s \}$ , an event of probability at least $1 - \delta - p ,$

$$
\left| f _ { a } - \mathbb { E } f _ { a } \right| \le \left| g - \mathbb { E } g \right| + \left| \mathbb { E } g - \mathbb { E } f _ { a } \right| < s + B _ { d } p .
$$

For the second moment, the Efron–Stein inequality gives $\mathrm { V a r } ( g ) \le N c ^ { 2 } / 2 = 8 ( 1 + \gamma ) ^ { 2 } V _ { \mathrm { m a x } } ^ { 2 } \Lambda _ { d } ^ { 2 } / N$ Since $f _ { a } = g$ on $G$ and $| f _ { a } - q | \le 2 V _ { \mathrm { m a x } }$ 2

$$
\begin{array} { r } { \mathbb { E } \big [ ( f _ { a } - q ) ^ { 2 } \big ] \leq \mathbb { E } \big [ ( g - q ) ^ { 2 } \big ] + 4 V _ { \operatorname* { m a x } } ^ { 2 } p = \mathrm { V a r } ( g ) + ( \mathbb { E } g - q ) ^ { 2 } + 4 V _ { \operatorname* { m a x } } ^ { 2 } p . } \end{array}
$$

Finally, $( \mathbb { E } g - q ) ^ { 2 } \leq 2 ( \mathbb { E } f _ { a } - q ) ^ { 2 } + 2 B _ { d } ^ { 2 } p ^ { 2 }$ , and $2 B _ { d } ^ { 2 } p ^ { 2 } + 4 V _ { \mathrm { m a x } } ^ { 2 } p \leq 3 B _ { d } ^ { 2 } p$ . The infimum defining g is universally measurable, which sufices for both inequalities. □

Lemma 8 (Balancedness under uniform selection). $I f d \leq 1$ , then $\mathcal { E } _ { \rho }$ holds surely. $I f d \geq 2 , q _ { \mathrm { m i n } } > 0$ and $| { \mathcal { A } } | M \geq 2$ , then

$$
\operatorname* { P r } ( \mathcal { E } _ { 2 } ^ { c } \mid \boldsymbol { b } _ { t } ^ { J } ) \leq 2 | \boldsymbol { \mathcal { A } } | ( | \boldsymbol { \mathcal { A } } | M ) ^ { d } \exp \biggl ( - \frac { N } { 3 2 } \Bigl ( \frac { q _ { \mathrm { m i n } } } { | \boldsymbol { \mathcal { A } } | } \Bigr ) ^ { d - 1 } \biggr ) .
$$

Proof. For $d \leq 1$ the tree has no internal node. Let $d \geq 2$ and fix a potential internal node v of $T _ { a }$ at depth $\ell \in [ 1 , d - 1 ]$ , i.e., a history with ℓ binned observations and $\ell - 1$ internal actions. By Lemma 5, the indicators that simulation m reaches v, and that it reaches v and selects $a ^ { \prime }$ there, are functions of $\xi _ { m }$ . They are therefore i.i.d. Bernoulli across $m ,$ , with parameters $p _ { v }$ and $p _ { v } / | \mathcal { A } |$ Each binned observation along the history has conditional probability at least $q _ { \mathrm { m i n } }$ and each internal action has probability $1 / | A |$ , so

$$
\frac { p _ { v } } { \left| \boldsymbol { A } \right| } \geq \Big ( \frac { q _ { \mathrm { m i n } } } { \left| \boldsymbol { A } \right| } \Big ) ^ { \ell } \geq \pi : = \Big ( \frac { q _ { \mathrm { m i n } } } { \left| \boldsymbol { A } \right| } \Big ) ^ { d - 1 } .
$$

The multiplicative Chernof bounds give

$$
\begin{array} { r l } & { \operatorname* { P r } \bigl ( N ( v ) \geq \frac { 3 } { 2 } N p _ { v } \bigr ) \leq e ^ { - N p _ { v } / 1 2 } \leq e ^ { - N \pi / 3 2 } , } \\ & { \operatorname* { P r } \bigl ( N ( v , a ^ { \prime } ) \leq \frac { 3 } { 4 } N p _ { v } / | A | \bigr ) \leq e ^ { - N p _ { v } / ( 3 2 | A | ) } \leq e ^ { - N \pi / 3 2 } . } \end{array}
$$

Outside these events, $N ( v ) < \frac { 3 } { 2 } N p _ { v } = 2 | A | \cdot \frac { 3 } { 4 } N p _ { v } / | A | < 2 | A | N ( v , a ^ { \prime } )$ , which is equation 27 with $\rho = 2$ . A union bound over the $| { \cal A } |$ actions and over the at most $\begin{array} { r } { \sum _ { \ell = 1 } ^ { d - 1 } M ^ { \ell } | \mathcal { A } | ^ { \ell - 1 } \leq ( | \mathcal { A } | M ) ^ { d } } \end{array}$ potential internal nodes gives the claim, since $1 + | { \mathcal { A } } | \leq 2 | { \mathcal { A } } |$ □

Proof of Proposition 3. When $d = 0$ the planner is exact, so let $d \geq 1$ and write $q _ { a } : = Q _ { \theta , d } ( b ^ { J } , a ) \in$ $[ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ]$

Step 1: Fluctuation. Apply Lemma 7 to each root action with $\delta / | { \mathcal { A } } |$ in place of $\delta ,$ and take a union bound over $a \in { \mathcal { A } }$ . With probability at least $1 - \delta - | \mathcal { A } | p _ { \rho } ,$ every a satisfies

$$
\begin{array} { r l } & { | \widehat { Q } ( { b } ^ { J } , a ) - q _ { a } | \leq | \widehat { Q } ( { b } ^ { J } , a ) - \mathbb { E } \widehat { Q } ( { b } ^ { J } , a ) | + | \mathbb { E } \widehat { Q } ( { b } ^ { J } , a ) - q _ { a } | } \\ & { \qquad \leq 2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } \Lambda _ { d } \sqrt { \frac { 2 \log \left( 2 | A | / \delta \right) } { N } } + B _ { d } p _ { \rho } + \mathrm { b i a s } ( N , d ) , } \end{array}
$$

and taking the supremum over a gives equation 30. For the second moment, Lemma $7$ with $q = q _ { a }$ gives

$$
\mathbb { E } \big [ ( \widehat { Q } ( b ^ { J } , a ) - q _ { a } ) ^ { 2 } \big ] \leq \frac { 8 ( 1 + \gamma ) ^ { 2 } V _ { \operatorname* { m a x } } ^ { 2 } \Lambda _ { d } ^ { 2 } } { N } + 2 \operatorname { b i a s } ( N , d ) ^ { 2 } + 3 B _ { d } ^ { 2 } p _ { \rho } .
$$

Summing over $a \in { \mathcal { A } }$ through $\begin{array} { r } { \epsilon _ { \mathrm { m c } } ( t , d ) ^ { 2 } \leq \sum _ { a } ( \widehat { Q } ( b ^ { J } , a ) - q _ { a } ) ^ { 2 } } \end{array}$ gives equation 31. The bounds on $p _ { \rho }$ are Lemma $8 .$

Step 2: DP backup estimation error. The simulations follow the uniform policy at internal nodes, and the DP backup ${ \hat { V } } ( v ) = \operatorname* { m a x } _ { a ^ { \prime } } { \hat { Q } } ( v , a ^ { \prime } )$ recovers the optimal internal actions. We bound the bias ${ \mathrm { b i a s } } _ { \mathrm { D P } } ( N , d ) : = \operatorname* { s u p } _ { a } | \mathbb { E } \widehat { Q } ( b ^ { J } , a ) - Q _ { \theta , d } ^ { \phi } ( b ^ { J } , a ) |$ of the resulting estimate by backward induction on depth, at the level of orders of magnitude, without any independence assumption on the returns. Fix a node v at depth $\ell \in [ 1 , d ]$ with $n _ { v } : = N ( v )$ visits, and condition on the set of simulations that reach v and on their states there. By Lemma $5 ,$ the continuation of each such simulation below v is driven by the components of its own seed at depths $\geq \ell ,$ which are independent across simulations and independent of the conditioning. Conditionally, the states in $B ( v )$ are therefore independent draws from the binned belief $b _ { v }$ at v, and each $\hat { Q } ( v , a ^ { \prime } )$ is formed from $n _ { v , a ^ { \prime } } : = N ( v , a ^ { \prime } )$ independent continuations, with $n _ { v , a ^ { \prime } } \geq n _ { v } / ( \rho | \mathcal { A } | )$ on $\mathcal { E } _ { \rho }$ . Concentration of these conditional averages gives

$$
\left| \hat { Q } ( v , a ^ { \prime } ) - Q _ { \theta , d - \ell } ^ { \phi } ( b _ { v } , a ^ { \prime } ) \right| \ \leq \ O \biggl ( \frac { V _ { \mathrm { m a x } } } { \sqrt { n _ { v , a ^ { \prime } } } } \biggr ) \ \leq \ O \biggl ( \frac { V _ { \mathrm { m a x } } \sqrt { | \mathcal { A } | } } { \sqrt { n _ { v } } } \biggr ) .
$$

At a leaf $( \ell = d )$ the same bound holds for the pooled averages $\begin{array} { r } { \frac { 1 } { \vert B ( v ) \vert } \sum _ { s \in B ( v ) } Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } ) } \end{array}$ around $\begin{array} { r } { \int Q _ { \theta } ^ { \star , S } ( s , a ^ { \prime } ) b _ { v } ( \mathrm { d } s ) } \end{array}$ . This level must be included, because the maximum over $a ^ { \prime }$ in equation 15 is biased upward at every finite $| B ( v ) |$ . Under the branching structure, a depth-ℓ node receives $n _ { v } ~ = ~ \Omega ( N / ( | \mathcal { A } | M ) ^ { \ell } )$ visits in expectation when the bins have comparable probabilities. Taking the maximum over estimated Q-values, the backup error at depth ℓ contributes at most $O ( V _ { \operatorname* { m a x } } \sqrt { | \mathcal { A } | ^ { \ell + 1 } M ^ { \ell } / N } )$ , discounted by $\gamma ^ { \ell }$ . The complement of $\mathcal { E } _ { \rho }$ changes the conditional mean by at most $2 V _ { \mathrm { m a x } } p _ { \rho } ,$ , which is absorbed into the constant once $p _ { \rho } \leq \dot { N } ^ { - 1 / 2 }$ (Remark 3). Summing over depths, leaves included,

$$
\mathrm { b i a s } _ { \mathrm { D P } } ( N , d ) \leq \sum _ { \ell = 1 } ^ { d } \gamma ^ { \ell } \cdot \frac { c _ { 1 } V _ { \mathrm { m a x } } \sqrt { | A | } } { \sqrt { N / ( | A | M ) ^ { \ell } } } = \frac { c _ { 0 } \Phi _ { d } V _ { \mathrm { m a x } } \sqrt { | A | } } { \sqrt { N } } .\tag{37}
$$

Step 3: Discretization error. For $o , o ^ { \prime }$ in the same bin, the Bayes update sensitivity [10, Section 4.3] gives

$$
\Vert \tau _ { \theta } ( b , a , o ) - \tau _ { \theta } ( b , a , o ^ { \prime } ) \Vert _ { \mathrm { T V } } \leq \frac { g _ { \operatorname* { m a x } } } { g _ { \operatorname* { m i n } } } \omega _ { Z } ( \mathrm { d i a m } ( \phi ) ) .
$$

Since $\tau _ { \theta } ^ { \phi }$ is a convex combination of $\{ \tau _ { \theta } ( b , a , o ^ { \prime } ) \} _ { o ^ { \prime } \in \phi ^ { - 1 } ( \bar { o } ) }$ , the same bound holds for $\| \tau _ { \theta } ^ { \phi } ( b , a , \bar { o } )$ $\tau _ { \theta } ( b , a , o ) \Vert _ { \mathrm { T V } }$ . By Lipschitz continuity of $\bar { V } _ { \theta , d - 1 } ^ { \star , S }$ in TV (constant $2 V _ { \mathrm { m a x } } ,$ , cf. Lemma 2), each of the d tree levels contributes $\begin{array} { r l } { \frac { 2 g _ { \mathrm { m a x } } } { g _ { \mathrm { m i n } } } V _ { \mathrm { m a x } } \omega _ { Z } ( \mathrm { d i a m } ( \phi ) ) } \end{array}$ discounted by γ:

$$
\operatorname* { s u p } _ { b , a } \bigl | Q _ { \theta , d } ^ { \phi } ( b , a ) - Q _ { \theta , d } ( b , a ) \bigr | \leq \frac { 2 \gamma ( 1 - \gamma ^ { d } ) } { 1 - \gamma } \cdot \frac { g _ { \operatorname* { m a x } } } { g _ { \operatorname* { m i n } } } V _ { \operatorname* { m a x } } \omega _ { Z } ( \dim ( \phi ) ) .\tag{38}
$$

Combining Steps 2 and 3 gives bias $( N , d ) \leq \mathrm { b i a s _ { D P } } ( N , d ) + \mathrm { b i a s _ { d i s c } } ( d )$ , yielding equation 32.

Remark 3 (Constants, a verification-free variant, and the choice of depth). (a) Order of the constants. Compared with an analysis that treats the simulation returns as $i . i . d .$ , the only changes are the depth factor $( 1 + \gamma ) \Lambda _ { d }$ in the fluctuation term and the exponentially small remainder in $p _ { \rho }$ . Since $\Lambda _ { d } \leq d \operatorname* { m a x } ( 1 , \rho \gamma | \boldsymbol { \mathcal { A } } | ) ^ { d - 1 }$ and $\Phi _ { d } \geq ( \gamma \sqrt { | { \cal A } | M } ) ^ { d }$ , the fluctuation term of equation 31 is at most of the same exponential-in-d order as the DP-backup term whenever $M = \Omega ( \rho ^ { 2 } | \mathcal { A } | )$ . Taking square roots in equation 31, with diam(ϕ) chosen as in Remark $^ { 4 , }$ then gives $\mathbb { E } [ \epsilon _ { \mathrm { m c } } ( t , d ) \mid b _ { t } ^ { J } ] \leq C _ { \mathrm { m c } } V _ { \mathrm { m a x } } / \sqrt { N }$ with $C _ { \mathrm { m c } }$ of the order stated in Theorem 1, whose order is therefore unchanged. For coarse binning, $M \ll | { \mathcal { A } } |$ , the fluctuation dominates and $C _ { \mathrm { m c } } \propto \sqrt { | \mathcal { A } | } \Lambda _ { d }$ . By Lemma $\delta ,$ the remainder $3 | \mathcal { A } | B _ { d } ^ { 2 } p _ { 2 }$ is at most $V _ { \operatorname* { m a x } } ^ { 2 } / N$ as soon as $N \geq 3 2 ( | A | / q _ { \operatorname* { m i n } } ) ^ { d - 1 } { \overset { \cdot } { \log } } { \big ( } 2 4 N | A | ^ { 2 } ( | A | M ) ^ { d } ( 1 + 4 \Lambda _ { d } ) ^ { 2 } { \big ) }$ , and it vanishes for $d \leq 1$ . At $d = 0 , \Lambda _ { 0 } = 0 $ and the fluctuation vanishes identically. For an internal rule satisfying $( S 1 ) \mathrm { - } ( S \mathcal { Z } )$ in place of uniform selection, the same proof applies with |A| replaced by $1 / \pi _ { \operatorname* { m i n } }$ in equation 27, equation 28 and Lemma 8.

(b) A verification-free variant. Sample splitting restores exactly conditionally i.i.d. returns. Use $N / 2$ seeds to build the tree, populate the particle sets, and run the DP backup, then freeze the leaf values and the greedy internal actions arg m $\operatorname { \ u x } _ { a ^ { \prime } } \hat { Q } ( v , a ^ { \prime } )$ . Value each root action with N/2 fresh seeds that follow the frozen greedy actions and stop at the frozen leaf values; a simulation that leaves the frozen tree stops and is evaluated by ma $\mathrm { x } _ { a ^ { \prime } } \bar { Q } _ { \theta } ^ { \star , S } ( s , \bar { a ^ { \prime } } )$ at its current state s. Conditionally on the first half, the second-half returns are i.i.d. and bounded by $V _ { \mathrm { m a x } } .$ , so Hoefding’s inequality applies to them directly, at the cost of a factor 2 in N, and the bias analysis of Step 2 applies to the first half. Evaluating each leaf at its single incoming state instead of equation 15 removes the pooling but not the coupling through the maximum in the DP backup, so it restores independent returns only at $d = 1$ , where it also changes the leaf target to the linear surrogate $\widetilde { V } _ { \boldsymbol { \theta } } ^ { \star }$

(c) Choice of depth. Propositions 2 and 3 concern diferent objects. The former bounds the planning target, $\lVert \bar { V } _ { \theta , d } ^ { \star , S } - \bar { V } _ { \theta } ^ { \star , B } \rVert _ { \infty } \leq \gamma ^ { d + 1 } \Xi _ { \theta } / ( 1 - \gamma )$ , which involves no particles and decays geometrically in d. The latter bounds the estimator $\widehat { Q }$ around $Q _ { \theta , d }$ , and its fluctuation $2 ( 1 + \gamma ) V _ { \operatorname* { m a x } } \Lambda _ { d } \sqrt { 2 \log ( 2 | \boldsymbol { A } | / \delta ) / N }$ and DP-backup bias $c _ { 0 } \Phi _ { d } V _ { \operatorname* { m a x } } \sqrt { | \mathcal { A } | / N }$ grow geometrically in d. Because increasing the depth reduces the structural bias but increases the Monte Carlo error, their sum is U-shaped in $d ,$ and each budget N has a finite optimal depth.

Remark 4 (Optimal discretization resolution). The discretization error in equation 32 is a fixed bias $O ( \omega _ { Z } ( \mathrm { d i a m } ( \phi ) ) )$ , while the DP backup estimation error vanishes as $N \to \infty$ at rate $\Phi _ { d } / \sqrt { N }$ . To make the discretization bias $O ( 1 / \sqrt { N } )$ , set diam $( \phi ) = \omega _ { Z } ^ { - 1 } \big ( ( 1 - \gamma ) g _ { \operatorname* { m i n } } / ( 2 \gamma g _ { \operatorname* { m a x } } V _ { \operatorname* { m a x } } \sqrt { N } ) \big )$ . When $Z _ { \theta } ( o \mid s )$ is $L _ { Z ^ { - } } L i p s c h i t z$ in o uniformly (so $\omega _ { Z } ( \delta ) \leq L _ { Z } \delta / g _ { \mathrm { m i n } } )$ and $\mathcal { O } \subseteq \mathbb { R } ^ { d _ { o } }$ with a uniform grid:

$$
M = \left\lceil \left( \frac { 2 \gamma L _ { Z } g _ { \mathrm { m a x } } V _ { \mathrm { m a x } } \mathrm { d i a m } ( \mathcal { O } ) \sqrt { N } } { \left( 1 - \gamma \right) g _ { \mathrm { m i n } } ^ { 2 } } \right) ^ { d _ { o } } \right\rceil .\tag{39}
$$

With this choice, M grows as $N ^ { d _ { o } / 2 } ;$ , and the simulation budget threshold for the DP backup error to be $O ( 1 / \sqrt { N } )$ is $N \geq c _ { 0 } ^ { 2 } | \mathcal { A } | \Phi _ { d } ^ { 2 }$ , which is exponential in $d \cdot d _ { o }$ via the dependence of $\Phi _ { d }$ on M.

Balancing the two bias terms instead gives an explicit rate. Take $\mathcal { O } \subseteq [ 0 , D ] ^ { d _ { o } }$ with a cubic grid of mesh h, so that $M = ( D / h ) ^ { d _ { o } }$ and d $\mathrm { i a m } ( \phi ) = \sqrt { d _ { o } } h$ , and assume $\omega _ { Z } ( \delta ) \le L _ { Z } \delta / g _ { \mathrm { m i n } }$ as above. For $\gamma \sqrt { | \mathcal { A } | M } \ge 2$ we have $\Phi _ { d } \leq 2 ( \gamma \sqrt { | { \cal A } | M } ) ^ { d } ,$ so the DP-backup and discretization terms of equation 32 are of order h $^ { - d d _ { o } / 2 } N ^ { - 1 / 2 }$ and h, respectively, with constants depending on $d , \gamma , | A |$ $D , L _ { Z } , g _ { \mathrm { { m a x } } } / g _ { \mathrm { { m i n } } }$ and $V _ { \mathrm { m a x } }$ . Balancing them gives

$$
\begin{array} { r } { \boldsymbol { h } ^ { \star } \times \boldsymbol { N } ^ { - 1 / ( d d _ { o } + 2 ) } , \qquad \epsilon _ { \mathrm { m c } } ( t , d ) = \widetilde { O } \big ( N ^ { - 1 / ( d d _ { o } + 2 ) } \big ) , } \end{array}\tag{40}
$$

since the fluctuation term of equation 30 does not depend on h and is of lower order. The $N ^ { - 1 / 2 }$ rate is recovered exactly when d $d _ { o } = 0$ , i.e., at $d = 0$ $A t d = 1$ , equation 40 is the familiar rate $N ^ { - 1 / ( d _ { o } + 2 ) }$ of a histogram estimate, here of the leaf value as a function of the observation. The balancedness condition of Lemma 8 holds at this mesh for large N: with the Lebesgue reference measure $q _ { \mathrm { m i n } } \geq g _ { \mathrm { m i n } } h ^ { d _ { o } }$ , so $N ( q _ { \mathrm { m i n } } / | \mathcal { A } | ) ^ { d - 1 } = \Omega ( N ^ { ( d _ { o } + 2 ) / ( d d _ { o } + 2 ) } ) \to \infty$ . The computational cost is $N | { \mathcal { A } } |$ simulations over at most $( | { \cal { A } } | { \cal { M } } ) ^ { d }$ nodes.

When $d = 0 \ o r \ | O | < \infty$ , this issue does not arise: set $\phi = \mathrm { i d }$ and $M = | \mathcal { O } |$

## D.3 Particle Filter Stability

We work under Assumption 1 throughout. Write $b _ { t } : = b _ { t } ^ { \theta ^ { \star } }$ for the true posterior and $b _ { t } ^ { J } : = b _ { t } ^ { J , ( \bar { \theta } _ { k ( s ) } ) _ { s \leq t } }$ for the particle belief filtered under the epoch-frozen model sequence. Since the model sequence is determined by the algorithm, we abbreviate the latter as $b _ { t } ^ { J }$

Proposition 4 (Particle filter stability in $d _ { Q } )$ . Under Assumption 1, there exist constants $C _ { \mathrm { p f } } , C _ { \mathrm { f i l t } } <$ ∞ and a forgetting rate $\rho : = 1 - \varepsilon _ { \mathrm { { m i x } } } \left( g _ { \mathrm { { m i n } } } / g _ { \mathrm { { m a x } } } \right) \in \left( 0 , 1 \right)$ such that for all $t \geq 1$

$$
\mathbb { E } \big [ d _ { Q } ( b _ { t } , b _ { t } ^ { J } ) \big ] \ \leq \ \underbrace { C _ { \mathrm { p f } } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p f } } } } } _ { p a r t i c l e a p p r o x . } + \underbrace { 2 V _ { \mathrm { m a x } } C _ { \mathrm { f l i t } } \sum _ { s = 1 } ^ { t } \rho ^ { t - s } d _ { \mathrm { m o d e l } } ^ { a _ { s } } ( \bar { \theta } _ { s } , \theta ^ { \star } ) } _ { f l i t e r \ m i s m a t c h } ,\tag{41}
$$

where $\begin{array} { r } { d _ { \mathrm { m o d e l } } ^ { a } ( \theta , \theta ^ { \prime } ) : = \operatorname* { s u p } _ { s ^ { \prime } } \| P _ { \theta } ( \cdot \mid s ^ { \prime } , a ) - P _ { \theta ^ { \prime } } ( \cdot \mid s ^ { \prime } , a ) \| _ { \mathrm { T V } } + \operatorname* { s u p } _ { s ^ { \prime } , o } | Z _ { \theta } ( o \mid s ^ { \prime } ) / Z _ { \theta ^ { \prime } } ( o \mid s ^ { \prime } ) - 1 | } \end{array}$ $C _ { \mathrm { f i l t } } : = g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } }$ , and $\kappa _ { \mathrm { p f } }$ is the particle filter exponent:

$$
\kappa _ { \mathrm { p f } } : = \operatorname* { m a x } \left( 1 , \ : \frac { \log ( 2 g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } ) } { \log ( 1 / \gamma ) } \right) .\tag{42}
$$

The truncation depth $H _ { \mathrm { p f } }$ in Lemma 1 is chosen to balance the sum and tail:

$$
H _ { \mathrm { p f } } : = \left\lceil \frac { \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) } { \kappa _ { \mathrm { p f } } \log ( 1 / \gamma ) } \right\rceil = \left\lceil \frac { \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) } { \log \operatorname* { m a x } ( 1 / \gamma , \ 2 g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } ) } \right\rceil ,\tag{43}
$$

and $C _ { \mathrm { p f } }$ depends polynomially on V<sub>max</sub>, $g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } } , 1 / \varepsilon _ { \mathrm { m i x } }$ , log $\mathcal { N } _ { \Theta } ( 1 / T )$ , and $H _ { \mathrm { p f } }$ . When $\kappa _ { \mathrm { p f } } = 1$ $( i . e . , \ : \dot { 2 } \gamma g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } \le 1 )$ , the rate recovers $J ^ { - 1 / 2 }$ . This $H _ { \mathrm { p f } }$ is distinct from the trajectory truncation horizon H in Section $D . 4$

Proof of Proposition 4. By the triangle inequality and Lemma 2:

$$
\begin{array} { r l r } { d _ { Q } ( b _ { t } ^ { \theta ^ { \star } } , b _ { t } ^ { J , \bar { \theta } _ { 1 : t } } ) } & { \le } & { \underbrace { d _ { Q } ( b _ { t } ^ { \theta ^ { \star } } , b _ { t } ^ { \bar { \theta } _ { 1 : t } } ) } _ { \mathrm { ( I ) ~ f l i t e r ~ m i s m a t c h } } + \underbrace { d _ { Q } ( b _ { t } ^ { \bar { \theta } _ { 1 : t } } , b _ { t } ^ { J , \bar { \theta } _ { 1 : t } } ) } _ { \mathrm { ( I I ) ~ p a r t i c l e ~ a p p r o x . } } . } \end{array}\tag{44}
$$

Term (I) involves two exact (continuous) posteriors, so the TV bound of Lemma 2 gives $d _ { Q } \le$ $2 V _ { \operatorname* { m a x } } \| b _ { t } ^ { \theta ^ { \star } } - b _ { t } ^ { \bar { \theta } _ { 1 : t } } \| _ { \mathrm { T V } }$ , which is bounded by the filter mismatch analysis below.

Term (II): test-function bound via Feynman-Kac. Conditional on $( a _ { 1 : T } , o _ { 1 : T } )$ , the particle filter is a Feynman-Kac system with kernels $M _ { s } ( \cdot \mid s ^ { \prime } ) : = P _ { \bar { \theta } _ { s } } ( \cdot \mid s ^ { \prime } , a _ { s } )$ and potentials $G _ { s } ( s ^ { \prime } ) : = $ $Z _ { \bar { \theta } _ { s } } \left( o _ { s } \mid s ^ { \prime } \right)$ . Under Assumption 1, the kernels satisfy $M _ { s } ( s ^ { \prime } \mid x ) \ge \varepsilon _ { \operatorname* { m i x } } \nu ( s ^ { \prime } )$ and the potentials satisfy $g _ { \mathrm { m i n } } \le G _ { s } ( x ) \le g _ { \mathrm { m a x } }$ uniformly. We invoke the following time-uniform particle-filter bound.

Lemma 9 ([21, Lemma 2.3]; see also [9, Theorem $7 . 4 . 4 ]$ . Let $( \eta _ { n } ) _ { n \ge 0 }$ and $( \eta _ { n } ^ { N } ) _ { n \geq 0 }$ be the exact and N-particle Feynman-Kac measures on a measurable space E. Suppose the Markov kernels $( M _ { \ell } ) _ { \ell \ge 1 }$ admit densities w.r.t. a common measure λ satisfying $\underline { { M } } \le M _ { \ell } ( x , y ) \le \overline { { M } }$ for all $\ell , x , y$ , and the potentials satisfy $\underline { { G } } \le G _ { \ell } ( x ) \le \overline { { G } }$ for all $\ell , x$ . Then for all $n \geq 0 , N \geq 2 , p \geq 1$ , and φ with os $; ( \varphi ) \le 1$

$$
\left( \mathbb { E } \left[ | \eta _ { n } ^ { N } ( \varphi ) - \eta _ { n } ( \varphi ) | ^ { p } \right] \right) ^ { 1 / p } \leq \frac { 2 d ( p ) ^ { 1 / p } } { \sqrt { N } } \frac { ( \overline { { M } } / \underline { { M } } ) ^ { 3 } \overline { { G } } } { \underline { { G } } } ,\tag{45}
$$

where $d ( p ) ^ { 1 / p } = { \cal O } ( \sqrt { p } )$ . The bound is uniform over n.

Our Assumption 1 implies the conditions of Lemma 9 with $\underline { { M } } = \varepsilon _ { \operatorname* { m i x } } , \overline { { M } } = 1 , \underline { { G } } = g _ { \operatorname* { m i n } }$ ${ \overline { { G } } } = g _ { \mathrm { m a x } } .$

Proposition 5 (Particle filter $L ^ { p }$ bound under Assumption 1). Under Assumption 1, for every measurable $f : { \mathcal { S } }  \mathbb { R }$ with $\| f \| _ { \infty } \leq M$ , all $p \geq 1$ , and all $t \geq 0$

$$
\big ( \mathbb { E } \big [ | \eta _ { t } ^ { J } ( f ) - \eta _ { t } ( f ) | ^ { p } \big ] \big ) ^ { 1 / p } \ \leq \ \frac { 4 d ( p ) ^ { 1 / p } M g _ { \operatorname* { m a x } } } { \varepsilon _ { \operatorname* { m i x } } ^ { 3 } g _ { \operatorname* { m i n } } \sqrt { J } } ,\tag{46}
$$

where $d ( p ) ^ { 1 / p } = { \cal O } ( \sqrt { p } )$ , so $\eta _ { t } ^ { J } ( f ) - \eta _ { t } ( f )$ is sub-Gaussian with parameter $\sigma ^ { 2 } = O ( M ^ { 2 } g _ { \mathrm { m a x } } ^ { 2 } / ( \varepsilon _ { \mathrm { m i x } } ^ { 6 } g _ { \mathrm { m i n } } ^ { 2 } J ) )$ The bound is uniform over t.

Proof. By Lemma 9 with $\underline { { M } } = \varepsilon _ { \mathrm { m i x } } , \overline { { M } } = 1 , \underline { { G } } = g _ { \mathrm { m i n } } , \overline { { G } } = g _ { \mathrm { m a x } }$ , for any φ with $\csc ( \varphi ) \leq 1$

$$
\bigl ( \mathbb { E } \bigl [ | \eta _ { t } ^ { J } ( \varphi ) - \eta _ { t } ( \varphi ) | ^ { p } \bigr ] \bigr ) ^ { 1 / p } \leq \frac { 2 d ( p ) ^ { 1 / p } g _ { \mathrm { m a x } } } { \varepsilon _ { \mathrm { m i x } } ^ { 3 } g _ { \mathrm { m i n } } \sqrt { J } } ,
$$

uniformly over t. For f with $\| f \| _ { \infty } \leq M$ , applying this to $\varphi : = f / ( 2 M )$ (which has $\csc ( \varphi ) \leq 1 )$ and rescaling by 2M yields equation 46. Sub-Gaussianity follows from the standard moment-to-tail conversion. □

Lemma 10 (Unconditional bound under adaptive actions). Let $( a _ { s } , o _ { s } ) _ { s \geq 1 }$ be an adapted actionobservation process where $a _ { s } = a _ { s } ( b _ { s } ^ { J } )$ may depend on the particle system. Under Assumption 1, for every measurable $f : S  \mathbb { R }$ with $\| f \| _ { \infty } \leq M$ , all $p \geq 1$ , and all $t \geq 0 .$

$$
\left( \mathbb { E } \left[ \vert b _ { t } ^ { J , \bar { \theta } _ { 1 : t } } ( f ) - b _ { t } ^ { \bar { \theta } _ { 1 : t } } ( f ) \vert ^ { p } \right] \right) ^ { 1 / p } \leq \frac { 4 d ( p ) ^ { 1 / p } M g _ { \operatorname* { m a x } } } { \varepsilon _ { \operatorname* { m i x } } ^ { 3 } g _ { \operatorname* { m i n } } \sqrt { J } } .\tag{47}
$$

Proof. Let $\mathcal { H } _ { T } : = ( a _ { 1 : T } , o _ { 1 : T } , \bar { \theta } _ { 1 : T } )$ . Conditional on $\mathcal { H } _ { T }$ , the Feynman-Kac kernels and potentials are deterministic, so Proposition 5 gives

$$
\mathbb { E } \big [ | b _ { t } ^ { J } ( f ) - b _ { t } ( f ) | ^ { p } \bigm | \mathcal { H } _ { T } \bigm ] \le \bigg ( \frac { 4 d ( p ) ^ { 1 / p } M g _ { \operatorname* { m a x } } } { \varepsilon _ { \operatorname* { m i x } } ^ { 3 } g _ { \operatorname* { m i n } } \sqrt { J } } \bigg ) ^ { p } = : C _ { p } ^ { p } .
$$

Since $C _ { p }$ is independent of $\mathcal { H } _ { T }$ , the tower property yields $\mathbb { E } [ | b _ { t } ^ { J } ( f ) - b _ { t } ( f ) | ^ { p } ] \leq C _ { p } ^ { p }$

By Lemma 1 with truncation depth $H _ { \mathrm { p f } }$ :

$$
d _ { Q } ( b _ { t } ^ { \bar { \theta } _ { 1 : t } } , b _ { t } ^ { \boldsymbol { J } , \bar { \theta } _ { 1 : t } } ) \ \leq \ \operatorname* { s u p } _ { \theta , a } \ \sum _ { \ell = 0 } ^ { H _ { \mathrm { p f } } - 1 } \gamma ^ { \ell } C _ { \mathrm { B U } } ^ { \ell } \operatorname* { s u p } _ { f \in \mathcal { F } _ { \ell } ( \theta , a ) } | b _ { t } ( f ) - b _ { t } ^ { J } ( f ) | + 2 \gamma ^ { H _ { \mathrm { p f } } } V _ { \mathrm { m a x } } .\tag{48}
$$

We optimize $H _ { \mathrm { p f } }$ to balance the sum and tail below.

Pointwise bound. For $f \in { \mathcal { F } } _ { \ell } ( \theta , a )$ with $\| f \| _ { \infty } \leq g _ { \operatorname* { m a x } } ^ { \ell } V _ { \operatorname* { m a x } }$ , Lemma 10 gives

$$
\big ( \mathbb { E } \big [ | b _ { t } ^ { J } ( f ) - b _ { t } ( f ) | ^ { p } \big ] \big ) ^ { 1 / p } \ \leq \ \frac { 4 d ( p ) ^ { 1 / p } g _ { \operatorname* { m a x } } ^ { \ell + 1 } V _ { \operatorname* { m a x } } } { \varepsilon _ { \operatorname* { m i x } } ^ { 3 } g _ { \operatorname* { m i n } } \sqrt { J } } = : \sigma _ { \ell } d ( p ) ^ { 1 / p } .\tag{49}
$$

Since $d ( p ) ^ { 1 / p } \leq C _ { 0 } \sqrt { p }$ (Corollary 5), $b _ { t } ^ { J } ( f ) - b _ { t } ( f )$ is sub-Gaussian with parameter $\sigma _ { \ell } ^ { 2 }$

Covering-number argument. For N sub-Gaussian random variables with common parameter $\sigma ^ { 2 }$

$$
\mathbb { E } \big [ \operatorname* { m a x } _ { i \leq N } | X _ { i } | \big ] \ \leq \ \sigma \sqrt { 2 \log ( 2 N ) } .\tag{50}
$$

At depth $\textstyle \ell , \bigcup _ { \theta , a } { \mathcal { F } } _ { \ell } ( \theta , a )$ is covered by $N _ { \ell } : = \mathcal { N } _ { \Theta } ( 1 / T ) \cdot | \mathbf { \mathcal { A } } | ^ { \ell }$ functions after discretizing Θ at resolution $1 / T$ , with approximation error $O ( 1 / T )$ . Applying equation 50 and summing over depths with weights $\gamma ^ { \ell } C _ { \mathrm { B U } } ^ { \ell } \colon$

$$
{  { \mathbb E } } \big [ d _ { Q } ( b _ { t } ^ {  { \bar { \theta } } _ { 1 : t } } , b _ { t } ^ { J ,  { \bar { \theta } } _ { 1 : t } } ) \big ] \le \sum _ { \ell = 0 } ^ { H _ { \mathrm { p t } } - 1 } \gamma ^ { \ell } C _ { \mathrm { B U } } ^ { \ell } \sigma _ { \ell } \sqrt { 2 \log ( 2 N _ { \ell } ) } + 2 \gamma ^ { H _ { \mathrm { p f } } } V _ { \mathrm { m a x } } + O ( 1 / T )\tag{51}
$$

Define $\alpha : = 2 \gamma g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } , \ r : = \ \alpha / \gamma \ = \ 2 g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 }$ , and $c _ { 1 } : = 4 g _ { \mathrm { m a x } } V _ { \mathrm { m a x } } / ( \varepsilon _ { \mathrm { m i x } } ^ { 3 } g _ { \mathrm { m i n } } )$ , so that $\sigma _ { \ell } = c _ { 1 } g _ { \mathrm { m a x } } ^ { \ell } / \sqrt { J }$ and the ℓ-th term equals

$$
{ \frac { c _ { 1 } } { \sqrt { J } } } \alpha ^ { \ell } \sqrt { 2 \log \left( 2 \Lambda \wp _ { } ( 1 / T ) \left| A \right| ^ { \ell } \right) } .
$$

Since ${ \sqrt { \log ( a b ) } } \leq { \sqrt { \log a } } + { \sqrt { \log b } }$ , the sum in equation 51 is bounded by

$$
\mathbb { E } [ d _ { Q } ^ { ( \Pi ) } ] \ \le \ \frac { c _ { 1 } \sqrt { 2 } } { \sqrt { J } } \left( \sqrt { \log ( 2  { N _ { \Theta } } ( 1 / T ) ) } \ \sum _ { \ell = 0 } ^ { H _ { \mathrm { p f } } - 1 } \alpha ^ { \ell } + \sqrt { \log | \mathscr { A } | } \ \sum _ { \ell = 0 } ^ { H _ { \mathrm { p f } } - 1 } \alpha ^ { \ell } \sqrt { \ell } \right) + 2 \gamma ^ { H _ { \mathrm { p f } } } V _ { \mathrm { m a x } } .\tag{52}
$$

We bound the two sums in two cases.

Case $\ : 1 \colon \alpha \geq 1 \ :$ . Choose

$$
H _ { \mathrm { p f } } = \left\lceil \frac { \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) } { \log r } \right\rceil = \left\lceil \frac { \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) } { \log ( 2 g _ { \mathrm { m a x } } ^ { 2 } / g _ { \mathrm { m i n } } ^ { 2 } ) } \right\rceil .
$$

Using $\textstyle \sum _ { \ell = 0 } ^ { H - 1 } \alpha ^ { \ell } \leq H \alpha ^ { H - 1 }$ and $\begin{array} { r } { \sum _ { \ell = 0 } ^ { H - 1 } \alpha ^ { \ell } \sqrt { \ell } \leq H ^ { 3 / 2 } \alpha ^ { H - 1 } \ ( \mathrm { s i n c e } \ \alpha \geq 1 ) } \end{array}$ :

$$
\mathrm { s u m } \ \leq \ \frac { c _ { 1 } \sqrt 2 H _ { \mathrm { p f } } ^ { 3 / 2 } \alpha ^ { H _ { \mathrm { p f } } - 1 } } { \sqrt J } \cdot \sqrt { \log \left( 2 \mathcal { N } _ { \Theta } ( 1 / T ) \left| \boldsymbol { \mathcal { A } } \right| ^ { H _ { \mathrm { p f } } } \right) } .\tag{53}
$$

Writing $\alpha ^ { H _ { \mathrm { p f } } - 1 } = \alpha ^ { H _ { \mathrm { p f } } } / \alpha = \gamma ^ { H _ { \mathrm { p f } } } r ^ { H _ { \mathrm { p f } } } / c$ α and substituting $r ^ { H _ { \mathrm { p f } } } = 4 V _ { \mathrm { m a x } } \sqrt { J } ;$

$$
\mathrm { s u m } ~ \leq ~ { \frac { 4 \sqrt { 2 } c _ { 1 } V _ { \mathrm { m a x } } H _ { \mathrm { p f } } ^ { 3 / 2 } } { \alpha } } \cdot \sqrt { { \log \left( 2 \mathcal { N } _ { \Theta } ( 1 / T ) \left| { \cal A } \right| ^ { H _ { \mathrm { p f } } } \right) } } \cdot \gamma ^ { H _ { \mathrm { p f } } } .\tag{54}
$$

For the tail:

$$
\gamma ^ { H _ { \mathrm { p f } } } = \gamma ^ { \mathrm { l o g } ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) / \log r } = ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) ^ { - \log ( 1 / \gamma ) / \log r } = ( 4 V _ { \mathrm { m a x } } ) ^ { - 1 / \kappa _ { \mathrm { p f } } } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p f } } } } ,\tag{55}
$$

where $\kappa _ { \mathrm { p f } } = \log r / \log ( 1 / \gamma ) > 1$ . Substituting $c _ { 1 } = 4 g _ { \mathrm { m a x } } V _ { \mathrm { m a x } } / ( \varepsilon _ { \mathrm { m i x } } ^ { 3 } g _ { \mathrm { m i n } } ) , H _ { \mathrm { p f } } = \lceil \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) / \log r \rceil$ and using log $( 2  { N _ { \Theta } } ( 1 / T )  { | \mathcal { A } | ^ { H _ { \mathrm { p f } } } } ) \le \log ( 2  { N _ { \Theta } } ( 1 / T ) ) + H _ { \mathrm { p f } } \log  { | \mathcal { A } | }$ , the total (sum + tail) is

$$
\begin{array} { r l r } { \mathbb { E } [ d _ { Q } ^ { ( \Pi ) } ] } & { \le } & { \left( \frac { 1 6 \sqrt { 2 } g _ { \operatorname* { m a x } } V _ { \operatorname* { m a x } } ^ { 2 } } { \alpha \varepsilon _ { \operatorname* { m i x } } ^ { 3 } \mathcal { G } _ { \operatorname* { m i x } } } \cdot \left( \frac { \log ( 4 V _ { \operatorname* { m a x } } \sqrt { J } ) } { \log r } \right) ^ { 3 / 2 } \sqrt { \log \left( 2 \mathcal { N } _ { \Theta } ( 1 / T ) \right) + \frac { \log ( 4 V _ { \operatorname* { m a x } } \sqrt { J } ) } { \log r } \log | A | } \right. } \\ & { } & { \left. + \ 2 V _ { \operatorname* { m a x } } \right) \cdot ( 4 V _ { \operatorname* { m a x } } ) ^ { - 1 / \kappa _ { \mathrm { p t } } } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p t } } } } . \ ~ ( 5 \frac { 1 } { \log r } ) ^ { 3 / 2 } , } \end{array}\tag{6}
$$

Case $\textstyle { \mathcal { Q } } \colon \alpha < 1$ . Choose

$$
H _ { \mathrm { p f } } = \left\lceil \frac { \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) } { \log ( 1 / \gamma ) } \right\rceil .
$$

The geometric series converges: $\textstyle \sum _ { \ell > 0 } \alpha ^ { \ell } = 1 / ( 1 - \alpha )$ and $\begin{array} { r } { \sum _ { \ell > 0 } \alpha ^ { \ell } \sqrt { \ell } \leq c _ { 2 } / ( 1 - \alpha ) ^ { 3 / 2 } } \end{array}$ for a universal $c _ { 2 } > 0$ . Hence

$$
\mathrm { s u m } \ \leq \ \frac { c _ { 1 } c _ { 2 } \sqrt { 2 } } { ( 1 - \alpha ) ^ { 3 / 2 } \sqrt { J } } \cdot \sqrt { \log \bigl ( 2  { \mathcal N } _ { \Theta } ( 1 / T ) \bigr ) + \log | A | } \ = \ O \Bigl ( J ^ { - 1 / 2 } \Bigr ) .
$$

For the tail: $\gamma ^ { H _ { \mathrm { p f } } } = ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) ^ { - 1 } = O ( J ^ { - 1 / 2 } )$ . Here $\kappa _ { \mathrm { p f } } = 1$ , and both terms are $O ( J ^ { - 1 / 2 } )$

Unified bound. The choice equation 43, $H _ { \mathrm { p f } } = \lceil \log ( 4 V _ { \mathrm { m a x } } \sqrt { J } ) / \log \operatorname* { m a x } ( 1 / \gamma , r ) \rceil$ , covers both cases and yields

$$
\mathbb { E } \big [ d _ { Q } ^ { ^ { ( \mathrm { I I } ) } } \big ] \ \le \ C _ { \mathrm { p f } } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p f } } } } , \qquad \kappa _ { \mathrm { p f } } = \operatorname* { m a x } \biggl ( 1 , \frac { \log r } { \log ( 1 / \gamma ) } \biggr ) ,
$$

where

$$
C _ { \mathrm { p f } } : = \frac { c _ { 3 } V _ { \mathrm { m a x } } g _ { \mathrm { m a x } } H _ { \mathrm { p f } } ^ { 3 / 2 } } { \varepsilon _ { \mathrm { m i x } } ^ { 3 } g _ { \mathrm { m i n } } } \cdot \sqrt { \log \left( 2 \mathcal { N } _ { \Theta } ( 1 / T ) \left| \boldsymbol { \mathcal { A } } \right| ^ { H _ { \mathrm { p f } } } \right) } ,\tag{57}
$$

with $c _ { 3 }$ depending on α and $g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } }$ . In particular, the |A| dependence enters through $\sqrt { H _ { \mathrm { p f } } \log | \mathcal { A } | }$

Term (I): filter mismatch. By Lemma 2,

$$
\begin{array} { r } { d _ { Q } \big ( b _ { t } ^ { \theta ^ { \star } } , b _ { t } ^ { \bar { \theta } _ { 1 : t } } \big ) \ \leq \ 2 V _ { \operatorname* { m a x } } \lVert b _ { t } ^ { \theta ^ { \star } } - b _ { t } ^ { \bar { \theta } _ { 1 : t } } \rVert _ { \mathrm { T V } } . } \end{array}
$$

Define $\delta _ { s } : = \operatorname* { s u p } _ { s ^ { \prime } } \Vert P _ { \bar { \theta } _ { s } } ( \cdot \mid s ^ { \prime } , a _ { s } ) - P _ { \theta ^ { \star } } ( \cdot \mid s ^ { \prime } , a _ { s } ) \Vert _ { \mathrm { T V } } + \operatorname* { s u p } _ { s ^ { \prime } } \mid Z _ { \bar { \theta } _ { s } } ( o _ { s } \mid s ^ { \prime } ) / Z _ { \theta ^ { \star } } ( o _ { s } \mid s ^ { \prime } ) - 1 \mid$ and the predict-update kernel

$$
K _ { \theta } ^ { a , o } ( s , d s ^ { \prime } ) : = \frac { Z _ { \theta } ( o \mid s ^ { \prime } ) P _ { \theta } ( s ^ { \prime } \mid s , a ) } { \int Z _ { \theta } ( o \mid s ^ { \prime \prime } ) P _ { \theta } ( s ^ { \prime \prime } \mid s , a ) d s ^ { \prime \prime } } d s ^ { \prime } .
$$

Under Assumption 1, bounding the numerator below and denominator above:

$$
\begin{array} { r } { K _ { \theta } ^ { a , o } ( s , d s ^ { \prime } ) \ge \frac { g _ { \mathrm { m i n } } \varepsilon _ { \mathrm { m i x } } } { g _ { \mathrm { m a x } } } \nu ( s ^ { \prime } ) d s ^ { \prime } \forall s \in \mathcal { S } , } \end{array}\tag{58}
$$

giving Dobrushin coeficient $\rho : = 1 - \varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } / g _ { \mathrm { m a x } }$ . Standard Dobrushin contraction [10, Theorem 4.3.1] under diferent models $\theta ^ { \star } , \bar { \theta } _ { s }$ yields

$$
\begin{array} { r } { \| b _ { s } ^ { \theta ^ { \star } } - b _ { s } ^ { \bar { \theta } _ { 1 : s } } \| _ { \mathrm { T V } } \leq \rho \| b _ { s - 1 } ^ { \theta ^ { \star } } - b _ { s - 1 } ^ { \bar { \theta } _ { 1 : s - 1 } } \| _ { \mathrm { T V } } + C _ { \mathrm { f l t } } \delta _ { s } , \qquad C _ { \mathrm { f l t } } : = g _ { \operatorname* { m a x } } / g _ { \operatorname* { m i n } } . } \end{array}\tag{59}
$$

Unrolling and using $\delta _ { s } \leq d _ { \mathrm { m o d e l } } ^ { a _ { s } } ( \bar { \theta } _ { s } , \theta ^ { \star } )$

$$
\lVert b _ { t } ^ { { \boldsymbol { \theta } } ^ { \star } } - b _ { t } ^ { \bar { \theta } _ { 1 : t } } \rVert _ { \mathrm { T V } } \leq C _ { \mathrm { f l t } } \sum _ { s = 1 } ^ { t } \rho ^ { t - s } d _ { \mathrm { m o d e l } } ^ { a _ { s } } ( \bar { \theta } _ { s } , { \boldsymbol { \theta } } ^ { \star } ) .
$$

Combining. Substituting into equation 44:

$$
\mathbb { E } d _ { Q } ( b _ { t } , b _ { t } ^ { J } ) \ \leq \ \frac { C _ { \mathrm { p f } } ^ { \prime } } { \sqrt { J } } + 2 V _ { \mathrm { m a x } } C _ { \mathrm { f l t } } \sum _ { s = 1 } ^ { t } \rho ^ { t - s } d _ { \mathrm { m o d e l } } ^ { a _ { s } } ( \bar { \theta } _ { s } , \theta ^ { \star } ) .
$$

When summed over $t ,$ exchanging summation order gives $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \sum _ { s = 1 } ^ { t } \rho ^ { t - s } d _ { \mathrm { m o d e l } } ^ { a _ { s } } \leq \frac { 1 } { 1 - \rho } \sum _ { s = 1 } ^ { T } d _ { \mathrm { m o d e l } } ^ { a _ { s } } . } \end{array}$ □

## D.4 Model Estimation Error (Terms $\mathbf { A } + \mathbf { C } )$ and Filter Mismatch

Truncation. Define $V _ { \theta } ^ { \pi , B , H } ( b ) : = \mathbb { E } _ { \theta } ^ { \pi } [ \sum _ { u = 0 } ^ { H - 1 } \gamma ^ { u } r _ { u } \mid b _ { 0 } = b ]$ . Since $\begin{array} { r } { | V _ { \theta } ^ { \pi , B } - V _ { \theta } ^ { \pi , B , H } | \le \gamma ^ { H } V _ { \mathrm { { m a x } } } . } \end{array}$

$$
\mathrm { ( A ) } _ { t } + \mathrm { ( C ) } _ { t } \leq | V _ { \theta ^ { \star } } ^ { \star , B , H } ( b _ { t } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B , H } ( b _ { t } ) | + | V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi _ { t } , B , H } ( b _ { t } ) - V _ { \theta ^ { \star } } ^ { \pi _ { t } , B , H } ( b _ { t } ) | + \frac { 4 \gamma ^ { H } R _ { \operatorname* { m a x } } } { 1 - \gamma } .\tag{60}
$$

Choose $H = \lceil \log ( 2 R _ { \operatorname* { m a x } } T / ( 1 - \gamma ) ) / \log ( 1 / \gamma ) \rceil$ so the tail contributes $O ( 1 )$

Confidence sets. Let $\operatorname* { P r } _ { \theta , H } ^ { \pi }$ denote the length-H trajectory distribution under $( \theta , \pi )$ . Define

$$
\mathcal { C } _ { k } : = \big \{ \theta \in \Theta : \mathcal { L } _ { T _ { k } } ( \theta ) \geq \mathcal { L } _ { T _ { k } } ( \bar { \theta } _ { k } ) - \beta \big \} ,\tag{61}
$$

where $\mathcal { L } _ { T _ { k } }$ is the log-likelihood over epochs $1 , \ldots , k - 1$ . By construction $\bar { \theta } _ { k } \in \mathcal { C } _ { k }$ , and $\theta ^ { \star } \in \mathcal { C } _ { k }$ for all k w.h.p. when $\beta \ge c _ { 0 } \log [ \mathcal { N } _ { \Theta } ( 1 / T ) T / \delta _ { \mathrm { o p t } } ] \ [ 2 9$ , Prop. B.1].

Proposition 6 (Likelihood-to-ℓ<sub>1</sub> concentration). There exists $c > 0$ such that $w . p . \ge 1 - \delta$ simultaneously for all $n \in [ T ]$ and all $\theta \in \Theta$ with ${ \mathcal { L } } _ { n } ( \theta ) \geq { \mathcal { L } } _ { n } ( \theta ^ { \star } ) - \beta .$

$$
\sum _ { i = 1 } ^ { n } \bigl \| \mathrm { P r } _ { \theta , H } ^ { \pi ^ { i } } - \mathrm { P r } _ { \theta ^ { \star } , H } ^ { \pi ^ { i } } \bigr \| _ { 1 } ^ { 2 } \leq c \beta _ { T } , \qquad \beta _ { T } : = \beta + \log [ \Lambda _ { \Theta } ( 1 / T ) T / \delta ] .\tag{62}
$$

Proof. The episodic version is [29, Prop. B.2]. We transfer to the online setting via a blocking argument: partition the trajectory into $\lfloor T / ( 2 H ) \rfloor$ blocks of length 2H, each split into a burn-in half and a data half. Choose H so that

$$
\rho ^ { H } = ( 1 - \varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } / g _ { \mathrm { m a x } } ) ^ { H } \leq 1 / T ^ { 2 } , \mathrm { i . e . , } H \geq \frac { 2 \log T } { \log ( 1 / \rho ) } .
$$

Our choice $H = O ( \log T / \log ( 1 / \gamma ) )$ satisfies this when $\varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } / g _ { \mathrm { m a x } } = \Omega ( 1 - \gamma )$ ; otherwise, increase H by a constant factor. With $\rho ^ { H } \leq 1 / T ^ { 2 }$ , the data halves are approximately independent, with a joint TV perturbation of $O ( 1 / T )$ . Applying the episodic bound to the data halves gives equation 62 up to a constant factor. □

Corollary 2 (Epoch-level MLE concentration). Under the same conditions as Proposition 6, w.p. $\geq 1 - \delta$ , simultaneously for all epochs $k \in [ K ]$

$$
\sum _ { i = 1 } ^ { T _ { k + 1 } - 1 } \big \| \mathrm { P r } _ { \bar { \theta } _ { k } , H } ^ { \pi ^ { i } } - \mathrm { P r } _ { \theta ^ { \star } , H } ^ { \pi ^ { i } } \big \| _ { 1 } ^ { 2 } \ \leq \ c \beta _ { T } ,\tag{63}
$$

where $c > 0$ is a universal constant (at most twice the constant in Proposition 6).

Proof. Fix $k \in [ K ]$ and write $\| \operatorname* { P r } _ { \theta } ^ { \pi } \| _ { 1 } : = \| \operatorname* { P r } _ { \theta , H } ^ { \pi } - \operatorname* { P r } _ { \theta ^ { \star } , H } ^ { \pi } \| _ { 1 }$

Historical data $( i \leq T _ { k } )$ . By MLE optimality, ${ \bar { \theta } _ { k } }$ lies in $\mathcal { C } _ { k }$ , so Proposition 6 at $n = T _ { k }$ gives

$$
\sum _ { i = 1 } ^ { T _ { k } } \bigl \| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { i } } \bigr \| _ { 1 } ^ { 2 } \leq c \beta _ { T } .\tag{64}
$$

Within-epoch data $( i \in E _ { k } )$ . Since ${ \bar { \theta } _ { k } }$ is $\mathcal { F } _ { T _ { k } }$ -measurable, we use a self-normalized argument for the fixed hypothesis ${ \bar { \theta } _ { k } }$ . Define

$$
L _ { n } ^ { ( k ) } : = \sum _ { i = T _ { k } + 1 } ^ { T _ { k } + n } \left[ \ell _ { i } ( \theta ^ { \star } ) - \ell _ { i } ( \bar { \theta } _ { k } ) \right] , \qquad M _ { n } ^ { ( k ) } : = \sum _ { i = T _ { k } + 1 } ^ { T _ { k } + n } \left[ \mathrm { K L } ( \mathrm { P r } _ { \theta ^ { \star } } ^ { \pi ^ { i } } \| \mathrm { P r } _ { \theta _ { k } } ^ { \pi ^ { i } } ) - \left( \ell _ { i } ( \theta ^ { \star } ) - \ell _ { i } ( \bar { \theta } _ { k } ) \right) \right] ,
$$

where $\ell _ { i } ( \theta ) : = \log p _ { \theta } ( o _ { i } \mid s _ { i } , a _ { i } )$ . By Pinsker’s inequality and blocking:

$$
\frac 1 2 \sum _ { i \in E _ { k } } \big \| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { i } } \big \| _ { 1 } ^ { 2 } \ \leq \ L _ { | E _ { k } | } ^ { ( k ) } + M _ { | E _ { k } | } ^ { ( k ) } .\tag{65}
$$

$M _ { n } ^ { ( k ) }$ is a supermartingale (each increment has non-positive conditional expectation). The maximal inequality and a covering-number union bound give

$$
\mathrm { P r } \left( \operatorname* { s u p } _ { n \leq | E _ { k } | } M _ { n } ^ { ( k ) } \geq c _ { 1 } \log \left[  { \mathcal { N } _ { \Theta } } ( 1 / T ) / \delta \right] \right) \leq \delta / K .
$$

For $L _ { | E _ { k } | } ^ { ( k ) } \mathsf { : }$ each $| \ell _ { i } ( \theta ^ { \star } ) - \ell _ { i } ( \bar { \theta } _ { k } ) | \leq \log ( g _ { \mathrm { { m a x } } } / g _ { \mathrm { { m i n } } } )$ and $\mathbb { E } [ \ell _ { i } ( \theta ^ { \star } ) - \ell _ { i } ( \bar { \theta } _ { k } ) \ | \ \mathcal { F } _ { i - 1 } ] \ge 0$ . By Azuma-Hoefding:

$$
\mathrm { P r } \Big ( L _ { | E _ { k } | } ^ { ( k ) } \geq c _ { 2 } \sqrt { | E _ { k } | \log ( 1 / \delta ) } \log ( g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } } ) \Big ) \leq \delta / K .
$$

Since $\sqrt { T } \log ( g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } } )$ is absorbed into $\beta _ { T }$ :

$$
\sum _ { i \in E _ { k } } \big \| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { i } } \big \| _ { 1 } ^ { 2 } \ \leq \ c \beta _ { T } .\tag{66}
$$

Combining. Summing equation 64 and equation 66, redefining $c  2 c$ , and taking a union bound over $k \in [ K ]$ gives equation 63. □

Transferring model error from historical to arbitrary policies requires action-space coverage via ε-greedy exploration.

Lemma 11 (Trajectory coverage under Doeblin mixing and ε-greedy). Under Assumption $^ { 1 , }$ for any two models $\theta , \theta ^ { \prime } \in \Theta$ , any policy $\pi _ { ; }$ , and any policy $\pi _ { \varepsilon }$ satisfying $\pi _ { \varepsilon } ( a \mid h ) \ge \varepsilon / | { \mathcal { A } } |$ for all $a , h$ $( w i t h \ \varepsilon \in ( 0 , 1 ] )$ ,

$$
\big \| \mathrm { P r } _ { \theta , H } ^ { \pi } - \mathrm { P r } _ { \theta ^ { \prime } , H } ^ { \pi } \big \| _ { 1 } \ \leq \ C _ { \mathrm { c o v } } \left\| \mathrm { P r } _ { \theta , H } ^ { \pi _ { \varepsilon } } - \mathrm { P r } _ { \theta ^ { \prime } , H } ^ { \pi _ { \varepsilon } } \right\| _ { 1 } , \qquad C _ { \mathrm { c o v } } : = \frac { H \left| \mathcal { A } \right| } { \varepsilon } \cdot \frac { g _ { \operatorname* { m a x } } ^ { 2 } } { \varepsilon _ { \mathrm { m i x } } g _ { \operatorname* { m i n } } ^ { 2 } } .\tag{67}
$$

Proof. Step 1: Telescoping. Let $\mathrm { P r } ^ { ( k ) }$ be the hybrid using θ for steps $1 , \ldots , k$ and $\theta ^ { \prime }$ for steps $k + 1 , \dots , H$ , under policy π. Let $\Delta _ { k } ( s , a ) : = \| P _ { \boldsymbol \theta } ( \cdot \mid s , a ) - P _ { \boldsymbol \theta ^ { \prime } } ( \cdot \mid s , a ) \| _ { \mathrm { T V } }$ and $\Delta ^ { Z } ( s ) : = \operatorname* { s u p } _ { o } | Z _ { \theta } ( o \mid$ $s ) / Z _ { \theta ^ { \prime } } ( o \mid s ) - 1 |$ . Then

$$
\Vert \mathrm { P r } _ { \theta , H } ^ { \pi } - \mathrm { P r } _ { \theta ^ { \prime } , H } ^ { \pi } \Vert _ { 1 } \leq \sum _ { k = 0 } ^ { H - 1 } \Vert \mathrm { P r } ^ { ( k + 1 ) } - \mathrm { P r } ^ { ( k ) } \Vert _ { 1 } ,\tag{68}
$$

$$
\| \operatorname* { P r } ^ { ( k + 1 ) } - \operatorname* { P r } ^ { ( k ) } \| _ { 1 } \ \leq \ \mathbb { E } _ { \mu _ { k } ^ { \pi } } [ \operatorname* { s u p } _ { a } \Delta _ { k } ( S _ { k } , a ) + \Delta ^ { Z } ( S _ { k } ) ] ,\tag{69}
$$

where $\mu _ { k } ^ { \pi }$ is the state marginal at step k under the hybrid.

Step 2: State-marginal ratio. By the Doeblin decomposition $\tilde { \mu } _ { k } ^ { \pi } \ge \varepsilon _ { \operatorname* { m i x } } \nu$ and $\tilde { \mu } _ { k } ^ { \pi ^ { \prime } } \geq \varepsilon _ { \operatorname* { m i x } } \nu$ , for any $h : S  [ 0 , 1 ]$ :

$$
\mathbb { E } _ { \tilde { \mu } _ { k } ^ { \pi } } [ h ] \le \varepsilon _ { \mathrm { m i x } } \mathbb { E } _ { \nu } [ h ] + ( 1 - \varepsilon _ { \mathrm { m i x } } ) \le \frac { 1 } { \varepsilon _ { \mathrm { m i x } } } \mathbb { E } _ { \tilde { \mu } _ { k } ^ { \pi ^ { \prime } } } [ h ] .
$$

Accounting for the factor $g _ { \mathrm { m a x } } / g _ { \mathrm { m i n } }$ from the Bayes update normalization gives

$$
\mathbb { E } _ { \mu _ { k } ^ { \pi } } [ h ] \ \leq \ \frac { g _ { \operatorname* { m a x } } } { g _ { \operatorname* { m i n } } \varepsilon _ { \operatorname* { m i x } } } \mathbb { E } _ { \mu _ { k } ^ { \pi ^ { \prime } } } [ h ] .
$$

Step 3: Action coverage. Since $\pi _ { \varepsilon } ( a \mid h ) \ge \varepsilon / | { \mathcal { A } } |$

$$
\operatorname* { s u p } _ { a } \Delta _ { k } ( s , a ) \ \leq \ { \frac { | A | } { \varepsilon } } \mathbb { E } _ { A \sim \pi _ { \varepsilon } } [ \Delta _ { k } ( s , A ) ] .
$$

Step $\it 4 .$ Assembly. Combining Steps 1–3 term-by-term:

$$
\| \operatorname* { P r } ^ { ( k + 1 ) } - \operatorname* { P r } ^ { ( k ) } \| _ { 1 } \leq \frac { | \mathcal { A } | } { \varepsilon } \cdot \frac { g _ { \operatorname* { m a x } } ^ { 2 } } { \varepsilon _ { \operatorname* { m i x } } g _ { \operatorname* { m i n } } ^ { 2 } } \cdot \| \operatorname* { P r } _ { \pi _ { \varepsilon } } ^ { ( k + 1 ) } - \operatorname* { P r } _ { \pi _ { \varepsilon } } ^ { ( k ) } \| _ { 1 } .
$$

Summing over $k = 0 , \ldots , H - 1$ yields equation 67.

Lemma 12 (Simulation lemma). For any θ, $\theta ^ { \prime } \in \Theta , \pi$ , and b:

$$
\vert V _ { \theta } ^ { \pi , B , H } ( b ) - V _ { \theta ^ { \prime } } ^ { \pi , B , H } ( b ) \vert \ \leq \ \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \Vert \mathrm { P r } _ { \theta , H } ^ { \pi } - \mathrm { P r } _ { \theta ^ { \prime } , H } ^ { \pi } \Vert _ { 1 } .
$$

Proof. Telescope the H-truncated value diference into one-step terms, each bounded by $R _ { \mathrm { m a x } }$ .

Proposition 7 (Control of $\mathrm { ( A ) + ( C ) + }$ filter mismatch). Under Assumption 1, consider the doubling-epoch variant of Algorithm 1 in which time is partitioned into epochs $E _ { k } = [ T _ { k } , T _ { k + 1 } )$ with $T _ { k } = 2 ^ { k - 1 } , k = 1 , \ldots , K : = \lceil \log _ { 2 } T \rceil$ . Within each epoch $E _ { k }$ , the model ${ \bar { \theta } _ { k } }$ is frozen (computed from data in epochs $1 , \ldots , k - 1 )$ and the agent uses $\varepsilon _ { k } { - } g r e e d y$ exploration with a constant rate $\varepsilon _ { k }$ . Let $K _ { \mathrm { c o v } } : = H | \mathbfcal { A } | g _ { \mathrm { m a x } } ^ { 2 } / ( \varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } ^ { 2 } )$ . Then with probability $\ge 1 - \delta _ { \mathrm { o p t } }$

$$
\sum _ { t = 1 } ^ { T } ( C ) _ { t } \leq \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \sqrt { c \beta _ { T } } \sum _ { k = 1 } ^ { K } \sqrt { | E _ { k } | } + \frac { 2 \gamma ^ { H } R _ { \operatorname* { m a x } } T } { 1 - \gamma } ,\tag{70}
$$

$$
\sum _ { t = 1 } ^ { T } { ( A ) _ { t } } \le \frac { R _ { \operatorname* { m a x } } K _ { \mathrm { c o v } } } { 1 - \gamma } \sqrt { c \beta _ { T } } \sum _ { k = 2 } ^ { K } { \frac { | E _ { k } | } { \sqrt { \sum _ { j = 1 } ^ { k - 1 } \varepsilon _ { j } ^ { 2 } | E _ { j } | } } } + \frac { 2 \gamma ^ { H } R _ { \operatorname* { m a x } } T } { 1 - \gamma } ,\tag{71}
$$

$$
\sum _ { t = 1 } ^ { T } d _ { \mathrm { m o d e l } } ^ { a _ { t } } ( { \bar { \theta } } _ { k ( t ) } , \theta ^ { \star } ) \leq C _ { \mathrm { m i x } } \sqrt { c \beta _ { T } } \sum _ { k = 1 } ^ { K } \sqrt { | E _ { k } | } ,\tag{72}
$$

where $\begin{array} { r } { d _ { \mathrm { m o d e l } } ^ { a } ( \theta , \theta ^ { \prime } ) : = \operatorname* { s u p } _ { s \in \mathcal { S } } \| P _ { \theta } ( \cdot \mid s , a ) - P _ { \theta ^ { \prime } } ( \cdot \mid s , a ) \| _ { \mathrm { T V } } + \operatorname* { s u p } _ { s \in \mathcal { S } , o \in \mathcal { O } } \left| Z _ { \theta } ( o \mid s ) / Z _ { \theta ^ { \prime } } ( o \mid s ) - 1 \right| } \end{array}$ is the per-action model discrepancy, and $C _ { \mathrm { m i x } }$ depends on $\varepsilon _ { \mathrm { m i x } } , g _ { \mathrm { m i n } } , g _ { \mathrm { m a x } } , H$

Proof. By Corollary 2, equation equation 63 holds for all $k \leq K$ . For brevity, write $\| \operatorname* { P r } _ { \theta } ^ { \pi } \| _ { 1 } : =$ $\| \operatorname* { P r } _ { \theta , H } ^ { \pi } - \operatorname* { P r } _ { \theta ^ { \star } , H } ^ { \pi } \| _ { 1 }$ throughout this proof.

Term (C). By Lemma 12,

$$
( \mathrm { C } ) _ { t } \ \leq \ \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \Vert \mathrm { P r } _ { \bar { \theta } _ { k ( t ) } } ^ { \pi ^ { t } } \Vert _ { 1 } + \frac { 2 \gamma ^ { H } R _ { \operatorname* { m a x } } } { 1 - \gamma } .
$$

Since $\theta _ { k }$ is fixed within $E _ { k }$ , Cauchy-Schwarz and equation 63 give

$$
\sum _ { t \in E _ { k } } \big \| \mathrm { P r } _ { \hat { \theta } _ { k } } ^ { \pi ^ { t } } \big \| _ { 1 } \leq \sqrt { | E _ { k } | \sum _ { t \in E _ { k } } \big \| \mathrm { P r } _ { \hat { \theta } _ { k } } ^ { \pi ^ { t } } \big \| _ { 1 } ^ { 2 } } \leq \sqrt { | E _ { k } | \cdot c \beta _ { T } } .
$$

Summing across epochs: $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \sqrt { | E _ { k } | \cdot c \beta _ { T } } = \sqrt { c \beta _ { T } } \sum _ { k } 2 ^ { ( k - 1 ) / 2 } = O ( \sqrt { T \beta _ { T } } ) } \end{array}$

Term (A). By Lemmas 12 and 11:

$$
\mathrm { ( A ) } _ { t } \ \leq \ \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \operatorname* { s u p } _ { \pi } \bigl \| \mathrm { P r } _ { \widetilde { \theta } _ { k ( t ) } } ^ { \pi } \bigr \| _ { 1 } + \frac { 2 \gamma ^ { H } R _ { \operatorname* { m a x } } } { 1 - \gamma } , \qquad \bigl \| \mathrm { P r } _ { \widetilde { \theta } _ { k } } ^ { \pi } \bigr \| _ { 1 } \ \leq \ \frac { K _ { \mathrm { c o v } } } { \varepsilon _ { j } } \bigl \| \mathrm { P r } _ { \widetilde { \theta } _ { k } } ^ { \pi ^ { j } } \bigr \| _ { 1 } \quad \forall j < k .
$$

Squaring, multiplying by $\varepsilon _ { j } ^ { 2 }$ , and summing over prior epochs using equation 63:

$$
\Vert \mathbf { P r } _ { \theta _ { k } } ^ { \pi } \Vert _ { 1 } ^ { 2 } \sum _ { j < k } \varepsilon _ { j } ^ { 2 } | E _ { j } | \ \leq \ K _ { \mathrm { c o v } } ^ { 2 } \sum _ { j < k } \sum _ { \ell \in E _ { j } } \Vert \mathbf { P r } _ { \theta _ { k } } ^ { \pi ^ { \ast } } \Vert _ { 1 } ^ { 2 } \ \leq \ K _ { \mathrm { c o v } } ^ { 2 } c \beta _ { T } \quad \implies \quad \underset { \pi } { \operatorname* { s u p } } \Vert \mathbf { P r } _ { \theta _ { k } } ^ { \pi } \Vert _ { 1 } \ \leq \ \frac { K _ { \mathrm { c o v } } \sqrt { c \beta _ { T } } } { \sqrt { \sum _ { j < k } \varepsilon _ { j } ^ { 2 } | E _ { j } | } } .
$$

Multiplying by $\lvert E _ { k } \rvert$ and summing gives equation 71.

Filter mismatch. By the telescoping argument of Lemma 11, using only Steps 1–2 and hence without the $| { \mathcal { A } } | / \varepsilon$ action-coverage factor, since we evaluate at the executed action $a _ { t }$

$$
d _ { \mathrm { m o d e l } } ^ { a _ { t } } ( \bar { \theta } _ { k ( t ) } , \theta ^ { \star } ) \leq C _ { \operatorname* { m i x } } \big \| \mathrm { P r } _ { \bar { \theta } _ { k ( t ) } } ^ { \pi ^ { t } } \big \| _ { 1 } , \qquad C _ { \operatorname* { m i x } } : = \frac { H g _ { \operatorname* { m a x } } ^ { 2 } } { \varepsilon _ { \operatorname* { m i x } } g _ { \operatorname* { m i n } } ^ { 2 } } .
$$

Cauchy-Schwarz and equation 63 give $\begin{array} { r } { \sum _ { t \in E _ { k } } \| \operatorname* { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { t } } \| _ { 1 } \le \sqrt { | E _ { k } | \cdot c \beta _ { T } } } \end{array}$ . Summing across epochs yields equation 72. □

## D.5 Proof of the Main Regret Bound (Theorem 1)

Proof of Theorem 1. Use doubling epochs $E _ { k } = [ T _ { k } , T _ { k + 1 } ) , T _ { k } = 2 ^ { k - 1 } , K = \lceil \log _ { 2 } T \rceil$ , with $\varepsilon _ { k } { \mathrm { - g r e e d y } }$ exploration within each epoch. Decompose:

$$
\mathrm { R e g } _ { T } \ \leq \ \underbrace { \frac { 2 R _ { \mathrm { m a x } } } { 1 - \gamma } \sum _ { k } | E _ { k } | \varepsilon _ { k } } _ { \mathrm { e x p l o r a t i o n } } + \mathbb { E } \sum _ { t = 1 } ^ { T } \bigl [ ( \mathrm { A } ) _ { t } + ( \mathrm { B } ) _ { t } + ( \mathrm { C } ) _ { t } \bigr ] .
$$

Term (C). By equation 70 (choosing $\delta _ { \mathrm { o p t } } = 1 / T )$

$$
\mathbb { E } \sum _ { t } \left( \mathrm { C } \right) _ { t } \ \leq \ \frac { R _ { \operatorname* { m a x } } \sqrt { c \beta _ { T } } } { 1 - \gamma } \sum _ { k = 1 } ^ { K } \sqrt { \left| E _ { k } \right| } + O ( 1 ) \ = \ \widetilde O \left( \frac { \sqrt { T } } { 1 - \gamma } \right) .
$$

Term (A). By equation 71 with $\varepsilon _ { k } = \eta / \sqrt { | E _ { k } | } \ ( \mathrm { s o } \ \varepsilon _ { k } ^ { 2 } | E _ { k } | = \eta ^ { 2 } )$

$$
\mathbb { E } \sum _ { t } \left( \mathrm { A } \right) _ { t } \ \leq \ \frac { R _ { \mathrm { m a x } } K _ { \mathrm { c o v } } \sqrt { c \beta _ { T } } } { ( 1 - \gamma ) \eta } \sum _ { k = 2 } ^ { K } \frac { \vert E _ { k } \vert } { \sqrt { k - 1 } } + { \cal O } ( 1 ) .
$$

Since ${ \textstyle \sum _ { k = 2 } ^ { K } 2 ^ { k - 1 } } / { \sqrt { k - 1 } } = \Theta ( T / { \sqrt { \log T } } )$ , this scales as $K _ { \mathrm { c o v } } \sqrt { \beta _ { T } } \cdot T / ( \eta \sqrt { \log T } ( 1 - \gamma ) )$

Term (B). By Theorem 2, Propositions 2, 3, 4:

$$
\mathbb { E } \sum _ { t } ( \mathrm { B } ) _ { t } \leq \frac { 2 C _ { \mathrm { p f } } T } { ( 1 - \gamma ) } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p f } } } } + \frac { C _ { \mathrm { m c } } V _ { \mathrm { m a x } } T } { ( 1 - \gamma ) \sqrt { N } } + \frac { 2 \gamma ^ { d + 1 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { t } \Xi _ { \bar { \theta } _ { k ( t ) } } + O ( \sqrt { T \beta _ { T } } ) .
$$

The $O ( \sqrt { T \beta _ { T } } )$ absorbs the filter mismatch via equation 72.

Optimization. The exploration cost is $O ( \eta \sqrt { T } R _ { \mathrm { m a x } } / ( 1 - \gamma ) )$ . Balancing it with term (A) gives $\eta ^ { \star } \sim K _ { \mathrm { c o v } } ^ { 1 / 2 } T ^ { 1 / 4 } ( \beta _ { T } / \log T ) ^ { 1 / 4 }$ and a total of $\widetilde { O } ( K _ { \mathrm { c o v } } ^ { 1 / 2 } T ^ { 3 / 4 } / ( 1 - \gamma ) ^ { 3 / 2 } )$ . Term (C) is $\widetilde { O } ( \sqrt { T } )$ , and the $T ^ { 3 / 4 }$ rate comes from the coverage transfer for term (A). □

## D.6 Optimal Exploration Schedule

Set $\varepsilon _ { k } = \eta / \sqrt { | E _ { k } | }$ so that $\varepsilon _ { k } ^ { 2 } | E _ { k } | = \eta ^ { 2 }$ and $\begin{array} { r } { \sum _ { j < k } \varepsilon _ { j } ^ { 2 } | E _ { j } | = ( k - 1 ) \eta ^ { 2 } } \end{array}$ . The exploration-dependen terms scale as:

Exploration cost: $\begin{array} { r } { \sum _ { k } | E _ { k } | \varepsilon _ { k } = \eta \sum _ { k } \sqrt { | E _ { k } | } = { \cal O } ( \eta \sqrt { T } ) . } \end{array}$

Term (A): $O \big ( K _ { \mathrm { c o v } } \sqrt { \beta _ { T } } \cdot T / ( \eta \sqrt { \log T } ( 1 - \gamma ) ) \big )$

Term (C): $O \big ( \sqrt { T \beta _ { T } } / ( 1 - \gamma ) \big )$ , independent of η.

Balancing term (A) against the exploration cost gives

$$
\eta ^ { \star } \sim \left( \frac { K _ { \mathrm { c o v } } \sqrt { T \beta _ { T } } } { \sqrt { \log T } } \right) ^ { 1 / 2 } , \qquad ( \mathrm { A } ) + \mathrm { e x p l o r a t i o n } = \widetilde O \left( \frac { K _ { \mathrm { c o v } } ^ { 1 / 2 } T ^ { 3 / 4 } } { ( 1 - \gamma ) } \right) .
$$

The total regret is:

$$
\mathrm { R e g } _ { T } = \widetilde { O } \Bigg ( \frac { K _ { \mathrm { c o v } } ^ { 1 / 2 } R _ { \mathrm { m a x } } T ^ { 3 / 4 } } { ( 1 - \gamma ) ^ { 3 / 2 } } \Bigg ) + \frac { C _ { \mathrm { p f } } R _ { \mathrm { m a x } } T } { ( 1 - \gamma ) ^ { 2 } } \cdot J ^ { - \frac { 1 } { 2 \kappa _ { \mathrm { p f } } } } + \frac { C _ { \mathrm { m c } } R _ { \mathrm { m a x } } T } { ( 1 - \gamma ) ^ { 2 } \sqrt { N } } + \frac { 2 \gamma ^ { d + 1 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { t } \Xi _ { \bar { \theta } _ { k ( t ) } } .\tag{73}
$$

The $T ^ { 3 / 4 }$ rate is driven by the $1 / \varepsilon$ coverage transfer for term (A). Using optimistic model selection $( \widetilde { \theta } _ { t } : = \arg \operatorname* { m a x } _ { \theta \in \mathcal { C } _ { t } } V _ { \theta } ^ { \star , B } )$ eliminates term (A) and recovers $\widetilde { O } ( \sqrt { T } )$ from term (C) alone, at the cost of solving an optimistic planning problem [29].

A constant ε gives the same ${ \widetilde O } ( T ^ { 3 / 4 } )$ rate after optimization and may be preferable when $T$ is unknown.

## D.7 Greedy Exploitation with Ofline Coverage

The $T ^ { 3 / 4 }$ rate in Theorem 1 arises from the coverage transfer for term (A), which requires ε-greedy exploration to ensure the MLE concentrates at all policies. An alternative is to replace online exploration with an ofline dataset that already provides coverage of the optimal policy. This mirrors the hybrid RL framework [35], where ofline data with suficient coverage enables greedy online exploitation.

Setting. Suppose the agent has access to an ofline dataset $\mathcal { D } _ { \mathrm { o f f } }$ of N i.i.d. length-H trajectories collected by a behavior policy $\pi _ { b }$ in the true POMDP (model $\theta ^ { \star } )$ . The agent then runs PF-MCTQ for T online steps with greedy exploitation $( \varepsilon _ { t } = 0$ , no exploration). The MLE is computed from the union $\mathcal { D } _ { \mathrm { o f f } } \cup \mathcal { D } _ { \mathrm { o n } }$

Definition 2 (Single-policy concentrability). The single-policy concentrability coeficient of the behavior policy $\pi _ { b }$ with respect to the optimal policy $\pi ^ { \star } : =$ arg max<sub>π</sub> $V _ { \theta ^ { \star } } ^ { \pi , B }$ is

$$
c _ { \mathrm { o f f } } ^ { \star } : = \operatorname* { s u p } _ { \theta , \theta ^ { \prime } \in \Theta } \frac { \Vert \operatorname* { P r } _ { \theta , H } ^ { \pi ^ { \star } } - \operatorname* { P r } _ { \theta ^ { \prime } , H } ^ { \pi ^ { \star } } \Vert _ { 1 } } { \Vert \operatorname* { P r } _ { \theta , H } ^ { \pi _ { b } } - \operatorname* { P r } _ { \theta ^ { \prime } , H } ^ { \pi _ { b } } \Vert _ { 1 } } .\tag{74}
$$

Under Assumption 1, if $\pi _ { b } ( a \mid h ) \geq \alpha _ { b } / | { \mathcal { A } } |$ for all $a , h ,$ Lemma 11 gives $c _ { \mathrm { o f f } } ^ { \star } \leq K _ { \mathrm { c o v } } / \alpha _ { b } =$ $\frac { H | \mathcal { A } | } { \alpha _ { b } } \cdot \frac { g _ { \mathrm { m a x } } ^ { 2 } } { \varepsilon _ { \mathrm { m i x } } g _ { \mathrm { m i n } } ^ { 2 } }$ . Unlike the all-policy concentrability provided by ε-greedy exploration, which takes a supremum over all policies, $c _ { \mathrm { o f f } } ^ { \star }$ involves only the optimal policy.

Proposition 8 (Regret with ofline coverage). Under Assumption 1, suppose the agent has N ofline trajectories from $\pi _ { b }$ and runs the doubling-epoch PF-MCTQ with greedy exploitation $( \varepsilon _ { k } = 0 )$ for T online steps. Let $\beta _ { T , N } : = \beta + \log [ \mathcal { N } _ { \Theta } ( 1 / ( T + N ) ) ( T + N ) / \delta ]$ . Then ${ \boldsymbol { v } } . { \boldsymbol { p } } . \geq 1 - \delta$ :

$$
\mathrm { R e g } _ { T } \le \underbrace { \widetilde { \cal O } \left( \frac { R _ { \operatorname* { m a x } } \sqrt { T } } { 1 - \gamma } \right) } _ { t e r m \mathrm { \it ~ ( C ) } } + \underbrace { \frac { c _ { \mathrm { o f f } } ^ { \star } R _ { \operatorname* { m a x } } \sqrt { c \beta _ { T , N } } } { 1 - \gamma } } _ { t e r m \mathrm { \it ~ ( A ) } } \cdot \frac { T } { \sqrt { N } } + c o m p u t a t i o n a l \ t e r m s ,\tag{75}
$$

where the computational terms are the same as in Theorem 1 (particle, MC, structural).

In particular, when $N \geq ( c _ { \mathrm { o f f } } ^ { \star } ) ^ { 2 } T$ , the total regret is $\widetilde { O } \big ( \sqrt { T } / ( 1 - \gamma ) \big )$ plus computational terms, so $\sqrt { T }$ regret is achieved without any online exploration.

Proof. The regret decomposition equation 11 gives

$$
\mathrm { R e g } _ { T } \ = \ \mathbb { E } \sum _ { t = 1 } ^ { T } \bigl [ ( \mathrm { A } ) _ { t } + ( \mathrm { B } ) _ { t } + ( \mathrm { C } ) _ { t } \bigr ] .
$$

There is no exploration cost since $\varepsilon _ { k } = 0$

MLE concentration with ofline data. The MLE is fit on $\mathcal { D } _ { \mathrm { o f f } } \cup \mathcal { D } _ { \mathrm { o n } }$ . The ofline trajectories are i.i.d. under $( \theta ^ { \star } , \pi _ { b } )$ , so the episodic MLE concentration [29, Prop. B.2] applies directly to the ofline portion. Combined with the blocking argument for the online portion (Proposition 6):

$$
N \left\| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi _ { b } } \right\| _ { 1 } ^ { 2 } + \sum _ { t = 1 } ^ { T _ { k + 1 } - 1 } \left\| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { t } } \right\| _ { 1 } ^ { 2 } \le c \beta _ { T , N } \qquad \forall k \le K ,\tag{76}
$$

using the shorthand $\| \operatorname* { P r } _ { \theta } ^ { \pi } \| _ { 1 } : = \| \operatorname* { P r } _ { \theta , H } ^ { \pi } - \operatorname* { P r } _ { \theta ^ { \star } , H } ^ { \pi } | \|$ <sub>1</sub>.

Term (C). As in the proof of Proposition 7, since ${ \bar { \theta } _ { k } }$ is fixed within $E _ { k }$ , applying Cauchy-Schwarz within each epoch to the online terms in equation 76 gives

$$
\sum _ { t = 1 } ^ { T } ( \mathrm { C } ) _ { t } \leq \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \sqrt { c \beta _ { T , N } } \sum _ { k = 1 } ^ { K } \sqrt { | E _ { k } | } + { \cal O } ( \gamma ^ { H } T ) = \widetilde { { \cal O } } \left( \frac { R _ { \operatorname* { m a x } } \sqrt { T } } { 1 - \gamma } \right) .
$$

Term (A). We use a tighter decomposition that avoids sup<sub>π</sub>:

$$
\mathrm { ( A ) } _ { t } = V _ { \theta ^ { \star } } ^ { \star , B } ( b _ { t } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B } ( b _ { t } ) \ \leq \ V _ { \theta ^ { \star } } ^ { \pi ^ { \star } , B } ( b _ { t } ) - V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi ^ { \star } , B } ( b _ { t } ) \ \leq \ \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \| \mathrm { P r } _ { \bar { \theta } _ { k ( t ) } } ^ { \pi ^ { \star } } \| _ { 1 } ,\tag{77}
$$

where the first inequality uses $V _ { \bar { \theta } _ { k ( t ) } } ^ { \star , B } \geq V _ { \bar { \theta } _ { k ( t ) } } ^ { \pi ^ { \star } , B }$ (optimality under $\bar { \theta } _ { k ( t ) } )$ and the second is the simulation lemma (Lemma 12).

By Definition 2:

$$
\left\| \operatorname* { P r } _ { \bar { \theta } _ { k } } ^ { \pi ^ { \star } } \right\| _ { 1 } \leq c _ { \mathrm { o f f } } ^ { \star } \left\| \operatorname* { P r } _ { \bar { \theta } _ { k } } ^ { \pi _ { b } } \right\| _ { 1 } .
$$

From the ofline portion of equation 76:

$$
\big \| \mathrm { P r } _ { \bar { \theta } _ { k } } ^ { \pi _ { b } } \big \| _ { 1 } \ \leq \ \sqrt { \frac { c \beta _ { T , N } } { N } } .
$$

Since $\bar { \theta } _ { k }$ is frozen within $E _ { k }$ , this bound is constant across the epoch:

$$
\sum _ { t \in E _ { k } } ( \mathrm { A } ) _ { t } \ \leq \ \frac { R _ { \operatorname* { m a x } } } { 1 - \gamma } \cdot c _ { \mathrm { o f f } } ^ { \star } \sqrt { \frac { c \beta _ { T , N } } { N } } \cdot | E _ { k } | .
$$

Summing across epochs:

$$
\sum _ { t = 1 } ^ { T } ( \mathrm { A } ) _ { t } \ \leq \ \frac { c _ { \mathrm { o f f } } ^ { \star } R _ { \operatorname* { m a x } } } { 1 - \gamma } \sqrt { \frac { c \beta _ { T , N } } { N } } \cdot T .
$$

Term $( B )$ and filter mismatch. Identical to the proof of Theorem 1. The filter mismatch $\textstyle \sum _ { t } d _ { \mathrm { m o d e l } } ^ { a _ { t } }$ is bounded by $\begin{array} { r } { C _ { \operatorname* { m i x } } \sqrt { c \beta _ { T , N } } \sum _ { k } \sqrt { \left| E _ { k } \right| } = \widetilde { O } ( \sqrt { T } ) } \end{array}$ via equation 72 (the online diagonal bound, which does not require coverage).

Combining.

$$
\mathrm { R e g } _ { T } \leq \frac { R _ { \mathrm { m a x } } } { 1 - \gamma } \tilde { O } ( \sqrt { T } ) + \frac { c _ { \mathrm { o f f } } ^ { * } R _ { \mathrm { m a x } } \sqrt { c \beta _ { T , N } } } { 1 - \gamma } \cdot \frac { T } { \sqrt { N } } + \frac { C _ { \mathrm { p f } } ^ { \prime } R _ { \mathrm { m a x } } T } { ( 1 - \gamma ) ^ { 2 } \sqrt { J } } + \frac { C _ { \mathrm { m a x } } R _ { \mathrm { m a x } } T } { ( 1 - \gamma ) ^ { 2 } \sqrt { N _ { \mathrm { m c } } } } + \frac { 2 \gamma ^ { d + 1 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { t } \Xi _ { \bar { \theta } _ { k ( t ) } } .
$$

When $N \geq ( c _ { \mathrm { o f f } } ^ { \star } ) ^ { 2 } T$ , the second term is $O ( \sqrt { T \beta _ { T , N } } / ( 1 - \gamma ) ) = \widetilde { O } ( \sqrt { T } / ( 1 - \gamma ) )$ , and the leading rate is $\widetilde { O } ( \sqrt { T } )$ □

## D.8 Variance Perspective on the QMDP Bias

We provide a variance-based interpretation of the Jensen gap $\Xi _ { \theta }$ defined in equation 13. We show that $\Xi _ { \theta }$ is controlled by the curvature of the value function along a suficient embedding and the posterior variance of the embedded features.

Value-relevant embedding. Let $g : S  \mathbb { R } ^ { k }$ be a measurable embedding, and write

$$
m _ { g } ( b ) : = \mathbb { E } _ { S \sim b } [ g ( S ) ] , \qquad \Sigma _ { g } ( b ) : = \operatorname { C o v } _ { S \sim b } \bigl ( g ( S ) \bigr ) .
$$

Assumption 3 (Value-relevant embedding). There exist twice continuously diferentiable maps $F _ { \theta } ^ { a } : \mathbb { R } ^ { k }  \mathbb { R }$ for each $a \in { \mathcal { A } }$ such that $Q _ { \theta } ^ { \star , S } ( s , a ) = F _ { \theta } ^ { a } ( g ( s ) )$ for all $( s , a ) \in \mathcal S \times \mathcal A$ , with curvature bounds

$$
\operatorname* { s u p } _ { a \in \mathcal { A } } \operatorname* { s u p } _ { x \in \mathrm { c o n v } ( g ( S ) ) } \left\| \nabla ^ { 2 } F _ { \theta } ^ { a } ( x ) \right\| _ { \mathrm { o p } } \leq L _ { \theta } \qquad a n d \qquad \operatorname* { s u p } _ { x \in \mathrm { c o n v } ( g ( S ) ) } \left\| \nabla ^ { 2 } F _ { \theta } ( x ) \right\| _ { \mathrm { o p } } \leq L _ { \theta } ,
$$

where $F _ { \theta } ( x ) : = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( x )$ and $V _ { \theta } ^ { \star , S } ( s ) = F _ { \theta } ( g ( s ) )$

The second curvature condition on $F _ { \theta } = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a }$ does not hold automatically, since $F _ { \theta }$ is not $C ^ { 2 }$ at switching boundaries where the maximizing action changes. The bound equation 83 uses $\| \nabla ^ { 2 } F _ { \theta } \| _ { \mathrm { o p } } \leq L _ { \theta }$ only for the upper bound on $\widetilde { V } _ { \boldsymbol { \theta } } ^ { \star } ( \boldsymbol { b } )$ ; the lower bound on $\bar { V } _ { \theta } ^ { \star , S } ( b )$ uses only the per-action bounds $\| \nabla ^ { 2 } F _ { \theta } ^ { a } \| _ { \mathrm { o p } } \leq L _ { \theta }$

We first state an auxiliary lemma used in the proof of Proposition 9.

Lemma 13 (Mean-belief value approximation). Under Assumption 3, for every belief $b \in \Delta ( \mathcal { S } )$

$$
\vert \bar { V } _ { \theta } ^ { \star , S } ( b ) - F _ { \theta } ( m _ { g } ( b ) ) \vert \ \leq \ \frac { L _ { \theta } } { 2 } \mathrm { t r } \big ( \Sigma _ { g } ( b ) \big ) .\tag{78}
$$

Proof. By Jensen’s inequality applied to the pointwise maximum,

$$
\bar { V } _ { \theta } ^ { \star , S } ( b ) = \operatorname* { m a x } _ { a \in \mathcal { A } } \int _ { S } Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) \leq \int _ { S } V _ { \theta } ^ { \star , S } ( s ) b ( \mathrm { d } s ) .
$$

By Assumption 3, $V _ { \theta } ^ { \star , S } ( s ) = F _ { \theta } ( g ( s ) )$ , so

$$
\bar { V } _ { \theta } ^ { \star , S } ( b ) \ \leq \ \int _ { S } F _ { \theta } ( g ( s ) ) b ( \mathrm { d } s ) .\tag{79}
$$

Expand $F _ { \theta }$ in a second-order Taylor expansion around $m _ { g } ( b )$ . For each $s ,$ there exists $\xi _ { s } \in$ conv $( g ( \cal S ) )$ on the segment between $m _ { g } ( b )$ and $g ( s )$ such that

$$
\begin{array} { r } { F _ { \theta } ( g ( s ) ) = F _ { \theta } ( m _ { g } ( b ) ) + \nabla F _ { \theta } ( m _ { g } ( b ) ) ^ { \top } \big ( g ( s ) - m _ { g } ( b ) \big ) + \frac { 1 } { 2 } \big ( g ( s ) - m _ { g } ( b ) \big ) ^ { \top } \nabla ^ { 2 } F _ { \theta } ( \xi _ { s } ) \big ( g ( s ) - m _ { g } ( b ) \big ) . } \end{array}
$$

Integrating against b and using $\begin{array} { r } { \int _ { \cal S } ( g ( s ) - m _ { g } ( b ) ) b ( \mathrm { d } s ) = 0 } \end{array}$

$$
\int _ { S } F _ { \theta } ( g ( s ) ) b ( \mathrm { d } s ) - F _ { \theta } ( m _ { g } ( b ) ) = \frac { 1 } { 2 } \int _ { S } \bigl ( g ( s ) - m _ { g } ( b ) \bigr ) ^ { \top } \nabla ^ { 2 } F _ { \theta } ( \xi _ { s } ) \bigl ( g ( s ) - m _ { g } ( b ) \bigr ) b ( \mathrm { d } s ) .\tag{80}
$$

Since $\| \nabla ^ { 2 } F _ { \theta } ( \xi _ { s } ) \| _ { \mathrm { o p } } \leq L _ { \theta }$ , each integrand is bounded in absolute value by $L _ { \theta } \left\| g ( s ) - m _ { g } ( b ) \right\| ^ { 2 }$ , so equation 80 lies in $\left[ - \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) , \ \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) \right]$ . Combined with equation 79, we obtain the upper bound

$$
\begin{array} { r } { \bar { V } _ { \theta } ^ { \star , S } ( b ) \ \leq \ F _ { \theta } ( m _ { g } ( b ) ) + \frac { L _ { \theta } } { 2 } \operatorname { t r } \big ( \Sigma _ { g } ( b ) \big ) . } \end{array}\tag{81}
$$

For the lower bound, fix any $a \in { \mathcal { A } }$ . By Assumption 3, $Q _ { \theta } ^ { \star , S } ( s , a ) = F _ { \theta } ^ { a } ( g ( s ) )$ with $\| \nabla ^ { 2 } F _ { \theta } ^ { a } \| _ { \mathrm { o p } } \leq L _ { \theta }$ The same Taylor expansion gives

$$
\int F _ { \theta } ^ { a } ( g ( s ) ) b ( \mathrm { d } s ) \geq F _ { \theta } ^ { a } ( m _ { g } ( b ) ) - { \textstyle { \frac { L _ { \theta } } { 2 } } } \operatorname { t r } \bigl ( \Sigma _ { g } ( b ) \bigr ) .
$$

Taking max<sub>a</sub>:

$$
\begin{array} { r } { \bar { V } _ { \theta } ^ { \star , S } ( b ) = \displaystyle \operatorname* { m a x } _ { a } \int F _ { \theta } ^ { a } ( g ( s ) ) b ( \mathrm { d } s ) \geq \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) = F _ { \theta } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) . } \end{array}\tag{82}
$$

Remark 5. The upper half equation 81 follows from Jensen’s inequality alone and is used to verify the hypothesis of Proposition 9 in concrete settings.

Proposition 9 (Curvature bound on $\Xi _ { \theta } )$ . Under Assumption 3, for every belief $b \in \Delta ( \mathcal { S } )$

$$
\Xi _ { \theta } ( b ) : = \widetilde { V } _ { \theta } ^ { \star } ( b ) - \bar { V } _ { \theta } ^ { \star , S } ( b ) \ \leq \ L _ { \theta } \operatorname { t r } \bigl ( \Sigma _ { g } ( b ) \bigr ) .\tag{83}
$$

In particular, $\begin{array} { r } { \Xi _ { \theta } = \operatorname* { s u p } _ { b } \Xi _ { \theta } ( b ) \leq L _ { \theta } \operatorname* { s u p } _ { b } \operatorname { t r } ( \Sigma _ { g } ( b ) ) } \end{array}$

Proof. We bound $\Xi _ { \theta } ( b ) = \widetilde { V } _ { \theta } ^ { \star } ( b ) - \bar { V } _ { \theta } ^ { \star , S } ( b )$ for each belief b.

Upper bound on $\widetilde { V } _ { \boldsymbol { \theta } } ^ { \star } ( \boldsymbol { b } )$ . By Assumption 3, $V _ { \theta } ^ { \star , S } ( s ) = F _ { \theta } ( g ( s ) )$ where $F _ { \theta } = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a }$ has curvature $\leq L _ { \theta }$ . Applying the standard Taylor upper bound (as in Lemma 13) to $F _ { \theta } { } _ { \mathrm { : } }$

$$
\widetilde { V } _ { \theta } ^ { \star } ( b ) = \int F _ { \theta } ( g ( s ) ) b ( \mathrm { d } s ) \leq F _ { \theta } ( m _ { g } ( b ) ) + \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) .\tag{84}
$$

Since $F _ { \theta } ( m _ { g } ( b ) ) = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( m _ { g } ( b ) )$ , this gives $\begin{array} { r } { \widetilde { V } _ { \theta } ^ { \star } ( b ) \leq \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( m _ { g } ( b ) ) + \frac { L _ { \theta } } { 2 } \mathrm { t r } \big ( \Sigma _ { g } ( b ) \big ) } \end{array}$

Lower bound on $\bar { V } _ { \theta } ^ { \star , S } ( b )$ . By Assumption 3, $Q _ { \theta } ^ { \star , S } ( s , a ) = F _ { \theta } ^ { a } ( g ( s ) )$ for each a. For any fixed $^ { a , }$

$$
\int Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) = \int F _ { \theta } ^ { a } ( g ( s ) ) b ( \mathrm { d } s ) \geq F _ { \theta } ^ { a } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \mathrm { t r } ( \Sigma _ { g } ( b ) ) ,
$$

where the inequality is the Taylor lower bound (each $F _ { \theta } ^ { a }$ has curvature $\leq L _ { \theta } )$ . Taking the maximum over a:

$$
\bar { V } _ { \theta } ^ { \star , S } ( b ) = \operatorname* { m a x } _ { a } \int F _ { \theta } ^ { a } ( g ( s ) ) b ( \mathrm { d } s ) \geq \operatorname* { m a x } _ { a } \Big [ F _ { \theta } ^ { a } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \mathrm { t r } ( \Sigma _ { g } ( b ) ) \Big ] = \operatorname* { m a x } _ { a } F _ { \theta } ^ { a } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \mathrm { t r } ( \Sigma _ { g } ( b ) ) .
$$

Combining.

$$
\Xi _ { \theta } ( b ) = \widetilde { V } _ { \theta } ^ { \star } ( b ) - V _ { \theta } ^ { \star , S } ( b ) \leq \left[ \operatorname* { m a x } _ { a } F _ { \theta } ^ { \alpha } ( m _ { g } ( b ) ) + \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) \right] - \left[ \operatorname* { m a x } _ { a } F _ { \theta } ^ { \alpha } ( m _ { g } ( b ) ) - \frac { L _ { \theta } } { 2 } \operatorname { t r } ( \Sigma _ { g } ( b ) ) \right] = L _ { \theta } \operatorname { t r } ( \Sigma _ { g } ( b ) ) .
$$

## D.9 Case Studies

Case 1: Discounted partially observed LQG. Consider linear-Gaussian dynamics $S _ { t + 1 } =$ $\Phi S _ { t } + \Gamma A _ { t } + W _ { t } , O _ { t } = H S _ { t } + \eta _ { t }$ , with $W _ { t } \sim { \mathcal { N } } ( 0 , \Sigma _ { W } ) , \eta _ { t } \sim { \mathcal { N } } ( 0 , \Sigma _ { \eta } )$ i.i.d., quadratic reward $\boldsymbol { r } ( s , a ) = - ( s ^ { \top } Q _ { \mathrm { c } } s + a ^ { \top } R a ) , Q _ { \mathrm { c } } \succeq 0 , R \succ 0$ . Assume $( \Phi , \Gamma )$ is stabilizable and $( \Phi , H )$ is detectable. For $\mathrm { L Q G }$ , we obtain the closed-form expression $\Xi _ { \theta } = \mathrm { t r } \big ( ( M - P ) \bar { \Sigma } \big )$ , where $P$ solve the Riccati equation, $M = Q _ { c } + \gamma \Phi ^ { \top } P \Phi$ , and Σ<sup>¯</sup> is the steady-state Kalman predictive covariance. When $\Gamma = 0$ we have $M = P$ and $\Xi _ { \theta } = 0$ . Note that for LQG, the QMDP policy is always optimal.

Corollary 3 (LQG). Let $M : = Q _ { \mathrm { c } } + \gamma \Phi ^ { \top } P \Phi$ where $P \succeq 0$ solves the discounted algebraic Riccati equation, and $l e t \bar { \Sigma } _ { \mathrm { p o s t } }$ be the steady-state Kalman posterior (filtered) covariance satisfying $\bar { \Sigma } _ { \mathrm { p o s t } } =$ $\bar { \Sigma } - \bar { \Sigma } H ^ { \top } ( H \bar { \Sigma } H ^ { \top } + \dot { \Sigma } _ { \eta } )$ <sup>−1</sup>HΣ<sup>¯</sup> with $\bar { \Sigma } : = \Phi \bar { \Sigma } _ { \mathrm { p o s t } } \Phi ^ { \top } + \Sigma _ { W }$ the steady-state predictive covariance. Then $\Xi _ { \theta } = \mathrm { t r } \big ( ( M - P ) \bar { \Sigma } \big )$ and the $Q M D P$ policy is belief-MDP optimal: $\operatorname* { s u p } _ { b } \big [ V _ { \boldsymbol { \theta } } ^ { \star , B } ( \boldsymbol { b } ) - V _ { \boldsymbol { \theta } } ^ { \pi _ { \mathrm { Q } } , B } ( \boldsymbol { b } ) \big ] = 0$

The QMDP action equals the certainty-equivalent LQR control $a = K m$ , which is belief-MDP optimal by the separation principle. When $\Gamma = 0$ , the DARE gives $M = P ,$ , so $\Xi _ { \theta } = 0$ and the QMDP value and the belief-MDP value coincide. When $\Gamma \neq 0 , M - P \succeq 0$ is generically nonzero, so $\Xi _ { \theta } > 0$ . In that case the QMDP policy remains optimal, but the QMDP value underestimates the belief-MDP value by $\mathrm { t r } ( ( M - P ) { \bar { \Sigma } } )$

Remark. LQG has unbounded reward and $V _ { \mathrm { m a x } } = \infty$ , so it lies outside the standing assumption $| r ( s , a ) | \leq R _ { \mathrm { m a x } }$ of Theorem 1. The corollary illustrates the structural analysis (Jensen gap and policy optimality) but does not directly instantiate the regret bound.

Case 2: Identifiable contextual MDP. Consider an episodic contextual MDP [24] with a fixed latent context $C \in { \mathcal { C } }$ sampled once per episode and Markov dynamics $S _ { t + 1 } \sim P _ { c } ( \cdot \mid S _ { t } , A _ { t } )$ $r _ { t } = r _ { c } ( S _ { t } , A _ { t } )$ given $C = c .$ Assume (i) there exists an estimator $\widehat { C } _ { t } = \widehat { C } _ { t } ( H _ { t } )$ with $\mathrm { P r } ( \widehat { C } _ { t } \neq C ) \leq \rho _ { t }$ $\rho _ { t } \downarrow 0 ;$ and (ii) su $\begin{array} { r } { \mathrm { p } _ { c \in \mathcal { C } } \| V _ { c } ^ { \star , S } \| _ { \infty } \leq V _ { \mathrm { m a x } } } \end{array}$ . Then $\Xi _ { \theta , t } ~ \leq ~ 2 V _ { \operatorname* { m a x } } \rho _ { t }$ and s $\begin{array} { r } { \operatorname* { u p } _ { b } \big [ V _ { \boldsymbol { \theta } } ^ { \star , B } ( b ) - V _ { \boldsymbol { \theta } } ^ { \pi _ { \mathrm { Q } } , B } ( b ) \big ] ~ \leq ~ } \end{array}$ $4 \gamma V _ { \operatorname* { m a x } } \rho _ { t } / ( 1 - \gamma ) ^ { 2 }$ . As the context becomes identifiable $( \rho _ { t } \to 0 )$ , both the structural residual and the QMDP policy suboptimality vanish. For the structural bias term $\begin{array} { r } { \sum _ { t } \Xi _ { t } \le 2 V _ { \operatorname* { m a x } } \sum _ { t } \rho _ { t } } \end{array}$ to be $o ( \sqrt { T } )$ , we need $\textstyle \sum _ { t = 1 } ^ { T } \rho _ { t } = o ( { \sqrt { T } } )$ . This result also suggests a schedule for the planning depth d that decays with $t ,$ reflecting diminishing structural bias.

Proofs. The fully observed value function is $V _ { \theta } ^ { \star , S } ( s ) = - s ^ { \top } P s - c _ { 0 }$ , where $P \succeq 0$ solves the discounted algebraic Riccati equation (DARE). The constant $c _ { 0 }$ is determined by substituting into the Bellman equation:

$$
- s ^ { \top } P s - c _ { 0 } = \operatorname* { m a x } _ { a } \{ r ( s , a ) + \gamma \mathbb { E } [ V ^ { \star , S } ( s ^ { \prime } ) ] \} ,
$$

which gives $c _ { 0 } = \gamma [ \mathrm { t r } ( P \Sigma _ { W } ) + c _ { 0 } ]$ from the noise covariance term, $_ \mathrm { y }$ ielding $c _ { 0 } = \gamma \mathrm { t r } ( P \Sigma _ { W } ) / ( 1 - \gamma )$ Hence

$$
\begin{array} { r } { Q _ { \theta } ^ { \star , S } ( s , a ) = - ( s ^ { \top } Q _ { \mathsf { c } } s + a ^ { \top } R a ) - \gamma ( \Phi s + \Gamma a ) ^ { \top } P ( \Phi s + \Gamma a ) - \gamma \mathrm { t r } ( P \Sigma _ { W } ) - \gamma c _ { 0 } . } \end{array}
$$

For belief $b = \mathcal { N } ( m , \Sigma )$ ),

$$
\int Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) = - m ^ { \top } Q _ { \mathsf { c } } m - a ^ { \top } R a - \gamma ( \Phi \boldsymbol { m } + \Gamma a ) ^ { \top } P ( \Phi \boldsymbol { m } + \Gamma a ) - \mathrm { t r } ( M \Sigma ) - \gamma \mathrm { t r } ( P \Sigma _ { W } ) - \gamma c _ { 0 } ,
$$

where $M : = Q _ { \mathrm { c } } + \gamma \Phi ^ { \top } P \Phi$ . All Σ-dependent terms are independent of $^ { a , }$ so the QMDP action is

$$
\arg \operatorname* { m a x } _ { a } \int Q _ { \theta } ^ { \star , S } ( s , a ) b ( \mathrm { d } s ) = \arg \operatorname* { m a x } _ { a } \big \{ - a ^ { \top } R a - \gamma ( \Phi m + \Gamma a ) ^ { \top } P ( \Phi m + \Gamma a ) \big \} = K m ,
$$

with

$$
K = - \gamma ( R + \gamma \Gamma ^ { \top } P \Gamma ) ^ { - 1 } \Gamma ^ { \top } P \Phi .
$$

By the separation principle for discounted LQG (stabilizable/detectable), the belief-MDP optimal action is also $K m ,$ so QMDP and belief-optimal policies coincide. This gives part 2.

Structural residual. We compute $\Xi _ { \theta } ( b ) = \widetilde { V } _ { \theta } ^ { \star } ( b ) - \bar { V } _ { \theta } ^ { \star , S } ( b )$ directly. From the expansion above,

$$
\bar { V } _ { \theta } ^ { \star , S } ( b ) = - m ^ { \top } P m - \mathrm { t r } ( M \Sigma ) - c _ { 0 } .
$$

Meanwhile,

$$
\widetilde { V } _ { \theta } ^ { \star } ( b ) = \int V _ { \theta } ^ { \star , S } ( s ) b ( \mathrm { d } s ) = \int ( - s ^ { \top } { \cal P } s - c _ { 0 } ) b ( \mathrm { d } s ) = - m ^ { \top } { \cal P } m - \mathrm { t r } ( { \cal P } \Sigma ) - c _ { 0 } .
$$

Therefore $\Xi _ { \theta } ( b ) = \mathrm { t r } ( ( M - P ) \Sigma )$ . Since

$$
M - P = \gamma ^ { 2 } \Phi ^ { \top } P \Gamma ( R + \gamma \Gamma ^ { \top } P \Gamma ) ^ { - 1 } \Gamma ^ { \top } P \Phi \succeq 0 ,
$$

this is nonnegative. Taking the supremum over beliefs generated by the stationary Kalman filter yields

$$
\Xi _ { \theta } \leq \mathrm { t r } ( ( M - P ) \bar { \Sigma } ) ,
$$

where $\bar { \Sigma }$ is the steady-state Kalman predictive covariance.

When $\Gamma = 0$ , the DARE gives $M = P , { \mathrm { s o } } \Xi _ { \theta } = 0 .$

Structural residual $( p a r t ~ 1 )$ . Let $\mu _ { t } ( \cdot ) : = \operatorname* { P r } ( C = \cdot \mid H _ { t } )$ be the posterior on context and $c ^ { \star }$ the true context. By the definition of the Bayes classifier

$$
\hat { C } _ { t } ^ { \mathrm { M A P } } = \arg \operatorname* { m a x } _ { c } \mu _ { t } ( c ) ,
$$

we have that

$$
\mathbb { E } [ 1 - \operatorname* { m a x } _ { c } \mu _ { t } ( c ) ] \leq \operatorname* { P r } ( \widehat { C } _ { t } \neq C ) \leq \rho _ { t } .
$$

For any bounded $f : \mathcal { C } \to [ - V _ { \mathrm { m a x } } , V _ { \mathrm { m a x } } ]$ , let $c _ { t } ^ { \mathrm { M A P } } = \arg \operatorname* { m a x } _ { c } \mu _ { t } ( c )$ . Then

$$
| \sum _ { c } \mu _ { t } ( c ) f ( c ) - f ( c _ { t } ^ { \mathrm { M A P } } ) | \leq 2 V _ { \mathrm { m a x } } ( 1 - \operatorname* { m a x } _ { c } \mu _ { t } ( c ) ) ,
$$

since the total posterior mass on contexts other than $c _ { t } ^ { \mathrm { M A P } }$ is $1 - \operatorname* { m a x } _ { c } \mu _ { t } ( c )$ . The structural residual is bounded directly (without the triangle inequality):

$$
\begin{array} { r l } { \Xi _ { \theta , t } ( b ) = \displaystyle \sum _ { c } \mu _ { t } ( c ) \operatorname* { m a x } _ { a } Q _ { c } ^ { \star , S } ( s , a ) - \displaystyle \operatorname* { m a x } _ { a } \sum _ { c } \mu _ { t } ( c ) Q _ { c } ^ { \star , S } ( s , a ) } \\ { \leq \displaystyle \sum _ { c } \mu _ { t } ( c ) \operatorname* { m a x } _ { a } Q _ { c } ^ { \star , S } ( s , a ) - \displaystyle \sum _ { c } \mu _ { t } ( c ) Q _ { c } ^ { \star , S } ( s , a _ { c _ { t } ^ { \mathrm { M A P } } } ^ { \star } ) } \\ { = \displaystyle \sum _ { c \neq c ^ { \mathrm { M A P } } } \mu _ { t } ( c ) \big [ \operatorname* { m a x } _ { a } Q _ { c } ^ { \star , S } ( s , a ) - Q _ { c } ^ { \star , S } ( s , a _ { c _ { t } ^ { \mathrm { M A P } } } ^ { \star } ) \big ] } \\ { \leq \displaystyle \sum _ { c \neq c ^ { \mathrm { M A P } } } \mu _ { t } ( c ) \cdot 2 V _ { \mathrm { m a x } } = \ : 2 V _ { \mathrm { m a x } } \big ( 1 - \operatorname* { m a x } _ { c } \mu _ { t } ( c ) \big ) , } \end{array}
$$

where $a _ { c _ { t } ^ { \mathrm { M A P } } } ^ { \star } : = \arg \operatorname* { m a x } _ { a } Q _ { c _ { t } ^ { \mathrm { M A P } } } ^ { \star , S } ( s , a )$ and the first inequality uses

$$
\operatorname* { m a x } _ { a } \sum _ { c } \mu _ { t } ( c ) Q _ { c } ( s , a ) \geq \sum _ { c } \mu _ { t } ( c ) Q _ { c } ( s , a _ { c _ { t } ^ { \mathrm { M A P } } } ^ { \star } ) ,
$$

Taking expectations yields

$$
\mathbb { E } [ \Xi _ { \theta , t } ] \le 2 V _ { \operatorname* { m a x } } \mathbb { E } [ 1 - \operatorname* { m a x } _ { c } \mu _ { t } ( c ) ] \le 2 V _ { \operatorname* { m a x } } \rho _ { t } .
$$

Policy suboptimality (part 2). From Proposition 2,

$$
\lVert Q _ { \theta } ^ { \star , B } - Q _ { \theta , 0 } \rVert _ { \infty } \leq \frac { \gamma } { 1 - \gamma } \Xi _ { \theta , t } .
$$

For any belief b, let

$$
a ^ { \star } = \arg \operatorname* { m a x } _ { a } Q _ { \theta } ^ { \star , B } ( b , a ) , \qquad a _ { \mathrm { Q } } = \pi _ { \mathrm { Q } } ( b ) .
$$

Then

$$
Q _ { \theta } ^ { \star , B } ( b , a ^ { \star } ) - Q _ { \theta } ^ { \star , B } ( b , a _ { \mathrm { Q } } ) \leq 2 \| Q _ { \theta } ^ { \star , B } - Q _ { \theta , 0 } \| _ { \infty } \leq \frac { 2 \gamma } { 1 - \gamma } \Xi _ { \theta , t } ,
$$

where the first inequality uses

$$
a _ { \mathrm { Q } } = \arg \operatorname* { m a x } _ { a } Q _ { \theta , 0 } ( b , a ) .
$$

By standard policy-loss arguments (telescoping the per-step suboptimality),

$$
\operatorname* { s u p } _ { b } [ V _ { \theta } ^ { \star , B } ( b ) - V _ { \theta } ^ { \pi _ { \mathrm { Q } } , B } ( b ) ] \leq \frac { 1 } { 1 - \gamma } \cdot \frac { 2 \gamma } { 1 - \gamma } \Xi _ { \theta , t } \leq \frac { 4 \gamma V _ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \rho _ { t } .
$$

## E Empirical Validation of the Regret Decomposition

Instance. The mixing Tiger of Section 5 is a variant of Tiger [19] that we constructed for diagnostic purposes, and it is not a standard benchmark. It has two latent states, three actions (listen, open-left, open-right), and two observations that report the state correctly with probability 0.85 under every action. Listening mixes the state, $P ( \cdot \mid s , \mathrm { l i s t e n } ) = 0 . 9 5 \delta _ { s } + 0 . 0 5 \mathrm { U n i f }$ , opening a door resets it uniformly, and the standard Tiger rewards $( - 1 , + 1 0 , - 1 0 0 )$ are mapped afinely to (0.9, 1, 0). Relative to the uniform measure every transition density lies in [0.05, 1.95], so Assumption 1 holds with $\varepsilon _ { \mathrm { m i x } } = 0 . 0 5 , g _ { \mathrm { m i n } } = 0 . 1 5$ , and $g _ { \mathrm { m a x } } = 0 . 8 5$ . The rewards lie in [0, 1], the discount factor is $\gamma = 0 . 8$ and $\Xi _ { \theta } = 0 . 1 0$

Isolating each term. Except in the estimation run, the planner and filter use the true model, so each experiment contains a single source of error. Belief-MDP quantities are computed by value iteration on a grid of 4001 belief points with linear interpolation. Particle filtering: a bootstrap SIR filter with J particles tracks the belief for 40 listening steps, and the error is the absolute error of its estimate of $\mathrm { P r } ( S = 1 )$ against the exact filter, averaged over steps (2000 replications). Monte Carlo: the root value $Q _ { \theta , 3 } ( b , a )$ is estimated from N sampled first transitions, $r ( s , a ) + \gamma \bar { V } _ { \theta , 2 } ^ { \star , S } ( \tau ( b , a , o ) )$ with the depth-2 continuation computed exactly and fixed before sampling, so the N returns are conditionally i.i.d. The error is the root-mean-squared error over five beliefs and all three actions, computed from 2000 replications. This design isolates the $N ^ { - 1 / 2 }$ fluctuation of Proposition 3. It does not include the pooling of leaf particles across simulations or the DP-backup bias of the full estimator, for which that proposition does not require independent returns. Structural: $\| \mathcal { T } _ { \theta } ^ { d } \bar { V } _ { \theta } ^ { \star , S } - V _ { \theta } ^ { \star , B } \| _ { \infty }$ is computed exactly for $d = 0 , \ldots , 1 2$ and compared with the envelope $\gamma ^ { d + 1 } { \Xi } _ { \theta } / ( 1 - \stackrel { . } { \gamma } )$ of Proposition 2. Rates are least-squares slopes of the log mean error against the log budget, and the structural contraction is the fitted per-level ratio. Figure 10 shows all three terms. The budgeting rule of Section 5 reads of, for each threshold, the smallest tested J, N, and d whose isolated error falls below it.

![](images/b46b44f6d5339b5cd3c21e85b20c0fe1d845050385b214b33ae55563c27707f7.jpg)

![](images/3f5c1e7ada848532cfdd113d290ae05b08ce889ddc42590f9dfdce8708abdd0b.jpg)

![](images/73acf5eb970858184095c6674b252d5e4de906a25c2b1257b0ac23870ff4b322.jpg)  
Figure 10: The three computational terms on the mixing Tiger. (a) Filtering error versus particles J (fitted slope −0.59). (b) Root-Q RMSE of the frozen-leaf estimator versus returns N (fitted slope $- 0 . 5 0 )$ . (c) Exact structural error versus belief lookahead depth d. The error contracts by a factor $0 . 8 0 = \gamma$ per level and stays below $\gamma ^ { d + 1 } { \Xi _ { \theta } } / { ( 1 - \gamma ) }$ . Means over 2000 replications in (a, b).

Estimation term. Here the model is learned online by maximum likelihood. In Figure $\mathrm { { 1 1 ( a ) } }$ the optimization error $\varepsilon _ { \mathrm { o p t } }$ decays as n<sup>−0.59</sup> and the model’s total-variation error as $n ^ { - 0 . 5 5 }$ in the number of model-estimation samples n. In panel (b), the cumulative estimation term $\textstyle \sum _ { t \leq T } \varepsilon _ { \mathrm { o p t } } ( t )$ grows as $T ^ { 0 . 5 7 }$ , close to $\sqrt { T }$ . With the known model, the same pipeline accumulates about $1 . 3 \cdot 1 0 ^ { - 4 }$ per round.

Structural term across instances. In Figure 11(c), the structural error of 32 instances stays below the envelope $\gamma ^ { d + 1 } { \Xi } / ( 1 - \gamma )$ and contracts by a factor of about $\gamma$ per level, with fitted ratios of 0.798–0.800.

![](images/645ad5958609c52f9a997330ec45f694b4ed522af1c38eb687d954503a011b13.jpg)

![](images/6d9f4e0f1b66504d0bf831f1c6a834227c27df018d412e51b6b8c7030052763a.jpg)

![](images/34e987e2b685e005501e54346a7b3152dd6f9154abaa2cf376a2acd5ba74a251.jpg)  
Figure 11: Estimation and structural terms. (a) $\varepsilon _ { \mathrm { o p t } }$ and the model’s total-variation error versus model-estimation samples $n _ { t } ,$ , with an $n ^ { - 1 / 2 }$ reference (fitted slopes −0.59 and −0.55). (b) Cumulative estimation term $\textstyle \sum _ { t } \varepsilon _ { \mathrm { o p t } } ( t )$ versus rounds $T$ , with a $T ^ { 1 / 2 }$ reference (fitted slope 0.57). (c) Structural error $\| \bar { V } _ { d } - V ^ { \star } \| _ { \infty }$ of 32 instances versus belief lookahead depth $d ,$ below the envelope $\gamma ^ { d + 1 } { \Xi } / ( 1 - \gamma )$ (per-level contraction 0.798–0.800).

## F Assumption Stress Tests

The two diagnostics below are intended to illustrate the mechanisms behind Assumption 1 and the structural term, and are not meant as benchmarks. Figure 12 plots both.

QMDP information bias (active sensing). In this Light–Dark-type problem, information can be obtained only by giving up immediate reward. Two latent contexts persist with mixing rate 0.02. Two exploit actions give reward 1 when they match the context and reveal nothing, while a sense action gives no reward and reveals the context with accuracy $q .$ The discount factor is $\gamma = 0 . 9$ . Table 9 reports the normalized return $( 1 - \gamma ) V ^ { \pi _ { d } , B } ( b _ { 0 } )$ at the uniform belief of the greedy depth-d policy, computed exactly on the belief grid. Because sensing has no immediate reward and the QMDP leaf values the state as if it will be revealed, depths 0 and 1 never sense and attain a normalized return of 0.5. Depth 3 comes within 0.006 of the optimum for $q \geq 0 . 7 5$ , and depth 2 does so for $q \geq 0 . 8 5$ . At $q = 0 . 6 5$ , even depth 3 does not sense.

Table 9: Active sensing: normalized return of the depth-d policy (exact computation).
<table><tr><td>Sensing accuracy q</td><td>QMDP  $( d = 0 )$ </td><td> $d = 1$ </td><td> $d = 2$ </td><td> $d = 3$ </td><td>Optimum</td></tr><tr><td>0.55</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td></tr><tr><td>0.65</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.565</td></tr><tr><td>0.75</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.641</td><td>0.643</td></tr><tr><td>0.85</td><td>0.500</td><td>0.500</td><td>0.717</td><td>0.717</td><td>0.722</td></tr><tr><td>0.95</td><td>0.500</td><td>0.500</td><td>0.793</td><td>0.794</td><td>0.798</td></tr></table>

Bounded-density violation (item search). An item lies in one of $K = 3 2$ boxes under a uniform prior, and the observation reveals the box exactly, so $g _ { \mathrm { m i n } } = 0$ . A particle set of size J drawn from the prior has all weights zero whenever no particle lies in the revealed box, which happens with probability $( 1 - 1 / K ) ^ { J }$ . In that case the particle policy chooses the wrong box, so this is also the decision-failure probability. Table 10 compares the empirical collapse rate over 10,000 replications with this formula. The efective sample size does not detect this failure. At $g _ { \mathrm { m i n } } = 0$ the mean ESS/J is about $1 / K$ for every J. With a small floor $g _ { \mathrm { m i n } } = 1 0 ^ { - 6 }$ , a particle set that misses the revealed box has equal weights and $\mathrm { E S S } / J = 1$ , so the mean $\mathrm { E S S } / J$ is 0.81 at $J = 8 .$ , even though 77% of the runs miss.

Table 10: Item search with $K = 3 2$ and $g _ { \mathrm { m i n } } = 0 \colon$ weight collapse versus particles $J .$
<table><tr><td>J</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>Empirical collapse rate</td><td>.773</td><td>.602</td><td>.360</td><td>.128</td><td>.017</td></tr><tr><td>Analytic  $( 1 - 1 / K ) ^ { J }$ </td><td>.776</td><td>.602</td><td>.362</td><td>.131</td><td>.017</td></tr><tr><td>Mean  $\mathrm { E S S } / J$ </td><td>.032</td><td>.031</td><td>.031</td><td>.032</td><td>.031</td></tr></table>

![](images/288d9b3e8d91746d3a91a6233744d48287b713e35ad1d1487ee6a62afb2e3690.jpg)

![](images/9da2521431efd6a0ac092eb275ed77386ec4471070a4951fcae372f1bfae4fea.jpg)

![](images/30fb86a92b54a42b82e35251b24d5e95b7ecc8a0a2581274ad04c40276d47d15.jpg)  
Figure 12: Stress tests illustrating the two mechanisms. (a) In active sensing, depths 0–1 never sense, while depths 2–3 approach the belief-optimal return when sensing is informative. (b) With exact observations over K boxes, the one-step filtering error grows when the particle set is too small to cover the observed box. (c) At $g _ { \mathrm { m i n } } = 0$ , weight collapse matches $\left( 1 - 1 / K \right) ^ { J } \left( K = 3 2 \right)$ , which is also the decision-failure probability. Means over 10,000 replications in (b, c).

## G Planner Comparison at Depth $d > 0$

Here we run the analyzed MCTQ at $d > 0$ on two finite POMDPs and compare it with POMCP [34] and a sparse particle-belief tree (Sparse-PFT) [26].

Setup. The environments are the active-sensing problem of Appendix F with $\gamma = 0 . 9$ and the mixing Tiger of Appendix E. The agent tracks the exact belief, and all planners share the same 256-particle root sample, with common random numbers. We report the mean $\pm \ \mathrm { s . e }$ . over 3 seeds of 40 episodes each. Calls per decision are counted in generative-model steps, of which MCTQ uses $| A | N d .$

Results. Depth roughly determines the value, as predicted by Proposition 2. At a fixed budget, performance is U-shaped in d. On the mixing Tiger, QMDP achieves the belief optimum (both 4.586), with every planner landing within 0.06 of it.

Table 11: Active sensing $( \gamma = 0 . 9 )$ : return, mean $\pm \ \mathrm { s . e }$ . over three seeds of 40 episodes. The belief optimum scores $7 . 4 5 \pm 0 . 2 0$ and the exact QMDP policy $5 . 5 4 \pm 0 . 3 2$
<table><tr><td>Planner</td><td>d</td><td>Calls per decision</td><td>Return</td></tr><tr><td>MCTQ, uniform</td><td>0</td><td>0</td><td> $5 . 0 4 \pm 0 . 1 1$ </td></tr><tr><td>MCTQ, uniform (= UCB1)</td><td>1</td><td>768</td><td> $5 . 0 6 \pm 0 . 1 3$ </td></tr><tr><td>POMCP</td><td></td><td>768  / 3072</td><td> $5 . 4 7 ~ / ~ 5 . 5 2$ </td></tr><tr><td> ${ \mathrm { M C T Q } } ,$  uniform</td><td>2</td><td>1536</td><td> $6 . 3 7 \pm 0 . 2 7$ </td></tr><tr><td>MCTQ, UCB1</td><td>2</td><td>1536</td><td> $6 . 9 6 \pm 0 . 1 1$ </td></tr><tr><td>Sparse-PFT,  $J = 6 4$ </td><td>2</td><td>2688</td><td> $7 . 1 8 \pm 0 . 0 8$ </td></tr><tr><td> ${ \mathrm { M C T Q } } ,$  uniform</td><td>2</td><td>6144</td><td> $7 . 3 0 \pm 0 . 1 2$ </td></tr><tr><td> ${ \mathrm { M C T Q } } ,$  uniform  $/ \mathrm { \ U C B 1 }$ </td><td>3</td><td>2304</td><td> $5 . 6 9 \ / \ 6 . 7 4$ </td></tr></table>

## H Ant: Correction Screen, Full-Horizon Runs, and Runtime

Correction screen. Unlike in the submitted version, Ant training is no longer stopped early, and Figure 3 shows the full Ant horizon. Training the released configuration over the full horizon revealed a systematic late collapse in both observation ablations. We report this result here and examined three choices in the released Ant configuration: it freezes the model after its random-data warm start, although the main text describes online updates; it uses AdamW weight decay 0.1; and it performs one policy update per environment step. At learning rate $3 \cdot 1 0 ^ { - 4 }$ , weight decay alone multiplies the parameters by about 0.22 over 50,000 optimizer steps. The controls in Table 12 show that the first two corrections are not suficient and that the collapse is prevented by reducing the update-to-data ratio (UTD). We screened corrections on position-hidden Ant (seed 0) using a fixed 15k–20k selection window (Figure 13). Neither online updates nor corrected decay alone prevents collapse. We selected UTD $1 / 1 0$ before launching the final multi-seed comparison. Table 5 lists the corrected configuration.

Table 12: Correction screen on position-hidden Ant (seed 0): mean 10-episode rolling return over 15k–20k steps and at 20.1k steps.
<table><tr><td>Variant</td><td>Online model</td><td>Decay</td><td>UTD</td><td>15k-20k mean</td><td>At 20.1k</td></tr><tr><td>Submitted</td><td>No</td><td>0.1</td><td>1</td><td>-279</td><td>-563</td></tr><tr><td>Online model only</td><td>Yes</td><td>0.1</td><td>1</td><td>-378</td><td>-834</td></tr><tr><td>Online model + corrected decay</td><td>Yes</td><td> $1 0 ^ { - 4 }$ </td><td>1</td><td>-259</td><td>-505</td></tr><tr><td> $\mathrm { C o r r e c t e d } + \mathrm { U T D \ 1 } / 4$ </td><td>Yes</td><td> $1 0 ^ { - 4 }$ </td><td> $1 / 4$ </td><td>902</td><td>881</td></tr><tr><td> $\mathrm { C o r r e c t e d } + \mathrm { U T D \ 1 / 1 0 }$ </td><td>Yes</td><td> $1 0 ^ { - 4 }$ </td><td> $1 / 1 0$ </td><td>980</td><td>975</td></tr></table>

Full horizon. All six corrected runs reach 49,200 steps. In the prespecified 40k–49.2k late window, corrected PF-MCTQ scores $8 9 3 \pm 1 7 7$ with position hidden and $8 2 2 \pm 2 6 8$ with velocity hidden (mean and 95% Student-t interval over three seeds). The submitted configuration, run with two seeds, scores −396 and −518, respectively. Independent final 10-episode evaluations are $8 6 2 \pm 1 7 8$ and $7 6 7 \pm 3 2 3$ for the corrected method, compared with means of −751 and −625 for the submitted configuration. Table 13 also reports the five-seed recurrent-SAC and GRU-PPO comparisons, and Table 14 gives PF-MCTQ at fixed checkpoints. All reported values are measured, and no curve or table entry is extrapolated or forward-filled. The submitted configuration is not asymptotically stable, while the targeted update-ratio correction prevents catastrophic collapse across the tested seeds.

![](images/c66ad7b3f66af1ceaca6d484173cf3a2f26757cf623b64d81c78a5d165d7885c.jpg)  
Figure 13: Correction screen on position-hidden Ant (seed 0). “Corrected” denotes online model updates with weight decay $1 0 ^ { - 4 } ;$ ; UTD $1 / k$ means one policy update per k environment steps. The shaded region is the 15k–20k selection window, and the dotted line marks the original 3k cutof.

Table 13: Ant return over the prespecified 40k–49.2k window. Values with ± are means and 95% Student-t intervals. For the submitted configuration’s two complete seeds, parentheses show both seed values. Recurrent baselines have five seeds.
<table><tr><td>Hidden variable</td><td>Method</td><td>Seeds</td><td>Late return</td><td>Final evaluation</td></tr><tr><td>Position</td><td>PF-MCTQ, corrected</td><td>3</td><td> $8 9 3 \pm 1 7 7$ </td><td> $8 6 2 \pm 1 7 8$ </td></tr><tr><td>Position</td><td>PF-MCTQ, submitted</td><td>2</td><td>-396 (-506, -287)</td><td>-751 (−1055, -448)</td></tr><tr><td>Position</td><td>Recurrent SAC</td><td>5</td><td>-123 ± 47</td><td>not recorded</td></tr><tr><td>Position</td><td>GRU-PPO</td><td>5</td><td>599 ± 147</td><td>not recorded</td></tr><tr><td>Velocity</td><td>PF-MCTQ, corrected</td><td>3</td><td>822 ± 268</td><td>767 ± 323</td></tr><tr><td>Velocity</td><td>PF-MCTQ, submitted</td><td>2</td><td>-518 (−513, -522)</td><td>-625 (-511, -739)</td></tr><tr><td>Velocity</td><td>Recurrent SAC</td><td>5</td><td>593 ± 154</td><td>not recorded</td></tr><tr><td>Velocity</td><td>GRU-PPO</td><td>5</td><td> $6 6 9 \pm 3 7$ </td><td>not recorded</td></tr></table>

The checkpoint values below are the logged 10-episode rolling training return at epoch ends, so every value is measured.

Runtime. Table 15 reports wall-clock cost on one RTX 3090 with position hidden. Training time is end-to-end (data collection and all updates, excluding the final evaluation). PF-MCTQ rates come from full 49.2k-step runs and the baseline rates from 3k-step runs with the same 300-step epochs. Initialization is the one-time cost before the first environment step, and for PF-MCTQ it is dominated by the 2000 model warm-start updates.

Table 14: Ant checkpoints. Corrected: mean and 95% Student-t half-width over three seeds $( t _ { 2 , 0 . 9 7 5 } = 4 . 3 0 )$ . Submitted: mean of the two complete seeds, with both seed values in parentheses. The last two rows are the mean over the prespecified 40k–49.2k window and a final 10-episode evaluation.
<table><tr><td rowspan="2">Env. steps</td><td colspan="2">Corrected (3 seeds)</td><td colspan="2">Submitted (2 seeds)</td></tr><tr><td></td><td>Pos. hidden Vel. hidden</td><td>Pos. hidden</td><td>Vel. hidden</td></tr><tr><td>3k</td><td> $8 2 5 \pm 1 4 6$ </td><td> $8 8 2 \pm 6 0$ </td><td>908 (945, 871)</td><td>897 (925, 870)</td></tr><tr><td>10.2k</td><td> $9 3 1 \pm 6 1$ </td><td> $9 3 4 \pm 1 1$ </td><td>600 (659, 541)</td><td>435 (606, 264)</td></tr><tr><td>20.1k</td><td> $9 5 9 \pm 3 6$ </td><td> $9 6 2 \pm 1 5$ </td><td>-911 (−563, -1260)</td><td>-620 (-697, -542)</td></tr><tr><td>49.2k</td><td> $8 6 8 \pm 2 4 9$ </td><td> $8 1 6 \pm 2 4 1$ </td><td>-448 (-678,-217)</td><td>-648 (-708, -589)</td></tr><tr><td>40k–49.2k mean</td><td>893 ± 177</td><td>822 ± 268</td><td>-396 (-506, -287)</td><td>-518 (−513, -522)</td></tr><tr><td>Final evaluation</td><td>862 ± 178</td><td> $7 6 7 \pm 3 2 3$ </td><td>-751 (−1055, -448)</td><td>-625 (−511, -739)</td></tr></table>

Table 15: Wall-clock cost on Ant with the released code (position hidden, one RTX 3090).
<table><tr><td>Method</td><td>Initialization (s)</td><td>Training s / 1k steps</td><td>Steps  $/ \mathrm { ~ s ~ }$ </td></tr><tr><td>PF-MCTQ (corrected)</td><td>86.3</td><td>15.8</td><td>63.4</td></tr><tr><td>PF-MCTQ (submitted cfg.)</td><td>92.9</td><td>41.7</td><td>24.0</td></tr><tr><td>Recurrent SAC</td><td>21.0</td><td>53.6</td><td>18.6</td></tr><tr><td>GRU-PPO</td><td>2.8</td><td>5.5</td><td>181.8</td></tr></table>