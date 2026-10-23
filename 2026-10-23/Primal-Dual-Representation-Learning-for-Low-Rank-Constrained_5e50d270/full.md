# Primal-Dual Representation Learning for Low-Rank Constrained MDPs

Chenhao Zhou Chao Zhang<sup>∗</sup> Wensong Bai Hanbin Zhao Hui Qian College of Computer Science and Technology, Zhejiang University zhouchenhao@zju.edu.cn zczju@zju.edu.cn

## Abstract

Representation learning has substantially improved the efficiency of unconstrained continuous-state Markov Decision Processes (MDPs), but its extension to Constrained MDPs (CMDPs) is difficult because the transition representation must be learned while constraint violation is controlled. In this paper, we study representation learning for low-rank CMDPs with continuous state spaces, and propose REP-PD-TEG for the soft cumulative-violation setting. In the feature learning period, the algorithm interacts with the environment using a decaying-ϵ-greedy mixture execution policy, and uses the collected samples to learn transition features by maximum likelihood. Based on the learned features, it constructs uncertainty bonuses and plans with an optimistic Lagrangian objective to obtain the return policy. We prove regret and soft-violation guarantees for the return policies, and quantify the influence of the decaying-ϵ-greedy exploration. In particular, REP-<sub>PD-TEG achieves the optimal Oe(</sub>√<sub>K) return policy bound, and gives a</sub> ${ \widetilde O } ( K ^ { 3 / 4 } )$ overall sample bound when exploratory actions are charged. For the stricter hard cumulative violation, where feasibility must be verified episode by episode, we propose REP-PD-H-TEG. It plans under a bonus-corrected lower-confidence utility certificate with the learned representation, and performs a stabilized dual search to obtain the return policy, yielding regret and hard-violation bounds. Together, these results give the first theoretically guaranteed representation-learning method for low-rank continuous-state CMDPs with both soft cumulative guarantees and certified hard-violation control.

## 1 Introduction

Constrained Markov Decision Processes (CMDPs) provide a principled framework for constrained reinforcement learning with both reward objectives and safety requirements [4, 22]. This framework is especially relevant in safety-critical domains, and a growing line of provably constrained RL algorithms has been developed within this framework [8, 9, 16, 24, 27, 39]. These works primarily focus on the tabular setting, where guarantees scale with the cardinality of the state space and become unsuitable for continuous or high-dimensional environments. Linear CMDP methods [5, 13, 17, 18, 26, 42] partially address this issue by assuming that transitions are linearly represented by known low-dimensional features, yielding state-dimension-independent bounds. However, existing tabular/linear CMDP methods all rely on a priori knowledge of the latent feature representations. In the absence of such prior feature knowledge, their guarantees no longer apply, leaving CMDPs with unknown features insufficiently addressed.

Representation learning has been effective for discovering low-dimensional transition features in unconstrained MDPs [2, 29, 31, 36, 37, 41, 44]. Extending this idea to CMDPs is more challenging because representation error now affects both reward optimization and constraint evaluation. The learner must identify latent dynamics from online data, use the learned representation to guide exploration, and still control cumulative constraint violation [2, 16]. This creates a coupling between representation coverage and constraint-aware policy optimization that is absent from unconstrained low-rank RL and is not handled by existing CMDP results with known features.

In this paper, we develop primal-dual representation-learning algorithms for low-rank continuousstate CMDPs with unknown transition features. During feature learning, the algorithms interact with the environment via an ϵ-greedy mixture execution policy and use the collected samples to learn transition features by maximum-likelihood estimation (MLE). For soft violation criterion, where only the total constraint shortfall across episodes is charged, our algorithm, REP-PD-TEG, uses the learned features to construct uncertainty bonuses, and plan with the bonus-augmented Lagrangian objective that couples reward maximization, utility optimization, and exploration. The algorithm outputs the return policies with a primal-dual update, and admits both return-policy and execution-policy guarantees. For the stricter per-episode hard-violation metric, where each episode’s shortfall is charged separately, we develop REP-PD-H-TEG, which keeps the representation-learning procedure but plans under a tight lower-confidence utility certificate and conducts a stabilized dual search to obtain the return policy. Specifically, we make the following contributions:

• We propose the first theoretically guaranteed representation-learning method for low-rank continuous-state CMDPs with unknown features. Starting from the occupancy of the current return policy, it uses ϵ-greedy auxiliary probes for action coverage, fits transition features by MLE, and constructs a merged-covariance uncertainty bonus. We quantify the effect of decaying-ϵ exploration for both return and execution policies.

• For the aggregate soft-violation criterion, we propose REP-PD-TEG, which updates the return policy through an optimistic Lagrangian whose reward and dual-weighted utility terms are both bonus-augmented based on the learned features. We show that REP-PD-TEG achieves return policy regret and soft violation of order ${ \tilde { O } } ( K ^ { 1 / 2 + p } )$ sample complexity under the schedule $\epsilon _ { k } = k ^ { - p }$ , where $\epsilon _ { k }$ is the exploration level in episode $k .$ In particular, this recovers the optimal ${ \tilde { O } } ( { \sqrt { K } } )$ return policy rate when $p = 0$ , which corresponds to uniform auxiliary probing over actions during the whole procedure. We also prove an execution policy overall bound: executing the ϵ<sub>k</sub>-greedy exploration adds $O ( K ^ { 1 - p } )$ complexity, and the balanced choice $p = 1 / 4$ yields a ${ \tilde { O } } ( K ^ { 3 / 4 } )$ ) execution policy regret and soft violation up to problem-dependent factors.

• For the stricter per-episode hard-violation criterion, we propose REP-PD-H-TEG, which uses the learned features to conduct a stabilized dual search and certifies feasibility through a bonus-corrected lower-confidence utility certificate. For return policies, we obtain regret of order ${ \tilde { O } } ( K ^ { 3 / 4 + p } )$ and hard violation of order ${ \tilde { O } } ( K ^ { 1 / 2 + p } ) ;$ taking $p = 0$ again gives uniform probing and the clean ${ \tilde { O } } ( { \sqrt { K } } )$ hard-violation dependence. For execution policies, we show that the cost of ϵ -greedy exploration is also $\scriptstyle { \dot { O } } ( K ^ { 1 - p } )$ and choosing $p = 1 / 8$ gives a ${ \tilde { O } } ( K ^ { 7 / 8 } )$ regret and hard violation up to problem-dependent factors.

Appendix H provides the empirical study of REP-PD-TEG on a simulated job-scheduling task. Besides, we also implement an approximate neural network version of REP-PD-TEG on the Safety Gym task to validate the effectiveness of our representation learning procedure for a complex state-action space.

Remark on the probing schedule. In tabular and linear CMDPs, the state or feature representation is known, so the analyzed policy is the execution policy. In low-rank CMDPs, the transition features must be learned online, and auxiliary $\epsilon _ { k } \cdot$ -greedy probing is needed to obtain coverage over the state-action space for representation learning. This creates a distinction between the return-policy sequence and the execution policy. The latter is especially relevant in constrained RL because exploratory behavior can incur real training-time costs—for example, robot collisions or excessive contact forces—even when the final return policy is safe. The two REP-PD-TEG rows in Table 1 report, respectively, the sample complexity of the best return policy specialization $( p = 0 )$ and the best execution policy specialization $( p = 1 / 4 )$ , where $p$ denotes the probing strength with $\dot { \epsilon _ { k } } = k ^ { - p }$ Similarly, the two REP-PD-H-TEG rows in Table 2 show the best return policy complexity with $p = 0$ and the balanced execution policy complexity under the hard specialization $p = 1 / 8$ , respectively.

Table 1: Comparison of soft-violation bounds. The two REP-PD-TEG rows report the complexity of the best return policy specialization and the best execution policy specialization, respectively.
<table><tr><td rowspan=1 colspan=1>MDP</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Known Feature</td><td rowspan=1 colspan=1>Regret</td><td rowspan=1 colspan=1>Violation</td></tr><tr><td rowspan=2 colspan=1>Tabular</td><td rowspan=1 colspan=1>Efroni et al. [16]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { | { \cal S } | ^ { 3 } | { \cal A } | H ^ { 4 } K } \right)$ </td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { | { \cal S } | ^ { 3 } | { \cal A } | H ^ { 4 } K } \right)$ </td></tr><tr><td rowspan=1 colspan=1>Liu et al. [27]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { | { \cal S } | ^ { 3 } | { \cal A } | H ^ { 6 } K } \right)$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=3 colspan=1>Linear</td><td rowspan=1 colspan=1>Ding et al. [13]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 2 } H ^ { 6 } K } \right)$ </td><td rowspan=1 colspan=1> $\overline { { { \tilde { O } } \left( \sqrt { d ^ { 2 } H ^ { 6 } K } \right) } }$ </td></tr><tr><td rowspan=1 colspan=1>Ghosh et al. [17]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\overline { { { \tilde { O } } \left( \sqrt { d ^ { 3 } H ^ { 3 } K } \right) } }$ </td><td rowspan=1 colspan=1> $\overline { { { \cal O } \left( \sqrt { d ^ { 3 } H ^ { 3 } K } \right) } }$ </td></tr><tr><td rowspan=1 colspan=1>Ghosh et al. [17]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 3 } H ^ { 5 } K } \right)$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=2 colspan=1>Low-rank</td><td rowspan=2 colspan=1>REP-PD-TEG</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1> $\widetilde { O } \left( | A | d ^ { 2 } \sqrt { K } ( 1 - \gamma ) ^ { - 2 } \right)$ [return policy]</td><td rowspan=1 colspan=1> $\widetilde { O } \left( | A | d ^ { 2 } \sqrt { K } ( 1 - \gamma ) ^ { - 2 } \right)$ [return policy]</td></tr><tr><td rowspan=1 colspan=1> $\tilde { O } \left( | A | d ^ { 2 } K ^ { 3 / 4 } ( 1 - \gamma ) ^ { - 2 } \right)$ [execution policy]</td><td rowspan=1 colspan=1> $\tilde { O } \left( | A | d ^ { 2 } K ^ { 3 / 4 } ( 1 { - } \gamma ) ^ { - 2 } \right)$ [execution policy]</td></tr></table>

Table 2: Comparison of hard-violation bounds. The two REP-PD-H-TEG rows report the complexity of the best return policy specialization and the balanced execution policy specialization, respectively.
<table><tr><td rowspan=1 colspan=1>MDP</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>KnownFeature</td><td rowspan=1 colspan=1>Known SafeAction Set</td><td rowspan=1 colspan=1>Regret</td><td rowspan=1 colspan=1>Violation</td></tr><tr><td rowspan=2 colspan=1>Tabular</td><td rowspan=1 colspan=1>Efroni et al. [16]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { | { \cal S } | ^ { 3 } | { \cal A } | H ^ { 4 } K } \right)$ </td><td rowspan=1 colspan=1> $\overline { { { \tilde { O } } \left( \sqrt { | S | ^ { 3 } | A | H ^ { 4 } K } \right) } }$ </td></tr><tr><td rowspan=1 colspan=1>Bura et al. [9]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { | { \cal S } | ^ { 2 } | { \cal A } | H ^ { 6 } K } \right)$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=4 colspan=1>Linear</td><td rowspan=1 colspan=1>Amani et al. [5]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 3 } H ^ { 4 } K } \right)$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Kitamura et al. [26]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 5 } H ^ { 8 } K } \right)$ </td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>Wei et al. [42]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> $\overline { { { \tilde { O } } \left( \sqrt { d ^ { 3 } H ^ { 4 } K } \right) } }$ </td><td rowspan=1 colspan=1> $\overline { { { \tilde { O } } \left( \sqrt { d H ^ { 2 } K } \right) } }$ </td></tr><tr><td rowspan=1 colspan=1>Ghosh et al. [18]</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 3 } H ^ { 4 } K } \right)$ </td><td rowspan=1 colspan=1> $\tilde { O } \left( \sqrt { d ^ { 3 } H ^ { 4 } K } \right)$ </td></tr><tr><td rowspan=2 colspan=1>Low-rank</td><td rowspan=2 colspan=1>REP-PD-H-TEG</td><td rowspan=2 colspan=1>No</td><td rowspan=2 colspan=1>No</td><td rowspan=1 colspan=1> $\tilde { O } \left( | A | d ^ { 2 } K ^ { 3 / 4 } ( 1 - \gamma ) ^ { - 2 } \right)$ [return policy]</td><td rowspan=1 colspan=1> $\overline { { \tilde { O } \left( | \mathcal { A } | d ^ { 2 } \sqrt { K } \left( 1 - \gamma \right) ^ { - 2 } \right) } }$ [return policy]</td></tr><tr><td rowspan=1 colspan=1> $\tilde { O } \left( | A | d ^ { 2 } K ^ { 7 / 8 } ( 1 - \gamma ) ^ { - 2 } \right)$ [execution policy]</td><td rowspan=1 colspan=1> $\tilde { O } \left( | A | d ^ { 2 } K ^ { 7 / 8 } ( 1 - \gamma ) ^ { - 2 } \right)$ [execution policy]</td></tr></table>

## 1.1 Related Work

Provably Constrained RL CMDPs provide a rigorous framework for safe RL, balancing reward maximization with cumulative cost constraints [4, 19]. Several practical constrained RL methods have been developed on this foundation, including policy-gradient, Lagrangian, shielding, and modelbased approaches [1, 6, 7, 11, 21, 35]. On the theoretical side, there has been a growing body of work on provably safe reinforcement learning. Many works [8, 9, 14, 16, 24, 25, 27, 39] consider the tabular setting. Among the best known regret and constraint-violation guarantees are of order ${ \tilde { O } } ( { \sqrt { K } } )$ [16, 27]. However, their polynomial dependence on the cardinality of state space |S| makes them unsuitable for large or continuous state spaces.

Recent advances in linear CMDPs [5, 13, 17, 18, 26, 30, 42, 43] achieve dimension-independent sample complexity by assuming known low-dimensional features $\phi ^ { \star } ( s , a )$ They enforce safety through confidence intervals, optimistic bonuses, primal-dual updates, pessimistic constraints, or safe baselines. However, their reliance on predefined latent features, assumed to be known a priori or handcrafted, limits applicability when transition structures are unknown and must be learned from data. Our setting instead has unknown latent features that must be learned online. To our knowledge, the only prior work on low-rank CMDPs is [20], which studies the sample complexity of identifying a near-optimal feasible policy after training. In contrast, our work analyzes the entire learning trajectory and provides regret and constraint-violation guarantees.

Representation Learning for Unconstrained MDPs A growing body of research has focused on theoretically grounded representation learning in RL, aiming to derive low-dimensional features that efficiently capture environment dynamics. For example, works such as [15, 28, 46] investigated representation learning in block MDPs, leveraging hierarchical or modular state decompositions to improve sample efficiency. [38] studied representations under a Gaussian noise model, establishing guarantees for feature learning in noisy environments. Low-Rank MDPs have emerged as a pivotal framework for representation learning in structured environments. Within this regime, model-based approaches [2, 36, 41, 44, 45] learn latent transition features by leveraging model classes of transition probabilities, achieving sample-efficient exploration under realizability assumptions. In contrast, model-free methods like [29] eschew explicit transition models and show efficiency under minimal reachability conditions. The framework has recently expanded into reward-free exploration [10, 20] and multi-agent Markov games [31, 34, 47, 48]. However, these works do not address online safety constraints in CMDPs. Consequently, this work gives the first representation-learning framework for low-rank CMDPs that bridges feature discovery and provable online safety.

## 2 Background

We consider an episodic constrained MDP, denoted by $( \mathcal { S } , \mathcal { A } , P ^ { * } , r , g , \gamma , d _ { 1 } )$ where $s$ is the state space, A is the action space, $P ^ { * }$ is the transition function, r is the reward function, g is the utility function, γ is the discounted factor, and $d _ { 1 }$ is the initial distribution. Following prior work [23, $4 0 , 4 1 ]$ , we assume trajectory reward is normalized, i.e., for any trajectory $\{ s _ { h } , a _ { h } \} _ { h = 1 } ^ { \infty }$ , we have ${ \textstyle \sum _ { h = 1 } ^ { \infty } \gamma ^ { h } r ( s _ { h } , a _ { h } ) \in [ 0 , 1 ] }$ and $\textstyle \sum _ { h = 1 } ^ { \infty } \gamma ^ { h } g ( s _ { h } , a _ { h } ) \in [ 0 , 1 ]$

In an infinite-horizon discounted CMDP, each episode starts with state $s _ { 1 } \sim d _ { 1 }$ . Then at each step $h ,$ the agent observes state $s _ { h } \in S ,$ , picks an action $a _ { h } \in \mathcal { A }$ , and receives a reward $r ( s _ { h } , a _ { h } )$ and a utility $g ( s _ { h } , a _ { h } )$ . The MDP then transitions to $s _ { h + 1 }$ which is drawn from $P ^ { * } ( \cdot | s _ { h } , a _ { h } )$ . In this paper, we assume access to reward and utility function feedback, but does not require access to the transition model. For any $s \in \mathcal S , \pi ( a | s )$ denotes the probability of selecting action $a \in { \mathcal { A } }$ when the state is s under policy π. Let $d _ { P ^ { * } } ^ { \pi } ( s )$ be the state occupancy measure

$$
d _ { P ^ { * } } ^ { \pi } ( s ) = ( 1 - \gamma ) \sum _ { h = 1 } ^ { \infty } \gamma ^ { h - 1 } P _ { \pi } ^ { \star } ( s _ { h } = s | s _ { 1 } \sim d _ { 1 } ) ,
$$

where $P _ { \pi } ^ { * } ( s _ { h } = s | s _ { 1 } \sim d _ { 1 } )$ is the probability that $s _ { h } = s$ , after starting at state $s _ { 1 } \sim d _ { 1 }$ and following $\pi$ thereafter. Also, let $d _ { P ^ { * } } ^ { \pi } ( s , a ) = \pi ( a | s ) d _ { P ^ { * } } ^ { \pi } ( s )$ be the state-action occupation measure. We define the expected value function for the discounted reward under policy π starting from state s as:

$$
V _ { P ^ { * } , r } ^ { \pi } ( s ) = \mathbb { E } _ { P ^ { * } , \pi } \left[ \sum _ { h = 1 } ^ { \infty } \gamma ^ { h } r ( s _ { h } , a _ { h } ) | s _ { 1 } = s \right] ,\tag{1}
$$

where E is taken with respect to the policy π and the transition function $P ^ { * }$ . Similarly, we define the action-value function for the reward as:

$$
Q _ { P ^ { * } , r } ^ { \pi } ( s , a ) = \mathbb { E } _ { P ^ { * } , \pi } \left[ \sum _ { h = 1 } ^ { \infty } \gamma ^ { h } r ( s _ { h } , a _ { h } ) | s _ { 1 } = s , a _ { 1 } = a \right] .\tag{2}
$$

For simplicity, when the context is clear, we sometimes abbreviate $V _ { P ^ { \star } , r } ^ { \pi }$ and $Q _ { P ^ { \star } , r } ^ { \pi }$ as $V _ { r } ^ { \pi }$ and $Q _ { r } ^ { \pi }$ respectively. We also define the value function for the utility $V _ { g } ^ { \pi } ( s )$ and the action-value function for the utility $Q _ { g } ^ { \pi } ( s , a )$ analogously by replacing r with g in (1) and (2), respectively. We denote $V _ { j } ^ { \pi } ( s )$ and $Q _ { j } ^ { \pi } ( s , \bar { a } )$ for $j = r , g$

Definition 2.1. For brevity, we denote $P ^ { * } V _ { j } ^ { \pi } ( s , a ) = \mathbb { E } _ { s ^ { \prime } \sim P ^ { * } ( \cdot \vert s , a ) } V _ { j } ^ { \pi } ( s ^ { \prime } ) \mathrm { f o r } j = r , g .$

Using this notation, the Bellman’s equation associated with the policy π becomes

$$
Q _ { j } ^ { \pi } ( s , a ) = ( j + \gamma P ^ { * } V _ { j } ^ { \pi } ) ( s , a ) , \qquad j \in \{ r , g \} .
$$

Note that $V _ { j } ^ { \pi } ( s ) = \langle \pi ( \cdot | s ) , Q _ { j } ^ { \pi } ( s , \cdot ) \rangle _ { A }$ , where $\begin{array} { r } { \langle \pi ( \cdot | s ) , Q _ { j } ^ { \pi } ( s , \cdot ) \rangle _ { \cal A } = \sum _ { a \in \cal A } \pi _ { h } ( a | s ) Q _ { j } ^ { \pi } ( s , a ) . } \end{array}$

The objective of the learning agent is to find an optimal policy that maximizes the expected reward while satisfying a safety constraint on the expected utility:

$$
\operatorname* { m a x } _ { \pi } V _ { r } ^ { \pi } ( s _ { 1 } ) ,
$$

$$
{ \mathrm { s u b j e c t } } \tan V _ { g } ^ { \pi } ( s _ { 1 } ) \geq b .\tag{3}
$$

where $b \in ( 0 , 1 ]$ is a given threshold. While we focus on a single constraint for simplicity, our method naturally extends to handle multiple constraints. We denote the optimal policy as $\pi ^ { * }$ which solves the above optimization problem.

In the absence of prior knowledge about the constraint, the agent must balance exploration and exploitation while ensuring that the policy satisfies the constraint. To this end, we allow the policy to occasionally violate the constraint and aim to minimize the regret and the total constraint violations over K episodes. This approach is consistent with existing literature on constrained RL [12, 13, 16].

To formally measure the performance of the policy sequence $\{ \pi _ { k } \} _ { 1 } ^ { K }$ , we define the regret as

$$
\operatorname { R e g r e t } ( K ) = \sum _ { k = 1 } ^ { K } V _ { r } ^ { \pi ^ { * } } ( s _ { 1 } ) - V _ { r } ^ { \pi _ { k } } ( s _ { 1 } ) ,
$$

and the soft/hard constraint violations as

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ( K ) = \left[ \sum _ { k = 1 } ^ { K } ( b - V _ { g } ^ { \pi _ { k } } ( s _ { 1 } ) ) \right] _ { + } , \mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ( K ) \quad = \sum _ { k = 1 } ^ { K } \left[ b - V _ { g } ^ { \pi _ { k } } ( s _ { 1 } ) \right] _ { + } ,
$$

where $[ z ] _ { + } = \operatorname* { m a x } \{ z , 0 \}$ . The regret measures the gap between the total reward value by following the optimal policy $\pi ^ { * }$ , and the total reward value obtained by following agent’s policy $\pi _ { k }$ at episode $k$ over K episodes. The soft constraint violation quantifies the cumulative difference between the desired constraint threshold and the achieved utility, allowing for both positive and negative deviations over time. In contrast, the hard constraint violation is a stricter metric that measures only the cumulative positive deviations. As shown in [18], even zero soft violation may lead to hard constraint violation that grows linearly with the number of episodes, while soft constraint violation can only grow sub-linearly with the number of episodes if the hard constraint violation grows sub-linearly.

These metrics capture cumulative notions of safety rather than strict zero-violation safe exploration. In particular, soft violation is the standard online-CMDP notion for budget-style or long-run average constraints, while hard violation is a stricter cumulative metric that counts only positive deviations. Throughout the paper, our guarantees concern cumulative violation control; they do not imply per-episode safety during learning.

Dual problem and Slater’s Condition. We then introduce some additional notation that will be used throughout this paper.

Definition 2.2. For any policy π and Lagrangian multiplier $Y$ , we define the composite value function and Q-function as $V ^ { \pi ^ { \smash { \prime } } , Y ^ { * } } ( s ) \dot { = } V _ { r } ^ { \pi } ( s ) \dot { + } Y \breve { V _ { g } ^ { \pi } } ( s )$ and $\ \dot { Q } ^ { \pi , Y } ( s , a ) = Q _ { r } ^ { \pi } ( s , a ) \dot { + } Y Q _ { g } ^ { \pi } ( s , a )$

We can cast problem (3) as a saddle point problem max<sub>π</sub> min<sub>Y</sub> ${ \mathcal { L } } ( \pi , Y )$ , with Lagrangian ${ \mathcal { L } } ( \pi , Y ) =$ $V _ { r } ^ { \pi } ( s _ { 1 } ) + \dot { Y ( } V _ { a } ^ { \pi } ( s _ { 1 } ) - b ) = V ^ { \pi , Y } \dot { - } Y b$ . Although the Lagrangian is non-concave in π [3], the strong duality holds $[ \bar { 3 } 2 ]$ . Hence, there exists an optimal Lagrangian multiplier $Y ^ { * }$ such that $\operatorname* { m a x } _ { \pi } \mathcal { L } ( \pi , Y ^ { * } )$ recovers the optimal reward value function.

We assume the following Slater’s condition in this paper.

Assumption 2.3 (Slater’s Condition). There exists $\theta > 0$ , and a policy π¯, such that $V _ { g } ^ { \bar { \pi } } ( s _ { 1 } ) \geq b + \theta$

The Slater’s condition is mild in practice and widely used in prior work $[ 1 3 , 1 6 , 3 3 ] ;$ it also implies that the optimal dual variable is bounded.

Lemma 2.4 (Boundedness of $Y ^ { * } )$ . The optimal dual-variable $Y ^ { \ast } \leq \frac { V _ { r } ^ { \pi ^ { \ast } } ( s _ { 1 } ) - V _ { r } ^ { \bar { \pi } } ( s _ { 1 } ) } { \theta } \leq \frac { 1 } { \theta } .$

Low-rank CMDP. In this paper, we study low-rank CMDPs, which are characterized by a low-rank decomposition of the transition function. Specifically, we assume the following:

Assumption 2.5 (low-rank CMDP). The transition function of the CMDP $P ^ { * } : \mathcal { S } \times \mathcal { A }  \Delta ( \mathcal { S } )$ admits a low rank decomposition with rank $d \in \mathbb { N }$ if there exists two embedding functions $\phi ^ { * } :$ $S \times \mathcal { A }  \mathbb { R } ^ { d } , \mu ^ { * } : S  \bar { \mathbb { R } ^ { d } }$ such that

$$
\forall s , s ^ { \prime } \in S , a \in A : P ^ { * } ( s ^ { \prime } | s , a ) = \langle \phi ^ { * } ( s , a ) , \mu ^ { * } ( s ^ { \prime } ) \rangle .
$$

For normalization, we assume $\| \phi ^ { * } ( s , a ) \| \leq 1$ for all $( s , a )$ and for any functions $f : { \mathcal { S } }  [ 0 , 1 ]$ $\begin{array} { r } { \| \int \mu ^ { * } ( s ) f ( s ) d s \| _ { 2 } \leq \sqrt { d } } \end{array}$

Remark 2.6. The low-rank CMDP assumption, characterized by the existence of low-dimensional embedding functions $\phi ^ { * } ( s , a )$ and $\mu ^ { * } ( s ^ { \prime } )$ , is widely adopted structural assumption in RL [2, 37, 41]. This assumption encompasses several common settings as special cases, including tabular CMDPs, linear CMDPs, and block CMDPs.

Stochastic-oracle feedback model. Following prior low-rank representation-learning work, we study an occupancy-based stochastic-oracle setting. The learner may query states distributed as $d _ { P } ^ { \pi }$ <sub>⋆</sub> together with auxiliary transitions that are used only for representation learning and bonus construction. Practically, a discounted-occupancy state can be sampled by resetting to the initial distribution and rolling in under π for a geometrically distributed number of steps. Here $K$ counts outer oracle rounds, not raw environment transitions; implementing K queries this way uses $O ( K / ( 1 - \gamma ) + 3 K )$ transitions in expectation. Our regret and violation bounds do not charge the complete roll-in trajectories. Likewise, the execution-policy corollaries transfer values to the stationary ϵ-mixture policy rather than giving an end-to-end guarantee for the entire three-transition probing trajectory.

Function approximation setup and MLE oracle. Since the transition function $P ^ { * }$ is unknown, we employ a function class $\mathcal { M } = \bar { \{ ( \mu , \phi ) : \mu \in \Theta , \phi \in \Gamma \} }$ to characterize $\mu ^ { \star }$ and $\phi ^ { \star }$ , and by extension, to approximate $P ^ { * }$ . Our formal guarantees use a finite realizable class, $| { \mathcal { M } } | < \infty$ and $P ^ { \star } \in { \mathcal { M } }$ ; this restricts the candidate representations rather than the cardinality of the state space. Finite realizability and logarithmic dependence on $| { \mathcal { M } } |$ are standard in low-rank representation-learning analyses [2, 41]. To select the appropriate transition $\hat { P }$ from the function class $\mathcal { M } .$ , we assume access to a supervised learning-style maximum-likelihood estimation (MLE) oracle.

Definition 2.7 (MLE Oracle). Given a model class M and a dataset D in the form of $( s , a , s ^ { \prime } )$ , the MLE oracle returns the maximum likelihood estimator $\begin{array} { c c l } { { { \hat { P } } } } & { { = } } & { { ( { \hat { \mu } } , { \hat { \phi } } ) } } \end{array} =$ arg $\begin{array} { r } { \operatorname* { m a x } _ { ( \mu , \phi ) \in \mathcal { M } } \mathbb { E } _ { \mathcal { D } } \ln ( \mu ( s ^ { \prime } ) ^ { \top } \phi ( s , a ) , } \end{array}$ ).

## 3 Soft Violation via Cumulative Primal-Dual Control

We first study low-rank CMDPs under the soft cumulative-violation metric, where constraint deficits and surpluses are aggregated before taking the positive part. The learner must identify an unknown low-rank transition representation while updating policies through a primal-dual procedure: returnpolicy data may lack action coverage, whereas always-uniform actions are less policy-adaptive. REP-PD-TEG (Representation-learning Primal-Dual with Tertiary decaying-ϵ- Greedy probing) resolves this tension by keeping the probing kernel close to the previous return policy while retaining a uniform action floor; the tertiary transition supplies the coverage needed for optimism. Unless stated otherwise, regret and soft violation are evaluated on the return policies $\{ \bar { \pi _ { k } } \} _ { k = 1 } ^ { K } ;$ auxiliary tuples are used only for model estimation and bonus construction. We later quantify the cost of charging the execution policy.

## 3.1 REP-PD-TEG Algorithm

Algorithm 1 gives REP-PD-TEG. At episode k, the algorithm interacts with the environment using the execution policy $q _ { k } ( \cdot  { | } s ) = q _ { k } ^ { \prime } ( \cdot  { | } \hat { s } ) = q _ { k } ^ { \prime \prime } ( \cdot  { | } s ) = \overline { { ( 1 - \epsilon _ { k } ) \pi _ { k - 1 } ( \cdot  { | } s ) + \epsilon _ { k } U ( \mathcal { A } ) } }$ , where $U ( { \bar { \mathcal { A } } } )$ is the uniform distribution over the action space and $\epsilon _ { 1 } \geq \epsilon _ { 2 } \geq \cdot \cdot \cdot \geq \epsilon _ { K } > 0$ . The oracle returns one tertiary tuple $( s , a , s ^ { \prime } , a ^ { \prime } , s ^ { \prime \prime } , a ^ { \prime \prime } , \bar { s } )$ : it first samples $s \sim d _ { P ^ { \star } } ^ { \pi _ { k - 1 } }$ , then alternates actions from $q _ { k } , q _ { k } ^ { \prime } , q _ { k } ^ { \prime \prime }$ with transitions from $P ^ { \star }$ . The resulting three transitions are stored in separate replay buffers denoted as $D _ { k } , D _ { k } ^ { \prime }$ and $D _ { k } ^ { \prime \prime }$ . After updating the replay buffers, REP-PD-TEG fits $\widehat { P } _ { k } = ( \widehat { \mu } _ { k } , \widehat { \phi } _ { k } )$ by MLE on $D _ { k } + \mathrm { \ddot { \cal D } } _ { k } ^ { \prime } + \ddot { \cal D } _ { k } ^ { \prime \prime }$ The third stream is required by the present merged-covariance design, not by MLE alone. Since the covariance contains both $D _ { k }$ and $D _ { k } ^ { \prime } ,$ , the one-step-back proof produces propagated quadratic terms under both primary and secondary designs: $D _ { k } ^ { \prime }$ controls the former, while the one-step pushforward $D _ { k } ^ { \prime \prime }$ supplies an MLE-controlled denominator for the latter. We therefore do not place $\bar { D } _ { k } ^ { \prime \prime }$ in the covariance, which would create another propagated level. The learned features are then utilized to calculate the empirical covariance matrix $\widehat { \Sigma } _ { k } ^ { \mathrm { m i x } }$ , which quantifies the uncertainty associated with these features. Based on the empirical covariance matrix, an exploration bonus, $\widehat { b } _ { k } ( \boldsymbol { x } , \boldsymbol { u } )$ , is constructed as a scaled Mahalanobis norm of the learned feature $\widehat { \phi } _ { k }$ with respect to $\widehat { \Sigma } _ { k } ^ { \mathrm { m i x } }$ . This bonus induces uncertainty-aware planning under the learned model by augmenting both the reward and utility value estimates, after which the return policy and dual variable are updated according to the final two steps of Algorithm 1.

Algorithm 1 REP-PD-TEG   
1: Input: Parameters $\{ \lambda _ { k } \} _ { k = 1 } ^ { K } , \{ \alpha _ { k } \} _ { k = 1 } ^ { K } , \eta , \xi , \{ \epsilon _ { k } \} _ { k = 1 } ^ { K }$ , model class $\mathcal { M } = \{ ( \mu , \phi ) : \mu \in \Theta , \phi \in$   
$\Gamma \}$ , number of episodes $K$   
2: Initialize return policy $\pi _ { 0 } ( \cdot \mid s ) = U ( A ) , Y _ { 1 } = 0 , \mathrm { a n d } D _ { 0 } = D _ { 0 } ^ { \prime } = D _ { 0 } ^ { \prime \prime } = \emptyset .$   
3: for $k = 1 , \dots , \overset { \cdot } { K }$ do   
4: Set execution policy $q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } = ( 1 - \epsilon _ { k } ) \pi _ { k - 1 } + \epsilon _ { k } U ( \mathcal { A } )$   
5: Query one auxiliary tertiary tuple $( s , \bar { a } , s ^ { \prime } , \bar { a ^ { \prime } } , s ^ { \prime \prime } , a ^ { \prime \prime } , \bar { s } )$ , where s ∼ $d _ { P ^ { \star } } ^ { \pi _ { k - 1 } } , a \sim q _ { k } ( \cdot \mid s )$   
$\bar { s ^ { \prime } } \sim \mathsf { \bar { P } } ^ { \star } ( \cdot \mid s , a ) , a ^ { \bar { \prime } } \sim q _ { k } ^ { \prime } ( \cdot \mid s ^ { \prime } ) ^ { \star } , s ^ { \prime \prime } \sim P ^ { \star } ( \cdot \mid s ^ { \prime } , a ^ { \prime } ) , a ^ { \prime \prime } \sim q _ { k } ^ { \prime \prime } ( \cdot \mid s ^ { \prime \prime } ) , \bar { s } \sim P ^ { \star } ( \cdot \mid s ^ { \prime \prime } , a ^ { \prime \prime } )$   
6: Update $\hat { D _ { k } } ^ { \prime } = \hat { D } _ { k - 1 } \cup \{ ( s , a , s ^ { \prime } ) \} , D _ { k } ^ { \prime } = \hat { D } _ { k - 1 } ^ { \prime ^ { \prime } } \cup \{ ( s ^ { \prime } , a ^ { \prime } , s ^ { \prime \prime } ) \} , \tilde { D } _ { k } ^ { \prime \prime ^ { \prime } } = D _ { k - 1 } ^ { \prime \prime } \cup \{ ( s ^ { \prime \prime } , a ^ { \prime \prime } , \bar { s } ) \}$   
7: Fit the low-rank transition model by MLE:   
$\widehat { P } _ { k } = \bigl ( \widehat { \mu } _ { k } , \widehat { \phi } _ { k } \bigr ) \in \arg \operatorname* { m a x } _ { ( u , \phi ) \in \mathcal { M } } \mathbb { E } _ { D _ { k } + D _ { k } ^ { \prime } + D _ { k } ^ { \prime \prime } } \bigl [ \log \mu ( x ^ { \prime } ) ^ { \top } \phi ( x , u ) \bigr ]$   
8: Form the merged empirical covariance   
$\widehat { \Sigma } _ { k } ^ { \mathrm { m i x } } = \mathrm { ~  ~ \xi ~ { ~ \sum ~ } ~ } \widehat { \phi } _ { k } ( x , u ) \widehat { \phi } _ { k } ( x , u ) ^ { \top } + \mathrm { ~  ~ \xi ~ { ~ \sum ~ } ~ } \widehat { \phi } _ { k } ( x , u ) \widehat { \phi } _ { k } ( x , u ) ^ { \top } + \lambda _ { k } I .$   
(x,u,x<sup>′</sup>)∈D<sub>k</sub> (x,u,x<sup>′</sup>)∈D<sup>′</sup>   
9: Set the covariance mixture bonus   
$\widehat { b } _ { k } ( x , u ) = \operatorname* { m i n } \left\{ \alpha _ { k } \sqrt { \widehat { \phi } _ { k } ( x , u ) ^ { \top } ( \widehat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } \widehat { \phi } _ { k } ( x , u ) } , \bar { b } \right\}$   
10: Update the return policy $\begin{array} { r } { \pi _ { k } \in \arg \operatorname* { m a x } _ { \pi } \left( Q _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \pi } + Y _ { k } Q _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } \right) . } \end{array}$   
11: Update the dual variable $\begin{array} { r } { Y _ { k + 1 } = \Pi _ { \left[ 0 , \xi \right] } \left( Y _ { k } + \eta \big ( b - V _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) \big ) \right) } \end{array}$   
12: end for   
13: Return $\pi _ { 1 } , \ldots , \pi _ { K } .$

The dual variable is projected into $[ 0 , \xi ]$ and we set $\xi = 2 / \theta$ in the analysis. Reward optimism promotes exploration, while utility optimism is valid for soft violation because episode-level deficits and surpluses are aggregated. Thus one episode consists of auxiliary tertiary sampling, representation learning, bonus construction, optimistic primal update, and projected dual update.

Algorithm 1 uses exact MLE and exact optimistic-Lagrangian planning to isolate the statistical challenge of learning unknown representations. The proof exposes a conditional interface for other learners: a representation method may replace MLE if it establishes the same prediction event under all three probing designs, with any excess prediction error added to the representation-error term. Likewise, full-planning suboptimality enters through the cumulative planning term. A raw MLE objective gap need not imply the required prediction event, so the present theorems do not claim guarantees for arbitrary approximate oracles, misspecified models, or infinite classes. Spectral representation learning [37] is an example of a non-MLE alternative, but its three-design event would have to be verified separately here.

## 3.2 Polynomial-Schedule Guarantees

We state the guarantee for polynomial schedules $\epsilon _ { k } = k ^ { - p } ,$ , which make the exploration–deployment tradeoff explicit. The return-policy bound captures the statistical cost of learning the representation from auxiliary probing data, while the execution-policy statement adds the standard performancedifference cost of actually deploying the mixed probing policy.

Theorem 3.1 (Return policy and execution policy guarantees for polynomial probing). Suppose Assumptions 2.3 and 2.5 hold and thefinite model class is realizable, $| \mathbf { \bar { \mathcal { M } } } | < \infty \mathrm { \Gamma } \dot { a n d } \bar { P ^ { \star } } \in \dot { \mathcal { M } } .$ Let $\epsilon _ { k } = k ^ { - p }$ for some $p \in [ 0 , 1 / 2 )$ . Run Algorithm 1 with $\alpha _ { k } = \widetilde { O } ( \sqrt { ( | A | k ^ { p } + d ^ { 2 } ) \gamma \log ( | \mathcal { M } | k / \delta ) } )$ $\begin{array} { r } { \lambda _ { k } = \widetilde { \Theta } ( ( | | A | { k } ^ { p } + d ^ { 2 } ) / d ) \log ( | { M } | { k } / \delta ) ) , \bar { b } \geq \operatorname* { m a x } _ { k \leq K } \alpha _ { k } / \sqrt { \lambda _ { k } } , \eta = 2 ( 1 - \gamma ) / ( \theta \bar { b } \sqrt { K } ) , \xi = 2 / \theta . } \end{array}$ Then, with probability at least $1 - \delta ,$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde { O } \left( \frac { | A | \bar { b } d ^ { 2 } K ^ { 1 / 2 + p } } { \theta ( 1 - \gamma ) ^ { 2 } } \right) , \qquad \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde { O } \left( \frac { | A | \bar { b } d ^ { 2 } K ^ { 1 / 2 + p } } { ( 1 - \gamma ) ^ { 2 } } \right) .
$$

Moreover, for the execution policy $\widetilde { \pi } _ { k } = ( 1 - \epsilon _ { k } ) \pi _ { k } + \epsilon _ { k } U ( A )$

$$
{ \mathrm { R e g r e t } } ^ { \mathrm { e x e c } } ( K ) \leq { \mathrm { R e g r e t } } ^ { \mathrm { r e t u r n } } ( K ) + { \frac { 2 } { 1 - \gamma } } \sum _ { k = 1 } ^ { K } k ^ { - p } ,
$$

$$
\operatorname { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) \le \operatorname { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) + \frac { 2 } { 1 - \gamma } \sum _ { k = 1 } ^ { K } k ^ { - p } .
$$

The proof is given in Appendix D. Generally, our analyses combine a tuple-level MLE event, a pathwise-to-averaged bridge, tertiary-aware optimism for the merged bonus, and a cumulative overestimation bound for the dual update. The schedule parameter $_ p$ trades coverage for deployment cost: smaller $p$ improves the return-policy rate, while larger $p$ reduces the cost of executing the exploratory mixture. Since $\textstyle \sum _ { k = 1 } ^ { K } k ^ { - p } \leq K ^ { 1 - p } / ( 1 - p )$ , the return-policy term scales as $\widetilde { O } ( K ^ { 1 / 2 + p } )$ whereas execution adds $O ( K ^ { 1 - p } / ( 1 - \gamma ) )$

Corollary 3.2 (Best return policy and execution policy rates). Under the same setting as Theorem 3.1, thefollowing two specializations hold.

• First, $i f p = 0 ;$ , equivalently $\epsilon _ { k } \equiv 1$ , then the probing kernel in Algorithm 1 becomes uniform: $q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } = U ( A )$ . The return policies satisfy

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { \mathcal { O } } \left( \frac { | \mathcal { A } | \bar { b } d ^ { 2 } \sqrt { K } } { \theta ( 1 - \gamma ) ^ { 2 } } \right) , \quad \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { \mathcal { O } } \left( \frac { | \mathcal { A } | \bar { b } d ^ { 2 } \sqrt { K } } { ( 1 - \gamma ) ^ { 2 } } \right) .
$$

This is the best return policy specialization, and reduces to uniform tertiary probing.

• Second, $i f p = 1 / 4$ , then the return policies satisfy

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \bigg ( \frac { | A | \bar { b } d ^ { 2 } K ^ { 3 / 4 } } { \theta ( 1 - \gamma ) ^ { 2 } } \bigg ) , \quad \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \bigg ( \frac { | A | \bar { b } d ^ { 2 } K ^ { 3 / 4 } } { ( 1 - \gamma ) ^ { 2 } } \bigg )
$$

The execution policy gap is also $O ( K ^ { 3 / 4 } / ( 1 - \gamma ) )$ , and the overall sample complexities of the execution policies become

$$
\mathrm { R e g r e t } ^ { \mathrm { e x e c } } ( K ) \le \widetilde O ( K ^ { 3 / 4 } ) , \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) \le \widetilde O ( K ^ { 3 / 4 } )
$$

up to logarithmic and problem-dependent factors.

Corollary 3.2 highlights the schedule tradeoff: $p = 0$ is statistically sharp for return policies, while $p = 1 / 4$ balances representation learning with the cost of using the execution policy. Thus the returnpolicy theorem isolates representation-learning cost, and the execution policy corollary quantifies the extra price of deploying the probes as real behavior.

## 4 Hard Violation via Per-Episode Certification

We now consider the hard cumulative-violation metric. Unlike soft violation, hard violation does not allow constraint surplus in one episode to offset constraint deficit in another. Therefore the optimistic utility estimate used by the soft primal-dual update is insufficient for certifying feasibility at the episode level: an optimistic estimate may exceed the threshold b even when the true utility remains below it. REP-PD-H-TEG (the Hard-violation version of REP-PD-TEG) therefore keeps the same representation-learning procedure but changes the policy decision: policy search remains optimistic, while a tight feasibility test on the lower-confidence utility of the policy is explicitly conducted.

## 4.1 Lower-Certificate Planning in REP-PD-H-TEG

As shown in Algorithm 2 in Appendix F, REP-PD-H-TEG uses the same tertiary data collection, MLE representation update, merged covariance, and bonus construction as REP-PD-TEG. The hard-specific change lies in how the policy is selected after the bonus is built. In REP-PD-TEG, the optimistic utility value can be used in the dual update because violation is aggregated across episodes. In the hard violation setting, however, the policy executed in the current episode must be certified feasible on its own. Thus REP-PD-H-TEG introduces a lower-confidence utility certificate. Write $\Gamma _ { k } : = | \mathcal { A } | / \epsilon _ { k }$ and $\zeta _ { k } : = \ln ( | \mathcal { M } | k / \delta ) / k$ , and choose a sufficiently large universal calibration constant $C _ { \mathrm { c e r t } }$ . The one-sided statistical correction and the certificate are

$$
\ell _ { k } : = C _ { \mathrm { c e r t } } \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } , \qquad \mathcal { U } _ { k } ( \pi ) : = \Big [ V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - \ell _ { k } \Big ] _ { + } ,
$$

where $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$ . On the representation-learning good event, the one-step-back argument gives, simultaneously for every queried policy,

$$
\begin{array} { r } { \underline { { \mathcal { U } } } _ { k } ( \pi ) \leq V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) . } \end{array}
$$

Hence $\underline { { { \mathcal U } } } _ { k } ( \pi ) \geq b$ implies true feasibility. The corresponding two-sided width is needed only in the analysis of the selected policy and is

$$
w _ { k } ( \pi ) : = 2 \ell _ { k } + 2 V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) , \qquad V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \leq \mathcal { U } _ { k } ( \pi ) + w _ { k } ( \pi ) .
$$

Separating $\ell _ { k }$ from $w _ { k } ( \pi )$ is essential: the former makes the stopping rule conservative, while the latter accounts for the pessimistic bonus and is summed only along the returned policy sequence.

Instead of choosing a single policy at the current dual variable, REP-PD-H-TEG performs a capped dual search within the episode in the planning step. For each queried dual level Y , it solves the optimistic Lagrangian planning problem

$$
\widehat { \pi } _ { k , Y } \in \arg \operatorname* { m a x } _ { \pi } \left( Q _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \pi } + Y Q _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } \right) ,
$$

and then forms a statewise softmax candidate from the corresponding optimistic reward-plus-utility scores. The bonus makes planning optimistic and uncertainty-aware, exactly as in REP-PD-TEG; the stopping rule is different. REP-PD-H-TEG deploys the first candidate whose lower-confidence certificate satisfies $\mathcal { U } _ { k } ( \pi _ { k , Y } ) \geq b$ . If no candidate is certified before the cap $Y = K ^ { 1 / 4 }$ , the algorithm deploys the final capped-dual candidate. Thus optimistic bonus-augmented values are used to search for good policies, while the pessimistic bonus-based certificate decides whether a candidate is safe enough to execute.

Capped certificate search in one outer round.

1: For each grid level $Y \le Y _ { \mathrm { m a x } } ,$ , compute the optimistic candidate $\pi _ { k , Y }$

2: Evaluate ${ \underline { { \mathcal { U } } } } _ { k } ( \pi _ { k , Y } )$ ; if it is at least b, deploy $\pi _ { k , Y }$ and stop.

3: Otherwise increase Y and repeat the optimistic-plan/certificate test.

4: If no earlier candidate passes, deploy the candidate at $Y _ { \mathrm { m a x } }$

## 4.2 Hard-Violation Guarantees

We state the sample complexity result of REP-PD-H-TEG in this section. The proof decomposes into a front-end regret bound, a lower-certificate reduction, and a Slater-induced search-success argument; the full proof is given in Appendices F and G.

<table><tr><td>Assumption/interface</td><td>Role in the proof</td><td>Results</td></tr><tr><td>Low rank + finite realizabil- ity</td><td>MLE prediction event and one-step-back control</td><td>Soft + hard</td></tr><tr><td>Three-stream coverage</td><td>Controls both propagated terms from the merged covariance</td><td>Soft + hard</td></tr><tr><td>Slater margin  $\theta > 0$ </td><td>Bounds the dual variable and cap-induced deficit</td><td>Soft + hard</td></tr><tr><td>Positive score gap</td><td>Controls the hard-search softmax approximation</td><td>Hard only</td></tr></table>

The positive score gap is not used to learn the representation or to establish the lower-certificate inequality; it is used by the current hard proof to control the softmax approximation along the capped dual search.

Theorem 4.1 (Return policy and execution policy hard guarantees for REP-PD-H-TEG). Suppose Assumptions 2.3, 2.5, and C.1 hold, the finite model class is realizable $( | { \mathcal { M } } | < \infty , \ P ^ { \star } \in \ M )$ and fix $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ Run Algorithm 2 with $\epsilon _ { k } \ = \ k ^ { - p }$ for some $p ~ \in ~ [ 0 , 1 / 2 )$ , and set $\alpha _ { k } =$ $\widetilde { \Theta } ( \sqrt { ( | A | k ^ { p } + d ^ { 2 } ) \gamma \log ( | \mathcal { M } | k / \delta ) } )$ with a sufficiently large leading constant, $\lambda _ { k } = \widetilde { \Theta } ( ( | \mathcal { A } | k ^ { p } +$ $\begin{array} { r } { d ^ { 2 } ) / d ) \log ( | { \mathcal { M } } | k / \delta ) ) , \bar { b } \geq \operatorname* { m a x } _ { k \leq K } \alpha _ { k } / \sqrt { \lambda _ { k } } , \ell _ { k } = C _ { \mathrm { c e r t } } \sqrt { \Gamma _ { k } \zeta _ { k } / ( 1 - \gamma ) } } \end{array}$ for a sufficiently large universal $C _ { \mathrm { c e r t } } , Y _ { \mathrm { m a x } } = K ^ { 1 / 4 } , \tau _ { h } = \Theta ( 1 / ( ( 1 - \gamma ) \sqrt { K } \log | { \cal A } | ) ) , \eta _ { h } = \Theta ( \tau _ { h } ( 1 - \gamma ) / K ^ { 3 / 4 } )$ . Then, with probability at least $1 - \delta ,$ the return policies satisfy

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \left( \left[ \frac { | A | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | A | } { 1 - \gamma } } \right] K ^ { 3 / 4 + p } \right) ,
$$

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde { \cal O } \left( \frac { 1 } { \theta } \left[ \frac { | A | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | A | } { 1 - \gamma } } \right] K ^ { 1 / 2 + p } \right) + { \cal O } ( \theta ^ { - 4 } ) .
$$

The hard-violation bound is sublinearfor $p < 1 / 2 ,$ , while the regret bound is sublinear only when $p < 1 / 4$ . Moreover,for the execution policy $\widetilde { \pi } _ { k } = \widetilde { \left( 1 - \epsilon _ { k } \right) } \pi _ { k } + \widetilde { \epsilon _ { k } } U ( A )$ , we have Regre ${ \mathrm { , } } ^ { \mathrm { e x e c } } ( K ) \leq$ Regret $\textstyle \mathrm { r e t u r n } ( K ) + { \frac { 2 } { 1 - \gamma } } \sum _ { k = 1 } ^ { K } k ^ { - p }$ and Violatio $\begin{array} { r } { \mathsf { \Pi } _ { \mathrm { h a r d } } ^ { \mathrm { e x e c } } ( K ) \le \mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) + \frac { 2 } { 1 - \gamma } \sum _ { k = 1 } ^ { K } k ^ { - p } } \end{array}$

The $K ^ { 1 / 2 + p }$ hard-violation term is the representation-learning price, and the extra $K ^ { 1 / 4 }$ in regret comes from capped in-episode dual search; the execution policy bounds follow similar performancedifference argument from Section 3.

Corollary 4.2 (Best return policy and execution policy hard rates). Under the same setting as Theorem 4.1, thefollowing two specializations hold.

• First, $i f p = 0 ,$ , equivalently $\epsilon _ { k } \equiv 1$ , then REP-PD-H-TEG uses $q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } = U ( A )$ and the return policies satisfy

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \left( \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] K ^ { 3 / 4 } \right) ,
$$

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde O \left( \frac { 1 } { \theta } \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] \sqrt { K } \right) + O ( \theta ^ { - 4 } ) .
$$

• Second, $i f p = 1 / 8 ,$ , then the return policies satisfy

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \left( \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] K ^ { 7 / 8 } \right) ,
$$

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde O \left( \frac { 1 } { \theta } \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] K ^ { 5 / 8 } \right) + O ( \theta ^ { - 4 } ) .
$$

For $p = 1 / 8 ,$ the execution-policy gap is $O ( K ^ { 7 / 8 } / ( 1 - \gamma ) )$ ; hence both execution-policy regret and hard violation are ${ \widetilde O } ( K ^ { 7 / 8 } )$ up to problem-dependentfactors, with exploratory actions determining the balanced rate.

Thus $p = 0$ is sharp for return policies, whereas $p = 1 / 8$ balances the hard-regret front end $K ^ { 3 / 4 + p }$ with the execution gap $K ^ { 1 - p }$

## 5 Conclusion

We develop representation-learning algorithms for low-rank continuous-state CMDPs with unknown transition features. REP-PD-TEG combines auxiliary probing, covariance bonuses, and optimistic primal-dual planning for soft-violation control; REP-PD-H-TEG adds a lower-confidence utility certificate and stabilized dual search for hard-violation control. Both provide return- and executionpolicy guarantees.

Limitations. The theorems assume finite realizability and exact MLE/planning; alternative learners require the same three-design prediction guarantee. The hard result additionally uses a positive optimistic-score gap. The occupancy oracle is implementable with resets and geometric roll-ins, but our bounds do not charge full roll-in trajectories. Finally, the experiments illustrate the mechanism rather than exhaustively evaluating hard certification.

# Acknowledgments and Disclosure of Funding

Funding. There was no funding in direct support of this work.

Competing interests. The authors declare no competing interests.

## References

[1] Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International conference on machine learning, pages 22–31. PMLR, 2017.

[2] Alekh Agarwal, Sham Kakade, Akshay Krishnamurthy, and Wen Sun. Flambe: Structural complexity and representation learning of low rank mdps. Advances in neural information processing systems, 33:20095–20107, 2020.

[3] Alekh Agarwal, Sham M Kakade, Jason D Lee, and Gaurav Mahajan. On the theory of policy gradient methods: Optimality, approximation, and distribution shift. Journal of Machine Learning Research, 22(98):1–76, 2021.

[4] Eitan Altman. Constrained Markov decision processes. Routledge, 2021.

[5] Sanae Amani, Christos Thrampoulidis, and Lin Yang. Safe reinforcement learning with linear function approximation. In International Conference on Machine Learning, pages 243–253. PMLR, 2021.

[6] Yarden As, Ilnura Usmanova, Sebastian Curi, and Andreas Krause. Constrained policy optimization via bayesian world models. arXiv preprint arXiv:2201.09802, 2022.

[7] Arko Banerjee, Kia Rahmani, Joydeep Biswas, and Isil Dillig. Dynamic model predictive shielding for provably safe reinforcement learning. Advances in Neural Information Processing Systems, 37:100131–100159, 2024.

[8] Kianté Brantley, Miro Dudik, Thodoris Lykouris, Sobhan Miryoosefi, Max Simchowitz, Aleksandrs Slivkins, and Wen Sun. Constrained episodic reinforcement learning in concave-convex and knapsack settings. Advances in Neural Information Processing Systems, 33:16315–16326, 2020.

[9] Archana Bura, Aria HasanzadeZonuzy, Dileep Kalathil, Srinivas Shakkottai, and Jean-Francois Chamberland. Dope: Doubly optimistic and pessimistic exploration for safe reinforcement learning. Advances in neural information processing systems, 35:1047–1059, 2022.

[10] Yuan Cheng, Ruiquan Huang, Jing Yang, and Yingbin Liang. Improved sample complexity for reward-free reinforcement learning under low-rank mdps. arXiv preprint arXiv:2303.10859, 2023.

[11] Juntao Dai, Jiaming Ji, Long Yang, Qian Zheng, and Gang Pan. Augmented proximal policy optimization for safe reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 7288–7295, 2023.

[12] Dongsheng Ding, Kaiqing Zhang, Tamer Basar, and Mihailo R Jovanovic. Natural policy gradient primal-dual method for constrained markov decision processes. In NeurIPS, 2020.

[13] Dongsheng Ding, Xiaohan Wei, Zhuoran Yang, Zhaoran Wang, and Mihailo Jovanovic. Provably efficient safe exploration via primal-dual policy optimization. In International Conference on Artificial Intelligence and Statistics, pages 3304–3312. PMLR, 2021.

[14] Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, and Alejandro Ribeiro. Last-iterate convergent policy gradient primal-dual methods for constrained mdps. Advances in Neural Information Processing Systems, 36:66138–66200, 2023.

[15] Simon Du, Akshay Krishnamurthy, Nan Jiang, Alekh Agarwal, Miroslav Dudik, and John Langford. Provably efficient rl with rich observations via latent state decoding. In International Conference on Machine Learning, pages 1665–1674. PMLR, 2019.

[16] Yonathan Efroni, Shie Mannor, and Matteo Pirotta. Exploration-exploitation in constrained mdps. arXiv preprint arXiv:2003.02189, 2020.

[17] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Provably efficient model-free constrained rl with linear function approximation. Advances in Neural Information Processing Systems, 35: 13303–13315, 2022.

[18] Arnob Ghosh, Xingyu Zhou, and Ness Shroff. Towards achieving sub-linear regret and hard constraint violation in model-free rl. In International Conference on Artificial Intelligence and Statistics, pages 1054–1062. PMLR, 2024.

[19] Shangding Gu, Long Yang, Yali Du, Guang Chen, Florian Walter, Jun Wang, and Alois Knoll. A review of safe reinforcement learning: Methods, theories and applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[20] Ruiquan Huang, Jing Yang, and Yingbin Liang. Safe exploration incurs nearly no additional sample complexity for reward-free rl. arXiv preprint arXiv:2206.14057, 2022.

[21] Weidong Huang, Jiaming Ji, Chunhe Xia, Borong Zhang, and Yaodong Yang. Safedreamer: Safe reinforcement learning with world models. arXiv preprint arXiv:2307.07176, 2023.

[22] Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, et al. Ai alignment: A comprehensive survey. arXiv preprint arXiv:2310.19852, 2023.

[23] Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, John Langford, and Robert E Schapire. Contextual decision processes with low bellman rank are pac-learnable. In International Conference on Machine Learning, pages 1704–1713. PMLR, 2017.

[24] Krishna C Kalagarla, Rahul Jain, and Pierluigi Nuzzo. A sample-efficient algorithm for episodic finite-horizon mdp with constraints. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 8030–8037, 2021.

[25] Toshinori Kitamura, Tadashi Kozuno, Masahiro Kato, Yuki Ichihara, Soichiro Nishimori, Akiyoshi Sannai, Sho Sonoda, Wataru Kumagai, and Yutaka Matsuo. A policy gradient primal-dual algorithm for constrained mdps with uniform pac guarantees. arXiv preprint arXiv:2401.17780, 2024.

[26] Toshinori Kitamura, Arnob Ghosh, Tadashi Kozuno, Wataru Kumagai, Kazumi Kasaura, Kenta Hoshino, Yohei Hosoe, and Yutaka Matsuo. Provably efficient rl under episode-wise safety in linear cmdps. arXiv preprint arXiv:2502.10138, 2025.

[27] Tao Liu, Ruida Zhou, Dileep Kalathil, Panganamala Kumar, and Chao Tian. Learning policies with zero or bounded constraint violation for constrained mdps. Advances in Neural Information Processing Systems, 34:17183–17193, 2021.

[28] Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, and John Langford. Kinematic state abstraction and provably efficient rich-observation reinforcement learning. In International conference on machine learning, pages 6961–6971. PMLR, 2020.

[29] Aditya Modi, Jinglin Chen, Akshay Krishnamurthy, Nan Jiang, and Alekh Agarwal. Modelfree representation learning and exploration in low-rank mdps. Journal of Machine Learning Research, 25(6):1–76, 2024.

[30] Gergely Neu and Ciara Pike-Burke. A unifying view of optimism in episodic reinforcement learning. Advances in Neural Information Processing Systems, 33:1392–1403, 2020.

[31] Chengzhuo Ni, Yuda Song, Xuezhou Zhang, Chi Jin, and Mengdi Wang. Representation learning for general-sum low-rank markov games. arXiv preprint arXiv:2210.16976, 2022.

[32] Santiago Paternain, Miguel Calvo-Fullana, Luiz FO Chamon, and Alejandro Ribeiro. Safe policies for reinforcement learning via primal-dual methods. IEEE Transactions on Automatic Control, 68(3):1321–1336, 2022.

[33] Shuang Qiu, Xiaohan Wei, Zhuoran Yang, Jieping Ye, and Zhaoran Wang. Upper confidence primal-dual optimization: Stochastically constrained markov decision processes with adversarial losses and unknown transitions. arXiv preprint arXiv:2003.00660, 2020.

[34] Shuang Qiu, Lingxiao Wang, Chenjia Bai, Zhuoran Yang, and Zhaoran Wang. Contrastive ucb: Provably efficient contrastive self-supervised learning in online reinforcement learning. In International Conference on Machine Learning, pages 18168–18210. PMLR, 2022.

[35] Alex Ray, Joshua Achiam, and Dario Amodei. Benchmarking safe exploration in deep reinforcement learning. arXiv preprint arXiv:1910.01708, 7(1):2, 2019.

[36] Tongzheng Ren, Chenjun Xiao, Tianjun Zhang, Na Li, Zhaoran Wang, Sujay Sanghavi, Dale Schuurmans, and Bo Dai. Latent variable representation for reinforcement learning. arXiv preprint arXiv:2212.08765, 2022.

[37] Tongzheng Ren, Tianjun Zhang, Lisa Lee, Joseph E Gonzalez, Dale Schuurmans, and Bo Dai. Spectral decomposition representation for reinforcement learning. arXiv preprint arXiv:2208.09515, 2022.

[38] Tongzheng Ren, Tianjun Zhang, Csaba Szepesvári, and Bo Dai. A free lunch from the noise: Provable and practical exploration for representation learning. In Uncertainty in Artificial Intelligence, pages 1686–1696. PMLR, 2022.

[39] Rahul Singh, Abhishek Gupta, and Ness B Shroff. Learning in markov decision processes under constraints. arXiv preprint arXiv:2002.12435, 2020.

[40] Wen Sun, Nan Jiang, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Model-based rl in contextual decision processes: Pac bounds and exponential improvements over model-free approaches. In Conference on learning theory, pages 2898–2933. PMLR, 2019.

[41] Masatoshi Uehara, Xuezhou Zhang, and Wen Sun. Representation learning for online and offline rl in low-rank mdps. arXiv preprint arXiv:2110.04652, 2021.

[42] Honghao Wei, Xin Liu, and Lei Ying. Safe reinforcement learning with instantaneous constraints: the role of aggressive exploration. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pages 21708–21716, 2024.

[43] Yunchang Yang, Tianhao Wu, Han Zhong, Evrard Garcelon, Matteo Pirotta, Alessandro Lazaric, Liwei Wang, and Simon S Du. A reduction-based framework for conservative bandits and reinforcement learning. arXiv preprint arXiv:2106.11692, 2021.

[44] Hongming Zhang, Tongzheng Ren, Chenjun Xiao, Dale Schuurmans, and Bo Dai. Provable representation with efficient planning for partially observable reinforcement learning. arXiv preprint arXiv:2311.12244, 2023.

[45] Tianjun Zhang, Tongzheng Ren, Mengjiao Yang, Joseph Gonzalez, Dale Schuurmans, and Bo Dai. Making linear mdps practical via contrastive representation learning. In International Conference on Machine Learning, pages 26447–26466. PMLR, 2022.

[46] Xuezhou Zhang, Yuda Song, Masatoshi Uehara, Mengdi Wang, Alekh Agarwal, and Wen Sun. Efficient reinforcement learning in block mdps: A model-free representation learning approach. In International Conference on Machine Learning, pages 26517–26547. PMLR, 2022.

[47] Chenhao Zhou and Jiyu Wei. Toward online robust zero-sum markov games with function approximation. arXiv preprint arXiv:2610.26286, 2026. Accepted at NeurIPS 2026.

[48] Chenhao Zhou, Zebang Shen, Hanbin Zhao, Hui Qian, et al. Solving zero-sum markov games with continuous state via spectral dynamic embedding. Advances in Neural Information Processing Systems, 37:68622–68657, 2024.

## Appendix Roadmap

The appendix is organized as follows. Appendix A fixes notation and records the basic covariance concentration tool. Appendix B collects the MLE, elliptical-potential, simulation, and primal-dual lemmas used throughout the proofs. Appendix C states the technical ingredients specific to the hard-violation analysis. Appendix D proves the tertiary-aware decaying-ϵ representation-learning result for REP-PD-TEG, while Appendix E gives the $p = 0$ uniform specialization. Appendix F derives the one-sided hard certificate and proves the decaying-schedule result, while Appendix G records its $p = 0$ specialization. Appendix H contains the experimental details.

## A Notation and Auxiliary Concentration

This appendix fixes the occupancy notation used throughout the proof and recalls the empirical-topopulation covariance comparison that underlies the uncertainty bonus. Later appendices replace the uniform probing distribution below by the tertiary decaying-ϵ design, but the same comparison principle is repeatedly used.

We use $d _ { P } ^ { \pi }$ <sub>⋆</sub> for the normalized discounted state occupancy and $d _ { P ^ { \star } } ^ { \pi } ( s , a ) = d _ { P ^ { \star } } ^ { \pi } ( s ) \pi ( a \mid s )$ for the state-action occupancy. For the uniform probing baseline, let $\begin{array} { r } { \rho _ { k } ( s ) = k ^ { - 1 } \sum _ { i = 0 } ^ { k - 1 } d _ { P ^ { \star } } ^ { \pi _ { i } } ( s ) } \end{array}$ , and let $\rho _ { k } ^ { \prime }$ be the marginal distribution of $s ^ { \prime }$ generated by first drawing $s \sim \rho _ { k }$ , then $a \sim U ( A )$ , and finally $s ^ { \tilde { \prime } } \sim P ^ { \star } ( \cdot \mid s , a )$ . For a feature map ϕ, define

$$
\begin{array} { r } { \sum _ { \rho _ { k } \times U ( A ) , \phi } : = k \mathbb { E } _ { s \sim \rho _ { k } , a \sim U ( A ) } [ \phi ( s , a ) \phi ( s , a ) ^ { \top } ] + \lambda _ { k } I . } \end{array}
$$

Lemma A.1 (Concentration of the bonus term). Let

$$
\hat { \Sigma } _ { k , \phi } : = \sum _ { i = 0 } ^ { k - 1 } \phi ( s ^ { ( i ) } , a ^ { ( i ) } ) \phi ( s ^ { ( i ) } , a ^ { ( i ) } ) ^ { \top } + \lambda _ { k } I , \qquad \lambda _ { k } = \Theta ( d \log ( k | \Phi | / \delta ) ) .
$$

Then, with probability at least $1 - \delta ,$ , simultaneously for all $k \in \mathbb { K } ^ { + }$ , all $\phi \in \Phi$ , and all $( s , a )$

$$
c _ { 1 } \| \phi ( s , a ) \| _ { { \Sigma } _ { \rho _ { k } \times U ( A ) , \phi } ^ { - 1 } } \leq \| \phi ( s , a ) \| _ { \widehat { \Sigma } _ { k , \phi } ^ { - 1 } } \leq c _ { 2 } \| \phi ( s , a ) \| _ { { \Sigma } _ { \rho _ { k } \times U ( A ) , \phi } ^ { - 1 } } ,
$$

where $c _ { 1 } , c _ { 2 } > 0$ are universal constants.

Proof. This is the standard empirical-to-population covariance comparison for the uniform probing design; see Lemma 11 of Uehara et al. [41]. In the repaired analysis we apply the same concentration argument to block increments generated by the predictable auxiliary streams. □

## B Auxiliary Lemmas

This appendix collects standard auxiliary tools used by both the soft and hard analyses. We first recall the MLE guarantee for low-rank transition models, then state the elliptical-potential and simulation lemmas used to aggregate model error and convert value differences into regret and violation bounds. The MLE statement is a direct consequence of Theorem 21 of Agarwal et al. [2], specialized to the notation of our algorithms.

Lemma B.1 (MLE guarantee). For a fixed episode k, with probability $1 - \delta ,$

$$
\mathbb { E } _ { s \sim \{ 0 . 5 \rho _ { k } + 0 . 5 \rho _ { k } ^ { \prime } \} , a \sim U ( \mathcal { A } ) } [ \| \hat { P } _ { k } ( \cdot \vert s , a ) - P ^ { \star } ( \cdot \vert s , a ) \| _ { 1 } ^ { 2 } ] \lesssim \zeta ,
$$

where $\begin{array} { r } { \zeta : = \frac { \ln ( | \mathcal { M } | / \delta ) } { k } } \end{array}$ . As a straightforward corollary, with probability $1 - \delta , \forall k \in \mathbb { K } ^ { + }$

$$
\mathbb { E } _ { s \sim \{ 0 . 5 \rho _ { k } + 0 . 5 \rho _ { k } ^ { \prime } \} , a \sim U ( A ) } [ \| \hat { P } _ { k } ( \cdot \vert s , a ) - P ^ { \star } ( \cdot \vert s , a ) \| _ { 1 } ^ { 2 } ] \lesssim 0 . 5 \zeta _ { k } ,
$$

where $\begin{array} { r } { \zeta _ { k } : = \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } } \end{array}$

The next two lemmas are the standard determinant-trace inequalities used in the elliptical-potential argument; see Lemma G.2 of Agarwal et al. [2].

Lemma B.2. Consider thefollowing process. For $k = 1 , \cdot \cdot \cdot , K , M _ { k } = M _ { k - 1 } + G _ { k }$ with $M _ { 0 } = \lambda _ { 0 } I$ and $G _ { k }$ being a positive semidefinite matrix with eigenvalues upper-bounded by 1. We have that:

$$
2 \ln \operatorname* { d e t } ( M _ { K } ) - 2 \ln \operatorname* { d e t } ( \lambda _ { 0 } I ) \geq \sum _ { k = 1 } ^ { K } T r ( G _ { k } M _ { k - 1 } ^ { - 1 } ) .
$$

Lemma B.3 (Potential function lemma). Suppose $T r ( G _ { k } ) \le B ^ { 2 }$

$$
2 \ln \operatorname * { d e t } ( M _ { K } ) - 2 \ln \operatorname * { d e t } ( \lambda _ { 0 } I ) \leq d \ln \left( 1 + \frac { K B ^ { 2 } } { d \lambda _ { 0 } } \right) .
$$

Lemma B.4 (Simulation lemma). For any $c \in \{ r , g \}$ , two CMDPs $( P ^ { \prime } , c + b )$ and $( P , c )$ , and any policy π, we have

$$
V _ { P ^ { \prime } , c + b } ^ { \pi } - V _ { P , c } ^ { \pi } = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \prime } } ^ { \pi } } \left[ b ( s , a ) + \gamma \mathbb { E } _ { P ^ { \prime } } Q _ { P , c } ^ { \pi } ( s ^ { \prime } , \pi ) - \gamma \mathbb { E } _ { P } Q _ { P , c } ^ { \pi } ( s ^ { \prime } , \pi ) \right] ,
$$

$$
V _ { P ^ { \prime } , c + b } ^ { \pi } - V _ { P , c } ^ { \pi } = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P } ^ { \pi } } \left[ b ( s , a ) + \gamma \mathbb { E } _ { P ^ { \prime } } Q _ { P , c + b } ^ { \pi } ( s ^ { \prime } , \pi ) - \gamma \mathbb { E } _ { P } Q _ { P ^ { \prime } , c + b } ^ { \pi } ( s ^ { \prime } , \pi ) \right] .
$$

Finally, we use the following primal-dual reduction, which is Lemma 9 of Ding et al. [13] in our notation.

Lemma B.5. Let $Y ^ { * }$ be the optimal dual variable, and $C \geq 2 Y ^ { * }$ , then, $i f$

$$
V _ { r } ^ { \pi ^ { * } } ( s _ { 1 } ) - V _ { r } ^ { \tilde { \pi } } ( s _ { 1 } ) + C [ b - V _ { g } ^ { \tilde { \pi } } ( s _ { 1 } ) ] _ { + } \leq \delta
$$

then

$$
[ b - V _ { g } ^ { \tilde { \pi } } ( s _ { 1 } ) ] _ { + } \leq \frac { 2 \delta } { C } .
$$

## C Hard-Constraint Technical Lemmas

This appendix collects the hard-constraint technical ingredients that are reused by the lower-certificate analyses in Appendices G and F. The hard-violation guarantees in this draft are stated through lower-confidence certificates and Slater-induced control of cap episodes; no optimistic feasibility check is used as a hard certificate.

Assumption C.1 (Positive action gap). For every episode $k ,$ queried dual level $Y ,$ , and state $s ,$ the maximizer of the optimistic Lagrangian action score used by Algorithm 2 is separated from every suboptimal action by at least $\delta _ { \mathrm { m i n } } > 0$ . The constant is uniform over $k , Y , s$

Lemma C.2 (Softmax sensitivity). Let $\pi _ { k } ^ { Y }$ and $\pi _ { k } ^ { Y + \eta }$ be the softmax policies produced by two neighboring dual levels Y and $Y + \eta$ with temperature τ. Then

$$
\left| V _ { g } ^ { \pi _ { k } ^ { Y } } \left( s _ { 1 } \right) - V _ { g } ^ { \pi _ { k } ^ { Y + \eta } } ( s _ { 1 } ) \right| \le O \left( \frac { \eta } { \tau ( 1 - \gamma ) } \right) .
$$

Proof. Let $\begin{array} { r } { \Delta = \operatorname* { s u p } _ { s } | V _ { g } ^ { \pi _ { k } ^ { Y } } ( s ) - V _ { g } ^ { \pi _ { k } ^ { Y + \eta } } ( s ) | } \end{array}$ . Comparing the two Bellman equations gives

$$
\Delta \le \gamma \Delta + \operatorname* { s u p } _ { s } \| \pi _ { k } ^ { Y } ( \cdot \mid s ) - \pi _ { k } ^ { Y + \eta } ( \cdot \mid s ) \| _ { 1 } .
$$

The softmax map is $O ( 1 / \tau )$ -Lipschitz in its input. Increasing the dual level by η changes the softmax score by ${ \cal { O } } ( \eta )$ under the normalized utility scale, and therefore

$$
\operatorname* { s u p } _ { s } \| \pi _ { k } ^ { Y } ( \cdot \mid s ) - \pi _ { k } ^ { Y + \eta } ( \cdot \mid s ) \| _ { 1 } \leq O ( \eta / \tau ) .
$$

Rearranging the Bellman inequality proves the claim.

## D Decaying ϵ-Greedy Probing: Tertiary-Aware Analysis

This appendix proves the decaying-ϵ tertiary-aware result used in Section 3. The proof is organized around three steps:

1. the naive two-level decaying-ϵ probing rule is not enough to close the optimism step;

2. a minimal tertiary-aware repair restores the missing denominator;

3. for this repaired algorithm, one can prove a regret guarantee for the return policy sequence $\{ \pi _ { k } \} _ { k = 1 } ^ { K }$

The guarantees in this appendix are for the return policies of Algorithm 1. The execution-policy transfer in Section 3 is obtained afterward through a performance-difference argument. The same tertiary-aware front end also supports soft-violation control once the cumulative optimistic overestima tion of the utility is bounded explicitly. The hard-violation analysis in later appendices uses the same representation-learning front end, but replaces optimistic feasibility checks with a lower-confidence certificate.

## D.1 Why the Naive Two-Level Variant Is Not Enough

The natural first attempt is the two-level decaying-ϵ rule

$$
q _ { i } ( \cdot \mid s ) = q _ { i } ^ { \prime } ( \cdot \mid s ) = ( 1 - \epsilon _ { i } ) \pi _ { i - 1 } ( \cdot \mid s ) + \epsilon _ { i } U ( { \cal A } ) .
$$

This already improves the sampling rule relative to pure uniform probing, but it does not fully solve the coverage problem needed by the optimism proof.

Proposition D.1 (Two-level obstruction). There is no universal constant $C < \infty$ such that the pointwise comparison

$$
\| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \mathrm { p r i } } ) ^ { - 1 } } \leq C \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } , \qquad \forall ( s , a ) ,
$$

holds for every episode k under the naive two-level decaying-ϵ probing rule.

Proof. Consider a deterministic two-state MDP with states $s _ { 0 } , s _ { 1 }$ and two actions $a _ { \mathrm { s t a y } } , a _ { \mathrm { j u m p } } .$ . Let the deployed policy always choose $a _ { \mathrm { s t a y } }$ at $s _ { 0 }$ , let $s _ { 1 }$ be absorbing, and let $a _ { \mathrm { j u m p } }$ be the only action that can move the process from $s _ { 0 }$ to s<sub>1</sub>. Choose a bounded feature map such that

$$
\hat { \phi } _ { k } \mathopen { } \mathclose \bgroup \left( s _ { 0 } , a \aftergroup \egroup \right) = e _ { 1 } , \qquad \hat { \phi } _ { k } \mathopen { } \mathclose \bgroup \left( s _ { 1 } , a \aftergroup \egroup \right) = e _ { 2 } , \qquad \forall a \in \mathcal { A } .
$$

Then all primary samples stay at $s _ { 0 } .$ , so

$$
\begin{array} { r } { \Sigma _ { k } ^ { \mathrm { p r i } } = k e _ { 1 } e _ { 1 } ^ { \top } + \lambda _ { k } I . } \end{array}
$$

However, the secondary stream reaches $s _ { 1 }$ with probability of order $\epsilon _ { k } / | \mathcal { A } |$ , hence

$$
\Sigma _ { k } ^ { \mathrm { m i x } } \succeq k e _ { 1 } e _ { 1 } ^ { \top } + \frac { k \epsilon _ { k } } { | \mathcal { A } | } e _ { 2 } e _ { 2 } ^ { \top } + \lambda _ { k } I .
$$

Evaluating the two norms at any point with feature vector $e _ { 2 }$ gives

$$
\| e _ { 2 } \| _ { ( \Sigma _ { k } ^ { \mathrm { p r i } } ) ^ { - 1 } } = \frac { 1 } { \sqrt { \lambda _ { k } } } , \qquad \| e _ { 2 } \| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } \leq \frac { 1 } { \sqrt { \lambda _ { k } + k \epsilon _ { k } / | \cal { A } | } } .
$$

Therefore

$$
\frac { \left\| e _ { 2 } \right\| _ { ( \Sigma _ { k } ^ { \mathrm { p r i } } ) ^ { - 1 } } } { \left\| e _ { 2 } \right\| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } } \ge \sqrt { 1 + \frac { k \epsilon _ { k } } { | \mathcal { A } | \lambda _ { k } } } ,
$$

which rules out a universal constant bridge.

Proposition D.1 explains why the clean theorem is not for the naive two-level rule. The minimal repair for the present merged-covariance design is to add one more probe level. This obstruction does not rule out a primary-only alternative that fits on $D + D ^ { \prime }$ but constructs its covariance from D alone, as in the architecture of Uehara et al. [41]. Such a design removes the secondary covariance term but requires a different potential argument; the current regret aggregation cannot be transferred to it directly.

## D.2 The Tertiary-Aware Repaired Algorithm

For each round $i \in [ K ]$ , define the primary, secondary, and tertiary state marginals by

$$
\rho _ { i } ( s ) : = d _ { P ^ { \star } } ^ { \pi _ { i - 1 } } ( s ) ,
$$

$$
\rho _ { i } ^ { \prime } ( s ^ { \prime } ) : = \mathbb { E } _ { ( s , a ) \sim \rho _ { i } \times q _ { i } } \big [ P ^ { \star } ( s ^ { \prime } \mid s , a ) \big ] ,
$$

$$
\rho _ { i } ^ { \prime \prime } ( s ^ { \prime \prime } ) : = \mathbb { E } _ { ( s , a ) \sim \rho _ { i } ^ { \prime } \times q _ { i } ^ { \prime } } \left[ P ^ { \star } ( s ^ { \prime \prime } \mid s , a ) \right] .
$$

The corresponding probing distributions are

$$
\nu _ { i } ( s , a ) : = \rho _ { i } ( s ) q _ { i } ( a \mid s ) , \quad \nu _ { i } ^ { \prime } ( s , a ) : = \rho _ { i } ^ { \prime } ( s ) q _ { i } ^ { \prime } ( a \mid s ) , \quad \nu _ { i } ^ { \prime \prime } ( s , a ) : = \rho _ { i } ^ { \prime \prime } ( s ) q _ { i } ^ { \prime \prime } ( a \mid s ) .
$$

We write

$$
\bar { \nu } _ { k } ^ { \mathrm { p r i } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } , \qquad \bar { \nu } _ { k } ^ { \mathrm { s e c } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } ^ { \prime } , \qquad \bar { \nu } _ { k } ^ { \mathrm { t e r } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } ^ { \prime \prime } ,
$$

and

$$
\bar { \nu } _ { k } ^ { \mathrm { a l l } } : = \frac { 1 } { 3 } \big ( \bar { \nu } _ { k } ^ { \mathrm { p r i } } + \bar { \nu } _ { k } ^ { \mathrm { s e c } } + \bar { \nu } _ { k } ^ { \mathrm { t e r } } \big ) .
$$

The repaired probing kernels are

$$
q _ { k } ( \cdot \mid s ) = q _ { k } ^ { \prime } ( \cdot \mid s ) = q _ { k } ^ { \prime \prime } ( \cdot \mid s ) = ( 1 - \epsilon _ { k } ) \pi _ { k - 1 } ( \cdot \mid s ) + \epsilon _ { k } U ( \mathcal { A } ) ,
$$

where

$$
\epsilon _ { 1 } \geq \epsilon _ { 2 } \geq \cdot \cdot \cdot \geq \epsilon _ { K } > 0 .
$$

We also write

$$
\Gamma _ { k } : = \frac { | \cal { A } | } { \epsilon _ { k } } , \qquad \Lambda _ { k } : = \frac { 1 } { 1 - \epsilon _ { k } + \epsilon _ { k } / | \cal { A } | } .
$$

The coverage coefficients used below are domination constants between the target distributions appearing in the proof and the three empirical probing streams. For two state-action distributions $\eta , \nu ,$ define

$$
\begin{array} { r } { \mathsf { D } ( \eta \| \nu ) : = \operatorname* { i n f } \left\{ c \ge 0 : \mathbb { E } _ { \eta } [ g ^ { 2 } ] \le c \mathbb { E } _ { \nu } [ g ^ { 2 } ] \mathrm { ~ f o r ~ a l l ~ } g \right\} . } \end{array}
$$

When densities exist, this is the essential supremum of $d \eta / d \nu$ . For a state-action distribution $\nu ,$ let ${ \sf T } _ { \pi } \nu$ denote the one-step pushforward obtained by drawing $( { \tilde { s } } , { \tilde { a } } ) \sim \nu , s \sim P ^ { \star } ( \cdot \mid { \tilde { s } } , { \tilde { a } } )$ , and $a \sim \pi ( \cdot \mid s )$ . With this notation,

$$
\begin{array} { r l } & { \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) : = \mathrm { D } \Big ( ( 1 - \gamma ) d _ { 1 } \otimes \pi \Big \Vert \bar { \nu } _ { k } ^ { \mathrm { p r i } } \Big ) , } \\ & { \mathfrak { C } _ { k } ^ { \mathrm { n e x t } } ( \pi ) : = \mathrm { D } \Big ( \top _ { \pi } \bar { \nu } _ { k } ^ { \mathrm { p r i } } \left. \bar { \nu } _ { k } ^ { \mathrm { s e c } } \right) , } \\ & { \mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i , n e x t } } ( \pi ) : = \mathfrak { C } _ { k } ^ { \mathrm { n e x t } } ( \pi ) , } \\ & { \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i , n e x t } } ( \pi ) : = \mathrm { D } \big ( \top _ { \pi } \bar { \nu } _ { k } ^ { \mathrm { s e c } } \big \Vert \bar { \nu } _ { k } ^ { \mathrm { t e r } } \big ) , } \\ & { \mathfrak { C } _ { k } ^ { \mathrm { t r i , n e x t } } ( \pi ) : = \operatorname* { m a x } \Big \{ \mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i , n e x t } } ( \pi ) , \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i , n e x t } } ( \pi ) \Big \} . } \end{array}
$$

Thus ${ \mathfrak { C } } _ { k } ^ { \mathrm { i n i t } }$ compares the initial discounted mass to the primary stream, ${ \mathfrak { C } } _ { k } ^ { \mathrm { n e x t } }$ compares the primaryto-secondary one-step design, and $\mathfrak { C } _ { k } ^ { \mathrm { { t r i } , \mathrm { { n e x t } } } }$ is the maximum of the two one-step comparisons needed for the tertiary one-step-back argument.

The algorithm analyzed in this appendix is the main-text Algorithm 1. The repair relative to the earlier two-stream protocol is entirely in the auxiliary sampling design: it adds one more probe-action / transition pair, while keeping the bonus covariance merged over the primary and secondary tuples.

## D.3 Supporting Lemmas

We now record the small collection of lemmas needed for the final theorem.

Lemma D.2 (Three-stream MLE event at tuple level). Under realizability $P ^ { \star } \in { \mathcal { M } } ,$ , with probability at least $1 - \delta ,$ , simultaneouslyfor all $k \in [ \bar { K } ]$

$$
\mathbb { E } _ { ( s , a ) \sim \tilde { \nu } _ { k } ^ { \mathrm { a l l } } } \left[ \| \hat { P } _ { k } ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 } ^ { 2 } \right] \lesssim \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } = : \zeta _ { k } .
$$

Here

$$
\widetilde { \nu } _ { k } ^ { \mathrm { s e c } } : = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \delta _ { s _ { i } ^ { \prime } } q _ { i } ^ { \prime } ( \cdot \mid s _ { i } ^ { \prime } ) , \qquad \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } : = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \delta _ { s _ { i } ^ { \prime \prime } } q _ { i } ^ { \prime \prime } ( \cdot \mid s _ { i } ^ { \prime \prime } ) ,
$$

and

$$
\widetilde { \nu } _ { k } ^ { \mathrm { a l l } } : = \frac { 1 } { 3 } \big ( \bar { \nu } _ { k } ^ { \mathrm { p r i } } + \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } + \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } \big ) .
$$

$P r o o f .$ For episode $i ,$ write the realized tertiary tuple as $\left( { { s _ { i } } , { a _ { i } } , { s _ { i } ^ { \prime } } , { a _ { i } ^ { \prime } } , { s _ { i } ^ { \prime \prime } } , { a _ { i } ^ { \prime \prime } } , { { \bar { s } } _ { i } } } \right)$ , where $\bar { s } _ { i } \sim P ^ { \star } ( \cdot \ |$ $s _ { i } ^ { \prime \prime } , \bar { a _ { i } ^ { \prime \prime } } )$ is the next state of the third probe transition. Define

$$
\begin{array} { r l } { x _ { 3 i - 2 } : = ( s _ { i } , a _ { i } ) , } & { \quad y _ { 3 i - 2 } : = s _ { i } ^ { \prime } , } \\ { x _ { 3 i - 1 } : = ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) , } & { \quad y _ { 3 i - 1 } : = s _ { i } ^ { \prime \prime } , } \end{array}
$$

and

$$
x _ { 3 i } : = ( s _ { i } ^ { \prime \prime } , a _ { i } ^ { \prime \prime } ) , \qquad y _ { 3 i } : = \bar { s } _ { i } .
$$

Let $\mathcal { F } _ { 3 i - 3 }$ be the sigma-field generated by all data collected before episode i starts. Recursively define

$$
\mathscr { F } _ { 3 i - 2 } : = \sigma ( \mathscr { F } _ { 3 i - 3 } , x _ { 3 i - 2 } , y _ { 3 i - 2 } ) , \quad \mathscr { F } _ { 3 i - 1 } : = \sigma ( \mathscr { F } _ { 3 i - 2 } , x _ { 3 i - 1 } , y _ { 3 i - 1 } ) ,
$$

and

$$
\mathcal { F } _ { 3 i } : = \sigma ( \mathcal { F } _ { 3 i - 1 } , x _ { 3 i } , y _ { 3 i } ) .
$$

Then the predictable covariate designs are

$$
\begin{array} { r l } & { \mathrm { ~ } D _ { 3 i - 2 } : = \mathrm { L a w } ( x _ { 3 i - 2 } \mid \mathcal { F } _ { 3 i - 3 } ) = \nu _ { i } , } \\ & { \mathrm { ~ } D _ { 3 i - 1 } : = \mathrm { L a w } ( x _ { 3 i - 1 } \mid \mathcal { F } _ { 3 i - 2 } ) = \delta _ { s _ { i } ^ { \prime } } q _ { i } ^ { \prime } ( \cdot \mid s _ { i } ^ { \prime } ) , } \\ & { \mathrm { ~ } D _ { 3 i } : = \mathrm { L a w } ( x _ { 3 i } \mid \mathcal { F } _ { 3 i - 1 } ) = \delta _ { s _ { i } ^ { \prime \prime } } q _ { i } ^ { \prime \prime } ( \cdot \mid s _ { i } ^ { \prime \prime } ) . } \end{array}
$$

Hence the full covariate sequence

$$
x _ { 1 } , x _ { 2 } , \ldots , x _ { 3 k }
$$

is a predictable design sequence with respect to the tuple-level filtration $\{ \mathcal { F } _ { t } \} _ { t = 0 } ^ { 3 k }$ in the sense of $[ 2 ,$ Theorem 21]. Moreover, the responses satisfy

$$
y _ { t } \sim P ^ { \star } ( \cdot \mid x _ { t } ) \qquad { \mathrm { f o r } } { \mathrm { e v e r y } } t \leq 3 k ,
$$

and realizability gives $P ^ { \star } \in { \mathcal { M } }$

Apply Theorem 21 of [2] to the first 3k samples, with confidence level $\delta / k ^ { 2 }$ . This yields, for every fixed $k ,$

$$
\frac { 1 } { 3 k } \sum _ { t = 1 } ^ { 3 k } \mathbb { E } _ { \boldsymbol { x } \sim D _ { t } } \Big [ \| \hat { P } _ { k } ( \cdot \mid \boldsymbol { x } ) - P ^ { \star } ( \cdot \mid \boldsymbol { x } ) \| _ { 1 } ^ { 2 } \Big ] \lesssim \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } ,
$$

where $D _ { t }$ denotes the predictable design of the t-th tuple. By the definitions above,

$$
\frac { 1 } { 3 k } \sum _ { t = 1 } ^ { 3 k } D _ { t } = \frac { 1 } { 3 } \big ( \hat { \nu } _ { k } ^ { \mathrm { p r i } } + \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } + \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } \big ) = \widetilde { \nu } _ { k } ^ { \mathrm { a l l } } ,
$$

as a random probability measure. Therefore the left-hand side is precisely

$$
\mathbb { E } _ { ( s , a ) \sim \widetilde { \nu } _ { k } ^ { \mathrm { a l l } } } \left[ \| \hat { P } _ { k } ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 } ^ { 2 } \right] .
$$

Finally, a union bound over all $k \in [ K ]$ turns the fixed-k statement into the simultaneous event claimed in the lemma, with the extra ln k absorbed into $\zeta _ { k }$ □

Remark D.3 (Route choice: pathwise-to-averaged bridge). Lemma D.2 is the formally correct tuplelevel statement: the secondary and tertiary designs are the pathwise predictable kernels $\delta _ { s _ { i } ^ { \prime } } q _ { i } ^ { \prime } ( \cdot \mid ^ { \cdot } s _ { i } ^ { \prime } )$ and $\delta _ { s _ { i } ^ { \prime \prime } } q _ { i } ^ { \prime \prime } ( \cdot \mid s _ { i } ^ { \prime \prime } )$ , not the averaged covariates $\nu _ { i } ^ { \prime } , \nu _ { i } ^ { \prime \prime }$ directly. For the remainder of this appendix, we keep the coverage coefficients and theorem statements in terms of the deterministic averaged designs $\bar { \nu } _ { k } ^ { \mathrm { p r i } } , \bar { \nu } _ { k } ^ { \mathrm { s e c } }$ , and $\bar { \nu } _ { k } ^ { \mathrm { t e r } }$ , rather than rewriting every later lemma in pathwise notation. We therefore choose the pathwise-to-averaged bridge route: whenever the tuple-level MLE event is used under $\bar { \nu } _ { k } ^ { \mathrm { s e c } }$ or $\bar { \nu } _ { k } ^ { \mathrm { t e r } }$ we apply the bridge lemma below to the relevant squared mismatch class. This ensures simultaneous control under the random pathwise designs $\widetilde \nu _ { k } ^ { \mathrm { s e c } } , \bar { \nu } _ { k } ^ { \mathrm { t e r } }$ and their conditional means $\bar { \nu } _ { k } ^ { \mathrm { s e c } } , \bar { \nu } _ { k } ^ { \mathrm { t e r } }$ . The covariance comparison lemma below already avoids this issue by working at the episode-block level. The only bridge actually needed later in this appendix is for the squared MLE-mismatch class, which we state next.

Lemma D.4 (Pathwise-to-averaged bridge for the MLE mismatch class). For each candidate model $P \in { \mathcal { M } } ,$ , define

$$
h _ { P } ( s , a ) : = \| P ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 } ^ { 2 } .
$$

Then there is a universal constant $C < \infty$ such that, with probability at least $1 - \delta ,$ , simultaneously for all $k \in [ K ]$ and all $P \in { \mathcal { M } }$

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } \left[ h _ { P } \right] \leq 2 \mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } } \left[ h _ { P } \right] + C \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } ,
$$

and

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ h _ { P } ] \leq 2 \mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } } [ h _ { P } ] + C \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } .
$$

Consequently, on the intersection of this event with Lemma $D . 2 ,$ , we have simultaneously for all $k \in [ \dot { K } ]$

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } , \qquad \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } .
$$

In particular, this lemma is used later only to transfer the internal MLE-mismatch terms from the pathwise designs to the averaged designs. The generic test function g in the one-step-back and optimism lemmas is handled directly under the deterministic averaged measures $\bar { \nu } _ { k } ^ { \mathrm { s e c } } , \bar { \nu } _ { k } ^ { \mathrm { t e r } }$ through the coverage coefficients, so no stronger uniform bridge over an adaptive class is invoked.

Proof. Fix a candidate model $P \in { \mathcal { M } }$ and write $h : = h _ { P }$ . Since $\| P ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 } \leq 2$ we have

$$
0 \leq h ( s , a ) \leq 4 \qquad \forall ( s , a ) .
$$

We first treat the secondary stream. For each episode $i ,$ define

$$
X _ { i } ^ { \mathrm { s e c } } ( h ) : = h ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) - \mathbb { E } _ { \bar { \nu } _ { i } ^ { \mathrm { s e c } } } [ h ] = h ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) - \mathbb { E } _ { \nu _ { i } ^ { \prime } } [ h ] .
$$

Let $\mathcal { G } _ { i }$ denote the sigma-field generated by all tuples up to the end of episode i. Because $( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } )$ has conditional mean design $\nu _ { i } ^ { \prime }$ given $\mathcal { G } _ { i - 1 }$ , we have

$$
\mathbb { E } [ X _ { i } ^ { \mathrm { s e c } } ( h ) \mid { \mathcal G } _ { i - 1 } ] = 0 .
$$

Moreover,

$$
| X _ { i } ^ { \mathrm { s e c } } ( h ) | \leq 4 ,
$$

and, using $h ^ { 2 } \leq 4 h$ on $[ 0 , 4 ]$

$$
{  { \mathbb E } } \Big [ \big ( X _ { i } ^ { \mathrm { s e c } } ( h ) \big ) ^ { 2 } \mid \mathcal G _ { i - 1 } \Big ] \le {  { \mathbb E } } \big [ h ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) ^ { 2 } \mid \mathcal G _ { i - 1 } \big ] \le 4 {  { \mathbb E } } [ h ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) \mid \mathcal G _ { i - 1 } ] = 4 {  { \mathbb E } } _ { \nu _ { i } ^ { \prime } } [ h ] .
$$

Hence the martingale $\begin{array} { r } { S _ { k } ^ { \mathrm { s e c } } ( h ) : = \sum _ { i = 1 } ^ { k } X _ { i } ^ { \mathrm { s e c } } ( h ) } \end{array}$ has predictable quadratic variation bounded by

$$
V _ { k } ^ { \mathrm { s e c } } ( h ) \leq 4 \sum _ { i = 1 } ^ { k } \mathbb { E } _ { \nu _ { i } ^ { \prime } } [ h ] = 4 k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] .
$$

Apply Freedman’s inequality with confidence level $\delta _ { k , P } ^ { \mathrm { s e c } } : = \delta / ( 4 | M | k ^ { 2 } )$ , and write $\ell _ { k } ^ { \mathrm { s e c } } : =$ $\ln ( 1 / \delta _ { k , P } ^ { \mathrm { s e c } } )$ . Then, with probability at least $1 - \delta _ { k , P } ^ { \mathrm { s e c } }$

$$
- S _ { k } ^ { \mathrm { s e c } } ( h ) \leq \sqrt { 2 V _ { k } ^ { \mathrm { s e c } } ( h ) \ell _ { k } ^ { \mathrm { s e c } } } + \frac { 4 } { 3 } \ell _ { k } ^ { \mathrm { s e c } } .
$$

Dividing by k and recalling that

$$
\frac { - S _ { k } ^ { \mathrm { s e c } } ( h ) } { k } = \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] - \mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] ,
$$

we obtain

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] - \mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] \leq \sqrt { \frac { 8 \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] \ell _ { k } ^ { \mathrm { s e c } } } { k } } + \frac { 4 } { 3 } \frac { \ell _ { k } ^ { \mathrm { s e c } } } { k } .
$$

Using $2 { \sqrt { a b } } \leq { \frac { a } { 2 } } + 2 b$ with $a = \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ]$ and $b = 8 \ell _ { k } ^ { \mathrm { s e c } } / k$ gives

$$
\sqrt { \frac { 8 \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] \ell _ { k } ^ { \mathrm { s e c } } } { k } } \leq \frac { 1 } { 2 } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] + 1 6 \frac { \ell _ { k } ^ { \mathrm { s e c } } } { k } .
$$

Therefore

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] \leq 2 \mathbb { E } _ { \tilde { \nu } _ { k } ^ { \mathrm { s e c } } } [ h ] + \frac { 1 0 4 } { 3 } \frac { \ell _ { k } ^ { \mathrm { s e c } } } { k } .
$$

Since $\ell _ { k } ^ { \mathrm { s e c } } = \ln ( 4 | \mathcal { M } | k ^ { 2 } / \delta ) \lesssim \ln ( | \mathcal { M } | k / \delta )$ , this yields the desired secondary bound after adjusting the universal constant C.

The tertiary bound is identical. Indeed, defining

$$
X _ { i } ^ { \mathrm { t e r } } ( h ) : = h ( s _ { i } ^ { \prime \prime } , a _ { i } ^ { \prime \prime } ) - \mathbb { E } _ { \nu _ { i } ^ { \prime \prime } } [ h ] ,
$$

we again have

$$
\mathbb { E } [ X _ { i } ^ { \mathrm { t e r } } ( h ) \mid { \mathcal G } _ { i - 1 } ] = 0 , \qquad | X _ { i } ^ { \mathrm { t e r } } ( h ) | \le 4 ,
$$

and

$$
\mathbb { E } \Big [ \big ( X _ { i } ^ { \mathrm { t e r } } ( h ) \big ) ^ { 2 } \mid \mathcal { G } _ { i - 1 } \Big ] \leq 4 \mathbb { E } _ { \nu _ { i } ^ { \prime \prime } } [ h ] .
$$

Running the same Freedman argument and taking a union bound over the secondary and tertiary streams, over all $P \in { \mathcal { M } }$ , and over all $k \in [ K ]$ , gives the first two displays in the lemma.

Finally, set $P = \hat { P } _ { k }$ . Since $\hat { P } _ { k } \in { \mathcal { M } } .$ , we may plug $h _ { \hat { P } _ { k } } = f _ { k } ^ { 2 }$ into the simultaneous event. By Lemma D.2,

$$
\mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { a l l } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } .
$$

Because

$$
\widetilde { \nu } _ { k } ^ { \mathrm { a l l } } = \frac { 1 } { 3 } \big ( \bar { \nu } _ { k } ^ { \mathrm { p r i } } + \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } + \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } \big ) ,
$$

and all three component expectations are nonnegative,

$$
\mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { s e c } } } [ f _ { k } ^ { 2 } ] \le 3 \mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { a l l } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } ,
$$

and likewise

$$
\mathbb { E } _ { \widetilde { \nu } _ { k } ^ { \mathrm { t e r } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } .
$$

Substituting these into the secondary and tertiary bridge inequalities and absorbing the extra ln $( | \mathcal { M } | k / \delta \mathbf { \breve { ) } } / k$ term into $\zeta _ { k }$ proves the final display. □

Lemma D.5 (Primary-covariance deployed-to-probe transfer). Fix an episode k and define

$$
\begin{array} { r } { \Sigma _ { k , \star } ^ { \mathrm { p r i } } : = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ \phi ^ { \star } ( s , a ) \phi ^ { \star } ( s , a ) ^ { \top } ] + \lambda _ { k } I . } \end{array}
$$

Thenfor every policy π and every measurable $g : { \mathcal { S } } \times { \mathcal { A } } $ R with $\| g \| _ { \infty } \leq B ,$

$$
\begin{array} { r l } & { \left| \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } [ g ] \right| \lesssim \sqrt { \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } \| \phi ^ { \star } ( \tilde { s } , \tilde { a } ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } \sqrt { k \mathfrak { C } _ { k } ^ { \mathrm { n e x t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] + \lambda _ { k } d B ^ { 2 } } } \\ & { \qquad + \sqrt { ( 1 - \gamma ) \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ g ^ { 2 } ] } . } \end{array}
$$

Proof. Define

$$
m _ { g } ( s ) : = \mathbb { E } _ { a \sim \pi ( \cdot | s ) } [ g ( s , a ) ] , \quad T _ { \star } ^ { \pi } g ( \tilde { s } , \tilde { a } ) : = \mathbb { E } _ { s \sim P ^ { \star } ( \cdot | \tilde { s } , \tilde { a } ) } [ m _ { g } ( s ) ] .
$$

Then $| m _ { g } ( s ) | \leq B$ , and the discounted-occupancy recursion gives

$$
\mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } [ g ] = \gamma \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } [ T _ { \star } ^ { \pi } g ] + ( 1 - \gamma ) \mathbb { E } _ { s \sim d _ { 1 } } [ m _ { g } ( s ) ] .\tag{4}
$$

The initialization term is bounded by Cauchy-Schwarz and the definition of ${ \mathfrak { C } } _ { k } ^ { \mathrm { i n i t } } ( \pi )$

$$
( 1 - \gamma ) | \mathbb { E } _ { s \sim d _ { 1 } } [ m _ { g } ( s ) ] | \leq \sqrt { ( 1 - \gamma ) \mathfrak { E } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ g ^ { 2 } ] } .
$$

Next write $\begin{array} { r } { v _ { g } ^ { \star } : = \int \mu ^ { \star } ( s ) m _ { g } ( s ) } \end{array}$ ds. Since $T _ { \star } ^ { \pi } g ( \widetilde s , \widetilde a ) = \phi ^ { \star } ( \widetilde s , \widetilde a ) ^ { \top } v _ { g } ^ { \star }$ , Cauchy-Schwarz gives

$$
\begin{array} { r } { \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } [ T _ { \star } ^ { \pi } g ] \leq \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi } } \| \phi ^ { \star } ( \tilde { s } , \tilde { a } ) \| _ { ( { \Sigma _ { k , \star } ^ { \mathrm { p r i } } } ) ^ { - 1 } } \cdot \| v _ { g } ^ { \star } \| _ { { \Sigma _ { k , \star } ^ { \mathrm { p r i } } } } . } \end{array}
$$

Moreover, $\| v _ { g } ^ { \star } \| _ { 2 } \leq \sqrt { d } B$ , and

$$
\begin{array} { r } { \lVert \boldsymbol { v } _ { g } ^ { \star } \rVert _ { \Sigma _ { k , \star } ^ { \mathrm { p r i } } } ^ { 2 } = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \big [ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \big ] + \lambda _ { k } \lVert \boldsymbol { v } _ { g } ^ { \star } \rVert _ { 2 } ^ { 2 } . } \end{array}
$$

By Jensen and the definition of ${ \mathfrak { C } } _ { k } ^ { \mathrm { n e x t } } ( \pi )$ ，

$$
\begin{array} { r } { \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \left[ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \right] \leq \mathfrak { C } _ { k } ^ { \mathrm { n e x t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] . } \end{array}
$$

Substituting these bounds into (4) gives the claim.

Lemma D.6 (Merged covariance comparison). For each episode $k ,$ define

$$
\Sigma _ { k } ^ { \mathrm { m i x } } : = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] + k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] + \lambda _ { k } I .
$$

There exist universal constants $0 < c _ { 1 } \leq c _ { 2 } <$ ∞ such that, with probability at least $1 - \delta ,$ simultaneouslyfor all $k \in [ K ]$ and all $( s , a )$

$$
c _ { 1 } \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } \leq \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \hat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } \leq c _ { 2 } \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } .
$$

Proof. Fix an episode k and first work with an arbitrary candidate feature map $\phi \in \Gamma$ . Define the episode-block matrix

$$
H _ { i } ( \phi ) : = \phi ( s _ { i } , a _ { i } ) \phi ( s _ { i } , a _ { i } ) ^ { \top } + \phi ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) \phi ( s _ { i } ^ { \prime } , a _ { i } ^ { \prime } ) ^ { \top } , \qquad i \in [ k ] .
$$

Let $\mathcal { G } _ { i }$ be the sigma-field generated by all tuples collected up to the end of episode i, with $\mathcal { G } _ { 0 }$ the history before episode 1. Define the predictable episode-level conditional mean $\mathbf { \bar { H } } _ { i } ( \phi ) : = \mathbb { E } [ H _ { i } ( \phi ) \mid \mathcal { G } _ { i - 1 } ]$ By construction, conditioned on $\mathcal { G } _ { i - 1 }$ the primary tuple is sampled from $\nu _ { i } .$ , while the secondary tuple has conditional mean design $\nu _ { i } ^ { \prime }$ after marginalizing over the primary transition and the secondary probe action. Therefore

$$
\bar { H } _ { i } ( \phi ) = \mathbb { E } _ { \nu _ { i } } [ \phi \phi ^ { \top } ] + \mathbb { E } _ { \nu _ { i } ^ { \prime } } [ \phi \phi ^ { \top } ] , \qquad \sum _ { i = 1 } ^ { k } \bar { H } _ { i } ( \phi ) = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ \phi \phi ^ { \top } ] + k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s c c } } } [ \phi \phi ^ { \top } ] .
$$

Now define

$$
\Sigma _ { k } ^ { \mathrm { m i x } } ( \phi ) : = \sum _ { i = 1 } ^ { k } \bar { H } _ { i } ( \phi ) + \lambda _ { k } I , \qquad \hat { \Sigma } _ { k } ^ { \mathrm { m i x } } ( \phi ) : = \sum _ { i = 1 } ^ { k } H _ { i } ( \phi ) + \lambda _ { k } I .
$$

Under the bounded-feature assumption used throughout the paper, $H _ { i } ( \phi )$ is positive semidefinite with operator norm at most 2. Hence the matrix martingale

$$
M _ { t } ( \phi ) : = \sum _ { u = 1 } ^ { t } \bigl ( H _ { u } ( \phi ) - \bar { H } _ { u } ( \phi ) \bigr )
$$

satisfies the standard bounded-increment conditions for episode-block matrix concentration with respect to $\{ \mathcal { G } _ { i } \} _ { i = 0 } ^ { k }$ . In particular, the same self-normalized matrix martingale argument used in the

proof of Lemma A.1 applies to these block increments. After choosing $\lambda _ { k } \gtrsim d \ln ( | \mathcal { M } | k / \delta )$ , there is an event of probability at least $1 - \delta$ on which

$$
\left\| \left( \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) \right) ^ { - 1 / 2 } M _ { k } ( \phi ) \left( \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) \right) ^ { - 1 / 2 } \right\| _ { \mathrm { o p } } \leq \frac { 1 } { 2 } ,
$$

simultaneously for all $k \in [ K ]$ and all $\phi \in \Gamma$ . Equivalently,

$$
- \frac { 1 } { 2 } \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) \preceq M _ { k } ( \phi ) \preceq \frac { 1 } { 2 } \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) .
$$

Since $\hat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ( \phi ) = \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) + M _ { k } ( \phi )$ , this immediately yields

$$
\frac { 1 } { 2 } \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) \preceq \hat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ( \phi ) \preceq \frac { 3 } { 2 } \Sigma _ { k } ^ { \operatorname* { m i x } } ( \phi ) .
$$

Specializing to $\phi = \hat { \phi } _ { k }$ and inverting the PSD inequalities gives

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { 2 } \Sigma _ { k } ^ { \mathrm { m i x } } \preceq \hat { \Sigma } _ { k } ^ { \mathrm { m i x } } \preceq \frac { 3 } { 2 } \Sigma _ { k } ^ { \mathrm { m i x } } , } \\ & { \displaystyle \frac { 2 } { 3 } ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } \preceq ( \hat { \Sigma } _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } \preceq 2 ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } . } \end{array}
$$

Applying the resulting quadratic-form comparison to $\hat { \phi } _ { k } ( s , a )$ proves

$$
\sqrt { \frac { 2 } { 3 } } \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } \leq \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \hat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } \leq \sqrt { 2 } \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } .
$$

Thus the claim holds with any universal constants $c _ { 1 } \leq \sqrt { 2 / 3 }$ and $c _ { 2 } \geq { \sqrt { 2 } }$

Proposition D.7 (Concrete tertiary coverage bound). Under Algorithm 1, for every episode k and every policy π,

$$
\begin{array} { r } { \mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le \Gamma _ { k } , \qquad \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le \Gamma _ { k } , \qquad \mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le \Gamma _ { k } . } \end{array}
$$

Proof. Because the schedule is nonincreasing,

$$
\epsilon _ { i } \geq \epsilon _ { k } \qquad \forall i \leq k .
$$

Hence every probing kernel among the first k episodes satisfies

$$
q _ { i } ( a \mid s ) \geq { \frac { \epsilon _ { k } } { | { \cal A } | } } , \qquad q _ { i } ^ { \prime } ( a \mid s ) \geq { \frac { \epsilon _ { k } } { | { \cal A } | } } , \qquad q _ { i } ^ { \prime \prime } ( a \mid s ) \geq { \frac { \epsilon _ { k } } { | { \cal A } | } } .
$$

For the first tertiary coefficient, let

$$
\mu _ { k , 1 } ^ { \pi } ( s , a ) : = \mathbb { E } _ { \stackrel { ( \tilde { s } , \tilde { a } ) \sim \bar { \nu } _ { k } ^ { \mathrm { p r i } } } { s \sim P ^ { \star } \left( \cdot | \tilde { s } , \tilde { a } \right) } } \left[ \pi ( a \mid s ) \mathbf { 1 } \{ s \in d s \} \right]
$$

denote the pushforward state-action measure appearing in the definition of ${ \mathfrak { C } } _ { k , 1 } ^ { \mathrm { t r i , n e x t } } ( \pi )$ . Since

$$
\bar { \nu } _ { k } ^ { \mathrm { s e c } } ( s , a ) = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \rho _ { i } ^ { \prime } ( s ) q _ { i } ^ { \prime } ( a \mid s ) \ge \frac { \epsilon _ { k } } { \mid A \mid } \cdot \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \rho _ { i } ^ { \prime } ( s ) ,
$$

we have, pointwise,

$$
\mu _ { k , 1 } ^ { \pi } ( s , a ) \leq \frac { | \mathcal { A } | } { \epsilon _ { k } } \bar { \nu } _ { k } ^ { \mathrm { s e c } } ( s , a ) = \Gamma _ { k } \bar { \nu } _ { k } ^ { \mathrm { s e c } } ( s , a ) .
$$

Therefore for every nonnegative measurable $g ^ { 2 }$

$$
\begin{array} { r } { \mathbb { E } _ { \mu _ { k , 1 } ^ { \pi } } [ g ^ { 2 } ] \leq \Gamma _ { k } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] , } \end{array}
$$

which proves

$$
\mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i , n e x t } } ( \pi ) \leq \Gamma _ { k } .
$$

The proof for ${ \mathfrak { C } } _ { k , 2 } ^ { \mathrm { t r i , n e x t } } ( \pi )$ is identical. If

$$
\mu _ { k , 2 } ^ { \pi } ( s , a ) : = \mathbb { E } _ { \stackrel { ( \tilde { s } , \tilde { a } ) \sim \bar { \nu } _ { k } ^ { \mathrm { s e c } } } { s \sim P ^ { \star } ( \cdot | \tilde { s } , \tilde { a } ) } } \left[ \pi ( a \mid s ) \mathbf { 1 } \{ s \in d s \} \right] ,
$$

then

$$
\bar { \nu } _ { k } ^ { \mathrm { t e r } } ( s , a ) \geq \frac { \epsilon _ { k } } { | { \cal A } | } \cdot \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \rho _ { i } ^ { \prime \prime } ( s ) ,
$$

while by construction

$$
\mu _ { k , 2 } ^ { \pi } ( s , a ) = { \frac { 1 } { k } } \sum _ { i = 1 } ^ { k } \rho _ { i } ^ { \prime \prime } ( s ) \pi ( a \mid s ) .
$$

Using $\pi ( a \mid s ) \leq 1$ and the lower bound on $\bar { \nu } _ { k } ^ { \mathrm { t e r } }$ , we obtain the pointwise domination

$$
\begin{array} { r } { \mu _ { k , 2 } ^ { \pi } ( s , a ) \leq \Gamma _ { k } \bar { \nu } _ { k } ^ { \mathrm { t e r } } ( s , a ) . } \end{array}
$$

Therefore, for every nonnegative measurable $g ^ { 2 }$

$$
\begin{array} { r } { \mathbb { E } _ { \mu _ { k , 2 } ^ { \pi } } [ g ^ { 2 } ] \leq \Gamma _ { k } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ g ^ { 2 } ] , } \end{array}
$$

which proves

$$
\mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i , n e x t } } ( \pi ) \le \Gamma _ { k } .
$$

Taking the maximum of the two coefficients completes the proof.

Lemma D.8 (Tertiary-aware mixed one-step-back inequality). Fix an episode k and define $f _ { k } ( s , a ) : =$ $\| \hat { P } _ { k } ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 }$ and

$$
\Sigma _ { k } ^ { \mathrm { m i x } } : = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] + k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] + \lambda _ { k } I .
$$

Also define

$$
\begin{array} { r } { M _ { k } ^ { \mathrm { m i x } } ( \pi ) : = \mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } \left[ \| \hat { \phi } _ { k } ( \tilde { s } , \tilde { a } ) \| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } \right] , \qquad G _ { k , 2 3 } ( g ) : = \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] + \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ g ^ { 2 } ] . } \end{array}
$$

Thenfor every policy π and every measurable $g : { \mathcal { S } } \times { \mathcal { A } } $ R with $\| g \| _ { \infty } \leq B ,$

$$
\begin{array} { r l } & { \left| \mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ g ] \right| \lesssim \sqrt { \gamma } M _ { k } ^ { \mathrm { m i x } } ( \pi ) \sqrt { k \mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) G _ { k , 2 3 } ( g ) + \lambda _ { k } d B ^ { 2 } + k B ^ { 2 } \zeta _ { k } } } \\ & { \qquad + \sqrt { ( 1 - \gamma ) \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ g ^ { 2 } ] } , } \end{array}
$$

on the events ofLemmas D.2 and $D . 4 .$

Proof. Define

$$
\begin{array} { r l } & { m _ { g } ( s ) : = \mathbb { E } _ { a \sim \pi ( \cdot \vert s ) } [ g ( s , a ) ] , } \\ & { T _ { \hat { P } _ { k } } ^ { \pi } g ( \tilde { s } , \tilde { a } ) : = \mathbb { E } _ { s \sim \hat { P } _ { k } ( \cdot \vert \tilde { s } , \tilde { a } ) } [ m _ { g } ( s ) ] , } \\ & { T _ { \star } ^ { \pi } g ( \tilde { s } , \tilde { a } ) : = \mathbb { E } _ { s \sim P ^ { \star } ( \cdot \vert \tilde { s } , \tilde { a } ) } [ m _ { g } ( s ) ] . } \end{array}
$$

Since $\| g \| _ { \infty } \leq B _ { \mathrm { : } }$ , we also have $\| m _ { g } \| _ { \infty } \leq B$ . The discounted occupancy recursion under $\hat { P } _ { k }$ gives

$$
\mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ g ] = \gamma \mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ T _ { \hat { P } _ { k } } ^ { \pi } g ] + ( 1 - \gamma ) \mathbb { E } _ { s \sim d _ { 1 } } [ m _ { g } ( s ) ] .
$$

We treat the two terms on the right separately.

For the initialization term, write

$$
\mathbb { E } _ { s \sim d _ { 1 } } [ m _ { g } ( s ) ] = \sum _ { s , a } d _ { 1 } ( s ) { \pi } ( a \mid s ) g ( s , a ) .
$$

By the definition of ${ \mathfrak { C } } _ { k } ^ { \mathrm { i n i t } } ( \pi )$

$$
( 1 - \gamma ) d _ { 1 } ( s ) \pi ( a \mid s ) \leq { \mathfrak { C } } _ { k } ^ { \mathrm { i n i t } } ( \pi ) { \bar { \nu } } _ { k } ^ { \mathrm { p r i } } ( s , a ) \qquad \forall ( s , a )
$$

on the support of $\bar { \nu } _ { k } ^ { \mathrm { p r i } }$ . Therefore Cauchy-Schwarz gives

$$
( 1 - \gamma ) | \mathbb { E } _ { s \sim d _ { 1 } } [ m _ { g } ( s ) ] | \leq \sqrt { ( 1 - \gamma ) \mathfrak { E } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ g ^ { 2 } ] } .
$$

For the propagated term, define $\begin{array} { r } { v _ { g } ^ { \hat { P } } : = \int \hat { \mu } _ { k } ( s ) m _ { g } ( s ) d s } \end{array}$ . Because $T _ { \hat { P } _ { k } } ^ { \pi } g ( \tilde { s } , \tilde { a } ) = \hat { \phi } _ { k } ( \tilde { s } , \tilde { a } ) ^ { \top } v _ { g } ^ { \hat { P } } ,$ Cauchy-Schwarz under $\big ( \Sigma _ { k } ^ { \mathrm { m i x } } \big ) ^ { - 1 }$ yields

$$
\mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ T _ { \hat { P } _ { k } } ^ { \pi } g ] \leq M _ { k } ^ { \operatorname* { m i x } } ( \pi ) \| v _ { g } ^ { \hat { P } } \| _ { \Sigma _ { k } ^ { \operatorname* { m i x } } } .
$$

Moreover, $\| v _ { g } ^ { \hat { P } } \| _ { 2 } \leq \sqrt { d } B _ { : }$ , so

$$
\| v _ { g } ^ { \hat { P } } \| _ { \Sigma _ { k } ^ { \operatorname* { m i x } } } ^ { 2 } = k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \operatorname { p r i } } } \big [ ( T _ { \hat { P } _ { k } } ^ { \pi } g ) ^ { 2 } \big ] + k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \operatorname { s e c } } } \big [ ( T _ { \hat { P } _ { k } } ^ { \pi } g ) ^ { 2 } \big ] + \lambda _ { k } \| v _ { g } ^ { \hat { P } } \| _ { 2 } ^ { 2 } .
$$

We now control the two quadratic terms. By definition of $f _ { k }$

$$
\big | T _ { \hat { P } _ { k } } ^ { \pi } g \big ( \tilde { s } , \tilde { a } \big ) - T _ { \star } ^ { \pi } g \big ( \tilde { s } , \tilde { a } \big ) \big | \le B f _ { k } \big ( \tilde { s } , \tilde { a } \big ) .
$$

Hence, using $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 } .$

$$
\begin{array} { r } { \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \left[ ( T _ { \hat { P } _ { k } } ^ { \pi } g ) ^ { 2 } \right] \lesssim \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \left[ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \right] + B ^ { 2 } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ f _ { k } ^ { 2 } ] , } \\ { \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } \left[ ( T _ { \hat { P } _ { k } } ^ { \pi } g ) ^ { 2 } \right] \lesssim \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } \left[ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \right] + B ^ { 2 } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ f _ { k } ^ { 2 } ] . } \end{array}
$$

On the events of Lemmas D.2 and D.4,

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } , \qquad \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } ,
$$

because the primary term is one component of the pathwise average $\widetilde \nu _ { k } ^ { \mathrm { a l l } }$ , while the secondary term is provided directly by Lemma D.4. This is the only place where a pathwise-to-averaged transfer is used: the generic g-dependence remains entirely inside $G _ { k , 2 3 } ( g ) \overset { * } { = } \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] + \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ g ^ { 2 } ]$ and is handled through the deterministic coverage coefficients.

Next, by Jensen’s inequality,

$$
\begin{array} { r l } & { \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \left[ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \right] \leq \mathbb { E } _ { \mu _ { k , 1 } ^ { \pi } } [ g ^ { 2 } ] \leq \mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ g ^ { 2 } ] , } \\ & { \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } \left[ ( T _ { \star } ^ { \pi } g ) ^ { 2 } \right] \leq \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { t e r } } } [ g ^ { 2 } ] . } \end{array}
$$

Combining the last four displays and using

$$
\mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) = \mathrm { m a x } \Big \{ \mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) , \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \Big \} ,
$$

we obtain

$$
\| v _ { g } ^ { \hat { P } } \| _ { \Sigma _ { k } ^ { \operatorname * { m i x } } } ^ { 2 } \lesssim k \mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) G _ { k , 2 3 } ( g ) + \lambda _ { k } d B ^ { 2 } + k B ^ { 2 } \zeta _ { k } .
$$

Substituting this and the initialization bound into the discounted-occupancy recursion proves the claimed inequality. □

Lemma D.9 (Direct optimism for the repaired merged bonus). Fix an episode k and a policy π. Define

$$
\begin{array} { r l } & { b _ { k , \mathrm { m i x } , \mathrm { p o p } } ( s , a ) : = \alpha _ { k } ^ { \mathrm { m i x } } \Vert \hat { \phi } _ { k } ( s , a ) \Vert _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } , } \\ & { b _ { k , \mathrm { m i x } , \mathrm { e m p } } ( s , a ) : = \alpha _ { k } ^ { \mathrm { e m p } } \Vert \hat { \phi } _ { k } ( s , a ) \Vert _ { ( \hat { \Sigma } _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } . } \end{array}
$$

Assume the events ofLemmas D.2, D.4, and the covariance comparison event ofLemma D.6 all hold. If

$$
\alpha _ { k } ^ { \mathrm { m i x } } \gtrsim \sqrt { \gamma \Big ( k \mathfrak { C } _ { k } ^ { \mathrm { t r i , n e x t } } ( \pi ) \zeta _ { k } + \lambda _ { k } d + k \zeta _ { k } \Big ) } ,
$$

and

$$
\alpha _ { k } ^ { \mathrm { e m p } } \geq \alpha _ { k } ^ { \mathrm { m i x } } / c _ { 1 } ,
$$

then

$$
V _ { \hat { P } _ { k } , r + b _ { k , \mathrm { m i x , e m p } } } ^ { \pi } - V _ { P ^ { \star } , r } ^ { \pi } \gtrsim - \sqrt { \frac { \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \zeta _ { k } } { 1 - \gamma } } .
$$

Under the nonbinding-cap condition $\hat { b } _ { k } = b _ { k , \mathrm { m i x , e m p } } ,$ , the same bound holds with $b _ { k , \mathrm { m i x , e m p } }$ replaced by the actual algorithmic bonus $\hat { b } _ { k }$

Proof. Apply the first form of the simulation lemma (Lemma B.4) to the pair $( \hat { P } _ { k } , r + b _ { k , \mathrm { m i x } , \mathrm { p o p } } )$ and $\bar { ( P ^ { \star } , r ) }$ . This gives

$$
\begin{array} { r l } { ( 1 - \gamma ) \big ( V _ { \widehat { P } _ { k } , r + b _ { k , \mathrm { m i x } , \mathrm { p o p } } } ^ { \pi } - V _ { P ^ { \star } , r } ^ { \pi } \big ) } & { = \mathbb { E } _ { d _ { P _ { k } } ^ { \pi } } \Big [ b _ { k , \mathrm { m i x } , \mathrm { p o p } } ( s , a ) \gamma \mathbb { E } _ { \widehat { P } _ { k } } \big [ V _ { P ^ { \star } , r } ^ { \pi } ( s ^ { \prime } ) \big ] - \gamma \mathbb { E } _ { P ^ { \star } } \big [ V _ { P ^ { \star } , r } ^ { \pi } ( s ^ { \prime } ) \big ] \Big ] . } \end{array}
$$

Because the total discounted reward is normalized to lie in [0, 1], we have

$$
\| V _ { \hat { P } ^ { \star } , r } ^ { \pi } \| _ { \infty } \leq 1 , \qquad \left| \mathbb { E } _ { \hat { P } _ { k } } [ V _ { { P } ^ { \star } , r } ^ { \pi } ( s ^ { \prime } ) ] - \mathbb { E } _ { P ^ { \star } } [ V _ { P ^ { \star } , r } ^ { \pi } ( s ^ { \prime } ) ] \right| \leq f _ { k } ( s , a ) .
$$

Therefore

$$
\begin{array} { r l } { ( 1 - \gamma ) \big ( { \cal V } _ { \hat { P } _ { k } , r + b _ { k , \mathrm { m i x } , \mathrm { p o p } } } ^ { \pi } - { \cal V } _ { P ^ { \star } , r } ^ { \pi } \big ) } & { \geq \mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } \big [ b _ { k , \mathrm { m i x } , \mathrm { p o p } } ( s , a ) \big ] - \gamma \mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ f _ { k } ] . } \end{array}
$$

For brevity, define

$$
\Omega _ { k } ( \pi ) : = k { \mathfrak { C } } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \zeta _ { k } + \lambda _ { k } d + k \zeta _ { k } .
$$

Now apply Lemma D.8 with $g = f _ { k }$ and $B = 2$ . On the events assumed in the lemma,

$$
\mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ f _ { k } ] \lesssim \sqrt { \gamma } M _ { k } ^ { \operatorname* { m i x } } ( \pi ) \sqrt { \Omega _ { k } ( \pi ) } + \sqrt { ( 1 - \gamma ) \mathfrak { C } _ { k } ^ { \operatorname* { i n i t } } ( \pi ) \zeta _ { k } } .
$$

By the choice of $\alpha _ { k } ^ { \mathrm { m i x } } , \ \alpha _ { k } ^ { \mathrm { m i x } } \ \gtrsim \ \sqrt { \gamma \Omega _ { k } ( \pi ) }$ , so the first term on the right is absorbed by $\mathbb { E } _ { d _ { \hat { P } _ { k } } ^ { \pi } } [ b _ { k , \mathrm { m i x } , \mathrm { p o p } } ] = \alpha _ { k } ^ { \mathrm { m i x } } M _ { k } ^ { \mathrm { m i x } } ( \pi )$ . Consequently,

$$
V _ { \hat { P } _ { k } , r + b _ { k , \mathrm { m i x , p o p } } } ^ { \pi } - V _ { P ^ { \star } , r } ^ { \pi } \gtrsim - \sqrt { \frac { \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \zeta _ { k } } { 1 - \gamma } } .
$$

Next, Lemma D.6 gives

$$
\| \hat { \phi } _ { k } ( s , a ) \| _ { ( \hat { \Sigma } _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } \geq c _ { 1 } \| \hat { \phi } _ { k } ( s , a ) \| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } ,
$$

and therefore $\alpha _ { k } ^ { \mathrm { e m p } } \geq \alpha _ { k } ^ { \mathrm { m i x } } / c _ { 1 }$ implies the pointwise domination $b _ { k , \mathrm { m i x , e m p } } ( s , a ) \geq b _ { k , \mathrm { m i x , p o p } } ( s , a )$ Monotonicity of the simulation lemma in the bonus term then yields

$$
V _ { \hat { P } _ { k } , r + b _ { k , \mathrm { m i x , e m p } } } ^ { \pi } - V _ { P ^ { \star } , r } ^ { \pi } \gtrsim - \sqrt { \frac { \mathfrak { C _ { k } ^ { \mathrm { i n i t } } } ( \pi ) \zeta _ { k } } { 1 - \gamma } } .
$$

Finally, under the nonbinding-cap condition, $\hat { b } _ { k } = b _ { k , \mathrm { m i x , e m p } } .$ , so the same bound holds for the actual algorithmic bonus. □

## D.4 Final Aggregation

We now combine the optimism step with the deployed-to-probe transfer and the elliptical-potential argument.

Lemma D.10 (Tertiary-aware primal-dual aggregation). Fix an episode k and define

$$
\begin{array} { r l } & { \Delta _ { k } : = \big ( V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) \big ) + Y _ { k } \big ( V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) \big ) , } \\ & { A _ { k } : = \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } \Big [ \| \phi ^ { \star } ( \tilde { s } , \tilde { a } ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } \Big ] , } \\ & { f _ { k } ( s , a ) : = \| \hat { P } _ { k } ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 } . } \end{array}
$$

Assume,for every episode k,

1. the cap is nonbinding:

$$
\bar { b } \ge \operatorname* { m a x } _ { 1 \le j \le K } \frac { \alpha _ { j } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { j } } } ;
$$

2. the events of Lemmas D.2 and D.4 hold;

3. Lemma D.9 holds with $\pi = \pi ^ { \star }$ for both the reward objective r and the utility objective $g .$

Then, under Algorithm $^ { l , }$

$$
\begin{array} { r l } & { \displaystyle \Delta _ { k } \lesssim \frac { 1 + \xi } { 1 - \gamma } A _ { k } \sqrt { \gamma d \Gamma _ { k } ( \alpha _ { k } ^ { \mathrm { e m p } } ) ^ { 2 } + \gamma \lambda _ { k } d \bar { b } ^ { 2 } } + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } A _ { k } \sqrt { \gamma \big ( k \Gamma _ { k } \zeta _ { k } + \lambda _ { k } d \big ) } } \\ & { \quad \quad \quad + \frac { 1 + \xi } { 1 - \gamma } \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \frac { \Gamma _ { k } d ( 1 - \gamma ) } { k } } + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \Gamma _ { k } \zeta _ { k } ( 1 - \gamma ) } + ( 1 + \xi ) \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l r } {  { \sum _ { k = 1 } ^ { K } \Delta _ { k } \lesssim \frac { 1 + \xi } { 1 - \gamma } \sum _ { k = 1 } ^ { K } A _ { k } \sqrt { \gamma d \Gamma _ { k } ( \alpha _ { k } ^ { \mathrm { e m p } } ) ^ { 2 } + \gamma \lambda _ { k } d \bar { b } ^ { 2 } } + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { k = 1 } ^ { K } A _ { k } \sqrt { \gamma \big ( k \Gamma _ { k } \zeta _ { k } + \lambda _ { k } d \big ) } } } \\ & { } & { \quad + \frac { 1 + \xi } { 1 - \gamma } \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \frac { \Gamma _ { k } d ( 1 - \gamma ) } { k } } } \\ & { } & { \quad + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { k = 1 } ^ { K } \sqrt { \Gamma _ { k } \zeta _ { k } ( 1 - \gamma ) } + ( 1 + \xi ) \sum _ { k = 1 } ^ { K } \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } . } \end{array}
$$

Proof. Fix an episode k. By the definition of $\Delta _ { k }$

$$
\Delta _ { k } = \left( V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } - V _ { P ^ { \star } , r } ^ { \pi _ { k } } \right) + Y _ { k } \left( V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } - V _ { P ^ { \star } , g } ^ { \pi _ { k } } \right) .
$$

Because $0 \leq Y _ { k } \leq \xi$ and Lemma D.9 holds at $\pi ^ { \star }$ for both the reward and utility objectives,

$$
V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } \leq V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi ^ { \star } } + \sqrt { \frac { \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ^ { \star } ) \zeta _ { k } } { 1 - \gamma } } , \qquad V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } \leq V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi ^ { \star } } + \sqrt { \frac { \mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ^ { \star } ) \zeta _ { k } } { 1 - \gamma } } .
$$

Since every primary probing distribution has the uniform floor $\epsilon _ { k } / | \mathcal { A } |$ and $\rho _ { i } ( s ) = d _ { P ^ { \star } } ^ { \pi _ { i - 1 } } ( s ) \geq$ $( 1 - \gamma ) d _ { 1 } ( s )$ , we have $\begin{array} { r } { \bar { \nu } _ { k } ^ { \mathrm { p r i } } ( s , a ) \geq ( 1 - \gamma ) d _ { 1 } ( s ) \epsilon _ { k } / | \mathcal { A } | } \end{array}$ , hence ${ \mathfrak { C } } _ { k } ^ { \mathrm { { i n i t } } } ( \pi ^ { \star } ) \leq \Gamma _ { k }$ . Therefore

$$
\Delta _ { k } \leq \widetilde { \Delta } _ { k } + ( 1 + \xi ) \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } ,
$$

where

$$
\widetilde { \Delta } _ { k } : = \big ( { { V } _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi ^ { \star } } } - { { V } _ { P ^ { \star } , r } ^ { \pi _ { k } } } \big ) + { { Y } _ { k } } \big ( { { V } _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi ^ { \star } } } - { { V } _ { P ^ { \star } , g } ^ { \pi _ { k } } } \big ) .
$$

By the definition of $\pi _ { k }$ as the maximizer of the optimistic empirical Lagrangian,

$$
\begin{array} { r l } & { V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi ^ { \star } } + Y _ { k } V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi ^ { \star } } \le V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi _ { k } } + Y _ { k } V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } . } \end{array}
$$

Hence

$$
\widetilde { \Delta } _ { k } \leq \big ( V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi _ { k } } - V _ { P ^ { \star } , r } ^ { \pi _ { k } } \big ) + Y _ { k } \big ( V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } - V _ { P ^ { \star } , g } ^ { \pi _ { k } } \big ) .
$$

Define

$$
\begin{array} { r l r } & { } & { W _ { k , r } : = V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi _ { k } } , \qquad } & { W _ { k , g } : = V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } , } \\ & { } & { \Delta _ { \hat { P } _ { k } , P ^ { \star } } W ( s , a ) : = \mathbb { E } _ { \hat { P } _ { k } } [ W ( s ^ { \prime } ) ] - \mathbb { E } _ { P ^ { \star } } [ W ( s ^ { \prime } ) ] . } \end{array}
$$

Apply the second form of the simulation lemma (Lemma B.4) to each bracket. For the reward term,

$$
V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi _ { k } } - V _ { P ^ { \star } , r } ^ { \pi _ { k } } = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } \Big [ \hat { b } _ { k } ( s , a ) + \gamma \Delta _ { \hat { P } _ { k } , P ^ { \star } } W _ { k , r } ( s , a ) \Big ] .
$$

The analogous identity holds for the utility term. Since $\begin{array} { r } { \| \hat { b } _ { k } \| _ { \infty } \leq \bar { b } , } \end{array}$ the normalized reward and utility assumptions imply, up to universal constants,

$$
\| V _ { \hat { P } _ { k } , r + \hat { b } _ { k } } ^ { \pi _ { k } } \| _ { \infty } \lesssim \frac { \bar { b } } { 1 - \gamma } , \qquad \| V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } \| _ { \infty } \lesssim \frac { \bar { b } } { 1 - \gamma } .
$$

Thus

$$
\widetilde { \Delta } _ { k } \lesssim \frac { 1 + \xi } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ \widehat { b } _ { k } ] + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ f _ { k } ] .
$$

Combining this with the previous reduction gives

$$
\Delta _ { k } \lesssim \frac { 1 + \xi } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { k } } ^ { \pi _ { k } } } [ \hat { b } _ { k } ] + \frac { ( 1 + \xi ) \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \mathbb { E } _ { d _ { P ^ { k } } ^ { \pi _ { k } } } [ f _ { k } ] + ( 1 + \xi ) \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } .
$$

We now bound the two expectations separately.

For the bonus term, Lemma D.6 implies

$$
\begin{array} { r } { \hat { b } _ { k } ( s , a ) \lesssim \alpha _ { k } ^ { \mathrm { e m p } } \Vert \hat { \phi } _ { k } ( s , a ) \Vert _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } . } \end{array}
$$

Moreover,

$$
k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } \left[ \| \hat { \phi } _ { k } \| _ { ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } } ^ { 2 } \right] \leq d ,
$$

and the same bound holds with $\bar { \nu } _ { k } ^ { \mathrm { p r i } }$ replaced by $\bar { \nu } _ { k } ^ { \mathrm { s e c } }$ , because each term is upper-bounded by

$$
\begin{array} { r } { \mathrm { T r } \Big ( \big ( k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] + k \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ \hat { \phi } _ { k } \hat { \phi } _ { k } ^ { \top } ] \big ) ( \Sigma _ { k } ^ { \mathrm { m i x } } ) ^ { - 1 } \Big ) \leq d . } \end{array}
$$

Applying Lemma D.5 with $\boldsymbol { g } = \boldsymbol { \hat { b } _ { k } }$ and $B = { \bar { b } } ,$ and using ${ \mathfrak { C } } _ { k } ^ { \mathrm { { n e x t } } } ( \pi _ { k } ) \leq \Gamma _ { k }$ and ${ \mathfrak { C } } _ { k } ^ { \mathrm { { i n i t } } } ( \pi _ { k } ) \leq \Gamma _ { k }$ , gives

$$
\mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ \widehat { b } _ { k } ] \lesssim A _ { k } \sqrt { \gamma d \Gamma _ { k } ( \alpha _ { k } ^ { \mathrm { e m p } } ) ^ { 2 } + \gamma \lambda _ { k } d \bar { b } ^ { 2 } } + \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \frac { \Gamma _ { k } d ( 1 - \gamma ) } { k } } .
$$

For the model-error term, apply Lemma D.5 with $g = f _ { k }$ and $B = 2$ . By Lemmas D.2 and D.4,

$$
\mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { p r i } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } , \qquad \mathbb { E } _ { \bar { \nu } _ { k } ^ { \mathrm { s e c } } } [ f _ { k } ^ { 2 } ] \lesssim \zeta _ { k } ,
$$

and again ${ \mathfrak { C } } _ { k } ^ { \mathrm { n e x t } } ( \pi _ { k } ) \leq \Gamma _ { k }$ and ${ \mathfrak { C } } _ { k } ^ { \mathrm { { i n i t } } } ( \pi _ { k } ) \leq \Gamma _ { k }$ . Hence

$$
\mathbb { E } _ { d _ { P ^ { k } } ^ { \pi _ { k } } } [ f _ { k } ] \lesssim A _ { k } \sqrt { \gamma \big ( k \Gamma _ { k } \zeta _ { k } + \lambda _ { k } d \big ) } + \sqrt { \Gamma _ { k } \zeta _ { k } ( 1 - \gamma ) } .
$$

Substituting the last two displays into the bound on $\Delta _ { k }$ yields the episode-wise inequality claimed in the lemma. Summing over $k = 1 , \ldots , K$ proves the cumulative bound. □

Lemma D.11 (Tertiary-aware deployed-to-probe potential reduction). Under Algorithm 1,

$$
\Psi _ { K } : = \sum _ { k = 1 } ^ { K } A _ { k } \lesssim \frac { 1 } { \sqrt { \lambda _ { 1 } } } + \sqrt { d \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) \sum _ { k = 2 } ^ { K } \Lambda _ { k } } .
$$

In particular, since $\Lambda _ { k } \leq \Gamma _ { k }$

$$
\Psi _ { K } \lesssim \frac { 1 } { \sqrt { \lambda _ { 1 } } } + \sqrt { d \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) \sum _ { k = 2 } ^ { K } \Gamma _ { k } } .
$$

Proof. The first episode is trivial:

$$
A _ { 1 } \leq \frac { 1 } { \sqrt { \lambda _ { 1 } } } ,
$$

because $( \Sigma _ { 1 , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } \preceq \lambda _ { 1 } ^ { - 1 } I$ and $\| \phi ^ { \star } ( s , a ) \| _ { 2 } \leq 1$

Now fix $k \le K - 1$ . The relation

$$
q _ { k + 1 } ( a \mid s ) = ( 1 - \epsilon _ { k + 1 } ) \pi _ { k } ( a \mid s ) + \epsilon _ { k + 1 } U ( a )
$$

implies the pointwise ratio bound

$$
{ \frac { \pi _ { k } ( a \mid s ) } { q _ { k + 1 } ( a \mid s ) } } \leq \Lambda _ { k + 1 } .
$$

Therefore

Indeed,

$$
\begin{array} { r } { A _ { k } \leq \sqrt { \Lambda _ { k + 1 } } \sqrt { \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \right] } . } \end{array}
$$

$$
\begin{array} { r l } & { A _ { k } = \mathbb { E } _ { ( s , a ) \sim d _ { P ^ { \star } } ^ { \pi _ { k } } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } \right] } \\ & { \quad \leq \sqrt { \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } \left[ \frac { d _ { P ^ { \star } } ^ { \pi _ { k } } ( s , a ) } { \nu _ { k + 1 } ( s , a ) } \right] } \sqrt { \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \right] } } \\ & { \quad \leq \sqrt { \Lambda _ { k + 1 } } \sqrt { \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \right] } . } \end{array}
$$

For each $k \le K - 1$ , define

$$
G _ { k + 1 } : = \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \big [ \phi ^ { \star } ( s , a ) \phi ^ { \star } ( s , a ) ^ { \top } \big ] .
$$

Then

$$
\Sigma _ { k , \star } ^ { \mathrm { p r i } } = \lambda _ { k } I + \sum _ { i = 1 } ^ { k } G _ { i } .
$$

Since $\lambda _ { k + 1 } \geq \lambda _ { k }$ , we may apply Lemma B.2 to the PSD increments $G _ { 2 } , \dots , G _ { K }$ and obtain

$$
\sum _ { k = 1 } ^ { K - 1 } \operatorname { T r } \bigl ( G _ { k + 1 } ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } \bigr ) \lesssim \ln \operatorname* { d e t } ( \Sigma _ { K , \star } ^ { \mathrm { p r i } } ) - \ln \operatorname* { d e t } ( \lambda _ { 1 } I ) .
$$

Because

$$
\begin{array} { r } { \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \Big [ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \Big ] = \operatorname { T r } \big ( G _ { k + 1 } ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } \big ) , } \end{array}
$$

we deduce

$$
\sum _ { k = 1 } ^ { K - 1 } \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \right] \lesssim \ln \operatorname* { d e t } ( \Sigma _ { K , \star } ^ { \mathrm { p r i } } ) - \ln \operatorname* { d e t } ( \lambda _ { 1 } I ) .
$$

Lemma B.3 and the bound $\| \phi ^ { \star } ( s , a ) \| _ { 2 } \leq 1$ imply

$$
\ln \operatorname* { d e t } ( \Sigma _ { K , \star } ^ { \mathrm { p r i } } ) - \ln \operatorname* { d e t } ( \lambda _ { 1 } I ) \lesssim d \ln \left( 1 + { \frac { K } { d \lambda _ { 1 } } } \right) .
$$

Finally, Cauchy-Schwarz gives

$$
\sum _ { k = 1 } ^ { K - 1 } A _ { k } \leq \sqrt { \sum _ { k = 2 } ^ { K } \Lambda _ { k } } \sqrt { \sum _ { k = 1 } ^ { K - 1 } \mathbb { E } _ { ( s , a ) \sim \nu _ { k + 1 } } \left[ \| \phi ^ { \star } ( s , a ) \| _ { ( \Sigma _ { k , \star } ^ { \mathrm { p r i } } ) ^ { - 1 } } ^ { 2 } \right] } \lesssim \sqrt { d \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) \sum _ { k = 2 } ^ { K } \Lambda _ { k } } .
$$

Adding the bound on $A _ { 1 }$ proves the result.

Lemma D.12 (Summable optimistic overestimation of the utility). Fix an episode k and define

$$
\begin{array} { r } { \hat { U } _ { k } : = V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) , \qquad U _ { k } : = V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) . } \end{array}
$$

Assume Algorithm 1, the nonbinding-cap condition, and the events ofLemmas D.2, D.4, and D.6. Then, episode-wise,

$$
\begin{array} { r l } & { \left[ \hat { U } _ { k } - U _ { k } \right] _ { + } \lesssim \displaystyle \frac { 1 } { 1 - \gamma } A _ { k } \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \gamma d \Gamma _ { k } } + \frac { \bar { b } } { 1 - \gamma } A _ { k } \sqrt { \gamma \lambda _ { k } d } } \\ & { \qquad + \displaystyle \frac { \bar { b } } { ( 1 - \gamma ) ^ { 2 } } A _ { k } \sqrt { \gamma \big ( k \Gamma _ { k } \zeta _ { k } + \lambda _ { k } d \big ) } } \\ & { \qquad + \displaystyle \frac { 1 } { 1 - \gamma } \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \frac { \Gamma _ { k } d ( 1 - \gamma ) } { k } } + \frac { \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \Gamma _ { k } \zeta _ { k } ( 1 - \gamma ) } . } \end{array}
$$

Consequently, under the tuning ofProposition D.13,

$$
\sum _ { k = 1 } ^ { K } \bigl [ \hat { U } _ { k } - U _ { k } \bigr ] _ { + } \leq \tilde { \mathcal { O } } \biggl ( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \biggr ) \left[ \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \right] .
$$

Proof. Apply the second form of the simulation lemma (Lemma B.4) to the utility objective:

$$
\hat { U } _ { k } - U _ { k } = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } } \Big [ \hat { b } _ { k } ( s , a ) + \gamma \Delta _ { \hat { P } _ { k } , P ^ { \star } } W _ { k , g } ( s , a ) \Big ] ,
$$

where

$$
\begin{array} { r } { W _ { k , g } : = V _ { \hat { P } _ { k } , g + \hat { b } _ { k } } ^ { \pi _ { k } } , \quad \Delta _ { \hat { P } _ { k } , P ^ { \star } } W ( s , a ) : = \mathbb { E } _ { \hat { P } _ { k } } [ W ( s ^ { \prime } ) ] - \mathbb { E } _ { P ^ { \star } } [ W ( s ^ { \prime } ) ] . } \end{array}
$$

Since $\| \hat { b } _ { k } \| _ { \infty } \leq \bar { b }$ and the utility is normalized to lie in $[ 0 , 1 ]$ , we have $\| W _ { k , g } \| _ { \infty } \lesssim \bar { b } / ( 1 - \gamma )$ Therefore

$$
[ \hat { U } _ { k } - U _ { k } ] _ { + } \leq \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ \hat { b } _ { k } ] + \frac { \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ f _ { k } ] .
$$

The bonus term is controlled exactly as in the proof of Lemma D.10:

$$
\mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ \widehat { b } _ { k } ] \lesssim A _ { k } \sqrt { \gamma d \Gamma _ { k } ( \alpha _ { k } ^ { \mathrm { e m p } } ) ^ { 2 } + \gamma \lambda _ { k } d \bar { b } ^ { 2 } } + \alpha _ { k } ^ { \mathrm { e m p } } \sqrt { \frac { \Gamma _ { k } d ( 1 - \gamma ) } { k } } ,
$$

and the model-mismatch term obeys

$$
\mathbb { E } _ { d _ { P ^ { k } } ^ { \pi _ { k } } } [ f _ { k } ] \lesssim A _ { k } \sqrt { \gamma \big ( k \Gamma _ { k } \zeta _ { k } + \lambda _ { k } d \big ) } + \sqrt { \Gamma _ { k } \zeta _ { k } ( 1 - \gamma ) } .
$$

Substituting the last two displays proves the episode-wise bound.

For the cumulative statement, one substitutes the same choices of $\alpha _ { k } ^ { \mathrm { e m p } } , \lambda _ { k } , \zeta _ { k }$ as in Proposition D.13. The two $A _ { k }$ -weighted sums are then of order

$$
\tilde { \mathcal { O } } \left( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \Gamma _ { K } } \Psi _ { K } \right) , \qquad \Psi _ { K } : = \sum _ { k = 1 } ^ { K } A _ { k } ,
$$

while the two remaining sums are of order

$$
\tilde { \mathcal O } \left( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \right) .
$$

Finally Lemma D.11 gives

$$
\sqrt { \Gamma _ { K } } \Psi _ { K } \leq \tilde { \mathcal { O } } \left( \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } \right) ,
$$

which proves the cumulative bound.

Proposition D.13 (Theorem-level regret and soft-violation dependence). Assume Algorithm 1 and suppose the high-probability events required by Lemmas D.9, D.10, and D.11 hold simultaneously. Choose

$$
\begin{array} { r l r } { \alpha _ { k } ^ { \mathrm { m i x } } = \alpha _ { k } ^ { \mathrm { e m p } } = \tilde { \mathcal { O } } \left( \sqrt { ( \Gamma _ { k } + d ^ { 2 } ) \gamma \ln ( | \mathcal { M } | k / \delta ) } \right) , } & { } & \\ { \lambda _ { k } = \tilde { \Theta } \left( \frac { \Gamma _ { k } + d ^ { 2 } } { d } \ln ( | \mathcal { M } | k / \delta ) \right) , } & { } & \end{array}
$$

and choose

$$
\eta = \frac { 2 ( 1 - \gamma ) } { \theta \bar { b } \sqrt { K } } , \qquad \xi = \tilde { \Theta } ( 1 / \theta ) ,
$$

and assume the cap is nonbinding:

$$
\bar { b } \ge \operatorname* { m a x } _ { 1 \le j \le K } \frac { \alpha _ { j } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { j } } } .
$$

For a nonnegative sequence $z _ { 2 } , \dots , z _ { K }$ , define

$$
\Xi _ { K } ( z ) : = \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } z _ { k } } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } .
$$

Then

$$
\sum _ { k = 1 } ^ { K } \Delta _ { k } \leq \tilde { \mathcal { O } } \biggl ( \frac { ( 1 + \xi ) \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \biggr ) .
$$

Consequently, taking the comparator dual level to be $Y = 0 y i e l d s$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \biggl ( \frac { \bar { b } d ^ { 2 } } { \theta ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \biggr ) .
$$

Moreover,

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \left( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \right) .
$$

In particular, since $\Lambda _ { k } \le \Gamma _ { k } = | \mathcal { A } | / \epsilon _ { k }$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \biggl ( \frac { \bar { b } d ^ { 2 } } { \theta ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Gamma ) \biggr ) .
$$

Likewise,

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \biggl ( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Gamma ) \biggr ) .
$$

For $\epsilon _ { k } = k ^ { - p }$ with $p \in ( 0 , 1 / 2 )$ , this becomes

$$
\mathrm { R e g r e t ^ { r e t u r n } } ( K ) \le \tilde { \mathcal { O } } \Bigl ( K ^ { 1 / 2 + p } \Bigr ) , \quad \mathrm { V i o l a t i o n _ { s o f t } ^ { r e t u r n } } ( K ) \le \tilde { \mathcal { O } } \Bigl ( K ^ { 1 / 2 + p } \Bigr ) .
$$

up to problem-dependent prefactors, with the regret bound carrying the extra $\theta ^ { - 1 }$ dependence. Moreover, under the displayed tuning of $\lambda _ { k } ,$

$$
\frac { \alpha _ { k } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { k } } } = \tilde { \mathcal { O } } ( \sqrt { d } ) ,
$$

so the nonbinding-cap condition is compatible with any $\bar { b } = \tilde { \Omega } ( \sqrt { d } )$ and does not introduce any additional K-dependence.

Proof. Start from Lemma D.10 and substitute the tunings

$$
\alpha _ { k } ^ { \mathrm { e m p } } = \tilde { \cal O } \left( \sqrt { ( \Gamma _ { k } + d ^ { 2 } ) \gamma \ln ( | { \cal M } | k / \delta ) } \right) , \quad \lambda _ { k } = \tilde { \Theta } \left( \frac { \Gamma _ { k } + d ^ { 2 } } { d } \ln ( | { \cal M } | k / \delta ) \right) ,
$$

together with

$$
\zeta _ { k } = \tilde { \mathcal { O } } ( 1 / k ) .
$$

Under these choices, $\lambda _ { k } d = \tilde { \Theta } ( ( \Gamma _ { k } + d ^ { 2 } ) \ln ( | \mathcal { M } | k / \delta ) )$ , so every place in Lemma D.10 where $\lambda _ { k } d$ appears is of the same order as the dominant $( \Gamma _ { k } \dot { + } \dot { d } ^ { 2 } )$ term already present in $\alpha _ { k } ^ { \mathrm { e m p 2 } }$ . Therefore enlarging $\lambda _ { k }$ to make the cap nonbinding does not change the final K-dependence.

We also check the cap condition explicitly:

$$
\frac { \alpha _ { k } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { k } } } = \tilde { \mathcal { O } } \left( \sqrt { \frac { ( \Gamma _ { k } + d ^ { 2 } ) \ln ( \lvert \mathcal { M } \rvert k / \delta ) } { ( ( \Gamma _ { k } + d ^ { 2 } ) / d ) \ln ( \lvert \mathcal { M } \rvert k / \delta ) } } \right) = \tilde { \mathcal { O } } ( \sqrt { d } ) .
$$

Hence any choice $\bar { b } = \tilde { \Omega } ( \sqrt { d } )$ is sufficient to ensure the nonbinding-cap condition uniformly over $k \leq K$

Because $\Gamma _ { k }$ is nondecreasing, the dominant dependence in every episode-wise term may be upper bounded by $\Gamma _ { K }$ inside the coefficients that multiply $A _ { k } .$ . Consequently the two $A _ { k }$ -weighted sums in Lemma D.10 are both of order

$$
\tilde { \mathcal { O } } \biggl ( \frac { ( 1 + \xi ) \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \sqrt { \Gamma _ { K } } \Psi _ { K } \biggr ) , \qquad \Psi _ { K } : = \sum _ { k = 1 } ^ { K } A _ { k } ,
$$

The three remaining sums in Lemma D.10 contain no factor $A _ { k } ;$ after substituting the same choices of $\alpha _ { k } ^ { \mathrm { e m p } } , \lambda _ { k } , \zeta _ { k }$ , each of them is bounded by

$$
\tilde { \mathcal { O } } \left( \frac { ( 1 + \xi ) \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \right) .
$$

Now apply Lemma D.11, which gives

$$
\Psi _ { K } \lesssim \frac { 1 } { \sqrt { \lambda _ { 1 } } } + \sqrt { d \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) \sum _ { k = 2 } ^ { K } \Lambda _ { k } } .
$$

After absorbing the harmless initial term $1 / \sqrt { \lambda _ { 1 } }$ into the polylogarithmic factor, we obtain

$$
\sqrt { \Gamma _ { K } } \Psi _ { K } \leq \tilde { \mathcal { O } } \left( \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } \right) .
$$

This is exactly the quantity $\Xi _ { K } ( \Lambda )$ defined in the proposition, so

$$
\sum _ { k = 1 } ^ { K } \Delta _ { k } \leq \tilde { \mathcal { O } } \biggl ( \frac { ( 1 + \xi ) \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \biggr ) .
$$

The projection step in Algorithm 1 is identical to that of the main algorithm, so the proof of Lemma 2.4 applies verbatim and gives, for every $Y \in [ 0 , \xi ]$

$$
\sum _ { k = 1 } ^ { K } ( Y - Y _ { k } ) { \big ( } b - V _ { { \hat { P } } _ { k } , g + { \hat { b } } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) { \big ) } \leq { \frac { Y ^ { 2 } } { 2 \eta } } + { \frac { \eta { \bar { b } } ^ { 2 } K } { 2 ( 1 - \gamma ) ^ { 2 } } } .
$$

Setting $Y = 0$ and combining this with Lemma D.10 yields

$$
\sum _ { k = 1 } ^ { K } \bigl ( V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) \bigr ) \lesssim \sum _ { k = 1 } ^ { K } \Delta _ { k } + \frac { \eta \bar { b } ^ { 2 } K } { 2 ( 1 - \gamma ) ^ { 2 } } .
$$

Substituting the bound on $\textstyle \sum _ { k } \Delta _ { k }$ gives a right-hand side

$$
\tilde { \mathcal { O } } \left( \frac { \bar { b } d ^ { 2 } } { \theta ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \right) + \mathcal { O } \left( \frac { \bar { b } \sqrt { K } } { \theta ( 1 - \gamma ) } \right) ,
$$

$\begin{array} { r } { \eta = \frac { 2 ( 1 - \gamma ) } { \theta \bar { b } \sqrt { K } } } \end{array}$ . The second term is lower order than the first whenever $\epsilon _ { k } = k ^ { - p }$ with $p > 0$ , since then $\Xi _ { K } ( \Lambda ) = \tilde { \Omega } ( K ^ { 1 / 2 + p } )$ . This proves the displayed regret bound.

We now turn to the soft-violation term. Define

$$
\hat { U } _ { k } : = V _ { \hat { P } _ { k , g + \hat { b } _ { k } } } ^ { \pi _ { k } } ( s _ { 1 } ) , \qquad U _ { k } : = V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) , \qquad S _ { K } : = \sum _ { k = 1 } ^ { K } ( b - U _ { k } ) .
$$

Because $V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } ( s _ { 1 } ) \geq b ,$ we have

$$
V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) + Y _ { k } ( b - U _ { k } ) \leq \Delta _ { k } \qquad \forall k \in [ K ] .
$$

Hence, for every $Y \in [ 0 , \xi ]$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) + Y S _ { K } \leq \sum _ { k = 1 } ^ { K } \Delta _ { k } + \sum _ { k = 1 } ^ { K } ( Y - Y _ { k } ) ( b - \hat { U } _ { k } ) + \sum _ { k = 1 } ^ { K } ( Y - Y _ { k } ) ( \hat { U } _ { k } - U _ { k } ) .
$$

Choose $Y = \xi$ when $S _ { K } > 0$ and $Y = 0$ otherwise. Then

$$
\mathrm { R e g r e t ^ { r e t u r n } } ( K ) + \xi [ S _ { K } ] _ { + } \leq \sum _ { k = 1 } ^ { K } \Delta _ { k } + \frac { \xi ^ { 2 } } { 2 \eta } + \frac { \eta \bar { b } ^ { 2 } K } { 2 ( 1 - \gamma ) ^ { 2 } } + \xi \sum _ { k = 1 } ^ { K } [ \hat { U } _ { k } - U _ { k } ] _ { + } .
$$

Now use Lemma D.12 together with the bound on $\sum _ { k } \Delta _ { k }$ . This gives

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) + \xi [ S _ { K } ] _ { + } \leq \tilde { \mathcal { O } } \biggl ( \frac { \xi \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \biggr ) + \mathcal { O } \biggl ( \frac { \xi \bar { b } \sqrt { K } } { 1 - \gamma } \biggr ) .
$$

By linearity of discounted occupancy measures, there exists a policy $\pi ^ { \prime }$ whose reward and utility values equal the episode averages:

$$
V _ { P ^ { \star } , r } ^ { \pi ^ { \prime } } ( s _ { 1 } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) , \qquad V _ { P ^ { \star } , g } ^ { \pi ^ { \prime } } ( s _ { 1 } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) .
$$

Therefore

$$
{ V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } } ( s _ { 1 } ) - { V _ { P ^ { \star } , r } ^ { \pi ^ { \prime } } } ( s _ { 1 } ) + \xi \big [ b - { V _ { P ^ { \star } , g } ^ { \pi ^ { \prime } } } ( s _ { 1 } ) \big ] _ { + } \le \frac { 1 } { K } \Big ( \mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) + \xi [ S _ { K } ] _ { + } \Big ) .
$$

By Lemma 2.4, the optimal dual variable satisfies $Y ^ { \star } \le 1 / \theta$ , so the constant hidden in $\xi = \tilde { \Theta } ( 1 / \theta )$ may be chosen so that $\xi \geq 2 Y ^ { \star }$ . Applying Lemma B.5 yields

$$
[ S _ { K } ] _ { + } \leq { \frac { 2 } { \xi } } { \Bigl ( } { \mathrm { R e g r e t } } ^ { { \mathrm { r e t u r n } } } ( K ) + \xi [ S _ { K } ] _ { + } { \Bigr ) } .
$$

Since $\xi ^ { - 1 } = { \tilde { \mathcal { O } } } ( \theta )$ and $\Xi _ { K } ( \Lambda ) = \tilde { \Omega } ( \sqrt { K } )$ , the lower-order $\mathcal { O } ( \xi \bar { b } \sqrt { K } / ( 1 - \gamma ) )$ term is absorbed, and we obtain

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) = [ S _ { K } ] _ { + } \leq \tilde { \mathcal { O } } \biggl ( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \biggr ) .
$$

This proves the displayed soft-violation bound.

Proposition D.14 (Execution policy corollary for a stationary-mixture variant). The guarantees in Proposition D.13 are statedfor the return policy sequence $\{ \pi _ { k } \} _ { k = 1 } ^ { K }$ under the discounted-occupancy oracle protocol. The present proposition records a separate $f u l l y$ online interpretation in which episode k executes the stationary mixed policy

$$
\tilde { \pi } _ { k } ( \cdot \mid s ) : = ( 1 - \epsilon _ { k } ) \pi _ { k } ( \cdot \mid s ) + \epsilon _ { k } U ( A ) .
$$

This online mixture is not the probing protocol analyzed above; it is included only to translate return policy guarantees into execution policy guarantees.

Define

$$
\mathrm { R e g r e t } ^ { \mathrm { e x e c } } ( K ) : = \sum _ { k = 1 } ^ { K } \bigl ( V _ { r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { r } ^ { \tilde { \pi } _ { k } } ( s _ { 1 } ) \bigr ) ,
$$

$$
\operatorname { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) : = \left[ \sum _ { k = 1 } ^ { K } \left( b - V _ { g } ^ { \tilde { \pi } _ { k } } ( s _ { 1 } ) \right) \right] _ { + } .
$$

Then

$$
\mathrm { R e g r e t } ^ { \mathrm { e x e c } } ( K ) \le \mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) + \frac { 2 } { 1 - \gamma } \sum _ { k = 1 } ^ { K } \epsilon _ { k } ,
$$

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) \le \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) + \frac { 2 } { 1 - \gamma } \sum _ { k = 1 } ^ { K } \epsilon _ { k } .
$$

In particular, $i f \epsilon _ { k } = k ^ { - p }$ with $p \in [ 1 / 4 , 1 / 2 )$ , then Proposition D.13 implies

$$
\mathrm { R e g r e t } ^ { \mathrm { e x e c } } ( K ) , \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) \le \tilde { \mathcal { O } } \left( K ^ { 1 / 2 + p } \right) ,
$$

which matches the asymptotic orders ofthe return policy bounds.

Proof. Fix an episode k. For $j \in \{ r , g \}$ , define $A _ { j } ^ { \pi _ { k } } ( s , a ) : = Q _ { j } ^ { \pi _ { k } } ( s , a ) - V _ { j } ^ { \pi _ { k } } ( s )$ . By the trajectorylevel normalization introduced in the background section, $| A _ { i } ^ { \pi _ { k } } \bar { ( s , a ) } | \leq 1$ for both $j = r$ and $j = g$ The performance-difference lemma with normalized discounted occupancy measure gives

$$
( 1 - \gamma ) \big ( V _ { j } ^ { \tilde { \pi } _ { k } } \big ( s _ { 1 } \big ) - V _ { j } ^ { \pi _ { k } } \big ( s _ { 1 } \big ) \big ) = \mathbb { E } _ { s \sim d _ { P ^ { k } } ^ { \tilde { \pi } _ { k } } } \left[ \sum _ { a \in \mathcal { A } } \big ( \tilde { \pi } _ { k } ( a \mid s ) - \pi _ { k } ( a \mid s ) \big ) A _ { j } ^ { \pi _ { k } } ( s , a ) \right] .
$$

Hence

$$
D _ { k , j } ( s ) : = \sum _ { a \in \mathcal { A } } \left| \tilde { \pi } _ { k } ( a \mid s ) - \pi _ { k } ( a \mid s ) \right| \cdot \left| A _ { j } ^ { \pi _ { k } } ( s , a ) \right| .
$$

Therefore

$$
\big | V _ { j } ^ { \tilde { \pi } _ { k } } ( s _ { 1 } ) - V _ { j } ^ { \pi _ { k } } ( s _ { 1 } ) \big | \le \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { P ^ { * } } ^ { \tilde { \pi } _ { k } } } \big [ D _ { k , j } ( s ) \big ] \le \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d _ { P ^ { * } } ^ { \tilde { \pi } _ { k } } } \Big [ \| \tilde { \pi } _ { k } ( \cdot \vert s ) - \pi _ { k } ( \cdot \vert s ) \| _ { 1 } \Big ] .
$$

Now

$$
\tilde { \pi } _ { k } ( \cdot  { | } s ) - \pi _ { k } ( \cdot  { | } s ) = \epsilon _ { k } \bigl ( U ( A ) - \pi _ { k } ( \cdot  { | } s ) \bigr ) , \quad \| \tilde { \pi } _ { k } ( \cdot  { | } s ) - \pi _ { k } ( \cdot  { | } s ) \| _ { 1 } \leq \epsilon _ { k } \| U ( A ) - \pi _ { k } ( \cdot  { | } s ) \| _ { 1 } \leq 2 \epsilon _ { k } .
$$

Substituting yields, for both $j = r$ and $j = g ,$

$$
\left. V _ { j } ^ { \tilde { \pi } _ { k } } ( s _ { 1 } ) - V _ { j } ^ { \pi _ { k } } ( s _ { 1 } ) \right. \leq \frac { 2 \epsilon _ { k } } { 1 - \gamma } .
$$

Applying this bound with $j = r ,$ , we obtain

$$
\begin{array} { r l } & { \mathrm { R e g r e t } ^ { \mathrm { e x e c } } ( K ) = \displaystyle \sum _ { k = 1 } ^ { K } \bigl ( V _ { r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { r } ^ { \pi _ { k } } ( s _ { 1 } ) \bigr ) + \displaystyle \sum _ { k = 1 } ^ { K } \bigl ( V _ { r } ^ { \pi _ { k } } ( s _ { 1 } ) - V _ { r } ^ { \pi _ { k } } ( s _ { 1 } ) \bigr ) } \\ & { \qquad \le \mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) + \displaystyle \frac { 2 } { 1 - \gamma } \displaystyle \sum _ { k = 1 } ^ { K } \epsilon _ { k } . } \end{array}
$$

Applying the same bound with $j = g ,$ , together with $[ x + y ] _ { + } \leq [ x ] _ { + } + y$ for $y \geq 0 ,$ , yields

$$
\begin{array} { l } { \displaystyle \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) = \left[ \sum _ { k = 1 } ^ { K } \bigl ( b - V _ { g } ^ { \bar { \pi } _ { k } } \bigl ( s _ { 1 } \bigr ) \bigr ) \right] _ { + } } \\ { \displaystyle \qquad \leq \left[ \sum _ { k = 1 } ^ { K } \bigl ( b - V _ { g } ^ { \pi _ { k } } \bigl ( s _ { 1 } \bigr ) \bigr ) \right] _ { + } + \sum _ { k = 1 } ^ { K } \bigl | V _ { g } ^ { \bar { \pi } _ { k } } \bigl ( s _ { 1 } \bigr ) - V _ { g } ^ { \pi _ { k } } \bigl ( s _ { 1 } \bigr ) \bigr | } \\ { \displaystyle \qquad \leq \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) + \frac { 2 } { 1 - \gamma } \sum _ { k = 1 } ^ { K } \epsilon _ { k } . } \end{array}
$$

Finally, if $\epsilon _ { k } = k ^ { - p }$ with $p \in [ 1 / 4 , 1 / 2 )$ , then Proposition D.13 gives

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \le \tilde { \mathcal { O } } ( K ^ { 1 / 2 + p } ) , \qquad \sum _ { k = 1 } ^ { K } \epsilon _ { k } = \sum _ { k = 1 } ^ { K } k ^ { - p } = \mathcal { O } ( K ^ { 1 - p } ) .
$$

Since $p \ge 1 / 4$ is equivalent to $1 - p \leq 1 / 2 + p .$ , the additional execution term is of no larger order than the returned-policy terms. Hence Regre $\cdot ^ { \mathrm { e x e c } } ( K ) \leq \tilde { \mathcal { O } } ( K ^ { 1 / 2 + p } )$ . The same comparison applies to Violatio $1 _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K )$ , because the returned-policy soft-violation term is already of order $\tilde { \mathcal { O } } ( K ^ { 1 / 2 + p } )$ Therefore Violation $\mathsf { l } _ { \mathrm { s o f t } } ^ { \mathrm { e x e c } } ( K ) \le \tilde { \mathcal { O } } ( K ^ { 1 / 2 + p } )$ □

## E Uniform Exploration as a Tertiary-Aware Repaired Variant

This appendix specializes the decaying-ϵ analysis in Appendix D to the constant schedule $\epsilon _ { k } \equiv 1$ This case is useful as a clean benchmark: all probe actions are drawn uniformly, while the tertiary probe level and merged bonus covariance are retained so that the repaired optimism proof still applies.

Throughout this appendix, we set

$$
q _ { k } ( \cdot \mid s ) = q _ { k } ^ { \prime } ( \cdot \mid s ) = q _ { k } ^ { \prime \prime } ( \cdot \mid s ) = U ( A ) , \qquad \forall k \in [ K ] .
$$

Equivalently, this is Algorithm 1 with $\epsilon _ { k } \equiv 1$ . Therefore

$$
\Gamma _ { k } = \Lambda _ { k } = | { \cal { A } } | , \qquad \forall k \in [ K ] .
$$

## E.1 Uniform Specialization

For each round $i \in [ K ]$ , define the primary, secondary, and tertiary state marginals by

$$
\begin{array} { r } { \rho _ { i } ( s ) : = d _ { P ^ { \star } } ^ { \pi _ { i - 1 } } ( s ) , \quad \rho _ { i } ^ { \prime } ( s ^ { \prime } ) : = \mathbb { E } _ { \rho _ { i } \times U ( A ) } P ^ { \star } ( s ^ { \prime } \mid s , a ) , \quad \rho _ { i } ^ { \prime \prime } ( s ^ { \prime \prime } ) : = \mathbb { E } _ { \rho _ { i } ^ { \prime } \times U ( A ) } P ^ { \star } ( s ^ { \prime \prime } \mid s , a ) . } \end{array}
$$

The corresponding probing distributions are

$$
\nu _ { i } ( s , a ) : = \rho _ { i } ( s ) U ( a ) , \quad \nu _ { i } ^ { \prime } ( s , a ) : = \rho _ { i } ^ { \prime } ( s ) U ( a ) , \quad \nu _ { i } ^ { \prime \prime } ( s , a ) : = \rho _ { i } ^ { \prime \prime } ( s ) U ( a ) .
$$

We write

$$
\bar { \nu } _ { k } ^ { \mathrm { p r i } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } , \qquad \bar { \nu } _ { k } ^ { \mathrm { s e c } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } ^ { \prime } , \qquad \bar { \nu } _ { k } ^ { \mathrm { t e r } } : = \frac 1 k \sum _ { i = 1 } ^ { k } \nu _ { i } ^ { \prime \prime } .
$$

No separate pseudocode is needed: this is exactly Algorithm 1 with the constant schedule $\epsilon _ { k } \equiv 1$ Relative to the general algorithm, the only change is that all probing kernels are fixed to $U ( { \mathcal { A } } ) ;$ ; the tertiary tuple, merged covariance, model fitting step, and primal-dual update remain the same.

## E.2 Supporting Specializations

The tuple-level MLE event, the pathwise-to-averaged bridge for the squared MLE-mismatch class, and the merged covariance comparison from Appendix D apply verbatim to Algorithm 1 with $\epsilon _ { k } \equiv 1$ The only new point is that all coverage coefficients collapse to explicit constants.

Proposition E.1 (Uniform coverage specialization). Under Algorithm 1 with $\epsilon _ { k } \equiv 1$ , for every episode k and every policy π,

$$
\mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) \le | \mathcal { A } | , \qquad \mathfrak { C } _ { k } ^ { \mathrm { n e x t } } ( \pi ) \le | \mathcal { A } | ,
$$

and

$$
\mathfrak { C } _ { k , 1 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le | A | , \qquad \mathfrak { C } _ { k , 2 } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le | A | , \qquad \mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le | A | .
$$

Proof. Because every probe kernel is uniform,

$$
\bar { \nu } _ { k } ^ { \mathrm { p r i } } ( s , a ) = \frac { 1 } { \vert \mathcal { A } \vert } \cdot \frac { 1 } { k } \sum _ { i = 1 } ^ { k } \rho _ { i } ( s ) ,
$$

and every discounted occupancy satisfies

$$
\begin{array} { r } { \rho _ { i } ( s ) = d _ { P ^ { \star } } ^ { \pi _ { i - 1 } } ( s ) \geq ( 1 - \gamma ) d _ { 1 } ( s ) . } \end{array}
$$

Hence

$$
\bar { \nu } _ { k } ^ { \mathrm { p r i } } ( s , a ) \geq \frac { ( 1 - \gamma ) d _ { 1 } ( s ) } { | \mathcal { A } | } ,
$$

which implies

$$
{ \mathfrak { C } } _ { k } ^ { \mathrm { { i n i t } } } ( \pi ) \leq | { \mathcal { A } } |
$$

for every policy π.

The bound on ${ \mathfrak { C } } _ { k } ^ { \mathrm { n e x t } } ( \pi )$ follows exactly as in the uniform version of the original proof: the next-step pushforward state-action measure has density at most $| { \cal A } |$ times $\bar { \nu } _ { k } ^ { \mathrm { s e c } }$ , because the action component of $\bar { \nu } _ { k } ^ { \mathrm { s e c } }$ is uniform.

The tertiary coefficients are the specialization of Proposition D.7 to the constant schedule $\epsilon _ { k } \equiv 1$ , for which $\Gamma _ { k } \overset { \vartriangle } { = } | \boldsymbol { \mathcal { A } } |$ □

Proposition E.2 (Uniform potential reduction). Under Algorithm 1 with $\epsilon _ { k } \equiv 1$

$$
\Psi _ { K } : = \sum _ { k = 1 } ^ { K } A _ { k } \lesssim \frac { 1 } { \sqrt { \lambda _ { 1 } } } + \sqrt { \vert A \vert d K \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) } .
$$

Proof. This is the specialization of Lemma D.11 to $\Lambda _ { k } = | { \mathcal { A } } |$ for all k:

$$
\Psi _ { K } \lesssim \frac { 1 } { \sqrt { \lambda _ { 1 } } } + \sqrt { d \ln \biggl ( 1 + \frac { K } { d \lambda _ { 1 } } \biggr ) \sum _ { k = 2 } ^ { K } \Lambda _ { k } } .
$$

Since $\textstyle \sum _ { k = 2 } ^ { K } \Lambda _ { k } = ( K - 1 ) | { \mathcal { A } } |$ |, the claim follows.

## E.3 Theorem-Level Derivation

Theorem E.3 (Return policy regret and soft violation for $p = 0 )$ . Assume Algorithm 1 is run with $\epsilon _ { k } \equiv 1$ , and suppose the high-probability events required by Lemmas D.9, D.10, and D.11 hold simultaneously. Choose

$$
\alpha _ { k } ^ { \mathrm { m i x } } = \alpha _ { k } ^ { \mathrm { e m p } } = \tilde { \mathcal { O } } \left( \sqrt { ( | A | + d ^ { 2 } ) \gamma \ln ( | \mathcal { M } | k / \delta ) } \right) ,
$$

$$
\lambda _ { k } = \tilde { \Theta } \bigg ( \frac { | \mathcal { A } | + d ^ { 2 } } { d } \ln ( | \mathcal { M } | k / \delta ) \bigg ) ,
$$

and

$$
\eta = \frac { 2 ( 1 - \gamma ) } { \theta \bar { b } \sqrt { K } } , \qquad \xi = \tilde { \Theta } ( 1 / \theta ) ,
$$

and assume the cap is nonbinding:

$$
\bar { b } \ge \operatorname* { m a x } _ { 1 \le j \le K } \frac { \alpha _ { j } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { j } } } .
$$

Then, with probability at least $1 - \delta ,$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \left( \frac { | \mathcal { A } | \bar { b } d ^ { 2 } \sqrt { K } } { \theta ( 1 - \gamma ) ^ { 2 } } \right) , \quad \mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \left( \frac { | \mathcal { A } | \bar { b } d ^ { 2 } \sqrt { K } } { ( 1 - \gamma ) ^ { 2 } } \right) .
$$

Proof. This is a direct specialization of Proposition D.13 to the constant schedule $\epsilon _ { k } \equiv 1$ . Under this choice,

$$
\Gamma _ { k } = \Lambda _ { k } = | { \cal { A } } | , \qquad \forall k \in [ K ] .
$$

Therefore the quantity

$$
\Xi _ { K } ( \Lambda ) : = \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } }
$$

satisfies

$$
\Xi _ { K } ( \Lambda ) = \sqrt { | { \cal A } | \cdot ( K - 1 ) | { \cal A } | } + | { \cal A } | \sum _ { k = 1 } ^ { K } \frac { 1 } { \sqrt { k } } \lesssim | { \cal A } | \sqrt { K } .
$$

Substituting this into Proposition D.13 yields

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \left( \frac { \bar { b } d ^ { 2 } } { \theta ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \right) \leq \tilde { \mathcal { O } } \left( \frac { | A | \bar { b } d ^ { 2 } \sqrt { K } } { \theta ( 1 - \gamma ) ^ { 2 } } \right) ,
$$

and

$$
\mathrm { V i o l a t i o n } _ { \mathrm { s o f t } } ^ { \mathrm { r e t u r n } } ( K ) \leq \tilde { \mathcal { O } } \left( \frac { \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \Xi _ { K } ( \Lambda ) \right) \leq \tilde { \mathcal { O } } \left( \frac { | A | \bar { b } d ^ { 2 } \sqrt { K } } { ( 1 - \gamma ) ^ { 2 } } \right) .
$$

It remains only to note that the displayed tuning of $\lambda _ { k }$ makes the cap condition compatible with a K-independent choice of <sup>¯</sup>b. Indeed,

$$
\frac { \alpha _ { k } ^ { \mathrm { e m p } } } { \sqrt { \lambda _ { k } } } = \tilde { \mathcal { O } } \left( \sqrt { \frac { \left( | A | + d ^ { 2 } \right) \ln ( | \mathcal { M } | k / \delta ) } { \left( \left( | A | + d ^ { 2 } \right) / d \right) \ln ( | \mathcal { M } | k / \delta ) } } \right) = \tilde { \mathcal { O } } ( \sqrt { d } ) ,
$$

so any $\bar { b } = \tilde { \Omega } ( \sqrt { d } )$ is sufficient, exactly as in Appendix D.

Constraint discussion. The regret proof uses the same representation-learning and optimism ingredients as Appendix D. For soft violation, one additional ingredient is needed: the cumulative overestimation lemma controls how much the optimistic utility used by the dual update can exceed the true utility. This term is summable under the repaired tertiary design, so the standard projected dual argument yields the stated soft-violation bound.

Remark E.4 (Relation to the $p = 0$ specialization). The uniform protocol is exactly the $\epsilon _ { k } \equiv 1$ specialization of Algorithm 1 in Section 3. Consequently, this appendix gives the fully explicit return-policy theorem for the $p = 0$ schedule, with the familiar ${ \tilde { \mathcal { O } } } ( | { \tilde { \mathcal { A } } } | d ^ { 2 } { \sqrt { K } } )$ dependence up to the same problem-dependent prefactors and logarithmic terms.

## F Hard Constraints Under Decaying Tertiary Probing

This appendix gives the repaired hard-violation analysis. The key distinction is between the one-sided correction used by the algorithm and the two-sided width used only in the proof. In particular, we do not assume an external certificate event and we do not require a summable radius uniformly over all policies queried by the inner loop.

## F.1 Algorithm and Certificate

The representation-learning front end is the same as in Algorithm 1. At episode k, let

$$
q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } : = ( 1 - \epsilon _ { k } ) \pi _ { k - 1 } + \epsilon _ { k } U ( A ) , \qquad \Gamma _ { k } : = \frac { | A | } { \epsilon _ { k } } ,
$$

and let

$$
\zeta _ { k } : = \frac { \ln ( | \mathcal { M } | k / \delta ) } { k } .
$$

After fitting $\widehat { P } _ { k } = ( \widehat { \mu } _ { k } , \widehat { \phi } _ { k } )$ on the three streams and forming $\widehat { \Sigma } _ { k } ^ { \mathrm { m i x } }$ , define

$$
\widehat { b } _ { k } ( s , a ) : = \operatorname* { m i n } \Bigl \{ \alpha _ { k } \| \widehat { \phi } _ { k } ( s , a ) \| _ { ( \widehat { \Sigma } _ { k } ^ { \operatorname* { m i x } } ) ^ { - 1 } } , \bar { b } \Bigr \} .
$$

For a sufficiently large universal calibration constant $C _ { \mathrm { c e r t } }$ , set

$$
\ell _ { k } : = C _ { \mathrm { c e r t } } \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } , \qquad \mathcal { U } _ { k } ( \pi ) : = \left[ V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - \ell _ { k } \right] _ { + } .\tag{G.1}
$$

The quantity $\ell _ { k }$ is the one-sided statistical correction used by the algorithm. For analysis, define the policy-dependent selected width

$$
w _ { k } ( \pi ) : = 2 \ell _ { k } + 2 V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) .\tag{G.2}
$$

Unlike the correction $\ell _ { k }$ , the width $w _ { k } ( \pi )$ is not required as an algorithmic input.

Let $Y _ { \mathrm { m a x } } : = K ^ { 1 / 4 }$ , and use the capped grid

$$
\mathcal { V } _ { K } : = \{ 0 , \eta _ { h } , 2 \eta _ { h } , \ldots \} \cap [ 0 , Y _ { \operatorname* { m a x } } ] \cup \{ Y _ { \operatorname* { m a x } } \} .
$$

Thus the cap itself is always queried even when $Y _ { \mathrm { m a x } } / \eta _ { h }$ is not an integer.

## F.2 Derived Certificate Event

Let $\mathcal { E } _ { \mathrm { r e p } }$ be the intersection of the tuple-level MLE, pathwise bridge, and covariance-comparison events in Lemmas D.2, D.4, and D.6. Replacing δ by a constant fraction in the individual events makes $\mathbb { P } ( \mathcal { E } _ { \mathrm { r e p } } ) \geq 1 - \delta$

Lemma F.1 (One-sided certificate and selected-policy interval). Assume $\mathcal { E } _ { \mathrm { r e p } }$ , the nonbinding-cap condition

$$
{ \bar { b } } \geq \operatorname* { m a x } _ { j \leq K } { \frac { \alpha _ { j } } { \sqrt { \lambda _ { j } } } } ,
$$

and the parameter lower bound on $\alpha _ { k }$ in Lemma D.9 with ${ \mathfrak { C } } _ { k } ^ { \mathrm { { t r i } , { n e x t } } } ( \pi ) \leq \Gamma _ { k }$ . For a sufficiently large $C _ { \mathrm { { c e r t } } } ,$ , simultaneously for every episode k and every policy π,

$$
\mathcal { U } _ { k } ( \pi ) \leq V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \leq \mathcal { U } _ { k } ( \pi ) + w _ { k } ( \pi ) .\tag{G.3}
$$

Proof. We first verify that the residual in the one-step-back argument is uniform in $\pi .$ . Since every primary occupancy contains the initial component $( 1 - \gamma ) d _ { 1 }$ , and every probing kernel up to episode k has action floor $\epsilon _ { k } / | \mathcal { A } |$

$$
\bar { \nu } _ { k } ^ { \mathrm { p r i } } ( s , a ) \geq ( 1 - \gamma ) d _ { 1 } ( s ) \frac { \epsilon _ { k } } { | \mathcal { A } | } .
$$

Consequently

$$
\mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) = \mathsf { D } \Big ( ( 1 - \gamma ) d _ { 1 } \otimes \pi \ \Big \| \ \bar { \nu } _ { k } ^ { \mathrm { p r i } } \Big ) \leq \Gamma _ { k }
$$

Algorithm 2 REP-PD-H-TEG with an explicit one-sided certificate   
1: Input: Parameters $\{ \lambda _ { k } , \alpha _ { k } , \epsilon _ { k } \} _ { k = 1 } ^ { K } ,$ confidence $\delta ,$ calibration $C _ { \mathrm { c e r t } }$ , temperature $\tau _ { h }$ , grid step   
$\eta _ { h } .$ , model class $\mathcal { M } ,$ and episodes $K$   
2: Initialize $\pi _ { 0 } ( \cdot \mid s ) = U ( { \dot { A } } )$ and $D _ { 0 } = D _ { 0 } ^ { \prime } = D _ { 0 } ^ { \prime \prime } = \emptyset .$   
3: for $k = 1 , \ldots , \tilde { K }$ do   
4: Set $q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } = ( 1 - \epsilon _ { k } ) \pi _ { k - 1 } + \epsilon _ { k } U ( \mathcal { A } ) .$   
5: Query one tertiary tuple and update $D _ { k } , \dot { D _ { k } ^ { \prime } } , \dot { D } _ { k } ^ { \prime \prime }$ exactly as in Algorithm 1.   
6: Fit   
$\widehat { P } _ { k } = ( \widehat { \mu } _ { k } , \widehat { \phi } _ { k } ) \in \arg \operatorname* { m a x } _ { ( \mu , \phi ) \in \mathcal { M } } \mathbb { E } _ { D _ { k } + D _ { k } ^ { \prime } + D _ { k } ^ { \prime \prime } } [ \log \mu ( x ^ { \prime } ) ^ { \top } \phi ( x , u ) ] .$   
7: Form $\widehat { \Sigma } _ { k } ^ { \mathrm { m i x } }$ and $\widehat { b } _ { k }$ as above, and set   
$\ell _ { k } = C _ { \mathrm { c e r t } } \sqrt { \frac { | \mathcal { A } | \ln ( | \mathcal { M } | k / \delta ) } { \epsilon _ { k } k ( 1 - \gamma ) } } .$   
8: Set found ← false.   
9: for $Y$ in the increasing grid $\mathcal { D } _ { K }$ do   
10: Compute   
$\widehat { \pi } _ { k , Y } \in \arg \operatorname* { m a x } _ { \pi } \left( Q _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \pi } + Y Q _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } \right)$   
11: Form the statewise softmax candidate   
$\pi _ { k , Y } ( \cdot \mid x ) = \operatorname { s o f t m a x } _ { \tau _ { h } } \left( Q _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \widehat { \pi } _ { k , Y } } ( x , \cdot ) + Y Q _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \widehat { \pi } _ { k , Y } } ( x , \cdot ) \right)$   
12: Compute   
$\underline { { \mathcal { U } } } _ { k } ( \pi _ { k , Y } ) = \left[ V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi _ { k , Y } } ( s _ { 1 } ) - \ell _ { k } \right] _ { + }$   
13: if $\underline { { \mathcal { U } } } _ { k } ( \pi _ { k , Y } ) \geq b$ then   
14: Set $( \pi _ { k } , Y _ { k } ) \gets ( \pi _ { k , Y } , Y )$ and found true.   
15: break   
16: end if   
17: end for   
18: if found = false then   
19: Set $( \pi _ { k } , Y _ { k } ) \gets ( \pi _ { k , Y _ { \operatorname* { m a x } } } , Y _ { \operatorname* { m a x } } )$   
20: end if   
21: end for   
22: Return $\pi _ { 1 } , \ldots , \pi _ { K } .$

for every policy π. Proposition D.7 likewise gives ${ \mathfrak { C } } _ { k } ^ { \mathrm { { t r i } , { n e x t } } } ( \pi ) \leq \Gamma _ { k }$ uniformly in π.   
Write $f _ { k } ( s , a ) = \| \widehat { P } _ { k } ( \cdot \mid s , a ) - P ^ { \star } ( \cdot \mid s , a ) \| _ { 1 }$ . The first form of the simulation lemma gives   
$\begin{array} { r l } & { ( 1 - \gamma ) \left( V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - V _ { { P } ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \right) } \\ & { \quad = \mathbb { E } _ { d _ { \widehat { P } _ { k } } ^ { \pi } } \left[ - \widehat { b } _ { k } ( s , a ) + \gamma \left( \mathbb { E } _ { \widehat { P } _ { k } } - \mathbb { E } _ { P ^ { \star } } \right) V _ { { P } ^ { \star } , g } ^ { \pi } ( s ^ { \prime } ) \right] . } \end{array}$ (G.4)   
The normalized utility assumption implies $\| V _ { P ^ { \star } , g } ^ { \pi } \| _ { \infty } \leq 1 .$ Applying Lemma D.8 with $g = f _ { k }$ , using   
$\mathfrak { C } _ { k } ^ { \mathrm { i n i t } } ( \pi ) , \mathfrak { C } _ { k } ^ { \mathrm { t r i } , \mathrm { n e x t } } ( \pi ) \le \Gamma _ { k }$ , and choosing $\alpha _ { k }$ to absorb the covariance-dependent term yields   
$V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \leq C _ { \mathrm { c e r t } } \sqrt { \frac { \Gamma _ { k } \zeta _ { k } } { 1 - \gamma } } = \ell _ { k } .$ (G.5)   
The proof is deterministic on $\mathcal { E } _ { \mathrm { r e p } } ,$ so no additional union bound over the adaptively queried policies   
is required. Since the true utility is nonnegative, (G.5) proves the first inequality in (G.3) after   
positive-part clipping.   
Repeating the direct-optimism argument of Lemma D.9 for the normalized utility signal g gives   
$\begin{array} { r } { V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \le V _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) + \ell _ { k } . } \end{array}$ (G.6)

For the same model and policy, discounted value is linear in the signal:

$$
\begin{array} { r } { V _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) = 2 V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) . } \end{array}\tag{G.7}
$$

Combining (G.6), (G.7), and the raw lower endpoint $V _ { \widehat { P } _ { k } , g - \widehat { b } _ { k } } ^ { \pi } - \ell _ { k }$ proves the upper inequality with $w _ { k } ( \pi ) = 2 \ell _ { k } + 2 V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi }$ . Replacing a negative raw endpoint by zero only tightens this upper bound because $V _ { P ^ { \star } , g } ^ { \pi } \geq 0$ □

For the selected policy sequence, define

$$
\mathcal { B } _ { K } : = \widetilde O \left( \frac { \overline { { b } } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } \left[ \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \right] \right) ,\tag{G.8}
$$

where

$$
\Lambda _ { k } : = \frac { 1 } { 1 - \epsilon _ { k } + \epsilon _ { k } / | \mathcal { A } | } .
$$

Lemma F.2 (Selected-policy width summability). On $\mathcal { E } _ { \mathrm { r e p } } ,$ , the policy sequence returned by $A l g o \mathrm { - }$ rithm 2 satisfies

$$
\sum _ { k = 1 } ^ { K } w _ { k } ( \pi _ { k } ) \leq { \cal B } _ { K } + 2 \sum _ { k = 1 } ^ { K } \ell _ { k } .\tag{G.9}
$$

Proof. Condition on the history before episode k. Apply the second form of the simulation lemma to the signal $\widehat { b } _ { k }$ , with no additional bonus:

$$
V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) - V _ { P ^ { \star } , \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) = \frac { \gamma } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } \left[ \left( \mathbb { E } _ { \widehat { P } _ { k } } - \mathbb { E } _ { P ^ { \star } } \right) V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi _ { k } } ( s ^ { \prime } ) \right] .
$$

Because $0 \leq \widehat { b } _ { k } \leq \bar { b } , \| V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi _ { k } } \| _ { \infty } \leq \bar { b } / ( 1 - \gamma )$ , and hence

$$
V _ { \widehat { P } _ { k } , \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) \leq \frac { 1 } { 1 - \gamma } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ \widehat { b } _ { k } ] + \frac { \gamma \bar { b } } { ( 1 - \gamma ) ^ { 2 } } \mathbb { E } _ { d _ { P ^ { \star } } ^ { \pi _ { k } } } [ f _ { k } ] .\tag{G.10}
$$

The two expectations in (G.10) are exactly the bonus and model-error terms bounded in the proof of Lemma D.10; their cumulative bound follows from Lemma D.11 and is $\boldsymbol { B } _ { K }$ up to universal constants. Summing (G.2) and using (G.10) proves (G.9). Importantly, this step is applied only to the returned sequence $\left\{ \pi _ { k } \right\}$ , for which the deployed-to-probe coupling used by Lemma D.11 is valid. □

For a polynomial schedule $\epsilon _ { k } = k ^ { - p }$

$$
\sum _ { k = 1 } ^ { K } \ell _ { k } \leq \widetilde O \left( \sqrt { \frac { | A | } { 1 - \gamma } } K ^ { ( 1 + p ) / 2 } \right) ,\tag{G.11}
$$

while

$$
B _ { K } \leq \widetilde { O } \left( \frac { | A | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } K ^ { 1 / 2 + p } \right) .\tag{G.12}
$$

Thus the correction term is no larger in K than the selected-width term.

## F.3 Hard Front End

For the selected dual level $Y _ { k }$ , define the Lagrangian front end

$$
\mathfrak { F } _ { K } : = \sum _ { k = 1 } ^ { K } \left[ V _ { P ^ { \star } , r } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) + Y _ { k } \left( V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } ( s _ { 1 } ) - V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) \right) \right] .\tag{G.13}
$$

Let $\sigma _ { k } ^ { \mathrm { s m } }$ be the value suboptimality introduced by replacing the exact optimistic planner at the selected dual level with its statewise softmax candidate. Under Assumption C.1,

$$
\sum _ { k = 1 } ^ { K } \sigma _ { k } ^ { \mathrm { s m } } \le \widetilde { O } \left( \frac { K | \cal A | } { ( 1 - \gamma ) ^ { 2 } } \exp ( - \delta _ { \mathrm { m i n } } / \tau _ { h } ) \right) .\tag{G.14}
$$

Theorem F.3 (Decaying hard front end). Assume ${ \mathcal { E } } _ { \mathrm { r e p } } ,$ , the nonbinding bonus cap, andAssumption C.1. With the same $\alpha _ { k } , \lambda _ { k }$ as in the soft analysis and $0 \overset { \cdot } { \leq } Y _ { k } \leq Y _ { \mathrm { m a x } }$

$$
\mathfrak { F } _ { K } \leq \widetilde { O } \left( \frac { \overline { { b } } d ^ { 2 } Y _ { \operatorname* { m a x } } } { ( 1 - \gamma ) ^ { 2 } } \left[ \sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \right] \right) + \sum _ { k = 1 } ^ { K } \sigma _ { k } ^ { \mathrm { s m } } .\tag{G.15}
$$

Proof. At each selected dual level, the optimistic Lagrangian comparison is the same as in Lemma D.10, with the utility-side statistical terms multiplied by $Y _ { k } \ \leq \ Y _ { \operatorname* { m a x } } .$ . The reward-side terms are unchanged. The returned policy is a softmax approximation of the optimistic planner, so its value loss contributes $\sigma _ { k } ^ { \mathrm { s m } }$ . Applying Lemma D.11 to the returned sequence and summing gives (G.15). No certificate width is used in this front-end argument. □

## F.4 Selected-Certificate Reduction

Let $\kappa _ { \mathrm { b r } }$ contain episodes that stop at a certified grid point, and let $\kappa _ { \mathrm { c a p } }$ contain episodes that reach $Y _ { \mathrm { m a x } }$ without certification. Define the certificate grid sensitivity

$$
s _ { \mathrm { g r i d } } : = C _ { \mathrm { g r i d } } \frac { \eta _ { h } } { \tau _ { h } ( 1 - \gamma ) } ,\tag{G.16}
$$

where $C _ { \mathrm { g r i d } }$ is the universal constant from the softmax-sensitivity Lemma C.2. The positive-part clipping is 1-Lipschitz, so neighboring certificates differ by at most $s _ { \mathrm { g r i d } }$

Proposition F.4 (Selected-certificate regret reduction). On $\mathcal { E } _ { \mathrm { r e p } }$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \mathfrak { F } _ { K } + Y _ { \operatorname* { m a x } } \left( \sum _ { k = 1 } ^ { K } w _ { k } ( \pi _ { k } ) + K s _ { \mathrm { g r i d } } \right) ,\tag{G.17}
$$

and

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \sum _ { k \in { \cal K } _ { \mathrm { c a p } } } \Big [ b - V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) \Big ] _ { + } .\tag{G.18}
$$

Proof. If $k \in \mathcal { K } _ { \mathrm { b r } } .$ , then $\underline { { { \mathcal U } } } _ { k } ( \pi _ { k } ) \ge b$ , so Lemma F.1 gives $V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) \geq b$ . Break episodes therefore contribute zero hard violation, proving (G.18).

To convert the Lagrangian front end into reward regret, use $V _ { P ^ { \star } , g } ^ { \pi ^ { \star } } ( s _ { 1 } ) \geq b \colon$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \mathfrak { F } _ { K } + \sum _ { k = 1 } ^ { K } Y _ { k } \left[ V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) - b \right] _ { + } .\tag{G.19}
$$

If a break occurs at $Y _ { k } ~ > ~ 0$ , the preceding grid point failed the certificate. Grid sensitivity and Lemma F.1 therefore imply

$$
\left[ V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) - b \right] _ { + } \leq s _ { \mathrm { g r i d } } + w _ { k } ( \pi _ { k } ) .
$$

When $Y _ { k } = 0$ , the corresponding term in (G.19) is zero. On a cap episode, the certificate at the explicitly queried cap failed, so

$$
\begin{array} { r } { \left[ V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) - b \right] _ { + } \leq w _ { k } ( \pi _ { k } ) . } \end{array}
$$

Using $Y _ { k } \le Y _ { \operatorname* { m a x } }$ proves (G.17).

## F.5 Hard Violation at the Executed Cap Policy

The remaining argument is performed at the cap policy that is actually returned and subsequently enters the probing kernel; it does not use an unexecuted fixed-Y<sub>θ</sub> candidate.

For $c \in \{ r , g \}$ , define the selected-policy optimistic overestimation

$$
\begin{array} { r } { o _ { k , c } : = \Big [ V _ { \widehat { P } _ { k } , c + \widehat { b } _ { k } } ^ { \pi _ { k } } \big ( s _ { 1 } \big ) - V _ { P ^ { \star } , c } ^ { \pi _ { k } } \big ( s _ { 1 } \big ) \Big ] _ { + } . } \end{array}\tag{G.20}
$$

The proof of Lemma D.12 applies to the returned sequence of the hard algorithm and to either normalized signal r or g; hence

$$
\sum _ { k = 1 } ^ { K } o _ { k , r } \leq B _ { K } , \qquad \sum _ { k = 1 } ^ { K } o _ { k , g } \leq B _ { K } .\tag{G.21}
$$

Lemma F.5 (Cap-level Slater pressure). Let π¯ satisfy the Slater condition $V _ { P ^ { \star } , g } ^ { \bar { \pi } } ( s _ { 1 } ) \geq b + \theta$ . For any $k \in \mathcal { K } _ { \mathrm { c a p } } , i f Y _ { \mathrm { m a x } } \theta \geq 1$ , then

$$
\left[ b - V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) \right] _ { + } \leq \frac { o _ { k , r } } { Y _ { \operatorname* { m a x } } } + o _ { k , g } + \left( 1 + \frac { 1 } { Y _ { \operatorname* { m a x } } } \right) \ell _ { k } + \frac { \sigma _ { k } ^ { \mathrm { s m } } } { Y _ { \operatorname* { m a x } } } .\tag{G.22}
$$

Proof. At the cap, softmax planning gives

$$
\begin{array} { r l } & { V _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) + Y _ { \operatorname* { m a x } } V _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi _ { k } } ( s _ { 1 } ) } \\ & { \quad \geq V _ { \widehat { P } _ { k } , r + \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) + Y _ { \operatorname* { m a x } } V _ { \widehat { P } _ { k } , g + \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) - \sigma _ { k } ^ { \mathrm { s m } } . } \end{array}\tag{G.23}
$$

Direct optimism, with ${ \mathfrak { C } } _ { k } ^ { \mathrm { { i n i t } } } ( { \bar { \pi } } ) \leq \Gamma _ { k }$ , yields

$$
V _ { \widehat { P } _ { k } , c + \widehat { b } _ { k } } ^ { \pi } ( s _ { 1 } ) \geq V _ { P ^ { \star } , c } ^ { \pi } ( s _ { 1 } ) - \ell _ { k } \qquad ( c \in \{ r , g \} ) .\tag{G.24}
$$

The definition of $o _ { k , c }$ upper-bounds the left-hand optimistic values in (G.23) by the corresponding true values plus $o _ { k , c } .$ Consequently,

$$
\begin{array} { r l } & { V _ { P ^ { \star } , r } ^ { \pi _ { k } } ( s _ { 1 } ) + Y _ { \operatorname* { m a x } } V _ { P ^ { \star } , g } ^ { \pi _ { k } } ( s _ { 1 } ) + o _ { k , r } + Y _ { \operatorname* { m a x } } o _ { k , g } } \\ & { \quad \geq V _ { P ^ { \star } , r } ^ { \bar { \pi } } ( s _ { 1 } ) + Y _ { \operatorname* { m a x } } V _ { P ^ { \star } , g } ^ { \bar { \pi } } ( s _ { 1 } ) - ( 1 + Y _ { \operatorname* { m a x } } ) \ell _ { k } - \sigma _ { k } ^ { \mathrm { s m } } . } \end{array}\tag{G.25}
$$

If $v _ { k } = [ b - V _ { P ^ { \star } , q } ^ { \pi _ { k } } ( s _ { 1 } ) ] _ { + } > 0 $ , the Slater condition and the normalization $| V _ { P ^ { \star } , r } ^ { \pi _ { k } } - V _ { P ^ { \star } , r } ^ { \bar { \pi } } | \leq 1$ imply

$$
Y _ { \mathrm { m a x } } ( \theta + v _ { k } ) \le 1 + o _ { k , r } + Y _ { \mathrm { m a x } } o _ { k , g } + ( 1 + Y _ { \mathrm { m a x } } ) \ell _ { k } + \sigma _ { k } ^ { \mathrm { s m } } .
$$

Since $Y _ { \mathrm { m a x } } \theta \geq 1$ , subtracting this term and dividing by $Y _ { \mathrm { m a x } }$ proves (G.22). The claim is trivial when $v _ { k } = 0$ □

Proposition F.6 (Decaying Slater-induced hard violation). Assume the hypotheses of Proposition F.4. $I f Y _ { \operatorname* { m a x } } \theta \geq 1$ , then

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \leq \left( 1 + \frac { 1 } { Y _ { \operatorname* { m a x } } } \right) \left( 2 {  { \mathcal { B } } } _ { K } + \sum _ { k = 1 } ^ { K } {  { \ell _ { k } } } + \sum _ { k = 1 } ^ { K } \sigma _ { k } ^ { \mathrm { s m } } \right)\tag{G.26}
$$

up to universal constants. $I f Y _ { \operatorname* { m a x } } \theta < 1$ , the trivial bound Violation ${ \mathfrak { r e t u r n } } ( K ) \leq K \leq \theta ^ { - 4 }$ applies.

Proof. Combine (G.18) with Lemma $\mathrm { F . 5 , }$ sum over cap episodes, and use (G.21). Extending each nonnegative sum from $\kappa _ { \mathrm { c a p } }$ to all episodes gives (G.26). The second statement follows from $Y _ { \mathrm { m a x } } = K ^ { 1 / 4 }$ □

## F.6 Polynomial Schedules

Corollary F.7 (Polynomial hard rates). Let $\epsilon _ { k } = k ^ { - p }$ with $p \in [ 0 , 1 / 2 )$ , and choose

$$
\alpha _ { k } = \widetilde \Theta \Big ( \sqrt { ( \Gamma _ { k } + d ^ { 2 } ) \gamma \log ( | \mathcal { M } | k / \delta ) } \Big ) \quad w i t h \ a \ s u f f c i e n t l y \ l a r g e \ l e a d i n g \ c o n s t a n t ,
$$

$$
\lambda _ { k } = \widetilde \Theta \bigg ( \frac { \Gamma _ { k } + d ^ { 2 } } { d } \log ( | \mathcal { M } | k / \delta ) \bigg ) .
$$

$$
\tau _ { h } = \Theta \left( \frac { 1 } { ( 1 - \gamma ) \sqrt { K } \log | { \cal A } | } \right) , \qquad \eta _ { h } = \Theta \left( \frac { \tau _ { h } ( 1 - \gamma ) } { K ^ { 3 / 4 } } \right) .
$$

Under the assumptions ofTheorem F.3 and Proposition F.6, with probability at least $1 - \delta ,$

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \left( \left[ \frac { | A | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | A | } { 1 - \gamma } } \right] K ^ { 3 / 4 + p } \right) ,\tag{G.27}
$$

and

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde { \cal O } \left( \frac { 1 } { \theta } \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] K ^ { 1 / 2 + p } \right) + { \cal O } ( \theta ^ { - 4 } ) .\tag{G.28}
$$

Proof. For polynomial schedules,

$$
\sqrt { \Gamma _ { K } \sum _ { k = 2 } ^ { K } \Lambda _ { k } } + \sum _ { k = 1 } ^ { K } \frac { \Gamma _ { k } } { \sqrt { k } } \le \widetilde { O } ( | A | K ^ { 1 / 2 + p } ) .
$$

Equations (G.11)–(G.12), Theorem F.3, Proposition F.4, and $Y _ { \mathrm { m a x } } = K ^ { 1 / 4 }$ give (G.27); the grid contribution is

$$
Y _ { \mathrm { m a x } } K s _ { \mathrm { g r i d } } = \widetilde { O } ( K ^ { 1 / 2 } ) .
$$

Proposition F.6 gives (G.28). We display the harmless factor $\theta ^ { - 1 }$ to match the theorem’s problemdependent Slater convention; for normalized utility $\theta \leq 1$ . The exponential softmax residual under the stated temperature is lower order. □

## G The $p = 0$ Hard Specialization

This appendix specializes the derived-certificate proof in Appendix F to uniform tertiary probing. It is included separately to make the clean return-policy rate transparent.

When $p = 0 , \epsilon _ { k } \equiv 1$ , and hence

$$
q _ { k } = q _ { k } ^ { \prime } = q _ { k } ^ { \prime \prime } = U ( \mathcal { A } ) , \qquad \Gamma _ { k } = | \mathcal { A } | , \qquad \Lambda _ { k } = | \mathcal { A } | .
$$

The algorithmic lower correction becomes

$$
\ell _ { k } = C _ { \mathrm { c e r t } } \sqrt { \frac { | \boldsymbol { A } | \ln ( | \boldsymbol { M } | k / \delta ) } { k ( 1 - \gamma ) } } ,\tag{H.1}
$$

and Lemma F.1 gives

$$
\mathcal { U } _ { k } ( \pi ) \leq V _ { P ^ { \star } , g } ^ { \pi } ( s _ { 1 } ) \leq \mathcal { U } _ { k } ( \pi ) + w _ { k } ( \pi )
$$

for every queried policy. The algorithm uses only the first inequality; the second is summed only for the returned sequence.

Lemma G.1 $( p = 0$ selected correction and width). On the representation-learning good event,

$$
\sum _ { k = 1 } ^ { K } \ell _ { k } \leq \widetilde { O } \left( \sqrt { \frac { \vert A \vert K } { 1 - \gamma } } \right) ,
$$

and

$$
\sum _ { k = 1 } ^ { K } w _ { k } ( \pi _ { k } ) \leq \widetilde { O } \left( \left[ \frac { | A | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | A | } { 1 - \gamma } } \right] \sqrt { K } \right) .\tag{H.2}
$$

Proof. The first display follows by summing $\ell _ { k } = \widetilde { O } ( k ^ { - 1 / 2 } )$ . The second follows from Lemma F.2 with $\mathrm { ~ T } _ { k } = \Lambda _ { k } = | \bar { \mathcal { A } } |$ □

Proposition G.2 $( p = 0$ certificate reduction). Under the hypotheses of Proposition F.4,

$$
\mathrm { R e g r e t } ^ { \mathrm { r e t u r n } } ( K ) \leq \widetilde { O } \left( \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] K ^ { 3 / 4 } \right) ,\tag{H.3}
$$

and hard violation is supported only on cap episodes.

Proof. For $p = 0$ , the hard front end in Theorem F.3 is $\widetilde { O } ( \vert \mathcal { A } \vert \bar { b } d ^ { 2 } K ^ { 3 / 4 } / ( 1 - \gamma ) ^ { 2 } )$ . Apply Proposition F.4 and Lemma G.1. With $\eta _ { h } = \Theta ( \tau _ { h } ( 1 - \gamma ) / K ^ { 3 / 4 } )$ , the grid term is

$$
Y _ { \mathrm { m a x } } K s _ { \mathrm { g r i d } } = \widetilde { O } ( K ^ { 1 / 2 } ) ,
$$

and is lower order.

Theorem G.3 (Slater-induced hard violation for $p = 0 )$ . Suppose Assumptions 2.3, 2.5, and C.1 hold, $\vert \mathcal { M } \vert < \infty , P ^ { \star } \in \mathcal { M } ,$ , and the representation-learning good event holds. Then Algorithm 2 with $\epsilon _ { k } \equiv 1$ satisfies

$$
\mathrm { V i o l a t i o n } _ { \mathrm { h a r d } } ^ { \mathrm { r e t u r n } } ( K ) \le \widetilde O \left( \frac { 1 } { \theta } \left[ \frac { | { \cal A } | \bar { b } d ^ { 2 } } { ( 1 - \gamma ) ^ { 2 } } + \sqrt { \frac { | { \cal A } | } { 1 - \gamma } } \right] \sqrt { K } \right) + O ( \theta ^ { - 4 } ) .\tag{H.4}
$$

Proof. Break episodes are feasible by the one-sided inequality in Lemma F.1. At a cap episode, the returned policy is the actually queried $Y _ { \mathrm { m a x } }$ policy, so Lemma F.5 applies and its overestimation terms are summable along the returned sequence. Proposition F.6 and Lemma G.1 give (H.4). If $K ^ { 1 / 4 } \theta < 1$ , the trivial $K \leq \theta ^ { - 4 }$ bound accounts for the last term. □

## H Experiments

## H.1 Job scheduling.

To validate the convergence of REP-PD-TEG in CMDPs, we first evaluate it in a simulated job scheduling environment. This toy domain lets us visualize how the learned representation, optimistic bonus, and primal-dual update interact under unknown dynamics.

Following [17, 18], we design a constrained MDP with state space $\mathcal { S } = \{ 0 , 1 , \ldots , 9 \}$ , where s represents the number of pending jobs. The agent chooses either to withhold jobs $( a = 0 )$ or submit them $( a = 1 )$ over a horizon $H = 1 0$ . The transition dynamics follow:

$$
s _ { h + 1 } = { \left\{ \begin{array} { l l } { \operatorname* { m a x } \{ s _ { h } - 2 a _ { h } , 0 \} } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 0 . 8 } \\ { \operatorname* { m a x } \{ s _ { h } - a _ { h } , 0 \} } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } } 0 . 1 } \\ { s _ { h } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

We assume that at time steps from 3 to 6, the reward is 1−0.9a, while at all other time steps the reward is $1 - 0 . 2 a$ . This mimics a system in which job processing becomes temporarily expensive. The utility function $\begin{array} { r } { g ( s , a , s ^ { \prime } ) = \frac { s - s ^ { \prime } } { 2 } } \end{array}$ quantifies job-processing efficiency, with per-episode constraint $\begin{array} { r } { \sum _ { h = 0 } ^ { H - 1 } g ( s _ { h } , a _ { h } , s _ { h + 1 } ) \geq b = \bar { 2 } . } \end{array}$

We do not hand-craft transition features in this domain. Instead, the appendix implementation learns latent transition factors $\hat { \phi } ( s , a )$ and $\hat { \mu } ( s ^ { \prime } )$ directly from the collected transitions using one-hot encodings of the discrete states and actions. After each episode, the newly collected transitions are appended to the dataset, the representation learner is updated on the accumulated data for multiple gradient steps, and the primal-dual policy update is then executed with the induced exploration bonus.

To demonstrate the effectiveness of our constraint-aware design, we compare against REP-UCB [41], an unconstrained low-rank MDP approach using similar representation learning but without dual-variable updates, and a linear method [17] where the transition feature is known. As shown in Figure 1, REP-PD-TEG has higher regret than REP-UCB on this toy instance but substantially lower cumulative constraint violation, illustrating the tradeoff induced by constraint-aware learning. Despite not knowing the transition feature, REP-PD-TEG is comparable in scale to the known-feature linear method of Ghosh et al. [17] on both metrics. Because this illustrative plot reports a single run without uncertainty bands, we do not infer statistical significance or an empirical asymptotic rate from it.

## H.2 Practical Neural Implementation

Neural Safety Gym implementation. To probe scalability beyond the toy domain, we instantiate a practical neural variant, denoted REP-PD-NN for neural REP-PD-TEG. The theoretical algorithm uses a per-round full-batch MLE representation-learning step, which is difficult to scale directly to continuous-control domains. REP-PD-NN therefore replaces this step with a mini-batch SPEDERstyle stochastic representation objective [37], while preserving the primal-dual safe RL structure. The learned state-action representation is used as the input to neural reward and cost critics, and the policy is trained with a Lagrangian-style constrained policy update. We evaluate this variant on Safety Gym tasks [35].

Comnaricon of Cumulative Regret Comnaricon of Constraint Violation   
REP-PD-TEG REP-PD-TEG   
REP-UCB REP-UCB   
80 Ghosh et al. (2022 200 Ghosh et al. (2022)   
20 50   
0   
0 10 20 30 40 50 60 70 80 0 10 20 30 40 50 60 70 80   
Episodes Episodes  
Figure 1: Illustrative single-run comparison of REP-PD-TEG, REP-UCB, and the known-feature linear method of Ghosh et al. [17] on the job-scheduling CMDP.

Algorithm 3 REP-PD-NN   
1: Input: CMDP environment $\mathcal { E } ,$ cost budget $d _ { \mathrm { c o s t } }$ , replay buffer $\mathcal { D } ,$ warm-up steps $W$ , gradient   
steps per update $G ,$ dual stepsize $\rho .$   
2: Parameters: representation $\eta ,$ actor $\theta ,$ reward critics $\{ \psi _ { i } \} _ { i = 1 } ^ { 2 } ,$ , cost critics $\{ \omega _ { i } \} _ { i = 1 } ^ { 2 }$ , multiplier λ.   
3: Initialize target networks $\bar { \eta }  \eta , \psi _ { i }  \psi _ { i } .$ , and $\bar { \omega } _ { i }  \omega _ { i }$   
4: Collect W transitions with a random or stochastic policy and store them in $\mathcal { D } .$   
5: for $t = 1 , \dots , T$ do   
6: Sample $a _ { t } \sim \pi _ { \theta } ( \cdot | s _ { t } )$ , execute $a _ { t }$ in $\mathcal { E } ,$ and store $( s _ { t } , a _ { t } , r _ { t } , c _ { t } , s _ { t + 1 } , m _ { t } )$ in $\mathcal { D } .$   
7: if an episode terminates with cumulative cost $\widehat { J } _ { c }$ then   
8: Update $\lambda  [ \lambda + \rho ( \widehat { J } _ { c } - d _ { \mathrm { c o s t } } ) ] _ { + } .$   
9: end if   
10: for $g = 1 , \ldots , G$ do   
11: Sample a mini-batch B from D.   
12: Update the representation network by minimizing $\mathcal { I } _ { \mathrm { r e p } } ( \eta )$ on $B .$   
13: Encode current pairs with $\phi _ { \eta } ( s , a )$ and target pairs with $\dot { \phi } _ { \bar { \eta } } ( s ^ { \prime } , a ^ { \prime } )$   
14: Update reward critics by minimizing the SAC reward Bellman loss.   
15: Update cost critics by minimizing the cost Bellman loss.   
16: Update the actor by minimizing ${ \dot { \mathcal { I } } } _ { \pi } ( \theta )$ with the current multiplier λ.   
17: Optionally update the entropy temperature α.   
18: Soft-update target networks $\bar { \eta } , \{ \bar { \psi _ { i } } \} _ { i = 1 } ^ { 2 }$ , and $\{ \bar { \omega } _ { i } \} _ { i = 1 } ^ { 2 } .$   
19: end for   
20: end for   
21: Return policy $\pi _ { \theta }$ and representation $\phi _ { \eta } .$

Let $\phi _ { \eta } ( s , a )$ denote the learned state-action representation, and $J _ { \mathrm { r e p } } ( \eta )$ denote the objective of representation learning. The actor is trained with the Lagrangian objective

$$
\mathcal { I } _ { \boldsymbol { \pi } } ( \boldsymbol { \theta } ) = \mathbb { E } \Big [ \alpha \log \pi _ { \boldsymbol { \theta } } ( a | s ) - \operatorname* { m i n } _ { i } Q _ { \psi _ { i } } ^ { r } ( \phi _ { \eta } ( s , a ) ) + \lambda \operatorname* { m i n } _ { i } Q _ { \omega _ { i } } ^ { c } ( \phi _ { \eta } ( s , a ) ) \Big ] ,
$$

where λ is the Lagrange multiplier for the cost constraint.

Raw-state baseline. For the raw-state baseline, we use the same Lagrangian training skeleton but remove the representation update. The critics and actor receive raw state-action pairs (s, a) instead of $\phi _ { \eta } ( s , a )$ . Thus, any performance difference is attributable to the learned representation rather than to a different safe RL backbone.

Table 3 reports the completed positive results after 1M interaction steps and 3 random seeds, using the final deterministic evaluation checkpoint (eval\_return / eval\_cost). Across the three reported tasks, REP-PD-NN substantially improves final evaluation return and also reduces final evaluation cost relative to the raw baseline. These results indicate that a learned representation can improve both task performance and safety on a meaningful subset of continuous-control safe RL tasks.

Table 3: Safety Gym results for the practical neural implementation REP-PD-NN. Metrics are the mean and standard deviation over 3 seeds at the final deterministic evaluation checkpoint after 1M environment steps.
<table><tr><td>Environment</td><td>Raw Ret.</td><td>Raw Cost</td><td></td><td>REP-PD-NN Ret. REP-PD-NN Cost</td></tr><tr><td>PointGoal1</td><td> $- 1 . 3 8 \pm 0 . 5 8$ </td><td> $2 7 . 8 5 \pm 1 8 . 9 9$ </td><td> $\mathbf { - 0 . 1 4 \pm 0 . 0 3 }$ </td><td> $\mathbf { 1 6 . 4 5 \pm 1 2 . 6 9 }$ </td></tr><tr><td>PointPush1</td><td> $- 1 . 4 0 \pm 0 . 6 0$ </td><td> $3 2 . 1 7 \pm 1 4 . 6 6$ </td><td> $\mathbf { - 0 . 1 7 \pm 0 . 1 2 }$ </td><td> ${ \bf 1 2 . 1 3 \pm 6 . 9 5 }$ </td></tr><tr><td>PointPush2</td><td> $- 1 . 4 0 \pm 0 . 7 1$ </td><td> $1 3 . 0 0 \pm 8 . 1 7$ </td><td> $\mathbf { - 0 . 0 3 \pm 0 . 0 1 }$ </td><td> ${ \bf 1 0 . 9 0 \pm 3 . 8 5 }$ </td></tr></table>

Table 4: Approximate compute required to reproduce the reported experiments. Runtime denotes wall-clock time for one run. Total compute is over the reported methods and seeds.
<table><tr><td>Experiment</td><td>Worker</td><td>Steps</td><td>Runs</td><td>Time/run</td><td>Total compute</td></tr><tr><td>Job scheduling</td><td>CPU (&lt; 1GB RAM)</td><td></td><td>1</td><td>&lt; 1 min</td><td>&lt; 1 CPU-hour</td></tr><tr><td>Safety Gym PointGoal1</td><td>1 RTX 4090 (24GB)</td><td> ${ 1 0 } ^ { 6 }$ </td><td>6</td><td>8-20 h</td><td>48–120 GPU-hours</td></tr><tr><td>Safety Gym PointPush1</td><td>1 RTX 4090 (24GB)</td><td> $1 0 ^ { 6 }$ </td><td>6</td><td>8-20 h</td><td>48–120 GPU-hours</td></tr><tr><td>Safety Gym PointPush2</td><td>1 RTX 4090 (24GB)</td><td> ${ 1 0 } ^ { 6 }$ </td><td>6</td><td>8-20 h</td><td>48–120 GPU-hours</td></tr></table>

## H.3 Computational Resources

All experiments were run on a local institutional server equipped with eight NVIDIA RTX 4090 GPUs, each with 24GB of memory, and standard x86\_64 CPUs. The tabular job-scheduling experiment is lightweight and can be reproduced on a CPU-only machine. Each Safety Gym training run uses one GPU; the eight-GPU server was used only to run independent seeds and methods in parallel, and is not required for reproducing any single run. The implementation uses Python with PyTorch, CUDA-enabled GPU execution, MuJoCo, and Safety Gym.

Table 4 summarizes the approximate compute required to reproduce the reported experiments. For Safety Gym, each environment is evaluated with two methods, raw and REP-PD-NN, and three random seeds, resulting in six runs per environment and eighteen final training runs in total.

Thus, the reported Safety Gym results require approximately 144–360 GPU-hours in total. When all eight GPUs are used in parallel, the final runs can be completed in roughly 18–45 wall-clock hours. During method development, we also ran preliminary hyperparameter sweeps and diagnostic experiments that are not required to reproduce the reported tables. These preliminary runs used additional GPU time of the same order as the final reported experiment set; failed or preliminary runs were excluded from the reported results.