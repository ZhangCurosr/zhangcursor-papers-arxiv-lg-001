# Locally-Guided Actor-Critic: Training a Goal-conditioned Actor with a Subgoal-aware Critic

Olivier Serris serris@isir.upmc.fr ISIR Sorbonne Université, CNRS

Stéphane Doncieux Stephane.Doncieux@isir.upmc.fr ISIR Sorbonne Université, CNRS

Olivier Sigaud Olivier.Sigaud@isir.upmc.fr ISIR Sorbonne Université, CNRS

## Abstract

Goal-conditioned reinforcement learning struggles with long horizons when rewards are sparse. While a planner can provide subgoals to guide a low-level policy, its use at test time may introduce practical subgoal management difficulties. An alternative paradigm utilizes a high-level planner to assist learning, while the policy remains conditioned only on the final goal, enabling planner-free deployment. Among these methods, Reinforcement Learning with Imagined Subgoals (RIS) introduces a regularization term that encourages the policy to take the same actions for the final goal as it does for an intermediate goal. This regularization, however, may lead to goal-chaining issues when intermediate goals are low-dimensional. Potential-based reward shaping (PBRS) translates plans into an additional reward while ensuring that the optimal policy remains unchanged. Yet, it can generate deceptive rewards in terminal states. We study these failure cases and first propose an alternative reward shaping method (RS) that removes these deceptive rewards at the expense of theoretical guarantees of PBRS. Similar to this RS variant, we then propose another method named Locally-Guided Actor Critic (LG-AC) that rewards the agent for reaching intermediate goals. Unlike RS, where intermediate rewards are implicit in the shaping signal, we explicitly condition a value estimator on the full sequence of intermediate goals but represent the value function as a sum of subgoal-conditioned value functions, enabling dense hindsight relabeling. We evaluate all these methods in tasks with challenging goal-chaining requirements and empirically highlight specific cases in which either action regularization or reward shaping yield low performance, while LG-AC achieves the best overall performance across tasks.

## 1 Introduction

In Reinforcement Learning (RL), an agent typically learns to master a task by maximizing a reward signal over a sequence of actions. However, standard RL methods often struggle when horizons are long and rewards are sparse. Goal-Conditioned (GC) policies [Schaul et al., 2015] are no exception, frequently failing to reach distant goals without additional guidance. To address this, hierarchical reinforcement learning (HRL) approaches decompose the task into a hierarchy of subtasks. While HRL encompasses a vast range of approaches, most follow three primary families: the option framework [Sutton et al., 1999], where a high-level policy iteratively selects low-level sub-policies to execute, the feudal framework [Dayan and Hinton, 1992] where a high-level policy outputs intermediate goals that a low-level policy must reach iteratively, or graph-based planning [Savinov et al., 2018] where the low-level policy is similarly conditioned on intermediate goals, but these goals are generated via search over a graph of previously visited states. While effective for long-horizon tasks, these approaches typically require the high-level planner to operate during inference to guide the low-level policy which can lead to increased computation cost or subgoal management difficulties [Serris et al., 2026]. An orthogonal line of research only uses the high-level as a guidance for a low-level policy during learning, while the low-level policy operates on its own, conditioned solely on the final goal. With this paradigm, the high-level planner can utilize privileged information, such as ground-truth simulation states or global maps, that is only available in simulation.

Within this category of methods, RIS [Chane-Sane et al., 2021] proposes to constrain the policy to output the same action for a far-away final goal as it would for a goal placed midway between its current location and the final goal. However, Chenu et al. [2025] highlights a general chaining issue that can arise in similar contexts: when intermediate goals are low-dimensional and thus do not fully specify the state, a goal can be reached in a state incompatible with the achievement of the rest of the trajectory. To avoid these problems, other methods use potential-based reward shaping (PBRS) [Ng et al., 1999] to convert high-level plans into rewards. This method preserves the optimal policy of the original MDP, but respecting PBRS constraints can create deceptive rewards in terminal states [Müller and Kudenko, 2025]. Consequently, the agent is either incentivized to terminate in any previously visited terminal state or discouraged from reaching any terminal state, including goals.

In this paper, we evaluate an alternative method based on reward shaping (RS) that rewards reaching each intermediate goal while removing deceptive rewards in terminal states, at the expense of PBRS’s theoretical guarantees. Additionally, we propose a second method inspired by the RS approach but with a key architectural difference. In RS, the agent receives implicit rewards for reaching subgoals through a shaped reward signal. In another approach named Locally-Guided Actor Critic (LG-AC), this structure is made explicit by conditioning a value estimator on the complete sequence of intermediate goals provided by a high-level planner. This structure enables hindsight relabeling for intermediate goals. To maintain a fixed input size regardless of the length of the plan, LG-AC decomposes the value into a sum of per-goal components. This decomposition leads to an estimator conditioned on two distinct goal types simultaneously: the final goal, corresponding to the goal targeted by the policy, and an intermediate goal that provides an auxiliary reward. The actor is optimized using this estimator to reach all intermediate goals, which constitutes the plan for a given final goal.

We present an experimental study comparing methods using the RIS loss, PBRS, RS, and LG-AC. All methods share the same SAC+HER backbone and a handcrafted planner to ensure that each success or failure can be directly attributed to the low-level policy. We evaluate these methods in five goal-conditioned environments where the agent must learn to reach any goal from any state. Our results show that RIS performs well in all environments except GC-Hopper and GC-Walker, where goal chaining issues arise. The PBRS variant struggles when the environment contains terminal states, while the RS variant performs well except in AntMaze. Our LG-AC method achieves the best overall mean and interquartile mean (IQM) across all environments.

## 2 Preliminaries

In this work, we study goal-conditioned reinforcement learning (GCRL) agents that leverage hindsight relabeling and formalize the task of completing sequences of goals within the framework of a Partially Observable Markov Decision Process. This section provides the necessary background for these elements.

## 2.1 Markov Decision Process

We consider the standard RL framework, where an agent learns to make decisions through interaction with an environment, modeled as a Markov Decision Process (MDP) defined by the tuple $\{ S , A , R , P , \rho , \mathcal { T } , \gamma \}$ , where S is the state space, A is the action space. $R : S \times A \times S { \stackrel { . } { \to } } \mathbb { R }$ is the reward function and $P : S \times A \times S \to [ 0 , 1 ]$ is the probability of transitioning to a new state given a state and action. $\tau$ corresponds to the set of terminal states. The discount factor $\gamma \in [ 0 , 1 )$ balances immediate and future rewards. At the beginning of an episode, an initial state is sampled from $\rho .$ At each discrete time step t, the agent chooses an action $a _ { t } \in A .$ , moves to a new state $s _ { t + 1 }$ and receives a reward $r _ { t }$ . The objective is to find a policy that maximizes the discounted cumulative reward:

$$
\underset { a _ { t } \sim \pi ( \cdot | s _ { t } ) } { \mathbb { E } } \left[ \sum _ { { t = 0 } } ^ { \infty } \gamma ^ { t } R ( s _ { t } , a _ { t } , s _ { t + 1 } ) \prod _ { i = 1 } ^ { t } \mathbb { 1 } \left[ s _ { i } \notin \mathcal { T } \right] \right] .
$$

Partially Observable Markov Decision Process: When the agent cannot directly observe the state $s ,$ Partially Observable Markov Decision Processes (POMDPs) extend MDPs by adding an observation space Ω and an observation function $O : S  \Omega$ , where the agent selects actions based solely on observations $o = O ( s )$

## 2.2 Goal-conditioned Reinforcement Learning

Goal-conditioned Reinforcement Learning (GCRL) extends the standard MDP framework to settings where an agent must learn to reach multiple objectives. We formalize this as a GC-MDP defined by the tuple $( \bar { S } _ { g c } , A , R , P , \rho , \mathcal { T } , \gamma )$ , where the state space $S _ { g c } = S \times G$ is the Cartesian product of the original state space $S$ and a goal space G. Within this framework, the agent aims to learn a universal policy $\pi ( a | s , g )$ that generalizes across different goals [Schaul et al., 2015]. To relate states to goals, we define a mapping function $\phi : S  G$ that projects a state into the goal space. In this work, G is a low-dimensional representation (e.g., the $( x , y )$ coordinates of an agent). We consider a sparse reward function for goal reaching, defined as $\bar { R _ { g c } } ( \bar { s } ^ { \prime } , g ) = \mathbb { 1 } [ s ^ { \prime } \in S _ { g } ]$ , where $S _ { g }$ represents the set of states that satisfy goal $g .$ Since this reward only depends on the achieved state ${ \overline { { s ^ { \prime } } } }$ and the goal $^ { g , }$ we simplify notation and write $R _ { g c } ( s ^ { \prime } , g )$ rather than $R _ { g c } ( s , a , s ^ { \prime } , g )$

Actor-Critic Methods: A popular approach for solving GCRL problems is to use actor-critic methods. In these methods, a critic learns a goal-conditioned action-value function $Q ( s , a , g )$ measuring the expected return of taking action a in state s while pursuing goal $g$ and then following the current policy. The actor $\pi ( s , g )$ is updated to maximize the value provided by the critic. Offpolicy algorithms are especially advantageous in GCRL because they let the agent learn from past experiences across different goals. During training, both the actor and the critic are updated using transitions $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } , g , { r _ { t } } } \right)$ sampled from a replay buffer. In our work, we use Soft Actor-Critic (SAC) [Haarnoja et al., 2018] which adds an entropy maximization term to promote exploration.

Hindsight experience replay: GCRL with sparse rewards is challenging: as long as an agent does not reach a given goal, it collects no reward, hindering learning. Hindsight Experience Replay (HER) [Andrychowicz et al., 2017] addresses this difficulty by replacing a failed goal in a trajectory with a goal the agent actually reached. In off-policy algorithms, each transition can be relabeled independently as $( s _ { t } , a _ { t } , s _ { t + 1 } , g ) \to ( s _ { t } , a _ { t } , s _ { t + 1 } , \phi ( s _ { t + k } ) )$ , where $k >$ t is a future time step in the same trajectory. These relabeled transitions are then used in standard temporal-difference updates.

## 3 Related Work

Our approach addresses goal-conditioned Markov decision processes (GC-MDPs) by leveraging high-level plans composed of low-dimensional goals during training. We combine an asymmetric actor-critic formulation with value decomposition to distill planner-based rewards into a low-level GC-policy. We review methods that leverage intermediate goals or plans to learn flat goal-conditioned policies.

Policy Regularization via Subgoals: In RIS [Chane-Sane et al., 2021], a high-level policy generates intermediate goals and the low-level policy is constrained to produce similar actions when targeting a distant goal as when targeting an intermediate goal along the same path. This is implemented by minimizing a KL divergence between the action distributions. The intuition is to re-use knowledge of how to reach nearby goals to make progress toward distant ones. PIG [Kim et al., 2023] follows a similar idea but replaces the KL divergence with a quadratic penalty between actions. Due to this similarity, we only evaluate RIS in our experiments. A similar loss to RIS is also used in SAW [Zhou and Kao, 2025] in an offline RL framework. Instead of learning a high-level policy to propose intermediate goals, it directly samples intermediate states from relevant trajectories in the offline dataset, with each subgoal weighted differently depending on its relevance to reaching the final goal.

In PIG and in our study, the goals are low-dimensional, often corresponding to the location of the agent. In contrast, RIS and SAW directly use states as goals but rewards are typically defined with respect to a low-dimensional goal such as the agent location. As a result, it remains to be shown whether these methods truly learn to reach exact states, or whether they effectively only target the rewarded element of the goal-state. In the latter case, goal chaining issues can arise because a valid action to reach an intermediate goal may differ significantly from a valid action to reach a distant goal along the same trajectory. In contrast, our agent is optimized to reach all the goals of a given plan, thus encouraging actions that are compatible with both immediate and future low-dimensional subgoals.

Plan-Based Reward Shaping: An alternative way to convert plans into learning signals for flat policies is reward shaping, a technique that augments the original reward function with additional rewards to guide learning. A particular instance is potential-based reward shaping (PBRS), in which the additional reward is defined through a difference of potential over states. This formulation comes with a theoretical guarantee that preserves the optimal policy of the initial MDP [Ng et al., 1999]. In Okudo and Yamada [2021], the authors design a PBRS reward defined over a sequence of goals, providing positive rewards when the agent reduces the remaining length of the sequence and penalties when it increases. Their PBRS reward is only defined for a single sequence of goals. We extend this idea to multi-goal RL in our baselines. In our method, we only provide positive rewards when the agent reaches intermediate goals. However, we use a critic explicitly conditioned on intermediate goals to allow relabeling for these goals. In Lo et al. [2024], the authors propose to solve a high-level options SMDP and then use the critic of this high-level SMDP to define a potential-based reward to guide a flat policy. Their approach, however, is primarily applied in tabular and discrete-action RL, whereas we target problems with continuous states and actions.

## 4 Methods

We propose a method for solving goal-reaching tasks. During training, an expert planner guides policy learning by providing a sequence of intermediate goals towards any final goal. At deployment, the learned policy operates without access to the planner and is conditioned only on the final goal. We first formalize a problem where an agent receives rewards for intermediate goals via an asymmetric actor-critic method: the actor only sees the final goal, while the critic has access to the full goal sequence. We then show that the critic’s value estimate for a state can be written as a sum of discounted returns associated with each intermediate goal. Finally, we present an algorithm called Locally-Guided Actor-Critic (LG-AC) that jointly trains the actor and the critic using this additive structure.

## 4.1 Intermediate goals as privileged information

In this section, we formalize an optimization problem where the agent is rewarded for reaching each intermediate goal of a given plan. By formulating the problem as a POMDP, we treat these intermediate goals as part of the hidden state, making them unobservable to the actor, ensuring it can be deployed without a planner. Additionally, to facilitate learning, we use an asymmetric actor-critic formulation [Pinto et al., 2018], in which the critic has access to the intermediate goals as privileged information.

We define a Sequential Goal POMDP (SG-POMDP) as a specific POMDP in which the agent is rewarded for intermediate goals that are part of the hidden state. We note it

$$
M _ { s g } = \{ S _ { s e q } , A , O , \Omega , G , R _ { s u m } , P , \rho , T , \gamma \} ,
$$

where each component inherits from the POMDP definition, Section 2.1. The key distinction lies in the state space $\bar { S } _ { s e q } ,$ , which augments the underlying environment state with the current plan. At the start of the episode the agent starts in state $s _ { 0 }$ with goal $g _ { \pi } \in G$ that stays fixed during the whole episode. The planner computes a sequence of goals

$$
{ \pmb g } _ { 0 } = ( g _ { 1 } , g _ { 2 } , \ldots , g _ { \pi } ) = p l a n ( s _ { 0 } , g _ { \pi } ) ,
$$

which defines a path from the initial state to the final goal. We use bold notation for sequences.

The hidden state of $M _ { s g }$ is defined as $( s _ { t } , \pmb { g } _ { t } , \pmb { g } _ { \pi } ) \in S _ { s e q } = S \times G ^ { + } \times G ,$ , which corresponds to augmenting the observation with a sequence of intermediate goals. $G ^ { + }$ denotes the set of non-empty finite sequences of goals from $\begin{array} { r } { G , G ^ { + } = \bigcup _ { i > 1 } G ^ { i } } \end{array}$ , where $G ^ { i }$ is the set of sequences of length i from G. After taking action $a _ { t }$ , the agent transitions to a new hidden state $( s _ { t + 1 } , \pmb { g } _ { t + 1 } , g _ { \pi } )$ , where $s _ { t + 1 } \sim P ( . | s _ { t } , a _ { t } )$ . The actor only observes $( s _ { t } , g _ { \pi } ) = O ( s _ { t } , \pmb { g } _ { t } , g _ { \pi } )$ , the environment state and the final goal. The intermediate goal sequence $\mathbf { \pmb { g } } _ { t + 1 }$ is updated by removing any goal that has been achieved according to

$$
\pmb { g } _ { t + 1 } = \pmb { g } _ { t } \setminus \{ g \in \pmb { g } _ { t } | s _ { t + 1 } \in S _ { g } \}\tag{1}
$$

where $S _ { g }$ corresponds to the set of states that satisfy goal $g .$ . The reward function is defined as

$$
R _ { s u m } ( s _ { t + 1 } , g _ { t } ) = \sum _ { g _ { r } \in { \pmb g } _ { t } } R _ { g c } ( s _ { t + 1 } , g _ { r } ) .
$$

Thus each goal contributes to rewards only until reached. The objective is to maximize the expected discounted return:

$$
V _ { s g } ( s , \pmb { g } , \pmb { g } _ { \pi } ) = \underset { a _ { t } \sim \pi ( \cdot | s _ { t } , g _ { \pi } ) } { \mathbb { E } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { s u m } ( s _ { t + 1 } , \pmb { g } _ { t } ) \prod _ { i = 1 } ^ { t } \mathbb { 1 } [ s _ { i } \not \in \mathcal { T } ] \bigg | _ { \pmb { g _ { 0 } = g } } ^ { s _ { 0 } = s } \right] ,\tag{2}
$$

where T denotes the set of terminal states and $\pmb { g } _ { t }$ is defined in (1).

Compared to the classical GC-MDP framework, the SG-POMDP framework provides intermediate rewards along a planned path, resulting in denser learning signals. However, this formulation requires the value function to be conditioned on $\mathbf { \pmb { g } } _ { t }$ , a sequence whose length varies as goals are achieved. This dependency introduces two related challenges: the critic must generalize over a combinatorial number of possible goal subsets, and it must operate on variable-size inputs, which significantly increases architectural complexity.

## 4.2 Value Decomposition over individual goals

To address the two challenges outlined in Section 4.1, we propose decomposing the critic into a sum of goal-specific values. Specifically, we aim to replace a critic function that takes a sequence of intermediate goals as input with a parameterized estimator that takes a single intermediate goal as input and returns the corresponding goal-related return. We can then sum the outputs of this estimator over the full sequence of goals to recover the original return estimate. As established in prior work on reward decomposition [Juozapaitis et al., 2019], such a summation remains valid in off-policy settings, and is compatible with temporal difference learning, ensuring the global value function is accurately preserved. This decomposition allows us to learn a parameterized estimator with fixed-size inputs, enabling the use of simple multi-layer neural networks, while still handling sequences of arbitrary length.

Because rewards are additive and goals can be reached independently of one another, the total discounted return for reaching all goals of the sequence admits an exact decomposition into a sum of per-goal contributions. Each of these contributions is defined as $V _ { s g a } ( s , g _ { r } , g _ { \pi } )$ , the expected discounted return associated with the rewards for achieving $g _ { r }$ under a policy conditioned on $g _ { \pi } .$

$$
V _ { s g a } ( s , g _ { r } , g _ { \pi } ) \triangleq \mathbb { E } _ { a \sim \pi ( \cdot | s , g _ { \pi } ) } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { g c } ( s _ { t + 1 } , g _ { r } ) \prod _ { i = 1 } ^ { t } \mathbb { 1 } [ s _ { i } \notin ( S _ { g _ { r } } \cup \mathcal { T } ) ] \Bigg | s _ { 0 } = s \right] .\tag{3}
$$

The indicator term ensures that rewards are accumulated only until the goal $g _ { r }$ is reached or a terminal state is reached. Under this definition, the objective of reaching all the goals of a sequence g can be expressed as the sum of the individual goal values:

$$
V _ { s g } ( s , g , g _ { \pi } ) = \sum _ { g _ { r } \in g } V _ { s g a } ( s , g _ { r } , g _ { \pi } ) .\tag{4}
$$

We provide a proof of (4) in Appendix A.

As $V _ { s g a }$ is only conditioned on a fixed-size input $( s , g _ { r } , g _ { \pi } )$ , we can use a fixed-size neural network architecture reducing the space over which the agent is required to generalize. Crucially, this decomposition does not introduce additional learning complexity. $V _ { s g a }$ corresponds to an expected discounted return with terminal conditions. As a consequence, $V _ { s g a }$ can be learned using standard temporal-difference learning methods. We name this value function subgoal-aware critic (SGAC).

## 4.3 Locally-Guided Actor Critic

Our method, Locally-Guided Actor Critic, leverages SGAC to optimize the policy not only for the final goal but also for a sequence of intermediate goals leading to it. This is particularly beneficial during the early stages of training, when the critic is expected to provide more accurate estimates for goals that are closer to the current state. More generally, the reliability of the critic decreases for goals that are farther from the current state, which can create difficulties even at later stages of training. From the action-gap [Farahmand, 2011] or the signal-to-noise [Park et al., 2023] perspective, optimization becomes challenging when value differences are small relative to estimator noise, as happens for discounted rewards far in the future. By including intermediate goals, the actor receives more immediate learning signals that mitigate this difficulty.

For a given state-goal pair $( s , g _ { \pi } )$ , we rely on a planner to compute a sequence of intermediate goals $\pmb { g } = \mathtt { p l a n } ( s , g _ { \pi } )$ , which is used during training only. The policy is then optimized to maximize the sum of critic values associated with each goal in the sequence:

$$
\theta = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { a \sim \pi _ { \theta } ( . | s , g _ { \pi } ) } \left[ \sum _ { g _ { r } \in { \pmb { g } } = p l a n ( s , g _ { \pi } ) } Q _ { s g a } ^ { \pi _ { \theta } } ( s , g _ { r } , g _ { \pi } , a ) \right] .\tag{5}
$$

By linearity of expectation, the inner sum of Q-values is a sum of expectations, each defining the state-value function $V _ { s g a } ^ { \pi _ { \theta } } ( s , g _ { r } , g _ { \pi } ) = \mathbb { E } _ { a \sim \pi _ { \theta } } [ Q _ { s g a } ^ { \pi _ { \theta } } ( s , g _ { r } , g _ { \pi } , a ) ]$ . This yields the equivalent objective:

$$
\theta = \arg \operatorname* { m a x } _ { \theta } \sum _ { g _ { r } \in g } V _ { s g a } ^ { \pi _ { \theta } } ( s , g _ { r } , g _ { \pi } ) .
$$

As shown in (4), this objective is identical to the value function defined under the SG-POMDP.

Hindsight Relabeling As is common in sparse-reward GCRL, we employ hindsight experience replay to densify training signals. Transitions of the form $\left( { { s _ { t } } , { a _ { t } } , { g _ { \pi } } , { s _ { t + 1 } } } \right)$ are relabeled as $( s _ { t } , a _ { t } , \phi ( s _ { t + k } ) , s _ { t + 1 } )$ where $s _ { t + k }$ with $k > 0$ is a future state of the trajectory.

In the case of SGAC, training requires specifying not only the goal $g _ { \pi }$ conditioning the policy but also $g _ { r }$ . Since the sole purpose of subgoal-aware critic parametrized by ψ is to guide the policy, we design the relabeling strategy to ensure that all critic values appearing in the actor loss are accurately learned. To this end, we train $Q _ { s g a } ^ { \psi }$ on all intermediate goals along the planned sequence from the current state to the target goal, as these are precisely the values appearing in the actor objective. In addition, we also train $Q _ { s g a } ^ { \bar { \psi } }$ on the currently achieved goal $\phi ( s )$ to densify rewards. Formally, for each sample $( s , a , s ^ { \prime } , g _ { \pi } )$ of a mini batch, we define the loss as

$$
l ( s , a , s ^ { \prime } , g _ { \pi } , \psi ) = \sum _ { g _ { r } \in p l a n ( s , g _ { \pi } ) \cup \{ \phi ( s ) \} } \left\| Q _ { s g a } ^ { \psi } ( s , g _ { r } , g _ { \pi } , a ) - y _ { \theta , \psi } ( s ^ { \prime } , g _ { r } , g _ { \pi } ) \right\| ^ { 2 } ,\tag{6}
$$

where the target is given by

$$
\begin{array} { r } { y _ { \theta , \psi } ( s ^ { \prime } , g _ { r } , g _ { \pi } ) = R _ { g c } ( s ^ { \prime } , g _ { r } ) + \gamma \mathbb { 1 } [ s ^ { \prime } \notin \mathcal { T } \cup S _ { g _ { r } } ] Q _ { s g a } ^ { \psi } ( s ^ { \prime } , g _ { r } , g _ { \pi } , \pi _ { \theta } ( . | s ^ { \prime } , g _ { \pi } ) ) . } \end{array}
$$

Locally-Guided Actor Critic uses SAC to learn a subgoal-aware critic by estimating the value defined in (3), applies the relabeling strategy (6), and trains an actor by maximizing the objective in (5).

## 5 Experiments

We empirically evaluate LG-AC and several baselines in goal-conditioned environments. We test whether each method can reach near and far goals, handle goal-chaining, and avoid terminal states that lead to failure. To learn general policies, agents are trained by sampling goals uniformly across the entire goal space. Since testing all possible goal and start combinations is too costly, we evaluate on a fixed set of goals with increasing difficulty.

## 5.1 Baselines

We evaluate LG-AC against the following baselines.

SAC+HER: An agent that combines Soft Actor-Critic (SAC) [Haarnoja et al., 2018] with Hindsight Experience Replay (HER) [Andrychowicz et al., 2017]. The agent is trained to reach the final goal using off-policy actor-critic learning with goal relabeling, without exploiting any planning information. All other baselines retain the same SAC+HER backbone but introduce auxiliary objectives that leverage intermediate goals.

RS: Unlike the SAC+HER baseline, RS leverages the planner by incorporating a Reward Shaping (RS) bonus. Reward shaping is a well-established technique that provides additional rewards to guide learning [Mataric, 1994], and we tailor it here specifically for sequences of goal-following tasks. Intuitively, the agent is rewarded for progressing toward the final goal and penalized for moving away from it. Progress is formally measured using the plan cost, defined as the number of intermediate goals. Let $p l a n ( s , g )$ denote the sequence of subgoals generated by the planner from state s to the final goal $^ { g , }$ and define the plan cost as the number of subgoals in this sequence, $C ( s , g ) = | p l a n ( s , g ) |$ |. The shaping reward is then given by

$$
R _ { \mathrm { r s } } ( s , a , s ^ { \prime } , g ) = R _ { g c } ( s , a , g ) + C ( s , g ) - C ( s ^ { \prime } , g ) .\tag{7}
$$

In practice, this means that the agent receives a reward bonus of $+ N$ if the plan length decreases by N and −N if it increases by $N$

PBRS: This baseline incorporates Potential-Based Reward Shaping (PBRS) [Ng et al., 1999], which provides a theoretical guarantee that the auxiliary shaping signal does not alter the optimal policy of the original task. To achieve this, a potential function $\Phi ( s , g )$ is defined over states. Here, we set the potential as minus the remaining plan cost, $\Phi ( s , g ) = \dot { - } C \bar { ( } s , g )$ . Intuitively, states closer to the final goal (with a smaller plan cost) have higher potential. The augmented reward function is then defined as

$$
\begin{array} { r l } & { R _ { p b r s } ( s , a , s ^ { \prime } , g ) = R _ { g c } ( s , a , g ) + \gamma \Phi ( s ^ { \prime } , g ) - \Phi ( s , g ) } \\ & { \qquad = R _ { g c } ( s , a , g ) + \left\{ \begin{array} { l l } { C ( s , g ) } & { \mathrm { i f ~ } s ^ { \prime } \mathrm { ~ i s ~ t e r m i n a l , } } \\ { C ( s , g ) - \gamma C ( s ^ { \prime } , g ) } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{8}
$$

Compared to the RS bonus defined in (7), the shaping term in (8) differs in two aspects required for policy invariance. First, the immediate reward signal for progress is $C ( s , g ) - \gamma C ( s ^ { \prime } , g )$ rather than the undiscounted difference $C ( s , g ) - C ( s ^ { \prime } , g )$ used by RS. Second, PBRS requires that the potential of terminal states is zero, $\Phi ( s , g ) = 0$ for all $s \in \tau$ [Grzes, 2017]. This ensures the total shaping´ contribution over any complete trajectory sums to zero, preserving the optimal policy of the original task, which is not guaranteed by the RS baseline.

RIS: This baseline incorporates ideas from "Reinforcement learning with Imagined Subgoals" (RIS) [Chane-Sane et al., 2021]. The core concept of the learning process is to guide the policy towards a final goal g by constraining it to stay close to the behavior needed to reach an easier, intermediate goal $g _ { \mathrm { m i d } } .$ This is achieved by adding a Kullback-Leibler (KL) divergence penalty to the standard actor update. The policy improvement step is defined as

$$
\pi _ { \theta _ { k + 1 } } = \arg \operatorname* { m a x } _ { \theta } \mathbb { E } _ { ( s , g ) \sim D , a \sim \pi _ { \theta } } \left[ Q ^ { \pi } ( s , a , g ) - \alpha D _ { \mathrm { K L } } \big ( \pi _ { \theta } ( \cdot | s , g ) \big | \big | \pi _ { k } ^ { \mathrm { p r i o r } } ( \cdot | s , g _ { \mathrm { m i d } } ) \big ) \right] .
$$

Here, $\pi ^ { \mathrm { p r i o r } } ( \cdot | s , g _ { \mathrm { m i d } } )$ represents the distribution of actions that would lead from the current state s to the imagined intermediate goal $g _ { \mathrm { m i d } }$ . Unlike the original RIS algorithm which learns a high-level policy to predict $g _ { \mathrm { m i d } }$ , our implementation uses the same expert planner as the other baselines. For a given state s and a final goal g, we query the planner for the full subgoal sequence and set the imagined goal $g _ { \mathrm { m i d } }$ to be the middle goal in that sequence.

## 5.2 Environments

We evaluate on five continuous control tasks. Dubins Hallway and SnakeMaze5 are 2D navigation tasks where a car or a ball must reach a goal location. AntMaze requires a quadruped to navigate to a goal location in a maze. GC-Hopper and GC-Walker are locomotion tasks in which the agent must reach a target x-position, posing a goal-chaining challenge as the optimal behaviors for reaching near and far goals differ. In all environments, a handcrafted planner provides intermediate goals. See Appendix C for complete details.

![](images/d52db8ab4acf24b6060521d9aafeab69aa2ed30195c775fc4e4ec325893d5dd1.jpg)  
Figure 1: (a) The five goal-conditioned environments and the corresponding evaluation goal sets. (b) Mean success rate on the evaluation goal set during training. Each evaluation is repeated 10 times, with results reported as the mean and 95% confidence interval (computed with bootstrapping) over 30 seeds. (c) Final performance per individual evaluation goal for each method, averaged over the last five evaluations. In GC-Hopper and GC-Walker, RS achieves the best performance with LG-AC following closely behind, whereas in AntMaze, RIS and LG-AC perform best. In Dubins Hallway, PBRS has the worst performance, often failing to reach far-away goals.

## 5.3 Empirical evaluation

We evaluate our method on the environments defined in Section 5.2. Performance is measured by the success rate on individual goals of increasing difficulty and the mean success rate during training.

SAC+HER performs reliably only when the target goal is temporally close to the starting configuration. For example, in Dubins Hallway the farthest goal can be reached in approximately 40 time steps. As shown in Figure 1, SAC+HER achieves a high success rate across all goals in this setting. In the other environments, the farthest goals are more than 100 time steps away, and SAC+HER fails to reach these distant goals, despite often succeeding on nearby ones.

In contrast, PBRS performs poorly across most environments, which we attribute to the formulation of the shaping bonus. First, the inclusion of the discount factor γ creates a persistent positive reward even when the agent does not progress. If the plan length $C ( s , g )$ remains unchanged between a state s and its successor $s ^ { \prime } ,$ the immediate shaping reward bonus $C ( s , g ) - \gamma C ( s ^ { \overline { { \prime } } } , g )$ simplifies to $C ( s , g ) ( 1 - \gamma )$ . This persistent positive signal appears sufficient to perturb the agent. In our experiments, PBRS underperforms relative to SAC, while the RS baseline, which omits the γ factor, outperforms both. This is consistent with previous findings that removing the discount factor from the PB reward can improve empirical performance [Jeon et al., 2023]. A second issue arises from terminal conditions in AntMaze, GC-Hopper, and GC-Walker, where falling triggers a terminal state. To maintain theoretical policy invariance, the potential of a terminal state must be zero. Consequently, upon termination, the agent receives a reward equivalent to C(s, g), the total number of remaining goals in the plan. This creates a deceptive reward signal that rewards early failure. This signal severely hinders the learning process and leads to the catastrophic failures observed for PBRS in these locomotion tasks. By addressing these issues, the RS baseline achieves success rates near 100% in SnakeMaze5 and Dubins Hallway. It also reaches the highest success rates in GC-Hopper and GC-Walker, closely followed by LG-AC. In AntMaze, while RS outperforms SAC, it achieves lower success rates than both RIS and LG-AC.

![](images/f1f15a310934c6219163700883aa46a09c0d62fb6b5a1a61c9439a09a0a2c0d3.jpg)  
Figure 2: Aggregated median, interquartile mean (IQM), and mean performance across environments, computed from the last five final evaluations of each run shown in Figure 1. Confidence intervals are estimated using percentile bootstrap with stratified sampling [Agarwal et al., 2021] from 30 seeds. While RS, RIS, and LG-AC achieve similar median performance, LG-AC outperforms the baselines on all other metrics.

![](images/1159726839acabcff8db39ff233c27a5d971348c379a596f35db2566d90fc43a.jpg)  
Figure 3: GC-Hopper trajectories for RIS, LG-AC, and RS agents after training. RIS fails due to a myopic jump, whereas LG-AC and RS complete the task successfully.

RIS performs well in Dubins Hallway, SnakeMaze5, and AntMaze, but its performance drops significantly in GC-Hopper and GC-Walker. This failure is likely due to the main assumption in RIS when applied to low-dimensional goals, that the best action to reach an intermediate goal is also the correct action for reaching a far-away goal. In locomotion tasks like GC-Hopper, the best action to reach a goal in the middle of the trajectory may lead to myopic behavior, as illustrated in Figure 3: the agent jumps toward the final goal and subsequently fails to recover balance. In these cases, the RIS auxiliary loss provides deceptive guidance.

The LG-AC method achieves success rates comparable to the best-performing method in each environment and avoids the failures seen in all other baselines. As shown in Figure 2, aggregate metrics over all environments show that while RIS, RS, and LG-AC yield similar median performance, LG-AC demonstrates statistically significant improvements in the mean, interquartile mean (IQM), and optimality gap as indicated by the non-overlapping confidence intervals.

## 6 Conclusion

This study investigated the problem of learning general goal-conditioned policies through formulations that distill planning guidance into a policy conditioned only on the final goal, enabling execution without a planner at deployment time.

We proposed LG-AC, an asymmetric actor-critic method in which the actor is rewarded for completing a plan through a decomposition of the critic per-goal. With this formulation, the actor receives local guidance provided by the critic for each individual goal. We evaluated this approach on five goalconditioned benchmark environments with varying levels of goal-chaining difficulty and compared it to RIS, PBRS, and RS.

Our empirical results indicate that RIS performs efficiently in most environments, but exhibits failure cases in GC-Hopper and GC-Walker, where the optimal action for reaching an intermediate goal may differ from valid actions for reaching the final goal. We also empirically confirm theoretical limitations previously identified for PBRS. The proposed RS variant, based on PBRS and combined with HER, achieves strong results across environments, with the exception of AntMaze. Overall, our method achieves the best average IQM and mean performance across the tested environments.

A limitation of our approach is the increased computational cost, which scales with the number of intermediate goals used to train both the actor and the critic.

Several directions remain for future work. Since our formulation allows the actor to optimize each future goal independently, one could prioritize learning on goals where the critic’s predictions are trustworthy. Critic outputs may be unreliable because they correspond to insufficiently explored goals, or because the reward signal is too temporally distant to provide accurate learning feedback.

We could therefore weight the critic’s contributions for each future goal according to a reliability measure, for example using ensemble disagreement. Furthermore, while the use of an expert planner enables a controlled evaluation in which successes and failures can be clearly attributed to each low-level method, it would be of interest to study how LG-AC behaves when the planner is learned jointly with the policy, and how robust it is to imperfect or evolving plans during training.

## References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Proceedings of the 35th International Conference on Neural Information Processing Systems, NIPS ’21, Red Hook, NY, USA, 2021. Curran Associates Inc. ISBN 9781713845393.

Marcin Andrychowicz, Filip Wolski, Alex Ray, Jonas Schneider, Rachel Fong, Peter Welinder, Bob McGrew, Josh Tobin, OpenAI Pieter Abbeel, and Wojciech Zaremba. Hindsight experience replay. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf.

Elliot Chane-Sane, Cordelia Schmid, and Ivan Laptev. Goal-Conditioned Reinforcement Learning with Imagined Subgoals. In Proceedings of the 38th International Conference on Machine Learning, pages 1430–1440. PMLR, July 2021. URL https://proceedings.mlr.press/ v139/chane-sane21a.html. ISSN: 2640-3498.

Alexandre Chenu, Olivier Serris, Olivier Sigaud, and Nicolas Perrin-Gilbert. Leveraging sequentiality in reinforcement learning from a single demonstration. In 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids), pages 405–412, 2025. doi: 10.1109/Humanoids65713.2025.11203179.

Peter Dayan and Geoffrey E. Hinton. Feudal reinforcement learning. In Proceedings of the 6th International Conference on Neural Information Processing Systems, NIPS’92, page 271–278, San Francisco, CA, USA, 1992. Morgan Kaufmann Publishers Inc. ISBN 1558602747.

Rodrigo de Lazcano, Kallinteris Andreas, Jun Jet Tai, Seungjae Ryan Lee, and Jordan Terry. Gymnasium robotics, 2024. URL http://github.com/Farama-Foundation/ Gymnasium-Robotics.

Amir-massoud Farahmand. Action-gap phenomenon in reinforcement learning. In J. Shawe-Taylor, R. Zemel, P. Bartlett, F. Pereira, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 24. Curran Associates, Inc., 2011. URL https://proceedings.neurips.cc/paper\_files/paper/2011/file/ 013d407166ec4fa56eb1e1f8cbe183b9-Paper.pdf.

Marek Grzes. Reward shaping in episodic reinforcement learning. In´ Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems, AAMAS ’17, page 565–573, Richland, SC, 2017. International Foundation for Autonomous Agents and Multiagent Systems.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. In Proceedings of the 35th International Conference on Machine Learning, pages 1861–1870. PMLR, July 2018. URL https://proceedings.mlr.press/v80/haarnoja18b.html. ISSN: 2640-3498.

Se Hwan Jeon, Steve Heim, Charles Khazoom, and Sangbae Kim. Benchmarking potential based rewards for learning humanoid locomotion. In IEEE International Conference on Robotics and Automation, ICRA 2023, London, UK, May 29 - June 2, 2023, pages 9204–9210. IEEE, 2023. doi: 10.1109/ICRA48891.2023.10160885. URL https://doi.org/10.1109/ICRA48891.2023. 10160885.

Z. Juozapaitis, A. Koul, A. Fern, M. Erwig, and F. Doshi-Velez. Explainable reinforcement learning via reward decomposition. In Proceedings at the International Joint Conference on Artificial Intelligence. A Workshop on Explainable Artificial Intelligence., 2019.

Junsu Kim, Younggyo Seo, Sungsoo Ahn, Kyunghwan Son, and Jinwoo Shin. Imitating graph-based planning with goal-conditioned policies. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=6lUEy1J5R7p.

Chunlok Lo, Kevin Roice, Parham Mohammad Panahi, Scott M. Jordan, Adam White, Gabor Mihucz, Farzane Aminmansour, and Martha White. Goal-space planning with subgoal models. Journal of Machine Learning Research, 25(330):1–57, 2024. URL http://jmlr.org/papers/v25/ 24-0040.html.

Maja J. Mataric. Reward functions for accelerated learning. In Proceedings of the Eleventh International Conference on International Conference on Machine Learning, ICML’94, page 181–189, San Francisco, CA, USA, 1994. Morgan Kaufmann Publishers Inc. ISBN 1558603352.

Henrik Müller and Daniel Kudenko. Improving the effectiveness of potential-based reward shaping in reinforcement learning. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’25, page 2684–2686, Richland, SC, 2025. International Foundation for Autonomous Agents and Multiagent Systems. ISBN 9798400714269.

Andrew Y. Ng, Daishi Harada, and Stuart J. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning, ICML ’99, page 278–287, San Francisco, CA, USA, 1999. Morgan Kaufmann Publishers Inc. ISBN 1558606122.

Takato Okudo and Seiji Yamada. Subgoal-based reward shaping to improve efficiency in reinforcement learning. IEEE Access, PP:1–1, 06 2021. doi: 10.1109/ACCESS.2021.3090364.

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, and Sergey Levine. HIQL: Offline goal-conditioned rl with latent states as actions. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 34866–34891. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf.

Lerrel Pinto, Marcin Andrychowicz, Peter Welinder, Wojciech Zaremba, and Pieter Abbeel. Asymmetric actor critic for image-based robot learning. In Robotics: Science and Systems XIV. Robotics: Science and Systems Foundation, 06 2018. doi: 10.15607/RSS.2018.XIV.008.

Nikolay Savinov, Alexey Dosovitskiy, and Vladlen Koltun. Semi-parametric topological memory for navigation. In International Conference on Learning Representations, 05 2018.

Tom Schaul, Daniel Horgan, Karol Gregor, and David Silver. Universal value function approximators. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings ofMachine Learning Research, pages 1312– 1320, Lille, France, 07–09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ schaul15.html.

Olivier Serris, Stéphane Doncieux, and Olivier Sigaud. A tale of two goals: leveraging short term goals performs best in multi-goal scenarios. Transactions in Machine Learning Research (TMLR), 2026. URL https://openreview.net/pdf?id=qsUeLwbErp.

Richard S. Sutton, Doina Precup, and Satinder Singh. Between mdps and semi-mdps: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112(1):181–211, 1999. ISSN 0004-3702. doi: https://doi.org/10.1016/S0004-3702(99)00052-1. URL https://www. sciencedirect.com/science/article/pii/S0004370299000521.

Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goulão, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A standard interface for reinforcement learning environments. arXiv preprint arXiv:2407.17032, 2024.

John Luoyu Zhou and Jonathan C. Kao. Flattening hierarchies with policy bootstrapping. In Workshop on Reinforcement Learning Beyond Rewards @ Reinforcement Learning Conference 2025, 2025. URL https://openreview.net/forum?id=iDxTYJB0FP.

## A Proof of Value Decomposition

In this section, we provide a proof for Equation (4), which relates the value function $V _ { s g }$ to a sum of per-goal contributions. Similarly to Juozapaitis et al. [2019], we apply decomposition across rewards for intermediate goals, but also with different terminal conditions, to ensure each intermediate goal only participates to rewards only until reached. First, we establish a lemma that relates $g \in { \mathfrak { g } } _ { t }$ to a product of indicator functions encoding that $g$ has not been reached previously. This product can then be used to stop rewards from accumulating in a GC value objective once $g$ is achieved.

Lemma 1: For any $g \in \pmb { g } _ { 0 }$ and any $t \geq 0 ,$

$$
\mathbb { 1 } \left[ { \boldsymbol g } \in { \boldsymbol g } _ { t } \right] = \prod _ { i = 1 } ^ { t } \mathbb { 1 } \left[ s _ { i } \notin S _ { g } \right] , \quad \mathrm { w h e r e ~ t h e ~ e m p t y ~ p r o d u c t } \left( t = 0 \right) \mathrm { i s ~ d e f i n e d ~ a s ~ 1 . }
$$

Proof: By induction on $t .$ For $t = 0 , g \in g _ { 0 } .$ , so the left side is 1 and the right is the empty product, which is 1. Assume the statement holds for $t - 1$ . From the update rule $( \mathrm { E q . } 1 )$

$$
g \in { \pmb { g } } _ { t } \iff ( g \in { \pmb { g } } _ { t - 1 } ) \land ( s _ { t } \notin S _ { g } ) .
$$

Using the induction hypothesis,

$$
\mathbb { 1 } [ g \in \pmb { g } _ { t } ] = \mathbb { 1 } [ g \in \pmb { g } _ { t - 1 } ] \cdot \mathbb { 1 } [ s _ { t } \notin S _ { g } ] = \Big ( \prod _ { i = 1 } ^ { t - 1 } \mathbb { 1 } [ s _ { i } \notin S _ { g } ] \Big ) \cdot \mathbb { 1 } [ s _ { t } \notin S _ { g } ] = \prod _ { i = 1 } ^ { t } \mathbb { 1 } [ s _ { i } \notin S _ { g } ] . \quad \mathbb { D } [ g \in \pmb { g } _ { t } ] = \pmb { \operatorname { N } } [ g \in \pmb { \operatorname { N } } _ { i = 1 } ] .
$$

With this lemma, we now derive the desired decomposition.

$$
\begin{array} { r l } { b _ { x } ( \xi , \xi , \xi ) \equiv } & { = \underset { n _ { \xi } = - \infty } { \overset { n _ { \xi } } { \longrightarrow } } [ \underset { ( n _ { \xi } = + \infty ) } { \overset { n _ { \xi } } { \longrightarrow } } [ \underset { n _ { \xi } = + \infty } { \overset { n _ { \xi } } { \longrightarrow } } ( 1 _ { \xi } , \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ] ] \overset { \mathrm { L } } [ \xi , \xi ^ { - \eta } ] \underset { ( n _ { \xi } = \eta ) } { \overset { n _ { \xi } = - \infty } { \longrightarrow } } ] } \\ & { = \underset { n _ { \xi } = - \infty } { \overset { n _ { \xi } } { \longrightarrow } } [ \underset { ( n _ { \xi } = + \infty ) } { \overset { n _ { \xi } } { \longrightarrow } } [ \underset { ( n _ { \xi } = + \infty ) } { \overset { n _ { \xi } } { \longrightarrow } } ( \underset { n _ { \xi } = + \infty } { \overset { n _ { \xi } } { \longrightarrow } } ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ) ] \underset { ( n _ { \xi } = \eta ) } { \overset { n _ { \xi } = - \infty } { \overset { n _ { \xi } } { \longrightarrow } } } ] } \\ &  \underset  ( n _ { \xi } = \eta ) \underset { n _ { \xi } = + \infty } { \overset { n _ { \xi } } { \longrightarrow } } [ \xi ( \xi ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ( \xi _ { \xi - \xi } ) ] \underset { ( n _ { \xi } = \eta ) } { \overset { n _ { \xi } = \eta } { \longrightarrow } } ] \underset { ( n _ { \xi } = \eta ) } { \overset { n _ { \xi } = \eta } { \longrightarrow } } [ \underset  n _ { \xi } = \eta )  \end{array}
$$

By defining the subgoal-aware critic,

$$
V _ { s g a } ( s , g _ { r } , g _ { \pi } ) \triangleq \mathbb { E } _ { a _ { t } \sim \pi ( \cdot | s _ { t } , g _ { \pi } ) } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } R _ { g c } ( s _ { t + 1 } , g _ { r } ) \prod _ { i = 1 } ^ { t } \mathbb { I } \left[ s _ { i } \notin ( S _ { g _ { r } } \cup \mathcal { T } ) \right] \middle | s _ { 0 } = s \right] .
$$

We therefore obtain the decomposition

$$
V _ { s g } ( s , { \pmb g } , { \pmb g } _ { \pi } ) = \sum _ { { \pmb g } _ { r } \in { \pmb g } } V _ { s g a } ( s , { \pmb g } _ { r } , { \pmb g } _ { \pi } ) . \quad \sqcup
$$

This decomposition shows that the value function for a set of goals is the sum of independent per-goal value functions, each measuring the expected discounted reward from that goal until it is reached or a terminal state is encountered.

## B Relabeling Strategies

![](images/a366adf4a43a2991eecc4570e25faee393f15a7540b1022821dc881d03bb78d8.jpg)  
Figure 4: Mean success rate on the evaluation goal set during training. Each evaluation was repeated 10 times, with results reported as the mean and 95% confidence interval (computed with bootstrapping) over 10 seeds. LG-AC-future leads to low SR, only learning in SnakeMaze5. LG-AC performs equally or better compared to LG-AC-sample.

This section investigates the selection of relabeling techniques for the subgoal-aware critic, denoted as $Q ( s , a , g _ { r } , g _ { \pi } )$ . For $g _ { \pi }$ , the final goal of the episode, we propose to always sample using HER. For $g _ { r }$ , the goal conditioning the reward, we investigate three distinct approaches.

The first strategy, LG-AC-future, randomly selects a goal achieved later in the same trajectory. Another variant, LG-AC-sample, samples $g _ { r }$ from $p l a n ( s , g _ { \pi } )$ so that the critic is trained on the values optimized by the actor. Finally, the $\mathrm { L G } { \cdot } \mathrm { A C }$ strategy trains the critic on all goals within $p l a n ( s , g _ { \pi } )$ , as described in Section 4.3, to reduce the risk of encountering out-of-distribution values during actor optimization.

As shown in Figure 4, LG-AC-future achieves the lowest performance, suggesting that sampling goals from the current plan is critical. Moreover, LG-AC consistently performs similarly to or better than LG-AC-sample, but at a higher computational cost.

## C Environment details

![](images/4e723ee72579b8c880098686294f5ad495dd1ac1f143e02cb7108b016318cd0e.jpg)  
Figure 5: Handcrafted graphs used for planning in Dubins Hallway, SnakeMaze5, and AntMaze. For GC-Hopper and GC-Walker, the planner outputs uniformly spaced points between the agent and the final goal.

Dubins Hallway: A navigation task in which the agent controls a car in a 2D maze. The state $s = \{ x , y , \cos ( \theta ) , \sin ( \theta ) \}$ } includes the position and orientation of the agent. Each goal g is defined as a position $( x , y )$ and is considered reached if the agent is within a ball of radius 0.1 centered at that position. At each step, the agent advances at a fixed speed, and the action specifies the rate of change of its orientation <sup>˙</sup>θ. During a training episode, both the initial position of the agent and the goal are randomly sampled within the maze boundaries. During an evaluation episode, the agent is tested on a fixed start state and multiple goal locations illustrated in Figure 1. If the agent hits a wall, it stays stuck until the episode ends. The state is only terminal when the agent reaches the goal.

SnakeMaze5: The agent controls a ball navigating in a 2D maze [de Lazcano et al., 2024]. The state $s = \{ x , y , \dot { x } , \dot { y } \}$ includes the position and velocity of the agent. Each goal g is defined as an (x, y) position and is considered reached if the agent is within a ball of radius 0.45 centered at that position. The action represents the linear force exerted on the ball in the x and y directions. In de Lazcano et al. [2024], the agent’s velocity was limited to the range [−5, 5] m/s. Within this range, the agent could easily take sharp turns at maximum speed, making the task relatively simple. To increase the difficulty, we extended this limit to [−10, 10] m/s<sup>1</sup>, requiring the agent to anticipate turns and decelerate in advance. During a training episode, both the agent’s initial position and the goal are randomly sampled within the maze boundaries. During an evaluation episode, the agent is tested on multiple goal locations illustrated in Figure 1. The state is only terminal when the agent reaches the goal.

GC-Hopper: The agent controls a two-dimensional, one-legged figure based on de Lazcano et al. [2024]. The state consists of the positions and velocities of the robot’s body parts. The goal is defined as the difference between the agent’s current x-location and a desired location $x _ { g }$ that remains fixed during an episode; notably, the absolute x-position is excluded from the state to promote generalization. Actions represent the torques applied to the hinge joints. An episode terminates either when the goal is reached, defined as $( x - x _ { g } ) < 0 . 1$ , or when the hopper’s posture becomes unhealthy: the height of the hopper falls below 0.7 or the torso angle leaves the range [−0.2, 0.2]. During training, a random goal is sampled in the range [0, 6.5]. During evaluation, the agent is tested on the specific goals [1.5, 3, 4.5, 6]. Hopper poses a goal chaining challenge because the optimal behavior needed to reach nearby goals differs from that needed to reach far ones.

GC-Walker: The agent controls a two-legged version of the Hopper. The state, actions, and goals follow the same setup as GC-Hopper, including the training and evaluation procedures.

AntMaze: The agent controls the ant quadruped from Towers et al. [2024], in a maze, following the implementation in de Lazcano et al. [2024]. The state consists of the positions and velocities of the robot’s body parts. The goal corresponds to an (x, y) location. Actions represent the torques applied to the hinge joints. An episode terminates either when the goal is reached, defined as $( x - x _ { g } ) < 0 . 5$ or when the robot’s posture becomes unhealthy: the height of the ant is not in the closed interval [0.2, 1]. During training, goals are sampled by adding uniform noise in [−0.25, 0.25] to the center of a random grid cell. During evaluation, the agent is tested on the goals shown in Figure 1.

Handcrafted Planner In Dubins Hallway, SnakeMaze5, and AntMaze, the planner computes the shortest path using the expert graph (see Figure 5). In GC-Hopper and GC-Walker, the planner instead sets intermediate goals evenly spaced between the agent and the final goal.

## D Hyperparameters

Table 1: Common hyperparameters. Exceptions are noted in the rightmost column.
<table><tr><td>Hyperparameter</td><td>Common value</td><td>Notes / Exceptions</td></tr><tr><td>Random actions</td><td>50k</td><td>5k for Dubins Hallway and SnakeMaze5</td></tr><tr><td>Critic hidden size</td><td>256</td><td></td></tr><tr><td>Policy hidden size</td><td>256</td><td></td></tr><tr><td>Activation function</td><td>ReLU</td><td></td></tr><tr><td>Discount factor (γ)</td><td>0.99</td><td></td></tr><tr><td>Critic learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Policy learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>Temperature learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td></td></tr><tr><td>RIS-specific α</td><td> $2 ^ { - 8 }$ </td><td>(only for RIS)</td></tr><tr><td>HER relabel strategy</td><td>20% original, 80% future</td><td>AntMaze: 20% original, 40% future, 40% random</td></tr></table>

Table 2: Batch sizes per environment and method. LG-AC uses a fixed batch size of 256 for all environments; the other methods have tuned values.
<table><tr><td>Environment</td><td>LG-AC</td><td>RIS</td><td>RS</td><td>PBRS</td><td>SAC+HER</td></tr><tr><td>Dubins Hallway</td><td>256</td><td>512</td><td>256</td><td>256</td><td>512</td></tr><tr><td> $S n a k e M a z e S$ </td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>GC-Hopper</td><td>256</td><td>512</td><td>512</td><td>256</td><td>256</td></tr><tr><td>GC-Walker</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>AntMaze</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td></tr></table>

The hyper-parameters used for each environment are presented in Table 1 and Table 2. All baselines and our method share the same hyper-parameters, taken from the original SAC paper [Haarnoja et al., 2018], with the following exceptions. The number of random actions taken before learning is scaled to the complexity of the environment. For most environments, we use future HER [Andrychowicz et al., 2017]. For AntMaze, we employ the same relabeling strategy as in RIS [Chane-Sane et al., 2021]: goals are relabeled either with future states from the same trajectory or with random achieved goals from the replay buffer. We observed that this variant is particularly beneficial in AntMaze, while it has little effect in other environments. In addition, the RIS method introduces an extra hyper-parameter α, the procedure for tuning this value is detailed in Section F. Finally, batch sizes are tuned per (baseline, environment) pair, as explained in Section E.

## E Impact of batch size on performance

Table 3: Analysis of the effective batch size for LG-AC. The effective batch size is the total number of targets processed per update; it is calculated by multiplying the base batch size (256) by the average plan length per transition. Statistics (mean ± std) are computed across all runs shown in Figure 1.
<table><tr><td>Environment</td><td>Avg. Goals per Transition</td><td>Effective Batch Size</td></tr><tr><td>Dubins Hallway</td><td> $2 . 4 4 \pm 0 . 0 2$ </td><td> $6 2 4 \pm 6$ </td></tr><tr><td>SnakeMaze5</td><td> $3 . 9 6 \pm 0 . 1 1$ </td><td> $1 , 0 1 3 \pm 2 7$ </td></tr><tr><td>GC-Hopper</td><td> $5 . 1 4 \pm 0 . 2 5$ </td><td> $1 , 3 1 5 \pm 6 4$ </td></tr><tr><td>GC-Walker</td><td> $5 . 0 3 \pm 0 . 4 0$ </td><td> $1 , 2 8 6 \pm 1 0 2$ </td></tr><tr><td>AntMaze</td><td> $3 . 0 1 \pm 0 . 1 1$ </td><td> $7 7 1 \pm 2 7$ </td></tr></table>

We analyze the impact of batch size to ensure a fair comparison between methods. As detailed in Section 4.3, the loss for each transition $( s , a , s ^ { \prime } , g _ { \pi } )$ in SGAC incorporates all intermediate goals. This increases the number of targets processed per update, resulting in a larger effective batch size (see Table 3) and higher computational cost. Consequently, a direct comparison with baselines would be biased if performance gains were simply a byproduct of this increased computation. To control for this factor, we independently tuned the batch size for each baseline (see Table 4). For each method and environment, we selected the smallest batch size that yielded performance statistically equivalent to its best-performing configuration. This protocol ensures a fair comparison while avoiding unnecessary computational overhead.

## F Impact of the KL penalty factor on RIS performance

We evaluate the sensitivity of the RIS baseline to its KL penalty factor α under different reward specifications. In the original work, the authors use a negative reward function $R ( s ^ { \prime } , g ) = - \mathbb { 1 } [ s ^ { \prime } \notin$ g] and set $\alpha = 1$ . In our study, we primarily employ a positive reward function $\boldsymbol { R } ( s ^ { \prime } , g ) = \mathbb { 1 } [ \boldsymbol { \mathscr { s } } ^ { \prime } \in g ]$ As illustrated in Table 5, RIS performance varies smoothly with α, remaining near-optimal within a specific range for both reward types.

In the negative reward setting, we observe that $\alpha = 1$ falls within the near-optimal range, which aligns with the choice made in the original article. However, the range of effective α values differs between the positive and negative reward settings, suggesting that RIS requires specific tuning depending on the reward type. We also note a slight decrease in RIS performance when using negative rewards. In

Table 4: Success rate of all baseline algorithms across all environments under varying batch sizes. Each cell reports the mean and standard deviation over 10 random seeds for a given (algorithm, environment, batch size) configuration. Bold values indicate either (i) the highest mean performance among the tested batch sizes for a given (algorithm, environment) pair, or (ii) results that are not statistically significantly different from the best-performing batch size for that pair.
<table><tr><td>Method</td><td>Environment</td><td>Batch 256</td><td> $\mathbf { B a t c h 5 1 } 2$ </td><td>Batch 1024</td></tr><tr><td rowspan="5">SAC+HER</td><td>Dubins Hallway</td><td> $0 . 8 0 \pm 0 . 1 6$ </td><td> ${ \bf 0 . 9 0 \pm 0 . 1 2 }$   ${ \bf 0 . 3 8 \pm 0 . 0 8 }$ </td><td> $0 . 7 8 \pm 0 . 1 6$   $0 . 3 3 \pm 0 . 1 0$ </td></tr><tr><td>SnakeMaze5 GC-Hopper</td><td> ${ \bf 0 . 3 8 \pm 0 . 0 9 }$   ${ \bf 0 . 1 8 \pm 0 . 1 0 }$ </td><td> ${ \bf 0 . 2 2 \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 1 9 \pm 0 . 1 0 }$ </td></tr><tr><td>GC-Walker</td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 2 \pm 0 . 0 6 }$ </td></tr><tr><td>AntMaze</td><td></td><td></td><td></td></tr><tr><td></td><td> ${ \bf 0 . 3 0 \pm 0 . 0 7 }$ </td><td> ${ \bf 0 . 3 1 } \pm 0 . 0 6$ </td><td> ${ \bf 0 . 3 0 \pm 0 . 0 8 }$ </td></tr><tr><td rowspan="5">PBRS</td><td>Dubins Hallway SnakeMaze5</td><td> ${ \bf 0 . 4 0 \pm 0 . 2 4 }$   ${ \bf 0 . 2 4 \pm 0 . 0 8 }$ </td><td> ${ \bf 0 . 3 4 \pm 0 . 1 3 }$   ${ \bf 0 . 2 3 \pm 0 . 1 1 }$ </td><td> $0 . 2 8 \pm 0 . 1 9$   ${ \bf 0 . 2 1 } \pm 0 . 0 9$ </td></tr><tr><td>GC-Hopper</td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>GC-Walker</td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>AntMaze</td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 0 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>Dubins Hallway</td><td> $0 . 9 7 \pm 0 . 0 7$ </td><td></td><td> $0 . 9 8 \pm 0 . 0 7$ </td></tr><tr><td rowspan="5"></td><td>SnakeMaze5</td><td> ${ \bf 1 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 0 }$   ${ \bf 1 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 0 }$ </td></tr><tr><td>GC-Hopper</td><td> $0 . 3 0 \pm 0 . 1 3$ </td><td> ${ \bf 0 . 3 7 \pm 0 . 1 4 }$ </td><td> ${ \bf 0 . 3 5 \pm 0 . 1 2 }$ </td></tr><tr><td>GC-Walker</td><td> ${ \bf 0 . 2 2 \pm 0 . 1 1 }$ </td><td></td><td></td></tr><tr><td>AntMaze</td><td></td><td> $0 . 1 7 \pm 0 . 1 3$ </td><td> ${ \bf 0 . 2 5 \pm 0 . 1 3 }$ </td></tr><tr><td></td><td> ${ \bf 0 . 7 3 \pm 0 . 0 9 }$ </td><td> $\mathbf { 0 . 7 7 \pm 0 . 0 7 }$ </td><td> $0 . 7 2 \pm 0 . 1 0$ </td></tr><tr><td rowspan="5">RS</td><td>Dubins Hallway</td><td> ${ \bf 0 . 9 0 \pm 0 . 1 3 }$ </td><td> ${ \bf 0 . 8 9 \pm 0 . 1 5 }$ </td><td> ${ \bf 0 . 9 0 \pm 0 . 1 4 }$ </td></tr><tr><td>SnakeMaze5</td><td> ${ \bf 0 . 9 8 \pm 0 . 0 5 }$ </td><td> ${ \bf 0 . 9 7 \pm 0 . 0 6 }$ </td><td> $0 . 9 5 \pm 0 . 0 7$ </td></tr><tr><td>GC-Hopper</td><td> $0 . 6 9 \pm 0 . 1 9$ </td><td> $\mathbf { 0 . 7 5 \pm 0 . 2 1 }$ </td><td></td></tr><tr><td>GC-Walker</td><td> ${ \bf 0 . 8 3 \pm 0 . 1 4 }$ </td><td></td><td> $\mathbf { 0 . 7 7 \pm 0 . 1 6 }$ </td></tr><tr><td>AntMaze</td><td> ${ \bf 0 . 4 9 \pm 0 . 1 9 }$ </td><td> $\mathbf { 0 . 8 3 \pm 0 . 1 6 }$   ${ \bf 0 . 4 2 \pm 0 . 2 6 }$ </td><td> ${ \bf 0 . 8 0 \pm 0 . 1 8 }$   $0 . 2 3 \pm 0 . 2 2$ </td></tr></table>

contrast, LG-AC does not introduce additional hyperparameters and maintains stable performance across both reward specifications. For a fair comparison in the main article, all experiments involving the RIS baseline use positive rewards with $\alpha = \bar { 2 } ^ { - 8 }$

Table 5: Sensitivity analysis of the RIS baseline to the KL penalty factor α under different reward specifications. We report the mean and standard deviation of the success rate, aggregated over two environments (SnakeMaze5 and Dubins Hallway) with 10 independent runs each. The ideal range for α differs across reward specifications.
<table><tr><td>Method</td><td>α</td><td>Positive Reward Negative Reward</td><td></td></tr><tr><td></td><td> $\overline { { 2 ^ { - 1 2 } } }$ </td><td> $0 . 8 5 \pm 0 . 1 7$ </td><td> $0 . 2 5 \pm 0 . 0 9$ </td></tr><tr><td></td><td> $2 ^ { - 1 1 }$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 6 }$ </td><td> $0 . 2 5 \pm 0 . 0 9$ </td></tr><tr><td></td><td> $2 ^ { - 1 0 }$ </td><td> ${ \bf 0 . 9 8 \pm 0 . 0 7 }$ </td><td> $0 . 2 6 \pm 0 . 1 0$ </td></tr><tr><td></td><td> $2 ^ { - 9 }$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 6 }$ </td><td> $0 . 2 4 \pm 0 . 0 8$ </td></tr><tr><td></td><td> $2 ^ { - 8 }$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 5 }$ </td><td> $0 . 2 6 \pm 0 . 1 0$ </td></tr><tr><td></td><td> $2 ^ { - 7 }$ </td><td> $\mathbf { 0 . 9 8 \pm 0 . 0 6 }$ </td><td> $0 . 2 6 \pm 0 . 0 9$ </td></tr><tr><td>RIS</td><td> $2 ^ { - 6 }$ </td><td> $\mathbf { 0 . 9 7 \pm 0 . 0 9 }$ </td><td> $0 . 3 6 \pm 0 . 1 6$ </td></tr><tr><td></td><td> $2 ^ { - 5 }$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 5 }$ </td><td> $0 . 4 6 \pm 0 . 2 6$ </td></tr><tr><td></td><td> $2 ^ { - 4 }$ </td><td> $0 . 9 5 \pm 0 . 1 3$ </td><td> ${ \bf 0 . 7 0 \pm 0 . 2 7 }$ </td></tr><tr><td></td><td> $2 ^ { - 3 }$ </td><td> $0 . 8 2 \pm 0 . 2 0$ </td><td> ${ \bf 0 . 7 2 \pm 0 . 3 2 }$ </td></tr><tr><td></td><td> $2 ^ { - 2 }$ </td><td> $0 . 4 7 \pm 0 . 2 2$ </td><td> ${ \bf 0 . 7 4 \pm 0 . 3 1 }$ </td></tr><tr><td></td><td> $2 ^ { - 1 }$ </td><td> $0 . 2 4 \pm 0 . 1 3$ </td><td> ${ \bf 0 . 7 6 \pm 0 . 2 9 }$ </td></tr><tr><td></td><td> $2 ^ { 0 }$ </td><td> $0 . 1 9 \pm 0 . 0 7$ </td><td> ${ \bf 0 . 7 4 \pm 0 . 2 9 }$ </td></tr><tr><td></td><td> $2 ^ { 1 }$ </td><td> $0 . 1 5 \pm 0 . 0 9$ </td><td> ${ \bf 0 . 6 9 \pm 0 . 3 3 }$ </td></tr><tr><td></td><td> $2 ^ { 2 }$ </td><td> $0 . 0 9 \pm 0 . 0 9$ </td><td> $0 . 6 0 \pm 0 . 4 0$ </td></tr><tr><td></td><td> $2 ^ { 3 }$ </td><td> $0 . 0 9 \pm 0 . 0 9$ </td><td> $0 . 6 3 \pm 0 . 3 7$ </td></tr><tr><td></td><td> $2 ^ { 4 }$ </td><td> $0 . 0 8 \pm 0 . 1 0$ </td><td> $0 . 6 1 \pm 0 . 2 9$ </td></tr><tr><td> $\mathrm { L G } { \cdot } \mathrm { A C }$ </td><td></td><td> $0 . 9 9 \pm 0 . 0 4$ </td><td> $0 . 9 9 \pm 0 . 0 3$ </td></tr><tr><td> $_ { \mathrm { S A C + H E R } }$ </td><td></td><td> $0 . 5 8 \pm 0 . 2 5$ </td><td> $0 . 2 8 \pm 0 . 1 0$ </td></tr></table>