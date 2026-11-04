# Cooperative Multi-Agent Reinforcement Learning with Submodular Trajectory Rewards

Yasaman Zolfimoselo EPFL, Switzerland

Nikolas Geroliminis EPFL, Switzerland

Maryam Kamgarpour EPFL, Switzerland

## Abstract

We study cooperative multi-agent reinforcement learning with decoupled dynamics and a monotone submodular reward function over joint trajectories. Due to the non-additive reward, an optimal policy may depend on the history of all agents. This dependence makes exact optimization exponential in the horizon and the number of agents. To overcome this challenge while retaining theoretical guarantees, we introduce pre-window curvature c<sub>m</sub>, which quantifies how states visited more than m steps earlier reduce current marginal gains. Using this curvature to control the error from restricting the policy of each agent to depend only on its last m + 1 states, we propose a greedy algorithm that exploits the decoupled dynamics to optimize local policies sequentially. For fixed m, the algorithm achieves a $\frac { 1 - c _ { m } } { 2 - c _ { m } }$ approximation guarantee up to estimation error, and its computation time is polynomial in the number of agents and the horizon. We recover the known 1/2-factor approximation for time-additive rewards and obtain an approximation factor no smaller than the known total-curvature bound for a single agent. We formulate multi-agent trafic monitoring as an example of this framework and evaluate our algorithm on it.

## 1 INTRODUCTION

In standard reinforcement learning (RL), a single agent maximizes an expected sum of stage rewards, which depend on the agent’s current state and action. Extending the single-agent formulation, cooperative multiagent reinforcement learning (MARL) studies a team of agents that coordinate their actions to maximize a common objective. Each stage reward then depends on the agents’ current joint state and action. A further simplifying assumption is that each stage reward is a sum of the agents’ local rewards [19]. These two forms of additivity separate rewards over time and across agents, respectively. However, in many cooperative applications, the objective is not additive over time or across agents. Without these forms of additivity, an optimal policy may depend on the history of all agents.

In this paper, we consider a common monotone submodular reward defined over the visits along the trajectories of all agents. A visit is a time-state pair (t, s) indicating that at least one agent is in state s at time t. Submod ularity captures diminishing returns both across agents and over time. This setting is relevant to informative sensing and path planning [10, 22], cooperative coverage and biodiversity monitoring [16], and multi-agent persistent monitoring [2, 5, 20]. Trafic monitoring with multiple unmanned aerial vehicles (UAVs) [9, 12, 15] is a form of persistent monitoring that we use as a running example. In this example, monitoring a road adds less value if it was monitored recently or if another UAV monitors it at the same time. To study such rewards, we formulate a submodular multi-agent Markov decision process (MDP) with decoupled dynamics, in which the next state of each agent depends only on its own current state and action.

## 1.1 Related Work

Submodular maximization and path planning. The greedy algorithm achieves a (1 − 1/e)- approximation for maximizing a monotone submodular set function under a cardinality constraint [14]. This guarantee improves when the function has total curvature c < 1 [6]. These results were extended to settings where the objective function depends on the order of selection [13, 23, 25]. In all these works, removing elements from a feasible selection leaves it feasible. In our setting, visits must form trajectories consistent with the dynamics, so removing a visit can make a feasible selection infeasible. Informative path planning maximizes submodular objectives under such constraints [3, 21]. For multiple robots, sequential allocation yields a team guarantee when each robot’s marginal problem under fixed predecessor policies is solved optimally or with an approximation guarantee [1, 21, 22]. These planning models assume deterministic transition dynamics. Under stochastic dynamics, agents must optimize policies that account for uncertain next states.

Submodular RL. Submodular RL extends these planning models to stochastic dynamics. In this setting, an agent maximizes an expected monotone submodular reward over visits along its trajectory [17, 24]. Global RL generalizes submodular RL to broader classes of trajectory rewards and proposes semi-gradient methods for their optimization [7]. For submodular rewards, constant-factor approximation is generally intractable [17], but a (1 − c)-factor approximation is possible with total curvature $c < 1$ [7, 17]. These results are limited to a single agent. Treating a team as one agent makes the joint state space exponential in the number of agents. Moreover, diminishing returns across agents at the same time step can make the $( 1 - c )$ guarantee vacuous even for time-additive rewards, as we show in Section 3.1.

Submodular MARL. In the multi-agent setting, Chen et al. [4] consider time-additive rewards that are monotone submodular across agents at each stage. Under known decoupled dynamics, they optimize the policies sequentially and obtain a <sup>1</sup> -approximation in time polynomial in the number of agents. Liu et al. [11] study stage rewards defined as marginal gains of a submodular trajectory reward, but their guarantee requires every choice to remain feasible regardless of earlier actions. Hence, the guarantees of Chen et al. [4] and Liu et al. [11] do not cover rewards with diminishing returns over time when the dynamics constrain future visits.

Multi-agent persistent monitoring. Persistent monitoring coordinates mobile agents to revisit important locations [2, 5]. Closest to our setting, Rezazadeh and Kia [20] consider rewards that grow with the time since the last visit and plan over a receding horizon under deterministic dynamics. Their <sup>1</sup> -factor approximation is relative to the optimum within each planning horizon, not to the full-horizon optimal return.

To our knowledge, no prior work gives a polynomialtime algorithm with a curvature-dependent guarantee for MARL with submodular trajectory rewards when stochastic dynamics constrain future visits. Our formulation addresses this setting under decoupled dynamics and contains the settings of Chen et al. [4] and Prajapat et al. [17] as special cases, as Remark 1 shows.

## 1.2 Contributions

Building on the above past work, our contributions are summarized as follows.

1. We introduce a new notion of curvature called prewindow curvature $c _ { m }$ , which measures the maximum relative reduction in marginal gains caused by visits before an m-step window.

2. Using this notion of curvature, we modify the objective so that stage rewards depend only on the visits of all agents in the last $m + 1$ steps. Jointly maximizing this objective yields a policy that achieves a $\left( 1 - c _ { m } \right)$ -approximation of the optimal return.

3. We develop Greedy Windowed Policy Optimization (GWPO), which sequentially optimizes each agent’s policy using marginal rewards based on complete predecessor trajectories and its own mstep history. For fixed m, it runs in polynomial time and achieves the $\frac { 1 - c _ { m } } { 2 - c _ { m } }$ approximation guarantee of Theorem 2 relative to the optimal global history-dependent return.

4. We formulate multi-UAV trafic monitoring as a submodular multi-agent MDP, capturing diminishing returns from repeated monitoring across UAVs and time, and we evaluate GWPO in this setting.

## 2 PROBLEM SETTING

In this section, we first introduce the notation. We then recall submodularity and curvature, which our analysis builds on. Finally, we define the submodular multi-agent MDP and the optimization problem, and illustrate the framework through trafic monitoring.

## 2.1 Notation

Throughout the paper, we write $[ n ] = \{ 1 , \dots , n \}$ for $n \geq 1$ , and $[ 0 ] = \emptyset$ . Bold symbols denote joint quantities over the N agents. For example, x denotes $( x _ { 1 } , \ldots , x _ { N } )$ . For a function g on a product set $\mathcal { X } \times \mathcal { V }$ and $z = ( x , y )$ , we write $g ( z )$ and $g ( x , y )$ interchangeably. Let $\mathcal { D } ( \mathcal { X } )$ denote the probability simplex over a finite set X. Fix a set ${ \mathcal { S } } .$ For a set $B \subseteq \{ 0 , 1 , 2 , \ldots \} \times \mathcal { S }$ and integers t and $t ^ { \prime } .$ , let

$$
B _ { t : t ^ { \prime } } = \{ ( u , s ) \in B : t \leq u \leq t ^ { \prime } \} .\tag{1}
$$

For $t ^ { \prime } = t .$ , we write $B _ { t }$ for $B _ { t : t }$ . For a set $B _ { i }$ of agent i, we write $B _ { i , t : t ^ { \prime } }$ , and for a tuple $\pmb { { B } } = ( B _ { 1 } , \ldots , B _ { N } )$ of such sets, we apply this definition componentwise, $\pmb { B } _ { t : t ^ { \prime } } = ( B _ { 1 , t : t ^ { \prime } } , \ldots , B _ { N , t : t ^ { \prime } } )$

## 2.2 Submodularity and Curvature

We recall monotone submodular functions and their total curvature.

Definition 1 (Monotone submodular function). Let V be a finite ground set and let $F : 2 ^ { \mathcal { V } } \to { \mathbb { R } } _ { > 0 }$ satisfy $F ( \varnothing ) = 0$ . For $X , B \subseteq \nu$ , the marginal gain of X given B is

$$
\Delta ( X \mid B ) = F ( X \cup B ) - F ( B ) ,
$$

abbreviated as $\Delta ( x \mid B )$ when $X = \{ x \}$ . The function $F$ is monotone if $\Delta ( x \mid B ) \geq 0$ for every $B \subseteq \nu$ and $x \in \nu$ . It is submodular if

$\Delta ( x \mid B ) \ge \Delta ( x \mid B ^ { \prime } )$ whenever $B \subseteq B ^ { \prime }$ and $x \in \nu$

Submodularity states that marginal gains decrease as B grows, but it does not quantify how much $\Delta ( x \mid B )$ can decrease relative to $\Delta ( x \mid \emptyset ) \ : = \ : F ( \{ x \} )$ . Total curvature measures this worst-case relative decrease.

Definition 2 (Total curvature). The total curvature of a normalized monotone submodular function $F$ is

$$
c = 1 - \operatorname* { m i n } _ { B \underset { F } { \subset } \mathcal { V } , \ x \in \mathcal { V } \backslash B } \frac { \Delta ( x \mid B ) } { F ( \{ x \} ) } .
$$

For every $B \subseteq \nu$ and $x \in \mathcal { V } \setminus B , \Delta ( x \mid B ) \geq ( 1 -$ $c ) F ( \{ x \} )$ , including when $F ( \{ x \} ) = 0$ . An additive function has $c = 0$ . At the other extreme, $c = 1$ occurs exactly when some element with $F ( \{ x \} ) > 0$ has zero marginal gain in the presence of a set of other elements.

## 2.3 Submodular Multi-Agent MDP

With submodularity and curvature defined, we now introduce the submodular multi-agent MDP on which our method builds.

Definition 3 (Submodular multi-agent MDP). A submodular multi-agent MDP is specified by the tuple

$$
\langle N , H , S , A , \{ P _ { i } \} _ { i \in [ N ] } , \{ \rho _ { i } \} _ { i \in [ N ] } , F \rangle ,
$$

where N denotes the number of agents and H is the horizon length. The finite state space of each agent is $s { \mathrm { . } }$ also called the local state space. The joint state space is $\mathbf { S } = S ^ { N }$ . The finite action space of each agent is ${ \mathcal { A } } ,$ and the joint action space is $\bar { \mathbf { A } } = \mathcal { A } ^ { N }$ . The transition dynamics of agent i are $P _ { i } = ( P _ { i , 0 } , \dots , P _ { i , H - 1 } )$ , where $P _ { i , t } : S \times { \mathcal { A } } \to { \mathcal { D } } ( S )$ and $\textstyle P _ { i , t } ( \cdot \mid s , a )$ is the distribution of its state at time $t + 1$ The transition dynamics are decoupled, so the joint transition dynamics $\mathbf { P } =$ $( P _ { 1 } , \dots , P _ { N } )$ give the joint next-state distribution

$$
\mathbf { P } _ { t } ( s ^ { \prime } \mid s , \pmb { a } ) = \prod _ { i \in [ N ] } P _ { i , t } ( s _ { i } ^ { \prime } \mid s _ { i } , a _ { i } ) .
$$

The initial state distribution of agent i is $\rho _ { i } \in { \mathcal { D } } ( S )$ Initial states are independent, so $\begin{array} { r } { \pmb { \rho } = \prod _ { i \in [ N ] } \rho _ { i } } \end{array}$ . The common reward is a monotone submodular function $F : 2 ^ { \nu }  \mathbb { R } _ { > 0 }$ over trajectories. Its ground set $\nu =$ $\{ 0 , \ldots , H \} \times { \mathcal { S } }$ contains all possible visits, with $\mathcal { V } _ { t : t ^ { \prime } }$ defined as in (1). We assume that the reward function satisfies $F ( B ) > 0$ for at least one $B \subseteq \nu$

Agent i’s trajectory is the set of its visits,

$$
\tau _ { i } = \{ ( t , s _ { i , t } ) : 0 \leq t \leq H \} \subseteq \mathcal { V } ,
$$

and the joint trajectory is $\boldsymbol \tau = ( \tau _ { 1 } , \dots , \tau _ { N } )$ . In set operations we identify a tuple of trajectories with the union of its components. Using the notation in (1), this gives

$$
F ( \pmb { \tau } _ { t : t ^ { \prime } } ) : = F \Big ( \bigcup _ { i \in [ N ] } \tau _ { i , t : t ^ { \prime } } \Big ) .
$$

For example, if $\tau _ { 1 } = \{ ( 0 , s _ { 1 } ) \}$ and $\tau _ { 2 } = \{ ( 0 , s _ { 2 } ) \}$ , then ${ \boldsymbol \tau } = ( \tau _ { 1 } , \tau _ { 2 } )$ is treated as $\{ ( 0 , s _ { 1 } ) , ( 0 , s _ { 2 } ) \}$ whenever it appears in a set operation.

The formulation and analysis extend to monotone submodular rewards on the ground set $\{ 0 , \dots , H - 1 \} \times S \times$ $\mathcal { A }$ by including actions in the trajectories. For simplicity, we present the results using $\mathcal { V } = \{ 0 , \ldots , H \} \times \mathcal { S }$

Histories and policies. For a memory length $m \in$ $\{ 0 , \ldots , H \}$ and a time $t \in \{ 0 , \ldots , H \}$ , the m-step history of agent i is

$$
h _ { i , t } ^ { m } = ( s _ { i , \operatorname* { m a x } ( t - m , 0 ) } , \ldots , s _ { i , t - 1 } ) \in \mathcal { H } _ { t } ^ { m } ,
$$

where $\mathcal { H } _ { t } ^ { m } = S ^ { \mathrm { m i n } ( t , m ) }$ . It contains up to m states of agent i before time t. The joint m-step history is $\pmb { h } _ { t } ^ { m } = \left( h _ { 1 , t } ^ { m } , \ldots , h _ { N , t } ^ { m } \right) \in \mathcal { H } _ { t } ^ { m }$ , where $\mathbf { \mathcal { H } } _ { t } ^ { m } = ( \mathcal { H } _ { t } ^ { \bar { m } } ) ^ { N }$ For $m = H$ , the current state $s _ { i , t }$ and the history $h _ { i , t } ^ { H }$ together form the full history $( s _ { i , 0 } , \ldots , s _ { i , t } )$

For a memory of length m, a global policy of agent i is a sequence $\pi _ { i } = ( \pi _ { i , 0 } , \ldots , \pi _ { i , H - 1 } )$ with $\pi _ { i , t } : \mathbf { S } \times \mathbf { \mathcal { H } } _ { t } ^ { m } \to$ $\mathcal { D } ( \mathcal { A } )$ , and $\Pi _ { m }$ denotes the class of such policies. Here, $\pi _ { i , t } ( a \mid s _ { t } , h _ { t } ^ { m } )$ denotes the probability that agent i takes action $a \in { \mathcal { A } }$ at time t given $\mathbf { \boldsymbol { s } } _ { t }$ and $\boldsymbol { h } _ { t } ^ { m }$ . A local policy of agent i instead depends only on its local state and m-step history $( s _ { i , t } , h _ { i , t } ^ { m } )$ , with $\pi _ { i , t } : S \times \mathcal { H } _ { t } ^ { m } \to$ $\mathcal { D } ( \mathcal { A } )$ . We denote the class of such policies by $\Pi _ { m } ^ { \mathrm { l o c } } . \ \mathrm { A }$ joint policy is denoted by $\pmb { \pi } = ( \pi _ { 1 } , \ldots , \pi _ { N } )$ , where $\pi _ { i }$ is a global or local policy of agent i. The joint policy at time t is $\pmb { \pi } _ { t } = ( \pi _ { 1 , t } , \ldots , \pi _ { N , t } )$ , and the actions of the agents are independent given $( s _ { t } , h _ { t } ^ { m } )$ ,

$$
\pi _ { t } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } , \boldsymbol { h } _ { t } ^ { m } ) = \prod _ { i \in [ N ] } \pi _ { i , t } ( a _ { i } \mid \boldsymbol { s } _ { t } , \boldsymbol { h } _ { t } ^ { m } ) .\tag{2}
$$

We use the notation $\Pi _ { m }$ and $\Pi _ { m } ^ { \mathrm { l o c } }$ for the sets of joint global and joint local policies with memory m, respectively. The cases $m = 0$ and $m = H$ give global Markov and global history-dependent policies.

Problem statement. Given a submodular multiagent MDP, our goal is to find a joint policy $\pmb { \pi } \in \Pi _ { H }$ that maximizes the expected reward:

$$
J ( \pi ) = \mathbb { E } _ { \pi , \rho , \mathbf { P } } \left[ F ( \tau ) \right] , \qquad \pi ^ { * } \in \arg \operatorname* { m a x } _ { \pi \in \Pi _ { H } } J ( \pi ) .\tag{3}
$$

The expectation is taken over joint trajectories $\tau$ induced by the policy $\pi$ under the initial distribution $\rho$ and the transition dynamics P. We abbreviate the expectation as $\mathbb { E } _ { \tau \sim \pi }$ in the rest of the paper. Similarly, $\mathbb { E } _ { \tau _ { i } \sim \pi _ { i } }$ denotes expectation under agent $i \mathrm { \ ' } _ { \mathrm { S } }$ local policy $\pi _ { i } ,$ with $P _ { i }$ and $\rho _ { i }$ implicit. We write $J ^ { * } = J ( \pi ^ { * } )$ for the optimal value of (3). Restricting $\pi ^ { * }$ to the product form (2) does not reduce $J ^ { * }$ , because an optimal policy of this form exists (Lemma 2).

In this work, we assume known transition dynamics $\{ P _ { i } \} _ { i \in [ N ] }$ and initial distributions $\{ \rho _ { i } \} _ { i \in [ N ] }$ . During policy optimization, we also assume that $F ( B )$ can be evaluated exactly for every $B \subseteq \nu$

Remark 1. For $N = 1$ , our formulation reduces to reinforcement learning with a submodular reward over the agent’s trajectory [17]. Chen et al. [4] instead consider a multi-agent setting, but their objective is a sum of monotone submodular stage rewards $f : 2 ^ { S \times A } \to { \mathbb { R } } _ { \geq 0 }$ each evaluated on the state-action pairs of all agents at one time step. Their setting is the special case of our framework with ground set $\{ 0 , \dotsc , H - 1 \} \times S \times { \mathcal { A } }$ and

$$
F ( B ) = \sum _ { t = 0 } ^ { H - 1 } f { \big ( } \{ ( s , a ) : ( t , s , a ) \in B _ { t } \} { \big ) } .
$$

## 2.4 Trafic-Monitoring Example

We illustrate the submodular multi-agent MDP with a trafic-monitoring example used throughout the paper. In this example, the N agents are moving UAVs that monitor an urban road network over a horizon of H time steps. The network is an $M \times L$ grid with road set E. The local state $s \in S$ of each UAV is its position at a grid node, and its actions move it to an adjacent node or keep it in place. The transition dynamics and initial state distributions of the UAVs are decoupled, since the UAVs move independently and their initial positions are drawn independently. From position s, a UAV monitors the set ${ \mathcal { C } } ( s ) \subseteq { \mathcal { E } }$ of roads connected to s. Figure 1 shows $\mathcal C ( s )$ for five positions.

Monitoring each road $e \in { \mathcal { E } }$ provides information such as its trafic density and occupancy [12]. We call the added value of this information the importance of the road. Before road e is first monitored, its importance takes its maximum value $w _ { e } \geq 0$ . When a UAV monitors a road $e ,$ the importance of the road drops to zero. Its importance then recovers over time as the information becomes outdated. After ℓ time steps, its importance is $w _ { e } \alpha \ell$ , where $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ and $\beta > 0$ is the recovery rate.

![](images/407b8392c764242238d5f0a0c09fe0b40f057713d7cec313b5d1e52bdf001dac.jpg)  
Figure 1: Part of the road grid with five UAV positions $s _ { 1 } , \ldots , s _ { 5 }$ . For $i \leq 4 .$ the roads $\mathcal { C } ( s _ { i } )$ monitored from $s _ { i }$ share the color of $s _ { i }$ . Each dashed road in $\mathcal { C } ( s _ { 5 } )$ is colored, so it is also monitored from one of $s _ { 1 } , \ldots , s _ { 4 }$

For a set $B \subseteq \nu$ and a road $e \in { \mathcal { E } }$ , the set of times at which road e is monitored by visits in B is

$T _ { e } ( B ) = \{ t : ( t , s ) \in B$ for some $s \in S$ with $e \in \mathcal { C } ( s ) \}$

For $t \in T _ { e } ( B )$ , let $\ell _ { e } ( t ) = t - \operatorname* { m a x } \{ t ^ { \prime } \in T _ { e } ( B ) : t ^ { \prime } < t \}$ be the number of time steps since road e was previously monitored. If t is the first time in $T _ { e } ( B )$ , we set $\ell _ { e } ( t ) =$ ∞ and $\alpha _ { \infty } = 1$ . The trafic-monitoring reward is

$$
F ( B ) = \sum _ { e \in \mathcal { E } } \sum _ { t \in T _ { e } ( B ) } w _ { e } \alpha _ { \ell _ { e } ( t ) } .\tag{4}
$$

Thus, $F ( \tau )$ is the total value of the information collected by the fleet along its joint trajectory $\tau ,$ and $F$ has diminishing returns across agents and over time. For example, when two UAVs monitor a road at the same time, the second UAV adds no new information about that road. To complete the formulation, we show in Appendix C that the reward (4) is monotone submodular for a general class of recovery profiles $\alpha _ { \ell } ,$ and this class includes $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ for every $\beta > 0$

## 3 HARDNESS AND CURVATURE

To further motivate our results, we first explain why finding a policy that maximizes $J ( \pi )$ in (3) is computationally hard. This hardness justifies restricting the class of rewards to obtain constant-factor guarantees. We then introduce pre-window curvature $c _ { m } .$ a property of the reward function that our approximation guarantees depend on.

## 3.1 Computational Hardness

Maximizing J poses two computational challenges. First, the single-agent case $N = 1$ admits no constantfactor approximation in polynomial time under standard complexity assumptions, even with deterministic dynamics [17, Theorem 1]. For stochastic dynamics, optimal policies of this case may require history dependence, and representing an arbitrary history-dependent policy can require space exponential in the horizon [17]. Second, the joint state and action spaces of N agents have sizes $| \mathcal { S } | ^ { N }$ and $| \mathcal { A } | ^ { N }$ , so optimizing directly over them takes time exponential in N.

Constant-factor guarantees therefore require further restricting the class of rewards. For a single agent, existing algorithms achieve a (1 − c)-approximation for submodular trajectory rewards with total curvature $c <$ 1 [7, 17]. For multiple agents under decoupled dynamics, Chen et al. [4] obtain a 1/2-approximation for timeadditive rewards without any curvature assumption. Our framework contains both settings, so we seek a single curvature condition that yields an approximation guarantee for our problem and recovers or improves the known guarantees when specialized to either setting.

Total curvature does not achieve this goal in the multiagent setting, since simultaneous visits can make $c = 1$ even for time-additive rewards. Consider Figure 1 with the reward (4) and $w _ { e } > 0$ for every road $e \in { \mathcal { E } }$ . The visits $( t , s _ { 1 } ) , \ldots , ( t , s _ { 4 } )$ monitor every road in $\mathcal { C } ( s _ { 5 } )$ Thus, $x = ( t , s _ { 5 } )$ has zero marginal gain given these four visits although $\begin{array} { r } { F ( \{ x \} ) = \sum _ { e \in \mathcal { C } ( s _ { 5 } ) } w _ { e } > 0 } \end{array}$ , so $c = 1$ regardless of the choice of $\alpha _ { \ell }$ in the reward (4). We therefore seek a condition that limits the efect of visits long before the current time on marginal gains.

## 3.2 Pre-Window Curvature

We quantify how visits before an m-step window decrease the marginal gain of a current visit. For the ground set V of Definition 3, the m-step window at time t is $\mathcal { V } _ { t - m : t - 1 } \subseteq \mathcal { V }$ (Figure 4 in Appendix B). For fixed $x _ { t } \in \mathcal V _ { t }$ and $Z \subseteq \nu$ , adding visits $B \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ to Z changes the marginal gain from $\Delta ( x _ { t } \mid Z )$ to $\Delta ( x _ { t } \mid$ $B \cup Z )$ . We measure this decrease relative to $\Delta ( x _ { t } \mid Z )$ and exclude cases with $\Delta ( x _ { t } \mid Z ) = 0$ , since monotonicity and submodularity then give $\Delta ( x _ { t } \mid B \cup Z ) = 0$ We define pre-window curvature as the largest relative decrease over all such $x _ { t } , Z$ , and B.

Definition 4 (Pre-window curvature). For $m \in$ $\{ 0 , \ldots , H \}$ , the pre-window curvature of $F$ is

$$
c _ { m } = 1 - \operatorname* { m i n } _ { t , \ x _ { t } , \ B , \ Z } \frac { \Delta ( x _ { t } \mid B \cup Z ) } { \Delta ( x _ { t } \mid Z ) } ,\tag{5}
$$

where the minimum ranges over times $t \in \{ 0 , \ldots , H \}$ visits $x _ { t } \in \mathcal V _ { t }$ , and sets $B \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ and $Z \subseteq \nu$

Pre-window curvature satisfies $0 ~ \leq ~ c _ { m } ~ \leq ~ c$ and is nonincreasing in $m _ { \colon }$ , with $c _ { H } = 0$ . Moreover, $c _ { m } = 0$ whenever F is additive over time, even if overlap across agents makes c equal one. We prove these properties in Appendix B.

Returning to the example of Section 2.4, we show that the reward (4) has pre-window curvature $c _ { m } < 1$ for every $\beta > 0$ . Proposition 3 in Appendix C.2 gives the explicit expression $c _ { m } = e ^ { - \beta ( \bar { m ^ { } } + 1 ) }$ for $0 \leq m < H$ Without recovery $( \beta = 0 )$ , the reward (4) values only the first monitoring of each road and has $c _ { m } = c = 1$ for every $m < H$ . In summary, $c _ { m } < 1$ describes rewards in which visits before an m-step window have a limited efect on the marginal gain of a current visit.

## 4 ALGORITHMS AND ANALYSIS

To address the challenges of Section 3.1, we first approximate the objective J by the expected sum of stage rewards that depend only on the current joint state and the joint m-step history. The error of this approximation is controlled by the pre-window curvature $c _ { m } ,$ , and the maximizer of this approximated objective achieves at least $( 1 - c _ { m } ) J ^ { * }$ . However, maximizing this objective over joint policies is still exponential in the number of agents N. In Section 4.3, we therefore exploit the decoupled dynamics to optimize the policies of the agents sequentially in polynomial time for fixed m. The resulting policies achieve a $\frac { 1 - c _ { m } } { 2 - c _ { m } }$ guarantee up to estimation error.

## 4.1 History-Augmented State

We introduce the history-augmented state used to define the finite-horizon MDPs in the next two subsections. We fix a memory length $m \in \{ 0 , \ldots , H \}$ and write $h _ { i , t } = h _ { i , t } ^ { m }$ for the histories of Section 2. At time t, the history-augmented state of agent i is the pair $\tilde { s } _ { i , t } ~ = ~ ( s _ { i , t } , h _ { i , t } )$ , which lies in the augmented state space $\tilde { S } _ { t } = \mathcal { S } \times \mathcal { H } _ { t } ^ { m }$ . By definition, policies in $\Pi _ { m } ^ { \mathrm { l o c } }$ and $\mathbf { I I } _ { m }$ are Markov policies on $\tilde { s } _ { i , t }$ and $\tilde { \mathbf { s } } _ { t } \in \tilde { \mathbf { S } } _ { t } = \tilde { S } _ { t } ^ { N }$ respectively. For $\pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } }$ , the input of $\pi _ { i , t }$ takes $| \tilde { S } _ { t } | = | S | ^ { \operatorname* { m i n } ( t , m ) + 1 } \leq | S | ^ { m + 1 }$ values, compared with $| S | ^ { i + 1 }$ values for a full history.

The augmented transition dynamics are Markov and remain decoupled across agents. In particular, the next state $s _ { i , t + 1 } \sim P _ { i , t } ( \cdot \mid s _ { i , t } , a _ { i , t } )$ depends only on the state and action of agent i. The history updates deterministically by appending $s _ { i , t }$ to $h _ { i , t }$ and dropping the oldest entry if its length exceeds m. Figure 6 in Appendix D.3 illustrates this update.

Since the reward function F is defined on sets of visits, we map each history to the visits it determines. The map $\nu _ { t } : \mathcal { H } _ { t } ^ { m } \ \to \ 2 ^ { \nu }$ assigns to each history $h = ( s _ { \operatorname* { m a x } ( t - m , 0 ) } , \ldots , s _ { t - 1 } ) \in \mathcal { H } _ { t } ^ { m }$ the set of visits

$$
\nu _ { t } ( h ) = \big \{ ( l , s _ { l } ) : \operatorname* { m a x } ( t - m , 0 ) \leq l \leq t - 1 \big \} .
$$

For a joint history, $\begin{array} { r } { \nu _ { t } ( h _ { t } ) = \bigcup _ { i \in [ N ] } \nu _ { t } ( h _ { i , t } ) } \end{array}$ . Along a trajectory, $\nu _ { t } ( h _ { i , t } ) = \tau _ { i , t - m : t - 1 }$ and $\nu _ { t } ( h _ { t } ) = \tau _ { t - m : t - 1 }$

## 4.2 Windowed Objective

Our goal is to approximate Problem (3) by a finitehorizon MDP with additive stage rewards on the joint augmented states. We therefore need stage rewards $r _ { t } ( \tilde { \boldsymbol { s } } _ { t } )$ that depend only on the joint augmented state and whose expected sum under each policy π approximates $J ( \pi )$ . Adding the team visits one time step at a time decomposes $F$ over time as

$$
F ( \pmb { \tau } ) = \sum _ { t = 0 } ^ { H } \Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { 0 : t - 1 } ) .\tag{6}
$$

The term at time t is the marginal gain of the current team visits $\tau _ { t }$ given all earlier visits $\tau _ { 0 : t - 1 }$ . The augmented state $\tilde { \mathbf { \boldsymbol { s } } } _ { t }$ determines only the part $\tau _ { t - m : t - 1 }$ of these earlier visits, so this term is not always a stage reward on the augmented states. We therefore define the stage reward $r _ { t } : \tilde { \mathbf { S } } _ { t }  \mathbb { R } _ { \geq 0 }$ as the marginal gain of the current team visits given only the visits $\nu _ { t } ( h _ { t } )$ within the m-step window,

$$
r _ { t } \big ( \tilde { s } _ { t } \big ) = \Delta \big ( \{ ( t , s _ { i , t } ) : i \in [ N ] \} \ | \ \nu _ { t } ( h _ { t } ) \big ) .\tag{7}
$$

Along a trajectory, $r _ { t } \big ( \tilde { \mathbf { s } } _ { t } \big ) = \Delta ( \pmb { \tau } _ { t } \ | \ \pmb { \tau } _ { t - m : t - 1 } )$ . Replacing each term of Equation (6) by $r _ { t } ( \tilde { \boldsymbol { s } } _ { t } )$ and taking the expectation gives the windowed objective,

$$
\Phi _ { m } ( \pi ) = \mathbb { E } _ { \tau \sim \pi } \left[ \sum _ { t = 0 } ^ { H } r _ { t } ( \tilde { \mathbf { s } } _ { t } ) \right] .
$$

Maximizing $\Phi _ { m }$ is thus a finite-horizon MDP problem on the augmented states. This problem has a deterministic optimal policy that lies in $\Pi _ { m } ~ [ 1 8 ]$ . Algorithm 2 in Appendix D.3 computes such a policy. It remains to show that $\Phi _ { m }$ approximates $J .$

Lemma 1. For every policy π,

$$
( 1 - c _ { m } ) \Phi _ { m } ( \pmb { \pi } ) \leq J ( \pmb { \pi } ) \leq \Phi _ { m } ( \pmb { \pi } ) .
$$

Proof sketch. The upper bound holds by submodularity, since $r _ { t }$ considers only $\pmb { \tau } _ { t - m : t - 1 } \subseteq \pmb { \tau } _ { 0 : t - 1 }$ . The lower bound holds by the definition of $c _ { m }$

Theorem 1. Let ${ \pi } ^ { \prime } \in \arg \operatorname* { m a x } _ { { \pmb { \pi } } \in { \pmb { \Pi } } _ { H } } \Phi _ { m } ( { \pmb { \pi } } )$ . Then

$$
J ( \pi ^ { \prime } ) \geq \left( 1 - c _ { m } \right) J ^ { * } .
$$

Proof sketch. The guarantee follows from Lemma 1 and the optimality of $\pi ^ { \prime }$ for $\Phi _ { m }$ . We prove both results in Appendix D.2.

For a single agent with $m = 0$ , the windowed objective is the expected modular reward $\textstyle \sum _ { x \in \tau } F ( \{ x \} )$ of Prajapat et al. [17, Proposition $3 ]$ , and our proof follows their argument with the factor $1 - c _ { 0 } \geq 1 - c$ . For $m = H , c _ { H } = 0$ and $\Phi _ { H } = J $ , so the theorem yields an optimal policy. However, the MDP has up to $| S |$ N(m+1) joint augmented states per time step and $| \dot { \mathcal { A } } | ^ { N }$ joint actions. This exponential dependence on $N$ motivates the sequential policy optimization introduced next.

## 4.3 Sequential Policy Optimization

We adopt the sequential policy optimization approach of prior work [4, 21] to avoid joint optimization. This approach optimizes the policy of one agent at a time while holding predecessor policies fixed. However, the guarantee of Chen et al. [4] cannot be applied directly to $\Phi _ { m }$ . The reason is that an added agent’s visits within the window can reduce current marginal gains, which violates the required monotonicity assumption. We illustrate this violation with an example in $\mathrm { A p - }$ pendix D.4.

Instead, we sequentially construct local stage rewards using complete predecessor trajectories and each agent’s own m-step history. We then use pre-window curvature $c _ { m }$ to bound the relative overestimate caused by ignoring the agent’s visits before the window, which allows a guarantee relative to the optimal global historydependent return $J ^ { * }$

We optimize the policies of the agents in the order $1 , \ldots , N$ . For agent i, we fix the predecessor policies $\bar { \pi } _ { [ i - 1 ] } = ( \bar { \pi } _ { 1 } , \dots , \bar { \pi } _ { i - 1 } )$ computed so far. Even with fixed policies, predecessor trajectories $\tau _ { [ i - 1 ] }$ can be random because of stochastic dynamics or random initial states. We therefore take the expectation over $\tau _ { [ i - 1 ] } \sim \bar { \pi } _ { [ i - 1 ] }$ to define the local stage reward $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ $\tilde { S } _ { t }  \mathbb { R } _ { \geq 0 }$ as

$$
r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) = \mathbb { E } \big [ \Delta \big ( ( t , s _ { i , t } ) \mid \nu _ { t } \big ( h _ { i , t } \big ) \cup \pmb { \tau } _ { [ i - 1 ] } \big ) \big ] .\tag{8}
$$

For $i = 1$ , the reward $r _ { 1 , t } ^ { \bar { \pi } _ { [ 0 ] } } ( \tilde { s } _ { 1 , t } ) = \Delta \big ( ( t , s _ { 1 , t } ) \mid \nu _ { t } ( h _ { 1 , t } ) \big )$ is the stage reward $( 7 )$ with $N = 1$ . For a policy $\pi _ { i }$ of agent i, the local objective is the expected sum of the rewards $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ along the trajectory $\tau _ { i }$ induced by $\pi _ { i } .$

$$
G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) = \mathbb { E } _ { \tau _ { i } \sim \pi _ { i } } \left[ \sum _ { { t = 0 } } ^ { H } r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) \right] .\tag{9}
$$

As in Section 4.2, maximizing $G _ { i } ^ { \bar { \pi } _ { { [ i - 1 ] } } }$ is an MDP problem on the augmented states of agent i. This problem has a deterministic optimal policy that lies in $\Pi _ { m } ^ { \mathrm { l o c } }$

The exact evaluation of (8) may require summing over up to $| S | ^ { ( i - 1 ) ( H + 1 ) }$ predecessor trajectories. For each agent $i \geq 2$ , we therefore estimate this expectation using K trajectories $\pmb { \tau } _ { [ i - 1 ] } ^ { 1 } , \dots , \pmb { \tau } _ { [ i - 1 ] } ^ { K }$ sampled independently under $\bar { \pi } _ { [ i - 1 ] }$ 2

$$
\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Delta \big ( ( t , s _ { i , t } ) \mid \nu _ { t } ( h _ { i , t } ) \cup \pmb { \tau } _ { [ i - 1 ] } ^ { k } \big ) .\tag{10}
$$

Replacing $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ with $\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ in (9) defines the empirical local objective $\widehat { G } _ { i } ^ { \bar { \pmb { \pi } } _ { [ i - 1 ] } }$ . For agent 1, no sampling is required and ${ \widehat { G } } _ { 1 } = G _ { 1 }$

Algorithm 1 GWPO   
Require: submodular multi-agent MDP; memory   
length m $\in \{ 0 , \ldots , H \}$ ; number of samples $K \geq 1$   
1: for $i \in [ N ]$ do   
2: if $i = 1$ then   
3: $\hat { r } _ { 1 , t } ^ { \bar { \pi } _ { [ 0 ] } } ( \tilde { s } _ { 1 , t } ) = \Delta \big ( ( t , s _ { 1 , t } ) \mid \nu _ { t } ( h _ { 1 , t } ) \big )$ $\forall t , \tilde { s } _ { 1 , t }$   
4: else   
5: $\pmb { \tau } _ { [ i - 1 ] } ^ { 1 } , \dots , \pmb { \tau } _ { [ i - 1 ] } ^ { K } \sim \bar { \pmb { \pi } } _ { [ i - 1 ] }$   
6: Estimate $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } )$ by $\hat { r } _ { i , t } ^ { \tilde { \pi } _ { [ i - 1 ] } } ( \tilde { s } )$ in (10) ∀t, s˜   
7: end if   
8: $\bar { \pi } _ { i }  \mathrm { A }$ lgorithm $2 ( \mathcal { S } , \mathcal { A } , m , P _ { i } , \hat { \boldsymbol { r } } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } )$   
9: end for   
10: return $\hat { \pi } = ( \hat { \pi } _ { 1 } , \dots , \hat { \pi } _ { N } )$

Greedy Windowed Policy Optimization (GWPO), stated in Algorithm 1, solves each agent’s augmented MDP in turn with any exact finite-horizon MDP solver, such as Algorithm 2. For agent $i ,$ it maximizes $\widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } }$ given the fixed predecessors’ policies $\bar { \pi } _ { [ i - 1 ] }$ , and then fixes $\bar { \pi } _ { i }$ before proceeding to agent $i + \mathrm { \bar { 1 } }$ . During centralized training, GWPO samples the trajectories of the predecessors and evaluates $F$ exactly, as assumed in Section 2. Execution is decentralized, since each agent’s policy depends only on its own augmented state.

The guarantee of GWPO depends on the number of agents. For N = 1, GWPO exactly maximizes $\Phi _ { m } ,$ so Theorem 1 gives $J ( \bar { \pi } ) \geq ( 1 - c _ { m } ) J ^ { * }$ . For $N \geq 2 .$ , the following theorem bounds $J ( \bar { \boldsymbol { \pi } } )$ relative to $J ^ { * }$ in terms of $c _ { m }$ and the reward-estimation error.

Theorem 2. Let $N \geq 2$ and fix $m \in \{ 0 , \ldots , H \}$ . For $\epsilon > 0$ and $\delta \in ( 0 , 1 )$ , choose the number of sampled predecessor trajectories K such that

$$
K \ge \frac { R ^ { 2 } } { 2 \epsilon ^ { 2 } } \ln \frac { 2 ( N - 1 ) ( H + 1 ) | S | ^ { m + 1 } } { \delta } ,\tag{11}
$$

where $R = \operatorname* { m a x } _ { v \in \mathcal { V } } F ( \{ v \} )$ is the largest reward of a single visit. Then, with probability at least $1 - \delta$ , the joint policy π¯ returned by Algorithm 1 satisfies

$$
J ( \bar { \pi } ) \geq \frac { 1 - c _ { m } } { 2 - c _ { m } } \left( J ^ { * } - 2 ( N - 1 ) ( H + 1 ) \epsilon \right) .\tag{12}
$$

The approximation factor $\frac { 1 - c _ { m } } { 2 - c _ { m } }$ is positive for $c _ { m } < 1$ equals $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ at $c _ { m } = 0$ , and is zero at $c _ { m } = 1$ . The additive error comes from GWPO optimizing the estimated rewards $\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ instead of the rewards $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ for each agent $i \geq 2$ and time t. By (11), the sample size K grows only logarithmically in N, H, and $1 / \delta$

For rewards additive over time, Proposition 1(iv) shows that $c _ { m } = 0$ for every m. Therefore, the factor equals ${ \frac { 1 } { 2 } } ,$ as in Chen et al. [4, Theorem 1]. The factor $\frac { \bar { 1 } - c _ { m } } { 2 - c _ { m } }$ has the form $\frac { q } { 1 + q }$ of sequential allocation [22, Theorem 4] with $q = 1 - c _ { m }$ . Their theorem assumes a q-approximation for the marginal reward of each agent and uses the same policy classes for the returned policies and the optimum. We do not assume such an approximation, and we compare our local policies with the optimal global history-dependent policy.

Proof sketch of Theorem 2. The proof addresses two challenges that do not arise in the time-additive setting [4]. First, the optimal policy $\pi ^ { * } \in \Pi _ { H }$ may depend on the full joint history, while the policy $\bar { \pmb { \pi } } \in \Pi _ { m } ^ { \mathrm { l o c } }$ returned by GWPO uses only the current state and m-step history of each agent. Second, the return $J ( \bar { \pi } )$ is not the sum of the objectives that GWPO maximizes, since the objective of each agent ignores its visits prior to the m-step window. To address the first challenge, we consider an arbitrary joint policy $\pmb { \pi } \in \Pi _ { H }$ and an arbitrary agent i. We show that there exists a local policy in $\Pi _ { m } ^ { \mathrm { l o c } }$ that reproduces the distribution of the augmented state of agent i under π at every time step. For the second, we use pre-window curvature to lower bound $J ( \bar { \pi } )$ by the objectives that GWPO maximizes.

The proof proceeds in three steps. First, the optimality gap $J ^ { * } - J ( \bar { \pi } )$ is bounded by the sum of the maximum local objectives of the agents. Second, we bound the suboptimality of maximizing the local objective of each agent by the errors of reward estimation. Following Chen et al. [4], Hoefding’s inequality and a union bound keep all these errors below ϵ with probability at least 1−δ. Finally, pre-window curvature lower bounds $J ( \bar { \pi } )$ by $1 - c _ { m }$ times the sum of the attained local objectives. Together, these three bounds imply (12). Appendix E gives the full proof. □

GWPO achieves the guarantee of Theorem 2 in polynomial time for fixed m. We count oracle evaluations and transition samples as unit-cost operations and first consider the cost for a single agent i. Sampling K trajectories of its at most $N - 1$ predecessors costs O(NKH). Estimating the rewards (10) at all times and augmented states costs $O ( H | S | ^ { m + 1 } K )$ . Solving the MDP of agent i on its augmented states with Algorithm 2 costs $O ( H | S | ^ { m + 2 } | A | )$ . Summing over the N agents gives the computation time of Algorithm 1,

$$
O \big ( N H | S | ^ { m + 1 } ( K + | S | | A | ) + N ^ { 2 } K H \big ) .
$$

## 5 NUMERICAL EXPERIMENTS

We empirically studied the return and computation time of GWPO with memory length m on the traficmonitoring example of Section 2.4. Our instance used a $2 0 \times 2 0$ road grid, the horizon $H \ : = \ : 2 0$ , and initial positions of the agents sampled independently and uniformly. We first studied how memory affected the return and the computation time of GWPO, and then compared its performance with two baselines. The baselines were Greedy Policy Optimization (GPO) [4] and Submodular Policy Optimization (SubPO) [17]. GPO optimizes a time-additive objective, and SubPO optimizes global policies by policy gradient. We used SubPO with Markov (SubPO-M) and history-dependent (SubPO-NM) policies. SubPO-NM encodes the history with a recurrent neural network. We evaluated all policies under the reward (4) on 10,000 common initial joint states. Appendix F gives further experimental details.

![](images/48215afb8b3b21b61b2c127cfbe20e93a6fcc5472d0b331ddaacd8ef0084d156.jpg)  
(a) Value of memory

![](images/20a30e401137d2d577ac64fa3bf88bb016a6e570f490040858ac677ab1a93025.jpg)  
(b) Computation time

Figure 2: Performance improvement and computational cost of memory in GWPO. (a) Return improvement of GWPO with diferent memory lengths m over GPO [4] for $N = 2 0$ agents and diferent recovery rules. (b) Computation time for diferent numbers of agents N, comparing GWPO with $m = 0 , 3$ and GPO at $\beta = 0 . 4$  
![](images/66e059cc57f58fe384df49dc3514d05ca89207dd26e2774d1145b5a10f78f5c7.jpg)  
(a) Computation time $T _ { N }$

![](images/90aa7dc8b16fea2df0017db4b5a21a86f7c422e32f8ea89ea23efb02ca8bf6d1.jpg)  
(b) Computation time $5 T _ { N }$

![](images/65de2dd8821124485eb433a7f9efb9827a1bc4f8cb6d935ab53892e08b1a5fc5.jpg)  
(c) Computation time $2 0 T _ { N }$  
Figure 3: Return gap between each baseline and GWPO with $m = 3$ at $\beta = 0 . 4$ , as a percentage of the GWPO return. Negative values indicate higher returns for GWPO. SubPO-M and SubPO-NM are allowed computation times of (a) $T _ { N }$ , (b) $5 T _ { N }$ , and (c) $2 0 T _ { N }$ , where $T _ { N }$ is the computation time of GWPO for N agents.

Figure 2 shows how the memory length afects the return and computation time of GWPO. For $N = 2 0$ agents, increasing memory improved returns over GPO for the non-additive rewards in Figure $2 ( \mathrm { a } )$ . For the recovery rule $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ , smaller $\beta$ means that road importance recovers more slowly after a visit. Longer memory helped more when recovery was slower, while GWPO approximately matched GPO in the timeadditive case $\alpha _ { \ell } = 1$ . Figure 2(b) shows that these gains came with higher computation time.

Figure 3 compares return gaps relative to GWPO across the number of agents and computation times. With $T _ { N }$ , GWPO and SubPO achieved comparable returns for $N \leq 6 .$ while GWPO achieved higher returns for $N \geq 1 0 .$ . With $2 0 T _ { N }$ , both SubPO variants achieved up to 6% higher returns than GWPO for $N \leq 1 5$ , but lower returns at $N = 3 0$ . Appendix F.4 reports results over five SubPO seeds and their confidence intervals.

## 6 CONCLUSION

We introduced pre-window curvature to quantify how visits before an m-step window afect marginal gains in submodular rewards over trajectories. Using this notion, we developed GWPO, which exploits decoupled dynamics to sequentially optimize local policies. For fixed $m ,$ its computation time is polynomial in the number of agents and the horizon. We established a $\frac { 1 - c _ { m } } { 2 - c _ { m } }$ approximation guarantee relative to the optimal global history-dependent return, up to estimation error. We formulated multi-agent trafic monitoring as a submodular multi-agent MDP and evaluated GWPO against the time-additive baseline GPO and SubPO, which optimizes global policies by policy gradient.

## Acknowledgements

This work was supported as a part of NCCR Automation, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40 225155).

## AI use statement

In this work, we used generative AI tools for providing feedback on experimental design, checking proofs and suggesting minor corrections, and assisting with implementing and editing code for numerical experiments. We have not used generative AI tools for developing theoretical models or conceptual frameworks, formulating mathematical claims, providing critical ingredients for proofs, proposing or refining hypotheses, or interpreting results. Generating synthetic datasets, cleaning or reformatting datasets, qualitative or thematic data analysis, and translation are not applicable to this work. Additionally, we used generative AI tools for creating and modifying figures, editing text to improve readability, and finding additional related work. We have reviewed all AI outputs used in the activities mentioned above. LLM-generated code was verified and tested for correctness by the authors. All related work identified using AI tools was read and checked by the authors. The authors checked all AI-edited text and AI-suggested revisions. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

[1] Nikolay Atanasov, Jerome Le Ny, Kostas Daniilidis, and George J Pappas. Decentralized active information acquisition: Theory and application to multi-robot SLAM. In International Conference on Robotics and Automation (ICRA), pages 4775–4782. IEEE, 2015.

[2] Christos G. Cassandras, Xuchao Lin, and Xu Chu Ding. An optimal control approach to the multiagent persistent monitoring problem. IEEE Transactions on Automatic Control, 58(4):947–961, 2013.

[3] Chandra Chekuri and Martin Pal. A recursive greedy algorithm for walks in directed graphs. In IEEE Symposium on Foundations of Computer Science (FOCS), pages 245–253. IEEE, 2005.

[4] Wenjing Chen, Chengyuan Qian, Shuo Xing, Yi Zhou, and Victoria Crawford. Multi-agent reinforcement learning with submodular reward. In International Conference on Machine Learning, 2026.

[5] Yann Chevaleyre. Theoretical analysis of the multi-

agent patrolling problem. In IEEE/WIC/ACM International Conference on Intelligent Agent Technology (IAT), pages 302–308, 2004.

[6] Michele Conforti and G´erard Cornu´ejols. Submodular set functions, matroids and the greedy algorithm: Tight worst-case bounds and some generalizations of the Rado-Edmonds theorem. Discrete Applied Mathematics, 7(3):251–274, 1984.

[7] Riccardo De Santi, Manish Prajapat, and Andreas Krause. Global reinforcement learning: Beyond linear and convex rewards via submodular semigradient methods. In International Conference on Machine Learning, pages 10235–10266. PMLR, 2024.

[8] Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963.

[9] Konstantinos Kanistras, Goncalo Martins, Matthew J Rutherford, and Kimon P Valavanis. A survey of unmanned aerial vehicles (UAVs) for trafic monitoring. In 2013 International Conference on Unmanned Aircraft Systems (ICUAS), pages 221–234. IEEE, 2013.

[10] Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-optimal sensor placements in Gaussian processes: Theory, eficient algorithms and empirical studies. Journal of Machine Learning Research, 9: 235–284, 2008.

[11] Jing Liu, Luca Ballotta, Yangyang Yang, Fangfei Li, Yang Tang, and Ruggero Carli. Submodular policy learning for distributed task allocation in open multi-agent systems. arXiv preprint arXiv:2608.14390, 2026.

[12] Marko Maljkovic and Nikolas Geroliminis. On learning-based trafic monitoring with a swarm of drones. In European Control Conference (ECC), pages 1842–1847. IEEE, 2025.

[13] Marko Mitrovic, Ehsan Kazemi, Moran Feldman, Andreas Krause, and Amin Karbasi. Adaptive sequence submodularity. In Advances in Neural Information Processing Systems, volume 32, 2019.

[14] George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming, 14(1):265–294, 1978.

[15] Fatma Outay, Hanan Abdullah Mengash, and Muhammad Adnan. Applications of unmanned aerial vehicle (UAV) in road safety, trafic and highway infrastructure management: Recent advances and challenges. Transportation Research Part A: Policy and Practice, 141:116–129, 2020.

[16] Manish Prajapat, Matteo Turchetta, Melanie Zeilinger, and Andreas Krause. Near-optimal multi-agent learning for safe coverage control. In Advances in Neural Information Processing Systems, volume 35, pages 14998–15012, 2022.

[17] Manish Prajapat, Mojmir Mutny, Melanie Zeilinger, and Andreas Krause. Submodular reinforcement learning. In International Conference on Learning Representations, pages 15190–15212, 2024.

[18] Martin L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, 1994.

[19] Guannan Qu, Adam Wierman, and Na Li. Scalable reinforcement learning of localized policies for multi-agent networked systems. In Learning for Dynamics and Control, pages 256–266. PMLR, 2020.

[20] Navid Rezazadeh and Solmaz S. Kia. A submodular receding horizon solution for mobile multiagent persistent monitoring. Automatica, 127: 109460, 2021.

[21] Amarjeet Singh, Andreas Krause, Carlos Guestrin, and William J. Kaiser. Eficient informative sensing using multiple robots. Journal of Artificial Intelligence Research, 34:707–755, 2009.

[22] Amarjeet Singh, Andreas Krause, and William J. Kaiser. Nonmyopic adaptive informative path planning for multiple robots. In International Joint Conference on Artificial Intelligence, 2009.

[23] Sebastian Tschiatschek, Adish Singla, and Andreas Krause. Selecting sequences of items via submodular maximization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 31, 2017.

[24] Ruosong Wang, Hanrui Zhang, Devendra Singh Chaplot, Denis Garagi´c, and Ruslan Salakhutdinov. Planning with submodular objective functions. arXiv preprint arXiv:2010.11863, 2020.

[25] Zhenliang Zhang, Edwin K. P. Chong, Ali Pezeshki, and William Moran. String submodular functions with curvature constraints. IEEE Transactions on Automatic Control, 61(3):601–616, 2016.

## Appendix

## A CORRELATED POLICIES

The class $\Pi _ { H }$ uses the product action distributions in (2) with $m = H$ . We show that allowing joint action distributions that do not factor in this way does not improve the optimal expected reward. The following lemma applies the standard existence result for deterministic optimal policies in finite-horizon MDPs [18, Proposition 4.4.3] to the augmented state $( s _ { t } , h _ { t } ^ { H } )$ .

Lemma 2. There exists a deterministic policy $\pi ^ { * } \in \Pi _ { H }$ that is optimal among all global history-dependent joint policies, including those with correlated actions. Consequently, allowing correlated action distributions does not increase $J ^ { * }$

Proof. At each time $t \in \{ 0 , \ldots , H - 1 \}$ , allow an arbitrary joint action distribution

$$
q _ { t } : \mathbf { S } \times \pmb { \mathcal { H } } _ { t } ^ { H }  \mathcal { D } ( \mathbf { A } ) .
$$

For every joint history $( s _ { t } , h _ { t } ^ { H } )$ , this means that

$$
q _ { t } ( a \mid s _ { t } , h _ { t } ^ { H } ) \geq 0 \quad { \mathrm { f o r ~ a l l ~ } } a \in \mathbf { A } , \qquad \sum _ { a \in \mathbf { A } } q _ { t } ( a \mid s _ { t } , h _ { t } ^ { H } ) = 1 .
$$

The joint action distribution $q _ { t }$ does not necessarily factor across agents as in (2). Taking $( s _ { t } , h _ { t } ^ { H } )$ as the state yields a finite-horizon MDP. The transition dynamics of this $\mathrm { M D P }$ draw the next joint state from $\mathbf { P } _ { t } ( \cdot \mid s _ { t } , a _ { t } )$ and update the history by appending $\mathbf { \boldsymbol { s } } _ { t }$ . Its reward is zero at intermediate stages and equals $F ( \tau )$ at time $H$ when the augmented state determines the full joint trajectory.

Fix optimal policies at time steps $t + 1 , \ldots , H - 1$ . At each joint state and history, the expected return is linear in $q _ { t } ( \cdot \mid s _ { t } , h _ { t } ^ { H } )$ . Its maximum over the probability simplex $\mathcal { D } ( \mathbf { A } )$ is therefore attained at a vertex, which assigns probability one to a single joint action. Backward induction thus yields a deterministic joint policy that is optima under any initial distribution, including $\rho .$

Let $\pmb { a } _ { t } ^ { * } ( \pmb { s } _ { t } , \pmb { h } _ { t } ^ { H } )$ denote the joint action selected by backward induction, and let $a _ { i , t } ^ { * } ( s _ { t } , h _ { t } ^ { H } )$ denote its i-th component. For each agent $i \in [ N ]$ , define

$$
\pi _ { i , t } ^ { * } ( a _ { i } \mid \boldsymbol { s } _ { t } , \boldsymbol { h } _ { t } ^ { H } ) = \boldsymbol { 1 } \Big \{ a _ { i } = a _ { i , t } ^ { * } \big ( \boldsymbol { s } _ { t } , \boldsymbol { h } _ { t } ^ { H } \big ) \Big \} .
$$

The product of these policies assigns probability one to the selected joint action. Hence, this optimal policy belongs to $\Pi _ { H }$ , and allowing correlated action distributions does not increase $J ^ { * }$ □

## B PROPERTIES OF PRE-WINDOW CURVATURE

Figure 4 illustrates the m-step window and the pre-window part on the trajectories of two agents.

## B.1 Properties of the Pre-Window Curvature

Proposition 1. For every $m \in \{ 0 , \ldots , H \}$

i. $0 \leq c _ { m } \leq c \leq 1$ ;

ii. $c _ { m + 1 } \leq c _ { m }$ whenever $m < H$ ;

![](images/815c0d2063baa7488aed906bb556882707d7bf7bb59be325dc7c4d46b23de438.jpg)  
Figure 4: The m-step window moving from time t to time $t + 1$ . Each row is the trajectory of one agent, and the arrows are transitions under its dynamics. (a) At time $t ,$ the window covers times t − m to t − 1, and earlier visits form the pre-window part. (b) After the transition to time $t + 1$ , the window shifts by one step, and the visits at time $t - m$ join the pre-window part. Pre-window curvature $c _ { m }$ bounds how much pre-window visits, of any agent, reduce the marginal gain of a current visit.

iii. $c _ { H } = 0 ;$

iv. if F is additive across time, then $c _ { m } = 0$ for every m.

Proof. i. We first show $c _ { m } \geq 0$ and $c \leq 1$ , and then show $c _ { m } \leq c$ . Definition 4 restricts the minimum to $x _ { t }$ and Z with $\Delta ( x _ { t } \mid Z ) > 0$ , so every ratio is well defined. Since $Z \subseteq B \cup Z$ , submodularity gives $\Delta ( x _ { t } \mid B \cup Z ) \leq \Delta ( x _ { t } \mid Z )$ , so every ratio is at most 1 and $c _ { m } \geq 0$ . In Definition 2, monotonicity makes the numerator of every ratio nonnegative, so every ratio is at least 0 and $c \leq 1$

To show $c _ { m } \leq c ,$ we prove that every ratio in (5) is at least $1 - c .$ . Fix t, $x _ { t } \in \mathcal { V } _ { t } , B \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ , and $Z \subseteq \nu$ with $\Delta ( x _ { t } \mid Z ) > 0$ . The element $x _ { t }$ does not lie in $Z ,$ , since otherwise $\Delta ( x _ { t } \mid Z ) = 0$ . It does not lie in B either, since every element of B has a time index $l < t - m \leq t$ . Moreover, submodularity with $\varnothing \subseteq Z$ and $F ( \varnothing ) = 0$ gives

$$
F ( \{ x _ { t } \} ) = \Delta ( x _ { t } \mid \emptyset ) \geq \Delta ( x _ { t } \mid Z ) > 0 .
$$

Hence $x _ { t } \in \mathcal { V } \setminus \left( B \cup Z \right)$ and $F ( \{ x _ { t } \} ) > 0$ . The ratio $\Delta ( x _ { t } \mid B \cup Z ) / F ( \{ x _ { t } \} )$ ) is therefore one of the ratios minimized in Definition 2, so it is at least $1 - c .$ Since $1 - c \geq 0$ and $F ( \{ x _ { t } \} ) \geq \Delta ( x _ { t } \mid Z )$ , we obtain

$$
\frac { \Delta ( x _ { t } \mid B \cup Z ) } { \Delta ( x _ { t } \mid Z ) } \geq ( 1 - c ) \frac { F ( \{ x _ { t } \} ) } { \Delta ( x _ { t } \mid Z ) } \geq 1 - c .
$$

The minimum in (5) is therefore at least $1 - c ,$ so $c _ { m } \leq c .$

ii. Since $\mathcal { V } _ { 0 : t - m - 2 } \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ , replacing m by m + 1 shrinks the set of admissible B while the rest of the minimization is unchanged. The minimum over a smaller set can only increase, so $c _ { m + 1 } \leq c _ { m }$

iii. For $m = H$ and any $t \in \{ 0 , \ldots , H \}$ , we have $t - H \leq 0$ , so $\mathcal { V } _ { 0 : t - H - 1 } = \emptyset$ and $B = \varnothing$ . Every ratio then equals 1, and $c _ { H } = 0 .$

iv. Suppose $\begin{array} { r } { F ( X ) = \sum _ { t = 0 } ^ { H } f _ { t } ( X \cap \mathcal { V } _ { t } ) } \end{array}$ . Adding $x _ { t } \in \mathcal V _ { t }$ changes only the term $f _ { t }$ , which depends on $X \cap \mathcal { V } _ { t }$ Since $B \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ is disjoint from $\nu _ { t }$ , we have $( B \cup Z ) \cap \mathcal { V } _ { t } = Z \cap \mathcal { V } _ { t }$ . Hence $\Delta ( x _ { t } \mid B \cup Z ) = \Delta ( x _ { t } \mid Z )$ every ratio equals one, and $c _ { m } = 0$

## C TRAFFIC-MONITORING REWARD PROPERTIES

Proposition 2 gives suficient conditions on $\alpha _ { \ell }$ under which the reward (4) of Section 2.4 is monotone submodular and has $c _ { m } < 1$ . Corollary 1 shows that the exponential profile $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ satisfies these conditions, so the trafic-monitoring reward is monotone submodular. Proposition 3 then computes $c _ { m }$ in closed form for this profile

## C.1 Discretely Concave Recovery Profiles

We call the sequence ${ \pmb \alpha } = ( \alpha _ { 0 } , \ldots , \alpha _ { H } )$ the recovery profile. In this subsection, α is general rather than $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ . We call α discretely concave if

$$
\alpha _ { \ell + 1 } - \alpha _ { \ell } \le \alpha _ { \ell } - \alpha _ { \ell - 1 } , \qquad 1 \le \ell < H .\tag{13}
$$

Each passing time step therefore recovers no more than the step before it. The concavity is strict when every inequality is strict.

Proposition 2. Suppose $\alpha _ { 0 } = 0 , \alpha _ { \ell } \in [ 0 , 1 ]$ for $1 \leq \ell \leq H$ , and α is nondecreasing and discretely concave. Then the reward in (4) is monotone submodular. $I f \alpha _ { 1 } > 0$ and the concavity is strict, then $c _ { m } < 1$ for every $m < H$

Proof. Fix $B \subseteq \nu$ and consider adding $( t , s )$ . The reward (4) is a sum over roads, and each term depends on B only through the monitoring times $T _ { e } ( B )$ . Adding $( t , s )$ leaves $T _ { e } ( B )$ unchanged for $e \not \in { \mathcal { C } } ( s )$ , and replaces it by $T _ { e } ( B ) \cup \{ t \}$ otherwise. Hence

$$
\Delta \big ( ( t , s ) \mid B \big ) = \sum _ { e \in \mathcal { C } ( s ) } \Delta _ { e } ( t \mid B ) ,
$$

where $\Delta _ { e } ( t \mid B )$ is road $e \mathrm { { s } }$ contribution. This contribution is zero if $t \in T _ { e } ( B )$ . Otherwise, let p and q be the nearest monitoring times in $T _ { e } ( B )$ before and after t, when they exist. If p exists, let $\ell _ { - } = t - p ;$ if q exists, let $\ell _ { + } = q - t$ . The four cases are

$$
\Delta _ { e } ( t \mid B ) = w _ { e } \left\{ \begin{array} { l l } { { 1 , } } & { { p , q \mathrm { ~ d o ~ n o t ~ e x i s t } , } } \\ { { \alpha _ { \ell _ { - } } , } } & { { \mathrm { o n l y ~ } p \mathrm { ~ e x i s t s } , } } \\ { { \alpha _ { \ell _ { + } } , } } & { { \mathrm { o n l y ~ } q \mathrm { ~ e x i s t s } , } } \\ { { \alpha _ { \ell _ { - } } + \alpha _ { \ell _ { + } } - \alpha _ { \ell _ { - } + \ell _ { + } } , } } & { { p , q \mathrm { ~ b o t h ~ e x i s t } . } } \end{array} \right.\tag{14}
$$

When both $p$ and q exist, adding $( t , s )$ replaces the gap $\ell _ { - } + \ell _ { + }$ with the gaps $\ell _ { - }$ and $\ell _ { + }$ .

Since the increments of α are nonincreasing by (13), $\alpha _ { \ell _ { - } + \ell _ { + } } - \alpha _ { \ell _ { + } } \leq \alpha _ { \ell _ { - } } - \alpha _ { 0 } = \alpha _ { \ell _ { - } }$ . Moreover, α is nondecreasing, so $\alpha _ { \ell _ { - } + \ell _ { + } } \geq \operatorname* { m a x } \{ \alpha _ { \ell _ { - } } , \alpha _ { \ell _ { + } } \}$ . These inequalities give

$$
0 \leq \alpha _ { \ell _ { - } } + \alpha _ { \ell _ { + } } - \alpha _ { \ell _ { - } + \ell _ { + } } \leq \operatorname* { m i n } \{ \alpha _ { \ell _ { - } } , \alpha _ { \ell _ { + } } \} .\tag{15}
$$

The lower bound shows that the contribution in the last case of (14) is nonnegative. The upper bound shows that it is no greater than the corresponding contribution when only p or only q exists.

The four cases in (14) difer only in which neighboring monitoring times exist. The convention $\alpha _ { \infty } = 1$ of Section 2.4 removes this distinction. Set $\ell _ { - } = \infty$ when $p$ does not exist and $\ell _ { + } = \infty$ when $q$ does not exist, so that $\alpha _ { \ell _ { - } + \ell _ { + } } = 1$ whenever a gap is infinite. Every case of (14) then reads

$$
\Delta _ { e } ( t \mid B ) = w _ { e } h ( \ell _ { - } , \ell _ { + } ) , \qquad h ( \ell _ { 1 } , \ell _ { 2 } ) = \alpha _ { \ell _ { 1 } } + \alpha _ { \ell _ { 2 } } - \alpha _ { \ell _ { 1 } + \ell _ { 2 } } ,\tag{16}
$$

since $h ( \infty , \infty ) = 1 , h ( \ell _ { 1 } , \infty ) = \alpha _ { \ell _ { 1 } }$ , and $h ( \infty , \ell _ { 2 } ) = \alpha _ { \ell _ { 2 } }$ . The contribution of road e therefore depends on $B$ only through the two gaps.

The function h is nondecreasing in each argument over its domain. For finite $\ell _ { 1 } , \ell _ { 2 }$ with $\ell _ { 1 } + \ell _ { 2 } + 1 \leq H$

$$
h ( \ell _ { 1 } + 1 , \ell _ { 2 } ) - h ( \ell _ { 1 } , \ell _ { 2 } ) = ( \alpha _ { \ell _ { 1 } + 1 } - \alpha _ { \ell _ { 1 } } ) - ( \alpha _ { \ell _ { 1 } + \ell _ { 2 } + 1 } - \alpha _ { \ell _ { 1 } + \ell _ { 2 } } ) \geq 0
$$

by discrete concavity. When $\ell _ { 2 } = \infty , h ( \ell _ { 1 } + 1 , \ell _ { 2 } ) - h ( \ell _ { 1 } , \ell _ { 2 } ) \geq 0$ follows from $h ( \ell _ { 1 } , \infty ) = \alpha _ { \ell } .$ and the monotonicity of α. Moreover, $h ( \ell _ { 1 } , \ell _ { 2 } ) \leq \alpha _ { \ell _ { 2 } } = h ( \infty , \ell _ { 2 } )$ , since $\alpha _ { \ell _ { 1 } } \leq \alpha _ { \ell _ { 1 } + \ell _ { 2 } }$ . The argument for $\ell _ { 2 }$ is symmetric.

Monotonicity. By (15), $h \geq 0$ for finite gaps, and $h ( \ell _ { 1 } , \infty ) = \alpha _ { \ell _ { 1 } } \geq 0$ . Hence $\Delta ( ( t , s ) \mid B ) \geq 0$ , and F is monotone. Submodularity. Let $B \subseteq B ^ { \prime }$ and $( t , s ) \notin B ^ { \prime }$ , and fix $e \in { \mathcal { C } } ( s )$ . If $t \in T _ { e } ( B ^ { \prime } )$ , then $\Delta _ { e } ( t \mid B ^ { \prime } ) = 0 \le \Delta _ { e } ( t \mid B )$ Otherwise $t \notin T _ { e } ( B )$ as well, since $T _ { e } ( B ) \subseteq T _ { e } ( B ^ { \prime } )$ . Let $( \ell _ { - } , \ell _ { + } )$ and $( \ell _ { - } ^ { \prime } , \ell _ { + } ^ { \prime } )$ be the gaps under $T _ { e } ( B )$ and $T _ { e } ( B ^ { \prime } )$ . Additional monitoring times can only move the nearest monitoring time on each side closer to $t ,$ and can only turn an absent neighbor into a present one, so $\ell _ { - } ^ { \prime } \leq \ell _ { - }$ and $\ell _ { + } ^ { \prime } \leq \ell _ { + }$ . By (16) and the monotonicity of $h ,$

$$
\Delta _ { e } ( t \mid B ^ { \prime } ) = w _ { e } h ( \ell _ { - } ^ { \prime } , \ell _ { + } ^ { \prime } ) \leq w _ { e } h ( \ell _ { - } , \ell _ { + } ) = \Delta _ { e } ( t \mid B ) .
$$

Summing over $e \in { \mathcal { C } } ( s )$ proves that $F$ is submodular.

Curvature. Suppose $\alpha _ { 1 } > 0$ and the concavity is strict. Since α is nondecreasing, $\alpha _ { \ell } \geq \alpha _ { 1 } > 0$ for every positive gap ℓ. For positive finite gaps $\ell _ { 1 } , \ell _ { 2 }$ with $\ell _ { 1 } + \ell _ { 2 } \leq H$ , strict discrete concavity gives $\alpha _ { \ell _ { 1 } + \ell _ { 2 } } - \alpha _ { \ell _ { 2 } } < \alpha _ { \ell _ { 1 } } - \alpha _ { 0 } = \alpha _ { \ell _ { 1 } }$ Hence the contribution when both p and q exist is also positive.

Consider a ratio in (5) with $\Delta ( x _ { t } \mid Z ) > 0$ , where $x _ { t } = ( t , s )$ . At least one road $e \in { \mathcal { C } } ( s )$ has a positive contribution under Z. Elements of B occur before time $t - m .$ , so they cannot add t to the monitoring times of road $e . \ \mathrm { B y \ ( 1 4 ) }$ this road still has a positive contribution under $B \cup Z$ . Thus, the numerator is positive whenever the denominator is positive. The number of admissible ratios is finite, so their minimum is positive. Therefore, $c _ { m } < 1$ □

Remark. Non-strict discrete concavity alone does not guarantee $c _ { m } < 1$ , even when $\alpha _ { 1 } > 0$ . Fix $m \le H - 2$ and suppose

$$
\alpha _ { \ell + 1 } - \alpha _ { \ell } = \alpha _ { 1 } > 0 , \qquad 0 \le \ell \le m + 1 .
$$

Set $t = m + 1 , Z = \{ ( t + 1 , s ) \}$ , and $B = \{ ( 0 , s ) \}$ , where s monitors at least one road with $w _ { e } > 0$ . Since $t - m = 1$ $B \subseteq \mathcal { V } _ { 0 }$ is admissible. The marginal gain under Z is positive. Under $B \cup Z$ , the contribution of each road in $\mathcal C ( s )$ is

$$
w _ { e } \big ( \alpha _ { m + 1 } + \alpha _ { 1 } - \alpha _ { m + 2 } \big ) = 0 .
$$

Hence the ratio in (5) is zero, and $c _ { m } = 1$

For the exponential profile of Section 2.4, Proposition 2 gives the following result.

Corollary 1. For every $\beta > 0$ , the trafic-monitoring reward (4) is monotone submodular.

Proof. The exponential profile $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ satisfies $\alpha _ { 0 } = 0$ , is nondecreasing, and takes values in [0, 1]. Its increments

$$
\alpha _ { \ell + 1 } - \alpha _ { \ell } = e ^ { - \beta \ell } { \left( 1 - e ^ { - \beta } \right) }
$$

are strictly decreasing in $\ell ,$ so α is strictly discretely concave. Proposition 2 gives the claim.

## C.2 Pre-Window Curvature of the Exponential Profile

Proposition 3. For every $\beta > 0$ , the trafic-monitoring reward (4) has pre-window curvature

$$
c _ { m } = e ^ { - \beta ( m + 1 ) } , \qquad 0 \leq m < H .\tag{17}
$$

Proof. We first show that every ratio $\Delta ( x _ { t } \mid B \cup Z ) / \Delta ( x _ { t } \mid Z )$ in (5) is at least $\alpha _ { m + 1 }$ , and then show that this bound is attained. Fix $x _ { t } = ( t , s )$ and a road $e \in \mathcal { C } ( s )$ . Let ℓ<sub>−</sub> and $\ell _ { + }$ be the gaps to the nearest monitoring times of e in Z before and after t, and let $\ell _ { B }$ be the gap to the nearest monitoring time of e in B. Set a gap to infinity when the corresponding monitoring time does not exist. Since B contains only elements before time $t - m , \ell _ { B } \geq m + 1$ . Thus, adding B changes ℓ to min $\{ \ell _ { - } , \ell _ { B } \}$ and leaves $\ell _ { + }$ unchanged. Figure 5 illustrates these monitoring times.

For finite gaps $\ell _ { - }$ and $\ell _ { + }$ , the exponential profile satisfies

$$
\begin{array} { r } { \alpha _ { \ell _ { - } } + \alpha _ { \ell _ { + } } - \alpha _ { \ell _ { - } + \ell _ { + } } = ( 1 - e ^ { - \beta \ell _ { - } } ) ( 1 - e ^ { - \beta \ell _ { + } } ) = \alpha _ { \ell _ { - } } \alpha _ { \ell _ { + } } . } \end{array}
$$

With the convention $\alpha _ { \infty } = 1$ , the four cases of (14) for $t \notin T _ { e } ( Z )$ can therefore be written as

$$
\Delta _ { e } ( t \mid Z ) = w _ { e } \alpha _ { \ell _ { - } } \alpha _ { \ell _ { + } } .\tag{18}
$$

![](images/ff478a436e17b40c79c56c10a8811b18eadbb694d99a1dcd86c1100928bebf12.jpg)  
Figure 5: Monitoring times of road e around the added element $x _ { t } = ( t , s )$ . Filled circles denote monitoring times from $Z ,$ the open square denotes the nearest monitoring time from B, and the cross denotes $x _ { t }$ . Elements of B occur before time $t - m$ , so adding B changes the earlier $\mathrm { g a p }$ from $\ell _ { - }$ to min $\{ \ell _ { - } , \ell _ { B } \}$ and leaves the later $\exp { \ell _ { + } }$ unchanged.

Consider any admissible $( t , x _ { t } , B , Z )$ in Definition 4, so that $\Delta ( x _ { t } \mid Z ) > 0$ . For each road with $\Delta _ { e } ( t \mid Z ) > 0$ , (18) gives

$$
\frac { \Delta _ { e } ( t \mid B \cup Z ) } { \Delta _ { e } ( t \mid Z ) } = \frac { \alpha _ { \operatorname* { m i n } \{ \ell _ { - } , \ell _ { B } \} } } { \alpha _ { \ell _ { - } } } = \left\{ 1 , \quad \quad \ell _ { B } \geq \ell _ { - } , \right.
$$

Both cases are at least $\alpha _ { m + 1 } \mathrm { : }$ the first follows from $\alpha _ { m + 1 } \leq 1$ , and the second from $\alpha _ { \ell _ { - } } \leq 1$ and $\alpha _ { \ell _ { B } } \geq \alpha _ { m + 1 }$ To extend this bound to the complete marginal gain, let $E ^ { + } = \{ e \in \mathcal { C } ( s ) : \Delta _ { e } ( t \mid Z ) > 0 \}$ . A road with zero contribution under $Z$ also has zero contribution under $B \cup Z$ . Therefore,

$$
\begin{array} { r l } { \frac { \Delta \left( x _ { t } \mid B \cup Z \right) } { \Delta \left( x _ { t } \mid Z \right) } = \frac { \sum _ { e \in E ^ { + } } \Delta _ { e } \left( t \mid B \cup Z \right) } { \sum _ { e \in E ^ { + } } \Delta _ { e } \left( t \mid Z \right) } } & { } \\ { \quad \geq \frac { \displaystyle \operatorname* { m i n } } { \displaystyle e \in E ^ { + } } \frac { \Delta _ { e } \left( t \mid B \cup Z \right) } { \Delta _ { e } \left( t \mid Z \right) } \geq \alpha _ { m + 1 } . } \end{array}\tag{19}
$$

Hence every admissible ratio is at least $\alpha _ { m + 1 }$ , and $c _ { m } \leq 1 - \alpha _ { m + 1 }$

For $m < H$ , set $t = m + 1 , Z = \emptyset$ , and $B = \{ ( 0 , s ) \}$ , where s monitors at least one road with $w _ { e } > 0$ . Since $t - m = 1$ , the set $B \subseteq \mathcal { V } _ { 0 }$ is admissible. Under $Z , x _ { t } = ( t , s )$ is the first monitoring of every road in $\mathcal C ( s )$ , whereas under B it follows the previous monitoring by $m + 1$ time steps. Thus

$$
{ \frac { \Delta ( x _ { t } \mid B ) } { \Delta ( x _ { t } \mid Z ) } } = \alpha _ { m + 1 } .
$$

The lower bound is attained, so

$$
c _ { m } = 1 - \alpha _ { m + 1 } = e ^ { - \beta ( m + 1 ) } .
$$

At $m = H$ , no visit can occur before the H-step window, and $c _ { H } = 0$

## D ANALYSIS OF THE WINDOWED OBJECTIVE

## D.1 Set Form of the Pre-Window Curvature Inequality

Definition 4 bounds the efect of visits prior to the window on the marginal gain of a single visit $x _ { t } \in \mathcal V _ { t }$ given a set Z. The proof of Lemma 1 needs the same bound for a set of visits at time $t ,$ so we extend it to any set $X _ { t } \subseteq \mathcal { V } _ { t }$

Lemma 3. For every $m , t \in \{ 0 , \ldots , H \} , X _ { t } \subseteq \mathcal { V } _ { t } , B \subseteq \mathcal { V } _ { 0 : t - m - 1 }$ , and $Z \subseteq \nu$

$$
\Delta \left( X _ { t } \mid B \cup Z \right) \ \geq \ ( 1 - c _ { m } ) \Delta \left( X _ { t } \mid Z \right) .
$$

Proof. Elements of $X _ { t } \cap Z$ add zero marginal gain on both sides, so it sufices to consider the elements of $X _ { t } \setminus Z .$ enumerated as $x _ { 1 } , \ldots , x _ { q } .$ . No element of $X _ { t }$ lies in $B ,$ since $X _ { t } \subseteq \mathcal { V } _ { t }$ while B contains only elements of earlier

times. Let $Z _ { j } = Z \cup \{ x _ { 1 } , \dotsc , x _ { j - 1 } \}$ collect the elements added before $x _ { j }$ . Both marginal gains then decompose into single-element gains whose j-th terms difer only in the presence of the set $B \colon$

$$
\Delta ( X _ { t } \mid B \cup Z ) = \sum _ { j = 1 } ^ { q } \Delta ( x _ { j } \mid B \cup Z _ { j } ) ,
$$

$$
\Delta ( X _ { t } \mid Z ) = \sum _ { j = 1 } ^ { q } \Delta ( x _ { j } \mid Z _ { j } ) .
$$

Each pair of terms is an instance of (5) with $x _ { t } = x _ { j }$ , the same $B ,$ and $Z = Z _ { j }$ . Hence $\Delta ( x _ { j } \mid B \cup Z _ { j } ) \geq$ $( 1 - c _ { m } ) \Delta ( x _ { j } \mid Z _ { j } )$ . When $\Delta ( x _ { j } \mid Z _ { j } ) = 0$ , submodularity and $Z _ { j } \subseteq B \cup Z _ { j }$ force $\Delta ( x _ { j } \mid B \cup Z _ { j } ) = 0 .$ , so the inequality holds trivially. Summing over j proves the lemma. □

## D.2 Proofs of Lemma 1 and Theorem 1

We first prove Lemma 1, restated below.

Lemma 1. For every policy π,

$$
( 1 - c _ { m } ) \Phi _ { m } ( \pmb { \pi } ) \leq J ( \pmb { \pi } ) \leq \Phi _ { m } ( \pmb { \pi } ) .
$$

Proof. Fix a realized joint trajectory τ . Telescoping its marginal gains in chronological order gives

$$
F ( \pmb { \tau } ) = \sum _ { t = 0 } ^ { H } \Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { 0 : t - 1 } ) .
$$

For every $t ,$ the full past decomposes as $\pmb { \tau } _ { 0 : t - 1 } = \pmb { \tau } _ { 0 : t - m - 1 } \cup \pmb { \tau } _ { t - m : t - 1 }$ , where $\tau _ { 0 : t - m - 1 }$ is empty when $t \leq m$ . For the lower bound, we apply Lemma 3 with $X _ { t } = \pmb { \tau } _ { t } \subseteq \mathscr { V } _ { t } , B = \pmb { \tau } _ { 0 : t - m - 1 } \subseteq \mathscr { V } _ { 0 : t - m - 1 }$ , and $Z = \tau _ { t - m : t - 1 }$ . Since $B \cup Z = \tau _ { 0 : t - 1 }$ , this gives

$$
\Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { 0 : t - 1 } ) \geq ( 1 - c _ { m } ) \Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { t - m : t - 1 } ) .
$$

For the upper bound, submodularity and $\pmb { \tau } _ { t - m : t - 1 } \subseteq \pmb { \tau } _ { 0 : t - 1 }$ give

$$
\Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { 0 : t - 1 } ) \leq \Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { t - m : t - 1 } ) .
$$

Summing both bounds over t and taking expectations under π proves the lemma. The left-hand sides give $J ( \pi )$ by the telescoping identity above, and the terms $\Delta ( \pmb { \tau } _ { t } \mid \pmb { \tau } _ { t - m : t - 1 } ) = r _ { t } ( \tilde { \pmb { s } } _ { t } ) \mathrm { ~ g i v e ~ } \Phi _ { m } ( \pmb { \pi } )$ □

We then prove Theorem 1, restated below.

Theorem 1. Let $\pi ^ { \prime } \in$ arg $\operatorname* { m a x } _ { \pi \in \Pi _ { H } } \Phi _ { m } ( \pi )$ . Then

$$
J ( \pi ^ { \prime } ) \geq \left( 1 - c _ { m } \right) J ^ { * } .
$$

Proof. Applying Lemma 1 to $\pi ^ { * }$ and $\pi ^ { \prime }$ , and using the optimality of $\pi ^ { \prime }$ for $\Phi _ { m } .$ gives

$$
\begin{array} { r l } & { ( 1 - { c } _ { m } ) J ^ { * } \leq ( 1 - { c } _ { m } ) \Phi _ { m } ( \pi ^ { * } ) } \\ & { \qquad \leq ( 1 - { c } _ { m } ) \Phi _ { m } ( \pi ^ { \prime } ) \leq J ( \pi ^ { \prime } ) . } \end{array}
$$

The guarantee does not require an exact maximizer of $\Phi _ { m } .$ . Suppose that a policy π satisfies $\Phi _ { m } ( \pmb { \pi } ) \geq \Phi _ { m } ( \pmb { \pi } ^ { \prime } ) - \eta$ for some $\eta \geq 0$ . Combining the first two steps above with the lower bound of Lemma 1 for π gives

$$
( 1 - c _ { m } ) J ^ { * } \leq ( 1 - c _ { m } ) \Phi _ { m } ( \pi ^ { \prime } ) \leq ( 1 - c _ { m } ) \big ( \Phi _ { m } ( \pi ) + \eta \big ) \leq J ( \pi ) + ( 1 - c _ { m } ) \eta ,
$$

so $J ( \pmb { \pi } ) \geq ( 1 - c _ { m } ) ( J ^ { * } - \eta )$

![](images/a5056dc3a7e1905f343650d9b3500c70d2cbe7b9bfa095917b06a6c13ccfe0b2.jpg)

Figure 6: History-augmented state of agent i with memory length $m ,$ for $t \geq m$ . The augmented state $\tilde { s } _ { i , t }$ contains the history $h _ { i , t }$ (blue) and the current state $s _ { i , t } \ ( \mathrm { o r a n g e } )$ . The next state $s _ { i , t + 1 }$ is drawn from $P _ { i , t } ,$ , and the history appends $s _ { i , t }$ and drops the oldest state $s _ { i , t - m } ~ ( \mathrm { d a s h e d } )$  
Algorithm 2 Backward Induction on the Augmented State Space   
Require: spaces $s , A ;$ memory length m; dynamics $P = ( P _ { 0 } , \dots , P _ { H - 1 } ) ;$ ; rewards $r _ { t }$   
1: $V _ { H } ( \tilde { s } )  r _ { H } ( \tilde { s } ) \quad \forall \tilde { s } \in \tilde { S } _ { H }$   
2: for ${ t = H - 1 , \ldots , 0 }$ do   
3: for all $\tilde { s } = ( s , h ) \in \tilde { S } _ { t }$ and $a \in { \mathcal { A } }$ do   
4: $Q _ { t } ( \tilde { s } , a ) \gets r _ { t } ( \tilde { s } ) + \mathbb { E } _ { s ^ { \prime } \sim P _ { t } ( \cdot \vert s , a ) } \big [ V _ { t + 1 } ( s ^ { \prime } , \varphi ( h , s ) ) \big ]$   
5: end for   
6: $V _ { t } ( \tilde { s } ) \gets \operatorname* { m a x } _ { a } Q _ { t } ( \tilde { s } , a ) \quad \forall \tilde { s } \in \tilde { \mathcal { S } } _ { t }$   
7: $\pi _ { t } ( \tilde { s } ) \in \arg \operatorname* { m a x } _ { a } Q _ { t } ( \tilde { s } , a ) \ \quad \forall \tilde { s } \in \tilde { S } _ { t }$   
8: end for   
9: return $\pi = ( \pi _ { 0 } , \ldots , \pi _ { H - 1 } )$

## D.3 Backward Induction on the Augmented State Space

Algorithm 2 solves a finite-horizon MDP on the history-augmented states by backward induction [18]. The expectation is over the next state, while the history updates deterministically through the map $\varphi : \mathcal { H } _ { t } ^ { m } \times \mathcal { S }  \mathcal { H } _ { t + 1 } ^ { m } .$ which appends a state to a history and drops the first entry if the result has more than m entries. Figure 6 illustrates this update for one agent. Section 4.2 applies the algorithm to the joint augmented states, with S and A replaced by S and A. Section 4.3 applies it once per agent, with the estimated rewards $\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$

Backward induction on the joint augmented MDP computes a Q-value for each of the at most $| S | ^ { N ( m + 1 ) } | A | ^ { N }$ pairs in $\tilde { \mathbf { S } } _ { t } \times \mathbf { A }$ at every stage. Each Q-value sums over at most $| S | ^ { N }$ next joint physical states, since the histories update deterministically through $\varphi .$ Over H stages, the total computation time is $O ( H | S | ^ { N ( m + 2 ) } | A | ^ { N } )$

## D.4 Non-Monotonicity of the Windowed Stage Reward

Adding an agent can decrease the windowed objective even when F is monotone submodular and $c _ { m } < 1$ Consider a reward of the form (4) on two states a and b with a single road e of maximum importance $w _ { e } = 1$ where $\mathcal { C } ( a ) = \{ e \}$ and ${ \mathcal { C } } ( b ) = \emptyset$ . Set $H = 2 , m = 1$ , and $\beta = \ln ( 4 / 3 )$ , so that $\alpha _ { 1 } = 1 / 4$ and $\alpha _ { 2 } = 7 / 1 6$ . By Corollary 1, the reward is monotone submodular, and (17) gives $c _ { 1 } = 9 / 1 6$ . Each agent can move between a and b deterministically, and two agents with fixed initial states follow the trajectories

$$
\tau _ { 1 } = \{ ( 0 , a ) , ( 1 , b ) , ( 2 , a ) \} , \qquad \tau _ { 2 } = \{ ( 0 , b ) , ( 1 , a ) , ( 2 , b ) \} ,
$$

shown in Figure 7. Since the dynamics and initial states are deterministic, the windowed objective of the policies that generate these trajectories equals the sum of the stage rewards $r _ { t } ( \tilde { \boldsymbol { s } } _ { t } )$ along them.

With agent 1 alone, its visit at time 0 lies outside the window at time 2. The stage rewards are then $r _ { 0 } ( \tilde { s } _ { 0 } ) = 1$ $r _ { 1 } ( \tilde { s } _ { 1 } ) = 0$ , and $r _ { 2 } ( \tilde { s } _ { 2 } ) = 1$ , which sum to 2. With both agents, road e is monitored at every time. The stage rewards are then $r _ { 0 } ( \tilde { \boldsymbol { s } } _ { 0 } ) = 1 , r _ { 1 } ( \tilde { \boldsymbol { s } } _ { 1 } ) = \alpha _ { 1 }$ , and $r _ { 2 } ( \tilde { s } _ { 2 } ) = \alpha _ { 1 }$ , which sum to $3 / 2$ . Adding agent 2 thus lowers the stage reward at time 2 from 1 to $1 / 4$ , because its visit (1, a) enters the window at time 2. The true reward instead increases, from $F ( \tau _ { 1 } ) = 1 + \alpha _ { 2 } = 2 3 / 1 6$ to $F ( \tau _ { 1 } \cup \tau _ { 2 } ) = 3 / 2$ . Hence, the stage rewards of $\Phi _ { m }$ are not monotone in the set of agents, and the condition required by the guarantee of Chen et al. [4] fails.

![](images/e63d72190cae0fe19dfc9c03cb4322092bc8809072a80c3110bf4b4a0ec54fa4.jpg)

![](images/1fd5d48c97d0fc3a46b1cd39552822fdd654fc4bcc258127e7a5ac06e3986649.jpg)  
(a) Agent 1 alone  
(b) Both agents  
Figure 7: Adding an agent lowers the windowed objective in the example of Appendix D.4. Shaded nodes are visits to state $^ { a , }$ which monitors road $e .$ The gray region denotes the window at time 2, and the values below the axis are the stage rewards, which sum to the windowed objective. (a) With agent 1 alone, the window at time 2 contains no visit to $a .$ (b) With both agents, the window at time 2 contains the visit $( 1 , a )$ of agent 2. This visit lowers the stage reward at time 2, although the true reward F increases.

## E ANALYSIS OF SEQUENTIAL POLICY OPTIMIZATION

We prove Theorem 2, restated below.

Theorem 2. Let $N \geq 2$ and fix $m \in \{ 0 , \ldots , H \}$ . For $\epsilon > 0$ and $\delta \in ( 0 , 1 )$ , choose the number of sampled predecessor trajectories K such that

$$
K \ge \frac { R ^ { 2 } } { 2 \epsilon ^ { 2 } } \ln \frac { 2 ( N - 1 ) ( H + 1 ) | S | ^ { m + 1 } } { \delta } ,\tag{11}
$$

where $R = \operatorname* { m a x } _ { v \in \mathcal { V } } F ( \{ v \} )$ is the largest reward of a single visit. Then, with probability at least $1 - \delta$ , the joint policy π¯ returned by Algorithm 1 satisfies

$$
J ( \bar { \pi } ) \geq \frac { 1 - c _ { m } } { 2 - c _ { m } } \left( J ^ { * } - 2 ( N - 1 ) ( H + 1 ) \epsilon \right) .\tag{12}
$$

Throughout this section, we use the following notation. For a joint policy π and a local policy $\pi _ { i }$ of agent $i ,$ let $d _ { i , t } ^ { \pi } , d _ { i , t } ^ { \pi _ { i } } \in \mathcal { D } ( \tilde { S } _ { t } )$ denote the distributions of the augmented state of agent i at time $t ,$

$$
d _ { i , t } ^ { \pi } ( \tilde { s } ) = \operatorname* { P r } _ { \tau \sim \pi } \big ( \tilde { s } _ { i , t } = \tilde { s } \big ) , \qquad d _ { i , t } ^ { \pi _ { i } } ( \tilde { s } ) = \operatorname* { P r } _ { \tau _ { i } \sim \pi _ { i } } \big ( \tilde { s } _ { i , t } = \tilde { s } \big ) ,
$$

for every $\tilde { s } \in \tilde { S } _ { t }$ . With this notation, the local objective (9) takes the equivalent form

$$
G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) = \sum _ { { t = 0 } } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \left[ r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) \right] .\tag{20}
$$

## E.1 Proof of Theorem 2

Proof. The argument has three steps. Step 1 bounds the optimality gap by the sum of the maximum loca objectives of the agents. $\mathrm { S t e p 2 }$ bounds each maximum by the attained local objective, up to the reward-estimation error. Step 3 bounds the attained local objectives by the return $J ( \bar { \pi } )$ through $c _ { m }$

The policy π¯ returned by Algorithm 1 depends on the sampled predecessor trajectories. Throughout the proof, we treat π¯ as fixed. Since each $\bar { \pi } _ { i }$ is deterministic, an expectation over $\tau \sim \bar { \pi }$ is then taken over the initial states and the transitions only.

Step 1 (bounding the optimality gap by the maximum local objectives). For $\tau ^ { * } \sim \pi ^ { * }$ and $\tau \sim \bar { \pi }$ sampled independently, with $\pi ^ { * }$ an optimal global history-dependent policy of (3) and $\bar { \pmb { \pi } } \in \Pi _ { m } ^ { \mathrm { l o c } }$ the policy returned by Algorithm 1, we have

$$
\begin{array} { r l } & { J ^ { * } = \mathbb { E } _ { \tau ^ { * } \sim \pi ^ { * } } [ F ( \tau ^ { * } ) ] \le \mathbb { E } _ { \tau ^ { * } \sim \pi ^ { * } , \tau \sim \bar { \pi } } [ F ( \tau ^ { * } \cup \tau ) ] = \mathbb { E } _ { \tau ^ { * } \sim \pi ^ { * } , \tau \sim \bar { \pi } } [ F ( \tau ) + \Delta ( \tau ^ { * } \mid \tau ) ] } \\ & { \phantom { x x x x x x x x x x x x x x } = \mathbb { E } _ { \tau \sim \bar { \pi } } [ F ( \tau ) ] + \mathbb { E } _ { \tau ^ { * } \sim \pi ^ { * } , \tau \sim \bar { \pi } } [ \Delta ( \tau ^ { * } \mid \tau ) ] } \\ & { \phantom { x x x x x x x x x x } = J ( \bar { \pi } ) + \mathbb { E } _ { \tau ^ { * } \sim \pi ^ { * } , \tau \sim \bar { \pi } } [ \Delta ( \tau ^ { * } \mid \tau ) ] , } \end{array}
$$

where the inequality holds by monotonicity, and the equalities use the definition of the marginal gain, linearity of expectation, and the definition of the return (3). For every fixed realization of τ and $\tau ^ { * }$ , the pairs $\tau _ { i , t } ^ { * }$ of $\tau ^ { * }$ are added to τ one at a time in lexicographic order of $( i , t )$ , meaning $\tau _ { [ i - 1 ] } ^ { * } \cup \tau _ { i , 0 : t - 1 } ^ { * }$ is the set of elements added before $\tau _ { i , t } ^ { * }$ . Telescoping these marginal gains gives

$$
\Delta ( \tau ^ { * } \mid \tau ) = \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \Delta \big ( \tau _ { i , t } ^ { * } \mid \tau \cup \tau _ { [ i - 1 ] } ^ { * } \cup \tau _ { i , 0 : t - 1 } ^ { * } \big ) ,
$$

where a term whose pair already lies in its given set, $\tau _ { i , t } ^ { * } \subseteq \tau \cup \tau _ { [ i - 1 ] } ^ { * } \cup \tau _ { i , 0 : t - 1 } ^ { * } ,$ contributes 0, since $\Delta ( x \mid B ) = 0$ whenever $x \in B$ . The set $\tau \cup \tau _ { [ i - 1 ] } ^ { * } \cup \tau _ { i , 0 : t - 1 } ^ { * }$ given in each term contains $\tau _ { i , t - m : t - 1 } ^ { * } \cup \tau _ { [ i - 1 ] }$ , because $\tau _ { [ i - 1 ] } \subseteq \tau$ and $\tau _ { i , t - m : t - 1 } ^ { * } \subseteq \tau _ { i , 0 : t - 1 } ^ { * }$ . By submodularity, the marginal gain of $\tau _ { i , t } ^ { * }$ does not decrease when the given set shrinks to $\tau _ { i , t - m : t - 1 } ^ { * } \cup \tau _ { [ i - 1 ] }$ , so

$$
\Delta ( \pmb { \tau } ^ { * } \mid \pmb { \tau } ) \leq \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \Delta \big ( \tau _ { i , t } ^ { * } \big | \tau _ { i , t - m : t - 1 } ^ { * } \cup \pmb { \tau } _ { [ i - 1 ] } \big ) .
$$

It remains to take the expectation of this bound over the two runs. Along the optimal run, the augmented state $\tilde { s } _ { i , t } ^ { * } = ( s _ { i , t } ^ { * } , h _ { i , t } ^ { * } )$ determines $\tau _ { i , t } ^ { * }$ and $\tau _ { i , t - m : t - 1 } ^ { * } = \nu _ { t } ( h _ { i , t } ^ { * } )$ , so each term depends on agent i’s trajectory only through $\tilde { s } _ { i , t } ^ { * }$ . Since $\tilde { s } _ { i , t } ^ { * } \sim d _ { i , t } ^ { \pi ^ { * } }$ is independent of the predecessors’ set $\tau _ { [ i - 1 ] }$ , the expectation of each term splits into an outer expectation over $\tilde { s } _ { i , t } ^ { * }$ and an inner expectation over $\tau _ { [ i - 1 ] } \colon$

$$
\begin{array} { r l } & { \mathbb { E } _ { \tau ^ { \star } \sim \pi ^ { \star } , \tau \sim \pi } \left[ \Delta \left( \tau _ { i , t } ^ { * } \middle \vert \tau _ { i , t - m : t - 1 } ^ { * } \cup \tau _ { [ i - 1 ] } \right) \right] = \mathbb { E } _ { \tilde { s } _ { i , t } ^ { * } \sim d _ { i , t } ^ { n ^ { \star } } } \left[ \mathbb { E } _ { \tau _ { [ i - 1 ] } \sim \pi _ { [ i - 1 ] } } \left[ \Delta \left( \left( t , s _ { i , t } ^ { * } \right) \middle \vert \ \nu _ { t } ( h _ { i , t } ^ { * } ) \cup \tau _ { [ i - 1 ] } \right) \right] \right] } \\ & { \qquad = \mathbb { E } _ { \tilde { s } _ { i , t } ^ { * } \sim d _ { i , t } ^ { n ^ { \star } } } \left[ \tau _ { i , t } ^ { \pi _ { [ i - 1 ] } } \left( \tilde { s } _ { i , t } ^ { * } \right) \right] , } \end{array}
$$

where the second equality applies the definition (8) of the reward. The occupancy $d _ { i , t } ^ { \pi ^ { * } }$ is induced under the global history-dependent policy $\pi ^ { * }$ , whereas the algorithm optimizes over local Markov policies. The following lemma closes this gap.

Lemma 4. Fix an agent i and any $\pmb { \pi } \in \Pi _ { H }$ . There is a policy $\pi _ { i } ^ { \prime \prime } \in \Pi _ { m } ^ { \mathrm { l o c } }$ with $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi }$ for every t.

Applying Lemma 4 to $\pi ^ { * }$ gives a local policy $\pi _ { i } ^ { \prime \prime } \in \Pi _ { m } ^ { \mathrm { l o c } }$ with the same distributions $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi ^ { * } }$ of the augmented state of agent i. By (20), it follows that $\begin{array} { r } { \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } ^ { * } \sim d _ { i , t } ^ { \pi ^ { * } } } \left[ r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ^ { * } ) \right] = G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ^ { \prime \prime } ) \le \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) } \end{array}$ Hence,

$$
J ^ { * } \leq J ( \bar { \pi } ) + \sum _ { i \in [ N ] } \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) .\tag{21}
$$

Step 2 (bounding the maximum local objectives by the attained ones). Hoefding’s inequality and a union bound over the at most $( N - 1 ) ( H + 1 ) | S | ^ { m + 1 }$ estimated rewards give the following guarantee.

Lemma 5. Under (11), with probability at least $1 - \delta$

$$
\left| \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) \right| \leq \epsilon
$$

for all $i \geq 2 , t$ , and $\tilde { s } \in \tilde { S } _ { t }$

For the rest of the proof, we assume that the bound of Lemma 5 holds for all $i \geq 2 , t ,$ and $\tilde { s } \in \tilde { S } _ { t }$ , which happens with probability at least $1 - \delta$ . The algorithm maximizes $\widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } }$ rather than $G _ { i } ^ { \bar { \pi } _ { { [ i - 1 ] } } }$ . The next lemma bounds the resulting gap.

Lemma 6. Let $\begin{array} { r } { \bar { \pi } _ { i } \in \arg \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } \widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) . \mathrm { ~ I f ~ } \left| \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) \right| \leq \epsilon } \end{array}$ for all t and $\tilde { s } \in \tilde { S } _ { t }$ , then

$$
G _ { i } ^ { \pi _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) \geq \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \log } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) - 2 ( H + 1 ) \epsilon .
$$

For $i = 1 , \bar { \pi } _ { 1 }$ attains the maximum exactly.

Applying Lemma 6, each term max $\mathsf { \Pi } _ { \tau _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } \left( \pi _ { i } \right)$ in (21) satisfies $\begin{array} { r } { \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) \leq G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) + 2 ( H + } \end{array}$ 1)ϵ for $i \geq 2$ , while $\begin{array} { r } { \operatorname { n a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) = G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) } \end{array}$ ) for $i = 1$ . Summing these bounds over the N agents, only the $N - 1$ agents with $i \geq 2$ carry the error term, so

$$
J ^ { * } \le J ( \bar { \pi } ) + \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) + 2 ( N - 1 ) ( H + 1 ) \epsilon .\tag{22}
$$

Step 3 (bounding the attained local objectives by the return). It remains to relate the sum of the attained local objectives to the achieved return $J ( \bar { \pi } )$ . The following lemma provides this connection and is the only step in which pre-window curvature enters.

Lemma 7. For the policy π¯ returned by Algorithm 1,

$$
( 1 - c _ { m } ) \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) \leq J ( \bar { \pi } ) \leq \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) .
$$

If $c _ { m } = 1$ , the claimed bound reads $J ( \bar { \pi } ) \geq 0$ , which holds because F is monotone with $F ( \varnothing ) = 0$ . Assume $c _ { m } < 1$ The lower bound in Lemma 7 gives

$$
J ( \bar { \pi } ) \geq ( 1 - c _ { m } ) \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) ,\tag{23}
$$

so $\textstyle \sum _ { i \in [ N ] } G _ { i } ^ { { \bar { \pi } } _ { [ i - 1 ] } } ( { \bar { \pi } } _ { i } ) \leq J ( { \bar { \pi } } ) / ( 1 - c _ { m } )$ . Substituting into (22),

$$
\begin{array} { c } { \displaystyle { J ^ { * } \le J ( \bar { \pi } ) + \frac { J ( \bar { \pi } ) } { 1 - c _ { m } } + 2 ( N - 1 ) ( H + 1 ) \epsilon } } \\ { \displaystyle { = \frac { 2 - c _ { m } } { 1 - c _ { m } } J ( \bar { \pi } ) + 2 ( N - 1 ) ( H + 1 ) \epsilon , } } \end{array}
$$

which rearranges to (12). Since the bound of Lemma 5 holds with probability at least $1 - \delta ,$ so does (12).

## E.2 Proof of Lemma 4

Lemma 4. Fix an agent i and any $\pmb { \pi } \in \Pi _ { H }$ . There is a policy $\pi _ { i } ^ { \prime \prime } \in \Pi _ { m } ^ { \mathrm { l o c } }$ with $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi }$ for every t.

Proof. Fix an agent i and any $\pmb { \pi } \in \Pi _ { H }$ . Let $d _ { i , t } ^ { \pi }$ and $q _ { i , t } ^ { \pi }$ denote the distributions of agent i’s augmented state and of its (augmented state, action) pair at time t:

$$
\begin{array} { r l r } & { } & { d _ { i , t } ^ { \pi } ( \tilde { s } ) = \mathrm { P r } _ { \pi } ( \tilde { s } _ { i , t } = \tilde { s } ) , \qquad 0 \leq t \leq H , \ } \\ & { } & { q _ { i , t } ^ { \pi } ( \tilde { s } , a ) = \mathrm { P r } _ { \pi } ( \tilde { s } _ { i , t } = \tilde { s } , a _ { i , t } = a ) , \qquad 0 \leq t < H , } \end{array}
$$

so that $\begin{array} { r } { \sum _ { a } q _ { i , t } ^ { \pi } ( \tilde { s } , a ) = d _ { i , t } ^ { \pi } ( \tilde { s } ) } \end{array}$ for $0 \leq t < H$ . We now construct a policy $\pi _ { i } ^ { \prime \prime } \in \Pi _ { m } ^ { \mathrm { l o c } }$ and show by induction on t that $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi }$ . Define the local policy $\pi _ { i } ^ { \prime \prime } = ( \pi _ { i , 0 } ^ { \prime \prime } , \ldots , \pi _ { i , H - 1 } ^ { \prime \prime } )$ . When $d _ { i , t } ^ { \pi } ( \widetilde { s } ) > 0$ , set

$$
\pi _ { i , t } ^ { \prime \prime } ( a \mid \tilde { s } ) = q _ { i , t } ^ { \pi } ( \tilde { s } , a ) / d _ { i , t } ^ { \pi } ( \tilde { s } ) .
$$

When $d _ { i , t } ^ { \pi } ( \widetilde { s } ) = 0$ , choose $\pi _ { i , t } ^ { \prime \prime } ( \cdot \mid \tilde { s } ) \in \mathcal { D } ( \mathcal { A } )$ arbitrarily.

Base case. $\mathrm { A t } ~ t = 0$ , the history $h _ { i , 0 }$ is empty and $s _ { i , 0 } \sim \rho _ { i }$ . Hence, $d _ { i , 0 } ^ { \pi _ { i } ^ { \prime \prime } } ( s , h _ { i , 0 } ) = d _ { i , 0 } ^ { \pi } ( s , h _ { i , 0 } ) = \rho _ { i } ( s )$ for every $s \in S$ , independently of the policy.

Inductive step. Assume $0 \leq t < H$ and the inductive hypothesis $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi }$ , and write $q _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } }$ for agent i’s (augmented state, action) distribution under $\pi _ { i } ^ { \prime \prime }$ . If $d _ { i , t } ^ { \pi } ( \widetilde { s } ) = 0$ , both state–action occupancies are zero by the hypothesis and the identity $\begin{array} { r } { \sum _ { a } q _ { i , t } ^ { \pi } ( \tilde { s } , a ) = d _ { i , t } ^ { \pi } ( \tilde { s } ) } \end{array}$ . Otherwise, the hypothesis and the definition of $\pi _ { i } ^ { \prime \prime }$ give

$$
q _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } ( \tilde { s } , a ) = d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } ( \tilde { s } ) \pi _ { i , t } ^ { \prime \prime } ( a \mid \tilde { s } ) = d _ { i , t } ^ { \pi } ( \tilde { s } ) \frac { q _ { i , t } ^ { \pi } ( \tilde { s } , a ) } { d _ { i , t } ^ { \pi } ( \tilde { s } ) } = q _ { i , t } ^ { \pi } ( \tilde { s } , a ) .
$$

It remains to prove the equality at time $t + 1$ . Write $\tilde { s } = ( s , h )$ . Under decoupled dynamics, agent i moves to $s ^ { \prime }$ with probability $P _ { i , t } ( s ^ { \prime } \mid s , a )$ , and its augmented state becomes $\left( s ^ { \prime } , \varphi ( h , s ) \right)$ . Both depend on the joint state only through agent i’s own pair $( \tilde { s } , a )$ . For an augmented state $( s ^ { \prime } , h ^ { \prime } )$ at time $t + 1$ , summing over the pairs that lead to it therefore gives

$$
d _ { i , t + 1 } ^ { \pi } ( s ^ { \prime } , h ^ { \prime } ) = \sum _ { \scriptstyle ( ( s , h ) , a ) : \varphi ( h , s ) = h ^ { \prime } } q _ { i , t } ^ { \pi } ( ( s , h ) , a ) P _ { i , t } ( s ^ { \prime } \mid s , a ) .
$$

The same relation holds for $\pi _ { i } ^ { \prime \prime }$ with $q _ { i , t } ^ { \pi }$ replaced by $q _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } }$ . Since $q _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = q _ { i , t } ^ { \pi }$ , we obtain $d _ { i , t + 1 } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t + 1 } ^ { \pi }$ , completing the induction.

By (20) and $d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } = d _ { i , t } ^ { \pi }$ , for every predecessor policy $\bar { \pi } _ { [ i - 1 ] }$

$$
G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ^ { \prime \prime } ) = \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } ^ { \prime \prime } } } \big [ r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) \big ] = \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi } } \big [ r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) \big ] .
$$

## E.3 Proof of Lemma 5

Lemma 5. Under (11), with probability at least $1 - \delta .$

$$
\left| \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) \right| \leq \epsilon
$$

for all $i \geq 2 ,$ , t, and $\tilde { s } \in \tilde { S } _ { t }$

Proof. We adapt the concentration argument of Chen et al. [4]. Fix an agent $i \geq 2$ , a time t, and a historyaugmented state $\tilde { s } _ { i , t } \in \tilde { S } _ { t }$ . The empirical reward

$$
\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \Delta \big ( ( t , s _ { i , t } ) \mid \nu _ { t } ( h _ { i , t } ) \cup \pmb { \tau } _ { [ i - 1 ] } ^ { k } \big )
$$

is an average of marginal gains over K newly sampled predecessor trajectories. Conditional on the previously computed policies $\bar { \pi } _ { [ i - 1 ] } .$ these trajectories are independent and each marginal gain has mean $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big )$ By monotonicity and submodularity, each marginal gain lies in $[ 0 , R ]$ , where $R = \operatorname* { m a x } _ { v \in \mathcal { V } } F ( \{ v \} )$ . Applying Hoefding’s inequality (Lemma 8) conditionally gives

$$
\operatorname* { P r } \left( \left| \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \left( \tilde { s } _ { i , t } \right) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \left( \tilde { s } _ { i , t } \right) \right| > \epsilon \ \Big | \ \bar { \pi } _ { [ i - 1 ] } \right) \leq 2 \exp \left( - \frac { 2 K \epsilon ^ { 2 } } { R ^ { 2 } } \right) .
$$

The bound holds for every realization of the predecessor policies. Averaging over the randomness used to compute these policies gives the same bound unconditionally.

$$
\begin{array} { r l } { \operatorname* { P r } \Big ( \underset { i , t , \hat { s } _ { i , t } } { \operatorname* { m a x } } \left| \hat { r } _ { i , t } ^ { \pi _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) - r _ { i , t } ^ { \pi _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) \right| > \epsilon \Big ) \leq \underset { i , t , \tilde { s } _ { i , t } } { \sum } \operatorname* { P r } \Big ( \big | \hat { r } _ { i , t } ^ { \pi _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) - r _ { i , t } ^ { \pi _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) \big | > \epsilon \Big ) } & { } \\ { \leq 2 ( N - 1 ) ( H + 1 ) | S | ^ { m + 1 } \exp \Big ( - \frac { 2 K \epsilon ^ { 2 } } { R ^ { 2 } } \Big ) . } & { } \end{array}
$$

This right-hand side is at most δ if and only if $\begin{array} { r } { \frac { 2 K \epsilon ^ { 2 } } { R ^ { 2 } } \geq \ln \frac { 2 ( N - 1 ) ( H + 1 ) | S | ^ { m + 1 } } { \delta } } \end{array}$ , that is, if and only if K satisfies (11). Therefore, with probability at least $1 - \delta , | \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) | \leq \epsilon$ for all $i , t , \tilde { s } _ { i , t }$ □

## E.4 Proof of Lemma 6

Lemma 6. Let $\begin{array} { r } { \bar { \pi } _ { i } \in \arg \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } \widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) . \mathrm { ~ I f ~ } \left| \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) - r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \tilde { s } ) \right| \leq \epsilon } \end{array}$ for all t and $\tilde { s } \in \tilde { S } _ { t }$ , then

$$
G _ { i } ^ { \pi _ { [ i - 1 ] } } ( \pi _ { i } ) \geq \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \log } } G _ { i } ^ { \pi _ { [ i - 1 ] } } ( \pi _ { i } ) - 2 ( H + 1 ) \epsilon .
$$

For $i = 1 , \bar { \pi } _ { 1 }$ attains the maximum exactly.

Proof. Fix the predecessor policies $\bar { \pi } _ { [ i - 1 ] }$ . For any $\pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } }$ , the exact objective (20) is

$$
G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) = \sum _ { { t = 0 } } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \big [ r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) \big ] .
$$

Algorithm 1 maximizes the estimated objective

$$
\widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) = \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \big [ \widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \tilde { s } _ { i , t } \big ) \big ] ,
$$

obtained from $\boldsymbol { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } }$ by replacing the reward $r _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ with its estimate $\widehat { r } _ { i , t } ^ { \bar { \pi } _ { [ i - 1 ] } }$ . The state marginal $d _ { i , t } ^ { \pi _ { i } }$ is identical in both objectives, since it is determined by agent i’s dynamics and policy alone and does not depend on the reward. We bound the diference between the true and estimated objectives:

$$
\begin{array} { r l r } { \big | G _ { i } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } ( \pi _ { i } ) - \widehat G _ { i } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } ( \pi _ { i } ) \big | = \Big | \displaystyle \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \big [ \bar { r } _ { i , t } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } \big ( \tilde { s } _ { i , t } \big ) - \widehat { r } _ { i , t } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } \big ( \tilde { s } _ { i , t } \big ) \big ] \Big | } & { \quad \mathrm { ( l i n e a r i t y ~ o f ~ e x p e c t a t i o n ) } } \\ { \leq \displaystyle \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \big [ \big | r _ { i , t } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } \big ( \tilde { s } _ { i , t } \big ) - \widehat { r } _ { i , t } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } \big ( \tilde { s } _ { i , t } \big ) \big | \big ] } & { \quad \mathrm { ( t r i a n g l e ~ i n e q u a l i t y ) } } \\ { \leq \displaystyle \sum _ { t = 0 } ^ { H } \epsilon = ( H + 1 ) \epsilon . } & { \quad \mathrm { ( l i r \tilde { r } _ { i , t } ^ { \bar { \pi } _ { \{ i ^ { \cdot } - 1 \} } } } \big ( \tilde { s } _ { i , t } \big ) - r _ { i , t } ^ { \bar { \pi } _ { \{ i , t } } - 1 } \big ( \tilde { s } _ { i , t } \big ) \big | \leq \epsilon )  \end{array}
$$

Since the upper bound does not depend on $\pi _ { i } .$ , removing the absolute value yields two inequalities that hold for all $\pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } }$ :

$$
\widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) - ( H + 1 ) \epsilon \leq G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) ,\tag{24}
$$

$$
G _ { i } ^ { \widehat \pi _ { [ i - 1 ] } } ( \pi _ { i } ) - ( H + 1 ) \epsilon \leq \widehat { G } _ { i } ^ { \widehat \pi _ { [ i - 1 ] } } ( \pi _ { i } ) .\tag{25}
$$

Because $\bar { \pi } _ { i } \in \arg \operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } \widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } \big ( \pi _ { i } \big )$ ，

$$
\widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) \leq \widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) \quad \mathrm { f o r ~ a l l ~ } \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } .\tag{26}
$$

Subtracting $( H + 1 ) \epsilon$ from both sides of (25) gives the left inequality below. Combining (26) with (24) at $\bar { \pi } _ { i }$ gives the right one. Hence, for all $\pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } }$

$$
G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) - 2 ( H + 1 ) \epsilon \leq \widehat { G } _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) - ( H + 1 ) \epsilon \leq G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) .
$$

Taking the maximum of the left-hand side over $\pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } }$ gives the desired result:

$$
\operatorname* { m a x } _ { \pi _ { i } \in \Pi _ { m } ^ { \mathrm { l o c } } } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \pi _ { i } ) - 2 ( H + 1 ) \epsilon \leq G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) .
$$

For $i = 1$ the reward $r _ { 1 , t } ^ { \bar { \pi } _ { [ 0 ] } }$ is computed exactly, so $\widehat { G } _ { 1 } ^ { \bar { \pi } _ { [ 0 ] } } = { G } _ { 1 } ^ { \bar { \pi } _ { [ 0 ] } }$ , the error terms vanish, and $\bar { \pi } _ { 1 }$ attains the maximum exactly. □

## E.5 Proof of Lemma 7

Lemma 7. For the policy π¯ returned by Algorithm 1,

$$
( 1 - c _ { m } ) \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) \leq J ( \bar { \pi } ) \leq \sum _ { i \in [ N ] } G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } ( \bar { \pi } _ { i } ) .
$$

Proof. Fix a realization of the returned policy π¯ and a joint trajectory τ generated under it. The trajectory determines the visited pairs $\tau _ { i , t }$ and the sets $\tau _ { i , t - m : t - 1 } , \ \tau _ { i , 0 : t - 1 }$ , and $\tau _ { [ i - 1 ] }$ . Every relation up to the final expectation step holds for each such τ separately, and all expectations below are over the initial states and the transition dynamics with π¯ held fixed. Adding the visits in lexicographic order of (i, t) gives

$$
F ( \pmb { \tau } ) = \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \Delta \big ( \tau _ { i , t } \big | \tau _ { i , 0 : t - 1 } \cup \pmb { \tau } _ { [ i - 1 ] } \big ) .\tag{27}
$$

For each agent i and time t, apply Lemma 3 with $X _ { t } = \tau _ { i , t } \subseteq \mathcal { V } _ { t } , B = \tau _ { i , 0 : t - m - 1 }$ , and $Z = \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] }$ Since $B \cup Z = \tau _ { i , 0 : t - 1 } \cup \tau _ { [ i - 1 ] }$ , we obtain

$$
\Delta \left( \tau _ { i , t } \mid \tau _ { i , 0 : t - 1 } \cup \tau _ { \left[ i - 1 \right] } \right) \geq \left( 1 - c _ { m } \right) \Delta \left( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { \left[ i - 1 \right] } \right) ,
$$

while submodularity $( \tau _ { i , t - m : t - 1 } \subseteq \tau _ { i , 0 : t - 1 } )$ gives

$$
\Delta { \left( \tau _ { i , t } \mid \tau _ { i , 0 : t - 1 } \cup \tau _ { \left[ i - 1 \right] } \right) } \leq \Delta { \left( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { \left[ i - 1 \right] } \right) } .
$$

For every joint trajectory τ , summing the preceding inequalities over $i \in [ N ]$ and $t \in \{ 0 , \ldots , H \}$ and using (27) yields

$$
( 1 - c _ { m } ) \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \Delta \bigl ( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] } \bigr ) \ \leq \ F ( \tau ) \ \leq \ \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \Delta \bigl ( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] } \bigr ) .
$$

Taking expectations over $\tau \sim \bar { \pi }$ gives $\mathbb { E } [ F ( \pmb { \tau } ) ] = J ( \bar { \pmb { \pi } } )$ , as defined in (3). By linearity of expectation, each expected sum of marginal gains equals the sum of the corresponding expectations, yielding

$$
( 1 - c _ { m } ) \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tau \sim \pi } \left[ \Delta \left( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] } \right) \right] \leq J ( \bar { \pi } ) \leq \sum _ { i \in [ N ] } \sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tau \sim \pi } \left[ \Delta \left( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] } \right) \right] .\tag{28}
$$

For each agent i and time $t ,$ the marginal gain in (28) depends only on the augmented state $\tilde { s } _ { i , t }$ and the predecessors’ visited set $\tau _ { [ i - 1 ] } .$ , since $\tilde { s } _ { i , t }$ determines $\tau _ { i , t }$ and $\tau _ { i , t - m : t - 1 } = \nu _ { t } ( h _ { i , t } )$ . Under the fixed policy $\bar { \pmb { \pi } } \in \Pi _ { m } ^ { \mathrm { l o c } }$ independent initial states and decoupled dynamics imply that $\tilde { s } _ { i , t }$ and $\tau _ { [ i - 1 ] }$ are independent. The expectation therefore decomposes as

$$
\begin{array} { r l } & { \mathbb { E } _ { \tau \sim \pi } \Big [ \Delta \big ( \tau _ { i , t } \mid \tau _ { i , t - m ; t - 1 } \cup \tau _ { [ i - 1 ] } \big ) \Big ] = \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \Big [ \mathbb { E } _ { \tau _ { [ i - 1 ] } \sim \pi } _ { [ i - 1 ] } \Big [ \Delta \big ( ( t , s _ { i , t } ) \mid \nu _ { t } ( h _ { i , t } ) \cup \tau _ { [ i - 1 ] } \big ) \Big ] \Big ] } \\ & { \qquad = \mathbb { E } _ { \tilde { s } _ { i , t } \sim d _ { i , t } ^ { \pi _ { i } } } \big [ r _ { i , t } ^ { \pi _ { [ i - 1 ] } } ( \tilde { s } _ { i , t } ) \big ] , } \end{array}
$$

where the second equality follows from the definition of the local stage reward in (8). Summing over $t \in \{ 0 , \ldots , H \}$ and using (20) gives

$$
\sum _ { t = 0 } ^ { H } \mathbb { E } _ { \tau \sim \bar { \pi } } \left[ \Delta \left( \tau _ { i , t } \mid \tau _ { i , t - m : t - 1 } \cup \tau _ { [ i - 1 ] } \right) \right] = G _ { i } ^ { \bar { \pi } _ { [ i - 1 ] } } \left( \bar { \pi } _ { i } \right) .
$$

Substituting this equality for each $i \in [ N ]$ into (28) establishes the claimed bounds.

## E.6 Hoefding’s Inequality

For completeness, we recall the concentration inequality used in the proof of Lemma 5.

Lemma 8 (Hoefding’s inequality [8]). Let $X _ { 1 } , \ldots , X _ { n }$ be independent random variables with $X _ { l } \in [ a , b ]$ , and let $\begin{array} { r } { \bar { X } = \frac { 1 } { n } \sum _ { l = 1 } ^ { n } X _ { l } } \end{array}$ with mean $\mu = \operatorname { \mathbb { E } } [ { \bar { X } } ]$ . Then for every $\epsilon > 0$

$$
\operatorname* { P r } { \big ( } | { \bar { X } } - \mu | \geq \epsilon { \big ) } \leq 2 \exp { \Big ( } - { \frac { 2 n \epsilon ^ { 2 } } { ( b - a ) ^ { 2 } } } { \Big ) } .
$$

## F EXPERIMENTAL DETAILS

## F.1 Environment and Evaluation

The trafic-monitoring instance uses a $2 0 \times 2 0$ grid and horizon $H = 2 0$ . Each UAV moves deterministically to an adjacent node or hovers. An action that leaves the grid keeps the UAV in place. Initial positions in the reported comparisons are sampled independently and uniformly over grid nodes.

The road map is generated with seed 17 and fixed across methods and algorithm seeds. Road importance is not uniform across the grid. As in real urban networks, a few congested regions require more frequent monitoring than the rest. We model this pattern with two regions of high-importance roads. For roads outside these regions, the maximum importance $w _ { e }$ is sampled from $\mathcal { N } ( 1 . 5 , 1 ^ { 2 } )$ and clipped to [0, 3.5]. For roads inside these regions, $w _ { e }$ is sampled from $\mathcal { N } ( 3 . 5 , 0 . 5 ^ { 2 } )$ and clipped to [2.5, 5.5].

Road importance recovers according to $\alpha _ { \ell } = 1 - e ^ { - \beta \ell }$ . In the time-additive case, $\alpha _ { \ell } = 1$ for every positive gap. Returns are divided by $\textstyle \sum _ { e \in { \mathcal { E } } } w _ { e }$ , the sum of the maximum importances of all roads. Repeated visits after recovery can contribute additional reward, so the normalized horizon return can exceed one.

Policy evaluation. Every policy is evaluated by its return J on 10,000 common initial joint states. During evaluation, SubPO selects the action with the highest probability for each UAV, while GWPO and GPO return deterministic policies. All reported confidence intervals are computed across algorithm seeds using Student’s t-distribution at the 95% confidence level, with a fixed map and evaluation states.

The memory comparison in Figure 2(a) uses $N = 2 0 , m \in \{ 0 , 1 , 2 , 3 \}$ , and five algorithm seeds. For each seed, we express the return improvement of GWPO over GPO as a percentage of the GPO return, and then average these percentages across seeds. In Figure 3, each SubPO variant uses five training seeds, while GWPO with $m = 3$ and GPO each use a single seed. In the figure, each return gap is measured relative to the GWPO return from this single seed. The confidence intervals of the gaps therefore reflect only the variation across SubPO seeds.

## F.2 Algorithm Settings and Hyperparameter Selection

GWPO and GPO. GWPO estimates local stage rewards using (10) and optimizes each local policy by backward induction on the augmented states. GPO uses the same number of sampled trajectories as GWPO with $m = 0$ . GWPO computes marginal gains given full predecessor trajectories and the agent’s own m-step history, whereas GPO uses only predecessor visits at the current time. Thus, the two methods difer even when $m = 0$ Table 1 reports the pre-window curvature and the multiplicative factor in Theorem 2. These factors exclude the additive error term.

Number of sampled trajectories. We fix $\delta = 1 0 ^ { - 6 }$ . For each sample count K, the sample-size bound (11) then gives the tolerance

$$
\epsilon = R \sqrt { \frac { 1 } { 2 K } \ln \frac { 2 ( N - 1 ) ( H + 1 ) | \boldsymbol { S } | ^ { m + 1 } } { \delta } } .
$$

Here $R \approx 0 . 0 1 0 5$ for the reward.

Table 2 reports the return J for five sample counts at each of $N = 6$ and N = 20. We used GWPO with $m = 3 .$ $\beta = 0 . 4$ , and uniform initial state distributions. Each setting was run with seeds {1, 2, 3}, and all policies were evaluated on the same 10,000 initial joint states. As increasing K did not noticeably change the return, we ran the remaining experiments with $\epsilon \approx 3 . 8 \times 1 0 ^ { - 3 }$ for all N to reduce computation.

Table 1: Pre-window curvature $c _ { m } = e ^ { - \beta ( m + 1 ) }$ and the approximation factor $( 1 - c _ { m } ) / ( 2 - c _ { m } )$ of Theorem 2 for the trafic-monitoring reward at the listed recovery rates $\beta$ and memory lengths m.
<table><tr><td> $\beta$ </td><td>m</td><td> $c _ { m }$ </td><td>Approximation factor</td></tr><tr><td>0.1</td><td>0</td><td>0.905</td><td>0.087</td></tr><tr><td>0.1</td><td>3</td><td>0.670</td><td>0.248</td></tr><tr><td>0.4</td><td>0</td><td>0.670</td><td>0.248</td></tr><tr><td>0.4</td><td>3</td><td>0.202</td><td>0.444</td></tr><tr><td>0.8</td><td>0</td><td>0.449</td><td>0.355</td></tr><tr><td>0.8</td><td>3</td><td>0.041</td><td>0.490</td></tr></table>

Table 2: Sample counts K, associated ϵ, and returns J for GWPO with $m = 3 , \beta = 0 . 4$ , and uniform initial state distributions. We fix $\delta = 1 0 ^ { - 6 }$ . Each return is the mean over seeds {1, 2, 3} with its 95% confidence interval. The second row gives the counts used in the remaining experiments.
<table><tr><td colspan="3"> $N = 6$ </td><td colspan="3"> $N = 2 0$ </td></tr><tr><td>K</td><td>E</td><td>J</td><td>K</td><td>E</td><td>J</td></tr><tr><td>17</td><td> $1 . 1 9 \times 1 0 ^ { - 2 }$ </td><td> $0 . 6 6 3 5 \pm 0 . 0 0 6 4$ </td><td>17</td><td> $1 . 2 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 0 2 8 \pm 0 . 0 1 1 6$ </td></tr><tr><td>166</td><td> $3 . 7 9 \times 1 0 ^ { - 3 }$ </td><td> $0 . 6 7 1 1 \pm 0 . 0 0 5 0$ </td><td>173</td><td> $3 . 7 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 0 5 0 \pm 0 . 0 0 5 8$ </td></tr><tr><td>1,656</td><td> $1 . 2 0 \times 1 0 ^ { - 3 }$ </td><td> $0 . 6 6 5 4 \pm 0 . 0 0 1 6$ </td><td> $1 { , } 7 3 0$ </td><td> $1 . 1 9 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 0 3 4 \pm 0 . 0 0 6 7$ </td></tr><tr><td> $^ { 1 6 , 5 5 6 }$ </td><td> $3 . 8 0 \times 1 0 ^ { - 4 }$ </td><td> $0 . 6 6 6 3 \pm 0 . 0 0 0 5$ </td><td> $^ \mathrm { 1 7 , 2 9 5 }$ </td><td> $3 . 7 7 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 0 2 1 \pm 0 . 0 0 5 5$ </td></tr><tr><td> $^ { 1 6 5 , 5 5 6 }$ </td><td> $1 . 2 0 \times 1 0 ^ { - 4 }$ </td><td> $0 . 6 6 6 1 \pm 0 . 0 0 0 2$ </td><td> $^ \mathrm { 1 7 2 , 9 5 2 }$ </td><td> $1 . 1 9 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 0 1 5 \pm 0 . 0 0 0 9$ </td></tr></table>

Table 3: Selected neural network architectures and hyperparameters for SubPO. Depth denotes the number of ReLU layers; the recurrent policy additionally uses one GRU layer.
<table><tr><td rowspan="2">N</td><td colspan="3">SUBPO-M</td><td colspan="3">SUBPO-NM</td></tr><tr><td>Depth</td><td>Width</td><td>Learning rate</td><td>Depth</td><td>Width</td><td>Learning rate</td></tr><tr><td>2</td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td><td>2</td><td>256</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>6</td><td>1</td><td>256</td><td> $1 0 ^ { - 3 }$ </td><td>1</td><td>512</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>10</td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td><td>1</td><td>512</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>15</td><td>1</td><td>512</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>20</td><td>1</td><td>256</td><td> $1 0 ^ { - 3 }$ </td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>25</td><td>1</td><td>256</td><td> $1 0 ^ { - 3 }$ </td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>30</td><td>1</td><td>512</td><td> $1 0 ^ { - 3 }$ </td><td>1</td><td>512</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr></table>

SubPO architectures. Both SubPO variants receive one-hot encodings of the joint state and time. SubPO-M processes these inputs with a feedforward ReLU network, while SubPO-NM uses a ReLU encoder followed by a single-layer GRU with the same hidden width as the encoder. Both networks output a probability distribution over the actions of each UAV.

SubPO hyperparameter selection. For each team size and SubPO variant, we considered networks with {1, 2} ReLU layers, hidden widths {256, 512}, and learning rates $\{ 1 0 ^ { - 3 } , 3 \times 1 0 ^ { - 4 } \}$ . For SubPO-NM, these layers precede the GRU. Batch sizes were 128 for SubPO-M and 256 for $\mathrm { S U B P O - N M } .$ . An epoch is one gradient update on a fresh batch of trajectories. Table 3 reports the selected hyperparameters.

Hyperparameter selection used $\beta = 0 . 1$ and independent nonuniform initial distributions concentrated around a fixed grid node for each UAV. Each configuration was trained for 10,000 epochs and evaluated every 500 epochs on 4,000 fixed validation initial joint states.

For each team size and variant, we selected the configuration with the highest validation return and used it for all comparisons with uniform initial positions and $\beta \in \{ 0 . 1 , 0 . 4 , 0 . 8 \}$ . Hyperparameter selection time is excluded from the computation-time budgets.

## F.3 Computation-Time Budgets and Policy Selection

Computation-time budgets. Let $T _ { N }$ be the measured computation time of GWPO with $m = 3$ for N agents. GWPO and GPO run one job at a time on one AMD Ryzen 9 9950X3D CPU core, while $\mathrm { S U B P O }$ uses an NVIDIA RTX PRO 6000 GPU with one CPU thread per run. For all methods, the computation time covers algorithm setup and policy optimization. For GWPO and GPO, it includes trajectory sampling and reward estimation, as well as building the augmented state spaces for GWPO. For SubPO, it includes training and the validations during training. Environment initialization and final test evaluation are excluded. The GWPO and GPO policies are held fixed across the budgets $T _ { N } , 5 T _ { N }$ , and $2 0 T _ { N }$

![](images/6d94b714823bd20f2098532ec3f35bef8ef6676b213eb5f1eb6ba6f44295d06b.jpg)  
(a) Time budget $T _ { N }$

![](images/67afe3ca9fb23b27acecabcf36acd61e74986acc0b82bc4a58def15bd9b9f6f3.jpg)  
(b) Time budget $5 T _ { N }$

![](images/e7bb5d75b0e873ef943bc56bcbd80239ee7303dab615a9dba376e5c73630886b.jpg)  
(c) Time budget $2 0 T _ { N }$  
Figure 8: Returns of GWPO with $m = 3 .$ GPO, and the two SubPO variants at $\beta = 0 . 4$ . From left to right, the allowed computation times for SubPO are $T _ { N }$ , $5 T _ { N }$ , and $2 0 T _ { N }$ . SubPO curves show means over five training seeds with 95% confidence intervals. GWPO and GPO each use a single seed.

We train several SubPO runs in parallel, so the elapsed time of each run may overstate its computation time. We therefore convert each computation-time budget into a number of epochs, using separate calibration runs executed one at a time. For each team size and variant, the calibration process uses three seeds at $\beta = 0 . 4$ . Using the number of epochs completed by $T _ { N }$ , we estimate the counts at $5 T _ { N }$ and $2 0 T _ { N }$ by extrapolating the number of epochs completed between $T _ { N } / 2$ and $T _ { N }$ . We take the largest estimate across calibration seeds and add a 10% allowance. We use the same epoch budgets for all training seeds and recovery rates, assuming that the cost of an epoch does not change across recovery rates or over training.

Training and policy selection. During each SubPO training run, we evaluated the greedy policy every 500 epochs on a fixed validation set of 4,000 uniformly sampled initial joint states, generated independently of the test set. We used these validation returns both to decide when to stop training and to select a policy for each epoch budget. After at least 10,000 epochs, training stopped once ten consecutive validations showed no significant improvement. An improvement was significant if the return exceeded $J _ { \mathrm { r e f } }$ by more than max $\{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } | J _ { \mathrm { r e f } } | \}$ Each such improvement updated $J _ { \mathrm { r e f } }$ and reset the counter.

For each computation-time budget, we selected the policy with the highest validation return reached within that budget. If a run stopped early because of the stopping criterion, we used its policy with the highest validation return for every larger budget. The selected policies were then evaluated on the test set.

## F.4 Additional Results

Figure 8 and Table 4 report returns J at $\beta = 0 . 4$ corresponding to the return gaps in Figure 3.

Figure 9 extends the comparison of Figure 3 to $\beta = 0 . 1$ and $\beta = 0 . 8$

Figure 10 shows the training curves. Among the runs shown, all except SubPO-NM at N = 20 stopped before the $2 0 T _ { N }$ budget.

Table 4: Returns of the selected policies at $\beta = 0 . 4$ with uniform initial state distributions. The SubPO budgets use the isolated timing calibration and 10% allowance described above. Each SubPO entry is the mean over five training seeds ± the half-width of its 95% confidence interval. GWPO with $m = 3$ uses seed 1. Intervals describe training-seed variation on the fixed map and evaluation states.
<table><tr><td>N</td><td>GWPO Policy</td><td></td><td> $T _ { N }$ </td><td> $5 T _ { N }$ </td><td> $2 0 T _ { N }$ </td></tr><tr><td>2</td><td>0.2484</td><td>SUBPO-M</td><td>0.2526 ± 0.0019</td><td> $0 . 2 5 5 1 \pm 0 . 0 0 1 4$ </td><td> $0 . 2 5 6 0 \pm 0 . 0 0 1 4$ </td></tr><tr><td></td><td></td><td> $\operatorname { S U B P O - N M }$ </td><td> $0 . 2 5 5 2 \pm 0 . 0 0 0 8$ </td><td> $0 . 2 5 7 0 \pm 0 . 0 0 0 3$ </td><td> $0 . 2 5 7 8 \pm 0 . 0 0 0 3$ </td></tr><tr><td>6</td><td>0.6691</td><td> $\mathrm { S U B P O - M }$ </td><td> $0 . 6 7 4 7 \pm 0 . 0 0 2 6$ </td><td> $0 . 6 8 8 8 \pm 0 . 0 0 1 3$ </td><td> $0 . 6 9 0 9 \pm 0 . 0 0 2 0$ </td></tr><tr><td></td><td></td><td> $\operatorname { S U B P O - N M }$ </td><td> $0 . 6 7 1 4 \pm 0 . 0 0 2 7$ </td><td> $0 . 6 9 7 9 \pm 0 . 0 0 1 9$ </td><td> $0 . 7 0 9 5 \pm 0 . 0 0 2 9$ </td></tr><tr><td>10</td><td>1.0312</td><td> $\mathrm { S U B P O - M }$ </td><td> $1 . 0 2 5 4 \pm 0 . 0 0 4 1$ </td><td> $1 . 0 5 4 5 \pm 0 . 0 0 3 5$ </td><td> $1 . 0 5 8 3 \pm 0 . 0 0 2 1$ </td></tr><tr><td></td><td></td><td>SUBPO-NM</td><td> $0 . 9 8 8 4 \pm 0 . 0 0 5 4$ </td><td> $1 . 0 6 5 7 \pm 0 . 0 0 4 0$ </td><td> $1 . 0 8 4 1 \pm 0 . 0 0 3 3$ </td></tr><tr><td>15</td><td>1.4427</td><td> $\mathrm { S U B P O - M }$ </td><td> $1 . 2 9 8 4 \pm 0 . 0 0 5 5$ </td><td> $1 . 4 6 6 4 \pm 0 . 0 0 5 4$ </td><td> $1 . 4 7 8 0 \pm 0 . 0 0 4 6$ </td></tr><tr><td></td><td></td><td>SUBPO-NM</td><td> $1 . 3 2 9 2 \pm 0 . 0 0 2 8$ </td><td> $1 . 4 3 7 4 \pm 0 . 0 0 5 8$ </td><td> $1 . 4 7 4 4 \pm 0 . 0 0 3 8$ </td></tr><tr><td>20</td><td>1.8076</td><td>SUBPO-M</td><td> $1 . 6 2 4 9 \pm 0 . 0 0 2 7$ </td><td> $1 . 8 0 5 8 \pm 0 . 0 0 8 6$ </td><td> $1 . 8 2 6 1 \pm 0 . 0 0 3 4$ </td></tr><tr><td></td><td></td><td>SUBPO-NM</td><td> $1 . 6 3 2 0 \pm 0 . 0 0 1 3$ </td><td> $1 . 6 9 9 8 \pm 0 . 0 0 5 0$ </td><td> $1 . 7 6 7 9 \pm 0 . 0 0 7 0$ </td></tr><tr><td>25</td><td>2.1429</td><td>SUBPO-M</td><td> $1 . 8 8 8 2 \pm 0 . 0 0 2 2$ </td><td> $2 . 0 5 3 6 \pm 0 . 0 1 1 4$ </td><td> $2 . 1 4 1 7 \pm 0 . 0 0 7 3$ </td></tr><tr><td></td><td></td><td>SUBPO-NM</td><td> $1 . 9 1 7 5 \pm 0 . 0 0 2 4$ </td><td> $1 . 9 6 0 8 \pm 0 . 0 0 1 0$ </td><td> $1 . 9 6 6 9 \pm 0 . 0 0 4 3$ </td></tr><tr><td>30</td><td>2.4447</td><td>SUBPO-M</td><td> $2 . 1 5 2 6 \pm 0 . 0 0 2 4$ </td><td> $2 . 2 5 1 8 \pm 0 . 0 0 5 4$ </td><td> $2 . 3 2 1 3 \pm 0 . 0 0 8 9$ </td></tr><tr><td></td><td></td><td> $\operatorname { S U B P O - N M }$ </td><td> $2 . 1 7 7 5 \pm 0 . 0 0 4 0$ </td><td> $2 . 2 5 1 7 \pm 0 . 0 0 1 7$ </td><td> $2 . 2 7 6 3 \pm 0 . 0 2 3 0$ </td></tr></table>

![](images/8d4c522f2d082d99456245b22ebd8be5818c4c9ae8a4dc87adea2d8ae2a70a6f.jpg)  
(a) Computation time $T _ { N }$

![](images/f4f6c79675942f819038ab43089f1d7d67620932d86d74602509a0f3f74377fc.jpg)  
(b) Computation time $5 T N$

![](images/e29276740826b0e9a6ef4ca28b357dcdcd90f7cda8502b90c992aafddc8e256c.jpg)  
(c) Computation time $2 0 T _ { N }$

![](images/b395595f78f54078e00be006653b5badbc9ab94bdaed64fbd28d925626ef1c78.jpg)  
(d) Computation time $T _ { N }$

![](images/dc696ffad341975c1299904ae00eaf27481d241ae6a06def096008dfe3f6795c.jpg)  
(e) Computation time $5 T N$

![](images/fe3de504330ad05b665391dd159be00344b5bda1a7278bbd74bbea1b20714043.jpg)  
(f) Computation time $2 0 T _ { N }$

Figure 9: Return diferences between each baseline and GWPO with $m = 3 ,$ expressed as percentages of the GWPO return, at $\beta = 0 . 1 \ ( \mathrm { t o p } )$ and $\beta = 0 . 8$ (bottom). Negative values indicate higher returns for GWPO. From left to right, the allowed computation times for SubPO are $T _ { N } , 5 T _ { N }$ , and $2 0 T _ { N }$ . SubPO curves show means over five training seeds with 95% confidence intervals. GWPO and GPO each use a single seed.

![](images/5b6ad8c8ddb3ebcaf53c9f55af112b9b3ed8f525bb06a626fcd6c95617e884c7.jpg)  
Training epochs (thousands)  
(a) N = 10

Training epochs (thousands)  
![](images/3bb46df315a09721087cf8ea2901a169f0393a97f113fd067bda7353eb713c76.jpg)  
(b) N = 20

Training epochs (thousands)  
![](images/2430cac13e2706fa8cd7a256559d9295aedc3d9a27cdb3fd21e8fb9d579d2710.jpg)  
(c) N = 30

Figure 10: Training of $\mathrm { S U B P O - M }$ and $\mathrm { S U B P O - N M }$ at $\beta = 0 . 4 .$ . Curves show the highest validation return reached by each epoch, averaged over five training runs with 95% confidence intervals. A curve ends when all five runs have stopped. Markers indicate the epoch budgets corresponding to $T _ { N } , 5 T _ { N }$ , and $2 0 T _ { N }$ that fall within each curve. The horizontal line shows the test return of GWPO with $m = 3$