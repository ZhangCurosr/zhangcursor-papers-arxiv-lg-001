# Fully Byzantine-Resilient Multi-Agent Reinforcement Learning

Haejoon Lee, Student Member, IEEE, and Dimitra Panagou, Senior Member, IEEE

Abstract— We study distributed Byzantine-resilient actor-critic multi-agent reinforcement learning (AC-MARL), where agents collectively learn policies through local interactions. Existing methods guarantee convergence of the agents’ parameters only to a neighborhood of the attack-free limit points, resulting in degraded performance. We propose Fully Resilient AC-MARL (FRAC-MARL), a decentralized method in which each agent leverages redundancy in two-hop messages to identify reliable messages. Under linear parameterizations of the value and team-reward functions and Byzantine edge attacks, where adversarial behavior is confined to the communication layer, we prove that agents’ parameters converge almost surely to the same limit points as in the attack-free case over time-varying communication graphs. We introduce a novel topological condition for the convergence of our method, present a systematic method to construct such networks, and prove that this condition can be verified in polynomial time. Finally, we demonstrate our method on cooperative multi-robot formation control tasks. [code]<sup>a</sup>

Index Terms— Multi-Agent reinforcement learning, networked control systems, fault-tolerant systems

## I. INTRODUCTION

of reinforcement learning (RL) [1] for learning optimal policies of multiple agents interacting within a shared environment [2]. Distributed cooperative MARL in particular considers multiple agents that aim to optimize a shared global objective - often expressed as the average of local rewards - through information sharing with their neighbors in a network [3]–[5].

In this paper, we consider fully distributed actor-critic MARL (AC-MARL) methods studied in [3], [6]–[8], where agents with heterogeneous reward functions perform consensus-based updates of their local policy and valuefunction parameter estimates over a communication network. Compared with distributed tabular Q-learning approaches [9], these methods offer improved scalability to problems with large state and action spaces. Representative deep actor-critic MARL methods include MADDPG [10], COMA [11], and MAPPO [12]. However, these methods typically rely on centralized training or centralized value estimation.

For fully distributed AC-MARL, [3] established convergence under linear function approximation, while finite-time convergence guarantees were provided in [6]. Later, [7] established convergence under linear function approximation for directed communication graphs. Finally, [8] presented an algorithm with nonlinear function approximation and established asymptotic convergence guarantees.

Despite these merits, distributed MARL algorithms, similarly to other distributed learning and optimization methods, are highly susceptible to adversarial attacks that corrupt or manipulate information. In the distributed systems literature, the Byzantine model represents an omniscient adversary capable of injecting arbitrary disruptions through compromised hardware, software, or communication channels [13], [14]. As a result, a wide range of resilient algorithms have been developed to contain the impact of Byzantine agents on distributed consensus [14]–[16], optimization [13], [17], [18], learning frameworks [19]–[21], and, more recently, multi-agent LLM decision-making systems [22], [23].

Similarly, Byzantine-resilient distributed MARL has received increasing attention in recent years. Early studies demonstrated that even a single adversarial agent can significantly degrade or destabilize learning in cooperative MARL settings [24], [25]. In fact, it has been shown in [26] that learning optimal policies is generally impossible in the presence of Byzantine agents.

To address this vulnerability, several Byzantine-resilient MARL algorithms have been proposed. The distributed tabular Q-learning algorithm in [9] was extended in [27] by incorporating trimmed-mean aggregation to guarantee convergence in the presence of Byzantine agents. The trimmed-mean strategy was subsequently incorporated into consensus-based AC-MARL algorithms with linear function approximation to achieve resilience against Byzantine agents [28], [29]. More recently, [30], [31] combined projection-based updates with trimmed-mean aggregation to further strengthen the resilience of AC-MARL. Similarly, [32] employs geometric-median aggregation to mitigate Byzantine attacks, although it does not provide convergence guarantees and considers only noncolluding adversaries.

Despite these advances, existing methods have several limitations. First, these approaches guarantee convergence only to a neighborhood of the attack-free limit point of the attackfree case. Thus, Byzantine agents may strategically manipulate the exchanged parameters to substantially degrade the performance of the non-Byzantine agents [30]. Although [30], [31] mitigate this issue, they still guarantee only neighborhood convergence, and the size of the resulting neighborhood is generally difficult to characterize.

Furthermore, many Byzantine-resilient AC-MARL methods, including [29]–[31], require the communication network to satisfy $( 2 F + 1 )$ )-robustness [14]. Since verifying such properties is coNP-complete [33], their applicability to large-scale systems may be limited. A separate line of work [34]–[36] avoids these r-robustness requirements, but instead relies on a trusted central coordinator.

In addition, decentralized MARL with heterogeneous local rewards can be viewed through the lens of decentralized stochastic optimization under non-i.i.d. data distributions. Compared with Byzantine-resilient deterministic optimization [13], [17] and stochastic optimization under i.i.d. assumptions [19], [37], the non-i.i.d. setting introduces additional challenges due to heterogeneity and stochasticity across agents, which induces intrinsic bias that is further exacerbated by Byzantine attacks [38]. The work in [39] studied Byzantine-resilient stochastic optimization with non-i.i.d. data from a consensus perspective under a complete communication graph. In [38], authors iteratively remove values farthest from the weighted mean to construct a doubly stochastic mixing matrix with a sufficiently small contraction factor, thereby ensuring convergence. In contrast, [40] proposed a clippingbased mechanism that clips received values to neighborhoods centered at the receiving agents’ values. Nevertheless, these methods only guarantee convergence to a neighborhood of a stationary point of the underlying optimization problem.

To address these limitations, we build on our prior work on resilient distributed Q-learning [41] and extend the framework to decentralized Byzantine-resilient AC-MARL over timevarying communication graphs. Specifically, we propose Fully Resilient AC-MARL (FRAC-MARL), in which agents use redundant information relayed through two-hop communication to identify reliable messages before incorporating them into their local updates. Under a sufficient topological condition on the communication networks and a weaker Byzantine attack model, we establish that the agents’ parameters converge almost surely to the exact limit points of the attack-free scenario, rather than to a neighborhood.

Contributions: Our contributions are as follows:

• We propose a novel decentralized Byzantine-resilient AC-MARL algorithm, termed Fully Resilient AC-MARL (FRAC-MARL), that leverages redundancy in two-hop communication to identify and incorporate reliable messages for learning over time-varying communication graphs.

• We introduce a novel topological condition, termed $( r , r ^ { \prime } )$ -redundancy, and prove that under linear function approximation it guarantees that FRAC-MARL converges almost surely to the same limit points as the attack-free case under Byzantine edge attacks, a weaker Byzantine model in which adversarial behavior is restricted to the communication layer.

• We provide a constructive procedure for designing $( r , r ^ { \prime } ) \boldsymbol { \cdot }$ redundant graphs and show that this property can be verified in polynomial time, in contrast to r-robustness [14], which is widely adopted in existing work [29]–[31] but is coNP-complete to verify [33].

• We demonstrate our method can be applied even with nonlinear function approximation in a cooperative multiagent formation task.

## II. NOTATION

We denote the cardinality of a set C as |C|. We denote the sets of non-negative and positive integers as $\mathbb { Z } _ { \geq 0 }$ and $\mathbb { Z } _ { > 0 }$ . A multiset C is a collection in which elements may occur with multiplicity. For a (multi)set C, we define mode(C) as any element with the highest number of occurrences in ${ \mathcal { C } } ,$ with ties broken uniformly at random and ${ \bmod { \mathrm { e } } } ( \emptyset ) = 0$ . We denote mode count(C) as the maximum number of occurrences of any element in a (multi)set C. We denote diag(·) as the diagonal matrix formed by the elements of its argument. We denote the probability and expectation of a probability space $( \Omega , \mathcal { F } , \mathbb { P } )$ by P(·) and $\mathbb { E } ( \cdot )$ . We denote the Kronecker product by ⊗. For finite sets $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ and functions $g _ { 1 } : { \mathcal { C } } _ { 1 } \to \mathbb { R }$ and $g _ { 2 } : \mathcal { C } _ { 1 } \times \mathcal { C } _ { 2 } \to \mathbb { R }$ , we write $[ g _ { 1 } ( c _ { 1 } ) , c _ { 1 } \in \mathring { \mathcal { C } } _ { 1 } ] ^ { \intercal } \in \mathbb { R } ^ { | \mathcal { C } _ { 1 } | }$ and $[ g _ { 2 } ( c _ { 1 } , c _ { 2 } ) , \ c _ { 1 } \in \mathcal { C } _ { 1 } , c _ { 2 } \in \mathcal { C } _ { 2 } ] ^ { \top } \ \in \ \mathbb { R } ^ { | \mathcal { C } _ { 1 } | \cdot | \mathcal { C } _ { 2 } | }$ for the column vector of values $g _ { 1 } ( c _ { 1 } )$ and $g _ { 2 } ( c _ { 1 } , c _ { 2 } )$ respectively, stacked according to fixed orderings of $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 1 } \times \mathcal { C } _ { 2 }$ , respectively, used consistently throughout the paper. A sequence of random variables $\{ X _ { t } \} _ { t \ge 0 }$ is said to converge to a random variable X almost surely (a.s.) if $\mathbb { P } ( \operatorname* { l i m } _ { t  \infty } X _ { t } = X ) = 1$

## III. PRELIMINARIES

We consider a system of n agents interacting over a simple, undirected, and time-varying communication graph $\mathcal G _ { t } ~ = ~ ( \nu , \mathcal E _ { t } )$ . The vertex set $\nu ~ = ~ \{ 1 , \ldots , n \}$ represents the agents, and the edge set $\mathcal { E } _ { t } \subseteq \mathcal { V } \times \mathcal { V }$ denotes a set of communication links between agents at time t. Since the graph is undirected, $( i , j ) \in \mathcal { E } _ { t }$ implies $( j , i ) \in \mathcal { E } _ { t }$ . For agent i, its one-hop and two-hop neighbor sets at time t are denoted by $\mathcal { N } _ { i , t } = \{ j \in \mathcal { V } \mid ( i , j ) \in \mathcal { E } _ { t } \}$ and $\mathcal { N } _ { i , t } ^ { ( 2 ) } = \{ k \in \mathcal { V } \mid \exists j \in$ $\mathcal { N } _ { i , t } \mathrm { ~ s . t . ~ } k \in \mathcal { N } _ { j , t } \setminus \{ i \} \}$ . The extended one-hop neighbor set is $B _ { i , t } ~ = ~ { \mathcal { N } } _ { i , t } \cup \{ i \} . ~ { \mathrm { ~ A ~ } }$ path of length $k \in \mathbb { Z } _ { > 0 }$ is a sequence of vertices $( y _ { 0 } , \ldots , y _ { k } )$ such that $( y _ { \ell - 1 } , y _ { \ell } ) \in \mathcal { E } _ { t }$ $\forall \ell \in \{ 1 , \ldots , k \}$ . A graph is connected if there exists a path between any pair of nodes.

In our setting, agents $i \in \mathcal V$ not only communicate with their direct neighbors $\mathcal { N } _ { i , t }$ at time t, but also with their twohop neighbors $\mathcal { N } _ { i , t } ^ { ( 2 ) }$ through relaying, which we refer to as two-hop communication:

Definition 1 (Two-hop communication). Let $m _ { j ^ { \prime } } ^ { j  i } ( t )$ denote the copy of the message $m _ { j ^ { \prime } } ( t )$ originating from any agent $j ^ { \prime } \in \mathcal { V }$ and delivered to agent i by agent j at time t. We set $m _ { i ^ { \prime } } ^ { j \to i } ( t ) = \emptyset$ if the message is not received by agent i.

At each time $t \in \mathbb { Z } _ { \geq 0 } ,$ agents exchange messages over $\mathcal G _ { t } ~ = ~ ( \nu , \mathcal E _ { t } )$ in two consecutive rounds. In round $z ~ = ~ 1$ (sending), each agent i transmits its own message $m _ { i } ( t )$ to every neighbor $\textit { j } \in \mathcal { N } _ { i , t }$ . In round $z \ = \ 2$ (relaying), each agent i forwards to every neighbor $j \in \mathcal { N } _ { i , t }$ <sub>t</sub> the messages it received in round 1, i.e., $\{ m _ { k } ^ { k  i } ( t ) \} _ { k \in \mathcal { N } _ { i , t } }$ . Thus, agent j obtains a copy of the messages originating from its two-hop neighbors $\hat { \mathcal { N } } _ { j , t } ^ { ( 2 ) }$ through an intermediate neighbor $i \in \mathcal { N } _ { j } ,$ <sub>t</sub>.

The environment is modeled as a networked multiagent Markov Decision Process (MDP) defined by a tuple $( \boldsymbol { S } , \{ \mathcal { A } ^ { i } \} _ { i \in \mathcal { V } } , P , \{ r ^ { i } \} _ { i \in \mathcal { V } } , \{ \mathcal { G } _ { t } \} _ { t \geq 0 } , \gamma )$ . Here, S denotes the finite state space shared by all agents, and $\mathcal { A } ^ { i }$ denotes the finite action space of agent i. The joint action space is given by $\begin{array} { r } { \mathcal { A } = \prod _ { i \in \mathcal { V } } \mathcal { A } ^ { i } } \end{array}$ . The function $P : \mathcal { S } \times \mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ specifies the state transition probability, where $P ( s ^ { \prime } \mid s , a )$ denotes the probability of transitioning to state $s ^ { \prime }$ from state s under action $a .$ Each agent i has a local reward function $r ^ { i } : S \times \mathcal { A }  \mathbb { R }$ and $\gamma \in ( 0 , 1 )$ is the discount factor. We assume that the state and joint action are globally observable, while rewards are private and observed only by the corresponding agents.

At each time step t, each agent $i \in \mathcal V$ observes the current state $s _ { t } \in S$ and independently selects an action

$$
a _ { t } ^ { i } \sim \pi ^ { i } ( { \cdot } | s _ { t } ) ,\tag{1}
$$

where $\pi ^ { i } : \mathcal { S } \times \mathcal { A } ^ { i }  [ 0 , 1 ]$ is the local stochastic policy. The resulting joint action is $a _ { t } \ = \ ( a _ { t } ^ { 1 } , \ldots , a _ { t } ^ { n } ) \in \ A .$ We assume that agent actions are conditionally independent given the current state, that is,

$$
\pi ( a \mid s ) = \prod _ { i \in \nu } \pi ^ { i } ( a ^ { i } \mid s ) .\tag{2}
$$

After executing $a _ { t } ^ { i }$ at the state $s _ { t } .$ , each agent i receives a local realized reward $r _ { t + 1 } ^ { i } = r ^ { i } ( s _ { t } , a _ { t } )$ , and the environment transitions to the next state according to $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , a _ { t } )$

## A. Objective and Policy Parameterization

For a joint policy π and a fixed initial-state distribution ν over S, the agents collectively maximize a globally averaged, discounted long-term reward

$$
\begin{array} { r } { J ( \pi ) : = \mathbb { E } _ { s _ { 0 } \sim \nu , \pi } \Big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \bar { r } ( s _ { t } , a _ { t } ) \Big ] , } \end{array}\tag{3}
$$

where $\begin{array} { r } { \bar { r } ( s , a ) = \frac { 1 } { n } \sum _ { i \in \mathcal { V } } r ^ { i } ( s , a ) } \end{array}$ represents the network-wise average reward at state-action pair $( s , a )$ . We also define the associated action- and state-value functions

$$
\begin{array} { r } { Q _ { \pi } ( s , a ) : = \mathbb { E } _ { \pi } \Big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \bar { r } ( s _ { t } , a _ { t } ) \ \Big | \ s _ { 0 } = s , a _ { 0 } = a \Big ] , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { V _ { \pi } ( s ) : = \sum _ { a \in \mathcal { A } } \pi ( a \mid s ) Q _ { \pi } ( s , a ) . } \end{array}\tag{5}
$$

Since $\begin{array} { r } { Q _ { \pi } ( s , a ) = { \bar { r } } ( s , a ) + \gamma \sum _ { s ^ { \prime } \in S } P ( s ^ { \prime } \mid s , a ) V _ { \pi } ( s ^ { \prime } ) } \end{array}$ , the temporal-difference (TD) error

$$
\delta _ { t } = \bar { r } ( s _ { t } , a _ { t } ) + \gamma V _ { \pi } ( s _ { t + 1 } ) - V _ { \pi } ( s _ { t } )\tag{6}
$$

is an unbiased sample of the global advantage, i.e., $\mathbb { E } [ \delta _ { t } \mid s _ { t } =$ $s , a _ { t } = a \} = Q _ { \pi } ( s , a ) - V _ { \pi } ( s )$

To enable scalable learning in large state and action spaces, we parameterize each local policy [42]. Let $\pi _ { \theta ^ { i } } ^ { i } : \mathcal { S } \times \mathcal { \bar { A } } ^ { i } $ [0, 1] denote the local policy of agent $i \in \mathcal V$ parametrized by $\bar { { \boldsymbol { \theta } } ^ { i } } \in \Theta ^ { i } \subset \mathbb { R } ^ { b _ { i } }$ . By the conditional independence in (2), we have

$$
\begin{array} { r l } & { \pi _ { \theta } ( a \mid s ) = \displaystyle \prod _ { i \in \mathcal { V } } \pi _ { \theta ^ { i } } ^ { i } ( a ^ { i } \mid s ) , } \\ & { P _ { \theta } ( s ^ { \prime } \mid s ) = \displaystyle \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) P ( s ^ { \prime } \mid s , a ) , } \end{array}
$$

where $\begin{array} { r } { \theta = [ ( \theta ^ { 1 } ) ^ { \top } , \dots , ( \theta ^ { n } ) ^ { \top } ] ^ { \top } \in \prod _ { i \in \mathcal { V } } \Theta ^ { i } = \Theta . } \end{array}$ . For any $\theta \in \Theta$ , the Markov process $\{ s _ { t } \} _ { t \in \mathbb { Z } _ { \geq 0 } }$ induced by $\pi _ { \theta }$ is irreducible and aperiodic. This guarantees unique, strictly positive stationary distributions $d _ { \theta } ( s )$ and $d _ { \theta } ^ { \prime } ( s , a ) = d _ { \theta } ( s ) \pi _ { \theta } ( a \mid s )$ of states and state-action pairs.

Writing $J ( \theta ) : = J ( \pi _ { \theta } ) , Q _ { \theta } : = Q _ { \pi _ { \theta } }$ , and $V _ { \theta } : = V _ { \pi _ { \theta } } :$ , our objective is to find $\theta ^ { \star } = \arg \operatorname* { m a x } _ { \theta \in \Theta } J ( \theta )$ . We follow the multi-agent policy gradient theorem from [30], where each agent improves its policy along the direction

$$
h ^ { i } ( \theta ) : = \mathbb { E } _ { d _ { \theta } , \pi _ { \theta } } \left[ \nabla _ { \theta ^ { i } } \log \pi _ { \theta ^ { i } } ^ { i } ( a ^ { i } \mid s ) \big ( Q _ { \theta } ( s , a ) - V _ { \theta } ( s ) \big ) \right] .\tag{7}
$$

Note that (7) follows the form given in [24], [30] which is the surrogate for $\nabla _ { \theta ^ { i } } J ( \theta )$ obtained by taking the expectation under the on-policy stationary distribution $d _ { \theta }$ rather than the discounted visitation measure.

## B. Decentralized Policy Evaluation

Evaluating (6) requires evaluations of ${ \bar { r } } ( s , a )$ and $V _ { \theta } ( s )$ which are not available to individual agents as rewards are private. Therefore, we adopt the consensus-based AC-MARL algorithm from [3], where each agent maintains approximations $\bar { r } ( s , a ; \lambda ^ { i } ) \approx \bar { r } ( s , a )$ and $V ( s ; v ^ { i } ) \approx V _ { \theta } ( s )$ , parameterized by $\lambda ^ { i }$ and $v ^ { i } .$ , respectively, while reaching consensus on $\lambda ^ { i } , v ^ { i }$ over the communication graph $\mathcal { G } _ { t }$ . In this paper, we consider linear approximation, i.e.,

$$
\bar { r } ( s , a ; \lambda ^ { i } ) = f ( s , a ) ^ { \top } \lambda ^ { i } , \qquad V ( s ; v ^ { i } ) = \phi ( s ) ^ { \top } v ^ { i } ,\tag{8}
$$

with the assumption:

Assumption 1. The feature vectors $\begin{array} { r l } { f ( s , a ) \quad } & { { } = } \end{array}$ $[ f _ { 1 } ( s , \bar { a } ) , \ldots , f _ { M } ( s , a ) ] ^ { \top } \qquad \in \qquad \mathbb { R } ^ { M }$ and $\begin{array} { r l r l } { \phi ( s ) } & { { } } & { = } & { } \end{array}$ $\begin{array} { r l r } { [ \phi _ { 1 } ( s ) , \ldots , \phi _ { L } ( s ) ] ^ { \top } } & { { } \in } & { \mathbb { R } ^ { L } } \end{array}$ are uniformly bounded for any $s \in \ S$ and $a \in A .$ Furthermore, if we define the feature matrix $\textbf { F } ~ \in ~ \mathbb { R } ^ { | S | \cdot | \mathcal { A } | \times M }$ with its m-th column $[ f _ { m } ( s , a ) , \ s \in S , a \in A ] ^ { \top }$ for any $m \in \{ 1 , \ldots , M \}$ , and the feature matrix Φ $\in \mathbb { R } ^ { | \mathcal { S } | \times L }$ with $[ \phi _ { l } ( s ) , \ s \in \mathcal { S } ] ^ { \top }$ as its l-th column for any $l \in \{ 1 , \ldots , L \}$ , then both Φ and F have full column rank.

The assumption of the full-rank feature matrices allow us to characterize a unique asymptotically stable equilibrium in the estimations of the critic and team-averaged reward functions.

In [3], at time $t \in \mathbb { Z }$ , each agent first uses the current state $s _ { t } ,$ next state $s _ { t + 1 }$ , and its private reward $r _ { t + 1 } ^ { i }$ to compute its local TD error and reward-estimation error:

$$
\psi _ { t } ^ { i } = r _ { t + 1 } ^ { i } + \gamma V ( s _ { t + 1 } ; v _ { t } ^ { i } ) - V ( s _ { t } ; v _ { t } ^ { i } ) ,\tag{9a}
$$

$$
\xi _ { t } ^ { i } = r _ { t + 1 } ^ { i } - \bar { r } \big ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } \big ) ,\tag{9b}
$$

respectively. Then, using (9), each agent updates its parameters by interleaving local stochastic approximation steps

$$
\tilde { v } _ { t } ^ { i } = v _ { t } ^ { i } + \alpha _ { t } ^ { v } \cdot \psi _ { t } ^ { i } \cdot \nabla _ { v ^ { i } } V ( s _ { t } ; v _ { t } ^ { i } ) ,\tag{10a}
$$

$$
\tilde { \lambda } _ { t } ^ { i } = \lambda _ { t } ^ { i } + \alpha _ { t } ^ { \lambda } \cdot \xi _ { t } ^ { i } \cdot \nabla _ { \lambda ^ { i } } \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } ) ,\tag{10b}
$$

with consensus over $\mathcal { G } _ { t } \colon$

$$
v _ { t + 1 } ^ { i } = \sum _ { k \in \mathcal { B } _ { i , t } } w _ { v , t } ^ { i , k } \tilde { v } _ { t } ^ { k } , \quad \lambda _ { t + 1 } ^ { i } = \sum _ { k \in \mathcal { B } _ { i , t } } w _ { \lambda , t } ^ { i , k } \tilde { \lambda } _ { t } ^ { k } ,\tag{11}
$$

where $\alpha _ { t } ^ { v } , \alpha _ { t } ^ { \lambda } > 0$ are step sizes and the consensus weights satisfy $\begin{array} { r } { \sum _ { k \in \mathcal { B } _ { i , t } } w _ { v , t } ^ { i , k } = \sum _ { k \in \mathcal { B } _ { i , t } } w _ { \lambda , t } ^ { i , k } = 1 } \end{array}$ for all $t \in \mathbb { Z } _ { \geq 0 }$

Because $\psi _ { t } ^ { i }$ and $\xi _ { t } ^ { i }$ are formed from $r _ { t + 1 } ^ { i }$ rather than $\bar { r } ( s _ { t } , a _ { t } )$ , the local steps (10) alone are generally biased with respect to the team objective. This discrepancy is compensated for by the consensus step in (11), which drives the agents toward agreement. Consequently, every agent converges to a common limit determined by the team-average reward r¯ under some assumptions.

## C. Byzantine Resilient AC-MARL

However, such AC-MARL methods are vulnerable to Byzantine agents who deviate arbitrarily from the prescribed protocols [13], [14]. Although numerous methods have been proposed to ensure resilience [28], [30], [31], they only guarantee convergence to a neighborhood of the attack-free limit that would be achieved in the absence of Byzantine attacks. In fact, finding the exact optimal value functions in the presence of Byzantine agents is generally impossible [26]. Therefore, we consider a slightly weaker but practical model in which Byzantine agents can only attack in the communication layer:

Definition 2 (F-total Byzantine Edge Attack). Consider the two-hop communication of Definition 1. An edge $( i , j ) \in$ $\mathcal { E } _ { B } ^ { z } ( t ) \subseteq \mathcal { E } _ { t }$ is said to be under a Byzantine edge attack during round $z \in \{ 1 , 2 \}$ at time t if the message sent by agent i is arbitrarily altered or dropped before being received by agent $j ;$ that $i s ,$

$$
\begin{array} { l l } { z = 1 : } & { m _ { i } ^ { i  j } ( t ) \neq m _ { i } ( t ) , } \\ { z = 2 : } & { m _ { k } ^ { i  j } ( t ) \neq m _ { k } ^ { k  i } ( t ) , f o r s o m e k \in \mathscr { N } _ { i , t } . } \end{array}
$$

The network $\mathcal G _ { t } = ( \nu , \mathcal E _ { t } )$ is said to be under an F-total Byzantine edge attack $\begin{array} { r } { i f \sum _ { z = 1 } ^ { 2 } | \mathcal { E } _ { B } ^ { z } ( t ) | \leq F , \forall t \in \mathbb { Z } _ { \geq 0 } . } \end{array}$

We assume that Byzantine edge attackers have limited resources and can compromise at most F communications during each time step t. By definition, an attacker may corrupt either an agent’s original message or a message relayed by an intermediate agent. This attack model is closely related to the communication attack models studied in [31], [43].

Unlike much of the Byzantine MARL literature that assumes agents themselves are unreliable, we consider only unreliable communication. Thus, we assume that

Assumption 2. All agents $i \in \mathcal V$ are cooperative and follow the prescribed protocol.

Leveraging this assumption on the agents’ behavior - equivalently, that adversarial behavior is confined to the communication layer - we aim to develop AC-MARL that is fully resilient to Byzantine attacks and recovers the learning performance achievable in the absence of attacks, rather than merely guaranteeing convergence to a neighborhood of the attack-free performance. Our problem is therefore as follows:

Problem 1. Design a resilient AC-MARL algorithm such that, under Assumptions 1-2 and an F-total Byzantine edge attack, the critic, team-reward, and policy parameter estimates $\{ v _ { t } ^ { i } \}$ $\{ \lambda _ { t } ^ { i } \}$ , and $\{ \theta _ { t } ^ { i } \}$ of every agent $i \in \mathcal V$ converge almost surely to the same limits attained in the absence of attacks, using two-hop communications over $\mathcal { G } _ { t } = ( \nu , \mathcal { E } _ { t } )$

By (11), each agent updates its parameters from the messages received from its neighbors. Under an F-total Byzantine edge attack some of these messages are corrupted, but an agent cannot tell which messages to trust and filter. The problem therefore reduces to replacing the weighted average in (11) with a robust aggregation operator R such that

$$
v _ { t + 1 } ^ { i } , \lambda _ { t + 1 } ^ { i } = \mathcal { R } ( \{ m _ { k } ^ { j  i } \} _ { k \in \mathcal { B } _ { i , t } \cup \mathcal { N } _ { i , t } ^ { ( 2 ) } } )\tag{12}
$$

using only the messages available to agent i. Thus, the main challenge is to design an aggregation rule that can identify and filter unreliable messages without disrupting the learning performance.

We develop a method that guarantees each agent determines which messages to trust and filter through two-hop communication, such that the (i) actual induced communication remains connected and undirected (Lemma 2) and (ii) R completely filters out the Byzantine-induced messages in its update (Lemma 3). Combining these two results, we demonstrate that our method ensures the same convergence guarantees as in the absence of attacks (Theorems 1-2).

## IV. METHOD

Before presenting our method, we first provide the intuition. From a local perspective, agents cannot directly identify which messages from one-hop neighbors are compromised by Byzantine edge attacks. Therefore, many Byzantine-resilient methods rely on blind trimmed-mean techniques to discard outlier parameters. While effective, such filtering approaches have two fundamental limitations. First, because each agent independently filters the received messages, different agents may retain different subsets of their neighbors’ parameters, breaking the symmetry of the underlying information flow and, consequently, the doubly stochastic property of the mixing weight matrix. Second, even after filtering, Byzantine messages that remain within the accepted range can still introduce a systematic bias, preventing the agents from fully recovering the attack-free learning behavior.

Our method leverages message redundancy through two-hop communication. Specifically, each agent relays the information received from its one-hop neighbors. As a result, agent $i \in \nu$ may receive information originating from the same agent $k \in \mathcal { N } _ { i , t } ^ { ( 2 ) } \cup \mathcal { N } _ { i , t }$ through multiple distinct one-hop neighbors, corresponding to different communication paths. Under an F-total Byzantine edge attack, only a limited number of messages through these paths can be compromised. As a result, the same parameter update from the same two-hop neighbor may be relayed through multiple independent paths, providing redundant copies of the information available to agent i. By cross-checking these redundant parameter updates, agent i can aggregate messages while filtering the influence of Byzantinecorrupted information.

```latex
Algorithm 1: Fully Resilient Actor-Critic MARL
(FRAC-MARL)
Inputs: Threshold τ and step sizes $\alpha _ { t } ^ { \theta } , \alpha _ { t } ^ { v } , \alpha _ { t } ^ { \lambda }$
$/ /$ Local Update
1 Take action $a _ { t } ^ { i } \sim \pi _ { \theta _ { + } ^ { i } } ^ { i } ( \cdot \mid s _ { t } )$ , and observe next state
$s _ { t + 1 }$ and local reward $r _ { t + 1 } ^ { i }$
2 Update actor
$\delta _ { t } ^ { i } = \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } ) + \gamma V ( s _ { t + 1 } ; v _ { t } ^ { i } ) - V ( s _ { t } ; v _ { t } ^ { i } )$
$\theta _ { t + 1 } ^ { i } = \theta _ { t } ^ { i } + \alpha _ { t } ^ { \theta } \delta _ { t } ^ { i } \nabla _ { \theta ^ { i } } \log \pi _ { \theta _ { t } ^ { i } } ^ { i } ( a _ { t } ^ { i } \mid s _ { t } )$
3 Update critic and reward function
$\psi _ { t } ^ { i } = r _ { t + 1 } ^ { i } + \gamma V ( s _ { t + 1 } ; v _ { t } ^ { i } ) - V ( s _ { t } ; v _ { t } ^ { i } )$ (13a)
$\tilde { v } _ { t } ^ { i } = v _ { t } ^ { i } + \alpha _ { t } ^ { v } \psi _ { t } ^ { i } \nabla _ { v ^ { i } } V ( s _ { t } ; v _ { t } ^ { i } )$ (13b)
$\tilde { \lambda } _ { t } ^ { i } = \lambda _ { t } ^ { i } + \alpha _ { t } ^ { \lambda } \left( r _ { t + 1 } ^ { i } - \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } ) \right) \nabla _ { \lambda ^ { i } } \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } )$
(13c)
$/ /$ Two-Hop Communication
4 send $m _ { i } ( t ) : = ( \tilde { v } _ { t } ^ { i } , \tilde { \lambda } _ { t } ^ { i } , i )$ to every $j \in \mathcal { N } _ { i , t }$
5 Relay $\{ m _ { k } ^ { k \to i } ( t ) \} _ { k \in \mathcal { N } _ { i , t } }$ to every ${ \boldsymbol { j } } \in \mathcal { N } _ { i , t }$
$/ /$ Redundancy-Based Filter/Consensus
6 For each $k \in \mathcal { V } \setminus \{ i \}$ , collect the received copies of
$m _ { k } ( t )$ , with at most one copy from each relaying
agent ${ \boldsymbol { j } } \in \mathcal { N } _ { i , t }$ , and define a multi-set
$\mathcal { K } _ { t } ^ { i , k } \gets \left\{ \begin{array} { l l } { m _ { k } ^ { j \to i } ( t ) } & { | } \end{array} \right. \boldsymbol { j } \in \mathcal { N } _ { i , t } , \ m _ { k } ^ { j \to i } ( t ) \neq \emptyset \left. \right\}$
7 Collect the indices received in the two rounds,
$\mathcal { T } _ { i , t }  \{ k \in \mathcal { V } \setminus \{ i \} \mid \mathcal { K } _ { t } ^ { i , k } \neq \emptyset \}$
8 for $k \in \mathcal { T } _ { i , t }$ do
9 Find the most repeated message,
$( \hat { v } _ { t } ^ { k } , \hat { \lambda } _ { t } ^ { k } , k ) \gets \mathrm { m o d e } \big ( \mathcal { K } _ { t } ^ { i , k } \big )$ (14)
10 $\mathcal { M } _ { i , t } = \left\{ k \in \mathcal { V } \backslash \{ i \} \mid \right.$ mode count $( \mathcal { K } _ { t } ^ { i , k } ) \geq \tau \Big \}$
11 Update the parameters
${ v } _ { t + 1 } ^ { i } = { w } _ { v , t } ^ { i , i } \tilde { v } _ { t } ^ { i } + \sum _ { j \in \mathcal { M } _ { i , t } } { w } _ { v , t } ^ { i , j } \hat { v } _ { t } ^ { j } ,$ (15a)
$\lambda _ { t + 1 } ^ { i } = w _ { \lambda , t } ^ { i , i } \tilde { \lambda } _ { t } ^ { i } + \sum _ { j \in \mathcal { M } _ { i , t } } w _ { \lambda , t } ^ { i , j } \hat { \lambda } _ { t } ^ { j } ,$ (15b)
where $\begin{array} { r } { \sum _ { j \in { \mathcal { M } } _ { i , t } \cup \{ i \} } w _ { v , t } ^ { i , j } = \sum _ { j \in { \mathcal { M } } _ { i , t } \cup \{ i \} } w _ { \lambda , t } ^ { i , j } = 1 } \end{array}$
```

## A. Fully Resilient Actor-Critic MARL (FRAC-MARL)

The FRAC-MARL (outlined in Algorithm 1) operates by combining local actor-critic updates (lines 1-3), adopted from [3, Algorithm 2], with redundancy-based resilient aggregation (lines 4-11), which corresponds to the robust aggregator $\mathcal { R }$ in (12). At each time step $t \in \mathbb { Z } _ { \geq 0 } .$ , agent $\textit { i } \in \textit { \textbf { V } }$ first performs the standard local actor-critic updates (lines 1-3), yielding intermediate value-function and reward-model parameters $\tilde { v } _ { t } ^ { i }$ and $\tilde { \lambda } _ { t } ^ { i } .$ , respectively.

After local learning, agent i broadcasts its message $m _ { i } ( t ) =$ $( \tilde { v } _ { t } ^ { i } , \tilde { \lambda } _ { t } ^ { i } , i )$ to its one-hop neighbors. Then each agent relays the messages it receives $\{ m _ { k } ^ { k \to i } ( t ) \} _ { k \in \mathcal { N } _ { i } } ,$ to its own onet

![](images/718353979eecf0b35a4d5e5a4dd43688ed0a067ef45996e2e3e293d07530adf6.jpg)  
Fig. 1. Visualizations of (a) graph $\mathcal { G } _ { t } = ( \nu , \pmb { \varepsilon } _ { t } )$ and (b) its 5-2-hop graph $\mathcal { G } _ { t } ^ { 5 } = ( \nu , \varepsilon _ { t } ^ { 5 } )$ at time t. The edge $( i , j ) \in \mathcal { E } _ { t } ^ { 5 }$ , since $| \pmb { \mathcal { B } } _ { i , t }$ ∩ $\bar { \mathcal { N } } _ { j , t } | \geq 5$

hop neighbors, allowing information to propagate over twohop communication paths (lines 4-5). Then agent i stores the messages with claimed index k into the multiset $\mathcal { K } _ { t } ^ { i , k }$ , one per relaying agent, for every $k \in \mathcal { V } \setminus \{ i \}$ (line 6). Note that, an honest relay forwards at most one message per origin, so $\mathcal { K } _ { t } ^ { i , k }$ receives more than one tuple from $j$ only if $( j , i )$ is attacked. In this case, agent i retains an arbitrary message for the index k. Then, to filter out Byzantine-influenced messages, agent i performs redundancy-based filtering by constructing the set $\mathcal { M } _ { i , t }$ containing only agents k whose relayed parameter updates in $\mathcal { K } _ { t } ^ { i , k }$ have a mode appearing at least τ times (lines 7- 10). Finally, the accepted critic and reward-model parameters are aggregated via weighted averaging to update $v _ { t } ^ { i }$ and $\lambda _ { t } ^ { i }$ (line 11).

## B. (r, r<sup>′</sup>)-redundancy

Note that FRAC-MARL relies on the redundancy-based filtering (lines 7-10). The required level of redundancy, denoted by τ, is a user-defined threshold that affects the filtering process and, consequently, the learning performance. In this subsection, we introduce a novel topological property that allows us to theoretically characterize the conditions under which the proposed filtering mechanism and FRAC-MARL achieve the desired performance.

First, we define an r-2-hop graph which is defined as below:

Definition 3 (r-2-hop Graph). Let $\begin{array} { r c l } { \mathcal { G } _ { t } } & { = } & { ( \mathcal { V } , \mathcal { E } _ { t } ) } \end{array}$ be an undirected graph at time t. We define the r-2-hop graph of $\mathcal { G } _ { t } ,$ , denoted as $\mathcal { G } _ { t } ^ { r } = ( \nu , \mathcal { E } _ { t } ^ { r } )$ , such that an edge $( i , j ) \in \mathcal { E } _ { t } ^ { r }$ if $| B _ { i , t } \cap { \cal N } _ { j , t } | \geq r ,$ where $B _ { i , t } = \mathcal { N } _ { i , t } \cup \{ i \}$

An illustrative example of an $r – 2$ -hop graph is given in Figure 1. An $r – 2 .$ -hop graph contains an edge $( i , j )$ if agents i and $j$ share at least r neighbors (including direct links). Equivalently, there are at least r vertex-disjoint paths of length at most 2 connecting them.

Definition 4 $( ( \boldsymbol { r } , \boldsymbol { r ^ { \prime } } )$ -redundant). Let $\mathcal G _ { t } ~ = ~ ( \nu , \mathcal E _ { t } )$ be an undirected graph at time t, and let $\mathcal { G } _ { t } ^ { r } = ( \nu , \mathcal { E } _ { t } ^ { r } )$ be its r-2-hop graph. We say that $\mathcal { G } _ { t }$ is $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundant with $r > r ^ { \prime } \geq 0$ at time t if:

1) $\mathcal { G } _ { t } ^ { r }$ is connected, and

2) for all $( i , j ) \notin \mathcal { E } _ { t } ^ { r } , | \boldsymbol { \mathcal { B } } _ { i , t } \cap \mathcal { N } _ { j , t } | \leq r ^ { \prime } .$

Figure 2 visualizes an $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundant graph and its $r { - } 2 { - }$ hop graph. A graph $\mathcal { G } _ { t }$ is $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundant if two things hold. First, its r-2-hop graph is connected. Second, for all agent pairs not connected in the r-2-hop graph, they share at most $r ^ { \prime }$ neighbors (including direct links) in the graph $\mathcal { G } _ { t }$ . That is, for any $i , j \in \mathcal { V }$ , they share either at least r or at most $r ^ { \prime }$ neighbors (including direct links).

![](images/645219e4c1c1730231c9a5f1c416f7a701d4a90c5b820feb40d95a7abfa17672.jpg)  
(a)

![](images/60c9cf798e32045e394da99d577e4b8b85d80c9a9c106f1a298be4d2d44775f6.jpg)  
(b)  
Fig. 2. (a) (1, 0)-redundant graph and (b) its 1-2-hop graph.

Going back to Figure 2 as an example, for every edge $( i , j )$ in the original graph (Figure $ { 2 } ( \mathrm { a } ) )$ , we have $| B _ { i , t } \cap { \cal N } _ { j , t } | = 1$ Setting $r = 1$ , its r-2-hop graph (Figure $2 \ ( \mathbf { b } ) )$ is connected. Because every pair $i , j \in \mathcal { V }$ of nodes not connected in the $r { - } 2 { - }$ hop graph satisfies $| B _ { i , t } \cap { \cal N } _ { j , t } | = 0$ , the graph in Figure 2 (a) is (1, 0)-redundant.

Remark 1. The $( r , r ^ { \prime } )$ -redundancy condition quantifies the topological condition required for agents to have symmetric validations of correct messages amidst a bounded number of adversarial communication attacks. The threshold r ensures that for any two adjacent agents in the r-2-hop graph, there is sufficient multi-path redundancy (at least r common neighbors) to verify the relayed messages.

Conversely, when two agents are not connected in the r-2-hop graph, the number of neighbors they share is bounded by $r ^ { \prime } . \ \mathrm { A s \ a }$ result, this prevents Byzantine edge attacks from manipulating relayed messages to create asymmetric validation (e.g., agent i having sufficient redundancy in messages from agent j while the reverse does not hold). This bound therefore is intended to ensure the symmetry of the underlying communication, which is shown later in Lemma 2.

Now, we provide a useful property related to $( r , r ^ { \prime } ) .$ redundancy:

Lemma 1. Let $\mathcal G _ { t } ~ = ~ ( \nu , \mathcal E _ { t } )$ be $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundant at time t. Then, its r-2-hop graph $\mathcal { G } _ { t } ^ { r } ~ = ~ ( \nu , \mathcal { E } _ { t } ^ { r } )$ is connected and undirected at time t.

Proof. By Definition 4, G<sup>r</sup> is connected. Furthermore, because $\mathcal { G } _ { t }$ is undirected, $| B _ { i , t } \cap \dot { N } _ { j , t } | = | B _ { j , t } \cap \mathcal { N } _ { i , t } |$ for any $i , j \in \mathcal { V }$ $i \neq j$ . Hence, $\mathcal { G } _ { t } ^ { r }$ is also undirected. □

Lemma 1 characterizes the connectivity and symmetry properties of $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundancy. These results provide the foundation for the subsequent analysis of the robust aggregation mechanism presented in the next subsection.

## C. Theoretical Analysis

To further facilitate the convergence analysis of our algorithm, we define an induced communication graph:

Definition 5 (Induced Communication Graph). Let each agent execute FRAC-MARL and construct the filtered-neighbor set $\mathcal { M } _ { i , t } = \left\{ k \in \mathcal { V } \backslash \{ i \} \right.$ | mode count $( \mathcal { K } _ { t } ^ { i , \bar { k } } ) \geq \tau \Big \}$ , where multisets $\mathcal { K } _ { t } ^ { i , k }$ are constructed in line 6 of Algorithm 1. The induced communication graph is $\mathcal { G } _ { t } ^ { \mathrm { i n d } } = ( \nu , \mathcal { E } _ { t } ^ { \mathrm { i n d } } )$ , where $( i , k ) \in \mathcal { E } _ { t } ^ { \mathrm { i n d } } \iff k \in \mathcal { M } _ { i , t } .$

The induced graph $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$ captures the information flow where agents receive the information for update even after filtering step (lines 7-10) in FRAC-MARL. In general, $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$ is a directed graph since it is possible that $k \in \mathcal { M } _ { i , t }$ while $i \notin \mathcal { M } _ { k , t }$

For each t, we define two weight matrices $W _ { \lambda , t } = [ w _ { \lambda , t } ^ { i , k } ] \in$ $\mathbb { R } ^ { n \times n }$ and $W _ { v , t } = [ w _ { v , t } ^ { i , k } ] \in \mathbb { R } ^ { n \times n }$ associated with $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$ such that

$$
w _ { \lambda , t } ^ { i , k } = w _ { v , t } ^ { i , k } = \left\{ \begin{array} { l l } { \frac { 1 } { n } } & { \mathrm { i f ~ } ( i , k ) \in \mathcal { E } _ { t } ^ { \mathrm { i n d } } } \\ { 1 - \frac { | \mathcal { M } _ { i , t } | } { n } } & { \mathrm { i f ~ } k = i } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{16}
$$

Lemma 2. Let Assumption 2 hold. Let $\mathcal { G } _ { t } = ( \nu , \mathcal { E } _ { t } )$ be $( r , r -$ $2 F { - } 1 )$ -redundant where $r > 2 F ,$ , and suppose that each agent $i \in \nu$ runs the FRAC-MARL algorithm with ${ \boldsymbol { \tau } } = { \boldsymbol { r } } - { \boldsymbol { F } } $ under an F-total Byzantine edge attack for all $t \in \mathbb { Z } _ { > 0 }$ . Then, for all $t \in \mathbb { Z } _ { > 0 }$ , the induced communication graph $\bar { \boldsymbol g } _ { t } ^ { \mathrm { i n d } }$ is equal to the $r { - } 2 { - } h o p$ graph $\mathcal { G } _ { t } ^ { r } ~ = ~ ( \nu , \mathcal { E } _ { t } ^ { r } )$ of $\mathcal { G } _ { t } ,$ i.e. $\mathcal { E } _ { t } ^ { \mathrm { i n d } } = \mathcal { E } _ { t } ^ { r }$ Furthermore, $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$ is connected and undirected.

Proof. By definition, $( i , k ) \in \mathcal { E } _ { t } ^ { \mathrm { i n d } }$ if and only if $k \in \mathcal { M } _ { i , t }$ , so it suffices to show

$$
( i , k ) \in \mathcal { E } _ { t } ^ { r } \iff k \in \mathcal { M } _ { i , t } .\tag{17}
$$

We show in three parts.

Part 1 (⇒): Suppose first that $( i , k ) \in \mathcal { E } _ { t } ^ { r }$ . Then, $| \boldsymbol { B } _ { i , t } \cap$ $\mathcal { N } _ { k , t } | ~ \geq ~ r$ . By Assumption 2, all agents $\textit { i } \in \textit { \textbf { V } }$ generate and transmit the correct parameter pairs $( \tilde { v } _ { t } ^ { i } , \tilde { \lambda } _ { t } ^ { i } )$ according to (13) (lines 1-3 in Algorithm 1). Since at most $F$ relayed transmissions can be corrupted under an F-total Byzantine edge attack, agents i and k receive at least $r - F$ identical copies of each other’s parameter tuple. Because FRAC-MARL uses the threshold ${ \boldsymbol { \tau } } = { \boldsymbol { r } } - { \boldsymbol { F } }$ , both agents accept one another, i.e., we have $k \in \mathcal { M } _ { i , t }$

Part 2 (⇐): Now assume to the contrary that $k \in \mathcal { M } _ { i , t }$ and $( i , k ) \notin \mathcal { E } _ { t } ^ { r }$ . Then since $\mathrm { ( i ) } ~ ( i , k ) \notin \mathcal { E } _ { t } ^ { r }$ and (ii) $\mathcal { G } _ { t }$ is $( r , r - 2 F - 1 )$ -redundant, $| B _ { i , t } \cap \mathcal { N } _ { k , t } | \leq r - 2 F - 1 . \mathrm { ~ A ~ }$ tuple claiming the origin of k can reach agent i in only two ways. First, along an uncorrupted path: (i) agent k broadcasts to $\mathcal { N } _ { k , t }$ in the first round, and $j \in \mathcal { N } _ { i , t } \cap \mathcal { N } _ { k , t }$ forwards it to i in the second round, and (ii) i receives it directly whenever $i \in \mathcal { N } _ { k , t } ;$ the number of such paths is exactly $| B _ { i , t } \cap \mathcal { N } _ { k , t } |$ . Second, along a compromised edge: an attacker may alter a transmission so that it carries a tuple labeled with origin k, whether or not the sender ever received one. Each compromised edge contributes at most one such tuple to $\mathcal { K } _ { t } ^ { i , k }$ , and at most $F$ edges are compromised in total. Hence

$$
\begin{array} { r } { | \mathcal { K } _ { t } ^ { i , k } | \leq | \mathcal { B } _ { i , t } \cap \mathcal { N } _ { k , t } | + F \leq ( r - 2 F - 1 ) + F < \tau = r - F , } \end{array}
$$

so mode co $\mathrm { u n t } ( \mathcal { K } _ { t } ^ { i , k } ) \leq | { \mathcal { K } } _ { t } ^ { i , k } | < \tau$ and $k \notin \mathcal { M } _ { i , t }$ , which is a contradiction.

Part 3: Combining Parts 1 and 2, we have shown that (17) holds. Thus, $\mathcal { E } _ { t } ^ { \mathrm { i n d } } = \mathcal { E } _ { t } ^ { r }$ . Also, together with Lemma 1, this implies $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$ is undirected and connected. □

Lemma 2 shows that the (i) communication graph induced by the FRAC-MARL filtering mechanism coincides with the r-2-hop graph of the underlying communication graph G<sub>t</sub>, and thus (ii) is undirected and connected at every t. Thus, the resulting weight matrices $W _ { v , t }$ and $W _ { \lambda , }$ <sub>t</sub> defined in (16) are doubly stochastic. Also their nonzero elements are uniformly bounded below by $1 / n$

Lemma 3. Let Assumption 2 hold. Let $\mathcal { G } _ { t } = ( \nu , \mathcal { E } _ { t } )$ be $( r , r -$ $2 F - 1 )$ )-redundant where $r \ > \ 2 F _ { \mathrm { ~ \scriptsize ~ 2 ~ 5 ~ } }$ , and suppose that each agent $i \in \mathcal V$ runs the FRAC-MARL algorithm with $\tau = r -$ $F$ under an F-total Byzantine edge attack for all $t \in \mathbb { Z } _ { \geq 0 } .$ Then, for any time $t \in \mathbb { Z } _ { \geq 0 }$ , agent $i \in \mathcal V ,$ , and every accepted agent indices $k \in \mathcal { M } _ { i , t } ,$ , we have $( \hat { v } _ { t } ^ { k } , \hat { \lambda } _ { t } ^ { k } ) = ( \tilde { v } _ { t } ^ { k } , \tilde { \lambda } _ { t } ^ { k } )$ , where $( \hat { v } _ { t } ^ { k } , \hat { \lambda } _ { t } ^ { k } )$ and $( \tilde { v } _ { t } ^ { k } , \tilde { \lambda } _ { t } ^ { k } )$ are defined in (14) and (13), respectively. Consequently, the consensus update laws (15) depend only on true local parameter estimates of agents and are independent of Byzantine-modified transmissions.

Proof. By Assumption 2, all agents $k \in \mathcal V$ generate parameter pairs $( \tilde { v } _ { t } ^ { k } , \tilde { \lambda } _ { t } ^ { k } )$ ) according to (13) and transmit them. Also, since $\mathcal { G } _ { t }$ is $( r , r - 2 F - 1 )$ )-redundant and is under F-total Byzantine edge attack, any pair of agents connected in the r-2-hop graph has at least $r - F$ independent relayed copies of the correct message at each time step t.

Now assume to the contrary that $( \hat { v } _ { t } ^ { k } , \hat { \lambda } _ { t } ^ { k } ) \neq ( \tilde { v } _ { t } ^ { k } , \tilde { \lambda } _ { t } ^ { k } )$ Acceptance requires at least $\tau = r - F$ identical occurrences. Since only F transmissions can be corrupted in total, at most $F$ copies of any altered value can exist. However, since $r > 2 F$ we get $\tau = r - F > F$ , which leads to a contradiction.

Using these lemmas, we now provide our main results. To establish almost sure convergence under linear function approximation with Assumption 1, we formalize the following standard regularity conditions regarding the reward bounds, learning step sizes, and parameter spaces.

Assumption 3. For every agent $i \in \mathcal { V } ,$ , state $s \in S .$ , action $a ^ { i } \in { \mathcal { A } } ^ { i }$ , and parameter $\theta ^ { i } \in \Theta ^ { i }$ , the local policy $\pi _ { \theta ^ { i } } ^ { i } ( a ^ { i } \mid s )$ is positive and continuously differentiable with respect to $\theta ^ { i }$ over $\Theta ^ { i }$

Assumption 4. The local reward function $r ^ { i } ( s , a )$ is uniformly bounded for every agent $i \in \mathcal V$ , state $s \in S$ , and joint action $a \in { \mathcal { A } }$

Assumption 5. The stochastic approximation step sizes $\alpha _ { t } ^ { v }$ $\alpha _ { t } ^ { \lambda }$ , and $\alpha _ { t } ^ { \theta }$ are positive sequences satisfying:

$$
\sum _ { t = 0 } ^ { \infty } \alpha _ { t } ^ { q } = \infty , \quad \sum _ { t = 0 } ^ { \infty } ( \alpha _ { t } ^ { q } ) ^ { 2 } < \infty , \quad \forall q \in \{ v , \lambda , \theta \} ,\tag{18}
$$

along with the two-time-scale condition $\alpha _ { t } ^ { \theta } = o ( \alpha _ { t } ^ { v } ) = o ( \alpha _ { t } ^ { \lambda } )$ and the limit tracking property $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } \alpha _ { t + 1 } ^ { q } / \alpha _ { t } ^ { q } = 1 } \end{array}$

Assumption 6. For agent $i \in \mathcal V .$ , the parameter space $\Theta ^ { i } \subset$ $\mathbb { R } ^ { b _ { i } }$ is compact and hyper-rectangular. The policy parameter update includes a projection operator $\Psi _ { \Theta ^ { i } } : \mathbb { R } ^ { b _ { i } }  \Theta ^ { i }$ that maps any exterior iterate back onto the boundary of $\Theta ^ { i }$

Assumptions 3 and 4 provide the regularity and boundedness conditions needed for the policy-gradient and TD-error terms to be well defined. Assumption 5 establishes the twotime-scale stochastic approximation framework, which is common in actor-critic setting [42]. Since the actor parameters $\theta _ { t }$ change asymptotically more slowly than the critic and reward parameters $v _ { t }$ and $\lambda _ { t } , \ v _ { t }$ and $\lambda _ { t }$ evolve on a faster timescale than $\theta _ { t } ,$ , allowing the critic and reward parameter estimates to track their equilibria for the current policy while the actor parameters are effectively quasi-static. Finally, Assumption 6 ensures that projection operator $\Psi _ { \Theta } .$ i satisfies the conditions required by the projected stochastic approximation analysis used to establish convergence of the limiting ODEs, following the analysis of [3], [30]. Using $\Psi _ { \Theta ^ { i } }$ , for $\boldsymbol { y } \in \mathbb { R } ^ { b _ { i } }$ and $\theta ^ { i } \in \Theta ^ { i }$ we further define

$$
\hat { \Psi } _ { \Theta ^ { i } } [ y ] : = \operatorname * { l i m } _ { 0 < \eta \to 0 } \frac { \Psi _ { \Theta ^ { i } } ( \theta ^ { i } + \eta y ) - \theta ^ { i } } { \eta } ,\tag{19}
$$

the projection of $y$ onto the tangent cone of $\Theta ^ { i }$ at $\theta ^ { i } .$ , which reduces to y whenever $\theta ^ { i }$ lies in the interior of $\Theta ^ { i }$ . When the limit in (19) is not unique, we define $\hat { \Psi } _ { \Theta ^ { i } } [ y ]$ as the set of all possible limit points.

We first show the convergence of the critic and team reward approximation functions. Let $\mathbf { D } _ { s } ^ { \theta } = \mathbf { \Phi }$ diag $\left( \left\lceil d _ { \theta } ( s ) , s \in \mathcal { S } \right\rceil \right) \in$ $\hat { \mathbb { P } } ^ { | \mathcal { S } | \times | \mathcal { S } | }$ and ${ \bf D } _ { s , a } ^ { \theta } = \mathrm { d i a g } \left( \bar { \left[ d _ { \theta } ^ { \prime } ( s , a ) , \ ^ { s } \in S , a \in A \right] } \right) ^ { \prime } \ \in$ $\mathbb { R } ^ { | \boldsymbol { S } | \cdot | \boldsymbol { A } | \times | \boldsymbol { S } | \cdot | \boldsymbol { A } | }$ denote the diagonal matrices corresponding to the state distribution and state-action distribution induced by policy $\pi _ { \theta } .$ respectively. Also, we denote $\textbf { R } = ~ \left\lceil \bar { r } ( s , a ) , ~ s \in \mathcal { S } , a \in \mathcal { A } \right\rceil ^ { \top } ~ \in ~ \mathbb { R } ^ { | \mathcal { S } | \cdot | \mathcal { A } | }$ and $\begin{array} { r l } { \mathbf { R } _ { \theta } } & { { } = } \end{array}$ $\begin{array} { r } { \left\lceil \sum _ { a \in \mathcal { A } } \pi _ { \theta } ( a \mid s ) \bar { r } ( s , a ) , \ s \in \mathcal { S } \right\rceil ^ { \top } \ \in \ \mathbb { R } ^ { | \mathcal { S } | } } \end{array}$ . We denote $\mathbf { P } _ { \theta } =$ $\left\lceil \overline { { P } } _ { \theta } ( s ^ { \prime } \mid s ) , s \in S , s ^ { \prime } \in S \right\rceil \in \mathbb { R } ^ { | \bar { \cal S } | \times | S | }$

Theorem 1. Let Assumptions 1-5 hold andfix $\theta \in \Theta .$ . Suppose that, for every $t \in \ \mathbb { Z } _ { \geq 0 } ,$ , (i) the graph $\mathcal G _ { t } ~ = ~ ( \nu , \mathcal E _ { t } )$ is $( r , r - 2 F - 1 )$ -redundant where $r > 2 F$ , and (ii) every agent $i \in \mathcal V$ selects $a _ { t } ^ { i } \sim \pi _ { \theta ^ { i } } ^ { i } ( { \cdot } \ | \ s _ { t } )$ and updates $\{ v _ { t } ^ { i } \}$ and $\{ \lambda _ { t } ^ { i } \}$ via (13) and (15) of FRAC-MARL with threshold $\tau = r - F$ and consensus weights (16), under an F-total Byzantine edge attack. Then

$$
\operatorname* { l i m } _ { t \to \infty } v _ { t } ^ { i } = v _ { \theta } , \quad \quad \operatorname* { l i m } _ { t \to \infty } \lambda _ { t } ^ { i } = \lambda _ { \theta } , \quad f o r a l l i \in \mathcal { V } ,\tag{20}
$$

almost surely, where $\lambda _ { \theta }$ and v<sub>θ</sub> are the unique solutions to

$$
\begin{array} { r } { \mathbf { F } ^ { \top } \mathbf { D } _ { s , a } ^ { \theta } ( \mathbf { R } - \mathbf { F } \lambda _ { \theta } ) = 0 , } \end{array}\tag{21a}
$$

$$
\begin{array} { r } { \Phi ^ { \top } \mathbf { D } _ { s } ^ { \theta } ( \mathbf { R } _ { \theta } + \gamma \mathbf { P } _ { \theta } \Phi v _ { \theta } - \Phi v _ { \theta } ) = 0 . } \end{array}\tag{21b}
$$

Proof. By Lemma $^ { 3 , }$ all Byzantine-modified transmissions are removed by FRAC-MARL before the consensus update. Therefore, the critic and reward parameter updates are equivalent to those of a standard decentralized actor-critic algorithm operating over the induced communication graph $\mathcal { G } _ { t } ^ { \mathrm { i n d } }$

Furthermore, by Lemma 2, the induced graph is connected and undirected. This means the corresponding weight matrices $W _ { \lambda , t }$ and $W _ { v , t }$ as defined in (16) are doubly stochastic by construction. Because the weight matrices are doubly stochastic with uniformly positive diagonal entries and satisfy the required connectivity condition, we get spectral radii $\begin{array} { r } { \operatorname* { s u p } _ { t } \rho \big ( \bar { W } _ { q , t } ( I \ - \ \frac { 1 } { n } \mathbf { 1 } \mathbf { 1 } ^ { \top } ) W _ { q , t } \big ) < 1 } \end{array}$ for $q \in$ $\{ v , \lambda \}$ . Furthermore, $\mathcal { G } _ { t }$ is independent of the sample path $\{ \left( s _ { t ^ { \prime } } , a _ { t ^ { \prime } } , r _ { t ^ { \prime } + 1 } \right) \} _ { t ^ { \prime } \geq 0 }$ and the iterates $\{ v _ { t ^ { \prime } } ^ { i } , \lambda _ { t ^ { \prime } } ^ { i } , \theta _ { t ^ { \prime } } ^ { i } \} _ { t ^ { \prime } \geq 0 , i \in \mathcal { V } }$

Then because $W _ { \lambda , t }$ and $W _ { v , t }$ are determined by $\mathcal { G } _ { t }$ alone, $W _ { \lambda , t }$ and $W _ { v , t }$ satisfy [24, Assumption 4].

We first stack the parameters into

$$
\lambda _ { t } = \left[ ( \lambda _ { t } ^ { 1 } ) ^ { \top } \quad \cdots \quad ( \lambda _ { t } ^ { n } ) ^ { \top } \right] ^ { \top } , \ v _ { t } = \left[ ( v _ { t } ^ { 1 } ) ^ { \top } \quad \cdots \quad ( v _ { t } ^ { n } ) ^ { \top } \right] ^ { \top } .
$$

Then the update laws for both parameters can be compactly written as

$$
\lambda _ { t + 1 } = \left( W _ { \lambda , t } \otimes I \right) \left( \lambda _ { t } + \alpha _ { t } ^ { \lambda } ( A _ { t } ^ { \lambda } \lambda _ { t } + b _ { t } ^ { \lambda } ) \right)\tag{22a}
$$

$$
v _ { t + 1 } = ( W _ { v , t } \otimes I ) \left( v _ { t } + \alpha _ { t } ^ { v } ( A _ { t } ^ { v } v _ { t } + b _ { t } ^ { v } ) \right) ,\tag{22b}
$$

where

$$
\begin{array} { r l } & { A _ { t } ^ { \lambda } = - I \otimes f ( s _ { t } , a _ { t } ) f ( s _ { t } , a _ { t } ) ^ { \top } , } \\ & { A _ { t } ^ { v } = I \otimes \phi ( s _ { t } ) ( \gamma \phi ( s _ { t + 1 } ) - \phi ( s _ { t } ) ) ^ { \top } , } \end{array}
$$

and

$$
\boldsymbol { b } _ { t } ^ { \lambda } = \left[ \begin{array} { c } { f ( s _ { t } , a _ { t } ) r _ { t + 1 } ^ { 1 } } \\ { \vdots } \\ { f ( s _ { t } , a _ { t } ) r _ { t + 1 } ^ { n } } \end{array} \right] , \boldsymbol { b } _ { t } ^ { v } = \left[ \begin{array} { c } { \phi ( s _ { t } ) r _ { t + 1 } ^ { 1 } } \\ { \vdots } \\ { \phi ( s _ { t } ) r _ { t + 1 } ^ { n } } \end{array} \right] .
$$

The compact form (22) is mathematically in the same form as those defined in [24, Lemmas 1 and 3]. By [24, Lemma 1], sup<sub>t</sub> $\| \lambda _ { t } \| < \infty$ and sup<sub>t</sub> $\left\| v _ { t } \right\| < \infty$ almost surely, and by [24, Lemma 3], lim $\begin{array} { r } { \mathsf { \iota } _ { t  \infty } \| \lambda _ { t } - 1 \otimes \bar { \lambda } _ { t } \| = \operatorname* { l i m } _ { t  \infty } \| v _ { t } - 1 \otimes \bar { v } _ { t } \| = 0 } \end{array}$ almost surely, where

$$
\bar { \lambda } _ { t } = \frac 1 n ( { \bf 1 } ^ { \top } \otimes I ) \lambda _ { t } , \qquad \bar { v } _ { t } = \frac 1 n ( { \bf 1 } ^ { \top } \otimes I ) v _ { t } .
$$

Premultiplying (22) by $\smash { \frac { 1 } { n } ( \mathbf { 1 } ^ { \top } \otimes I ) }$ and using $\mathbf { 1 } ^ { \top } W _ { \lambda , t } ~ =$ $\mathbf { 1 } ^ { \top } W _ { v , t } \bar { = } \mathbf { 1 } ^ { \top }$ yields the closed recursions

$$
\begin{array} { r l } & { { { \bar { \lambda } } _ { t + 1 } } = { { \bar { \lambda } } _ { t } } + \alpha _ { t } ^ { \lambda } \left( { - f ( s _ { t } , a _ { t } ) f ( s _ { t } , a _ { t } ) ^ { \top } { { \bar { \lambda } } _ { t } } + f ( s _ { t } , a _ { t } ) { { \bar { r } } _ { t + 1 } } } \right) , } \\ & { { { \bar { v } } _ { t + 1 } } = { { \bar { v } } _ { t } } + \alpha _ { t } ^ { v } \left( { \phi ( s _ { t } ) ( \gamma \phi ( s _ { t + 1 } ) - \phi ( s _ { t } ) ) ^ { \top } { { \bar { v } } _ { t } } + \phi ( s _ { t } ) { { \bar { r } } _ { t + 1 } } } \right) } \end{array}
$$

where $\begin{array} { r c l } { \bar { r } _ { t + 1 } } & { = } & { \frac { 1 } { n } \sum _ { i \in \mathcal { V } } r _ { t + 1 } ^ { i } } \end{array}$ is the team-averaged realized reward. By [3, Appendix B.4 Step 2], under Assumptions 1, 3, 4, 5, and that since $\left\{ \left( s _ { t } , a _ { t } \right) \right\}$ is irreducible and aperiodic with stationary distribution $d _ { \theta } ^ { \prime }$ , the asymptotic behaviors of $\bar { \lambda } _ { t }$ and $\bar { v } _ { t }$ are described by

$$
\dot { \lambda } = - \mathbf { F } ^ { \top } \mathbf { D } _ { s , a } ^ { \theta } \mathbf { F } \lambda + \mathbf { F } ^ { \top } \mathbf { D } _ { s , a } ^ { \theta } \mathbf { R } ,\tag{23a}
$$

$$
\dot { \boldsymbol { v } } = \Phi ^ { \top } \mathbf { D } _ { s } ^ { \theta } ( \gamma \mathbf { P } _ { \theta } - I ) \Phi \boldsymbol { v } + \Phi ^ { \top } \mathbf { D } _ { s } ^ { \theta } \mathbf { R } _ { \theta } .\tag{23b}
$$

Since $- \mathbf { F } ^ { \top } \mathbf { D } _ { s , a } ^ { \theta } \mathbf { F }$ and $\Phi ^ { \top } \mathbf { D } _ { s } ^ { \theta } ( \gamma \mathbf { P } _ { \theta } - I ) \Phi$ are Hurwitz under Assumptions 1 and 3, the ODEs (23) admit the unique globally asymptotically stable equilibria $\lambda _ { \theta }$ and $v _ { \theta }$ , respectively (and thus solutions to (21)). Combined with $\mathrm { s u p } _ { t } \| \lambda _ { t } \| < \infty$ and sup<sub>t</sub> $\left\| v _ { t } \right\| < \infty$ , this gives $\bar { \lambda } _ { t } \ \to \ \lambda _ { \theta }$ and $\bar { v } _ { t } \ \to \ v _ { \theta }$ almost surely. Therefore, since $\| \lambda _ { t } ^ { i } - \bar { \lambda } _ { t } \| \leq \| \lambda _ { t } - 1 \otimes \bar { \lambda } _ { t } \|$

$$
\begin{array} { r } { \| \lambda _ { t } ^ { i } - \lambda _ { \theta } \| \leq \| \lambda _ { t } ^ { i } - \bar { \lambda } _ { t } \| + \| \bar { \lambda } _ { t } - \lambda _ { \theta } \|  0 , } \end{array}
$$

and similarly,

$$
\| v _ { t } ^ { i } - v _ { \theta } \| \leq \| v _ { t } ^ { i } - \bar { v } _ { t } \| + \| \bar { v } _ { t } - v _ { \theta } \|  0
$$

almost surely. Thus, lim $1 _ { t  \infty } \lambda _ { t } ^ { i } = \lambda _ { \theta }$ and ${ \operatorname* { l i m } } _ { t \to \infty } v _ { t } ^ { i } = v _ { \theta }$ almost surely for every $i \in \mathcal V$ , completing the proof. □

Theorem 1 depends on Assumptions 1-5. Except for ${ \mathrm { A s } } -$ sumption 2, all other assumptions are standard in reinforcement learning and can be applied to a broad class of multiagent systems, including multi-robot task allocation, navigation, and formation control, where agents employ stochastic policies with bounded rewards over a prescribed operating domain. We discuss the implications of Assumption 2 in more detail later in this section.

Theorem 1 establishes that, despite Byzantine edge attacks, the proposed algorithm enables all agents to reach consensus on the critic and team-averaged reward approximations almost surely. Specifically, the parameter estimates converge to $\lambda _ { \theta }$ and $v _ { \theta } ,$ which are, respectively, the least-squares approximation of the global reward function $\bar { r }$ and the unique solution to the mean square projected Bellman equation under the linear function approximation.

Having established convergence of the critic and teamreward parameter estimates, we next discuss the convergence of the actor in a slower timescale:

Theorem 2. Let Assumptions 1-6 hold. Suppose that,for every $t \in \mathbb { Z } _ { \geq 0 } , ( i )$ the graph $\mathcal { G } _ { t } = ( \nu , \mathcal { E } _ { t } )$ is $( r , r { - } 2 F { - } 1 )$ -redundant where $r > 2 F$ , and (ii) every agent $i \in \mathcal V$ executes the FRAC-MARL algorithm with threshold $\tau = r - F$ and consensus weights (16), under an F-total Byzantine edge attack. Then, for every agent $\textit { i } \in \mathcal { V } _ { : }$ , the policy parameter $\theta _ { t } ^ { i }$ converges almost surely to a point in the set of locally asymptotically stable equilibria of the ordinary differential equation

$$
\begin{array} { r } { \dot { \theta } ^ { i } = \hat { \Psi } _ { \boldsymbol { \Theta } ^ { i } } \left[ \mathbb { E } _ { d _ { \theta } , \pi _ { \theta } , P } \left[ \delta _ { \theta } ( \boldsymbol { s } _ { t } , \boldsymbol { a } _ { t } , \boldsymbol { s } _ { t + 1 } ) \nabla _ { \theta ^ { i } } \log \pi _ { \theta ^ { i } } ^ { i } ( \boldsymbol { a } _ { t } ^ { i } \mid \boldsymbol { s } _ { t } ) \right] \right] , } \end{array}\tag{24}
$$

with

$$
\delta _ { \theta } ( s _ { t } , a _ { t } , s _ { t + 1 } ) = \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { \theta } ) + \gamma V ( s _ { t + 1 } ; v _ { \theta } ) - V ( s _ { t } ; v _ { \theta } ) ,
$$

where parameters $\lambda _ { \theta }$ and v<sub>θ</sub> are the globally asymptotically stable equilibria under the global policy $\pi _ { \theta } .$

Proof. The proof proceeds by combining the convergence result established in Theorem 1 with the analysis developed in [3, Theorem 4.10] and [30, Theorem 6]. The actor update of each agent can be written as

$$
\theta _ { t + 1 } ^ { i } = \Psi _ { \Theta ^ { i } } \left( \theta _ { t } ^ { i } + \alpha _ { t } ^ { \theta } \cdot \delta _ { t } ^ { i } \cdot \nabla _ { \theta ^ { i } } \log \pi _ { \theta _ { t } ^ { i } } ^ { i } ( a _ { t } ^ { i } \mid s _ { t } ) \right) ,\tag{25}
$$

where

$$
\delta _ { t } ^ { i } = \bar { r } ( s _ { t } , a _ { t } ; \lambda _ { t } ^ { i } ) + \gamma V ( s _ { t + 1 } ; v _ { t } ^ { i } ) - V ( s _ { t } ; v _ { t } ^ { i } ) .\tag{26}
$$

By Theorem 1, the critic and reward parameters $( v _ { t } ^ { i } , \lambda _ { t } ^ { i } )$ almost surely converge to $( v _ { \theta } , \lambda _ { \theta } )$ for every fixed $\theta \in \Theta . \operatorname { A l s o } .$ the map $\theta \mapsto \left( v _ { \theta } , \lambda _ { \theta } \right)$ is continuous on Θ by Assumption 3 and the continuity of $d _ { \theta }$ in θ [3, Appendix B.3]. Since the actor evolves on a slower timescale by Assumption $5 , \theta _ { t }$ can be held constant when analyzing the faster recursions, and the standard two-time-scale argument from [3, Theorem 4.10] and [30, Theorem 6] applies, yielding

$$
\operatorname* { l i m } _ { t \to \infty } \| v _ { t } ^ { i } - v _ { \theta _ { t } } \| = \operatorname* { l i m } _ { t \to \infty } \| \lambda _ { t } ^ { i } - \lambda _ { \theta _ { t } } \| = 0 \quad \mathrm { a . s . , ~ } \forall i \in \mathcal { V } .\tag{27}
$$

Hence the bias $\varepsilon _ { t } ^ { i } : = \delta _ { t } ^ { i } - \delta _ { \theta _ { t } } \left( s _ { t } , a _ { t } , s _ { t + 1 } \right)$ introduced by using $( v _ { t } ^ { i } , \lambda _ { t } ^ { i } )$ in place of $( v _ { \theta _ { t } } , \lambda _ { \theta _ { t } } )$ satisfies $\varepsilon _ { t } ^ { i } \to 0$ almost surely by (27).

Combining this with Assumptions 1 and 4, we know that

$\varepsilon _ { t } ^ { i }$ in the actor TD error $\delta _ { t } ^ { i } = \delta _ { \theta _ { t } } ( s _ { t } , a _ { t } , s _ { t + 1 } ) + \varepsilon _ { t } ^ { i }$ is bounded and $\varepsilon _ { t } ^ { i } \to 0$ almost surely, and the actor update asymptotically becomes

$$
\begin{array} { r } { \theta _ { t + 1 } ^ { i } = \Psi _ { \Theta ^ { i } } \left( \theta _ { t } ^ { i } + \alpha _ { t } ^ { \theta } \cdot \delta _ { \theta _ { t } } ( s _ { t } , a _ { t } , s _ { t + 1 } ) \nabla _ { \theta ^ { i } } \log \pi _ { \theta _ { t } ^ { i } } ^ { i } ( a _ { t } ^ { i } \mid s _ { t } ) \right) . } \end{array}\tag{28}
$$

At this point, the actor recursion has the same limiting stochastic approximation form as those analyzed in [3, Theorem 4.10] and [30, Theorem 6]. It now remains to verify that the assumptions required for their stochastic approximation analysis hold in our setting. More specifically, we have

$\begin{array} { r } { \operatorname* { s u p } _ { t } \mathbb { E } \big ( \lVert \delta _ { t } ^ { i } \nabla _ { \theta ^ { i } } \log \pi _ { \theta _ { \star } ^ { i } } ^ { i } ( a _ { t } ^ { i } | s _ { t } ) \rVert ~ | ~ \theta _ { t ^ { \prime } } , t ^ { \prime } \ \leq \ t \big ) \ < \ \infty } \end{array}$ by Assumptions $1 , 3 , 4$ and 6 with Theorem 1;

• compact and hyper-rectangular projection set $\Theta ^ { i }$ by ${ \mathrm { A s } } -$ sumption $6 ;$

$\begin{array} { r l r l r l r } { \sum _ { t = 0 } ^ { \infty } \alpha _ { t } ^ { \theta } } & { { } = } & { \infty , } & { \sum _ { t = 0 } ^ { \infty } ( \alpha _ { t } ^ { \theta } ) ^ { 2 } } & { { } < } & { \infty . } \end{array}$ , and $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } \alpha _ { t + 1 } ^ { \theta } / \alpha _ { t } ^ { \theta } = 1 } \end{array}$ by Assumption $5 ;$

• bias $\varepsilon _ { t } ^ { i } \to 0$ almost surely (which is established above);

• continuous limiting mean update $\begin{array} { r l r l } { g ( \theta _ { t } ^ { i } ) } & { { } } & { = } & { { } } \end{array}$ $\mathbb { E } _ { d _ { \theta } , \pi _ { \theta } , P } \left[ \delta _ { \theta } ( s _ { t } , a _ { t } , s _ { t + 1 } ) \nabla _ { \theta ^ { i } } \log \pi _ { \theta _ { * } ^ { i } } ^ { i } ( a _ { t } ^ { i } | s _ { t } ) \right]$ by Assumption 3;

• continuity of $\theta \mapsto \left( v _ { \theta } , \lambda _ { \theta } \right)$ (which is established above).

Hence, by [30, Theorem 6], the asymptotic behavior of the actor is governed by

$$
\begin{array} { r } { \dot { \theta } ^ { i } = \hat { \Psi } _ { \boldsymbol { \Theta } ^ { i } } \left[ \mathbb { E } _ { d _ { \theta } , \pi _ { \theta } , P } \left[ \delta _ { \theta } ( s _ { t } , a _ { t } , s _ { t + 1 } ) \nabla _ { \theta ^ { i } } \log \pi _ { \theta ^ { i } } ^ { i } ( a _ { t } ^ { i } | s _ { t } ) \right] \right] , } \end{array}
$$

and $\theta ^ { i }$ converges almost surely to a point in the set of locally asymptotically stable equilibria of (24). □

Our analysis is in the same spirit as those developed in [3], [24], [30]. The convergence guarantees established in Theorem 2 show that every agent converges almost surely to an asymptotically stable equilibrium of (24), which is the standard convergence guarantee for actor-critic algorithms even in the single-agent setting [3], [42]. Our result nevertheless represents a notable improvement over existing Byzantineresilient MARL methods (e.g., [30], [31], [44]), which guarantee convergence only to a neighborhood.

The key distinction is that FRAC-MARL completely removes the Byzantine-induced bias that affects the existing work before it enters the consensus updates. Specifically, Byzantine attacks introduce bias in two ways: (i) corrupted messages that, if accepted, directly perturb the consensus updates, and (ii) asymmetric information flow created by filtering, which itself introduces consensus bias. As established by Lemmas 2-3, our method removes such biases through $( r , r ^ { \prime } )$ redundancy. Consequently, the critic and reward estimates converge to the same values as in the Byzantine-free setting.

We note that such stronger convergence guarantees are obtained under Assumption 2, where all agents are assumed to follow the prescribed protocol, and Byzantine attacks are introduced only through communication corruption. This setting constitutes a weaker attack model than the classical Byzantine-agent model [13], [14], in which Byzantine agents may arbitrarily deviate from the prescribed protocol.

However, this weakened attack model at the same time allows our method to remain completely independent of how the Byzantine attack unfolds, as long as the number of corrupted communications stays bounded. This is also an improvement over some of the existing work on Byzantineresilient AC-MARL [30], [31], as they assume that Byzantine agents’ policies converge to some stationary policy, which ensures stationary MDP from the perspective of non-Byzantine agents. This effectively restricts the learning-dynamics of the Byzantine attackers in the asymptotic sense. FRAC-MARL requires no restriction on the temporal or learning-dynamics behavior of Byzantine attacks. As a result, as long as the attack occurs at the communication level, our guarantee holds regardless of the behavior of the Byzantine attack.

## V. REDUNDANT NETWORK GRAPH

Through Theorems 1-2, we have shown that $( r , r - 2 F - 1 ) .$ redundancy where $r > 2 F$ plays a pivotal role in achieving resilience against the F-total Byzantine edge attacks. In this section, we explore different aspects of the notion of $( r , r ^ { \prime } ) .$ redundancy by providing (i) a systematic construction of an $( r , r ^ { \prime } )$ -redundant graph (Proposition 1) and (ii) its computation time (Proposition 2). As the underlying structural requirements are independent of the learning dynamics, we focus on timeinvariant graphs and drop the argument t on a graph throughout this discussion.

We first present a systematic method to construct $( r , r ^ { \prime } ) .$ redundant graphs for any r and $r ^ { \prime } { : }$

Proposition 1. Let $\mathcal { V } = \{ 1 , \ldots , n \}$ where $n > r ,$ , and $\mathcal { V } _ { c } =$ $\{ 1 , \dots , r \} \subset \mathcal { V }$ . Then, a graph $\mathcal { G } = \left( \mathcal { V } , \mathcal { E } \right) i s \left( \boldsymbol { r } , \boldsymbol { r ^ { \prime } } \right)$ -redundant $f o r \ r > r ^ { \prime } \ge 0 \ i f \left( i \right)$ every node in $\mathcal { V } _ { c }$ is connected to every other node in $\mathcal { V } _ { c }$ and (ii) every node $i \in \mathcal { V } \setminus \mathcal { V } _ { c }$ is connected to all r nodes in $\nu _ { c } .$

Proof. By (i) and (ii), every $i \in \mathcal { V } _ { c }$ is adjacent to every other node of $\nu ,$ so $\mathcal { N } _ { i } = \mathcal { V } \backslash \{ i \}$ and $B _ { i } = \nu ;$ and every $i \in \mathcal { V } \backslash \mathcal { V } _ { c }$ is adjacent to all of $\gamma _ { c } ,$ so $\mathcal { N } _ { i } \supseteq \mathcal { V } _ { c }$ and $B _ { i } \supseteq \mathcal { V } _ { c } \cup \{ i \}$

We show $| B _ { i } \cap \mathcal { N } _ { j } | \geq r$ for every $i \neq j .$ considering four cases. (a) If $i , j \in \mathcal { V } _ { c }$ , then $\ B _ { i } \cap \mathcal { N } _ { j } = \mathcal { V } \setminus \{ j \}$ , so $| B _ { i } \cap \mathcal { N } _ { j } | =$ $n - 1 \geq r$ since $n > r . \ ( \mathsf { b } )$ If $i \in \mathcal { V } _ { c }$ and $j \notin \mathcal { V } _ { c }$ , then $\mathcal { B } _ { i } \cap \mathcal { N } _ { j } = \mathcal { N } _ { j } \ \supseteq \ \mathcal { V } _ { c } , \ \mathrm { s o } \ | \mathcal { B } _ { i } \cap \mathcal { N } _ { j } | \geq r .$ (c) If $i \notin \mathcal { V } _ { c }$ and $j \in \mathcal { V } _ { c } ,$ , then $| B _ { i } \cap \mathcal { N } _ { j } | \geq | ( \mathcal { V } _ { c } \cup \{ i \} ) \setminus \{ j \} |$ . Since $j \in \mathcal { V } _ { c }$ and $i \notin \mathcal { V } _ { c } ,$ , this set has $( r - 1 ) + 1 = r$ elements. (d) If $i , j \notin \mathcal { V } _ { c } ,$ then $\begin{array} { r } { B _ { i } \cap \mathcal { N } _ { j } \supseteq \mathcal { V } _ { c } , } \end{array}$ , so $| B _ { i } \cap \mathcal { N } _ { j } | \geq r$

In every case $| B _ { i } \cap \mathcal { N } _ { j } | \geq r ,$ , so $( i , j ) \in \mathcal { E } ^ { r }$ for all $i \neq j .$ Hence G is $( r , r ^ { \prime } )$ -redundant for every $r ^ { \prime }$ with $r > r ^ { \prime } \geq 0$ □

While Proposition 1 provides a method to construct $( r , r ^ { \prime } ) .$ redundant graphs, the following result establishes that we can verify the redundancy of an arbitrary graph efficiently.

Proposition 2. Given an $r , r ^ { \prime } \in \mathbb { Z } _ { > 0 }$ and a communication graph $\mathcal { G } = ( \nu , \mathcal { E } )$ with $| \nu | = n$ , one can verify whether $\mathcal { G }$ is $( r , r ^ { \prime } )$ -redundant in $O ( n ^ { 3 } )$

Proof. Let A be an adjacency matrix of ${ \mathcal { G } } .$ Then, ${ \bar { A } } : = A ^ { 2 } + A$ will contain elements $\bar { a } _ { i j }$ that counts the number of shared neighbors between nodes i and $j$ (including node i itself) i.e., $| B _ { i } \cap \mathcal { N } _ { j } |$ . Computing $\bar { A }$ using standard matrix multiplication requires $O ( n ^ { 3 } )$ operations [45, Sec. 4], and checking whether $\bar { a } _ { i j } \ge r \ \mathrm { o r } \ \bar { a } _ { i j } \le r ^ { \prime }$ adds at most $O ( n ^ { 2 } )$ operations. Next, to verify that the r-2-hop graph $\mathcal G ^ { r } = ( \nu , \mathcal E ^ { r } )$ of $\mathcal { G }$ is connected, one can perform a Breadth-First Search (BFS), which requires $O ( n + m _ { r } )$ time, where $m _ { r } = \left| \mathcal { E } ^ { r } \right| \left[ 4 5 \right.$ , Sec. 22.2]. Therefore, the total required computation is $O ( n ^ { 3 } + m _ { r } )$ . Since $m _ { r } \ \leq$ $\left( \ l _ { 2 } ^ { n } \right) = O ( n ^ { 2 } { \bar { ) } } , O ( n ^ { 3 } + { \bar { m } } _ { r } ) = O ( n ^ { 3 } )$ □

Proposition 2 establishes that $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ )-redundancy can be verified efficiently. Compare this with r-robustness [14], whose definition is given below:

Definition 6 (r-robustness [14]). A graph $\mathcal { G } = ( \nu , \mathcal { E } )$ is rrobust iffor every pair of nonempty, disjoint subsets $\mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } \subset$ V, at least one of the subsets contains a node with at least r neighbors outside the subset. That is, there exists a node $i \in \mathcal { P } _ { k }$ such that $| \mathcal { N } _ { i } \ \backslash \ \mathcal { P } _ { k } | \ge r \ f o r$ some $k \in \{ 1 , 2 \}$

While $( 2 F + 1 )$ -robustness provides a sufficient condition for many Byzantine-resilient AC-MARL frameworks [27], [29]–[31], determining whether a graph satisfies this property is coNP-complete [33]. Consequently, there is no known efficient algorithm for verifying r-robustness in general, making robustness-based design impractical for large-scale and dynamic networks. In contrast, $( r , r ^ { \prime } )$ -redundancy offers a tractable alternative, making our method more suitable.

Lemma 4. Let G be an $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundant graph constructed according to Proposition 1 with $n \geq 2 r - 1$ . Then, G is rrobust.

Proof. By Proposition 1, each $i \in \mathcal { V } _ { c }$ satisfies $\mathcal { N } _ { i } = \mathcal { V } \backslash \{ i \}$ and each $i \in \mathcal { V } \setminus \mathcal { V } _ { c }$ satisfies $\mathcal { N } _ { i } \supseteq \mathcal { V } _ { c }$ . Let $\mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } \subset \mathcal { V }$ be nonempty and disjoint; since $| \mathcal { P } _ { 1 } | + | \mathcal { P } _ { 2 } | \leq n ,$ , assume without loss of generality $| \mathcal { P } _ { 1 } | \le \lfloor n / 2 \rfloor$ . If $\mathcal { P } _ { 1 } \cap \mathcal { V } _ { c } \neq \emptyset$ any $i \in \mathcal { P } _ { 1 } \cap \mathcal { V } _ { c }$ gives $| \mathcal { N } _ { i } \setminus \mathcal { P } _ { 1 } | = \overline { { n - | \mathcal { P } _ { 1 } | } } \ge \lceil n / 2 \rceil \ge r .$ Otherwise $\begin{array} { r } { \mathcal { V } _ { c } \cap \mathcal { P } _ { 1 } = \emptyset . } \end{array}$ , so any $i \in \mathcal { P } _ { 1 }$ gives $\mathcal { N } _ { i } \setminus \mathcal { P } _ { 1 } \supseteq \mathcal { V } _ { c }$ and thus $\left| { \mathcal { N } } _ { i } \right. \left. \mathcal { P } _ { 1 } \right| \geq r .$ . In either case $\mathcal { P } _ { 1 }$ contains a node i such that $\begin{array} { r } { | \mathcal { N } _ { i }  \mathcal { P } _ { 1 } | \geq r , } \end{array}$ completing the proof. □

Lemma 4 connects $( r , r ^ { \prime } )$ -redundancy of graphs constructed via Proposition 1 to the notion of r-robustness. Note that this result does not establish a general characterization between the two properties; it applies only to the specific class of graphs. Establishing a full characterization of the relationship between these two topological conditions remains future work.

## VI. SIMULATION RESULTS

We evaluate the proposed algorithm on a cooperative formation task built on the Multi-Particle Environments 2 (MPE2) [10]. We consider a team of $n ~ = ~ 1 0$ agents that must arrange themselves into a circular formation of radius $R _ { \mathrm { c i r c l e } } = 0 . 5$ around a stationary landmark located at $p ^ { \mathrm { l m } } \in$ $\mathbb { R } ^ { 2 }$ . Each agent estimates its actor, critic, and team-average reward functions using neural networks with a single hidden layer of 30 units and Leaky ReLU activation functions with negative slope 0.1.

State Space: At each time $t ,$ agent $i \in \mathcal { V } = \{ 1 , \dotsc , 1 0 \}$ observes the state

$$
s _ { t } = \left[ ( o _ { t } ^ { 1 } ) ^ { \top } , ( o _ { t } ^ { 2 } ) ^ { \top } , \ldots , ( o _ { t } ^ { n } ) ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { 6 n } ,\tag{29}
$$

where $o _ { t } ^ { i } = \left\lceil ( \dot { p } _ { t } ^ { i } ) ^ { \top } , \ : ( p _ { t } ^ { i } ) ^ { \top } , ( p _ { t } ^ { i , \mathrm { r e l } } ) ^ { \top } \right\rceil ^ { \top }$ contains the agent i’s velocity p˙<sup>i</sup>, position $p _ { t } ^ { i } \in \mathbb { R } ^ { 2 }$ , and its relative position to the landmark, $\bar { p _ { t } ^ { i , \mathrm { r e l } } } = p _ { t } ^ { i } - \bar { p } ^ { \mathrm { l m } }$

Action Space: Each agent selects a discrete action $a _ { t } ^ { i } \in \mathcal { A } ^ { i } =$ {stay, left, right, down, up}, corresponding to stay still or a unit force applied along one of the four cardinal directions.

Reward Space: Each agent i receives a reward $r _ { t + 1 } ^ { i } ~ =$ $- \mathrm { c l i p } ( \| p _ { t } ^ { i } - g _ { t } ^ { i } \| _ { 2 } , 0 , 2 )$ based on its distance to an assigned formation goal $g _ { t } ^ { i } .$ . At each time step, n goal positions $\{ g _ { t } ^ { i } \} _ { i \in \mathcal { V } }$ are evenly spaced on a circle of radius $R _ { \mathrm { c i r c l e } }$ around the landmark, with the orientation determined by the current agents’ configurations, and are assigned to the agents by minimizing the total assignment distance using the Hungarian algorithm.

Agent Dynamics: We use the default dynamics defined in MPE2, where each agent i is modeled as a point mass with damped double-integrator dynamics:

$$
p _ { t + 1 } ^ { i } = p _ { t } ^ { i } + { \dot { p } } _ { t } ^ { i } \Delta t ,\tag{30}
$$

$$
\dot { p } _ { t + 1 } ^ { i } = \left( 1 - \beta \right) \dot { p } _ { t } ^ { i } + \frac { \Delta t } { m } u _ { t } ^ { i } ,\tag{31}
$$

where $u _ { t } ^ { i } ~ \in ~ \{ ( 0 , 0 ) , ( \pm 1 , 0 ) , ( 0 , \pm 1 ) \}$ is the control input selected by $a _ { t } ^ { i } .$ with sampling time $\Delta t \ = \ 0 . 1$ , damping coefficient $\beta = 0 . 2 5$ , and mass $m = 1$

Training: We compare our method against four baselines:

• Normal: the vanilla decentralized AC-MARL from [3, $\mathrm { A l g } . \ 2 ]$ without attacks;

• Naive: the vanilla decentralized AC-MARL from [3, Alg. 2] under F Byzantine edge attacks;

• Projection: the resilient AC-MARL method from [30, Alg. 2] that uses a projection-based defense mechanism. Following the authors’ implementation, trimmed-mean aggregation is applied to the hidden-layer parameters before the projection-based updates; and

• Trimmed-Mean: the resilient AC-MARL from [44, Alg. 1] that performs an element-wise trimmed-mean operation.

We simulate Byzantine edge attacks by randomly selecting F edges incident to agent 1, replacing the transmitted messages with parameter tuples obtained by adding a positive offset to each parameter tensor. The offset scaled according to the mean absolute magnitude of the corresponding tensor. Then, it is upper bounded by 1 to ensure numerical stability.

We train all methods for 10000 episodes using five different random seeds, with each episode consisting of 35 steps, using 10 critic and reward-function updates per actor update. The learning rates are $\alpha _ { t } ^ { v } ~ = ~ \alpha _ { t } ^ { \lambda } ~ = ~ 0 . 0 1$ and $\alpha _ { t } ^ { \theta } ~ = ~ 0 . 0 0 1$ Updates are performed in batches every 20 episodes. We set the discount factor to $\gamma = 0 . 9$ and each agent selects a random action with probability $\mu = 0 . 1$

We consider $F = 1$ and $F = 2$ with $r = 2 F + 1$ . Every 20 episodes we generate a $( 2 F + 1 , 0 )$ -redundant network using the construction mechanism in Proposition 1, while randomly permuting the agent indices to emulate a time-varying communication topology. By construction, the resulting networks satisfy the topological conditions required by Theorems 1-2. By Lemma 4, the network is also $( 2 F + 1 )$ -robust for all time, satisfying the sufficient topological condition required for the convergence of the projection-based method in [30].

![](images/f8bcc3ba6bb926dd6e50e88ba3f194cbf4de051c384c68f08e068c725b7249ae.jpg)  
Fig. 3. Reward curves under F-total Byzantine edge attack, with (a) ${ \pmb F } = { \pmb 1 }$ and (b) ${ \mathbf { F } } = { \mathbf { 2 } } .$ . Our method attains the same reward level as the attack-free Normal baseline, whereas the other methods converge to suboptimal policies with visibly lower rewards.

Figure 3 reports the reward curves under $F ~ = ~ 1$ and $F = 2$ Byzantine edge attacks. Our method matches the attackfree Normal baseline in both settings. This is consistent with Theorems 1-2, which guarantee convergence to equilibria of the limiting ODEs rather than a neighborhood of them. In contrast, the other methods converge to lower reward levels, reflecting the residual errors introduced by their consensus steps under F-total Byzantine edge attacks.

## VII. CONCLUSION

We study resilient actor-critic multi-agent reinforcement learning under Byzantine edge attacks. Our method exploits the redundancy of two-hop communication to decide which messages to trust and filter. We introduce a novel topological condition, $( \boldsymbol { r } , \boldsymbol { r } ^ { \prime } )$ -redundancy, to provide the conditions under which the policy parameters converge almost surely to a locally asymptotically stable equilibrium of the attack-free limiting ODE. We validate our method on a multi-agent formation control task.

## REFERENCES

[1] R. S. Sutton and A. Barto, Reinforcement learning: An introduction. MIT press Cambridge, 1998, vol. 1, no. 1.

[2] K. Zhang, Z. Yang, and T. Bas¸ar, “Multi-agent reinforcement learning: A selective overview of theories and algorithms,” Handbook of reinforcement learning and control, pp. 321–384, 2021.

[3] K. Zhang, Z. Yang, H. Liu, T. Zhang, and T. Basar, “Fully decentralized multi-agent reinforcement learning with networked agents,” in Proceedings of the 35th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 80. PMLR, 10–15 Jul 2018, pp. 5872–5881.

[4] Y. Lin, G. Qu, L. Huang, and A. Wierman, “Multi-agent reinforcement learning in stochastic networked systems,” in Advances in Neural Information Processing Systems, vol. 34. Curran Associates, Inc., 2021, pp. 7825–7837.

[5] G. Qu, A. Wierman, and N. Li, “Scalable reinforcement learning of localized policies for multi-agent networked systems,” in Proceedings of the 2nd Conference on Learning for Dynamics and Control, ser. Proceedings of Machine Learning Research, vol. 120. PMLR, 10–11 Jun 2020, pp. 256–266.

[6] S. Zeng, T. Chen, A. Garcia, and M. Hong, “Learning to coordinate in multi-agent systems: A coordinated actor-critic algorithm and finitetime guarantees,” in Proceedings of The 4th Annual Learning for Dynamics and Control Conference, ser. Proceedings of Machine Learning Research, vol. 168. PMLR, 23–24 Jun 2022, pp. 278–290.

[7] P. Dai, W. Yu, H. Wang, and S. Baldi, “Distributed actor–critic algorithms for multiagent reinforcement learning over directed graphs,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 10, pp. 7210–7221, 2023.

[8] P. Dai, Y. Mo, W. Yu, and W. Ren, “Distributed neural policy gradient algorithm for global convergence of networked multiagent reinforcement learning,” IEEE Transactions on Automatic Control, vol. 70, no. 11, pp. 7109–7124, 2025.

[9] S. Kar, J. M. F. Moura, and H. V. Poor, “QD-learning: A collaborative distributed strategy for multi-agent reinforcement learning through consensus + innovations,” IEEE Transactions on Signal Processing, vol. 61, no. 7, pp. 1848–1862, 2013.

[10] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, “Multiagent actor-critic for mixed cooperative-competitive environments,” Neural Information Processing Systems (NIPS), 2017.

[11] J. Foerster, G. Farquhar, T. Afouras, N. Nardelli, and S. Whiteson, “Counterfactual multi-agent policy gradients,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[12] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of ppo in cooperative multi-agent games,” Advances in neural information processing systems, vol. 35, pp. 24 611– 24 624, 2022.

[13] L. Su and N. H. Vaidya, “Byzantine-resilient multiagent optimization,” IEEE Transactions on Automatic Control, vol. 66, no. 5, pp. 2227–2233, 2021.

[14] H. J. LeBlanc, H. Zhang, X. Koutsoukos, and S. Sundaram, “Resilient asymptotic consensus in robust networks,” IEEE Journal on Selected Areas in Communications, vol. 31, no. 4, pp. 766–781, 2013.

[15] L. Yuan and H. Ishii, “Resilient average consensus with adversaries via distributed detection and recovery,” IEEE Transactions on Automatic Control, vol. 70, no. 1, pp. 415–430, 2025.

[16] H. Lee and D. Panagou, “Distributed resilience-aware control in multirobot networks,” in 2025 IEEE 64th Conference on Decision and Control (CDC), 2025, pp. 3868–3875.

[17] S. Sundaram and B. Gharesifard, “Distributed optimization under adversarial nodes,” IEEE Transactions on Automatic Control, vol. 64, no. 3, pp. 1063–1076, 2019.

[18] M. Yemini, A. Nedic, A. J. Goldsmith, and S. Gil, “Resilient distributed´ optimization for multiagent cyberphysical systems,” IEEE Transactions on Automatic Control, vol. 70, no. 6, pp. 3952–3967, 2025.

[19] C. Fang, Z. Yang, and W. U. Bajwa, “Bridge: Byzantine-resilient decentralized gradient descent,” IEEE Transactions on Signal and Information Processing over Networks, vol. 8, pp. 610–626, 2022.

[20] Y. Chen, L. Su, and J. Xu, “Distributed statistical machine learning in adversarial settings: Byzantine gradient descent,” vol. 1, no. 2, Dec. 2017.

[21] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,” in Advances in Neural Information Processing Systems, vol. 30. Curran Associates, Inc., 2017.

[22] H. Lee, V.-D. Yun, H. Oh, D. Panagou, and S. P. Karimireddy, “Robust multi-agent llms under byzantine faults,” arXiv preprint arXiv:2605.09076, 2026.

[23] H. Luo, G. Sun, Y. Liu, D. Zhao, D. Niyato, H. Yu, and S. Dustdar, “A weighted byzantine fault tolerance consensus driven trusted multiple large language models network,” IEEE Transactions on Cognitive Communications and Networking, 2025.

[24] M. Figura, K. C. Kosaraju, and V. Gupta, “Adversarial attacks in consensus-based multi-agent reinforcement learning,” in 2021 American Control Conference (ACC), 2021, pp. 3050–3055.

[25] Y. Xie, S. Mou, and S. Sundaram, “Towards resilience for multi-agent qd-learning,” in 2021 60th IEEE Conference on Decision and Control (CDC), 2021, pp. 1250–1255.

[26] Hairi, M. Fang, Z. Zhang, A. Velasquez, and J. Liu, “On the hardness of decentralized multi-agent policy evaluation under byzantine attacks,” in 2024 International Symposium on Modeling and Optimization in Mobile, Ad Hoc, and Wireless Networks (WiOpt), 2024, pp. 257–264.

[27] Y. Xie, S. Mou, and S. Sundaram, “Communication-efficient and resilient distributed q-learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 3, pp. 3351–3364, 2023.

[28] Z. Wu, H. Shen, T. Chen, and Q. Ling, “Byzantine-resilient decentralized policy evaluation with linear function approximation,” IEEE Transactions on Signal Processing, vol. 69, pp. 3839–3853, 2021.

[29] J. Yao and X. Gong, “Communication-efficient and resilient distributed deep reinforcement learning for multi-agent systems,” in 2024 IEEE International Conference on Unmanned Systems (ICUS), 2024, pp. 1521–1526.

[30] L. Ye, M. Figura, Y. Lin, M. Pal, P. Das, J. Liu, and V. Gupta, “Resilient multiagent reinforcement learning with function approximation,” IEEE Transactions on Automatic Control, vol. 69, no. 12, pp. 8497–8512, 2024.

[31] X. Gong, Y. Lu, J. Gui, and T. Yu, “Resilient fully-distributed reinforcement learning for uav swarms against general byzantine attacks,” Journal of the Franklin Institute, vol. 363, no. 12, p. 108732, 2026.

[32] J. K. Medhi, R. Liu, Q. Wang, and X. Chen, “Robust multiagent reinforcement learning for uav systems: Countering byzantine attacks,” Information, vol. 14, no. 11, 2023.

[33] H. Zhang, E. Fata, and S. Sundaram, “A notion of robustness in complex networks,” IEEE Transactions on Control of Network Systems, vol. 2, no. 3, pp. 310–320, 2015.

[34] Y. Lin, S. Gade, R. Sandhu, and J. Liu, “Toward resilient multi-agent actor-critic algorithms for distributed reinforcement learning,” in 2020 American Control Conference (ACC), 2020, pp. 3953–3958.

[35] Q. Lin and Q. Ling, “Robust reward-free actor–critic for cooperative multiagent reinforcement learning,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 12, pp. 17 318–17 329, 2024.

[36] M. Fang, X. Wang, and N. Z. Gong, “Provably robust federated reinforcement learning,” in Proceedings of the ACM on Web Conference 2025, ser. WWW ’25. New York, NY, USA: Association for Computing Machinery, 2025, p. 896–909.

[37] S. Guo, T. Zhang, H. Yu, X. Xie, L. Ma, T. Xiang, and Y. Liu, “Byzantine-resilient decentralized stochastic gradient descent,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 32, no. 6, pp. 4096–4106, 2022.

[38] Z. Wu, T. Chen, and Q. Ling, “Byzantine-resilient decentralized stochastic optimization with robust aggregation rules,” IEEE Transactions on Signal Processing, vol. 71, pp. 3179–3195, 2023.

[39] E. M. El-Mhamdi, S. Farhadkhani, R. Guerraoui, A. Guirguis, L.-N. Hoang, and S. Rouault, “Collaborative learning in the jungle (decentralized, byzantine, heterogeneous, asynchronous and nonconvex learning),” Advances in neural information processing systems, vol. 34, pp. 25 044– 25 057, 2021.

[40] L. He, S. P. Karimireddy, and M. Jaggi, “Byzantine-robust decentralized learning via clippedgossip,” arXiv preprint arXiv:2202.01545, 2022.

[41] H. Lee and D. Panagou, “Fully byzantine-resilient distributed multiagent q-learning,” arXiv preprint arXiv:2604.02791, 2026.

[42] S. Bhatnagar, R. S. Sutton, M. Ghavamzadeh, and M. Lee, “Natural actor–critic algorithms,” Automatica, vol. 45, no. 11, pp. 2471–2482, 2009.

[43] X. Lei, G. Wen, and M. M. Polycarpou, “Distributed secure consensus tracking for multi-agent systems: From asymptotic to finite-/fixed-time convergence,” IEEE Transactions on Automatic Control, pp. 1–8, 2026.

[44] Z. Wu, H. Shen, T. Chen, and Q. Ling, “Byzantine-resilient decentralized td learning with linear function approximation,” in ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 5040–5044.

[45] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, Introduction to Algorithms, 3rd Edition. MIT Press, 2009.

![](images/093dc819b524b1ee1fde54ebda4224caf645a1d71750da11159eef691b50ab50.jpg)

Haejoon Lee (Student Member, IEEE) received the B.S. degree in applied math and statistics from Stony Brook University, Stony Brook, NY, USA, in 2023. He earned the M.S. degree in robotics in 2025 from the University of Michigan, Ann Arbor, MI, USA, where he is currently working toward the Ph.D. degree in robotics, advised by Prof. Dimitra Panagou.

His research interests include safety, resilience, and security of autonomous systems, with particular emphasis on distributed consensus, optimization, and learning for multi-agent systems in adversarial and uncertain environments.

![](images/56a096cefd94ac79651138670e0d53487eb47ca67bc7ac3c963ec6ddee1c78b4.jpg)

Dimitra Panagou (Diploma (2006) and PhD (2012) in Mechanical Engineering from the National Technical University of Athens, Greece) is an Associate Professor with the Department of Robotics, with a courtesy appointment with the Department of Aerospace Engineering, University of Michigan. Her research program spans the areas of nonlinear systems and control; multi-agent systems; autonomy; and aerospace robotics. She is particularly interested in the development of provably-correct methods for the safe and secure (resilient) operation of autonomous systems with applications in robot/sensor networks and multi-vehicle systems under uncertainty. She is a recipient of the NASA Early Career Faculty Award, the AFOSR Young Investigator Award, the NSF CAREER Award, the George J. Huebner, Jr. Research Excellence Award, and a Senior Member of the IEEE and the AIAA.